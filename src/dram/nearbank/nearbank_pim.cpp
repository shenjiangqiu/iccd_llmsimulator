#include "dram/nearbank/nearbank_pim.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace llm_system {

// =============================================================================
// computeGEMVLatency
// =============================================================================
//
// Latency Model (per bank, pipelined):
//
//   Given a GEMV operation M×K @ K×N → M×N:
//
//   1. Total data movement per GEMV:
//      - Read matrix A: M*K elements
//      - Read matrix B: K*N elements
//      - Write result:   M*N elements
//      - Total elements = M*K + K*N + M*N
//      - Total bytes = total_elements * element_size_bytes
//
//   2. Work distribution across banks:
//      - bytes_per_bank = total_bytes / num_pim_banks
//
//   3. Row buffer processing (per bank):
//      - Each row buffer fill reads rowbuffer_size_bytes from DRAM
//      - num_rb_fills = ceil(bytes_per_bank / rowbuffer_size_bytes)
//      - Time per fill: T_fill = rowbuffer_size_bytes / dram_to_rb_bw_per_bank
//      - Total fill time: T_fill_total = num_rb_fills * T_fill
//
//   4. PE computation (per bank):
//      - PE processes pe_width_bytes per cycle
//      - PE cycle time = pe_cycle_time_ns
//      - PE throughput = pe_width_bytes / pe_cycle_time_ns (bytes/ns)
//      - Total PE time: T_pe = bytes_per_bank / pe_throughput
//        = bytes_per_bank * pe_cycle_time_ns / pe_width_bytes
//
//   5. Pipelined latency:
//      - Fill row buffer 0 (T_fill), then PE starts computing on it
//      - While PE computes on rb_i, rb_{i+1} is being filled
//      - latency = T_fill + max((num_rb_fills - 1) * T_fill,
//                                num_rb_fills * T_pe_per_rb)
//      where T_pe_per_rb = rowbuffer_size_bytes * pe_cycle_time_ns
//                           / pe_width_bytes
//
//   6. Special case: if bytes_per_bank <= rowbuffer_size_bytes:
//      - Only one row buffer needed
//      - latency = max(bytes_per_bank / dram_to_rb_bw,
//                       bytes_per_bank * pe_cycle_time_ns / pe_width_bytes)
//      (fill and compute can't fully pipeline with one buffer, but the
//       overlap still exists as data streams through)
//
// For simplicity and consistency, we use the general formula even for the
// single-buffer case, as it provides a reasonable approximation.
// =============================================================================
NearbankGEMVResult NearbankPIMUnit::computeGEMVLatency(
    int M, int K, int N,
    int data_element_size_bytes,
    int compute_element_size_bytes) const {

    NearbankGEMVResult result;

    // Data bytes: based on stored KV cache format (determines rowbuffer volume)
    //  data_element_size_bytes >= 2: FP16, 2B/element
    //  data_element_size_bytes <= 1: quantized, bit-width from kv_cache_bits
    double bytes_per_element = data_element_size_bytes;
    double pe_elements_per_cycle = 1.0;
    if (data_element_size_bytes <= 1) {
        int bits = config_.kv_cache_bits;  // 2 or 4
        bytes_per_element = bits / 8.0;     // 0.25 (2-bit) or 0.5 (4-bit)
        // PE element throughput: pe_width_bytes * (8 / bits) elem/cycle
        //   e.g., 16B PE: 2-bit→64, 4-bit→32.  32B PE: 2-bit→128, 4-bit→64.
        double pe_width = config_.pe_width_bytes;
        if (compute_element_size_bytes <= 1) {
            pe_width = config_.pe_lowbit_width_bytes;
        }
        pe_elements_per_cycle = pe_width * (8.0 / bits);
    } else {
        pe_elements_per_cycle = config_.pe_width_bytes / compute_element_size_bytes;
    }
    double pe_cycle_for_compute = (compute_element_size_bytes <= 1)
        ? config_.pe_lowbit_cycle_time_ns : config_.pe_cycle_time_ns;

    // Flag: data is low-bit but we compute in FP16 → need dequantization
    bool implicit_dequant = (data_element_size_bytes <= 1 && compute_element_size_bytes >= 2);

    // --- Step 1: Calculate total data movement ---
    // Elements: read A (M*K) + read B (K*N) + write result (M*N)
    double total_elements = static_cast<double>(M) * K +
                            static_cast<double>(K) * N +
                            static_cast<double>(M) * N;
    result.total_bytes = total_elements * bytes_per_element;

    // --- Step 2: Distribute across banks ---
    int num_banks = config_.getTotalBanks();
    result.bytes_per_bank = result.total_bytes / num_banks;
    double elements_per_bank = total_elements / num_banks;

    // --- Step 3: Row buffer processing ---
    double rb_size = config_.rowbuffer_size_bytes;
    double dram_bw = config_.dram_to_rb_bw_per_bank;

    // Number of row buffer fills needed (ceiling)
    result.num_rowbuffer_fills = static_cast<int>(
        std::ceil(result.bytes_per_bank / rb_size));
    if (result.num_rowbuffer_fills < 1) {
        result.num_rowbuffer_fills = 1;
    }

    // Time to fill one row buffer (full size)
    double time_per_rb_fill = rb_size / dram_bw * 1e9;  // Convert to ns

    // Total row buffer fill time (actual data volume)
    result.rowbuffer_time_ns = result.bytes_per_bank / dram_bw * 1e9;

    // --- Step 4: PE computation (in elements) ---
    // Total PE time (without pipelining): elements / elements_per_cycle * cycle_time
    result.pe_compute_time_ns = elements_per_bank / pe_elements_per_cycle * pe_cycle_for_compute;

    // --- Step 5: Pipelined latency (base GEMV only) ---
    if (result.bytes_per_bank <= rb_size) {
        double fill_time = result.bytes_per_bank / dram_bw * 1e9;
        double compute_time = result.pe_compute_time_ns;
        result.base_gemv_time_ns = std::max(fill_time, compute_time);
    } else {
        double first_fill = time_per_rb_fill;
        double remaining_fill =
            (result.num_rowbuffer_fills - 1) * time_per_rb_fill;
        double total_pe = result.pe_compute_time_ns;
        result.base_gemv_time_ns =
            first_fill + std::max(remaining_fill, total_pe);
    }

    // --- Step 6: Asymmetric quantization reduction overhead ---
    // Q·K ≈ s_Q·s_K · [ Σq̂k̂ - z_K·Σq̂ - z_Q·Σk̂ + d·z_Q·z_K ]
    //
    // Extra work beyond main GEMV (Σq̂k̂):
    //   z_K · Σq̂:  M*K element-sums across the K dimension of Q
    //   z_Q · Σk̂:  N*K element-sums across the K dimension of K
    //   d·z_Q·z_K: constant, negligible
    //
    // These sums are computed in PE alongside the main GEMV on the same
    // data — the rowbuffer already contains Q and K rows.  So only PE
    // time increases, not DRAM read time.
    if (config_.enable_asymmetric_quant) {
        double reduction_elements = static_cast<double>(M) * K   // Σq̂
                                 + static_cast<double>(N) * K;  // Σk̂
        result.reduction_ops = reduction_elements;

        // Reduction distributed across banks like main GEMV
        double red_per_bank_elems = reduction_elements / num_banks;
        result.reduction_time_ns = red_per_bank_elems / pe_elements_per_cycle * pe_cycle_for_compute;

        // PE time with reduction (no extra DRAM reads)
        double total_pe_all = result.pe_compute_time_ns + result.reduction_time_ns;

        // Recompute pipelined latency with combined PE time
        if (result.bytes_per_bank <= rb_size) {
            double fill_time = result.bytes_per_bank / dram_bw * 1e9;
            result.latency_ns = std::max(fill_time, total_pe_all);
        } else {
            double first_fill = time_per_rb_fill;
            double remaining_fill =
                (result.num_rowbuffer_fills - 1) * time_per_rb_fill;
            result.latency_ns =
                first_fill + std::max(remaining_fill, total_pe_all);
        }
    } else {
        result.latency_ns = result.base_gemv_time_ns;
    }

    // --- Step 6b: PIM dequantization overhead ---
    // Triggers when:
    //   a) enable_pim_dequant flag AND compute is FP16 (avoid double-counting
    //      with asymmetric quant which operates in low-bit space)
    //   b) implicit_dequant: data is low-bit but compute is FP16
    // Per element: 1 MUL (scale×int) + 1 ADD (offset).
    // Dequant ops use low-bit PE rate; subsequent GEMV uses FP16 PE rate.
    bool need_dequant = implicit_dequant ||
        (config_.enable_pim_dequant && compute_element_size_bytes >= 2);

    if (need_dequant) {
        double dequant_elements = static_cast<double>(K) * N;  // K or V matrix
        double dequant_ops = 2.0 * dequant_elements;  // 1 MUL + 1 ADD per element
        double dequant_per_bank = dequant_ops / num_banks;
        // Dequant uses low-bit PE (quantized data processing)
        double lowbit_pe_rate = config_.pe_lowbit_width_bytes * (8.0 / config_.kv_cache_bits);
        double dequant_pe_time = dequant_per_bank / lowbit_pe_rate * config_.pe_lowbit_cycle_time_ns;

        double total_pe_all = result.pe_compute_time_ns
                            + result.reduction_time_ns + dequant_pe_time;

        if (result.bytes_per_bank <= rb_size) {
            double fill_time = result.bytes_per_bank / dram_bw * 1e9;
            result.latency_ns = std::max(fill_time, total_pe_all);
        } else {
            double first_fill = time_per_rb_fill;
            double remaining_fill =
                (result.num_rowbuffer_fills - 1) * time_per_rb_fill;
            result.latency_ns =
                first_fill + std::max(remaining_fill, total_pe_all);
        }
        result.pe_compute_time_ns += dequant_pe_time;
    }

    // --- Step 7: Build description ---
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "GEMV(" << M << "x" << K << " @ " << K << "x" << N
        << ", data=" << data_element_size_bytes << "B"
        << ", comp=" << compute_element_size_bytes << "B)"
        << " total=" << result.total_bytes << "B"
        << " per_bank=" << result.bytes_per_bank << "B"
        << " rb_fills=" << result.num_rowbuffer_fills
        << " latency=" << result.latency_ns << "ns"
        << " (rb=" << result.rowbuffer_time_ns
        << "ns, pe=" << result.pe_compute_time_ns << "ns";
    if (config_.enable_asymmetric_quant) {
        oss << ", red=" << result.reduction_time_ns << "ns"
            << ", red_ops=" << result.reduction_ops;
    }
    if (implicit_dequant) {
        oss << ", impl_deq";
    }
    if (config_.enable_pim_dequant) {
        oss << ", expl_deq";
    }
    oss << ")";
    result.description = oss.str();

    return result;
}

}  // namespace llm_system
