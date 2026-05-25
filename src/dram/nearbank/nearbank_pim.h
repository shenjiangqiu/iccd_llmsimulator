#pragma once

#include <memory>
#include <string>

#include "common/type.h"
#include "dram/nearbank/nearbank_config.h"

namespace llm_system {

// =============================================================================
// NearbankGEMVResult - Result of a GEMV latency calculation
// =============================================================================
struct NearbankGEMVResult {
    time_ns latency_ns = 0.0;          // Total latency for one bank's work
    time_ns rowbuffer_time_ns = 0.0;   // Time spent filling row buffers (total)
    time_ns pe_compute_time_ns = 0.0;  // Time spent in PE computation (total)
    time_ns base_gemv_time_ns = 0.0;   // Latency of main GEMV only (w/o reduction)
    time_ns reduction_time_ns = 0.0;   // Extra PE time for reduction sums
    hw_metric total_bytes = 0.0;       // Total bytes processed by all banks
    hw_metric bytes_per_bank = 0.0;    // Bytes processed per bank
    hw_metric reduction_ops = 0.0;     // Reduction ops: (M+N)*K element-sums
    int num_rowbuffer_fills = 0;       // Number of row buffer fills per bank
    std::string description;           // Human-readable summary

    NearbankGEMVResult& operator+=(const NearbankGEMVResult& rhs) {
        latency_ns += rhs.latency_ns;
        rowbuffer_time_ns += rhs.rowbuffer_time_ns;
        pe_compute_time_ns += rhs.pe_compute_time_ns;
        base_gemv_time_ns += rhs.base_gemv_time_ns;
        reduction_time_ns += rhs.reduction_time_ns;
        total_bytes += rhs.total_bytes;
        reduction_ops += rhs.reduction_ops;
        return *this;
    }
};

// =============================================================================
// NearbankPIMUnit - Simulates Nearbank PIM processing for GEMV operations
//
// This unit models the data movement and computation overhead of performing
// GEMV (General Matrix-Vector multiply) operations inside near-bank PIM units.
//
// Processing Model:
//   1. Data is loaded from DRAM into row buffers (rowbuffer fill)
//   2. PE computes on data in the row buffer (pe_width_bytes per cycle)
//   3. While PE computes on row buffer i, row buffer i+1 is being filled
//   4. This pipeline continues until all bank data is processed
//
// The total latency is computed for ONE bank (all banks work in parallel):
//   latency = first_rb_fill_time + max(remaining_fill_time, total_pe_time)
//
// Key Assumptions:
//   - Work is evenly distributed across all PIM banks
//   - Virtual addresses are constructed from operand sizes (no real data layout)
//   - All banks work in parallel, so total latency = single bank latency
//   - Only GEMV operations are modeled (not softmax, not reductions)
// =============================================================================
class NearbankPIMUnit {
 public:
    using Ptr = std::shared_ptr<NearbankPIMUnit>;

    [[nodiscard]] static Ptr Create(const NearbankPIMConfig& config) {
        return Ptr(new NearbankPIMUnit(config));
    }

    // -------------------------------------------------------------------------
    // computeGEMVLatency - Compute latency for a single GEMV operation
    //
    // Models the operation: matrix_A[M×K] @ matrix_B[K×N] → result[M×N]
    // In attention context:
    //   Scoring:  Q[1×head_dim] @ K^T[head_dim×seq_len] → scores[1×seq_len]
    //   Context:  scores[1×seq_len] @ V[seq_len×head_dim] → output[1×head_dim]
    //
    // Parameters:
    //   M: rows of first matrix (batch dimension)
    //   K: inner dimension (shared between matrices)
    //   N: columns of second matrix
    //   data_element_size_bytes: size of the stored (K/V) matrix elements
    //     (2 for FP16, 1 for 2-bit packed -> 0.25B real storage)
    //     Determines rowbuffer data volume.
    //   compute_element_size_bytes: size for PE computation precision
    //     (2 for FP16, 1 for 2-bit)
    //     Determines PE element throughput. Different from data size when
    //     dequantization is needed (e.g., Score@V: V stored 2-bit, computed FP16).
    //
    // Returns: NearbankGEMVResult with latency breakdown
    // -------------------------------------------------------------------------
    NearbankGEMVResult computeGEMVLatency(int M, int K, int N,
                                          int data_element_size_bytes,
                                          int compute_element_size_bytes) const;

    // Backward-compatible overload: data and compute use same precision
    NearbankGEMVResult computeGEMVLatency(int M, int K, int N,
                                          int element_size_bytes) const {
        return computeGEMVLatency(M, K, N, element_size_bytes, element_size_bytes);
    }

    // Access config
    const NearbankPIMConfig& getConfig() const { return config_; }

 private:
    explicit NearbankPIMUnit(const NearbankPIMConfig& config)
        : config_(config) {}

    NearbankPIMConfig config_;
};

}  // namespace llm_system
