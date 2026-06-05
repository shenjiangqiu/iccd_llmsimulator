#pragma once

#include "common/type.h"

namespace llm_system {

// =============================================================================
// NearbankPIMConfig - Configuration for Nearbank PIM (Processing-in-Memory) Unit
//
// This models a near-bank PIM architecture where each DRAM bank has a simple
// processing element (PE) that can perform GEMV operations on data in the
// row buffer, avoiding the need to move data to the GPU.
//
// Key concept: Data is loaded from DRAM arrays into row buffers, then the PE
// computes on that data. Row buffer fills and PE computation are pipelined.
// =============================================================================
struct NearbankPIMConfig {
    // --- Bank Configuration ---
    // Total number of PIM-enabled banks across all cubes
    int num_pim_banks = 2048;

    // Number of PIM cubes (stacked DRAM dies with PIM capability)
    int num_pim_cubes = 8;

    // Channels per cube
    int channels_per_cube = 32;

    // --- Row Buffer Configuration ---
    // Size of one row buffer in bytes (e.g., 2KB = 2048)
    // This is the granularity at which data is read from DRAM arrays
    hw_metric rowbuffer_size_bytes = 2048.0;

    // --- PE (Processing Element) Configuration ---
    // FP16 PE: width and cycle time for float16 operations
    hw_metric pe_width_bytes = 16.0;
    hw_metric pe_cycle_time_ns = 0.5;

    // Low-bit PE: separate PE for 2-bit/4-bit quantized operations
    // When kv_bits < 16, this PE handles the quantized K matrix operations
    hw_metric pe_lowbit_width_bytes = 32.0;
    hw_metric pe_lowbit_cycle_time_ns = 0.5;

    // --- Bandwidth Configuration ---
    // DRAM array to row buffer bandwidth per bank (bytes/second)
    // e.g., 32 GB/s per bank
    hw_metric dram_to_rb_bw_per_bank = 32.0e9;

    // --- Feature Toggles ---
    // Enable near-bank PIM modeling (if false, falls back to original model)
    bool enable_nearbank_model = true;

    // Process softmax in PIM (false = softmax on GPU)
    bool enable_softmax_in_pim = false;

    // Process context stage (score@V) in PIM (true = GEMV in PIM)
    bool enable_context_in_pim = true;

    // Process scoring stage (Q@K^T) in PIM (true = GEMV in PIM)
    bool enable_scoring_in_pim = true;

    // --- Asymmetric Quantization ---
    // When enabled, models the extra reduction terms needed for
    // asymmetric (affine) quantized dot products:
    //   Q·K ≈ s_Q·s_K · [ Σq̂k̂ - z_K·Σq̂ - z_Q·Σk̂ + d·z_Q·z_K ]
    // The reduction sums (Σq̂, Σk̂) are computed in PE alongside the
    // main GEMV, reusing the same data stream — zero extra DRAM reads.
    bool enable_asymmetric_quant = false;

    // Zero point for Q matrix (dequant: q = s_Q * (q̂ - z_Q))
    hw_metric zero_point_q = 0.0;

    // Zero point for K matrix (dequant: k = s_K * (k̂ - z_K))
    hw_metric zero_point_k = 0.0;

    // --- PIM Dequantization ---
    // When enabled, 2-bit KV data is first dequantized to FP16 inside PIM
    // before computing the GEMV.  Model:
    //   1. Read 2-bit data into rowbuffer (small, fast)
    //   2. PE dequantizes: 1 MUL + 1 ADD per element → FP16
    //   3. PE computes FP16 GEMV on dequantized data
    // Cost: dequant ops + FP16 GEMV ops, all in PE (no extra DRAM reads)
    bool enable_pim_dequant = false;

    // --- KV Cache Bit Width ---
    // 2: 2-bit quantization (4 elements/byte packed)
    // 4: 4-bit quantization (2 elements/byte packed)
    int kv_cache_bits = 2;

    // --- Derived Values (computed after config is set) ---
    // PE throughput in bytes/second = pe_width_bytes / pe_cycle_time_ns * 1e9
    hw_metric getPEThroughput() const {
        return pe_width_bytes / pe_cycle_time_ns * 1e9;
    }

    // Total number of banks = cubes * channels * banks_per_channel
    // Default HBM3: 8 cube * 32 channel * (4 bankgroup * 4 bank / (2 rank?))
    // But user configures this directly since it varies by architecture
    int getTotalBanks() const {
        return num_pim_banks;
    }
};

}  // namespace llm_system
