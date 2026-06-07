#include <memory>

#include "common/type.h"
#include "dram/dram_interface.h"
#include "dram/dram_request.h"
#include "dram/nearbank/nearbank_pim.h"
#include "hardware/layer_impl.h"
#include "module/tensor.h"

namespace llm_system {
class DRAMRequest;
class Tensor;

using Tensor_Ptr = std::shared_ptr<Tensor>;
using DRAMRequest_Ptr = std::shared_ptr<DRAMRequest>;

ExecStatus AttentionGenExecutionGPU(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator) {
  Tensor_Ptr input = tensor.at(0);
  Tensor_Ptr k_cache = tensor.at(1);
  Tensor_Ptr v_cache = tensor.at(2);

  auto config = device->config;
  hw_metric compute_peak_flops = config.compute_peak_flops;
  hw_metric memory_bandwidth = config.memory_bandwidth;

  // GPU GEMV efficiency: for M=1 (decode), arithmetic intensity << roofline
  // ridge, so the operation is purely memory-bound.  Peak FLOPS unreachable.
  // Effective FLOPS ≈ memory_bandwidth × arithmetic_intensity.
  // Use peak as-is (compute ≈ 0 for M=1, dominated by memory_duration in max()).
  // For M > 1 (prefill / large batch), consider a factor ~0.3-0.5 × peak.
  const hw_metric gpu_gemv_eff_factor = (input->precision_byte <= 1) ? 0.5 : 0.5;
  hw_metric compute_eff_flops = compute_peak_flops * gpu_gemv_eff_factor;

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int attention_group_size = layer_info.attention_group_size;

  time_ns time = 0;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;

  ExecStatus exec_status;
  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  // Scoring //
  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_gen();
  Sequence::Ptr seq;

  std::vector<int> shape = {1, head_dim};

  int num_seq = seq_list.size();

  int accumul_len = 0;
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = head_dim;
    n = seq->current_len + seq->num_process_token;

    for (int kv_idx = 0; kv_idx < num_kv_heads; kv_idx++) {
      flops = m * k * n * 2.0 * attention_group_size;
      total_flops += flops;

      memory_size = 1.0 * (m * k * num_heads / num_kv_heads + k * n + m * n * num_heads / num_kv_heads) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_eff_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
    }
    accumul_len += n;
  }

  if (use_ramulator) {
    k_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp =
        issueRamulator(device, LayerType::ATTENTION_GEN, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, k_cache);
    exec_status += temp;
    accumul_memory_duration = temp.memory_duration;
  }
  else {
    k_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, k_cache);
    exec_status += temp;
  }

  // GPU dequantization overhead: when KV cache is 2-bit, GPU must dequantize
  // to FP16 before GEMV.  2 ops/element (subtract zero_point + multiply by scale).
  // K_matrix: accumul_len × head_dim_per_kv_head elements
  if (input->precision_byte <= 1) {
    double dequant_elements = static_cast<double>(accumul_len) * k;
    double dequant_flops = 2.0 * dequant_elements;
    time_ns dequant_time = dequant_flops / compute_eff_flops * 1000 * 1000 * 1000;
    accumul_compute_duration += dequant_time;
    exec_status.compute_duration += dequant_time;
  }

  time_ns qk_total = std::max(accumul_compute_duration, accumul_memory_duration);
  exec_status.total_duration += qk_total;
  exec_status.qk_duration = qk_total;

  // =========================================================================
  // Context stage: scores @ V → output
  // GEMV(1, seq_len) @ (seq_len, head_dim) → (1, head_dim)
  // Repeated for each KV head, multiplied by attention_group_size
  // =========================================================================
  accumul_len = 0;
  time_ns ctx_compute_duration = 0;
  time_ns ctx_memory_duration = 0;

  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = seq->current_len + seq->num_process_token;
    n = head_dim;

    for (int kv_idx = 0; kv_idx < num_kv_heads; kv_idx++) {
      flops = m * k * n * 2.0 * attention_group_size;
      total_flops += flops;

      memory_size = 1.0 * (m * k * num_heads / num_kv_heads + k * n + m * n * num_heads / num_kv_heads) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_eff_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      ctx_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      ctx_memory_duration += memory_duration;
    }
    int _k = (k + 3) / 4 * 4;
    accumul_len += _k;
  }

  if (use_ramulator) {
    v_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp =
        issueRamulator(device, LayerType::ATTENTION_GEN, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, v_cache);
    exec_status += temp;
    ctx_memory_duration = temp.memory_duration;
  }
  else {
    v_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, v_cache);
    exec_status += temp;
  }

  // GPU dequant for V matrix (2-bit → FP16)
  if (input->precision_byte <= 1) {
    double dequant_elements = static_cast<double>(accumul_len) * n;
    double dequant_flops = 2.0 * dequant_elements;
    time_ns dequant_time = dequant_flops / compute_eff_flops * 1000 * 1000 * 1000;
    ctx_compute_duration += dequant_time;
    exec_status.compute_duration += dequant_time;
  }

  time_ns ctx_total = std::max(ctx_compute_duration, ctx_memory_duration);
  exec_status.total_duration += ctx_total;
  exec_status.score_v_duration = ctx_total;

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};

ExecStatus AttentionGenExecutionLogic(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info,
                                      bool use_ramulator) {
  Tensor_Ptr input = tensor.at(0);
  Tensor_Ptr k_cache = tensor.at(1);
  Tensor_Ptr v_cache = tensor.at(2);

  auto config = device->config;
  hw_metric compute_peak_flops =
      config.logic_memory_bandwidth * config.logic_op_b;
  hw_metric memory_bandwidth = config.logic_memory_bandwidth;
  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int attention_group_size = layer_info.attention_group_size;

  time_ns time = 0;

  int m, n, k;
  double flops = 0;
  double memory_size = 0;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;

  ExecStatus exec_status;
  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  // Scoring //
  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_gen();
  Sequence::Ptr seq;

  std::vector<int> shape = {1, head_dim};

  int num_seq = seq_list.size();

  int accumul_len = 0;
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;

  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = head_dim;
    n = seq->current_len + seq->num_process_token;

    for (int kv_idx = 0; kv_idx < num_kv_heads; kv_idx++) {
      flops = m * k * n * 2.0 * attention_group_size;
      total_flops += flops;

      memory_size = (k * n) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
    }
    int _n = (n + 3) / 4 * 4;
    accumul_len += _n;
  }

  if (use_ramulator) {
    k_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp =
        issueRamulator(device, LayerType::ATTENTION_GEN, ProcessorType::LOGIC,
                       DRAMRequestType::kGEMV, PIMOperandType::kSrc, k_cache);
    exec_status += temp;
    accumul_memory_duration = temp.memory_duration;
  }
  else {
    k_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, k_cache);
    exec_status += temp;
  }

  exec_status.total_duration +=
      std::max(accumul_compute_duration, accumul_memory_duration);

  // Context //
  accumul_len = 0;
  accumul_compute_duration = 0;
  accumul_memory_duration = 0;
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    seq = seq_list.at(seq_idx);

    m = seq->num_process_token;
    k = seq->current_len + seq->num_process_token;
    n = head_dim;

    for (int kv_idx = 0; kv_idx < num_kv_heads; kv_idx++) {
      flops = m * k * n * 2.0 * attention_group_size;
      total_flops += flops;

      memory_size = (k * n) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
    }
    int _k = (k + 3) / 4 * 4;
    accumul_len += _k;
  }

  if (use_ramulator) {
    v_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp =
        issueRamulator(device, LayerType::ATTENTION_GEN, ProcessorType::LOGIC,
                       DRAMRequestType::kGEMV, PIMOperandType::kSrc, v_cache);
    exec_status += temp;
    accumul_memory_duration = temp.memory_duration;
  }
  else {
    v_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, v_cache);
    exec_status += temp;
  }

  exec_status.total_duration +=
      std::max(accumul_compute_duration, accumul_memory_duration);

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};

ExecStatus AttentionGenExecutionPIM(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator) {
  Tensor_Ptr input = tensor.at(0);
  Tensor_Ptr k_cache = tensor.at(1);
  Tensor_Ptr v_cache = tensor.at(2);

  auto config = device->config;
  auto& nb_config = config.nearbank_pim_config;
  hw_metric pim_compute_peak_flops = config.pim_memory_bandwidth * config.pim_op_b;
  hw_metric pim_memory_bandwidth = config.pim_memory_bandwidth;
  hw_metric gpu_compute_peak_flops = config.compute_peak_flops;
  hw_metric gpu_memory_bandwidth = config.memory_bandwidth;
  if(input->precision_byte == 1){
    pim_compute_peak_flops *= 2;
    gpu_compute_peak_flops *= 2;
  }

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int attention_group_size = layer_info.attention_group_size;

  time_ns time = 0;
  int m, n, k;
  double flops = 0;
  double memory_size = 0;
  double total_flops = 0;
  double total_memory_size = 0;
  time_ns total_duration = 0;

  ExecStatus exec_status;
  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_gen();
  Sequence::Ptr seq;
  int num_seq = seq_list.size();

  bool use_nearbank = nb_config.enable_nearbank_model;
  NearbankPIMUnit::Ptr nearbank_unit = nullptr;
  if (use_nearbank) {
    nearbank_unit = NearbankPIMUnit::Create(nb_config);
  }

  int accumul_len = 0;
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;

  // =========================================================================
  // Scoring stage: Q @ K^T → attention scores
  // GEMV(1, head_dim) @ (head_dim, seq_len) → (1, seq_len)
  // Repeated for each KV head, multiplied by attention_group_size (for Q heads)
  // =========================================================================
  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    seq = seq_list.at(seq_idx);
    m = seq->num_process_token;
    k = head_dim;
    n = seq->current_len + seq->num_process_token;

    for (int kv_idx = 0; kv_idx < num_kv_heads; kv_idx++) {
      flops = m * k * n * 2.0 * attention_group_size;
      total_flops += flops;
      memory_size = (k * n) * input->precision_byte;
      total_memory_size += memory_size;

      if (use_nearbank && nb_config.enable_scoring_in_pim) {
        int comp_pb = input->precision_byte;
        if (nb_config.enable_pim_dequant && input->precision_byte <= 1) {
            comp_pb = 2;  // Q@K dequant: K is 2-bit, compute in FP16
        }
        NearbankGEMVResult gemv_result = nearbank_unit->computeGEMVLatency(
            m, k, n, input->precision_byte, comp_pb);
        // GQA: K loaded ONCE per kv_head, computed group_size times (one per Q head).
        // Latency = max(rb, pe_all × group_size) — pipelined rb + G PE passes.
        time_ns gemv_rb = gemv_result.rowbuffer_time_ns;
        time_ns gemv_pe_all = gemv_result.pe_compute_time_ns + 
            (nb_config.enable_asymmetric_quant ? gemv_result.reduction_time_ns : 0.0);
        time_ns kv_latency = std::max(gemv_rb, gemv_pe_all * attention_group_size);
        time_ns gemv_pe = gemv_result.pe_compute_time_ns * attention_group_size;
        exec_status.compute_duration += kv_latency;
        accumul_compute_duration += kv_latency;
        accumul_memory_duration += gemv_rb;
        exec_status.pim_rb_duration += gemv_rb;
        exec_status.pim_pe_duration += gemv_pe;
        exec_status.pim_rb_qk += gemv_rb;
        exec_status.pim_pe_qk += gemv_pe;
      } else if (use_nearbank) {
        time_ns comp_dur = flops / gpu_compute_peak_flops * 1000 * 1000 * 1000;
        exec_status.compute_duration += comp_dur;
        accumul_compute_duration += comp_dur;
        time_ns mem_dur = memory_size / gpu_memory_bandwidth * 1000 * 1000 * 1000;
        accumul_memory_duration += mem_dur;
      } else {
        time_ns comp_dur = flops / pim_compute_peak_flops * 1000 * 1000 * 1000;
        exec_status.compute_duration += comp_dur;
        accumul_compute_duration += comp_dur;
        time_ns mem_dur = memory_size / pim_memory_bandwidth * 1000 * 1000 * 1000;
        accumul_memory_duration += mem_dur;
      }
    }
    int _n = (n + 3) / 4 * 4;
    accumul_len += _n;
  }

  if (use_ramulator) {
    k_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp =
        issueRamulator(device, LayerType::ATTENTION_GEN, ProcessorType::PIM,
                       DRAMRequestType::kGEMV, PIMOperandType::kSrc, k_cache);
    exec_status += temp;
    if (!use_nearbank) {
      accumul_memory_duration = temp.memory_duration;
    }
  } else {
    k_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, k_cache);
    exec_status += temp;
  }

  if (!use_nearbank) {
    double opb = total_flops / total_memory_size;
    exec_status.total_duration += accumul_memory_duration * opb;
  } else {
    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }

  // =========================================================================
  // Softmax stage
  // =========================================================================
  time_ns softmax_accum = 0;
  if (!use_nearbank || nb_config.enable_softmax_in_pim) {
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);
      m = seq->num_process_token;
      n = seq->current_len + seq->num_process_token;
      double softmax_flops = 7.0 * m * n * num_heads;
      if (!use_nearbank) {
        time_ns softmax_dur = softmax_flops / pim_compute_peak_flops * 1000 * 1000 * 1000;
        exec_status.total_duration += softmax_dur;
        softmax_accum += softmax_dur;
      }
    }
  } else if (use_nearbank) {
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);
      m = seq->num_process_token;
      n = seq->current_len + seq->num_process_token;
      double softmax_flops = 7.0 * m * n * num_heads;
      time_ns softmax_dur = softmax_flops / gpu_compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.total_duration += softmax_dur;
      softmax_accum += softmax_dur;
    }
  }
  exec_status.softmax_duration = softmax_accum;

  // =========================================================================
  // Context stage: scores @ V → output
  // GEMV(1, seq_len) @ (seq_len, head_dim) → (1, head_dim)
  // =========================================================================

  // Reset accumulators for context stage
  time_ns ctx_compute_duration = 0;
  time_ns ctx_memory_duration = 0;
  accumul_len = 0;

  for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
    seq = seq_list.at(seq_idx);
    m = seq->num_process_token;
    k = seq->current_len + seq->num_process_token;
    n = head_dim;

    for (int kv_idx = 0; kv_idx < num_kv_heads; kv_idx++) {
      flops = m * k * n * 2.0 * attention_group_size;
      total_flops += flops;
      memory_size = (k * n) * input->precision_byte;
      total_memory_size += memory_size;

      if (use_nearbank && nb_config.enable_context_in_pim) {
        int comp_pb = input->precision_byte;
        if (input->precision_byte <= 1) {
            comp_pb = 2;  // Score@V: scores are FP16, need FP16 dequant
        }
        NearbankGEMVResult gemv_result = nearbank_unit->computeGEMVLatency(
            m, k, n, input->precision_byte, comp_pb);
        // GQA: V loaded ONCE per kv_head, computed group_size times
        time_ns gemv_rb = gemv_result.rowbuffer_time_ns;
        time_ns gemv_pe_all = gemv_result.pe_compute_time_ns + 
            (nb_config.enable_asymmetric_quant ? gemv_result.reduction_time_ns : 0.0);
        time_ns kv_latency = std::max(gemv_rb, gemv_pe_all * attention_group_size);
        time_ns gemv_pe = gemv_result.pe_compute_time_ns * attention_group_size;
        ctx_compute_duration += kv_latency;
        ctx_memory_duration += gemv_rb;
        exec_status.pim_rb_duration += gemv_rb;
        exec_status.pim_pe_duration += gemv_pe;
        exec_status.pim_rb_sv += gemv_rb;
        exec_status.pim_pe_sv += gemv_pe;
      } else if (use_nearbank) {
        time_ns comp_dur = flops / gpu_compute_peak_flops * 1000 * 1000 * 1000;
        ctx_compute_duration += comp_dur;
        time_ns mem_dur = memory_size / gpu_memory_bandwidth * 1000 * 1000 * 1000;
        ctx_memory_duration += mem_dur;
      } else {
        time_ns comp_dur = flops / pim_compute_peak_flops * 1000 * 1000 * 1000;
        ctx_compute_duration += comp_dur;
        time_ns mem_dur = memory_size / pim_memory_bandwidth * 1000 * 1000 * 1000;
        ctx_memory_duration += mem_dur;
      }
    }
    int _k = (k + 3) / 4 * 4;
    accumul_len += _k;
  }

  if (use_ramulator) {
    v_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp =
        issueRamulator(device, LayerType::ATTENTION_GEN, ProcessorType::PIM,
                       DRAMRequestType::kGEMV, PIMOperandType::kSrc, v_cache);
    exec_status += temp;
    if (!use_nearbank) {
      ctx_memory_duration = temp.memory_duration;
    }
  } else {
    v_cache->setShape({accumul_len, head_dim * num_kv_heads});
    ExecStatus temp;
    temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, v_cache);
    exec_status += temp;
  }

  if (!use_nearbank) {
    double opb = total_flops / total_memory_size;
    exec_status.total_duration += ctx_memory_duration * opb;
  } else {
    exec_status.total_duration += std::max(ctx_compute_duration, ctx_memory_duration);
    exec_status.compute_duration += ctx_compute_duration;
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             pim_compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            pim_memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  // Per-stage timing
  if (use_nearbank) {
    exec_status.qk_duration = std::max(accumul_compute_duration, accumul_memory_duration);
    exec_status.score_v_duration = std::max(ctx_compute_duration, ctx_memory_duration);
    if (!nb_config.enable_softmax_in_pim) {
      // softmax duration set in the softmax block above
    }
  }

  // KV cache quantization overhead (decode mode, asymmetric quant on GPU)
  // Each new token's K and V must be quantized before storing to cache.
  // d_kv = num_kv_heads * head_dim elements per matrix
  // Ops: 2 subtracts + 2 multiplies per element (asymmetric quant)
  // Memory: read FP16 K,V + write quantized K,V
  {
    int d_kv = num_kv_heads * head_dim;
    double kv_elements = 2.0 * d_kv * num_seq;  // K + V, all sequences
    double kv_flops = 4.0 * kv_elements;          // sub + mul per element
    double kv_bytes = kv_elements * 2.0;           // FP16 read
    time_ns kv_comp = kv_flops / gpu_compute_peak_flops * 1000 * 1000 * 1000;
    time_ns kv_mem = kv_bytes / gpu_memory_bandwidth * 1000 * 1000 * 1000;
    exec_status.kv_quant_duration = std::max(kv_comp, kv_mem);
    exec_status.kv_quant_bytes = kv_bytes;
  }

  return exec_status;
};

ExecStatus MultiLatentAttentionGenExecutionGPU(Device_Ptr device,
  std::vector<Tensor_Ptr> tensor,
  BatchedSequence::Ptr sequences_metadata,
  LayerInfo layer_info, bool use_ramulator) {

  Tensor_Ptr input = tensor.at(0);
  Tensor_Ptr k_cache = tensor.at(0);
  Tensor_Ptr v_cache = tensor.at(0);
  bool compressed_kv = true;
  if(tensor.size() > 1){ // not use compressed_kv
    compressed_kv = false;
    k_cache = tensor.at(1);
    v_cache = tensor.at(2);
  }

  auto config = device->config;
  hw_metric compute_peak_flops = config.compute_peak_flops;
  hw_metric memory_bandwidth = config.memory_bandwidth;

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  bool use_flash_mla = layer_info.use_flash_mla;

  time_ns time = 0;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;

  ExecStatus exec_status;
  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  std::vector<int> orig_shape = input->shape;
  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_gen();
  Sequence::Ptr seq;

  std::vector<int> shape = {1, head_dim};

  int num_seq = seq_list.size();

  int accumul_len = 0;
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;

  if(use_flash_mla){
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      accumul_len += n;

      flops = 2.0 * m * (head_dim + qk_rope_head_dim) * n * num_heads + // score
              2.0 * m * (head_dim) * n * num_heads; // context
      total_flops += flops;


      memory_size = 1.0 * (m * (head_dim + qk_rope_head_dim) + // query
                    1.0 * n * (head_dim + qk_rope_head_dim) + // key
                    1.0 * n * head_dim + // value
                    1.0 * m * head_dim) * // output 
                    num_heads * input->precision_byte;
      total_memory_size += memory_size;

      accumul_compute_duration += flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += accumul_compute_duration;

      accumul_memory_duration +=
          memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      exec_status.memory_duration += accumul_memory_duration;
    }

    if(use_ramulator) {
      ExecStatus temp;

      // read query
      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      // read key
      input->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);                        

      // read value
      input->setShape({accumul_len, head_dim * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);                        

      // write output
      input->setShape({num_seq, num_heads * head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);                        

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len, head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({num_seq, num_heads * head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }
  else{

    // Score //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      for (int head_idx = 0; head_idx < num_heads; head_idx++) {
        flops = m * k * n * 2.0;
        total_flops += flops;

        memory_size = (m * k + k * n + m * n) * input->precision_byte;
        total_memory_size += memory_size;

        compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
        exec_status.compute_duration += compute_duration;
        accumul_compute_duration += compute_duration;

        memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
        accumul_memory_duration += memory_duration;
      }
      accumul_len += n;
    }

    if(use_ramulator) {
      ExecStatus temp;
      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      k_cache->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, k_cache);                        

      input->setShape({num_heads, accumul_len});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);                        

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      k_cache->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, k_cache);
      exec_status += temp;

      input->setShape({num_heads, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);

    // Softmax //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      n = seq->current_len + seq->num_process_token;

      flops = 7.0 * m * n * num_heads; // scale + mask + softmax
      total_flops += flops;

      memory_size = (2.0 * m * n * num_heads) *input->precision_byte;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if(use_ramulator){
        memory_duration = 0;
        
        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;

        // store output
        temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;
      }
      else{
        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
        exec_status += temp;

        // store output
        temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }

      exec_status.total_duration += std::max(compute_duration, memory_duration);;
    }

    // Context //
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = seq->current_len + seq->num_process_token;
      n = head_dim;

      for (int head_idx = 0; head_idx < num_heads; head_idx++) {
        flops = m * k * n * 2.0; 
        total_flops += flops;

        memory_size = (m * k + k * n + m * n) * input->precision_byte;
        total_memory_size += memory_size;

        compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
        exec_status.compute_duration += compute_duration;
        accumul_compute_duration += compute_duration;

        memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
        accumul_memory_duration += memory_duration;
      }
      accumul_len += k;
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;
      ExecStatus temp;

      input->setShape({num_heads, accumul_len});
      temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
      
      v_cache->setShape({accumul_len, head_dim * num_heads});
      temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, v_cache);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      input->setShape({num_seq, num_heads * head_dim});
      temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    }
    else{
      ExecStatus temp;
      
      input->setShape({num_heads, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      v_cache->setShape({accumul_len, head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, v_cache);
      exec_status += temp;

      input->setShape({num_seq, num_heads * head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
  compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
  memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;
  input->setShape(orig_shape); // restore orig shape of input
  return exec_status;
};

ExecStatus MultiLatentAttentionGenExecutionLogic(Device_Ptr device,
  std::vector<Tensor_Ptr> tensor,
  BatchedSequence::Ptr sequences_metadata,
  LayerInfo layer_info, bool use_ramulator) {

  Tensor_Ptr input = tensor.at(0);
  Tensor_Ptr k_cache = tensor.at(0);
  Tensor_Ptr v_cache = tensor.at(0);
  bool compressed_kv = true;
  if(tensor.size() > 1){ // not use compressed_kv
    compressed_kv = false;
    k_cache = tensor.at(1);
    v_cache = tensor.at(2);
  }

  auto config = device->config;
  hw_metric compute_peak_flops =
    config.logic_memory_bandwidth * config.logic_op_b;
  hw_metric memory_bandwidth = config.logic_memory_bandwidth;
  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  int use_flash_mla = layer_info.use_flash_mla;

  time_ns time = 0;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;

  ExecStatus exec_status;
  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  // Scoring //
  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_gen();
  Sequence::Ptr seq;
  std::vector<int> orig_shape = input->shape;

  std::vector<int> shape = {1, head_dim};

  int num_seq = seq_list.size();

  int accumul_len = 0;
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;

  if(use_flash_mla){
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      accumul_len += n;

      flops = 2.0 * m * (head_dim + qk_rope_head_dim) * n * num_heads + // score
              2.0 * m * (head_dim) * n * num_heads; // context
      total_flops += flops;


      memory_size = 1.0 * (m * (head_dim + qk_rope_head_dim) + // query
                    1.0 * n * (head_dim + qk_rope_head_dim) + // key
                    1.0 * n * head_dim + // value
                    1.0 * m * head_dim) * // output 
                    num_heads * input->precision_byte;
      total_memory_size += memory_size;

      accumul_compute_duration += flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += accumul_compute_duration;

      accumul_memory_duration +=
          memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      exec_status.memory_duration += accumul_memory_duration;
    }

    if(use_ramulator) {
      ExecStatus temp;

      // read query
      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      // read key
      input->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);                        

      // read value
      input->setShape({accumul_len, head_dim * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);                        

      // write output
      input->setShape({num_seq, num_heads * head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);                        

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len, head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({num_seq, num_heads * head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }
  else{

    // Score //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      for (int head_idx = 0; head_idx < num_heads; head_idx++) {
        flops = m * k * n * 2.0;
        total_flops += flops;

        memory_size = (m * k + k * n + m * n) * input->precision_byte;
        total_memory_size += memory_size;

        compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
        exec_status.compute_duration += compute_duration;
        accumul_compute_duration += compute_duration;

        memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
        accumul_memory_duration += memory_duration;
      }
      accumul_len += n;
    }

    if(use_ramulator) {
      ExecStatus temp;
      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      k_cache->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, k_cache);                        

      input->setShape({num_heads, accumul_len});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);                        

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      k_cache->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, k_cache);
      exec_status += temp;

      input->setShape({num_heads, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);

    // Softmax //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      n = seq->current_len + seq->num_process_token;

      flops = 7.0 * m * n * num_heads; // scale + mask + softmax
      total_flops += flops;

      memory_size = (2.0 * m * n * num_heads) *input->precision_byte;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if(use_ramulator){
        memory_duration = 0;
        
        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;

        // store output
        temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;
      }
      else{
        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
        exec_status += temp;

        // store output
        temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }

      exec_status.total_duration += std::max(compute_duration, memory_duration);;
    }

    // Context //
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = seq->current_len + seq->num_process_token;
      n = head_dim;

      for (int head_idx = 0; head_idx < num_heads; head_idx++) {
        flops = m * k * n * 2.0; 
        total_flops += flops;

        memory_size = (m * k + k * n + m * n) * input->precision_byte;
        total_memory_size += memory_size;

        compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
        exec_status.compute_duration += compute_duration;
        accumul_compute_duration += compute_duration;

        memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
        accumul_memory_duration += memory_duration;
      }
      accumul_len += k;
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;
      ExecStatus temp;

      input->setShape({num_heads, accumul_len});
      temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
      
      v_cache->setShape({accumul_len, head_dim * num_heads});
      temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, v_cache);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      input->setShape({num_seq, num_heads * head_dim});
      temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    }
    else{
      ExecStatus temp;
      
      input->setShape({num_heads, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      v_cache->setShape({accumul_len, head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, v_cache);
      exec_status += temp;

      input->setShape({num_seq, num_heads * head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
  compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
  memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;
  input->setShape(orig_shape); // restore orig shape of input
  return exec_status;
};

ExecStatus MultiLatentAttentionGenExecutionPIM(Device_Ptr device,
  std::vector<Tensor_Ptr> tensor,
  BatchedSequence::Ptr sequences_metadata,
  LayerInfo layer_info, bool use_ramulator) {

  Tensor_Ptr input = tensor.at(0);
  Tensor_Ptr k_cache = tensor.at(0);
  Tensor_Ptr v_cache = tensor.at(0);
  bool compressed_kv = true;
  if(tensor.size() > 1){ // not use compressed_kv
    compressed_kv = false;
    k_cache = tensor.at(1);
    v_cache = tensor.at(2);
  }

  auto config = device->config;
  hw_metric compute_peak_flops = config.pim_memory_bandwidth * config.pim_op_b;
  hw_metric memory_bandwidth = config.pim_memory_bandwidth;
  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  int use_flash_mla = layer_info.use_flash_mla;

  time_ns time = 0;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;

  ExecStatus exec_status;
  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  // Scoring //
  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_gen();
  Sequence::Ptr seq;
  std::vector<int> orig_shape = input->shape;

  std::vector<int> shape = {1, head_dim};

  int num_seq = seq_list.size();

  int accumul_len = 0;
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;

  if(use_flash_mla){
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      accumul_len += n;

      flops = 2.0 * m * (head_dim + qk_rope_head_dim) * n * num_heads + // score
              2.0 * m * (head_dim) * n * num_heads; // context
      total_flops += flops;


      memory_size = 1.0 * (m * (head_dim + qk_rope_head_dim) + // query
                    1.0 * n * (head_dim + qk_rope_head_dim) + // key
                    1.0 * n * head_dim + // value
                    1.0 * m * head_dim) * // output 
                    num_heads * input->precision_byte;
      total_memory_size += memory_size;

      accumul_compute_duration += flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += accumul_compute_duration;

      accumul_memory_duration +=
          memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      exec_status.memory_duration += accumul_memory_duration;
    }

    if(use_ramulator) {
      ExecStatus temp;

      // read query
      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      // read key
      input->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);                        

      // read value
      input->setShape({accumul_len, head_dim * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);                        

      // write output
      input->setShape({num_seq, num_heads * head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);                        

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len, head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({num_seq, num_heads * head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }
  else{
    // Score //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = head_dim + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      for (int head_idx = 0; head_idx < num_heads; head_idx++) {
        flops = m * k * n * 2.0;
        total_flops += flops;

        memory_size = (m * k + k * n + m * n) * input->precision_byte;
        total_memory_size += memory_size;

        compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
        exec_status.compute_duration += compute_duration;
        accumul_compute_duration += compute_duration;

        memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
        accumul_memory_duration += memory_duration;
      }
      accumul_len += n;
    }

    if(use_ramulator) {
      ExecStatus temp;
      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      k_cache->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, k_cache);                        

      input->setShape({num_heads, accumul_len});
      temp += issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);                        

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({num_seq * num_heads, head_dim + qk_rope_head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      k_cache->setShape({accumul_len, (head_dim + qk_rope_head_dim) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, k_cache);
      exec_status += temp;

      input->setShape({num_heads, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);

    // Softmax //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      n = seq->current_len + seq->num_process_token;

      flops = 7.0 * m * n * num_heads; // scale + mask + softmax
      total_flops += flops;

      memory_size = (2.0 * m * n * num_heads) *input->precision_byte;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if(use_ramulator){
        memory_duration = 0;
        
        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;

        // store output
        temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;
      }
      else{
        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
        exec_status += temp;

        // store output
        temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }

      exec_status.total_duration += std::max(compute_duration, memory_duration);;
    }

    // Context //
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = seq->current_len + seq->num_process_token;
      n = head_dim;

      for (int head_idx = 0; head_idx < num_heads; head_idx++) {
        flops = m * k * n * 2.0; 
        total_flops += flops;

        memory_size = (m * k + k * n + m * n) * input->precision_byte;
        total_memory_size += memory_size;

        compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
        exec_status.compute_duration += compute_duration;
        accumul_compute_duration += compute_duration;

        memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
        accumul_memory_duration += memory_duration;
      }
      accumul_len += k;
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;
      ExecStatus temp;

      input->setShape({num_heads, accumul_len});
      temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
      
      v_cache->setShape({accumul_len, head_dim * num_heads});
      temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, v_cache);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      input->setShape({num_seq, num_heads * head_dim});
      temp = issueRamulator(device, LayerType::MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    }
    else{
      ExecStatus temp;
      
      input->setShape({num_heads, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      v_cache->setShape({accumul_len, head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, v_cache);
      exec_status += temp;

      input->setShape({num_seq, num_heads * head_dim});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }
    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
  compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
  memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;
  input->setShape(orig_shape); // restore orig shape of input
  return exec_status;
};

ExecStatus AbsorbMLAGenExecutionGPU(Device_Ptr device,
  std::vector<Tensor_Ptr> tensor,
  BatchedSequence::Ptr sequences_metadata,
  LayerInfo layer_info, bool use_ramulator) {

  Tensor_Ptr input = tensor.at(0);

  auto config = device->config;
  hw_metric compute_peak_flops = config.compute_peak_flops;
  hw_metric memory_bandwidth = config.memory_bandwidth;

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  int kv_lora_rank = layer_info.kv_lora_rank;
  bool use_flash_mla = layer_info.use_flash_mla;

  time_ns time = 0;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;

  ExecStatus exec_status;
  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_gen();
  Sequence::Ptr seq;

  std::vector<int> orig_shape = input->shape;

  int num_seq = seq_list.size();

  int accumul_len = 0; // num_seq x seqLen
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;

  if(use_flash_mla){
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = kv_lora_rank + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      accumul_len += n;

      flops = 2.0 * num_heads * (kv_lora_rank + qk_rope_head_dim) * n + // score
              2.0 * num_heads * (kv_lora_rank) * n; // context
      total_flops += flops;


      memory_size = 1.0 * (num_heads * (kv_lora_rank + qk_rope_head_dim) + // query
                    1.0 * n * (kv_lora_rank + qk_rope_head_dim) + // latent kv and pe cache
                    1.0 * num_heads * kv_lora_rank) * // output 
                    input->precision_byte;
      total_memory_size += memory_size;

      accumul_compute_duration += flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += accumul_compute_duration;

      accumul_memory_duration +=
          memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      exec_status.memory_duration += accumul_memory_duration;
    }

    if(use_ramulator) {
      ExecStatus temp;

      // read query
      input->setShape({num_seq * num_heads, (kv_lora_rank + qk_rope_head_dim)});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      // read latent kv and pe cache
      input->setShape({accumul_len, (kv_lora_rank + qk_rope_head_dim)});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);                                          

      // write output
      input->setShape({num_seq, num_heads * kv_lora_rank});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);                        

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({num_seq * num_heads, (kv_lora_rank + qk_rope_head_dim)});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len, (kv_lora_rank + qk_rope_head_dim)});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({num_seq, num_heads * kv_lora_rank});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }
  else{
    // Scoring for NoPE//
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = kv_lora_rank;
      n = seq->current_len + seq->num_process_token;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n + 1.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
      accumul_len += n;
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read input
      input->setShape({num_seq, (1 * kv_lora_rank) * num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read compressed_KV
      input->setShape({kv_lora_rank, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // store intermediate value
      input->setShape({num_heads, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
              DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    }
    else{
      ExecStatus temp;

      // read input
      input->setShape({num_seq, (1 * kv_lora_rank) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read compressed_KV
      input->setShape({kv_lora_rank, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // store intermediate value
      input->setShape({num_heads, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);

    // Scoring for RoPE//
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n + 1.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
      
      accumul_len += n;
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read input
      input->setShape({num_seq, qk_rope_head_dim * num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read latent_PE
      input->setShape({qk_rope_head_dim, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // store intermediate value
      input->setShape({num_heads, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
              DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      // read input
      input->setShape({num_seq, qk_rope_head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read latent_PE
      input->setShape({qk_rope_head_dim, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // store intermediate value
      input->setShape({num_heads, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }
    
    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);


    // Scale + mask + Softmax //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      n = seq->current_len + seq->num_process_token;

      flops = 7.0 * m * n * num_heads; // scale + mask + softmax
      total_flops += flops;

      memory_size = (2.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;
      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if(use_ramulator){
        memory_duration = 0;
        
        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
                DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;

        // store output
        temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
                DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;
      }
      else{
        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
        exec_status += temp;

        // store output
        temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }

      exec_status.total_duration += std::max(compute_duration, memory_duration);;
    }

    // Context //
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = seq->current_len + seq->num_process_token;
      n = kv_lora_rank;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;
      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n +
        1.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
      accumul_len += k;
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read score output
      input->setShape({num_heads, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read compressed_kv
      input->setShape({accumul_len, kv_lora_rank});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;


      // store context out
      input->setShape({num_seq * kv_lora_rank, num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::GPU,
              DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    }
    else{
      ExecStatus temp;

      // read score output
      input->setShape({num_heads, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read compressed_kv
      input->setShape({accumul_len, kv_lora_rank});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);
      exec_status += temp;

      // store context out
      input->setShape({num_seq * kv_lora_rank, num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }

  // exec_status.total_duration = total_duration;

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
  compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
  memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  input->setShape(orig_shape); // restore orig shape of input
  return exec_status;
};

ExecStatus AbsorbMLAGenExecutionLogic(Device_Ptr device,
  std::vector<Tensor_Ptr> tensor,
  BatchedSequence::Ptr sequences_metadata,
  LayerInfo layer_info, bool use_ramulator) {

  Tensor_Ptr input = tensor.at(0);

  auto config = device->config;
  hw_metric compute_peak_flops =
      config.logic_memory_bandwidth * config.logic_op_b;
  hw_metric memory_bandwidth = config.logic_memory_bandwidth;
  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  int kv_lora_rank = layer_info.kv_lora_rank;
  bool use_flash_mla = layer_info.use_flash_mla;

  time_ns time = 0;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;

  ExecStatus exec_status;
  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_gen();
  Sequence::Ptr seq;

  std::vector<int> orig_shape = input->shape;

  int num_seq = seq_list.size();

  int accumul_len = 0; // num_seq x seqLen
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;

  if(use_flash_mla){
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = kv_lora_rank + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      accumul_len += n;

      flops = 2.0 * num_heads * (kv_lora_rank + qk_rope_head_dim) * n + // score
              2.0 * num_heads * (kv_lora_rank) * n; // context
      total_flops += flops;


      memory_size = 1.0 * (num_heads * (kv_lora_rank + qk_rope_head_dim) + // query
                    1.0 * n * (kv_lora_rank + qk_rope_head_dim) + // latent kv and pe cache
                    1.0 * num_heads * kv_lora_rank) * // output 
                    input->precision_byte;
      total_memory_size += memory_size;

      accumul_compute_duration += flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += accumul_compute_duration;

      accumul_memory_duration +=
          memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      exec_status.memory_duration += accumul_memory_duration;
    }

    if(use_ramulator) {
      ExecStatus temp;

      // read query
      input->setShape({num_seq * num_heads, (kv_lora_rank + qk_rope_head_dim)});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      // read latent kv and pe cache
      input->setShape({accumul_len, (kv_lora_rank + qk_rope_head_dim)});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);                                          

      // write output
      input->setShape({num_seq, num_heads * kv_lora_rank});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);                        

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({num_seq * num_heads, (kv_lora_rank + qk_rope_head_dim)});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len, (kv_lora_rank + qk_rope_head_dim)});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({num_seq, num_heads * kv_lora_rank});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }
  else{
    
    // Scoring for NoPE//
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = kv_lora_rank;
      n = seq->current_len + seq->num_process_token;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n + 1.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
      accumul_len += n;
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read input
      input->setShape({num_seq, (1 * kv_lora_rank) * num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read compressed_KV
      input->setShape({kv_lora_rank, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // store intermediate value
      input->setShape({num_heads, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
              DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    }
    else{
      ExecStatus temp;

      // read input
      input->setShape({num_seq, (1 * kv_lora_rank) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read compressed_KV
      input->setShape({kv_lora_rank, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      // store intermediate value
      input->setShape({num_heads, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);

    // Scoring for RoPE//
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n + 1.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
      
      accumul_len += n;
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read input
      input->setShape({num_seq, qk_rope_head_dim * num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read latent_PE
      input->setShape({qk_rope_head_dim, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // store intermediate value
      input->setShape({num_heads, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
              DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      // read input
      input->setShape({num_seq, qk_rope_head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read latent_PE
      input->setShape({qk_rope_head_dim, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      // store intermediate value
      input->setShape({num_heads, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }
    
    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);


    // Scale + mask + Softmax //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      n = seq->current_len + seq->num_process_token;

      flops = 7.0 * m * n * num_heads; // scale + mask + softmax
      total_flops += flops;

      memory_size = (2.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;
      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if(use_ramulator){
        memory_duration = 0;
        
        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
                DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;

        // store output
        temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
                DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;
      }
      else{
        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
        exec_status += temp;

        // store output
        temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }

      exec_status.total_duration += std::max(compute_duration, memory_duration);;
    }

    // Context //
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = seq->current_len + seq->num_process_token;
      n = kv_lora_rank;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;
      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n +
        1.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;

      // k_cache->setShape(shape);
      accumul_len += k;
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read score output
      input->setShape({num_heads, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read compressed_kv
      input->setShape({accumul_len, kv_lora_rank});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // store context out
      input->setShape({num_seq * kv_lora_rank, num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::LOGIC,
              DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    }
    else{
      ExecStatus temp;

      // read score output
      input->setShape({num_heads, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read compressed_kv
      input->setShape({accumul_len, kv_lora_rank});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, input);
      exec_status += temp;
      
      // store context out
      input->setShape({num_seq * kv_lora_rank, num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
  compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
  memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  input->setShape(orig_shape); // restore orig shape of input
  return exec_status;
};

ExecStatus AbsorbMLAGenExecutionPIM(Device_Ptr device,
  std::vector<Tensor_Ptr> tensor,
  BatchedSequence::Ptr sequences_metadata,
  LayerInfo layer_info, bool use_ramulator) {

  Tensor_Ptr input = tensor.at(0);

  auto config = device->config;
  hw_metric compute_peak_flops = config.pim_memory_bandwidth * config.pim_op_b;
  hw_metric memory_bandwidth = config.pim_memory_bandwidth;
  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  int head_dim = layer_info.head_dim;
  int num_heads = layer_info.num_heads;
  int num_kv_heads = layer_info.num_kv_heads;
  int qk_rope_head_dim = layer_info.qk_rope_head_dim;
  int kv_lora_rank = layer_info.kv_lora_rank;
  bool use_flash_mla = layer_info.use_flash_mla;

  time_ns time = 0;

  int m, n, k;
  double flops, memory_size;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  time_ns total_duration = 0;

  ExecStatus exec_status;
  if (sequences_metadata->get_gen_process_token() == 0) {
    return exec_status;
  }

  std::vector<Sequence::Ptr> seq_list = sequences_metadata->get_gen();
  Sequence::Ptr seq;

  std::vector<int> orig_shape = input->shape;

  int num_seq = seq_list.size();

  int accumul_len = 0; // num_seq x seqLen
  time_ns accumul_compute_duration = 0;
  time_ns accumul_memory_duration = 0;
  if(use_flash_mla){
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = kv_lora_rank + qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      accumul_len += n;

      flops = 2.0 * num_heads * (kv_lora_rank + qk_rope_head_dim) * n + // score
              2.0 * num_heads * (kv_lora_rank) * n; // context
      total_flops += flops;


      memory_size = 1.0 * (num_heads * (kv_lora_rank + qk_rope_head_dim) + // query
                    1.0 * n * (kv_lora_rank + qk_rope_head_dim) + // latent kv and pe cache
                    1.0 * num_heads * kv_lora_rank) * // output 
                    input->precision_byte;
      total_memory_size += memory_size;

      accumul_compute_duration += flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += accumul_compute_duration;

      accumul_memory_duration +=
          memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      exec_status.memory_duration += accumul_memory_duration;
    }

    if(use_ramulator) {
      ExecStatus temp;

      // read query
      input->setShape({num_seq * num_heads, (kv_lora_rank + qk_rope_head_dim)});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);

      // read latent kv and pe cache
      input->setShape({accumul_len, (kv_lora_rank + qk_rope_head_dim)});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kRead, PIMOperandType::kDRAM, input);                                          

      // write output
      input->setShape({num_seq, num_heads * kv_lora_rank});
      temp += issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
                            DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);                        

      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      input->setShape({num_seq * num_heads, (kv_lora_rank + qk_rope_head_dim)});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({accumul_len, (kv_lora_rank + qk_rope_head_dim)});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      input->setShape({num_seq, num_heads * kv_lora_rank});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }
  else{  
    // Scoring for NoPE//
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = kv_lora_rank;
      n = seq->current_len + seq->num_process_token;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n + 1.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
      accumul_len += n;
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read input
      input->setShape({num_seq, (1 * kv_lora_rank) * num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read compressed_KV
      input->setShape({kv_lora_rank, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // store intermediate value
      input->setShape({num_heads, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
              DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    }
    else{
      ExecStatus temp;

      // read input
      input->setShape({num_seq, (1 * kv_lora_rank) * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read compressed_KV
      input->setShape({kv_lora_rank, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      // store intermediate value
      input->setShape({num_heads, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);

    // Scoring for RoPE//
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = qk_rope_head_dim;
      n = seq->current_len + seq->num_process_token;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;

      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n + 1.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
      
      accumul_len += n;
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read input
      input->setShape({num_seq, qk_rope_head_dim * num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read latent_PE
      input->setShape({qk_rope_head_dim, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // store intermediate value
      input->setShape({num_heads, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
              DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration = temp.memory_duration;
    }
    else{
      ExecStatus temp;

      // read input
      input->setShape({num_seq, qk_rope_head_dim * num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read latent_PE
      input->setShape({qk_rope_head_dim, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      // store intermediate value
      input->setShape({num_heads, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }
    
    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);


    // Scale + mask + Softmax //
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      time_ns compute_duration = 0;
      time_ns memory_duration = 0;

      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      n = seq->current_len + seq->num_process_token;

      flops = 7.0 * m * n * num_heads; // scale + mask + softmax
      total_flops += flops;

      memory_size = (2.0 * m * n * num_heads) * input->precision_byte;
      total_memory_size += memory_size;
      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

      if(use_ramulator){
        memory_duration = 0;
        
        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
                DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;

        // store output
        temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
                DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
        exec_status += temp;
        memory_duration += temp.memory_duration;
      }
      else{
        ExecStatus temp;

        // read input
        input->setShape({m, n * num_heads});
        temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
        exec_status += temp;

        // store output
        temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
        exec_status += temp;
      }

      exec_status.total_duration += std::max(compute_duration, memory_duration);;
    }

    // Context //
    accumul_len = 0;
    accumul_compute_duration = 0;
    accumul_memory_duration = 0;
    for (int seq_idx = 0; seq_idx < num_seq; seq_idx++) {
      seq = seq_list.at(seq_idx);

      m = seq->num_process_token;
      k = seq->current_len + seq->num_process_token;
      n = kv_lora_rank;

      flops = 1.0 * m * k * n * 2.0 * num_heads;
      total_flops += flops;
      memory_size = (1.0 * m * k * num_heads + 1.0 * k * n +
        1.0 * m * n * num_heads) * input->precision_byte;
    
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration += memory_duration;
      accumul_len += k;
    }

    if (use_ramulator) {
      accumul_memory_duration = 0;

      ExecStatus temp;

      // read score output
      input->setShape({num_heads, accumul_len});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;

      // read compressed_kv
      input->setShape({accumul_len, kv_lora_rank});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
              DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    
      // store context out
      input->setShape({num_seq * kv_lora_rank, num_heads});
      temp = issueRamulator(device, LayerType::ABSORBED_MLA_GEN, ProcessorType::PIM,
              DRAMRequestType::kWrite, PIMOperandType::kDRAM, input);
      exec_status += temp;
      accumul_memory_duration += temp.memory_duration;
    }
    else{
      ExecStatus temp;

      // read score output
      input->setShape({num_heads, accumul_len});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      // read compressed_kv
      input->setShape({accumul_len, kv_lora_rank});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, input);
      exec_status += temp;

      // store context out
      input->setShape({num_seq * kv_lora_rank, num_heads});
      temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kWrite, input);
      exec_status += temp;
    }

    exec_status.total_duration += std::max(accumul_compute_duration, accumul_memory_duration);
  }

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
  compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
  memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  input->setShape(orig_shape); // restore orig shape of input
  return exec_status;
};

}  // namespace llm_system