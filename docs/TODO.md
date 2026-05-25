# TODO — 问题分析与修复计划

## 问题 1: exp5 (2bit All-PIM) Score@V 缺少 V 反量化开销

### 问题描述

在 exp5 中，Score@V 阶段：
- 输入的 attention scores (S) 是 FP16 精度（来自 Q@K 的 GEMV 输出）
- V 矩阵存储在 KV cache 中为 2-bit 量化格式
- 当前代码在 `AttentionGenExecutionPIM` (attention_gen_impl.cpp:449-451) 对 Context 阶段调用：
  ```cpp
  nearbank_unit->computeGEMVLatency(m, k, n, input->precision_byte);
  ```
  其中 `input->precision_byte = 1`，意味着 S 和 V 都被当作 2-bit 处理。

### 为什么这是错误的

- Q@K 阶段可以用 2-bit 空间计算（通过 asymmetric quant 分解公式），因为 Q 虽为 FP16 但只有 1 个 token（~512 元素），相比 K 矩阵（~2M 元素）数据量可忽略。
- Score@V 阶段不同：S 有 16528 个 FP16 元素（~33KB），V 有 ~2M 个 2-bit 元素（~528KB）。S 不可忽略，且 S 是 FP16 没有量化参数可用。
- 正确的做法：V 必须先反量化到 FP16，再做 FP16 GEMV（这正是 exp6 的 `enable_pim_dequant` 做的事）。

### 代码位置

- `src/hardware/attention_gen_impl.cpp:449-451` — Context 阶段调用 `computeGEMVLatency`
- `src/dram/nearbank/nearbank_pim.cpp:167-187` — PIM dequant 开销只检查 `enable_pim_dequant` flag

### 修复方案

**方案 A（推荐）：exp5 配置中为 Context 启用 dequant**

修改 `config_exp5_2bit_allpim.yaml`，添加 `enable_pim_dequant: true`。

但这会让 Q@K 也被 dequantize（因为 dequant 作用于整个 GEMV 的 K 矩阵）。对 Q@K 而言，asymmetric quant 已经在 2-bit 空间正确计算了，再加 dequant 反而是重复的。

更好的做法：**让 `computeGEMVLatency` 区分 Scoring 和 Context 阶段的 element_size**。Context 阶段传入 `element_size_bytes=2` (FP16)，Rowbuffer 数据量仍用 2-bit（因为 V 以 2-bit 存储），PE 时间需要额外加 dequant 开销。

**方案 B：在 `computeGEMVLatency` 中增加参数区分 S 和 V 的精度**

修改接口，允许分别指定矩阵 A（S，FP16）和矩阵 B（V，2-bit）的元素大小：
- `rowbuffer_time` 基于 V 的 2-bit 数据量 = K*N*0.25 + M*N*2 (output FP16)
- `pe_time` 基于反量化后的 FP16 数据量 = total_elements * 2

**结论：** 当前代码低估了 exp5 Score@V 的延迟。Exp6 已经正确建模了 V 反量化开销。

---

## 问题 2: exp3 (2bit GPU) GPU 反量化开销缺失

### 问题描述

在 exp3 中，KV cache 数据为 2-bit 存储，GPU 需要先反量化到 FP16 再进行 GEMV 计算。当前 `AttentionGenExecutionGPU` 函数：

- 用 `precision_byte = 1` 计算 memory_size（减少了 8x 数据量）
- 但没有添加反量化的计算开销

### 为什么 2bit GPU 看起来太快

|  | FP16 GPU (exp1) | 2bit GPU (exp3) | 比值 |
|---|---|---|---|
| KV 数据量 | 34.9 MB | 4.4 MB | 8:1 |
| 反量化 | 不需要 | 需要（~33M ops） | — |
| 报告 Q@K | 97 us | 48 us | 2:1 |

如果 2-bit 只省 load 时间（8x 带宽节省），那么 Q@K 应该接近 97/8 ≈ 12us。但由于 RAMULATOR 建模 DRAM 延迟开销、加上反量化计算，实际应该比 12us 大。

48us 的结果比预期理想带宽节省（~12us）慢 4x，但比 FP16 GPU（97us）快 2x。通常 GPU 做 GEMV 是 memory-bound 的，48us 可能合理，但**缺少了反量化 compute 开销**。

### 修复方案

在 `AttentionGenExecutionGPU` 中添加 2-bit KV 反量化开销：
```cpp
if (input->precision_byte <= 1) {
    // Dequant overhead: 2 ops/element (subtract zero_point + multiply by scale)
    double dequant_elems = k * n + m * k * num_heads / num_kv_heads;
    double dequant_flops = 2.0 * dequant_elems;
    time_ns dequant_time = dequant_flops / compute_peak_flops * 1e9;
    accumul_compute_duration += dequant_time;
}
```

另外需要考虑：反量化后数据膨胀到 FP16，相当于增加了 register file / L1 的数据量。对于 memory-bound 的 GEMV，这可能不是瓶颈（因为 HBM→SM 的带宽才是瓶颈），但反量化本身消耗了 SM 的 ALU 时间。

---

## 问题 3: GPU GEMV 建模需要体现 memory-bandwidth bound 特性

### 问题描述

Attention decode 阶段全是 GEMV（M=1），算术强度低：

```
Q@K:  GEMV(1, 128) × (128, 16528) → (1, 16528)
      FLOPs = 1×128×16528×2 = 4.2M
      Bytes (FP16) = (128 + 2,115,584 + 16,528) × 2 = 4.3 MB
      算术强度 = 4.2M / 4.3M = 0.98 FLOP/B

Score@V: GEMV(1, 16528) × (16528, 128) → (1, 128)
      算术强度类似 (~0.98 FLOP/B)
```

B100 的 FP16 "roofline" 拐点 = 1750 TFLOPS / 8 TB/s = 218.75 FLOP/B。

0.98 << 218.75 → **纯 memory-bound**，GPU 的 FLOPS 利用率极低（<1%）。

### 当前代码处理

当前代码用 `max(compute_duration, memory_duration)`：
- `compute_duration` = 4.2M / 1750 TFLOPS ≈ 0.002 ns（几乎为零）
- `memory_duration` ≈ 4.3MB / 8 TB/s ≈ 0.54 us（per kv_head）
- 16 seqs × 8 kv_heads: memory_duration ≈ 69 us

RAMULATOR 替换了 memory_duration，所以实际数字取决于 DRAM simulator。

### 建议

当前模型在概念上是正确的（memory-bound → memory_duration 主导，compute 可忽略）。主要问题是：
1. 缺少 2-bit dequant compute 开销
2. 可能低估了 memory latency 开销（queueing, bank conflicts）
3. `max(compute, memory)` 假设 compute 和 memory 可以完全重叠，对于小 batch GEMV 可能过于乐观

---

## 问题 4: `experiment_report.md` 数据与 `experiment_comparison.md` 不一致

`experiment_report.md` 是用 256 banks 的分析性结果，而 `experiment_comparison.md` 是用 2048 banks 的实际仿真结果。两者差异巨大：

| Exp | report.md Q@K | comparison.md Q@K | 差距 |
|---|---|---|---|
| exp2 FP16 PIM | 299 us | 66 us | 4.5x |
| exp4 2bit Hybrid | 298 us | 8 us | 37x |
| exp5 2bit All-PIM | 298 us | 8 us | 37x |

`experiment_report.md` 需要更新为实际仿真结果。

---

## 问题 5: GPU qk_duration 赋值错误

`attention_gen_impl.cpp:116` 中 `qk_duration` 被设为 **sum** (compute + memory)，而 `total_duration` 使用 `max`。在 GPU 场景下 compute 可忽略，所以影响不大，但语义上不一致。

---

## 修复优先级

1. **P0**: exp5 Score@V dequant 缺失 (配置修复)
2. **P0**: exp3 GPU 反量化开销缺失 (代码修复)
3. **P1**: `computeGEMVLatency` 接口改进（区分矩阵精度）
4. **P1**: 更新 `experiment_report.md` 为实际仿真数据
5. **P2**: GPU qk_duration sum→max 修复
