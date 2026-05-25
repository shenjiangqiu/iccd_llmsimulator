# GPU GEMV 模型分析与缺陷

## 1. 当前 GPU 模型

`AttentionGenExecutionGPU` 对每个 GEMV：

```cpp
compute_duration = flops / peak_flops      // 假设 100% 峰值算力
memory_duration = bytes / bandwidth         // 假设 100% 内存带宽
total = max(compute, memory)               // 假设完美重叠
```

## 2. 为什么不正确

### 2.1 GEMV 无法达到峰值算力

Decode 阶段 (M=1) 的 GEMV 算术强度极低：

| 操作 | FLOPs | Bytes (FP16) | AI (FLOP/B) |
|------|-------|-------------|-------------|
| Q@K | 4.2M | 4.3M | 0.98 |
| Score@V | 4.2M | 4.3M | 0.98 |

B100 roofline 拐点 = 1750 TFLOPS / 8 TB/s = 219 FLOP/B。

0.98 << 219 → **纯 memory-bound**。

GPU GEMV 效率受限于：
- 小 M → 低 warp occupancy → 无法隐藏 memory latency
- 每 SM 的 L1 bandwidth 有限
- 计算单元大部分时间空闲等待数据

**实际效率**：通常 5-30% 峰值算力对 M=1 的 GEMV。模型用 100% 峰值高估了 compute 能力，但幸好 compute ≈ 0（memory-bound），所以 `max()` 退化到 memory_duration，影响不大。

### 2.2 precision_byte=1 的歧义

`precision_byte=1` 作为 2-bit 存储的代理值：

```
实际 2-bit packed: 0.25 B/elem
precision_byte=1:  1.00 B/elem  → 高估 4x

实际 FP16:         2.00 B/elem
precision_byte=1:  1.00 B/elem  → 低估 2x
```

这个值既不代表存储格式，也不代表计算精度。它的来源是 `tensor->getSize() = product(shape) × precision_byte`，影响整个 RAMULATOR 通路。

GPU 读 K 矩阵的真实内存流量：
- HBM → L2 (2-bit packed): K_elems × 0.25 B
- L2 → L1 (FP16, after decompression): K_elems × 2 B
- 最紧缺的路径是 HBM → L2（带宽瓶颈）

当前用 `precision_byte=1`：K_elems × 1 B → 比实际 2-bit 大 4x，比 FP16 小 2x。恰好是一个 _fudge factor_。

### 2.3 2-bit 必须 dequantize 再计算

2-bit KV 数据在 GPU 上的计算链路：

```
HBM ──[2-bit packed]──→ L2 ──[decompress]──→ L1 ──[dequant]──→ Reg (FP16) ──[GEMV]──→ Reg
  0.25B/elem             2B/elem               2 ops/elem         FP16 MAD
```

**缺失**：
1. 反量化计算开销（2 ops/elem）
2. 反量化后数据膨胀（寄存器占用 2B/elem）
3. 反量化消耗 SM ALU，占用 GEMV 执行单元

### 2.4 先 load 再计算 vs 完美重叠

`max(compute, memory)` 假设计算和数据搬运可以完美重叠。但：
- 小 GEMV 的操作粒度有限
- 必须先 load 完一批数据才能开始计算该批数据
- 大 seq_len 下虽然可以做 double-buffering，但效率有限
- 实际 = memory + 少量 compute 尾巴

## 3. 修复方案

### 3.1 已修：GPU 反量化开销

```cpp
// attention_gen_impl.cpp: 在 RAMULATOR 前
if (input->precision_byte <= 1) {
    double dequant_elements = accumul_len × k;
    double dequant_flops = 2.0 × dequant_elements;
    time_ns dequant_time = dequant_flops / effective_flops;
    accumul_compute_duration += dequant_time;
}
```

### 3.2 GPU GEMV 有效算力

不应使用 `peak_flops`，应使用 memory-bandwidth 限定的有效算力：

```
effective_flops = min(peak_flops, memory_bandwidth × flops / bytes)
```

由于 GEMV 是 memory-bound，`effective_flops ≈ memory_bandwidth × AI`，远小于 peak。
等效于 `time = max(compute_eff, memory) ≈ memory`。

对 M=1 decode 场景，这个修正不影响结果（compute 已经 ≈ 0）。
对 M>1（prefill）场景才有实际影响。

### 3.3 建议的完整模型

```
GPU GEMV 延迟:
  read_time  = bytes_2bit / hbm_bw          (HBM read, 2-bit packed)
  deq_time   = 2 × elems / sm_ops_per_sec   (反量化, SM ALU)
  gemv_time  = flops_fp16 / gemv_eff_flops  (GEMV, FP16, 低效)
  
  total = read_time + max(0, gemv_time - overlapped_read_time)
        ≈ read_time   (因为 read_time >> gemv_time)
```

当前模型 `max(memory, compute)` 的含义：
- memory: 数据搬运时间（RAMULATOR 返回值）
- compute: 计算时间（几乎为 0）
- 结果 ≈ memory → 本质正确

**核心**：只要 memory_duration 正确（含 RAMULATOR 的 DRAM 建模），GPU GEMV 延迟就大致正确。compute 的 0% vs 50% 效率差异对结果影响可忽略。

### 3.4 precision_byte 的处理建议

选项 A（激进）：将 2-bit 实验的 `precision_byte` 改为 2，在代码中显式处理 2-bit 数据量。
选项 B（保守）：保持 `precision_byte=1`，在注释中标明是 fudge factor。
选项 C（折中）：在 GPU 路径显式使用 0.25B/elem 的存储因子，与 `precision_byte` 脱钩。

**推荐 C**：在 GPU 函数内部计算真实 bytes，不依赖 `precision_byte` 的值。
