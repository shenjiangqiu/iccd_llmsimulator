# 实验设计与预期分析

## 1. 实验环境

| 参数 | 值 |
|------|-----|
| 模型 | Llama-3-8B (dense, 32层, hidden=4096, heads=32, kv_heads=8, head_dim=128) |
| GPU | B100 (1750 TFLOPS FP16, 8 TB/s HBM带宽) |
| Prefill 长度 | 16400 tokens |
| Decode 长度 | 128 steps |
| Batch size | 64 (16 seqs/DP group) |
| PIM banks | 2048 |
| Row buffer | 2048 B |
| PE FP16 | 16 B/cycle @ 0.5ns (2 GHz) |
| PE lowbit | 32 B/cycle @ 0.5ns (2 GHz) |
| DRAM→RB 带宽 | 32 GB/s per bank |

---

## 2. 各实验设计

### 2.1 Exp1: FP16 GPU (Baseline)

**设想**：标准 GPU 注意力计算，KV cache 存 FP16。

**数据流**：
1. GPU 从 HBM 读取 Q（FP16，1×128×32 = 4096 元素）
2. GPU 从 HBM 读取 K（FP16，16528×128×8 = 16.9M 元素）
3. GPU SM 计算 `Q @ K^T`（GEMV，memory-bound）
4. GPU SM 计算 Softmax
5. GPU 从 HBM 读取 V（FP16，16528×128×8 = 16.9M 元素）
6. GPU SM 计算 `Score @ V`（GEMV，memory-bound）

**关键参数**：
- Q@K 算术强度 = 4.2M FLOPs / 4.3 MB ≈ 0.98 FLOP/B
- Memory-bound：GPU 利用率 << peak
- 总 GEMV 次：16 seqs × 8 kv_heads × 4 attention_group = 512 次/step

**预期**：Q@K 时间由 HBM 带宽主导。FP16 数据量 ~34.9 MB per step（16 seqs，含 Q+K+result）。

---

### 2.2 Exp2: FP16 PIM

**设想**：Q@K 和 Score@V 都在 PIM 中完成，Softmax 在 GPU。

**数据流**：
1. Q 从 GPU 发送到 PIM（仅 512 元素×2B = 1KB per attention_group，可忽略）
2. PIM rowbuffer 从 DRAM 加载 K 矩阵（FP16，2MB per kv_head）
3. PIM PE 计算 `Q @ K^T`（rowbuffer fill 和 PE compute 流水线化）
4. 结果（scores）送回 GPU 做 Softmax
5. GPU 将 scores 送回 PIM
6. PIM rowbuffer 加载 V 矩阵（FP16，2MB per kv_head）
7. PIM PE 计算 `Score @ V`

**关键参数**：
- per-bank bytes = total_bytes / 2048 banks
- FP16: total_bytes = (128+2115584+16528)×2 = 4,264,480 B
  - per_bank = 2082 B, rb_fills = 2
  - rb_time = 128 ns (2 fills × 64ns)
  - pe_time = 2082/16×0.5 = 65.1 ns
  - latency = 64 + max(64, 65.1) = 129.1 ns
- per_direction = 129.1 × 4 × 8 × 16 = 66 us

**预期 PIM vs GPU**：PIM 的 PE 吞吐远低于 GPU（2048 bank × 16B/cycle × 2GHz = 65.5 TB/s per bank 总计算吞吐，但分散到各 bank）。PIM 优势在于**零数据搬运**（K、V 不需要出 DRAM）。

---

### 2.3 Exp3: 2-bit GPU

**设想**：KV cache 使用 2-bit 量化（4 元素/字节紧凑排列），GPU 做全部计算。

**数据流**：
1. GPU 从 HBM 读取 Q（FP16）
2. GPU 从 HBM 读取 K（2-bit，528KB per kv_head vs FP16 的 4.2MB → 8x 带宽节省）
3. **GPU 反量化**：2-bit K → FP16 K（subtract zero_point × scale）→ ~33.8M ops
4. GPU SM 计算 `Q @ K^T`
5. 同理对 V：读取 2-bit → 反量化 → 计算 Score@V

**关键参数**：
- K 数据量：16528×128×0.25 = 528,896 B per kv_head（2-bit 紧凑排列）
- 反量化 FLOPs：2×16528×128 = 4.2M ops per kv_head
- 数据量减少 8x，但添加了反量化开销

**⚠️ 当前代码问题**：`AttentionGenExecutionGPU` 未建模反量化开销。
- 当前：compute_duration ≈ 0（仅 GEMV FLOPS），memory_duration 基于 2-bit 数据量
- 缺少：dequant compute_time ≈ 4.2M ops / 1750 TFLOPS ≈ 2.4 ps（可忽略）
- **关键缺失**：反量化后数据膨胀到 FP16 占用 register file，实际 GEMV 效率更低

**预期**：2bit GPU 比 FP16 GPU 快约 8x（纯带宽收益），减去反量化开销后约 6~7x。

---

### 2.4 Exp4: 2-bit Hybrid

**设想**：Q@K 在 PIM（利用 2-bit 数据量小的优势），Score@V 在 GPU（避免 PIM PE 瓶颈）。

**数据流**：
1. PIM 用 2-bit K 做 Q@K（asymmetric quant 分解在 2-bit 空间）
2. PIM PE 额外计算 asymmetric quant 的 reduction 项（Σq̂, Σk̂）
3. Scores（FP16）送回 GPU
4. GPU 做 Softmax
5. GPU 从 HBM 读取 V（2-bit）→ 反量化 → 计算 Score@V

**关键参数**：
- PIM Q@K（2-bit + asymmetric quant）：
  - total_bytes = (128+2115584+16528)×0.25 = 533,060 B
  - per_bank = 260 B, rb_fills = 1
  - rb_time = 260/32e9×1e9 = 8.1 ns
  - pe_time (main GEMV) = 1041/64×0.5 = 8.1 ns
  - reduction_time = 1033/64×0.5 = 8.1 ns
  - latency = max(8.1, 8.1+8.1) = 16.2 ns
  - per_direction = 16.2 × 4 × 8 × 16 = 8.3 us

- GPU Score@V：与 Exp3 的 Score@V 相同（数据量小 + 反量化）
  - V 数据量: 528KB per kv_head（2-bit）× 8 kv_heads × 16 seqs = 67.5 MB
  - memory_time = 67.5MB / 8TB/s = 8.4 us（ideal bandwidth）
  - 加上反量化时间（GPU 很快）≈ 合计 ~10 us

- 但 report 显示 S@V=34us（可能是 RAMULATOR 建模的 DRAM 开销）

**预期**：Q@K 在 PIM 极快（8us），Score@V 在 GPU 中等（~34us），总体比全 PIM 快很多。

---

### 2.5 Exp5: 2-bit All-PIM

**设想**：Q@K 和 Score@V 全在 PIM。2-bit KV，asymmetric quant。

**数据流**：
1. PIM Q@K：与 Exp4 相同（asymmetric quant 在 2-bit 空间）
2. Scores（FP16）留在 PIM（或通过共享 buffer）
3. Softmax 在 GPU（scores 送回 GPU → softmax → 送回 PIM）
4. **PIM Score@V**：Scores（FP16）× V（2-bit）

**⚠️ 问题：Score@V 需要 V 反量化**

Scores 是 FP16（Q@K 输出，16528 元素 × 2B = 33KB），V 是 2-bit（528KB）。
Score@V = S(f16) × V(2bit) → 这要求 V 先反量化到 FP16。

- 如果不反量化：直接用 2-bit V 乘 FP16 分数→ 数值无意义
- 正确做法：V 反量化后再做 FP16 GEMV（即 Exp6 的方案）

**当前代码问题**：Exp5 的 `computeGEMVLatency` 对 Context 阶段传入 `precision_byte=1`，将 S 和 V 都当 2-bit，低估了 PE 时间。

正确模型：
- Rowbuffer 加载：V(2bit) = 528KB（仍用 2-bit 数据量）
- PE dequantize：2 ops/element × 2M elements = 4M ops → ~8.1 ns PE 时间
- PE GEMV：dequantized V(FP16) × S(FP16) → 使用 FP16 PE 吞吐（8 elements/cycle）
  - elements_per_bank = 2132240/2048 = 1041
  - pe_time = 1041/8×0.5 = 65.1 ns
- 总 PE 时间 = 65.1 + 8.1 = 73.2 ns（远大于当前建模的 ~16 ns）

**预期**：Score@V 应该明显慢于 Q@K（因为需要 dequant + FP16 GEMV）。Exp5 目前低估了 Score@V 时间。

---

### 2.6 Exp6: 2-bit PIM Dequant

**设想**：与 Exp5 相同，但显式启用 V 反量化（`enable_pim_dequant: true`）。

**数据流**：同 Exp5，但 Score@V 阶段：
1. PIM rowbuffer 加载 V（2-bit，528KB）
2. PIM PE 反量化：V(int2) → V(FP16)
3. PIM PE 计算 FP16 GEMV：S(FP16) × V(FP16)

**关键参数**：
- Rowbuffer：528KB → per_bank = 258B → 1 fill → 8.1 ns
- PE dequant：2×2115584/2048 = 2066 ops/bank → 2066/64×0.5 = 16.1 ns
- PE GEMV（FP16）：1041 elements → 1041/8×0.5 = 65.1 ns
- 总 PE = 16.1 + 65.1 = 81.2 ns
- latency = max(8.1, 81.2) = 81.2 ns
- per_direction = 81.2 × 32 × 16 = 41.6 us

**对比**：
- Exp5（无 dequant，错误模型）：Score@V ≈ 8us（与 Q@K 相同）
- Exp6（有 dequant，正确模型）：Score@V ≈ 17us（约 2x Exp5）

**实际上 FP16 Score@V 是否一定要先 dequant 再算？**

可以考虑另一种架构：PIM PE 直接支持 "混合精度 GEMV"，即 2-bit 权重 × FP16 激活，省去显式反量化步骤。但当前 nearbank PE 模型不支持此功能，两种精度分开建模。

---

## 3. GPU GEMV 建模原则

### 为什么 GPU GEMV 不能简单用 peak FLOPS

Attention decode 阶段所有操作都是 GEMV（M=1），算术强度极低：

| 操作 | M | K | N | FLOPs | Bytes (FP16) | AI (FLOP/B) |
|------|---|---|---|--------|-------------|-------------|
| Q@K | 1 | 128 | 16528 | 4.2M | 4.3M | 0.98 |
| Score@V | 1 | 16528 | 128 | 4.2M | 4.3M | 0.98 |

B100 roofline 拐点 = 1750 TFLOPS / 8 TB/s = 219 FLOP/B。0.98 << 219 → **纯 memory-bound**。

**GPU 建模要点**：
1. **先 load 再计算**：K/V 必须先从 HBM 加载到 SM 的 register/L1
2. **内存带宽决定上限**：`T ≥ bytes / bandwidth`（不可逾越）
3. **GEMV 效率远低于 peak**：小 M 的 GEMV 达不到 GPU peak FLOPs
4. **2-bit 场景**：load 节省了带宽，但反量化消耗 ALU，且反量化后数据膨胀

### 当前 GPU 模型

```
compute_duration = flops / peak_flops  (≈ 0，对于 memory-bound GEMV)
memory_duration = bytes / bandwidth     (主导项)
total = max(compute, memory)           (= memory_duration)
```

对于 memory-bound 操作，`max()` 近似于只用 memory_duration，这基本正确。但缺少：
- 反量化 compute 开销（对 2-bit GPU 很重要）
- Memory latency overhead（RAMULATOR 部分覆盖了这点）

---

## 4. PIM 流水线模型验证

PIM 延迟 = `first_rb_fill + max(remaining_fills, total_pe_time)`

### FP16 PIM (Exp2)

```
M=1, K=128, N=16528, elem=2B, 2048 banks
total_bytes = 4,264,480 B
per_bank = 2,082 B, rb_fills = 2
T_fill = 2048/32e9 = 64 ns
rb_time = 2×64 = 128 ns
pe_time = 2082/16×0.5 = 65.1 ns
latency = 64 + max(64, 65.1) = 129.1 ns
per_direction = 129.1 × 4 × 8 × 16 = 66,100 ns = 66.1 us ✓
```

### 2-bit PIM Q@K (Exp4/Exp5, with asymmetric quant)

```
elem=0.25B, 2048 banks
total_bytes = 533,060 B
per_bank = 260.3 B, rb_fills = 1
pe_time_main = 1041/64×0.5 = 8.13 ns
reduction = (1+16528)×128 = 2,115,712
pe_time_red = 2,115,712/2048/64×0.5 = 8.07 ns
total_pe = 8.13 + 8.07 = 16.2 ns
latency = max(8.13, 16.2) = 16.2 ns
per_direction = 16.2 × 4 × 8 × 16 = 8,294 ns = 8.3 us ✓
```

### 2-bit PIM Score@V with Dequant (Exp6)

```
V dequant + FP16 GEMV:
  Rowbuffer: 528KB → per_bank 258B → 1 fill → 8.1 ns
  PE dequant: 2×2115584 = 4.2M ops → per_bank 2066 → 2066/64×0.5 = 16.1 ns
  PE GEMV FP16: 1041 elements → 1041/8×0.5 = 65.1 ns
  Total PE = 16.1 + 65.1 = 81.2 ns
  latency = max(8.1, 81.2) = 81.2 ns
  per_direction = 81.2 × 32 × 16 = 41.6 us
```

**Exp6 报告 Score@V ≈ 17us？** → 这说明 RAMULATOR / 实际代码可能有不同的 bank 分配或更多流水线重叠。需要验证。

---

## 5. 数据一致性问题

`experiment_report.md` 使用 **256 banks** 的分析性结果（较早），`experiment_comparison.md` 使用 **2048 banks** 的仿真结果（新）。

| Exp | report.md Q@K | comparison.md Q@K | 说明 |
|-----|-------------|-------------------|------|
| exp2 | 299 us | 66 us | 256 vs 2048 banks 差异 |
| exp4 | 298 us | 8 us | 同上 + comparison 可能是只算 scoring 不算 context |
| exp5 | 298 us | 8 us | 同上 |

**`experiment_report.md` 需要更新为 2048 banks 的正确数据。**

---

## 6. 代码缺陷汇总

| # | 位置 | 问题 | 严重度 |
|---|------|------|--------|
| 1 | `attention_gen_impl.cpp:449-451` | exp5 Score@V 用 `precision_byte=1` 建模，未考虑 S 是 FP16、V 需反量化 | **高** |
| 2 | `attention_gen_impl.cpp:50-121` | exp3 GPU 路径未建模 2-bit KV 反量化开销 | **高** |
| 3 | `attention_gen_impl.cpp:116` | GPU qk_duration 用 sum 而非 max，与 total_duration 不一致 | 低 |
| 4 | `nearbank_pim.cpp:167-187` | PIM dequant 模型只作用于单精度场景，无法区分 Score@V 的 S(f16)×V(2bit) | **中** |
| 5 | `experiment_report.md` | 使用 256 banks 旧数据，需更新 | **中** |
