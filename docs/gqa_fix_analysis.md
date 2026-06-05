# GQA Rowbuffer Bug Fix — 分析与验证

## 1. Bug 描述

在 `attention_gen_impl.cpp` 中，PIM 的 Score/Context 阶段对 `rowbuffer_time_ns` 错误地乘了 `attention_group_size`：

```cpp
// ✗ 错误（修复前）
accumul_memory_duration += gemv_result.rowbuffer_time_ns * attention_group_size;
pim_rb_qk += gemv_result.rowbuffer_time_ns * attention_group_size;

// ✓ 正确（修复后）
accumul_memory_duration += gemv_result.rowbuffer_time_ns;  // K/V 只读一次
pim_rb_qk += gemv_result.rowbuffer_time_ns;
```

## 2. 原理

GQA 中一个 KV head 对应 `attention_group_size` 个 Q head。K/V 矩阵对所有 Q head **相同**，应只从 DRAM 加载一次到 rowbuffer，然后广播不同 Q 做多次 GEMV：

```
kv_head i:
  K_i → rowbuffer (DRAM read, 1 次)
  for j = 1..G:
    Q_{i*G+j} (global bus) × K_i → score_{i*G+j} (GEMV, G 次)
```

```
时间轴 (G=4):
  Rowbuffer: |=== K load ===|                         ← 1 次
  PE:        |Q1 × K|Q2 × K|Q3 × K|Q4 × K|           ← G 次
  Total:       rb + G × pe
```

修复前错误假设 K 被加载 G 次：

```
  Rowbuffer: |==K1==|==K2==|==K3==|==K4==|          ← 错误，G 次
  PE:        |Q1×K1|Q2×K2|Q3×K3|Q4×K4|
  Total:       G × max(rb, pe)
```

## 3. 影响

修复后 rowbuffer 时间（rb）减少 `attention_group_size` 倍。

| 模型 | group_size | rb 变化 | 影响最大的配置 |
|------|-----------|---------|--------------|
| glm4 | 8 | rb ÷ 8 | All-PIM（rb 占比最大） |
| llama3/qwen3 | 4 | rb ÷ 4 | All-PIM |
| opt | 1 | 无影响 | 无 |

## 4. 修复后数据

### Speedup vs FP16 GPU (Gen latency)

| Config | glm4(8:1) | llama3(4:1) | opt(1:1) | qwen3(4:1) |
|--------|-----------|-------------|----------|-------------|
| FP16 GPU | 1.0× | 1.0× | 1.0× | 1.0× |
| FP16 PIM | 0.7× | 1.5× | 11.5× | 1.5× |
| 2-bit GPU | 2.0× | 2.0× | 2.0× | 2.0× |
| **2-bit Hyb.** | **4.6×** | **5.0×** | **5.5×** | **5.0×** |
| 2-bit All | 0.7× | 5.0× | 5.4× | 1.4× |
| 2-bit Deq. | 1.3× | 2.6× | 10.3× | 2.6× |
| All (16B/32B) | 1.3× | 9.1× | 10.3× | 2.6× |

### 关键发现

1. **All-PIM 修复后显著提升**：llama3 从 1.4× → 5.0×（修复前 rb 被高估 4×，掩盖了 All-PIM 的真实性能）

2. **glm4 (group_size=8) 的 All-PIM 仍然慢（0.7×）**：因为 Aggregate 阶段走 PIM 的 FP16 GEMV，瓶颈在 PE 算力而非 rowbuffer。即使 rb 减少 8×，FP16 GEMV 的 65ns/GEMV 仍是瓶颈。

3. **Hybrid 不受影响**：Aggregate 在 GPU，修复只影响 rb 上报值，不影响总延迟。

4. **opt (group_size=1) 完全不变**：无 GQA，`attention_group_size = 1`，修复前后一致。

## 5. 结论

GQA 模型的 PIM 实现应区分**一次性 K/V load** 与**多次 Q head 计算**。修复后 All-PIM 的 rb 开销正确反映了 GQA 的共享特性，对于 group_size ≥ 4 的模型影响显著。
