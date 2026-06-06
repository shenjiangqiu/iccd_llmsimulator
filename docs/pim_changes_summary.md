# PIM 改动总结 (GQA Rowbuffer Bug + 4-bit KV)

## 1. GQA Rowbuffer Bug 修复

### Bug

`attention_gen_impl.cpp` 中 PIM 的 `rowbuffer_time_ns` 被错误地乘了 `attention_group_size`。

```cpp
// ✗ 修复前：K/V 被加载 group_size 次
pim_rb_qk += gemv_result.rowbuffer_time_ns * attention_group_size;

// ✓ 修复后：K/V 每个 kv_head 只加载一次
pim_rb_qk += gemv_result.rowbuffer_time_ns;
```

### 原理

GQA 中同一 kv_head 的所有 Q head 共享 K/V 矩阵，K/V 只需从 DRAM 加载一次到 rowbuffer。

```
kv_head i: K_i → rowbuffer (1 次)
  for Q in group:
    Q × K_i → GEMV (group_size 次)
```

### 影响

| 模型 | group_size | rb 修复前被高估 | 影响最大的配置 |
|------|-----------|--------------|--------------|
| llama3 | 4 | 4× | All-PIM: 1.4× → **5.0×** |
| qwen3 | 4 | 4× | All-PIM: 1.4× → **5.0×** |
| glm4 | 8 | 8× | All-PIM: 0.7× → 0.7× (FP16 PE 瓶颈) |
| opt | 1 | 1× (无影响) | 不变 |

### 修改文件

- `src/hardware/attention_gen_impl.cpp`：Scoring 和 Context 阶段的 rb 计数不再乘 group_size
- `src/hardware/attention_mixed_impl.cpp`：同步修复
- 分析文档：`docs/gqa_fix_analysis.md`

---

## 2. 4-bit KV Cache 支持

### 新增配置字段

`nearbank_config.h`:
```cpp
int kv_cache_bits = 2;  // 2: 2-bit, 4: 4-bit
```

YAML:
```yaml
nearbank_pim:
  kv_cache_bits: 4
```

### PE 模型泛化

`nearbank_pim.cpp` 中将硬编码的 2-bit 模型替换为通用 bit-width 模型：

```cpp
int bits = config_.kv_cache_bits;
bytes_per_element = bits / 8.0;               // 2-bit→0.25, 4-bit→0.5
pe_elements_per_cycle = pe_width * (8.0 / bits); // 2-bit→128, 4-bit→64 (32B PE)
```

### 4-bit vs 2-bit 对比 (llama3_8B, Hybrid)

| Metric | 2-bit | 4-bit | 比值 |
|--------|-------|-------|------|
| 存储 (B/elem) | 0.25 | 0.50 | 2× |
| PE 吞吐 (elem/cycle, 32B PE) | 128 | 64 | 0.5× |
| Score (us) | 4.16 | 8.31 | **2×** |
| qk_rb (us) | 1.04 | 2.08 | 2× |
| qk_pe (us) | 2.08 | 4.16 | 2× |

### 新增实验配置

- `config_exp4_4bit_hybrid.yaml`
- `config_exp5_4bit_allpim.yaml`
- `config_exp3_4bit_gpu.yaml`

### 修改文件

- `src/dram/nearbank/nearbank_config.h`：新增 `kv_cache_bits`
- `src/dram/nearbank/nearbank_pim.cpp`：泛化为可变 bit-width
- `eval/test.cpp`：YAML 解析 `kv_cache_bits`
- 新增 4-bit 配置文件
