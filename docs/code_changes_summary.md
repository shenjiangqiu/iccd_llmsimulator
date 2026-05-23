# Nearbank PIM 代码改动总结

## 1. 新增文件

| 文件 | 用途 |
|------|------|
| `src/dram/nearbank/nearbank_config.h` | PIM 配置结构（bank 数、rowbuffer、PE 参数、功能开关） |
| `src/dram/nearbank/nearbank_pim.h` | `NearbankPIMUnit` 类：GEMV 延迟计算接口 |
| `src/dram/nearbank/nearbank_pim.cpp` | 流水线延迟模型：rowbuffer fill + PE compute，pipeline `max(rb, pe)` |
| `src/dram/nearbank/CMakeLists.txt` | 编译配置 |
| `config_exp1~6_*.yaml` | 6 组试验配置文件 |
| `tools/parse_layer.py` | 解析模拟器输出 → 嵌套树 JSON |
| `tools/report.py` | 生成对比报告 + PIM cycle 验算 |
| `tools/run_all.sh` | 一键运行全部试验 + 生成报告 |
| `eval/test_nearbank.cpp` | 35 个 NearbankPIMUnit 单元测试 |
| `docs/experiment_report.md` | 试验报告 |
| `docs/experiment_comparison.md` | 自动生成的对比表格 |
| `docs/pim_rb_pe_analysis.md` | PIM rb/pe 时间拆解分析 |
| `docs/papers/pim_attention/` | ICML 格式论文 |
| `docs/papers/iccd26/` | IEEE ICCD 格式论文 |

## 2. 修改的现有文件

| 文件 | 改动 |
|------|------|
| `src/module/status.h` | `ExecStatus` + `StatusBoard` 新增字段：`qk_duration`, `softmax_duration`, `score_v_duration`, `kv_quant_duration`, `pim_rb/pe_duration`, `pim_rb/pe_qk`, `pim_rb/pe_sv` |
| `src/module/timeboard.cpp` | `print_util()` 输出 per-stage timing + PIM rb/pe 分解；`set_status()` 做差值得 per-op 值 |
| `src/module/module_graph.cpp` | 从 `ExecStatus` 拷贝 per-stage 字段到 `StatusBoard` |
| `src/hardware/attention_gen_impl.cpp` | `AttentionGenExecutionPIM` 使用 NearbankPIMUnit；per-stage 计时；hybrid 模式 softmax/context 走 GPU；KV 量化开销；PIM rb/pe 分解填充 |
| `src/hardware/attention_mixed_impl.cpp` | `AttentionMixedExecutionPIM` 同样适配 nearbank + hybrid 路由 |
| `src/hardware/hardware_config.h` | `SystemConfig` 加入 `NearbankPIMConfig` |
| `src/model/model_config.h` | 新增 `llama3_8B` 模型配置（dense, 无 MoE） |
| `eval/test.cpp` | 注册 `llama3_8B`；解析 YAML `nearbank_pim` 段 |
| `src/dram/CMakeLists.txt` | 加入 `nearbank` 子目录 |
| `eval/CMakeLists.txt` | 加入 `test_nearbank` 编译目标 |
| `config.yaml` | 加入 `nearbank_pim` 配置段 |

## 3. 核心模型

### 3.1 PIM 延迟模型 (`nearbank_pim.cpp`)

```
per_bank_bytes = total_data / num_banks
rb_fills = ceil(per_bank_bytes / rowbuffer_size)
T_rb = per_bank_bytes / dram_bw       (实际数据量，非满行缓冲)
T_pe = per_bank_elements / pe_elements_per_cycle × pe_cycle
T_asym = reduction_elements / banks / pe_elements_per_cycle × pe_cycle   (非对称量化)
T_dequant = dequant_ops / banks / pe_elements_per_cycle × pe_cycle       (PIM 反量化)

Pipelined: T = first_fill + max(remaining_fills, total_pe)
```

### 3.2 2-bit 紧凑排列处理

- FP16：2 字节/元素，PE 8 元素/cycle（16B ÷ 2B）
- 2-bit：4 元素/字节（紧凑排列），PE 64 元素/cycle（16B × 4），Q 和 K 均为 2-bit
- 数据量：`bytes = elements / 4`

### 3.3 非对称量化分解

```
Q·K ≈ s_Q·s_K · [Σ q̂·k̂ - z_K·Σ q̂ - z_Q·Σ k̂ + d·z_Q·z_K]
```

4 项操作均在 2-bit PE 上完成，reduction 复用 rowbuffer 数据流，无额外 DRAM 读取。

## 4. 6 组试验

| # | 名称 | KV | Q@K | Softmax | Score@V | Asym Quant | PIM Dequant |
|---|------|----|-----|---------|---------|-----------|-------------|
| 1 | FP16 GPU | FP16 | GPU | GPU | GPU | — | — |
| 2 | FP16 PIM | FP16 | PIM | GPU | PIM | — | — |
| 3 | 2bit GPU | 2-bit | GPU | GPU | GPU | ✓ | — |
| 4 | 2bit Hybrid | 2-bit | PIM | GPU | GPU | ✓ | — |
| 5 | 2bit All-PIM | 2-bit | PIM | GPU | PIM | ✓ | — |
| 6 | 2bit Dequant | 2-bit | PIM | GPU | PIM | ✓ | ✓ |

## 5. 最终结果 (per_seq Q@K)

| 试验 | per_seq | vs GPU |
|------|---------|--------|
| FP16 GPU | 6.0us | — |
| FP16 PIM | 4.1us | 1.5× faster |
| 2bit GPU | 3.0us | — |
| **2bit Hybrid** | **0.5us** | **6× faster** |
| 2bit All-PIM | 0.5us | 6× faster |
| 2bit Dequant | 1.0us | dequant 2× slower |

PIM cycle 验算：FP16 PIM ratio 1.00×，2-bit PIM ratio 1.00×。
