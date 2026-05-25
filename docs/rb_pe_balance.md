# 为什么 2-bit 下 RB 和 PE 时间相等

## 1. 核心等式

PIM 的 Rowbuffer 搬运和 PE 计算是两个独立硬件单元，可以流水线化。当两者吞吐匹配时，处理同批数据的时间必然相等。

```
rb_time = per_bank_bytes / dram_bw
pe_time = per_bank_elems / pe_elem_per_cycle × pe_cycle
```

代入 `per_bank_bytes = per_bank_elems × bytes_per_element`，两边消去 `per_bank_elems`：

```
rb_time = pe_time
⟺ bytes_per_element / dram_bw = pe_cycle / pe_elem_per_cycle
⟺ dram_bw / bytes_per_element = pe_elem_per_cycle / pe_cycle
```

| 左边 | 右边 |
|------|------|
| DRAM 的元素吞吐 (elem/s) | PE 的元素吞吐 (elem/s) |

---

## 2. FP16 验算

```
dram_bw / bytes_per_elem = 32e9 / 2       = 16e9 elem/s
pe_elem_per_cycle / pe_cycle = 8 / 0.5e-9 = 16e9 elem/s

16e9 = 16e9  ✓
```

代入实际数据（M=1, K=128, N=16528, 2048 banks）：

```
per_bank_data  = (1×128 + 128×16528 + 1×16528) × 2 / 2048
               = 4,264,480 / 2048 = 2082 B

per_bank_elems = 2,132,240 / 2048 = 1041

rb_time = 2082 / 32e9 × 1e9  = 65.1 ns
pe_time = 1041 / 8 × 0.5     = 65.1 ns        ← 精确相等
```

---

## 3. 2-bit 验算

2-bit 紧凑排列：4 元素/B，PE 64 元素/cycle（16B × 4 elem/B）。

```
dram_bw / bytes_per_elem = 32e9 / 0.25    = 128e9 elem/s
pe_elem_per_cycle / pe_cycle = 64 / 0.5e-9 = 128e9 elem/s

128e9 = 128e9  ✓
```

代入实际数据：

```
per_bank_data  = 2,132,240 × 0.25 / 2048
               = 533,060 / 2048 = 260.3 B

per_bank_elems = 2,132,240 / 2048 = 1041

rb_time = 260.3 / 32e9 × 1e9 = 8.13 ns
pe_time = 1041 / 64 × 0.5    = 8.13 ns        ← 精确相等
```

---

## 4. 物理意义

所有精度的平衡来自同一个硬件事实：

```
PE 宽度        = 16 B/cycle
PE 频率        = 2 GHz
PE 字节吞吐    = 16 B / 0.5 ns = 32 GB/s

DRAM→RB 带宽   = 32 GB/s per bank

32 GB/s = 32 GB/s  →  PE 吞吐 = 内存带宽
```

PE 刚好在一个 cycle 里消费完 rowbuffer 一次 fill 送过来的那批数据（16B）。

不同精度的差异仅在于**元素粒度**：

| 精度 | bytes/elem | DRAM elem/s | PE elem/cycle | PE elem/s |
|------|-----------|-------------|--------------|-----------|
| FP16 | 2 | 16e9 | 8 (=16B÷2B) | 16e9 |
| 2-bit | 0.25 | 128e9 | 64 (=16B×4) | 128e9 |

每行左右两列始终相等（16e9=16e9, 128e9=128e9），消去 `per_bank_elems` 后 rb_time = pe_time。

---

## 5. 这是配置巧合而非必须

上述相等是因为配置中 `dram_to_rb_bw_per_bank = 32 GB/s` 且 `pe_width_bytes = 16 B / (0.5 ns)`。如果改变任一参数：

```
将带宽翻倍:  dram_bw = 64 GB/s 
  → FP16: rb=32.5ns, pe=65.1ns → pe 成为瓶颈

将 PE 翻倍:  pe_width = 32 B
  → FP16: rb=65.1ns, pe=32.5ns → rb 成为瓶颈
```

当前配置恰好让两者平衡，意味着流水线效率最优——rb 和 pe 都不会闲置等待。
