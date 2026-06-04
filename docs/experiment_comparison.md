# Full Comparison Report

## Model: glm4_9B (kv_heads=4)

### Per-Step Attention Latency (us, 16 seqs/DP)

| Config | Score | Aggregate | Gen | Softmax | KV Quant | pim_rb | pim_pe | qk_rb | qk_pe | sv_rb | sv_pe | per_seq |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FP16 GPU | 49 | 48 | 97 | - | - | - | - | - | - | - | - | 3.04 |
| FP16 PIM | 66 | 66 | 132 | 0 | 0 | 66 | 66 | 33 | 33 | 33 | 33 | 4.13 |
| 2-bit GPU | 24 | 25 | 49 | - | - | - | - | - | - | - | - | 1.50 |
| 2-bit Hyb. | 4 | 17 | 21 | 0 | 0 | 4 | 2 | 4 | 2 | - | - | 0.26 |
| 2-bit All | 4 | 71 | 75 | 0 | 0 | 8 | 39 | 4 | 2 | 4 | 37 | 0.26 |
| 2-bit Deq. | 37 | 37 | 75 | 0 | 0 | 8 | 75 | 4 | 37 | 4 | 37 | 2.34 |

### Throughput (64 seqs, 4x GPU, TP=1)

| Config | Layer(us) | Decode(tok/s) | Decode Step(us) |
|---|---|---|---|
| FP16 GPU | - | 11,027 | - |
| FP16 PIM | - | 8,079 | - |
| 2-bit GPU | - | 21,919 | - |
| 2-bit Hyb. | - | 50,680 | - |
| 2-bit All | - | 14,265 | - |
| 2-bit Deq. | - | 14,271 | - |


## Model: llama3_8B (kv_heads=8)

### Per-Step Attention Latency (us, 16 seqs/DP)

| Config | Score | Aggregate | Gen | Softmax | KV Quant | pim_rb | pim_pe | qk_rb | qk_pe | sv_rb | sv_pe | per_seq |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FP16 GPU | 95 | 96 | 192 | - | - | - | - | - | - | - | - | 5.96 |
| FP16 PIM | 66 | 66 | 132 | 0 | 0 | 66 | 66 | 33 | 33 | 33 | 33 | 4.13 |
| 2-bit GPU | 48 | 48 | 95 | - | - | - | - | - | - | - | - | 2.98 |
| 2-bit Hyb. | 4 | 34 | 38 | 0 | 0 | 4 | 2 | 4 | 2 | - | - | 0.26 |
| 2-bit All | 4 | 71 | 75 | 0 | 0 | 8 | 39 | 4 | 2 | 4 | 37 | 0.26 |
| 2-bit Deq. | 37 | 37 | 75 | 0 | 0 | 8 | 75 | 4 | 37 | 4 | 37 | 2.34 |

### Throughput (64 seqs, 4x GPU, TP=1)

| Config | Layer(us) | Decode(tok/s) | Decode Step(us) |
|---|---|---|---|
| FP16 GPU | 271 | 7,370 | 8,684 |
| FP16 PIM | 212 | 9,428 | 6,788 |
| 2-bit GPU | 136 | 14,668 | 4,363 |
| 2-bit Hyb. | 79 | 25,262 | 2,533 |
| 2-bit All | 116 | 17,240 | 3,712 |
| 2-bit Deq. | 116 | 17,244 | 3,711 |


## Model: opt_6_7B (kv_heads=32)

### Per-Step Attention Latency (us, 16 seqs/DP)

| Config | Score | Aggregate | Gen | Softmax | KV Quant | pim_rb | pim_pe | qk_rb | qk_pe | sv_rb | sv_pe | per_seq |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FP16 GPU | 280 | 280 | 560 | - | - | - | - | - | - | - | - | 17.51 |
| FP16 PIM | 24 | 24 | 49 | 0 | 0 | 49 | 49 | 24 | 24 | 24 | 24 | 1.52 |
| 2-bit GPU | 140 | 140 | 280 | - | - | - | - | - | - | - | - | 8.76 |
| 2-bit Hyb. | 3 | 99 | 102 | 0 | 0 | 3 | 2 | 3 | 2 | - | - | 0.19 |
| 2-bit All | 3 | 52 | 55 | 0 | 0 | 6 | 29 | 3 | 2 | 3 | 27 | 0.19 |
| 2-bit Deq. | 27 | 27 | 55 | 0 | 0 | 6 | 55 | 3 | 27 | 3 | 27 | 1.71 |

### Throughput (64 seqs, 4x GPU, TP=1)

| Config | Layer(us) | Decode(tok/s) | Decode Step(us) |
|---|---|---|---|
| FP16 GPU | - | 2,379 | - |
| FP16 PIM | - | 27,451 | - |
| 2-bit GPU | - | 4,757 | - |
| 2-bit Hyb. | - | 13,106 | - |
| 2-bit All | - | 24,412 | - |
| 2-bit Deq. | - | 24,422 | - |


## Model: qwen3_8B (kv_heads=8)

### Per-Step Attention Latency (us, 16 seqs/DP)

| Config | Score | Aggregate | Gen | Softmax | KV Quant | pim_rb | pim_pe | qk_rb | qk_pe | sv_rb | sv_pe | per_seq |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FP16 GPU | 95 | 96 | 192 | - | - | - | - | - | - | - | - | 5.96 |
| FP16 PIM | 66 | 66 | 132 | 0 | 0 | 66 | 66 | 33 | 33 | 33 | 33 | 4.13 |
| 2-bit GPU | 48 | 48 | 95 | - | - | - | - | - | - | - | - | 2.98 |
| 2-bit Hyb. | 4 | 34 | 38 | 0 | 0 | 4 | 2 | 4 | 2 | - | - | 0.26 |
| 2-bit All | 4 | 71 | 75 | 0 | 0 | 8 | 39 | 4 | 2 | 4 | 37 | 0.26 |
| 2-bit Deq. | 37 | 37 | 75 | 0 | 0 | 8 | 75 | 4 | 37 | 4 | 37 | 2.34 |

### Throughput (64 seqs, 4x GPU, TP=1)

| Config | Layer(us) | Decode(tok/s) | Decode Step(us) |
|---|---|---|---|
| FP16 GPU | - | 6,186 | - |
| FP16 PIM | - | 8,976 | - |
| 2-bit GPU | - | 12,412 | - |
| 2-bit Hyb. | - | 31,240 | - |
| 2-bit All | - | 15,850 | - |
| 2-bit Deq. | - | 15,856 | - |


## Cross-Model Comparison (Hybrid)

| Model | kv_heads | Score | Aggregate | Gen | Decode tok/s |
|---|---|---|---|---|---|
| glm4_9B | 4 | 4 | 17 | 21 | 50,680 |
| llama3_8B | 8 | 4 | 34 | 38 | 25,262 |
| opt_6_7B | 32 | 3 | 99 | 102 | 13,106 |
| qwen3_8B | 8 | 4 | 34 | 38 | 31,240 |

