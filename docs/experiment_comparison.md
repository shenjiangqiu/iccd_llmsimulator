# Full Comparison Report

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


## Cross-Model Comparison (Hybrid)

| Model | kv_heads | Score | Aggregate | Gen | Decode tok/s |
|---|---|---|---|---|---|
| llama3_8B | 8 | 4 | 34 | 38 | 25,262 |

