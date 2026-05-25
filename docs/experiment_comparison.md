# Experiment Comparison Report

| Experiment | Total(us) | Layer(us) | Gen(us) | Proc | Q@K(us) | per_seq | S@V(us) | qk_rb | qk_pe | sv_rb | sv_pe | pim_rb | pim_pe |
|------------|-----------|-----------|---------|------|---------|---------|---------|-------|-------|-------|-------|--------|--------|
| exp1_fp16_gpu                |      8684 |       271 |     192 | GPU  |      95 |     6.0 |      96 |     0 |     0 |     0 |     0 |      0 |      0 |
| exp2_fp16_pim                |      6788 |       212 |     132 | PIM  |      66 |     4.1 |      66 |    33 |    33 |    33 |    33 |     66 |     66 |
| exp3_2bit_gpu                |      4363 |       136 |      95 | GPU  |      48 |     3.0 |      48 |     0 |     0 |     0 |     0 |      0 |      0 |
| exp4_2bit_hybrid             |      2534 |        79 |      38 | PIM  |       4 |     0.3 |      34 |     4 |     2 |     0 |     0 |      4 |      2 |
| exp5_2bit_allpim             |      3712 |       116 |      75 | PIM  |       4 |     0.3 |      71 |     4 |     2 |     4 |    37 |      8 |     39 |
| exp6_2bit_dequant_pim        |      3711 |       116 |      75 | PIM  |      37 |     2.3 |      37 |     4 |    37 |     4 |    37 |      8 |     75 |

## PIM Cycle Verification (exp2 FP16 PIM)

- per-GEMV: rb=128.0ns pe=65.1ns
- per-direction (×4grp ×8heads ×16seqs): rb=65.5us pe=33.3us
- Reported qk_rb=33.2us qk_pe=33.2us sv_rb=33.2us sv_pe=33.2us
- ✓ Verified

## PIM Cycle Verification (exp5 2-bit All-PIM)

- 2-bit packed: 4 elements/B, PE 64 elements/cycle
- per-bank: 260.3B, rb_fills=1
- per-GEMV: rb=8.1ns pe=8.1ns
- Reported qk_rb=4.2us qk_pe=2.1us
- ✓ Verified
