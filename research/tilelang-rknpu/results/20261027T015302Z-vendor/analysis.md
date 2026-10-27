# Vendor baseline comparison

- status: **PASS**
- full test images per profile: 10000
- rounds: 2

Warm values are per batch call; vendor includes its graph preprocessing.

Speedups are `current_ms / arm_ms`; >1 means the arm is faster.

| profile | preopt ms* | current ms | vendor ms | CPU ms | vendor speedup | CPU speedup |
|---|---:|---:|---:|---:|---:|---:|
| mnist_mlp.b32 | 2.881 | 1.968 | 0.819 | 6.838 | 2.402x | 0.288x |
| mnist_mlp.b8 | 2.027 | 1.366 | 0.537 | 1.752 | 2.544x | 0.779x |
| mnist_mlp.b1 | 2.031 | 1.235 | 0.432 | 0.254 | 2.862x | 4.864x |
| tiny_vit.b8 | 34.631 | 10.421 | 4.120 | 5.867 | 2.529x | 1.776x |
| tiny_vit.b1 | 7.138 | 3.065 | 1.018 | 0.872 | 3.012x | 3.515x |

\* preopt comes from the earlier final campaign (`results/20261026T120625Z-final`); the other three columns share this vendor campaign.

## Accuracy and numeric

| profile | vendor acc | vendor top-1 | vendor max err | packed atol |
|---|---:|---:|---:|---:|
| mnist_mlp.b32 | 0.98310 | 1.00000 | 0.03049 | 0.0625 |
| mnist_mlp.b8 | 0.98310 | 1.00000 | 0.03049 | 0.0625 |
| mnist_mlp.b1 | 0.98310 | 1.00000 | 0.03049 | 0.0625 |
| tiny_vit.b8 | 0.98210 | 1.00000 | 0.12277 | 0.0625 |
| tiny_vit.b1 | 0.98210 | 1.00000 | 0.10715 | 0.0625 |

Vendor accuracy/top-1 equals the CPU reference for all five profiles.  MLP logits stay within the packed task-logit tolerance; ViT max errors (0.1228/0.1072) exceed the packed 0.0625 tolerance, so vendor fusion is classification-preserving but not a drop-in per-logit equivalence.

## Setup and structure

| profile | current setup ms | vendor setup ms | current task/submit | vendor rknn calls | CPU calls |
|---|---:|---:|---:|---:|---:|
| mnist_mlp.b32 | 37.5 | 203.3 | 3454/2512 | 314 | 314 |
| mnist_mlp.b8 | 17.6 | 60.0 | 13750/10000 | 1250 | 1250 |
| mnist_mlp.b1 | 18.6 | 61.2 | 110000/80000 | 10000 | 10000 |
| tiny_vit.b8 | 61.7 | 64.2 | 87500/10000 | 1250 | 1250 |
| tiny_vit.b1 | 25.0 | 61.7 | 140000/80000 | 10000 | 10000 |

## Wall time

| arm | wall seconds (2 rounds) |
|---|---:|
| current | 143.777286103 |
| vendor | 106.51869673 |
| CPU | 85.00899900900001 |

Wall includes CPU/declared reference calculations and trace capture; it is not the application-only interval.

Vendor RKNN uses toolkit/runtime 2.3.2, non-quantized float16, `single_core_mode=True`, runtime core 0; current uses the same single NPU core and CPU affinity 6.

**Interpretation:** vendor is the strongest batched path; CPU remains faster than both backends at b1 for MLP and ViT. The gap is therefore not a single QKV transport issue; it points to graph-level scheduling/fusion and Host boundaries.
