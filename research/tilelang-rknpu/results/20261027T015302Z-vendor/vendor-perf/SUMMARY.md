# Vendor operator placement probe

The probe uses the local RKNN Runtime 2.3.2 with
`RKNN_FLAG_COLLECT_PERF_MASK` and writes the per-layer target/time report.
`COLLECT_PERF_MASK` adds collection overhead, so the reported total operator
times are **not** used as benchmark latency; only target placement and relative
layer ranking are used.  The application-side warm measurements are in
`../analysis.md`.

| profile | NPU layers | CPU layers |
|---|---:|---:|
| mnist_mlp b1 | 6 | 3 (InputOperator, final Reshape, OutputOperator) |
| mnist_mlp b8 | 7 | 3 (same boundary ops) |
| mnist_mlp b32 | 7 | 3 (same boundary ops) |
| tiny_vit b1 | 28 | 5 (InputOperator, 2 RMSNorm Div, final Reshape, OutputOperator) |
| tiny_vit b8 | 50 | 5 (same boundary ops) |

For Tiny ViT b8, the largest NPU entries in the probe report are the reshape
chain (16 calls, ~923 us with collection overhead), `exSDPAttention` (~675 us),
transpose (10 calls, ~468 us), and the eight convolutions (~323 us); the two
RMSNorm `Div` operations remain on CPU (~58 us).  This is placement evidence
only, not a second application timing claim.

Raw reports are the `*.txt` files in this directory.  The probe source is
`tools/probe_rknn_perf.c`.
