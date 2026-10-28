# Current / RKNN 2.3.2 / CPU FP32 comparison

Fresh three arm comparison on the RK3588 board using the 74f6266 source snapshot, the frozen compiled manifest and all 10,000 MNIST test images. Each timed call starts from raw uint8 input and ends with owned contiguous float32 logits. Three rounds rotate current, vendor and CPU order per batch. Argmax, numeric checks, references and artifact writes are outside the timer.

## Results

Each latency cell is `median of the three round means / median of the three round p50s` in microseconds. Ratios are raw `current / vendor` and `current / CPU` latency quotients; values below 1 mean current took less time. Throughput is the median of the three round image rates and includes all batches and tails.

| Profile | Current mean / p50 (µs) | RKNN 2.3.2 mean / p50 (µs) | CPU FP32 mean / p50 (µs) | Current/vendor | Current/CPU | Throughput current / vendor / CPU (img/s) |
|---|---:|---:|---:|---:|---:|---:|
| MLP32 | 1411.8 / 1407.7 | 600.3 / 595.0 | 6889.1 / 6887.0 | 2.352 | 0.205 | 22590.2 / 53164.9 / 4644.8 |
| MLP8 | 843.5 / 760.7 | 349.5 / 337.9 | 1775.2 / 1774.5 | 2.414 | 0.475 | 9484.0 / 22893.1 / 4506.5 |
| MLP1 | 865.0 / 914.1 | 277.1 / 279.4 | 271.1 / 271.2 | 3.122 | 3.190 | 1156.1 / 3609.2 / 3688.1 |
| ViT8 | 3607.5 / 3681.1 | 1692.2 / 1665.0 | 5984.1 / 5988.7 | 2.132 | 0.603 | 2217.6 / 4727.7 / 1336.9 |
| ViT1 | 1474.2 / 1558.5 | 604.7 / 660.9 | 928.1 / 929.2 | 2.438 | 1.588 | 678.3 / 1653.7 / 1077.4 |

MLP32 schedules 312 primary batches of 32 and two tail batches of 8. Tail timing is separate; overall throughput above includes both tails.

| MLP32 tail calls (batch 8) | Current mean / p50 (µs) | RKNN 2.3.2 mean / p50 (µs) | CPU FP32 mean / p50 (µs) |
|---|---:|---:|---:|
| 2 calls per round | 1079.3 / 1085.3 | 384.8 / 393.7 | 1775.5 / 1777.7 |

## Correctness and precision policy

All five profiles processed 10,000 images in each of three rounds. Current and vendor produced the same top-1 accuracy as the CPU FP32 reference on every round: MLP profiles 98.31%, ViT profiles 98.21%; top-1 agreement with CPU was 100% for both NPU arms. Maximum absolute logit error against CPU was:

| Profile family | Current | Vendor RKNN |
|---|---:|---:|
| MLP32 | 0.050686 | 0.030495 |
| MLP8 | 0.050686 | 0.030495 |
| MLP1 | 0.050686 | 0.030495 |
| ViT8 | 0.038425 | 0.122771 |
| ViT1 | 0.038425 | 0.107146 |

These are descriptive metrics for this comparison and do not alter the frozen admission limits. The vendor ViT models exceed the frozen packed precision gate, so ViT rows are performance measurements under a separate precision policy and are excluded from same-policy ranking. CPU FP32 is also a separate precision policy. No pre-optimization results are included.

## Run identity and evidence

- Source HEAD: `74f62665c027c87922679b74ce630ba39325e444`.
- Compiled manifest SHA-256: `16fc690686729cc4c11029b533b1ea95ac5dafc5d5c4e5149c19cc3d03a83869`.
- Frozen packed numeric policy SHA-256: `2ccefee0d19dbb558622c67f7020331b573357c44434e7dde24e26d98514ecb2`.
- Current runtime SHA-256: `7fd5a94b200568b942c2920348d9fae8d4d50ebab7cefb282f4b24bd4f34d0d7`.
- Vendor runtime SHA-256: `d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8`; loaded mapping path `/usr/lib/librknnrt.so` matched mapped device/inode and this SHA. The temporary bind mount was removed; the prior system runtime hash `0ebc1b408f897863a91a1b9ed60f3838a801386c7b1ef7c54d55ead624cd8347` was restored.
- RKNN Toolkit Lite reported runtime 2.3.2 and driver 0.9.8. Vendor contexts used `RKNNLite.NPU_CORE_0`.
- CPU calls ran pinned to CPU6 with one process thread and BLAS/OpenMP thread counts set to 1. The harness observed CPU6 maximum frequency 2.304 GHz; the worker required performance governors and CPU6 2.304 GHz, NPU 1 GHz and DMC 2.112 GHz before both smoke and full runs.
- 32-image coexistence smoke passed in `current → vendor → CPU → current` order with bitwise equal current logits.
- The report records `all` mode, three rounds, complete per-call timings and output hashes. Full JSON exit code: 0.

Artifacts are under [`results/20261028T012703Z-current-backend-comparison/`](results/20261028T012703Z-current-backend-comparison/). `board/full.json` has per-round summaries, exact schedules, numeric metrics and identities; `board/full_artifacts/raw_samples.jsonl` has every call and arm duration; `board/full_artifacts/logits.*.npy` retains each arm/round output and references. `board/full.json.stdout`, `.stderr`, `.exit`, and `.runtime-mount.txt` preserve the worker outcome, runtime log, exit status and mount restoration evidence. `campaign/SHA256SUMS` covers deployed campaign files except itself; `SHA256SUMS` in the result directory covers the fetched run evidence except itself. The command wrapper and native helper are in `scripts/run-current-backend-comparison-host.sh` and `scripts/run-current-backend-comparison-native.sh`.
