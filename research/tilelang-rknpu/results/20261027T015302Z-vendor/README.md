# Vendor baseline campaign

Date: 2026-10-27 (UTC).  Two rounds, full 10,000-image MNIST test split for all
five static profiles, same data/weights/batch-tail, NPU core 1, Host CPU
affinity 6, single-thread BLAS.

## Arms

- **current**: current custom RKPlan/hybrid default, runtime source `60f4a82`
  (same compiled package as `results/20261026T120625Z-final`).
- **vendor**: RKNN-Toolkit2 2.3.2 -> RKNN Lite2 2.3.2, non-quantized FP16,
  `single_core_mode=True`, runtime NPU core 0.  ONNX input is raw `[B,784]`
  float32; normalization/patchification are in the vendor graph.
- **CPU**: NumPy FP32 model reference with the same static batch schedule and
  the same raw inputs.

The pre-optimization column in `analysis.md` is from the earlier final campaign
`results/20261026T120625Z-final/`, not from this vendor campaign.

## Fixed identities

- Vendor compiled manifest:
  `../../vendor/rknn/rknn-manifest.json`
- ONNX manifest:
  `../../vendor/onnx/export-manifest.json`
- Runtime 2.3.2 sha256:
  `d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8`
- Lite2 wheel sha256:
  `821e80c95e6838308c133915660b1a6ae78bb8d079b2cbbd46a02dae61192d33`
- Board: RK3588 OPi 5 Plus, kernel `6.1.43-rockchip-rk3588-rknpu098`,
  driver `v0.9.8`, NPU frequency 1.0 GHz.
- Campaign manifest:
  `campaign/manifest.json`; current compiled manifest:
  `72f898e989240ad45a864d88596838a21e9c5e7d0f74feb416efef97c20ddadd`.

## Evidence

- `analysis.json` / `analysis.md`: warm per-call latency, throughput, setup
  cost, structure and accuracy/numeric comparison.
- `model-results/rN-{current,vendor,cpu}/`: per-round JSON, logs and wall time.
- `../../vendor/onnx/`, `../../vendor/rknn/`: exact comparison artifacts.

## Headline

Vendor is **2.40x / 2.54x / 2.86x** faster than current on MLP b32/b8/b1 and
**2.53x / 3.01x** faster on Tiny ViT b8/b1.  CPU is faster than both NPU paths
at b1 (MLP 4.86x, ViT 3.51x faster than current).  Vendor precision matches
accuracy/top-1 but ViT max logit error 0.1228/0.1072 exceeds the packed 0.0625
per-logit tolerance; it needs an independent numeric policy.

Replay build and board commands are in
[`../../vendor/README.md`](../../vendor/README.md); the analysis command is
`PYTHONPATH=.:python python3 tools/analyze_vendor_baseline.py ...`.
