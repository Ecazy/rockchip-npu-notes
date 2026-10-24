# Next milestone: two complete models

Status: **complete for the declared first-round scope** (2026-10-24).  This is
the independent N0-N3 milestone described in [NEXT-MILESTONE.md](NEXT-MILESTONE.md);
it does not retroactively change the already accepted MVP.

## Delivered models

### MNIST MLP

- Learned weights: `models/mnist_mlp.npz` (safe `npz`, no pickle).
- Structure: `784 -> 256 -> 128 -> 10`, bias and exact ReLU at each hidden layer.
- Training: fixed seed 20261024, MNIST train split only, 12 epochs AdamW.
- CPU FP32 test accuracy: **0.9831** (9,831 / 10,000).
- NPU path: all linear layers, biases and ReLUs execute through real TileLang
  staged PrimFuncs -> RKIR -> RKPlan -> production runtime.
- Intermediate activations remain in device BOs; the plan has one final output
  download and no intermediate download/re-upload bridge.
- Static profiles: batch 1, 8 and 32 are separately compiled.  The logical
  784-wide input is zero-padded to 1024 because the locked ordered-K primitive
  uses full 256-wide tiles; the first 10 of the padded 16 logits are returned.
  The 10,000-image split is processed as 312 × batch 32 plus two compiled batch-8
  tail calls; no dynamic shape is used.
- Source: `models/tilelang_mlp.py`; plan compiler: `tools/compile_models.py`.

### Tiny Transformer image classifier

- Learned weights: `models/tiny_vit.npz` (safe `npz`, no pickle).
- Structure: 16 raster 7x7 patch tokens, D=64, 4 heads, one pre-RMSNorm encoder
  layer, pre-RMSNorm SwiGLU FFN=128, residual connections, mean pooling, 10-way
  linear classifier.
- Training: fixed seed 20261024, MNIST train split only, 30 epochs AdamW.
- CPU FP32 test accuracy: **0.9821** (9,821 / 10,000).
- NPU matrix subgraphs: patch projection, QKV, QK, PV, attention output
  projection, FFN gate/up/down and classifier.  Every one is a real TileLang
  PrimFunc -> RKIR -> RKPlan -> production runtime.
- First-round Host boundary, explicitly recorded: patch rasterization,
  position add, RMSNorm, stable softmax, head layout merge/split, SiLU/gated
  multiply, residual adds and mean pooling.  These are Host compute, not an
  opaque NPU fallback.
- Static profiles: batch 1 and 8; QK/PV use one 16x16 dynamic-B plan reused
  across all batches/heads.
- Source: `models/tilelang_matmul.py`; plan compiler: `tools/compile_models.py`.

## Numerical and task acceptance

Full 10,000-image MNIST test split was used; training never saw the test split.

| Model | Batch/full split | CPU FP32 acc | RK3588 acc | Δ | CPU top-1 agreement | Declared top-1 agreement |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 32 / 10,000 | 0.9831 | 0.9831 | 0.0000 | 1.0000 | 1.0000 |
| Tiny Transformer | 8 / 10,000 | 0.9821 | 0.9821 | 0.0000 | 1.0000 | 1.0000 |

Additional profile checks on the deterministic 512-image subset:

- MLP batch 8 and batch 1: both 0.986328125, CPU/declared top-1 agreement 1.0;
- Tiny Transformer batch 1: 0.986328125, CPU/declared top-1 agreement 1.0;
- masks 2 and 4 use the same 512-image subset; mask 1 is the complete split.

Logit error, full split:

- MLP: max |hardware − CPU FP32| = 0.05069; max |hardware − declared FP16 path|
  = 0.03125;
- Tiny Transformer: max |hardware − CPU FP32| = 0.03843; max |hardware −
  declared FP16 path| = 0.03125.

The declared path is a reproducible Host-side emulation of ordered FP16 writeback
and the explicit Host boundary; it is not a substitute for the CPU FP32 model.

## Board evidence and cost

Latest complete-model campaign:
`results/20261024T031249Z-models/`.

- core 1 (fixed core): MLP full split, 10,000 images, **6.223 s**,
  **1,607 images/s**; Transformer full split, **65.294 s**, **153 images/s**.
- core 2 and core 4: 512-image deterministic admission subset, all checks pass.
- MLP full-split transfer: 20.50 MB upload / 0.64 MB download.
- Transformer full-split transfer: 391.04 MB upload / 287.68 MB download,
  88,750 NPU tasks/submits, 3.20 s Host compute; this is the first-round hybrid
  bridge cost, not hidden device arithmetic.
- CPU baseline on the same board/data/preprocessing: MLP 4,609 images/s,
  Transformer 1,214 images/s; CPU frequency range 408–2,304 MHz, 8 CPUs.
  Therefore this report does **not** claim NPU is faster: the MLP and first-round
  Transformer bridge are slower than this CPU reference.
- Board verifier: `results/20261024T031249Z-models/verify.log` **PASS**.
- Host/mock full evaluation: `evidence/model_eval_mock.json`; CPU benchmark:
  `evidence/cpu_benchmark.json`.

## Reuse and failure behavior

- `PlanCall` creates the CPlan and workspace once, uploads constants once and
  updates input bindings on each call.
- The MLP test runs different inputs through the same workspace and compares
  each against the declared reference.
- A model-level injected submit failure in
  `tests/fixtures/model_failure_worker.py` returns no result, publishes no output
  and leaves the workspace POISONED; no retry is attempted.
- Existing BUSY/quarantine/no-publish runtime tests remain active.

## Reproduce

```bash
# Host: compile all static model profiles from real TileLang sources
PYTHONPATH=$PWD:$PWD/python:$PWD/.src/p0-build-src/tilelang:$PWD/.src/p0-source-env \
LD_LIBRARY_PATH=$PWD/.src/p0-build-tilelang/lib:$PWD/.src/p0-build-tvmffi-python/lib:$PWD/.pydeps-p0/z3/lib \
python3 tools/compile_models.py

# Host: complete test-set execution through the mock C ABI
PYTHONPATH=$PWD/python:$PWD/.src/p0-build-src/tilelang:$PWD/.src/p0-source-env \
LD_LIBRARY_PATH=$PWD/.src/p0-build-tilelang/lib:$PWD/.src/p0-build-tvmffi-python/lib:$PWD/.pydeps-p0/z3/lib \
python3 tools/evaluate_models.py --backend mock --models both \
  --out evidence/model_eval_mock.json

# Full model campaign on RK3588 (core 1 full split; masks 2/4 512-image subset)
./scripts/run-model-admission.sh
```

The model package is described by `models/manifest.json`; exact source, plan,
weight and dataset hashes are recorded there and in the campaign manifest.

## Explicit non-claims / next boundary

This milestone does **not** add dynamic shapes, generic ONNX import, arbitrary
TileLang control flow/masks, CNN operator coverage, NPU training, INT8/INT4,
full LLM loading, KV cache/state, RK1228 support, multi-board/concurrent
throughput, or post-fault recovery.  Those are separate extensions.  The
Transformer's first-round Host bridge is intentionally reported rather than
silently treated as a pure-NPU implementation.
