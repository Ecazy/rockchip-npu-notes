# Vendor RKNN baseline

This directory is the reproducible vendor comparison artifact for the frozen
MNIST MLP and Tiny ViT models.  It is not part of the production backend and it
does not replace any numeric acceptance policy.

## Fixed identities

- RKNN-Toolkit2: **2.3.2**
- Board runtime: `runtime/2.3.2/librknnrt.so`, version **2.3.2**
  - sha256 `d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8`
- RKNN-Toolkit-Lite2 wheel:
  `wheel/rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl`
  - sha256 `821e80c95e6838308c133915660b1a6ae78bb8d079b2cbbd46a02dae61192d33`
- Target: `rk3588`, non-quantized `float16`, `single_core_mode=True`,
  runtime NPU core 0.
- ONNX graphs: `onnx/`; RKNN graphs: `rknn/`; manifests contain per-file
  SHA-256 and model bytes.

The board used for the recorded campaign has kernel
`6.1.43-rockchip-rk3588-rknpu098`, driver `v0.9.8`, NPU frequency 1.0 GHz.
The system runtime is 1.4.0, so the campaign bind-mounts the local 2.3.2
runtime over `/usr/lib/librknnrt.so` while measuring and unmounts it afterwards.

## Build

Host Python needs `rknn-toolkit2==2.3.2`, `onnx==1.17.0`, PyTorch and
`onnxruntime`.  The repository does not install these into the system Python.

```bash
cd research/tilelang-rknpu
PYTHONPATH=.pydeps-p0:/tmp/rknn-host-site \
  python3 tools/export_vendor_models.py --check
PYTHONPATH=/tmp/rknn-host-site \
  python3 tools/build_vendor_rknn.py
```

`tools/export_vendor_models.py` uses the same weights and FP32 model math as
`python/rknpu_backend/model_reference.py`; ONNX Runtime checks on the deterministic
sample report max abs error below 1e-5 and matching top-1.

## Board measurement

Install the Lite2 wheel and its Python dependencies into a local target:

```bash
python3 -m pip install --no-deps --target "$PWD/site" vendor/wheel/*.whl
python3 -m pip install --no-deps --target "$PWD/site" ruamel.yaml psutil
```

Then, as root, bind-mount the 2.3.2 runtime for the duration of the run:

```bash
mount --bind "$PWD/vendor/runtime/2.3.2/librknnrt.so" /usr/lib/librknnrt.so
trap 'umount /usr/lib/librknnrt.so' EXIT
```

Vendor and CPU runs use the same static batch schedule and full 10,000-image
test split:

```bash
export PYTHONPATH="$PWD/site:$PWD/python"
taskset -c 6 python3 tools/benchmark_vendor_models.py \
  --mode vendor --data-dir .data/mnist --weights-dir models \
  --rknn-dir vendor/rknn --out evidence/vendor.json --limit 10000
taskset -c 6 python3 tools/benchmark_vendor_models.py \
  --mode cpu --data-dir .data/mnist --weights-dir models \
  --rknn-dir vendor/rknn --out evidence/cpu.json --limit 10000
```

The current RKPlan/hybrid arm in the recorded campaign was run by
`scripts/run-final-delivery-native.sh eval` with the current source/build/plans,
the same 10,000-image test split and full profile samples.  The two recorded
rounds and their raw JSON/logits are under
`results/20261027T015302Z-vendor/`.

## Numeric interpretation

Vendor accuracy and top-1 agreement match the CPU reference for all five
profiles.  Vendor logits are not a drop-in replacement for the packed per-logit
policy: MLP stays within 0.0625, while Tiny ViT reaches 0.1228/0.1072.  Any
deployment using this vendor path needs its own numeric policy/evidence; it is
not claimed as a bitwise or packed-tolerance equivalent of the custom backend.

Per-layer target placement can be inspected with `tools/probe_rknn_perf.c`:

```bash
gcc -O2 -I /path/to/rknn_api/include tools/probe_rknn_perf.c \
  -L vendor/runtime/2.3.2 -Wl,-rpath,$PWD/vendor/runtime/2.3.2 \
  -lrknnrt -o /tmp/probe_rknn_perf
taskset -c 6 /tmp/probe_rknn_perf vendor/rknn/tiny_vit_b8.rknn 8
```

The probe enables `RKNN_FLAG_COLLECT_PERF_MASK`, which adds overhead; use its
target assignment/relative ranking, not its absolute total time.  Recorded
reports are in `results/20261027T015302Z-vendor/vendor-perf/`.
