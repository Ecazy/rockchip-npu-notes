# tilelang-rknpu

**Experimental, incomplete checkpoint. Not approved for device execution.**
The review reproduced missing C validation for upload strides and task ordering;
M/N tiling and the full frontend subset remain unfinished. See
[review findings](REVIEW-2026-10-20.md) and [status](IMPLEMENTATION-STATUS.md).

Minimal vertical slice for the design in [overview.md](overview.md) and
[specific.md](specific.md):

```text
real TileLang PrimFunc
  -> RKIR (frozen dataclass)
  -> RKPlan (MatMul K chain + optional SiLU/sigmoid)
  -> C descriptor (independent validation + deep copy)
  -> existing co_emit_* / co_execute_profiled seam
```

The compile/validate/explain path never opens a device.  Runtime is synchronous and
uses a caller-owned fd.  Host-only tests use BO/execution mocks; no board acceptance
is claimed in this checkout.

## Build and test

```bash
cmake -S research/custom-ops -B research/custom-ops/build-host-mvp
cmake --build research/custom-ops/build-host-mvp -j4
ctest --test-dir research/custom-ops/build-host-mvp --output-on-failure

PYTHONPATH=research/tilelang-rknpu/python \
RKNPU_PLAN_LIB=research/custom-ops/build-host-mvp/librknpu_plan.so \
python3 -m pytest research/tilelang-rknpu/tests
```

TileLang is optional.  The pinned wheel environment used for the frontend test is
recorded in [provenance.json](provenance.json) and can be recreated with
`scripts/setup-tilelang-env.sh`.  The real fixture can be compiled without touching a
device:

```bash
cd research/tilelang-rknpu
export PYTHONPATH=$PWD/python:$PWD/.pydeps
export TILELANG_CACHE_DIR=$PWD/.cache
export LD_LIBRARY_PATH=$PWD/.pydeps/z3/lib:$PWD/.pydeps/tilelang/lib:$PWD/.pydeps/tvm_ffi/lib
export RKNPU_PLAN_LIB=$PWD/../custom-ops/build-host-mvp/librknpu_plan.so
python -m rknpu_backend.cli compile-tilelang \
  --file examples/matmul_silu.py --symbol kernel \
  --numeric-mode rk_f16_tiled --allow-approx-lut --out /tmp/mm_silu.rkplan.json
python -m rknpu_backend.cli validate /tmp/mm_silu.rkplan.json
python -m rknpu_backend.cli explain /tmp/mm_silu.rkplan.json --format json
```

`swiglu.py` is intentionally a real multi-GEMM TileLang negative fixture until the
next work package extends the extractor.  It is not silently lowered.
