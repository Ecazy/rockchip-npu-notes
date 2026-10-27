# Runtime layer isolation harness

These are experiment tools, not default runtime code. They create the A/B/C/D
arms used in `RUNTIME-LAYER-ISOLATION.md`.

## Build the case bundles

From `research/tilelang-rknpu` with the board-independent Python deps available:

```bash
PYTHONPATH="$PWD/python:$PWD" \
python3 tools/runtime_layers/build_cases.py --out /tmp/runtime-cases
```

Five deterministic cases are produced: MLP b1/b32, QKV b8, QK group b8 and PV
group b8. `case.def` and `bundle.bin` are stable; `frames.bin` is regenerated
from the fixed seed, so it is not committed.

## Build the native C probe

On the board, using the final RKPlan library from the stage result:

```bash
gcc -std=c11 -O2 -I custom-ops/include -I custom-ops/src \
  -I upstream-components/rocket-userspace/include \
  runtime_layer_probe.c -L build-fastdl -lrknpu_plan -lm \
  -Wl,-rpath,$PWD/build-fastdl -o runtime_layer_probe
```

The source is `runtime_layers/runtime_layer_probe.c` relative to the repo root.
`b/c/d` are native C RKPlan/direct/resident modes; `battr` and `cprof` are the
separate attribution samples. Build the shared full-model adapter with:

```bash
gcc -std=c11 -O2 -fPIC -shared -DRL_DIRECT_SHARED -Dmain=rl_probe_main \
  -I custom-ops/include -I custom-ops/src \
  -I upstream-components/rocket-userspace/include \
  runtime_layer_probe.c -L build-fastdl -lrknpu_plan -lm \
  -Wl,-rpath,$PWD/build-fastdl -o librl_direct_group.so
```

## A arms

A0 is ordinary `PlanCall.run_into`/`run_group_into`; A1 installs prepared
bindings once and calls the legacy C entry directly from Python ctypes:

```bash
PYTHONPATH="$PWD/python" RKNPU_CORE_MASK=1 taskset -c 6 \
python3 tools/runtime_layers/benchmark_python_layers.py \
  --case /tmp/runtime-cases/qk_b8 \
  --lib build-fastdl/librknpu_plan.so --mode a1 --rounds 1 \
  --plans-dir models/compiled --weights-dir models \
  --out /tmp/qk-a1.bin
```

## B/C/D and profiles

```bash
./runtime_layer_probe /tmp/runtime-cases/qk_b8 b 1 /tmp/qk-b.bin
./runtime_layer_probe /tmp/runtime-cases/qk_b8 c 1 /tmp/qk-c.bin
./runtime_layer_probe /tmp/runtime-cases/qk_b8 d 1 /tmp/qk-d.bin
./runtime_layer_probe /tmp/runtime-cases/qk_b8 battr 1 /tmp/qk-battr.bin
./runtime_layer_probe /tmp/runtime-cases/qk_b8 cprof 1 /tmp/qk-cprof.bin
```

## Full-model targeted integration

`full_model_direct_group.py` constructs two identical Tiny ViT b8 compiled
graphs and replaces only the qk/pv group PlanCalls in the second with the C
direct context. `two_runner_control.py` is the no-adapter placement control.

```bash
PYTHONPATH="$PWD/python" RKNPU_CORE_MASK=1 taskset -c 6 \
python3 tools/runtime_layers/full_model_direct_group.py \
  --plans-dir models/compiled --weights-dir models \
  --lib build-fastdl/librknpu_plan.so \
  --adapter-lib librl_direct_group.so \
  --data-dir .data/mnist \
  --qk-case /tmp/runtime-cases/qk_b8 --pv-case /tmp/runtime-cases/pv_b8 \
  --batch 8 --limit 256 --rounds 3 --out /tmp/full-model/result.json
```

The result directory contains result JSON, current/variant logits `.npy`,
one `control256.json`, and timing logs. The C direct path is an experimental
adapter; it is not linked into the shipped backend and does not alter the
default safety contract.
