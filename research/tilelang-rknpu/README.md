# tilelang-rknpu

下一阶段目标：[完整 MLP 与 hybrid Tiny Transformer 模型推理](NEXT-MILESTONE.md)；以真实模型补齐能力，保持已完成 MVP 的验收边界。

**Experimental checkpoint; declared MVP scope complete.  Not approved as a general release or for
unbounded device execution.**  Review blockers 1-4 and their follow-ups are fixed on
the Host with negative tests.  P0 is `complete`: the locked TileLang/TVM sources
were independently built, imported from an isolated source-build tree, and exercised
by the existing frontend/planner/FFI regressions.  The native `tvm_ffi` loaded by
the source environment is the source-built library, verified through
`find_libtvm_ffi()` and `/proc/self/maps`.  A source-built-plan RK3588 admission
campaign passes on three serial single-core masks.  Failure handling is explicitly
*safe isolation and stop*: recovery/continued computation after unknown device state
is out of scope and not claimed.  RK1228 remains independent.  See
[P0-SOURCE-LOCK.md](P0-SOURCE-LOCK.md), [review findings](REVIEW-2026-10-20.md),
[status](IMPLEMENTATION-STATUS.md) and [RK1228-TARGET.md](RK1228-TARGET.md).

Minimal vertical slice for the design in [overview.md](overview.md) and
[specific.md](specific.md):

```text
real TileLang PrimFunc
  -> RKIR (frozen dataclass)
  -> RKPlan (MatMul K chain, optional activation, same-shape add/mul)
  -> C descriptor (independent validation + deep copy)
  -> existing co_emit_* / co_execute_profiled seam
```

Plan JSON is schema v2: multi-tile outputs publish explicit placement rectangles
(`dst_row`, `dst_col`, `rows`, `cols`).  v1 fixtures are rejected, not migrated.
Current status and remaining review items: [IMPLEMENTATION-STATUS.md](IMPLEMENTATION-STATUS.md).

The compile/validate/explain path never opens a device.  Runtime is synchronous and
uses a caller-owned fd.  Host tests use BO/execution mocks.  RK3588 real-device
admission is reproducible through `scripts/run-board-admission.sh`; the latest
evidence is `results/20261023T144054Z-rk3588/`: representative small-shape
admission, 27 cases on each of three **serially run** single-core masks, with raw per-round outputs,
declared + unquantized float64 math references, byte-level chain/separate
comparison, final-publication BO guard, numeric guards, tiled per-node
dataflow, the board CLI path and exact C runtime target identity check.  P8 is
complete for the declared finite support domain; real-device recovery after an
unknown fault is an explicit out-of-scope limitation, not an implicit MVP gate.
Support boundaries are in
[SUPPORT-DOMAIN.md](SUPPORT-DOMAIN.md), necessary-condition evidence in
[ACCEPTANCE-MATRIX.md](ACCEPTANCE-MATRIX.md), runtime failure ownership in
[RUNTIME-LIFECYCLE.md](RUNTIME-LIFECYCLE.md) and replay commands in
[REPLAY.md](REPLAY.md).
RK1228 is a separate target and must be identified independently
([RK1228-TARGET.md](RK1228-TARGET.md)).

## Build and test

```bash
cmake -S research/custom-ops -B research/custom-ops/build-host-mvp
cmake --build research/custom-ops/build-host-mvp -j4
ctest --test-dir research/custom-ops/build-host-mvp --output-on-failure

PYTHONPATH=research/tilelang-rknpu/python \
RKNPU_PLAN_LIB=research/custom-ops/build-host-mvp/librknpu_plan.so \
python3 -m pytest research/tilelang-rknpu/tests
```

TileLang is optional.  Its source identity (tag `v0.1.15` -> commit `a35f8df`,
`3rdparty/tvm` -> `4211874e`), the no-patch statement, the hash-locked dependency
set and the clean-install procedure are recorded in
[P0-SOURCE-LOCK.md](P0-SOURCE-LOCK.md) and [provenance.json](provenance.json).
Reproduce the environment with:

```bash
scripts/lock_deps.sh lock
scripts/lock_deps.sh install                       # fresh .pydeps-p0
TILELANG_DEPS_DIR=$PWD/.pydeps-p0 scripts/lock_deps.sh verify
scripts/run-verification.sh              # defaults to .pydeps-p0, exports its env
```

P1 batch 1 extracts real TileLang K loops: the canonical
`for ko in T.Pipelined(K // BK): copy(A[0, ko*BK]); copy(B[ko*BK, 0]); gemm(...)`
becomes one RKIR matmul with logical `K = trips * BK`, and the planner splits it
into 256-wide K tiles while recording the source tiling and the dropped pipelining
hint in the plan.  The MVP semantic K block is fixed at 256 (specific.md 12.4), so
a source block of 128 is rejected rather than merged.  Ragged K, copy/GEMM
mis-ordering, a cleared accumulator, a duplicated or extra GEMM, an unregistered
tileop, a loop variable leaking into another axis and work inside the loop are all
rejected (`tests/test_frontend_kloop.py`).

SwiGLU FFN: `Y = (SiLU(X@Wg^T) * (X@Wu^T)) @ Wd^T + X` is lowered stage by stage
(gate, up, SiLU, mul, projection, residual), one submit per primitive stage with
every intermediate staying on device.  The canonical graph uses the generic
per-node path by default.  `allow_experimental=True` may additionally select the
whole-graph specialisation and its `swiglu_ffn_v1` recipe; the capability
`co.swiglu_ffn.f16.v1` remains **experimental**, but it is no longer a gateway for
compiling the semantically equivalent graph (`tests/test_swiglu.py`,
`tests/test_ffn_execution.py`).  The real three-GEMM TileLang FFN fixture is
extracted when every FP16 point is explicit, including projection -> FP16 before
the residual add.  A variant that leaves the residual in the FP32 accumulator and
narrows only in the final `T.copy` is rejected with `RK_E_NUMERIC_CONTRACT`
(`tests/test_frontend_ffn.py`).

### RK3588 real-device admission

Connection details are read only from the ignored repository `.env`; the campaign
never prints or archives the password.  It generates plans/data on the Host,
uploads an isolated source bundle, rebuilds the production `librknpu_plan.so`,
and runs through the real C ABI on core masks `1`, `2` and `4` in chain and
separate modes:

```bash
research/tilelang-rknpu/scripts/run-board-admission.sh
# evidence: results/<utc>-rk3588/board/summary.md
```

The latest run is `results/20261023T144054Z-rk3588/`: RK3588 OPi 5 Plus, driver
`v0.9.8`, ABI 1, **27 cases × 3 masks = 81/81 small-shape case runs PASS**.
Coverage includes MatMul, projection, staged-FP16 FFN, one explicit guarded
ragged K/M/N tail shape, scalar/per-channel constant Add/Mul, independent
activation, a 256x256 physical LUT split, dual output, M/N multi-tile, K=512/257
shard landing, and tiled per-node projection/FFN/branch/activation at
`M=N=K=300`.  Every core mask also runs the CLI `compile-rkir` -> `validate` ->
`explain` -> two `run` rounds -> bad-binding path.  Raw outputs and both
declared-path and unquantized float64 math references are archived per round;
math references are not narrowed on storage/read.  The acceptance predicate is
explicitly `abs(got-declared) <= 8e-3 + 8e-3*abs(declared)` on the declared
staged path.  Chain/separate raw outputs are byte-identical.  At final
publication, after all submits and staging downloads, the runtime checks every
data BO's slack/guard bytes with `co_guard_check`; a Host regression with a task
after a download verifies no publish and workspace poisoning.  The source-bundle
hash is independently recomputed; the remote loaded-library hash is recorded but
not independently rehashed.

The three core masks are serial single-core runs, not a concurrent-throughput
claim.  Multi-board, concurrent three-core, performance and long-duration stress
are extension validation, not MVP/P8 blockers.  P8 is complete for the declared
MVP scope; post-fault recovery is an explicit out-of-scope limitation.

### CLI path

`python -m rknpu_backend.cli` exposes `compile-tilelang`, `compile-rkir`,
`validate`, `explain` and `run`.  The board uses the persisted-RKIR path because
the isolated bundle does not ship the pinned TileLang build:

```bash
python -m rknpu_backend.cli compile-rkir input.rkir.json \
  --profile rk3588_vendor_co_v1 --numeric-mode rk_f16_tiled --out plan.json
python -m rknpu_backend.cli validate plan.json
python -m rknpu_backend.cli explain plan.json --format json
python -m rknpu_backend.cli run plan.json --bindings bindings.json \
  --output-dir out/
```

Bindings values are `.npy` paths resolved relative to the bindings JSON.  The CLI
opens `/dev/dri` and calls the vendor's real `void rocket_close(int)` only after
workspace destroy confirms quiescence; `rocket_close` has no status result, so
successful fd release is not separately claimed.  The additive `--fd N` option
uses a caller-owned descriptor that the CLI never closes.  If destroy returns
POISONED/BUSY, the provider fd/task BO are retained for process-level recovery
rather than closed.  Success publishes `<output>.npy` atomically;
schema/binding/numeric failures return 2 without publishing, and resource/device
failures return 3.  Board evidence is under `board/core-*/cli/cli-report.json` in
the campaign result.

Semantic guards shared by both extraction paths: the GEMM `clear_accum` flag must
be false, `tl.region` access masks must match the load/store role the pattern
expects, block and K loops must start at zero and step by one, RKIR tensors must be
dense row-major, and a submit's `recipe_id` must match its task sequence
(`tests/test_ir.py`, `tests/test_plan_verify.py`, `tests/test_frontend*.py`).

P1 batch 2 extracts M/N-tiled kernels: a 1-D or 2-D block grid is accepted only
when `grid * block_tile` covers the output exactly once and the block variables
appear on the right operand axes (`A[by*BM, ...]`, `B[..., bx*BN]`,
`C[by*BM, bx*BN]`).  The plan records the source tiling
(`source_mn_tiling.coverage = "exact_once"`) next to the 256-wide lowering tiling,
so the re-tiling is explicit rather than implicit (`tests/test_frontend_mntile.py`).

The restricted lowering follows the real graph edges: the add/mul operand is
resolved from the RKIR Value IDs and the module output root is checked, so a graph
that does not match `matmul -> (activation?) -> (add|mul)` is rejected instead of
being silently rewritten (`tests/test_dataflow.py`).

The no-K-loop path also accepts the closed
`Z=X@W1; S=SiLU(Z); P=S@W2; Y=P+X` static dataflow.  It emits a four-node SSA
RKIR graph, lowers each node to a device cube view, keeps Z/S/P in device BOs,
and lets the second GEMM consume the activation output directly.  The real
TileLang fixture has five explicit FP16 narrowing points and is executed through
the mock FFI with an independent numpy reference over two input rounds
(`tests/test_frontend_silu_proj.py`, `tests/test_silu_proj_execution.py`).
Wrong-branch, missing-narrowing and unsupported-op neighbours are refused.

The generic static-dataflow path also supports multiple exported values and
branch graphs.  Each output gets its own download/publish, and an exported
intermediate stays live through later device consumers.  A reordered SwiGLU
branch graph is executed with the per-node path (without the experimental
whole-graph `swiglu_ffn_v1` matcher) and checked against the numpy reference
(`tests/test_multi_output_execution.py`, `tests/test_ffn_execution.py`).  A real
TileLang two-GEMM/two-output fixture is also extracted and executed through the
mock runtime.  The real three-GEMM TileLang FFN frontend is covered by the
explicit-FP16 positive fixture and its FP32-residual negative control.

Ragged M/N/K GEMMs are accepted only through one explicit guarded form: both
physical input tiles are zero-filled on every K iteration, valid elements are
loaded under affine rectangle bounds, and the output store is guarded to the
logical M/N rectangle.  `ceildiv` alone is not treated as an access-safety proof.
The RKIR keeps the full logical shapes plus a `source_tail_contract`; the planner
uses its existing 256-wide K chain and M/N output-tail publish tiles.  Missing
zero-fill, off-by-one, wrong-axis and unguarded-write negatives are covered by
`tests/test_frontend_ragged.py` and `tests/test_ragged_execution.py`.

Compile-time constant broadcast is supported for same-shape Add/Mul with a rank-0
scalar or rank-1 `[N]` per-channel constant.  The constant is expanded into the
full `[M,N]` cube by the explicit Host upload action once at workspace creation;
the EW emitter is unchanged and still consumes two same-shape cube views.  Host
budget/accounting includes the expanded tile, and runtime tensor/dynamic/arbitrary
broadcast remains an explicit rejection (`tests/test_frontend_broadcast.py`,
`tests/test_broadcast_execution.py`).

The verification campaign runs every Python step under `.venv-p0/bin/python`
(created with `python3 -m venv --without-pip`, so it does not inherit
`/usr/lib/python3/dist-packages`) and fails if any third-party module resolves
outside `.pydeps-p0`.

`scripts/setup-tilelang-env.sh` builds the older ad-hoc `.pydeps` tree that is kept
only as the migration baseline.  The real fixture can be compiled without touching a
device:

```bash
cd research/tilelang-rknpu
export PYTHONPATH=$PWD/python:$PWD/.pydeps-p0
export TILELANG_CACHE_DIR=$PWD/.cache
export LD_LIBRARY_PATH=$PWD/.pydeps-p0/z3/lib:$PWD/.pydeps-p0/tilelang/lib:$PWD/.pydeps-p0/tvm_ffi/lib
export RKNPU_PLAN_LIB=$PWD/../custom-ops/build-host-mvp/librknpu_plan.so
python -m rknpu_backend.cli compile-tilelang \
  --file examples/matmul_silu.py --symbol kernel \
  --numeric-mode rk_f16_tiled --allow-approx-lut --out /tmp/mm_silu.rkplan.json
python -m rknpu_backend.cli validate /tmp/mm_silu.rkplan.json
python -m rknpu_backend.cli explain /tmp/mm_silu.rkplan.json --format json
```

`swiglu.py` is intentionally a real multi-GEMM TileLang negative fixture until the
next work package extends the extractor.  It is not silently lowered.
