# Supported domain and explicit boundaries

Updated for the MVP closure checkpoint, 2026-10-23.

## Accepted source forms

- **TileLang classic paths**: canonical blocked MatMul (including K loop), M/N
  multi-tile single-GEMM, projection, staged-FP16 FFN, scalar/per-channel
  compile-time broadcast, real two-output branches, and the explicit guarded
  ragged K/M/N tail fixture.
- **Real multi-block TileLang stage pipelines**: separately launched `T.Kernel`
  stages whose global intermediate parameters are materialised between launches.
  Each stage writes exactly one parameter; later stages may read it; a stage may
  not read the parameter it writes.  Accepted positives cover a multi-block
  projection, a multi-block SwiGLU FFN and a multi-block two-output branch.
  K loops must write the FP16 accumulator back after every source K tile.  The
  extractor additionally rejects incomplete block coverage and any same-launch
  cross-block read that would require an unproved global barrier.
- **Persisted/generated RKIR**: static rank-2 SSA graphs using `matmul`,
  registered unary activations, and same-shape `add`/`mul`, plus one rank-0 or
  rank-`[N]` compile-time constant broadcast operand. Logical values may be
  tiled: the graph builder maps each RKIR value to a set of 256x256 output tiles
  and connects consumers by the owning tile.
- **Public offline builders used for tiled tests/campaigns**:
  `projection_module`, `swiglu_module` (generic path when it exceeds the
  experimental single-tile specialisation), `branch_outputs_module` and
  `activation_module` in `python/rknpu_backend/ir.py`.
- **CLI on the board**: `compile-rkir`, `validate`, `explain` and `run` are
  exercised on the real RK3588. `compile-tilelang` remains a Host command because
  the board bundle does not carry the pinned TileLang build.

## Shapes and tiling

- M, N and K are static integers in **1..4096**.
- Primitive extents are at most 256. Tiled per-node graphs use a 256x256
  logical output grid and 256-wide K slices.
- MatMul K chains use alternating accumulator BOs; every intermediate value
  stays device-resident. Residual inputs and exported outputs are kept live
  through all consumers and downloads.
- Multi-tile projection, generic FFN, branch, independent activation, scalar
  and per-channel broadcast were exercised on RK3588 at `M=N=K=300` (with a
  `300x200` second branch output).

## Numerical policy

- Declared FP16 path: staged/ordered FP16 writeback at each K-tile boundary and
  explicit FP16 stores at activation/gated/residual points.
- Acceptance is the per-plan `NumericContract` allclose policy
  `abs(got-declared) <= atol + rtol*abs(declared)`, normally `8e-3/8e-3` for
  `rk_f16_tiled`.
- An unquantized float64 mathematical reference is stored separately and is
  never the acceptance predicate.
- Strict FP32 fused/unspecified accumulation is **not supported**.
- The whole-graph SwiGLU specialisation remains experimental; the canonical
  graph defaults to the generic per-node path.

## Explicitly unsupported / future extension

- 3-D grids, dynamic shapes, state/recurrent or hybrid ops.
- Runtime tensor/dynamic/arbitrary broadcast; only compile-time rank-0 and
  `[N]` constants are accepted.
- Arbitrary masks, non-rectangular predicates, else branches, and unproved
  overlap.
- Device-produced MatMul weights.
- External arbitrary IOVA import.
- Multi-block TileLang extraction is supported only for the separately launched
  stage form above.  A single parallel grid whose blocks depend on each other's
  global writes is explicitly rejected; it is not reinterpreted as sequential
  stages.  Arbitrary block-grid tails in this staged source form are not accepted
  yet (tail semantics remain covered by the dedicated ragged/tiled fixtures).
- Multi-board/fleet qualification, concurrent multi-core throughput, performance
  advantages and long-duration stress are extension validation, not MVP/P8
  blockers.
- RK1228 is a separate target; no RK3588 recipe or acceptance conclusion is
  inherited. See `RK1228-TARGET.md`.

## Recovery boundary

Unknown submit/device errors are handled by quarantine and stop: the provider
context rejects further submissions, the workspace is poisoned, outputs are not
published and resources are retained for process-level disposal.  Driver
quiescence, context reinitialization, physical release and continued computation
after such a fault are **not implemented, not tested and not promised**.  This is
an explicit scope boundary, not a hidden MVP acceptance condition.

## CLI contract

`run` loads a bindings JSON whose values are `.npy` paths relative to the JSON
file (or absolute), verifies the plan's input/constant/output identities and
shapes, opens `/dev/dri` unless `--fd` is supplied, publishes outputs atomically
as `<output>.npy`, and returns:

- `0` success
- `2` schema/binding/numeric/unsupported error
- `3` resource/device/BUSY/POISONED error

If `run` opens the fd it closes it only after workspace destruction confirms
quiescence; with `--fd` the caller retains ownership.  The vendor
`void rocket_close(int)` has no status result, so the CLI does not claim confirmed
fd closure.  A workspace-destroy failure is written to stderr and returns a
resource error; when the workspace is poisoned/BUSY the provider fd and task BO
are retained for process-level recovery rather than closed.  A simultaneous
execution error remains the primary exit code and cleanup diagnostics are still
reported.

## Next-milestone model bridge

The separate complete-model milestone is documented in
[MODEL-MILESTONE-STATUS.md](MODEL-MILESTONE-STATUS.md).  It adds a learned
784→256→128→10 MLP with exact NPU ReLU and bias, and a 16-token/64-wide/4-head
one-layer image Transformer.  For the Transformer first round, patch
rasterization/position add, RMSNorm, stable softmax, head layout bridges,
SwiGLU elementwise operations, residual adds and mean pooling are explicit Host
work; all matrix products and the classifier run through real TileLang-derived
RKPlans and the production runtime.  This is a reported hybrid boundary, not an
implicit CPU matrix fallback.  Dynamic shape, arbitrary ONNX, full LLM/KV cache,
quantization, NPU training, multi-board/concurrent throughput and post-fault
recovery remain outside scope.
