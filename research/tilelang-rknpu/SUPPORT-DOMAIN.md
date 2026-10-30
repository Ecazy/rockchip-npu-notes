# Supported domain and explicit boundaries

Updated 2026-10-29 for the MVP, complete-model, SmolLM2 v5 and graph-fusion
checkpoints.

## Accepted source forms

- **TileLang classic paths**: canonical blocked MatMul (including K loop), M/N
  multi-tile single-GEMM, projection, staged-FP16 FFN, scalar/per-channel
  compile-time broadcast, real two-output branches, and the explicit guarded
  ragged K/M/N tail fixture.
- **Full logical single-GEMM TileLang source**: a one-launch PrimFunc whose
  `T.copy` regions prove complete A/B/C coverage and whose one `T.gemm`
  describes a logical M/N/K in 1..4096. The source result is a logical matmul;
  the target planner lowers it into 256x256 output tiles and 256-wide K slices.
  This is the form used by every SmolLM2 projection and lm_head shard.
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

- M, N and K are static integers in **1..4096** for the logical operation.
- Primitive extents are at most 256. Tiled per-node graphs use a 256x256
  logical output grid and 256-wide K slices. A full logical GEMM larger than 256
  is accepted from the closed single-GEMM source form only; arbitrary large
  source tiles or loop nests are not reinterpreted.
- External runtime operand B is supported for the dynamic-weight matmul plans
  (`b_is_input=True`). Device-produced B remains unsupported; SmolLM2 attention
  therefore keeps the append-only KV cache on the Host and passes a copied
  window into QK/PV RKPlans.
- MatMul K chains use alternating accumulator BOs; every intermediate value
  stays device-resident. Residual inputs and exported outputs are kept live
  through all consumers and downloads.
- Multi-tile projection, generic FFN, branch, independent activation, scalar
  and per-channel broadcast were exercised on RK3588 at `M=N=K=300` (with a
  `300x200` second branch output).

## Numerical policy

- Declared FP16 path: staged/ordered FP16 writeback at each K-tile boundary and
  explicit FP16 stores at activation/gated/residual points.
- Per-plan/operator acceptance remains the declared `NumericContract` allclose
  policy.  The complete-model milestone adds a separate frozen layer/task gate:
  deterministic traces, 33 per-layer and 5 per-profile absolute limits
  (`rtol=0`).  Limits are frozen from a 32-image MNIST **train** calibration
  subset only, using
  `max(2*observed_error, 2*FP16-spacing-at-train-reference-scale)` for device
  outputs and `2*observed_error` for exact Host layouts, rounded to 2^-12.
  Labels and test maxima are not used.  A disjoint fixed test subset is then
  observed and verified; a prior test-calibrated envelope is retained only as
  regression evidence.  This is an empirical RK3588 envelope, not a theoretical
  error guarantee.
- An unquantized float64 mathematical reference is stored separately and is
  never the acceptance predicate.
- Strict FP32 fused/unspecified accumulation is **not supported**.
- The whole-graph SwiGLU specialisation remains experimental; the canonical
  graph defaults to the generic per-node path.

## Explicitly unsupported / future extension

- 3-D grids, dynamic shapes, state/recurrent graphs, and arbitrary hybrid IR.
  The frozen complete-model package supports a finite versioned hybrid form:
  an `rkhybrid` manifest may call registered Host actions and immutable RKPlans;
  unknown actions, shapes, plans and fallbacks are rejected.
- Runtime tensor/dynamic/arbitrary broadcast; only compile-time rank-0 and
  `[N]` constants are accepted.
- Arbitrary masks, non-rectangular predicates, else branches, and unproved
  overlap.
- Device-produced MatMul weights. Runtime Host-provided B is supported; a
  MatMul output cannot be consumed as another MatMul's weight.
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

The complete-model milestone is documented in
[MODEL-MILESTONE-STATUS.md](ACCEPTANCE.md#models).  It adds a learned
784→256→128→10 MLP with exact NPU ReLU and bias, and a 16-token/64-wide/4-head
one-layer image Transformer.  A bounded versioned `rkhybrid` manifest records
shape/axes, dependencies, numeric contracts and bridge accounting.  Transformer
patch rasterization/position add, RMSNorm, stable softmax, head layout bridges,
SwiGLU elementwise operations, residual adds and mean pooling are named registered
Host actions; all matrix products and the classifier execute through immutable
real-TileLang-derived RKPlans.  Unknown Host actions, RKPlans, shapes, numeric
contracts or module fallback are rejected.  Dynamic shape, arbitrary ONNX, full
LLM/KV cache, quantization, NPU training, multi-board/concurrent throughput and
post-fault recovery remain outside scope.

## SmolLM2-135M-Instruct execution domain

The fixed model revision `12fd25f77366fa6b3b4b768ec3050bf629380bac` adds a
single-model LLM domain, not a generic LLM importer:

- batch 1, total prompt-plus-generated capacity 256 tokens, one fixed prefill
  bucket of 256 and M=1 decode plans;
- persistent Host KV cache `[30,3,256,64]` FP16, with an explicitly registered
  in-place append action and a runtime valid length;
- 30 layers x {QKV combined projection, O projection, gate/up combined
  projection, down projection} in both decode and prefill profiles;
- attention QK/PV plans with runtime operand B, three query heads per KV group;
- twelve 4096-wide lm_head shards covering the full 49152 vocabulary;
- embedding lookup, RMSNorm, RoPE, GQA layout/merge, stable causal softmax,
  SwiGLU elementwise product, residual add, vocabulary concatenation and greedy
  sampling are named, metered Host actions;
- unknown Host actions, non-C-contiguous or wrong-shape bindings, plans without
  only `co.mm*` NPU tasks, and requests beyond capacity fail closed;
- the fixed package has 268 immutable RKPlans, 5660 NPU tasks and 1712 submits,
  with about 537.9 MB of deployed plan constants (decode and prefill copies) and
  a 269.1 MB canonical weight file.

The current audited complete acceptance is policy v5
(`models/smollm2_numeric_acceptance_v5.json`): 6 calibration prompts, 8 disjoint
acceptance prompts, 5x256-token disjoint text windows, sealed calibration and
acceptance observation bundles, 56 numeric gates and a complete quality gate.
The current persistent KV path is native-layout Host orchestration plus per-layer
device QK/PV workspaces; it is not a generic device-resident KV API. The verifier
is fail-closed and rejects missing gates, shape broadcasting, non-finite
observations/thresholds, identity mismatches and incomplete quality evidence.

This closes only the named SmolLM2 geometry. It does not establish arbitrary
GQA/MQA/MLA geometry, dynamic prefill lengths, beam search, sampling policies,
quantization, multi-request batching, device-resident KV cache, multi-board
qualification, post-fault recovery, or RK1228 applicability.

## Graph-fusion experimental domain (2026-10-28)

- Dependency-ordered mixed-template chains are accepted only when the planner
  proves action-id contiguity and dependency order and the C verifier
  re-proves it.  The current device-evidence domain is the seven-task staged
  FP16 SwiGLU FFN (gate/up MM, sigmoid LUT, EW multiply, down MM, residual EW
  add) and the two-task MM + EW_ADD residual chain.
- `co.lut.rsqrt.f16.v1` is not in the default capability profile.  The isolated
  profile `capabilities/rk3588_vendor_co_v1_rmsnorm_lut.json` admits the
  approximate RMSNorm composition.  Its declared table domain is
  `[0.125, 6.32]`; with `lut_scale=0.75` the corresponding original mean-square
  domain is `[0.1667, 8.43]`, and the calibrated Tiny ViT subrange is
  `[0.17, 6.10]`.  The manifest records the scale, gamma folding and every FP16
  narrowing point.  `models/rmsnorm_approx_acceptance.json` freezes a disjoint
  calibration/acceptance operator gate: acceptance max-abs `0.0234375` and
  RMSE `0.0012227` against gates `0.0390625` / `0.009765625`, PASS.  This gate
  belongs only to the experimental policy and does not relax default RMSNorm.
- RoPE remains a mixed Host/NPU prototype in `custom-ops/src/co_rope.c`; it is
  not part of the RKIR default path and does not count as a removed Host
  boundary.

## Stage-3 device-layout and RoPE constraints

- CUBE 16×16 and QK B/WEIGHT 16×16 do not share a legal view.  The required
  mapping is `dst_weight[n*32+k] = src_cube[(k//8)*128+n*8+k%8]`; the existing
  fixed p0312 permutation mismatches 255 of 256 logical entries.  QKV→QK remains
  a Host-pack boundary until a real `co.reorder.cube_to_weight.f16.v1` device
  command or an equivalent variable B view exists.
- Smol device RoPE has an opt-in model path.  Prefill Cos/Sin/P are plan
  constants uploaded once and reused across 30 layers; decode Cos/Sin are runtime
  inputs because there is no dynamic position-indexed device view.  The candidate
  policy is `models/smollm2_device_rope_policy.json`; it passes an independent
  prompt acceptance but is not a default-path tolerance.
- PV→O b8 can execute as eight legal 11-task Submit groups (88 tasks total) in
  one workspace.  This proves the 64-task single-chain cap is not a blocker, but
  the current operator shape is slower than the merged Host reference and is not
  a model-level default.
