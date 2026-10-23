# Handoff — per-node lowering for the MatMul→SiLU→proj+residual loop

_Updated 2026-10-23 after host closure, explicit ragged tails, and the first
RK3588 real-device admission campaign._

## Delivered in this pass

The no-K-loop path now returns a real multi-node SSA `Module` for the target
kernel; `_KBlock` and the K-loop path were not changed.

1. **Extraction (`tilelang_adapter._extract_static_dataflow`)**
   - Iterates every no-K-loop GEMM and runs the existing `_gemm_header`,
     full-region, offset, dense-layout and zero-fill checks per GEMM.
   - Associates fills, A/B copies and stores by **buffer identity + accessed
     region + statement-order def-use**.  It does not group by "one copy per
     GEMM" or by variable names.
   - Builds versioned tile definitions, so repeated stores to the same fragment
     (`p16 = acc2; p16 = p16 + X`) resolve to the correct earlier value.
   - Emits `matmul → silu → matmul → add` with the residual input edge
     `add(P, X)`, and records the five explicit source narrowing points:
     `gemm1_accumulator`, `sigmoid`, `silu_mul`, `gemm2_accumulator`,
     `residual_add`.
   - Refuses a two-GEMM branch, a wrong second-GEMM input, missing explicit FP16
     narrowing, extra stores/copies/fills and unsupported elementwise calls.

2. **Legalize (`passes/legalize._legalize_static_dataflow`)**
   - Multi-output graphs, graphs with more than one matmul, and graphs with more
     than three nodes go through per-node capability and `NumericContract`
     checks.  The experimental whole-graph SwiGLU matcher is only consulted when
     `allow_experimental=True`; otherwise canonical SwiGLU falls through to the
     same per-node legalizer.
   - Before any node is accepted it proves: at least one exported output, every
     op is reachable from an export, no non-exported temporary is unconsumed,
     role/producer combinations are consistent, and every output is produced.
     Exported intermediates may feed later nodes.
   - The old matmul/activation/binary path remains for the existing ≤3-op,
     K-loop and multi-tile cases.  The intermediate-state safety rule is now:
     **an extraction result is not planner input until legalize proves the whole
     graph connected**; do not add a bypass that sends a half-built SSA graph to
     `dag`.

3. **Lowering (`passes/dag._Builder.build_dataflow`)**
   - Maintains `value_id -> (device view, producer action)`.
   - Uploads each external binding once; weights with `b_layout="kn"` still use
     `transpose_source=True` into the custom-ops `[N,K]` weight layout.
   - Allocates one cube view per produced value and emits one submit group per
     primitive stage.  There is **no host round trip for Z/S/P**; the second GEMM
     consumes the activation's cube view directly.
   - Keeps the input X view alive through the final residual add.  The target
     graph has one download/publish; generic graphs may export several values,
     each with its own download/publish and extended lifetime.

4. **Additional graph-shape coverage (same pass)**
   - The generic static-dataflow lowerer now supports **multiple exported
     values**.  An exported intermediate may still feed a later node; each
     output gets its own download/publish rectangle and its buffer lifetime
     covers both the last device consumer and its own download.
   - `tests/test_multi_output_execution.py` covers two independent outputs and
     an exported pre-activation `Z` that is later consumed by SiLU while `P` is
     also exported.
   - The same per-node path lowers a reordered SwiGLU branch graph without using
     the whole-graph `co.swiglu_ffn.f16.v1` matcher.  `tests/test_ffn_execution.py`
     executes that graph through the mock and matches the independent numpy
     reference.  This proves the branch/shared-input/residual structure is not
     dependent on the experimental whole-graph optimisation.
   - Added `examples/silu_proj.py` and a CLI compile/validate/explain regression.
   - Added real TileLang multi-output branch extraction: two independent GEMMs
     share X and export two FP16 values.  The frontend emits a 2-node multi-output
     RKIR graph, and the plan produces two downloads/publishes
     (`tests/test_frontend_multi_output.py`, `tests/test_multi_output_execution.py`).
   - Added the explicit guarded rectangular-tail K loop (P1 batch 3): A/B tiles
     are zeroed on every K iteration, guarded affine loads cover only logical
     elements, and the output store is guarded to the logical M/N rectangle.
     The RKIR retains full logical shapes plus `source_tail_contract`; the
     existing planner generates M/N/K tails.  `ceildiv` alone is never accepted
     as a memory-safety proof.  Combined M=13/N=33/K=257 and multi-grid
     M=N=257/K=300 execute through the mock; missing zero-fill, off-by-one,
     wrong-axis and unguarded-output-store mutations are rejected.
   - Implemented the first P4 broadcast slice: explicit **constant** rank-0
     scalar and rank-1 `[N]` per-channel Add/Mul.  The source value is uploaded
     to an expanded `[M,N]` cube once at workspace creation; the existing EW
     emitter still sees two same-shape cube views.  Expansion bytes are charged
     to Host accounting, the constant upload counts once, and runtime tensor/
     dynamic/arbitrary broadcast remains rejected
     (`tests/test_frontend_broadcast.py`, `tests/test_broadcast_execution.py`).
   - Added the real three-GEMM TileLang FFN frontend for the explicit-FP16
     variant (`tests/fixtures/tilelang_ffn.py::swiglu_ffn_f16`).  It extracts the
     six-node branch graph and executes through the generic per-node path.
     `swiglu_ffn_fp32_residual` preserves the original FP32-accumulator residual
     and is a hard `RK_E_NUMERIC_CONTRACT` negative even when LUT/experimental/
     separate-submit options are enabled: this backend cannot fuse FP32
     projection+residual and then narrow once at the output.

5. **Closed-loop evidence**
   - `tests/test_silu_proj_execution.py`: real PrimFunc → RKIR → RKPlan → C ABI
     → mock NPU; independent numpy reference with all five FP16 roundings; same
     workspace and BOs run twice with different X; zero-weight case proves
     `Y == X` after both GEMMs, i.e. the residual input survived.
   - `tests/test_frontend_silu_proj.py`: SSA shape/edges and plan topology; the
     second GEMM's A view aliases the SiLU EW output buffer, and X's buffer
     lifetime extends through the add.
   - Adjacent negatives in `tests/fixtures/tilelang_silu_proj.py`:
     wrong branch (`z16` copied where `s16` is required), missing FP16 narrowing
     (FP32 SiLU fragment), unsupported `T.tanh` gate; plus dead-graph/contract
     negatives in `tests/test_dataflow.py`.
   - Both the canonical and reordered **RKIR** branch graphs use the generic
     per-node path.  The real three-GEMM TileLang FFN is also positive when the
     source materialises project16 before the residual add; the original
     FP32-residual variant stays a numeric-contract negative.

## Review closure

The four evidence/semantic review items from the prior round are closed in
[REVIEW-CLOSURE.md](REVIEW-CLOSURE.md): unquantized float64 math reference,
final-publication BO guard, verifier byte/hash semantics, and exact NUL-token /
driver target parsing.  The table links each item to code, regressions and
archived evidence, and retains the historical limitations.

## RK3588 real-device admission (2026-10-23)

Scope: **small shapes, three single-core masks run serially**.  This is real RK3588
backend execution evidence, not a general P8/release qualification and not a
three-core concurrent-throughput result.

The board is reached through the ignored repository `.env` using the existing
host/bundle/remote-results flow.  Credentials are never printed or copied into
artifacts; the campaign performs an exact-value credential scan and fails closed.
Because this Host has no usable PTY, the SSH path uses
`SSH_ASKPASS_REQUIRE=force` instead of `sshpass`, without changing the deployment
shape.

Evidence: `results/20261023T121507Z-rk3588/`

- Device: `RK3588 OPi 5 Plus`; kernel `6.1.43-rockchip-rk3588-rknpu098`
- Driver: `RKNPU driver: v0.9.8`; NPU 1 GHz; runtime ABI 1
- `profile_hash = 954d0713...e62eb692d`; every plan matches the generated manifest
- C runtime independently validates an exact NUL-token match in the RK3588
  device-tree compatible list and the exact locked driver string before
  workspace creation (`rk_target_check`); the host regression rejects
  `rockchip,rk3588-extra`, `v0.9.80` and prefix/suffix version lookalikes.
  This remains narrower than a full fleet/driver-variation qualification.
- Source archive: `source-bundle.tgz` + `source-bundle.sha256`
- Loaded production library hash: `board/library.sha256`, also repeated in the
  runner environment record

Result: **27/27 cases passed on each of masks 1, 2, 4 (81/81)**:

- MatMul chain; projection chain / separate; staged-FP16 FFN chain / separate
- explicit guarded ragged K/M/N chain / separate
- scalar and per-channel constant add/mul (chain or separate)
- independent SiLU activation and a 256x256 physical LUT split into two chunks
- real two-output branch and M/N multi-tile MatMul
- even and odd K shard landing (`K=512`, `K=257`)
- tiled per-node projection, generic FFN, branch and activation (`M=N=K=300`,
  branch second output `300x200`)
- **real multi-block TileLang staged projection/FFN/branch** (`512x256x512`),
  each stage a separate `T.Kernel` launch with materialised global intermediates;
  same-grid cross-block dependencies and FP32-only K accumulation are rejected
- NaN input guard, NaN constant guard (workspace-create failure; no output
  binding is executed), finite-input FP16 overflow -> non-finite output guard
- full board CLI path on every core mask: `compile-rkir`, `validate`,
  `explain`, two successful `run` calls with stable fd count/output publication,
  and a missing-constant binding returning rc2 without publishing; Host
  regressions cover caller-owned fd and destroy/close failure propagation

Numeric evidence:

- MatMul: bit-exact against the declared staged reference
- ragged: bit-exact against the declared staged reference; one real K/M/N tail
  shape only (`M=13,N=33,K=257`), not a claim over all 1..4096 shapes
- projection: declared-path `max_abs = 4.8828125e-4`
- FFN: declared-path `max_abs = 2.44140625e-4`
- multi-block TileLang staged projection / FFN / branch declared `max_abs`:
  `9.765625e-4` / `4.8828125e-4` / `2.44140625e-4`
- acceptance is `np.allclose(abs(got-declared) <= atol + rtol*abs(declared))`
  with `atol=rtol=8e-3`, sourced from the per-plan `rk_f16_tiled`
  `NumericContract.tolerance`; it is not an independent mathematical-FP64
  precision guarantee
- a second, **unquantized FP64 math** reference is retained as float64 per round
  for context; it is not narrowed to FP16 when stored/read and is not the
  acceptance predicate
- chain and separate hardware raw outputs are bit-identical for projection, FFN
  and ragged on all three masks

Guards:

- The runtime calls the existing `co_guard_check` helper for every allocated data
  BO at the **final publication gate**, after all submits/download staging and
  before any caller output bytes are committed.  A host mock regression includes
  a legal plan with task A -> download A -> task B, where task B corrupts a guard:
  publication is still refused, both outputs remain unchanged and the workspace
  is poisoned.
- NaN input and non-finite output rejections keep the pre-run output bytes
  unchanged.
- NaN constant is rejected during workspace creation before any output binding
  is executed; no "output unchanged" claim is made for that case.

Raw outputs, inputs and references are stored per round in the campaign.  The
verifier re-checks rounds, output sets/shapes/element counts, per-round task and
submit counts, declared-path numerics, raw chain/separate equality by
`(dtype, shape, bytes)`, driver, profile, runtime ABI, target check and credential
scan rather than trusting `status=PASS`.  It independently recomputes the source
bundle SHA-256; the remote loaded-library SHA-256 is labelled **recorded-only**
because the library bytes are not rehashed on the Host.

Offline recomputation of the archived raw outputs is recorded at
`results/20261023T121507Z-rk3588/recomputed/summary.md` (and for the prior archive
at `results/20261023T101551Z-rk3588/recomputed/summary.md`).  The recomputation
uses unquantized float64 math references; the latest declared-path and math-path
max errors match the campaign summary values above.

Multi-board coverage, concurrent three-core throughput, performance advantages
and long-duration stress are **extension validation**, not MVP/P8 blockers.

## MVP closure artifacts

- [SUPPORT-DOMAIN.md](SUPPORT-DOMAIN.md): accepted source forms, tiling,
  numerical policy and explicit non-support boundaries.
- [ACCEPTANCE-MATRIX.md](ACCEPTANCE-MATRIX.md): one-row-per-necessary-condition
  map to code, tests and archived logs, with implementation/verification/
  unsupported/future/blocked classifications.
- [REPLAY.md](REPLAY.md): Host, board, CLI, offline-recompute and P0 retry commands.
- [REVIEW-CLOSURE.md](REVIEW-CLOSURE.md): the four evidence-semantic review items.
- [RUNTIME-LIFECYCLE.md](RUNTIME-LIFECYCLE.md): command/task/data BO ownership,
  pre-submit safety, unknown-state poisoning/retention, BUSY and CLI cleanup.
- P0 bounded retries: `exit 124` (`provenance/p0-rebuild-attempt.log`) and
  `curl_rc=28` (`provenance/p0-rebuild-attempt-2.log`); P0 remains externally
  blocked and is not claimed complete.

## Final MVP status

The implemented and board-verified domain is complete for the finite representative
matrix: real TileLang multi-block staged projection/FFN/branch, tiled RKIR graphs,
CLI, guards, bindings/BO/budget/BUSY/pre-submit/unknown-poison host regressions and
same-board execution.

The entire normative MVP is **not claimed closed** because exactly two concrete
conditions remain: (1) P0 source-built `libtvm` is externally blocked; (2) a real
RK3588 submit/recovery fault was not injected, so driver quiescence and physical
release after unknown device state remain unmeasured.  These are recorded as
limitations/external blockers, not hidden behind an undefined release-gate label.

## Verification state

* `pytest tests` → **207 passed, 38 subtests**, 0 skipped.
* `ctest --test-dir research/custom-ops/build-host-mvp` → **26/26 passed**.
* `scripts/run-verification.sh --keep-build` → exit 0;
  `results/20261023T121654Z-host/`:
  - pytest 207 passed / 0 failed / 0 skipped;
  - ASan+UBSan contract/host checks pass;
  - evidence complete.  This host record opens no device; the RK3588 evidence is
    the separate campaign above.

## Resolved path decision (2026-10-23)

Approved behaviour: the canonical SwiGLU graph uses the **generic per-node path by
default**, just like any equivalent reordered graph.  `allow_experimental=True`
merely permits the optional whole-graph `swiglu_ffn_v1` specialisation; it is not
a gateway for compilation.  The specialisation still enforces its own capability,
numeric, layout and range checks, and `co.swiglu_ffn.f16.v1` remains
**experimental**.

Implemented and pinned by:

* `tests/test_swiglu.py`: canonical default selects `linear_k_acc_v1` /
  `gate_ew_v1` per-stage recipes, while `allow_experimental=True` selects
  `swiglu_ffn_v1`; strict / missing-LUT / >256-width rejections are checked on
  both paths; the capability still cannot be used through `profile.enabled()`.
* `tests/test_ffn_execution.py`: canonical default and gate/up-swapped graphs both
  execute via the generic path and match the independent numpy reference.
* The obsolete "experimental, no composed numeric evidence" rejection text was
  removed from `legalize`.

## Remaining boundaries (do not overclaim)

* The no-K-loop frontend supports the 2-GEMM projection chain, the 3-GEMM
  staged-FP16 SwiGLU branch, two independent matmuls exporting two values, a
  single GEMM followed by additive/multiplicative rank-0 or rank-1 constant
  broadcast, and the explicit guarded ragged K/M/N form.  Composite graphs are
  limited to one primitive tile per node; the ragged K form retains logical
  M/N/K and delegates tail generation to the existing planner.  An FP32
  projection+residual narrowed only by the final copy is a deliberate
  numeric-contract rejection; no implicit reordering is allowed.
* Broadcast is deliberately narrow: only compile-time constants, only scalar or
  `[N]` per-channel, only Add/Mul, and only a full rank-2 peer operand.  Runtime
  tensor broadcast, dynamic Host scalars, arbitrary NumPy broadcasting and
  native scalar EW templates remain unsupported.
* The classic `_detect_k_block` path is untouched; the new ragged K form is a
  separate guarded-tail extractor.  K reduction, M/N multi-tile and the
  `co.mm_acc` ping-pong remain on the old planner path.
* The generic graph lowerer supports matmul, the registered unary activations,
  and same-shape add/mul.  It is not a general arbitrary-DAG compiler.
* The RK3588 campaign above is real-device evidence for the listed small shapes
  and policies, not a general performance or production qualification.  The
  broader release policy in `IMPLEMENTATION-STATUS.md` still applies.
* RK1228 is a separate target.  Its identification, profile, runtime/driver ABI
  and memory contract must be captured from the real device; see
  [RK1228-TARGET.md](RK1228-TARGET.md).  No RK3588 register recipe or acceptance
  result transfers.

## Two invariants for the next change

1. **Never hand a partially connected SSA graph to the planner.**  Extraction
   may produce several ops, but `legalize` must first prove reachability,
   consumer closure and the exported-root set.  Keep the current dead-node and
   unconsumed-temporary rejects when extending the op set.
2. **Keep intermediates in device BOs.**  The second GEMM must consume the
   previous stage's cube view; do not add a download/upload bridge and do not let
   storage reuse X before the residual add.  The zero-weight `Y == X` test and
   the X lifetime assertion are the regression guards for this.
