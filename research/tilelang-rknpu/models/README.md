# Frozen model package

This directory contains the learned weights and source bodies for the two
NEXT-MILESTONE models:

- `mnist_mlp.npz`: all learned MLP weights and biases, safe NumPy arrays;
- `tiny_vit.npz`: all learned Tiny Transformer weights;
- `manifest.json`: preprocessing, source/weight/plan hashes, architecture,
  training provenance and acceptance thresholds;
- `tilelang_mlp.py`: real staged TileLang MLP PrimFunc generator;
- `tilelang_matmul.py`: real one-tile TileLang matrix PrimFunc generator;
- `compiled/`: offline RKPlan artifacts and their source/plan hash manifest.

The `.npz` files are loaded with `allow_pickle=False`; no Python callable or
pickled object is stored in a model or RKPlan.

## Current execution manifests

| manifest | role |
|---|---|
| `compiled/tiny_vit_bN.rkhybrid.json` | default: packed FFN, QK/PV shared group, mapped QKV input |
| `compiled/tiny_vit_bN_splitheads.rkhybrid.json` | materialised q/k/v control; use for B1/separate submission-policy comparisons |
| `compiled/tiny_vit_bN_deviceffn.rkhybrid.json` | opt-in device FFN with an independent numeric policy |
| `compiled/tiny_vit_bN_hostffn.rkhybrid.json` | historical unpacked FFN control |
| `compiled/tiny_vit_bN_rmsnorm.rkhybrid.json` | fused residual+RMSNorm control (no measured default gain) |
| `compiled/mnist_mlp_bN.rkhybrid.json` | static MLP profiles |

Default vs control attribution and final acceptance are in
[OPTIMIZATION-SUMMARY.md](../OPTIMIZATION-SUMMARY.md); replay commands are in
[REPLAY.md](../REPLAY.md).

<a id="smollm2-135m-instruct"></a>
## SmolLM2-135M-Instruct

The fixed external package is generated into the ignored local directory
`.data/models/smollm2_135m_instruct_package/`; it is not committed because the
canonical FP16 weight file is 269.1 MB.  The fixed model is
`HuggingFaceTB/SmolLM2-135M-Instruct` at revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`.

Files:

| path | role |
|---|---|
| `manifest.json` | model/file identities, verified config, plan/source hashes, resource counts, coverage and Host-action registry |
| `weights.npz` | canonical FP16 `embed`, 30 layers of combined QKV/O/gate-up/down weights and norm gammas |
| `plans/*.rkplan.json` | 268 real-TileLang-derived RKPlans: 240 layer projections, 24 lm_head shards and 4 dynamic-B QK/PV plans |

Verified model semantics: 30 layers, hidden 576, intermediate 1536, vocab
49152, 9 Q heads, 3 KV heads, head_dim 64, SiLU MLP, RMSNorm eps 1e-5,
RoPE theta 100000 with HF half-split rotation, tied word embeddings, EOS token
`<|im_end|>` id 2, BOS `<|im_start|>` id 1, and the tokenizer's embedded chat
template.

Execution boundary:

- every Q/K/V/O, gate/up/down, attention QK/PV and lm_head shard is an immutable
  RKPlan whose only NPU task templates are `co.mm.f16.v1` /
  `co.mm_acc.f16.v1`;
- embedding lookup, RMSNorm, RoPE, GQA layout/merge, stable causal softmax,
  in-place KV append, SwiGLU elementwise product, residual add, vocabulary
  concatenation and greedy sampling are explicitly registered and metered Host
  actions;
- prefill uses one 256-token bucket; decode uses M=1 plans and a persistent
  `[30,3,256,64]` FP16 Host KV cache; no RKNN/RKLLM or CPU matrix fallback is
  used.

Host-only tokenizer/reference versions are fixed in
`requirements-smollm2.lock`. Fetch, compile and replay commands are in
[REPLAY.md](../REPLAY.md).  The frozen
numeric acceptance policy is `models/smollm2_numeric_acceptance.json`; its
calibration and disjoint acceptance evidence is under
`results/20261028T094650Z-smollm2/`.

## Graph-fusion variants (2026-10-28)

The frozen default packages under `models/compiled/` are unchanged.  Explicit
graph-fusion experiments are built into separate directories:

- `models/compiled_graph_fusion/` (default capability profile):
  - `tiny_vit_bN_deviceffn_fused.rkhybrid.json` — same seven tasks and byte
    transfers as the device-FFN control, one dependency-ordered Submit;
  - `tiny_vit_bN_addfused.rkhybrid.json` — device residual Add after attention
    output and down projection, bitwise equal to the Host add path.
- `models/compiled_rmsnorm_lut/` (independent profile
  `rk3588_vendor_co_v1_rmsnorm_lut.json`):
  - `tiny_vit_bN_rmsnorm_approx.rkhybrid.json` — full-device square, row sum,
    scaled bounded-rsqrt LUT, gamma expansion and multiply.  This is an
    independent approximate numeric policy and is not a default replacement.

The b1/b8 paired measurements and capability matrix are in
[GRAPH-FUSION-REPORT.md](../GRAPH-FUSION-REPORT.md).


## Stage-2 experimental probes (2026-10-28)

- `rope_device_module` / `tools/evaluate_rope_device.py`: 4-task, one-Submit
  device RoPE reorder (`X @ P`, then three EW tasks).  Cos/Sin are still runtime
  inputs, so coefficient residency and Smol integration remain open.
- `pv_o_device_module` / `tools/evaluate_pv_o_device.py`: 4 PV head matmuls
  directly feed 4 O-slice matmuls and three EW adds; 11 tasks in one Submit.
- `models/rmsnorm_approx_acceptance.json`: frozen disjoint calibration/
  acceptance gates for the isolated approximate RMSNorm policy.  It is not a
  default-path tolerance.


## Stage-3 additions

- `models/rmsnorm_approx_acceptance.json` + `.json.sha256`: frozen calibration
  spec for the isolated RMSNorm policy; use `evaluate_rmsnorm_approx.py --mode
  verify` in a separate process for acceptance.
- `models/smollm2_device_rope_policy.json`: independent prefill/decode candidate
  numeric policy for the opt-in Smol device-RoPE integration.
- `rope_device_constant_module`: fixed-prefill Cos/Sin constants, reused across
  layers; decode still takes runtime coefficients.
- `pv_o_device_batches_module`: eight legal 11-task PV→O Submit groups in one
  device workspace for the b8 experiment.


## Stage-4 additions

- `models/smollm2_device_rope_policy_cont32.json`: 32-step continuous candidate
  policy, calibrated on prompt-0 and independently accepted on prompt-1.
- `rope_decode_indexed_module` / `tools/evaluate_smollm_rope_indexed.py`:
  resident CosW/SinW/P tables plus one-hot position gather; Host uploads only
  position metadata, not the coefficient table.


- `models/smollm2_device_rope_policy_cont32_indexed.json`: 32-step continuous
  policy for the position-indexed decode path (resident CosW/SinW/P tables plus
  one-hot device gather).
