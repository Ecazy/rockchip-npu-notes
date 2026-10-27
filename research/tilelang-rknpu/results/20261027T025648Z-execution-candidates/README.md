# Execution-plan candidate campaign

Date: 2026-10-27 (UTC).  Same RK3588 OPi 5 Plus, same model files, same
10,000-image MNIST test split, same static batch/tail schedule, NPU core mask 1,
CPU affinity 6, and single-thread BLAS.

Kernel `6.1.43-rockchip-rk3588-rknpu098`, driver `v0.9.8`, NPU frequency
1,000,000,000 Hz.  The full process is pinned to CPU 6; NPU submissions use
`RKNPU_CORE_MASK=1` / one device core.

## Arms

| arm | graph | execution | numeric policy |
|---|---|---|---|
| generic | `plans/current`, frozen `rkhybrid` | generic per-call Python bindings | packed per-logit; same arithmetic |
| prepared | same graph | stable Host slots + prebound ctypes/RKPlan I/O | packed per-logit; bit-identical |
| auto | same graph | static binding-fanout selector: `<32` generic, `>=32` prepared | packed per-logit; bit-identical |
| hostfused | opt-in mapped-QKV graph with residual+RMSNorm fused Host action | prepared | packed per-logit; bit-identical |
| vendor | RKNN-Toolkit2/Lite2 2.3.2, non-quantized FP16, one core | one runtime invocation per static batch | classification-preserving; ViT max logit error exceeds 0.0625 |
| CPU | NumPy FP32 model reference | same batch schedule | FP32 reference, separate numeric contract |

The selected `auto` path chooses **generic** for all MLP profiles (13 static
binding slots) and **prepared** for Tiny ViT b1 (42 slots) and b8 (210 slots).
The threshold is 32 and is a graph/layout predicate, not a model-name branch.

## Results

Warm latency is per static batch call.  It includes raw input preparation,
model execution, output materialisation and excludes argmax/reference time.
Ratios against the generic control use paired-round medians.  Generic, prepared
and auto have five rounds; hostfused three; vendor/CPU two.

CPU frequency was not locked.  The evaluator pinned CPU 6 and sampled
`/proc/cpuinfo` at min/mean/max 408/1578/2304 MHz.  NPU sysfs read 1.0 GHz and
was not changed by the campaign; one NPU core (`RKNPU_CORE_MASK=1`) and vendor
runtime core 0 were used.

| profile | generic ms | prepared ms | auto ms | hostfused ms | vendor ms | CPU ms |
|---|---:|---:|---:|---:|---:|---:|
| mnist_mlp b32 | 2.152 | 2.187 | 2.163 | 2.127 | 0.834 | 6.861 |
| mnist_mlp b8 | 1.271 | 1.368 | 1.304 | 1.330 | 0.527 | 1.751 |
| mnist_mlp b1 | 1.233 | 1.293 | 1.206 | 1.234 | 0.409 | 0.255 |
| tiny_vit b8 | 10.238 | 7.442 | 7.303 | 7.550 | 4.272 | 5.873 |
| tiny_vit b1 | 3.075 | 2.499 | 2.358 | 2.283 | 1.022 | 0.874 |

- Prepared vs generic on the two Tiny ViT profiles: **1.367x** (b8) and
  **1.230x** (b1), paired medians.  Auto executes that same prepared path.
- Auto vs generic on MLP is not an optimization claim: both execute the generic
  path and emit identical tasks/submits/transfers; observed MLP differences are
  run-to-run variance and were not used as a speedup claim.
- Five-profile process wall (includes reference/trace work, not a pure app
  interval): generic median **143.585 s**, prepared **134.245 s**, auto
  **131.936 s**, hostfused **132.433 s**, vendor **107.320 s**, CPU **86.216 s**.
- Accuracy/top-1 is unchanged for every profile: MLP **0.98310**, Tiny ViT
  **0.98210**, top-1 agreement **1.00000**.
- All custom-arm raw logits and layer traces are bit-identical to the generic
  control in every paired round.

Prepared residency is 505,664 bytes for ViT b8 and 63,208 bytes for ViT b1.
Setup is separate: auto b8 64.3 ms and b1 45.3 ms, versus generic 60.9/44.0 ms;
this one-time cost is not represented as a warm-latency gain.

## Numeric acceptance

`verify-auto-summary.json` is **PASS / 0 problems** against the unchanged frozen
`numeric_acceptance_packed.json`.  The production compiled manifest remains
`72f898e989240ad45a864d88596838a21e9c5e7d0f74feb416efef97c20ddadd`;
the auto overlay changes runtime binding/residency selection, not model plans or
weights.  The vendor path remains separately qualified: MLP max logit error
**0.03049** (within packed 0.0625), Tiny ViT b8/b1 **0.12277 / 0.10715**
(exceeds packed 0.0625), so it is not a drop-in packed per-logit equivalent.

## Cost model

`cost-model.md` and `cost-model-*.json` retain the generic profile ledgers,
the static binding-slot count, the prepared write-back estimate, paired campaign
validation and the unpriced device-QKV screen.  Tiny ViT b8 is predicted at
2.795 ms saving and measured at 2.796 ms saving in this campaign.

The device-QKV screen removes 49,152 bytes of QKV download plus 163,840 bytes of
QKV-attributable QK/PV upload at b8 and models only **0.377 ms**, excluding any
device CUBE/bank reorder.  It keeps all 8 submits and the Host Softmax boundary,
so it is not selected.

## Identity and replay

- Runtime library SHA-256:
  `878ef3fa6c81cb651e02783db2b6bb72829cb2b0941444eaf3bc755784051d85`
- Initial campaign source: `60bfd68`; auto overlay source: `f54a215`
  (`auto-overlay.sha256` identifies the actual runtime/tool bundle).
- Production plans hash: `72f898...`; campaign candidate plan hash:
  `6fc2f94...`; host-fused hybrid manifests are listed in
  `campaign/manifest.json`.
- Campaign manifest SHA-256:
  `2c26057216f72749e13bb46e1b2943b614382bb65abd26f2db7cce7b31789123`
- Analysis SHA-256:
  `40bb90e78003b05add5ad4ae5ffdfd6a86ef57758c955d02e8e6cfa7d3838d56`
  (`PASS / 0 problems`); Markdown:
  `fe19c87c25f12184fad0d6c995e49b0a1bade3efe5f6b174c5874a6e4dcca85c`.
- Cost model SHA-256:
  `04631a3afec4a992d63ebed0687863078c9d3eb608997338f981d983fa27c4bc`.
- Frozen numeric verifier SHA-256:
  `9cd83510b859570aaf2c3001f70fafb7a7981e26a8305b822e7f3c26fc5edfb4`.
- Raw logits: selected `model_eval.*.logits.npy` files; layer traces:
  `model_eval.layers.npz`; frozen observations: `numeric-auto*/observation.npz`.

Re-run full campaign:

```bash
cd research/tilelang-rknpu
RK_CANDIDATE_LIMIT=10000 RK_CANDIDATE_ROUNDS=3 \
RK_CANDIDATE_EXTERNAL_ROUNDS=2 ./scripts/run-execution-candidates.sh
```

Re-run analysis/cost model against this checkpoint:

```bash
latest=results/20261027T025648Z-execution-candidates
PYTHONPATH=python:. python3 tools/analyze_execution_candidates.py \
  --campaign "$latest/campaign" --results "$latest/model-results" \
  --arms generic prepared auto hostfused vendor cpu \
  --out "$latest/analysis.json" --markdown "$latest/analysis.md"
for spec in 'mnist_mlp 32' 'mnist_mlp 8' 'mnist_mlp 1' 'tiny_vit 8' 'tiny_vit 1'; do
  read -r model batch <<< "$spec"
  PYTHONPATH=python:. python3 tools/execution_cost_model.py \
    --profile "$latest/model-results/profiles/${model}_b${batch}.generic.json" \
    --plans-dir "$latest/campaign/plans/current" \
    --manifest "${model}_b${batch}" \
    --prepared-profile "$latest/model-results/profiles/${model}_b${batch}.prepared.json" \
    --campaign-analysis "$latest/analysis.json" \
    --profile-key "${model}.b${batch}" \
    --out "$latest/cost-model-${model}_b${batch}.json"
done
PYTHONPATH=python:. python3 tools/summarize_execution_cost.py \
  --results "$latest" --out "$latest/cost-model.md"
PYTHONPATH=python:. python3 tools/verify_numeric_acceptance.py \
  --spec "$latest/campaign/numeric/numeric_acceptance_packed.json" \
  --calibration-observation "$latest/model-results/numeric-auto-calibration/observation.npz" \
  --acceptance-observation "$latest/model-results/numeric-auto/observation.npz" \
  --out "$latest/verify-auto-summary.json"
```
