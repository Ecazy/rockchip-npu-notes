# Final model delivery campaign

Date: 2026-10-26 (UTC).  RK3588 OPi 5 Plus, NPU core mask 1, Host CPU affinity
6, three interleaved rounds, full 10,000-image MNIST test split for all five
static profiles.

## Identity

- Baseline source: `42f870f` (the commit before the first optimisation package;
  tree `cf2b7f2d75940ddec51d283d318d90306ce49dac`).
- Current runtime: `60f4a82`; delivery harness: `8cb57d5` (campaign ran with
  harness `96cb0dc`; only the local analysis aggregation changed afterwards).
- Baseline compiled manifest:
  `039cfaddcffde7e5e8fbcf75a76fb393f3362c1110f1641f6204685634654249`.
- Current compiled manifest:
  `72f898e989240ad45a864d88596838a21e9c5e7d0f74feb416efef97c20ddadd`.
- Campaign manifest:
  `26e30f88868940248c7fa9e458d8d2f36f4f810efdf70ce9585e2771aeab32da`.
- Weights:
  - `mnist_mlp.npz`: `c7cdd120350625ddf39decfde71c94eaed58e60144230ff29dfe7e4d7690c98e`
  - `tiny_vit.npz`: `0190412619e2603853e52c308fc06caa90bf54f92c15caa0ad3ea4e27e37000f`
- Test data:
  - images: `8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6`
  - labels: `f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6`
- Packed numeric spec:
  `f49d876b724b07ea417472ea1ac0c2b271c904e2cd6c7081e9a4c87cb3ddbe66`.

## Result files

- `analysis.json` / `analysis.md`: independent reference recomputation,
  original accuracy/top-1 thresholds, packed task-logit tolerances, warm
  latency/throughput, creation cost and structural comparison.
- `verify-packed-summary.json`: fresh current-build packed calibration and
  acceptance observation replay; **PASS / 0 problems**.
- `model-results/rN-{baseline,current}/model_eval.json`: raw per-round reports,
  logits, references and layer traces.
- `model-results/numeric-current{,-calibration}/`: fresh packed numeric
  observation.

## Replay

From `research/tilelang-rknpu`:

```bash
RK_FINAL_LIMIT=10000 RK_FINAL_ROUNDS=3 ./scripts/run-final-delivery.sh
```

The script reads the ignored repository `.env`, rebuilds both source arms,
uploads one shared payload, interleaves full five-profile evaluations, captures
the current packed numeric observation, retrieves raw evidence, and runs
`tools/analyze_final_delivery.py` plus `tools/verify_numeric_acceptance.py`.
