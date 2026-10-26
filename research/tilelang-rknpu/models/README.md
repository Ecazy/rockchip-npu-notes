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
