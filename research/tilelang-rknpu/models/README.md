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
