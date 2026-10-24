#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Reconstruct the large campaign weight/data payload from committed package files
# and replay the train-calibrated numeric verifier against committed observations.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)
NUMERIC=${1:-"$HERE/results/20261024T091634Z-numeric"}
MODEL_CAMPAIGN=${2:-"$HERE/results/20261024T071144Z-models"}
tmp=$(mktemp -d /tmp/rk-numeric-verify-XXXXXX)
trap 'rm -rf -- "$tmp"' EXIT
mkdir -p "$tmp/campaign"/{weights,data}
cp "$HERE"/models/mnist_mlp.npz "$HERE"/models/tiny_vit.npz \
   "$tmp/campaign/weights/"
cp "$HERE"/.data/mnist/t10k-images-idx3-ubyte.gz \
   "$HERE"/.data/mnist/t10k-labels-idx1-ubyte.gz "$tmp/campaign/data/"
cp "$MODEL_CAMPAIGN/campaign/manifest.json" "$tmp/campaign/manifest.json"
PYTHONPATH="$HERE/python:$HERE" python3 "$HERE/tools/verify_numeric_acceptance.py" \
  --spec "$HERE/models/numeric_acceptance_train.json" \
  --calibration-observation "$NUMERIC/calibration/observation.npz" \
  --acceptance-observation "$NUMERIC/acceptance/observation.npz" \
  --campaign-manifest "$tmp/campaign/manifest.json" \
  --campaign-results "$MODEL_CAMPAIGN/model-results" \
  --out "$tmp/verify-summary.json"
