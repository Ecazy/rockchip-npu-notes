#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Reconstruct the immutable campaign payload from committed package files and run
# the independent verifier against the committed raw board logits/layer traces.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/.." && pwd)
RESULT=${1:-"$HERE/results/20261024T071144Z-models"}
[[ -f "$RESULT/campaign/manifest.json" ]] || {
  echo "missing campaign manifest: $RESULT/campaign/manifest.json" >&2; exit 2; }
tmp=$(mktemp -d /tmp/rk-final-campaign-XXXXXX)
trap 'rm -rf -- "$tmp"' EXIT
mkdir -p "$tmp/campaign"/{plans,weights,data,numeric}
cp "$HERE"/models/compiled/*.rkplan.json "$tmp/campaign/plans/"
cp "$HERE"/models/compiled/*.rkhybrid.json "$tmp/campaign/plans/"
cp "$HERE"/models/compiled/manifest.json "$tmp/campaign/plans/manifest.json"
cp "$HERE"/models/mnist_mlp.npz "$HERE"/models/tiny_vit.npz \
   "$HERE"/models/numeric_golden.npz "$tmp/campaign/weights/"
cp "$HERE"/models/numeric_acceptance.json "$HERE"/models/numeric_golden.npz \
   "$tmp/campaign/numeric/"
if [[ -f "$HERE/.data/mnist/t10k-images-idx3-ubyte.gz" ]]; then
  cp "$HERE"/.data/mnist/t10k-images-idx3-ubyte.gz \
     "$HERE"/.data/mnist/t10k-labels-idx1-ubyte.gz "$tmp/campaign/data/"
else
  cp "$RESULT"/campaign/data/t10k-images-idx3-ubyte.gz \
     "$RESULT"/campaign/data/t10k-labels-idx1-ubyte.gz "$tmp/campaign/data/"
fi
cp "$RESULT/campaign/manifest.json" "$tmp/campaign/manifest.json"
PYTHONPATH="$HERE/python" python3 "$HERE/tools/verify_model_results.py" \
  --manifest "$tmp/campaign/manifest.json" --results "$RESULT/model-results" \
  --out "$tmp/verify-summary.json"
