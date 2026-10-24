#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Add reproducible package/plan metadata to models/manifest.json."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def main() -> int:
    path = ROOT / 'models' / 'manifest.json'
    d = json.loads(path.read_text())
    compiled = json.loads((ROOT / 'models' / 'compiled' / 'manifest.json').read_text())
    d['milestone'] = 'NEXT-MILESTONE two complete models'
    d['package'] = {
        'format': 'npz_no_pickle_safe_arrays',
        'logical_weight_dtype': 'float32',
        'npu_weight_dtype': 'float16',
        'input_preprocessing': 'MNIST mean=0.1307 std=0.3081; 784 values zero-padded to 1024 for ordered-K NPU primitive; TinyViT uses 16 raster 7x7 patches',
        'output_postprocessing': 'argmax on first 10 logits; no threshold tuning',
        'weights_are_learned': True,
        'random_operator_demo': False,
    }
    d['model_sources'] = {
        'tools/train_mnist_models.py': sha(ROOT/'tools/train_mnist_models.py'),
        'models/tilelang_mlp.py': sha(ROOT/'models/tilelang_mlp.py'),
        'models/tilelang_matmul.py': sha(ROOT/'models/tilelang_matmul.py'),
        'python/rknpu_backend/models.py': sha(ROOT/'python/rknpu_backend/models.py'),
        'python/rknpu_backend/model_reference.py': sha(ROOT/'python/rknpu_backend/model_reference.py'),
        'models/compiled/manifest.json': sha(ROOT/'models/compiled/manifest.json'),
    }
    d['npu_plans'] = {
        'profile_hash': compiled['profile_hash'],
        'manifest': 'models/compiled/manifest.json',
        'plans': {name: {**info, 'sha256': sha(ROOT/'models/compiled'/info['path'])}
                  for name, info in compiled['plans'].items()},
    }
    d['host_boundaries'] = {
        'mnist_mlp': ['input normalization','zero padding to 1024','argmax'],
        'tiny_vit': ['patch rasterization','position add','pre-RMSNorm','stable softmax',
                     'head layout bridge','SwiGLU elementwise','residual adds',
                     'mean pooling','argmax'],
        'npu': {'mnist_mlp': ['all Linear','all bias','all ReLU'],
                'tiny_vit': ['patch projection','QKV','QK','PV','attention output projection',
                             'FFN gate/up/down','classifier']},
    }
    d['acceptance'] = {
        'full_test_images': 10000,
        'mnist_mlp_accuracy_min': 0.97,
        'tiny_vit_accuracy_min': 0.95,
        'accuracy_drop_max': 0.005,
        'top1_agreement_min': 0.99,
        'batch_profiles': {'mnist_mlp': [1,8,32], 'tiny_vit': [1,8]},
        'evidence': {'host_mock':'evidence/model_eval_mock.json',
                     'cpu_baseline':'evidence/cpu_benchmark.json',
                     'board_campaign':'results/20261024T031249Z-models/'},
    }
    d['numeric_policy'] = {
        'matmul':'ordered 256-wide K tiles with explicit FP16 writeback',
        'relu':'exact NPU elementwise MAX against zero, no LUT approximation',
        'activation':'explicit Host SiLU/SwiGLU first-round bridge',
        'softmax':'stable Max/Exp/Sum Host computation on FP32 scores',
        'fallback':'error; no implicit CPU matrix fallback',
    }
    path.write_text(json.dumps(d, indent=2, sort_keys=True) + '\n')
    print(path)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
