#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Add final reproducible model/plan/hybrid/numeric metadata to the package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def latest(prefix: str) -> str | None:
    rows = sorted((ROOT / "results").glob(f"*{prefix}"))
    if not rows:
        return None
    return rows[-1].relative_to(ROOT).as_posix() + "/"


def main() -> int:
    path = ROOT / "models" / "manifest.json"
    d = json.loads(path.read_text())
    compiled_path = ROOT / "models" / "compiled" / "manifest.json"
    compiled = json.loads(compiled_path.read_text())
    numeric_spec_path = ROOT / "models" / "numeric_acceptance.json"
    numeric_golden_path = ROOT / "models" / "numeric_golden.npz"
    numeric_spec = json.loads(numeric_spec_path.read_text())
    board_campaign = latest("-models")
    host_campaign = latest("-host")
    d["milestone"] = "NEXT-MILESTONE two complete models"
    d["package"] = {
        "format": "npz_no_pickle_safe_arrays",
        "logical_weight_dtype": "float32",
        "npu_weight_dtype": "float16",
        "input_preprocessing": "MNIST mean=0.1307 std=0.3081; 784 values "
                               "zero-padded to 1024 for ordered-K NPU primitive; "
                               "TinyViT uses 16 raster 7x7 patches",
        "output_postprocessing": "argmax on first 10 logits; no threshold tuning",
        "weights_are_learned": True,
        "random_operator_demo": False,
        "execution_manifest": "versioned rkhybrid; registered Host actions and "
                              "immutable RKPlan references only",
        "numeric_acceptance": "deterministic 8-image layer trace + real-TileLang "
                              "exact-ReLU probe + frozen per-layer/profile limits",
    }
    source_files = [
        "tools/train_mnist_models.py",
        "tools/compile_models.py",
        "tools/compile_numeric_probe.py",
        "tools/build_hybrid_manifests.py",
        "tools/build_numeric_acceptance.py",
        "tools/calibrate_numeric_acceptance.py",
        "tools/evaluate_models.py",
        "tools/benchmark_cpu_models.py",
        "tools/verify_model_results.py",
        "models/tilelang_mlp.py",
        "models/tilelang_matmul.py",
        "python/rknpu_backend/models.py",
        "python/rknpu_backend/model_reference.py",
        "python/rknpu_backend/hybrid.py",
        "python/rknpu_backend/numeric.py",
        "python/rknpu_backend/hostinfo.py",
        "models/compiled/manifest.json",
        "models/numeric_acceptance.json",
        "models/numeric_golden.npz",
    ]
    d["model_sources"] = {rel: sha(ROOT / rel) for rel in source_files}
    d["npu_plans"] = {
        "profile_hash": compiled["profile_hash"],
        "manifest": "models/compiled/manifest.json",
        "plans": {name: {**info,
                         "sha256": sha(ROOT / "models" / "compiled" /
                                       info["path"])}
                  for name, info in compiled["plans"].items()},
        "hybrid_manifests": {
            name: {**info,
                   "sha256": sha(ROOT / "models" / "compiled" /
                                 info["path"])}
            for name, info in compiled["hybrid_manifests"].items()
        },
    }
    d["hybrid_execution"] = {
        "schema_version": compiled.get("hybrid_schema_version"),
        "kind": "rkhybrid",
        "policy": compiled.get("hybrid_policy"),
        "host_actions": "rknpu_backend.hybrid.HOST_ACTIONS",
        "host_actions_version": 1,
        "node_counts": {
            "mnist_mlp_b8": {"total": 4, "host": 3, "rkplan": 1},
            "tiny_vit_b8": {"total": 23, "host": 14, "rkplan": 9},
        },
        "plan_reuse": "QK and PV use one immutable 16x16 dynamic-B RKPlan and "
                      "one workspace per runner",
        "no_opaque_model_node": True,
        "fallback": "error",
    }
    d["timing_contract"] = {
        "schema_version": 2,
        "excluded_from_model_interval": [
            "weight load", "runner/constant setup", "warm-up",
            "CPU FP32 reference", "declared FP16 reference",
        ],
        "included_in_model_interval": [
            "preprocessing", "all registered Host action nodes",
            "all RKPlan node calls and their binding overhead",
            "postprocessing/logits export", "argmax",
        ],
        "phase_names": ["preprocess", "execution", "postprocess"],
        "throughput_basis": "n / end_to_end_seconds; references excluded",
        "cpu_baseline": "matched static batch/tail schedule; explicit single-"
                        "thread BLAS env recorded",
        "invalid_history": "results/20261024T031249Z-models/"
                           "PERFORMANCE-RETRACTION.md",
    }
    d["layer_numeric_acceptance"] = {
        "spec": "models/numeric_acceptance.json",
        "spec_sha256": sha(numeric_spec_path),
        "golden": "models/numeric_golden.npz",
        "golden_sha256": sha(numeric_golden_path),
        "status": numeric_spec["status"],
        "calibration_status": numeric_spec["calibration"]["status"],
        "sample_indices": numeric_spec["sample_policy"]["indices"],
        "layer_tolerance_count": len(numeric_spec["layer_tolerances"]),
        "profile_logit_tolerance_count": len(
            numeric_spec["task_logit_tolerances"]),
        "tolerance_rule": numeric_spec["calibration"]["rule"],
        "classification_used_for_tolerance": numeric_spec["calibration"][
            "uses_classification_metric"],
        "relu_probe_plan": "models/compiled/mnist_relu_probe.rkplan.json",
        "relu_probe_templates": ["co.ew_add.f16.v1", "co.ew_max.f16.v1"],
        "calibration_evidence": "evidence/numeric-calibration/",
    }
    d["host_boundaries"] = {
        "mnist_mlp": [
            "mnist.normalize_flatten.v1",
            "tensor.pad_2d.v1",
            "logits.prefix.v1",
            "argmax",
        ],
        "tiny_vit": [
            "mnist.patchify_norm.v1", "tensor.pad_2d.v1",
            "tiny_vit.position_add.v1", "rmsnorm.last_axis.v1",
            "tiny_vit.qkv_split_heads.v1",
            "attention.scaled_softmax.v1", "tiny_vit.merge_heads.v1",
            "tensor.add.v1", "swiglu.gate_up.v1", "tensor.reshape.v1",
            "tensor.mean_axis.v1", "logits.prefix.v1", "argmax",
        ],
        "npu": {
            "mnist_mlp": ["all Linear", "all bias", "all exact ReLU"],
            "tiny_vit": ["patch projection", "QKV", "QK", "PV",
                         "attention output projection", "FFN gate/up/down",
                         "classifier"],
        },
        "npu_fallback": "forbidden",
        "host_fallback": "forbidden",
    }
    evidence_hashes = {}
    if board_campaign:
        base = ROOT / board_campaign
        for label, rel in (
                ("board_campaign_manifest",
                 "campaign/manifest.json"),
                ("board_verify_summary",
                 "model-results/verify-summary.json"),
                ("board_raw_core1_model_eval",
                 "model-results/core-1/model_eval.json"),
                ("board_layer_trace_core1",
                 "model-results/core-1/model_eval.layers.npz"),
                ("board_cpu_benchmark",
                 "model-results/core-1/cpu_benchmark.json")):
            evidence_path = base / rel
            if evidence_path.is_file():
                evidence_hashes[label] = sha(evidence_path)
    if host_campaign:
        host_manifest = ROOT / host_campaign / "manifest.json"
        if host_manifest.is_file():
            evidence_hashes["host_verification_manifest"] = sha(host_manifest)
    evidence_hashes["host_mock_eval"] = sha(
        ROOT / "evidence" / "model_eval_mock_v2.json")
    evidence_hashes["host_cpu_benchmark"] = sha(
        ROOT / "evidence" / "cpu_benchmark_v2.json")
    d["acceptance"] = {
        "full_test_images": 10000,
        "mnist_mlp_accuracy_min": 0.97,
        "tiny_vit_accuracy_min": 0.95,
        "accuracy_drop_max": 0.005,
        "top1_agreement_min": 0.99,
        "batch_profiles": {"mnist_mlp": [1, 8, 32], "tiny_vit": [1, 8]},
        "evidence": {
            "host_mock": "evidence/model_eval_mock_v2.json",
            "cpu_baseline": "evidence/cpu_benchmark_v2.json",
            "board_campaign": board_campaign,
            "host_verification": host_campaign,
            "numeric_calibration": "evidence/numeric-calibration/",
        },
        "evidence_sha256": evidence_hashes,
    }
    d["numeric_policy"] = {
        "matmul": "ordered 256-wide K tiles with explicit FP16 writeback",
        "relu": "exact NPU elementwise MAX against zero; no LUT approximation",
        "activation": "explicit registered Host SiLU/SwiGLU nodes",
        "softmax": "stable Max/Exp/Sum Host node on FP32 scores",
        "fallback": "error; no implicit CPU matrix fallback",
        "layer_gate": "frozen per-layer and per-profile absolute tolerances; "
                      "top-1 cannot replace the raw-logit/layer checks",
    }
    path.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
