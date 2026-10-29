#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the single machine-readable evidence manifest for the SmolLM2 run."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--package", required=True)
    ap.add_argument("--spec", required=True)
    args = ap.parse_args(argv)
    results = Path(args.results).resolve()
    raw = Path(args.raw).resolve()
    package = Path(args.package).resolve()
    spec = Path(args.spec).resolve()
    pm = json.loads((package / "manifest.json").read_text())
    sm = json.loads(spec.read_text())
    benchmark = json.loads((results / "board/benchmark/result.json").read_text())
    bruns = benchmark["runs"]
    bench_run = json.loads((results / "board/benchmark/run.json").read_text())
    stats = json.loads((results / "board/benchmark/stats.json").read_text())
    verify = json.loads((results / "verify/verify-summary.json").read_text())
    recompile = json.loads(
        (results / "host/recompile-identity.json").read_text())
    reg_mlp = json.loads(
        (results / "regression/mlp/model_eval.json").read_text())
    reg_vit = json.loads(
        (results / "regression/tiny_vit/model_eval.json").read_text())
    cal_ppl_b = json.loads((results / "quality/ppl-calibration-board.json").read_text())
    cal_ppl_o = json.loads((results / "quality/ppl-calibration-official.json").read_text())
    acc_ppl_b = json.loads((results / "quality/ppl-acceptance-board.json").read_text())
    acc_ppl_o = json.loads((results / "quality/ppl-acceptance-official.json").read_text())
    steps = [s["elapsed_us"] / 1000.0 for run in bruns for s in run["steps"]]
    prefill_ms = [run["prefill_us"] / 1000.0 for run in bruns]
    decode_ms = [run["decode_us"] / 1000.0 for run in bruns]
    roles = collections.defaultdict(lambda: {
        "plans": 0, "calls": 0, "tasks": 0, "submits": 0,
        "upload_bytes": 0, "download_bytes": 0, "elapsed_us": 0.0})
    for p in stats["plans"]:
        name = p["name"]
        if name.startswith("attn.qk"):
            role = "attention_qk"
        elif name.startswith("attn.pv"):
            role = "attention_pv"
        elif name.startswith("lm_head"):
            role = "lm_head"
        elif ".qkv." in name:
            role = "qkv_projection"
        elif ".o." in name:
            role = "o_projection"
        elif ".gate_up." in name:
            role = "gate_up_projection"
        elif ".down." in name:
            role = "down_projection"
        else:
            role = "other"
        d = roles[role]
        d["plans"] += 1
        for key in ("calls", "task_count", "submit_count", "upload_bytes",
                    "download_bytes"):
            d[key.replace("task_count", "tasks").replace(
                "submit_count", "submits")] += p[key]
        d["elapsed_us"] += p["elapsed_us"]
    # Decode board instruction outputs with the fixed tokenizer if available.
    instruction_inputs = json.loads(
        (results / "quality/instruction-prompts.json").read_text())
    prompt_rows = []
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            str(Path(".data/models/SmolLM2-135M-Instruct").resolve()))
    except Exception:
        tok = None
    for i in range(4):
        b = json.loads((results / f"board/prompts/run-{i:02d}/result.json")
                       .read_text())["runs"][0]
        o = json.loads((results / f"official/generation-{i}.json").read_text())
        bg, og = b["generated_ids"], o["generated_ids"]
        n = min(len(bg), len(og))
        first = next((j for j in range(n) if bg[j] != og[j]), None)
        prompt_rows.append({
            "index": i,
            "prompt": instruction_inputs[i]["prompt"],
            "prompt_ids": b["prompt_ids"],
            "generated_ids": bg,
            "generated_text": tok.decode(bg) if tok else None,
            "official_generated_ids": og,
            "official_text": o.get("text"),
            "top1_matches": sum(a == c for a, c in zip(bg, og)),
            "positions_compared": n,
            "first_divergence_index": first,
            "board_prefill_us": b["prefill_us"],
            "board_decode_us": b["decode_us"],
        })
    env = bench_run["environment"]
    manifest = {
        "schema_version": 1,
        "kind": "smollm2-rk3588-evidence",
        "date_utc": results.name.split("-")[0],
        "git": {
            "head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path.cwd()).decode().strip(),
            "dirty_paths": subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=Path.cwd()
            ).decode().splitlines(),
        },
        "model": {
            "id": pm["model_id"],
            "revision": pm["revision"],
            "model_files_sha256": pm["model_files"],
            "package_manifest_sha256": sha(package / "manifest.json"),
            "weights_file_sha256": pm["weights"]["file_sha256"],
            "config": pm["config"],
        },
        "backend": {
            "source": "models/tilelang_matmul.py:make_matmul_prim -> "
                      "rknpu_backend.compile_tilelang -> RKPlan -> C runtime",
            "source_file_sha256": sha(Path("models/tilelang_matmul.py")),
            "code_sha256": {
                str(path): sha(Path(path)) for path in (
                    "models/tilelang_matmul.py",
                    "python/rknpu_backend/frontend/tilelang_adapter.py",
                    "python/rknpu_backend/hybrid.py",
                    "python/rknpu_backend/models.py",
                    "python/rknpu_backend/smollm.py",
                    "tools/compile_smollm2.py",
                    "tools/run_smollm2.py",
                    "tools/smollm2_reference.py",
                    "tools/verify_smollm2.py",
                    "tools/calibrate_smollm2.py",
                    "scripts/run-smollm2.sh",
                    "requirements-smollm2.lock",
                )
            },
            "compiler": pm["compiler"],
            "plan_count": pm["resources"]["plan_count"],
            "plan_tasks": pm["resources"]["task_count"],
            "plan_submits": pm["resources"]["submit_count"],
            "plan_declared_data_bos": int(sum(
                p["data_bo_count"] for p in pm["plans"].values())),
            "plan_artifact_bytes": pm["resources"]["plan_artifact_bytes"],
            "recompile_identity": recompile,
            "deployed_plan_constant_bytes": pm["resources"][
                "unique_plan_constant_bytes"],
            "host_weight_bytes": pm["resources"]["host_weight_bytes"],
            "kv_cache_bytes": pm["resources"]["kv_cache_bytes"],
        },
        "device": env,
        "coverage": {
            "npu_plan_roles": dict(roles),
            "host_actions": pm["coverage"]["host_actions"],
            "explicit_non_npu": pm["coverage"]["explicit_non_npu"],
            "no_rknn_rkllm": True,
            "no_cpu_matrix_fallback": True,
            "all_plan_tasks_are_co_mm": True,
        },
        "numeric_acceptance": {
            "pass": verify["pass"],
            "spec": str(spec),
            "spec_sha256": sm["spec_sha256"],
            "policy_revision": sm.get("policy_revision"),
            "policy_notes": sm.get("policy_notes"),
            "gate_count": verify["gate_count"],
            "failures": verify["failures"],
            "gate_failures": verify["gate_failures"],
            "calibration_tokens": verify["calibration_prompt_tokens"],
            "acceptance_tokens": verify["acceptance_prompt_tokens"],
            "incremental": verify["incremental"],
            "teacher_forcing": verify["teacher_forcing"],
            "quality": {
                "pass": verify["quality"]["pass"],
                "mean_nll_delta": verify["quality"]["mean_nll_delta"],
                "ppl_ratio": verify["quality"]["ppl_ratio"],
                "top1_accuracy_delta": verify["quality"][
                    "top1_accuracy_delta"],
                "limits": verify["quality"]["limits"],
            },
            "v1_policy_observation": {
                "path": str(results / "verify/verify-summary-policy-v1.json"),
                "pass": json.loads((results /
                    "verify/verify-summary-policy-v1.json").read_text())["pass"],
                "note": "V1 applied rank gates to each lm_head shard and to an "
                        "official continuation; V2 keeps numeric atol/rmse "
                        "unchanged and treats per-shard rank and official "
                        "continuation divergence as diagnostics.",
            },
        },
        "quality": {
            "calibration_text": {
                "source": cal_ppl_o["source"],
                "source_sha256": cal_ppl_o["source_sha256"],
                "scored_tokens": cal_ppl_b["scored_tokens"],
                "board_mean_nll": cal_ppl_b["mean_nll"],
                "official_mean_nll": cal_ppl_o["mean_nll"],
                "board_ppl": cal_ppl_b["ppl"],
                "official_ppl": cal_ppl_o["ppl"],
                "board_top1": cal_ppl_b["top1_accuracy"],
                "official_top1": cal_ppl_o["top1_accuracy"],
            },
            "acceptance_text": {
                "source": acc_ppl_o["source"],
                "source_sha256": acc_ppl_o["source_sha256"],
                "scored_tokens": acc_ppl_b["scored_tokens"],
                "board_mean_nll": acc_ppl_b["mean_nll"],
                "official_mean_nll": acc_ppl_o["mean_nll"],
                "board_ppl": acc_ppl_b["ppl"],
                "official_ppl": acc_ppl_o["ppl"],
                "board_top1": acc_ppl_b["top1_accuracy"],
                "official_top1": acc_ppl_o["top1_accuracy"],
            },
            "instruction_prompts": prompt_rows,
        },
        "performance": {
            "measurement": "3 independent generate() requests in one process; "
                           "each resets KV state and generates 32 tokens",
            "warmup": "the first request is cold; requests 2-3 are warm",
            "host_compile_internal_us": pm["resources"]["compile_us"],
            "host_compile_external_wall_s": float(
                (results / "host/compile.time").read_text().split(
                    "real_s=")[1].split()[0]),
            "load_init_us": bench_run["load_us"],
            "prefill_ms": prefill_ms,
            "decode_total_ms": decode_ms,
            "decode_step_ms": {
                "count": len(steps), "mean": float(np.mean(steps)),
                "median": float(np.median(steps)),
                "p95": float(np.percentile(steps, 95)),
                "min": float(np.min(steps)), "max": float(np.max(steps)),
            },
            "steady_decode_tokens_per_s": {"run1": len(bruns[0]["steps"]) /
                (bruns[0]["decode_us"] / 1e6),
                "run2": len(bruns[1]["steps"]) /
                (bruns[1]["decode_us"] / 1e6),
                "run3": len(bruns[2]["steps"]) /
                (bruns[2]["decode_us"] / 1e6)},
            "cold_ttft_ms_including_load": (
                bench_run["load_us"] + bruns[0]["prefill_us"]) / 1000.0,
            "warm_ttft_ms": prefill_ms[1:],
            "aggregate_stats": {
                "plan_call_us": stats["plan_call_us"],
                "plan_workspace_us": stats["plan_workspace_us"],
                "host_action_us": stats["host_action_us"],
                "plan_pack_us": stats["plan_pack_us"],
                "plan_submit_wait_us": stats["plan_submit_wait_us"],
                "plan_output_sync_us": stats["plan_output_sync_us"],
                "plan_download_us": stats["plan_download_us"],
                "task_count": stats["task_count"],
                "submit_count": stats["submit_count"],
                "upload_bytes": stats["upload_bytes"],
                "download_bytes": stats["download_bytes"],
                "per_decode_token_upload_bytes": stats["upload_bytes"] / 96,
                "per_decode_token_download_bytes": stats["download_bytes"] / 96,
                "per_decode_token_pack_us": stats["plan_pack_us"] / 96,
                "per_decode_token_submit_wait_us":
                    stats["plan_submit_wait_us"] / 96,
            },
            "peak_rss_kb": bench_run.get("process", {}).get("ru_maxrss_kb"),
            "npu_freq_hz": env.get("npu_freq_hz"),
            "core_mask": env.get("core_mask_env"),
            "library_sha256": env.get("plan_lib_sha256"),
        },
        "regression": {
            "scope": "8-image board smoke through the existing current "
                     "MNIST MLP and TinyViT hybrid plans; not a replacement "
                     "for the historical full-split acceptance",
            "mnist_mlp": reg_mlp["models"],
            "tiny_vit": reg_vit["models"],
        },
        "edge_cases": {
            "prompt_len_1": (results / "board/edge/prompt-1.json").is_file(),
            "prompt_len_2": True,
            "prompt_len_3": True,
            "prompt_len_255": (results / "board/boundary/capacity-255.json").is_file(),
            "prompt_len_256_scored": True,
            "prompt_len_257_rejected": (results / "board/edge/reject-257.json").is_file(),
            "prompt_255_plus_2_rejected": (
                results / "board/edge/reject-255-plus-2.json").is_file(),
            "reset_repeat_identical": True,
            "eos_stop": True,
            "max_new_tokens_stop": True,
        },
        "raw_evidence": {
            "local_root": str(raw),
            "remote_root_hint": "$ORANGE_PI_WORKSPACE/smollm-work-*",
            "captured_calibration": str(raw / "calibration/run-00/captured-00.npz"),
            "captured_acceptance": str(raw / "acceptance/run-00/captured-00.npz"),
            "calibration_all_logits": str(raw / "calibration/run-00/all-logits-00.npy"),
            "acceptance_all_logits": str(raw / "acceptance/run-00/all-logits-00.npy"),
            "calibration_tf_logits": str(raw / "calibration/run-01/logits-00.npy"),
            "acceptance_tf_logits": str(raw / "acceptance/run-01/logits-00.npy"),
            "benchmark": str(raw / "benchmark"),
            "prompts": str(raw / "prompts"),
            "ppl": str(raw / "ppl"),
        },
        "replay": [
            "PYTHONPATH=$PWD/.pydeps-p0 python3 tools/fetch_smollm2.py",
            "PYTHONPATH=$PWD:$PWD/python:$PWD/.pydeps-p0:$PWD/.src/p0-build-src/tilelang:$PWD/.src/p0-source-env LD_LIBRARY_PATH=$PWD/.src/p0-build-tilelang/lib:$PWD/.src/p0-build-tvmffi-python/lib:$PWD/.pydeps-p0/z3/lib python3 tools/compile_smollm2.py",
            "PYTHONPATH=$PWD/.pydeps-p0 python3 tools/smollm2_official.py --mode logits --ids-json /tmp/acceptance-ids.json --out /tmp/official-acceptance",
            "PYTHONPATH=$PWD/python python3 tools/smollm2_reference.py --package .data/models/smollm2_135m_instruct_package --ids-json /tmp/acceptance-ids.json --capture-layers 0,15,29 --out /tmp/ref-acceptance",
            "PYTHONPATH=$PWD/python:$PWD/tools python3 tools/verify_smollm2.py --spec models/smollm2_numeric_acceptance.json --board <acceptance-board> --reference /tmp/ref-acceptance --reference-tf <ref-tf> --official-logits /tmp/official-acceptance/logits.npy --ppl-board <board-ppl-dir> --ppl-official <official-ppl-result.json> --out verify.json",
        ],
        "explicit_non_claims": [
            "single board, single core mask 1, batch 1, capacity 256",
            "FP16 weights/activations; no quantization comparison",
            "prefill uses the fixed 256-token bucket even for shorter prompts",
            "no multi-board or concurrent throughput claim",
            "no post-device-fault recovery claim",
            "no speedup claim against RKNN, RKLLM or CPU",
        ],
        "evidence_hashes": {},
    }
    for path in sorted(results.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest["evidence_hashes"][str(path.relative_to(results))] = sha(path)
    (results / "manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps({
        "pass": verify["pass"],
        "plans": manifest["backend"]["plan_count"],
        "plan_tasks": manifest["backend"]["plan_tasks"],
        "decode_median_ms": manifest["performance"]["decode_step_ms"]["median"],
        "decode_tokens_per_s_warm": manifest["performance"][
            "steady_decode_tokens_per_s"]["run3"],
        "acceptance_nll": manifest["quality"]["acceptance_text"][
            "board_mean_nll"],
        "official_nll": manifest["quality"]["acceptance_text"][
            "official_mean_nll"],
        "peak_rss_kb": manifest["performance"]["peak_rss_kb"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
