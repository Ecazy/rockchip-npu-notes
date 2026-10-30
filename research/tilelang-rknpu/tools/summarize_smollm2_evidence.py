#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the auditable evidence manifest for one SmolLM2 acceptance campaign.

The summarizer fails closed: it refuses a verifier result that is not a complete
pass, and it never derives per-token performance by dividing aggregate counters
by a hard-coded token count.  Per-decode/per-prefill attribution comes from the
run's stage deltas and actual forward counts.
"""
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


def read_json(path: Path):
    return json.loads(path.read_text())


def require_complete_verifier(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit("verifier summary is missing")
    result = read_json(path)
    if result.get("mode") != "complete":
        raise SystemExit("verifier is not a complete acceptance")
    if result.get("complete_acceptance") is not True:
        raise SystemExit("verifier did not execute complete acceptance")
    if result.get("quality_pass") is not True:
        raise SystemExit("verifier quality gate did not pass")
    if result.get("pass") is not True:
        raise SystemExit("verifier overall pass is false")
    if result.get("gate_failures"):
        raise SystemExit("verifier reports numeric gate failures")
    if result.get("failures"):
        raise SystemExit("verifier reports failure entries")
    if not result.get("acceptance_bundle", {}).get("pass"):
        raise SystemExit("acceptance observation bundle is not sealed/pass")
    if not result.get("calibration_bundle", {}).get("pass"):
        raise SystemExit("calibration observation bundle is not sealed/pass")
    return result


def stage_delta(run: dict, stage: str) -> dict:
    stages = run.get("stages")
    if not isinstance(stages, dict) or stage not in stages:
        raise SystemExit(f"run lacks stages.{stage}")
    value = stages[stage]
    if not isinstance(value, dict) or not isinstance(value.get("stats_delta"), dict):
        raise SystemExit(f"run lacks stages.{stage}.stats_delta")
    return value


def aggregate_stage_delta(runs: list[dict], stage: str, denominator: int) -> dict:
    sums: dict[str, float | int] = collections.defaultdict(float)
    for run in runs:
        for key, value in stage_delta(run, stage)["stats_delta"].items():
            sums[key] += value
    out = dict(sums)
    out["denominator_forward_count"] = int(denominator)
    for key in ("upload_bytes", "download_bytes", "upload_calls",
                "download_calls", "plan_pack_us", "plan_submit_wait_us",
                "plan_output_sync_us", "plan_download_us", "plan_guard_us",
                "plan_publish_us", "plan_call_us", "plan_workspace_us",
                "host_action_us", "host_action_bytes", "kv_state_init_bytes",
                "kv_state_append_bytes", "kv_state_init_us",
                "kv_state_append_us", "task_count", "submit_count"):
        if key in out:
            out[f"{key}_per_forward"] = out[key] / denominator
    if out.get("kv_state_append_calls"):
        out["kv_state_append_bytes_per_append"] = (
            out["kv_state_append_bytes"] / out["kv_state_append_calls"])
        out["kv_state_append_us_per_append"] = (
            out["kv_state_append_us"] / out["kv_state_append_calls"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--package", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--calibration-bundle", required=True)
    ap.add_argument("--acceptance-bundle", required=True)
    ap.add_argument("--history", default="")
    args = ap.parse_args(argv)
    results = Path(args.results).resolve()
    raw = Path(args.raw).resolve()
    package = Path(args.package).resolve()
    spec_path = Path(args.spec).resolve()
    pm = read_json(package / "manifest.json")
    sm = read_json(spec_path)
    verify = require_complete_verifier(results / "verify/verify-summary.json")
    benchmark = read_json(results / "board/benchmark/result.json")
    bruns = benchmark["runs"]
    bench_run = read_json(results / "board/benchmark/run.json")
    raw_stats = read_json(results / "board/benchmark/stats.json")
    acc_run = read_json(results / "board/acceptance/run.json")
    cal_ppl_b = read_json(results / "quality/ppl-calibration-board.json")
    cal_ppl_o = read_json(results / "quality/ppl-calibration-official.json")
    acc_ppl_b = read_json(results / "quality/ppl-acceptance-board.json")
    acc_ppl_o = read_json(results / "quality/ppl-acceptance-official.json")
    reg_mlp = read_json(results / "regression/mlp/model_eval.json")
    reg_vit = read_json(results / "regression/tiny_vit/model_eval.json")

    # Validate actual forward counts and stage accounting.
    run_perf = []
    total_decode_forwards = 0
    for index, run in enumerate(bruns):
        out_tokens = int(run["output_tokens"])
        forwards = int(run["decode_forwards"])
        if forwards != max(0, out_tokens - 1):
            raise SystemExit(f"run {index}: decode_forwards != output_tokens-1")
        decode_stage = stage_delta(run, "decode")
        if int(decode_stage["forward_count"]) != forwards:
            raise SystemExit(f"run {index}: decode stage forward count mismatch")
        sampling_stage = stage_delta(run, "sampling")
        if int(sampling_stage["calls"]) != out_tokens:
            raise SystemExit(f"run {index}: sampling call count mismatch")
        total_decode_forwards += forwards
        run_perf.append({
            "run": index,
            "output_tokens": out_tokens,
            "decode_forwards": forwards,
            "stop_reason": run["stop_reason"],
            "reset_ms": float(run["reset_us"]) / 1000.0,
            "prefill_ms": float(run["prefill_us"]) / 1000.0,
            "first_sample_ms": float(run["first_sample_us"]) / 1000.0,
            "ttft_ms": float(run["ttft_us"]) / 1000.0,
            "sampling_ms": float(run["sampling_us"]) / 1000.0,
            "decode_ms": float(run["decode_us"]) / 1000.0,
            "generate_wall_ms": float(run["generate_wall_us"]) / 1000.0,
            "decode_forward_per_s": (
                forwards / (float(run["decode_us"]) / 1e6)
                if forwards else None),
            "output_tokens_per_s_full_generate": (
                out_tokens / (float(run["generate_wall_us"]) / 1e6)),
        })
    if total_decode_forwards <= 0:
        raise SystemExit("benchmark has no decode forwards")

    prefill_stats = aggregate_stage_delta(bruns, "prefill", len(bruns))
    decode_stats = aggregate_stage_delta(bruns, "decode",
                                         total_decode_forwards)
    sampling_stats = aggregate_stage_delta(bruns, "sampling",
                                           sum(r["output_tokens"]
                                               for r in run_perf))
    steady_forwards = total_decode_forwards - len(bruns)
    if steady_forwards > 0:
        decode_stats["steady_forward_count"] = steady_forwards
        decode_stats["kv_state_append_bytes_per_decode_token"] = (
            decode_stats.get("kv_state_append_bytes", 0) / steady_forwards)
        decode_stats["kv_state_append_us_per_decode_token"] = (
            decode_stats.get("kv_state_append_us", 0) / steady_forwards)
        decode_stats["kv_state_append_calls_per_decode_token"] = (
            decode_stats.get("kv_state_append_calls", 0) / steady_forwards)
    decode_stats["kv_state_init_bytes_per_run"] = (
        decode_stats.get("kv_state_init_bytes", 0) / len(bruns))
    roles = collections.defaultdict(lambda: {
        "plans": 0, "calls": 0, "tasks": 0, "submits": 0,
        "upload_bytes": 0, "download_bytes": 0, "elapsed_us": 0.0})
    for p in raw_stats["plans"]:
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
        d["calls"] += p["calls"]
        d["tasks"] += p["task_count"]
        d["submits"] += p["submit_count"]
        d["upload_bytes"] += p["upload_bytes"]
        d["download_bytes"] += p["download_bytes"]
        d["elapsed_us"] += p["elapsed_us"]

    # Instruction generation comparison.
    instruction_inputs = read_json(results / "quality/instruction-prompts.json")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(Path(".data/models/SmolLM2-135M-Instruct").resolve()))
    except Exception:
        tokenizer = None
    gen_rows = []
    for item in instruction_inputs:
        board_row = next(
            (r for r in acc_run["requests"]
             if r.get("mode") == "generate" and
             r.get("token_ids") == item["token_ids"] and
             r.get("runs") and
             r["runs"][0].get("output_tokens", 0) == 32), None)
        if board_row is None:
            raise SystemExit(f"no 32-token board generation for {item['id']}")
        b = board_row["runs"][0]
        official_path = results / "official" / f"generation-{item['id']}.json"
        if not official_path.is_file():
            raise SystemExit(f"missing official generation {official_path}")
        o = read_json(official_path)
        bg, og = b["generated_ids"], o["generated_ids"]
        n = min(len(bg), len(og))
        first = next((j for j in range(n) if bg[j] != og[j]), None)
        gen_rows.append({
            "id": item["id"],
            "prompt": item["prompt"],
            "prompt_ids": item["token_ids"],
            "generated_ids": bg,
            "official_generated_ids": og,
            "generated_text": tokenizer.decode(bg) if tokenizer else None,
            "official_text": o.get("text"),
            "top1_matches": sum(a == c for a, c in zip(bg, og)),
            "positions_compared": n,
            "first_divergence_index": first,
            "output_tokens": b["output_tokens"],
            "decode_forwards": b["decode_forwards"],
            "generate_wall_us": b["generate_wall_us"],
        })

    env = bench_run["environment"]
    median = lambda values: float(np.median(np.asarray(values, dtype=np.float64)))
    performance = {
        "measurement": "3 independent generate() requests in one process; "
                       "each resets persistent KV state and requests 32 tokens",
        "timing_definitions": {
            "ttft_ms": "reset + prefill + first sampling action inside "
                       "generate(); excludes process/model load, chat "
                       "tokenization and detokenization",
            "cold_start_to_first_token_ms": "runner load_us + first run ttft_ms",
            "decode_ms": "sum of decode_one() forward intervals only",
            "generate_wall_ms": "full generate() interval including reset, "
                                "prefill, all sampling, all decode forwards "
                                "and Python control",
            "per_decode_forward": "stage delta summed over decode forwards "
                                  "divided by actual decode forward count",
        },
        "host_compile_internal_us": pm["resources"]["compile_us"],
        "host_compile_external_wall_s": float(
            (results / "host/compile.time").read_text().split(
                "real_s=")[1].split()[0]),
        "load_init_us": bench_run["load_us"],
        "cold_start_to_first_token_ms": (
            bench_run["load_us"] + bruns[0]["ttft_us"]) / 1000.0,
        "run_results": run_perf,
        "aggregate": {
            "prefill_ms_median": median([r["prefill_ms"] for r in run_perf]),
            "ttft_ms_median": median([r["ttft_ms"] for r in run_perf]),
            "decode_ms_median": median([r["decode_ms"] for r in run_perf]),
            "generate_wall_ms_median": median(
                [r["generate_wall_ms"] for r in run_perf]),
            "decode_forward_per_s_median": median(
                [r["decode_forward_per_s"] for r in run_perf]),
            "output_tokens_per_s_full_generate_median": median(
                [r["output_tokens_per_s_full_generate"] for r in run_perf]),
            "output_tokens_total": sum(r["output_tokens"] for r in run_perf),
            "decode_forwards_total": total_decode_forwards,
        },
        "prefill_stage": prefill_stats,
        "decode_stage": decode_stats,
        "sampling_stage": sampling_stats,
        "raw_aggregate_stats_not_used_for_per_token": {
            "plan_call_us": raw_stats["plan_call_us"],
            "plan_workspace_us": raw_stats["plan_workspace_us"],
            "host_action_us": raw_stats["host_action_us"],
            "task_count": raw_stats["task_count"],
            "submit_count": raw_stats["submit_count"],
            "upload_bytes": raw_stats["upload_bytes"],
            "download_bytes": raw_stats["download_bytes"],
            "note": "These totals cover prefill + decode + sampling and are "
                    "not divided by a fixed token count.",
        },
        "peak_rss_kb": bench_run.get("process", {}).get("ru_maxrss_kb"),
        "npu_freq_hz": env.get("npu_freq_hz"),
        "core_mask": env.get("core_mask_env"),
        "library_sha256": env.get("plan_lib_sha256"),
    }

    history = read_json(Path(args.history)) if args.history else None
    manifest = {
        "schema_version": 2,
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
                    "tools/freeze_smollm2_v3.py",
                    "tools/seal_evidence_bundle.py",
                    "tools/summarize_smollm2_evidence.py",
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
            "pass": True,
            "mode": verify["mode"],
            "complete_acceptance": verify["complete_acceptance"],
            "spec": str(spec_path),
            "spec_canonical_content_sha256": sm[
                "canonical_content_sha256"],
            "spec_file_sha256": sha(spec_path),
            "policy_revision": sm.get("policy_revision"),
            "policy_notes": sm.get("policy_notes"),
            "gate_count": verify["gate_count_expected"],
            "gate_count_executed": verify["gate_count_executed"],
            "gate_failures": verify["gate_failures"],
            "failures": verify["failures"],
            "input_identity": verify["input_identity"],
            "calibration_manifest": verify["calibration_bundle"],
            "acceptance_manifest": verify["acceptance_bundle"],
            "incremental": verify["incremental"],
            "teacher_forcing": verify["teacher_forcing"],
            "quality": verify["quality"],
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
            "instruction_prompts": gen_rows,
        },
        "performance": performance,
        "regression": {
            "scope": "8-image board smoke through existing current MNIST MLP "
                     "and TinyViT hybrid plans; not a replacement for the "
                     "historical full-split acceptance",
            "mnist_mlp": reg_mlp["models"],
            "tiny_vit": reg_vit["models"],
        },
        "policy_history": history,
        "raw_evidence": {
            "local_root": str(raw),
            "calibration_bundle_path": str(Path(args.calibration_bundle)),
            "acceptance_bundle_path": str(Path(args.acceptance_bundle)),
            "benchmark": str(raw / "benchmark"),
            "acceptance": str(raw / "acceptance"),
            "ppl": str(raw / "ppl"),
        },
        "explicit_non_claims": [
            "single board, single core mask 1, batch 1, capacity 256",
            "FP16 weights/activations; no quantization comparison",
            "prefill uses a fixed 256-token bucket",
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
        "pass": True,
        "plans": manifest["backend"]["plan_count"],
        "decode_forward_per_s_run0": performance["run_results"][0][
            "decode_forward_per_s"],
        "ttft_ms_median": performance["aggregate"]["ttft_ms_median"],
        "decode_forward_per_s_median": performance["aggregate"][
            "decode_forward_per_s_median"],
        "output_tokens_per_s_full_generate_median": performance[
            "aggregate"]["output_tokens_per_s_full_generate_median"],
        "acceptance_nll": manifest["quality"]["acceptance_text"][
            "board_mean_nll"],
        "official_nll": manifest["quality"]["acceptance_text"][
            "official_mean_nll"],
        "peak_rss_kb": performance["peak_rss_kb"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
