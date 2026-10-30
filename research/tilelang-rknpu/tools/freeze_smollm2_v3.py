#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Freeze the SmolLM2 v3 acceptance rules from multi-prompt calibration data.

Inputs are frozen before the acceptance board run.  This tool consumes only
calibration observations and writes:
  * a versioned numeric/quality spec;
  * an explicit expected-gate set;
  * input identities for calibration and future acceptance;
  * the calibration observation-bundle SHA-256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import verify_smollm2 as vfy

UNIT = 1.0 / 4096.0


def sha(path: Path) -> str:
    return vfy.sha_file(path)


def ceil_unit(x: float) -> float:
    return math.ceil(max(0.0, float(x)) / UNIT) * UNIT


def f16_spacing(scale: float) -> float:
    value = np.float16(max(abs(float(scale)), np.finfo(np.float16).tiny))
    return float(abs(np.spacing(value)))


def merge_metrics(rows: list[dict]) -> dict:
    positions = sum(int(r["positions"]) for r in rows)
    matches = sum(int(r["top1_matches"]) for r in rows)
    means = [float(r["top5_overlap_mean"]) for r in rows]
    weights = [int(r["positions"]) for r in rows]
    return {
        "finite": all(bool(r["finite"]) for r in rows),
        "max_abs": max(float(r["max_abs"]) for r in rows),
        "rmse": max(float(r["rmse"]) for r in rows),
        "ref_absmax": max(float(r["ref_absmax"]) for r in rows),
        "ref_rms": max(float(r["ref_rms"]) for r in rows),
        "top1_matches": int(matches),
        "positions": int(positions),
        "top1_rate": matches / positions if positions else 0.0,
        "top5_overlap_mean": (
            sum(m * w for m, w in zip(means, weights)) / positions
            if positions else 0.0),
        "top5_overlap_min": min(int(r["top5_overlap_min"]) for r in rows),
        "sources": len(rows),
    }


def gate(m: dict, *, exact: bool = False) -> dict:
    spacing = f16_spacing(m["ref_absmax"])
    atol = ceil_unit(max(2.0 * m["max_abs"],
                         1.0 * UNIT if exact else 2.0 * spacing))
    rmse = ceil_unit(max(
        2.0 * m["rmse"],
        1.0 * UNIT if exact else max(0.25 * atol, 2 * spacing)))
    return {"atol": atol, "rmse_limit": rmse, "finite_required": True,
            "calibration_max_abs": m["max_abs"],
            "calibration_rmse": m["rmse"],
            "ref_absmax": m["ref_absmax"],
            "f16_spacing_at_ref_absmax": spacing,
            "calibration_sources": m.get("sources", 1)}


def quality_gate(m: dict, rank_policy: dict | None = None) -> dict:
    top1 = float(m["top1_rate"])
    top5 = float(m["top5_overlap_mean"])
    if rank_policy:
        top1_min = float(rank_policy["top1_rate_min"])
        top5_min = float(rank_policy["top5_overlap_mean_min"])
        source = "predeclared_fixed_policy"
    else:
        top1_min = max(0.95, top1 - 0.01)
        top5_min = max(4.0, top5 - 0.5)
        source = "calibration_derived"
    return {"top1_rate_min": top1_min,
            "top5_overlap_mean_min": top5_min,
            "rank_threshold_source": source,
            "calibration_top1_rate": top1,
            "calibration_top5_overlap_mean": top5}


def board_rows(board: Path) -> list[dict]:
    return json.loads((board / "run.json").read_text())["requests"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-root", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--calibration-bundle", required=True)
    ap.add_argument("--out-spec", required=True)
    args = ap.parse_args(argv)
    root = Path(args.evidence_root).resolve()
    board = root / "board"
    inp = json.loads(Path(args.inputs).read_text())
    rank_policy = inp.get("rank_policy")
    rows = board_rows(board)
    captures = [r for r in rows if r["mode"] == "capture"]
    tfs = [r for r in rows if r["mode"] == "teacher_forcing"]
    incs = [r for r in rows if r["mode"] == "incremental"]
    cal_prompts = inp["calibration_prompts"]
    if len(captures) != len(cal_prompts):
        raise SystemExit(f"capture count {len(captures)} != calibration prompts "
                         f"{len(cal_prompts)}")
    if len(tfs) != len(cal_prompts):
        raise SystemExit("teacher-forcing count does not match calibration prompts")
    if len(incs) != 1:
        raise SystemExit("calibration must contain exactly one incremental run")

    gate_rows: dict[str, list[dict]] = {}
    full_rows = []
    official_rows = []
    shard_rows: dict[str, list[dict]] = {}
    tf_rows = []
    incremental_metrics = None
    capture_file_hashes: dict[str, dict] = {}

    for index, row in enumerate(captures):
        req_index = int(row["index"])
        run_dir = board / f"run-{req_index:02d}"
        ref_dir = root / "reference" / f"capture-{index}"
        bcap = np.load(run_dir / "captured-00.npz", allow_pickle=False)
        rcap = np.load(ref_dir / "captured.npz", allow_pickle=False)
        blog = np.load(run_dir / "all-logits-00.npy", allow_pickle=False)
        rlog = np.load(ref_dir / "logits.npy", allow_pickle=False)
        prompt_id = cal_prompts[index]["id"]
        capture_file_hashes[prompt_id] = {
            "board_captured": sha(run_dir / "captured-00.npz"),
            "board_logits": sha(run_dir / "all-logits-00.npy"),
            "reference_captured": sha(ref_dir / "captured.npz"),
            "reference_logits": sha(ref_dir / "logits.npy"),
            "reference_meta": sha(ref_dir / "meta.json"),
        }
        if set(bcap.files) != set(rcap.files):
            raise SystemExit(f"capture tensor set mismatch for {prompt_id}")
        n = int(json.loads((ref_dir / "meta.json").read_text())["tokens"])
        blog = blog[:n]
        rlog = rlog[:n]
        if blog.shape != rlog.shape:
            raise SystemExit(f"logits shape mismatch for {prompt_id}")
        for name in sorted(rcap.files):
            m = vfy.metrics(bcap[name], rcap[name])
            if not m["finite"]:
                raise SystemExit(f"nonfinite capture {prompt_id}:{name}")
            gate_rows.setdefault(name, []).append(m)
        full_rows.append(vfy.metrics(blog, rlog))
        off_path = root / "official-logits" / prompt_id / "logits.npy"
        official = np.load(off_path, allow_pickle=False)
        official_rows.append(vfy.metrics(blog, official))
        shards = 12
        width = (blog.shape[1] + shards - 1) // shards
        for j in range(shards):
            start, stop = j * width, min(blog.shape[1], (j + 1) * width)
            shard_rows.setdefault(f"logits.shard{j:02d}", []).append(
                vfy.metrics(blog[:, start:stop], rlog[:, start:stop]))

    for index, row in enumerate(tfs):
        req_index = int(row["index"])
        run_dir = board / f"run-{req_index:02d}"
        ref_dir = root / "reference" / f"tf-{index}"
        btf = np.load(run_dir / "logits-00.npy", allow_pickle=False)
        rtf = np.load(ref_dir / "logits.npy", allow_pickle=False)
        if btf.shape != rtf.shape:
            raise SystemExit(f"teacher-forcing shape mismatch {index}")
        tf_rows.append(vfy.metrics(btf, rtf))

    inc_row = incs[0]
    inc_index = int(inc_row["index"])
    inc_board = np.load(board / f"run-{inc_index:02d}" /
                        "incremental-logits-00.npy", allow_pickle=False)
    # Locate the matching teacher-forcing request with the same prompt.
    match_tf = None
    for index, row in enumerate(tfs):
        if row["token_ids"][:len(inc_row["token_ids"])] == inc_row["token_ids"]:
            match_tf = (index, row)
            break
    if match_tf is None:
        raise SystemExit("no matching teacher-forcing row for incremental")
    tf_index, tf_row = match_tf
    tf_board = np.load(board / f"run-{int(tf_row['index']):02d}" /
                       "logits-00.npy", allow_pickle=False)
    prompt_len = int(inc_row["token_count"])
    tf_slice = tf_board[prompt_len - 1:prompt_len - 1 + inc_board.shape[0]]
    incremental_metrics = vfy.metrics(inc_board, tf_slice)

    gates: dict[str, dict] = {}
    calibration_metrics: dict[str, dict] = {}
    for name, rows_for_tensor in sorted(gate_rows.items()):
        merged = merge_metrics(rows_for_tensor)
        calibration_metrics[name] = merged
        gates[name] = gate(merged)
    full = merge_metrics(full_rows)
    calibration_metrics["logits.full"] = full
    gates["logits.full"] = {**gate(full),
                            **quality_gate(full, rank_policy)}
    official = merge_metrics(official_rows)
    calibration_metrics["official.logits"] = official
    gates["official.logits"] = {**gate(official),
                                **quality_gate(official, rank_policy)}
    for name, rows_for_shard in sorted(shard_rows.items()):
        merged = merge_metrics(rows_for_shard)
        calibration_metrics[name] = merged
        gates[name] = gate(merged)
    tf_merged = merge_metrics(tf_rows)
    calibration_metrics["teacher_forcing.logits"] = tf_merged
    gates["teacher_forcing.logits"] = {
        **gate(tf_merged), **quality_gate(tf_merged, rank_policy)}
    calibration_metrics["incremental_vs_full_prefix.logits"] = \
        incremental_metrics
    gates["incremental_vs_full_prefix.logits"] = {
        **gate(incremental_metrics, exact=True),
        **quality_gate(incremental_metrics, rank_policy)}

    # Quality calibration from a text corpus disjoint from the future text.
    cal_windows = inp["calibration_nll_windows"]
    ppl_board = vfy.load_board_ppl(root / "ppl" / "board", cal_windows)
    ppl_official = vfy.load_official_ppl(
        root / "ppl" / "official" / "result.json", cal_windows)
    delta = ppl_board["mean_nll"] - ppl_official["mean_nll"]
    ratio = ppl_board["ppl"] / ppl_official["ppl"]
    top1_delta = ppl_board["top1_accuracy"] - ppl_official["top1_accuracy"]
    quality_limits = {
        "scored_tokens_min": 1024,
        "mean_nll_delta_max": max(0.02, ceil_unit(2.0 * abs(delta))),
        "ppl_ratio_max": max(1.02, 1.0 + 2.0 * abs(ratio - 1.0)),
        "top1_accuracy_delta_min": min(-0.01, top1_delta - 0.01),
    }
    quality = {
        "dataset": {
            "source": cal_windows["source"],
            "source_sha256": cal_windows["source_sha256"],
            "window_size": cal_windows["window_size"],
            "windows": cal_windows["window_meta"],
        },
        "board": {k: ppl_board[k] for k in
                  ("scored_tokens", "sum_nll", "mean_nll", "ppl",
                   "top1_accuracy", "nll_sha256")},
        "official": {k: ppl_official[k] for k in
                     ("scored_tokens", "sum_nll", "mean_nll", "ppl",
                      "top1_accuracy", "nll_sha256")},
        "calibration_delta_mean_nll": delta,
        "calibration_ppl_ratio": ratio,
        "calibration_top1_delta": top1_delta,
        "limits": quality_limits,
        "policy": "limits are derived only from calibration data with a "
                  "fixed margin; the acceptance text windows are disjoint",
    }

    acceptance_request_path = Path.cwd() / inp["acceptance_request"]["path"]
    if sha(acceptance_request_path) != inp["acceptance_request"]["sha256"]:
        raise SystemExit("acceptance request identity changed before freeze")
    official_reference = {}
    for prompt_id in (inp["calibration_prompts"][0]["id"],):
        meta = json.loads(
            (root / "official-logits" / prompt_id / "result.json").read_text())
        official_reference = {
            "config_sha256": meta.get("config_sha256"),
            "weights_sha256": meta.get("weights_sha256"),
            "tokenizer_sha256": meta.get("tokenizer_sha256"),
            "torch": meta.get("torch"),
            "transformers": meta.get("transformers"),
            "dtype": meta.get("dtype"),
            "attn_implementation": meta.get("attn_implementation"),
        }
    board_meta = json.loads((board / "run.json").read_text())
    env = board_meta["environment"]
    ppl_meta = json.loads((root / "ppl" / "official" / "result.json").read_text())
    common_weights = ppl_meta.get("weights_sha256")
    if common_weights and official_reference.get("weights_sha256") and \
            common_weights != official_reference["weights_sha256"]:
        raise SystemExit("official reference weight identity differs")

    spec = {
        "schema_version": 2,
        "kind": "smollm2-numeric-acceptance",
        "policy_revision": 3,
        "policy_notes": [
            f"Thresholds are frozen from {len(cal_prompts)} calibration prompts, 39 captured tensors from layers 0/15/29, full and sharded logits, teacher-forcing logits and one incremental decode sequence.",
            "Per-lm_head-shard rank is diagnostic; each shard is still gated numerically and by exact vocabulary position through the declared-reference slice.",
            "Official free-generation divergence is reported, not used as the backend numeric gate.",
            "Complete acceptance requires numeric and quality evidence plus sealed calibration and acceptance observation bundles.",
        ],
        "model_id": inp["model_id"],
        "revision": inp["revision"],
        "official_reference": official_reference,
        "input_identity": {
            "input_manifest_path": str(Path(args.inputs).resolve()),
            "input_manifest_sha256": sha(Path(args.inputs)),
            "input_plan_frozen_sha256": inp["frozen_sha256"],
            "calibration": {
                "prompt_token_ids": [x["token_ids"]
                                     for x in inp["calibration_prompts"]],
                "nll_windows": cal_windows["window_meta"],
            },
            "acceptance": {
                "prompt_token_ids": [x["token_ids"]
                                     for x in inp["acceptance_prompts"]],
                "nll_windows": inp["acceptance_nll_windows"]["window_meta"],
            },
            "numeric_acceptance_prompt_ids": inp[
                "numeric_acceptance_prompt_ids"],
            "teacher_forcing_token_ids": inp["teacher_forcing_token_ids"],
            "incremental_continuation_ids": inp[
                "incremental_continuation_ids"],
            "acceptance_request_sha256": inp["acceptance_request"]["sha256"],
            "ppl_acceptance_request_sha256": inp[
                "ppl_acceptance_request"]["sha256"],
        },
        "calibration": {
            "token_ids": inp["calibration_prompts"][0]["token_ids"],
            "prompt_tokens": len(inp["calibration_prompts"][0]["token_ids"]),
            "prompt_text": inp["calibration_prompts"][0]["text"],
            "prompt_ids": [x["id"] for x in inp["calibration_prompts"]],
            "package_manifest_sha256": env["package_manifest_sha256"],
            "package_weights_sha256": env["package_weights_sha256"],
            "board_run_sha256": sha(board / "run.json"),
            "board_stats_sha256": sha(board / "stats.json"),
            "observation_bundle_sha256": sha(
                Path(args.calibration_bundle).resolve()),
            "observation_bundle_file_sha256": sha(
                Path(args.calibration_bundle).resolve()),
            "capture_file_hashes": capture_file_hashes,
        },
        "expected_gates": list(gates),
        "gates": gates,
        "calibration_metrics": calibration_metrics,
        "quality": quality,
    }
    body = dict(spec)
    spec["canonical_content_sha256"] = vfy.canonical_json_sha(body)
    out = Path(args.out_spec)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n")
    print(json.dumps({
        "spec": str(out),
        "canonical_content_sha256": spec["canonical_content_sha256"],
        "gates": len(gates),
        "calibration_prompts": len(cal_prompts),
        "quality_limits": quality_limits,
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
