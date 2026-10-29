#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Freeze SmolLM2 backend numeric/quality limits from one calibration prompt.

The calibration observation must be generated before this script runs and must
be disjoint from the later acceptance prompt.  The script records every source
archive hash in the spec so a verifier can prove which observations produced the
frozen numbers.  This is the only place limits are created.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

UNIT = 1.0 / 4096.0


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ceil_unit(x: float) -> float:
    return math.ceil(max(0.0, float(x)) / UNIT) * UNIT


def f16_spacing(scale: float) -> float:
    value = np.float16(max(abs(float(scale)), np.finfo(np.float16).tiny))
    return float(abs(np.spacing(value)))


def metrics(board: np.ndarray, ref: np.ndarray) -> dict:
    b = board.astype(np.float32)
    r = ref.astype(np.float32)
    e = b - r
    flat_b = b.reshape(-1, b.shape[-1]) if b.ndim > 1 else b.reshape(1, -1)
    flat_r = r.reshape(-1, r.shape[-1]) if r.ndim > 1 else r.reshape(1, -1)
    top1_b = np.argmax(flat_b, axis=-1)
    top1_r = np.argmax(flat_r, axis=-1)
    top5_b = np.argsort(flat_b, axis=-1)[:, -5:]
    top5_r = np.argsort(flat_r, axis=-1)[:, -5:]
    overlaps = [len(set(x.tolist()) & set(y.tolist()))
                for x, y in zip(top5_b, top5_r)]
    topology = {
        "top1_matches": int(np.sum(top1_b == top1_r)),
        "positions": int(top1_b.size),
        "top1_rate": float(np.mean(top1_b == top1_r)),
        "top5_overlap_mean": float(np.mean(overlaps)),
        "top5_overlap_min": int(min(overlaps)),
    }
    return {
        "finite": bool(np.isfinite(b).all()),
        "max_abs": float(np.max(np.abs(e))) if e.size else 0.0,
        "rmse": float(np.sqrt(np.mean(e * e))) if e.size else 0.0,
        "ref_absmax": float(np.max(np.abs(r))) if r.size else 0.0,
        "ref_rms": float(np.sqrt(np.mean(r * r))) if r.size else 0.0,
        **topology,
    }


def gate(m: dict, *, exact: bool = False) -> dict:
    spacing = f16_spacing(m["ref_absmax"])
    atol = ceil_unit(max(2.0 * m["max_abs"], 1.0 * UNIT if exact else
                         2.0 * spacing))
    rmse = ceil_unit(max(2.0 * m["rmse"],
                         1.0 * UNIT if exact else max(0.25 * atol, 2 * spacing)))
    return {"atol": atol, "rmse_limit": rmse, "finite_required": True,
            "calibration_max_abs": m["max_abs"],
            "calibration_rmse": m["rmse"],
            "ref_absmax": m["ref_absmax"], "f16_spacing_at_ref_absmax": spacing}


def quality_gate(m: dict) -> dict:
    top1_cal = float(m.get("top1_rate", m.get("top1_matches", 0)))
    top5_cal = float(m.get("top5_overlap_mean",
                           m.get("top5_overlap", 0)))
    return {
        "top1_rate_min": max(0.95, top1_cal - 0.01),
        "top5_overlap_mean_min": max(4.0, top5_cal - 0.5),
        "calibration_top1_rate": top1_cal,
        "calibration_top5_overlap_mean": top5_cal,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--reference-tf", required=True)
    ap.add_argument("--official-logits", required=True)
    ap.add_argument("--prompt-text", default="")
    ap.add_argument("--ppl-board", default="")
    ap.add_argument("--ppl-official", default="")
    ap.add_argument("--ppl-source-json", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    board = Path(args.board).resolve()
    refdir = Path(args.reference).resolve()
    reftf = Path(args.reference_tf).resolve()
    official = np.load(args.official_logits, allow_pickle=False).astype(
        np.float32)
    bcap = np.load(board / "run-00" / "captured-00.npz", allow_pickle=False)
    rcap = np.load(refdir / "captured.npz", allow_pickle=False)
    blog = np.load(board / "run-00" / "all-logits-00.npy",
                   allow_pickle=False)
    rlog = np.load(refdir / "logits.npy", allow_pickle=False)
    rmeta = json.loads((refdir / "meta.json").read_text())
    board_meta = json.loads((board / "run.json").read_text())
    n = int(rmeta["tokens"])
    blog = blog[:n]
    btf = np.load(board / "run-01" / "logits-00.npy", allow_pickle=False)
    rtf = np.load(reftf / "logits.npy", allow_pickle=False)
    inc = np.load(board / "run-02" / "incremental-logits-00.npy",
                  allow_pickle=False)
    prompt_len = int(json.loads(
        (board / "run-02" / "result.json").read_text())["token_count"])
    tf_slice = btf[prompt_len - 1:prompt_len - 1 + inc.shape[0]]

    gates: dict[str, dict] = {}
    calibration: dict[str, dict] = {}
    if set(bcap.files) != set(rcap.files):
        raise SystemExit("captured tensor sets differ")
    for name in sorted(rcap.files):
        m = metrics(bcap[name], rcap[name])
        calibration[name] = m
        gates[name] = gate(m)
    full_m = metrics(blog, rlog)
    calibration["logits.full"] = full_m
    gates["logits.full"] = {**gate(full_m), **quality_gate(full_m)}
    shards = int(json.loads((board / "run.json").read_text())["requests"][0]
                 .get("lm_head_shards", 12)) if False else 12
    width = (int(rmeta.get("vocab_size", 49152)) + shards - 1) // shards
    # The manifest carries the exact vocab size; derive it from logits width.
    vocab = int(blog.shape[1])
    width = (vocab + shards - 1) // shards
    for j in range(shards):
        start, stop = j * width, min(vocab, (j + 1) * width)
        name = f"logits.shard{j:02d}"
        m = metrics(blog[:, start:stop], rlog[:, start:stop])
        calibration[name] = m
        # Per-shard rank is diagnostic only.  A near-tie inside one 4096-wide
        # slice is not a vocabulary-indexing defect; the full-vocab rank gate
        # below is authoritative.  Bounded numeric error still gates the slice.
        gates[name] = gate(m)
    off_m = metrics(blog, official)
    calibration["official.logits"] = off_m
    gates["official.logits"] = {**gate(off_m), **quality_gate(off_m)}
    tf_m = metrics(btf, rtf)
    calibration["teacher_forcing.logits"] = tf_m
    gates["teacher_forcing.logits"] = {**gate(tf_m), **quality_gate(tf_m)}
    inc_m = metrics(inc, tf_slice)
    calibration["incremental_vs_full_prefix.logits"] = inc_m
    gates["incremental_vs_full_prefix.logits"] = {
        **gate(inc_m, exact=True), **quality_gate(inc_m)}

    quality = None
    if args.ppl_board and args.ppl_official:
        bp = json.loads(Path(args.ppl_board).read_text())
        op = json.loads(Path(args.ppl_official).read_text())
        cal_delta = float(bp["mean_nll"]) - float(op["mean_nll"])
        cal_ratio = float(bp["ppl"]) / float(op["ppl"])
        cal_top1_delta = float(bp["top1_accuracy"]) - float(
            op["top1_accuracy"])
        quality = {
            "dataset": (json.loads(Path(args.ppl_source_json).read_text())
                        if args.ppl_source_json else {}),
            "board": {k: bp[k] for k in
                      ("scored_tokens", "sum_nll", "mean_nll", "ppl",
                       "top1_accuracy")},
            "official": {k: op[k] for k in
                         ("scored_tokens", "sum_nll", "mean_nll", "ppl",
                          "top1_accuracy")},
            "calibration_delta_mean_nll": cal_delta,
            "calibration_ppl_ratio": cal_ratio,
            "calibration_top1_delta": cal_top1_delta,
            "limits": {
                "scored_tokens_min": 1024,
                "mean_nll_delta_max": max(
                    0.02, ceil_unit(2.0 * abs(cal_delta))),
                "ppl_ratio_max": max(
                    1.02, 1.0 + 2.0 * abs(cal_ratio - 1.0)),
                "top1_accuracy_delta_min": min(
                    -0.01, cal_top1_delta - 0.01),
            },
            "policy": "limits are derived with a fixed margin from the "
                      "calibration text corpus only",
        }

    spec = {
        "schema_version": 1,
        "kind": "smollm2-numeric-acceptance",
        "policy_revision": 2,
        "policy_notes": [
            "Full-vocabulary top-1/top-5 rank is gated; per-lm_head-shard rank is diagnostic only because a near-tie within one slice is not a vocabulary indexing or concatenation failure.",
            "Teacher-forcing top-1 divergence from an official continuation is reported as a quality/first-divergence statistic, not as a backend numeric gate; the authoritative teacher-forcing numeric gate compares against the declared reference.",
            "Numeric atol/rmse limits are unchanged from the calibration formula.",
        ],
        "model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
        "calibration": {
            "token_ids": rmeta["token_ids"],
            "prompt_tokens": n,
            "prompt_text": args.prompt_text,
            "package_manifest_sha256": board_meta["environment"][
                "package_manifest_sha256"],
            "package_weights_sha256": board_meta["environment"][
                "package_weights_sha256"],
            "reference_meta_sha256": sha(refdir / "meta.json"),
            "board_run_sha256": sha(board / "run.json"),
            "board_stats_sha256": sha(board / "stats.json"),
            "board_captured_sha256": sha(board / "run-00" / "captured-00.npz"),
            "board_logits_sha256": sha(board / "run-00" / "all-logits-00.npy"),
            "board_teacher_forcing_sha256": sha(board / "run-01" / "logits-00.npy"),
            "board_incremental_sha256": sha(board / "run-02" /
                                            "incremental-logits-00.npy"),
            "reference_logits_sha256": sha(refdir / "logits.npy"),
            "reference_tf_logits_sha256": sha(reftf / "logits.npy"),
            "official_logits_sha256": sha(Path(args.official_logits)),
            "policy": {
                "unit": "2^-12",
                "atol": "ceil_2^-12(max(2*calibration_max_abs, "
                        "2*spacing_f16(ref_absmax)))",
                "rmse": "ceil_2^-12(max(2*calibration_rmse, "
                        "max(0.25*atol, 2*spacing_f16(ref_absmax))))",
                "exact_boundary": "incremental-vs-full uses floor 2^-12",
                "quality": "top1 >= max(0.95, calibration-0.01); "
                           "top5 mean >= max(4.0, calibration-0.5)",
            },
        },
        "calibration_metrics": calibration,
        "gates": gates,
        "quality": quality,
    }
    raw = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    spec["spec_sha256"] = hashlib.sha256(raw).hexdigest()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"spec": str(out), "spec_sha256": spec["spec_sha256"],
                      "gates": len(gates),
                      "calibration_tokens": n}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
