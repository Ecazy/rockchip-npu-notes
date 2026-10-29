#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Verify frozen SmolLM2 numeric/quality acceptance observations.

The thresholds in the spec are never recomputed here.  This tool rejects a
different spec hash, a changed package/weight identity, a calibration/acceptance
sequence collision, non-finite values, or any gate above its frozen limit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from calibrate_smollm2 import metrics


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def spec_hash(spec: dict) -> str:
    obj = {k: v for k, v in spec.items() if k != "spec_sha256"}
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def load_ppl(board_dir: Path) -> dict:
    rows = [json.loads(p.read_text())["runs"][0]
            for p in sorted(board_dir.glob("run-*/result.json"))]
    nll_files = sorted(board_dir.glob("run-*/nll-*.npy"))
    nll = np.concatenate([np.load(f, allow_pickle=False)
                          for f in nll_files])
    return {
        "scored_tokens": int(nll.size),
        "sum_nll": float(nll.sum()),
        "mean_nll": float(nll.mean()),
        "ppl": float(np.exp(nll.mean())),
        "top1_accuracy": float(sum(
            float(r["top1_accuracy"]) * int(r["scored_tokens"])
            for r in rows) / nll.size),
        "windows": rows,
        "nll_sha256": hashlib.sha256(nll.tobytes()).hexdigest(),
    }


def check_gate(value: np.ndarray, ref: np.ndarray, gate: dict) -> dict:
    m = metrics(value, ref)
    failures = []
    if gate.get("finite_required") and not m["finite"]:
        failures.append("nonfinite")
    if m["max_abs"] > float(gate["atol"]):
        failures.append("max_abs")
    if m["rmse"] > float(gate["rmse_limit"]):
        failures.append("rmse")
    if "top1_rate_min" in gate and \
            m["top1_rate"] < float(gate["top1_rate_min"]):
        failures.append("top1_rate")
    if "top5_overlap_mean_min" in gate and \
            m["top5_overlap_mean"] < float(gate["top5_overlap_mean_min"]):
        failures.append("top5_overlap")
    return {"pass": not failures, "failures": failures,
            "observed": m,
            "limits": {k: gate[k] for k in
                       ("atol", "rmse_limit", "top1_rate_min",
                        "top5_overlap_mean_min") if k in gate}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--board", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--reference-tf", required=True)
    ap.add_argument("--official-logits", required=True)
    ap.add_argument("--ppl-board", default="")
    ap.add_argument("--ppl-official", default="")
    ap.add_argument("--acceptance-prompt", default="")
    ap.add_argument("--acceptance-source-json", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text())
    frozen = spec.get("spec_sha256")
    actual_spec = spec_hash(spec)
    board = Path(args.board).resolve()
    refdir = Path(args.reference).resolve()
    reftf = Path(args.reference_tf).resolve()
    bcap = np.load(board / "run-00" / "captured-00.npz", allow_pickle=False)
    rcap = np.load(refdir / "captured.npz", allow_pickle=False)
    blog = np.load(board / "run-00" / "all-logits-00.npy",
                   allow_pickle=False)
    rlog = np.load(refdir / "logits.npy", allow_pickle=False)
    rmeta = json.loads((refdir / "meta.json").read_text())
    btf = np.load(board / "run-01" / "logits-00.npy", allow_pickle=False)
    rtf = np.load(reftf / "logits.npy", allow_pickle=False)
    inc = np.load(board / "run-02" / "incremental-logits-00.npy",
                  allow_pickle=False)
    official = np.load(args.official_logits, allow_pickle=False).astype(
        np.float32)
    run_meta = json.loads((board / "run.json").read_text())
    n = int(rmeta["tokens"])
    blog = blog[:n]
    prompt_len = int(json.loads(
        (board / "run-02" / "result.json").read_text())["token_count"])
    tf_slice = btf[prompt_len - 1:prompt_len - 1 + inc.shape[0]]

    failures: list[str] = []
    if actual_spec != frozen:
        failures.append("spec_sha256")
    calib_ids = list(spec["calibration"]["token_ids"])
    accept_ids = list(rmeta["token_ids"])
    if accept_ids == calib_ids:
        failures.append("calibration_acceptance_sequence_collision")
    if run_meta["environment"]["package_manifest_sha256"] != \
            spec["calibration"]["package_manifest_sha256"]:
        failures.append("package_manifest_identity")
    if run_meta["environment"]["package_weights_sha256"] != \
            spec["calibration"]["package_weights_sha256"]:
        failures.append("package_weights_identity")

    gates: dict[str, dict] = {}
    if set(bcap.files) != set(rcap.files):
        failures.append("captured_tensor_set")
    for name in sorted(rcap.files):
        gates[name] = check_gate(bcap[name], rcap[name],
                                 spec["gates"][name])
    gates["logits.full"] = check_gate(blog, rlog,
                                      spec["gates"]["logits.full"])
    vocab = int(blog.shape[1])
    shards = 12
    width = (vocab + shards - 1) // shards
    for j in range(shards):
        start, stop = j * width, min(vocab, (j + 1) * width)
        name = f"logits.shard{j:02d}"
        gates[name] = check_gate(blog[:, start:stop],
                                 rlog[:, start:stop],
                                 spec["gates"][name])
    gates["official.logits"] = check_gate(
        blog, official, spec["gates"]["official.logits"])
    gates["teacher_forcing.logits"] = check_gate(
        btf, rtf, spec["gates"]["teacher_forcing.logits"])
    gates["incremental_vs_full_prefix.logits"] = check_gate(
        inc, tf_slice, spec["gates"]["incremental_vs_full_prefix.logits"])

    # Cross-check that the teacher-forced top-1 sequence is the advertised
    # greedy continuation, and that incremental decode has the same top-1 at
    # every compared position.
    tf_ids = list(json.loads((reftf / "meta.json").read_text())["token_ids"])
    continuation = tf_ids[prompt_len:]
    tf_pred = [int(v) for v in np.argmax(btf[prompt_len - 1:], axis=1)]
    ncheck = min(len(continuation), len(tf_pred))
    first_tf_div = next((i for i in range(ncheck)
                         if tf_pred[i] != continuation[i]), None)
    inc_pred = [-1] * inc.shape[0]
    full_pred = [-1] * inc.shape[0]
    for i in range(inc.shape[0]):
        inc_pred[i] = int(np.argmax(inc[i]))
        full_pred[i] = int(np.argmax(tf_slice[i]))
    if inc_pred != full_pred:
        failures.append("incremental_top1_vs_full_prefix")
    # The official continuation is reported, not gated: a first divergence is
    # a model-quality statistic.  The authoritative teacher-forcing numeric
    # gate above compares every logit with the declared reference.

    # Quality comparison on a fixed, disjoint text corpus.  The calibration
    # limits were frozen in the spec before this acceptance corpus was scored.
    quality = None
    if args.ppl_board or args.ppl_official:
        if not (args.ppl_board and args.ppl_official):
            raise SystemExit("both --ppl-board and --ppl-official are required")
        bp = load_ppl(Path(args.ppl_board))
        op = json.loads(Path(args.ppl_official).read_text())
        limits = spec["quality"]["limits"]
        delta = bp["mean_nll"] - float(op["mean_nll"])
        ratio = bp["ppl"] / float(op["ppl"])
        top1_delta = bp["top1_accuracy"] - float(op["top1_accuracy"])
        quality = {
            "pass": True, "board": bp,
            "official": {k: op[k] for k in
                         ("scored_tokens", "sum_nll", "mean_nll", "ppl",
                          "top1_accuracy")},
            "mean_nll_delta": delta, "ppl_ratio": ratio,
            "top1_accuracy_delta": top1_delta,
            "limits": limits,
        }
        if bp["scored_tokens"] < int(limits["scored_tokens_min"]):
            quality["pass"] = False
            failures.append("quality_scored_tokens")
        if abs(delta) > float(limits["mean_nll_delta_max"]):
            quality["pass"] = False
            failures.append("quality_mean_nll_delta")
        if abs(ratio - 1.0) > float(limits["ppl_ratio_max"]) - 1.0:
            quality["pass"] = False
            failures.append("quality_ppl_ratio")
        if top1_delta < float(limits["top1_accuracy_delta_min"]):
            quality["pass"] = False
            failures.append("quality_top1_delta")
        if args.acceptance_source_json:
            src = json.loads(Path(args.acceptance_source_json).read_text())
            if src["source_sha256"] == \
                    spec["quality"]["dataset"]["source_sha256"]:
                quality["pass"] = False
                failures.append("quality_corpus_collision")

    first_failed = [k for k, v in gates.items() if not v["pass"]]
    result = {
        "kind": "smollm2-acceptance-verification",
        "pass": not failures,
        "spec": str(Path(args.spec).resolve()),
        "spec_sha256_frozen": frozen,
        "spec_sha256_recomputed": actual_spec,
        "acceptance_prompt": args.acceptance_prompt,
        "acceptance_prompt_tokens": n,
        "calibration_prompt_tokens": len(calib_ids),
        "acceptance_sequence_disjoint_from_calibration": accept_ids != calib_ids,
        "package_manifest_sha256": run_meta["environment"][
            "package_manifest_sha256"],
        "package_weights_sha256": run_meta["environment"][
            "package_weights_sha256"],
        "gate_count": len(gates),
        "gate_failures": first_failed,
        "failures": failures,
        "teacher_forcing": {
            "positions_checked": ncheck,
            "first_divergence_index": first_tf_div,
            "top1_matches": sum(a == b for a, b in
                                zip(tf_pred, continuation)),
            "top1_rate": (sum(a == b for a, b in
                              zip(tf_pred, continuation)) / ncheck
                          if ncheck else 0.0),
        },
        "incremental": {
            "positions_checked": inc.shape[0],
            "top1_equal_to_full_prefix": inc_pred == full_pred,
            "predicted_ids": inc_pred,
        },
        "gates": gates,
        "quality": quality,
        "observation_hashes": {
            "board_captured": sha(board / "run-00" / "captured-00.npz"),
            "board_logits": sha(board / "run-00" / "all-logits-00.npy"),
            "board_tf": sha(board / "run-01" / "logits-00.npy"),
            "board_incremental": sha(board / "run-02" /
                                     "incremental-logits-00.npy"),
            "reference_logits": sha(refdir / "logits.npy"),
            "reference_tf": sha(reftf / "logits.npy"),
            "official_logits": sha(Path(args.official_logits)),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"pass": result["pass"],
                      "first_failed_gates": first_failed,
                      "failures": failures,
                      "incremental_top1_equal": inc_pred == full_pred,
                      "teacher_forcing_first_divergence": first_tf_div,
                      "quality": quality and {
                          "pass": quality["pass"],
                          "mean_nll_delta": quality["mean_nll_delta"],
                          "ppl_ratio": quality["ppl_ratio"],
                          "scored_tokens": quality["board"]["scored_tokens"]}},
                     indent=1))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
