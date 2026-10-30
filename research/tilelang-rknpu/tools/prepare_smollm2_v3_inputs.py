#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tokenize and freeze the SmolLM2 v3 calibration/acceptance input identities.

Run once without ``--official-continuation-json`` to produce a draft, then run
again with the official greedy continuation to freeze the teacher-forcing
sequence.  The final file records full token sequences and SHA-256 digests and
rejects reuse of any known earlier prompt or text-window identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha(obj: dict) -> str:
    return sha_bytes(json.dumps(obj, sort_keys=True,
                                separators=(",", ":")).encode())


def token_ids_sha(ids: list[int]) -> str:
    import numpy as np
    return sha_bytes(np.asarray(ids, dtype="<i8").tobytes())


def tokenize(tokenizer, text: str) -> list[int]:
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=True,
        add_generation_prompt=True)
    if not isinstance(ids, list) or not ids:
        raise SystemExit("chat template returned no token ids")
    return [int(v) for v in ids]


def windows_for(tokenizer, source: str, width: int, count: int) -> dict:
    source_path = (ROOT / source).resolve()
    text_bytes = source_path.read_bytes()
    ids = tokenizer(text_bytes.decode("utf-8"),
                    add_special_tokens=False)["input_ids"]
    need = width * count
    if len(ids) < need:
        raise SystemExit(f"{source} has {len(ids)} tokens, need {need}")
    windows = [ids[i:i + width] for i in range(0, need, width)]
    meta = []
    for i, window in enumerate(windows):
        labels = window[1:]
        meta.append({
            "window": i,
            "token_ids": [int(v) for v in window],
            "token_seq_sha256": token_ids_sha(window),
            "scored_label_ids": [int(v) for v in labels],
            "scored_label_sha256": token_ids_sha(labels),
            "scored_tokens": len(labels),
        })
    return {
        "kind": "smollm2-nll-windows",
        "source": str(source_path),
        "source_sha256": sha_bytes(text_bytes),
        "tokenizer_file_sha256": sha_bytes(
            (ROOT / ".data/models/SmolLM2-135M-Instruct/tokenizer.json")
            .read_bytes()),
        "add_special_tokens": False,
        "window_size": width,
        "windows": [[int(v) for v in w] for w in windows],
        "window_meta": meta,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=str(
        ROOT / "models/smollm2_v3_input_plan.json"))
    ap.add_argument("--model-dir", default=str(
        ROOT / ".data/models/SmolLM2-135M-Instruct"))
    ap.add_argument("--official-continuation-json", default="")
    ap.add_argument("--out", default=str(
        ROOT / "models/smollm2_v3_inputs.json"))
    ap.add_argument("--calibration-windows-out", default=str(
        ROOT / "models/smollm2_v3_ppl_calibration_windows.json"))
    ap.add_argument("--acceptance-windows-out", default=str(
        ROOT / "models/smollm2_v3_ppl_acceptance_windows.json"))
    args = ap.parse_args(argv)
    from transformers import AutoTokenizer
    plan = json.loads(Path(args.plan).read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    cal = [{
        "id": item["id"], "text": item["text"],
        "token_ids": tokenize(tokenizer, item["text"]),
    } for item in plan["calibration_prompts"]]
    acc = [{
        "id": item["id"], "text": item["text"],
        "token_ids": tokenize(tokenizer, item["text"]),
    } for item in plan["acceptance_prompts"]]
    for item in cal + acc:
        item["token_seq_sha256"] = token_ids_sha(item["token_ids"])

    # Reject exact prompt reuse against all known earlier SmolLM2 inputs.
    old_sequences: set[tuple[int, ...]] = set()
    old_paths = [
        ROOT / "models/specs/smollm2_v2_20261028T094650Z.json",
        ROOT / "results/20261028T094650Z-smollm2/quality/instruction-prompts.json",
        ROOT / ".data/evidence/smollm2-20261028T094650Z/acceptance-reference/meta.json",
    ]
    for path in old_paths:
        if not path.is_file():
            continue
        obj = json.loads(path.read_text())
        if isinstance(obj, list):
            for row in obj:
                ids = row.get("token_ids") or row.get("prompt_ids")
                if ids:
                    old_sequences.add(tuple(int(v) for v in ids))
            continue
        if "calibration" in obj:
            old_sequences.add(tuple(obj["calibration"]["token_ids"]))
        for row in obj.get("prompts", obj.get("rows", [])):
            ids = row.get("token_ids") or row.get("prompt_ids")
            if ids:
                old_sequences.add(tuple(int(v) for v in ids))
        if "token_ids" in obj:
            old_sequences.add(tuple(int(v) for v in obj["token_ids"]))
    new_sequences = [tuple(x["token_ids"]) for x in cal + acc]
    if len(set(new_sequences)) != len(new_sequences):
        raise SystemExit("new calibration/acceptance prompt sequences collide")
    if set(new_sequences) & old_sequences:
        raise SystemExit("new prompt reuses an earlier development prompt")

    cw = windows_for(tokenizer, plan["calibration_text"]["source"],
                     int(plan["calibration_text"]["window_size"]),
                     int(plan["calibration_text"]["windows"]))
    aw = windows_for(tokenizer, plan["acceptance_text"]["source"],
                     int(plan["acceptance_text"]["window_size"]),
                     int(plan["acceptance_text"]["windows"]))
    cw_seqs = {tuple(w["token_ids"]) for w in cw["window_meta"]}
    aw_seqs = {tuple(w["token_ids"]) for w in aw["window_meta"]}
    if cw_seqs & aw_seqs:
        raise SystemExit("calibration/acceptance NLL windows reuse identity")
    for old in (ROOT / ".data/evidence/smollm2-20261028T094650Z/official/"
                "ppl-calibration-windows.json",
                ROOT / ".data/evidence/smollm2-20261028T094650Z/official/"
                "ppl-acceptance-windows.json"):
        if old.is_file():
            obj = json.loads(old.read_text())
            old_windows = {tuple(int(v) for v in w)
                           for w in obj.get("windows", [])}
            if (cw_seqs | aw_seqs) & old_windows:
                raise SystemExit("new NLL window reuses an earlier text window")

    numeric_id = plan["numeric_acceptance_prompt_id"]
    numeric = next(x for x in acc if x["id"] == numeric_id)
    continuation = None
    if args.official_continuation_json:
        off = json.loads(Path(args.official_continuation_json).read_text())
        if off.get("prompt_ids") != numeric["token_ids"]:
            raise SystemExit("official continuation prompt ids differ")
        generated = [int(v) for v in off.get("generated_ids", [])]
        need = int(plan["teacher_forcing_max_tokens"])
        if len(generated) < need:
            raise SystemExit(
                f"official continuation has {len(generated)} tokens, need {need}")
        continuation = generated[:need]
    result = {
        "schema_version": 1,
        "kind": "smollm2-v3-input-identity",
        "plan_path": str(Path(args.plan).resolve()),
        "plan_sha256": sha_bytes(Path(args.plan).read_bytes()),
        "model_id": plan["model_id"],
        "revision": plan["revision"],
        "tokenizer_file_sha256": sha_bytes(
            (Path(args.model_dir) / "tokenizer.json").read_bytes()),
        "calibration_prompts": cal,
        "acceptance_prompts": acc,
        "numeric_acceptance_prompt_id": numeric_id,
        "numeric_acceptance_prompt_ids": numeric["token_ids"],
        "teacher_forcing_continuation_ids": continuation,
        "teacher_forcing_token_ids": (
            numeric["token_ids"] + continuation if continuation else None),
        "incremental_continuation_ids": (
            continuation[:int(plan["incremental_continuation_tokens"])]
            if continuation else None),
        "calibration_nll_windows": cw,
        "acceptance_nll_windows": aw,
        "boundary_cases": plan["boundary_cases"],
        "policy": plan["policy"],
        "rank_policy": plan.get("rank_policy"),
    }
    body = dict(result)
    result["frozen_sha256"] = canonical_sha(body)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    Path(args.calibration_windows_out).write_text(
        json.dumps(cw, indent=1, sort_keys=True) + "\n")
    Path(args.acceptance_windows_out).write_text(
        json.dumps(aw, indent=1, sort_keys=True) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "frozen_sha256": result["frozen_sha256"],
        "calibration_prompts": len(cal),
        "acceptance_prompts": len(acc),
        "teacher_forcing_continuation": len(continuation) if continuation else 0,
        "calibration_scored_tokens": 5 * 255,
        "acceptance_scored_tokens": 5 * 255,
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
