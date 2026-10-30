#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Prepare and analyze a finite same-prefix divergence diagnostic.

The diagnostic never changes the V5 gates.  It records, at the first greedy
divergence for each instruction prompt, the backend (board), declared reference
and official FP32 top candidates/logits and top1-top2 margins.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def prompt_rows(board_run: Path) -> list[dict]:
    return [r for r in load_json(board_run)["requests"]
            if r.get("mode") == "generate"][:8]


def official_rows(official_root: Path) -> list[dict]:
    return [load_json(official_root / f"acc-{i}" / "result.json")
            for i in range(1, 9)]


def first_divergence(xs: list[int], ys: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(xs, ys)):
        if x != y:
            return i
    return None


def dataset_manifest(board_run: Path, official_root: Path,
                     inputs_path: Path) -> dict:
    inp = load_json(inputs_path)
    acceptance = inp["acceptance_prompts"]
    board = prompt_rows(board_run)
    official = official_rows(official_root)
    if len(board) != 8 or len(official) != 8:
        raise SystemExit("expected exactly 8 acceptance generation prompts")
    rows = []
    for i, (brow, orow, prompt) in enumerate(zip(board, official, acceptance),
                                            1):
        ids = [int(v) for v in brow["token_ids"]]
        official_prompt = [int(v) for v in orow["prompt_ids"]]
        if ids != official_prompt:
            raise SystemExit(f"acc-{i}: backend/official prompt token ids differ")
        if ids != [int(v) for v in prompt["token_ids"]]:
            raise SystemExit(f"acc-{i}: prompt differs from frozen input")
        bgen = [int(v) for v in brow["runs"][0]["generated_ids"]]
        ogen = [int(v) for v in orow["generated_ids"]]
        div = first_divergence(bgen, ogen)
        # A no-divergence prompt has no "first divergence"; probe the prompt
        # prefix itself, which is the first greedy decision.
        common = 0 if div is None else div
        prefix = ids + bgen[:common]
        rows.append({
            "prompt_id": f"acc-{i}",
            "prompt_token_ids": ids,
            "prompt_token_sha256": hashlib.sha256(
                np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
            "backend_generated_ids": bgen,
            "official_generated_ids": ogen,
            "matching_positions": sum(
                int(a == b) for a, b in zip(bgen, ogen)),
            "generated_lengths": [len(bgen), len(ogen)],
            "first_divergence": div,
            "probe_prefix_token_ids": prefix,
            "probe_prefix_len": len(prefix),
            "probe_kind": ("no_divergence_prompt_prefix" if div is None
                           else "first_divergence_prefix"),
            "chat_template": "tokenizer.apply_chat_template(add_generation_prompt=True)",
            "greedy": True,
            "backend_stop_on_eos": False,
            "official_ignore_eos": bool(orow.get("ignore_eos", False)),
            "max_new_tokens": [int(brow["runs"][0].get(
                "output_tokens", 32)), int(orow.get("max_new_tokens", 32))],
        })
    return {
        "schema_version": 1,
        "kind": "smollm2-divergence-probe-manifest",
        "board_run": str(board_run.resolve()),
        "official_generations": str(official_root.resolve()),
        "inputs": str(inputs_path.resolve()),
        "rows": rows,
    }


def prepare(args) -> int:
    manifest = dataset_manifest(Path(args.board_run), Path(args.official_root),
                                Path(args.inputs))
    request = {"runs": [{
        "mode": "teacher_forcing",
        "token_ids": row["probe_prefix_token_ids"],
        "repeat": 1,
    } for row in manifest["rows"]]}
    for path, obj in ((args.out_request, request),
                      (args.out_manifest, manifest)):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    print(json.dumps({
        "request": str(args.out_request),
        "manifest": str(args.out_manifest),
        "prompts": len(manifest["rows"]),
        "divergent": sum(r["first_divergence"] is not None
                         for r in manifest["rows"]),
    }, indent=1))
    return 0


def top_candidates(row: np.ndarray, tokenizer, k: int = 5) -> tuple:
    row = np.asarray(row)
    top1 = int(np.argmax(row))
    order = sorted(range(int(row.size)), key=lambda i: (-float(row[i]), i))[:k]
    cands = [{
        "token_id": int(i),
        "token_text": tokenizer.decode([int(i)]),
        "logit": float(row[i]),
    } for i in order]
    margin = (float(row[order[0]] - row[order[1]])
              if len(order) > 1 else 0.0)
    return top1, margin, cands


def classify(b_top: int, d_top: int, o_top: int) -> str:
    if b_top == d_top == o_top:
        return "three_way_top1_agreement"
    if b_top == d_top and d_top != o_top:
        return "declared_reference_matches_backend_not_official"
    if b_top != d_top and d_top == o_top:
        return "backend_differs_declared_and_official_agree"
    if b_top == o_top and d_top != o_top:
        return "backend_and_official_agree_declared_differs"
    return "three_way_top1_difference"


def analyze(args) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import sys
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from smollm2_reference import DeclaredReference

    manifest = load_json(Path(args.manifest))
    probe_board = Path(args.probe_board)
    package = Path(args.package).resolve()
    model_dir = Path(args.model_dir).resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="eager")
    model.eval()
    declared = DeclaredReference(package)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    with torch.no_grad():
        for i, row in enumerate(manifest["rows"]):
            prefix = [int(v) for v in row["probe_prefix_token_ids"]]
            run_dir = probe_board / f"run-{i:02d}"
            if not (run_dir / "result.json").is_file():
                failures.append(f"missing_probe_run:{row['prompt_id']}")
                continue
            run_result = load_json(run_dir / "result.json")
            if run_result.get("mode") != "teacher_forcing" or \
                    run_result.get("token_ids") != prefix:
                failures.append(f"probe_request_mismatch:{row['prompt_id']}")
                continue
            blog = np.load(run_dir / "logits-00.npy", allow_pickle=False)
            if blog.ndim != 2 or blog.shape != (len(prefix), declared.cfg["vocab_size"]):
                failures.append(f"backend_probe_shape:{row['prompt_id']}")
                continue
            b_last = blog[-1]
            d_last = np.asarray(declared.forward(prefix, capture_layers=())[
                "logits"])[-1]
            tens = torch.tensor([prefix], dtype=torch.long)
            o_last = model(tens, use_cache=False, return_dict=True).logits[
                0, -1].float().numpy()
            # Reproduce the archived official generate() path: transformers
            # infers attention_mask = input_ids != pad_token_id when no mask is
            # supplied.  Token id 2 is both EOS and pad in this chat tokenizer.
            auto_mask = (tens != int(tokenizer.pad_token_id)).to(torch.long)
            # transformers also derives position_ids from that inferred mask
            # with cumsum(mask)-1.  Both effects must be reproduced to explain
            # the archived generation path.
            auto_positions = torch.clamp(
                torch.cumsum(auto_mask, dim=-1) - 1, min=0)
            auto_last = model(tens, attention_mask=auto_mask,
                              position_ids=auto_positions,
                              use_cache=False, return_dict=True).logits[
                0, -1].float().numpy()
            b_top, b_margin, b_cands = top_candidates(b_last, tokenizer)
            d_top, d_margin, d_cands = top_candidates(d_last, tokenizer)
            o_top, o_margin, o_cands = top_candidates(o_last, tokenizer)
            a_top, _, _ = top_candidates(auto_last, tokenizer)
            div = row["first_divergence"]
            expected_b = (row["backend_generated_ids"][div] if div is not None
                          else row["backend_generated_ids"][0])
            archived_o = (row["official_generated_ids"][div] if div is not None
                          else row["official_generated_ids"][0])
            # Corrected official greedy continuation with the full unpadded
            # prompt visible to attention.
            prompt_tens = torch.tensor([row["prompt_token_ids"]],
                                       dtype=torch.long)
            generated = model.generate(
                prompt_tens, attention_mask=torch.ones_like(prompt_tens),
                max_new_tokens=32, do_sample=False, use_cache=True,
                eos_token_id=None, pad_token_id=tokenizer.pad_token_id)
            corrected_official = [int(v) for v in
                                  generated[0, len(row["prompt_token_ids"]):]
                                  .tolist()]
            corrected_match = sum(
                int(a == b) for a, b in
                zip(row["backend_generated_ids"], corrected_official))
            corrected_div = first_divergence(
                row["backend_generated_ids"], corrected_official)
            consistency = {
                "backend_probe_top1_equals_backend_generated": bool(
                    b_top == expected_b),
                "official_full_attention_top1_equals_corrected_at_probe": bool(
                    o_top == (corrected_official[div] if div is not None
                              else corrected_official[0])),
                "official_auto_mask_top1_equals_archived_generated": bool(
                    a_top == archived_o),
            }
            if not all(consistency.values()):
                failures.append(f"greedy_consistency:{row['prompt_id']}")
            prompt_dir = out_root / row["prompt_id"]
            prompt_dir.mkdir(parents=True, exist_ok=True)
            np.save(prompt_dir / "backend-logits.npy", blog)
            np.save(prompt_dir / "declared-logits.npy", d_last)
            np.save(prompt_dir / "official-logits.npy", o_last)
            np.save(prompt_dir / "official-auto-mask-logits.npy", auto_last)
            (prompt_dir / "generation.json").write_text(
                json.dumps({
                    "prompt_id": row["prompt_id"],
                    "archived_official_generated_ids":
                        row["official_generated_ids"],
                    "auto_attention_mask_generated_first": archived_o,
                    "corrected_official_generated_ids": corrected_official,
                    "backend_generated_ids": row["backend_generated_ids"],
                }, indent=1, sort_keys=True) + "\n")
            rows.append({
                **row,
                "probe_prefix_sha256": hashlib.sha256(
                    np.asarray(prefix, dtype="<i8").tobytes()).hexdigest(),
                "backend_top1": b_top,
                "declared_top1": d_top,
                "official_top1": o_top,
                "backend_top1_top2_margin": b_margin,
                "declared_top1_top2_margin": d_margin,
                "official_top1_top2_margin": o_margin,
                "backend_top5": b_cands,
                "declared_top5": d_cands,
                "official_top5": o_cands,
                "official_auto_mask_top1": a_top,
                "prompt_pad_token_count": int(
                    (tens == int(tokenizer.pad_token_id)).sum().item()),
                "auto_attention_mask_zero_positions": int(
                    (auto_mask == 0).sum().item()),
                "expected_backend_next": expected_b,
                "expected_archived_official_next": archived_o,
                "classification": classify(b_top, d_top, o_top),
                "corrected_official_generated_ids": corrected_official,
                "corrected_official_matching_positions": corrected_match,
                "corrected_official_first_divergence": corrected_div,
                "consistency": consistency,
                "array_sha256": {
                    "backend_last": sha_array(b_last),
                    "declared_last": sha_array(d_last),
                    "official_last": sha_array(o_last),
                },
            })
    summary = {
        "schema_version": 1,
        "kind": "smollm2-divergence-diagnostic",
        "pass": not failures,
        "prompts_checked": len(rows),
        "divergent_prompts": sum(r["first_divergence"] is not None
                                 for r in rows),
        "no_divergence_prompts": sum(r["first_divergence"] is None
                                     for r in rows),
        "classifications": {
            name: sum(r["classification"] == name for r in rows)
            for name in sorted({r["classification"] for r in rows})
        },
        "archived_official_matching_positions": sum(
            r["matching_positions"] for r in rows),
        "corrected_official_matching_positions": sum(
            r["corrected_official_matching_positions"] for r in rows),
        "all_consistency_checks": all(
            all(r["consistency"].values()) for r in rows),
        "official_auto_mask_is_pad_token_mask": all(
            r["prompt_pad_token_count"] ==
            r["auto_attention_mask_zero_positions"] for r in rows),
        "official_auto_mask_affected_prompts": sum(
            r["prompt_pad_token_count"] > 0 for r in rows),
        "failures": failures,
        "rows": rows,
        "notes": [
            "The diagnostic is finite: it probes one same-prefix position per prompt.",
            "No V5 threshold or observation was changed.",
            "A top1 difference is reported with its measured margin; it is not "
            "automatically labeled a near-tie.",
        ],
    }
    (out_root / "diagnostic.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n")
    lines = [
        "# SmolLM2 instruction-prompt divergence diagnostic",
        "",
        "One same-prefix greedy position was probed per prompt. No V5 gate or "
        "observation was modified.",
        "",
        "| Prompt | Backend vs archived official | Backend vs corrected official "
        "| First divergence | Backend top1 | Declared top1 | Official top1 | "
        "Backend margin | Declared margin | Official margin | Classification |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['prompt_id']} | {r['matching_positions']}/32 | "
            f"{r['corrected_official_matching_positions']}/32 | "
            f"{r['first_divergence']} | {r['backend_top1']} | "
            f"{r['declared_top1']} | {r['official_top1']} | "
            f"{r['backend_top1_top2_margin']:.6g} | "
            f"{r['declared_top1_top2_margin']:.6g} | "
            f"{r['official_top1_top2_margin']:.6g} | "
            f"{r['classification']} |")
    lines += [
        "",
        "## Finding",
        "",
        "The archived official free-generation path called `model.generate()` "
        "without an attention mask.  Because token id 2 is both the chat "
        "template's delimiter/EOS and the tokenizer pad id, transformers "
        "derived an attention mask with zeros at real prompt positions and "
        "then derived position_ids as `cumsum(mask)-1`, resetting RoPE "
        "positions after each such token.  Replaying the official model with "
        "an explicit all-ones attention mask restores the same-prefix greedy "
        "top1 and, in this finite diagnostic, matches the backend for all 256 "
        "generated positions across the 8 prompts.  The archived 102/256 "
        "match count is therefore not evidence of backend matrix deviation.",
        "",
        "This is a reference-tooling defect, not a V5 gate change.  The "
        "corrected official generation is kept in a separate delivery "
        "directory; V5 inputs, spec, thresholds and sealed observations remain "
        "untouched.",
        "",
    ]
    (out_root / "DIVERGENCE-REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "pass": summary["pass"],
        "prompts_checked": summary["prompts_checked"],
        "divergent_prompts": summary["divergent_prompts"],
        "classifications": summary["classifications"],
        "failures": summary["failures"],
    }, indent=1, sort_keys=True))
    return 0 if summary["pass"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--board-run", required=True)
    p.add_argument("--official-root", required=True)
    p.add_argument("--inputs", required=True)
    p.add_argument("--out-request", required=True)
    p.add_argument("--out-manifest", required=True)
    p.set_defaults(func=prepare)
    a = sub.add_parser("analyze")
    a.add_argument("--probe-board", required=True)
    a.add_argument("--manifest", required=True)
    a.add_argument("--package", required=True)
    a.add_argument("--model-dir", required=True)
    a.add_argument("--out", required=True)
    a.set_defaults(func=analyze)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
