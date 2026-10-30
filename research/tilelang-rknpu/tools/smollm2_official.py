#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Official Hugging Face SmolLM2-135M-Instruct reference entry point.

This is a quality/semantic reference only.  It is never imported or called by
the RKNPU runtime path.  The recorded dtype is float32; the backend package is
FP16 with declared staged K-tile writeback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
    ap.add_argument("--model-dir", default=".data/models/SmolLM2-135M-Instruct")
    ap.add_argument("--mode", choices=("logits", "generate", "score"),
                    required=True)
    ap.add_argument("--ids-json")
    ap.add_argument("--windows-json")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.float32,
        attn_implementation="eager")
    model.eval()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "kind": "smollm2-official-reference",
        "model_dir": args.model_dir,
        "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
        "dtype": "float32",
        "attn_implementation": "eager",
        "deterministic": True,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "config_sha256": sha(Path(args.model_dir) / "config.json"),
        "weights_sha256": sha(Path(args.model_dir) / "model.safetensors"),
        "tokenizer_sha256": sha(Path(args.model_dir) / "tokenizer.json"),
    }
    with torch.no_grad():
        if args.mode == "logits":
            if not args.ids_json:
                raise SystemExit("--ids-json is required for logits")
            raw = json.loads(Path(args.ids_json).read_text())
            ids = raw["token_ids"] if isinstance(raw, dict) else raw
            t = torch.tensor([ids], dtype=torch.long)
            result = model(t, use_cache=False, return_dict=True)
            logits = result.logits[0].float().numpy()
            np.save(out / "logits.npy", logits)
            meta.update({"token_ids": ids, "logits_shape": list(logits.shape),
                         "logits_sha256": hashlib.sha256(
                             logits.tobytes()).hexdigest()})
        elif args.mode == "generate":
            if not args.ids_json:
                raise SystemExit("--ids-json is required for generate")
            raw = json.loads(Path(args.ids_json).read_text())
            ids = raw["token_ids"] if isinstance(raw, dict) else raw
            t = torch.tensor([ids], dtype=torch.long)
            # The tokenizer's pad token is also an in-vocabulary chat
            # sentinel (id 2).  Never let generate() derive an attention mask
            # from ``input_ids != pad_token_id``: that would mark real prompt
            # tokens as padding.  The prompt is a single unpadded sequence.
            attention_mask = torch.ones_like(t)
            generated = model.generate(
                t, attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens, do_sample=False,
                use_cache=True,
                eos_token_id=None if args.ignore_eos else
                tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id)
            gen = generated[0, len(ids):].tolist()
            meta.update({"prompt_ids": ids, "generated_ids": gen,
                         "text": tokenizer.decode(gen),
                         "max_new_tokens": args.max_new_tokens,
                         "ignore_eos": bool(args.ignore_eos),
                         "attention_mask": "explicit_all_ones",
                         "pad_token_id": tokenizer.pad_token_id})
        else:
            if not args.windows_json:
                raise SystemExit("--windows-json is required for score")
            windows = json.loads(Path(args.windows_json).read_text())
            nlls = []
            rows = []
            correct = 0
            total = 0
            for i, ids in enumerate(windows["windows"]):
                t = torch.tensor([ids], dtype=torch.long)
                result = model(t, use_cache=False, return_dict=True)
                logits = result.logits[0, :-1].float()
                labels = t[0, 1:]
                nll = -torch.log_softmax(logits, dim=-1)[
                    torch.arange(labels.numel()), labels]
                nlls.append(nll.numpy())
                top1 = int((logits.argmax(-1) == labels).sum())
                correct += top1
                total += int(labels.numel())
                label_ids = labels.cpu().tolist()
                rows.append({
                    "window": i,
                    "token_ids": [int(v) for v in ids],
                    "token_seq_sha256": hashlib.sha256(
                        np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
                    "scored_label_ids": [int(v) for v in label_ids],
                    "scored_label_sha256": hashlib.sha256(
                        np.asarray(label_ids, dtype="<i8").tobytes()
                    ).hexdigest(),
                    "scored_tokens": int(labels.numel()),
                    "sum_nll": float(nll.sum()),
                    "mean_nll": float(nll.mean()),
                    "ppl": float(torch.exp(nll.mean())),
                    "top1_accuracy": top1 / int(labels.numel()),
                })
            nll = np.concatenate(nlls)
            np.save(out / "nll.npy", nll)
            meta.update({
                "source": windows["source"],
                "source_sha256": windows["source_sha256"],
                "window_size": windows["window_size"],
                "windows": rows,
                "scored_tokens": int(nll.size),
                "sum_nll": float(nll.sum()),
                "mean_nll": float(nll.mean()),
                "ppl": float(np.exp(nll.mean())),
                "top1_accuracy": correct / total,
                "nll_sha256": hashlib.sha256(nll.tobytes()).hexdigest(),
            })
    (out / "result.json").write_text(
        json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: meta[k] for k in ("kind", "dtype")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
