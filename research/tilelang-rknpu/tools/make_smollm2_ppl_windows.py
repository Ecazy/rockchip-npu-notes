#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tokenize a fixed text source into non-overlapping SmolLM2 NLL windows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=".data/models/SmolLM2-135M-Instruct")
    ap.add_argument("--source", required=True)
    ap.add_argument("--window-size", type=int, default=256)
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    from transformers import AutoTokenizer
    text_bytes = Path(args.source).read_bytes()
    text = text_bytes.decode("utf-8")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    need = args.window_size * args.windows
    if len(ids) < need:
        raise SystemExit(f"source has {len(ids)} tokens, need {need}")
    windows = [ids[i:i + args.window_size]
               for i in range(0, need, args.window_size)]
    result = {
        "kind": "smollm2-nll-windows",
        "model_dir": args.model_dir,
        "tokenizer_file_sha256": sha_bytes(
            (Path(args.model_dir) / "tokenizer.json").read_bytes()),
        "source": args.source,
        "source_sha256": sha_bytes(text_bytes),
        "add_special_tokens": False,
        "tokenization": "AutoTokenizer(text).input_ids",
        "window_size": args.window_size,
        "windows": windows,
    }
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"source": args.source,
                      "source_sha256": result["source_sha256"],
                      "tokens": len(ids),
                      "windows": [len(w) for w in windows],
                      "scored_tokens": args.windows * (args.window_size - 1)},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
