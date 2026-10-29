#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fetch the fixed SmolLM2-135M-Instruct revision and verify file identities."""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
PATTERNS = [
    "config.json", "generation_config.json", "model.safetensors",
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
    "vocab.json", "merges.txt", "README.md",
]
EXPECTED = {
    "config.json": "8eb740e8bbe4cff95ea7b4588d17a2432deb16e8075bc5828ff7ba9be94d982a",
    "generation_config.json": "87b916edaaab66b3899b9d0dd0752727dff6666686da0504d89ae0a6e055a013",
    "model.safetensors": "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c",
    "special_tokens_map.json": "2b7379f3ae813529281a5c602bc5a11c1d4e0a99107aaa597fe936c1e813ca52",
    "tokenizer.json": "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
    "tokenizer_config.json": "4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df",
    "vocab.json": "82b84012e3add4d01d12ba14442026e49b8cbbaead1f79ecf3d919784f82dc79",
    "merges.txt": "0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510",
    "README.md": "4f97533ad95b1b2fea15fbc075c01b94578ebdd7c8138888fa43fa3abd530dc4",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".data/models/SmolLM2-135M-Instruct")
    ap.add_argument("--endpoint", default=os.environ.get(
        "HF_ENDPOINT", "https://hf-mirror.com"))
    args = ap.parse_args(argv)
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise SystemExit(f"huggingface_hub is required: {exc}")
    os.environ.setdefault("HF_ENDPOINT", args.endpoint)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    snapshot_download(MODEL_ID, revision=REVISION, local_dir=str(out),
                      allow_patterns=PATTERNS)
    bad = []
    for name, expected in EXPECTED.items():
        path = out / name
        if not path.is_file():
            bad.append(f"{name}: missing")
        elif sha(path) != expected:
            bad.append(f"{name}: hash mismatch")
    if bad:
        raise SystemExit("fixed model identity failure: " + "; ".join(bad))
    print(f"model={MODEL_ID} revision={REVISION} dir={out} identity=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
