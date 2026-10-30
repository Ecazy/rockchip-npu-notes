#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fail-closed elementwise equivalence check for Host-weight caching.

The candidate is allowed to cache immutable package arrays, but must not alter
any Host action result, plan input/output, incremental KV result or sampled
token.  This tool compares two board evidence roots produced with the same
request and runtime package.
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


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def compare_arrays(name: str, left: np.ndarray, right: np.ndarray,
                   rows: list[dict]) -> None:
    la, ra = np.asarray(left), np.asarray(right)
    row = {
        "name": name,
        "left_shape": list(la.shape),
        "right_shape": list(ra.shape),
        "left_dtype": str(la.dtype),
        "right_dtype": str(ra.dtype),
        "left_sha256": array_sha(la),
        "right_sha256": array_sha(ra),
        "shape_equal": list(la.shape) == list(ra.shape),
        "dtype_equal": la.dtype == ra.dtype,
        "array_equal": bool(la.shape == ra.shape and la.dtype == ra.dtype and
                            np.array_equal(la, ra)),
    }
    rows.append(row)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    b = Path(args.baseline).resolve()
    c = Path(args.candidate).resolve()
    rows: list[dict] = []
    failures: list[str] = []
    for run_dir in sorted(p for p in b.glob("run-*") if p.is_dir()):
        rel = run_dir.relative_to(b)
        other = c / rel
        if not other.is_dir():
            failures.append(f"missing_candidate_run:{rel}")
            continue
        for path in sorted(run_dir.iterdir()):
            if not path.is_file():
                continue
            q = other / path.name
            if not q.is_file():
                failures.append(f"missing_candidate_file:{rel / path.name}")
                continue
            if path.suffix == ".npz":
                with np.load(path, allow_pickle=False) as za, \
                        np.load(q, allow_pickle=False) as zb:
                    if set(za.files) != set(zb.files):
                        failures.append(f"npz_key_mismatch:{rel / path.name}")
                        continue
                    for key in sorted(za.files):
                        compare_arrays(f"{rel / path.name}:{key}",
                                       za[key], zb[key], rows)
            elif path.suffix == ".npy":
                compare_arrays(str(rel / path.name),
                               np.load(path, allow_pickle=False),
                               np.load(q, allow_pickle=False), rows)
    for row in rows:
        if not row["array_equal"]:
            failures.append(f"array_mismatch:{row['name']}")
    # Generation requests do not write intermediate arrays for every token;
    # compare their sampled-token lists explicitly as well.
    for run_dir in sorted(p for p in b.glob("run-*") if p.is_dir()):
        rel = run_dir.relative_to(b)
        rb = json.loads((run_dir / "result.json").read_text())
        rc = json.loads((c / rel / "result.json").read_text())
        if rb.get("mode") != rc.get("mode"):
            failures.append(f"mode_mismatch:{rel}")
        if rb.get("token_ids") != rc.get("token_ids"):
            failures.append(f"request_token_mismatch:{rel}")
        for i, (x, y) in enumerate(zip(rb.get("runs", []), rc.get("runs", []))):
            if x.get("generated_ids") != y.get("generated_ids"):
                failures.append(f"generated_ids_mismatch:{rel}:{i}")
            if x.get("output_tokens") != y.get("output_tokens"):
                failures.append(f"output_tokens_mismatch:{rel}:{i}")
            if x.get("decode_forwards") != y.get("decode_forwards"):
                failures.append(f"decode_forwards_mismatch:{rel}:{i}")
    result = {
        "schema_version": 1,
        "kind": "smollm2-host-cache-equivalence",
        "pass": not failures,
        "baseline": str(b),
        "candidate": str(c),
        "numeric_arrays_compared": len(rows),
        "numeric_arrays_equal": sum(r["array_equal"] for r in rows),
        "generated_token_lists_checked": len(list(b.glob("run-*/result.json"))),
        "failures": failures,
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in
                      ("pass", "numeric_arrays_compared",
                       "numeric_arrays_equal",
                       "generated_token_lists_checked", "failures")},
                     indent=1, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
