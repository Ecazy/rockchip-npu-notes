#!/usr/bin/env python3
"""Summarize retained battr output_sync attribution samples."""
import argparse
import json
import statistics
from pathlib import Path

FIELDS = ("validate", "pack", "emit", "command", "submit_alloc",
          "submit_wait", "output_sync", "download", "guard", "publish",
          "alloc", "total", "pack_layout")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cases = {}
    for case in ("qk_b8", "pv_b8"):
        cases[case] = {}
        for arm in ("fast", "reference"):
            path = args.profile_root / case / f"{arm}.stdout"
            rows, counts = [], []
            for line in path.read_text().splitlines():
                if line.startswith("ATTR_US "):
                    rows.append([float(x) for x in line.split()[1:]])
                elif line.startswith("ATTR_COUNTS "):
                    counts.append([int(x) for x in line.split()[1:]])
            if not rows or any(len(row) != len(FIELDS) for row in rows):
                raise SystemExit(f"bad ATTR_US sample: {path}")
            cases[case][arm] = {
                "n": len(rows),
                "output_sync_us_median": statistics.median(row[6] for row in rows),
                "output_sync_us_min": min(row[6] for row in rows),
                "output_sync_us_max": max(row[6] for row in rows),
                "stage_medians_us": {
                    key: statistics.median(row[i] for row in rows)
                    for i, key in enumerate(FIELDS)
                    if key in ("submit_wait", "download", "guard", "total")},
                "ATTR_COUNTS_unique": [list(row) for row in sorted(set(map(tuple, counts)))],
                "exit": int((args.profile_root / f"{case}-{arm}.exit").read_text().strip()),
            }
    result = {"kind": "output-sync-attribution", "fields": list(FIELDS), "cases": cases}
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
