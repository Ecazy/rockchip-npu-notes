#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare two projection files produced by ``tools/refactor_plan_projection.py``.

Output is the ``comparison.json`` the refactor spec asks for: per-case identity
result, the explicit ignore list, and any unexplained execution difference.

The comparison is deliberately strict.  ``plan`` and ``descriptor`` must be equal
field for field; only these paths are allowed to differ, and only for the reasons
named next to them:

  tests may not add entries to this list without a written justification in
  ``refactor-specific.md`` section 15.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# (path, reason).  Only identity/provenance fields may appear here.
ALLOWED_DIFFERENCES: dict[str, str] = {
    "environment.backend_module": "the package location differs by definition",
    "environment.executable": "informational",
    "environment.platform": "informational",
}


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key in sorted(obj):
            flat.update(_flatten(obj[key], f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            flat.update(_flatten(item, f"{prefix}[{index}]"))
    else:
        flat[prefix] = obj
    return flat


def _diff(left: Any, right: Any, limit: int = 40) -> list[dict[str, Any]]:
    a, b = _flatten(left), _flatten(right)
    out: list[dict[str, Any]] = []
    for key in sorted(set(a) | set(b)):
        if key not in a:
            out.append({"path": key, "left": None, "right": b[key]})
        elif key not in b:
            out.append({"path": key, "left": a[key], "right": None})
        elif a[key] != b[key]:
            out.append({"path": key, "left": a[key], "right": b[key]})
        if len(out) >= limit:
            break
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    baseline = json.loads(Path(args.baseline).read_text())
    current = json.loads(Path(args.current).read_text())

    report: dict[str, Any] = {
        "allowed_differences": ALLOWED_DIFFERENCES,
        "cases": {},
        "negatives": {},
        "identity_differences": [],
    }

    for name in sorted(set(baseline["cases"]) | set(current["cases"])):
        left = baseline["cases"].get(name)
        right = current["cases"].get(name)
        if left is None or right is None:
            report["cases"][name] = {"identical": False,
                                     "reason": "case missing on one side"}
            continue
        if left.get("ok") != right.get("ok"):
            report["cases"][name] = {"identical": False,
                                     "reason": "compile outcome differs",
                                     "left": left.get("error"),
                                     "right": right.get("error")}
            continue
        if not left.get("ok"):
            same = left.get("error") == right.get("error")
            report["cases"][name] = {"identical": same, "rejected": True,
                                     "error": right.get("error")}
            continue
        plan_diff = _diff(left["projection"]["plan"], right["projection"]["plan"])
        desc_diff = _diff(left["projection"]["descriptor"],
                          right["projection"]["descriptor"])
        for item in plan_diff + desc_diff:
            if item["path"] in ALLOWED_DIFFERENCES:
                report["identity_differences"].append(
                    {"case": name, "path": item["path"],
                     "reason": ALLOWED_DIFFERENCES[item["path"]]})
        plan_diff = [d for d in plan_diff if d["path"] not in ALLOWED_DIFFERENCES]
        desc_diff = [d for d in desc_diff if d["path"] not in ALLOWED_DIFFERENCES]
        report["cases"][name] = {
            "identical": not plan_diff and not desc_diff,
            "plan_hash_baseline": left["plan_hash"],
            "plan_hash_current": right["plan_hash"],
            "plan_differences": plan_diff,
            "descriptor_differences": desc_diff,
            "counts": {
                "actions": len(right["projection"]["plan"]["actions"]),
                "submits": len(right["projection"]["plan"]["submits"]),
                "tasks": right["projection"]["plan"]["accounting"]["task_count"],
                "publish": len(right["projection"]["plan"]["publish"]),
            },
        }

    for name in sorted(set(baseline["negatives"]) | set(current["negatives"])):
        left = baseline["negatives"].get(name)
        right = current["negatives"].get(name)
        report["negatives"][name] = {
            "identical": left == right,
            "baseline": left,
            "current": right,
        }

    identical = [n for n, c in report["cases"].items() if c["identical"]]
    differing = [n for n, c in report["cases"].items() if not c["identical"]]
    neg_ok = [n for n, c in report["negatives"].items() if c["identical"]]
    neg_bad = [n for n, c in report["negatives"].items() if not c["identical"]]
    report["summary"] = {
        "cases": len(report["cases"]),
        "identical": len(identical),
        "differing": differing,
        "negatives": len(report["negatives"]),
        "negatives_identical": len(neg_ok),
        "negatives_differing": neg_bad,
        "unexplained_execution_differences": [
            n for n in differing
            if report["cases"][n].get("plan_differences")
            or report["cases"][n].get("descriptor_differences")
        ],
    }
    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"cases={len(report['cases'])} identical={len(identical)} "
          f"differing={len(differing)} negatives={len(neg_ok)}/"
          f"{len(report['negatives'])} -> {args.out}")
    if differing or neg_bad:
        for name in differing[:5]:
            print(f"  DIFF {name}: {report['cases'][name]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
