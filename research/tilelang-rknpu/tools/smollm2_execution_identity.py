#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Create the expected effective execution identity for a SmolLM board run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--policy", default="")
    ap.add_argument("--lib", required=True)
    ap.add_argument("--lib-sha256", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    package = Path(args.package).resolve()
    pm = json.loads((package / "manifest.json").read_text())
    identity = {
        "schema_version": 1,
        "kind": "smollm2-effective-execution-identity",
        "base_package": {
            "manifest_sha256": sha(package / "manifest.json"),
            "weights_sha256": sha(package / pm["weights"]["path"]),
        },
        "profile_hash": pm["compiler"]["profile_hash"],
        "numeric_mode": pm["compiler"]["numeric_mode"],
        "plan_library_sha256": (args.lib_sha256 or
                                sha(Path(args.lib).resolve())),
        "overrides": {},
        "schedule_policy_sha256": None,
    }
    if args.policy:
        policy_path = Path(args.policy).resolve()
        policy = json.loads(policy_path.read_text())
        if policy.get("kind") != "smollm2-plan-overrides-v1":
            raise SystemExit("unsupported schedule policy kind")
        identity["schedule_policy_sha256"] = sha(policy_path)
        for key, value in policy["overrides"].items():
            base = pm["plans"].get(key)
            if base is None:
                raise SystemExit(f"override key {key!r} is not in package")
            out = {
                "replacement_kind": value.get("replacement_kind"),
                "base_plan_hash": value.get("base_plan_hash"),
                "plan_hash": value.get("plan_hash"),
                "plan_sha256": value.get("plan_sha256"),
                "task_count": value.get("task_count"),
                "cache_key": value.get("cache_key"),
            }
            if out["base_plan_hash"] != base["plan_hash"]:
                raise SystemExit(f"override base hash mismatch for {key!r}")
            identity["overrides"][key] = out
    Path(args.out).write_text(json.dumps(identity, indent=1, sort_keys=True))
    print(json.dumps({"out": args.out, "sha256": sha(Path(args.out)),
                      "overrides": sorted(identity["overrides"])}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
