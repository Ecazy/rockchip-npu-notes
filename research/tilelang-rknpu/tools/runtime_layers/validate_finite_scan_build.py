#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Verify the finite-scan A/B build differs only by the reference macro."""
import argparse, json, re
from pathlib import Path

MACRO = "-DRK_PLAN_FINITE_SCAN_REFERENCE"


def cache(path):
    out = {}
    for line in path.read_text().splitlines():
        if line and not line.startswith(("#", "//")) and ":" in line and "=" in line:
            lhs, value = line.split("=", 1)
            out[lhs.split(":", 1)[0]] = value
    return out


def flags(path):
    out = {}
    for line in path.read_text().splitlines():
        m = re.match(r"(C_DEFINES|C_INCLUDES|C_FLAGS) = (.*)", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def commands(path):
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    p = args.artifacts
    fc = cache(p / "fast-CMakeCache.txt")
    rc = cache(p / "reference-CMakeCache.txt")
    ff = flags(p / "fast-runtime-flags.make")
    rf = flags(p / "reference-runtime-flags.make")
    fcmds = commands(p / "fast-compile_commands.json")
    rcmds = commands(p / "reference-compile_commands.json")
    assert MACRO not in " ".join(ff.values()), "fast target unexpectedly defines reference macro"
    assert MACRO in " ".join(rf.values()), "reference target lacks reference macro"
    assert fc.get("CMAKE_C_FLAGS", "") == "", "fast CMAKE_C_FLAGS must be empty"
    assert rc.get("CMAKE_C_FLAGS", "") == MACRO, "reference CMAKE_C_FLAGS must contain only reference macro"
    for key in ("CMAKE_BUILD_TYPE", "CMAKE_C_COMPILER", "CMAKE_EXPORT_COMPILE_COMMANDS", "UPSTREAM_ROOT"):
        assert fc.get(key) == rc.get(key), f"CMake setting differs: {key}"
    for key in ("C_DEFINES", "C_INCLUDES", "C_FLAGS"):
        normalized = re.sub(r"\s*-DRK_PLAN_FINITE_SCAN_REFERENCE\b", "", rf.get(key, "")).strip()
        assert normalized == ff.get(key, "").strip(), f"runtime target flags differ beyond macro: {key}"
    assert len(fcmds) == len(rcmds), "compile command counts differ"
    for index, (fast, reference) in enumerate(zip(fcmds, rcmds)):
        assert fast["file"] == reference["file"], f"compile source differs at entry {index}"
        assert MACRO not in fast["command"], f"fast command unexpectedly defines macro at entry {index}"
        assert MACRO in reference["command"], f"reference command lacks macro at entry {index}"
        normalized = " ".join(re.sub(r"\s*-DRK_PLAN_FINITE_SCAN_REFERENCE\b", "", reference["command"]).split())
        assert normalized == " ".join(fast["command"].split()), f"compile command differs beyond macro at entry {index}"
    report = {
        "status": "PASS",
        "assertions": {
            "reference_macro_present": True, "fast_macro_absent": True,
            "runtime_flags_match_after_macro_removal": True,
            "all_compile_commands_match_after_macro_removal": True,
            "CMake_build_identity_matches": True,
        },
        "fast_CMAKE_C_FLAGS": fc["CMAKE_C_FLAGS"],
        "reference_CMAKE_C_FLAGS": rc["CMAKE_C_FLAGS"],
        "fast_runtime_C_FLAGS": ff["C_FLAGS"],
        "reference_runtime_C_FLAGS": rf["C_FLAGS"],
        "compile_commands": len(fcmds),
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
