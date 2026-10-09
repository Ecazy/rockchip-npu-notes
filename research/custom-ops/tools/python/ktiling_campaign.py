#!/usr/bin/env python3
"""K-tiling campaign coverage, integrity and kernel gate.

Every one of the 6 process configurations (core masks 1/2/4 x 2 repetitions)
must contain:
  * a probe.jsonl whose summary is PASS,
  * one-submit invocations covering the contract/raw/perf matrices with the
    expected task counts and external 16/64 B offsets,
  * an independent ktiling_gate.py oracle report over the witnesses,
  * a recorded zero exit status.

The aggregate is hashed into an immutable campaign.json; an existing manifest is
verified and never silently replaced.
"""
import hashlib
import json
from pathlib import Path
import re
import sys

REQUIRED_KS = {255, 256, 257, 511, 512, 513, 769, 1024, 2049, 4096}
REQUIRED_RAW = {"chan-k255", "chan-k256", "chan-k257", "chan-k511",
                "chan-k512", "chan-k4095", "cancel-k513", "prefix"}
LUT_ELEMENTS = 32768
K_TILE = 256


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def round_up(v, a):
    return (v + a - 1) // a * a


def tiles(k):
    return (k + K_TILE - 1) // K_TILE


def lut_chunks(m, n):
    rows = round_up(12 if m < 12 else m, 4)
    cols = round_up(n, 32)
    return (rows * cols + LUT_ELEMENTS - 1) // LUT_ELEMENTS


def expected_tasks(m, n, k, kind):
    nt = tiles(k)
    if kind == "raw":
        return nt
    if kind == "perf-unfused":
        return 2 * nt + lut_chunks(m, n)
    return nt + 1 + lut_chunks(m, n)


def percentile(samples, p):
    if not samples:
        return None
    s = sorted(samples)
    i = int(p * (len(s) - 1) + 0.5)
    return s[min(i, len(s) - 1)]


def read_jsonl(path, errors, name):
    records = []
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    except (OSError, ValueError) as exc:
        errors.append(f"{name}: unreadable probe.jsonl ({exc})")
    return records


def verify_process(directory, reps):
    """Return (stats, errors) for one core-MASK-run-REP directory."""
    errors = []
    records = read_jsonl(directory / "probe.jsonl", errors, directory.name)
    invocations = [r for r in records if r.get("event") == "invocation"]
    perf = [r for r in records if r.get("event") == "perf"]
    summary = [r for r in records if r.get("event") == "summary"]

    if len(summary) != 1 or summary[0].get("status") != "PASS":
        errors.append(f"{directory.name}: missing/passless summary")
    if not invocations:
        errors.append(f"{directory.name}: no invocation records")

    contract_ks = set()
    contract_patterns = set()
    raw_names = set()
    offsets = set()
    stresses = {}
    checks = submits = tasks = 0
    for rec in invocations:
        checks += 1
        kind = rec.get("kind")
        m, n, k = rec.get("m"), rec.get("n"), rec.get("k")
        if rec.get("submits") != 1:
            errors.append(f"{directory.name}: {kind} m{m} n{n} k{k} submits != 1")
        submits += rec.get("submits", 0)
        exp = expected_tasks(m, n, k, kind)
        if rec.get("tasks") != exp or rec.get("expected_tasks") != exp:
            errors.append(f"{directory.name}: {kind} m{m} n{n} k{k} tasks {rec.get('tasks')}/{rec.get('expected_tasks')} != {exp}")
        tasks += rec.get("tasks", 0)
        for flag in ("finite", "input_unchanged", "guards", "pass"):
            if rec.get(flag) is not True:
                errors.append(f"{directory.name}: {kind} m{m} n{n} k{k} {flag} not true")
        if m * n >= 16 and kind == "contract" and rec.get("unsaturated_frac", 0.0) < 0.5:
            errors.append(f"{directory.name}: {kind} m{m} n{n} k{k} unsaturated_frac {rec.get('unsaturated_frac')} < 0.5")
        if kind == "contract":
            contract_ks.add(k)
            contract_patterns.add(rec.get("pattern"))
            offsets.add(rec.get("offset_x"))
            offsets.add(rec.get("offset_y"))
            if rec.get("lut_chunks") != lut_chunks(m, n):
                errors.append(f"{directory.name}: contract m{m} n{n} k{k} lut_chunks wrong")
        elif kind == "raw":
            raw_names.add(rec.get("who"))
            offsets.add(rec.get("offset_x"))
        elif kind == "stress":
            if rec.get("cube") not in (True, False):
                errors.append(f"{directory.name}: stress entry missing")
            key=(m,n,k,rec.get("iteration"))
            if key in stresses:errors.append(f"{directory.name}: duplicate stress iteration")
            stresses[key]=rec.get("cube")
        else:
            errors.append(f"{directory.name}: unknown invocation kind {kind}")

    expected_stress={(m,n,k,i): bool(i%2) for m,n,k in ((29,45,769),(16,32,4096)) for i in range(8)}
    if stresses!=expected_stress:
        errors.append(f"{directory.name}: changing-input context coverage incomplete")
    if contract_ks != REQUIRED_KS:
        errors.append(f"{directory.name}: contract K coverage {sorted(contract_ks)} != {sorted(REQUIRED_KS)}")
    if contract_patterns != {0, 1}:
        errors.append(f"{directory.name}: contract patterns {sorted(contract_patterns)} do not cover dyadic+random")
    missing_raw = REQUIRED_RAW - raw_names
    if missing_raw:
        errors.append(f"{directory.name}: raw coverage missing {sorted(missing_raw)}")
    if not {16, 64} <= offsets:
        errors.append(f"{directory.name}: external offsets not both 16 and 64: {sorted(o for o in offsets if o)}")

    perf_arms = {}
    for rec in perf:
        checks += 1
        key = (rec.get("m"), rec.get("n"), rec.get("k"), rec.get("arm"))
        perf_arms[key] = rec
        if rec.get("submits") != 1:
            errors.append(f"{directory.name}: perf {key} submits != 1")
        submits += rec.get("submits", 0)
        exp = expected_tasks(rec["m"], rec["n"], rec["k"], f"perf-{rec.get('arm')}")
        if rec.get("tasks") != exp or rec.get("expected_tasks") != exp:
            errors.append(f"{directory.name}: perf {key} tasks {rec.get('tasks')} != {exp}")
        tasks += rec.get("tasks", 0)
        if rec.get("pass") is not True or rec.get("finite") is not True:
            errors.append(f"{directory.name}: perf {key} failed numerical check")
        if rec.get("unsaturated_frac", 0.0) < 0.5:
            errors.append(f"{directory.name}: perf {key} unsaturated_frac {rec.get('unsaturated_frac')} < 0.5")
        if rec.get("reps", 0) < reps:
            errors.append(f"{directory.name}: perf {key} reps {rec.get('reps')} < {reps}")
        for field in ("wall_ns", "hw_ns"):
            samples = rec.get(field)
            if not isinstance(samples, list) or len(samples) != rec.get("reps"):
                errors.append(f"{directory.name}: perf {key} {field} sample count wrong")
                continue
            for p, keyname in ((0.1, f"{field.split('_')[0]}_p10"),
                               (0.5, f"{field.split('_')[0]}_median"),
                               (0.9, f"{field.split('_')[0]}_p90")):
                if percentile(samples, p) != rec.get(keyname):
                    errors.append(f"{directory.name}: perf {key} {keyname} inconsistent")

    shapes = {(r.get("m"), r.get("n"), r.get("k")) for r in perf}
    for shape in shapes:
        for arm in ("fused", "unfused"):
            if (*shape, arm) not in perf_arms:
                errors.append(f"{directory.name}: perf shape {shape} missing {arm} arm")
    primary = perf_arms.get((64, 64, 4096, "fused"))
    second = perf_arms.get((64, 64, 4096, "unfused"))
    if not primary or not second or primary.get("tasks") != 18 or second.get("tasks") != 33:
        errors.append(f"{directory.name}: primary perf task counts not 18/33")
    big = perf_arms.get((256, 256, 1024, "fused"))
    if not big or big.get("lut_chunks") != 2:
        errors.append(f"{directory.name}: large-output perf case missing two LUT chunks")

    gate = directory / "gate.json"
    if not gate.exists():
        errors.append(f"{directory.name}: missing gate.json")
    else:
        try:
            report = json.loads(gate.read_text())
            if report.get("status") != "PASS" or report.get("errors"):
                errors.append(f"{directory.name}: oracle gate failed")
            cases = len(list((directory / "witnesses").glob("case-*.json")))
            if report.get("cases") != cases or cases == 0:
                errors.append(f"{directory.name}: oracle case count {report.get('cases')} != witnesses {cases}")
        except (OSError, ValueError) as exc:
            errors.append(f"{directory.name}: unreadable gate.json ({exc})")

    try:
        exitcode = int((directory / "exit.txt").read_text())
    except (OSError, ValueError):
        exitcode = None
        errors.append(f"{directory.name}: missing exit status")
    if exitcode != 0:
        errors.append(f"{directory.name}: nonzero exit {exitcode}")
    return dict(name=directory.name, exit=exitcode, checks=checks, submits=submits,
                tasks=tasks, passes=not errors), errors


def verify(root, reps):
    errors = []
    runs = []
    for mask in (1, 2, 4):
        for rep in (1, 2):
            name = f"core-{mask}-run-{rep}"
            directory = root / name
            if not directory.is_dir():
                errors.append(f"missing {name}")
                continue
            stats, run_errors = verify_process(directory, reps)
            errors.extend(run_errors)
            runs.append(stats)

    kernel = root / "kernel-after.log"
    if not kernel.exists():
        errors.append("missing kernel-after.log")
    else:
        faults = [s for s in kernel.read_text().splitlines()
                  if re.search(r"(rknpu|npu|iommu).*(timeout|timed out|fault|reset|failed)", s, re.I)]
        errors.extend(faults)
    host = root / "host-tests.log"
    if not host.exists() or "100% tests passed" not in host.read_text():
        errors.append("host/provider ctest did not pass")

    manifest = dict(status="FAIL" if errors else "PASS", reps=reps, errors=errors,
                    runs=runs,
                    checks=sum(r["checks"] for r in runs),
                    submits=sum(r["submits"] for r in runs),
                    tasks=sum(r["tasks"] for r in runs),
                    sha256={p.relative_to(root).as_posix(): digest(p)
                            for p in sorted(root.rglob("*"))
                            if p.is_file() and p.name != "campaign.json"})
    return manifest


def verify_manifest(root):
    """Re-check a recorded campaign without rewriting it."""
    manifest = json.loads((root / "campaign.json").read_text())
    errors = []
    if manifest.get("status") != "PASS" or manifest.get("errors"):
        errors.append("recorded campaign did not pass")
    runs = manifest.get("runs", [])
    expected = {f"core-{mask}-run-{rep}" for mask in (1, 2, 4) for rep in (1, 2)}
    if {r.get("name") for r in runs} != expected:
        errors.append("missing required runs")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p.name != "campaign.json"}
    if actual != set(manifest.get("sha256", {})):
        errors.append("file set changed since the manifest was written")
    for name, sha in manifest.get("sha256", {}).items():
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            errors.append(f"unsafe evidence path: {name}")
            continue
        path = root / rel
        if not path.is_file() or digest(path) != sha:
            errors.append(f"SHA256 mismatch: {name}")
    for run in runs:
        directory = root / run["name"]
        if not directory.is_dir():
            errors.append(f"missing run: {run['name']}")
            continue
        if run.get("exit") != 0 or not run.get("passes"):
            errors.append(f"failed run: {run['name']}")
    print(f"KTILING_VERIFY {'FAIL' if errors else 'PASS'} runs={len(runs)} recorded_checks={manifest.get('checks')}")
    for error in errors:
        print(f"VERIFY ERROR {error}")
    return 1 if errors else 0


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit("usage: ktiling_campaign.py RESULTS [REPS]")
    root = Path(sys.argv[1])
    if (root / "campaign.json").exists():
        return verify_manifest(root)
    if len(sys.argv) != 3:
        sys.exit("requested a new campaign but no REPS was given")
    reps = int(sys.argv[2])
    manifest = verify(root, reps)
    (root / "campaign.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"KTILING_CAMPAIGN {manifest['status']} runs={len(manifest['runs'])} "
          f"checks={manifest['checks']} submits={manifest['submits']} tasks={manifest['tasks']}")
    for error in manifest["errors"]:
        print(error)
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
