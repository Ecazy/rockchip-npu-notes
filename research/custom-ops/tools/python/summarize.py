#!/usr/bin/env python3
"""Portable evidence manifest and campaign gate; no third-party packages."""
import hashlib
import json
from pathlib import Path
import re
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources(project, upstream):
    paths = [("custom-ops/" + str(p.relative_to(project)), p) for p in project.rglob("*")
             if p.is_file() and not p.is_symlink() and
             not any(part.startswith("build") or part in ("results","artifacts","__pycache__") for part in p.relative_to(project).parts)
             and p.suffix in (".c", ".h", ".py", ".sh", ".txt")]
    for name in ("rocket-userspace", "rknpu-submit"):
        root = upstream / name
        for subdir in ("include", "src"):
            paths.extend((name + "/" + str(p.relative_to(root)), p)
                         for p in (root / subdir).rglob("*")
                         if p.is_file() and p.suffix in (".c", ".h"))
    return {label: digest(p) for label, p in sorted(paths)}


def campaign(root):
    runs = []
    errors = []
    kernel = root / "kernel-after.log"
    fault_pattern = re.compile(r"(?:rknpu|npu|iommu).*(?:timed out|timeout|fault|soft reset|failed)", re.I)
    faults = [line for line in kernel.read_text().splitlines() if fault_pattern.search(line)] if kernel.exists() else []
    if not kernel.exists():
        errors.append("missing kernel-after.log")
    host = root / "host-tests.log"
    if not host.exists() or "100% tests passed" not in host.read_text():
        errors.append("host/provider ctest did not pass")
    for mask in (1, 2, 4):
        for rep in (1, 2):
            run = root / f"core-{mask}-run-{rep}"
            log, exitfile = run / "probe.log", run / "exit.txt"
            if not log.exists() or not exitfile.exists():
                errors.append(f"missing {run.name}")
                continue
            lines = log.read_text().splitlines()
            result = re.search(r"SUMMARY checks=(\d+) failures=(\d+) status=(\w+)", "\n".join(lines))
            checks = [line for line in lines if line.startswith("CHECK ")]
            exitcode = int(exitfile.read_text())
            ok = (exitcode == 0 and result is not None and result[3] == "PASS"
                  and int(result[2]) == 0 and len(checks) == int(result[1])
                  and len(checks) == 349 and all(line.endswith(" PASS") for line in checks))
            cli = run / "cli-report.json"
            cli_checks = 0
            if cli.exists():
                data = json.loads(cli.read_text())
                cli_checks = len(data.get("checks", []))
                positives = {(tuple(c.get("shape", [])), c.get("mode"))
                             for c in data.get("checks", []) if "shape" in c}
                expected = {(shape, mode) for shape in ((1,1,1),(4,32,32),(29,37,45),
                                                       (37,65,33),(1,256,1),(256,1,256))
                            for mode in ("separate", "chain")}
                cli_ok = (data.get("status") == "PASS" and cli_checks == 17
                          and data.get("core_mask") == str(mask) and positives == expected
                          and all(c.get("pass") is True for c in data["checks"]))
                ok = ok and cli_ok
            else:
                ok = False
            if not ok:
                errors.append(f"failed/incomplete {run.name}")
            runs.append({"name": run.name, "core_mask": mask, "exit": exitcode,
                         "checks": len(checks), "cli_checks": cli_checks, "pass": ok,
                         "sha256": {p.name: digest(p) for p in sorted(run.iterdir()) if p.is_file()}})
    manifest = {"status": "PASS" if not errors and not faults else "FAIL",
                "checks": sum(r["checks"] for r in runs),
                "cli_checks": sum(r["cli_checks"] for r in runs), "runs": runs,
                "kernel_faults": faults, "errors": errors,
                "evidence_sha256": {p.name: digest(p) for p in sorted(root.iterdir())
                                    if p.is_file() and p.name != "summary.json"}}
    (root / "summary.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"CAMPAIGN {manifest['status']} runs={len(runs)} checks={manifest['checks']} kernel_faults={len(faults)}")
    return 0 if manifest["status"] == "PASS" else 1


def verify(root):
    """Verify recorded evidence against its saved hashes; never rewrite a record."""
    manifest = json.loads((root / "summary.json").read_text())
    errors = []
    if manifest.get("status") != "PASS" or manifest.get("errors") or manifest.get("kernel_faults"):
        errors.append("recorded campaign did not pass")
    runs = manifest.get("runs", [])
    if {r.get("name") for r in runs} != {f"core-{mask}-run-{rep}"
                                          for mask in (1,2,4) for rep in (1,2)}:
        errors.append("missing required runs")
    if sum(r.get("checks", 0) for r in runs) != manifest.get("checks"):
        errors.append("inconsistent total check count")
    def check_files(directory, expected, ignore=()):
        actual = {p.name for p in directory.iterdir() if p.is_file() and p.name not in ignore}
        if actual != set(expected):
            errors.append(f"file set changed: {directory.name}")
        for name, sha in expected.items():
            if Path(name).name != name:
                errors.append(f"unsafe file name: {name}")
                continue
            path = directory / name
            if not path.is_file() or digest(path) != sha:
                errors.append(f"SHA256 mismatch: {directory.name}/{name}")
    check_files(root, manifest.get("evidence_sha256", {}), ("summary.json",))
    for run in runs:
        directory = root / run["name"]
        if not directory.is_dir():
            errors.append(f"missing run: {directory.name}")
            continue
        check_files(directory, run.get("sha256", {}))
        if not run.get("pass") or run.get("exit") != 0:
            errors.append(f"failed run: {directory.name}")
        log = directory / "probe.log"
        if log.is_file():
            lines = log.read_text().splitlines()
            checks = [line for line in lines if line.startswith("CHECK ")]
            if len(checks) != run.get("checks") or not all(line.endswith(" PASS") for line in checks):
                errors.append(f"check log disagrees with manifest: {directory.name}")
    for error in errors:
        print(f"VERIFY ERROR {error}")
    print(f"VERIFY {'FAIL' if errors else 'PASS'} runs={len(runs)} recorded_checks={manifest.get('checks')}")
    return 1 if errors else 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--sources":
        print(json.dumps(sources(Path(sys.argv[2]), Path(sys.argv[3])), indent=2))
    elif len(sys.argv) == 2:
        root = Path(sys.argv[1])
        sys.exit(verify(root) if (root / "summary.json").exists() else campaign(root))
    elif len(sys.argv) == 3 and sys.argv[1] == "--create":
        root = Path(sys.argv[2])
        if (root / "summary.json").exists():
            sys.exit("refusing to replace an existing manifest; use a fresh result directory")
        sys.exit(campaign(root))
    else:
        sys.exit("usage: summarize.py RESULTS | --create RESULTS | --sources PROJECT UPSTREAM")
