#!/usr/bin/env python3
"""Verify a retrieved three-core campaign, including artifact hashes and row math."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
from row_gate import CASES, reference
from graph_gate import SHAPES


def load_half(path, count):
    raw=path.read_bytes()
    if len(raw)!=count*2:raise ValueError(f"wrong byte count: {path}")
    return struct.unpack(f"<{count}e",raw)


def artifact_report(root,name):
    report=json.loads((root/name).read_text())
    if report.get("status")!="PASS" or not report.get("artifacts"):
        raise ValueError(f"missing complete report: {root/name}")
    for rel,expected in report["artifacts"].items():
        path=Path(rel)
        if path.is_absolute() or ".." in path.parts:raise ValueError("unsafe artifact path")
        actual=hashlib.sha256((root/path).read_bytes()).hexdigest()
        if actual!=expected:raise ValueError(f"artifact hash mismatch: {root/path}")
    return report


def verify(root):
    checks=[]
    host=(root/"host-tests.log").read_text()
    if "100% tests passed" not in host:raise ValueError("host tests incomplete")
    for mask in (1,2,4):
        rowroot=root/f"core-{mask}-row"
        row=artifact_report(rowroot,"row-report.json")
        graphroot=root/f"core-{mask}-graph"
        graph=artifact_report(graphroot,"graph-report.json")
        cli=json.loads((root/f"core-{mask}-cli-regression/cli-report.json").read_text())
        expected_rows={(kind,m,n,distribution,mode) for kind,m,n,distribution in CASES for mode in ("chain","separate")}
        actual_rows={(c["kind"],*c["shape"],c["distribution"],c["mode"]) for c in row["checks"]}
        if actual_rows!=expected_rows:raise ValueError("row shape/mode coverage differs")
        expected_ffn={(m,d,h,bias,mode) for m,d,h in SHAPES for bias in (False,True) for mode in ("chain","separate")}
        actual_ffn={(*c["shape"],c["bias"],c["mode"]) for c in graph["checks"]}
        if actual_ffn!=expected_ffn:raise ValueError("normalized FFN coverage differs")
        if len(row["checks"])!=22 or row["negative_checks"]!=3:raise ValueError("row coverage incomplete")
        if len(graph["checks"])!=16 or graph["negative_checks"]!=2 or not all(c["rmsnorm"] for c in graph["checks"]):
            raise ValueError("normalized FFN coverage incomplete")
        if cli["status"]!="PASS" or len(cli["checks"])!=25:raise ValueError("CLI regression incomplete")
        log=(root/f"core-{mask}-device-regression.log").read_text()
        if len(re.findall(r"^MULTITILE .* PASS$",log,re.M))!=8:raise ValueError("device regression incomplete")
        errors=[]
        for case in row["checks"]:
            kind=case["kind"];m,n=case["shape"];args=case["argv"]
            x=load_half(rowroot/Path(args[4]).name,m*n)
            gamma=load_half(rowroot/Path(args[5]).name,n) if kind.endswith("rms") else None
            y=load_half(rowroot/Path(args[6]).name,m*n)
            ref=reference(kind,m,n,x,gamma)
            if not all(math.isfinite(v) for v in y):raise ValueError("nonfinite row output")
            error=max(abs(a-b) for a,b in zip(y,ref))
            if not math.isclose(error,case["max_abs"],rel_tol=1e-9,abs_tol=1e-12):raise ValueError("reported row error differs from recomputed error")
            if error>(0.004 if gamma is not None else 0.002):raise ValueError("row math mismatch")
            if gamma is None:
                for r in range(m):
                    expected=1 if any(ref[r*n:(r+1)*n]) else 0
                    if any(v<0 for v in y[r*n:(r+1)*n]) or abs(math.fsum(y[r*n:(r+1)*n])-expected)>0.002:
                        raise ValueError("softmax mass mismatch")
            errors.append(error)
        checks.append({"core_mask":mask,"row_checks":22,"row_negative_checks":3,
            "rms_max_abs":max(c["max_abs"] for c in row["checks"] if c["kind"].endswith("rms")),
            "softmax_max_abs":max(c["max_abs"] for c in row["checks"] if c["kind"].endswith("softmax")),
            "softmax_mass_error":max(c["row_mass_error"] for c in row["checks"]),
            "normalized_ffn_checks":16,"ffn_negative_checks":2,
            "ffn_max_abs":max(c["max_abs"] for c in graph["checks"]),
            "cli_regression_checks":25,"device_regression_cases":8})
    interval=root/"kernel-interval.log"
    if interval.exists() and re.search(r"rknpu.*(?:timeout|fault)|iommu.*fault|Unhandled fault",interval.read_text(),re.I):
        raise ValueError("kernel fault/timeout in campaign interval")
    return {"status":"PASS","checks":checks,"kernel_interval_available":interval.exists(),
            "artifact_hashes_verified":True,"row_math_recomputed":True}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("results",type=Path)
    args=parser.parse_args()
    summary=verify(args.results)
    target=args.results/"row-summary.json"
    if target.exists():
        if json.loads(target.read_text())!=summary:raise ValueError("existing summary differs")
    else:target.write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
