#!/usr/bin/env python3
"""Independent stdlib oracle for the K-tiling witnesses.

Reads the case-*.json / case-*.bin witnesses written by ktiling_probe.c and
re-derives every numerical result from scratch (no reuse of the C code): the full
FP64 reference and the tile-aware FP16-narrowed reference, the activated output
for contract/perf arms and the raw unactivated accumulator for raw arms. It also
recomputes the LUT chunk count and the expected one-submit task count.

This oracle is intentionally independent: it must be possible for it to disagree
with the probe. A disagreement is a failure, never silently tolerated.

Usage: ktiling_gate.py WITNESS_DIR [--tol ACT [--raw-tol RAW]]
Writes a JSON report to stdout; exits 0 on PASS, 1 on FAIL.
"""
import argparse
import json
import math
import struct
import sys
from pathlib import Path

K_TILE = 256
MAX_WORDS = 2048
LUT_ELEMENTS = 32768
TOL_ACT = 0.004
TOL_RAW = 0.004


def f16_bits(value):
    return struct.pack("<e", value)


def f16(value):
    """Round a Python float to the nearest FP16 (via half-precision round trip)."""
    return struct.unpack("<e", struct.pack("<e", value))[0]


def load_f16(path, count):
    raw = Path(path).read_bytes()
    if len(raw) != count * 2:
        raise ValueError(f"{path}: expected {count * 2} bytes, got {len(raw)}")
    return list(struct.unpack(f"<{count}e", raw))


def round_up(v, a):
    return (v + a - 1) // a * a


def physical(m, n):
    rows = round_up(12 if m < 12 else m, 4)
    cols = round_up(n, 32)
    return rows, cols


def tiles(k):
    return (k + K_TILE - 1) // K_TILE


def tile_len(k, i):
    return min(K_TILE, k - i * K_TILE)


def lut_chunks(m, n):
    rows, cols = physical(m, n)
    return (rows * cols + LUT_ELEMENTS - 1) // LUT_ELEMENTS


def expected_tasks(m, n, k, kind):
    nt = tiles(k)
    if kind == "raw":
        return nt
    if kind == "perf-unfused":
        return 2 * nt + lut_chunks(m, n)
    return nt + 1 + lut_chunks(m, n)


def hardsigmoid(v):
    r = v / 6.0 + 0.5
    return 0.0 if r < 0 else 1.0 if r > 1 else r


def reference(m, n, k, x, w, bias, unfused=False):
    """Return (raw64, raw16, act64, act16) row-major [m][n] Python lists.

    raw64/raw16 are the pre-activation accumulation; act* fold in bias and
    HardSigmoid. raw16 narrows through FP16 once per K tile, exactly as the DPU
    EW accumulation does; this is NOT an fp32-exact claim.
    """
    nt = tiles(k)
    raw64 = [[0.0] * n for _ in range(m)]
    raw16 = [[0.0] * n for _ in range(m)]
    act64 = [[0.0] * n for _ in range(m)]
    act16 = [[0.0] * n for _ in range(m)]
    for r in range(m):
        xrow = x[r * k:(r + 1) * k]
        for j in range(n):
            wrow = w[j * k:(j + 1) * k]
            s64 = 0.0
            s16 = 0.0
            for t in range(nt):
                k0 = t * K_TILE
                kt = tile_len(k, t)
                ts = 0.0
                for c in range(k0, k0 + kt):
                    ts += xrow[c] * wrow[c]
                s64 += ts
                part = f16(ts) if unfused else ts
                s16 = f16(part) if t == 0 else f16(s16 + part)
            raw64[r][j] = s64
            raw16[r][j] = s16
            if bias is None:
                act64[r][j] = s64
                act16[r][j] = s16
            else:
                act64[r][j] = hardsigmoid(s64 + bias[j])
                act16[r][j] = hardsigmoid(f16(s16 + bias[j]))
    return raw64, raw16, act64, act16


def score(got, ref64, ref16, tol):
    bad = 0
    finite = True
    peak64 = peak16 = 0.0
    unsat = 0
    for i, g in enumerate(got):
        if not math.isfinite(g):
            finite = False
            bad += 1
            continue
        d64 = abs(g - ref64[i])
        d16 = abs(g - ref16[i])
        peak64 = max(peak64, d64)
        peak16 = max(peak16, d16)
        ok = d64 <= tol and d16 <= tol
        if not ok:
            bad += 1
        if 0.02 < ref64[i] < 0.98:
            unsat += 1
    return dict(bad=bad, finite=finite, peak64=peak64, peak16=peak16,
                unsaturated=(unsat / len(got) if got else 0.0))


def check_case(path, tol_act, tol_raw):
    case = json.loads(Path(path).read_text())
    kind = case["kind"]
    m, n, k = case["m"], case["n"], case["k"]
    base = Path(path).parent
    x = load_f16(base / case["x"], m * k)
    w = load_f16(base / case["weights"], n * k)
    bias = load_f16(base / case["bias"], n)
    y = load_f16(base / case["y"], m * n)
    raw = kind == "raw"
    # raw witnesses carry a zero placeholder bias and read the accumulator.
    use_bias = None if raw else bias
    raw64, raw16, act64, act16 = reference(m, n, k, x, w, use_bias, unfused=kind=="perf-unfused")
    if raw:
        ref64 = [raw64[r][j] for r in range(m) for j in range(n)]
        ref16 = [raw16[r][j] for r in range(m) for j in range(n)]
        tol = tol_raw
    else:
        ref64 = [act64[r][j] for r in range(m) for j in range(n)]
        ref16 = [act16[r][j] for r in range(m) for j in range(n)]
        tol = tol_act
    st = score(y, ref64, ref16, tol)

    errors = []
    if not st["finite"]:
        errors.append("non-finite output")
    if st["bad"]:
        errors.append(f"{st['bad']}/{m * n} outside tolerance {tol}")
    # The probe's own recorded peaks must agree with this independent recomputation.
    if abs(st["peak64"] - case.get("peak_abs_fp64", -1)) > 1e-6:
        errors.append(f"peak_abs_fp64 mismatch outer={case.get('peak_abs_fp64')} oracle={st['peak64']:.9g}")
    if abs(st["peak16"] - case.get("peak_abs_fp16", -1)) > 1e-6:
        errors.append(f"peak_abs_fp16 mismatch outer={case.get('peak_abs_fp16')} oracle={st['peak16']:.9g}")
    exp = expected_tasks(m, n, k, kind)
    if case.get("tasks") != exp:
        errors.append(f"tasks {case.get('tasks')} != expected {exp}")
    if case.get("submits") != 1:
        errors.append(f"submits {case.get('submits')} != 1")
    if case.get("pass") is not True:
        errors.append("probe marked case failed")
    return dict(id=case.get("id"), kind=kind, m=m, n=n, k=k,
                tiles=tiles(k), lut_chunks=lut_chunks(m, n),
                tasks=case.get("tasks"), expected_tasks=exp,
                peak64=st["peak64"], peak16=st["peak16"],
                unsaturated=st["unsaturated"], pass_=not errors, errors=errors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("witness_dir")
    ap.add_argument("--tol", type=float, default=TOL_ACT)
    ap.add_argument("--raw-tol", type=float, default=TOL_RAW)
    args = ap.parse_args()
    root = Path(args.witness_dir)
    cases = sorted(root.glob("case-*.json"))
    results = []
    errors = []
    if not cases:
        errors.append("no case witnesses found")
    for case in cases:
        try:
            r = check_case(case, args.tol, args.raw_tol)
        except Exception as exc:  # noqa: BLE001 - report, never crash the campaign
            r = dict(id=str(case), pass_=False, errors=[f"oracle exception: {exc}"])
        results.append(r)
        for e in r.get("errors", []):
            errors.append(f"{case.name}: {e}")
    report = dict(status="FAIL" if errors else "PASS", cases=len(cases),
                  errors=errors, results=results)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
