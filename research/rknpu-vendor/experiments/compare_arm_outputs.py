#!/usr/bin/env python3
"""Compare the output dumps of the marker experiment arms.

The marker experiment replaces one 8-byte command in an otherwise identical
stream.  The arms therefore have the same length and the same arithmetic, so the
ONLY acceptable outcome is that the outputs are **bit-identical**:

    original   untouched baseline
    nop        the slot becomes OP_NONE   -> must equal original
    marker     the slot becomes a PPU write -> must equal original

If the marker arm differs from the baseline, the register write was not inert and
that register cannot be used as a marker, no matter how the timing looks.  A nop
arm that differs means the duplicate-write assumption was wrong for this slot.

This tool never decides "close enough" by itself: a tolerance verdict is
available (`--atol` / `--rtol`) but is reported separately from bit-exactness and
is only meant for outputs that go through a float reduction whose order changed.

Usage
-----
    python3 compare_arm_outputs.py --baseline original.bin \
        --arm nop=nop.bin --arm marker=marker.bin --out-dir results/arms_cmp
    # interpret as float32 to get magnitudes / ULP instead of raw bytes:
    python3 compare_arm_outputs.py --baseline original.bin --arm marker=marker.bin \
        --dtype float32 --atol 1e-6
    python3 compare_arm_outputs.py --selftest
"""

import argparse
import array
import json
import math
import os
import struct
import sys

DTYPES = {
    "int8": ("b", 1), "uint8": ("B", 1),
    "int16": ("h", 2), "uint16": ("H", 2),
    "int32": ("i", 4), "uint32": ("I", 4),
    "float16": ("e", 2), "float32": ("f", 4), "float64": ("d", 8),
}


def read_bytes(path):
    with open(path, "rb") as fh:
        return fh.read()


def unpack(data, dtype):
    code, size = DTYPES[dtype]
    if len(data) % size:
        raise ValueError("%s length %d is not a multiple of %d" % (dtype, len(data), size))
    values = array.array(code)
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    return list(values)


def ulp_distance(a, b):
    """Distance in representable steps; handles sign and NaN/Inf bit patterns."""
    if a == b:
        return 0
    # map the IEEE bit pattern to a monotone integer
    pa = struct.unpack("<q", struct.pack("<d", float(a)))[0]
    pb = struct.unpack("<q", struct.pack("<d", float(b)))[0]
    def mono(v):
        return v if v >= 0 else (-(v & 0x7FFFFFFFFFFFFFFF))
    return abs(mono(pa) - mono(pb))


def compare(baseline_bytes, other_bytes, dtype=None, atol=0.0, rtol=0.0):
    result = {
        "baseline_size": len(baseline_bytes),
        "size": len(other_bytes),
        "bit_exact": False,
        "verdict": None,
    }
    if len(baseline_bytes) != len(other_bytes):
        result["verdict"] = "size_mismatch"
        result["note"] = ("the two dumps have different lengths (%d vs %d); the arms are "
                          "not comparable" % (len(baseline_bytes), len(other_bytes)))
        return result

    differing = [i for i in range(len(baseline_bytes))
                 if baseline_bytes[i] != other_bytes[i]]
    result["different_bytes"] = len(differing)
    result["first_diff_offset"] = differing[0] if differing else None
    result["bit_exact"] = not differing

    if dtype is None:
        if differing:
            result["verdict"] = "different"
            result["note"] = ("%d byte(s) differ; first at offset %d"
                              % (len(differing), differing[0]))
        else:
            result["verdict"] = "identical"
            result["note"] = "byte-identical"
        return result

    a = unpack(baseline_bytes, dtype)
    b = unpack(other_bytes, dtype)
    nan_mismatch = 0
    max_abs = 0.0
    max_ulp = 0
    worst_index = None
    out_of_tolerance = 0
    for i, (x, y) in enumerate(zip(a, b)):
        x_nan = isinstance(x, float) and math.isnan(x)
        y_nan = isinstance(y, float) and math.isnan(y)
        if x_nan or y_nan:
            if not (x_nan and y_nan):
                nan_mismatch += 1
            continue
        diff = abs(float(x) - float(y))
        tol = atol + rtol * abs(float(x))
        if diff > tol:
            out_of_tolerance += 1
        if diff > max_abs:
            max_abs = diff
            worst_index = i
        if dtype in ("float16", "float32", "float64") and diff > 0:
            max_ulp = max(max_ulp, ulp_distance(x, y))

    result.update({
        "dtype": dtype,
        "elements": len(a),
        "nan_mismatch": nan_mismatch,
        "max_abs_diff": max_abs,
        "max_ulp_diff": max_ulp,
        "worst_index": worst_index,
        "out_of_tolerance": out_of_tolerance,
        "atol": atol,
        "rtol": rtol,
    })
    if not differing:
        result["verdict"] = "identical"
        result["note"] = "byte-identical"
    elif nan_mismatch or out_of_tolerance:
        result["verdict"] = "different"
        result["note"] = ("%d element(s) outside tolerance, %d NaN mismatch(es), "
                          "max |diff| = %g, max ULP = %d"
                          % (out_of_tolerance, nan_mismatch, max_abs, max_ulp))
    else:
        result["verdict"] = "within_tolerance"
        result["note"] = ("not bit-identical but inside tolerance: max |diff| = %g, "
                          "max ULP = %d, %d differing byte(s)"
                          % (max_abs, max_ulp, len(differing)))
    return result


def write_outputs(pairs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "comparison.json"), "w") as fh:
        json.dump(pairs, fh, indent=2)
    lines = ["# 三臂输出比对\n"]
    lines.append("判据：`original` / `nop` / `marker` 三臂的命令流长度与算术完全相同，"
                 "**只有逐位一致**才算标记无副作用。容差结论单独标注，不能替代逐位一致。\n")
    lines.append("| 对照 | 结果 | 不同字节 | 首个偏移 | max abs | max ULP | 说明 |")
    lines.append("|---|---|---|---|---|---|---|")
    for pair in pairs:
        r = pair["result"]
        lines.append("| %s | **%s** | %s | %s | %s | %s | %s |" % (
            pair["name"], r["verdict"],
            r.get("different_bytes", "n/a"), r.get("first_diff_offset", "n/a"),
            ("%.6g" % r["max_abs_diff"]) if "max_abs_diff" in r else "n/a",
            r.get("max_ulp_diff", "n/a"), r["note"]))
    bad = [p for p in pairs if p["result"]["verdict"] not in ("identical", "within_tolerance")]
    lines.append("")
    if bad:
        lines.append("**未通过**：%s —— 这些臂与基线不一致，对应的寄存器/槽位不能当标记用。"
                     % ", ".join(p["name"] for p in bad))
    else:
        lines.append("所有对照通过。注意这**只**说明替换这个槽位没有改变输出，"
                     "不说明标记寄存器可读、也不说明标记可见时间等于命令执行时间。")
    lines.append("")
    open(os.path.join(out_dir, "report.md"), "w").write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------

def selftest():
    import tempfile
    failures = 0

    def group(name, fn):
        nonlocal failures
        print(name)
        try:
            fn()
        except AssertionError as exc:
            print("  FAIL " + str(exc))
            failures += 1

    with tempfile.TemporaryDirectory() as tmpdir:
        def path(name, data):
            p = os.path.join(tmpdir, name)
            with open(p, "wb") as fh:
                fh.write(data)
            return p

        def identical():
            data = bytes(range(256))
            r = compare(data, data)
            assert r["verdict"] == "identical" and r["bit_exact"], r
            print("  ok  identical dumps -> identical")

        group("identical dumps", identical)

        def one_byte():
            a = bytes(range(64))
            b = bytearray(a); b[17] ^= 0x01
            r = compare(a, bytes(b))
            assert r["verdict"] == "different", r
            assert r["first_diff_offset"] == 17, r
            assert r["different_bytes"] == 1, r
            print("  ok  one byte differs -> offset %d" % r["first_diff_offset"])

        group("a single differing byte is located", one_byte)

        def size_mismatch():
            r = compare(b"\x00" * 8, b"\x00" * 9)
            assert r["verdict"] == "size_mismatch", r
            print("  ok  different lengths are refused")

        group("length mismatch is not silently padded", size_mismatch)

        def float_within_tolerance():
            a = [1.0, 2.0, 3.0, 4.0]
            # 1e-9 is below float32 resolution and would make the dumps identical;
            # use a difference that survives the rounding but stays inside atol
            b = [1.0, 2.000001, 3.0, 4.0]
            da = struct.pack("<4f", *a)
            db = struct.pack("<4f", *b)
            assert da != db, "the test case must actually differ"
            r = compare(da, db, dtype="float32", atol=1e-5)
            assert r["verdict"] == "within_tolerance", r
            assert not r["bit_exact"] and r["max_abs_diff"] <= 1e-5, r
            print("  ok  float32 within atol -> within_tolerance (max %g)"
                  % r["max_abs_diff"])

        group("tolerance verdict is separate from bit-exactness", float_within_tolerance)

        def float_out_of_tolerance():
            da = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
            db = struct.pack("<4f", 1.0, 2.0, 3.0, 4.5)
            r = compare(da, db, dtype="float32", atol=1e-6)
            assert r["verdict"] == "different", r
            assert r["worst_index"] == 3, r
            print("  ok  a real difference is not excused by tolerance")

        group("tolerance does not hide a real difference", float_out_of_tolerance)

        def nan_handling():
            da = struct.pack("<2f", float("nan"), 1.0)
            db = struct.pack("<2f", float("nan"), 1.0)
            r = compare(da, db, dtype="float32")
            assert r["verdict"] == "identical", r
            dc = struct.pack("<2f", 1.0, 1.0)
            r = compare(da, dc, dtype="float32")
            assert r["verdict"] == "different" and r["nan_mismatch"] == 1, r
            print("  ok  NaN==NaN is identical, NaN vs number is a mismatch")

        group("NaN handling", nan_handling)

        def int_tolerance():
            da = bytes([1, 2, 3])
            db = bytes([1, 3, 3])
            r = compare(da, db, dtype="uint8", atol=1.0)
            assert r["verdict"] == "within_tolerance", r
            r = compare(da, db, dtype="uint8", atol=0.0)
            assert r["verdict"] == "different", r
            print("  ok  integer tolerance behaves as documented")

        group("integer tolerance", int_tolerance)

        def report_written():
            a = bytes(16)
            pairs = [{"name": "nop vs original", "result": compare(a, a)},
                     {"name": "marker vs original", "result": compare(a, b"\x00" * 15 + b"\x01")}]
            out = os.path.join(tmpdir, "out")
            write_outputs(pairs, out)
            text = open(os.path.join(out, "report.md")).read()
            assert "未通过" in text and "marker vs original" in text, text
            assert os.path.exists(os.path.join(out, "comparison.json"))
            print("  ok  report marks the failing arm")

        group("the report names the failing arm", report_written)

    print()
    if failures:
        print("%d selftest group(s) failed" % failures)
        return 1
    print("all arm-comparison selftest checks passed")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", help="reference dump (the 'original' arm)")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=PATH",
                    help="arm dump to compare against the baseline (repeatable)")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default=None,
                    help="interpret both dumps as this type to report magnitudes")
    ap.add_argument("--atol", type=float, default=0.0)
    ap.add_argument("--rtol", type=float, default=0.0)
    ap.add_argument("--out-dir")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.baseline or not args.arm or not args.out_dir:
        ap.error("--baseline, at least one --arm and --out-dir are required")

    base = read_bytes(args.baseline)
    pairs = []
    for spec in args.arm:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            ap.error("--arm expects NAME=PATH, got %r" % spec)
        if not os.path.exists(path):
            ap.error("arm dump not found: %s" % path)
        result = compare(base, read_bytes(path), args.dtype, args.atol, args.rtol)
        result["baseline"] = args.baseline
        result["file"] = path
        pairs.append({"name": name, "result": result})

    write_outputs(pairs, args.out_dir)
    print("wrote %s" % args.out_dir)
    worst = 0
    for pair in pairs:
        r = pair["result"]
        print("  %-24s %-16s %s" % (pair["name"], r["verdict"], r["note"]))
        if r["verdict"] not in ("identical", "within_tolerance"):
            worst = 1
    return worst


if __name__ == "__main__":
    sys.exit(main())
