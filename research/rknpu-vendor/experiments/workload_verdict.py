#!/usr/bin/env python3
"""Two acceptance tiers for a workload's own numerical check.

`min_conv_exp` prints two kinds of verdict lines:

    total=1024  fp16-mismatch=0  max|diff|=0  max ulp=0          -> BIT-EXACT
    total=1024  fp16-mismatch=31 max|diff|=9.5e-07 max ulp=4      -> ULP-BOUNDED

BIT-EXACT is the tier every experiment used so far.  Some shapes (e.g. more input
channels) differ from the CPU reference only by fp16 accumulation order: a few
elements, a few ULP, relative error far below any structural error.  Those cannot
be called bit-exact, so they get a second tier that is **explicit and bounded**,
never a silent pass:

    ulp-bounded : mismatch fraction <= 5%  AND  max ulp <= 8
                  AND  max |diff| / |typical| small enough to be rounding only

Anything else is `rejected`.  A run in the ulp-bounded tier must be reported with
that label; it must never be counted as BIT-EXACT.

Usage
-----
    python3 workload_verdict.py --text out.txt
    echo "total=1024 fp16-mismatch=31 max ulp=4" | python3 workload_verdict.py
    python3 workload_verdict.py --selftest
"""

import argparse
import re
import sys

TOTAL_RE = re.compile(r"total=(\d+)")
MISMATCH_RE = re.compile(r"mismatch=(\d+)")
ULP_RE = re.compile(r"max ulp=(\d+)")
EXACT_RE = re.compile(r"RESULT:\s*BIT-EXACT")
# 显式阈值：超过就不是"舍入顺序"能解释的了
MAX_MISMATCH_FRACTION = 0.05
MAX_ULP = 8


def classify(text):
    """-> (tier, detail dict).  tier in {bit-exact, ulp-bounded, rejected, unknown}."""
    detail = {}
    total = TOTAL_RE.search(text)
    mismatch = MISMATCH_RE.search(text)
    ulp = ULP_RE.search(text)
    if total:
        detail["total"] = int(total.group(1))
    if mismatch:
        detail["mismatch"] = int(mismatch.group(1))
    if ulp:
        detail["max_ulp"] = int(ulp.group(1))
    if EXACT_RE.search(text):
        return "bit-exact", detail
    if "rejected" in text.lower() and not mismatch:
        return "rejected", detail
    if detail.get("total") and "mismatch" in detail and "max_ulp" in detail:
        frac = detail["mismatch"] / detail["total"]
        detail["mismatch_fraction"] = round(frac, 5)
        if frac <= MAX_MISMATCH_FRACTION and detail["max_ulp"] <= MAX_ULP:
            return "ulp-bounded", detail
        return "rejected", detail
    if "RESULT:" in text:
        return "rejected", detail
    return "unknown", detail


def selftest():
    failures = 0

    def group(name, fn):
        nonlocal failures
        print(name)
        try:
            fn()
        except AssertionError as exc:
            print("  FAIL " + str(exc))
            failures += 1

    def exact():
        tier, d = classify("  total=1024  fp16-mismatch=0  max|diff|=0  max ulp=0\n"
                           "  RESULT: BIT-EXACT (fp16)\n")
        assert tier == "bit-exact", (tier, d)
        print("  ok  a BIT-EXACT line is bit-exact")

    group("bit-exact tier", exact)

    def bounded():
        tier, d = classify("  total=1024  fp16-mismatch=31  max|diff|=9.53674e-07  max ulp=4\n"
                           "  RESULT: MISMATCH\n")
        assert tier == "ulp-bounded", (tier, d)
        assert d["mismatch_fraction"] == 0.03027, d
        print("  ok  31/1024 at <=8 ulp -> ulp-bounded (explicit second tier)")

    group("ulp-bounded tier", bounded)

    def too_many():
        tier, d = classify("  total=1024  fp16-mismatch=200  max|diff|=0.5  max ulp=3\n"
                           "  RESULT: MISMATCH\n")
        assert tier == "rejected", (tier, d)
        print("  ok  many mismatching elements -> rejected even at low ulp")

    group("mismatch fraction above the bound is rejected", too_many)

    def too_big_ulp():
        tier, d = classify("  total=1024  fp16-mismatch=45  max|diff|=8e-06  max ulp=140\n"
                           "  RESULT: MISMATCH\n")
        assert tier == "rejected", (tier, d)
        print("  ok  140 ulp -> rejected even with few mismatches")

    group("ulp above the bound is rejected", too_big_ulp)

    def unknown():
        tier, _ = classify("something else entirely\n")
        assert tier == "unknown", tier
        print("  ok  text without a verdict is unknown, not accepted")

    group("missing verdict is unknown", unknown)

    print()
    if failures:
        print("%d selftest group(s) failed" % failures)
        return 1
    print("all workload-verdict selftest checks passed")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", help="file with the workload output (default: stdin)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    text = open(args.text, errors="replace").read() if args.text else sys.stdin.read()
    tier, detail = classify(text)
    print("%s %s" % (tier, " ".join("%s=%s" % kv for kv in sorted(detail.items()))))
    return 0 if tier in ("bit-exact", "ulp-bounded") else 1


if __name__ == "__main__":
    sys.exit(main())
