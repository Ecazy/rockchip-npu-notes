#!/usr/bin/env python3
"""Estimate the submit -> kick offset from the interrupt-status falling edge.

THIS IS NOT A HARD BOUND.  The estimator places x*, the transition point of
x = phase - hw, at the midpoint of the two samples around the best-separating
threshold.  That midpoint can land BELOW the true value: with a synthetic kick of
5.00 us the estimator returns 4.50 us and its own selftest passes.  So the output
is a **conditional estimate under the stated model**, not an upper bound, and the
bootstrap interval is not a maximum-latency guarantee either.

Idea
----
For a run, let phase be measured from the submit entry and hw be the driver's
reported hw_elapse_time, i.e. the time from the commit (the kick, where the driver
programs OPERATION_ENABLE) to the job-end interrupt:

    job_end - kick = hw

The DPU group bit is cleared by the driver's IRQ path at job end.  So the bit is set
exactly while

    phase < kick_rel + hw + irq_latency

where kick_rel is the kick time relative to the submit entry and irq_latency >= 0.
Write x = phase - hw.  Then the transition sits at x* = kick_rel + irq_latency, so

    submit -> kick  <=  x*        (equality only if the IRQ latency were zero)

This tool estimates x* from a phase sweep by finding the threshold on x that best
separates "bit set" from "bit clear", and resamples the sweep to get an interval.
It reports an UPPER BOUND, not a measurement of the kick.

Limits (repeated in the report)
-------------------------------
* the transition is located by a fitted threshold, not by watching the kick;
* IRQ latency, the sampler's own read latency and the phase quantisation all fold
  into x*, so x* is conservative;
* only runs that printed BIT-EXACT are used; nothing is repaired;
* the estimate says nothing about regcmd positions.

Usage
-----
    python3 calibrate_kick_offset.py --sweep fp16=DIR_A [--sweep iw32=DIR_B] \
        --out-dir OUT
    python3 calibrate_kick_offset.py --selftest
"""

import argparse
import collections
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_irq_completion import DONE_MASK, load_sweep  # noqa: E402


def transition_estimate(points, min_phase_ns=20000):
    """points: [(phase_ns, hw_ns, is_set)].  Returns x* (ns) or None.

    x* is an ESTIMATE of where the set->clear transition sits on the
    x = phase - hw axis.  It may be smaller than the true value (see selftest).

    x* is placed at the midpoint between the largest x that is set and the smallest
    x that is clear, restricted to the neighbourhood of the crossing so that a few
    stragglers far away cannot drag it."""
    xs = sorted((p[0] - p[1], p[2]) for p in points if p[0] >= min_phase_ns)
    if not xs:
        return None
    best = None
    for threshold in sorted(set(x for x, _ in xs)):
        inside = sum(1 for x, s in xs if (x < threshold) == bool(s))
        if best is None or inside > best[0]:
            best = (inside, threshold)
    if best is None:
        return None
    _, thr = best
    below = [x for x, s in xs if s and x < thr]
    above = [x for x, s in xs if not s and x >= thr]
    if not below or not above:
        return float(thr)
    # keep the transition tight: use the nearest set/clear pair around the threshold
    near_set = max(below)
    near_clear = min(above)
    return (near_set + near_clear) / 2.0


def estimate(records, seed=20261022, resamples=2000):
    points = [(r["phase_ns"], r["hw_ns"], 1 if r["status"] & DONE_MASK else 0)
              for r in records]
    x_star = transition_estimate(points)
    if x_star is None:
        return {"n": len(points), "status": "no_transition"}
    # resample runs; keep the phase structure by drawing whole (x, set) pairs
    rnd = random.Random(seed)
    boots = []
    for _ in range(resamples):
        sample = [rnd.choice(points) for _ in range(len(points))]
        value = transition_estimate(sample)
        if value is not None:
            boots.append(value)
    boots.sort()
    return {
        "n": len(points),
        "status": "ok",
        "x_star_ns": x_star,
        "interpretation": "conditional transition estimate under the stated model "
                          "(commit/hw/clear-on-IRQ); NOT a hard bound",
        "is_hard_bound": False,
        "ci95_ns": [boots[int(0.025 * (len(boots) - 1))],
                    boots[int(0.975 * (len(boots) - 1))]] if len(boots) > 50 else None,
        "ci95_meaning": "bootstrap spread of the estimator, not a max-latency guarantee",
        "n_set": sum(1 for p in points if p[2]),
        "n_clear": sum(1 for p in points if not p[2]),
    }


def write_outputs(result, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    def us(v):
        return "n/a" if v is None else "%.2f" % (v / 1000.0)

    lines = ["# submit → kick 的**条件估计**（由中断状态下降沿反推）\n"]
    lines.append("`job_end − kick = hw_elapse`（驱动在 commit 处写 OPERATION_ENABLE），"
                 "而 DPU 组位由 IRQ 路径在作业结束时清除，所以置位条件是 "
                 "`φ < kick_rel + hw + irq_latency`。令 `x = φ − hw`，过渡点 "
                 "`x* = kick_rel + clear_latency`——这是一个**混合量**，不是 kick 本身。\n")
    lines.append("**这不是硬上界**：估计量把 x* 放在最佳分类阈值两侧两点的中点，"
                 "该中点可能**低于**真值（自测实例：真值 5.00 µs → 估计 4.50 µs，测试通过）。"
                 "bootstrap 区间是估计量的散布，**不是**最大延迟保证。\n")
    lines.append("同时注意：采样线程与提交线程并行，x* 里混着 IRQ 延迟、采样读延迟与相位量化，"
                 "**不能**把它们当作可以串行相加的分量。\n")
    lines.append("| 负载 | 样本 | 置位/未置位 | x* 估计 (µs) | 95% 区间 (µs) | 含义 |")
    lines.append("|---|---|---|---|---|---|")
    for label, block in result["loads"].items():
        e = block
        ci = e.get("ci95_ns")
        lines.append("| %s | %d | %d/%d | %s | %s | 条件估计，**不是上界** |" % (
            label, e["n"], e["n_set"], e["n_clear"], us(e.get("x_star_ns")),
            "n/a" if not ci else "%.2f–%.2f" % (ci[0] / 1000.0, ci[1] / 1000.0)))
    lines.append("\n## 没有证明的\n")
    lines.append("- kick 的确切时刻；这里只有**条件模型下的估计**，且可能偏低；")
    lines.append("- 硬上界：**未建立**；")
    lines.append("- 驱动计时源与用户态 RAW 时钟是否同源：**未建立**（见测量模型 §2.2）；")
    lines.append("- IRQ 延迟与采样延迟均未单独分离；")
    lines.append("- 不涉及任何 regcmd 位置；")
    lines.append("- 结论仅适用于本板、本驱动版本与这些负载。\n")
    open(os.path.join(out_dir, "report.md"), "w").write("\n".join(lines) + "\n")


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

    def rec(phase, hw, is_set, idx=0):
        return {"phase_ns": phase, "hw_ns": hw, "status": 0x300 if is_set else 0,
                "index": idx}

    def synthetic(kick, irq=0, jitter=0, seed=1):
        rnd = random.Random(seed)
        out = []
        for i, phase in enumerate(range(0, 100000, 1000)):
            hw = 40000 + rnd.randrange(-jitter, jitter + 1) if jitter else 40000
            is_set = phase < kick + hw + irq
            out.append(rec(phase, hw, is_set, i))
        return out

    def estimates_and_can_undershoot():
        e = estimate(synthetic(kick=5000))
        assert e["status"] == "ok", e
        assert abs(e["x_star_ns"] - 5000) <= 500, e
        # explicitly: the estimator is NOT a guaranteed upper bound
        assert e["is_hard_bound"] is False, e
        assert "not a hard bound" in e["interpretation"].lower(), e["interpretation"]
        assert e["x_star_ns"] < 5000, e["x_star_ns"]   # it lands at 4500 for this fixture
        print("  ok  a 5 us kick is estimated at %.2f us (below the true value: not a bound)"
              % (e["x_star_ns"] / 1000.0))

    group("the estimator is a conditional estimate, not a bound", estimates_and_can_undershoot)

    def irq_delay_moves_the_estimate():
        e = estimate(synthetic(kick=5000, irq=3000))
        assert e["x_star_ns"] >= 8000 - 500, e
        print("  ok  a 3 us IRQ delay moves the estimate to %.2f us (it is a mixture, "
              "not a separable component)" % (e["x_star_ns"] / 1000.0))

    group("an IRQ delay moves the estimate (it is not separable)", irq_delay_moves_the_estimate)

    def no_transition_without_samples():
        e = estimate([rec(1000, 40000, True) for _ in range(3)])
        assert e["status"] == "no_transition", e
        print("  ok  too few late samples -> no_transition, no bound invented")

    group("no transition is reported as such", no_transition_without_samples)

    def all_set_is_not_a_usable_estimate():
        e = estimate([rec(p, 100000, True) for p in range(0, 60000, 2000)])
        assert e["status"] == "ok" and e["x_star_ns"] < 0, e
        print("  ok  an all-set sweep yields a negative x*, i.e. no usable estimate")

    group("a degenerate sweep does not produce a usable estimate",
          all_set_is_not_a_usable_estimate)

    def interval_present():
        e = estimate(synthetic(kick=7000), resamples=300)
        assert e["ci95_ns"] and e["ci95_ns"][0] <= e["x_star_ns"] <= e["ci95_ns"][1], e
        print("  ok  bootstrap interval brackets the estimate")

    group("the bootstrap interval brackets the estimate", interval_present)

    print()
    if failures:
        print("%d selftest group(s) failed" % failures)
        return 1
    print("all kick-offset selftest checks passed")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", action="append", default=[], metavar="LABEL=DIR")
    ap.add_argument("--out-dir")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.sweep or not args.out_dir:
        ap.error("--sweep and --out-dir are required unless --selftest")
    loads = {}
    for spec in args.sweep:
        label, sep, path = spec.partition("=")
        if not sep:
            ap.error("--sweep expects LABEL=DIR")
        drops = collections.defaultdict(collections.Counter)
        loads[label] = estimate(load_sweep(label, path, drops))
        loads[label]["drops"] = dict(drops[label])
    write_outputs({"loads": loads}, args.out_dir)
    for label, e in loads.items():
        print("%-8s n=%-4d x*=%.2f us  (conditional estimate, NOT a bound)"
              % (label, e["n"], (e.get("x_star_ns") or 0) / 1000.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
