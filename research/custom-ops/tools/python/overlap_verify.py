#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent evidence verifier for the same-core overlap / concurrency probe.

Handles both suites:
  matmul-lut : the original 256^3 MatMul vs DPU-only LUT experiment.
  gemm-gemv  : GEMM vs GEMV, amplified B8/B16, two-thread same/cross-core.

Checks the JSONL records are internally consistent before any timing claim:
missing, duplicated, failed, wrong-mask, incomplete, guard-failing or
numerically bad submissions are refused. Timing and descriptive
classifications are reported, not gated: a serialized result is a valid
outcome. Records without a `suite`/`phase` field default to
matmul-lut / main so the earlier evidence stays valid.

Usage: overlap_verify.py RESULTS_DIR [--json OUT]
"""
import json
import os
import statistics
import sys

# ---------------------------------------------------------------------------
# Per-suite arm tables. Each non-pair arm lists its blocking submits; each
# submit is the list of instance indices chained into that one submit.
# ---------------------------------------------------------------------------

GEMV8 = list(range(4, 12))
GEMV16 = list(range(4, 20))

ML_ARMS = {
    "A":    {"subs": [[0]]},
    "B":    {"subs": [[1]]},
    "S1":   {"subs": [[0], [1]]},
    "C_AB": {"subs": [[0, 1]]},
    "C_BA": {"subs": [[1, 0]]},
    "C_AA": {"subs": [[0, 2]]},
    "C_BB": {"subs": [[1, 3]]},
    "S3_A_CORE": {"subs": [[0]], "unwritten": [0]},
    "S3_CORE_B": {"subs": [[0, 1]], "unwritten": [0]},
    "S3_B_CORE": {"subs": [[1, 0]], "unwritten": [0]},
}

GG_ARMS = {
    "A":    {"subs": [[0]]},
    "B":    {"subs": [[1]]},
    "B8":   {"subs": [GEMV8]},
    "B16":  {"subs": [GEMV16]},
    "S_AB":   {"subs": [[0], [1]]},
    "S_AB8":  {"subs": [[0], GEMV8]},
    "S_AB16": {"subs": [[0], GEMV16]},
    "C_AB":   {"subs": [[0, 1]]},
    "C_BA":   {"subs": [[1, 0]]},
    "C_AB8":  {"subs": [[0] + GEMV8]},
    "C_B8A":  {"subs": [GEMV8 + [0]]},
    "C_AB16": {"subs": [[0] + GEMV16]},
    "C_B16A": {"subs": [GEMV16 + [0]]},
    "C_AA":   {"subs": [[0, 2]]},
    "C_BB":   {"subs": [[1, 3]]},
    "P_same":  {"pair": True, "masks": [1, 1]},
    "P_cross": {"pair": True, "masks": [1, 2]},
}

ML_KINDS = {0: "mm", 1: "lut", 2: "mm", 3: "lut"}
GG_KINDS = {i: "mm" for i in range(20)}
GG_PHYS = {0: 256, 2: 256}
GG_PHYS.update({i: 12 for i in range(1, 20) if i not in (0, 2)})

SUITES = {
    "matmul-lut": {"arms": ML_ARMS, "kinds": ML_KINDS, "physical_rows": {}},
    "gemm-gemv": {"arms": GG_ARMS, "kinds": GG_KINDS, "physical_rows": GG_PHYS},
}

TOL = {"mm": 0.02, "lut": 0.002}


def suite_of(rec):
    return rec.get("suite", "matmul-lut")


def phase_of(rec):
    return rec.get("phase", "main")


def load_run(path):
    submits, verifies, pairs, windows, wall, summary = {}, {}, {}, {}, {}, None
    jl = os.path.join(path, "probe.jsonl")
    if not os.path.isfile(jl):
        raise ValueError("probe.jsonl missing")
    with open(jl) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("probe.jsonl:%d not JSON: %s" % (lineno, exc))
            typ = rec.get("type")
            if typ == "submit":
                idx = rec.get("sub_index", rec.get("submit_index"))
                submits.setdefault((suite_of(rec), phase_of(rec), rec["arm"], rec["rep"]), {})[idx] = rec
            elif typ == "verify":
                verifies.setdefault((suite_of(rec), phase_of(rec), rec["arm"], rec["rep"]), {})[rec["inst"]] = rec
            elif typ == "arm":
                wall.setdefault((suite_of(rec), phase_of(rec), rec["arm"], rec["rep"]), rec)
            elif typ == "pair":
                pairs.setdefault((suite_of(rec), rec["arm"], rec["rep"]), {})[rec["thread"]] = rec
            elif typ == "pair_window":
                windows[(suite_of(rec), rec["arm"], rec["rep"])] = rec
            elif typ == "summary":
                summary = rec
            else:
                raise ValueError("probe.jsonl:%d unknown record type %r" % (lineno, typ))
    if summary is None:
        raise ValueError("summary record missing")
    return submits, verifies, pairs, windows, wall, summary


def check_run(path, errors):
    def err(msg):
        errors.append("%s: %s" % (os.path.basename(path), msg))

    try:
        submits, verifies, pairs, windows, wall, summary = load_run(path)
    except ValueError as exc:
        err(str(exc))
        return None

    if summary.get("status") != "PASS":
        err("summary status is %r" % summary.get("status"))
    if summary.get("failures"):
        err("summary reports %s probe-level failures" % summary["failures"])

    timed_out = {}
    seen = set()
    keys = set(submits) | set(verifies)
    for (suite, phase, arm, rep) in sorted(keys):
        spec = SUITES.get(suite, {}).get("arms", {}).get(arm)
        if spec is None:
            err("unknown arm %r in suite %r" % (arm, suite))
            continue
        if spec.get("pair"):
            err("pair arm %r has submit/verify records" % arm)
            continue
        seen.add((suite, phase, arm, rep))
        by_sub = submits.get((suite, phase, arm, rep), {})
        if len(by_sub) != len(spec["subs"]):
            err("%s/%s %s rep=%s has %d submits, expected %d" %
                (suite, phase, arm, rep, len(by_sub), len(spec["subs"])))
            continue
        arm_to = False
        for s, sub in enumerate(spec["subs"]):
            rec = by_sub.get(s)
            if rec is None:
                err("%s/%s %s rep=%s missing submit %d" % (suite, phase, arm, rep, s))
                continue
            if rec.get("requested_mask") != 1 or rec.get("actual_mask") != 1:
                err("%s/%s %s rep=%s sub=%d mask %s/%s not Core 0" %
                    (suite, phase, arm, rep, s, rec.get("requested_mask"), rec.get("actual_mask")))
            if rec.get("task_count") != len(sub):
                err("%s/%s %s rep=%s sub=%d task_count=%s expected %d" %
                    (suite, phase, arm, rep, s, rec.get("task_count"), len(sub)))
            res = rec.get("result")
            if res != 0:
                if arm.startswith("S3_") and res == -110:
                    arm_to = True
                    if rec.get("int_status") != 0 or rec.get("driver_service_ns") != 0:
                        err("%s rep=%s sub=%d timeout carries completion/service evidence" %
                            (arm, rep, s))
                else:
                    err("%s/%s %s rep=%s sub=%d result=%s" % (suite, phase, arm, rep, s, res))
            else:
                if not isinstance(rec.get("driver_service_ns"), int) or rec["driver_service_ns"] <= 0:
                    err("%s/%s %s rep=%s sub=%d service=%s" %
                        (suite, phase, arm, rep, s, rec.get("driver_service_ns")))
                if not isinstance(rec.get("int_status"), int) or (rec["int_status"] & 0x300) == 0:
                    err("%s/%s %s rep=%s sub=%d int_status=%s lacks a completion bit" %
                        (suite, phase, arm, rep, s, rec.get("int_status")))
                if "irq_delta" in rec and rec["irq_delta"] != [1, 0, 0]:
                    err("%s/%s %s rep=%s sub=%d irq_delta=%s expected exactly one Core 0 IRQ" %
                        (suite, phase, arm, rep, s, rec["irq_delta"]))
        timed_out[(suite, phase, arm, rep)] = arm_to
        used = {i for sub in spec["subs"] for i in sub}
        missing = used - set(verifies.get((suite, phase, arm, rep), {}))
        if missing:
            err("%s/%s %s rep=%s missing verify record for inst %s" %
                (suite, phase, arm, rep, sorted(missing)))

    if not seen and not pairs:
        err("no submissions recorded")
        return None

    # Pair groups.
    pkeys = set(pairs) | set(windows)
    for (suite, arm, rep) in sorted(pkeys):
        spec = SUITES.get(suite, {}).get("arms", {}).get(arm)
        if spec is None or not spec.get("pair"):
            err("pair record for unknown arm %r" % arm)
            continue
        recs = pairs.get((suite, arm, rep), {})
        if len(recs) != 2:
            err("%s %s rep=%s has %d pair submits, expected 2" % (suite, arm, rep, len(recs)))
        for t in (0, 1):
            rec = recs.get(t)
            if rec is None:
                err("%s %s rep=%s missing thread %d" % (suite, arm, rep, t))
                continue
            want = spec["masks"][t]
            if rec.get("requested_mask") != want or rec.get("actual_mask") != want:
                err("%s %s rep=%s thread %d mask %s/%s expected %d" %
                    (suite, arm, rep, t, rec.get("requested_mask"), rec.get("actual_mask"), want))
            if rec.get("result") != 0:
                err("%s %s rep=%s thread %d result=%s" % (suite, arm, rep, t, rec.get("result")))
            if not isinstance(rec.get("driver_service_ns"), int) or rec["driver_service_ns"] <= 0:
                err("%s %s rep=%s thread %d service=%s" % (suite, arm, rep, t, rec.get("driver_service_ns")))
            if not rec.get("written") or rec.get("bad"):
                err("%s %s rep=%s thread %d written=%s bad=%s" %
                    (suite, arm, rep, t, rec.get("written"), rec.get("bad")))
        win = windows.get((suite, arm, rep))
        if win is None:
            err("%s %s rep=%s missing pair_window" % (suite, arm, rep))
        else:
            irq = win.get("irq_delta")
            expect = [2, 0, 0] if spec["masks"] == [1, 1] else [1, 1, 0]
            if irq != expect:
                err("%s %s rep=%s irq_delta=%s expected %s" % (suite, arm, rep, irq, expect))
            if not isinstance(win.get("t_pair_ns"), int) or win["t_pair_ns"] <= 0:
                err("%s %s rep=%s t_pair_ns=%s" % (suite, arm, rep, win.get("t_pair_ns")))

    # Numerical / guard checks on the verified instances.
    for (suite, phase, arm, rep), recs in sorted(verifies.items()):
        spec = SUITES.get(suite, {}).get("arms", {}).get(arm)
        if spec is None or spec.get("pair"):
            continue
        used = {i for sub in spec["subs"] for i in sub}
        exp_unwritten = set(spec.get("unwritten", []))
        to = timed_out.get((suite, phase, arm, rep), False)
        for inst, rec in sorted(recs.items()):
            kind = rec.get("kind") or SUITES[suite]["kinds"].get(inst, "mm")
            tol = TOL.get(kind, 0.02)
            if rec.get("download_result") != 0:
                err("%s %s rep=%s inst=%d download failed" % (arm, phase, rep, inst))
            if "guard_ok" in rec and not rec["guard_ok"]:
                err("%s %s rep=%s inst=%d guard failed" % (arm, phase, rep, inst))
            if inst in used:
                if to:
                    if rec.get("written") and rec.get("bad"):
                        err("%s %s rep=%s inst=%d bad=%s after timeout" % (arm, phase, rep, inst, rec.get("bad")))
                elif inst in exp_unwritten:
                    if rec.get("written"):
                        err("%s %s rep=%s inst=%d expected unwritten but written" % (arm, phase, rep, inst))
                else:
                    if not rec.get("written"):
                        err("%s %s rep=%s inst=%d expected written but all-NaN" % (arm, phase, rep, inst))
                    if rec.get("bad"):
                        err("%s %s rep=%s inst=%d bad=%s max_abs=%s" %
                            (arm, phase, rep, inst, rec.get("bad"), rec.get("max_abs")))
                    if not (rec.get("max_abs") is not None and rec["max_abs"] <= tol):
                        err("%s %s rep=%s inst=%d max_abs=%s above %s" %
                            (arm, phase, rep, inst, rec.get("max_abs"), tol))
            else:
                if rec.get("written"):
                    err("%s %s rep=%s inst=%d touched though unused" % (arm, phase, rep, inst))
    return {"submits": submits, "verifies": verifies, "pairs": pairs,
            "windows": windows, "wall": wall, "summary": summary, "timed_out": timed_out}


def median(values):
    return statistics.median(values) if values else None


def summarize(data, phase="main"):
    submits = data["submits"]
    per_arm = {}
    for (suite, ph, arm, rep), by_sub in submits.items():
        if ph != phase:
            continue
        per_arm.setdefault((suite, arm), []).extend(
            (r["driver_service_ns"], r.get("host_done_ns", 0) - r.get("host_begin_ns", 0))
            for r in by_sub.values())
    stats = {}
    wall = {}
    for key, rec in data.get("wall", {}).items():
        suite, ph, arm, rep = key
        if ph != phase:
            continue
        wall.setdefault((suite, arm), []).append((rec["wall_ns"], rec["device_service_sum_ns"]))
    for key, vals in sorted(per_arm.items()):
        svc = [v[0] for v in vals]
        e2e = [v[1] for v in vals]
        w = wall.get(key, [])
        stats["%s/%s" % key] = {
            "n": len(svc),
            "median_service_ns": median(svc),
            "p10_service_ns": sorted(svc)[max(0, int(len(svc) * 0.1) - 1)],
            "p90_service_ns": sorted(svc)[min(len(svc) - 1, int(len(svc) * 0.9))],
            "median_api_response_ns": median(e2e),
            # wall covers submit..output download/verify; not a device figure.
            "median_wall_ns": median([x[0] for x in w]),
            "median_device_sum_ns": median([x[1] for x in w]),
        }
    derived = {}
    ml = {k.split("/", 1)[1]: v for k, v in stats.items() if k.startswith("matmul-lut/")}
    if all(a in ml for a in ("A", "B", "C_AB", "C_BB", "C_AA", "C_BA")):
        ma, mb = ml["A"]["median_service_ns"], ml["B"]["median_service_ns"]
        derived["matmul-lut"] = {
            "saving_AB_ns": ma + mb - ml["C_AB"]["median_service_ns"],
            "saving_BA_ns": ma + mb - ml["C_BA"]["median_service_ns"],
            "saving_BB_ns": 2 * mb - ml["C_BB"]["median_service_ns"],
            "saving_AA_ns": 2 * ma - ml["C_AA"]["median_service_ns"],
        }
        d = derived["matmul-lut"]
        d["hetero_extra_over_BB_ns"] = d["saving_AB_ns"] - d["saving_BB_ns"]

    gk = {k.split("/", 1)[1]: v for k, v in stats.items() if k.startswith("gemm-gemv/")}
    if all(a in gk for a in ("A", "B", "B8", "B16", "C_BB", "C_AB", "C_AB8", "C_AB16")):
        def s(a):
            return gk[a]["median_service_ns"]
        envelope = 2 * s("B") - s("C_BB")
        d = {
            "saving_AB_ns": s("A") + s("B") - s("C_AB"),
            "saving_AB8_ns": s("A") + s("B8") - s("C_AB8"),
            "saving_AB16_ns": s("A") + s("B16") - s("C_AB16"),
            "saving_BB_ns": envelope,
            "saving_2xB8_over_B16_ns": 2 * s("B8") - s("B16"),
        }
        d["hetero_extra_over_BB_ns"] = d["saving_AB_ns"] - d["saving_BB_ns"]
        d["hetero_extra_over_BB_8_ns"] = d["saving_AB8_ns"] - d["saving_BB_ns"]
        d["hetero_extra_over_BB_16_ns"] = d["saving_AB16_ns"] - d["saving_BB_ns"]
        derived["gemm-gemv"] = d

    pair_stats = {}
    for (suite, arm, rep), win in data["windows"].items():
        key = "%s/%s" % (suite, arm)
        pair_stats.setdefault(key, {"t_pair": [], "sum_service": [], "overlap": [], "wall": []})
        pair_stats[key]["t_pair"].append(win["t_pair_ns"])
        pair_stats[key]["sum_service"].append(win["sum_service_ns"])
        pair_stats[key]["overlap"].append(win["service_overlap_ns"])
        pair_stats[key]["wall"].append(win.get("wall_ns", 0))
    for key, v in sorted(pair_stats.items()):
        t = median(v["t_pair"])
        ss = median(v["sum_service"])
        pair_stats[key] = {
            "n": len(v["t_pair"]),
            "median_t_pair_ns": t,
            "median_sum_service_ns": ss,
            "median_service_overlap_ns": median(v["overlap"]),
            "median_wall_ns": median(v["wall"]),
            "sum_service_over_t_pair": (ss / t) if t else None,
        }
    return {"per_arm": stats, "derived": derived, "pair": pair_stats}


def classify(derived, pair, min_reps):
    out = {}
    if min_reps is not None and min_reps < 8:
        out["verdict"] = "insufficient-samples"
        out["min_reps"] = min_reps
        return out
    ml = derived.get("matmul-lut")
    if ml:
        margin = max(5000.0, 0.05 * max(ml["saving_BB_ns"], 1.0))
        if ml["hetero_extra_over_BB_ns"] > margin:
            out["matmul-lut"] = "cross-module-overlap"
        elif ml["hetero_extra_over_BB_ns"] < -margin:
            out["matmul-lut"] = "worse-than-same-kind"
        else:
            out["matmul-lut"] = "serialized-envelope-amortized"
        out["matmul-lut_min_reps"] = min_reps
    gg = derived.get("gemm-gemv")
    if gg:
        margin = max(5000.0, 0.05 * max(gg["saving_BB_ns"], 1.0))
        extras = [gg["hetero_extra_over_BB_ns"], gg["hetero_extra_over_BB_8_ns"],
                  gg["hetero_extra_over_BB_16_ns"]]
        if max(extras) > margin:
            out["gemm-gemv"] = "cross-module-overlap"
        elif all(e < -margin for e in extras):
            out["gemm-gemv"] = "worse-than-same-kind"
        else:
            out["gemm-gemv"] = "serialized-envelope-amortized"
        out["gemm-gemv_min_reps"] = min_reps
    for key, v in pair.items():
        ratio = v.get("sum_service_over_t_pair")
        if ratio is None:
            continue
        if ratio > 1.2:
            out[key] = "device-overlap"
        elif ratio < 1.05:
            out[key] = "serialized-or-envelope"
        else:
            out[key] = "ambiguous"
    return out


def verify_root(root):
    errors = []
    runs = {}
    entries = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and os.path.isfile(os.path.join(root, d, "probe.jsonl"))
    )
    if not entries:
        errors.append("%s: no run directory with probe.jsonl" % root)
    for name in entries:
        data = check_run(os.path.join(root, name), errors)
        if data:
            runs[name] = data
    report = {"root": root, "runs": {}, "errors": errors,
              "status": "PASS" if not errors else "FAIL"}
    for name, data in runs.items():
        timing = summarize(data)
        suite = data["summary"].get("suite", "matmul-lut")
        gk = {k: v["n"] for k, v in timing["per_arm"].items()}
        min_reps = min(gk.values()) if gk else 0
        report["runs"][name] = {
            "summary": data["summary"],
            "timing": timing,
            "timed_out": {"%s/%s/%s/rep%s" % k: v for k, v in sorted(data["timed_out"].items()) if v},
            "classification": classify(timing["derived"], timing["pair"], min_reps),
            "suite": suite,
        }
    return report


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    root = argv[1]
    out = None
    if len(argv) >= 4 and argv[2] == "--json":
        out = argv[3]
    if not os.path.isdir(root):
        print("not a directory: %s" % root, file=sys.stderr)
        return 2
    report = verify_root(root)
    if out:
        with open(out, "w") as f:
            f.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for name, run in sorted(report["runs"].items()):
        d = run["timing"]["derived"]
        print("%-10s suite=%-10s %-4s derived=%s" % (
            name, run["suite"], run["summary"]["status"],
            {k: (round(v, 1) if isinstance(v, (int, float)) else v) for k, v in d.items()}))
        for k, v in sorted(run["timing"]["pair"].items()):
            print("   %-16s %s" % (k, {kk: (round(vv, 3) if isinstance(vv, float) else vv)
                                       for kk, vv in v.items()}))
    if report["errors"]:
        for e in report["errors"]:
            print("ERROR " + e, file=sys.stderr)
        print("OVERLAP_VERIFY FAIL")
        return 1
    print("OVERLAP_VERIFY PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
