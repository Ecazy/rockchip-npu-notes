#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Same-process Tiny ViT A/B with only the compiled finite scan changed."""
from __future__ import annotations

import argparse, gzip, json, statistics, struct, time
from pathlib import Path

import numpy as np

from rknpu_backend.hybrid import HybridRunner, load_hybrid_program
from rknpu_backend.models import load_weight_npz
from rknpu_backend.runtime.ffi import LibRKPlan, configure_rocket_abi


def read_idx(path):
    with gzip.open(path, "rb") as f:
        f.read(4)
        dims = struct.unpack(">III", f.read(12))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)


def summary(values):
    values = sorted(float(x) for x in values)
    return {"n": len(values), "median_us": float(statistics.median(values)),
            "p10_us": values[(len(values)-1)//10], "min_us": values[0],
            "max_us": values[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans-dir", type=Path, required=True)
    ap.add_argument("--weights-dir", type=Path, required=True)
    ap.add_argument("--lib-fast", required=True)
    ap.add_argument("--lib-reference", required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--batch", type=int, choices=(1, 8), required=True)
    ap.add_argument("--limit", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.limit % args.batch:
        raise SystemExit("limit must be divisible by batch")
    images = read_idx(args.data_dir / "t10k-images-idx3-ubyte.gz")[:args.limit]
    program = load_hybrid_program(args.plans_dir / f"tiny_vit_b{args.batch}.rkhybrid.json",
                                  plans_dir=args.plans_dir)
    weights = load_weight_npz(args.weights_dir / "tiny_vit.npz")
    arms = {}
    runners = {}
    old_cache = __import__("os").environ.get("RK_PLAN_PREPARED_VALIDATION")
    __import__("os").environ["RK_PLAN_PREPARED_VALIDATION"] = "0"
    try:
        for name, libpath in (("fast", args.lib_fast),
                              ("reference", args.lib_reference)):
            lib = LibRKPlan(libpath)
            configure_rocket_abi(lib.cdll)
            fd = int(lib.cdll.rocket_open())
            if fd < 0:
                raise RuntimeError(f"rocket_open {name} failed: {fd}")
            arms[name] = (lib, fd)
            runner = HybridRunner(lib, program, weights,
                                  weights_path=args.weights_dir / "tiny_vit.npz",
                                  fd=fd, execution_mode="graph")
            runners[name] = runner
    except BaseException:
        for name, runner in list(runners.items()):
            try:
                runner.close()
            except Exception:
                pass
            else:
                lib, fd = arms[name]
                try:
                    lib.cdll.rocket_close(fd)
                except Exception:
                    pass
        for name, (lib, fd) in list(arms.items()):
            if name not in runners:
                try:
                    lib.cdll.rocket_close(fd)
                except Exception:
                    pass
        raise
    finally:
        import os
        if old_cache is None:
            os.environ.pop("RK_PLAN_PREPARED_VALIDATION", None)
        else:
            os.environ["RK_PLAN_PREPARED_VALIDATION"] = old_cache

    args.out.parent.mkdir(parents=True, exist_ok=True)
    samples = {name: [] for name in runners}
    paired_ratios = []
    sample_rows = []
    ratios = []
    round_rows = []
    all_logits = {name: np.empty((args.limit, 10), dtype=np.float32)
                  for name in runners}
    try:
        runners["fast"].run(images[:args.batch], stats=None)
        runners["reference"].run(images[:args.batch], stats=None)
        for ri in range(args.rounds):
            round_calls = {name: [] for name in runners}
            for ci in range(args.limit // args.batch):
                start, stop = ci * args.batch, (ci + 1) * args.batch
                order = ("fast", "reference") if (ci + ri) % 2 == 0 else ("reference", "fast")
                outputs, times = {}, {}
                for name in order:
                    t0 = time.perf_counter()
                    result = runners[name].run(images[start:stop], stats=None)
                    times[name] = (time.perf_counter() - t0) * 1e6
                    outputs[name] = np.array(result.outputs["logits"], copy=True)
                if not np.array_equal(outputs["fast"], outputs["reference"]):
                    raise RuntimeError(f"logits differ bitwise at round={ri+1}, call={ci}")
                paired_ratios.append(times["reference"] / times["fast"])
                sample_rows.append({"round": ri + 1, "call": ci + 1,
                                    "image_start": start, "image_stop": stop,
                                    "fast_us": times["fast"],
                                    "reference_us": times["reference"],
                                    "ratio_reference_over_fast": paired_ratios[-1]})
                for name in runners:
                    samples[name].append(times[name])
                    round_calls[name].append(times[name])
                    all_logits[name][start:stop] = outputs[name]
            row = {name: float(statistics.median(vals)) for name, vals in round_calls.items()}
            row["round"] = ri + 1
            row["ratio_reference_over_fast"] = row["reference"] / row["fast"]
            ratios.append(row["ratio_reference_over_fast"])
            round_rows.append(row)
        for name, arr in all_logits.items():
            np.save(args.out.parent / f"{name}_logits.npy", arr)
        result = {"schema_version": 1, "kind": "full-model-finite-scan-ab",
                  "batch": args.batch, "limit": args.limit, "rounds": args.rounds,
                  "prepared_validation": 0, "bitwise": True,
                  "timing": {k: summary(v) for k, v in samples.items()},
                  "rounds_detail": round_rows,
                  "samples_us": {"fast": samples["fast"],
                                 "reference": samples["reference"],
                                 "paired_reference_over_fast": paired_ratios,
                                 "paired_rows": sample_rows},
                  "round_ratio_median": float(statistics.median(ratios)),
                  "round_ratio_min": min(ratios), "round_ratio_max": max(ratios),
                  "round_ratio_p10": sorted(ratios)[(len(ratios)-1)//10],
                  "output_files": {k: f"{k}_logits.npy" for k in all_logits}}
        (args.out.parent / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        cleanup_errors = []
        for name, runner in runners.items():
            try:
                runner.close()
            except Exception as exc:
                cleanup_errors.append(f"{name} runner.close failed: {exc}")
            else:
                lib, fd = arms[name]
                try:
                    lib.cdll.rocket_close(fd)
                except Exception as exc:
                    cleanup_errors.append(f"{name} rocket_close failed: {exc}")
        if cleanup_errors:
            raise RuntimeError("teardown requires quarantine: " + "; ".join(cleanup_errors))


if __name__ == "__main__":
    raise SystemExit(main())
