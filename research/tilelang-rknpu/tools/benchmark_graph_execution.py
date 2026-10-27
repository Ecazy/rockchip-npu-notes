#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Full-test application timing for same-policy Tiny ViT graph execution arms.

The timed path deliberately passes ``stats=None``: ``RunStats`` is an
instrumentation surface and its per-node bookkeeping is not application work.
The tool records a separate stats-instrumented sample for attribution and keeps
the two intervals clearly separated.

Arms are executed round-robin per static batch, with the order rotated for every
batch/round, so CPU DVFS and thermal drift are not assigned to one arm.  Every
arm consumes the same raw images and writes the same per-batch logits.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import statistics
import struct
import time
from pathlib import Path

import numpy as np

from rknpu_backend.hostinfo import thread_config
from rknpu_backend.hybrid import HybridRunner, load_hybrid_program
from rknpu_backend.models import load_weight_npz, RunStats
from rknpu_backend.runtime.ffi import (LibRKPlan, configure_rocket_abi,
                                      default_library)

ROOT = Path(__file__).resolve().parents[1]


def read_idx(path: Path, rank: int) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        f.read(4)
        dims = struct.unpack(">" + "I" * rank, f.read(4 * rank))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def median(values):
    return float(statistics.median(values)) if values else None


def parse_arm(text: str) -> tuple[str, str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            "arm must be NAME=EXECUTION_MODE[:MANIFEST_SUFFIX]")
    name, rest = text.split("=", 1)
    mode, sep, suffix = rest.partition(":")
    if not name or mode not in ("generic", "prepared", "graph", "auto"):
        raise argparse.ArgumentTypeError(f"invalid arm {text!r}")
    if sep and not suffix.startswith("_"):
        raise argparse.ArgumentTypeError("manifest suffix must start with _")
    return name, mode, suffix


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans-dir", type=Path,
                    default=ROOT / "models" / "compiled")
    ap.add_argument("--manifest-dir", type=Path,
                    help="directory containing rkhybrid manifests (default: "
                         "plans-dir)")
    ap.add_argument("--independent-policy", action="append", default=[],
                    help="arm name whose numeric policy is separately qualified")
    ap.add_argument("--weights-dir", type=Path, default=ROOT / "models")
    ap.add_argument("--data-dir", type=Path, default=ROOT / ".data" / "mnist")
    ap.add_argument("--lib", default=os.environ.get("RKNPU_PLAN_LIB", ""))
    ap.add_argument("--backend", choices=("mock", "board"), default="board")
    ap.add_argument("--limit", type=int, default=10000)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--stats-calls", type=int, default=100)
    ap.add_argument("--batch", type=int, action="append", default=[])
    ap.add_argument("--arm", type=parse_arm, action="append", default=[])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    batches = args.batch or [8, 1]
    arms = args.arm or [("generic", "generic", ""),
                        ("prepared", "prepared", ""),
                        ("auto", "auto", "")]
    if args.limit % max(batches) != 0:
        raise SystemExit("--limit must be divisible by every batch")
    if args.rounds < 1 or args.stats_calls < 1:
        raise SystemExit("rounds and stats-calls must be positive")
    plans_dir = args.plans_dir.resolve()
    data_dir = args.data_dir.resolve()
    images = read_idx(data_dir / "t10k-images-idx3-ubyte.gz", 3)[:args.limit]
    labels = read_idx(data_dir / "t10k-labels-idx1-ubyte.gz", 1)[:args.limit]
    weights_path = args.weights_dir.resolve() / "tiny_vit.npz"
    weights = load_weight_npz(weights_path)
    if args.backend == "mock":
        lib_path = args.lib or str(ROOT.parent / "custom-ops" /
                                   "build-host-mvp" /
                                   "librknpu_plan_mock.so")
        lib = LibRKPlan(lib_path)
        fd = 0
    else:
        lib = default_library() if not args.lib else LibRKPlan(args.lib)
        configure_rocket_abi(lib.cdll)
        fd = int(lib.cdll.rocket_open())
        if fd < 0:
            raise RuntimeError(f"rocket_open failed: {fd}")
    result = {
        "schema_version": 1,
        "kind": "tiny-vit-graph-execution-campaign",
        "backend": args.backend,
        "limit": args.limit,
        "rounds": args.rounds,
        "arms": [{"name": name, "requested_mode": mode,
                  "manifest_suffix": suffix,
                  "numeric_policy": ("independent"
                                     if name in args.independent_policy
                                     else "same")}
                 for name, mode, suffix in arms],
        "batches": batches,
        "cpu_thread_config": thread_config(),
        "timing": {
            "application": ("runner.run(..., stats=None); preprocess+execution+"
                            "output materialisation"),
            "instrumented": "separate RunStats sample; not included in app time",
            "argmax": "outside application interval",
        },
        "profiles": {},
    }
    try:
        for batch in batches:
            calls = args.limit // batch
            manifest_dir = (args.manifest_dir or plans_dir).resolve()
            programs = {}
            runners = {}
            setup_ms = {}
            for name, mode, suffix in arms:
                if suffix not in programs:
                    manifest = manifest_dir / \
                        f"tiny_vit_b{batch}{suffix}.rkhybrid.json"
                    programs[suffix] = load_hybrid_program(
                        manifest, plans_dir=plans_dir)
                t0 = time.perf_counter()
                runner = HybridRunner(lib, programs[suffix], weights,
                                      weights_path=weights_path, fd=fd,
                                      execution_mode=mode)
                setup_ms[name] = (time.perf_counter() - t0) * 1e3
                runners[name] = runner
            try:
                # Warm every workspace with the same first batch.
                for runner in runners.values():
                    runner.run(images[:batch], stats=None)
                round_rows = []
                numeric = {name: [] for name, _, _ in arms}
                generic_logits = None
                final_logits = {}
                for round_index in range(args.rounds):
                    logits = {
                        name: np.empty((args.limit, 10), dtype=np.float32)
                        for name, _, _ in arms
                    }
                    durations = {name: [] for name, _, _ in arms}
                    paired_ratios = {name: [] for name, _, _ in arms
                                     if name != "generic"}
                    for call_index in range(calls):
                        start = call_index * batch
                        stop = start + batch
                        order = list(runners)
                        shift = (call_index + round_index) % len(order)
                        order = order[shift:] + order[:shift]
                        timing = {}
                        for name in order:
                            t0 = time.perf_counter()
                            out = runners[name].run(images[start:stop],
                                                    stats=None)
                            timing[name] = (time.perf_counter() - t0) * 1e6
                            logits[name][start:stop] = out.outputs["logits"]
                        for name, value in timing.items():
                            durations[name].append(value)
                        if "generic" in timing:
                            for name in paired_ratios:
                                paired_ratios[name].append(
                                    timing["generic"] / timing[name])
                    round_logits = {}
                    generic_round = logits.get("generic")
                    for name, _, _ in arms:
                        values = logits[name]
                        round_logits[name] = values
                        final_logits[name] = values
                        accuracy = float(np.mean(values.argmax(axis=1) == labels))
                        row = {
                            "round": round_index,
                            "accuracy": accuracy,
                            "warm_ms_per_call": statistics.fmean(
                                durations[name]) / 1e3,
                            "warm_median_ms_per_call": median(durations[name]) / 1e3,
                            "throughput_images_s": args.limit / max(
                                1e-12, sum(durations[name]) / 1e6),
                            "logits_sha256": sha_bytes(values.tobytes()),
                            "paired_vs_generic_median": median(
                                paired_ratios.get(name, [])),
                        }
                        if generic_round is not None and name != "generic":
                            delta = np.abs(values.astype(np.float64) -
                                           generic_round.astype(np.float64))
                            row["bitwise_equal_generic"] = bool(
                                np.array_equal(values, generic_round))
                            row["max_abs_delta_generic"] = float(delta.max())
                            row["top1_agreement_generic"] = float(np.mean(
                                values.argmax(axis=1) ==
                                generic_round.argmax(axis=1)))
                        round_rows.append({"arm": name, **row})
                    generic_logits = generic_round if generic_round is not None \
                        else generic_logits
                    for name, _, _ in arms:
                        fname = args.out.with_name(
                            f"{args.out.stem}.vit_b{batch}.r{round_index + 1}."
                            f"{name}.logits.npy")
                        fname.parent.mkdir(parents=True, exist_ok=True)
                        np.save(fname, round_logits[name])
                # Separate instrumentation and one capture equality check.
                stats_rows = {}
                trace_rows = {}
                for name, runner in runners.items():
                    stats = RunStats()
                    t0 = time.perf_counter()
                    for i in range(args.stats_calls):
                        runner.run(images[i * batch:(i + 1) * batch], stats=stats)
                    stats_sec = time.perf_counter() - t0
                    stats_rows[name] = {
                        "calls": args.stats_calls,
                        "median_ms_per_call": 1e3 * stats_sec / args.stats_calls,
                        "stats_per_call": {
                            **{key: stats.to_dict()[key] / args.stats_calls
                               for key in (
                                   "plan_call_us", "plan_workspace_us",
                                   "host_action_us", "executor_overhead_us",
                                   "task_count", "submit_count",
                                   "upload_calls", "download_calls",
                                   "upload_bytes", "download_bytes",
                                   "graph_view_ops",
                                   "graph_skipped_trace_output_writes")},
                        },
                        "host_actions": {
                            row["name"]: row["elapsed_us"] /
                            args.stats_calls
                            for row in stats.host_actions},
                        "execution_selection": stats_rows.get(
                            name, {}).get("execution_selection"),
                    }
                for name, runner in runners.items():
                    trace = runner.run(images[:batch], capture=True).trace
                    trace_rows[name] = trace
                trace_equal = {}
                if "generic" in trace_rows:
                    reference = trace_rows["generic"]
                    for name in runners:
                        if name == "generic":
                            continue
                        trace_equal[name] = {
                            key: bool(np.array_equal(reference[key],
                                                     trace_rows[name][key]))
                            for key in reference
                            if key in trace_rows[name]
                        }
                summary = {}
                for name, runner in runners.items():
                    selected = runner.execution_selection
                    summary[name] = {
                        "requested_mode": runner.requested_execution_mode,
                        "selected_mode": runner.execution_mode,
                        "backend": runner.execution_backend,
                        "selection": selected,
                        "runner_setup_ms": setup_ms[name],
                        "resident_bytes": (
                            runner.residency.total_bytes
                            if runner.residency is not None else 0),
                        "rounds": [row for row in round_rows
                                   if row["arm"] == name],
                        "instrumented": stats_rows[name],
                        "trace_bitwise_equal_generic": trace_equal.get(name),
                    }
                    stats_rows[name]["execution_selection"] = selected
                result["profiles"][f"tiny_vit.b{batch}"] = summary
            finally:
                for runner in runners.values():
                    runner.close()
    finally:
        if args.backend == "board":
            lib.cdll.rocket_close(fd)
    # All same-policy arms must be bitwise equal in every round.
    problems = []
    for profile, summary in result["profiles"].items():
        for name, row in summary.items():
            independent = name in args.independent_policy
            for round_row in row["rounds"]:
                if round_row["accuracy"] < 0.95:
                    problems.append(
                        f"{profile}/{name}/r{round_row['round'] + 1}: "
                        "accuracy below 0.95")
                if not independent and name != "generic" and round_row.get(
                        "bitwise_equal_generic") is not True:
                    problems.append(
                        f"{profile}/{name}/r{round_row['round'] + 1}: "
                        "logits differ from generic")
            if name != "generic" and not independent:
                failed = [key for key, ok in
                          (row["trace_bitwise_equal_generic"] or {}).items()
                          if not ok]
                if failed:
                    problems.append(f"{profile}/{name}: trace differs: {failed}")
    result["status"] = "PASS" if not problems else "FAIL"
    result["problems"] = problems
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "out": str(args.out),
                      "problems": problems[:20]}, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
