#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Three-arm, same-schedule full-test benchmark for current, RKNN and CPU.

Every timed sample is one raw uint8 batch through arm preprocessing/inference
to an owned contiguous float32 [B,10] logits array.  Metrics, references,
artifact writes and JSON serialization happen after the sample timer.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import statistics
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "python"
TOOLS = ROOT / "tools"
for path in (str(PYTHON), str(TOOLS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from evaluate_models import make_batch_schedule, metrics  # noqa: E402
from rknpu_backend.hybrid import HybridRunner, load_hybrid_program  # noqa: E402
from rknpu_backend.hostinfo import thread_config  # noqa: E402
from rknpu_backend.model_reference import (  # noqa: E402
    mlp_logits, mlp_logits_declared, tiny_vit_logits,
    tiny_vit_logits_declared,
)
from rknpu_backend.models import load_weight_npz  # noqa: E402
from rknpu_backend.runtime.ffi import (  # noqa: E402
    LibRKPlan, configure_rocket_abi,
)

PROFILES = (
    ("mlp_b32", "mnist_mlp", 32),
    ("mlp_b8", "mnist_mlp", 8),
    ("mlp_b1", "mnist_mlp", 1),
    ("vit_b8", "tiny_vit", 8),
    ("vit_b1", "tiny_vit", 1),
)
ARMS = ("current", "vendor", "cpu")


def _add_error_context(error: BaseException, message: str) -> None:
    """Preserve the original exception on Python 3.10 while adding context."""
    add_note = getattr(error, "add_note", None)
    if add_note is not None:
        add_note(message)
    else:
        try:
            error.args = (*error.args, message)
        except BaseException:
            pass


def read_idx(path: Path, rank: int) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        header = stream.read(4 + 4 * rank)
        if len(header) != 4 + 4 * rank:
            raise ValueError(f"truncated IDX header: {path}")
        magic = struct.unpack(">I", header[:4])[0]
        if magic != (0x0800 + rank):
            raise ValueError(f"invalid IDX magic {magic:#x}: {path}")
        dims = struct.unpack(">" + "I" * rank, header[4:])
        payload = stream.read()
    expected = int(np.prod(dims))
    if len(payload) != expected:
        raise ValueError(f"IDX payload size mismatch: {path}")
    return np.frombuffer(payload, dtype=np.uint8).reshape(dims)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def loaded_vendor_runtime_maps(expected_path: Path | None = None) -> list[dict[str, Any]]:
    """Return the actual librknnrt mappings, including kernel file identity."""
    maps_path = Path("/proc/self/maps")
    if not maps_path.exists():
        return []
    files: dict[tuple[str, str, str], dict[str, Any]] = {}
    for line in maps_path.read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) < 6 or "librknnrt" not in parts[5].lower():
            continue
        key = (parts[5], parts[3], parts[4])
        entry = files.setdefault(key, {
            "path": parts[5], "device": parts[3], "inode": int(parts[4]),
            "ranges": [],
        })
        entry["ranges"].append({"address_range": parts[0],
                                "permissions": parts[1], "offset": parts[2]})
    mapped = [files[key] for key in sorted(files)]
    if expected_path is not None:
        expected = expected_path.resolve()
        stat = expected.stat()
        import os
        expected_dev = stat.st_dev
        expected_hash = sha256_file(expected)
        matches = []
        for entry in mapped:
            path = Path(entry["path"].removesuffix(" (deleted)"))
            try:
                mapped_stat = path.stat()
                mapped_hash = sha256_file(path)
            except OSError:
                continue
            if (mapped_stat.st_ino == entry["inode"] and
                    mapped_stat.st_dev == os.makedev(
                        int(entry["device"].split(":")[0], 16),
                        int(entry["device"].split(":")[1], 16))):
                entry["mapped_file_sha256"] = mapped_hash
                entry["matches_expected_runtime"] = (
                    mapped_stat.st_ino == stat.st_ino and
                    mapped_stat.st_dev == expected_dev and
                    mapped_hash == expected_hash)
                matches.append(entry["matches_expected_runtime"])
        if not any(matches):
            raise RuntimeError(
                f"loaded librknnrt mapping does not identify expected runtime {expected}")
        for entry in mapped:
            entry["expected_runtime_path"] = str(expected)
            entry["expected_runtime_sha256"] = expected_hash
    return mapped


def digest_array(array: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_samples(samples: list[dict[str, Any]], arm: str,
                      *, role: str | None = None) -> dict[str, Any]:
    times = [float(row["seconds_by_arm"][arm]) for row in samples
             if role is None or row["role"] == role]
    return {
        "calls": len(times),
        "mean_us": float(statistics.fmean(times) * 1e6) if times else None,
        "median_us": float(statistics.median(times) * 1e6) if times else None,
        "p90_us": percentile(times, 90.0) * 1e6 if times else None,
        "sum_seconds": float(sum(times)),
    }


def _profile_schedule(model: str, batch: int, limit: int) -> list[dict[str, Any]]:
    tails = (8, 1) if model == "mnist_mlp" else ()
    return make_batch_schedule(limit, batch, tail_profiles=tails)


def _program_path(manifest: dict[str, Any], plans_dir: Path, key: str) -> Path:
    entry = manifest.get("hybrid_manifests", {}).get(key)
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise ValueError(f"compiled manifest has no hybrid program {key!r}")
    return plans_dir / entry["path"]


def _owned_logits(value: Any, batch: int, *, force_copy: bool = False) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (batch, 10) or array.dtype != np.dtype(np.float32):
        raise RuntimeError(
            f"expected float32 logits [{batch},10], got {array.dtype} {array.shape}")
    if (force_copy or not array.flags.c_contiguous or
            not array.flags.owndata):
        array = np.array(array, dtype=np.float32, order="C", copy=True)
    return array


def _cpu_call(model: str, weights: dict[str, np.ndarray], raw: np.ndarray) -> np.ndarray:
    fn = mlp_logits if model == "mnist_mlp" else tiny_vit_logits
    return _owned_logits(fn(weights, raw), len(raw))


def _declared_reference(model: str, weights: dict[str, np.ndarray],
                        images: np.ndarray, schedule: list[dict[str, Any]]) -> np.ndarray:
    fn = mlp_logits_declared if model == "mnist_mlp" else tiny_vit_logits_declared
    out = np.empty((len(images), 10), dtype=np.float32)
    for row in schedule:
        start, size = int(row["offset"]), int(row["size"])
        out[start:start + size] = fn(weights, images[start:start + size])
    return out


def _rknn_input(raw: np.ndarray) -> np.ndarray:
    return raw.reshape(len(raw), 784).astype(np.float32)


def _close_vendor(contexts: dict[tuple[str, int], Any],
                  active_error: BaseException | None) -> None:
    errors: list[BaseException] = []
    for context in reversed(list(contexts.values())):
        try:
            rc = context.release()
            if rc not in (None, 0):
                errors.append(RuntimeError(f"RKNNLite.release failed: {rc}"))
        except BaseException as exc:
            errors.append(exc)
    if errors:
        if active_error is not None:
            _add_error_context(active_error, "vendor context cleanup also failed: " +
                               "; ".join(repr(error) for error in errors))
        else:
            raise errors[0]


def _close_current(runners: dict[tuple[str, int], HybridRunner], lib: LibRKPlan,
                   fd: int | None, *, quarantined: bool,
                   active_error: BaseException | None) -> None:
    errors: list[BaseException] = []
    for runner in reversed(list(runners.values())):
        try:
            runner.close()
        except BaseException as exc:
            errors.append(exc)
    if errors:
        quarantined = True
    if fd is not None and not quarantined and not errors:
        try:
            lib.cdll.rocket_close(fd)
        except BaseException as exc:
            errors.append(exc)
    if errors:
        if active_error is not None:
            _add_error_context(active_error, "current runtime cleanup also failed; fd "
                               "retained when workspace state was uncertain: " +
                               "; ".join(repr(error) for error in errors))
        else:
            raise errors[0]


def _call_current(runner: HybridRunner, raw: np.ndarray) -> np.ndarray:
    result = runner.run(raw, stats=None)
    return _owned_logits(result.outputs["logits"], len(raw))


def _call_vendor(context: Any, raw: np.ndarray) -> np.ndarray:
    prepared = _rknn_input(raw)
    outputs = context.inference(inputs=[prepared], data_type="float32")
    if outputs is None or len(outputs) != 1:
        raise RuntimeError("RKNNLite returned an unexpected output list")
    # RKNNLite may recycle its output storage, so take one owned copy inside the
    # timed call.  The current and CPU arms already return owned arrays normally.
    return _owned_logits(outputs[0], len(raw), force_copy=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    plans_dir = args.plans_dir.resolve()
    weights_dir = args.weights_dir.resolve()
    data_dir = args.data_dir.resolve()
    rknn_dir = args.rknn_dir.resolve()
    out_path = args.out.resolve()
    artifact_dir = out_path.parent / f"{out_path.stem}_artifacts"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if not 1 <= args.limit <= 10000:
        raise ValueError("--limit must be in [1,10000]")
    if args.rounds < 1:
        raise ValueError("--rounds must be positive")
    selected = [profile for profile in PROFILES
                if not args.profile or profile[0] in args.profile]
    if not selected:
        raise ValueError("no profile selected")
    selected_arms = ARMS if args.mode == "all" else (args.mode,)

    data_load_start = time.perf_counter()
    images_all = read_idx(data_dir / "t10k-images-idx3-ubyte.gz", 3)
    labels_all = read_idx(data_dir / "t10k-labels-idx1-ubyte.gz", 1)
    data_load_seconds = time.perf_counter() - data_load_start
    if len(images_all) != 10000 or len(labels_all) != 10000:
        raise ValueError("MNIST test IDX files must each contain exactly 10000 items")
    if len(images_all) != len(labels_all):
        raise ValueError("MNIST test image/label count mismatch")
    images = images_all[:args.limit]
    labels = labels_all[:args.limit].astype(np.int64)

    weights_paths = {
        "mnist_mlp": weights_dir / "mnist_mlp.npz",
        "tiny_vit": weights_dir / "tiny_vit.npz",
    }
    weights: dict[str, dict[str, np.ndarray]] = {}
    weight_load_seconds: dict[str, float] = {}
    for _, model, _ in selected:
        if model not in weights:
            t0 = time.perf_counter()
            weights[model] = load_weight_npz(weights_paths[model])
            weight_load_seconds[model] = time.perf_counter() - t0

    compiled_manifest = json.loads((plans_dir / "manifest.json").read_text())
    vendor_manifest: dict[str, Any] = {}
    if "vendor" in selected_arms:
        vendor_manifest = json.loads((rknn_dir / "rknn-manifest.json").read_text())

    schedules: dict[str, list[dict[str, Any]]] = {}
    required_contexts: dict[str, set[tuple[str, int]]] = {
        arm: set() for arm in ARMS
    }
    for profile_name, model, batch in selected:
        schedule = _profile_schedule(model, batch, args.limit)
        if not schedule:
            raise ValueError(f"empty schedule for {profile_name}")
        if not any(row["size"] == batch and row["role"] == "primary"
                   for row in schedule):
            raise ValueError(f"--limit must include at least one primary batch for "
                             f"{profile_name} (need >= {batch})")
        schedules[profile_name] = schedule
        needed = {int(row["profile"]) for row in schedule}
        for size in needed:
            required_contexts["current"].add((model, size))
            required_contexts["vendor"].add((model, size))

    lib: LibRKPlan | None = None
    fd: int | None = None
    runners: dict[tuple[str, int], HybridRunner] = {}
    current_program_identity: dict[str, dict[str, str]] = {}
    vendor_contexts: dict[tuple[str, int], Any] = {}
    vendor_model_identity: dict[str, dict[str, str]] = {}
    current_quarantined = False
    active_error: BaseException | None = None
    setup_seconds: dict[str, dict[str, float]] = {arm: {} for arm in selected_arms}
    warmup_seconds: dict[str, dict[str, float]] = {arm: {} for arm in selected_arms}
    results: dict[str, Any] = {}
    samples_by_profile: dict[str, list[dict[str, Any]]] = {}
    logits_by_round: dict[tuple[str, int, str], np.ndarray] = {}
    declared_by_profile: dict[str, np.ndarray] = {}
    cpu_reference_by_profile: dict[str, np.ndarray] = {}
    reference_seconds: dict[str, float] = {}
    device_open_seconds = 0.0
    vendor_runtime_maps: list[dict[str, Any]] = []
    compatibility_smoke: dict[str, Any] = {"status": "not_run"}
    rknnlite_type = None

    def init_current_context(key: tuple[str, int]) -> None:
        if key in runners:
            return
        if lib is None or fd is None:
            raise RuntimeError("current runtime was not opened")
        model, batch = key
        program_key = f"{model}_b{batch}"
        program_path = _program_path(compiled_manifest, plans_dir, program_key)
        program_entry = compiled_manifest["hybrid_manifests"][program_key]
        expected_hash = program_entry.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"compiled manifest has no valid sha256 for {program_key}")
        actual_hash = sha256_file(program_path)
        if actual_hash != expected_hash:
            raise ValueError(f"hybrid program hash mismatch for {program_key}: "
                             f"manifest={expected_hash}, actual={actual_hash}")
        current_program_identity[program_key] = {
            "path": str(program_path), "manifest_sha256": expected_hash,
            "actual_sha256": actual_hash,
        }
        program = load_hybrid_program(program_path, plans_dir=plans_dir)
        t0 = time.perf_counter()
        runners[key] = HybridRunner(
            lib, program, weights[model], weights_path=weights_paths[model],
            fd=fd, execution_mode="auto")
        setup_seconds["current"][program_key] = time.perf_counter() - t0

    def init_vendor_context(key: tuple[str, int]) -> None:
        if key in vendor_contexts:
            return
        if rknnlite_type is None:
            raise RuntimeError("RKNN Lite is unavailable")
        model, batch = key
        profile_key = f"{model}_b{batch}"
        entry = vendor_manifest.get("profiles", {}).get(profile_key)
        if not isinstance(entry, dict) or not isinstance(entry.get("rknn"), str):
            raise ValueError(f"vendor manifest has no RKNN profile {profile_key!r}")
        rknn_path = rknn_dir / entry["rknn"]
        expected_hash = entry.get("rknn_sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"vendor manifest has no valid rknn_sha256 for {profile_key}")
        actual_hash = sha256_file(rknn_path)
        if actual_hash != expected_hash:
            raise ValueError(f"RKNN file hash mismatch for {profile_key}: "
                             f"manifest={expected_hash}, actual={actual_hash}")
        vendor_model_identity[profile_key] = {
            "path": str(rknn_path), "manifest_sha256": expected_hash,
            "actual_sha256": actual_hash,
        }
        t0 = time.perf_counter()
        context = rknnlite_type(verbose=False)
        rc = context.load_rknn(str(rknn_path))
        if rc:
            raise RuntimeError(f"load_rknn failed {profile_key}: {rc}")
        rc = context.init_runtime(target=None,
                                  core_mask=rknnlite_type.NPU_CORE_0)
        if rc:
            init_error = RuntimeError(f"init_runtime failed {profile_key}: {rc}")
            try:
                release_rc = context.release()
                if release_rc not in (None, 0):
                    _add_error_context(
                        init_error,
                        f"RKNNLite.release during failed init returned {release_rc}")
            except BaseException as cleanup:
                _add_error_context(
                    init_error,
                    f"RKNNLite.release during failed init also failed: {cleanup!r}")
            raise init_error
        vendor_contexts[key] = context
        setup_seconds["vendor"][profile_key] = time.perf_counter() - t0

    try:
        if "current" in selected_arms:
            lib = LibRKPlan(str(args.lib.resolve()))
            configure_rocket_abi(lib.cdll)
            t0 = time.perf_counter()
            opened = int(lib.cdll.rocket_open())
            if opened < 0:
                raise RuntimeError(f"rocket_open failed: {opened}")
            fd = opened
            device_open_seconds = time.perf_counter() - t0

        if "vendor" in selected_arms:
            from rknnlite.api import RKNNLite
            rknnlite_type = RKNNLite

        # Prove current/vendor coexistence before the long run.  The final
        # current call must reproduce the first logits after vendor and CPU work.
        if args.mode == "all":
            profile_name, model, _primary_batch = selected[0]
            first = schedules[profile_name][0]
            raw = images[int(first["offset"]):
                         int(first["offset"]) + int(first["size"])]
            batch_size = len(raw)
            smoke_key = (model, batch_size)
            init_current_context(smoke_key)
            init_vendor_context(smoke_key)
            runner = runners[(model, batch_size)]
            vendor = vendor_contexts[(model, batch_size)]
            t0 = time.perf_counter()
            logits_first = _call_current(runner, raw)
            current_first_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            vendor_logits = _call_vendor(vendor, raw)
            vendor_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            cpu_logits = _cpu_call(model, weights[model], raw)
            cpu_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            logits_last = _call_current(runner, raw)
            current_last_s = time.perf_counter() - t0
            equal = bool(np.array_equal(logits_first, logits_last))
            if not equal:
                raise RuntimeError("coexistence smoke failed: current logits changed "
                                   "across vendor/CPU calls")
            compatibility_smoke = {
                "status": "passed", "profile": profile_name,
                "batch_size": batch_size,
                "order": ["current", "vendor", "cpu", "current"],
                "current_logits_bitwise_equal": equal,
                "seconds": {"current_first": current_first_s,
                            "vendor": vendor_s, "cpu": cpu_s,
                            "current_last": current_last_s},
                "logits_sha256": digest_array(logits_first),
                "vendor_logits_sha256": digest_array(vendor_logits),
                "cpu_logits_sha256": digest_array(cpu_logits),
            }
            del logits_first, logits_last, vendor_logits, cpu_logits

        for key in sorted(required_contexts["current"]):
            if "current" in selected_arms:
                init_current_context(key)
        for key in sorted(required_contexts["vendor"]):
            if "vendor" in selected_arms:
                init_vendor_context(key)
        if "vendor" in selected_arms:
            expected_runtime = Path(os.environ.get(
                "RKNN_RUNTIME_EXPECTED", "/usr/lib/librknnrt.so"))
            vendor_runtime_maps = loaded_vendor_runtime_maps(expected_runtime)
            if not vendor_runtime_maps:
                raise RuntimeError("RKNNLite contexts initialized but no "
                                   "librknnrt mapping is visible in /proc/self/maps")

        for profile_name, model, batch in selected:
            schedule = schedules[profile_name]
            used_sizes = sorted({int(row["profile"]) for row in schedule},
                                reverse=True)
            # One warmup per distinct context/profile, outside every timed round.
            for arm in selected_arms:
                t0 = time.perf_counter()
                for size in used_sizes:
                    first = next(row for row in schedule
                                 if int(row["profile"]) == size)
                    start = int(first["offset"])
                    raw = images[start:start + int(first["size"])]
                    if arm == "current":
                        _call_current(runners[(model, size)], raw)
                    elif arm == "vendor":
                        _call_vendor(vendor_contexts[(model, size)], raw)
                    else:
                        _cpu_call(model, weights[model], raw)
                warmup_seconds[arm][profile_name] = time.perf_counter() - t0

            profile_samples: list[dict[str, Any]] = []
            round_cpu_logits: dict[int, np.ndarray] = {}
            for round_index in range(args.rounds):
                round_logits: dict[str, np.ndarray] = {
                    arm: np.empty((args.limit, 10), dtype=np.float32)
                    for arm in selected_arms
                }
                for call_index, row in enumerate(schedule):
                    start, size = int(row["offset"]), int(row["size"])
                    raw = images[start:start + size]
                    order = list(selected_arms)
                    if len(order) > 1:
                        shift = (call_index + round_index) % len(order)
                        order = order[shift:] + order[:shift]
                    timings: dict[str, float] = {}
                    for arm in order:
                        t0 = time.perf_counter()
                        if arm == "current":
                            output = _call_current(
                                runners[(model, int(row["profile"]))], raw)
                        elif arm == "vendor":
                            output = _call_vendor(
                                vendor_contexts[(model, int(row["profile"]))], raw)
                        else:
                            output = _cpu_call(model, weights[model], raw)
                        timings[arm] = time.perf_counter() - t0
                        # Store after the timer, then drop this batch's owned
                        # output before allocating the next one.  Holding 10k
                        # small arrays until round end perturbs allocator/GC and
                        # cache pressure without helping the measurement.
                        round_logits[arm][start:start + size] = output
                        del output
                    profile_samples.append({
                        "profile": profile_name, "round": round_index + 1,
                        "call_index": call_index,
                        "offset": start, "size": size,
                        "role": row["role"], "profile_batch": row["profile"],
                        "order": order,
                        "seconds_by_arm": timings,
                    })

                for arm in selected_arms:
                    logits_by_round[(profile_name, round_index + 1, arm)] = \
                        round_logits[arm]
                if "cpu" in round_logits:
                    round_cpu_logits[round_index + 1] = round_logits["cpu"]

            profile_samples.sort(key=lambda row: (row["round"], row["call_index"]))
            samples_by_profile[profile_name] = profile_samples

            reference_start = time.perf_counter()
            if "cpu" in selected_arms:
                cpu_reference = round_cpu_logits[1]
            else:
                cpu_reference = np.empty((args.limit, 10), dtype=np.float32)
                for row in schedule:
                    start, size = int(row["offset"]), int(row["size"])
                    cpu_reference[start:start + size] = _cpu_call(
                        model, weights[model], images[start:start + size])
            cpu_reference_by_profile[profile_name] = cpu_reference
            declared = _declared_reference(model, weights[model], images, schedule)
            reference_seconds[profile_name] = time.perf_counter() - reference_start
            declared_by_profile[profile_name] = declared

            round_summaries = []
            for round_index in range(1, args.rounds + 1):
                this_samples = [row for row in profile_samples
                                if row["round"] == round_index]
                ref = round_cpu_logits.get(round_index, cpu_reference)
                arm_metrics: dict[str, Any] = {}
                for arm in selected_arms:
                    arm_logits = logits_by_round[(profile_name, round_index, arm)]
                    arm_metrics[arm] = metrics(arm_logits, ref, labels, declared)
                per_arm_timing = {}
                for arm in selected_arms:
                    primary = summarize_samples(this_samples, arm, role="primary")
                    tail = summarize_samples(this_samples, arm, role="tail")
                    total_seconds = sum(
                        float(sample["seconds_by_arm"][arm])
                        for sample in this_samples)
                    per_arm_timing[arm] = {
                        "primary": primary,
                        "tail": tail,
                        "all_calls_sum_seconds": total_seconds,
                        "throughput_images_s": args.limit / max(1e-12, total_seconds),
                    }
                round_summaries.append({
                    "round": round_index,
                    "timing_by_arm": per_arm_timing,
                    "metrics_by_arm_vs_cpu_reference": arm_metrics,
                    "logits_sha256_by_arm": {
                        arm: digest_array(logits_by_round[
                            (profile_name, round_index, arm)])
                        for arm in selected_arms
                    },
                })

            overall = {}
            for arm in selected_arms:
                prim_means = [row["timing_by_arm"][arm]["primary"]["mean_us"]
                              for row in round_summaries]
                prim_medians = [row["timing_by_arm"][arm]["primary"]["median_us"]
                                for row in round_summaries]
                tails = [row["timing_by_arm"][arm]["tail"]["mean_us"]
                         for row in round_summaries
                         if row["timing_by_arm"][arm]["tail"]["calls"]]
                tail_medians = [row["timing_by_arm"][arm]["tail"]["median_us"]
                                for row in round_summaries
                                if row["timing_by_arm"][arm]["tail"]["calls"]]
                overall[arm] = {
                    "primary_round_mean_us": prim_means,
                    "primary_round_median_us": prim_medians,
                    "median_of_primary_round_means_us": float(
                        statistics.median(prim_means)),
                    "median_of_primary_round_medians_us": float(
                        statistics.median(prim_medians)),
                    "tail_round_mean_us": tails,
                    "tail_round_median_us": tail_medians,
                    "tail_mean_us_across_rounds": (
                        float(statistics.mean(tails)) if tails else None),
                    "tail_median_us_across_rounds": (
                        float(statistics.median(tail_medians))
                        if tail_medians else None),
                    "per_round_throughput_images_s": [
                        row["timing_by_arm"][arm]["throughput_images_s"]
                        for row in round_summaries
                    ],
                }

            # Keep all expensive reference and metrics work out of timed samples.
            results[profile_name] = {
                "model": model, "primary_batch": batch, "images": args.limit,
                "batch_schedule": {
                    "calls": len(schedule),
                    "batch_histogram": {
                        str(size): sum(row["size"] == size for row in schedule)
                        for size in sorted({row["size"] for row in schedule})
                    },
                    "tail_calls": sum(row["role"] == "tail" for row in schedule),
                    "rows": schedule,
                },
                "setup_seconds_by_context": {
                    arm: {f"{model}_b{size}": setup_seconds[arm].get(
                        f"{model}_b{size}") for size in used_sizes}
                    for arm in selected_arms if arm != "cpu"
                },
                "current_execution_by_context": ({
                    f"{model}_b{size}": {
                        "requested_mode": runners[(model, size)].requested_execution_mode,
                        "selected_mode": runners[(model, size)].execution_mode,
                        "backend": runners[(model, size)].execution_backend,
                        "selection": runners[(model, size)].execution_selection,
                    }
                    for size in used_sizes
                } if "current" in selected_arms else None),
                "warmup_seconds_by_arm": {
                    arm: warmup_seconds[arm][profile_name]
                    for arm in selected_arms
                },
                "reference_seconds": reference_seconds[profile_name],
                "rounds": round_summaries,
                "overall": overall,
                "numeric_policy_note": (
                    "vendor logits are reported diagnostically against CPU and "
                    "declared references; no packed-path admission gate is applied"
                    if model == "tiny_vit" else
                    "metrics are diagnostic; existing current admission limits "
                    "are not modified by this benchmark"),
            }

    except BaseException as exc:
        active_error = exc
        if "current" in selected_arms:
            current_quarantined = True
        raise
    finally:
        cleanup_errors: list[BaseException] = []
        if vendor_contexts:
            try:
                _close_vendor(vendor_contexts, active_error)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if runners and lib is not None:
            try:
                _close_current(runners, lib, fd,
                              quarantined=current_quarantined,
                              active_error=active_error)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            if active_error is not None:
                _add_error_context(active_error, "benchmark cleanup failed: " +
                                   "; ".join(repr(e) for e in cleanup_errors))
            else:
                raise cleanup_errors[0]

    # Persist only after every selected arm/profile/round and cleanup succeeded.
    raw_path = artifact_dir / "raw_samples.jsonl"
    with raw_path.open("w", encoding="utf-8") as stream:
        for profile_name in [profile[0] for profile in selected]:
            for sample in samples_by_profile[profile_name]:
                stream.write(json.dumps(sample, sort_keys=True) + "\n")
    logits_paths: dict[str, dict[str, str]] = {}
    for (profile_name, round_index, arm), logits in logits_by_round.items():
        path = artifact_dir / f"logits.{profile_name}.round{round_index}.{arm}.npy"
        np.save(path, logits, allow_pickle=False)
        logits_paths.setdefault(profile_name, {})[
            f"round{round_index}.{arm}"] = path.name
    for profile_name, declared in declared_by_profile.items():
        path = artifact_dir / f"logits.{profile_name}.declared.npy"
        np.save(path, declared, allow_pickle=False)
        logits_paths.setdefault(profile_name, {})["declared"] = path.name
    for profile_name, cpu_reference in cpu_reference_by_profile.items():
        path = artifact_dir / f"logits.{profile_name}.cpu_reference.npy"
        np.save(path, cpu_reference, allow_pickle=False)
        logits_paths.setdefault(profile_name, {})["cpu_reference"] = path.name

    input_paths = [data_dir / "t10k-images-idx3-ubyte.gz",
                   data_dir / "t10k-labels-idx1-ubyte.gz"]
    weight_files = [weights_paths[model] for model in sorted(weights)]
    report = {
        "schema_version": 1,
        "kind": "current-vendor-cpu-three-arm-benchmark",
        "mode": args.mode,
        "arms": list(selected_arms),
        "limit": args.limit,
        "rounds": args.rounds,
        "profiles": results,
        "compatibility_smoke": compatibility_smoke,
        "vendor_runtime_loaded_maps": vendor_runtime_maps,
        "setup_seconds": {
            "data_load": data_load_seconds,
            "device_open": device_open_seconds,
            "weights_by_model": weight_load_seconds,
        },
        "timing_contract": {
            "timed_call": "raw uint8 batch -> arm preprocessing/inference -> "
                           "owned C-contiguous float32 [B,10] logits",
            "current": "HybridRunner.run(raw, stats=None)",
            "vendor": "uint8 reshape/cast float32 + RKNNLite inference + owned copy",
            "cpu": "mlp_logits/tiny_vit_logits raw preprocessing + FP32 forward",
            "excluded": ["argmax", "numeric metrics", "reference calculation",
                         "artifact writes", "JSON serialization"],
            "round_order": "rotated by (batch_call_index + round_index) modulo arms",
            "primary_and_tail": "summarized separately; throughput uses sum of all "
                                "per-call durations including tails",
        },
        "artifacts": {
            "directory": str(artifact_dir),
            "raw_samples_jsonl": raw_path.name,
            "logits": logits_paths,
        },
        "reference_seconds_by_profile": reference_seconds,
        "weight_load_seconds_by_model": weight_load_seconds,
        "identity": {
            "compiled_manifest": {"path": str(plans_dir / "manifest.json"),
                                  "sha256": sha256_file(plans_dir / "manifest.json")},
            "rknn_manifest": ({"path": str(rknn_dir / "rknn-manifest.json"),
                               "sha256": sha256_file(rknn_dir / "rknn-manifest.json")}
                              if "vendor" in selected_arms else None),
            "current_programs": current_program_identity,
            "vendor_models": vendor_model_identity,
            "lib": ({"path": str(args.lib.resolve()),
                     "sha256": sha256_file(args.lib.resolve())}
                    if "current" in selected_arms else None),
            "vendor_core_mask": ("RKNNLite.NPU_CORE_0"
                                 if "vendor" in selected_arms else None),
            "weights": {path.name: sha256_file(path) for path in weight_files},
            "data": {path.name: sha256_file(path) for path in input_paths},
        },
        "cpu_thread_config": thread_config(),
    }
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans-dir", type=Path,
                        default=ROOT / "models" / "compiled")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--data-dir", type=Path, default=ROOT / ".data" / "mnist")
    parser.add_argument("--rknn-dir", type=Path, default=ROOT / "models" / "vendor")
    parser.add_argument("--lib", type=Path, default=Path(os.environ.get(
        "RKNPU_PLAN_LIB", "")) if os.environ.get("RKNPU_PLAN_LIB") else
        ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan.so")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", action="append", choices=[p[0] for p in PROFILES],
                        help="restrict to one or more profiles; repeatable")
    parser.add_argument("--mode", choices=("all", "current", "vendor", "cpu"),
                        default="all",
                        help="all first proves current/vendor coexistence; use a "
                             "single-arm mode in separate processes if it fails")
    args = parser.parse_args()
    try:
        report = run(args)
    except BaseException as exc:
        print(f"benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        for note in getattr(exc, "__notes__", ()):
            print(note, file=sys.stderr)
        if args.mode == "all":
            print("If arm initialization or the coexistence smoke failed, rerun "
                  "with --mode current, --mode vendor, and --mode cpu in separate "
                  "processes; no result JSON was written.", file=sys.stderr)
        return 1
    print(json.dumps({"out": str(args.out.resolve()),
                      "profiles": list(report["profiles"]),
                      "mode": report["mode"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
