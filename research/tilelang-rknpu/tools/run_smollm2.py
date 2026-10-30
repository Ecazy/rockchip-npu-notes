#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""RK3588 board runner for the frozen SmolLM2 package.

The Host prepares a request JSON with token ids (chat-template rendering is a
Host responsibility).  This process opens the NPU provider exactly once, loads
the compiled package, executes one or more requests, and writes raw arrays plus
run statistics.  It never opens RKNN/RKLLM or a CPU model implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np


def _read(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except Exception:
        return ""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def process_info() -> dict:
    info = {"ru_maxrss_kb": int(resource.getrusage(
        resource.RUSAGE_SELF).ru_maxrss)}
    try:
        status = Path("/proc/self/status").read_text()
        for line in status.splitlines():
            if line.startswith(("VmPeak:", "VmHWM:", "VmRSS:")):
                key, value = line.split(":", 1)
                info[key.lower()] = value.strip()
    except Exception:
        pass
    info["npu_load_after"] = _read("/sys/kernel/debug/rknpu/load")
    return info


def environment(package: Path, lib) -> dict:
    compat = _read("/proc/device-tree/compatible")
    lib_path = os.environ.get("RKNPU_PLAN_LIB", "")
    return {
        "collected_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "board_model": _read("/proc/device-tree/model").replace("\x00", ""),
        "chip": "rk3588" if "rk3588" in compat else "unknown",
        "machine": platform.machine(),
        "kernel": platform.release(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "driver_version": _read("/sys/kernel/debug/rknpu/version"),
        "npu_freq_hz": _read("/sys/kernel/debug/rknpu/freq"),
        "npu_load_before": _read("/sys/kernel/debug/rknpu/load"),
        "core_mask_env": os.environ.get("RKNPU_CORE_MASK", "unset"),
        "runtime_abi_version": int(lib.abi_version()),
        "plan_lib_basename": os.path.basename(lib_path),
        "plan_lib_sha256": sha256(Path(lib_path)) if lib_path and
                           Path(lib_path).is_file() else "",
        "package_manifest_sha256": sha256(package / "manifest.json"),
        "package_weights_sha256": sha256(package / "weights.npz"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--request", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-share-constants", action="store_true",
                    help="A/B control: repack/upload immutable weights per plan")
    ap.add_argument("--no-persistent-kv", action="store_true",
                    help="A/B control: re-pack/upload the full KV cache per step")
    ap.add_argument("--no-cache-host-weights", action="store_true",
                    help="A/B control: re-read embedding/norm weights from the "
                         "NPZ on every use instead of the immutable cached views")
    ap.add_argument("--profile", action="store_true",
                    help="collect layered decode timing; keep this off for "
                         "final performance measurements")
    ap.add_argument("--schedule-policy", default="",
                    help="external plan-override policy for an opt-in schedule")
    args = ap.parse_args(argv)

    package = Path(args.package).resolve()
    request_path = Path(args.request).resolve()
    request = json.loads(request_path.read_text())
    request_sha256 = sha256(request_path)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    from rknpu_backend.models import RunStats
    from rknpu_backend.runtime.ffi import LibRKPlan, configure_rocket_abi
    from rknpu_backend.smollm import SmolLM2Runner

    lib = LibRKPlan(os.environ.get("RKNPU_PLAN_LIB") or None)
    configure_rocket_abi(lib.cdll)
    fd = int(lib.cdll.rocket_open())
    if fd < 0:
        print(f"rocket_open failed: {fd}", file=sys.stderr)
        return 3
    overrides = None
    policy_identity = None
    if args.schedule_policy:
        policy_path = Path(args.schedule_policy).resolve()
        policy = json.loads(policy_path.read_text())
        if policy.get("kind") != "smollm2-plan-overrides-v1":
            raise SystemExit("unsupported schedule policy kind")
        overrides = policy.get("overrides")
        if not isinstance(overrides, dict):
            raise SystemExit("schedule policy overrides must be an object")
        policy_identity = {
            "path": str(policy_path),
            "sha256": sha256(policy_path),
            "kind": policy["kind"],
            "source": policy.get("source", ""),
            "override_keys": sorted(overrides),
        }
    overall = {"kind": "smollm2-board-run", "fd_owned_by_runner": True,
               "request_sha256": request_sha256,
               "schedule_policy": policy_identity,
               "environment": environment(package, lib), "requests": []}
    status = 0
    runner = None
    try:
        stats = RunStats()
        layer_indices = request.get("layer_indices")
        load_t0 = time.perf_counter()
        runner = SmolLM2Runner(
            lib, package, fd=fd, stats=stats,
            layer_indices=(tuple(int(v) for v in layer_indices)
                           if layer_indices is not None else None),
            share_constants=not bool(args.no_share_constants),
            persistent_kv=not bool(args.no_persistent_kv),
            profile=bool(args.profile),
            cache_host_weights=not bool(args.no_cache_host_weights),
            plan_overrides=overrides)
        load_elapsed = (time.perf_counter() - load_t0) * 1e6
        overall["load_us"] = load_elapsed
        overall["loaded_plan_count"] = len(runner._calls)
        overall["backend_options"] = {
            "share_constants": bool(runner.share_constants),
            "persistent_kv": bool(runner.persistent_kv),
            "profile": bool(runner.profile),
            "cache_host_weights": bool(runner.cache_host_weights),
            "plan_override_keys": sorted(runner.plan_overrides),
        }
        for index, item in enumerate(request["runs"]):
            mode = item.get("mode", "generate")
            run_dir = out / f"run-{index:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            ids = item["token_ids"]
            repeat = int(item.get("repeat", 1))
            results = []
            t0 = time.perf_counter()
            for rep in range(repeat):
                if mode == "generate":
                    res = runner.generate(
                        ids, max_new_tokens=int(item["max_new_tokens"]),
                        return_all_logits=bool(item.get("return_all_logits",
                                                         False)),
                        stop_on_eos=bool(item.get("stop_on_eos", True)))
                    payload = {
                        "prompt_ids": res.prompt_ids,
                        "generated_ids": res.generated_ids,
                        "all_ids": res.all_ids,
                        "stopped_by_eos": res.stopped_by_eos,
                        "stop_reason": res.stop_reason,
                        "output_tokens": res.output_tokens,
                        "decode_forwards": res.decode_forwards,
                        "prefill_us": res.prefill_us,
                        "decode_us": res.decode_us,
                        "reset_us": res.reset_us,
                        "sampling_us": res.sampling_us,
                        "first_sample_us": res.first_sample_us,
                        "ttft_us": res.ttft_us,
                        "generate_wall_us": res.generate_wall_us,
                        "stages": res.stages,
                        "steps": res.steps,
                    }
                    if args.profile:
                        payload["profile"] = res.profile
                elif mode == "teacher_forcing":
                    res = runner.teacher_forcing_logits(ids)
                    np.save(run_dir / f"logits-{rep:02d}.npy", res["logits"])
                    payload = {"valid_len": res["valid_len"],
                               "elapsed_us": res["elapsed_us"],
                               "logits_shape": list(res["logits"].shape),
                               "logits_sha256": hashlib.sha256(
                                   res["logits"].tobytes()).hexdigest()}
                elif mode == "prefill":
                    runner.reset()
                    res = runner.prefill(
                        ids, return_all_logits=bool(
                            item.get("return_all_logits", True)))
                    np.save(run_dir / f"logits-{rep:02d}.npy", res["logits"])
                    if res["all_logits"] is not None:
                        np.save(run_dir / f"all-logits-{rep:02d}.npy",
                                res["all_logits"])
                    payload = {"valid_len": res["valid_len"],
                               "elapsed_us": res["elapsed_us"],
                               "logits_shape": list(res["logits"].shape)}
                elif mode == "capture":
                    res = runner.capture_prefill(
                        ids, layers=tuple(item.get("capture_layers",
                                                    [0, 15, 29])),
                        return_all_logits=bool(
                            item.get("return_all_logits", True)))
                    np.savez(run_dir / f"captured-{rep:02d}.npz",
                             **res["captured"])
                    np.save(run_dir / f"logits-{rep:02d}.npy", res["logits"])
                    if res["all_logits"] is not None:
                        np.save(run_dir / f"all-logits-{rep:02d}.npy",
                                res["all_logits"])
                    payload = {"valid_len": res["valid_len"],
                               "elapsed_us": res["elapsed_us"],
                               "captured": sorted(res["captured"]),
                               "logits_shape": list(res["logits"].shape)}
                elif mode == "score":
                    res = runner.teacher_forcing_logits(ids)
                    logits = res["logits"]
                    ids_arr = np.asarray(ids, dtype=np.int64)
                    # Logits at position i score token i+1.  The final row has
                    # no target and is intentionally excluded.
                    pred = logits[:-1].astype(np.float32)
                    labels = ids_arr[1:]
                    if pred.shape[0] != labels.size:
                        raise RuntimeError("score logits/label length mismatch")
                    mx = np.max(pred, axis=1, keepdims=True)
                    with np.errstate(over="ignore"):
                        lse = mx[:, 0] + np.log(
                            np.sum(np.exp(pred - mx), axis=1))
                    logp = pred[np.arange(labels.size), labels] - lse
                    nll = -logp.astype(np.float64)
                    top1 = np.argmax(pred, axis=1).astype(np.int64)
                    np.save(run_dir / f"nll-{rep:02d}.npy", nll)
                    payload = {
                        "valid_len": res["valid_len"],
                        "token_ids": [int(v) for v in ids],
                        "token_seq_sha256": hashlib.sha256(
                            ids_arr.astype("<i8").tobytes()).hexdigest(),
                        "scored_label_ids": [int(v) for v in labels],
                        "scored_label_sha256": hashlib.sha256(
                            labels.astype("<i8").tobytes()).hexdigest(),
                        "scored_tokens": int(labels.size),
                        "sum_nll": float(nll.sum()),
                        "mean_nll": float(nll.mean()),
                        "ppl": float(np.exp(nll.mean())),
                        "top1_accuracy": float(np.mean(top1 == labels)),
                        "nll_sha256": hashlib.sha256(nll.tobytes()).hexdigest(),
                        "logits_shape": list(logits.shape),
                        "logits_sha256": hashlib.sha256(
                            logits.tobytes()).hexdigest(),
                    }
                elif mode == "expect_reject":
                    from rknpu_backend.errors import BackendError, code_name
                    expected = str(item.get("expected_code", "RK_E_SHAPE"))
                    try:
                        runner.reset()
                        runner.generate(
                            ids,
                            max_new_tokens=int(item.get("max_new_tokens", 1)),
                            stop_on_eos=bool(item.get("stop_on_eos", False)))
                    except BackendError as exc:
                        actual = code_name(exc.code)
                        if actual != expected:
                            raise
                        payload = {"rejected": True, "expected_code": expected,
                                   "actual_code": actual,
                                   "message": str(exc)}
                    else:
                        payload = {"rejected": False,
                                   "expected_code": expected,
                                   "actual_code": None,
                                   "message": "request was accepted"}
                elif mode == "incremental":
                    continuation = [int(v) for v in item.get(
                        "continuation_ids", [])]
                    reset_before = runner._stats_snapshot()
                    reset_t0 = time.perf_counter()
                    runner.reset()
                    reset_us = (time.perf_counter() - reset_t0) * 1e6
                    reset_stats = runner._stats_delta(
                        reset_before, runner._stats_snapshot())
                    prefill_before = runner._stats_snapshot()
                    first = runner.prefill(ids, return_all_logits=False)
                    prefill_stats = runner._stats_delta(
                        prefill_before, runner._stats_snapshot())
                    logits_rows = [first["logits"]]
                    step_logs = [{"kind": "prefill_after_prompt",
                                  "valid_len": runner.valid_len,
                                  "elapsed_us": first["elapsed_us"],
                                  "stats_delta": prefill_stats}]
                    decode_rows = []
                    for pos, token in enumerate(continuation):
                        before = runner._stats_snapshot()
                        decode_t0 = time.perf_counter()
                        step = runner.decode_one(token)
                        step_us = (time.perf_counter() - decode_t0) * 1e6
                        delta = runner._stats_delta(
                            before, runner._stats_snapshot())
                        decode_rows.append(delta)
                        logits_rows.append(step["logits"])
                        step_logs.append({"kind": "decode_after_token",
                                          "token_id": token,
                                          "valid_len": step["valid_len"],
                                          "elapsed_us": step_us,
                                          "stats_delta": delta})
                    # Shape [prompt+continuation, vocab]; each row is the
                    # prediction that selects the next supplied token.
                    np.save(run_dir / f"incremental-logits-{rep:02d}.npy",
                            np.concatenate(logits_rows, axis=0))
                    payload = {"prompt_len": len(ids),
                               "prompt_ids": [int(v) for v in ids],
                               "continuation_ids": continuation,
                               "valid_len": runner.valid_len,
                               "reset_us": reset_us,
                               "reset_stats_delta": reset_stats,
                               "prefill_stats_delta": prefill_stats,
                               "decode_stats_delta": runner._stats_sum(
                                   decode_rows),
                               "decode_forwards": len(decode_rows),
                               "steps": step_logs,
                               "logits_shape": [len(logits_rows),
                                                runner.config.vocab_size]}
                else:
                    raise SystemExit(f"unknown request mode {mode!r}")
                results.append(payload)
            elapsed = (time.perf_counter() - t0) * 1e6
            row = {"index": index, "mode": mode, "repeat": repeat,
                   "token_ids": [int(v) for v in ids],
                   "token_count": len(ids), "wall_us": elapsed,
                   "runs": results,
                   "valid_len_after": runner.valid_len}
            (run_dir / "result.json").write_text(
                json.dumps(row, indent=1, sort_keys=True) + "\n")
            overall["requests"].append(row)
            print(json.dumps({k: row[k] for k in
                              ("index", "mode", "repeat", "token_count",
                               "wall_us", "valid_len_after")},
                             sort_keys=True), flush=True)
        overall["stats"] = stats.to_dict()
        overall["process"] = process_info()
        (out / "stats.json").write_text(
            json.dumps(overall["stats"], indent=1, sort_keys=True) + "\n")
    except BaseException as exc:
        status = 1
        overall["error"] = {"type": type(exc).__name__, "text": str(exc)}
        (out / "error.json").write_text(
            json.dumps(overall["error"], indent=1, sort_keys=True) + "\n")
        raise
    finally:
        if runner is not None:
            try:
                runner.close()
            except BaseException as exc:
                overall["close_error"] = repr(exc)
                status = 1
        # Only close the fd after workspace destruction has quiesced.  A
        # poisoned workspace raises during close and keeps the fd/context.
        try:
            lib.cdll.rocket_close(fd)
        except Exception as exc:
            overall["fd_close_error"] = repr(exc)
        (out / "run.json").write_text(
            json.dumps(overall, indent=1, sort_keys=True) + "\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
