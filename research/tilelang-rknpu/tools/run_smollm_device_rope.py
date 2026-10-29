#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run SmolLM2 prefill/decode with an opt-in device RoPE plan.

Prefill uses constant Cos/Sin plans for the fixed 256-token bucket and reuses
them across all layers.  Decode uses runtime Cos/Sin inputs because the current
runtime has no dynamic position-indexed device view.  Baseline and device modes
run on the same loaded runner/package for a paired logits comparison.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def cos_sin(tokens, heads, hd, theta, positions):
    inv = np.power(np.float32(theta),
                   -np.arange(0, hd, 2, dtype=np.float32) / np.float32(hd))
    freqs = positions.astype(np.float32)[:, None] * inv[None, :]
    emb = np.concatenate((freqs, freqs), axis=-1)
    cos = np.cos(emb).astype(np.float32)
    sin = np.sin(emb).astype(np.float32)
    return (
        np.ascontiguousarray(np.broadcast_to(
            cos[:, None, :], (tokens, heads, hd)).reshape(
                tokens, heads * hd).astype(np.float16)),
        np.ascontiguousarray(np.broadcast_to(
            sin[:, None, :], (tokens, heads, hd)).reshape(
                tokens, heads * hd).astype(np.float16)),
    )


def rope_perm(width, hd):
    p = np.zeros((width, width), dtype=np.float16)
    half = hd // 2
    for base in range(0, width, hd):
        for d in range(hd):
            pair = d + half if d < half else d - half
            p[base + d, base + pair] = -1.0 if d < half else 1.0
    return np.ascontiguousarray(p)


from rknpu_backend.smollm import SmolLM2Runner


def policy_binding_problems(policy, report):
    """Return policy binding problems for mode, steps and actual calls."""
    problems = []
    if policy.get("decode_mode") != report.get("decode_mode"):
        problems.append("policy_decode_mode")
    if int(policy.get("decode_steps", -1)) != int(report.get("decode_steps", -2)):
        problems.append("policy_decode_steps")
    if int(policy.get("expected_decode_calls", -1)) != \
            int(report.get("actual_decode_calls", -2)):
        problems.append("policy_actual_decode_calls")
    return problems


class DeviceRopeRunner(SmolLM2Runner):
    """SmolLM2 runner with an opt-in device RoPE implementation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device_rope = False
        self.decode_indexed = False
        self.rope_stats = {
            "prefill_calls": 0, "decode_calls": 0,
            "prefill_wall_us": 0.0, "decode_wall_us": 0.0,
        }
        self._rope_calls = {}
        self._rope_plans = {}
        self._build_rope_calls()

    def _build_rope_calls(self):
        from rknpu_backend import (CompileOptions, compile_rkir,
                                   default_profile,
                                   rope_device_constant_module,
                                   rope_device_module)
        from rknpu_backend.models import PlanCall
        cfg = self.config
        hd = cfg.head_dim
        widths = {cfg.num_attention_heads: cfg.hidden_size,
                  cfg.num_key_value_heads: cfg.num_key_value_heads * hd}
        prefill_pos = np.arange(self.prefill_bucket, dtype=np.int32)
        for heads, width in widths.items():
            cos, sin = cos_sin(self.prefill_bucket, heads, hd,
                               cfg.rope_theta, prefill_pos)
            pp = compile_rkir(
                rope_device_constant_module(
                    m=self.prefill_bucket, width=width,
                    name=f"smollm_prefill_rope_{heads}h"),
                profile=default_profile(),
                options=CompileOptions(numeric_mode="rk_f16_tiled",
                                       fuse_task_chain=True)).plan
            pc = {"Cos": cos, "Sin": sin, "P": rope_perm(width, hd)}
            for spec in pp.constants:
                if spec.generator == "identity":
                    n = int(spec.shape[0])
                    pc[spec.name] = np.eye(n, dtype=np.float16)
            self._rope_calls[("prefill", heads)] = PlanCall(
                self.lib, pp, pc, fd=self.fd, share_constants=True)
            self._rope_plans[("prefill", heads)] = pp
            from rknpu_backend import rope_decode_indexed_module
            ip = compile_rkir(
                rope_decode_indexed_module(
                    width=width, capacity=self.capacity,
                    name=f"smollm_decode_indexed_rope_{heads}h"),
                profile=default_profile(),
                options=CompileOptions(numeric_mode="rk_f16_tiled",
                                       fuse_task_chain=True)).plan
            ic = {"CosW": np.ascontiguousarray(cos.T),
                  "SinW": np.ascontiguousarray(sin.T),
                  "P": rope_perm(width, hd)}
            for spec in ip.constants:
                if spec.generator == "identity":
                    n = int(spec.shape[0])
                    ic[spec.name] = np.eye(n, dtype=np.float16)
            self._rope_calls[("decode_indexed", heads)] = PlanCall(
                self.lib, ip, ic, fd=self.fd, share_constants=True)
            self._rope_plans[("decode_indexed", heads)] = ip
            dp = compile_rkir(
                rope_device_module(m=1, width=width,
                                   name=f"smollm_decode_rope_{heads}h"),
                profile=default_profile(),
                options=CompileOptions(numeric_mode="rk_f16_tiled",
                                       fuse_task_chain=True)).plan
            dc = {"P": rope_perm(width, hd)}
            for spec in dp.constants:
                if spec.generator == "identity":
                    n = int(spec.shape[0])
                    dc[spec.name] = np.eye(n, dtype=np.float16)
            self._rope_calls[("decode", heads)] = PlanCall(
                self.lib, dp, dc, fd=self.fd, share_constants=True)
            self._rope_plans[("decode", heads)] = dp

    def _rope(self, x, positions):
        if not self.device_rope:
            return super()._rope(x, positions)
        x = np.asarray(x)
        tokens, heads, hd = x.shape
        width = heads * hd
        flat = np.ascontiguousarray(x.reshape(tokens, width))
        t0 = time.perf_counter()
        if tokens == self.prefill_bucket:
            call = self._rope_calls[("prefill", heads)]
            out = np.empty((tokens, width), dtype=np.float16)
            call.run_into([flat], [out])
            self.rope_stats["prefill_calls"] += 1
            self.rope_stats["prefill_wall_us"] += (time.perf_counter() - t0) * 1e6
        else:
            out = np.empty((tokens, width), dtype=np.float16)
            if self.decode_indexed:
                pos = int(np.asarray(positions).reshape(-1)[0])
                if pos < 0 or pos >= self.capacity:
                    raise ValueError(f"decode position {pos} outside capacity")
                onehot = np.zeros((1, self.capacity), dtype=np.float16)
                onehot[0, pos] = 1.0
                call = self._rope_calls[("decode_indexed", heads)]
                call.run_into([flat, onehot], [out])
            else:
                cos, sin = cos_sin(tokens, heads, hd, self.config.rope_theta,
                                   positions)
                call = self._rope_calls[("decode", heads)]
                call.run_into([flat, cos, sin], [out])
            self.rope_stats["decode_calls"] += 1
            self.rope_stats["decode_wall_us"] += (time.perf_counter() - t0) * 1e6
        return out.reshape(tokens, heads, hd)

    def close(self, *, ignore_errors: bool = False):
        for call in self._rope_calls.values():
            try:
                call.close()
            except Exception:
                if not ignore_errors:
                    raise
        self._rope_calls.clear()
        return super().close(ignore_errors=ignore_errors)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--prompt-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", default="")
    ap.add_argument("--decode-steps", type=int, default=32)
    ap.add_argument("--decode-indexed", action="store_true")
    ap.add_argument("--lib", default="")
    ap.add_argument("--backend", choices=("mock", "board"), default="board")
    args = ap.parse_args(argv)
    from rknpu_backend.runtime.ffi import LibRKPlan, configure_rocket_abi
    ids = np.asarray(json.loads(Path(args.prompt_json).read_text())["token_ids"],
                     dtype=np.int32)
    lib = LibRKPlan(args.lib) if args.lib else LibRKPlan()
    fd = 0
    owns_fd = False
    if args.backend == "board":
        configure_rocket_abi(lib.cdll)
        fd = int(lib.cdll.rocket_open())
        if fd < 0:
            raise SystemExit(f"rocket_open failed: {fd}")
        owns_fd = True
    runner = DeviceRopeRunner(lib, args.package, fd=fd,
                              share_constants=True, persistent_kv=True)
    runner.decode_indexed = bool(args.decode_indexed)
    try:
        # Baseline greedy run: prefill then the requested number of decodes.
        runner.device_rope = False
        t0 = time.perf_counter()
        ref_prefill = runner.prefill(ids, return_all_logits=False)
        ref_prefill_us = (time.perf_counter() - t0) * 1e6
        ref_pred = [ref_prefill["logits"].copy()]
        forced_tokens = []
        ref_decode_times = []
        for _ in range(args.decode_steps):
            tok = int(np.argmax(ref_pred[-1][0]))
            forced_tokens.append(tok)
            t0 = time.perf_counter()
            step = runner.decode_one(tok)
            ref_decode_times.append((time.perf_counter() - t0) * 1e6)
            ref_pred.append(step["logits"].copy())
        # Device run uses the exact same forced-token prefix.
        runner.reset()
        runner.device_rope = True
        t0 = time.perf_counter()
        dev_prefill = runner.prefill(ids, return_all_logits=False)
        dev_prefill_us = (time.perf_counter() - t0) * 1e6
        dev_pred = [dev_prefill["logits"].copy()]
        dev_decode_times = []
        for tok in forced_tokens:
            t0 = time.perf_counter()
            step = runner.decode_one(int(tok))
            dev_decode_times.append((time.perf_counter() - t0) * 1e6)
            dev_pred.append(step["logits"].copy())
        per_step_max = []
        per_step_top1 = []
        for ref_l, dev_l in zip(ref_pred, dev_pred):
            per_step_max.append(float(np.max(np.abs(
                ref_l.astype(np.float32) - dev_l.astype(np.float32)))))
            per_step_top1.append(bool(int(np.argmax(ref_l[0])) ==
                                      int(np.argmax(dev_l[0]))))
        d0 = np.abs(dev_pred[0].astype(np.float32) -
                    ref_pred[0].astype(np.float32))
        # Candidate self-consistency: ALL prefill+decode predictions vs a fresh
        # full-prefix run over prompt + the same forced tokens.
        full_ids = np.concatenate([ids, np.asarray(forced_tokens, dtype=np.int32)])
        runner.reset()
        full = runner.teacher_forcing_logits(full_ids)
        inc = np.asarray([p[0] for p in dev_pred], dtype=np.float32)
        start = int(ids.size) - 1
        full_tail = np.asarray(full["logits"][start:start + len(inc)],
                               dtype=np.float32)
        if full_tail.shape != inc.shape:
            raise SystemExit(
                f"incremental/full comparison shape mismatch: {inc.shape} vs {full_tail.shape}")
        full_diff = np.abs(inc - full_tail)
        # A/B/A isolation with decode on every request (not prefill-only).
        def run_sequence(seq_ids, nsteps):
            runner.reset()
            preds = [runner.prefill(seq_ids, return_all_logits=False)["logits"].copy()]
            toks = []
            for _ in range(nsteps):
                tok = int(np.argmax(preds[-1][0]))
                toks.append(tok)
                preds.append(runner.decode_one(tok)["logits"].copy())
            return np.stack([p[0] for p in preds], axis=0), toks

        aba_decode_steps = min(8, args.decode_steps)
        a1, a_tokens = run_sequence(ids, aba_decode_steps)
        b_ids = np.roll(ids, 1)
        b1, b_tokens = run_sequence(b_ids, aba_decode_steps)
        a2, a2_tokens = run_sequence(ids, aba_decode_steps)
        aba_bitwise = bool(np.array_equal(a1, a2))
        aba_shape_equal = a1.shape == a2.shape
        aba_tokens_equal = a_tokens == a2_tokens
        report = {
            "schema_version": 1,
            "kind": "smollm-device-rope-integration",
            "package": str(Path(args.package).resolve()),
            "prompt_tokens": int(ids.size),
            "decode_steps": args.decode_steps,
            "decode_mode": "position_indexed_device_gather" if args.decode_indexed else "runtime_coefficient_inputs",
            "baseline_prefill_us": ref_prefill_us,
            "device_prefill_us": dev_prefill_us,
            "baseline_decode_total_us": sum(ref_decode_times),
            "device_decode_total_us": sum(dev_decode_times),
            "baseline_decode_median_us": statistics.median(ref_decode_times),
            "device_decode_median_us": statistics.median(dev_decode_times),
            "rope_stats": runner.rope_stats,
            "prefill_logits_max_abs_diff": float(d0.max()),
            "prefill_logits_top1_equal": bool(
                int(np.argmax(dev_pred[0][0])) ==
                int(np.argmax(ref_pred[0][0]))),
            "decode_steps_max_abs_diff": per_step_max,
            "decode_steps_top1_equal": per_step_top1,
            "decode_logits_max_abs_diff": float(max(per_step_max[1:])),
            "decode_all_top1_equal": all(per_step_top1[1:]),
            "incremental_vs_full_prefix_max_abs": float(full_diff.max()),
            "incremental_vs_full_prefix_rmse": float(np.sqrt(np.mean(
                full_diff * full_diff))),
            "incremental_vs_full_prefix_top1_equal": bool(np.array_equal(
                np.argmax(inc, axis=1), np.argmax(full_tail, axis=1))),
            "incremental_vs_full_compare_count": int(len(inc)),
            "incremental_vs_full_expected_count": int(args.decode_steps + 1),
            "aba_request_isolation_bitwise": aba_bitwise,
            "aba_decode_steps": int(aba_decode_steps),
            "aba_sequence_shape": list(a1.shape),
            "aba_sequence_shape_equal": bool(aba_shape_equal),
            "aba_tokens_equal": bool(aba_tokens_equal),
            "actual_decode_calls": int(runner.rope_stats["decode_calls"]),
            "forced_tokens": forced_tokens,
            "finite": bool(np.isfinite(dev_pred[0]).all() and
                           all(np.isfinite(x).all() for x in dev_pred[1:]) and
                           np.isfinite(full_tail).all()),
            "plan_hashes": {f"{k[0]}_{k[1]}h": p.plan_hash
                            for k, p in runner._rope_plans.items()},
            "coefficient_residency": (
                "prefill constants resident and reused across layers; "
                + ("decode one-hot position gather from resident CosW/SinW tables"
                   if args.decode_indexed else
                   "decode per-token runtime Cos/Sin inputs")),
        }
        problems = []
        if not report["finite"]:
            problems.append("nonfinite")
        gates = {"prefill_max_abs": 0.25, "decode_max_abs": 0.25,
                 "top1_required": True}
        policy_sha = None
        if args.policy:
            policy_raw = Path(args.policy).read_bytes()
            policy = json.loads(policy_raw)
            policy_sha = __import__("hashlib").sha256(policy_raw).hexdigest()
            gates = policy["gates"]
            if policy.get("plan_hashes") != report["plan_hashes"]:
                problems.append("policy_plan_hashes")
            problems.extend(policy_binding_problems(policy, report))
            if policy.get("package_manifest_sha256") and \
                    policy["package_manifest_sha256"] != \
                    __import__("hashlib").sha256(
                        (Path(args.package) / "manifest.json").read_bytes()
                    ).hexdigest():
                problems.append("policy_package")
        report["policy_sha256"] = policy_sha
        report["gates"] = gates
        if report["prefill_logits_max_abs_diff"] > gates["prefill_max_abs"]:
            problems.append("prefill_logits")
        if report["decode_logits_max_abs_diff"] > gates["decode_max_abs"]:
            problems.append("decode_logits")
        if report.get("incremental_vs_full_prefix_max_abs", 0.0) > \
                gates.get("incremental_vs_full_max_abs", 0.25):
            problems.append("incremental_vs_full")
        if report["incremental_vs_full_compare_count"] != \
                report["incremental_vs_full_expected_count"]:
            problems.append("incremental_compare_count")
        if gates.get("aba_bitwise_required", True) and (
                not report["aba_request_isolation_bitwise"] or
                not report["aba_sequence_shape_equal"] or
                not report["aba_tokens_equal"]):
            problems.append("aba_isolation")
        if gates.get("top1_required", True):
            if not report["prefill_logits_top1_equal"]:
                problems.append("prefill_top1")
            if not report["decode_all_top1_equal"]:
                problems.append("decode_top1")
        report["problems"] = problems
        report["pass"] = not problems
        out = Path(args.out).resolve(); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        print(json.dumps(report, indent=1))
        return 0 if not problems else 1
    finally:
        runner.close(ignore_errors=True)
        if owns_fd:
            lib.cdll.rocket_close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
