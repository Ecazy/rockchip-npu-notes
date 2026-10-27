#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Integrate the direct-C QK/PV group path into the full Tiny ViT b8 model.

This is a targeted model-level transmission probe, not a new default backend.
All other plans and Host actions remain the current compiled graph.  The two
QK/PV group PlanCalls are replaced after construction by the experimental C
direct-group context; the runner still owns the resident arrays and Host
actions.
"""
from __future__ import annotations

import argparse
import ctypes
import dataclasses
import gzip
import hashlib
import json
import statistics
import struct
import time
from pathlib import Path

import numpy as np

from rknpu_backend.hybrid import (HybridRunner, _CompiledGraph,
                                  _GraphNpuOp, load_hybrid_program)
from rknpu_backend.models import load_weight_npz
from rknpu_backend.runtime.ffi import LibRKPlan, configure_rocket_abi


def read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        f.read(4)
        dims = struct.unpack(">III", f.read(12))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)


class DirectGroupCall:
    """Duck-typed group PlanCall using the experimental direct-C context."""

    def __init__(self, case_dir: Path, lib_path: Path, fd: int):
        self.lib = ctypes.CDLL(str(lib_path))
        self.lib.rl_group_open.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.rl_group_open.restype = ctypes.c_void_p
        self.lib.rl_group_run.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_double),
        ]
        self.lib.rl_group_run.restype = ctypes.c_int
        self.lib.rl_group_close.argtypes = [ctypes.c_void_p]
        self.ctx = self.lib.rl_group_open(str(case_dir).encode(), int(fd))
        if not self.ctx:
            raise RuntimeError(f"rl_group_open failed: {case_dir}")
        self.flat_inputs: list[np.ndarray] = []
        self.flat_outputs: list[np.ndarray] = []
        self.binding_key = None
        self.last_us = 0.0

    def bind_flat(self, flat_inputs, flat_outputs, *,
                  binding_key: str = "direct") -> None:
        self.flat_inputs = list(flat_inputs)
        self.flat_outputs = list(flat_outputs)
        self.binding_key = binding_key
        for arr in self.flat_inputs + self.flat_outputs:
            if not arr.flags.c_contiguous:
                raise RuntimeError("direct group binding is not C-contiguous")

    def bind_group_prepared_into(self, inputs_list, outputs_list, *,
                                 binding_key: str = "direct") -> None:
        self.bind_flat([a for row in inputs_list for a in row],
                       [a for row in outputs_list for a in row],
                       binding_key=binding_key)

    def bind_prepared_into(self, inputs, outputs, *,
                           binding_key: str = "direct") -> None:
        self.bind_flat(inputs, outputs, binding_key=binding_key)

    def _run(self, binding_key: str):
        if binding_key != self.binding_key:
            raise RuntimeError("direct group binding key mismatch")
        in_ptrs = (ctypes.c_void_p * len(self.flat_inputs))(
            *[int(a.ctypes.data) for a in self.flat_inputs])
        out_ptrs = (ctypes.c_void_p * len(self.flat_outputs))(
            *[int(a.ctypes.data) for a in self.flat_outputs])
        us = ctypes.c_double()
        rc = self.lib.rl_group_run(
            self.ctx, in_ptrs, out_ptrs,
            len(self.flat_inputs), len(self.flat_outputs), ctypes.byref(us))
        if rc:
            raise RuntimeError(f"direct group run failed: {rc}")
        self.last_us = float(us.value)
        return None, self.last_us

    def run_prepared(self, *, binding_key: str = "direct"):
        return self._run(binding_key)

    def run_group_prepared(self, *, binding_key: str = "direct"):
        return self._run(binding_key)

    def close(self):
        if self.ctx:
            self.lib.rl_group_close(self.ctx)
            self.ctx = None


class TimedPlanProxy:
    def __init__(self, call):
        self.call = call
        self.last_us = 0.0

    def run_prepared(self, *, binding_key: str = "default"):
        t0 = time.perf_counter()
        result = self.call.run_prepared(binding_key=binding_key)
        self.last_us = (time.perf_counter() - t0) * 1e6
        return result


def replace_group_calls(runner: HybridRunner, adapters: dict[str, DirectGroupCall]):
    graph = runner._compiled_production
    if graph is None:
        raise RuntimeError("runner is not using the compiled graph executor")
    ops = []
    replaced = 0
    for op in graph.ops:
        if isinstance(op, _GraphNpuOp) and op.node_id in adapters:
            ops.append(dataclasses.replace(op, call=adapters[op.node_id]))
            replaced += 1
        else:
            ops.append(op)
    runner._compiled_production = _CompiledGraph(
        ops=tuple(ops), produced_tensors=graph.produced_tensors,
        view_ops=graph.view_ops,
        skipped_trace_output_writes=graph.skipped_trace_output_writes,
        capture=graph.capture)
    if replaced != len(adapters):
        raise RuntimeError(f"replaced {replaced} graph ops, expected {len(adapters)}")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans-dir", type=Path, required=True)
    ap.add_argument("--weights-dir", type=Path, required=True)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--adapter-lib", required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--qk-case", type=Path, required=True)
    ap.add_argument("--pv-case", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.limit % args.batch:
        raise SystemExit("limit must be divisible by batch")
    images = read_idx(args.data_dir / "t10k-images-idx3-ubyte.gz")[:args.limit]
    program = load_hybrid_program(
        args.plans_dir / f"tiny_vit_b{args.batch}.rkhybrid.json",
        plans_dir=args.plans_dir)
    weights_path = args.weights_dir / "tiny_vit.npz"
    weights = load_weight_npz(weights_path)
    lib = LibRKPlan(args.lib)
    configure_rocket_abi(lib.cdll)
    fd = int(lib.cdll.rocket_open())
    if fd < 0:
        raise RuntimeError(f"rocket_open failed: {fd}")
    current = variant = None
    adapters = {}
    try:
        t0 = time.perf_counter()
        current = HybridRunner(lib, program, weights, weights_path=weights_path,
                               fd=fd, execution_mode="graph")
        current_setup_us = (time.perf_counter() - t0) * 1e6
        t0 = time.perf_counter()
        variant = HybridRunner(lib, program, weights, weights_path=weights_path,
                               fd=fd, execution_mode="graph")
        variant_setup_us = (time.perf_counter() - t0) * 1e6
        plan_keys = {id(info.plan): key for key, info in variant.program.plans.items()}
        t0 = time.perf_counter()
        for node_id, item in variant.prepared_nodes.items():
            kind, call, repeats, _, _ = item
            plan_key = plan_keys.get(id(call.plan))
            if plan_key not in ("attn_qk_qkv_group", "attn_pv_qkv_group"):
                continue
            case_dir = (args.qk_case if plan_key == "attn_qk_qkv_group"
                        else args.pv_case)
            prepared = call._prepared_single.get(node_id)
            if prepared is None:
                raise RuntimeError(
                    f"direct adapter expects a single prepared binding: {node_id}")
            in_arrays, out_arrays, _, _ = prepared
            adapter = DirectGroupCall(case_dir, args.adapter_lib, fd)
            adapter.bind_flat(in_arrays, out_arrays, binding_key=node_id)
            adapters[node_id] = adapter
        adapter_setup_us = (time.perf_counter() - t0) * 1e6
        if set(adapters) != {"qk", "pv"}:
            raise RuntimeError(f"unexpected adapter node ids: {sorted(adapters)}")
        # Keep a small proxy around the current qk/pv calls so the model test
        # can report the actual per-node group interval, not infer it from the
        # application delta.
        current_proxies = {}
        current_ops = []
        for op in current._compiled_production.ops:
            if isinstance(op, _GraphNpuOp):
                proxy = TimedPlanProxy(op.call)
                current_proxies[op.node_id] = proxy
                current_ops.append(dataclasses.replace(op, call=proxy))
            else:
                current_ops.append(op)
        current._compiled_production = _CompiledGraph(
            ops=tuple(current_ops),
            produced_tensors=current._compiled_production.produced_tensors,
            view_ops=current._compiled_production.view_ops,
            skipped_trace_output_writes=current._compiled_production.skipped_trace_output_writes,
            capture=current._compiled_production.capture)
        replace_group_calls(variant, adapters)
        variant_proxies = {}
        variant_ops = []
        for op in variant._compiled_production.ops:
            if isinstance(op, _GraphNpuOp) and op.node_id not in adapters:
                proxy = TimedPlanProxy(op.call)
                variant_proxies[op.node_id] = proxy
                variant_ops.append(dataclasses.replace(op, call=proxy))
            else:
                variant_ops.append(op)
        variant._compiled_production = _CompiledGraph(
            ops=tuple(variant_ops),
            produced_tensors=variant._compiled_production.produced_tensors,
            view_ops=variant._compiled_production.view_ops,
            skipped_trace_output_writes=variant._compiled_production.skipped_trace_output_writes,
            capture=variant._compiled_production.capture)
        all_nodes = sorted(set(current_proxies) | set(variant_proxies) | set(adapters))

        # Warm both applications with the first batch before any timed calls.
        current.run(images[:args.batch], stats=None)
        variant.run(images[:args.batch], stats=None)
        calls = args.limit // args.batch
        samples = {"current": [], "variant": []}
        node_samples = {node: {"current": [], "variant": []}
                        for node in all_nodes}
        paired = []
        bitwise = True
        output_hashes = {}
        all_logits = {
            "current": np.empty((args.limit, 10), dtype=np.float32),
            "variant": np.empty((args.limit, 10), dtype=np.float32),
        }
        order = ("current", "variant")
        for round_index in range(args.rounds):
            for call_index in range(calls):
                start = call_index * args.batch
                stop = start + args.batch
                rotate = (call_index + round_index) % 2
                this_order = order[rotate:] + order[:rotate]
                outputs = {}
                timing = {}
                for name in this_order:
                    runner = current if name == "current" else variant
                    t0 = time.perf_counter()
                    result = runner.run(images[start:stop], stats=None)
                    timing[name] = (time.perf_counter() - t0) * 1e6
                    outputs[name] = np.array(result.outputs["logits"],
                                             copy=True)
                if not np.array_equal(outputs["current"], outputs["variant"]):
                    bitwise = False
                    raise RuntimeError("full-model outputs differ bitwise")
                for name in ("current", "variant"):
                    all_logits[name][start:stop] = outputs[name]
                for name in ("current", "variant"):
                    samples[name].append(timing[name])
                for node in all_nodes:
                    node_samples[node]["current"].append(
                        current_proxies[node].last_us)
                    variant_calls = (adapters.get(node) or
                                     variant_proxies.get(node))
                    node_samples[node]["variant"].append(
                        variant_calls.last_us)
                paired.append(timing["current"] / timing["variant"])
                if call_index == calls - 1 and round_index == args.rounds - 1:
                    for name, value in outputs.items():
                        output_hashes[name] = sha_bytes(value.tobytes())
        def stats(values):
            if not values:
                return {}
            return {"median_us": float(statistics.median(values)),
                    "min_us": float(min(values)),
                    "p10_us": float(sorted(values)[(len(values) - 1) // 10]),
                    "max_us": float(max(values)),
                    "n": len(values)}
        payload = {
            "schema_version": 1,
            "kind": "full-model-direct-group-integration",
            "limit": args.limit, "batch": args.batch, "rounds": args.rounds,
            "setup_us": {"current": current_setup_us,
                         "variant": variant_setup_us,
                         "adapter": adapter_setup_us},
            "timing": {name: stats(values) for name, values in samples.items()},
            "paired_current_over_variant": stats(paired),
            "bitwise": bitwise and output_hashes.get("current") ==
                       output_hashes.get("variant"),
            "output_sha256": output_hashes,
            "adapter_nodes": sorted(adapters),
            "group_node_timing": {
                node: {arm: stats(values) for arm, values in by_arm.items()}
                for node, by_arm in node_samples.items()
            },
            "note": "Only qk/pv group calls are replaced; Host actions and all "
                    "other plan calls remain the current compiled graph.",
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        for name, value in all_logits.items():
            np.save(args.out.parent / f"{name}_logits.npy", value)
        payload["output_files"] = {
            name: f"{name}_logits.npy" for name in all_logits
        }
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        for adapter in adapters.values():
            adapter.close()
        if current is not None:
            current.close()
        if variant is not None:
            variant.close()
        lib.cdll.rocket_close(fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
