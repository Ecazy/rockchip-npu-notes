#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare legal execution-plan candidates with one measured profile ledger.

This is intentionally a small pre-implementation model, not a general
autotuner.  It consumes the representative-run JSON emitted by
``profile_bridge_envelope.py`` and a frozen rkhybrid graph.  The ledger keeps
preparation, Host action, transfer/pack, submit, guard and executor terms
separate, then applies only candidate transformations that can be described from
the graph and measured envelopes.

Candidate groups:

``prepared_host_residency``
    Stable Host slots and prebound RKPlan descriptors.  The C validation and
    transfer work is unchanged; the model removes the duplicated Python binding
    surface and adds write-back into resident Host slots.

``host_boundary_fusion``
    Adjacent registered Host actions with an existing bit-preserving fused
    action.  It can remove a Python/Host boundary, but no RKPlan submit or
    transfer is removed.

``device_resident_qkv``
    Keep the QKV output in a device BO for QK/PV consumers.  The model removes
    the measured QKV download and the QK/PV uploads attributable to QKV.  Device
    reorder/recipe cost and required contract work are deliberately left
    unpriced and marked missing; this is a screening bound, not an implementation
    claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rknpu_backend.hybrid import HostNode, NpuNode, load_hybrid_program
from rknpu_backend.passes.residency import MIN_PREPARED_BINDING_SLOTS
from rknpu_backend.plan import Plan


def _ms(us: float) -> float:
    return float(us) / 1e3


def _profile_run(profile: dict[str, Any]) -> dict[str, Any]:
    wall = profile["wall_us"]
    run_us = float(wall["median"] if isinstance(wall, dict) else wall)
    env = dict(profile["envelope"])
    plans = {row["name"]: row for row in profile.get("plans", [])}
    actions = {row["name"]: row for row in profile.get("host_actions", [])}
    return {"run_us": run_us, "envelope": env, "plans": plans,
            "host_actions": actions}


def _host_output_bytes(program) -> int:
    total = 0
    for node in program.nodes:
        if isinstance(node, HostNode):
            total += sum(program.tensors[ref.tensor].nbytes
                         for ref in node.outputs.values())
    return total


def _npu_binding_count(program, runner_plans: dict[str, Any]) -> int:
    """Count static ctypes binding slots touched in one graph invocation."""
    total = 0
    for node in program.nodes:
        if not isinstance(node, NpuNode):
            continue
        binding_plan = runner_plans[node.id]
        if binding_plan.mode == "shared_group":
            info = program.plans[node.group_plan]
            total += len(info.plan.bindings)
        else:
            info = program.plans[node.plan]
            total += len(info.plan.bindings) * max(
                1, len(binding_plan.repeats))
    return total


def _upload_bytes_from_tensor(plan: Plan, program, node,
                              tensor: str) -> int:
    """Upload payload bytes in ``plan`` sourced from one input tensor.

    For a mapped group the A_i/B_i source binding is matched to the graph's
    group_layout role.  A normal plan is matched through the node input names.
    The byte quantity is the physical upload destination size, the same unit the
    C runtime reports as ``upload_bytes``.
    """
    role_to_tensor: dict[str, str] = {}
    if node.group_layout is not None:
        role_to_tensor = {
            name: spec.tensor for name, spec in node.group_layout.inputs.items()
        }
    else:
        role_to_tensor = {
            name: ref.tensor for name, ref in node.inputs.items()
        }
    bindings = {binding.id: binding for binding in plan.bindings}
    views = {view.id: view for view in plan.views}
    total = 0
    for action in plan.actions:
        if action.kind != "upload":
            continue
        try:
            source_id = int(action.attrs["source_binding"])
        except (KeyError, TypeError, ValueError):
            continue
        binding = bindings.get(source_id)
        if binding is None:
            continue
        base = binding.name.rsplit("_", 1)[0] if node.group_layout else binding.name
        if role_to_tensor.get(base) != tensor:
            continue
        if action.outputs:
            view = views.get(action.outputs[0])
            if view is not None:
                total += int(view.tensor.bytes)
    return total


def _device_qkv_candidate(program, run: dict[str, Any]) -> dict[str, Any]:
    """Screen direct device-consumer handoffs using measured transfer rows."""
    rows = run["plans"]
    candidates = []
    for producer in program.nodes:
        if not isinstance(producer, NpuNode):
            continue
        if producer.repeat or producer.group_layout is not None:
            continue
        produced = {ref.tensor for ref in producer.outputs.values()}
        consumers = []
        for node in program.nodes:
            if not isinstance(node, NpuNode) or node.id == producer.id:
                continue
            touched = produced & {ref.tensor for ref in node.inputs.values()}
            if touched:
                consumers.append((node, touched))
        if not consumers:
            continue
        producer_row = rows.get(producer.id)
        if producer_row is None:
            continue
        saved_download_bytes = int(producer_row.get("download_bytes", 0))
        saved_download_us = float(producer_row.get("download_us", 0.0))
        saved_upload_bytes = 0
        saved_pack_us = 0.0
        consumer_ids = []
        for consumer, touched in consumers:
            consumer_row = rows.get(consumer.id)
            info = program.plans[
                consumer.group_plan if consumer.group_plan else consumer.plan]
            plan_uploads = int(consumer_row.get("upload_bytes", 0)) if consumer_row else 0
            wanted = sum(_upload_bytes_from_tensor(info.plan, program, consumer,
                                                   tensor)
                         for tensor in touched)
            wanted = min(wanted, plan_uploads)
            saved_upload_bytes += wanted
            if consumer_row is not None and plan_uploads:
                saved_pack_us += (float(consumer_row.get("pack_us", 0.0)) *
                                  wanted / plan_uploads)
            consumer_ids.append(consumer.id)
        if not saved_download_bytes and not saved_upload_bytes:
            continue
        candidates.append({
            "producer": producer.id,
            "consumers": consumer_ids,
            "removed_producer_download_bytes": saved_download_bytes,
            "removed_consumer_upload_bytes": saved_upload_bytes,
            "modeled_download_us": saved_download_us,
            "modeled_upload_pack_us": saved_pack_us,
            "modeled_saving_us": saved_download_us + saved_pack_us,
            "submit_count_change": 0,
            "status": "screened_only",
            "missing": [
                "device-source binding with versioned BO lifetime/owner checks",
                "C layout proof or device CUBE->WEIGHT/tile reorder template",
                "numeric contract for the resident boundary if conversion changes "
                "rounding",
            ],
            "notes": (
                "The bound excludes any device reorder cost.  Host Softmax after "
                "QK means scores/probabilities still cross the Host boundary, so "
                "this candidate does not collapse the whole attention segment."
            ),
        })
    return {"candidates": candidates}


def _host_fusion_candidate(program, run: dict[str, Any]) -> dict[str, Any]:
    """Find the registered add+rmsnorm composition and price its Host boundary."""
    actions = run["host_actions"]
    matches = []
    for node in program.nodes:
        if not isinstance(node, HostNode) or node.action.action_id != "tensor.add.v1":
            continue
        for consumer in program.nodes:
            if not isinstance(consumer, HostNode) or node.id not in consumer.depends_on:
                continue
            if consumer.action.action_id != "rmsnorm.last_axis.v1":
                continue
            a = float(actions.get(node.id, {}).get("elapsed_us", 0.0))
            b = float(actions.get(consumer.id, {}).get("elapsed_us", 0.0))
            # The registered fused implementation performs the same two passes
            # over the same bytes.  Use a small boundary factor rather than
            # claiming the arithmetic disappears.
            fused = 0.98 * (a + b)
            matches.append({
                "replace": [node.id, consumer.id],
                "with": "tensor.add_rmsnorm.last_axis.v1",
                "modeled_removed_us": a + b,
                "modeled_added_us": fused,
                "modeled_saving_us": a + b - fused,
                "submit_count_change": 0,
                "transfer_change": 0,
                "status": "screened_only",
            })
            break
    return {
        "candidates": matches,
        "note": "Host fusion can remove a Host boundary but no RKPlan submit or "
                "bridge transfer; prepared residency already reduces the Python "
                "plan-node surface.",
    }


def _prepared_candidate(program, run: dict[str, Any],
                        prepared_profile: Path | None) -> dict[str, Any]:
    env = run["envelope"]
    py_us = float(env.get("python_plan_node_us", 0.0))
    host_output_bytes = _host_output_bytes(program)
    # Resident write-back is an explicit copy into a stable slot.  Use a
    # conservative 8 GB/s Host memcpy rate and expose it rather than hiding it.
    copy_us = host_output_bytes / 8000.0
    residual_py_us = 0.05 * py_us
    predicted_us = (run["run_us"] - py_us + residual_py_us + copy_us)
    out = {
        "modeled_total_ms": _ms(predicted_us),
        "modeled_saving_ms": _ms(run["run_us"] - predicted_us),
        "components": {
            "generic_python_plan_node_us": py_us,
            "modeled_residual_python_us": residual_py_us,
            "resident_host_output_bytes": host_output_bytes,
            "modeled_writeback_us": copy_us,
            "submit_change": 0,
            "transfer_change": 0,
        },
        "conditions": [
            "one producer per resident tensor and no Host-slot aliasing",
            "all RKPlan views resolve against stable buffers for runner lifetime",
            "C per-run binding, finite, alias and publication checks remain",
            "public outputs copied so caller retention does not alias residency",
        ],
        "missing": [],
        "status": "candidate",
    }
    if prepared_profile is not None:
        measured = _profile_run(json.loads(prepared_profile.read_text()))
        out["measured"] = {
            "total_ms": _ms(measured["run_us"]),
            "saving_ms": _ms(run["run_us"] - measured["run_us"]),
            "python_plan_node_ms": _ms(float(
                measured["envelope"].get("python_plan_node_us", 0.0))),
            "host_action_ms": _ms(measured["envelope"]["host_action_us"]),
            "workspace_ms": _ms(measured["envelope"]["plan_workspace_us"]),
        }
        out["prediction_error_ms"] = _ms(
            predicted_us - measured["run_us"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", type=Path, required=True,
                    help="generic profile_bridge_envelope JSON")
    ap.add_argument("--plans-dir", type=Path, required=True)
    ap.add_argument("--manifest", required=True,
                    help="rkhybrid manifest stem inside --plans-dir")
    ap.add_argument("--prepared-profile", type=Path,
                    help="optional measured prepared profile for model validation")
    ap.add_argument("--campaign-analysis", type=Path,
                    help="optional same-payload campaign analysis.json; compares "
                         "the modeled saving with interleaved round medians")
    ap.add_argument("--profile-key", default="",
                    help="profile key for --campaign-analysis, e.g. tiny_vit.b8")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    profile = json.loads(args.profile.read_text())
    run = _profile_run(profile)
    program = load_hybrid_program(
        args.plans_dir / f"{args.manifest}.rkhybrid.json",
        plans_dir=args.plans_dir)
    from rknpu_backend.hybrid import HybridRunner
    # Resolve the exact binding-plan shape without creating a workspace.
    runner_plans = {}
    for node in program.nodes:
        if not isinstance(node, NpuNode):
            continue
        # Use the runner's pure static resolver through an unbound instance;
        # no workspace or device resource is allocated by this cost model.
        resolver = HybridRunner.__new__(HybridRunner)
        resolver.program = program
        resolver.repeat_submit = "manifest"
        binding_spec = HybridRunner._static_binding_plan(resolver, node)
        runner_plans[node.id] = binding_spec
    binding_slots = _npu_binding_count(program, runner_plans)
    prepared_estimate = _prepared_candidate(
        program, run, args.prepared_profile)
    auto_selected = ("prepared"
                     if binding_slots >= MIN_PREPARED_BINDING_SLOTS
                     else "generic")
    summary = {
        "schema_version": 1,
        "kind": "execution-cost-screening",
        "manifest": args.manifest,
        "generic_profile": args.profile.name,
        "ledger_ms": {
            "whole_run": _ms(run["run_us"]),
            "c_total": _ms(run["envelope"]["plan_total_us"]),
            "rkplan_workspace": _ms(run["envelope"]["plan_workspace_us"]),
            "python_plan_node": _ms(run["envelope"]["python_plan_node_us"]),
            "host_actions": _ms(run["envelope"]["host_action_us"]),
            "executor": _ms(run["envelope"]["executor_overhead_us"]),
            "pack": _ms(run["envelope"]["pack_us"]),
            "submit_wait": _ms(run["envelope"]["submit_wait_us"]),
            "download": _ms(run["envelope"]["download_us"]),
            "guard": _ms(run["envelope"]["guard_us"]),
        },
        "structure": {
            "npu_binding_slots": binding_slots,
            "npu_plan_nodes": sum(isinstance(n, NpuNode) for n in program.nodes),
            "npu_submits": int(run["envelope"]["submit_count"]),
            "host_action_count": sum(
                isinstance(n, HostNode) for n in program.nodes
                if not getattr(n, "capture_only", False)),
            "host_output_bytes": _host_output_bytes(program),
        },
        "candidates": {
            "prepared_host_residency": prepared_estimate,
            "auto_selection": {
                "selected": auto_selected,
                "rule": ("static_binding_slots >= "
                         f"{MIN_PREPARED_BINDING_SLOTS}"),
                "binding_slots": binding_slots,
                "modeled_total_ms": (
                    prepared_estimate["modeled_total_ms"]
                    if auto_selected == "prepared" else _ms(run["run_us"])),
                "modeled_saving_ms": (
                    prepared_estimate["modeled_saving_ms"]
                    if auto_selected == "prepared" else 0.0),
            },
            "host_boundary_fusion": _host_fusion_candidate(program, run),
            "device_resident_qkv": _device_qkv_candidate(program, run),
        },
    }
    if args.campaign_analysis is not None:
        analysis = json.loads(args.campaign_analysis.read_text())
        profile_key = (args.profile_key or
                       f"{program.model}.b{program.batch}")
        profile = analysis.get("profiles", {}).get(profile_key)
        if profile is None:
            raise SystemExit(
                f"campaign analysis has no profile {profile_key!r}")
        reference = float(profile["generic"]["warm_ms"])
        prepared = float(profile["prepared"]["warm_ms"])
        modeled_total = float(
            summary["candidates"]["prepared_host_residency"]["modeled_total_ms"])
        modeled_saving = float(
            summary["candidates"]["prepared_host_residency"]["modeled_saving_ms"])
        summary["same_campaign_validation"] = {
            "profile": profile_key,
            "reference_arm": "generic",
            "prepared_arm": "prepared",
            "reference_ms": reference,
            "prepared_ms": prepared,
            "measured_saving_ms": reference - prepared,
            "modeled_saving_ms": modeled_saving,
            "modeled_total_ms": modeled_total,
            "prediction_error_total_ms": modeled_total - prepared,
            "prediction_error_saving_ms": modeled_saving - (reference - prepared),
            "note": "The profile JSON supplies component coefficients; this "
                    "validation uses interleaved campaign medians to avoid "
                    "sequential-run frequency drift.",
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
