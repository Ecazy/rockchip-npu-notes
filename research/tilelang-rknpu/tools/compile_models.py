#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compile the two frozen model graphs from real TileLang source to RKPlan."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from rknpu_backend import (CompileOptions, compile_rkir, compile_tilelang,
                           default_profile, load_profile, matmul_module,
                           rmsnorm_approx_module, save_plan)

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def transformer_shapes(batch: int):
    m = batch * 16
    return {
        "patch": (m, 64, 64, False),
        "qkv": (m, 64, 192, False),
        "attn_out": (m, 64, 64, False),
        "gate": (m, 64, 128, False),
        "up": (m, 64, 128, False),
        # One packed N=2H GEMM: Wgu = concat(Wg, Wu) on N, so the shared input
        # is uploaded and submitted once rather than twice.
        "gate_up": (m, 64, 256, False),
        "down": (m, 128, 64, False),
        "classifier": (batch, 64, 16, False),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "models" / "compiled"))
    ap.add_argument("--profile", default="",
                    help="optional capability profile path for an isolated "
                         "experimental package")
    ap.add_argument("--include-rmsnorm-approx", action="store_true",
                    help="compile the independent approximate full-device "
                         "RMSNorm plans into this package")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    profile = load_profile(args.profile) if args.profile else default_profile()
    options = CompileOptions(numeric_mode="rk_f16_tiled",
                             allow_approx_lut=False)
    # The numeric probe is intentionally the frozen two-task Add+Max oracle.
    probe_options = CompileOptions(numeric_mode="rk_f16_tiled",
                                   allow_approx_lut=False,
                                   fuse_add_relu=False)

    import models.tilelang_matmul as tmm
    import models.tilelang_mlp as tmlp

    manifest = {
        "schema_version": 1,
        "profile_hash": profile.profile_hash,
        "compiler": "real TileLang PrimFunc -> rknpu_backend.compile_tilelang",
        "numeric_mode": options.numeric_mode,
        "allow_approx_lut": options.allow_approx_lut,
        "plans": {},
        "source_files": {
            "models/tilelang_mlp.py": sha256(ROOT / "models/tilelang_mlp.py"),
            "models/tilelang_matmul.py": sha256(ROOT / "models/tilelang_matmul.py"),
        },
    }

    def write(plan, name):
        path = out / f"{name}.rkplan.json"
        save_plan(plan, str(path))
        manifest["plans"][name] = {
            "path": path.name, "plan_hash": plan.plan_hash,
            "source_hash": plan.source_hash, "batch": plan.bindings[0].shape[0],
        }
        return path

    for batch in (1, 8, 32):
        plan = compile_tilelang(tmlp.make_mlp_prim(batch), profile=profile,
                                options=options).plan
        write(plan, f"mnist_mlp_b{batch}")
        print("wrote", f"mnist_mlp_b{batch}", plan.plan_hash, flush=True)

    probe = compile_tilelang(tmlp.make_relu_probe_prim(), profile=profile,
                             options=probe_options).plan
    templates = [a.template_id for a in probe.actions
                 if a.kind == "npu_task"]
    if templates != ["co.ew_add.f16.v1", "co.ew_max.f16.v1"]:
        raise SystemExit(f"unexpected ReLU probe templates: {templates}")
    write(probe, "mnist_relu_probe")
    manifest["plans"]["mnist_relu_probe"].update({
        "role": "numeric-acceptance-relu-probe",
        "templates": templates,
        "constant_inputs": ["X", "B"],
    })
    print("wrote mnist_relu_probe", probe.plan_hash, flush=True)

    diag_specs = [
        ("mnist_mlp_diag_l1_bias", tmlp.make_mlp_layer1_bias_prim,
         "layer1-pre-relu-bias"),
        ("mnist_mlp_diag_l1_relu", tmlp.make_mlp_layer1_relu_prim,
         "layer1-exact-relu"),
        ("mnist_mlp_diag_l2_bias", tmlp.make_mlp_layer2_bias_prim,
         "layer2-pre-relu-bias"),
        ("mnist_mlp_diag_l2_relu", tmlp.make_mlp_layer2_relu_prim,
         "layer2-exact-relu"),
        ("mnist_mlp_diag_l3_bias", tmlp.make_mlp_layer3_bias_prim,
         "layer3-final-preactivation-bias"),
    ]
    manifest["mlp_hidden_diagnostics"] = {}
    for name, factory, role in diag_specs:
        diag = compile_tilelang(factory(), profile=profile,
                                options=options).plan
        write(diag, name)
        diag_templates = [a.template_id for a in diag.actions
                          if a.kind == "npu_task"]
        manifest["plans"][name].update({
            "role": f"numeric-acceptance-mlp-{role}",
            "templates": diag_templates,
        })
        manifest["mlp_hidden_diagnostics"][name] = {
            "role": role,
            "plan_hash": diag.plan_hash,
            "templates": diag_templates,
            "output_binding": "Y",
        }
        print("wrote", name, diag.plan_hash, flush=True)

    for batch in (1, 8):
        plans = {}
        for name, (m, k, n, binp) in transformer_shapes(batch).items():
            plans[name] = compile_tilelang(
                tmm.make_matmul_prim(m, k, n), profile=profile,
                options=options, b_is_input=binp).plan
        # Same-policy residual Add on device: the existing EW_ADD template is
        # consumed directly by the following resident value.  The plan keeps
        # the GEMM and Add in one chain; no arithmetic or transfer is silently
        # removed.  The host-add manifests remain the control.
        add_options = CompileOptions(numeric_mode="rk_f16_tiled",
                                     allow_approx_lut=False,
                                     fuse_task_chain=True)
        m = batch * 16
        plans["attn_out_add"] = compile_rkir(
            matmul_module(m=m, k=64, n=64, residual="add", b_layout="kn",
                          name=f"tiny_vit_b{batch}_attn_out_add"),
            profile=profile, options=add_options).plan
        plans["down_add"] = compile_rkir(
            matmul_module(m=m, k=128, n=64, residual="add", b_layout="kn",
                          name=f"tiny_vit_b{batch}_down_add"),
            profile=profile, options=add_options).plan
        write(plans["attn_out_add"], f"tiny_vit_b{batch}_attn_out_add")
        manifest["plans"][f"tiny_vit_b{batch}_attn_out_add"].update({
            "role": "device-residual-add-attn-out",
            "submit_count": len(plans["attn_out_add"].submits),
        })
        write(plans["down_add"], f"tiny_vit_b{batch}_down_add")
        manifest["plans"][f"tiny_vit_b{batch}_down_add"].update({
            "role": "device-residual-add-down",
            "submit_count": len(plans["down_add"].submits),
        })
        # Full-device RMSNorm candidate: FP16 square/reduction, bounded rsqrt
        # LUT and gamma expansion are all device tasks in one chain.  This is a
        # distinct approximate numeric policy; the Host FP32 path stays default.
        if args.include_rmsnorm_approx:
            rmsnorm_options = CompileOptions(
                numeric_mode="rk_f16_tiled", allow_approx_lut=True,
                fuse_task_chain=True)
            for norm_id in (1, 2):
                rmsnorm_plan = compile_rkir(
                    rmsnorm_approx_module(
                        m=m, d=64,
                        name=f"tiny_vit_b{batch}_rmsnorm{norm_id}_approx"),
                    profile=profile, options=rmsnorm_options).plan
                plan_name = f"tiny_vit_b{batch}_rmsnorm{norm_id}_approx"
                write(rmsnorm_plan, plan_name)
                manifest["plans"][plan_name].update({
                    "role": f"device-approx-rmsnorm-{norm_id}",
                    "templates": [a.template_id for a in rmsnorm_plan.actions
                                  if a.kind == "npu_task"],
                    "submit_count": len(rmsnorm_plan.submits),
                    "transfers": rmsnorm_plan.accounting.transfers,
                })
        # QK and PV use the same 16x16 dynamic-B primitive; one immutable plan is
        # intentionally reused by the model runner for both operations.
        plans["qk"] = compile_tilelang(tmm.make_matmul_prim(16, 16, 16),
                                       profile=profile, options=options,
                                       b_is_input=True).plan
        plans["pv"] = plans["qk"]
        # Device-resident SwiGLU FFN: one real TileLang source lowers to the
        # complete gate/up -> SiLU -> gated product -> down -> residual chain.
        # The LUT is explicitly approved for this plan; the manifest records the
        # changed numeric contract separately from the host-swiglu path.
        ffn_options = CompileOptions(numeric_mode="rk_f16_tiled",
                                     allow_approx_lut=True)
        m = batch * 16
        ffn_kernel = tmm.make_swiglu_ffn_f16(m=m, d=64, h=128)
        ffn = compile_tilelang(ffn_kernel, profile=profile,
                               options=ffn_options).plan
        write(ffn, f"tiny_vit_b{batch}_ffn")
        manifest["plans"][f"tiny_vit_b{batch}_ffn"].update({
            "role": "device-resident-swiglu-ffn",
            "templates": [a.template_id for a in ffn.actions
                          if a.kind == "npu_task"],
            "transfers": ffn.accounting.transfers,
        })
        # Same tasks, buffers, arithmetic and transfers as the device FFN
        # control, but one dependency-ordered CO_CHAIN Submit instead of six
        # stage envelopes.  The mixed-template chain is independently validated
        # by the C verifier; the per-stage plan above stays as the control.
        ffn_fused_options = CompileOptions(
            numeric_mode="rk_f16_tiled", allow_approx_lut=True,
            fuse_task_chain=True)
        ffn_fused = compile_tilelang(ffn_kernel, profile=profile,
                                     options=ffn_fused_options).plan
        write(ffn_fused, f"tiny_vit_b{batch}_ffn_fused")
        manifest["plans"][f"tiny_vit_b{batch}_ffn_fused"].update({
            "role": "device-resident-swiglu-ffn-fused-submit",
            "source_plan": f"tiny_vit_b{batch}_ffn",
            "templates": [a.template_id for a in ffn_fused.actions
                          if a.kind == "npu_task"],
            "transfers": ffn_fused.accounting.transfers,
            "submit_count": len(ffn_fused.submits),
        })
        # Capture-only trace plan: same extracted graph and primitive tasks, but
        # exports the intermediates that the frozen numeric layer gates require.
        # It is never executed in the measured production path.
        from rknpu_backend.frontend.tilelang_adapter import extract
        trace_module = extract(ffn_kernel, options=ffn_options,
                               profile_hash=profile.profile_hash)
        trace_names = {4: "gate", 5: "up", 7: "act", 8: "ffn"}
        trace_outputs = tuple(trace_names)
        # Drop the final residual add/Y from the trace module: the production
        # plan owns h2, while the frozen FFN layer gates describe the down
        # projection before that residual add.
        trace_values = tuple(
            replace(value,
                    id=9 if value.id == 10 else value.id,
                    name=trace_names.get(value.id, value.name),
                    role="output" if value.id in trace_names else value.role)
            for value in trace_module.values if value.id != 9)
        trace_ops = tuple(op for op in trace_module.ops if op.id != 5)
        trace_module = replace(trace_module,
                               name=f"{trace_module.name}:trace",
                               values=trace_values, ops=trace_ops,
                               inputs=(0, 9), outputs=trace_outputs)
        trace_plan = compile_rkir(trace_module, profile=profile,
                                  options=ffn_options).plan
        write(trace_plan, f"tiny_vit_b{batch}_ffn_trace")
        manifest["plans"][f"tiny_vit_b{batch}_ffn_trace"].update({
            "role": "numeric-capture-trace",
            "source_plan": f"tiny_vit_b{batch}_ffn",
            "outputs": list(trace_names.values()),
        })
        for name, plan in plans.items():
            write(plan, f"tiny_vit_b{batch}_{name}")
        # Static-repeat groups: the QK/PV node is declared N = batch * heads
        # times, so pre-expand the primitive into one plan with N independent
        # tasks (three shared BOs, disjoint slices, one chained submit).  This is
        # the only geometry the admission rules accept, so the expansion cannot
        # silently apply to a different shape.
        from rknpu_backend.passes.repeat_expand import (expand_static_repeat,
                                                        remap_group_inputs)
        heads = 4
        nrepeat = batch * heads
        group = expand_static_repeat(plans["qk"], nrepeat, shared_bo=True)
        write(group, f"tiny_vit_b{batch}_qk_group")
        manifest["plans"][f"tiny_vit_b{batch}_qk_group"].update({
            "role": "static-repeat-group",
            "source_plan": f"tiny_vit_b{batch}_qk",
            "repeat": nrepeat,
        })
        # Restricted QKV input mapping: source bindings carry the full
        # [16,192] per-batch QKV block; each per-head upload action selects a
        # 16-column window and, for K, transposes it into the weight tile.
        qk_mapped = remap_group_inputs(group, nrepeat, [
            {"shape": (16, 192), "row_stride": 192, "col_base": 0,
             "col_stride": 16, "col_modulus": heads, "transpose": False},
            {"shape": (16, 192), "row_stride": 192, "col_base": 64,
             "col_stride": 16, "col_modulus": heads, "transpose": False},
        ])
        write(qk_mapped, f"tiny_vit_b{batch}_qk_qkv_group")
        manifest["plans"][f"tiny_vit_b{batch}_qk_qkv_group"].update({
            "role": "mapped-qkv-attn-group",
            "source_plan": f"tiny_vit_b{batch}_qk",
            "repeat": nrepeat,
            "input_map_version": 1,
        })
        # PV keeps probs as a native [16,16] slice, while V is read directly
        # from the QKV block's V columns (no precomputed v_heads array).
        pv_mapped = remap_group_inputs(group, nrepeat, [
            {"shape": (16, 16), "row_stride": 16, "col_base": 0,
             "col_stride": 0, "col_modulus": 1, "transpose": False},
            {"shape": (16, 192), "row_stride": 192, "col_base": 128,
             "col_stride": 16, "col_modulus": heads, "transpose": True},
        ])
        write(pv_mapped, f"tiny_vit_b{batch}_pv_qkv_group")
        manifest["plans"][f"tiny_vit_b{batch}_pv_qkv_group"].update({
            "role": "mapped-qkv-attn-group",
            "source_plan": f"tiny_vit_b{batch}_qk",
            "repeat": nrepeat,
            "input_map_version": 1,
        })
        print("wrote", f"tiny_vit_b{batch}_qk_group", group.plan_hash, flush=True)
        print("wrote", f"tiny_vit_b{batch}", flush=True)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    subprocess.run([
        sys.executable, str(ROOT / "tools" / "build_hybrid_manifests.py"),
        "--plans-dir", str(out),
    ], check=True)
    print(out / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
