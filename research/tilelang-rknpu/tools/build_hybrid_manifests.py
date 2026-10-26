#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Emit the frozen, versioned rkhybrid execution manifests.

The tool is a package writer, not an importer: every graph below is an explicit
function of the two frozen model architectures and the already-compiled RKPlan
manifest.  It never accepts an arbitrary model file or shape.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from rknpu_backend import load_plan

ROOT = Path(__file__).resolve().parents[1]
HYBRID_SCHEMA_VERSION = 1


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor(name: str, dtype: str, shape, axes) -> dict:
    return {"name": name, "dtype": dtype, "shape": list(shape),
            "axes": list(axes)}


def ref(name: str, index=()) -> dict:
    value = {"tensor": name}
    if index:
        value["index"] = list(index)
    return value


def host_node(node_id: str, action: str, inputs: dict, outputs: dict,
              *, deps=(), attrs=None, phase="execution",
              numeric=None, depends_nodes=None) -> dict:
    return {
        "id": node_id, "kind": "host_action", "action": action,
        "action_version": 1, "inputs": inputs, "outputs": outputs,
        "attrs": attrs or {}, "depends_on": list(depends_nodes or deps),
        "repeat": [], "phase": phase,
        "numeric_contract": numeric or {
            "storage": "float16", "accumulation": "float32",
            "writeback": "float16", "semantics": "registered_host_action"},
        "bridge": {"kind": "host_tensor_materialization",
                   "accounting": "tensor_bytes"},
    }


def plan_node(node_id: str, plan_key: str, inputs: dict, outputs: dict,
              constants: dict, *, deps, numeric, repeat=None,
              repeat_submit=None, group_plan=None) -> dict:
    node = {
        "id": node_id, "kind": "rkplan", "plan": plan_key,
        "inputs": inputs, "outputs": outputs, "constants": constants,
        "depends_on": list(deps), "repeat": list(repeat or []),
        "phase": "execution", "numeric_contract": numeric,
        "bridge": {"kind": "rkplan_transfer",
                   "accounting": "runtime_report"},
    }
    if repeat_submit is not None:
        node["repeat_submit"] = repeat_submit
    if group_plan is not None:
        node["group_plan"] = group_plan
    return node


def plan_entry(compiled: dict, key: str, dynamic_operand=None) -> dict:
    info = compiled["plans"][key]
    return {"path": info["path"], "plan_hash": info["plan_hash"],
            "source_hash": info["source_hash"],
            "dynamic_operand": dynamic_operand}


def weight_constants(plan, mapping: dict[str, dict]) -> dict:
    out = {}
    for spec in plan.constants:
        if spec.name in mapping:
            desc = dict(mapping[spec.name])
        elif spec.generator is not None:
            desc = {"kind": "generated", "generator": spec.generator}
        else:
            raise SystemExit(f"no explicit constant transform for {spec.name}")
        desc["kind"] = desc.get("kind", "weight")
        out[spec.name] = desc
    return out


def weight_meta(weights: dict[str, np.ndarray]) -> dict:
    return {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in sorted(weights.items())}


def policy() -> tuple[dict, dict]:
    return ({
        "mode": "explicit_hybrid",
        "host_actions_version": 1,
        "plan_reuse": "per_plan_workspace",
        "host_fallback": "forbidden",
        "npu_fallback": "forbidden",
        "dynamic_shapes": "forbidden",
        "unknown_action": "reject",
    }, {
        "host_action": "sum_resolved_input_output_tensor_bytes",
        "rkplan": "C_RKRunReport_upload_plus_download_bytes",
        "final_output": "host_materialized",
        "npu_fallback": "forbidden",
    })


def mlp_manifest(batch: int, compiled: dict, weights: dict,
                 weights_sha: str, profile_hash: str) -> dict:
    m = batch
    tensors = [
        tensor("x_flat", "float16", (m, 784), ("batch", "feature")),
        tensor("x_padded", "float16", (m, 1024), ("batch", "ordered_k")),
        tensor("logits_padded", "float16", (m, 16), ("batch", "padded_class")),
        tensor("logits", "float32", (m, 10), ("batch", "class")),
    ]
    plan = load_plan(str((ROOT / "models" / "compiled" /
                         compiled["plans"][f"mnist_mlp_b{batch}"]["path"])))
    constants = weight_constants(plan, {
        "W1": {"kind": "weight_repeat_pad", "name": "W1",
               "reshape": [784, 256]},
        "B1": {"kind": "weight_repeat_pad", "name": "b1",
               "reshape": [1, 256], "repeat_rows": batch},
        "W2": {"kind": "weight_repeat_pad", "name": "W2",
               "reshape": [256, 128]},
        "B2": {"kind": "weight_repeat_pad", "name": "b2",
               "reshape": [1, 128], "repeat_rows": batch},
        "W3": {"kind": "weight_repeat_pad", "name": "W3",
               "reshape": [128, 10]},
        "B3": {"kind": "weight_repeat_pad", "name": "b3",
               "reshape": [1, 10], "repeat_rows": batch},
    })
    nodes = [
        host_node("normalize", "mnist.normalize_flatten.v1",
                  {"raw": ref("raw")}, {"x": ref("x_flat")},
                  phase="preprocess",
                  numeric={"storage": "float16", "accumulation": "float32",
                           "writeback": "float16",
                           "semantics": "normalize_uint8_with_float32_then_one_float16_store"}),
        host_node("pad_k", "tensor.pad_2d.v1",
                  {"x": ref("x_flat")}, {"y": ref("x_padded")},
                  depends_nodes=["normalize"], phase="preprocess",
                  attrs={"input_shape": [m, 784],
                         "copy_shape": [m, 784],
                         "output_shape": [m, 1024]},
                  numeric={"storage": "float16", "accumulation": "none",
                           "writeback": "zero_pad_top_left",
                           "semantics": "logical_784_to_ordered_1024_k"}),
        plan_node("mlp", "mlp", {"X": ref("x_padded")},
                  {"Y": ref("logits_padded")}, constants, deps=["pad_k"],
                  numeric={"storage": "float16",
                           "accumulation": "float16_per_256_k_tile",
                           "writeback": "explicit_float16_each_k_tile",
                           "ordered_k_tile": 256,
                           "semantics": "real_tilelang_staged_mlp_with_device_resident_activations"}),
        host_node("export_logits", "logits.prefix.v1",
                  {"logits": ref("logits_padded")}, {"y": ref("logits")},
                  depends_nodes=["mlp"], phase="postprocess",
                  attrs={"cols": 10},
                  numeric={"storage": "float32", "accumulation": "none",
                           "writeback": "float32_prefix",
                           "semantics": "first_10_of_padded_16_logits"}),
    ]
    exec_policy, bridge_policy = policy()
    return {
        "schema_version": HYBRID_SCHEMA_VERSION, "kind": "rkhybrid",
        "name": f"mnist_mlp_b{batch}", "model": "mnist_mlp", "batch": batch,
        "profile_hash": profile_hash,
        "execution_policy": exec_policy, "bridge_policy": bridge_policy,
        "input": tensor("raw", "uint8", (m, 28, 28), ("batch", "row", "col")),
        "host_constants": {}, "tensors": tensors,
        "plans": {"mlp": plan_entry(compiled, f"mnist_mlp_b{batch}")},
        "nodes": nodes, "outputs": {"logits": "logits"},
        "trace_tensors": ["x_padded", "logits"],
        "weights": {"file": "mnist_mlp.npz", "sha256": weights_sha,
                    "keys": weight_meta(weights),
                    "host_precision": "float16"},
    }


def vit_manifest(batch: int, compiled: dict, weights: dict,
                 weights_sha: str, profile_hash: str) -> dict:
    b, t, d, h, hd = batch, 16, 64, 4, 16
    m = b * t
    tensors = [
        tensor("patches", "float16", (b, t, 49), ("batch", "token", "patch")),
        tensor("patch_flat", "float16", (m, 64), ("batch_token", "d")),
        tensor("patch_projection", "float16", (m, 64), ("batch_token", "d")),
        tensor("positioned", "float16", (m, 64), ("batch_token", "d")),
        tensor("norm1", "float16", (m, 64), ("batch_token", "d")),
        tensor("qkv", "float16", (m, 192), ("batch_token", "qkv_head_dim")),
        tensor("q_heads", "float16", (b, h, t, hd), ("batch", "head", "token", "d")),
        tensor("k_t_heads", "float16", (b, h, hd, t), ("batch", "head", "d", "token")),
        tensor("v_heads", "float16", (b, h, t, hd), ("batch", "head", "token", "d")),
        tensor("scores_heads", "float16", (b, h, t, t), ("batch", "head", "query", "key")),
        tensor("probs_heads", "float16", (b, h, t, t), ("batch", "head", "query", "key")),
        tensor("attn_heads", "float16", (b, h, t, hd), ("batch", "head", "token", "d")),
        tensor("attn_merged", "float16", (m, 64), ("batch_token", "head_d")),
        tensor("attn_projection", "float16", (m, 64), ("batch_token", "d")),
        tensor("h1", "float16", (m, 64), ("batch_token", "d")),
        tensor("norm2", "float16", (m, 64), ("batch_token", "d")),
        tensor("gate", "float16", (m, 128), ("batch_token", "ffn")),
        tensor("up", "float16", (m, 128), ("batch_token", "ffn")),
        tensor("act", "float16", (m, 128), ("batch_token", "ffn")),
        tensor("ffn", "float16", (m, 64), ("batch_token", "d")),
        tensor("h2", "float16", (m, 64), ("batch_token", "d")),
        tensor("h2_tokens", "float16", (b, t, d), ("batch", "token", "d")),
        tensor("pooled", "float16", (b, 64), ("batch", "d")),
        tensor("logits_padded", "float16", (b, 16), ("batch", "padded_class")),
        tensor("logits", "float32", (b, 10), ("batch", "class")),
    ]
    patch_plan = load_plan(str(ROOT / "models" / "compiled" /
                               compiled["plans"][f"tiny_vit_b{batch}_patch"]["path"]))
    qkv_plan = load_plan(str(ROOT / "models" / "compiled" /
                             compiled["plans"][f"tiny_vit_b{batch}_qkv"]["path"]))
    attn_plan = load_plan(str(ROOT / "models" / "compiled" /
                              compiled["plans"][f"tiny_vit_b{batch}_attn_out"]["path"]))
    gate_plan = load_plan(str(ROOT / "models" / "compiled" /
                              compiled["plans"][f"tiny_vit_b{batch}_gate"]["path"]))
    up_plan = load_plan(str(ROOT / "models" / "compiled" /
                            compiled["plans"][f"tiny_vit_b{batch}_up"]["path"]))
    down_plan = load_plan(str(ROOT / "models" / "compiled" /
                              compiled["plans"][f"tiny_vit_b{batch}_down"]["path"]))
    cls_plan = load_plan(str(ROOT / "models" / "compiled" /
                             compiled["plans"][f"tiny_vit_b{batch}_classifier"]["path"]))
    dyn_plan = load_plan(str(ROOT / "models" / "compiled" /
                             compiled["plans"][f"tiny_vit_b{batch}_qk"]["path"]))
    plans = {
        "patch": plan_entry(compiled, f"tiny_vit_b{batch}_patch"),
        "qkv": plan_entry(compiled, f"tiny_vit_b{batch}_qkv"),
        "attn_out": plan_entry(compiled, f"tiny_vit_b{batch}_attn_out"),
        "gate": plan_entry(compiled, f"tiny_vit_b{batch}_gate"),
        "up": plan_entry(compiled, f"tiny_vit_b{batch}_up"),
        "down": plan_entry(compiled, f"tiny_vit_b{batch}_down"),
        "classifier": plan_entry(compiled, f"tiny_vit_b{batch}_classifier"),
        "attn_matmul": plan_entry(compiled, f"tiny_vit_b{batch}_qk",
                                  dynamic_operand="B"),
        # Pre-expanded static-repeat group (batch * heads independent tasks in one
        # chained submit).  Selected explicitly by the QK/PV nodes below; the
        # per-repeat plan above stays for the separate and B1 control paths.
        "attn_matmul_group": plan_entry(compiled,
                                        f"tiny_vit_b{batch}_qk_group"),
    }
    plan_numeric = {
        "storage": "float16", "accumulation": "float32_single_tile",
        "writeback": "float16_after_gemm",
        "semantics": "real_tilelang_one_tile_matmul_fp32_accumulate_one_fp16_writeback",
    }
    dynamic_numeric = {
        "storage": "float16", "accumulation": "float32_single_tile",
        "writeback": "float16_after_gemm", "dynamic_b": True,
        "semantics": "real_tilelang_16x16_dynamic_b_matmul_reused_across_batch_heads",
    }
    layout_numeric = {
        "storage": "float16", "accumulation": "none",
        "writeback": "layout_preserving",
        "semantics": "explicit_static_axis_transform",
    }
    host_norm = {
        "storage": "float16", "accumulation": "float32",
        "writeback": "float16",
        "semantics": "float32_reduction_then_one_float16_store",
    }
    nodes = [
        host_node("patchify", "mnist.patchify_norm.v1",
                  {"raw": ref("raw")}, {"patches": ref("patches")},
                  phase="preprocess",
                  numeric={"storage": "float16", "accumulation": "float32",
                           "writeback": "float16",
                           "semantics": "normalize_then_7x7_raster_patchify_float16"}),
        host_node("patch_pad", "tensor.pad_2d.v1",
                  {"x": ref("patches")}, {"y": ref("patch_flat")},
                  depends_nodes=["patchify"], phase="preprocess",
                  attrs={"input_shape": [m, 49], "copy_shape": [m, 49],
                         "output_shape": [m, 64]},
                  numeric={"storage": "float16", "accumulation": "none",
                           "writeback": "zero_pad_top_left",
                           "semantics": "flatten_tokens_and_pad_49_to_64"}),
        plan_node("patch", "patch", {"A": ref("patch_flat")},
                  {"Y": ref("patch_projection")},
                  weight_constants(patch_plan, {
                      "B": {"kind": "weight_repeat_pad", "name": "patch_w",
                            "reshape": [49, 64]}}),
                  deps=["patch_pad"], numeric=plan_numeric),
        host_node("position_add", "tiny_vit.position_add.v1",
                  {"x": ref("patch_projection"), "pos": ref("pos")},
                  {"y": ref("positioned")}, depends_nodes=["patch"],
                  attrs={"batch": b, "tokens": t}, numeric=host_norm),
        host_node("rmsnorm1", "rmsnorm.last_axis.v1",
                  {"x": ref("positioned"), "gamma": ref("norm1_gamma")},
                  {"y": ref("norm1")}, depends_nodes=["position_add"],
                  attrs={"eps": 1e-6}, numeric=host_norm),
        plan_node("qkv", "qkv", {"A": ref("norm1")}, {"Y": ref("qkv")},
                  weight_constants(qkv_plan, {
                      "B": {"kind": "weight", "name": "qkv_w"}}),
                  deps=["rmsnorm1"], numeric=plan_numeric),
        host_node("split_heads", "tiny_vit.qkv_split_heads.v1",
                  {"qkv": ref("qkv")},
                  {"q": ref("q_heads"), "k_t": ref("k_t_heads"),
                   "v": ref("v_heads")}, depends_nodes=["qkv"],
                  attrs={"batch": b, "tokens": t, "heads": h,
                         "head_dim": hd}, numeric=layout_numeric),
        plan_node("qk", "attn_matmul",
                  {"A": ref("q_heads", ("bi", "hi")),
                   "B": ref("k_t_heads", ("bi", "hi"))},
                  {"Y": ref("scores_heads", ("bi", "hi"))}, {},
                  deps=["split_heads"], numeric=dynamic_numeric,
                  repeat=[{"name": "bi", "extent": b, "axis": "batch"},
                          {"name": "hi", "extent": h, "axis": "head"}],
                  repeat_submit="shared_group",
                  group_plan="attn_matmul_group"),
        host_node("softmax", "attention.scaled_softmax.v1",
                  {"scores": ref("scores_heads")}, {"y": ref("probs_heads")},
                  depends_nodes=["qk"], attrs={"scale": 0.25, "axis": -1},
                  numeric={"storage": "float16", "accumulation": "float32",
                           "writeback": "float16",
                           "scale": 0.25,
                           "semantics": "stable_max_subtract_exp_normalize_float32_then_float16"}),
        plan_node("pv", "attn_matmul",
                  {"A": ref("probs_heads", ("bi", "hi")),
                   "B": ref("v_heads", ("bi", "hi"))},
                  {"Y": ref("attn_heads", ("bi", "hi"))}, {},
                  deps=["softmax", "split_heads"], numeric=dynamic_numeric,
                  repeat=[{"name": "bi", "extent": b, "axis": "batch"},
                          {"name": "hi", "extent": h, "axis": "head"}],
                  repeat_submit="shared_group",
                  group_plan="attn_matmul_group"),
        host_node("merge_heads", "tiny_vit.merge_heads.v1",
                  {"heads": ref("attn_heads")}, {"y": ref("attn_merged")},
                  depends_nodes=["pv"],
                  attrs={"batch": b, "tokens": t, "heads": h,
                         "head_dim": hd}, numeric=layout_numeric),
        plan_node("attn_out", "attn_out", {"A": ref("attn_merged")},
                  {"Y": ref("attn_projection")},
                  weight_constants(attn_plan, {
                      "B": {"kind": "weight", "name": "attn_out_w"}}),
                  deps=["merge_heads"], numeric=plan_numeric),
        host_node("residual_attention", "tensor.add.v1",
                  {"a": ref("positioned"), "b": ref("attn_projection")},
                  {"y": ref("h1")},
                  depends_nodes=["position_add", "attn_out"], numeric=host_norm),
        host_node("rmsnorm2", "rmsnorm.last_axis.v1",
                  {"x": ref("h1"), "gamma": ref("norm2_gamma")},
                  {"y": ref("norm2")}, depends_nodes=["residual_attention"],
                  attrs={"eps": 1e-6}, numeric=host_norm),
        plan_node("gate", "gate", {"A": ref("norm2")}, {"Y": ref("gate")},
                  weight_constants(gate_plan, {
                      "B": {"kind": "weight", "name": "ffn_gate_w"}}),
                  deps=["rmsnorm2"], numeric=plan_numeric),
        plan_node("up", "up", {"A": ref("norm2")}, {"Y": ref("up")},
                  weight_constants(up_plan, {
                      "B": {"kind": "weight", "name": "ffn_up_w"}}),
                  deps=["rmsnorm2"], numeric=plan_numeric),
        host_node("swiglu", "swiglu.gate_up.v1",
                  {"gate": ref("gate"), "up": ref("up")}, {"y": ref("act")},
                  depends_nodes=["gate", "up"], numeric=host_norm),
        plan_node("down", "down", {"A": ref("act")}, {"Y": ref("ffn")},
                  weight_constants(down_plan, {
                      "B": {"kind": "weight", "name": "ffn_down_w"}}),
                  deps=["swiglu"], numeric=plan_numeric),
        host_node("residual_ffn", "tensor.add.v1",
                  {"a": ref("h1"), "b": ref("ffn")}, {"y": ref("h2")},
                  depends_nodes=["residual_attention", "down"], numeric=host_norm),
        host_node("reshape_tokens", "tensor.reshape.v1",
                  {"x": ref("h2")}, {"y": ref("h2_tokens")},
                  depends_nodes=["residual_ffn"],
                  attrs={"shape": [b, t, d]}, numeric=layout_numeric),
        host_node("mean_pool", "tensor.mean_axis.v1",
                  {"x": ref("h2_tokens")}, {"y": ref("pooled")},
                  depends_nodes=["reshape_tokens"], attrs={"axis": 1},
                  numeric={"storage": "float16", "accumulation": "float32",
                           "writeback": "float16",
                           "semantics": "mean_over_16_tokens_in_float32_then_float16"}),
        plan_node("classifier", "classifier", {"A": ref("pooled")},
                  {"Y": ref("logits_padded")},
                  weight_constants(cls_plan, {
                      "B": {"kind": "weight_repeat_pad", "name": "class_w",
                            "reshape": [64, 10]}}),
                  deps=["mean_pool"], numeric=plan_numeric),
        host_node("export_logits", "logits.prefix.v1",
                  {"logits": ref("logits_padded")}, {"y": ref("logits")},
                  depends_nodes=["classifier"], phase="postprocess",
                  attrs={"cols": 10},
                  numeric={"storage": "float32", "accumulation": "none",
                           "writeback": "float32_prefix",
                           "semantics": "first_10_of_padded_16_logits"}),
    ]
    host_constants = {
        "pos": {"weight": "pos", "dtype": "float16",
                "shape": [t, d], "axes": ["token", "d"]},
        "norm1_gamma": {"weight": "norm1_gamma", "dtype": "float16",
                        "shape": [d], "axes": ["d"]},
        "norm2_gamma": {"weight": "norm2_gamma", "dtype": "float16",
                        "shape": [d], "axes": ["d"]},
    }
    trace_tensors = [
        "patches", "patch_flat", "patch_projection", "positioned", "norm1",
        "qkv", "q_heads", "k_t_heads", "v_heads", "scores_heads",
        "probs_heads", "attn_heads", "attn_merged", "attn_projection", "h1",
        "norm2", "gate", "up", "act", "ffn", "h2", "h2_tokens", "pooled",
        "logits_padded", "logits",
    ]
    exec_policy, bridge_policy = policy()
    return {
        "schema_version": HYBRID_SCHEMA_VERSION, "kind": "rkhybrid",
        "name": f"tiny_vit_b{batch}", "model": "tiny_vit", "batch": batch,
        "profile_hash": profile_hash,
        "execution_policy": exec_policy, "bridge_policy": bridge_policy,
        "input": tensor("raw", "uint8", (b, 28, 28), ("batch", "row", "col")),
        "host_constants": host_constants, "tensors": tensors, "plans": plans,
        "nodes": nodes, "outputs": {"logits": "logits"},
        "trace_tensors": trace_tensors,
        "weights": {"file": "tiny_vit.npz", "sha256": weights_sha,
                    "keys": weight_meta(weights),
                    "host_precision": "float16"},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans-dir", default=str(ROOT / "models" / "compiled"))
    ap.add_argument("--weights-dir", default=str(ROOT / "models"))
    args = ap.parse_args()
    plans_dir = Path(args.plans_dir).resolve()
    weights_dir = Path(args.weights_dir).resolve()
    compiled_path = plans_dir / "manifest.json"
    compiled = json.loads(compiled_path.read_text())
    if compiled.get("schema_version") != 1:
        raise SystemExit("unexpected compiled manifest schema")
    weights_files = {
        "mnist_mlp": weights_dir / "mnist_mlp.npz",
        "tiny_vit": weights_dir / "tiny_vit.npz",
    }
    weights = {name: dict(np.load(path, allow_pickle=False))
               for name, path in weights_files.items()}
    weights_sha = {name: sha256(path) for name, path in weights_files.items()}
    written = {}
    for batch in (1, 8, 32):
        path = plans_dir / f"mnist_mlp_b{batch}.rkhybrid.json"
        data = mlp_manifest(batch, compiled, weights["mnist_mlp"],
                            weights_sha["mnist_mlp"], compiled["profile_hash"])
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        written[f"mnist_mlp_b{batch}"] = {"path": path.name, "model": "mnist_mlp",
                              "batch": batch, "sha256": sha256(path)}
        print("wrote", path, flush=True)
    for batch in (1, 8):
        path = plans_dir / f"tiny_vit_b{batch}.rkhybrid.json"
        data = vit_manifest(batch, compiled, weights["tiny_vit"],
                            weights_sha["tiny_vit"], compiled["profile_hash"])
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        written[f"tiny_vit_b{batch}"] = {"path": path.name, "model": "tiny_vit",
                              "batch": batch, "sha256": sha256(path)}
        print("wrote", path, flush=True)
    compiled["hybrid_schema_version"] = HYBRID_SCHEMA_VERSION
    compiled["hybrid_manifests"] = written
    compiled["hybrid_policy"] = {
        "kind": "explicit_versioned_rkhybrid",
        "host_actions": "rknpu_backend.hybrid.HOST_ACTIONS",
        "host_actions_version": 1,
        "unknown_action": "reject",
        "npu_fallback": "forbidden",
        "host_fallback": "forbidden",
        "dynamic_shapes": "forbidden",
    }
    compiled_path.write_text(json.dumps(compiled, indent=2, sort_keys=True) + "\n")
    print(compiled_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
