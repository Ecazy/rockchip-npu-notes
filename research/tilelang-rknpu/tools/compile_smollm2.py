#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compile the frozen SmolLM2-135M-Instruct package to real TileLang RKPlans.

Required environment (the repository's pinned source build):

  PYTHONPATH=$PWD:$PWD/python:$PWD/.src/p0-build-src/tilelang:$PWD/.src/p0-source-env
  LD_LIBRARY_PATH=$PWD/.src/p0-build-tilelang/lib:$PWD/.src/p0-build-tvmffi-python/lib:$PWD/.pydeps-p0/z3/lib

The generated package contains:
  * one canonical FP16 weight file (decoder, embedding, norms);
  * 30 layers x 2 profiles x 4 projections = 240 real-TileLang RKPlans;
  * 12 lm_head shards x 2 profiles = 24 real-TileLang RKPlans;
  * 2 QK and 2 PV dynamic-weight real-TileLang RKPlans;
  * a manifest with fixed model/file identities and complete plan metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from rknpu_backend import (CompileOptions, compile_tilelang, default_profile,
                           save_plan)

ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
EXPECTED_FILE_SHA256 = {
    "config.json": "8eb740e8bbe4cff95ea7b4588d17a2432deb16e8075bc5828ff7ba9be94d982a",
    "model.safetensors": "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c",
    "tokenizer.json": "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
    "tokenizer_config.json": "4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df",
    "generation_config.json": "87b916edaaab66b3899b9d0dd0752727dff6666686da0504d89ae0a6e055a013",
    "special_tokens_map.json": "2b7379f3ae813529281a5c602bc5a11c1d4e0a99107aaa597fe936c1e813ca52",
}
CAPACITY = 256
LM_HEAD_SHARDS = 12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_arrays(path: Path) -> str:
    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as z:
        for key in sorted(z.files):
            arr = np.asarray(z[key])
            h.update(key.encode())
            h.update(str(arr.dtype).encode())
            h.update(str(arr.shape).encode())
            h.update(arr.tobytes())
    return h.hexdigest()


def _f16(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float16)


def load_weights(model_dir: Path, out: Path) -> dict[str, np.ndarray]:
    from safetensors.torch import load_file
    import torch

    src = model_dir / "model.safetensors"
    state = load_file(str(src), device="cpu")
    out_weights: dict[str, np.ndarray] = {}

    def fp16(name: str):
        return _f16(state[name].to(torch.float32).numpy())

    out_weights["embed"] = fp16("model.embed_tokens.weight")
    out_weights["norm"] = fp16("model.norm.weight")
    for i in range(30):
        base = f"model.layers.{i}"
        q = state[f"{base}.self_attn.q_proj.weight"].to(torch.float32)
        k = state[f"{base}.self_attn.k_proj.weight"].to(torch.float32)
        v = state[f"{base}.self_attn.v_proj.weight"].to(torch.float32)
        gate = state[f"{base}.mlp.gate_proj.weight"].to(torch.float32)
        up = state[f"{base}.mlp.up_proj.weight"].to(torch.float32)
        out_weights[f"layer.{i}.qkv"] = _f16(
            torch.cat((q, k, v), dim=0).numpy())
        out_weights[f"layer.{i}.o"] = fp16(
            f"{base}.self_attn.o_proj.weight")
        out_weights[f"layer.{i}.gate_up"] = _f16(
            torch.cat((gate, up), dim=0).numpy())
        out_weights[f"layer.{i}.down"] = fp16(f"{base}.mlp.down_proj.weight")
        out_weights[f"layer.{i}.input_norm"] = fp16(
            f"{base}.input_layernorm.weight")
        out_weights[f"layer.{i}.post_norm"] = fp16(
            f"{base}.post_attention_layernorm.weight")
    np.savez(out, **out_weights)
    return out_weights


def compile_one(name: str, m: int, k: int, n: int, *, b_input: bool,
                out_dir: Path, profile, options) -> dict:
    import models.tilelang_matmul as tmm

    t0 = time.perf_counter()
    prim = tmm.make_matmul_prim(m, k, n)
    result = compile_tilelang(prim, profile=profile, options=options,
                              b_is_input=b_input)
    plan = result.plan
    path = out_dir / f"{name}.rkplan.json"
    save_plan(plan, str(path))
    elapsed = time.perf_counter() - t0
    templates = [a.template_id for a in plan.actions if a.kind == "npu_task"]
    if not templates or any(not t.startswith("co.mm") for t in templates):
        raise SystemExit(f"{name}: unexpected task templates {templates}")
    constants = [{"name": c.name, "shape": list(c.shape),
                  "nbytes": int(c.nbytes), "sha256": c.sha256}
                 for c in plan.constants]
    if not b_input:
        if [c["name"] for c in constants] != ["B"]:
            raise SystemExit(f"{name}: expected one B constant")
    elif constants:
        raise SystemExit(f"{name}: dynamic-weight plan has constants")
    entry = {
        "path": f"plans/{path.name}",
        "plan_hash": plan.plan_hash,
        "source_hash": plan.source_hash,
        "profile_hash": plan.profile_hash,
        "m": m, "k": k, "n": n, "b_is_input": bool(b_input),
        "task_count": int(plan.accounting.task_count),
        "submit_count": int(plan.accounting.submit_count),
        "publish_count": len(plan.publish),
        "data_bo_count": int(plan.accounting.data_bo_count),
        "constant_bytes": int(sum(c.nbytes for c in plan.constants)),
        "constants": constants,
        "templates": templates,
        "compile_us": elapsed * 1e6,
        "artifact_bytes": path.stat().st_size,
    }
    print(f"compiled {name:34s} tasks={entry['task_count']:4d} "
          f"submits={entry['submit_count']:3d} "
          f"const={entry['constant_bytes']/1e6:8.3f}MB "
          f"{elapsed:6.2f}s", flush=True)
    return entry


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(
        ROOT / ".data/models/SmolLM2-135M-Instruct"))
    ap.add_argument("--out", default=str(
        ROOT / ".data/models/smollm2_135m_instruct_package"))
    args = ap.parse_args(argv)
    model_dir = Path(args.model_dir).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    plans_dir = out / "plans"
    plans_dir.mkdir(exist_ok=True)

    for name, expected in EXPECTED_FILE_SHA256.items():
        path = model_dir / name
        if not path.is_file():
            raise SystemExit(f"missing fixed model file: {path}")
        got = sha256(path)
        if got != expected:
            raise SystemExit(f"{name} hash mismatch: {got} != {expected}")

    config_raw = json.loads((model_dir / "config.json").read_text())
    tokenizer_cfg = json.loads((model_dir / "tokenizer_config.json").read_text())
    generation = json.loads((model_dir / "generation_config.json").read_text())
    cfg = {
        "hidden_size": int(config_raw["hidden_size"]),
        "intermediate_size": int(config_raw["intermediate_size"]),
        "vocab_size": int(config_raw["vocab_size"]),
        "num_hidden_layers": int(config_raw["num_hidden_layers"]),
        "num_attention_heads": int(config_raw["num_attention_heads"]),
        "num_key_value_heads": int(config_raw["num_key_value_heads"]),
        "head_dim": int(config_raw["hidden_size"]) //
                    int(config_raw["num_attention_heads"]),
        "rms_norm_eps": float(config_raw["rms_norm_eps"]),
        "rope_theta": float(config_raw["rope_theta"]),
        "eos_token_id": int(generation["eos_token_id"]),
        "bos_token_id": int(generation["bos_token_id"]),
        "pad_token_id": int(generation["pad_token_id"]),
        "tie_word_embeddings": bool(config_raw["tie_word_embeddings"]),
        "hidden_act": str(config_raw["hidden_act"]),
        "max_position_embeddings": int(config_raw["max_position_embeddings"]),
    }
    checks = {
        "architecture": config_raw["architectures"] == ["LlamaForCausalLM"],
        "rope_scaling": config_raw.get("rope_scaling") is None,
        "rope_interleaved": config_raw.get("rope_interleaved") is False,
        "attention_bias": config_raw.get("attention_bias") is False,
        "mlp_bias": config_raw.get("mlp_bias") is False,
        "hidden_act": cfg["hidden_act"] == "silu",
        "vocab": cfg["vocab_size"] == 49152,
        "heads": cfg["num_attention_heads"] == 9,
        "kv_heads": cfg["num_key_value_heads"] == 3,
        "layers": cfg["num_hidden_layers"] == 30,
        "hidden": cfg["hidden_size"] == 576,
        "intermediate": cfg["intermediate_size"] == 1536,
        "head_dim": cfg["head_dim"] == 64,
        "tied": cfg["tie_word_embeddings"],
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise SystemExit(f"unexpected SmolLM2 config: {failed}")

    weights_path = out / "weights.npz"
    print("materialising canonical FP16 weights", flush=True)
    load_weights(model_dir, weights_path)
    weights_sha = hash_arrays(weights_path)
    file_sha = sha256(weights_path)
    print(f"weights written {weights_path.stat().st_size/1e6:.1f} MB "
          f"sha256={file_sha}", flush=True)

    profile = default_profile()
    options = CompileOptions(numeric_mode="rk_f16_tiled",
                             allow_approx_lut=False)
    plans: dict[str, dict] = {}
    t_start = time.perf_counter()
    for layer in range(cfg["num_hidden_layers"]):
        shapes = {
            "qkv": (1, 576, 960),
            "o": (1, 576, 576),
            "gate_up": (1, 576, 3072),
            "down": (1, 1536, 576),
        }
        for role, (m, k, n) in shapes.items():
            plans[f"layer.{layer}.{role}.decode"] = compile_one(
                f"layer_{layer:02d}_{role}_decode", m, k, n, b_input=False,
                out_dir=plans_dir, profile=profile, options=options)
        shapes = {
            "qkv": (CAPACITY, 576, 960),
            "o": (CAPACITY, 576, 576),
            "gate_up": (CAPACITY, 576, 3072),
            "down": (CAPACITY, 1536, 576),
        }
        for role, (m, k, n) in shapes.items():
            plans[f"layer.{layer}.{role}.prefill"] = compile_one(
                f"layer_{layer:02d}_{role}_prefill", m, k, n, b_input=False,
                out_dir=plans_dir, profile=profile, options=options)

    head_width = (cfg["vocab_size"] + LM_HEAD_SHARDS - 1) // LM_HEAD_SHARDS
    for j in range(LM_HEAD_SHARDS):
        start, stop = j * head_width, min(cfg["vocab_size"],
                                          (j + 1) * head_width)
        width = stop - start
        plans[f"lm_head.decode.{j:02d}"] = compile_one(
            f"lm_head_decode_{j:02d}", 1, 576, width, b_input=False,
            out_dir=plans_dir, profile=profile, options=options)
        plans[f"lm_head.prefill.{j:02d}"] = compile_one(
            f"lm_head_prefill_{j:02d}", CAPACITY, 576, width, b_input=False,
            out_dir=plans_dir, profile=profile, options=options)

    plans["attn.qk.decode"] = compile_one(
        "attn_qk_decode", 3, 64, CAPACITY, b_input=True, out_dir=plans_dir,
        profile=profile, options=options)
    plans["attn.pv.decode"] = compile_one(
        "attn_pv_decode", 3, CAPACITY, 64, b_input=True, out_dir=plans_dir,
        profile=profile, options=options)
    per_group = cfg["num_attention_heads"] // cfg["num_key_value_heads"]
    plans["attn.qk.prefill"] = compile_one(
        "attn_qk_prefill", per_group * CAPACITY, 64, CAPACITY, b_input=True,
        out_dir=plans_dir, profile=profile, options=options)
    plans["attn.pv.prefill"] = compile_one(
        "attn_pv_prefill", per_group * CAPACITY, CAPACITY, 64, b_input=True,
        out_dir=plans_dir, profile=profile, options=options)
    compile_total = time.perf_counter() - t_start

    total_const = sum(p["constant_bytes"] for p in plans.values())
    total_artifacts = sum(p["artifact_bytes"] for p in plans.values())
    total_tasks = sum(p["task_count"] for p in plans.values())
    total_submits = sum(p["submit_count"] for p in plans.values())
    manifest = {
        "schema_version": 1,
        "kind": "smollm2-rkplan-package",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "model_files": {name: sha256(model_dir / name)
                        for name in EXPECTED_FILE_SHA256},
        "config_source": config_raw,
        "tokenizer": {
            "class": tokenizer_cfg.get("tokenizer_class"),
            "bos_token": tokenizer_cfg.get("bos_token"),
            "eos_token": tokenizer_cfg.get("eos_token"),
            "pad_token": tokenizer_cfg.get("pad_token"),
            "chat_template": tokenizer_cfg.get("chat_template"),
        },
        "config": cfg,
        "capacity": CAPACITY,
        "prefill_bucket": CAPACITY,
        "lm_head_shards": LM_HEAD_SHARDS,
        "lm_head_shard_width": head_width,
        "weights": {
            "path": weights_path.name,
            "file_sha256": file_sha,
            "semantic_sha256": weights_sha,
            "dtype": "float16",
            "nbytes": weights_path.stat().st_size,
        },
        "compiler": {
            "entry": "rknpu_backend.compile_tilelang",
            "source": "models/tilelang_matmul.py:make_matmul_prim",
            "numeric_mode": options.numeric_mode,
            "allow_approx_lut": options.allow_approx_lut,
            "profile_hash": profile.profile_hash,
            "chip_id": profile.chip_id,
            "task_primitive": "co.mm.f16.v1 / co.mm_acc.f16.v1",
        },
        "plans": plans,
        "resources": {
            "plan_count": len(plans),
            "task_count": total_tasks,
            "submit_count": total_submits,
            "unique_plan_constant_bytes": total_const,
            "plan_artifact_bytes": total_artifacts,
            "host_weight_bytes": int(weights_path.stat().st_size),
            "kv_cache_bytes": int(2 * cfg["num_hidden_layers"] *
                                  cfg["num_key_value_heads"] * CAPACITY *
                                  cfg["head_dim"] * 2),
            "compile_us": compile_total * 1e6,
        },
        "coverage": {
            "npu_plan_operations": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "attention_qk", "attention_pv", "lm_head_shards",
            ],
            "host_actions": [
                "smollm.embed_padded.v1", "rmsnorm.last_axis.v1",
                "smollm.qkv_split_gqa.v1", "smollm.rope_half.v1",
                "smollm.kv_append_inplace.v1", "smollm.qk_layout_gqa.v1",
                "smollm.causal_softmax.v1", "smollm.merge_gqa_heads.v1",
                "swiglu.packed_gate_up.v1", "tensor.add.v1",
                "smollm.concat_logits.v1", "smollm.greedy_sample.v1",
            ],
            "explicit_non_npu": [
                "embedding lookup", "RMSNorm", "RoPE", "stable softmax",
                "GQA layout/merge", "SwiGLU elementwise", "residual add",
                "logits concatenation", "greedy sampling",
            ],
            "no_rknn_rkllm": True,
            "no_cpu_matrix_fallback": True,
        },
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps(manifest["resources"], indent=2, sort_keys=True))
    print(f"package: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
