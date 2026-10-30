# SPDX-License-Identifier: GPL-3.0-or-later
"""SmolLM2-135M-Instruct execution over compiled TileLang->RKPlan artifacts.

The runner is deliberately a Host state machine around a frozen set of RKPlans:

* token embedding, RMSNorm, RoPE, GQA layout/merge, stable causal softmax,
  in-place KV append, logits concatenation and greedy selection are named
  registered Host actions;
* every Q/K/V/O projection, gate/up/down projection, attention QK/PV and
  lm_head shard is an immutable RKPlan lowered from a real TileLang PrimFunc;
* prefill uses a fixed 256-token bucket and decode uses M=1 plans;
* the persistent KV cache lives across token steps and is zeroed on reset.

No model matrix operation is recomputed on the Host.  A failed RKPlan raises the
existing runtime error and the runner does not continue to a later submission.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .errors import (RK_E_INVALID, RK_E_NUMERIC_CONTRACT, RK_E_SCHEMA,
                     RK_E_SHAPE, RK_E_UNSUPPORTED, reject)
from .hybrid import HOST_ACTIONS
from .models import PlanCall, RunStats
from .plan import Plan
from .runtime.ffi import LibRKPlan


PACKAGE_SCHEMA_VERSION = 1
PACKAGE_KIND = "smollm2-rkplan-package"
DECODE = "decode"
PREFILL = "prefill"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _f16(a: Any) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float16)


@dataclass(frozen=True)
class SmolLM2Config:
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    eos_token_id: int
    bos_token_id: int
    pad_token_id: int
    tie_word_embeddings: bool
    hidden_act: str
    max_position_embeddings: int

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "SmolLM2Config":
        required = {
            "hidden_size", "intermediate_size", "vocab_size",
            "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "head_dim", "rms_norm_eps",
            "rope_theta", "eos_token_id", "bos_token_id", "pad_token_id",
            "tie_word_embeddings", "hidden_act", "max_position_embeddings",
        }
        missing, unknown = sorted(required - obj.keys()), sorted(
            obj.keys() - required)
        if missing or unknown:
            reject(RK_E_SCHEMA, "SmolLM2 config fields missing or unknown",
                   stage="smollm", actual={"missing": missing,
                                           "unknown": unknown})
        model = cls(**{k: obj[k] for k in required})
        if model.hidden_size % model.num_attention_heads:
            reject(RK_E_SHAPE, "hidden_size must divide attention heads",
                   stage="smollm")
        if model.head_dim != model.hidden_size // model.num_attention_heads:
            reject(RK_E_SHAPE,
                   "declared head_dim disagrees with hidden_size/num_heads",
                   stage="smollm")
        if model.num_attention_heads % model.num_key_value_heads:
            reject(RK_E_SHAPE, "GQA q/kv head counts must divide",
                   stage="smollm")
        if not model.tie_word_embeddings:
            reject(RK_E_UNSUPPORTED,
                   "this package requires the verified tied word embedding",
                   stage="smollm")
        return model

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "eos_token_id": self.eos_token_id,
            "bos_token_id": self.bos_token_id,
            "pad_token_id": self.pad_token_id,
            "tie_word_embeddings": self.tie_word_embeddings,
            "hidden_act": self.hidden_act,
            "max_position_embeddings": self.max_position_embeddings,
        }


_STAGE_STAT_FIELDS = (
    "upload_calls", "upload_bytes", "download_calls", "download_bytes",
    "task_count", "submit_count", "plan_call_us", "plan_workspace_us",
    "plan_pack_us", "plan_submit_wait_us", "plan_output_sync_us",
    "plan_download_us", "plan_guard_us", "plan_publish_us",
    "host_action_us", "host_action_bytes",
    "kv_state_init_calls", "kv_state_init_bytes", "kv_state_init_us",
    "kv_state_append_calls", "kv_state_append_bytes", "kv_state_append_us",
    "kv_state_clear_calls", "kv_state_clear_bytes", "kv_state_clear_us",
)


@dataclass
class GenerationResult:
    prompt_ids: list[int]
    generated_ids: list[int]
    stopped_by_eos: bool
    stop_reason: str
    prefill_us: float
    decode_us: float
    steps: list[dict[str, Any]] = field(default_factory=list)
    reset_us: float = 0.0
    sampling_us: float = 0.0
    first_sample_us: float = 0.0
    ttft_us: float = 0.0
    generate_wall_us: float = 0.0
    decode_forwards: int = 0
    output_tokens: int = 0
    stages: dict[str, Any] = field(default_factory=dict)
    profile: dict[str, Any] = field(default_factory=dict)

    @property
    def all_ids(self) -> list[int]:
        return self.prompt_ids + self.generated_ids


class SmolLM2Runner:
    """Load a frozen SmolLM2 package and run stateful batch=1 generation."""

    def __init__(self, lib: LibRKPlan, package_dir: str | Path, *, fd: int,
                 stats: RunStats | None = None,
                 layer_indices: tuple[int, ...] | None = None,
                 share_constants: bool = True,
                 persistent_kv: bool = True,
                 profile: bool = False,
                 cache_host_weights: bool = True,
                 plan_overrides: dict[str, dict[str, Any]] | None = None):
        if not isinstance(fd, int) or fd < 0:
            reject(RK_E_INVALID, "SmolLM2Runner requires a caller-owned fd>=0",
                   stage="smollm")
        if not isinstance(share_constants, bool) or not isinstance(
                persistent_kv, bool) or not isinstance(profile, bool) or                 not isinstance(cache_host_weights, bool):
            reject(RK_E_INVALID,
                   "share_constants, persistent_kv, profile and "
                   "cache_host_weights must be boolean and plan_overrides "
                   "must be a mapping", stage="smollm")
        self.share_constants = share_constants
        self.persistent_kv = persistent_kv
        self.profile = bool(profile)
        self.cache_host_weights = bool(cache_host_weights)
        self.plan_overrides = dict(plan_overrides or {})
        self._profile_us: dict[str, float] = {}
        self._profile_count: dict[str, int] = {}
        self._cached_embed: np.ndarray | None = None
        self._cached_norm_gamma: dict[str, np.ndarray] = {}
        self.lib = lib
        self.package_dir = Path(package_dir).resolve()
        self.fd = fd
        self.stats = stats or RunStats()
        load_t0 = time.perf_counter()
        self._load_plan_call_us = 0.0
        self._closed = False
        manifest_path = self.package_dir / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            reject(RK_E_SCHEMA, f"cannot read SmolLM2 package manifest: {exc}",
                   stage="smollm")
        if manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION or \
                manifest.get("kind") != PACKAGE_KIND:
            reject(RK_E_SCHEMA, "unsupported SmolLM2 package manifest",
                   stage="smollm")
        self.manifest = manifest
        self.config = SmolLM2Config.from_dict(manifest["config"])
        self.capacity = int(manifest["capacity"])
        self.prefill_bucket = int(manifest["prefill_bucket"])
        if self.prefill_bucket != self.capacity:
            reject(RK_E_UNSUPPORTED,
                   "this package expects prefill_bucket == capacity",
                   stage="smollm", actual=[self.prefill_bucket, self.capacity])
        if self.capacity < 1 or self.capacity > 4096:
            reject(RK_E_SHAPE, "capacity must be in 1..4096", stage="smollm")
        if layer_indices is None:
            self.layer_indices = tuple(range(self.config.num_hidden_layers))
        else:
            self.layer_indices = tuple(layer_indices)
            if not self.layer_indices or any(
                    i < 0 or i >= self.config.num_hidden_layers
                    for i in self.layer_indices):
                reject(RK_E_INVALID, "layer_indices are outside the model",
                       stage="smollm")
        weights_path = self.package_dir / manifest["weights"]["path"]
        if _sha256_file(weights_path) != \
                manifest["weights"]["file_sha256"]:
            reject(RK_E_SCHEMA, "SmolLM2 weight package hash mismatch",
                   stage="smollm")
        self.weights = np.load(weights_path, allow_pickle=False)
        # The weights are immutable after the package-hash validation above.
        # The legacy access path indexed a compressed NPZ member and narrowed
        # it on every decode step; the candidate caches the identical FP16
        # arrays for the runner lifetime.  Identity/values are unchanged.
        if self.cache_host_weights:
            self._cached_embed = _f16(np.asarray(self.weights["embed"]))
        self._arrays: dict[str, np.ndarray] = {}
        self._plans: dict[str, Plan] = {}
        self._calls: dict[str, PlanCall] = {}
        self._persistent_action_ids: dict[tuple[str, int, int], int] = {}
        self._attn_b_inputs: dict[str, np.ndarray] = {}
        self._kv_state_ready: dict[tuple[int, int], bool] = {}
        self._load_plans(manifest["plans"])
        self.k_cache = np.zeros(
            (self.config.num_hidden_layers, self.config.num_key_value_heads,
             self.capacity, self.config.head_dim), dtype=np.float16)
        self.v_cache = np.zeros_like(self.k_cache)
        self.valid_len = 0
        self.reset_count = 0
        # Optional diagnostic capture.  The production path leaves this None,
        # so no extra arrays are retained during generation.
        self.capture_layers: set[int] | None = None
        self.captured: dict[str, np.ndarray] = {}
        self.load_us = (time.perf_counter() - load_t0) * 1e6
        self.stats.load_us += float(self.load_us)
        self.stats.load_plan_call_us += float(self._load_plan_call_us)
        for call in self._calls.values():
            self.stats.load_shared_constant_actions += int(
                call.shared_constant_actions)
            self.stats.load_shared_constant_created += int(
                call.shared_constant_created)
            self.stats.load_shared_constant_reused += int(
                call.shared_constant_reused)
            self.stats.load_shared_constant_bytes += int(
                call.shared_constant_bytes)

    # ------------------------------------------------------------------
    # Package and plan validation
    # ------------------------------------------------------------------
    def _weight_for_plan(self, key: str) -> np.ndarray:
        c = self.config
        parts = key.split(".")
        if key.startswith("layer.") and len(parts) >= 4:
            layer = int(parts[1])
            role = parts[2]
            w = self.weights[f"layer.{layer}.{role}"]
            expected = {
                "qkv": (c.num_attention_heads * c.head_dim +
                        2 * c.num_key_value_heads * c.head_dim, c.hidden_size),
                "o": (c.hidden_size, c.hidden_size),
                "gate_up": (2 * c.intermediate_size, c.hidden_size),
                "down": (c.hidden_size, c.intermediate_size),
            }[role]
            if tuple(w.shape) != expected:
                reject(RK_E_SHAPE, f"weight {key} has wrong shape",
                       stage="smollm", actual=list(w.shape),
                       required=list(expected))
            cache_key = f"w.{layer}.{role}"
            if cache_key not in self._arrays:
                # RKPlan B is [K,N] row-major; HF stores Linear as [N,K].
                self._arrays[cache_key] = _f16(np.asarray(w).T)
            return self._arrays[cache_key]
        if key.startswith("lm_head."):
            shard = int(parts[-1])
            shards = int(self.manifest["lm_head_shards"])
            if shard < 0 or shard >= shards:
                reject(RK_E_SHAPE, "lm_head shard index out of range",
                       stage="smollm")
            if "embedding_transposed" not in self._arrays:
                self._arrays["embedding_transposed"] = _f16(
                    np.asarray(self.weights["embed"]).T)
            base = self._arrays["embedding_transposed"]
            width = (c.vocab_size + shards - 1) // shards
            start, stop = shard * width, min(c.vocab_size, (shard + 1) * width)
            cache_key = f"lm_head_shard.{shard}"
            if cache_key not in self._arrays:
                # Decode and prefill profiles consume the same immutable shard.
                # Keep one Host copy instead of duplicating it per plan name.
                self._arrays[cache_key] = _f16(base[:, start:stop])
            return self._arrays[cache_key]
        reject(RK_E_SCHEMA, f"no constant mapping for plan {key!r}",
               stage="smollm")

    def _load_plans(self, entries: dict[str, Any]) -> None:
        needed: list[str] = []
        for layer in self.layer_indices:
            for role in ("qkv", "o", "gate_up", "down"):
                for profile in (DECODE, PREFILL):
                    needed.append(f"layer.{layer}.{role}.{profile}")
        for profile in (DECODE, PREFILL):
            for shard in range(int(self.manifest["lm_head_shards"])):
                needed.append(f"lm_head.{profile}.{shard:02d}")
        needed.extend(("attn.qk.decode", "attn.pv.decode",
                       "attn.qk.prefill", "attn.pv.prefill"))
        unknown_overrides = sorted(set(self.plan_overrides) - set(needed))
        if unknown_overrides:
            reject(RK_E_UNSUPPORTED,
                   "schedule policy names unknown SmolLM2 plan keys",
                   stage="smollm", actual=unknown_overrides)
        for key in needed:
            if key not in entries:
                reject(RK_E_SCHEMA, f"package manifest lacks plan {key!r}",
                       stage="smollm")
            info = entries[key]
            override = self.plan_overrides.get(key)
            if override is not None:
                if not isinstance(override, dict) or "plan" not in override:
                    reject(RK_E_INVALID,
                           f"schedule override for {key!r} must contain a plan",
                           stage="smollm")
                plan = Plan.from_dict(override["plan"])
                expected_hash = override.get("plan_hash")
                if expected_hash is not None and plan.plan_hash != expected_hash:
                    reject(RK_E_SCHEMA,
                           f"schedule override plan hash mismatch for {key!r}",
                           stage="smollm")
                expected_task_count = override.get("task_count")
                if expected_task_count is not None and int(
                        plan.accounting.task_count) != int(expected_task_count):
                    reject(RK_E_SCHEMA,
                           f"schedule override task count mismatch for {key!r}",
                           stage="smollm")
            else:
                path = self.package_dir / info["path"]
                plan = Plan.from_dict(json.loads(path.read_text()))
                if plan.plan_hash != info["plan_hash"]:
                    reject(RK_E_SCHEMA, f"plan hash mismatch for {key!r}",
                           stage="smollm")
                if plan.accounting.task_count != int(info["task_count"]):
                    reject(RK_E_SCHEMA, f"task count mismatch for {key!r}",
                           stage="smollm")
            if plan.profile_hash != self.manifest["compiler"]["profile_hash"]:
                reject(RK_E_SCHEMA, f"plan profile mismatch for {key!r}",
                       stage="smollm")
            templates = [a.template_id for a in plan.actions
                         if a.kind == "npu_task"]
            if not templates or any(not t.startswith("co.mm") for t in templates):
                reject(RK_E_UNSUPPORTED,
                       f"plan {key!r} does not contain only matrix NPU tasks",
                       stage="smollm", actual=templates)
            constants: dict[str, np.ndarray] = {}
            if plan.constants:
                if [c.name for c in plan.constants] != ["B"]:
                    reject(RK_E_SCHEMA,
                           f"unexpected constant set for plan {key!r}",
                           stage="smollm",
                           actual=[c.name for c in plan.constants])
                constants["B"] = self._weight_for_plan(key)
            self._plans[key] = plan
            try:
                t_call = time.perf_counter()
                self._calls[key] = PlanCall(
                    self.lib, plan, constants, fd=self.fd,
                    entry="attributed", reuse_bindings=True,
                    share_constants=self.share_constants,
                    profile=self.profile)
                self._load_plan_call_us += (
                    time.perf_counter() - t_call) * 1e6
            except BaseException as exc:
                self.close(ignore_errors=True)
                raise exc
        # Decode attention needs one persistent native-layout workspace per
        # layer and KV head.  Sharing one PlanCall across layers/heads made it
        # impossible to skip the KV upload safely.  The prefill QK/PV plans stay
        # shared; they run once and use ordinary full uploads.
        try:
            if self.persistent_kv:
                for layer in self.layer_indices:
                    for g in range(self.config.num_key_value_heads):
                        for kind in ("qk", "pv"):
                            template_key = f"attn.{kind}.decode"
                            template_plan = self._plans[template_key]
                            call_key = f"attn.{kind}.decode.{layer}.{g}"
                            call = PlanCall(
                                self.lib, template_plan, {}, fd=self.fd,
                                entry="attributed", reuse_bindings=True,
                                share_constants=False,
                                profile=self.profile)
                            b_binding = next(
                                b for b in template_plan.bindings
                                if b.name == "B")
                            if kind not in self._attn_b_inputs:
                                self._attn_b_inputs[kind] = np.zeros(
                                    tuple(b_binding.shape), dtype=np.float16)
                            uploads = [
                                a for a in template_plan.actions
                                if a.kind == "upload" and
                                int(a.attrs.get("source_binding", -1)) ==
                                b_binding.id]
                            if len(uploads) != 1:
                                call.close()
                                reject(RK_E_SCHEMA,
                                       f"{template_key} must have exactly one "
                                       "B upload action", stage="smollm",
                                       actual=len(uploads))
                            try:
                                call.set_persistent_upload(uploads[0].id)
                            except BaseException:
                                call.close()
                                raise
                            self._calls[call_key] = call
                            self._persistent_action_ids[(kind, layer, g)] = \
                                int(uploads[0].id)
                            self._kv_state_ready[(layer, g)] = False
                # The template decode calls are no longer on the execution path.
                for key in ("attn.qk.decode", "attn.pv.decode"):
                    template_call = self._calls.pop(key, None)
                    if template_call is not None:
                        template_call.close()
        except BaseException as exc:
            self.close(ignore_errors=True)
            raise exc
        self.plan_keys = tuple(needed) + tuple(
            k for k in self._calls if k.startswith("attn."))

    # ------------------------------------------------------------------
    # Metered Host action and RKPlan boundaries
    # ------------------------------------------------------------------
    def _act(self, action_id: str, inputs: dict[str, np.ndarray],
             attrs: dict[str, Any], *, nbytes: int | None = None
             ) -> dict[str, np.ndarray]:
        action = HOST_ACTIONS.get(action_id)
        if action is None:
            reject(RK_E_UNSUPPORTED, f"unregistered Host action {action_id!r}",
                   stage="smollm")
        if set(inputs) != set(action.input_names):
            reject(RK_E_SCHEMA,
                   f"Host action {action_id!r} input names mismatch",
                   stage="smollm", actual=sorted(inputs),
                   required=sorted(action.input_names))
        t0 = time.perf_counter()
        outputs = action.func(inputs, attrs)
        elapsed = (time.perf_counter() - t0) * 1e6
        if set(outputs) != set(action.output_names):
            reject(RK_E_SCHEMA,
                   f"Host action {action_id!r} output names mismatch",
                   stage="smollm", actual=sorted(outputs),
                   required=sorted(action.output_names))
        self.stats.add_host_action(
            action_id, "execution", elapsed,
            tuple(inputs.values()), tuple(outputs.values()), nbytes=nbytes)
        self._profile_add("host_action.us", elapsed)
        self._profile_add("host_action", 0.0)
        return outputs

    def _profile_add(self, name: str, elapsed_us: float,
                     *, count: int = 1) -> None:
        if not self.profile:
            return
        self._profile_us[name] = self._profile_us.get(name, 0.0) + float(
            elapsed_us)
        self._profile_count[name] = self._profile_count.get(name, 0) +             int(count)

    def _profile_reset(self) -> None:
        self._profile_us = {}
        self._profile_count = {}

    def _profile_report(self, *, decode_wall_us: float,
                        decode_forwards: int) -> dict[str, Any]:
        """Return a layered decode profile with explicit non-overlap rules.

        ``top_level`` sums to the runner's decode wall minus a residual.
        ``layer_ops`` is a subdivision of ``top_level.layer_total_us``.
        ``plan_node`` is a cross-cut over all PlanCall nodes and is not added
        to the top level.  ``c_runtime`` is a child of ``plan_node.call_us``.
        """
        if not getattr(self, "profile", False):
            return {}
        us = dict(getattr(self, "_profile_us", {}))
        counts = dict(getattr(self, "_profile_count", {}))
        top_names = ("embed", "layer_total", "final_norm", "lm_head_total",
                     "concat")
        top = {name: float(us.get("top." + name, 0.0))
               for name in top_names}
        top_sum = sum(top.values())
        layer_ops = {
            name: float(us.get("layer." + name, 0.0))
            for name in ("norm_input", "qkv_plan", "qkv_split", "rope",
                         "kv_append", "attention_total", "o_plan",
                         "residual_attn", "norm_post", "gate_up_plan",
                         "swiglu", "down_plan", "residual_ffn")
        }
        layer_ops_sum = sum(layer_ops.values())
        attention_ops = {
            name: float(us.get("attn." + name, 0.0))
            for name in ("qk_layout", "qk_plan", "softmax", "pv_plan",
                         "merge")
        }
        attention_ops_sum = sum(attention_ops.values())
        plan_node = {
            name: float(us.get("plan." + name, 0.0))
            for name in ("alloc", "call", "finite", "stats", "residual")
        }
        plan_call = {
            name: float(us.get("plan_call." + name, 0.0))
            for name in ("lock_wait", "input_prepare", "output_prepare",
                         "binding", "c_runtime", "inner_unattributed",
                         "wrapper_unattributed")
        }
        c_runtime = {
            name: float(us.get("c_runtime." + name, 0.0))
            for name in ("validate", "pack", "emit", "command",
                         "submit_alloc", "submit_wait", "output_sync",
                         "download", "guard", "publish", "alloc")
        }
        c_runtime_total = float(us.get("c_runtime.total", 0.0))
        c_runtime_children_sum = sum(c_runtime.values())
        counters = {
            "decode_forwards": int(decode_forwards),
            "plan_nodes": int(counts.get("plan.node", 0)),
            "host_actions": int(counts.get("host_action", 0)),
        }
        return {
            "enabled": True,
            "decode_wall_us": float(decode_wall_us),
            "decode_forwards": int(decode_forwards),
            "top_level": top,
            "top_level_sum_us": top_sum,
            "top_level_residual_us": max(
                0.0, float(decode_wall_us) - top_sum),
            "top_level_residual_fraction": (
                max(0.0, float(decode_wall_us) - top_sum) /
                float(decode_wall_us) if decode_wall_us else 0.0),
            "layer_total_us": top["layer_total"],
            "layer_ops": layer_ops,
            "layer_ops_sum_us": layer_ops_sum,
            "layer_residual_us": max(
                0.0, top["layer_total"] - layer_ops_sum),
            "attention_total_us": float(
                us.get("layer.attention_total", 0.0)),
            "attention_ops": attention_ops,
            "attention_ops_sum_us": attention_ops_sum,
            "attention_residual_us": max(
                0.0, float(us.get("layer.attention_total", 0.0)) -
                attention_ops_sum),
            "plan_node_subdivision": plan_node,
            "plan_node_sum_us": sum(plan_node.values()),
            "plan_node_residual_us": max(
                0.0, plan_node["call"] - sum(plan_call.values())),
            "plan_call_subdivision": plan_call,
            "plan_call_sum_us": sum(plan_call.values()),
            "c_runtime_total_us": c_runtime_total,
            "c_runtime_subdivision": c_runtime,
            "c_runtime_children_sum_us": c_runtime_children_sum,
            "c_runtime_residual_us": max(
                0.0, c_runtime_total - c_runtime_children_sum),
            "counters": counters,
            "host_action_us": float(us.get("host_action.us", 0.0)),
            "measurement_note": (
                "top_level=runner decode wall components; layer_ops is a "
                "subdivision of layer_total; plan_node is a cross-cut over "
                "all plans; c_runtime is a subdivision of plan_node.call; "
                "residuals are measured, not reassigned."),
        }

    def _run_plan(self, key: str, inputs: list[np.ndarray],
                  *, phase: str = "execution") -> list[np.ndarray]:
        call = self._calls.get(key)
        if call is None:
            reject(RK_E_SCHEMA, f"runner has no compiled plan {key!r}",
                   stage="smollm")
        t_node = time.perf_counter() if self.profile else 0.0
        t_alloc = t_node
        outputs = [np.empty(tuple(b.shape), dtype=np.float16)
                   for b in call.outputs]
        t_call = time.perf_counter() if self.profile else 0.0
        report, elapsed = call.run_into(inputs, outputs)
        t_finite = time.perf_counter() if self.profile else 0.0
        workspace_us = float(getattr(report, "total_us", 0.0)) or None
        self.stats.add_report(key, report, elapsed, phase=phase,
                              workspace_us=workspace_us)
        t_stats = time.perf_counter() if self.profile else 0.0
        if not all(np.isfinite(o).all() for o in outputs):
            reject(RK_E_NUMERIC_CONTRACT,
                   f"RKPlan {key!r} produced non-finite output",
                   stage="smollm")
        if self.profile:
            done = time.perf_counter()
            known = ((t_call - t_alloc) + (t_finite - t_call) +
                     (t_stats - t_finite) + (done - t_stats))
            self._profile_add("plan.node", (done - t_node) * 1e6)
            self._profile_add("plan.alloc", (t_call - t_alloc) * 1e6)
            self._profile_add("plan.call", (t_finite - t_call) * 1e6)
            self._profile_add("plan.finite", (t_stats - t_finite) * 1e6)
            self._profile_add("plan.stats", (done - t_stats) * 1e6)
            self._profile_add(
                "plan.residual",
                max(0.0, (done - t_node) * 1e6 - known * 1e6))
            if call.last_profile:
                for sub, value in call.last_profile.items():
                    self._profile_add(
                        "plan_call." + sub.removesuffix("_us"), value)
                if hasattr(report, "total_us"):
                    self._profile_add(
                        "c_runtime.total", float(report.total_us))
                for field in ("validate_us", "pack_us", "emit_us",
                              "command_us", "submit_alloc_us",
                              "submit_wait_us", "output_sync_us",
                              "download_us", "guard_us", "publish_us",
                              "alloc_us"):
                    if hasattr(report, field):
                        self._profile_add(
                            "c_runtime." + field.removesuffix("_us"),
                            float(getattr(report, field)))
        return outputs

    def _norm(self, x: np.ndarray, gamma_key: str) -> np.ndarray:
        if self.cache_host_weights:
            gamma = self._cached_norm_gamma.get(gamma_key)
            if gamma is None:
                gamma = _f16(np.asarray(self.weights[gamma_key]))
                self._cached_norm_gamma[gamma_key] = gamma
        else:
            gamma = _f16(self.weights[gamma_key])
        return self._act("rmsnorm.last_axis.v1",
                         {"x": _f16(x), "gamma": gamma},
                         {"eps": float(self.config.rms_norm_eps)})["y"]

    def _add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self._act("tensor.add.v1", {"a": _f16(a), "b": _f16(b)}, {})["y"]

    def _embed(self, ids: np.ndarray, *, bucket: int) -> np.ndarray:
        table = self._cached_embed if self.cache_host_weights else _f16(
            self.weights["embed"])
        return self._act(
            "smollm.embed_padded.v1",
            {"token_ids": np.ascontiguousarray(ids, dtype=np.int32),
             "table": table},
            {"bucket": int(bucket),
             "hidden": int(self.config.hidden_size)})["x"]

    def _rope(self, x: np.ndarray, positions: np.ndarray) -> np.ndarray:
        return self._act(
            "smollm.rope_half.v1",
            {"x": _f16(x), "positions": np.ascontiguousarray(
                positions, dtype=np.int32)},
            {"theta": float(self.config.rope_theta),
             "interleaved": False})["y"]

    def _split_qkv(self, qkv: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                    np.ndarray]:
        out = self._act(
            "smollm.qkv_split_gqa.v1", {"qkv": _f16(qkv)},
            {"q_heads": self.config.num_attention_heads,
             "kv_heads": self.config.num_key_value_heads,
             "head_dim": self.config.head_dim})
        return out["q"], out["k"], out["v"]

    def _append_kv(self, layer: int, k: np.ndarray, v: np.ndarray,
                   start: int) -> None:
        ck = self.k_cache[layer]
        cv = self.v_cache[layer]
        nbytes = 3 * (int(k.nbytes) + int(v.nbytes))
        self._act(
            "smollm.kv_append_inplace.v1",
            {"cache_k": ck, "cache_v": cv,
             "k": np.ascontiguousarray(k, dtype=np.float16),
             "v": np.ascontiguousarray(v, dtype=np.float16),
             "start": np.asarray([start], dtype=np.int32)},
            {"capacity": int(self.capacity)}, nbytes=nbytes)
        self._append_native_kv(layer, start, k, v)

    def _causal_softmax(self, scores: np.ndarray,
                        valid_lengths: np.ndarray) -> np.ndarray:
        return self._act(
            "smollm.causal_softmax.v1",
            {"scores": _f16(scores),
             "valid_lengths": np.ascontiguousarray(
                 valid_lengths, dtype=np.int32)},
            {"scale": 1.0 / math.sqrt(float(self.config.head_dim))})["probs"]

    def _qk_layout(self, q: np.ndarray) -> np.ndarray:
        return self._act(
            "smollm.qk_layout_gqa.v1", {"q": _f16(q)},
            {"q_heads": self.config.num_attention_heads,
             "kv_heads": self.config.num_key_value_heads,
             "head_dim": self.config.head_dim})["a"]

    def _merge_heads(self, contexts: np.ndarray, tokens: int) -> np.ndarray:
        return self._act(
            "smollm.merge_gqa_heads.v1",
            {"contexts": _f16(contexts)},
            {"tokens": int(tokens),
             "q_heads": self.config.num_attention_heads,
             "kv_heads": self.config.num_key_value_heads,
             "head_dim": self.config.head_dim})["merged"]

    def _swiglu(self, gate_up: np.ndarray) -> np.ndarray:
        return self._act("swiglu.packed_gate_up_act.v1",
                         {"gate_up": _f16(gate_up)}, {})["act"]

    def _concat_logits(self, shards: list[np.ndarray],
                       *, rows: int) -> np.ndarray:
        count = int(self.manifest["lm_head_shards"])
        if len(shards) != count:
            reject(RK_E_SHAPE, "lm_head shard count mismatch", stage="smollm",
                   actual=len(shards), required=count)
        inputs = {f"shard{i}": shard for i, shard in enumerate(shards)}
        return self._act(
            "smollm.concat_logits.v1", inputs,
            {"rows": int(rows), "vocab_size": self.config.vocab_size,
             "shards": count})["logits"]

    # ------------------------------------------------------------------
    # Persistent KV native-layout state
    # ------------------------------------------------------------------
    def _kv_state(self, kind: str, layer: int, group: int) -> PlanCall:
        key = f"attn.{kind}.decode.{layer}.{group}"
        call = self._calls.get(key)
        if call is None:
            reject(RK_E_SCHEMA, f"runner has no persistent {kind} state {key!r}",
                   stage="smollm")
        return call

    def _state_update(self, call: PlanCall, action_id: int,
                      array: np.ndarray, *, src_row: int, src_col: int,
                      dst_row: int, dst_col: int, rows: int, cols: int,
                      transpose_source: bool, init: bool) -> None:
        t0 = time.perf_counter()
        call.update_upload_region(
            action_id, array, src_row=src_row, src_col=src_col,
            dst_row=dst_row, dst_col=dst_col, rows=rows, cols=cols,
            transpose_source=transpose_source)
        elapsed = (time.perf_counter() - t0) * 1e6
        nbytes = int(rows) * int(cols) * 2
        if init:
            self.stats.kv_state_init_calls += 1
            self.stats.kv_state_init_bytes += nbytes
            self.stats.kv_state_init_us += elapsed
        else:
            self.stats.kv_state_append_calls += 1
            self.stats.kv_state_append_bytes += nbytes
            self.stats.kv_state_append_us += elapsed

    def _ensure_native_kv(self, layer: int, group: int) -> None:
        """Initialize one layer/head's persistent packed QK/PV B destinations.

        The first decode after reset (or after a prefill) writes the complete
        capacity once from the Host cache.  All later decode steps only write
        the newly appended token stripe.  Different layers and heads own
        different PlanCall workspaces and can never overwrite each other.
        """
        key = (layer, group)
        if self._kv_state_ready.get(key, False):
            return
        c = self.config
        qk_id = self._persistent_action_ids[("qk", layer, group)]
        pv_id = self._persistent_action_ids[("pv", layer, group)]
        # QK B destination is logically [capacity, head_dim]; the Host K cache
        # is already [capacity, head_dim] row-major, so no transpose is needed.
        self._state_update(
            self._kv_state("qk", layer, group), qk_id,
            self.k_cache[layer, group], src_row=0, src_col=0,
            dst_row=0, dst_col=0, rows=self.capacity,
            cols=c.head_dim, transpose_source=False, init=True)
        # PV B destination is logically [head_dim, capacity]: each destination
        # row is one output channel and each column is a KV position.
        self._state_update(
            self._kv_state("pv", layer, group), pv_id,
            self.v_cache[layer, group], src_row=0, src_col=0,
            dst_row=0, dst_col=0, rows=c.head_dim,
            cols=self.capacity, transpose_source=True, init=True)
        self._kv_state_ready[key] = True

    def _append_native_kv(self, layer: int, start: int,
                          k: np.ndarray, v: np.ndarray) -> None:
        """Append only new token stripes to already-initialized persistent KV."""
        if not self.persistent_kv:
            return
        c = self.config
        k = np.asarray(k)
        v = np.asarray(v)
        for t in range(k.shape[0]):
            position = int(start) + t
            for g in range(c.num_key_value_heads):
                if not self._kv_state_ready.get((layer, g), False):
                    # The first decode path performs a full initialization
                    # that already includes this token.
                    continue
                kt = np.ascontiguousarray(k[t, g, :], dtype=np.float16)
                vt = np.ascontiguousarray(v[t, g, :], dtype=np.float16)
                self._state_update(
                    self._kv_state("qk", layer, g),
                    self._persistent_action_ids[("qk", layer, g)], kt,
                    src_row=0, src_col=0, dst_row=position, dst_col=0,
                    rows=1, cols=c.head_dim, transpose_source=False,
                    init=False)
                self._state_update(
                    self._kv_state("pv", layer, g),
                    self._persistent_action_ids[("pv", layer, g)], vt,
                    src_row=0, src_col=0, dst_row=0, dst_col=position,
                    rows=c.head_dim, cols=1, transpose_source=True,
                    init=False)

    # ------------------------------------------------------------------
    # Layer computation
    # ------------------------------------------------------------------
    def _attention(self, layer: int, q: np.ndarray, k: np.ndarray,
                   v: np.ndarray, *, tokens: int, profile: str,
                   start: int, valid_lengths: np.ndarray) -> np.ndarray:
        c = self.config
        kh = c.num_key_value_heads
        if profile == DECODE:
            t_attn = time.perf_counter() if self.profile else 0.0
            t0 = t_attn
            a = self._qk_layout(q)
            if self.profile:
                self._profile_add(
                    "attn.qk_layout", (time.perf_counter() - t0) * 1e6)
            contexts = []
            for g in range(kh):
                if self.persistent_kv:
                    self._ensure_native_kv(layer, g)
                    qk_key = f"attn.qk.decode.{layer}.{g}"
                    pv_key = f"attn.pv.decode.{layer}.{g}"
                    qk_b = self._attn_b_inputs["qk"]
                    pv_b = self._attn_b_inputs["pv"]
                else:
                    qk_key = "attn.qk.decode"
                    pv_key = "attn.pv.decode"
                    # Historical A/B control: materialise the transposed K
                    # cache and upload both dynamic operands every step.
                    qk_b = _f16(self.k_cache[layer, g].T)
                    pv_b = self.v_cache[layer, g]
                t0 = time.perf_counter() if self.profile else 0.0
                scores = self._run_plan(qk_key, [a[g], qk_b])[0]
                if self.profile:
                    self._profile_add(
                        "attn.qk_plan", (time.perf_counter() - t0) * 1e6)
                # Every query head in each KV group has the same causal
                # prefix length during decode, so the same per-row vector is
                # valid for each of the three independent group plans.
                t0 = time.perf_counter() if self.profile else 0.0
                probs = self._causal_softmax(scores, valid_lengths)
                if self.profile:
                    self._profile_add(
                        "attn.softmax", (time.perf_counter() - t0) * 1e6)
                t0 = time.perf_counter() if self.profile else 0.0
                contexts.append(self._run_plan(
                    pv_key, [probs, pv_b])[0])
                if self.profile:
                    self._profile_add(
                        "attn.pv_plan", (time.perf_counter() - t0) * 1e6)
            t0 = time.perf_counter() if self.profile else 0.0
            merged = self._merge_heads(np.stack(contexts, axis=0), tokens)
            if self.profile:
                self._profile_add(
                    "attn.merge", (time.perf_counter() - t0) * 1e6)
                self._profile_add(
                    "layer.attention_total", (time.perf_counter() - t_attn) * 1e6)
            return merged
        if profile != PREFILL:
            reject(RK_E_INVALID, f"unknown attention profile {profile!r}",
                   stage="smollm")
        a = self._qk_layout(q)
        contexts = []
        for g in range(kh):
            kt = _f16(self.k_cache[layer, g].T)
            scores = self._run_plan(
                "attn.qk.prefill", [a[g], kt])[0]
            probs = self._causal_softmax(scores, valid_lengths)
            contexts.append(self._run_plan(
                "attn.pv.prefill", [probs, self.v_cache[layer, g]])[0])
        return self._merge_heads(np.stack(contexts, axis=0), tokens)

    def _layer_prefill(self, layer: int, x: np.ndarray,
                       tokens: int) -> np.ndarray:
        c = self.config
        bucket = self.prefill_bucket
        capture = self.capture_layers is not None and layer in self.capture_layers
        h = self._norm(x, f"layer.{layer}.input_norm")
        qkv = self._run_plan(
            f"layer.{layer}.qkv.{PREFILL}", [h])[0]
        q, k, v = self._split_qkv(qkv)
        positions = np.arange(bucket, dtype=np.int32)
        q, k = self._rope(q, positions), self._rope(k, positions)
        self._append_kv(layer, k[:tokens], v[:tokens], 0)
        per = c.num_attention_heads // c.num_key_value_heads
        # qk_layout orders each group as [q_local * bucket + token].
        valid_lengths = np.zeros(per * bucket, dtype=np.int32)
        valid_lengths.reshape(per, bucket)[:, :tokens] = \
            np.arange(1, tokens + 1, dtype=np.int32)[None, :]
        attn = self._attention(layer, q, k, v, tokens=bucket,
                               profile=PREFILL, start=0,
                               valid_lengths=valid_lengths)
        o = self._run_plan(f"layer.{layer}.o.{PREFILL}", [attn])[0]
        x_attn = self._add(x, o)
        h2 = self._norm(x_attn, f"layer.{layer}.post_norm")
        gate_up = self._run_plan(
            f"layer.{layer}.gate_up.{PREFILL}", [h2])[0]
        act = self._swiglu(gate_up)
        down = self._run_plan(f"layer.{layer}.down.{PREFILL}", [act])[0]
        x = self._add(x_attn, down)
        if capture:
            prefix = f"layer.{layer}."
            self.captured.update({
                prefix + "input_norm": h[:tokens].copy(),
                prefix + "qkv": qkv[:tokens].copy(),
                prefix + "q_rope": q[:tokens].copy(),
                prefix + "k_rope": k[:tokens].copy(),
                prefix + "v": v[:tokens].copy(),
                prefix + "attn_context": attn[:tokens].copy(),
                prefix + "o_proj": o[:tokens].copy(),
                prefix + "resid_attn": x_attn[:tokens].copy(),
                prefix + "post_norm": h2[:tokens].copy(),
                prefix + "gate_up": gate_up[:tokens].copy(),
                prefix + "swiglu": act[:tokens].copy(),
                prefix + "down_proj": down[:tokens].copy(),
                prefix + "resid_ffn": x[:tokens].copy(),
            })
        return x

    def _layer_decode(self, layer: int, x: np.ndarray,
                      position: int) -> np.ndarray:
        c = self.config
        t0 = time.perf_counter() if self.profile else 0.0
        h = self._norm(x, f"layer.{layer}.input_norm")
        if self.profile:
            self._profile_add(
                "layer.norm_input", (time.perf_counter() - t0) * 1e6)
        t0 = time.perf_counter() if self.profile else 0.0
        qkv = self._run_plan(f"layer.{layer}.qkv.{DECODE}", [h])[0]
        if self.profile:
            self._profile_add(
                "layer.qkv_plan", (time.perf_counter() - t0) * 1e6)
        t0 = time.perf_counter() if self.profile else 0.0
        q, k, v = self._split_qkv(qkv)
        if self.profile:
            self._profile_add(
                "layer.qkv_split", (time.perf_counter() - t0) * 1e6)
        pos = np.asarray([position], dtype=np.int32)
        t0 = time.perf_counter() if self.profile else 0.0
        q, k = self._rope(q, pos), self._rope(k, pos)
        if self.profile:
            self._profile_add(
                "layer.rope", (time.perf_counter() - t0) * 1e6)
        t0 = time.perf_counter() if self.profile else 0.0
        self._append_kv(layer, k, v, position)
        if self.profile:
            self._profile_add(
                "layer.kv_append", (time.perf_counter() - t0) * 1e6)
        per = c.num_attention_heads // c.num_key_value_heads
        # Decode calls one QK/PV plan per KV group.  The plan has three rows
        # (the query heads sharing that KV head); softmax sees one valid length
        # per row.
        valid_lengths = np.full(per, position + 1, dtype=np.int32)
        attn = self._attention(layer, q, k, v, tokens=1,
                               profile=DECODE, start=position,
                               valid_lengths=valid_lengths)
        t0 = time.perf_counter() if self.profile else 0.0
        o = self._run_plan(f"layer.{layer}.o.{DECODE}", [attn])[0]
        if self.profile:
            self._profile_add(
                "layer.o_plan", (time.perf_counter() - t0) * 1e6)
        t0 = time.perf_counter() if self.profile else 0.0
        x = self._add(x, o)
        if self.profile:
            self._profile_add(
                "layer.residual_attn", (time.perf_counter() - t0) * 1e6)
        t0 = time.perf_counter() if self.profile else 0.0
        h2 = self._norm(x, f"layer.{layer}.post_norm")
        if self.profile:
            self._profile_add(
                "layer.norm_post", (time.perf_counter() - t0) * 1e6)
        t0 = time.perf_counter() if self.profile else 0.0
        gate_up = self._run_plan(
            f"layer.{layer}.gate_up.{DECODE}", [h2])[0]
        if self.profile:
            self._profile_add(
                "layer.gate_up_plan", (time.perf_counter() - t0) * 1e6)
        t0 = time.perf_counter() if self.profile else 0.0
        act = self._swiglu(gate_up)
        if self.profile:
            self._profile_add(
                "layer.swiglu", (time.perf_counter() - t0) * 1e6)
        t0 = time.perf_counter() if self.profile else 0.0
        down = self._run_plan(f"layer.{layer}.down.{DECODE}", [act])[0]
        if self.profile:
            self._profile_add(
                "layer.down_plan", (time.perf_counter() - t0) * 1e6)
        t0 = time.perf_counter() if self.profile else 0.0
        result = self._add(x, down)
        if self.profile:
            self._profile_add(
                "layer.residual_ffn", (time.perf_counter() - t0) * 1e6)
        return result

    # ------------------------------------------------------------------
    # Public request API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        # Clear host state and every initialized native persistent destination.
        # The next decode would also overwrite the complete configured capacity,
        # but an explicit clear makes reset semantics independent of that
        # initialization shortcut and prevents stale bytes from a prior request
        # from being observable if capacity/layout paths diverge later.
        for (layer, group), ready in list(self._kv_state_ready.items()):
            if not ready:
                continue
            for kind in ("qk", "pv"):
                call = self._kv_state(kind, layer, group)
                action_id = self._persistent_action_ids[(kind, layer, group)]
                t0 = time.perf_counter()
                call.clear_upload_destination(action_id)
                elapsed = (time.perf_counter() - t0) * 1e6
                self.stats.kv_state_clear_calls += 1
                self.stats.kv_state_clear_bytes += int(
                    call.plan.views[next(
                        a for a in call.plan.actions
                        if a.id == action_id).outputs[0]].tensor.bytes)
                self.stats.kv_state_clear_us += elapsed
            self._kv_state_ready[(layer, group)] = False
        self.k_cache.fill(0)
        self.v_cache.fill(0)
        self.valid_len = 0
        self.reset_count += 1

    def _ensure_capacity(self, needed: int) -> None:
        if needed < 0 or needed > self.capacity:
            reject(RK_E_SHAPE,
                   "request exceeds persistent KV cache capacity",
                   stage="smollm", actual=needed, required=f"<= {self.capacity}")

    def prefill(self, token_ids: list[int] | np.ndarray, *,
                return_all_logits: bool = False) -> dict[str, Any]:
        if self.valid_len != 0:
            reject(RK_E_INVALID,
                   "prefill requires a reset request; call reset() first",
                   stage="smollm", actual=self.valid_len)
        ids = np.ascontiguousarray(token_ids, dtype=np.int32)
        if ids.ndim != 1 or ids.size < 1:
            reject(RK_E_SHAPE, "prefill token_ids must be a non-empty vector",
                   stage="smollm")
        self._ensure_capacity(int(ids.size))
        t0 = time.perf_counter()
        if ids.size == 1:
            x = self._embed(ids, bucket=1)
            for layer in self.layer_indices:
                x = self._layer_decode(layer, x, position=0)
            self.valid_len = 1
            normed = self._norm(x, "norm")
            shards = [self._run_plan(
                f"lm_head.decode.{j:02d}", [normed])[0]
                for j in range(int(self.manifest["lm_head_shards"]))]
            logits = self._concat_logits(shards, rows=1)
            all_logits = logits if return_all_logits else None
        else:
            x = self._embed(ids, bucket=self.prefill_bucket)
            for layer in self.layer_indices:
                x = self._layer_prefill(layer, x, int(ids.size))
            self.valid_len = int(ids.size)
            normed = self._norm(x, "norm")
            if self.capture_layers is not None:
                self.captured["final_norm"] = normed[:ids.size].copy()
            if return_all_logits:
                shards = [self._run_plan(
                    f"lm_head.prefill.{j:02d}", [normed])[0]
                    for j in range(int(self.manifest["lm_head_shards"]))]
                all_logits = self._concat_logits(
                    shards, rows=self.prefill_bucket)
                logits = all_logits[ids.size - 1:ids.size]
            else:
                last = np.ascontiguousarray(normed[ids.size - 1:ids.size])
                shards = [self._run_plan(
                    f"lm_head.decode.{j:02d}", [last])[0]
                    for j in range(int(self.manifest["lm_head_shards"]))]
                logits = self._concat_logits(shards, rows=1)
                all_logits = None
        elapsed_us = (time.perf_counter() - t0) * 1e6
        return {"logits": logits, "all_logits": all_logits,
                "valid_len": self.valid_len, "elapsed_us": elapsed_us}

    def decode_one(self, token_id: int) -> dict[str, Any]:
        if not 0 <= int(token_id) < self.config.vocab_size:
            reject(RK_E_SHAPE, "decode token id is outside vocabulary",
                   stage="smollm", actual=int(token_id))
        self._ensure_capacity(self.valid_len + 1)
        if self.valid_len >= self.capacity:
            reject(RK_E_SHAPE, "decode would exceed persistent KV cache capacity",
                   stage="smollm", actual=self.valid_len + 1,
                   required=f"<= {self.capacity}")
        t0 = time.perf_counter()
        t_embed = time.perf_counter() if self.profile else 0.0
        x = self._embed(np.asarray([int(token_id)], dtype=np.int32), bucket=1)
        if self.profile:
            self._profile_add(
                "top.embed", (time.perf_counter() - t_embed) * 1e6)
        for layer in self.layer_indices:
            t_layer = time.perf_counter() if self.profile else 0.0
            x = self._layer_decode(layer, x, position=self.valid_len)
            if self.profile:
                self._profile_add(
                    "top.layer_total", (time.perf_counter() - t_layer) * 1e6)
        self.valid_len += 1
        t_norm = time.perf_counter() if self.profile else 0.0
        normed = self._norm(x, "norm")
        if self.profile:
            self._profile_add(
                "top.final_norm", (time.perf_counter() - t_norm) * 1e6)
        t_lm = time.perf_counter() if self.profile else 0.0
        shards = [self._run_plan(f"lm_head.decode.{j:02d}", [normed])[0]
                  for j in range(int(self.manifest["lm_head_shards"]))]
        if self.profile:
            self._profile_add(
                "top.lm_head_total", (time.perf_counter() - t_lm) * 1e6)
        t_concat = time.perf_counter() if self.profile else 0.0
        logits = self._concat_logits(shards, rows=1)
        if self.profile:
            self._profile_add(
                "top.concat", (time.perf_counter() - t_concat) * 1e6)
        elapsed_us = (time.perf_counter() - t0) * 1e6
        return {"logits": logits, "valid_len": self.valid_len,
                "elapsed_us": elapsed_us}

    def capture_prefill(self, token_ids: list[int] | np.ndarray, *,
                        layers: tuple[int, ...],
                        return_all_logits: bool = True) -> dict[str, Any]:
        """Diagnostic prefill that retains selected intermediate tensors.

        The captured values are FP16 exactly as consumed/produced at the Host
        boundary.  This path is never used by the production generation loop.
        """
        self.reset()
        self.captured = {}
        self.capture_layers = set(int(v) for v in layers)
        try:
            result = self.prefill(token_ids,
                                  return_all_logits=return_all_logits)
        finally:
            self.capture_layers = None
        result["captured"] = {
            name: np.ascontiguousarray(value)
            for name, value in sorted(self.captured.items())}
        return result

    def teacher_forcing_logits(self, token_ids: list[int] | np.ndarray
                               ) -> dict[str, Any]:
        """Run one fresh request and return logits for every supplied position."""
        self.reset()
        result = self.prefill(token_ids, return_all_logits=True)
        if result["all_logits"] is None:
            reject(RK_E_INVALID, "teacher forcing requires a multi-token prefix",
                   stage="smollm")
        n = len(token_ids)
        return {"logits": result["all_logits"][:n],
                "valid_len": self.valid_len,
                "elapsed_us": result["elapsed_us"]}

    def _stats_snapshot(self) -> dict[str, float | int]:
        return {name: getattr(self.stats, name) for name in _STAGE_STAT_FIELDS}

    @staticmethod
    def _stats_delta(before: dict[str, float | int],
                     after: dict[str, float | int]) -> dict[str, float | int]:
        return {name: after[name] - before[name] for name in _STAGE_STAT_FIELDS}

    @staticmethod
    def _stats_sum(rows: list[dict[str, float | int]]
                   ) -> dict[str, float | int]:
        out: dict[str, float | int] = {}
        for name in _STAGE_STAT_FIELDS:
            values = [row[name] for row in rows]
            out[name] = sum(values) if values else 0
        return out

    def generate(self, token_ids: list[int] | np.ndarray, *,
                 max_new_tokens: int, return_all_logits: bool = False,
                 stop_on_eos: bool = True) -> GenerationResult:
        if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            reject(RK_E_INVALID, "max_new_tokens must be >= 1", stage="smollm")
        ids = [int(v) for v in np.asarray(token_ids).reshape(-1)]
        self._ensure_capacity(len(ids) + max_new_tokens)
        self._profile_reset()
        wall_t0 = time.perf_counter()

        reset_before = self._stats_snapshot()
        reset_t0 = time.perf_counter()
        self.reset()
        reset_us = (time.perf_counter() - reset_t0) * 1e6
        reset_stats = self._stats_delta(reset_before, self._stats_snapshot())

        prefill_before = self._stats_snapshot()
        prompt = self.prefill(ids, return_all_logits=return_all_logits)
        prefill_stats = self._stats_delta(prefill_before,
                                          self._stats_snapshot())

        generated: list[int] = []
        steps: list[dict[str, Any]] = []
        decode_rows: list[dict[str, float | int]] = []
        sample_rows: list[dict[str, float | int]] = []
        t_decode = 0.0
        sampling_us = 0.0
        first_sample_us = 0.0
        current_logits = prompt["logits"]
        stopped_by_eos = False
        stop_reason = "max_new_tokens"
        for step_index in range(max_new_tokens):
            sample_before = self._stats_snapshot()
            sample_t0 = time.perf_counter()
            token = int(self._act(
                "smollm.greedy_sample.v1", {"logits": current_logits},
                {"vocab_size": self.config.vocab_size})["token_ids"][0])
            sample_us = (time.perf_counter() - sample_t0) * 1e6
            sampling_us += sample_us
            if step_index == 0:
                first_sample_us = sample_us
            sample_rows.append(self._stats_delta(
                sample_before, self._stats_snapshot()))
            generated.append(token)
            if stop_on_eos and token == self.config.eos_token_id:
                stopped_by_eos = True
                stop_reason = "eos"
                break
            if len(ids) + len(generated) >= self.capacity:
                stop_reason = "capacity"
                break
            # The final sampled token has no consumer.  Do not execute an
            # extra decode forward merely to produce an unused logits row.
            if step_index + 1 >= max_new_tokens:
                stop_reason = "max_new_tokens"
                break
            decode_before = self._stats_snapshot()
            decode_t0 = time.perf_counter()
            step = self.decode_one(token)
            step_us = (time.perf_counter() - decode_t0) * 1e6
            decode_rows.append(self._stats_delta(
                decode_before, self._stats_snapshot()))
            t_decode += step_us
            steps.append({"token_id": token,
                          "step_index": step_index,
                          "valid_len": step["valid_len"],
                          "wall_us": float(step_us),
                          "elapsed_us": float(step_us),
                          "stats_delta": decode_rows[-1]})
            current_logits = step["logits"]
        output_tokens = len(generated)
        decode_forwards = len(decode_rows)
        if decode_forwards != max(0, output_tokens - 1):
            reject(RK_E_INVALID,
                   "generation forward count does not match output tokens",
                   stage="smollm",
                   actual={"output_tokens": output_tokens,
                           "decode_forwards": decode_forwards})
        wall_us = (time.perf_counter() - wall_t0) * 1e6
        decode_stats = self._stats_sum(decode_rows)
        sampling_stats = self._stats_sum(sample_rows)
        accounted = reset_us + float(prompt["elapsed_us"]) + sampling_us + \
            t_decode
        stages = {
            "reset": {"wall_us": reset_us, "stats_delta": reset_stats},
            "prefill": {"wall_us": float(prompt["elapsed_us"]),
                        "stats_delta": prefill_stats},
            "sampling": {"wall_us": sampling_us,
                         "calls": len(sample_rows),
                         "stats_delta": sampling_stats},
            "decode": {"wall_us": t_decode,
                       "forward_count": decode_forwards,
                       "stats_delta": decode_stats},
            "generate_overhead_us": max(0.0, wall_us - accounted),
            "generate_wall_us": wall_us,
        }
        return GenerationResult(
            prompt_ids=ids, generated_ids=generated,
            stopped_by_eos=stopped_by_eos, stop_reason=stop_reason,
            prefill_us=float(prompt["elapsed_us"]), decode_us=t_decode,
            steps=steps, reset_us=reset_us, sampling_us=sampling_us,
            first_sample_us=first_sample_us,
            ttft_us=reset_us + float(prompt["elapsed_us"]) + first_sample_us,
            generate_wall_us=wall_us,
            decode_forwards=decode_forwards, output_tokens=output_tokens,
            stages=stages,
            profile=self._profile_report(
                decode_wall_us=t_decode, decode_forwards=decode_forwards))

    def close(self, *, ignore_errors: bool = False) -> None:
        if self._closed:
            return
        errors: list[BaseException] = []
        for key, call in reversed(list(self._calls.items())):
            try:
                call.close()
            except BaseException as exc:
                errors.append(exc)
        self._calls.clear()
        if errors and not ignore_errors:
            raise errors[0]
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            self.close(ignore_errors=exc is not None)
        except BaseException as cleanup:
            if exc is not None:
                exc.add_note("SmolLM2Runner cleanup also failed: " +
                             repr(cleanup))
                return False
            raise
        return False
