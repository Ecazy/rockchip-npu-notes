# SPDX-License-Identifier: GPL-3.0-or-later
"""Versioned TargetProfile and capability selection.

Only ``enabled`` capabilities are eligible for default lowering.  The manifest is
data, not a claim: an enabled entry must still have a matching lowering emitter and
the host tests must exercise it.  Existing experimental entries are never selected
implicitly.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import (RK_E_BUDGET, RK_E_SCHEMA, RK_E_TARGET_MISMATCH,
                     RK_E_UNSUPPORTED, reject)
from .serialize import load_json_safe, sha256_json

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE_PATH = PACKAGE_ROOT / "capabilities" / "rk3588_vendor_co_v1.json"


@dataclass(frozen=True)
class Capability:
    id: str
    op: str
    emitter: str
    target_profile: str
    status: str
    inputs: tuple[str, ...]
    output: str
    numeric: str
    alias: str
    chain_recipe: str
    evidence: str
    validation_domain: str = ""
    uncovered_risks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"enabled", "experimental", "unsupported"}:
            reject(RK_E_SCHEMA, f"capability {self.id} has unknown status {self.status!r}",
                   stage="capabilities")
        if not self.id or not self.emitter or not self.target_profile:
            reject(RK_E_SCHEMA, "capability id/emitter/profile must be non-empty",
                   stage="capabilities")

    @classmethod
    def from_dict(cls, obj: Any) -> "Capability":
        if not isinstance(obj, dict):
            reject(RK_E_SCHEMA, "capability must be an object", stage="capabilities")
        required = {"id", "op", "emitter", "target_profile", "status", "inputs",
                    "output", "numeric", "alias", "chain_recipe", "evidence"}
        optional = {"validation_domain", "uncovered_risks"}
        if set(obj) - (required | optional) or required - set(obj):
            reject(RK_E_SCHEMA, "capability fields are missing or unknown",
                   stage="capabilities")
        return cls(
            id=obj["id"], op=obj["op"], emitter=obj["emitter"],
            target_profile=obj["target_profile"], status=obj["status"],
            inputs=tuple(obj["inputs"]), output=obj["output"],
            numeric=obj["numeric"], alias=obj["alias"],
            chain_recipe=obj["chain_recipe"], evidence=obj["evidence"],
            validation_domain=obj.get("validation_domain", ""),
            uncovered_risks=tuple(obj.get("uncovered_risks", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "op": self.op,
            "emitter": self.emitter,
            "target_profile": self.target_profile,
            "status": self.status,
            "inputs": list(self.inputs),
            "output": self.output,
            "numeric": self.numeric,
            "alias": self.alias,
            "chain_recipe": self.chain_recipe,
            "evidence": self.evidence,
            "validation_domain": self.validation_domain,
            "uncovered_risks": list(self.uncovered_risks),
        }


@dataclass(frozen=True)
class TargetProfile:
    schema_version: int
    chip_id: str
    driver_abi_id: str
    runtime_abi_id: str
    custom_ops_source_hash: str
    generator_source_hash: str
    capability_manifest_hash: str
    tensor_layout_versions: tuple[str, ...]
    template_versions: tuple[str, ...]
    max_programs_per_submit: int = 64
    max_words_per_program: int = 2048
    primitive_extent_limit: int = 256
    execution_policy: tuple[str, ...] = ("synchronous", "single_core_per_submit")
    numeric_policy: str = "rk_f16_tiled_or_strict"
    fallback_policy: str = "error"
    device_budget: int = 256 * 1024 * 1024
    host_budget: int = 256 * 1024 * 1024
    capabilities: tuple[Capability, ...] = field(default_factory=tuple)
    profile_hash: str = ""
    manifest_path: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_dict(cls, obj: Any, *, manifest_path: str = "",
                  verify_hash: bool = True) -> "TargetProfile":
        if not isinstance(obj, dict):
            reject(RK_E_SCHEMA, "profile manifest must be an object", stage="capabilities")
        if obj.get("schema_version") != 1:
            reject(RK_E_SCHEMA, "unsupported capability schema version",
                   stage="capabilities")
        # The manifest has a small stable envelope; profile entries are explicit.
        if "profile" not in obj or "capabilities" not in obj:
            reject(RK_E_SCHEMA, "manifest must contain profile and capabilities",
                   stage="capabilities")
        profile = obj["profile"]
        caps = tuple(Capability.from_dict(entry) for entry in obj["capabilities"])
        required = {
            "schema_version", "chip_id", "driver_abi_id", "runtime_abi_id",
            "custom_ops_source_hash", "generator_source_hash",
            "capability_manifest_hash", "tensor_layout_versions", "template_versions",
            "max_programs_per_submit", "max_words_per_program",
            "primitive_extent_limit", "execution_policy", "numeric_policy",
            "fallback_policy", "device_budget", "host_budget",
        }
        if set(profile) != required:
            reject(RK_E_SCHEMA,
                   "profile fields are missing or unknown; schema must be versioned",
                   stage="capabilities")
        for key in ("max_programs_per_submit", "max_words_per_program",
                    "primitive_extent_limit", "device_budget", "host_budget"):
            value = profile[key]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                reject(RK_E_SCHEMA, f"profile.{key} must be a positive integer",
                       stage="capabilities")
        if profile["max_programs_per_submit"] > 64:
            reject(RK_E_SCHEMA, "max_programs_per_submit cannot exceed the ABI limit 64",
                   stage="capabilities")
        caps_hash = sha256_json([cap.to_dict() for cap in caps])
        if profile["capability_manifest_hash"] != caps_hash:
            reject(RK_E_SCHEMA, "capability_manifest_hash does not match capabilities",
                   stage="capabilities")
        profile_hash = sha256_json({"profile": profile,
                                    "capabilities": [cap.to_dict() for cap in caps]})
        return cls(
            schema_version=1, chip_id=profile["chip_id"],
            driver_abi_id=profile["driver_abi_id"],
            runtime_abi_id=profile["runtime_abi_id"],
            custom_ops_source_hash=profile["custom_ops_source_hash"],
            generator_source_hash=profile["generator_source_hash"],
            capability_manifest_hash=profile["capability_manifest_hash"],
            tensor_layout_versions=tuple(profile["tensor_layout_versions"]),
            template_versions=tuple(profile["template_versions"]),
            max_programs_per_submit=profile["max_programs_per_submit"],
            max_words_per_program=profile["max_words_per_program"],
            primitive_extent_limit=profile["primitive_extent_limit"],
            execution_policy=tuple(profile["execution_policy"]),
            numeric_policy=profile["numeric_policy"],
            fallback_policy=profile["fallback_policy"],
            device_budget=profile["device_budget"],
            host_budget=profile["host_budget"],
            capabilities=caps, profile_hash=profile_hash,
            manifest_path=manifest_path,
            raw=copy.deepcopy(obj),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "chip_id": self.chip_id,
            "driver_abi_id": self.driver_abi_id,
            "runtime_abi_id": self.runtime_abi_id,
            "custom_ops_source_hash": self.custom_ops_source_hash,
            "generator_source_hash": self.generator_source_hash,
            "capability_manifest_hash": self.capability_manifest_hash,
            "tensor_layout_versions": list(self.tensor_layout_versions),
            "template_versions": list(self.template_versions),
            "max_programs_per_submit": self.max_programs_per_submit,
            "max_words_per_program": self.max_words_per_program,
            "primitive_extent_limit": self.primitive_extent_limit,
            "execution_policy": list(self.execution_policy),
            "numeric_policy": self.numeric_policy,
            "fallback_policy": self.fallback_policy,
            "device_budget": self.device_budget,
            "host_budget": self.host_budget,
        }

    def capability(self, capability_id: str) -> Capability:
        for cap in self.capabilities:
            if cap.id == capability_id:
                return cap
        reject(RK_E_TARGET_MISMATCH, f"target profile has no capability {capability_id}",
               stage="capabilities", target_profile=self.profile_hash)

    def enabled(self, capability_id: str) -> Capability:
        cap = self.capability(capability_id)
        if cap.target_profile != self.chip_id + "_vendor_co_v1" and \
                cap.target_profile not in {"rk3588_vendor_co_v1"}:
            reject(RK_E_TARGET_MISMATCH,
                   f"capability {capability_id} belongs to {cap.target_profile}",
                   stage="capabilities", target_profile=self.profile_hash)
        if cap.status != "enabled":
            reject(RK_E_UNSUPPORTED,
                   f"capability {capability_id} is {cap.status}, not enabled",
                   stage="capabilities", target_profile=self.profile_hash)
        return cap

    def validate_environment(self) -> None:
        """Validate explicit host hints without opening a device.

        The real board check is performed by the runtime adapter before BO
        allocation.  Here, if the operator sets ``RKNPU_CHIP`` or
        ``ROCKET_CHIP``, a mismatch is a hard error instead of silently widening
        the capability table.
        """
        hint = os.environ.get("RKNPU_CHIP") or os.environ.get("ROCKET_CHIP")
        if hint and hint.lower() != self.chip_id.lower():
            reject(RK_E_TARGET_MISMATCH,
                   f"host environment reports {hint!r}, profile requires {self.chip_id!r}",
                   stage="capabilities", actual=hint, required=self.chip_id,
                   target_profile=self.profile_hash)


def load_profile(path: str | os.PathLike[str] | None = None) -> TargetProfile:
    manifest_path = Path(path) if path is not None else DEFAULT_PROFILE_PATH
    obj = load_json_safe(manifest_path)
    return TargetProfile.from_dict(obj, manifest_path=str(manifest_path))


def default_profile() -> TargetProfile:
    return load_profile(DEFAULT_PROFILE_PATH)


def check_budgets(profile: TargetProfile, *, device_bytes: int,
                  host_bytes: int) -> None:
    own_device = profile.device_budget
    own_host = profile.host_budget
    if device_bytes > own_device:
        reject(RK_E_BUDGET, "device budget exceeded", stage="storage",
               actual=device_bytes, required=own_device,
               target_profile=profile.profile_hash)
    if host_bytes > own_host:
        reject(RK_E_BUDGET, "host budget exceeded", stage="storage",
               actual=host_bytes, required=own_host,
               target_profile=profile.profile_hash)


# Keep a tiny alias for callers that used the longer name while the package was
# being built.  The stable public spelling is load_profile.
load_target_profile = load_profile
