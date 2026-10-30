# SPDX-License-Identifier: GPL-3.0-or-later
"""Model-level runners for the frozen MNIST MLP and hybrid Tiny Transformer.

The runners only orchestrate already-compiled RKPlan files (or compile them from
real TileLang sources on the Host).  All matrix computation is delegated to the
production ``rkplan`` C ABI; the Transformer's first-round RMSNorm, stable
softmax, layout bridge and mean pooling are explicit Host work.
"""
from __future__ import annotations

import ctypes
import hashlib
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .errors import (RK_E_BUSY, RK_E_INVALID, RK_E_SHAPE, RK_E_UNSUPPORTED,
                     reject)
from .plan import Plan
from .runtime.ffi import (RK_ABI_VERSION, RK_DTYPE_F16, RKBinding, CWorkspace,
                          LibRKPlan, make_binding, validate_numpy_binding)

MNIST_MEAN = np.float32(0.1307)
MNIST_STD = np.float32(0.3081)


def _f16(a: Any) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float16)


def preprocess_mnist(raw: np.ndarray) -> np.ndarray:
    """Convert uint8/float raw ``[N,1,28,28]`` or ``[N,784]`` to normalized f32."""
    a = np.asarray(raw)
    if a.ndim == 3:
        if a.shape[1:] != (28, 28):
            reject(RK_E_SHAPE, "MNIST images must be [N,28,28]", stage="preprocess",
                   actual=list(a.shape))
        a = a.reshape(a.shape[0], 784)
    elif a.ndim == 4:
        if a.shape[1] != 1 or a.shape[2:] != (28, 28):
            reject(RK_E_SHAPE, "MNIST images must be [N,1,28,28]", stage="preprocess",
                   actual=list(a.shape))
        a = a[:, 0, :, :]
    if a.ndim != 2 or a.shape[1] != 784:
        reject(RK_E_SHAPE, "MNIST images must flatten to [N,784]", stage="preprocess",
               actual=list(a.shape))
    af = a.astype(np.float32)
    if af.size and float(np.max(af)) > 1.5:
        af /= 255.0
    return (af - MNIST_MEAN) / MNIST_STD


def patchify_mnist(raw: np.ndarray) -> np.ndarray:
    """Return normalized ``[N,16,49]`` 7x7 patches in raster token order."""
    a = np.asarray(raw)
    if a.ndim == 2 and a.shape[1] == 784:
        a = a.reshape(-1, 1, 28, 28)
    elif a.ndim == 3 and a.shape[1:] == (28, 28):
        a = a[:, None, :, :]
    if a.ndim != 4 or a.shape[1] != 1 or a.shape[2:] != (28, 28):
        reject(RK_E_SHAPE, "MNIST images must be [N,1,28,28]", stage="preprocess",
               actual=list(a.shape))
    af = a.astype(np.float32)
    if af.size and float(np.max(af)) > 1.5:
        af /= 255.0
    af = (af - MNIST_MEAN) / MNIST_STD
    # [N,1,4,7,4,7] -> [N,1,4,4,7,7] -> [N,16,49]
    p = af.reshape(a.shape[0], 1, 4, 7, 4, 7).transpose(0, 1, 2, 4, 3, 5)
    return p.reshape(a.shape[0], 16, 49).astype(np.float32)


TIMING_SCHEMA_VERSION = 2
TIMING_PHASES = ("preprocess", "execution", "postprocess")

# Envelope attribution fields reported by the C runtime, in definition order.
# Each is a disjoint interval inside ``plan_workspace_us``; their sum plus the
# published residual explains the whole native workspace call.
ENVELOPE_FIELDS = ("validate_us", "pack_us", "emit_us", "command_us",
                   "submit_alloc_us", "submit_wait_us", "output_sync_us",
                   "download_us", "guard_us", "publish_us", "alloc_us")
# Self-consistency witnesses reported alongside the attribution intervals.
ENVELOPE_WITNESS_FIELDS = ("total_us", "clock_us")
# Sub-breakdowns of pack_us; they are not additional disjoint intervals.
PACK_BREAKDOWN_FIELDS = ("pack_prep_us", "pack_clear_us", "pack_layout_us",
                         "pack_fini_us", "pack_prep_calls", "pack_fini_calls")


def _nbytes(arrays: tuple[np.ndarray, ...]) -> int:
    return sum(int(np.asarray(a).nbytes) for a in arrays)


@dataclass
class RunStats:
    """Timing and transfer accounting with non-overlapping model phases.

    ``preprocess + execution + postprocess`` is the timed model interval.  CPU
    and declared-reference calculations are deliberately outside this object;
    the evaluator records them separately.  ``host_action_us`` and
    ``plan_call_us`` are breakdowns of the three phase totals, not additional
    time to add again.
    """

    run_calls: int = 0
    end_to_end_us: float = 0.0
    preprocess_us: float = 0.0
    execution_us: float = 0.0
    postprocess_us: float = 0.0
    host_action_us: float = 0.0
    executor_overhead_us: float = 0.0
    plan_call_us: float = 0.0
    plan_workspace_us: float = 0.0
    plan_pack_us: float = 0.0
    plan_emit_us: float = 0.0
    plan_submit_wait_us: float = 0.0
    plan_output_sync_us: float = 0.0
    # Envelope attribution reported by the C runtime: the RKPlan dwell is
    # decomposed into these disjoint intervals so the residual is visible
    # instead of being an unexplained remainder.
    plan_validate_us: float = 0.0
    plan_command_us: float = 0.0
    plan_submit_alloc_us: float = 0.0
    plan_download_us: float = 0.0
    plan_guard_us: float = 0.0
    plan_publish_us: float = 0.0
    plan_alloc_us: float = 0.0
    plan_total_us: float = 0.0
    plan_clock_us: float = 0.0
    plan_pack_prep_us: float = 0.0
    plan_pack_clear_us: float = 0.0
    plan_pack_layout_us: float = 0.0
    plan_pack_fini_us: float = 0.0
    plan_pack_prep_calls: int = 0
    plan_pack_fini_calls: int = 0
    plan_upload_coalesce: int = 0
    plan_upload_fast_pack: int = 0
    upload_calls: int = 0
    upload_bytes: int = 0
    download_calls: int = 0
    download_bytes: int = 0
    task_count: int = 0
    submit_count: int = 0
    # Static-repeat group entries: one native call covers `plan_repeats`
    # independent executions of the same immutable plan.
    plan_group_calls: int = 0
    plan_repeats: int = 0
    # Load-time immutable packed-constant reuse.  These are construction/load
    # costs, deliberately separate from steady-state plan pack counters.
    load_us: float = 0.0
    load_plan_call_us: float = 0.0
    load_shared_constant_actions: int = 0
    load_shared_constant_created: int = 0
    load_shared_constant_reused: int = 0
    load_shared_constant_bytes: int = 0
    # Persistent KV native-layout state updates.  These are separate from the
    # plan upload counters because the C runtime owns the destination BO and the
    # model only writes newly required stripes.
    kv_state_init_calls: int = 0
    kv_state_init_bytes: int = 0
    kv_state_init_us: float = 0.0
    kv_state_append_calls: int = 0
    kv_state_append_bytes: int = 0
    kv_state_append_us: float = 0.0
    kv_state_clear_calls: int = 0
    kv_state_clear_bytes: int = 0
    kv_state_clear_us: float = 0.0
    # Compiled graph execution: metadata-only view nodes and trace-only writes
    # elided from the production path.  These are structural counters, not time.
    graph_view_ops: int = 0
    graph_skipped_trace_output_writes: int = 0
    host_action_bytes: int = 0
    plans: list[dict[str, Any]] = field(default_factory=list)
    host_actions: list[dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def _phase(phase: str) -> str:
        if phase not in TIMING_PHASES:
            reject(RK_E_INVALID, f"unknown timing phase {phase!r}", stage="model")
        return phase

    def _phase_add(self, phase: str, elapsed_us: float) -> None:
        phase = self._phase(phase)
        setattr(self, f"{phase}_us", float(getattr(self, f"{phase}_us")) +
                float(elapsed_us))

    def add_host_action(self, name: str, phase: str, elapsed_us: float,
                        inputs: tuple[np.ndarray, ...] = (),
                        outputs: tuple[np.ndarray, ...] = (),
                        *, nbytes: int | None = None) -> None:
        """Record one registered Host boundary action and its Host bytes.

        ``nbytes`` may be supplied by actions such as an in-place KV append
        where the bound arrays span the whole persistent cache but only one
        slot is actually touched.  The override is explicit; the default keeps
        the historical resolved-input-plus-output byte count.
        """
        phase = self._phase(phase)
        if nbytes is None:
            nbytes = _nbytes(tuple(inputs) + tuple(outputs))
        elif not isinstance(nbytes, int) or isinstance(nbytes, bool) or nbytes < 0:
            reject(RK_E_INVALID, "host action nbytes override must be >= 0",
                   stage="model")
        self.host_action_us += float(elapsed_us)
        self.host_action_bytes += nbytes
        self._phase_add(phase, elapsed_us)
        row = next((r for r in self.host_actions if r["name"] == name), None)
        if row is None:
            row = {"name": name, "phase": phase, "calls": 0,
                   "elapsed_us": 0.0, "host_bytes": 0}
            self.host_actions.append(row)
        row["calls"] += 1
        row["elapsed_us"] += float(elapsed_us)
        row["host_bytes"] += nbytes

    def _accumulate(self, name: str, report: Any, elapsed_us: float,
                    workspace_us: float | None, phase: str, *,
                    calls: int = 1, repeats: int = 1) -> None:
        self.plan_call_us += float(elapsed_us)
        self.plan_workspace_us += (float(elapsed_us) if workspace_us is None
                                   else float(workspace_us))
        # The four legacy fields are members of ENVELOPE_FIELDS; they are
        # accumulated once by the loop below.  Adding them here as well would
        # double-count them and make the envelope residual negative.
        for field in ENVELOPE_FIELDS + ENVELOPE_WITNESS_FIELDS:
            if hasattr(report, field):
                setattr(self, f"plan_{field}",
                        getattr(self, f"plan_{field}") + float(
                            getattr(report, field)))
        for field in PACK_BREAKDOWN_FIELDS:
            if hasattr(report, field):
                setattr(self, f"plan_{field}",
                        getattr(self, f"plan_{field}") + getattr(report, field))
        for field in ("upload_coalesce", "upload_fast_pack"):
            if hasattr(report, field):
                setattr(self, f"plan_{field}", int(getattr(report, field)))
        self.upload_calls += int(report.upload_calls)
        self.upload_bytes += int(report.upload_bytes)
        self.download_calls += int(report.download_calls)
        self.download_bytes += int(report.download_bytes)
        self.task_count += int(report.task_count)
        self.submit_count += int(report.submit_count)
        self.plan_repeats += int(repeats)
        if calls and repeats > 1:
            self.plan_group_calls += int(calls)
        self._phase_add(phase, elapsed_us)
        # Aggregate repeated calls so a full test-set evaluation has a bounded
        # report (one row per logical plan) rather than one row per batch.
        row = next((r for r in self.plans if r["name"] == name), None)
        if row is None:
            row = {"name": name, "kind": "rkplan", "phase": phase, "calls": 0,
                   "repeat_count": 0,
                   "elapsed_us": 0.0, "workspace_us": 0.0,
                   "pack_us": 0.0, "emit_us": 0.0,
                   "submit_wait_us": 0.0, "output_sync_us": 0.0,
                   "validate_us": 0.0, "command_us": 0.0,
                   "submit_alloc_us": 0.0, "download_us": 0.0,
                   "guard_us": 0.0, "publish_us": 0.0, "alloc_us": 0.0,
                   "total_us": 0.0, "clock_us": 0.0,
                   **{field: 0.0 for field in PACK_BREAKDOWN_FIELDS},
                   "upload_coalesce": 0, "upload_fast_pack": 0,
                   "upload_calls": 0, "upload_bytes": 0,
                   "download_calls": 0, "download_bytes": 0,
                   "task_count": 0, "submit_count": 0,
                   "bridge_bytes": 0}
            self.plans.append(row)
        row["calls"] += int(calls)
        row["repeat_count"] += int(repeats)
        row["elapsed_us"] += float(elapsed_us)
        row["workspace_us"] += (float(elapsed_us) if workspace_us is None
                                else float(workspace_us))
        for field in ENVELOPE_FIELDS + ENVELOPE_WITNESS_FIELDS:
            if hasattr(report, field):
                row[field] = row.get(field, 0.0) + float(getattr(report, field))
        for field in PACK_BREAKDOWN_FIELDS:
            if hasattr(report, field):
                row[field] = row.get(field, 0.0) + getattr(report, field)
        for field in ("upload_coalesce", "upload_fast_pack"):
            if hasattr(report, field):
                row[field] = int(getattr(report, field))
        row["upload_calls"] += int(report.upload_calls)
        row["upload_bytes"] += int(report.upload_bytes)
        row["download_calls"] += int(report.download_calls)
        row["download_bytes"] += int(report.download_bytes)
        row["task_count"] += int(report.task_count)
        row["submit_count"] += int(report.submit_count)
        row["bridge_bytes"] = (int(row["upload_bytes"]) +
                               int(row["download_bytes"]))

    def add_report(self, name: str, report: Any, elapsed_us: float,
                   *, phase: str = "execution",
                   workspace_us: float | None = None) -> None:
        """Record one production RKPlan node interval.

        ``elapsed_us`` is the full plan-node dwell (input resolution, binding
        preparation, C workspace run and output-target handling).  The optional
        ``workspace_us`` is only the C ``rk_workspace_run`` interval, so callers can
        separate model-runtime overhead from native execution.
        """
        self._accumulate(name, report, elapsed_us, workspace_us,
                         self._phase(phase))

    def add_group_report(self, name: str, report: Any, elapsed_us: float,
                         *, repeats: int, phase: str = "execution",
                         workspace_us: float | None = None) -> None:
        """Record one static-repeat group entry covering ``repeats`` runs.

        The C side reports one aggregate row per group call, so the counters and
        attribution intervals are already summed over the repeats.  Publication
        is all-or-nothing, therefore a successful return must show every repeat
        completed and no failing repeat.
        """
        if int(report.repeat_count) != int(repeats):
            reject(RK_E_INVALID, f"group report covers {report.repeat_count} "
                   f"repeats, expected {repeats}", stage="model")
        if int(report.repeat_completed) != int(repeats) or \
                int(report.fail_repeat) != 0xFFFFFFFF:
            reject(RK_E_INVALID, "group entry reported a partially completed "
                   "repeat set as success", stage="model",
                   actual={"completed": int(report.repeat_completed),
                           "fail_repeat": int(report.fail_repeat)})
        self._accumulate(name, report, elapsed_us, workspace_us,
                         self._phase(phase), calls=1, repeats=repeats)

    def add_executor_overhead(self, elapsed_us: float) -> None:
        """Attribute graph dispatch/binding overhead that is not a Host action."""
        self.executor_overhead_us += float(elapsed_us)
        self._phase_add("execution", elapsed_us)

    def record_run(self, end_to_end_us: float) -> None:
        self.run_calls += 1
        self.end_to_end_us += float(end_to_end_us)

    @property
    def plan_envelope_attributed_us(self) -> float:
        """Sum of the C-side attributed intervals inside the workspace dwell."""
        return sum(float(getattr(self, f"plan_{field}"))
                   for field in ENVELOPE_FIELDS)

    @property
    def plan_envelope_residual_us(self) -> float:
        """Workspace dwell not yet attributed by the C runtime.

        The reference interval is the C-side total when the library reports one
        (it is measured by the same clock as the attribution intervals), and the
        caller-observed dwell otherwise.  A large residual means some cost inside
        ``rk_workspace_run`` is still unattributed; it is reported rather than
        folded into a named bucket.
        """
        base = self.plan_total_us or self.plan_workspace_us
        return base - self.plan_envelope_attributed_us

    def _phases(self) -> dict[str, float]:

        return {phase: float(getattr(self, f"{phase}_us"))
                for phase in TIMING_PHASES}

    def to_dict(self) -> dict[str, Any]:
        phases = self._phases()
        attributed = sum(phases.values())
        return {
            "timing_schema_version": TIMING_SCHEMA_VERSION,
            "reference_timing": "excluded",
            "run_calls": self.run_calls,
            "end_to_end_us": self.end_to_end_us,
            "preprocess_us": phases["preprocess"],
            "execution_us": phases["execution"],
            "postprocess_us": phases["postprocess"],
            "unattributed_us": max(0.0, self.end_to_end_us - attributed),
            "host_action_us": self.host_action_us,
            "host_compute_us": self.host_action_us,
            "executor_overhead_us": self.executor_overhead_us,
            "plan_call_us": self.plan_call_us,
            "plan_workspace_us": self.plan_workspace_us,
            "plan_pack_us": self.plan_pack_us,
            "plan_emit_us": self.plan_emit_us,
            "plan_submit_wait_us": self.plan_submit_wait_us,
            "plan_output_sync_us": self.plan_output_sync_us,
            **{f"plan_{field}": float(getattr(self, f"plan_{field}"))
               for field in ENVELOPE_FIELDS + ENVELOPE_WITNESS_FIELDS},
            **{f"plan_{field}": float(getattr(self, f"plan_{field}"))
               for field in PACK_BREAKDOWN_FIELDS},
            "plan_upload_coalesce": int(self.plan_upload_coalesce),
            "plan_upload_fast_pack": int(self.plan_upload_fast_pack),
            "plan_envelope_attributed_us": self.plan_envelope_attributed_us,
            "plan_envelope_residual_us": self.plan_envelope_residual_us,
            # Backward-compatible summary name: this is only the RKPlan interval,
            # never the model wall interval.
            "elapsed_us": self.plan_call_us,
            "upload_calls": self.upload_calls,
            "upload_bytes": self.upload_bytes,
            "download_calls": self.download_calls,
            "download_bytes": self.download_bytes,
            "task_count": self.task_count,
            "submit_count": self.submit_count,
            "host_action_bytes": self.host_action_bytes,
            "graph_view_ops": int(self.graph_view_ops),
            "graph_skipped_trace_output_writes":
                int(self.graph_skipped_trace_output_writes),
            "bridge_bytes": self.upload_bytes + self.download_bytes,
            "load_us": float(self.load_us),
            "load_plan_call_us": float(self.load_plan_call_us),
            "load_shared_constant_actions": int(
                self.load_shared_constant_actions),
            "load_shared_constant_created": int(
                self.load_shared_constant_created),
            "load_shared_constant_reused": int(
                self.load_shared_constant_reused),
            "load_shared_constant_bytes": int(
                self.load_shared_constant_bytes),
            "kv_state_init_calls": int(self.kv_state_init_calls),
            "kv_state_init_bytes": int(self.kv_state_init_bytes),
            "kv_state_init_us": float(self.kv_state_init_us),
            "kv_state_append_calls": int(self.kv_state_append_calls),
            "kv_state_append_bytes": int(self.kv_state_append_bytes),
            "kv_state_append_us": float(self.kv_state_append_us),
            "kv_state_clear_calls": int(self.kv_state_clear_calls),
            "kv_state_clear_bytes": int(self.kv_state_clear_bytes),
            "kv_state_clear_us": float(self.kv_state_clear_us),
            "plans": list(self.plans),
            "host_actions": list(self.host_actions),
        }

def _shared_constant_pack_key(plan: Plan, action: Any,
                                source: Any, source_sha256: str) -> str:
    """Identity for one immutable packed upload destination.

    The key binds the actual source bytes to the source region, stride and
    transpose plus the destination tensor's complete physical geometry.  It
    deliberately excludes the plan/profile/binding id: two profiles may share a
    weight tile, but they must not share it if any byte-level or layout-level
    input differs.  The C runtime independently re-checks the same structural
    fields and the same fd/context.
    """
    if action.kind != "upload" or len(action.outputs) != 1:
        reject(RK_E_SCHEMA, "shared constant must be a single-output upload",
               stage="model", action_id=action.id)
    view = plan.views[action.outputs[0]]
    tensor = view.tensor
    buffer = plan.buffers[view.buffer_id]
    attrs = action.attrs
    h = hashlib.sha256()
    h.update(b"rk-shared-packed-constant-v1\0")
    h.update(bytes.fromhex(source_sha256))
    h.update(struct.pack(
        "<QQQQQQQQQQQ",
        int(source.nbytes),
        int(attrs.get("src_row_offset", 0)),
        int(attrs.get("src_col_offset", 0)),
        int(attrs.get("src_row_stride", 0)),
        int(bool(attrs.get("transpose_source", False))),
        int(view.offset_bytes),
        int(tensor.rows), int(tensor.cols),
        int(tensor.physical_rows), int(tensor.physical_cols),
        int(tensor.atom)))
    h.update(tensor.layout.encode("ascii"))
    h.update(b"\0")
    h.update(tensor.dtype.encode("ascii"))
    h.update(b"\0")
    h.update(struct.pack("<Q", int(tensor.bytes)))
    h.update(str(attrs.get("broadcast", "")).encode("ascii"))
    h.update(b"\0")
    h.update(buffer.layout_id.encode("ascii"))
    h.update(b"\0")
    h.update(buffer.padding_state.encode("ascii"))
    h.update(struct.pack("<QQQ", int(buffer.capacity), int(buffer.offset),
                         int(buffer.alignment)))
    return h.hexdigest()


class PlanCall:
    """Reusable workspace for one immutable plan and its constants.

    Construction validates/materialises every Host constant before asking C to
    create the plan.  If a later C call fails, already-created handles are
    destroyed without replacing the original exception.  Destruction is
    idempotent; a failed workspace destroy keeps both workspace and CPlan alive
    because the runtime owns quarantine retry state.
    """

    def __init__(self, lib: LibRKPlan, plan: Plan, constants: dict[str, np.ndarray],
                 *, fd: int = 0, entry: str = "attributed",
                 reuse_bindings: bool = True,
                 share_constants: bool = False,
                 profile: bool = False):
        if entry not in ("attributed", "legacy"):
            reject(RK_E_INVALID, f"unknown plan entry {entry!r}", stage="model")
        if not isinstance(reuse_bindings, bool):
            reject(RK_E_INVALID, "reuse_bindings must be boolean", stage="model")
        if not isinstance(share_constants, bool):
            reject(RK_E_INVALID, "share_constants must be boolean", stage="model")
        if not isinstance(profile, bool):
            reject(RK_E_INVALID, "profile must be boolean", stage="model")
        self.entry = entry
        self.reuse_bindings = reuse_bindings
        self.share_constants = bool(share_constants)
        self.profile = bool(profile)
        self.last_profile: dict[str, float] | None = None
        self.shared_constant_actions = 0
        self.shared_constant_created = 0
        self.shared_constant_reused = 0
        self.shared_constant_bytes = 0
        self.lib = lib
        self.plan = plan
        self.cplan: CPlan | None = None
        self.workspace: CWorkspace | None = None
        self._closed = False
        self._const_arrays: list[np.ndarray] = []
        bindings = []
        try:
            # A missing/mismatched Host constant is a construction error before
            # any C resource exists.
            digest_by_binding: dict[int, str] = {}
            for spec in plan.constants:
                if spec.name not in constants:
                    raise KeyError(
                        f"missing constant {spec.name!r} for {plan.name}")
                arr = _f16(constants[spec.name])
                expected = tuple(spec.shape)
                if arr.shape != expected:
                    reject(RK_E_SHAPE, f"constant {spec.name!r} shape mismatch",
                           stage="model", actual=list(arr.shape),
                           required=list(expected))
                self._const_arrays.append(arr)
                bindings.append(make_binding(spec.binding_id, arr))
                digest_by_binding[spec.binding_id] = hashlib.sha256(
                    memoryview(arr).cast("B")).hexdigest()
            self.cplan = lib.create_plan(plan)
            shared_entries: list[tuple[int, Any]] = []
            shared_handles: list[Any] = []
            if self.share_constants:
                if not getattr(lib, "supports_shared_constants", False):
                    reject(RK_E_UNSUPPORTED,
                           "this rknpu_plan library does not support shared "
                           "immutable constants", stage="model")
                constant_by_binding = {b.binding_id: b for b in plan.constants}
                try:
                    for action in plan.actions:
                        if action.kind != "upload":
                            continue
                        source_id = int(action.attrs.get("source_binding", -1))
                        spec = constant_by_binding.get(source_id)
                        if spec is None:
                            continue
                        source = next(b for b in bindings
                                      if int(b.id) == source_id)
                        key = _shared_constant_pack_key(
                            plan, action, spec, digest_by_binding[source_id])
                        handle, created = lib.create_shared_constant_on(
                            fd, self.cplan, action.id, source, key)
                        shared_handles.append(handle)
                        shared_entries.append((action.id, handle))
                        self.shared_constant_actions += 1
                        self.shared_constant_created += int(created)
                        self.shared_constant_reused += int(not created)
                        self.shared_constant_bytes += int(
                            lib.cdll.rk_shared_constant_bytes(handle))
                    self.workspace = lib.create_workspace_shared(
                        self.cplan, bindings, shared_entries, fd=fd)
                finally:
                    # The workspace retained every unique handle.  Drop this
                    # builder's references even on failure so a failed load
                    # cannot keep cache BOs alive.
                    for handle in shared_handles:
                        lib.release_shared_constant(handle)
            else:
                self.workspace = lib.create_workspace(
                    self.cplan, bindings, fd=fd)
        except BaseException as exc:
            cleanup_errors = []
            workspace, cplan = self.workspace, self.cplan
            if workspace is not None:
                try:
                    workspace.destroy()
                except BaseException as cleanup:
                    cleanup_errors.append(cleanup)
            # A live workspace may still point at the plan, so only destroy the
            # plan after a successful workspace release.
            if not cleanup_errors and cplan is not None:
                try:
                    cplan.destroy()
                except BaseException as cleanup:
                    cleanup_errors.append(cleanup)
            self.workspace = None
            self.cplan = None
            self._closed = not cleanup_errors
            if cleanup_errors:
                exc.add_note(
                    "model PlanCall construction cleanup also failed: " +
                    "; ".join(repr(e) for e in cleanup_errors))
            raise
        self.inputs = [b for b in plan.bindings if b.role == "input"]
        self.outputs = [b for b in plan.bindings if b.role == "output"]
        self._input_bind_cache = [self._binding_slot(b) for b in self.inputs]
        self._output_bind_cache = [self._binding_slot(b) for b in self.outputs]
        # Shared descriptor arrays are protected from the first pointer update
        # through the synchronous C call.  Non-blocking acquisition preserves
        # the established BUSY result for concurrent users of one PlanCall.
        self._descriptor_lock = threading.Lock()
        # A compiled runner can bind stable resident arrays once.  Keeping the
        # arrays here also keeps the ctypes pointers and their backing storage
        # alive for the whole workspace lifetime.
        self._prepared_single: dict[str, tuple[Any, ...]] = {}
        self._prepared_group: dict[str, tuple[Any, ...]] = {}
        self.persistent_actions: set[int] = set()

    def set_persistent_upload(self, action_id: int) -> None:
        """Mark one workspace upload action as caller-managed persistent state.

        The normal run no longer rewrites that destination.  The caller must
        initialize/update it through :meth:`update_upload_region` and clear it on
        reset.  The C runtime still owns the BO and validates every region.
        """
        if self.workspace is None:
            reject(RK_E_INVALID, "model plan workspace is already closed",
                   stage="model")
        self.lib.set_persistent_upload(self.workspace, int(action_id))
        self.persistent_actions.add(int(action_id))

    def update_upload_region(self, action_id: int, array: np.ndarray, *,
                             src_row: int, src_col: int,
                             dst_row: int, dst_col: int,
                             rows: int, cols: int,
                             transpose_source: bool = False) -> None:
        if int(action_id) not in self.persistent_actions:
            reject(RK_E_INVALID,
                   "upload action is not marked persistent", stage="model",
                   actual=int(action_id))
        self.lib.update_upload_region(
            self.workspace, int(action_id), array, src_row=int(src_row),
            src_col=int(src_col), dst_row=int(dst_row), dst_col=int(dst_col),
            rows=int(rows), cols=int(cols),
            transpose_source=bool(transpose_source))

    def clear_upload_destination(self, action_id: int) -> None:
        if int(action_id) not in self.persistent_actions:
            reject(RK_E_INVALID,
                   "upload action is not marked persistent", stage="model",
                   actual=int(action_id))
        self.lib.clear_upload_destination(self.workspace, int(action_id))

    def _lock_descriptors(self) -> None:
        if not self._descriptor_lock.acquire(blocking=False):
            reject(RK_E_BUSY,
                   "PlanCall is busy; concurrent use requires separate instances",
                   stage="model")

    @staticmethod
    def _binding_slot(spec: Any) -> RKBinding:
        shape = list(spec.shape) + [0] * (2 - len(spec.shape))
        return RKBinding(
            abi_version=RK_ABI_VERSION,
            struct_size=ctypes.sizeof(RKBinding),
            id=spec.id,
            dtype=RK_DTYPE_F16,
            rank=len(spec.shape),
            shape=(ctypes.c_uint32 * 2)(*shape),
            data=None,
            nbytes=spec.nbytes)

    def _prepare_inputs(self, inputs: list[np.ndarray],
                        finite_cache: set[tuple[int, int, str]] | None = None,
                        *, check_finite: bool = True
                        ) -> list[np.ndarray]:
        if len(inputs) != len(self.inputs):
            reject(RK_E_SHAPE, "model plan input count mismatch", stage="model",
                   actual=len(inputs), required=len(self.inputs))
        prepared = []
        for spec, arr in zip(self.inputs, inputs):
            if not isinstance(arr, np.ndarray):
                reject(RK_E_INVALID, "model plan input must be numpy.ndarray",
                       stage="model")
            if arr.dtype != np.dtype("<f2"):
                reject(RK_E_INVALID,
                       "model plan input must be native FP16; implicit astype is "
                       "forbidden at the runtime boundary", stage="model",
                       actual=str(arr.dtype), required="float16")
            if arr.shape != tuple(spec.shape):
                reject(RK_E_SHAPE, "model plan input shape mismatch", stage="model",
                       actual=list(arr.shape), required=list(spec.shape))
            should_check = bool(check_finite)
            if should_check and finite_cache is not None:
                key = (int(arr.ctypes.data), int(arr.nbytes), arr.dtype.str)
                should_check = key not in finite_cache
                finite_cache.add(key)
            prepared.append(validate_numpy_binding(
                arr, tuple(spec.shape), writable=False, name=spec.name,
                check_finite=should_check))
        return prepared

    def _prepare_outputs(self, outputs: list[np.ndarray]) -> list[np.ndarray]:
        if len(outputs) != len(self.outputs):
            reject(RK_E_SHAPE, "model plan output count mismatch", stage="model",
                   actual=len(outputs), required=len(self.outputs))
        prepared = []
        for spec, arr in zip(self.outputs, outputs):
            if not isinstance(arr, np.ndarray):
                reject(RK_E_INVALID, "model plan output must be numpy.ndarray",
                       stage="model")
            if arr.dtype != np.dtype("<f2"):
                reject(RK_E_INVALID,
                       "model plan output must be native FP16", stage="model",
                       actual=str(arr.dtype), required="float16")
            if arr.shape != tuple(spec.shape):
                reject(RK_E_SHAPE, "model plan output shape mismatch", stage="model",
                       actual=list(arr.shape), required=list(spec.shape))
            prepared.append(validate_numpy_binding(
                arr, tuple(spec.shape), writable=True, name=spec.name))
        return prepared

    def run_into(self, inputs: list[np.ndarray],
                 outputs: list[np.ndarray]) -> tuple[Any, float]:
        """One plan call.

        ``entry="attributed"`` (the default) uses ``rk_workspace_run_ex`` and
        returns the envelope attribution; ``entry="legacy"`` uses the frozen
        88-byte ``rk_workspace_run`` report and exists so a campaign can measure
        what the attribution itself costs on an otherwise identical binary.
        """
        if self.workspace is None:
            reject(RK_E_INVALID, "model plan workspace is already closed",
                   stage="model")
        t_wrapper = time.perf_counter() if self.profile else 0.0
        t_lock = time.perf_counter() if self.profile else 0.0
        self._lock_descriptors()
        t_locked = time.perf_counter() if self.profile else 0.0
        try:
            report, elapsed = self._run_into_locked(inputs, outputs)
        finally:
            self._descriptor_lock.release()
        if self.profile:
            done = time.perf_counter()
            profile = dict(self.last_profile or {})
            profile["lock_wait_us"] = max(0.0, (t_locked - t_lock) * 1e6)
            profile["wrapper_total_us"] = max(0.0, (done - t_wrapper) * 1e6)
            profile["wrapper_unattributed_us"] = max(
                0.0, profile["wrapper_total_us"] -
                profile.get("input_prepare_us", 0.0) -
                profile.get("output_prepare_us", 0.0) -
                profile.get("binding_us", 0.0) -
                profile.get("c_runtime_us", 0.0) -
                profile["lock_wait_us"])
            self.last_profile = profile
        return report, elapsed

    def _run_into_locked(self, inputs: list[np.ndarray],
                         outputs: list[np.ndarray],
                         *, check_finite: bool = True
                         ) -> tuple[Any, float]:
        t_in0 = time.perf_counter() if self.profile else 0.0
        in_arrays = self._prepare_inputs(inputs, finite_cache=set(),
                                         check_finite=check_finite)
        t_in1 = time.perf_counter() if self.profile else 0.0
        out_arrays = self._prepare_outputs(outputs)
        t_out1 = time.perf_counter() if self.profile else 0.0
        if self.reuse_bindings:
            in_bind = self._input_bind_cache
            out_bind = self._output_bind_cache
            for bind, arr in zip(in_bind, in_arrays):
                bind.data = arr.ctypes.data
                bind.nbytes = arr.nbytes
            for bind, arr in zip(out_bind, out_arrays):
                bind.data = arr.ctypes.data
                bind.nbytes = arr.nbytes
        else:
            in_bind = [make_binding(spec.id, arr)
                       for spec, arr in zip(self.inputs, in_arrays)]
            out_bind = [make_binding(spec.id, arr)
                        for spec, arr in zip(self.outputs, out_arrays)]
        t_bind = time.perf_counter() if self.profile else 0.0
        # The mock harness snapshots each emitted co_program.  Clear that
        # test-only cache between calls; production builds have no such symbol.
        if hasattr(self.lib.cdll, "rk_mock_reset_tags"):
            self.lib.mock_control("rk_mock_reset_tags")
        t0 = time.perf_counter()
        if self.entry == "legacy":
            report = self.workspace.run(in_bind, out_bind)
        else:
            report = self.workspace.run_attributed(in_bind, out_bind)
        elapsed_us = (time.perf_counter() - t0) * 1e6
        t_done = time.perf_counter() if self.profile else 0.0
        if self.profile:
            self.last_profile = {
                "input_prepare_us": max(0.0, (t_in1 - t_in0) * 1e6),
                "output_prepare_us": max(0.0, (t_out1 - t_in1) * 1e6),
                "binding_us": max(0.0, (t_bind - t_out1) * 1e6),
                "c_runtime_us": max(0.0, elapsed_us),
                "inner_unattributed_us": max(
                    0.0, (t_done - t_bind) * 1e6 - elapsed_us),
            }
        return report, elapsed_us

    def bind_prepared_into(self, inputs: list[np.ndarray],
                           outputs: list[np.ndarray], *,
                           binding_key: str = "default") -> None:
        """Validate stable resident views once and prebuild their descriptors.

        This is an internal compiler-facing entry.  It does *not* weaken the
        per-run guard: every C workspace run still checks binding ids, dtypes,
        shapes, storage extents, aliases and finite inputs.  Python skips only
        the duplicate shape/dtype/finite scans that a compiled graph has already
        discharged over immutable internal buffers.
        """
        if self.workspace is None:
            reject(RK_E_INVALID, "model plan workspace is already closed",
                   stage="model")
        in_arrays = tuple(self._prepare_inputs(
            inputs, finite_cache=None, check_finite=False))
        out_arrays = tuple(self._prepare_outputs(outputs))
        if self.reuse_bindings:
            in_bind = self._input_bind_cache
            out_bind = self._output_bind_cache
        else:
            in_bind = [make_binding(spec.id, arr)
                       for spec, arr in zip(self.inputs, in_arrays)]
            out_bind = [make_binding(spec.id, arr)
                        for spec, arr in zip(self.outputs, out_arrays)]
        for bind, arr in zip(in_bind, in_arrays):
            bind.data = arr.ctypes.data
            bind.nbytes = arr.nbytes
        for bind, arr in zip(out_bind, out_arrays):
            bind.data = arr.ctypes.data
            bind.nbytes = arr.nbytes
        in_bound = (RKBinding * max(len(in_bind), 1))()
        out_bound = (RKBinding * max(len(out_bind), 1))()
        for i, binding in enumerate(in_bind):
            in_bound[i] = binding
        for i, binding in enumerate(out_bind):
            out_bound[i] = binding
        if binding_key in self._prepared_single:
            reject(RK_E_INVALID,
                   f"prepared binding key {binding_key!r} is already installed",
                   stage="model")
        self._prepared_single[binding_key] = (
            in_arrays, out_arrays, in_bound, out_bound)

    def run_prepared(self, *, binding_key: str = "default"
                     ) -> tuple[Any, float]:
        """Run a workspace whose stable Host bindings were compiled once."""
        if self.workspace is None:
            reject(RK_E_INVALID, "model plan workspace is already closed",
                   stage="model")
        if binding_key not in self._prepared_single:
            reject(RK_E_INVALID,
                   f"prepared binding key {binding_key!r} is not installed",
                   stage="model")
        if self.entry == "legacy":
            reject(RK_E_UNSUPPORTED,
                   "prepared bindings require the attributed execution entry",
                   stage="model")
        self._lock_descriptors()
        try:
            in_arrays, out_arrays, in_bind, out_bind = \
                self._prepared_single[binding_key]
            if hasattr(self.lib.cdll, "rk_mock_reset_tags"):
                self.lib.mock_control("rk_mock_reset_tags")
            t0 = time.perf_counter()
            report = self.lib.workspace_run_attributed_bound(
                self.workspace, in_bind, out_bind,
                len(in_arrays), len(out_arrays))
            elapsed_us = (time.perf_counter() - t0) * 1e6
            return report, elapsed_us
        finally:
            self._descriptor_lock.release()

    def _group_binding_slots(self, nrepeat: int):
        """Cache one flat binding array for ``nrepeat`` repeats of this plan.

        Only ``data`` differs between repeats; identity, dtype, rank, shape and
        nbytes come from the immutable plan, so they are written once.  The
        caller's arrays are still validated per repeat before a pointer is
        stored, so a wrong dtype/shape/alias cannot reach the C side.
        """
        cache = getattr(self, "_group_cache", None)
        if cache is not None and cache[0] == nrepeat:
            return cache[1]
        specs = self.inputs + self.outputs
        dummy: dict[tuple[int, ...], np.ndarray] = {}
        arr = (RKBinding * (nrepeat * len(specs)))()
        for r in range(nrepeat):
            for i, spec in enumerate(specs):
                key = tuple(spec.shape)
                if key not in dummy:
                    dummy[key] = np.empty(key, dtype=np.float16)
                arr[r * len(specs) + i] = make_binding(spec.id, dummy[key])
        self._group_cache = (nrepeat, arr)
        return arr

    def run_group_into(self, inputs_list: list[list[np.ndarray]],
                       outputs_list: list[list[np.ndarray]]) -> tuple[Any, float]:
        """Execute ``n`` independent binding sets inside one native call.

        Every repeat still validates its own inputs and outputs exactly as
        ``run_into`` does.  The submit count is unchanged: the group entry only
        removes the per-repeat Host transition, the per-repeat guard pass and the
        per-repeat scratch allocation.
        """
        if self.workspace is None:
            reject(RK_E_INVALID, "model plan workspace is already closed",
                   stage="model")
        self._lock_descriptors()
        try:
            return self._run_group_into_locked(inputs_list, outputs_list)
        finally:
            self._descriptor_lock.release()

    def _run_group_into_locked(self, inputs_list: list[list[np.ndarray]],
                               outputs_list: list[list[np.ndarray]]
                               ) -> tuple[Any, float]:
        nrepeat = len(inputs_list)
        if nrepeat < 1 or nrepeat != len(outputs_list):
            reject(RK_E_SHAPE, "group entry needs matching, non-empty input and "
                   "output repeat lists", stage="model",
                   actual={"inputs": nrepeat, "outputs": len(outputs_list)})
        flat = self._group_binding_slots(nrepeat)
        step = len(self.inputs) + len(self.outputs)
        finite_cache: set[tuple[int, int, str]] = set()
        for r in range(nrepeat):
            in_arrays = self._prepare_inputs(inputs_list[r],
                                             finite_cache=finite_cache)
            out_arrays = self._prepare_outputs(outputs_list[r])
            base = r * step
            for k, arr in enumerate(in_arrays):
                flat[base + k].data = arr.ctypes.data
            for k, arr in enumerate(out_arrays):
                flat[base + len(self.inputs) + k].data = arr.ctypes.data
        # The mock harness snapshots each emitted co_program.  Clear that
        # test-only cache between calls; production builds have no such symbol.
        if hasattr(self.lib.cdll, "rk_mock_reset_tags"):
            self.lib.mock_control("rk_mock_reset_tags")
        t0 = time.perf_counter()
        report = self.workspace.run_group(flat, nrepeat)
        elapsed_us = (time.perf_counter() - t0) * 1e6
        return report, elapsed_us

    def bind_group_prepared_into(
            self, inputs_list: list[list[np.ndarray]],
            outputs_list: list[list[np.ndarray]], *,
            binding_key: str = "default") -> None:
        """Build one stable repeat-major binding array for a compiled group.

        Structural shape/dtype/contiguity checks run once here.  The C group
        entry still performs all per-repeat binding, finite-value, alias and
        publication checks on every execution.
        """
        if self.workspace is None:
            reject(RK_E_INVALID, "model plan workspace is already closed",
                   stage="model")
        nrepeat = len(inputs_list)
        if nrepeat < 1 or nrepeat != len(outputs_list):
            reject(RK_E_SHAPE, "group entry needs matching, non-empty input and "
                   "output repeat lists", stage="model",
                   actual={"inputs": nrepeat, "outputs": len(outputs_list)})
        specs = self.inputs + self.outputs
        flat = (RKBinding * (nrepeat * len(specs)))()
        flat_inputs: list[np.ndarray] = []
        flat_outputs: list[np.ndarray] = []
        step = len(specs)
        for r in range(nrepeat):
            in_arrays = self._prepare_inputs(
                inputs_list[r], finite_cache=None, check_finite=False)
            out_arrays = self._prepare_outputs(outputs_list[r])
            base = r * step
            flat_inputs.extend(in_arrays)
            flat_outputs.extend(out_arrays)
            for k, (spec, arr) in enumerate(zip(self.inputs, in_arrays)):
                flat[base + k] = make_binding(spec.id, arr)
            for k, (spec, arr) in enumerate(zip(self.outputs, out_arrays)):
                flat[base + len(self.inputs) + k] = make_binding(spec.id, arr)
        if binding_key in self._prepared_group:
            reject(RK_E_INVALID,
                   f"prepared group key {binding_key!r} is already installed",
                   stage="model")
        self._prepared_group[binding_key] = (
            flat, nrepeat, tuple(flat_inputs), tuple(flat_outputs))

    def run_group_prepared(self, *, binding_key: str = "default"
                           ) -> tuple[Any, float]:
        """Run a stable repeat-major group whose bindings were compiled once."""
        if self.workspace is None:
            reject(RK_E_INVALID, "model plan workspace is already closed",
                   stage="model")
        if binding_key not in self._prepared_group:
            reject(RK_E_INVALID,
                   f"prepared group key {binding_key!r} is not installed",
                   stage="model")
        self._lock_descriptors()
        try:
            flat, nrepeat, _, _ = self._prepared_group[binding_key]
            if hasattr(self.lib.cdll, "rk_mock_reset_tags"):
                self.lib.mock_control("rk_mock_reset_tags")
            t0 = time.perf_counter()
            report = self.lib.workspace_run_group_bound(
                self.workspace, flat, nrepeat)
            elapsed_us = (time.perf_counter() - t0) * 1e6
            return report, elapsed_us
        finally:
            self._descriptor_lock.release()

    def run(self, inputs: list[np.ndarray]) -> tuple[list[np.ndarray], Any, float]:
        out_arrays = [np.empty(tuple(b.shape), dtype=np.float16)
                      for b in self.outputs]
        report, elapsed_us = self.run_into(inputs, out_arrays)
        return out_arrays, report, elapsed_us

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            self.close()
        except BaseException as cleanup:
            if exc is not None:
                exc.add_note("PlanCall cleanup also failed: " + repr(cleanup))
                return False
            raise
        return False

    def close(self) -> None:
        """Release handles once; preserve the first cleanup error if any."""
        if self._closed:
            return
        errors: list[BaseException] = []
        if self.workspace is not None:
            try:
                self.workspace.destroy()
            except BaseException as exc:
                # The runtime retains ownership after BUSY/POISONED.  Keep the
                # CPlan and workspace handle intact so a later close can retry.
                errors.append(exc)
            else:
                self.workspace = None
        if not errors and self.cplan is not None:
            try:
                self.cplan.destroy()
            except BaseException as exc:
                errors.append(exc)
            else:
                self.cplan = None
        if not errors:
            self._closed = True
        if errors:
            raise errors[0]


class HybridModelRunner:
    """Thin model-safe wrapper around the versioned hybrid graph executor.

    There is no second Python orchestration path: preprocessing, bridge layout,
    every matrix RKPlan and postprocessing are nodes in the manifest.
    """

    def __init__(self, lib: LibRKPlan, program: Any,
                 weights: dict[str, np.ndarray], batch: int, *,
                 weights_path: str | Path | None = None, fd: int = 0,
                 execution_mode: str = "auto"):
        from .hybrid import HybridRunner, HybridProgram, load_hybrid_program
        if isinstance(program, (str, Path)):
            program = load_hybrid_program(program)
        if not isinstance(program, HybridProgram):
            reject(RK_E_UNSUPPORTED,
                   "model runner requires a versioned rkhybrid manifest",
                   stage="model")
        if int(batch) != program.batch:
            reject(RK_E_SHAPE, "raw batch does not match hybrid profile",
                   stage="model", actual=int(batch), required=program.batch)
        self.program = program
        self.batch = int(batch)
        self.runner = HybridRunner(lib, program, weights,
                                   weights_path=weights_path, fd=fd,
                                   execution_mode=execution_mode)
        self.last_trace: dict[str, np.ndarray] = {}

    def run_result(self, raw_images: np.ndarray, *, stats: RunStats | None = None,
                   capture: bool = False):
        result = self.runner.run(raw_images, stats=stats, capture=capture)
        self.last_trace = result.trace
        return result

    def close(self) -> None:
        self.runner.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return self.runner.__exit__(exc_type, exc, tb)


class MLPRunner(HybridModelRunner):
    """Compatibility facade returning the public ``[B,10]`` logits array."""

    def run(self, raw_images: np.ndarray, *, stats: RunStats | None = None
            ) -> np.ndarray:
        return self.run_result(raw_images, stats=stats).outputs["logits"]


class TinyViTRunner(HybridModelRunner):
    """Compatibility facade returning the public ``[B,10]`` logits array."""

    def run(self, raw_images: np.ndarray, *, stats: RunStats | None = None
            ) -> np.ndarray:
        return self.run_result(raw_images, stats=stats).outputs["logits"]


def load_weight_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}
