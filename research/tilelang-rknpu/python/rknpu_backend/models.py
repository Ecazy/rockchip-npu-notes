# SPDX-License-Identifier: GPL-3.0-or-later
"""Model-level runners for the frozen MNIST MLP and hybrid Tiny Transformer.

The runners only orchestrate already-compiled RKPlan files (or compile them from
real TileLang sources on the Host).  All matrix computation is delegated to the
production ``rkplan`` C ABI; the Transformer's first-round RMSNorm, stable
softmax, layout bridge and mean pooling are explicit Host work.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .errors import RK_E_INVALID, RK_E_SHAPE, RK_E_UNSUPPORTED, reject
from .plan import Plan
from .runtime.ffi import (CWorkspace, LibRKPlan, make_binding,
                          validate_numpy_binding)

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
    plan_call_us: float = 0.0
    plan_pack_us: float = 0.0
    plan_emit_us: float = 0.0
    plan_submit_wait_us: float = 0.0
    plan_output_sync_us: float = 0.0
    upload_calls: int = 0
    upload_bytes: int = 0
    download_calls: int = 0
    download_bytes: int = 0
    task_count: int = 0
    submit_count: int = 0
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
                        outputs: tuple[np.ndarray, ...] = ()) -> None:
        """Record one registered Host boundary action and its Host bytes."""
        phase = self._phase(phase)
        nbytes = _nbytes(tuple(inputs) + tuple(outputs))
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

    def add_report(self, name: str, report: Any, elapsed_us: float,
                   *, phase: str = "execution") -> None:
        """Record one production RKPlan call as an ``execution`` node."""
        phase = self._phase(phase)
        self.plan_call_us += float(elapsed_us)
        self.plan_pack_us += float(report.pack_us)
        self.plan_emit_us += float(report.emit_us)
        self.plan_submit_wait_us += float(report.submit_wait_us)
        self.plan_output_sync_us += float(report.output_sync_us)
        self.upload_calls += int(report.upload_calls)
        self.upload_bytes += int(report.upload_bytes)
        self.download_calls += int(report.download_calls)
        self.download_bytes += int(report.download_bytes)
        self.task_count += int(report.task_count)
        self.submit_count += int(report.submit_count)
        self._phase_add(phase, elapsed_us)
        # Aggregate repeated calls so a full test-set evaluation has a bounded
        # report (one row per logical plan) rather than one row per batch.
        row = next((r for r in self.plans if r["name"] == name), None)
        if row is None:
            row = {"name": name, "kind": "rkplan", "phase": phase, "calls": 0,
                   "elapsed_us": 0.0,
                   "pack_us": 0.0, "emit_us": 0.0,
                   "submit_wait_us": 0.0, "output_sync_us": 0.0,
                   "upload_calls": 0, "upload_bytes": 0,
                   "download_calls": 0, "download_bytes": 0,
                   "task_count": 0, "submit_count": 0,
                   "bridge_bytes": 0}
            self.plans.append(row)
        row["calls"] += 1
        row["elapsed_us"] += float(elapsed_us)
        row["pack_us"] += float(report.pack_us)
        row["emit_us"] += float(report.emit_us)
        row["submit_wait_us"] += float(report.submit_wait_us)
        row["output_sync_us"] += float(report.output_sync_us)
        row["upload_calls"] += int(report.upload_calls)
        row["upload_bytes"] += int(report.upload_bytes)
        row["download_calls"] += int(report.download_calls)
        row["download_bytes"] += int(report.download_bytes)
        row["task_count"] += int(report.task_count)
        row["submit_count"] += int(report.submit_count)
        row["bridge_bytes"] = (int(row["upload_bytes"]) +
                               int(row["download_bytes"]))

    def record_run(self, end_to_end_us: float) -> None:
        self.run_calls += 1
        self.end_to_end_us += float(end_to_end_us)

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
            "plan_call_us": self.plan_call_us,
            "plan_pack_us": self.plan_pack_us,
            "plan_emit_us": self.plan_emit_us,
            "plan_submit_wait_us": self.plan_submit_wait_us,
            "plan_output_sync_us": self.plan_output_sync_us,
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
            "bridge_bytes": self.upload_bytes + self.download_bytes,
            "plans": list(self.plans),
            "host_actions": list(self.host_actions),
        }

class PlanCall:
    """Reusable workspace for one immutable plan and its constants.

    Construction validates/materialises every Host constant before asking C to
    create the plan.  If a later C call fails, already-created handles are
    destroyed without replacing the original exception.  Destruction is
    idempotent; a failed workspace destroy keeps both workspace and CPlan alive
    because the runtime owns quarantine retry state.
    """

    def __init__(self, lib: LibRKPlan, plan: Plan, constants: dict[str, np.ndarray],
                 *, fd: int = 0):
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
            self.cplan = lib.create_plan(plan)
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

    def _prepare_inputs(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
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
            prepared.append(validate_numpy_binding(
                arr, tuple(spec.shape), writable=False, name=spec.name))
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
        if self.workspace is None:
            reject(RK_E_INVALID, "model plan workspace is already closed",
                   stage="model")
        in_arrays = self._prepare_inputs(inputs)
        out_arrays = self._prepare_outputs(outputs)
        in_bind = [make_binding(spec.id, arr)
                   for spec, arr in zip(self.inputs, in_arrays)]
        out_bind = [make_binding(spec.id, arr)
                    for spec, arr in zip(self.outputs, out_arrays)]
        # The mock harness snapshots each emitted co_program.  Clear that
        # test-only cache between calls; production builds have no such symbol.
        if hasattr(self.lib.cdll, "rk_mock_reset_tags"):
            self.lib.mock_control("rk_mock_reset_tags")
        t0 = time.perf_counter()
        report = self.workspace.run(in_bind, out_bind)
        elapsed_us = (time.perf_counter() - t0) * 1e6
        return report, elapsed_us

    def run(self, inputs: list[np.ndarray]) -> tuple[list[np.ndarray], Any, float]:
        out_arrays = [np.empty(tuple(b.shape), dtype=np.float16)
                      for b in self.outputs]
        report, elapsed_us = self.run_into(inputs, out_arrays)
        return out_arrays, report, elapsed_us

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
                 weights_path: str | Path | None = None, fd: int = 0):
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
                                   weights_path=weights_path, fd=fd)
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
