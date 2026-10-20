# SPDX-License-Identifier: GPL-3.0-or-later
"""Runtime bindings and the caller-fd :class:`Workspace` lifecycle."""
from __future__ import annotations

import ctypes
from typing import Any

import numpy as np

from ..errors import (RK_E_INVALID, RK_E_UNSUPPORTED, BackendError,
                      Diagnostic, reject)
from ..plan import Plan
from .ffi import (RKBinding, RKDiag, RKRunReport, default_library,
                  make_binding, validate_numpy_binding)


class Workspace:
    def __init__(self, plan: Plan, fd: int, *, _lib=None, _ptr=None):
        self.plan = plan
        self.fd = fd
        self._lib = _lib
        self._ptr = _ptr
        self._closed = False

    @classmethod
    def create(cls, plan: Plan, *, fd: int,
               constants: dict[str, np.ndarray]) -> "Workspace":
        if not isinstance(plan, Plan):
            reject(RK_E_INVALID, "Workspace.create expects a Plan", stage="runtime")
        if not isinstance(fd, int) or isinstance(fd, bool) or fd < 0:
            reject(RK_E_INVALID, "Workspace.create requires a caller-owned fd>=0",
                   stage="runtime")
        lib = default_library()
        cplan = lib.create_plan(plan)
        try:
            prepared: list[np.ndarray] = []
            for spec in plan.constants:
                if spec.generator == "identity":
                    if spec.shape[0] != spec.shape[1]:
                        reject(RK_E_INVALID, "identity constant must be square",
                               stage="runtime")
                    array = np.eye(spec.shape[0], dtype=np.float16)
                else:
                    if spec.name not in constants:
                        reject(RK_E_INVALID,
                               f"missing runtime constant {spec.name!r}",
                               stage="runtime")
                    array = constants[spec.name]
                array = validate_numpy_binding(array, spec.shape,
                                               writable=False, name=spec.name)
                prepared.append(array)
            bindings = (RKBinding * len(prepared))()
            for i, (spec, array) in enumerate(zip(plan.constants, prepared)):
                bindings[i] = make_binding(spec.binding_id, array)
            diag = RKDiag()
            out = ctypes.c_void_p()
            rc = lib.fn.rk_workspace_create(fd, cplan._ptr,
                                            bindings if prepared else None,
                                            len(prepared), ctypes.byref(out),
                                            ctypes.byref(diag))
            if rc:
                from .ffi import _diag_to_exception
                raise _diag_to_exception(diag, "workspace_create")
            if not out.value:
                reject(RK_E_INVALID, "rk_workspace_create returned no handle",
                       stage="runtime")
            return cls(plan, fd, _lib=lib, _ptr=out)
        finally:
            cplan.destroy()

    def run(self, inputs: dict[str, np.ndarray], *,
            outputs: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
        if self._closed or not self._ptr:
            reject(RK_E_INVALID, "workspace is closed", stage="runtime")
        if not isinstance(inputs, dict):
            reject(RK_E_INVALID, "inputs must be a dict", stage="runtime")
        input_bindings = [b for b in self.plan.bindings if b.role == "input"]
        output_bindings = [b for b in self.plan.bindings if b.role == "output"]
        if set(inputs) != {b.name for b in input_bindings}:
            reject(RK_E_INVALID, "inputs must exactly match plan input names",
                   stage="runtime",
                   actual=sorted(inputs), required=sorted(b.name for b in input_bindings))
        checked_inputs: list[np.ndarray] = []
        for binding in input_bindings:
            checked_inputs.append(validate_numpy_binding(
                inputs[binding.name], binding.shape, writable=False,
                name=binding.name))
        if outputs is None:
            result = {b.name: np.empty(b.shape, dtype=np.float16)
                      for b in output_bindings}
        else:
            if set(outputs) != {b.name for b in output_bindings}:
                reject(RK_E_INVALID,
                       "outputs must exactly match plan output names",
                       stage="runtime", actual=sorted(outputs),
                       required=sorted(b.name for b in output_bindings))
            result = {}
            for binding in output_bindings:
                result[binding.name] = validate_numpy_binding(
                    outputs[binding.name], binding.shape, writable=True,
                    name=binding.name)
        for a in checked_inputs:
            for b in result.values():
                if np.shares_memory(a, b):
                    reject(RK_E_INVALID, "input/output host buffers overlap",
                           stage="runtime")
        out_names = [b.name for b in output_bindings]
        for i, a in enumerate(out_names):
            for b in out_names[:i]:
                if np.shares_memory(result[a], result[b]):
                    reject(RK_E_INVALID, "output host buffers overlap",
                           stage="runtime")
        input_arrays = (RKBinding * len(input_bindings))()
        for i, binding in enumerate(input_bindings):
            input_arrays[i] = make_binding(binding.id, inputs[binding.name])
        output_arrays = (RKBinding * len(output_bindings))()
        for i, binding in enumerate(output_bindings):
            output_arrays[i] = make_binding(binding.id, result[binding.name])
        report = RKRunReport()
        diag = RKDiag()
        rc = self._lib.fn.rk_workspace_run(
            self._ptr, input_arrays if input_bindings else None,
            len(input_bindings),
            output_arrays if output_bindings else None,
            len(output_bindings), ctypes.byref(report), ctypes.byref(diag))
        if rc:
            from .ffi import _diag_to_exception
            raise _diag_to_exception(diag, "workspace_run")
        self.last_report = {
            "task_count": int(report.task_count),
            "submit_count": int(report.submit_count),
            "upload_calls": int(report.upload_calls),
            "upload_bytes": int(report.upload_bytes),
            "download_calls": int(report.download_calls),
            "download_bytes": int(report.download_bytes),
            "pack_us": float(report.pack_us),
            "emit_us": float(report.emit_us),
            "submit_wait_us": float(report.submit_wait_us),
            "output_sync_us": float(report.output_sync_us),
            "numeric_mode": "rk_f16_tiled",
        }
        return result

    def close(self) -> None:
        if self._closed:
            return
        if not self._ptr:
            self._closed = True
            return
        diag = RKDiag()
        rc = self._lib.fn.rk_workspace_destroy(self._ptr, ctypes.byref(diag))
        if rc:
            from .ffi import _diag_to_exception
            # The C contract returns BUSY/POISONED with ownership retained.
            raise _diag_to_exception(diag, "workspace_destroy")
        self._ptr = None
        self._closed = True

    def __enter__(self) -> "Workspace":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
