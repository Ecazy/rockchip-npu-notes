# SPDX-License-Identifier: GPL-3.0-or-later
"""ctypes bridge for the public ``rk_plan.h`` C ABI.

The Python layer constructs a plan descriptor, calls ``rk_plan_create`` and lets
the C side independently validate/deep-copy it.  It never writes a device pointer
into JSON and never treats a Python ``verified`` flag as authority.

The RKPlan descriptor *encoding* lives in :mod:`rknpu_backend.codegen.descriptor`
so codegen and runtime share exactly one implementation.  Its ctypes types and ABI
constants are re-exported here because this module is the long-standing public
spelling used by the board campaign tools.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
import struct
from pathlib import Path
from typing import Any

import numpy as np

from ..codegen.descriptor import (BUFFER_ROLE_MAP, KIND_MAP, LAYOUT_MAP,
                                  RK_ABI_VERSION, RK_ACTION_COMMIT,
                                  RK_ACTION_DOWNLOAD,
                                  RK_ACTION_FLAG_BROADCAST_CHANNEL,
                                  RK_ACTION_FLAG_BROADCAST_SCALAR,
                                  RK_ACTION_FLAG_CACHE_CONSTANT,
                                  RK_ACTION_FLAG_TRANSPOSE_SOURCE,
                                  RK_ACTION_HOST_CHECK, RK_ACTION_HOST_COMPUTE,
                                  RK_ACTION_HOST_EXPAND, RK_ACTION_HOST_PACK,
                                  RK_ACTION_NPU_TASK, RK_ACTION_UPLOAD,
                                  RK_CORE_AUTO, RK_CORE_SINGLE, RK_DTYPE_F16,
                                  RK_LAYOUT_CUBE_F16, RK_LAYOUT_LINEAR_F16,
                                  RK_LAYOUT_WEIGHT_F16, RK_ROLE_ACCUMULATOR,
                                  RK_ROLE_CONSTANT, RK_ROLE_GATE, RK_ROLE_IDENTITY,
                                  RK_ROLE_INPUT, RK_ROLE_INPUT_SLICE,
                                  RK_ROLE_OUTPUT, RK_ROLE_SCRATCH, RK_ROLE_STATE,
                                  RK_ROLE_WEIGHT, RK_SCHEMA_VERSION,
                                  RK_SUBMIT_CHAIN, RK_SUBMIT_SEPARATE,
                                  RK_TEMPLATE_EW_ADD_F16, RK_TEMPLATE_EW_MAX_F16,
                                  RK_TEMPLATE_EW_MUL_F16,
                                  RK_TEMPLATE_LUT_GELU_GATE_F16,
                                  RK_TEMPLATE_LUT_HARDSIGMOID_F16,
                                  RK_TEMPLATE_LUT_SIGMOID_F16,
                                  RK_TEMPLATE_MM_ACC_F16, RK_TEMPLATE_MM_F16,
                                  TEMPLATE_MAP, RKActionDesc, RKBufferDesc,
                                  RKPlanDesc, RKPublishDesc, RKSubmitDesc,
                                  RKTensorDesc, RKValueDesc, RKViewDesc,
                                  build_plan_descriptor)
from ..errors import (RK_E_INVALID, RK_E_NOMEM, RK_E_UNSUPPORTED, BackendError,
                      Diagnostic, reject)
from ..plan import Plan


class RKBinding(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32), ("struct_size", ctypes.c_uint32),
        ("id", ctypes.c_uint32), ("dtype", ctypes.c_uint32),
        ("rank", ctypes.c_uint32), ("shape", ctypes.c_uint32 * 2),
        ("data", ctypes.c_void_p), ("nbytes", ctypes.c_uint64),
    ]


class RKRunReport(ctypes.Structure):
    """Mirror of ``rk_run_report``: the frozen 88-byte legacy report.

    ``rk_workspace_run`` writes exactly this struct and never consults
    ``struct_size``, so a caller from before the attribution extension keeps
    working without initialising anything.  Extended attribution is requested
    explicitly through ``RKAttribution`` and ``rk_workspace_run_ex``.
    """

    _fields_ = [
        ("abi_version", ctypes.c_uint32), ("struct_size", ctypes.c_uint32),
        ("task_count", ctypes.c_uint32), ("submit_count", ctypes.c_uint32),
        ("fail_action_id", ctypes.c_uint32), ("numeric_mode", ctypes.c_uint32),
        ("upload_calls", ctypes.c_uint64), ("upload_bytes", ctypes.c_uint64),
        ("download_calls", ctypes.c_uint64), ("download_bytes", ctypes.c_uint64),
        ("pack_us", ctypes.c_double), ("emit_us", ctypes.c_double),
        ("submit_wait_us", ctypes.c_double), ("output_sync_us", ctypes.c_double),
    ]


class RKAttribution(ctypes.Structure):
    """Mirror of ``rk_run_attribution`` (the ``rk_workspace_run_ex`` report).

    Every interval is disjoint and inside the enclosing call, so their sum plus
    the residual accounts for the whole dwell.  ``struct_size`` is mandatory and
    the C side refuses to run when it is not a usable declaration.
    """

    _fields_ = [
        ("abi_version", ctypes.c_uint32), ("struct_size", ctypes.c_uint32),
        ("task_count", ctypes.c_uint32), ("submit_count", ctypes.c_uint32),
        ("fail_action_id", ctypes.c_uint32), ("numeric_mode", ctypes.c_uint32),
        ("upload_calls", ctypes.c_uint64), ("upload_bytes", ctypes.c_uint64),
        ("download_calls", ctypes.c_uint64), ("download_bytes", ctypes.c_uint64),
        ("validate_us", ctypes.c_double), ("pack_us", ctypes.c_double),
        ("emit_us", ctypes.c_double), ("command_us", ctypes.c_double),
        ("submit_alloc_us", ctypes.c_double),
        ("submit_wait_us", ctypes.c_double),
        ("output_sync_us", ctypes.c_double), ("download_us", ctypes.c_double),
        ("guard_us", ctypes.c_double), ("publish_us", ctypes.c_double),
        ("alloc_us", ctypes.c_double), ("total_us", ctypes.c_double),
        ("clock_us", ctypes.c_double),
        ("pack_prep_us", ctypes.c_double),
        ("pack_clear_us", ctypes.c_double),
        ("pack_layout_us", ctypes.c_double),
        ("pack_fini_us", ctypes.c_double),
        ("pack_prep_calls", ctypes.c_uint64),
        ("pack_fini_calls", ctypes.c_uint64),
        ("upload_coalesce", ctypes.c_uint32),
        ("upload_fast_pack", ctypes.c_uint32),
        ("upload_policy_reserved", ctypes.c_uint32 * 2),
    ]


class RKGroupReport(ctypes.Structure):
    """Mirror of ``rk_group_report`` for the static-repeat group entry."""

    _fields_ = [
        ("abi_version", ctypes.c_uint32), ("struct_size", ctypes.c_uint32),
        ("repeat_count", ctypes.c_uint32), ("repeat_completed", ctypes.c_uint32),
        ("fail_repeat", ctypes.c_uint32), ("fail_action_id", ctypes.c_uint32),
        ("task_count", ctypes.c_uint32), ("submit_count", ctypes.c_uint32),
        ("numeric_mode", ctypes.c_uint32), ("reserved0", ctypes.c_uint32),
        ("upload_calls", ctypes.c_uint64), ("upload_bytes", ctypes.c_uint64),
        ("download_calls", ctypes.c_uint64), ("download_bytes", ctypes.c_uint64),
        ("validate_us", ctypes.c_double), ("pack_us", ctypes.c_double),
        ("emit_us", ctypes.c_double), ("command_us", ctypes.c_double),
        ("submit_alloc_us", ctypes.c_double),
        ("submit_wait_us", ctypes.c_double),
        ("output_sync_us", ctypes.c_double), ("download_us", ctypes.c_double),
        ("guard_us", ctypes.c_double), ("publish_us", ctypes.c_double),
        ("alloc_us", ctypes.c_double), ("total_us", ctypes.c_double),
        ("clock_us", ctypes.c_double),
    ]


def _new_attribution() -> "RKAttribution":
    """Zero-initialised attribution with this process's declared size.

    ``struct_size`` is mandatory for ``rk_workspace_run_ex``: the C side
    validates it and refuses to run rather than guessing, so a wrong value can
    never overrun the buffer.
    """
    report = RKAttribution()
    report.struct_size = ctypes.sizeof(RKAttribution)
    return report


class RKDiag(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32), ("struct_size", ctypes.c_uint32),
        ("code", ctypes.c_int32), ("native_status", ctypes.c_int32),
        ("action_id", ctypes.c_uint32), ("has_action", ctypes.c_uint32),
        ("stage", ctypes.c_char * 32), ("message", ctypes.c_char * 512),
        ("source", ctypes.c_char * 128), ("profile", ctypes.c_char * 64),
    ]




def _diag_to_exception(diag: RKDiag, stage: str = "c_abi") -> BackendError:
    def text(field: Any) -> str:
        return bytes(field).split(b"\0", 1)[0].decode("utf-8", "replace")

    action_id = int(diag.action_id) if diag.has_action else None
    return BackendError(
        int(diag.code) if diag.code else RK_E_INVALID,
        text(diag.message) or "C ABI call failed",
        stage=text(diag.stage) or stage,
        source=text(diag.source), action_id=action_id,
        native_status=int(diag.native_status) if diag.native_status else None)


class CPlan:
    def __init__(self, lib: "LibRKPlan", ptr: Any):
        self._lib = lib
        self._ptr = ptr

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass

    def destroy(self) -> None:
        if self._ptr:
            self._lib.fn.rk_plan_destroy(self._ptr)
            self._ptr = None


class LibRKPlan:
    def __init__(self, path: str | os.PathLike[str] | None = None):
        self.path = str(path) if path else find_library()
        try:
            self.cdll = ctypes.CDLL(self.path)
        except OSError as exc:
            raise BackendError(
                RK_E_UNSUPPORTED,
                f"cannot load rknpu_plan shared library: {exc}",
                stage="ffi",
                alternative="build research/custom-ops target rknpu_plan and set "
                            "RKNPU_PLAN_LIB") from exc
        self._bind()

    def _bind(self) -> None:
        fn = self.cdll
        fn.rk_abi_version.argtypes = []
        fn.rk_abi_version.restype = ctypes.c_uint32
        fn.rk_plan_create.argtypes = [ctypes.POINTER(RKPlanDesc),
                                      ctypes.POINTER(ctypes.c_void_p),
                                      ctypes.POINTER(RKDiag)]
        fn.rk_plan_create.restype = ctypes.c_int
        fn.rk_plan_destroy.argtypes = [ctypes.c_void_p]
        fn.rk_plan_destroy.restype = None
        fn.rk_workspace_create.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(RKBinding),
            ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(RKDiag)]
        fn.rk_workspace_create.restype = ctypes.c_int
        fn.rk_workspace_run.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(RKBinding), ctypes.c_uint32,
            ctypes.POINTER(RKBinding), ctypes.c_uint32,
            ctypes.POINTER(RKRunReport), ctypes.POINTER(RKDiag)]
        fn.rk_workspace_run.restype = ctypes.c_int
        fn.rk_workspace_run_ex.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(RKBinding), ctypes.c_uint32,
            ctypes.POINTER(RKBinding), ctypes.c_uint32,
            ctypes.POINTER(RKAttribution), ctypes.POINTER(RKDiag)]
        fn.rk_workspace_run_ex.restype = ctypes.c_int
        fn.rk_workspace_run_group.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(RKBinding), ctypes.c_uint32,
            ctypes.POINTER(RKGroupReport), ctypes.POINTER(RKDiag)]
        fn.rk_workspace_run_group.restype = ctypes.c_int
        fn.rk_workspace_destroy.argtypes = [ctypes.c_void_p,
                                            ctypes.POINTER(RKDiag)]
        fn.rk_workspace_destroy.restype = ctypes.c_int
        self.fn = fn
        if self.abi_version() != RK_ABI_VERSION:
            reject(RK_E_UNSUPPORTED, "rknpu_plan ABI version mismatch",
                   stage="ffi", actual=self.abi_version(), required=RK_ABI_VERSION)

    def abi_version(self) -> int:
        return int(self.cdll.rk_abi_version())

    def _missing(self, name: str) -> bool:
        return not hasattr(self.cdll, name)

    _MOCK_SIGNATURES = {
        "rk_mock_execs": (ctypes.c_uint, ()),
        "rk_mock_live_bos": (ctypes.c_int, ()),
        "rk_mock_resident": (ctypes.c_size_t, ()),
        "rk_mock_fail_at": (None, (ctypes.c_uint,)),
        "rk_mock_inject_nan": (None, (ctypes.c_uint,)),
        "rk_mock_corrupt_guard": (None, (ctypes.c_uint,)),
        "rk_mock_corrupt_guard_at": (None, (ctypes.c_uint,)),
        "rk_mock_delay_us": (None, (ctypes.c_uint,)),
        "rk_mock_fail_prep_at": (None, (ctypes.c_uint,)),
        "rk_mock_prep_calls": (ctypes.c_uint, ()),
        "rk_mock_fail_fini_at": (None, (ctypes.c_uint,)),
        "rk_mock_fini_calls": (ctypes.c_uint, ()),
        "rk_mock_reset_tags": (None, ()),
        "rk_mock_context_fault": (None, (ctypes.c_uint, ctypes.c_uint)),
        "rk_mock_context_faulted": (ctypes.c_int, (ctypes.c_int,)),
        "rk_mock_reset": (None, ()),
        "rk_mock_allocs": (ctypes.c_uint, ()),
        "rk_mock_tag_count": (ctypes.c_uint, ()),
        "rk_mock_tag_overflow": (ctypes.c_uint, ()),
        "rk_mock_untracked": (ctypes.c_uint, ()),
        "rk_mock_bo_count": (ctypes.c_uint, ()),
        "rk_mock_tag_capacity": (ctypes.c_uint, ()),
        "rk_mock_fail_alloc": (None, (ctypes.c_uint,)),
        "rk_mock_force_tag_overflow": (None, (ctypes.c_uint,)),
        "rk_mock_dma_of": (ctypes.c_uint32, (ctypes.c_uint32,)),
        "rk_mock_bo_dma_address": (ctypes.c_uint64, (ctypes.c_uint32,)),
        "rk_mock_dma_range": (ctypes.c_int, (ctypes.c_uint, ctypes.POINTER(ctypes.c_uint32),
                                             ctypes.POINTER(ctypes.c_uint32),
                                             ctypes.POINTER(ctypes.c_size_t))),
        "rk_mock_find_half": (ctypes.c_int, (ctypes.c_uint16, ctypes.POINTER(ctypes.c_uint),
                                             ctypes.POINTER(ctypes.c_size_t))),
    }

    #: Every rk_mock_* export with its exact ctypes signature.  The C side is
    #: checked against this at compile time by tests/mock/mock_signature_probe.c
    #: (which pins the same prototypes from mock_npu.h); nothing parses C at run
    #: time.
    def mock_exports(self) -> dict:
        return dict(self._MOCK_SIGNATURES)

    def mock_control(self, name: str, *args):
        """Call a test-only ``rk_mock_*`` symbol if this library exposes one."""
        fn = getattr(self.cdll, name, None)
        if fn is not None:
            restype, argtypes = self._MOCK_SIGNATURES[name]
            fn.restype = restype
            fn.argtypes = argtypes
        if fn is None:
            raise RuntimeError(
                f"{name} is not exported by {self.path!r}; use "
                "librknpu_plan_mock.so for execution tests")
        return fn(*args)

    def create_workspace(self, plan: CPlan, constants: list, fd: int = 0) -> CWorkspace:
        for name in ("rk_workspace_create", "rk_workspace_run",
                     "rk_workspace_destroy"):
            if self._missing(name):
                raise RuntimeError(f"library does not export {name}")
        arr = (RKBinding * max(len(constants), 1))()
        for i, binding in enumerate(constants):
            arr[i] = binding
        handle = ctypes.c_void_p()
        diag = RKDiag()
        rc = self.cdll.rk_workspace_create(fd, plan._ptr, arr, len(constants),
                                           ctypes.byref(handle), ctypes.byref(diag))
        if rc:
            raise _diag_to_exception(diag, "workspace_create")
        return CWorkspace(lib=self, handle=handle)

    @staticmethod
    def _binding_array(bindings: list):
        arr = (RKBinding * max(len(bindings), 1))()
        for i, binding in enumerate(bindings):
            arr[i] = binding
        return arr

    def workspace_run(self, workspace: CWorkspace, inputs: list, outputs: list):
        """Legacy single-run entry: the frozen 88-byte report, no attribution."""
        in_arr = self._binding_array(inputs)
        out_arr = self._binding_array(outputs)
        report = RKRunReport()
        diag = RKDiag()
        rc = self.cdll.rk_workspace_run(workspace.handle, in_arr, len(inputs),
                                        out_arr, len(outputs),
                                        ctypes.byref(report), ctypes.byref(diag))
        if rc:
            raise _diag_to_exception(diag, "workspace_run")
        return report

    def workspace_run_attributed(self, workspace: CWorkspace, inputs: list,
                                 outputs: list):
        """Single run with envelope attribution (explicit ``_ex`` entry)."""
        in_arr = self._binding_array(inputs)
        out_arr = self._binding_array(outputs)
        return self.workspace_run_attributed_bound(
            workspace, in_arr, out_arr, len(inputs), len(outputs))

    def workspace_run_attributed_bound(self, workspace: CWorkspace,
                                       in_arr, out_arr,
                                       ninputs: int, noutputs: int):
        """Attributed run with caller-owned, already packed binding arrays.

        A compiled runner may populate the arrays once after proving that the
        backing Host buffers and all their views are stable.  The C workspace
        still validates every binding and every finite input on every call; this
        entry only removes duplicate Python descriptor construction.
        """
        report = _new_attribution()
        diag = RKDiag()
        rc = self.cdll.rk_workspace_run_ex(
            workspace.handle, in_arr if ninputs else None, int(ninputs),
            out_arr if noutputs else None, int(noutputs),
            ctypes.byref(report), ctypes.byref(diag))
        if rc:
            raise _diag_to_exception(diag, "workspace_run_ex")
        return report

    def workspace_run_group(self, workspace: CWorkspace, bindings: list,
                            nrepeat: int):
        """One native entry for ``nrepeat`` independent binding sets.

        ``bindings`` is repeat-major: repeat 0 inputs then outputs, repeat 1
        inputs then outputs, ...  The C side performs every per-repeat check and
        submit, then publishes all repeats only after the shared guard gate.
        """
        count = len(bindings)
        arr = (RKBinding * max(count, 1))()
        for i, binding in enumerate(bindings):
            arr[i] = binding
        return self.workspace_run_group_bound(workspace, arr, nrepeat)

    def workspace_run_group_bound(self, workspace: CWorkspace, arr,
                                  nrepeat: int):
        """Group entry with a prebuilt repeat-major binding array."""
        report = RKGroupReport()
        report.struct_size = ctypes.sizeof(RKGroupReport)
        diag = RKDiag()
        rc = self.cdll.rk_workspace_run_group(
            workspace.handle, arr, int(nrepeat), ctypes.byref(report),
            ctypes.byref(diag))
        if rc:
            raise _diag_to_exception(diag, "workspace_run_group")
        return report

    def workspace_destroy(self, workspace: CWorkspace) -> None:
        diag = RKDiag()
        rc = self.cdll.rk_workspace_destroy(workspace.handle, ctypes.byref(diag))
        if rc:
            # BUSY / POISONED keep ownership: the caller must retry, so the handle
            # must stay valid instead of being dropped here.
            raise _diag_to_exception(diag, "workspace_destroy")
        workspace.handle = ctypes.c_void_p()

    def create_plan(self, plan: Plan) -> CPlan:
        desc, holder = build_plan_descriptor(plan)
        diag = RKDiag()
        out = ctypes.c_void_p()
        rc = self.fn.rk_plan_create(ctypes.byref(desc), ctypes.byref(out),
                                    ctypes.byref(diag))
        # holder is intentionally alive through this call; C deep-copies.
        _ = holder
        if rc:
            raise _diag_to_exception(diag, "plan_create")
        if not out.value:
            raise BackendError(RK_E_INVALID, "rk_plan_create returned no handle",
                               stage="ffi")
        return CPlan(self, out)


def find_library() -> str:
    env = os.environ.get("RKNPU_PLAN_LIB")
    if env:
        return env
    root = Path(__file__).resolve().parents[3]
    candidates = [
        root.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan.so",
        root.parent / "custom-ops" / "build-host" / "librknpu_plan.so",
        root.parent / "custom-ops" / "build" / "librknpu_plan.so",
        Path.cwd() / "librknpu_plan.so",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    try:
        import ctypes.util
        found = ctypes.util.find_library("rknpu_plan")
        if found:
            return found
    except Exception:
        pass
    reject(RK_E_UNSUPPORTED,
           "rknpu_plan shared library was not found",
           stage="ffi",
           alternative="cmake --build research/custom-ops/build-host-mvp "
                       "--target rknpu_plan and set RKNPU_PLAN_LIB")


_DEFAULT_LIB: LibRKPlan | None = None


@dataclass
class CWorkspace:
    """Wraps ``rk_workspace *``; the mock library exposes run counters."""

    lib: "LibRKPlan"
    handle: ctypes.c_void_p

    def run(self, inputs, outputs):
        """Legacy report-only entry (no attribution)."""
        return self.lib.workspace_run(self, inputs, outputs)

    def run_attributed(self, inputs, outputs):
        """Single run plus envelope attribution."""
        return self.lib.workspace_run_attributed(self, inputs, outputs)

    def run_group(self, bindings, nrepeat: int):
        return self.lib.workspace_run_group(self, bindings, nrepeat)

    def destroy(self) -> None:
        if not self.handle:
            return  # already released: cleanup must be idempotent
        self.lib.workspace_destroy(self)


def configure_rocket_abi(cdll) -> None:
    """Apply the vendor header's real ABI to ctypes declarations.

    ``rocket_open`` returns int; ``rocket_close`` is declared void and has no
    status result.  Callers must not infer closure success from a fabricated
    return value.
    """
    cdll.rocket_open.argtypes = []
    cdll.rocket_open.restype = ctypes.c_int
    cdll.rocket_close.argtypes = [ctypes.c_int]
    cdll.rocket_close.restype = None
    if hasattr(cdll, "rocket_context_faulted"):
        cdll.rocket_context_faulted.argtypes = [ctypes.c_int]
        cdll.rocket_context_faulted.restype = ctypes.c_int


def default_library() -> LibRKPlan:
    global _DEFAULT_LIB
    if _DEFAULT_LIB is None:
        _DEFAULT_LIB = LibRKPlan()
    return _DEFAULT_LIB


def make_binding(binding_id: int, array: np.ndarray) -> RKBinding:
    if array.ndim > 2:
        reject(RK_E_INVALID, "bindings must be rank 0, 1 or 2",
               stage="binding", actual=array.shape)
    shape = list(array.shape) + [0] * (2 - array.ndim)
    return RKBinding(
        abi_version=RK_ABI_VERSION, struct_size=ctypes.sizeof(RKBinding),
        id=binding_id, dtype=RK_DTYPE_F16, rank=array.ndim,
        shape=(ctypes.c_uint32 * 2)(*shape),
        data=array.ctypes.data_as(ctypes.c_void_p),
        nbytes=array.nbytes)


def validate_numpy_binding(array: Any, shape: tuple[int, int], *,
                           writable: bool, name: str,
                           check_finite: bool = True) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        reject(RK_E_INVALID, f"binding {name!r} must be an exact numpy.ndarray",
               stage="binding")
    if array.dtype != np.dtype("<f2"):
        reject(RK_E_INVALID,
               f"binding {name!r} must be native little-endian FP16; implicit "
               "astype is forbidden", stage="binding", actual=str(array.dtype))
    if array.shape != shape:
        reject(RK_E_INVALID, f"binding {name!r} exact shape is {shape}",
               stage="binding", actual=list(array.shape))
    if not array.flags.c_contiguous:
        reject(RK_E_INVALID, f"binding {name!r} must be C-contiguous",
               stage="binding")
    if writable and not array.flags.writeable:
        reject(RK_E_INVALID, f"binding {name!r} must be writable", stage="binding")
    if not writable and check_finite and not np.isfinite(array).all():
        from ..errors import RK_E_NUMERIC_CONTRACT
        reject(RK_E_NUMERIC_CONTRACT,
               f"binding {name!r} contains non-finite values", stage="binding")
    return array
