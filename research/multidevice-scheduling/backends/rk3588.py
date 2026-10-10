#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""RK3588 backend adapter.

This adapter does not re-implement NPU execution.  It loads the shared library built
from ``bridge/co_backend_bridge.c`` (which links the existing ``custom_ops`` and
``co_schedule`` targets) with :mod:`ctypes`, and forwards every operation to the real
custom-ops entry points.

Execution model: custom-ops submit is *synchronous*.  ``submit`` therefore performs the
blocking ``co_execute`` call and returns a token that is already COMPLETE; ``wait`` only
observes that finished token.  ``ASYNC_COMPLETION`` is reported UNAVAILABLE, and no
timeout is fabricated for an operation that has already returned.

If the bridge library is missing, or the device cannot be opened, the adapter fails
closed: all execution capabilities are UNAVAILABLE and operations raise
:class:`UnsupportedOperation` / :class:`DeviceUnavailable`.  There is no fallback path.

Build the bridge (see bridge/CMakeLists.txt)::

    cmake -S research/multidevice-scheduling/backends/bridge -B build/co-backend-bridge
    cmake --build build/co-backend-bridge -j4
    export CO_BACKEND_BRIDGE=$PWD/build/co-backend-bridge/libco_backend_bridge.so
"""

from __future__ import annotations

import ctypes
import ctypes.util
import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Sequence, Tuple

from .interface import (
    BackendError,
    BaseBackend,
    BufferHandle,
    BufferOwnershipError,
    Capability,
    CapabilityState,
    CompletionToken,
    DeviceIdentity,
    DeviceUnavailable,
    Evidence,
    InvalidPayload,
    JobHandle,
    MemoryBoundsError,
    Payload,
    TokenState,
    UnknownSDKBehavior,
    UnsupportedOperation,
    now_ns,
)

__all__ = [
    "RK3588Payload",
    "RK3588Buffer_t",
    "RK3588Backend",
    "CtypesBridge",
    "BridgeCallError",
    "TensorKind",
    "MODE_SEPARATE",
    "MODE_CHAIN",
    "BRIDGE_ENV",
    "default_bridge_candidates",
]

BRIDGE_ENV = "CO_BACKEND_BRIDGE"
BRIDGE_ABI_VERSION = 2
MODE_SEPARATE = 0
MODE_CHAIN = 1

# rocket_status codes mirrored from rocket_npu.h
ROCKET_OK = 0
ROCKET_E_SHAPE = -1
ROCKET_E_TILING = -2
ROCKET_E_NOMEM = -3
ROCKET_E_DEVICE = -4
ROCKET_E_UNSUPPORTED = -5


class BridgeCallError(BackendError):
    """A bridge call returned a negative rocket_status."""

    def __init__(self, op: str, code: int, name: str) -> None:
        super().__init__(f"bridge {op} failed: {name} ({code})")
        self.op = op
        self.code = code
        self.name = name


class TensorKind(str):
    """Marker values for :meth:`RK3588Backend.configure_tensor`."""

    MATRIX = "matrix"
    INPUT = "input"
    WEIGHT = "weight"
    LINEAR = "linear"


def default_bridge_candidates() -> List[Path]:
    root = Path(__file__).resolve().parents[3]
    names = ["libco_backend_bridge.so", "libco_backend_bridge.dylib"]
    bases = [
        root / "research" / "multidevice-scheduling" / "build-bridge",
        root / "build" / "co-backend-bridge",
        root / "research" / "multidevice-scheduling" / "backends" / "bridge" / "build",
        Path(__file__).resolve().parent / "bridge" / "build",
    ]
    out: List[Path] = []
    for base in bases:
        for name in names:
            out.append(base / name)
    return out


def find_bridge_library(explicit: Optional[str] = None) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    env = os.environ.get(BRIDGE_ENV)
    if env:
        p = Path(env)
        return p if p.exists() else None
    for cand in default_bridge_candidates():
        if cand.exists():
            return cand
    found = ctypes.util.find_library("co_backend_bridge")
    if found:
        return Path(found)
    return None


class CtypesBridge:
    """Typed ctypes wrapper over ``libco_backend_bridge``.

    Every method raises :class:`BridgeCallError` on a negative rocket status and returns
    the decoded value otherwise.  The device fd is passed through by the adapter, so a
    single library handle serves any number of open devices.
    """

    def __init__(self, library_path: Optional[str] = None) -> None:
        path = find_bridge_library(library_path)
        if path is None:
            raise DeviceUnavailable(
                "libco_backend_bridge not found; build it or set " + BRIDGE_ENV
            )
        self.path = path
        self._lib = ctypes.CDLL(str(path))
        self._lib.cobs_abi_version.restype = ctypes.c_uint32
        version = int(self._lib.cobs_abi_version())
        if version != BRIDGE_ABI_VERSION:
            raise UnknownSDKBehavior(
                f"bridge ABI {version} != expected {BRIDGE_ABI_VERSION}"
            )
        try:
            self._bind()
        except AttributeError as e:
            raise UnknownSDKBehavior("bridge missing required ABI symbols") from e

    # -- binding -----------------------------------------------------------------

    def _bind(self) -> None:
        L = self._lib
        L.cobs_abi_version.restype = ctypes.c_uint32
        L.cobs_abi_version.argtypes = []

        L.cobs_device_name.restype = ctypes.c_char_p
        L.cobs_device_name.argtypes = []

        L.cobs_open.restype = ctypes.c_int
        L.cobs_open.argtypes = []

        L.cobs_close.restype = None
        L.cobs_close.argtypes = [ctypes.c_int]

        L.cobs_buffer_alloc.restype = ctypes.c_int
        L.cobs_buffer_alloc.argtypes = [ctypes.c_int, ctypes.c_size_t,
                                        ctypes.POINTER(ctypes.c_uint64)]

        L.cobs_buffer_free.restype = ctypes.c_int
        L.cobs_buffer_free.argtypes = [ctypes.c_uint64]

        L.cobs_buffer_payload.restype = ctypes.c_size_t
        L.cobs_buffer_payload.argtypes = [ctypes.c_uint64]

        L.cobs_buffer_allocated.restype = ctypes.c_size_t
        L.cobs_buffer_allocated.argtypes = [ctypes.c_uint64]

        L.cobs_tensor_bytes.restype = ctypes.c_size_t
        L.cobs_tensor_bytes.argtypes = [ctypes.c_uint64]

        for fn, argtypes in (
            ("cobs_buffer_set_matrix", [ctypes.c_uint64, ctypes.c_uint, ctypes.c_uint,
                                        ctypes.c_int]),
            ("cobs_buffer_set_input", [ctypes.c_uint64, ctypes.c_uint, ctypes.c_uint]),
            ("cobs_buffer_set_weight", [ctypes.c_uint64, ctypes.c_uint, ctypes.c_uint]),
            ("cobs_buffer_set_linear", [ctypes.c_uint64, ctypes.c_uint]),
        ):
            f = getattr(L, fn)
            f.restype = ctypes.c_int
            f.argtypes = argtypes

        L.cobs_upload.restype = ctypes.c_int
        L.cobs_upload.argtypes = [ctypes.c_uint64, ctypes.c_size_t, ctypes.c_char_p,
                                  ctypes.c_size_t]

        L.cobs_download.restype = ctypes.c_int
        L.cobs_download.argtypes = [ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p,
                                    ctypes.c_size_t]

        L.cobs_job_create.restype = ctypes.c_int
        L.cobs_job_create.argtypes = [ctypes.c_uint64, ctypes.c_size_t,
                                      ctypes.POINTER(ctypes.c_uint64)]

        L.cobs_job_add_read.restype = ctypes.c_int
        L.cobs_job_add_read.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t]

        L.cobs_job_set_words.restype = ctypes.c_int
        L.cobs_job_set_words.argtypes = [ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64),
                                         ctypes.c_uint, ctypes.c_uint]

        L.cobs_job_validate.restype = ctypes.c_int
        L.cobs_job_validate.argtypes = [ctypes.c_uint64]

        L.cobs_job_emit_lut.restype = ctypes.c_int
        L.cobs_job_emit_lut.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_uint]
        L.cobs_guard_check.restype = ctypes.c_int
        L.cobs_guard_check.argtypes = [ctypes.c_uint64]

        L.cobs_job_free.restype = None
        L.cobs_job_free.argtypes = [ctypes.c_uint64]

        L.cobs_execute.restype = ctypes.c_int
        L.cobs_execute.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_uint]

        L.cobs_execute_batch.restype = ctypes.c_int
        L.cobs_execute_batch.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint64),
                                         ctypes.c_uint, ctypes.c_uint]

        L.cobs_schedule_execute.restype = ctypes.c_int
        L.cobs_schedule_execute.argtypes = [
            ctypes.c_int, ctypes.c_uint64, ctypes.c_uint, ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]

        L.cobs_status_name.restype = ctypes.c_char_p
        L.cobs_status_name.argtypes = [ctypes.c_int]

    # -- helpers -----------------------------------------------------------------

    def status_name(self, code: int) -> str:
        name = self._lib.cobs_status_name(int(code))
        return name.decode() if name else "E_UNKNOWN"

    def _check(self, op: str, rc: int) -> int:
        if rc != ROCKET_OK:
            raise BridgeCallError(op, rc, self.status_name(rc))
        return rc

    # -- methods -----------------------------------------------------------------

    def device_name(self) -> Optional[str]:
        name = self._lib.cobs_device_name()
        return name.decode() if name else None

    def open_device(self) -> int:
        fd = int(self._lib.cobs_open())
        if fd < 0:
            raise BridgeCallError("open", fd, os.strerror(-fd) if fd != 0 else "unknown")
        return fd

    def close_device(self, fd: int) -> None:
        self._lib.cobs_close(int(fd))

    def buffer_alloc(self, fd: int, payload: int) -> int:
        handle = ctypes.c_uint64(0)
        self._check("buffer_alloc",
                    self._lib.cobs_buffer_alloc(int(fd), int(payload),
                                                ctypes.byref(handle)))
        return int(handle.value)

    def buffer_free(self, handle: int) -> None:
        self._check("buffer_free", self._lib.cobs_buffer_free(int(handle)))

    def buffer_payload(self, handle: int) -> int:
        return int(self._lib.cobs_buffer_payload(int(handle)))

    def buffer_allocated(self, handle: int) -> int:
        return int(self._lib.cobs_buffer_allocated(int(handle)))

    def tensor_bytes(self, handle: int) -> int:
        return int(self._lib.cobs_tensor_bytes(int(handle)))

    def buffer_set_matrix(self, handle: int, rows: int, cols: int, type_: int) -> None:
        self._check("buffer_set_matrix",
                    self._lib.cobs_buffer_set_matrix(int(handle), int(rows), int(cols),
                                                     int(type_)))

    def buffer_set_input(self, handle: int, rows: int, cols: int) -> None:
        self._check("buffer_set_input",
                    self._lib.cobs_buffer_set_input(int(handle), int(rows), int(cols)))

    def buffer_set_weight(self, handle: int, outputs: int, inputs: int) -> None:
        self._check("buffer_set_weight",
                    self._lib.cobs_buffer_set_weight(int(handle), int(outputs),
                                                     int(inputs)))

    def buffer_set_linear(self, handle: int, elements: int) -> None:
        self._check("buffer_set_linear",
                    self._lib.cobs_buffer_set_linear(int(handle), int(elements)))

    def upload(self, handle: int, offset: int, data: bytes) -> None:
        self._check("upload",
                    self._lib.cobs_upload(int(handle), int(offset), data, len(data)))

    def download(self, handle: int, offset: int, size: int) -> bytes:
        buf = ctypes.create_string_buffer(size)
        self._check("download",
                    self._lib.cobs_download(int(handle), int(offset), buf, size))
        return buf.raw

    def job_create(self, out_handle: int, out_offset: int) -> int:
        job = ctypes.c_uint64(0)
        self._check("job_create",
                    self._lib.cobs_job_create(int(out_handle), int(out_offset),
                                              ctypes.byref(job)))
        return int(job.value)

    def job_add_read(self, job: int, buffer: int, offset: int) -> None:
        self._check("job_add_read",
                    self._lib.cobs_job_add_read(int(job), int(buffer), int(offset)))

    def job_set_words(self, job: int, words: Sequence[int], flags: int) -> None:
        arr = (ctypes.c_uint64 * len(words))(*[int(w) & 0xFFFFFFFFFFFFFFFF for w in words])
        self._check("job_set_words",
                    self._lib.cobs_job_set_words(int(job), arr, len(words), int(flags)))

    def job_validate(self, job: int) -> None:
        self._check("job_validate", self._lib.cobs_job_validate(int(job)))

    def emit_lut(self, job, input_handle, offset, activation):
        self._check('emit_lut', self._lib.cobs_job_emit_lut(job, input_handle, offset, activation))

    def guard_check(self, handle):
        self._check('guard_check', self._lib.cobs_guard_check(handle))

    def job_free(self, job: int) -> None:
        self._lib.cobs_job_free(int(job))

    def execute(self, fd: int, job: int, mode: int) -> None:
        self._check("execute", self._lib.cobs_execute(int(fd), int(job), int(mode)))

    def execute_batch(self, fd: int, jobs: Sequence[int], mode: int) -> None:
        arr = (ctypes.c_uint64 * len(jobs))(*[int(j) for j in jobs])
        self._check("execute_batch",
                    self._lib.cobs_execute_batch(int(fd), arr, len(jobs), int(mode)))

    def schedule_execute(self, fd: int, job: int, mask: int, request_id: int) -> Dict[str, int]:
        host_begin = ctypes.c_uint64()
        ioctl_begin = ctypes.c_uint64()
        ioctl_end = ctypes.c_uint64()
        host_done = ctypes.c_uint64()
        service = ctypes.c_int64()
        actual = ctypes.c_uint()
        tasks = ctypes.c_uint()
        int_status = ctypes.c_int()
        result = ctypes.c_int()
        rc = self._lib.cobs_schedule_execute(
            int(fd), int(job), int(mask), int(request_id),
            ctypes.byref(host_begin), ctypes.byref(ioctl_begin), ctypes.byref(ioctl_end),
            ctypes.byref(host_done), ctypes.byref(service), ctypes.byref(actual),
            ctypes.byref(tasks), ctypes.byref(int_status), ctypes.byref(result))
        if rc != ROCKET_OK:
            raise BridgeCallError("schedule_execute", rc, self.status_name(rc))
        return {
            "host_begin_ns": int(host_begin.value),
            "ioctl_begin_ns": int(ioctl_begin.value),
            "ioctl_end_ns": int(ioctl_end.value),
            "host_done_ns": int(host_done.value),
            "driver_service_ns": int(service.value),
            "actual_mask": int(actual.value),
            "task_count": int(tasks.value),
            "int_status": int(int_status.value),
            "result": int(result.value),
        }


@dataclass
class RK3588Payload(Payload):
    """A single custom-ops register-command program.

    ``words`` are the real 64-bit regcmd words produced by the existing custom-ops
    emitters; ``reads`` and ``output`` are ``(BufferHandle, offset)`` pairs that must
    have been configured with :meth:`RK3588Backend.configure_tensor`.
    """

    backend: ClassVar[str] = "rk3588"
    words: Tuple[int, ...] = ()
    output: Optional[Tuple[BufferHandle, int]] = None
    reads: Tuple[Tuple[BufferHandle, int], ...] = ()
    flags: int = 0
    mode: int = MODE_SEPARATE

    def describe(self) -> str:
        n = len(self.words)
        out = self.output[0].token_id if self.output else None
        return f"rk3588(regcmd_words={n},output_buffer={out},mode={self.mode})"


# A stable alias so callers can name the concrete handle type explicitly.
RK3588Buffer_t = BufferHandle


class RK3588Backend(BaseBackend):
    """The independent RK3588 adapter over the custom-ops bridge."""

    def __init__(self, *, bridge_factory=None, library_path: Optional[str] = None,
                 clock=now_ns) -> None:
        super().__init__(clock=clock)
        self._library_path = library_path
        self._factory = bridge_factory or (lambda: CtypesBridge(library_path))
        self._bridge = None
        self._bridge_error: Optional[Exception] = None
        self._fd: Optional[int] = None
        self._tensor_kind: Dict[int, str] = {}
        self._executed_on_device = False
        try:
            self._bridge = self._factory()
            abi = getattr(self._bridge, "abi_version", None)
            if callable(abi):
                version = int(abi())
                if version != BRIDGE_ABI_VERSION:
                    raise UnknownSDKBehavior(
                        f"bridge ABI {version} != expected {BRIDGE_ABI_VERSION}"
                    )
        except (BackendError, OSError) as exc:
            self._bridge = None
            self._bridge_error = exc
        # Only the real ctypes loader over an actually-built library can back a hardware
        # claim.  An injected bridge double (used in tests) may mirror the Python surface
        # exactly, but nothing about it proves a physical device ran anything.
        self._bridge_is_real = isinstance(self._bridge, CtypesBridge)
        self._identity = DeviceIdentity(
            backend="rk3588",
            device_name="RK3588",
            vendor="Rockchip",
            platform="rk3588",
            driver="rocket / rknpu uapi via custom-ops",
            api_version=str(BRIDGE_ABI_VERSION),
            evidence=Evidence.BUILD,
            notes=("ctypes bridge to the existing custom-ops implementation; "
                   "synchronous submit, no fallback"),
        )

    # -- identity / capabilities -------------------------------------------------

    @property
    def identity(self) -> DeviceIdentity:
        # HARDWARE requires both a real library and an observed execution.  A fake or
        # injected bridge never promotes past BUILD, so no test double can fabricate a
        # silicon claim.
        ev = (Evidence.HARDWARE
              if (self._executed_on_device and self._bridge_is_real)
              else Evidence.BUILD)
        return dataclasses.replace(self._identity, evidence=ev)

    def _capabilities(self) -> Dict[Capability, CapabilityState]:
        if self._bridge is None:
            return {cap: CapabilityState.UNAVAILABLE for cap in Capability}
        return {
            Capability.SUBMIT: CapabilityState.AVAILABLE,
            Capability.WAIT: CapabilityState.AVAILABLE,
            Capability.UPLOAD: CapabilityState.AVAILABLE,
            Capability.DOWNLOAD: CapabilityState.AVAILABLE,
            Capability.QUERY_MEMORY: CapabilityState.AVAILABLE,
            # co_execute is a blocking call; there is no asynchronous completion.
            Capability.ASYNC_COMPLETION: CapabilityState.UNAVAILABLE,
            Capability.HOST_STAGED_MIGRATION: CapabilityState.AVAILABLE,
            # No cross-device P2P path is implemented or proven here.
            Capability.PEER_TO_PEER_TRANSFER: CapabilityState.UNAVAILABLE,
            Capability.REGCMD_PROGRAM: CapabilityState.AVAILABLE,
            Capability.RKNN3_MODEL: CapabilityState.UNAVAILABLE,
        }

    # -- error mapping -----------------------------------------------------------

    def _ensure_bridge(self) -> None:
        if self._bridge is None:
            raise DeviceUnavailable(
                f"rk3588 bridge unavailable: {self._bridge_error}"
            )

    def _ensure_open(self) -> None:
        self._ensure_bridge()
        if self._fd is None:
            raise DeviceUnavailable("rk3588 device is not open")

    @staticmethod
    def _map_error(exc: BridgeCallError) -> None:
        code = exc.code
        if code == ROCKET_E_UNSUPPORTED:
            raise UnsupportedOperation(str(exc)) from exc
        if code == ROCKET_E_NOMEM:
            raise MemoryBoundsError(str(exc)) from exc
        if code in (ROCKET_E_SHAPE, ROCKET_E_TILING):
            raise InvalidPayload(str(exc)) from exc
        raise BackendError(str(exc)) from exc

    # -- lifecycle ---------------------------------------------------------------

    def _open_native(self) -> None:
        self._ensure_bridge()
        name = self._bridge.device_name()
        if name is not None and name != "rk3588":
            raise UnsupportedOperation(
                f"rk3588 adapter refuses device profile '{name}'"
            )
        try:
            self._fd = self._bridge.open_device()
        except BridgeCallError as exc:
            raise DeviceUnavailable(str(exc)) from exc

    def _close_native(self) -> None:
        for job in list(self._jobs.values()):
            self.release_job(job)
        for buf in list(self._buffers.values()):
            self.free_buffer(buf)
        if self._fd is not None:
            self._bridge.close_device(self._fd)
            self._fd = None

    # -- buffers -----------------------------------------------------------------

    def _alloc_native(self, token_id: int, payload_bytes: int, purpose: str):
        self._ensure_open()
        try:
            handle = self._bridge.buffer_alloc(self._fd, payload_bytes)
        except BridgeCallError as exc:
            self._map_error(exc)
            raise
        allocated = self._bridge.buffer_allocated(handle)
        if allocated < payload_bytes:
            self._bridge.buffer_free(handle)
            raise MemoryBoundsError(
                f"bridge allocated {allocated} < requested {payload_bytes}"
            )
        return allocated, handle

    def _free_native(self, buffer: BufferHandle) -> None:
        self._tensor_kind.pop(buffer.token_id, None)
        self._bridge.buffer_free(buffer.native)

    def configure_tensor(self, buffer: BufferHandle, kind: str, **kwargs) -> None:
        """Attach the custom-ops tensor descriptor a buffer needs before it can be a
        read/output view.  ``kind`` is one of the :class:`TensorKind` strings."""
        self._check_buffer(buffer)
        if any(any(b is buffer for b in self._job_buffers(j)) for j in self._jobs.values()):
            raise BufferOwnershipError("buffer descriptor pinned by prepared job")
        self._ensure_open()
        try:
            if kind == TensorKind.MATRIX:
                self._bridge.buffer_set_matrix(buffer.native, kwargs["rows"],
                                               kwargs["cols"], int(kwargs.get("type", 0)))
            elif kind == TensorKind.INPUT:
                self._bridge.buffer_set_input(buffer.native, kwargs["rows"], kwargs["cols"])
            elif kind == TensorKind.WEIGHT:
                self._bridge.buffer_set_weight(buffer.native, kwargs["outputs"],
                                               kwargs["inputs"])
            elif kind == TensorKind.LINEAR:
                self._bridge.buffer_set_linear(buffer.native, kwargs["elements"])
            else:
                raise InvalidPayload(f"unknown tensor kind {kind!r}")
        except BridgeCallError as exc:
            self._map_error(exc)
            raise
        self._tensor_kind[buffer.token_id] = kind

    def tensor_bytes(self, buffer: BufferHandle) -> int:
        self._check_buffer(buffer)
        return self._bridge.tensor_bytes(buffer.native)

    def _upload_native(self, buffer: BufferHandle, offset: int, data: bytes) -> None:
        self._ensure_open()
        try:
            self._bridge.upload(buffer.native, offset, data)
        except BridgeCallError as exc:
            self._map_error(exc)
            raise

    def _download_native(self, buffer: BufferHandle, offset: int, length: int) -> bytes:
        self._ensure_open()
        try:
            return self._bridge.download(buffer.native, offset, length)
        except BridgeCallError as exc:
            self._map_error(exc)
            raise

    # -- jobs / submit -----------------------------------------------------------

    def _require_owned(self, handle: BufferHandle) -> None:
        if not isinstance(handle, BufferHandle):
            raise InvalidPayload("buffer entries must be BufferHandle instances")
        self._check_buffer(handle)

    def _create_job_native(self, payload: RK3588Payload) -> int:
        if not isinstance(payload, RK3588Payload):
            raise InvalidPayload("rk3588 expects an RK3588Payload")
        if not payload.words:
            raise InvalidPayload("regcmd program must contain at least one word")
        if payload.mode not in (MODE_SEPARATE, MODE_CHAIN):
            raise InvalidPayload(f"invalid execution mode {payload.mode}")
        if payload.output is None:
            raise InvalidPayload("payload must name an output buffer")
        self._ensure_open()
        out_buf, out_offset = payload.output
        self._require_owned(out_buf)
        job = None
        try:
            job = self._bridge.job_create(out_buf.native, out_offset)
            for buf, offset in payload.reads:
                self._require_owned(buf)
                self._bridge.job_add_read(job, buf.native, offset)
            self._bridge.job_set_words(job, payload.words, payload.flags)
            self._bridge.job_validate(job)
        except Exception as exc:
            if job is not None:
                self._bridge.job_free(job)
            if isinstance(exc, BridgeCallError):
                self._map_error(exc)
            raise
        return job

    def create_lut_job(self, input_buffer, output_buffer, *, activation=0):
        """Prepare using existing co_emit_lut; no new Regcmd format or generator."""
        self._ensure_open()
        self._require_owned(input_buffer); self._require_owned(output_buffer)
        if type(activation) is not int or not 0 <= activation <= 5:
            raise InvalidPayload('invalid LUT activation')
        payload = RK3588Payload(output=(output_buffer, 0), reads=((input_buffer, 0),), mode=MODE_CHAIN)
        native = self._bridge.job_create(output_buffer.native, 0)
        try:
            self._bridge.emit_lut(native, input_buffer.native, 0, activation)
        except Exception:
            self._bridge.job_free(native)
            raise
        job = JobHandle(self.identity.backend, self._new_id(), payload, native=native)
        self._jobs[job.job_id] = job
        return job

    def check_guard(self, buffer):
        self._check_buffer(buffer)
        self._bridge.guard_check(buffer.native)

    def _free_job_native(self, job: JobHandle) -> None:
        self._bridge.job_free(job.native)

    def _submit_native(self, job: JobHandle, token: CompletionToken) -> None:
        self._ensure_open()
        payload: RK3588Payload = job.payload
        try:
            self._bridge.execute(self._fd, job.native, payload.mode)
        except BridgeCallError as exc:
            self._map_error(exc)
            raise
        self._executed_on_device = True
        token.state = TokenState.COMPLETE
        token.completed_ns = self._clock()

    def _wait_native(self, token: CompletionToken, timeout_ns: Optional[int]) -> TokenState:
        # Synchronous backend: submit already ran the blocking execute.  wait observes.
        if token.state is TokenState.PENDING:
            raise BackendError(
                "synchronous rk3588 backend returned a pending token; refusing to wait"
            )
        return token.state

    def submit_profiled(self, job: JobHandle, core_mask: int, request_id: int,
                        timeout_ns: Optional[int] = None):
        """Single-core submission through ``co_schedule_execute`` with the driver
        profile.  Returns ``(token, profile_dict)``; the token is already COMPLETE."""
        if timeout_ns is not None:
            raise UnsupportedOperation("synchronous profiled submit has no cancellable deadline")
        self.require(Capability.SUBMIT)
        self._check_job(job)
        self._ensure_open()
        payload: RK3588Payload = job.payload
        if payload.mode != MODE_SEPARATE:
            raise InvalidPayload("profiled submission requires MODE_SEPARATE")
        token = CompletionToken(
            token_id=self._new_id(),
            backend=self.identity.backend,
            job=job,
            state=TokenState.PENDING,
            submitted_ns=self._clock(),
            deadline_ns=None,
        )
        self._tokens[token.token_id] = token
        try:
            profile = self._bridge.schedule_execute(
                self._fd, job.native, core_mask, request_id)
        except BridgeCallError as exc:
            token.state = TokenState.FAILED
            token.error = str(exc)
            self._map_error(exc)
            raise
        if profile["result"] != ROCKET_OK:
            token.state = TokenState.FAILED
            token.error = self._bridge.status_name(profile["result"])
            raise BackendError(f"schedule_execute: {token.error}")
        self._executed_on_device = True
        token.state = TokenState.COMPLETE
        token.completed_ns = self._clock()
        return token, profile

    def submit_batch(self, jobs: Sequence[JobHandle], mode: int = MODE_SEPARATE) -> None:
        """Execute several prepared jobs through one ``co_execute`` call."""
        self.require(Capability.SUBMIT)
        self._ensure_open()
        if not jobs:
            raise InvalidPayload("no jobs to submit")
        for job in jobs:
            self._check_job(job)
            if job.payload.mode != mode:
                raise InvalidPayload("all batched jobs must use the requested mode")
        try:
            self._bridge.execute_batch(self._fd, [j.native for j in jobs], mode)
        except BridgeCallError as exc:
            self._map_error(exc)
            raise
        self._executed_on_device = True

    # -- memory ------------------------------------------------------------------

    def _memory_totals_native(self) -> Dict[str, Optional[int]]:
        # The custom-ops / rknpu path does not publish a device or system total here.
        # Leaving them None is the honest answer; owned bytes come from the base class.
        return {
            "reserved_bytes": None,
            "device_total_bytes": None,
            "system_total_bytes": None,
            "scope": ("owned payload/allocated bytes are exact; device and system totals "
                      "are not exposed by the custom-ops path and stay None"),
        }


# ctypes releases the GIL: guard the process-wide C handle registry across instances.
# Direct C ABI callers must serialize calls as documented in the bridge source.
import functools as _functools
import threading as _threading
_NATIVE_LOCK = _threading.RLock()
def _serial_native(method):
    @_functools.wraps(method)
    def call(*args, **kwargs):
        with _NATIVE_LOCK:
            return method(*args, **kwargs)
    return call
for _name, _method in list(CtypesBridge.__dict__.items()):
    if not _name.startswith('_') and callable(_method):
        setattr(CtypesBridge, _name, _serial_native(_method))
