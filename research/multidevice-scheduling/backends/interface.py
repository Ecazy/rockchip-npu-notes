#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Uniform multi-device backend interface (Python standard library only).

This module is a *contract*, not a driver.  It names the small set of operations a
scheduler may use across devices that do not share a command ABI, and it pins the
ownership, timeout, completion-lifetime and error rules those operations obey.  The
per-device adapters (:mod:`backends.rk3588`, :mod:`backends.rk182x`) and the in-memory
:mod:`backends.mock` reference implement it; none of them may silently fall back to a
different device or to CPU compute.

Design rules enforced here and by the adapters:

* A handle (Buffer/Job/CompletionToken) is *opaque* and device-specific.  The only
  thing a scheduler may do with a foreign handle is fail closed: it is an error to
  pass an RK182X payload to an RK3588 adapter, or vice versa.
* Capabilities are tri-state.  ``UNAVAILABLE`` and ``UNKNOWN`` both make an operation
  raise :class:`UnsupportedOperation`; nothing fabricates a success.
* A field that the device does not expose is ``None``, never 0 and never derived.
  ``query_memory`` reports the bytes the adapter actually owns and leaves the device
  and system totals unknown when the driver does not publish them.
* Transfers are bounded by both the buffer's payload size and a per-call byte cap.
* A completion token is single-use.  Releasing it ends its lifetime; waiting on a
  released token is an error, not a silent success.
* ``submit``/``wait`` keep their real execution model.  A synchronous device returns a
  token that is already COMPLETE and ``wait`` only observes it; an asynchronous device
  returns PENDING and ``wait`` blocks with the caller's timeout.
"""

from __future__ import annotations

import abc
import enum
import itertools
import time
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

__all__ = [
    "Capability",
    "CapabilityState",
    "TokenState",
    "Evidence",
    "BackendError",
    "UnsupportedOperation",
    "BackendMismatch",
    "DeviceUnavailable",
    "InvalidPayload",
    "MemoryBoundsError",
    "BufferOwnershipError",
    "OperationTimeout",
    "UnknownSDKBehavior",
    "Payload",
    "JobHandle",
    "BufferHandle",
    "CompletionToken",
    "DeviceIdentity",
    "MemoryInfo",
    "Backend",
    "BaseBackend",
    "MAX_TRANSFER_BYTES",
    "now_ns",
]

#: Largest single upload/download a caller may request in one call.  This is a hard
#: host-side bound; it is independent of any device-specific DMA chunk size.
MAX_TRANSFER_BYTES = 64 * 1024 * 1024


def now_ns() -> int:
    """Monotonic nanosecond clock, the timing base for every token timestamp."""
    return time.monotonic_ns()


# --------------------------------------------------------------------------- enums


class Capability(str, enum.Enum):
    """Operations a scheduler may ask a backend about before using them."""

    SUBMIT = "submit"
    WAIT = "wait"
    UPLOAD = "upload"
    DOWNLOAD = "download"
    QUERY_MEMORY = "query_memory"
    ASYNC_COMPLETION = "async_completion"
    #: Move bytes through Host staging; needs usable source/destination bindings.
    HOST_STAGED_MIGRATION = "host_staged_migration"
    #: Device-to-device DMA with no host copy.  Never claimed without SDK proof.
    PEER_TO_PEER_TRANSFER = "peer_to_peer_transfer"
    #: Device consumes RK3588 64-bit register-command words.
    REGCMD_PROGRAM = "regcmd_program"
    #: Device consumes an RKNN3 model/session buffer.
    RKNN3_MODEL = "rknn3_model"


class CapabilityState(str, enum.Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    #: The adapter could not determine the state; treated as not usable.
    UNKNOWN = "unknown"


class TokenState(str, enum.Enum):
    PENDING = "pending"
    COMPLETE = "complete"
    FAILED = "failed"
    #: The token's lifetime is over; it may no longer be waited on.
    CONSUMED = "consumed"


class Evidence(str, enum.Enum):
    """How far the claim about this backend has actually been taken."""

    #: Executed and observed on the physical device.
    HARDWARE = "hardware"
    #: Compiled/linked against the real implementation but not run on hardware.
    BUILD = "build"
    #: In-memory contract reference; never touches a device.
    MOCK = "mock"
    #: Binding checks against an injected runtime contract; never the real SDK.
    CONTRACT = "contract"


# --------------------------------------------------------------------------- errors


class BackendError(RuntimeError):
    """Base class for every backend failure."""


class UnsupportedOperation(BackendError):
    """Capability is unavailable/unknown, or the operation was never valid here."""


class BackendMismatch(BackendError):
    """A handle/payload/runtime belongs to a different device or ABI."""


class DeviceUnavailable(BackendError):
    """The real device or its library could not be opened."""


class InvalidPayload(BackendError):
    """The payload is malformed or violates the device's payload ABI."""


class MemoryBoundsError(BackendError):
    """A transfer or buffer request is outside the owned region."""


class BufferOwnershipError(BackendError):
    """A buffer/token was used after release or freed twice."""


class OperationTimeout(BackendError):
    """An asynchronous operation did not finish inside the caller's timeout."""


class UnknownSDKBehavior(BackendError):
    """The injected SDK returned a version/platform/capability we cannot interpret."""


# --------------------------------------------------------------------------- handles


class Payload(abc.ABC):
    """Device-specific work description.  Base type is never submitted directly.

    ``backend`` is a class attribute (a ``ClassVar`` in the concrete dataclasses), not
    a constructor field, so each payload type is permanently bound to one ABI.
    """

    backend: str = "abstract"

    @abc.abstractmethod
    def describe(self) -> str:  # pragma: no cover - trivial per adapter
        raise NotImplementedError


@dataclass
class BufferHandle:
    """Opaque device buffer.  Only the owning adapter may touch ``token_id``."""

    backend: str
    token_id: int
    payload_bytes: int
    allocated_bytes: int
    purpose: str = "generic"
    released: bool = False
    #: Device-private reference (ctypes handle, runtime object, ...).  Schedulers
    #: must treat this as opaque and never share it across adapters.
    native: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.payload_bytes < 0 or self.allocated_bytes < self.payload_bytes:
            raise InvalidPayload("buffer byte counts are inconsistent")


@dataclass
class JobHandle:
    """Opaque, device-specific prepared job."""

    backend: str
    job_id: int
    payload: Payload
    released: bool = False
    native: object = field(default=None, repr=False)


@dataclass
class CompletionToken:
    """Single-use completion identity returned by ``submit``.

    The token is the scheduler's only proof that a job was observed to finish.  Its
    lifetime ends at :meth:`BaseBackend.release_token`; after that any use raises.
    """

    token_id: int
    backend: str
    job: JobHandle
    state: TokenState
    submitted_ns: int
    deadline_ns: Optional[int] = None
    completed_ns: Optional[int] = None
    error: Optional[str] = None

    def is_terminal(self) -> bool:
        return self.state in (TokenState.COMPLETE, TokenState.FAILED)


# --------------------------------------------------------------------------- info


@dataclass(frozen=True)
class DeviceIdentity:
    backend: str
    device_name: str
    vendor: str
    platform: str
    driver: Optional[str]
    api_version: str
    evidence: Evidence
    notes: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {
            "backend": self.backend,
            "device_name": self.device_name,
            "vendor": self.vendor,
            "platform": self.platform,
            "driver": self.driver,
            "api_version": self.api_version,
            "evidence": self.evidence.value,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class MemoryInfo:
    """Snapshot of *owned* memory plus whatever totals the device exposes.

    ``owned_payload_bytes`` / ``owned_allocated_bytes`` are computed from live buffers
    this adapter allocated and are always exact.  Every other field is ``None`` when the
    device/driver does not publish it; no adapter may estimate a total from the owned
    figure.
    """

    backend: str
    owned_payload_bytes: int
    owned_allocated_bytes: int
    live_buffers: int
    reserved_bytes: Optional[int]
    device_total_bytes: Optional[int]
    system_total_bytes: Optional[int]
    scope: str = ""

    @property
    def totals_known(self) -> bool:
        return self.device_total_bytes is not None and self.system_total_bytes is not None


# --------------------------------------------------------------------------- protocol


@runtime_checkable
class Backend(Protocol):
    """The operations a scheduler needs from any device.  Implementations are named
    explicitly in :data:`backends.ADAPTERS`; there is no auto-detection."""

    @property
    def identity(self) -> DeviceIdentity:
        ...

    def capabilities(self) -> Mapping[Capability, CapabilityState]:
        ...

    def capability_state(self, capability: Capability) -> CapabilityState:
        ...

    def require(self, capability: Capability) -> None:
        ...

    def open(self) -> None:
        ...

    def close(self) -> None:
        ...

    def query_memory(self) -> MemoryInfo:
        ...

    def transfer(self, buffer: BufferHandle, *, direction: str, offset: int = 0,
                 data: bytes | None = None, length: int | None = None):
        ...

    def alloc_buffer(self, payload_bytes: int, purpose: str = "generic") -> BufferHandle:
        ...

    def free_buffer(self, buffer: BufferHandle) -> None:
        ...

    def upload(self, buffer: BufferHandle, offset: int, data: bytes) -> int:
        ...

    def download(self, buffer: BufferHandle, offset: int, length: int) -> bytes:
        ...

    def create_job(self, payload: Payload) -> JobHandle:
        ...

    def submit(self, job: JobHandle, timeout_ns: Optional[int] = None) -> CompletionToken:
        ...

    def wait(self, token: CompletionToken, timeout_ns: Optional[int] = None) -> TokenState:
        ...

    def release_token(self, token: CompletionToken) -> None:
        ...


# --------------------------------------------------------------------------- base


class BaseBackend(abc.ABC):
    """Shared ownership/validation machinery for concrete adapters.

    Subclasses supply the native operations (``_open_native``, ``_alloc_native``,
    ``_upload_native``, ``_submit_native``, ``_wait_native``, ...) and the static
    :attr:`identity` / capability table.  Everything in between -- handle registries,
    stream identity, bounds checks, token lifetime and the fail-closed capability gate
    -- lives here so all adapters behave identically.
    """

    def __init__(self, *, clock=now_ns) -> None:
        self._clock = clock
        self._ids = itertools.count(1)
        self._buffers: Dict[int, BufferHandle] = {}
        self._jobs: Dict[int, JobHandle] = {}
        self._tokens: Dict[int, CompletionToken] = {}
        self._opened = False

    # -- identity / capabilities -------------------------------------------------

    @property
    @abc.abstractmethod
    def identity(self) -> DeviceIdentity:
        raise NotImplementedError

    @abc.abstractmethod
    def _capabilities(self) -> Dict[Capability, CapabilityState]:
        raise NotImplementedError

    def capabilities(self) -> Mapping[Capability, CapabilityState]:
        return dict(self._capabilities())

    def capability_state(self, capability: Capability) -> CapabilityState:
        return self.capabilities().get(capability, CapabilityState.UNKNOWN)

    def require(self, capability: Capability) -> None:
        state = self.capability_state(capability)
        if state is not CapabilityState.AVAILABLE:
            raise UnsupportedOperation(
                f"{self.identity.backend}: capability {capability.value} is {state.value}"
            )

    # -- lifecycle ---------------------------------------------------------------

    def open(self) -> None:
        if self._opened:
            return
        self._open_native()
        self._opened = True

    def close(self) -> None:
        if not self._opened:
            return
        if any(t.state is TokenState.PENDING for t in self._tokens.values()):
            raise BufferOwnershipError("pending work must complete before close")
        buffers, jobs, tokens = list(self._buffers.values()), list(self._jobs.values()), list(self._tokens.values())
        self._close_native()
        for handle in buffers + jobs:
            handle.released = True
        for token in tokens:
            token.state = TokenState.CONSUMED
        self._buffers.clear(); self._jobs.clear(); self._tokens.clear()
        self._opened = False

    def _ensure_open(self):
        if not self._opened:
            raise DeviceUnavailable("backend is closed")

    @staticmethod
    def _job_buffers(job):
        payload = job.payload
        result = []
        output = getattr(payload, 'output', None)
        if output:
            result.append(output[0])
        result.extend(x[0] for x in getattr(payload, 'reads', ()))
        result.extend(x for x in getattr(payload, 'inputs', ()) if isinstance(x, BufferHandle))
        result.extend(x for x in getattr(payload, 'outputs', ()) if isinstance(x, BufferHandle))
        return result

    def _buffer_pending(self, buffer):
        return any(t.state is TokenState.PENDING and
                   any(b is buffer for b in self._job_buffers(t.job))
                   for t in self._tokens.values())

    def __enter__(self) -> "BaseBackend":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- buffer ownership --------------------------------------------------------

    def _new_id(self) -> int:
        return next(self._ids)

    def _check_buffer(self, buffer: BufferHandle) -> None:
        self._ensure_open()
        if not isinstance(buffer, BufferHandle):
            raise BackendMismatch(
                f"{self.identity.backend}: expected a BufferHandle, got {type(buffer).__name__}"
            )
        if buffer.backend != self.identity.backend:
            raise BackendMismatch(
                f"{self.identity.backend}: buffer belongs to {buffer.backend}"
            )
        if buffer.token_id not in self._buffers or self._buffers[buffer.token_id] is not buffer:
            raise BufferOwnershipError("buffer is not owned by this adapter")
        if buffer.released:
            raise BufferOwnershipError("buffer has been released")

    def alloc_buffer(self, payload_bytes: int, purpose: str = "generic") -> BufferHandle:
        self._ensure_open()
        if type(payload_bytes) is not int or payload_bytes <= 0:
            raise MemoryBoundsError("payload_bytes must be a positive integer")
        token_id = self._new_id()
        allocated, native = self._alloc_native(token_id, payload_bytes, purpose)
        buffer = BufferHandle(
            backend=self.identity.backend,
            token_id=token_id,
            payload_bytes=payload_bytes,
            allocated_bytes=allocated,
            purpose=purpose,
            native=native,
        )
        self._buffers[token_id] = buffer
        return buffer

    def free_buffer(self, buffer: BufferHandle) -> None:
        self._check_buffer(buffer)
        if any(any(b is buffer for b in self._job_buffers(j)) for j in self._jobs.values()):
            raise BufferOwnershipError("buffer pinned by a prepared job")
        self._free_native(buffer)
        buffer.released = True
        self._buffers.pop(buffer.token_id, None)

    # -- transfer ----------------------------------------------------------------

    def _check_transfer(self, buffer: BufferHandle, offset: int, size: int) -> None:
        self._check_buffer(buffer)
        if type(offset) is not int or type(size) is not int or offset < 0 or size < 0:
            raise MemoryBoundsError("offset and size must be non-negative integers")
        if size > MAX_TRANSFER_BYTES:
            raise MemoryBoundsError(
                f"transfer {size} exceeds per-call cap {MAX_TRANSFER_BYTES}"
            )
        if offset > buffer.payload_bytes or size > buffer.payload_bytes - offset:
            raise MemoryBoundsError(
                f"transfer [{offset},{offset + size}) outside payload {buffer.payload_bytes}"
            )

    def upload(self, buffer: BufferHandle, offset: int, data: bytes) -> int:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise InvalidPayload("upload data must be bytes-like")
        self.require(Capability.UPLOAD)
        blob = bytes(data)
        self._check_transfer(buffer, offset, len(blob))
        if self._buffer_pending(buffer):
            raise BufferOwnershipError("buffer used by pending work")
        if not blob:
            return 0
        self._upload_native(buffer, offset, blob)
        return len(blob)

    def download(self, buffer: BufferHandle, offset: int, length: int) -> bytes:
        self.require(Capability.DOWNLOAD)
        self._check_transfer(buffer, offset, length)
        if self._buffer_pending(buffer):
            raise BufferOwnershipError("buffer used by pending work")
        if not length:
            return b''
        data = bytes(self._download_native(buffer, offset, length))
        if len(data) != length:
            raise BackendError("runtime returned a short transfer")
        return data

    def transfer(self, buffer, *, direction, offset=0, data=None, length=None):
        """Bounded Host transfer; native handles never cross backend boundaries."""
        if direction == 'upload' and data is not None and length is None:
            return self.upload(buffer, offset, data)
        if direction == 'download' and data is None and length is not None:
            return self.download(buffer, offset, length)
        raise InvalidPayload("transfer requires upload+data or download+length")

    # -- job / submit / wait -----------------------------------------------------

    def create_job(self, payload: Payload) -> JobHandle:
        self._ensure_open()
        if not isinstance(payload, Payload):
            raise InvalidPayload(
                f"{self.identity.backend}: expected Payload, got {type(payload).__name__}"
            )
        if payload.backend != self.identity.backend:
            raise BackendMismatch(
                f"{self.identity.backend}: payload ABI belongs to {payload.backend}"
            )
        job_id = self._new_id()
        native = self._create_job_native(payload)
        job = JobHandle(
            backend=self.identity.backend,
            job_id=job_id,
            payload=payload,
            native=native,
        )
        self._jobs[job_id] = job
        return job

    def _check_job(self, job: JobHandle) -> None:
        self._ensure_open()
        if not isinstance(job, JobHandle):
            raise BackendMismatch(
                f"{self.identity.backend}: expected a JobHandle, got {type(job).__name__}"
            )
        if job.backend != self.identity.backend:
            raise BackendMismatch(f"{self.identity.backend}: job belongs to {job.backend}")
        if job.job_id not in self._jobs or self._jobs[job.job_id] is not job:
            raise BufferOwnershipError("job is not owned by this adapter")
        if job.released:
            raise BufferOwnershipError("job has been released")

    def release_job(self, job: JobHandle) -> None:
        self._check_job(job)
        if any(t.job is job and t.state is TokenState.PENDING for t in self._tokens.values()):
            raise BufferOwnershipError("job has pending work")
        self._free_job_native(job)
        job.released = True
        self._jobs.pop(job.job_id, None)

    def submit(self, job: JobHandle, timeout_ns: Optional[int] = None) -> CompletionToken:
        self.require(Capability.SUBMIT)
        self._check_job(job)
        if timeout_ns is not None and (type(timeout_ns) is not int or timeout_ns < 0):
            raise InvalidPayload("timeout_ns must be a non-negative integer or None")
        if timeout_ns is not None and self.capability_state(Capability.ASYNC_COMPLETION) is not CapabilityState.AVAILABLE:
            raise UnsupportedOperation("synchronous submit does not provide cancellation deadlines")
        for b in self._job_buffers(job):
            self._check_buffer(b)
        if any(t.job is job and t.state is TokenState.PENDING for t in self._tokens.values()):
            raise BufferOwnershipError("job already pending")
        token_id = self._new_id()
        submitted = self._clock()
        deadline = None if timeout_ns is None else submitted + timeout_ns
        token = CompletionToken(
            token_id=token_id,
            backend=self.identity.backend,
            job=job,
            state=TokenState.PENDING,
            submitted_ns=submitted,
            deadline_ns=deadline,
        )
        self._tokens[token_id] = token
        try:
            self._submit_native(job, token)
        except Exception as exc:
            token.state = TokenState.FAILED
            token.error = str(exc)
            self._tokens.pop(token_id, None)
            raise
        return token

    def wait(self, token: CompletionToken, timeout_ns: Optional[int] = None) -> TokenState:
        self.require(Capability.WAIT)
        self._check_token(token)
        if timeout_ns is not None and (type(timeout_ns) is not int or timeout_ns < 0):
            raise InvalidPayload("timeout_ns must be a non-negative integer or None")
        if token.state is TokenState.PENDING and token.deadline_ns is not None:
            remaining = max(0, token.deadline_ns - self._clock())
            timeout_ns = remaining if timeout_ns is None else min(remaining, timeout_ns)
        return self._wait_native(token, timeout_ns)

    def _check_token(self, token: CompletionToken) -> None:
        self._ensure_open()
        if not isinstance(token, CompletionToken):
            raise BackendMismatch(
                f"{self.identity.backend}: expected CompletionToken, got {type(token).__name__}"
            )
        if token.backend != self.identity.backend:
            raise BackendMismatch(
                f"{self.identity.backend}: token belongs to {token.backend}"
            )
        if token.token_id not in self._tokens or self._tokens[token.token_id] is not token:
            raise BufferOwnershipError("completion token is not owned by this adapter")
        if token.state is TokenState.CONSUMED:
            raise BufferOwnershipError("completion token has been released")

    def release_token(self, token: CompletionToken) -> None:
        self._check_token(token)
        if token.state is TokenState.PENDING:
            raise BufferOwnershipError("pending token cannot be released")
        self._tokens.pop(token.token_id, None)
        token.state = TokenState.CONSUMED

    # -- memory ------------------------------------------------------------------

    def query_memory(self) -> MemoryInfo:
        self._ensure_open()
        self.require(Capability.QUERY_MEMORY)
        owned_payload = sum(b.payload_bytes for b in self._buffers.values())
        owned_alloc = sum(b.allocated_bytes for b in self._buffers.values())
        totals = self._memory_totals_native()
        return MemoryInfo(
            backend=self.identity.backend,
            owned_payload_bytes=owned_payload,
            owned_allocated_bytes=owned_alloc,
            live_buffers=len(self._buffers),
            reserved_bytes=totals.get("reserved_bytes"),
            device_total_bytes=totals.get("device_total_bytes"),
            system_total_bytes=totals.get("system_total_bytes"),
            scope=totals.get("scope", "owned bytes are exact; device/system totals are "
                                    "None unless the driver publishes them"),
        )

    def _memory_totals_native(self) -> Dict[str, Optional[int]]:
        return {"reserved_bytes": None, "device_total_bytes": None, "system_total_bytes": None}

    # -- native hooks ------------------------------------------------------------

    @abc.abstractmethod
    def _open_native(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _close_native(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _alloc_native(self, token_id: int, payload_bytes: int, purpose: str):
        """Return ``(allocated_bytes, native_handle)`` for a new buffer."""
        raise NotImplementedError

    @abc.abstractmethod
    def _free_native(self, buffer: BufferHandle) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _upload_native(self, buffer: BufferHandle, offset: int, data: bytes) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _download_native(self, buffer: BufferHandle, offset: int, length: int) -> bytes:
        raise NotImplementedError

    @abc.abstractmethod
    def _create_job_native(self, payload: Payload) -> object:
        raise NotImplementedError

    def _free_job_native(self, job: JobHandle) -> None:
        return None

    @abc.abstractmethod
    def _submit_native(self, job: JobHandle, token: CompletionToken) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _wait_native(self, token: CompletionToken, timeout_ns: Optional[int]) -> TokenState:
        raise NotImplementedError


def transfer_between(source_backend, source, target_backend, target, *, length, source_offset=0, target_offset=0):
    """Synchronous Host staging; failure never frees the source or reports success."""
    source_backend.require(Capability.HOST_STAGED_MIGRATION)
    target_backend.require(Capability.HOST_STAGED_MIGRATION)
    source_backend._check_transfer(source, source_offset, length)
    target_backend._check_transfer(target, target_offset, length)
    begin = now_ns()
    data = source_backend.transfer(source, direction='download', offset=source_offset, length=length)
    downloaded = now_ns()
    target_backend.transfer(target, direction='upload', offset=target_offset, data=data)
    end = now_ns()
    return {'bytes': length, 'download_ns': downloaded-begin, 'upload_ns': end-downloaded,
            'total_ns': end-begin, 'scope': 'Host staging API wall time; includes copies/sync',
            'evidence': 'observed_api_wall_time'}
