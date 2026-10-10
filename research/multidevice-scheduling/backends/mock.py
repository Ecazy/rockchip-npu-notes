#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""In-memory contract reference backend.

``MockBackend`` exists so the *contract* in :mod:`backends.interface` can be exercised
without hardware: deterministic clocks, real byte buffers, owner-checked handles and an
asynchronous token path.  It never touches a device and must not be presented as
evidence about any physical part.  Its identity carries ``Evidence.MOCK``.

The simulated clock is injected, so tests advance time explicitly rather than sleeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Dict, Optional

from .interface import (
    BackendError,
    BaseBackend,
    Capability,
    CapabilityState,
    CompletionToken,
    DeviceIdentity,
    Evidence,
    InvalidPayload,
    MemoryBoundsError,
    OperationTimeout,
    Payload,
    TokenState,
)

__all__ = ["MockPayload", "MockBackend", "FakeClock"]


class FakeClock:
    """Deterministic monotonic clock in nanoseconds."""

    def __init__(self, start_ns: int = 1_000_000_000) -> None:
        self._now = int(start_ns)

    def __call__(self) -> int:
        return self._now

    def advance(self, delta_ns: int) -> int:
        if delta_ns < 0:
            raise ValueError("cannot move a monotonic clock backwards")
        self._now += int(delta_ns)
        return self._now


@dataclass
class MockPayload(Payload):
    """A unit of fake work.  ``duration_ns`` lets the token stay PENDING."""

    backend: ClassVar[str] = "mock"
    kind: str = "noop"
    duration_ns: int = 0
    fail: bool = False

    def describe(self) -> str:
        return f"mock(kind={self.kind},duration_ns={self.duration_ns})"


class MockBackend(BaseBackend):
    """Contract-only backend.  Evidence is always MOCK."""

    def __init__(self, *, clock: FakeClock, capacity_bytes: Optional[int] = None,
                 name: str = "mock0") -> None:
        super().__init__(clock=clock)
        self._name = name
        self._capacity_bytes = capacity_bytes
        self._data: Dict[int, bytearray] = {}
        self._ready: Dict[int, int] = {}
        self._identity = DeviceIdentity(
            backend="mock",
            device_name=name,
            vendor="test",
            platform="in-memory",
            driver="none",
            api_version="1",
            evidence=Evidence.MOCK,
            notes="in-memory contract reference; never a hardware claim",
        )

    @property
    def identity(self) -> DeviceIdentity:
        return self._identity

    def _capabilities(self) -> Dict[Capability, CapabilityState]:
        return {
            Capability.SUBMIT: CapabilityState.AVAILABLE,
            Capability.WAIT: CapabilityState.AVAILABLE,
            Capability.UPLOAD: CapabilityState.AVAILABLE,
            Capability.DOWNLOAD: CapabilityState.AVAILABLE,
            Capability.QUERY_MEMORY: CapabilityState.AVAILABLE,
            Capability.ASYNC_COMPLETION: CapabilityState.AVAILABLE,
            Capability.HOST_STAGED_MIGRATION: CapabilityState.AVAILABLE,
            Capability.PEER_TO_PEER_TRANSFER: CapabilityState.UNAVAILABLE,
            Capability.REGCMD_PROGRAM: CapabilityState.UNAVAILABLE,
            Capability.RKNN3_MODEL: CapabilityState.UNAVAILABLE,
        }

    # -- native hooks ------------------------------------------------------------

    def _open_native(self) -> None:
        if self._capacity_bytes is not None and self._capacity_bytes <= 0:
            raise BackendError("mock capacity must be positive")

    def _close_native(self) -> None:
        self._data.clear()
        self._ready.clear()

    def _alloc_native(self, token_id: int, payload_bytes: int, purpose: str):
        if self._capacity_bytes is not None:
            used = sum(b.allocated_bytes for b in self._buffers.values())
            if used + payload_bytes > self._capacity_bytes:
                raise MemoryBoundsError(
                    f"mock allocation of {payload_bytes} exceeds capacity {self._capacity_bytes}"
                )
        return payload_bytes, token_id

    def _free_native(self, buffer) -> None:
        self._data.pop(buffer.token_id, None)

    def _upload_native(self, buffer, offset: int, data: bytes) -> None:
        store = self._data.setdefault(buffer.token_id, bytearray(buffer.payload_bytes))
        store[offset:offset + len(data)] = data

    def _download_native(self, buffer, offset: int, length: int) -> bytes:
        store = self._data.get(buffer.token_id)
        if store is None:
            return bytes(length)
        return bytes(store[offset:offset + length])

    def _create_job_native(self, payload: MockPayload) -> object:
        if not isinstance(payload, MockPayload):
            raise InvalidPayload("mock backend expects a MockPayload")
        if payload.duration_ns < 0:
            raise InvalidPayload("duration_ns must be non-negative")
        return {"kind": payload.kind, "duration_ns": payload.duration_ns}

    def _submit_native(self, job, token: CompletionToken) -> None:
        payload: MockPayload = job.payload
        if payload.fail:
            raise BackendError(f"mock job {payload.kind} failed on purpose")
        ready = self._clock() + payload.duration_ns
        self._ready[token.token_id] = ready
        if payload.duration_ns == 0:
            token.state = TokenState.COMPLETE
            token.completed_ns = self._clock()

    def _wait_native(self, token: CompletionToken, timeout_ns: Optional[int]) -> TokenState:
        if token.is_terminal():
            return token.state
        ready = self._ready.get(token.token_id, self._clock())
        remaining = ready - self._clock()
        if remaining <= 0:
            token.state = TokenState.COMPLETE
            token.completed_ns = self._clock()
            return token.state
        if timeout_ns is not None and timeout_ns < remaining:
            raise OperationTimeout(
                f"mock job did not complete within {timeout_ns}ns (needs {remaining}ns)"
            )
        # Deterministic advance: a generous (or absent) timeout simply runs the clock
        # forward to completion instead of sleeping.
        self._clock.advance(remaining) if hasattr(self._clock, "advance") else None
        token.state = TokenState.COMPLETE
        token.completed_ns = self._clock()
        return token.state

    def _memory_totals_native(self) -> Dict[str, Optional[int]]:
        return {
            "reserved_bytes": None,
            "device_total_bytes": self._capacity_bytes,
            "system_total_bytes": None,
            "scope": ("owned bytes are exact; mock device capacity is configured, "
                      "system total is unknown"),
        }
