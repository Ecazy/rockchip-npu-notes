#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""RK182X backend adapter (RKNN3 coprocessor).

The RK182X is a *different device with a different ABI*.  It consumes RKNN3 model /
session buffers and a runtime API; it does not consume RK3588 register-command words,
it has no RK3588 ioctl/struct, and no cross-device DMA alias is assumed.  This adapter
therefore never reuses the RK3588 bridge and rejects any RK3588 payload outright.

No native RKNN3 execution binding is installed in this workspace, so the adapter **enforces the
unavailable state**: execution capabilities are UNAVAILABLE and every execution
operation raises :class:`UnsupportedOperation`.  What is provided instead is an
*injected runtime contract* -- a small Protocol plus a version/platform/capability
check -- so a future real SDK can be bound without changing the scheduler.  The only
runtime shipped here is a fake marked ``contract_only``; it exercises the binding
checks and can never execute.

Migration: the planned path is host staging (download on the source
adapter, upload on the destination), unavailable until native execution is bound.  Peer-to-peer transfer is UNAVAILABLE and is not
claimed until an SDK proves a device-to-device path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Dict, Mapping, Optional, Protocol, Tuple, runtime_checkable

from .interface import (
    BackendError,
    BaseBackend,
    BackendMismatch,
    Capability,
    CapabilityState,
    CompletionToken,
    DeviceIdentity,
    DeviceUnavailable,
    Evidence,
    InvalidPayload,
    Payload,
    TokenState,
    UnknownSDKBehavior,
    UnsupportedOperation,
    now_ns,
)

__all__ = [
    "RK182XPayload",
    "RKNN3RuntimeContract",
    "RKNN3Runtime",
    "ContractOnlyRKNN3Runtime",
    "RK182XBackend",
]


@dataclass
class RK182XPayload(Payload):
    """Opaque RKNN3 model/session work description.

    This is a *distinct ABI* from :class:`backends.rk3588.RK3588Payload`: it names an
    RKNN3 model blob, not 64-bit register-command words.
    """

    backend: ClassVar[str] = "rk182x"
    model_blob: bytes = b""
    model_name: str = ""
    session: bool = False
    inputs: Tuple[object, ...] = ()
    outputs: Tuple[object, ...] = ()

    def describe(self) -> str:
        kind = "session" if self.session else "model"
        return (f"rk182x({kind}={self.model_name or '<unnamed>'},"
                f"blob_bytes={len(self.model_blob)})")


@dataclass(frozen=True)
class RKNN3RuntimeContract:
    """What a future real RKNN3 runtime must satisfy to be bound."""

    required_api_major: int = 1
    required_api_minor: int = 1
    required_platform: str = "rk182x"
    #: Public entry points the adapter will call once a real SDK is available.
    required_entry_points: Tuple[str, ...] = (
        "rknn3_run",
        "rknn3_create_mem",
        "rknn3_mem_sync",
    )
    notes: str = (
        "Contract mirrors the public rknn3-api surface (rknn3_find_devices, "
        "RKNN3_QUERY_SDK_VERSION, rknn3_create_mem/_sync, rknn3_run).  It is a binding "
        "check only; no SDK library is present in this workspace."
    )


@runtime_checkable
class RKNN3Runtime(Protocol):
    """The minimal surface the adapter needs from an injected runtime.

    A future implementation wraps ``librknn3_api`` with ctypes or a C shim.  An object
    with these three methods (and ``contract_only``) can be bound, but execution still
    requires ``contract_only`` to be explicitly False.
    """

    def api_version(self) -> Tuple[int, int, int]:
        ...

    def platform(self) -> str:
        ...

    def capabilities(self) -> Mapping[str, bool]:
        ...


@dataclass
class ContractOnlyRKNN3Runtime:
    """Fake runtime used by tests to exercise the binding checks.

    It is *always* marked ``contract_only``: the adapter will bind it, verify the
    version/platform/capabilities, and still refuse to execute.  It never loads a
    library or touches a device.
    """

    api_version_value: Tuple[int, int, int] = (1, 1, 0)
    platform_name: str = "rk182x"
    capability_map: Mapping[str, bool] = field(default_factory=lambda: {
        "rknn3_run": True,
        "rknn3_create_mem": True,
        "rknn3_mem_sync": True,
    })
    contract_only: bool = True

    def api_version(self) -> Tuple[int, int, int]:
        return tuple(self.api_version_value)  # type: ignore[return-value]

    def platform(self) -> str:
        return self.platform_name

    def capabilities(self) -> Mapping[str, bool]:
        return dict(self.capability_map)


class RK182XBackend(BaseBackend):
    """Separate RK182X adapter.  Execution is unavailable until a real SDK is bound."""

    def __init__(self, *, runtime: Optional[object] = None,
                 contract: Optional[RKNN3RuntimeContract] = None, clock=now_ns) -> None:
        super().__init__(clock=clock)
        self._contract = contract or RKNN3RuntimeContract()
        self._runtime: Optional[object] = None
        self._bound_info: Dict[str, object] = {}
        self._identity = DeviceIdentity(
            backend="rk182x",
            device_name="RK182X",
            vendor="Rockchip",
            platform="rk182x",
            driver="RKNN3 runtime (not present)",
            api_version="unbound",
            evidence=Evidence.CONTRACT,
            notes=("opaque RKNN3 model/session ABI; no RK3588 regcmd, no reused "
                   "ioctl/struct, host-staged migration only, P2P unproven"),
        )
        if runtime is not None:
            self.bind_runtime(runtime)

    # -- runtime binding ---------------------------------------------------------

    def bind_runtime(self, runtime: object) -> Dict[str, object]:
        """Validate an injected runtime against the contract and bind it.

        Version/platform/capability problems raise immediately.  A ``contract_only``
        runtime binds but does not enable execution.
        """
        if runtime is None:
            raise DeviceUnavailable("no runtime supplied")
        if not isinstance(runtime, RKNN3Runtime):
            # Structural check via runtime_checkable Protocol.
            raise UnknownSDKBehavior("runtime does not implement RKNN3Runtime")

        try:
            raw_version = runtime.api_version()
        except Exception as exc:  # pragma: no cover - defensive
            raise UnknownSDKBehavior(f"runtime.api_version() failed: {exc}") from exc
        if (not isinstance(raw_version, tuple) or len(raw_version) < 2
                or not all(type(v) is int and v >= 0 for v in raw_version[:3])):
            raise UnknownSDKBehavior(f"unrecognised runtime version {raw_version!r}")
        major, minor = raw_version[0], raw_version[1]
        if major != self._contract.required_api_major or (major, minor) < (self._contract.required_api_major,
                             self._contract.required_api_minor):
            raise UnsupportedOperation(
                f"RKNN3 runtime {major}.{minor} is older than required "
                f"{self._contract.required_api_major}.{self._contract.required_api_minor}"
            )

        try:
            platform = runtime.platform()
        except Exception as exc:  # pragma: no cover - defensive
            raise UnknownSDKBehavior(f"runtime.platform() failed: {exc}") from exc
        if platform != self._contract.required_platform:
            raise BackendMismatch(
                f"runtime platform {platform!r} != {self._contract.required_platform!r}"
            )

        try:
            caps = dict(runtime.capabilities())
        except Exception as exc:  # pragma: no cover - defensive
            raise UnknownSDKBehavior(f"runtime.capabilities() failed: {exc}") from exc
        if any(type(v) is not bool for v in caps.values()):
            raise UnknownSDKBehavior("capabilities must be Boolean")
        missing = [e for e in self._contract.required_entry_points if not caps.get(e)]
        if missing:
            raise UnsupportedOperation(f"runtime lacks required entry points: {missing}")

        contract_only = getattr(runtime, "contract_only", True)
        if contract_only is not True:
            raise UnsupportedOperation("this adapter currently accepts contract-only runtime binding; native SDK binding not implemented")
        self._runtime = runtime
        self._bound_info = {
            "api_version": (major, minor, raw_version[2] if len(raw_version) > 2 else 0),
            "platform": platform,
            "contract_only": contract_only,
        }
        self._identity = DeviceIdentity(
            backend="rk182x",
            device_name="RK182X",
            vendor="Rockchip",
            platform=platform,
            driver="RKNN3 runtime",
            api_version=".".join(str(v) for v in self._bound_info["api_version"]),
            evidence=Evidence.CONTRACT,
            notes=("contract-only binding; execution remains UnsupportedOperation until "
                   "a real, non-contract runtime is supplied"),
        )
        return dict(self._bound_info)

    def runtime_bound(self) -> bool:
        return self._runtime is not None

    def runtime_contract_only(self) -> bool:
        return bool(self._bound_info.get("contract_only", True))

    # -- identity / capabilities -------------------------------------------------

    @property
    def identity(self) -> DeviceIdentity:
        return self._identity

    def _execution_available(self) -> bool:
        return False  # a future native binding must implement the full SDK contract

    def _capabilities(self) -> Dict[Capability, CapabilityState]:
        executable = self._execution_available()
        gate = CapabilityState.AVAILABLE if executable else CapabilityState.UNAVAILABLE
        return {
            Capability.SUBMIT: gate,
            Capability.WAIT: gate,
            Capability.UPLOAD: gate,
            Capability.DOWNLOAD: gate,
            Capability.QUERY_MEMORY: gate,
            Capability.ASYNC_COMPLETION: gate,
            # Host staging is a modelling claim we can always make; it does not need
            # the device.  It is not a hardware or bandwidth claim.
            Capability.HOST_STAGED_MIGRATION: gate,
            # No P2P path is proven by the public SDK; never claim it.
            Capability.PEER_TO_PEER_TRANSFER: CapabilityState.UNAVAILABLE,
            # RK182X does not consume RK3588 register-command words.
            Capability.REGCMD_PROGRAM: CapabilityState.UNAVAILABLE,
            Capability.RKNN3_MODEL: gate,
        }

    # -- lifecycle ---------------------------------------------------------------

    def _open_native(self) -> None:
        # There is no local runtime to open.  Opening succeeds (the adapter can be
        # inspected and bound) but execution stays gated by capabilities.
        if self._runtime is None:
            return

    def _close_native(self) -> None:
        for job in list(self._jobs.values()):
            job.released = True
        self._jobs.clear()
        for buf in list(self._buffers.values()):
            buf.released = True
        self._buffers.clear()

    # -- native hooks ------------------------------------------------------------

    def _require_executable(self) -> None:
        if self._runtime is None:
            raise UnsupportedOperation(
                "rk182x: no RKNN3 runtime bound; execution unavailable"
            )
        if self.runtime_contract_only():
            raise UnsupportedOperation(
                "rk182x: bound runtime is contract-only; execution unavailable"
            )

    def _alloc_native(self, token_id: int, payload_bytes: int, purpose: str):
        self._require_executable()
        # A real binding would call rknn3_create_mem here.
        runtime = self._runtime
        if not hasattr(runtime, "create_mem"):
            raise UnsupportedOperation("rk182x runtime has no create_mem binding")
        native = runtime.create_mem(payload_bytes, purpose)  # pragma: no cover
        return payload_bytes, native

    def _free_native(self, buffer) -> None:
        runtime = self._runtime
        if runtime is not None and hasattr(runtime, "free_mem"):
            runtime.free_mem(buffer.native)  # pragma: no cover

    def _upload_native(self, buffer, offset: int, data: bytes) -> None:
        self._require_executable()
        runtime = self._runtime
        if not hasattr(runtime, "mem_sync"):
            raise UnsupportedOperation("rk182x runtime has no mem_sync binding")
        runtime.mem_sync(buffer.native, offset, data, True)  # pragma: no cover

    def _download_native(self, buffer, offset: int, length: int) -> bytes:
        self._require_executable()
        runtime = self._runtime
        if not hasattr(runtime, "mem_sync"):
            raise UnsupportedOperation("rk182x runtime has no mem_sync binding")
        return runtime.mem_sync(buffer.native, offset, length, False)  # pragma: no cover

    def _create_job_native(self, payload: RK182XPayload) -> object:
        if not isinstance(payload, RK182XPayload):
            raise InvalidPayload("rk182x expects an RK182XPayload")
        # Refuse any RK3588 regcmd payload that slipped past the backend check.
        if hasattr(payload, "words"):
            raise BackendMismatch("rk182x refuses RK3588 register-command payloads")
        self._require_executable()
        if not payload.model_blob and not payload.model_name:
            raise InvalidPayload("payload must name an RKNN3 model or session")
        return {"model_blob": payload.model_blob, "session": payload.session}

    def _submit_native(self, job, token: CompletionToken) -> None:
        self._require_executable()
        runtime = self._runtime
        if not hasattr(runtime, "run_model"):
            raise UnsupportedOperation("rk182x runtime has no run_model binding")
        runtime.run_model(job.native)  # pragma: no cover
        token.state = TokenState.COMPLETE
        token.completed_ns = self._clock()

    def _wait_native(self, token, timeout_ns: Optional[int]) -> TokenState:
        return token.state

    # -- memory ------------------------------------------------------------------

    def _memory_totals_native(self) -> Dict[str, Optional[int]]:
        # A real binding would read RKNN3_QUERY_DEVICE_MEM_INFO / _CORE_MEM_SIZE.
        # Without one, no total is known.
        return {
            "reserved_bytes": None,
            "device_total_bytes": None,
            "system_total_bytes": None,
            "scope": ("RK182X memory totals require a bound RKNN3 runtime "
                      "(RKNN3_QUERY_DEVICE_MEM_INFO); none is available"),
        }

    # -- migration ---------------------------------------------------------------

    def migration_plan(self) -> Dict[str, object]:
        """Describe the only migration path this adapter will admit."""
        return {
            "host_staged": True,
            "peer_to_peer": False,
            "bytes_scope": "payload bytes only; alignment and runtime overhead unknown",
            "timing_scope": "unmeasured; no bandwidth claim without an SDK and hardware",
            "note": ("stage by download() on the source adapter and upload() on the "
                     "destination; P2P stays unavailable until the SDK proves it"),
        }
