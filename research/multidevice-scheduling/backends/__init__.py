#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Multi-device backend adapters.

Each device gets its own adapter behind the uniform contract in
:mod:`backends.interface`.  There is deliberately no auto-detection: a scheduler names
the backend it wants.  The known adapters are:

* ``rk3588``  - calls the existing custom-ops implementation through a built ctypes
  bridge (synchronous submit; real register-command ABI).
* ``rk182x``  - an RKNN3 coprocessor boundary.  No SDK is present, so execution is
  unavailable and only an injected runtime contract is offered.
* ``mock``    - an in-memory contract reference; never a hardware claim.
"""

from .interface import (
    Backend,
    BackendError,
    BackendMismatch,
    BaseBackend,
    transfer_between,
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
    MAX_TRANSFER_BYTES,
    MemoryBoundsError,
    MemoryInfo,
    OperationTimeout,
    Payload,
    TokenState,
    UnknownSDKBehavior,
    UnsupportedOperation,
)
from .mock import FakeClock, MockBackend, MockPayload
from .rk3588 import (
    BRIDGE_ENV,
    MODE_CHAIN,
    MODE_SEPARATE,
    RK3588Backend,
    RK3588Payload,
    find_bridge_library,
)
from .rk182x import (
    ContractOnlyRKNN3Runtime,
    RK182XBackend,
    RK182XPayload,
    RKNN3Runtime,
    RKNN3RuntimeContract,
)

#: Named adapter constructors.  Callers that need a configured instance (custom clock,
#: injected bridge/runtime) construct the class directly.
ADAPTERS = {
    "rk3588": RK3588Backend,
    "rk182x": RK182XBackend,
    "mock": MockBackend,
}

__all__ = [
    "ADAPTERS",
    "Backend",
    "BackendError",
    "BackendMismatch",
    "BaseBackend",
    "transfer_between",
    "BufferHandle",
    "BufferOwnershipError",
    "Capability",
    "CapabilityState",
    "CompletionToken",
    "DeviceIdentity",
    "DeviceUnavailable",
    "Evidence",
    "InvalidPayload",
    "JobHandle",
    "MAX_TRANSFER_BYTES",
    "MemoryBoundsError",
    "MemoryInfo",
    "OperationTimeout",
    "Payload",
    "TokenState",
    "UnknownSDKBehavior",
    "UnsupportedOperation",
    "FakeClock",
    "MockBackend",
    "MockPayload",
    "BRIDGE_ENV",
    "MODE_CHAIN",
    "MODE_SEPARATE",
    "RK3588Backend",
    "RK3588Payload",
    "find_bridge_library",
    "ContractOnlyRKNN3Runtime",
    "RK182XBackend",
    "RK182XPayload",
    "RKNN3Runtime",
    "RKNN3RuntimeContract",
]
