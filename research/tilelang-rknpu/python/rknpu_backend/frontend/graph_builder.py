# SPDX-License-Identifier: GPL-3.0-or-later
"""Graph construction helper for the remaining adapter source forms.

The adapter recognizer decides what a kernel computes; this module provides the
single RKIR builder it uses.  The former native/Relax carrier has been retired,
so there is no second container or record type here.
"""
from __future__ import annotations

from typing import Any

from .. import ir


class RkirBuilder:
    """Construct internal RKIR values, ops and modules."""

    kind = "rkir"

    def value(self, *args: Any, **kwargs: Any):
        return ir.Value(*args, **kwargs)

    def op(self, *args: Any, **kwargs: Any):
        return ir.Op(*args, **kwargs)

    def finish(self, *, name: str, values: Any, ops: Any, inputs: Any,
               outputs: Any, profile_hash: str = "", source_hash: str = "",
               state_signature: str = "") -> Any:
        return ir.Module(name=name, values=tuple(values), ops=tuple(ops),
                         inputs=tuple(inputs), outputs=tuple(outputs),
                         profile_hash=profile_hash, source_hash=source_hash,
                         state_signature=state_signature)


_DEFAULT_BUILDER = RkirBuilder()


def active_builder() -> RkirBuilder:
    """The one surviving graph builder used by the adapter."""
    return _DEFAULT_BUILDER
