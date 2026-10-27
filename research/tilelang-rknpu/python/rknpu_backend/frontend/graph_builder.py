# SPDX-License-Identifier: GPL-3.0-or-later
"""Graph-builder seam for the structural recognizer.

The TileLang recognizer decides *what* the kernel computes; it must not decide
*which container* records that result.  This module is the seam between the two:

* :class:`RkirBuilder` (the default) constructs the internal RKIR records exactly
  as before -- ``ir.Value``/``ir.Op``/``ir.Module`` and therefore every existing
  structural and numeric check;
* :class:`RecordBuilder` constructs neutral, container-free records that the
  native TVM builder turns into ``tvm.relax`` nodes without ever instantiating an
  RKIR value, op or module.

The recognizer calls ``active_builder()`` in its eight graph-construction sites,
so recognition code has exactly one copy while the two carriers stay independent.

Neutral records carry no schema validation of their own.  The RKIR carrier keeps
its checks where they always were (the record ``__post_init__``), and the native
carrier relies on upstream Relax well-formedness plus the shared planner checks --
which is precisely the trade-off the native-graph evaluation has to measure.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from .. import ir


@dataclass(frozen=True)
class ValueSpec:
    """Container-free description of one recognized value."""

    id: int
    name: str
    type: Any
    producer: int | None
    role: str


@dataclass(frozen=True)
class OpSpec:
    """Container-free description of one recognized operation."""

    id: int
    opcode: str
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    attrs: dict[str, Any]
    source: str


@dataclass(frozen=True)
class GraphSpec:
    """Container-free recognition result handed to a concrete graph builder."""

    name: str
    values: tuple[ValueSpec, ...]
    ops: tuple[OpSpec, ...]
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    profile_hash: str = ""
    source_hash: str = ""
    state_signature: str = ""


class RkirBuilder:
    """Default carrier: the internal RKIR module (unchanged behaviour)."""

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


class RecordBuilder:
    """Neutral carrier: no RKIR object is created at recognition time."""

    kind = "recognized"

    def value(self, *args: Any, **kwargs: Any) -> ValueSpec:
        if kwargs:
            return ValueSpec(**kwargs)
        return ValueSpec(*args)

    def op(self, *args: Any, **kwargs: Any) -> OpSpec:
        if kwargs:
            return OpSpec(**kwargs)
        return OpSpec(*args)

    def finish(self, *, name: str, values: Any, ops: Any, inputs: Any,
               outputs: Any, profile_hash: str = "", source_hash: str = "",
               state_signature: str = "") -> GraphSpec:
        return GraphSpec(name=name, values=tuple(values), ops=tuple(ops),
                         inputs=tuple(inputs), outputs=tuple(outputs),
                         profile_hash=profile_hash, source_hash=source_hash,
                         state_signature=state_signature)


_DEFAULT_BUILDER = RkirBuilder()
# A ContextVar keeps two concurrent recognitions from seeing each other's carrier
# without threading a builder argument through every extraction helper.
_ACTIVE: ContextVar[Any] = ContextVar("rknpu_graph_builder")


def active_builder() -> Any:
    return _ACTIVE.get(_DEFAULT_BUILDER)


class use_builder:
    """Context manager selecting the carrier for one recognition."""

    def __init__(self, builder: Any):
        self._builder = builder
        self._token = None

    def __enter__(self) -> Any:
        self._token = _ACTIVE.set(self._builder)
        return self._builder

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._token is not None:
            _ACTIVE.reset(self._token)
            self._token = None
