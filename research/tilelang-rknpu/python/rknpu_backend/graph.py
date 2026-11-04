# SPDX-License-Identifier: GPL-3.0-or-later
"""The compute-graph access protocol shared by the planner and its carriers.

The planner is written against a handful of read-only attributes rather than a
concrete graph class.  The remaining adapter graph form is the internal RKIR
:class:`rknpu_backend.ir.Module`; migrated paths pass an ordered target-task
sequence to the task consumer instead.  A graph view must expose ``name``,
``source_hash``, ``values``, ``ops``, ``inputs``, ``outputs`` and ``value(id)``
and must not accumulate a second copy of the graph.

This module has no imports from the rest of the package, so both ``passes`` and
``codegen`` can depend on it without a cycle.
"""
from __future__ import annotations

from typing import Any, Iterable, Protocol, runtime_checkable

GRAPH_ATTRIBUTES = ("name", "source_hash", "values", "ops", "inputs", "outputs",
                    "value")


@runtime_checkable
class ComputeGraph(Protocol):
    name: str
    source_hash: str
    values: Iterable[Any]
    ops: Iterable[Any]
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]

    def value(self, value_id: int) -> Any:  # pragma: no cover - protocol
        ...


def is_compute_graph(obj: Any) -> bool:
    return all(hasattr(obj, name) for name in GRAPH_ATTRIBUTES)


def missing_graph_attributes(obj: Any) -> list[str]:
    return sorted(name for name in GRAPH_ATTRIBUTES if not hasattr(obj, name))
