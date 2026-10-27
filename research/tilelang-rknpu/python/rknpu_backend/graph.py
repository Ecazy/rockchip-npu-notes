# SPDX-License-Identifier: GPL-3.0-or-later
"""The compute-graph access protocol shared by the planner and its carriers.

The planner is written against a handful of read-only attributes rather than a
concrete graph class, so the same legalization/draft/storage code can run over
different carriers without copying the algorithm:

* the internal RKIR :class:`rknpu_backend.ir.Module` (legacy and default), and
* the experimental native TVM graph view in
  :mod:`rknpu_backend.frontend.native_graph`.

A carrier must expose ``name``, ``source_hash``, ``values``, ``ops``, ``inputs``,
``outputs`` and ``value(id)``.  It must not accumulate a second copy of the graph:
these are projections over the carrier's own storage.

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
