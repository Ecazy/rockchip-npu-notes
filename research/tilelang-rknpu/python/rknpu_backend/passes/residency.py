# SPDX-License-Identifier: GPL-3.0-or-later
"""Static Host-residency and execution-surface analysis for rkhybrid graphs.

The generic hybrid executor materialises every tensor into a fresh NumPy array,
then revalidates each plan binding on every call.  That contract is convenient
for an importer, but a frozen hybrid graph already proves which node produces
each tensor and that tensor names never alias.  This pass turns those facts into
one immutable residency plan:

* every graph tensor gets exactly one caller-owned Host slot for the lifetime of
  a runner;
* the producer kind of every slot is recorded so a lowering pass can preserve
  the C runtime's per-run finite-value gate instead of duplicating it in Python;
* all direct RKPlan inputs and public outputs are enumerated without consulting
  model names or dimensions.

Tensor views inside a slot (repeat-head slices, row windows, transposes consumed
by ``PlanCall``) remain the runner's responsibility.  This pass only licenses the
buffer-identity and lifetime facts on which those prebound views depend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..errors import (RK_E_ALIAS, RK_E_INVALID, RK_E_SCHEMA,
                      RK_E_SHAPE, reject)

_DTYPES = {
    "uint8": np.dtype(np.uint8),
    "float16": np.dtype(np.float16),
    "float32": np.dtype(np.float32),
}
ProducerKind = Literal["host_action", "rkplan"]


@dataclass(frozen=True)
class ResidentSlot:
    """One stable Host buffer owned by a compiled runner."""

    tensor: str
    dtype: str
    shape: tuple[int, ...]
    producer: str
    producer_kind: ProducerKind
    consumers: tuple[str, ...]
    is_model_output: bool
    is_trace: bool
    capture_only: bool

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) * _DTYPES[
            self.dtype].itemsize


@dataclass(frozen=True)
class ResidencyPlan:
    """Compiler result consumed by ``HybridRunner``'s prepared lowering."""

    slots: tuple[ResidentSlot, ...]
    npu_input_tensors: tuple[str, ...]
    host_output_tensors: tuple[str, ...]
    public_output_tensors: tuple[str, ...]
    total_bytes: int

    def slot_map(self) -> dict[str, ResidentSlot]:
        return {slot.tensor: slot for slot in self.slots}

    def allocate(self) -> dict[str, np.ndarray]:
        """Allocate one writable, C-contiguous backing array per slot.

        The arrays intentionally remain uninitialised.  They are outputs of a
        producer in the graph and must be written before any consumer runs; a
        finite-value check at this point would inspect uninitialised storage and
        would not be a producer guarantee.  The C plan boundary performs the
        authoritative finite scan when a resident value first crosses into an
        RKPlan, exactly as it does in the generic path.
        """
        values: dict[str, np.ndarray] = {}
        identities: dict[tuple[int, int], str] = {}
        for slot in self.slots:
            array = np.empty(slot.shape, dtype=_DTYPES[slot.dtype])
            if array.dtype != _DTYPES[slot.dtype] or array.shape != slot.shape:
                reject(RK_E_SHAPE, "resident buffer materialisation mismatch",
                       stage="lowering", actual=list(array.shape),
                       required=list(slot.shape))
            if not array.flags.c_contiguous or not array.flags.writeable:
                reject(RK_E_INVALID, "resident buffer is not writable and "
                       "C-contiguous", stage="lowering")
            key = (int(array.ctypes.data), int(array.nbytes))
            if key in identities:
                reject(RK_E_ALIAS,
                       "resident buffers must not share storage",
                       stage="lowering",
                       actual=[identities[key], slot.tensor])
            identities[key] = slot.tensor
            values[slot.tensor] = array
        return values

    def total_bytes_for(self, tensors) -> int:
        by_name = self.slot_map()
        return sum(by_name[name].nbytes for name in set(tensors))


def _node_refs(node) -> tuple[tuple[str, ...], tuple[str, ...]]:
    inputs = tuple(ref.tensor for ref in node.inputs.values())
    outputs = tuple(ref.tensor for ref in node.outputs.values())
    return inputs, outputs


def analyze_residency(program) -> ResidencyPlan:
    """Prove and describe stable Host residency for a frozen hybrid graph.

    Rejections are intentionally independent of the manifest parser.  The parser
    proves graph well-formedness; this pass proves the stronger facts required by
    prebound descriptors: one producer per slot, no tensor alias between slots,
    and no consumer before the slot has a producer in the declared order.
    """
    producers: dict[str, tuple[str, ProducerKind, bool]] = {
        program.input_spec.name: ("$input", "host_action", False),
    }
    producers.update({
        name: ("$host_constant", "host_action", False)
        for name in program.host_constants
    })
    consumers: dict[str, list[str]] = {name: [] for name in program.tensors}
    npu_inputs: set[str] = set()
    host_outputs: set[str] = set()

    for node in program.nodes:
        inputs, outputs = _node_refs(node)
        for name in inputs:
            if name not in producers:
                reject(RK_E_SCHEMA,
                       f"{node.id} reads resident tensor {name!r} before its "
                       "producer", stage="lowering")
            consumers.setdefault(name, []).append(node.id)
        # NpuNode/HostNode are deliberately not imported here to avoid a cycle.
        kind = "host_action" if hasattr(node, "action") else "rkplan"
        if kind == "rkplan":
            npu_inputs.update(name for name in inputs
                              if name in program.tensors)
        for name in outputs:
            if name in producers:
                reject(RK_E_SCHEMA,
                       f"resident tensor {name!r} has more than one producer",
                       stage="lowering", actual=[producers[name][0], node.id])
            if name not in program.tensors:
                reject(RK_E_SCHEMA,
                       f"node {node.id} writes undeclared tensor {name!r}",
                       stage="lowering")
            producers[name] = (node.id, kind, bool(
                getattr(node, "capture_only", False)))
            if kind == "host_action":
                host_outputs.add(name)

    missing = set(program.tensors) - set(producers)
    if missing:
        reject(RK_E_SCHEMA, "resident graph has unproduced tensors",
               stage="lowering", actual=sorted(missing))

    public_tensors = set(program.outputs.values())
    trace_tensors = set(program.trace_tensors)
    slots: list[ResidentSlot] = []
    for name, spec in program.tensors.items():
        producer, kind, capture_only = producers[name]
        if spec.dtype not in _DTYPES:
            reject(RK_E_SCHEMA, "resident tensor dtype is not materialisable",
                   stage="lowering", actual=spec.dtype)
        consumers_for_slot = tuple(consumers.get(name, ()))
        # A declared tensor may legitimately be trace-only (the capture split
        # outputs are one example); unused executable tensors are not silently
        # treated as residency roots.
        if not consumers_for_slot and name not in public_tensors and \
                name not in trace_tensors:
            reject(RK_E_SCHEMA,
                   "resident tensor has no consumer, public output or trace use",
                   stage="lowering", actual=name)
        slots.append(ResidentSlot(
            tensor=name, dtype=spec.dtype, shape=tuple(spec.shape),
            producer=producer, producer_kind=kind,
            consumers=consumers_for_slot,
            is_model_output=name in public_tensors,
            is_trace=name in trace_tensors,
            capture_only=capture_only))

    # Public outputs are copied to fresh arrays by the runner.  They must not be
    # capture-only because exposing a possibly-unwritten buffer would change the
    # generic output contract.
    by_name = {slot.tensor: slot for slot in slots}
    for name in sorted(public_tensors):
        if name not in by_name:
            reject(RK_E_SCHEMA, "public output has no resident slot",
                   stage="lowering", actual=name)
        if by_name[name].capture_only:
            reject(RK_E_SCHEMA, "capture-only tensor cannot be a public output",
                   stage="lowering", actual=name)

    total = sum(slot.nbytes for slot in slots)
    if total <= 0:
        reject(RK_E_SHAPE, "resident execution plan has no bytes",
               stage="lowering")
    return ResidencyPlan(
        slots=tuple(slots),
        npu_input_tensors=tuple(sorted(npu_inputs)),
        host_output_tensors=tuple(sorted(host_outputs)),
        public_output_tensors=tuple(sorted(public_tensors)),
        total_bytes=total,
    )
