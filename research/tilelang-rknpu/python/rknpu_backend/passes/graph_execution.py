# SPDX-License-Identifier: GPL-3.0-or-later
"""Graph-level boundary analysis and metadata-only view lowering.

The prepared residency lowering removes repeated RKPlan binding construction,
but the executor still walks the manifest node list and reconstructs each Host
action dictionary on every call.  This pass extracts the parts that can be
compiled once from the graph and its tensor layout:

* metadata-only reshaping becomes a stable view of its input slot;
* trace-only values are distinguished from values required by production;
* every Host boundary carries the reason it exists, without treating all Host
  time as recoverable.

It deliberately does not turn model names into graph rules.  A reshape is
eligible only from the registered Host op, matching input/output dtypes and
element counts, and is checked against the actual C-contiguous resident view by
the runtime.  Boundary causes describe operator/layout/dependency facts and are
reported for planning; they do not change execution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from ..errors import RK_E_SCHEMA, RK_E_SHAPE, RK_E_UNSUPPORTED, reject

BoundaryCause = Literal[
    "numeric_semantics",
    "missing_device_operator",
    "physical_layout",
    "resource_interface",
    "current_implementation",
]

# Cause classes for the closed Host-action registry.  These are operator facts,
# not model branches.  ``missing_device_operator`` means the current verified
# device recipe set cannot express the operation; it does not claim impossibility.
_BOUNDARY_CAUSES: dict[str, tuple[BoundaryCause, ...]] = {
    "mnist.normalize_flatten.v1": (
        "numeric_semantics", "current_implementation"),
    "mnist.patchify_norm.v1": (
        "numeric_semantics", "physical_layout", "current_implementation"),
    "tensor.pad_2d.v1": (
        "physical_layout", "current_implementation"),
    "tensor.reshape.v1": ("physical_layout",),
    "tensor.add.v1": ("numeric_semantics",),
    "tiny_vit.position_add.v1": (
        "numeric_semantics", "missing_device_operator"),
    "rmsnorm.last_axis.v1": (
        "numeric_semantics", "missing_device_operator"),
    "attention.scaled_softmax.v1": (
        "numeric_semantics", "missing_device_operator"),
    "tiny_vit.qkv_split_heads.v1": (
        "physical_layout", "current_implementation"),
    "tiny_vit.merge_heads.v1": (
        "physical_layout", "current_implementation"),
    "swiglu.packed_gate_up.v1": (
        "numeric_semantics", "missing_device_operator"),
    "swiglu.gate_up.v1": (
        "numeric_semantics", "missing_device_operator"),
    "tensor.mean_axis.v1": (
        "numeric_semantics", "missing_device_operator"),
    "logits.prefix.v1": (
        "numeric_semantics", "resource_interface"),
    "tensor.add_rmsnorm.last_axis.v1": (
        "numeric_semantics", "missing_device_operator"),
}

_DEVICE_GAPS: dict[str, tuple[str, ...]] = {
    "tiny_vit.position_add.v1": (
        "exact-shape FP16 FW add is available; broadcast position-add is not",
    ),
    "rmsnorm.last_axis.v1": (
        "device row reduction + rsqrt + coefficient materialisation recipe",
    ),
    "attention.scaled_softmax.v1": (
        "device stable max/exp/sum/normalize row reduction and bounded-domain "
        "numeric contract",
    ),
    "swiglu.packed_gate_up.v1": (
        "device exact sigmoid/SwiGLU or an independently accepted approximate "
        "recipe; current LUT path has a separate numeric policy",
    ),
    "swiglu.gate_up.v1": (
        "device exact sigmoid/SwiGLU or an independently accepted approximate "
        "recipe; current LUT path has a separate numeric policy",
    ),
    "tensor.mean_axis.v1": (
        "device FP32-preserving token reduction then one FP16 writeback",
    ),
}


@dataclass(frozen=True)
class HostBoundary:
    node_id: str
    action_id: str
    phase: str
    causes: tuple[BoundaryCause, ...]
    input_tensors: tuple[str, ...]
    output_tensors: tuple[str, ...]
    trace_only_outputs: tuple[str, ...]
    device_missing: tuple[str, ...]
    capture_only: bool
    lowered_as_view: bool = False


@dataclass(frozen=True)
class ViewAlias:
    node_id: str
    input_tensor: str
    output_tensor: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class GraphExecutionPlan:
    boundaries: tuple[HostBoundary, ...]
    view_aliases: tuple[ViewAlias, ...]
    production_tensors: tuple[str, ...]
    trace_only_tensors: tuple[str, ...]
    public_tensors: tuple[str, ...]

    def alias_for_node(self, node_id: str) -> ViewAlias | None:
        return next((item for item in self.view_aliases
                     if item.node_id == node_id), None)


def analyze_graph_execution(program) -> GraphExecutionPlan:
    """Analyze Host boundaries and prove metadata-only reshape candidates."""
    public_tensors = set(program.outputs.values())
    required = set(public_tensors)
    production_nodes = [
        node for node in program.nodes
        if not bool(getattr(node, "capture_only", False))
    ]
    for node in production_nodes:
        required.update(ref.tensor for ref in node.inputs.values())
    # A tensor may be both a public output and a trace value.
    trace_only = (set(program.trace_tensors) - required)

    aliases: list[ViewAlias] = []
    alias_ids: set[str] = set()
    for node in program.nodes:
        if not hasattr(node, "action") or node.capture_only:
            continue
        if node.action.action_id != "tensor.reshape.v1":
            continue
        if len(node.inputs) != 1 or len(node.outputs) != 1:
            reject(RK_E_SHAPE,
                   "reshape view lowering requires one input and one output",
                   stage="graph_execution", actual=node.id)
        input_ref = next(iter(node.inputs.values()))
        output_ref = next(iter(node.outputs.values()))
        if input_ref.index or output_ref.index:
            reject(RK_E_UNSUPPORTED,
                   "indexed reshape cannot become a graph view",
                   stage="graph_execution", actual=node.id)
        input_spec = program.tensors.get(input_ref.tensor)
        output_spec = program.tensors.get(output_ref.tensor)
        if input_spec is None or output_spec is None:
            reject(RK_E_SCHEMA, "reshape view names unknown tensor",
                   stage="graph_execution", actual=node.id)
        shape = tuple(int(v) for v in node.attrs.get("shape", ()))
        if tuple(output_spec.shape) != shape:
            reject(RK_E_SHAPE, "reshape view output does not match attrs",
                   stage="graph_execution", actual=node.id)
        if input_spec.dtype != output_spec.dtype:
            reject(RK_E_UNSUPPORTED,
                   "reshape view cannot change dtype",
                   stage="graph_execution", actual=node.id)
        if int(math.prod(input_spec.shape)) != int(math.prod(shape)):
            reject(RK_E_SHAPE, "reshape view must preserve element count",
                   stage="graph_execution", actual=node.id)
        aliases.append(ViewAlias(node.id, input_ref.tensor,
                                 output_ref.tensor, shape))
        alias_ids.add(node.id)

    boundaries: list[HostBoundary] = []
    for node in program.nodes:
        if not hasattr(node, "action"):
            continue
        inputs, outputs = [], []
        for ref in node.inputs.values():
            if ref.tensor not in inputs:
                inputs.append(ref.tensor)
        for ref in node.outputs.values():
            if ref.tensor not in outputs:
                outputs.append(ref.tensor)
        trace_outputs = tuple(name for name in outputs
                              if name in trace_only)
        boundaries.append(HostBoundary(
            node_id=node.id,
            action_id=node.action.action_id,
            phase=node.phase,
            causes=_BOUNDARY_CAUSES.get(
                node.action.action_id, ("current_implementation",)),
            input_tensors=tuple(inputs),
            output_tensors=tuple(outputs),
            trace_only_outputs=trace_outputs,
            device_missing=_DEVICE_GAPS.get(node.action.action_id, ()),
            capture_only=bool(node.capture_only),
            lowered_as_view=node.id in alias_ids,
        ))
    return GraphExecutionPlan(
        boundaries=tuple(boundaries),
        view_aliases=tuple(aliases),
        production_tensors=tuple(sorted(required)),
        trace_only_tensors=tuple(sorted(trace_only)),
        public_tensors=tuple(sorted(public_tensors)),
    )
