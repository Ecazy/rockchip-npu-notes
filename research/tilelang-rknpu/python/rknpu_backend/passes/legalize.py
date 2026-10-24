# SPDX-License-Identifier: GPL-3.0-or-later
"""Capability legalization for the minimal FP16 path."""
from __future__ import annotations

from dataclasses import dataclass

from ..capabilities import TargetProfile
from ..errors import (RK_E_NUMERIC_CONTRACT, RK_E_SHAPE, RK_E_UNSUPPORTED, reject)
from ..ir import Module, NumericContract, Op
from ..options import CompileOptions

ACTIVATION_CAPABILITY = {
    "relu": "co.ew_max.f16.v1",
    "silu": "co.lut.sigmoid.f16.v1",
    "gelu": "co.lut.gelu_gate.f16.v1",
    "sigmoid": "co.lut.sigmoid.f16.v1",
    "hardsigmoid": "co.lut.hardsigmoid.f16.v1",
}
BINARY_CAPABILITY = {"add": "co.ew_add.f16.v1", "mul": "co.ew_mul.f16.v1"}


@dataclass(frozen=True)
class LegalizedOp:
    op_id: int
    capability_id: str
    template_id: str


@dataclass(frozen=True)
class SwiGLUFFN:
    """The matched ``Y = (SiLU(X@Wg^T) * (X@Wu^T)) @ Wd^T + X`` graph."""
    matmul_gate: Op
    matmul_up: Op
    activation: Op
    multiply: Op
    matmul_proj: Op
    add: Op
    input_id: int
    hidden: int
    output: int


@dataclass(frozen=True)
class LegalizedGraph:
    module: Module
    ops: tuple[LegalizedOp, ...]
    matmul_op: Op | None
    activation_op: Op | None
    binary_op: Op | None = None
    # The runtime-tensor operand of the add/mul, resolved from the actual graph
    # edges.  ``dag`` must use this instead of assuming ``inputs[1]``.
    binary_operand_id: int | None = None
    swiglu: "SwiGLUFFN | None" = None
    # Multi-node static dataflow.  ``graph_ops`` is the per-node capability list
    # in module.topo order; ``None`` keeps the older closed-pattern builders on
    # their existing paths.
    graph_ops: tuple[LegalizedOp, ...] | None = None


def legalize(module: Module, *, profile: TargetProfile,
             options: CompileOptions) -> LegalizedGraph:
    """Select registered capabilities for a connected static graph.

    The classic closed shapes (one matmul, optionally followed by one activation
    and one same-shape add/mul) retain their existing path.  Multi-node and
    multi-output graphs use the per-node legalizer.

    The experimental whole-graph SwiGLU matcher is an **opt-in optimisation**,
    not a gateway: when ``allow_experimental`` is false the canonical SwiGLU
    graph falls through to the same per-node legalizer as any equivalent
    reordered graph.  When the opt-in is set, a matched specialisation must still
    satisfy its own capability, numeric and range constraints; a failure there is
    reported rather than silently rewritten.
    """
    ops = module.ops
    if options.allow_experimental:
        swiglu = _match_swiglu(module, options)
        if swiglu is not None:
            profile.experimental("co.swiglu_ffn.f16.v1")
            profile.enabled("co.mm.f16.v1")
            profile.enabled("co.lut.sigmoid.f16.v1")
            profile.enabled("co.ew_mul.f16.v1")
            profile.enabled("co.ew_add.f16.v1")
            return LegalizedGraph(
                module=module, ops=tuple(), matmul_op=swiglu.matmul_gate,
                activation_op=swiglu.activation,
                binary_op=swiglu.multiply, swiglu=swiglu)
    # Multi-output graphs, graphs with more than one matmul, and graphs with
    # more than three nodes use the explicit per-node static-dataflow
    # legalizer.  It proves every op is reachable from an export and has a
    # capability/numeric contract; it never hands an unconnected module to the
    # planner.  The classic single-matmul path remains for the K-loop and
    # multi-tile cases whose lowering is materially different.
    has_constant_broadcast = any(
        op.opcode in BINARY_CAPABILITY and
        any(module.value(vid).role == "constant" and
            module.value(vid).type.rank < 2 for vid in op.inputs)
        for op in ops)
    classic_shape = (1 <= len(ops) <= 3 and len(module.outputs) == 1 and
                     sum(op.opcode == "matmul" for op in ops) == 1 and
                     ops[0].opcode == "matmul" and
                     not has_constant_broadcast)
    if not classic_shape:
        return _legalize_static_dataflow(module, profile=profile, options=options)
    if not 1 <= len(ops) <= 3:
        reject(RK_E_UNSUPPORTED,
               "minimal MVP path accepts matmul plus at most one activation and "
               "one same-shape add/mul", stage="legalize", actual=len(ops))
    matmul = ops[0]
    if matmul.opcode != "matmul":
        reject(RK_E_UNSUPPORTED, "minimal MVP path must start with matmul",
               stage="legalize", op=matmul.opcode, source=matmul.source)
    contract = NumericContract.from_dict(matmul.attrs["numeric_contract"])
    if options.numeric_mode == "strict":
        reject(RK_E_NUMERIC_CONTRACT,
               "gemm requests FP32 strict accumulation; co.mm_acc.f16.v1 writes "
               "FP16 at each K tile",
               stage="legalize", op=matmul.opcode, source=matmul.source,
               target_profile=profile.profile_hash,
               actual=contract.to_dict(), required="rk_f16_tiled opt-in",
               alternative="pass numeric_mode=rk_f16_tiled to accept the declared "
                           "ordered K-tile writeback path")
    if not contract.allows_f16_tiled_path:
        reject(RK_E_NUMERIC_CONTRACT,
               "numeric contract does not declare ordered K-tile FP16 writeback",
               stage="legalize", op=matmul.opcode, source=matmul.source,
               target_profile=profile.profile_hash,
               actual=contract.to_dict(),
               alternative="declare narrowing at k_tile_boundary")
    profile.enabled("co.mm.f16.v1")
    profile.enabled("co.mm_acc.f16.v1")

    legalized = [LegalizedOp(matmul.id, "co.mm.f16.v1", "co.mm.f16.v1")]
    index = 1
    activation_op: Op | None = None
    produced = matmul.outputs[0]
    if index < len(ops) and ops[index].opcode in ACTIVATION_CAPABILITY:
        activation_op = ops[index]
        # The activation must consume the matmul result.  Matching only on the
        # opcode would let an unrelated activation slip into the chain and the
        # lowering would silently rewire the graph.
        if tuple(activation_op.inputs) != (produced,):
            reject(RK_E_UNSUPPORTED,
                   "activation does not consume the matmul result; the MVP only "
                   "lowers a single connected chain",
                   stage="legalize", op=activation_op.opcode,
                   source=activation_op.source,
                   actual={"activation_inputs": list(activation_op.inputs),
                           "matmul_output": produced},
                   alternative="express the program as one matmul followed by one "
                               "activation of its result")
        if activation_op.opcode != "relu" and not options.allow_approx_lut:
            reject(RK_E_NUMERIC_CONTRACT,
                   "activation uses a LUT approximation; explicit "
                   "allow_approx_lut is required",
                   stage="legalize", op=activation_op.opcode,
                   source=activation_op.source,
                   target_profile=profile.profile_hash,
                   alternative="pass allow_approx_lut=True after reviewing the "
                               "capability tolerance")
        capability = ACTIVATION_CAPABILITY[activation_op.opcode]
        profile.enabled(capability)
        if activation_op.opcode in {"silu", "gelu"}:
            profile.enabled("co.ew_mul.f16.v1")
        legalized.append(LegalizedOp(activation_op.id, capability, capability))
        produced = activation_op.outputs[0]
        index += 1

    binary_op: Op | None = None
    binary_operand_id: int | None = None
    swiglu: "SwiGLUFFN | None" = None
    if index < len(ops) and ops[index].opcode in BINARY_CAPABILITY:
        binary_op = ops[index]
        capability = BINARY_CAPABILITY[binary_op.opcode]
        profile.enabled(capability)
        # The binary must consume the value the previous node produced.  add/mul
        # are commutative, so either operand position is accepted, but the
        # lowering must then use the *other* operand, not a fixed index.
        if produced in binary_op.inputs:
            other = binary_op.inputs[0] if binary_op.inputs[1] == produced \
                else binary_op.inputs[1]
        else:
            reject(RK_E_UNSUPPORTED,
                   "add/mul does not consume the matmul/activation result; the "
                   "MVP only lowers a single connected chain",
                   stage="legalize", op=binary_op.opcode, source=binary_op.source,
                   actual={"binary_inputs": list(binary_op.inputs),
                           "produced_value": produced},
                   alternative="connect add/mul to the produced value or split the "
                               "graph into separate compilations")
        binary_operand_id = other
        operand = module.value(other)
        result = module.value(binary_op.outputs[0])
        if operand.role not in {"input", "constant"}:
            reject(RK_E_UNSUPPORTED,
                   "the second binary operand must be a runtime input or constant; "
                   "device-produced broadcast operands are not in this path",
                   stage="legalize", op=binary_op.opcode,
                   source=binary_op.source, actual=operand.role,
                   alternative="materialise the operand as an input binding")
        produced_value = module.value(produced)
        if operand.type != produced_value.type or operand.type != result.type:
            reject(RK_E_UNSUPPORTED,
                   "MVP add/mul requires exact same-shape operands; broadcast is "
                   "rejected", stage="legalize", op=binary_op.opcode,
                   source=binary_op.source,
                   actual=[list(operand.type.shape),
                           list(produced_value.type.shape)])
        legalized.append(LegalizedOp(binary_op.id, capability, capability))
        produced = binary_op.outputs[0]
        index += 1

    if index != len(ops):
        reject(RK_E_UNSUPPORTED,
               "unsupported operation after the closed matmul/activation/binary "
               "pattern", stage="legalize", op=ops[index].opcode,
               source=ops[index].source)

    # The exported output must be exactly the value the closed chain produces:
    # otherwise the plan would publish a different node than the program asks for.
    if tuple(module.outputs) != (produced,):
        reject(RK_E_UNSUPPORTED,
               "module output is not the value produced by the lowered chain",
               stage="legalize", actual={"module_outputs": list(module.outputs),
                                         "chain_output": produced},
               alternative="export only the final value of the supported chain")
    for value in module.values:
        if value.role == "output" and value.id != produced:
            reject(RK_E_UNSUPPORTED,
                   "module exports a value outside the lowered chain",
                   stage="legalize", actual=value.id)
    return LegalizedGraph(module=module, ops=tuple(legalized), matmul_op=matmul,
                          activation_op=activation_op, binary_op=binary_op,
                          binary_operand_id=binary_operand_id)


def _legalize_static_dataflow(module: Module, *, profile: TargetProfile,
                                options: CompileOptions) -> LegalizedGraph:
    """Per-node capability and numeric-contract legalization for a small DAG.

    The accepted node set is deliberately primitive: matmul, a registered
    unary activation, and same-shape add/mul.  There is no whole-function shape
    match.  Graph legality (one or more exported values, every node reachable
    from an export, every non-exported produced value consumed) is proved first,
    so a node cannot disappear into a partial plan.
    """
    ops = list(module.ops)
    if not ops:
        reject(RK_E_UNSUPPORTED, "static dataflow module has no nodes",
               stage="legalize")
    output_ids = set(module.outputs)
    if not output_ids:
        reject(RK_E_UNSUPPORTED,
               "static dataflow requires at least one exported value",
               stage="legalize")
    # Role and producer are orthogonal fields in the frozen IR.  Check their
    # combination here before any binding/view is created: input/constant must be
    # external, output must be exported and produced, and temporary must be
    # produced by a node.
    for value in module.values:
        if value.role == "input":
            if value.producer is not None or value.id not in module.inputs:
                reject(RK_E_UNSUPPORTED,
                       "input role requires an external value listed in "
                       "module.inputs", stage="legalize",
                       actual={"value": value.id, "name": value.name,
                               "producer": value.producer})
        elif value.role == "constant":
            if value.producer is not None:
                reject(RK_E_UNSUPPORTED,
                       "constant role requires an external value",
                       stage="legalize", actual={"value": value.id,
                                                  "name": value.name})
        elif value.role == "output":
            if value.producer is None or value.id not in output_ids:
                reject(RK_E_UNSUPPORTED,
                       "output role requires a producer and an entry in "
                       "module.outputs", stage="legalize",
                       actual={"value": value.id, "name": value.name,
                               "producer": value.producer})
        elif value.role == "temporary":
            if value.producer is None:
                reject(RK_E_UNSUPPORTED,
                       "temporary role requires a device producer",
                       stage="legalize", actual={"value": value.id,
                                                  "name": value.name})
    for output_id in output_ids:
        output_value = module.value(output_id)
        if output_value.producer is None:
            reject(RK_E_UNSUPPORTED,
                   "module output has no producing node", stage="legalize",
                   actual={"value": output_id, "name": output_value.name})
        if output_value.role != "output":
            reject(RK_E_UNSUPPORTED,
                   "module output value must have output role",
                   stage="legalize", actual=output_value.role)

    # Reachability from all exported values.  Ops are already topologically
    # ordered by Module.__post_init__; walking backwards is sufficient.  An
    # exported intermediate may also feed a later op, so outputs are not assumed
    # to be the final node.
    needed_values = set(output_ids)
    reachable_ops: set[int] = set()
    for op in reversed(ops):
        if any(out in needed_values for out in op.outputs):
            reachable_ops.add(op.id)
            needed_values.update(op.inputs)
    if len(reachable_ops) != len(ops):
        dead = [op.id for op in ops if op.id not in reachable_ops]
        reject(RK_E_UNSUPPORTED,
               "static dataflow contains an operation that does not contribute "
               "to any exported value", stage="legalize", actual=dead,
               alternative="remove the dead node or export its result")
    produced = {v.producer for v in module.values if v.producer is not None}
    if produced != reachable_ops:
        reject(RK_E_UNSUPPORTED,
               "static dataflow has a value producer outside the reachable "
               "graph", stage="legalize")

    # Every produced value is either consumed or exported.  Operands without a
    # producer must be declared runtime inputs/constants; an unclassified
    # storage value must not reach the view planner.
    consumers: dict[int, int] = {v.id: 0 for v in module.values}
    for op in ops:
        for vid in op.inputs:
            consumers[vid] = consumers.get(vid, 0) + 1
    for value in module.values:
        if value.producer is None:
            if consumers.get(value.id, 0) and \
                    value.role not in {"input", "constant"}:
                reject(RK_E_UNSUPPORTED,
                       "a consumed external operand must be a module input or "
                       "constant", stage="legalize",
                       actual={"value": value.id, "name": value.name,
                               "role": value.role})
            if value.role == "input" and value.id not in module.inputs:
                reject(RK_E_UNSUPPORTED,
                       "an input-role value is missing from module.inputs",
                       stage="legalize",
                       actual={"value": value.id, "name": value.name})
            continue
        if value.id in output_ids:
            continue
        if value.role == "output":
            reject(RK_E_UNSUPPORTED,
                   "module declares an output-role value that is not exported",
                   stage="legalize", actual={"value": value.id,
                                              "name": value.name})
        if consumers.get(value.id, 0) == 0:
            reject(RK_E_UNSUPPORTED,
                   "static dataflow produces an unconsumed temporary",
                   stage="legalize", actual={"value": value.id,
                                              "name": value.name})

    legalized: list[LegalizedOp] = []
    matmuls: list[Op] = []
    activations: list[Op] = []
    binaries: list[Op] = []
    for op in ops:
        if op.opcode == "matmul":
            contract = NumericContract.from_dict(op.attrs["numeric_contract"])
            if options.numeric_mode == "strict":
                reject(RK_E_NUMERIC_CONTRACT,
                       "gemm requests FP32 strict accumulation; "
                       "co.mm_acc.f16.v1 writes FP16 at each K tile",
                       stage="legalize", op=op.opcode, source=op.source,
                       target_profile=profile.profile_hash,
                       actual=contract.to_dict(),
                       required="rk_f16_tiled opt-in",
                       alternative="pass numeric_mode=rk_f16_tiled to accept "
                                   "the declared ordered K-tile writeback path")
            if not contract.allows_f16_tiled_path:
                reject(RK_E_NUMERIC_CONTRACT,
                       "numeric contract does not declare ordered K-tile FP16 "
                       "writeback", stage="legalize", op=op.opcode,
                       source=op.source,
                       target_profile=profile.profile_hash,
                       actual=contract.to_dict(),
                       alternative="declare narrowing at k_tile_boundary")
            a = module.value(op.inputs[0])
            b = module.value(op.inputs[1])
            out = module.value(op.outputs[0])
            if not (1 <= out.type.rows <= 4096 and
                    1 <= out.type.cols <= 4096 and
                    1 <= a.type.cols <= 4096):
                reject(RK_E_SHAPE,
                       "tiled static dataflow requires M/N/K in 1..4096",
                       stage="legalize", op=op.opcode, source=op.source,
                       actual={"M": out.type.rows, "N": out.type.cols,
                               "K": a.type.cols})
            if b.producer is not None:
                reject(RK_E_UNSUPPORTED,
                       "matmul weights must be runtime constants or inputs, not "
                       "device-produced values", stage="legalize",
                       op=op.opcode, source=op.source)
            profile.enabled("co.mm.f16.v1")
            profile.enabled("co.mm_acc.f16.v1")
            legalized.append(LegalizedOp(op.id, "co.mm.f16.v1",
                                         "co.mm.f16.v1"))
            matmuls.append(op)
        elif op.opcode in ACTIVATION_CAPABILITY:
            if op.opcode != "relu" and not options.allow_approx_lut:
                reject(RK_E_NUMERIC_CONTRACT,
                       "activation uses a LUT approximation; explicit "
                       "allow_approx_lut is required", stage="legalize",
                       op=op.opcode, source=op.source,
                       target_profile=profile.profile_hash,
                       alternative="pass allow_approx_lut=True after reviewing "
                                   "the capability tolerance")
            capability = ACTIVATION_CAPABILITY[op.opcode]
            profile.enabled(capability)
            if op.opcode in {"silu", "gelu"}:
                profile.enabled("co.ew_mul.f16.v1")
            legalized.append(LegalizedOp(op.id, capability, capability))
            activations.append(op)
        elif op.opcode in BINARY_CAPABILITY:
            capability = BINARY_CAPABILITY[op.opcode]
            profile.enabled(capability)
            a = module.value(op.inputs[0])
            b = module.value(op.inputs[1])
            out = module.value(op.outputs[0])
            if not (a.type == b.type == out.type):
                constant_side = None
                tensor_side = None
                for value, other in ((a, b), (b, a)):
                    if value.role == "constant" and value.type.rank < 2:
                        if constant_side is not None:
                            reject(RK_E_UNSUPPORTED,
                                   "two broadcast constants require an explicit "
                                   "materialised tensor", stage="legalize",
                                   op=op.opcode, source=op.source)
                        constant_side, tensor_side = value, other
                if constant_side is None or tensor_side is None or \
                        tensor_side.type.rank != 2 or \
                        tensor_side.type != out.type:
                    reject(RK_E_UNSUPPORTED,
                           "static dataflow add/mul accepts same-shape operands "
                           "or one explicit rank-0/rank-1 constant broadcast",
                           stage="legalize", op=op.opcode, source=op.source,
                           actual={"a": list(a.type.shape),
                                   "b": list(b.type.shape),
                                   "out": list(out.type.shape)})
                if constant_side.type.rank == 1 and \
                        constant_side.type.shape[0] != out.type.cols:
                    reject(RK_E_SHAPE,
                           "per-channel constant length must equal N",
                           stage="legalize", op=op.opcode, source=op.source,
                           actual=list(constant_side.type.shape),
                           required=out.type.cols)
            legalized.append(LegalizedOp(op.id, capability, capability))
            binaries.append(op)
        else:
            reject(RK_E_UNSUPPORTED,
                   f"unsupported static dataflow node {op.opcode!r}",
                   stage="legalize", op=op.opcode, source=op.source,
                   actual={"op_id": op.id})

    return LegalizedGraph(
        module=module, ops=tuple(legalized),
        matmul_op=matmuls[0] if matmuls else None,
        activation_op=activations[0] if activations else None,
        binary_op=binaries[0] if binaries else None,
        graph_ops=tuple(legalized))


def _match_swiglu(module: Module, options: CompileOptions) -> SwiGLUFFN | None:
    """Match the closed SwiGLU FFN dataflow, or return None.

    The pattern is verified by Value ID connections, not by opcode order:

        Zg = X @ Wg^T   Zu = X @ Wu^T
        S  = SiLU(Zg)   A = S * Zu
        H  = A @ Wd^T   Y = H + X
    """
    ops = module.ops
    if len(ops) != 6:
        return None
    by_id = {op.id: op for op in ops}
    gate, up, act, mul, proj, add = ops
    if [op.opcode for op in ops] != ["matmul", "matmul", "silu", "mul", "matmul",
                                     "add"]:
        return None
    x = module.value(module.inputs[0]) if module.inputs else None
    if x is None or x.role != "input":
        return None
    if tuple(gate.inputs[0:1]) + tuple(up.inputs[0:1]) != (x.id, x.id):
        return None
    if tuple(gate.outputs) != tuple(act.inputs):
        return None
    if tuple(act.outputs) + tuple(up.outputs) != tuple(mul.inputs) and \
            tuple(up.outputs) + tuple(act.outputs) != tuple(mul.inputs):
        return None
    if tuple(mul.outputs) != tuple(proj.inputs[0:1]):
        return None
    if tuple(proj.outputs) + (x.id,) != tuple(add.inputs) and \
            (x.id,) + tuple(proj.outputs) != tuple(add.inputs):
        return None
    if tuple(module.outputs) != tuple(add.outputs):
        return None
    gate_shape = module.value(gate.outputs[0]).type.shape
    proj_shape = module.value(proj.outputs[0]).type.shape
    hidden = gate_shape[1]
    # This function is a *matcher* for the optional whole-graph
    # specialisation.  If any specialisation constraint does not hold, return
    # None so the generic per-node legalizer can handle the same semantic graph
    # (and produce the authoritative diagnostic if that path also rejects).
    for matmul in (gate, up, proj):
        contract = module.value(matmul.inputs[0]).type.shape[1]
        if contract > 256 or matmul.attrs.get("b_layout", "kn") != "nk":
            return None
        contract_attr = NumericContract.from_dict(matmul.attrs["numeric_contract"])
        if options.numeric_mode == "strict" or \
                not contract_attr.allows_f16_tiled_path:
            return None
    if max(gate_shape + proj_shape) > 256:
        return None
    if not options.allow_approx_lut:
        return None
    return SwiGLUFFN(matmul_gate=gate, matmul_up=up, activation=act,
                     multiply=mul, matmul_proj=proj, add=add, input_id=x.id,
                     hidden=hidden, output=proj_shape[1])
