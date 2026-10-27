# SPDX-License-Identifier: GPL-3.0-or-later
"""Native TVM carrier for the recognized compute graph (refactor R4 pilot).

This module answers one question with evidence: *can the separate RKIR graph
container be replaced by upstream TVM without losing target semantics?*

What it does
------------
The structural recognizer in :mod:`rknpu_backend.frontend.tilelang_adapter` runs
with the neutral carrier of :mod:`rknpu_backend.frontend.graph_builder`, so it
returns container-free records instead of RKIR objects.  Those records are turned
directly into ``tvm.relax`` nodes here: shapes, dtypes, operators and data-flow
edges live in the native graph (``TensorStructInfo`` plus ``relax.Var`` /
``relax.Call`` / ``VarBinding``), never in an attribute.

What it deliberately keeps
--------------------------
Only the *target contract* -- the part ordinary Relax cannot express -- is stored,
as a single ``DictAttrs`` entry on the ``relax.Function``:

* per-value role (input / constant / temporary / output) and binding name,
* per-op numeric contract, accumulator dtype, FP16 narrowing points, K-loop,
  MN-tiling and tail hints, plus the source span used by diagnostics.

The table is re-validated against the native graph every time a carrier is built
or a pass is applied (:func:`validate_contract`).  Parameter order, binding count,
operator kind, result shape, dtype and the emitted variable names must all agree
with the contract; a pass that renames, reorders or rewrites nodes therefore fails
closed with an explicit rejection instead of silently planning a different kernel.

Why names appear at all: built-in Relax ops carry their own typed ``Attrs``, so a
per-node custom contract is impossible without either a private dialect (forbidden
by the refactor spec) or an op-composition policy that breaks the one-record-per-
node association.  The pilot therefore proves the association on every use.  That
re-validation cost is part of the R4 adoption evidence, not a hidden assumption.

Coverage is intentionally partial: an opcode with no direct Relax operator in the
pinned TVM version (``hardsigmoid``) is rejected rather than composed.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from ..errors import RK_E_SCHEMA, RK_E_UNSUPPORTED, reject
from ..ir import TensorType

CONTRACT_KIND = "rk-native-contract.v1"
CONTRACT_ATTR = "rknpu_contract"
# The only upstream passes the native carrier declares.  Anything that changes
# association or rounding semantics must rebuild the contract.
PASS_SET = ("Normalize", "CanonicalizeBindings", "DeadCodeElimination")
OPCODE_PREFIX = "v"


def _import_relax():
    try:
        from tvm import relax
        return relax
    except Exception:
        pass
    # The pinned TileLang wheel ships its own TVM tree and puts it on ``sys.path``
    # on import, exactly as the structural extractor already relies on.
    try:
        import tilelang  # noqa: F401
        from tvm import relax
        return relax
    except Exception as exc:  # pragma: no cover - optional dependency
        reject(RK_E_UNSUPPORTED,
               f"TVM Relax is unavailable or failed to import: {exc}",
               stage="native",
               alternative="install the pinned optional TileLang dependency")


# opcode -> (dotted Relax builder path, arity).  One table means the builder, the
# contract's ``native_op`` field and the coverage report cannot disagree.
NATIVE_OPS: dict[str, tuple[str, int]] = {
    "matmul": ("relax.op.matmul", 2),
    "add": ("relax.op.add", 2),
    "mul": ("relax.op.multiply", 2),
    "relu": ("relax.op.nn.relu", 1),
    "silu": ("relax.op.nn.silu", 1),
    "gelu": ("relax.op.nn.gelu", 1),
    "sigmoid": ("relax.op.sigmoid", 1),
}


def _resolve_relax_path(relax: Any, path: str) -> Any:
    node = relax
    for part in path.split(".")[1:]:
        node = getattr(node, part)
    return node


def native_op_table(relax: Any) -> dict[str, tuple[Any, int]]:
    """opcode -> (Relax operator builder, arity)."""
    return {opcode: (_resolve_relax_path(relax, path), arity)
            for opcode, (path, arity) in NATIVE_OPS.items()}


def native_op_name(opcode: str) -> str:
    """The Relax operator name the contract must record for ``opcode``."""
    try:
        return NATIVE_OPS[opcode][0].replace("op.", "", 1).replace(".nn.", ".nn.")
    except KeyError:
        reject(RK_E_UNSUPPORTED,
               f"native carrier has no Relax operator for opcode {opcode!r}",
               stage="native", op=opcode)


def supported_opcodes() -> tuple[str, ...]:
    return tuple(sorted(NATIVE_OPS))


# ---------------------------------------------------------------- view types

@dataclass(frozen=True)
class NativeValue:
    """Read-only projection of one ``relax.Var`` for the planner."""

    id: int
    name: str
    type: TensorType
    producer: int | None
    role: str


@dataclass(frozen=True)
class NativeOp:
    """Read-only projection of one ``relax.Call`` for the planner."""

    id: int
    opcode: str
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    attrs: dict[str, Any]
    source: str
    native_op: str = ""
    side_effects: bool = False


class NativeGraph:
    """The planner-facing carrier backed by a real ``tvm.relax`` module.

    ``values``/``ops`` are frozen projections derived from the native module and
    validated against the target contract; the native module stays the single
    source of truth and no second mutable graph is kept.
    """

    kind = "native"
    contract_version = CONTRACT_KIND

    def __init__(self, *, ir_module: Any, contract: dict[str, Any],
                 values: tuple[NativeValue, ...], ops: tuple[NativeOp, ...]):
        self._ir_module = ir_module
        self._contract = contract
        self._values = values
        self._ops = ops
        self._by_id = {value.id: value for value in values}
        self.name = contract["name"]
        self.source_hash = contract["source_hash"]
        self.profile_hash = contract["profile_hash"]
        self.inputs = tuple(contract["inputs"])
        self.outputs = tuple(contract["outputs"])

    # -- planner protocol -------------------------------------------------
    @property
    def values(self) -> tuple[NativeValue, ...]:
        return self._values

    @property
    def ops(self) -> tuple[NativeOp, ...]:
        return self._ops

    def value(self, value_id: int) -> NativeValue:
        found = self._by_id.get(value_id)
        if found is None:
            reject(RK_E_SCHEMA, f"native graph has no value {value_id}",
                   stage="native")
        return found

    # -- diagnostics ------------------------------------------------------
    @property
    def ir_module(self) -> Any:
        return self._ir_module

    @property
    def contract(self) -> dict[str, Any]:
        return copy.deepcopy(self._contract)

    def tvmscript(self) -> str:
        return self._ir_module.script()

    def release_native_graph(self) -> None:
        """Drop the Relax module; a plan never depends on it."""
        self._ir_module = None

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (f"NativeGraph(name={self.name!r}, values={len(self._values)}, "
                f"ops={len(self._ops)}, inputs={self.inputs}, "
                f"outputs={self.outputs})")


# ------------------------------------------------------------------ building

def _tensor_shape(value: Any, where: str) -> tuple[tuple[int, ...], str]:
    tensor = value.type
    shape = tuple(int(v) for v in getattr(tensor, "shape", ()))
    dtype = str(getattr(tensor, "dtype", "float16"))
    if dtype != "float16":
        reject(RK_E_UNSUPPORTED, f"{where} must be float16", stage="native",
               actual=dtype)
    if any(dim <= 0 for dim in shape):
        reject(RK_E_SCHEMA, f"{where} has a non-static extent", stage="native",
               actual=list(shape))
    return shape, dtype


def build_spec(spec: Any) -> NativeGraph:
    """Turn one container-free recognition result into a native Relax graph."""
    relax = _import_relax()
    import tvm

    builders = native_op_table(relax)
    values = list(spec.values)
    ops = list(spec.ops)
    if [value.id for value in values] != list(range(len(values))):
        reject(RK_E_SCHEMA, "native build requires contiguous value IDs",
               stage="native")
    if [op.id for op in ops] != list(range(len(ops))):
        reject(RK_E_SCHEMA, "native build requires contiguous op IDs",
               stage="native")

    expressions: dict[int, Any] = {}
    params = []
    for value in values:
        shape, dtype = _tensor_shape(value, f"native value {value.id}")
        if value.producer is None:
            if value.role not in {"input", "constant"}:
                reject(RK_E_UNSUPPORTED,
                       f"native value {value.id} has no producer but role "
                       f"{value.role!r}", stage="native")
            var = relax.Var(value.name, relax.TensorStructInfo(shape, dtype))
            expressions[value.id] = var
            params.append(var)

    contract_ops: list[dict[str, Any]] = []
    builder = relax.BlockBuilder()
    with builder.function(spec.name, params=params):
        with builder.dataflow():
            for op in ops:
                entry = builders.get(op.opcode)
                if entry is None:
                    reject(RK_E_UNSUPPORTED,
                           f"native carrier has no Relax operator for opcode "
                           f"{op.opcode!r}", stage="native", op=op.opcode,
                           source=op.source,
                           alternative="keep RKIR for this kernel or extend the "
                                       "native operator table")
                native_fn, arity = entry
                if len(op.inputs) != arity:
                    reject(RK_E_SCHEMA,
                           f"{op.opcode} needs arity {arity} in the native carrier",
                           stage="native", op=op.opcode, actual=len(op.inputs))
                args = []
                for value_id in op.inputs:
                    expr = expressions.get(value_id)
                    if expr is None:
                        reject(RK_E_SCHEMA,
                               f"native op {op.id} reads value {value_id} before "
                               "its definition", stage="native", op=op.opcode)
                    args.append(expr)
                call = native_fn(*args)
                (out_id,) = op.outputs
                expressions[out_id] = builder.emit(
                    call, f"{OPCODE_PREFIX}{out_id}")
                contract_ops.append({
                    "id": op.id,
                    "value_id": out_id,
                    "opcode": op.opcode,
                    "native_op": call.op.name,
                    "source": op.source,
                    "contract": copy.deepcopy(dict(op.attrs)),
                })
            outputs = [expressions[value_id] for value_id in spec.outputs]
            if len(outputs) == 1:
                result = builder.emit_output(outputs[0])
            else:
                result = builder.emit_output(relax.Tuple(outputs))
        builder.emit_func_output(result)
    ir_module = builder.get()

    contract = {
        "kind": CONTRACT_KIND,
        "name": spec.name,
        "profile_hash": spec.profile_hash,
        "source_hash": spec.source_hash,
        "state_signature": getattr(spec, "state_signature", ""),
        "values": [
            {"id": value.id, "name": value.name, "role": value.role,
             "producer": value.producer,
             "shape": list(_tensor_shape(value, "value")[0]),
             "dtype": str(value.type.dtype),
             # Broadcast sources may be rank 0/1; the axis labels are part of the
             # recognized type, so they travel with the contract instead of being
             # re-guessed from the shape.
             "axes": list(getattr(value.type, "axes", ()))}
            for value in values
        ],
        "ops": contract_ops,
        "inputs": list(spec.inputs),
        "outputs": list(spec.outputs),
        "pass_set": list(PASS_SET),
    }
    ir_module = _attach_contract(ir_module, contract)
    # Bind the contract to the native graph through a renumbering-invariant
    # signature, then require that signature on every later use.
    contract["signature"] = graph_signature(_function(ir_module, contract["name"]),
                                            contract)
    ir_module = _attach_contract(ir_module, contract)
    return _build_view(ir_module, contract)


def _function(ir_module: Any, name: str | None = None) -> Any:
    """Resolve the single Relax function, preferring the recognized name."""
    functions = dict(ir_module.functions)
    if not functions:
        reject(RK_E_SCHEMA, "native module has no function", stage="native")
    for gvar, func in functions.items():
        if name and gvar.name_hint == name:
            return func
    if len(functions) != 1:
        reject(RK_E_SCHEMA, "native module has more than one function",
               stage="native", actual=sorted(g.name_hint for g in functions))
    return next(iter(functions.values()))


def _attach_contract(ir_module: Any, contract: dict[str, Any]) -> Any:
    import tvm
    from tvm import relax

    func = _function(ir_module, contract.get("name"))
    attrs = dict(func.attrs) if func.attrs else {}
    attrs[CONTRACT_ATTR] = tvm.runtime.convert(contract)
    new_func = relax.Function(func.params, func.body,
                              ret_struct_info=func.ret_struct_info,
                              is_pure=func.is_pure,
                              attrs=tvm.ir.DictAttrs(attrs))
    return tvm.IRModule({contract["name"]: new_func})


def _op_bindings(func: Any) -> list[Any]:
    """``VarBinding`` nodes that produce a call, in data-flow order."""
    relax = _import_relax()
    body = func.body
    if not isinstance(body, relax.SeqExpr):
        reject(RK_E_SCHEMA, "native function body is not a SeqExpr",
               stage="native", actual=type(body).__name__)
    bindings = []
    for block in body.blocks:
        for binding in getattr(block, "bindings", ()):
            if isinstance(binding, relax.VarBinding) and \
                    isinstance(binding.value, relax.Call):
                bindings.append(binding)
    return bindings


def _shape_of(struct_info: Any) -> list[int]:
    shape = getattr(struct_info, "shape", None)
    if shape is None:
        reject(RK_E_SCHEMA, "native node has no tensor shape", stage="native",
               actual=type(struct_info).__name__)
    return [int(v) for v in shape]


def _operand_refs(func: Any, contract: dict[str, Any]) -> list[list[str]]:
    """Express each op's operands as ``p<param>`` / ``v<producing op>``.

    This is a *validation* signature, not a second copy of the graph: it is
    derived from the native nodes on demand, is invariant under node renumbering
    and renaming, and changes as soon as a transformation alters data flow.
    """
    relax = _import_relax()
    params = {param: index for index, param in enumerate(func.params)}
    produced: dict[Any, int] = {}
    refs: list[list[str]] = []
    position = 0
    for block in func.body.blocks:
        for binding in getattr(block, "bindings", ()):
            if not isinstance(binding, relax.VarBinding):
                continue
            if not isinstance(binding.value, relax.Call):
                continue
            row = []
            for arg in binding.value.args:
                if arg in params:
                    row.append(f"p{params[arg]}")
                    continue
                producer = produced.get(arg)
                if producer is None:
                    reject(RK_E_SCHEMA,
                           "native operand is neither a parameter nor a recognized "
                           "operation result", stage="native",
                           action_id=position)
                row.append(f"v{producer}")
            refs.append(row)
            produced[binding.var] = position
            position += 1
    return refs


def _output_refs(func: Any, contract: dict[str, Any]) -> list[str]:
    relax = _import_relax()
    params = {param: index for index, param in enumerate(func.params)}
    produced: dict[Any, int] = {}
    position = 0
    for block in func.body.blocks:
        for binding in getattr(block, "bindings", ()):
            if not isinstance(binding, relax.VarBinding):
                continue
            if isinstance(binding.value, relax.Call):
                produced[binding.var] = position
                position += 1
            elif binding.var not in produced:
                # Alias bindings created by ``emit_output`` (``gv = v3``).
                produced[binding.var] = None  # type: ignore[assignment]
    resolved: dict[Any, str] = {}
    for expr, index in produced.items():
        if index is not None:
            resolved[expr] = f"v{index}"
    for param, index in params.items():
        resolved[param] = f"p{index}"
    # Second sweep for aliases whose source is known.
    for block in func.body.blocks:
        for binding in getattr(block, "bindings", ()):
            if isinstance(binding, relax.VarBinding) and \
                    not isinstance(binding.value, relax.Call):
                source = resolved.get(binding.value)
                if source is not None:
                    resolved[binding.var] = source
                elif isinstance(binding.value, relax.Tuple):
                    pass

    out: list[str] = []

    def flatten(expr: Any) -> None:
        if isinstance(expr, relax.Tuple):
            for field in expr.fields:
                flatten(field)
            return
        if expr in tuple_map:
            out.extend(tuple_map[expr])
            return
        ref = resolved.get(expr)
        if ref is None:
            reject(RK_E_SCHEMA, "native graph output is not a recognized value",
                   stage="native", actual=type(expr).__name__)
        out.append(ref)

    tuple_map: dict[Any, list[str]] = {}
    for block in func.body.blocks:
        for binding in getattr(block, "bindings", ()):
            if isinstance(binding, relax.VarBinding) and \
                    isinstance(binding.value, relax.Tuple):
                fields: list[str] = []
                ok = True
                for field in binding.value.fields:
                    ref = resolved.get(field)
                    if ref is None:
                        ok = False
                        break
                    fields.append(ref)
                if ok:
                    tuple_map[binding.var] = fields
    flatten(func.body.body)
    return out


def graph_signature(func: Any, contract: dict[str, Any]) -> dict[str, Any]:
    """The canonical, renumbering-invariant description the contract is bound to."""
    relax = _import_relax()
    params = list(func.params)
    bindings = _op_bindings(func)
    if len(bindings) != len(contract["ops"]):
        reject(RK_E_SCHEMA,
               "native graph no longer matches its target contract: op count "
               "changed", stage="native", actual=len(bindings),
               required=len(contract["ops"]))
    refs = _operand_refs(func, contract)
    ops = []
    for index, (binding, operands) in enumerate(zip(bindings, refs)):
        struct_info = binding.var.struct_info
        ops.append({
            "op": binding.value.op.name,
            "operands": operands,
            "shape": _shape_of(struct_info),
            "dtype": str(struct_info.dtype),
        })
    return {
        "params": [{"shape": _shape_of(param.struct_info),
                    "dtype": str(param.struct_info.dtype)} for param in params],
        "ops": ops,
        "outputs": _output_refs(func, contract),
    }


def _native_op_for(opcode: str) -> str:
    return native_op_name(opcode)


def cross_validate_contract(contract: dict[str, Any]) -> None:
    """Check the contract is internally consistent before it is believed.

    The signature proves the *graph* still matches; this proves the *contract*
    fields agree with each other, so a corrupted ``opcode``/``value_id``/role can
    never be turned into a silently different plan.
    """
    if contract.get("kind") != CONTRACT_KIND:
        reject(RK_E_SCHEMA, "native contract has an unknown kind", stage="native",
               actual=contract.get("kind"))
    signature = contract.get("signature")
    if signature is None:
        reject(RK_E_SCHEMA, "native contract has no graph signature",
               stage="native")
    values = {value["id"]: value for value in contract["values"]}
    ops = contract["ops"]
    if [op["id"] for op in ops] != list(range(len(ops))):
        reject(RK_E_SCHEMA, "native contract op IDs are not contiguous",
               stage="native")
    if len(signature["ops"]) != len(ops):
        reject(RK_E_SCHEMA, "native contract signature length mismatch",
               stage="native")
    for position, (op, recorded) in enumerate(zip(ops, signature["ops"])):
        expected = _native_op_for(op["opcode"])
        if op["native_op"] != expected:
            reject(RK_E_SCHEMA,
                   "native contract operator does not match its opcode",
                   stage="native", action_id=position, actual=op["native_op"],
                   required=expected)
        if recorded["op"] != expected:
            reject(RK_E_SCHEMA,
                   "native contract signature no longer describes the recorded "
                   "opcode", stage="native", action_id=position,
                   actual=recorded["op"], required=expected)
        produced = values.get(op["value_id"])
        if produced is None or produced["producer"] != op["id"]:
            reject(RK_E_SCHEMA,
                   "native contract value no longer belongs to its op",
                   stage="native", action_id=position, actual=op["value_id"])
        if list(recorded["shape"]) != produced["shape"] or \
                recorded["dtype"] != produced["dtype"]:
            reject(RK_E_SCHEMA,
                   "native contract signature shape/dtype disagrees with the "
                   "value table", stage="native", action_id=position,
                   actual={"shape": recorded["shape"], "dtype": recorded["dtype"]},
                   required={"shape": produced["shape"],
                             "dtype": produced["dtype"]})
    param_values = [value for value in contract["values"]
                    if value["producer"] is None]
    if len(param_values) != len(signature["params"]):
        reject(RK_E_SCHEMA, "native contract parameter table disagrees with the "
               "signature", stage="native")
    for value, recorded in zip(param_values, signature["params"]):
        if value["shape"] != recorded["shape"] or value["dtype"] != recorded["dtype"]:
            reject(RK_E_SCHEMA,
                   "native contract parameter shape disagrees with the signature",
                   stage="native", actual=recorded,
                   required={"shape": value["shape"], "dtype": value["dtype"]})
    expected_outputs = []
    for ref in signature["outputs"]:
        kind, index = ref[0], int(ref[1:])
        if kind == "v":
            expected_outputs.append(ops[index]["value_id"])
        else:
            expected_outputs.append(param_values[index]["id"])
    if expected_outputs != list(contract["outputs"]):
        reject(RK_E_SCHEMA, "native contract outputs disagree with the signature",
               stage="native", actual=expected_outputs,
               required=list(contract["outputs"]))
    for position, value in enumerate(contract["values"]):
        if value["id"] != position:
            reject(RK_E_SCHEMA, "native contract values must be ordered by id",
                   stage="native")
        if value["role"] not in {"input", "constant", "temporary", "output"}:
            reject(RK_E_SCHEMA, "native contract has an unknown value role",
                   stage="native", actual=value["role"])
        if value["producer"] is None and value["role"] not in {"input", "constant"}:
            reject(RK_E_SCHEMA,
                   "native contract binds a producer-less value to a computed "
                   "role", stage="native", action_id=value["id"],
                   actual=value["role"])
        if value["producer"] is not None and \
                value["role"] not in {"temporary", "output"}:
            reject(RK_E_SCHEMA,
                   "native contract binds a computed value to a source role",
                   stage="native", action_id=value["id"], actual=value["role"])
    by_role = {
        "input": sorted(value["id"] for value in contract["values"]
                        if value["role"] == "input"),
        "constant": sorted(value["id"] for value in contract["values"]
                           if value["role"] == "constant"),
        "output": sorted(value["id"] for value in contract["values"]
                         if value["role"] == "output"),
    }
    if by_role["input"] != sorted(contract["inputs"]):
        reject(RK_E_SCHEMA,
               "native contract inputs disagree with the value roles",
               stage="native", actual=by_role["input"],
               required=sorted(contract["inputs"]))
    if by_role["output"] != sorted(contract["outputs"]):
        reject(RK_E_SCHEMA,
               "native contract outputs disagree with the value roles",
               stage="native", actual=by_role["output"],
               required=sorted(contract["outputs"]))
    if set(by_role["constant"]) & set(contract["inputs"]):
        reject(RK_E_SCHEMA, "native contract lists a constant as an input",
               stage="native")


def validate_contract(ir_module: Any, contract: dict[str, Any]) -> None:
    """Prove the target contract still describes the native graph."""
    func = _function(ir_module, contract.get("name"))
    cross_validate_contract(contract)
    recorded = contract["signature"]
    actual = graph_signature(func, contract)
    if actual != recorded:
        reject(RK_E_SCHEMA,
               "native graph no longer matches its target contract", stage="native",
               actual=actual, required=recorded)


def _build_view(ir_module: Any, contract: dict[str, Any]) -> NativeGraph:
    validate_contract(ir_module, contract)
    func = _function(ir_module, contract.get("name"))
    param_values = [value for value in contract["values"]
                    if value["producer"] is None]
    signature = contract["signature"]
    # Positional association: operand reference ``p<k>`` is the k-th parameter
    # block and ``v<j>`` is the result of the j-th op; both were validated above.
    ops: list[NativeOp] = []
    for position, op in enumerate(contract["ops"]):
        entry = signature["ops"][position]
        inputs = []
        for ref in entry["operands"]:
            kind, index = ref[0], int(ref[1:])
            if kind == "p":
                inputs.append(param_values[index]["id"])
            else:
                inputs.append(contract["ops"][index]["value_id"])
        ops.append(NativeOp(id=op["id"], opcode=op["opcode"],
                            inputs=tuple(inputs),
                            outputs=(op["value_id"],),
                            attrs=copy.deepcopy(op["contract"]),
                            source=op["source"],
                            native_op=op["native_op"]))
    exprs: dict[Any, int] = {}
    for index, param in enumerate(func.params):
        exprs[param] = param_values[index]["id"]
    for position, binding in enumerate(_op_bindings(func)):
        exprs[binding.var] = contract["ops"][position]["value_id"]
    values = []
    for value in contract["values"]:
        expr = next((node for node, ident in exprs.items() if ident == value["id"]),
                    None)
        shape = tuple(value["shape"])
        dtype = value["dtype"]
        if expr is not None:
            shape = tuple(_shape_of(expr.struct_info))
            dtype = str(expr.struct_info.dtype)
        values.append(NativeValue(
            id=value["id"], name=value["name"],
            type=TensorType(dtype, shape, axes=tuple(value.get("axes") or ())),
            producer=value["producer"], role=value["role"]))
    return NativeGraph(ir_module=ir_module, contract=contract,
                       values=tuple(values), ops=tuple(ops))


def build(prim_func: Any, *, options: Any, profile_hash: str = "",
          b_is_input: bool = False) -> NativeGraph:
    """Recognize a real TileLang PrimFunc straight into a native TVM graph."""
    from . import tilelang_adapter
    from .graph_builder import RecordBuilder, use_builder

    with use_builder(RecordBuilder()):
        spec = tilelang_adapter.extract(prim_func, options=options,
                                        profile_hash=profile_hash,
                                        b_is_input=b_is_input)
    return build_spec(spec)


def apply_minimal_passes(graph: NativeGraph) -> NativeGraph:
    """Run only the declared upstream passes, then re-validate the contract."""
    import tvm

    contract = graph.contract
    module = graph.ir_module
    if module is None:
        reject(RK_E_SCHEMA, "native graph module was released", stage="native")
    applied = []
    for name in PASS_SET:
        pass_ = getattr(tvm.relax.transform, name, None)
        if pass_ is None:
            reject(RK_E_UNSUPPORTED,
                   f"required upstream pass {name} is unavailable", stage="native")
        module = pass_()(module)
        validate_contract(module, contract)
        applied.append(name)
    view = _build_view(module, contract)
    return view
