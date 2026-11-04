# SPDX-License-Identifier: GPL-3.0-or-later
"""RKNPU pass chain: localise tile ops on TIR, instantiate tasks, build RKPlan.

The program is normalised with upstream passes, its local tile operations are
proved and rewritten into RKNPU operations *inside the IR*, the loops are then
instantiated into an explicit task sequence, and the plan is assembled from the
ordered operations, operand regions and accumulation chain read back from that
sequence.  The planner still owns pack layouts, accumulator ping-pong, templates,
submit grouping and downloads; none of those are derived from the program.

Status: the guarded-tail opt-in entry.  Nothing in this module registers with
tilelang.backend; ``compile_guarded_tail`` drives the upstream transforms itself
and binds target kind "c" because this build has no rknpu target kind.  The
registered backend reaches this entry through
:mod:`rknpu_backend.frontend.tilelang_lowering`; closed full-logical GEMMs use
:mod:`rknpu_backend.frontend.logical_schedule` and the same task consumer.

Chain (each stage is a real IR -> IR transform; the IR text is what the next
stage consumes):

    TIR
      -> BindTarget + MaterializeKernelLaunch(lower_thread_binding=False)
      -> pass_localise      every tile op becomes an RKNPU call whose arguments
                            are the proved operand, tile and K facts
      -> pass_instantiate   grid and K loops are expanded; guards are decided on
                            the iteration domain; the body becomes one SeqStmt
                            of RKNPU calls in program order
      -> tasks_from_module  the sequence is read back as task records
      -> build_from_tasks   the planner consumes the ordered target operations,
                            their source/destination regions, tile/K geometry,
                            accumulator chain and store rectangles.  Upload
                            pack layouts, accumulator ping-pong, templates,
                            submit grouping and downloads remain planner
                            decisions; they are not regenerated from a shape
                            summary.

Proofs are domain based: an access is legal when it is inside the operand for
every iteration satisfying the enclosing path conditions, and its index is
exactly tile_extent * tile_coordinate + local_index.  A guard that normalisation
removed is therefore not needed; a guard that is still required and cannot be
discharged is a rejection.

Two rules make those local facts a plan that restates the program:

* One computation.  The block part of every access is proved per axis
  (_prove_tile_index) and then the operand reads and the output write must agree
  on it over one domain, with the K axis on the K loop variable.  Three source
  expressions agreeing is an obligation; nothing downstream re-derives a block
  coordinate from a default grid.
* Per-axis output proof.  Each output axis is either bounded by a consumed guard
  naming the logical extent, or covered exactly by the grid, and the store must
  be inside the output for every executed iteration.  A coordinate that folded
  to zero is accepted only when the axis has one block, and the two axes may not
  share one grid loop: that would write the same block twice.

The ordered target operations themselves are the planner input.  The planner
validates their fill/copy/gemm/store relationships, tile/K partition,
accumulation order and publication source; it does not reduce them to a tile
set first and regenerate the execution graph from shapes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..errors import RK_E_BUDGET, RK_E_NUMERIC_CONTRACT, RK_E_SHAPE, \
    RK_E_UNSUPPORTED, reject

# The one definition of the tile operations this path consumes, shared by the
# proof and the rewrite.  An operation outside this table has an effect the chain
# cannot restate, so it is refused; nothing defaults an unknown operation into a
# copy or into any other category by name shape.
TILE_OPS = {"tl.tileop.gemm": "gemm", "tl.tileop.fill": "fill",
            "tl.tileop.copy": "copy"}

RK_GEMM = "rk_gemm"
RK_COPY = "rk_copy"
RK_FILL = "rk_fill"
RK_STORE = "rk_store"
RK_SILU = "rk_silu"
_PIPELINE_NOT_APPLIED_REASON = (
    "no software pipeline is implemented on this target: the K chain is "
    "emitted as an ordered accumulation of 256-wide tasks, so the request "
    "is recorded but not applied")


def _tir():
    import tilelang  # noqa: F401  (loads the vendored TVM)
    from tvm import tirx

    return tirx


def _arith():
    import tvm.arith

    return tvm.arith.Analyzer()


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _walk(node: Any, depth: int = 0) -> Iterable[tuple[int, Any]]:
    yield depth, node
    for name in ("block", "iter_values", "predicate", "body", "value",
                 "then_case", "else_case", "condition", "a", "b"):
        child = getattr(node, name, None)
        if child is None or not hasattr(child, "__class__"):
            continue
        if type(child).__name__ in ("int", "str", "bool", "float"):
            continue
        yield from _walk(child, depth + 1)
    for item in getattr(node, "seq", ()) or ():
        yield from _walk(item, depth + 1)
    for arg in getattr(node, "args", ()) or ():
        if hasattr(arg, "__class__") and type(arg).__name__ not in (
                "int", "str", "bool", "float"):
            yield from _walk(arg, depth + 1)


def _op_name(call: Any) -> str:
    return str(getattr(getattr(call, "op", None), "name", "") or "")


def _scope_of(buffer: Any) -> str:
    scope = getattr(buffer, "scope", "")
    return str(scope() if callable(scope) else scope)


def _const(value: Any):
    tir = _tir()
    if isinstance(value, tir.IntImm):
        return int(value)
    if isinstance(value, int):
        return value
    return None


def _known_const(value: Any):
    """Read literals and call-free expressions that simplify to a literal."""
    direct = _const(value)
    if direct is not None:
        return direct
    try:
        return _const(_arith().simplify(value))
    except (AttributeError, TypeError, ValueError):
        return None


def _conjuncts(condition: Any) -> list[Any]:
    tir = _tir()
    out: list[Any] = []
    stack = [condition]
    while stack:
        node = stack.pop()
        if isinstance(node, tir.And):
            stack.extend([node.b, node.a])
        else:
            out.append(node)
    return out


@dataclass
class Stage:
    """Audit record for one stage: kind says whether it rewrote the IR."""

    name: str
    kind: str
    ir_digest: str
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind,
                "ir_digest": self.ir_digest, "notes": self.notes}


# --------------------------------------------------------------- upstream
def normalise(prim_func: Any, *, target: Any = None,
              simplify: bool = False) -> tuple[Any, list[Stage]]:
    """Run the upstream normalisation the RKNPU backend relies on."""
    import tvm
    from tvm import tirx
    from tilelang import transform as TT

    target = target or tvm.target.Target("c")
    module = prim_func if isinstance(prim_func, tvm.IRModule) else \
        tvm.IRModule({"main": prim_func})
    stages: list[Stage] = []
    module = tirx.transform.BindTarget(target)(module)
    stages.append(Stage("bind-target", "transform",
                        _digest(module.script()), {"target": str(target)}))
    module = TT.MaterializeKernelLaunch(lower_thread_binding=False)(module)
    stages.append(Stage("materialize-kernel-launch", "transform",
                        _digest(module.script()),
                        {"lower_thread_binding": False}))
    if simplify:
        module = TT.Simplify()(module)
        stages.append(Stage("simplify", "transform", _digest(module.script()), {}))
    return module["main"], stages


def _statements(node: Any) -> list[Any]:
    seq = getattr(node, "seq", None)
    if seq is not None:
        return list(seq)
    return [node]


def _store_inside(node: Any):
    tir = _tir()
    cursor = node
    while type(cursor).__name__ in ("For", "IfThenElse"):
        cursor = cursor.then_case if type(cursor).__name__ == "IfThenElse" \
            else cursor.body
    return cursor if type(cursor).__name__ == "BufferStore" else None


def _structurally_equal(left: Any, right: Any) -> bool:
    import tvm.ir

    try:
        return bool(tvm.ir.structural_equal(left, right))
    except (AttributeError, ValueError, TypeError):
        return False


def _region_exprs(call: Any, *, keep_symbolic_extents: bool = False
                  ) -> list[dict[str, Any]]:
    """Regions of a tile-op call with expression offsets kept.

    ``keep_symbolic_extents`` is used by the staged full-logical rewrite while
    its scheduled TIR still carries ``min(...)`` tail extents.  Existing callers
    keep the original integer-extent behaviour.
    """
    tir = _tir()
    out = []
    for arg in call.args:
        if not isinstance(arg, tir.Call) or _op_name(arg) != "tl.region":
            continue
        elements = list(arg.args)
        element = elements[0]
        buffer = getattr(element, "buffer", None)
        if buffer is None:
            continue
        offsets = list(getattr(element, "indices", ()))
        extents = ([x if keep_symbolic_extents else int(x)
                    for x in elements[2:]]
                   if len(elements) > 2 else [])
        out.append({"buffer": buffer, "offsets": offsets, "extents": extents})
    return out


def _loop_domain(loop: Any, *, role: str) -> int:
    """The one rule every consumed loop obeys.

    Analysis (binding a range) and instantiation (iterating) must agree on the
    domain, so both call this.  A loop this path does not model is rejected
    before any proof or rewrite; nothing here assumes a zero minimum.
    """
    minimum = _const(getattr(loop, "min", None)) if hasattr(loop, "min") else 0
    extent = _const(loop.extent)
    # this TVM fork carries no loop step: a For without one is a unit stride
    raw_step = getattr(loop, "step", None)
    step = 1 if raw_step is None else _const(raw_step)
    if minimum is None or extent is None or step is None:
        reject(RK_E_UNSUPPORTED,
               "a consumed loop must have constant bounds", stage="pass",
               actual={"role": role, "loop": str(loop.loop_var),
                       "min": str(getattr(loop, "min", None)),
                       "extent": str(loop.extent),
                       "step": str(getattr(loop, "step", None))})
    if minimum != 0 or step != 1 or extent < 1:
        reject(RK_E_UNSUPPORTED,
               "the pass chain only handles loops that run 0..extent with step 1",
               stage="pass",
               actual={"role": role, "loop": str(loop.loop_var),
                       "min": minimum, "extent": extent, "step": step})
    return int(extent)


def _bind_domains(loops):
    import tvm

    analyzer = _arith()
    for loop in loops:
        extent = _loop_domain(loop, role="analysis")
        analyzer.bind(loop.loop_var, tvm.ir.Range.from_min_extent(0, extent))
    return analyzer


def _consume_conditions(conditions: list[Any], *, indices: list[Any],
                        limits: list[int], analyzer: Any, role: str) -> dict:
    """Classify every path condition: consumed, redundant, or rejected.

    A condition either names one axis bound (index_axis < extent_axis) and is
    consumed as that bound, or it is provably true over the loop domain and is
    therefore redundant.  Anything else restricts the region beyond the buffer,
    which changes what is read or written, so it is rejected instead of dropped.
    """
    tir = _tir()
    consumed: dict[int, int] = {}
    for condition in conditions:
        if analyzer.can_prove(condition):
            continue
        matched = None
        if type(condition).__name__ == "LT":
            for axis in range(len(indices)):
                if analyzer.can_prove(tir.EQ(condition.a, indices[axis])) and                         _const(condition.b) == int(limits[axis]):
                    matched = axis
        if matched is None:
            reject(RK_E_SHAPE,
                   "a path condition is neither a buffer bound nor redundant; it "
                   "would change " + role, stage="pass",
                   actual={"role": role, "condition": str(condition)})
        if matched in consumed:
            reject(RK_E_SHAPE,
                   "two path conditions bound the same axis", stage="pass",
                   actual={"role": role, "axis": matched})
        consumed[matched] = _const(condition.b)
    return consumed


def _prove_under(analyzer: Any, conditions: list[Any], claim: Any) -> bool:
    """Prove the claim assuming every condition holds.

    Encoded as one disjunction so the arithmetic analyser is asked a single
    question: (not c1) or (not c2) or ... or claim.
    """
    tir = _tir()
    expression = claim
    for condition in reversed(conditions):
        expression = tir.Or(tir.Not(condition), expression)
    return bool(analyzer.can_prove(expression))


def _require_pure_control(expression: Any, *, role: str) -> None:
    """Do not let a Boolean proof erase evaluation of a call expression."""
    import tvm.tirx.stmt_functor as stmt_functor

    calls = []
    stmt_functor.post_order_visit(
        expression,
        lambda node: calls.append(node)
        if type(node).__name__ == "Call" else None)
    if calls:
        reject(RK_E_UNSUPPORTED,
               "a control expression contains a call whose evaluation is not "
               "consumed by the pass chain", stage="pass",
               actual={"role": role, "calls": [str(call) for call in calls]})


def _validate_call_positions(prim_func: Any) -> dict[str, Any]:
    """Calls are allowed only where a pass has a complete consumer for them.

    Tile operations and their tl.region descriptors are consumed by localise;
    launch_thread_idx placeholders are metadata-only and separately proved
    unused; generated rk_* extern calls are consumed by instantiate. Any call
    in an index, value, loop domain, buffer declaration, or unrelated statement
    has no such consumer and is refused before simplification can erase it.
    """
    import tvm.tirx.stmt_functor as stmt_functor

    all_calls = []
    stmt_functor.post_order_visit(
        prim_func.body,
        lambda node: all_calls.append(node)
        if type(node).__name__ == "Call" else None)
    allowed = []

    def expression_calls(expression: Any) -> list[Any]:
        import tvm

        if not isinstance(expression, tvm.ir.PrimExpr):
            return []
        calls = []
        stmt_functor.post_order_visit(
            expression,
            lambda node: calls.append(node)
            if type(node).__name__ == "Call" else None)
        return calls

    def reject_calls_in_metadata(value: Any, *, role: str) -> None:
        if value is None:
            return
        if type(value).__name__ in ("int", "str", "bool", "float"):
            return
        if type(value).__name__ in ("list", "tuple"):
            for item in value:
                reject_calls_in_metadata(item, role=role)
            return
        if hasattr(value, "items"):
            for key, item in value.items():
                reject_calls_in_metadata(item, role=role + "." + str(key))
            return
        if expression_calls(value):
            reject(RK_E_UNSUPPORTED,
                   "metadata contains a call expression with no consumer",
                   stage="pass", actual={"role": role, "value": str(value)})

    for _, node in _walk(prim_func.body):
        kind = type(node).__name__
        if kind == "For":
            for role, expr in (("loop minimum", getattr(node, "min", None)),
                               ("loop extent", node.extent),
                               ("loop step", getattr(node, "step", None))):
                if expr is not None and expression_calls(expr):
                    reject(RK_E_UNSUPPORTED,
                           "a loop domain contains a call with no consumer",
                           stage="pass", actual={"role": role, "expr": str(expr)})
            reject_calls_in_metadata(getattr(node, "annotations", None),
                                     role="loop annotations")
        elif kind == "Bind":
            calls = expression_calls(node.value)
            if (len(calls) != 1 or _op_name(calls[0]) != "tl.launch_thread_idx"
                    or not calls[0].same_as(node.value)):
                if calls:
                    reject(RK_E_UNSUPPORTED,
                           "a Bind contains an unconsumed call", stage="pass",
                           actual=str(node))
            else:
                allowed.extend(calls)
        elif kind == "Evaluate" and type(node.value).__name__ == "Call":
            call = node.value
            name = _op_name(call)
            if name in TILE_OPS:
                nested = expression_calls(call)
                for child in nested:
                    if not child.same_as(call) and _op_name(child) != "tl.region":
                        reject(RK_E_UNSUPPORTED,
                               "a tile operation argument contains a call other "
                               "than a consumed region descriptor", stage="pass",
                               actual={"op": name, "nested": str(child)})
                allowed.extend(nested)
            elif name.endswith("call_extern") and call.args and \
                    _name_of(call.args[0]) in (RK_GEMM, RK_COPY, RK_FILL,
                                                            RK_STORE, RK_SILU):
                nested = expression_calls(call)
                if any(not child.same_as(call) for child in nested):
                    reject(RK_E_UNSUPPORTED,
                           "an RKNPU task argument contains an unconsumed call",
                           stage="pass", actual=str(call))
                allowed.extend(nested)
            else:
                reject(RK_E_UNSUPPORTED,
                       "a call statement has no lowering consumer", stage="pass",
                       actual={"op": name, "call": str(call)})
        elif kind == "BufferStore":
            # The only math call admitted in a source value is the sigmoid used
            # by the separately proved SiLU accumulator update.
            nested = expression_calls(node.value)
            for child in nested:
                if not _op_name(child).endswith("sigmoid"):
                    reject(RK_E_UNSUPPORTED,
                           "a store value contains a call expression with no "
                           "lowering consumer", stage="pass",
                           actual={"call": str(child)})
            allowed.extend(nested)

    unknown = [call for call in all_calls
               if not any(call.same_as(consumer) for consumer in allowed)]
    if unknown:
        reject(RK_E_UNSUPPORTED,
               "a call expression occurs outside a consumed call position",
               stage="pass", actual=[str(call) for call in unknown[:8]])

    # Buffer/PrimFunc declarations are not necessarily visited as statements.
    # Their expressions must also be literal or ordinary call-free arithmetic.
    buffers = list((getattr(prim_func, "buffer_map", None) or {}).values())
    allocations = [buffer for _, node in _walk(prim_func.body)
                   if type(node).__name__ == "SBlock"
                   for buffer in (getattr(node, "alloc_buffers", ()) or ())]
    for buffer in buffers + allocations:
        for expression in (list(buffer.shape)
                           + list(getattr(buffer, "strides", ()) or ())
                           + [getattr(buffer, "elem_offset", None)]):
            if expression is not None and expression_calls(expression):
                reject(RK_E_UNSUPPORTED,
                       "a parameter buffer declaration contains an "
                       "unconsumed call", stage="pass",
                       actual={"buffer": str(buffer.name),
                               "expression": str(expression)})
    reject_calls_in_metadata(getattr(prim_func, "attrs", None),
                             role="PrimFunc attributes")
    for _, node in _walk(prim_func.body):
        if type(node).__name__ == "SBlock":
            reject_calls_in_metadata(getattr(node, "annotations", None),
                                     role="SBlock annotations")
        elif type(node).__name__ in ("For", "Call"):
            reject_calls_in_metadata(getattr(node, "annotations", None),
                                     role=type(node).__name__ + " annotations")
    return {"calls": len(all_calls), "allowed_positions": [
        "tileop/region at Evaluate", "unused launch-thread Bind",
        "generated rk_* task at Evaluate"]}


def _validate_container_semantics(prim_func: Any) -> dict[str, Any]:
    """Check container fields before any pass can erase them.

    The pilot consumes only blocks with no iteration bindings, init, or buffer
    aliases.  Their declarations and allocations are handled by the existing
    buffer contract.  Block predicates may be omitted only when they are true
    for every point in the enclosing loop/path domain.
    """
    tir = _tir()
    counts = {name: 0 for name in ("SBlockRealize", "SBlock", "AllocBuffer",
              "AttrStmt", "Bind", "DeclBuffer")}
    variable_uses: list[Any] | None = None
    thread_tags: list[str] = []

    def uses_of(variable: Any) -> list[Any]:
        nonlocal variable_uses
        if variable_uses is None:
            variable_uses = []
            import tvm.tirx.stmt_functor as stmt_functor
            stmt_functor.post_order_visit(
                prim_func.body,
                lambda expr: variable_uses.append(expr)
                if type(expr).__name__ == "Var" else None)
        return [expr for expr in variable_uses if expr.same_as(variable)]

    def visit(node: Any, loops: list[Any], conditions: list[Any]) -> None:
        nonlocal variable_uses
        kind = type(node).__name__
        if kind in counts:
            counts[kind] += 1
        if kind == "For":
            _loop_domain(node, role="container predicate")
            binding = getattr(node, "thread_binding", None)
            if binding is not None:
                tag = str(getattr(binding, "thread_tag", ""))
                if tag not in ("blockIdx.x", "blockIdx.y") or \
                        int(node.kind) != 4:
                    reject(RK_E_UNSUPPORTED,
                           "a kernel launch loop has an unsupported thread "
                           "binding", stage="pass",
                           actual={"loop": str(node.loop_var), "tag": tag,
                                   "binding_var": str(binding.var)})
                binding_uses = uses_of(binding.var)
                if binding_uses:
                    reject(RK_E_UNSUPPORTED,
                           "a launch binding variable is used in expressions "
                           "instead of the loop variable", stage="pass",
                           actual={"loop": str(node.loop_var),
                                   "binding_var": str(binding.var),
                                   "uses": len(binding_uses)})
                if tag in thread_tags:
                    reject(RK_E_UNSUPPORTED,
                           "kernel launch grid repeats a block axis", stage="pass",
                           actual=tag)
                thread_tags.append(tag)
            annotations = getattr(node, "annotations", None) or {}
            unknown = [str(key) for key in annotations
                       if str(key) != "num_stages"
                       and not str(key).startswith("rknpu.")]
            if unknown:
                reject(RK_E_UNSUPPORTED,
                       "a loop annotation has no execution/metadata contract",
                       stage="pass", actual=unknown)
            visit(node.body, loops + [node], conditions)
            return
        if kind == "IfThenElse":
            _require_pure_control(node.condition, role="IfThenElse condition")
            visit(node.then_case, loops, conditions + _conjuncts(node.condition))
            if node.else_case is not None:
                visit(node.else_case, loops, conditions + [tir.Not(node.condition)])
            return
        if kind == "SBlockRealize":
            block = node.block
            analyzer = _bind_domains(loops)
            predicate = node.predicate
            _require_pure_control(predicate, role="SBlockRealize predicate")
            if not _prove_under(analyzer, conditions, predicate):
                reject(RK_E_UNSUPPORTED,
                       "an SBlockRealize predicate is not provably true over its "
                       "enclosing domain", stage="pass",
                       actual={"predicate": str(predicate),
                               "path_conditions": [str(x) for x in conditions]})
            if list(getattr(node, "iter_values", ()) or ()):
                reject(RK_E_UNSUPPORTED,
                       "SBlockRealize iteration bindings are not consumed by "
                       "the pass chain", stage="pass",
                       actual={"iter_values": [str(x) for x in node.iter_values],
                               "iter_vars": [str(x) for x in block.iter_vars]})
            visit(block, loops, conditions)
            return
        if kind == "SBlock":
            if getattr(node, "init", None) is not None:
                reject(RK_E_UNSUPPORTED,
                       "SBlock.init has execution effects not consumed by the "
                       "pass chain", stage="pass", actual=str(node.init))
            if list(getattr(node, "match_buffers", ()) or ()):
                reject(RK_E_UNSUPPORTED,
                       "SBlock.match_buffers aliases are not consumed by the "
                       "pass chain", stage="pass",
                       actual=[str(x) for x in node.match_buffers])
            if list(getattr(node, "iter_vars", ()) or ()):
                reject(RK_E_UNSUPPORTED,
                       "SBlock.iter_vars bindings are not consumed by the "
                       "pass chain", stage="pass",
                       actual=[str(x) for x in node.iter_vars])
            annotations = getattr(node, "annotations", None) or {}
            known_metadata = {"tl.launch_threads"}
            unknown = [str(key) for key in annotations
                       if str(key) not in known_metadata
                       and not str(key).startswith("rknpu.")]
            if unknown:
                reject(RK_E_UNSUPPORTED,
                       "an SBlock annotation has no metadata contract", stage="pass",
                       actual=unknown)
            # alloc_buffers are checked below for identity, layout, and role;
            # launch metadata is recorded by pass_localise.
            visit(node.body, loops, conditions)
            return
        if kind == "Bind":
            value = getattr(node, "value", None)
            op_name = _op_name(value)
            args = list(getattr(value, "args", ()) or ())
            axis = _const(args[0]) if len(args) == 1 else None
            if op_name != "tl.launch_thread_idx" or axis not in (0, 1, 2):
                reject(RK_E_UNSUPPORTED,
                       "a Bind is not a recognized TileLang launch-thread "
                       "placeholder", stage="pass",
                       actual=str(node))
            # Only erase the placeholder if the bound value has no expression
            # use anywhere in this PrimFunc. The Bind itself has no body; later
            # statements in its scope remain siblings in their containing SeqStmt.
            uses = uses_of(node.var)
            if uses:
                reject(RK_E_UNSUPPORTED,
                       "a TileLang launch-thread binding is referenced by the "
                       "kernel body and has no RKNPU execution equivalent",
                       stage="pass", actual={"var": str(node.var),
                                              "axis": axis,
                                              "uses": len(uses)})
            return
        if kind in ("AllocBuffer", "AttrStmt", "DeclBuffer"):
            reject(RK_E_UNSUPPORTED,
                   kind + " carries storage/control semantics that are not "
                   "consumed by the pass chain", stage="pass",
                   actual=str(node))
        if kind == "SeqStmt":
            for item in node.seq:
                visit(item, loops, conditions)
        elif kind == "BufferStore":
            if node.predicate is not None:
                _require_pure_control(node.predicate, role="BufferStore predicate")
                analyzer = _bind_domains(loops)
                if not _prove_under(analyzer, conditions, node.predicate):
                    reject(RK_E_UNSUPPORTED,
                           "a BufferStore predicate is not provably true over "
                           "its enclosing domain", stage="pass",
                           actual={"buffer": str(node.buffer.name),
                                   "predicate": str(node.predicate)})
            return
        elif kind == "Evaluate":
            # These statements are consumed by localise/instantiate. Their
            # expression fields are analyzed by those passes.
            return
        else:
            reject(RK_E_UNSUPPORTED,
                   "a statement kind has no container/effect contract in this "
                   "pass chain", stage="pass",
                   actual={"kind": kind, "statement": str(node)})

    visit(prim_func.body, [], [])
    for _, node in _walk(prim_func.body):
        annotations = getattr(node, "annotations", None) or {}
        for key in annotations:
            name = str(key)
            if name.startswith("rknpu."):
                continue
            if type(node).__name__ == "SBlock" and name == "tl.launch_threads":
                launch = _plain(annotations[key])
                if not isinstance(launch, list) or len(launch) != 3 or any(
                        type(value) is not int or value < 1 for value in launch):
                    reject(RK_E_UNSUPPORTED,
                           "tl.launch_threads must be three positive integer "
                           "metadata values", stage="pass",
                           actual={"scope": _declaration_scope(node),
                                   "value": launch})
                continue
            if type(node).__name__ == "For" and name == "num_stages":
                stages = _declaration_value(annotations[key])
                if type(stages) is not int or stages < 1:
                    reject(RK_E_UNSUPPORTED,
                           "num_stages must be a positive integer metadata value",
                           stage="pass",
                           actual={"scope": _declaration_scope(node),
                                   "value": str(annotations[key])})
                continue
            if type(node).__name__ == "Call" and name == "coalesced_width":
                value = _declaration_value(annotations[key])
                if _op_name(node) != "tl.tileop.copy" or \
                        type(value) is not int or value < 1:
                    reject(RK_E_UNSUPPORTED,
                           "coalesced_width is only a positive integer hint on "
                           "a tile copy call", stage="pass",
                           actual={"scope": _declaration_scope(node),
                                   "value": str(annotations[key])})
                continue
            reject(RK_E_UNSUPPORTED,
                   "an annotation has no metadata contract at this scope",
                   stage="pass",
                   actual={"scope": _declaration_scope(node), "key": name,
                           "value": str(annotations[key])})
    calls_contract = _validate_call_positions(prim_func)
    if thread_tags and (
            not set(thread_tags) <= {"blockIdx.x", "blockIdx.y"} or
            len(thread_tags) != len(set(thread_tags))):
        reject(RK_E_UNSUPPORTED,
               "the kernel launch may bind only distinct blockIdx.x/blockIdx.y "
               "axes", stage="pass", actual=thread_tags)
    # SBlock.alloc_buffers are real storage declarations, not harmless wrapper
    # fields. The pilot accepts only distinct, compact, zero-offset allocations
    # that pass_localise can identify by name and supported scope.
    allocations = []
    for _, block in _walk(prim_func.body):
        if type(block).__name__ != "SBlock":
            continue
        allocations.extend(getattr(block, "alloc_buffers", ()) or ())
    names: set[str] = set()
    parameters = list((getattr(prim_func, "buffer_map", None) or {}).values())
    for buffer in parameters + allocations:
        name = str(buffer.name)
        is_allocation = any(buffer is item for item in allocations)
        if is_allocation and (name in names or
                              any(str(param.name) == name for param in parameters)):
            reject(RK_E_UNSUPPORTED,
                   "an allocated buffer name is ambiguous in the pass chain",
                   stage="pass", actual=name)
        if is_allocation:
            names.add(name)
        _buffer_descriptor(buffer, role="buffer " + name)
        if is_allocation and any(getattr(buffer, "data", None) is not None
               and getattr(param, "data", None) is not None
               and buffer.data.same_as(param.data) for param in parameters):
            reject(RK_E_UNSUPPORTED,
                   "an allocated buffer aliases a parameter buffer",
                   stage="pass", actual=name)
        for previous in allocations:
            if previous is buffer:
                break
            if previous.data.same_as(buffer.data):
                reject(RK_E_UNSUPPORTED,
                       "two allocated buffers share one data variable",
                       stage="pass",
                       actual={"first": str(previous.name), "second": name})
    declared = parameters + allocations
    references = []
    import tvm.tirx.stmt_functor as stmt_functor
    stmt_functor.post_order_visit(
        prim_func.body,
        lambda node: references.append(node.buffer)
        if type(node).__name__ in ("BufferLoad", "BufferStore") else None)
    for buffer in references:
        owners = [candidate for candidate in declared
                  if buffer.data.same_as(candidate.data)]
        if len(owners) != 1:
            reject(RK_E_UNSUPPORTED,
                   "a buffer access does not resolve to one declared storage "
                   "object", stage="pass",
                   actual={"buffer": str(buffer.name),
                           "matching_declarations": [str(x.name) for x in owners]})
        candidate = owners[0]
        if _buffer_descriptor(buffer, role="buffer access " + str(buffer.name)) != \
                _buffer_descriptor(candidate,
                                   role="declared buffer " + str(candidate.name)):
            reject(RK_E_UNSUPPORTED,
                   "a buffer access uses a view whose storage/layout differs "
                   "from its declaration", stage="pass",
                   actual={"access": str(buffer), "declaration": str(candidate)})
    # BufferLoad carries its own optional vector mask. Such masks have no
    # lowering representation in the task sequence; accept only a mask that
    # the complete enclosing loop domain proves true.
    tir = _tir()
    analyzer = _arith()
    loops = [node for _, node in _walk(prim_func.body)
             if type(node).__name__ == "For"]
    unique_loops = []
    for loop in loops:
        if not any(loop.loop_var.same_as(previous.loop_var)
                   for previous in unique_loops):
            unique_loops.append(loop)
    analyzer = _bind_domains(unique_loops)
    import tvm.tirx.stmt_functor as stmt_functor
    load_predicates = []
    stmt_functor.post_order_visit(
        prim_func.body,
        lambda expr: load_predicates.append(expr.predicate)
        if type(expr).__name__ == "BufferLoad"
        and getattr(expr, "predicate", None) is not None else None)
    for predicate in load_predicates:
        _require_pure_control(predicate, role="BufferLoad predicate")
        if not analyzer.can_prove(predicate):
            reject(RK_E_UNSUPPORTED,
                   "a BufferLoad predicate is not provably true over the "
                   "program loop domain", stage="pass",
                   actual=str(predicate))
    param_data = []
    for buffer in parameters:
        data = getattr(buffer, "data", None)
        if data is None:
            reject(RK_E_UNSUPPORTED,
                   "a parameter buffer has no data variable binding",
                   stage="pass", actual=str(buffer.name))
        if any(data.same_as(previous) for previous in param_data):
            reject(RK_E_UNSUPPORTED,
                   "parameter buffers may not alias in the supported task "
                   "binding", stage="pass", actual=str(buffer.name))
        param_data.append(data)
    return {"field_counts": counts,
            "calls": calls_contract,
            "alloc_buffers": sorted(names),
            "buffer_references": len(references),
            "contract": "predicate/masks true over domain; empty iter/init/alias; "
                        "distinct compact zero-offset alloc_buffers; known metadata"}


def _require_zero_elem_offset(buffer: Any, *, role: str) -> None:
    raw_offset = getattr(buffer, "elem_offset", None)
    offset = _known_const(raw_offset)
    if raw_offset is None or offset is None:
        reject(RK_E_UNSUPPORTED,
               "a buffer element offset is not statically known", stage="pass",
               actual={"role": role, "offset": str(raw_offset)})
    if offset != 0:
        reject(RK_E_UNSUPPORTED,
               "a buffer has a nonzero element offset not represented by the "
               "task binding", stage="pass",
               actual={"role": role, "offset": offset})


def _buffer_descriptor(buffer: Any, *, role: str) -> tuple:
    """The physical view fields this planner can represent for one buffer."""
    shape = tuple(_constant_shape(buffer, role=role))
    _require_zero_elem_offset(buffer, role=role)
    compact = []
    running = 1
    for extent in reversed(shape):
        compact.append(running)
        running *= extent
    compact.reverse()
    raw_strides = list(getattr(buffer, "strides", ()) or ())
    strides = tuple(_known_const(value) for value in raw_strides) if raw_strides else \
        tuple(compact)
    if strides != tuple(compact):
        reject(RK_E_UNSUPPORTED,
               "a buffer view has noncompact or unknown strides", stage="pass",
               actual={"role": role, "buffer": str(buffer.name),
                       "strides": [str(x) for x in raw_strides],
                       "compact": compact})
    separators = tuple(getattr(buffer, "axis_separators", ()) or ())
    if separators:
        reject(RK_E_UNSUPPORTED,
               "a buffer with axis separators has no task layout mapping",
               stage="pass", actual={"role": role, "buffer": str(buffer.name),
                                      "axis_separators": separators})
    return (str(buffer.name), str(buffer.dtype), shape, strides,
            _scope_of(buffer), separators)


def _loop_label(loop: Any) -> str:
    return str(getattr(loop, "loop_var", loop))


def _operand_load(entry: dict):
    """The single buffer read a guarded tile copy stores."""
    load = None
    for _, inner in _walk(entry["store"].value):
        if type(inner).__name__ == "BufferLoad":
            load = inner
    return load


def _prove_tile_index(analyzer: Any, *, reads: list[Any], locals: list[Any],
                      extents: list[int], loops: list[Any], message: str,
                      role: str) -> tuple[list[Any], list[Any]]:
    """Prove each axis is one consumed loop times its extent, plus the local index.

    Returns the loop variable that owns each axis (None when the block part is
    identically zero) together with the block part of each index, so two accesses
    can be shown to describe the same block instead of each being well formed on
    its own.  The domain is the caller's: nothing here assumes a loop name or a
    default grid.
    """
    tir = _tir()
    variables: list[Any] = []
    blocks: list[Any] = []
    for axis, (read, local) in enumerate(zip(reads, locals)):
        extent = int(extents[axis])
        part = analyzer.simplify(tir.Sub(read, local))
        # which consumed loop the index names is decided by the index itself, not
        # by which loop the analyser can prove equal to it: a single-block loop
        # is equal to zero without indexing anything
        named = _vars_in(part)
        owners = [loop for loop in loops
                  if any(loop.loop_var.same_as(var) for var in named)]
        if len(owners) > 1:
            reject(RK_E_SHAPE,
                   "the tile index names more than one block loop", stage="pass",
                   actual={"role": role, "axis": axis, "index": str(read),
                           "loops": [_loop_label(loop) for loop in owners]})
        if not owners:
            if not analyzer.can_prove(tir.EQ(part, tir.IntImm("int32", 0))):
                reject(RK_E_SHAPE, message, stage="pass",
                       actual={"role": role, "axis": axis, "index": str(read),
                               "block": str(part)})
            variables.append(None)
            blocks.append(part)
            continue
        expected = tir.Mul(owners[0].loop_var, tir.IntImm("int32", extent))
        if not analyzer.can_prove(tir.EQ(part, expected)):
            reject(RK_E_SHAPE, message, stage="pass",
                   actual={"role": role, "axis": axis, "index": str(read),
                           "block": str(part), "expected": str(expected)})
        variables.append(owners[0].loop_var)
        blocks.append(part)
    return variables, blocks

def _constant_shape(buffer: Any, *, role: str) -> list[int]:
    """Every extent this path reads must be a constant, or it is not consumed."""
    shape = [_const(value) for value in buffer.shape]
    if any(value is None for value in shape):
        reject(RK_E_UNSUPPORTED,
               "a " + role + " has a symbolic extent, which this pass chain does "
               "not consume", stage="pass",
               actual={"buffer": str(buffer.name),
                       "shape": [str(value) for value in buffer.shape]})
    return [int(value) for value in shape]


def _same_buffer(left: Any, right: Any) -> bool:
    """Whether two Buffer nodes denote the same allocation.

    Names are not identity: the regions a call carries and the allocation table
    are separate node graphs, so a chain must be checked on the object (or its
    data variable) and on dtype and shape, not on how the buffers happen to be
    spelled.
    """
    if left is None or right is None:
        return False
    if str(left.dtype) != str(right.dtype):
        return False
    if [int(x) for x in left.shape] != [int(x) for x in right.shape]:
        return False
    try:
        if bool(left.same_as(right)):
            return True
        return bool(left.data.same_as(right.data))
    except AttributeError:
        return False


def _require_whole_region(region: dict, buffer: Any, *, role: str) -> None:
    """A region must be the whole buffer: offset zero in every axis."""
    extents = [int(x) for x in region["extents"]]
    offsets = [_const(x) for x in region["offsets"]]
    shape = [int(x) for x in buffer.shape]
    if extents != shape or offsets != [0] * len(shape):
        reject(RK_E_NUMERIC_CONTRACT,
               "a tile operation region is not the whole buffer it names, so the "
               "chain would read or write part of it", stage="pass",
               actual={"role": role, "buffer": str(buffer.name),
                       "offsets": [str(x) for x in region["offsets"]],
                       "extents": extents, "shape": shape})


def _prove_tile_coverage(analyzer: Any, *, indices: list[Any], nest: list[Any],
                         extents: list[int], role: str) -> None:
    """Prove the store map covers the whole tile exactly once.

    Each tile axis must be written by exactly one loop variable of the copy nest,
    that variable must take exactly the tile extent, and two axes may not share
    it.  Together with "the nest iterates the tile extent" this is a bijection
    from the iterations onto the tile elements: every element is written once, so
    the loop kind (parallel or serial, repeated writes or not) cannot change the
    result.  An index that is not one loop variable -- a constant column, a
    strided or remapped element -- does not cover the tile and is refused.
    """
    tir = _tir()
    owners: list[Any] = []
    for axis, index in enumerate(indices):
        extent = int(extents[axis])
        named = _vars_in(index)
        matched = [loop for loop in nest
                   if any(loop.loop_var.same_as(var) for var in named)]
        if len(matched) != 1 or not analyzer.can_prove(
                tir.EQ(index, matched[0].loop_var)):
            reject(RK_E_SHAPE,
                   "a tile store index is not one loop variable of its own nest, "
                   "so the store does not cover the tile", stage="pass",
                   actual={"role": role, "axis": axis, "index": str(index),
                           "loops": [_loop_label(loop) for loop in matched]})
        if int(matched[0].extent) != extent:
            reject(RK_E_SHAPE,
                   "a tile store axis is indexed by a loop of a different "
                   "extent, so the store does not cover the tile", stage="pass",
                   actual={"role": role, "axis": axis, "index": str(index),
                           "iterates": int(matched[0].extent),
                           "tile": extent})
        owners.append(matched[0])
    if owners[0].loop_var.same_as(owners[1].loop_var):
        reject(RK_E_SHAPE,
               "one loop variable indexes both tile axes, so the store writes a "
               "row or column repeatedly", stage="pass",
               actual={"role": role, "loop": _loop_label(owners[0])})


def _loop_chain(node: Any) -> list[Any]:
    chain = []
    cursor = node
    while type(cursor).__name__ == "For":
        chain.append(cursor)
        cursor = cursor.body
    return chain


def _tile_op_role(call: Any, *, where: str) -> str:
    """Classify a tile operation against the exact table, or refuse it."""
    name = _op_name(call)
    role = TILE_OPS.get(name)
    if role is None:
        reject(RK_E_UNSUPPORTED,
               "a tile operation is not consumed by the pass chain and would "
               "change the result if it were dropped", stage="pass",
               actual={"op": name, "where": where,
                       "supported": sorted(TILE_OPS)})
    return role


def _is_tile_op_call(node: Any) -> bool:
    if type(node).__name__ != "Evaluate":
        return False
    return _op_name(node.value).startswith("tl.tileop.")


@dataclass
class LocalFacts:
    logical_m: int
    logical_k: int
    logical_n: int
    tile_m: int
    tile_n: int
    block_k: int
    rounding: str
    rounding_points: str
    output_name: str
    operand_a: str
    operand_b: str
    hints: dict[str, Any] = field(default_factory=dict)
    # The BufferStore statements this pass proved, by node identity.  Not part of
    # to_dict: instantiation must be able to say which particular store it is
    # allowed to consume, and a name is not that answer.
    drained: tuple = ()
    order: str = "m_major"
    b_role: str = "constant"
    source_hash: str = "rknpu-pass"
    activation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"logical": [self.logical_m, self.logical_k, self.logical_n],
                "tile": [self.tile_m, self.tile_n, self.block_k],
                "rounding": self.rounding, "order": self.order,
                "operands": [self.operand_a, self.operand_b],
                "output": self.output_name, "b_role": self.b_role,
                "activation": self.activation, "hints": self.hints}


def _grid_and_k_loops(prim_func: Any):
    """K loop = innermost serial loop holding tile ops; its ancestors = grid."""
    tir = _tir()
    nodes = list(_walk(prim_func.body))
    stack: list[tuple[int, Any]] = []
    k_loop = None
    k_loop_depth = -1
    grid: list[Any] = []
    for depth, node in nodes:
        while stack and stack[-1][0] >= depth:
            stack.pop()
        if not isinstance(node, tir.For):
            continue
        stack.append((depth, node))
        if int(node.kind) != int(tir.ForKind.SERIAL):
            continue
        def contains_compute(inner_node: Any) -> bool:
            if _is_tile_op_call(inner_node):
                return True
            if type(inner_node).__name__ != "Evaluate":
                return False
            call = inner_node.value
            return _op_name(call).endswith("call_extern") and bool(call.args) \
                and _name_of(call.args[0]) in {
                    RK_GEMM, RK_COPY, RK_FILL, RK_STORE, RK_SILU}
        if not any(contains_compute(inner) for _, inner in _walk(node)):
            continue
        if depth > k_loop_depth:
            k_loop_depth = depth
            k_loop = node
            grid = [loop for _, loop in stack[:-1]]
    for loop in ([k_loop] if k_loop is not None else []) + grid:
        _loop_domain(loop, role="grid or K")
    if k_loop is None or len(grid) != 2:
        reject(RK_E_SHAPE,
               "the RKNPU pass chain expects a serial K loop under a 2-D grid",
               stage="pass", actual={"k_loop": k_loop is not None,
                                     "grid": len(grid)})
    return grid[0], grid[1], k_loop


SUPPORTED_DECLARATIONS = {
    "rknpu.schedule": "schedule",
    "rknpu.tile_m": "tile_m",
    "rknpu.tile_n": "tile_n",
    "rknpu.k_tile": "k_tile",
    "rknpu.mn_order": "order",
    "rknpu.source_k_rounding": "source_rounding",
}


def _declared_order(declared: list[dict], *, stage: str) -> str:
    """Read one consistent M/N traversal request without guessing from names."""
    values = [row["value"] for row in declared
              if row["key"] == "rknpu.mn_order"]
    if not values:
        return "m_major"
    if len(set(values)) != 1 or values[0] not in {"m_major", "n_major"}:
        reject(RK_E_UNSUPPORTED,
               "rknpu.mn_order must declare one admitted traversal order",
               stage=stage, actual=values, required=["m_major", "n_major"])
    return str(values[0])


def _validate_declarations(declared: list[dict], *, tile_m: int, tile_n: int,
                           block_k: int, order: str,
                           source_rounding: str,
                           stage: str) -> list[dict]:
    """Every rknpu.* declaration must be understood and must agree with the IR.

    An unknown declaration is rejected rather than dropped: a caller that asks
    for something this path does not implement must be told so.
    """
    proven = {"tile_m": tile_m, "tile_n": tile_n, "k_tile": block_k,
              "schedule": "mn_tiles_v1", "order": order,
              "source_rounding": source_rounding}
    consumed = []
    for declaration in declared:
        key = declaration["key"]
        value = declaration["value"]
        if key not in SUPPORTED_DECLARATIONS:
            reject(RK_E_UNSUPPORTED,
                   "an rknpu annotation is not supported by the pass chain",
                   stage="pass", actual={**declaration})
        field = SUPPORTED_DECLARATIONS[key]
        expected = proven[field]
        if isinstance(expected, str):
            if type(value) is not str:
                reject(RK_E_UNSUPPORTED,
                       "an rknpu schedule declaration has the wrong value type",
                       stage="pass", actual=declaration,
                       required={"type": "str", "value": expected})
            if str(value) != expected:
                reject(RK_E_UNSUPPORTED,
                       "an rknpu schedule declaration disagrees with the "
                       "proven geometry", stage="pass",
                       actual=declaration, required={key: expected})
        else:
            if type(value) is not int:
                reject(RK_E_UNSUPPORTED,
                       "an rknpu tile declaration has the wrong value type",
                       stage="pass", actual=declaration,
                       required={"type": "int", "value": int(expected)})
            if value != int(expected):
                reject(RK_E_UNSUPPORTED,
                       "an rknpu tile declaration disagrees with the proven geometry",
                       stage="pass", actual=declaration,
                       required={key: int(expected)})
        consumed.append({**declaration, "required": expected,
                         "validation_stage": stage,
                         "consumed": True,
                         "reason": "matched the geometry proved from source accesses"})
    return consumed


def pass_localise(prim_func: Any, *,
                  source_declarations: list[dict] | None = None,
                  source_hints: list[dict] | None = None
                  ) -> tuple[Any, LocalFacts, Stage]:
    """Prove the local tile operations and lower them into RKNPU operations.

    The rewrite keeps the surrounding loops, guards and allocations and replaces
    each tile-op call with an RKNPU call whose arguments are facts proved here:
    the operand and accumulator buffers, the block-grid and K variables, and the
    tile geometry.
    """
    tir = _tir()
    container_contract = _validate_container_semantics(prim_func)
    grid_m, grid_n, k_loop = _grid_and_k_loops(prim_func)
    buffer_map = getattr(prim_func, "buffer_map", None) or {}
    params = {}
    for param in getattr(prim_func, "params", ()):
        buffer = buffer_map.get(param) if hasattr(buffer_map, "get") else None
        if buffer is not None:
            params[str(buffer.name)] = buffer
    allocations = {}
    for _, node in _walk(prim_func.body):
        if type(node).__name__ != "SBlock":
            continue
        for buffer in getattr(node, "alloc_buffers", ()) or ():
            allocations[str(buffer.name)] = buffer
    if len(params) != 3:
        reject(RK_E_SHAPE,
               "the guarded source chain expects A, B and one output parameter",
               stage="pass", actual=sorted(params))
    for name, buffer in list(params.items()) + list(allocations.items()):
        _constant_shape(buffer, role="buffer " + name)
    shared = [b for b in allocations.values() if _scope_of(b).startswith("shared")]
    fragments = [b for b in allocations.values()
                 if _scope_of(b).startswith("local")]
    fp16 = [b for b in fragments if str(b.dtype) == "float16"]
    fp32 = [b for b in fragments if str(b.dtype) == "float32"]
    if len(shared) != 2 or len(fp16) > 1 or len(fp32) != 1:
        reject(RK_E_NUMERIC_CONTRACT,
               "the pass chain expects two shared tiles and one FP32 fragment "
               "with an optional FP16 running accumulator", stage="pass",
               actual={"shared": len(shared), "fp16": len(fp16),
                       "fp32": len(fp32)})
    final_only = not fp16
    gemm_acc = fp32[0]
    if final_only:
        running = _tir().decl_buffer(
            gemm_acc.shape, "float16", name="Acc16",
            scope="local.fragment")
        allocations[str(running.name)] = running
    else:
        running = fp16[0]
    normalized_hints = _unapplied_annotation_hints(prim_func)
    launch_requests = [row for row in normalized_hints
                       if row["key"] == "tl.launch_threads"]
    launch_threads = (launch_requests[-1]["requested"]
                      if launch_requests else None)
    # TileLang thread extents are a GPU launch hint; an RKNPU plan has no
    # thread dimension, so they are recorded and never applied.
    hints_note = ({}
                  if launch_threads is None
                  else {"ignored_launch_threads": launch_threads})
    pipeline_records = [row for row in normalized_hints
                        if row["key"] == "num_stages"
                        and row["scope"] == _declaration_scope(k_loop)]
    pipeline_hint = (pipeline_records[-1]["requested"]
                     if pipeline_records else None)
    if pipeline_hint is not None:
        hints_note["pipeline_hint"] = {
            "num_stages": pipeline_hint, "applied": False,
            "reason": "an RKNPU plan has no software pipeline; the K chain is "
                      "executed as ordered submissions"}
    declared = _rknpu_declarations(prim_func)

    fills: list[dict[str, Any]] = []
    copies: list[dict[str, Any]] = []
    guarded: list[dict[str, Any]] = []
    proven: list[Any] = []
    gemm = None
    statements = []
    for position, node in enumerate(_statements(k_loop.body)):
        statements.append(type(node).__name__)
        if _is_tile_op_call(node):
            proven.append(node)
            call = node.value
            name = _op_name(call)
            role = _tile_op_role(call, where="the K loop body")
            regions = _region_exprs(call)
            expected_regions = {"gemm": 3, "fill": 1, "copy": 2}[role]
            if len(regions) != expected_regions:
                reject(RK_E_UNSUPPORTED,
                       "a tile operation has an unexpected number of region "
                       "arguments", stage="pass",
                       actual={"op": name, "regions": len(regions),
                               "required": expected_regions})
            for region_index, region in enumerate(regions):
                _require_whole_region(
                    region, region["buffer"],
                    role=("the tile " + role + " region " + str(region_index)))
            if role == "gemm":
                if gemm is not None:
                    reject(RK_E_SHAPE, "one tile GEMM per K iteration",
                           stage="pass")
                if len(call.args) != 13:
                    reject(RK_E_UNSUPPORTED,
                           "the tile GEMM call must use its complete consumed "
                           "argument signature", stage="pass",
                           actual=len(call.args), required=13)
                transpose_a = _known_const(call.args[3])
                transpose_b = _known_const(call.args[4])
                if transpose_a not in (0, 1) or transpose_b not in (0, 1):
                    reject(RK_E_UNSUPPORTED,
                           "tile GEMM transpose flags must be statically known",
                           stage="pass",
                           actual={"transpose_a": str(call.args[3]),
                                   "transpose_b": str(call.args[4])})
                if transpose_a or transpose_b:
                    reject(RK_E_UNSUPPORTED,
                           "tile GEMM transposition is not supported",
                           stage="pass",
                           actual={"transpose_a": transpose_a,
                                   "transpose_b": transpose_b})
                gemm_shape = tuple(_known_const(call.args[index])
                                   for index in (5, 6, 7))
                if any(value is None for value in gemm_shape):
                    reject(RK_E_UNSUPPORTED,
                           "tile GEMM geometry must be statically known",
                           stage="pass",
                           actual=[str(call.args[index]) for index in (5, 6, 7)])
                clear_accum = _known_const(call.args[9])
                if clear_accum not in (0, 1):
                    reject(RK_E_UNSUPPORTED,
                           "tile GEMM clear_accum must be statically known",
                           stage="pass", actual=str(call.args[9]))
                if clear_accum:
                    reject(RK_E_UNSUPPORTED,
                           "tile GEMM clear_accum is not supported: the pass "
                           "chain proves one accumulating K chain",
                           stage="pass", actual={"clear_accum": clear_accum})
                barrier = call.args[10]
                barrier_value = None if barrier is None else _known_const(barrier)
                if barrier is not None and barrier_value != 0:
                    reject(RK_E_UNSUPPORTED,
                           "a tile GEMM barrier argument is not supported",
                           stage="pass", actual=str(barrier))
                coords = tuple(_known_const(call.args[index]) for index in (11, 12))
                if any(value is None for value in coords):
                    reject(RK_E_UNSUPPORTED,
                           "tile GEMM accumulate coordinates must be "
                           "statically known", stage="pass",
                           actual=[str(call.args[index]) for index in (11, 12)])
                if any(value != 0 for value in coords):
                    reject(RK_E_UNSUPPORTED,
                           "tile GEMM accumulate coordinates are not supported",
                           stage="pass", actual=list(coords))
                policy = _known_const(call.args[8])
                if policy != 0:
                    reject(RK_E_UNSUPPORTED,
                           "tile GEMM policy is not the supported default",
                           stage="pass", actual=str(call.args[8]), required=0)
                gemm = {"regions": regions, "position": position,
                        "shape": gemm_shape,
                        "relaxed": {"policy": policy, "coords": coords}}
            elif role == "fill":
                if len(call.args) != 2:
                    reject(RK_E_UNSUPPORTED,
                           "tile fill must use its two-argument signature",
                           stage="pass", actual=len(call.args))
                value = _known_const(call.args[1])
                if value is None:
                    reject(RK_E_UNSUPPORTED,
                           "tile fill value must be statically known",
                           stage="pass", actual=str(call.args[1]))
                fills.append({"buffer": regions[0]["buffer"], "value": value,
                              "extents": regions[0]["extents"],
                              "position": position})
            elif role == "copy":
                if len(call.args) != 2:
                    reject(RK_E_UNSUPPORTED,
                           "tile copy must use its two-region signature",
                           stage="pass", actual=len(call.args))
                copies.append({"from": regions[0]["buffer"],
                               "to": regions[1]["buffer"],
                               "from_region": regions[0],
                               "to_region": regions[1],
                               "from_extents": regions[0]["extents"],
                               "to_extents": regions[1]["extents"],
                               "position": position})
        elif type(node).__name__ == "For":
            conditions: list[Any] = []
            shared_stores = 0
            guarded_stores = 0
            for _, inner in _walk(node):
                if type(inner).__name__ != "IfThenElse":
                    continue
                if getattr(inner, "else_case", None) is not None:
                    reject(RK_E_UNSUPPORTED,
                           "a guarded tile copy must not have an else branch",
                           stage="pass")
                conditions.extend(_conjuncts(inner.condition))
                store = _store_inside(inner.then_case)
                if store is not None:
                    guarded_stores += 1
                    proven.append(store)
                    guarded.append({"store": store, "conditions": conditions,
                                    "nest": _loop_chain(node),
                                    "position": position})
            for _, inner in _walk(node):
                if type(inner).__name__ != "BufferStore":
                    continue
                if not _scope_of(inner.buffer).startswith(("shared", "local")):
                    continue
                shared_stores += 1
            if shared_stores != guarded_stores or guarded_stores != 1:
                reject(RK_E_UNSUPPORTED,
                       "every non-global store inside the K loop must be the "
                       "one guarded copy of a tile", stage="pass",
                       actual={"stores": shared_stores,
                               "guarded": guarded_stores})
        else:
            reject(RK_E_UNSUPPORTED,
                   "a statement inside the K loop is not consumed",
                   stage="pass", actual=type(node).__name__)
    if gemm is None:
        reject(RK_E_UNSUPPORTED, "no tile GEMM inside the K loop", stage="pass")
    # Effect closure: the K body must contain exactly the operations this path
    # consumes, one per role.  A fill, copy or store with any other target keeps
    # a state change that the instantiated task sequence would drop, so it is
    # rejected here instead of being recognised and then lost.
    if len(fills) != 2:
        reject(RK_E_UNSUPPORTED,
               "a fill inside the K loop is not consumed by the pass chain",
               stage="pass",
               actual={"fills": [str(f["buffer"].name) for f in fills]})
    if (not final_only and len(copies) != 2) or (final_only and copies):
        # final-only has no round trip; explicit mode has exactly one load and
        # one FP16 writeback.
        reject(RK_E_UNSUPPORTED,
               "a copy inside the K loop is not consumed by the pass chain",
               stage="pass",
               actual={"copies": [[str(c["from"].name), str(c["to"].name)]
                                  for c in copies]})
    if len(guarded) != 2:
        reject(RK_E_SHAPE,
               "the K loop must fill both shared tiles through guarded copies",
               stage="pass", actual=len(guarded))

    sources: list[str] = []
    for entry in guarded:
        store = entry["store"]
        target = store.buffer
        if not _scope_of(target).startswith("shared"):
            reject(RK_E_SHAPE, "a guarded tile copy must write a shared tile",
                   stage="pass", actual=str(target.name))
        if not any(_same_buffer(target, tile) for tile in shared):
            reject(RK_E_SHAPE,
                   "a guarded tile copy writes a buffer that is not the shared "
                   "tile the rest of the chain reads", stage="pass",
                   actual=str(target.name))
        load = None
        for _, inner in _walk(store.value):
            if type(inner).__name__ == "BufferLoad":
                load = inner
        if load is None:
            reject(RK_E_SHAPE, "guarded copy has no operand read", stage="pass")
        if not _structurally_equal(store.value, load):
            reject(RK_E_NUMERIC_CONTRACT,
                   "the guarded copy must store exactly the operand read",
                   stage="pass")
        nest_extents = [_loop_domain(loop, role="operand copy")
                        for loop in entry["nest"]]
        if nest_extents != \
                [int(x) for x in target.shape]:
            reject(RK_E_SHAPE, "the guarded copy must iterate the whole tile",
                   stage="pass",
                   actual={"iterates": nest_extents,
                           "tile": [int(x) for x in target.shape]})
        name = str(load.buffer.name)
        operand = params.get(name)
        if operand is None:
            reject(RK_E_SHAPE, "the guarded copy must read a parameter",
                   stage="pass", actual=name)
        domains = [grid_m, grid_n, k_loop] + list(entry["nest"])
        entry_analyzer = _bind_domains(domains)
        entry_consumed = _consume_conditions(
            entry["conditions"], indices=list(load.indices),
            limits=[int(operand.shape[0]), int(operand.shape[1])],
            analyzer=entry_analyzer,
            role="the copy of " + name + " into " + str(target.name))
        _prove_tile_coverage(entry_analyzer, indices=list(store.indices),
                             nest=list(entry["nest"]),
                             extents=[int(x) for x in target.shape],
                             role="the copy of " + name + " into " + str(target.name))
        for axis in range(2):
            limit = int(operand.shape[axis])
            offset = load.indices[axis]
            # the copy is element wise: this iteration reads exactly one element,
            # so the legality claim is "index + 1 <= extent" under the path
            # conditions, on the same domain the instantiation will use
            legal = tir.And(tir.LE(0, offset), tir.LE(tir.Add(offset, 1), limit))
            if not _prove_under(entry_analyzer, entry["conditions"], legal):
                reject(RK_E_SHAPE,
                       "the operand access is not provably inside the buffer "
                       "over the enclosing domain", stage="pass",
                       actual={"operand": name, "axis": axis,
                               "offset": str(offset), "limit": limit})
        # every axis is one block loop times the tile plus the fragment index;
        # which loop, and whether the coordinate folded to zero, is proved here
        # and consumed below against the output block coordinates.
        entry["variables"], entry["blocks"] = _prove_tile_index(
            entry_analyzer, reads=list(load.indices),
            locals=list(store.indices),
            extents=[int(x) for x in target.shape],
            loops=[grid_m, grid_n, k_loop],
            message="the operand read index is not tile_extent * tile + "
                    "local_index",
            role="the copy of " + name + " into " + str(target.name))

        sources.append(name)
    if len(set(sources)) != 2:
        reject(RK_E_SHAPE,
               "the two guarded copies must read two distinct operands",
               stage="pass", actual=sources)

    for tile in shared:
        matching = [f for f in fills if _same_buffer(f["buffer"], tile)]
        if len(matching) != 1:
            reject(RK_E_NUMERIC_CONTRACT,
                   "each shared tile is zero filled exactly once per K block",
                   stage="pass", actual=str(tile.name))
        if matching[0]["value"] != 0:
            reject(RK_E_NUMERIC_CONTRACT, "the tile fill must be zero",
                   stage="pass", actual=str(tile.name))
        if matching[0]["extents"] != [int(x) for x in tile.shape]:
            reject(RK_E_SHAPE, "the tile fill must cover the whole tile",
                   stage="pass", actual=str(tile.name))
        copy_positions = [entry["position"] for entry in guarded
                          if str(entry["store"].buffer.name) == str(tile.name)]
        if copy_positions and matching[0]["position"] > copy_positions[0]:
            reject(RK_E_NUMERIC_CONTRACT,
                   "a shared tile is filled after the copy that fills it",
                   stage="pass", actual=str(tile.name))

    loads: list[dict[str, Any]] = []
    writebacks: list[dict[str, Any]] = []
    if not final_only:
        loads = [c for c in copies
                 if str(c["from"].name) == str(running.name)
                 and str(c["to"].name) == str(gemm_acc.name)]
        writebacks = [c for c in copies
                      if str(c["from"].name) == str(gemm_acc.name)
                      and str(c["to"].name) == str(running.name)]
        if len(loads) != 1 or len(writebacks) != 1:
            reject(RK_E_NUMERIC_CONTRACT,
                   "the K loop must load and write back the running "
                   "accumulator once", stage="pass",
                   actual={"loads": len(loads),
                           "writebacks": len(writebacks)})
        if loads[0]["position"] > gemm["position"]:
            reject(RK_E_NUMERIC_CONTRACT,
                   "the accumulator load must precede the tile GEMM",
                   stage="pass")
        if writebacks[0]["position"] < gemm["position"]:
            reject(RK_E_NUMERIC_CONTRACT,
                   "the FP16 writeback must follow the tile GEMM",
                   stage="pass")
        for entry in (loads[0], writebacks[0]):
            if entry["from_extents"] != [int(x) for x in running.shape] or \
                    entry["to_extents"] != [int(x) for x in gemm_acc.shape]:
                reject(RK_E_NUMERIC_CONTRACT,
                       "an accumulator copy must cover the whole fragment",
                       stage="pass")
    for entry in guarded:
        if entry["position"] > gemm["position"]:
            reject(RK_E_NUMERIC_CONTRACT,
                   "a guarded operand copy must precede the tile GEMM",
                   stage="pass")

    # ---- operand roles from the proved block map, never from the names
    k_var = k_loop.loop_var
    tile_by_name = {str(b.name): b for b in shared}
    gemm_regions = gemm["regions"]
    gemm_tiles = [str(region["buffer"].name) for region in gemm_regions]
    tile_source = {}
    for entry in guarded:
        load = _operand_load(entry)
        tile_source[str(entry["store"].buffer.name)] = str(load.buffer.name)
    k_trips = _const(k_loop.extent)
    k_axis: dict[str, int] = {}
    for entry in guarded:
        name = str(_operand_load(entry).buffer.name)
        destination = str(entry["store"].buffer.name)
        expected_axis = (1 if destination == gemm_tiles[0]
                         else 0 if destination == gemm_tiles[1] else None)
        axes = [axis for axis in range(2)
                if entry["variables"][axis] is not None
                and entry["variables"][axis].same_as(k_var)]
        if len(axes) == 0 and expected_axis is not None and k_trips == 1:
            # A single-trip K loop may fold its variable to the actual zero
            # coordinate.  The role still fixes which tile axis is K; the
            # degenerate coordinate must prove to the actual constant zero.
            block = entry["blocks"][expected_axis]
            if _const(block) != 0:
                reject(RK_E_SHAPE,
                       "a single-trip K coordinate must fold to zero",
                       stage="pass", actual={"operand": name,
                                             "block": str(block)})
            axes = [expected_axis]
        if len(axes) != 1 or name in k_axis or axes[0] != expected_axis:
            reject(RK_E_SHAPE,
                   "an operand tile must read one K axis from the K block loop",
                   stage="pass", actual={"operand": name, "k_axes": axes,
                                         "expected_axis": expected_axis})
        k_axis[name] = axes[0]
    arg_a = tile_source.get(gemm_tiles[0])
    arg_b = tile_source.get(gemm_tiles[1])
    if k_axis.get(arg_a) != 1 or k_axis.get(arg_b) != 0:
        reject(RK_E_SHAPE,
               "the GEMM operands do not match the tile they are filled into",
               stage="pass", actual={"gemm_args": gemm_tiles,
                                     "k_axis": {k: v for k, v in k_axis.items()}})
    operand_a = params[arg_a]
    operand_b = params[arg_b]
    for label, buffer in (("A", operand_a), ("B", operand_b)):
        _require_operand_contract(buffer, label=label)
    for tile in shared:
        _require_operand_contract(tile, label=str(tile.name), parameter=False)
    logical_m = int(operand_a.shape[0])
    logical_k = int(operand_a.shape[1])
    logical_n = int(operand_b.shape[1])
    if int(operand_b.shape[0]) != logical_k:
        reject(RK_E_SHAPE, "the two operands disagree on K", stage="pass",
               actual=[int(operand_a.shape[1]), int(operand_b.shape[0])])
    tile_a = tile_by_name[gemm_tiles[0]]
    tile_b = tile_by_name[gemm_tiles[1]]
    tile_m = int(tile_a.shape[0])
    block_k = int(tile_a.shape[1])
    tile_n = int(tile_b.shape[1])
    if int(tile_b.shape[0]) != block_k:
        reject(RK_E_SHAPE, "the two operand tiles disagree on the K extent",
               stage="pass")
    if gemm.get("shape") != (tile_m, tile_n, block_k):
        reject(RK_E_SHAPE,
               "the tile GEMM geometry disagrees with the tiles it consumes",
               stage="pass",
               actual={"gemm": gemm.get("shape"),
                       "tiles": [tile_m, tile_n, block_k]})

    # ---- one def/use chain: the tile GEMM must read the tiles the copies fill
    # and write the accumulator the copies and the epilogue exchange.  Each link
    # is checked on the buffer object and as a whole region, so a source that
    # computes into one buffer and reads another is refused instead of being
    # planned as the default accumulating GEMM.
    gemm_regions = gemm["regions"]
    for role, region, buffer in (
            ("first operand", gemm_regions[0], tile_a),
            ("second operand", gemm_regions[1], tile_b),
            ("result accumulator", gemm_regions[2], gemm_acc)):
        if not _same_buffer(region["buffer"], buffer):
            reject(RK_E_SHAPE,
                   "the tile GEMM " + role + " is not the buffer the rest of the "
                   "chain reads and writes", stage="pass",
                   actual={"role": role, "gemm_region": str(region["buffer"].name),
                           "dtype": str(region["buffer"].dtype),
                           "expected": str(buffer.name),
                           "expected_dtype": str(buffer.dtype)})
        _require_whole_region(region, buffer, role="the tile GEMM " + role)
    if not final_only:
        for role, region, buffer in (
                ("load target", loads[0]["to_region"], gemm_acc),
                ("load source", loads[0]["from_region"], running),
                ("writeback source", writebacks[0]["from_region"], gemm_acc),
                ("writeback target", writebacks[0]["to_region"], running)):
            if not _same_buffer(region["buffer"], buffer):
                reject(RK_E_SHAPE,
                       "an accumulator copy " + role + " is not the buffer the "
                       "chain binds", stage="pass",
                       actual={"role": role,
                               "region": str(region["buffer"].name),
                               "dtype": str(region["buffer"].dtype),
                               "expected": str(buffer.name)})
    # ---- output store: exactly one, tile local, guarded or exactly covered
    output_store_buffer = gemm_acc if final_only else running
    output = _prove_output_store(prim_func, k_loop, output_store_buffer,
                                 (grid_m, grid_n), tile_m, tile_n,
                                 logical_m, logical_n, params)
    source_declarations = (declared if source_declarations is None
                           else source_declarations)
    if final_only and any(row["key"] == "rknpu.schedule"
                          for row in source_declarations):
        reject(RK_E_NUMERIC_CONTRACT,
               "scheduled source narrows only after the full K reduction; the "
               "tiled schedule writes FP16 back at every 256-wide K block, so "
               "the two describe different numerical programs",
               stage="source")
    source_order = _declared_order(source_declarations, stage="source")
    normalized_order = _declared_order(declared, stage="normalized")
    if source_order != normalized_order:
        reject(RK_E_UNSUPPORTED,
               "source and normalized rknpu.mn_order declarations disagree",
               stage="pass", actual={"source": source_order,
                                     "normalized": normalized_order})
    consumed_source_declarations = _validate_declarations(
        source_declarations, tile_m=tile_m, tile_n=tile_n,
        block_k=block_k, order=source_order,
        source_rounding=("fp32_final_narrowing" if final_only
                         else "explicit_per_k_fp16"), stage="source")
    consumed_normalized_declarations = _validate_declarations(
        declared, tile_m=tile_m, tile_n=tile_n,
        block_k=block_k, order=normalized_order,
        source_rounding=("fp32_final_narrowing" if final_only
                         else "explicit_per_k_fp16"), stage="normalized")
    if consumed_source_declarations or consumed_normalized_declarations:
        hints_note["rknpu_declarations"] = {
            "source": consumed_source_declarations,
            "normalized": consumed_normalized_declarations,
            "explanation": "every original and normalized declaration was checked "
                           "against the proved geometry; equal duplicates remain "
                           "separate records"}
    source_hints = (normalized_hints if source_hints is None else source_hints)
    if source_hints or normalized_hints:
        hints_note["unapplied_annotations"] = {
            "source": source_hints,
            "normalized": normalized_hints,
            "explanation": "each known non-RKNPU hint occurrence is retained with "
                           "its scope and reason it is not applied"}
    # ---- one computation: operand slices, K blocks and the output block must be
    # the same coordinate map, so the plan is assembled from these proved
    # instances instead of from a default GEMM assumption.  Each of the three is
    # a different expression in the source, so agreeing is a real obligation.
    chain = _bind_domains([grid_m, grid_n, k_loop]
                          + [loop for entry in guarded
                             for loop in entry["nest"]]
                          + list(output["nest"]))
    consumed = {}
    for entry in guarded:
        consumed[str(_operand_load(entry).buffer.name)] = entry
    for role, name, axis, index, left, right in (
            ("A", arg_a, 0, "row", consumed[arg_a]["blocks"][0], output["rows"]),
            ("B", arg_b, 1, "column", consumed[arg_b]["blocks"][1],
             output["cols"])):
        if chain.can_prove(tir.EQ(left, right)):
            continue
        operand_folded = consumed[name]["variables"][axis] is None
        output_folded = output["variables"][axis] is None
        if operand_folded and not output_folded:
            message = ("the " + role + " block coordinate folded away while the "
                       "output block is still indexed by a block loop: the plan "
                       "would read one slice for every block")
        elif output_folded and not operand_folded:
            message = ("the output block coordinate folded away while the " + role
                       + " is still indexed by a block loop: the plan would "
                       "write one block for every slice the program reads")
        else:
            message = ("the " + role + " block coordinate and the output block "
                       "coordinate are different computations; the plan would "
                       "read a slice this program never writes")
        reject(RK_E_SHAPE, message, stage="pass",
               actual={"operand": name, "role": role, "axis": index,
                       "operand_block": str(left), "output_block": str(right)})
    if not chain.can_prove(tir.EQ(consumed[arg_a]["blocks"][1],
                                  consumed[arg_b]["blocks"][0])):
        reject(RK_E_SHAPE,
               "the two operands do not share one K block coordinate",
               stage="pass",
               actual={"a": str(consumed[arg_a]["blocks"][1]),
                       "b": str(consumed[arg_b]["blocks"][0])})
    proven.extend(output["proven"])
    _require_effect_closure(prim_func, proven)
    facts = LocalFacts(logical_m=logical_m, logical_k=logical_k,
                       logical_n=logical_n, tile_m=tile_m, tile_n=tile_n,
                       block_k=block_k,
                       rounding=("fp32_final_narrowing" if final_only
                                 else "explicit_per_k_fp16"),
                       rounding_points=("final_output_only" if final_only
                                        else "after_each_k_block"),
                       output_name=output["name"], operand_a=arg_a,
                       operand_b=arg_b, hints=hints_note,
                       order=source_order,
                       drained=tuple(node for node in proven
                                     if type(node).__name__ == "BufferStore"))
    def block_offset(variable: Any, extent: int) -> Any:
        return tir.Mul(variable, tir.IntImm("int32", extent))

    def tail_extent(extent: int, base: Any, logical: int) -> Any:
        return tir.min(tir.IntImm("int32", extent),
                       tir.Sub(tir.IntImm("int32", logical), base))

    tile_role = {str(tile_a.name): "A", str(tile_b.name): "B"}
    operand_copies: list[dict[str, Any]] = []
    for entry in guarded:
        destination = str(entry["store"].buffer.name)
        role = tile_role.get(destination)
        if role is None:
            reject(RK_E_UNSUPPORTED,
                   "a guarded store inside the K loop is not an A/B tile copy",
                   stage="pass", actual=destination)
        load = _operand_load(entry)
        if role == "A":
            src_row = block_offset(output["variables"][0], tile_m)
            src_col = block_offset(k_loop.loop_var, block_k)
            rows = tail_extent(tile_m, src_row, logical_m)
            cols = tail_extent(block_k, src_col, logical_k)
        else:
            src_row = block_offset(k_loop.loop_var, block_k)
            src_col = block_offset(output["variables"][1], tile_n)
            rows = tail_extent(block_k, src_row, logical_k)
            cols = tail_extent(tile_n, src_col, logical_n)
        operand_copies.append({
            "loop": entry["nest"][0],
            "source": str(load.buffer.name),
            "destination": destination,
            "source_region": {"row": src_row, "col": src_col,
                              "rows": rows, "cols": cols},
            "destination_region": {"row": tir.IntImm("int32", 0),
                                   "col": tir.IntImm("int32", 0),
                                   "rows": rows, "cols": cols},
            "zero_padding": True,
        })
    if {copy["destination"] for copy in operand_copies} != set(tile_role):
        reject(RK_E_UNSUPPORTED,
               "the K loop must stage both A and B through guarded tile copies",
               stage="pass", actual=[copy["destination"]
                                     for copy in operand_copies])
    rewritten = _rewrite_tile_ops(prim_func, k_loop, output["variables"],
                                  running, gemm_acc, tile_m, tile_n, block_k,
                                  (arg_a, arg_b, output["name"]), output,
                                  operand_copies, logical_m, logical_n,
                                  final_only=final_only)

    return rewritten, facts, Stage(
        "localise", "transform", _digest(rewritten.script()),
        {"guarded_copies": len(guarded), "fills": len(fills),
         "copies": len(copies), "statements": statements,
         "facts": facts.to_dict(),
         "container_contract": container_contract})


def _vars_in(expr):
    tir = _tir()
    found = []

    def visit(node):
        if isinstance(node, tir.Var):
            found.append(node)

    import tvm.tirx.stmt_functor as stmt_functor

    stmt_functor.post_order_visit(expr, visit)
    return found


def _require_effect_closure(prim_func: Any, proven: list[Any]) -> None:
    """Every statement with a state effect must be one the proof consumed.

    The proofs walk the statements they were handed -- the sequence that owns the
    K loop, the loop body, the epilogue -- so a write in an ancestor or a sibling
    sequence was never looked at.  This walks the whole PrimFunc: a BufferStore
    or a call whose effect this chain cannot restate is refused, whatever buffer
    it names.  Sharing a name with a buffer the plan owns is not a proof, so
    nothing is dropped for that reason.
    """
    for _, node in _walk(prim_func.body):
        kind = type(node).__name__
        if kind not in ("BufferStore", "Evaluate"):
            continue
        if any(node.same_as(other) for other in proven):
            continue
        if kind == "BufferStore":
            detail = {"kind": kind, "target": str(node.buffer.name),
                      "indices": [str(value) for value in node.indices]}
        else:
            detail = {"kind": kind, "call": _op_name(node.value)}
        reject(RK_E_UNSUPPORTED,
               "a statement with a state effect is not consumed by the pass "
               "chain; dropping it would change the result", stage="pass",
               actual=detail)


def _root_sequence(prim_func: Any, k_loop: Any):
    """Statements of the sequence that owns the K loop, plus the loop index."""
    nodes = list(_walk(prim_func.body))
    for _, node in nodes:
        seq = getattr(node, "seq", None)
        if not seq:
            continue
        items = list(seq)
        for position, item in enumerate(items):
            if _structurally_equal(item, k_loop):
                return items, position
    reject(RK_E_SHAPE,
           "cannot enumerate the statements outside the K loop; refusing to "
           "skip the output proof", stage="pass")


def _require_operand_contract(buffer: Any, *, label: str,
                              parameter: bool = True) -> None:
    """Prove the buffer the plan will bind, instead of rewriting its type.

    The plan binds FP16 operands with row-major strides, and the plan's own
    numeric contract is the FP16 one.  A source whose parameter or shared tile
    has another dtype or a strided layout cannot be represented, so it is
    refused at the entry rather than accepted and silently retyped.
    """
    dtype = str(buffer.dtype)
    if dtype != "float16":
        reject(RK_E_NUMERIC_CONTRACT,
               "the pass chain proves an FP16 interface; this "
               + ("parameter" if parameter else "tile") + " is " + dtype
               + ", and retyping it would change the signature",
               stage="pass", actual={"buffer": label, "dtype": dtype},
               required={"dtype": "float16"})
    strides = list(getattr(buffer, "strides", ()) or ())
    if parameter and strides:
        layout = [_const(value) for value in strides]
        if layout != [int(buffer.shape[1]), 1]:
            reject(RK_E_UNSUPPORTED,
                   "a parameter whose rows are not contiguous is not consumed "
                   "by the pass chain", stage="pass",
                   actual={"buffer": label,
                           "strides": [str(value) for value in strides]},
                   required={"strides": [int(buffer.shape[1]), 1]})


def _prove_output_store(prim_func: Any, k_loop: Any, running: Any,
                        grid_loops: tuple, tile_m: int, tile_n: int,
                        logical_m: int, logical_n: int,
                        params: dict) -> dict[str, Any]:
    """Prove the single epilogue store writes exactly the published blocks.

    Each axis is decided on the epilogue's own iteration domain.  A guard is
    either consumed as that axis' bound (and must name the logical extent) or
    proven redundant over the domain; an axis with no consumed guard must be
    covered exactly by the grid, which is what makes an unguarded store safe and
    complete at the same time.  The store coordinate must then be that axis'
    block loop times the tile plus the fragment index, so the epilogue writes the
    block the plan publishes and no other.  A block coordinate that folded to
    zero is accepted only when the axis really has one block, and the loop the
    plan will use is returned so nothing falls back to a default grid.
    """
    tir = _tir()
    items, loop_position = _root_sequence(prim_func, k_loop)
    fill_position = None
    store_position = None
    store = None
    conditions: list[Any] = []
    nest: list[Any] = []
    for position, item in enumerate(items):
        if _structurally_equal(item, k_loop):
            continue
        if type(item).__name__ == "Evaluate" and _op_name(item.value).endswith(
                "fill"):
            regions = _region_exprs(item.value)
            value = _const(item.value.args[1]) if len(item.value.args) > 1 else None
            if not _same_buffer(regions[0]["buffer"], running):
                reject(RK_E_UNSUPPORTED,
                       "an unsupported fill outside the K loop", stage="pass")
            _require_whole_region(regions[0], running,
                                 role="the accumulator clear outside the K loop")
            if value != 0 or regions[0]["extents"] != [int(x) for x in running.shape]:
                reject(RK_E_NUMERIC_CONTRACT,
                       "the running accumulator must be cleared to zero over the "
                       "whole fragment", stage="pass")
            if fill_position is not None:
                reject(RK_E_UNSUPPORTED,
                       "a second accumulator clear outside the K loop is not "
                       "consumed", stage="pass")
            fill_position = position
            clear_statement = item
            continue
        if type(item).__name__ == "For":
            depth_stores = []
            for _, inner in _walk(item):
                if type(inner).__name__ in ("Evaluate", "Allocate",
                                            "AllocateConst"):
                    reject(RK_E_UNSUPPORTED,
                           "a tile operation or allocation outside the K loop is "
                           "not consumed", stage="pass",
                           actual=type(inner).__name__)
                if type(inner).__name__ == "IfThenElse":
                    if getattr(inner, "else_case", None) is not None:
                        reject(RK_E_UNSUPPORTED,
                               "the output store must not have an else branch",
                               stage="pass")
                    conditions.extend(_conjuncts(inner.condition))
                if type(inner).__name__ == "BufferStore":
                    depth_stores.append(inner)
            for candidate in depth_stores:
                if store is not None:
                    reject(RK_E_UNSUPPORTED,
                           "the kernel must write exactly one output value",
                           stage="pass",
                           actual=str(candidate.buffer.name))
                store = candidate
            nest = _loop_chain(item)
            nest_extents = [_loop_domain(loop, role="epilogue")
                            for loop in nest]
            if nest_extents != [tile_m, tile_n]:
                reject(RK_E_SHAPE,
                       "the epilogue must iterate the whole tile",
                       stage="pass",
                       actual={"iterates": nest_extents,
                               "tile": [tile_m, tile_n]})
            store_position = position
            continue
        reject(RK_E_UNSUPPORTED,
               "a statement outside the K loop is not consumed", stage="pass",
               actual=type(item).__name__)
    if store is None or fill_position is None:
        reject(RK_E_UNSUPPORTED,
               "the kernel must clear the accumulator and write the output",
               stage="pass")
    if not (fill_position < loop_position < store_position):
        reject(RK_E_NUMERIC_CONTRACT,
               "the accumulator must be cleared before the K loop and the output "
               "store must follow it", stage="pass",
               actual={"fill": fill_position, "loop": loop_position,
                       "store": store_position})
    stored = store.value
    if type(stored).__name__ == "Cast":
        stored = stored.value
    if type(stored).__name__ != "BufferLoad":
        reject(RK_E_NUMERIC_CONTRACT,
               "the output store must store the accumulator value itself",
               stage="pass")
    if not _same_buffer(stored.buffer, running):
        reject(RK_E_NUMERIC_CONTRACT,
               "the output store must read the running accumulator",
               stage="pass")
    target = store.buffer
    if _scope_of(target).startswith(("shared", "local")):
        reject(RK_E_SHAPE, "the output store must write a parameter",
               stage="pass", actual=str(target.name))
    candidates = [name for name, buffer in params.items()
                  if _same_buffer(target, buffer)]
    if len(candidates) != 1:
        reject(RK_E_SHAPE,
               "the output parameter must be unambiguous", stage="pass",
               actual=candidates)
    output_name = candidates[0]
    if [int(x) for x in target.shape] != [logical_m, logical_n]:
        reject(RK_E_SHAPE,
               "the single global store does not write the logical output "
               "shape", stage="pass",
               actual={"store": str(target.name), "shape": list(target.shape)})
    domain = _bind_domains(list(grid_loops) + list(nest))
    _prove_tile_coverage(domain, indices=list(stored.indices), nest=list(nest),
                         extents=[tile_m, tile_n], role="the output store")
    guard_limits = _consume_conditions(
        conditions, indices=list(store.indices),
        limits=[logical_m, logical_n], analyzer=domain, role="the output region")
    variables, blocks = _prove_tile_index(
        domain, reads=list(store.indices), locals=list(stored.indices),
        extents=[tile_m, tile_n], loops=list(grid_loops),
        message="the output store index is not tile local",
        role="the output store")

    def owner(variable: Any) -> Any:
        for loop in grid_loops:
            if loop.loop_var.same_as(variable):
                return loop
        return None

    for axis in (0, 1):
        if variables[axis] is not None:
            continue
        free = [loop for loop in grid_loops
                if not any(variables[other] is not None and
                           variables[other].same_as(loop.loop_var)
                           for other in (0, 1))]
        if not free or any(int(loop.extent) != 1 for loop in free):
            reject(RK_E_SHAPE,
                   "the output block coordinate cannot be folded while the axis "
                   "has more than one block", stage="pass",
                   actual={"axis": axis,
                           "blocks": [int(loop.extent) for loop in grid_loops]})
        variables[axis] = free[axis if len(free) > 1 else 0].loop_var
    if variables[0].same_as(variables[1]):
        reject(RK_E_SHAPE,
               "both output axes are indexed by one block loop, so another grid "
               "loop would write the same block again", stage="pass",
               actual={"loop": str(variables[0]),
                       "blocks": [int(loop.extent) for loop in grid_loops]})
    for axis in (0, 1):
        loop = owner(variables[axis])
        blocks_axis = int(loop.extent)
        tile = tile_m if axis == 0 else tile_n
        logical = logical_m if axis == 0 else logical_n
        limit = guard_limits.get(axis)
        if limit is not None and limit != logical:
            reject(RK_E_SHAPE,
                   "the output guard bound is not the logical extent",
                   stage="pass", actual={"axis": axis, "bound": limit,
                                         "logical": logical})
        if limit is None and blocks_axis * tile != logical:
            reject(RK_E_SHAPE,
                   "an unguarded epilogue axis needs a grid that covers it "
                   "exactly, and this one does not", stage="pass",
                   actual={"axis": axis, "blocks": blocks_axis, "tile": tile,
                           "logical": logical})
        if limit is not None and blocks_axis * tile < logical:
            reject(RK_E_SHAPE,
                   "the epilogue grid does not cover the output axis",
                   stage="pass",
                   actual={"axis": axis, "blocks": blocks_axis, "tile": tile,
                           "logical": logical})
        index = store.indices[axis]
        inside = tir.And(tir.LE(0, index), tir.LE(tir.Add(index, 1), logical))
        if not _prove_under(domain, conditions, inside):
            reject(RK_E_SHAPE,
                   "the output store is not provably inside the output over the "
                   "enclosing domain", stage="pass",
                   actual={"axis": axis, "index": str(index), "limit": logical})
    return {"name": str(target.name), "rows": blocks[0], "cols": blocks[1],
            "variables": (variables[0], variables[1]), "nest": tuple(nest),
            "store": store, "clear": clear_statement,
            "proven": (clear_statement, store)}



def _rewrite_tile_ops(prim_func: Any, k_loop: Any, block_loops: tuple,
                      running: Any, gemm_acc: Any, tile_m: int, tile_n: int,
                      block_k: int, params: tuple, output: dict,
                      operand_copies: list[dict[str, Any]],
                      logical_m: int, logical_n: int,
                      *, final_only: bool = False) -> Any:
    """Rewrite every tile-op call into an RKNPU extern call, in place.

    The arguments are the proved facts: the operand and accumulator buffer names,
    the block coordinate of each axis and the tile/K offsets, all as expressions
    of the loop variables the proof identified.  A coordinate that folded to zero
    is emitted as the constant the plan will use, never as a default grid.
    """
    tir = _tir()
    import tvm.tirx.stmt_functor as stmt_functor

    def offset(variable: Any, extent: int) -> Any:
        if variable is None:
            return tir.IntImm("int32", 0)
        return tir.Mul(variable, tir.IntImm("int32", extent))

    def region_args(region: dict) -> tuple[Any, ...]:
        offsets = list(region["offsets"])
        extents = list(region["extents"])
        if len(offsets) != 2 or len(extents) != 2:
            reject(RK_E_SHAPE,
                   "a localised copy region is not rank 2", stage="pass",
                   actual={"buffer": str(region["buffer"].name),
                           "offsets": [str(value) for value in offsets],
                           "extents": extents})
        return (offsets[0], offsets[1], tir.IntImm("int32", extents[0]),
                tir.IntImm("int32", extents[1]))

    def post(node):
        # The epilogue store loop is a target task too.  Replace the whole
        # tile-local loop nest with one store operation; replacing only the
        # BufferStore would turn the iteration bound into a task count and
        # duplicate publication for every element.
        if type(node).__name__ == "For":
            if output.get("nest") and node.same_as(output["nest"][0]):
                m0 = offset(block_loops[0], tile_m)
                n0 = offset(block_loops[1], tile_n)
                rows = tir.min(tir.IntImm("int32", tile_m),
                               tir.Sub(tir.IntImm("int32", logical_m), m0))
                cols = tir.min(tir.IntImm("int32", tile_n),
                               tir.Sub(tir.IntImm("int32", logical_n), n0))
                new = tir.call_extern(
                    "void", RK_STORE, _string(str(running.name)),
                    _string(output["name"]), m0, n0, rows, cols)
                return tir.Evaluate(new)
            for copy in operand_copies:
                if not node.same_as(copy["loop"]):
                    continue
                src = copy["source_region"]
                dst = copy["destination_region"]
                def as_extent(value: Any) -> Any:
                    return (tir.IntImm("int32", value)
                            if type(value) is int else value)

                new = tir.call_extern(
                    "void", RK_COPY, _string(copy["source"]),
                    _string(copy["destination"]),
                    src["row"], src["col"],
                    as_extent(src["rows"]), as_extent(src["cols"]),
                    dst["row"], dst["col"],
                    as_extent(dst["rows"]), as_extent(dst["cols"]))
                return tir.Evaluate(new)
        if type(node).__name__ != "Evaluate":
            return None
        call = node.value
        if not _op_name(call).startswith("tl.tileop."):
            return None
        role = _tile_op_role(call, where="the localised body")
        regions = _region_exprs(call)
        names = [str(region["buffer"].name) for region in regions]
        if role == "gemm":
            gemm = tir.call_extern(
                "void", RK_GEMM,
                _string(params[0]), _string(params[1]),
                _string(params[2]),
                _string(names[0]), _string(names[1]), _string(names[2]),
                offset(block_loops[0], tile_m), offset(block_loops[1], tile_n),
                tir.IntImm("int32", tile_m), tir.IntImm("int32", tile_n),
                offset(k_loop.loop_var, block_k),
                tir.IntImm("int32", block_k))
            if not final_only:
                new = gemm
            else:
                load = tir.call_extern(
                    "void", RK_COPY, _string(str(running.name)),
                    _string(str(gemm_acc.name)),
                    *region_args(regions[2]), *region_args(regions[2]))
                writeback = tir.call_extern(
                    "void", RK_COPY, _string(str(gemm_acc.name)),
                    _string(str(running.name)),
                    *region_args(regions[2]), *region_args(regions[2]))
                new = tir.SeqStmt([tir.Evaluate(load), tir.Evaluate(gemm),
                                   tir.Evaluate(writeback)])
        elif role == "fill":
            value = _const(call.args[1]) if len(call.args) > 1 else 0
            target = (str(running.name)
                      if final_only and names[0] == str(gemm_acc.name)
                      else names[0])
            new = tir.call_extern("void", RK_FILL, _string(target),
                                  tir.IntImm("int32", int(value or 0)),
                                  *region_args(regions[0]))
        else:
            src, dst = regions
            new = tir.call_extern("void", RK_COPY,
                                  _string(names[0]), _string(names[1]),
                                  *region_args(src), *region_args(dst))
        if type(new).__name__ == "SeqStmt":
            return new
        return tir.Evaluate(new)

    body = stmt_functor.ir_transform(prim_func.body, None, post)
    if final_only:
        root = None
        for _, node in _walk(prim_func.body):
            if type(node).__name__ == "SBlock" and \
                    list(getattr(node, "alloc_buffers", ()) or ()):
                root = node
                break
        if root is None:
            reject(RK_E_SHAPE,
                   "final-only lowering cannot find the allocation block",
                   stage="pass")

        def add_running(node):
            if not node.same_as(root):
                return None
            allocations = list(getattr(node, "alloc_buffers", ()) or ())
            allocations.append(running)
            return tir.SBlock(
                iter_vars=list(getattr(node, "iter_vars", ()) or ()),
                reads=list(getattr(node, "reads", ()) or ()),
                writes=list(getattr(node, "writes", ()) or ()),
                name_hint=str(getattr(node, "name_hint", "tilelang_root")),
                body=getattr(node, "body"), init=getattr(node, "init", None),
                alloc_buffers=allocations,
                match_buffers=list(getattr(node, "match_buffers", ()) or ()),
                annotations=dict(getattr(node, "annotations", None) or {}),
                span=getattr(node, "span", None))

        body = stmt_functor.ir_transform(body, None, add_running)
    return tir.PrimFunc(prim_func.params, body, prim_func.ret_type,
                        prim_func.buffer_map, prim_func.attrs)



# --------------------------------------------------------------- entry
def _minimal_module(facts: LocalFacts, *, name: str) -> Any:
    """RKIR module carrying only binding identity for the proven geometry."""
    from .. import ir

    contract = ir.NumericContract(
        accumulator_dtype="float16",
        accumulation="ordered_k_tiles_with_f16_writeback",
        narrowing_points=("k_tile_boundary",), approximate=False,
        tolerance={"atol": 8e-3, "rtol": 8e-3})
    a = ir.Value(0, facts.operand_a, ir.TensorType(
        ir.DTYPE_F16, (facts.logical_m, facts.logical_k)), None, "input")
    b = ir.Value(1, facts.operand_b, ir.TensorType(
        ir.DTYPE_F16, (facts.logical_k, facts.logical_n)), None, facts.b_role)
    matmul_out = 3 if facts.activation is not None else 2
    out_name = facts.output_name if facts.activation is None else "Z"
    out = ir.Value(2, out_name, ir.TensorType(
        ir.DTYPE_F16, (facts.logical_m, facts.logical_n)),
        0, "output" if facts.activation is None else "temporary")
    op = ir.Op(0, "matmul", (0, 1), (2,),
               {"b_layout": "kn",
                "numeric_contract": contract.to_dict(),
                "source_schedule": {"kind": "mn_tiles_v1",
                                    "tile_m": facts.tile_m,
                                    "tile_n": facts.tile_n,
                                    "order": facts.order,
                                    "k_tile": facts.block_k,
                                    "accumulation":
                                        "ordered_ascending_f16_writeback"},
                "source_schedule_origin": "pass-instantiated",
                "source_k_rounding": facts.rounding,
                "source_k_loop": {"logical_k": facts.logical_k,
                                  "rounding_points": facts.rounding_points}},
               "rknpu-pass:localise")
    values = [a, b, out]
    ops = [op]
    outputs = (2,)
    if facts.activation is not None:
        final = ir.Value(3, facts.output_name, ir.TensorType(
            ir.DTYPE_F16, (facts.logical_m, facts.logical_n)), 1, "output")
        values.append(final)
        ops.append(ir.Op(1, facts.activation, (2,), (3,), {},
                         "rknpu-pass:localise"))
        outputs = (3,)
    return ir.Module(
        name=name, values=tuple(values), ops=tuple(ops),
        inputs=((0, 1) if facts.b_role == "input" else (0,)),
        outputs=outputs, source_hash=facts.source_hash)


def _compile_guarded_tail(prim_func: Any, *, profile: Any = None,
                          options: Any = None, simplify: bool = False,
                          name: str = "rknpu-pass",
                          pin_baseline: bool = True,
                          honor_grid_order: bool = False,
                          b_is_input: bool = False):
    """Shared lowering body; returns compilation, stages, facts and tasks."""
    from ..capabilities import default_profile
    from ..codegen.pipeline import compile_from_tasks
    from ..options import CompileOptions

    profile = profile or default_profile()
    options = options or CompileOptions(numeric_mode="rk_f16_tiled",
                                       allow_approx_lut=False)
    pinned_policy = None
    if pin_baseline and options.linear_task_order == "baseline":
        # Preserve the historical direct/registered default hash while the
        # formal entries pass their actual policy through to task assembly.
        import dataclasses

        pinned_policy = options.linear_task_order
        options = dataclasses.replace(
            options, linear_task_order="scheduled_baseline")
    # Check original fields before upstream transforms can erase their source
    # representation. Keep this list as provenance; pass_localise validates the
    # original list and the normalized list independently against proven facts.
    _validate_container_semantics(prim_func)
    source_declarations = _rknpu_declarations(prim_func)
    source_hints = _unapplied_annotation_hints(prim_func)
    normalised, stages = normalise(prim_func, simplify=simplify)
    localised, facts, localise_stage = pass_localise(
        normalised, source_declarations=source_declarations,
        source_hints=source_hints)
    if b_is_input:
        import dataclasses

        facts = dataclasses.replace(facts, b_role="input")
    stages.append(localise_stage)
    local_nodes = len(list(_walk(localised.body)))
    if local_nodes > options.max_ir_nodes:
        reject(RK_E_BUDGET, "max_ir_nodes exceeded by the localised IR",
               stage="pass", actual=local_nodes,
               required=options.max_ir_nodes)
    if options.numeric_mode != "rk_f16_tiled":
        reject(RK_E_UNSUPPORTED,
               "the pass chain only proves the rk_f16_tiled contract",
               stage="pass", actual=options.numeric_mode)
    instantiated, tasks, instantiate_stage = pass_instantiate(
        localised, facts, max_tasks=options.max_actions,
        honor_grid_order=honor_grid_order)
    stages.append(instantiate_stage)
    generated = [task for task in tasks if task["kind"] == "gemm"]
    if not generated:
        reject(RK_E_SHAPE,
               "the instantiated task sequence has no GEMM", stage="pass")
    if pinned_policy is not None:
        facts.hints["pinned_linear_task_order"] = {
            "requested": pinned_policy, "applied": "scheduled_baseline"}
    # The target task sequence is the execution authority.  The synthetic
    # matmul module below carries binding identity and the NumericContract only;
    # it is not used to regenerate tiles, K slices, task order or publication.
    facts_dict = {"m": facts.logical_m, "k": facts.logical_k,
                  "n": facts.logical_n, "tile_m": facts.tile_m,
                  "tile_n": facts.tile_n, "block_k": facts.block_k,
                  "order": facts.order, "rounding": facts.rounding,
                  "hints": dict(facts.hints), "guarded_tail": True,
                  "schedule_declared": bool(
                      honor_grid_order and
                      facts.hints.get("rknpu_declarations", {})
                      .get("source")),
                  "formal_guarded_entry": bool(honor_grid_order)}
    module = _minimal_module(facts, name=name)
    compilation = compile_from_tasks(module, tasks, facts_dict,
                                     profile=profile, options=options)
    first_task = next(action for action in compilation.plan.actions
                      if action.kind == "npu_task")
    compilation.report["source"] = {
        "module": name, "source_hash": facts.source_hash,
        "numeric_mode": options.numeric_mode,
        "source_k_rounding": facts.rounding,
        "source_schedule_origin": (
            "annotated" if source_declarations else None),
        "mn_grid_source": ("annotation" if source_declarations
                           else "default_grid"),
        "source_k_loop": dict(first_task.attrs.get("source_k_loop") or {}),
        "source_mn_tiling": dict(
            first_task.attrs.get("source_mn_tiling") or {}),
        "lowering": "guarded-tail localise -> target TIR -> task consumer",
    }
    stages.append(Stage("plan", "analysis", _digest(
        compilation.plan.plan_hash), {"plan_hash": compilation.plan.plan_hash,
                                      "tasks": len(generated)}))
    return compilation, stages, facts, tasks


def compile_guarded_tail(prim_func: Any, *, profile: Any = None,
                         options: Any = None, simplify: bool = False,
                         name: str = "rknpu-pass", b_is_input: bool = False):
    """Direct/registered entry; returns (plan, stages, facts, tasks)."""
    compilation, stages, facts, tasks = _compile_guarded_tail(
        prim_func, profile=profile, options=options, simplify=simplify,
        name=name, pin_baseline=True, honor_grid_order=False,
        b_is_input=b_is_input)
    return compilation.plan, stages, facts, tasks


def compile_guarded_tail_compilation(prim_func: Any, *, profile: Any = None,
                                     options: Any = None,
                                     simplify: bool = False,
                                     name: str = "rknpu-guarded-task",
                                     b_is_input: bool = False):
    """Formal compile-entry wrapper preserving the source order policy."""
    compilation, _, _, _ = _compile_guarded_tail(
        prim_func, profile=profile, options=options, simplify=simplify,
        name=name, pin_baseline=False, honor_grid_order=True,
        b_is_input=b_is_input)
    return compilation


def _plain(value: Any) -> Any:
    """An annotation value as plain python data."""
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, (int, float, str, bool)):
        return value
    try:
        items = list(value)
    except TypeError:
        return str(value)
    return [_plain(item) for item in items]


def _rknpu_annotations(node: Any) -> dict:
    found = {}
    annotations = getattr(node, "annotations", None)
    if not annotations:
        return found
    for key in annotations:
        text = str(key)
        if text.startswith("rknpu."):
            found[text] = _plain(annotations[key])
    return found


def _declaration_value(value: Any) -> Any:
    """Resolve call-free constant expressions without erasing their source text."""
    constant = _known_const(value)
    if constant is not None:
        return constant
    if type(value).__name__ == "StringImm":
        return str(value.value)
    if type(value).__name__ == "String":
        return str(value)
    return _plain(value)


def _declaration_scope(node: Any) -> str:
    kind = type(node).__name__
    if kind == "SBlock":
        return kind + ":" + str(getattr(node, "name_hint", ""))
    if kind == "For":
        return kind + ":" + str(node.loop_var)
    if kind == "Call":
        return kind + ":" + _op_name(node)
    return kind


def _rknpu_declarations(prim_func: Any) -> list[dict]:
    """Collect every explicit request with its source expression and scope."""
    declarations = []
    for ordinal, (depth, node) in enumerate(_walk(prim_func.body)):
        annotations = getattr(node, "annotations", None) or {}
        for key in annotations:
            name = str(key)
            if name.startswith("rknpu."):
                raw_value = annotations[key]
                declarations.append({"key": name,
                                     "value": _declaration_value(raw_value),
                                     "expression": str(raw_value),
                                     "scope": _declaration_scope(node),
                                     "depth": depth, "ordinal": ordinal})
    for key, raw_value in (getattr(prim_func, "attrs", None) or {}).items():
        name = str(key)
        if name.startswith("rknpu."):
            declarations.append({"key": name,
                                 "value": _declaration_value(raw_value),
                                 "expression": str(raw_value),
                                 "scope": "PrimFunc.attrs",
                                 "depth": -1, "ordinal": -1})
    return declarations


def _unapplied_annotation_hints(prim_func: Any) -> list[dict]:
    """Keep every accepted pure GPU hint with a scope-specific reason."""
    records = []
    reasons = {
        "tl.launch_threads":
            "an RKNPU plan has no GPU thread extent",
        "num_stages":
            "an RKNPU plan has no software pipeline; K tasks remain ordered",
        "coalesced_width":
            "an RKNPU copy has no GPU vector coalescing width",
    }
    for ordinal, (depth, node) in enumerate(_walk(prim_func.body)):
        annotations = getattr(node, "annotations", None) or {}
        for key in annotations:
            name = str(key)
            if name not in reasons:
                continue
            records.append({"key": name,
                            "requested": _declaration_value(annotations[key]),
                            "scope": _declaration_scope(node),
                            "depth": depth, "ordinal": ordinal,
                            "applied": False, "reason": reasons[name]})
    return records


def _string(value: str) -> Any:
    tir = _tir()

    return tir.StringImm(value)


def _name_of(arg: Any) -> str:
    return str(getattr(arg, "value", arg))


def pass_instantiate(prim_func: Any, facts: LocalFacts, *,
                     max_tasks: int | None = None,
                     honor_grid_order: bool = False) -> tuple[Any, list[dict],
                                                              Stage]:
    """Expand the grid and K loops into an explicit RKNPU task sequence.

    Guards are decided on the iteration domain: one that holds for every instance
    is dropped, one that fails for every instance removes the statement, and one
    that cannot be decided rejects the program.  The body becomes a single
    SeqStmt of RKNPU calls in program order, and that sequence drives the plan.
    """
    tir = _tir()
    container_contract = _validate_container_semantics(prim_func)
    current_declarations = _rknpu_declarations(prim_func)
    _validate_declarations(current_declarations, tile_m=facts.tile_m,
                           tile_n=facts.tile_n, block_k=facts.block_k,
                           order=facts.order,
                           source_rounding=facts.rounding,
                           stage="instantiate")
    analyzer = _arith()
    import tvm.tirx.stmt_functor as stmt_functor

    emitted: list[Any] = []
    drained = tuple(getattr(facts, "drained", ()))

    grid_pair = None
    if honor_grid_order:
        grid0, grid1, _ = _grid_and_k_loops(prim_func)
        gemm_call = None
        for _, node in _walk(prim_func.body):
            if type(node).__name__ == "Evaluate" and \
                    _op_name(node.value).endswith("call_extern") and \
                    len(node.value.args) >= 10 and \
                    _name_of(node.value.args[0]) == RK_GEMM:
                gemm_call = node.value
                break
        if gemm_call is None:
            reject(RK_E_SHAPE,
                   "ordered guarded-tail instantiation needs a GEMM task",
                   stage="pass")
        m_vars = set(_vars_in(gemm_call.args[7]))
        n_vars = set(_vars_in(gemm_call.args[8]))
        if grid0.loop_var in m_vars and grid1.loop_var in n_vars:
            m_loop, n_loop = grid0, grid1
        elif grid0.loop_var in n_vars and grid1.loop_var in m_vars:
            m_loop, n_loop = grid1, grid0
        else:
            reject(RK_E_SHAPE,
                   "cannot map the guard-tail grid loops to M/N", stage="pass")
        grid_pair = (grid0, grid1, m_loop, n_loop)

    def substitute(expr: Any, env: dict) -> Any:
        if not env:
            return expr
        return stmt_functor.substitute(expr, env)

    def expand(node: Any, env: dict) -> None:
        kind = type(node).__name__
        if kind == "For":
            if grid_pair is not None and node.same_as(grid_pair[0]):
                grid0, grid1, m_loop, n_loop = grid_pair
                m_extent = _loop_domain(m_loop, role="M grid")
                n_extent = _loop_domain(n_loop, role="N grid")
                if facts.order == "m_major":
                    values = ((m, n) for m in range(m_extent)
                              for n in range(n_extent))
                else:
                    values = ((m, n) for n in range(n_extent)
                              for m in range(m_extent))
                for m_value, n_value in values:
                    inner = dict(env)
                    inner[m_loop.loop_var] = tir.IntImm("int32", m_value)
                    inner[n_loop.loop_var] = tir.IntImm("int32", n_value)
                    expand(grid1.body, inner)
                return
            extent = _loop_domain(node, role="instantiation")
            for value in range(extent):
                inner = dict(env)
                inner[node.loop_var] = tir.IntImm("int32", value)
                expand(node.body, inner)
            return
        if kind == "IfThenElse":
            condition = substitute(node.condition, env)
            if analyzer.can_prove(condition):
                expand(node.then_case, env)
                return
            if analyzer.can_prove(tir.Not(condition)):
                if node.else_case is not None:
                    expand(node.else_case, env)
                return
            reject(RK_E_UNSUPPORTED,
                   "a guard cannot be decided on the iteration domain",
                   stage="pass", actual=str(condition))
        if kind == "SeqStmt":
            for item in node.seq:
                expand(item, env)
            return
        if kind == "SBlockRealize":
            predicate = substitute(node.predicate, env)
            if not analyzer.can_prove(predicate):
                reject(RK_E_UNSUPPORTED,
                       "an SBlockRealize predicate is not true for this "
                       "instantiated point", stage="pass",
                       actual=str(predicate))
            expand(node.block, env)
            return
        if kind == "SBlock":
            body = getattr(node, "body", None)
            if body is not None:
                expand(body, env)
            return
        if kind in ("AttrStmt", "DeclBuffer", "Allocate", "AllocateConst",
                    "Bind"):
            body = getattr(node, "body", None)
            if body is not None:
                expand(body, env)
            return
        if kind == "Evaluate":
            call = node.value
            if _op_name(call).endswith("call_extern"):
                name = _name_of(call.args[0])
                args = [analyzer.simplify(substitute(argument, env))
                        for argument in call.args[1:]]
                if name in (RK_GEMM, RK_COPY, RK_FILL, RK_STORE, RK_SILU):
                    emitted.append(tir.Evaluate(tir.call_extern("void", name,
                                                                *args)))
                    if max_tasks is not None and len(emitted) > max_tasks:
                        reject(RK_E_BUDGET,
                               "the instantiated task count exceeds "
                               "the budget", stage="pass",
                               actual=len(emitted),
                               required=max_tasks)
                    return
        if kind == "BufferStore":
            # A store is consumed only when it is the statement pass_localise
            # proved -- the two guarded operand copies and the single output
            # write, matched by node identity.  Those do not become tasks of
            # their own: the GEMM task's source slice was proved equal to what
            # the copy stages, and the output block is the tile instance.  Any
            # other store is a state change this path never proved, so it is
            # refused rather than dropped for sharing a buffer name.
            if not any(node.same_as(other) for other in drained):
                reject(RK_E_UNSUPPORTED,
                       "a store in the lowered program was not proved by "
                       "pass_localise, so dropping it would change the result",
                       stage="pass", actual=str(node.buffer.name))
            return
        reject(RK_E_UNSUPPORTED,
               "a statement kind is not consumed by instantiation",
               stage="pass", actual=kind)

    expand(prim_func.body, {})
    if not emitted:
        reject(RK_E_SHAPE, "the instantiated program is empty", stage="pass")
    body = tir.SeqStmt(emitted) if len(emitted) > 1 else emitted[0]
    instantiated = tir.PrimFunc(prim_func.params, body, prim_func.ret_type,
                                prim_func.buffer_map, prim_func.attrs)
    return instantiated, tasks_from_module(instantiated), Stage(
        "instantiate", "transform", _digest(instantiated.script()),
        {"tasks": len(emitted), "container_contract": container_contract})


def tasks_from_module(prim_func: Any) -> list[dict]:
    """Read the instantiated RKNPU task sequence back out of the IR."""
    tasks = []
    for statement in _statements(prim_func.body):
        if type(statement).__name__ != "Evaluate":
            continue
        call = statement.value
        if not _op_name(call).endswith("call_extern"):
            continue
        name = _name_of(call.args[0])
        args = list(call.args[1:])
        if name == RK_GEMM:
            tasks.append({"kind": "gemm", "ordinal": len(tasks),
                          "operands": [_name_of(args[0]), _name_of(args[1]),
                                       _name_of(args[2])],
                          "tiles": [_name_of(args[3]), _name_of(args[4]),
                                    _name_of(args[5])],
                          "m0": _const(args[6]), "n0": _const(args[7]),
                          "rows": _const(args[8]), "cols": _const(args[9]),
                          "k0": _const(args[10]), "kt": _const(args[11])})
        elif name == RK_COPY:
            if len(args) != 10:
                reject(RK_E_UNSUPPORTED,
                       "a localised copy task must carry both rank-2 regions",
                       stage="pass", actual=len(args))
            tasks.append({
                "kind": "copy", "ordinal": len(tasks),
                "operands": [_name_of(args[0]), _name_of(args[1])],
                "source": _name_of(args[0]),
                "destination": _name_of(args[1]),
                "source_region": {"row": _const(args[2]),
                                  "col": _const(args[3]),
                                  "rows": _const(args[4]),
                                  "cols": _const(args[5])},
                "destination_region": {"row": _const(args[6]),
                                       "col": _const(args[7]),
                                       "rows": _const(args[8]),
                                       "cols": _const(args[9])},
            })
        elif name == RK_FILL:
            if len(args) != 6:
                reject(RK_E_UNSUPPORTED,
                       "a localised fill task must carry its rank-2 region",
                       stage="pass", actual=len(args))
            tasks.append({
                "kind": "fill", "ordinal": len(tasks),
                "operands": [_name_of(args[0])],
                "destination": _name_of(args[0]),
                "value": _const(args[1]),
                "region": {"row": _const(args[2]), "col": _const(args[3]),
                           "rows": _const(args[4]), "cols": _const(args[5])},
            })
        elif name == RK_STORE:
            if len(args) != 6:
                reject(RK_E_UNSUPPORTED,
                       "a localised store task must carry its rectangle",
                       stage="pass", actual=len(args))
            tasks.append({
                "kind": "store", "ordinal": len(tasks),
                "operands": [_name_of(args[0]), _name_of(args[1])],
                "source": _name_of(args[0]),
                "destination": _name_of(args[1]),
                "source_region": {"row": 0, "col": 0,
                                  "rows": _const(args[4]), "cols": _const(args[5])},
                "destination_region": {"row": _const(args[2]), "col": _const(args[3]),
                                       "rows": _const(args[4]), "cols": _const(args[5])},
                "m0": _const(args[2]), "n0": _const(args[3]),
                "rows": _const(args[4]), "cols": _const(args[5]),
            })
        elif name == RK_SILU:
            if len(args) != 3:
                reject(RK_E_UNSUPPORTED,
                       "a localised SiLU task must carry source and rectangle",
                       stage="pass", actual=len(args))
            tasks.append({
                "kind": "silu", "ordinal": len(tasks),
                "operands": [_name_of(args[0])],
                "source": _name_of(args[0]),
                "rows": _const(args[1]), "cols": _const(args[2]),
            })
    return tasks
