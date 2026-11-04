# SPDX-License-Identifier: GPL-3.0-or-later
"""Two-stage source-operation rewrite for closed full-logical GEMMs.

Stage 1 rewrites the actual TileLang fill/copy/GEMM/[SiLU store]/output-copy
nodes into a tile/K schedule while keeping them as TileLang operations.  Stage 2
visits those scheduled operations individually and lowers them to the existing
``rk_*`` representation, inserting the target accumulator, padding and writeback
where the target policy requires them.  A SiLU BufferStore is matched locally and
lowered to ``rk_silu``; it is not a switch for rebuilding a kernel template.

The surrounding PrimFunc, launch, SBlock, parameter declarations and original
tile buffers stay in place.  Parent loops/SeqStmts are rebuilt only as needed to
hold the rewritten child operations.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from ..errors import (RK_E_NUMERIC_CONTRACT, RK_E_SHAPE, RK_E_UNSUPPORTED,
                      reject)
from ..options import CompileOptions
from .rknpu_pass import (LocalFacts, RK_COPY, RK_FILL, RK_GEMM, RK_SILU,
                         RK_STORE, Stage, _arith, _const, _digest, _op_name,
                         _known_const, _region_exprs, _same_buffer, _scope_of,
                         _structurally_equal, _string, _tile_op_role, _vars_in,
                         _validate_call_positions, _walk, pass_instantiate)


@dataclass(frozen=True)
class ScheduledFullLogical:
    """Stage 1 result: TileLang operations after tiling/access rewriting."""

    source: Any
    scheduled: Any
    facts: LocalFacts
    stage: Stage


@dataclass(frozen=True)
class LocatedFullLogical:
    """Stage-1 location result, before any scheduling transformation."""

    root: Any
    statements: tuple[Any, ...]
    gemm_index: int
    gemm: Any
    reads: dict[int, tuple[Any, ...]]
    writes: dict[int, tuple[Any, ...]]
    roles: dict[str, Any]
    store: Any | None
    store_expression: Any | None
    store_statement_index: int | None
    store_loops: tuple[Any, ...]


@dataclass(frozen=True)
class FullLogicalRewrite:
    """Two conversion stages plus the existing task-instantiation result."""

    source: Any
    scheduled: Any
    target: Any
    instantiated: Any
    facts: LocalFacts
    tasks: list[dict[str, Any]]
    stages: tuple[Stage, Stage, Stage]


def _tir() -> Any:
    from .rknpu_pass import _tir as tir
    return tir()


def _imm(value: int) -> Any:
    return _tir().IntImm("int32", int(value))


def _add(base: Any, delta: Any) -> Any:
    return _arith().simplify(_tir().Add(base, delta))


def _extent(value: Any) -> Any:
    return value if type(value).__name__ != "int" else _imm(value)


def _predicate_true(node: Any) -> bool:
    if node is None or node is True:
        return True
    if type(node).__name__ != "IntImm":
        return False
    try:
        return str(node.dtype) == "bool" and bool(node.value)
    except Exception:
        return False


def _validate_region_contract(nodes: dict[str, Any]) -> None:
    """Preserve the old source boundaries at the actual-tile-op boundary."""
    gemm = nodes["gemm"].value
    region_calls = [arg for arg in gemm.args
                    if type(arg).__name__ == "Call" and
                    _op_name(arg) == "tl.region"]
    if len(region_calls) != 3:
        reject(RK_E_UNSUPPORTED,
               "GEMM must carry three region descriptors", stage="rewrite")
    masks = [_const(call.args[1]) for call in region_calls]
    if masks[:2] != [1, 1]:
        reject(RK_E_UNSUPPORTED,
               "GEMM A/B regions must be loads", stage="rewrite",
               actual=masks)
    if masks[2] != 3:
        reject(RK_E_UNSUPPORTED,
               "the GEMM accumulator pattern requires a load+store",
               stage="rewrite", actual=masks[2])
    for node in list(nodes["copies"]) + [nodes["fill"], nodes["gemm"]]:
        for arg in node.value.args:
            if type(arg).__name__ != "Call" or _op_name(arg) != "tl.region":
                continue
            load = arg.args[0]
            if not _predicate_true(getattr(load, "predicate", None)):
                reject(RK_E_UNSUPPORTED,
                       "region load carries a predicate", stage="rewrite",
                       actual=str(getattr(load, "predicate", None)))


def _region_call(region_call: Any, *, offsets: list[Any],
                 extents: list[Any]) -> Any:
    """Rebuild one real ``tl.region`` call with new access coordinates."""
    tir = _tir()
    args = list(region_call.args)
    load = args[0]
    new_load = tir.BufferLoad(load.buffer, list(offsets))
    return tir.Call(region_call.dtype, region_call.op,
                    [new_load, args[1], _extent(extents[0]),
                     _extent(extents[1])],
                    dict(getattr(region_call, "annotations", None) or {}))


def _call_with_args(call: Any, args: list[Any]) -> Any:
    return _tir().Call(call.dtype, call.op, args,
                       dict(getattr(call, "annotations", None) or {}))


def _source_tile_nodes(prim_func: Any) -> dict[str, Any]:
    """Locate the actual nodes this one source form needs, by operation role."""
    gemms: list[Any] = []
    fills: list[Any] = []
    copies: list[Any] = []
    stores: list[Any] = []
    root = None
    for _, node in _walk(prim_func.body):
        if type(node).__name__ == "BufferStore":
            stores.append(node)
        if type(node).__name__ == "SBlock" and \
                list(getattr(node, "alloc_buffers", ()) or ()) and \
                any(type(inner).__name__ == "Evaluate" and
                    _op_name(inner.value).startswith("tl.tileop.")
                    for _, inner in _walk(node)):
            root = node
        if type(node).__name__ != "Evaluate":
            continue
        call = node.value
        if not _op_name(call).startswith("tl.tileop."):
            continue
        role = _tile_op_role(call, where="the source-operation rewrite")
        if role == "gemm":
            gemms.append(node)
        elif role == "fill":
            fills.append(node)
        else:
            copies.append(node)
    if root is None:
        reject(RK_E_SHAPE,
               "the source has no tile block containing a tile operation",
               stage="rewrite")
    if len(gemms) != 1 or len(fills) != 1 or len(copies) != 3:
        reject(RK_E_UNSUPPORTED,
               "the source-operation rewrite expects one GEMM, one fill and "
               "three copies", stage="rewrite",
               actual={"gemm": len(gemms), "fill": len(fills),
                       "copy": len(copies)})
    return {"root": root, "gemm": gemms[0], "fill": fills[0],
            "copies": copies, "stores": stores}


def _copy_regions(node: Any, *, keep_symbolic: bool = False
                  ) -> tuple[dict[str, Any], dict[str, Any]]:
    regions = _region_exprs(node.value, keep_symbolic_extents=keep_symbolic)
    if len(regions) != 2:
        reject(RK_E_SHAPE, "a source copy must have two regions",
               stage="rewrite", actual=len(regions))
    return regions[0], regions[1]


def _prove_silu_expression(store: Any, acc: Any) -> None:
    """Match only the concrete accumulator update, not a kernel-shape switch."""
    if not _predicate_true(getattr(store, "predicate", None)):
        reject(RK_E_UNSUPPORTED,
               "the SiLU store must be unconditional", stage="rewrite")
    indices = list(store.indices)
    if len(indices) != 2:
        reject(RK_E_UNSUPPORTED,
               "the SiLU store must be indexed by two tile loops",
               stage="rewrite", actual=len(indices))

    def load_ok(expr: Any) -> bool:
        return type(expr).__name__ == "BufferLoad" and \
            _same_buffer(expr.buffer, acc) and \
            _predicate_true(getattr(expr, "predicate", None)) and \
            len(expr.indices) == 2 and \
            all(_structurally_equal(left, right)
                for left, right in zip(expr.indices, indices))

    def sigmoid_ok(expr: Any) -> bool:
        if type(expr).__name__ != "Call":
            return False
        if not str(getattr(getattr(expr, "op", None), "name", "")).endswith(
                "sigmoid"):
            return False
        return len(expr.args) == 1 and load_ok(expr.args[0])

    value = store.value
    if type(value).__name__ != "Mul":
        reject(RK_E_UNSUPPORTED,
               "the only supported activation is acc * sigmoid(acc)",
               stage="rewrite", actual=str(value))
    factors = (value.a, value.b)
    if not ((load_ok(factors[0]) and sigmoid_ok(factors[1])) or
            (load_ok(factors[1]) and sigmoid_ok(factors[0]))):
        reject(RK_E_UNSUPPORTED,
               "the only supported activation is acc * sigmoid(acc)",
               stage="rewrite", actual=str(value))


def _classify_dataflow(nodes: dict[str, Any], *,
                       keep_symbolic: bool = False) -> dict[str, Any]:
    """Identify operand/output copies from the actual GEMM region identities."""
    gemm_regions = _region_exprs(
        nodes["gemm"].value, keep_symbolic_extents=keep_symbolic)
    if len(gemm_regions) != 3:
        reject(RK_E_SHAPE, "the source GEMM must carry three regions",
               stage="rewrite")
    a_stage, b_stage, gemm_acc = (region["buffer"]
                                  for region in gemm_regions)
    fill_regions = _region_exprs(
        nodes["fill"].value, keep_symbolic_extents=keep_symbolic)
    if len(fill_regions) != 1 or \
            not _same_buffer(fill_regions[0]["buffer"], gemm_acc):
        reject(RK_E_UNSUPPORTED,
               "the source fill must initialize the GEMM accumulator",
               stage="rewrite")
    stores = [store for store in (nodes.get("stores") or ())
              if _same_buffer(store.buffer, gemm_acc)]
    if len(stores) > 1:
        reject(RK_E_UNSUPPORTED,
               "the source has more than one accumulator update",
               stage="rewrite", actual=len(stores))
    store = stores[0] if stores else None
    if store is not None:
        _prove_silu_expression(store, gemm_acc)

    a_copy = b_copy = output_copy = None
    for node in nodes["copies"]:
        source, destination = _copy_regions(node,
                                             keep_symbolic=keep_symbolic)
        if _same_buffer(destination["buffer"], a_stage):
            a_copy = node
        elif _same_buffer(destination["buffer"], b_stage):
            b_copy = node
        elif _same_buffer(source["buffer"], gemm_acc):
            output_copy = node
    if a_copy is None or b_copy is None or output_copy is None:
        reject(RK_E_UNSUPPORTED,
               "the source copies do not connect A/B and the output",
               stage="rewrite")
    a_source, a_destination = _copy_regions(
        a_copy, keep_symbolic=keep_symbolic)
    b_source, b_destination = _copy_regions(
        b_copy, keep_symbolic=keep_symbolic)
    out_source, out_destination = _copy_regions(
        output_copy, keep_symbolic=keep_symbolic)
    if len({id(a_source["buffer"]), id(b_source["buffer"]),
            id(out_destination["buffer"])}) != 3:
        reject(RK_E_SHAPE,
               "A, B and the output must be three distinct parameter buffers",
               stage="rewrite")
    return {
        "a_copy": a_copy, "b_copy": b_copy, "output_copy": output_copy,
        "a_param": a_source["buffer"], "b_param": b_source["buffer"],
        "y_param": out_destination["buffer"],
        "a_stage": a_stage, "b_stage": b_stage, "gemm_acc": gemm_acc,
        "a_source_region": a_source,
        "a_destination_region": a_destination,
        "b_source_region": b_source,
        "b_destination_region": b_destination,
        "out_source_region": out_source,
        "out_destination_region": out_destination,
        "gemm_regions": gemm_regions,
        "store": store,
        "store_expression": None if store is None else store.value,
    }


def _validate_policy(prim_func: Any, options: Any) -> None:
    if options is None or not isinstance(options, CompileOptions):
        reject(RK_E_UNSUPPORTED, "the source rewrite needs CompileOptions",
               stage="rewrite")
    if options.numeric_mode != "rk_f16_tiled":
        reject(RK_E_NUMERIC_CONTRACT,
               "the source rewrite only supports the rk_f16_tiled policy",
               stage="rewrite", actual=options.numeric_mode)
    for _, node in _walk(prim_func.body):
        kind = type(node).__name__
        if kind == "IfThenElse":
            reject(RK_E_UNSUPPORTED,
                   "the full-logical task scheduler does not consume "
                   "conditional execution", stage="frontend")
        if kind == "SBlockRealize" and not _predicate_true(node.predicate):
            reject(RK_E_UNSUPPORTED,
                   "block predicate is not provably true", stage="frontend",
                   actual=str(node.predicate))
        if kind == "SBlockRealize" and list(
                getattr(node, "iter_values", ()) or ()):
            reject(RK_E_UNSUPPORTED,
                   "block iteration bindings have no target equivalent",
                   stage="frontend")
        if kind == "SBlock":
            if list(getattr(node, "iter_vars", ()) or ()):
                reject(RK_E_UNSUPPORTED,
                       "block iteration variables have no target equivalent",
                       stage="frontend")
            if getattr(node, "init", None) is not None or list(
                    getattr(node, "match_buffers", ()) or ()):
                reject(RK_E_UNSUPPORTED,
                       "block init/alias effects have no target consumer",
                       stage="frontend")
    _validate_call_positions(prim_func)


def _source_geometry(data: dict[str, Any]) -> tuple[int, int, int]:
    a_region = data["a_source_region"]
    b_region = data["b_source_region"]
    c_region = data["out_source_region"]
    gm, gk = (int(extent) for extent in a_region["extents"])
    b_gk, gn = (int(extent) for extent in b_region["extents"])
    c_gm, c_gn = (int(extent) for extent in c_region["extents"])
    if (gm, gk) != (c_gm, b_gk) or (b_gk, gn) != (gk, c_gn):
        reject(RK_E_SHAPE,
               "GEMM/copy regions disagree on the logical M/N/K geometry",
               stage="rewrite",
               actual={"a": [gm, gk], "b": [b_gk, gn], "c": [c_gm, c_gn]})
    return gm, gk, gn


def _ordered_statements(body: Any) -> tuple[Any, ...]:
    return tuple(getattr(body, "seq", ())) if \
        type(body).__name__ == "SeqStmt" else (body,)


def _region_masks(call: Any) -> list[int]:
    masks = []
    tir = _tir()
    for arg in call.args:
        if isinstance(arg, tir.Call) and _op_name(arg) == "tl.region":
            masks.append(_const(arg.args[1]))
    return masks


def _statement_effects(node: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Read/write buffer relations used by the movement proof."""
    stores = [inner for _, inner in _walk(node)
              if type(inner).__name__ == "BufferStore"]
    if stores:
        reads = []
        writes = []
        for store in stores:
            writes.append(store.buffer)
            for _, inner in _walk(store.value):
                if type(inner).__name__ == "BufferLoad":
                    reads.append(inner.buffer)
        return tuple(reads), tuple(writes)
    if type(node).__name__ != "Evaluate":
        return (), ()
    call = node.value
    if not _op_name(call).startswith("tl.tileop."):
        return (), ()
    role = _tile_op_role(call, where="the source movement proof")
    regions = _region_exprs(call)
    if role == "fill":
        return (), (regions[0]["buffer"],)
    if role == "copy":
        return (regions[0]["buffer"],), (regions[1]["buffer"],)
    masks = _region_masks(call)
    reads = tuple(region["buffer"] for region, mask in zip(regions, masks)
                  if mask is not None and mask & 1)
    writes = tuple(region["buffer"] for region, mask in zip(regions, masks)
                   if mask is not None and mask & 2)
    return reads, writes


def locate_full_logical_source(prim_func: Any) -> LocatedFullLogical:
    """Return actual ordered statements, GEMM anchor and read/write relations."""
    nodes = _source_tile_nodes(prim_func)
    statements = _ordered_statements(nodes["root"].body)
    gemm_index = next((index for index, statement in enumerate(statements)
                       if statement.same_as(nodes["gemm"])), None)
    if gemm_index is None:
        reject(RK_E_SHAPE,
               "the actual GEMM node is not in the root statement sequence",
               stage="rewrite")
    reads: dict[int, tuple[Any, ...]] = {}
    writes: dict[int, tuple[Any, ...]] = {}
    for index, statement in enumerate(statements):
        reads[index], writes[index] = _statement_effects(statement)
    stores = [store for store in (nodes.get("stores") or ())]
    store = stores[0] if stores else None
    store_index = None
    store_loops: tuple[Any, ...] = ()
    if store is not None:
        for index, statement in enumerate(statements):
            if any(inner.same_as(store) for _, inner in _walk(statement)):
                store_index = index
                store_loops = tuple(
                    inner for _, inner in _walk(statement)
                    if type(inner).__name__ == "For")
                break
        if store_index is None:
            reject(RK_E_UNSUPPORTED,
                   "the actual SiLU store is not in the root statement tree",
                   stage="rewrite")
    return LocatedFullLogical(
        root=nodes["root"], statements=statements, gemm_index=gemm_index,
        gemm=nodes["gemm"], reads=reads, writes=writes, roles=nodes,
        store=store,
        store_expression=None if store is None else store.value,
        store_statement_index=store_index, store_loops=store_loops)


def _effects_conflict(left: Any, right: Any, relation: str) -> bool:
    if relation == "read-read":
        return False
    return _same_buffer(left, right)


def _move_past_is_legal(index: int, gemm_index: int,
                        moved: set[int],
                        located: LocatedFullLogical) -> None:
    """Refuse a producer move that would cross a conflicting statement."""
    for other in range(index + 1, gemm_index):
        if other in moved:
            continue
        for left in located.writes[index]:
            for right in located.reads[other]:
                if _effects_conflict(left, right, "write-read"):
                    reject(RK_E_UNSUPPORTED,
                           "moving a producer across a statement would change "
                           "its consumer order", stage="rewrite",
                           actual={"producer": index, "crossed": other})
            for right in located.writes[other]:
                if _effects_conflict(left, right, "write-write"):
                    reject(RK_E_UNSUPPORTED,
                           "moving a producer across another writer would "
                           "change the program", stage="rewrite",
                           actual={"producer": index, "crossed": other})
        for left in located.reads[index]:
            for right in located.writes[other]:
                if _effects_conflict(left, right, "read-write"):
                    reject(RK_E_UNSUPPORTED,
                           "moving a producer across a writer would change "
                           "the value it reads", stage="rewrite",
                           actual={"producer": index, "crossed": other})


def _store_loops(prim_func: Any, store: Any) -> tuple[tuple[Any, ...], Any | None]:
    """Return the two store-index loops and the outer one in scheduled TIR."""
    if store is None:
        return (), None
    indices = list(store.indices)
    loops = []
    for _, node in _walk(prim_func.body):
        if type(node).__name__ != "For":
            continue
        if any(node.loop_var.same_as(index) for index in indices):
            loops.append(node)
    if len(loops) != 2:
        reject(RK_E_UNSUPPORTED,
               "the scheduled SiLU store must have two tile loops",
               stage="rewrite", actual=len(loops))
    outer = None
    for loop in loops:
        other = next(item for item in loops if not item.same_as(loop))
        if any(inner.same_as(other) for _, inner in _walk(loop.body)):
            outer = loop
            break
    if outer is None:
        reject(RK_E_UNSUPPORTED,
               "the scheduled SiLU loops have no outer tile loop",
               stage="rewrite")
    return tuple(loops), outer


def _rewrite_copy(node: Any, *, source_offsets: list[Any],
                  source_extents: list[Any], destination_offsets: list[Any],
                  destination_extents: list[Any]) -> Any:
    tir = _tir()
    call = node.value
    args = list(call.args)
    args[0] = _region_call(call.args[0], offsets=source_offsets,
                           extents=source_extents)
    args[1] = _region_call(call.args[1], offsets=destination_offsets,
                           extents=destination_extents)
    return tir.Evaluate(_call_with_args(call, args))


def _rewrite_fill(node: Any, *, offsets: list[Any],
                  extents: list[Any]) -> Any:
    tir = _tir()
    call = node.value
    args = list(call.args)
    args[0] = _region_call(call.args[0], offsets=offsets, extents=extents)
    return tir.Evaluate(_call_with_args(call, args))


def _rewrite_store_loop(statement: Any, *, store: Any, loops: tuple[Any, ...],
                        rows: Any, cols: Any, gm: int, gn: int) -> Any:
    """Retile the actual SiLU store loop to consume the current output tile."""
    tir = _tir()
    import tvm.tirx.stmt_functor as stmt_functor
    if len(loops) != 2:
        reject(RK_E_UNSUPPORTED,
               "the SiLU update must be enclosed by two tile loops",
               stage="rewrite", actual=len(loops))
    indices = list(store.indices)
    if len(indices) != 2:
        reject(RK_E_UNSUPPORTED,
               "the SiLU store must be indexed by two loop vars",
               stage="rewrite")
    by_var = {}
    for loop in loops:
        for index in indices:
            if loop.loop_var.same_as(index):
                by_var[index] = loop
    if len(by_var) != 2:
        reject(RK_E_UNSUPPORTED,
               "the SiLU store indices must be the enclosing loop vars",
               stage="rewrite")
    ordered = [by_var[index] for index in indices]
    bounds = [_const(getattr(loop, "min", 0)) for loop in ordered]
    steps = [1 if getattr(loop, "step", None) is None else _const(loop.step)
             for loop in ordered]
    extents = [_const(loop.extent) for loop in ordered]
    if bounds != [0, 0]:
        reject(RK_E_UNSUPPORTED,
               f"activation loop lower bounds {bounds} must be [0, 0]",
               stage="rewrite", actual=bounds)
    if steps != [1, 1]:
        reject(RK_E_UNSUPPORTED,
               "activation loops must step by one", stage="rewrite",
               actual=steps)
    if extents != [gm, gn]:
        reject(RK_E_SHAPE,
               f"activation loop extents {extents} do not cover the GEMM tile",
               stage="rewrite", actual=extents)
    def replace_loop(node: Any) -> Any:
        if type(node).__name__ != "For":
            return None
        target = None
        for index in indices:
            if node.loop_var.same_as(index):
                target = rows if index.same_as(indices[0]) else cols
                break
        if target is None:
            return None
        return tir.For(node.loop_var, _imm(0), target, node.kind, node.body,
                       node.thread_binding, dict(node.annotations or {}))

    rewritten = stmt_functor.ir_transform(statement, None, replace_loop)
    if not any(type(inner).__name__ == "BufferStore"
               for _, inner in _walk(rewritten)):
        reject(RK_E_UNSUPPORTED,
               "the SiLU rewrite dropped its store", stage="rewrite")
    return rewritten


def _rewrite_gemm(node: Any, *, offsets: list[Any], extents: list[Any]) -> Any:
    tir = _tir()
    call = node.value
    args = list(call.args)
    args[0] = _region_call(call.args[0], offsets=list(offsets),
                           extents=[extents[0], extents[2]])
    args[1] = _region_call(call.args[1], offsets=list(offsets),
                           extents=[extents[2], extents[1]])
    args[2] = _region_call(call.args[2], offsets=list(offsets),
                           extents=[extents[0], extents[1]])
    args[5], args[6], args[7] = extents[0], extents[1], extents[2]
    return tir.Evaluate(_call_with_args(call, args))


# ---------------------------------------------------------------------------
# Stage 1: tile/access rewrite, keeping the real TileLang operations visible


def schedule_full_logical_source(prim_func: Any, *, options: Any,
                                 tile_m: int = 256, tile_n: int = 256,
                                 block_k: int = 256,
                                 order: str = "m_major",
                                 b_is_input: bool = False) -> ScheduledFullLogical:
    tir = _tir()
    import tvm.tirx.stmt_functor as stmt_functor
    _validate_policy(prim_func, options)
    located = locate_full_logical_source(prim_func)
    data = _classify_dataflow(located.roles)
    gemm_call = located.gemm.value
    if len(gemm_call.args) < 10:
        reject(RK_E_UNSUPPORTED, "malformed source GEMM", stage="rewrite")
    if _const(gemm_call.args[3]) or _const(gemm_call.args[4]):
        reject(RK_E_UNSUPPORTED,
               "GEMM transposition is not supported", stage="rewrite")
    if _const(gemm_call.args[9]):
        reject(RK_E_NUMERIC_CONTRACT,
               "GEMM clear_accum is not supported; the zero fill is the only "
               "accumulator clear", stage="rewrite",
               actual=_const(gemm_call.args[9]))
    gm, gk, gn = _source_geometry(data)
    _validate_region_contract(located.roles)
    fill_value = _const(located.roles["fill"].value.args[1])
    if fill_value is None or int(fill_value) != 0:
        reject(RK_E_NUMERIC_CONTRACT,
               "the accumulator fill must be the zero clear", stage="rewrite",
               actual=fill_value)
    if not (1 <= gm <= 4096 and 1 <= gn <= 4096 and 1 <= gk <= 4096):
        reject(RK_E_SHAPE, "M/N/K must be in 1..4096", stage="rewrite",
               actual=[gm, gn, gk])
    if not (1 <= tile_m <= 256 and 1 <= tile_n <= 256 and
            1 <= block_k <= 256):
        reject(RK_E_SHAPE, "target tile sizes must be in 1..256",
               stage="rewrite")
    attrs = dict(getattr(prim_func, "attrs", None) or {})
    allowed_attrs = {"rknpu.schedule", "rknpu.tile_m", "rknpu.tile_n",
                     "rknpu.mn_order", "rknpu.source_k_rounding"}
    unknown_attrs = sorted(str(key) for key in attrs
                           if str(key).startswith("rknpu.") and
                           str(key) not in allowed_attrs)
    if unknown_attrs:
        reject(RK_E_UNSUPPORTED,
               "unknown source schedule annotations are not consumed",
               stage="rewrite", actual=unknown_attrs)
    schedule_present = any(key in attrs for key in (
        "rknpu.schedule", "rknpu.tile_m", "rknpu.tile_n", "rknpu.mn_order"))
    for _, loop in _walk(prim_func.body):
        if type(loop).__name__ != "For" or loop.thread_binding is None:
            continue
        if _const(getattr(loop, "min", 0)) != 0:
            reject(RK_E_UNSUPPORTED,
                   "block grid loops must start at zero", stage="rewrite",
                   actual=_const(getattr(loop, "min", 0)))
        if getattr(loop, "step", None) is not None and _const(loop.step) != 1:
            reject(RK_E_UNSUPPORTED,
                   "block grid loops must step by one", stage="rewrite",
                   actual=_const(loop.step))
        if _const(loop.extent) != 1:
            reject(RK_E_NUMERIC_CONTRACT,
                   "scheduled source narrows only after the full K reduction; "
                   "the tiled schedule writes FP16 back at every 256-wide K "
                   "block, so the two describe different numerical programs "
                   "(write the K loop, or drop the M/N declaration to stay on "
                   "the legacy default path)", stage="rewrite",
                   actual=_const(loop.extent))
    marker = attrs.get("rknpu.schedule")
    if marker is not None and str(marker) != "mn_tiles_v1":
        reject(RK_E_UNSUPPORTED,
               "unsupported full-logical M/N schedule kind", stage="rewrite",
               actual=str(marker), required="mn_tiles_v1")
    if "rknpu.tile_m" in attrs:
        tile_m = attrs["rknpu.tile_m"]
    if "rknpu.tile_n" in attrs:
        tile_n = attrs["rknpu.tile_n"]
    if type(tile_m) is not int or type(tile_n) is not int or \
            not 1 <= tile_m <= 256 or not 1 <= tile_n <= 256:
        reject(RK_E_UNSUPPORTED,
               "source M/N tile sizes must be integers in 1..256",
               stage="rewrite", actual=[tile_m, tile_n])
    if "rknpu.mn_order" in attrs:
        order = str(attrs["rknpu.mn_order"])
    if order not in {"m_major", "n_major"}:
        reject(RK_E_UNSUPPORTED, "unsupported tile traversal order",
               stage="rewrite", actual=order)
    rounding = attrs.get("rknpu.source_k_rounding")
    if schedule_present and gk > block_k and \
            str(rounding) != "explicit_per_k_fp16":
        reject(RK_E_NUMERIC_CONTRACT,
               "scheduled source has a different K-rounding contract",
               stage="rewrite", actual=rounding,
               required=["explicit_per_k_fp16", "single_k_tile_fp16"])

    # Locate the original ordered statements and the actual producer/consumer
    # positions.  The schedule below is assembled from this sequence, not from
    # a role-ordered operation template.
    def index_of(node: Any) -> int:
        return next(index for index, statement in enumerate(located.statements)
                    if statement.same_as(node))

    fill_index = index_of(located.roles["fill"])
    output_index = index_of(data["output_copy"])
    a_copy_index = index_of(data["a_copy"])
    b_copy_index = index_of(data["b_copy"])
    store_index = located.store_statement_index
    moved = {a_copy_index, b_copy_index}
    for producer in sorted(moved):
        _move_past_is_legal(producer, located.gemm_index, moved, located)

    m_tiles = (gm + tile_m - 1) // tile_m
    n_tiles = (gn + tile_n - 1) // tile_n
    k_tiles = (gk + block_k - 1) // block_k
    mi = tir.Var("mi", "int32")
    ni = tir.Var("ni", "int32")
    ko = tir.Var("ko", "int32")

    def mul(var: Any, extent: int) -> Any:
        return tir.Mul(var, _imm(extent))

    def tail(var: Any, extent: int, total: int) -> Any:
        return tir.min(_imm(extent),
                       tir.Sub(_imm(total), mul(var, extent)))

    m0 = mul(mi, tile_m)
    n0 = mul(ni, tile_n)
    k0 = mul(ko, block_k)
    rows = tail(mi, tile_m, gm)
    cols = tail(ni, tile_n, gn)
    kt = tail(ko, block_k, gk)

    def rewrite_original(index: int) -> Any:
        statement = located.statements[index]
        if index == fill_index:
            return _rewrite_fill(statement, offsets=[_imm(0), _imm(0)],
                                 extents=[rows, cols])
        if index == a_copy_index:
            return _rewrite_copy(
                statement,
                source_offsets=[
                    _add(data["a_source_region"]["offsets"][0], m0),
                    _add(data["a_source_region"]["offsets"][1], k0)],
                source_extents=[rows, kt],
                destination_offsets=list(
                    data["a_destination_region"]["offsets"]),
                destination_extents=[rows, kt])
        if index == b_copy_index:
            return _rewrite_copy(
                statement,
                source_offsets=[
                    _add(data["b_source_region"]["offsets"][0], k0),
                    _add(data["b_source_region"]["offsets"][1], n0)],
                source_extents=[kt, cols],
                destination_offsets=list(
                    data["b_destination_region"]["offsets"]),
                destination_extents=[kt, cols])
        if index == located.gemm_index:
            return _rewrite_gemm(statement, offsets=[_imm(0), _imm(0)],
                                 extents=[rows, cols, kt])
        if store_index is not None and index == store_index:
            return _rewrite_store_loop(
                statement, store=located.store, loops=located.store_loops,
                rows=rows, cols=cols, gm=gm, gn=gn)
        if index == output_index:
            return _rewrite_copy(
                statement,
                source_offsets=[_imm(0), _imm(0)],
                source_extents=[rows, cols],
                destination_offsets=[
                    _add(data["out_destination_region"]["offsets"][0], m0),
                    _add(data["out_destination_region"]["offsets"][1], n0)],
                destination_extents=[rows, cols])
        reject(RK_E_UNSUPPORTED,
               "a statement in the local rewrite range has no tile-op "
               "rewrite", stage="rewrite", actual=index)

    involved = moved | {fill_index, located.gemm_index, output_index}
    if store_index is not None:
        involved.add(store_index)
    local_start = min(involved)
    local_end = max(involved)
    k_body = tir.SeqStmt(
        [rewrite_original(index) for index in sorted(moved)] +
        [rewrite_original(located.gemm_index)])
    k_loop = tir.For(ko, _imm(0), _imm(k_tiles), tir.ForKind.SERIAL, k_body)
    local_sequence: list[Any] = []
    for index in range(local_start, local_end + 1):
        if index in moved:
            continue
        if index == located.gemm_index:
            local_sequence.append(k_loop)
        else:
            local_sequence.append(rewrite_original(index))
    local_body = tir.SeqStmt(local_sequence)
    if order == "m_major":
        tile_loop = tir.For(
            mi, _imm(0), _imm(m_tiles), tir.ForKind.SERIAL,
            tir.For(ni, _imm(0), _imm(n_tiles), tir.ForKind.SERIAL,
                    local_body))
    else:
        tile_loop = tir.For(
            ni, _imm(0), _imm(n_tiles), tir.ForKind.SERIAL,
            tir.For(mi, _imm(0), _imm(m_tiles), tir.ForKind.SERIAL,
                    local_body))
    body_sequence = list(located.statements[:local_start]) + [tile_loop] + \
        list(located.statements[local_end + 1:])
    scheduled_body = tir.SeqStmt(body_sequence) if len(body_sequence) > 1 \
        else body_sequence[0]
    root = located.root

    def replace_root(node: Any) -> Any:
        if not node.same_as(root):
            return None
        return tir.SBlock(
            iter_vars=list(getattr(node, "iter_vars", ()) or ()),
            reads=list(getattr(node, "reads", ()) or ()),
            writes=list(getattr(node, "writes", ()) or ()),
            name_hint=str(getattr(node, "name_hint", "tilelang_root")),
            body=scheduled_body,
            init=getattr(node, "init", None),
            alloc_buffers=list(getattr(node, "alloc_buffers", ()) or ()),
            match_buffers=list(getattr(node, "match_buffers", ()) or ()),
            annotations=dict(getattr(node, "annotations", None) or {}),
            span=getattr(node, "span", None))

    body = stmt_functor.ir_transform(prim_func.body, None, replace_root)
    scheduled = tir.PrimFunc(prim_func.params, body, prim_func.ret_type,
                             prim_func.buffer_map, prim_func.attrs)
    facts = LocalFacts(
        logical_m=gm, logical_k=gk, logical_n=gn,
        tile_m=tile_m, tile_n=tile_n, block_k=block_k,
        rounding="fp32_final_narrowing", rounding_points="final_output_only",
        output_name="Y",
        operand_a=str(data["a_param"].name),
        operand_b=str(data["b_param"].name),
        hints={}, order=order,
        b_role="input" if b_is_input else "constant",
        source_hash=_digest(prim_func.script()),
        activation="silu" if located.store is not None else None)
    stage = Stage(
        "schedule-source-ops", "transform", _digest(scheduled.script()),
        {"loops": {"m_tiles": m_tiles, "n_tiles": n_tiles,
                   "k_tiles": k_tiles, "order": order},
         "original_statements": len(located.statements),
         "moved_to_k": sorted(moved), "gemm_index": located.gemm_index,
         "store_statement": store_index,
         "local_range": [local_start, local_end]})
    return ScheduledFullLogical(source=prim_func, scheduled=scheduled,
                                facts=facts, stage=stage)


# ---------------------------------------------------------------------------
# Stage 2: visit each scheduled tile operation and replace it with rk_* calls


def lower_scheduled_full_logical(scheduled: Any, *,
                                 facts: LocalFacts) -> tuple[Any, Stage]:
    tir = _tir()
    import tvm.tirx.stmt_functor as stmt_functor
    nodes = _source_tile_nodes(scheduled)
    data = _classify_dataflow(nodes, keep_symbolic=True)
    store = data["store"]
    store_loops, outer_store_loop = _store_loops(scheduled, store)
    silu_rows = silu_cols = None
    if store is not None:
        by_var = {}
        for loop in store_loops:
            for index in store.indices:
                if loop.loop_var.same_as(index):
                    by_var[index] = loop.extent
        if len(by_var) != 2:
            reject(RK_E_UNSUPPORTED,
                   "scheduled SiLU loop vars do not match the store indices",
                   stage="rewrite")
        silu_rows = by_var[store.indices[0]]
        silu_cols = by_var[store.indices[1]]
    a_region = data["a_source_region"]
    b_region = data["b_source_region"]
    rows, kt = a_region["extents"]
    _b_kt, cols = b_region["extents"]
    m0, k0 = a_region["offsets"]
    n0 = b_region["offsets"][1]

    def extern(name: str, *args: Any) -> Any:
        return tir.Evaluate(tir.call_extern("void", name, *args))

    def region_args(region: dict[str, Any]) -> tuple[Any, ...]:
        return (region["offsets"][0], region["offsets"][1],
                region["extents"][0], region["extents"][1])

    def copy_call(source: Any, destination: Any,
                  source_region: dict[str, Any],
                  destination_region: dict[str, Any]) -> Any:
        return tir.call_extern(
            "void", RK_COPY, _string(str(source.name)),
            _string(str(destination.name)),
            *region_args(source_region), *region_args(destination_region))

    a_stage = data["a_stage"]
    b_stage = data["b_stage"]
    gemm_acc = data["gemm_acc"]
    a_param = data["a_param"]
    b_param = data["b_param"]
    y_param = data["y_param"]
    acc_name = _string("Acc16")
    a_stage_name = _string(str(a_stage.name))
    b_stage_name = _string(str(b_stage.name))
    cl_name = _string(str(gemm_acc.name))

    def lower_op(node: Any) -> Any:
        call = node.value
        name = _op_name(call)
        if not name.startswith("tl.tileop."):
            return None
        role = _tile_op_role(call, where="scheduled target lowering")
        if role == "fill":
            region = _region_exprs(call, keep_symbolic_extents=True)[0]
            return extern(RK_FILL, acc_name, call.args[1],
                          region["offsets"][0], region["offsets"][1],
                          region["extents"][0], region["extents"][1])
        if role == "copy":
            source, destination = _copy_regions(node, keep_symbolic=True)
            if _same_buffer(destination["buffer"], a_stage):
                padding = extern(
                    RK_FILL, a_stage_name, _imm(0),
                    _imm(0), _imm(0), destination["extents"][0],
                    destination["extents"][1])
                return tir.SeqStmt([
                    padding,
                    tir.Evaluate(copy_call(
                        source["buffer"], destination["buffer"],
                        source, destination))])
            if _same_buffer(destination["buffer"], b_stage):
                padding = extern(
                    RK_FILL, b_stage_name, _imm(0),
                    _imm(0), _imm(0), destination["extents"][0],
                    destination["extents"][1])
                return tir.SeqStmt([
                    padding,
                    tir.Evaluate(copy_call(
                        source["buffer"], destination["buffer"],
                        source, destination))])
            if _same_buffer(source["buffer"], gemm_acc) and \
                    _same_buffer(destination["buffer"], y_param):
                return extern(RK_STORE, acc_name,
                              _string(facts.output_name),
                              destination["offsets"][0],
                              destination["offsets"][1],
                              destination["extents"][0],
                              destination["extents"][1])
            reject(RK_E_UNSUPPORTED,
                   "a scheduled copy has no target lowering", stage="rewrite",
                   actual=str(call))
        if role == "gemm":
            regions = _region_exprs(call, keep_symbolic_extents=True)
            if len(regions) != 3:
                reject(RK_E_SHAPE, "scheduled GEMM lost its regions",
                       stage="rewrite")
            acc_region = regions[2]
            convert_in = tir.call_extern(
                "void", RK_COPY, acc_name, cl_name,
                *region_args(acc_region), *region_args(acc_region))
            gemm_out = "Z" if facts.activation else facts.output_name
            gemm = tir.call_extern(
                "void", RK_GEMM, _string(str(a_param.name)),
                _string(str(b_param.name)), _string(gemm_out),
                _string(str(a_stage.name)), _string(str(b_stage.name)),
                cl_name, m0, n0, rows, cols, k0, kt)
            convert_out = tir.call_extern(
                "void", RK_COPY, cl_name, acc_name,
                *region_args(acc_region), *region_args(acc_region))
            # The actual scheduled GEMM is one source operation.  The target
            # policy expands it into conversion, GEMM and FP16 writeback.
            return tir.SeqStmt([
                tir.Evaluate(convert_in), tir.Evaluate(gemm),
                tir.Evaluate(convert_out)])
        reject(RK_E_UNSUPPORTED,
               "a scheduled tile operation has no target lowering",
               stage="rewrite", actual=name)

    def replace_op(node: Any) -> Any:
        if type(node).__name__ == "For":
            if outer_store_loop is not None and node.same_as(outer_store_loop):
                return extern(RK_SILU, acc_name, silu_rows, silu_cols)
            return None
        if type(node).__name__ != "Evaluate":
            return None
        return lower_op(node)

    target_body = stmt_functor.ir_transform(
        scheduled.body, None, replace_op)

    acc_buffer = tir.decl_buffer((facts.tile_m, facts.tile_n), "float16",
                                 name="Acc16", scope="local.fragment")

    def add_accumulator(node: Any) -> Any:
        if type(node).__name__ != "SBlock" or \
                not list(getattr(node, "alloc_buffers", ()) or ()):
            return None
        return tir.SBlock(
            iter_vars=list(getattr(node, "iter_vars", ()) or ()),
            reads=list(getattr(node, "reads", ()) or ()),
            writes=list(getattr(node, "writes", ()) or ()),
            name_hint=str(getattr(node, "name_hint", "tilelang_root")),
            body=getattr(node, "body"),
            init=getattr(node, "init", None),
            alloc_buffers=list(getattr(node, "alloc_buffers", ()) or ()) +
                           [acc_buffer],
            match_buffers=list(getattr(node, "match_buffers", ()) or ()),
            annotations=dict(getattr(node, "annotations", None) or {}),
            span=getattr(node, "span", None))

    # `target_body` already contains the individually lowered child operations;
    # this second pass only adds the policy-owned accumulator allocation.
    target_body = stmt_functor.ir_transform(target_body, None, add_accumulator)
    target = tir.PrimFunc(scheduled.params, target_body, scheduled.ret_type,
                          scheduled.buffer_map, scheduled.attrs)
    stage = Stage(
        "lower-scheduled-ops", "transform", _digest(target.script()),
        {"lowered_nodes": {"fill": 1, "copy": 3, "gemm": 1,
                           "silu_store": 0 if store is None else 1},
         "inserted": {"accumulator": "Acc16", "padding_fill": 2,
                      "convert": 2}})
    return target, stage


@dataclass(frozen=True)
class KLoopRewrite:
    """Source-operation lowering for an explicit unguarded K-loop GEMM."""

    source: Any
    target: Any
    instantiated: Any
    facts: LocalFacts
    tasks: list[dict[str, Any]]
    stages: tuple[Stage, Stage]
    source_k_loop: dict[str, Any]
    schedule_declared: bool


def _k_loop_structure(prim_func: Any) -> dict[str, Any] | None:
    """Structural locator for final-only or explicit-roundtrip K-loop GEMMs."""
    params = list((getattr(prim_func, "buffer_map", None) or {}).values())
    if len(params) != 3 or any(str(buffer.dtype) != "float16"
                               for buffer in params):
        return None
    tile_nodes: list[Any] = []
    block_loops: list[Any] = []
    tile_loops: list[Any] = []
    root = None
    tir = _tir()
    for _, node in _walk(prim_func.body):
        if type(node).__name__ == "IfThenElse":
            return None
        if type(node).__name__ == "SBlock" and \
                list(getattr(node, "alloc_buffers", ()) or ()):
            root = node
        if type(node).__name__ == "For":
            if node.thread_binding is not None:
                block_loops.append(node)
            elif any(_op_name(inner.value).startswith("tl.tileop.")
                     for _, inner in _walk(node)
                     if type(inner).__name__ == "Evaluate"):
                tile_loops.append(node)
        if type(node).__name__ == "Evaluate" and \
                _op_name(node.value).startswith("tl.tileop."):
            tile_nodes.append(node)
    if root is None or not (1 <= len(block_loops) <= 2) or \
            len(tile_loops) != 1:
        return None
    k_loop = tile_loops[0]
    inside = [node for _, node in _walk(k_loop.body)
              if type(node).__name__ == "Evaluate" and
              _op_name(node.value).startswith("tl.tileop.")]

    def in_inside(node: Any) -> bool:
        return any(node.same_as(item) for item in inside)

    outside = [node for node in tile_nodes if not in_inside(node)]
    gemms = [node for node in inside
             if _op_name(node.value) == "tl.tileop.gemm"]
    fills = [node for node in outside
             if _op_name(node.value) == "tl.tileop.fill"]
    if len(gemms) != 1 or len(fills) != 1:
        return None
    gemm_regions = _region_exprs(gemms[0].value)
    if len(gemm_regions) != 3:
        return None
    a_stage, b_stage, gemm_acc = (region["buffer"]
                                  for region in gemm_regions)
    a_copy = b_copy = None
    roundtrip: list[Any] = []
    for node in inside:
        name = _op_name(node.value)
        if name != "tl.tileop.copy":
            continue
        regions = _region_exprs(node.value)
        if len(regions) != 2:
            return None
        source, destination = regions[0]["buffer"], regions[1]["buffer"]
        if _same_buffer(destination, a_stage):
            a_copy = node
        elif _same_buffer(destination, b_stage):
            b_copy = node
        elif _scope_of(source).startswith("local") and \
                _scope_of(destination).startswith("local"):
            roundtrip.append(node)
        else:
            return None
    if a_copy is None or b_copy is None or len(inside) != 3 + len(roundtrip):
        return None
    output_copies = [node for node in outside
                     if _op_name(node.value) == "tl.tileop.copy"]
    if len(output_copies) != 1:
        return None
    out_regions = _region_exprs(output_copies[0].value)
    if len(out_regions) != 2 or not any(
            _same_buffer(out_regions[1]["buffer"], param)
            for param in params):
        return None
    fill_regions = _region_exprs(fills[0].value)
    if len(fill_regions) != 1:
        return None
    fill_buffer = fill_regions[0]["buffer"]
    output_source = out_regions[0]["buffer"]
    if not _same_buffer(fill_buffer, output_source):
        return None
    a_regions = _region_exprs(a_copy.value)
    b_regions = _region_exprs(b_copy.value)
    a_source, a_destination = a_regions
    b_source, b_destination = b_regions
    a_param = a_source["buffer"]
    b_param = b_source["buffer"]
    y_param = out_regions[1]["buffer"]
    if not any(_same_buffer(a_param, param) for param in params) or \
            not any(_same_buffer(b_param, param) for param in params):
        return None
    if len({id(a_param), id(b_param), id(y_param)}) != 3:
        return None
    if len(roundtrip) == 0:
        if not _same_buffer(fill_buffer, gemm_acc):
            return None
        mode = "final_only"
    elif len(roundtrip) == 2:
        if str(fill_buffer.dtype) != "float16" or \
                not _scope_of(fill_buffer).startswith("local"):
            return None
        has_load = has_writeback = False
        for node in roundtrip:
            source, destination = _region_exprs(node.value)
            if _same_buffer(source["buffer"], fill_buffer) and \
                    _same_buffer(destination["buffer"], gemm_acc):
                has_load = True
            elif _same_buffer(source["buffer"], gemm_acc) and \
                    _same_buffer(destination["buffer"], fill_buffer):
                has_writeback = True
        if not has_load or not has_writeback:
            return None
        mode = "roundtrip"
    else:
        return None
    return {
        "root": root, "k_loop": k_loop, "block_loops": tuple(block_loops),
        "inside": tuple(inside), "outside": tuple(outside),
        "gemm": gemms[0], "fill": fills[0],
        "a_copy": a_copy, "b_copy": b_copy,
        "output_copy": output_copies[0], "roundtrip": tuple(roundtrip),
        "mode": mode, "params": tuple(params),
        "a_param": a_param, "b_param": b_param, "y_param": y_param,
        "a_stage": a_stage, "b_stage": b_stage, "gemm_acc": gemm_acc,
        "running": fill_buffer,
        "a_source": a_source, "a_destination": a_destination,
        "b_source": b_source, "b_destination": b_destination,
        "out_source": out_regions[0], "out_destination": out_regions[1],
        "fill_region": fill_regions[0],
        "gemm_regions": gemm_regions,
    }


def is_k_loop_candidate(prim_func: Any) -> bool:
    """Structural classifier for the two migrated K-loop forms."""
    return _k_loop_structure(prim_func) is not None


def _region_args(region: dict[str, Any]) -> tuple[Any, ...]:
    return (region["offsets"][0], region["offsets"][1],
            region["extents"][0], region["extents"][1])


def _retile_k_loop_source(prim_func: Any, structure: dict[str, Any], *,
                          tile_m: int, tile_n: int) -> Any:
    """Apply explicit M/N tiling to the source stage before task lowering.

    Only exact coverage is represented here: a source with M/N tails under a
    different explicit tile is reported as an unsupported local retile rather
    than silently keeping the physical source tiling.
    """
    tir = _tir()
    import tvm.tirx.stmt_functor as stmt_functor
    y_shape = tuple(int(value) for value in structure["y_param"].shape)
    logical_m, logical_n = y_shape
    logical_k = int(structure["a_param"].shape[1])
    block_k = int(structure["a_destination"]["extents"][1])
    if tile_m != int(structure["a_destination"]["extents"][0]) or \
            tile_n != int(structure["b_destination"]["extents"][1]):
        if logical_m % tile_m or logical_n % tile_n or \
                logical_k % block_k:
            reject(RK_E_UNSUPPORTED,
                   "explicit M/N retiling with M/N/K tails is not implemented "
                   "by the local K-loop source rewrite", stage="rewrite",
                   actual={"logical": [logical_m, logical_k, logical_n],
                           "tile": [tile_m, tile_n, block_k]})
    a_vars = [var for var in _vars_in(
        structure["a_source"]["offsets"][0])
        if any(var.same_as(loop.loop_var)
               for loop in structure["block_loops"])]
    b_vars = [var for var in _vars_in(
        structure["b_source"]["offsets"][1])
        if any(var.same_as(loop.loop_var)
               for loop in structure["block_loops"])]
    if len(a_vars) != 1 or len(b_vars) != 1:
        reject(RK_E_SHAPE,
               "explicit M/N retiling needs one M and one N block variable",
               stage="rewrite")
    m_var, n_var = a_vars[0], b_vars[0]
    k_var = structure["k_loop"].loop_var
    old_a, old_b = structure["a_stage"], structure["b_stage"]
    old_cl = structure["gemm_acc"]
    old_running = structure["running"]
    new_a = tir.decl_buffer((tile_m, block_k), "float16",
                            name=str(old_a.name), scope=str(old_a.scope()))
    new_b = tir.decl_buffer((block_k, tile_n), "float16",
                            name=str(old_b.name), scope=str(old_b.scope()))
    new_cl = tir.decl_buffer((tile_m, tile_n), "float32",
                             name=str(old_cl.name),
                             scope=str(old_cl.scope()))
    new_running = tir.decl_buffer((tile_m, tile_n), "float16",
                                  name=str(old_running.name),
                                  scope=str(old_running.scope()))

    def mapped(buffer: Any) -> Any:
        for old, new in ((old_a, new_a), (old_b, new_b),
                         (old_cl, new_cl), (old_running, new_running)):
            if _same_buffer(buffer, old):
                return new
        return buffer

    def region(call: Any, buffer: Any, offsets: list[Any],
               extents: list[int]) -> Any:
        args = list(call.args)
        new_load = tir.BufferLoad(buffer, list(offsets))
        return tir.Call(call.dtype, call.op,
                        [new_load, args[1], _imm(extents[0]),
                         _imm(extents[1])],
                        dict(getattr(call, "annotations", None) or {}))

    def lower(node: Any) -> Any:
        if type(node).__name__ != "Evaluate" or \
                not _op_name(node.value).startswith("tl.tileop."):
            return None
        call = node.value
        if node.same_as(structure["a_copy"]):
            args = list(call.args)
            source, dest = _region_exprs(call)
            args[0] = region(call.args[0], structure["a_param"],
                             [tir.Mul(m_var, _imm(tile_m)),
                              tir.Mul(k_var, _imm(block_k))],
                             [tile_m, block_k])
            args[1] = region(call.args[1], new_a, [0, 0],
                             [tile_m, block_k])
            return tir.Evaluate(_call_with_args(call, args))
        if node.same_as(structure["b_copy"]):
            args = list(call.args)
            args[0] = region(call.args[0], structure["b_param"],
                             [tir.Mul(k_var, _imm(block_k)),
                              tir.Mul(n_var, _imm(tile_n))],
                             [block_k, tile_n])
            args[1] = region(call.args[1], new_b, [0, 0],
                             [block_k, tile_n])
            return tir.Evaluate(_call_with_args(call, args))
        if node.same_as(structure["fill"]):
            args = list(call.args)
            target = mapped(structure["fill_region"]["buffer"])
            args[0] = region(call.args[0], target, [0, 0],
                             [int(target.shape[0]), int(target.shape[1])])
            return tir.Evaluate(_call_with_args(call, args))
        if node.same_as(structure["gemm"]):
            args = list(call.args)
            args[0] = region(call.args[0], new_a, [0, 0],
                             [tile_m, block_k])
            args[1] = region(call.args[1], new_b, [0, 0],
                             [block_k, tile_n])
            args[2] = region(call.args[2], new_cl, [0, 0],
                             [tile_m, tile_n])
            args[5], args[6], args[7] = (_imm(tile_m), _imm(tile_n),
                                         _imm(block_k))
            return tir.Evaluate(_call_with_args(call, args))
        if any(node.same_as(item) for item in structure["roundtrip"]):
            args = list(call.args)
            source, dest = _region_exprs(call)
            ns, nd = mapped(source["buffer"]), mapped(dest["buffer"])
            args[0] = region(call.args[0], ns, [0, 0],
                             [int(ns.shape[0]), int(ns.shape[1])])
            args[1] = region(call.args[1], nd, [0, 0],
                             [int(nd.shape[0]), int(nd.shape[1])])
            return tir.Evaluate(_call_with_args(call, args))
        if node.same_as(structure["output_copy"]):
            args = list(call.args)
            args[0] = region(call.args[0], new_running, [0, 0],
                             [tile_m, tile_n])
            args[1] = region(call.args[1], structure["y_param"],
                             [tir.Mul(m_var, _imm(tile_m)),
                              tir.Mul(n_var, _imm(tile_n))],
                             [tile_m, tile_n])
            return tir.Evaluate(_call_with_args(call, args))
        return None

    new_m = logical_m // tile_m
    new_n = logical_n // tile_n

    def replace_schedule(node: Any) -> Any:
        if type(node).__name__ == "For":
            var = node.loop_var
            extent = (new_m if var.same_as(m_var)
                      else new_n if var.same_as(n_var)
                      else None)
            if extent is not None:
                return tir.For(
                    node.loop_var, node.min, _imm(extent), node.kind,
                    node.body, node.thread_binding,
                    dict(getattr(node, "annotations", None) or {}),
                    getattr(node, "step", None), getattr(node, "span", None))
        if type(node).__name__ == "SBlock" and node.same_as(structure["root"]):
            old_allocs = list(getattr(node, "alloc_buffers", ()) or ())
            allocs = [mapped(buffer) for buffer in old_allocs]
            for new in (new_a, new_b, new_cl, new_running):
                if not any(_same_buffer(new, buffer) for buffer in allocs):
                    allocs.append(new)
            return tir.SBlock(
                iter_vars=list(getattr(node, "iter_vars", ()) or ()),
                reads=list(getattr(node, "reads", ()) or ()),
                writes=list(getattr(node, "writes", ()) or ()),
                name_hint=str(getattr(node, "name_hint", "tilelang_root")),
                body=getattr(node, "body"), init=getattr(node, "init", None),
                alloc_buffers=allocs,
                match_buffers=list(getattr(node, "match_buffers", ()) or ()),
                annotations=dict(getattr(node, "annotations", None) or {}),
                span=getattr(node, "span", None))
        return None

    body = stmt_functor.ir_transform(prim_func.body, None, replace_schedule)
    body = stmt_functor.ir_transform(body, None, lower)
    return tir.PrimFunc(prim_func.params, body, prim_func.ret_type,
                        prim_func.buffer_map, prim_func.attrs)


def rewrite_k_loop_tir(prim_func: Any, *, options: Any,
                       b_is_input: bool = False,
                       max_tasks: int | None = None) -> KLoopRewrite:
    """Lower actual K-loop copy/GEMM relations item by item."""
    from ..errors import RK_E_NUMERIC_CONTRACT
    if options is None or getattr(options, "numeric_mode", None) != "rk_f16_tiled":
        reject(RK_E_NUMERIC_CONTRACT,
               "the K-loop lowering only supports rk_f16_tiled",
               stage="frontend",
               actual=None if options is None else options.numeric_mode)
    if getattr(options, "linear_task_order", "baseline") not in {
            "baseline", "scheduled_baseline"}:
        reject(RK_E_UNSUPPORTED,
               "the K-loop lowering preserves program order; the requested "
               "linear_task_order would regroup it", stage="frontend",
               actual=options.linear_task_order,
               required=["baseline", "scheduled_baseline"])
    if getattr(options, "linear_group_n_tiles", 0):
        reject(RK_E_UNSUPPORTED,
               "the K-loop lowering does not apply submit regrouping",
               stage="frontend", actual=options.linear_group_n_tiles)
    _validate_policy(prim_func, options)
    structure = _k_loop_structure(prim_func)
    if structure is None:
        reject(RK_E_SHAPE,
               "the source is not one of the migrated unguarded K-loop forms",
               stage="frontend")
    if any(type(inner).__name__ == "BufferStore"
           for _, inner in _walk(prim_func.body)):
        reject(RK_E_UNSUPPORTED,
               "activation inside a K-loop program is not lowered",
               stage="frontend")
    k_loop = structure["k_loop"]
    try:
        loop_kind = int(k_loop.kind)
    except Exception:
        loop_kind = -1
    if loop_kind != int(_tir().ForKind.SERIAL):
        reject(RK_E_UNSUPPORTED,
               "only a serial/pipelined K loop is lowered",
               stage="frontend", actual=loop_kind)
    if len(structure["outside"]) != 2:
        reject(RK_E_UNSUPPORTED,
               "unexpected tile operations outside the K loop",
               stage="frontend",
               actual=len(structure["outside"]))
    block_k = int(structure["a_destination"]["extents"][1])
    if block_k != 256:
        reject(RK_E_UNSUPPORTED,
               "the source K block is fixed at 256", stage="frontend",
               actual=block_k)
    trips = _const(k_loop.extent)
    a_shape = tuple(int(value) for value in structure["a_param"].shape)
    b_shape = tuple(int(value) for value in structure["b_param"].shape)
    if trips is None or trips <= 0 or \
            trips * block_k != a_shape[1] or trips * block_k != b_shape[0]:
        reject(RK_E_SHAPE,
               "K-loop tiling does not tile the parameter shapes exactly",
               stage="frontend",
               actual={"trips": trips, "block_k": block_k,
                       "A": list(a_shape), "B": list(b_shape)})
    gemm_call = structure["gemm"].value
    if len(gemm_call.args) < 10:
        reject(RK_E_UNSUPPORTED, "malformed K-loop GEMM", stage="frontend")
    if _const(gemm_call.args[9]):
        reject(RK_E_NUMERIC_CONTRACT,
               "GEMM clear_accum is not supported inside the K loop",
               stage="frontend", actual=True)
    transpose_a = _known_const(gemm_call.args[3])
    transpose_b = _known_const(gemm_call.args[4])
    if transpose_a not in (0, 1) or transpose_b not in (0, 1):
        reject(RK_E_UNSUPPORTED,
               "tile GEMM transpose flags must be statically known",
               stage="frontend",
               actual={"transpose_a": str(gemm_call.args[3]),
                       "transpose_b": str(gemm_call.args[4])})
    if transpose_a or transpose_b:
        reject(RK_E_UNSUPPORTED,
               "tile GEMM transposition is not supported",
               stage="frontend",
               actual={"transpose_a": transpose_a,
                       "transpose_b": transpose_b})
    a_offsets = structure["a_source"]["offsets"]
    b_offsets = structure["b_source"]["offsets"]
    k_var = k_loop.loop_var
    if k_var not in _vars_in(a_offsets[1]) or \
            k_var in _vars_in(a_offsets[0]) or \
            k_var not in _vars_in(b_offsets[0]) or \
            k_var in _vars_in(b_offsets[1]):
        reject(RK_E_UNSUPPORTED,
               "K-loop slice indices must advance by the block size only on "
               "the K axis", stage="frontend")
    for axis, offset in (("A", a_offsets[1]), ("B", b_offsets[0])):
        residual = _arith().simplify(
            _tir().Sub(offset,
                       _tir().Mul(k_var, _imm(block_k))))
        if k_var in _vars_in(residual):
            reject(RK_E_UNSUPPORTED,
                   f"{axis} K index must advance by the block size with the "
                   "K loop", stage="frontend")
    sequence = [inner for _, inner in _walk(k_loop.body)
                if type(inner).__name__ == "Evaluate" and
                _op_name(inner.value).startswith("tl.tileop.")]
    a_index = sequence.index(structure["a_copy"])
    b_index = sequence.index(structure["b_copy"])
    gemm_index = sequence.index(structure["gemm"])
    if a_index >= gemm_index or b_index >= gemm_index:
        reject(RK_E_UNSUPPORTED,
               "A/B slice copies must precede the GEMM", stage="frontend")
    tir = _tir()
    import tvm.tirx.stmt_functor as stmt_functor
    attrs = dict(getattr(prim_func, "attrs", None) or {})
    allowed = {"rknpu.schedule", "rknpu.tile_m", "rknpu.tile_n",
               "rknpu.mn_order", "rknpu.source_k_rounding"}
    unknown = sorted(str(key) for key in attrs
                     if str(key).startswith("rknpu.") and
                     str(key) not in allowed)
    if unknown:
        reject(RK_E_UNSUPPORTED,
               "unknown K-loop schedule annotations", stage="frontend",
               actual=unknown)
    mode = structure["mode"]
    actual_tile_m = int(structure["a_destination"]["extents"][0])
    actual_tile_n = int(structure["b_destination"]["extents"][1])
    tile_m = attrs.get("rknpu.tile_m", actual_tile_m)
    tile_n = attrs.get("rknpu.tile_n", actual_tile_n)
    order = str(attrs.get("rknpu.mn_order", "m_major"))
    if type(tile_m) is not int or type(tile_n) is not int or \
            not 1 <= tile_m <= 256 or not 1 <= tile_n <= 256:
        reject(RK_E_UNSUPPORTED,
               "source M/N tile sizes must be integers in 1..256",
               stage="frontend", actual=[tile_m, tile_n])
    if order not in {"m_major", "n_major"}:
        reject(RK_E_UNSUPPORTED, "unsupported M/N traversal order",
               stage="frontend", actual=order)
    if tile_m != actual_tile_m or tile_n != actual_tile_n:
        prim_func = _retile_k_loop_source(
            prim_func, structure, tile_m=tile_m, tile_n=tile_n)
        structure = _k_loop_structure(prim_func)
        if structure is None:
            reject(RK_E_SHAPE,
                   "the explicitly retiled K-loop source is not consumable",
                   stage="rewrite")
    schedule_present = any(key in attrs for key in (
        "rknpu.schedule", "rknpu.tile_m", "rknpu.tile_n", "rknpu.mn_order"))
    k_trips = _const(structure["k_loop"].extent)
    if k_trips is None or k_trips <= 0:
        reject(RK_E_SHAPE, "the K loop needs a static positive trip count",
               stage="frontend")
    declared_rounding = attrs.get("rknpu.source_k_rounding")
    rounding = ("explicit_per_k_fp16" if mode == "roundtrip"
                else ("single_k_tile_fp16" if k_trips == 1
                      else "fp32_final_narrowing"))
    if declared_rounding is not None and \
            str(declared_rounding) != rounding:
        reject(RK_E_NUMERIC_CONTRACT,
               "the declared K-rounding contract does not match the actual "
               "K-loop copies", stage="frontend", actual=declared_rounding,
               required=rounding)
    block_k = int(structure["a_destination"]["extents"][1])
    a_source = structure["a_source"]
    b_source = structure["b_source"]
    out_destination = structure["out_destination"]
    gemm_regions = structure["gemm_regions"]
    rows, kt = a_source["extents"]
    b_kt, cols = b_source["extents"]
    if int(kt) != int(b_kt):
        reject(RK_E_SHAPE, "K-loop A/B slice extents disagree",
               stage="frontend", actual=[kt, b_kt])
    a_name = _string(str(structure["a_param"].name))
    b_name = _string(str(structure["b_param"].name))
    y_name = _string("Y")
    a_stage_name = _string(str(structure["a_stage"].name))
    b_stage_name = _string(str(structure["b_stage"].name))
    cl_name = _string(str(structure["gemm_acc"].name))
    running_name = _string("Acc16" if mode == "final_only"
                           else str(structure["running"].name))
    m0, k0 = a_source["offsets"]
    n0 = b_source["offsets"][1]
    logical_rows = int(rows)
    logical_cols = int(cols)
    logical_kt = int(kt)

    def extern(name: str, *args: Any) -> Any:
        return tir.Evaluate(tir.call_extern("void", name, *args))

    def lower_op(node: Any) -> Any:
        call = node.value
        if node.same_as(structure["fill"]):
            region = structure["fill_region"]
            return extern(RK_FILL, running_name, call.args[1],
                          *_region_args(region))
        if node.same_as(structure["a_copy"]):
            destination = structure["a_destination"]
            return tir.SeqStmt([
                extern(RK_FILL, a_stage_name, _imm(0),
                       _imm(0), _imm(0), *destination["extents"]),
                extern(RK_COPY, a_name, a_stage_name,
                       *_region_args(structure["a_source"]),
                       *_region_args(destination)),
            ])
        if node.same_as(structure["b_copy"]):
            destination = structure["b_destination"]
            return tir.SeqStmt([
                extern(RK_FILL, b_stage_name, _imm(0),
                       _imm(0), _imm(0), *destination["extents"]),
                extern(RK_COPY, b_name, b_stage_name,
                       *_region_args(structure["b_source"]),
                       *_region_args(destination)),
            ])
        if any(node.same_as(item) for item in structure["roundtrip"]):
            source, destination = _region_exprs(call)
            return extern(RK_COPY, _string(str(source["buffer"].name)),
                          _string(str(destination["buffer"].name)),
                          *_region_args(source), *_region_args(destination))
        if node.same_as(structure["gemm"]):
            gemm = extern(RK_GEMM, a_name, b_name, y_name,
                          a_stage_name, b_stage_name, cl_name,
                          m0, n0, rows, cols, k0, kt)
            if mode == "roundtrip":
                return gemm
            acc_region = structure["gemm_regions"][2]
            convert_in = extern(RK_COPY, running_name, cl_name,
                                *_region_args(acc_region),
                                *_region_args(acc_region))
            convert_out = extern(RK_COPY, cl_name, running_name,
                                 *_region_args(acc_region),
                                 *_region_args(acc_region))
            return tir.SeqStmt([convert_in, gemm, convert_out])
        if node.same_as(structure["output_copy"]):
            return extern(RK_STORE, running_name, y_name,
                          *_region_args(out_destination))
        reject(RK_E_UNSUPPORTED,
               "a K-loop tile operation has no target lowering",
               stage="frontend", actual=str(call))

    def replace_op(node: Any) -> Any:
        if type(node).__name__ != "Evaluate":
            return None
        return lower_op(node)

    target_body = stmt_functor.ir_transform(
        prim_func.body, None, replace_op)
    root = structure["root"]
    allocations = list(getattr(root, "alloc_buffers", ()) or ())
    if mode == "final_only":
        allocations.append(tir.decl_buffer(
            (tile_m, tile_n), "float16", name="Acc16",
            scope="local.fragment"))

    def replace_root(node: Any) -> Any:
        if not node.same_as(root):
            return None
        return tir.SBlock(
            iter_vars=list(getattr(node, "iter_vars", ()) or ()),
            reads=list(getattr(node, "reads", ()) or ()),
            writes=list(getattr(node, "writes", ()) or ()),
            name_hint=str(getattr(node, "name_hint", "tilelang_root")),
            body=getattr(node, "body"),
            init=getattr(node, "init", None),
            alloc_buffers=allocations,
            match_buffers=list(getattr(node, "match_buffers", ()) or ()),
            annotations=dict(getattr(node, "annotations", None) or {}),
            span=getattr(node, "span", None))

    target_body = stmt_functor.ir_transform(target_body, None, replace_root)
    target = tir.PrimFunc(prim_func.params, target_body, prim_func.ret_type,
                          prim_func.buffer_map, prim_func.attrs)
    k_annotations = {}
    for key, value in dict(getattr(structure["k_loop"],
                                   "annotations", {}) or {}).items():
        parsed = _const(value)
        k_annotations[str(key)] = (parsed if parsed is not None
                                   else str(value))
    pipeline_stages = int(k_annotations.get("num_stages", 0) or 0)
    hints = {}
    if pipeline_stages > 0:
        hints["pipeline_hint"] = {
            "num_stages": pipeline_stages, "applied": False,
            "reason": "an RKNPU plan has no software pipeline; the K chain is "
                      "executed as ordered submissions",
        }
    m_dim = int(structure["y_param"].shape[0])
    k_dim = int(structure["a_param"].shape[1])
    n_dim = int(structure["y_param"].shape[1])
    facts = LocalFacts(
        logical_m=m_dim, logical_k=k_dim, logical_n=n_dim,
        tile_m=tile_m, tile_n=tile_n, block_k=block_k,
        rounding=rounding,
        rounding_points=("after_each_k_block" if mode == "roundtrip"
                         else "final_output_only"),
        output_name="Y", operand_a=str(structure["a_param"].name),
        operand_b=str(structure["b_param"].name),
        hints=hints, order=order,
        b_role="input" if b_is_input else "constant",
        source_hash=_digest(prim_func.script()), activation=None)
    stage = Stage("rewrite-k-loop-ops", "transform",
                  _digest(target.script()),
                  {"mode": mode, "k_trips": k_trips,
                   "tile": [tile_m, tile_n, block_k]})
    instantiated, tasks, instantiate_stage = pass_instantiate(
        target, facts, max_tasks=max_tasks,
        honor_grid_order=(mode == "roundtrip" and
                          len(structure["block_loops"]) == 2))
    k_blocks = [{"index": index, "k0": index * block_k,
                 "extent": min(block_k, k_dim - index * block_k)}
                for index in range(k_trips)]
    source_k_loop = {
        "source_k_block": block_k,
        "source_k_trips": k_trips,
        "logical_k": k_dim,
        "k_blocks": k_blocks,
        "source_rounding": rounding,
        "rounding_points": ("after_each_k_block" if mode == "roundtrip"
                            else "final_output_only"),
        "lowering_k_tile": 256,
        "source_loop_annotations": dict(k_annotations),
        "source_pipeline_stages": pipeline_stages,
        "pipeline_hint_dropped": bool(pipeline_stages),
        "pipeline_not_applied_reason": (
            "no software pipeline is implemented on this target: the K chain "
            "is emitted as an ordered accumulation of 256-wide tasks, so the "
            "request is recorded but not applied" if pipeline_stages else None),
    }
    return KLoopRewrite(
        source=prim_func, target=target, instantiated=instantiated,
        facts=facts, tasks=tasks, stages=(stage, instantiate_stage),
        source_k_loop=source_k_loop, schedule_declared=schedule_present)


@dataclass(frozen=True)
class DataflowRewrite:
    """Carrier metadata and the target TIR produced from actual statements."""

    source: Any
    target: Any
    module: Any
    tasks: list[dict[str, Any]]
    facts: dict[str, Any]
    stages: tuple[Stage, Stage]


def _statement_buffer_key(buffer: Any) -> str:
    return str(buffer.name)


def _store_loads(node: Any) -> list[Any]:
    return [inner for _, inner in _walk(node)
            if type(inner).__name__ == "BufferLoad"]


def _has_sigmoid(node: Any) -> bool:
    return any("sigmoid" in str(inner) for _, inner in _walk(node)
               if type(inner).__name__ == "Call")


def is_local_dataflow_candidate(prim_func: Any) -> bool:
    """Structural gate for the admitted local 2/3-GEMM dataflow forms."""
    tir = _tir()
    gemms = 0
    stores = 0
    thread_loops = 0
    for _, node in _walk(prim_func.body):
        if type(node).__name__ == "BufferStore":
            stores += 1
        elif type(node).__name__ == "Evaluate" and \
                _op_name(node.value) == "tl.tileop.gemm":
            gemms += 1
        elif isinstance(node, tir.For) and node.thread_binding is not None:
            thread_loops += 1
    if gemms not in {1, 2, 3} or stores < 2 or thread_loops not in {1, 2}:
        return False
    if gemms != 1:
        return True
    params = list((getattr(prim_func, "buffer_map", None) or {}).values())
    low_rank = [param for param in params
                if len(tuple(int(v) for v in param.shape)) < 2]
    if not low_rank:
        return False
    for _, node in _walk(prim_func.body):
        if type(node).__name__ != "BufferStore" or \
                type(node.value).__name__ not in {"Add", "Mul"}:
            continue
        for load in _store_loads(node.value):
            if any(_same_buffer(param, load.buffer) for param in low_rank):
                return True
    return False


is_ffn_candidate = is_local_dataflow_candidate


def extract_rk_tasks(target: Any) -> list[dict[str, Any]]:
    """Translate ordered rk_* calls from target TIR into region-carrying tasks."""
    tasks: list[dict[str, Any]] = []

    def integer(arg: Any, what: str) -> int:
        value = _const(arg)
        if value is None:
            reject(RK_E_SHAPE, f"{what} must be a static integer",
                   stage="rewrite", actual=str(arg))
        return int(value)

    def region(args: list[Any], start: int) -> dict[str, int]:
        return {
            "row": integer(args[start], "region row"),
            "col": integer(args[start + 1], "region col"),
            "rows": integer(args[start + 2], "region rows"),
            "cols": integer(args[start + 3], "region cols"),
        }

    for _, node in _walk(target.body):
        if type(node).__name__ != "Evaluate":
            continue
        call = node.value
        name = _op_name(call)
        args = list(call.args)
        if name.endswith("call_extern"):
            if not args:
                continue
            name = str(getattr(args[0], "value", args[0]))
            args = args[1:]
        if name == "rk_fill":
            tasks.append({
                "kind": "fill",
                "value_id": integer(args[0], "fill value id"),
                "buffer_token": integer(args[1], "fill buffer token"),
                "value": integer(args[2], "fill value"),
                "region": region(args, 3),
            })
        elif name in {"rk_copy", "rk_cast"}:
            tasks.append({
                "kind": "copy" if name == "rk_copy" else "cast",
                "source_token": integer(args[0], "copy source token"),
                "destination_token": integer(args[1], "copy destination token"),
                "source_value": integer(args[2], "copy source value"),
                "source_region": region(args, 3),
                "destination_region": region(args, 7),
            })
        elif name == "rk_gemm":
            tasks.append({
                "kind": "gemm", "op_id": integer(args[0], "gemm op id"),
                "input_tokens": [integer(args[1], "gemm A"),
                                 integer(args[2], "gemm B")],
                "output_value": integer(args[3], "gemm output value"),
                "output_token": integer(args[4], "gemm output token"),
                "a_region": region(args, 5),
                "b_region": region(args, 9),
                "region": region(args, 13),
                "k0": integer(args[17], "gemm k0"),
                "kt": integer(args[18], "gemm kt"),
            })
        elif name == "rk_silu":
            tasks.append({
                "kind": "silu", "op_id": integer(args[0], "silu op id"),
                "input_tokens": [integer(args[1], "silu input")],
                "output_value": integer(args[2], "silu output value"),
                "output_token": integer(args[3], "silu output token"),
                "region": region(args, 4),
            })
        elif name == "rk_sigmoid":
            tasks.append({
                "kind": "sigmoid", "op_id": integer(args[0], "sigmoid op id"),
                "input_tokens": [integer(args[1], "sigmoid input")],
                "output_value": integer(args[2], "sigmoid output value"),
                "output_token": integer(args[3], "sigmoid output token"),
                "region": region(args, 4),
            })
        elif name == "rk_ew_mul":
            tasks.append({
                "kind": "mul", "op_id": integer(args[0], "mul op id"),
                "group_op_id": integer(args[1], "mul group op id"),
                "group_kind": str(getattr(args[2], "value", args[2])),
                "input_tokens": [integer(args[3], "left"),
                                 integer(args[4], "right")],
                "input_values": [integer(args[5], "left value"),
                                 integer(args[6], "right value")],
                "output_value": integer(args[7], "mul output value"),
                "output_token": integer(args[8], "mul output token"),
                "region": region(args, 9),
                "input_broadcasts": [
                    str(getattr(args[13], "value", args[13])),
                    str(getattr(args[14], "value", args[14]))],
            })
        elif name == "rk_ew_add":
            tasks.append({
                "kind": "add", "op_id": integer(args[0], "add op id"),
                "input_tokens": [integer(args[1], "left"),
                                 integer(args[2], "right")],
                "input_values": [integer(args[3], "left value"),
                                 integer(args[4], "right value")],
                "output_value": integer(args[5], "add output value"),
                "output_token": integer(args[6], "add output token"),
                "region": region(args, 7),
                "input_broadcasts": [
                    str(getattr(args[11], "value", args[11])),
                    str(getattr(args[12], "value", args[12]))],
            })
        elif name == "rk_store":
            tasks.append({
                "kind": "store",
                "source_token": integer(args[0], "store source token"),
                "destination_token": integer(args[1], "store destination token"),
                "output_value": integer(args[2], "store output value"),
                "source_region": region(args, 3),
                "destination_region": region(args, 7),
            })
        elif name.startswith("rk_"):
            reject(RK_E_UNSUPPORTED,
                   "a target operation has no task consumer", stage="rewrite",
                   actual=name)
    return tasks


def _full_region(region: dict[str, Any], label: str) -> dict[str, int]:
    offsets = [_const(value) for value in region["offsets"]]
    extents = [int(value) for value in region["extents"]]
    shape = [int(value) for value in region["buffer"].shape]
    if offsets != [0] * len(shape) or extents != shape:
        reject(RK_E_UNSUPPORTED,
               f"{label} must be a whole-buffer region", stage="rewrite",
               actual={"offsets": [str(x) for x in region["offsets"]],
                       "extents": extents, "shape": shape})
    return {"row": 0, "col": 0, "rows": shape[0], "cols": shape[1]}


def _classify_store_load(load: Any, target_shape: list[int],
                         variables: list[Any]) -> str:
    """Classify one actual store load as full-tile, scalar or column broadcast."""
    shape = [int(v) for v in load.buffer.shape]
    indices = [str(index) for index in load.indices]
    if shape == target_shape and indices == [str(var) for var in variables]:
        return "none"
    if not shape and not indices:
        return "scalar"
    if shape == [target_shape[1]] and indices == [str(variables[1])]:
        return "channel"
    reject(RK_E_SHAPE,
           "an elementwise load must be the full tile, a rank-0 scalar or "
           "a column-indexed rank-1 value", stage="rewrite",
           actual={"buffer": str(load.buffer.name), "shape": shape,
                   "indices": indices})


def _prove_loop_store(prim_func: Any, store: Any) -> tuple[dict[str, int], list[Any]]:
    """Prove a source BufferStore is the whole elementwise tile loop."""
    tir = _tir()
    loops = [node for _, node in _walk(prim_func.body)
             if isinstance(node, tir.For) and
             any(inner.same_as(store) for _, inner in _walk(node.body))]
    serial = [loop for loop in loops if loop.thread_binding is None]
    if len(serial) != 2:
        reject(RK_E_UNSUPPORTED,
               "an elementwise write must be the body of exactly two tile loops",
               stage="rewrite", actual=len(serial))
    target_shape = [int(v) for v in store.buffer.shape]
    variables = [loop.loop_var for loop in serial]
    for loop in serial:
        try:
            kind = int(loop.kind)
        except Exception:
            kind = -1
        if kind != int(tir.ForKind.PARALLEL):
            reject(RK_E_UNSUPPORTED,
                   "an elementwise write must use parallel tile loops",
                   stage="rewrite", actual=kind)
        if _const(loop.min) not in (0, None) or \
                [int(v) for v in [loop.extent]] != [target_shape[serial.index(loop)]]:
            reject(RK_E_SHAPE,
                   "an elementwise tile loop must cover the whole buffer",
                   stage="rewrite", actual=str(loop.extent))
        step = getattr(loop, "step", None)
        if step is not None and _const(step) != 1:
            reject(RK_E_UNSUPPORTED,
                   "an elementwise tile loop must step by one",
                   stage="rewrite")
    if [str(index) for index in store.indices] != [str(var) for var in variables]:
        reject(RK_E_SHAPE,
               "an elementwise store must write the tile-loop coordinates",
               stage="rewrite",
               actual=[str(x) for x in store.indices])
    kinds = [_classify_store_load(load, target_shape, variables)
             for load in _store_loads(store.value)]
    if "none" not in kinds:
        reject(RK_E_UNSUPPORTED,
               "an elementwise write must read at least one full-tile value",
               stage="rewrite", actual=kinds)
    if sum(kind != "none" for kind in kinds) > 1:
        reject(RK_E_UNSUPPORTED,
               "an elementwise write admits at most one broadcast operand",
               stage="rewrite", actual=kinds)
    return ({"row": 0, "col": 0,
             "rows": target_shape[0], "cols": target_shape[1]},
            variables)


def rewrite_ffn_tir(prim_func: Any, *, profile_hash: str = "",
                    name: str = "rknpu-ffn-dataflow") -> DataflowRewrite:
    """Rewrite actual statements into region-carrying ordered target calls."""
    from .. import ir
    params = list((getattr(prim_func, "buffer_map", None) or {}).values())
    gemms = [node for _, node in _walk(prim_func.body)
             if type(node).__name__ == "Evaluate" and
             _op_name(node.value) == "tl.tileop.gemm"]
    copies = [node for _, node in _walk(prim_func.body)
              if type(node).__name__ == "Evaluate" and
              _op_name(node.value) == "tl.tileop.copy"]
    stores = [node for _, node in _walk(prim_func.body)
              if type(node).__name__ == "BufferStore"]
    tir = _tir()
    thread_loops = [node for _, node in _walk(prim_func.body)
                    if isinstance(node, tir.For) and
                    node.thread_binding is not None]
    if len(gemms) not in {1, 2, 3} or len(stores) < 1 or \
            len(thread_loops) not in {1, 2}:
        reject(RK_E_UNSUPPORTED,
               "the local dataflow rewrite requires one to three GEMMs, their "
               "explicit stores and one launch", stage="rewrite")

    tokens: list[Any] = []
    token_names: list[str] = []

    def token_for(buffer: Any) -> int:
        for index, candidate in enumerate(tokens):
            if _same_buffer(candidate, buffer):
                return index
        token = len(tokens)
        tokens.append(buffer)
        token_names.append(str(buffer.name))
        return token

    for param in params:
        token_for(param)
    last_write_kind: dict[int, str] = {}
    cast_source: dict[int, int] = {}
    for _, node in _walk(prim_func.body):
        if type(node).__name__ == "Evaluate":
            call_name = _op_name(node.value)
            if call_name == "tl.tileop.gemm":
                regions = _region_exprs(node.value)
                last_write_kind[token_for(regions[2]["buffer"])] = "gemm"
            elif call_name == "tl.tileop.fill":
                regions = _region_exprs(node.value)
                last_write_kind[token_for(regions[0]["buffer"])] = "fill"
        elif type(node).__name__ == "BufferStore":
            token = token_for(node.buffer)
            if type(node.value).__name__ == "Cast" and \
                    type(node.value.value).__name__ == "BufferLoad":
                last_write_kind[token] = "cast"
                cast_source[token] = token_for(node.value.value.buffer)
            else:
                last_write_kind[token] = "value"
    gemm_regions = [_region_exprs(node.value) for node in gemms]
    a_stage_tokens = {token_for(regions[0]["buffer"])
                      for regions in gemm_regions}
    b_stage_tokens = {token_for(regions[1]["buffer"])
                      for regions in gemm_regions}
    param_roles: dict[int, str] = {}
    copy_order: list[int] = []
    residual_params: list[int] = []
    broadcast_params: list[int] = []
    output_params: list[Any] = []
    output_pairs: list[tuple[int, int]] = []
    output_source_tokens: set[int] = set()
    for node in copies:
        regions = _region_exprs(node.value)
        src, dst = regions[0]["buffer"], regions[1]["buffer"]
        source_param = next((p for p in params if _same_buffer(p, src)), None)
        dest_param = next((p for p in params if _same_buffer(p, dst)), None)
        if source_param is not None:
            token = token_for(source_param)
            if token not in copy_order:
                copy_order.append(token)
            role = ("constant" if token_for(dst) in b_stage_tokens
                    else "input")
            previous = param_roles.get(token)
            if previous is not None and previous != role:
                reject(RK_E_UNSUPPORTED,
                       "one parameter is used in incompatible operand roles",
                       stage="rewrite", actual=token_names[token])
            param_roles[token] = role
        elif dest_param is not None:
            if not any(_same_buffer(dest_param, item)
                       for item in output_params):
                output_params.append(dest_param)
            source_token = token_for(src)
            output_token = token_for(dest_param)
            output_pairs.append((source_token, output_token))
            output_source_tokens.add(source_token)
            param_roles[output_token] = "output"
    for store in stores:
        if type(store.value).__name__ not in {"Add", "Mul"}:
            continue
        destination_shape = [int(v) for v in store.buffer.shape]
        for load in _store_loads(store.value):
            source_param = next((p for p in params
                                 if _same_buffer(p, load.buffer)), None)
            if source_param is None:
                continue
            token = token_for(source_param)
            load_shape = [int(v) for v in load.buffer.shape]
            if load_shape == destination_shape:
                param_roles.setdefault(token, "input")
                if token not in residual_params:
                    residual_params.append(token)
            elif not load_shape or load_shape == [destination_shape[1]]:
                param_roles[token] = "constant"
                if token not in broadcast_params:
                    broadcast_params.append(token)
            else:
                reject(RK_E_SHAPE,
                       "a binary parameter load is neither full-tile nor a "
                       "scalar/column broadcast", stage="rewrite",
                       actual={"buffer": token_names[token],
                               "shape": load_shape,
                               "output": destination_shape})
    primary_input = None
    for regions in gemm_regions:
        stage = token_for(regions[0]["buffer"])
        for node in copies:
            c = _region_exprs(node.value)
            if token_for(c[1]["buffer"]) != stage:
                continue
            source_param = next((p for p in params
                                 if _same_buffer(p, c[0]["buffer"])), None)
            if source_param is not None:
                primary_input = token_for(source_param)
                param_roles[primary_input] = "input"
                break
        if primary_input is not None:
            break
    if not output_params:
        reject(RK_E_SHAPE, "the source has no output copy", stage="rewrite")
    ordered_external: list[int] = []
    if primary_input is not None:
        ordered_external.append(primary_input)
    for token in copy_order:
        if param_roles.get(token) == "constant":
            ordered_external.append(token)
    for token in broadcast_params:
        if token not in ordered_external:
            ordered_external.append(token)
    output_tokens = [token_for(param) for param in output_params]
    ordered_external.extend(output_tokens)
    for token, role in param_roles.items():
        if token not in ordered_external and role == "input":
            ordered_external.append(token)

    values: list[Any] = []
    value_by_token: dict[int, int] = {}
    token_values: dict[int, int] = {}
    ops: list[Any] = []
    calls: list[Any] = []
    zero_values: set[int] = set()
    final_output_by_token: dict[int, int] = {}

    def add_value(buffer: Any, role: str, producer: int | None) -> int:
        token = token_for(buffer)
        if token in final_output_by_token:
            value_id = final_output_by_token[token]
            value_by_token[token] = value_id
            return value_id
        if token in value_by_token:
            return value_by_token[token]
        shape = tuple(int(v) for v in buffer.shape)
        axes = (() if len(shape) == 0 else
                ("col",) if len(shape) == 1 else ("row", "col"))
        vid = len(values)
        values.append(ir.Value(
            vid, token_names[token],
            ir.TensorType("float16", shape, axes=axes),
            producer, role))
        value_by_token[token] = vid
        return vid

    for token in ordered_external:
        token_values[token] = add_value(tokens[token], param_roles[token], None)
    external_tokens = dict(token_values)
    final_output_by_token.clear()
    for source_token, output_binding_token in output_pairs:
        token = source_token
        seen: set[int] = set()
        while last_write_kind.get(token) == "cast":
            if token in seen or token not in cast_source:
                reject(RK_E_SHAPE,
                       "a cast alias cycle prevents output attribution",
                       stage="rewrite")
            seen.add(token)
            token = cast_source[token]
        final_output_by_token[token] = token_values[output_binding_token]

    contract_tol = {"atol": 8e-3, "rtol": 8e-3}

    def add_op(opcode: str, inputs: tuple[int, ...], output: int,
               points: tuple[str, ...]) -> int:
        op_id = len(ops)
        contract = ir.NumericContract(
            accumulator_dtype="float16",
            accumulation="ordered_k_tiles_with_f16_writeback",
            narrowing_points=points, approximate=False,
            tolerance=contract_tol)
        attrs = {"numeric_contract": contract.to_dict()}
        if opcode == "matmul":
            attrs.update({"b_layout": "kn",
                          "source_accumulator_dtype": "float16",
                          "source_k_rounding": "explicit_per_k_fp16",
                          "source_k_loop": None, "source_mn_tiling": None,
                          "source_narrowing_points": [
                              "gate_accumulator", "up_accumulator", "silu",
                              "act_mul", "projection_accumulator",
                              "residual_add"]})
        ops.append(ir.Op(op_id, opcode, inputs, (output,), attrs,
                         f"source:{opcode}:{op_id}"))
        values[output] = ir.Value(
            values[output].id, values[output].name, values[output].type,
            op_id, values[output].role)
        return op_id

    def emit(name: str, *args: Any) -> None:
        calls.append(tir.Evaluate(tir.call_extern(
            "void", name, *(_imm(value) if type(value) is int else value
                            for value in args))))

    def token_value(token: int, role: str) -> int:
        value = token_values.get(token)
        if value is None:
            reject(RK_E_SHAPE, f"{role} has no established value",
                   stage="rewrite", actual=token_names[token])
        return value

    if len(gemms) == 1:
        matmul_labels = ("gemm_accumulator",)
    elif len(gemms) == 3:
        matmul_labels = ("gate_accumulator", "up_accumulator",
                         "projection_accumulator")
    elif len(stores) >= 5:
        matmul_labels = ("gemm1_accumulator", "gemm2_accumulator")
    else:
        matmul_labels = ("gemm0_accumulator", "gemm1_accumulator")
    matmul_index = 0
    sigmoid_groups: dict[int, int] = {}
    for _, node in _walk(prim_func.body):
        if type(node).__name__ == "Evaluate":
            call = node.value
            name = _op_name(call)
            if name == "tl.tileop.fill":
                regions = _region_exprs(call)
                region = _full_region(regions[0], "fill")
                buffer = regions[0]["buffer"]
                value = _const(call.args[1]) if len(call.args) > 1 else None
                if value is None:
                    reject(RK_E_UNSUPPORTED,
                           "a local fill value must be static", stage="rewrite")
                value_id = add_value(buffer, "temporary", None)
                if value != 0:
                    reject(RK_E_NUMERIC_CONTRACT,
                           "the local dataflow rewrite only consumes zero fills",
                           stage="rewrite", actual=value)
                zero_values.add(value_id)
                emit("rk_fill", value_id, token_for(buffer), value,
                     region["row"], region["col"],
                     region["rows"], region["cols"])
                continue
            if name == "tl.tileop.copy":
                regions = _region_exprs(call)
                source_region = _full_region(regions[0], "copy source")
                destination_region = _full_region(regions[1], "copy destination")
                src, dst = regions[0]["buffer"], regions[1]["buffer"]
                if str(src.dtype) != str(dst.dtype):
                    reject(RK_E_NUMERIC_CONTRACT,
                           "copy fusion does not admit a dtype conversion; "
                           "residual must be explicitly narrowed from FP32 to "
                           "FP16 before the final T.copy",
                           stage="rewrite",
                           actual=[str(src.dtype), str(dst.dtype)])
                src_token, dst_token = token_for(src), token_for(dst)
                dest_param = next((p for p in params
                                   if _same_buffer(p, dst)), None)
                if dest_param is not None:
                    source_value = token_value(src_token, "output copy source")
                    output_value = token_value(dst_token, "output binding")
                    emit("rk_store", src_token, dst_token, output_value,
                         source_region["row"], source_region["col"],
                         source_region["rows"], source_region["cols"],
                         destination_region["row"], destination_region["col"],
                         destination_region["rows"], destination_region["cols"])
                else:
                    source_value = token_value(src_token, "copy source")
                    token_values[dst_token] = source_value
                    emit("rk_copy", src_token, dst_token, source_value,
                         source_region["row"], source_region["col"],
                         source_region["rows"], source_region["cols"],
                         destination_region["row"], destination_region["col"],
                         destination_region["rows"], destination_region["cols"])
                continue
            if name == "tl.tileop.gemm":
                regions = _region_exprs(call)
                a_token = token_for(regions[0]["buffer"])
                b_token = token_for(regions[1]["buffer"])
                a_value = token_value(a_token, "GEMM A")
                b_value = token_value(b_token, "GEMM B")
                out_buffer = regions[2]["buffer"]
                out_token = token_for(out_buffer)
                out_id = add_value(out_buffer, "temporary", None)
                if out_id not in zero_values:
                    reject(RK_E_NUMERIC_CONTRACT,
                           "a GEMM accumulator write has no preceding zero fill",
                           stage="rewrite", actual=token_names[out_token])
                label = matmul_labels[min(matmul_index, 2)]
                matmul_index += 1
                op_id = add_op("matmul", (a_value, b_value), out_id,
                               ("k_tile_boundary", label))
                token_values[out_token] = out_id
                region = _full_region(regions[2], "GEMM output")
                a_region = _full_region(regions[0], "GEMM A")
                b_region = _full_region(regions[1], "GEMM B")
                emit("rk_gemm", op_id, a_token, b_token, out_id, out_token,
                     a_region["row"], a_region["col"],
                     a_region["rows"], a_region["cols"],
                     b_region["row"], b_region["col"],
                     b_region["rows"], b_region["cols"],
                     region["row"], region["col"],
                     region["rows"], region["cols"],
                     0, int(regions[0]["extents"][1]))
                continue
        if type(node).__name__ != "BufferStore":
            continue
        store = node
        region, loop_vars = _prove_loop_store(prim_func, store)
        dest_token = token_for(store.buffer)
        value = store.value
        if type(value).__name__ == "Cast":
            load = value.value
            if type(load).__name__ != "BufferLoad" or \
                    str(load.buffer.dtype) != "float32" or \
                    str(store.buffer.dtype) != "float16":
                reject(RK_E_NUMERIC_CONTRACT,
                       "cast fusion requires an explicit FP32 residual narrowing "
                       "to FP16 before the final T.copy",
                       stage="rewrite",
                       actual={"load": type(load).__name__,
                               "load_dtype": str(getattr(load, "buffer", None).dtype
                                                 if hasattr(load, "buffer") else ""),
                               "store_dtype": str(store.buffer.dtype)})
            source_token = token_for(load.buffer)
            source_value = token_value(source_token, "cast source")
            token_values[dest_token] = source_value
            emit("rk_cast", source_token, dest_token, source_value,
                 region["row"], region["col"],
                 region["rows"], region["cols"],
                 region["row"], region["col"],
                 region["rows"], region["cols"])
            continue
        if type(value).__name__ == "Call" and "sigmoid" in str(value):
            loads = _store_loads(value)
            if len(loads) != 1:
                reject(RK_E_UNSUPPORTED,
                       "sigmoid store must read exactly one value",
                       stage="rewrite")
            if str(store.buffer.dtype) != "float16" or \
                    str(loads[0].buffer.dtype) != "float16":
                reject(RK_E_NUMERIC_CONTRACT,
                       "explicit sigmoid requires FP16 input and output",
                       stage="rewrite")
            source_token = token_for(loads[0].buffer)
            source_value = token_value(source_token, "sigmoid input")
            out_id = add_value(store.buffer, "temporary", None)
            token_values[dest_token] = out_id
            op_id = add_op("sigmoid", (source_value,), out_id,
                           ("k_tile_boundary", "sigmoid"))
            sigmoid_groups[dest_token] = op_id
            emit("rk_sigmoid", op_id, source_token, out_id, dest_token,
                 region["row"], region["col"],
                 region["rows"], region["cols"])
            continue
        if type(value).__name__ == "Mul":
            loads = _store_loads(value)
            if len(loads) != 2:
                reject(RK_E_UNSUPPORTED, "multiply must read two values",
                       stage="rewrite")
            load_kinds = [_classify_store_load(load, [region["rows"],
                                                      region["cols"]],
                                               loop_vars)
                          for load in loads]
            source_tokens = [token_for(load.buffer) for load in loads]
            source_ids = [token_value(token, "multiply source")
                          for token in source_tokens]
            out_id = add_value(store.buffer, "temporary", None)
            token_values[dest_token] = out_id
            broadcast = any(kind != "none" for kind in load_kinds)
            if _has_sigmoid(value):
                op_id = add_op("silu", (source_ids[0],), out_id,
                               ("k_tile_boundary", "silu"))
                emit("rk_silu", op_id, source_tokens[0], out_id, dest_token,
                     region["row"], region["col"],
                     region["rows"], region["cols"])
            else:
                group_op = None
                group_kind = "ew"
                for token in source_tokens:
                    if token in sigmoid_groups:
                        group_op = sigmoid_groups[token]
                        group_kind = "act"
                        break
                point = ("broadcast_mul" if broadcast else
                         "silu_mul" if group_kind == "act" else "act_mul")
                op_id = add_op(
                    "mul", tuple(source_ids), out_id,
                    ("k_tile_boundary", point))
                if group_op is None:
                    group_op = op_id
                emit("rk_ew_mul", op_id, group_op, _string(group_kind),
                     source_tokens[0], source_tokens[1],
                     source_ids[0], source_ids[1],
                     out_id, dest_token, region["row"], region["col"],
                     region["rows"], region["cols"],
                     _string(load_kinds[0]), _string(load_kinds[1]))
            continue
        if type(value).__name__ == "Add":
            loads = _store_loads(value)
            if len(loads) != 2:
                reject(RK_E_UNSUPPORTED, "add must read two values",
                       stage="rewrite")
            load_kinds = [_classify_store_load(load, [region["rows"],
                                                      region["cols"]],
                                               loop_vars)
                          for load in loads]
            source_tokens = [token_for(load.buffer) for load in loads]
            source_ids = [token_value(token, "add source")
                          for token in source_tokens]
            if dest_token not in output_source_tokens:
                reject(RK_E_UNSUPPORTED,
                       "the binary write is not the source of an output copy",
                       stage="rewrite", actual=token_names[dest_token])
            out_id = final_output_by_token.get(dest_token)
            if out_id is None:
                reject(RK_E_UNSUPPORTED,
                       "the binary write has no output binding",
                       stage="rewrite", actual=token_names[dest_token])
            token_values[dest_token] = out_id
            broadcast = any(kind != "none" for kind in load_kinds)
            op_id = add_op(
                "add", tuple(source_ids), out_id,
                ("k_tile_boundary",
                 "broadcast_add" if broadcast else "residual_add"))
            emit("rk_ew_add", op_id, source_tokens[0], source_tokens[1],
                 source_ids[0], source_ids[1],
                 out_id, dest_token, region["row"], region["col"],
                 region["rows"], region["cols"],
                 _string(load_kinds[0]), _string(load_kinds[1]))
            continue
        reject(RK_E_UNSUPPORTED,
               "a source write is not consumed by the local dataflow rewrite: "
               + (_op_name(value) or str(value)), stage="rewrite",
               actual={"kind": type(value).__name__,
                       "expression": str(value)})

    target = tir.PrimFunc(
        prim_func.params, tir.SeqStmt(calls), prim_func.ret_type,
        prim_func.buffer_map, prim_func.attrs)
    tasks = extract_rk_tasks(target)
    if not any(task["kind"] == "store" for task in tasks):
        reject(RK_E_SHAPE,
               "the source output copy did not produce an rk_store task",
               stage="rewrite")
    module = ir.Module(
        name=name, values=tuple(values), ops=tuple(ops),
        inputs=tuple(value.id for value in values
                     if value.role == "input"),
        outputs=tuple(task["output_value"] for task in tasks
                      if task["kind"] == "store"),
        profile_hash=profile_hash, source_hash=_digest(prim_func.script()))
    stage = Stage("rewrite-source-statements", "transform",
                  _digest(target.script()),
                  {"calls": len(calls), "ops": len(ops),
                   "buffer_tokens": len(tokens)})
    return DataflowRewrite(
        source=prim_func, target=target, module=module, tasks=tasks,
        facts={"external_tokens": dict(external_tokens)},
        stages=(stage, stage))


def rewrite_full_logical_tir(prim_func: Any, *, options: Any,
                             tile_m: int = 256, tile_n: int = 256,
                             block_k: int = 256, order: str = "m_major",
                             b_is_input: bool = False,
                             max_tasks: int | None = None) -> FullLogicalRewrite:
    """Run both visible conversion stages and existing task instantiation."""
    scheduled = schedule_full_logical_source(
        prim_func, options=options, tile_m=tile_m, tile_n=tile_n,
        block_k=block_k, order=order, b_is_input=b_is_input)
    target, lower_stage = lower_scheduled_full_logical(
        scheduled.scheduled, facts=scheduled.facts)
    instantiated, tasks, instantiate_stage = pass_instantiate(
        target, scheduled.facts, max_tasks=max_tasks)
    return FullLogicalRewrite(
        source=prim_func, scheduled=scheduled.scheduled, target=target,
        instantiated=instantiated, facts=scheduled.facts, tasks=tasks,
        stages=(scheduled.stage, lower_stage, instantiate_stage))
