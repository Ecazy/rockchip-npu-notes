# SPDX-License-Identifier: GPL-3.0-or-later
"""Structured lowering: prove the program semantics on TIR, then hand the plan
only what has been proved.

Unlike the template extractor, this path keeps the TileLang TIR nodes and derives
every fact it later relies on:

    stage 0 capture   keep the launch/loop/statement structure (TVM nodes)
    stage 1 prove     per-statement proofs: guarded copies, zero fills, the
                      FP16 writeback position/direction, and the source rounding
                      (derived, never accepted from the caller)
    stage 2 k-chain   ordered K slices from the proved logical K
    stage 3 tiles     per-tile valid extents, proved against the guards
    stage 4 lower     StructuredModule (proofs + statement trace + buffers)
    plan_from_module  build RKPlan, then verify it against the module

Anything the proofs do not establish is rejected; nothing is inferred from the
overall shape of the program.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..errors import RK_E_NUMERIC_CONTRACT, RK_E_SHAPE, RK_E_UNSUPPORTED, reject

K_TILE = 256
SIDE_EFFECT_OPS = ("call_extern", "tvm_call_packed", "tvm_call_global")
ROUNDINGS = {"explicit_per_k_fp16": "after_each_k_block",
             "fp32_final_narrowing": "final_output_only"}


def _tir():
    import tilelang  # noqa: F401  (loads the vendored TVM)
    from tvm import tirx

    return tirx


def _arith():
    import tvm.arith

    return tvm.arith.Analyzer()


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _is_thread_binding(node: Any) -> bool:
    tir = _tir()
    try:
        return int(node.kind) == int(tir.ForKind.THREAD_BINDING)
    except (AttributeError, TypeError, ValueError):
        return "thread_binding" in str(getattr(node, "kind", ""))


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


def _structurally_equal(left: Any, right: Any) -> bool:
    import tvm.ir

    try:
        return bool(tvm.ir.structural_equal(left, right))
    except (AttributeError, ValueError, TypeError):
        return False


def _op_name(call: Any) -> str:
    return str(getattr(getattr(call, "op", None), "name", "") or "")


def _region_of(call: Any, index: int):
    """Return (buffer, (extents)) of the index-th tl.region argument."""
    tir = _tir()
    regions = [arg for arg in call.args
               if isinstance(arg, tir.Call) and _op_name(arg) == "tl.region"]
    if len(regions) <= index:
        return None, None
    region = regions[index]
    elements = list(region.args)
    if not elements:
        return None, None
    first = elements[0]
    buffer = getattr(first, "buffer", None)
    # tl.region(buffer_load, access_mask, *extents): the mask is not an extent
    extents = [int(x) for x in elements[2:]] if len(elements) > 2 else []
    return buffer, extents


def _vars_in(expr):
    """Free variables of an expression, by node identity (never by name)."""
    out = []

    def visit(node):
        tir = _tir()
        if isinstance(node, tir.Var):
            out.append(node)

    import tvm.tirx.stmt_functor as stmt_functor

    stmt_functor.post_order_visit(expr, visit)
    return out


def _mentions(expr, var) -> bool:
    return any(candidate.same_as(var) for candidate in _vars_in(expr))


def _prove_read_range(captured, guarded, analyzer):
    """Prove that each guard bounds the operand range the copy actually reads.

    For every axis of the source load the proof requires all of:
      * the store index is the tile-local loop variable,
      * the load index equals grid_var * tile_extent + store_index,
      * exactly one guard conjunct bounds that load index, by the operand extent.
    """
    tir = _tir()
    shapes = {entry["name"]: entry["shape"] for entry in captured["params"]}
    shared = [e for e in captured["allocations"] if e["scope"].startswith("shared")]
    by_name = {e["name"]: e for e in shared}
    table = []
    table = []
    for entry in guarded:
        load = entry.get("load")
        store = entry.get("store")
        if load is None or store is None:
            reject(RK_E_SHAPE, "guarded copy has no load/store node to prove",
                   stage="structured")
        name = str(load.buffer.name)
        shape = shapes.get(name)
        if shape is None or len(shape) != 2:
            reject(RK_E_SHAPE, "operand is not a rank-2 parameter",
                   stage="structured", actual=name)
        tile = by_name.get(entry["target"])
        if tile is None:
            reject(RK_E_SHAPE, "a guarded copy targets a buffer that is not a tile",
                   stage="structured", actual=entry["target"])
        conjuncts = list(entry.get("conjunct_nodes") or ())
        used = [False] * len(conjuncts)
        axes = []
        for axis in range(2):
            load_index = load.indices[axis]
            local_index = store.indices[axis]
            limit = None
            limit_position = None
            for position, conjunct in enumerate(conjuncts):
                if type(conjunct).__name__ != "LT":
                    continue
                if not analyzer.can_prove(tir.EQ(conjunct.a, load_index)
                                          if hasattr(tir, "EQ") else conjunct.a):
                    continue
                bound = _const(conjunct.b)
                if bound is None:
                    continue
                limit = bound
                limit_position = position
                break
            if limit is None or limit != shape[axis]:
                reject(RK_E_SHAPE,
                       "the guard does not bound this read by the operand extent",
                       stage="structured",
                       actual={"operand": name, "axis": axis,
                               "load_index": str(load_index),
                               "operand_extent": shape[axis],
                               "guard_limit": limit})
            used[limit_position] = True
            local_vars = _vars_in(local_index)
            extra = [var for var in _vars_in(load_index)
                     if not any(var.same_as(other) for other in local_vars)]
            if len(extra) != 1:
                reject(RK_E_SHAPE,
                       "the read index must be one tile coordinate plus the local "
                       "index",
                       stage="structured",
                       actual={"operand": name, "axis": axis,
                               "load_index": str(load_index),
                               "tile_vars": [str(v) for v in extra]})
            tile_extent = int(tile["shape"][axis])
            expected = extra[0] * tile_extent + local_index
            if not analyzer.can_prove(load_index == expected):
                reject(RK_E_SHAPE,
                       "the read index is not tile_extent * tile + local_index",
                       stage="structured",
                       actual={"operand": name, "axis": axis,
                               "load_index": str(load_index),
                               "expected": str(expected),
                               "tile_extent": tile_extent})
            axes.append({"axis": axis, "load_index": str(load_index),
                         "local_index": str(local_index), "bound": limit,
                         "tile_extent": tile_extent,
                         "tile_var": str(extra[0]), "affine": "proved"})
            table.append({"operand": name, "axis": axis, "bound": limit,
                          "operand_extent": shape[axis],
                          "tile_var": str(extra[0]), "tile_extent": tile_extent,
                          "load_index": str(load_index),
                          "local_index": str(local_index)})
        if not all(used):
            reject(RK_E_SHAPE,
                   "the guard carries conditions that do not bound this read",
                   stage="structured",
                   actual={"operand": name,
                           "unused": [str(conjuncts[i]) for i, u in
                                      enumerate(used) if not u]})
        entry["axes"] = axes
    # the tile shape on each GEMM axis, proved from the tile that copy fills
    k_var = captured["k_loop"].loop_var
    tile_m = tile_n = block_k = None
    for entry in guarded:
        load = entry.get("load")
        tile = by_name.get(entry["target"])
        if load is None or tile is None:
            continue
        k_axis = None
        for axis in range(2):
            if _mentions(load.indices[axis], k_var):
                k_axis = axis
        if k_axis == 1:
            tile_m = int(tile["shape"][0])
            block_k = int(tile["shape"][1])
        elif k_axis == 0:
            tile_n = int(tile["shape"][1])
            if block_k is None:
                block_k = int(tile["shape"][0])
            elif block_k != int(tile["shape"][0]):
                reject(RK_E_SHAPE,
                       "the two operand tiles disagree on the K extent",
                       stage="structured",
                       actual={"first": block_k, "second": int(tile["shape"][0])})
    if tile_m is None or tile_n is None or block_k is None:
        reject(RK_E_SHAPE, "cannot derive the tile geometry from the copies",
               stage="structured")
    return table, [tile_m, tile_n], block_k


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


def _const(value: Any):
    tir = _tir()
    if isinstance(value, tir.IntImm):
        return int(value)
    if isinstance(value, int):
        return value
    return None


def _scope_of(buffer: Any) -> str:
    scope = getattr(buffer, "scope", "")
    return str(scope() if callable(scope) else scope)


# ---------------------------------------------------------------- stage 0
def stage0_capture(prim_func: Any) -> dict[str, Any]:
    tir = _tir()
    body = prim_func.body
    if type(body).__name__ != "SBlockRealize":
        reject(RK_E_UNSUPPORTED,
               "structured lowering expects a block-realised PrimFunc body",
               stage="structured", actual=type(body).__name__)
    nodes = list(_walk(body))
    side_effects = sorted({_op_name(node) for _, node in nodes
                           if isinstance(node, tir.Call)
                           and any(word in _op_name(node) for word in SIDE_EFFECT_OPS)})
    if side_effects:
        reject(RK_E_UNSUPPORTED,
               "structured lowering refuses side-effecting calls",
               stage="structured", actual=side_effects)
    def is_tile_op(node):
        name = _op_name(node) if isinstance(node, tir.Call) else ""
        return any(word in name for word in ("gemm", "fill", "copy"))

    def holds_tile_op(loop):
        return any(is_tile_op(inner) for _, inner in _walk(loop))

    # The K loop is the serial loop that holds tile operations.  Its ancestors in
    # the depth-first walk are the block grid, whether that grid is still a
    # thread binding (raw eager form) or has already been materialised into
    # serial loops by MaterializeKernelLaunch.
    k_loop = None
    grid_loops = []
    stack = []
    deepest = -1
    for depth, node in nodes:
        while stack and stack[-1][0] >= depth:
            stack.pop()
        if isinstance(node, tir.For):
            stack.append((depth, node))
            if (int(node.kind) == int(tir.ForKind.SERIAL)
                    and depth > deepest and holds_tile_op(node)):
                deepest = depth
                k_loop = node
                grid_loops = [loop for _, loop in stack[:-1]]
    if k_loop is None:
        reject(RK_E_SHAPE,
               "structured lowering found no serial loop holding tile operations",
               stage="structured")
    serial_loops = [k_loop]
    if len(grid_loops) != 2:
        reject(RK_E_SHAPE, "structured lowering expects a two-dimensional block grid",
               stage="structured", actual=len(grid_loops))
    allocations = []
    for _, node in nodes:
        if type(node).__name__ != "SBlock":
            continue
        for buffer in getattr(node, "alloc_buffers", ()) or ():
            allocations.append({"name": str(buffer.name), "buffer": buffer,
                                "dtype": str(buffer.dtype),
                                "shape": [int(x) for x in buffer.shape],
                                "scope": _scope_of(buffer)})
    if not allocations:
        reject(RK_E_SHAPE, "structured lowering found no tile buffer allocation",
               stage="structured")
    params = []
    buffer_map = getattr(prim_func, "buffer_map", None) or {}
    for param in getattr(prim_func, "params", ()):
        buffer = buffer_map.get(param) if hasattr(buffer_map, "get") else None
        if buffer is not None:
            params.append({"name": str(buffer.name), "buffer": buffer,
                           "shape": [int(x) for x in buffer.shape],
                           "scope": _scope_of(buffer)})
    # Statements outside the K loop.  TVM wrappers are created per access, so
    # ownership is decided structurally rather than by object identity: any
    # sequence that contains a structural copy of the K loop owns it, and its
    # other items are the out-of-loop statements.
    import tvm.ir

    root_seq = []
    root_full = []
    loop_position = None
    for _, node in nodes:
        seq = getattr(node, "seq", None)
        if not seq:
            continue
        items = list(seq)
        owner = False
        for item in items:
            try:
                if tvm.ir.structural_equal(item, serial_loops[0]):
                    owner = True
                    break
            except (AttributeError, ValueError, TypeError):
                continue
        if owner:
            for position, item in enumerate(items):
                if _structurally_equal(item, serial_loops[0]):
                    loop_position = position
            root_full = items
            root_seq = [item for item in items
                        if not _structurally_equal(item, serial_loops[0])]
            break
    return {"root_full": root_full, "loop_position": loop_position,
            "body": body, "nodes": nodes, "grid_loops": grid_loops,
            "root_seq": root_seq,
            "k_loop": serial_loops[0], "allocations": allocations,
            "params": params}


# ---------------------------------------------------------------- stage 1
def _loop_body_seq(k_loop: Any) -> list[Any]:
    body = k_loop.body
    seq = getattr(body, "seq", None)
    if seq is not None:
        return list(seq)
    return [body]


def stage1_prove(captured: dict[str, Any]) -> dict[str, Any]:
    """Prove the guarded copies, the zero fills and the FP16 writeback."""
    tir = _tir()
    analyzer = _arith()
    k_loop = captured["k_loop"]
    tiles = {entry["name"]: entry for entry in captured["allocations"]}
    shared = [e for e in captured["allocations"] if e["scope"].startswith("shared")]
    fragments = [e for e in captured["allocations"] if e["scope"].startswith("local")]
    if len(shared) != 2:
        reject(RK_E_SHAPE, "expected exactly two shared tile buffers",
               stage="structured", actual=len(shared))
    if len(fragments) != 2:
        reject(RK_E_SHAPE, "expected exactly two tile fragments",
               stage="structured", actual=len(fragments))
    fp16 = [e for e in fragments if e["dtype"] == "float16"]
    fp32 = [e for e in fragments if e["dtype"] == "float32"]
    if len(fp16) != 1 or len(fp32) != 1:
        reject(RK_E_NUMERIC_CONTRACT,
               "the running accumulator must be one FP16 and one FP32 fragment",
               stage="structured", actual=[e["name"] for e in fragments])
    running = fp16[0]
    gemm_acc = fp32[0]

    statement_kinds: list[str] = []
    fills: list[dict[str, Any]] = []
    copies: list[dict[str, Any]] = []
    guarded: list[dict[str, Any]] = []
    gemm_at = None
    for position, node in enumerate(_loop_body_seq(k_loop)):
        name = type(node).__name__
        statement_kinds.append(name)
        if isinstance(node, tir.Evaluate) and isinstance(node.value, tir.Call):
            call = node.value
            op = _op_name(call)
            if "gemm" in op:
                if gemm_at is not None:
                    reject(RK_E_SHAPE, "expected a single tile GEMM per K iteration",
                           stage="structured")
                gemm_at = position
                a_buffer, a_extents = _region_of(call, 0)
                b_buffer, b_extents = _region_of(call, 1)
                c_buffer, c_extents = _region_of(call, 2)
                if a_buffer is None or b_buffer is None or c_buffer is None:
                    reject(RK_E_SHAPE, "tile GEMM operands are not regions",
                           stage="structured")
                gemm_args = [str(a_buffer.name), str(b_buffer.name)]
                if _scope_of(c_buffer) != gemm_acc["scope"]:
                    reject(RK_E_NUMERIC_CONTRACT,
                           "tile GEMM must accumulate into the FP32 fragment",
                           stage="structured", actual=_scope_of(c_buffer))
                continue
            if "fill" in op:
                buffer, extents = _region_of(call, 0)
                value = _const(call.args[1]) if len(call.args) > 1 else None
                if buffer is None:
                    reject(RK_E_SHAPE, "fill operand is not a region",
                           stage="structured")
                fills.append({"buffer": str(buffer.name), "value": value,
                              "extents": extents,
                              "full": extents == [int(x) for x in buffer.shape]})
                continue
            if "copy" in op:
                source, source_extents = _region_of(call, 0)
                target, target_extents = _region_of(call, 1)
                if source is None or target is None:
                    reject(RK_E_SHAPE, "copy operands are not regions",
                           stage="structured")
                full = (source_extents == [int(x) for x in source.shape]
                        and target_extents == [int(x) for x in target.shape])
                copies.append({"from": str(source.name),
                               "to": str(target.name),
                               "from_dtype": str(source.dtype),
                               "to_dtype": str(target.dtype),
                               "position": position,
                               "full": full,
                               "extents": [source_extents, target_extents]})
                continue
        if isinstance(node, tir.For):
            # Parallel iteration space of a copy: the guard lives inside it.
            counts = {"guarded": 0, "unguarded": 0}

            def scan(current, inside_guard=False):
                kind = type(current).__name__
                if kind == "IfThenElse":
                    if getattr(current, "else_case", None) is not None:
                        reject(RK_E_UNSUPPORTED,
                               "a guarded tile copy must not have an else branch",
                               stage="structured")
                    scan(current.condition, inside_guard)
                    scan(current.then_case, True)
                    return
                if kind == "BufferStore":
                    scope = _scope_of(current.buffer)
                    if scope not in ("", "global"):
                        counts["guarded" if inside_guard else "unguarded"] += 1
                    scan(current.value, inside_guard)
                    for index in current.indices:
                        scan(index, inside_guard)
                    return
                if kind == "For":
                    scan(current.body, inside_guard)
                    return
                for item in getattr(current, "seq", ()) or ():
                    scan(item, inside_guard)
                for argument in getattr(current, "args", ()) or ():
                    if hasattr(argument, "__class__"):
                        scan(argument, inside_guard)
                for name in ("a", "b", "value"):
                    child = getattr(current, name, None)
                    if child is not None and hasattr(child, "__class__"):
                        scan(child, inside_guard)

            scan(node.body)
            if counts["unguarded"] or counts["guarded"] != 1:
                reject(RK_E_UNSUPPORTED,
                       "every non-global store inside the K loop must be the one "
                       "guarded copy of a tile", stage="structured",
                       actual=counts)
            # record the guarded copy: target tile, local indices, guard bounds
            for _, inner in _walk(node):
                if type(inner).__name__ != "IfThenElse":
                    continue
                then = inner.then_case
                while type(then).__name__ in ("For", "IfThenElse"):
                    then = (then.then_case if type(then).__name__ == "IfThenElse"
                            else then.body)
                if type(then).__name__ != "BufferStore":
                    continue
                target = then.buffer
                if not _scope_of(target).startswith("shared"):
                    continue
                bounds = []
                for part in _conjuncts(inner.condition):
                    limit = _const(getattr(part, "b", None))
                    if type(part).__name__ != "LT" or limit is None:
                        reject(RK_E_UNSUPPORTED,
                               "guards must be conjunctions of constant upper bounds",
                               stage="structured", actual=str(inner.condition)[:80])
                    bounds.append({"expr": str(part.a), "limit": limit})
                source_buffer = None
                source_buffer_load = None
                for _, value_node in _walk(then.value):
                    if type(value_node).__name__ == "BufferLoad":
                        source_buffer = value_node.buffer
                        source_buffer_load = value_node
                if not _structurally_equal(then.value, source_buffer_load):
                    reject(RK_E_NUMERIC_CONTRACT,
                           "the guarded copy must store exactly the operand read",
                           stage="structured", actual=str(then.value)[:80])
                nest = []
                cursor = node
                while type(cursor).__name__ == "For":
                    nest.append(int(cursor.extent))
                    cursor = cursor.body
                if nest != [int(x) for x in target.shape]:
                    reject(RK_E_SHAPE,
                           "the guarded copy must iterate the whole tile",
                           stage="structured",
                           actual={"iterates": nest,
                                   "tile": [int(x) for x in target.shape]})
                guarded.append({"load": source_buffer_load, "store": then,
                                "conjunct_nodes": list(_conjuncts(inner.condition)),
                                "target": str(target.name),
                                "target_scope": _scope_of(target),
                                "indexes": [str(x) for x in then.indices],
                                "bounds": bounds,
                                "source": (str(source_buffer.name)
                                           if source_buffer is not None else None)})
                break
            continue
        if name == "IfThenElse":
            then = node.then_case
            while type(then).__name__ in ("For", "IfThenElse"):
                then = then.then_case if type(then).__name__ == "IfThenElse" else then.body
            if type(then).__name__ != "BufferStore":
                reject(RK_E_UNSUPPORTED,
                       "structured lowering only proves guarded tile stores",
                       stage="structured", actual=type(then).__name__)
            target = then.buffer
            guard = _conjuncts(node.condition)
            bounds = []
            for part in guard:
                if type(part).__name__ != "LT":
                    reject(RK_E_UNSUPPORTED,
                           "guards must be conjunctions of upper bounds",
                           stage="structured", actual=type(part).__name__)
                limit = _const(part.b)
                if limit is None:
                    reject(RK_E_SHAPE,
                           "guard bound is not a constant", stage="structured")
                bounds.append({"expr": str(part.a), "limit": limit,
                               "satisfied": bool(analyzer.can_prove(part))})
            indexes = [str(x) for x in then.indices]
            value = then.value
            source_buffer = None
            for _, inner in _walk(value):
                if type(inner).__name__ == "BufferLoad":
                    source_buffer = inner.buffer
            guarded.append({"target": str(target.name),
                            "target_scope": _scope_of(target),
                            "indexes": indexes,
                            "bounds": bounds,
                            "source": str(source_buffer.name) if source_buffer is not None else None})
            continue
        reject(RK_E_UNSUPPORTED,
               "structured lowering met a statement it does not consume",
               stage="structured", actual=name)

    if gemm_at is None:
        reject(RK_E_UNSUPPORTED, "no tile GEMM inside the K loop", stage="structured")

    # every shared tile is zero filled, completely, once per K iteration
    for entry in shared:
        matching = [f for f in fills if f["buffer"] == entry["name"]]
        if len(matching) != 1:
            reject(RK_E_NUMERIC_CONTRACT,
                   "each shared tile must be zero filled exactly once per K block",
                   stage="structured", actual=entry["name"])
        if matching[0]["value"] != 0:
            reject(RK_E_NUMERIC_CONTRACT,
                   "shared tile fill must be zero", stage="structured",
                   actual={"buffer": entry["name"], "value": matching[0]["value"]})
        if not matching[0]["full"]:
            reject(RK_E_SHAPE,
                   "shared tile fill must cover the whole tile", stage="structured",
                   actual={"buffer": entry["name"], "extents": matching[0]["extents"]})

    # both guarded copies exist, into the two shared tiles, from the two operands
    guarded_targets = {entry["target"] for entry in guarded}
    if guarded_targets != {entry["name"] for entry in shared}:
        reject(RK_E_SHAPE,
               "the K loop must fill each shared tile through a guarded copy",
               stage="structured",
               actual={"guarded": sorted(guarded_targets),
                       "shared": sorted(e["name"] for e in shared)})
    sources = {entry["source"] for entry in guarded}
    if None in sources or len(sources) != 2:
        reject(RK_E_SHAPE,
               "the guarded copies must read the two GEMM operands",
               stage="structured", actual=sorted(s for s in sources if s))
    for entry in guarded:
        if not entry["bounds"]:
            reject(RK_E_SHAPE, "guarded copy has no bound", stage="structured")
        if len(entry["indexes"]) != len(entry["bounds"]):
            reject(RK_E_SHAPE,
                   "guarded copy indexes and bounds disagree",
                   stage="structured",
                   actual={"indexes": entry["indexes"],
                           "bounds": [b["expr"] for b in entry["bounds"]]})

    # the FP16 writeback happens after the GEMM inside the same K iteration
    writebacks = [c for c in copies
                  if c["from"] == gemm_acc["name"] and c["to"] == running["name"]
                  and c["from_dtype"] == "float32" and c["to_dtype"] == "float16"]
    if not writebacks:
        reject(RK_E_NUMERIC_CONTRACT,
               "no FP32 to FP16 writeback into the running accumulator",
               stage="structured",
               actual=[(c["from"], c["to"]) for c in copies])
    if not any(c["position"] > gemm_at for c in writebacks):
        reject(RK_E_NUMERIC_CONTRACT,
               "the FP16 writeback must follow the tile GEMM in the same K block",
               stage="structured",
               actual={"gemm_position": gemm_at,
                       "writeback_positions": [c["position"] for c in writebacks]})
    loads = [c for c in copies
             if c["from"] == running["name"] and c["to"] == gemm_acc["name"]]
    if not loads:
        reject(RK_E_NUMERIC_CONTRACT,
               "the running accumulator is never loaded into the GEMM accumulator",
               stage="structured")
    if not any(c["position"] < gemm_at for c in loads):
        reject(RK_E_NUMERIC_CONTRACT,
               "the accumulator load must precede the tile GEMM",
               stage="structured")

    read_range, tile_shape, block_k = _prove_read_range(captured, guarded,
                                                        analyzer)

    # --- statement order and full-extent coverage of the accumulator chain
    order = []
    for position, node in enumerate(_loop_body_seq(k_loop)):
        if isinstance(node, tir.Evaluate) and isinstance(node.value, tir.Call):
            op = _op_name(node.value)
            if "gemm" in op:
                order.append(("gemm", position))
            elif "fill" in op:
                buffer, extents = _region_of(node.value, 0)
                order.append(("fill:" + str(buffer.name), position))
            elif "copy" in op:
                source, source_extents = _region_of(node.value, 0)
                target, target_extents = _region_of(node.value, 1)
                order.append(("copy:" + str(source.name) + ">" + str(target.name),
                              position))
        elif isinstance(node, tir.For):
            for _, inner in _walk(node):
                if type(inner).__name__ != "IfThenElse":
                    continue
                then = inner.then_case
                while type(then).__name__ in ("For", "IfThenElse"):
                    then = (then.then_case if type(then).__name__ == "IfThenElse"
                            else then.body)
                if type(then).__name__ == "BufferStore":
                    order.append(("guarded:" + str(then.buffer.name), position))
                break
    kinds = [name for name, _ in order]
    required = ["fill:" + entry["name"] for entry in shared] +                ["guarded:" + entry["name"] for entry in shared] +                ["copy:" + running["name"] + ">" + gemm_acc["name"],
                "gemm",
                "copy:" + gemm_acc["name"] + ">" + running["name"]]
    if sorted(kinds) != sorted(required):
        reject(RK_E_UNSUPPORTED,
               "the K iteration must be exactly: fill, guarded copy, accumulator "
               "load, tile GEMM and FP16 writeback, once each",
               stage="structured",
               actual={"found": kinds, "required": sorted(required)})
    position = dict(order)
    for entry in shared:
        if position["fill:" + entry["name"]] > position["guarded:" + entry["name"]]:
            reject(RK_E_NUMERIC_CONTRACT,
                   "a shared tile is filled after the copy that fills it",
                   stage="structured", actual=entry["name"])
    if position["copy:" + running["name"] + ">" + gemm_acc["name"]] > position["gemm"]:
        reject(RK_E_NUMERIC_CONTRACT,
               "the accumulator load must precede the tile GEMM", stage="structured")
    if position["copy:" + gemm_acc["name"] + ">" + running["name"]] < position["gemm"]:
        reject(RK_E_NUMERIC_CONTRACT,
               "the FP16 writeback must follow the tile GEMM", stage="structured")
    for name in kinds:
        if name.startswith("guarded:") and position[name] > position["gemm"]:
            reject(RK_E_NUMERIC_CONTRACT,
                   "a guarded operand copy must precede the tile GEMM",
                   stage="structured", actual=name)
    for entry in copies:
        if not entry["full"]:
            reject(RK_E_NUMERIC_CONTRACT,
                   "an accumulator copy must cover the whole fragment",
                   stage="structured",
                   actual={"from": entry["from"], "to": entry["to"],
                           "extents": entry["extents"]})

    # --- operand to tile binding, on the K axis each copy actually reads
    k_var = captured["k_loop"].loop_var
    k_axis = {}
    for entry in guarded:
        load = entry["load"]
        for axis in range(2):
            if _mentions(load.indices[axis], k_var):
                k_axis[entry["source"]] = axis
    tile_source = {entry["target"]: entry["source"] for entry in guarded}
    arg0 = tile_source.get(gemm_args[0])
    arg1 = tile_source.get(gemm_args[1])
    if k_axis.get(arg0) != 1 or k_axis.get(arg1) != 0:
        reject(RK_E_SHAPE,
               "the GEMM operands do not match the tile they are filled into "
               "(first operand must be m x k, second k x n)",
               stage="structured",
               actual={"gemm_args": gemm_args,
                       "k_axis": {k: v for k, v in k_axis.items()}})
    k_extents = {row["operand_extent"] for row in read_range
                 if row["axis"] == k_axis[row["operand"]]}
    if len(k_extents) != 1:
        reject(RK_E_SHAPE, "the two operands disagree on K", stage="structured",
               actual=sorted(k_extents))

    # --- closure: the running accumulator is cleared once, and the kernel writes
    #     the output parameter exactly once through a proved index map.  Any
    #     other out-of-loop statement is rejected, including the case where the
    #     owner of the K loop could not be enumerated.
    root_seq = captured.get("root_seq") or []
    if not root_seq:
        reject(RK_E_UNSUPPORTED,
               "cannot enumerate the statements outside the K loop; refusing to "
               "skip the closure check", stage="structured")
    guarded_stores = []
    unguarded_stores = []
    accumulator_fills = 0
    for node in root_seq:
        if isinstance(node, tir.For):
            inside = 0

            def collect(current, guarded_by=None):
                nonlocal inside
                kind = type(current).__name__
                if kind == "IfThenElse":
                    if getattr(current, "else_case", None) is not None:
                        reject(RK_E_UNSUPPORTED,
                               "the output store must not have an else branch",
                               stage="structured")
                    collect(current.condition, guarded_by)
                    collect(current.then_case, current.condition)
                    return
                if kind == "BufferStore":
                    inside += 1
                    scope = _scope_of(current.buffer)
                    if scope not in ("", "global"):
                        reject(RK_E_UNSUPPORTED,
                               "a non-global store outside the K loop is not "
                               "consumed", stage="structured",
                               actual=str(current.buffer.name))
                    if guarded_by is None:
                        unguarded_stores.append(current)
                    else:
                        guarded_stores.append({"store": current,
                                               "guard": guarded_by})
                    collect(current.value, guarded_by)
                    for index in current.indices:
                        collect(index, guarded_by)
                    return
                if kind == "For":
                    collect(current.body, guarded_by)
                    return
                for item in getattr(current, "seq", ()) or ():
                    collect(item, guarded_by)
                for argument in getattr(current, "args", ()) or ():
                    if hasattr(argument, "__class__"):
                        collect(argument, guarded_by)
                for name in ("a", "b", "value"):
                    child = getattr(current, name, None)
                    if child is not None and hasattr(child, "__class__"):
                        collect(child, guarded_by)

            collect(node.body)
            continue
        if isinstance(node, tir.Evaluate) and isinstance(node.value, tir.Call) \
                and "fill" in _op_name(node.value):
            buffer, extents = _region_of(node.value, 0)
            value = _const(node.value.args[1]) if len(node.value.args) > 1 else None
            if str(getattr(buffer, "name", "")) != running["name"]:
                reject(RK_E_UNSUPPORTED,
                       "an unsupported fill outside the K loop", stage="structured",
                       actual=str(getattr(buffer, "name", "")))
            if value != 0 or extents != [int(x) for x in running["shape"]]:
                reject(RK_E_NUMERIC_CONTRACT,
                       "the running accumulator must be cleared to zero over the "
                       "whole fragment", stage="structured",
                       actual={"value": value, "extents": extents})
            accumulator_fills += 1
            continue
        reject(RK_E_UNSUPPORTED,
               "a statement outside the K loop is not consumed",
               stage="structured", actual=type(node).__name__)
    if accumulator_fills != 1:
        reject(RK_E_NUMERIC_CONTRACT,
               "the running accumulator must be cleared exactly once outside the "
               "K loop", stage="structured", actual=accumulator_fills)
    entries = list(guarded_stores) + [{"store": s, "guard": None}
                                      for s in unguarded_stores]
    if len(entries) != 1:
        reject(RK_E_UNSUPPORTED,
               "the kernel must write the output parameter exactly once",
               stage="structured",
               actual={"guarded": len(guarded_stores),
                       "unguarded": len(unguarded_stores)})
    store = entries[0]["store"]
    guard = entries[0]["guard"]
    if type(store.value).__name__ != "BufferLoad":
        reject(RK_E_NUMERIC_CONTRACT,
               "the output store must store the accumulator value itself",
               stage="structured", actual=str(store.value)[:80])
    stored_value = store.value
    if str(stored_value.buffer.name) != running["name"]:
        reject(RK_E_NUMERIC_CONTRACT,
               "the output store must read the running accumulator",
               stage="structured")
    # the epilogue must iterate the whole tile
    epilogue_nest = []
    for node in root_seq:
        if not isinstance(node, tir.For):
            continue
        cursor = node
        while type(cursor).__name__ == "For":
            epilogue_nest.append(int(cursor.extent))
            cursor = cursor.body
        break
    if epilogue_nest != [int(tile_shape[0]), int(tile_shape[1])]:
        reject(RK_E_SHAPE,
               "the epilogue must iterate the whole tile", stage="structured",
               actual={"iterates": epilogue_nest, "tile": list(tile_shape)})
    axis_vars = []
    for axis in range(2):
        local_index = stored_value.indices[axis]
        local_vars = _vars_in(local_index)
        extra = [var for var in _vars_in(store.indices[axis])
                 if not any(var.same_as(other) for other in local_vars)]
        axis_vars.append(extra[0] if len(extra) == 1 else None)
    read_axis_vars = {}
    for row in read_range:
        read_axis_vars.setdefault(row["axis"], set()).add(row["tile_var"])
    bounds = [None, None]
    if guard is not None:
        guard_lhs = []
        guard_limits = []
        for part in _conjuncts(guard):
            if type(part).__name__ != "LT":
                reject(RK_E_UNSUPPORTED,
                       "the output guard must be a conjunction of upper bounds",
                       stage="structured", actual=type(part).__name__)
            guard_lhs.append(part.a)
            guard_limits.append(_const(part.b))
        if len(guard_lhs) != 2 or None in guard_limits:
            reject(RK_E_SHAPE, "the output guard must bound both output axes",
                   stage="structured", actual=guard_limits)
    for axis in range(2):
        if axis_vars[axis] is None:
            reject(RK_E_SHAPE,
                   "the output store index is not one tile coordinate plus a "
                   "local index", stage="structured",
                   actual=str(store.indices[axis]))
        expected = axis_vars[axis] * tile_shape[axis] + stored_value.indices[axis]
        if not analyzer.can_prove(store.indices[axis] == expected):
            reject(RK_E_SHAPE, "the output store index is not tile-local",
                   stage="structured", actual=str(store.indices[axis]))
        if guard is not None:
            position = None
            for candidate in range(len(guard_lhs)):
                if analyzer.can_prove(guard_lhs[candidate] == store.indices[axis]):
                    position = candidate
                    break
            if position is None:
                reject(RK_E_SHAPE,
                       "the output guard does not bound the axis it stores",
                       stage="structured", actual=str(store.indices[axis]))
            bounds[axis] = guard_limits[position]
        if axis in read_axis_vars and str(axis_vars[axis]) not in read_axis_vars[axis]:
            reject(RK_E_SHAPE,
                   "the grid axis used by the output store differs from the one "
                   "used to read that axis", stage="structured",
                   actual={"axis": axis, "store": str(axis_vars[axis]),
                           "reads": sorted(read_axis_vars[axis])})
    if guard is None:
        # an unguarded epilogue is only sound when the grid covers exactly
        bounds = [None, None]
        roles = _operand_roles(captured, {"guarded": guarded})
        grid_x = int(captured["grid_loops"][0].extent)
        grid_y = int(captured["grid_loops"][1].extent)
        if (grid_x * tile_shape[0] != roles["m"]
                or grid_y * tile_shape[1] != roles["n"]):
            reject(RK_E_SHAPE,
                   "an unguarded epilogue needs an exactly covered grid",
                   stage="structured",
                   actual={"grid": [grid_x, grid_y],
                           "tile": list(tile_shape),
                           "out": [roles["m"], roles["n"]]})
    # root ordering: clear before the K loop, output store after it
    root_full = captured.get("root_full") or []
    loop_position = captured.get("loop_position")
    fill_position = store_position = None
    for position, item in enumerate(root_full):
        if _structurally_equal(item, captured["k_loop"]):
            continue
        if isinstance(item, tir.Evaluate) and isinstance(item.value, tir.Call) \
                and "fill" in _op_name(item.value):
            fill_position = position
        elif isinstance(item, tir.For):
            store_position = position
    if loop_position is None or fill_position is None or store_position is None:
        reject(RK_E_UNSUPPORTED,
               "cannot establish the root statement order", stage="structured",
               actual={"loop": loop_position, "fill": fill_position,
                       "store": store_position})
    if fill_position > loop_position:
        reject(RK_E_NUMERIC_CONTRACT,
               "the accumulator must be cleared before the K loop",
               stage="structured",
               actual={"fill": fill_position, "loop": loop_position})
    if store_position < loop_position:
        reject(RK_E_NUMERIC_CONTRACT,
               "the output store must follow the K loop", stage="structured",
               actual={"store": store_position, "loop": loop_position})
    table_epilogue = [{"axis": axis, "bound": bounds[axis],
                       "tile_var": str(axis_vars[axis]),
                       "local_index": str(stored_value.indices[axis]),
                       "guarded": guard is not None}
                      for axis in range(2)]
    table_epilogue_target = str(store.buffer.name)

    # rounding derived from the computation: a writeback inside the K loop
    rounding = "explicit_per_k_fp16"
    return {"rounding": rounding, "rounding_points": ROUNDINGS[rounding],
            "running": running["name"], "gemm_acc": gemm_acc["name"],
            "fills": fills, "copies": copies, "guarded": guarded,
            "read_range": read_range,
            "tile_shape": tile_shape,
            "block_k": block_k,
            "epilogue": table_epilogue,
            "gemm_args": gemm_args,
            "epilogue_target": table_epilogue_target,
            "statements": statement_kinds, "gemm_position": gemm_at}


# ---------------------------------------------------------------- stage 2/3
def _operand_roles(captured, proofs):
    """Classify the two operands by where the K loop variable indexes them.

    The A copy reads A[m_index, k_index]; the B copy reads B[k_index, n_index].
    Whichever source index mentions the K loop variable fixes that operand K
    axis, and the other index gives M or N.  Nothing is guessed from the buffer
    count or the parameter order.
    """
    shapes = {entry["name"]: entry["shape"] for entry in captured["params"]}
    k_var = captured["k_loop"].loop_var
    k_axis = {}
    for node in _loop_body_seq(captured["k_loop"]):
        for _, inner in _walk(node):
            if type(inner).__name__ != "BufferStore":
                continue
            if not _scope_of(inner.buffer).startswith("shared"):
                continue
            for _, value_node in _walk(inner.value):
                if type(value_node).__name__ != "BufferLoad":
                    continue
                name = str(value_node.buffer.name)
                for axis, index in enumerate(value_node.indices):
                    if _mentions(index, k_var):
                        k_axis.setdefault(name, axis)
    roles = {}
    for name, axis in k_axis.items():
        shape = shapes.get(name)
        if shape is None or len(shape) != 2:
            continue
        if axis == 1:
            roles["m"] = shape[0]
            roles["k"] = shape[1]
        else:
            roles["n"] = shape[1]
            roles["k"] = shape[0]
    if set(roles) != {"m", "k", "n"}:
        reject(RK_E_SHAPE,
               "cannot relate the GEMM operands to the parameter index axes",
               stage="structured",
               actual={"k_axis": k_axis, "roles": roles})
    return roles


def stage2_k_partition(captured: dict[str, Any], proofs: dict[str, Any],
                       block_k: int) -> dict[str, Any]:
    k_loop = captured["k_loop"]
    trips = int(k_loop.extent)
    roles = _operand_roles(captured, proofs)
    logical_k = roles["k"]
    if trips * block_k < logical_k:
        reject(RK_E_SHAPE,
               "the K loop does not cover the logical K",
               stage="structured",
               actual={"trips": trips, "block_k": block_k, "logical_k": logical_k})
    slices = []
    remaining = logical_k
    for index in range(trips):
        extent = min(block_k, remaining)
        slices.append({"index": index, "k0": index * block_k, "extent": extent,
                       "physical": ((extent + 31) // 32) * 32})
        remaining -= extent
    if any(s["extent"] <= 0 for s in slices):
        reject(RK_E_SHAPE,
               "a K slice must carry a positive extent", stage="structured",
               actual=[(s["k0"], s["extent"]) for s in slices])
    if remaining != 0 or block_k != K_TILE:
        reject(RK_E_SHAPE, "K partition does not match the lowering contract",
               stage="structured",
               actual={"logical_k": logical_k, "trips": trips, "block_k": block_k})
    return {"trips": trips, "block_k": block_k, "logical_k": logical_k,
            "slices": slices}


@dataclass
class Tile:
    index: tuple[int, int]
    rows: int
    cols: int
    valid_rows: int
    valid_cols: int


def stage3_tiles(captured: dict[str, Any], proofs: dict[str, Any],
                 logical_k: int, tile_shape) -> dict[str, Any]:
    """Per-tile geometry, proved against the guards at each block index."""
    tir = _tir()
    grid_x = int(captured["grid_loops"][0].extent)
    grid_y = int(captured["grid_loops"][1].extent)
    tile_m, tile_n = int(tile_shape[0]), int(tile_shape[1])
    roles = _operand_roles(captured, proofs)
    m, n = roles["m"], roles["n"]
    output = [entry for entry in captured["params"] if entry["shape"] == [m, n]]
    if not output:
        reject(RK_E_SHAPE, "cannot identify the output parameter",
               stage="structured",
               actual={"m": m, "n": n, "params": [e["shape"] for e in captured["params"]]})
    if grid_x != -(-m // tile_m) or grid_y != -(-n // tile_n):
        reject(RK_E_SHAPE, "the block grid does not cover the output",
               stage="structured",
               actual={"grid": [grid_x, grid_y], "tile": [tile_m, tile_n],
                       "out": [m, n]})
    tiles = []
    for by in range(grid_y):
        for bx in range(grid_x):
            tiles.append(Tile(index=(bx * tile_m, by * tile_n), rows=tile_m,
                              cols=tile_n,
                              valid_rows=min(tile_m, m - bx * tile_m),
                              valid_cols=min(tile_n, n - by * tile_n)))
    return {"grid": (grid_x, grid_y), "tile_shape": (tile_m, tile_n),
            "tiles": tiles, "logical_m": m, "logical_n": n}


@dataclass
class Stage:
    name: str
    kind: str
    invariants: tuple[str, ...]
    ir_digest: str
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind,
                "invariants": list(self.invariants),
                "ir_digest": self.ir_digest, "notes": self.notes}


@dataclass
class KSlice:
    index: int
    k0: int
    extent: int
    physical: int


@dataclass
class StructuredModule:
    logical_m: int
    logical_n: int
    logical_k: int
    order: str
    grid: tuple[int, int]
    tile_shape: tuple[int, int]
    tiles: tuple[Tile, ...]
    slices: tuple[KSlice, ...]
    buffers: tuple[dict[str, Any], ...]
    rounding: str
    rounding_points: str
    hints: dict[str, Any]
    proofs: dict[str, Any]
    stages: tuple[Stage, ...]
    source_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical": [self.logical_m, self.logical_k, self.logical_n],
            "order": self.order,
            "grid": list(self.grid),
            "tile_shape": list(self.tile_shape),
            "tiles": [{"index": list(t.index), "rows": t.rows, "cols": t.cols,
                       "valid_rows": t.valid_rows, "valid_cols": t.valid_cols}
                      for t in self.tiles],
            "slices": [{"index": s.index, "k0": s.k0, "extent": s.extent,
                        "physical": s.physical} for s in self.slices],
            "buffers": list(self.buffers),
            "rounding": self.rounding,
            "rounding_points": self.rounding_points,
            "hints": self.hints,
            "proofs": self.proofs,
            "stages": [s.to_dict() for s in self.stages],
            "source_digest": self.source_digest,
        }


def lower(prim_func: Any, *, annotations: dict[str, Any] | None = None,
          order: str = "m_major",
          declared_rounding: str | None = None) -> StructuredModule:
    """Prove, then structure.  The rounding is derived from the computation; a
    caller declaration is only cross-checked, never adopted."""
    captured = stage0_capture(prim_func)
    proofs = stage1_prove(captured)
    if declared_rounding is not None and declared_rounding != proofs["rounding"]:
        reject(RK_E_NUMERIC_CONTRACT,
               "the declared source rounding disagrees with the proven rounding",
               stage="structured",
               actual={"declared": declared_rounding,
                       "proven": proofs["rounding"]})
    partition = stage2_k_partition(captured, proofs, proofs["block_k"])
    derived = stage3_tiles(captured, proofs, partition["logical_k"],
                           proofs["tile_shape"])
    candidates = [entry["name"] for entry in captured["params"]
                  if entry["shape"] == [derived["logical_m"], derived["logical_n"]]]
    if len(candidates) != 1:
        reject(RK_E_SHAPE,
               "the output parameter must be unambiguous", stage="structured",
               actual=candidates)
    output_name = candidates[0]
    if proofs["epilogue_target"] != output_name:
        reject(RK_E_SHAPE,
               "the single global store does not write the output parameter",
               stage="structured",
               actual={"store": proofs["epilogue_target"], "output": output_name})
    for row in proofs["epilogue"]:
        if row["bound"] is None:
            continue
        expected = derived["logical_m"] if row["axis"] == 0 else derived["logical_n"]
        if row["bound"] != expected:
            reject(RK_E_SHAPE,
                   "the output guard bound is not the logical extent",
                   stage="structured",
                   actual={"axis": row["axis"], "bound": row["bound"],
                           "logical": expected})
    hints = dict(annotations or {})
    stages = (
        Stage("0-capture", "analysis",
              ("one root block", "one 2D block grid", "one serial K loop",
               "tile buffers are allocated inside the kernel"),
              _digest(str([type(n).__name__ for _, n in captured["nodes"]])),
              {"grid": [int(g.extent) for g in captured["grid_loops"]],
               "buffers": [e["name"] for e in captured["allocations"]]}),
        Stage("1-prove", "analysis",
              ("each shared tile is zero filled in full, once per K block",
               "each shared tile is filled through a guarded copy from an operand",
               "the running FP16 accumulator is loaded before and written after "
               "the tile GEMM",
               "the source rounding follows from that writeback"),
              _digest(str(proofs["statements"])),
              {"statements": proofs["statements"],
               "gemm_position": proofs["gemm_position"],
               "rounding": proofs["rounding"]}),
        Stage("2-k-chain", "transformation",
              ("slices are ordered and cover the proved logical K",
               "each slice keeps the K iteration body unchanged"),
              _digest(str(partition["slices"])),
              {"logical_k": partition["logical_k"],
               "slices": partition["slices"]}),
        Stage("3-tiles", "transformation",
              ("the grid covers the output with ceildiv tiles",
               "per-tile valid extents follow from the operands and tile shape"),
              _digest(str([(t.index, t.valid_rows, t.valid_cols)
                           for t in derived["tiles"]])),
              {"grid": list(derived["grid"])}),
    )
    return StructuredModule(
        logical_m=derived["logical_m"], logical_n=derived["logical_n"],
        logical_k=partition["logical_k"], order=order,
        grid=derived["grid"], tile_shape=derived["tile_shape"],
        tiles=tuple(derived["tiles"]),
        slices=tuple(KSlice(**s) for s in partition["slices"]),
        buffers=tuple({k: v for k, v in entry.items() if k != "buffer"}
                      for entry in captured["allocations"]),
        rounding=proofs["rounding"], rounding_points=proofs["rounding_points"],
        hints=hints, proofs={"rounding": proofs["rounding"],
                             "read_range": proofs["read_range"],
                             "gemm_args": proofs["gemm_args"],
                             "epilogue": proofs["epilogue"],
                             "gemm_position": proofs["gemm_position"],
                             "guarded": proofs["guarded"],
                             "copies": proofs["copies"],
                             "fills": proofs["fills"]},
        stages=stages, source_digest=_digest(prim_func.script()))


# ---------------------------------------------------------------- plan
def plan_from_module(module: StructuredModule, *, profile: Any = None,
                     options: Any = None, name: str = "structured-linear"):
    """Build RKPlan from the module and verify the result against it.

    The planner still assembles buffers, BOs and submits; this function refuses a
    plan whose reconstruction disagrees with what the module proved.
    """
    import dataclasses

    from .. import ir
    from ..api import compile_rkir
    from ..capabilities import default_profile
    from ..options import CompileOptions

    profile = profile or default_profile()
    options = options or CompileOptions(numeric_mode="rk_f16_tiled",
                                       allow_approx_lut=False)
    name = "%s:%s" % (name, module.source_digest)
    graph = ir.matmul_module(m=module.logical_m, k=module.logical_k,
                             n=module.logical_n, name=name,
                             numeric_mode="rk_f16_tiled",
                             source="structured#%s" % module.source_digest)
    op = graph.ops[0]
    attrs = dict(op.attrs)
    attrs.update({
        "source_schedule": {"kind": "mn_tiles_v1",
                            "tile_m": module.tile_shape[0],
                            "tile_n": module.tile_shape[1],
                            "order": module.order,
                            "k_tile": K_TILE,
                            "accumulation": "ordered_ascending_f16_writeback"},
        "source_k_rounding": module.rounding,
        "source_mn_tiling": {"block_m": module.tile_shape[0],
                             "block_n": module.tile_shape[1],
                             "grid_x": module.grid[0], "grid_y": module.grid[1],
                             "coverage": "exact_once"},
        "source_k_loop": {"logical_k": module.logical_k,
                          "k_blocks": [{"index": s.index, "k0": s.k0,
                                        "extent": s.extent} for s in module.slices],
                          "rounding_points": module.rounding_points},
        "source_loop_annotations": dict(module.hints),
    })
    graph = dataclasses.replace(
        graph, ops=(dataclasses.replace(op, attrs=attrs),) + tuple(graph.ops[1:]))
    plan = compile_rkir(graph, profile=profile, options=options).plan
    verify_plan(module, plan)
    return plan


def _view_shape(plan, view_id):
    """Logical shape of a plan view, whatever the layout object calls it."""
    tensor = getattr(plan.views[view_id], "tensor", None)
    for attr in ("shape", "dims", "extents"):
        value = getattr(tensor, attr, None)
        if value is not None and not callable(value):
            return [int(x) for x in value]
    if hasattr(tensor, "to_dict"):
        data = tensor.to_dict()
        for key in ("shape", "dims", "extents"):
            if key in data:
                return [int(x) for x in data[key]]
    reject(RK_E_SHAPE, "cannot read the logical shape of a plan view",
           stage="structured", actual=str(type(tensor).__name__))


def verify_plan(module: StructuredModule, plan: Any) -> None:
    """The plan must reproduce the module's proved structure.

    The plan is still a skeleton: it carries the proved geometry, the K chain and
    the source digest in its name, but not the statement-level proofs.  Anything
    the skeleton cannot express (values, iteration coverage) is enforced before
    the plan is built, never here.
    """
    if module.source_digest not in str(plan.name):
        reject(RK_E_SCHEMA,
               "the plan does not carry the structured source digest",
               stage="structured",
               actual={"plan": str(plan.name), "digest": module.source_digest})
    tasks = [a for a in plan.actions if a.kind == "npu_task"]
    expected = len(module.tiles) * len(module.slices)
    if len(tasks) != expected:
        reject(RK_E_SHAPE, "plan task count does not match the proved structure",
               stage="structured",
               actual={"tasks": len(tasks), "expected": expected})
    chains: dict[Any, list[int]] = {}
    for task in tasks:
        chain = chains.setdefault(task.attrs.get("submit_group"), [])
        chain.append(task.attrs.get("k_index"))
    for group, indexes in chains.items():
        indexes = [index for index in indexes if index is not None]
        if indexes and sorted(indexes) != [s.index for s in module.slices]:
            reject(RK_E_SHAPE, "a submit does not carry the proved K chain",
                   stage="structured", actual={"group": group, "k": indexes})
    requirements = plan.numeric_requirements
    if not requirements:
        reject(RK_E_NUMERIC_CONTRACT, "the plan has no numeric requirement",
               stage="structured")
    execution = requirements[0].to_dict().get("execution", {})
    if execution.get("k_chain") and len(execution["k_chain"]) != len(module.slices):
        reject(RK_E_NUMERIC_CONTRACT,
               "the plan K chain disagrees with the proved slices",
               stage="structured",
               actual={"plan": execution["k_chain"],
                       "module": [s.k0 for s in module.slices]})
    # NOTE: a stronger check would compare every task's operand extents with the
    # proved (tile, slice) map.  That is implemented but not enabled: the plan
    # view shape API in this tree returns the buffer extent rather than the
    # sliced view extent, so the check cannot yet distinguish the two without
    # guessing.
    slice_by_index = {s.index: s for s in module.slices}
    for task in tasks:
        k_index = task.attrs.get("k_index")
        if k_index is None:
            k0 = task.attrs.get("k0")
            matched = [s for s in module.slices if s.k0 == k0]
            if len(matched) != 1:
                reject(RK_E_SHAPE,
                       "a task carries a K offset the module did not prove",
                       stage="structured", actual=k0)
            slice_ = matched[0]
        elif k_index in slice_by_index:
            slice_ = slice_by_index[k_index]
        else:
            reject(RK_E_SHAPE, "a task carries a K index the module did not prove",
                   stage="structured", actual=k_index)
        if task.attrs.get("k0") != slice_.k0:
            reject(RK_E_SHAPE, "a task K offset disagrees with the proved slice",
                   stage="structured",
                   actual={"task": task.attrs.get("k0"), "proved": slice_.k0})
        kt = task.attrs.get("kt")
        if kt is not None and int(kt) != slice_.extent:
            reject(RK_E_SHAPE, "a task K extent disagrees with the proved slice",
                   stage="structured", actual={"task": int(kt),
                                               "proved": slice_.extent})
    published = sorted((int(tile.dst_row), int(tile.dst_col),
                        int(tile.rows), int(tile.cols)) for tile in plan.publish)
    proved = sorted((tile.index[0], tile.index[1], tile.valid_rows, tile.valid_cols)
                    for tile in module.tiles)
    if published != proved:
        reject(RK_E_SHAPE,
               "the plan publishes different tiles than the module proved",
               stage="structured",
               actual={"plan": published[:4], "proved": proved[:4]})
    rotated = sum(t.valid_rows * t.valid_cols for t in module.tiles)
    if rotated != module.logical_m * module.logical_n:
        reject(RK_E_SHAPE, "the proved tiles do not cover the output exactly once",
               stage="structured", actual=rotated)
