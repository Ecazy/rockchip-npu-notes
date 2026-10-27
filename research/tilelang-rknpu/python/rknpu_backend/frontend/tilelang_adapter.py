# SPDX-License-Identifier: GPL-3.0-or-later
"""A deliberately small, structural TileLang PrimFunc extractor.

This module does not parse ``str(prim_func)`` and does not guess operators from
variable names or buffer shapes.  It accepts one closed pattern over the actual
TVM/TileLang TIR objects:

    rank-2 FP16 A, B, C parameters
    explicit zero fill of the FP32 accumulator
    A->shared, B->shared, accumulator->C tile copies in that order
    exactly one tile GEMM with trans_A=false, trans_B=false between them
    optional ``acc = acc * sigmoid(acc)`` on the same (i, j) elements
    and nothing else

Parameter roles come from the copy data flow (which parameter feeds GEMM operand
A, operand B, and which parameter receives the accumulator), so square shapes are
not ambiguous.  A buffer is identified by its underlying TVM var, not by name.

The recognizer does not choose its output container.  Node and module
construction goes through :mod:`rknpu_backend.frontend.graph_builder`, whose
default carrier is the internal RKIR module and whose alternative carrier returns
container-free records for :mod:`rknpu_backend.frontend.native_graph`.  Both
carriers therefore share one copy of the recognition logic.

The optional TileLang import stays inside this module, so importing
:mod:`rknpu_backend` has no TileLang dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ..errors import (RK_E_NUMERIC_CONTRACT, RK_E_SHAPE, RK_E_UNSUPPORTED,
                      reject)
from ..ir import NumericContract, TensorType
from ..options import CompileOptions
from .graph_builder import active_builder

ALLOWED_TILEOPS = {"tl.tileop.fill", "tl.tileop.copy", "tl.tileop.gemm"}
FORBIDDEN_TOKENS = ("atomic", "barrier", "shuffle", "sync", "reduce")


@dataclass(frozen=True)
class _Region:
    buffer: Any
    physical: tuple[int, int]
    # Constant (row, col) offset, or None when the index is not a literal.  The
    # K-loop path analyses the raw expressions; every other path requires
    # constants and rejects None.
    offset: tuple[int, int] | None
    access_mask: int
    raw_indices: tuple[Any, Any] = (None, None)


@dataclass(frozen=True)
class _Item:
    order: int
    kind: str
    node: Any
    loops: tuple[Any, ...] = ()
    blocks: tuple[Any, ...] = ()
    guards: tuple[Any, ...] = ()


def tilelang_available() -> bool:
    try:
        import tilelang  # noqa: F401
        import tvm  # noqa: F401
        return True
    except Exception:
        return False


def _import_tirx():
    try:
        from tvm import tirx
        return tirx
    except Exception as exc:
        reject(RK_E_UNSUPPORTED,
               f"TileLang/TVM is unavailable or failed to import: {exc}",
               stage="frontend",
               alternative="install the pinned optional TileLang dependency from "
                           "provenance.json")


def _value(*args, **kwargs):
    """Construct one recognized value through the active carrier."""
    return active_builder().value(*args, **kwargs)


def _op(*args, **kwargs):
    """Construct one recognized operation through the active carrier."""
    return active_builder().op(*args, **kwargs)


def _finish(*, name: str, values, ops, inputs, outputs,
            profile_hash: str = "", source_hash: str = "",
            state_signature: str = ""):
    """Finish a recognized graph in the active carrier."""
    return active_builder().finish(
        name=name, values=values, ops=ops, inputs=inputs, outputs=outputs,
        profile_hash=profile_hash, source_hash=source_hash,
        state_signature=state_signature)


def _int_imm(node: Any) -> int | None:
    try:
        if type(node).__name__ == "IntImm":
            return int(node.value)
    except Exception:
        return None
    return None


def _bool_imm(node: Any) -> bool | None:
    value = _int_imm(node)
    if value is not None:
        return bool(value)
    try:
        if type(node).__name__ == "BoolImm":
            return bool(node.value)
    except Exception:
        return None
    return None


def _op_name(call: Any) -> str:
    try:
        return str(call.op.name)
    except Exception:
        return ""


def _affine_index(expr: Any, tirx: Any) -> tuple[Any, int, int] | None:
    """Match ``var * stride + base`` (either operand order) or a constant.

    Returns ``(var_or_None, stride, base)``; a constant yields ``(None, 0, value)``.
    Only single-variable linear forms are accepted, which is what a K-blocked
    TileLang copy produces.
    """
    value = _int_imm(expr)
    if value is not None:
        return (None, 0, value)
    name = type(expr).__name__
    if name == "Add":
        left = _affine_index(expr.a, tirx)
        right = _affine_index(expr.b, tirx)
        if left is None or right is None:
            return None
        if left[0] is None and right[0] is not None:
            return (right[0], right[1], right[2] + left[2])
        if right[0] is None and left[0] is not None:
            return (left[0], left[1], left[2] + right[2])
        if left[0] is None and right[0] is None:
            return (None, 0, left[2] + right[2])
        return None
    if name == "Mul":
        for factor, other in ((expr.a, expr.b), (expr.b, expr.a)):
            if type(factor).__name__ == "Var":
                scale = _int_imm(other)
                if scale is None:
                    return None
                return (factor, scale, 0)
        return None
    if name == "Var":
        return (expr, 1, 0)
    return None


def _gemm_header(gemm: Any, tirx: Any, span: str):
    """Validate the GEMM call header shared by both extraction paths.

    Scalar flags are checked before the regions: with ``clear_accum=True``
    TileLang also downgrades the accumulator region to a plain store, and the
    clearer diagnosis is the flag itself.
    """
    if len(gemm.args) < 10:
        reject(RK_E_UNSUPPORTED, "malformed tl.tileop.gemm", stage="frontend")
    trans_a = _bool_imm(gemm.args[3])
    trans_b = _bool_imm(gemm.args[4])
    if trans_a is None or trans_b is None:
        reject(RK_E_UNSUPPORTED, "dynamic GEMM transpose flags", stage="frontend")
    if trans_a or trans_b:
        reject(RK_E_UNSUPPORTED,
               "the minimal TileLang adapter accepts only trans_A=false, "
               "trans_B=false", stage="frontend", source=span)
    gm, gn, gk = (_int_imm(gemm.args[i]) for i in range(5, 8))
    if gm is None or gn is None or gk is None:
        reject(RK_E_SHAPE, "dynamic GEMM extents are unsupported", stage="frontend")
    # arg 9 is ``clear_accum``: the GEMM would zero the accumulator before adding,
    # which destroys a K accumulation built across loop iterations.
    clear = _bool_imm(gemm.args[9])
    if clear is None:
        reject(RK_E_UNSUPPORTED, "dynamic GEMM clear_accum flag",
               stage="frontend", source=span)
    if clear:
        reject(RK_E_NUMERIC_CONTRACT,
               "GEMM requests clear_accum=True; the accumulator must be zero "
               "filled once and accumulated across K tiles", stage="frontend",
               source=span, actual=True)
    a_reg = _region(gemm.args[0], tirx, "gemm operand A", expect_mask=1)
    b_reg = _region(gemm.args[1], tirx, "gemm operand B", expect_mask=1)
    c_reg = _region(gemm.args[2], tirx, "gemm accumulator", expect_mask=3)
    return a_reg, b_reg, c_reg, gm, gn, gk


def _check_plain_tile_buffer(region: _Region, label: str, span: str) -> None:
    """A tile buffer used by the GEMM must be plain row-major with no offset.

    The lowering treats the tile as a dense [rows, cols] buffer; a strided or
    offset view would silently change which elements are contracted.
    """
    try:
        strides = (tuple(int(x) for x in region.buffer.strides)
                   if region.buffer.strides else ())
        elem_offset = int(region.buffer.elem_offset)
    except Exception:
        reject(RK_E_UNSUPPORTED,
               f"{label} buffer layout is not inspectable", stage="frontend",
               source=span)
    if strides not in ((), (region.physical[1], 1)) or elem_offset != 0:
        reject(RK_E_UNSUPPORTED,
               f"{label} uses a strided or offset buffer layout; only plain "
               "row-major tiles are lowered", stage="frontend", source=span,
               actual={"strides": list(strides), "elem_offset": elem_offset})


def _role_of(var: Any, roles: dict[str, Any]) -> str | None:
    for role, candidate in roles.items():
        if _same_var(var, candidate):
            return role
    return None


def _affine_expr(expr: Any, tirx: Any,
                 roles: dict[str, Any]) -> tuple[dict[str, int], int] | None:
    """Parse ``sum(coeff * role) + const`` over the allowed loop/block vars."""
    value = _int_imm(expr)
    if value is not None:
        return ({}, value)
    name = type(expr).__name__
    if name == "Add":
        left = _affine_expr(expr.a, tirx, roles)
        right = _affine_expr(expr.b, tirx, roles)
        if left is None or right is None:
            return None
        coeffs = dict(left[0])
        for role, coeff in right[0].items():
            coeffs[role] = coeffs.get(role, 0) + coeff
        return (coeffs, left[1] + right[1])
    if name == "Mul":
        for factor, other in ((expr.a, expr.b), (expr.b, expr.a)):
            if isinstance(factor, tirx.Var):
                scale = _int_imm(other)
                if scale is None:
                    return None
                role = _role_of(factor, roles)
                if role is None:
                    return None
                return ({role: scale}, 0)
        return None
    if isinstance(expr, tirx.Var):
        role = _role_of(expr, roles)
        return ({role: 1}, 0) if role is not None else None
    return None


def _require_affine(region: _Region, axis: int, tirx: Any,
                    roles: dict[str, Any], expected: dict[str, int],
                    label: str, span: str) -> None:
    expr = region.raw_indices[axis]
    parsed = _affine_expr(expr, tirx, roles)
    if parsed is None:
        reject(RK_E_UNSUPPORTED,
               f"{label}: index on axis {axis} is not an affine expression of "
               "the loop/block variables", stage="frontend", source=span,
               actual=repr(expr))
    coeffs, const = parsed
    if const != 0 or {k: v for k, v in coeffs.items() if v} != \
            {k: v for k, v in expected.items() if v}:
        reject(RK_E_UNSUPPORTED,
               f"{label}: index on axis {axis} does not match the supported "
               "tiling", stage="frontend", source=span,
               actual={"coeffs": coeffs, "const": const},
               required=expected)


def _same_buffer(a: Any, b: Any) -> bool:
    """Buffer identity by underlying TVM var, never by display name."""
    try:
        return bool(a.data.same_as(b.data))
    except Exception:
        return False


def _same_var(a: Any, b: Any) -> bool:
    try:
        return bool(a.same_as(b))
    except Exception:
        return False


def _buffer_name(buffer: Any) -> str:
    try:
        return str(buffer.name)
    except Exception:
        return "<unnamed>"


def _region(call: Any, tirx: Any, where: str,
            expect_mask: int | None = None) -> _Region:
    if not isinstance(call, tirx.Call) or _op_name(call) != "tl.region":
        reject(RK_E_UNSUPPORTED, f"{where}: expected a tl.region argument",
               stage="frontend")
    args = list(call.args)
    if len(args) < 4:
        reject(RK_E_UNSUPPORTED, f"{where}: malformed tl.region", stage="frontend")
    load = args[0]
    if not isinstance(load, tirx.BufferLoad):
        reject(RK_E_UNSUPPORTED,
               f"{where}: tl.region does not start with a BufferLoad",
               stage="frontend")
    mask = _int_imm(args[1])
    shape = tuple(_int_imm(x) for x in args[2:])
    if mask is None or any(v is None or v <= 0 for v in shape):
        reject(RK_E_UNSUPPORTED, f"{where}: dynamic or malformed region extent",
               stage="frontend")
    if len(shape) != 2:
        reject(RK_E_SHAPE, "MVP TileLang subset only accepts rank-2 regions",
               stage="frontend", actual=list(shape))
    if len(load.indices) != 2:
        reject(RK_E_UNSUPPORTED, f"{where}: expected a rank-2 region start",
               stage="frontend")
    # A predicated load/store executes conditionally; the MVP cannot represent
    # that, so only an unconditional access is accepted.
    if not _predicate_true(getattr(load, "predicate", None)):
        reject(RK_E_UNSUPPORTED,
               f"{where}: region load carries a predicate; conditional loads are "
               "not lowered", stage="frontend", actual=str(getattr(load,
                                                                    "predicate",
                                                                    None)))
    # ``mask`` is a load/store access mask: 1 = load, 2 = store, 3 = load+store.
    if mask not in (1, 2, 3):
        reject(RK_E_UNSUPPORTED,
               f"{where}: unknown region access mask {mask}", stage="frontend")
    if expect_mask is not None and mask != expect_mask:
        name = {1: "load", 2: "store", 3: "load+store"}.get(mask, str(mask))
        want = {1: "load", 2: "store", 3: "load+store"}[expect_mask]
        reject(RK_E_UNSUPPORTED,
               f"{where}: region is a {name} but the pattern requires a {want}",
               stage="frontend", actual=mask, required=expect_mask)
    raw = tuple(load.indices)
    constants = tuple(_int_imm(x) for x in raw)
    offset = (None if any(v is None or v < 0 for v in constants)
              else (constants[0], constants[1]))
    return _Region(load.buffer, (shape[0], shape[1]), offset, mask, raw)


def _predicate_true(node: Any) -> bool:
    """A predicate is only acceptable when it is absent or provably always true."""
    if node is None or node is True:
        return True
    try:
        if type(node).__name__ != "IntImm":
            return False
        return str(node.dtype) == "bool" and bool(node.value)
    except Exception:
        return False


def _scan(node: Any, tirx: Any, out: list[_Item], counter: list[int],
          loops: tuple[Any, ...], span: str = "",
          blocks: tuple[Any, ...] = (), guards: tuple[Any, ...] = ()) -> None:
    """Collect semantic items in execution order, skipping launch bindings."""
    if isinstance(node, tirx.SBlockRealize):
        # A guarded block may be skipped at run time.  The MVP cannot represent
        # conditional execution, so anything but a provably-true predicate would
        # silently execute work the source program may not do.
        if not _predicate_true(node.predicate):
            reject(RK_E_UNSUPPORTED,
                   "block predicate is not provably true; conditional execution "
                   "is not supported", stage="frontend", source=span,
                   actual=str(node.predicate))
        if list(node.iter_values):
            reject(RK_E_UNSUPPORTED,
                   "block iteration bindings are not lowered by the MVP",
                   stage="frontend", source=span,
                   actual=[str(v) for v in node.iter_values])
        _scan(node.block, tirx, out, counter, loops, span, blocks, guards)
        return
    if isinstance(node, tirx.SBlock):
        if list(node.iter_vars):
            reject(RK_E_UNSUPPORTED,
                   "block iteration variables are not lowered by the MVP",
                   stage="frontend", source=span,
                   actual=[str(v.var) for v in node.iter_vars])
        if node.init is not None:
            _scan(node.init, tirx, out, counter, loops, span, blocks, guards)
        _scan(node.body, tirx, out, counter, loops, span, blocks, guards)
        return
    if isinstance(node, tirx.SeqStmt):
        for stmt in node.seq:
            _scan(stmt, tirx, out, counter, loops, span, blocks, guards)
        return
    if isinstance(node, tirx.For):
        try:
            thread = node.thread_binding is not None
        except Exception:
            thread = False
        if thread:
            # Block/grid loops are launch structure.  They are recorded so the
            # index analysis can prove how the grid covers the output; whether a
            # wider grid is acceptable depends on the pattern being lowered.
            _scan(node.body, tirx, out, counter, loops, span, blocks + (node,),
                  guards)
            return
        out.append(_Item(counter[0], "for", node, loops, blocks, guards))
        counter[0] += 1
        _scan(node.body, tirx, out, counter, loops + (node,), span, blocks,
              guards)
        return
    if isinstance(node, tirx.Evaluate):
        _scan(node.value, tirx, out, counter, loops, span, blocks, guards)
        return
    if isinstance(node, tirx.IfThenElse):
        if node.else_case is not None:
            reject(RK_E_UNSUPPORTED,
                   "if/else tail guards are not lowered; use one provable "
                   "in-bounds condition", stage="frontend", source=span)
        _scan(node.then_case, tirx, out, counter, loops, span, blocks,
              guards + (node.condition,))
        return
    if isinstance(node, tirx.Call):
        out.append(_Item(counter[0], "call", node, loops, blocks, guards))
        counter[0] += 1
        name = _op_name(node)
        if not name.startswith("tl."):
            for arg in node.args:
                if type(arg).__module__.startswith("tvm"):
                    _scan(arg, tirx, out, counter, loops, span, blocks, guards)
        return
    if isinstance(node, tirx.BufferStore):
        if not _predicate_true(getattr(node, "predicate", None)):
            reject(RK_E_UNSUPPORTED,
                   "buffer store carries a predicate; conditional stores are not "
                   "lowered", stage="frontend", source=span,
                   actual=str(getattr(node, "predicate", None)))
        out.append(_Item(counter[0], "store", node, loops, blocks, guards))
        counter[0] += 1
        return
    name = type(node).__name__
    if name == "Bind":
        # Launch-time thread indices are metadata: they never contribute data
        # values to a region, and every region index we accept is a constant.
        value = getattr(node, "value", None)
        if not (isinstance(value, tirx.Call) and
                _op_name(value).endswith("launch_thread_idx")):
            reject(RK_E_UNSUPPORTED,
                   "a Bind carries a real data value; only launch thread index "
                   "bindings are ignored", stage="frontend", source=span,
                   actual=_op_name(value) if isinstance(value, tirx.Call) else
                   type(value).__name__)
        return
    if name == "AssertStmt":
        # A runtime assertion is a semantic precondition.  Dropping it would let
        # the device run inputs the source program rejects.
        if not _predicate_true(getattr(node, "condition", None)):
            reject(RK_E_UNSUPPORTED,
                   "runtime assertion is not provably true; MVP does not lower "
                   "guards", stage="frontend", source=span,
                   actual=str(getattr(node, "condition", "")))
        return
    if name == "LaunchThread":
        # Pure launch metadata (thread identity), same reasoning as Bind.
        return
    if name in {"BufferLoad", "IntImm", "FloatImm", "BoolImm", "Var",
                "StringImm"}:
        return
    reject(RK_E_UNSUPPORTED,
           f"unsupported TileLang node {type(node).__name__}", stage="frontend")


def _check_no_forbidden_calls(calls: Iterable[Any], span: str) -> None:
    for call in calls:
        name = _op_name(call)
        if any(token in name for token in FORBIDDEN_TOKENS):
            reject(RK_E_UNSUPPORTED,
                   f"GPU communication primitive {name} cannot be ignored",
                   stage="frontend", source=span, op=name)


def _param_buffers(prim_func: Any) -> list[Any]:
    try:
        buffers = [buffer for _, buffer in prim_func.buffer_map.items()]
    except Exception:
        reject(RK_E_UNSUPPORTED, "PrimFunc has no buffer_map", stage="frontend")
    if len(buffers) < 3:
        reject(RK_E_UNSUPPORTED,
               "a matmul program needs at least three parameter buffers "
               "(A, B and an output)", stage="frontend",
               actual=[_buffer_name(b) for b in buffers])
    for buffer in buffers:
        try:
            shape = tuple(int(x) for x in buffer.shape)
            dtype = str(buffer.dtype)
        except Exception:
            reject(RK_E_SHAPE, "parameter buffer is not static rank-2",
                   stage="frontend")
        if len(shape) > 2 or dtype != "float16":
            reject(RK_E_SHAPE,
                   "TileLang MVP buffers must be static rank-0/1/2 FP16",
                   stage="frontend",
                   actual={"name": _buffer_name(buffer), "shape": list(shape),
                           "dtype": dtype})
        strides = tuple(int(x) for x in buffer.strides) if buffer.strides else ()
        if len(shape) == 2:
            valid_strides = ((), (shape[1], 1))
        elif len(shape) == 1:
            valid_strides = ((), (1,))
        else:
            valid_strides = ((),)
        if strides not in valid_strides:
            reject(RK_E_UNSUPPORTED,
                   "parameter buffer is not plain row-major; strided/transposed "
                   "views are not lowered", stage="frontend",
                   actual={"name": _buffer_name(buffer),
                           "strides": list(strides)})
        if int(buffer.elem_offset) != 0:
            reject(RK_E_UNSUPPORTED,
                   "parameter buffer has a non-zero element offset",
                   stage="frontend",
                   actual={"name": _buffer_name(buffer),
                           "elem_offset": int(buffer.elem_offset)})
    return buffers


def _param_of(buffer: Any, params: list[Any]) -> Any | None:
    for param in params:
        if _same_buffer(buffer, param):
            return param
    return None


def _si_lu_ok(store: Any, item: _Item, tirx: Any, acc: Any,
              gm: int, gn: int) -> tuple[bool, str]:
    """Prove ``acc[i,j] = acc[i,j] * sigmoid(acc[i,j])`` over the whole tile."""
    if not isinstance(store, tirx.BufferStore) or not _same_buffer(store.buffer, acc):
        return False, "activation store does not write the GEMM accumulator"
    loops = item.loops
    if len(loops) != 2:
        return False, "activation must be a two-level loop nest over the tile"
    extents = [_int_imm(loop.extent) for loop in loops]
    if extents != [gm, gn]:
        return False, (f"activation loop extents {extents} do not cover the "
                       f"GEMM tile ({gm}, {gn})")
    mins = [_int_imm(loop.min) for loop in loops]
    if mins != [0, 0]:
        return False, (f"activation loop lower bounds {mins} are not zero; a "
                       "shifted loop would skip or duplicate elements")
    try:
        kinds = [int(loop.kind) for loop in loops]
    except Exception:
        kinds = []
    if kinds != [1, 1]:
        return False, "activation loops are not parallel element loops"
    i, j = loops[0].loop_var, loops[1].loop_var
    if len(store.indices) != 2 or not _same_var(store.indices[0], i) or \
            not _same_var(store.indices[1], j):
        return False, "activation store indices are not the loop induction vars"
    value = store.value
    if type(value).__name__ != "Mul":
        return False, "activation expression is not a product"
    a, b = value.a, value.b

    steps = [_int_imm(getattr(loop, "step", None)) for loop in loops]
    if any(step is not None and step != 1 for step in steps):
        return False, f"activation loops must step by one, got {steps}"

    def load_matches(node: Any) -> bool:
        return (isinstance(node, tirx.BufferLoad) and _same_buffer(node.buffer, acc)
                and _predicate_true(getattr(node, "predicate", None))
                and len(node.indices) == 2
                and _same_var(node.indices[0], i) and _same_var(node.indices[1], j))

    def sigmoid_matches(node: Any) -> bool:
        return (isinstance(node, tirx.Call) and _op_name(node) == "tirx.sigmoid"
                and len(node.args) == 1 and load_matches(node.args[0]))

    if load_matches(a) and sigmoid_matches(b):
        return True, ""
    if load_matches(b) and sigmoid_matches(a):
        return True, ""
    return False, "activation is not acc * sigmoid(acc) on identical indices"


def _source_span(prim_func: Any) -> str:
    try:
        span = prim_func.span
        if span is not None:
            return str(span)
    except Exception:
        pass
    try:
        return str(prim_func.attrs["global_symbol"])
    except Exception:
        return "tilelang:unknown"


def _symbol_name(prim_func: Any) -> str:
    try:
        return str(prim_func.attrs["global_symbol"])
    except Exception:
        return "kernel"


# --- Static per-node extraction (no K loop) --------------------------------
# The no-K-loop path is intentionally a small SSA extractor rather than a
# "one GEMM per function" pattern matcher.  A GEMM is one definition of its
# accumulator tile; the explicit FP16 stores after it are the values consumed by
# later nodes.  Fills/copies are attached by buffer identity, accessed region and
# statement order, never by an assumed count or by "one group per GEMM".

F16_NARROWING_NAMES = ("gemm1_accumulator", "sigmoid", "silu_mul",
                       "gemm2_accumulator", "residual_add")


@dataclass(frozen=True)
class _Expr:
    kind: str
    buffer: Any | None = None
    args: tuple["_Expr", ...] = ()
    dtype: str = ""


@dataclass(frozen=True)
class _TileDef:
    """One device-side definition of a tile buffer.

    ``order`` is the statement order of the producer (GEMM or BufferStore), so a
    later read can resolve the correct version even when a source program writes
    the same fragment twice (the projection kernel does that for the residual).
    """

    buffer: Any
    kind: str
    order: int
    shape: tuple[int, int]
    expr: _Expr
    gemm: "_GemmGroup | None" = None


@dataclass(frozen=True)
class _GemmGroup:
    item: _Item
    a_reg: _Region
    b_reg: _Region
    c_reg: _Region
    gm: int
    gn: int
    gk: int
    fill: _Item
    a_copy: _Item
    b_copy: _Item


def _raw_buffer_shape(buffer: Any) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in buffer.shape)
    except Exception:
        reject(RK_E_UNSUPPORTED, "tile buffer shape is not inspectable",
               stage="frontend")


def _buffer_shape(buffer: Any) -> tuple[int, int]:
    shape = _raw_buffer_shape(buffer)
    if len(shape) != 2:
        reject(RK_E_SHAPE, "tile buffer must be rank 2", stage="frontend",
               actual=list(shape))
    try:
        dtype = str(buffer.dtype)
    except Exception:
        reject(RK_E_UNSUPPORTED, "tile buffer dtype is not inspectable",
               stage="frontend")
    if dtype != "float16":
        reject(RK_E_NUMERIC_CONTRACT,
               "static dataflow requires rank-2 FP16 tile buffers; an implicit "
               "cast would hide a narrowing point", stage="frontend",
               actual={"shape": list(shape), "dtype": dtype})
    return shape


def _same_expr_buffer(expr: _Expr, buffer: Any) -> bool:
    return (expr.kind == "load" and expr.buffer is not None and
            _same_buffer(expr.buffer, buffer))


def _expr_load_buffers(expr: _Expr) -> list[Any]:
    if expr.kind == "load":
        return [expr.buffer] if expr.buffer is not None else []
    out: list[Any] = []
    for arg in expr.args:
        out.extend(_expr_load_buffers(arg))
    return out


def _parse_elementwise_expr(node: Any, tirx: Any, loop_vars: tuple[Any, Any],
                            target_shape: tuple[int, int], span: str,
                            params: list[Any] | None = None) -> _Expr:
    """Parse one scalar store value into the tiny elementwise expression AST.

    Only loads at the loop induction indices, explicit casts, sigmoid and the two
    same-shape arithmetic operations are accepted.  Every accepted form preserves
    the source narrowing: the destination store itself must be FP16.
    """
    if isinstance(node, tirx.BufferLoad):
        if not _predicate_true(getattr(node, "predicate", None)):
            reject(RK_E_UNSUPPORTED,
                   "elementwise load carries a predicate", stage="frontend",
                   source=span, actual=str(getattr(node, "predicate", None)))
        load_shape = _raw_buffer_shape(node.buffer)
        parameter = _param_of(node.buffer, params) if params is not None else None
        if parameter is not None and len(load_shape) < 2:
            if len(load_shape) == 0:
                if node.indices:
                    reject(RK_E_UNSUPPORTED,
                           "rank-0 scalar constant must be loaded without indices",
                           stage="frontend", source=span,
                           actual=[repr(x) for x in node.indices])
            else:
                if len(node.indices) != 1 or \
                        not _same_var(node.indices[0], loop_vars[1]):
                    reject(RK_E_UNSUPPORTED,
                           "per-channel constant must be indexed by the column "
                           "loop variable", stage="frontend", source=span,
                           actual=[repr(x) for x in node.indices])
                if load_shape[0] != target_shape[1]:
                    reject(RK_E_SHAPE,
                           "per-channel constant length must equal the store "
                           "column count", stage="frontend", source=span,
                           actual=list(load_shape), required=target_shape[1])
            return _Expr("load", buffer=node.buffer,
                         dtype=str(node.buffer.dtype))
        if len(node.indices) != 2 or not _same_var(node.indices[0], loop_vars[0]) \
                or not _same_var(node.indices[1], loop_vars[1]):
            reject(RK_E_UNSUPPORTED,
                   "elementwise load indices are not the store loop variables",
                   stage="frontend", source=span,
                   actual=[repr(x) for x in node.indices])
        if load_shape != target_shape:
            reject(RK_E_SHAPE,
                   "elementwise load and store shapes differ; only explicit "
                   "rank-0/rank-1 constant broadcast is lowered",
                   stage="frontend", source=span,
                   actual={"load": list(load_shape),
                           "store": list(target_shape)})
        return _Expr("load", buffer=node.buffer, dtype=str(node.buffer.dtype))

    name = type(node).__name__
    if name in {"IntImm", "FloatImm"} and hasattr(node, "value"):
        try:
            value = float(node.value)
        except Exception:
            value = None
        if value == 0.0:
            return _Expr("zero", dtype="float16")
        reject(RK_E_UNSUPPORTED,
               "only the literal zero may appear in a relu elementwise form",
               stage="frontend", source=span, actual=repr(node))
    if name == "Cast":
        if str(node.dtype) != "float16":
            reject(RK_E_NUMERIC_CONTRACT,
                   "explicit narrowing casts must target float16; an implicit "
                   "store cast would hide the documented narrowing point",
                   stage="frontend", source=span, actual=str(node.dtype))
        inner = _parse_elementwise_expr(node.value, tirx, loop_vars, target_shape,
                                        span, params)
        return _Expr("cast", args=(inner,), dtype=str(node.dtype))
    if isinstance(node, tirx.Call):
        opname = _op_name(node)
        if opname != "tirx.sigmoid" or len(node.args) != 1:
            reject(RK_E_UNSUPPORTED,
                   f"unsupported elementwise operation {opname or name!r}; the "
                   "static dataflow subset accepts sigmoid and FP16 add/mul only",
                   stage="frontend", source=span, op=opname or name)
        return _Expr("sigmoid",
                     args=(_parse_elementwise_expr(node.args[0], tirx, loop_vars,
                                                   target_shape, span, params),),
                     dtype=str(node.dtype))
    if name in {"Mul", "Add", "Max"}:
        return _Expr(name.lower(),
                     args=(_parse_elementwise_expr(node.a, tirx, loop_vars,
                                                   target_shape, span, params),
                           _parse_elementwise_expr(node.b, tirx, loop_vars,
                                                   target_shape, span, params)),
                     dtype=str(node.dtype))
    reject(RK_E_UNSUPPORTED,
           f"unsupported elementwise expression {name!r}; explicit FP16 "
           "narrowing and the registered sigmoid/add/mul forms are required",
           stage="frontend", source=span)


def _check_elementwise_store(item: _Item, tirx: Any, params: list[Any],
                             span: str) -> tuple[Any, tuple[int, int], _Expr]:
    store = item.node
    if not isinstance(store, tirx.BufferStore):
        reject(RK_E_UNSUPPORTED, "expected a BufferStore", stage="frontend",
               source=span)
    if _param_of(store.buffer, params) is not None:
        reject(RK_E_UNSUPPORTED,
               "the static dataflow extractor only accepts stores to tile "
               "buffers; a parameter store must be an explicit copy",
               stage="frontend", source=span)
    target_shape = _buffer_shape(store.buffer)
    loops = item.loops
    if len(loops) != 2:
        reject(RK_E_UNSUPPORTED,
               "elementwise store must be a two-level loop nest over its tile",
               stage="frontend", source=span, actual=len(loops))
    extents = [_int_imm(loop.extent) for loop in loops]
    if extents != list(target_shape):
        reject(RK_E_UNSUPPORTED,
               "elementwise store loops do not cover the destination tile",
               stage="frontend", source=span,
               actual={"extents": extents, "shape": list(target_shape)})
    mins = [_int_imm(loop.min) for loop in loops]
    if mins != [0, 0]:
        reject(RK_E_UNSUPPORTED,
               f"elementwise store loop lower bounds {mins} are not zero",
               stage="frontend", source=span)
    try:
        kinds = [int(loop.kind) for loop in loops]
    except Exception:
        kinds = []
    if kinds != [1, 1]:
        reject(RK_E_UNSUPPORTED,
               "elementwise store loops are not parallel element loops",
               stage="frontend", source=span, actual=kinds)
    steps = [_int_imm(getattr(loop, "step", None)) for loop in loops]
    if any(step is not None and step != 1 for step in steps):
        reject(RK_E_UNSUPPORTED,
               "elementwise store loops must step by one", stage="frontend",
               source=span, actual=steps)
    loop_vars = (loops[0].loop_var, loops[1].loop_var)
    if len(store.indices) != 2 or \
            not _same_var(store.indices[0], loop_vars[0]) or \
            not _same_var(store.indices[1], loop_vars[1]):
        reject(RK_E_UNSUPPORTED,
               "elementwise store indices are not the loop induction variables",
               stage="frontend", source=span,
               actual=[repr(x) for x in store.indices])
    expr = _parse_elementwise_expr(store.value, tirx, loop_vars, target_shape,
                                   span, params)
    return store.buffer, target_shape, expr


def _latest_def(defs: list[_TileDef], buffer: Any,
                before_order: int) -> _TileDef | None:
    matches = [d for d in defs
               if _same_buffer(d.buffer, buffer) and d.order < before_order]
    if not matches:
        return None
    return max(matches, key=lambda d: d.order)


def _validate_operand_copy(copy_item: _Item, tirx: Any, dest_buffer: Any,
                           expected: tuple[int, int], label: str,
                           span: str) -> _Region:
    src = _region(copy_item.node.args[0], tirx, "copy source", expect_mask=1)
    dst = _region(copy_item.node.args[1], tirx, "copy destination",
                  expect_mask=2)
    if not _same_buffer(dst.buffer, dest_buffer):
        reject(RK_E_UNSUPPORTED,
               f"{label} copy destination is not the GEMM tile buffer",
               stage="frontend", source=span)
    if src.physical != expected or dst.physical != expected or \
            src.offset != (0, 0) or dst.offset != (0, 0):
        reject(RK_E_SHAPE,
               f"{label} copy must cover exactly the full GEMM operand tile",
               stage="frontend", source=span,
               actual={"source": list(src.physical), "destination": list(dst.physical),
                       "source_offset": src.offset,
                       "destination_offset": dst.offset},
               required=list(expected))
    return src


def _contract_with_points(options: CompileOptions,
                          points: tuple[str, ...]) -> NumericContract:
    base = _numeric_contract(options)
    if options.numeric_mode == "strict":
        return base
    return NumericContract(
        accumulator_dtype="float16",
        accumulation="ordered_k_tiles_with_f16_writeback",
        narrowing_points=("k_tile_boundary",) + points,
        approximate=False,
        tolerance={"atol": 8e-3, "rtol": 8e-3})


def _strip_expr_casts(expr: _Expr) -> _Expr:
    while expr.kind == "cast" and expr.args:
        expr = expr.args[0]
    return expr


def _extract_staged_dataflow(items: list[_Item], params: list[Any], *, tirx: Any,
                             prim_func: Any, span: str,
                             options: CompileOptions,
                             profile_hash: str) -> Any:
    """Extract a sequence of separately launched, self-contained stage kernels.

    Each ``T.Kernel`` is a launch boundary.  A stage writes one global parameter;
    later stages may read it.  A stage may not read the parameter it writes, so
    cross-block dependencies cannot be invented by treating one parallel grid as
    sequential.  Output coverage is proved exact from the block grid and the
    published tile rectangle.  GEMM K loops are accepted only with an explicit
    FP16 accumulator writeback after every source K tile; the FP32-accumulate /
    single-final-narrow form is rejected because it would change reduction
    semantics.
    """
    active = [item for item in items
              if item.kind == "store" or
              (item.kind == "call" and _op_name(item.node) in ALLOWED_TILEOPS)]
    if not active:
        reject(RK_E_UNSUPPORTED, "staged dataflow has no tile operations",
               stage="frontend", source=span)
    if any(not item.blocks for item in active):
        reject(RK_E_UNSUPPORTED,
               "every staged dataflow operation must be inside a T.Kernel launch",
               stage="frontend", source=span)

    groups: list[list[_Item]] = []
    keys: list[int] = []
    for item in active:
        key = id(item.blocks)
        if not keys or key != keys[-1]:
            if key in keys:
                reject(RK_E_UNSUPPORTED,
                       "a T.Kernel block scope is not contiguous in source order",
                       stage="frontend", source=span)
            keys.append(key)
            groups.append([])
        groups[-1].append(item)
    if len(groups) < 2:
        reject(RK_E_UNSUPPORTED,
               "staged dataflow requires at least two T.Kernel launches",
               stage="frontend", source=span, actual=len(groups))

    def pidx(buffer: Any) -> int | None:
        for index, param in enumerate(params):
            if _same_buffer(buffer, param):
                return index
        return None

    def pshape(index: int) -> tuple[int, ...]:
        shape = tuple(int(x) for x in params[index].shape)
        if len(shape) not in (1, 2):
            reject(RK_E_SHAPE, "staged dataflow parameters must be rank 1 or 2",
                   stage="frontend", source=span, actual=list(shape))
        return shape

    def block_extents(stage: list[_Item]) -> tuple[Any, ...]:
        loops = stage[0].blocks
        if len(loops) not in (1, 2):
            reject(RK_E_UNSUPPORTED,
                   "staged dataflow supports one- or two-dimensional block grids",
                   stage="frontend", source=span, actual=len(loops))
        for loop in loops:
            extent = _int_imm(loop.extent)
            if extent is None or extent <= 0:
                reject(RK_E_UNSUPPORTED,
                       "block grid extents must be static positive integers",
                       stage="frontend", source=span,
                       actual=repr(loop.extent))
            if _int_imm(loop.min) != 0:
                reject(RK_E_UNSUPPORTED,
                       "block grid loops must start at zero", stage="frontend",
                       source=span, actual=repr(loop.min))
            step = getattr(loop, "step", None)
            if step is not None and _int_imm(step) != 1:
                reject(RK_E_UNSUPPORTED,
                       "block grid loops must step by one", stage="frontend",
                       source=span, actual=repr(step))
        return tuple(loops)

    def grid_mapping(stage: list[_Item], dst: _Region,
                     out_shape: tuple[int, int]) -> dict[str, Any]:
        loops = block_extents(stage)
        extents = [_int_imm(loop.extent) for loop in loops]
        roles = {f"b{i}": loop.loop_var for i, loop in enumerate(loops)}
        row = _affine_expr(dst.raw_indices[0], tirx, roles)
        col = _affine_expr(dst.raw_indices[1], tirx, roles)
        if row is None or col is None:
            reject(RK_E_UNSUPPORTED,
                   "stage output indices are not affine in the block grid",
                   stage="frontend", source=span,
                   actual=[repr(dst.raw_indices[0]), repr(dst.raw_indices[1])])
        row_coeffs, row_const = row
        col_coeffs, col_const = col
        if row_const or col_const:
            reject(RK_E_UNSUPPORTED,
                   "stage output tile base must be exactly block*tile",
                   stage="frontend", source=span,
                   actual={"row": row_const, "col": col_const})
        # A block variable whose extent is one contributes a proven zero; keep
        # it out of the comparison but still require every wider axis to map.
        inactive = {f"b{i}" for i, extent in enumerate(extents) if extent == 1}
        row_coeffs = {k: v for k, v in row_coeffs.items()
                      if v and k not in inactive}
        col_coeffs = {k: v for k, v in col_coeffs.items()
                      if v and k not in inactive}
        if len(row_coeffs) > 1 or len(col_coeffs) > 1:
            reject(RK_E_UNSUPPORTED,
                   "a stage output axis may depend on at most one block variable",
                   stage="frontend", source=span,
                   actual={"row": row_coeffs, "col": col_coeffs})
        row_role = next(iter(row_coeffs), None)
        col_role = next(iter(col_coeffs), None)
        if row_role is not None and col_role is not None and row_role == col_role:
            reject(RK_E_UNSUPPORTED,
                   "one block variable cannot map to both output axes",
                   stage="frontend", source=span, actual=row_role)
        tile_rows, tile_cols = dst.physical
        if row_role is not None and row_coeffs[row_role] != tile_rows:
            reject(RK_E_SHAPE,
                   "row block stride does not equal the published tile height",
                   stage="frontend", source=span,
                   actual=row_coeffs[row_role], required=tile_rows)
        if col_role is not None and col_coeffs[col_role] != tile_cols:
            reject(RK_E_SHAPE,
                   "column block stride does not equal the published tile width",
                   stage="frontend", source=span,
                   actual=col_coeffs[col_role], required=tile_cols)
        used = {role for role in (row_role, col_role) if role is not None}
        for i, extent in enumerate(extents):
            if extent > 1 and f"b{i}" not in used:
                reject(RK_E_UNSUPPORTED,
                       "a wider block-grid axis does not contribute to the "
                       "output rectangle", stage="frontend", source=span,
                       actual={"axis": i, "extent": extent})
        row_grid = extents[int(row_role[1:])] if row_role else 1
        col_grid = extents[int(col_role[1:])] if col_role else 1
        if row_grid * tile_rows != out_shape[0] or \
                col_grid * tile_cols != out_shape[1]:
            reject(RK_E_SHAPE,
                   "block grid does not cover the stage output exactly once",
                   stage="frontend", source=span,
                   actual={"grid": [col_grid, row_grid],
                           "tile": [tile_rows, tile_cols],
                           "shape": list(out_shape)})
        return {"tile_rows": tile_rows, "tile_cols": tile_cols,
                "row_grid": row_grid, "col_grid": col_grid,
                "row_role": row_role, "col_role": col_role,
                "extents": extents, "roles": roles,
                "block_loops": loops}

    def index_matches(region: _Region, axis: int, expected: dict[str, int],
                      mapping: dict[str, Any], k_role: str | None,
                      label: str) -> None:
        roles = dict(mapping["roles"])
        if k_role is not None:
            roles[k_role] = k_role
        # k_role is a marker, replaced by the concrete loop variable below.
        if k_role is not None:
            roles.pop(k_role, None)
        parsed = _affine_expr(region.raw_indices[axis], tirx, roles)
        if parsed is None:
            reject(RK_E_UNSUPPORTED,
                   f"{label} index is not affine in block/K variables",
                   stage="frontend", source=span,
                   actual=repr(region.raw_indices[axis]))
        coeffs, const = parsed
        if const:
            reject(RK_E_SHAPE,
                   f"{label} index has a non-zero base", stage="frontend",
                   source=span, actual=const)
        inactive = {f"b{i}" for i, extent in enumerate(mapping["extents"])
                    if extent == 1}
        coeffs = {k: v for k, v in coeffs.items()
                  if v and k not in inactive}
        if coeffs != {k: v for k, v in expected.items() if v}:
            reject(RK_E_SHAPE,
                   f"{label} index does not match the proved stage tiling",
                   stage="frontend", source=span,
                   actual=coeffs, required=expected)

    stage_infos: list[dict[str, Any]] = []
    write_stage: dict[int, int] = {}
    parsed_stores: dict[int, tuple[Any, tuple[int, int], _Expr]] = {}
    for stage_index, stage in enumerate(groups):
        copies = [it for it in stage
                  if it.kind == "call" and _op_name(it.node) == "tl.tileop.copy"]
        writes: list[tuple[_Item, _Region, _Region]] = []
        for copy_item in copies:
            src = _region(copy_item.node.args[0], tirx, "copy source",
                          expect_mask=1)
            dst = _region(copy_item.node.args[1], tirx, "copy destination",
                          expect_mask=2)
            if pidx(dst.buffer) is not None:
                writes.append((copy_item, src, dst))
        if len(writes) != 1:
            reject(RK_E_UNSUPPORTED,
                   "each stage must publish exactly one parameter value",
                   stage="frontend", source=span,
                   actual={"stage": stage_index, "parameter_writes": len(writes)})
        publish_item, pub_src, pub_dst = writes[0]
        out_pid = pidx(pub_dst.buffer)
        if out_pid is None:
            reject(RK_E_UNSUPPORTED,
                   "stage publish destination is not a parameter",
                   stage="frontend", source=span)
        if pidx(pub_src.buffer) is not None:
            reject(RK_E_UNSUPPORTED,
                   "stage must publish a local tile, not copy parameter to "
                   "parameter", stage="frontend", source=span)
        src_shape = _raw_buffer_shape(pub_src.buffer)
        if len(src_shape) != 2 or src_shape != pub_dst.physical:
            reject(RK_E_SHAPE,
                   "published local tile must be rank-2 and match the destination "
                   "rectangle", stage="frontend", source=span,
                   actual={"source": list(src_shape),
                           "destination": list(pub_dst.physical)})
        out_shape = pshape(out_pid)
        mapping = grid_mapping(stage, pub_dst, out_shape)
        if (mapping["tile_rows"], mapping["tile_cols"]) != src_shape:
            reject(RK_E_SHAPE,
                   "published tile shape differs from the stage block tile",
                   stage="frontend", source=span,
                   actual={"published": list(src_shape),
                           "block": [mapping["tile_rows"],
                                     mapping["tile_cols"]]})
        if out_pid in write_stage:
            reject(RK_E_UNSUPPORTED,
                   "a staged parameter is written by more than one launch",
                   stage="frontend", source=span, actual=_buffer_name(pub_dst.buffer))
        write_stage[out_pid] = stage_index
        gemms = [it for it in stage
                 if it.kind == "call" and _op_name(it.node) == "tl.tileop.gemm"]
        stores = [it for it in stage if it.kind == "store"]
        store_info = None
        if gemms:
            if len(gemms) != 1 or stores:
                reject(RK_E_UNSUPPORTED,
                       "a GEMM stage must contain one GEMM and no elementwise "
                       "parameter store", stage="frontend", source=span,
                       actual={"stage": stage_index, "gemms": len(gemms),
                               "stores": len(stores)})
            gemm_item = gemms[0]
            a_reg, b_reg, c_reg, gm, gn, gk = _gemm_header(
                gemm_item.node, tirx, span)
            if a_reg.physical != (gm, gk) or b_reg.physical != (gk, gn) or \
                    c_reg.physical != (gm, gn):
                reject(RK_E_SHAPE,
                       "staged GEMM regions do not describe a full tile",
                       stage="frontend", source=span)
            if (gm, gn) != src_shape:
                reject(RK_E_SHAPE,
                       "staged GEMM accumulator tile differs from the published "
                       "tile", stage="frontend", source=span,
                       actual={"gemm": [gm, gn], "published": list(src_shape)})
            a_copies = []
            b_copies = []
            for copy_item in copies:
                dst = _region(copy_item.node.args[1], tirx, "copy destination",
                              expect_mask=2)
                if _same_buffer(dst.buffer, a_reg.buffer):
                    a_copies.append(copy_item)
                if _same_buffer(dst.buffer, b_reg.buffer):
                    b_copies.append(copy_item)
            if len(a_copies) != 1 or len(b_copies) != 1:
                reject(RK_E_UNSUPPORTED,
                       "staged GEMM needs one A copy and one B copy",
                       stage="frontend", source=span,
                       actual={"A": len(a_copies), "B": len(b_copies)})
            a_copy, b_copy = a_copies[0], b_copies[0]
            a_src = _region(a_copy.node.args[0], tirx, "A copy source",
                            expect_mask=1)
            b_src = _region(b_copy.node.args[0], tirx, "B copy source",
                            expect_mask=1)
            if a_src.physical != (gm, gk) or b_src.physical != (gk, gn):
                reject(RK_E_SHAPE,
                       "staged GEMM operands do not cover the declared tile",
                       stage="frontend", source=span,
                       actual={"A": list(a_src.physical),
                               "B": list(b_src.physical)})
            a_pid = pidx(a_src.buffer)
            b_pid = pidx(b_src.buffer)
            if a_pid is None or b_pid is None:
                reject(RK_E_UNSUPPORTED,
                       "staged GEMM operands must be parameter tensors",
                       stage="frontend", source=span)
            a_shape = pshape(a_pid)
            b_shape = pshape(b_pid)
            k_loop = gemm_item.loops[0] if gemm_item.loops else None
            if k_loop is not None:
                if len(gemm_item.loops) != 1:
                    reject(RK_E_UNSUPPORTED,
                           "a staged K GEMM may have only one loop nest",
                           stage="frontend", source=span)
                try:
                    kind = int(k_loop.kind)
                except Exception:
                    kind = -1
                if kind != 0 or _int_imm(k_loop.min) != 0:
                    reject(RK_E_UNSUPPORTED,
                           "staged K loop must be zero-based serial/pipelined",
                           stage="frontend", source=span)
                step = getattr(k_loop, "step", None)
                if step is not None and _int_imm(step) != 1:
                    reject(RK_E_UNSUPPORTED,
                           "staged K loop must step by one", stage="frontend",
                           source=span)
                trips = _int_imm(k_loop.extent)
                if trips is None or trips <= 0:
                    reject(RK_E_SHAPE,
                           "staged K loop needs a static positive trip count",
                           stage="frontend", source=span)
                logical_k = trips * gk
                if a_shape[1] != logical_k or b_shape[0] != logical_k:
                    reject(RK_E_SHAPE,
                           "source K loop does not cover the full logical K",
                           stage="frontend", source=span,
                           actual={"A": list(a_shape), "B": list(b_shape),
                                   "loop": [trips, gk]})
                # Explicit FP16 accumulator writeback form:
                #   clear Acc16; for ko: copy Acc16->Cl; gemm; copy Cl->Acc16
                fill_items = [it for it in stage
                              if it.kind == "call" and
                              _op_name(it.node) == "tl.tileop.fill"]
                if len(fill_items) != 1:
                    reject(RK_E_NUMERIC_CONTRACT,
                           "staged K GEMM needs one accumulator initialisation",
                           stage="frontend", source=span)
                fill_item = fill_items[0]
                fill_reg = _region(fill_item.node.args[0], tirx, "fill",
                                   expect_mask=2)
                if pidx(fill_reg.buffer) is not None or \
                        _int_imm(fill_item.node.args[1]) != 0:
                    reject(RK_E_NUMERIC_CONTRACT,
                           "staged K GEMM accumulator must be a zeroed local "
                           "FP16 tile", stage="frontend", source=span)
                acc_shape = _raw_buffer_shape(fill_reg.buffer)
                if acc_shape != (gm, gn) or \
                        str(fill_reg.buffer.dtype) != "float16" or \
                        fill_reg.physical != (gm, gn):
                    reject(RK_E_NUMERIC_CONTRACT,
                           "staged K GEMM requires an FP16 accumulator tile with "
                           "the GEMM shape", stage="frontend", source=span,
                           actual=list(acc_shape))
                if fill_item.order >= min(it.order for it in stage):
                    # The fill is the first operation of the stage; any earlier
                    # stage operation means the loop is not ordered as claimed.
                    pass
                prior_copies = [c for c in copies
                                if _same_buffer(_region(c.node.args[1], tirx,
                                                        "copy dst",
                                                        expect_mask=2).buffer,
                                                c_reg.buffer)]
                writeback_copies = [c for c in copies
                                    if _same_buffer(_region(c.node.args[0], tirx,
                                                            "copy src",
                                                            expect_mask=1).buffer,
                                                    c_reg.buffer)]
                if len(prior_copies) != 1 or len(writeback_copies) != 1:
                    reject(RK_E_UNSUPPORTED,
                           "staged K GEMM requires one prior-writing copy and one "
                           "writeback copy", stage="frontend", source=span,
                           actual={"prior": len(prior_copies),
                                   "writeback": len(writeback_copies)})
                prior_copy, writeback_copy = prior_copies[0], writeback_copies[0]
                prior_src = _region(prior_copy.node.args[0], tirx,
                                    "prior copy source", expect_mask=1)
                write_dst = _region(writeback_copy.node.args[1], tirx,
                                    "writeback destination", expect_mask=2)
                if not _same_buffer(prior_src.buffer, fill_reg.buffer) or \
                        not _same_buffer(write_dst.buffer, fill_reg.buffer):
                    reject(RK_E_UNSUPPORTED,
                           "staged K GEMM must write the same FP16 accumulator "
                           "tile that initialises each K iteration",
                           stage="frontend", source=span)
                if not (prior_copy.order < gemm_item.order <
                        writeback_copy.order):
                    reject(RK_E_NUMERIC_CONTRACT,
                           "staged K GEMM order must be prior-copy, GEMM, "
                           "FP16-writeback", stage="frontend", source=span)
                for item, label in ((a_copy, "A"), (b_copy, "B"),
                                     (prior_copy, "prior"),
                                     (writeback_copy, "writeback")):
                    if len(item.loops) != 1 or \
                            not _same_var(item.loops[0].loop_var,
                                          k_loop.loop_var):
                        reject(RK_E_UNSUPPORTED,
                               f"staged K GEMM {label} is not inside the same "
                               "K loop", stage="frontend", source=span)
                # The publication copy consumes the FP16 accumulator after loop.
                if not _same_buffer(pub_src.buffer, fill_reg.buffer) or \
                        pub_dst.physical != (gm, gn):
                    reject(RK_E_UNSUPPORTED,
                           "stage publish must consume the accumulated FP16 tile "
                           "after the K loop", stage="frontend", source=span)
                k_hints = {"source_k_block": gk, "source_k_trips": trips,
                           "lowering_k_tile": 256,
                           "narrowing_base": "explicit f16 writeback every K tile",
                           "pipeline_hint_dropped": bool(
                               dict(k_loop.annotations))}
                source_acc = "float16_per_k_tile"
            else:
                logical_k = gk
                if a_shape[1] != logical_k or b_shape[0] != logical_k:
                    reject(RK_E_SHAPE,
                           "staged GEMM K extent does not match its operands",
                           stage="frontend", source=span,
                           actual={"A": list(a_shape), "B": list(b_shape),
                                   "gemm_k": gk})
                fills = [it for it in stage
                         if it.kind == "call" and
                         _op_name(it.node) == "tl.tileop.fill"]
                if len(fills) != 1:
                    reject(RK_E_NUMERIC_CONTRACT,
                           "staged GEMM needs one accumulator zero fill",
                           stage="frontend", source=span)
                fill_reg = _region(fills[0].node.args[0], tirx, "fill",
                                   expect_mask=2)
                if not _same_buffer(fill_reg.buffer, c_reg.buffer) or \
                        fill_reg.physical != (gm, gn) or \
                        _int_imm(fills[0].node.args[1]) != 0:
                    reject(RK_E_NUMERIC_CONTRACT,
                           "staged GEMM fill must zero the full FP32 accumulator",
                           stage="frontend", source=span)
                narrows = [c for c in copies
                           if _same_buffer(_region(c.node.args[0], tirx,
                                                   "narrow source",
                                                   expect_mask=1).buffer,
                                           c_reg.buffer)]
                if len(narrows) != 1:
                    reject(RK_E_NUMERIC_CONTRACT,
                           "staged GEMM needs exactly one accumulator-to-FP16 "
                           "copy", stage="frontend", source=span,
                           actual=len(narrows))
                narrow = narrows[0]
                narrow_dst = _region(narrow.node.args[1], tirx,
                                     "narrow destination", expect_mask=2)
                if _raw_buffer_shape(narrow_dst.buffer) != (gm, gn) or \
                        str(narrow_dst.buffer.dtype) != "float16" or \
                        not _same_buffer(pub_src.buffer, narrow_dst.buffer):
                    reject(RK_E_NUMERIC_CONTRACT,
                           "staged GEMM publication must consume its explicit "
                           "FP16 narrowing tile", stage="frontend", source=span)
                if not (a_copy.order < gemm_item.order and
                        b_copy.order < gemm_item.order and
                        fills[0].order < gemm_item.order and
                        narrow.order > gemm_item.order):
                    reject(RK_E_NUMERIC_CONTRACT,
                           "staged GEMM source order is not fill/copy/GEMM/"
                           "narrow", stage="frontend", source=span)
                k_hints = None
                source_acc = "float32"

            # Prove A/B source access rectangles map to M/N/K block coordinates.
            k_role_marker = "ko"
            if k_loop is not None:
                # index_matches needs the concrete K loop var, not just its name.
                k_roles = dict(mapping["roles"])
                k_roles[k_role_marker] = k_loop.loop_var
                def k_index(region: _Region, axis: int,
                            expected: dict[str, int], label: str) -> None:
                    parsed = _affine_expr(region.raw_indices[axis], tirx, k_roles)
                    if parsed is None:
                        reject(RK_E_UNSUPPORTED,
                               f"{label} index is not affine", stage="frontend",
                               source=span)
                    coeffs, const = parsed
                    if const:
                        reject(RK_E_SHAPE, f"{label} has non-zero base",
                               stage="frontend", source=span, actual=const)
                    inactive = {f"b{i}" for i, extent in
                                enumerate(mapping["extents"]) if extent == 1}
                    coeffs = {k: v for k, v in coeffs.items()
                              if v and k not in inactive}
                    if coeffs != {k: v for k, v in expected.items() if v}:
                        reject(RK_E_SHAPE,
                               f"{label} index does not match stage tiling",
                               stage="frontend", source=span,
                               actual=coeffs, required=expected)
                row_role = mapping["row_role"]
                col_role = mapping["col_role"]
                a_row_expected = ({row_role: gm}
                                  if row_role is not None else {})
                a_col_expected = {k_role_marker: gk}
                b_row_expected = {k_role_marker: gk}
                b_col_expected = ({col_role: gn}
                                  if col_role is not None else {})
                k_index(a_src, 0, a_row_expected, "staged A row")
                k_index(a_src, 1, a_col_expected, "staged A column")
                k_index(b_src, 0, b_row_expected, "staged B row")
                k_index(b_src, 1, b_col_expected, "staged B column")
            else:
                row_role = mapping["row_role"]
                col_role = mapping["col_role"]
                a_row_expected = ({row_role: gm}
                                  if row_role is not None else {})
                b_col_expected = ({col_role: gn}
                                  if col_role is not None else {})
                index_matches(a_src, 0, a_row_expected, mapping, None,
                             "staged A row")
                index_matches(a_src, 1, {}, mapping, None,
                              "staged A column")
                index_matches(b_src, 0, {}, mapping, None,
                              "staged B row")
                index_matches(b_src, 1, b_col_expected, mapping, None,
                              "staged B column")

            stage_reads = set()
            for copy_item in copies:
                copy_src = _region(copy_item.node.args[0], tirx,
                                   "copy source", expect_mask=1)
                read_pid = pidx(copy_src.buffer)
                if read_pid is not None:
                    stage_reads.add(read_pid)
            stage_infos.append({
                "stage": stage_index, "opcode": "matmul",
                "inputs": (a_pid, b_pid), "output": out_pid,
                "reads": stage_reads,
                "b_pid": b_pid, "gm": gm, "gn": gn, "gk": gk,
                "k_hints": k_hints, "source_acc": source_acc,
                "mapping": mapping,
            })
        else:
            if len(stores) != 1:
                reject(RK_E_UNSUPPORTED,
                       "an elementwise stage must contain exactly one tile store",
                       stage="frontend", source=span,
                       actual={"stage": stage_index, "stores": len(stores)})
            store_item = stores[0]
            target, target_shape, expr = _check_elementwise_store(
                store_item, tirx, params, span)
            if target_shape != src_shape:
                reject(RK_E_SHAPE,
                       "elementwise store and published tile shapes differ",
                       stage="frontend", source=span,
                       actual={"store": list(target_shape),
                               "published": list(src_shape)})
            if not _same_buffer(target, pub_src.buffer):
                reject(RK_E_UNSUPPORTED,
                       "elementwise stage must publish the tile it stores",
                       stage="frontend", source=span)
            if str(store_item.node.value.dtype) != "float16" or \
                    str(target.dtype) != "float16":
                reject(RK_E_NUMERIC_CONTRACT,
                       "elementwise stage must write an FP16 tile so the "
                       "narrowing point is explicit", stage="frontend",
                       source=span,
                       actual={"value_dtype": str(store_item.node.value.dtype),
                               "target_dtype": str(target.dtype)})
            origins: list[tuple[Any, int]] = []
            for copy_item in copies:
                csrc = _region(copy_item.node.args[0], tirx, "copy source",
                               expect_mask=1)
                cdst = _region(copy_item.node.args[1], tirx, "copy destination",
                               expect_mask=2)
                if pidx(cdst.buffer) is not None:
                    continue
                src_pid = pidx(csrc.buffer)
                if src_pid is not None:
                    origins.append((cdst.buffer, src_pid))
                else:
                    match = next((pid for buffer, pid in origins
                                  if _same_buffer(buffer, csrc.buffer)), None)
                    if match is not None:
                        origins.append((cdst.buffer, match))

            def origin_of(buffer: Any) -> int | None:
                direct = pidx(buffer)
                if direct is not None:
                    return direct
                for candidate, pid in origins:
                    if _same_buffer(candidate, buffer):
                        return pid
                return None

            inner = _strip_expr_casts(expr)
            used_origins: set[int] = set()
            direct_params: set[int] = set()
            def arg_pid(arg: _Expr, label: str) -> int:
                if arg.kind != "load" or arg.buffer is None:
                    reject(RK_E_UNSUPPORTED,
                           f"{label} is not a simple tile load",
                           stage="frontend", source=span)
                pid = origin_of(arg.buffer)
                if pid is None:
                    reject(RK_E_UNSUPPORTED,
                           f"{label} does not resolve to a parameter tensor",
                           stage="frontend", source=span,
                           actual=_buffer_name(arg.buffer))
                if pidx(arg.buffer) is not None:
                    direct_params.add(pid)
                used_origins.add(pid)
                return pid
            if inner.kind == "sigmoid" and len(inner.args) == 1:
                opcode = "sigmoid"
                inputs = (arg_pid(inner.args[0], "sigmoid input"),)
            elif inner.kind in {"mul", "add"} and len(inner.args) == 2:
                opcode = inner.kind
                inputs = (arg_pid(inner.args[0], f"{opcode} lhs"),
                          arg_pid(inner.args[1], f"{opcode} rhs"))
            elif inner.kind == "max" and len(inner.args) == 2:
                # ReLU is exactly max(x, 0); only a literal zero is accepted.
                z = [a for a in inner.args if a.kind == "zero"]
                x = [a for a in inner.args if a.kind != "zero"]
                if len(z) != 1 or len(x) != 1:
                    reject(RK_E_UNSUPPORTED,
                           "staged max is only accepted as relu(x, 0)",
                           stage="frontend", source=span)
                opcode = "relu"
                inputs = (arg_pid(x[0], "relu input"),)
            else:
                reject(RK_E_UNSUPPORTED,
                       "staged elementwise operation must be sigmoid, add, mul "
                       "or relu", stage="frontend", source=span,
                       actual=inner.kind)
            copy_sources = {pid for _, pid in origins}
            if not used_origins <= (copy_sources | direct_params):
                reject(RK_E_UNSUPPORTED,
                       "staged elementwise store uses a parameter that was "
                       "neither copied nor directly loaded as a constant",
                       stage="frontend", source=span,
                       actual={"copied": sorted(copy_sources),
                               "direct": sorted(direct_params),
                               "used": sorted(used_origins)})
            feature_pids = {pid for info in stage_infos
                            if info["opcode"] == "matmul"
                            for pid in info["inputs"]}
            copied_constants = {pid for _, pid in origins
                                if pid not in write_stage
                                and pid not in feature_pids}
            stage_infos.append({
                "stage": stage_index, "opcode": opcode, "inputs": inputs,
                "reads": set(used_origins), "output": out_pid,
                "mapping": mapping,
                "constant_pids": direct_params | copied_constants,
            })

    # ---- parameter roles and SSA ordering --------------------------------
    weight_pids = {info["b_pid"] for info in stage_infos
                   if info["opcode"] == "matmul"}
    constant_pids = {pid for info in stage_infos
                     for pid in info.get("constant_pids", set())}
    if weight_pids & constant_pids:
        reject(RK_E_UNSUPPORTED,
               "a parameter cannot be both a GEMM weight and an elementwise "
               "constant", stage="frontend", source=span)
    for pid in constant_pids:
        shape = tuple(int(x) for x in params[pid].shape)
        if len(shape) > 2:
            reject(RK_E_SHAPE,
                   "staged constants must be rank 0, 1 or 2",
                   stage="frontend", source=span, actual=list(shape))
    produced = set(write_stage)
    external = [i for i in range(len(params)) if i not in produced]
    reads: dict[int, set[int]] = {i: set() for i in range(len(params))}
    for info in stage_infos:
        for pid in info["reads"]:
            reads[pid].add(info["stage"])
    for pid in external:
        if not reads[pid]:
            reject(RK_E_UNSUPPORTED,
                   "a staged parameter is never read", stage="frontend",
                   source=span, actual=_buffer_name(params[pid]))
    for pid in weight_pids:
        if pid not in external:
            reject(RK_E_UNSUPPORTED,
                   "a staged GEMM weight must be an external constant",
                   stage="frontend", source=span,
                   actual=_buffer_name(params[pid]))
        for info in stage_infos:
            if pid not in info["reads"]:
                continue
            if info["opcode"] != "matmul" or info.get("b_pid") != pid:
                reject(RK_E_UNSUPPORTED,
                       "a binding cannot be both a packed weight and a feature "
                       "tensor", stage="frontend", source=span,
                       actual=_buffer_name(params[pid]))
    for info in stage_infos:
        stage_idx = info["stage"]
        for pid in info["reads"]:
            if pid in write_stage and write_stage[pid] >= stage_idx:
                reject(RK_E_UNSUPPORTED,
                       "a stage reads a parameter written by the same or a "
                       "later launch; cross-block ordering is unproved",
                       stage="frontend", source=span,
                       actual={"reader_stage": stage_idx,
                               "writer_stage": write_stage[pid],
                               "parameter": _buffer_name(params[pid])})
        if stage_idx in reads[info["output"]]:
            reject(RK_E_UNSUPPORTED,
                   "a stage reads the parameter it writes; a parallel grid has "
                   "no implicit intra-launch global sync", stage="frontend",
                   source=span, actual=_buffer_name(params[info["output"]]))

    # Outputs are produced parameters with no reader in any stage.
    output_pids = [pid for pid in range(len(params))
                   if pid in produced and not reads[pid]]
    if not output_pids:
        reject(RK_E_UNSUPPORTED,
               "staged dataflow has no exported parameter", stage="frontend",
               source=span)

    # Build the op list in launch order.
    ops: list[Any] = []
    op_for_stage: dict[int, int] = {}
    gemm_count = 0
    for info in stage_infos:
        op_id = len(ops)
        op_for_stage[info["stage"]] = op_id
        if info["opcode"] == "matmul":
            gemm_count += 1
            contract = _contract_with_points(
                options, (f"gemm{gemm_count}_accumulator",))
            attrs = {
                "b_layout": "kn",
                "numeric_contract": contract.to_dict(),
                "source_accumulator_dtype": info["source_acc"],
                "source_mn_tiling": {
                    "block_m": info["gm"], "block_n": info["gn"],
                    "grid_x": info["mapping"]["col_grid"],
                    "grid_y": info["mapping"]["row_grid"],
                    "coverage": "exact_once",
                },
                "source_stage_index": info["stage"],
                "source_stage_count": len(groups),
                "source_stage_boundary": "kernel_launch",
                "source_narrowing_points": list(F16_NARROWING_NAMES),
            }
            if info["k_hints"] is not None:
                attrs["source_k_loop"] = info["k_hints"]
            ops.append(_op(op_id, "matmul", info["inputs"],
                          (info["output"],), attrs, span))
        else:
            points = {
                "sigmoid": ("sigmoid",),
                "mul": ("silu_mul",),
                "add": ("residual_add",),
                "relu": ("relu",),
            }[info["opcode"]]
            contract = _contract_with_points(options, points)
            ops.append(_op(op_id, info["opcode"], info["inputs"],
                          (info["output"],),
                          {"numeric_contract": contract.to_dict()}, span))

    values: list[Any] = []
    for pid in range(len(params)):
        shape = pshape(pid)
        if pid in produced:
            role = "output" if pid in output_pids else "temporary"
            producer = op_for_stage[write_stage[pid]]
        else:
            role = "constant" if (pid in weight_pids or pid in constant_pids) \
                else "input"
            producer = None
        axes = () if len(shape) == 0 else (("col",) if len(shape) == 1
                                           else ("row", "col"))
        values.append(_value(pid, _buffer_name(params[pid]),
                            TensorType("float16", shape, axes=axes),
                            producer, role))
    inputs = tuple(pid for pid in range(len(params))
                   if pid in external and pid not in weight_pids
                   and pid not in constant_pids)
    outputs = tuple(output_pids)
    return _finish(name=f"tilelang:{_symbol_name(prim_func)}", values=tuple(values),
                  ops=tuple(ops), inputs=inputs, outputs=outputs,
                  profile_hash=profile_hash, source_hash=span,
                  state_signature="")


def _extract_static_dataflow(items: list[_Item], params: list[Any], *, tirx: Any,
                             prim_func: Any, span: str,
                             options: CompileOptions,
                             profile_hash: str) -> Any:
    """Extract the closed MatMul -> SiLU -> MatMul + residual SSA graph.

    This is the no-K-loop, one-primitive-tile subset.  It deliberately refuses
    branch graphs (the SwiGLU fixture) rather than returning a partial Module.
    """
    calls = [item for item in items if item.kind == "call"]
    gemm_items = [item for item in calls
                  if _op_name(item.node) == "tl.tileop.gemm"]
    fill_items = [item for item in calls
                  if _op_name(item.node) == "tl.tileop.fill"]
    copy_items = [item for item in calls
                  if _op_name(item.node) == "tl.tileop.copy"]
    store_items = [item for item in items if item.kind == "store"]
    if len(gemm_items) != 2:
        reject(RK_E_UNSUPPORTED,
               "multi-GEMM static dataflow currently supports exactly one "
               "MatMul -> SiLU -> projection chain (2 GEMMs); branch graphs are "
               "not lowered", stage="frontend", source=span,
               actual=len(gemm_items))

    # ---- 1. Per-GEMM header / region / fill / copy validation ------------
    groups: list[_GemmGroup] = []
    used_fills: set[int] = set()
    used_copies: dict[int, Any] = {}
    for gemm_item in gemm_items:
        gemm = gemm_item.node
        a_reg, b_reg, c_reg, gm, gn, gk = _gemm_header(gemm, tirx, span)
        if a_reg.physical != (gm, gk) or b_reg.physical != (gk, gn) or \
                c_reg.physical != (gm, gn):
            reject(RK_E_SHAPE,
                   "GEMM regions do not describe a full MxK * KxN tile",
                   stage="frontend", source=span)
        for reg, label in ((a_reg, "A"), (b_reg, "B"),
                           (c_reg, "accumulator")):
            if _param_of(reg.buffer, params) is not None:
                reject(RK_E_SHAPE,
                       f"GEMM operand {label} must be a tile buffer, not a "
                       "parameter", stage="frontend", source=span)
            if reg.offset != (0, 0):
                reject(RK_E_SHAPE,
                       f"GEMM operand {label} must start at tile offset zero",
                       stage="frontend", source=span, actual=reg.offset)
            _check_plain_tile_buffer(reg, f"GEMM operand {label}", span)
        if not (1 <= gm <= 256 and 1 <= gn <= 256 and 1 <= gk <= 256):
            reject(RK_E_UNSUPPORTED,
                   "static dataflow is limited to one primitive tile per node "
                   "(M/N/K <= 256)", stage="frontend", source=span,
                   actual=[gm, gn, gk])

        # Fills are attached by the exact accumulator buffer/region they write.
        matching_fills = []
        for fill_item in fill_items:
            reg = _region(fill_item.node.args[0], tirx, "fill", expect_mask=2)
            if _same_buffer(reg.buffer, c_reg.buffer):
                matching_fills.append((fill_item, reg))
        if len(matching_fills) != 1:
            reject(RK_E_NUMERIC_CONTRACT,
                   "each GEMM accumulator needs exactly one explicit zero fill",
                   stage="frontend", source=span,
                   actual={"accumulator": _buffer_name(c_reg.buffer),
                           "fills": len(matching_fills)})
        fill_item, fill_reg = matching_fills[0]
        if fill_reg.physical != (gm, gn) or fill_reg.offset != (0, 0):
            reject(RK_E_NUMERIC_CONTRACT,
                   "accumulator fill does not cover the whole GEMM tile",
                   stage="frontend", source=span)
        if _int_imm(fill_item.node.args[1]) != 0:
            reject(RK_E_NUMERIC_CONTRACT,
                   "GEMM accumulator must be zero filled", stage="frontend",
                   source=span, actual=repr(fill_item.node.args[1]))
        if fill_item.order >= gemm_item.order:
            reject(RK_E_NUMERIC_CONTRACT,
                   "accumulator fill must precede its GEMM", stage="frontend",
                   source=span)
        used_fills.add(fill_item.order)

        # Copies define the A/B operand tiles.  A may be a parameter or a tile
        # produced by an earlier node; B is a static weight parameter.
        a_candidates = []
        b_candidates = []
        for copy_item in copy_items:
            dst = _region(copy_item.node.args[1], tirx, "copy destination",
                          expect_mask=2)
            if _same_buffer(dst.buffer, a_reg.buffer):
                a_candidates.append(copy_item)
            if _same_buffer(dst.buffer, b_reg.buffer):
                b_candidates.append(copy_item)
        if len(a_candidates) != 1 or len(b_candidates) != 1:
            reject(RK_E_UNSUPPORTED,
                   "each GEMM operand tile must have exactly one defining copy",
                   stage="frontend", source=span,
                   actual={"A": len(a_candidates), "B": len(b_candidates)})
        a_copy, b_copy = a_candidates[0], b_candidates[0]
        if a_copy is b_copy:
            reject(RK_E_UNSUPPORTED,
                   "GEMM A and B cannot be filled by the same copy",
                   stage="frontend", source=span)
        a_src = _validate_operand_copy(a_copy, tirx, a_reg.buffer, (gm, gk),
                                       "A", span)
        b_src = _validate_operand_copy(b_copy, tirx, b_reg.buffer, (gk, gn),
                                       "B", span)
        if a_copy.order >= gemm_item.order or b_copy.order >= gemm_item.order:
            reject(RK_E_UNSUPPORTED,
                   "A/B copies must precede the GEMM that consumes them",
                   stage="frontend", source=span)
        if _param_of(b_src.buffer, params) is None:
            reject(RK_E_UNSUPPORTED,
                   "GEMM B must be copied from a static parameter weight",
                   stage="frontend", source=span)
        used_copies[a_copy.order] = ("A", gemm_item.order)
        used_copies[b_copy.order] = ("B", gemm_item.order)
        groups.append(_GemmGroup(
            item=gemm_item, a_reg=a_reg, b_reg=b_reg, c_reg=c_reg,
            gm=gm, gn=gn, gk=gk, fill=fill_item, a_copy=a_copy,
            b_copy=b_copy))

    if len(used_fills) != len(fill_items):
        extra = [order for order in (i.order for i in fill_items)
                 if order not in used_fills]
        reject(RK_E_NUMERIC_CONTRACT,
               "fill does not target a GEMM accumulator", stage="frontend",
               source=span, actual=extra)
    c_buffers = [g.c_reg.buffer for g in groups]
    if any(any(_same_buffer(c_buffers[i], c_buffers[j])
               for j in range(i)) for i in range(len(c_buffers))):
        reject(RK_E_UNSUPPORTED,
               "each GEMM needs a distinct accumulator tile", stage="frontend",
               source=span)

    # A branch graph can have two GEMMs and still be statically shaped.  The
    # second GEMM must consume an earlier device-produced tile through its A
    # copy; if it reads a parameter instead, this is a two-branch or two-
    # independent-GEMM program, not the supported projection chain.
    if len(groups) == 2:
        a2_src = _region(groups[1].a_copy.node.args[0], tirx, "copy source",
                         expect_mask=1)
        if _param_of(a2_src.buffer, params) is not None:
            reject(RK_E_UNSUPPORTED,
                   "multi-GEMM static dataflow currently supports one "
                   "MatMul -> SiLU -> projection chain (2 GEMMs); branch "
                   "graphs sharing a parameter input are not lowered",
                   stage="frontend", source=span)

    # ---- 2. Remaining copy must be the single parameter output copy ------
    output_copies: list[_Item] = []
    for copy_item in copy_items:
        if copy_item.order in used_copies:
            continue
        src = _region(copy_item.node.args[0], tirx, "copy source", expect_mask=1)
        dst = _region(copy_item.node.args[1], tirx, "copy destination",
                      expect_mask=2)
        if _param_of(dst.buffer, params) is not None:
            if src.offset != (0, 0) or dst.offset != (0, 0) or \
                    src.physical != dst.physical:
                reject(RK_E_SHAPE,
                       "output copy must move one whole tile to the parameter",
                       stage="frontend", source=span)
            output_copies.append(copy_item)
            continue
        reject(RK_E_UNSUPPORTED,
               "tile copy is not connected to a GEMM operand or the output",
               stage="frontend", source=span,
               actual=[_buffer_name(src.buffer), _buffer_name(dst.buffer)])
    if len(output_copies) != 1:
        reject(RK_E_UNSUPPORTED,
               "static dataflow requires exactly one accumulator-to-output copy",
               stage="frontend", source=span, actual=len(output_copies))
    output_copy = output_copies[0]
    output_src = _region(output_copy.node.args[0], tirx, "copy source",
                         expect_mask=1)
    output_dst = _region(output_copy.node.args[1], tirx, "copy destination",
                         expect_mask=2)
    y_param = _param_of(output_dst.buffer, params)
    if y_param is None:
        reject(RK_E_UNSUPPORTED,
               "output copy destination is not a parameter", stage="frontend",
               source=span)

    # ---- 3. Build SSA versions from explicit FP16 stores -----------------
    defs: list[_TileDef] = [
        _TileDef(g.c_reg.buffer, "matmul", g.item.order,
                 (g.gm, g.gn), _Expr("matmul"), gemm=g)
        for g in groups
    ]
    store_defs: list[_TileDef] = []
    for store_item in sorted(store_items, key=lambda i: i.order):
        target, shape, expr = _check_elementwise_store(store_item, tirx, params,
                                                       span)
        # Resolve every load against the most recent definition.  Parameters are
        # external values and need no device definition.
        for load_buffer in _expr_load_buffers(expr):
            if _param_of(load_buffer, params) is not None:
                continue
            if _latest_def(defs, load_buffer, store_item.order) is None:
                reject(RK_E_UNSUPPORTED,
                       "elementwise store reads a tile before it is defined",
                       stage="frontend", source=span,
                       actual=_buffer_name(load_buffer))
        kind = expr.kind
        if kind == "cast":
            kind = "narrow"
        elif kind in {"load", "sigmoid", "mul", "add"}:
            pass
        else:
            reject(RK_E_UNSUPPORTED,
                   f"unsupported store definition {kind!r}", stage="frontend",
                   source=span)
        definition = _TileDef(target, kind, store_item.order, shape, expr)
        defs.append(definition)
        store_defs.append(definition)

    if len(store_defs) != 5:
        reject(RK_E_NUMERIC_CONTRACT,
               "the projection closure requires five explicit FP16 narrowing "
               "stores (GEMM1, sigmoid, SiLU product, GEMM2, residual)",
               stage="frontend", source=span, actual=len(store_defs))

    # The store value of the accumulator cast is the SSA result of the GEMM.
    narrow_defs: list[_TileDef] = []
    for group in groups:
        matches = [
            d for d in store_defs
            if d.kind == "narrow" and d.expr.args and
            _same_expr_buffer(d.expr.args[0], group.c_reg.buffer) and
            d.order > group.item.order
        ]
        if len(matches) != 1:
            reject(RK_E_NUMERIC_CONTRACT,
                   "each GEMM accumulator must be consumed by exactly one "
                   "explicit FP16 store", stage="frontend", source=span,
                   actual={"gemm_order": group.item.order,
                           "narrowing_stores": len(matches)})
        narrow_defs.append(matches[0])
    z_def, p_def = narrow_defs
    if z_def.shape != (groups[0].gm, groups[0].gn) or \
            p_def.shape != (groups[1].gm, groups[1].gn):
        reject(RK_E_SHAPE,
               "FP16 narrowing store shape does not match its GEMM result",
               stage="frontend", source=span)

    # ---- 4. Match SiLU from def-use, not from variable names -------------
    sig_matches = [
        d for d in store_defs
        if d.kind == "sigmoid" and d.order > z_def.order and
        d.expr.args and _same_expr_buffer(d.expr.args[0], z_def.buffer)
    ]
    if len(sig_matches) != 1:
        reject(RK_E_UNSUPPORTED,
               "SiLU gate must be one sigmoid of the first GEMM's FP16 result",
               stage="frontend", source=span, actual=len(sig_matches))
    sig_def = sig_matches[0]
    if sig_def.shape != z_def.shape:
        reject(RK_E_SHAPE, "sigmoid store shape differs from its input",
               stage="frontend", source=span)
    mul_matches = []
    for d in store_defs:
        if d.kind != "mul" or d.order <= max(z_def.order, sig_def.order):
            continue
        loads = [a for a in d.expr.args if a.kind == "load"]
        if len(loads) != 2:
            continue
        if any(_same_expr_buffer(a, z_def.buffer) for a in loads) and \
                any(_same_expr_buffer(a, sig_def.buffer) for a in loads):
            mul_matches.append(d)
    if len(mul_matches) != 1:
        reject(RK_E_UNSUPPORTED,
               "SiLU must be matched as one FP32/FP16 multiply of the first "
               "GEMM result and its sigmoid", stage="frontend", source=span,
               actual=len(mul_matches))
    s_def = mul_matches[0]
    if s_def.shape != z_def.shape:
        reject(RK_E_SHAPE, "SiLU store shape differs from the gate tile",
               stage="frontend", source=span)

    # Every load of the first result / gate after its definition must belong to
    # the matched SiLU pair; a side consumer would make this more than one chain.
    for d in store_defs:
        for load_buffer in _expr_load_buffers(d.expr):
            if _same_buffer(load_buffer, z_def.buffer) and d is not sig_def \
                    and d is not s_def and d.order > z_def.order:
                reject(RK_E_UNSUPPORTED,
                       "first GEMM result has an unsupported consumer",
                       stage="frontend", source=span,
                       actual=_buffer_name(load_buffer))
            if _same_buffer(load_buffer, sig_def.buffer) and d is not s_def:
                reject(RK_E_UNSUPPORTED,
                       "sigmoid result has an unsupported consumer",
                       stage="frontend", source=span)

    # The second GEMM must consume the SiLU result through its A copy.
    g1, g2 = groups
    a2 = _region(g2.a_copy.node.args[0], tirx, "copy source", expect_mask=1)
    a2_def = _latest_def(store_defs, a2.buffer, g2.item.order)
    if a2_def is None or a2_def is not s_def:
        reject(RK_E_UNSUPPORTED,
               "second GEMM A operand is not the matched SiLU result; a wrong "
               "branch would change the computation", stage="frontend",
               source=span,
               actual={"copy_source": _buffer_name(a2.buffer),
                       "silu_result": _buffer_name(s_def.buffer)})

    # The first GEMM must read a parameter feature tile.  Its result shape and
    # the second GEMM's K extent tie the graph together.
    a1 = _region(g1.a_copy.node.args[0], tirx, "copy source", expect_mask=1)
    x_param = _param_of(a1.buffer, params)
    b1 = _region(g1.b_copy.node.args[0], tirx, "copy source", expect_mask=1)
    w1_param = _param_of(b1.buffer, params)
    b2 = _region(g2.b_copy.node.args[0], tirx, "copy source", expect_mask=1)
    w2_param = _param_of(b2.buffer, params)
    if x_param is None or w1_param is None or w2_param is None:
        reject(RK_E_UNSUPPORTED,
               "GEMM A must read the feature parameter and GEMM B must read a "
               "static weight parameter", stage="frontend", source=span)
    if g1.gm != g2.gm or g1.gk != g2.gn or g1.gn != g2.gk:
        reject(RK_E_SHAPE,
               "two-GEMM projection dimensions are inconsistent",
               stage="frontend", source=span,
               actual={"gemm1": [g1.gm, g1.gn, g1.gk],
                       "gemm2": [g2.gm, g2.gn, g2.gk]})

    # ---- 5. Residual add and final copy ----------------------------------
    add_matches = []
    for d in store_defs:
        if d.kind != "add" or d.order <= p_def.order:
            continue
        loads = [a for a in d.expr.args if a.kind == "load"]
        if len(loads) != 2:
            continue
        if any(_same_expr_buffer(a, p_def.buffer) for a in loads) and \
                any(a.buffer is not None and
                    _param_of(a.buffer, params) is not None and
                    _same_buffer(a.buffer, x_param) for a in loads):
            add_matches.append(d)
    if len(add_matches) != 1:
        reject(RK_E_UNSUPPORTED,
               "projection must be followed by exactly one same-shape residual "
               "add of the input parameter", stage="frontend", source=span,
               actual=len(add_matches))
    y_def = add_matches[0]
    if y_def.shape != p_def.shape:
        reject(RK_E_SHAPE, "residual add shape differs from projection",
               stage="frontend", source=span)

    expected_defs = {id(z_def), id(sig_def), id(s_def), id(p_def), id(y_def)}
    if len(expected_defs) != 5 or \
            expected_defs != {id(d) for d in store_defs}:
        reject(RK_E_UNSUPPORTED,
               "static dataflow contains a store that is not part of the single "
               "MatMul -> SiLU -> projection + residual chain",
               stage="frontend", source=span,
               actual=[d.kind for d in store_defs])

    final_src = _region(output_copy.node.args[0], tirx, "copy source",
                        expect_mask=1)
    final_def = _latest_def(store_defs, final_src.buffer, output_copy.order)
    if final_def is None or final_def is not y_def:
        reject(RK_E_UNSUPPORTED,
               "output copy does not read the residual result", stage="frontend",
               source=span)
    if output_copy.order <= y_def.order:
        reject(RK_E_UNSUPPORTED,
               "output copy must follow the residual add", stage="frontend",
               source=span)

    # ---- 6. Parameter roles and dimension closure ------------------------
    param_set = [p for p in params]
    named = [x_param, w1_param, w2_param, y_param]
    if len({id(p) for p in named}) != 4 or len(param_set) != 4:
        reject(RK_E_UNSUPPORTED,
               "silu_proj expects exactly four parameters: input X, W1, W2 and "
               "output Y", stage="frontend", source=span,
               actual=[_buffer_name(p) for p in param_set])
    if any(not any(_same_buffer(p, q) for q in named) for p in param_set):
        reject(RK_E_UNSUPPORTED,
               "a parameter is not connected to the two-GEMM chain",
               stage="frontend", source=span,
               actual=[_buffer_name(p) for p in param_set])

    m, d = (g1.gm, g1.gk)
    h = g1.gn
    if tuple(int(x) for x in x_param.shape) != (m, d) or \
            tuple(int(x) for x in w1_param.shape) != (d, h) or \
            tuple(int(x) for x in w2_param.shape) != (h, d) or \
            tuple(int(x) for x in y_param.shape) != (m, d):
        reject(RK_E_SHAPE,
               "parameter shapes do not match X[M,D], W1[D,H], W2[H,D] and "
               "Y[M,D]", stage="frontend", source=span,
               actual={"X": list(x_param.shape), "W1": list(w1_param.shape),
                       "W2": list(w2_param.shape), "Y": list(y_param.shape)})

    # ---- 7. Emit SSA RKIR -------------------------------------------------
    mm1_points = ("gemm1_accumulator",)
    mm2_points = ("gemm2_accumulator",)
    silu_points = ("sigmoid", "silu_mul")
    add_points = ("residual_add",)
    c1 = _contract_with_points(options, mm1_points)
    c2 = _contract_with_points(options, mm2_points)
    c_act = _contract_with_points(options, silu_points)
    c_add = _contract_with_points(options, add_points)
    shared = {"source_narrowing_points": list(F16_NARROWING_NAMES)}
    mm1_attrs = {"b_layout": "kn", "numeric_contract": c1.to_dict(),
                 "source_accumulator_dtype": "float32",
                 "tile_m": g1.gm, "tile_n": g1.gn, "tile_k": g1.gk, **shared}
    mm2_attrs = {"b_layout": "kn", "numeric_contract": c2.to_dict(),
                 "source_accumulator_dtype": "float32",
                 "tile_m": g2.gm, "tile_n": g2.gn, "tile_k": g2.gk, **shared}
    values = (
        _value(0, _buffer_name(x_param), TensorType("float16", (m, d)),
              None, "input"),
        _value(1, _buffer_name(w1_param), TensorType("float16", (d, h)),
              None, "constant"),
        _value(2, _buffer_name(w2_param), TensorType("float16", (h, d)),
              None, "constant"),
        _value(3, "Z", TensorType("float16", (m, h)), 0, "temporary"),
        _value(4, "S", TensorType("float16", (m, h)), 1, "temporary"),
        _value(5, "P", TensorType("float16", (m, d)), 2, "temporary"),
        _value(6, _buffer_name(y_param), TensorType("float16", (m, d)),
              3, "output"),
    )
    ops = (
        _op(0, "matmul", (0, 1), (3,), mm1_attrs, span),
        _op(1, "silu", (3,), (4,),
           {"numeric_contract": c_act.to_dict()}, span),
        _op(2, "matmul", (4, 2), (5,), mm2_attrs, span),
        _op(3, "add", (5, 0), (6,),
           {"numeric_contract": c_add.to_dict()}, span),
    )
    return _finish(name=f"tilelang:{_symbol_name(prim_func)}", values=values,
                  ops=ops, inputs=(0,), outputs=(6,),
                  profile_hash=profile_hash, source_hash=span,
                  state_signature="")


# --- Three-GEMM SwiGLU branch extraction (no K loop) -----------------------

FFN_NARROWING_NAMES = (
    "gate_accumulator", "up_accumulator", "silu", "act_mul",
    "projection_accumulator", "residual_add",
)


def _expr_direct_load(expr: _Expr, buffer: Any) -> bool:
    return expr.kind == "load" and expr.buffer is not None and \
        _same_buffer(expr.buffer, buffer)


def _expr_is_silu(expr: _Expr, value_buffer: Any) -> bool:
    """Match ``x * sigmoid(x)`` (or the commutative operand order)."""
    if expr.kind != "mul" or len(expr.args) != 2:
        return False
    for left, right in ((expr.args[0], expr.args[1]),
                        (expr.args[1], expr.args[0])):
        if _expr_direct_load(left, value_buffer) and \
                right.kind == "sigmoid" and len(right.args) == 1 and \
                _expr_direct_load(right.args[0], value_buffer):
            return True
    return False


def _expr_has_direct_load(expr: _Expr, buffer: Any) -> bool:
    if _expr_direct_load(expr, buffer):
        return True
    return any(_expr_has_direct_load(arg, buffer) for arg in expr.args)


def _raw_load_buffers(node: Any, tirx: Any) -> list[Any]:
    if isinstance(node, tirx.BufferLoad):
        return [node.buffer]
    out: list[Any] = []
    name = type(node).__name__
    if name in {"Add", "Mul"}:
        out.extend(_raw_load_buffers(node.a, tirx))
        out.extend(_raw_load_buffers(node.b, tirx))
    elif name == "Cast":
        out.extend(_raw_load_buffers(node.value, tirx))
    elif isinstance(node, tirx.Call):
        for arg in node.args:
            out.extend(_raw_load_buffers(arg, tirx))
    return out


def _check_unfused_fp32_residual(store_items: list[_Item], groups: list[_GemmGroup],
                                 params: list[Any], tirx: Any, span: str) -> None:
    """Reject a residual add that is narrowed only by the output T.copy.

    The hardware path performs same-shape add in FP16 and cannot represent a
    fused FP32 projection+residual followed by one final rounding.  Silently
    inserting a projection narrowing before the add changes the numeric contract,
    so this is a hard RK_E_NUMERIC_CONTRACT until a separate capability exists.
    """
    for item in store_items:
        store = item.node
        try:
            dtype = str(store.buffer.dtype)
        except Exception:
            continue
        if dtype != "float32":
            continue
        target_group = next(
            (g for g in groups if _same_buffer(store.buffer, g.c_reg.buffer)), None)
        if target_group is None or type(store.value).__name__ != "Add":
            continue
        loads = _raw_load_buffers(store.value, tirx)
        if not any(_same_buffer(b, target_group.c_reg.buffer) for b in loads):
            continue
        if not any(_param_of(b, params) is not None for b in loads):
            continue
        reject(RK_E_NUMERIC_CONTRACT,
               "residual add is written in the GEMM accumulator's FP32 buffer and "
               "narrowed only by the final T.copy; this backend has no capability "
               "for fused FP32 projection+residual followed by one final rounding. "
               "Use an explicit FP16 residual store, or define a separate numeric "
               "contract and hardware evidence", stage="frontend", source=span,
               op="add", actual={"accumulator": _buffer_name(store.buffer),
                                  "store_dtype": dtype},
               alternative="materialise proj16 before the add")


def _extract_ffn_branch(items: list[_Item], params: list[Any], *, tirx: Any,
                        prim_func: Any, span: str, options: CompileOptions,
                        profile_hash: str) -> Any:
    """Extract the staged-FP16 SwiGLU branch graph for the generic lowerer.

    Required source dataflow::

        Zg = X @ Wg;  Zu = X @ Wu;  S = SiLU(Zg);  A = S*Zu
        H32 = A @ Wd; H16 = H32; Y = H16 + X

    All definitions are associated by buffer identity and statement-order def-use.
    The X -> shared copy may legally have several GEMM consumers.
    """
    calls = [item for item in items if item.kind == "call"]
    gemm_items = [i for i in calls if _op_name(i.node) == "tl.tileop.gemm"]
    fill_items = [i for i in calls if _op_name(i.node) == "tl.tileop.fill"]
    copy_items = [i for i in calls if _op_name(i.node) == "tl.tileop.copy"]
    store_items = [i for i in items if i.kind == "store"]
    if len(gemm_items) != 3:
        reject(RK_E_UNSUPPORTED,
               "SwiGLU branch extraction requires exactly three GEMMs",
               stage="frontend", source=span, actual=len(gemm_items))

    groups: list[_GemmGroup] = []
    used_fills: set[int] = set()
    used_copies: set[int] = set()
    for gemm_item in gemm_items:
        gemm = gemm_item.node
        a_reg, b_reg, c_reg, gm, gn, gk = _gemm_header(gemm, tirx, span)
        if a_reg.physical != (gm, gk) or b_reg.physical != (gk, gn) or \
                c_reg.physical != (gm, gn):
            reject(RK_E_SHAPE,
                   "GEMM regions do not describe a full MxK * KxN tile",
                   stage="frontend", source=span)
        for reg, label in ((a_reg, "A"), (b_reg, "B"),
                           (c_reg, "accumulator")):
            if _param_of(reg.buffer, params) is not None:
                reject(RK_E_SHAPE,
                       f"GEMM operand {label} must be a tile buffer",
                       stage="frontend", source=span)
            if reg.offset != (0, 0):
                reject(RK_E_SHAPE,
                       f"GEMM operand {label} must start at tile offset zero",
                       stage="frontend", source=span, actual=reg.offset)
            _check_plain_tile_buffer(reg, f"GEMM operand {label}", span)
        if not (1 <= gm <= 256 and 1 <= gn <= 256 and 1 <= gk <= 256):
            reject(RK_E_UNSUPPORTED,
                   "the FFN frontend is limited to one primitive tile per GEMM "
                   "(M/N/K <= 256)", stage="frontend", source=span,
                   actual=[gm, gn, gk])

        matching_fills = []
        for fill_item in fill_items:
            reg = _region(fill_item.node.args[0], tirx, "fill", expect_mask=2)
            if _same_buffer(reg.buffer, c_reg.buffer):
                matching_fills.append((fill_item, reg))
        if len(matching_fills) != 1:
            reject(RK_E_NUMERIC_CONTRACT,
                   "each FFN GEMM accumulator needs one explicit zero fill",
                   stage="frontend", source=span,
                   actual={"fills": len(matching_fills)})
        fill_item, fill_reg = matching_fills[0]
        if fill_reg.physical != (gm, gn) or fill_reg.offset != (0, 0) or \
                _int_imm(fill_item.node.args[1]) != 0:
            reject(RK_E_NUMERIC_CONTRACT,
                   "FFN accumulator fill must zero the whole GEMM tile",
                   stage="frontend", source=span)
        if fill_item.order >= gemm_item.order:
            reject(RK_E_NUMERIC_CONTRACT,
                   "FFN accumulator fill must precede its GEMM",
                   stage="frontend", source=span)
        used_fills.add(fill_item.order)

        def defining_copy(dest_buffer: Any, expected: tuple[int, int],
                          label: str) -> tuple[_Item, _Region]:
            candidates = []
            for copy_item in copy_items:
                dst = _region(copy_item.node.args[1], tirx, "copy destination",
                              expect_mask=2)
                if _same_buffer(dst.buffer, dest_buffer):
                    candidates.append(copy_item)
            if len(candidates) != 1:
                reject(RK_E_UNSUPPORTED,
                       f"FFN GEMM {label} needs exactly one defining copy",
                       stage="frontend", source=span,
                       actual={"copies": len(candidates)})
            copy_item = candidates[0]
            src = _validate_operand_copy(copy_item, tirx, dest_buffer,
                                         expected, label, span)
            if copy_item.order >= gemm_item.order:
                reject(RK_E_UNSUPPORTED,
                       f"FFN GEMM {label} copy must precede the GEMM",
                       stage="frontend", source=span)
            used_copies.add(copy_item.order)
            return copy_item, src

        a_copy, a_src = defining_copy(a_reg.buffer, (gm, gk), "A")
        b_copy, b_src = defining_copy(b_reg.buffer, (gk, gn), "B")
        if _param_of(b_src.buffer, params) is None:
            reject(RK_E_UNSUPPORTED,
                   "FFN GEMM B must be copied from a static parameter weight",
                   stage="frontend", source=span)
        groups.append(_GemmGroup(
            item=gemm_item, a_reg=a_reg, b_reg=b_reg, c_reg=c_reg,
            gm=gm, gn=gn, gk=gk, fill=fill_item, a_copy=a_copy,
            b_copy=b_copy))

    if len(used_fills) != len(fill_items):
        reject(RK_E_NUMERIC_CONTRACT,
               "FFN fill does not target a GEMM accumulator", stage="frontend",
               source=span)
    for i in range(len(groups)):
        for j in range(i):
            if _same_buffer(groups[i].c_reg.buffer, groups[j].c_reg.buffer):
                reject(RK_E_UNSUPPORTED,
                       "each FFN GEMM needs a distinct accumulator tile",
                       stage="frontend", source=span)

    # A fused FP32 residual must be rejected before generic store parsing, with
    # a diagnosis that names the missing numeric capability rather than silently
    # inserting a projection narrowing.
    _check_unfused_fp32_residual(store_items, groups, params, tirx, span)

    # Any remaining copy must be the single FP16 parameter output copy.
    output_copies: list[_Item] = []
    for copy_item in copy_items:
        if copy_item.order in used_copies:
            continue
        src = _region(copy_item.node.args[0], tirx, "copy source", expect_mask=1)
        dst = _region(copy_item.node.args[1], tirx, "copy destination",
                      expect_mask=2)
        if _param_of(dst.buffer, params) is not None and \
                src.physical == dst.physical and src.offset == (0, 0) and \
                dst.offset == (0, 0):
            output_copies.append(copy_item)
            continue
        reject(RK_E_UNSUPPORTED,
               "FFN tile copy is not connected to a GEMM operand or the output",
               stage="frontend", source=span)
    if len(output_copies) != 1:
        reject(RK_E_UNSUPPORTED,
               "FFN requires exactly one parameter output copy",
               stage="frontend", source=span, actual=len(output_copies))
    output_copy = output_copies[0]
    output_dst = _region(output_copy.node.args[1], tirx, "copy destination",
                         expect_mask=2)
    y_param = _param_of(output_dst.buffer, params)
    if y_param is None:
        reject(RK_E_UNSUPPORTED,
               "FFN output copy destination is not a parameter",
               stage="frontend", source=span)

    # Build versioned SSA definitions from the explicit FP16 stores.
    defs: list[_TileDef] = [
        _TileDef(g.c_reg.buffer, "matmul", g.item.order,
                 (g.gm, g.gn), _Expr("matmul"), gemm=g)
        for g in groups
    ]
    store_defs: list[_TileDef] = []
    for store_item in sorted(store_items, key=lambda i: i.order):
        try:
            if str(store_item.node.buffer.dtype) != "float16":
                reject(RK_E_NUMERIC_CONTRACT,
                       "FFN elementwise definition is not an explicit FP16 store",
                       stage="frontend", source=span,
                       actual=str(store_item.node.buffer.dtype))
        except AttributeError:
            pass
        target, shape, expr = _check_elementwise_store(store_item, tirx, params,
                                                       span)
        for load_buffer in _expr_load_buffers(expr):
            if _param_of(load_buffer, params) is not None:
                continue
            if _latest_def(defs, load_buffer, store_item.order) is None:
                reject(RK_E_UNSUPPORTED,
                       "FFN store reads a tile before it is defined",
                       stage="frontend", source=span,
                       actual=_buffer_name(load_buffer))
        kind = "narrow" if expr.kind == "cast" else expr.kind
        if kind not in {"narrow", "sigmoid", "mul", "add"}:
            reject(RK_E_UNSUPPORTED,
                   f"unsupported FFN store definition {kind!r}",
                   stage="frontend", source=span)
        definition = _TileDef(target, kind, store_item.order, shape, expr)
        defs.append(definition)
        store_defs.append(definition)

    if len(store_defs) != 6:
        reject(RK_E_NUMERIC_CONTRACT,
               "staged FP16 FFN requires six explicit narrowing stores "
               "(gate, up, SiLU, act_mul, projection, residual)",
               stage="frontend", source=span, actual=len(store_defs))

    # One direct FP16 cast per accumulator is the SSA result of that GEMM.
    narrows: list[_TileDef] = []
    for group in groups:
        matches = [d for d in store_defs
                   if d.kind == "narrow" and d.expr.args and
                   _same_expr_buffer(d.expr.args[0], group.c_reg.buffer) and
                   d.order > group.item.order]
        if len(matches) != 1:
            reject(RK_E_NUMERIC_CONTRACT,
                   "each FFN GEMM accumulator needs exactly one FP16 cast",
                   stage="frontend", source=span, actual=len(matches))
        narrows.append(matches[0])
    n0, n1, proj_n = narrows

    # Identify gate vs. up by which branch result is consumed by x*sigmoid(x).
    silu_matches: list[tuple[_TileDef, _TileDef]] = []
    for branch in (n0, n1):
        for d in store_defs:
            if d.kind == "mul" and d.order > branch.order and \
                    _expr_is_silu(d.expr, branch.buffer):
                silu_matches.append((branch, d))
    if len(silu_matches) != 1:
        reject(RK_E_UNSUPPORTED,
               "FFN branch needs exactly one x*sigmoid(x) SiLU store",
               stage="frontend", source=span, actual=len(silu_matches))
    gate_n, silu_def = silu_matches[0]
    up_n = n1 if gate_n is n0 else n0
    if gate_n.shape != up_n.shape or silu_def.shape != gate_n.shape:
        reject(RK_E_SHAPE, "FFN gate/up/SiLU shapes differ", stage="frontend",
               source=span)

    mul_matches = []
    for d in store_defs:
        if d.kind != "mul" or d is silu_def or d.order <= silu_def.order:
            continue
        loads = [a for a in d.expr.args if a.kind == "load"]
        if len(loads) == 2 and \
                any(_same_expr_buffer(a, silu_def.buffer) for a in loads) and \
                any(_same_expr_buffer(a, up_n.buffer) for a in loads):
            mul_matches.append(d)
    if len(mul_matches) != 1:
        reject(RK_E_UNSUPPORTED,
               "FFN SiLU result must be multiplied by the up branch exactly once",
               stage="frontend", source=span, actual=len(mul_matches))
    act_def = mul_matches[0]

    # Projection GEMM A copy must consume the gated activation value.
    g0, g1, g2 = groups
    a2 = _region(g2.a_copy.node.args[0], tirx, "copy source", expect_mask=1)
    if _latest_def(store_defs, a2.buffer, g2.item.order) is not act_def:
        reject(RK_E_UNSUPPORTED,
               "FFN projection does not consume the gated activation",
               stage="frontend", source=span)

    # Residual must add the same X parameter that feeds both branches, after an
    # explicit FP16 projection store.
    residual_matches = []
    for d in store_defs:
        if d.kind != "add" or d.order <= proj_n.order:
            continue
        loads = [a for a in d.expr.args if a.kind == "load"]
        if len(loads) != 2:
            continue
        if any(_same_expr_buffer(a, proj_n.buffer) for a in loads) and \
                any(a.buffer is not None and
                    _param_of(a.buffer, params) is not None for a in loads):
            residual_matches.append(d)
    if len(residual_matches) != 1:
        reject(RK_E_UNSUPPORTED,
               "FFN projection must be followed by one explicit same-shape "
               "residual add", stage="frontend", source=span,
               actual=len(residual_matches))
    y_def = residual_matches[0]

    expected_defs = {id(gate_n), id(up_n), id(silu_def), id(act_def),
                     id(proj_n), id(y_def)}
    if len(expected_defs) != 6 or \
            expected_defs != {id(d) for d in store_defs}:
        reject(RK_E_UNSUPPORTED,
               "FFN contains a store outside the single SwiGLU branch chain",
               stage="frontend", source=span,
               actual=[d.kind for d in store_defs])

    final_src = _region(output_copy.node.args[0], tirx, "copy source",
                        expect_mask=1)
    if _latest_def(store_defs, final_src.buffer, output_copy.order) is not y_def:
        reject(RK_E_UNSUPPORTED,
               "FFN output copy does not read the residual result",
               stage="frontend", source=span)

    # Parameter roles and dimension closure.
    a0_src = _region(g0.a_copy.node.args[0], tirx, "copy source", expect_mask=1)
    a1_src = _region(g1.a_copy.node.args[0], tirx, "copy source", expect_mask=1)
    x0 = _param_of(a0_src.buffer, params)
    x1 = _param_of(a1_src.buffer, params)
    if x0 is None or x1 is None or not _same_buffer(x0, x1):
        reject(RK_E_UNSUPPORTED,
               "FFN gate/up branches must copy the same X parameter",
               stage="frontend", source=span)
    x_param = x0
    x_loads = [a for a in y_def.expr.args if a.kind == "load" and
               a.buffer is not None and _param_of(a.buffer, params) is not None]
    if len(x_loads) != 1:
        reject(RK_E_UNSUPPORTED,
               "FFN residual add must consume exactly one external parameter",
               stage="frontend", source=span)
    residual_param = _param_of(x_loads[0].buffer, params)
    if residual_param is None:
        reject(RK_E_UNSUPPORTED,
               "FFN residual add must consume an external parameter",
               stage="frontend", source=span)
    residual_is_x = _same_buffer(residual_param, x_param)
    b0 = _region(g0.b_copy.node.args[0], tirx, "copy source", expect_mask=1)
    b1 = _region(g1.b_copy.node.args[0], tirx, "copy source", expect_mask=1)
    b2 = _region(g2.b_copy.node.args[0], tirx, "copy source", expect_mask=1)
    w0 = _param_of(b0.buffer, params)
    w1 = _param_of(b1.buffer, params)
    wd = _param_of(b2.buffer, params)
    if w0 is None or w1 is None or wd is None:
        reject(RK_E_UNSUPPORTED,
               "FFN GEMM B operands must be parameter weights",
               stage="frontend", source=span)
    wg, wu = (w0, w1) if gate_n is n0 else (w1, w0)
    named = ([x_param, wg, wu, wd, y_param] if residual_is_x else
             [x_param, wg, wu, wd, residual_param, y_param])
    if len({id(p) for p in named}) != len(named) or \
            len(params) != len(named) or \
            any(not any(_same_buffer(p, q) for q in named) for p in params):
        reject(RK_E_UNSUPPORTED,
               "SwiGLU FFN expects X, Wg, Wu, Wd, optional residual input H, "
               "and Y parameters",
               stage="frontend", source=span,
               actual=[_buffer_name(p) for p in params])

    m, d = (g0.gm, g0.gk)
    h = g0.gn
    if g1.gm != m or g1.gk != d or g1.gn != h or \
            g2.gm != m or g2.gk != h or g2.gn != d:
        reject(RK_E_SHAPE,
               "FFN GEMM dimensions are inconsistent", stage="frontend",
               source=span,
               actual={"branch": [g0.gm, g0.gn, g0.gk],
                       "projection": [g2.gm, g2.gn, g2.gk]})
    if tuple(int(x) for x in x_param.shape) != (m, d) or \
            tuple(int(x) for x in wg.shape) != (d, h) or \
            tuple(int(x) for x in wu.shape) != (d, h) or \
            tuple(int(x) for x in wd.shape) != (h, d) or \
            tuple(int(x) for x in y_param.shape) != (m, d):
        reject(RK_E_SHAPE,
               "FFN parameter shapes must be X[M,D], Wg/Wu[D,H], Wd[H,D], Y[M,D]",
               stage="frontend", source=span,
               actual={"X": list(x_param.shape), "Wg": list(wg.shape),
                       "Wu": list(wu.shape), "Wd": list(wd.shape),
                       "Y": list(y_param.shape)})

    gate_contract = _contract_with_points(options, ("gate_accumulator",))
    up_contract = _contract_with_points(options, ("up_accumulator",))
    proj_contract = _contract_with_points(options,
                                          ("projection_accumulator",))
    silu_contract = _contract_with_points(options, ("silu",))
    mul_contract = _contract_with_points(options, ("act_mul",))
    add_contract = _contract_with_points(options, ("residual_add",))
    shared = {"source_narrowing_points": list(FFN_NARROWING_NAMES)}

    def mm_attrs(contract: NumericContract, group: _GemmGroup) -> dict[str, Any]:
        return {"b_layout": "kn", "numeric_contract": contract.to_dict(),
                "source_accumulator_dtype": "float32",
                "tile_m": group.gm, "tile_n": group.gn,
                "tile_k": group.gk, **shared}

    values = (
        _value(0, _buffer_name(x_param), TensorType("float16", (m, d)),
              None, "input"),
        _value(1, _buffer_name(wg), TensorType("float16", (d, h)),
              None, "constant"),
        _value(2, _buffer_name(wu), TensorType("float16", (d, h)),
              None, "constant"),
        _value(3, _buffer_name(wd), TensorType("float16", (h, d)),
              None, "constant"),
        _value(4, "Zg", TensorType("float16", (m, h)), 0, "temporary"),
        _value(5, "Zu", TensorType("float16", (m, h)), 1, "temporary"),
        _value(6, "S", TensorType("float16", (m, h)), 2, "temporary"),
        _value(7, "A", TensorType("float16", (m, h)), 3, "temporary"),
        _value(8, "H", TensorType("float16", (m, d)), 4, "temporary"),
        _value(9, _buffer_name(y_param), TensorType("float16", (m, d)),
              5, "output"),
    )
    if not residual_is_x:
        values = values + (
            _value(10, _buffer_name(residual_param),
                  TensorType("float16", (m, d)), None, "input"),
        )
    gate_group = g0 if gate_n is n0 else g1
    up_group = g1 if gate_n is n0 else g0
    ops = (
        _op(0, "matmul", (0, 1), (4,), mm_attrs(gate_contract, gate_group), span),
        _op(1, "matmul", (0, 2), (5,), mm_attrs(up_contract, up_group), span),
        _op(2, "silu", (4,), (6,),
           {"numeric_contract": silu_contract.to_dict()}, span),
        _op(3, "mul", (6, 5), (7,),
           {"numeric_contract": mul_contract.to_dict()}, span),
        _op(4, "matmul", (7, 3), (8,),
           mm_attrs(proj_contract, g2), span),
        _op(5, "add", (8, 0 if residual_is_x else 10), (9,),
           {"numeric_contract": add_contract.to_dict()}, span),
    )
    inputs = (0,) if residual_is_x else (0, 10)
    return _finish(name=f"tilelang:{_symbol_name(prim_func)}", values=values,
                  ops=ops, inputs=inputs, outputs=(9,),
                  profile_hash=profile_hash, source_hash=span,
                  state_signature="")


# --- Two independent GEMM branches with multiple outputs -------------------

MULTI_OUTPUT_NARROWING_NAMES = ("gemm0_accumulator", "gemm1_accumulator")


def _extract_branch_outputs(items: list[_Item], params: list[Any], *, tirx: Any,
                            prim_func: Any, span: str,
                            options: CompileOptions,
                            profile_hash: str) -> Any:
    """Extract two independent GEMMs that export two FP16 outputs.

    The two branches may share one parameter->tile copy.  Each GEMM accumulator
    is consumed by exactly one explicit FP16 cast, and each such value is copied
    to one distinct output parameter.
    """
    calls = [item for item in items if item.kind == "call"]
    gemm_items = [i for i in calls if _op_name(i.node) == "tl.tileop.gemm"]
    fill_items = [i for i in calls if _op_name(i.node) == "tl.tileop.fill"]
    copy_items = [i for i in calls if _op_name(i.node) == "tl.tileop.copy"]
    store_items = [i for i in items if i.kind == "store"]
    if len(gemm_items) != 2:
        reject(RK_E_UNSUPPORTED,
               "multi-output branch extraction requires exactly two GEMMs",
               stage="frontend", source=span, actual=len(gemm_items))

    groups: list[_GemmGroup] = []
    used_fills: set[int] = set()
    used_copies: set[int] = set()
    for gemm_item in gemm_items:
        gemm = gemm_item.node
        a_reg, b_reg, c_reg, gm, gn, gk = _gemm_header(gemm, tirx, span)
        if a_reg.physical != (gm, gk) or b_reg.physical != (gk, gn) or \
                c_reg.physical != (gm, gn):
            reject(RK_E_SHAPE,
                   "GEMM regions do not describe a full MxK * KxN tile",
                   stage="frontend", source=span)
        for reg, label in ((a_reg, "A"), (b_reg, "B"),
                           (c_reg, "accumulator")):
            if _param_of(reg.buffer, params) is not None or \
                    reg.offset != (0, 0):
                reject(RK_E_SHAPE,
                       f"GEMM operand {label} must be a zero-offset tile buffer",
                       stage="frontend", source=span)
            _check_plain_tile_buffer(reg, f"GEMM operand {label}", span)
        if not (1 <= gm <= 256 and 1 <= gn <= 256 and 1 <= gk <= 256):
            reject(RK_E_UNSUPPORTED,
                   "multi-output branches are limited to one primitive tile per "
                   "GEMM", stage="frontend", source=span, actual=[gm, gn, gk])

        fills = []
        for fill_item in fill_items:
            reg = _region(fill_item.node.args[0], tirx, "fill", expect_mask=2)
            if _same_buffer(reg.buffer, c_reg.buffer):
                fills.append((fill_item, reg))
        if len(fills) != 1:
            reject(RK_E_NUMERIC_CONTRACT,
                   "each branch GEMM needs one explicit zero fill",
                   stage="frontend", source=span, actual=len(fills))
        fill_item, fill_reg = fills[0]
        if fill_reg.physical != (gm, gn) or fill_reg.offset != (0, 0) or \
                _int_imm(fill_item.node.args[1]) != 0 or \
                fill_item.order >= gemm_item.order:
            reject(RK_E_NUMERIC_CONTRACT,
                   "branch accumulator must be zero-filled before its GEMM",
                   stage="frontend", source=span)
        used_fills.add(fill_item.order)

        def operand_copy(dest_buffer: Any, expected: tuple[int, int],
                         label: str) -> tuple[_Item, _Region]:
            candidates = []
            for copy_item in copy_items:
                dst = _region(copy_item.node.args[1], tirx, "copy destination",
                              expect_mask=2)
                if _same_buffer(dst.buffer, dest_buffer):
                    candidates.append(copy_item)
            if len(candidates) != 1:
                reject(RK_E_UNSUPPORTED,
                       f"branch GEMM {label} needs exactly one defining copy",
                       stage="frontend", source=span,
                       actual={"copies": len(candidates)})
            copy_item = candidates[0]
            src = _validate_operand_copy(copy_item, tirx, dest_buffer,
                                         expected, label, span)
            if copy_item.order >= gemm_item.order:
                reject(RK_E_UNSUPPORTED,
                       f"branch GEMM {label} copy must precede the GEMM",
                       stage="frontend", source=span)
            used_copies.add(copy_item.order)
            return copy_item, src

        a_copy, a_src = operand_copy(a_reg.buffer, (gm, gk), "A")
        b_copy, b_src = operand_copy(b_reg.buffer, (gk, gn), "B")
        if _param_of(a_src.buffer, params) is None or \
                _param_of(b_src.buffer, params) is None:
            reject(RK_E_UNSUPPORTED,
                   "independent branch operands must come from parameters",
                   stage="frontend", source=span)
        groups.append(_GemmGroup(
            item=gemm_item, a_reg=a_reg, b_reg=b_reg, c_reg=c_reg,
            gm=gm, gn=gn, gk=gk, fill=fill_item, a_copy=a_copy,
            b_copy=b_copy))

    if len(used_fills) != len(fill_items):
        reject(RK_E_NUMERIC_CONTRACT,
               "branch fill does not target a GEMM accumulator",
               stage="frontend", source=span)
    for i in range(len(groups)):
        for j in range(i):
            if _same_buffer(groups[i].c_reg.buffer, groups[j].c_reg.buffer):
                reject(RK_E_UNSUPPORTED,
                       "branch GEMMs need distinct accumulator tiles",
                       stage="frontend", source=span)

    # Remaining copies are the two parameter outputs.
    output_copies: list[tuple[_Item, Any]] = []
    for copy_item in copy_items:
        if copy_item.order in used_copies:
            continue
        src = _region(copy_item.node.args[0], tirx, "copy source", expect_mask=1)
        dst = _region(copy_item.node.args[1], tirx, "copy destination",
                      expect_mask=2)
        y_param = _param_of(dst.buffer, params)
        if y_param is None or src.physical != dst.physical or \
                src.offset != (0, 0) or dst.offset != (0, 0):
            reject(RK_E_UNSUPPORTED,
                   "branch copy is not a full parameter output copy",
                   stage="frontend", source=span)
        output_copies.append((copy_item, y_param))
    if len(output_copies) != 2:
        reject(RK_E_UNSUPPORTED,
               "two independent branches require two output copies",
               stage="frontend", source=span, actual=len(output_copies))

    # Build SSA definitions from the two explicit FP16 accumulator casts.
    defs: list[_TileDef] = [
        _TileDef(g.c_reg.buffer, "matmul", g.item.order,
                 (g.gm, g.gn), _Expr("matmul"), gemm=g)
        for g in groups
    ]
    store_defs: list[_TileDef] = []
    for store_item in sorted(store_items, key=lambda i: i.order):
        target, shape, expr = _check_elementwise_store(store_item, tirx, params,
                                                       span)
        for load_buffer in _expr_load_buffers(expr):
            if _param_of(load_buffer, params) is not None:
                continue
            if _latest_def(defs, load_buffer, store_item.order) is None:
                reject(RK_E_UNSUPPORTED,
                       "branch store reads a tile before it is defined",
                       stage="frontend", source=span)
        if expr.kind != "cast":
            reject(RK_E_UNSUPPORTED,
                   "branch output must be an explicit accumulator->FP16 cast",
                   stage="frontend", source=span, actual=expr.kind)
        definition = _TileDef(target, "narrow", store_item.order, shape, expr)
        defs.append(definition)
        store_defs.append(definition)
    if len(store_defs) != 2:
        reject(RK_E_NUMERIC_CONTRACT,
               "multi-output branches require exactly two explicit FP16 stores",
               stage="frontend", source=span, actual=len(store_defs))

    narrows: list[_TileDef] = []
    for group in groups:
        matches = [d for d in store_defs
                   if d.kind == "narrow" and d.expr.args and
                   _same_expr_buffer(d.expr.args[0], group.c_reg.buffer) and
                   d.order > group.item.order]
        if len(matches) != 1:
            reject(RK_E_NUMERIC_CONTRACT,
                   "each branch accumulator needs exactly one FP16 cast",
                   stage="frontend", source=span, actual=len(matches))
        narrows.append(matches[0])
    if {id(d) for d in store_defs} != {id(d) for d in narrows}:
        reject(RK_E_UNSUPPORTED,
               "branch store is not the accumulator narrowing point",
               stage="frontend", source=span)

    # Map each narrowed value to exactly one output parameter.
    output_by_def: dict[int, Any] = {}
    for copy_item, y_param in output_copies:
        src = _region(copy_item.node.args[0], tirx, "copy source", expect_mask=1)
        definition = _latest_def(store_defs, src.buffer, copy_item.order)
        if definition is None or definition not in narrows:
            reject(RK_E_UNSUPPORTED,
                   "branch output copy does not read an accumulator cast",
                   stage="frontend", source=span)
        key = id(definition)
        if key in output_by_def:
            reject(RK_E_UNSUPPORTED,
                   "two outputs cannot copy the same branch value",
                   stage="frontend", source=span)
        output_by_def[key] = y_param

    # Parameter roles and shapes.
    a0 = _region(groups[0].a_copy.node.args[0], tirx, "copy source",
                 expect_mask=1)
    a1 = _region(groups[1].a_copy.node.args[0], tirx, "copy source",
                 expect_mask=1)
    x0 = _param_of(a0.buffer, params)
    x1 = _param_of(a1.buffer, params)
    if x0 is None or x1 is None or not _same_buffer(x0, x1):
        reject(RK_E_UNSUPPORTED,
               "independent branches must share one X parameter",
               stage="frontend", source=span)
    b0 = _region(groups[0].b_copy.node.args[0], tirx, "copy source",
                 expect_mask=1)
    b1 = _region(groups[1].b_copy.node.args[0], tirx, "copy source",
                 expect_mask=1)
    w0 = _param_of(b0.buffer, params)
    w1 = _param_of(b1.buffer, params)
    if w0 is None or w1 is None:
        reject(RK_E_UNSUPPORTED,
               "branch GEMM B operands must be parameters", stage="frontend",
               source=span)
    y0 = output_by_def.get(id(narrows[0]))
    y1 = output_by_def.get(id(narrows[1]))
    if y0 is None or y1 is None:
        reject(RK_E_UNSUPPORTED,
               "each narrowed branch must have one output copy",
               stage="frontend", source=span)
    named = [x0, w0, w1, y0, y1]
    if len({id(p) for p in named}) != 5 or len(params) != 5 or \
            any(not any(_same_buffer(p, q) for q in named) for p in params):
        reject(RK_E_UNSUPPORTED,
               "multi-output branch expects exactly X, two weights and two "
               "outputs", stage="frontend", source=span,
               actual=[_buffer_name(p) for p in params])

    m, k = groups[0].gm, groups[0].gk
    if groups[1].gm != m or groups[1].gk != k:
        reject(RK_E_SHAPE,
               "independent branch GEMMs must share M and K", stage="frontend",
               source=span)
    n0, n1 = groups[0].gn, groups[1].gn
    if tuple(int(x) for x in x0.shape) != (m, k) or \
            tuple(int(x) for x in w0.shape) != (k, n0) or \
            tuple(int(x) for x in w1.shape) != (k, n1) or \
            tuple(int(x) for x in y0.shape) != (m, n0) or \
            tuple(int(x) for x in y1.shape) != (m, n1):
        reject(RK_E_SHAPE,
               "multi-output parameter shapes do not match their GEMMs",
               stage="frontend", source=span)

    names = list(MULTI_OUTPUT_NARROWING_NAMES)
    c0 = _contract_with_points(options, ("gemm0_accumulator",))
    c1 = _contract_with_points(options, ("gemm1_accumulator",))
    shared = {"source_narrowing_points": names}
    attrs0 = {"b_layout": "kn", "numeric_contract": c0.to_dict(),
              "source_accumulator_dtype": "float32",
              "tile_m": m, "tile_n": n0, "tile_k": k, **shared}
    attrs1 = {"b_layout": "kn", "numeric_contract": c1.to_dict(),
              "source_accumulator_dtype": "float32",
              "tile_m": m, "tile_n": n1, "tile_k": k, **shared}
    values = (
        _value(0, _buffer_name(x0), TensorType("float16", (m, k)),
              None, "input"),
        _value(1, _buffer_name(w0), TensorType("float16", (k, n0)),
              None, "constant"),
        _value(2, _buffer_name(w1), TensorType("float16", (k, n1)),
              None, "constant"),
        _value(3, _buffer_name(y0), TensorType("float16", (m, n0)),
              0, "output"),
        _value(4, _buffer_name(y1), TensorType("float16", (m, n1)),
              1, "output"),
    )
    ops = (
        _op(0, "matmul", (0, 1), (3,), attrs0, span),
        _op(1, "matmul", (0, 2), (4,), attrs1, span),
    )
    return _finish(name=f"tilelang:{_symbol_name(prim_func)}", values=values,
                  ops=ops, inputs=(0,), outputs=(3, 4),
                  profile_hash=profile_hash, source_hash=span,
                  state_signature="")


# --- MatMul followed by explicit constant broadcast ------------------------

BROADCAST_TAIL_NARROWING_NAMES = ("gemm_accumulator", "broadcast_add")


def _extract_broadcast_tail(items: list[_Item], params: list[Any], *, tirx: Any,
                            prim_func: Any, span: str,
                            options: CompileOptions,
                            profile_hash: str) -> Any:
    """Extract one GEMM followed by Add/Mul with a rank-0/1 constant.

    The source must materialise the accumulator in FP16 and write the binary
    result to a second FP16 tile before copying it to the output.  Runtime tensor
    operands stay same-shape only; this helper exists solely for an explicit
    constant broadcast source.
    """
    calls = [item for item in items if item.kind == "call"]
    gemm_items = [i for i in calls if _op_name(i.node) == "tl.tileop.gemm"]
    fill_items = [i for i in calls if _op_name(i.node) == "tl.tileop.fill"]
    copy_items = [i for i in calls if _op_name(i.node) == "tl.tileop.copy"]
    store_items = [i for i in items if i.kind == "store"]
    if len(gemm_items) != 1:
        reject(RK_E_UNSUPPORTED,
               "broadcast-tail extraction requires exactly one GEMM",
               stage="frontend", source=span)

    gemm_item = gemm_items[0]
    a_reg, b_reg, c_reg, gm, gn, gk = _gemm_header(gemm_item.node, tirx, span)
    if a_reg.physical != (gm, gk) or b_reg.physical != (gk, gn) or \
            c_reg.physical != (gm, gn):
        reject(RK_E_SHAPE,
               "GEMM regions do not describe a full MxK * KxN tile",
               stage="frontend", source=span)
    for reg, label in ((a_reg, "A"), (b_reg, "B"), (c_reg, "accumulator")):
        if _param_of(reg.buffer, params) is not None or reg.offset != (0, 0):
            reject(RK_E_SHAPE,
                   f"GEMM operand {label} must be a zero-offset tile buffer",
                   stage="frontend", source=span)
        _check_plain_tile_buffer(reg, f"GEMM operand {label}", span)
    if not (1 <= gm <= 256 and 1 <= gn <= 256 and 1 <= gk <= 256):
        reject(RK_E_UNSUPPORTED,
               "broadcast tail is limited to one primitive tile (M/N/K <= 256)",
               stage="frontend", source=span, actual=[gm, gn, gk])

    fills = []
    for fill_item in fill_items:
        reg = _region(fill_item.node.args[0], tirx, "fill", expect_mask=2)
        if _same_buffer(reg.buffer, c_reg.buffer):
            fills.append((fill_item, reg))
    if len(fills) != 1 or len(fill_items) != 1:
        reject(RK_E_NUMERIC_CONTRACT,
               "broadcast tail requires exactly one accumulator zero fill",
               stage="frontend", source=span, actual=len(fill_items))
    fill_item, fill_reg = fills[0]
    if fill_reg.physical != (gm, gn) or fill_reg.offset != (0, 0) or \
            _int_imm(fill_item.node.args[1]) != 0 or \
            fill_item.order >= gemm_item.order:
        reject(RK_E_NUMERIC_CONTRACT,
               "accumulator fill must zero the GEMM tile before the GEMM",
               stage="frontend", source=span)

    def operand_copy(dest_buffer: Any, expected: tuple[int, int],
                     label: str) -> tuple[_Item, _Region]:
        candidates = []
        for copy_item in copy_items:
            dst = _region(copy_item.node.args[1], tirx, "copy destination",
                          expect_mask=2)
            if _same_buffer(dst.buffer, dest_buffer):
                candidates.append(copy_item)
        if len(candidates) != 1:
            reject(RK_E_UNSUPPORTED,
                   f"broadcast-tail GEMM {label} needs exactly one defining copy",
                   stage="frontend", source=span, actual=len(candidates))
        copy_item = candidates[0]
        src = _validate_operand_copy(copy_item, tirx, dest_buffer, expected,
                                     label, span)
        if copy_item.order >= gemm_item.order:
            reject(RK_E_UNSUPPORTED,
                   f"broadcast-tail {label} copy must precede the GEMM",
                   stage="frontend", source=span)
        return copy_item, src

    a_copy, a_src = operand_copy(a_reg.buffer, (gm, gk), "A")
    b_copy, b_src = operand_copy(b_reg.buffer, (gk, gn), "B")
    x_param = _param_of(a_src.buffer, params)
    w_param = _param_of(b_src.buffer, params)
    if x_param is None or w_param is None:
        reject(RK_E_UNSUPPORTED,
               "broadcast-tail GEMM operands must come from parameters",
               stage="frontend", source=span)

    used_copies = {a_copy.order, b_copy.order}
    output_copies: list[_Item] = []
    for copy_item in copy_items:
        if copy_item.order in used_copies:
            continue
        src = _region(copy_item.node.args[0], tirx, "copy source", expect_mask=1)
        dst = _region(copy_item.node.args[1], tirx, "copy destination",
                      expect_mask=2)
        if _param_of(dst.buffer, params) is not None and \
                src.physical == dst.physical and src.offset == (0, 0) and \
                dst.offset == (0, 0):
            output_copies.append(copy_item)
            continue
        reject(RK_E_UNSUPPORTED,
               "broadcast-tail copy is not connected to a GEMM operand or the "
               "output", stage="frontend", source=span)
    if len(output_copies) != 1:
        reject(RK_E_UNSUPPORTED,
               "broadcast tail requires exactly one output copy",
               stage="frontend", source=span, actual=len(output_copies))
    output_copy = output_copies[0]
    output_dst = _region(output_copy.node.args[1], tirx, "copy destination",
                         expect_mask=2)
    y_param = _param_of(output_dst.buffer, params)
    bias_candidates = [p for p in params
                       if p is not x_param and p is not w_param and
                       p is not y_param]
    if y_param is None or len(bias_candidates) != 1:
        reject(RK_E_UNSUPPORTED,
               "broadcast tail expects X, W, one broadcast constant and Y",
               stage="frontend", source=span,
               actual=[_buffer_name(p) for p in params])
    bias_param = bias_candidates[0]
    bias_shape = tuple(int(x) for x in bias_param.shape)
    if len(bias_shape) not in {0, 1}:
        reject(RK_E_SHAPE,
               "only a rank-0 scalar or rank-1 [N] constant may broadcast",
               stage="frontend", source=span, actual=list(bias_shape))
    if len(bias_shape) == 1 and bias_shape[0] != gn:
        reject(RK_E_SHAPE,
               "per-channel broadcast length must equal the GEMM N extent",
               stage="frontend", source=span, actual=list(bias_shape),
               required=gn)

    defs: list[_TileDef] = [
        _TileDef(c_reg.buffer, "matmul", gemm_item.order, (gm, gn),
                 _Expr("matmul"))
    ]
    store_defs: list[_TileDef] = []
    for store_item in sorted(store_items, key=lambda i: i.order):
        target, shape, expr = _check_elementwise_store(store_item, tirx, params,
                                                       span)
        for load_buffer in _expr_load_buffers(expr):
            if _param_of(load_buffer, params) is not None:
                continue
            if _latest_def(defs, load_buffer, store_item.order) is None:
                reject(RK_E_UNSUPPORTED,
                       "broadcast-tail store reads a tile before it is defined",
                       stage="frontend", source=span)
        definition = _TileDef(target,
                              "narrow" if expr.kind == "cast" else expr.kind,
                              store_item.order, shape, expr)
        defs.append(definition)
        store_defs.append(definition)
    if len(store_defs) != 2:
        reject(RK_E_NUMERIC_CONTRACT,
               "broadcast tail requires two explicit FP16 stores "
               "(accumulator cast, binary result)", stage="frontend",
               source=span, actual=len(store_defs))
    narrow_matches = [d for d in store_defs
                      if d.kind == "narrow" and d.expr.args and
                      _same_expr_buffer(d.expr.args[0], c_reg.buffer) and
                      d.order > gemm_item.order]
    if len(narrow_matches) != 1:
        reject(RK_E_NUMERIC_CONTRACT,
               "accumulator needs exactly one explicit FP16 cast",
               stage="frontend", source=span, actual=len(narrow_matches))
    z_def = narrow_matches[0]
    binary_matches = []
    for d in store_defs:
        if d is z_def or d.kind not in {"add", "mul"} or d.order <= z_def.order:
            continue
        loads = [a for a in d.expr.args if a.kind == "load"]
        if len(loads) != 2:
            continue
        if any(_same_expr_buffer(a, z_def.buffer) for a in loads) and \
                any(a.buffer is not None and _same_buffer(a.buffer, bias_param)
                    for a in loads):
            binary_matches.append(d)
    if len(binary_matches) != 1:
        reject(RK_E_UNSUPPORTED,
               "broadcast tail needs one Add/Mul of the FP16 GEMM result and the "
               "declared constant", stage="frontend", source=span,
               actual=len(binary_matches))
    binary_def = binary_matches[0]
    if {id(d) for d in store_defs} != {id(z_def), id(binary_def)}:
        reject(RK_E_UNSUPPORTED,
               "broadcast tail contains an extra store definition",
               stage="frontend", source=span)
    final_src = _region(output_copy.node.args[0], tirx, "copy source",
                        expect_mask=1)
    if _latest_def(store_defs, final_src.buffer, output_copy.order) is not binary_def:
        reject(RK_E_UNSUPPORTED,
               "output copy does not read the broadcast result",
               stage="frontend", source=span)
    if tuple(int(x) for x in x_param.shape) != (gm, gk) or \
            tuple(int(x) for x in w_param.shape) != (gk, gn) or \
            tuple(int(x) for x in y_param.shape) != (gm, gn):
        reject(RK_E_SHAPE,
               "broadcast-tail parameter shapes do not match the GEMM",
               stage="frontend", source=span)

    bias_ty = TensorType("float16", bias_shape,
                         axes=() if not bias_shape else ("col",))
    mm_contract = _contract_with_points(options, ("gemm_accumulator",))
    bin_contract = _contract_with_points(
        options, ("broadcast_" + binary_def.expr.kind,))
    points = ["gemm_accumulator", "broadcast_" + binary_def.expr.kind]
    mm_attrs = {"b_layout": "kn", "numeric_contract": mm_contract.to_dict(),
                "source_accumulator_dtype": "float32",
                "tile_m": gm, "tile_n": gn, "tile_k": gk,
                "source_narrowing_points": points}
    values = (
        _value(0, _buffer_name(x_param), TensorType("float16", (gm, gk)),
              None, "input"),
        _value(1, _buffer_name(w_param), TensorType("float16", (gk, gn)),
              None, "constant"),
        _value(2, "Z", TensorType("float16", (gm, gn)), 0, "temporary"),
        _value(3, _buffer_name(bias_param), bias_ty, None, "constant"),
        _value(4, _buffer_name(y_param), TensorType("float16", (gm, gn)),
              1, "output"),
    )
    ops = (
        _op(0, "matmul", (0, 1), (2,), mm_attrs, span),
        _op(1, binary_def.expr.kind, (2, 3), (4,),
           {"numeric_contract": bin_contract.to_dict()}, span),
    )
    return _finish(name=f"tilelang:{_symbol_name(prim_func)}", values=values,
                  ops=ops, inputs=(0,), outputs=(4,),
                  profile_hash=profile_hash, source_hash=span,
                  state_signature="")


# --- K-loop (blocked reduction) extraction ---------------------------------
# TileLang's canonical GEMM writes a serial/pipelined loop over the K axis with
# one A slice copy, one B slice copy and one accumulating ``tl.tileop.gemm``
# inside it.  The lowering maps that to a K-tiled device chain, so the extractor
# must prove:
#   * exactly one serial loop, zero-based, static trip count, no nested loops,
#   * the loop variable only selects the K slice of the two copies,
#   * the accumulator is zero-filled before the loop and read only after it,
#   * trips * block_k equals the logical K of the parameter tensors.

K_LOOP_ALLOWED_ANNOTATIONS = ("num_stages", "software_pipeline_order",
                              "software_pipeline_stage", "tl.disable_warp_group_reg_alloc")


@dataclass(frozen=True)
class _KBlock:
    loop_item: _Item
    loop_var: Any
    trips: int
    block_k: int
    rows: int
    cols: int
    a_param: Any
    b_param: Any
    c_param: Any
    a_copy: _Item
    b_copy: _Item
    gemm_item: _Item
    fill_item: _Item
    c_copy: _Item
    annotations: dict
    grid_x: int = 1
    grid_y: int = 1
    block_var_x: Any = None
    block_var_y: Any = None


def _require_constant(region: _Region, label: str, span: str) -> tuple[int, int]:
    if region.offset is None:
        reject(RK_E_UNSUPPORTED,
               f"{label} must use a constant index", stage="frontend",
               source=span)
    return region.offset


def _k_slice_index(region: _Region, axis: int, block_k: int, loop_var: Any,
                   label: str, span: str, tirx: Any,
                   roles: dict[str, Any] | None = None,
                   expected: dict[str, int] | None = None) -> None:
    """Prove the indices are exactly the supported tiling.

    ``axis`` is the K axis: it must advance by exactly ``block_k`` with the K loop
    variable.  The other axis may only carry the block offset given in
    ``expected`` (empty for a single-block launch).
    """
    roles = roles or {}
    expected = {(k): v for k, v in (expected or {}).items() if v}
    for dim in (0, 1):
        if dim == axis:
            affine = _affine_index(region.raw_indices[dim], tirx)
            if affine is None:
                reject(RK_E_UNSUPPORTED,
                       f"{label}: index on axis {dim} is not an affine loop "
                       "expression", stage="frontend", source=span,
                       actual=repr(region.raw_indices[dim]))
            var, stride, base = affine
            if var is None or not _same_var(var, loop_var) or stride != block_k:
                reject(RK_E_UNSUPPORTED,
                       f"{label}: axis {dim} must advance by the block size "
                       f"({block_k}) with the loop variable", stage="frontend",
                       source=span,
                       actual={"stride": stride, "base": base,
                               "uses_loop_var": var is not None})
            if base != 0:
                reject(RK_E_UNSUPPORTED,
                       f"{label}: axis {dim} has a non-zero loop base",
                       stage="frontend", source=span, actual=base)
        else:
            _require_affine(region, dim, tirx, roles, expected, label, span)


def _flatten_conjuncts(expr: Any, tirx: Any) -> list[Any]:
    if type(expr).__name__ == "And":
        return (_flatten_conjuncts(expr.a, tirx) +
                _flatten_conjuncts(expr.b, tirx))
    return [expr]


def _affine_coeffs(expr: Any, tirx: Any, roles: dict[str, Any],
                   label: str, span: str) -> tuple[dict[str, int], int]:
    parsed = _affine_expr(expr, tirx, roles)
    if parsed is None:
        reject(RK_E_UNSUPPORTED,
               f"{label} is not an affine expression of the block/tile loop "
               "variables", stage="frontend", source=span, actual=repr(expr))
    return parsed


def _guard_terms(guards: tuple[Any, ...], tirx: Any, roles: dict[str, Any],
                 label: str, span: str) -> list[tuple[dict[str, int], int]]:
    terms: list[tuple[dict[str, int], int]] = []
    for guard in guards:
        for cmp in _flatten_conjuncts(guard, tirx):
            if type(cmp).__name__ != "LT":
                reject(RK_E_UNSUPPORTED,
                       f"{label} guard must be a conjunction of strict "
                       "upper-bound comparisons", stage="frontend", source=span,
                       actual=repr(cmp))
            bound = _int_imm(cmp.b)
            if bound is None or bound <= 0:
                reject(RK_E_SHAPE,
                       f"{label} guard bound must be a positive static constant",
                       stage="frontend", source=span, actual=repr(cmp.b))
            coeffs, const = _affine_coeffs(cmp.a, tirx, roles, label, span)
            if const != 0:
                reject(RK_E_UNSUPPORTED,
                       f"{label} guard has a non-zero base; only block*tile + "
                       "local index < logical_extent is accepted", stage="frontend",
                       source=span, actual=repr(cmp.a))
            terms.append(({key: value for key, value in coeffs.items() if value},
                          bound))
    return terms


def _match_guard_terms(terms: list[tuple[dict[str, int], int]],
                       expected: list[tuple[dict[str, int], int]],
                       label: str, span: str) -> None:
    remaining = list(terms)
    for coeffs, bound in expected:
        match = next((i for i, item in enumerate(remaining)
                      if item[0] == coeffs and item[1] == bound), None)
        if match is None:
            reject(RK_E_SHAPE,
                   f"{label} guard does not match the logical tensor boundary",
                   stage="frontend", source=span,
                   actual=[{"coeffs": c, "bound": b} for c, b in terms],
                   required=[{"coeffs": c, "bound": b}
                             for c, b in expected])
        remaining.pop(match)
    if remaining:
        reject(RK_E_UNSUPPORTED,
               f"{label} guard has extra conditions beyond the rectangle tail",
               stage="frontend", source=span,
               actual=[{"coeffs": c, "bound": b} for c, b in remaining])


def _match_index(coeffs: dict[str, int], const: int,
                 expected: dict[str, int], label: str, span: str) -> None:
    if const != 0 or coeffs != {k: v for k, v in expected.items() if v}:
        reject(RK_E_SHAPE,
               f"{label} does not match the canonical guarded tail indexing",
               stage="frontend", source=span,
               actual={"coeffs": coeffs, "const": const},
               required=expected)


def _tile_parallel_axes(item: _Item, rows: int, cols: int,
                        label: str, span: str) -> tuple[Any, Any]:
    loops = item.loops
    if len(loops) < 2:
        reject(RK_E_UNSUPPORTED,
               f"{label} must have two trailing parallel tile loops",
               stage="frontend", source=span, actual=len(loops))
    i_loop, j_loop = loops[-2], loops[-1]
    try:
        kinds = [int(i_loop.kind), int(j_loop.kind)]
    except Exception:
        kinds = []
    if kinds != [1, 1]:
        reject(RK_E_UNSUPPORTED,
               f"{label} must use two parallel loops; guarded serial loops are "
               "not a tile copy", stage="frontend", source=span, actual=kinds)
    if [_int_imm(i_loop.extent), _int_imm(j_loop.extent)] != [rows, cols] or \
            [_int_imm(i_loop.min), _int_imm(j_loop.min)] != [0, 0]:
        reject(RK_E_SHAPE,
               f"{label} tile loops must be zero-based and cover the full tile",
               stage="frontend", source=span,
               actual={"extents": [repr(i_loop.extent), repr(j_loop.extent)],
                       "mins": [repr(i_loop.min), repr(j_loop.min)]},
               required={"extents": [rows, cols], "mins": [0, 0]})
    return i_loop.loop_var, j_loop.loop_var


def _extract_ragged_tail(items: list[_Item], params: list[Any], *, tirx: Any,
                         prim_func: Any, span: str,
                         options: CompileOptions,
                         profile_hash: str) -> Any:
    """Extract the canonical explicit guarded rectangular-tail K loop.

    Source shape (the only accepted form)::

        with T.Kernel(ceildiv(N, BN), ceildiv(M, BM)) as (bx, by):
            clear(acc)
            for ko in Pipelined(ceildiv(K, BK)):
                fill(A_tile, 0); fill(B_tile, 0)
                for i,j in Parallel(BM,BK):
                    if by*BM+i < M and ko*BK+j < K:
                        A_tile[i,j] = A[by*BM+i, ko*BK+j]
                for i,j in Parallel(BK,BN):
                    if ko*BK+i < K and bx*BN+j < N:
                        B_tile[i,j] = B[ko*BK+i, bx*BN+j]
                gemm(A_tile, B_tile, acc)
            for i,j in Parallel(BM,BN):
                if by*BM+i < M and bx*BN+j < N:
                    Y[by*BM+i, bx*BN+j] = acc[i,j]

    Every out-of-range read is covered by an in-loop zero fill; the output store
    is guarded.  The planner consumes the logical M/N/K and emits its own exact
    padded tail tiles, so the guards are not silently discarded.
    """
    calls = [item for item in items if item.kind == "call"]
    tile_calls = [item for item in calls
                  if _op_name(item.node) in ALLOWED_TILEOPS]
    stores = [item for item in items if item.kind == "store"]
    fills = [item for item in tile_calls
             if _op_name(item.node) == "tl.tileop.fill"]
    gemms = [item for item in tile_calls
             if _op_name(item.node) == "tl.tileop.gemm"]
    copies = [item for item in tile_calls
              if _op_name(item.node) == "tl.tileop.copy"]
    if len(gemms) != 1 or copies:
        reject(RK_E_UNSUPPORTED,
               "guarded tail extraction requires one GEMM and no tile copies",
               stage="frontend", source=span,
               actual={"gemms": len(gemms), "copies": len(copies)})
    gemm_item = gemms[0]
    if gemm_item.guards or len(gemm_item.loops) != 1 or gemm_item.blocks is None:
        reject(RK_E_UNSUPPORTED,
               "guarded tail GEMM must be unguarded inside exactly one K loop",
               stage="frontend", source=span)
    k_loop = gemm_item.loops[0]
    try:
        k_kind = int(k_loop.kind)
    except Exception:
        k_kind = -1
    if k_kind != 0:
        reject(RK_E_UNSUPPORTED,
               "guarded tail K loop must be serial/pipelined", stage="frontend",
               source=span, actual=k_kind)
    if _int_imm(k_loop.min) != 0 or \
            (getattr(k_loop, "step", None) is not None and
             _int_imm(k_loop.step) != 1):
        reject(RK_E_UNSUPPORTED,
               "guarded tail K loop must start at zero and step by one",
               stage="frontend", source=span)
    trips = _int_imm(k_loop.extent)
    if trips is None or trips <= 0:
        reject(RK_E_SHAPE,
               "guarded tail K loop needs a static positive trip count",
               stage="frontend", source=span, actual=repr(k_loop.extent))

    a_reg, b_reg, c_reg, gm, gn, gk = _gemm_header(gemm_item.node, tirx, span)
    if a_reg.physical != (gm, gk) or b_reg.physical != (gk, gn) or \
            c_reg.physical != (gm, gn):
        reject(RK_E_SHAPE,
               "guarded tail GEMM regions do not describe a full tile",
               stage="frontend", source=span)
    for reg, label in ((a_reg, "A"), (b_reg, "B"), (c_reg, "accumulator")):
        if _param_of(reg.buffer, params) is not None or reg.offset != (0, 0):
            reject(RK_E_SHAPE,
                   f"guarded tail GEMM operand {label} must be a zero-offset "
                   "tile", stage="frontend", source=span)
        _check_plain_tile_buffer(reg, f"guarded tail GEMM operand {label}", span)
    if not (1 <= gm <= 256 and 1 <= gn <= 256 and gk == 256):
        reject(RK_E_UNSUPPORTED,
               "guarded tail lowering requires BM/BN <= 256 and BK == 256",
               stage="frontend", source=span, actual=[gm, gn, gk])

    # Block grid is (x=N tiles, y=M tiles); each axis has exactly ceil coverage.
    block_loops = gemm_item.blocks
    if len(block_loops) != 2:
        reject(RK_E_UNSUPPORTED,
               "guarded tail extraction requires a 2-D block grid (including "
               "extent-1 axes)", stage="frontend", source=span,
               actual=len(block_loops))
    bx_loop, by_loop = block_loops
    grid_x, grid_y = _int_imm(bx_loop.extent), _int_imm(by_loop.extent)
    if grid_x is None or grid_y is None or grid_x < 1 or grid_y < 1:
        reject(RK_E_SHAPE,
               "guarded tail block-grid extents must be static positive integers",
               stage="frontend", source=span,
               actual=[repr(bx_loop.extent), repr(by_loop.extent)])
    for block in block_loops:
        if _int_imm(block.min) != 0 or \
                (getattr(block, "step", None) is not None and
                 _int_imm(block.step) != 1):
            reject(RK_E_UNSUPPORTED,
                   "guarded tail block grid must start at zero and step by one",
                   stage="frontend", source=span)

    # Locate the three parameters and prove their roles by the guarded accesses.
    a_param = b_param = c_param = None
    a_guarded = b_guarded = y_guarded = None
    for item in stores:
        if not item.guards:
            continue
        store = item.node
        target = _param_of(store.buffer, params)
        if target is not None:
            if y_guarded is not None:
                reject(RK_E_UNSUPPORTED,
                       "guarded tail has more than one output store",
                       stage="frontend", source=span)
            y_guarded = item
            c_param = target
            continue
        if _same_buffer(store.buffer, a_reg.buffer):
            a_guarded = item
        elif _same_buffer(store.buffer, b_reg.buffer):
            b_guarded = item
        else:
            reject(RK_E_UNSUPPORTED,
                   "guarded store targets an unrelated buffer", stage="frontend",
                   source=span, actual=_buffer_name(store.buffer))
    if a_guarded is None or b_guarded is None or y_guarded is None:
        reject(RK_E_UNSUPPORTED,
               "guarded tail requires guarded A, B and output stores",
               stage="frontend", source=span,
               actual={"A": a_guarded is not None, "B": b_guarded is not None,
                       "Y": y_guarded is not None})
    if len(stores) != 3:
        reject(RK_E_UNSUPPORTED,
               "guarded tail contains an extra store outside the A/B/output "
               "rectangle", stage="frontend", source=span,
               actual=len(stores))

    # No unguarded tile store may bypass the tail conditions.
    for item in stores:
        if item is a_guarded or item is b_guarded or item is y_guarded:
            continue
        if _param_of(item.node.buffer, params) is not None or \
                _same_buffer(item.node.buffer, a_reg.buffer) or \
                _same_buffer(item.node.buffer, b_reg.buffer) or \
                _same_buffer(item.node.buffer, c_reg.buffer):
            reject(RK_E_UNSUPPORTED,
                   "unguarded tail store would access data outside the logical "
                   "rectangle", stage="frontend", source=span,
                   actual=_buffer_name(item.node.buffer))

    # A/B switches: the same K loop variable must advance both.
    roles = {"bx": bx_loop.loop_var, "by": by_loop.loop_var,
             "ko": k_loop.loop_var}
    if a_guarded.loops[:1] != (k_loop,) or b_guarded.loops[:1] != (k_loop,):
        reject(RK_E_UNSUPPORTED,
               "guarded input loads must be inside the K loop", stage="frontend",
               source=span)
    a_i, a_j = _tile_parallel_axes(a_guarded, gm, gk, "A guarded load", span)
    b_i, b_j = _tile_parallel_axes(b_guarded, gk, gn, "B guarded load", span)
    for item, i_var, j_var, label in ((a_guarded, a_i, a_j, "A"),
                                      (b_guarded, b_i, b_j, "B")):
        indices = item.node.indices
        if len(indices) != 2 or not _same_var(indices[0], i_var) or \
                not _same_var(indices[1], j_var):
            reject(RK_E_UNSUPPORTED,
                   f"guarded {label} tile store indices are not its tile loops",
                   stage="frontend", source=span)
    roles.update({"ai": a_i, "aj": a_j, "bi": b_i, "bj": b_j})
    a_terms = _guard_terms(a_guarded.guards, tirx, roles, "A tail", span)
    b_terms = _guard_terms(b_guarded.guards, tirx, roles, "B tail", span)
    # The logical bounds are resolved from the parameter shapes below; the
    # affine source indices are checked after that discovery.

    a_store = a_guarded.node
    b_store = b_guarded.node
    if type(a_store.value).__name__ != "BufferLoad" or \
            type(b_store.value).__name__ != "BufferLoad":
        reject(RK_E_UNSUPPORTED,
               "guarded input store value must be a direct parameter load",
               stage="frontend", source=span)
    a_loaded = a_store.value
    b_loaded = b_store.value
    a_param = _param_of(a_loaded.buffer, params)
    b_param = _param_of(b_loaded.buffer, params)
    if a_param is None or b_param is None:
        reject(RK_E_UNSUPPORTED,
               "guarded input stores must load the A/B parameters",
               stage="frontend", source=span)
    a_shape = tuple(int(x) for x in a_param.shape)
    b_shape = tuple(int(x) for x in b_param.shape)
    c_shape = tuple(int(x) for x in c_param.shape) if c_param is not None else ()
    if len(a_shape) != 2 or len(b_shape) != 2 or len(c_shape) != 2:
        reject(RK_E_SHAPE,
               "guarded tail parameters must be rank 2", stage="frontend",
               source=span)
    logical_m, logical_k = a_shape
    if b_shape[0] != logical_k:
        reject(RK_E_SHAPE,
               "guarded tail A/B disagree on logical K", stage="frontend",
               source=span, actual={"A": list(a_shape), "B": list(b_shape)})
    logical_n = b_shape[1]
    if c_shape != (logical_m, logical_n):
        reject(RK_E_SHAPE,
               "guarded tail output shape does not match M/N", stage="frontend",
               source=span, actual=list(c_shape))
    if grid_x != (logical_n + gn - 1) // gn or \
            grid_y != (logical_m + gm - 1) // gm or \
            trips != (logical_k + gk - 1) // gk:
        reject(RK_E_SHAPE,
               "guarded tail grid/trip counts do not equal ceildiv coverage",
               stage="frontend", source=span,
               actual={"grid_x": grid_x, "grid_y": grid_y, "trips": trips,
                       "logical": [logical_m, logical_n, logical_k],
                       "tile": [gm, gn, gk]})
    _match_guard_terms(a_terms,
                       [({"by": gm, "ai": 1}, logical_m),
                        ({"ko": gk, "aj": 1}, logical_k)],
                       "A tail", span)
    _match_guard_terms(b_terms,
                       [({"ko": gk, "bi": 1}, logical_k),
                        ({"bx": gn, "bj": 1}, logical_n)],
                       "B tail", span)
    _match_index(*_affine_coeffs(a_loaded.indices[0], tirx, roles,
                                 "A load row", span),
                 {"by": gm, "ai": 1}, "A load row", span)
    _match_index(*_affine_coeffs(a_loaded.indices[1], tirx, roles,
                                 "A load column", span),
                 {"ko": gk, "aj": 1}, "A load column", span)
    _match_index(*_affine_coeffs(b_loaded.indices[0], tirx, roles,
                                 "B load row", span),
                 {"ko": gk, "bi": 1}, "B load row", span)
    _match_index(*_affine_coeffs(b_loaded.indices[1], tirx, roles,
                                 "B load column", span),
                 {"bx": gn, "bj": 1}, "B load column", span)

    # A/B tiles are explicitly redefined to zero on every K iteration before the
    # guarded loads.  This is the proof that no stale tail bytes survive.
    for reg, label, guarded, dims in (
            (a_reg, "A", a_guarded, (gm, gk)),
            (b_reg, "B", b_guarded, (gk, gn))):
        matches = [f for f in fills
                   if k_loop in f.loops and
                   _same_buffer(_region(f.node.args[0], tirx, "fill",
                                        expect_mask=2).buffer, reg.buffer)]
        if len(matches) != 1:
            reject(RK_E_NUMERIC_CONTRACT,
                   f"guarded tail {label} tile needs exactly one zero fill each "
                   "K iteration", stage="frontend", source=span,
                   actual=len(matches))
        fill = matches[0]
        fill_reg = _region(fill.node.args[0], tirx, "fill", expect_mask=2)
        if fill.guards or fill_reg.physical != dims or \
                fill_reg.offset != (0, 0) or \
                _int_imm(fill.node.args[1]) != 0:
            reject(RK_E_NUMERIC_CONTRACT,
                   f"guarded tail {label} zero fill does not cover the full tile",
                   stage="frontend", source=span)
        if fill.order >= guarded.order:
            reject(RK_E_NUMERIC_CONTRACT,
                   f"guarded tail {label} tile must be zeroed before its "
                   "conditional load", stage="frontend", source=span)

    # Accumulator is zeroed once, outside the K loop.
    acc_fills = [f for f in fills
                 if not f.loops and
                 _same_buffer(_region(f.node.args[0], tirx, "fill",
                                      expect_mask=2).buffer, c_reg.buffer)]
    if len(acc_fills) != 1 or len(fills) != 3:
        reject(RK_E_NUMERIC_CONTRACT,
               "guarded tail requires one accumulator fill outside the K loop "
               "and exactly two tile fills inside it", stage="frontend",
               source=span, actual=len(fills))
    acc_fill = acc_fills[0]
    acc_fill_reg = _region(acc_fill.node.args[0], tirx, "fill", expect_mask=2)
    loop_order = next(item.order for item in items
                      if item.kind == "for" and item.node is k_loop)
    if acc_fill.guards or acc_fill_reg.physical != (gm, gn) or \
            acc_fill_reg.offset != (0, 0) or \
            _int_imm(acc_fill.node.args[1]) != 0 or \
            acc_fill.order >= loop_order:
        reject(RK_E_NUMERIC_CONTRACT,
               "guarded tail accumulator must be zeroed once before the K loop",
               stage="frontend", source=span)

    # Output stays in the accumulator until after the reduction; only the
    # logical rectangle is written.
    y_store = y_guarded.node
    if k_loop in y_guarded.loops or y_guarded.blocks != gemm_item.blocks or \
            y_guarded.order <= gemm_item.order:
        reject(RK_E_UNSUPPORTED,
               "guarded tail output store must run after the K loop under the "
               "same block grid", stage="frontend", source=span)
    y_i, y_j = _tile_parallel_axes(y_guarded, gm, gn, "output guarded store",
                                   span)
    y_roles = {"bx": bx_loop.loop_var, "by": by_loop.loop_var,
               "yi": y_i, "yj": y_j}
    y_terms = _guard_terms(y_guarded.guards, tirx, y_roles, "output tail",
                           span)
    _match_guard_terms(y_terms,
                       [({"by": gm, "yi": 1}, logical_m),
                        ({"bx": gn, "yj": 1}, logical_n)],
                       "output tail", span)
    if type(y_store.value).__name__ != "Cast" or \
            type(y_store.value.value).__name__ != "BufferLoad":
        reject(RK_E_UNSUPPORTED,
               "guarded output store must cast the accumulator fragment",
               stage="frontend", source=span)
    y_load = y_store.value.value
    if not _same_buffer(y_load.buffer, c_reg.buffer):
        reject(RK_E_UNSUPPORTED,
               "guarded output store does not read the GEMM accumulator",
               stage="frontend", source=span)
    if len(y_load.indices) != 2 or not _same_var(y_load.indices[0], y_i) or \
            not _same_var(y_load.indices[1], y_j):
        reject(RK_E_UNSUPPORTED,
               "guarded output accumulator indices are not the tile loops",
               stage="frontend", source=span)
    _match_index(*_affine_coeffs(y_store.indices[0], tirx, y_roles,
                                 "output row", span),
                 {"by": gm, "yi": 1}, "output row", span)
    _match_index(*_affine_coeffs(y_store.indices[1], tirx, y_roles,
                                 "output column", span),
                 {"bx": gn, "yj": 1}, "output column", span)

    # All three parameters are now role-proven and no fourth parameter is allowed.
    if len(params) != 3 or a_param is b_param or a_param is c_param or \
            b_param is c_param:
        reject(RK_E_UNSUPPORTED,
               "guarded tail expects exactly A, B and the output parameter",
               stage="frontend", source=span,
               actual=[_buffer_name(p) for p in params])

    k_tail = logical_k - (trips - 1) * gk
    m_tail = logical_m - (grid_y - 1) * gm
    n_tail = logical_n - (grid_x - 1) * gn
    contract = _numeric_contract(options)
    tail_contract = {
        "form": "guarded_rect_tail_v1",
        "zero_fill": ["A_tile_each_k_iteration", "B_tile_each_k_iteration"],
        "guarded_reads": {"A": [logical_m, logical_k],
                          "B": [logical_k, logical_n]},
        "guarded_write": [logical_m, logical_n],
        "tiles": {"M": gm, "N": gn, "K": gk},
        "tails": {"M": m_tail, "N": n_tail, "K": k_tail},
        "grid": {"x": grid_x, "y": grid_y},
        "k_trips": trips,
    }
    hints = {"source_k_block": gk, "source_k_trips": trips,
             "logical_k": logical_k, "k_tail": k_tail,
             "source_loop_annotations": {},
             "lowering_k_tile": 256,
             "pipeline_hint_dropped": False,
             "narrowing_base": "every 256 K elements",
             "tail_contract": tail_contract}
    mn_tiling = {"block_m": gm, "block_n": gn,
                 "grid_x": grid_x, "grid_y": grid_y,
                 "coverage": "ragged_tail",
                 "m_tail": m_tail, "n_tail": n_tail}
    values = (
        _value(0, _buffer_name(a_param), TensorType("float16",
                                                   (logical_m, logical_k)),
              None, "input"),
        _value(1, _buffer_name(b_param), TensorType("float16",
                                                   (logical_k, logical_n)),
              None, "constant"),
        _value(2, _buffer_name(c_param), TensorType("float16",
                                                   (logical_m, logical_n)),
              0, "output"),
    )
    ops = (_op(0, "matmul", (0, 1), (2,),
              {"b_layout": "kn", "numeric_contract": contract.to_dict(),
               "source_accumulator_dtype": "float32",
               "source_k_loop": hints,
               "source_mn_tiling": mn_tiling,
               "source_tail_contract": tail_contract}, span),)
    return _finish(name=f"tilelang:{_symbol_name(prim_func)}", values=values,
                  ops=ops, inputs=(0,), outputs=(2,),
                  profile_hash=profile_hash, source_hash=span,
                  state_signature="")


def _detect_k_block(items: list[_Item], tirx: Any, span: str,
                    params: list[Any]) -> _KBlock | None:
    tile_items = [item for item in items
                  if item.kind in {"call", "store"}
                  and _op_name(item.node) in ALLOWED_TILEOPS]
    nests = {tuple(item.loops) for item in tile_items if item.loops}
    if not nests:
        return None
    if len(nests) != 1:
        reject(RK_E_UNSUPPORTED,
               "tile operations are spread across several loops; the MVP "
               "extractor accepts a single K loop", stage="frontend",
               source=span, actual=len(nests))
    nest = next(iter(nests))
    if len(nest) != 1:
        reject(RK_E_UNSUPPORTED,
               "nested loops around tile operations are not expanded by the MVP "
               "extractor", stage="frontend", source=span, actual=len(nest))
    loop = nest[0]
    try:
        kind = int(loop.kind)
    except Exception:
        kind = -1
    if kind != 0:  # ForKind.SERIAL
        reject(RK_E_UNSUPPORTED,
               "only a serial/pipelined K loop is lowered; parallel loops around "
               "tile operations are not", stage="frontend", source=span,
               actual=kind)
    trips = _int_imm(loop.extent)
    if trips is None or trips <= 0:
        reject(RK_E_UNSUPPORTED,
               "K loop must have a static positive trip count", stage="frontend",
               source=span, actual=repr(loop.extent))
    if _int_imm(loop.min) != 0:
        reject(RK_E_UNSUPPORTED,
               "K loop must start at zero", stage="frontend", source=span,
               actual=repr(loop.min))
    k_step = getattr(loop, "step", None)
    if k_step is not None and _int_imm(k_step) != 1:
        reject(RK_E_UNSUPPORTED,
               "K loop must step by one", stage="frontend", source=span,
               actual=repr(k_step))
    # Block grid: block loops are launch structure, but the index analysis has
    # to know their extents and variables to prove the output coverage.
    block_nests = {tuple(item.blocks) for item in tile_items}
    if len(block_nests) != 1:
        reject(RK_E_UNSUPPORTED,
               "tile operations are nested under different block grids",
               stage="frontend", source=span, actual=len(block_nests))
    block_loops = next(iter(block_nests))
    if len(block_loops) > 2:
        reject(RK_E_UNSUPPORTED,
               "at most a 2-D block grid is lowered", stage="frontend",
               source=span, actual=len(block_loops))
    grid_extents: list[int] = []
    for block in block_loops:
        extent = _int_imm(block.extent)
        if extent is None or extent < 1:
            reject(RK_E_UNSUPPORTED,
                   "block grid extents must be static and positive",
                   stage="frontend", source=span, actual=repr(block.extent))
        if _int_imm(block.min) != 0:
            reject(RK_E_UNSUPPORTED,
                   "block grid loops must start at zero", stage="frontend",
                   source=span, actual=repr(block.min))
        step = getattr(block, "step", None)
        if step is not None and _int_imm(step) != 1:
            reject(RK_E_UNSUPPORTED,
                   "block grid loops must step by one", stage="frontend",
                   source=span, actual=repr(step))
        grid_extents.append(extent)
    grid_x = grid_y = 1
    block_var_x = block_var_y = None

    annotations: dict[str, Any] = {}
    for key, value in dict(loop.annotations).items():
        as_int = _int_imm(value)
        if as_int is not None:
            annotations[str(key)] = as_int
        elif isinstance(value, str):
            annotations[str(key)] = value
        else:
            reject(RK_E_UNSUPPORTED,
                   "K loop annotation value is not a plain scalar",
                   stage="frontend", source=span,
                   actual={str(key): type(value).__name__})
    unknown = sorted(set(annotations) - set(K_LOOP_ALLOWED_ANNOTATIONS))
    if unknown:
        reject(RK_E_UNSUPPORTED,
               "K loop carries annotations the extractor cannot prove are "
               "scheduling-only", stage="frontend", source=span, actual=unknown)

    inside = [item for item in tile_items if item.loops == nest]
    outside = [item for item in tile_items if not item.loops]
    fills = [i for i in outside if _op_name(i.node) == "tl.tileop.fill"]
    copies_out = [i for i in outside if _op_name(i.node) == "tl.tileop.copy"]
    copies_in = [i for i in inside if _op_name(i.node) == "tl.tileop.copy"]
    gemms = [i for i in inside if _op_name(i.node) == "tl.tileop.gemm"]
    if len(gemms) != 1:
        reject(RK_E_UNSUPPORTED,
               f"K loop must contain exactly one GEMM, found {len(gemms)}",
               stage="frontend", source=span)
    for item in copies_in:
        dst = _region(item.node.args[1], tirx, "copy dst", expect_mask=2)
        if _param_of(dst.buffer, params) is not None:
            reject(RK_E_UNSUPPORTED,
                   "accumulator must be copied to the output after the K loop, "
                   "not inside it", stage="frontend", source=span)
    if len(copies_in) != 2:
        reject(RK_E_UNSUPPORTED,
               "K loop must contain exactly two slice copies (A and B), found "
               f"{len(copies_in)}", stage="frontend", source=span)
    if len(fills) != 1:
        reject(RK_E_NUMERIC_CONTRACT,
               "K-loop GEMM needs exactly one zero fill outside the loop",
               stage="frontend", source=span)
    if len(copies_out) != 1:
        reject(RK_E_UNSUPPORTED,
               "K-loop program must copy the accumulator to the output exactly "
               f"once outside the loop, found {len(copies_out)}",
               stage="frontend", source=span)
    if len(inside) != len(gemms) + len(copies_in):
        reject(RK_E_UNSUPPORTED,
               "K loop body may contain only the two slice copies and the GEMM",
               stage="frontend", source=span, actual=len(inside))
    # Everything the program does must be consumed by this closed pattern: an
    # extra GEMM/copy/fill outside the loop would otherwise be silently dropped.
    stray = [item for item in outside
             if item is not fills[0] and item is not copies_out[0]]
    if stray:
        shapes = [_op_name(item.node) for item in stray]
        reject(RK_E_UNSUPPORTED,
               "unexpected tile operations outside the K loop body",
               stage="frontend", source=span, actual=shapes,
               alternative="a K-loop program may contain exactly one zero fill "
                           "and one accumulator-to-output copy outside the loop")
    if len(tile_items) != len(inside) + len(outside):
        reject(RK_E_UNSUPPORTED,
               "tile operations are not partitioned between the K loop and its "
               "surroundings", stage="frontend", source=span)

    gemm = gemms[0].node
    a_reg, b_reg, c_reg, gm, gn, gk = _gemm_header(gemm, tirx, span)
    for reg, label, expected in ((a_reg, "A tile", (gm, gk)),
                                 (b_reg, "B tile", (gk, gn)),
                                 (c_reg, "accumulator", (gm, gn))):
        _require_constant(reg, f"GEMM operand {label}", span)
        _check_plain_tile_buffer(reg, f"GEMM operand {label}", span)
        if reg.offset != (0, 0):
            reject(RK_E_SHAPE, f"GEMM operand {label} starts at a non-zero offset",
                   stage="frontend", source=span)
        if reg.physical != expected:
            reject(RK_E_SHAPE,
                   f"GEMM operand {label} region does not match the GEMM extents",
                   stage="frontend", source=span)
    if gm > 256 or gn > 256:
        reject(RK_E_UNSUPPORTED,
               "K-loop extraction still requires a single M/N primitive tile "
               "(<=256 rows/cols)", stage="frontend", source=span,
               actual=[gm, gn])
    if gk != 256:
        # specific.md section 12.4: the MVP GEMM semantic K block is 256 (the
        # last slice may be shorter).  Any other block size implies a different
        # reduction order / narrowing pattern and must be rejected rather than
        # merged into 256 by the lowering.
        reject(RK_E_UNSUPPORTED,
               "MVP K blocking is fixed at 256; a source block of "
               f"{gk} would change the reduction/narrowing order",
               stage="frontend", source=span, actual=gk, required=256,
               alternative="rewrite the kernel with a 256-wide K block")

    # A copy: A[0, ko*BK] -> A_tile
    a_copy = next((i for i in copies_in
                   if _same_buffer(_region(i.node.args[1], tirx, "copy dst",
                                           expect_mask=2).buffer, a_reg.buffer)),
                  None)
    b_copy = next((i for i in copies_in
                   if _same_buffer(_region(i.node.args[1], tirx, "copy dst",
                                           expect_mask=2).buffer, b_reg.buffer)),
                  None)
    if a_copy is None or b_copy is None or a_copy is b_copy:
        reject(RK_E_UNSUPPORTED,
               "K loop must copy A and B into the two GEMM tile buffers",
               stage="frontend", source=span)
    # Execution order inside the loop: both slice copies must precede the GEMM,
    # otherwise the source program would read tiles it has not filled yet.
    if not (a_copy.order < gemms[0].order and b_copy.order < gemms[0].order):
        reject(RK_E_UNSUPPORTED,
               "the two slice copies must precede the GEMM inside the K loop",
               stage="frontend", source=span,
               actual={"a_copy_order": a_copy.order, "b_copy_order": b_copy.order,
                       "gemm_order": gemms[0].order})
    if copies_in[0].order > gemms[0].order or copies_in[1].order > gemms[0].order:
        reject(RK_E_UNSUPPORTED,
               "no tile operation may follow the GEMM inside the K loop body",
               stage="frontend", source=span)
    params_by_role: dict[str, Any] = {}
    c_copy = copies_out[0]
    c_src = _region(c_copy.node.args[0], tirx, "copy source", expect_mask=1)
    c_dst = _region(c_copy.node.args[1], tirx, "copy dst", expect_mask=2)
    # Assign the block variables to the output axes.  A 2-D grid is (x=N, y=M) by
    # TileLang convention; a 1-D grid may cover either axis, so infer it from how
    # the single block variable indexes the output tile.
    if len(block_loops) == 2:
        grid_x, block_var_x = grid_extents[0], block_loops[0].loop_var
        grid_y, block_var_y = grid_extents[1], block_loops[1].loop_var
    elif len(block_loops) == 1 and grid_extents[0] > 1:
        single = block_loops[0]
        role = {"b0": single.loop_var}
        axis0 = _affine_expr(c_dst.raw_indices[0], tirx, role)
        axis1 = _affine_expr(c_dst.raw_indices[1], tirx, role)
        uses0 = axis0 is not None and axis0[0].get("b0", 0) != 0
        uses1 = axis1 is not None and axis1[0].get("b0", 0) != 0
        if uses0 and not uses1 and axis0[0].get("b0") == gm:
            grid_y, block_var_y = grid_extents[0], single.loop_var
        elif uses1 and not uses0 and axis1[0].get("b0") == gn:
            grid_x, block_var_x = grid_extents[0], single.loop_var
        else:
            reject(RK_E_UNSUPPORTED,
                   "cannot prove which output axis the single block grid maps to",
                   stage="frontend", source=span,
                   actual={"output_indices": [repr(c_dst.raw_indices[0]),
                                              repr(c_dst.raw_indices[1])]})
    roles = {"ko": loop.loop_var}
    if block_var_x is not None:
        roles["bx"] = block_var_x
    if block_var_y is not None:
        roles["by"] = block_var_y
    # A[by*BM, ko*K] and B[ko*K, bx*BN]
    operand_indices = {
        "A": (1, {"by": gm} if grid_y > 1 else {}),
        "B": (0, {"bx": gn} if grid_x > 1 else {}),
    }
    for role, item, axis, extent in (("A", a_copy, 1, (gm, gk)),
                                     ("B", b_copy, 0, (gk, gn))):
        src = _region(item.node.args[0], tirx, "copy source", expect_mask=1)
        dst = _region(item.node.args[1], tirx, "copy dst", expect_mask=2)
        _require_constant(dst, f"{role} copy destination", span)
        if dst.offset != (0, 0) or dst.physical != extent:
            reject(RK_E_SHAPE,
                   f"{role} copy destination must cover the whole GEMM tile",
                   stage="frontend", source=span)
        if src.physical != extent:
            reject(RK_E_SHAPE,
                   f"{role} copy source extent must equal the GEMM tile",
                   stage="frontend", source=span)
        param = _param_of(src.buffer, params)
        if param is None:
            reject(RK_E_UNSUPPORTED,
                   f"{role} K-block source is not a parameter buffer; tile-to-"
                   "tile copies are not part of the K-loop pattern",
                   stage="frontend", source=span)
        _k_slice_index(src, axis, gk, loop.loop_var, f"{role} copy source",
                       span, tirx, roles, operand_indices[role][1])
        params_by_role[role] = param

    c_src = _region(c_copy.node.args[0], tirx, "copy source", expect_mask=1)
    c_dst = _region(c_copy.node.args[1], tirx, "copy dst", expect_mask=2)
    # Assign the block variables to the output axes.  A 2-D grid is (x=N, y=M) by
    # TileLang convention; a 1-D grid may cover either axis, so infer it from how
    # the single block variable indexes the output tile.
    if len(block_loops) == 2:
        grid_x, block_var_x = grid_extents[0], block_loops[0].loop_var
        grid_y, block_var_y = grid_extents[1], block_loops[1].loop_var
    elif len(block_loops) == 1 and grid_extents[0] > 1:
        single = block_loops[0]
        role = {"b0": single.loop_var}
        axis0 = _affine_expr(c_dst.raw_indices[0], tirx, role)
        axis1 = _affine_expr(c_dst.raw_indices[1], tirx, role)
        uses0 = axis0 is not None and axis0[0].get("b0", 0) != 0
        uses1 = axis1 is not None and axis1[0].get("b0", 0) != 0
        if uses0 and not uses1 and axis0[0].get("b0") == gm:
            grid_y, block_var_y = grid_extents[0], single.loop_var
        elif uses1 and not uses0 and axis1[0].get("b0") == gn:
            grid_x, block_var_x = grid_extents[0], single.loop_var
        else:
            reject(RK_E_UNSUPPORTED,
                   "cannot prove which output axis the single block grid maps to",
                   stage="frontend", source=span,
                   actual={"output_indices": [repr(c_dst.raw_indices[0]),
                                              repr(c_dst.raw_indices[1])]})
    _require_constant(c_src, "accumulator copy source", span)
    _require_affine(c_dst, 0, tirx, roles,
                    {"by": gm} if grid_y > 1 else {},
                    "output copy destination", span)
    _require_affine(c_dst, 1, tirx, roles,
                    {"bx": gn} if grid_x > 1 else {},
                    "output copy destination", span)
    if not _same_buffer(c_src.buffer, c_reg.buffer):
        reject(RK_E_UNSUPPORTED,
               "output copy must read the GEMM accumulator", stage="frontend",
               source=span)
    if c_src.physical != (gm, gn) or c_dst.physical != (gm, gn) or \
            c_src.offset != (0, 0):
        reject(RK_E_SHAPE,
               "output copy must move the whole accumulator tile", stage="frontend",
               source=span)
    c_param = _param_of(c_dst.buffer, params)
    if c_param is None:
        reject(RK_E_UNSUPPORTED,
               "output copy destination is not a parameter buffer",
               stage="frontend", source=span)

    fill_item = fills[0]
    fill_reg = _region(fill_item.node.args[0], tirx, "fill", expect_mask=2)
    _require_constant(fill_reg, "accumulator fill", span)
    if not _same_buffer(fill_reg.buffer, c_reg.buffer) or \
            fill_reg.physical != (gm, gn) or fill_reg.offset != (0, 0):
        reject(RK_E_NUMERIC_CONTRACT,
               "the fill must zero the whole accumulator outside the loop",
               stage="frontend", source=span)
    if _int_imm(fill_item.node.args[1]) != 0:
        reject(RK_E_NUMERIC_CONTRACT,
               "K-loop accumulator must be zero filled; a non-zero initial value "
               "needs a separate legalization rule", stage="frontend", source=span)

    loop_order = next(item.order for item in items
                      if item.kind == "for" and item.node is loop)
    if fill_item.order >= loop_order:
        reject(RK_E_NUMERIC_CONTRACT,
               "accumulator zero fill must precede the K loop", stage="frontend",
               source=span)
    if c_copy.order <= loop_order:
        reject(RK_E_UNSUPPORTED,
               "accumulator must be copied to the output after the K loop",
               stage="frontend", source=span)

    logical_k = trips * gk
    a_shape = tuple(int(x) for x in params_by_role["A"].shape)
    b_shape = tuple(int(x) for x in params_by_role["B"].shape)
    c_shape = tuple(int(x) for x in c_param.shape)
    # Single-coverage proof: the grid times the block tile must tile the whole
    # output exactly once.  Re-arranging the M/N tiling is then numerically
    # neutral because each output element still comes from exactly one dot
    # product (only the K order affects rounding, and that is fixed at 256).
    logical_m, logical_n = grid_y * gm, grid_x * gn
    if logical_m != c_shape[0] or logical_n != c_shape[1]:
        reject(RK_E_SHAPE,
               "block grid times block tile does not cover the output exactly "
               "once (ragged M/N tiles are not supported yet)",
               stage="frontend", source=span,
               actual={"grid": [grid_x, grid_y], "tile": [gm, gn],
                       "output": list(c_shape)})
    if a_shape != (logical_m, logical_k) or b_shape != (logical_k, logical_n):
        reject(RK_E_SHAPE,
               "K-loop tiling does not tile the parameter shapes exactly "
               "(no ragged K is supported yet)", stage="frontend", source=span,
               actual={"A": list(a_shape), "B": list(b_shape), "C": list(c_shape),
                       "trips": trips, "block_k": gk})
    return _KBlock(loop_item=next(i for i in items
                                  if i.kind == "for" and i.node is loop),
                   loop_var=loop.loop_var, trips=trips, block_k=gk, rows=gm,
                   cols=gn, a_param=params_by_role["A"],
                   b_param=params_by_role["B"], c_param=c_param,
                   a_copy=a_copy, b_copy=b_copy, gemm_item=gemms[0],
                   fill_item=fill_item, c_copy=c_copy, annotations=annotations,
                   grid_x=grid_x, grid_y=grid_y, block_var_x=block_var_x,
                   block_var_y=block_var_y)


def _k_loop_module(block: _KBlock, *, prim_func: Any, span: str,
                   options: CompileOptions, profile_hash: str) -> Any:
    logical_k = block.trips * block.block_k
    if not 1 <= logical_k <= 4096:
        reject(RK_E_SHAPE, "logical K must be in 1..4096", stage="frontend",
               source=span, actual=logical_k)
    contract = _numeric_contract(options)
    logical_m = block.grid_y * block.rows
    logical_n = block.grid_x * block.cols
    hints = {"source_k_block": block.block_k,
             "source_k_trips": block.trips,
             "source_loop_annotations": block.annotations,
             "source_loop_span": str(getattr(block.loop_item.node, "span", "") or ""),
             "lowering_k_tile": 256,
             "pipeline_hint_dropped": bool(block.annotations),
             "narrowing_base": "every 256 K elements"}
    mn_tiling = {"block_m": block.rows, "block_n": block.cols,
                 "grid_x": block.grid_x, "grid_y": block.grid_y,
                 "coverage": "exact_once"}
    values = [
        _value(0, "A", TensorType("float16", (logical_m, logical_k)), None,
              "input"),
        _value(1, "B", TensorType("float16", (logical_k, logical_n)), None,
              "constant"),
        _value(2, "Y", TensorType("float16", (logical_m, logical_n)), 0, "output"),
    ]
    ops = [_op(0, "matmul", (0, 1), (2,),
              {"b_layout": "kn", "numeric_contract": contract.to_dict(),
               "source_accumulator_dtype": "float32",
               "source_k_loop": hints,
               "source_mn_tiling": mn_tiling}, span)]
    return _finish(name=f"tilelang:{_symbol_name(prim_func)}", values=tuple(values),
                  ops=tuple(ops), inputs=(0,), outputs=(2,),
                  profile_hash=profile_hash, source_hash=span,
                  state_signature="")


def _numeric_contract(options: CompileOptions) -> NumericContract:
    if options.numeric_mode == "strict":
        return NumericContract(
            accumulator_dtype="float32", accumulation="strict_unspecified",
            narrowing_points=(), approximate=False,
            tolerance={"atol": 1e-3, "rtol": 1e-3})
    return NumericContract(
        accumulator_dtype="float16",
        accumulation="ordered_k_tiles_with_f16_writeback",
        narrowing_points=("k_tile_boundary",), approximate=False,
        tolerance={"atol": 8e-3, "rtol": 8e-3})


def extract(prim_func: Any, *, options: CompileOptions,
            profile_hash: str = "", b_is_input: bool = False) -> Any:
    tirx = _import_tirx()
    if not isinstance(prim_func, tirx.PrimFunc):
        reject(RK_E_UNSUPPORTED,
               "compile_tilelang expects a tvm.tirx.PrimFunc, got "
               f"{type(prim_func).__name__}", stage="frontend")
    span = _source_span(prim_func)
    params = _param_buffers(prim_func)

    items: list[_Item] = []
    _scan(prim_func.body, tirx, items, [0], (), span)

    calls = [item for item in items if item.kind == "call"]
    stores = [item for item in items if item.kind == "store"]
    _check_no_forbidden_calls([item.node for item in calls], span)
    # Complete call whitelist: the only calls the MVP lowers are the registered
    # tile ops.  Anything else (intrinsics, atomics, backend hooks) is rejected
    # here, before the K path can dispatch, so a side-effecting call can never be
    # silently dropped.
    for item in calls:
        name = _op_name(item.node)
        if name not in ALLOWED_TILEOPS:
            reject(RK_E_UNSUPPORTED, f"unhandled call {name!r} in the kernel body",
                   stage="frontend", source=span, op=name,
                   alternative="only tl.tileop.fill/copy/gemm are lowered")

    # Guarded parallel stores are only accepted by the explicit rectangular-tail
    # K-loop form.  They must never fall through to a path that ignores guards.
    if any(item.guards for item in items):
        return _extract_ragged_tail(
            items, params, tirx=tirx, prim_func=prim_func, span=span,
            options=options, profile_hash=profile_hash)

    # Several distinct T.Kernel block scopes are an explicit staged pipeline.
    # It is dispatched before the single-K-loop detector because each launch is
    # a separate global-synchronisation boundary.
    stage_items = [item for item in items
                   if item.kind == "store" or
                   (item.kind == "call" and
                    _op_name(item.node) in ALLOWED_TILEOPS)]
    stage_keys: list[int] = []
    for item in stage_items:
        key = id(item.blocks)
        if not stage_keys or key != stage_keys[-1]:
            stage_keys.append(key)
    if len(stage_keys) > 1:
        return _extract_staged_dataflow(
            items, params, tirx=tirx, prim_func=prim_func, span=span,
            options=options, profile_hash=profile_hash)

    # A K-blocked reduction takes the dedicated path; everything else must be
    # loop free.
    k_block = _detect_k_block(items, tirx, span, params)
    # Block-grid structure is validated for every pattern, not only the K-loop
    # path: a grid that starts at 1 or steps by 2 would change which output tiles
    # are produced even in the single-tile case.
    for item in items:
        if item.kind in {"call", "store"} and _op_name(item.node) in ALLOWED_TILEOPS:
            for block in item.blocks:
                if _int_imm(block.min) != 0:
                    reject(RK_E_UNSUPPORTED,
                           "block grid loops must start at zero",
                           stage="frontend", source=span,
                           actual=repr(block.min))
                step = getattr(block, "step", None)
                if step is not None and _int_imm(step) != 1:
                    reject(RK_E_UNSUPPORTED,
                           "block grid loops must step by one", stage="frontend",
                           source=span, actual=repr(step))
                if k_block is None and _int_imm(block.extent) != 1:
                    reject(RK_E_UNSUPPORTED,
                           "a wider block grid is only lowered together with "
                           "a K loop", stage="frontend", source=span,
                           actual=_int_imm(block.extent))
    if k_block is not None:
        if stores:
            reject(RK_E_UNSUPPORTED,
                   "activation inside a K-loop program is not lowered yet",
                   stage="frontend", source=span)
        return _k_loop_module(k_block, prim_func=prim_func, span=span,
                              options=options, profile_hash=profile_hash)

    # Any tile-level operation inside a non-thread loop is a reduction, a
    # repetition or a pipeline that this extractor cannot lower.  Silently
    # treating it as one operation would change semantics, so reject it.  Only
    # the activation store may live under loops, and its nest is verified below.
    for item in calls:
        if item.loops:
            reject(RK_E_UNSUPPORTED,
                   "tile operation inside a serial/pipelined loop is not "
                   "expanded by the MVP extractor",
                   stage="frontend", source=span, op=_op_name(item.node),
                   actual=len(item.loops))

    def pick(name: str) -> list[_Item]:
        return [item for item in calls if _op_name(item.node) == name]

    fills = pick("tl.tileop.fill")
    copies = pick("tl.tileop.copy")
    gemms = pick("tl.tileop.gemm")
    if len(gemms) == 1 and any(len(tuple(int(x) for x in p.shape)) < 2
                               for p in params):
        return _extract_broadcast_tail(
            items, params, tirx=tirx, prim_func=prim_func, span=span,
            options=options, profile_hash=profile_hash)
    if len(gemms) > 1:
        # Per-node SSA extraction: the module may contain several GEMMs, and
        # fills/copies/stores are associated by buffer identity and def-use.
        if len(gemms) == 3:
            return _extract_ffn_branch(
                items, params, tirx=tirx, prim_func=prim_func, span=span,
                options=options, profile_hash=profile_hash)
        if len(gemms) == 2:
            # Two-GEMM graphs split into the projection chain (one output) and
            # independent branch outputs (two parameter destinations).
            output_copy_count = 0
            for item in calls:
                if _op_name(item.node) != "tl.tileop.copy":
                    continue
                dst = _region(item.node.args[1], tirx, "copy destination",
                              expect_mask=2)
                if _param_of(dst.buffer, params) is not None:
                    output_copy_count += 1
            if output_copy_count > 1:
                return _extract_branch_outputs(
                    items, params, tirx=tirx, prim_func=prim_func, span=span,
                    options=options, profile_hash=profile_hash)
            return _extract_static_dataflow(
                items, params, tirx=tirx, prim_func=prim_func, span=span,
                options=options, profile_hash=profile_hash)
        reject(RK_E_UNSUPPORTED,
               "multi-GEMM static dataflow supports the projection chain (2 "
               "GEMMs) or staged SwiGLU branch (3 GEMMs) only",
               stage="frontend", source=span, actual=len(gemms))
    if len(gemms) != 1:
        reject(RK_E_UNSUPPORTED,
               "minimal TileLang extractor requires exactly one tile GEMM, "
               f"found {len(gemms)}", stage="frontend", source=span)
    if len(fills) != 1:
        reject(RK_E_NUMERIC_CONTRACT,
               "GEMM accumulator needs exactly one explicit zero fill",
               stage="frontend", source=span)
    if len(copies) != 3:
        reject(RK_E_UNSUPPORTED,
               "minimal TileLang subset requires exactly three tile copies "
               f"(A, B, accumulator), found {len(copies)}",
               stage="frontend", source=span)
    if len(stores) > 1:
        reject(RK_E_UNSUPPORTED,
               "only one activation store is supported by the minimal adapter",
               stage="frontend", source=span)

    gemm = gemms[0].node
    a_reg, b_reg, c_reg, gm, gn, gk = _gemm_header(gemm, tirx, span)
    if a_reg.physical != (gm, gk) or b_reg.physical != (gk, gn) or \
            c_reg.physical != (gm, gn):
        reject(RK_E_SHAPE,
               "GEMM regions do not describe a full MxK * KxN tile",
               stage="frontend", source=span)
    for reg, label in ((a_reg, "A"), (b_reg, "B"), (c_reg, "accumulator")):
        if _param_of(reg.buffer, params) is not None:
            reject(RK_E_SHAPE,
                   f"GEMM operand {label} must be a tile buffer, not a parameter",
                   stage="frontend", source=span)
        if reg.offset is None:
            reject(RK_E_UNSUPPORTED,
                   f"GEMM operand {label} uses a non-constant index; only a "
                   "K-blocked copy may depend on a loop variable",
                   stage="frontend", source=span)
        if reg.offset != (0, 0):
            reject(RK_E_SHAPE, f"GEMM operand {label} starts at a non-zero offset",
                   stage="frontend", source=span)
        try:
            tile_strides = tuple(int(x) for x in reg.buffer.strides) \
                if reg.buffer.strides else ()
            tile_offset = int(reg.buffer.elem_offset)
        except Exception:
            reject(RK_E_UNSUPPORTED,
                   f"GEMM operand {label} buffer layout is not inspectable",
                   stage="frontend", source=span)
        if tile_strides not in ((), (reg.physical[1], 1)) or tile_offset != 0:
            reject(RK_E_UNSUPPORTED,
                   f"GEMM operand {label} uses a strided or offset buffer layout",
                   stage="frontend", source=span,
                   actual={"strides": list(tile_strides),
                           "elem_offset": tile_offset})

    # ---- Parameter roles from the copy data flow, not from shapes. ----
    roles: dict[str, tuple[Any, _Item]] = {}
    c_copy: _Item | None = None
    for item in copies:
        copy = item.node
        src = _region(copy.args[0], tirx, "copy source", expect_mask=1)
        dst = _region(copy.args[1], tirx, "copy destination", expect_mask=2)
        if _same_buffer(dst.buffer, a_reg.buffer):
            role, expected = "A", (gm, gk)
        elif _same_buffer(dst.buffer, b_reg.buffer):
            role, expected = "B", (gk, gn)
        elif _same_buffer(src.buffer, c_reg.buffer):
            role, expected = "C", (gm, gn)
        else:
            reject(RK_E_UNSUPPORTED,
                   "tile copy does not connect a parameter to the GEMM",
                   stage="frontend", source=span,
                   actual=[_buffer_name(src.buffer), _buffer_name(dst.buffer)])
        if role == "C":
            param = _param_of(dst.buffer, params)
        else:
            param = _param_of(src.buffer, params)
        if param is None:
            reject(RK_E_UNSUPPORTED,
                   f"copy for GEMM operand {role} does not touch a parameter "
                   "buffer", stage="frontend", source=span)
        if role in roles:
            reject(RK_E_UNSUPPORTED,
                   f"more than one copy feeds GEMM operand {role}",
                   stage="frontend", source=span)
        if role == "C":
            c_copy = item
        roles[role] = (param, item)
        if src.offset is None or dst.offset is None:
            reject(RK_E_UNSUPPORTED,
                   f"copy for GEMM operand {role} uses a non-constant index",
                   stage="frontend", source=span, op=role)
        if src.physical != expected or dst.physical != expected or \
                src.offset != (0, 0) or dst.offset != (0, 0):
            reject(RK_E_SHAPE,
                   "copy region must cover exactly the full logical tensor",
                   stage="frontend", source=span, op=role)
    if set(roles) != {"A", "B", "C"} or c_copy is None:
        reject(RK_E_UNSUPPORTED,
               "copy graph does not connect A, B and the accumulator to C",
               stage="frontend", source=span, actual=sorted(roles))
    if len({id(roles[k][0]) for k in roles}) != 3:
        reject(RK_E_SHAPE,
               "A, B and C must be three distinct parameter buffers",
               stage="frontend", source=span)
    a_param, a_copy = roles["A"]
    b_param, b_copy = roles["B"]
    c_param, _ = roles["C"]

    # ---- Order and initialisation must match the closed pattern. ----
    fill = fills[0]
    fill_reg = _region(fill.node.args[0], tirx, "fill", expect_mask=2)
    if not _same_buffer(fill_reg.buffer, c_reg.buffer):
        reject(RK_E_NUMERIC_CONTRACT,
               "the explicit fill does not target the GEMM accumulator",
               stage="frontend", source=span)
    if fill_reg.offset is None:
        reject(RK_E_UNSUPPORTED, "the accumulator fill uses a non-constant index",
               stage="frontend", source=span)
    if fill_reg.physical != (gm, gn) or fill_reg.offset != (0, 0):
        reject(RK_E_NUMERIC_CONTRACT,
               "the accumulator fill does not cover the whole tile",
               stage="frontend", source=span)
    zero = _int_imm(fill.node.args[1])
    if zero != 0:
        reject(RK_E_NUMERIC_CONTRACT,
               "GEMM accumulator must be zero filled; a non-zero initial value "
               "needs a separate legalization rule",
               stage="frontend", source=span, actual=zero)
    gemm_order = gemms[0].order
    if not (fill.order < gemm_order and a_copy.order < gemm_order and
            b_copy.order < gemm_order):
        reject(RK_E_NUMERIC_CONTRACT,
               "accumulator fill and A/B copies must precede the GEMM",
               stage="frontend", source=span)
    if c_copy.order <= gemm_order:
        reject(RK_E_UNSUPPORTED,
               "accumulator must be written to C after the GEMM",
               stage="frontend", source=span)

    activation: str | None = None
    if stores:
        ok, why = _si_lu_ok(stores[0].node, stores[0], tirx, c_reg.buffer, gm, gn)
        if not ok:
            reject(RK_E_UNSUPPORTED,
                   f"unsupported accumulator update: {why}", stage="frontend",
                   source=span)
        if not (gemm_order < stores[0].order < c_copy.order):
            reject(RK_E_UNSUPPORTED,
                   "activation must run after the GEMM and before the copy to C",
                   stage="frontend", source=span)
        activation = "silu"

    if not (1 <= gm <= 4096 and 1 <= gn <= 4096 and 1 <= gk <= 4096):
        reject(RK_E_SHAPE, "M/N/K must be in 1..4096", stage="frontend",
               source=span, actual=[gm, gn, gk])
    if max(gm, gn, gk) > 256:
        reject(RK_E_UNSUPPORTED,
               "the minimal structural extractor accepts one primitive-sized "
               "M/N/K tile (<=256); K tiling and multi-tile extraction are not "
               "implemented yet", stage="frontend", source=span,
               actual=[gm, gn, gk])

    a_shape = tuple(int(x) for x in a_param.shape)
    b_shape = tuple(int(x) for x in b_param.shape)
    c_shape = tuple(int(x) for x in c_param.shape)
    if a_shape != (gm, gk) or b_shape != (gk, gn) or c_shape != (gm, gn):
        reject(RK_E_SHAPE,
               "parameter shapes do not match the GEMM tile extents",
               stage="frontend", source=span,
               actual={"A": list(a_shape), "B": list(b_shape), "C": list(c_shape)})

    if options.numeric_mode == "strict":
        contract = NumericContract(
            accumulator_dtype="float32", accumulation="strict_unspecified",
            narrowing_points=(), approximate=False,
            tolerance={"atol": 1e-3, "rtol": 1e-3})
    else:
        contract = NumericContract(
            accumulator_dtype="float16",
            accumulation="ordered_k_tiles_with_f16_writeback",
            narrowing_points=("k_tile_boundary",), approximate=False,
            tolerance={"atol": 8e-3, "rtol": 8e-3})

    b_role = "input" if b_is_input else "constant"
    values = [
        _value(0, "A", TensorType("float16", (gm, gk)), None, "input"),
        _value(1, "B", TensorType("float16", (gk, gn)), None, b_role),
        _value(2, "Z" if activation else "Y", TensorType("float16", (gm, gn)),
              0, "temporary" if activation else "output"),
    ]
    ops = [_op(0, "matmul", (0, 1), (2,),
              {"b_layout": "kn", "numeric_contract": contract.to_dict(),
               "source_accumulator_dtype": "float32",
               "tile_m": gm, "tile_n": gn, "tile_k": gk},
              span)]
    if activation:
        values.append(_value(3, "Y", TensorType("float16", (gm, gn)), 1, "output"))
        ops.append(_op(1, activation, (2,), (3,), {}, span))
        outputs = (3,)
    else:
        outputs = (2,)
    module_inputs = (0, 1) if b_is_input else (0,)
    return _finish(name=f"tilelang:{_symbol_name(prim_func)}", values=tuple(values),
                  ops=tuple(ops), inputs=module_inputs, outputs=outputs,
                  profile_hash=profile_hash, source_hash=span,
                  state_signature="")
