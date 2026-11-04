# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared TileLang TIR scanning/expression helpers (adapter retired)."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
from typing import Any, Iterable

from ..errors import (RK_E_NUMERIC_CONTRACT, RK_E_SHAPE, RK_E_UNSUPPORTED,
                      reject)
from ..ir import NumericContract, TensorType
from ..options import CompileOptions
from .graph_builder import active_builder

ALLOWED_TILEOPS = {"tl.tileop.fill", "tl.tileop.copy", "tl.tileop.gemm"}
FORBIDDEN_TOKENS = ("atomic", "barrier", "shuffle", "sync", "reduce")
SCHEDULE_ATTRS = {
    "rknpu.schedule", "rknpu.tile_m", "rknpu.tile_n", "rknpu.mn_order",
    "rknpu.source_k_rounding",
}
PIPELINE_NOT_APPLIED_REASON = (
    "no software pipeline is implemented on this target: the K chain is "
    "emitted as an ordered accumulation of 256-wide tasks, so the request "
    "is recorded but not applied")

# not by itself request a non-default M/N plan.
SCHEDULE_ROUNDING_ATTR = "rknpu.source_k_rounding"
SOURCE_ROUNDINGS = {"explicit_per_k_fp16", "single_k_tile_fp16",
                    "fp32_final_narrowing"}
SCHEDULE_K_TILE = 256
MN_ORDERS = {"m_major", "n_major"}
F16_NARROWING_NAMES = ("gemm1_accumulator", "sigmoid", "silu_mul",
                       "gemm2_accumulator", "residual_add")
FFN_NARROWING_NAMES = (
    "gate_accumulator", "up_accumulator", "silu", "act_mul",
    "projection_accumulator", "residual_add",
)



@dataclass(frozen=True)
class _ScheduleAnnotation:
    present: bool
    tile_m: int | None
    tile_n: int | None
    order: str | None

def _schedule_annotation(prim_func: Any, span: str) -> _ScheduleAnnotation:
    """Read and validate explicit ``rknpu.*`` schedule annotations.

    Any schedule-shaped annotation outside the admitted set is rejected.  This
    is the frontend half of "no silent schedule drops": an unknown K order,
    tile size or traversal key cannot disappear between TileLang and RKIR.
    """
    attrs = dict(getattr(prim_func, "attrs", {}) or {})
    # The whole ``rknpu.*`` namespace is the backend's schedule namespace: a key
    # there is a request this backend either honours or refuses.  Silently
    # ignoring one (an older prefix list did that for anything outside
    # schedule/tile_/mn_/k_/source_) would claim support the runtime never
    # provides.
    for key in attrs:
        if key.startswith("rknpu.") and key not in SCHEDULE_ATTRS:
            reject(RK_E_UNSUPPORTED,
                   f"unsupported TileLang schedule annotation {key!r}",
                   stage="frontend", source=span,
                   actual=key, required=sorted(SCHEDULE_ATTRS))
    present = any(key in attrs for key in SCHEDULE_ATTRS
                  if key != SCHEDULE_ROUNDING_ATTR)
    if not present:
        return _ScheduleAnnotation(False, None, None, None)
    marker = attrs.get("rknpu.schedule")
    if marker is not None and marker != "mn_tiles_v1":
        reject(RK_E_UNSUPPORTED,
               "unsupported TileLang schedule contract", stage="frontend",
               source=span, actual=marker, required="mn_tiles_v1")
    raw_m = attrs.get("rknpu.tile_m")
    raw_n = attrs.get("rknpu.tile_n")
    if (raw_m is None) != (raw_n is None):
        reject(RK_E_UNSUPPORTED,
               "rknpu.tile_m and rknpu.tile_n must be supplied together",
               stage="frontend", source=span)
    def schedule_int(value: Any) -> int | None:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return _int_imm(value)
    tile_m = None if raw_m is None else schedule_int(raw_m)
    tile_n = None if raw_n is None else schedule_int(raw_n)
    if raw_m is not None and (tile_m is None or tile_n is None):
        reject(RK_E_UNSUPPORTED, "schedule tile sizes must be static integers",
               stage="frontend", source=span)
    order = attrs.get("rknpu.mn_order")
    if order is not None and not isinstance(order, str):
        order = str(order)
    if order is not None and order not in MN_ORDERS:
        reject(RK_E_UNSUPPORTED, "unsupported M/N traversal order",
               stage="frontend", source=span, actual=order,
               required=sorted(MN_ORDERS))
    if tile_m is not None and (not 1 <= tile_m <= 256 or
                               not 1 <= tile_n <= 256):
        reject(RK_E_UNSUPPORTED,
               "M/N schedule tiles must be within 1..256", stage="frontend",
               source=span, actual=[tile_m, tile_n])
    return _ScheduleAnnotation(True, tile_m, tile_n, order)

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

def _int_imm(node: Any) -> int | None:
    try:
        if type(node).__name__ == "IntImm":
            return int(node.value)
    except Exception:
        return None
    return None

def _op_name(call: Any) -> str:
    try:
        return str(call.op.name)
    except Exception:
        return ""

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

def _canonical_source_name(path: str) -> str:
    """Stable source identity independent of checkout path and TVM address.

    ``str(tvm.Span)`` embeds the Python object address of the ``SourceName``.
    Using it directly made otherwise identical recompilations produce different
    plan hashes.  We replace that address with a fixed marker and bind the
    identity to the source file's relative path and content digest.
    """
    resolved = Path(path).resolve()
    package_root = Path(__file__).resolve().parents[3]
    try:
        relative = str(resolved.relative_to(package_root))
    except ValueError:
        relative = resolved.name
    try:
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()[:16]
    except OSError:
        digest = "unreadable"
    return f"{relative}#sha256:{digest}"

def _source_span(prim_func: Any) -> str:
    try:
        span = prim_func.span
        if span is not None:
            text = str(span)
            text = _SOURCE_NAME.sub(
                lambda match: f"SourceName({_canonical_source_name(match.group(1))}, 0x0)",
                text)
            # Any remaining TVM address is identity-irrelevant.
            return re.sub(r"0x[0-9a-fA-F]+", "0x0", text)
    except Exception:
        pass
    try:
        return str(prim_func.attrs["global_symbol"])
    except Exception:
        return "tilelang:unknown"

@dataclass(frozen=True)
class _Expr:
    kind: str
    buffer: Any | None = None
    args: tuple["_Expr", ...] = ()
    dtype: str = ""

def _raw_buffer_shape(buffer: Any) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in buffer.shape)
    except Exception:
        reject(RK_E_UNSUPPORTED, "tile buffer shape is not inspectable",
               stage="frontend")

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

def _flatten_conjuncts(expr: Any, tirx: Any) -> list[Any]:
    if type(expr).__name__ == "And":
        return (_flatten_conjuncts(expr.a, tirx) +
                _flatten_conjuncts(expr.b, tirx))
    return [expr]






@dataclass(frozen=True)
class _Region:
    buffer: Any
    physical: tuple[int, int]
    offset: tuple[int, int] | None
    access_mask: int
    raw_indices: tuple[Any, Any] = (None, None)


def _region(call: Any, tirx: Any, where: str,
            expect_mask: int | None = None) -> _Region:
    if type(call).__name__ != "Call" or len(call.args) < 4:
        reject(RK_E_UNSUPPORTED, f"{where}: malformed region",
               stage="frontend")
    load = call.args[0]
    if type(load).__name__ != "BufferLoad":
        reject(RK_E_UNSUPPORTED, f"{where}: region is not a buffer load",
               stage="frontend")
    mask = _int_imm(call.args[1])
    shape = [_int_imm(call.args[2]), _int_imm(call.args[3])]
    if mask is None or any(v is None or v <= 0 for v in shape):
        reject(RK_E_UNSUPPORTED, f"{where}: dynamic or malformed region extent",
               stage="frontend")
    if len(shape) != 2:
        reject(RK_E_SHAPE, "MVP TileLang subset only accepts rank-2 regions",
               stage="frontend", actual=list(shape))
    if len(load.indices) != 2:
        reject(RK_E_UNSUPPORTED, f"{where}: expected a rank-2 region start",
               stage="frontend")
    if not _predicate_true(getattr(load, "predicate", None)):
        reject(RK_E_UNSUPPORTED,
               f"{where}: region load carries a predicate; conditional loads "
               "are not lowered", stage="frontend")
    if mask not in (1, 2, 3):
        reject(RK_E_UNSUPPORTED,
               f"{where}: unknown region access mask {mask}",
               stage="frontend")
    if expect_mask is not None and mask != expect_mask:
        reject(RK_E_UNSUPPORTED,
               f"{where}: region access mask {mask} does not match "
               f"{expect_mask}", stage="frontend")
    raw = tuple(load.indices)
    constants = tuple(_int_imm(x) for x in raw)
    offset = (None if any(v is None or v < 0 for v in constants)
              else (constants[0], constants[1]))
    return _Region(load.buffer, (shape[0], shape[1]), offset, mask, raw)


def _buffer_shape(buffer: Any) -> tuple[int, int]:
    shape = _raw_buffer_shape(buffer)
    if len(shape) != 2:
        reject(RK_E_SHAPE, "tile buffer must be rank 2", stage="frontend",
               actual=list(shape))
    if str(buffer.dtype) != "float16":
        reject(RK_E_NUMERIC_CONTRACT,
               "static dataflow requires rank-2 FP16 tile buffers; an "
               "implicit cast would hide a narrowing point", stage="frontend",
               actual={"shape": list(shape), "dtype": str(buffer.dtype)})
    return shape


def _strip_expr_casts(expr: Any) -> Any:
    while type(expr).__name__ == "Cast":
        expr = expr.value
    return expr


def _check_elementwise_store(item: Any, tirx: Any, params: list[Any],
                             span: str):
    store = item.node
    if type(store).__name__ != "BufferStore":
        reject(RK_E_UNSUPPORTED, "elementwise stage needs a BufferStore",
               stage="frontend", source=span)
    target = store.buffer
    target_shape = _buffer_shape(target)
    expr = _parse_elementwise_expr(_strip_expr_casts(store.value), tirx,
                                   params, span)
    if expr is None:
        reject(RK_E_UNSUPPORTED,
               "elementwise store expression is not a supported local op",
               stage="frontend", source=span)
    return target, target_shape, expr


def _pipeline_record(annotations):
    """Record a source pipeline request without claiming it was applied.

    num_stages is the TileLang buffering hint (0 means no pipeline); the
    software pipeline order/stage lists are the manual form of the hint.
    """
    raw = {}
    for key, value in dict(annotations or {}).items():
        try:
            raw[str(key)] = int(value)
        except (TypeError, ValueError):
            raw[str(key)] = str(value)
    stages = 0
    for value in (raw.get("num_stages"), raw.get("software_pipeline_stages")):
        try:
            stages = max(stages, int(value))
        except (TypeError, ValueError):
            continue
    if any(key in raw for key in ("software_pipeline_order",
                                  "software_pipeline_stage")):
        stages = max(stages, 1)
    return {"source_loop_annotations": raw,
            "source_pipeline_stages": stages,
            "pipeline_hint_dropped": bool(stages),
            "pipeline_not_applied_reason":
                PIPELINE_NOT_APPLIED_REASON if stages else None}


def _match_local_elementwise(node: Any, *, store_cast: bool = False
                             ) -> tuple[str, tuple[Any, ...]]:
    """Match one supported scalar operation, preserving its exact operands.

    A store cast is permitted only for an already-FP16 expression.  Arithmetic
    below a load operand is never discarded or flattened into a load list.
    """
    if store_cast and type(node).__name__ == "Cast" and \
            str(node.dtype) == "float16" and str(node.value.dtype) == "float16":
        node = node.value

    def load(expr: Any) -> bool:
        return type(expr).__name__ == "BufferLoad"

    def sigmoid(expr: Any) -> Any:
        if type(expr).__name__ == "Call" and \
                _op_name(expr) == "tirx.sigmoid" and len(expr.args) == 1 and \
                load(expr.args[0]):
            return expr.args[0]
        return None

    operand = sigmoid(node)
    if operand is not None:
        return "sigmoid", (operand,)
    kind = type(node).__name__
    if kind in {"Add", "Mul"}:
        if load(node.a) and load(node.b):
            return kind.lower(), (node.a, node.b)
        if kind == "Mul":
            for direct, call in ((node.a, node.b), (node.b, node.a)):
                arg = sigmoid(call)
                if load(direct) and arg is not None and \
                        _same_buffer(direct.buffer, arg.buffer) and \
                        len(direct.indices) == len(arg.indices) and \
                        all(_same_var(a, b) or str(a) == str(b)
                            for a, b in zip(direct.indices, arg.indices)):
                    return "silu", (direct,)
    reject(RK_E_UNSUPPORTED,
           "unsupported nested elementwise expression " + (_op_name(node) or kind) +
           "; expected direct "
           "load Add/Mul, sigmoid(load), or load * sigmoid(the same load)",
           stage="rewrite", actual=str(node))
