# SPDX-License-Identifier: GPL-3.0-or-later
"""A deliberately small, structural TileLang PrimFunc extractor.

This module does not parse ``str(prim_func)`` and does not pattern-match variable
names.  It inspects the actual TVM/TileLang TIR objects and accepts one closed
pattern:

    rank-2 FP16 A, B, C parameters
    zero-filled FP32 accumulator
    one tile-level GEMM with trans_A/trans_B false
    optional SiLU store on the accumulator
    one copy to the FP16 output

Anything else is rejected with a source span.  The optional TileLang import stays
inside this module, so importing :mod:`rknpu_backend` has no TileLang dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ..errors import (RK_E_NUMERIC_CONTRACT, RK_E_UNSUPPORTED, RK_E_SHAPE, reject)
from ..ir import Module, NumericContract, Op, TensorType, Value
from ..options import CompileOptions


@dataclass(frozen=True)
class _Region:
    buffer_name: str
    physical: tuple[int, int]
    offset: tuple[int, int]
    access_mask: int


@dataclass(frozen=True)
class _ScanItem:
    order: int
    kind: str
    node: Any


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
               alternative="install the pinned optional TileLang dependency")


def _int_imm_value(node: Any) -> int | None:
    try:
        if type(node).__name__ == "IntImm":
            return int(node.value)
    except Exception:
        return None
    return None


def _bool_imm_value(node: Any) -> bool | None:
    value = _int_imm_value(node)
    if value is None:
        try:
            if type(node).__name__ == "BoolImm":
                return bool(node.value)
        except Exception:
            return None
    return None if value is None else bool(value)


def _op_name(call: Any) -> str:
    try:
        return str(call.op.name)
    except Exception:
        return ""


def _region(call: Any, tirx: Any) -> _Region:
    if not isinstance(call, tirx.Call) or _op_name(call) != "tl.region":
        reject(RK_E_UNSUPPORTED, "expected a tl.region argument", stage="frontend")
    args = list(call.args)
    if len(args) < 4:
        reject(RK_E_UNSUPPORTED, "malformed tl.region", stage="frontend")
    load = args[0]
    if not isinstance(load, tirx.BufferLoad):
        reject(RK_E_UNSUPPORTED, "tl.region does not start with a BufferLoad",
               stage="frontend")
    mask = _int_imm_value(args[1])
    shape = tuple(_int_imm_value(x) for x in args[2:])
    if mask is None or any(v is None or v <= 0 for v in shape):
        reject(RK_E_UNSUPPORTED, "dynamic or malformed tl.region extent",
               stage="frontend")
    indices = tuple(_int_imm_value(x) for x in load.indices)
    if any(v is None or v < 0 for v in indices):
        reject(RK_E_UNSUPPORTED, "dynamic tl.region index", stage="frontend")
    try:
        name = str(load.buffer.name)
    except Exception:
        reject(RK_E_UNSUPPORTED, "tl.region buffer has no name", stage="frontend")
    if len(shape) != 2:
        reject(RK_E_SHAPE, "MVP TileLang subset only accepts rank-2 regions",
               stage="frontend")
    return _Region(name, (shape[0], shape[1]), (indices[0], indices[1]), mask)


def _scan_block(node: Any, tirx: Any, out: list[_ScanItem], counter: list[int],
                entered_pipeline: bool = False) -> None:
    """Collect semantic calls/stores in execution order, skipping launch bindings."""
    if isinstance(node, tirx.SBlockRealize):
        _scan_block(node.block, tirx, out, counter, entered_pipeline)
        return
    if isinstance(node, tirx.SBlock):
        if node.init is not None:
            _scan_block(node.init, tirx, out, counter, entered_pipeline)
        _scan_block(node.body, tirx, out, counter, entered_pipeline)
        return
    if isinstance(node, tirx.SeqStmt):
        for stmt in node.seq:
            _scan_block(stmt, tirx, out, counter, entered_pipeline)
        return
    if isinstance(node, tirx.For):
        # Kernel/thread-binding loops are launch structure.  Serial, Parallel and
        # Pipelined loops inside the tile body are semantic; the MVP accepts only
        # the activation Parallel loops and rejects any K pipeline.
        try:
            thread = node.thread_binding is not None
        except Exception:
            thread = False
        if thread:
            _scan_block(node.body, tirx, out, counter, entered_pipeline)
            return
        out.append(_ScanItem(counter[0], "for", node))
        counter[0] += 1
        _scan_block(node.body, tirx, out, counter, True)
        return
    if isinstance(node, tirx.Evaluate):
        _scan_block(node.value, tirx, out, counter, entered_pipeline)
        return
    if isinstance(node, tirx.Call):
        out.append(_ScanItem(counter[0], "call", node))
        counter[0] += 1
        # Do not recurse into nested tl.region internals; they are not semantics
        # by themselves.  Recurse into arithmetic calls to catch unsupported ops.
        name = _op_name(node)
        if not name.startswith("tl."):
            for arg in node.args:
                if hasattr(arg, "__class__") and type(arg).__module__.startswith("tvm"):
                    _scan_block(arg, tirx, out, counter, entered_pipeline)
        return
    if isinstance(node, tirx.BufferStore):
        out.append(_ScanItem(counter[0], "store", node))
        counter[0] += 1
        return
    if type(node).__name__ in {"Bind", "LaunchThread", "AssertStmt", "BufferLoad",
                               "IntImm", "FloatImm", "BoolImm", "Var"}:
        return
    # Unknown semantic nodes are not silently ignored.
    reject(RK_E_UNSUPPORTED,
           f"unsupported TileLang node {type(node).__name__}", stage="frontend")


def _same_load(expr: Any, buffer_name: str, tirx: Any) -> bool:
    return (isinstance(expr, tirx.BufferLoad) and
            str(expr.buffer.name) == buffer_name)


def _detect_silu(store: Any, acc_name: str, tirx: Any) -> bool:
    """Match exactly ``acc[i,j] = acc[i,j] * sigmoid(acc[i,j])``."""
    if not isinstance(store, tirx.BufferStore) or str(store.buffer.name) != acc_name:
        return False
    value = store.value
    if type(value).__name__ != "Mul":
        return False
    a, b = value.a, value.b
    if _same_load(a, acc_name, tirx) and isinstance(b, tirx.Call) and \
            _op_name(b) == "tirx.sigmoid" and len(b.args) == 1 and \
            _same_load(b.args[0], acc_name, tirx):
        return True
    if _same_load(b, acc_name, tirx) and isinstance(a, tirx.Call) and \
            _op_name(a) == "tirx.sigmoid" and len(a.args) == 1 and \
            _same_load(a.args[0], acc_name, tirx):
        return True
    return False


def _parameter_buffers(prim_func: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        items = list(prim_func.buffer_map.items())
    except Exception:
        reject(RK_E_UNSUPPORTED, "PrimFunc has no buffer_map", stage="frontend")
    for _, buffer in items:
        name = str(buffer.name)
        out[name] = buffer
    if len(out) != 3:
        reject(RK_E_UNSUPPORTED,
               "minimal TileLang subset requires exactly A, B and C buffers",
               stage="frontend", actual=sorted(out))
    for name, buffer in out.items():
        try:
            shape = tuple(int(x) for x in buffer.shape)
            dtype = str(buffer.dtype)
        except Exception:
            reject(RK_E_SHAPE, f"buffer {name} is not static rank-2 FP16",
                   stage="frontend")
        if len(shape) != 2 or dtype != "float16":
            reject(RK_E_SHAPE, "TileLang MVP buffers must be static rank-2 FP16",
                   stage="frontend", actual={"name": name, "shape": shape,
                                             "dtype": dtype})
    return out


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


def extract(prim_func: Any, *, options: CompileOptions,
            profile_hash: str = "") -> Module:
    tirx = _import_tirx()
    if not isinstance(prim_func, tirx.PrimFunc):
        reject(RK_E_UNSUPPORTED,
               "compile_tilelang expects a tvm.tirx.PrimFunc, got "
               f"{type(prim_func).__name__}", stage="frontend")
    params = _parameter_buffers(prim_func)
    span = _source_span(prim_func)
    items: list[_ScanItem] = []
    _scan_block(prim_func.body, tirx, items, [0])

    # A serial/pipelined loop around GEMM is not silently expanded in this first
    # extractor.  The MVP fixture is one complete tile.
    for item in items:
        if item.kind == "for":
            # Activation Parallel loops are allowed only when they contain no
            # tile-level operation.  Any loop wrapping copy/fill/gemm is a K or
            # dataflow loop and is outside this minimal extractor.
            nested = _collect_call_names(item.node, tirx)
            if any(name.startswith("tl.tileop.") for name in nested):
                reject(RK_E_UNSUPPORTED,
                       "serial/pipelined loops around GEMM are not in the minimal "
                       "TileLang subset", stage="frontend", source=span)

    calls = [item.node for item in items if item.kind == "call"]
    stores = [item.node for item in items if item.kind == "store"]
    for call in calls:
        name = _op_name(call)
        if name.startswith("tl.tileop.") and name not in {
                "tl.tileop.fill", "tl.tileop.copy", "tl.tileop.gemm"}:
            reject(RK_E_UNSUPPORTED, f"unsupported TileLang tileop {name}",
                   stage="frontend", source=span, op=name)
        if "atomic" in name or "barrier" in name or "shuffle" in name:
            reject(RK_E_UNSUPPORTED,
                   f"GPU communication primitive {name} cannot be ignored",
                   stage="frontend", source=span, op=name)

    gemms = [call for call in calls if _op_name(call) == "tl.tileop.gemm"]
    if len(gemms) != 1:
        reject(RK_E_UNSUPPORTED,
               "minimal TileLang extractor requires exactly one tl.tileop.gemm",
               stage="frontend", source=span)
    gemm = gemms[0]
    if len(gemm.args) < 10:
        reject(RK_E_UNSUPPORTED, "malformed tl.tileop.gemm", stage="frontend")
    a_reg, b_reg, c_reg = (_region(gemm.args[i], tirx) for i in range(3))
    trans_a = _bool_imm_value(gemm.args[3])
    trans_b = _bool_imm_value(gemm.args[4])
    if trans_a is None or trans_b is None:
        reject(RK_E_UNSUPPORTED, "dynamic GEMM transpose flags", stage="frontend")
    if trans_a or trans_b:
        reject(RK_E_UNSUPPORTED,
               "the minimal TileLang adapter accepts only trans_A=false, "
               "trans_B=false; explicit packing rules are required for transposes",
               stage="frontend", source=span)
    gm, gn, gk = (_int_imm_value(gemm.args[i]) for i in range(5, 8))
    if gm is None or gn is None or gk is None:
        reject(RK_E_SHAPE, "dynamic GEMM extents are unsupported", stage="frontend")

    # Identify parameter roles from logical shapes and copy regions.  Names are
    # used only to index the actual TIR buffer objects, not to guess semantics.
    # The three names vary by user program, so match region extents to parameters.
    a_name = b_name = c_name = None
    for name, buffer in params.items():
        shape = tuple(int(x) for x in buffer.shape)
        if shape == (gm, gk):
            a_name = a_name or name
        if shape == (gk, gn):
            b_name = b_name or name
        if shape == (gm, gn):
            c_name = c_name or name
    if not (a_name and b_name and c_name):
        reject(RK_E_SHAPE,
               "minimal GEMM requires A[M,K], B[K,N], C[M,N] parameter shapes",
               stage="frontend", source=span,
               actual={k: list(map(int, v.shape)) for k, v in params.items()})
    if len({a_name, b_name, c_name}) != 3:
        # Square shapes can make the role ambiguous; we deliberately require a
        # unique structural mapping instead of guessing.
        reject(RK_E_SHAPE, "ambiguous parameter roles for square GEMM shapes",
               stage="frontend", source=span)

    # The accumulator must be zero-filled before the GEMM and copied to C after.
    fill_ok = False
    for call in calls:
        if _op_name(call) == "tl.tileop.fill" and len(call.args) == 2:
            reg = _region(call.args[0], tirx)
            zero_value = _int_imm_value(call.args[1])
            if (reg.buffer_name == c_reg.buffer_name and zero_value == 0 and
                    reg.physical == (gm, gn) and reg.offset == (0, 0)):
                fill_ok = True
    if not fill_ok:
        reject(RK_E_NUMERIC_CONTRACT,
               "GEMM accumulator must have an explicit zero fill; non-zero initial "
               "accumulation needs a separate legalization rule",
               stage="frontend", source=span)
    if c_reg.physical != (gm, gn) or a_reg.physical != (gm, gk) or \
            b_reg.physical != (gk, gn):
        reject(RK_E_SHAPE, "GEMM region extents do not describe a full MxK * KxN tile",
               stage="frontend", source=span)

    copies = [call for call in calls if _op_name(call) == "tl.tileop.copy"]
    if len(copies) != 3:
        reject(RK_E_UNSUPPORTED,
               "minimal TileLang subset requires exactly three copies: "
               "A->shared, B->shared, accumulator->C",
               stage="frontend", source=span, actual=len(copies))
    copy_roles: set[str] = set()
    for call in copies:
        src, dst = (_region(call.args[i], tirx) for i in range(2))
        if src.buffer_name == a_name and dst.buffer_name == a_reg.buffer_name:
            expected = (gm, gk)
            role = "A"
        elif src.buffer_name == b_name and dst.buffer_name == b_reg.buffer_name:
            expected = (gk, gn)
            role = "B"
        elif src.buffer_name == c_reg.buffer_name and dst.buffer_name == c_name:
            expected = (gm, gn)
            role = "C"
        else:
            reject(RK_E_UNSUPPORTED,
                   "unsupported copy edge in minimal TileLang subset",
                   stage="frontend", source=span,
                   actual=[src.buffer_name, dst.buffer_name])
        if src.physical != expected or dst.physical != expected or \
                src.offset != (0, 0) or dst.offset != (0, 0):
            reject(RK_E_SHAPE,
                   "copy region must cover exactly the full logical tensor",
                   stage="frontend", source=span, op=role)
        copy_roles.add(role)
    if copy_roles != {"A", "B", "C"}:
        reject(RK_E_SHAPE, "copy graph does not cover A, B and C",
               stage="frontend", source=span)

    activation: str | None = None
    if stores:
        if len(stores) != 1:
            reject(RK_E_UNSUPPORTED,
                   "only one activation store is supported by the minimal adapter",
                   stage="frontend", source=span)
        if not _detect_silu(stores[0], c_reg.buffer_name, tirx):
            reject(RK_E_UNSUPPORTED,
                   "unsupported accumulator store; the only registered expression "
                   "is acc = acc * sigmoid(acc) (SiLU)",
                   stage="frontend", source=span)
        activation = "silu"

    if not (1 <= gm <= 4096 and 1 <= gn <= 4096 and 1 <= gk <= 4096):
        reject(RK_E_SHAPE, "M/N/K must be in 1..4096", stage="frontend",
               source=span, actual=[gm, gn, gk])
    if max(gm, gn, gk) > 256:
        reject(RK_E_UNSUPPORTED,
               "the minimal structural extractor accepts one primitive-sized "
               "M/N/K tile (<=256); K tiling and multi-tile extraction are not "
               "implemented yet",
               stage="frontend", source=span, actual=[gm, gn, gk])

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

    values = [
        Value(0, "A", TensorType("float16", (gm, gk)), None, "input"),
        Value(1, "B", TensorType("float16", (gk, gn)), None, "constant"),
        Value(2, "Z" if activation else "Y", TensorType("float16", (gm, gn)),
              0, "temporary" if activation else "output"),
    ]
    ops = [Op(0, "matmul", (0, 1), (2,),
              {"b_layout": "kn", "numeric_contract": contract.to_dict(),
               "source_accumulator_dtype": "float32",
               "tile_m": gm, "tile_n": gn, "tile_k": gk},
              span)]
    if activation:
        values.append(Value(3, "Y", TensorType("float16", (gm, gn)), 1, "output"))
        ops.append(Op(1, activation, (2,), (3,), {}, span))
        outputs = (3,)
    else:
        outputs = (2,)
    return Module(name=f"tilelang:{_symbol_name(prim_func)}", values=tuple(values),
                  ops=tuple(ops), inputs=(0,), outputs=outputs,
                  profile_hash=profile_hash, source_hash=_source_span(prim_func),
                  state_signature="")


def _symbol_name(prim_func: Any) -> str:
    try:
        return str(prim_func.attrs["global_symbol"])
    except Exception:
        return "kernel"


def _collect_call_names(node: Any, tirx: Any) -> list[str]:
    names: list[str] = []
    if isinstance(node, tirx.Call):
        names.append(_op_name(node))
        for arg in node.args:
            if hasattr(arg, "__class__") and type(arg).__module__.startswith("tvm"):
                names.extend(_collect_call_names(arg, tirx))
    elif isinstance(node, tirx.SeqStmt):
        for stmt in node.seq:
            names.extend(_collect_call_names(stmt, tirx))
    elif isinstance(node, (tirx.For, tirx.SBlock, tirx.SBlockRealize)):
        body = getattr(node, "body", None)
        if body is None and isinstance(node, tirx.SBlockRealize):
            body = node.block
        if body is not None:
            names.extend(_collect_call_names(body, tirx))
    elif isinstance(node, tirx.Evaluate):
        names.extend(_collect_call_names(node.value, tirx))
    elif isinstance(node, tirx.BufferStore):
        names.extend(_collect_call_names(node.value, tirx))
    return names
