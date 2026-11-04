# SPDX-License-Identifier: GPL-3.0-or-later
"""Launch-boundary aware staging frontend.

This module owns the actual top-level ``T.Kernel`` segmentation.  Stage-local
operation parsing is kept behind the existing validation helpers; the formal
entries build a carrier from the actual operation chain and then use the common compile pipeline.  The
adapter's ``extract`` no longer dispatches staged sources.
"""
from __future__ import annotations

from typing import Any

from dataclasses import dataclass
from .tir_utils import (
    ALLOWED_TILEOPS, PIPELINE_NOT_APPLIED_REASON,
    RK_E_NUMERIC_CONTRACT, RK_E_SHAPE, RK_E_UNSUPPORTED, reject,
    _Item, _Region, _affine_expr, _buffer_name, _buffer_shape,
    _check_elementwise_store, _import_tirx, _int_imm, _op_name,
    _param_buffers, _param_of, _parse_elementwise_expr,
    _predicate_true, _raw_buffer_shape, _region, _same_buffer,
    _same_var, _scan, _source_span, _strip_expr_casts,
)



def stage_groups(prim_func: Any) -> list[list[Any]]:
    """Return contiguous top-level launch groups in actual source order."""
    tirx = _import_tirx()
    items: list[Any] = []
    _scan(prim_func.body, tirx, items, [0], (),
                _source_span(prim_func))
    for item in items:
        if item.kind == "call" and _op_name(item.node) not in ALLOWED_TILEOPS:
            reject(RK_E_UNSUPPORTED, "unconsumed source call in staged program",
                   stage="frontend", source=_source_span(prim_func),
                   actual=str(item.node))
    active = [item for item in items
              if item.kind == "store" or
              (item.kind == "call" and
               _op_name(item.node) in ALLOWED_TILEOPS)]
    groups: list[list[Any]] = []
    keys: list[int] = []
    for item in active:
        key = id(item.blocks)
        if not keys or key != keys[-1]:
            keys.append(key)
            groups.append([])
        groups[-1].append(item)
    return groups


def is_staged_candidate(prim_func: Any) -> bool:
    try:
        return len(stage_groups(prim_func)) > 1
    except Exception:
        return False









# Stage contract parsers migrated from the adapter.









def _prove_elementwise_stage(stage_index: int, stage_items: list[_Item],
                              params: list[Any], *, tirx: Any,
                              span: str) -> dict[str, Any]:
    """Prove local loop/region contracts without generating an execution chain."""
    if any(item.guards for item in stage_items):
        reject(RK_E_UNSUPPORTED, "conditional staged elementwise effects are not lowered",
               stage="frontend", source=span)
    calls = [item for item in stage_items if item.kind == "call"]
    tile_calls = [item for item in calls
                  if _op_name(item.node) in ALLOWED_TILEOPS]
    stores = [item for item in stage_items if item.kind == "store"]
    copies = [item for item in tile_calls
              if _op_name(item.node) == "tl.tileop.copy"]
    fills = [item for item in tile_calls
             if _op_name(item.node) == "tl.tileop.fill"]
    gemms = [item for item in tile_calls
             if _op_name(item.node) == "tl.tileop.gemm"]
    if gemms or fills:
        reject(RK_E_UNSUPPORTED,
               f"stage {stage_index} is an elementwise stage; GEMM/fill is "
               "unexpected", stage="frontend", source=span)
    if len(stores) != 1:
        reject(RK_E_UNSUPPORTED,
               f"stage {stage_index} needs exactly one elementwise store",
               stage="frontend", source=span, actual=len(stores))
    def pidx(buffer: Any) -> int | None:
        for i, param in enumerate(params):
            if _same_buffer(buffer, param):
                return i
        return None

    store_item = stores[0]
    target, target_shape, _ = _check_elementwise_store(
        store_item, tirx, params, span)
    writes = []
    for copy_item in copies:
        src = _region(copy_item.node.args[0], tirx, "copy source",
                      expect_mask=1)
        dst = _region(copy_item.node.args[1], tirx, "copy destination",
                      expect_mask=2)
        if pidx(dst.buffer) is not None:
            writes.append((copy_item, src, dst))
    if len(writes) != 1:
        reject(RK_E_UNSUPPORTED,
               f"stage {stage_index} must publish exactly one parameter value",
               stage="frontend", source=span, actual=len(writes))
    publish_item, pub_src, pub_dst = writes[0]
    out_pid = pidx(pub_dst.buffer)
    if out_pid is None:
        reject(RK_E_UNSUPPORTED,
               f"stage {stage_index} publish destination is not a parameter",
               stage="frontend", source=span)
    if not _same_buffer(target, pub_src.buffer):
        reject(RK_E_UNSUPPORTED,
               f"stage {stage_index} store target is not the published tile",
               stage="frontend", source=span)
    if pub_dst.physical != target_shape:
        reject(RK_E_SHAPE,
               f"stage {stage_index} published rectangle differs from tile",
               stage="frontend", source=span,
               actual={"published": list(pub_dst.physical),
                       "tile": list(target_shape)})
    out_shape = tuple(int(x) for x in params[out_pid].shape)
    source_regions: list[_Region] = []
    for copy_item in copies:
        if copy_item is publish_item:
            continue
        csrc = _region(copy_item.node.args[0], tirx, "copy source",
                       expect_mask=1)
        _region(copy_item.node.args[1], tirx, "copy destination", expect_mask=2)
        src_pid = pidx(csrc.buffer)
        if src_pid is not None:
            source_regions.append(csrc)
    from .tir_utils import _match_local_elementwise
    opcode, _ = _match_local_elementwise(store_item.node.value,
                                          store_cast=True)
    if opcode not in {"sigmoid", "mul", "add"}:
        reject(RK_E_UNSUPPORTED, "unsupported staged elementwise operation",
               stage="frontend", actual=opcode)
    # Prove copy source bases exactly against the output tile indices.
    block_roles = {f"b{i}": loop.loop_var
                   for i, loop in enumerate(store_item.blocks)}
    row_parsed = _affine_expr(pub_dst.raw_indices[0], tirx, block_roles)
    col_parsed = _affine_expr(pub_dst.raw_indices[1], tirx, block_roles)
    if row_parsed is None or col_parsed is None:
        reject(RK_E_UNSUPPORTED,
               f"stage {stage_index} store indices are not affine",
               stage="frontend", source=span)
    row_block = next((k for k, v in row_parsed[0].items()
                      if v == target_shape[0] and k.startswith("b")), None)
    col_block = next((k for k, v in col_parsed[0].items()
                      if v == target_shape[1] and k.startswith("b")), None)
    if row_block is None or col_block is None:
        reject(RK_E_UNSUPPORTED,
               f"stage {stage_index} cannot identify M/N block variables",
               stage="frontend", source=span)
    block_extents = [_int_imm(loop.extent) for loop in store_item.blocks]
    row_grid = block_extents[int(row_block[1:])] if row_block else 1
    col_grid = block_extents[int(col_block[1:])] if col_block else 1
    if row_grid is None or col_grid is None or             row_grid * target_shape[0] != out_shape[0] or             col_grid * target_shape[1] != out_shape[1]:
        reject(RK_E_SHAPE,
               f"stage {stage_index} elementwise tile does not cover its output",
               stage="frontend", source=span,
               actual={"grid": [row_grid, col_grid],
                       "tile": list(target_shape),
                       "output": list(out_shape)})
    for csrc in source_regions:
        parsed_row = _affine_expr(csrc.raw_indices[0], tirx, block_roles)
        parsed_col = _affine_expr(csrc.raw_indices[1], tirx, block_roles)
        if parsed_row is None or parsed_col is None:
            reject(RK_E_UNSUPPORTED,
                   f"stage {stage_index} copy source indices are not affine",
                   stage="frontend", source=span)
        if parsed_row[1] != 0 or parsed_row[0] != {row_block: target_shape[0]}:
            reject(RK_E_SHAPE,
                   f"stage {stage_index} copy source row is not the M tile",
                   stage="frontend", source=span,
                   actual=parsed_row)
        expected_col = {col_block: target_shape[1]}
        if {k: v for k, v in parsed_col[0].items() if v} != expected_col:
            reject(RK_E_SHAPE,
                   f"stage {stage_index} copy source column is not the N tile",
                   stage="frontend", source=span,
                   actual=parsed_col[0], required=expected_col)
        if csrc.physical != target_shape:
            reject(RK_E_SHAPE,
                   f"stage {stage_index} copy source does not cover the tile",
                   stage="frontend", source=span,
                   actual=list(csrc.physical))
    return {"opcode": opcode, "store": store_item,
            "publish": publish_item, "output": out_pid,
            "tile": target_shape}



# Shared expression/region validators migrated with the stage parsers.


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

def _strip_expr_casts(expr: _Expr) -> _Expr:
    while expr.kind == "cast" and expr.args:
        expr = expr.args[0]
    return expr








# Remaining region/affine primitives used by stage contract checks.

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



# Region value object shared by the migrated stage parsers.
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


def _simple_gemm_stage_tasks(stage_index: int, stage_items: list[Any],
                             params: list[Any], *, tirx: Any,
                             span: str) -> dict[str, Any]:
    from . import rknpu_pass
    import itertools
    import tvm.tirx.stmt_functor as stmt_functor

    blocks = list(stage_items[0].blocks) if stage_items else []
    extents = [int(loop.extent) for loop in blocks]
    calls = [item for item in stage_items if item.kind == "call" and
             _op_name(item.node).startswith("tl.tileop.")]

    def eval_expr(expr: Any, values: dict[Any, Any]) -> int:
        if type(expr) is int:
            return expr
        reduced = stmt_functor.substitute(expr, values) if values else expr
        reduced = rknpu_pass._arith().simplify(reduced)
        value = rknpu_pass._const(reduced)
        if value is None:
            reject(RK_E_UNSUPPORTED,
                         "simple staged region is not static after block binding",
                         stage="frontend", actual=str(reduced))
        return int(value)

    tasks: list[dict[str, Any]] = []
    gemm_info = None
    copy_sources: list[str] = []
    output_name = None
    last_gemm_task = None
    for combo in itertools.product(*[range(e) for e in extents]):
        last_gemm_task = None
        values = {loop.loop_var: tirx.IntImm("int32", idx)
                  for loop, idx in zip(blocks, combo)}
        for item in calls:
            name = _op_name(item.node)
            regions = rknpu_pass._region_exprs(item.node)
            if name == "tl.tileop.fill":
                reg = regions[0]
                tasks.append({
                    "kind": "fill",
                    "destination": str(reg["buffer"].name),
                    "value": eval_expr(item.node.args[1], values),
                    "region": {"row": eval_expr(reg["offsets"][0], values),
                               "col": eval_expr(reg["offsets"][1], values),
                               "rows": eval_expr(reg["extents"][0], values),
                               "cols": eval_expr(reg["extents"][1], values)},
                })
            elif name == "tl.tileop.copy":
                src_reg, dst_reg = regions[0], regions[1]
                src, dst = src_reg["buffer"], dst_reg["buffer"]
                source_region = {"row": eval_expr(src_reg["offsets"][0], values),
                                 "col": eval_expr(src_reg["offsets"][1], values),
                                 "rows": eval_expr(src_reg["extents"][0], values),
                                 "cols": eval_expr(src_reg["extents"][1], values)}
                destination_region = {
                    "row": eval_expr(dst_reg["offsets"][0], values),
                    "col": eval_expr(dst_reg["offsets"][1], values),
                    "rows": eval_expr(dst_reg["extents"][0], values),
                    "cols": eval_expr(dst_reg["extents"][1], values)}
                dest_param = next((p for p in params if _same_buffer(p, dst)), None)
                if dest_param is not None:
                    output_name = str(dst.name)
                    if last_gemm_task is not None:
                        last_gemm_task["m0"] = destination_region["row"]
                        last_gemm_task["n0"] = destination_region["col"]
                    tasks.append({"kind": "store", "source": str(src.name),
                                  "destination": str(dst.name),
                                  "source_region": source_region,
                                  "destination_region": destination_region,
                                  "m0": destination_region["row"],
                                  "n0": destination_region["col"],
                                  "rows": destination_region["rows"],
                                  "cols": destination_region["cols"]})
                else:
                    source_param = next((p for p in params
                                         if _same_buffer(p, src)), None)
                    if source_param is not None:
                        copy_sources.append(str(src.name))
                    tasks.append({"kind": "copy", "source": str(src.name),
                                  "destination": str(dst.name),
                                  "source_region": source_region,
                                  "destination_region": destination_region})
            elif name == "tl.tileop.gemm":
                a_reg, b_reg, c_reg = regions[0], regions[1], regions[2]
                gemm_info = {
                    "tiles": [str(a_reg["buffer"].name),
                              str(b_reg["buffer"].name),
                              str(c_reg["buffer"].name)],
                    "m0": eval_expr(a_reg["offsets"][0], values),
                    "n0": eval_expr(b_reg["offsets"][1], values),
                    "rows": eval_expr(c_reg["extents"][0], values),
                    "cols": eval_expr(c_reg["extents"][1], values),
                    "k0": eval_expr(a_reg["offsets"][1], values),
                    "kt": eval_expr(a_reg["extents"][1], values),
                }
                last_gemm_task = {"kind": "gemm", **gemm_info}
                tasks.append(last_gemm_task)
    if gemm_info is None or output_name is None:
        reject(RK_E_SHAPE,
                     "simple staged GEMM has no GEMM/output store",
                     stage="frontend")
    output_value = next(v for v in params if str(v.name) == output_name)
    input_name = copy_sources[0] if copy_sources else ""
    input_value = next(v for v in params if str(v.name) == input_name)
    constants = []
    for name in copy_sources[1:]:
        if name not in constants:
            constants.append(name)
    writes = [t for t in tasks if t.get("kind") == "store"]
    return {
        "index": stage_index, "source_nodes": len(stage_items),
        "operand_a": input_name, "operand_b": constants[0] if constants else "",
        "opcode": "matmul",
        "op_inputs": [input_name] + ([constants[0]] if constants else []),
        "output_name": output_name,
        "logical_m": int(output_value.shape[0]),
        "logical_k": int(input_value.shape[1]),
        "logical_n": int(output_value.shape[1]),
        "tile_m": gemm_info["rows"], "tile_n": gemm_info["cols"],
        "order": "m_major", "tasks": tasks, "global_writes": writes,
        "global_reads": [t for t in tasks if t.get("kind") == "copy"],
        "boundary": "kernel_launch",
    }


launch_map: dict[int, dict[Any, Any]] = {}


def two_stage_chain(prim_func: Any, *, options: Any,
                    carrier: Any = None,
                    schedules: list[Any] | None = None) -> dict[str, Any]:
    """Consume actual launch scopes and lower each one to ordered tasks."""
    from tvm import ir as tvm_ir
    from tvm import tirx as tvm_tirx
    from . import rknpu_pass

    span = _source_span(prim_func)
    params = _param_buffers(prim_func)
    groups = stage_groups(prim_func)
    body = prim_func.body
    if type(body).__name__ == "SBlockRealize":
        body = body.block.body
    launches = list(body.seq)
    if len(groups) != len(launches):
        reject(RK_E_SHAPE,
                     "launch scopes and lowered groups disagree",
                     stage="frontend")
    launch_map.clear()
    for launch in launches:
        used = set()
        for _, node in rknpu_pass._walk(launch):
            if type(node).__name__ in {"BufferLoad", "BufferStore"}:
                used.add(str(node.buffer.name))
        launch_map[id(launch)] = {
            var: buf for var, buf in prim_func.buffer_map.items()
            if str(buf.name) in used}

    attrs_map = dict(getattr(prim_func, "attrs", {}) or {})
    if any(key in attrs_map for key in
           ("rknpu.schedule", "rknpu.tile_m", "rknpu.tile_n")):
        reject(RK_E_UNSUPPORTED,
                     "a single global schedule attribute cannot describe "
                     "multiple launches", stage="frontend")
    for key in attrs_map:
        name = str(key)
        if name.startswith("rknpu.stage"):
            try:
                stage_token = name.split(".")[1]
                stage_index = int(stage_token[len("stage"):])
            except ValueError:
                continue
            if stage_index >= len(launches):
                reject(RK_E_UNSUPPORTED,
                             "a stage attribute is outside the source",
                             stage="frontend", actual=name)
    if schedules is None:
        schedules = [None] * len(launches)
        if carrier is not None and len(carrier.ops) == len(launches):
            schedules = [op.attrs.get("source_schedule")
                         for op in carrier.ops]
    applied_m = [None] * len(launches)
    for index, schedule in enumerate(schedules):
        if not schedule:
            continue
        rounding = schedule.get("source_k_rounding")
        if rounding is not None and rounding != "explicit_per_k_fp16":
            reject(RK_E_NUMERIC_CONTRACT,
                         f"stage{index} rounding declaration does not match "
                         "the proven per-K FP16 source",
                         stage="frontend", actual=rounding)
    if len(launches) == 2 and all(schedules):
        if int(schedules[0].get("tile_n", 0)) != 256:
            reject(RK_E_UNSUPPORTED,
                         "stage0 applied tile_n must be 256",
                         stage="frontend", actual=schedules[0].get("tile_n"))
        applied_m = [int(s.get("tile_m", 0)) for s in schedules]
        if applied_m[0] != applied_m[1]:
            reject(RK_E_UNSUPPORTED,
                         "applied stage0/stage1 M tile must match",
                         stage="frontend", actual=applied_m)
        out_shapes = [_launch_output_shape(launch, params)
                      for launch in launches]
        for i, shape in enumerate(out_shapes):
            if shape is None:
                continue
            logical_n = int(shape[1])
            n_tail = logical_n % int(schedules[i].get("tile_n", logical_n))
            if i == 0 and n_tail and n_tail % 32:
                reject(RK_E_UNSUPPORTED,
                             "stage0 N tail must be 32-element aligned",
                             stage="frontend", actual=n_tail)

    stages = []
    for index, launch in enumerate(launches):
        group = groups[index]
        has_gemm = any(item.kind == "call" and
                       _op_name(item.node) == "tl.tileop.gemm"
                       for item in group)
        if not has_gemm:
            stage = _elementwise_stage_tasks(index, group, params,
                                             tirx=tvm_tirx, span=span)
        else:
            has_serial = any(type(node).__name__ == "For" and
                             node.thread_binding is None and
                             any(_op_name(inner.value) == "tl.tileop.gemm"
                                 for _, inner in rknpu_pass._walk(node)
                                 if type(inner).__name__ == "Evaluate")
                             for _, node in rknpu_pass._walk(launch))
            if not has_serial:
                stage = _simple_gemm_stage_tasks(index, group, params,
                                                 tirx=tvm_tirx, span=span)
            else:
                schedule = schedules[index]
                if schedule:
                    shape = _launch_output_shape(launch, params)
                    if shape is not None:
                        logical_m = int(shape[0])
                        source_m = _source_block_m(
                            launch, logical_m=logical_m,
                            logical_n=int(shape[1]),
                            applied_n=int(schedule.get("tile_n", shape[1])))
                        target_m = int(schedule.get("tile_m", source_m))
                        if target_m != source_m:
                            launch = _retile_launch_source(
                                launch, logical_m=logical_m,
                                old_tile_m=source_m, new_tile_m=target_m)
                buffer_map = launch_map[id(launches[index])]
                sub = tvm_tirx.PrimFunc(
                    list(buffer_map.keys()), launch, prim_func.ret_type,
                    buffer_map, tvm_ir.DictAttrs({}), prim_func.span)
                has_guards = any(type(node).__name__ == "IfThenElse"
                                 for _, node in rknpu_pass._walk(launch))
                if not has_guards:
                    from .full_logical_rewrite import rewrite_k_loop_tir
                    rewritten = rewrite_k_loop_tir(sub, options=options)
                    tasks = [dict(t) for t in rewritten.tasks]
                    a_name = b_name = None
                    for task in tasks:
                        if task.get("kind") == "copy" and a_name is None:
                            a_name = task["source"]
                        elif task.get("kind") == "copy" and b_name is None and                                 task["source"] != a_name:
                            b_name = task["source"]
                    names = [str(buf.name) for buf in buffer_map.values()]
                    output_name = next(name for name in names
                                       if name not in {a_name, b_name})
                    for task in tasks:
                        if task.get("destination") == "Y":
                            task["destination"] = output_name
                        ops = task.get("operands")
                        if isinstance(ops, list) and ops and ops[-1] == "Y":
                            ops[-1] = output_name
                    gemms = [t for t in tasks if t.get("kind") == "gemm"]
                    store = next(t for t in tasks if t.get("kind") == "store")
                    out_buf = next(buf for buf in buffer_map.values()
                                   if str(buf.name) == output_name)
                    a_buf = next(buf for buf in buffer_map.values()
                                 if str(buf.name) == a_name)
                    stage = {
                        "index": index, "source_nodes": len(group),
                        "operand_a": a_name, "operand_b": b_name,
                        "opcode": "matmul",
                        "op_inputs": [a_name, b_name],
                        "output_name": output_name,
                        "logical_m": int(out_buf.shape[0]),
                        "logical_k": int(rewritten.source_k_loop["logical_k"]),
                        "logical_n": int(out_buf.shape[1]),
                        "tile_m": int(gemms[0]["rows"]),
                        "tile_n": int(gemms[0]["cols"]), "order": "m_major",
                        "source_k_loop": {
                            **dict(rewritten.source_k_loop),
                            "narrowing_base":
                                "explicit f16 writeback after each K block"},
                        "tasks": tasks,
                        "global_writes": [dict(t) for t in tasks
                                          if t.get("kind") == "store"],
                        "global_reads": [dict(t) for t in tasks
                                         if t.get("kind") == "copy"],
                        "boundary": "kernel_launch",
                    }
                else:
                    _, _, facts, tasks = rknpu_pass.compile_guarded_tail(
                        sub, options=options, name=f"stage:{index}")
                    stage = {
                        "index": index, "source_nodes": len(group),
                        "operand_a": facts.operand_a,
                        "operand_b": facts.operand_b,
                        "opcode": "matmul",
                        "op_inputs": [facts.operand_a, facts.operand_b],
                        "output_name": facts.output_name,
                        "logical_m": facts.logical_m,
                        "logical_k": facts.logical_k,
                        "logical_n": facts.logical_n,
                        "tile_m": facts.tile_m, "tile_n": facts.tile_n,
                        "order": facts.order,
                        "source_k_loop": {
                            "logical_k": facts.logical_k,
                            "source_k_trips": (facts.logical_k + 255) // 256,
                            "narrowing_base":
                                "explicit f16 writeback after each K block"},
                        "tasks": [dict(t) for t in tasks],
                        "global_writes": [dict(t) for t in tasks
                                          if t.get("kind") == "store"],
                        "global_reads": [dict(t) for t in tasks
                                         if t.get("kind") == "copy"],
                        "boundary": "kernel_launch",
                    }
        stages.append(stage)

    produced = {}
    for stage in stages:
        for write in stage["global_writes"]:
            produced.setdefault(write["destination"], []).append({
                "stage": stage["index"], "source": write.get("source"),
                "m0": write.get("m0", write.get("source_region", {}).get("row", 0)),
                "n0": write.get("n0", write.get("source_region", {}).get("col", 0)),
                "rows": write.get("rows", write.get("source_region", {}).get("rows", 0)),
                "cols": write.get("cols", write.get("source_region", {}).get("cols", 0)),
            })
    relations = []
    for stage in stages:
        for read in stage["global_reads"]:
            name = read["source"]
            if name in produced:
                relations.append({"buffer": name,
                                  "producer_stage": produced[name][0]["stage"],
                                  "consumer_stage": stage["index"],
                                  "producer_regions": produced[name],
                                  "consumer_region": dict(read["source_region"])})
    if "rknpu.staged_schedule" in attrs_map and len(stages) == 2 and             not relations:
        reject(RK_E_UNSUPPORTED,
                     "two-stage schedule needs one distinct intermediate",
                     stage="frontend")
    return {"launch_count": len(stages), "stages": stages,
            "global_relations": relations}


def two_stage_plan_inputs(prim_func: Any, *, options: Any,
                          profile_hash: str = "") -> tuple[Any, dict[str, Any]]:
    """Carrier module plus the actual per-launch task streams."""
    count = len(stage_groups(prim_func))
    schedules = schedules_from_attrs(prim_func, count)
    chain = two_stage_chain(prim_func, options=options,
                            schedules=schedules)
    carrier = carrier_from_chain(prim_func, chain,
                                 profile_hash=profile_hash,
                                 schedules=schedules)
    stage_ops = [op.id for op in carrier.ops if op.opcode != "view"]
    attrs = dict(getattr(prim_func, "attrs", {}) or {})
    scheduled = "rknpu.staged_schedule" in attrs
    facts = {"staged_stages": chain["stages"],
             "staged_launches": chain["launch_count"],
             "staged_global_relations": chain["global_relations"],
             "staged_general": not scheduled,
             "staged_op_ids": stage_ops}
    if scheduled:
        facts["staged_schedule_contract"] = "staged_mn_tiles_v1"
    return carrier, facts



def _launch_output_shape(launch: Any, params: list[Any]):
    from . import rknpu_pass
    for _, node in rknpu_pass._walk(launch):
        buffer = None
        if type(node).__name__ == "BufferStore":
            buffer = node.buffer
        elif type(node).__name__ == "Evaluate" and \
                _op_name(node.value) == "tl.tileop.copy":
            regs = rknpu_pass._region_exprs(node.value)
            buffer = regs[1]["buffer"]
        if buffer is None:
            continue
        for param in params:
            if _same_buffer(param, buffer):
                return tuple(int(v) for v in param.shape)
    return None


def _source_block_m(launch: Any, *, logical_m: int, logical_n: int,
                    applied_n: int) -> int:
    from . import rknpu_pass
    extents = [int(loop.extent) for _, loop in rknpu_pass._walk(launch)
               if type(loop).__name__ == "For" and
               loop.thread_binding is not None]
    n_grid = (logical_n + applied_n - 1) // applied_n
    m_grids = [value for value in extents if value != n_grid]
    if len(m_grids) != 1 or m_grids[0] <= 0:
        reject(RK_E_SHAPE,
                     "cannot derive the source M block grid",
                     stage="frontend", actual=extents)
    return (logical_m + m_grids[0] - 1) // m_grids[0]


def schedules_from_attrs(prim_func: Any, count: int) -> list[Any]:
    attrs = dict(getattr(prim_func, "attrs", {}) or {})
    schedules = []
    found = False
    for index in range(count):
        prefix = f"rknpu.stage{index}."
        if prefix + "tile_m" not in attrs:
            schedules.append(None)
            continue
        found = True
        schedules.append({
            "kind": "mn_tiles_v1",
            "tile_m": int(attrs[prefix + "tile_m"]),
            "tile_n": int(attrs[prefix + "tile_n"]),
            "order": str(attrs.get(prefix + "mn_order", "m_major")),
            "k_tile": 256,
            "accumulation": "ordered_ascending_f16_with_writeback",
            "source_k_rounding": attrs.get(prefix + "source_k_rounding"),
        })
    return schedules if found else [None] * count


def carrier_from_chain(prim_func: Any, chain: dict[str, Any], *,
                       profile_hash: str, name: str = "rknpu-staged",
                       schedules: list[Any] | None = None) -> Any:
    from .. import ir
    params = list((getattr(prim_func, "buffer_map", None) or {}).values())
    param_by_name = {str(buf.name): buf for buf in params}
    stages = chain["stages"]
    written = set()
    read = set()
    for stage in stages:
        for task in stage["tasks"]:
            if task.get("kind") == "store":
                written.add(str(task["destination"]))
            elif task.get("kind") == "copy":
                read.add(str(task["source"]))
        read.update(str(x) for x in (stage.get("op_inputs") or ()))
    constant = {str(stage.get("operand_b")) for stage in stages
                if stage.get("opcode") == "matmul" and
                stage.get("operand_b")}
    output = {name for name in written if name not in read}
    input_names = [name for name in param_by_name
                   if name not in written and name not in constant]
    constant_names = [name for name in param_by_name if name in constant]
    output_names = [name for name in param_by_name if name in output]
    values = []
    by_name = {}
    role_order = [("input", input_names), ("constant", constant_names),
                  ("output", output_names)]
    for role, names in role_order:
        for pname in names:
            buf = param_by_name[pname]
            vid = len(values)
            values.append(ir.Value(vid, pname, ir.TensorType(
                "float16", tuple(int(v) for v in buf.shape)), None, role))
            by_name[pname] = vid
    for pname in param_by_name:
        if pname in by_name:
            continue
        buf = param_by_name[pname]
        vid = len(values)
        values.append(ir.Value(vid, pname, ir.TensorType(
            "float16", tuple(int(v) for v in buf.shape)), None, "temporary"))
        by_name[pname] = vid
    ops = []
    for index, stage in enumerate(stages):
        opcode = str(stage.get("opcode") or "matmul")
        out_id = by_name[str(stage["output_name"])]
        if opcode == "matmul":
            op_inputs = tuple(by_name[name]
                              for name in stage.get("op_inputs") or ())
        else:
            op_inputs_list = []
            offsets = list(stage.get("op_offsets") or
                           [0] * len(stage.get("op_inputs") or ()))
            for operand_index, name in enumerate(stage.get("op_inputs") or ()):
                source_id = by_name[name]
                source_value = values[source_id]
                if source_value.role in {"input", "constant"} and                         source_value.type.shape == values[out_id].type.shape:
                    op_inputs_list.append(source_id)
                    continue
                vid = len(values)
                values.append(ir.Value(vid, f"stage{index}_in{operand_index}",
                                       values[out_id].type, None,
                                       "temporary"))
                view_op = len(ops)
                ops.append(ir.Op(view_op, "view", (source_id,), (vid,), {
                    "row_offset": 0, "col_offset": int(offsets[operand_index]),
                    "rows": values[out_id].type.shape[0],
                    "cols": values[out_id].type.shape[1]}, f"staged:view:{index}:{operand_index}"))
                values[vid] = ir.Value(values[vid].id, values[vid].name,
                                       values[vid].type, view_op, "temporary")
                op_inputs_list.append(vid)
            op_inputs = tuple(op_inputs_list)
        contract = ir.NumericContract(
            accumulator_dtype="float16",
            accumulation="ordered_k_tiles_with_f16_writeback",
            narrowing_points=("k_tile_boundary",),
            approximate=False, tolerance={"atol": 8e-3, "rtol": 8e-3})
        attrs = {"numeric_contract": contract.to_dict(),
                 "source_stage_index": index,
                 "source_stage_count": len(stages),
                 "source_stage_boundary": "kernel_launch",
                 **({"source_stage_schedule_contract":
                     "staged_mn_tiles_v1"}
                    if schedules and any(schedules) else {})}
        if opcode == "matmul":
            attrs.update({"b_layout": "kn",
                          "source_k_rounding": "explicit_per_k_fp16",
                          "source_k_loop": stage.get("source_k_loop")})
            if schedules and index < len(schedules) and schedules[index]:
                attrs["source_schedule"] = dict(schedules[index])
                attrs["source_stage_schedule_contract"] =                     "staged_mn_tiles_v1"
        op_id = len(ops)
        ops.append(ir.Op(op_id, opcode, op_inputs, (out_id,), attrs,
                         f"staged:{index}"))
        values[out_id] = ir.Value(
            values[out_id].id, values[out_id].name, values[out_id].type,
            op_id, values[out_id].role)
    return ir.Module(name=name, values=tuple(values), ops=tuple(ops),
                     inputs=tuple(by_name[name] for name in input_names),
                     outputs=tuple(by_name[name] for name in output_names),
                     profile_hash=profile_hash,
                     source_hash=_source_span(prim_func))


def _retile_launch_source(launch: Any, *, logical_m: int,
                          old_tile_m: int, new_tile_m: int) -> Any:
    """Apply an explicit M tile to the actual guarded launch TIR."""
    if old_tile_m == new_tile_m:
        return launch
    from tvm import tirx as tir
    import tvm.tirx.stmt_functor as stmt_functor
    from . import rknpu_pass

    m_var = None
    for _, node in rknpu_pass._walk(launch):
        if type(node).__name__ == "For" and node.thread_binding is not None:
            extent = rknpu_pass._const(node.extent)
            if extent == (logical_m + old_tile_m - 1) // old_tile_m:
                m_var = node.loop_var
                break
    if m_var is None:
        reject(RK_E_SHAPE,
                     "cannot identify the launch M block variable",
                     stage="frontend")
    replacements = {}
    for _, node in rknpu_pass._walk(launch):
        if type(node).__name__ != "SBlock":
            continue
        for buffer in list(getattr(node, "alloc_buffers", ()) or ()):
            shape = tuple(int(v) for v in buffer.shape)
            if len(shape) == 2 and shape[0] == old_tile_m:
                replacements[buffer] = tir.decl_buffer(
                    (new_tile_m, shape[1]), buffer.dtype,
                    name=str(buffer.name), data=buffer.data,
                    scope=str(buffer.scope()))

    def mapped(buffer):
        for old, new in replacements.items():
            if _same_buffer(old, buffer):
                return new
        return buffer

    new_m_grid = (logical_m + new_tile_m - 1) // new_tile_m

    def rewrite(node):
        kind = type(node).__name__
        if kind == "Mul":
            for var, other in ((node.a, node.b), (node.b, node.a)):
                if type(var).__name__ == "Var" and var.same_as(m_var) and \
                        rknpu_pass._const(other) == old_tile_m:
                    return tir.Mul(m_var, tir.IntImm("int32", new_tile_m))
        if kind == "For":
            extent = node.extent
            if node.loop_var.same_as(m_var):
                extent = tir.IntImm("int32", new_m_grid)
            elif node.thread_binding is None and \
                    rknpu_pass._const(extent) == old_tile_m:
                extent = tir.IntImm("int32", new_tile_m)
            return tir.For(node.loop_var, node.min, extent, node.kind,
                           node.body, node.thread_binding,
                           dict(getattr(node, "annotations", None) or {}),
                           getattr(node, "step", None),
                           getattr(node, "span", None))
        if kind == "BufferLoad":
            buffer = mapped(node.buffer)
            if buffer is node.buffer:
                return None
            return tir.BufferLoad(buffer, list(node.indices))
        if kind == "BufferStore":
            buffer = mapped(node.buffer)
            if buffer is node.buffer:
                return None
            return tir.BufferStore(buffer, node.value, list(node.indices))
        if kind == "Evaluate" and _op_name(node.value).startswith("tl.tileop."):
            call = node.value
            name = _op_name(call)
            args = list(call.args)
            def fix_region(arg):
                if type(arg).__name__ != "Call" or len(arg.args) < 4:
                    return arg
                parts = list(arg.args)
                if type(parts[0]).__name__ == "BufferLoad":
                    parts[0] = tir.BufferLoad(mapped(parts[0].buffer),
                                              list(parts[0].indices))
                ex0 = rknpu_pass._const(parts[2])
                if ex0 == old_tile_m:
                    parts[2] = tir.IntImm("int32", new_tile_m)
                return tir.Call(arg.dtype, arg.op, parts,
                                dict(getattr(arg, "annotations", None) or {}))
            if name == "tl.tileop.fill":
                args[0] = fix_region(args[0])
            elif name == "tl.tileop.copy":
                args[0] = fix_region(args[0])
                args[1] = fix_region(args[1])
            elif name == "tl.tileop.gemm":
                for i in range(3):
                    args[i] = fix_region(args[i])
                if len(args) > 5 and rknpu_pass._const(args[5]) == old_tile_m:
                    args[5] = tir.IntImm("int32", new_tile_m)
            return tir.Evaluate(tir.Call(
                call.dtype, call.op, args,
                dict(getattr(call, "annotations", None) or {})))
        if kind == "SBlock":
            allocs = [mapped(buffer) for buffer in
                      list(getattr(node, "alloc_buffers", ()) or ())]
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

    return stmt_functor.ir_transform(launch, None, rewrite)


def _elementwise_stage_tasks(stage_index: int, stage_items: list[Any],
                            params: list[Any], *, tirx: Any,
                            span: str) -> dict[str, Any]:
    # Proof constrains the accepted source geometry.  Task order and operands
    # below come from each original copy/store, never from a stage summary.
    from . import rknpu_pass
    from .tir_utils import _match_local_elementwise
    import itertools
    import tvm.tirx.stmt_functor as stmt_functor

    proof = _prove_elementwise_stage(stage_index, stage_items, params,
                                     tirx=tirx, span=span)
    rows, cols = proof["tile"]
    blocks = list(proof["store"].blocks)

    def evaluate(expr, env):
        if type(expr) is int:
            return expr
        value = rknpu_pass._const(rknpu_pass._arith().simplify(
            stmt_functor.substitute(expr, env)))
        if value is None:
            reject(RK_E_UNSUPPORTED, "staged copy region is not static",
                   stage="frontend", actual=str(expr))
        return int(value)

    def region(raw, env):
        return dict(zip(("row", "col", "rows", "cols"),
                        [evaluate(x, env) for x in
                         (*raw["offsets"], *raw["extents"])]))

    tasks = []
    op_inputs = input_offsets = None
    for combo in itertools.product(*[range(int(loop.extent)) for loop in blocks]):
        env = {loop.loop_var: tirx.IntImm("int32", index)
               for loop, index in zip(blocks, combo)}
        # Origins are updated at the actual copy, so a store reads the version
        # visible at that point, including a later overwrite of the same tile.
        origins = {}
        established = set()
        pub = rknpu_pass._region_exprs(proof["publish"].node)[1]
        output_region = region(pub, env)
        for item in stage_items:
            if item.kind == "call":
                regs = rknpu_pass._region_exprs(item.node)
                if _op_name(item.node) != "tl.tileop.copy":
                    reject(RK_E_UNSUPPORTED, "unconsumed elementwise call",
                           stage="frontend", actual=str(item.node))
                src, dst = regs
                sn, dn = str(src["buffer"].name), str(dst["buffer"].name)
                sr, dr = region(src, env), region(dst, env)
                src_param = next((p for p in params
                                  if _same_buffer(p, src["buffer"])), None)
                dst_param = next((p for p in params
                                  if _same_buffer(p, dst["buffer"])), None)
                if src_param is None and sn not in established:
                    reject(RK_E_SHAPE, "copy reads an unestablished local value",
                           stage="frontend", actual=sn)
                task = {"kind": "store" if dst_param is not None else "copy",
                        "source": sn, "destination": dn,
                        "source_region": sr, "destination_region": dr}
                if dst_param is not None:
                    task.update(m0=dr["row"], n0=dr["col"],
                                rows=dr["rows"], cols=dr["cols"])
                else:
                    origins[dn] = ((sn, sr["col"] - output_region["col"])
                                   if src_param is not None else origins.get(sn))
                    established.add(dn)
                tasks.append(task)
            elif item.kind == "store":
                opcode, loads = _match_local_elementwise(item.node.value,
                                                        store_cast=True)
                names = [str(load.buffer.name) for load in loads]
                if any(name not in established or origins.get(name) is None
                       for name in names):
                    reject(RK_E_SHAPE, "elementwise reads an unestablished value",
                           stage="frontend", actual=names)
                op_inputs = [origins[name][0] for name in names]
                input_offsets = [origins[name][1] for name in names]
                if opcode == "mul":
                    logical_n = int(params[proof["output"]].shape[1])
                    for operand_index, (name, off) in enumerate(
                            zip(op_inputs, input_offsets)):
                        param = next(p for p in params if str(p.name) == name)
                        if operand_index and int(param.shape[1]) == 2 * logical_n and off == 0:
                            reject(RK_E_UNSUPPORTED,
                                   "packed second multiply operand must read the up half at column H",
                                   stage="frontend")
                local_region = {"row": 0, "col": 0, "rows": rows, "cols": cols}
                destination = str(item.node.buffer.name)
                tasks.append({"kind": opcode, "inputs": names,
                              "input_offsets": list(input_offsets),
                              "input_regions": [dict(local_region) for _ in names],
                              "output": destination, "region": local_region,
                              "m0": output_region["row"], "n0": output_region["col"],
                              "rows": rows, "cols": cols})
                established.add(destination)
                origins[destination] = None
            else:
                reject(RK_E_UNSUPPORTED, "unconsumed staged elementwise effect",
                       stage="frontend", actual=item.kind)
    return {"index": stage_index, "source_nodes": len(stage_items),
            "operand_a": None, "operand_b": None, "opcode": proof["opcode"],
            "op_inputs": op_inputs, "op_offsets": input_offsets,
            "output_name": str(params[proof["output"]].name),
            "logical_m": int(params[proof["output"]].shape[0]), "logical_k": 0,
            "logical_n": int(params[proof["output"]].shape[1]),
            "tile_m": rows, "tile_n": cols, "order": "m_major", "tasks": tasks,
            "global_writes": [t for t in tasks if t["kind"] == "store"],
            "global_reads": [t for t in tasks if t["kind"] == "copy"],
            "boundary": "kernel_launch", "elementwise": True}
