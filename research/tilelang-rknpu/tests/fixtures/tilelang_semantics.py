# SPDX-License-Identifier: GPL-3.0-or-later
"""Semantic-guard fixtures built by TIR surgery on valid kernels.

Each function starts from a real, accepted TileLang program and changes exactly one
structure that the extractor must refuse:

* ``gemm_accum_store_only``  — accumulator region downgraded from load+store to store
* ``predicated_gemm_load``   — a GEMM operand load carries ``predicate=False``
* ``grid_min_one``/``grid_step_two`` — block grid starts at 1 / steps by 2
* ``k_loop_min_one``/``k_loop_step_two`` — K loop starts at 1 / steps by 2
"""
import tilelang
import tilelang.language as T

from tilelang_negative import _rebuild, _with_body


def _tirx():
    from tvm import tirx
    return tirx


def _valid_kernel(m=16, k=256, n=16):
    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"), B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            A_s = T.alloc_shared((m, k), "float16")
            B_s = T.alloc_shared((k, n), "float16")
            C_l = T.alloc_fragment((m, n), "float32")
            T.clear(C_l)
            T.copy(A, A_s)
            T.copy(B, B_s)
            T.gemm(A_s, B_s, C_l)
            T.copy(C_l, C)
    return kernel


def _patch_gemm_arg(base, index, transform):
    tirx = _tirx()

    def mutate(node):
        if isinstance(node, tirx.Call) and str(node.op.name) == "tl.tileop.gemm":
            args = list(node.args)
            args[index] = transform(args[index], tirx)
            return tirx.Call(node.dtype, node.op, args, dict(node.annotations or {}))
        return None

    return _with_body(base, _rebuild(base.body, tirx, mutate))


def gemm_accum_store_only():
    """Accumulator region declares only a store (mask 2) instead of load+store."""
    def change(region, tirx):
        args = list(region.args)
        args[1] = tirx.const(2, "int32")
        return tirx.Call(region.dtype, region.op, args,
                         dict(region.annotations or {}))
    return _patch_gemm_arg(_valid_kernel(), 2, change)


def predicated_gemm_load():
    """The A operand load is guarded by ``predicate=False``."""
    def change(region, tirx):
        args = list(region.args)
        load = args[0]
        args[0] = tirx.BufferLoad(load.buffer, list(load.indices),
                                  tirx.const(False, "bool"))
        return tirx.Call(region.dtype, region.op, args,
                         dict(region.annotations or {}))
    return _patch_gemm_arg(_valid_kernel(), 0, change)


def _collect_for_nodes(node, tirx, out):
    if isinstance(node, tirx.For):
        out.append(node)
        _collect_for_nodes(node.body, tirx, out)
        return
    if isinstance(node, tirx.SBlockRealize):
        _collect_for_nodes(node.block, tirx, out)
        return
    if isinstance(node, tirx.SBlock):
        _collect_for_nodes(node.body, tirx, out)
        return
    if isinstance(node, tirx.SeqStmt):
        for stmt in node.seq:
            _collect_for_nodes(stmt, tirx, out)
        return
    if isinstance(node, tirx.Evaluate):
        _collect_for_nodes(node.value, tirx, out)
        return


def _patch_for(base, predicate, *, first=True):
    """Rewrite the outer- or innermost ``For``, matched by its loop variable."""
    tirx = _tirx()
    loops: list = []
    _collect_for_nodes(base.body, tirx, loops)
    if not loops:
        raise RuntimeError("fixture has no For node to patch")
    target = loops[0] if first else loops[-1]
    state = {"done": False}

    def mutate(node):
        if isinstance(node, tirx.For) and not state["done"] and \
                node.loop_var.same_as(target.loop_var) and \
                node.extent.same_as(target.extent):
            state["done"] = True
            return predicate(node, tirx)
        return None

    return _with_body(base, _rebuild(base.body, tirx, mutate))


def grid_min_one():
    def change(node, tirx):
        return tirx.For(node.loop_var, tirx.const(1, "int32"), node.extent,
                        node.kind, _rebuild(node.body, tirx, lambda n: None),
                        node.thread_binding, dict(node.annotations))
    return _patch_for(_valid_kernel(), change, first=True)


def grid_step_two():
    def change(node, tirx):
        return tirx.For(node.loop_var, node.min, node.extent, node.kind,
                        _rebuild(node.body, tirx, lambda n: None),
                        node.thread_binding, dict(node.annotations),
                        tirx.const(2, "int32"))
    return _patch_for(_valid_kernel(), change, first=True)


def _valid_k_loop(m=16, n=16, k=512, bk=256):
    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"), B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            A_s = T.alloc_shared((m, bk), "float16")
            B_s = T.alloc_shared((bk, n), "float16")
            C_l = T.alloc_fragment((m, n), "float32")
            T.clear(C_l)
            for ko in T.Pipelined(k // bk, num_stages=2):
                T.copy(A[0, ko * bk], A_s)
                T.copy(B[ko * bk, 0], B_s)
                T.gemm(A_s, B_s, C_l)
            T.copy(C_l, C)
    return kernel


def k_loop_min_one():
    base = _valid_k_loop()
    tirx = _tirx()
    state = {"done": False}

    def mutate(node):
        if isinstance(node, tirx.For) and node.thread_binding is None and \
                not state["done"]:
            state["done"] = True
            return tirx.For(node.loop_var, tirx.const(1, "int32"), node.extent,
                            node.kind, _rebuild(node.body, tirx, mutate),
                            node.thread_binding, dict(node.annotations))
        return None

    return _with_body(base, _rebuild(base.body, tirx, mutate))


def k_loop_step_two():
    base = _valid_k_loop()
    tirx = _tirx()
    state = {"done": False}

    def mutate(node):
        if isinstance(node, tirx.For) and node.thread_binding is None and \
                not state["done"]:
            state["done"] = True
            return tirx.For(node.loop_var, node.min, node.extent, node.kind,
                            _rebuild(node.body, tirx, mutate),
                            node.thread_binding, dict(node.annotations),
                            tirx.const(2, "int32"))
        return None

    return _with_body(base, _rebuild(base.body, tirx, mutate))


def unhandled_call_inside_kernel():
    """A side-effecting intrinsic call inside the kernel body."""
    @T.prim_func
    def kernel(A: T.Tensor((16, 256), "float16"),
               B: T.Tensor((256, 16), "float16"),
               C: T.Tensor((16, 16), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            A_s = T.alloc_shared((16, 256), "float16")
            B_s = T.alloc_shared((256, 16), "float16")
            C_l = T.alloc_fragment((16, 16), "float32")
            T.clear(C_l)
            T.copy(A, A_s)
            T.copy(B, B_s)
            T.gemm(A_s, B_s, C_l)
            T.call_extern("handle", "some_side_effect", 1)
            T.copy(C_l, C)
    return kernel


def _patch_store_value(base, transform):
    tirx = _tirx()
    state = {"done": False}

    def mutate(node):
        if isinstance(node, tirx.BufferStore):
            state["done"] = True
            return tirx.BufferStore(node.buffer, transform(node.value, tirx),
                                    node.indices, node.predicate)
        return None

    return _with_body(base, _rebuild(base.body, tirx, mutate))


def _silu_kernel(m=16, k=256, n=16, loop_step=None):
    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"), B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            A_s = T.alloc_shared((m, k), "float16")
            B_s = T.alloc_shared((k, n), "float16")
            C_l = T.alloc_fragment((m, n), "float32")
            T.clear(C_l)
            T.copy(A, A_s)
            T.copy(B, B_s)
            T.gemm(A_s, B_s, C_l)
            for i, j in T.Parallel(m, n):
                C_l[i, j] = C_l[i, j] * T.sigmoid(C_l[i, j])
            T.copy(C_l, C)
    return kernel


def silu_inner_load_predicated():
    """The sigmoid argument load carries predicate=False."""
    base = _silu_kernel()
    tirx = _tirx()

    def rewrite(expr):
        if isinstance(expr, tirx.Call) and str(expr.op.name) == "tirx.sigmoid":
            arg = expr.args[0]
            guarded = tirx.BufferLoad(arg.buffer, list(arg.indices),
                                      tirx.const(False, "bool"))
            return tirx.Call(expr.dtype, expr.op, [guarded],
                             dict(expr.annotations or {}))
        if type(expr).__name__ == "Mul":
            return tirx.Mul(rewrite(expr.a), rewrite(expr.b))
        return expr

    def mutate(node):
        if isinstance(node, tirx.BufferStore):
            return tirx.BufferStore(node.buffer, rewrite(node.value),
                                    list(node.indices), node.predicate)
        return None

    return _with_body(base, _rebuild(base.body, tirx, mutate))


def _rewrite_calls(node, tirx, mutate):
    replacement = mutate(node)
    if replacement is not None:
        return replacement
    if isinstance(node, tirx.Call):
        args = [_rewrite_calls(a, tirx, mutate) for a in node.args]
        return tirx.Call(node.dtype, node.op, args, dict(node.annotations or {}))
    return node


def activation_loop_step_two():
    """The activation Parallel loops advance by two elements."""
    base = _silu_kernel()
    tirx = _tirx()
    loops = []
    _collect_for_nodes(base.body, tirx, loops)
    parallel = [loop for loop in loops if int(loop.kind) == 1]
    target = parallel[0].loop_var

    def mutate(node):
        if isinstance(node, tirx.For) and node.loop_var.same_as(target):
            return tirx.For(node.loop_var, node.min, node.extent, node.kind,
                            _rebuild(node.body, tirx, mutate),
                            node.thread_binding, dict(node.annotations),
                            tirx.const(2, "int32"))
        return None

    return _with_body(base, _rebuild(base.body, tirx, mutate))
