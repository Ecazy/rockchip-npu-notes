# SPDX-License-Identifier: GPL-3.0-or-later
"""Real TileLang programs used as extractor fixtures.

Positive cases live in ``examples/``.  This module holds the adjacent negative
cases the structural extractor must reject, plus a square-shape case that must now
be accepted because parameter roles come from the copy data flow rather than from
guessed shapes.  Importing this module requires the pinned TileLang wheel; the
tests skip when it is absent.
"""
import tilelang
import tilelang.language as T


def square_matmul(m=16, k=16, n=16):
    """Square A/B/C: shapes cannot disambiguate roles, the copies must."""

    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"),
               B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            A_shared = T.alloc_shared((m, k), "float16")
            B_shared = T.alloc_shared((k, n), "float16")
            C_local = T.alloc_fragment((m, n), "float32")
            T.clear(C_local)
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C)

    return kernel


def nonzero_fill(m=16, k=16, n=16):
    """Accumulator starts at one: this is a non-zero GEMM initial value."""

    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"),
               B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            A_shared = T.alloc_shared((m, k), "float16")
            B_shared = T.alloc_shared((k, n), "float16")
            C_local = T.alloc_fragment((m, n), "float32")
            T.fill(C_local, 1)
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C)

    return kernel


def extra_copy(m=16, k=16, n=16):
    """A fourth tile copy: not part of the recognised closed pattern."""

    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"),
               B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            A_shared = T.alloc_shared((m, k), "float16")
            B_shared = T.alloc_shared((k, n), "float16")
            C_local = T.alloc_fragment((m, n), "float32")
            T.clear(C_local)
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C)
            T.copy(C_local, C)

    return kernel


def wrong_activation_index(m=16, k=16, n=16):
    """Square tile where the sigmoid reads the transposed index."""

    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"),
               B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            A_shared = T.alloc_shared((m, k), "float16")
            B_shared = T.alloc_shared((k, n), "float16")
            C_local = T.alloc_fragment((m, n), "float32")
            T.clear(C_local)
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.gemm(A_shared, B_shared, C_local)
            for i, j in T.Parallel(m, n):
                C_local[i, j] = C_local[i, j] * T.sigmoid(C_local[j, i])
            T.copy(C_local, C)

    return kernel


def incorrect_silu_exponent(m=16, k=16, n=16):
    """``z * exp(-z)`` is not SiLU and must not be matched as one."""

    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"),
               B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            A_shared = T.alloc_shared((m, k), "float16")
            B_shared = T.alloc_shared((k, n), "float16")
            C_local = T.alloc_fragment((m, n), "float32")
            T.clear(C_local)
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.gemm(A_shared, B_shared, C_local)
            for i, j in T.Parallel(m, n):
                C_local[i, j] = C_local[i, j] * T.exp(-C_local[i, j])
            T.copy(C_local, C)

    return kernel


def transposed_gemm(m=16, k=16, n=16):
    """trans_B=True: the MVP has no transpose packing rule for this shape."""

    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"),
               B: T.Tensor((n, k), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            A_shared = T.alloc_shared((m, k), "float16")
            B_shared = T.alloc_shared((n, k), "float16")
            C_local = T.alloc_fragment((m, n), "float32")
            T.clear(C_local)
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C)

    return kernel


def looped_gemm(m=16, k=16, n=16):
    """The GEMM runs twice inside a serial loop: two accumulations, one task."""
    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"),
               B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            A_shared = T.alloc_shared((m, k), "float16")
            B_shared = T.alloc_shared((k, n), "float16")
            C_local = T.alloc_fragment((m, n), "float32")
            T.clear(C_local)
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            for _rep in T.serial(2):
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C)
    return kernel


def _rebuild(node, tirx, mutate):
    """Rebuild a TIR statement tree, applying ``mutate`` to each statement."""
    replacement = mutate(node)
    if replacement is not None:
        return replacement
    if isinstance(node, tirx.SBlockRealize):
        return tirx.SBlockRealize(list(node.iter_values), node.predicate,
                                  _rebuild(node.block, tirx, mutate))
    if isinstance(node, tirx.SBlock):
        body = _rebuild(node.body, tirx, mutate)
        init = (_rebuild(node.init, tirx, mutate)
                if node.init is not None else None)
        return tirx.SBlock(iter_vars=list(node.iter_vars),
                           reads=list(node.reads), writes=list(node.writes),
                           name_hint=node.name_hint, body=body, init=init,
                           alloc_buffers=list(node.alloc_buffers),
                           match_buffers=list(node.match_buffers),
                           annotations=dict(node.annotations))
    if isinstance(node, tirx.SeqStmt):
        return tirx.SeqStmt([_rebuild(x, tirx, mutate) for x in node.seq])
    if isinstance(node, tirx.For):
        return tirx.For(node.loop_var, node.min, node.extent, node.kind,
                        _rebuild(node.body, tirx, mutate), node.thread_binding,
                        dict(node.annotations))
    if isinstance(node, tirx.Evaluate):
        return tirx.Evaluate(_rebuild(node.value, tirx, mutate))
    if isinstance(node, tirx.BufferStore):
        return tirx.BufferStore(node.buffer, _rebuild(node.value, tirx, mutate),
                                list(node.indices), node.predicate)
    return node


def _with_body(prim_func, body):
    from tvm import tirx
    return tirx.PrimFunc(prim_func.params, body, prim_func.ret_type,
                         prim_func.buffer_map, prim_func.attrs, prim_func.span)


def _with_predicate(base, predicate_fn, *, nested_only=False, tirx):
    """Rebuild ``base`` replacing the chosen SBlockRealize predicate."""
    seen = {"count": 0}

    def mutate(node):
        if isinstance(node, tirx.SBlockRealize):
            index = seen["count"]
            seen["count"] += 1
            new_predicate = predicate_fn(index, node.predicate)
            return tirx.SBlockRealize(
                list(node.iter_values),
                node.predicate if new_predicate is None else new_predicate,
                _rebuild(node.block, tirx, mutate))
        return None

    return _with_body(base, _rebuild(base.body, tirx, mutate))


def _body(base):
    from tvm import tirx
    body = base.body
    while isinstance(body, (tirx.SBlockRealize,)):
        body = body.block
    return body


def false_predicate(m=16, k=16, n=16):
    """The whole kernel body is guarded by a predicate that is always false.

    The source program performs no work, so emitting a GEMM task would execute
    computation the program never does.
    """
    from tvm import tirx
    base = square_matmul(m, k, n)
    return _with_predicate(
        base, lambda index, old: tirx.const(False, "bool") if index == 0 else None,
        tirx=tirx)


def variable_predicate(m=16, k=16, n=16):
    """Predicate is a free boolean variable: true or false at run time."""
    from tvm import tirx
    base = square_matmul(m, k, n)
    guard = tirx.Var("guard", "bool")

    def choose(index, old):
        return guard if index == 0 else None

    return _with_predicate(base, choose, tirx=tirx)


def compound_predicate(m=16, k=16, n=16):
    """Predicate is ``True and True``: constant, but not a literal IntImm."""
    from tvm import tirx
    base = square_matmul(m, k, n)
    compound = tirx.And(tirx.const(True, "bool"), tirx.const(True, "bool"))

    def choose(index, old):
        return compound if index == 0 else None

    return _with_predicate(base, choose, tirx=tirx)


def nested_guarded_block(m=16, k=16, n=16):
    """The inner kernel block is guarded false while the outer one is true."""
    from tvm import tirx
    base = square_matmul(m, k, n)

    def choose(index, old):
        return tirx.const(False, "bool") if index == 1 else None

    return _with_predicate(base, choose, tirx=tirx)


def _shift_activation_loop(m, n, *, which, new_min):
    """Move one activation loop's lower bound away from zero."""
    from tvm import tirx
    base = silu_matmul(m, k=16, n=n)
    seen = {"par": 0}

    def mutate(node):
        if isinstance(node, tirx.For) and node.kind == tirx.ForKind.PARALLEL:
            index = seen["par"]
            seen["par"] += 1
            if index == which:
                return tirx.For(node.loop_var, tirx.const(new_min, "int32"),
                                node.extent, node.kind,
                                _rebuild(node.body, tirx, mutate),
                                node.thread_binding, dict(node.annotations))
        return None

    return _with_body(base, _rebuild(base.body, tirx, mutate))


def shifted_outer_activation_loop(m=8, n=12):
    """Outer SiLU loop starts at 1: row 0 is skipped."""
    return _shift_activation_loop(m, n, which=0, new_min=1)


def shifted_inner_activation_loop(m=8, n=12):
    """Inner SiLU loop starts at 1: column 0 is skipped."""
    return _shift_activation_loop(m, n, which=1, new_min=1)


def negative_activation_loop(m=8, n=12):
    """Outer SiLU loop starts at -1: reads one row before the tile."""
    return _shift_activation_loop(m, n, which=0, new_min=-1)


def multi_block_launch(m=16, k=16, n=16):
    """Two blocks each computing the same output tile is not lowered."""
    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"),
               B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(2, threads=128) as _bx:
            A_shared = T.alloc_shared((m, k), "float16")
            B_shared = T.alloc_shared((k, n), "float16")
            C_local = T.alloc_fragment((m, n), "float32")
            T.clear(C_local)
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C)
    return kernel


def silu_matmul(m=16, k=16, n=16):
    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"),
               B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            A_shared = T.alloc_shared((m, k), "float16")
            B_shared = T.alloc_shared((k, n), "float16")
            C_local = T.alloc_fragment((m, n), "float32")
            T.clear(C_local)
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.gemm(A_shared, B_shared, C_local)
            for i, j in T.Parallel(m, n):
                C_local[i, j] = C_local[i, j] * T.sigmoid(C_local[i, j])
            T.copy(C_local, C)
    return kernel


def clear_accum_matmul(m=16, k=16, n=16):
    """The GEMM clears its accumulator: a zero fill would be pointless here and
    the K accumulation semantics would change."""
    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"),
               B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            A_shared = T.alloc_shared((m, k), "float16")
            B_shared = T.alloc_shared((k, n), "float16")
            C_local = T.alloc_fragment((m, n), "float32")
            T.clear(C_local)
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.gemm(A_shared, B_shared, C_local, clear_accum=True)
            T.copy(C_local, C)
    return kernel
