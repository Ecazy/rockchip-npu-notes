# SPDX-License-Identifier: GPL-3.0-or-later
"""Real TileLang K-loop fixtures for the restricted extractor.

``k_loop`` is the canonical blocked GEMM:

    C_local = 0
    for ko in T.Pipelined(K // BK):
        copy(A[0, ko*BK], A_shared)
        copy(B[ko*BK, 0], B_shared)
        gemm(A_shared, B_shared, C_local)
    copy(C_local, C)

Every ``bad_*`` function is the same program with one semantic property broken,
so each negative exercises exactly one extractor rule.
"""
import tilelang
import tilelang.language as T


def k_loop(m=16, n=16, k=512, bk=256, num_stages=2, serial=False):
    """Canonical blocked GEMM.

    TileLang's eager builder needs the loop frame written inline, so the serial
    and pipelined variants are two separate prim_func bodies.
    """
    if serial:
        @T.prim_func
        def kernel(A: T.Tensor((m, k), "float16"),
                   B: T.Tensor((k, n), "float16"),
                   C: T.Tensor((m, n), "float16")):
            with T.Kernel(1, threads=128) as _bx:
                A_shared = T.alloc_shared((m, bk), "float16")
                B_shared = T.alloc_shared((bk, n), "float16")
                C_local = T.alloc_fragment((m, n), "float32")
                T.clear(C_local)
                for ko in T.serial(k // bk):
                    T.copy(A[0, ko * bk], A_shared)
                    T.copy(B[ko * bk, 0], B_shared)
                    T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, C)
        return kernel

    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"), B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            A_shared = T.alloc_shared((m, bk), "float16")
            B_shared = T.alloc_shared((bk, n), "float16")
            C_local = T.alloc_fragment((m, n), "float32")
            T.clear(C_local)
            for ko in T.Pipelined(k // bk, num_stages=num_stages):
                T.copy(A[0, ko * bk], A_shared)
                T.copy(B[ko * bk, 0], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C)
    return kernel


def bad_both_k_axes_shifted(m=16, n=16, k=512, bk=256):
    """The A slice advances on the M axis instead of the K axis."""
    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"), B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            A_shared = T.alloc_shared((m, bk), "float16")
            B_shared = T.alloc_shared((bk, n), "float16")
            C_local = T.alloc_fragment((m, n), "float32")
            T.clear(C_local)
            for ko in T.Pipelined(k // bk, num_stages=2):
                # wrong: rows advance with the K loop, columns stay at 0
                T.copy(A[ko * 0, ko * 0], A_shared)
                T.copy(B[ko * bk, 0], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C)
    return kernel


def bad_ragged_k(m=16, n=16, k=300, bk=256):
    """The loop does not tile K exactly (300 = 2*128 + 44)."""
    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"), B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            A_shared = T.alloc_shared((m, bk), "float16")
            B_shared = T.alloc_shared((bk, n), "float16")
            C_local = T.alloc_fragment((m, n), "float32")
            T.clear(C_local)
            for ko in T.Pipelined(k // bk, num_stages=2):
                T.copy(A[0, ko * bk], A_shared)
                T.copy(B[ko * bk, 0], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C)
    return kernel


def bad_clear_inside_loop(m=16, n=16, k=512, bk=256):
    """The accumulator is cleared every iteration: only the last slice survives."""
    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"), B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            A_shared = T.alloc_shared((m, bk), "float16")
            B_shared = T.alloc_shared((bk, n), "float16")
            C_local = T.alloc_fragment((m, n), "float32")
            for ko in T.Pipelined(k // bk, num_stages=2):
                T.clear(C_local)
                T.copy(A[0, ko * bk], A_shared)
                T.copy(B[ko * bk, 0], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C)
    return kernel


def bad_two_gemms(m=16, n=16, k=512, bk=256):
    """Two GEMMs per iteration double-count the slice."""
    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"), B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            A_shared = T.alloc_shared((m, bk), "float16")
            B_shared = T.alloc_shared((bk, n), "float16")
            C_local = T.alloc_fragment((m, n), "float32")
            T.clear(C_local)
            for ko in T.Pipelined(k // bk, num_stages=2):
                T.copy(A[0, ko * bk], A_shared)
                T.copy(B[ko * bk, 0], B_shared)
                T.gemm(A_shared, B_shared, C_local)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C)
    return kernel


def bad_activation_in_loop(m=16, n=16, k=512, bk=256):
    """Element-wise work inside the reduction loop is not lowered."""
    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"), B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            A_shared = T.alloc_shared((m, bk), "float16")
            B_shared = T.alloc_shared((bk, n), "float16")
            C_local = T.alloc_fragment((m, n), "float32")
            T.clear(C_local)
            for ko in T.Pipelined(k // bk, num_stages=2):
                T.copy(A[0, ko * bk], A_shared)
                T.copy(B[ko * bk, 0], B_shared)
                T.gemm(A_shared, B_shared, C_local)
                for i, j in T.Parallel(m, n):
                    C_local[i, j] = C_local[i, j] * T.sigmoid(C_local[i, j])
            T.copy(C_local, C)
    return kernel


def bad_parallel_loop(m=16, n=16, k=512, bk=256):
    """A parallel (element) loop around the GEMM changes the reduction order."""
    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"), B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            A_shared = T.alloc_shared((m, bk), "float16")
            B_shared = T.alloc_shared((bk, n), "float16")
            C_local = T.alloc_fragment((m, n), "float32")
            T.clear(C_local)
            for ko in T.Parallel(k // bk):
                T.copy(A[0, ko * bk], A_shared)
                T.copy(B[ko * bk, 0], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C)
    return kernel


def bad_copy_from_shared(m=16, n=16, k=512, bk=256):
    """The GEMM inputs are copied from a tile rather than the parameters."""
    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"), B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            A_shared = T.alloc_shared((m, bk), "float16")
            B_shared = T.alloc_shared((bk, n), "float16")
            A_other = T.alloc_shared((m, bk), "float16")
            C_local = T.alloc_fragment((m, n), "float32")
            T.clear(C_local)
            for ko in T.Pipelined(k // bk, num_stages=2):
                # exactly two in-loop copies, but A comes from a tile
                T.copy(A_other, A_shared)
                T.copy(B[ko * bk, 0], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C)
    return kernel


def bad_output_copy_inside_loop(m=16, n=16, k=512, bk=256):
    """Writing C inside the loop publishes partial accumulations."""
    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"), B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            A_shared = T.alloc_shared((m, bk), "float16")
            B_shared = T.alloc_shared((bk, n), "float16")
            C_local = T.alloc_fragment((m, n), "float32")
            T.clear(C_local)
            for ko in T.Pipelined(k // bk, num_stages=2):
                T.copy(A[0, ko * bk], A_shared)
                T.copy(B[ko * bk, 0], B_shared)
                T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, C)
    return kernel


def bad_gemm_before_copies(m=16, n=16, k=512, bk=256):
    """The GEMM runs on tiles that the loop fills only afterwards."""
    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"), B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            A_shared = T.alloc_shared((m, bk), "float16")
            B_shared = T.alloc_shared((bk, n), "float16")
            C_local = T.alloc_fragment((m, n), "float32")
            T.clear(C_local)
            for ko in T.Pipelined(k // bk, num_stages=2):
                T.gemm(A_shared, B_shared, C_local)
                T.copy(A[0, ko * bk], A_shared)
                T.copy(B[ko * bk, 0], B_shared)
            T.copy(C_local, C)
    return kernel


def bad_gemm_between_copies(m=16, n=16, k=512, bk=256):
    """A is copied, then the GEMM runs before B has been copied."""
    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"), B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            A_shared = T.alloc_shared((m, bk), "float16")
            B_shared = T.alloc_shared((bk, n), "float16")
            C_local = T.alloc_fragment((m, n), "float32")
            T.clear(C_local)
            for ko in T.Pipelined(k // bk, num_stages=2):
                T.copy(A[0, ko * bk], A_shared)
                T.gemm(A_shared, B_shared, C_local)
                T.copy(B[ko * bk, 0], B_shared)
            T.copy(C_local, C)
    return kernel


def bad_block_k_128(m=16, n=16, k=512, bk=128):
    """A 128-wide source K block implies a different reduction/narrowing order."""
    return k_loop(m=m, n=n, k=k, bk=bk)


def bad_extra_gemm_outside_loop(m=16, n=16, k=512, bk=256):
    """A second GEMM after the loop would be silently dropped by the lowering."""
    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"), B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            A_shared = T.alloc_shared((m, bk), "float16")
            B_shared = T.alloc_shared((bk, n), "float16")
            C_local = T.alloc_fragment((m, n), "float32")
            T.clear(C_local)
            for ko in T.Pipelined(k // bk, num_stages=2):
                T.copy(A[0, ko * bk], A_shared)
                T.copy(B[ko * bk, 0], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C)
    return kernel


def bad_extra_copy_outside_loop(m=16, n=16, k=512, bk=256):
    """An extra tile-to-tile copy outside the loop is not part of the pattern."""
    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"), B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            A_shared = T.alloc_shared((m, bk), "float16")
            B_shared = T.alloc_shared((bk, n), "float16")
            A_copy = T.alloc_shared((m, bk), "float16")
            C_local = T.alloc_fragment((m, n), "float32")
            T.clear(C_local)
            for ko in T.Pipelined(k // bk, num_stages=2):
                T.copy(A[0, ko * bk], A_shared)
                T.copy(B[ko * bk, 0], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(A_shared, A_copy)
            T.copy(C_local, C)
    return kernel


def bad_unknown_tileop_in_loop(m=16, n=16, k=512, bk=256):
    """An unregistered tileop inside the K loop must be rejected.

    ``T.copy`` has a registered name, so use ``T.transpose``, which lowers to
    ``tl.tileop.transpose``: a tileop the extractor does not accept.
    """
    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"), B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            A_shared = T.alloc_shared((m, bk), "float16")
            B_shared = T.alloc_shared((bk, n), "float16")
            A_t = T.alloc_shared((bk, m), "float16")
            C_local = T.alloc_fragment((m, n), "float32")
            T.clear(C_local)
            for ko in T.Pipelined(k // bk, num_stages=2):
                T.copy(A[0, ko * bk], A_shared)
                T.transpose(A_shared, A_t)
                T.copy(B[ko * bk, 0], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C)
    return kernel


def bad_clear_accum_in_loop(m=16, n=16, k=512, bk=256):
    """clear_accum=True inside the K loop would discard earlier K slices."""
    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"), B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            A_shared = T.alloc_shared((m, bk), "float16")
            B_shared = T.alloc_shared((bk, n), "float16")
            C_local = T.alloc_fragment((m, n), "float32")
            T.clear(C_local)
            for ko in T.Pipelined(k // bk, num_stages=2):
                T.copy(A[0, ko * bk], A_shared)
                T.copy(B[ko * bk, 0], B_shared)
                T.gemm(A_shared, B_shared, C_local, clear_accum=True)
            T.copy(C_local, C)
    return kernel
