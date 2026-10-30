# SPDX-License-Identifier: GPL-3.0-or-later
"""Real TileLang M/N-tiled GEMM fixtures (P1 batch 2).

``mn_tiled`` is the canonical multi-block kernel::

    with T.Kernel(N // BN, M // BM, threads=128) as (bx, by):
        for ko in T.Pipelined(K // BK):
            copy(A[by*BM, ko*BK], A_shared)
            copy(B[ko*BK, bx*BN], B_shared)
            gemm(A_shared, B_shared, C_local)
        copy(C_local, C[by*BM, bx*BN])

Every ``bad_*`` function breaks exactly one coverage/index property.
"""
import tilelang
import tilelang.language as T


def mn_tiled(m=512, n=384, k=512, bm=256, bn=128, bk=256, grid_y=None,
             order="m_major"):
    gy = grid_y if grid_y is not None else m // bm
    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"), B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        T.func_attr({"rknpu.schedule": "mn_tiles_v1",
                     "rknpu.mn_order": order})
        with T.Kernel(T.ceildiv(n, bn), gy, threads=128) as (bx, by):
            A_shared = T.alloc_shared((bm, bk), "float16")
            B_shared = T.alloc_shared((bk, bn), "float16")
            C_local = T.alloc_fragment((bm, bn), "float32")
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(k, bk), num_stages=2):
                T.copy(A[by * bm, ko * bk], A_shared)
                T.copy(B[ko * bk, bx * bn], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * bm, bx * bn])
    return kernel


def mn_tiled_n_only(m=256, n=512, k=512, bm=256, bn=256, bk=256):
    """One M tile, several N tiles: a 1-D grid of extent N/BN."""
    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"), B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(T.ceildiv(n, bn), threads=128) as bx:
            A_shared = T.alloc_shared((bm, bk), "float16")
            B_shared = T.alloc_shared((bk, bn), "float16")
            C_local = T.alloc_fragment((bm, bn), "float32")
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(k, bk), num_stages=2):
                T.copy(A[0, ko * bk], A_shared)
                T.copy(B[ko * bk, bx * bn], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[0, bx * bn])
    return kernel


def bad_ragged_output(m=512, n=384, k=512, bm=256, bn=256, bk=256):
    """grid*tile overshoots N: the last tile would write past the output."""
    return mn_tiled(m=m, n=n, k=k, bm=bm, bn=bn, bk=bk)


def bad_block_var_on_wrong_axis(m=512, n=384, k=512, bm=256, bn=128, bk=256):
    """A advances with bx instead of by: blocks read the wrong rows."""
    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"), B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(T.ceildiv(n, bn), m // bm, threads=128) as (bx, by):
            A_shared = T.alloc_shared((bm, bk), "float16")
            B_shared = T.alloc_shared((bk, bn), "float16")
            C_local = T.alloc_fragment((bm, bn), "float32")
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(k, bk), num_stages=2):
                T.copy(A[bx * bm, ko * bk], A_shared)
                T.copy(B[ko * bk, bx * bn], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * bm, bx * bn])
    return kernel


def bad_output_tile_swapped(m=512, n=384, k=512, bm=256, bn=128, bk=256):
    """The result is stored at (bx*BM, by*BN): tiles overlap/gap."""
    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"), B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(T.ceildiv(n, bn), m // bm, threads=128) as (bx, by):
            A_shared = T.alloc_shared((bm, bk), "float16")
            B_shared = T.alloc_shared((bk, bn), "float16")
            C_local = T.alloc_fragment((bm, bn), "float32")
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(k, bk), num_stages=2):
                T.copy(A[by * bm, ko * bk], A_shared)
                T.copy(B[ko * bk, bx * bn], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[bx * bm, by * bn])
    return kernel


def bad_three_dim_grid(m=512, n=384, k=512, bm=256, bn=128, bk=256):
    """A third grid dimension is not part of the supported tiling."""
    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"), B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(1, T.ceildiv(n, bn), m // bm, threads=128) as (bz, bx, by):
            A_shared = T.alloc_shared((bm, bk), "float16")
            B_shared = T.alloc_shared((bk, bn), "float16")
            C_local = T.alloc_fragment((bm, bn), "float32")
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(k, bk), num_stages=2):
                T.copy(A[by * bm, ko * bk], A_shared)
                T.copy(B[ko * bk, bx * bn], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * bm, bx * bn])
    return kernel


def mn_tiled_m_only(m=512, n=256, k=512, bm=256, bn=256, bk=256):
    """One N tile, several M tiles: a 1-D grid that covers the M axis."""
    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"), B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(T.ceildiv(m, bm), threads=128) as bx:
            A_shared = T.alloc_shared((bm, bk), "float16")
            B_shared = T.alloc_shared((bk, bn), "float16")
            C_local = T.alloc_fragment((bm, bn), "float32")
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(k, bk), num_stages=2):
                T.copy(A[bx * bm, ko * bk], A_shared)
                T.copy(B[ko * bk, 0], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[bx * bm, 0])
    return kernel


def bad_single_grid_axis_ambiguous(m=512, n=512, k=512, bm=256, bn=256, bk=256):
    """The single block variable indexes both output axes: ambiguous."""
    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"), B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(T.ceildiv(n, bn), threads=128) as bx:
            A_shared = T.alloc_shared((bm, bk), "float16")
            B_shared = T.alloc_shared((bk, bn), "float16")
            C_local = T.alloc_fragment((bm, bn), "float32")
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(k, bk), num_stages=2):
                T.copy(A[bx * bm, ko * bk], A_shared)
                T.copy(B[ko * bk, bx * bn], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[bx * bm, bx * bn])
    return kernel
