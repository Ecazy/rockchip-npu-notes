# SPDX-License-Identifier: GPL-3.0-or-later
"""One-tile real TileLang matrix primitive used by the hybrid Transformer."""
from __future__ import annotations

import tilelang
import tilelang.language as T


def make_matmul_prim(m: int, k: int, n: int):
    """Return a real one-launch FP16 matrix PrimFunc with explicit narrowing."""

    @T.prim_func
    def matmul_tile(
        A: T.Tensor((m, k), "float16"),
        B: T.Tensor((k, n), "float16"),
        Y: T.Tensor((m, n), "float16"),
    ):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            As = T.alloc_shared((m, k), "float16")
            Bs = T.alloc_shared((k, n), "float16")
            Cl = T.alloc_fragment((m, n), "float32")
            T.copy(A[0, 0], As)
            T.copy(B[0, 0], Bs)
            T.clear(Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Y[0, 0])

    return matmul_tile


def make_matmul_tiled_prim(m: int, k: int, n: int, *, bm: int, bn: int,
                          bk: int = 256, order: str = "m_major"):
    """A real TileLang M/N/K-tiled GEMM with an explicit traversal annotation.

    This is the source-level experiment form: block extents and the block grid
    are real TileLang schedule choices.  The frontend proves exact coverage and
    carries the chosen M/N tile plus order into RKIR.
    """
    if k % bk != 0:
        raise ValueError("explicit K-rounding fixture requires K % BK == 0")
    if m % bm != 0 or n % bn != 0:
        raise ValueError("fixture requires exact M/N block coverage")

    @T.prim_func
    def matmul_tiled(
        A: T.Tensor((m, k), "float16"),
        B: T.Tensor((k, n), "float16"),
        Y: T.Tensor((m, n), "float16"),
    ):
        T.func_attr({
            "rknpu.schedule": "mn_tiles_v1",
            "rknpu.mn_order": order,
            "rknpu.source_k_rounding": "explicit_per_k_fp16",
        })
        with T.Kernel(T.ceildiv(n, bn), T.ceildiv(m, bm),
                      threads=128) as (bx, by):
            As = T.alloc_shared((bm, bk), "float16")
            Bs = T.alloc_shared((bk, bn), "float16")
            Cl = T.alloc_fragment((bm, bn), "float32")
            Acc16 = T.alloc_fragment((bm, bn), "float16")
            T.clear(Acc16)
            for ko in T.Pipelined(T.ceildiv(k, bk), num_stages=2):
                T.copy(A[by * bm, ko * bk], As)
                T.copy(B[ko * bk, bx * bn], Bs)
                # Explicit source-level rounding point: the running FP16
                # accumulator is loaded into the FP32 GEMM accumulator, then
                # narrowed back to FP16 at the end of every K block.
                T.copy(Acc16, Cl)
                T.gemm(As, Bs, Cl)
                T.copy(Cl, Acc16)
            T.copy(Acc16, Y[by * bm, bx * bn])

    return matmul_tiled


def make_matmul_tiled_tail_prim(m: int, k: int, n: int, *, bm: int,
                                bn: int, bk: int = 256,
                                order: str = "m_major",
                                swapped: bool = False):
    """Real guarded M/N/K-tail GEMM with explicit per-K FP16 rounding.

    This is the scheduled source used when a projection crosses a 256-wide K
    boundary or when M/N is not divisible by the requested logical tile.  The
    physical A/B tiles are cleared on every K iteration, guarded loads touch
    only logical elements, and the running FP16 accumulator is round-tripped
    through the FP32 GEMM accumulator at every K block.
    """
    if bk != 256:
        raise ValueError("the supported source K block is exactly 256")
    attrs = {"rknpu.schedule": "mn_tiles_v1",
             "rknpu.mn_order": order,
             "rknpu.source_k_rounding": "explicit_per_k_fp16"}
    if swapped:
        @T.prim_func
        def matmul_tiled_tail(
            A: T.Tensor((m, k), "float16"),
            B: T.Tensor((k, n), "float16"),
            Y: T.Tensor((m, n), "float16"),
        ):
            T.func_attr(attrs)
            with T.Kernel(T.ceildiv(m, bm), T.ceildiv(n, bn),
                          threads=128) as (bm_i, bn_i):
                As = T.alloc_shared((bm, bk), "float16")
                Bs = T.alloc_shared((bk, bn), "float16")
                Cl = T.alloc_fragment((bm, bn), "float32")
                Acc16 = T.alloc_fragment((bm, bn), "float16")
                T.clear(Acc16)
                for ko in T.Pipelined(T.ceildiv(k, bk), num_stages=2):
                    T.fill(As, 0)
                    T.fill(Bs, 0)
                    for i, j in T.Parallel(bm, bk):
                        if bm_i * bm + i < m and ko * bk + j < k:
                            As[i, j] = A[bm_i * bm + i, ko * bk + j]
                    for i, j in T.Parallel(bk, bn):
                        if ko * bk + i < k and bn_i * bn + j < n:
                            Bs[i, j] = B[ko * bk + i, bn_i * bn + j]
                    T.copy(Acc16, Cl)
                    T.gemm(As, Bs, Cl)
                    T.copy(Cl, Acc16)
                for i, j in T.Parallel(bm, bn):
                    if bm_i * bm + i < m and bn_i * bn + j < n:
                        Y[bm_i * bm + i, bn_i * bn + j] = Acc16[i, j]
        return matmul_tiled_tail

    @T.prim_func
    def matmul_tiled_tail(
        A: T.Tensor((m, k), "float16"),
        B: T.Tensor((k, n), "float16"),
        Y: T.Tensor((m, n), "float16"),
    ):
        T.func_attr(attrs)
        with T.Kernel(T.ceildiv(n, bn), T.ceildiv(m, bm),
                      threads=128) as (bn_i, bm_i):
            As = T.alloc_shared((bm, bk), "float16")
            Bs = T.alloc_shared((bk, bn), "float16")
            Cl = T.alloc_fragment((bm, bn), "float32")
            Acc16 = T.alloc_fragment((bm, bn), "float16")
            T.clear(Acc16)
            for ko in T.Pipelined(T.ceildiv(k, bk), num_stages=2):
                T.fill(As, 0)
                T.fill(Bs, 0)
                for i, j in T.Parallel(bm, bk):
                    if bm_i * bm + i < m and ko * bk + j < k:
                        As[i, j] = A[bm_i * bm + i, ko * bk + j]
                for i, j in T.Parallel(bk, bn):
                    if ko * bk + i < k and bn_i * bn + j < n:
                        Bs[i, j] = B[ko * bk + i, bn_i * bn + j]
                T.copy(Acc16, Cl)
                T.gemm(As, Bs, Cl)
                T.copy(Cl, Acc16)
            for i, j in T.Parallel(bm, bn):
                if bm_i * bm + i < m and bn_i * bn + j < n:
                    Y[bm_i * bm + i, bn_i * bn + j] = Acc16[i, j]
    return matmul_tiled_tail


def make_matmul_prim_scheduled(m: int, k: int, n: int, *,
                               tile_m: int, tile_n: int,
                               order: str = "m_major"):
    """Same GEMM as ``make_matmul_prim`` with an explicit M/N schedule.

    TileLang still expresses the source GEMM and its full logical operands; the
    PrimFunc annotations are the schedule contract consumed by the RKIR lowerer.
    K blocking and accumulation are intentionally not parameters here.
    """
    @T.prim_func
    def matmul_tile(
        A: T.Tensor((m, k), "float16"),
        B: T.Tensor((k, n), "float16"),
        Y: T.Tensor((m, n), "float16"),
    ):
        T.func_attr({
            "rknpu.schedule": "mn_tiles_v1",
            "rknpu.tile_m": tile_m,
            "rknpu.tile_n": tile_n,
            "rknpu.mn_order": order,
        })
        with T.Kernel(1, 1, threads=128) as (bx, by):
            As = T.alloc_shared((m, k), "float16")
            Bs = T.alloc_shared((k, n), "float16")
            Cl = T.alloc_fragment((m, n), "float32")
            T.copy(A[0, 0], As)
            T.copy(B[0, 0], Bs)
            T.clear(Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Y[0, 0])

    return matmul_tile


def make_swiglu_ffn_f16(m: int, d: int, h: int):
    """Return the staged-FP16 SwiGLU FFN used by the device-resident FFN plan.

    The source materialises every narrowing point explicitly:
    gate/up accumulators -> FP16, SiLU -> FP16, gated product -> FP16,
    projection -> FP16, then residual add -> FP16.  The frontend lowers the
    graph as seven primitive device tasks; only X/Y cross the device boundary.
    """
    @T.prim_func
    def swiglu_ffn_f16(
        X: T.Tensor((m, d), "float16"),
        H: T.Tensor((m, d), "float16"),
        Wg: T.Tensor((d, h), "float16"),
        Wu: T.Tensor((d, h), "float16"),
        Wd: T.Tensor((h, d), "float16"),
        Y: T.Tensor((m, d), "float16"),
    ):
        with T.Kernel(1, threads=128) as _bx:
            X_shared = T.alloc_shared((m, d), "float16")
            Wg_shared = T.alloc_shared((d, h), "float16")
            Wu_shared = T.alloc_shared((d, h), "float16")
            Wd_shared = T.alloc_shared((h, d), "float16")
            act_shared = T.alloc_shared((m, h), "float16")
            gate32 = T.alloc_fragment((m, h), "float32")
            up32 = T.alloc_fragment((m, h), "float32")
            gate16 = T.alloc_fragment((m, h), "float16")
            up16 = T.alloc_fragment((m, h), "float16")
            silu16 = T.alloc_fragment((m, h), "float16")
            mul16 = T.alloc_fragment((m, h), "float16")
            proj32 = T.alloc_fragment((m, d), "float32")
            proj16 = T.alloc_fragment((m, d), "float16")
            T.clear(gate32)
            T.clear(up32)
            T.clear(proj32)
            T.copy(X, X_shared)
            T.copy(Wg, Wg_shared)
            T.copy(Wu, Wu_shared)
            T.gemm(X_shared, Wg_shared, gate32)
            T.gemm(X_shared, Wu_shared, up32)
            for i, j in T.Parallel(m, h):
                gate16[i, j] = gate32[i, j]
            for i, j in T.Parallel(m, h):
                up16[i, j] = up32[i, j]
            for i, j in T.Parallel(m, h):
                silu16[i, j] = gate16[i, j] * T.sigmoid(gate16[i, j])
            for i, j in T.Parallel(m, h):
                mul16[i, j] = silu16[i, j] * up16[i, j]
            T.copy(mul16, act_shared)
            T.copy(Wd, Wd_shared)
            T.gemm(act_shared, Wd_shared, proj32)
            for i, j in T.Parallel(m, d):
                proj16[i, j] = proj32[i, j]
            for i, j in T.Parallel(m, d):
                proj16[i, j] = proj16[i, j] + H[i, j]
            T.copy(proj16, Y)

    return swiglu_ffn_f16
