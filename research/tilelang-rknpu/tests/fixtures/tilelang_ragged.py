# SPDX-License-Identifier: GPL-3.0-or-later
"""Canonical explicit guarded rectangular-tail GEMM.

The source spells out the only accepted ragged form:

* zero-fill both shared input tiles on **every** K iteration;
* conditionally load only in-bounds A/B elements;
* guard the output store to the logical M/N rectangle.

There is no assumption that ``T.copy`` inserts bounds predicates.  Tails in M,
N and K may all be non-zero simultaneously.
"""
import tilelang
import tilelang.language as T


def ragged_tail_f16(m=13, n=33, k=257, bm=16, bn=64, bk=256,
                    order="m_major", swapped=False, bad_index=False):
    """Guarded M/N/K tail source with explicit per-K FP16 rounding.

    The running accumulator is FP16.  Every K iteration loads it into the FP32
    GEMM accumulator and stores the result back to FP16, so the last partial K
    block has exactly the same rounding point as every full block.  Both the
    canonical (x=N, y=M) grid and the swapped (x=M, y=N) grid are expressed by
    real index expressions, not by relabelled axes.
    """
    attrs = {"rknpu.schedule": "mn_tiles_v1",
             "rknpu.mn_order": order,
             "rknpu.source_k_rounding": "explicit_per_k_fp16"}
    k_off = 1 if bad_index else 0
    if swapped:
        @T.prim_func
        def kernel(A: T.Tensor((m, k), "float16"),
                   B: T.Tensor((k, n), "float16"),
                   C: T.Tensor((m, n), "float16")):
            T.func_attr(attrs)
            with T.Kernel(T.ceildiv(m, bm), T.ceildiv(n, bn),
                          threads=128) as (bm_i, bn_i):
                As = T.alloc_shared((bm, bk), "float16")
                Bs = T.alloc_shared((bk, bn), "float16")
                Cl = T.alloc_fragment((bm, bn), "float32")
                Acc = T.alloc_fragment((bm, bn), "float16")
                T.clear(Acc)
                for ko in T.Pipelined(T.ceildiv(k, bk), num_stages=2):
                    T.fill(As, 0)
                    T.fill(Bs, 0)
                    for i, j in T.Parallel(bm, bk):
                        if bm_i * bm + i < m and ko * bk + j < k:
                            As[i, j] = A[bm_i * bm + i,
                                       ko * bk + j + k_off]
                    for i, j in T.Parallel(bk, bn):
                        if ko * bk + i < k and bn_i * bn + j < n:
                            Bs[i, j] = B[ko * bk + i, bn_i * bn + j]
                    T.copy(Acc, Cl)
                    T.gemm(As, Bs, Cl)
                    T.copy(Cl, Acc)
                for i, j in T.Parallel(bm, bn):
                    if bm_i * bm + i < m and bn_i * bn + j < n:
                        C[bm_i * bm + i, bn_i * bn + j] = Acc[i, j]
        return kernel

    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"),
               B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        T.func_attr(attrs)
        with T.Kernel(T.ceildiv(n, bn), T.ceildiv(m, bm),
                      threads=128) as (bn_i, bm_i):
            As = T.alloc_shared((bm, bk), "float16")
            Bs = T.alloc_shared((bk, bn), "float16")
            Cl = T.alloc_fragment((bm, bn), "float32")
            Acc = T.alloc_fragment((bm, bn), "float16")
            T.clear(Acc)
            for ko in T.Pipelined(T.ceildiv(k, bk), num_stages=2):
                T.fill(As, 0)
                T.fill(Bs, 0)
                for i, j in T.Parallel(bm, bk):
                    if bm_i * bm + i < m and ko * bk + j < k:
                        As[i, j] = A[bm_i * bm + i,
                                   ko * bk + j + k_off]
                for i, j in T.Parallel(bk, bn):
                    if ko * bk + i < k and bn_i * bn + j < n:
                        Bs[i, j] = B[ko * bk + i, bn_i * bn + j]
                T.copy(Acc, Cl)
                T.gemm(As, Bs, Cl)
                T.copy(Cl, Acc)
            for i, j in T.Parallel(bm, bn):
                if bm_i * bm + i < m and bn_i * bn + j < n:
                    C[bm_i * bm + i, bn_i * bn + j] = Acc[i, j]
    return kernel


def ragged_tail(m=13, n=33, k=257, bm=16, bn=64, bk=256):
    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"),
               B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(T.ceildiv(n, bn), T.ceildiv(m, bm),
                      threads=128) as (bx, by):
            As = T.alloc_shared((bm, bk), "float16")
            Bs = T.alloc_shared((bk, bn), "float16")
            Cl = T.alloc_fragment((bm, bn), "float32")
            T.clear(Cl)
            for ko in T.Pipelined(T.ceildiv(k, bk), num_stages=2):
                # Redefine the whole physical tiles so a short K slice never
                # inherits bytes from the previous iteration.
                T.fill(As, 0)
                T.fill(Bs, 0)
                for i, j in T.Parallel(bm, bk):
                    if by * bm + i < m and ko * bk + j < k:
                        As[i, j] = A[by * bm + i, ko * bk + j]
                for i, j in T.Parallel(bk, bn):
                    if ko * bk + i < k and bx * bn + j < n:
                        Bs[i, j] = B[ko * bk + i, bx * bn + j]
                T.gemm(As, Bs, Cl)
            for i, j in T.Parallel(bm, bn):
                if by * bm + i < m and bx * bn + j < n:
                    C[by * bm + i, bx * bn + j] = Cl[i, j]
    return kernel


def ragged_tail_legacy_scheduled(m=13, n=33, k=576, bm=16, bn=64,
                                 bk=256, order="m_major"):
    """Legacy final-only narrowing carrying schedule annotations.

    The schedule annotation is deliberately present but the source has no
    per-K FP16 round trip.  The frontend must reject it instead of treating a
    final FP32 narrowing as the backend's per-K writeback.
    """
    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"),
               B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        T.func_attr({"rknpu.schedule": "mn_tiles_v1",
                     "rknpu.mn_order": order,
                     "rknpu.source_k_rounding": "fp32_final_narrowing"})
        with T.Kernel(T.ceildiv(n, bn), T.ceildiv(m, bm),
                      threads=128) as (bx, by):
            As = T.alloc_shared((bm, bk), "float16")
            Bs = T.alloc_shared((bk, bn), "float16")
            Cl = T.alloc_fragment((bm, bn), "float32")
            T.clear(Cl)
            for ko in T.Pipelined(T.ceildiv(k, bk), num_stages=2):
                T.fill(As, 0)
                T.fill(Bs, 0)
                for i, j in T.Parallel(bm, bk):
                    if by * bm + i < m and ko * bk + j < k:
                        As[i, j] = A[by * bm + i, ko * bk + j]
                for i, j in T.Parallel(bk, bn):
                    if ko * bk + i < k and bx * bn + j < n:
                        Bs[i, j] = B[ko * bk + i, bx * bn + j]
                T.gemm(As, Bs, Cl)
            for i, j in T.Parallel(bm, bn):
                if by * bm + i < m and bx * bn + j < n:
                    C[by * bm + i, bx * bn + j] = Cl[i, j]
    return kernel
