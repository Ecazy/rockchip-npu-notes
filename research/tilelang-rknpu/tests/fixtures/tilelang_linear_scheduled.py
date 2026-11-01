"""Real TileLang single GEMM with an explicit M/N schedule and per-K FP16.

The production decode kernel (``models/tilelang_matmul.make_matmul_prim``) is a
one-launch GEMM whose K loop materializes no FP16 round trip: for a K=576 source
that makes the extractor stamp no schedule and the planner take the historical
grid.  This fixture is the same GEMM written the way the schedule contract asks
for it -- a K loop that writes FP16 back after every 256-wide block, plus the
explicit M/N annotation -- so the declared path can be exercised on a real
TileLang source rather than only through direct RKIR.
"""
from __future__ import annotations

from tilelang import language as T


def make_linear(m=128, k=576, n=3072, bm=256, bn=256, bk=256,
                order="m_major"):
    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"),
               B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        T.func_attr({"rknpu.schedule": "mn_tiles_v1",
                     "rknpu.tile_m": bm, "rknpu.tile_n": bn,
                     "rknpu.mn_order": order,
                     "rknpu.source_k_rounding": "explicit_per_k_fp16"})
        with T.Kernel(T.ceildiv(n, bn), T.ceildiv(m, bm),
                      threads=128) as (bx, by):
            As = T.alloc_shared((bm, bk), "float16")
            Bs = T.alloc_shared((bk, bn), "float16")
            Cl = T.alloc_fragment((bm, bn), "float32")
            Acc = T.alloc_fragment((bm, bn), "float16")
            T.clear(Acc)
            # One FP16 writeback per 256-wide K block, including the 64 tail.
            for ko in T.Pipelined(T.ceildiv(k, bk), num_stages=2):
                T.fill(As, 0)
                T.fill(Bs, 0)
                for i, j in T.Parallel(bm, bk):
                    if by * bm + i < m and ko * bk + j < k:
                        As[i, j] = A[by * bm + i, ko * bk + j]
                for i, j in T.Parallel(bk, bn):
                    if ko * bk + i < k and bx * bn + j < n:
                        Bs[i, j] = B[ko * bk + i, bx * bn + j]
                T.copy(Acc, Cl)
                T.gemm(As, Bs, Cl)
                T.copy(Cl, Acc)
            for i, j in T.Parallel(bm, bn):
                if by * bm + i < m and bx * bn + j < n:
                    C[by * bm + i, bx * bn + j] = Acc[i, j]
    return kernel
