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
