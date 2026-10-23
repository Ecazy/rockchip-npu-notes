# SPDX-License-Identifier: GPL-3.0-or-later
"""Real multi-block two-output branch with separately launched GEMM stages."""
import tilelang
import tilelang.language as T

M, K, N1, N2 = 512, 256, 512, 256
BM, BN1, BN2 = 256, 64, 256


@T.prim_func
def branch_staged(X: T.Tensor((M, K), "float16"),
                  W1: T.Tensor((K, N1), "float16"),
                  W2: T.Tensor((K, N2), "float16"),
                  Y1: T.Tensor((M, N1), "float16"),
                  Y2: T.Tensor((M, N2), "float16")):
    with T.Kernel(T.ceildiv(N1, BN1), T.ceildiv(M, BM), threads=128) as (bx, by):
        As = T.alloc_shared((BM, K), "float16")
        Bs = T.alloc_shared((K, BN1), "float16")
        Cl = T.alloc_fragment((BM, BN1), "float32")
        C16 = T.alloc_fragment((BM, BN1), "float16")
        T.copy(X[by * BM, 0], As)
        T.copy(W1[0, bx * BN1], Bs)
        T.clear(Cl)
        T.gemm(As, Bs, Cl)
        T.copy(Cl, C16)
        T.copy(C16, Y1[by * BM, bx * BN1])
    with T.Kernel(T.ceildiv(N2, BN2), T.ceildiv(M, BM), threads=128) as (bx, by):
        As = T.alloc_shared((BM, K), "float16")
        Bs = T.alloc_shared((K, BN2), "float16")
        Cl = T.alloc_fragment((BM, BN2), "float32")
        C16 = T.alloc_fragment((BM, BN2), "float16")
        T.copy(X[by * BM, 0], As)
        T.copy(W2[0, bx * BN2], Bs)
        T.clear(Cl)
        T.gemm(As, Bs, Cl)
        T.copy(Cl, C16)
        T.copy(C16, Y2[by * BM, bx * BN2])
