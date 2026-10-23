# SPDX-License-Identifier: GPL-3.0-or-later
"""Real multi-block SwiGLU FFN with separately launched materialised stages."""
import tilelang
import tilelang.language as T

M, D, H = 512, 256, 512
BM, BH, BD, BK = 256, 64, 256, 256


@T.prim_func
def ffn_staged(X: T.Tensor((M, D), "float16"),
               Wg: T.Tensor((D, H), "float16"),
               Wu: T.Tensor((D, H), "float16"),
               Wd: T.Tensor((H, D), "float16"),
               Zg: T.Tensor((M, H), "float16"),
               Zu: T.Tensor((M, H), "float16"),
               Sg: T.Tensor((M, H), "float16"),
               Silu: T.Tensor((M, H), "float16"),
               A: T.Tensor((M, H), "float16"),
               P: T.Tensor((M, D), "float16"),
               Y: T.Tensor((M, D), "float16")):
    # gate = X @ Wg
    with T.Kernel(T.ceildiv(H, BH), T.ceildiv(M, BM), threads=128) as (bx, by):
        As = T.alloc_shared((BM, D), "float16")
        Bs = T.alloc_shared((D, BH), "float16")
        Cl = T.alloc_fragment((BM, BH), "float32")
        C16 = T.alloc_fragment((BM, BH), "float16")
        T.copy(X[by * BM, 0], As)
        T.copy(Wg[0, bx * BH], Bs)
        T.clear(Cl)
        T.gemm(As, Bs, Cl)
        T.copy(Cl, C16)
        T.copy(C16, Zg[by * BM, bx * BH])
    # up = X @ Wu
    with T.Kernel(T.ceildiv(H, BH), T.ceildiv(M, BM), threads=128) as (bx, by):
        As = T.alloc_shared((BM, D), "float16")
        Bs = T.alloc_shared((D, BH), "float16")
        Cl = T.alloc_fragment((BM, BH), "float32")
        C16 = T.alloc_fragment((BM, BH), "float16")
        T.copy(X[by * BM, 0], As)
        T.copy(Wu[0, bx * BH], Bs)
        T.clear(Cl)
        T.gemm(As, Bs, Cl)
        T.copy(Cl, C16)
        T.copy(C16, Zu[by * BM, bx * BH])
    # sigmoid(gate)
    with T.Kernel(T.ceildiv(H, BH), T.ceildiv(M, BM), threads=128) as (bx, by):
        Zs = T.alloc_shared((BM, BH), "float16")
        Sl = T.alloc_fragment((BM, BH), "float16")
        T.copy(Zg[by * BM, bx * BH], Zs)
        for i, j in T.Parallel(BM, BH):
            Sl[i, j] = T.cast(T.sigmoid(Zs[i, j]), "float16")
        T.copy(Sl, Sg[by * BM, bx * BH])
    # silu = gate * sigmoid(gate)
    with T.Kernel(T.ceildiv(H, BH), T.ceildiv(M, BM), threads=128) as (bx, by):
        Zs = T.alloc_shared((BM, BH), "float16")
        Ss = T.alloc_shared((BM, BH), "float16")
        Gl = T.alloc_fragment((BM, BH), "float16")
        T.copy(Zg[by * BM, bx * BH], Zs)
        T.copy(Sg[by * BM, bx * BH], Ss)
        for i, j in T.Parallel(BM, BH):
            Gl[i, j] = T.cast(Zs[i, j] * Ss[i, j], "float16")
        T.copy(Gl, Silu[by * BM, bx * BH])
    # act = silu(gate) * up
    with T.Kernel(T.ceildiv(H, BH), T.ceildiv(M, BM), threads=128) as (bx, by):
        Gs = T.alloc_shared((BM, BH), "float16")
        Us = T.alloc_shared((BM, BH), "float16")
        Al = T.alloc_fragment((BM, BH), "float16")
        T.copy(Silu[by * BM, bx * BH], Gs)
        T.copy(Zu[by * BM, bx * BH], Us)
        for i, j in T.Parallel(BM, BH):
            Al[i, j] = T.cast(Gs[i, j] * Us[i, j], "float16")
        T.copy(Al, A[by * BM, bx * BH])
    # projection with explicit FP16 K-tile writeback
    with T.Kernel(T.ceildiv(D, BD), T.ceildiv(M, BM), threads=128) as (bx, by):
        As = T.alloc_shared((BM, BK), "float16")
        Bs = T.alloc_shared((BK, BD), "float16")
        Cl = T.alloc_fragment((BM, BD), "float32")
        Acc16 = T.alloc_fragment((BM, BD), "float16")
        T.clear(Acc16)
        for ko in T.Pipelined(T.ceildiv(H, BK), num_stages=2):
            T.copy(A[by * BM, ko * BK], As)
            T.copy(Wd[ko * BK, bx * BD], Bs)
            T.copy(Acc16, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc16)
        T.copy(Acc16, P[by * BM, bx * BD])
    # residual
    with T.Kernel(T.ceildiv(D, BD), T.ceildiv(M, BM), threads=128) as (bx, by):
        Ps = T.alloc_shared((BM, BD), "float16")
        Xs = T.alloc_shared((BM, BD), "float16")
        Yl = T.alloc_fragment((BM, BD), "float16")
        T.copy(P[by * BM, bx * BD], Ps)
        T.copy(X[by * BM, bx * BD], Xs)
        for i, j in T.Parallel(BM, BD):
            Yl[i, j] = T.cast(Ps[i, j] + Xs[i, j], "float16")
        T.copy(Yl, Y[by * BM, bx * BD])
