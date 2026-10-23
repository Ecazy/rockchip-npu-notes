# SPDX-License-Identifier: GPL-3.0-or-later
"""Real multi-block, multi-stage projection with explicit stage boundaries.

Shapes are chosen exact for the composition-domain fixture:
  M=512, D=256, H=512; BM=256, BH=64, BD=256, BK=256.

Each ``T.Kernel`` is a separate launch.  Global intermediates Z/S/G/P are
materialised between launches, so the two GEMM stages have a real global
synchronisation boundary rather than an assumed cross-block ordering.
The projection K loop explicitly writes the FP16 accumulator back after every
256-wide K tile, matching the backend's ordered K-tile numeric contract.
"""
import tilelang
import tilelang.language as T

M, D, H = 512, 256, 512
BM, BH, BD, BK = 256, 64, 256, 256


@T.prim_func
def projection_staged(X: T.Tensor((M, D), "float16"),
                      W1: T.Tensor((D, H), "float16"),
                      W2: T.Tensor((H, D), "float16"),
                      Z: T.Tensor((M, H), "float16"),
                      S: T.Tensor((M, H), "float16"),
                      G: T.Tensor((M, H), "float16"),
                      P: T.Tensor((M, D), "float16"),
                      Y: T.Tensor((M, D), "float16")):
    with T.Kernel(T.ceildiv(H, BH), T.ceildiv(M, BM), threads=128) as (bx, by):
        As = T.alloc_shared((BM, D), "float16")
        Bs = T.alloc_shared((D, BH), "float16")
        Cl = T.alloc_fragment((BM, BH), "float32")
        C16 = T.alloc_fragment((BM, BH), "float16")
        T.copy(X[by * BM, 0], As)
        T.copy(W1[0, bx * BH], Bs)
        T.clear(Cl)
        T.gemm(As, Bs, Cl)
        T.copy(Cl, C16)
        T.copy(C16, Z[by * BM, bx * BH])
    with T.Kernel(T.ceildiv(H, BH), T.ceildiv(M, BM), threads=128) as (bx, by):
        Zs = T.alloc_shared((BM, BH), "float16")
        Sl = T.alloc_fragment((BM, BH), "float16")
        T.copy(Z[by * BM, bx * BH], Zs)
        for i, j in T.Parallel(BM, BH):
            Sl[i, j] = T.cast(T.sigmoid(Zs[i, j]), "float16")
        T.copy(Sl, S[by * BM, bx * BH])
    with T.Kernel(T.ceildiv(H, BH), T.ceildiv(M, BM), threads=128) as (bx, by):
        Zs = T.alloc_shared((BM, BH), "float16")
        Ss = T.alloc_shared((BM, BH), "float16")
        Gl = T.alloc_fragment((BM, BH), "float16")
        T.copy(Z[by * BM, bx * BH], Zs)
        T.copy(S[by * BM, bx * BH], Ss)
        for i, j in T.Parallel(BM, BH):
            Gl[i, j] = T.cast(Ss[i, j] * Zs[i, j], "float16")
        T.copy(Gl, G[by * BM, bx * BH])
    with T.Kernel(T.ceildiv(D, BD), T.ceildiv(M, BM), threads=128) as (bx, by):
        As = T.alloc_shared((BM, BK), "float16")
        Bs = T.alloc_shared((BK, BD), "float16")
        Cl = T.alloc_fragment((BM, BD), "float32")
        Acc16 = T.alloc_fragment((BM, BD), "float16")
        T.clear(Acc16)
        for ko in T.Pipelined(T.ceildiv(H, BK), num_stages=2):
            T.copy(G[by * BM, ko * BK], As)
            T.copy(W2[ko * BK, bx * BD], Bs)
            T.copy(Acc16, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc16)
        T.copy(Acc16, P[by * BM, bx * BD])
    with T.Kernel(T.ceildiv(D, BD), T.ceildiv(M, BM), threads=128) as (bx, by):
        Ps = T.alloc_shared((BM, BD), "float16")
        Xs = T.alloc_shared((BM, BD), "float16")
        Yl = T.alloc_fragment((BM, BD), "float16")
        T.copy(P[by * BM, bx * BD], Ps)
        T.copy(X[by * BM, bx * BD], Xs)
        for i, j in T.Parallel(BM, BD):
            Yl[i, j] = T.cast(Ps[i, j] + Xs[i, j], "float16")
        T.copy(Yl, Y[by * BM, bx * BD])


@T.prim_func
def projection_fp32_accum_only(X: T.Tensor((512, 512), "float16"),
                               W: T.Tensor((512, 256), "float16"),
                               Z: T.Tensor((512, 512), "float16"),
                               Y: T.Tensor((512, 256), "float16")):
    """Negative: K loop accumulates in FP32 and narrows only after the loop."""
    with T.Kernel(T.ceildiv(512, BM), T.ceildiv(512, BM), threads=128) as (bx, by):
        Xs = T.alloc_shared((BM, BM), "float16")
        Zs = T.alloc_fragment((BM, BM), "float16")
        T.copy(X[by * BM, bx * BM], Xs)
        for i, j in T.Parallel(BM, BM):
            Zs[i, j] = T.cast(T.sigmoid(Xs[i, j]), "float16")
        T.copy(Zs, Z[by * BM, bx * BM])
    with T.Kernel(T.ceildiv(256, BD), T.ceildiv(512, BM), threads=128) as (bx, by):
        As = T.alloc_shared((BM, BK), "float16")
        Bs = T.alloc_shared((BK, BD), "float16")
        Cl = T.alloc_fragment((BM, BD), "float32")
        C16 = T.alloc_fragment((BM, BD), "float16")
        T.clear(Cl)
        for ko in T.Pipelined(T.ceildiv(512, BK), num_stages=2):
            T.copy(Z[by * BM, ko * BK], As)
            T.copy(W[ko * BK, bx * BD], Bs)
            T.gemm(As, Bs, Cl)
        T.copy(Cl, C16)
        T.copy(C16, Y[by * BM, bx * BD])


@T.prim_func
def projection_cross_block_read(X: T.Tensor((512, 256), "float16"),
                                W: T.Tensor((256, 512), "float16"),
                                Z: T.Tensor((512, 512), "float16"),
                                Y: T.Tensor((512, 512), "float16")):
    """Negative: one launch writes Z and then reads Z without a global barrier."""
    with T.Kernel(T.ceildiv(512, 64), T.ceildiv(512, 256), threads=128) as (bx, by):
        As = T.alloc_shared((256, 256), "float16")
        Bs = T.alloc_shared((256, 64), "float16")
        Cl = T.alloc_fragment((256, 64), "float32")
        C16 = T.alloc_fragment((256, 64), "float16")
        Unused = T.alloc_shared((256, 64), "float16")
        T.copy(X[by * 256, 0], As)
        T.copy(W[0, bx * 64], Bs)
        T.clear(Cl)
        T.gemm(As, Bs, Cl)
        T.copy(Cl, C16)
        T.copy(C16, Z[by * 256, bx * 64])
        # Same parallel launch, different blocks cannot rely on this value.
        T.copy(Z[by * 256, bx * 64], Unused)
    with T.Kernel(T.ceildiv(512, 64), T.ceildiv(512, 256), threads=128) as (bx, by):
        Zs = T.alloc_shared((256, 64), "float16")
        Yl = T.alloc_fragment((256, 64), "float16")
        T.copy(Z[by * 256, bx * 64], Zs)
        for i, j in T.Parallel(256, 64):
            Yl[i, j] = T.cast(T.sigmoid(Zs[i, j]), "float16")
        T.copy(Yl, Y[by * 256, bx * 64])


@T.prim_func
def projection_bad_coverage(X: T.Tensor((512, 256), "float16"),
                            W: T.Tensor((256, 512), "float16"),
                            Z: T.Tensor((512, 512), "float16"),
                            Y: T.Tensor((512, 512), "float16")):
    """Negative: seven N tiles publish only 448 of 512 columns."""
    with T.Kernel(7, T.ceildiv(512, 256), threads=128) as (bx, by):
        As = T.alloc_shared((256, 256), "float16")
        Bs = T.alloc_shared((256, 64), "float16")
        Cl = T.alloc_fragment((256, 64), "float32")
        C16 = T.alloc_fragment((256, 64), "float16")
        T.copy(X[by * 256, 0], As)
        T.copy(W[0, bx * 64], Bs)
        T.clear(Cl)
        T.gemm(As, Bs, Cl)
        T.copy(Cl, C16)
        T.copy(C16, Z[by * 256, bx * 64])
    with T.Kernel(T.ceildiv(512, 64), T.ceildiv(512, 256), threads=128) as (bx, by):
        Zs = T.alloc_shared((256, 64), "float16")
        Yl = T.alloc_fragment((256, 64), "float16")
        T.copy(Z[by * 256, bx * 64], Zs)
        for i, j in T.Parallel(256, 64):
            Yl[i, j] = T.cast(T.sigmoid(Zs[i, j]), "float16")
        T.copy(Yl, Y[by * 256, bx * 64])
