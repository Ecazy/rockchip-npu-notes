# SPDX-License-Identifier: GPL-3.0-or-later
"""Real TileLang source for the frozen MNIST MLP.

The logical 784-wide input is zero-padded to 1024 for the NPU primitive because
the locked ordered-K-tile contract uses 256-wide full K tiles.  The source still
represents the complete three-layer network; the model wrapper slices the first
784 values and the first 10 logits.
"""
from __future__ import annotations

import tilelang
import tilelang.language as T


def make_mlp_prim(m: int, *, k_pad: int = 1024):
    """Return a real staged PrimFunc for one static batch profile."""
    n1, n2, n3 = 256, 128, 16
    bk = 256
    bn = 64

    @T.prim_func
    def mnist_mlp_staged(
        X: T.Tensor((m, k_pad), "float16"),
        W1: T.Tensor((k_pad, n1), "float16"),
        B1: T.Tensor((m, n1), "float16"),
        W2: T.Tensor((n1, n2), "float16"),
        B2: T.Tensor((m, n2), "float16"),
        W3: T.Tensor((n2, n3), "float16"),
        B3: T.Tensor((m, n3), "float16"),
        Z1: T.Tensor((m, n1), "float16"),
        H1: T.Tensor((m, n1), "float16"),
        R1: T.Tensor((m, n1), "float16"),
        Z2: T.Tensor((m, n2), "float16"),
        H2: T.Tensor((m, n2), "float16"),
        R2: T.Tensor((m, n2), "float16"),
        Z3: T.Tensor((m, n3), "float16"),
        Y: T.Tensor((m, n3), "float16"),
    ):
        # Layer 1: X @ W1, explicit FP16 writeback after each 256-wide K tile.
        with T.Kernel(T.ceildiv(n1, bn), 1, threads=128) as (bx, by):
            As = T.alloc_shared((m, bk), "float16")
            Bs = T.alloc_shared((bk, bn), "float16")
            Cl = T.alloc_fragment((m, bn), "float32")
            Acc = T.alloc_fragment((m, bn), "float16")
            T.clear(Acc)
            for ko in T.Pipelined(T.ceildiv(k_pad, bk), num_stages=2):
                T.copy(X[0, ko * bk], As)
                T.copy(W1[ko * bk, bx * bn], Bs)
                T.copy(Acc, Cl)
                T.gemm(As, Bs, Cl)
                T.copy(Cl, Acc)
            T.copy(Acc, Z1[0, bx * bn])

        # Layer 1 bias, then exact ReLU.
        with T.Kernel(T.ceildiv(n1, bn), 1, threads=128) as (bx, by):
            Zs = T.alloc_shared((m, bn), "float16")
            Bl = T.alloc_shared((m, bn), "float16")
            Hl = T.alloc_fragment((m, bn), "float16")
            T.copy(Z1[0, bx * bn], Zs)
            T.copy(B1[0, bx * bn], Bl)
            for i, j in T.Parallel(m, bn):
                Hl[i, j] = T.cast(Zs[i, j] + Bl[i, j], "float16")
            T.copy(Hl, H1[0, bx * bn])
        with T.Kernel(T.ceildiv(n1, bn), 1, threads=128) as (bx, by):
            Hs = T.alloc_shared((m, bn), "float16")
            Rl = T.alloc_fragment((m, bn), "float16")
            T.copy(H1[0, bx * bn], Hs)
            for i, j in T.Parallel(m, bn):
                Rl[i, j] = T.cast(T.max(Hs[i, j], T.float16(0.0)), "float16")
            T.copy(Rl, R1[0, bx * bn])

        # Layer 2.
        with T.Kernel(T.ceildiv(n2, bn), 1, threads=128) as (bx, by):
            As = T.alloc_shared((m, n1), "float16")
            Bs = T.alloc_shared((n1, bn), "float16")
            Cl = T.alloc_fragment((m, bn), "float32")
            C16 = T.alloc_fragment((m, bn), "float16")
            T.copy(R1[0, 0], As)
            T.copy(W2[0, bx * bn], Bs)
            T.clear(Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, C16)
            T.copy(C16, Z2[0, bx * bn])
        with T.Kernel(T.ceildiv(n2, bn), 1, threads=128) as (bx, by):
            Zs = T.alloc_shared((m, bn), "float16")
            Bl = T.alloc_shared((m, bn), "float16")
            Hl = T.alloc_fragment((m, bn), "float16")
            T.copy(Z2[0, bx * bn], Zs)
            T.copy(B2[0, bx * bn], Bl)
            for i, j in T.Parallel(m, bn):
                Hl[i, j] = T.cast(Zs[i, j] + Bl[i, j], "float16")
            T.copy(Hl, H2[0, bx * bn])
        with T.Kernel(T.ceildiv(n2, bn), 1, threads=128) as (bx, by):
            Hs = T.alloc_shared((m, bn), "float16")
            Rl = T.alloc_fragment((m, bn), "float16")
            T.copy(H2[0, bx * bn], Hs)
            for i, j in T.Parallel(m, bn):
                Rl[i, j] = T.cast(T.max(Hs[i, j], T.float16(0.0)), "float16")
            T.copy(Rl, R2[0, bx * bn])

        # Layer 3 and final bias (N=16; first 10 logits are exported).
        with T.Kernel(1, 1, threads=128) as (bx, by):
            As = T.alloc_shared((m, n2), "float16")
            Bs = T.alloc_shared((n2, n3), "float16")
            Cl = T.alloc_fragment((m, n3), "float32")
            C16 = T.alloc_fragment((m, n3), "float16")
            T.copy(R2[0, 0], As)
            T.copy(W3[0, 0], Bs)
            T.clear(Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, C16)
            T.copy(C16, Z3[0, 0])
        with T.Kernel(1, 1, threads=128) as (bx, by):
            Zs = T.alloc_shared((m, n3), "float16")
            Bl = T.alloc_shared((m, n3), "float16")
            Yl = T.alloc_fragment((m, n3), "float16")
            T.copy(Z3[0, 0], Zs)
            T.copy(B3[0, 0], Bl)
            for i, j in T.Parallel(m, n3):
                Yl[i, j] = T.cast(Zs[i, j] + Bl[i, j], "float16")
            T.copy(Yl, Y[0, 0])

    return mnist_mlp_staged


def make_relu_probe_prim(m: int = 2, n: int = 256):
    """Return a real TileLang bias-add -> exact-ReLU diagnostic PrimFunc.

    Two explicit kernel launches keep the same production extraction/lowering
    path as the MLP activation stages.  ``co.ew_max.f16.v1`` is exercised against
    a zero operand; this is not a LUT/HardSigmoid approximation.
    """

    @T.prim_func
    def mnist_relu_probe(
        X: T.Tensor((m, n), "float16"),
        B: T.Tensor((m, n), "float16"),
        Z: T.Tensor((m, n), "float16"),
        Y: T.Tensor((m, n), "float16"),
    ):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            Xs = T.alloc_shared((m, n), "float16")
            Bs = T.alloc_shared((m, n), "float16")
            Zl = T.alloc_fragment((m, n), "float16")
            T.copy(X[0, 0], Xs)
            T.copy(B[0, 0], Bs)
            for i, j in T.Parallel(m, n):
                Zl[i, j] = T.cast(Xs[i, j] + Bs[i, j], "float16")
            T.copy(Zl, Z[0, 0])

        with T.Kernel(1, 1, threads=128) as (bx, by):
            Zs = T.alloc_shared((m, n), "float16")
            Yl = T.alloc_fragment((m, n), "float16")
            T.copy(Z[0, 0], Zs)
            for i, j in T.Parallel(m, n):
                Yl[i, j] = T.cast(T.max(Zs[i, j], T.float16(0.0)),
                                  "float16")
            T.copy(Yl, Y[0, 0])

    return mnist_relu_probe
