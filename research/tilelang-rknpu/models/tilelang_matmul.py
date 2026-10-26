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
