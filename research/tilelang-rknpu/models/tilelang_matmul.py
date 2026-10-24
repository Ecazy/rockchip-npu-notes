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
