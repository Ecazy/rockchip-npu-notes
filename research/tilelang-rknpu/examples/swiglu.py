# SPDX-License-Identifier: GPL-3.0-or-later
"""SwiGLU intent fixture for the next work package.

This is a real multi-GEMM TileLang PrimFunc, intentionally rejected by the
minimal one-GEMM structural extractor.  Keeping it here prevents claiming the
SwiGLU graph is implemented when only the matmul->SiLU vertical slice is done.
"""
import tilelang
import tilelang.language as T


def make_kernel(m=16, k=32, h=24):
    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"),
               Wg: T.Tensor((k, h), "float16"),
               Wu: T.Tensor((k, h), "float16"),
               C: T.Tensor((m, h), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            Ag = T.alloc_shared((m, k), "float16")
            Au = T.alloc_shared((m, k), "float16")
            Wg_shared = T.alloc_shared((k, h), "float16")
            Wu_shared = T.alloc_shared((k, h), "float16")
            gate = T.alloc_fragment((m, h), "float32")
            up = T.alloc_fragment((m, h), "float32")
            T.clear(gate)
            T.clear(up)
            T.copy(A, Ag)
            T.copy(A, Au)
            T.copy(Wg, Wg_shared)
            T.copy(Wu, Wu_shared)
            T.gemm(Ag, Wg_shared, gate)
            T.gemm(Au, Wu_shared, up)
            for i, j in T.Parallel(m, h):
                gate[i, j] = gate[i, j] * T.sigmoid(gate[i, j]) * up[i, j]
            T.copy(gate, C)
    return kernel


kernel = make_kernel()
