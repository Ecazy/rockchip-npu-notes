# SPDX-License-Identifier: GPL-3.0-or-later
"""Real TileLang PrimFunc fixture: MatMul followed by the registered SiLU form.

The expression ``z * sigmoid(z)`` is matched structurally from the actual TIR
BufferStore/Mul/Call nodes.  No string parsing is used.
"""
import tilelang
import tilelang.language as T


def make_kernel(m=32, k=64, n=48):
    @T.prim_func
    def kernel(A: T.Tensor((m, k), "float16"),
               B: T.Tensor((k, n), "float16"),
               C: T.Tensor((m, n), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            A_shared = T.alloc_shared((m, k), "float16")
            B_shared = T.alloc_shared((k, n), "float16")
            C_local = T.alloc_fragment((m, n), "float32")
            T.clear(C_local)
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.gemm(A_shared, B_shared, C_local)
            for i, j in T.Parallel(m, n):
                C_local[i, j] = C_local[i, j] * T.sigmoid(C_local[i, j])
            T.copy(C_local, C)
    return kernel


kernel = make_kernel()
