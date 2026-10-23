# SPDX-License-Identifier: GPL-3.0-or-later
"""Real TileLang fixture with two independent matmul outputs.

    Y1 = X @ W1      Y2 = X @ W2

The two branches share one ``X -> X_shared`` copy but have distinct weights,
accumulators and exported FP16 values.  Both accumulator-to-FP16 stores are
explicit; the final copies are type-preserving.
"""
import tilelang
import tilelang.language as T


def branch_outputs(m=8, k=64, n1=32, n2=48):
    @T.prim_func
    def kernel(X: T.Tensor((m, k), "float16"),
               W1: T.Tensor((k, n1), "float16"),
               W2: T.Tensor((k, n2), "float16"),
               Y1: T.Tensor((m, n1), "float16"),
               Y2: T.Tensor((m, n2), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            X_shared = T.alloc_shared((m, k), "float16")
            W1_shared = T.alloc_shared((k, n1), "float16")
            W2_shared = T.alloc_shared((k, n2), "float16")
            acc1 = T.alloc_fragment((m, n1), "float32")
            acc2 = T.alloc_fragment((m, n2), "float32")
            y1_16 = T.alloc_fragment((m, n1), "float16")
            y2_16 = T.alloc_fragment((m, n2), "float16")
            T.clear(acc1)
            T.clear(acc2)
            T.copy(X, X_shared)
            T.copy(W1, W1_shared)
            T.copy(W2, W2_shared)
            T.gemm(X_shared, W1_shared, acc1)
            T.gemm(X_shared, W2_shared, acc2)
            for i, j in T.Parallel(m, n1):
                y1_16[i, j] = acc1[i, j]
            for i, j in T.Parallel(m, n2):
                y2_16[i, j] = acc2[i, j]
            T.copy(y1_16, Y1)
            T.copy(y2_16, Y2)
    return kernel
