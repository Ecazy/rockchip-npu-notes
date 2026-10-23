# SPDX-License-Identifier: GPL-3.0-or-later
"""Example TileLang kernel: MatMul -> SiLU -> MatMul(projection) + residual.

    Z = X @ W1          (X[M,D] @ W1[D,H])
    S = Z * sigmoid(Z)  (SiLU)
    P = S @ W2          (S[M,H] @ W2[H,D])
    Y = P + X           (residual)

The declared FP16 narrowing points are spelled out with explicit buffers, so the
numeric contract is visible in the TIR rather than inferred from names:

    1. GEMM1 accumulator (fp32) -> Z16  (fp16)
    2. sigmoid(Z16)             -> G16  (fp16)
    3. Z16 * G16                -> S16  (fp16)
    4. GEMM2 accumulator (fp32) -> P16  (fp16)
    5. P16 + X                  -> Y    (fp16)

Weights use the natural TileLang [K, N] layout; mapping them onto the custom-ops
[N, K] packing is the adapter's job and is asserted separately.
"""
import tilelang
import tilelang.language as T


def silu_proj(m=13, d=64, h=32):
    @T.prim_func
    def kernel(X: T.Tensor((m, d), "float16"),
               W1: T.Tensor((d, h), "float16"),
               W2: T.Tensor((h, d), "float16"),
               Y: T.Tensor((m, d), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            X1_shared = T.alloc_shared((m, d), "float16")
            W1_shared = T.alloc_shared((d, h), "float16")
            S_shared = T.alloc_shared((m, h), "float16")
            W2_shared = T.alloc_shared((h, d), "float16")
            acc1 = T.alloc_fragment((m, h), "float32")
            z16 = T.alloc_fragment((m, h), "float16")
            g16 = T.alloc_fragment((m, h), "float16")
            s16 = T.alloc_fragment((m, h), "float16")
            acc2 = T.alloc_fragment((m, d), "float32")
            p16 = T.alloc_fragment((m, d), "float16")
            T.clear(acc1)
            T.clear(acc2)
            T.copy(X, X1_shared)
            T.copy(W1, W1_shared)
            T.gemm(X1_shared, W1_shared, acc1)          # narrowing point 1 below
            for i, j in T.Parallel(m, h):
                z16[i, j] = acc1[i, j]                  # 1. fp32 acc -> fp16 Z
            for i, j in T.Parallel(m, h):
                g16[i, j] = T.sigmoid(z16[i, j])        # 2. gate narrowed to fp16
            for i, j in T.Parallel(m, h):
                s16[i, j] = z16[i, j] * g16[i, j]       # 3. SiLU narrowed to fp16
            T.copy(s16, S_shared)
            T.copy(W2, W2_shared)
            T.gemm(S_shared, W2_shared, acc2)
            for i, j in T.Parallel(m, d):
                p16[i, j] = acc2[i, j]                  # 4. projection -> fp16
            for i, j in T.Parallel(m, d):
                p16[i, j] = p16[i, j] + X[i, j]         # 5. residual -> fp16
            T.copy(p16, Y)
    return kernel

kernel = silu_proj()
