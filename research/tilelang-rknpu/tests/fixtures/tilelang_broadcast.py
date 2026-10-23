# SPDX-License-Identifier: GPL-3.0-or-later
"""Real TileLang matmul followed by scalar / per-channel constant Add or Mul."""

import tilelang
import tilelang.language as T


def _tail_kernel(*, m, k, n, op, bias_shape):
    if bias_shape == ():
        bias_ty = T.Tensor((), "float16")
    else:
        bias_ty = T.Tensor((n,), "float16")

    @T.prim_func
    def kernel(X: T.Tensor((m, k), "float16"),
               W: T.Tensor((k, n), "float16"),
               B: bias_ty,
               Y: T.Tensor((m, n), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            X_shared = T.alloc_shared((m, k), "float16")
            W_shared = T.alloc_shared((k, n), "float16")
            acc = T.alloc_fragment((m, n), "float32")
            z16 = T.alloc_fragment((m, n), "float16")
            y16 = T.alloc_fragment((m, n), "float16")
            T.clear(acc)
            T.copy(X, X_shared)
            T.copy(W, W_shared)
            T.gemm(X_shared, W_shared, acc)
            for i, j in T.Parallel(m, n):
                z16[i, j] = acc[i, j]
            if bias_shape == ():
                if op == "add":
                    for i, j in T.Parallel(m, n):
                        y16[i, j] = z16[i, j] + B[()]
                else:
                    for i, j in T.Parallel(m, n):
                        y16[i, j] = z16[i, j] * B[()]
            else:
                if op == "add":
                    for i, j in T.Parallel(m, n):
                        y16[i, j] = z16[i, j] + B[j]
                else:
                    for i, j in T.Parallel(m, n):
                        y16[i, j] = z16[i, j] * B[j]
            T.copy(y16, Y)
    return kernel


def scalar_add(m=8, k=64, n=33):
    return _tail_kernel(m=m, k=k, n=n, op="add", bias_shape=())


def scalar_mul(m=8, k=64, n=33):
    return _tail_kernel(m=m, k=k, n=n, op="mul", bias_shape=())


def channel_add(m=8, k=64, n=33):
    return _tail_kernel(m=m, k=k, n=n, op="add", bias_shape=(n,))


def channel_mul(m=8, k=64, n=33, bias_n=None):
    """Per-channel multiply; ``bias_n`` may be set to an invalid length in a
    negative fixture where the channel axis does not match the GEMM N extent."""
    length = n if bias_n is None else bias_n
    return _tail_kernel(m=m, k=k, n=n, op="mul", bias_shape=(length,))
