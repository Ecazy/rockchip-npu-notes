# SPDX-License-Identifier: GPL-3.0-or-later
"""Real TileLang PrimFunc fixture: one primitive-sized FP16 MatMul tile.

Run with the pinned optional environment:
  python -m rknpu_backend.cli compile-tilelang --file examples/matmul.py \
    --symbol kernel --numeric-mode rk_f16_tiled --out /tmp/matmul.rkplan.json
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
            T.copy(C_local, C)
    return kernel


kernel = make_kernel()
