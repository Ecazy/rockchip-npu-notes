# SPDX-License-Identifier: GPL-3.0-or-later
"""Real TileLang SwiGLU FFN fixtures with explicit and implicit narrowing.

The target graph is:

    Zg = X @ Wg       Zu = X @ Wu
    S  = SiLU(Zg)     A  = S * Zu
    H  = A @ Wd       Y  = H + X

Both branch GEMMs share one X tile.  The positive ``swiglu_ffn_f16`` variant
spells out every FP16 narrowing point, including projection -> FP16 *before* the
residual add.  ``swiglu_ffn_fp32_residual`` keeps the residual in the projection's
FP32 accumulator and narrows only in the final ``T.copy``; that is a different
numeric contract and must be refused rather than silently rewritten.
``swiglu_ffn`` is the all-FP32 expression negative control.
"""
import tilelang
import tilelang.language as T


def swiglu_ffn(m=64, d=256, h=128, bk=256):
    """One primitive tile per stage: the smallest honest FFN kernel."""
    @T.prim_func
    def kernel(X: T.Tensor((m, d), "float16"),
               Wg: T.Tensor((d, h), "float16"),
               Wu: T.Tensor((d, h), "float16"),
               Wd: T.Tensor((h, d), "float16"),
               Y: T.Tensor((m, d), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            X_shared = T.alloc_shared((m, bk), "float16")
            Wg_shared = T.alloc_shared((bk, h), "float16")
            Wu_shared = T.alloc_shared((bk, h), "float16")
            Wd_shared = T.alloc_shared((h, d), "float16")
            gate = T.alloc_fragment((m, h), "float32")
            up = T.alloc_fragment((m, h), "float32")
            act = T.alloc_fragment((m, h), "float16")
            proj = T.alloc_fragment((m, d), "float32")
            T.clear(gate)
            T.clear(up)
            T.clear(proj)
            T.copy(X, X_shared)
            T.copy(Wg, Wg_shared)
            T.copy(Wu, Wu_shared)
            T.gemm(X_shared, Wg_shared, gate)
            T.gemm(X_shared, Wu_shared, up)
            for i, j in T.Parallel(m, h):
                act[i, j] = gate[i, j] * T.sigmoid(gate[i, j]) * up[i, j]
            act_shared = T.alloc_shared((m, h), "float16")
            T.copy(act, act_shared)
            T.copy(Wd, Wd_shared)
            T.gemm(act_shared, Wd_shared, proj)
            for i, j in T.Parallel(m, d):
                proj[i, j] = proj[i, j] + X[i, j]
            T.copy(proj, Y)
    return kernel


def swiglu_ffn_fp32_residual(m=64, d=256, h=128, bk=256):
    """Negative: FP32 residual add narrowed only by the final output copy.

    The gate/up/SiLU/product points are explicit FP16, but ``proj`` stays FP32
    through ``proj = proj + X`` and narrows only in ``T.copy(proj, Y)``.  The
    current hardware path cannot preserve that ordering (it would insert a
    projection narrowing before the add), so the frontend must reject it with
    ``RK_E_NUMERIC_CONTRACT``.
    """
    @T.prim_func
    def kernel(X: T.Tensor((m, d), "float16"),
               Wg: T.Tensor((d, h), "float16"),
               Wu: T.Tensor((d, h), "float16"),
               Wd: T.Tensor((h, d), "float16"),
               Y: T.Tensor((m, d), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            X_shared = T.alloc_shared((m, bk), "float16")
            Wg_shared = T.alloc_shared((bk, h), "float16")
            Wu_shared = T.alloc_shared((bk, h), "float16")
            Wd_shared = T.alloc_shared((h, d), "float16")
            act_shared = T.alloc_shared((m, h), "float16")
            gate = T.alloc_fragment((m, h), "float32")
            up = T.alloc_fragment((m, h), "float32")
            gate16 = T.alloc_fragment((m, h), "float16")
            up16 = T.alloc_fragment((m, h), "float16")
            silu16 = T.alloc_fragment((m, h), "float16")
            mul16 = T.alloc_fragment((m, h), "float16")
            proj = T.alloc_fragment((m, d), "float32")
            T.clear(gate)
            T.clear(up)
            T.clear(proj)
            T.copy(X, X_shared)
            T.copy(Wg, Wg_shared)
            T.copy(Wu, Wu_shared)
            T.gemm(X_shared, Wg_shared, gate)
            T.gemm(X_shared, Wu_shared, up)
            # explicit narrowing: gate/up -> FP16
            for i, j in T.Parallel(m, h):
                gate16[i, j] = gate[i, j]
            for i, j in T.Parallel(m, h):
                up16[i, j] = up[i, j]
            # SiLU on the FP16 value, then narrow the product
            for i, j in T.Parallel(m, h):
                silu16[i, j] = gate16[i, j] * T.sigmoid(gate16[i, j])
            for i, j in T.Parallel(m, h):
                mul16[i, j] = silu16[i, j] * up16[i, j]
            T.copy(mul16, act_shared)
            T.copy(Wd, Wd_shared)
            T.gemm(act_shared, Wd_shared, proj)
            for i, j in T.Parallel(m, d):
                proj[i, j] = proj[i, j] + X[i, j]
            T.copy(proj, Y)
    return kernel


def swiglu_ffn_f16(m=64, d=256, h=128, bk=256):
    """Positive staged-FP16 FFN with an explicit residual narrowing point.

    Six explicit FP16 stores:
      1 gate accumulator -> gate16
      2 up accumulator   -> up16
      3 gate16 * sigmoid(gate16) -> silu16
      4 silu16 * up16 -> mul16
      5 projection accumulator -> proj16
      6 proj16 + X -> proj16 (residual)
    The final T.copy(proj16, Y) is therefore type-preserving.
    """
    @T.prim_func
    def kernel(X: T.Tensor((m, d), "float16"),
               Wg: T.Tensor((d, h), "float16"),
               Wu: T.Tensor((d, h), "float16"),
               Wd: T.Tensor((h, d), "float16"),
               Y: T.Tensor((m, d), "float16")):
        with T.Kernel(1, threads=128) as _bx:
            X_shared = T.alloc_shared((m, bk), "float16")
            Wg_shared = T.alloc_shared((bk, h), "float16")
            Wu_shared = T.alloc_shared((bk, h), "float16")
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
                gate16[i, j] = gate32[i, j]                  # 1
            for i, j in T.Parallel(m, h):
                up16[i, j] = up32[i, j]                      # 2
            for i, j in T.Parallel(m, h):
                silu16[i, j] = gate16[i, j] * T.sigmoid(gate16[i, j])  # 3
            for i, j in T.Parallel(m, h):
                mul16[i, j] = silu16[i, j] * up16[i, j]      # 4
            T.copy(mul16, act_shared)
            T.copy(Wd, Wd_shared)
            T.gemm(act_shared, Wd_shared, proj32)
            for i, j in T.Parallel(m, d):
                proj16[i, j] = proj32[i, j]                  # 5
            for i, j in T.Parallel(m, d):
                proj16[i, j] = proj16[i, j] + X[i, j]        # 6
            T.copy(proj16, Y)
    return kernel
