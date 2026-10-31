# SPDX-License-Identifier: GPL-3.0-or-later
"""Scheduled, device-resident SmolLM-shaped SwiGLU FFN.

Logical stages:
  1. packed gate/up projection: Zgu = X @ Wgu
  2. SwiGLU: S=sigmoid(Zgu[:, :H]); Silu=Zgu[:, :H]*S;
     A=Silu*Zgu[:, H:]
  3. down projection + residual: Y = A @ Wd + X
Every stage is a separate T.Kernel launch.  The two GEMMs use guarded 256-wide
K blocks with explicit FP16 accumulator writeback.  The elementwise stages are
separate launches and materialise FP16 intermediates.
"""
import tilelang
import tilelang.language as T


def make_ffn(m=256, d=576, h=1536, bm=256, bn=256, tn=192,
             order0="m_major", order1="m_major", order2="m_major",
             order3="m_major", order4="m_major", order5="m_major"):
    @T.prim_func
    def kernel(H: T.Tensor((m, d), "float16"),
               R: T.Tensor((m, d), "float16"),
               Wgu: T.Tensor((d, 2 * h), "float16"),
               Wd: T.Tensor((h, d), "float16"),
               Zgu: T.Tensor((m, 2 * h), "float16"),
               S: T.Tensor((m, h), "float16"),
               Silu: T.Tensor((m, h), "float16"),
               A: T.Tensor((m, h), "float16"),
               P: T.Tensor((m, d), "float16"),
               Y: T.Tensor((m, d), "float16")):
        T.func_attr({
            "rknpu.staged_schedule": "staged_mn_tiles_v1",
            "rknpu.stage0.tile_m": bm, "rknpu.stage0.tile_n": 256,
            "rknpu.stage0.mn_order": order0,
            "rknpu.stage0.source_k_rounding": "explicit_per_k_fp16",
            "rknpu.stage1.tile_m": bm, "rknpu.stage1.tile_n": bn,
            "rknpu.stage1.mn_order": order1,
            "rknpu.stage2.tile_m": bm, "rknpu.stage2.tile_n": bn,
            "rknpu.stage2.mn_order": order2,
            "rknpu.stage3.tile_m": bm, "rknpu.stage3.tile_n": bn,
            "rknpu.stage3.mn_order": order3,
            "rknpu.stage4.tile_m": bm, "rknpu.stage4.tile_n": tn,
            "rknpu.stage4.mn_order": order4,
            "rknpu.stage4.source_k_rounding": "explicit_per_k_fp16",
            "rknpu.stage5.tile_m": bm, "rknpu.stage5.tile_n": tn,
            "rknpu.stage5.mn_order": order5,
        })

        # stage 0: packed gate/up projection, K=576 tail
        with T.Kernel(T.ceildiv(2 * h, 256), T.ceildiv(m, bm),
                      threads=128) as (nb, mb):
            As = T.alloc_shared((bm, 256), "float16")
            Bs = T.alloc_shared((256, 256), "float16")
            Cl = T.alloc_fragment((bm, 256), "float32")
            Acc = T.alloc_fragment((bm, 256), "float16")
            T.clear(Acc)
            for ko in T.Pipelined(T.ceildiv(d, 256), num_stages=2):
                T.fill(As, 0)
                T.fill(Bs, 0)
                for i, j in T.Parallel(bm, 256):
                    if mb * bm + i < m and ko * 256 + j < d:
                        As[i, j] = H[mb * bm + i, ko * 256 + j]
                for i, j in T.Parallel(256, 256):
                    if ko * 256 + i < d and nb * 256 + j < 2 * h:
                        Bs[i, j] = Wgu[ko * 256 + i, nb * 256 + j]
                T.copy(Acc, Cl)
                T.gemm(As, Bs, Cl)
                T.copy(Cl, Acc)
            for i, j in T.Parallel(bm, 256):
                if mb * bm + i < m and nb * 256 + j < 2 * h:
                    Zgu[mb * bm + i, nb * 256 + j] = Acc[i, j]

        # stage 1: sigmoid(gate half)
        with T.Kernel(T.ceildiv(h, bn), T.ceildiv(m, bm),
                      threads=128) as (nb, mb):
            Zs = T.alloc_shared((bm, bn), "float16")
            Sl = T.alloc_fragment((bm, bn), "float16")
            T.copy(Zgu[mb * bm, nb * bn], Zs)
            for i, j in T.Parallel(bm, bn):
                Sl[i, j] = T.cast(T.sigmoid(Zs[i, j]), "float16")
            T.copy(Sl, S[mb * bm, nb * bn])

        # stage 2: silu(gate) = gate * sigmoid(gate)
        with T.Kernel(T.ceildiv(h, bn), T.ceildiv(m, bm),
                      threads=128) as (nb, mb):
            Zs = T.alloc_shared((bm, bn), "float16")
            Ss = T.alloc_shared((bm, bn), "float16")
            Gl = T.alloc_fragment((bm, bn), "float16")
            T.copy(Zgu[mb * bm, nb * bn], Zs)
            T.copy(S[mb * bm, nb * bn], Ss)
            for i, j in T.Parallel(bm, bn):
                Gl[i, j] = T.cast(Zs[i, j] * Ss[i, j], "float16")
            T.copy(Gl, Silu[mb * bm, nb * bn])

        # stage 3: act = silu(gate) * up half
        with T.Kernel(T.ceildiv(h, bn), T.ceildiv(m, bm),
                      threads=128) as (nb, mb):
            Gs = T.alloc_shared((bm, bn), "float16")
            Us = T.alloc_shared((bm, bn), "float16")
            Al = T.alloc_fragment((bm, bn), "float16")
            T.copy(Silu[mb * bm, nb * bn], Gs)
            T.copy(Zgu[mb * bm, h + nb * bn], Us)
            for i, j in T.Parallel(bm, bn):
                Al[i, j] = T.cast(Gs[i, j] * Us[i, j], "float16")
            T.copy(Al, A[mb * bm, nb * bn])

        # stage 4: down projection, K=1536 exact
        with T.Kernel(T.ceildiv(d, tn), T.ceildiv(m, bm),
                      threads=128) as (nb, mb):
            As = T.alloc_shared((bm, 256), "float16")
            Bs = T.alloc_shared((256, tn), "float16")
            Cl = T.alloc_fragment((bm, tn), "float32")
            Acc = T.alloc_fragment((bm, tn), "float16")
            T.clear(Acc)
            for ko in T.Pipelined(T.ceildiv(h, 256), num_stages=2):
                T.fill(As, 0)
                T.fill(Bs, 0)
                for i, j in T.Parallel(bm, 256):
                    if mb * bm + i < m and ko * 256 + j < h:
                        As[i, j] = A[mb * bm + i, ko * 256 + j]
                for i, j in T.Parallel(256, tn):
                    if ko * 256 + i < h and nb * tn + j < d:
                        Bs[i, j] = Wd[ko * 256 + i, nb * tn + j]
                T.copy(Acc, Cl)
                T.gemm(As, Bs, Cl)
                T.copy(Cl, Acc)
            for i, j in T.Parallel(bm, tn):
                if mb * bm + i < m and nb * tn + j < d:
                    P[mb * bm + i, nb * tn + j] = Acc[i, j]

        # stage 5: residual add
        with T.Kernel(T.ceildiv(d, tn), T.ceildiv(m, bm),
                      threads=128) as (nb, mb):
            Ps = T.alloc_shared((bm, tn), "float16")
            Xs = T.alloc_shared((bm, tn), "float16")
            Yl = T.alloc_fragment((bm, tn), "float16")
            T.copy(P[mb * bm, nb * tn], Ps)
            T.copy(R[mb * bm, nb * tn], Xs)
            for i, j in T.Parallel(bm, tn):
                Yl[i, j] = T.cast(Ps[i, j] + Xs[i, j], "float16")
            T.copy(Yl, Y[mb * bm, nb * tn])

    return kernel


def bad_gate_up_swap(m=256, d=576, h=1536, bm=256, bn=256, tn=192):
    """Negative: act multiplies silu by the gate half instead of the up half."""
    @T.prim_func
    def kernel(H: T.Tensor((m, d), "float16"),
               R: T.Tensor((m, d), "float16"),
               Wgu: T.Tensor((d, 2 * h), "float16"),
               Wd: T.Tensor((h, d), "float16"),
               Zgu: T.Tensor((m, 2 * h), "float16"),
               S: T.Tensor((m, h), "float16"),
               Silu: T.Tensor((m, h), "float16"),
               A: T.Tensor((m, h), "float16"),
               P: T.Tensor((m, d), "float16"),
               Y: T.Tensor((m, d), "float16")):
        T.func_attr({
            "rknpu.staged_schedule": "staged_mn_tiles_v1",
            "rknpu.stage0.tile_m": bm, "rknpu.stage0.tile_n": 256,
            "rknpu.stage0.mn_order": "m_major",
            "rknpu.stage0.source_k_rounding": "explicit_per_k_fp16",
            "rknpu.stage1.tile_m": bm, "rknpu.stage1.tile_n": bn,
            "rknpu.stage2.tile_m": bm, "rknpu.stage2.tile_n": bn,
            "rknpu.stage3.tile_m": bm, "rknpu.stage3.tile_n": bn,
            "rknpu.stage4.tile_m": bm, "rknpu.stage4.tile_n": tn,
            "rknpu.stage4.source_k_rounding": "explicit_per_k_fp16",
            "rknpu.stage5.tile_m": bm, "rknpu.stage5.tile_n": tn,
        })
        # The gate/up tail here is intentionally incomplete: the stage-3 source
        # reads the gate half again.  The frontend must reject the pairing.
        with T.Kernel(T.ceildiv(2 * h, 256), T.ceildiv(m, bm),
                      threads=128) as (nb, mb):
            As = T.alloc_shared((bm, 256), "float16")
            Bs = T.alloc_shared((256, 256), "float16")
            Cl = T.alloc_fragment((bm, 256), "float32")
            Acc = T.alloc_fragment((bm, 256), "float16")
            T.clear(Acc)
            for ko in T.Pipelined(T.ceildiv(d, 256), num_stages=2):
                T.fill(As, 0); T.fill(Bs, 0)
                for i, j in T.Parallel(bm, 256):
                    if mb * bm + i < m and ko * 256 + j < d:
                        As[i, j] = H[mb * bm + i, ko * 256 + j]
                for i, j in T.Parallel(256, 256):
                    if ko * 256 + i < d and nb * 256 + j < 2 * h:
                        Bs[i, j] = Wgu[ko * 256 + i, nb * 256 + j]
                T.copy(Acc, Cl); T.gemm(As, Bs, Cl); T.copy(Cl, Acc)
            for i, j in T.Parallel(bm, 256):
                if mb * bm + i < m and nb * 256 + j < 2 * h:
                    Zgu[mb * bm + i, nb * 256 + j] = Acc[i, j]
        with T.Kernel(T.ceildiv(h, bn), T.ceildiv(m, bm),
                      threads=128) as (nb, mb):
            Zs = T.alloc_shared((bm, bn), "float16")
            Sl = T.alloc_fragment((bm, bn), "float16")
            T.copy(Zgu[mb * bm, nb * bn], Zs)
            for i, j in T.Parallel(bm, bn):
                Sl[i, j] = T.cast(T.sigmoid(Zs[i, j]), "float16")
            T.copy(Sl, S[mb * bm, nb * bn])
        with T.Kernel(T.ceildiv(h, bn), T.ceildiv(m, bm),
                      threads=128) as (nb, mb):
            Zs = T.alloc_shared((bm, bn), "float16")
            Ss = T.alloc_shared((bm, bn), "float16")
            Gl = T.alloc_fragment((bm, bn), "float16")
            T.copy(Zgu[mb * bm, nb * bn], Zs)
            T.copy(S[mb * bm, nb * bn], Ss)
            for i, j in T.Parallel(bm, bn):
                Gl[i, j] = T.cast(Zs[i, j] * Ss[i, j], "float16")
            T.copy(Gl, Silu[mb * bm, nb * bn])
        with T.Kernel(T.ceildiv(h, bn), T.ceildiv(m, bm),
                      threads=128) as (nb, mb):
            Gs = T.alloc_shared((bm, bn), "float16")
            Us = T.alloc_shared((bm, bn), "float16")
            Al = T.alloc_fragment((bm, bn), "float16")
            T.copy(Silu[mb * bm, nb * bn], Gs)
            T.copy(Zgu[mb * bm, nb * bn], Us)
            for i, j in T.Parallel(bm, bn):
                Al[i, j] = T.cast(Gs[i, j] * Us[i, j], "float16")
            T.copy(Al, A[mb * bm, nb * bn])
        # Per-stage schedules are present; the wrong up offset is rejected by
        # the FFN source pairing proof before planning.
        with T.Kernel(T.ceildiv(d, tn), T.ceildiv(m, bm),
                      threads=128) as (nb, mb):
            As = T.alloc_shared((bm, 256), "float16")
            Bs = T.alloc_shared((256, tn), "float16")
            Cl = T.alloc_fragment((bm, tn), "float32")
            Acc = T.alloc_fragment((bm, tn), "float16")
            T.clear(Acc)
            for ko in T.Pipelined(T.ceildiv(h, 256), num_stages=2):
                T.fill(As, 0); T.fill(Bs, 0)
                for i, j in T.Parallel(bm, 256):
                    if mb * bm + i < m and ko * 256 + j < h:
                        As[i, j] = A[mb * bm + i, ko * 256 + j]
                for i, j in T.Parallel(256, tn):
                    if ko * 256 + i < h and nb * tn + j < d:
                        Bs[i, j] = Wd[ko * 256 + i, nb * tn + j]
                T.copy(Acc, Cl); T.gemm(As, Bs, Cl); T.copy(Cl, Acc)
            for i, j in T.Parallel(bm, tn):
                if mb * bm + i < m and nb * tn + j < d:
                    P[mb * bm + i, nb * tn + j] = Acc[i, j]
        with T.Kernel(T.ceildiv(d, tn), T.ceildiv(m, bm),
                      threads=128) as (nb, mb):
            Ps = T.alloc_shared((bm, tn), "float16")
            Xs = T.alloc_shared((bm, tn), "float16")
            Yl = T.alloc_fragment((bm, tn), "float16")
            T.copy(P[mb * bm, nb * tn], Ps)
            T.copy(R[mb * bm, nb * tn], Xs)
            for i, j in T.Parallel(bm, tn):
                Yl[i, j] = T.cast(Ps[i, j] + Xs[i, j], "float16")
            T.copy(Yl, Y[mb * bm, nb * tn])

    return kernel
