# SPDX-License-Identifier: GPL-3.0-or-later
"""Two independent TileLang GEMM launches with a device-resident FP16 Z.

Z = X @ W1, Y = Z @ W2.  Each stage has its own block grid, logical tile and
schedule annotation.  Both stages use guarded rectangular-tail loads and an
explicit FP16 accumulator round trip after every 256-wide K block.
"""
import tilelang
import tilelang.language as T


def make_two_stage(m=65, k=576, h=576, n=97, bm0=64, bn0=256,
                   bm1=32, bn1=64, order0="m_major", order1="n_major",
                   swapped0=False, swapped1=False,
                   src_bm0=None, src_bm1=None,
                   bad_rounding1=False, extra_stage_attr=False):
    sbm0 = bm0 if src_bm0 is None else src_bm0
    sbm1 = bm1 if src_bm1 is None else src_bm1
    attrs = {
        "rknpu.staged_schedule": "staged_mn_tiles_v1",
        "rknpu.stage0.tile_m": bm0,
        "rknpu.stage0.tile_n": bn0,
        "rknpu.stage0.mn_order": order0,
        "rknpu.stage0.source_k_rounding": "explicit_per_k_fp16",
        "rknpu.stage1.tile_m": bm1,
        "rknpu.stage1.tile_n": bn1,
        "rknpu.stage1.mn_order": order1,
        "rknpu.stage1.source_k_rounding": (
            "fp32_final_narrowing" if bad_rounding1
            else "explicit_per_k_fp16"),
    }
    if extra_stage_attr:
        attrs["rknpu.stage2.tile_m"] = 32

    @T.prim_func
    def kernel(X: T.Tensor((m, k), "float16"),
               W1: T.Tensor((k, h), "float16"),
               W2: T.Tensor((h, n), "float16"),
               Z: T.Tensor((m, h), "float16"),
               Y: T.Tensor((m, n), "float16")):
        T.func_attr(attrs)
        if swapped0:
            with T.Kernel(T.ceildiv(m, sbm0), T.ceildiv(h, bn0),
                          threads=128) as (mb0, nb0):
                As = T.alloc_shared((sbm0, 256), "float16")
                Bs = T.alloc_shared((256, bn0), "float16")
                Cl = T.alloc_fragment((sbm0, bn0), "float32")
                Acc = T.alloc_fragment((sbm0, bn0), "float16")
                T.clear(Acc)
                for ko in T.Pipelined(T.ceildiv(k, 256), num_stages=2):
                    T.fill(As, 0)
                    T.fill(Bs, 0)
                    for i, j in T.Parallel(sbm0, 256):
                        if mb0 * sbm0 + i < m and ko * 256 + j < k:
                            As[i, j] = X[mb0 * sbm0 + i, ko * 256 + j]
                    for i, j in T.Parallel(256, bn0):
                        if ko * 256 + i < k and nb0 * bn0 + j < h:
                            Bs[i, j] = W1[ko * 256 + i, nb0 * bn0 + j]
                    T.copy(Acc, Cl)
                    T.gemm(As, Bs, Cl)
                    T.copy(Cl, Acc)
                for i, j in T.Parallel(sbm0, bn0):
                    if mb0 * sbm0 + i < m and nb0 * bn0 + j < h:
                        Z[mb0 * sbm0 + i, nb0 * bn0 + j] = Acc[i, j]
        else:
            with T.Kernel(T.ceildiv(h, bn0), T.ceildiv(m, sbm0),
                          threads=128) as (nb0, mb0):
                As = T.alloc_shared((sbm0, 256), "float16")
                Bs = T.alloc_shared((256, bn0), "float16")
                Cl = T.alloc_fragment((sbm0, bn0), "float32")
                Acc = T.alloc_fragment((sbm0, bn0), "float16")
                T.clear(Acc)
                for ko in T.Pipelined(T.ceildiv(k, 256), num_stages=2):
                    T.fill(As, 0)
                    T.fill(Bs, 0)
                    for i, j in T.Parallel(sbm0, 256):
                        if mb0 * sbm0 + i < m and ko * 256 + j < k:
                            As[i, j] = X[mb0 * sbm0 + i, ko * 256 + j]
                    for i, j in T.Parallel(256, bn0):
                        if ko * 256 + i < k and nb0 * bn0 + j < h:
                            Bs[i, j] = W1[ko * 256 + i, nb0 * bn0 + j]
                    T.copy(Acc, Cl)
                    T.gemm(As, Bs, Cl)
                    T.copy(Cl, Acc)
                for i, j in T.Parallel(sbm0, bn0):
                    if mb0 * sbm0 + i < m and nb0 * bn0 + j < h:
                        Z[mb0 * sbm0 + i, nb0 * bn0 + j] = Acc[i, j]

        if swapped1:
            with T.Kernel(T.ceildiv(m, sbm1), T.ceildiv(n, bn1),
                          threads=128) as (mb1, nb1):
                As = T.alloc_shared((sbm1, 256), "float16")
                Bs = T.alloc_shared((256, bn1), "float16")
                Cl = T.alloc_fragment((sbm1, bn1), "float32")
                Acc = T.alloc_fragment((sbm1, bn1), "float16")
                T.clear(Acc)
                for ko in T.Pipelined(T.ceildiv(h, 256), num_stages=2):
                    T.fill(As, 0)
                    T.fill(Bs, 0)
                    for i, j in T.Parallel(sbm1, 256):
                        if mb1 * sbm1 + i < m and ko * 256 + j < h:
                            As[i, j] = Z[mb1 * sbm1 + i, ko * 256 + j]
                    for i, j in T.Parallel(256, bn1):
                        if ko * 256 + i < h and nb1 * bn1 + j < n:
                            Bs[i, j] = W2[ko * 256 + i, nb1 * bn1 + j]
                    T.copy(Acc, Cl)
                    T.gemm(As, Bs, Cl)
                    T.copy(Cl, Acc)
                for i, j in T.Parallel(sbm1, bn1):
                    if mb1 * sbm1 + i < m and nb1 * bn1 + j < n:
                        Y[mb1 * sbm1 + i, nb1 * bn1 + j] = Acc[i, j]
        else:
            with T.Kernel(T.ceildiv(n, bn1), T.ceildiv(m, sbm1),
                          threads=128) as (nb1, mb1):
                As = T.alloc_shared((sbm1, 256), "float16")
                Bs = T.alloc_shared((256, bn1), "float16")
                Cl = T.alloc_fragment((sbm1, bn1), "float32")
                Acc = T.alloc_fragment((sbm1, bn1), "float16")
                T.clear(Acc)
                for ko in T.Pipelined(T.ceildiv(h, 256), num_stages=2):
                    T.fill(As, 0)
                    T.fill(Bs, 0)
                    for i, j in T.Parallel(sbm1, 256):
                        if mb1 * sbm1 + i < m and ko * 256 + j < h:
                            As[i, j] = Z[mb1 * sbm1 + i, ko * 256 + j]
                    for i, j in T.Parallel(256, bn1):
                        if ko * 256 + i < h and nb1 * bn1 + j < n:
                            Bs[i, j] = W2[ko * 256 + i, nb1 * bn1 + j]
                    T.copy(Acc, Cl)
                    T.gemm(As, Bs, Cl)
                    T.copy(Cl, Acc)
                for i, j in T.Parallel(sbm1, bn1):
                    if mb1 * sbm1 + i < m and nb1 * bn1 + j < n:
                        Y[mb1 * sbm1 + i, nb1 * bn1 + j] = Acc[i, j]

    return kernel


def two_stage_bad_no_intermediate(m=65, k=576, h=576, n=97,
                                  bm=64, bn=256):
    """Negative: both launches read X; no producer supplies stage1's A."""
    @T.prim_func
    def kernel(X: T.Tensor((m, k), "float16"),
               W1: T.Tensor((k, h), "float16"),
               W2: T.Tensor((k, n), "float16"),
               Z: T.Tensor((m, h), "float16"),
               Y: T.Tensor((m, n), "float16")):
        T.func_attr({
            "rknpu.staged_schedule": "staged_mn_tiles_v1",
            "rknpu.stage0.tile_m": bm,
            "rknpu.stage0.tile_n": bn,
            "rknpu.stage0.mn_order": "m_major",
            "rknpu.stage0.source_k_rounding": "explicit_per_k_fp16",
            "rknpu.stage1.tile_m": bm,
            "rknpu.stage1.tile_n": bn,
            "rknpu.stage1.mn_order": "m_major",
            "rknpu.stage1.source_k_rounding": "explicit_per_k_fp16",
        })
        with T.Kernel(T.ceildiv(h, bn), T.ceildiv(m, bm),
                      threads=128) as (nb0, mb0):
            As = T.alloc_shared((bm, 256), "float16")
            Bs = T.alloc_shared((256, bn), "float16")
            Cl = T.alloc_fragment((bm, bn), "float32")
            Acc = T.alloc_fragment((bm, bn), "float16")
            T.clear(Acc)
            for ko in T.Pipelined(T.ceildiv(k, 256), num_stages=2):
                T.fill(As, 0)
                T.fill(Bs, 0)
                for i, j in T.Parallel(bm, 256):
                    if mb0 * bm + i < m and ko * 256 + j < k:
                        As[i, j] = X[mb0 * bm + i, ko * 256 + j]
                for i, j in T.Parallel(256, bn):
                    if ko * 256 + i < k and nb0 * bn + j < h:
                        Bs[i, j] = W1[ko * 256 + i, nb0 * bn + j]
                T.copy(Acc, Cl)
                T.gemm(As, Bs, Cl)
                T.copy(Cl, Acc)
            for i, j in T.Parallel(bm, bn):
                if mb0 * bm + i < m and nb0 * bn + j < h:
                    Z[mb0 * bm + i, nb0 * bn + j] = Acc[i, j]
        with T.Kernel(T.ceildiv(n, bn), T.ceildiv(m, bm),
                      threads=128) as (nb1, mb1):
            As = T.alloc_shared((bm, 256), "float16")
            Bs = T.alloc_shared((256, bn), "float16")
            Cl = T.alloc_fragment((bm, bn), "float32")
            Acc = T.alloc_fragment((bm, bn), "float16")
            T.clear(Acc)
            for ko in T.Pipelined(T.ceildiv(k, 256), num_stages=2):
                T.fill(As, 0)
                T.fill(Bs, 0)
                for i, j in T.Parallel(bm, 256):
                    if mb1 * bm + i < m and ko * 256 + j < k:
                        As[i, j] = X[mb1 * bm + i, ko * 256 + j]
                for i, j in T.Parallel(256, bn):
                    if ko * 256 + i < k and nb1 * bn + j < n:
                        Bs[i, j] = W2[ko * 256 + i, nb1 * bn + j]
                T.copy(Acc, Cl)
                T.gemm(As, Bs, Cl)
                T.copy(Cl, Acc)
            for i, j in T.Parallel(bm, bn):
                if mb1 * bm + i < m and nb1 * bn + j < n:
                    Y[mb1 * bm + i, nb1 * bn + j] = Acc[i, j]

    return kernel
