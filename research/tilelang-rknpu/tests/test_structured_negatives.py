# SPDX-License-Identifier: GPL-3.0-or-later
"""Semantic negatives as independent source files.

Every kernel here is a complete TileLang program written out in full: the
mutations cannot be expressed through factory branches, because the eager builder
resolves a kernel from its own source text.  Each negative changes exactly one
thing relative to the reference kernel in this file.
"""
from __future__ import annotations

import unittest

import tilelang.language as T

from rknpu_backend.errors import BackendError
from rknpu_backend.frontend import structured as S

M, K, N = 256, 576, 256
BM = BN = 256


@T.prim_func
def reference(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"),
              Y: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN), threads=128) as (im, inn):
        As = T.alloc_shared((BM, 256), "float16")
        Bs = T.alloc_shared((256, BN), "float16")
        Cl = T.alloc_fragment((BM, BN), "float32")
        Acc = T.alloc_fragment((BM, BN), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(T.ceildiv(K, 256), num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(BM, 256):
                if im * BM + i < M and ko * 256 + j < K:
                    As[i, j] = A[im * BM + i, ko * 256 + j]
            for i, j in T.Parallel(256, BN):
                if ko * 256 + i < K and inn * BN + j < N:
                    Bs[i, j] = B[ko * 256 + i, inn * BN + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(BM, BN):
            if im * BM + i < M and inn * BN + j < N:
                Y[im * BM + i, inn * BN + j] = Acc[i, j]


@T.prim_func
def clear_after_loop(A: T.Tensor((M, K), "float16"),
                     B: T.Tensor((K, N), "float16"),
                     Y: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN), threads=128) as (im, inn):
        As = T.alloc_shared((BM, 256), "float16")
        Bs = T.alloc_shared((256, BN), "float16")
        Cl = T.alloc_fragment((BM, BN), "float32")
        Acc = T.alloc_fragment((BM, BN), "float16")
        for ko in T.Pipelined(T.ceildiv(K, 256), num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(BM, 256):
                if im * BM + i < M and ko * 256 + j < K:
                    As[i, j] = A[im * BM + i, ko * 256 + j]
            for i, j in T.Parallel(256, BN):
                if ko * 256 + i < K and inn * BN + j < N:
                    Bs[i, j] = B[ko * 256 + i, inn * BN + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        T.clear(Acc)
        for i, j in T.Parallel(BM, BN):
            if im * BM + i < M and inn * BN + j < N:
                Y[im * BM + i, inn * BN + j] = Acc[i, j]


@T.prim_func
def epilogue_first(A: T.Tensor((M, K), "float16"),
                   B: T.Tensor((K, N), "float16"),
                   Y: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN), threads=128) as (im, inn):
        As = T.alloc_shared((BM, 256), "float16")
        Bs = T.alloc_shared((256, BN), "float16")
        Cl = T.alloc_fragment((BM, BN), "float32")
        Acc = T.alloc_fragment((BM, BN), "float16")
        T.clear(Acc)
        for i, j in T.Parallel(BM, BN):
            if im * BM + i < M and inn * BN + j < N:
                Y[im * BM + i, inn * BN + j] = Acc[i, j]
        for ko in T.Pipelined(T.ceildiv(K, 256), num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(BM, 256):
                if im * BM + i < M and ko * 256 + j < K:
                    As[i, j] = A[im * BM + i, ko * 256 + j]
            for i, j in T.Parallel(256, BN):
                if ko * 256 + i < K and inn * BN + j < N:
                    Bs[i, j] = B[ko * 256 + i, inn * BN + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)


@T.prim_func
def loose_guard(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"),
                Y: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN), threads=128) as (im, inn):
        As = T.alloc_shared((BM, 256), "float16")
        Bs = T.alloc_shared((256, BN), "float16")
        Cl = T.alloc_fragment((BM, BN), "float32")
        Acc = T.alloc_fragment((BM, BN), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(T.ceildiv(K, 256), num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(BM, 256):
                if im * BM + i < M and ko * 256 + j < K:
                    As[i, j] = A[im * BM + i, ko * 256 + j]
            for i, j in T.Parallel(256, BN):
                if ko * 256 + i < K and inn * BN + j < N:
                    Bs[i, j] = B[ko * 256 + i, inn * BN + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(BM, BN):
            if im * BM + i < 1024 and inn * BN + j < 4096:
                Y[im * BM + i, inn * BN + j] = Acc[i, j]


@T.prim_func
def gemm_first(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"),
               Y: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN), threads=128) as (im, inn):
        As = T.alloc_shared((BM, 256), "float16")
        Bs = T.alloc_shared((256, BN), "float16")
        Cl = T.alloc_fragment((BM, BN), "float32")
        Acc = T.alloc_fragment((BM, BN), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(T.ceildiv(K, 256), num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            for i, j in T.Parallel(BM, 256):
                if im * BM + i < M and ko * 256 + j < K:
                    As[i, j] = A[im * BM + i, ko * 256 + j]
            for i, j in T.Parallel(256, BN):
                if ko * 256 + i < K and inn * BN + j < N:
                    Bs[i, j] = B[ko * 256 + i, inn * BN + j]
            T.copy(Cl, Acc)
        for i, j in T.Parallel(BM, BN):
            if im * BM + i < M and inn * BN + j < N:
                Y[im * BM + i, inn * BN + j] = Acc[i, j]


@T.prim_func
def else_branch(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"),
                Y: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN), threads=128) as (im, inn):
        As = T.alloc_shared((BM, 256), "float16")
        Bs = T.alloc_shared((256, BN), "float16")
        Cl = T.alloc_fragment((BM, BN), "float32")
        Acc = T.alloc_fragment((BM, BN), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(T.ceildiv(K, 256), num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(BM, 256):
                if im * BM + i < M and ko * 256 + j < K:
                    As[i, j] = A[im * BM + i, ko * 256 + j]
                else:
                    As[i, j] = A[0, ko * 256 + j]
            for i, j in T.Parallel(256, BN):
                if ko * 256 + i < K and inn * BN + j < N:
                    Bs[i, j] = B[ko * 256 + i, inn * BN + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(BM, BN):
            if im * BM + i < M and inn * BN + j < N:
                Y[im * BM + i, inn * BN + j] = Acc[i, j]


@T.prim_func
def extra_output_store(A: T.Tensor((M, K), "float16"),
                       B: T.Tensor((K, N), "float16"),
                       Y: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN), threads=128) as (im, inn):
        As = T.alloc_shared((BM, 256), "float16")
        Bs = T.alloc_shared((256, BN), "float16")
        Cl = T.alloc_fragment((BM, BN), "float32")
        Acc = T.alloc_fragment((BM, BN), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(T.ceildiv(K, 256), num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(BM, 256):
                if im * BM + i < M and ko * 256 + j < K:
                    As[i, j] = A[im * BM + i, ko * 256 + j]
            for i, j in T.Parallel(256, BN):
                if ko * 256 + i < K and inn * BN + j < N:
                    Bs[i, j] = B[ko * 256 + i, inn * BN + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(BM, BN):
            Y[im * BM + i, inn * BN + j] = T.float16(0)
        for i, j in T.Parallel(BM, BN):
            if im * BM + i < M and inn * BN + j < N:
                Y[im * BM + i, inn * BN + j] = Acc[i, j]


@T.prim_func
def write_to_z(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"),
               Z: T.Tensor((M, N), "float16"), Y: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN), threads=128) as (im, inn):
        As = T.alloc_shared((BM, 256), "float16")
        Bs = T.alloc_shared((256, BN), "float16")
        Cl = T.alloc_fragment((BM, BN), "float32")
        Acc = T.alloc_fragment((BM, BN), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(T.ceildiv(K, 256), num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(BM, 256):
                if im * BM + i < M and ko * 256 + j < K:
                    As[i, j] = A[im * BM + i, ko * 256 + j]
            for i, j in T.Parallel(256, BN):
                if ko * 256 + i < K and inn * BN + j < N:
                    Bs[i, j] = B[ko * 256 + i, inn * BN + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(BM, BN):
            if im * BM + i < M and inn * BN + j < N:
                Z[im * BM + i, inn * BN + j] = Acc[i, j]


@T.prim_func
def non_square(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"),
               Y: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(M, 128), T.ceildiv(N, BN), threads=128) as (im, inn):
        As = T.alloc_shared((128, 256), "float16")
        Bs = T.alloc_shared((256, BN), "float16")
        Cl = T.alloc_fragment((128, BN), "float32")
        Acc = T.alloc_fragment((128, BN), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(T.ceildiv(K, 256), num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(128, 256):
                if im * 128 + i < M and ko * 256 + j < K:
                    As[i, j] = A[im * 128 + i, ko * 256 + j]
            for i, j in T.Parallel(256, BN):
                if ko * 256 + i < K and inn * BN + j < N:
                    Bs[i, j] = B[ko * 256 + i, inn * BN + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(128, BN):
            if im * 128 + i < M and inn * BN + j < N:
                Y[im * 128 + i, inn * BN + j] = Acc[i, j]


class ReferenceTests(unittest.TestCase):
    def test_the_reference_kernel_is_proved(self):
        module = S.lower(reference)
        self.assertEqual(module.rounding, "explicit_per_k_fp16")
        self.assertEqual([(s.k0, s.extent) for s in module.slices],
                         [(0, 256), (256, 256), (512, 64)])
        self.assertEqual(module.grid, (1, 1))

    def test_a_non_square_tile_is_accepted(self):
        module = S.lower(non_square)
        self.assertEqual(module.tile_shape, (128, 256))
        self.assertEqual(module.grid, (2, 1))
        self.assertEqual([(t.valid_rows, t.valid_cols) for t in module.tiles],
                         [(128, 256), (128, 256)])


class MutationTests(unittest.TestCase):
    def _rejects(self, kernel, needle=None):
        with self.assertRaises(BackendError) as caught:
            S.lower(kernel)
        message = caught.exception.diagnostics[0].message
        if needle is not None:
            self.assertIn(needle, message)
        return message

    def test_clear_after_loop(self):
        self._rejects(clear_after_loop, "cleared before the K loop")

    def test_epilogue_before_loop(self):
        self._rejects(epilogue_first, "output store must follow")

    def test_loose_output_guard(self):
        self._rejects(loose_guard, "not the logical extent")

    def test_gemm_before_copies(self):
        self._rejects(gemm_first, "must precede the tile GEMM")

    def test_else_branch_in_a_guarded_copy(self):
        self._rejects(else_branch, "else branch")

    def test_extra_unguarded_output_store(self):
        self._rejects(extra_output_store)

    def test_writing_a_non_output_parameter(self):
        self._rejects(write_to_z)




@T.prim_func
def guard_on_loop_vars(A: T.Tensor((M, K), "float16"),
                       B: T.Tensor((K, N), "float16"),
                       Y: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN), threads=128) as (im, inn):
        As = T.alloc_shared((BM, 256), "float16")
        Bs = T.alloc_shared((256, BN), "float16")
        Cl = T.alloc_fragment((BM, BN), "float32")
        Acc = T.alloc_fragment((BM, BN), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(T.ceildiv(K, 256), num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(BM, 256):
                if im * BM + i < M and ko * 256 + j < K:
                    As[i, j] = A[im * BM + i, ko * 256 + j]
            for i, j in T.Parallel(256, BN):
                if ko * 256 + i < K and inn * BN + j < N:
                    Bs[i, j] = B[ko * 256 + i, inn * BN + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(BM, BN):
            if i < BM and j < BN:
                Y[im * BM + i, inn * BN + j] = Acc[i, j]


@T.prim_func
def sibling_shared_store(A: T.Tensor((M, K), "float16"),
                         B: T.Tensor((K, N), "float16"),
                         Y: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN), threads=128) as (im, inn):
        As = T.alloc_shared((BM, 256), "float16")
        Bs = T.alloc_shared((256, BN), "float16")
        Cl = T.alloc_fragment((BM, BN), "float32")
        Acc = T.alloc_fragment((BM, BN), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(T.ceildiv(K, 256), num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(BM, 256):
                if im * BM + i < M and ko * 256 + j < K:
                    As[i, j] = A[im * BM + i, ko * 256 + j]
                As[i, j] = T.float16(0)
            for i, j in T.Parallel(256, BN):
                if ko * 256 + i < K and inn * BN + j < N:
                    Bs[i, j] = B[ko * 256 + i, inn * BN + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(BM, BN):
            if im * BM + i < M and inn * BN + j < N:
                Y[im * BM + i, inn * BN + j] = Acc[i, j]


class MoreMutationTests(unittest.TestCase):
    def test_a_guard_without_the_tile_offset_is_refused(self):
        with self.assertRaises(BackendError) as caught:
            S.lower(guard_on_loop_vars)
        self.assertIn("does not bound the axis it stores",
                      caught.exception.diagnostics[0].message)

    def test_a_sibling_shared_store_is_refused(self):
        with self.assertRaises(BackendError) as caught:
            S.lower(sibling_shared_store)
        self.assertIn("the one guarded copy of a tile",
                      caught.exception.diagnostics[0].message)




M_WIDE, N_WIDE = 256, 128


@T.prim_func
def copy_value_plus_one(A: T.Tensor((M, K), "float16"),
                        B: T.Tensor((K, N), "float16"),
                        Y: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN), threads=128) as (im, inn):
        As = T.alloc_shared((BM, 256), "float16")
        Bs = T.alloc_shared((256, BN), "float16")
        Cl = T.alloc_fragment((BM, BN), "float32")
        Acc = T.alloc_fragment((BM, BN), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(T.ceildiv(K, 256), num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(BM, 256):
                if im * BM + i < M and ko * 256 + j < K:
                    As[i, j] = A[im * BM + i, ko * 256 + j] + T.float16(1)
            for i, j in T.Parallel(256, BN):
                if ko * 256 + i < K and inn * BN + j < N:
                    Bs[i, j] = B[ko * 256 + i, inn * BN + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(BM, BN):
            if im * BM + i < M and inn * BN + j < N:
                Y[im * BM + i, inn * BN + j] = Acc[i, j]


@T.prim_func
def copy_value_twice(A: T.Tensor((M, K), "float16"),
                     B: T.Tensor((K, N), "float16"),
                     Y: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN), threads=128) as (im, inn):
        As = T.alloc_shared((BM, 256), "float16")
        Bs = T.alloc_shared((256, BN), "float16")
        Cl = T.alloc_fragment((BM, BN), "float32")
        Acc = T.alloc_fragment((BM, BN), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(T.ceildiv(K, 256), num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(BM, 256):
                if im * BM + i < M and ko * 256 + j < K:
                    As[i, j] = A[im * BM + i, ko * 256 + j] + A[im * BM + i, ko * 256 + j]
            for i, j in T.Parallel(256, BN):
                if ko * 256 + i < K and inn * BN + j < N:
                    Bs[i, j] = B[ko * 256 + i, inn * BN + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(BM, BN):
            if im * BM + i < M and inn * BN + j < N:
                Y[im * BM + i, inn * BN + j] = Acc[i, j]


@T.prim_func
def output_value_doubled(A: T.Tensor((M, K), "float16"),
                         B: T.Tensor((K, N), "float16"),
                         Y: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN), threads=128) as (im, inn):
        As = T.alloc_shared((BM, 256), "float16")
        Bs = T.alloc_shared((256, BN), "float16")
        Cl = T.alloc_fragment((BM, BN), "float32")
        Acc = T.alloc_fragment((BM, BN), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(T.ceildiv(K, 256), num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(BM, 256):
                if im * BM + i < M and ko * 256 + j < K:
                    As[i, j] = A[im * BM + i, ko * 256 + j]
            for i, j in T.Parallel(256, BN):
                if ko * 256 + i < K and inn * BN + j < N:
                    Bs[i, j] = B[ko * 256 + i, inn * BN + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(BM, BN):
            if im * BM + i < M and inn * BN + j < N:
                Y[im * BM + i, inn * BN + j] = Acc[i, j] * T.float16(2)


@T.prim_func
def partial_operand_copy(A: T.Tensor((M, K), "float16"),
                         B: T.Tensor((K, N), "float16"),
                         Y: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN), threads=128) as (im, inn):
        As = T.alloc_shared((BM, 256), "float16")
        Bs = T.alloc_shared((256, BN), "float16")
        Cl = T.alloc_fragment((BM, BN), "float32")
        Acc = T.alloc_fragment((BM, BN), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(T.ceildiv(K, 256), num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(BM, 128):
                if im * BM + i < M and ko * 256 + j < K:
                    As[i, j] = A[im * BM + i, ko * 256 + j]
            for i, j in T.Parallel(256, BN):
                if ko * 256 + i < K and inn * BN + j < N:
                    Bs[i, j] = B[ko * 256 + i, inn * BN + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(BM, BN):
            if im * BM + i < M and inn * BN + j < N:
                Y[im * BM + i, inn * BN + j] = Acc[i, j]


@T.prim_func
def partial_epilogue(A: T.Tensor((M, K), "float16"),
                     B: T.Tensor((K, N), "float16"),
                     Y: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN), threads=128) as (im, inn):
        As = T.alloc_shared((BM, 256), "float16")
        Bs = T.alloc_shared((256, BN), "float16")
        Cl = T.alloc_fragment((BM, BN), "float32")
        Acc = T.alloc_fragment((BM, BN), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(T.ceildiv(K, 256), num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(BM, 256):
                if im * BM + i < M and ko * 256 + j < K:
                    As[i, j] = A[im * BM + i, ko * 256 + j]
            for i, j in T.Parallel(256, BN):
                if ko * 256 + i < K and inn * BN + j < N:
                    Bs[i, j] = B[ko * 256 + i, inn * BN + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(128, BN):
            if im * BM + i < M and inn * BN + j < N:
                Y[im * BM + i, inn * BN + j] = Acc[i, j]


@T.prim_func
def local_sibling_store(A: T.Tensor((M, K), "float16"),
                        B: T.Tensor((K, N), "float16"),
                        Y: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN), threads=128) as (im, inn):
        As = T.alloc_shared((BM, 256), "float16")
        Bs = T.alloc_shared((256, BN), "float16")
        Cl = T.alloc_fragment((BM, BN), "float32")
        Acc = T.alloc_fragment((BM, BN), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(T.ceildiv(K, 256), num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(BM, 256):
                if im * BM + i < M and ko * 256 + j < K:
                    As[i, j] = A[im * BM + i, ko * 256 + j]
            Cl[0, 0] = T.float32(0)
            for i, j in T.Parallel(256, BN):
                if ko * 256 + i < K and inn * BN + j < N:
                    Bs[i, j] = B[ko * 256 + i, inn * BN + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(BM, BN):
            if im * BM + i < M and inn * BN + j < N:
                Y[im * BM + i, inn * BN + j] = Acc[i, j]


@T.prim_func
def store_into_a(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"),
                 Y: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN), threads=128) as (im, inn):
        As = T.alloc_shared((BM, 256), "float16")
        Bs = T.alloc_shared((256, BN), "float16")
        Cl = T.alloc_fragment((BM, BN), "float32")
        Acc = T.alloc_fragment((BM, BN), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(T.ceildiv(K, 256), num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(BM, 256):
                if im * BM + i < M and ko * 256 + j < K:
                    As[i, j] = A[im * BM + i, ko * 256 + j]
            for i, j in T.Parallel(256, BN):
                if ko * 256 + i < K and inn * BN + j < N:
                    Bs[i, j] = B[ko * 256 + i, inn * BN + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(BM, BN):
            if im * BM + i < M and inn * BN + j < N:
                A[im * BM + i, inn * BN + j] = Acc[i, j]


@T.prim_func
def reordered_guard(A: T.Tensor((M_WIDE, K), "float16"),
                    B: T.Tensor((K, N_WIDE), "float16"),
                    Y: T.Tensor((M_WIDE, N_WIDE), "float16")):
    with T.Kernel(T.ceildiv(M_WIDE, 256), T.ceildiv(N_WIDE, 128),
                  threads=128) as (im, inn):
        As = T.alloc_shared((256, 256), "float16")
        Bs = T.alloc_shared((256, 128), "float16")
        Cl = T.alloc_fragment((256, 128), "float32")
        Acc = T.alloc_fragment((256, 128), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(T.ceildiv(K, 256), num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(256, 256):
                if im * 256 + i < M_WIDE and ko * 256 + j < K:
                    As[i, j] = A[im * 256 + i, ko * 256 + j]
            for i, j in T.Parallel(256, 128):
                if ko * 256 + i < K and inn * 128 + j < N_WIDE:
                    Bs[i, j] = B[ko * 256 + i, inn * 128 + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(256, 128):
            if inn * 128 + j < N_WIDE and im * 256 + i < M_WIDE:
                Y[im * 256 + i, inn * 128 + j] = Acc[i, j]


@T.prim_func
def bs_first(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"),
             Y: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(M, 128), T.ceildiv(N, BN), threads=128) as (im, inn):
        Bs = T.alloc_shared((256, BN), "float16")
        As = T.alloc_shared((128, 256), "float16")
        Cl = T.alloc_fragment((128, BN), "float32")
        Acc = T.alloc_fragment((128, BN), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(T.ceildiv(K, 256), num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(128, 256):
                if im * 128 + i < M and ko * 256 + j < K:
                    As[i, j] = A[im * 128 + i, ko * 256 + j]
            for i, j in T.Parallel(256, BN):
                if ko * 256 + i < K and inn * BN + j < N:
                    Bs[i, j] = B[ko * 256 + i, inn * BN + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(128, BN):
            if im * 128 + i < M and inn * BN + j < N:
                Y[im * 128 + i, inn * BN + j] = Acc[i, j]


class ValueAndCoverageTests(unittest.TestCase):
    def _rejects(self, kernel, needle):
        with self.assertRaises(BackendError) as caught:
            S.lower(kernel)
        message = caught.exception.diagnostics[0].message
        self.assertIn(needle, message)
        return message

    def test_operand_value_arithmetic_is_refused(self):
        self._rejects(copy_value_plus_one, "exactly the operand read")

    def test_operand_value_read_twice_is_refused(self):
        self._rejects(copy_value_twice, "exactly the operand read")

    def test_output_value_arithmetic_is_refused(self):
        self._rejects(output_value_doubled, "accumulator value itself")

    def test_a_partial_operand_copy_is_refused(self):
        self._rejects(partial_operand_copy, "iterate the whole tile")

    def test_a_partial_epilogue_is_refused(self):
        self._rejects(partial_epilogue, "epilogue must iterate the whole tile")

    def test_a_local_sibling_store_is_refused(self):
        with self.assertRaises(BackendError) as caught:
            S.lower(local_sibling_store)
        message = caught.exception.diagnostics[0].message
        self.assertTrue("one guarded copy" in message or "not consume" in message,
                        message)

    def test_storing_into_a_non_output_parameter_is_refused(self):
        self._rejects(store_into_a, "does not write the output parameter")


class PrecisionTests(unittest.TestCase):
    def test_reordered_guard_conjuncts_are_accepted(self):
        module = S.lower(reordered_guard)
        self.assertEqual(module.tile_shape, (256, 128))
        self.assertEqual((module.logical_m, module.logical_n),
                         (M_WIDE, N_WIDE))

    def test_declaration_order_of_the_tiles_does_not_matter(self):
        module = S.lower(bs_first)
        self.assertEqual(module.tile_shape, (128, 256))
        self.assertEqual(module.grid, (2, 1))


class PlanProvenanceTests(unittest.TestCase):
    def test_two_different_sources_do_not_share_a_plan(self):
        from rknpu_backend import CompileOptions, default_profile

        options = CompileOptions(numeric_mode="rk_f16_tiled",
                                 allow_approx_lut=False)
        good = S.plan_from_module(S.lower(reference), profile=default_profile(),
                                  options=options)
        other = S.plan_from_module(S.lower(bs_first), profile=default_profile(),
                                   options=options)
        self.assertNotEqual(good.plan_hash, other.plan_hash)
        self.assertIn(S.lower(reference).source_digest, good.name)




@T.prim_func
def unguarded_epilogue(A: T.Tensor((M, K), "float16"),
                       B: T.Tensor((K, N), "float16"),
                       Y: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN), threads=128) as (im, inn):
        As = T.alloc_shared((BM, 256), "float16")
        Bs = T.alloc_shared((256, BN), "float16")
        Cl = T.alloc_fragment((BM, BN), "float32")
        Acc = T.alloc_fragment((BM, BN), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(T.ceildiv(K, 256), num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(BM, 256):
                if im * BM + i < M and ko * 256 + j < K:
                    As[i, j] = A[im * BM + i, ko * 256 + j]
            for i, j in T.Parallel(256, BN):
                if ko * 256 + i < K and inn * BN + j < N:
                    Bs[i, j] = B[ko * 256 + i, inn * BN + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(BM, BN):
            Y[im * BM + i, inn * BN + j] = Acc[i, j]


@T.prim_func
def root_else_branch(A: T.Tensor((M, K), "float16"),
                     B: T.Tensor((K, N), "float16"),
                     Y: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN), threads=128) as (im, inn):
        As = T.alloc_shared((BM, 256), "float16")
        Bs = T.alloc_shared((256, BN), "float16")
        Cl = T.alloc_fragment((BM, BN), "float32")
        Acc = T.alloc_fragment((BM, BN), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(T.ceildiv(K, 256), num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(BM, 256):
                if im * BM + i < M and ko * 256 + j < K:
                    As[i, j] = A[im * BM + i, ko * 256 + j]
            for i, j in T.Parallel(256, BN):
                if ko * 256 + i < K and inn * BN + j < N:
                    Bs[i, j] = B[ko * 256 + i, inn * BN + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(BM, BN):
            if im * BM + i < M and inn * BN + j < N:
                Y[im * BM + i, inn * BN + j] = Acc[i, j]
            else:
                Y[im * BM + i, inn * BN + j] = T.float16(0)


class Precision2Tests(unittest.TestCase):
    def test_an_unguarded_epilogue_is_accepted(self):
        module = S.lower(unguarded_epilogue)
        self.assertEqual(module.grid, (1, 1))
        self.assertEqual([(s.k0, s.extent) for s in module.slices],
                         [(0, 256), (256, 256), (512, 64)])

    def test_a_root_else_branch_is_refused_with_its_own_message(self):
        with self.assertRaises(BackendError) as caught:
            S.lower(root_else_branch)
        self.assertIn("else branch", caught.exception.diagnostics[0].message)




@T.prim_func
def extra_k_trip(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"),
                 Y: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN), threads=128) as (im, inn):
        As = T.alloc_shared((BM, 256), "float16")
        Bs = T.alloc_shared((256, BN), "float16")
        Cl = T.alloc_fragment((BM, BN), "float32")
        Acc = T.alloc_fragment((BM, BN), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(T.ceildiv(1024, 256), num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(BM, 256):
                if im * BM + i < M and ko * 256 + j < K:
                    As[i, j] = A[im * BM + i, ko * 256 + j]
            for i, j in T.Parallel(256, BN):
                if ko * 256 + i < K and inn * BN + j < N:
                    Bs[i, j] = B[ko * 256 + i, inn * BN + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(BM, BN):
            if im * BM + i < M and inn * BN + j < N:
                Y[im * BM + i, inn * BN + j] = Acc[i, j]


@T.prim_func
def swapped_axes_square(A: T.Tensor((M, K), "float16"),
                        B: T.Tensor((K, N), "float16"),
                        Y: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN), threads=128) as (im, inn):
        As = T.alloc_shared((BM, 256), "float16")
        Bs = T.alloc_shared((256, BN), "float16")
        Cl = T.alloc_fragment((BM, BN), "float32")
        Acc = T.alloc_fragment((BM, BN), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(T.ceildiv(K, 256), num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(BM, 256):
                if im * BM + i < M and ko * 256 + j < K:
                    As[i, j] = A[im * BM + i, ko * 256 + j]
            for i, j in T.Parallel(256, BN):
                if ko * 256 + i < K and inn * BN + j < N:
                    Bs[i, j] = B[ko * 256 + i, inn * BN + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(BM, BN):
            if inn * BN + j < M and im * BM + i < N:
                Y[inn * BN + j, im * BM + i] = Acc[i, j]


class SliceAndAxisTests(unittest.TestCase):
    def test_an_extra_k_trip_is_refused(self):
        with self.assertRaises(BackendError):
            S.lower(extra_k_trip)

    def test_swapped_output_axes_on_a_square_output_are_refused(self):
        with self.assertRaises(BackendError) as caught:
            S.lower(swapped_axes_square)
        message = caught.exception.diagnostics[0].message
        self.assertTrue("grid axis" in message or "tile-local" in message
                        or "one tile coordinate" in message,
                        message)


if __name__ == "__main__":
    unittest.main()
