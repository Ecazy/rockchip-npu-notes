# SPDX-License-Identifier: GPL-3.0-or-later
"""Structured lowering: proofs, semantic negatives and plan conformance."""
from __future__ import annotations

import unittest
from pathlib import Path

import tilelang.language as T

from rknpu_backend import CompileOptions, compile_tilelang, default_profile
from rknpu_backend.errors import BackendError
from rknpu_backend.frontend import structured as S

PROFILE = default_profile()
M, K, N = 256, 576, 3072
BM = BN = 256


def _opt():
    return CompileOptions(numeric_mode="rk_f16_tiled", allow_approx_lut=False)


def _kernel(shape_m, shape_k, shape_n, *, guard_a=True, swap_axis=False,
            fill_b=True, writeback=True, rename=False, epilogue="ok",
            fill_order="before", crosswire=False, partial_writeback=False):
    """One guarded-tail GEMM; each keyword breaks exactly one property."""
    ma = "rowtile" if rename else "im"
    na = "coltile" if rename else "inn"

    @T.prim_func
    def kernel(A: T.Tensor((shape_m, shape_k), "float16"),
               B: T.Tensor((shape_k, shape_n), "float16"),
               Y: T.Tensor((shape_m, shape_n), "float16")):
        with T.Kernel(T.ceildiv(shape_m, BM), T.ceildiv(shape_n, BN),
                      threads=128) as (i0, i1):
            im = i1 if swap_axis else i0
            inn = i0 if swap_axis else i1
            As = T.alloc_shared((BM, 256), "float16")
            Bs = T.alloc_shared((256, BN), "float16")
            Cl = T.alloc_fragment((BM, BN), "float32")
            Acc = T.alloc_fragment((BM, BN), "float16")
            T.clear(Acc)
            for ko in T.Pipelined(T.ceildiv(shape_k, 256), num_stages=2):
                if fill_order == "before":
                    T.fill(As, 0)
                    if fill_b:
                        T.fill(Bs, 0)
                for i, j in T.Parallel(BM, 256):
                    if guard_a:
                        if im * BM + i < shape_m and ko * 256 + j < shape_k:
                            As[i, j] = A[im * BM + i, ko * 256 + j]
                    else:
                        As[i, j] = A[im * BM + i, ko * 256 + j]
                for i, j in T.Parallel(256, BN):
                    if ko * 256 + i < shape_k and inn * BN + j < shape_n:
                        if crosswire:
                            Bs[i, j] = A[ko * 256 + i, inn * BN + j]
                        else:
                            Bs[i, j] = B[ko * 256 + i, inn * BN + j]
                if fill_order == "after":
                    T.fill(As, 0)
                    if fill_b:
                        T.fill(Bs, 0)
                T.copy(Acc, Cl)
                T.gemm(As, Bs, Cl)
                if writeback:
                    if partial_writeback:
                        T.copy(Cl[0:128, 0:BN], Acc[0:128, 0:BN])
                    else:
                        T.copy(Cl, Acc)
            if epilogue == "ok":
                for i, j in T.Parallel(BM, BN):
                    if im * BM + i < shape_m and inn * BN + j < shape_n:
                        Y[im * BM + i, inn * BN + j] = Acc[i, j]
            elif epilogue == "transposed":
                for i, j in T.Parallel(BM, BN):
                    if im * BM + i < shape_m and inn * BN + j < shape_n:
                        Y[inn * BN + j, im * BM + i] = Acc[i, j]

    return kernel


def _module(**kwargs):
    return S.lower(_kernel(M, K, N, **kwargs), annotations={"num_stages": 2})


class ProofTests(unittest.TestCase):
    def test_the_pilot_proves_the_expected_structure(self):
        module = _module(rename=True)  # renaming must not matter
        self.assertEqual(module.rounding, "explicit_per_k_fp16")
        self.assertEqual(module.rounding_points, "after_each_k_block")
        self.assertEqual([(s.k0, s.extent) for s in module.slices],
                         [(0, 256), (256, 256), (512, 64)])
        self.assertEqual(module.grid, (1, 12))
        self.assertEqual(module.tile_shape, (256, 256))
        self.assertEqual(module.logical_m, M)
        self.assertEqual(module.logical_n, N)
        self.assertEqual(module.logical_k, K)
        self.assertEqual(len(module.tiles), 12)
        self.assertEqual(len(module.proofs["guarded"]), 2)
        self.assertEqual(len(module.proofs["fills"]), 2)

    def test_a_swapped_grid_proves_the_same_structure(self):
        swapped = _module(swap_axis=True)
        plain = _module()
        self.assertEqual((swapped.grid, swapped.tile_shape, swapped.slices,
                          swapped.rounding),
                         (plain.grid, plain.tile_shape, plain.slices,
                          plain.rounding))

    def test_the_declared_rounding_is_only_cross_checked(self):
        source = _kernel(M, K, N)
        with self.assertRaises(BackendError) as caught:
            S.lower(source, declared_rounding="fp32_final_narrowing")
        self.assertIn("disagrees with the proven rounding",
                      caught.exception.diagnostics[0].message)
        module = S.lower(source, declared_rounding="explicit_per_k_fp16")
        self.assertEqual(module.rounding, "explicit_per_k_fp16")

    def test_k_257_partitions_into_256_plus_1(self):
        module = S.lower(_kernel(256, 257, 256))
        self.assertEqual([(s.k0, s.extent) for s in module.slices],
                         [(0, 256), (256, 1)])
        self.assertEqual(module.logical_k, 257)

    def test_m_tail_is_reported_per_tile(self):
        module = S.lower(_kernel(257, K, 256))
        self.assertEqual(module.grid, (2, 1))
        self.assertEqual([(t.valid_rows, t.valid_cols) for t in module.tiles],
                         [(256, 256), (1, 256)])


class NegativeTests(unittest.TestCase):
    def test_a_missing_operand_guard_is_refused(self):
        with self.assertRaises(BackendError) as caught:
            S.lower(_kernel(M, K, N, guard_a=False))
        self.assertIn("guarded", caught.exception.diagnostics[0].message.lower())

    def test_a_missing_zero_fill_is_refused(self):
        with self.assertRaises(BackendError) as caught:
            S.lower(_kernel(M, K, N, fill_b=False))
        self.assertIn("zero filled", caught.exception.diagnostics[0].message)

    def test_a_missing_writeback_is_refused(self):
        with self.assertRaises(BackendError) as caught:
            S.lower(_kernel(M, K, N, writeback=False))
        self.assertIn("writeback", caught.exception.diagnostics[0].message)

    def test_a_source_that_never_consumes_a_statement_is_refused(self):
        @T.prim_func
        def kernel(A: T.Tensor((256, 576), "float16"),
                   B: T.Tensor((576, 256), "float16"),
                   Y: T.Tensor((256, 256), "float16")):
            with T.Kernel(1, 1, threads=128) as (im, inn):
                As = T.alloc_shared((256, 256), "float16")
                Bs = T.alloc_shared((256, 256), "float16")
                Cl = T.alloc_fragment((256, 256), "float32")
                Acc = T.alloc_fragment((256, 256), "float16")
                T.clear(Acc)
                for ko in T.Pipelined(T.ceildiv(576, 256), num_stages=0):
                    T.fill(As, 0)
                    T.fill(Bs, 0)
                    for i, j in T.Parallel(256, 256):
                        if im * 256 + i < 256 and ko * 256 + j < 576:
                            As[i, j] = A[im * 256 + i, ko * 256 + j]
                    for i, j in T.Parallel(256, 256):
                        if ko * 256 + i < 576 and inn * 256 + j < 256:
                            Bs[i, j] = B[ko * 256 + i, inn * 256 + j]
                    T.copy(Acc, Cl)
                    T.gemm(As, Bs, Cl)
                    T.copy(Cl, Acc)
                    for i, j in T.Parallel(256, 256):
                        if im * 256 + i < 256 and inn * 256 + j < 256:
                            Y[im * 256 + i, inn * 256 + j] = Acc[i, j]

        with self.assertRaises(BackendError) as caught:
            S.lower(kernel)
        message = caught.exception.diagnostics[0].message
        self.assertTrue("does not consume" in message or "guarded copy" in message,
                        message)


class ClosureNegativeTests(unittest.TestCase):
    def test_a_missing_epilogue_is_refused(self):
        with self.assertRaises(BackendError) as caught:
            S.lower(_kernel(M, K, N, epilogue="none"))
        self.assertIn("output parameter", caught.exception.diagnostics[0].message)

    def test_a_transposed_epilogue_is_refused(self):
        with self.assertRaises(BackendError):
            S.lower(_kernel(M, K, N, epilogue="transposed"))

    def test_a_fill_after_the_copy_is_refused(self):
        with self.assertRaises(BackendError) as caught:
            S.lower(_kernel(M, K, N, fill_order="after"))
        self.assertIn("filled after", caught.exception.diagnostics[0].message)

    def test_a_partial_writeback_is_refused(self):
        with self.assertRaises(BackendError) as caught:
            S.lower(_kernel(M, K, N, partial_writeback=True))
        self.assertIn("whole fragment", caught.exception.diagnostics[0].message)

    def test_crosswired_operands_are_refused(self):
        with self.assertRaises(BackendError) as caught:
            S.lower(_kernel(M, K, N, crosswire=True))
        self.assertIn("GEMM operands", caught.exception.diagnostics[0].message)


class PlanConformanceTests(unittest.TestCase):
    def test_the_plan_reproduces_the_proved_structure(self):
        module = _module()
        plan = S.plan_from_module(module, profile=PROFILE, options=_opt())
        tasks = [a for a in plan.actions if a.kind == "npu_task"]
        self.assertEqual(len(tasks), len(module.tiles) * len(module.slices))
        S.verify_plan(module, plan)

    def test_the_plan_matches_the_old_path_field_for_field(self):
        module = _module()
        old = compile_tilelang(_kernel(M, K, N), profile=PROFILE,
                               options=_opt()).plan
        new = S.plan_from_module(module, profile=PROFILE, options=_opt())

        def signature(plan):
            tasks = [a for a in plan.actions if a.kind == "npu_task"]
            return {
                "tasks": [(a.template_id, tuple(a.inputs), tuple(a.outputs),
                           a.attrs.get("k0"), a.attrs.get("submit_group"),
                           len(a.deps)) for a in tasks],
                "submits": [(len(s.task_ids), s.recipe_id, s.mode, s.core_policy)
                            for s in plan.submits],
                "publish": len(plan.publish),
                "constants": [(c.name, str(c.dtype), tuple(c.shape), c.pack_key)
                              for c in plan.constants],
                "device": plan.accounting.device_request_bytes,
                "host": plan.accounting.host_bytes,
                "bo": plan.accounting.data_bo_count,
            }

        self.assertEqual(signature(new), signature(old))

    def test_the_new_plan_passes_the_independent_verifier(self):
        from rknpu_backend.runtime.ffi import LibRKPlan

        lib_path = str(Path(__file__).resolve().parents[1].parent
                       / "custom-ops" / "build-host-mvp" / "librknpu_plan.so")
        if not Path(lib_path).is_file():
            self.skipTest("host plan library not built")
        module = _module()
        plan = S.plan_from_module(module, profile=PROFILE, options=_opt())
        lib = LibRKPlan(lib_path)
        lib.create_plan(plan).destroy()


if __name__ == "__main__":
    unittest.main()
