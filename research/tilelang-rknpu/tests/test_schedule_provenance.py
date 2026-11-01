"""F3: where the M/N block decision came from must be recorded, not guessed."""
from __future__ import annotations

import unittest
from dataclasses import replace

from rknpu_backend import (CompileOptions, RK_E_NUMERIC_CONTRACT,
                           RK_E_UNSUPPORTED, compile_rkir, compile_tilelang,
                           default_profile, matmul_module)
from rknpu_backend.codegen.pipeline import normalize_tilelang
from rknpu_backend.errors import BackendError
from rknpu_backend.frontend.tilelang_adapter import tilelang_available

PROFILE = default_profile()


def options(**extra):
    return CompileOptions(numeric_mode="rk_f16_tiled", allow_approx_lut=False,
                          **extra)


def schedule(tile_m, tile_n):
    return {"kind": "mn_tiles_v1", "tile_m": tile_m, "tile_n": tile_n,
            "order": "m_major", "k_tile": 256,
            "accumulation": "ordered_ascending_f16_writeback"}


def rectangles(plan):
    return sorted({(p.rows, p.cols) for p in plan.publish})


class DirectRkirProvenanceTests(unittest.TestCase):
    def test_hand_written_schedule_is_treated_as_a_requirement(self):
        module = matmul_module(m=256, k=256, n=256)
        attrs = dict(module.ops[0].attrs)
        attrs["source_schedule"] = schedule(128, 256)
        attrs["source_k_rounding"] = "explicit_per_k_fp16"
        result = compile_rkir(replace(module,
                                       ops=(replace(module.ops[0], attrs=attrs),)),
                              profile=PROFILE, options=options())
        source = result.report["source"]
        # No frontend stamp: the attribute is the caller's own requirement and is
        # never reported as a frontend derivation.
        self.assertEqual(source["source_schedule_origin"], "caller")
        self.assertEqual(source["mn_grid_source"], "caller")
        self.assertEqual(result.plan.accounting.task_count, 2)
        self.assertEqual(rectangles(result.plan), [(128, 256)])

    def test_default_grid_is_reported_when_nothing_is_declared(self):
        result = compile_rkir(matmul_module(m=256, k=256, n=256),
                              profile=PROFILE, options=options())
        source = result.report["source"]
        self.assertEqual(source["mn_grid_source"], "default_grid")
        self.assertEqual(rectangles(result.plan), [(256, 256)])


@unittest.skipUnless(tilelang_available(),
                     "pinned TileLang optional dependency absent")
class TileLangProvenanceTests(unittest.TestCase):
    def _kernel(self, *, annotate=None, rounding="explicit_per_k_fp16",
                acc16=True, m=256, k=256, n=256, bm=128, bn=256, bk=256):
        import tilelang.language as T
        attrs = {}
        if annotate is not None:
            tile_m, tile_n = annotate
            attrs = {"rknpu.schedule": "mn_tiles_v1",
                     "rknpu.tile_m": tile_m, "rknpu.tile_n": tile_n,
                     "rknpu.mn_order": "m_major"}
            if rounding is not None:
                attrs["rknpu.source_k_rounding"] = rounding

        @T.prim_func
        def gemm(A: T.Tensor((m, k), "float16"),
                 B: T.Tensor((k, n), "float16"),
                 Y: T.Tensor((m, n), "float16")):
            if attrs:
                T.func_attr(attrs)
            with T.Kernel(T.ceildiv(n, bn), T.ceildiv(m, bm),
                          threads=128) as (bx, by):
                As = T.alloc_shared((bm, bk), "float16")
                Bs = T.alloc_shared((bk, bn), "float16")
                Cl = T.alloc_fragment((bm, bn), "float32")
                if acc16:
                    Acc = T.alloc_fragment((bm, bn), "float16")
                    T.clear(Acc)
                    for ko in T.Pipelined(T.ceildiv(k, bk), num_stages=2):
                        T.copy(A[by * bm, ko * bk], As)
                        T.copy(B[ko * bk, bx * bn], Bs)
                        T.copy(Acc, Cl)
                        T.gemm(As, Bs, Cl)
                        T.copy(Cl, Acc)
                    T.copy(Acc, Y[by * bm, bx * bn])
                else:
                    T.clear(Cl)
                    for ko in T.Pipelined(T.ceildiv(k, bk), num_stages=2):
                        T.copy(A[by * bm, ko * bk], As)
                        T.copy(B[ko * bk, bx * bn], Bs)
                        T.gemm(As, Bs, Cl)
                    T.copy(Cl, Y[by * bm, bx * bn])

        return gemm

    def test_derived_geometry_is_recorded_as_derived(self):
        result = compile_tilelang(self._kernel(), profile=PROFILE,
                                  options=options())
        source = result.report["source"]
        self.assertEqual(source["source_schedule_origin"], "derived")
        self.assertEqual(source["mn_grid_source"], "source_geometry")
        self.assertEqual(rectangles(result.plan), [(128, 256)])

    def test_matching_annotation_gives_the_same_plan_and_a_different_origin(self):
        derived = compile_tilelang(self._kernel(), profile=PROFILE,
                                   options=options())
        annotated = compile_tilelang(self._kernel(annotate=(128, 256)),
                                     profile=PROFILE, options=options())
        self.assertEqual(derived.report["source"]["mn_grid_source"],
                         "source_geometry")
        self.assertEqual(annotated.report["source"]["mn_grid_source"],
                         "annotation")
        # Provenance differs, execution does not.
        self.assertEqual(rectangles(derived.plan), rectangles(annotated.plan))
        self.assertEqual(derived.plan.accounting.task_count,
                         annotated.plan.accounting.task_count)
        self.assertEqual(derived.plan.accounting.submit_count,
                         annotated.plan.accounting.submit_count)

    def test_annotation_can_reblock_the_source(self):
        result = compile_tilelang(self._kernel(annotate=(256, 256)),
                                  profile=PROFILE, options=options())
        self.assertEqual(result.report["source"]["mn_grid_source"],
                         "annotation")
        self.assertEqual(rectangles(result.plan), [(256, 256)])
        self.assertEqual(result.plan.accounting.task_count, 1)

    def test_plain_accumulator_uses_the_documented_target_policy(self):
        result = compile_tilelang(self._kernel(acc16=False), profile=PROFILE,
                                  options=options())
        source = result.report["source"]
        self.assertIsNone(source["source_schedule_origin"])
        self.assertEqual(source["mn_grid_source"], "default_grid")
        self.assertEqual(source["source_k_rounding"], "single_k_tile_fp16")
        self.assertEqual(rectangles(result.plan), [(256, 256)])

    def test_annotation_conflicting_with_the_proven_rounding_is_refused(self):
        with self.assertRaises(BackendError) as caught:
            compile_tilelang(self._kernel(annotate=(128, 256),
                                          rounding="fp32_final_narrowing"),
                             profile=PROFILE, options=options())
        self.assertEqual(caught.exception.code, RK_E_NUMERIC_CONTRACT)


class RaggedProvenanceTests(unittest.TestCase):
    @unittest.skipUnless(tilelang_available(), "TileLang absent")
    def test_guarded_tail_without_annotation_keeps_the_target_policy(self):
        from tests.fixtures import tilelang_ragged
        result = compile_tilelang(tilelang_ragged.ragged_tail(),
                                  profile=PROFILE, options=options())
        source = result.report["source"]
        self.assertIsNone(source["source_schedule_origin"])
        self.assertEqual(source["mn_grid_source"], "default_grid")

    @unittest.skipUnless(tilelang_available(), "TileLang absent")
    def test_guarded_tail_with_annotation_reports_annotation(self):
        from tests.fixtures import tilelang_ragged
        result = compile_tilelang(tilelang_ragged.ragged_tail_f16(),
                                  profile=PROFILE, options=options())
        source = result.report["source"]
        self.assertEqual(source["source_schedule_origin"], "annotated")
        self.assertEqual(source["mn_grid_source"], "annotation")


class AdmissionTests(unittest.TestCase):
    """Origin must not decide admission; the checks decide, provenance explains."""

    @staticmethod
    def contract():
        from rknpu_backend import NumericContract
        return NumericContract(
            accumulator_dtype="float16",
            accumulation="ordered_k_tiles_with_f16_writeback",
            narrowing_points=("k_tile_boundary",), approximate=False,
            tolerance={"atol": 8e-3, "rtol": 8e-3}).to_dict()

    def chain(self, origin, tile_m=128, tile_n=256,
              rounding="explicit_per_k_fp16"):
        from rknpu_backend import Module, Op, TensorType, Value
        attrs = {"b_layout": "kn", "numeric_contract": self.contract(),
                 "source_schedule": schedule(tile_m, tile_n),
                 "source_k_rounding": rounding}
        if origin != "absent":
            attrs["source_schedule_origin"] = origin
        values = (Value(0, "X", TensorType("float16", (256, 256)), None, "input"),
                  Value(1, "W1", TensorType("float16", (256, 256)), None, "constant"),
                  Value(2, "Z", TensorType("float16", (256, 256)), 0, "temporary"),
                  Value(3, "W2", TensorType("float16", (256, 256)), None, "constant"),
                  Value(4, "Y", TensorType("float16", (256, 256)), 1, "output"))
        ops = (Op(0, "matmul", (0, 1), (2,), dict(attrs)),
               Op(1, "matmul", (2, 3), (4,), dict(attrs)))
        return Module(name="chain", values=values, ops=ops, inputs=(0,),
                      outputs=(4,), source_hash="admission",
                      profile_hash=PROFILE.profile_hash)

    def compile_chain(self, **kwargs):
        return compile_rkir(self.chain(**kwargs), profile=PROFILE,
                            options=options())

    def test_origin_does_not_change_the_two_node_plan(self):
        plans = {origin: self.compile_chain(origin=origin).plan
                 for origin in ("derived", "annotated", "absent")}
        self.assertEqual(len({plan.plan_hash for plan in plans.values()}), 1)
        for plan in plans.values():
            self.assertEqual(plan.accounting.task_count, 4)
            self.assertEqual(plan.accounting.submit_count, 4)
            self.assertEqual(rectangles(plan), [(128, 256)])

    def test_derived_does_not_bypass_the_two_stage_contract(self):
        for origin in ("derived", "annotated", "absent"):
            with self.subTest(origin=origin):
                with self.assertRaises(BackendError) as caught:
                    self.compile_chain(origin=origin, tile_m=128, tile_n=128)
                self.assertEqual(caught.exception.code, RK_E_UNSUPPORTED)

    def test_unknown_origin_is_refused(self):
        with self.assertRaises(BackendError) as caught:
            self.compile_chain(origin="guess")
        self.assertEqual(caught.exception.code, RK_E_UNSUPPORTED)
        self.assertIn("source_schedule_origin",
                      caught.exception.diagnostics[0].message)

    def test_origin_does_not_bypass_the_numeric_contract(self):
        for origin in ("derived", "annotated", "absent"):
            with self.subTest(origin=origin):
                with self.assertRaises(BackendError) as caught:
                    self.compile_chain(origin=origin,
                                       rounding="fp32_final_narrowing")
                self.assertEqual(caught.exception.code, RK_E_NUMERIC_CONTRACT)

    def test_three_node_dataflow_refuses_a_schedule_with_the_right_words(self):
        from rknpu_backend import Module, Op, TensorType, Value

        def chain3(origin):
            attrs = {"b_layout": "kn", "numeric_contract": self.contract(),
                     "source_schedule": schedule(128, 256),
                     "source_k_rounding": "explicit_per_k_fp16"}
            if origin is not None:
                attrs["source_schedule_origin"] = origin
            values = (
                Value(0, "X", TensorType("float16", (256, 256)), None, "input"),
                Value(1, "W1", TensorType("float16", (256, 256)), None, "constant"),
                Value(2, "Z1", TensorType("float16", (256, 256)), 0, "temporary"),
                Value(3, "W2", TensorType("float16", (256, 256)), None, "constant"),
                Value(4, "Z2", TensorType("float16", (256, 256)), 1, "temporary"),
                Value(5, "W3", TensorType("float16", (256, 256)), None, "constant"),
                Value(6, "Y", TensorType("float16", (256, 256)), 2, "output"))
            ops = (Op(0, "matmul", (0, 1), (2,), dict(attrs)),
                   Op(1, "matmul", (2, 3), (4,), dict(attrs)),
                   Op(2, "matmul", (4, 5), (6,), dict(attrs)))
            return Module(name="chain3", values=values, ops=ops, inputs=(0,),
                          outputs=(6,), source_hash="chain3",
                          profile_hash=PROFILE.profile_hash)

        with self.assertRaises(BackendError) as derived:
            compile_rkir(chain3("derived"), profile=PROFILE,
                         options=options())
        self.assertIn("source proves M/N blocks",
                      derived.exception.diagnostics[0].message)

        with self.assertRaises(BackendError) as annotated:
            compile_rkir(chain3("annotated"), profile=PROFILE,
                         options=options())
        self.assertIn("explicit M/N scheduling",
                      annotated.exception.diagnostics[0].message)

    def test_persistence_keeps_the_adopted_geometry(self):
        import tempfile
        from pathlib import Path as PathType
        from rknpu_backend import load_plan, save_plan
        plan = self.compile_chain(origin="derived").plan
        with tempfile.TemporaryDirectory() as tmp:
            path = PathType(tmp) / "chain.rkplan.json"
            save_plan(plan, str(path))
            loaded = load_plan(str(path))
        self.assertEqual(loaded.plan_hash, plan.plan_hash)
        self.assertEqual(rectangles(loaded), [(128, 256)])
        self.assertEqual(loaded.accounting.task_count, 4)

if __name__ == "__main__":
    unittest.main()