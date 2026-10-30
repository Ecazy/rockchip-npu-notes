"""Direct-RKIR and rejection tests for the minimal M/N schedule contract."""
from __future__ import annotations

import unittest
from dataclasses import replace

from rknpu_backend import (CompileOptions, Module, Op, RK_E_NUMERIC_CONTRACT,
                           RK_E_SHAPE, RK_E_UNSUPPORTED, TensorType, Value,
                           compile_rkir, compile_tilelang, default_profile,
                           matmul_module)
from rknpu_backend.errors import BackendError

PROFILE = default_profile()
OPTIONS = CompileOptions(numeric_mode="rk_f16_tiled")


def schedule(tile_m=256, tile_n=128, order="m_major"):
    return {
        "kind": "mn_tiles_v1", "tile_m": tile_m, "tile_n": tile_n,
        "order": order, "k_tile": 256,
        "accumulation": "ordered_ascending_f16_writeback",
    }


def with_schedule(module, value, rounding="explicit_per_k_fp16"):
    attrs = dict(module.ops[0].attrs)
    attrs["source_schedule"] = value
    if rounding is not None:
        attrs["source_k_rounding"] = rounding
    return replace(module, ops=(replace(module.ops[0], attrs=attrs),))


class ScheduleContractTests(unittest.TestCase):
    def test_direct_rkir_can_express_mn_schedule(self):
        module = matmul_module(m=512, k=512, n=384)
        result = compile_rkir(with_schedule(module, schedule(128, 128)),
                              profile=PROFILE, options=OPTIONS)
        self.assertEqual(result.plan.accounting.task_count, 24)
        first = next(a for a in result.plan.actions
                     if a.template_id == "co.mm.f16.v1")
        self.assertEqual(first.attrs["source_schedule"]["tile_m"], 128)
        self.assertEqual(first.attrs["source_schedule"]["tile_n"], 128)
        self.assertEqual(first.attrs["applied_schedule"]["tile_m"], 128)
        self.assertEqual(first.attrs["applied_schedule"]["tile_n"], 128)
        self.assertEqual(result.report["source"]["source_schedule"],
                         schedule(128, 128))
        self.assertEqual(len(result.report["schedule"]["applied"]), 1)

    def test_direct_rkir_traversal_order_changes_publish_order(self):
        module = matmul_module(m=512, k=512, n=384)
        row = compile_rkir(with_schedule(module, schedule(256, 128, "m_major")),
                           profile=PROFILE, options=OPTIONS).plan
        col = compile_rkir(with_schedule(module, schedule(256, 128, "n_major")),
                           profile=PROFILE, options=OPTIONS).plan
        self.assertNotEqual(row.plan_hash, col.plan_hash)
        self.assertNotEqual([(p.dst_row, p.dst_col) for p in row.publish],
                            [(p.dst_row, p.dst_col) for p in col.publish])

    def test_unknown_schedule_key_is_rejected(self):
        module = matmul_module(m=64, k=256, n=64)
        bad = schedule(64, 64)
        bad["split_k"] = 2
        with self.assertRaises(BackendError) as ctx:
            compile_rkir(with_schedule(module, bad), profile=PROFILE,
                         options=OPTIONS)
        self.assertEqual(ctx.exception.diagnostics[0].code, RK_E_UNSUPPORTED)
        self.assertIn("unsupported or missing fields",
                      ctx.exception.diagnostics[0].message)

    def test_non_256_k_block_is_rejected_even_with_valid_mn_schedule(self):
        module = matmul_module(m=64, k=512, n=64)
        attrs = dict(module.ops[0].attrs)
        attrs["source_schedule"] = schedule(64, 64)
        attrs["source_k_rounding"] = "explicit_per_k_fp16"
        attrs["source_k_loop"] = {"source_k_block": 128,
                                  "lowering_k_tile": 256}
        module = replace(module, ops=(replace(module.ops[0], attrs=attrs),))
        with self.assertRaises(BackendError) as ctx:
            compile_rkir(module, profile=PROFILE, options=OPTIONS)
        self.assertEqual(ctx.exception.diagnostics[0].code, RK_E_UNSUPPORTED)
        self.assertIn("source K block must be 256",
                      ctx.exception.diagnostics[0].message)

    def test_explicit_schedule_on_multi_node_dataflow_is_rejected(self):
        base = matmul_module(m=32, k=64, n=48)
        contract = base.ops[0].attrs["numeric_contract"]
        attrs = {"b_layout": "kn", "numeric_contract": contract,
                 "source_schedule": schedule(32, 48)}
        values = (
            Value(0, "X", TensorType("float16", (32, 64)), None, "input"),
            Value(1, "W1", TensorType("float16", (64, 48)), None, "constant"),
            Value(2, "Z", TensorType("float16", (32, 48)), 0, "temporary"),
            Value(3, "W2", TensorType("float16", (48, 32)), None, "constant"),
            Value(4, "Y", TensorType("float16", (32, 32)), 1, "output"),
        )
        ops = (
            Op(0, "matmul", (0, 1), (2,), dict(attrs)),
            Op(1, "matmul", (2, 3), (4,), dict(attrs)),
        )
        module = Module(name="multi-node-schedule-test", values=values,
                        ops=ops, inputs=(0,), outputs=(4,),
                        source_hash="test")
        with self.assertRaises(BackendError) as ctx:
            compile_rkir(module, profile=PROFILE, options=OPTIONS)
        self.assertEqual(ctx.exception.diagnostics[0].code, RK_E_UNSUPPORTED)
        self.assertIn("multi-node dataflow",
                      ctx.exception.diagnostics[0].message)

    def test_scheduled_legacy_final_narrowing_source_is_rejected(self):
        from models.tilelang_matmul import make_matmul_prim_scheduled
        with self.assertRaises(BackendError) as ctx:
            compile_tilelang(
                make_matmul_prim_scheduled(64, 512, 64, tile_m=64, tile_n=64),
                profile=PROFILE, options=OPTIONS)
        self.assertEqual(ctx.exception.diagnostics[0].code,
                         RK_E_NUMERIC_CONTRACT)
        self.assertIn("K-rounding", ctx.exception.diagnostics[0].message)

    def test_scheduled_explicit_k_rounding_source_is_accepted(self):
        from models.tilelang_matmul import make_matmul_tiled_prim
        result = compile_tilelang(
            make_matmul_tiled_prim(128, 512, 128, bm=64, bn=64),
            profile=PROFILE, options=OPTIONS)
        first = next(a for a in result.plan.actions
                     if a.template_id == "co.mm.f16.v1")
        self.assertEqual(first.attrs["source_k_loop"]["source_rounding"],
                         "explicit_per_k_fp16")
        self.assertEqual(first.attrs["source_schedule"]["tile_m"], 64)

    def test_schedule_without_a_rounding_contract_is_rejected(self):
        module = matmul_module(m=64, k=512, n=64)
        with self.assertRaises(BackendError) as ctx:
            compile_rkir(with_schedule(module, schedule(64, 64),
                                       rounding=None), profile=PROFILE,
                         options=OPTIONS)
        self.assertEqual(ctx.exception.diagnostics[0].code,
                         RK_E_NUMERIC_CONTRACT)
        self.assertIn("K-rounding", ctx.exception.diagnostics[0].message)

    def test_multi_block_schedule_needs_explicit_per_k_rounding(self):
        module = matmul_module(m=64, k=576, n=64)
        attrs = dict(module.ops[0].attrs)
        attrs["source_schedule"] = schedule(64, 64)
        attrs["source_k_rounding"] = "single_k_tile_fp16"
        attrs["source_k_loop"] = {
            "source_k_block": 256,
            "source_k_trips": 3,
            "logical_k": 576,
            "lowering_k_tile": 256,
            "k_blocks": [
                {"index": 0, "k0": 0, "extent": 256},
                {"index": 1, "k0": 256, "extent": 256},
                {"index": 2, "k0": 512, "extent": 64},
            ],
        }
        module = replace(module, ops=(replace(module.ops[0], attrs=attrs),))
        with self.assertRaises(BackendError) as ctx:
            compile_rkir(module, profile=PROFILE, options=OPTIONS)
        self.assertEqual(ctx.exception.diagnostics[0].code,
                         RK_E_NUMERIC_CONTRACT)
        self.assertIn("logical K > 256", ctx.exception.diagnostics[0].message)

    def test_schedule_k_blocks_must_be_contiguous(self):
        module = matmul_module(m=64, k=576, n=64)
        attrs = dict(module.ops[0].attrs)
        attrs["source_schedule"] = schedule(64, 64)
        attrs["source_k_rounding"] = "explicit_per_k_fp16"
        attrs["source_k_loop"] = {
            "source_k_block": 256,
            "source_k_trips": 3,
            "logical_k": 576,
            "lowering_k_tile": 256,
            "k_blocks": [
                {"index": 0, "k0": 0, "extent": 256},
                {"index": 1, "k0": 256, "extent": 256},
                {"index": 2, "k0": 511, "extent": 65},
            ],
        }
        module = replace(module, ops=(replace(module.ops[0], attrs=attrs),))
        with self.assertRaises(BackendError) as ctx:
            compile_rkir(module, profile=PROFILE, options=OPTIONS)
        self.assertEqual(ctx.exception.diagnostics[0].code, RK_E_SHAPE)
        self.assertIn("contiguous ascending",
                      ctx.exception.diagnostics[0].message)

    def test_scheduled_tail_default_matches_legacy_geometry(self):
        from models.tilelang_matmul import (make_matmul_prim,
                                             make_matmul_tiled_tail_prim)
        legacy = compile_tilelang(
            make_matmul_prim(256, 576, 960), profile=PROFILE,
            options=OPTIONS).plan
        scheduled = compile_tilelang(
            make_matmul_tiled_tail_prim(256, 576, 960, bm=256, bn=256),
            profile=PROFILE, options=OPTIONS).plan
        self.assertEqual([(b.name, b.role, tuple(b.shape))
                          for b in legacy.bindings],
                         [(b.name, b.role, tuple(b.shape))
                          for b in scheduled.bindings])
        self.assertEqual([(p.dst_row, p.dst_col, p.rows, p.cols)
                          for p in legacy.publish],
                         [(p.dst_row, p.dst_col, p.rows, p.cols)
                          for p in scheduled.publish])
        self.assertEqual(
            [(a.template_id, a.attrs["k0"], a.attrs["kt"])
             for a in legacy.actions if a.kind == "npu_task"],
            [(a.template_id, a.attrs["k0"], a.attrs["kt"])
             for a in scheduled.actions if a.kind == "npu_task"])
        first = next(a for a in scheduled.actions
                     if a.template_id == "co.mm.f16.v1")
        self.assertEqual(first.attrs["source_k_rounding"],
                         "explicit_per_k_fp16")
        self.assertEqual(first.attrs["source_k_loop"]["k_blocks"], [
            {"index": 0, "k0": 0, "extent": 256},
            {"index": 1, "k0": 256, "extent": 256},
            {"index": 2, "k0": 512, "extent": 64},
        ])

    def test_unsupported_order_is_rejected(self):
        module = matmul_module(m=64, k=256, n=64)
        with self.assertRaises(BackendError) as ctx:
            compile_rkir(with_schedule(module, schedule(64, 64, "row_major")),
                         profile=PROFILE, options=OPTIONS)
        self.assertEqual(ctx.exception.diagnostics[0].code, RK_E_UNSUPPORTED)
        self.assertIn("traversal order", ctx.exception.diagnostics[0].message)


if __name__ == "__main__":
    unittest.main()
