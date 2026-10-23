import unittest

from rknpu_backend import (CompileOptions, matmul_module, compile_rkir,
                           default_profile)
from rknpu_backend.errors import BackendError, RK_E_ALIAS


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.profile = default_profile()

    def _plan(self, **kwargs):
        return compile_rkir(
            matmul_module(m=13, k=257, n=33, activation="silu"),
            profile=self.profile,
            options=CompileOptions(numeric_mode="rk_f16_tiled",
                                   allow_approx_lut=True, **kwargs)).plan

    def test_spec_example_counts_and_budget(self):
        plan = self._plan()
        self.assertEqual(plan.accounting.task_count, 4)
        self.assertEqual(len(plan.submits), 2)
        self.assertEqual(plan.accounting.data_bo_count, 8)
        self.assertEqual(plan.accounting.peak_payload_bytes, 60416)
        self.assertEqual(plan.accounting.reserved_bytes, 323072)

    def test_accumulator_ping_pong_output_reuses_free_slot(self):
        plan = self._plan()
        accs = [b for b in plan.buffers if b.role == "accumulator"]
        self.assertEqual(len(accs), 2)
        self.assertNotEqual(accs[0].slot_id, accs[1].slot_id)
        out = [b for b in plan.buffers if b.role == "output"][0]
        # P's last use is before Y's first write, so the spec's first-fit rule
        # reuses P's slot for Y while Z remains a different BO.
        self.assertEqual(out.slot_id, accs[0].slot_id)
        self.assertNotEqual(out.slot_id, accs[1].slot_id)

    def test_separate_mode_has_four_submits(self):
        self.assertEqual(len(self._plan(submit_mode="separate").submits), 4)


class ConstantLifetimeTests(unittest.TestCase):
    def test_constant_slot_is_exclusive(self):
        from dataclasses import replace
        from rknpu_backend.plan import Plan

        plan = compile_rkir(
            matmul_module(m=13, k=257, n=33, activation="silu"),
            profile=default_profile(),
            options=CompileOptions(numeric_mode="rk_f16_tiled",
                                   allow_approx_lut=True)).plan
        weight = next(b for b in plan.buffers if b.role == "weight")
        output = next(b for b in plan.buffers if b.role == "output")
        buffers = [replace(b, slot_id=weight.slot_id) if b.id == output.id else b
                   for b in plan.buffers]
        with self.assertRaises(BackendError) as ctx:
            Plan(plan.name, plan.profile_hash, plan.source_hash, plan.bindings,
                 plan.constants, tuple(buffers), plan.views, plan.actions,
                 plan.submits, plan.publish, plan.accounting)
        self.assertEqual(ctx.exception.code, RK_E_ALIAS)


class MultitileTests(unittest.TestCase):
    """M/N above one tile must publish per-tile rectangles that cover the output."""

    def setUp(self):
        self.profile = default_profile()
        self.options = CompileOptions(numeric_mode="rk_f16_tiled",
                                      allow_approx_lut=True)

    def _plan(self, m, n):
        return compile_rkir(matmul_module(m=m, k=64, n=n),
                            profile=self.profile,
                            options=self.options).plan

    def test_m_tiles_publish_two_rectangles(self):
        plan = self._plan(257, 33)
        tiles = sorted((p.dst_row, p.dst_col, p.rows, p.cols)
                       for p in plan.publish)
        self.assertEqual(tiles, [(0, 0, 256, 33), (256, 0, 1, 33)])
        self.assertEqual(sum(r * c for _, _, r, c in tiles), 257 * 33)

    def test_n_tiles_publish_two_rectangles(self):
        plan = self._plan(13, 257)
        tiles = sorted((p.dst_row, p.dst_col, p.rows, p.cols)
                       for p in plan.publish)
        self.assertEqual(tiles, [(0, 0, 13, 256), (0, 256, 13, 1)])

    def test_four_tiles_cover_the_binding(self):
        plan = self._plan(257, 257)
        self.assertEqual(len(plan.publish), 4)
        self.assertEqual(sum(p.rows * p.cols for p in plan.publish), 257 * 257)
        # Each output tile is downloaded by exactly one host boundary action.
        downloads = [a for a in plan.actions if a.kind == "download"]
        self.assertEqual(len(downloads), 4)
        self.assertEqual(len({a.inputs[0] for a in downloads}), 4)

    def test_lut_chunk_views_use_byte_offsets(self):
        # 256x256 has 65536 physical elements, so the second chunk must start at
        # byte 65536 rather than element 32768.
        plan = compile_rkir(
            matmul_module(m=256, k=256, n=256, activation="silu"),
            profile=self.profile, options=self.options).plan
        lut = [a for a in plan.actions
               if a.template_id == "co.lut.sigmoid.f16.v1"]
        self.assertEqual(len(lut), 2)
        offsets = sorted(plan.views[a.inputs[0]].offset_bytes for a in lut)
        self.assertEqual(offsets, [0, 65536])


if __name__ == "__main__":
    unittest.main()
