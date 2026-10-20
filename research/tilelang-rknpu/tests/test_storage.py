import unittest

from rknpu_backend import CompileOptions, compile_rkir, default_profile, matmul_module


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


if __name__ == "__main__":
    unittest.main()
