# SPDX-License-Identifier: GPL-3.0-or-later
"""TileLang scheduling controls: namespace, traversal and the CBUF register.

The backend accepts five scheduling annotations.  Everything else in the
'rknpu.*' namespace is a request it cannot honour and is refused; the reuse flag
is checked where it is actually programmed, in the CBUF_CON0 register word.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from rknpu_backend import CompileOptions, compile_tilelang, default_profile
from rknpu_backend.errors import BackendError
from rknpu_backend.models import PlanCall
from rknpu_backend.runtime.ffi import LibRKPlan
from tests.fixtures.tilelang_linear_scheduled import make_linear

PROFILE = default_profile()
ROOT = Path(__file__).resolve().parents[1]
MOCK = str(ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan_mock.so")


def options(**extra):
    settings = {"numeric_mode": "rk_f16_tiled", "allow_approx_lut": True}
    settings.update(extra)
    return CompileOptions(**settings)


def compile_with(extra_attrs=None, **kwargs):
    func = make_linear(**kwargs)
    for key, value in (extra_attrs or {}).items():
        func = func.with_attr(key, value)
    return compile_tilelang(func, profile=PROFILE, options=options()).plan


def tile_sequence(plan):
    sequence = []
    for action in plan.actions:
        if action.kind != "npu_task":
            continue
        group = action.attrs.get("submit_group")
        if group and (not sequence or sequence[-1] != group):
            sequence.append(group)
    return sequence


class ScheduleNamespaceTests(unittest.TestCase):
    def test_the_declared_annotations_are_accepted(self):
        plan = compile_with(m=128, k=256, n=256, bm=256, bn=256, order="n_major")
        self.assertTrue([a for a in plan.actions if a.kind == "npu_task"])

    def test_unknown_rknpu_annotations_are_refused(self):
        # These used to be ignored because only the schedule/tile_/mn_/k_/source_
        # prefixes were scanned: a request silently became a no-op.
        for key, value in (("rknpu.pipeline_stages", 3),
                           ("rknpu.double_buffer", True),
                           ("rknpu.async_copy", 1),
                           ("rknpu.tile_num", 4)):
            with self.subTest(key=key):
                with self.assertRaises(BackendError) as caught:
                    compile_with({key: value}, m=128, k=256, n=256, bm=256, bn=256)
                diagnostic = caught.exception.diagnostics[0]
                self.assertIn("unsupported TileLang schedule annotation",
                              diagnostic.message)
                self.assertEqual(diagnostic.to_dict().get("actual"), key)

    def test_bad_values_of_declared_annotations_are_refused(self):
        for extra, needle in (({"rknpu.mn_order": "diagonal"},
                               "unsupported M/N traversal order"),
                              ({"rknpu.tile_m": 512}, "within 1..256")):
            with self.subTest(extra=extra):
                with self.assertRaises(BackendError) as caught:
                    compile_with(extra, m=128, k=256, n=256, bm=256, bn=256)
                self.assertIn(needle, caught.exception.diagnostics[0].message)


class TraversalOrderTests(unittest.TestCase):
    def test_mn_order_moves_the_emitted_tile_sequence(self):
        row = compile_with(m=512, k=576, n=512, bm=256, bn=256, order="m_major")
        col = compile_with(m=512, k=576, n=512, bm=256, bn=256, order="n_major")
        self.assertEqual(len(row.submits), len(col.submits))
        # The declared schedule names its tiles by grid index, so m_major visits
        # every N tile of one M tile first and n_major the other way round.
        self.assertEqual(tile_sequence(row),
                         ["mm:0:0", "mm:0:1", "mm:1:0", "mm:1:1"])
        self.assertEqual(tile_sequence(col),
                         ["mm:0:0", "mm:1:0", "mm:0:1", "mm:1:1"])

    def test_tile_size_changes_the_task_grid(self):
        big = compile_with(m=256, k=576, n=512, bm=256, bn=256)
        small = compile_with(m=256, k=576, n=512, bm=128, bn=256)
        tasks_big = sum(1 for a in big.actions if a.kind == "npu_task")
        tasks_small = sum(1 for a in small.actions if a.kind == "npu_task")
        self.assertEqual(tasks_big, 6)
        self.assertEqual(tasks_small, 12)


@unittest.skipUnless(Path(MOCK).is_file(), "mock library not built")
class ReuseRegisterTests(unittest.TestCase):
    """The reuse flag has to appear in the emitted CBUF_CON0 word."""

    def _reuse_words(self, task_order):
        plan = compile_tilelang(make_linear(256, 576, 512, bm=256, bn=256),
                                profile=PROFILE,
                                options=options(linear_task_order=task_order)).plan
        constants = {"B": np.zeros((576, 512), dtype=np.float16)}
        for spec in plan.constants:
            if spec.name in constants:
                continue
            constants[spec.name] = (np.eye(spec.shape[0], dtype=np.float16)
                                    if spec.generator == "identity"
                                    else np.zeros(tuple(spec.shape),
                                                  dtype=np.float16))
        lib = LibRKPlan(MOCK)
        call = PlanCall(lib, plan, constants, fd=0)
        flagged = 0
        declared = 0
        try:
            for action in plan.actions:
                if action.kind != "npu_task":
                    continue
                if action.attrs.get("data_reuse"):
                    declared += 1
                info, _ = lib.task_cbuf(call.workspace, action.id, 4096)
                if int(info.first_cbuf_con0) & 0x1000:
                    flagged += 1
        finally:
            call.close()
        return declared, flagged

    def test_reuse_flag_reaches_the_register_word(self):
        declared_off, flagged_off = self._reuse_words("interleaved")
        declared_on, flagged_on = self._reuse_words("interleaved_reuse")
        self.assertEqual((declared_off, flagged_off), (0, 0))
        self.assertGreater(declared_on, 0)
        self.assertEqual(declared_on, flagged_on)


if __name__ == "__main__":
    unittest.main()
