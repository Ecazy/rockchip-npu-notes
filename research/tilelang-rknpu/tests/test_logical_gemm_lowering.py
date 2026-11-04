# SPDX-License-Identifier: GPL-3.0-or-later
"""Full-logical GEMM target-TIR entry, dispatcher and backend boundaries."""
from __future__ import annotations

from unittest import mock
import unittest

import numpy as np
import tilelang
import tvm
from tvm import tirx

import models.tilelang_matmul as tmm
from rknpu_backend import CompileOptions, default_profile
from rknpu_backend.errors import BackendError, RK_E_SHAPE, RK_E_UNSUPPORTED
from rknpu_backend.frontend import logical_schedule as S
from rknpu_backend.frontend import tilelang_backend as B
from rknpu_backend.frontend import tilelang_lowering as TL
from rknpu_backend.frontend.rknpu_pass import _op_name

M, K, N = 1, 576, 960


def options(**extra):
    base = {"numeric_mode": "rk_f16_tiled", "allow_approx_lut": False}
    base.update(extra)
    return CompileOptions(**base)


def _duplicate_output_copy(prim_func):
    def change(node):
        if type(node).__name__ != "Evaluate" or \
                _op_name(node.value) != "tl.tileop.copy":
            return None
        destination = node.value.args[1]
        if str(destination.args[0].buffer.name) != "Y":
            return None
        return tirx.SeqStmt([node, node])
    return prim_func.with_body(tirx.stmt_functor.ir_transform(
        prim_func.body, None, change))


def _shift_output_copy(prim_func):
    def change(node):
        if type(node).__name__ != "Evaluate" or \
                _op_name(node.value) != "tl.tileop.copy":
            return None
        call = node.value
        args = list(call.args)
        destination = args[1]
        if str(destination.args[0].buffer.name) != "Y":
            return None
        region_args = list(destination.args)
        load = region_args[0]
        indices = list(load.indices)
        indices[0] = indices[0] + 1
        region_args[0] = tirx.BufferLoad(load.buffer, indices)
        args[1] = tirx.Call(destination.dtype, destination.op, region_args)
        return tirx.Evaluate(tirx.Call(call.dtype, call.op, args))
    return prim_func.with_body(tirx.stmt_functor.ir_transform(
        prim_func.body, None, change))


class ConversionTests(unittest.TestCase):
    def test_the_dispatcher_selects_the_task_chain(self):
        prim = tmm.make_matmul_prim(M, K, N)
        self.assertFalse(TL.selects_guarded_tail(prim))
        self.assertIs(TL.entry_for(prim), S.compile_logical_gemm_tasks)
        plan, stages, facts, tasks = TL.compile_tilelang_ir(
            prim, profile=default_profile(), options=options())
        self.assertEqual([stage.name for stage in stages],
                         ["schedule-source-ops", "lower-scheduled-ops", "instantiate", "plan"])
        self.assertEqual(len(tasks), 92)
        self.assertEqual(plan.accounting.data_bo_count, 17)

    def test_the_registered_backend_uses_the_same_conversion(self):
        prim = tmm.make_matmul_prim(M, K, N)
        B.clear_lowerings()
        try:
            if True:
                direct = S.compile_logical_gemm_tasks(
                    prim, profile=default_profile(), options=options(),
                    name="rknpu-pass")[0]
                with B.registered():
                    artifact = tilelang.engine.lower(
                        tvm.IRModule({"main": prim}), target=B.rknpu_target())
                    plan = B.plan_from_artifact(artifact)
                    record = B.last_lowering()
        finally:
            B.clear_lowerings()
        self.assertEqual(plan.to_dict(), direct.to_dict())
        self.assertIs(record.entry, S.compile_logical_gemm_tasks)
        self.assertEqual([name for name, _ in record.stages],
                         ["schedule-source-ops", "lower-scheduled-ops", "instantiate", "plan"])
        self.assertEqual(record.facts["logical"], [M, K, N])


class GuardedExecutionTests(unittest.TestCase):
    ENTRIES = (
        ("direct", lambda prim: S.compile_logical_gemm_tasks(
            prim, profile=default_profile(), options=options())),
        ("dispatcher", lambda prim: TL.compile_tilelang_ir(
            prim, profile=default_profile(), options=options())),
        ("backend", lambda prim: B.compile_to_plan(
            prim, profile=default_profile(), options=options())),
    )

    def _assert_guarded_rejected(self, prim):
        for name, compile_one in self.ENTRIES:
            with self.subTest(entry=name):
                with self.assertRaises(BackendError) as caught:
                    compile_one(prim)
                error = caught.exception
                self.assertEqual(error.code, RK_E_UNSUPPORTED)
                diagnostic = error.diagnostics[0]
                self.assertEqual(diagnostic.stage, "frontend")
                self.assertIn("conditional execution", diagnostic.message)

    def test_a_guarded_gemm_is_rejected(self):
        prim = tmm.make_matmul_prim(M, K, N)

        def change(node):
            if type(node).__name__ == "Evaluate" and \
                    _op_name(node.value) == "tl.tileop.gemm":
                return tirx.IfThenElse(tirx.IntImm("bool", 0), node, None)
            return None

        self._assert_guarded_rejected(
            prim.with_body(tirx.stmt_functor.ir_transform(
                prim.body, None, change)))

    def test_a_guarded_output_copy_is_rejected(self):
        prim = tmm.make_matmul_prim(M, K, N)

        def change(node):
            if type(node).__name__ == "Evaluate" and \
                    _op_name(node.value) == "tl.tileop.copy":
                if str(node.value.args[1].args[0].buffer.name) == "Y":
                    return tirx.IfThenElse(tirx.IntImm("bool", 0), node, None)
            return None

        self._assert_guarded_rejected(
            prim.with_body(tirx.stmt_functor.ir_transform(
                prim.body, None, change)))


class NegativeTests(unittest.TestCase):
    def test_an_extra_copy_is_not_swallowed(self):
        prim = _duplicate_output_copy(tmm.make_matmul_prim(M, K, N))
        with self.assertRaises(BackendError):
            S.compile_logical_gemm_tasks(
                prim, profile=default_profile(), options=options())

    def test_a_shifted_region_is_not_swallowed(self):
        prim = _shift_output_copy(tmm.make_matmul_prim(M, K, N))
        with self.assertRaises(BackendError):
            S.compile_logical_gemm_tasks(
                prim, profile=default_profile(), options=options())

    def test_the_guarded_tail_form_is_not_accepted_as_logical(self):
        prim = tmm.make_matmul_tiled_tail_prim(M, K, N, bm=1, bn=256)
        with self.assertRaises(BackendError):
            S.compile_logical_gemm_tasks(
                prim, profile=default_profile(), options=options())


if __name__ == "__main__":
    unittest.main()
