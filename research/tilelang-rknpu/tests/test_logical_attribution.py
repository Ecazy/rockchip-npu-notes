# SPDX-License-Identifier: GPL-3.0-or-later
"""The plan must restate its logical graph, and every task must name its op.

Two questions are answered here and nowhere else:

* which device tasks implement each logical op -- recorded when the lowering
  emits the task, never re-derived from an origin string; and
* does a numeric requirement still describe the graph it names -- checked
  against the plan own logical-op table, not against itself.

The tamper cases all rewrite plan_hash first, so a rejection can only come from
the structural cross-checks.  Nothing here claims the device meets a tolerance.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from rknpu_backend import (CompileOptions, Module, Op, TensorType, Value,
                           compile_rkir, default_profile, explain_plan,
                           load_plan, matmul_module, save_plan)
from rknpu_backend.errors import BackendError
from rknpu_backend.frontend.tir_utils import tilelang_available
from rknpu_backend.serialize import sha256_json

PROFILE = default_profile()
ROOT = Path(__file__).resolve().parents[1]


def options(**extra):
    settings = {"numeric_mode": "rk_f16_tiled", "allow_approx_lut": False}
    settings.update(extra)
    return CompileOptions(**settings)


def schedule(tile_m=128, tile_n=256):
    return {"kind": "mn_tiles_v1", "tile_m": tile_m, "tile_n": tile_n,
            "order": "m_major", "k_tile": 256,
            "accumulation": "ordered_ascending_f16_writeback"}


def with_schedule(module, tile_m=128, tile_n=256):
    attrs = dict(module.ops[0].attrs)
    attrs["source_schedule"] = schedule(tile_m, tile_n)
    attrs["source_k_rounding"] = "explicit_per_k_fp16"
    return replace(module, ops=(replace(module.ops[0], attrs=attrs),))


def two_stage_module():
    base_attrs = dict(matmul_module(m=256, k=256, n=256).ops[0].attrs)
    first = dict(base_attrs, source_schedule=schedule(),
                 source_k_rounding="explicit_per_k_fp16")
    second = dict(first, numeric_contract=dict(base_attrs["numeric_contract"],
                                               tolerance={"atol": 4e-3,
                                                          "rtol": 4e-3}))
    values = (Value(0, "X", TensorType("float16", (256, 256)), None, "input"),
              Value(1, "W1", TensorType("float16", (256, 256)), None, "constant"),
              Value(2, "Z", TensorType("float16", (256, 256)), 0, "temporary"),
              Value(3, "W2", TensorType("float16", (256, 256)), None, "constant"),
              Value(4, "Y", TensorType("float16", (256, 256)), 1, "output"))
    return Module(name="chain", values=values,
                  ops=(Op(0, "matmul", (0, 1), (2,), first),
                       Op(1, "matmul", (2, 3), (4,), second)),
                  inputs=(0,), outputs=(4,), source_hash="chain",
                  profile_hash=PROFILE.profile_hash)


def add_relu_module(m=13, k=64, n=33, contract=True):
    attrs = {"b_layout": "kn"}
    if contract:
        attrs["numeric_contract"] = matmul_module(
            m=m, k=k, n=n).ops[0].attrs["numeric_contract"]
    values = (Value(0, "A", TensorType("float16", (m, k)), None, "input"),
              Value(1, "B", TensorType("float16", (k, n)), None, "constant"),
              Value(2, "R", TensorType("float16", (m, n)), None, "input"),
              Value(3, "Z", TensorType("float16", (m, n)), 0, "temporary"),
              Value(4, "S", TensorType("float16", (m, n)), 1, "temporary"),
              Value(5, "Y", TensorType("float16", (m, n)), 2, "output"))
    return Module(name="add-relu", values=values,
                  ops=(Op(0, "matmul", (0, 1), (3,), attrs),
                       Op(1, "add", (3, 2), (4,), {}),
                       Op(2, "relu", (4,), (5,), {})),
                  inputs=(0, 2), outputs=(5,), source_hash="add-relu",
                  profile_hash=PROFILE.profile_hash)


def rehash(payload):
    """Rewrite plan_hash so only structural checks can reject the payload."""
    body = {key: value for key, value in payload.items() if key != "plan_hash"}
    payload = json.loads(json.dumps(body))
    payload["plan_hash"] = sha256_json(payload)
    return payload


class AttributionCoverageTests(unittest.TestCase):
    def _attributed(self, plan):
        claimed = {}
        for entry in plan.logical_ops:
            for task in entry.tasks:
                claimed.setdefault(task, []).append(entry.index)
        return claimed

    def _assert_closed(self, plan):
        tasks = {action.id for action in plan.actions
                 if action.kind == "npu_task"}
        claimed = self._attributed(plan)
        self.assertEqual(set(claimed), tasks,
                         "every device task must name its logical op")

    def test_single_matmul_op_is_attributed_to_both_templates(self):
        plan = compile_rkir(with_schedule(matmul_module(m=256, k=512, n=256)),
                            profile=PROFILE, options=options()).plan
        self.assertEqual([entry.index for entry in plan.logical_ops], [0])
        entry = plan.logical_ops[0]
        self.assertEqual(entry.opcode, "matmul")
        self.assertEqual(entry.inputs, ("A", "B"))
        self.assertEqual(entry.outputs, ("Y",))
        self.assertTrue(entry.tasks)
        self.assertIsNotNone(entry.contract)
        self.assertFalse(entry.auxiliary)
        self._assert_closed(plan)
        self.assertEqual(entry.tasks, plan.numeric_requirements[0].tasks)

    def test_two_stage_stages_own_their_own_k_chains(self):
        plan = compile_rkir(two_stage_module(), profile=PROFILE,
                            options=options()).plan
        self._assert_closed(plan)
        self.assertEqual([entry.index for entry in plan.logical_ops], [0, 1])
        first, second = plan.logical_ops
        self.assertEqual(first.inputs, ("X", "W1"))
        self.assertEqual(second.inputs, ("Z", "W2"))
        self.assertFalse(set(first.tasks) & set(second.tasks))
        self.assertEqual(plan.numeric_requirements[0].contract["tolerance"],
                         {"atol": 8e-3, "rtol": 8e-3})
        self.assertEqual(plan.numeric_requirements[1].contract["tolerance"],
                         {"atol": 4e-3, "rtol": 4e-3})

    def test_fused_add_relu_task_serves_both_logical_ops(self):
        plan = compile_rkir(add_relu_module(), profile=PROFILE,
                            options=options()).plan
        self._assert_closed(plan)
        by_opcode = {entry.opcode: entry for entry in plan.logical_ops}
        self.assertIn("co.ew_add_relu.f16.v1",
                      {action.template_id for action in plan.actions
                       if action.kind == "npu_task"})
        self.assertEqual(by_opcode["add"].tasks, by_opcode["relu"].tasks)
        self.assertTrue(by_opcode["add"].tasks)
        self.assertEqual([requirement.opcode
                          for requirement in plan.numeric_requirements],
                         ["matmul"])

    def test_contractless_elementwise_ops_are_still_recorded(self):
        # Only the matmul carries a contract here, so only it has a
        # requirement; the elementwise op must still name its tasks.
        plan = compile_rkir(matmul_module(m=13, k=64, n=33,
                                         activation="sigmoid"),
                            profile=PROFILE,
                            options=options(allow_approx_lut=True)).plan
        self.assertEqual([requirement.opcode
                          for requirement in plan.numeric_requirements],
                         ["matmul"])
        self.assertEqual([entry.opcode for entry in plan.logical_ops],
                         ["matmul", "sigmoid"])
        self.assertTrue(plan.logical_ops[1].tasks)
        self.assertIsNone(plan.logical_ops[1].contract)
        self._assert_closed(plan)

    def test_explain_reports_the_stored_graph(self):
        plan = compile_rkir(with_schedule(matmul_module(m=256, k=512, n=256)),
                            profile=PROFILE, options=options()).plan
        logical = explain_plan(plan)["logical"]
        self.assertTrue(logical["stored"])
        self.assertEqual(logical["source"], "plan")
        self.assertEqual(logical["ops"][0]["opcode"], "matmul")
        self.assertEqual(logical["without_tasks"], [])


@unittest.skipUnless(tilelang_available(), "TileLang absent")
class FusedPathAttributionTests(unittest.TestCase):
    def _roundtrip(self, plan):
        tmp = tempfile.mkdtemp()
        path = Path(tmp) / "plan.rkplan.json"
        save_plan(plan, str(path))
        return str(path)

    def test_swiglu_ffn_elementwise_tasks_are_attributed(self):
        from rknpu_backend import compile_tilelang
        from tests.fixtures import tilelang_ffn
        plan = compile_tilelang(tilelang_ffn.swiglu_ffn_f16(64, 256, 128),
                                profile=PROFILE,
                                options=options(allow_approx_lut=True)).plan
        by_opcode = {entry.opcode: entry for entry in plan.logical_ops}
        self.assertEqual(sorted(by_opcode), ["add", "matmul", "mul", "silu"])
        for opcode in ("matmul", "silu", "mul", "add"):
            self.assertTrue(by_opcode[opcode].tasks,
                            f"{opcode} must name the tasks that implement it")
        silu = by_opcode["silu"]
        templates = {action.template_id for action in plan.actions
                     if action.id in silu.tasks}
        self.assertIn("co.lut.sigmoid.f16.v1", templates)
        self.assertIn("co.ew_mul.f16.v1", templates)
        requirement = next(r for r in plan.numeric_requirements
                           if r.opcode == "silu")
        self.assertTrue(requirement.execution["approximated"])
        self.assertFalse(requirement.contract["approximate"])

    def test_attribution_survives_upload_hoisting_and_chain_fusion(self):
        from rknpu_backend import compile_tilelang
        from tests.fixtures import tilelang_ffn
        plan = compile_tilelang(
            tilelang_ffn.swiglu_ffn_f16(64, 256, 128), profile=PROFILE,
            options=options(allow_approx_lut=True, fuse_task_chain=True,
                            submit_mode="chain")).plan
        loaded = load_plan(self._roundtrip(plan))
        templates = {action.id: action.template_id
                     for action in loaded.actions if action.kind == "npu_task"}
        for entry in loaded.logical_ops:
            for task in entry.tasks:
                self.assertIn(task, templates)
        for requirement in loaded.numeric_requirements:
            self.assertEqual(
                sorted({templates[task] for task in requirement.tasks}),
                sorted(requirement.execution["templates"]))
            self.assertEqual(requirement.execution["attributed"],
                             bool(requirement.tasks))
        self.assertEqual(sum(len(s.task_ids) for s in loaded.submits),
                         len(templates))

    def test_scheduled_ffn_view_ops_are_auxiliary(self):
        # The eight-op packed FFN splits the gate/up halves with two view ops:
        # they are real logical ops with no task of their own.
        import importlib.util
        import sys
        fixtures = ROOT / "tests" / "fixtures"
        if str(fixtures) not in sys.path:
            sys.path.insert(0, str(fixtures))
        spec = importlib.util.spec_from_file_location(
            "tl_ffn_scheduled", str(fixtures / "tilelang_ffn_scheduled.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from rknpu_backend import compile_tilelang
        plan = compile_tilelang(module.make_ffn(), profile=PROFILE,
                                options=options(allow_approx_lut=True)).plan
        auxiliary = [entry for entry in plan.logical_ops if entry.auxiliary]
        self.assertGreaterEqual(len(auxiliary), 2)
        self.assertTrue(all(entry.opcode == "view" for entry in auxiliary))
        for entry in auxiliary:
            self.assertEqual(entry.tasks, ())
        for entry in plan.logical_ops:
            if not entry.auxiliary:
                self.assertTrue(entry.tasks, entry.opcode)


class CrossCheckTests(unittest.TestCase):
    """Tampering is re-hashed first, so only the cross-checks can reject it."""

    def _payload(self):
        plan = compile_rkir(two_stage_module(), profile=PROFILE,
                            options=options()).plan
        return plan, json.loads(json.dumps(plan.to_dict()))

    def _load(self, payload):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tampered.rkplan.json"
            path.write_text(json.dumps(payload))
            return load_plan(str(path))

    def _reject(self, payload, needle):
        with self.assertRaises(BackendError) as caught:
            self._load(payload)
        self.assertIn(needle, caught.exception.diagnostics[0].message)

    def test_hash_rewrite_alone_is_accepted(self):
        plan, payload = self._payload()
        loaded = self._load(rehash(payload))
        self.assertEqual(loaded.plan_hash, plan.plan_hash)

    def test_op_index_moved_to_another_real_op_is_rejected(self):
        _, payload = self._payload()
        payload["numeric_requirements"][0]["op_index"] = 1
        self._reject(rehash(payload), "disagrees")

    def test_opcode_change_is_rejected(self):
        _, payload = self._payload()
        payload["numeric_requirements"][1]["opcode"] = "silu"
        self._reject(rehash(payload), "disagrees")

    def test_inputs_change_is_rejected(self):
        _, payload = self._payload()
        payload["numeric_requirements"][0]["inputs"] = ["X", "W2"]
        self._reject(rehash(payload), "disagrees")

    def test_task_moved_between_ops_is_rejected(self):
        _, payload = self._payload()
        payload["numeric_requirements"][0]["tasks"] = \
            payload["numeric_requirements"][0]["tasks"][:1]
        self._reject(rehash(payload), "disagrees")

    def test_graph_entry_losing_its_task_is_rejected(self):
        _, payload = self._payload()
        payload["logical_ops"][0]["tasks"] = \
            list(payload["logical_ops"][0]["tasks"]) + [
                payload["logical_ops"][1]["tasks"][0]]
        self._reject(rehash(payload), "disagrees")

    def test_upload_cannot_implement_a_logical_op(self):
        plan, payload = self._payload()
        upload = next(action.id for action in plan.actions
                      if action.kind == "upload")
        payload["logical_ops"][0]["tasks"] = \
            list(payload["logical_ops"][0]["tasks"]) + [upload]
        self._reject(rehash(payload), "non-NPU action")

    def test_unattributed_device_task_is_rejected(self):
        _, payload = self._payload()
        payload["logical_ops"][1]["tasks"] = []
        payload["numeric_requirements"][1]["tasks"] = []
        payload["numeric_requirements"][1]["execution"]["attributed"] = False
        self._reject(rehash(payload), "not attributed")

    def test_requirement_without_the_graph_is_rejected(self):
        _, payload = self._payload()
        del payload["logical_ops"]
        self._reject(rehash(payload), "need the logical op table")

    def test_contract_change_is_rejected(self):
        _, payload = self._payload()
        payload["numeric_requirements"][0]["contract"]["tolerance"] = {
            "atol": 1.0, "rtol": 1.0}
        self._reject(rehash(payload), "disagrees")

    def test_consistent_rewrite_is_accepted(self):
        plan, payload = self._payload()
        # A different but still self-consistent request: rename the input
        # everywhere it is recorded -- the binding, both graph entries and the
        # requirement.  Execution is untouched.
        payload["bindings"][0]["name"] = "X_renamed"
        for key in ("logical_ops", "numeric_requirements"):
            payload[key][0]["inputs"] = ["X_renamed", "W1"]
        loaded = self._load(rehash(payload))
        self.assertEqual(loaded.logical_ops[0].inputs, ("X_renamed", "W1"))
        self.assertEqual(loaded.bindings[0].name, "X_renamed")
        self.assertEqual(loaded.actions, plan.actions)
        self.assertEqual(loaded.submits, plan.submits)

    def test_renaming_a_binding_without_the_description_is_rejected(self):
        _, payload = self._payload()
        payload["bindings"][0]["name"] = "X_renamed"
        self._reject(rehash(payload), "does not define")

    def test_old_plan_without_the_graph_still_loads_and_resaves(self):
        plan = load_plan(str(ROOT / "models" / "compiled" /
                             "tiny_vit_b1_attn_out.rkplan.json"))
        self.assertEqual(plan.logical_ops, ())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.rkplan.json"
            save_plan(plan, str(path))
            payload = json.loads(path.read_text())
        self.assertNotIn("logical_ops", payload)
        self.assertNotIn("numeric_requirements", payload)
        self.assertEqual(payload["plan_hash"], plan.plan_hash)


class RoundTripTests(unittest.TestCase):
    def test_saved_plan_round_trips_the_graph_and_the_hash(self):
        plan = compile_rkir(two_stage_module(), profile=PROFILE,
                            options=options()).plan
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.rkplan.json"
            save_plan(plan, str(path))
            loaded = load_plan(str(path))
        self.assertEqual(loaded.plan_hash, plan.plan_hash)
        self.assertEqual(loaded.logical_ops, plan.logical_ops)
        self.assertEqual(loaded.numeric_requirements, plan.numeric_requirements)


if __name__ == "__main__":
    unittest.main()
