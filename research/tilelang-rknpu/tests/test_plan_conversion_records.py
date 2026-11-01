# SPDX-License-Identifier: GPL-3.0-or-later
"""Plan-to-plan conversions must carry the records, not drop them.

A repeat group is N copies of one computation.  Each copy keeps the numeric
requirement it was compiled with, gets its own logical op index and its own task
id, and says which plan and which parent op it came from.  The checks here are
about *relations*, so every tamper case rewrites ``plan_hash`` first: a rejection
can only come from the structural cross-checks.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from rknpu_backend import (CompileOptions, Module, Op, TensorType, Value,
                           compile_rkir, default_profile, explain_plan,
                           load_plan, matmul_module, save_plan)
from rknpu_backend.errors import BackendError
from rknpu_backend.passes.repeat_expand import (expand_static_repeat,
                                                remap_group_inputs)
from rknpu_backend.serialize import sha256_json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PROFILE = default_profile()
OPTIONS = CompileOptions(numeric_mode="rk_f16_tiled")
MOCK_LIB = os.environ.get("RKNPU_PLAN_MOCK_LIB") or str(
    ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan_mock.so")
OLD_PARENT = ROOT / "models" / "compiled" / "tiny_vit_b8_qk.rkplan.json"


def recorded_parent(m=16, k=16, n=16, name="qk"):
    """A 16x16x16 matmul whose weight is an input, which is the admitted shape."""
    contract = matmul_module(m=m, k=k, n=n, b_layout="kn").ops[0].attrs[
        "numeric_contract"]
    module = Module(
        name=name,
        values=(Value(0, "Q", TensorType("float16", (m, k)), None, "input"),
                Value(1, "K", TensorType("float16", (k, n)), None, "input"),
                Value(2, "S", TensorType("float16", (m, n)), 0, "output")),
        ops=(Op(0, "matmul", (0, 1), (2,),
                {"b_layout": "kn", "numeric_contract": contract}),),
        inputs=(0, 1), outputs=(2,), source_hash=name)
    return compile_rkir(module, profile=PROFILE, options=OPTIONS).plan


def rehash(payload):
    body = {key: value for key, value in payload.items() if key != "plan_hash"}
    payload = json.loads(json.dumps(body))
    payload["plan_hash"] = sha256_json(payload)
    return payload


class ConversionRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent = recorded_parent()

    def _expand(self, plan=None, nrepeat=3, **kwargs):
        return expand_static_repeat(plan or self.parent, nrepeat, **kwargs)

    def test_one_repeat_returns_the_plan_and_keeps_its_records(self):
        group = expand_static_repeat(self.parent, 1)
        self.assertIs(group, self.parent)
        self.assertEqual(len(group.logical_ops), 1)
        self.assertEqual(len(group.numeric_requirements), 1)

    def test_each_copy_gets_its_own_op_and_requirement(self):
        for shared in (False, True):
            with self.subTest(shared_bo=shared):
                group = self._expand(nrepeat=3, shared_bo=shared)
                self.assertEqual([entry.index for entry in group.logical_ops],
                                 [0, 1, 2])
                self.assertEqual([requirement.op_index
                                  for requirement in group.numeric_requirements],
                                 [0, 1, 2])
                for i, entry in enumerate(group.logical_ops):
                    self.assertEqual(entry.opcode, "matmul")
                    self.assertEqual(entry.inputs, (f"Q_{i}", f"K_{i}"))
                    self.assertEqual(entry.outputs, (f"S_{i}",))
                    self.assertEqual(len(entry.tasks), 1)
                    self.assertFalse(entry.auxiliary)
                    self.assertEqual(entry.derived_from,
                                     {"plan_hash": self.parent.plan_hash,
                                      "op_index": 0})
                    requirement = group.numeric_requirements[i]
                    self.assertEqual(requirement.contract,
                                     self.parent.numeric_requirements[0].contract)
                    self.assertEqual(requirement.tasks, entry.tasks)
                    self.assertEqual(requirement.inputs, entry.inputs)
                    self.assertEqual(requirement.outputs, entry.outputs)
                    self.assertTrue(requirement.execution["attributed"])
                    self.assertEqual(requirement.execution["templates"],
                                     ["co.mm.f16.v1"])

    def test_copies_point_at_their_own_renumbered_task(self):
        group = self._expand(nrepeat=4, shared_bo=True)
        first_task = 2 * 4
        tasks = [action for action in group.actions
                 if action.kind == "npu_task"]
        self.assertEqual([entry.tasks for entry in group.logical_ops],
                         [(first_task + i,) for i in range(4)])
        for i, entry in enumerate(group.logical_ops):
            chosen = group.actions[entry.tasks[0]]
            self.assertEqual(chosen.template_id, "co.mm.f16.v1")
            self.assertEqual(chosen.id, first_task + i)
            self.assertEqual(len(tasks), 4)
        # The parent task id must not survive into the converted plan.
        self.assertNotIn(self.parent.logical_ops[0].tasks[0],
                         [entry.tasks[0] for entry in group.logical_ops])

    def test_saved_plan_reloaded_converts_the_same_way(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "parent.rkplan.json"
            save_plan(self.parent, str(path))
            loaded = load_plan(str(path))
        self.assertEqual(loaded.plan_hash, self.parent.plan_hash)
        here = self._expand(nrepeat=2)
        there = expand_static_repeat(loaded, 2)
        self.assertEqual(here.logical_ops, there.logical_ops)
        self.assertEqual(here.numeric_requirements,
                         there.numeric_requirements)
        self.assertEqual(here.plan_hash, there.plan_hash)

    def test_records_survive_a_second_conversion_and_a_save_load(self):
        group = self._expand(nrepeat=2, shared_bo=True)
        mapped = remap_group_inputs(group, 2, [
            {"shape": (16, 192), "row_stride": 192, "col_base": 0,
             "col_stride": 16, "col_modulus": 4, "transpose": False},
            {"shape": (16, 192), "row_stride": 192, "col_base": 64,
             "col_stride": 16, "col_modulus": 4, "transpose": True},
        ])
        self.assertEqual(mapped.logical_ops, group.logical_ops)
        self.assertEqual(mapped.numeric_requirements,
                         group.numeric_requirements)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mapped.rkplan.json"
            save_plan(mapped, str(path))
            loaded = load_plan(str(path))
            self.assertEqual(loaded.logical_ops, mapped.logical_ops)
            self.assertEqual(loaded.numeric_requirements,
                             mapped.numeric_requirements)
            payload = json.loads(path.read_text())
        self.assertEqual(payload["logical_ops"][0]["derived_from"],
                         {"plan_hash": self.parent.plan_hash, "op_index": 0})

    def test_remap_makes_each_copy_read_a_different_window(self):
        group = self._expand(nrepeat=4, shared_bo=True)
        mapped = remap_group_inputs(group, 4, [
            {"shape": (16, 192), "row_stride": 192, "col_base": 0,
             "col_stride": 16, "col_modulus": 4, "transpose": False},
            {"shape": (16, 192), "row_stride": 192, "col_base": 64,
             "col_stride": 16, "col_modulus": 4, "transpose": False},
        ])
        ops = explain_plan(mapped)["logical"]["ops"]
        windows = []
        for record in ops:
            reads = {(read["binding"], read["col_offset"], read["row_stride"])
                     for read in record["reads"]}
            self.assertEqual(len(reads), 2)
            windows.append(sorted(offset for _, offset, _ in reads))
            # The description still names the bindings the uploads pack.
            self.assertEqual({read["binding"] for read in record["reads"]},
                             set(record["inputs"]))
        self.assertEqual(windows, [[0, 64], [16, 80], [32, 96], [48, 112]])

    def test_instances_may_share_one_source_region(self):
        group = self._expand(nrepeat=3, shared_bo=True)
        mapped = remap_group_inputs(group, 3, [
            {"shape": (16, 16), "row_stride": 16, "col_base": 0,
             "col_stride": 0, "col_modulus": 1, "transpose": False},
            {"shape": (16, 192), "row_stride": 192, "col_base": 128,
             "col_stride": 16, "col_modulus": 4, "transpose": True},
        ])
        ops = explain_plan(mapped)["logical"]["ops"]
        for record in ops:
            first = [read for read in record["reads"]
                     if read["binding"].startswith("Q_")][0]
            self.assertEqual((first["col_offset"], first["row_stride"]),
                             (0, 16))
        self.assertEqual(len(mapped.numeric_requirements), 3)

    def test_old_plan_without_records_converts_and_stays_unrecorded(self):
        parent = load_plan(str(OLD_PARENT))
        self.assertEqual(parent.logical_ops, ())
        group = expand_static_repeat(parent, 4, shared_bo=True)
        self.assertEqual(group.logical_ops, ())
        self.assertEqual(group.numeric_requirements, ())
        self.assertEqual(len(group.submits[0].task_ids), 4)
        report = explain_plan(group)["logical"]
        self.assertFalse(report["stored"])
        self.assertEqual(report["source"], "not stored")
        # No contract is invented for the copies.
        self.assertEqual(explain_plan(group)["numeric"]["source"], "not stored")

    def test_requirements_without_a_table_are_refused_at_the_door(self):
        # The two fields can only be kept or dropped together; a plan that
        # keeps requirements without the table is not even constructible.
        with self.assertRaises(BackendError) as caught:
            replace(self.parent, logical_ops=())
        self.assertIn("need the logical op table", str(caught.exception))

    def test_a_parent_op_naming_a_non_task_cannot_even_be_built(self):
        # The conversion's own "op does not name this task" branch is
        # defensive: a plan whose op names a non-task is rejected before any
        # conversion can see it, which is asserted here.
        download = next(action.id for action in self.parent.actions
                        if action.kind == "download")
        with self.assertRaises(BackendError) as caught:
            replace(self.parent,
                    logical_ops=(replace(self.parent.logical_ops[0],
                                         tasks=(download,)),))
        self.assertIn("non-NPU action", str(caught.exception))

    def test_conversion_needs_a_hashed_parent_when_records_exist(self):
        unhashed = replace(self.parent, plan_hash="")
        with self.assertRaises(BackendError) as caught:
            expand_static_repeat(unhashed, 2)
        self.assertIn("hashed", str(caught.exception))


class ConvertedPlanTamperTests(unittest.TestCase):
    """Every case rewrites plan_hash, so only the relations can reject it."""

    @classmethod
    def setUpClass(cls):
        cls.parent = recorded_parent()
        cls.group = expand_static_repeat(cls.parent, 3, shared_bo=True)

    def _payload(self):
        return json.loads(json.dumps(self.group.to_dict()))

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
        loaded = self._load(rehash(self._payload()))
        self.assertEqual(loaded.plan_hash, self.group.plan_hash)
        self.assertEqual(loaded.logical_ops, self.group.logical_ops)

    def test_dropping_one_copys_requirement_is_rejected(self):
        payload = self._payload()
        payload["numeric_requirements"] = payload["numeric_requirements"][:2]
        self._reject(rehash(payload), "no numeric requirement")

    def test_pointing_a_requirement_at_the_wrong_copy_is_rejected(self):
        payload = self._payload()
        payload["numeric_requirements"][1]["op_index"] = 2
        self._reject(rehash(payload), "disagrees")

    def test_moving_a_task_to_another_copy_is_rejected(self):
        payload = self._payload()
        payload["logical_ops"][0]["tasks"] = payload["logical_ops"][1]["tasks"]
        self._reject(rehash(payload), "not attributed")

    def test_keeping_the_parent_input_names_is_rejected(self):
        payload = self._payload()
        for key in ("logical_ops", "numeric_requirements"):
            payload[key][0]["inputs"] = ["Q", "K"]
        self._reject(rehash(payload), "does not define")

    def test_reusing_a_parent_action_id_is_rejected(self):
        payload = self._payload()
        parent_task = self.parent.logical_ops[0].tasks[0]
        payload["logical_ops"][0]["tasks"] = [parent_task]
        payload["numeric_requirements"][0]["tasks"] = [parent_task]
        # The parent's id exists in the converted plan, but as an upload: a
        # stale id is refused because it is not a device task.
        self._reject(rehash(payload), "non-NPU action")

    def test_losing_the_provenance_is_still_self_consistent(self):
        # derived_from is the only optional part: dropping it keeps every other
        # relation intact, so it loads.  The copies then no longer say who they
        # came from, which is why the field is written by the conversion.
        payload = self._payload()
        for entry in payload["logical_ops"]:
            entry.pop("derived_from", None)
        loaded = self._load(rehash(payload))
        self.assertIsNone(loaded.logical_ops[0].derived_from)

    def test_a_consistent_alternative_conversion_is_accepted(self):
        payload = self._payload()
        for key in ("logical_ops", "numeric_requirements"):
            payload[key][0]["inputs"] = ["Q_alt", "K_alt"]
        payload["bindings"][0]["name"] = "Q_alt"
        payload["bindings"][1]["name"] = "K_alt"
        loaded = self._load(rehash(payload))
        self.assertEqual(loaded.logical_ops[0].inputs, ("Q_alt", "K_alt"))
        self.assertEqual(loaded.actions, self.group.actions)

    def test_derived_from_must_be_a_real_pointer(self):
        payload = self._payload()
        payload["logical_ops"][0]["derived_from"]["plan_hash"] = "nope"
        self._reject(rehash(payload), "derived_from.plan_hash")
        payload = self._payload()
        payload["logical_ops"][1]["derived_from"]["op_index"] = -1
        self._reject(rehash(payload), "op_index")


@unittest.skipUnless(Path(MOCK_LIB).is_file(), "mock library not built")
class ConvertedPlanExecutionTests(unittest.TestCase):
    """Every copy must still run exactly like the single-instance plan."""

    @classmethod
    def setUpClass(cls):
        from rknpu_backend.runtime.ffi import LibRKPlan
        cls.lib = LibRKPlan(MOCK_LIB)
        cls.parent = recorded_parent()

    def _run(self, plan, inputs, nrepeat):
        from rknpu_backend.models import PlanCall
        outputs = [[np.zeros((16, 16), dtype=np.float16)] for _ in range(nrepeat)]
        call = PlanCall(self.lib, plan, {}, fd=0)
        try:
            report, _ = call.run_into(inputs, [out for group in outputs
                                               for out in group])
        finally:
            call.close()
        return report, outputs

    def test_each_copy_matches_its_own_separate_run(self):
        rng = np.random.default_rng(77)
        repeats = 4
        per_repeat = [[np.ascontiguousarray(
            rng.standard_normal((16, 16)).astype(np.float16)) for _ in range(2)]
            for _ in range(repeats)]
        separate = []
        for i in range(repeats):
            report, outputs = self._run(self.parent, per_repeat[i], 1)
            self.assertEqual(int(report.task_count), 1)
            separate.append(outputs[0][0])
        for shared in (False, True):
            with self.subTest(shared_bo=shared):
                group = expand_static_repeat(self.parent, repeats,
                                             shared_bo=shared)
                flat = [array for repeat in per_repeat for array in repeat]
                report, outputs = self._run(group, flat, repeats)
                self.assertEqual(int(report.task_count), repeats)
                self.assertEqual(int(report.submit_count), 1)
                for i in range(repeats):
                    self.assertTrue(
                        np.array_equal(separate[i], outputs[i][0]),
                        f"copy {i} differs from its own separate run")

    def test_mapped_group_still_runs_and_keeps_its_records(self):
        import numpy as np
        repeats = 3
        group = expand_static_repeat(self.parent, repeats, shared_bo=True)
        weight_transposed = next(
            action.attrs["transpose_source"] for action in group.actions
            if action.kind == "upload" and "upload_w" in action.origin)
        mapped = remap_group_inputs(group, repeats, [
            {"shape": (16, 48), "row_stride": 48, "col_base": 0,
             "col_stride": 16, "col_modulus": 3, "transpose": False},
            {"shape": (16, 48), "row_stride": 48, "col_base": 0,
             "col_stride": 16, "col_modulus": 3,
             "transpose": weight_transposed},
        ])
        rng = np.random.default_rng(11)
        q = np.ascontiguousarray(
            rng.standard_normal((16, 48)).astype(np.float16))
        k = np.ascontiguousarray(
            rng.standard_normal((16, 48)).astype(np.float16))
        flat = [q, k] * repeats
        report, outputs = self._run(mapped, flat, repeats)
        self.assertEqual(int(report.task_count), repeats)
        expected = []
        for i in range(repeats):
            _, single = self._run(
                self.parent,
                [np.ascontiguousarray(q[:, 16 * i:16 * (i + 1)]),
                 np.ascontiguousarray(k[:, 16 * i:16 * (i + 1)])], 1)
            expected.append(single[0][0])
        for i in range(repeats):
            self.assertTrue(np.array_equal(expected[i], outputs[i][0]),
                            f"mapped copy {i} reads the wrong window")
        self.assertEqual(len(mapped.numeric_requirements), repeats)


class ConvertedPlanVerifierTests(unittest.TestCase):
    def test_real_c_verifier_accepts_a_converted_plan(self):
        real = ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan.so"
        if not Path(real).is_file():
            self.skipTest(f"host plan library not built: {real}")
        from rknpu_backend.runtime.ffi import LibRKPlan
        lib = LibRKPlan(str(real))
        parent = recorded_parent()
        for shared in (False, True):
            group = expand_static_repeat(parent, 3, shared_bo=shared)
            cplan = lib.create_plan(group)
            cplan.destroy()
        mapped = remap_group_inputs(expand_static_repeat(parent, 3,
                                                        shared_bo=True), 3, [
            {"shape": (16, 192), "row_stride": 192, "col_base": 0,
             "col_stride": 16, "col_modulus": 4, "transpose": False},
            {"shape": (16, 192), "row_stride": 192, "col_base": 64,
             "col_stride": 16, "col_modulus": 4, "transpose": True},
        ])
        cplan = lib.create_plan(mapped)
        cplan.destroy()


if __name__ == "__main__":
    unittest.main()
