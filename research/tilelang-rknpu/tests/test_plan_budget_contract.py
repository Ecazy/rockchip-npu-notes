"""Plan budget persistence and descriptor separation."""
from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rknpu_backend import CompileOptions, compile_tilelang, default_profile, load_plan
from rknpu_backend.codegen.descriptor import build_plan_descriptor
from rknpu_backend.hybrid import load_hybrid_program
from rknpu_backend.plan import Plan, PlanBudget, with_plan_hash
from rknpu_backend.serialize import save_plan
from models.tilelang_mlp import make_mlp_prim


class PlanBudgetContractTests(unittest.TestCase):
    def test_budgeted_compile_roundtrip_hash_and_descriptor(self):
        device_budget = 8 * 1024 * 1024
        host_budget = 16 * 1024 * 1024
        options = CompileOptions(
            numeric_mode="rk_f16_tiled", allow_approx_lut=False,
            device_budget_bytes=device_budget,
            host_budget_bytes=host_budget)
        plan = compile_tilelang(
            make_mlp_prim(1), profile=default_profile(), options=options).plan

        self.assertIsNotNone(plan.budget)
        self.assertEqual(plan.budget.device_budget_bytes, device_budget)
        self.assertEqual(plan.budget.host_budget_bytes, host_budget)
        self.assertEqual(plan.to_dict()["budget_policy"],
                         {"schema_version": 1,
                          "device_budget_bytes": device_budget,
                          "host_budget_bytes": host_budget})

        roundtrip = Plan.from_dict(plan.to_dict())
        self.assertEqual(roundtrip.budget, plan.budget)
        self.assertEqual(roundtrip.plan_hash, plan.plan_hash)

        changed = with_plan_hash(dataclasses.replace(
            plan, plan_hash="", budget=PlanBudget(
                device_budget_bytes=device_budget + 4096,
                host_budget_bytes=host_budget)))
        self.assertNotEqual(changed.plan_hash, plan.plan_hash)

        desc, _holder = build_plan_descriptor(plan)
        self.assertEqual(desc.device_budget_bytes, device_budget)
        self.assertEqual(desc.host_budget_bytes, host_budget)
        self.assertEqual(desc.claimed_device_bytes,
                         plan.accounting.device_request_bytes)
        self.assertEqual(desc.claimed_host_bytes, plan.accounting.host_bytes)

    def test_old_plan_without_budget_keeps_strict_descriptor_limits(self):
        plan = load_plan(ROOT / "models" / "compiled" /
                         "mnist_mlp_b32.rkplan.json")
        self.assertIsNone(plan.budget)
        self.assertNotIn("budget_policy", plan.to_dict())
        desc, _holder = build_plan_descriptor(plan)
        self.assertEqual(desc.device_budget_bytes,
                         plan.accounting.device_request_bytes)
        self.assertEqual(desc.host_budget_bytes, plan.accounting.host_bytes)
        self.assertEqual(desc.claimed_device_bytes,
                         plan.accounting.device_request_bytes)
        self.assertEqual(desc.claimed_host_bytes, plan.accounting.host_bytes)

    def test_budgeted_plan_loads_through_hybrid_manifest(self):
        options = CompileOptions(
            numeric_mode="rk_f16_tiled", allow_approx_lut=False,
            device_budget_bytes=8 * 1024 * 1024,
            host_budget_bytes=16 * 1024 * 1024)
        plan = compile_tilelang(
            make_mlp_prim(1), profile=default_profile(), options=options).plan
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            save_plan(plan, str(tmpdir / "mnist_mlp_b1.rkplan.json"))
            manifest = json.loads((ROOT / "models" / "compiled" /
                                   "mnist_mlp_b1.rkhybrid.json").read_text())
            manifest["plans"]["mlp"]["path"] = "mnist_mlp_b1.rkplan.json"
            manifest["plans"]["mlp"]["plan_hash"] = plan.plan_hash
            hybrid_path = tmpdir / "mnist_mlp_b1.rkhybrid.json"
            hybrid_path.write_text(json.dumps(manifest))
            program = load_hybrid_program(hybrid_path, plans_dir=tmpdir)
            loaded = program.plans["mlp"].plan
            self.assertEqual(loaded.budget, plan.budget)

    def test_malformed_or_unknown_budget_policy_is_rejected(self):
        for bad in (
            {"schema_version": 0, "device_budget_bytes": 1,
             "host_budget_bytes": 1},
            {"schema_version": 1, "device_budget_bytes": 0,
             "host_budget_bytes": 1},
            {"schema_version": 1, "device_budget_bytes": -1,
             "host_budget_bytes": 1},
            {"schema_version": 1, "device_budget_bytes": 1,
             "host_budget_bytes": 1, "extra": 1},
        ):
            with self.assertRaises(Exception):
                PlanBudget.from_dict(bad)
