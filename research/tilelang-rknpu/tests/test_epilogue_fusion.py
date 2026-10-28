"""Generic Linear -> bias -> ReLU Submit-envelope fusion on MLP."""
from __future__ import annotations

import gzip
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

from rknpu_backend import CompileOptions, compile_tilelang, default_profile
from rknpu_backend.hybrid import load_hybrid_program
from rknpu_backend.models import MLPRunner, load_weight_npz
from rknpu_backend.runtime.ffi import LibRKPlan
from rknpu_backend.serialize import save_plan
from models.tilelang_mlp import make_mlp_prim

MOCK_LIB = os.environ.get("RKNPU_PLAN_MOCK_LIB") or str(
    ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan_mock.so")


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        f.read(4)
        dims = struct.unpack(">III", f.read(12))
        return np.frombuffer(f.read(), np.uint8).reshape(dims)


class EpilogueFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not Path(MOCK_LIB).is_file():
            raise unittest.SkipTest(f"mock library not built: {MOCK_LIB}")
        cls.lib = LibRKPlan(MOCK_LIB)
        cls.weights = load_weight_npz(ROOT / "models" / "mnist_mlp.npz")
        cls.images = _read_idx(ROOT / ".data" / "mnist" /
                               "t10k-images-idx3-ubyte.gz")

    def _compile(self, fuse_epilogue: bool, fuse_add_relu: bool = False):
        return compile_tilelang(
            make_mlp_prim(1), profile=default_profile(),
            options=CompileOptions(numeric_mode="rk_f16_tiled",
                                   allow_approx_lut=False,
                                   fuse_epilogue=fuse_epilogue,
                                   fuse_add_relu=fuse_add_relu)).plan

    def _run(self, plan, tmp: Path) -> np.ndarray:
        tmp.mkdir(parents=True, exist_ok=True)
        save_plan(plan, str(tmp / "mnist_mlp_b1.rkplan.json"))
        hybrid = json.loads((ROOT / "models" / "compiled" /
                             "mnist_mlp_b1.rkhybrid.json").read_text())
        hybrid["plans"]["mlp"]["path"] = "mnist_mlp_b1.rkplan.json"
        hybrid["plans"]["mlp"]["plan_hash"] = plan.plan_hash
        hybrid["profile_hash"] = plan.profile_hash
        node = next(n for n in hybrid["nodes"] if n.get("plan") == "mlp")
        names = {spec.name for spec in plan.constants}
        node["constants"] = {k: v for k, v in
                             node["constants"].items() if k in names}
        path = tmp / "mnist_mlp_b1.rkhybrid.json"
        path.write_text(json.dumps(hybrid))
        program = load_hybrid_program(path, plans_dir=tmp)
        runner = MLPRunner(self.lib, program, self.weights, 1, fd=0,
                           weights_path=ROOT / "models" / "mnist_mlp.npz")
        try:
            return runner.run(self.images[:1])
        finally:
            runner.close()

    def test_mlp_submit_grouping_fusion_is_structural_and_bitwise(self):
        control = self._compile(False, False)
        fused = self._compile(True, False)
        self.assertEqual(len(control.submits), 8)
        self.assertEqual(len(fused.submits), 3)
        self.assertEqual(control.accounting.task_count,
                         fused.accounting.task_count)
        self.assertEqual(control.accounting.transfers,
                         fused.accounting.transfers)
        self.assertEqual([s.recipe_id for s in fused.submits],
                         ["linear_epilogue_v1"] * 3)
        self.assertEqual(
            [control.actions[t].template_id
             for s in control.submits for t in s.task_ids],
            [fused.actions[t].template_id
             for s in fused.submits for t in s.task_ids])
        # The fused fallback command workspace grows with max tasks; record the
        # exchange rather than pretending resources are unchanged.
        self.assertEqual(fused.accounting.command_workspace_bytes,
                         control.accounting.command_workspace_bytes + 32768)
        self.assertEqual(fused.accounting.device_request_bytes,
                         control.accounting.device_request_bytes + 32768)
        self.assertEqual(fused.accounting.host_bytes,
                         control.accounting.host_bytes + 33248)

        with tempfile.TemporaryDirectory() as tmp:
            c_out = self._run(control, Path(tmp) / "control")
            f_out = self._run(fused, Path(tmp) / "fused")
        self.assertEqual(c_out.shape, f_out.shape)
        np.testing.assert_array_equal(c_out, f_out)

    def test_mlp_add_relu_primitive_is_structural_and_bitwise(self):
        control = self._compile(True, False)
        fused = self._compile(True, True)
        self.assertEqual(len(control.submits), 3)
        self.assertEqual(len(fused.submits), 3)
        self.assertEqual(control.accounting.task_count, 11)
        self.assertEqual(fused.accounting.task_count, 9)
        self.assertEqual(control.accounting.data_bo_count, 19)
        self.assertEqual(fused.accounting.data_bo_count, 15)
        self.assertEqual(control.accounting.transfers, 16)
        self.assertEqual(fused.accounting.transfers, 14)
        self.assertEqual(
            [control.actions[t].template_id
             for s in control.submits for t in s.task_ids].count(
                 "co.ew_add_relu.f16.v1"), 0)
        self.assertEqual(
            [fused.actions[t].template_id
             for s in fused.submits for t in s.task_ids].count(
                 "co.ew_add_relu.f16.v1"), 2)
        with tempfile.TemporaryDirectory() as tmp:
            c_out = self._run(control, Path(tmp) / "control")
            f_out = self._run(fused, Path(tmp) / "fused")
        np.testing.assert_array_equal(c_out, f_out)
