"""Complete-model milestone tests: real TileLang source, learned weights, hybrid runner."""
from __future__ import annotations

import gzip
import importlib.util
import json
import os
import struct
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MOCK_LIB = os.environ.get("RKNPU_PLAN_MOCK_LIB") or str(
    ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan_mock.so")
PLANS = ROOT / "models" / "compiled"
WEIGHTS = ROOT / "models"


def read_idx(path: Path, rank: int) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        f.read(4)
        dims = struct.unpack(">" + "I" * rank, f.read(4 * rank))
        return np.frombuffer(f.read(), np.uint8).reshape(dims)


class CompleteModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not Path(MOCK_LIB).is_file():
            raise unittest.SkipTest(f"mock library not built: {MOCK_LIB}")
        from rknpu_backend import default_profile
        from rknpu_backend.runtime.ffi import LibRKPlan
        cls.lib = LibRKPlan(MOCK_LIB)
        cls.profile = default_profile()
        cls.x = read_idx(ROOT / ".data" / "mnist" /
                         "t10k-images-idx3-ubyte.gz", 3)
        cls.y = read_idx(ROOT / ".data" / "mnist" /
                         "t10k-labels-idx1-ubyte.gz", 1).astype(np.int64)
        cls.mlp_weights = dict(np.load(WEIGHTS / "mnist_mlp.npz",
                                       allow_pickle=False))
        cls.vit_weights = dict(np.load(WEIGHTS / "tiny_vit.npz",
                                       allow_pickle=False))

    def _plan(self, name):
        from rknpu_backend import load_plan
        return load_plan(str(PLANS / f"{name}.rkplan.json"))

    def test_weight_manifest_and_learned_model_baselines(self):
        manifest = json.loads((ROOT / "models" / "manifest.json").read_text())
        names = {m["name"]: m for m in manifest["models"]}
        self.assertGreaterEqual(names["mnist_mlp"]["test_accuracy"], 0.97)
        self.assertGreaterEqual(names["tiny_vit"]["test_accuracy"], 0.95)
        for name in ("mnist_mlp.npz", "tiny_vit.npz"):
            sha = names["mnist_mlp" if name.startswith("mnist") else "tiny_vit"]["weights_sha256"]
            import hashlib
            self.assertEqual(hashlib.sha256((WEIGHTS / name).read_bytes()).hexdigest(), sha)

    def test_mlp_plan_contains_exact_relu_and_device_residency(self):
        plan = self._plan("mnist_mlp_b32")
        templates = [a.template_id for a in plan.actions if a.kind == "npu_task"]
        self.assertIn("co.ew_max.f16.v1", templates)
        self.assertEqual(templates.count("co.ew_max.f16.v1"), 2)
        self.assertEqual(sum(a.kind == "download" for a in plan.actions), 1)
        self.assertEqual([b.shape for b in plan.bindings if b.role == "input"],
                         [(32, 1024)])

    def test_mlp_source_compiles_from_real_tilelang(self):
        from models.tilelang_mlp import make_mlp_prim
        from rknpu_backend import CompileOptions, compile_tilelang
        result = compile_tilelang(
            make_mlp_prim(1), profile=self.profile,
            options=CompileOptions(numeric_mode="rk_f16_tiled",
                                   allow_approx_lut=False))
        self.assertIn("co.ew_max.f16.v1",
                      [a.template_id for a in result.plan.actions
                       if a.kind == "npu_task"])
        self.assertEqual(len([a for a in result.plan.actions
                              if a.kind == "download"]), 1)

    def test_mlp_learned_weights_input_update_and_batches(self):
        from rknpu_backend.models import MLPRunner
        from rknpu_backend.model_reference import mlp_logits_declared
        for batch in (1, 8, 32):
            runner = MLPRunner(self.lib, self._plan(f"mnist_mlp_b{batch}"),
                               self.mlp_weights, batch, fd=0)
            try:
                for start in (0, batch):
                    raw = self.x[start:start + batch]
                    got = runner.run(raw)
                    ref = mlp_logits_declared(self.mlp_weights, raw)
                    np.testing.assert_allclose(got, ref, atol=0.08, rtol=0.02)
                    self.assertEqual(got.shape, (batch, 10))
            finally:
                runner.close()

    def test_model_failure_does_not_publish(self):
        import subprocess
        env = dict(os.environ)
        env["RKNPU_PLAN_MOCK_LIB"] = MOCK_LIB
        env["PYTHONPATH"] = (str(ROOT / "python") +
                             (os.pathsep + env["PYTHONPATH"]
                              if env.get("PYTHONPATH") else ""))
        proc = subprocess.run(
            [sys.executable, str(ROOT / "tests" / "fixtures" /
                                  "model_failure_worker.py")],
            cwd=str(ROOT), env=env, text=True, capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + "\n" + proc.stderr)
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertFalse(result["published"])
        self.assertNotEqual(result["run_code"], "NO_ERROR")
        self.assertNotEqual(result["close_code"], "NO_ERROR")
        self.assertEqual(result["execs_after"], result["execs_before"] + 1)

    def test_transformer_source_dynamic_b_and_hybrid_runner(self):
        from models.tilelang_matmul import make_matmul_prim
        from rknpu_backend import CompileOptions, compile_tilelang
        from rknpu_backend.models import TinyViTRunner
        from rknpu_backend.model_reference import tiny_vit_logits_declared
        dyn = compile_tilelang(
            make_matmul_prim(16, 16, 16), profile=self.profile,
            options=CompileOptions(numeric_mode="rk_f16_tiled",
                                   allow_approx_lut=False),
            b_is_input=True).plan
        self.assertEqual([b.role for b in dyn.bindings if b.name in {"A", "B"}],
                         ["input", "input"])
        names = ("patch", "qkv", "qk", "pv", "attn_out", "gate", "up",
                 "down", "classifier")
        plans = {n: self._plan(f"tiny_vit_b1_{n}") for n in names}
        runner = TinyViTRunner(self.lib, plans, self.vit_weights, 1, fd=0)
        try:
            raw = self.x[:2]
            got = np.concatenate([runner.run(raw[i:i + 1]) for i in range(2)])
            ref = tiny_vit_logits_declared(self.vit_weights, raw)
            np.testing.assert_allclose(got, ref, atol=0.08, rtol=0.02)
            self.assertEqual((got.argmax(1) == ref.argmax(1)).all(), True)
            # Every matrix plan is a real one-task RKPlan; Host work is explicit
            # in TinyViTRunner, never a co_execute/graph fallback.
            for plan in plans.values():
                self.assertTrue(all(a.template_id == "co.mm.f16.v1"
                                    for a in plan.actions
                                    if a.kind == "npu_task"))
        finally:
            runner.close()


if __name__ == "__main__":
    unittest.main()
