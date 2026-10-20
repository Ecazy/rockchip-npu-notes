import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rknpu_backend import CompileOptions, compile_rkir, default_profile, matmul_module
from rknpu_backend.serialize import canonical_json

ROOT = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def _env(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "python")
        lib = ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan.so"
        if lib.is_file():
            env["RKNPU_PLAN_LIB"] = str(lib)
        return env

    def _run(self, *args):
        return subprocess.run([sys.executable, "-m", "rknpu_backend.cli", *args],
                              cwd=str(ROOT), env=self._env(),
                              text=True, capture_output=True)

    def test_compile_validate_explain_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            module = matmul_module(m=13, k=64, n=33, activation="silu")
            rkir = tmp / "input.rkir.json"
            rkir.write_text(canonical_json(module.to_dict()))
            plan = tmp / "plan.json"
            proc = self._run("compile-rkir", str(rkir), "--profile",
                             "rk3588_vendor_co_v1", "--numeric-mode",
                             "rk_f16_tiled", "--allow-approx-lut",
                             "--out", str(plan))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(plan.is_file())
            proc = self._run("validate", str(plan))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            proc = self._run("explain", str(plan), "--format", "json")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads(proc.stdout)
            self.assertEqual(report["counts"]["tasks"], 3)
            self.assertEqual(report["kind"], "rkplan")

    def test_bad_numeric_mode_is_compile_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rkir = tmp / "input.rkir.json"
            rkir.write_text(canonical_json(
                matmul_module(m=2, k=3, n=4).to_dict()))
            proc = self._run("compile-rkir", str(rkir), "--numeric-mode",
                             "strict", "--out", str(tmp / "plan.json"))
            self.assertEqual(proc.returncode, 2)
            self.assertIn("RK_E_NUMERIC_CONTRACT", proc.stderr)


if __name__ == "__main__":
    unittest.main()
