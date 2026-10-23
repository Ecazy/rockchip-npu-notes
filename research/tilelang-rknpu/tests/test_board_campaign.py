"""Host preflight for the RK3588 board campaign generator and runner."""
import json
import os
import numpy as np
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rknpu_backend.frontend.tilelang_adapter import tilelang_available

ROOT = Path(__file__).resolve().parents[1]
MOCK_LIB = os.environ.get("RKNPU_PLAN_MOCK_LIB") or str(
    ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan_mock.so")


@unittest.skipUnless(tilelang_available(), "pinned TileLang optional dependency absent")
@unittest.skipUnless(Path(MOCK_LIB).is_file(), "mock library not built")
class BoardCampaignHostPreflightTests(unittest.TestCase):
    def test_generator_and_runner_close_all_cases_on_mock(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp) / "campaign"
            results = Path(tmp) / "results"
            env = dict(os.environ)
            env["PYTHONPATH"] = (str(ROOT / "python") +
                                 (os.pathsep + env["PYTHONPATH"]
                                  if env.get("PYTHONPATH") else ""))
            proc = subprocess.run(
                [sys.executable, str(ROOT / "tools" / "make_board_campaign.py"),
                 "--out-dir", str(campaign)],
                cwd=str(ROOT), env=env, text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            manifest = json.loads((campaign / "manifest.json").read_text())
            self.assertTrue(manifest["cases"])
            self.assertEqual(len({c["name"] for c in manifest["cases"]}),
                             len(manifest["cases"]))
            projection = next(c for c in manifest["cases"]
                              if c["name"] == "projection_chain")
            refs = np.load(campaign / projection["data"], allow_pickle=False)
            self.assertEqual(refs["declared.Y.0"].dtype, np.float16)
            self.assertEqual(refs["fp64_math.Y.0"].dtype, np.float64)
            self.assertFalse(np.array_equal(
                refs["fp64_math.Y.0"],
                refs["fp64_math.Y.0"].astype(np.float16).astype(np.float64)))

            env["RKNPU_PLAN_LIB"] = MOCK_LIB
            env["RKNPU_CORE_MASK"] = "1"
            proc = subprocess.run(
                [sys.executable, str(campaign / "tools" / "board_runner.py"),
                 "--campaign", str(campaign), "--out", str(results), "--fd", "0"],
                cwd=str(ROOT), env=env, text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0,
                             proc.stdout + "\n" + proc.stderr)
            reports = [json.loads(p.read_text())
                       for p in results.glob("*.json")
                       if p.name != "environment.json"]
            self.assertEqual(len(reports), len(manifest["cases"]))
            self.assertTrue(all(r["status"] == "PASS" for r in reports),
                            [r for r in reports if r["status"] != "PASS"])
            self.assertTrue(any(r.get("guard_code") == -4 for r in reports))
            self.assertTrue(any(r.get("layout_ids") for r in reports))

            env["RKNPU_CHIP"] = "rk3588"
            cli_out = Path(tmp) / "cli"
            proc = subprocess.run(
                [sys.executable, str(campaign / "tools" /
                                     "board_cli_check.py"),
                 "--campaign", str(campaign), "--out", str(cli_out),
                 "--fd", "0"],
                cwd=str(ROOT), env=env, text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0,
                             proc.stdout + "\n" + proc.stderr)
            cli_report = json.loads((cli_out / "cli-report.json").read_text())
            self.assertEqual(cli_report["status"], "PASS")
            self.assertEqual(cli_report["steps"]["bad_binding"]["rc"], 2)
            self.assertFalse(cli_report["steps"]["bad_binding"]["published"])
            self.assertTrue(all(item["before"] == item["after"]
                                for item in cli_report["fd"]))


if __name__ == "__main__":
    unittest.main()
