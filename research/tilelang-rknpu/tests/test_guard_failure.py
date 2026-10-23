"""Guard failure must poison the workspace and publish nothing.

The failed workspace cannot be destroyed by design; the actual scenario runs in a
subprocess so process-level recovery is tested without leaking mock BOs into the
pytest process.
"""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOCK_LIB = os.environ.get("RKNPU_PLAN_MOCK_LIB") or str(
    ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan_mock.so")


class GuardFailureTests(unittest.TestCase):
    def test_guard_corruption_refuses_publish_and_poisons(self):
        if not Path(MOCK_LIB).is_file():
            self.skipTest(f"mock library not built: {MOCK_LIB}")
        env = dict(os.environ)
        env["RKNPU_PLAN_MOCK_LIB"] = MOCK_LIB
        env["PYTHONPATH"] = (str(ROOT / "python") +
                             (os.pathsep + env["PYTHONPATH"]
                              if env.get("PYTHONPATH") else ""))
        proc = subprocess.run(
            [sys.executable,
             str(ROOT / "tests" / "fixtures" / "guard_failure_worker.py")],
            cwd=str(ROOT), env=env, text=True, capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + "\n" + proc.stderr)
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["guard_code"], -9)
        self.assertEqual(result["poison_code"], -11)
        self.assertTrue(result["output_unchanged"])

    def test_later_submit_after_download_still_blocks_publication(self):
        if not Path(MOCK_LIB).is_file():
            self.skipTest(f"mock library not built: {MOCK_LIB}")
        env = dict(os.environ)
        env["RKNPU_PLAN_MOCK_LIB"] = MOCK_LIB
        env["PYTHONPATH"] = (str(ROOT / "python") +
                             (os.pathsep + env["PYTHONPATH"]
                              if env.get("PYTHONPATH") else ""))
        proc = subprocess.run(
            [sys.executable, str(ROOT / "tests" / "fixtures" /
                                  "guard_after_download_worker.py")],
            cwd=str(ROOT), env=env, text=True, capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + "\n" + proc.stderr)
        self.assertIn("PASS post-download task guard corruption", proc.stdout)


if __name__ == "__main__":
    unittest.main()
