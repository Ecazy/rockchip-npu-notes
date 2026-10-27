"""The artifact-smoke script must fail when *any* step fails.

The reviewed defect was in the exit aggregation: the manifest step's status was
recorded but not part of the final decision, and the failure count was recomputed by
re-reading the very artifact that step writes -- so a manifest failure could still
exit 0.

This test extracts the aggregation tail of ``scripts/run-artifact-smoke.sh`` and runs
it against synthetic step tables, so the decision logic is pinned without any device
access.
"""
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run-artifact-smoke.sh"
TAIL_MARKER = "manifest_rc=$?\n"


def _tail() -> str:
    text = SCRIPT.read_text(encoding="utf-8")
    if TAIL_MARKER not in text:
        raise AssertionError("aggregation tail marker not found in the script")
    return text.split(TAIL_MARKER, 1)[1]


class SmokeScriptExitTests(unittest.TestCase):
    """Drive the real tail with a synthetic status file."""

    def run_tail(self, *, steps, manifest_rc=0, remote_rc=0, verify_rc=0,
                 status_written=True, status_dir_exists=True):
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp)
            status = (local if status_dir_exists else local / "missing") / ".status"
            if status_written and status_dir_exists:
                status.write_text("".join(f"{name}\t{code}\n"
                                          for name, code in steps))
            harness = local / "harness.sh"
            harness.write_text(
                "#!/usr/bin/env bash\n"
                "set -uo pipefail\n"
                f"LOCAL='{local}'\n"
                f"status_file='{status}'\n"
                f"manifest_rc={manifest_rc}\n"
                f"remote_rc={remote_rc}\n"
                f"verify_rc={verify_rc}\n"
                'record() { printf \'%s\\t%s\\n\' "$1" "$2" >>"$status_file"; }\n'
                "if (( manifest_rc != 0 )); then :; fi\n"
                + _tail())
            harness.chmod(0o755)
            result = subprocess.run(["bash", str(harness)], text=True,
                                    capture_output=True)
            return result

    def test_all_steps_ok_passes(self):
        result = self.run_tail(steps=[("build_campaign", 0), ("remote_run", 0),
                                      ("verify", 0)])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("artifact smoke: PASS", result.stderr)

    def test_manifest_failure_fails_even_when_everything_else_passed(self):
        """The reviewed case: manifest_rc was recorded but never consulted."""
        result = self.run_tail(steps=[("build_campaign", 0), ("remote_run", 0),
                                      ("verify", 0), ("manifest", 1)],
                               manifest_rc=1)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("manifest(1)", result.stderr)

    def test_manifest_failure_without_status_entry_still_fails(self):
        # The status file lost the manifest row: the explicit check must still fire.
        result = self.run_tail(steps=[("build_campaign", 0), ("verify", 0)],
                               manifest_rc=1)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("manifest(1)", result.stderr)

    def test_any_failed_step_fails(self):
        result = self.run_tail(steps=[("build_campaign", 0), ("remote_run", 2),
                                      ("verify", 0), ("manifest", 0)])
        self.assertEqual(result.returncode, 1)
        self.assertIn("remote_run(2)", result.stderr)

    def test_verify_and_remote_are_double_checked(self):
        result = self.run_tail(steps=[("manifest", 0)], verify_rc=1)
        self.assertEqual(result.returncode, 1)
        self.assertIn("verify(1)", result.stderr)
        result = self.run_tail(steps=[("manifest", 0)], remote_rc=3)
        self.assertEqual(result.returncode, 1)
        self.assertIn("remote_run(3)", result.stderr)

    def test_manifest_row_is_always_recorded(self):
        """The tail records its own step, so a bare table is not an empty one.

        This documents why the ``no-steps-recorded`` guard is a *fallback* for an
        unwritable status file rather than a check on the pre-existing rows.
        """
        result = self.run_tail(steps=[])
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unwritable_status_file_is_not_a_pass(self):
        """If nothing can be recorded, the run cannot be claimed as passing."""
        result = self.run_tail(steps=[], status_dir_exists=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no-steps-recorded", result.stderr)

    def test_status_file_is_removed_after_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp)
            status = local / ".status"
            status.write_text("verify\t0\n")
            harness = local / "harness.sh"
            harness.write_text(
                "#!/usr/bin/env bash\nset -uo pipefail\n"
                f"LOCAL='{local}'\nstatus_file='{status}'\n"
                "manifest_rc=0\nremote_rc=0\nverify_rc=0\n"
                'record() { printf \'%s\\t%s\\n\' "$1" "$2" >>"$status_file"; }\n'
                + _tail())
            harness.chmod(0o755)
            subprocess.run(["bash", str(harness)], capture_output=True, text=True)
            self.assertFalse(status.exists())


if __name__ == "__main__":
    unittest.main()
