"""Targeted verifier regressions: pair bytes, dtype and source-bundle hash."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROFILE = "a" * 64
LIB_HASH = "b" * 64
NAMES = ("projection_chain", "projection_separate",
         "ffn_chain", "ffn_separate",
         "ragged_chain", "ragged_separate")


def _write_source_bundle(run_root: Path) -> None:
    payload = run_root / "source.txt"
    payload.write_text("source bundle payload\n")
    archive = run_root / "source-bundle.tgz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(payload, arcname="source.txt")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (run_root / "source-bundle.sha256").write_text(
        f"{digest}  {archive}\n")
    payload.unlink()


def _make_case(campaign: Path, name: str) -> None:
    data = campaign / "cases" / f"{name}.npz"
    data.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        data,
        **{"declared.Y.0": np.asarray([[1.0]], dtype=np.float16),
           "fp64_math.Y.0": np.asarray([[1.0]], dtype=np.float64)})


def make_campaign(tmp: Path) -> tuple[Path, Path, Path]:
    run_root = tmp / "run"
    campaign = run_root / "campaign"
    board = run_root / "board"
    campaign.mkdir(parents=True)
    board.mkdir(parents=True)
    _write_source_bundle(run_root)
    (board / "library.sha256").write_text(f"{LIB_HASH}  remote/lib.so\n")
    cases = []
    for name in NAMES:
        _make_case(campaign, name)
        cases.append({
            "name": name, "kind": "matmul",
            "plan": f"plans/{name}.json", "data": f"cases/{name}.npz",
            "rounds": 1, "guard": None,
            "input_names": ["X"], "constant_names": [],
            "output_names": ["Y"], "output_shapes": {"Y": [1, 1]},
            "tolerance": {"atol": 8e-3, "rtol": 8e-3},
            "task_count": 1, "submit_count": 1,
            "templates": ["co.mm.f16.v1"],
        })
    manifest = {
        "schema": 1, "kind": "synthetic",
        "chip": "rk3588", "profile_hash": PROFILE,
        "expected_driver": "RKNPU driver: v0.9.8",
        "expected_runtime_abi_version": 1,
        "tolerance_contract": {"atol": 8e-3, "rtol": 8e-3},
        "cases": cases,
    }
    manifest_path = campaign / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    for mask in (1, 2, 4):
        core = board / f"core-{mask}"
        (core / "outputs").mkdir(parents=True)
        env = {
            "driver_version": "RKNPU driver: v0.9.8",
            "chip": "rk3588", "core_mask_env": str(mask),
            "target_check_rc": 0, "target_check_message": "",
            "expected_profile_hash": PROFILE,
            "runtime_abi_version": 1,
            "plan_lib_sha256": LIB_HASH,
        }
        (core / "environment.json").write_text(json.dumps(env))
        (core / "exit.txt").write_text("0\n")
        for name in NAMES:
            report = {
                "case": name, "status": "PASS", "kind": "matmul",
                "guard": None, "profile_hash": PROFILE,
                "task_count": 1, "submit_count": 1,
                "rounds": [{"round": 0, "task_count": 1,
                            "submit_count": 1}],
            }
            (core / f"{name}.json").write_text(json.dumps(report))
            arrays = {"output.Y.0": np.asarray([[1.0]], dtype=np.float16)}
            if name == "ffn_separate":
                # Same numeric value, wrong bytes/dtype: this must still fail the
                # byte-level chain/separate comparison.
                arrays["output.Y.0"] = np.asarray([[1.0]], dtype=np.float32)
            np.savez_compressed(core / "outputs" / f"{name}.npz", **arrays)
    return run_root, manifest_path, board


class BoardVerifierTests(unittest.TestCase):
    def run_verifier(self, manifest: Path, board: Path):
        env = dict(os.environ)
        env["PYTHONPATH"] = (str(ROOT / "python") +
                             (os.pathsep + env["PYTHONPATH"]
                              if env.get("PYTHONPATH") else ""))
        return subprocess.run(
            [sys.executable, str(ROOT / "tools" / "verify_board_campaign.py"),
             "--manifest", str(manifest), "--results", str(board)],
            cwd=str(ROOT), env=env, text=True, capture_output=True)

    def test_verifier_rejects_chain_separate_byte_or_dtype_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            _run, manifest, board = make_campaign(Path(tmp))
            proc = self.run_verifier(manifest, board)
            self.assertNotEqual(proc.returncode, 0)
            summary = json.loads((board / "summary.json").read_text())
            self.assertTrue(any("raw chain/separate mismatch" in p
                                for p in summary["problems"]))

    def test_verifier_rejects_source_bundle_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, manifest, board = make_campaign(Path(tmp))
            with open(run / "source-bundle.tgz", "ab") as fp:
                fp.write(b"tampered")
            proc = self.run_verifier(manifest, board)
            self.assertNotEqual(proc.returncode, 0)
            summary = json.loads((board / "summary.json").read_text())
            self.assertIn("source bundle hash recomputation mismatch",
                          summary["problems"])
            self.assertFalse(summary["artifact_verification"][
                "source_bundle_sha256"]["verified"])


if __name__ == "__main__":
    unittest.main()
