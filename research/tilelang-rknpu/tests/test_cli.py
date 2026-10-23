import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rknpu_backend import (BackendError, CompileOptions, RK_E_DEVICE,
                           RK_E_INVALID, RK_E_NUMERIC_CONTRACT,
                           RK_E_POISONED, compile_rkir, default_profile,
                           matmul_module, save_plan)
from rknpu_backend.frontend.tilelang_adapter import tilelang_available
from rknpu_backend.serialize import canonical_json

ROOT = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def _env(self):
        """Child interpreter environment.

        Inherit the caller's environment (PYTHONPATH/LD_LIBRARY_PATH point at the
        locked dependency tree when the suite runs under the isolation venv) and
        only prepend this package, so the CLI subprocess resolves the same
        third-party modules as the test process itself.
        """
        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (str(ROOT / "python") +
                             (os.pathsep + existing if existing else ""))
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

    @unittest.skipUnless(tilelang_available(),
                         "pinned TileLang optional dependency absent")
    def test_compile_tilelang_silu_proj_through_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            plan = tmp / "silu_proj.rkplan.json"
            proc = self._run(
                "compile-tilelang", "--file",
                str(ROOT / "examples" / "silu_proj.py"), "--symbol", "kernel",
                "--numeric-mode", "rk_f16_tiled", "--allow-approx-lut",
                "--out", str(plan))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(plan.is_file())
            proc = self._run("validate", str(plan))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            proc = self._run("explain", str(plan), "--format", "json")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads(proc.stdout)
            self.assertEqual(report["counts"]["tasks"], 5)
            self.assertEqual(report["counts"]["submits"], 4)
            kinds = [action["kind"] for action in report["actions"]]
            self.assertEqual(kinds.count("upload"), 5)
            self.assertEqual(kinds.count("download"), 1)

    def test_run_uses_caller_fd_and_publishes_outputs(self):
        mock = ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan_mock.so"
        if not mock.is_file():
            self.skipTest(f"mock library not built: {mock}")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            plan = tmp / "matmul.rkplan.json"
            module = matmul_module(m=13, k=64, n=33)
            result = compile_rkir(module, profile=default_profile(),
                                  options=CompileOptions(numeric_mode="rk_f16_tiled"))
            save_plan(result.plan, plan)
            rng = np.random.default_rng(91)
            a = np.asarray(rng.normal(0, 0.3, (13, 64)), dtype=np.float16)
            b = np.asarray(rng.normal(0, 0.2, (64, 33)), dtype=np.float16)
            np.save(tmp / "A.npy", a)
            np.save(tmp / "B.npy", b)
            bindings = tmp / "bindings.json"
            bindings.write_text(json.dumps({"A": "A.npy", "B": "B.npy"}))
            bad = tmp / "bad.json"
            bad.write_text(json.dumps({"A": "A.npy"}))
            out = tmp / "out"
            bad_out = tmp / "bad-out"
            env = self._env()
            env["RKNPU_PLAN_LIB"] = str(mock)
            proc = subprocess.run(
                [sys.executable, "-m", "rknpu_backend.cli", "run", str(plan),
                 "--bindings", str(bindings), "--output-dir", str(out),
                 "--fd", "0"], cwd=str(ROOT), env=env, text=True,
                capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            got = np.load(out / "Y.npy", allow_pickle=False)
            expected = np.asarray(a.astype(np.float64) @ b.astype(np.float64),
                                  dtype=np.float16)
            np.testing.assert_array_equal(got, expected)

            proc = subprocess.run(
                [sys.executable, "-m", "rknpu_backend.cli", "run", str(plan),
                 "--bindings", str(bad), "--output-dir", str(bad_out),
                 "--fd", "0"], cwd=str(ROOT), env=env, text=True,
                capture_output=True)
            self.assertEqual(proc.returncode, 2, proc.stderr)
            self.assertFalse((bad_out / "Y.npy").exists())

    def test_cli_does_not_close_caller_owned_fd(self):
        mock = ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan_mock.so"
        if not mock.is_file():
            self.skipTest(f"mock library not built: {mock}")
        from rknpu_backend import cli
        from rknpu_backend.runtime import ffi
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            plan = tmp / "matmul.rkplan.json"
            module = matmul_module(m=8, k=64, n=33)
            result = compile_rkir(module, profile=default_profile(),
                                  options=CompileOptions(numeric_mode="rk_f16_tiled"))
            save_plan(result.plan, plan)
            rng = np.random.default_rng(92)
            a = np.asarray(rng.normal(0, 0.2, (8, 64)), dtype=np.float16)
            b = np.asarray(rng.normal(0, 0.2, (64, 33)), dtype=np.float16)
            np.save(tmp / "A.npy", a)
            np.save(tmp / "B.npy", b)
            bindings = tmp / "bindings.json"
            bindings.write_text(json.dumps({"A": "A.npy", "B": "B.npy"}))
            fd = os.open(os.devnull, os.O_RDONLY)
            old = ffi._DEFAULT_LIB
            ffi._DEFAULT_LIB = ffi.LibRKPlan(str(mock))
            try:
                rc = cli.main(["run", str(plan), "--bindings", str(bindings),
                               "--output-dir", str(tmp / "out"),
                               "--fd", str(fd)])
                self.assertEqual(rc, 0)
                os.fstat(fd)  # caller-owned fd must still be open
            finally:
                ffi._DEFAULT_LIB = old
                os.close(fd)

    def test_cli_reports_cleanup_failure_without_masking_run_error(self):
        import contextlib
        import io
        from unittest import mock
        from rknpu_backend import cli
        from rknpu_backend.runtime import bindings, ffi

        class FakeFn:
            def __init__(self, value):
                self.value = value
                self.calls = []

            def __call__(self, *args):
                self.calls.append(args)
                return self.value

        class FakeCDLL:
            def __init__(self):
                self.rocket_open = FakeFn(7)
                self.rocket_close = FakeFn(0)

        class FakeLib:
            def __init__(self):
                self.cdll = FakeCDLL()

        class FakeWorkspace:
            instance = None

            @classmethod
            def create(cls, *args, **kwargs):
                return cls.instance

            def __init__(self, *, run_error=None):
                self.run_error = run_error

            def run(self, inputs):
                if self.run_error is not None:
                    raise self.run_error
                return {"Y": np.zeros((8, 33), dtype=np.float16)}

            def close(self):
                raise BackendError(RK_E_POISONED,
                                   "workspace destroy failed", stage="cli")

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            plan = tmp / "plan.json"
            save_plan(compile_rkir(
                matmul_module(m=8, k=64, n=33), profile=default_profile(),
                options=CompileOptions(numeric_mode="rk_f16_tiled")).plan,
                plan)
            rng = np.random.default_rng(93)
            np.save(tmp / "A.npy", np.asarray(
                rng.normal(0, 0.2, (8, 64)), dtype=np.float16))
            np.save(tmp / "B.npy", np.asarray(
                rng.normal(0, 0.2, (64, 33)), dtype=np.float16))
            binding_path = tmp / "bindings.json"
            binding_path.write_text(json.dumps({"A": "A.npy", "B": "B.npy"}))

            for run_error, expected_rc in (
                    (None, 3),
                    (BackendError(RK_E_NUMERIC_CONTRACT,
                                  "numeric rejection", stage="cli"), 2)):
                FakeWorkspace.instance = FakeWorkspace(run_error=run_error)
                fake_lib = FakeLib()
                stderr = io.StringIO()
                with mock.patch.object(ffi, "default_library",
                                       return_value=fake_lib), \
                        mock.patch.object(bindings, "Workspace", FakeWorkspace), \
                        contextlib.redirect_stderr(stderr):
                    rc = cli.main(["run", str(plan), "--bindings",
                                   str(binding_path), "--output-dir",
                                   str(tmp / f"out-{expected_rc}")])
                self.assertEqual(rc, expected_rc, stderr.getvalue())
                self.assertIn("cleanup:", stderr.getvalue())
                if run_error is not None:
                    self.assertIn("RK_E_NUMERIC_CONTRACT", stderr.getvalue())
                self.assertEqual(len(fake_lib.cdll.rocket_open.calls), 1)
                self.assertEqual(len(fake_lib.cdll.rocket_close.calls), 1)

            # Caller-owned fd: neither rocket_open nor rocket_close is invoked.
            FakeWorkspace.instance = FakeWorkspace()
            fake_lib = FakeLib()
            with mock.patch.object(ffi, "default_library",
                                   return_value=fake_lib), \
                    mock.patch.object(bindings, "Workspace", FakeWorkspace), \
                    contextlib.redirect_stderr(io.StringIO()):
                rc = cli.main(["run", str(plan), "--bindings",
                               str(binding_path), "--output-dir",
                               str(tmp / "out-fd"), "--fd", "0"])
            self.assertEqual(rc, 3)
            self.assertEqual(fake_lib.cdll.rocket_open.calls, [])
            self.assertEqual(fake_lib.cdll.rocket_close.calls, [])

    def test_cli_error_exit_code_mapping(self):
        from unittest import mock
        from rknpu_backend import cli

        for code, expected in ((RK_E_INVALID, 2), (RK_E_DEVICE, 3)):
            with self.subTest(code=code), \
                    mock.patch.object(cli, "_cmd_run",
                                      side_effect=BackendError(code, "test",
                                                               stage="cli")):
                argv = ["run", "unused", "--bindings", "unused",
                        "--output-dir", "unused"]
                self.assertEqual(cli.main(argv), expected)

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
