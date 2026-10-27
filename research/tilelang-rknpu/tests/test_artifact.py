"""The compile-artifact boundary: device-free compile, direct instantiation.

These tests pin the properties the refactor promises for the new entry point:

* ``compile_artifact`` -> ``explain`` -> ``instantiate`` -> ``run`` needs no JSON
  round-trip and no prior ``export_plan``;
* ``explain`` cannot mutate the authoritative plan, and neither the artifact nor
  its report is a device handle;
* an artifact outlives the PrimFunc it came from;
* two instantiations are independent resource sets;
* the exported file reloads through the *existing* ``load_plan`` and runs with the
  same result as the in-memory artifact.
"""
import copy
import gc
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rknpu_backend import (CompiledArtifact, CompileOptions, compile_artifact,
                           compile_rkir, compile_tilelang, default_profile,
                           load_plan, matmul_module)
from rknpu_backend.frontend.tilelang_adapter import tilelang_available
from rknpu_backend.runtime import ffi
from rknpu_backend.runtime.bindings import Workspace

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
MOCK_LIB = os.environ.get("RKNPU_PLAN_MOCK_LIB") or str(
    ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan_mock.so")


def f16(x):
    return np.asarray(x, dtype=np.float16)


def kchain(x, w):
    a = x.astype(np.float64)
    b = w.astype(np.float64)
    acc = f16(a[:, :256] @ b[:256, :])
    for k0 in range(256, a.shape[1], 256):
        acc = f16(acc.astype(np.float64) + a[:, k0:k0 + 256] @ b[k0:k0 + 256, :])
    return acc


def load_fixture(module_name: str, filename: str):
    # Some fixtures build on each other as top-level modules
    # (``tilelang_semantics`` imports ``tilelang_negative``).
    if str(FIXTURES) not in sys.path:
        sys.path.insert(0, str(FIXTURES))
    spec = importlib.util.spec_from_file_location(module_name, FIXTURES / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ArtifactStructureTests(unittest.TestCase):
    """Properties that need no shared library at all."""

    def setUp(self):
        self.profile = default_profile()
        self.options = CompileOptions(numeric_mode="rk_f16_tiled")
        self.module = matmul_module(m=13, k=64, n=33, activation="silu")
        self.options = CompileOptions(numeric_mode="rk_f16_tiled",
                                      allow_approx_lut=True)

    def test_compile_artifact_matches_legacy_plan(self):
        artifact = compile_artifact(self.module, profile=self.profile,
                                    options=self.options)
        legacy = compile_rkir(self.module, profile=self.profile,
                              options=self.options)
        self.assertIsInstance(artifact, CompiledArtifact)
        self.assertEqual(artifact.plan.plan_hash, legacy.plan.plan_hash)
        self.assertEqual(artifact.plan.to_dict(), legacy.plan.to_dict())
        self.assertEqual(artifact.explain()["counts"],
                         legacy.report["counts"])

    def test_artifact_holds_no_device_or_caller_state(self):
        artifact = compile_artifact(self.module, profile=self.profile,
                                    options=self.options)
        # Nothing in the artifact tree is an fd, a numpy array, a ctypes handle or
        # a callable.  This is the "compilation product is not a runtime object"
        # rule expressed mechanically.
        def walk(value, path):
            self.assertFalse(callable(value) and not isinstance(value, type),
                             f"{path} is callable")
            self.assertNotIsInstance(value, np.ndarray, f"{path} is an array")
            self.assertNotIsInstance(value, ffi.RKBinding, f"{path} is a binding")
            if hasattr(value, "__dict__") and not isinstance(value, type):
                for key, child in vars(value).items():
                    walk(child, f"{path}.{key}")
            elif isinstance(value, (tuple, list)):
                for index, child in enumerate(value):
                    walk(child, f"{path}[{index}]")

        walk(artifact, "artifact")
        self.assertFalse(hasattr(artifact, "fd"))
        self.assertFalse(artifact.has_retained_graph)

    def test_explain_report_cannot_mutate_the_plan(self):
        artifact = compile_artifact(self.module, profile=self.profile,
                                    options=self.options)
        before = artifact.plan.to_dict()
        report = artifact.explain()
        # Mutate every place a caller could reach execution semantics.
        for action in report["actions"]:
            action["attrs"]["source_binding"] = 999
            action["deps"].append(12345)
            action["inputs"].append(12345)
        report["submits"][0]["task_ids"].append(999)
        report["counts"]["tasks"] = -1
        report["accounting"]["device_request_bytes"] = 1
        report["source"]["narrowing_points"].append("bogus")
        report["policy"]["numeric_mode"] = "tampered"
        self.assertEqual(artifact.plan.to_dict(), before)
        fresh = artifact.explain()
        self.assertNotEqual(fresh["counts"]["tasks"], -1)
        self.assertNotEqual(fresh["policy"]["numeric_mode"], "tampered")
        self.assertNotIn("bogus", fresh["source"]["narrowing_points"])
        for action in fresh["actions"]:
            self.assertNotIn(12345, action["deps"])
            self.assertNotEqual(action["attrs"].get("source_binding"), 999)

    def test_explain_reports_the_actual_compile_snapshot(self):
        artifact = compile_artifact(self.module, profile=self.profile,
                                    options=self.options)
        report = artifact.explain()
        self.assertEqual(report["policy"]["numeric_mode"], "rk_f16_tiled")
        self.assertEqual(report["policy"]["fallback"], "error")
        self.assertTrue(report["policy"]["approx_lut"])
        self.assertEqual(report["compile"]["options"]["allow_approx_lut"], True)
        self.assertEqual(report["compile"]["profile"]["profile_hash"],
                         self.profile.profile_hash)
        self.assertEqual(report["source"]["module"], self.module.name)
        self.assertEqual(report["compile"]["retained_graph"], None)
        self.assertEqual(report["policy"]["device_access"],
                         "none (compile/explain)")

    def test_options_snapshot_is_not_a_live_reference(self):
        options = CompileOptions(numeric_mode="rk_f16_tiled",
                                 allow_approx_lut=True)
        artifact = compile_artifact(self.module, profile=self.profile,
                                    options=options)
        # CompileOptions is frozen; the artifact must report the snapshot, not a
        # caller-owned mutable object.
        self.assertIs(artifact.options, options)
        self.assertNotIsInstance(type(artifact.options).allow_approx_lut, dict)
        with self.assertRaises(Exception):
            options.allow_approx_lut = False  # frozen dataclass

    def test_descriptor_cannot_be_steered_by_editing_the_plan(self):
        """The review's reproduction: flip an upload flag through ``artifact.plan``.

        The descriptor is encoded from ``Action.attrs``, so a writable ``attrs``
        could change execution while ``plan_hash`` still reported the old plan.
        Every attempt must be refused, and the encoded descriptor must be
        unchanged afterwards.
        """
        from rknpu_backend.runtime.ffi import build_plan_descriptor

        artifact = compile_artifact(self.module, profile=self.profile,
                                    options=self.options)

        def projection():
            desc, holder = build_plan_descriptor(artifact.plan)
            _ = holder
            return {
                "flags": [int(desc.actions[i].flags)
                          for i in range(int(desc.nactions))],
                "source_binding": [int(desc.actions[i].source_binding)
                                   for i in range(int(desc.nactions))],
                "kinds": [int(desc.actions[i].kind)
                          for i in range(int(desc.nactions))],
                "counts": [int(desc.nvalues), int(desc.nbuffers),
                           int(desc.nviews), int(desc.nactions),
                           int(desc.nsubmits), int(desc.npublish)],
            }

        before = projection()
        upload = next(action for action in artifact.plan.actions
                      if action.kind == "upload")
        with self.assertRaises(TypeError):
            upload.attrs["transpose_source"] = True
        with self.assertRaises(TypeError):
            upload.attrs.update({"transpose_source": True})
        with self.assertRaises(TypeError):
            upload.attrs.clear()
        with self.assertRaises((TypeError, AttributeError)):
            setattr(upload, "origin", "tampered")
        with self.assertRaises((TypeError, AttributeError)):
            setattr(artifact.plan, "actions", ())
        self.assertEqual(projection(), before)
        self.assertEqual(artifact.plan.plan_hash, artifact.plan_hash)

    def test_retained_graph_is_optional_and_releasable(self):
        artifact = compile_artifact(self.module, profile=self.profile,
                                    options=self.options)
        self.assertFalse(artifact.has_retained_graph)
        retained = artifact.with_retained_graph({"diagnostic": True})
        self.assertTrue(retained.has_retained_graph)
        self.assertEqual(retained.explain()["compile"]["retained_graph"], "dict")
        released = retained.release_diagnostics()
        self.assertFalse(released.has_retained_graph)
        # Dropping the diagnostic graph cannot change identity or the plan.
        self.assertEqual(released.plan.to_dict(), artifact.plan.to_dict())
        self.assertEqual(released.explain()["counts"], artifact.explain()["counts"])

    def test_instantiate_requires_a_caller_fd(self):
        artifact = compile_artifact(self.module, profile=self.profile,
                                    options=self.options)
        with self.assertRaises(TypeError):
            artifact.instantiate(constants={})
        with self.assertRaises(Exception):
            artifact.instantiate(fd=-1, constants={})

    def test_import_and_compile_happen_without_the_device_library(self):
        """No shared library is resolved by importing or compiling."""
        from rknpu_backend.codegen import descriptor
        self.assertTrue(hasattr(descriptor, "build_plan_descriptor"))
        artifact = compile_artifact(self.module, profile=self.profile,
                                    options=self.options)
        descriptor.build_plan_descriptor(artifact.plan)


class ArtifactIsolationTests(unittest.TestCase):
    """The legacy paths keep working when TileLang/TVM is unavailable."""

    def test_no_tvm_import_at_package_import(self):
        code = (
            "import sys, rknpu_backend\n"
            "assert 'tvm' not in sys.modules, 'tvm imported eagerly'\n"
            "assert 'tilelang' not in sys.modules, 'tilelang imported eagerly'\n"
            "print('ok')\n")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "python")
        # Block the optional tree entirely to prove it is not imported.
        env["PYTHONPATH"] += os.pathsep + str(ROOT / "tests" / "nodeps")
        result = subprocess.run([sys.executable, "-c", code], env=env,
                                cwd=str(ROOT), text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)

    def test_compile_rkir_and_replay_without_tilelang(self):
        code = r"""
import sys, tempfile, os
import rknpu_backend
from rknpu_backend import (CompileOptions, compile_artifact, compile_rkir,
                           default_profile, load_plan, matmul_module)
assert 'tilelang' not in sys.modules
profile = default_profile()
options = CompileOptions(numeric_mode="rk_f16_tiled")
module = matmul_module(m=13, k=64, n=33)
artifact = compile_artifact(module, profile=profile, options=options)
legacy = compile_rkir(module, profile=profile, options=options)
assert artifact.plan.plan_hash == legacy.plan.plan_hash
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "plan.json")
    artifact.export_plan(path)
    reloaded = load_plan(path)
    assert reloaded.plan_hash == artifact.plan.plan_hash
    assert reloaded.to_dict() == artifact.plan.to_dict()
print("ok")
"""
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "python")
        env["PYTHONPATH"] += os.pathsep + str(ROOT / "tests" / "nodeps")
        env.pop("RKNPU_PLAN_LIB", None)
        result = subprocess.run([sys.executable, "-c", code], env=env,
                                cwd=str(ROOT), text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)


@unittest.skipUnless(Path(MOCK_LIB).is_file(), f"mock library not built: {MOCK_LIB}")
class ArtifactExecutionTests(unittest.TestCase):
    def setUp(self):
        self.profile = default_profile()
        self.options = CompileOptions(numeric_mode="rk_f16_tiled",
                                      allow_approx_lut=True)
        self.module = matmul_module(m=13, k=300, n=33, activation="silu")
        self._arrays = []
        lib = ffi.LibRKPlan(MOCK_LIB)
        self._old_default = ffi._DEFAULT_LIB
        ffi._DEFAULT_LIB = lib
        self.addCleanup(self._restore)
        bos = lib.mock_control("rk_mock_live_bos")
        res = lib.mock_control("rk_mock_resident")
        self.addCleanup(self.assert_no_leak, lib, bos, res)

    def _restore(self):
        lib = ffi._DEFAULT_LIB
        try:
            if lib is not None:
                lib.cdll.rk_mock_reset()
        finally:
            ffi._DEFAULT_LIB = self._old_default
        self._arrays.clear()

    def assert_no_leak(self, lib, bos, res):
        self.assertEqual(lib.mock_control("rk_mock_untracked"), 0)
        self.assertEqual(lib.mock_control("rk_mock_tag_overflow"), 0)
        self.assertEqual(lib.mock_control("rk_mock_live_bos"), bos)
        self.assertEqual(lib.mock_control("rk_mock_resident"), res)

    @staticmethod
    def _reference(x, w):
        z = kchain(x, w)
        sig = f16(1.0 / (1.0 + np.exp(-z.astype(np.float32))))
        return f16(z.astype(np.float32) * sig.astype(np.float32))

    def _call(self, artifact):
        """Inputs and constants for the plan, plus the expected result."""
        rng = np.random.default_rng(17)
        inputs = {b.name: f16(rng.normal(0, 0.15, b.shape))
                  for b in artifact.plan.bindings if b.role == "input"}
        constants = {}
        for spec in artifact.plan.constants:
            # ``identity`` constants are materialised by the runtime itself.
            if spec.generator == "identity":
                continue
            constants[spec.name] = f16(rng.normal(0, 0.1, spec.shape))
        (input_name,) = tuple(inputs)
        (constant_name,) = tuple(constants)
        outputs = [b.name for b in artifact.plan.bindings if b.role == "output"]
        return (inputs, constants,
                {outputs[0]: self._reference(inputs[input_name],
                                             constants[constant_name])})

    def test_compile_instantiate_run_without_json(self):
        artifact = compile_artifact(self.module, profile=self.profile,
                                    options=self.options)
        inputs, constants, expected = self._call(artifact)
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                workspace = artifact.instantiate(fd=0, constants=constants)
                try:
                    result = workspace.run(inputs)
                finally:
                    workspace.close()
            finally:
                os.chdir(cwd)
            # The whole compile -> instantiate -> run path must not have written a
            # plan or any other intermediate file.
            self.assertEqual(os.listdir(tmp), [])
        for name, array in result.items():
            self.assertEqual(array.shape,
                             next(b.shape for b in artifact.plan.bindings
                                  if b.name == name))
            self.assertTrue(np.isfinite(array).all())
        for name, reference in expected.items():
            np.testing.assert_allclose(result[name].astype(np.float64),
                                       reference.astype(np.float64),
                                       atol=8e-3, rtol=8e-3, equal_nan=False)

    def test_two_instantiations_are_independent(self):
        artifact = compile_artifact(self.module, profile=self.profile,
                                    options=self.options)
        inputs, constants, _ = self._call(artifact)
        first = artifact.instantiate(fd=0, constants=constants)
        second = artifact.instantiate(fd=0, constants=constants)
        try:
            a = first.run(inputs)
            b = second.run(inputs)
            for name in a:
                np.testing.assert_array_equal(a[name], b[name])
        finally:
            first.close()
            second.close()

    def test_export_load_run_matches_in_memory_run(self):
        artifact = compile_artifact(self.module, profile=self.profile,
                                    options=self.options)
        inputs, constants, _expected = self._call(artifact)
        direct = artifact.instantiate(fd=0, constants=constants)
        try:
            expected_run = direct.run(inputs)
        finally:
            direct.close()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "plan.json")
            artifact.export_plan(path)
            reloaded = load_plan(path)
            self.assertEqual(reloaded.to_dict(), artifact.plan.to_dict())
            replayed = Workspace.create(reloaded, fd=0, constants=constants)
            try:
                actual = replayed.run(inputs)
            finally:
                replayed.close()
        for name in expected_run:
            np.testing.assert_array_equal(expected_run[name], actual[name])

    def test_attempted_plan_edits_do_not_change_execution(self):
        artifact = compile_artifact(self.module, profile=self.profile,
                                    options=self.options)
        inputs, constants, expected = self._call(artifact)

        def one_run():
            workspace = artifact.instantiate(fd=0, constants=constants)
            try:
                return workspace.run(inputs)
            finally:
                workspace.close()

        reference = one_run()
        for action in artifact.plan.actions:
            for key in list(action.attrs):
                with self.assertRaises(TypeError):
                    action.attrs[key] = 1
            with self.assertRaises(TypeError):
                action.attrs["submit_group"] = "tampered"
        again = one_run()
        for name in reference:
            np.testing.assert_array_equal(reference[name], again[name])
        for name, wanted in expected.items():
            np.testing.assert_allclose(again[name].astype(np.float64),
                                       wanted.astype(np.float64),
                                       atol=8e-3, rtol=8e-3, equal_nan=False)

    @unittest.skipUnless(tilelang_available(),
                         "pinned TileLang optional dependency absent")
    def test_primfunc_released_before_instantiate_and_run(self):
        """A compiled artifact must not keep the source PrimFunc alive.

        ``tilelang_semantics._valid_kernel`` is a plain factory, so the object it
        returns has no other owner: a weak reference dying proves no strong
        reference survives anywhere in the process, including inside the artifact.
        The run afterwards then goes through the production workspace path, so
        "execution does not depend on the source module" is a tested property
        rather than a comment.
        """
        import weakref

        semantics = load_fixture("artifact_semantics", "tilelang_semantics.py")
        # Negative control: a live reference must keep a weakref alive, otherwise
        # the assertion below could pass for the wrong reason.
        control = semantics._valid_kernel(16, 256, 16)
        control_ref = weakref.ref(control)
        gc.collect()
        self.assertIsNotNone(control_ref())
        del control
        gc.collect()
        self.assertIsNone(control_ref())

        prim = semantics._valid_kernel(16, 256, 16)
        reference = weakref.ref(prim)
        artifact = compile_artifact(prim, profile=self.profile,
                                    options=self.options)
        plan_dict = copy.deepcopy(artifact.plan.to_dict())
        del prim
        gc.collect()
        self.assertIsNone(reference(),
                          "the artifact still keeps the source PrimFunc alive")
        self.assertFalse(artifact.has_retained_graph)
        self.assertEqual(artifact.plan.to_dict(), plan_dict)

        rng = np.random.default_rng(31)
        x = f16(rng.normal(0, 0.15, (16, 256)))
        w = f16(rng.normal(0, 0.1, (256, 16)))
        inputs = {b.name: x for b in artifact.plan.bindings if b.role == "input"}
        constants = {spec.name: w for spec in artifact.plan.constants
                     if spec.generator is None}
        self.assertEqual(len(constants), 1)
        outputs = [b.name for b in artifact.plan.bindings if b.role == "output"]
        workspace = artifact.instantiate(fd=0, constants=constants)
        try:
            result = workspace.run(inputs)
        finally:
            workspace.close()
        np.testing.assert_allclose(result[outputs[0]].astype(np.float64),
                                   kchain(x, w).astype(np.float64),
                                   atol=8e-3, rtol=8e-3, equal_nan=False)

    def test_artifact_survives_source_module_release(self):
        module = matmul_module(m=13, k=300, n=33, activation="silu")
        artifact = compile_artifact(module, profile=self.profile,
                                    options=self.options)
        expected = artifact.plan.to_dict()
        del module
        gc.collect()
        self.assertEqual(artifact.plan.to_dict(), expected)
        report = artifact.explain()
        self.assertEqual(report["counts"]["tasks"],
                         artifact.plan.accounting.task_count)


@unittest.skipUnless(tilelang_available(),
                     "pinned TileLang optional dependency absent")
class ArtifactTileLangTests(unittest.TestCase):
    """TileLang-specific end-to-end checks for the artifact entry point."""

    @classmethod
    def setUpClass(cls):
        cls.profile = default_profile()
        cls.options = CompileOptions(numeric_mode="rk_f16_tiled",
                                     allow_approx_lut=True)
        cls.projection = load_fixture("artifact_projection",
                                      "tilelang_projection_staged.py")

    def test_artifact_matches_compile_tilelang(self):
        artifact = compile_artifact(self.projection.projection_staged,
                                    profile=self.profile, options=self.options)
        legacy = compile_tilelang(self.projection.projection_staged,
                                  profile=self.profile, options=self.options)
        self.assertEqual(artifact.plan.plan_hash, legacy.plan.plan_hash)
        self.assertEqual(artifact.explain()["counts"], legacy.report["counts"])
        self.assertEqual(artifact.explain()["source"]["module"],
                         legacy.report["source"]["module"])

    def test_compile_result_exposes_the_same_artifact(self):
        legacy = compile_tilelang(self.projection.projection_staged,
                                  profile=self.profile, options=self.options)
        artifact = legacy.artifact
        self.assertIsInstance(artifact, CompiledArtifact)
        self.assertEqual(artifact.plan.plan_hash, legacy.plan.plan_hash)
        self.assertIs(artifact.plan, legacy.plan)

    def test_artifact_outlives_the_primfunc_object(self):
        prim = self.projection.projection_staged
        artifact = compile_artifact(prim, profile=self.profile,
                                    options=self.options)
        plan_dict = copy.deepcopy(artifact.plan.to_dict())
        del prim
        gc.collect()
        self.assertEqual(artifact.plan.to_dict(), plan_dict)
        self.assertFalse(artifact.has_retained_graph)

    def test_retained_graph_does_not_gate_export(self):
        artifact = compile_artifact(self.projection.projection_staged,
                                    profile=self.profile, options=self.options,
                                    retain_graph=True)
        self.assertTrue(artifact.has_retained_graph)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "plan.json")
            artifact.export_plan(path)
            reloaded = load_plan(path)
            self.assertEqual(reloaded.to_dict(), artifact.plan.to_dict())
            payload = json.loads(Path(path).read_text())
            self.assertEqual(payload["kind"], "rkplan")
            self.assertNotIn("retained_graph", payload)


if __name__ == "__main__":
    unittest.main()
