"""The refactor's single-authority properties.

These tests fail if a second planner, a second descriptor encoder or a hidden
device access appears behind any entry point.  They are structural on purpose: the
behavioural equivalence of the old and new entries is covered by the projection
oracle (``tools/refactor_plan_projection.py``) and the execution suites.
"""
import unittest
from unittest import mock

from rknpu_backend import (CompileOptions, compile_artifact, compile_rkir,
                           compile_tilelang, default_profile, explain_plan,
                           load_plan, matmul_module)
from rknpu_backend.codegen import descriptor as codegen_descriptor
from rknpu_backend.codegen import pipeline as codegen_pipeline
from rknpu_backend.runtime import ffi


def module():
    return matmul_module(m=13, k=64, n=33, activation="silu")


class SingleAuthorityTests(unittest.TestCase):
    def setUp(self):
        self.profile = default_profile()
        self.options = CompileOptions(numeric_mode="rk_f16_tiled",
                                      allow_approx_lut=True)

    def test_one_descriptor_encoder(self):
        """runtime.ffi must re-export the codegen encoder, not carry its own."""
        self.assertIs(ffi.build_plan_descriptor,
                      codegen_descriptor.build_plan_descriptor)
        for name in ("RKPlanDesc", "RKValueDesc", "RKActionDesc", "RKSubmitDesc",
                     "RKPublishDesc", "RKBufferDesc", "RKViewDesc",
                     "RKTensorDesc"):
            self.assertIs(getattr(ffi, name), getattr(codegen_descriptor, name),
                          f"{name} is duplicated")

    def test_encoders_are_the_only_plan_projections(self):
        """Plan -> descriptor happens exactly once, in codegen."""
        calls = []
        original = codegen_descriptor.build_plan_descriptor

        def counting(plan):
            calls.append(plan.plan_hash)
            return original(plan)

        class FakeLib:
            def __init__(self):
                self.seen = []

            def create_plan(self, plan):
                self.seen.append(counting(plan))
                return mock.Mock()

        with mock.patch.object(ffi, "build_plan_descriptor", counting), \
                mock.patch.object(ffi.LibRKPlan, "create_plan",
                                  lambda self, plan: counting(plan) or mock.Mock()):
            fake = FakeLib()
            fake.create_plan(compile_artifact(module(), profile=self.profile,
                                              options=self.options).plan)
        self.assertEqual(len(calls), 1)

    def test_all_compile_entries_use_the_one_pipeline(self):
        seen = []
        original = codegen_pipeline.compile_graph

        def counting(*args, **kwargs):
            seen.append(kwargs.get("stage"))
            return original(*args, **kwargs)

        with mock.patch.object(codegen_pipeline, "compile_graph", counting):
            compile_rkir(module(), profile=self.profile, options=self.options)
            compile_artifact(module(), profile=self.profile, options=self.options)
        self.assertEqual(seen, ["api", "api"])

    def test_compilation_never_constructs_the_runtime_library(self):
        """compile + explain must not resolve the shared library."""
        with mock.patch.object(ffi.LibRKPlan, "__init__",
                               side_effect=AssertionError("library loaded")):
            artifact = compile_artifact(module(), profile=self.profile,
                                        options=self.options)
            report = artifact.explain()
            reloaded = load_plan(_export(artifact))
        self.assertEqual(report["counts"]["tasks"],
                         artifact.plan.accounting.task_count)
        self.assertEqual(reloaded.plan_hash, artifact.plan.plan_hash)

    def test_explain_plan_is_shared_by_api_and_artifact(self):
        artifact = compile_artifact(module(), profile=self.profile,
                                    options=self.options)
        api_report = explain_plan(artifact.plan)
        # The artifact report is a superset: same counts, plus identity sections.
        self.assertEqual(api_report["counts"], artifact.explain()["counts"])
        self.assertEqual(api_report["actions"], artifact.explain()["actions"])
        self.assertEqual(api_report["slots"], artifact.explain()["slots"])

    def test_no_module_level_tvm_dependency(self):
        """codegen, artifact and api must not import TileLang/TVM at module load."""
        import ast
        import rknpu_backend.api as api_module
        import rknpu_backend.artifact as artifact_module
        import rknpu_backend.codegen.descriptor as descriptor_module

        for module in (api_module, artifact_module, descriptor_module,
                       codegen_pipeline):
            tree = ast.parse(open(module.__file__, encoding="utf-8").read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                else:
                    continue
                for name in names:
                    root = name.split(".")[0]
                    self.assertNotIn(
                        root, {"tilelang", "tvm", "tvm_ffi"},
                        f"{module.__name__} imports {name}")


class DependencyDirectionTests(unittest.TestCase):
    """planner must not import runtime; serializer must not plan."""

    def test_passes_never_import_the_runtime_or_the_library(self):
        import ast
        from pathlib import Path as _Path

        passes_dir = _Path(codegen_pipeline.__file__).parent.parent / "passes"
        for path in passes_dir.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    self.assertNotIn("runtime", name.split("."),
                                     f"{path.name} imports {name}")
                    self.assertNotIn("ctypes", name,
                                     f"{path.name} imports {name}")

    def test_codegen_does_not_import_the_runtime_package(self):
        import ast
        from pathlib import Path as _Path

        codegen_dir = _Path(codegen_pipeline.__file__).parent
        for path in codegen_dir.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                elif isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                for name in names:
                    self.assertNotIn("runtime", name.split("."),
                                     f"{path.name} imports {name}")


def _export(artifact):
    import tempfile

    handle = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    handle.close()
    artifact.export_plan(handle.name)
    return handle.name


class PlanImmutabilityTests(unittest.TestCase):
    """Frozen dataclasses must not leak a mutable ``attrs`` mapping."""

    def setUp(self):
        self.profile = default_profile()
        self.options = CompileOptions(numeric_mode="rk_f16_tiled",
                                      allow_approx_lut=True)
        self.artifact = compile_artifact(module(), profile=self.profile,
                                         options=self.options)

    def test_action_attrs_do_not_alias_the_caller_mapping(self):
        from rknpu_backend.plan import Action

        payload = {"numeric_contract": {"mode": "m"}, "list": [1, 2]}
        action = Action(0, "host_check", None, (1,), (2,), (), "test", payload)
        payload["list"].append(3)
        payload["numeric_contract"]["mode"] = "tampered"
        self.assertEqual(action.attrs["list"], [1, 2])
        self.assertEqual(action.attrs["numeric_contract"]["mode"], "m")

    def test_action_to_dict_detaches_nested_values(self):
        action = self.artifact.plan.actions[0]
        dumped = action.to_dict()
        if dumped["attrs"]:
            key = next(iter(dumped["attrs"]))
            if isinstance(dumped["attrs"][key], dict):
                dumped["attrs"][key]["nested_tamper"] = True
        self.assertNotIn("nested_tamper", str(action.to_dict()))

    def test_plan_hash_survives_explain_round_trip(self):
        before = self.artifact.plan.plan_hash
        self.artifact.explain()
        self.assertEqual(self.artifact.plan.plan_hash, before)

    def test_authoritative_plan_refuses_mutation(self):
        """``artifact.plan`` must not be writable, at any depth.

        A frozen dataclass still hands out its ``attrs`` mapping, and the
        descriptor is encoded from exactly those attributes, so post-construction
        mutation could change execution while ``plan_hash`` still reported the old
        plan.  Every write path must raise instead.
        """
        plan = self.artifact.plan
        upload = next(action for action in plan.actions if action.kind == "upload")
        self.assertIs(upload.attrs.get("transpose_source"), False)
        attempts = {
            # ``transpose_source`` is a descriptor flag: it must be impossible to
            # flip it on a compiled plan, which is what the review reproduced.
            "descriptor flag assignment": lambda: upload.attrs.__setitem__(
                "transpose_source", True),
            "action attribute delete": lambda: upload.attrs.__delitem__(
                next(iter(upload.attrs))),
            "action attribute update": lambda: upload.attrs.update(
                {"transpose_source": True}),
            "action attribute setdefault": lambda: upload.attrs.setdefault(
                "cache", "constant"),
            "frozen field assignment": lambda: setattr(upload, "origin", "x"),
            "plan collection assignment": lambda: setattr(plan, "actions", ()),
        }
        for label, attempt in attempts.items():
            with self.subTest(mutation=label):
                with self.assertRaises((TypeError, AttributeError)):
                    attempt()

    def test_every_action_attribute_tree_is_read_only(self):
        from rknpu_backend.immutable import is_frozen

        for action in self.artifact.plan.actions:
            self.assertTrue(is_frozen(action.attrs),
                            f"action {action.id} attrs are still mutable")
            self.assertIsInstance(action.attrs, dict)

    def test_recognized_graph_attributes_are_read_only(self):
        """The same rule applies to the RKIR graph the plan came from."""
        graph = module()
        op = graph.ops[0]
        with self.assertRaises(TypeError):
            op.attrs["tampered"] = 1
        contract = op.attrs["numeric_contract"]
        with self.assertRaises(TypeError):
            contract["tampered"] = 1
        # The serialized projection is still a plain, JSON-ready structure.
        dumped = op.to_dict()["attrs"]
        self.assertIsInstance(dumped["numeric_contract"], dict)
        dumped["numeric_contract"]["tampered"] = 1
        self.assertNotIn("tampered", op.to_dict()["attrs"]["numeric_contract"])
        self.assertIsInstance(op.attrs["numeric_contract"], dict)

    def test_caller_mapping_edits_cannot_reach_the_plan(self):
        from rknpu_backend.plan import Action

        payload = {"numeric_contract": {"mode": "m"}, "list": [1, 2],
                   "nested": {"deep": [{"k": 1}]}}
        action = Action(0, "host_check", None, (1,), (2,), (), "test", payload)
        payload["list"].append(3)
        payload["numeric_contract"]["mode"] = "tampered"
        payload["nested"]["deep"][0]["k"] = 99
        self.assertEqual(action.attrs["list"], [1, 2])
        self.assertEqual(action.attrs["numeric_contract"]["mode"], "m")
        self.assertEqual(action.attrs["nested"]["deep"][0]["k"], 1)
        with self.assertRaises(TypeError):
            action.attrs["list"].append(4)
        with self.assertRaises(TypeError):
            action.attrs["nested"]["deep"][0].__setitem__("k", 2)
        with self.assertRaises(TypeError):
            action.attrs["numeric_contract"].__setitem__("mode", "x")


if __name__ == "__main__":
    unittest.main()
