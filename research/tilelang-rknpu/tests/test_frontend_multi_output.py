"""Real TileLang multi-output branch frontend extraction."""
import importlib.util
import sys
import unittest
from pathlib import Path

from rknpu_backend import CompileOptions, compile_tilelang, default_profile
from rknpu_backend.frontend.tilelang_adapter import (MULTI_OUTPUT_NARROWING_NAMES,
                                                     extract,
                                                     tilelang_available)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def load_fixture():
    if str(FIXTURES) not in sys.path:
        sys.path.insert(0, str(FIXTURES))
    spec = importlib.util.spec_from_file_location(
        "tl_multi_output_frontend", str(FIXTURES / "tilelang_multi_output.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(tilelang_available(), "pinned TileLang optional dependency absent")
class MultiOutputFrontendTests(unittest.TestCase):
    def setUp(self):
        self.profile = default_profile()
        self.fixture = load_fixture()

    def test_shared_input_branch_extracts_two_ssa_outputs(self):
        kernel = self.fixture.branch_outputs(m=8, k=64, n1=32, n2=48)
        options = CompileOptions(numeric_mode="rk_f16_tiled")
        module = extract(kernel, options=options,
                         profile_hash=self.profile.profile_hash)
        self.assertEqual([op.opcode for op in module.ops],
                         ["matmul", "matmul"])
        self.assertEqual([(v.name, v.type.shape, v.role)
                          for v in module.values],
                         [("X", (8, 64), "input"),
                          ("W1", (64, 32), "constant"),
                          ("W2", (64, 48), "constant"),
                          ("Y1", (8, 32), "output"),
                          ("Y2", (8, 48), "output")])
        self.assertEqual(module.ops[0].inputs[0], module.ops[1].inputs[0])
        self.assertEqual(module.outputs, (3, 4))
        self.assertEqual(module.ops[0].attrs["source_narrowing_points"],
                         list(MULTI_OUTPUT_NARROWING_NAMES))

    def test_plan_downloads_and_publishes_each_output(self):
        result = compile_tilelang(
            self.fixture.branch_outputs(m=8, k=64, n1=32, n2=48),
            profile=self.profile,
            options=CompileOptions(numeric_mode="rk_f16_tiled"))
        plan = result.plan
        tasks = [a for a in plan.actions if a.kind == "npu_task"]
        uploads = [a for a in plan.actions if a.kind == "upload"]
        downloads = [a for a in plan.actions if a.kind == "download"]
        self.assertEqual(len(tasks), 2)
        self.assertEqual(len(uploads), 3)
        self.assertEqual(len(downloads), 2)
        self.assertEqual(len(plan.publish), 2)
        self.assertLess(max(a.id for a in uploads), min(a.id for a in tasks))
        self.assertEqual([b.name for b in plan.bindings if b.role == "output"],
                         ["Y1", "Y2"])


if __name__ == "__main__":
    unittest.main()
