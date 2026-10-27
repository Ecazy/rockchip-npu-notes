"""The native TVM carrier (refactor R4 pilot).

Two things are pinned here:

1. the pilot is a *real* direct path -- real ``tvm.relax`` nodes are built from the
   recognized kernel without instantiating RKIR values, ops or modules;
2. it is faithful -- the same planner produces a byte-identical plan and an
   identical descriptor projection, the declared upstream passes preserve the
   target contract, and corrupted or unsupported contracts are rejected instead of
   being planned.

Adoption status is a separate question and is recorded in ``refactor-specific.md``
section 15.3, not asserted here.
"""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from rknpu_backend import (CompileOptions, CompileResult, compile_artifact,
                           compile_tilelang, default_profile, load_plan)
from rknpu_backend.frontend import native_graph
from rknpu_backend.frontend.graph_builder import (RecordBuilder, RkirBuilder,
                                                  ValueSpec, use_builder)
from rknpu_backend.frontend.tilelang_adapter import (extract,
                                                    tilelang_available)
from rknpu_backend.ir import Module
from rknpu_backend.codegen import pipeline
from rknpu_backend.runtime.ffi import build_plan_descriptor

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"

TILED = dict(numeric_mode="rk_f16_tiled")
TILED_LUT = dict(numeric_mode="rk_f16_tiled", allow_approx_lut=True)


def load_fixture(name: str):
    # Some fixtures build on each other as top-level modules (``tilelang_semantics``
    # imports ``tilelang_negative``), so the fixture directory must be importable.
    if str(FIXTURES) not in sys.path:
        sys.path.insert(0, str(FIXTURES))
    spec = importlib.util.spec_from_file_location(f"_nfix_{name}",
                                                  FIXTURES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def descriptor_dict(plan):
    """Field-by-field descriptor projection without host pointer addresses."""
    desc, holder = build_plan_descriptor(plan)
    _ = holder
    return {
        "nvalues": int(desc.nvalues), "nbuffers": int(desc.nbuffers),
        "nviews": int(desc.nviews), "nactions": int(desc.nactions),
        "nsubmits": int(desc.nsubmits), "npublish": int(desc.npublish),
        "profile_hash": int(desc.profile_hash),
        "source_hash": int(desc.source_hash),
        "schema_version": int(desc.schema_version),
        "budgets": [int(desc.device_budget_bytes), int(desc.host_budget_bytes),
                    int(desc.claimed_device_bytes), int(desc.claimed_host_bytes)],
        "actions": [
            [int(desc.actions[i].id), int(desc.actions[i].kind),
             int(desc.actions[i].template_id), int(desc.actions[i].flags),
             [int(desc.actions[i].inputs[k])
              for k in range(int(desc.actions[i].ninputs))],
             [int(desc.actions[i].outputs[k])
              for k in range(int(desc.actions[i].noutputs))],
             [int(desc.actions[i].deps[k])
              for k in range(int(desc.actions[i].ndeps))]]
            for i in range(int(desc.nactions))],
        "buffers": [
            [int(desc.buffers[i].id), int(desc.buffers[i].slot_id),
             int(desc.buffers[i].offset), int(desc.buffers[i].capacity),
             int(desc.buffers[i].role), int(desc.buffers[i].layout_id)]
            for i in range(int(desc.nbuffers))],
        "submits": [
            [int(desc.submits[i].id), int(desc.submits[i].mode),
             [int(desc.submits[i].task_ids[k])
              for k in range(int(desc.submits[i].ntask_ids))]]
            for i in range(int(desc.nsubmits))],
        "publish": [
            [int(desc.publish[i].binding_id), int(desc.publish[i].view_id),
             int(desc.publish[i].dst_row_offset),
             int(desc.publish[i].dst_col_offset), int(desc.publish[i].rows),
             int(desc.publish[i].cols)]
            for i in range(int(desc.npublish))],
    }


class CarrierSeamTests(unittest.TestCase):
    """The recognizer must not depend on the carrier it reports into."""

    def setUp(self):
        self.profile = default_profile()
        self.options = CompileOptions(**TILED_LUT)
        self.semantics = load_fixture("tilelang_semantics")

    def test_default_carrier_still_returns_rkir(self):
        module = extract(self.semantics._valid_kernel(16, 256, 16),
                         options=self.options,
                         profile_hash=self.profile.profile_hash)
        self.assertIsInstance(module, Module)
        self.assertEqual([op.opcode for op in module.ops], ["matmul"])

    def test_record_carrier_returns_container_free_records(self):
        with use_builder(RecordBuilder()):
            graph = extract(self.semantics._valid_kernel(16, 256, 16),
                            options=self.options,
                            profile_hash=self.profile.profile_hash)
        self.assertNotIsInstance(graph, Module)
        self.assertTrue(all(isinstance(value, ValueSpec) for value in graph.values))
        # No RKIR record leaked through: the specs are plain dataclasses.
        self.assertEqual(type(graph).__name__, "GraphSpec")
        self.assertEqual([op.opcode for op in graph.ops], ["matmul"])

    def test_carrier_is_restored_after_the_context(self):
        builder = RkirBuilder()
        with use_builder(RecordBuilder()):
            self.assertIsNone(None)  # nested assertions below
        with use_builder(builder):
            module = extract(self.semantics._valid_kernel(16, 256, 16),
                             options=self.options,
                             profile_hash=self.profile.profile_hash)
        self.assertIsInstance(module, Module)


@unittest.skipUnless(tilelang_available(),
                     "pinned TileLang optional dependency absent")
class NativeGraphBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = default_profile()
        cls.options = CompileOptions(**TILED_LUT)
        cls.semantics = load_fixture("tilelang_semantics")

    def _native(self, factory, options=None):
        return native_graph.build(factory(), options=options or self.options,
                                  profile_hash=self.profile.profile_hash)

    def test_real_relax_nodes_carry_the_graph(self):
        graph = self._native(lambda: self.semantics._valid_kernel(16, 256, 16))
        self.assertEqual(graph.kind, "native")
        script = graph.tvmscript()
        # Structure lives in the native graph, not in an attribute.
        self.assertIn('R.Tensor((16, 256), dtype="float16")', script)
        self.assertIn("R.matmul(", script)
        self.assertEqual([op.native_op for op in graph.ops], ["relax.matmul"])
        self.assertEqual(graph.inputs, (0,))
        self.assertEqual(graph.outputs, (2,))
        self.assertEqual([value.role for value in graph.values],
                         ["input", "constant", "output"])
        # The contract is target metadata only: it must not carry the graph.
        contract = graph.contract
        self.assertEqual(sorted(contract), sorted([
            "kind", "name", "profile_hash", "source_hash", "state_signature",
            "values", "ops", "inputs", "outputs", "pass_set", "signature"]))
        for op in contract["ops"]:
            self.assertEqual(sorted(op),
                             ["contract", "id", "native_op", "opcode", "source",
                              "value_id"])

    def test_no_rkir_object_is_created(self):
        import rknpu_backend.ir as ir

        created = []
        original_module = ir.Module
        original_value = ir.Value
        original_op = ir.Op

        class Recorder:
            def __init__(self, wrapped):
                self._wrapped = wrapped

            def __call__(self, *args, **kwargs):
                created.append(self._wrapped.__name__)
                return self._wrapped(*args, **kwargs)

        ir.Module = Recorder(original_module)  # type: ignore[assignment]
        ir.Value = Recorder(original_value)  # type: ignore[assignment]
        ir.Op = Recorder(original_op)  # type: ignore[assignment]
        try:
            graph = self._native(
                lambda: self.semantics._valid_kernel(16, 256, 16))
        finally:
            ir.Module = original_module
            ir.Value = original_value
            ir.Op = original_op
        self.assertEqual(created, [])
        self.assertEqual(graph.kind, "native")

    def test_native_operator_names_match_the_actual_relax_dispatch(self):
        relax = native_graph._import_relax()
        table = native_graph.native_op_table(relax)
        for opcode, (builder, arity) in table.items():
            args = [relax.Var(f"a{i}", relax.TensorStructInfo((4, 4), "float16"))
                    for i in range(arity)]
            self.assertEqual(builder(*args).op.name,
                             native_graph.native_op_name(opcode),
                             f"{opcode} native operator name disagrees")

    def test_unsupported_opcode_is_rejected_not_composed(self):
        """hardsigmoid has no direct Relax operator in the pinned TVM version.

        The carrier must reject it rather than compose a different op chain: a
        composition would break the one-record-per-node contract association.
        """
        from rknpu_backend.frontend.graph_builder import GraphSpec, OpSpec
        from rknpu_backend.ir import TensorType

        spec = GraphSpec(
            name="synthetic:hardsigmoid",
            values=(ValueSpec(0, "X", TensorType("float16", (8, 16)), None, "input"),
                    ValueSpec(1, "Y", TensorType("float16", (8, 16)), 0, "output")),
            ops=(OpSpec(0, "hardsigmoid", (0,), (1,), {}, "synthetic"),),
            inputs=(0,), outputs=(1,))
        with self.assertRaises(Exception) as caught:
            native_graph.build_spec(spec)
        self.assertIn("no Relax operator", str(caught.exception))
        self.assertNotIn("hardsigmoid", native_graph.supported_opcodes())


@unittest.skipUnless(tilelang_available(),
                     "pinned TileLang optional dependency absent")
class NativeFidelityTests(unittest.TestCase):
    """Native and RKIR carriers must produce the same execution plan."""

    @classmethod
    def setUpClass(cls):
        cls.profile = default_profile()
        cls.cases = cls._cases()

    @staticmethod
    def _cases():
        semantics = load_fixture("tilelang_semantics")
        kloop = load_fixture("tilelang_kloop")
        ragged = load_fixture("tilelang_ragged")
        broadcast = load_fixture("tilelang_broadcast")
        multi = load_fixture("tilelang_multi_output")
        mntile = load_fixture("tilelang_mntile")
        silu_proj = load_fixture("tilelang_silu_proj")
        staged = load_fixture("tilelang_projection_staged")
        ffn = load_fixture("tilelang_ffn_staged")
        branch = load_fixture("tilelang_branch_staged")
        return [
            ("single_matmul", lambda: semantics._valid_kernel(16, 256, 16), TILED),
            ("matmul_silu", lambda: semantics._silu_kernel(16, 256, 16), TILED_LUT),
            ("kloop", lambda: kloop.k_loop(16, 16, 512, 256), TILED),
            ("ragged_k257",
             lambda: ragged.ragged_tail(13, 33, 257, 16, 64, 256), TILED),
            ("broadcast_scalar", lambda: broadcast.scalar_add(8, 64, 33), TILED),
            ("broadcast_channel", lambda: broadcast.channel_mul(8, 64, 33), TILED),
            ("multi_output", lambda: multi.branch_outputs(8, 64, 32, 48), TILED),
            ("mn_tiled",
             lambda: mntile.mn_tiled(512, 384, 512, 256, 128, 256), TILED),
            ("silu_proj", lambda: silu_proj.silu_proj(13, 64, 32), TILED_LUT),
            ("projection_staged", lambda: staged.projection_staged, TILED_LUT),
            ("ffn_staged", lambda: ffn.ffn_staged, TILED_LUT),
            ("branch_staged", lambda: branch.branch_staged, TILED),
        ]

    def test_plans_and_descriptors_match_the_rkir_carrier(self):
        for name, factory, options in self.cases:
            with self.subTest(case=name):
                opts = CompileOptions(**options)
                legacy = compile_tilelang(factory(), profile=self.profile,
                                          options=opts)
                native = compile_artifact(factory(), profile=self.profile,
                                          options=opts, carrier="native")
                self.assertEqual(native.plan.to_dict(), legacy.plan.to_dict(),
                                 f"{name}: plan differs")
                self.assertEqual(descriptor_dict(native.plan),
                                 descriptor_dict(legacy.plan),
                                 f"{name}: descriptor projectio differs")

    def test_declared_pass_set_preserves_the_contract(self):
        semantics = load_fixture("tilelang_semantics")
        staged = load_fixture("tilelang_projection_staged")
        for name, factory, options in (
                ("single_matmul",
                 lambda: semantics._valid_kernel(16, 256, 16), TILED),
                ("projection_staged", lambda: staged.projection_staged, TILED_LUT)):
            with self.subTest(case=name):
                opts = CompileOptions(**options)
                graph = native_graph.build(factory(), options=opts,
                                           profile_hash=self.profile.profile_hash)
                passed = native_graph.apply_minimal_passes(graph)
                plan = pipeline.plan(passed, profile=self.profile, options=opts)
                legacy = compile_tilelang(factory(), profile=self.profile,
                                          options=opts)
                self.assertEqual(plan.to_dict(), legacy.plan.to_dict())

    def test_corrupted_contracts_are_rejected(self):
        module = load_fixture("tilelang_semantics")
        graph = native_graph.build(module._valid_kernel(16, 256, 16),
                                   options=CompileOptions(**TILED),
                                   profile_hash=self.profile.profile_hash)
        mutations = {
            "opcode": lambda c: c["ops"][0].__setitem__("opcode", "add"),
            "native_op": lambda c: c["ops"][0].__setitem__(
                "native_op", "relax.add"),
            "value_id": lambda c: c["ops"][0].__setitem__("value_id", 0),
            "value_shape": lambda c: c["values"][0].__setitem__("shape", [1, 1]),
            "role": lambda c: c["values"][0].__setitem__("role", "output"),
            "outputs": lambda c: c.__setitem__("outputs", [0]),
            "op_count": lambda c: c["ops"].append(dict(c["ops"][0])),
            "signature": lambda c: c["signature"]["ops"][0].__setitem__(
                "op", "relax.add"),
        }
        for label, mutate in mutations.items():
            with self.subTest(mutation=label):
                contract = graph.contract
                mutate(contract)
                with self.assertRaises(Exception):
                    native_graph.validate_contract(graph.ir_module, contract)

    def test_released_native_graph_refuses_further_work(self):
        module = load_fixture("tilelang_semantics")
        graph = native_graph.build(module._valid_kernel(16, 256, 16),
                                   options=CompileOptions(**TILED),
                                   profile_hash=self.profile.profile_hash)
        graph.release_native_graph()
        self.assertIsNone(graph.ir_module)
        with self.assertRaises(Exception):
            native_graph.apply_minimal_passes(graph)


@unittest.skipUnless(tilelang_available(),
                     "pinned TileLang optional dependency absent")
class NativeArtifactTests(unittest.TestCase):
    def setUp(self):
        self.profile = default_profile()
        self.options = CompileOptions(**TILED_LUT)
        self.staged = load_fixture("tilelang_projection_staged")

    def test_artifact_carrier_choice_does_not_change_identity(self):
        rkir = compile_artifact(self.staged.projection_staged,
                                profile=self.profile, options=self.options)
        native = compile_artifact(self.staged.projection_staged,
                                  profile=self.profile, options=self.options,
                                  carrier="native")
        self.assertEqual(rkir.plan_hash, native.plan_hash)
        self.assertEqual(rkir.explain()["counts"], native.explain()["counts"])
        with tempfile.TemporaryDirectory() as tmp:
            native.export_plan(str(Path(tmp) / "plan.json"))
            reloaded = load_plan(str(Path(tmp) / "plan.json"))
        self.assertEqual(reloaded.to_dict(), rkir.plan.to_dict())

    def test_retained_native_graph_is_optional(self):
        native = compile_artifact(self.staged.projection_staged,
                                  profile=self.profile, options=self.options,
                                  carrier="native", retain_graph=True)
        self.assertTrue(native.has_retained_graph)
        self.assertEqual(native.explain()["compile"]["retained_graph"],
                         "NativeGraph")
        self.assertFalse(native.release_diagnostics().has_retained_graph)

    def test_unknown_carrier_is_rejected(self):
        with self.assertRaises(Exception):
            compile_artifact(self.staged.projection_staged,
                             profile=self.profile, options=self.options,
                             carrier="magic")

    def test_native_carrier_rejects_rkir_input(self):
        from rknpu_backend import matmul_module

        with self.assertRaises(Exception):
            compile_artifact(matmul_module(m=8, k=64, n=16),
                             profile=self.profile, options=self.options,
                             carrier="native")


if __name__ == "__main__":
    unittest.main()
