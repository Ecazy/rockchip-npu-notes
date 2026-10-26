"""Complete-model milestone tests: real TileLang source, learned weights, hybrid runner."""
from __future__ import annotations

import gzip
import importlib.util
import json
import os
import struct
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MOCK_LIB = os.environ.get("RKNPU_PLAN_MOCK_LIB") or str(
    ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan_mock.so")
PLANS = ROOT / "models" / "compiled"
WEIGHTS = ROOT / "models"


def read_idx(path: Path, rank: int) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        f.read(4)
        dims = struct.unpack(">" + "I" * rank, f.read(4 * rank))
        return np.frombuffer(f.read(), np.uint8).reshape(dims)


class CompleteModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not Path(MOCK_LIB).is_file():
            raise unittest.SkipTest(f"mock library not built: {MOCK_LIB}")
        from rknpu_backend import default_profile
        from rknpu_backend.runtime.ffi import LibRKPlan
        cls.lib = LibRKPlan(MOCK_LIB)
        cls.profile = default_profile()
        cls.x = read_idx(ROOT / ".data" / "mnist" /
                         "t10k-images-idx3-ubyte.gz", 3)
        cls.y = read_idx(ROOT / ".data" / "mnist" /
                         "t10k-labels-idx1-ubyte.gz", 1).astype(np.int64)
        cls.mlp_weights = dict(np.load(WEIGHTS / "mnist_mlp.npz",
                                       allow_pickle=False))
        cls.vit_weights = dict(np.load(WEIGHTS / "tiny_vit.npz",
                                       allow_pickle=False))

    def _plan(self, name):
        from rknpu_backend import load_plan
        return load_plan(str(PLANS / f"{name}.rkplan.json"))

    def _hybrid(self, name):
        return PLANS / f"{name}.rkhybrid.json"

    def test_weight_manifest_and_learned_model_baselines(self):
        manifest = json.loads((ROOT / "models" / "manifest.json").read_text())
        names = {m["name"]: m for m in manifest["models"]}
        self.assertGreaterEqual(names["mnist_mlp"]["test_accuracy"], 0.97)
        self.assertGreaterEqual(names["tiny_vit"]["test_accuracy"], 0.95)
        for name in ("mnist_mlp.npz", "tiny_vit.npz"):
            sha = names["mnist_mlp" if name.startswith("mnist") else "tiny_vit"]["weights_sha256"]
            import hashlib
            self.assertEqual(hashlib.sha256((WEIGHTS / name).read_bytes()).hexdigest(), sha)

    def test_mlp_plan_contains_exact_relu_and_device_residency(self):
        plan = self._plan("mnist_mlp_b32")
        templates = [a.template_id for a in plan.actions if a.kind == "npu_task"]
        self.assertIn("co.ew_max.f16.v1", templates)
        self.assertEqual(templates.count("co.ew_max.f16.v1"), 2)
        self.assertEqual(sum(a.kind == "download" for a in plan.actions), 1)
        self.assertEqual([b.shape for b in plan.bindings if b.role == "input"],
                         [(32, 1024)])

    def test_mlp_source_compiles_from_real_tilelang(self):
        from models.tilelang_mlp import make_mlp_prim
        from rknpu_backend import CompileOptions, compile_tilelang
        result = compile_tilelang(
            make_mlp_prim(1), profile=self.profile,
            options=CompileOptions(numeric_mode="rk_f16_tiled",
                                   allow_approx_lut=False))
        self.assertIn("co.ew_max.f16.v1",
                      [a.template_id for a in result.plan.actions
                       if a.kind == "npu_task"])
        self.assertEqual(len([a for a in result.plan.actions
                              if a.kind == "download"]), 1)

    def test_mlp_learned_weights_input_update_and_batches(self):
        from rknpu_backend.models import MLPRunner
        from rknpu_backend.model_reference import mlp_logits_declared
        for batch in (1, 8, 32):
            runner = MLPRunner(self.lib, self._hybrid(f"mnist_mlp_b{batch}"),
                               self.mlp_weights, batch, fd=0,
                               weights_path=WEIGHTS / "mnist_mlp.npz")
            try:
                for start in (0, batch):
                    raw = self.x[start:start + batch]
                    got = runner.run(raw)
                    ref = mlp_logits_declared(self.mlp_weights, raw)
                    np.testing.assert_allclose(got, ref, atol=0.08, rtol=0.02)
                    self.assertEqual(got.shape, (batch, 10))
            finally:
                runner.close()

    def test_timing_phases_are_explicit_and_reference_is_outside(self):
        from rknpu_backend.models import MLPRunner, RunStats, TIMING_SCHEMA_VERSION
        runner = MLPRunner(self.lib, self._hybrid("mnist_mlp_b1"),
                           self.mlp_weights, 1, fd=0,
                           weights_path=WEIGHTS / "mnist_mlp.npz")
        stats = RunStats()
        try:
            runner.run(self.x[:1], stats=stats)
        finally:
            runner.close()
        data = stats.to_dict()
        self.assertEqual(data["timing_schema_version"], TIMING_SCHEMA_VERSION)
        self.assertEqual(data["reference_timing"], "excluded")
        self.assertEqual(data["run_calls"], 1)
        self.assertGreaterEqual(data["end_to_end_us"], 0.0)
        phase_sum = (data["preprocess_us"] + data["execution_us"] +
                     data["postprocess_us"])
        self.assertLessEqual(phase_sum, data["end_to_end_us"] + 1.0)
        action_names = {row["name"] for row in data["host_actions"]}
        self.assertIn("normalize", action_names)
        self.assertIn("pad_k", action_names)
        self.assertIn("export_logits", action_names)
        self.assertNotIn("reference", " ".join(action_names).lower())

    def test_repeat_submit_policies_agree_bitwise(self):
        """The three declared repeat paths must produce identical logits.

        `manifest` uses the node's declared policy (shared group for QK/PV),
        `b1` keeps the per-repeat group entry and `separate` issues one call per
        repeat.  Keeping the controls executable is what makes the default
        switch reversible.
        """
        from rknpu_backend.hybrid import HybridRunner, load_hybrid_program
        program = load_hybrid_program(self._hybrid("tiny_vit_b8"))
        results = {}
        reports = {}
        for mode in ("manifest", "b1", "separate", "chain_shared"):
            runner = HybridRunner(self.lib, program, self.vit_weights,
                                  weights_path=WEIGHTS / "tiny_vit.npz", fd=0,
                                  repeat_submit=mode)
            stats = __import__("rknpu_backend.models",
                               fromlist=["RunStats"]).RunStats()
            try:
                results[mode] = runner.run(self.x[:8], stats=stats).outputs["logits"]
                reports[mode] = (stats.submit_count, stats.task_count)
            finally:
                runner.close()
        for mode in ("b1", "separate", "chain_shared"):
            self.assertTrue(np.array_equal(results["manifest"], results[mode]),
                            f"{mode} path differs from the manifest path")
        # Structural difference, per batch-8 call: 2 chained QK/PV submits plus
        # the 7 other nodes, versus 64 per-repeat submits in the B1 path.
        self.assertEqual(reports["manifest"], (8, 70))
        self.assertEqual(reports["b1"], (70, 70))
        self.assertEqual(reports["separate"], (70, 70))
        self.assertEqual(reports["chain_shared"], (8, 70))

    def test_plan_envelope_attribution_tiles_the_whole_call(self):
        """The C attribution must explain the entire RKPlan dwell.

        This is the invariant the first optimisation package depends on: every
        named interval is disjoint, their sum plus the reported residual covers
        the C-side total, and the instrumentation itself is not a hidden cost.
        It also pins the double-counting regression where the four legacy fields
        were accumulated both explicitly and through the envelope field list.
        """
        from rknpu_backend.models import (ENVELOPE_FIELDS,
                                          ENVELOPE_WITNESS_FIELDS, RunStats,
                                          TinyViTRunner)
        runner = TinyViTRunner(self.lib, self._hybrid("tiny_vit_b8"),
                               self.vit_weights, 8, fd=0,
                               weights_path=WEIGHTS / "tiny_vit.npz")
        stats = RunStats()
        try:
            runner.run(self.x[:8], stats=stats)
        finally:
            runner.close()
        self.assertGreater(stats.plan_total_us, 0.0)
        # Caller-observed dwell and C-side total describe the same interval.
        self.assertLess(abs(stats.plan_workspace_us - stats.plan_total_us),
                        0.10 * stats.plan_total_us)
        attributed = sum(float(getattr(stats, f"plan_{f}"))
                         for f in ENVELOPE_FIELDS)
        self.assertAlmostEqual(attributed, stats.plan_envelope_attributed_us,
                               places=3)
        residual = stats.plan_envelope_residual_us
        self.assertGreaterEqual(residual, -0.02 * stats.plan_total_us)
        self.assertLessEqual(residual, 0.05 * stats.plan_total_us,
                             "envelope does not tile the RKPlan dwell")
        for field in ENVELOPE_FIELDS + ENVELOPE_WITNESS_FIELDS:
            value = float(getattr(stats, f"plan_{field}"))
            self.assertGreaterEqual(value, 0.0, field)
            self.assertLessEqual(value, stats.plan_total_us,
                                 f"{field} exceeds the whole call")
        # Every attribution row is also present per plan.
        for row in stats.plans:
            for field in ENVELOPE_FIELDS:
                self.assertIn(field, row)
        data = stats.to_dict()
        self.assertIn("plan_envelope_residual_us", data)
        # The legacy keys keep their meaning and are not double-counted.
        self.assertAlmostEqual(data["plan_submit_wait_us"],
                               stats.plan_submit_wait_us, places=3)
        self.assertAlmostEqual(data["plan_pack_us"], stats.plan_pack_us, places=3)

    def test_static_batch_schedule_matches_compiled_tails(self):
        spec = importlib.util.spec_from_file_location(
            "evaluate_models", ROOT / "tools" / "evaluate_models.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        schedule = module.make_batch_schedule(10000, 32, tail_profiles=(8, 1))
        self.assertEqual(len(schedule), 314)
        self.assertEqual(sum(row["size"] for row in schedule), 10000)
        self.assertEqual([row["size"] for row in schedule[-2:]], [8, 8])
        self.assertTrue(all(row["profile"] == row["size"] for row in schedule))
        self.assertTrue(module.schedule_summary(schedule)["no_dynamic_shapes"])

    def test_hybrid_manifest_is_explicit_versioned_and_reuses_plans(self):
        from rknpu_backend.hybrid import (HOST_ACTIONS, HostNode, HybridRunner,
                                          NpuNode, load_hybrid_program)
        program = load_hybrid_program(self._hybrid("tiny_vit_b8"))
        self.assertEqual(program.execution_policy["mode"], "explicit_hybrid")
        self.assertEqual(program.execution_policy["npu_fallback"], "forbidden")
        self.assertEqual(program.execution_policy["host_fallback"], "forbidden")
        self.assertEqual(len(program.tensors), 26)
        node_ids = [node.id for node in program.nodes]
        self.assertEqual(node_ids, [
            "patchify", "patch_pad", "patch", "position_add", "rmsnorm1",
            "qkv", "split_heads", "qk", "softmax", "pv", "merge_heads",
            "attn_out", "residual_attention", "rmsnorm2", "gate_up", "swiglu",
            "down", "residual_ffn", "reshape_tokens", "mean_pool", "classifier",
            "export_logits"])
        host_ids = [node.id for node in program.nodes
                    if isinstance(node, HostNode)]
        self.assertIn("patchify", host_ids)
        self.assertIn("position_add", host_ids)
        self.assertIn("split_heads", host_ids)
        self.assertIn("softmax", host_ids)
        self.assertIn("merge_heads", host_ids)
        self.assertIn("residual_attention", host_ids)
        self.assertIn("swiglu", host_ids)
        self.assertIn("residual_ffn", host_ids)
        for node in program.nodes:
            if isinstance(node, HostNode):
                action = HOST_ACTIONS.get(node.action.action_id)
                self.assertIsNotNone(action)
                self.assertEqual(action.version, node.action.version)
        npu = [node for node in program.nodes if isinstance(node, NpuNode)]
        self.assertEqual(len(npu), 8)
        # QK/PV share one immutable per-repeat plan, plus one pre-expanded
        # static-repeat group plan.  The FFN gate/up branches are one packed
        # N=2H plan, and no capture-only probe enters the production manifest.
        self.assertEqual(len(program.plans), 8)
        gate_up = next(n for n in npu if n.id == "gate_up")
        self.assertEqual(sorted(gate_up.inputs), ["A"])
        self.assertEqual(gate_up.outputs["Y"].tensor, "gate_up")
        self.assertFalse(any(n.capture_only for n in npu))
        for node_id in ("qk", "pv"):
            node = next(n for n in program.nodes if n.id == node_id)
            self.assertEqual(node.repeat_submit, "shared_group")
            self.assertEqual(node.group_plan, "attn_matmul_group")
            group = program.plans[node.group_plan].plan
            self.assertEqual(group.accounting.submit_count, 1)
            self.assertEqual(group.accounting.task_count, 32)
            self.assertEqual(len(group.bindings), 96)
        for node in program.nodes:
            if getattr(node, "repeat_submit", None) and node.id not in ("qk", "pv"):
                self.assertEqual(node.repeat_submit, "per_repeat_submits")
        qk = next(node for node in npu if node.id == "qk")
        pv = next(node for node in npu if node.id == "pv")
        self.assertEqual(qk.plan, pv.plan)
        self.assertEqual([(r.name, r.extent) for r in qk.repeat],
                         [("bi", 8), ("hi", 4)])
        self.assertTrue(program.plans[qk.plan].dynamic_operand == "B")
        for spec in program.tensors.values():
            self.assertEqual(len(spec.shape), len(spec.axes))
        with HybridRunner(self.lib, program, self.vit_weights,
                          weights_path=WEIGHTS / "tiny_vit.npz", fd=0) as runner:
            self.assertEqual(len(runner.calls), 8)
            stats = __import__("rknpu_backend.models", fromlist=["RunStats"]).RunStats()
            result = runner.run(self.x[:8], stats=stats, capture=True)
            self.assertEqual(result.outputs["logits"].shape, (8, 10))
            self.assertTrue(np.isfinite(result.outputs["logits"]).all())
            actions = {row["name"] for row in stats.to_dict()["host_actions"]}
            for name in ("patchify", "patch_pad", "position_add", "rmsnorm1",
                         "split_heads", "softmax", "merge_heads",
                         "residual_attention", "rmsnorm2", "swiglu",
                         "residual_ffn", "reshape_tokens", "mean_pool",
                         "export_logits"):
                self.assertIn(name, actions)
            plans = {row["name"]: row for row in stats.to_dict()["plans"]}
            # QK/PV stay per-(batch,head) work: 32 repeats each, and the device
            # submit count is unchanged.  What changed is that the static repeat
            # set now uses one native group entry instead of 32 calls.
            # QK/PV now run as one pre-expanded plan with one chained submit:
            # 32 tasks in a single native call, so the node's task count is
            # unchanged while its submit count drops from 32 to 1.
            for name in ("qk", "pv"):
                self.assertEqual(plans[name]["calls"], 1)
                self.assertEqual(plans[name]["task_count"], 32)
                self.assertEqual(plans[name]["submit_count"], 1)
            self.assertEqual(stats.task_count, 70)
            # 2 chained QK/PV submits + patch/qkv/attn_out/classifier +
            # packed gate/up + down.
            self.assertEqual(stats.submit_count, 8)
            self.assertIn("gate_up", plans)
            self.assertIn("down", plans)
            self.assertEqual(plans["gate_up"]["task_count"], 1)
            self.assertEqual(plans["gate_up"]["submit_count"], 1)

    def test_packed_ffn_is_bitwise_host_equivalent_and_device_ffn_is_opt_in(self):
        """The default FFN packs gate/up into one N=2H GEMM and narrows SwiGLU
        once on Host, exactly like the old control.  The fully device-resident
        FFN remains a separate, explicitly calibrated numeric path."""
        from rknpu_backend.hybrid import HybridRunner, load_hybrid_program
        from rknpu_backend.models import RunStats

        packed_plan = self._plan("tiny_vit_b8_gate_up")
        self.assertEqual([a.template_id for a in packed_plan.actions
                          if a.kind == "npu_task"], ["co.mm.f16.v1"])
        self.assertEqual(packed_plan.accounting.task_count, 1)
        self.assertEqual(packed_plan.accounting.submit_count, 1)
        self.assertEqual(sum(a.kind == "download"
                             for a in packed_plan.actions), 1)

        results = {}
        traces = {}
        for key in ("tiny_vit_b8", "tiny_vit_b8_hostffn",
                    "tiny_vit_b8_deviceffn"):
            program = load_hybrid_program(self._hybrid(key))
            with HybridRunner(self.lib, program, self.vit_weights,
                              weights_path=WEIGHTS / "tiny_vit.npz", fd=0) as runner:
                stats = RunStats()
                result = runner.run(self.x[:8], stats=stats, capture=True)
                results[key] = result.outputs["logits"]
                traces[key] = result.trace
                if key == "tiny_vit_b8":
                    self.assertEqual((stats.task_count, stats.submit_count),
                                     (70, 8))
                elif key == "tiny_vit_b8_hostffn":
                    self.assertEqual((stats.task_count, stats.submit_count),
                                     (71, 9))
                else:
                    self.assertEqual((stats.task_count, stats.submit_count),
                                     (81, 17))
        # The packed path preserves the exact host-SwiGLU output and every
        # frozen trace tensor bit for bit.
        for key in ("tiny_vit_b8", "tiny_vit_b8_hostffn"):
            for name in ("gate", "up", "act", "ffn", "h2"):
                np.testing.assert_array_equal(traces[key][name],
                                              traces["tiny_vit_b8_hostffn"][name])
        np.testing.assert_array_equal(results["tiny_vit_b8"],
                                      results["tiny_vit_b8_hostffn"])
        # The fully device-resident FFN is a different numeric policy.
        self.assertFalse(np.array_equal(results["tiny_vit_b8_deviceffn"],
                                        results["tiny_vit_b8_hostffn"]))

        device_manifest = (PLANS / "tiny_vit_b8_deviceffn.rkhybrid.json").read_text()
        self.assertIn('"capture_only": true', device_manifest)

        # The capture-only plan must not change the production output: it is a
        # diagnostic sibling, not an extra step in the measured graph.
        device_program = load_hybrid_program(self._hybrid("tiny_vit_b8_deviceffn"))
        with HybridRunner(self.lib, device_program, self.vit_weights,
                          weights_path=WEIGHTS / "tiny_vit.npz", fd=0) as runner:
            plain = runner.run(self.x[:8])
        with HybridRunner(self.lib, device_program, self.vit_weights,
                          weights_path=WEIGHTS / "tiny_vit.npz", fd=0) as runner:
            captured = runner.run(self.x[:8], capture=True)
        np.testing.assert_array_equal(plain.outputs["logits"],
                                      captured.outputs["logits"])
        self.assertNotIn("gate", plain.trace)
        self.assertIn("gate", captured.trace)

    def test_shared_group_upload_coalescing_and_failure_atomicity(self):
        """One prep/fini per shared BO must be bitwise equivalent to per-slice
        sync, and an injected provider failure must not submit consumers."""
        from rknpu_backend.hybrid import HybridRunner, load_hybrid_program
        from rknpu_backend.models import RunStats
        from rknpu_backend.errors import BackendError

        program = load_hybrid_program(self._hybrid("tiny_vit_b8"))
        old = os.environ.get("RK_PLAN_UPLOAD_COALESCE")
        outputs = {}
        stats = {}
        try:
            for mode in ("1", "0"):
                os.environ["RK_PLAN_UPLOAD_COALESCE"] = mode
                with HybridRunner(self.lib, program, self.vit_weights,
                                  weights_path=WEIGHTS / "tiny_vit.npz", fd=0) as runner:
                    first = RunStats()
                    outputs[mode] = runner.run(self.x[:8], stats=first).outputs["logits"]
                    stats[mode] = first
                    second = runner.run(self.x[8:16]).outputs["logits"]
                if mode == "0":
                    outputs["changed"] = second
            np.testing.assert_array_equal(outputs["1"], outputs["0"])
            self.assertLess(stats["1"].plan_pack_prep_calls,
                            stats["0"].plan_pack_prep_calls)
            self.assertEqual(stats["1"].plan_pack_fini_calls,
                             stats["1"].plan_pack_prep_calls)
            self.assertEqual(stats["1"].upload_calls,
                             stats["0"].upload_calls)
            self.assertEqual(stats["1"].upload_bytes,
                             stats["0"].upload_bytes)

            # Changed inputs through the coalesced path still re-pack every slice.
            os.environ["RK_PLAN_UPLOAD_COALESCE"] = "1"
            with HybridRunner(self.lib, program, self.vit_weights,
                              weights_path=WEIGHTS / "tiny_vit.npz", fd=0) as runner:
                changed = runner.run(self.x[8:16]).outputs["logits"]
            np.testing.assert_array_equal(changed, outputs["changed"])

            # Provider failure in the shared-group upload itself: isolate the
            # qk group plan so there is no preceding unrelated submit.
            from rknpu_backend import load_plan
            from rknpu_backend.models import PlanCall
            group_plan = load_plan(str(PLANS / "tiny_vit_b8_qk_group.rkplan.json"))
            with PlanCall(self.lib, group_plan, {}, fd=0) as call:
                in_arrays = [np.zeros(tuple(b.shape), dtype=np.float16)
                             for b in group_plan.bindings if b.role == "input"]
                out_arrays = [np.zeros(tuple(b.shape), dtype=np.float16)
                              for b in group_plan.bindings if b.role == "output"]
                before = self.lib.cdll.rk_mock_execs()
                prep = self.lib.cdll.rk_mock_prep_calls()
                self.lib.cdll.rk_mock_fail_prep_at(prep + 1)
                with self.assertRaises(BackendError):
                    call.run_into(in_arrays, out_arrays)
                self.assertEqual(self.lib.cdll.rk_mock_execs(), before)
                self.lib.cdll.rk_mock_fail_prep_at(0)
                _report, _elapsed = call.run_into(in_arrays, out_arrays)

                before = self.lib.cdll.rk_mock_execs()
                fini = self.lib.cdll.rk_mock_fini_calls()
                self.lib.cdll.rk_mock_fail_fini_at(fini + 1)
                with self.assertRaises(BackendError):
                    call.run_into(in_arrays, out_arrays)
                self.assertEqual(self.lib.cdll.rk_mock_execs(), before)
                self.lib.cdll.rk_mock_fail_fini_at(0)
                _report, _elapsed = call.run_into(in_arrays, out_arrays)
        finally:
            if old is None:
                os.environ.pop("RK_PLAN_UPLOAD_COALESCE", None)
            else:
                os.environ["RK_PLAN_UPLOAD_COALESCE"] = old

    def test_fused_residual_rmsnorm_is_bitwise_control(self):
        """Fusing residual add + RMSNorm removes a Host action boundary while
        preserving the FP16 residual narrowing point and every downstream value."""
        from rknpu_backend.hybrid import HybridRunner, load_hybrid_program
        from rknpu_backend.models import RunStats

        outputs = {}
        traces = {}
        for key in ("tiny_vit_b8", "tiny_vit_b8_rmsnorm"):
            program = load_hybrid_program(self._hybrid(key))
            with HybridRunner(self.lib, program, self.vit_weights,
                              weights_path=WEIGHTS / "tiny_vit.npz", fd=0) as runner:
                stats = RunStats()
                result = runner.run(self.x[:8], stats=stats, capture=True)
                outputs[key] = result.outputs["logits"]
                traces[key] = result.trace
                self.assertEqual((stats.task_count, stats.submit_count), (70, 8))
                actions = [row["name"] for row in stats.host_actions]
                if key == "tiny_vit_b8":
                    self.assertIn("residual_attention", actions)
                    self.assertIn("rmsnorm2", actions)
                else:
                    self.assertIn("residual_rmsnorm2", actions)
                    self.assertNotIn("residual_attention", actions)
                    self.assertNotIn("rmsnorm2", actions)
        np.testing.assert_array_equal(outputs["tiny_vit_b8"],
                                      outputs["tiny_vit_b8_rmsnorm"])
        for name in ("h1", "norm2", "gate", "up", "act", "ffn", "h2", "logits"):
            np.testing.assert_array_equal(traces["tiny_vit_b8"][name],
                                          traces["tiny_vit_b8_rmsnorm"][name])

    def test_capture_only_outputs_cannot_enter_ordinary_execution(self):
        """A capture-only producer is diagnostic-only at load time, not merely
        skipped at run time. Ordinary consumers and model outputs must fail
        closed even though the manifest parses structurally."""
        from rknpu_backend.hybrid import load_hybrid_program
        from rknpu_backend.errors import BackendError
        import tempfile

        source = self._hybrid("tiny_vit_b8_deviceffn")
        base = json.loads(source.read_text())

        consumer = json.loads(json.dumps(base))
        reshape = next(n for n in consumer["nodes"]
                       if n["id"] == "reshape_tokens")
        reshape["inputs"]["x"]["tensor"] = "act"
        reshape["depends_on"] = ["ffn_trace"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_consumer.rkhybrid.json"
            path.write_text(json.dumps(consumer))
            with self.assertRaises(BackendError) as ctx:
                load_hybrid_program(path, plans_dir=PLANS)
            self.assertIn("capture-only", ctx.exception.diagnostics[0].message)

        exposed = json.loads(json.dumps(base))
        exposed["outputs"]["leak"] = "gate"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_output.rkhybrid.json"
            path.write_text(json.dumps(exposed))
            with self.assertRaises(BackendError) as ctx:
                load_hybrid_program(path, plans_dir=PLANS)
            self.assertIn("capture-only", ctx.exception.diagnostics[0].message)

    def test_hybrid_manifest_rejects_unknown_action(self):
        from rknpu_backend.hybrid import load_hybrid_program
        from rknpu_backend.errors import BackendError
        import tempfile
        data = json.loads(self._hybrid("tiny_vit_b1").read_text())
        for node in data["nodes"]:
            if node["kind"] == "host_action":
                node["action"] = "transformer.run_everything.v1"
                break
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.rkhybrid.json"
            path.write_text(json.dumps(data))
            with self.assertRaises(BackendError) as ctx:
                load_hybrid_program(path, plans_dir=PLANS)
            self.assertIn("unregistered Host action", ctx.exception.diagnostics[0].message)

    def test_plan_call_creation_failure_cleans_c_plan(self):
        from rknpu_backend.models import PlanCall
        plan = self._plan("tiny_vit_b1_qk")  # no constants
        class Handle:
            def __init__(self):
                self.destroyed = 0
            def destroy(self):
                self.destroyed += 1
        class FakeLib:
            def __init__(self):
                self.handle = Handle()
                self.plan_creates = 0
                self.workspace_creates = 0
            def create_plan(self, _plan):
                self.plan_creates += 1
                return self.handle
            def create_workspace(self, _cplan, _bindings, fd=0):
                self.workspace_creates += 1
                raise RuntimeError("injected workspace failure")
        fake = FakeLib()
        with self.assertRaisesRegex(RuntimeError, "injected workspace failure"):
            PlanCall(fake, plan, {}, fd=0)
        self.assertEqual(fake.plan_creates, 1)
        self.assertEqual(fake.workspace_creates, 1)
        self.assertEqual(fake.handle.destroyed, 1)

    def test_plan_call_validates_constants_before_c_plan(self):
        from rknpu_backend.models import PlanCall
        plan = self._plan("tiny_vit_b1_patch")  # has B constant
        class FakeLib:
            plan_creates = 0
            def create_plan(self, _plan):
                self.plan_creates += 1
                raise AssertionError("must not create C plan")
        fake = FakeLib()
        with self.assertRaises(KeyError):
            PlanCall(fake, plan, {}, fd=0)
        self.assertEqual(fake.plan_creates, 0)

    def test_historical_test_envelope_and_mock_layer_tolerances(self):
        import importlib.util
        from rknpu_backend.hybrid import HybridRunner, load_hybrid_program
        from rknpu_backend.models import PlanCall
        from rknpu_backend.numeric import SAMPLE_INDICES, relu_probe_constants
        spec_data = json.loads((WEIGHTS / "numeric_acceptance.json").read_text())
        self.assertEqual(spec_data["status"], "frozen")
        self.assertEqual(spec_data["calibration"]["status"], "frozen")
        self.assertEqual(spec_data["calibration"]["uses_classification_metric"],
                         False)
        self.assertEqual(set(spec_data["task_logit_tolerances"]),
                         {"mnist_mlp.b32", "mnist_mlp.b8", "mnist_mlp.b1",
                          "tiny_vit.b8", "tiny_vit.b1"})
        for key, tol in {**spec_data["layer_tolerances"],
                         **spec_data["task_logit_tolerances"]}.items():
            self.assertIsNotNone(tol, key)
            self.assertGreaterEqual(tol["atol"], 0.0, key)
            self.assertEqual(tol["rtol"], 0.0, key)
            self.assertIn("basis", tol)
            self.assertTrue(tol["observed_max_abs_by_source"])
        loader_spec = importlib.util.spec_from_file_location(
            "verify_model_results", ROOT / "tools" / "verify_model_results.py")
        verifier = importlib.util.module_from_spec(loader_spec)
        loader_spec.loader.exec_module(verifier)
        golden = np.load(WEIGHTS / "numeric_golden.npz", allow_pickle=False)
        sample = self.x[list(SAMPLE_INDICES)]
        program = load_hybrid_program(self._hybrid("tiny_vit_b8"))
        with HybridRunner(self.lib, program, self.vit_weights,
                          weights_path=WEIGHTS / "tiny_vit.npz", fd=0) as runner:
            result = runner.run(sample, capture=True)
        for key, value in result.trace.items():
            golden_key = f"tiny_vit.{key}"
            ok, _err = verifier.logits_within_frozen_tolerance(
                value, golden[golden_key],
                spec_data["layer_tolerances"][golden_key])
            self.assertTrue(ok, golden_key)
        probe_plan = self._plan("mnist_relu_probe")
        with PlanCall(self.lib, probe_plan,
                      relu_probe_constants(probe_plan, self.mlp_weights),
                      fd=0) as probe:
            outputs, _report, _elapsed = probe.run([])
        ok, _err = verifier.logits_within_frozen_tolerance(
            outputs[0], golden["relu_probe.output"],
            spec_data["layer_tolerances"]["relu_probe.output"])
        self.assertTrue(ok)

    def test_model_runner_rejects_non_manifest_bypass(self):
        from rknpu_backend.models import MLPRunner
        from rknpu_backend.errors import BackendError
        with self.assertRaises(BackendError):
            MLPRunner(self.lib, self._plan("mnist_mlp_b1"),
                      self.mlp_weights, 1, fd=0)

    def test_train_calibration_and_test_acceptance_samples_are_disjoint(self):
        spec = json.loads((WEIGHTS / "numeric_acceptance_train.json").read_text())
        self.assertEqual(spec["calibration"]["split"], "train")
        self.assertEqual(spec["acceptance"]["split"], "test")
        self.assertFalse(spec["calibration"]["labels_read"])
        self.assertFalse(spec["acceptance"]["labels_read"])
        cal = set(spec["calibration"]["sample_sha256"])
        acc = set(spec["acceptance"]["sample_sha256"])
        self.assertTrue(cal.isdisjoint(acc))
        self.assertNotEqual(spec["calibration"]["source_sha256"],
                            spec["acceptance"]["source_sha256"])

    def test_hidden_trace_rejects_while_final_top1_is_unchanged(self):
        from rknpu_backend.numeric_policy import within_tolerance
        spec = json.loads((WEIGHTS / "numeric_acceptance_train.json").read_text())
        expected = np.load(WEIGHTS / "numeric_acceptance_test.npz",
                           allow_pickle=False)
        hidden = expected["mlp.h1_pre_relu"]
        tampered_hidden = hidden.copy()
        tolerance = spec["layer_tolerances"]["mlp.h1_pre_relu"]
        tampered_hidden[0, 0] = np.float16(
            float(tampered_hidden[0, 0]) + 2.0 * float(tolerance["atol"]))
        logits_before = expected["profile.mnist_mlp.b32.logits"].copy()
        logits_after = logits_before.copy()  # hidden tamper must not be inferred
        self.assertTrue(np.array_equal(logits_before.argmax(axis=1),
                                       logits_after.argmax(axis=1)))
        hidden_ok, hidden_error = within_tolerance(
            tampered_hidden, hidden, tolerance)
        self.assertFalse(hidden_ok)
        self.assertGreater(hidden_error, float(tolerance["atol"]))
        # A top-1-only verifier would pass because the final logits are unchanged;
        # the frozen layer gate must still reject the tampered hidden trace.
        logits_ok, _ = within_tolerance(logits_after, logits_before, {
            "atol": spec["task_logit_tolerances"]["mnist_mlp.b32"]["atol"],
            "rtol": 0.0})
        self.assertTrue(logits_ok)

    def test_mlp_hidden_diagnostic_plans_match_declared_layers(self):
        from rknpu_backend.numeric_observation import _run_mlp_diagnostics
        from rknpu_backend.numeric_policy import within_tolerance
        spec = json.loads((WEIGHTS / "numeric_acceptance_train.json").read_text())
        expected = np.load(WEIGHTS / "numeric_calibration_train.npz",
                           allow_pickle=False)
        images = expected["images"][:8]
        manifest = json.loads((PLANS / "manifest.json").read_text())
        got = _run_mlp_diagnostics(
            self.lib, manifest, PLANS, self.mlp_weights, images, 0)
        mapping = {
            "h1_pre_relu": "mlp.h1_pre_relu",
            "r1_relu": "mlp.r1_relu",
            "h2_pre_relu": "mlp.h2_pre_relu",
            "r2_relu": "mlp.r2_relu",
            "final_preactivation": "mlp.final_preactivation",
        }
        for observed_key, expected_key in mapping.items():
            ok, error = within_tolerance(
                got[observed_key], expected[expected_key][:8],
                spec["layer_tolerances"][expected_key])
            self.assertTrue(ok, (observed_key, error))
        templates = {
            key: [a.template_id for a in self._plan(key).actions
                  if a.kind == "npu_task"]
            for key in (
                "mnist_mlp_diag_l1_bias", "mnist_mlp_diag_l1_relu",
                "mnist_mlp_diag_l2_bias", "mnist_mlp_diag_l2_relu",
                "mnist_mlp_diag_l3_bias")
        }
        self.assertNotIn("co.ew_max.f16.v1", templates["mnist_mlp_diag_l1_bias"])
        self.assertIn("co.ew_max.f16.v1", templates["mnist_mlp_diag_l1_relu"])
        self.assertNotIn("co.ew_max.f16.v1", templates["mnist_mlp_diag_l2_bias"])
        self.assertIn("co.ew_max.f16.v1", templates["mnist_mlp_diag_l2_relu"])
        self.assertEqual(sum(a.kind == "download"
                             for a in self._plan("mnist_mlp_b8").actions), 1)

    def test_train_calibrated_numeric_evidence_replays(self):
        import importlib.util
        loader_spec = importlib.util.spec_from_file_location(
            "verify_numeric_acceptance",
            ROOT / "tools" / "verify_numeric_acceptance.py")
        verifier = importlib.util.module_from_spec(loader_spec)
        loader_spec.loader.exec_module(verifier)
        for spec_name, result_name in (
                ("numeric_acceptance_packed.json",
                 "20261026T093427Z-numeric-replay"),
                ("numeric_acceptance_packed.json",
                 "20261026T091612Z-numeric-replay"),
                ("numeric_acceptance_device_ffn.json",
                 "20261026T091954Z-numeric")):
            evidence = ROOT / "results" / result_name
            summary = verifier.verify(
                WEIGHTS / spec_name,
                evidence / "calibration" / "observation.npz",
                evidence / "acceptance" / "observation.npz")
            self.assertEqual(summary["status"], "PASS", summary["problems"])
            self.assertEqual(summary["problems"], [])
            self.assertTrue(
                summary["sample_disjoint"]["train_test_splits_distinct"])
            self.assertTrue(summary["sample_disjoint"]["pixel_hash_sets_disjoint"])
            self.assertEqual(len(summary["calibration"]["layers"]), 33)
            self.assertEqual(len(summary["acceptance"]["layers"]), 33)
            self.assertEqual(len(summary["acceptance"]["profiles"]), 5)
            for key in ("mlp.h1_pre_relu", "mlp.r1_relu", "mlp.h2_pre_relu",
                        "mlp.r2_relu", "mlp.final_preactivation"):
                self.assertIn(key, summary["acceptance"]["layers"])
                self.assertTrue(summary["acceptance"]["layers"][key]["ok"])

    def test_verifier_rejects_same_top1_amplified_logits(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "verify_model_results", ROOT / "tools" / "verify_model_results.py")
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
        expected = np.asarray([[1.0, 2.0, 3.0],
                               [4.0, 1.0, 0.0]], dtype=np.float32)
        amplified = expected * np.float32(1.5)
        np.testing.assert_array_equal(amplified.argmax(1), expected.argmax(1))
        ok, error = verifier.logits_within_frozen_tolerance(
            amplified, expected, {"atol": 0.01, "rtol": 0.0})
        self.assertFalse(ok)
        self.assertGreater(error, 0.01)

    def test_cleanup_failure_does_not_replace_original_exception(self):
        from rknpu_backend.hybrid import HybridRunner, load_hybrid_program
        program = load_hybrid_program(self._hybrid("tiny_vit_b1"))
        runner = HybridRunner(self.lib, program, self.vit_weights,
                              weights_path=WEIGHTS / "tiny_vit.npz", fd=0)
        call = next(iter(runner.calls.values()))
        original_close = call.close
        def fail_close():
            raise RuntimeError("injected cleanup failure")
        call.close = fail_close
        try:
            with self.assertRaisesRegex(RuntimeError, "original run failure"):
                with runner:
                    raise RuntimeError("original run failure")
        finally:
            call.close = original_close
            try:
                runner.close()
            except Exception:
                pass

    def test_relu_probe_uses_real_exact_ew_max_plan(self):
        from rknpu_backend.models import PlanCall
        from rknpu_backend.numeric import (relu_probe_constants,
                                           relu_probe_vectors)
        plan = self._plan("mnist_relu_probe")
        templates = [a.template_id for a in plan.actions
                     if a.kind == "npu_task"]
        self.assertEqual(templates, ["co.ew_add.f16.v1",
                                     "co.ew_max.f16.v1"])
        constants = relu_probe_constants(plan, self.mlp_weights)
        _x, _b, _pre, expected = relu_probe_vectors(self.mlp_weights)
        with PlanCall(self.lib, plan, constants, fd=0) as call:
            outputs, _report, _elapsed = call.run([])
        np.testing.assert_array_equal(outputs[0], expected)

    def test_deterministic_layer_golden_is_reproducible(self):
        import hashlib
        from rknpu_backend.numeric import golden_arrays
        golden_path = WEIGHTS / "numeric_golden.npz"
        spec = json.loads((WEIGHTS / "numeric_acceptance.json").read_text())
        self.assertEqual(spec["golden_sha256"],
                         hashlib.sha256(golden_path.read_bytes()).hexdigest())
        golden = np.load(golden_path, allow_pickle=False)
        rebuilt = golden_arrays(self.mlp_weights, self.vit_weights, self.x)
        self.assertEqual(set(golden.files), set(rebuilt))
        for key in golden.files:
            np.testing.assert_array_equal(golden[key], rebuilt[key])
        contract_keys = set()
        for item in spec["trace_contract"].values():
            contract_keys.update(item.get("keys", []))
        self.assertTrue(contract_keys <= set(golden.files))
        self.assertEqual(spec["sample_policy"]["indices"],
                         golden["sample_indices"].tolist())

    def test_model_failure_does_not_publish(self):
        import subprocess
        env = dict(os.environ)
        env["RKNPU_PLAN_MOCK_LIB"] = MOCK_LIB
        env["PYTHONPATH"] = (str(ROOT / "python") +
                             (os.pathsep + env["PYTHONPATH"]
                              if env.get("PYTHONPATH") else ""))
        proc = subprocess.run(
            [sys.executable, str(ROOT / "tests" / "fixtures" /
                                  "model_failure_worker.py")],
            cwd=str(ROOT), env=env, text=True, capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + "\n" + proc.stderr)
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertFalse(result["published"])
        self.assertNotEqual(result["run_code"], "NO_ERROR")
        self.assertNotEqual(result["close_code"], "NO_ERROR")
        self.assertEqual(result["execs_after"], result["execs_before"] + 1)

    def test_transformer_source_dynamic_b_and_hybrid_runner(self):
        from models.tilelang_matmul import make_matmul_prim
        from rknpu_backend import CompileOptions, compile_tilelang
        from rknpu_backend.models import TinyViTRunner
        from rknpu_backend.model_reference import tiny_vit_logits_declared
        dyn = compile_tilelang(
            make_matmul_prim(16, 16, 16), profile=self.profile,
            options=CompileOptions(numeric_mode="rk_f16_tiled",
                                   allow_approx_lut=False),
            b_is_input=True).plan
        self.assertEqual([b.role for b in dyn.bindings if b.name in {"A", "B"}],
                         ["input", "input"])
        names = ("patch", "qkv", "qk", "pv", "attn_out", "gate", "up",
                 "down", "classifier")
        plan_objects = [self._plan(f"tiny_vit_b1_{n}") for n in names]
        runner = TinyViTRunner(self.lib, self._hybrid("tiny_vit_b1"),
                               self.vit_weights, 1, fd=0,
                               weights_path=WEIGHTS / "tiny_vit.npz")
        try:
            raw = self.x[:2]
            got = np.concatenate([runner.run(raw[i:i + 1]) for i in range(2)])
            ref = tiny_vit_logits_declared(self.vit_weights, raw)
            np.testing.assert_allclose(got, ref, atol=0.08, rtol=0.02)
            self.assertEqual((got.argmax(1) == ref.argmax(1)).all(), True)
            # Every matrix plan is a real one-task RKPlan; the graph executor
            # never bypasses it with a co_execute/graph fallback.
            for plan in plan_objects:
                self.assertTrue(all(a.template_id == "co.mm.f16.v1"
                                    for a in plan.actions
                                    if a.kind == "npu_task"))
        finally:
            runner.close()


if __name__ == "__main__":
    unittest.main()
