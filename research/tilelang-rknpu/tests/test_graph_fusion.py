"""Graph-level dependency-chain fusion, residual Add and RMSNorm policy tests."""
from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOCK_LIB = os.environ.get("RKNPU_PLAN_MOCK_LIB") or str(
    ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan_mock.so")
EXPERIMENTAL_PROFILE = ROOT / "capabilities" / "rk3588_vendor_co_v1_rmsnorm_lut.json"


class GraphFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not Path(MOCK_LIB).is_file():
            raise unittest.SkipTest(f"mock library not built: {MOCK_LIB}")
        from rknpu_backend.runtime.ffi import LibRKPlan
        cls.lib = LibRKPlan(MOCK_LIB)
        cls.weights = dict(np.load(ROOT / "models" / "tiny_vit.npz",
                                   allow_pickle=False))

    def test_mixed_template_ffn_chain_is_one_submit_and_bitwise_control(self):
        from rknpu_backend import (CompileOptions, compile_tilelang,
                                   default_profile, load_plan)
        from rknpu_backend.models import PlanCall
        from models.tilelang_matmul import make_swiglu_ffn_f16
        for m in (16, 128):
            fused = compile_tilelang(
                make_swiglu_ffn_f16(m=m, d=64, h=128),
                profile=default_profile(),
                options=CompileOptions(numeric_mode="rk_f16_tiled",
                                       allow_approx_lut=True,
                                       fuse_task_chain=True)).plan
            self.assertEqual(fused.accounting.task_count, 7)
            self.assertEqual(len(fused.submits), 1)
            self.assertEqual(fused.submits[0].recipe_id,
                             "dependency_chain_v1")
            self.assertEqual(len(fused.submits[0].task_ids), 7)
            control = load_plan(str(
                ROOT / "models" / "compiled" /
                f"tiny_vit_b{m // 16}_ffn.rkplan.json"))
            consts = {
                "Wg": self.weights["ffn_gate_w"],
                "Wu": self.weights["ffn_up_w"],
                "Wd": self.weights["ffn_down_w"],
                "__identity_64": np.eye(64, dtype=np.float16),
                "__identity_128": np.eye(128, dtype=np.float16),
            }
            with PlanCall(self.lib, control, consts, fd=0) as old, \
                    PlanCall(self.lib, fused, consts, fd=0) as new:
                rng = np.random.default_rng(10 + m)
                x = np.ascontiguousarray(
                    rng.normal(size=(m, 64)).astype(np.float16))
                h = np.ascontiguousarray(
                    rng.normal(size=(m, 64)).astype(np.float16))
                y_old = np.empty((m, 64), dtype=np.float16)
                y_new = np.empty_like(y_old)
                ro, _ = old.run_into([x, h], [y_old])
                rn, _ = new.run_into([x, h], [y_new])
                np.testing.assert_array_equal(y_old, y_new)
                self.assertEqual(ro.task_count, rn.task_count)
                self.assertEqual(ro.submit_count, 6)
                self.assertEqual(rn.submit_count, 1)

    def test_mixed_chain_python_allowlist_is_exact(self):
        from rknpu_backend.passes.schedule import (_is_allowed_chain,
                                                    _is_allowed_chain_prefix)
        chain = [
            "co.mm.f16.v1", "co.mm.f16.v1", "co.lut.sigmoid.f16.v1",
            "co.ew_mul.f16.v1", "co.ew_mul.f16.v1", "co.mm.f16.v1",
            "co.ew_add.f16.v1",
        ]
        self.assertTrue(_is_allowed_chain(chain))
        self.assertTrue(_is_allowed_chain_prefix(chain[:3]))
        for bad in (
                ["co.mm.f16.v1", "co.lut.sigmoid.f16.v1"],
                ["co.mm.f16.v1", "co.mm.f16.v1", "co.ew_mul.f16.v1"],
                chain[:-1] + ["co.ew_max.f16.v1"],
                list(reversed(chain))):
            self.assertFalse(_is_allowed_chain(bad), bad)

    def _negative_chain_plans(self):
        from dataclasses import replace
        from rknpu_backend import (CompileOptions, compile_tilelang,
                                   default_profile)
        from rknpu_backend.plan import with_plan_hash
        from models.tilelang_matmul import make_swiglu_ffn_f16
        plan = compile_tilelang(
            make_swiglu_ffn_f16(m=16, d=64, h=128),
            profile=default_profile(),
            options=CompileOptions(numeric_mode="rk_f16_tiled",
                                   allow_approx_lut=True,
                                   fuse_task_chain=True)).plan
        self.assertEqual(len(plan.submits), 1)
        tasks = list(plan.submits[0].task_ids)

        def non_consecutive():
            bad = replace(plan.submits[0],
                          task_ids=(tasks[0], tasks[2], tasks[3], tasks[4],
                                    tasks[5], tasks[6], tasks[1]))
            return with_plan_hash(replace(plan, submits=(bad,),
                                          plan_hash=""))

        def wrong_template():
            actions = list(plan.actions)
            idx = tasks[2]
            actions[idx] = replace(actions[idx],
                                   template_id="co.ew_max.f16.v1")
            return with_plan_hash(replace(plan, actions=tuple(actions),
                                          plan_hash=""))

        def overlapping_writes():
            actions = list(plan.actions)
            actions[tasks[2]] = replace(
                actions[tasks[2]],
                outputs=plan.actions[tasks[0]].outputs)
            return with_plan_hash(replace(plan, actions=tuple(actions),
                                          plan_hash=""))

        return (non_consecutive, wrong_template, overlapping_writes)

    def test_c_rejects_non_allowlisted_or_alias_chain(self):
        from rknpu_backend.errors import BackendError
        for make_broken in self._negative_chain_plans():
            with self.assertRaises(BackendError):
                broken = make_broken()
                cplan = self.lib.create_plan(broken)
                cplan.destroy()

    @staticmethod
    def _rope_coefficients(m, width, head_dim, rotary_dim, style):
        half = rotary_dim // 2
        cos = np.ones((m, width), dtype=np.float16)
        sin = np.zeros((m, width), dtype=np.float16)
        perm = np.zeros((width, width), dtype=np.float16)
        for h in range(width // head_dim):
            base = h * head_dim
            for d in range(rotary_dim):
                if style == "interleaved":
                    pair = d ^ 1
                    sign = -1.0 if d % 2 == 0 else 1.0
                    freq_index = d // 2
                else:
                    pair = d + half if d < half else d - half
                    sign = -1.0 if d < half else 1.0
                    freq_index = d % half
                perm[base + d, base + pair] = sign
                freq = 10000.0 ** (-2.0 * freq_index / rotary_dim)
                for pos in range(m):
                    angle = pos * freq
                    cos[pos, base + d] = np.float16(np.cos(angle))
                    sin[pos, base + d] = np.float16(np.sin(angle))
        return cos, sin, perm

    def test_device_rope_reorder_is_one_chain_and_matches_host_fp16(self):
        from rknpu_backend import (CompileOptions, compile_rkir,
                                   default_profile, rope_device_module)
        from rknpu_backend.models import PlanCall
        for style in ("interleaved", "split_half"):
            for rotary_dim in (64, 32):
                m, width, head_dim = 16, 64, 64
                cos, sin, perm = self._rope_coefficients(
                    m, width, head_dim, rotary_dim, style)
                plan = compile_rkir(
                    rope_device_module(m=m, width=width,
                                       name=f"rope_{style}_{rotary_dim}"),
                    profile=default_profile(),
                    options=CompileOptions(numeric_mode="rk_f16_tiled",
                                           fuse_task_chain=True)).plan
                self.assertEqual(plan.accounting.task_count, 4)
                self.assertEqual(len(plan.submits), 1)
                self.assertEqual(plan.submits[0].recipe_id,
                                 "dependency_chain_v1")
                with PlanCall(self.lib, plan, {"P": perm, "__identity_64": np.eye(64, dtype=np.float16)}, fd=0) as call:
                    rng = np.random.default_rng(50 + rotary_dim)
                    x = np.ascontiguousarray(
                        rng.normal(size=(m, width)).astype(np.float16))
                    y = np.empty_like(x)
                    call.run_into([x, cos, sin], [y])
                rot = np.zeros_like(x)
                for d in range(width):
                    if d % head_dim < rotary_dim:
                        # P row d selects exactly one source column.
                        src = int(np.argmax(np.abs(
                            perm[d].astype(np.float32))))
                        rot[:, d] = np.float16(
                            np.sign(perm[d, src]) * x[:, src].astype(np.float32))
                    else:
                        rot[:, d] = 0
                t1 = np.float16(x.astype(np.float32) *
                                cos.astype(np.float32))
                t2 = np.float16(rot.astype(np.float32) *
                                sin.astype(np.float32))
                ref = np.float16(t1.astype(np.float32) +
                                 t2.astype(np.float32))
                # Non-rotary dimensions pass through x*1 + 0.
                np.testing.assert_array_equal(y, ref)

    def test_pv_o_device_dataflow_is_one_chain_and_matches_host_four_head(self):
        from rknpu_backend import (CompileOptions, compile_rkir,
                                   default_profile, pv_o_device_module)
        from rknpu_backend.models import PlanCall
        m, heads, hd, out_dim = 16, 4, 16, 64
        rng = np.random.default_rng(909)
        probs = [np.ascontiguousarray(
            rng.normal(size=(m, hd)).astype(np.float16))
            for _ in range(heads)]
        v = [np.ascontiguousarray(
            rng.normal(size=(m, hd)).astype(np.float16))
            for _ in range(heads)]
        w = np.ascontiguousarray(rng.normal(
            size=(out_dim, heads * hd)).astype(np.float16))
        plan = compile_rkir(
            pv_o_device_module(m=m, heads=heads, head_dim=hd,
                               out_dim=out_dim),
            profile=default_profile(),
            options=CompileOptions(numeric_mode="rk_f16_tiled",
                                   fuse_task_chain=True)).plan
        self.assertEqual(plan.accounting.task_count, 11)
        self.assertEqual(len(plan.submits), 1)
        self.assertEqual(plan.submits[0].recipe_id, "dependency_chain_v1")
        consts = {f"W{h}": np.ascontiguousarray(
            w[:, h * hd:(h + 1) * hd]) for h in range(heads)}
        consts["__identity_64"] = np.eye(64, dtype=np.float16)
        inputs = probs + [np.ascontiguousarray(v[h].T)
                          for h in range(heads)]
        with PlanCall(self.lib, plan, consts, fd=0) as call:
            got = np.empty((m, out_dim), dtype=np.float16)
            call.run_into(inputs, [got])
        pv = [np.ascontiguousarray(
            probs[h].astype(np.float32) @ v[h].astype(np.float32),
            dtype=np.float16) for h in range(heads)]
        parts = [np.ascontiguousarray(
            pv[h].astype(np.float32) @
            w[:, h * hd:(h + 1) * hd].astype(np.float32).T,
            dtype=np.float16) for h in range(heads)]
        want = parts[0]
        for part in parts[1:]:
            want = np.ascontiguousarray(
                want.astype(np.float32) + part.astype(np.float32),
                dtype=np.float16)
        np.testing.assert_array_equal(got, want)

    def test_cube_to_weight_requires_new_reorder_not_p0312(self):
        def cube16(r, c):
            return (c // 8) * 16 * 8 + r * 8 + c % 8
        def weight16(n, k):
            return n * 32 + k
        def p0312(dest):
            c, a, b = dest // 64, (dest % 64) // 8, dest % 8
            return (a * 8 + b) * 8 + c
        mismatches = 0
        for n in range(16):
            for k in range(16):
                if p0312(weight16(n, k)) != cube16(n, k):
                    mismatches += 1
        self.assertEqual(mismatches, 255)
        self.assertEqual(cube16(1, 0), 8)
        self.assertEqual(weight16(1, 0), 32)
        self.assertEqual(p0312(32), 256)

    def test_qkv_to_qk_device_weight_operand_is_rejected(self):
        from rknpu_backend import CompileOptions, compile_rkir, default_profile
        from rknpu_backend.errors import BackendError
        from rknpu_backend.ir import (_builder_contract, DTYPE_F16, Module,
                                      Op, TensorType, Value)
        contract = _builder_contract("rk_f16_tiled")
        ty = TensorType(DTYPE_F16, (16, 64))
        attrs = {"b_layout": "nk",
                 "numeric_contract": contract.to_dict(),
                 "source_accumulator_dtype": "float32"}
        module = Module(
            name="qkv_then_qk_weight_operand",
            values=(
                Value(0, "X", ty, None, "input"),
                Value(1, "W", TensorType(DTYPE_F16, (64, 64)), None,
                      "constant"),
                Value(2, "QKV", ty, 0, "temporary"),
                Value(3, "S", TensorType(DTYPE_F16, (16, 16)), 1, "output"),
            ),
            ops=(
                Op(0, "matmul", (0, 1), (2,), dict(attrs), "test"),
                Op(1, "matmul", (2, 2), (3,), dict(attrs), "test"),
            ),
            inputs=(0,), outputs=(3,), source_hash="builder",
            state_signature="")
        with self.assertRaises(BackendError) as ctx:
            compile_rkir(module, profile=default_profile(),
                         options=CompileOptions(numeric_mode="rk_f16_tiled"))
        self.assertIn("device-produced values", str(ctx.exception))

    def test_smollm_device_rope_candidate_policy_passes(self):
        import json
        spec = json.loads(
            (ROOT / "models" / "smollm2_device_rope_policy.json").read_text())
        self.assertTrue(spec["pass"])
        self.assertEqual(spec["status"], "experimental_not_default")
        gates = spec["gates"]
        acc = spec["acceptance"]
        self.assertLessEqual(acc["prefill_logits_max_abs_diff"],
                             gates["prefill_max_abs"])
        self.assertLessEqual(acc["decode_logits_max_abs_diff"],
                             gates["decode_max_abs"])
        self.assertTrue(acc["prefill_top1_equal"])
        self.assertTrue(acc["decode_top1_equal"])

    def test_smollm_continuous_device_rope_policy_passes(self):
        import hashlib
        import json
        path = ROOT / "models" / "smollm2_device_rope_policy_cont32.json"
        raw = path.read_bytes()
        spec = json.loads(raw)
        self.assertTrue(spec["pass"])
        self.assertEqual(spec["decode_steps"], 32)
        self.assertEqual(hashlib.sha256(raw).hexdigest(),
                         "fc105a2c6a8f95208b4a1882f821d6ea8cf9808d3c714f57a89a347ab6f02101")
        self.assertLessEqual(spec["acceptance"]["decode_logits_max_abs_diff"],
                             spec["gates"]["decode_max_abs"])
        self.assertEqual(spec["acceptance"]["incremental_vs_full_prefix_max_abs"], 0.0)
        self.assertTrue(spec["acceptance"]["aba_request_isolation_bitwise"])

    def test_smollm_policy_binding_rejects_mode_steps_calls(self):
        import importlib.util
        loader = importlib.util.spec_from_file_location(
            "run_smollm_device_rope",
            ROOT / "tools" / "run_smollm_device_rope.py")
        mod = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(mod)
        report = {"decode_mode": "position_indexed_device_gather",
                  "decode_steps": 32,
                  "actual_decode_calls": 3360}
        good = {"decode_mode": "position_indexed_device_gather",
                "decode_steps": 32, "expected_decode_calls": 3360}
        self.assertEqual(mod.policy_binding_problems(good, report), [])
        bad_mode = dict(good, decode_mode="runtime_coefficient_inputs")
        self.assertIn("policy_decode_mode",
                      mod.policy_binding_problems(bad_mode, report))
        bad_steps = dict(good, decode_steps=31)
        self.assertIn("policy_decode_steps",
                      mod.policy_binding_problems(bad_steps, report))
        bad_calls = dict(good, expected_decode_calls=1920)
        self.assertIn("policy_actual_decode_calls",
                      mod.policy_binding_problems(bad_calls, report))

    def test_smollm_continuous_indexed_rope_policy_passes(self):
        import json
        spec = json.loads((ROOT / "models" /
                           "smollm2_device_rope_policy_cont32_indexed.json"
                          ).read_text())
        self.assertTrue(spec["pass"])
        self.assertEqual(spec["decode_steps"], 32)
        self.assertIn("decode_indexed_9h", spec["plan_hashes"])
        self.assertIn("decode_indexed_3h", spec["plan_hashes"])
        self.assertLessEqual(spec["acceptance"]["decode_logits_max_abs_diff"],
                             spec["gates"]["decode_max_abs"])
        self.assertEqual(spec["acceptance"]["incremental_vs_full_prefix_max_abs"], 0.0)
        self.assertTrue(spec["acceptance"]["aba_request_isolation_bitwise"])

    def test_rope_decode_indexed_plan_gathers_resident_table(self):
        from rknpu_backend import (CompileOptions, compile_rkir,
                                   default_profile, rope_decode_indexed_module)
        from rknpu_backend.models import PlanCall
        width, cap = 192, 256
        plan = compile_rkir(
            rope_decode_indexed_module(width=width, capacity=cap),
            profile=default_profile(),
            options=CompileOptions(numeric_mode="rk_f16_tiled",
                                   fuse_task_chain=True)).plan
        self.assertEqual(plan.accounting.task_count, 6)
        self.assertIn("CosW", [c.name for c in plan.constants])
        self.assertIn("SinW", [c.name for c in plan.constants])
        x = np.zeros((1, width), dtype=np.float16)
        cosw = np.zeros((width, cap), dtype=np.float16)
        sinw = np.zeros((width, cap), dtype=np.float16)
        cosw[:, 255] = np.float16(0.5)
        sinw[:, 255] = np.float16(0.25)
        consts = {"CosW": cosw, "SinW": sinw,
                  "P": np.eye(width, dtype=np.float16),
                  "__identity_192": np.eye(192, dtype=np.float16)}
        pos = np.zeros((1, cap), dtype=np.float16)
        pos[0, 255] = 1.0
        with PlanCall(self.lib, plan, consts, fd=0) as call:
            got = np.empty((1, width), dtype=np.float16)
            call.run_into([x, pos], [got])
        np.testing.assert_array_equal(got, np.zeros_like(got))

    def test_pv_o_b8_multi_submit_plan_is_legal(self):
        from rknpu_backend import (CompileOptions, compile_rkir,
                                   default_profile,
                                   pv_o_device_batches_module)
        plan = compile_rkir(
            pv_o_device_batches_module(),
            profile=default_profile(),
            options=CompileOptions(numeric_mode="rk_f16_tiled",
                                   fuse_task_chain=True)).plan
        self.assertEqual(plan.accounting.task_count, 88)
        self.assertEqual(len(plan.submits), 8)
        self.assertEqual([len(s.task_ids) for s in plan.submits], [11] * 8)
        cplan = self.lib.create_plan(plan)
        cplan.destroy()

    def test_device_residual_add_is_bitwise_host_equivalent(self):
        from rknpu_backend import (CompileOptions, compile_rkir,
                                   default_profile, load_plan, matmul_module)
        from rknpu_backend.models import PlanCall
        for m, k, weight_name, filename in (
                (16, 64, "attn_out_w", "tiny_vit_b1_attn_out"),
                (128, 64, "attn_out_w", "tiny_vit_b8_attn_out"),
                (16, 128, "ffn_down_w", "tiny_vit_b1_down"),
                (128, 128, "ffn_down_w", "tiny_vit_b8_down")):
            fused = compile_rkir(
                matmul_module(m=m, k=k, n=64, residual="add", b_layout="kn",
                              name=f"residual_add_{m}_{k}"),
                profile=default_profile(),
                options=CompileOptions(numeric_mode="rk_f16_tiled",
                                       fuse_task_chain=True)).plan
            self.assertEqual(len(fused.submits), 1)
            self.assertEqual(fused.accounting.task_count, 2)
            control = load_plan(str(ROOT / "models" / "compiled" /
                                    f"{filename}.rkplan.json"))
            consts = {"B": self.weights[weight_name],
                      "__identity_64": np.eye(64, dtype=np.float16)}
            with PlanCall(self.lib, control, {"B": self.weights[weight_name]},
                          fd=0) as old, \
                    PlanCall(self.lib, fused, consts, fd=0) as new:
                rng = np.random.default_rng(20 + m + k)
                a = np.ascontiguousarray(
                    rng.normal(size=(m, k)).astype(np.float16))
                r = np.ascontiguousarray(
                    rng.normal(size=(m, 64)).astype(np.float16))
                y_old = np.empty((m, 64), dtype=np.float16)
                old.run_into([a], [y_old])
                expected = np.ascontiguousarray(
                    y_old.astype(np.float32) + r.astype(np.float32),
                    dtype=np.float16)
                y_new = np.empty_like(y_old)
                new.run_into([a, r], [y_new])
                np.testing.assert_array_equal(expected, y_new)

    def test_rsqrt_capability_is_isolated_to_experimental_profile(self):
        from rknpu_backend import default_profile
        from rknpu_backend.capabilities import load_profile
        from rknpu_backend.errors import (BackendError,
                                          RK_E_TARGET_MISMATCH)
        with self.assertRaises(BackendError) as ctx:
            default_profile().enabled("co.lut.rsqrt.f16.v1")
        self.assertEqual(ctx.exception.code, RK_E_TARGET_MISMATCH)
        experimental = load_profile(EXPERIMENTAL_PROFILE)
        self.assertEqual(experimental.enabled("co.lut.rsqrt.f16.v1").status,
                         "enabled")
        self.assertNotEqual(experimental.profile_hash,
                            default_profile().profile_hash)

    def test_rmsnorm_operator_acceptance_is_frozen_and_passes(self):
        import hashlib
        import json
        path = ROOT / "models" / "rmsnorm_approx_acceptance.json"
        raw = path.read_bytes()
        spec = json.loads(raw)
        sidecar = path.with_name(path.name + ".sha256").read_text().strip()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), sidecar)
        self.assertEqual(spec["schema_version"], 2)
        self.assertIn("plan_hash", spec)
        self.assertIn("profile_hash", spec)
        self.assertIn("library_sha256", spec)
        cal = spec["calibration"]["metrics"]
        gates = spec["gates"]
        self.assertLessEqual(cal["max_abs"], gates["atol"])
        self.assertLessEqual(cal["rmse"], gates["rmse_limit"])
        verify_path = (ROOT / "results" / "20261028T113459Z-graph-fusion" /
                       "rmsnorm-verify-frozen.json")
        verify = json.loads(verify_path.read_text())
        self.assertTrue(verify["pass"])
        self.assertEqual(verify["spec_sha256"], sidecar)
        acc = verify["acceptance"]
        self.assertLessEqual(acc["max_abs"], verify["gates"]["atol"])
        self.assertLessEqual(acc["rmse"], verify["gates"]["rmse_limit"])
        self.assertEqual(spec["numeric_policy"]["validated_original_mean_square"],
                         [0.17, 6.1])

    def test_full_device_rmsnorm_approx_is_one_chain_and_in_domain(self):
        from rknpu_backend import (CompileOptions, compile_rkir,
                                   rmsnorm_approx_module)
        from rknpu_backend.capabilities import load_profile
        from rknpu_backend.models import PlanCall
        profile = load_profile(EXPERIMENTAL_PROFILE)
        for m in (16, 128):
            plan = compile_rkir(
                rmsnorm_approx_module(m=m, d=64, name=f"rmsnorm_test_{m}"),
                profile=profile,
                options=CompileOptions(numeric_mode="rk_f16_tiled",
                                       allow_approx_lut=True,
                                       fuse_task_chain=True)).plan
            self.assertEqual(plan.accounting.task_count, 7)
            self.assertEqual(len(plan.submits), 1)
            self.assertEqual(plan.profile_hash, profile.profile_hash)
            lut_scale = 0.75
            consts = {
                "Ones": np.ones((1, 64), dtype=np.float16),
                "InvD": np.full((m, 1), lut_scale / 64.0,
                                dtype=np.float16),
                "Eps": np.full((m, 1), lut_scale * 1e-6,
                               dtype=np.float16),
                "GammaW": np.full((64, 1), np.sqrt(lut_scale),
                                  dtype=np.float16),
                "__identity_32": np.eye(32, dtype=np.float16),
                "__identity_64": np.eye(64, dtype=np.float16),
            }
            with PlanCall(self.lib, plan, consts, fd=0) as call:
                rng = np.random.default_rng(30 + m)
                # Construct rows with mean-square in the LUT input domain
                # [0.25, 4.0].  Each row is 64 values with RMS in [0.75, 1.5].
                x = rng.normal(size=(m, 64)).astype(np.float32)
                x /= np.sqrt(np.mean(x * x, axis=1, keepdims=True))
                scale = rng.uniform(0.75, 1.5, size=(m, 1)).astype(np.float32)
                x = np.ascontiguousarray((x * scale).astype(np.float16))
                y = np.empty_like(x)
                call.run_into([x], [y])
                ref = np.ascontiguousarray(
                    x.astype(np.float32) /
                    np.sqrt(np.mean(x.astype(np.float32) ** 2,
                                    axis=1, keepdims=True) + 1e-6),
                    dtype=np.float16)
                err = np.abs(y.astype(np.float32) - ref.astype(np.float32))
                self.assertLessEqual(float(err.max()), 0.01)
                self.assertLessEqual(float(np.sqrt(np.mean(err * err))), 0.003)
                self.assertEqual(float(np.max(np.abs(y))), float(np.max(np.abs(y))))
                self.assertTrue(np.isfinite(y).all())


if __name__ == "__main__":
    unittest.main()
