# SPDX-License-Identifier: GPL-3.0-or-later
"""The opt-in TileLang backend registration: nothing on import, the new chain afterwards.

Every registration here is explicit and scoped: importing
rknpu_backend.frontend.tilelang_backend registers nothing, registered() restores
the previous state on exit, and the compiles go through the registered backend
entry (create_backend_context / tilelang.engine.lower) rather than through
api.compile_tilelang or the structured prototype.
"""
from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import pytest

import tilelang  # noqa: F401  (loads the vendored TVM)
import tilelang.language as T
import tvm

from rknpu_backend.capabilities import default_profile
from rknpu_backend.errors import (RK_E_NUMERIC_CONTRACT, RK_E_SHAPE,
                                  RK_E_UNSUPPORTED, BackendError)
from rknpu_backend.frontend import rknpu_pass as R
from rknpu_backend.frontend import tilelang_backend as TB
from rknpu_backend.options import CompileOptions

ROOT = Path(__file__).resolve().parents[1]
PILOT_STAGES = ["bind-target", "materialize-kernel-launch", "localise",
                "instantiate", "plan"]
M, K, N, BM, BN = 256, 576, 512, 256, 256


@T.prim_func
def gemm_small(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"),
               Y: T.Tensor((M, N), "float16")):
    """The pilot domain: guarded-tail FP16 GEMM, one tile-K tail at K=576."""
    with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN), threads=128) as (im, inn):
        As = T.alloc_shared((BM, 256), "float16")
        Bs = T.alloc_shared((256, BN), "float16")
        Cl = T.alloc_fragment((BM, BN), "float32")
        Acc = T.alloc_fragment((BM, BN), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(T.ceildiv(K, 256), num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(BM, 256):
                if im * BM + i < M and ko * 256 + j < K:
                    As[i, j] = A[im * BM + i, ko * 256 + j]
            for i, j in T.Parallel(256, BN):
                if ko * 256 + i < K and inn * BN + j < N:
                    Bs[i, j] = B[ko * 256 + i, inn * BN + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(BM, BN):
            if im * BM + i < M and inn * BN + j < N:
                Y[im * BM + i, inn * BN + j] = Acc[i, j]


@T.prim_func
def add_only(A: T.Tensor((256, 256), "float16"),
             B: T.Tensor((256, 256), "float16"),
             Y: T.Tensor((256, 256), "float16")):
    """Outside the pilot domain: elementwise, no guarded-tail GEMM at all."""
    with T.Kernel(1, threads=128):
        for i, j in T.Parallel(256, 256):
            Y[i, j] = A[i, j] + B[i, j]


@T.prim_func
def gemm_fp32(A: T.Tensor((256, 256), "float32"),
              B: T.Tensor((256, 256), "float32"),
              Y: T.Tensor((256, 256), "float32")):
    """Outside the pilot domain: the chain only proves the rk_f16_tiled contract."""
    with T.Kernel(1, 1, threads=128) as (im, inn):
        As = T.alloc_shared((256, 256), "float32")
        Bs = T.alloc_shared((256, 256), "float32")
        Cl = T.alloc_fragment((256, 256), "float32")
        T.clear(Cl)
        for ko in T.Pipelined(1, num_stages=2):
            T.copy(A, As)
            T.copy(B, Bs)
            T.gemm(As, Bs, Cl)
        T.copy(Cl, Y)


def options():
    return CompileOptions(numeric_mode="rk_f16_tiled", allow_approx_lut=False)


def _root_predicate(prim_func, predicate):
    def change(node):
        if type(node).__name__ == "SBlockRealize" and \
                str(node.block.name_hint) == "tilelang_root":
            return tvm.tirx.SBlockRealize(node.iter_values, predicate,
                                         node.block)
        return None
    return prim_func.with_body(tvm.tirx.stmt_functor.ir_transform(
        prim_func.body, None, change))


def _output_mask(prim_func, predicate):
    def change(node):
        if type(node).__name__ == "BufferStore" and str(node.buffer.name) == "Y":
            return tvm.tirx.BufferStore(node.buffer, node.value, node.indices,
                                        predicate)
        return None
    return prim_func.with_body(tvm.tirx.stmt_functor.ir_transform(
        prim_func.body, None, change))


def _effectful_output_index(prim_func):
    def change(node):
        if type(node).__name__ == "BufferStore" and str(node.buffer.name) == "Y":
            dead = tvm.tirx.Mul(
                tvm.tirx.IntImm("int32", 0),
                tvm.tirx.call_extern("int32", "audit_effect"))
            return tvm.tirx.BufferStore(
                node.buffer, node.value,
                [tvm.tirx.Add(node.indices[0], dead), node.indices[1]],
                node.predicate)
        return None
    return prim_func.with_body(tvm.tirx.stmt_functor.ir_transform(
        prim_func.body, None, change))


def _parameter_offset(prim_func, name, offset, extra_param=None):
    key, old = next((key, buffer) for key, buffer in prim_func.buffer_map.items()
                    if str(buffer.name) == name)
    view = tvm.tirx.decl_buffer(
        old.shape, old.dtype, name=str(old.name), data=old.data,
        strides=old.strides, elem_offset=offset, scope=R._scope_of(old))

    def change(node):
        if type(node).__name__ == "BufferLoad" and node.buffer.same_as(old):
            return tvm.tirx.BufferLoad(view, node.indices, node.predicate)
        if type(node).__name__ == "BufferStore" and node.buffer.same_as(old):
            return tvm.tirx.BufferStore(view, node.value, node.indices,
                                        node.predicate)
        return None
    body = tvm.tirx.stmt_functor.ir_transform(prim_func.body, None, change)
    mapping = dict(prim_func.buffer_map)
    mapping[key] = view
    params = list(prim_func.params)
    if extra_param is not None:
        params.append(extra_param)
    return tvm.tirx.PrimFunc(params, body, prim_func.ret_type,
                             mapping, prim_func.attrs)


def _gemm_argument(prim_func, index, value, extra_param=None):
    def change(node):
        if type(node).__name__ == "Evaluate" and \
                type(node.value).__name__ == "Call" and \
                str(getattr(node.value.op, "name", "")) == "tl.tileop.gemm":
            args = list(node.value.args)
            args[index] = value
            call = tvm.tirx.Call(node.value.dtype, node.value.op, args,
                                 getattr(node.value, "annotations", None))
            return tvm.tirx.Evaluate(call)
        return None
    body = tvm.tirx.stmt_functor.ir_transform(prim_func.body, None, change)
    params = list(prim_func.params)
    if extra_param is not None:
        params.append(extra_param)
    return tvm.tirx.PrimFunc(params, body, prim_func.ret_type,
                             prim_func.buffer_map, prim_func.attrs)


def _rknpu_declarations(prim_func, *, function=None, block=None,
                        loop=None, call=None):
    def change(node):
        if type(node).__name__ == "SBlock" and \
                str(node.name_hint) == "tilelang_root" and block:
            annotations = dict(node.annotations)
            annotations.update(block)
            return tvm.tirx.SBlock(node.iter_vars, node.reads, node.writes,
                                   node.name_hint, node.body, node.init,
                                   node.alloc_buffers, node.match_buffers,
                                   annotations)
        if type(node).__name__ == "For" and \
                str(node.loop_var) == "ko" and loop:
            annotations = dict(node.annotations)
            annotations.update(loop)
            return tvm.tirx.For(node.loop_var, node.min, node.extent,
                                node.kind, node.body, node.thread_binding,
                                annotations, node.step)
        if type(node).__name__ == "Call" and \
                str(getattr(node.op, "name", "")) == "tl.tileop.gemm" and call:
            annotations = dict(getattr(node, "annotations", None) or {})
            for key, value in call.items():
                annotations[key] = (tvm.tirx.IntImm("int32", value)
                                    if type(value) is int else
                                    tvm.tirx.StringImm(value)
                                    if type(value) is str else value)
            return tvm.tirx.Call(node.dtype, node.op, list(node.args), annotations)
        return None

    body = tvm.tirx.stmt_functor.ir_transform(prim_func.body, None, change)
    attrs = dict(getattr(prim_func, "attrs", None) or {})
    attrs.update(function or {})
    return prim_func.with_body(body).with_attrs(attrs)


def _copy_hint(prim_func, key, value):
    def change(node):
        if type(node).__name__ == "Call" and \
                str(getattr(node.op, "name", "")) == "tl.tileop.copy":
            annotations = dict(getattr(node, "annotations", None) or {})
            annotations[key] = tvm.tirx.IntImm("int32", value)
            return tvm.tirx.Call(node.dtype, node.op, list(node.args), annotations)
        return None
    return prim_func.with_body(tvm.tirx.stmt_functor.ir_transform(
        prim_func.body, None, change))


def _operand_load_mask(prim_func, predicate):
    def change(node):
        if type(node).__name__ == "BufferLoad" and str(node.buffer.name) == "A":
            return tvm.tirx.BufferLoad(node.buffer, node.indices, predicate)
        return None
    return prim_func.with_body(tvm.tirx.stmt_functor.ir_transform(
        prim_func.body, None, change))


def _effectful_output_value(prim_func):
    def change(node):
        if type(node).__name__ == "BufferStore" and str(node.buffer.name) == "Y":
            dead = tvm.tirx.Mul(
                tvm.tirx.FloatImm("float16", 0.0),
                tvm.tirx.call_extern("float16", "audit_effect"))
            return tvm.tirx.BufferStore(node.buffer,
                                        tvm.tirx.Add(node.value, dead),
                                        node.indices, node.predicate)
        return None
    return prim_func.with_body(tvm.tirx.stmt_functor.ir_transform(
        prim_func.body, None, change))


def require_backend_api():
    """Skip with the installed build's own reason when nothing can be registered."""
    if TB.available():
        return
    try:
        TB.backend_api()
        reason = "the installed TileLang exposes no registrable backend API"
    except BackendError as err:
        reason = err.diagnostics[0].message + " :: " + str(err.diagnostics[0].actual)
    except Exception as err:  # noqa: BLE001 - report whatever the build raised
        reason = f"{type(err).__name__}: {err}"
    pytest.skip(f"a TileLang backend registration test needs tilelang.backend: {reason}")


def child_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "python"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return env


class ImportPurityTests(unittest.TestCase):
    """Importing the module must not register anything, anywhere."""

    def test_a_fresh_interpreter_imports_the_module_and_registers_nothing(self):
        script = "\n".join([
            "import rknpu_backend",
            "import rknpu_backend.frontend.tilelang_backend as tb",
            "from tilelang.backend import list_backends, list_target_detectors",
            "from tilelang.backend import target as target_api",
            "print('available', tb.available())",
            "print('registered', tb.is_registered())",
            "print('backends', ','.join(sorted(list_backends())))",
            "print('normalizers', 'rknpu' in getattr(target_api, '_TARGET_NORMALIZERS', {}))",
            "print('detectors', ','.join(sorted(list_target_detectors())))",
        ])
        done = subprocess.run([sys.executable, "-c", script], cwd=str(ROOT),
                              env=child_env(), capture_output=True, text=True,
                              timeout=900)
        self.assertEqual(done.returncode, 0, done.stderr)
        reported = dict(line.split(" ", 1)
                        for line in done.stdout.strip().splitlines())
        self.assertEqual(reported["registered"], "False")
        self.assertNotIn("rknpu", reported["backends"].split(","))
        self.assertEqual(reported["normalizers"], "False")
        self.assertNotIn("rknpu", reported["detectors"].split(","))

    def test_a_fresh_interpreter_compiles_through_the_entry_without_the_old_frontends(self):
        require_backend_api()
        script = "\n".join([
            "import importlib.util, sys",
            "spec = importlib.util.spec_from_file_location('under_test', sys.argv[1])",
            "module = importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(module)",
            "import tilelang",
            "import tvm",
            "tb = module.TB",
            "tb.register_rknpu_backend()",
            "context = tilelang.backend.create_backend_context('rknpu')",
            "context.lower(tvm.IRModule({'main': module.gemm_small}))",
            "record = tb.last_lowering()",
            "print('plan_hash', record.plan_hash)",
            "print('stages', ','.join(name for name, _ in record.stages))",
            "print('entry', record.to_dict()['entry'])",
            "for name in ('structured',):",
            "    full = 'rknpu_backend.frontend.' + name",
            "    print(name, full in sys.modules)",
            "print('unregistered', tb.unregister_rknpu_backend())",
            "print('registered_after', tb.is_registered())",
        ])
        done = subprocess.run(
            [sys.executable, "-c", script,
             str(Path(__file__).resolve())],
            cwd=str(ROOT), env=child_env(), capture_output=True, text=True,
            timeout=900)
        self.assertEqual(done.returncode, 0, done.stderr)
        reported = dict(line.split(" ", 1)
                        for line in done.stdout.strip().splitlines())
        self.assertEqual(reported["stages"], ",".join(PILOT_STAGES))
        self.assertEqual(reported["entry"],
                         "rknpu_backend.frontend.rknpu_pass.compile_guarded_tail")
        self.assertEqual(reported["structured"], "False")
        self.assertEqual(reported["unregistered"], "True")
        self.assertEqual(reported["registered_after"], "False")

    def test_this_process_is_not_registered(self):
        require_backend_api()
        self.assertFalse(TB.is_registered())
        self.assertNotIn("rknpu", tilelang.backend.list_backends())

    def test_importing_this_module_installs_no_engine_shim(self):
        require_backend_api()
        self.assertFalse(TB.is_registered())
        self.assertFalse(hasattr(tvm.IRModule, "inspect_source"))


class RegistrationSurfaceTests(unittest.TestCase):
    """What register_rknpu_backend() adds, and what it leaves alone."""

    def setUp(self):
        require_backend_api()
        self.assertFalse(TB.is_registered())

    def test_registration_adds_exactly_the_rknpu_backend(self):
        before = set(tilelang.backend.list_backends())
        with TB.registered() as backend:
            after = set(tilelang.backend.list_backends())
            self.assertEqual(after - before, {"rknpu"})
            self.assertIs(backend, TB.backend_module())
            self.assertIs(tilelang.backend.get_backend("rknpu"), backend)
            self.assertEqual(backend.target_kinds, ("ext_dev",))
            self.assertEqual(tuple(spec.name for spec in backend.execution_backends),
                             ("rknpu_plan",))
            self.assertEqual(sorted(backend.pipelines), [TB.RKNPU_TARGET_KIND])
            self.assertEqual(sorted(backend.device_codegens), [TB.RKNPU_TARGET_KIND])
            self.assertFalse(backend.callbacks)
            self.assertFalse(backend.host_codegens)
        self.assertFalse(TB.is_registered())
        self.assertEqual(set(tilelang.backend.list_backends()), before)

    def test_the_rknpu_target_name_resolves_to_a_marked_ext_dev_target(self):
        with TB.registered():
            context = tilelang.backend.create_backend_context(TB.RKNPU_TARGET_NAME)
            self.assertEqual(context.name, "rknpu")
            self.assertIs(context.module, TB.backend_module())
            self.assertEqual(context.target.kind.name, "ext_dev")
            self.assertIn(TB.RKNPU_TARGET_KEY, list(context.target.keys))
            self.assertEqual(context.execution_backend.name, "rknpu_plan")
            self.assertEqual(context.execution_backend.enable_host_codegen, False)
            self.assertEqual(context.execution_backend.enable_device_compile, False)

    def test_a_bare_ext_dev_target_is_not_claimed(self):
        with TB.registered():
            with self.assertRaises(ValueError) as caught:
                tilelang.backend.create_backend_context({"kind": "ext_dev"})
            self.assertIn("No backend matches target", str(caught.exception))

    def test_the_old_entry_points_still_work_while_registered(self):
        from rknpu_backend import compile_tilelang

        before = compile_tilelang(gemm_small, profile=default_profile(),
                                  options=options()).plan.plan_hash
        with TB.registered():
            during = compile_tilelang(gemm_small, profile=default_profile(),
                                      options=options()).plan.plan_hash
        self.assertEqual(during, before)

    def test_the_explicit_entry_is_idempotent_and_removable(self):
        try:
            first = TB.register_rknpu_backend()
            second = TB.register_rknpu_backend()
            self.assertIs(first, second)
            self.assertTrue(TB.is_registered())
        finally:
            self.assertTrue(TB.unregister_rknpu_backend())
        self.assertFalse(TB.is_registered())
        self.assertNotIn("rknpu", tilelang.backend.list_backends())
        self.assertFalse(hasattr(tvm.IRModule, "inspect_source"))


class RegisteredLoweringTests(unittest.TestCase):
    """A compile through the registered entry runs the new RKNPU chain."""

    @classmethod
    def setUpClass(cls):
        require_backend_api()
        TB.clear_lowerings()
        cls._stack = contextlib.ExitStack()
        cls.backend = cls._stack.enter_context(TB.registered())
        cls.old_frontend_calls = 0

        # Imported here, not at module scope, so that a child interpreter which
        # loads this file can prove the old frontends are never imported.
        from rknpu_backend import api as backend_api_module
        from rknpu_backend.frontend import structured

        def never_called(*args, **kwargs):
            del args, kwargs
            cls.old_frontend_calls += 1
            raise AssertionError(
                "the registered lowering called an old frontend entry point")

        for owner, name in ((backend_api_module, "compile_tilelang"),
                            (structured, "stage0_capture")):
            if hasattr(owner, name):
                cls._stack.enter_context(mock.patch.object(
                    owner, name, never_called))

        cls.context = tilelang.backend.create_backend_context(TB.RKNPU_TARGET_NAME)
        cls.lowered = cls.context.lower(tvm.IRModule({"main": gemm_small}))
        cls.record = TB.last_lowering()
        (cls.direct_plan, cls.direct_stages, cls.direct_facts,
         cls.direct_tasks) = R.compile_guarded_tail(
            gemm_small, profile=default_profile(), options=options())

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()
        TB.clear_lowerings()

    def test_the_registered_entry_produces_the_same_plan_as_the_chain(self):
        self.assertIsNotNone(self.record)
        self.assertEqual(self.record.plan_hash, self.direct_plan.plan_hash)
        self.assertEqual(self.record.plan, self.direct_plan)
        self.assertEqual(len(self.record.tasks), len(self.direct_tasks))
        self.assertEqual(self.record.facts, self.direct_facts.to_dict())

    def test_the_chain_that_ran_is_the_new_one(self):
        self.assertEqual([name for name, _ in self.record.stages], PILOT_STAGES)
        self.assertEqual([kind for _, kind in self.record.stages],
                         ["transform", "transform", "transform", "transform",
                          "analysis"])
        self.assertIs(self.record.entry, R.compile_guarded_tail)
        self.assertEqual(self.record.entry.__module__,
                         "rknpu_backend.frontend.rknpu_pass")
        self.assertEqual(self.record.facts["tile"], [BM, BN, 256])
        gemms = [task for task in self.record.tasks if task["kind"] == "gemm"]
        self.assertEqual(len(gemms), 6)

    def test_the_lowered_module_carries_the_plan_identity(self):
        attrs = self.lowered.attrs
        self.assertEqual(str(attrs["rknpu_plan_hash"]), self.direct_plan.plan_hash)
        self.assertEqual(str(attrs["rknpu_stages"]), ",".join(PILOT_STAGES))
        self.assertEqual(int(attrs["rknpu_tasks"]), len(self.record.tasks))
        self.assertEqual(str(attrs["rknpu_target_key"]), TB.RKNPU_TARGET_KEY)

    def test_no_old_frontend_was_called(self):
        self.assertEqual(self.old_frontend_calls, 0)

    def test_the_engine_entry_runs_the_chain_and_returns_the_plan(self):
        recorded = len(TB.lowerings())
        product = tilelang.engine.lower(tvm.IRModule({"main": gemm_small}),
                                        target=TB.RKNPU_TARGET_NAME)
        self.assertEqual(type(product).__name__, "CompiledArtifact")
        self.assertEqual(json.loads(product.kernel_source),
                         self.direct_plan.to_dict())
        self.assertEqual(len(TB.lowerings()), recorded + 1)
        self.assertEqual(TB.last_lowering().plan_hash, self.direct_plan.plan_hash)
        self.assertEqual([name for name, _ in TB.last_lowering().stages],
                         PILOT_STAGES)

    def test_the_lowered_function_carries_the_artifact_too(self):
        carried = TB.plan_json_from_module(self.lowered["main"])
        self.assertIsNotNone(carried)
        self.assertEqual(json.loads(carried), self.direct_plan.to_dict())
        self.assertEqual(
            str(self.lowered["main"].attrs[TB.RKNPU_PLAN_HASH_ATTR]),
            self.direct_plan.plan_hash)


class EngineProductTests(unittest.TestCase):
    """The plan is one compilation's product, carried by the artifact itself."""

    @classmethod
    def setUpClass(cls):
        require_backend_api()
        cls._stack = contextlib.ExitStack()
        cls._stack.enter_context(TB.registered())
        cls.context = tilelang.backend.create_backend_context(TB.RKNPU_TARGET_NAME)
        (cls.direct_plan, cls.direct_stages, cls.direct_facts,
         cls.direct_tasks) = R.compile_guarded_tail(
            gemm_small, profile=default_profile(), options=options())
        cls.product = tilelang.engine.lower(
            tvm.IRModule({"main": gemm_small}), target=TB.RKNPU_TARGET_NAME)
        cls.compilation = TB.compile_to_plan(gemm_small,
                                             profile=default_profile(),
                                             options=options())

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()
        TB.clear_lowerings()

    def test_the_engine_source_parses_as_this_compilations_plan(self):
        document = json.loads(self.product.kernel_source)
        self.assertEqual(document["kind"], "rkplan")
        self.assertEqual(document["plan_hash"], self.direct_plan.plan_hash)
        self.assertEqual(document, self.direct_plan.to_dict())

    def test_the_artifact_reconstructs_the_plan_the_chain_returned(self):
        plan = TB.plan_from_artifact(self.product)
        self.assertEqual(plan.plan_hash, self.direct_plan.plan_hash)
        self.assertEqual(plan.to_dict(), self.direct_plan.to_dict())
        self.assertEqual(plan, self.direct_plan)
        from_device_mod = TB.plan_from_module(self.product.device_mod)
        self.assertEqual(from_device_mod.to_dict(), self.direct_plan.to_dict())

    def test_the_global_lowering_log_is_not_part_of_the_contract(self):
        TB.clear_lowerings()
        try:
            self.assertEqual(TB.lowerings(), ())
            self.assertIsNone(TB.last_lowering())
            plan = TB.plan_from_artifact(self.product)
            self.assertEqual(plan.plan_hash, self.direct_plan.plan_hash)
            self.assertEqual(plan.to_dict(), self.direct_plan.to_dict())
        finally:
            TB.clear_lowerings()

    def test_the_module_and_its_function_both_carry_the_artifact(self):
        module = self.compilation.module
        module_text = str(module.attrs[TB.RKNPU_PLAN_JSON_ATTR])
        function_text = str(module["main"].attrs[TB.RKNPU_PLAN_JSON_ATTR])
        self.assertEqual(module_text, function_text)
        self.assertEqual(json.loads(module_text), self.direct_plan.to_dict())
        self.assertEqual(str(module.attrs[TB.RKNPU_PLAN_HASH_ATTR]),
                         self.direct_plan.plan_hash)
        self.assertEqual(str(module["main"].attrs[TB.RKNPU_PLAN_HASH_ATTR]),
                         self.direct_plan.plan_hash)
        # the engine's host module keeps the function, so it keeps that copy
        self.assertEqual(json.loads(str(self.product.host_mod["main"].attrs[
            TB.RKNPU_PLAN_JSON_ATTR])), self.direct_plan.to_dict())

    def test_compile_to_plan_returns_the_product_without_recording_a_global(self):
        TB.clear_lowerings()
        try:
            self.assertEqual(self.compilation.plan_hash,
                             self.direct_plan.plan_hash)
            self.assertEqual(json.loads(self.compilation.plan_json),
                             self.direct_plan.to_dict())
            self.assertEqual([stage.name for stage in self.compilation.stages],
                             PILOT_STAGES)
            self.assertEqual(self.compilation.facts.to_dict(),
                             self.direct_facts.to_dict())
            self.assertEqual(len(self.compilation.tasks), len(self.direct_tasks))
            self.assertEqual(TB.lowerings(), ())
            self.assertEqual(TB.plan_from_module(self.compilation.module)
                             .to_dict(), self.direct_plan.to_dict())
            record = self.compilation.to_dict()
            self.assertEqual(record["plan_hash"], self.direct_plan.plan_hash)
            self.assertIn(TB.RKNPU_PLAN_JSON_ATTR, record["module_attrs"])
            self.assertEqual(record["tasks"], len(self.direct_tasks))
        finally:
            TB.clear_lowerings()

    def test_device_codegen_refuses_a_module_this_pipeline_did_not_lower(self):
        for compile_device in (False, True):
            with self.assertRaises(BackendError) as caught:
                self.context.codegen_device(tvm.IRModule({"main": gemm_small}),
                                            compile_device=compile_device)
            err = caught.exception
            self.assertEqual(err.code, RK_E_UNSUPPORTED, compile_device)
            self.assertEqual(err.diagnostics[0].stage, "codegen", compile_device)
        with self.assertRaises(BackendError):
            TB.plan_from_module(tvm.IRModule({"main": gemm_small}))
        foreign = type("Foreign", (), {"kernel_source": "",
                                       "device_mod": tvm.IRModule(
                                           {"main": gemm_small})})()
        with self.assertRaises(BackendError):
            TB.plan_from_artifact(foreign)

    def test_the_missing_engine_method_is_a_scoped_shim(self):
        self.assertIs(tvm.IRModule.inspect_source, TB.inspect_source_shim)
        self.assertEqual(self.product.device_mod.inspect_source(),
                         self.product.kernel_source)
        with self.assertRaises(BackendError) as caught:
            tvm.IRModule({"main": gemm_small}).inspect_source()
        self.assertEqual(caught.exception.code, RK_E_UNSUPPORTED)


class PilotDomainRejectionTests(unittest.TestCase):
    """Everything outside guarded-tail FP16 GEMM is rejected, not accepted."""

    @classmethod
    def setUpClass(cls):
        require_backend_api()
        cls._stack = contextlib.ExitStack()
        cls._stack.enter_context(TB.registered())
        cls.context = tilelang.backend.create_backend_context(TB.RKNPU_TARGET_NAME)

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()

    def assert_rejected(self, prim_func, why):
        TB.clear_lowerings()
        with self.assertRaises(BackendError) as caught:
            self.context.lower(tvm.IRModule({"main": prim_func}))
        err = caught.exception
        self.assertIn(err.code,
                      {RK_E_SHAPE, RK_E_UNSUPPORTED, RK_E_NUMERIC_CONTRACT}, why)
        self.assertEqual(err.diagnostics[0].stage, "pass", why)
        self.assertEqual(err.diagnostics[0].code_name,
                         {RK_E_SHAPE: "RK_E_SHAPE",
                          RK_E_UNSUPPORTED: "RK_E_UNSUPPORTED",
                          RK_E_NUMERIC_CONTRACT: "RK_E_NUMERIC_CONTRACT"}[err.code],
                         why)
        self.assertEqual(TB.lowerings(), (), why)

    def assert_engine_rejected(self, prim_func, phrase):
        with self.assertRaises(BackendError) as caught:
            tilelang.engine.lower(tvm.IRModule({"main": prim_func}),
                                  target=TB.RKNPU_TARGET_NAME)
        self.assertIn(phrase, caught.exception.diagnostics[0].message)

    def test_registered_engine_rejects_false_selective_and_store_mask(self):
        _, grid_n, _ = R._grid_and_k_loops(gemm_small)
        zero = tvm.tirx.IntImm(grid_n.loop_var.dtype, 0)
        one = tvm.tirx.IntImm(grid_n.loop_var.dtype, 1)
        cases = (
            (_root_predicate(gemm_small, tvm.tirx.IntImm("bool", 0)),
             "SBlockRealize predicate"),
            (_root_predicate(gemm_small, tvm.tirx.LT(grid_n.loop_var, one)),
             "SBlockRealize predicate"),
            (_root_predicate(gemm_small, tvm.tirx.EQ(grid_n.loop_var, zero)),
             "SBlockRealize predicate"),
            (_output_mask(gemm_small, tvm.tirx.IntImm("bool", 0)),
             "BufferStore predicate"),
        )
        for prim, phrase in cases:
            with self.subTest(phrase=phrase):
                self.assert_engine_rejected(prim, phrase)

    def test_registered_engine_accepts_domain_proven_true_block_predicate(self):
        _, grid_n, _ = R._grid_and_k_loops(gemm_small)
        predicate = tvm.tirx.LT(
            grid_n.loop_var,
            tvm.tirx.IntImm(grid_n.loop_var.dtype, int(grid_n.extent)))
        product = tilelang.engine.lower(
            tvm.IRModule({"main": _root_predicate(gemm_small, predicate)}),
            target=TB.RKNPU_TARGET_NAME)
        self.assertEqual(TB.plan_from_artifact(product).plan_hash,
                         R.compile_guarded_tail(
                             gemm_small, options=options())[0].plan_hash)

    def test_registered_engine_rejects_unconsumed_calls_masks_offsets_and_flags(self):
        false = tvm.tirx.IntImm("bool", 0)
        _, grid_n, _ = R._grid_and_k_loops(gemm_small)
        input_a = next(buffer for buffer in gemm_small.buffer_map.values()
                       if str(buffer.name) == "A")
        data_condition = tvm.tirx.GT(
            tvm.tirx.BufferLoad(input_a, [tvm.tirx.IntImm("int32", 0),
                                           tvm.tirx.IntImm("int32", 0)]),
            tvm.tirx.FloatImm("float16", 0.0))
        clear_true = tvm.tirx.EQ(tvm.tirx.IntImm("int32", 0),
                                 tvm.tirx.IntImm("int32", 0))
        unknown_clear = tvm.tirx.Var("engine_clear_flag", "bool")
        unknown_offset = tvm.tirx.Var("engine_offset", "int32")
        cases = [
            (_effectful_output_index(gemm_small), "call expression"),
            (_effectful_output_value(gemm_small), "call expression"),
            (_output_mask(gemm_small, false), "BufferStore predicate"),
            (_operand_load_mask(gemm_small, false), "BufferLoad predicate"),
            (_root_predicate(gemm_small, data_condition),
             "SBlockRealize predicate"),
            (_gemm_argument(gemm_small, 9, clear_true),
             "clear_accum is not supported"),
            (_gemm_argument(gemm_small, 9, unknown_clear,
                            extra_param=unknown_clear),
             "clear_accum must be statically known"),
        ]
        for buffer_name in ("A", "B", "Y"):
            cases.append((_parameter_offset(gemm_small, buffer_name, 576),
                          "nonzero element offset"))
        cases.append((_parameter_offset(gemm_small, "A", unknown_offset,
                                        extra_param=unknown_offset),
                      "element offset is not statically known"))
        for prim, phrase in cases:
            with self.subTest(phrase=phrase):
                self.assert_engine_rejected(prim, phrase)

    def test_registered_engine_consumes_supported_equivalent_zero_fields(self):
        zero = tvm.tirx.Add(tvm.tirx.IntImm("int32", 0),
                            tvm.tirx.IntImm("int32", 0))
        false_expr = tvm.tirx.EQ(tvm.tirx.IntImm("int32", 0),
                                 tvm.tirx.IntImm("int32", 1))
        prim = _parameter_offset(gemm_small, "A", zero)
        prim = _gemm_argument(prim, 9, false_expr)
        prim = _gemm_argument(prim, 3, false_expr)
        artifact = tilelang.engine.lower(
            tvm.IRModule({"main": prim}), target=TB.RKNPU_TARGET_NAME)
        direct = R.compile_guarded_tail(
            gemm_small, profile=default_profile(), options=options())[0]
        self.assertEqual(TB.plan_from_artifact(artifact).plan_hash,
                         direct.plan_hash)

    def test_registered_engine_consumes_duplicate_declarations_from_each_scope(self):
        prim = _rknpu_declarations(
            gemm_small,
            function={"rknpu.schedule": "mn_tiles_v1"},
            block={"rknpu.tile_m": 256},
            loop={"rknpu.tile_m": 256},
            call={"rknpu.tile_n": 256, "rknpu.k_tile": 256})
        artifact = tilelang.engine.lower(
            tvm.IRModule({"main": prim}), target=TB.RKNPU_TARGET_NAME)
        self.assertIsNotNone(TB.plan_from_artifact(artifact))
        trace = TB.last_lowering().facts["hints"]["rknpu_declarations"]
        self.assertEqual(len(trace["source"]), 5)
        self.assertEqual(len(trace["normalized"]), 5)
        self.assertTrue(all(row["consumed"] for row in trace["source"]))
        self.assertEqual(sum(row["key"] == "rknpu.tile_m"
                             for row in trace["source"]), 2)

    def test_registered_engine_rejects_conflicting_and_invalid_declarations(self):
        cases = [
            (_rknpu_declarations(gemm_small,
                                 block={"rknpu.tile_m": 128},
                                 loop={"rknpu.tile_m": 256}),
             "declaration disagrees"),
            (_rknpu_declarations(gemm_small,
                                 block={"rknpu.tile_m": 256},
                                 loop={"rknpu.tile_m": 128}),
             "declaration disagrees"),
            (_rknpu_declarations(gemm_small,
                                 block={"rknpu.unimplemented": 1}),
             "not supported by the pass chain"),
            (_rknpu_declarations(gemm_small,
                                 block={"rknpu.tile_m": "256"}),
             "wrong value type"),
        ]
        for prim, phrase in cases:
            with self.subTest(phrase=phrase, source=prim.script()[:180]):
                self.assert_engine_rejected(prim, phrase)

    def test_module_level_rknpu_requests_are_rejected_at_that_scope(self):
        self.assertIsNotNone(TB.compile_to_plan(
            tvm.IRModule({"main": gemm_small})).plan)
        for key, value in (("rknpu.tile_m", 128),
                           ("rknpu.unimplemented", 1)):
            module = tvm.IRModule({"main": gemm_small}, attrs={key: value})
            with self.subTest(key=key):
                with self.assertRaises(BackendError) as caught:
                    TB.compile_to_plan(module)
                diagnostic = caught.exception.diagnostics[0]
                self.assertEqual(diagnostic.stage, "backend")
                self.assertIn("module-level rknpu declarations are not supported",
                              diagnostic.message)
                self.assertEqual(diagnostic.to_dict()["actual"][0]["key"], key)
                with self.assertRaises(BackendError) as engine_caught:
                    tilelang.engine.lower(module, target=TB.RKNPU_TARGET_NAME)
                self.assertIn("module-level rknpu declarations",
                              engine_caught.exception.diagnostics[0].message)

    def test_registered_engine_records_copy_launch_and_pipeline_hints(self):
        from rknpu_backend.codegen.pipeline import explain_plan

        prim = _copy_hint(gemm_small, "coalesced_width", 8)
        artifact = tilelang.engine.lower(
            tvm.IRModule({"main": prim}), target=TB.RKNPU_TARGET_NAME)
        plan = TB.plan_from_artifact(artifact)
        hints = TB.last_lowering().facts["hints"]["unapplied_annotations"]
        for stage in ("source", "normalized"):
            rows = hints[stage]
            request = next(row for row in rows
                           if row["key"] == "coalesced_width")
            self.assertEqual(request["requested"], 8)
            self.assertFalse(request["applied"])
            self.assertTrue(request["reason"])
            self.assertTrue(any(row["key"] == "tl.launch_threads" for row in rows))
            self.assertTrue(any(row["key"] == "num_stages" for row in rows))
        explanation = json.dumps(explain_plan(plan), sort_keys=True)
        self.assertIn("coalesced_width", explanation)
        self.assertIn("RKNPU copy has no GPU vector coalescing width", explanation)

    def test_registered_engine_rejects_unknown_copy_hint(self):
        self.assert_engine_rejected(
            _copy_hint(gemm_small, "unsupported_copy_hint", 8),
            "annotation has no metadata contract")

    def test_an_elementwise_kernel_is_rejected(self):
        self.assert_rejected(add_only, "no guarded-tail GEMM to prove")

    def test_a_float32_gemm_is_rejected(self):
        self.assert_rejected(gemm_fp32, "the chain only proves rk_f16_tiled")

    def test_the_pilot_program_is_accepted(self):
        TB.clear_lowerings()
        self.context.lower(tvm.IRModule({"main": gemm_small}))
        self.assertEqual(len(TB.lowerings()), 1)
        self.assertEqual(TB.last_lowering().plan_hash,
                         R.compile_guarded_tail(
                             gemm_small, profile=default_profile(),
                             options=options())[0].plan_hash)
