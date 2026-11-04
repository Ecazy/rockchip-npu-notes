# SPDX-License-Identifier: GPL-3.0-or-later
"""The opt-in RKNPU pass chain: proof, IR lowering, instantiation, plan.

Every case here goes through compile_guarded_tail, i.e. through the pass entry,
not through the TileLang extractor and not through the structured prototype.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import tilelang  # noqa: F401  (loads the vendored TVM)
import tilelang.language as T
import tvm
from tvm import tirx
from tilelang import transform as TT

from rknpu_backend import CompileOptions, compile_tilelang, default_profile
from rknpu_backend.errors import BackendError, RK_E_UNSUPPORTED
from rknpu_backend.frontend import rknpu_pass as R
from rknpu_backend import load_plan, save_plan
import tests.test_structured_negatives as negatives

PROFILE = default_profile()
M, K, N, BM, BN = 256, 576, 3072, 256, 256
# The explicit source-hint trace changed this canonical JSON/hash; the frozen
# reference_anchor evidence proves C execution fields, constants and contracts
# still match the historical 70d70281 plan.
REFERENCE_PLAN_WITH_HINT_TRACE = \
    "12590578f133ecfa7307f885e02ee3927a1087413453a92a02e5da8ea1e30742"


def options(**extra):
    base = {"numeric_mode": "rk_f16_tiled", "allow_approx_lut": False}
    base.update(extra)
    return CompileOptions(**base)


@T.prim_func
def gate_up(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"),
            Y: T.Tensor((M, N), "float16")):
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


def materialise(prim_func):
    module = tvm.IRModule({"main": prim_func})
    module = tirx.transform.BindTarget(tvm.target.Target("c"))(module)
    module = TT.MaterializeKernelLaunch(lower_thread_binding=False)(module)
    return module["main"]


def simplify(prim_func):
    return TT.Simplify()(tvm.IRModule({"main": prim_func}))["main"]


def _plan_dict_across_normalisation_forms(plan):
    """Compare full plans while excluding the identity hash and source locations."""
    payload = plan.to_dict()
    payload.pop("plan_hash", None)
    for action in payload.get("actions", ()):
        source_hints = action.get("attrs", {}).get("source_hints", {})
        unapplied = source_hints.get("unapplied_annotations", {})
        for occurrence in unapplied.get("source", ()):
            occurrence.pop("depth", None)
            occurrence.pop("ordinal", None)
    return payload


def with_root_predicate(prim_func, predicate):
    def change(node):
        if type(node).__name__ == "SBlockRealize" and \
                str(node.block.name_hint) == "tilelang_root":
            return tirx.SBlockRealize(node.iter_values, predicate, node.block,
                                      getattr(node, "span", None))
        return None
    return prim_func.with_body(tirx.stmt_functor.ir_transform(
        prim_func.body, None, change))


def with_output_store_predicate(prim_func, predicate):
    def change(node):
        if type(node).__name__ == "BufferStore" and str(node.buffer.name) == "Y":
            return tirx.BufferStore(node.buffer, node.value, node.indices,
                                    predicate, getattr(node, "span", None))
        return None
    return prim_func.with_body(tirx.stmt_functor.ir_transform(
        prim_func.body, None, change))


def with_load_predicate(prim_func, buffer_name, predicate):
    def change(node):
        if type(node).__name__ == "BufferLoad" and \
                str(node.buffer.name) == buffer_name:
            return tirx.BufferLoad(node.buffer, node.indices, predicate,
                                   getattr(node, "span", None))
        return None
    return prim_func.with_body(tirx.stmt_functor.ir_transform(
        prim_func.body, None, change))


def with_effectful_output_index(prim_func):
    def change(node):
        if type(node).__name__ == "BufferStore" and str(node.buffer.name) == "Y":
            call = tirx.call_extern("int32", "audit_effect")
            dead = tirx.Mul(tirx.IntImm("int32", 0), call)
            return tirx.BufferStore(
                node.buffer, node.value,
                [tirx.Add(node.indices[0], dead), node.indices[1]],
                node.predicate)
        return None
    return prim_func.with_body(tirx.stmt_functor.ir_transform(
        prim_func.body, None, change))


def with_parameter_offset(prim_func, name, offset, extra_param=None):
    key, old = next((key, buffer) for key, buffer in prim_func.buffer_map.items()
                    if str(buffer.name) == name)
    view = tirx.decl_buffer(
        old.shape, old.dtype, name=str(old.name), data=old.data,
        strides=old.strides, elem_offset=offset, scope=R._scope_of(old))

    def change(node):
        if type(node).__name__ == "BufferLoad" and node.buffer.same_as(old):
            return tirx.BufferLoad(view, node.indices, node.predicate)
        if type(node).__name__ == "BufferStore" and node.buffer.same_as(old):
            return tirx.BufferStore(view, node.value, node.indices,
                                    node.predicate)
        return None
    body = tirx.stmt_functor.ir_transform(prim_func.body, None, change)
    mapping = dict(prim_func.buffer_map)
    mapping[key] = view
    params = list(prim_func.params)
    if extra_param is not None:
        params.append(extra_param)
    return tirx.PrimFunc(params, body, prim_func.ret_type,
                         mapping, prim_func.attrs)


def with_gemm_arg(prim_func, index, value, extra_param=None):
    def change(node):
        if type(node).__name__ == "Evaluate" and \
                type(node.value).__name__ == "Call" and \
                str(getattr(node.value.op, "name", "")) == "tl.tileop.gemm":
            args = list(node.value.args)
            args[index] = value
            call = tirx.Call(node.value.dtype, node.value.op, args,
                             getattr(node.value, "annotations", None))
            return tirx.Evaluate(call)
        return None
    body = tirx.stmt_functor.ir_transform(prim_func.body, None, change)
    params = list(prim_func.params)
    if extra_param is not None:
        params.append(extra_param)
    return tirx.PrimFunc(params, body, prim_func.ret_type,
                         prim_func.buffer_map, prim_func.attrs)


def with_rknpu_declarations(prim_func, *, function=None, block=None,
                            loop=None, call=None):
    def change(node):
        if type(node).__name__ == "SBlock" and \
                str(node.name_hint) == "tilelang_root" and block:
            annotations = dict(node.annotations)
            annotations.update(block)
            return tirx.SBlock(node.iter_vars, node.reads, node.writes,
                               node.name_hint, node.body, node.init,
                               node.alloc_buffers, node.match_buffers,
                               annotations)
        if type(node).__name__ == "For" and \
                str(node.loop_var) == "ko" and loop:
            annotations = dict(node.annotations)
            annotations.update(loop)
            return tirx.For(node.loop_var, node.min, node.extent, node.kind,
                            node.body, node.thread_binding, annotations,
                            node.step)
        if type(node).__name__ == "Call" and \
                str(getattr(node.op, "name", "")) == "tl.tileop.gemm" and call:
            annotations = dict(getattr(node, "annotations", None) or {})
            for key, value in call.items():
                annotations[key] = (tirx.IntImm("int32", value)
                                    if type(value) is int else
                                    tirx.StringImm(value)
                                    if type(value) is str else value)
            return tirx.Call(node.dtype, node.op, list(node.args), annotations)
        return None

    body = tirx.stmt_functor.ir_transform(prim_func.body, None, change)
    attrs = dict(getattr(prim_func, "attrs", None) or {})
    attrs.update(function or {})
    return prim_func.with_body(body).with_attrs(attrs)


def with_copy_hint(prim_func, key, value):
    def change(node):
        if type(node).__name__ == "Call" and \
                str(getattr(node.op, "name", "")) == "tl.tileop.copy":
            annotations = dict(getattr(node, "annotations", None) or {})
            annotations[key] = tirx.IntImm("int32", value)
            return tirx.Call(node.dtype, node.op, list(node.args), annotations)
        return None
    return prim_func.with_body(tirx.stmt_functor.ir_transform(
        prim_func.body, None, change))


def mutate_root_block(prim_func, mutate):
    def change(node):
        if type(node).__name__ == "SBlockRealize" and \
                str(node.block.name_hint) == "tilelang_root":
            block, values = mutate(node.block, list(node.iter_values))
            return tirx.SBlockRealize(values, node.predicate, block)
        return None
    return prim_func.with_body(tirx.stmt_functor.ir_transform(
        prim_func.body, None, change))


class NormalisationTests(unittest.TestCase):
    def test_all_three_input_forms_produce_the_same_plan(self):
        raw = gate_up
        materialised = materialise(raw)
        folded = simplify(materialised)
        plans = []
        for name, prim in (("raw", raw), ("materialised", materialised),
                           ("simplified", folded)):
            with self.subTest(form=name):
                plan, stages, facts, tasks = R.compile_guarded_tail(
                    prim, profile=PROFILE, options=options())
                plans.append(plan)
                self.assertEqual(len([a for a in plan.actions
                                      if a.kind == "npu_task"]), 36)
                self.assertEqual(len(plan.submits), 12)
                self.assertEqual(facts.to_dict()["tile"], [256, 256, 256])
        self.assertEqual(facts.to_dict()["hints"]["pipeline_hint"]["num_stages"], 2)
        comparable = [_plan_dict_across_normalisation_forms(plan)
                      for plan in plans]
        self.assertEqual(comparable[1:], [comparable[0], comparable[0]])


    def test_the_stages_report_what_they_rewrote(self):
        plan, stages, facts, tasks = R.compile_guarded_tail(
            gate_up, profile=PROFILE, options=options())
        kinds = [(stage.name, stage.kind) for stage in stages]
        self.assertEqual(kinds, [("bind-target", "transform"),
                                 ("materialize-kernel-launch", "transform"),
                                 ("localise", "transform"),
                                 ("instantiate", "transform"),
                                 ("plan", "analysis")])
        gemms = [task for task in tasks if task["kind"] == "gemm"]
        self.assertEqual(len(gemms), 36)
        self.assertEqual(sorted({(task["m0"], task["n0"]) for task in gemms}),
                         [(0, n0) for n0 in range(0, N, BN)])
        self.assertEqual(sorted({task["k0"] for task in gemms}), [0, 256, 512])
        self.assertEqual({task["kt"] for task in gemms}, {256})

    def test_the_instantiated_program_is_a_task_sequence(self):
        _, stages, _, tasks = R.compile_guarded_tail(
            gate_up, profile=PROFILE, options=options())
        self.assertGreater(len(tasks), 36)  # fills and copies are kept too
        self.assertEqual(tasks[0]["kind"], "fill")


class ContainerExecutionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference_plan = R.compile_guarded_tail(
            gate_up, profile=PROFILE, options=options())[0]

    def _compile(self, prim, *, simplify_ir=False):
        return R.compile_guarded_tail(prim, profile=PROFILE, options=options(),
                                      simplify=simplify_ir)

    def test_constant_true_block_predicate_is_accepted_before_and_after_normalise(self):
        for source in (gate_up, materialise(gate_up), simplify(materialise(gate_up))):
            with self.subTest(source=type(source).__name__):
                predicated = with_root_predicate(
                    source, tirx.IntImm("bool", 1))
                plan, _, _, _ = self._compile(predicated)
                self.assertEqual(_plan_dict_across_normalisation_forms(plan),
                                 _plan_dict_across_normalisation_forms(
                                     self.reference_plan))

    def test_domain_proven_true_block_predicate_is_accepted(self):
        _, grid_n, _ = R._grid_and_k_loops(gate_up)
        predicate = tirx.LT(grid_n.loop_var,
                            tirx.IntImm(grid_n.loop_var.dtype, int(grid_n.extent)))
        prim = with_root_predicate(gate_up, predicate)
        for source in (prim, materialise(prim), simplify(materialise(prim))):
            with self.subTest(source=type(source).__name__):
                plan = self._compile(source)[0]
                self.assertEqual(_plan_dict_across_normalisation_forms(plan),
                                 _plan_dict_across_normalisation_forms(
                                     self.reference_plan))

    def test_false_selective_and_unproven_block_predicates_are_refused(self):
        _, grid_n, _ = R._grid_and_k_loops(gate_up)
        zero = tirx.IntImm(grid_n.loop_var.dtype, 0)
        one = tirx.IntImm(grid_n.loop_var.dtype, 1)
        input_a = next(buffer for buffer in gate_up.buffer_map.values()
                       if str(buffer.name) == "A")
        data_dependent = tirx.GT(
            tirx.BufferLoad(input_a, [tirx.IntImm("int32", 0),
                                      tirx.IntImm("int32", 0)]),
            tirx.FloatImm("float16", 0.0))
        cases = (
            (tirx.IntImm("bool", 0), "SBlockRealize predicate"),
            (tirx.LT(grid_n.loop_var, one), "SBlockRealize predicate"),
            (data_dependent, "SBlockRealize predicate"),
        )
        for predicate, message in cases:
            prim = with_root_predicate(gate_up, predicate)
            for source in (prim, simplify(materialise(prim))):
                with self.subTest(predicate=str(predicate), source=type(source).__name__):
                    with self.assertRaises(BackendError) as caught:
                        self._compile(source)
                    self.assertIn(message, caught.exception.diagnostics[0].message)

    def test_store_and_load_masks_must_be_provably_true(self):
        false = tirx.IntImm("bool", 0)
        for prim, message in (
                (with_output_store_predicate(gate_up, false), "BufferStore predicate"),
                (with_load_predicate(gate_up, "A", false), "BufferLoad predicate")):
            with self.subTest(message=message):
                with self.assertRaises(BackendError) as caught:
                    self._compile(prim)
                self.assertIn(message, caught.exception.diagnostics[0].message)

    def test_effectful_calls_in_indices_or_values_are_not_erased(self):
        index_case = with_effectful_output_index(gate_up)

        def effectful_value(node):
            if type(node).__name__ == "BufferStore" and \
                    str(node.buffer.name) == "Y":
                dead = tirx.Mul(tirx.FloatImm("float16", 0.0),
                                tirx.call_extern("float16", "audit_effect"))
                return tirx.BufferStore(node.buffer,
                                        tirx.Add(node.value, dead),
                                        node.indices, node.predicate)
            return None
        value_case = gate_up.with_body(tirx.stmt_functor.ir_transform(
            gate_up.body, None, effectful_value))
        for prim in (index_case, value_case):
            with self.subTest(source=prim.script()[-160:]):
                with self.assertRaises(BackendError) as caught:
                    self._compile(prim)
                self.assertIn("call expression", caught.exception.diagnostics[0].message)

    def test_parameter_offsets_are_rejected_and_equivalent_zero_is_accepted(self):
        for name in ("A", "B", "Y"):
            with self.subTest(buffer=name):
                prim = with_parameter_offset(gate_up, name, 576)
                with self.assertRaises(BackendError) as caught:
                    self._compile(prim)
                self.assertIn("nonzero element offset",
                              caught.exception.diagnostics[0].message)
        unknown = tirx.Var("parameter_offset", "int32")
        with self.assertRaises(BackendError) as caught:
            self._compile(with_parameter_offset(
                gate_up, "A", unknown, extra_param=unknown))
        self.assertIn("element offset is not statically known",
                      caught.exception.diagnostics[0].message)
        zero = tirx.Add(tirx.IntImm("int32", 0), tirx.IntImm("int32", 0))
        prim = with_parameter_offset(gate_up, "A", zero)
        self.assertEqual(self._compile(prim)[0].plan_hash,
                         self.reference_plan.plan_hash)

    def test_allocated_buffer_offset_is_rejected(self):
        base = next(buffer for _, node in R._walk(gate_up.body)
                    if type(node).__name__ == "SBlock"
                    for buffer in node.alloc_buffers
                    if str(buffer.name) == "As")
        shifted = tirx.decl_buffer(
            base.shape, base.dtype, name=str(base.name), data=base.data,
            strides=base.strides, elem_offset=1, scope=R._scope_of(base))

        def shift_references_and_declaration(node):
            if type(node).__name__ == "BufferLoad" and node.buffer.same_as(base):
                return tirx.BufferLoad(shifted, node.indices, node.predicate)
            if type(node).__name__ == "BufferStore" and node.buffer.same_as(base):
                return tirx.BufferStore(shifted, node.value, node.indices,
                                        node.predicate)
            if type(node).__name__ == "SBlock" and \
                    any(buffer.same_as(base) for buffer in node.alloc_buffers):
                allocs = [shifted if buffer.same_as(base) else buffer
                          for buffer in node.alloc_buffers]
                return tirx.SBlock(
                    node.iter_vars, node.reads, node.writes, node.name_hint,
                    node.body, node.init, allocs, node.match_buffers,
                    node.annotations)
            return None
        body = tirx.stmt_functor.ir_transform(
            gate_up.body, None, shift_references_and_declaration)
        with self.assertRaises(BackendError) as caught:
            self._compile(gate_up.with_body(body))
        self.assertIn("nonzero element offset",
                      caught.exception.diagnostics[0].message)

    def test_clear_accum_literal_simplified_unknown_and_equivalent_flags(self):
        false_expr = tirx.EQ(tirx.IntImm("int32", 0), tirx.IntImm("int32", 1))
        true_expr = tirx.EQ(tirx.IntImm("int32", 0), tirx.IntImm("int32", 0))
        unknown = tirx.Var("clear_flag", "bool")
        equivalent_false = with_gemm_arg(gate_up, 9, false_expr)
        for source in (equivalent_false,
                       simplify(materialise(equivalent_false))):
            plan = self._compile(source)[0]
            self.assertEqual(_plan_dict_across_normalisation_forms(plan),
                             _plan_dict_across_normalisation_forms(
                                 self.reference_plan))
        for source in (with_gemm_arg(gate_up, 9, true_expr),
                       with_gemm_arg(gate_up, 9, unknown, extra_param=unknown)):
            for simplify_ir in (False, True):
                with self.subTest(simplify=simplify_ir,
                                  clear=str(source.script()[-320:])):
                    with self.assertRaises(BackendError) as caught:
                        self._compile(source, simplify_ir=simplify_ir)
                    self.assertTrue(any(term in caught.exception.diagnostics[0].message
                                        for term in ("clear_accum is not supported",
                                                     "clear_accum must be statically known")))

    def test_transpose_flags_accept_equivalent_false_and_reject_unknown(self):
        false_expr = tirx.EQ(tirx.IntImm("int32", 0), tirx.IntImm("int32", 1))
        transposed = with_gemm_arg(gate_up, 3, false_expr)
        self.assertEqual(self._compile(transposed)[0].plan_hash,
                         self.reference_plan.plan_hash)
        unknown = tirx.Var("transpose_flag", "bool")
        for simplify_ir in (False, True):
            prim = with_gemm_arg(gate_up, 3, unknown, extra_param=unknown)
            with self.subTest(simplify=simplify_ir):
                with self.assertRaises(BackendError) as caught:
                    self._compile(prim, simplify_ir=simplify_ir)
                self.assertIn("transpose flags must be statically known",
                              caught.exception.diagnostics[0].message)

    def test_true_store_mask_preserves_the_reference_plan(self):
        prim = with_output_store_predicate(gate_up, tirx.IntImm("bool", 1))
        self.assertEqual(self._compile(prim)[0].plan_hash,
                         self.reference_plan.plan_hash)

    def test_block_init_and_iteration_bindings_are_refused(self):
        def add_init(block, values):
            return tirx.SBlock(
                block.iter_vars, block.reads, block.writes, block.name_hint,
                block.body, tirx.Evaluate(tirx.IntImm("int32", 1)),
                block.alloc_buffers, block.match_buffers, block.annotations), values

        prim = mutate_root_block(gate_up, add_init)
        with self.assertRaises(BackendError) as caught:
            self._compile(prim)
        self.assertIn("SBlock.init", caught.exception.diagnostics[0].message)

        variable = tirx.Var("direct_block_iter", "int32")
        iv = tirx.IterVar(tvm.ir.Range.from_min_extent(0, 1), variable, 0)
        direct_block = tirx.SBlock([iv], [], [], "direct_block", gate_up.body)
        with self.assertRaises(BackendError) as caught:
            self._compile(gate_up.with_body(direct_block))
        self.assertIn("SBlock.iter_vars",
                      caught.exception.diagnostics[0].message)

        def add_iter_var_and_realize_value(block, values):
            variable = tirx.Var("realized_block_iter", "int32")
            iv = tirx.IterVar(tvm.ir.Range.from_min_extent(0, 1), variable, 0)
            updated = tirx.SBlock(
                [iv], block.reads, block.writes, block.name_hint, block.body,
                block.init, block.alloc_buffers, block.match_buffers,
                block.annotations)
            return updated, [tirx.IntImm("int32", 0)]
        prim = mutate_root_block(gate_up, add_iter_var_and_realize_value)
        with self.assertRaises(BackendError) as caught:
            self._compile(prim)
        self.assertIn("SBlockRealize iteration bindings",
                      caught.exception.diagnostics[0].message)

    def test_alias_annotations_and_unknown_statements_are_refused(self):
        def add_alias(block, values):
            base = block.alloc_buffers[0]
            alias = tirx.decl_buffer(base.shape, base.dtype, name="alias",
                                     data=base.data, scope=R._scope_of(base))
            region = tirx.BufferRegion(
                base, [tvm.ir.Range.from_min_extent(0, extent)
                       for extent in base.shape])
            match = tirx.MatchBufferRegion(alias, region)
            updated = tirx.SBlock(
                block.iter_vars, block.reads, block.writes, block.name_hint,
                block.body, block.init, block.alloc_buffers, [match],
                block.annotations)
            return updated, values

        alias_prim = mutate_root_block(gate_up, add_alias)
        with self.assertRaises(BackendError) as caught:
            self._compile(alias_prim)
        self.assertIn("match_buffers", caught.exception.diagnostics[0].message)

        def add_duplicate_storage(block, values):
            base = block.alloc_buffers[0]
            alias = tirx.decl_buffer(
                base.shape, base.dtype, name="aliased_allocation",
                data=base.data, strides=base.strides,
                elem_offset=base.elem_offset, scope=R._scope_of(base))
            updated = tirx.SBlock(
                block.iter_vars, block.reads, block.writes, block.name_hint,
                block.body, block.init, [*block.alloc_buffers, alias],
                block.match_buffers, block.annotations)
            return updated, values
        with self.assertRaises(BackendError) as caught:
            self._compile(mutate_root_block(gate_up, add_duplicate_storage))
        self.assertIn("share one data variable",
                      caught.exception.diagnostics[0].message)

        def unknown_annotation(block, values):
            annotations = dict(block.annotations)
            annotations["tl.unverified_execution_hint"] = tirx.IntImm("int32", 1)
            return tirx.SBlock(block.iter_vars, block.reads, block.writes,
                               block.name_hint, block.body, block.init,
                               block.alloc_buffers, block.match_buffers,
                               annotations), values
        with self.assertRaises(BackendError) as caught:
            self._compile(mutate_root_block(gate_up, unknown_annotation))
        self.assertIn("annotation has no metadata contract",
                      caught.exception.diagnostics[0].message)

    def test_else_domain_uses_not_of_the_complete_condition(self):
        a, b = tirx.Var("a", "bool"), tirx.Var("b", "bool")
        condition = tirx.And(a, b)
        prim = with_root_predicate(gate_up, tirx.Not(a))
        body = tirx.IfThenElse(condition, tirx.Evaluate(tirx.IntImm("int32", 0)),
                               prim.body)
        prim = tirx.PrimFunc([*prim.params, a, b], body, prim.ret_type,
                             prim.buffer_map, prim.attrs)
        with self.assertRaises(BackendError) as caught:
            self._compile(prim)
        self.assertIn("SBlockRealize predicate",
                      caught.exception.diagnostics[0].message)

    def test_unhandled_bind_allocation_and_attribute_statements_are_refused(self):
        body = gate_up.body
        bad_bind = tirx.Bind(tirx.Var("unhandled_bind", "int32"),
                             tirx.IntImm("int32", 1))
        prim = gate_up.with_body(tirx.SeqStmt([bad_bind, body]))
        cases = [(prim, "Bind is not a recognized")]

        allocation = next(buffer for _, node in R._walk(body)
                          for buffer in (getattr(node, "alloc_buffers", ()) or ()))
        alloc_stmt = tirx.AllocBuffer(allocation)
        cases.append((gate_up.with_body(tirx.SeqStmt([alloc_stmt, body])),
                      "AllocBuffer carries storage"))
        decl_stmt = tirx.DeclBuffer(allocation)
        cases.append((gate_up.with_body(tirx.SeqStmt([decl_stmt, body])),
                      "DeclBuffer carries storage"))
        attr_stmt = tirx.AttrStmt(tirx.Var("attr_node", "int32"),
                                  "unverified_scope", tirx.IntImm("int32", 1), body)
        cases.append((gate_up.with_body(attr_stmt),
                      "AttrStmt carries storage/control"))
        for prim, phrase in cases:
            with self.subTest(phrase=phrase):
                with self.assertRaises(BackendError) as caught:
                    self._compile(prim)
                self.assertIn(phrase, caught.exception.diagnostics[0].message)

    def test_launch_thread_binding_is_refused_when_kernel_uses_it(self):
        bind_vars = []
        tirx.stmt_functor.post_order_visit(
            gate_up.body,
            lambda node: bind_vars.append(node.var)
            if type(node).__name__ == "Bind" else None)
        self.assertTrue(bind_vars)
        thread_var = bind_vars[0]

        def replace_output_index(node):
            if type(node).__name__ == "BufferStore" and \
                    str(node.buffer.name) == "Y":
                return tirx.BufferStore(node.buffer, node.value,
                                        [thread_var, *node.indices[1:]],
                                        getattr(node, "predicate", None))
            return None

        prim = gate_up.with_body(tirx.stmt_functor.ir_transform(
            gate_up.body, None, replace_output_index))
        with self.assertRaises(BackendError) as caught:
            self._compile(prim)
        self.assertIn("launch-thread binding is referenced",
                      caught.exception.diagnostics[0].message)


class DeclarationContractTests(unittest.TestCase):
    def test_each_supported_scope_and_matching_duplicate_is_consumed_and_saved(self):
        prim = with_rknpu_declarations(
            gate_up,
            function={"rknpu.schedule": "mn_tiles_v1"},
            block={"rknpu.tile_m": 256},
            loop={"rknpu.tile_m": 256},
            call={"rknpu.tile_n": 256, "rknpu.k_tile": 256})
        plan, _, facts, _ = R.compile_guarded_tail(
            prim, profile=PROFILE, options=options())
        trace = facts.to_dict()["hints"]["rknpu_declarations"]
        for stage in ("source", "normalized"):
            records = trace[stage]
            self.assertEqual(len(records), 5)
            self.assertTrue(all(record["consumed"] for record in records))
            self.assertEqual(sum(record["key"] == "rknpu.tile_m"
                                 for record in records), 2)
        self.assertEqual({record["scope"] for record in trace["source"]},
                         {"PrimFunc.attrs", "SBlock:tilelang_root",
                          "For:ko", "Call:tl.tileop.gemm"})
        serialized = json.dumps(plan.to_dict(), sort_keys=True)
        self.assertIn("rknpu_declarations", serialized)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "declarations.rkplan.json"
            save_plan(plan, path)
            loaded = load_plan(path)
        self.assertEqual(loaded.to_dict(), plan.to_dict())
        self.assertIn("rknpu_declarations", json.dumps(loaded.to_dict()))

    def test_conflicting_scope_values_are_rejected_in_both_orders(self):
        cases = (
            (128, 256, "SBlock:tilelang_root"),
            (256, 128, "For:ko"),
        )
        for outer, inner, scope in cases:
            prim = with_rknpu_declarations(
                gate_up, block={"rknpu.tile_m": outer},
                loop={"rknpu.tile_m": inner})
            with self.subTest(outer=outer, inner=inner):
                with self.assertRaises(BackendError) as caught:
                    R.compile_guarded_tail(prim, profile=PROFILE,
                                           options=options())
                diagnostic = caught.exception.diagnostics[0]
                self.assertEqual(diagnostic.code, RK_E_UNSUPPORTED)
                self.assertIn("declaration disagrees", diagnostic.message)
                self.assertEqual(diagnostic.to_dict()["actual"]["scope"], scope)

    def test_unknown_keys_and_wrong_value_types_are_rejected(self):
        cases = (
            ({"rknpu.unimplemented": 1}, "not supported by the pass chain"),
            ({"rknpu.tile_m": "256"}, "wrong value type"),
        )
        for annotations, phrase in cases:
            with self.subTest(annotations=annotations):
                prim = with_rknpu_declarations(gate_up, block=annotations)
                with self.assertRaises(BackendError) as caught:
                    R.compile_guarded_tail(prim, profile=PROFILE,
                                           options=options())
                diagnostic = caught.exception.diagnostics[0]
                self.assertEqual(diagnostic.code, RK_E_UNSUPPORTED)
                self.assertIn(phrase, diagnostic.message)

    def test_copy_hint_and_existing_launch_hints_are_saved_and_explained(self):
        from rknpu_backend.codegen.pipeline import explain_plan

        prim = with_copy_hint(gate_up, "coalesced_width", 8)
        plan, _, facts, _ = R.compile_guarded_tail(
            prim, profile=PROFILE, options=options())
        unapplied = facts.to_dict()["hints"]["unapplied_annotations"]
        for stage in ("source", "normalized"):
            rows = unapplied[stage]
            copy_hint = next(row for row in rows
                             if row["key"] == "coalesced_width")
            self.assertEqual(copy_hint["requested"], 8)
            self.assertFalse(copy_hint["applied"])
            self.assertTrue(copy_hint["reason"])
            self.assertTrue(any(row["key"] == "tl.launch_threads" for row in rows))
            self.assertTrue(any(row["key"] == "num_stages" for row in rows))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "copy-hint.rkplan.json"
            save_plan(plan, path)
            restored = load_plan(path)
        explanation = explain_plan(restored)
        self.assertIn("coalesced_width",
                      json.dumps(explanation["schedule"]["source_hints"]))
        self.assertIn("RKNPU copy has no GPU vector coalescing width",
                      json.dumps(explanation["schedule"]["source_hints"]))

class PlanParityTests(unittest.TestCase):
    def test_the_plan_matches_the_scheduled_baseline_builder(self):
        old = compile_tilelang(
            gate_up, profile=PROFILE,
            options=options(linear_task_order="scheduled_baseline")).plan
        new, _, _, _ = R.compile_guarded_tail(gate_up, profile=PROFILE,
                                              options=options())

        def signature(plan):
            tasks = [a for a in plan.actions if a.kind == "npu_task"]
            return {
                "tasks": [(a.template_id, tuple(a.inputs), tuple(a.outputs),
                           a.attrs.get("k_index"), a.attrs.get("k0"),
                           a.attrs.get("k_count"),
                           a.attrs.get("submit_group")) for a in tasks],
                "submits": [(len(s.task_ids), s.recipe_id, s.mode)
                            for s in plan.submits],
                "publish": [(p.dst_row, p.dst_col, p.rows, p.cols)
                            for p in plan.publish],
                "device": plan.accounting.device_request_bytes,
                "host": plan.accounting.host_bytes,
                "bo": plan.accounting.data_bo_count,
                "ops": len(plan.logical_ops),
                "reqs": len(plan.numeric_requirements),
            }

        self.assertEqual(signature(new), signature(old))

    def test_the_plan_saves_and_reloads(self):
        plan, _, _, _ = R.compile_guarded_tail(gate_up, profile=PROFILE,
                                               options=options())
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "plan.rkplan.json")
            save_plan(plan, path)
            loaded = load_plan(path)
        self.assertEqual(loaded.plan_hash, plan.plan_hash)
        self.assertEqual(len([a for a in loaded.actions
                              if a.kind == "npu_task"]), 36)


class GeometryTests(unittest.TestCase):
    def test_a_non_square_tile_is_supported(self):
        plan, _, facts, _ = R.compile_guarded_tail(
            negatives.non_square, profile=PROFILE, options=options())
        self.assertEqual(facts.to_dict()["tile"], [128, 256, 256])
        # two M tiles times three K slices
        self.assertEqual(len([a for a in plan.actions
                              if a.kind == "npu_task"]), 6)

    def test_a_declaration_order_swap_is_supported(self):
        plan, _, facts, _ = R.compile_guarded_tail(
            negatives.bs_first, profile=PROFILE, options=options())
        self.assertEqual(facts.to_dict()["tile"], [128, 256, 256])

    def test_reordered_output_guards_are_supported(self):
        plan, _, facts, _ = R.compile_guarded_tail(
            negatives.reordered_guard, profile=PROFILE, options=options())
        self.assertEqual(facts.to_dict()["logical"], [256, 576, 128])

    def test_an_exactly_covered_unguarded_epilogue_is_supported(self):
        R.compile_guarded_tail(negatives.unguarded_epilogue, profile=PROFILE,
                               options=options())


class NegativeTests(unittest.TestCase):
    # loose_guard and guard_on_loop_vars left this list: their extra conditions
    # are provably true over the iteration domain, so they are the reference
    # program and are checked for plan identity in RedundantConditionTests.
    CASES = ("clear_after_loop", "epilogue_first", "gemm_first",
             "else_branch", "extra_output_store", "write_to_z",
             "sibling_shared_store", "copy_value_plus_one",
             "copy_value_twice", "output_value_doubled",
             "partial_operand_copy", "partial_epilogue", "local_sibling_store",
             "store_into_a", "swapped_axes_square", "root_else_branch",
             "extra_k_trip")

    def test_every_semantic_mutation_is_refused(self):
        for name in self.CASES:
            with self.subTest(kernel=name):
                kernel = getattr(negatives, name)
                with self.assertRaises(BackendError):
                    R.compile_guarded_tail(kernel, profile=PROFILE,
                                           options=options())




M2, K2, N2 = 256, 576, 256


@T.prim_func
def transpose_b(A: T.Tensor((M2, K2), "float16"),
                B: T.Tensor((K2, N2), "float16"),
                Y: T.Tensor((M2, N2), "float16")):
    with T.Kernel(T.ceildiv(M2, 256), T.ceildiv(N2, 256), threads=128) as (im, inn):
        As = T.alloc_shared((256, 256), "float16")
        Bs = T.alloc_shared((256, 256), "float16")
        Cl = T.alloc_fragment((256, 256), "float32")
        Acc = T.alloc_fragment((256, 256), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(T.ceildiv(K2, 256), num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(256, 256):
                if im * 256 + i < M2 and ko * 256 + j < K2:
                    As[i, j] = A[im * 256 + i, ko * 256 + j]
            for i, j in T.Parallel(256, 256):
                if ko * 256 + i < K2 and inn * 256 + j < N2:
                    Bs[i, j] = B[ko * 256 + i, inn * 256 + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl, transpose_B=True)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(256, 256):
            if im * 256 + i < M2 and inn * 256 + j < N2:
                Y[im * 256 + i, inn * 256 + j] = Acc[i, j]


@T.prim_func
def reversed_k(A: T.Tensor((M2, K2), "float16"),
               B: T.Tensor((K2, N2), "float16"),
               Y: T.Tensor((M2, N2), "float16")):
    with T.Kernel(T.ceildiv(M2, 256), T.ceildiv(N2, 256), threads=128) as (im, inn):
        As = T.alloc_shared((256, 256), "float16")
        Bs = T.alloc_shared((256, 256), "float16")
        Cl = T.alloc_fragment((256, 256), "float32")
        Acc = T.alloc_fragment((256, 256), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(T.ceildiv(K2, 256), num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(256, 256):
                if im * 256 + i < M2 and (2 - ko) * 256 + j < K2:
                    As[i, j] = A[im * 256 + i, (2 - ko) * 256 + j]
            for i, j in T.Parallel(256, 256):
                if (2 - ko) * 256 + i < K2 and inn * 256 + j < N2:
                    Bs[i, j] = B[(2 - ko) * 256 + i, inn * 256 + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(256, 256):
            if im * 256 + i < M2 and inn * 256 + j < N2:
                Y[im * 256 + i, inn * 256 + j] = Acc[i, j]


N3 = 512


@T.prim_func
def pinned_weight(A: T.Tensor((M2, K2), "float16"),
                  B: T.Tensor((K2, N3), "float16"),
                  Y: T.Tensor((M2, N3), "float16")):
    with T.Kernel(T.ceildiv(M2, 256), T.ceildiv(N3, 256), threads=128) as (im, inn):
        As = T.alloc_shared((256, 256), "float16")
        Bs = T.alloc_shared((256, 256), "float16")
        Cl = T.alloc_fragment((256, 256), "float32")
        Acc = T.alloc_fragment((256, 256), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(T.ceildiv(K2, 256), num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(256, 256):
                if im * 256 + i < M2 and ko * 256 + j < K2:
                    As[i, j] = A[im * 256 + i, ko * 256 + j]
            for i, j in T.Parallel(256, 256):
                if ko * 256 + i < K2 and j < N3:
                    Bs[i, j] = B[ko * 256 + i, j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(256, 256):
            if im * 256 + i < M2 and inn * 256 + j < N3:
                Y[im * 256 + i, inn * 256 + j] = Acc[i, j]


class ReviewMutationTests(unittest.TestCase):
    """The three mutations an independent review found accepted, plus threads."""

    def _rejects(self, kernel, needle):
        with self.assertRaises(BackendError) as caught:
            R.compile_guarded_tail(kernel, profile=PROFILE, options=options())
        self.assertIn(needle, caught.exception.diagnostics[0].message)

    def test_a_transposed_gemm_operand_is_refused(self):
        self._rejects(transpose_b, "transposition is not supported")

    def test_a_reversed_k_accumulation_is_refused(self):
        self._rejects(reversed_k, "not tile_extent * tile + local_index")

    def test_a_weight_tile_not_bound_to_the_block_index_is_refused(self):
        self._rejects(pinned_weight, "block coordinate folded away")

    def test_gpu_thread_extents_are_recorded_and_not_applied(self):
        _, _, facts, _ = R.compile_guarded_tail(gate_up, profile=PROFILE,
                                                options=options())
        self.assertEqual(facts.to_dict()["hints"]["ignored_launch_threads"],
                         [128, 1, 1])

    def test_the_task_ir_states_its_parameters_and_k_tile(self):
        _, _, _, tasks = R.compile_guarded_tail(gate_up, profile=PROFILE,
                                                options=options())
        gemm = [task for task in tasks if task["kind"] == "gemm"][0]
        self.assertEqual(gemm["operands"], ["A", "B", "Y"])
        self.assertEqual(gemm["tiles"], ["As", "Bs", "Cl"])
        plan, _, _, _ = R.compile_guarded_tail(gate_up, profile=PROFILE,
                                               options=options())
        task = [a for a in plan.actions if a.kind == "npu_task"][0]
        self.assertEqual(task.attrs.get("kt"), 256)






M4, K4, N4 = 512, 576, 256


@T.prim_func
def folded_block_index(A: T.Tensor((M4, K4), "float16"),
                       B: T.Tensor((K4, N4), "float16"),
                       Y: T.Tensor((M4, N4), "float16")):
    """Two M blocks, but the operand copy drops the block index."""
    with T.Kernel(T.ceildiv(M4, 256), T.ceildiv(N4, 256), threads=128) as (im, inn):
        As = T.alloc_shared((256, 256), "float16")
        Bs = T.alloc_shared((256, 256), "float16")
        Cl = T.alloc_fragment((256, 256), "float32")
        Acc = T.alloc_fragment((256, 256), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(T.ceildiv(K4, 256), num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(256, 256):
                if i < M4 and ko * 256 + j < K4:
                    As[i, j] = A[i, ko * 256 + j]
            for i, j in T.Parallel(256, 256):
                if ko * 256 + i < K4 and inn * 256 + j < N4:
                    Bs[i, j] = B[ko * 256 + i, inn * 256 + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(256, 256):
            if im * 256 + i < M4 and inn * 256 + j < N4:
                Y[im * 256 + i, inn * 256 + j] = Acc[i, j]


@T.prim_func
def half_k_tile(A: T.Tensor((256, K4), "float16"),
                B: T.Tensor((K4, N4), "float16"),
                Y: T.Tensor((256, N4), "float16")):
    """A legal kernel whose K tile is 128, not the planner's 256."""
    with T.Kernel(1, T.ceildiv(N4, 256), threads=128) as (im, inn):
        As = T.alloc_shared((256, 128), "float16")
        Bs = T.alloc_shared((128, 256), "float16")
        Cl = T.alloc_fragment((256, 256), "float32")
        Acc = T.alloc_fragment((256, 256), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(T.ceildiv(K4, 128), num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(256, 128):
                if i < 256 and ko * 128 + j < K4:
                    As[i, j] = A[i, ko * 128 + j]
            for i, j in T.Parallel(128, 256):
                if ko * 128 + i < K4 and inn * 256 + j < N4:
                    Bs[i, j] = B[ko * 128 + i, inn * 256 + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(256, 256):
            if i < 256 and inn * 256 + j < N4:
                Y[i, inn * 256 + j] = Acc[i, j]


class GateTests(unittest.TestCase):
    def test_a_folded_block_index_is_refused(self):
        with self.assertRaises(BackendError) as caught:
            R.compile_guarded_tail(folded_block_index, profile=PROFILE,
                                   options=options())
        self.assertIn("block coordinate folded away",
                      caught.exception.diagnostics[0].message)

    def test_a_non_256_k_tile_is_refused_by_the_plan(self):
        with self.assertRaises(BackendError):
            R.compile_guarded_tail(half_k_tile, profile=PROFILE,
                                   options=options())

    def test_the_task_budget_is_enforced(self):
        with self.assertRaises(BackendError) as caught:
            R.compile_guarded_tail(gate_up, profile=PROFILE,
                                   options=options(max_actions=5))
        self.assertIn("budget", caught.exception.diagnostics[0].message)

    def test_an_unsupported_numeric_mode_is_refused(self):
        with self.assertRaises(BackendError):
            R.compile_guarded_tail(gate_up, profile=PROFILE,
                                   options=options(numeric_mode="fp32_ref"))




import tilelang.language as T

MR, KR, NR = 256, 576, 256


@T.prim_func
def row_zero_only(A: T.Tensor((MR, KR), "float16"), B: T.Tensor((KR, NR), "float16"),
                  Y: T.Tensor((MR, NR), "float16")):
    """Full tile loop, but the A copy writes only row 0."""
    with T.Kernel(T.ceildiv(MR, 256), T.ceildiv(NR, 256), threads=128) as (im, inn):
        As = T.alloc_shared((256, 256), "float16")
        Bs = T.alloc_shared((256, 256), "float16")
        Cl = T.alloc_fragment((256, 256), "float32")
        Acc = T.alloc_fragment((256, 256), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(T.ceildiv(KR, 256), num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(256, 256):
                if im * 256 + i < MR and i < 1 and ko * 256 + j < KR:
                    As[i, j] = A[im * 256 + i, ko * 256 + j]
            for i, j in T.Parallel(256, 256):
                if ko * 256 + i < KR and inn * 256 + j < NR:
                    Bs[i, j] = B[ko * 256 + i, inn * 256 + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(256, 256):
            if im * 256 + i < MR and inn * 256 + j < NR:
                Y[im * 256 + i, inn * 256 + j] = Acc[i, j]


@T.prim_func
def clearing_gemm(A: T.Tensor((MR, KR), "float16"),
                  B: T.Tensor((KR, NR), "float16"),
                  Y: T.Tensor((MR, NR), "float16")):
    """The GEMM clears its accumulator, so the KR chain must not accumulate."""
    with T.Kernel(T.ceildiv(MR, 256), T.ceildiv(NR, 256), threads=128) as (im, inn):
        As = T.alloc_shared((256, 256), "float16")
        Bs = T.alloc_shared((256, 256), "float16")
        Cl = T.alloc_fragment((256, 256), "float32")
        Acc = T.alloc_fragment((256, 256), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(T.ceildiv(KR, 256), num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(256, 256):
                if im * 256 + i < MR and ko * 256 + j < KR:
                    As[i, j] = A[im * 256 + i, ko * 256 + j]
            for i, j in T.Parallel(256, 256):
                if ko * 256 + i < KR and inn * 256 + j < NR:
                    Bs[i, j] = B[ko * 256 + i, inn * 256 + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl, clear_accum=True)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(256, 256):
            if im * 256 + i < MR and inn * 256 + j < NR:
                Y[im * 256 + i, inn * 256 + j] = Acc[i, j]


@T.prim_func
def k_from_one(A: T.Tensor((MR, KR), "float16"), B: T.Tensor((KR, NR), "float16"),
               Y: T.Tensor((MR, NR), "float16")):
    """The KR loop runs 1..3, but the produced plan assumes 0..2."""
    with T.Kernel(T.ceildiv(MR, 256), T.ceildiv(NR, 256), threads=128) as (im, inn):
        As = T.alloc_shared((256, 256), "float16")
        Bs = T.alloc_shared((256, 256), "float16")
        Cl = T.alloc_fragment((256, 256), "float32")
        Acc = T.alloc_fragment((256, 256), "float16")
        T.clear(Acc)
        for ko in T.serial(1, 4):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(256, 256):
                if im * 256 + i < MR and ko * 256 + j < KR:
                    As[i, j] = A[im * 256 + i, ko * 256 + j]
            for i, j in T.Parallel(256, 256):
                if ko * 256 + i < KR and inn * 256 + j < NR:
                    Bs[i, j] = B[ko * 256 + i, inn * 256 + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(256, 256):
            if im * 256 + i < MR and inn * 256 + j < NR:
                Y[im * 256 + i, inn * 256 + j] = Acc[i, j]


@T.prim_func
def duplicated_output_block(A: T.Tensor((MR, KR), "float16"),
                            B: T.Tensor((KR, NR), "float16"),
                            Y: T.Tensor((MR, NR), "float16")):
    """Two MR blocks, but every block writes the first one."""
    with T.Kernel(T.ceildiv(MR, 128), T.ceildiv(NR, 256), threads=128) as (im, inn):
        As = T.alloc_shared((128, 256), "float16")
        Bs = T.alloc_shared((256, 256), "float16")
        Cl = T.alloc_fragment((128, 256), "float32")
        Acc = T.alloc_fragment((128, 256), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(T.ceildiv(KR, 256), num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(128, 256):
                if im * 128 + i < MR and ko * 256 + j < KR:
                    As[i, j] = A[im * 128 + i, ko * 256 + j]
            for i, j in T.Parallel(256, 256):
                if ko * 256 + i < KR and inn * 256 + j < NR:
                    Bs[i, j] = B[ko * 256 + i, inn * 256 + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(128, 256):
            if i < MR and inn * 256 + j < NR:
                Y[i, inn * 256 + j] = Acc[i, j]


@T.prim_func
def declared_tile_mismatch(A: T.Tensor((MR, KR), "float16"),
                           B: T.Tensor((KR, NR), "float16"),
                           Y: T.Tensor((MR, NR), "float16")):
    """An explicit rknpu schedule that disagrees with the proven tile."""
    with T.Kernel(T.ceildiv(MR, 256), T.ceildiv(NR, 256), threads=128) as (im, inn):
        T.sblock_attr({"rknpu.schedule": "mn_tiles_v1", "rknpu.tile_m": 128,
                      "rknpu.tile_n": 256})
        As = T.alloc_shared((256, 256), "float16")
        Bs = T.alloc_shared((256, 256), "float16")
        Cl = T.alloc_fragment((256, 256), "float32")
        Acc = T.alloc_fragment((256, 256), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(T.ceildiv(KR, 256), num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(256, 256):
                if im * 256 + i < MR and ko * 256 + j < KR:
                    As[i, j] = A[im * 256 + i, ko * 256 + j]
            for i, j in T.Parallel(256, 256):
                if ko * 256 + i < KR and inn * 256 + j < NR:
                    Bs[i, j] = B[ko * 256 + i, inn * 256 + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(256, 256):
            if im * 256 + i < MR and inn * 256 + j < NR:
                Y[im * 256 + i, inn * 256 + j] = Acc[i, j]


@T.prim_func
def unknown_annotation(A: T.Tensor((MR, KR), "float16"),
                       B: T.Tensor((KR, NR), "float16"),
                       Y: T.Tensor((MR, NR), "float16")):
    """An rknpu annotation this path does not implement."""
    with T.Kernel(T.ceildiv(MR, 256), T.ceildiv(NR, 256), threads=128) as (im, inn):
        T.sblock_attr({"rknpu.double_buffer": True})
        As = T.alloc_shared((256, 256), "float16")
        Bs = T.alloc_shared((256, 256), "float16")
        Cl = T.alloc_fragment((256, 256), "float32")
        Acc = T.alloc_fragment((256, 256), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(T.ceildiv(KR, 256), num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(256, 256):
                if im * 256 + i < MR and ko * 256 + j < KR:
                    As[i, j] = A[im * 256 + i, ko * 256 + j]
            for i, j in T.Parallel(256, 256):
                if ko * 256 + i < KR and inn * 256 + j < NR:
                    Bs[i, j] = B[ko * 256 + i, inn * 256 + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(256, 256):
            if im * 256 + i < MR and inn * 256 + j < NR:
                Y[im * 256 + i, inn * 256 + j] = Acc[i, j]

class ReviewerCounterexampleTests(unittest.TestCase):
    """The miscompiles a direct review reproduced against the previous version."""

    def _refuses(self, kernel, needle, **option_overrides):
        with self.assertRaises(BackendError) as caught:
            R.compile_guarded_tail(kernel, profile=PROFILE,
                                   options=options(**option_overrides))
        self.assertIn(needle, caught.exception.diagnostics[0].message)

    def test_a_copy_that_only_writes_one_row_is_refused(self):
        # the extra "i < 1" is neither an axis bound nor redundant, so the copy
        # would cover one row of the tile and leave the rest of the zero fill
        self._refuses(row_zero_only,
                      "a path condition is neither a buffer bound nor redundant")

    def test_a_clearing_gemm_is_refused(self):
        self._refuses(clearing_gemm, "clear_accum is not supported")

    def test_a_k_loop_that_does_not_start_at_zero_is_refused(self):
        self._refuses(k_from_one, "only handles loops that run 0..extent")

    def test_two_blocks_writing_one_output_region_are_refused(self):
        self._refuses(duplicated_output_block, "cannot be folded")

    def test_a_declared_tile_that_disagrees_with_the_ir_is_refused(self):
        self._refuses(declared_tile_mismatch, "disagrees with the proven geometry")

    def test_an_unknown_rknpu_annotation_is_refused(self):
        self._refuses(unknown_annotation, "not supported by the pass chain")

    def test_an_explicit_regrouping_policy_reorders_consumed_tasks(self):
        plan, _, _, _ = R.compile_guarded_tail(
            gate_up, profile=PROFILE,
            options=options(linear_task_order="merged"))
        tasks = [a for a in plan.actions if a.kind == "npu_task"]
        self.assertTrue(tasks)
        self.assertEqual(len(plan.submits), 1)
        self.assertEqual({task.attrs["linear_task_policy"] for task in tasks},
                         {"merged"})

    def test_a_pinned_policy_and_a_pipeline_hint_are_recorded(self):
        _, _, facts, _ = R.compile_guarded_tail(gate_up, profile=PROFILE,
                                                options=options())
        hints = facts.to_dict()["hints"]
        self.assertEqual(hints["pinned_linear_task_order"]["applied"],
                         "scheduled_baseline")
        self.assertEqual(hints["pipeline_hint"]["num_stages"], 2)
        self.assertFalse(hints["pipeline_hint"]["applied"])



# ---------------------------------------------------------------------------
# 65 review round: the block map, the per-axis epilogue proof and the
# conditions that prove redundant over the iteration domain.  The two
# counterexamples are rebuilt from the same factory the audit scripts use, so
# the tests pin the reproducers instead of a re-derived variant.
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402
from tvm.tirx import stmt_functor as sf  # noqa: E402
from tests.test_structured_lowering import _kernel as pilot  # noqa: E402
from rknpu_backend.codegen.pipeline import explain_plan  # noqa: E402

SQUARE, DEEP = 512, 576
TILES_512 = [(0, 0), (0, 256), (256, 0), (256, 256)]


def _swap_operand_blocks(prim_func):
    """The reviewed mutation: exchange im and inn in the two operand copies."""
    grid_m, grid_n, _ = R._grid_and_k_loops(prim_func)

    def swap(node):
        if type(node).__name__ == "IfThenElse":
            store = R._store_inside(node.then_case)
            if store is not None and str(store.buffer.name) in ("As", "Bs"):
                return sf.substitute(node, {grid_m.loop_var: grid_n.loop_var,
                                            grid_n.loop_var: grid_m.loop_var})
        return None

    return prim_func.with_body(sf.ir_transform(prim_func.body, None, swap))


def _drop_output_row_guard(prim_func):
    """The reviewed mutation: keep the column guard, drop the row guard."""
    def drop(node):
        if type(node).__name__ == "IfThenElse":
            store = R._store_inside(node.then_case)
            if store is not None and str(store.buffer.name) == "Y":
                keep = [part for part in R._conjuncts(node.condition)
                        if not R._structurally_equal(part.a, store.indices[0])]
                assert len(keep) == 1
                return tirx.IfThenElse(keep[0], node.then_case,
                                       node.else_case)
        return None

    return prim_func.with_body(sf.ir_transform(prim_func.body, None, drop))


def _shift_grid_origin(prim_func):
    """The first grid loop runs 1..blocks+1 instead of 0..blocks."""
    grid_m, _, _ = R._grid_and_k_loops(prim_func)
    extent = int(grid_m.extent)

    def shift(node):
        if type(node).__name__ == "For" and node.loop_var.same_as(
                grid_m.loop_var):
            return tirx.For(node.loop_var, 1, extent + 1, node.kind, node.body,
                            node.thread_binding, node.annotations)
        return None

    return prim_func.with_body(sf.ir_transform(prim_func.body, None, shift))


@T.prim_func
def short_k_chain(A: T.Tensor((256, 576), "float16"),
                  B: T.Tensor((576, 256), "float16"),
                  Y: T.Tensor((256, 256), "float16")):
    """Two K blocks of a 576 reduction: the chain stops at 512."""
    with T.Kernel(1, 1, threads=128) as (im, inn):
        As = T.alloc_shared((256, 256), "float16")
        Bs = T.alloc_shared((256, 256), "float16")
        Cl = T.alloc_fragment((256, 256), "float32")
        Acc = T.alloc_fragment((256, 256), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(2, num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(256, 256):
                if im * 256 + i < 256 and ko * 256 + j < 576:
                    As[i, j] = A[im * 256 + i, ko * 256 + j]
            for i, j in T.Parallel(256, 256):
                if ko * 256 + i < 576 and inn * 256 + j < 256:
                    Bs[i, j] = B[ko * 256 + i, inn * 256 + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(256, 256):
            if im * 256 + i < 256 and inn * 256 + j < 256:
                Y[im * 256 + i, inn * 256 + j] = Acc[i, j]


def _normalised(shape_m, shape_k, shape_n, **kwargs):
    return R.normalise(pilot(shape_m, shape_k, shape_n, **kwargs))[0]


def _tiled_geometry(plan):
    """The rectangles the plan publishes, the K slices, and the task count."""
    tasks = [a for a in plan.actions if a.kind == "npu_task"]
    return {
        "publish": sorted((p.dst_row, p.dst_col, p.rows, p.cols)
                          for p in plan.publish),
        "k0": sorted({a.attrs.get("k0") for a in tasks}),
        "kt": sorted({a.attrs.get("kt") for a in tasks}),
        "tasks": len(tasks),
        "bo": plan.accounting.data_bo_count,
    }


class BlockMapTests(unittest.TestCase):
    """Operand slices, K blocks and the output block are one computation."""

    def test_a_swapped_operand_block_is_refused(self):
        prim = _swap_operand_blocks(_normalised(SQUARE, DEEP, SQUARE))
        with self.assertRaises(BackendError) as caught:
            R.compile_guarded_tail(prim, profile=PROFILE, options=options())
        self.assertIn("different computations",
                      caught.exception.diagnostics[0].message)

    def test_the_accepted_square_program_publishes_every_block(self):
        plan, _, facts, tasks = R.compile_guarded_tail(
            _normalised(SQUARE, DEEP, SQUARE), profile=PROFILE,
            options=options())
        gemms = [task for task in tasks if task["kind"] == "gemm"]
        self.assertEqual(sorted({(task["m0"], task["n0"]) for task in gemms}),
                         TILES_512)
        self.assertEqual(sorted({task["k0"] for task in gemms}), [0, 256, 512])

    def test_exchanging_the_grid_axes_everywhere_is_supported(self):
        plain = R.compile_guarded_tail(
            _normalised(SQUARE, DEEP, SQUARE), profile=PROFILE,
            options=options())[0]
        swapped = R.compile_guarded_tail(
            _normalised(SQUARE, DEEP, SQUARE, swap_axis=True), profile=PROFILE,
            options=options())[0]
        # the grid positions changed, the computation did not: the plan must
        # publish the same rectangles for the same K slices
        self.assertEqual(_tiled_geometry(swapped), _tiled_geometry(plain))

    def test_a_grid_loop_that_does_not_start_at_zero_is_refused(self):
        prim = _shift_grid_origin(_normalised(SQUARE, DEEP, SQUARE))
        with self.assertRaises(BackendError) as caught:
            R.compile_guarded_tail(prim, profile=PROFILE, options=options())
        self.assertIn("only handles loops that run 0..extent",
                      caught.exception.diagnostics[0].message)

    def test_renaming_every_loop_variable_keeps_the_plan(self):
        base = R.compile_guarded_tail(_normalised(256, 576, 256),
                                      profile=PROFILE, options=options())[0]
        renamed = R.compile_guarded_tail(
            _normalised(256, 576, 256, rename=True), profile=PROFILE,
            options=options())[0]
        self.assertEqual(renamed.plan_hash, base.plan_hash)

    def test_k_257_is_one_full_slice_and_one_one_wide_slice(self):
        plan = R.compile_guarded_tail(_normalised(256, 257, 256),
                                       profile=PROFILE, options=options())[0]
        self.assertEqual(_tiled_geometry(plan)["kt"], [1, 256])

    def test_k_576_is_two_full_slices_and_one_64_wide_slice(self):
        plan = R.compile_guarded_tail(_normalised(256, 576, 256),
                                       profile=PROFILE, options=options())[0]
        self.assertEqual(_tiled_geometry(plan)["kt"], [64, 256])

    def test_a_k_chain_that_stops_short_is_refused(self):
        # every K block in this source is legal on its own; what is missing is
        # the third block, so the reduction would be published incomplete
        with self.assertRaises(BackendError) as caught:
            R.compile_guarded_tail(short_k_chain, profile=PROFILE,
                                   options=options())
        self.assertIn("do not cover the whole reduction",
                      caught.exception.diagnostics[0].message)

    def test_a_tailed_m_tile_publishes_the_clipped_rectangle(self):
        plan = R.compile_guarded_tail(_normalised(257, 576, 256),
                                       profile=PROFILE, options=options())[0]
        self.assertEqual(_tiled_geometry(plan)["publish"],
                         [(0, 0, 256, 256), (256, 0, 1, 256)])


class OutputGuardTests(unittest.TestCase):
    """Every output axis is proved safe and covering on its own domain."""

    def test_an_uncovered_tail_axis_without_its_guard_is_refused(self):
        prim = _drop_output_row_guard(_normalised(257, 576, 256))
        with self.assertRaises(BackendError) as caught:
            R.compile_guarded_tail(prim, profile=PROFILE, options=options())
        self.assertIn("an unguarded epilogue axis needs a grid that covers it",
                      caught.exception.diagnostics[0].message)

    def test_the_guarded_tail_is_accepted_and_clipped(self):
        plan = R.compile_guarded_tail(_normalised(257, 576, 256),
                                       profile=PROFILE, options=options())[0]
        self.assertEqual(_tiled_geometry(plan)["publish"],
                         [(0, 0, 256, 256), (256, 0, 1, 256)])

    def test_a_tail_operand_copy_without_its_guard_is_refused(self):
        with self.assertRaises(BackendError) as caught:
            R.compile_guarded_tail(_normalised(257, 576, 256, guard_a=False),
                                   profile=PROFILE, options=options())
        # an unguarded operand read is not a guarded tile copy at all, and the
        # tail row it would read is outside A
        self.assertIn("the one guarded copy of a tile",
                      caught.exception.diagnostics[0].message)

    def test_a_narrowing_epilogue_condition_is_refused(self):
        def narrow(node):
            if type(node).__name__ == "IfThenElse":
                store = R._store_inside(node.then_case)
                if store is not None and str(store.buffer.name) == "Y":
                    return tirx.IfThenElse(
                        tirx.And(node.condition, tirx.LT(store.indices[0], 200)),
                        node.then_case, node.else_case)
            return None

        prim = _normalised(257, 576, 256)
        with self.assertRaises(BackendError) as caught:
            R.compile_guarded_tail(
                prim.with_body(sf.ir_transform(prim.body, None, narrow)),
                profile=PROFILE, options=options())
        self.assertIn("a path condition is neither a buffer bound nor redundant",
                      caught.exception.diagnostics[0].message)


class RedundantConditionTests(unittest.TestCase):
    """A condition the domain proves true is dropped, not refused.

    Both sources below are the reference kernel with an extra condition that
    holds for every iteration, so they must produce the reference plan exactly.
    """

    def _reference_plan(self):
        return R.compile_guarded_tail(negatives.reference, profile=PROFILE,
                                      options=options())[0]

    def test_a_vacuous_epilogue_guard_produces_the_reference_plan(self):
        plan = R.compile_guarded_tail(negatives.loose_guard, profile=PROFILE,
                                      options=options())[0]
        self.assertEqual(plan.plan_hash, self._reference_plan().plan_hash)

    def test_a_guard_on_the_loop_extents_produces_the_reference_plan(self):
        plan = R.compile_guarded_tail(negatives.guard_on_loop_vars,
                                      profile=PROFILE, options=options())[0]
        self.assertEqual(plan.plan_hash, self._reference_plan().plan_hash)


class ExplainTests(unittest.TestCase):
    def test_explain_restates_the_instantiated_tasks(self):
        plan, stages, facts, tasks = R.compile_guarded_tail(
            gate_up, profile=PROFILE, options=options())
        report = explain_plan(plan)
        self.assertEqual(report["plan_hash"], plan.plan_hash)
        self.assertEqual(report["source_hash"], "rknpu-pass")
        self.assertEqual(len(report["logical"]["ops"]), 1)
        self.assertEqual(report["logical"]["without_tasks"], [])
        self.assertFalse(report["schedule"]["reblocked"])
        self.assertGreater(report["counts"]["tasks"], 0)
        self.assertEqual([stage.name for stage in stages][-1], "plan")

class MockExecutionTests(unittest.TestCase):
    """The independent C mock executes the plan the pass chain produced.

    Every plan below is compiled through compile_guarded_tail; the mock supplies
    the device-side GEMM, so a wrong block map, a missing K slice or an unguarded
    tail shows up as a wrong number rather than as a shape complaint.
    """

    LIB = (Path(__file__).resolve().parents[1].parent / "custom-ops"
           / "build-host-mvp" / "librknpu_plan_mock.so")

    def setUp(self):
        if not self.LIB.is_file():
            self.skipTest("host plan mock library not built")
        from rknpu_backend.models import PlanCall
        from rknpu_backend.runtime.ffi import LibRKPlan
        self._plan_call = PlanCall
        self._lib = LibRKPlan(str(self.LIB))

    def _signature(self, rows):
        return ((np.arange(1, rows + 1) % 8) + 1).astype(np.float16)

    def _execute(self, shape_m, shape_k, shape_n, **kwargs):
        prim = _normalised(shape_m, shape_k, shape_n, **kwargs)
        plan = R.compile_guarded_tail(prim, profile=PROFILE,
                                       options=options())[0]
        a = np.repeat(self._signature(shape_m)[:, None], shape_k, axis=1)
        b = np.ones((shape_k, shape_n), dtype=np.float16)
        out = np.zeros((shape_m, shape_n), dtype=np.float16)
        call = self._plan_call(self._lib, plan, {"B": b}, fd=0)
        try:
            call.run_into([a], [out])
        finally:
            call.close()
        expected = np.repeat((self._signature(shape_m).astype(np.float64) * shape_k
                              )[:, None], shape_n, axis=1)
        return plan, out, expected

    def test_the_square_program_computes_every_block(self):
        plan, out, expected = self._execute(SQUARE, DEEP, SQUARE)
        self.assertEqual(float(np.abs(out.astype(np.float64) - expected).max()),
                         0.0)

    def test_the_exchanged_grid_axes_compute_the_same_output(self):
        _, plain, _ = self._execute(256, DEEP, 256)
        swapped, swapped_out, expected = self._execute(256, DEEP, 256,
                                                       swap_axis=True)
        self.assertEqual(float(np.abs(swapped_out.astype(np.float64)
                                      - expected).max()), 0.0)
        self.assertTrue(np.array_equal(plain, swapped_out))

    def test_the_tailed_program_writes_only_the_published_rows(self):
        plan, out, expected = self._execute(257, DEEP, 256)
        self.assertEqual(float(np.abs(out.astype(np.float64) - expected).max()),
                         0.0)
        self.assertEqual(int(np.count_nonzero(out)), 257 * 256)



# ---------------------------------------------------------------------------
# 68 review round: a tile store must cover its tile, and the parameter interface
# must be the one the plan binds.  Both counterexamples are the reviewer's own
# sources, retyped here so the rules are pinned by tests.
# ---------------------------------------------------------------------------
@T.prim_func
def column_only_staging(A: T.Tensor((256, 576), "float16"),
                        B: T.Tensor((576, 256), "float16"),
                        Y: T.Tensor((256, 256), "float16")):
    """68.1: repeated serial writes, but only to column 0 of As."""
    with T.Kernel(1, 1, threads=128) as (im, inn):
        As = T.alloc_shared((256, 256), "float16")
        Bs = T.alloc_shared((256, 256), "float16")
        Cl = T.alloc_fragment((256, 256), "float32")
        Acc = T.alloc_fragment((256, 256), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(3, num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i in T.Parallel(256):
                for j in T.serial(256):
                    if im * 256 + i < 256 and ko * 256 < 576:
                        As[i, 0] = A[im * 256 + i, ko * 256]
            for i, j in T.Parallel(256, 256):
                if ko * 256 + i < 576 and inn * 256 + j < 256:
                    Bs[i, j] = B[ko * 256 + i, inn * 256 + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(256, 256):
            if im * 256 + i < 256 and inn * 256 + j < 256:
                Y[im * 256 + i, inn * 256 + j] = Acc[i, j]


@T.prim_func
def diagonal_staging(A: T.Tensor((256, 576), "float16"),
                     B: T.Tensor((576, 256), "float16"),
                     Y: T.Tensor((256, 256), "float16")):
    """One loop variable indexes both tile axes: only the diagonal is written."""
    with T.Kernel(1, 1, threads=128) as (im, inn):
        As = T.alloc_shared((256, 256), "float16")
        Bs = T.alloc_shared((256, 256), "float16")
        Cl = T.alloc_fragment((256, 256), "float32")
        Acc = T.alloc_fragment((256, 256), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(3, num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(256, 256):
                if im * 256 + i < 256 and ko * 256 + i < 576:
                    As[i, i] = A[im * 256 + i, ko * 256 + i]
            for i, j in T.Parallel(256, 256):
                if ko * 256 + i < 576 and inn * 256 + j < 256:
                    Bs[i, j] = B[ko * 256 + i, inn * 256 + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(256, 256):
            if im * 256 + i < 256 and inn * 256 + j < 256:
                Y[im * 256 + i, inn * 256 + j] = Acc[i, j]


@T.prim_func
def remapped_staging(A: T.Tensor((256, 576), "float16"),
                     B: T.Tensor((576, 256), "float16"),
                     Y: T.Tensor((256, 256), "float16")):
    """The column index is reversed: every element is written, but remapped."""
    with T.Kernel(1, 1, threads=128) as (im, inn):
        As = T.alloc_shared((256, 256), "float16")
        Bs = T.alloc_shared((256, 256), "float16")
        Cl = T.alloc_fragment((256, 256), "float32")
        Acc = T.alloc_fragment((256, 256), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(3, num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(256, 256):
                if im * 256 + i < 256 and ko * 256 + j < 576:
                    As[i, 255 - j] = A[im * 256 + i, ko * 256 + j]
            for i, j in T.Parallel(256, 256):
                if ko * 256 + i < 576 and inn * 256 + j < 256:
                    Bs[i, j] = B[ko * 256 + i, inn * 256 + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(256, 256):
            if im * 256 + i < 256 and inn * 256 + j < 256:
                Y[im * 256 + i, inn * 256 + j] = Acc[i, j]


@T.prim_func
def column_only_epilogue(A: T.Tensor((256, 576), "float16"),
                         B: T.Tensor((576, 256), "float16"),
                         Y: T.Tensor((256, 256), "float16")):
    """The epilogue writes column 0 of the accumulator to every output column."""
    with T.Kernel(1, 1, threads=128) as (im, inn):
        As = T.alloc_shared((256, 256), "float16")
        Bs = T.alloc_shared((256, 256), "float16")
        Cl = T.alloc_fragment((256, 256), "float32")
        Acc = T.alloc_fragment((256, 256), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(3, num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(256, 256):
                if im * 256 + i < 256 and ko * 256 + j < 576:
                    As[i, j] = A[im * 256 + i, ko * 256 + j]
            for i, j in T.Parallel(256, 256):
                if ko * 256 + i < 576 and inn * 256 + j < 256:
                    Bs[i, j] = B[ko * 256 + i, inn * 256 + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(256, 256):
            if im * 256 + i < 256 and inn * 256 + j < 256:
                Y[im * 256 + i, inn * 256 + j] = Acc[i, 0]


@T.prim_func
def fp32_operands(A: T.Tensor((256, 576), "float32"),
                  B: T.Tensor((576, 256), "float32"),
                  Y: T.Tensor((256, 256), "float16")):
    """68.2: a resolvable FP32 source whose interface the plan cannot keep."""
    with T.Kernel(1, 1, threads=128) as (im, inn):
        As = T.alloc_shared((256, 256), "float32")
        Bs = T.alloc_shared((256, 256), "float32")
        Cl = T.alloc_fragment((256, 256), "float32")
        Acc = T.alloc_fragment((256, 256), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(3, num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(256, 256):
                if im * 256 + i < 256 and ko * 256 + j < 576:
                    As[i, j] = A[im * 256 + i, ko * 256 + j]
            for i, j in T.Parallel(256, 256):
                if ko * 256 + i < 576 and inn * 256 + j < 256:
                    Bs[i, j] = B[ko * 256 + i, inn * 256 + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(256, 256):
            if im * 256 + i < 256 and inn * 256 + j < 256:
                Y[im * 256 + i, inn * 256 + j] = Acc[i, j]


def _strided_parameter(prim_func):
    """Declare A with a gapped row stride: one element per row pair."""
    params = list(prim_func.params)
    buffer_map = dict(prim_func.buffer_map)
    original = buffer_map[params[0]]
    buffer_map[params[0]] = tirx.decl_buffer(
        original.shape, original.dtype, name=original.name, data=original.data,
        strides=[int(original.shape[1]) * 2, 1], scope=original.scope())
    return tirx.PrimFunc(params, prim_func.body, prim_func.ret_type,
                          buffer_map, prim_func.attrs)


class LocalWriteCoverageTests(unittest.TestCase):
    """A tile store must write every tile element exactly once (68.1)."""

    def _refuses(self, kernel, needle):
        with self.assertRaises(BackendError) as caught:
            R.compile_guarded_tail(kernel, profile=PROFILE, options=options())
        self.assertIn(needle, caught.exception.diagnostics[0].message)

    def test_a_copy_that_writes_one_column_is_refused(self):
        self._refuses(column_only_staging, "does not cover the tile")

    def test_a_copy_that_writes_only_the_diagonal_is_refused(self):
        self._refuses(diagonal_staging, "indexes both tile axes")

    def test_a_remapped_copy_index_is_refused(self):
        self._refuses(remapped_staging,
                      "a tile store index is not one loop variable")

    def test_the_same_rule_covers_the_epilogue(self):
        self._refuses(column_only_epilogue, "does not cover the tile")

    def test_the_reference_program_still_covers_its_tiles(self):
        plan = R.compile_guarded_tail(negatives.reference, profile=PROFILE,
                                       options=options())[0]
        self.assertEqual(plan.plan_hash, REFERENCE_PLAN_WITH_HINT_TRACE)


class InterfaceContractTests(unittest.TestCase):
    """The plan's bindings are the source's parameters, or the source is refused (68.2)."""

    def test_fp32_operands_are_refused_instead_of_retyped(self):
        with self.assertRaises(BackendError) as caught:
            R.compile_guarded_tail(fp32_operands, profile=PROFILE,
                                   options=options())
        message = caught.exception.diagnostics[0].message
        self.assertIn("proves an FP16 interface", message)
        self.assertIn("float32", message)

    def test_a_strided_parameter_is_refused(self):
        prim = _strided_parameter(R.normalise(pilot(256, 576, 256))[0])
        with self.assertRaises(BackendError) as caught:
            R.compile_guarded_tail(prim, profile=PROFILE, options=options())
        diagnostic = caught.exception.diagnostics[0]
        self.assertEqual(diagnostic.code, RK_E_UNSUPPORTED)
        self.assertEqual(diagnostic.stage, "pass")
        self.assertIn("noncompact or unknown strides", diagnostic.message)
        actual = diagnostic.to_dict()["actual"]
        self.assertEqual(actual["buffer"], "A")
        self.assertEqual(actual["strides"], ["1152", "1"])

    def test_the_reference_interface_is_bound_as_source_declares_it(self):
        plan, _, facts, _ = R.compile_guarded_tail(
            negatives.reference, profile=PROFILE, options=options())
        types = {binding.name: str(binding.dtype) for binding in plan.bindings}
        self.assertEqual(types, {"A": "float16", "B": "float16",
                                 "Y": "float16"})
# ---------------------------------------------------------------------------
# 70 review round: the accumulator chain must be one def/use chain, checked on
# the buffers themselves rather than on their names or their order.
# ---------------------------------------------------------------------------
@T.prim_func
def gemm_into_the_accumulator(A: T.Tensor((256, 576), "float16"),
                              B: T.Tensor((576, 256), "float16"),
                              Y: T.Tensor((256, 256), "float16")):
    """70.2: the GEMM writes Acc, then the saved Cl overwrites that result."""
    with T.Kernel(1, 1, threads=128) as (im, inn):
        As = T.alloc_shared((256, 256), "float16")
        Bs = T.alloc_shared((256, 256), "float16")
        Cl = T.alloc_fragment((256, 256), "float32")
        Acc = T.alloc_fragment((256, 256), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(3, num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(256, 256):
                if im * 256 + i < 256 and ko * 256 + j < 576:
                    As[i, j] = A[im * 256 + i, ko * 256 + j]
            for i, j in T.Parallel(256, 256):
                if ko * 256 + i < 576 and inn * 256 + j < 256:
                    Bs[i, j] = B[ko * 256 + i, inn * 256 + j]
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Acc)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(256, 256):
            if im * 256 + i < 256 and inn * 256 + j < 256:
                Y[im * 256 + i, inn * 256 + j] = Acc[i, j]


@T.prim_func
def reversed_gemm_operands(A: T.Tensor((256, 576), "float16"),
                           B: T.Tensor((576, 256), "float16"),
                           Y: T.Tensor((256, 256), "float16")):
    """The GEMM reads the two tiles in the opposite order."""
    with T.Kernel(1, 1, threads=128) as (im, inn):
        As = T.alloc_shared((256, 256), "float16")
        Bs = T.alloc_shared((256, 256), "float16")
        Cl = T.alloc_fragment((256, 256), "float32")
        Acc = T.alloc_fragment((256, 256), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(3, num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(256, 256):
                if im * 256 + i < 256 and ko * 256 + j < 576:
                    As[i, j] = A[im * 256 + i, ko * 256 + j]
            for i, j in T.Parallel(256, 256):
                if ko * 256 + i < 576 and inn * 256 + j < 256:
                    Bs[i, j] = B[ko * 256 + i, inn * 256 + j]
            T.copy(Acc, Cl)
            T.gemm(Bs, As, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(256, 256):
            if im * 256 + i < 256 and inn * 256 + j < 256:
                Y[im * 256 + i, inn * 256 + j] = Acc[i, j]


def _symbolic_parameter(prim_func):
    """Declare A with a symbolic row extent instead of a constant one."""
    params = list(prim_func.params)
    buffer_map = dict(prim_func.buffer_map)
    original = buffer_map[params[0]]
    rows = tirx.Var("dyn_m", "int32")
    buffer_map[params[0]] = tirx.decl_buffer(
        (rows, original.shape[1]), original.dtype, name=original.name,
        data=original.data, scope=original.scope())
    return tirx.PrimFunc(params, prim_func.body, prim_func.ret_type,
                         buffer_map, prim_func.attrs)


class AccumulatorChainTests(unittest.TestCase):
    """The chain is bound by buffer identity, region and dtype (70.2)."""

    def test_a_gemm_that_writes_another_buffer_is_refused(self):
        with self.assertRaises(BackendError) as caught:
            R.compile_guarded_tail(gemm_into_the_accumulator, profile=PROFILE,
                                   options=options())
        message = caught.exception.diagnostics[0].message
        self.assertIn("tile GEMM result accumulator", message)
        self.assertIn("not the buffer the rest of the chain reads", message)

    def test_a_gemm_with_reversed_operands_is_refused(self):
        with self.assertRaises(BackendError) as caught:
            R.compile_guarded_tail(reversed_gemm_operands, profile=PROFILE,
                                   options=options())
        self.assertIn("an operand tile must read one K axis",
                      caught.exception.diagnostics[0].message)

    def test_the_reference_chain_still_binds_as_one_chain(self):
        plan = R.compile_guarded_tail(negatives.reference, profile=PROFILE,
                                       options=options())[0]
        self.assertEqual(plan.plan_hash, REFERENCE_PLAN_WITH_HINT_TRACE)

    def test_a_symbolic_extent_is_refused_before_it_is_read(self):
        prim = _symbolic_parameter(R.normalise(pilot(256, 576, 256))[0])
        with self.assertRaises(BackendError) as caught:
            R.compile_guarded_tail(prim, profile=PROFILE, options=options())
        self.assertIn("symbolic extent",
                      caught.exception.diagnostics[0].message)

    def test_the_wrong_accumulator_source_is_also_refused_by_localise(self):
        with self.assertRaises(BackendError):
            R.pass_localise(R.normalise(gemm_into_the_accumulator)[0])
# ---------------------------------------------------------------------------
# 72 review round: the consumed tile operations are an exact table, and every
# region a fill or a copy names must be the whole buffer.
# ---------------------------------------------------------------------------
@T.prim_func
def cumsum_staging(A: T.Tensor((256, 576), "float16"),
                   B: T.Tensor((576, 256), "float16"),
                   Y: T.Tensor((256, 256), "float16")):
    """72.2: a supported TileLang scan that the RKNPU pilot cannot restate."""
    with T.Kernel(1, 1, threads=128) as (im, inn):
        As = T.alloc_shared((256, 256), "float16")
        Bs = T.alloc_shared((256, 256), "float16")
        Cl = T.alloc_fragment((256, 256), "float32")
        Acc = T.alloc_fragment((256, 256), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(3, num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(256, 256):
                if im * 256 + i < 256 and ko * 256 + j < 576:
                    As[i, j] = A[im * 256 + i, ko * 256 + j]
            for i, j in T.Parallel(256, 256):
                if ko * 256 + i < 576 and inn * 256 + j < 256:
                    Bs[i, j] = B[ko * 256 + i, inn * 256 + j]
            T.cumsum(As, As, dim=1)
            T.copy(Acc, Cl)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, Acc)
        for i, j in T.Parallel(256, 256):
            if im * 256 + i < 256 and inn * 256 + j < 256:
                Y[im * 256 + i, inn * 256 + j] = Acc[i, j]


@T.prim_func
def explicit_full_regions(A: T.Tensor((256, 576), "float16"),
                          B: T.Tensor((576, 256), "float16"),
                          Y: T.Tensor((256, 256), "float16")):
    """The same computation written with explicit full-buffer slices."""
    with T.Kernel(1, 1, threads=128) as (im, inn):
        As = T.alloc_shared((256, 256), "float16")
        Bs = T.alloc_shared((256, 256), "float16")
        Cl = T.alloc_fragment((256, 256), "float32")
        Acc = T.alloc_fragment((256, 256), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(3, num_stages=2):
            T.fill(As, 0)
            T.fill(Bs, 0)
            for i, j in T.Parallel(256, 256):
                if im * 256 + i < 256 and ko * 256 + j < 576:
                    As[i, j] = A[im * 256 + i, ko * 256 + j]
            for i, j in T.Parallel(256, 256):
                if ko * 256 + i < 576 and inn * 256 + j < 256:
                    Bs[i, j] = B[ko * 256 + i, inn * 256 + j]
            T.copy(Acc[0:256, 0:256], Cl[0:256, 0:256])
            T.gemm(As, Bs, Cl)
            T.copy(Cl[0:256, 0:256], Acc[0:256, 0:256])
        for i, j in T.Parallel(256, 256):
            if im * 256 + i < 256 and inn * 256 + j < 256:
                Y[im * 256 + i, inn * 256 + j] = Acc[i, j]


def _shift_region(prim_func, choose):
    """Move the selected tile-op region one element along its first axis."""
    def shift(node):
        if type(node).__name__ != "Evaluate":
            return None
        call = node.value
        if not R._op_name(call).startswith("tl.tileop."):
            return None
        regions = R._region_exprs(call)
        index = choose(call, regions)
        if index is None:
            return None
        args = list(call.args)
        region = args[index]
        region_args = list(region.args)
        load = region_args[0]
        indices = list(load.indices)
        indices[0] = indices[0] + 1
        region_args[0] = tirx.BufferLoad(load.buffer, indices)
        args[index] = tirx.Call(region.dtype, region.op, region_args)
        return tirx.Evaluate(tirx.Call(call.dtype, call.op, args))

    return prim_func.with_body(sf.ir_transform(prim_func.body, None, shift))


def _shifted_writeback(prim_func):
    def choose(call, regions):
        if R._op_name(call) == "tl.tileop.copy" and len(regions) == 2 \
                and str(regions[0]["buffer"].name) == "Cl" \
                and str(regions[1]["buffer"].name) == "Acc":
            return 1
        return None

    return _shift_region(prim_func, choose)


def _shifted_clear(prim_func):
    def choose(call, regions):
        if R._op_name(call) == "tl.tileop.fill" \
                and str(regions[0]["buffer"].name) == "Acc":
            return 0
        return None

    return _shift_region(prim_func, choose)


class UnsupportedOperationTests(unittest.TestCase):
    """An operation outside the table is refused, never rewritten (72.2)."""

    def test_an_unsupported_tile_operation_is_refused(self):
        with self.assertRaises(BackendError) as caught:
            R.compile_guarded_tail(cumsum_staging, profile=PROFILE,
                                   options=options())
        diagnostic = caught.exception.diagnostics[0]
        self.assertEqual(diagnostic.code, RK_E_UNSUPPORTED)
        self.assertEqual(diagnostic.stage, "pass")
        self.assertIn("no lowering consumer", diagnostic.message)
        self.assertEqual(diagnostic.to_dict()["actual"]["op"], "tl.tileop.cumsum")

    def test_the_supported_table_is_exact(self):
        self.assertEqual(R.TILE_OPS,
                         {"tl.tileop.gemm": "gemm", "tl.tileop.fill": "fill",
                          "tl.tileop.copy": "copy"})


class RegionContractTests(unittest.TestCase):
    """Every fill and copy region must be the whole buffer (72.3)."""

    def test_a_writeback_that_starts_one_row_late_is_refused(self):
        prim = _shifted_writeback(R.normalise(negatives.reference)[0])
        with self.assertRaises(BackendError) as caught:
            R.compile_guarded_tail(prim, profile=PROFILE, options=options())
        self.assertIn("not the whole buffer it names",
                      caught.exception.diagnostics[0].message)

    def test_a_clear_that_starts_one_row_late_is_refused(self):
        prim = _shifted_clear(R.normalise(negatives.reference)[0])
        with self.assertRaises(BackendError) as caught:
            R.compile_guarded_tail(prim, profile=PROFILE, options=options())
        self.assertIn("not the whole buffer it names",
                      caught.exception.diagnostics[0].message)

    def test_explicit_full_regions_are_still_accepted(self):
        plan, _, _, _ = R.compile_guarded_tail(explicit_full_regions,
                                               profile=PROFILE,
                                               options=options())
        self.assertEqual(plan.plan_hash, REFERENCE_PLAN_WITH_HINT_TRACE)
# ---------------------------------------------------------------------------
# 81 review round: every statement with a state effect must be consumed at the
# place it occurs, not merely target a buffer the plan happens to own.
# ---------------------------------------------------------------------------
@T.prim_func
def blocked_compute(A: T.Tensor((256, 576), "float16"),
                    B: T.Tensor((576, 256), "float16"),
                    Y: T.Tensor((256, 256), "float16")):
    """The supported chain wrapped in a named block, with nothing outside it."""
    with T.Kernel(1, 1, threads=128) as (im, inn):
        with T.sblock("compute"):
            As = T.alloc_shared((256, 256), "float16")
            Bs = T.alloc_shared((256, 256), "float16")
            Cl = T.alloc_fragment((256, 256), "float32")
            Acc = T.alloc_fragment((256, 256), "float16")
            T.clear(Acc)
            for ko in T.Pipelined(3, num_stages=2):
                T.fill(As, 0)
                T.fill(Bs, 0)
                for i, j in T.Parallel(256, 256):
                    if im * 256 + i < 256 and ko * 256 + j < 576:
                        As[i, j] = A[im * 256 + i, ko * 256 + j]
                for i, j in T.Parallel(256, 256):
                    if ko * 256 + i < 576 and inn * 256 + j < 256:
                        Bs[i, j] = B[ko * 256 + i, inn * 256 + j]
                T.copy(Acc, Cl)
                T.gemm(As, Bs, Cl)
                T.copy(Cl, Acc)
            for i, j in T.Parallel(256, 256):
                if im * 256 + i < 256 and inn * 256 + j < 256:
                    Y[im * 256 + i, inn * 256 + j] = Acc[i, j]


@T.prim_func
def blocked_with_extra_output(A: T.Tensor((256, 576), "float16"),
                              B: T.Tensor((576, 256), "float16"),
                              Y: T.Tensor((256, 256), "float16")):
    """81.1: the reviewed source -- a scalar write to Y after the block."""
    with T.Kernel(1, 1, threads=128) as (im, inn):
        with T.sblock("compute"):
            As = T.alloc_shared((256, 256), "float16")
            Bs = T.alloc_shared((256, 256), "float16")
            Cl = T.alloc_fragment((256, 256), "float32")
            Acc = T.alloc_fragment((256, 256), "float16")
            T.clear(Acc)
            for ko in T.Pipelined(3, num_stages=2):
                T.fill(As, 0)
                T.fill(Bs, 0)
                for i, j in T.Parallel(256, 256):
                    if im * 256 + i < 256 and ko * 256 + j < 576:
                        As[i, j] = A[im * 256 + i, ko * 256 + j]
                for i, j in T.Parallel(256, 256):
                    if ko * 256 + i < 576 and inn * 256 + j < 256:
                        Bs[i, j] = B[ko * 256 + i, inn * 256 + j]
                T.copy(Acc, Cl)
                T.gemm(As, Bs, Cl)
                T.copy(Cl, Acc)
            for i, j in T.Parallel(256, 256):
                if im * 256 + i < 256 and inn * 256 + j < 256:
                    Y[im * 256 + i, inn * 256 + j] = Acc[i, j]
        Y[0, 0] = T.float16(0)


@T.prim_func
def blocked_with_extra_input(A: T.Tensor((256, 576), "float16"),
                             B: T.Tensor((576, 256), "float16"),
                             Y: T.Tensor((256, 256), "float16")):
    """The extra write targets an input parameter instead of the output."""
    with T.Kernel(1, 1, threads=128) as (im, inn):
        with T.sblock("compute"):
            As = T.alloc_shared((256, 256), "float16")
            Bs = T.alloc_shared((256, 256), "float16")
            Cl = T.alloc_fragment((256, 256), "float32")
            Acc = T.alloc_fragment((256, 256), "float16")
            T.clear(Acc)
            for ko in T.Pipelined(3, num_stages=2):
                T.fill(As, 0)
                T.fill(Bs, 0)
                for i, j in T.Parallel(256, 256):
                    if im * 256 + i < 256 and ko * 256 + j < 576:
                        As[i, j] = A[im * 256 + i, ko * 256 + j]
                for i, j in T.Parallel(256, 256):
                    if ko * 256 + i < 576 and inn * 256 + j < 256:
                        Bs[i, j] = B[ko * 256 + i, inn * 256 + j]
                T.copy(Acc, Cl)
                T.gemm(As, Bs, Cl)
                T.copy(Cl, Acc)
            for i, j in T.Parallel(256, 256):
                if im * 256 + i < 256 and inn * 256 + j < 256:
                    Y[im * 256 + i, inn * 256 + j] = Acc[i, j]
        A[0, 0] = T.float16(0)


@T.prim_func
def blocked_with_extra_prefix(A: T.Tensor((256, 576), "float16"),
                              B: T.Tensor((576, 256), "float16"),
                              Y: T.Tensor((256, 256), "float16")):
    """A write before the compute block instead of after it."""
    with T.Kernel(1, 1, threads=128) as (im, inn):
        Y[0, 0] = T.float16(0)
        with T.sblock("compute"):
            As = T.alloc_shared((256, 256), "float16")
            Bs = T.alloc_shared((256, 256), "float16")
            Cl = T.alloc_fragment((256, 256), "float32")
            Acc = T.alloc_fragment((256, 256), "float16")
            T.clear(Acc)
            for ko in T.Pipelined(3, num_stages=2):
                T.fill(As, 0)
                T.fill(Bs, 0)
                for i, j in T.Parallel(256, 256):
                    if im * 256 + i < 256 and ko * 256 + j < 576:
                        As[i, j] = A[im * 256 + i, ko * 256 + j]
                for i, j in T.Parallel(256, 256):
                    if ko * 256 + i < 576 and inn * 256 + j < 256:
                        Bs[i, j] = B[ko * 256 + i, inn * 256 + j]
                T.copy(Acc, Cl)
                T.gemm(As, Bs, Cl)
                T.copy(Cl, Acc)
            for i, j in T.Parallel(256, 256):
                if im * 256 + i < 256 and inn * 256 + j < 256:
                    Y[im * 256 + i, inn * 256 + j] = Acc[i, j]


def _output_parameter(prim_func):
    return next(buffer for buffer in prim_func.buffer_map.values()
                if str(buffer.name) == "Y")


def _extra_grid_sibling(prim_func):
    """Append Y[0,0] = 0 beside the compute block, inside the grid loop."""
    target = _output_parameter(prim_func)
    store = tirx.BufferStore(target, tirx.FloatImm("float16", 0), [0, 0])

    grid = [node for _, node in R._walk(prim_func.body)
            if type(node).__name__ == "For"]
    target_loop = grid[0]

    def append(node):
        if type(node).__name__ == "For" and node.loop_var.same_as(
                target_loop.loop_var):
            return tirx.For(node.loop_var, node.min, node.extent, node.kind,
                             tirx.SeqStmt([node.body, store]),
                             node.thread_binding, node.annotations)
        return None

    return prim_func.with_body(sf.ir_transform(prim_func.body, None, append))


def _extra_outside_grid(prim_func):
    """Append Y[0,0] = 0 beside the whole kernel body."""
    target = _output_parameter(prim_func)
    store = tirx.BufferStore(target, tirx.FloatImm("float16", 0), [0, 0])
    return prim_func.with_body(tirx.SeqStmt([prim_func.body, store]))


class EffectClosureTests(unittest.TestCase):
    """A state effect is consumed where it occurs, or refused (81.1)."""

    def _refuses(self, kernel, needle="not consumed by the pass chain"):
        with self.assertRaises(BackendError) as caught:
            R.compile_guarded_tail(kernel, profile=PROFILE, options=options())
        self.assertIn(needle, caught.exception.diagnostics[0].message)

    def test_a_block_wrapped_compute_is_still_supported(self):
        plan = R.compile_guarded_tail(blocked_compute, profile=PROFILE,
                                       options=options())[0]
        plain = R.compile_guarded_tail(negatives.reference, profile=PROFILE,
                                        options=options())[0]
        tasks = lambda value: [(a.template_id, tuple(a.inputs), tuple(a.outputs),
                                a.attrs.get("k_index"), a.attrs.get("k0"))
                               for a in value.actions if a.kind == "npu_task"]
        self.assertEqual(tasks(plan), tasks(plain))

    def test_a_write_after_the_block_is_refused(self):
        self._refuses(blocked_with_extra_output)

    def test_a_write_before_the_block_is_refused(self):
        self._refuses(blocked_with_extra_prefix)

    def test_a_write_beside_the_block_inside_the_grid_is_refused(self):
        prim = _extra_grid_sibling(R.normalise(blocked_compute)[0])
        self._refuses(prim, "state effect is not consumed")

    def test_a_write_outside_the_grid_is_refused(self):
        prim = _extra_outside_grid(R.normalise(blocked_compute)[0])
        self._refuses(prim, "state effect is not consumed")

    def test_a_write_to_an_input_parameter_is_refused(self):
        self._refuses(blocked_with_extra_input)

    def test_the_reference_writes_only_what_was_proved(self):
        _, _, facts, _ = R.compile_guarded_tail(negatives.reference,
                                                profile=PROFILE,
                                                options=options())
        self.assertEqual(len(facts.drained), 3)
        self.assertEqual([str(node.buffer.name) for node in facts.drained],
                         ["As", "Bs", "Y"])
if __name__ == "__main__":
    unittest.main()
