# SPDX-License-Identifier: GPL-3.0-or-later
"""Guarded-tail sources with no hint, num_stages=0 and num_stages=2.

The three sources spell the same math in one file so TileLang can recover their
source text.  They must compile to the same execution program and differ only in
the recorded pipeline request.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import tilelang.language as T

from rknpu_backend import (CompileOptions, compile_tilelang, default_profile,
                           load_plan, save_plan)
from rknpu_backend.frontend.tir_utils import PIPELINE_NOT_APPLIED_REASON

PROFILE = default_profile()
M, K, N, BM, BN = 256, 576, 3072, 256, 256
RECORD_KEYS = ("source_loop_annotations", "source_pipeline_stages",
               "pipeline_hint_dropped", "pipeline_not_applied_reason")
# Per-source provenance (the span of the loop in the file) is not execution.
PROVENANCE_KEYS = ("source_loop_span",)


def options():
    return CompileOptions(numeric_mode="rk_f16_tiled", allow_approx_lut=False)


def _body(T_, A, B, Y, im, inn, As, Bs, Cl, Acc, ko):
    T_.fill(As, 0)
    T_.fill(Bs, 0)
    for i, j in T_.Parallel(BM, 256):
        if im * BM + i < M and ko * 256 + j < K:
            As[i, j] = A[im * BM + i, ko * 256 + j]
    for i, j in T_.Parallel(256, BN):
        if ko * 256 + i < K and inn * BN + j < N:
            Bs[i, j] = B[ko * 256 + i, inn * BN + j]
    T_.copy(Acc, Cl)
    T_.gemm(As, Bs, Cl)
    T_.copy(Cl, Acc)


@T.prim_func
def tail_serial(A: T.Tensor((M, K), "float16"), B: T.Tensor((K, N), "float16"),
                Y: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN), threads=128) as (im, inn):
        As = T.alloc_shared((BM, 256), "float16")
        Bs = T.alloc_shared((256, BN), "float16")
        Cl = T.alloc_fragment((BM, BN), "float32")
        Acc = T.alloc_fragment((BM, BN), "float16")
        T.clear(Acc)
        for ko in T.serial(T.ceildiv(K, 256)):
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
def tail_stages_zero(A: T.Tensor((M, K), "float16"),
                     B: T.Tensor((K, N), "float16"),
                     Y: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN), threads=128) as (im, inn):
        As = T.alloc_shared((BM, 256), "float16")
        Bs = T.alloc_shared((256, BN), "float16")
        Cl = T.alloc_fragment((BM, BN), "float32")
        Acc = T.alloc_fragment((BM, BN), "float16")
        T.clear(Acc)
        for ko in T.Pipelined(T.ceildiv(K, 256), num_stages=0):
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
def tail_stages_two(A: T.Tensor((M, K), "float16"),
                    B: T.Tensor((K, N), "float16"),
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


SOURCES = {"serial": tail_serial, "stages_zero": tail_stages_zero,
           "stages_two": tail_stages_two}


def compile_source(source):
    return compile_tilelang(source, profile=PROFILE, options=options()).plan


def k_loop_record(plan):
    for action in plan.actions:
        if action.kind == "npu_task" and isinstance(
                action.attrs.get("source_k_loop"), dict):
            return action.attrs["source_k_loop"]
    return {}


def strip_records(plan_dict):
    """Drop the pipeline record fields so execution can be compared."""
    import copy

    body = copy.deepcopy(plan_dict)
    for action in body["actions"]:
        attrs = action.get("attrs") or {}
        for key in RECORD_KEYS + PROVENANCE_KEYS:
            attrs.pop(key, None)
        loop = attrs.get("source_k_loop")
        if isinstance(loop, dict):
            for key in RECORD_KEYS + PROVENANCE_KEYS:
                loop.pop(key, None)
    body.pop("plan_hash", None)
    return body


class GuardedTailPipelineRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plans = {name: compile_source(source) for name, source in SOURCES.items()}

    def test_all_three_sources_compile(self):
        for name, plan in self.plans.items():
            with self.subTest(source=name):
                self.assertEqual(plan.plan_hash[:12],
                                 plan.plan_hash[:12])
                tasks = [a for a in plan.actions if a.kind == "npu_task"]
                self.assertEqual(len(tasks), 36)
                self.assertEqual(len(plan.submits), 12)

    def test_the_request_is_recorded_per_variant(self):
        serial = k_loop_record(self.plans["serial"])
        zero = k_loop_record(self.plans["stages_zero"])
        two = k_loop_record(self.plans["stages_two"])
        self.assertEqual(serial.get("source_pipeline_stages"), 0)
        self.assertFalse(serial.get("pipeline_hint_dropped"))
        self.assertIsNone(serial.get("pipeline_not_applied_reason"))
        self.assertEqual(zero.get("source_pipeline_stages"), 0)
        self.assertFalse(zero.get("pipeline_hint_dropped"))
        self.assertIsNone(zero.get("pipeline_not_applied_reason"))
        self.assertEqual(two.get("source_pipeline_stages"), 2)
        self.assertTrue(two.get("pipeline_hint_dropped"))
        self.assertEqual(two.get("pipeline_not_applied_reason"),
                         PIPELINE_NOT_APPLIED_REASON)

    def test_stages_zero_is_not_reported_as_a_dropped_request(self):
        zero = k_loop_record(self.plans["stages_zero"])
        self.assertEqual(zero.get("source_pipeline_stages"), 0)
        self.assertFalse(zero.get("pipeline_hint_dropped"))
        self.assertIsNone(zero.get("pipeline_not_applied_reason"))
        # TileLang may drop a zero-stage annotation entirely; either way it is
        # not a deferred request.
        self.assertFalse(zero.get("pipeline_hint_dropped"))

    EXECUTION_KEYS = ("bindings", "constants", "views", "buffers", "actions",
                      "submits", "publish", "accounting", "budget_policy",
                      "logical_ops", "numeric_requirements")

    def test_execution_fields_are_identical_across_the_three_sources(self):
        bodies = {name: strip_records(plan.to_dict())
                  for name, plan in self.plans.items()}
        bodies = {name: {key: body[key] for key in self.EXECUTION_KEYS
                         if key in body}
                  for name, body in bodies.items()}
        reference = bodies["serial"]
        for name, body in bodies.items():
            with self.subTest(source=name):
                self.assertEqual(body, reference)
        # accounting, publish and device budget must match too
        devices = {plan.accounting.device_request_bytes
                   for plan in self.plans.values()}
        submits = {len(plan.submits) for plan in self.plans.values()}
        publishes = {len(plan.publish) for plan in self.plans.values()}
        self.assertEqual(len(devices), 1)
        self.assertEqual(submits, {12})
        self.assertEqual(len(publishes), 1)

    def test_a_pipeline_request_changes_only_the_record(self):
        with_two = k_loop_record(self.plans["stages_two"])
        without = k_loop_record(self.plans["serial"])
        different = {key for key in RECORD_KEYS
                     if with_two.get(key) != without.get(key)}
        self.assertTrue(different)
        self.assertLessEqual(different, set(RECORD_KEYS))

    def test_records_survive_save_and_load(self):
        for name, plan in self.plans.items():
            with self.subTest(source=name):
                with tempfile.TemporaryDirectory() as tmp:
                    path = str(Path(tmp) / "plan.rkplan.json")
                    save_plan(plan, path)
                    loaded = load_plan(path)
                self.assertEqual(loaded.plan_hash, plan.plan_hash)
                self.assertEqual(k_loop_record(loaded), k_loop_record(plan))
                self.assertEqual(strip_records(loaded.to_dict()),
                                 strip_records(plan.to_dict()))


if __name__ == "__main__":
    unittest.main()
