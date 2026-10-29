"""Immutable packed-constant sharing and persistent upload-state contracts."""
from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOCK_LIB = os.environ.get("RKNPU_PLAN_MOCK_LIB") or str(
    ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan_mock.so")


class SharedConstantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not Path(MOCK_LIB).is_file():
            raise unittest.SkipTest(f"mock library not built: {MOCK_LIB}")
        from rknpu_backend import CompileOptions, default_profile
        from rknpu_backend.runtime.ffi import LibRKPlan
        from models.tilelang_matmul import make_matmul_prim
        from rknpu_backend import compile_tilelang
        cls.lib = LibRKPlan(MOCK_LIB)
        cls.profile = default_profile()
        cls.options = CompileOptions(numeric_mode="rk_f16_tiled",
                                     allow_approx_lut=False)
        cls.plan = compile_tilelang(
            make_matmul_prim(2, 64, 128), profile=cls.profile,
            options=cls.options, b_is_input=False).plan
        cls.a_packed_bytes = next(
            cls.plan.views[a.outputs[0]].tensor.bytes
            for a in cls.plan.actions if a.kind == "upload" and
            cls.plan.bindings[a.attrs["source_binding"]].name == "A")

    def setUp(self):
        self.calls = []
        self.lib.mock_control("rk_mock_reset")
        self.b = np.ascontiguousarray(
            np.random.default_rng(123).normal(size=(64, 128)).astype(np.float16))
        self.x = np.ascontiguousarray(
            np.random.default_rng(124).normal(size=(2, 64)).astype(np.float16))

    def tearDown(self):
        for call in reversed(self.calls):
            call.close()
        self.calls.clear()
        self.assertEqual(self.lib.shared_constant_live(), 0)

    def _call(self, weight=None, fd=0):
        from rknpu_backend.models import PlanCall
        call = PlanCall(self.lib, self.plan,
                        {"B": self.b if weight is None else weight},
                        fd=fd, share_constants=True)
        self.calls.append(call)
        return call

    def test_same_content_reuses_every_packed_tile_and_runs_numerically(self):
        c1 = self._call()
        self.assertGreater(c1.shared_constant_actions, 0)
        self.assertEqual(c1.shared_constant_created, c1.shared_constant_actions)
        c2 = self._call(weight=self.b.copy())
        self.assertEqual(c2.shared_constant_reused, c2.shared_constant_actions)
        self.assertEqual(c2.shared_constant_created, 0)
        self.assertEqual(self.lib.shared_constant_live(),
                         c1.shared_constant_actions)
        y1 = np.empty((2, 128), dtype=np.float16)
        y2 = np.empty_like(y1)
        r1, _ = c1.run_into([self.x], [y1])
        r2, _ = c2.run_into([self.x], [y2])
        np.testing.assert_array_equal(y1, y2)
        np.testing.assert_allclose(
            y1.astype(np.float64),
            self.x.astype(np.float64) @ self.b.astype(np.float64),
            atol=2e-2, rtol=2e-2)
        # The B uploads happened once when the shared BOs were packed, not once
        # per run.  Only the A activation upload remains in the run report.
        self.assertEqual(int(r1.upload_calls), 1)
        self.assertEqual(int(r2.upload_calls), 1)
        # Upload accounting is packed destination bytes, not the borrowed Host
        # source span.  It must be exactly one packed A tile and exclude B.
        self.assertEqual(int(r1.upload_bytes), self.a_packed_bytes)

    def test_changed_content_does_not_hit_the_cache(self):
        c1 = self._call()
        changed = self.b.copy()
        changed[0, 0] = np.nextafter(changed[0, 0], np.float16(np.inf))
        c2 = self._call(weight=changed)
        self.assertEqual(c2.shared_constant_created, c2.shared_constant_actions)
        self.assertEqual(c2.shared_constant_reused, 0)
        self.assertEqual(self.lib.shared_constant_live(),
                         2 * c1.shared_constant_actions)

    def test_same_pointer_content_change_invalidates_new_pack_identity(self):
        c1 = self._call()
        y1 = np.empty((2, 128), dtype=np.float16)
        c1.run_into([self.x], [y1])
        old_first = np.array(y1, copy=True)
        # Mutate the caller's original buffer in place, then ask for another
        # PlanCall.  Identity is content-derived, never the Host pointer.
        self.b[0, :] = np.float16(1.0)
        c2 = self._call()
        self.assertEqual(c2.shared_constant_created, c2.shared_constant_actions)
        self.assertEqual(c2.shared_constant_reused, 0)
        y2 = np.empty_like(y1)
        c2.run_into([self.x], [y2])
        self.assertFalse(np.array_equal(old_first, y2))

    def test_release_order_and_aba_reuse(self):
        c1 = self._call()
        c2 = self._call()
        n = c1.shared_constant_actions
        self.assertEqual(self.lib.shared_constant_live(), n)
        c1.close()
        self.assertEqual(self.lib.shared_constant_live(), n)
        c3 = self._call()
        self.assertEqual(c3.shared_constant_reused, c3.shared_constant_actions)
        self.assertEqual(c3.shared_constant_created, 0)
        c2.close()
        self.assertEqual(self.lib.shared_constant_live(), n)
        c3.close()
        self.assertEqual(self.lib.shared_constant_live(), 0)

    def test_fd_context_isolation_is_rejected(self):
        from rknpu_backend.errors import BackendError, RK_E_DOMAIN
        from rknpu_backend.runtime.ffi import make_binding
        from rknpu_backend.models import _shared_constant_pack_key
        import hashlib
        action = next(a for a in self.plan.actions if a.kind == "upload" and
                      self.plan.bindings[a.attrs["source_binding"]].name == "B")
        spec = next(c for c in self.plan.constants if c.name == "B")
        source = make_binding(spec.binding_id, self.b)
        digest = hashlib.sha256(memoryview(self.b).cast("B")).hexdigest()
        key = _shared_constant_pack_key(self.plan, action, spec, digest)
        cplan0 = self.lib.create_plan(self.plan)
        handle, created = self.lib.create_shared_constant_on(
            7, cplan0, action.id, source, key)
        self.assertTrue(created)
        cplan = self.lib.create_plan(self.plan)
        try:
            with self.assertRaises(BackendError) as ctx:
                self.lib.create_workspace_shared(
                    cplan, [source], [(action.id, handle)], fd=8)
            self.assertEqual(ctx.exception.code, RK_E_DOMAIN)
        finally:
            self.lib.release_shared_constant(handle)
            cplan0.destroy()
            cplan.destroy()

    def test_incompatible_destination_geometry_is_rejected(self):
        from rknpu_backend import compile_tilelang
        from rknpu_backend.errors import BackendError, RK_E_SHAPE
        from rknpu_backend.runtime.ffi import make_binding
        from rknpu_backend.models import _shared_constant_pack_key
        from models.tilelang_matmul import make_matmul_prim
        import hashlib
        other = compile_tilelang(
            make_matmul_prim(2, 64, 192), profile=self.profile,
            options=self.options, b_is_input=False).plan
        action = next(a for a in self.plan.actions if a.kind == "upload" and
                      self.plan.bindings[a.attrs["source_binding"]].name == "B")
        spec = next(c for c in self.plan.constants if c.name == "B")
        source = make_binding(spec.binding_id, self.b)
        digest = hashlib.sha256(memoryview(self.b).cast("B")).hexdigest()
        key = _shared_constant_pack_key(self.plan, action, spec, digest)
        cplan1 = self.lib.create_plan(self.plan)
        handle, _ = self.lib.create_shared_constant_on(
            0, cplan1, action.id, source, key)
        cplan2 = self.lib.create_plan(other)
        try:
            with self.assertRaises(BackendError) as ctx:
                self.lib.create_workspace_shared(
                    cplan2, [source], [(action.id, handle)], fd=0)
            self.assertEqual(ctx.exception.code, RK_E_SHAPE)
        finally:
            self.lib.release_shared_constant(handle)
            cplan1.destroy()
            cplan2.destroy()


class PersistentUploadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not Path(MOCK_LIB).is_file():
            raise unittest.SkipTest(f"mock library not built: {MOCK_LIB}")
        from rknpu_backend import CompileOptions, default_profile, compile_tilelang
        from rknpu_backend.runtime.ffi import LibRKPlan
        from models.tilelang_matmul import make_matmul_prim
        cls.lib = LibRKPlan(MOCK_LIB)
        cls.plan = compile_tilelang(
            make_matmul_prim(2, 8, 16), profile=default_profile(),
            options=CompileOptions(numeric_mode="rk_f16_tiled",
                                   allow_approx_lut=False),
            b_is_input=True).plan
        cls.a_packed_bytes = next(
            cls.plan.views[a.outputs[0]].tensor.bytes
            for a in cls.plan.actions if a.kind == "upload" and
            cls.plan.bindings[a.attrs["source_binding"]].name == "A")

    def setUp(self):
        from rknpu_backend.models import PlanCall
        self.lib.mock_control("rk_mock_reset")
        self.call = PlanCall(self.lib, self.plan, {}, fd=0)
        self.b_binding = next(b for b in self.plan.bindings if b.name == "B")
        self.upload = next(
            a for a in self.plan.actions if a.kind == "upload" and
            a.attrs["source_binding"] == self.b_binding.id)
        self.call.set_persistent_upload(self.upload.id)
        self.x = np.ascontiguousarray(
            np.random.default_rng(9).normal(size=(2, 8)).astype(np.float16))
        self.b = np.ascontiguousarray(
            np.arange(8 * 16, dtype=np.float16).reshape(8, 16) / 16)
        self.dummy_b = np.zeros((8, 16), dtype=np.float16)

    def tearDown(self):
        self.call.close()

    def _run(self):
        y = np.empty((2, 16), dtype=np.float16)
        report, _ = self.call.run_into([self.x, self.dummy_b], [y])
        return y, report

    def test_full_init_incremental_row_update_and_clear(self):
        self.call.update_upload_region(
            self.upload.id, self.b, src_row=0, src_col=0,
            dst_row=0, dst_col=0, rows=16, cols=8,
            transpose_source=True)
        y, report = self._run()
        np.testing.assert_allclose(
            y.astype(np.float64),
            self.x.astype(np.float64) @ self.b.astype(np.float64),
            atol=2e-2, rtol=2e-2)
        # B is persistent: the run only uploads A.
        self.assertEqual(int(report.upload_calls), 1)
        self.assertEqual(int(report.upload_bytes),
                         PersistentUploadTests.a_packed_bytes)

        b2 = self.b.copy()
        b2[3, :] = np.float16(1.5)
        self.call.update_upload_region(
            self.upload.id, np.ascontiguousarray(b2[3, :]),
            src_row=0, src_col=0, dst_row=0, dst_col=3,
            rows=16, cols=1, transpose_source=True)
        y2, report2 = self._run()
        np.testing.assert_allclose(
            y2.astype(np.float64),
            self.x.astype(np.float64) @ b2.astype(np.float64),
            atol=2e-2, rtol=2e-2)
        self.assertEqual(int(report2.upload_calls), 1)

        self.call.clear_upload_destination(self.upload.id)
        y3, _ = self._run()
        self.assertEqual(float(np.max(np.abs(y3))), 0.0)

    def test_dynamic_a_same_pointer_is_uploaded_fresh_each_run(self):
        self.call.update_upload_region(
            self.upload.id, self.b, src_row=0, src_col=0,
            dst_row=0, dst_col=0, rows=16, cols=8,
            transpose_source=True)
        x_ptr = self.x
        y1, _ = self._run()
        x_ptr[...] = np.float16(0.25)
        y2, _ = self._run()
        ref = x_ptr.astype(np.float64) @ self.b.astype(np.float64)
        np.testing.assert_allclose(y2.astype(np.float64), ref,
                                   atol=2e-2, rtol=2e-2)
        self.assertFalse(np.array_equal(y1, y2))

    def test_region_outside_destination_is_rejected_before_write(self):
        from rknpu_backend.errors import BackendError, RK_E_SHAPE
        with self.assertRaises(BackendError) as ctx:
            self.call.update_upload_region(
                self.upload.id, self.b, src_row=0, src_col=0,
                dst_row=15, dst_col=0, rows=2, cols=8,
                transpose_source=True)
        self.assertEqual(ctx.exception.code, RK_E_SHAPE)
        # The workspace remains healthy after a pre-submit validation failure.
        self.call.update_upload_region(
            self.upload.id, self.b, src_row=0, src_col=0,
            dst_row=0, dst_col=0, rows=16, cols=8,
            transpose_source=True)
        y, _ = self._run()
        self.assertTrue(np.isfinite(y).all())


if __name__ == "__main__":
    unittest.main()
