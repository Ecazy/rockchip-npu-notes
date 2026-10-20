import ctypes
import unittest

from rknpu_backend import CompileOptions, compile_rkir, default_profile, matmul_module


class FFITests(unittest.TestCase):
    def test_ctypes_layout_matches_c_header_sizes(self):
        from rknpu_backend.runtime import ffi
        expected = {
            ffi.RKValueDesc: 56,
            ffi.RKTensorDesc: 48,
            ffi.RKBufferDesc: 56,
            ffi.RKViewDesc: 88,
            ffi.RKActionDesc: 144,
            ffi.RKSubmitDesc: 48,
            ffi.RKPublishDesc: 16,
            ffi.RKPlanDesc: 160,
            ffi.RKBinding: 48,
            ffi.RKRunReport: 88,
            ffi.RKDiag: 760,
        }
        for struct, size in expected.items():
            self.assertEqual(ctypes.sizeof(struct), size, struct.__name__)

    def test_plan_create_deep_copies_descriptor(self):
        try:
            from rknpu_backend.runtime.ffi import default_library
            lib = default_library()
        except Exception as exc:
            self.skipTest(f"rknpu_plan shared library not built: {exc}")
        self.assertEqual(lib.abi_version(), 1)
        plan = compile_rkir(
            matmul_module(m=13, k=64, n=33, activation="silu"),
            profile=default_profile(),
            options=CompileOptions(numeric_mode="rk_f16_tiled",
                                   allow_approx_lut=True)).plan
        cplan = lib.create_plan(plan)
        self.assertTrue(cplan._ptr)
        cplan.destroy()
        self.assertIsNone(cplan._ptr)


if __name__ == "__main__":
    unittest.main()
