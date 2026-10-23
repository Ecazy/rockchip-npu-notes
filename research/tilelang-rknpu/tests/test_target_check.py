"""Target identity parsing: exact NUL-token chip and exact driver version."""
import ctypes
import os
import unittest
from pathlib import Path

from rknpu_backend.runtime.ffi import LibRKPlan, RKDiag

ROOT = Path(__file__).resolve().parents[1]
PLAN_LIB = os.environ.get("RKNPU_PLAN_LIB") or str(
    ROOT.parent / "custom-ops" / "build-host-mvp" / "librknpu_plan.so")


class TargetCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not Path(PLAN_LIB).is_file():
            raise unittest.SkipTest(f"production library not built: {PLAN_LIB}")
        cls.lib = LibRKPlan(PLAN_LIB)
        fn = getattr(cls.lib.cdll, "rk_target_validate_text", None)
        if fn is None:
            raise unittest.SkipTest("rk_target_validate_text is not exported")
        fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p,
                       ctypes.POINTER(RKDiag)]
        fn.restype = ctypes.c_int
        cls.fn = fn

    def check(self, compat: bytes, version: bytes) -> int:
        buf = ctypes.create_string_buffer(compat)
        diag = RKDiag()
        return int(self.fn(buf, len(compat), version, ctypes.byref(diag)))

    def test_accepts_exact_token_in_nul_separated_list(self):
        self.assertEqual(
            self.check(b"other,board\0rockchip,rk3588\0",
                       b"RKNPU driver: v0.9.8"), 0)

    def test_rejects_chip_name_only_as_substring(self):
        self.assertNotEqual(
            self.check(b"other,board\0rockchip,rk3588-extra\0",
                       b"RKNPU driver: v0.9.8"), 0)

    def test_rejects_nearby_driver_version_text(self):
        for version in (b"RKNPU driver: v0.9.80",
                        b"prefix RKNPU driver: v0.9.8 suffix",
                        b"RKNPU driver: v0.9.8-extra"):
            with self.subTest(version=version):
                self.assertNotEqual(
                    self.check(b"rockchip,rk3588\0", version), 0)


if __name__ == "__main__":
    unittest.main()
