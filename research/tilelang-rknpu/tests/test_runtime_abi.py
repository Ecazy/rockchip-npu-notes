"""Real vendor ABI checks for rocket_open/rocket_close."""
import ctypes
import os
import re
import subprocess
import unittest
from pathlib import Path

from rknpu_backend.runtime.ffi import configure_rocket_abi

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT.parent
HEADER = (RESEARCH / "upstream-components" / "rocket-userspace" /
          "include" / "rocket_npu.h")
PLAN_LIB = os.environ.get("RKNPU_PLAN_LIB") or str(
    RESEARCH / "custom-ops" / "build-host-mvp" / "librknpu_plan.so")
PROBE = RESEARCH / "custom-ops" / "build-host-mvp" / "mock_signature_probe"


class RuntimeABITests(unittest.TestCase):
    def test_vendor_header_declares_void_rocket_close(self):
        text = HEADER.read_text()
        self.assertRegex(text, r"\bint\s+rocket_open\s*\(\s*void\s*\)\s*;")
        self.assertRegex(text, r"\bvoid\s+rocket_close\s*\(\s*int\s+fd\s*\)\s*;")

    def test_ctypes_configuration_matches_vendor_header(self):
        self.assertTrue(Path(PLAN_LIB).is_file(), PLAN_LIB)
        lib = ctypes.CDLL(PLAN_LIB)
        configure_rocket_abi(lib)
        self.assertEqual(lib.rocket_open.argtypes, [])
        self.assertIs(lib.rocket_open.restype, ctypes.c_int)
        self.assertEqual(lib.rocket_close.argtypes, [ctypes.c_int])
        self.assertIsNone(lib.rocket_close.restype)
        if hasattr(lib, "rocket_context_faulted"):
            self.assertEqual(lib.rocket_context_faulted.argtypes, [ctypes.c_int])
            self.assertIs(lib.rocket_context_faulted.restype, ctypes.c_int)

    def test_poisoned_flag_is_atomic_and_ordered_after_busy_gate(self):
        header = (RESEARCH / "custom-ops" / "src" /
                  "rk_plan_internal.h").read_text()
        self.assertIn("atomic_int poisoned;", header)
        source = (RESEARCH / "custom-ops" / "src" /
                  "rk_plan_runtime.c").read_text()
        run_body = source[source.index("int rk_workspace_run("):
                          source.index("int rk_workspace_destroy(")]
        busy_gate = run_body.index("atomic_exchange_explicit(&ws->busy, 1")
        poison_load = run_body.index(
            "atomic_load_explicit(&ws->poisoned")
        self.assertLess(busy_gate, poison_load)
        self.assertNotIn("ws->poisoned =", source)
        self.assertIn("atomic_store_explicit(&ws->poisoned, 1", source)

    def test_c_probe_compiles_vendor_close_as_void(self):
        if not PROBE.is_file():
            self.skipTest(f"signature probe not built: {PROBE}")
        out = subprocess.run([str(PROBE)], capture_output=True, text=True,
                             check=True).stdout
        signatures = {}
        for line in out.splitlines():
            if line.startswith("VENDOR_SIG\t"):
                _, name, ret, args = line.split("\t")
                signatures[name] = (ret, args)
        self.assertEqual(signatures.get("rocket_open"), ("int", ""))
        self.assertEqual(signatures.get("rocket_close"), ("void", "int"))
        self.assertEqual(signatures.get("rocket_context_faulted"),
                         ("int", "int"))


if __name__ == "__main__":
    unittest.main()
