"""The Python ctypes table and the C mock prototypes must agree exactly.

`mock_signature_probe` (tests/mock/mock_signature_probe.c) prints one
`MOCK_SIG<TAB>name<TAB>return<TAB>args` line per export.  The printed text and the
function-pointer assignments in that file are **two hand-written sources**: the
assignments make a C-side prototype change a build error, but the strings are not
derived from them, so the two must be kept in step by hand.

This test maps each printed C type name to the ctypes type it must be and compares
the *type objects* (pointers recursively), preserving arity, order and return type -
including `size_t`, the fixed-width integers and the pointer parameters.  Because
the comparison goes C-name -> ctypes, a platform where `ctypes.c_size_t is
ctypes.c_uint64` passes naturally, while a real sign/width/arity error still fails.
"""
import ctypes
import os
import subprocess
import unittest
from pathlib import Path

from rknpu_backend.runtime.ffi import LibRKPlan

ROOT = Path(__file__).resolve().parents[1]
PROBE = os.environ.get("RK_MOCK_PROBE_BIN") or str(
    ROOT.parent / "custom-ops" / "build-host-mvp" / "mock_signature_probe")

# The C probe prints type *names*; this table maps a name to the ctypes type it
# must be.  Mapping C -> ctypes (not ctypes -> name) keeps the comparison exact per
# platform: where ctypes aliases two names to one object (here c_size_t is
# c_uint64 and c_uint is c_uint32) the alias is simply the right answer, while a
# real width/sign mismatch still fails.
CTYPES_BY_NAME = {
    "void": None,
    "int": ctypes.c_int,
    "unsigned": ctypes.c_uint,
    "size_t": ctypes.c_size_t,
    "uint16_t": ctypes.c_uint16,
    "uint32_t": ctypes.c_uint32,
    "uint64_t": ctypes.c_uint64,
}


def ctypes_for(name):
    name = name.strip()
    if name.endswith("*"):
        return ctypes.POINTER(ctypes_for(name[:-1]))
    return CTYPES_BY_NAME[name]


def compare(declared, c_ret, c_args):
    """Return a list of mismatches (empty list means the signatures agree)."""
    problems = []
    if declared[0] is not ctypes_for(c_ret):
        problems.append(f"return: {declared[0]!r} vs C {c_ret!r}")
    want = [ctypes_for(a) for a in c_args.split(",") if a.strip()]
    got = list(declared[1])
    if len(got) != len(want):
        problems.append(f"arity: {len(got)} vs {len(want)}")
    for i, (g, w) in enumerate(zip(got, want)):
        if g is not w:
            problems.append(f"param {i}: {g!r} vs C {w!r}")
    return problems


class MockApiParityTests(unittest.TestCase):
    def setUp(self):
        if not Path(PROBE).is_file():
            self.skipTest(f"mock_signature_probe not built: {PROBE}")
        proc = subprocess.run([PROBE], capture_output=True, text=True, check=True)
        self.c_sigs = {}
        for line in proc.stdout.splitlines():
            if not line.startswith("MOCK_SIG\t"):
                continue
            _, name, ret, args = line.split("\t")
            self.c_sigs[name] = (ret, args)

    def test_export_sets_match(self):
        self.assertEqual(set(self.c_sigs), set(LibRKPlan._MOCK_SIGNATURES))

    def test_return_and_parameter_types_match(self):
        for name, (ret, args) in sorted(self.c_sigs.items()):
            with self.subTest(symbol=name):
                self.assertEqual(
                    compare(LibRKPlan._MOCK_SIGNATURES[name], ret, args), [])

    def test_size_t_and_pointer_parameters_are_covered(self):
        sig = LibRKPlan._MOCK_SIGNATURES
        self.assertIs(sig["rk_mock_resident"][0], ctypes.c_size_t)
        self.assertIs(sig["rk_mock_dma_range"][1][3], ctypes.POINTER(ctypes.c_size_t))
        self.assertIs(sig["rk_mock_dma_range"][1][1], ctypes.POINTER(ctypes.c_uint32))
        self.assertIs(sig["rk_mock_find_half"][1][0], ctypes.c_uint16)
        self.assertIs(sig["rk_mock_find_half"][1][2], ctypes.POINTER(ctypes.c_size_t))
        self.assertIs(sig["rk_mock_bo_dma_address"][0], ctypes.c_uint64)

    def test_comparator_rejects_wrong_sign_width_and_arity(self):
        sig = LibRKPlan._MOCK_SIGNATURES
        self.assertTrue(compare(sig["rk_mock_execs"], "int", ""))          # sign
        self.assertTrue(compare(sig["rk_mock_fail_at"], "void", "uint16_t"))  # width
        self.assertTrue(compare(sig["rk_mock_dma_range"], "int",
                                "unsigned,uint32_t *,uint32_t *"))          # arity
        self.assertTrue(compare(sig["rk_mock_resident"], "unsigned", ""))  # width
        self.assertEqual(compare(sig["rk_mock_execs"], "unsigned", ""), [])  # control


if __name__ == "__main__":
    unittest.main()
