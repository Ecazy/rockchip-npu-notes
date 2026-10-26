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
        # The gate is one function so every entry point shares it; the ordering
        # invariant (busy acquired before poison is observed) is checked there.
        gate_body = source[source.index("static int run_gate("):
                           source.index("static int validate_repeat_bindings(")]
        busy_gate = gate_body.index("atomic_exchange_explicit(&ws->busy, 1")
        poison_load = gate_body.index(
            "atomic_load_explicit(&ws->poisoned")
        self.assertLess(busy_gate, poison_load)
        self.assertNotIn("ws->poisoned =", source)
        self.assertIn("atomic_store_explicit(&ws->poisoned, 1", source)
        # Every path that can reach device work must take the gate first: the
        # shared single-run body (used by both rk_workspace_run entries) and the
        # group entry, each before its first repeat.
        body = source[source.index("static uint32_t plan_task_count("):
                      source.index("int rk_workspace_run_group(")]
        self.assertLess(body.index("run_gate(ws"), body.index("run_repeat("))
        run_body = source[source.index("int rk_workspace_run("):
                          source.index("int rk_workspace_destroy(")]
        self.assertEqual(run_body.count("workspace_run_body("), 2)
        group_body = source[source.index("int rk_workspace_run_group("):
                            source.index("int rk_workspace_destroy(")]
        self.assertLess(group_body.index("run_gate(ws"),
                        group_body.index("run_repeat("))

    def test_legacy_report_entry_cannot_write_past_its_struct(self):
        """The pre-extension report size is a hard write boundary.

        ``rk_workspace_run`` is called by userspace that predates the
        attribution extension, where ``struct_size`` was never an input.  It must
        therefore write exactly ``sizeof(rk_run_report)`` bytes and must not use
        ``struct_size`` to decide how much to write; the extended fields live in
        a separate struct behind ``rk_workspace_run_ex``.
        """
        header = (RESEARCH / "custom-ops" / "include" / "rk_plan.h").read_text()
        report = header[header.index("typedef struct rk_run_report {"):
                        header.index("} rk_run_report;")]
        for extended in ("validate_us", "guard_us", "total_us", "clock_us"):
            self.assertNotIn(extended, report,
                             f"{extended} must not be part of rk_run_report")
        self.assertIn("typedef struct rk_run_attribution {", header)
        self.assertIn("RK_RUN_ATTRIBUTION_V1_SIZE", header)
        source = (RESEARCH / "custom-ops" / "src" /
                  "rk_plan_runtime.c").read_text()
        clear = source[source.index("static void run_report_clear("):
                       source.index("static void attribution_clear(")]
        self.assertIn("memset(report, 0, sizeof(*report))", clear)
        self.assertNotIn("report->struct_size", clear.replace(
            "report->struct_size = (uint32_t)sizeof(*report)", ""))
        # The extended entries validate the declaration instead of guessing it.
        self.assertIn("is not a usable ABI ", source)
        for entry in ("attribution_size(", "group_size("):
            self.assertIn(entry, source)

    def test_no_workspace_access_after_the_run_lock_is_released(self):
        """Reporting must not dereference the workspace after the busy release.

        ``workspace_run_body`` releases the single-run lock before returning, so
        another thread may legally destroy the workspace immediately afterwards.
        The public entries therefore have to fill their report from local state
        copied while the lock was held (``st.task_count``), not from
        ``ws->plan``.
        """
        source = (RESEARCH / "custom-ops" / "src" /
                  "rk_plan_runtime.c").read_text()
        body = source[source.index("static int workspace_run_body("):
                      source.index("int rk_workspace_run_group(")]
        # The lock is released inside the body, before it returns.
        self.assertIn("atomic_store_explicit(&ws->busy, 0", body)
        self.assertLess(body.index("st->task_count = plan_task_count(ws->plan)"),
                        body.index("atomic_store_explicit(&ws->busy, 0"))
        for entry in ("int rk_workspace_run(", "int rk_workspace_run_ex("):
            start = body.index(entry)
            end = body.index("\n}\n", start)
            text = body[start:end]
            call = text.index("workspace_run_body(")
            tail = text[text.index(")", call):]
            self.assertNotIn("ws->", tail,
                             f"{entry} dereferences the workspace after the run "
                             "body (lock) has returned")

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
