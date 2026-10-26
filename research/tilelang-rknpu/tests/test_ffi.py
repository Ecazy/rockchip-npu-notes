import ctypes
import unittest

from rknpu_backend import (CompileOptions, compile_rkir, default_profile,
                           matmul_module)


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
            ffi.RKPublishDesc: 32,
            ffi.RKPlanDesc: 168,
            ffi.RKBinding: 48,
            ffi.RKRunReport: 88,
            ffi.RKAttribution: 160,
            ffi.RKGroupReport: 176,
            ffi.RKDiag: 760,
        }
        for struct, size in expected.items():
            self.assertEqual(ctypes.sizeof(struct), size, struct.__name__)

    def test_long_k_chain_is_accepted_by_the_c_verifier(self):
        """Regression: plans with >=3 K tiles were once rejected.

        The accumulator ping-pong writes the same BO several times, so a
        "single writer per buffer" rule rejected every chain longer than two
        tiles.  The verifier now requires the *latest* producer before each read
        to be a declared dependency.
        """
        try:
            from rknpu_backend.runtime.ffi import default_library
            lib = default_library()
        except Exception as exc:
            self.skipTest(f"rknpu_plan shared library not built: {exc}")
        for k in (256, 257, 512, 513, 768, 1024, 4096):
            plan = compile_rkir(
                matmul_module(m=13, k=k, n=33),
                profile=default_profile(),
                options=CompileOptions(numeric_mode="rk_f16_tiled")).plan
            cplan = lib.create_plan(plan)
            cplan.destroy()
            self.assertGreaterEqual(plan.accounting.task_count,
                                    (k + 255) // 256)

    def test_ctypes_layout_matches_the_c_header_exactly(self):
        """Total sizes and every field offset/size are compared with C ``offsetof``.

        The field set must match exactly: a missing, extra, swapped or resized
        field fails, even when the total struct size happens to be unchanged.
        """
        import os
        import subprocess
        from pathlib import Path

        from rknpu_backend.runtime import ffi

        binary = os.environ.get("RK_PLAN_CONTRACT_BIN")
        if not binary:
            candidate = (Path(__file__).resolve().parents[2] / "custom-ops" /
                         "build-host-mvp" / "rk_plan_contract_test")
            binary = str(candidate) if candidate.is_file() else ""
        if not binary or not Path(binary).is_file():
            self.skipTest("rk_plan_contract_test binary not built")
        out = subprocess.run([binary], capture_output=True, text=True, check=True)
        c_sizes, c_fields = {}, {}
        for line in out.stdout.splitlines():
            parts = line.split()
            if parts[0] == "ABI_SIZE":
                c_sizes[parts[1]] = int(parts[2])
            elif parts[0] == "ABI_FIELD":
                c_fields.setdefault(parts[1], {})[parts[2]] = (int(parts[3]),
                                                               int(parts[4]))
        self.assertTrue(c_sizes, "C layout dump missing ABI_SIZE lines")
        self.assertTrue(c_fields, "C layout dump missing ABI_FIELD lines")
        structs = {
            "rk_value_desc": ffi.RKValueDesc,
            "rk_tensor_desc": ffi.RKTensorDesc,
            "rk_buffer_desc": ffi.RKBufferDesc,
            "rk_view_desc": ffi.RKViewDesc,
            "rk_action_desc": ffi.RKActionDesc,
            "rk_submit_desc": ffi.RKSubmitDesc,
            "rk_publish_desc": ffi.RKPublishDesc,
            "rk_plan_desc": ffi.RKPlanDesc,
            "rk_binding": ffi.RKBinding,
            "rk_run_report": ffi.RKRunReport,
            "rk_run_attribution": ffi.RKAttribution,
            "rk_group_report": ffi.RKGroupReport,
            "rk_diag": ffi.RKDiag,
        }
        self.assertEqual(set(structs), set(c_sizes),
                         "C and Python disagree on the struct set")
        for name, cls in structs.items():
            self.assertEqual(ctypes.sizeof(cls), c_sizes[name], f"{name} size")
            python_fields = {field[0] for field in cls._fields_}
            self.assertEqual(python_fields, set(c_fields[name]),
                             f"{name} field set differs between C and ctypes")
            for field, (offset, size) in c_fields[name].items():
                descriptor = getattr(cls, field)
                self.assertEqual(descriptor.offset, offset,
                                 f"{name}.{field} offset")
                self.assertEqual(descriptor.size, size, f"{name}.{field} size")

    def test_co_program_bytes_matches_the_c_seam(self):
        """The Host requirement formula mirrors ``sizeof(co_program)``.

        ``passes/storage.py`` uses ``CO_PROGRAM_BYTES`` to size the per-submit
        emission array in the plan's declared Host requirement, and the C verifier
        adds the same term when it recomputes that requirement.  The two must
        agree, so the Python constant is pinned to the C ABI dump here.
        """
        import os
        import subprocess
        from pathlib import Path

        from rknpu_backend.passes import storage

        binary = os.environ.get("RK_PLAN_CONTRACT_BIN")
        if not binary:
            candidate = (Path(__file__).resolve().parents[2] / "custom-ops" /
                         "build-host-mvp" / "rk_plan_contract_test")
            binary = str(candidate) if candidate.is_file() else ""
        if not binary or not Path(binary).is_file():
            self.skipTest("rk_plan_contract_test binary not built")
        out = subprocess.run([binary], capture_output=True, text=True, check=True)
        consts = {}
        for line in out.stdout.splitlines():
            parts = line.split()
            if parts and parts[0] == "ABI_CONST":
                consts[parts[1]] = int(parts[2])
        self.assertEqual(consts.get("co_program"), storage.CO_PROGRAM_BYTES)

    def test_repeat_group_admission_domain_matches_the_c_verifier(self):
        """Python and C must gate the independent-MM group on identical limits.

        Structure independence is a necessary condition, not a validation domain:
        both sides accept only the geometry and task count the board probe
        covered.  Pinning the Python constants to the C header keeps the two from
        drifting apart, which is exactly how an unmeasured 64-task or 8x8x8 chain
        could otherwise slip through one of them.
        """
        import os
        import subprocess
        from pathlib import Path

        from rknpu_backend.passes import repeat_expand

        binary = os.environ.get("RK_PLAN_CONTRACT_BIN")
        if not binary:
            candidate = (Path(__file__).resolve().parents[2] / "custom-ops" /
                         "build-host-mvp" / "rk_plan_contract_test")
            binary = str(candidate) if candidate.is_file() else ""
        if not binary or not Path(binary).is_file():
            self.skipTest("rk_plan_contract_test binary not built")
        out = subprocess.run([binary], capture_output=True, text=True, check=True)
        consts = {}
        for line in out.stdout.splitlines():
            parts = line.split()
            if parts and parts[0] == "ABI_CONST":
                consts[parts[1]] = int(parts[2])
        self.assertEqual(consts.get("independent_mm_tasks"),
                         repeat_expand.MAX_GROUP_TASKS)
        self.assertEqual(consts.get("independent_mm_rows"), repeat_expand.GROUP_M)
        self.assertEqual(consts.get("independent_mm_cols"), repeat_expand.GROUP_N)
        self.assertEqual(consts.get("independent_mm_k"), repeat_expand.GROUP_K)
        self.assertEqual(consts.get("independent_mm_phys_cols"),
                         repeat_expand.GROUP_PHYS_COLS)

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
