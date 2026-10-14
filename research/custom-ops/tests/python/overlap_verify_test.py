#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Negative/positive fixtures for overlap_verify.py; no NPU required."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/python"))

from overlap_verify import SUITES, verify_root  # noqa: E402

REPS = 12
ELEMENTS = 65536  # GEMM output logical elements; GEMV is 256


def _gemv_inst(inst):
    return inst not in (0, 2)


def _submit(suite, phase, arm, rep, s, tasks):
    return {"type": "submit", "suite": suite, "arm": arm, "phase": phase, "rep": rep,
            "sub_index": s, "task_count": tasks, "read_views": 1,
            "requested_mask": 1, "actual_mask": 1, "int_status": 0x100,
            "irq_delta": [1, 0, 0], "driver_service_ns": 100000,
            "ioctl_begin_ns": 1, "ioctl_end_ns": 2, "host_begin_ns": 1, "host_done_ns": 3,
            "result": 0}


def _verify(suite, phase, arm, rep, inst, kind, written=True, guard_ok=True):
    elements = 256 if kind == "mm" and _gemv_inst(inst) else ELEMENTS
    return {"type": "verify", "suite": suite, "arm": arm, "phase": phase, "rep": rep,
            "inst": inst, "kind": kind,
            "physical_rows": 12 if kind == "mm" and _gemv_inst(inst) else 256,
            "used": 1, "written": 1 if written else 0,
            "nan": 0 if written else elements, "elements": elements,
            "bad": 0, "max_abs": 0.001, "guard_ok": 1 if guard_ok else 0, "download_result": 0}


def _pair(suite, arm, rep, t, mask, written=True):
    return {"type": "pair", "suite": suite, "arm": arm, "rep": rep, "thread": t,
            "requested_mask": mask, "actual_mask": mask, "int_status": 0x100,
            "driver_service_ns": 100000, "ioctl_begin_ns": 1, "ioctl_end_ns": 2,
            "written": 1 if written else 0, "bad": 0, "max_abs": 0.001, "result": 0}


def _window(suite, arm, rep, masks):
    irq = [2, 0, 0] if masks == [1, 1] else [1, 1, 0]
    return {"type": "pair_window", "suite": suite, "arm": arm, "rep": rep,
            "t_pair_ns": 150000, "sum_service_ns": 200000,
            "service_overlap_ns": 50000, "irq_delta": irq}


def records(suite):
    arms = SUITES[suite]["arms"]
    kinds = SUITES[suite]["kinds"]
    out = []
    for arm, spec in arms.items():
        for rep in range(REPS):
            if spec.get("pair"):
                for t in (0, 1):
                    out.append(_pair(suite, arm, rep, t, spec["masks"][t]))
                out.append(_window(suite, arm, rep, spec["masks"]))
                continue
            unwritten = set(spec.get("unwritten", []))
            timeout = arm.startswith("S3_")
            used = []
            for s, sub in enumerate(spec["subs"]):
                sub_rec = _submit(suite, "main", arm, rep, s, len(sub))
                if timeout:
                    sub_rec.update(result=-110, int_status=0, driver_service_ns=0)
                out.append(sub_rec)
                used.extend(sub)
            for inst in sorted(set(used)):
                kind = kinds.get(inst, "mm")
                out.append(_verify(suite, "main", arm, rep, inst, kind,
                                   written=inst not in unwritten))
    out.append({"type": "summary", "suite": suite, "samples": REPS, "warmups": 8,
                "seed": 1, "checks": 100, "failures": 0, "explore": False, "status": "PASS"})
    return out


def write_run(root, name, recs):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "probe.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


class OverlapVerifyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, recs):
        write_run(self.root, "run", recs)

    def _find(self, recs, **match):
        for r in recs:
            if all(r.get(k) == v for k, v in match.items()):
                return r
        raise AssertionError("no record %r" % (match,))

    def expect_pass(self, recs):
        self._write(recs)
        report = verify_root(str(self.root))
        self.assertEqual(report["status"], "PASS", report["errors"])

    def expect_fail(self, recs, needle):
        self._write(recs)
        report = verify_root(str(self.root))
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any(needle in e for e in report["errors"]), report["errors"])

    # -- positive --------------------------------------------------------
    def test_valid_matmul_lut(self):
        self.expect_pass(records("matmul-lut"))

    def test_valid_gemm_gemv(self):
        self.expect_pass(records("gemm-gemv"))

    def test_backward_compatible_records(self):
        # Old evidence has no suite/phase/samples fields.
        recs = records("matmul-lut")
        for r in recs:
            r.pop("suite", None)
            r.pop("phase", None)
            if r["type"] == "submit":
                r["submit_index"] = r.pop("sub_index")
        self._find(recs, type="summary")["reps"] = REPS
        self.expect_pass(recs)

    # -- submit integrity ------------------------------------------------
    def test_wrong_mask(self):
        recs = records("gemm-gemv")
        self._find(recs, type="submit", arm="A")["actual_mask"] = 2
        self.expect_fail(recs, "Core 0")

    def test_bad_result(self):
        recs = records("gemm-gemv")
        self._find(recs, type="submit", arm="C_AB")["result"] = -1
        self.expect_fail(recs, "result")

    def test_zero_service(self):
        recs = records("gemm-gemv")
        self._find(recs, type="submit", arm="B")["driver_service_ns"] = 0
        self.expect_fail(recs, "service")

    def test_missing_completion_bit(self):
        recs = records("gemm-gemv")
        self._find(recs, type="submit", arm="B16")["int_status"] = 0
        self.expect_fail(recs, "completion")

    def test_wrong_irq_delta(self):
        recs = records("gemm-gemv")
        self._find(recs, type="submit", arm="C_AB8")["irq_delta"] = [0, 1, 0]
        self.expect_fail(recs, "Core 0 IRQ")

    def test_wrong_task_count(self):
        recs = records("gemm-gemv")
        self._find(recs, type="submit", arm="C_AB8")["task_count"] = 8
        self.expect_fail(recs, "task_count")

    def test_missing_submit(self):
        recs = [r for r in records("gemm-gemv")
                if not (r.get("type") == "submit" and r.get("arm") == "C_BA")]
        self.expect_fail(recs, "expected 1")

    def test_summary_status(self):
        recs = records("gemm-gemv")
        self._find(recs, type="summary")["status"] = "FAIL"
        self.expect_fail(recs, "status")

    # -- verify integrity ------------------------------------------------
    def test_used_instance_unwritten(self):
        recs = records("gemm-gemv")
        v = self._find(recs, type="verify", arm="C_AB", inst=1)
        v["written"], v["nan"] = 0, v["elements"]
        self.expect_fail(recs, "expected written")

    def test_used_instance_bad(self):
        recs = records("gemm-gemv")
        self._find(recs, type="verify", arm="A", inst=0)["bad"] = 3
        self.expect_fail(recs, "bad=")

    def test_over_tolerance(self):
        recs = records("gemm-gemv")
        self._find(recs, type="verify", arm="B", inst=1)["max_abs"] = 1.0
        self.expect_fail(recs, "above")

    def test_guard_failed(self):
        recs = records("gemm-gemv")
        self._find(recs, type="verify", arm="B8", inst=4)["guard_ok"] = 0
        self.expect_fail(recs, "guard failed")

    def test_unused_touched(self):
        recs = records("gemm-gemv")
        recs.append(_verify("gemm-gemv", "main", "A", 0, 19, "mm", written=True))
        self.expect_fail(recs, "touched though unused")

    def test_missing_verify(self):
        recs = [r for r in records("gemm-gemv")
                if not (r.get("type") == "verify" and r.get("arm") == "C_BB" and r.get("inst") == 3)]
        self.expect_fail(recs, "missing verify")

    # -- explore timeout -------------------------------------------------
    def test_s3_timeout_with_service_rejected(self):
        recs = records("matmul-lut")
        self._find(recs, type="submit", arm="S3_A_CORE")["driver_service_ns"] = 5
        self.expect_fail(recs, "timeout carries")

    def test_non_explore_timeout_rejected(self):
        recs = records("gemm-gemv")
        self._find(recs, type="submit", arm="A")["result"] = -110
        self.expect_fail(recs, "result")

    # -- pair groups -----------------------------------------------------
    def test_pair_wrong_mask(self):
        recs = records("gemm-gemv")
        self._find(recs, type="pair", arm="P_cross", thread=1)["actual_mask"] = 1
        self.expect_fail(recs, "expected 2")

    def test_pair_missing_thread(self):
        recs = [r for r in records("gemm-gemv")
                if not (r.get("type") == "pair" and r.get("arm") == "P_same" and r.get("thread") == 1)]
        self.expect_fail(recs, "expected 2")

    def test_pair_bad_output(self):
        recs = records("gemm-gemv")
        self._find(recs, type="pair", arm="P_cross", thread=0)["bad"] = 1
        self.expect_fail(recs, "bad=")

    def test_pair_wrong_irq(self):
        recs = records("gemm-gemv")
        self._find(recs, type="pair_window", arm="P_cross")["irq_delta"] = [2, 0, 0]
        self.expect_fail(recs, "irq_delta")

    def test_pair_missing_window(self):
        recs = [r for r in records("gemm-gemv")
                if not (r.get("type") == "pair_window" and r.get("arm") == "P_same")]
        self.expect_fail(recs, "missing pair_window")

    # -- files -----------------------------------------------------------
    def test_missing_file(self):
        report = verify_root(str(self.root))
        self.assertEqual(report["status"], "FAIL")

    def test_malformed_json(self):
        self._write(records("gemm-gemv"))
        p = self.root / "run" / "probe.jsonl"
        p.write_text(p.read_text() + "{not json\n")
        self.assertEqual(verify_root(str(self.root))["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
