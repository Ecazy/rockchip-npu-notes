#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run the verifier/summarizer negative matrix and emit an audit JSON."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))

import test_verify_smollm2 as fixture  # noqa: E402
import summarize_smollm2_evidence as summary  # noqa: E402
import verify_smollm2 as verifier  # noqa: E402


def run_case(name, mutate, *, numeric_only=True, complete=False,
             expect_exit_zero=False):
    with tempfile.TemporaryDirectory() as tmp:
        case = fixture.SyntheticAcceptance(Path(tmp))
        mutate(case)
        args = case.args(numeric_only=numeric_only, complete=complete)
        try:
            rc = verifier.main(args)
            result = json.loads((case.root / "verify.json").read_text())
            row = {"case": name, "exit_code": rc,
                   "overall_pass": result.get("pass"),
                   "gate_failures": result.get("gate_failures"),
                   "failures": result.get("failures"),
                   "complete_acceptance": result.get("complete_acceptance")}
        except SystemExit as exc:
            row = {"case": name, "exit_code": int(exc.code or 0),
                   "overall_pass": None, "gate_failures": [],
                   "failures": ["command_rejected"],
                   "complete_acceptance": False}
        row["expected_failure"] = not expect_exit_zero
        row["ok"] = (not expect_exit_zero and row["exit_code"] != 0 and
                     row["overall_pass"] is not True) or \
                    (expect_exit_zero and row["exit_code"] == 0 and
                     row["overall_pass"] is True)
        return row


def main(argv=None) -> int:
    rows = []
    rows.append(run_case(
        "baseline_numeric_only", lambda c: None, expect_exit_zero=True))

    def numeric_error(c):
        c.logits_val[0, 0] = 0.5
        import numpy as np
        np.save(c.board / "run-00" / "all-logits-00.npy", c.logits_val)
        c.write_spec(gates={
            "logits.full": {"atol": 0.0, "rmse_limit": 0.0,
                            "finite_required": True}})
    rows.append(run_case("numeric_error_top1_unchanged", numeric_error))

    def rank_error(c):
        import numpy as np
        c.logits_val[0, 3] = 0.0
        c.logits_val[0, 2] = 0.01
        np.save(c.board / "run-00" / "all-logits-00.npy", c.logits_val)
    rows.append(run_case("rank_gate_failure", rank_error))

    def missing_gate(c):
        c.write_spec(gates={"missing.tensor": {
            "atol": 0.1, "rmse_limit": 0.1, "finite_required": True}})
    rows.append(run_case("missing_capture_gate", missing_gate))

    def shape_error(c):
        import numpy as np
        np.savez(c.ref / "captured.npz",
                 x=np.zeros((2, 1), dtype=np.float16))
        np.savez(c.board / "run-00" / "captured-00.npz",
                 x=np.zeros((2, 3), dtype=np.float16))
    rows.append(run_case("broadcastable_shape_mismatch", shape_error))

    def nonfinite_obs(c):
        import numpy as np
        bad = c.x_val.copy()
        bad[0, 0] = np.nan
        np.savez(c.board / "run-00" / "captured-00.npz", x=bad)
    rows.append(run_case("nonfinite_observation", nonfinite_obs))

    def nonfinite_ref(c):
        import numpy as np
        bad = c.x_ref.copy()
        bad[0, 0] = np.nan
        np.savez(c.ref / "captured.npz", x=bad)
    rows.append(run_case("nonfinite_reference", nonfinite_ref))

    def bad_threshold(c):
        c.write_spec(gates={"x": {"atol": float("nan"),
                                  "rmse_limit": 0.1,
                                  "finite_required": True}})
    rows.append(run_case("nan_threshold", bad_threshold))

    def package_mismatch(c):
        run = json.loads((c.board / "run.json").read_text())
        run["environment"]["package_manifest_sha256"] = "0" * 64
        fixture.write_json(c.board / "run.json", run)
    rows.append(run_case("package_identity_mismatch", package_mismatch))

    def token_mismatch(c):
        run = json.loads((c.board / "run.json").read_text())
        run["requests"][0]["token_ids"] = [7, 8]
        fixture.write_json(c.board / "run.json", run)
    rows.append(run_case("input_token_mismatch", token_mismatch))

    rows.append(run_case(
        "complete_missing_quality", lambda c: None,
        numeric_only=False, complete=False))

    def quality_nonfinite(c):
        import numpy as np
        bad = np.asarray([np.nan, 1.0], dtype=np.float64)
        np.save(c.ppl_board / "run-00" / "nll-00.npy", bad)
        c.seal("calibration", c.cal_manifest)
        c.seal("acceptance", c.acc_manifest)
    rows.append(run_case(
        "quality_nonfinite", quality_nonfinite,
        numeric_only=False, complete=True))

    def quality_window_mismatch(c):
        result_path = c.ppl_board / "run-00" / "result.json"
        row = json.loads(result_path.read_text())
        row["token_ids"] = [7, 8, 9]
        row["runs"][0]["token_ids"] = [7, 8, 9]
        fixture.write_json(result_path, row)
        c.seal("calibration", c.cal_manifest)
        c.seal("acceptance", c.acc_manifest)
    rows.append(run_case(
        "quality_window_token_mismatch", quality_window_mismatch,
        numeric_only=False, complete=True))

    def quality_count_mismatch(c):
        result_path = c.ppl_official / "result.json"
        row = json.loads(result_path.read_text())
        row["windows"][0]["scored_tokens"] = 1
        fixture.write_json(result_path, row)
        c.seal("calibration", c.cal_manifest)
        c.seal("acceptance", c.acc_manifest)
    rows.append(run_case(
        "quality_scored_token_count_mismatch", quality_count_mismatch,
        numeric_only=False, complete=True))

    # Summarizer rejection.
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "verify.json"
        p.write_text(json.dumps({
            "mode": "complete", "complete_acceptance": True,
            "quality_pass": True, "pass": False,
            "gate_failures": ["logits.full"], "failures": [],
            "calibration_bundle": {"pass": True},
            "acceptance_bundle": {"pass": True},
        }) + "\n")
        try:
            summary.require_complete_verifier(p)
            srow = {"case": "summarizer_rejects_failed_verifier",
                    "exit_code": 0, "ok": False}
        except SystemExit as exc:
            try:
                code = int(exc.code or 0)
            except (TypeError, ValueError):
                code = 1
            srow = {"case": "summarizer_rejects_failed_verifier",
                    "exit_code": code,
                    "overall_pass": None,
                    "failures": [str(exc)], "ok": code != 0}
        rows.append(srow)
    out = ROOT / "results/20261029T120416Z-smollm2-v5/verify/negative-matrix.json"
    out.write_text(json.dumps({"cases": rows,
                               "all_ok": all(r["ok"] for r in rows)},
                              indent=1, sort_keys=True) + "\n")
    print(json.dumps({"all_ok": all(r["ok"] for r in rows),
                      "out": str(out), "cases": len(rows)}, indent=1))
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
