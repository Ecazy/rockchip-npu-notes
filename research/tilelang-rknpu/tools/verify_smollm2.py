#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Audit a frozen SmolLM2 numeric/quality acceptance observation.

Fail-closed properties:
  * every declared gate is executed; expected and executed gate sets must match;
  * array shapes must be exactly equal (no NumPy broadcasting);
  * observations, references, thresholds and quality summaries must be finite;
  * any numeric/rank gate failure makes ``pass`` false and the CLI non-zero;
  * a complete acceptance requires quality evidence;
  * ``--numeric-only`` is marked explicitly and can never be summarized as a
    complete acceptance.

The tool does not recompute thresholds.  A spec's canonical-content hash and
file SHA-256 are kept as separate identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

SPECIAL_GATES = {
    "logits.full",
    "official.logits",
    "teacher_forcing.logits",
    "incremental_vs_full_prefix.logits",
}
SHARD_RE = re.compile(r"^logits\.shard(\d+)$")
CAPTURE_SPECIAL = SPECIAL_GATES


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_sha(obj: dict) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def spec_hashes(spec: dict) -> dict:
    if "canonical_content_sha256" in spec:
        body = {k: v for k, v in spec.items()
                if k not in {"canonical_content_sha256", "file_sha256"}}
        frozen = spec["canonical_content_sha256"]
        scheme = "canonical_content_v1"
    elif "spec_sha256" in spec:
        body = {k: v for k, v in spec.items() if k != "spec_sha256"}
        frozen = spec["spec_sha256"]
        scheme = "legacy_spec_sha256_v1"
    else:
        body = dict(spec)
        frozen = None
        scheme = "missing"
    return {"frozen": frozen, "recomputed": canonical_json_sha(body),
            "scheme": scheme, "body": body}


def _finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and \
        math.isfinite(float(value))


def validate_gate_definition(gate: dict) -> list[str]:
    errors: list[str] = []
    if not isinstance(gate, dict):
        return ["gate_definition_not_object"]
    for key in ("atol", "rmse_limit"):
        value = gate.get(key)
        if not _finite_number(value) or float(value) < 0:
            errors.append(f"{key}_invalid")
    for key in ("top1_rate_min", "top5_overlap_mean_min"):
        if key in gate:
            value = gate[key]
            if not _finite_number(value):
                errors.append(f"{key}_invalid")
    return errors


def metrics(value: np.ndarray, ref: np.ndarray) -> dict:
    b = value.astype(np.float32)
    r = ref.astype(np.float32)
    e = b - r
    flat_b = b.reshape(-1, b.shape[-1]) if b.ndim > 1 else b.reshape(1, -1)
    flat_r = r.reshape(-1, r.shape[-1]) if r.ndim > 1 else r.reshape(1, -1)
    top1_b = np.argmax(flat_b, axis=-1)
    top1_r = np.argmax(flat_r, axis=-1)
    top5_b = np.argsort(flat_b, axis=-1)[:, -5:]
    top5_r = np.argsort(flat_r, axis=-1)[:, -5:]
    overlaps = [len(set(x.tolist()) & set(y.tolist()))
                for x, y in zip(top5_b, top5_r)]
    return {
        "finite": bool(np.isfinite(b).all() and np.isfinite(r).all()),
        "max_abs": float(np.max(np.abs(e))) if e.size else 0.0,
        "rmse": float(np.sqrt(np.mean(np.square(e)))) if e.size else 0.0,
        "ref_absmax": float(np.max(np.abs(r))) if r.size else 0.0,
        "ref_rms": float(np.sqrt(np.mean(np.square(r)))) if r.size else 0.0,
        "top1_matches": int(np.sum(top1_b == top1_r)),
        "positions": int(top1_b.size),
        "top1_rate": float(np.mean(top1_b == top1_r)) if top1_b.size else 0.0,
        "top5_overlap_mean": float(np.mean(overlaps)) if overlaps else 0.0,
        "top5_overlap_min": int(min(overlaps)) if overlaps else 0,
    }


def check_gate(value: np.ndarray | None, ref: np.ndarray | None,
               gate: dict) -> dict:
    definition_errors = validate_gate_definition(gate)
    if definition_errors:
        return {"pass": False, "failures": definition_errors,
                "observed": None, "limits": gate,
                "shape": {"value": None, "reference": None}}
    if value is None or ref is None:
        return {"pass": False, "failures": ["missing_observation"],
                "observed": None, "limits": gate,
                "shape": {"value": None, "reference": None}}
    value = np.asarray(value)
    ref = np.asarray(ref)
    shape = {"value": list(value.shape), "reference": list(ref.shape)}
    failures: list[str] = []
    if value.shape != ref.shape:
        failures.append("shape_mismatch")
        return {"pass": False, "failures": failures,
                "observed": {"finite": bool(np.isfinite(value).all() and
                                            np.isfinite(ref).all()),
                             "value_shape": list(value.shape),
                             "reference_shape": list(ref.shape)},
                "limits": gate, "shape": shape}
    observed = metrics(value, ref)
    if not observed["finite"]:
        failures.append("nonfinite_observation_or_reference")
    else:
        if observed["max_abs"] > float(gate["atol"]):
            failures.append("max_abs")
        if observed["rmse"] > float(gate["rmse_limit"]):
            failures.append("rmse")
        if "top1_rate_min" in gate and \
                observed["top1_rate"] < float(gate["top1_rate_min"]):
            failures.append("top1_rate")
        if "top5_overlap_mean_min" in gate and \
                observed["top5_overlap_mean"] < \
                float(gate["top5_overlap_mean_min"]):
            failures.append("top5_overlap")
    return {"pass": not failures, "failures": failures,
            "observed": observed, "limits": gate, "shape": shape}


def _row_by_mode(rows: list[dict], mode_names: tuple[str, ...]) -> dict | None:
    for row in rows:
        if row.get("mode") in mode_names:
            return row
    return None


def _check_token_sequence(name: str, actual, expected, failures: list[str],
                          details: dict) -> None:
    actual_list = [int(v) for v in (actual or [])]
    expected_list = [int(v) for v in (expected or [])]
    equal = actual_list == expected_list
    details[name] = {
        "match": equal,
        "actual_sha256": hashlib.sha256(
            np.asarray(actual_list, dtype="<i8").tobytes()).hexdigest(),
        "expected_sha256": hashlib.sha256(
            np.asarray(expected_list, dtype="<i8").tobytes()).hexdigest(),
        "actual_tokens": len(actual_list),
        "expected_tokens": len(expected_list),
    }
    if not equal:
        failures.append(name)


def _validate_input_identity(spec: dict, board_meta: dict, rmeta: dict,
                             reftf_meta: dict, official_meta: dict,
                             failures: list[str]) -> dict:
    rows = board_meta.get("requests", [])
    capture = _row_by_mode(rows, ("capture", "prefill"))
    tf_row = _row_by_mode(rows, ("teacher_forcing",))
    inc_row = _row_by_mode(rows, ("incremental",))
    details: dict = {}
    if capture is None or tf_row is None or inc_row is None:
        failures.append("required_board_request_mode_missing")
        return details
    _check_token_sequence("capture_token_ids_vs_reference",
                          capture.get("token_ids"), rmeta.get("token_ids"),
                          failures, details)
    _check_token_sequence("teacher_forcing_token_ids_vs_reference",
                          tf_row.get("token_ids"), reftf_meta.get("token_ids"),
                          failures, details)
    _check_token_sequence("incremental_prompt_ids_vs_reference",
                          inc_row.get("token_ids"), rmeta.get("token_ids"),
                          failures, details)
    inc_run = (inc_row.get("runs") or [{}])[0]
    inc_actual = [int(v) for v in inc_run.get("continuation_ids", [])]
    tf_tail = list(reftf_meta.get("token_ids", []))[
        len(rmeta.get("token_ids", [])):]
    identity = spec.get("input_identity") or {}
    if identity.get("incremental_continuation_ids") is not None:
        expected_inc = [int(v) for v in identity[
            "incremental_continuation_ids"]]
    elif identity:
        expected_inc = tf_tail
    else:
        # Legacy observations compared only the decoded prefix.
        expected_inc = tf_tail[:len(inc_actual)]
    _check_token_sequence(
        "incremental_continuation_ids_vs_reference",
        inc_actual, expected_inc, failures, details)
    _check_token_sequence("official_logits_token_ids_vs_reference",
                          official_meta.get("token_ids"), rmeta.get("token_ids"),
                          failures, details)

    identity = spec.get("input_identity")
    if identity:
        numeric_ids = identity.get("numeric_acceptance_prompt_ids")
        tf_ids = identity.get("teacher_forcing_token_ids")
        inc_cont = identity.get("incremental_continuation_ids")
        request_sha = identity.get("acceptance_request_sha256")
        _check_token_sequence("capture_token_ids_vs_frozen_input",
                              capture.get("token_ids"), numeric_ids,
                              failures, details)
        _check_token_sequence("teacher_forcing_token_ids_vs_frozen_input",
                              tf_row.get("token_ids"), tf_ids,
                              failures, details)
        _check_token_sequence("incremental_continuation_vs_frozen_input",
                              inc_run.get("continuation_ids"), inc_cont,
                              failures, details)
        if request_sha:
            actual_request = board_meta.get("request_sha256")
            details["request_sha256"] = {
                "match": actual_request == request_sha,
                "actual": actual_request, "expected": request_sha}
            if actual_request != request_sha:
                failures.append("acceptance_request_sha256")
        # Full-sequence independence: do not reuse a prompt or a text-window
        # identity between calibration and acceptance.
        calibration = identity.get("calibration", {})
        acceptance = identity.get("acceptance", {})
        calib_prompts = calibration.get("prompt_token_ids", [])
        accept_prompts = acceptance.get("prompt_token_ids", [])
        calib_set = {tuple(int(v) for v in seq) for seq in calib_prompts}
        accept_set = {tuple(int(v) for v in seq) for seq in accept_prompts}
        details["prompt_sequence_overlap"] = len(calib_set & accept_set)
        if calib_set & accept_set:
            failures.append("calibration_acceptance_prompt_reuse")
        calib_windows = {
            tuple(int(v) for v in w.get("token_ids", []))
            for w in calibration.get("nll_windows", [])}
        accept_windows = {
            tuple(int(v) for v in w.get("token_ids", []))
            for w in acceptance.get("nll_windows", [])}
        details["nll_window_overlap"] = len(calib_windows & accept_windows)
        if calib_windows & accept_windows:
            failures.append("calibration_acceptance_nll_window_reuse")
    else:
        calib_ids = list(spec.get("calibration", {}).get("token_ids", []))
        if [int(v) for v in rmeta.get("token_ids", [])] == calib_ids:
            failures.append("calibration_acceptance_sequence_collision")
    return details


def load_board_ppl(board_dir: Path, windows_json: dict) -> dict:
    expected = windows_json.get("window_meta")
    if expected is None:
        expected = [
            {"window": i, "token_ids": [int(v) for v in window],
             "scored_label_ids": [int(v) for v in window[1:]],
             "scored_tokens": len(window) - 1}
            for i, window in enumerate(windows_json.get("windows", []))]
    if not isinstance(expected, list) or not expected:
        raise ValueError("windows JSON lacks non-empty window metadata")
    result_files = sorted(board_dir.glob("run-*/result.json"))
    if len(result_files) != len(expected):
        raise ValueError(
            f"board PPL has {len(result_files)} windows, expected {len(expected)}")
    rows = []
    nlls = []
    for index, (path, meta) in enumerate(zip(result_files, expected)):
        row_data = json.loads(path.read_text())
        payload = row_data["runs"][0]
        token_ids = [int(v) for v in (
            payload.get("token_ids") or row_data.get("token_ids") or [])]
        expected_ids = [int(v) for v in meta.get("token_ids", [])]
        if token_ids != expected_ids:
            raise ValueError(f"board PPL window {index} token ids differ")
        labels = [int(v) for v in (
            payload.get("scored_label_ids") or expected_ids[1:])]
        if labels != expected_ids[1:]:
            raise ValueError(f"board PPL window {index} scored labels differ")
        nll_path = path.parent / f"nll-{0:02d}.npy"
        if not nll_path.is_file():
            nll_paths = sorted(path.parent.glob("nll-*.npy"))
            if len(nll_paths) != 1:
                raise ValueError(f"board PPL window {index} NLL file missing")
            nll_path = nll_paths[0]
        nll = np.load(nll_path, allow_pickle=False).astype(np.float64)
        if nll.ndim != 1 or nll.size != len(labels):
            raise ValueError(f"board PPL window {index} NLL shape mismatch")
        if not np.isfinite(nll).all():
            raise ValueError(f"board PPL window {index} contains non-finite NLL")
        if payload.get("nll_sha256") and \
                payload["nll_sha256"] != hashlib.sha256(
                    nll.tobytes()).hexdigest():
            raise ValueError(f"board PPL window {index} NLL hash mismatch")
        nlls.append(nll)
        rows.append({
            "window": index,
            "token_ids": token_ids,
            "token_seq_sha256": hashlib.sha256(
                np.asarray(token_ids, dtype="<i8").tobytes()).hexdigest(),
            "scored_label_ids": labels,
            "scored_label_sha256": hashlib.sha256(
                np.asarray(labels, dtype="<i8").tobytes()).hexdigest(),
            "scored_tokens": int(nll.size),
            "sum_nll": float(nll.sum()),
            "mean_nll": float(nll.mean()),
            "ppl": float(math.exp(float(nll.mean()))),
            "top1_accuracy": float(payload.get("top1_accuracy", float("nan"))),
        })
    nll = np.concatenate(nlls)
    return {
        "scored_tokens": int(nll.size),
        "sum_nll": float(nll.sum()),
        "mean_nll": float(nll.mean()),
        "ppl": float(math.exp(float(nll.mean()))),
        "top1_accuracy": float(np.mean(
            [r["top1_accuracy"] for r in rows])),
        "windows": rows,
        "nll_sha256": hashlib.sha256(nll.tobytes()).hexdigest(),
    }


def load_official_ppl(path: Path, windows_json: dict) -> dict:
    result = json.loads(path.read_text())
    expected = windows_json.get("window_meta")
    if expected is None:
        expected = [
            {"window": i, "token_ids": [int(v) for v in window],
             "scored_label_ids": [int(v) for v in window[1:]],
             "scored_tokens": len(window) - 1}
            for i, window in enumerate(windows_json.get("windows", []))]
    if not isinstance(expected, list) or not expected:
        raise ValueError("windows JSON lacks non-empty window metadata")
    rows = result.get("windows")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("official PPL window count mismatch")
    nll_path = path.parent / "nll.npy"
    if not nll_path.is_file():
        raise ValueError("official PPL nll.npy is missing")
    nll_raw = np.load(nll_path, allow_pickle=False)
    if result.get("nll_sha256") and \
            result["nll_sha256"] != hashlib.sha256(
                nll_raw.tobytes()).hexdigest():
        raise ValueError("official PPL nll hash mismatch")
    nll_file = nll_raw.astype(np.float64)
    if not np.isfinite(nll_file).all():
        raise ValueError("official PPL nll.npy contains non-finite values")
    scored = 0
    checked_rows = []
    for index, (row, meta) in enumerate(zip(rows, expected)):
        token_ids = [int(v) for v in row.get("token_ids", [])]
        expected_ids = [int(v) for v in meta.get("token_ids", [])]
        if token_ids != expected_ids:
            raise ValueError(f"official PPL window {index} token ids differ")
        labels = [int(v) for v in row.get("scored_label_ids", [])]
        if labels != expected_ids[1:]:
            raise ValueError(f"official PPL window {index} labels differ")
        values = {k: row.get(k) for k in
                  ("scored_tokens", "sum_nll", "mean_nll", "ppl",
                   "top1_accuracy")}
        if any(not _finite_number(v) for v in values.values()):
            raise ValueError(f"official PPL window {index} non-finite summary")
        if int(values["scored_tokens"]) != len(labels):
            raise ValueError(f"official PPL window {index} scored count mismatch")
        scored += int(values["scored_tokens"])
        checked_rows.append({k: values[k] for k in values})
    if scored != nll_file.size:
        raise ValueError("official PPL aggregate NLL length mismatch")
    aggregate = {
        "scored_tokens": int(nll_file.size),
        "sum_nll": float(nll_file.sum()),
        "mean_nll": float(nll_file.mean()),
        "ppl": float(math.exp(float(nll_file.mean()))),
        "top1_accuracy": float(np.mean(
            [r["top1_accuracy"] for r in checked_rows])),
        "windows": checked_rows,
        "nll_sha256": hashlib.sha256(nll_file.tobytes()).hexdigest(),
    }
    for key in ("scored_tokens", "sum_nll", "mean_nll", "ppl",
                "top1_accuracy"):
        if key in result and _finite_number(result[key]):
            if key == "scored_tokens" and int(result[key]) != aggregate[key]:
                raise ValueError("official PPL aggregate scored_tokens mismatch")
    return aggregate


def validate_quality(board: dict, official: dict, limits: dict) -> dict:
    failures: list[str] = []
    for key in ("scored_tokens_min", "mean_nll_delta_max", "ppl_ratio_max",
                "top1_accuracy_delta_min"):
        if not _finite_number(limits.get(key)):
            failures.append(f"quality_limit_{key}_invalid")
    if failures:
        return {"pass": False, "failures": failures, "limits": limits}
    if board["scored_tokens"] != official["scored_tokens"]:
        failures.append("quality_scored_token_count_mismatch")
    if board["windows"] != official["windows"]:
        # Per-window numeric metrics must not be compared here because NLL
        # values differ; identity was already checked separately.  Keep this
        # field for callers but do not force byte equality.
        pass
    delta = float(board["mean_nll"]) - float(official["mean_nll"])
    ratio = float(board["ppl"]) / float(official["ppl"])
    top1_delta = float(board["top1_accuracy"]) - float(
        official["top1_accuracy"])
    if not all(math.isfinite(v) for v in (delta, ratio, top1_delta)):
        failures.append("quality_metric_nonfinite")
    if board["scored_tokens"] < int(limits["scored_tokens_min"]):
        failures.append("quality_scored_tokens")
    if abs(delta) > float(limits["mean_nll_delta_max"]):
        failures.append("quality_mean_nll_delta")
    if abs(ratio - 1.0) > float(limits["ppl_ratio_max"]) - 1.0:
        failures.append("quality_ppl_ratio")
    if top1_delta < float(limits["top1_accuracy_delta_min"]):
        failures.append("quality_top1_delta")
    return {"pass": not failures, "failures": failures,
            "mean_nll_delta": delta, "ppl_ratio": ratio,
            "top1_accuracy_delta": top1_delta, "limits": limits}


def _manifest_label_path(label: str, roots: dict) -> Path | None:
    if label.startswith("board/"):
        return roots["board"] / label[len("board/"):]
    if label.startswith("reference/"):
        return roots["reference"] / label[len("reference/"):]
    if label.startswith("reference_tf/"):
        return roots["reference_tf"] / label[len("reference_tf/"):]
    if label.startswith("official/"):
        return roots["official"] / label[len("official/"):]
    if label.startswith("ppl_board/"):
        root = roots.get("ppl_board")
        return (root / label[len("ppl_board/"):]) if root else None
    if label.startswith("ppl_official/"):
        root = roots.get("ppl_official")
        return (root / label[len("ppl_official/"):]) if root else None
    return None


def validate_observation_bundle(path: Path, role: str, roots: dict,
                                spec: dict, calib: dict, spec_path: Path,
                                failures: list[str]) -> dict:
    result = {"present": True, "role": role, "path": str(path)}
    try:
        manifest = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001 - converted to audit failure
        failures.append(f"observation_manifest_unreadable:{role}:{exc}")
        return {"present": True, "role": role, "pass": False}
    if manifest.get("kind") != "smollm2-observation-bundle" or \
            manifest.get("role") != role:
        failures.append(f"observation_manifest_identity:{role}")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        failures.append(f"observation_manifest_empty:{role}")
        files = {}
    checked = 0
    for label, entry in sorted(files.items()):
        actual_path = _manifest_label_path(label, roots)
        if actual_path is None or not actual_path.is_file():
            failures.append(f"observation_manifest_missing:{role}:{label}")
            continue
        expected = entry.get("sha256") if isinstance(entry, dict) else None
        if not expected or sha_file(actual_path) != expected:
            failures.append(f"observation_manifest_hash:{role}:{label}")
        checked += 1
    result["files_checked"] = checked
    result["file_sha256"] = sha_file(path)
    if manifest.get("package_manifest_sha256") != calib.get(
            "package_manifest_sha256"):
        failures.append(f"observation_package_manifest_identity:{role}")
    if manifest.get("package_weights_sha256") != calib.get(
            "package_weights_sha256"):
        failures.append(f"observation_package_weights_identity:{role}")
    if role == "acceptance":
        frozen = spec.get("canonical_content_sha256") or spec.get("spec_sha256")
        actual = manifest.get("spec_canonical_content_sha256") or \
            manifest.get("spec_legacy_sha256")
        if frozen != actual:
            failures.append("acceptance_manifest_spec_identity")
        if manifest.get("spec_file_sha256") and \
                manifest["spec_file_sha256"] != sha_file(spec_path):
            failures.append("acceptance_manifest_spec_file_sha256")
    if role == "calibration":
        calibration_hashes = {
            "board/run.json": "board_run_sha256",
            "board/stats.json": "board_stats_sha256",
            "board/run-00/captured-00.npz": "board_captured_sha256",
            "board/run-00/all-logits-00.npy": "board_logits_sha256",
            "board/run-01/logits-00.npy": "board_teacher_forcing_sha256",
            "board/run-02/incremental-logits-00.npy":
                "board_incremental_sha256",
            "reference/meta.json": "reference_meta_sha256",
            "reference/logits.npy": "reference_logits_sha256",
            "reference_tf/logits.npy": "reference_tf_logits_sha256",
            "official/logits.npy": "official_logits_sha256",
        }
        for label, key in calibration_hashes.items():
            if key not in calib:
                continue
            entry = files.get(label)
            expected = entry.get("sha256") if isinstance(entry, dict) else None
            if expected != calib[key]:
                failures.append(f"calibration_manifest_spec_hash:{key}")
    result["pass"] = not any(
        item.startswith("observation_") or
        item.startswith("calibration_manifest_") or
        item == "acceptance_manifest_spec_identity" for item in failures)
    return result


def validate_generic_bundle(path: Path, role: str, root: Path, spec: dict,
                            calib: dict, spec_path: Path,
                            failures: list[str]) -> dict:
    result = {"present": True, "role": role, "path": str(path),
              "root": str(root)}
    try:
        manifest = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        failures.append(f"generic_bundle_unreadable:{role}:{exc}")
        return {**result, "pass": False}
    if manifest.get("kind") != "smollm2-evidence-bundle" or \
            manifest.get("role") != role:
        failures.append(f"generic_bundle_identity:{role}")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        failures.append(f"generic_bundle_empty:{role}")
        files = {}
    checked = 0
    for label, entry in sorted(files.items()):
        actual = root / label
        if not actual.is_file():
            failures.append(f"generic_bundle_missing:{role}:{label}")
            continue
        expected = entry.get("sha256") if isinstance(entry, dict) else None
        if not expected or sha_file(actual) != expected:
            failures.append(f"generic_bundle_hash:{role}:{label}")
        checked += 1
    if manifest.get("package_manifest_sha256") != calib.get(
            "package_manifest_sha256"):
        failures.append(f"generic_bundle_package_manifest:{role}")
    if manifest.get("package_weights_sha256") != calib.get(
            "package_weights_sha256"):
        failures.append(f"generic_bundle_package_weights:{role}")
    if role == "acceptance":
        frozen = spec.get("canonical_content_sha256") or spec.get("spec_sha256")
        actual = manifest.get("spec_canonical_content_sha256") or \
            manifest.get("spec_legacy_sha256")
        if frozen != actual:
            failures.append("generic_acceptance_bundle_spec")
        if manifest.get("spec_file_sha256") and \
                manifest["spec_file_sha256"] != sha_file(spec_path):
            failures.append("generic_acceptance_bundle_spec_file")
    else:
        expected_file = calib.get("observation_bundle_sha256")
        if expected_file and sha_file(path) != expected_file:
            failures.append("generic_calibration_bundle_file_sha256")
    result.update({"files_checked": checked,
                   "file_sha256": sha_file(path),
                   "pass": not any(
                       item.startswith("generic_bundle") or
                       item.startswith("generic_acceptance_bundle") or
                       item.startswith("generic_calibration_bundle")
                       for item in failures)})
    return result


def check_execution_identity(expected: dict, env: dict, board_meta: dict) -> list[str]:
    """Compare the expected effective execution identity with the board record."""
    failures: list[str] = []
    if expected.get("kind") != "smollm2-effective-execution-identity":
        return ["execution_identity_kind"]
    base = expected.get("base_package", {})
    if base.get("manifest_sha256") != env.get("package_manifest_sha256"):
        failures.append("execution_identity_package_manifest")
    if base.get("weights_sha256") != env.get("package_weights_sha256"):
        failures.append("execution_identity_package_weights")
    if expected.get("profile_hash") != env.get("profile_hash"):
        failures.append("execution_identity_profile")
    if expected.get("numeric_mode") != env.get("numeric_mode"):
        failures.append("execution_identity_numeric_mode")
    if expected.get("plan_library_sha256") != env.get("plan_lib_sha256"):
        failures.append("execution_identity_plan_library")
    actual_policy = board_meta.get("schedule_policy") or {}
    expected_overrides = expected.get("overrides") or {}
    actual_overrides = actual_policy.get("overrides") or {}
    if expected.get("schedule_policy_sha256") != actual_policy.get("sha256"):
        failures.append("execution_identity_policy_hash")
    if set(expected_overrides) != set(actual_overrides):
        failures.append("execution_identity_override_keys")
    for key in sorted(set(expected_overrides) & set(actual_overrides)):
        want = expected_overrides[key]
        have = actual_overrides[key]
        for field in ("replacement_kind", "base_plan_hash", "plan_hash",
                      "plan_sha256", "task_count", "cache_key"):
            if want.get(field) != have.get(field):
                failures.append(f"execution_identity_override_{key}_{field}")
    expected_loaded = {key: value.get("plan_hash")
                       for key, value in expected_overrides.items()}
    actual_loaded = actual_policy.get("loaded_plan_hashes") or {}
    if expected_loaded != actual_loaded:
        failures.append("execution_identity_loaded_plan_hashes")
    if not expected_overrides and actual_policy.get("override_keys"):
        failures.append("execution_identity_unexpected_override")
    return failures


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--board", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--reference-tf", required=True)
    ap.add_argument("--official-logits", required=True)
    ap.add_argument("--ppl-board", default="")
    ap.add_argument("--ppl-official", default="")
    ap.add_argument("--ppl-windows", default="")
    ap.add_argument("--acceptance-prompt", default="")
    ap.add_argument("--acceptance-request", default="")
    ap.add_argument("--ppl-acceptance-request", default="")
    ap.add_argument("--calibration-manifest", default="")
    ap.add_argument("--acceptance-manifest", default="")
    ap.add_argument("--calibration-bundle", default="")
    ap.add_argument("--acceptance-bundle", default="")
    ap.add_argument("--calibration-root", default="")
    ap.add_argument("--acceptance-root", default="")
    ap.add_argument("--calibration-board", default="")
    ap.add_argument("--calibration-reference", default="")
    ap.add_argument("--calibration-reference-tf", default="")
    ap.add_argument("--calibration-official-logits", default="")
    ap.add_argument("--calibration-ppl-board", default="")
    ap.add_argument("--calibration-ppl-official", default="")
    ap.add_argument("--expected-spec-file-sha256", default="")
    ap.add_argument("--expected-execution-identity", default="")
    ap.add_argument("--numeric-only", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    spec_path = Path(args.spec).resolve()
    spec = json.loads(spec_path.read_text())
    hashes = spec_hashes(spec)
    board = Path(args.board).resolve()
    refdir = Path(args.reference).resolve()
    reftf = Path(args.reference_tf).resolve()
    bcap = np.load(board / "run-00" / "captured-00.npz", allow_pickle=False)
    rcap = np.load(refdir / "captured.npz", allow_pickle=False)
    blog = np.load(board / "run-00" / "all-logits-00.npy",
                   allow_pickle=False)
    rlog = np.load(refdir / "logits.npy", allow_pickle=False)
    btf = np.load(board / "run-01" / "logits-00.npy", allow_pickle=False)
    rtf = np.load(reftf / "logits.npy", allow_pickle=False)
    inc = np.load(board / "run-02" / "incremental-logits-00.npy",
                  allow_pickle=False)
    official = np.load(args.official_logits, allow_pickle=False)
    rmeta = json.loads((refdir / "meta.json").read_text())
    reftf_meta = json.loads((reftf / "meta.json").read_text())
    official_meta = json.loads(
        (Path(args.official_logits).parent / "result.json").read_text()
        if (Path(args.official_logits).parent / "result.json").is_file()
        else "{}")
    board_meta = json.loads((board / "run.json").read_text())

    failures: list[str] = []
    if hashes["frozen"] is None or hashes["frozen"] != hashes["recomputed"]:
        failures.append("spec_canonical_content_sha256")
    if args.expected_spec_file_sha256 and \
            sha_file(spec_path) != args.expected_spec_file_sha256:
        failures.append("spec_file_sha256")
    if args.acceptance_request:
        request = json.loads(Path(args.acceptance_request).read_text())
        request_sha = sha_file(Path(args.acceptance_request))
        if board_meta.get("request_sha256") != request_sha:
            failures.append("acceptance_request_file_sha256")
        if spec.get("input_identity", {}).get("acceptance_request_sha256") and \
                spec["input_identity"]["acceptance_request_sha256"] != request_sha:
            failures.append("acceptance_request_vs_frozen_spec")

    env = board_meta.get("environment", {})
    if args.expected_execution_identity:
        expected_identity = json.loads(
            Path(args.expected_execution_identity).read_text())
        failures.extend(check_execution_identity(
            expected_identity, env, board_meta))
    calib = spec.get("calibration", {})
    if env.get("package_manifest_sha256") != calib.get(
            "package_manifest_sha256"):
        failures.append("package_manifest_identity")
    if env.get("package_weights_sha256") != calib.get(
            "package_weights_sha256"):
        failures.append("package_weights_identity")
    if rmeta.get("package_manifest_sha256") != calib.get(
            "package_manifest_sha256"):
        failures.append("reference_package_manifest_identity")
    if rmeta.get("weights_file_sha256") != calib.get(
            "package_weights_sha256"):
        failures.append("reference_package_weights_identity")
    if official_meta and official_meta.get("weights_sha256") and \
            spec.get("official_reference", {}).get("weights_sha256") and \
            official_meta["weights_sha256"] != \
            spec["official_reference"]["weights_sha256"]:
        failures.append("official_weights_identity")
    if official_meta and official_meta.get("config_sha256") and \
            spec.get("official_reference", {}).get("config_sha256") and \
            official_meta["config_sha256"] != \
            spec["official_reference"]["config_sha256"]:
        failures.append("official_config_identity")

    observation_roots = {
        "board": board,
        "reference": refdir,
        "reference_tf": reftf,
        "official": Path(args.official_logits).parent,
        "ppl_board": Path(args.ppl_board).resolve() if args.ppl_board else None,
        "ppl_official": (Path(args.ppl_official).resolve().parent
                         if args.ppl_official else None),
    }
    calibration_roots = dict(observation_roots)
    if args.calibration_board:
        calibration_roots["board"] = Path(args.calibration_board).resolve()
    if args.calibration_reference:
        calibration_roots["reference"] = Path(
            args.calibration_reference).resolve()
    if args.calibration_reference_tf:
        calibration_roots["reference_tf"] = Path(
            args.calibration_reference_tf).resolve()
    if args.calibration_official_logits:
        calibration_roots["official"] = Path(
            args.calibration_official_logits).resolve().parent
    if args.calibration_ppl_board:
        calibration_roots["ppl_board"] = Path(
            args.calibration_ppl_board).resolve()
    if args.calibration_ppl_official:
        calibration_roots["ppl_official"] = Path(
            args.calibration_ppl_official).resolve().parent
    calibration_manifest = None
    acceptance_manifest = None
    calibration_bundle = None
    acceptance_bundle = None
    if args.calibration_manifest:
        calibration_manifest = validate_observation_bundle(
            Path(args.calibration_manifest).resolve(), "calibration",
            calibration_roots, spec, calib, spec_path, failures)
    if args.acceptance_manifest:
        acceptance_manifest = validate_observation_bundle(
            Path(args.acceptance_manifest).resolve(), "acceptance",
            observation_roots, spec, calib, spec_path, failures)
    if args.calibration_bundle:
        if not args.calibration_root:
            failures.append("calibration_bundle_root_missing")
        else:
            calibration_bundle = validate_generic_bundle(
                Path(args.calibration_bundle).resolve(), "calibration",
                Path(args.calibration_root).resolve(), spec, calib, spec_path,
                failures)
    if args.acceptance_bundle:
        if not args.acceptance_root:
            failures.append("acceptance_bundle_root_missing")
        else:
            acceptance_bundle = validate_generic_bundle(
                Path(args.acceptance_bundle).resolve(), "acceptance",
                Path(args.acceptance_root).resolve(), spec, calib, spec_path,
                failures)

    expected_gates = set(spec.get("expected_gates", spec["gates"].keys()))
    if set(spec["gates"].keys()) != expected_gates:
        failures.append("spec_gate_set_definition")
    for name in expected_gates:
        if name not in spec["gates"]:
            failures.append(f"gate_definition_missing:{name}")

    input_details = _validate_input_identity(
        spec, board_meta, rmeta, reftf_meta, official_meta, failures)

    n = int(rmeta.get("tokens", 0))
    if n <= 0 or blog.ndim != 2 or rlog.ndim != 2:
        failures.append("logits_rank_or_token_count")
    if blog.shape[0] < n or rlog.shape[0] < n:
        failures.append("logits_token_count")
    blog_n = blog[:n]
    rlog_n = rlog[:n]
    if official.shape != rlog_n.shape:
        failures.append("official_logits_shape")
    if btf.shape != rtf.shape:
        failures.append("teacher_forcing_shape")
    if inc.ndim != 2 or inc.shape[1] != blog.shape[1]:
        failures.append("incremental_shape")
    prompt_len = int(rmeta.get("tokens", 0))
    inc_row = _row_by_mode(board_meta.get("requests", []), ("incremental",))
    if inc_row is not None:
        prompt_len = int(inc_row.get("token_count", prompt_len))
    tf_slice = btf[prompt_len - 1:prompt_len - 1 + inc.shape[0]]
    if tf_slice.shape != inc.shape:
        failures.append("incremental_tf_slice_shape")

    capture_expected = {
        name for name in expected_gates
        if name not in CAPTURE_SPECIAL and not SHARD_RE.match(name)}
    capture_actual = set(bcap.files)
    if capture_actual != capture_expected:
        failures.append("capture_tensor_set")
    if set(rcap.files) != capture_expected:
        failures.append("reference_capture_tensor_set")

    gates: dict[str, dict] = {}
    for name in sorted(capture_expected):
        gates[name] = check_gate(
            bcap[name] if name in bcap.files else None,
            rcap[name] if name in rcap.files else None,
            spec["gates"].get(name, {}))
    if "logits.full" in expected_gates:
        gates["logits.full"] = check_gate(
            blog_n, rlog_n, spec["gates"]["logits.full"])
    if "official.logits" in expected_gates:
        gates["official.logits"] = check_gate(
            blog_n, official, spec["gates"]["official.logits"])
    if "teacher_forcing.logits" in expected_gates:
        gates["teacher_forcing.logits"] = check_gate(
            btf, rtf, spec["gates"]["teacher_forcing.logits"])
    if "incremental_vs_full_prefix.logits" in expected_gates:
        gates["incremental_vs_full_prefix.logits"] = check_gate(
            inc, tf_slice, spec["gates"]["incremental_vs_full_prefix.logits"])

    shard_indices = sorted(int(SHARD_RE.match(name).group(1))
                           for name in expected_gates
                           if SHARD_RE.match(name))
    if shard_indices:
        if shard_indices != list(range(len(shard_indices))):
            failures.append("lm_head_shard_gate_not_contiguous")
        shards = len(shard_indices)
        vocab = int(blog_n.shape[1])
        width = (vocab + shards - 1) // shards
        for j in shard_indices:
            name = f"logits.shard{j:02d}"
            start, stop = j * width, min(vocab, (j + 1) * width)
            gates[name] = check_gate(
                blog_n[:, start:stop], rlog_n[:, start:stop],
                spec["gates"][name])
    executed_gates = set(gates)
    if executed_gates != expected_gates:
        failures.append("executed_gate_set")

    gate_failures = [name for name, row in gates.items() if not row["pass"]]
    if gate_failures:
        failures.append("numeric_gate_failures")

    # Diagnostic teacher-forcing sequence check: a top-1 divergence from the
    # official continuation is quality information, not a compiler gate.
    tf_ids = [int(v) for v in reftf_meta.get("token_ids", [])]
    continuation = tf_ids[prompt_len:]
    tf_pred = ([int(v) for v in np.argmax(btf[prompt_len - 1:], axis=1)]
               if btf.ndim == 2 and btf.shape[0] >= prompt_len else [])
    ncheck = min(len(continuation), len(tf_pred))
    first_tf_div = next((i for i in range(ncheck)
                         if tf_pred[i] != continuation[i]), None)
    inc_pred = ([int(v) for v in np.argmax(inc, axis=1)]
                if inc.ndim == 2 else [])
    full_pred = ([int(v) for v in np.argmax(tf_slice, axis=1)]
                 if tf_slice.ndim == 2 else [])
    if inc.ndim == 2 and tf_slice.shape == inc.shape and inc_pred != full_pred:
        failures.append("incremental_top1_vs_full_prefix")

    quality = None
    quality_pass = None
    if args.numeric_only:
        complete_acceptance = False
        mode = "numeric-only"
    else:
        if not (args.ppl_board and args.ppl_official and args.ppl_windows):
            ap.error("complete acceptance requires --ppl-board, "
                     "--ppl-official and --ppl-windows; use --numeric-only "
                     "for an explicitly incomplete numeric check")
        if not ((args.calibration_manifest or args.calibration_bundle) and
                (args.acceptance_manifest or args.acceptance_bundle)):
            ap.error("complete acceptance requires sealed calibration and "
                     "acceptance observation bundles/manifests")
        if not (args.calibration_board and args.calibration_reference and
                args.calibration_reference_tf and
                args.calibration_official_logits and
                args.calibration_ppl_board and args.calibration_ppl_official):
            ap.error("complete acceptance requires the full calibration "
                     "board/reference/ppl paths")
        windows = json.loads(Path(args.ppl_windows).read_text())
        ppl_run_path = Path(args.ppl_board) / "run.json"
        if ppl_run_path.is_file():
            ppl_run = json.loads(ppl_run_path.read_text())
            expected_ppl_sha = spec.get("input_identity", {}).get(
                "ppl_acceptance_request_sha256")
            if expected_ppl_sha and \
                    ppl_run.get("request_sha256") != expected_ppl_sha:
                failures.append("ppl_acceptance_request_sha256")
        if args.ppl_acceptance_request:
            req_sha = sha_file(Path(args.ppl_acceptance_request))
            expected_ppl_sha = spec.get("input_identity", {}).get(
                "ppl_acceptance_request_sha256")
            if expected_ppl_sha and req_sha != expected_ppl_sha:
                failures.append("ppl_acceptance_request_file_sha256")
        quality_spec = spec.get("quality")
        if not isinstance(quality_spec, dict) or \
                not isinstance(quality_spec.get("limits"), dict):
            failures.append("quality_spec_missing")
            quality = {"pass": False, "failures": ["quality_spec_missing"]}
        else:
            try:
                bp = load_board_ppl(Path(args.ppl_board), windows)
                op = load_official_ppl(Path(args.ppl_official), windows)
                quality = validate_quality(bp, op, quality_spec["limits"])
                quality["board"] = bp
                quality["official"] = op
                quality["windows_identity"] = "exact_token_ids_and_scored_labels"
            except Exception as exc:  # noqa: BLE001 - converted to audit failure
                quality = {"pass": False,
                           "failures": [f"quality_evidence_invalid:{exc}"]}
        quality_pass = bool(quality.get("pass"))
        if not quality_pass:
            failures.append("quality_gate_failures")
        complete_acceptance = True
        mode = "complete"

    result = {
        "kind": "smollm2-acceptance-verification",
        "mode": mode,
        "complete_acceptance": complete_acceptance,
        "pass": not failures,
        "numeric_pass": not failures or (
            failures == ["quality_gate_failures"]),
        "quality_pass": quality_pass,
        "spec": str(spec_path),
        "spec_hash_scheme": hashes["scheme"],
        "spec_canonical_content_sha256_frozen": hashes["frozen"],
        "spec_canonical_content_sha256_recomputed": hashes["recomputed"],
        "spec_file_sha256": sha_file(spec_path),
        "expected_spec_file_sha256": args.expected_spec_file_sha256 or None,
        "acceptance_prompt": args.acceptance_prompt,
        "acceptance_prompt_tokens": n,
        "calibration_prompt_tokens": len(calib.get("token_ids", [])),
        "input_identity": input_details,
        "calibration_manifest": calibration_manifest,
        "acceptance_manifest": acceptance_manifest,
        "calibration_bundle": calibration_bundle,
        "acceptance_bundle": acceptance_bundle,
        "package_manifest_sha256": env.get("package_manifest_sha256"),
        "package_weights_sha256": env.get("package_weights_sha256"),
        "gate_count_expected": len(expected_gates),
        "gate_count_executed": len(executed_gates),
        "gate_failures": gate_failures,
        "failures": failures,
        "teacher_forcing": {
            "positions_checked": ncheck,
            "first_divergence_index": first_tf_div,
            "top1_matches": sum(a == b for a, b in
                                zip(tf_pred, continuation)),
            "top1_rate": (sum(a == b for a, b in
                              zip(tf_pred, continuation)) / ncheck
                          if ncheck else 0.0),
        },
        "incremental": {
            "positions_checked": int(inc.shape[0]) if inc.ndim == 2 else 0,
            "top1_equal_to_full_prefix": (
                inc.ndim == 2 and tf_slice.shape == inc.shape and
                inc_pred == full_pred),
            "predicted_ids": inc_pred,
        },
        "gates": gates,
        "quality": quality,
        "observation_hashes": {
            "board_captured": sha_file(board / "run-00" / "captured-00.npz"),
            "board_logits": sha_file(board / "run-00" / "all-logits-00.npy"),
            "board_tf": sha_file(board / "run-01" / "logits-00.npy"),
            "board_incremental": sha_file(board / "run-02" /
                                           "incremental-logits-00.npy"),
            "reference_logits": sha_file(refdir / "logits.npy"),
            "reference_tf": sha_file(reftf / "logits.npy"),
            "official_logits": sha_file(Path(args.official_logits)),
        },
    }
    # A gate failure must never be hidden by another failure bucket.
    if gate_failures and "numeric_gate_failures" not in result["failures"]:
        result["failures"].append("numeric_gate_failures")
    result["pass"] = not result["failures"]
    result["numeric_pass"] = not (
        set(result["failures"]) - {"quality_gate_failures"})
    result["quality_pass"] = quality_pass
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({
        "pass": result["pass"],
        "mode": result["mode"],
        "complete_acceptance": result["complete_acceptance"],
        "numeric_pass": result["numeric_pass"],
        "quality_pass": result["quality_pass"],
        "gate_count_expected": result["gate_count_expected"],
        "gate_count_executed": result["gate_count_executed"],
        "gate_failures": result["gate_failures"],
        "failures": result["failures"],
        "incremental_top1_equal": result["incremental"][
            "top1_equal_to_full_prefix"],
        "teacher_forcing_first_divergence": result[
            "teacher_forcing"]["first_divergence_index"],
    }, indent=1))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
