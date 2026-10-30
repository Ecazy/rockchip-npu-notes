#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Executable SmolLM2 replay entry with fail-fast campaign ordering.

Commands
--------
reuse
    Re-verify an existing sealed V5 observation.  The expected spec file and
    canonical hashes are read from an independent hash record, never from the
    spec file being checked.
rerun-fixed-acceptance
    Re-run the V5 acceptance half on the board against the frozen V5 spec and
    frozen requests, rebuild the declared/official references, seal the
    acceptance observation and verify it.  Calibration is deliberately reused
    because this command is for implementation/timing changes, not re-policy.
campaign
    Full new campaign: tokenize/freeze inputs, generate preliminary official
    continuations, run calibration board/reference/PPL, seal calibration,
    freeze the spec, then run acceptance board/reference/PPL, seal acceptance
    and verify.  Every stage checks its predecessor and exits non-zero on
    failure.

The script never silently skips a stage.  Use --reuse-board only for an
explicit evidence-only replay where the board output is already sealed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / ".data/models/SmolLM2-135M-Instruct"
PACKAGE = ROOT / ".data/models/smollm2_135m_instruct_package"
DEFAULT_SPEC = ROOT / "models/smollm2_numeric_acceptance_v5.json"
DEFAULT_INPUTS = ROOT / "models/smollm2_v5_inputs.json"
DEFAULT_REQUESTS = ROOT / "models/smollm2_v5_requests"
DEFAULT_HASH_RECORD = ROOT / "models/specs/smollm2_spec_hashes.json"
DEFAULT_CAL_BUNDLE = ROOT / "models/specs/smollm2_v4_calibration_bundle.json"
DEFAULT_ACC_BUNDLE = ROOT / "models/specs/smollm2_v5_acceptance_bundle.json"
DEFAULT_RAW = ROOT / ".data/evidence/smollm2-20261029T120416Z"
DEFAULT_RESULTS = ROOT / "results/20261029T120416Z-smollm2-v5"
DEFAULT_PPL_WINDOWS = ROOT / "models/smollm2_v5_ppl_acceptance_windows.json"
DEFAULT_ACCEPTANCE_PROMPT = (
    "Write a short Python function that returns the largest number in a list.")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")


def canonical_json_sha(value: dict) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def run(cmd: list[str], *, log_path: Path | None = None,
        env: dict[str, str] | None = None, cwd: Path = ROOT) -> None:
    printable = " ".join(str(x) for x in cmd)
    print(f"+ {printable}", flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as log:
            log.write(f"+ {printable}\n")
            log.flush()
            proc = subprocess.run(cmd, cwd=str(cwd), env=env, text=True,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT)
            log.write(proc.stdout)
            log.flush()
    else:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, text=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
        if proc.stdout:
            print(proc.stdout, end="")
    if proc.returncode != 0:
        raise SystemExit(
            f"stage failed (rc={proc.returncode}): {printable}")


def tool(name: str, *args: str, log: Path | None = None,
         env: dict[str, str] | None = None) -> None:
    run([sys.executable, str(ROOT / "tools" / name), *args], log_path=log,
        env=env)


def board_run(request: Path, out: Path, *, runner_args: str = "",
              log: Path | None = None) -> None:
    env = os.environ.copy()
    if runner_args:
        env["RK_SMOLLM_RUNNER_ARGS"] = runner_args
    run([str(ROOT / "scripts/run-smollm2.sh"), "board-run",
         str(request), str(out)], log_path=log, env=env)


def reward_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def seal(root: Path, role: str, board_run_path: Path, out: Path,
         spec: Path | None = None, log: Path | None = None) -> None:
    args = ["--root", str(root), "--role", role,
            "--board-run", str(board_run_path), "--out", str(out)]
    if spec is not None:
        args += ["--spec", str(spec)]
    tool("seal_evidence_bundle.py", *args, log=log)


def spec_expectation(spec: Path, record: Path) -> tuple[str, str]:
    """Read expected identities from the independent record."""
    rec = read_json(record)
    key = None
    try:
        rel = spec.resolve().relative_to(ROOT.resolve()).as_posix()
        if rel in rec:
            key = rel
    except ValueError:
        pass
    if key is None:
        key = spec.name
    row = rec.get(key)
    if not isinstance(row, dict):
        raise SystemExit(f"independent spec hash record has no {key!r}")
    return row["file_sha256"], row["canonical_content_sha256"]


def check_spec_identity(spec: Path, record: Path,
                        expected_file_sha: str = "") -> dict:
    want_file, want_canonical = spec_expectation(spec, record)
    if expected_file_sha and expected_file_sha != want_file:
        raise SystemExit("caller expected spec file hash disagrees with record")
    body = read_json(spec)
    actual_file = sha(spec)
    if actual_file != want_file:
        raise SystemExit(
            f"spec file SHA-256 mismatch: expected {want_file}, got {actual_file}")
    if "canonical_content_sha256" in body:
        canonical_body = {k: v for k, v in body.items()
                          if k not in {"canonical_content_sha256",
                                       "file_sha256"}}
        actual_canonical = canonical_json_sha(canonical_body)
        if actual_canonical != want_canonical or \
                body["canonical_content_sha256"] != want_canonical:
            raise SystemExit("spec canonical-content hash mismatch")
    return {"file_sha256": actual_file,
            "canonical_content_sha256": want_canonical,
            "record": str(record.resolve())}


def load_request_prompts(request: Path) -> list[dict]:
    return read_json(request)["runs"]


def reference_set(request: Path, out_root: Path, package: Path,
                  log: Path | None = None) -> None:
    """Build declared-reference capture/TF arrays in freeze/verifier layout."""
    reward_dir(out_root)
    capture_index = 0
    tf_index = 0
    ids_root = out_root / "_ids"
    for run in load_request_prompts(request):
        mode = run.get("mode", "generate")
        if mode not in ("capture", "teacher_forcing"):
            continue
        if mode == "capture":
            out = out_root / f"capture-{capture_index}"
            capture_index += 1
            layers = ",".join(str(int(v)) for v in
                              run.get("capture_layers", [0, 15, 29]))
        else:
            out = out_root / f"tf-{tf_index}"
            tf_index += 1
            layers = ""
        ids_path = ids_root / f"{out.name}.json"
        write_json(ids_path, {"token_ids": run["token_ids"]})
        args = ["--package", str(package), "--ids-json", str(ids_path),
                "--out", str(out)]
        if layers:
            args += ["--capture-layers", layers]
        else:
            args += ["--capture-layers", ""]
        tool("smollm2_reference.py", *args, log=log)
    if capture_index == 0 or tf_index == 0:
        raise SystemExit("request set lacks capture or teacher-forcing rows")


def official_logits(prompts: list[dict], out_root: Path, model: Path,
                    log: Path | None = None) -> None:
    ids_root = out_root / "_ids"
    for prompt in prompts:
        ids_path = ids_root / f"{prompt['id']}.json"
        write_json(ids_path, {"token_ids": prompt["token_ids"]})
        tool("smollm2_official.py", "--mode", "logits",
             "--model-dir", str(model), "--ids-json", str(ids_path),
             "--out", str(out_root / prompt["id"]), log=log)


def official_ppl(windows: Path, out: Path, model: Path,
                 log: Path | None = None) -> None:
    reward_dir(out)
    tool("smollm2_official.py", "--mode", "score",
         "--model-dir", str(model), "--windows-json", str(windows),
         "--out", str(out), log=log)


def official_generations(prompts: list[dict], out_root: Path, model: Path,
                         max_new: int = 32,
                         log: Path | None = None) -> None:
    ids_root = out_root / "_ids"
    for prompt in prompts:
        ids_path = ids_root / f"{prompt['id']}.json"
        write_json(ids_path, {"token_ids": prompt["token_ids"]})
        tool("smollm2_official.py", "--mode", "generate",
             "--model-dir", str(model), "--ids-json", str(ids_path),
             "--max-new-tokens", str(max_new), "--ignore-eos",
             "--out", str(out_root / prompt["id"]), log=log)


def build_request_set(inputs_path: Path, out: Path, model: Path,
                      calibration_generations: Path,
                      numeric_generation: Path) -> Path:
    """Derive the four board request files from a frozen input manifest."""
    inp = read_json(inputs_path)
    numeric_id = inp["numeric_acceptance_prompt_id"]
    numeric_prompt = next(x for x in inp["acceptance_prompts"]
                          if x["id"] == numeric_id)
    num = read_json(numeric_generation)
    num_gen = [int(v) for v in num["generated_ids"]]
    if num.get("prompt_ids") != numeric_prompt["token_ids"]:
        raise SystemExit("numeric official generation prompt identity differs")
    need = 32
    if len(num_gen) < need:
        raise SystemExit("numeric official generation is shorter than 32")

    cal_runs = []
    for prompt in inp["calibration_prompts"]:
        cal_runs.append({
            "mode": "capture", "token_ids": prompt["token_ids"],
            "capture_layers": [0, 15, 29], "return_all_logits": True,
            "repeat": 1})
    for i, prompt in enumerate(inp["calibration_prompts"]):
        gen = read_json(calibration_generations / f"cal-{i + 1}" /
                        "result.json")
        if gen.get("prompt_ids") != prompt["token_ids"]:
            raise SystemExit(f"calibration official prompt mismatch {i}")
        continuation = [int(v) for v in gen["generated_ids"][:16]]
        if len(continuation) != 16:
            raise SystemExit("calibration official continuation too short")
        cal_runs.append({"mode": "teacher_forcing",
                         "token_ids": prompt["token_ids"] + continuation,
                         "repeat": 1})
    first_cal = inp["calibration_prompts"][0]
    first_gen = read_json(calibration_generations / "cal-1" / "result.json")
    cal_inc = [int(v) for v in first_gen["generated_ids"][:16]]
    cal_runs.append({"mode": "incremental",
                     "token_ids": first_cal["token_ids"],
                     "continuation_ids": cal_inc, "repeat": 1})

    acceptance_runs = [
        {"mode": "capture", "token_ids": numeric_prompt["token_ids"],
         "capture_layers": [0, 15, 29], "return_all_logits": True,
         "repeat": 1},
        {"mode": "teacher_forcing",
         "token_ids": numeric_prompt["token_ids"] + num_gen[:32],
         "repeat": 1},
        {"mode": "incremental", "token_ids": numeric_prompt["token_ids"],
         "continuation_ids": num_gen[:16], "repeat": 1},
    ]
    for prompt in inp["acceptance_prompts"]:
        acceptance_runs.append({
            "mode": "generate", "token_ids": prompt["token_ids"],
            "max_new_tokens": 32, "stop_on_eos": False, "repeat": 1})
    boundary = inp["boundary_cases"]
    acceptance_runs.extend([
        {"mode": "generate", "token_ids": boundary["short_1"],
         "max_new_tokens": 2, "stop_on_eos": False, "repeat": 1},
        {"mode": "generate", "token_ids": boundary["short_2"],
         "max_new_tokens": 2, "stop_on_eos": False, "repeat": 1},
    ])
    # The non-aligned boundary prompt is tokenized with the same chat template.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model))
    non_aligned = tok.apply_chat_template(
        [{"role": "user", "content": boundary["non_aligned_prompt"]}],
        tokenize=True, add_generation_prompt=True)
    acceptance_runs.append({
        "mode": "generate", "token_ids": [int(v) for v in non_aligned],
        "max_new_tokens": 1, "stop_on_eos": False, "repeat": 1})
    capacity = [int(boundary["capacity_token_id"])] * (
        int(boundary["capacity_prompt_tokens"]) - 1)
    capacity = [1] + capacity
    if len(capacity) != int(boundary["capacity_prompt_tokens"]):
        raise SystemExit("capacity boundary token construction failed")
    acceptance_runs.extend([
        {"mode": "generate", "token_ids": capacity,
         "max_new_tokens": 1, "stop_on_eos": False, "repeat": 1},
        {"mode": "expect_reject", "token_ids": capacity,
         "max_new_tokens": 2, "expected_code": "RK_E_SHAPE"},
        {"mode": "expect_reject", "token_ids": capacity + [1, 2],
         "max_new_tokens": 1, "expected_code": "RK_E_SHAPE"},
        {"mode": "expect_reject", "token_ids": [],
         "max_new_tokens": 1, "expected_code": "RK_E_SHAPE"},
        {"mode": "generate", "token_ids": numeric_prompt["token_ids"],
         "max_new_tokens": 32, "stop_on_eos": True, "repeat": 1},
    ])

    def ppl_runs(windows: dict) -> dict:
        return {"runs": [{"mode": "score", "token_ids": w,
                          "repeat": 1} for w in windows["windows"]]}

    out.mkdir(parents=True, exist_ok=True)
    files = {
        "calibration.json": {"runs": cal_runs},
        "ppl-calibration.json": ppl_runs(
            inp["calibration_nll_windows"]),
        "acceptance.json": {"runs": acceptance_runs},
        "ppl-acceptance.json": ppl_runs(inp["acceptance_nll_windows"]),
    }
    for name, obj in files.items():
        write_json(out / name, obj)
    extended = dict(inp)
    extended.pop("frozen_sha256", None)
    for key, name in (("calibration_request", "calibration.json"),
                      ("ppl_calibration_request", "ppl-calibration.json"),
                      ("acceptance_request", "acceptance.json"),
                      ("ppl_acceptance_request", "ppl-acceptance.json")):
        p = (out / name).resolve()
        extended[key] = {"path": str(p), "sha256": sha(p)}
    extended["frozen_sha256"] = canonical_json_sha(extended)
    extended_path = out / "inputs-with-requests.json"
    write_json(extended_path, extended)
    # Keep the tokenizer-produced windows next to the requests.
    write_json(out / "calibration-windows.json",
               inp["calibration_nll_windows"])
    write_json(out / "acceptance-windows.json",
               inp["acceptance_nll_windows"])
    return extended_path


def verifier_command(*, spec: Path, expected_spec_file_sha: str,
                     acceptance_root: Path, calibration_root: Path,
                     calibration_bundle: Path, acceptance_bundle: Path,
                     requests: Path, inputs_manifest: Path,
                     ppl_windows: Path, acceptance_prompt: str,
                     out: Path, execution_identity: str = "") -> list[str]:
    inp = read_json(inputs_manifest)
    numeric_id = inp["numeric_acceptance_prompt_id"]
    first_cal_id = inp["calibration_prompts"][0]["id"]
    return [
        sys.executable, str(ROOT / "tools/verify_smollm2.py"),
        "--spec", str(spec),
        "--expected-spec-file-sha256", expected_spec_file_sha,
        "--board", str(acceptance_root / "board"),
        "--reference", str(acceptance_root / "reference/capture-0"),
        "--reference-tf", str(acceptance_root / "reference/tf-0"),
        "--official-logits", str(acceptance_root / "official-logits" /
                                  numeric_id / "logits.npy"),
        "--ppl-board", str(acceptance_root / "ppl/board"),
        "--ppl-official", str(acceptance_root / "ppl/official/result.json"),
        "--ppl-windows", str(ppl_windows),
        "--acceptance-prompt", acceptance_prompt,
        "--acceptance-request", str(requests / "acceptance.json"),
        "--ppl-acceptance-request", str(requests / "ppl-acceptance.json"),
        *(["--expected-execution-identity", str(execution_identity)]
          if execution_identity else []),
        "--calibration-bundle", str(calibration_bundle),
        "--acceptance-bundle", str(acceptance_bundle),
        "--calibration-root", str(calibration_root),
        "--acceptance-root", str(acceptance_root),
        "--calibration-board", str(calibration_root / "board"),
        "--calibration-reference", str(calibration_root / "reference/capture-0"),
        "--calibration-reference-tf", str(calibration_root / "reference/tf-0"),
        "--calibration-official-logits", str(
            calibration_root / "official-logits" /
            first_cal_id / "logits.npy"),
        "--calibration-ppl-board", str(calibration_root / "ppl/board"),
        "--calibration-ppl-official", str(
            calibration_root / "ppl/official/result.json"),
        "--out", str(out),
    ]


def run_reuse(args) -> int:
    spec = Path(args.spec).resolve()
    record = Path(args.hash_record).resolve()
    identity = check_spec_identity(spec, record,
                                   args.expected_spec_file_sha256)
    raw = Path(args.raw).resolve()
    results = Path(args.results).resolve()
    results.mkdir(parents=True, exist_ok=True)
    # The archived raw tree uses the canonical layout directly.
    requests = Path(args.requests).resolve()
    ppl_windows = Path(args.ppl_windows).resolve()
    out = results / "verify/verify-summary.json"
    cmd = verifier_command(
        spec=spec, expected_spec_file_sha=identity["file_sha256"],
        acceptance_root=raw / "acceptance",
        calibration_root=raw / "calibration",
        calibration_bundle=Path(args.calibration_bundle).resolve(),
        acceptance_bundle=Path(args.acceptance_bundle).resolve(),
        requests=requests, inputs_manifest=Path(args.inputs).resolve(),
        ppl_windows=ppl_windows, acceptance_prompt=args.acceptance_prompt,
        out=out)
    run(cmd, log_path=results / "host/replay-verify.log")
    ver = read_json(out)
    if ver.get("pass") is not True or ver.get("complete_acceptance") is not True:
        raise SystemExit("reuse verifier did not produce a complete pass")
    summary = {
        "schema_version": 1,
        "kind": "smollm2-v5-reuse-replay",
        "pass": True,
        "spec": identity,
        "raw_root": str(raw),
        "results": str(results),
        "verify_summary": str(out),
        "verify_summary_sha256": sha(out),
        "verifier": {k: ver.get(k) for k in (
            "mode", "complete_acceptance", "numeric_pass", "quality_pass",
            "gate_count_expected", "gate_count_executed", "gate_failures",
            "failures")},
    }
    write_json(results / "replay-reuse.json", summary)
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


def _acceptance_stage(*, spec: Path, inputs_manifest: Path, requests: Path,
                      root: Path, results: Path, package: Path,
                      model: Path, spec_hash: str,
                      calibration_root: Path, calibration_bundle: Path,
                      ppl_windows: Path, acceptance_prompt: str,
                      reuse_board: bool, execution_identity: str = "") -> dict:
    acc = root / "acceptance"
    acc.mkdir(parents=True, exist_ok=True)
    if not reuse_board:
        board_run(requests / "acceptance.json", acc / "board",
                  log=results / "host/acceptance-board.log")
        reward_dir(acc / "ppl/board")
        board_run(requests / "ppl-acceptance.json", acc / "ppl/board",
                  log=results / "host/acceptance-ppl-board.log")
    else:
        if not (acc / "board/run.json").is_file():
            raise SystemExit("--reuse-board requested but board/run.json missing")
    reference_set(requests / "acceptance.json", acc / "reference", package,
                  log=results / "host/acceptance-reference.log")
    inp = read_json(inputs_manifest)
    numeric_id = inp["numeric_acceptance_prompt_id"]
    numeric_prompt = next(x for x in inp["acceptance_prompts"]
                          if x["id"] == numeric_id)
    official_logits([numeric_prompt], acc / "official-logits", model,
                    log=results / "host/acceptance-official-logits.log")
    official_ppl(ppl_windows, acc / "ppl/official", model,
                 log=results / "host/acceptance-official-ppl.log")
    copy_file(inputs_manifest, acc / "smollm2_inputs.json")
    copy_file(spec, acc / spec.name)
    copy_file(requests / "acceptance.json", acc / "acceptance.json")
    copy_file(requests / "ppl-acceptance.json", acc / "ppl-acceptance.json")
    seal(acc, "acceptance", acc / "board/run.json",
         root / "acceptance-bundle.json", spec=spec,
         log=results / "host/acceptance-seal.log")
    out = results / "verify/verify-summary.json"
    cmd = verifier_command(
        spec=spec, expected_spec_file_sha=spec_hash,
        acceptance_root=acc, calibration_root=calibration_root,
        calibration_bundle=calibration_bundle,
        acceptance_bundle=root / "acceptance-bundle.json",
        requests=requests, inputs_manifest=inputs_manifest,
        ppl_windows=ppl_windows, acceptance_prompt=acceptance_prompt,
        out=out, execution_identity=execution_identity)
    run(cmd, log_path=results / "host/replay-verify.log")
    ver = read_json(out)
    if ver.get("pass") is not True or ver.get("complete_acceptance") is not True:
        raise SystemExit("verifier did not produce a complete acceptance")
    return {
        "acceptance_root": str(acc),
        "acceptance_bundle": str(root / "acceptance-bundle.json"),
        "acceptance_bundle_sha256": sha(root / "acceptance-bundle.json"),
        "verify_summary": str(out),
        "verify_summary_sha256": sha(out),
        "verifier": {k: ver.get(k) for k in (
            "mode", "complete_acceptance", "numeric_pass", "quality_pass",
            "gate_count_expected", "gate_count_executed", "gate_failures",
            "failures")},
    }


def run_rerun_fixed(args) -> int:
    spec = Path(args.spec).resolve()
    record = Path(args.hash_record).resolve()
    identity = check_spec_identity(spec, record,
                                   args.expected_spec_file_sha256)
    root = Path(args.evidence_root).resolve()
    results = Path(args.results).resolve()
    root.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    result = _acceptance_stage(
        spec=spec, inputs_manifest=Path(args.inputs).resolve(),
        requests=Path(args.requests).resolve(), root=root, results=results,
        package=Path(args.package).resolve(), model=Path(args.model).resolve(),
        spec_hash=identity["file_sha256"],
        calibration_root=Path(args.calibration_root).resolve(),
        calibration_bundle=Path(args.calibration_bundle).resolve(),
        ppl_windows=Path(args.ppl_windows).resolve(),
        acceptance_prompt=args.acceptance_prompt,
        reuse_board=args.reuse_board,
        execution_identity=args.execution_identity)
    summary = {
        "schema_version": 1,
        "kind": "smollm2-v5-fixed-acceptance-rerun",
        "pass": True,
        "spec": identity,
        "calibration_reused": {
            "root": str(Path(args.calibration_root).resolve()),
            "bundle": str(Path(args.calibration_bundle).resolve()),
            "bundle_sha256": sha(Path(args.calibration_bundle).resolve()),
        },
        "acceptance": result,
        "profile_or_runner_args": os.environ.get("RK_SMOLLM_RUNNER_ARGS", ""),
    }
    write_json(results / "replay-rerun-fixed.json", summary)
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


def run_campaign(args) -> int:
    root = Path(args.evidence_root).resolve()
    results = Path(args.results).resolve()
    package = Path(args.package).resolve()
    model = Path(args.model).resolve()
    plan = Path(args.input_plan).resolve()
    root.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    if args.fetch:
        run([str(ROOT / "scripts/run-smollm2.sh"), "fetch"],
            log_path=results / "host/fetch.log")
    if args.compile:
        run([str(ROOT / "scripts/run-smollm2.sh"), "compile"],
            log_path=results / "host/compile.log")
    for p in (package / "manifest.json", model / "config.json"):
        if not p.is_file():
            raise SystemExit(f"campaign prerequisite missing: {p}")

    # 1. Frozen input preparation.
    prep = results / "inputs"
    draft_inputs = prep / "draft-inputs.json"
    draft_cal = prep / "draft-cal-windows.json"
    draft_acc = prep / "draft-acc-windows.json"
    tool("prepare_smollm2_v3_inputs.py", "--plan", str(plan),
         "--model-dir", str(model), "--out", str(draft_inputs),
         "--calibration-windows-out", str(draft_cal),
         "--acceptance-windows-out", str(draft_acc),
         log=results / "host/prepare-inputs-draft.log")
    draft = read_json(draft_inputs)
    pre = results / "official-preaccept"
    cal_prompts = draft["calibration_prompts"]
    numeric_id = draft["numeric_acceptance_prompt_id"]
    numeric_prompt = next(x for x in draft["acceptance_prompts"]
                          if x["id"] == numeric_id)
    official_generations(cal_prompts, pre, model, max_new=16,
                         log=results / "host/official-preaccept-cal.log")
    official_generations([numeric_prompt], pre, model, max_new=32,
                         log=results / "host/official-preaccept-numeric.log")
    numeric_gen = pre / numeric_id / "result.json"
    final_inputs = prep / "inputs.json"
    final_cal = results / "requests/calibration-windows.json"
    final_acc = results / "requests/acceptance-windows.json"
    tool("prepare_smollm2_v3_inputs.py", "--plan", str(plan),
         "--model-dir", str(model),
         "--official-continuation-json", str(numeric_gen),
         "--out", str(final_inputs),
         "--calibration-windows-out", str(final_cal),
         "--acceptance-windows-out", str(final_acc),
         log=results / "host/prepare-inputs-final.log")
    requests = results / "requests"
    extended_inputs = build_request_set(
        final_inputs, requests, model, pre, numeric_gen)

    # 2. Calibration -> seal -> freeze (strict order).
    cal = root / "calibration"
    cal.mkdir(parents=True, exist_ok=True)
    board_run(requests / "calibration.json", cal / "board",
              log=results / "host/calibration-board.log")
    reward_dir(cal / "ppl/board")
    board_run(requests / "ppl-calibration.json", cal / "ppl/board",
              log=results / "host/calibration-ppl-board.log")
    reference_set(requests / "calibration.json", cal / "reference", package,
                  log=results / "host/calibration-reference.log")
    cal_inp = read_json(extended_inputs)
    official_logits(cal_inp["calibration_prompts"],
                    cal / "official-logits", model,
                    log=results / "host/calibration-official-logits.log")
    official_ppl(requests / "calibration-windows.json",
                 cal / "ppl/official", model,
                 log=results / "host/calibration-official-ppl.log")
    copy_file(extended_inputs, cal / "inputs-with-requests.json")
    copy_file(plan, cal / "input-plan.json")
    for name in ("calibration.json", "ppl-calibration.json",
                 "acceptance.json", "ppl-acceptance.json"):
        copy_file(requests / name, cal / name)
    cal_bundle = root / "calibration-bundle.json"
    seal(cal, "calibration", cal / "board/run.json", cal_bundle,
         log=results / "host/calibration-seal.log")
    spec = root / "spec.json"
    tool("freeze_smollm2_v3.py", "--evidence-root", str(cal),
         "--inputs", str(extended_inputs),
         "--calibration-bundle", str(cal_bundle),
         "--out-spec", str(spec), log=results / "host/freeze-spec.log")
    spec_file_sha = sha(spec)
    spec_body = read_json(spec)
    if spec_body.get("canonical_content_sha256") != canonical_json_sha(
            {k: v for k, v in spec_body.items()
             if k not in {"canonical_content_sha256", "file_sha256"}}):
        raise SystemExit("newly frozen spec canonical hash is inconsistent")

    # 3. Acceptance -> seal -> verify.
    acc_root = root
    acc = _acceptance_stage(
        spec=spec, inputs_manifest=extended_inputs, requests=requests,
        root=acc_root, results=results, package=package, model=model,
        spec_hash=spec_file_sha, calibration_root=cal,
        calibration_bundle=cal_bundle,
        ppl_windows=requests / "acceptance-windows.json",
        acceptance_prompt=args.acceptance_prompt,
        reuse_board=False)
    summary = {
        "schema_version": 1,
        "kind": "smollm2-replay-campaign",
        "pass": True,
        "order": ["prepare-inputs", "calibration", "seal-calibration",
                  "freeze-spec", "acceptance", "seal-acceptance", "verify"],
        "spec_file_sha256": spec_file_sha,
        "spec_canonical_content_sha256":
            spec_body["canonical_content_sha256"],
        "inputs": str(extended_inputs),
        "inputs_sha256": sha(extended_inputs),
        "calibration_bundle_sha256": sha(cal_bundle),
        "acceptance": acc,
    }
    write_json(results / "replay-campaign.json", summary)
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("reuse")
    p.add_argument("--raw", default=str(DEFAULT_RAW))
    p.add_argument("--results", default=str(DEFAULT_RESULTS))
    p.add_argument("--spec", default=str(DEFAULT_SPEC))
    p.add_argument("--inputs", default=str(DEFAULT_INPUTS))
    p.add_argument("--requests", default=str(DEFAULT_REQUESTS))
    p.add_argument("--hash-record", default=str(DEFAULT_HASH_RECORD))
    p.add_argument("--expected-spec-file-sha256", default="")
    p.add_argument("--calibration-bundle", default=str(DEFAULT_CAL_BUNDLE))
    p.add_argument("--acceptance-bundle", default=str(DEFAULT_ACC_BUNDLE))
    p.add_argument("--ppl-windows", default=str(DEFAULT_PPL_WINDOWS))
    p.add_argument("--acceptance-prompt", default=DEFAULT_ACCEPTANCE_PROMPT)
    p.set_defaults(func=run_reuse)

    p = sub.add_parser("rerun-fixed-acceptance")
    p.add_argument("--evidence-root", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--package", default=str(PACKAGE))
    p.add_argument("--model", default=str(MODEL))
    p.add_argument("--spec", default=str(DEFAULT_SPEC))
    p.add_argument("--inputs", default=str(DEFAULT_INPUTS))
    p.add_argument("--requests", default=str(DEFAULT_REQUESTS))
    p.add_argument("--hash-record", default=str(DEFAULT_HASH_RECORD))
    p.add_argument("--expected-spec-file-sha256", default="")
    p.add_argument("--calibration-root", required=True)
    p.add_argument("--calibration-bundle", default=str(DEFAULT_CAL_BUNDLE))
    p.add_argument("--ppl-windows", default=str(DEFAULT_PPL_WINDOWS))
    p.add_argument("--acceptance-prompt", default=DEFAULT_ACCEPTANCE_PROMPT)
    p.add_argument("--reuse-board", action="store_true")
    p.add_argument("--execution-identity", default="")
    p.set_defaults(func=run_rerun_fixed)

    p = sub.add_parser("campaign")
    p.add_argument("--evidence-root", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--package", default=str(PACKAGE))
    p.add_argument("--model", default=str(MODEL))
    p.add_argument("--input-plan",
                   default=str(ROOT / "models/smollm2_v5_input_plan.json"))
    p.add_argument("--acceptance-prompt", default=DEFAULT_ACCEPTANCE_PROMPT)
    p.add_argument("--fetch", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.set_defaults(func=run_campaign)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
