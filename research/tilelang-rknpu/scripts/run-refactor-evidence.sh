#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Collect the refactor evidence bundle described in refactor-specific.md section 14.
#
#   results/<stamp>-refactor/
#     manifest.json              source/dependency identity, commands, exit codes
#     baseline/projection.json   representative plan+descriptor projection at BASE_REF
#     current/projection.json    the same projection for the working tree
#     comparison.json            field-by-field diff, explicit ignore list
#     native-ir-evaluation.json  native carrier coverage, contracts, adoption numbers
#     logs/                      raw command output
#
# The baseline is a throw-away git worktree of the recorded base commit, so the
# comparison never depends on a hand-copied oracle.  Nothing here modifies the
# working tree, and no dependency is installed or upgraded.
set -uo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(cd "$HERE/../.." && pwd)
BASE_REF=${RK_REFACTOR_BASE_REF:-}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT_ROOT=${1:-$HERE/results/$STAMP-refactor}
DEPS=${RK_REFACTOR_DEPS:-$HERE/.pydeps}

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/baseline" "$OUT_ROOT/current"
LOG="$OUT_ROOT/logs"
status_file="$OUT_ROOT/.status"
: >"$status_file"

record() { # name exit_code
  printf '%s\t%s\n' "$1" "$2" >>"$status_file"
}

run() { # label command...
  local label=$1
  shift
  echo "== $label: $*" >&2
  "$@" >"$LOG/$label.log" 2>&1
  local rc=$?
  record "$label" "$rc"
  echo "   rc=$rc" >&2
  return $rc
}

HEAD_SHA=$(git -C "$REPO" rev-parse HEAD)
if [ -z "$BASE_REF" ]; then
  # Default: the commit this refactor started from, as recorded in git history.
  BASE_REF=$(git -C "$REPO" rev-list --max-parents=0 HEAD | head -n1)
  BASE_REF=${RK_REFACTOR_BASE_REF_DEFAULT:-$BASE_REF}
fi

echo "evidence -> $OUT_ROOT" >&2
echo "HEAD=$HEAD_SHA BASE_REF=$BASE_REF" >&2

WORKTREE=$(mktemp -d -t rk-refactor-baseline-XXXXXX)
cleanup() {
  git -C "$REPO" worktree remove --force "$WORKTREE" >/dev/null 2>&1 || true
  rm -rf "$WORKTREE"
}
trap cleanup EXIT

git -C "$REPO" worktree add --detach "$WORKTREE" "$BASE_REF" >"$LOG/worktree.log" 2>&1
record worktree $?

BASE_TREE="$WORKTREE/research/tilelang-rknpu"

# --- 1. baseline and current projections -----------------------------------
( cd "$BASE_TREE" && PYTHONPATH="$BASE_TREE/python:$DEPS:$BASE_TREE" \
    python3 "$HERE/tools/refactor_plan_projection.py" \
      --out "$OUT_ROOT/baseline/projection.json" ) >"$LOG/baseline_projection.log" 2>&1
record baseline_projection $?

( cd "$HERE" && PYTHONPATH="$HERE/python:$DEPS:$HERE" \
    python3 "$HERE/tools/refactor_plan_projection.py" \
      --out "$OUT_ROOT/current/projection.json" ) >"$LOG/current_projection.log" 2>&1
record current_projection $?

# --- 2. field-by-field comparison ------------------------------------------
python3 "$HERE/tools/compare_plan_projections.py" \
  --baseline "$OUT_ROOT/baseline/projection.json" \
  --current "$OUT_ROOT/current/projection.json" \
  --out "$OUT_ROOT/comparison.json" >"$LOG/comparison.log" 2>&1
record comparison $?

# --- 3. native carrier evaluation ------------------------------------------
( cd "$HERE" && PYTHONPATH="$HERE/python:$DEPS:$HERE" \
    python3 "$HERE/tools/evaluate_native_graph.py" \
      --out "$OUT_ROOT/native-ir-evaluation.json" ) \
    >"$LOG/native_ir_evaluation.log" 2>&1
record native_ir_evaluation $?

# --- 3b. optional: full model-package regeneration --------------------------
# Off by default because it rewrites tracked files under models/compiled (the
# campaign tools work in place).  When enabled it snapshots the package, recompiles
# every model, compares content and restores the tracked files, so the only lasting
# effect is model-package-regeneration.json.
if [ "${RK_REFACTOR_WITH_MODELS:-0}" = "1" ]; then
  MODEL_DIR="$HERE/models/compiled"
  if [ -n "$(git -C "$REPO" status --short -- "$MODEL_DIR" 2>/dev/null)" ]; then
    echo "models/compiled has uncommitted changes; skipping regeneration" >&2
    record model_package_regeneration 99
  else
    SNAPSHOT=$(mktemp -d -t rk-refactor-models-XXXXXX)
    cp -r "$MODEL_DIR/." "$SNAPSHOT/"
    ( cd "$HERE" && PYTHONPATH="$HERE:$HERE/python:$DEPS" python3 tools/compile_models.py ) \
      >"$LOG/model_compile.log" 2>&1
    record model_compile $?
    python3 "$HERE/tools/compare_compiled_models.py" \
      --reference "$SNAPSHOT" --candidate "$MODEL_DIR" --source-root "$HERE" \
      --out "$OUT_ROOT/model-package-regeneration.json" \
      >"$LOG/model_package_regeneration.log" 2>&1
    record model_package_regeneration $?
    git -C "$REPO" checkout -- "$MODEL_DIR" >>"$LOG/model_restore.log" 2>&1
    record model_restore $?
    rm -rf "$SNAPSHOT"
  fi
fi

# --- 4. test suites ---------------------------------------------------------
( cd "$HERE" && PYTHONPATH="$DEPS:$HERE/python:$HERE" \
    python3 -m pytest -q -o addopts="" ) >"$LOG/pytest.log" 2>&1
record pytest $?

( ctest --test-dir "$HERE/../custom-ops/build-host-mvp" --output-on-failure ) \
  >"$LOG/ctest.log" 2>&1
record ctest $?

# --- 5. manifest ------------------------------------------------------------
PYTHONPATH="$HERE/python:$DEPS" python3 - "$OUT_ROOT" "$HEAD_SHA" \
  "$BASE_REF" "$status_file" "$HERE" <<'PY'
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

out, head, base, status_path = sys.argv[1:5]
here = Path(sys.argv[5]).resolve()
repo = here.parents[1]


def sh(*argv):
    try:
        return subprocess.run(argv, cwd=str(repo), text=True, capture_output=True,
                              check=False).stdout.strip()
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"<error: {exc}>"


commands = {}
for line in Path(status_path).read_text().splitlines():
    name, code = line.split("\t")
    commands[name] = {"exit_code": int(code)}

modules = {}
for name in ("numpy", "jsonschema", "pytest"):
    try:
        module = importlib.import_module(name)
        modules[name] = getattr(module, "__file__", "builtin")
    except Exception as exc:  # pragma: no cover - diagnostic only
        modules[name] = f"unavailable: {exc}"
try:  # the pinned TileLang wheel registers its own TVM tree on import
    import tilelang
    modules["tilelang"] = tilelang.__file__
    modules["tilelang_version"] = getattr(tilelang, "__version__", "unknown")
    import tvm
    modules["tvm"] = tvm.__file__
    modules["tvm_version"] = getattr(tvm, "__version__", "unknown")
except Exception as exc:
    modules["tilelang"] = f"unavailable: {exc}"

manifest = {
    "kind": "refactor-evidence",
    "created_utc": Path(out).name.split("-")[0],  # results/<STAMP>-refactor
    "evidence_root": str(out),
    "git": {
        "head": head,
        "base_ref": base,
        "dirty_before": sh("git", "status", "--short"),
        "log": sh("git", "log", "--oneline", "-3"),
    },
    "environment": {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "modules": modules,
    },
    "commands": commands,
    "artifacts": sorted(str(path.relative_to(out))
                        for path in Path(out).rglob("*")
                        if path.is_file() and path.name != ".status"),
    "notes": [
        "baseline/ and current/ are produced by the same script against a git "
        "worktree of base_ref and against this working tree",
        "source_hash for TileLang-derived plans embeds a TVM SourceName object "
        "address that differs between processes (this is true at base_ref as "
        "well); the projection normalises exactly that substring and recomputes "
        "the canonical plan hash, see tools/refactor_plan_projection.py",
    ],
}
Path(out, "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
print("manifest written")
PY
record manifest $?

echo "--- evidence summary ---" >&2
python3 - "$OUT_ROOT" "$status_file" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
status_path = Path(sys.argv[2])
if not (out / "manifest.json").is_file():
    print("manifest.json was not written; the evidence bundle is incomplete",
          file=sys.stderr)
manifest_path = out / "manifest.json"
if manifest_path.is_file():
    manifest = json.loads(manifest_path.read_text())
    for name, entry in manifest["commands"].items():
        print(f"{name:24s} rc={entry['exit_code']}")
for name in ("comparison.json", "native-ir-evaluation.json"):
    path = out / name
    if not path.is_file():
        print(f"{name:24s} MISSING")
        continue
    payload = json.loads(path.read_text())
    if name == "comparison.json":
        summary = payload["summary"]
        print(f"{name:24s} identical={summary['identical']}/{summary['cases']} "
              f"negatives={summary['negatives_identical']}/"
              f"{summary['negatives']}")
    elif name == "native-ir-evaluation.json":
        ok = sum(1 for c in payload["positive"].values()
                 if c.get("plan_identical") and c.get("descriptor_identical"))
        print(f"{name:24s} identical={ok}/{len(payload['positive'])}")
    else:
        counts = payload.get("counts", {})
        print(f"{name:24s} {counts} "
              f"failures={len(payload.get('failures', []))}")

failed = []
for line in status_path.read_text().splitlines():
    if not line.strip():
        continue
    step, code = line.split("\t")
    if int(code) != 0:
        failed.append(f"{step} (rc={code})")
sys.exit(1 if failed else 0)
PY
summary_rc=$?
rm -f "$status_file"

# Every step is attempted first, so one failure still produces a complete bundle.
# The bundle is only accepted when every requested step succeeded: a skipped
# optional step records its own non-zero code and fails the run as well.
if [ "$summary_rc" -ne 0 ]; then
  echo "EVIDENCE FAILED: see $OUT_ROOT/manifest.json commands and $LOG" >&2
else
  echo "evidence complete: $OUT_ROOT" >&2
fi
exit "$summary_rc"
