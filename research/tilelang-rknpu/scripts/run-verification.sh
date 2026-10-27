#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Reproducible host verification campaign for the TileLang -> RKPlan slice.
#
# Guarantees this script provides:
#   * the Python/TileLang/device-library environment is exported BEFORE anything
#     is measured, so the recorded identity is the identity actually tested;
#   * both build trees are created from scratch for every run, and the sanitizer
#     flags are passed explicitly on the configure line;
#   * the flags that the compiler and linker really received are archived;
#   * evidence generation (hashes, manifest, summary) is a checked step: a
#     missing artifact or an unparseable statistic fails the run.
#
# No step opens a device.  This proves Host behaviour only.
set -uo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(cd "$HERE/../.." && pwd)
CUSTOM_OPS="$REPO/research/custom-ops"
BUILD_HOST="$CUSTOM_OPS/build-host-mvp"
BUILD_SAN="$CUSTOM_OPS/build-sanitize-rk"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$HERE/results/$STAMP-host"

# Inherited sanitizer settings must never reach cc1/venv/cmake.  Snapshot the
# ORIGINAL values first (before this script sets any), then drop them; only the
# dedicated sanitizer commands re-introduce what they need, per command.
RKNPU_ENV_WHITELIST=(LD_PRELOAD ASAN_OPTIONS UBSAN_OPTIONS LSAN_OPTIONS CC CFLAGS
                     LDFLAGS RKNPU_PLAN_MOCK_LIB RKNPU_POISON_CHILD)
mkdir -p "$OUT"
{
    printf '%s\n' '=== inherited (pre-script) sanitizer-relevant environment ==='
    for v in "${RKNPU_ENV_WHITELIST[@]}"; do printf '%s=%s\n' "$v" "${!v-<unset>}"; done
} > "$OUT/environment-whitelist.log"
for v in "${RKNPU_ENV_WHITELIST[@]}"; do
    case "$v" in CC|CFLAGS|LDFLAGS) : ;; *) unset "$v" ;; esac
done
KEEP_BUILD=0
# Which dependency tree to test.  P0 requires proving a *clean* install, so this
# is an explicit, recorded input rather than an implicit `.pydeps` default.
# Default to the P0 hash-locked install; `.pydeps` is only the legacy baseline.
DEPS_DIR=${TILELANG_DEPS_DIR:-$HERE/.pydeps-p0}
if [ ! -d "$DEPS_DIR/tilelang" ]; then
    echo "no tilelang package under TILELANG_DEPS_DIR=$DEPS_DIR" >&2
    echo "run scripts/lock_deps.sh install (or point TILELANG_DEPS_DIR at .pydeps)" >&2
    exit 2
fi

VENV_DIR=${TILELANG_VENV_DIR:-$HERE/.venv-p0}
PY=python3
if [ "${TILELANG_NO_VENV:-0}" != "1" ]; then
    # A venv (created with --without-pip: ensurepip is unavailable here) has its
    # own site-packages and does NOT inherit /usr/lib/python3/dist-packages, which
    # `pip --target` alone cannot prevent.  `python3 -S` is the fallback below.
    if [ ! -x "$VENV_DIR/bin/python" ]; then
        python3 -m venv --without-pip "$VENV_DIR" || {
            echo "cannot create the isolation venv at $VENV_DIR" >&2; exit 2; }
    fi
    PY="$VENV_DIR/bin/python"
fi

for arg in "$@"; do
    case "$arg" in
        --keep-build) KEEP_BUILD=1 ;;
        -h|--help) echo "usage: $0 [--keep-build]"; exit 0 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

mkdir -p "$OUT" "$OUT/compile-flags"
CMDS="$OUT/commands.log"
: >"$CMDS"
FAILED=0
declare -A STEP_RC=()

log() { printf '%s\n' "$*" >>"$CMDS"; }

# ---- 0. environment first: everything measured below sees exactly this ------
# Export the dependency root: check_isolation.py (and any nested tool) reads it.
export TILELANG_DEPS_DIR="$DEPS_DIR"
export TILELANG_LOCK_FILE="$HERE/requirements.lock"
export PYTHONPATH="$HERE/python:$DEPS_DIR"
export TILELANG_CACHE_DIR="$HERE/.cache"
export RKNPU_PLAN_LIB="$BUILD_HOST/librknpu_plan.so"
export LD_LIBRARY_PATH="$DEPS_DIR/z3/lib:$DEPS_DIR/tilelang/lib:$DEPS_DIR/tvm_ffi/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Sanitizer options are NOT exported globally: the C sanitizer steps and the two
# Python workers set them per command (see below), so an ordinary step can never
# inherit them.
export MALLOC_PERTURB_=128

run_step() {
    local name=$1
    shift
    local log_file="$OUT/$name.log"
    log "=== [$name] $*"
    log "--- started: $(date -u +%FT%TZ)"
    ( "$@" ) >"$log_file" 2>&1
    local rc=$?
    STEP_RC["$name"]=$rc
    log "--- exit: $rc"
    if [ "$rc" -ne 0 ]; then
        FAILED=1
        log "--- FAILED, tail follows"
        tail -n 25 "$log_file" >>"$CMDS"
    fi
}

run_step_fn() {
    local name=$1 fn=$2
    local log_file="$OUT/$name.log"
    log "=== [$name] shell function $fn"
    log "--- started: $(date -u +%FT%TZ)"
    "$fn" >"$log_file" 2>&1
    local rc=$?
    STEP_RC["$name"]=$rc
    log "--- exit: $rc"
    if [ "$rc" -ne 0 ]; then
        FAILED=1
        log "--- FAILED, tail follows"
        tail -n 25 "$log_file" >>"$CMDS"
    fi
}

# ---- 1. environment identity (after the exports above) ----------------------
capture_environment() {
    echo "# environment (captured after this script exported its own paths)"
    echo "collected_utc=$(date -u +%FT%TZ)"
    echo "repo=$REPO"
    echo "host=$(uname -a)"
    echo "git_head=$(git -C "$REPO" rev-parse HEAD 2>/dev/null)"
    echo "PYTHONPATH=$PYTHONPATH"
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
    echo "TILELANG_CACHE_DIR=$TILELANG_CACHE_DIR"
    echo "RKNPU_PLAN_LIB=$RKNPU_PLAN_LIB"
    echo "TILELANG_DEPS_DIR=$DEPS_DIR"
    echo "TILELANG_LOCK_FILE=$HERE/requirements.lock"
    echo "TEST_INTERPRETER=$PY"
    "$PY" - <<'PY'
import json
import sys
print("interpreter=" + json.dumps({
    "executable": sys.executable,
    "prefix": sys.prefix,
    "base_prefix": sys.base_prefix,
    "no_site": bool(sys.flags.no_site),
    "isolated_venv": sys.prefix != sys.base_prefix,
}))
PY
    echo "ASAN_OPTIONS=${ASAN_OPTIONS-<unset>}"
    echo "UBSAN_OPTIONS=${UBSAN_OPTIONS-<unset>}"
    echo "MALLOC_PERTURB_=$MALLOC_PERTURB_"
    echo
    echo "# git status (scoped)"
    git -C "$REPO" status --short research/custom-ops research/tilelang-rknpu
    echo
    echo "# worktree diffstat (scoped)"
    git -C "$REPO" diff --stat research/custom-ops research/tilelang-rknpu
    echo
    echo "# toolchain"
    (cc --version | head -1) 2>&1
    (cmake --version | head -1) 2>&1
    (python3 --version) 2>&1
    echo
    echo "# installed distributions visible to the test interpreter"
    "$PY" - <<'PY'
import importlib
import importlib.metadata as md
import sys

roots = [p for p in sys.path if p.endswith((".pydeps", ".pydeps-p0", ".pydeps-src"))]
print("search_roots=" + (", ".join(roots) if roots else "<none>"))
dists = sorted(md.distributions(path=roots) if roots else md.distributions(),
               key=lambda d: (d.metadata["Name"] or "").lower())
names = []
for dist in dists:
    name = dist.metadata["Name"]
    names.append((name or "").lower())
    print(f"{name}=={dist.version}")
duplicates = sorted({n for n in names if names.count(n) > 1})
print(f"duplicate_distribution_names={duplicates}")
print()
print("# imported module identity")
for name in ("numpy", "jsonschema", "pytest", "tilelang", "tvm", "tvm_ffi",
             "z3", "torch"):
    try:
        mod = importlib.import_module(name)
        print(f"{name}={getattr(mod, '__version__', 'unknown')} "
              f"origin={getattr(mod, '__file__', 'builtin')}")
    except Exception as exc:  # noqa: BLE001 - identity capture only
        print(f"{name}=ABSENT ({type(exc).__name__})")
if duplicates:
    raise SystemExit(f"ambiguous environment: duplicate distributions {duplicates}")
PY
    local rc=$?
    echo
    echo "# optional dependency root"
    du -sh "$DEPS_DIR" 2>/dev/null || echo "$DEPS_DIR absent"
    return "$rc"
}

# ---- 2. source hashes -------------------------------------------------------
capture_hashes() {
    "$PY" - "$REPO" "$OUT/source-hashes.json" <<'PY'
import hashlib
import json
import pathlib
import sys

repo = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
roots = [repo / "research/custom-ops/src", repo / "research/custom-ops/include",
         repo / "research/custom-ops/tests/c",
         # the mock seam and its header are compiled into the mock library and the
         # C host test, so they belong in the source snapshot
         repo / "research/custom-ops/tests/mock",
         repo / "research/tilelang-rknpu/python",
         repo / "research/tilelang-rknpu/tests",
         repo / "research/tilelang-rknpu/schema",
         repo / "research/tilelang-rknpu/capabilities"]
hashes = {}
for root in roots:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".npy"}:
            continue
        hashes[path.relative_to(repo).as_posix()] = hashlib.sha256(
            path.read_bytes()).hexdigest()
for rel in ("research/custom-ops/CMakeLists.txt",
            "research/tilelang-rknpu/provenance.json",
            "research/tilelang-rknpu/archive/development-history.md",
            "research/tilelang-rknpu/scripts/run-verification.sh"):
    path = repo / rel
    if path.is_file():
        hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
if len(hashes) < 50:
    raise SystemExit(f"refusing to record only {len(hashes)} hashes")
out.write_text(json.dumps({"file_count": len(hashes), "sha256": hashes},
                          indent=2, sort_keys=True) + "\n")
print(f"{len(hashes)} files hashed -> {out}")
PY
    [ -s "$OUT/source-hashes.json" ] || { echo "source-hashes.json missing" >&2; return 1; }
}

# ---- 3. fresh build trees ---------------------------------------------------
prepare_build_dirs() {
    if [ "$KEEP_BUILD" -eq 1 ]; then
        # Keep mode is for re-running the campaign against an existing tree, so a
        # cache is expected.  Instead of demanding freshness, prove the cached
        # configuration belongs to this source tree and carries the expected
        # compiler/linker flags.
        echo "keeping existing build trees (--keep-build)"
        local dir cache home flags_ok=0
        for dir in "$BUILD_HOST" "$BUILD_SAN"; do
            cache="$dir/CMakeCache.txt"
            if [ ! -s "$cache" ]; then
                echo "ERROR: --keep-build needs an existing cache at $cache" >&2
                return 1
            fi
            home=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$cache")
            if [ "$home" != "$CUSTOM_OPS" ]; then
                echo "ERROR: $cache belongs to $home, not $CUSTOM_OPS" >&2
                return 1
            fi
            echo "reusing $dir (source $home)"
        done
        if grep -q -- "-fsanitize=address,undefined" "$BUILD_SAN/CMakeCache.txt"; then
            flags_ok=1
        fi
        if [ "$flags_ok" -ne 1 ]; then
            echo "ERROR: kept sanitizer tree has no sanitizer flags; re-run without"                  "--keep-build to reconfigure" >&2
            return 1
        fi
        echo "kept build trees validated (source root + sanitizer flags)"
        return 0
    fi
    for dir in "$BUILD_HOST" "$BUILD_SAN"; do
        if [ -d "$dir" ]; then
            echo "removing stale build tree $dir"
            rm -rf "$dir" || return 1
        fi
    done
    for dir in "$BUILD_HOST" "$BUILD_SAN"; do
        if [ -e "$dir/CMakeCache.txt" ]; then
            echo "ERROR: $dir still has a cache; refusing a non-fresh run" >&2
            return 1
        fi
    done
    echo "build trees are fresh"
}

# ---- 4. explicit sanitizer configuration with post-configure verification ---
configure_sanitize() {
    cmake -S "$CUSTOM_OPS" -B "$BUILD_SAN" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_C_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
        -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" \
        -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address,undefined" || return 1
    grep -q -- "-fsanitize=address,undefined" "$BUILD_SAN/CMakeCache.txt" || {
        echo "sanitizer flags are absent from CMakeCache.txt" >&2; return 1; }
    local link="$BUILD_SAN/CMakeFiles/rk_plan_host_test.dir/link.txt"
    [ -s "$link" ] || { echo "link.txt not generated yet (expected before build)" >&2; }
    echo "sanitizer configure flags verified in CMakeCache.txt"
}

# ---- 5. archive the real compile/link command lines ------------------------
archive_flags() {
    "$PY" - "$BUILD_HOST" "$BUILD_SAN" "$OUT/compile-flags" <<'PY'
import json
import pathlib
import sys

host, san, out = (pathlib.Path(x) for x in sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)
interest = ("rk_plan_", "custom_ops/src", "tilelang")
written = 0
for label, build in (("host", host), ("sanitize", san)):
    commands = build / "compile_commands.json"
    if not commands.is_file():
        raise SystemExit(f"{label}: compile_commands.json missing")
    entries = [e for e in json.loads(commands.read_text())
               if any(key in e["file"] for key in interest)]
    if not entries:
        raise SystemExit(f"{label}: no interesting compiler invocations found")
    (out / f"compile-commands-{label}.json").write_text(
        json.dumps(entries, indent=2) + "\n")
    written += 1
    print(f"{label}: {len(entries)} compiler invocations archived")
    for link in sorted(build.glob("CMakeFiles/*/link.txt")):
        if "rk_plan" not in str(link) and "rknpu_plan" not in str(link):
            continue
        (out / f"link-{label}-{link.parent.name}.txt").write_text(link.read_text())
        written += 1
    cache = build / "CMakeCache.txt"
    if cache.is_file():
        lines = [line for line in cache.read_text().splitlines()
                 if "SANITIZE" in line or "CMAKE_C_FLAGS" in line
                 or "LINKER_FLAGS" in line]
        (out / f"cmakecache-flags-{label}.txt").write_text("\n".join(lines) + "\n")
        written += 1
san = out / "compile-commands-sanitize.json"
if not san.is_file() or "-fsanitize=address,undefined" not in san.read_text():
    raise SystemExit("archived sanitizer compile commands lack the flag")
if written < 4:
    raise SystemExit(f"only {written} flag artifacts archived")
print(f"{written} flag artifacts archived")
PY
}

# ---- 6. summary as a checked step -----------------------------------------
build_summary() {
    "$PY" - "$OUT" "$FAILED" <<'SUMMARY_PY'
import json
import pathlib
import re
import sys

out = pathlib.Path(sys.argv[1])
failed = int(sys.argv[2])
problems = []

required = ["isolation.log", "source-hashes.json", "environment.log", "commands.log",
            "pytest.log", "ctest.log", "environment-whitelist.log", "rk_plan_contract.log",
            "rk_plan_host.log", "sanitize-contract.log", "sanitize-host.log", "sanitize-mock-build.log",
            "sanitize-poison-worker.log", "sanitize-ffn-worker.log",
            "compile-flags/compile-commands-sanitize.json"]
for name in required:
    path = out / name
    if not path.is_file() or path.stat().st_size == 0:
        problems.append(f"missing or empty artifact: {name}")


def read(name):
    try:
        return (out / name).read_text()
    except OSError:
        return ""


cmds = read("commands.log").splitlines()


def step_exit(name):
    inside = False
    for line in cmds:
        if line.startswith(f"=== [{name}] "):
            inside = True
            continue
        if inside and line.startswith("--- exit: "):
            return int(line.split(":")[1])
    return None


steps = ("environment", "isolation", "source-hashes", "prepare-build", "configure", "build",
         "ctest", "rk_plan_contract", "rk_plan_host", "graph_host",
         "mock_scenarios",
         "configure-sanitize",
         "build-sanitize", "sanitize-contract", "sanitize-host",
         "sanitize-mock-build", "sanitize-poison-worker", "sanitize-ffn-worker",
         "archive-flags", "pytest")
step_map = {name: step_exit(name) for name in steps}
for name, rc in step_map.items():
    if rc is None:
        problems.append(f"could not parse exit code for step {name}")

pytest_log = read("pytest.log")
failed_match = re.search(r"(\d+) failed", pytest_log)
failed_count = int(failed_match.group(1)) if failed_match else 0
if failed_count:
    problems.append(f"pytest reported {failed_count} failing tests")
passed_match = re.search(r"(\d+) passed", pytest_log)
passed = int(passed_match.group(1)) if passed_match else None
if passed is None:
    problems.append("could not parse the pytest summary line")
skipped_match = re.search(r"(\d+) skipped", pytest_log)
skipped = int(skipped_match.group(1)) if skipped_match else 0
skip_reasons = sorted({line.strip() for line in pytest_log.splitlines()
                       if line.strip().startswith("SKIPPED")})

ctest_ratio = re.search(r"(\d+)% tests passed, (\d+) tests failed out of (\d+)",
                        read("ctest.log"))
if ctest_ratio is None:
    problems.append("could not parse the ctest summary line")


def checks(log_name):
    match = re.search(r"checks=(\d+) failures=(\d+)", read(log_name))
    if not match:
        problems.append(f"could not parse check counts from {log_name}")
        return {"checks": None, "failures": None}
    result = {"checks": int(match.group(1)), "failures": int(match.group(2))}
    if result["failures"]:
        problems.append(f"{log_name} reported {result['failures']} failures")
    return result


contract = checks("rk_plan_contract.log")
host = checks("rk_plan_host.log")
graph = checks("graph_host.log")
for label, values, floor in (("rk_plan_contract", contract, 20),
                             ("rk_plan_host", host, 50),
                             ("graph_host", graph, 1_000_000)):
    if values["checks"] is not None and values["checks"] < floor:
        problems.append(f"{label} recorded only {values['checks']} checks")

if "ISOLATION OK" not in read("isolation.log"):
    problems.append("isolation check did not report ISOLATION OK")
sanitize_compiles = read("compile-flags/compile-commands-sanitize.json")
if sanitize_compiles and "-fsanitize=address,undefined" not in sanitize_compiles:
    problems.append("archived sanitizer compile commands lack the flag")

manifest = {
    "kind": "tilelang-rknpu-host-verification",
    "run_id": out.name,
    "overall_exit": failed,
    "evidence_complete": not problems,
    "evidence_problems": problems,
    "steps": step_map,
    "ctest": ({"passed_percent": int(ctest_ratio.group(1)),
               "failed": int(ctest_ratio.group(2)),
               "total": int(ctest_ratio.group(3))} if ctest_ratio else None),
    "c_checks": {"rk_plan_contract": contract, "rk_plan_host": host,
                 "graph_host": graph},
    "pytest": {"passed": passed, "failed": failed_count, "skipped": skipped,
               "skip_reasons": skip_reasons},
    "deps_dir": ((out / "environment.log").read_text().split(
        "TILELANG_DEPS_DIR=", 1)[1].splitlines()[0]
        if "TILELANG_DEPS_DIR=" in (out / "environment.log").read_text()
        else None),
    "sanitizer": {"enabled_explicitly": True,
                  "flags": "-fsanitize=address,undefined"},
    "device_access": "none",
    "board_acceptance": "not_run",
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

lines = [f"# Host verification {out.name}", "",
         f"overall exit: **{failed}**",
         f"evidence complete: **{not problems}**", "",
         "| step | exit |", "| --- | --- |"]
lines += [f"| {name} | {rc} |" for name, rc in step_map.items()]
lines += ["",
          f"ctest: {manifest['ctest']}",
          f"C checks: contract={contract} host={host}",
          f"sanitizer: {manifest['sanitizer']}",
          f"pytest: {passed} passed, {failed_count} failed, {skipped} skipped"
          + (f" ({'; '.join(skip_reasons)})" if skip_reasons else "")]
if problems:
    lines += ["", "## Evidence problems", ""]
    lines += [f"- {item}" for item in problems]
lines += ["",
          "Device access: none. Board acceptance: not run. This record proves "
          "Host behaviour only; it is not an RK3588 acceptance report."]
(out / "summary.md").write_text("\n".join(lines) + "\n")
print(json.dumps(manifest, indent=2))
if problems:
    raise SystemExit(f"evidence incomplete: {problems}")
SUMMARY_PY
}

# ---- run the campaign -------------------------------------------------------
run_step_fn environment capture_environment
check_isolation() { "$PY" "$HERE/scripts/check_isolation.py"; }
run_step_fn isolation check_isolation
run_step_fn source-hashes capture_hashes
run_step_fn prepare-build prepare_build_dirs
run_step configure cmake -S "$CUSTOM_OPS" -B "$BUILD_HOST" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DPython3_EXECUTABLE="$PY"
run_step build cmake --build "$BUILD_HOST" -j4
run_step ctest ctest --test-dir "$BUILD_HOST" --output-on-failure
run_step rk_plan_contract "$BUILD_HOST/rk_plan_contract_test"
run_step rk_plan_host "$BUILD_HOST/rk_plan_host_test"
run_step graph_host "$BUILD_HOST/graph_host_test"
run_step mock_scenarios "$BUILD_HOST/rk_mock_scenarios"

run_step_fn configure-sanitize configure_sanitize
run_step build-sanitize cmake --build "$BUILD_SAN" \
    --target rk_plan_contract_test rk_plan_host_test -j4
SAN_C_ENV=(ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1
           MALLOC_PERTURB_=128)
run_step sanitize-contract env "${SAN_C_ENV[@]}" "$BUILD_SAN/rk_plan_contract_test"
run_step sanitize-host env "${SAN_C_ENV[@]}" "$BUILD_SAN/rk_plan_host_test"

# Python workers under the fully instrumented sanitizer build.  The mock library is
# rebuilt here so the archive always matches current sources.  LD_PRELOAD and
# detect_leaks=0 apply ONLY to these commands; the C checks above keep leaks fatal.
ASAN_RT="$("${CC:-cc}" -print-file-name=libasan.so)"
SAN_MOCK="$BUILD_SAN/librknpu_plan_mock.so"
run_step sanitize-mock-build cmake --build "$BUILD_SAN" --target rknpu_plan_mock -j4
run_step sanitize-poison-worker env LD_PRELOAD="$ASAN_RT" \
    ASAN_OPTIONS=detect_leaks=0:abort_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    RKNPU_POISON_CHILD=1 RKNPU_PLAN_MOCK_LIB="$SAN_MOCK" \
    TILELANG_CACHE_DIR="$HERE/.cache" PYTHONPATH="$PYTHONPATH" \
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH" "$PY" "$HERE/tests/test_ws_poison.py"
run_step sanitize-ffn-worker env LD_PRELOAD="$ASAN_RT" \
    ASAN_OPTIONS=detect_leaks=0:abort_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    RKNPU_PLAN_MOCK_LIB="$SAN_MOCK" \
    TILELANG_CACHE_DIR="$HERE/.cache" PYTHONPATH="$PYTHONPATH" \
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
    "$PY" -m pytest "$HERE/tests/test_ffn_execution.py" -o addopts= -q
run_step_fn archive-flags archive_flags
run_step pytest "$PY" -m pytest "$HERE/tests" -o addopts= -q -rs --tb=short
run_step_fn summary build_summary

if [ "$FAILED" -ne 0 ]; then
    echo "verification FAILED; see $OUT" >&2
    tail -n 30 "$CMDS" >&2
    exit 1
fi
echo "verification artifacts: $OUT" >&2
echo "verification PASSED" >&2
exit 0
