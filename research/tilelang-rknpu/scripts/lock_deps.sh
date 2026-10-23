#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# P0 dependency lock + clean install.
#
#   scripts/lock_deps.sh lock     # (re)generate requirements.lock from pinned artifacts
#   scripts/lock_deps.sh install  # install into a brand-new directory from that lock
#   scripts/lock_deps.sh verify   # assert no duplicate dist metadata + import identity
#
# The lock pins every artifact by exact version AND SHA-256 and installs with
# `--require-hashes --no-deps`, so a fresh environment contains exactly these
# files and nothing else.  Nothing here touches the network at install time
# beyond fetching those exact artifacts.
set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
LOCK="$HERE/requirements.lock"
DEPS_DIR=${TILELANG_DEPS_DIR:-$HERE/.pydeps-p0}
VENV_DIR=${TILELANG_VENV_DIR:-$HERE/.venv-p0}
# `pip --target` fills a directory but does not stop the interpreter from also
# reading the OS dist-packages.  Module resolution is therefore checked with an
# isolated venv interpreter (created with --without-pip: ensurepip is absent in
# this image), which is the interpreter used by the verification campaign too.
PY=python3
if [ -x "$VENV_DIR/bin/python" ]; then
    PY="$VENV_DIR/bin/python"
fi
CPU_INDEX="https://download.pytorch.org/whl/cpu"

# Every artifact the backend needs, at the version recorded in provenance.json.
# torch is pinned to the CPU build: the extractor builds TIR and never runs a GPU
# kernel, so pulling the CUDA wheels would add ~3 GB for nothing.
PINS=(
  "tilelang==0.1.15"
  "apache-tvm-ffi==0.1.12"
  "torch-c-dlpack-ext==0.1.5"
  "numpy==2.5.4"
  "cloudpickle==3.1.2"
  "ml-dtypes==0.6.0"
  "psutil==7.2.2"
  "tqdm==4.70.1"
  "typing-extensions==4.16.0"
  "z3-solver==4.15.4.0"
  "pytest==9.1.1"
  "pluggy==1.7.0"
  "iniconfig==2.3.1"
  "packaging==26.3"
  "pygments==2.21.0"
  "filelock==3.20.4"
  "sympy==1.14.0"
  "networkx==3.6.1"
  "jinja2==3.1.6"
  "markupsafe==3.0.4"
  "fsspec==2026.9.0"
  "setuptools==84.2.0"
  # Declared project dependency: the verifier uses jsonschema for the schema
  # tests.  It must come from the locked environment, not from the OS
  # dist-packages directory.
  "jsonschema==4.25.1"
  "attrs==26.1.0"
  "jsonschema-specifications==2025.9.1"
  "referencing==0.37.0"
  "rpds-py==2026.9.1"
)
TORCH_PIN="torch==2.9.0+cpu"

usage() {
    echo "usage: $0 {lock|install|verify}" >&2
    exit 2
}
[ $# -eq 1 ] || usage

case "$1" in
lock)
    work=$(mktemp -d)
    trap 'rm -rf "$work"' EXIT
    echo "downloading pinned artifacts into $work" >&2
    python3 -m pip download --no-deps -d "$work" "${PINS[@]}" >/dev/null
    python3 -m pip download --no-deps -d "$work" --index-url "$CPU_INDEX" \
        "$TORCH_PIN" >/dev/null
    {
        echo "# P0 dependency lock: exact artifacts, verified by SHA-256."
        echo "# Regenerate with scripts/lock_deps.sh lock"
        echo "# Install with   scripts/lock_deps.sh install"
        echo "--extra-index-url $CPU_INDEX"
        python3 - "$work" <<'PY'
import hashlib
import pathlib
import sys

wheel_dir = pathlib.Path(sys.argv[1])
entries = []
for path in sorted(wheel_dir.glob("*.whl")):
    name = path.name.split("-")
    distribution, version = name[0], name[1]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    entries.append((distribution.lower(), f"{distribution}=={version}",
                    digest, path.name))
for _, spec, digest, wheel in sorted(entries):
    print(f"{spec} --hash=sha256:{digest}  # {wheel}")
PY
    } >"$LOCK"
    echo "wrote $LOCK with $(grep -c -- '--hash=' "$LOCK") pinned artifacts" >&2
    ;;

install)
    [ -s "$LOCK" ] || { echo "missing $LOCK; run the lock step first" >&2; exit 1; }
    if [ -e "$DEPS_DIR" ]; then
        echo "$DEPS_DIR already exists; a clean install needs an empty target" >&2
        echo "remove it or point TILELANG_DEPS_DIR elsewhere" >&2
        exit 1
    fi
    echo "installing $(grep -c -- '--hash=' "$LOCK") artifacts into $DEPS_DIR" >&2
    echo "(runtime interpreter for verification: $PY)" >&2
    python3 -m pip install --no-deps --require-hashes \
        --target "$DEPS_DIR" -r "$LOCK" >"$DEPS_DIR.install.log" 2>&1 || {
        tail -n 30 "$DEPS_DIR.install.log" >&2
        exit 1
    }
    echo "installed; log at $DEPS_DIR.install.log" >&2
    ;;

verify)
    export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-$HERE/.cache}"
    export LD_LIBRARY_PATH="$DEPS_DIR/z3/lib:$DEPS_DIR/tilelang/lib:$DEPS_DIR/tvm_ffi/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    "$PY" - "$DEPS_DIR" <<'PY'
import importlib
import importlib.metadata as md
import pathlib
import sys

deps = pathlib.Path(sys.argv[1])
if not deps.is_dir():
    raise SystemExit(f"{deps} does not exist")
dists = sorted(md.distributions(path=[str(deps)]),
               key=lambda d: (d.metadata["Name"] or "").lower())
names = [(d.metadata["Name"] or "").lower() for d in dists]
duplicates = sorted({n for n in names if names.count(n) > 1})
for dist in dists:
    print(f"{dist.metadata['Name']}=={dist.version}")
print(f"distributions={len(names)} duplicates={duplicates}")
py_dir = deps / "tilelang"
if not py_dir.is_dir():
    raise SystemExit("tilelang package missing from the clean install")
for name in ("tilelang", "tvm", "tvm_ffi", "z3", "numpy"):
    sys.path.insert(0, str(deps))
    try:
        mod = importlib.import_module(name)
        print(f"import {name}: {getattr(mod, '__version__', 'unknown')} "
              f"{getattr(mod, '__file__', '')}")
    except Exception as exc:  # noqa: BLE001 - import identity check only
        raise SystemExit(f"import {name} failed: {exc}")
if duplicates:
    raise SystemExit(f"duplicate distribution metadata: {duplicates}")
PY
    ;;

*)
    usage
    ;;
esac
