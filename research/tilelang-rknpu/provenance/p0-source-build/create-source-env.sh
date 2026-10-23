#!/usr/bin/env bash
# Recreate the isolated source validation Python environment without changing the
# locked .pydeps-p0 tree. The source tvm_ffi package, built core extension and
# source native libraries take precedence; the wheel tvm_ffi package and its
# RECORD metadata are deliberately excluded.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/../.." && pwd)
DEPS="$ROOT/.pydeps-p0"
SRC="$ROOT/.src/p0-build-src/tvm/3rdparty/tvm-ffi/python/tvm_ffi"
ENV="$ROOT/.src/p0-source-env"
CORE="$ROOT/.src/p0-build-tvmffi-python/core.abi3.so"

[[ -d "$DEPS" ]] || { echo "missing $DEPS" >&2; exit 2; }
[[ -d "$SRC" ]] || { echo "missing $SRC" >&2; exit 2; }
[[ -f "$CORE" ]] || { echo "missing $CORE" >&2; exit 2; }

rm -rf "$ENV"
mkdir -p "$ENV"
python3 - "$DEPS" "$ENV" "$SRC" "$CORE" <<'PY'
import shutil
import sys
from pathlib import Path

deps, env, src, core = map(Path, sys.argv[1:])
for item in deps.iterdir():
    name = item.name
    if name == "tvm_ffi" or (name.startswith("apache_tvm_ffi-") and name.endswith(".dist-info")):
        continue
    (env / name).symlink_to(item.resolve())

# Use the source Python wrapper, not the wheel package. The built Cython core is
# placed beside it so ``import tvm_ffi.core`` resolves to the source-built object.
shutil.copytree(src, env / "tvm_ffi", ignore=shutil.ignore_patterns("__pycache__"))
(env / "tvm_ffi" / "core.abi3.so").symlink_to(core.resolve())
PY
printf 'source environment: %s\n' "$ENV"
printf 'source tvm_ffi core: %s\n' "$CORE"
printf 'wheel tvm_ffi package excluded; apache-tvm-ffi RECORD metadata excluded\n'
