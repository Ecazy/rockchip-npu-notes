#!/usr/bin/env bash
# Host-only build + all available tests.  Not a board acceptance command.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(cd "$HERE/../.." && pwd)
cd "$REPO"
cmake -S research/custom-ops -B research/custom-ops/build-host-mvp
cmake --build research/custom-ops/build-host-mvp -j4
ctest --test-dir research/custom-ops/build-host-mvp --output-on-failure
export PYTHONPATH="$HERE/python${PYTHONPATH:+:$PYTHONPATH}"
export RKNPU_PLAN_LIB="$REPO/research/custom-ops/build-host-mvp/librknpu_plan.so"
if [ -d "$HERE/.pydeps" ]; then
  export PYTHONPATH="$HERE/python:$HERE/.pydeps${PYTHONPATH:+:$PYTHONPATH}"
  export TILELANG_CACHE_DIR="$HERE/.cache"
  export LD_LIBRARY_PATH="$HERE/.pydeps/z3/lib:$HERE/.pydeps/tilelang/lib:$HERE/.pydeps/tvm_ffi/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  python3 -m pytest "$HERE/tests" -q
else
  echo "optional .pydeps absent; running pytest without real TileLang fixtures" >&2
  python3 -m pytest "$HERE/tests" -q
fi
