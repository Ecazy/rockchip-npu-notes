#!/usr/bin/env bash
# Reproducible local optional environment for the structural TileLang extractor.
# It deliberately avoids pip's default torch-CUDA dependency graph: only a CPU
# torch wheel is installed, because the frontend tests build TIR and never run a
# GPU kernel.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"
python3 -m pip install --target .pydeps --no-deps \
  'tilelang==0.1.15' 'apache-tvm-ffi==0.1.12' 'z3-solver==4.15.4.0' \
  numpy cloudpickle ml_dtypes psutil tqdm typing-extensions \
  pytest pluggy iniconfig packaging pygments
python3 -m pip install --target .pydeps --no-deps \
  --index-url https://download.pytorch.org/whl/cpu 'torch==2.9.0+cpu'
python3 -m pip install --target .pydeps --no-deps torch-c-dlpack-ext
# The z3 wheel declares a SONAME that differs from its file name on some hosts.
if [ -e .pydeps/z3/lib/libz3.so ]; then
  ln -sf libz3.so .pydeps/z3/lib/libz3.so.4.15
fi
