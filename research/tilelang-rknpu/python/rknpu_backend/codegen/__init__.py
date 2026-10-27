# SPDX-License-Identifier: GPL-3.0-or-later
"""The RKNPU codegen boundary: normalize, plan, verify, encode.

Importing this package has no device and no TileLang/TVM dependency.
"""
from .descriptor import build_plan_descriptor
from .pipeline import (Compilation, CompileSnapshot, compile_graph,
                       explain_plan, normalize, normalize_tilelang,
                       normalize_tilelang_native, options_to_dict, plan,
                       plan_verified, source_identity)

__all__ = ["Compilation", "CompileSnapshot", "build_plan_descriptor",
           "compile_graph", "explain_plan", "normalize",
           "normalize_tilelang", "normalize_tilelang_native", "options_to_dict",
           "plan", "plan_verified", "source_identity"]
