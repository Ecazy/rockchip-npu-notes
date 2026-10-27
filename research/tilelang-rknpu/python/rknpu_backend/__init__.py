# SPDX-License-Identifier: GPL-3.0-or-later
"""TileLang -> RKIR -> RKPlan offline backend and caller-owned runtime.

New code should use :func:`compile_artifact` and the
:class:`~rknpu_backend.artifact.CompiledArtifact` it returns.  The
``compile_tilelang``/``compile_rkir``/``CompileResult`` spellings remain supported
and share the same core pipeline.
"""
from .api import (CompileResult, compile_rkir, compile_tilelang, explain_plan,
                  load_plan, save_plan)
from .artifact import CompiledArtifact, compile_artifact
from .capabilities import (Capability, TargetProfile, default_profile,
                           load_profile, load_target_profile)
from .errors import (BackendError, Diagnostic, RK_E_ALIAS, RK_E_BUDGET,
                     RK_E_DEVICE, RK_E_DOMAIN, RK_E_INVALID, RK_E_NOMEM,
                     RK_E_NUMERIC_CONTRACT, RK_E_POISONED, RK_E_SCHEMA,
                     RK_E_SHAPE, RK_E_TARGET_MISMATCH, RK_E_UNSUPPORTED,
                     RK_E_BUSY)
from .ir import (Module, NumericContract, Op, TensorType, Value,
                 activation_module, branch_outputs_module, matmul_module,
                 projection_module, swiglu_module)
from .options import CompileOptions
from .plan import Plan
from .runtime.bindings import Workspace

__all__ = [
    "BackendError", "Capability", "CompileOptions", "CompileResult",
    "CompiledArtifact",
    "Diagnostic", "Module", "NumericContract", "Op", "Plan", "TargetProfile",
    "TensorType", "Value", "Workspace", "compile_artifact", "compile_rkir",
    "compile_tilelang",
    "default_profile", "explain_plan", "load_plan", "load_profile",
    "load_target_profile", "activation_module", "branch_outputs_module",
    "matmul_module", "projection_module", "swiglu_module", "save_plan",
    "RK_E_INVALID", "RK_E_UNSUPPORTED", "RK_E_SHAPE", "RK_E_NUMERIC_CONTRACT",
    "RK_E_BUDGET", "RK_E_TARGET_MISMATCH", "RK_E_ALIAS", "RK_E_NOMEM",
    "RK_E_DEVICE", "RK_E_BUSY", "RK_E_POISONED", "RK_E_SCHEMA", "RK_E_DOMAIN",
]
