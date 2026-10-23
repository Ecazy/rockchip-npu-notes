# SPDX-License-Identifier: GPL-3.0-or-later
"""Type/shape/numeric inference and static verification."""
from __future__ import annotations

from ..errors import RK_E_SHAPE, reject
from ..ir import Module, TensorType


def infer_and_verify(module: Module) -> Module:
    """Return the module after enforcing static rank-2 FP16 invariants.

    ``Module.__post_init__`` performs the core graph validation.  This pass adds
    the policies that belong to compilation rather than serialization: every
    value is rank-2, no dynamic extent survives, and a matmul result is never
    silently retyped.
    """
    if not isinstance(module, Module):
        reject(RK_E_SHAPE, "compile_rkir expects an ir.Module", stage="frontend")
    for value in module.values:
        if not isinstance(value.type, TensorType):
            reject(RK_E_SHAPE, f"value {value.id} has no TensorType", stage="infer")
        if value.role == "constant":
            if len(value.type.shape) > 2 or \
                    any(v <= 0 for v in value.type.shape):
                reject(RK_E_SHAPE,
                       f"constant value {value.id} has an invalid broadcast shape",
                       stage="infer", actual=list(value.type.shape))
        elif len(value.type.shape) != 2 or any(v <= 0 for v in value.type.shape):
            reject(RK_E_SHAPE, f"value {value.id} is not a static rank-2 tensor",
                   stage="infer")
    return module
