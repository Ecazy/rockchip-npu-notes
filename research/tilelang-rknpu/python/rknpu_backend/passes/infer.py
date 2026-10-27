# SPDX-License-Identifier: GPL-3.0-or-later
"""Type/shape/numeric inference and static verification."""
from __future__ import annotations

from ..errors import RK_E_SHAPE, reject
from ..graph import is_compute_graph, missing_graph_attributes
from ..ir import TensorType


def infer_and_verify(module):
    """Return the graph after enforcing static rank-2 FP16 invariants.

    ``Module.__post_init__`` performs the core graph validation for the internal
    RKIR carrier.  This pass adds the policies that belong to compilation rather
    than serialization: every value is rank-2, no dynamic extent survives, and a
    matmul result is never silently retyped.  It accepts any carrier satisfying
    :mod:`rknpu_backend.graph`, so the native-graph pilot is checked by exactly
    the same code as the legacy RKIR entry.
    """
    if not is_compute_graph(module):
        reject(RK_E_SHAPE, "compile_rkir expects an ir.Module", stage="frontend",
               actual=type(module).__name__,
               required=missing_graph_attributes(module))
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
