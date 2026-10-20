# SPDX-License-Identifier: GPL-3.0-or-later
"""Capability legalization for the minimal FP16 path."""
from __future__ import annotations

from dataclasses import dataclass

from ..capabilities import TargetProfile
from ..errors import (RK_E_NUMERIC_CONTRACT, RK_E_UNSUPPORTED, reject)
from ..ir import Module, NumericContract, Op
from ..options import CompileOptions


@dataclass(frozen=True)
class LegalizedOp:
    op_id: int
    capability_id: str
    template_id: str
    activation: str | None = None


@dataclass(frozen=True)
class LegalizedGraph:
    module: Module
    ops: tuple[LegalizedOp, ...]
    matmul_op: Op
    activation_op: Op | None


def legalize(module: Module, *, profile: TargetProfile,
             options: CompileOptions) -> LegalizedGraph:
    """Match only explicitly enabled capabilities and one closed graph shape.

    The minimal vertical slice is one MatMul optionally followed by one supported
    activation.  Anything else is rejected here rather than passed to an
    unverified emitter.
    """
    if len(module.ops) not in {1, 2}:
        reject(RK_E_UNSUPPORTED,
               "minimal MVP path accepts one matmul and one optional activation",
               stage="legalize")
    matmul = module.ops[0]
    if matmul.opcode != "matmul":
        reject(RK_E_UNSUPPORTED, "minimal MVP path must start with matmul",
               stage="legalize", op=matmul.opcode, source=matmul.source)
    contract = NumericContract.from_dict(matmul.attrs["numeric_contract"])
    if options.numeric_mode == "strict":
        reject(RK_E_NUMERIC_CONTRACT,
               "gemm requests FP32 strict accumulation; co.mm_acc.f16.v1 writes "
               "FP16 at each K tile",
               stage="legalize", op=matmul.opcode, source=matmul.source,
               target_profile=profile.profile_hash,
               actual=contract.to_dict(), required="rk_f16_tiled opt-in",
               alternative="pass numeric_mode=rk_f16_tiled to accept the declared "
                           "ordered K-tile writeback path")
    if not contract.allows_f16_tiled_path:
        reject(RK_E_NUMERIC_CONTRACT,
               "numeric contract does not declare ordered K-tile FP16 writeback",
               stage="legalize", op=matmul.opcode, source=matmul.source,
               target_profile=profile.profile_hash,
               actual=contract.to_dict(),
               alternative="declare narrowing at k_tile_boundary")
    profile.enabled("co.mm.f16.v1")
    profile.enabled("co.mm_acc.f16.v1")

    activation_op: Op | None = None
    activation = None
    if len(module.ops) == 2:
        activation_op = module.ops[1]
        if activation_op.opcode not in {"silu", "gelu", "sigmoid", "hardsigmoid"}:
            reject(RK_E_UNSUPPORTED,
                   "only SiLU/GELU/sigmoid/HardSigmoid are in the minimal path",
                   stage="legalize", op=activation_op.opcode,
                   source=activation_op.source)
        if not options.allow_approx_lut:
            reject(RK_E_NUMERIC_CONTRACT,
                   "activation uses a LUT approximation; explicit allow_approx_lut "
                   "is required",
                   stage="legalize", op=activation_op.opcode,
                   source=activation_op.source,
                   target_profile=profile.profile_hash,
                   alternative="pass allow_approx_lut=True after reviewing the "
                               "capability tolerance")
        activation = activation_op.opcode
        cap = {
            "silu": "co.lut.sigmoid.f16.v1",
            "gelu": "co.lut.gelu_gate.f16.v1",
            "sigmoid": "co.lut.sigmoid.f16.v1",
            "hardsigmoid": "co.lut.hardsigmoid.f16.v1",
        }[activation]
        profile.enabled(cap)
        if activation in {"silu", "gelu"}:
            profile.enabled("co.ew_mul.f16.v1")
    legalized = [
        LegalizedOp(matmul.id, "co.mm.f16.v1", "co.mm.f16.v1", activation),
    ]
    if activation_op is not None:
        cap = {
            "silu": "co.lut.sigmoid.f16.v1",
            "gelu": "co.lut.gelu_gate.f16.v1",
            "sigmoid": "co.lut.sigmoid.f16.v1",
            "hardsigmoid": "co.lut.hardsigmoid.f16.v1",
        }[activation_op.opcode]
        legalized.append(LegalizedOp(activation_op.id, cap, cap))
    return LegalizedGraph(module=module, ops=tuple(legalized),
                          matmul_op=matmul, activation_op=activation_op)
