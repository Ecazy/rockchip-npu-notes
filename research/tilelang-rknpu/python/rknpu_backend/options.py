# SPDX-License-Identifier: GPL-3.0-or-later
"""Frozen compile options with MVP-only values."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import RK_E_BUDGET, RK_E_UNSUPPORTED, reject


@dataclass(frozen=True)
class CompileOptions:
    numeric_mode: str = "strict"
    allow_approx_lut: bool = False
    fallback: str = "error"
    submit_mode: str = "chain"
    tile_limit: int = 256
    device_budget_bytes: int = 256 * 1024 * 1024
    host_budget_bytes: int = 256 * 1024 * 1024
    # Generic submit-level epilogue fusion: chain a Linear output's bias add
    # and ReLU (or either epilogue suffix) into the same submit when the direct
    # dependency and template sequence prove it is safe.  This changes no task,
    # arithmetic or transfer; it only reduces Submit envelopes.
    fuse_epilogue: bool = True
    # Experimental patterns are opt-in: only ``enabled`` capabilities are selected
    # by default (P2).
    allow_experimental: bool = False
    max_actions: int = 1_000_000
    max_ir_nodes: int = 4096

    def __post_init__(self) -> None:
        if self.numeric_mode not in {"strict", "rk_f16_tiled"}:
            reject(RK_E_UNSUPPORTED, f"unknown numeric mode {self.numeric_mode!r}",
                   stage="options")
        if self.fallback not in {"error", "host_prepare"}:
            reject(RK_E_UNSUPPORTED, f"unknown fallback {self.fallback!r}",
                   stage="options")
        if self.fallback != "error":
            reject(RK_E_UNSUPPORTED,
                   "MVP only permits fallback=error; host_prepare needs a declared "
                   "rule and is not part of this minimal path",
                   stage="options")
        if self.submit_mode not in {"chain", "separate"}:
            reject(RK_E_UNSUPPORTED, f"unknown submit mode {self.submit_mode!r}",
                   stage="options")
        if self.tile_limit != 256:
            reject(RK_E_UNSUPPORTED, "MVP tile_limit is fixed at 256",
                   stage="options", actual=self.tile_limit, required=256)
        if not isinstance(self.fuse_epilogue, bool):
            reject(RK_E_UNSUPPORTED, "fuse_epilogue must be a boolean",
                   stage="options")
        for name in ("device_budget_bytes", "host_budget_bytes", "max_actions",
                     "max_ir_nodes"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                reject(RK_E_BUDGET, f"{name} must be a positive integer",
                       stage="options")
