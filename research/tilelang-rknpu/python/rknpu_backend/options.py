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
    # Primitive-level add+ReLU fusion: replace an EW_ADD immediately consumed
    # by an EW_MAX(zero) with one co.ew_add_relu template.  This is separate
    # from Submit grouping and is gated on the capability manifest.
    fuse_add_relu: bool = True
    # General task-chain fusion: merge adjacent, dependency-connected Submit
    # groups into one CO_CHAIN envelope when the tasks are contiguous in action
    # order and every NPU dependency is satisfied by an earlier task in the
    # merged envelope.  This changes no task, buffer, arithmetic or transfer;
    # it only reduces submit/command/sync envelopes.  Opt-in until the mixed
    # template chain has its own board probe.
    fuse_task_chain: bool = False
    # Experimental patterns are opt-in: only ``enabled`` capabilities are selected
    # by default (P2).
    allow_experimental: bool = False
    max_actions: int = 1_000_000
    max_ir_nodes: int = 4096
    # Explicit single-matmul execution policy.  The source computes the same
    # graph in every case; this only selects task order, submit grouping and
    # proven CBUF input reuse.
    linear_task_order: str = "baseline"
    # How many whole N tiles one submit may hold for the grouped linear
    # policies.  0 keeps the automatic choice (the largest count that stays
    # inside the 64-task bound); any other value is honoured and validated.
    linear_group_n_tiles: int = 0

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
        if self.linear_task_order not in {"baseline", "scheduled_baseline", "merged",
                                          "interleaved", "interleaved_reuse",
                                          "weight_stationary",
                                          "weight_stationary_reuse",
                                          "grouped_linear"}:
            reject(RK_E_UNSUPPORTED,
                   f"unknown linear_task_order {self.linear_task_order!r}",
                   stage="options")
        if self.linear_task_order in {"interleaved_reuse", "weight_stationary",
                                      "weight_stationary_reuse",
                                      "grouped_linear"} and \
                self.submit_mode != "chain":
            reject(RK_E_UNSUPPORTED,
                   f"{self.linear_task_order} requires submit_mode='chain'",
                   stage="options")
        if not isinstance(self.linear_group_n_tiles, int) or \
                isinstance(self.linear_group_n_tiles, bool) or \
                self.linear_group_n_tiles < 0:
            reject(RK_E_UNSUPPORTED,
                   "linear_group_n_tiles must be a non-negative integer",
                   stage="options", actual=self.linear_group_n_tiles)
        if self.submit_mode not in {"chain", "separate"}:
            reject(RK_E_UNSUPPORTED, f"unknown submit mode {self.submit_mode!r}",
                   stage="options")
        if self.tile_limit != 256:
            reject(RK_E_UNSUPPORTED, "MVP tile_limit is fixed at 256",
                   stage="options", actual=self.tile_limit, required=256)
        for name in ("fuse_epilogue", "fuse_add_relu", "fuse_task_chain"):
            if not isinstance(getattr(self, name), bool):
                reject(RK_E_UNSUPPORTED, f"{name} must be a boolean",
                       stage="options")
        for name in ("device_budget_bytes", "host_budget_bytes", "max_actions",
                     "max_ir_nodes"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                reject(RK_E_BUDGET, f"{name} must be a positive integer",
                       stage="options")
