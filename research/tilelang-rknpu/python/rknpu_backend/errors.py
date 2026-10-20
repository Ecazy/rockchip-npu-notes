# SPDX-License-Identifier: GPL-3.0-or-later
"""Stable error codes and diagnostics for the offline RKNPU backend.

The numeric values mirror the C ABI in ``research/custom-ops/include/rk_plan.h``.
Python never invents a second meaning for these values.  Native rocket/custom-ops
statuses are stored separately in :attr:`Diagnostic.native_status`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

RK_E_INVALID = -1
RK_E_UNSUPPORTED = -2
RK_E_SHAPE = -3
RK_E_NUMERIC_CONTRACT = -4
RK_E_BUDGET = -5
RK_E_TARGET_MISMATCH = -6
RK_E_ALIAS = -7
RK_E_NOMEM = -8
RK_E_DEVICE = -9
RK_E_BUSY = -10
RK_E_POISONED = -11
RK_E_SCHEMA = -12
RK_E_DOMAIN = -13

CODE_NAMES: dict[int, str] = {
    RK_E_INVALID: "RK_E_INVALID",
    RK_E_UNSUPPORTED: "RK_E_UNSUPPORTED",
    RK_E_SHAPE: "RK_E_SHAPE",
    RK_E_NUMERIC_CONTRACT: "RK_E_NUMERIC_CONTRACT",
    RK_E_BUDGET: "RK_E_BUDGET",
    RK_E_TARGET_MISMATCH: "RK_E_TARGET_MISMATCH",
    RK_E_ALIAS: "RK_E_ALIAS",
    RK_E_NOMEM: "RK_E_NOMEM",
    RK_E_DEVICE: "RK_E_DEVICE",
    RK_E_BUSY: "RK_E_BUSY",
    RK_E_POISONED: "RK_E_POISONED",
    RK_E_SCHEMA: "RK_E_SCHEMA",
    RK_E_DOMAIN: "RK_E_DOMAIN",
}
NAME_CODES: dict[str, int] = {v: k for k, v in CODE_NAMES.items()}


def code_name(code: int) -> str:
    return CODE_NAMES.get(int(code), f"RK_E_UNKNOWN({code})")


@dataclass
class Diagnostic:
    """A machine-readable rejection/explanation record.

    ``source`` is the source span retained by the frontend.  ``op`` is the opcode,
    ``action_id`` is the plan action id when one exists.  The target profile hash is
    kept as text so a rejected plan can be compared without loading a profile.
    """

    code: int
    stage: str
    message: str
    source: str = ""
    op: str = ""
    action_id: int | None = None
    target_profile: str = ""
    actual: Any = None
    required: Any = None
    alternative: str = ""
    native_status: int | None = None
    severity: str = "error"

    @property
    def code_name(self) -> str:
        return code_name(self.code)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": int(self.code),
            "code_name": self.code_name,
            "severity": self.severity,
            "stage": self.stage,
            "message": self.message,
            "source": self.source,
            "op": self.op,
            "action_id": self.action_id,
            "target_profile": self.target_profile,
            "actual": self.actual,
            "required": self.required,
            "alternative": self.alternative,
            "native_status": self.native_status,
        }

    def format(self) -> str:
        where = f" at {self.source}" if self.source else ""
        op = f" {self.op}" if self.op else ""
        return f"{self.code_name}{where}{op}: {self.message}"


class BackendError(Exception):
    """Exception carrying one or more :class:`Diagnostic` objects."""

    def __init__(self, code: int, message: str, *, stage: str = "unknown",
                 source: str = "", op: str = "", action_id: int | None = None,
                 target_profile: str = "", actual: Any = None,
                 required: Any = None, alternative: str = "",
                 native_status: int | None = None):
        self.diagnostics = [Diagnostic(
            code=code, stage=stage, message=message, source=source, op=op,
            action_id=action_id, target_profile=target_profile, actual=actual,
            required=required, alternative=alternative,
            native_status=native_status,
        )]
        super().__init__(self.diagnostics[0].format())

    @classmethod
    def from_diagnostics(cls, diagnostics: list[Diagnostic]) -> "BackendError":
        if not diagnostics:
            raise ValueError("at least one diagnostic is required")
        obj = cls(diagnostics[0].code, diagnostics[0].message,
                  stage=diagnostics[0].stage, source=diagnostics[0].source,
                  op=diagnostics[0].op, action_id=diagnostics[0].action_id,
                  target_profile=diagnostics[0].target_profile,
                  actual=diagnostics[0].actual, required=diagnostics[0].required,
                  alternative=diagnostics[0].alternative,
                  native_status=diagnostics[0].native_status)
        obj.diagnostics = list(diagnostics)
        return obj

    @property
    def code(self) -> int:
        return self.diagnostics[0].code

    def to_list(self) -> list[dict[str, Any]]:
        return [d.to_dict() for d in self.diagnostics]


def reject(code: int, message: str, **kwargs: Any) -> None:
    """Raise a one-diagnostic :class:`BackendError`."""
    raise BackendError(code, message, **kwargs)
