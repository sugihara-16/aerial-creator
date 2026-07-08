from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from amsrr.assembly.construction_state import AssemblyStep, ConstructionState
from amsrr.schemas.common import SchemaBase
from amsrr.schemas.feasibility import Violation


@dataclass
class AssemblyExecutionResult(SchemaBase):
    step_id: int
    success: bool
    updated_state: ConstructionState | None = None
    violations: list[Violation] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    message: str | None = None

    def validate(self) -> None:
        if self.step_id < 0:
            from amsrr.schemas.common import SchemaValidationError

            raise SchemaValidationError("AssemblyExecutionResult.step_id must be non-negative")


class AssemblyExecutorInterface(Protocol):
    """Execution backend contract for future simulator/real assembly code."""

    def execute_step(self, step: AssemblyStep, state: ConstructionState) -> AssemblyExecutionResult:
        ...
