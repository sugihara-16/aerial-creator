from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from amsrr.schemas.common import SchemaBase, SchemaValidationError, StrEnum, require_non_empty
from amsrr.schemas.contact_candidates import AssignmentFeasibilityResult
from amsrr.schemas.policies import ContactWrenchContractVersion


class ViolationSeverity(StrEnum):
    HARD = "hard"
    SOFT = "soft"
    WARNING = "warning"


@dataclass
class Violation(SchemaBase):
    code: str
    severity: ViolationSeverity
    message: str
    node_or_edge_ref: str | None = None
    margin: float | None = None
    threshold: float | None = None

    def validate(self) -> None:
        require_non_empty(self.code, "Violation.code")
        require_non_empty(self.message, "Violation.message")


@dataclass
class FeasibilityResult(SchemaBase):
    feasible: bool
    hard_violations: list[Violation]
    soft_violations: list[Violation]
    margins: dict[str, float]
    proxy_scores: dict[str, float]
    checker_version: str
    metadata: dict[str, str | float | int | bool] = field(default_factory=dict)

    def validate(self) -> None:
        require_non_empty(self.checker_version, "FeasibilityResult.checker_version")
        if self.feasible and self.hard_violations:
            from amsrr.schemas.common import SchemaValidationError

            raise SchemaValidationError("FeasibilityResult cannot be feasible with hard_violations")


@dataclass
class TrajectoryKnotFeasibilityResult(SchemaBase):
    """Auditable result for one unmodified high-level trajectory knot."""

    knot_index: int
    t_rel_s: float
    assignment_result: AssignmentFeasibilityResult
    qp_evaluated: bool
    collision_evaluated: bool
    wrench_evaluated: bool
    margins: dict[str, float] = field(default_factory=dict)
    violation_codes: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.knot_index < 0:
            raise SchemaValidationError(
                "TrajectoryKnotFeasibilityResult.knot_index must be non-negative"
            )
        if not math.isfinite(float(self.t_rel_s)) or self.t_rel_s < 0.0:
            raise SchemaValidationError(
                "TrajectoryKnotFeasibilityResult.t_rel_s must be finite and non-negative"
            )
        if len(self.violation_codes) != len(set(self.violation_codes)):
            raise SchemaValidationError(
                "TrajectoryKnotFeasibilityResult.violation_codes must be unique"
            )
        for name, value in self.margins.items():
            require_non_empty(name, "TrajectoryKnotFeasibilityResult.margins.key")
            if not math.isfinite(float(value)):
                raise SchemaValidationError(
                    f"TrajectoryKnotFeasibilityResult.margins[{name!r}] must be finite"
                )


@dataclass
class TrajectoryFeasibilityResult(SchemaBase):
    """Hard-check result for a complete ``ContactWrenchTrajectory`` proposal.

    Evaluation flags on each knot prevent a structural proxy from being
    mistaken for a completed QP/collision/wrench check.
    """

    feasible: bool
    hard_violations: list[Violation]
    warnings: list[Violation]
    knot_results: list[TrajectoryKnotFeasibilityResult]
    margins: dict[str, float]
    checker_version: str
    contract_version: ContactWrenchContractVersion
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        require_non_empty(
            self.checker_version,
            "TrajectoryFeasibilityResult.checker_version",
        )
        if self.feasible and self.hard_violations:
            raise SchemaValidationError(
                "TrajectoryFeasibilityResult cannot be feasible with hard_violations"
            )
        for name, value in self.margins.items():
            require_non_empty(name, "TrajectoryFeasibilityResult.margins.key")
            if not math.isfinite(float(value)):
                raise SchemaValidationError(
                    f"TrajectoryFeasibilityResult.margins[{name!r}] must be finite"
                )
        indices = [result.knot_index for result in self.knot_results]
        if indices != list(range(len(indices))):
            raise SchemaValidationError(
                "TrajectoryFeasibilityResult.knot_results must use contiguous ordered indices"
            )
