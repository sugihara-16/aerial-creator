from __future__ import annotations

from dataclasses import dataclass, field

from amsrr.schemas.common import SchemaBase, StrEnum, require_non_empty


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

