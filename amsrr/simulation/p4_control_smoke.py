from __future__ import annotations

from dataclasses import dataclass, field

from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_non_empty


P4_CONTROL_REQUIRED_SMOKES = (
    "single_module_hover",
    "fixed_morphology_hover",
    "fixed_morphology_waypoint",
)


@dataclass
class P4ControlSmokeResult(SchemaBase):
    smoke_name: str
    attempted: bool
    passed: bool
    skipped: bool = False
    isaac_backed: bool = False
    backend: str = "isaac_lab"
    skip_reason: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        require_non_empty(self.smoke_name, "P4ControlSmokeResult.smoke_name")
        require_non_empty(self.backend, "P4ControlSmokeResult.backend")
        if self.passed and (not self.attempted or self.skipped):
            raise SchemaValidationError("P4ControlSmokeResult cannot pass when not attempted or skipped")
