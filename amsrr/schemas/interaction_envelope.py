from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from amsrr.schemas.common import ContactMode, SchemaBase, SchemaValidationError, require_non_empty, require_non_negative


@dataclass
class TargetRegionSet(SchemaBase):
    entity_id: str
    region_ids: list[str] = field(default_factory=list)
    region_types: list[str] = field(default_factory=list)

    def validate(self) -> None:
        require_non_empty(self.entity_id, "TargetRegionSet.entity_id")


@dataclass
class WrenchSpaceRequirement(SchemaBase):
    applies_to: str
    effect: str
    lower_bound_description: str | None = None
    wrench_lower: list[float] | None = None
    wrench_upper: list[float] | None = None
    target_wrench: list[float] | None = None
    priority: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        require_non_empty(self.applies_to, "WrenchSpaceRequirement.applies_to")
        require_non_empty(self.effect, "WrenchSpaceRequirement.effect")
        if self.wrench_lower is not None and len(self.wrench_lower) != 6:
            raise SchemaValidationError("WrenchSpaceRequirement.wrench_lower must have length 6")
        if self.wrench_upper is not None and len(self.wrench_upper) != 6:
            raise SchemaValidationError("WrenchSpaceRequirement.wrench_upper must have length 6")
        if self.target_wrench is not None and len(self.target_wrench) != 6:
            raise SchemaValidationError("WrenchSpaceRequirement.target_wrench must have length 6")


@dataclass
class SupportRatioRequirement(SchemaBase):
    min_contact_support_ratio: float | None = None
    max_vertical_thrust_ratio: float | None = None
    allow_thrust_for_stabilization: bool = True

    def validate(self) -> None:
        if self.min_contact_support_ratio is not None:
            require_non_negative(self.min_contact_support_ratio, "SupportRatioRequirement.min_contact_support_ratio")
        if self.max_vertical_thrust_ratio is not None:
            require_non_negative(self.max_vertical_thrust_ratio, "SupportRatioRequirement.max_vertical_thrust_ratio")


@dataclass
class PrecisionRequirement(SchemaBase):
    target: str
    tolerance_pos_m: float | None = None
    tolerance_rot_rad: float | None = None
    tolerance_q: list[float] | None = None

    def validate(self) -> None:
        require_non_empty(self.target, "PrecisionRequirement.target")
        if self.tolerance_pos_m is not None:
            require_non_negative(self.tolerance_pos_m, "PrecisionRequirement.tolerance_pos_m")
        if self.tolerance_rot_rad is not None:
            require_non_negative(self.tolerance_rot_rad, "PrecisionRequirement.tolerance_rot_rad")


@dataclass
class DurationRequirement(SchemaBase):
    phase_label: str | None = None
    min_duration_s: float | None = None
    max_duration_s: float | None = None

    def validate(self) -> None:
        if self.min_duration_s is not None:
            require_non_negative(self.min_duration_s, "DurationRequirement.min_duration_s")
        if self.max_duration_s is not None:
            require_non_negative(self.max_duration_s, "DurationRequirement.max_duration_s")
        if self.min_duration_s is not None and self.max_duration_s is not None and self.max_duration_s < self.min_duration_s:
            raise SchemaValidationError("DurationRequirement.max_duration_s must be >= min_duration_s")


@dataclass
class CapabilityRequirement(SchemaBase):
    capability_type: str
    min_force_n: float | None = None
    min_torque_nm: float | None = None
    pose_accuracy_m: float | None = None
    pose_accuracy_rad: float | None = None
    stiffness_requirement: float | None = None

    def validate(self) -> None:
        require_non_empty(self.capability_type, "CapabilityRequirement.capability_type")
        for name in ("min_force_n", "min_torque_nm", "pose_accuracy_m", "pose_accuracy_rad", "stiffness_requirement"):
            value = getattr(self, name)
            if value is not None:
                require_non_negative(value, f"CapabilityRequirement.{name}")


@dataclass
class EnvelopeBranchOption(SchemaBase):
    branch_id: str
    label: str
    required_contact_modes: list[ContactMode] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        require_non_empty(self.branch_id, "EnvelopeBranchOption.branch_id")


@dataclass
class InteractionEnvelope(SchemaBase):
    envelope_id: str
    task_id: str
    required_contact_count_range: tuple[int, int]
    required_contact_modes: list[ContactMode]
    target_region_sets: list[TargetRegionSet]
    wrench_space_requirements: list[WrenchSpaceRequirement]
    support_ratio_requirements: SupportRatioRequirement | None = None
    vertical_thrust_ratio_limit: float | None = None
    precision_requirements: list[PrecisionRequirement] = field(default_factory=list)
    duration_requirements: list[DurationRequirement] = field(default_factory=list)
    capability_requirements: list[CapabilityRequirement] = field(default_factory=list)
    branch_options: list[EnvelopeBranchOption] = field(default_factory=list)

    def validate(self) -> None:
        require_non_empty(self.envelope_id, "InteractionEnvelope.envelope_id")
        require_non_empty(self.task_id, "InteractionEnvelope.task_id")
        low, high = self.required_contact_count_range
        if low < 0 or high < low:
            raise SchemaValidationError("InteractionEnvelope.required_contact_count_range must be [min, max] with max >= min")
        if self.vertical_thrust_ratio_limit is not None:
            require_non_negative(self.vertical_thrust_ratio_limit, "InteractionEnvelope.vertical_thrust_ratio_limit")

