from __future__ import annotations

"""Simulator-evidence reduction for the shadow half of production ``C_H``."""

import math
from dataclasses import dataclass, field
from typing import Sequence

from amsrr.feasibility.contact_wrench_hybrid import (
    active_numeric_object_wrench_requirements,
    intersect_numeric_wrench_requirements,
    wrench_reference_world,
)
from amsrr.geometry.wrench import contact_wrench_to_world
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.schemas.policies import InteractionKnot


SHADOW_CONTACT_WRENCH_METRIC_VERSION = "order9_shadow_contact_wrench_metric_v1"
_ACTIVE_CONTACT_STATES = frozenset({"attach", "maintain", "slide"})


@dataclass(frozen=True)
class MeasuredCandidateWrench:
    candidate_id: int
    wrench_contact: tuple[float, float, float, float, float, float]
    evidence_valid: bool = True
    sample_count: int = 1

    def __post_init__(self) -> None:
        if self.candidate_id < 0 or self.sample_count < 0:
            raise ValueError("measured contact wrench ids/counts must be non-negative")
        if len(self.wrench_contact) != 6 or any(
            not math.isfinite(float(value)) for value in self.wrench_contact
        ):
            raise ValueError("measured contact wrench must contain six finite values")


@dataclass(frozen=True)
class ShadowContactWrenchMetric:
    residual: float
    margins: dict[str, float] = field(default_factory=dict)
    metric_version: str = SHADOW_CONTACT_WRENCH_METRIC_VERSION

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.residual)) or self.residual < 0.0:
            raise ValueError("shadow contact-wrench residual must be non-negative")
        if any(not math.isfinite(float(value)) for value in self.margins.values()):
            raise ValueError("shadow contact-wrench margins must be finite")


def evaluate_shadow_contact_wrench_residual(
    context: HighLevelPolicyContext,
    knot: InteractionKnot,
    measurements: Sequence[MeasuredCandidateWrench],
    *,
    force_scale_n: float = 30.0,
    torque_scale_nm: float = 5.0,
    missing_or_invalid_residual: float = 1.0,
) -> ShadowContactWrenchMetric:
    """Reduce measured per-patch net wrenches to one dimensionless residual.

    The metric checks the executed wrench against the unmodified policy range,
    the circular friction cone, and active numeric object-effect requirements.
    It does not require equality to the policy's preferred target because the
    approved contract gives ``pi_L`` freedom to realize any wrench inside the
    accepted range.
    """

    for name, value in (
        ("force_scale_n", force_scale_n),
        ("torque_scale_nm", torque_scale_nm),
        ("missing_or_invalid_residual", missing_or_invalid_residual),
    ):
        if not math.isfinite(float(value)) or value <= 0.0:
            raise ValueError(f"shadow wrench {name} must be positive")
    measurement_by_id = {item.candidate_id: item for item in measurements}
    if len(measurement_by_id) != len(measurements):
        return _failed_metric(missing_or_invalid_residual, "duplicate_measurement")
    candidate_by_id = {
        item.candidate_id: item
        for item in context.contact_candidate_set.candidates
    }
    active = [
        item
        for item in knot.contact_assignments
        if item.schedule_state in _ACTIVE_CONTACT_STATES
    ]
    scales = [force_scale_n] * 3 + [torque_scale_nm] * 3
    residual = 0.0
    minimum_cone_margin = math.inf
    measured_active_count = 0
    net_force = [0.0, 0.0, 0.0]
    net_torque = [0.0, 0.0, 0.0]
    candidates = []
    resolved: list[tuple[object, MeasuredCandidateWrench]] = []
    for assignment in active:
        candidate = candidate_by_id.get(assignment.candidate_id)
        measurement = measurement_by_id.get(assignment.candidate_id)
        if (
            candidate is None
            or measurement is None
            or not measurement.evidence_valid
            or measurement.sample_count < 1
            or assignment.wrench_lower is None
            or assignment.wrench_upper is None
            or candidate.friction is None
        ):
            return _failed_metric(
                missing_or_invalid_residual,
                "missing_or_invalid_active_measurement",
            )
        values = [float(value) for value in measurement.wrench_contact]
        for value, lower, upper, scale in zip(
            values,
            assignment.wrench_lower,
            assignment.wrench_upper,
            scales,
        ):
            residual = max(
                residual,
                max(float(lower) - value, value - float(upper), 0.0) / scale,
            )
        world = contact_wrench_to_world(values, candidate)
        inward = tuple(-float(value) for value in candidate.normal_world)
        normal = sum(world[index] * inward[index] for index in range(3))
        tangent = math.sqrt(
            max(
                0.0,
                sum(world[index] ** 2 for index in range(3)) - normal**2,
            )
        )
        cone_margin = float(candidate.friction) * normal - tangent
        minimum_cone_margin = min(minimum_cone_margin, cone_margin)
        residual = max(
            residual,
            max(-normal, -cone_margin, 0.0) / force_scale_n,
        )
        candidates.append(candidate)
        resolved.append((candidate, measurement))
        measured_active_count += 1

    if candidates:
        reference = wrench_reference_world(candidates, context)
        for candidate, measurement in resolved:
            world = contact_wrench_to_world(measurement.wrench_contact, candidate)
            for axis in range(3):
                net_force[axis] += float(world[axis])
                net_torque[axis] += float(world[3 + axis])
            arm = [
                float(candidate.contact_pose_world[axis]) - float(reference[axis])
                for axis in range(3)
            ]
            cross = (
                arm[1] * world[2] - arm[2] * world[1],
                arm[2] * world[0] - arm[0] * world[2],
                arm[0] * world[1] - arm[1] * world[0],
            )
            for axis in range(3):
                net_torque[axis] += cross[axis]

    requirements = active_numeric_object_wrench_requirements(context, knot)
    if requirements:
        if not candidates or len({item.target_entity_id for item in candidates}) != 1:
            return _failed_metric(
                missing_or_invalid_residual,
                "unresolved_object_wrench_requirement",
            )
        lower, upper = intersect_numeric_wrench_requirements(
            requirements,
            force_scale_n=force_scale_n,
            torque_scale_nm=torque_scale_nm,
        )
        if lower is None or upper is None:
            return _failed_metric(
                missing_or_invalid_residual,
                "invalid_object_wrench_requirement",
            )
        normalized = [
            *(value / force_scale_n for value in net_force),
            *(value / torque_scale_nm for value in net_torque),
        ]
        for value, left, right in zip(normalized, lower, upper):
            residual = max(residual, max(left - value, value - right, 0.0))

    cone_margin = (
        0.0 if math.isinf(minimum_cone_margin) else minimum_cone_margin
    )
    return ShadowContactWrenchMetric(
        residual=float(residual),
        margins={
            "measured_active_contact_count": float(measured_active_count),
            "active_assignment_count": float(len(active)),
            "minimum_measured_friction_cone_margin_n": float(cone_margin),
            "measured_net_force_x_n": float(net_force[0]),
            "measured_net_force_y_n": float(net_force[1]),
            "measured_net_force_z_n": float(net_force[2]),
            "measured_net_torque_x_nm": float(net_torque[0]),
            "measured_net_torque_y_nm": float(net_torque[1]),
            "measured_net_torque_z_nm": float(net_torque[2]),
            "active_numeric_wrench_requirement_count": float(len(requirements)),
        },
    )


def _failed_metric(residual: float, reason: str) -> ShadowContactWrenchMetric:
    reason_hash = float(
        sum((index + 1) * ord(character) for index, character in enumerate(reason))
    )
    return ShadowContactWrenchMetric(
        residual=float(residual),
        margins={
            "shadow_wrench_metric_failed": 1.0,
            "shadow_wrench_failure_reason_hash": reason_hash,
        },
    )


__all__ = [
    "SHADOW_CONTACT_WRENCH_METRIC_VERSION",
    "MeasuredCandidateWrench",
    "ShadowContactWrenchMetric",
    "evaluate_shadow_contact_wrench_residual",
]
