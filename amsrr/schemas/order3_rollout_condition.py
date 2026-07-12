from __future__ import annotations

"""Canonical, hash-bound simulator condition for Order-3 free-flight rollouts."""

from dataclasses import dataclass, field
import math
from typing import Any, Literal, Mapping, Sequence

from amsrr.schemas.common import (
    SchemaBase,
    SchemaValidationError,
    canonical_json,
    require_len,
    require_non_empty,
)
from amsrr.utils.hashing import stable_hash


ORDER3_ROLLOUT_CONDITION_VERSION = "order3_free_flight_rollout_condition_v1"
Order3RolloutTaskMode = Literal["hover", "waypoint", "takeoff"]


@dataclass
class Order3RolloutCondition(SchemaBase):
    condition_version: str
    stage_id: str
    task_mode: Order3RolloutTaskMode
    seed: int
    initial_position_offset_world: list[float] = field(
        default_factory=lambda: [0.0] * 3
    )
    initial_orientation_rpy_rad: list[float] = field(
        default_factory=lambda: [0.0] * 3
    )
    initial_linear_velocity_world: list[float] = field(
        default_factory=lambda: [0.0] * 3
    )
    initial_angular_velocity_body: list[float] = field(
        default_factory=lambda: [0.0] * 3
    )
    waypoint_position_offset_world: list[float] = field(
        default_factory=lambda: [0.0] * 3
    )
    waypoint_orientation_rpy_rad: list[float] = field(
        default_factory=lambda: [0.0] * 3
    )
    waypoint_ramp_s: float = 1.0
    hold_s: float = 1.0
    external_wrench_body: list[float] = field(default_factory=lambda: [0.0] * 6)
    disturbance_start_s: float = 3.0
    disturbance_duration_s: float = 0.0
    mass_scale: float = 1.0
    inertia_scale: float = 1.0
    thrust_scale: float = 1.0
    condition_hash: str = ""

    def validate(self) -> None:
        if self.condition_version != ORDER3_ROLLOUT_CONDITION_VERSION:
            raise SchemaValidationError(
                "Order3RolloutCondition.condition_version mismatch"
            )
        require_non_empty(self.stage_id, "Order3RolloutCondition.stage_id")
        if self.task_mode not in {"hover", "waypoint", "takeoff"}:
            raise SchemaValidationError(
                "Order3RolloutCondition.task_mode must be hover, waypoint, or takeoff"
            )
        if self.seed < 0:
            raise SchemaValidationError(
                "Order3RolloutCondition.seed must be non-negative"
            )
        vectors = {
            "initial_position_offset_world": (self.initial_position_offset_world, 3),
            "initial_orientation_rpy_rad": (self.initial_orientation_rpy_rad, 3),
            "initial_linear_velocity_world": (self.initial_linear_velocity_world, 3),
            "initial_angular_velocity_body": (self.initial_angular_velocity_body, 3),
            "waypoint_position_offset_world": (self.waypoint_position_offset_world, 3),
            "waypoint_orientation_rpy_rad": (self.waypoint_orientation_rpy_rad, 3),
            "external_wrench_body": (self.external_wrench_body, 6),
        }
        for name, (values, width) in vectors.items():
            require_len(values, width, f"Order3RolloutCondition.{name}")
            if not all(math.isfinite(float(value)) for value in values):
                raise SchemaValidationError(
                    f"Order3RolloutCondition.{name} must be finite"
                )
        bounded_vectors = {
            "initial_position_offset_world": (
                self.initial_position_offset_world,
                10.0,
            ),
            "initial_orientation_rpy_rad": (
                self.initial_orientation_rpy_rad,
                math.pi,
            ),
            "initial_linear_velocity_world": (
                self.initial_linear_velocity_world,
                10.0,
            ),
            "initial_angular_velocity_body": (
                self.initial_angular_velocity_body,
                5.0,
            ),
            "waypoint_position_offset_world": (
                self.waypoint_position_offset_world,
                10.0,
            ),
            "waypoint_orientation_rpy_rad": (
                self.waypoint_orientation_rpy_rad,
                math.pi,
            ),
            "external_wrench_body": (self.external_wrench_body, 10_000.0),
        }
        for name, (values, absolute_limit) in bounded_vectors.items():
            if any(abs(float(value)) > absolute_limit for value in values):
                raise SchemaValidationError(
                    f"Order3RolloutCondition.{name} exceeds its safe absolute bound "
                    f"{absolute_limit}"
                )
        for name in ("waypoint_ramp_s", "hold_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise SchemaValidationError(
                    f"Order3RolloutCondition.{name} must be finite and positive"
                )
        for name in ("disturbance_start_s", "disturbance_duration_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise SchemaValidationError(
                    f"Order3RolloutCondition.{name} must be finite and non-negative"
                )
        for name in ("mass_scale", "inertia_scale", "thrust_scale"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.5 <= value <= 1.5:
                raise SchemaValidationError(
                    f"Order3RolloutCondition.{name} must be finite and in [0.5, 1.5]"
                )
        expected_hash = order3_rollout_condition_hash(self.to_dict())
        if self.condition_hash != expected_hash:
            raise SchemaValidationError(
                "Order3RolloutCondition.condition_hash does not match its canonical payload"
            )

    def identity_payload(self) -> dict[str, Any]:
        return order3_rollout_condition_payload(self.to_dict())

    def to_canonical_json(self) -> str:
        return canonical_json(self)


def order3_rollout_condition_payload(
    value: Order3RolloutCondition | Mapping[str, Any],
) -> dict[str, Any]:
    data = value.to_dict() if isinstance(value, Order3RolloutCondition) else dict(value)
    data.pop("condition_hash", None)
    return data


def order3_rollout_condition_hash(
    value: Order3RolloutCondition | Mapping[str, Any],
) -> str:
    return stable_hash(order3_rollout_condition_payload(value))


def build_order3_rollout_condition(
    *,
    stage_id: str,
    task_mode: Order3RolloutTaskMode,
    seed: int,
    initial_position_offset_world: Sequence[float] = (0.0, 0.0, 0.0),
    initial_orientation_rpy_rad: Sequence[float] = (0.0, 0.0, 0.0),
    initial_linear_velocity_world: Sequence[float] = (0.0, 0.0, 0.0),
    initial_angular_velocity_body: Sequence[float] = (0.0, 0.0, 0.0),
    waypoint_position_offset_world: Sequence[float] = (0.0, 0.0, 0.0),
    waypoint_orientation_rpy_rad: Sequence[float] = (0.0, 0.0, 0.0),
    waypoint_ramp_s: float = 1.0,
    hold_s: float = 1.0,
    external_wrench_body: Sequence[float] = (0.0,) * 6,
    disturbance_start_s: float = 3.0,
    disturbance_duration_s: float = 0.0,
    mass_scale: float = 1.0,
    inertia_scale: float = 1.0,
    thrust_scale: float = 1.0,
) -> Order3RolloutCondition:
    payload: dict[str, Any] = {
        "condition_version": ORDER3_ROLLOUT_CONDITION_VERSION,
        "stage_id": stage_id,
        "task_mode": task_mode,
        "seed": seed,
        "initial_position_offset_world": list(initial_position_offset_world),
        "initial_orientation_rpy_rad": list(initial_orientation_rpy_rad),
        "initial_linear_velocity_world": list(initial_linear_velocity_world),
        "initial_angular_velocity_body": list(initial_angular_velocity_body),
        "waypoint_position_offset_world": list(waypoint_position_offset_world),
        "waypoint_orientation_rpy_rad": list(waypoint_orientation_rpy_rad),
        "waypoint_ramp_s": waypoint_ramp_s,
        "hold_s": hold_s,
        "external_wrench_body": list(external_wrench_body),
        "disturbance_start_s": disturbance_start_s,
        "disturbance_duration_s": disturbance_duration_s,
        "mass_scale": mass_scale,
        "inertia_scale": inertia_scale,
        "thrust_scale": thrust_scale,
    }
    payload["condition_hash"] = order3_rollout_condition_hash(payload)
    return Order3RolloutCondition.from_dict(payload)
