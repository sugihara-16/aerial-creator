from __future__ import annotations

"""Versioned contracts for P4-full Order 4 free-flight planning."""

from dataclasses import dataclass
import math
from typing import Any, Literal, Mapping, Sequence

from amsrr.schemas.common import (
    SchemaBase,
    SchemaValidationError,
    canonical_json,
    require_len,
    require_non_empty,
)
from amsrr.schemas.policies import InteractionKnot
from amsrr.utils.hashing import stable_hash


ORDER4_FREE_FLIGHT_MISSION_VERSION = "order4_free_flight_mission_v1"
ORDER4_FREE_FLIGHT_RUNTIME_VERSION = "order4_free_flight_runtime_v1"
ORDER4_FREE_FLIGHT_REPORT_VERSION = "order4_free_flight_isaac_report_v1"
Order4FreeFlightPhase = Literal[
    "floor_settle",
    "takeoff",
    "hover_acquisition",
    "waypoint",
    "final_hover",
    "complete",
    "safe_hold",
]
Order4ReachabilityStatus = Literal[
    "not_applicable_no_active_assignments",
    "required_and_passed",
    "required_and_failed",
]


@dataclass
class Order4FreeFlightWaypoint(SchemaBase):
    waypoint_id: str
    position_offset_world: list[float]
    orientation_rpy_rad: list[float]
    transition_duration_s: float = 2.0
    dwell_s: float = 0.5
    timeout_s: float = 8.0

    def validate(self) -> None:
        require_non_empty(self.waypoint_id, "Order4FreeFlightWaypoint.waypoint_id")
        require_len(
            self.position_offset_world,
            3,
            "Order4FreeFlightWaypoint.position_offset_world",
        )
        require_len(
            self.orientation_rpy_rad,
            3,
            "Order4FreeFlightWaypoint.orientation_rpy_rad",
        )
        for name, values in (
            ("position_offset_world", self.position_offset_world),
            ("orientation_rpy_rad", self.orientation_rpy_rad),
        ):
            if not all(math.isfinite(float(value)) for value in values):
                raise SchemaValidationError(
                    f"Order4FreeFlightWaypoint.{name} must be finite"
                )
        if any(abs(float(value)) > 2.0 for value in self.position_offset_world):
            raise SchemaValidationError(
                "Order4FreeFlightWaypoint.position_offset_world exceeds the 2 m Order-4 bound"
            )
        if any(abs(float(value)) > math.pi for value in self.orientation_rpy_rad):
            raise SchemaValidationError(
                "Order4FreeFlightWaypoint.orientation_rpy_rad must be in [-pi, pi]"
            )
        for name in ("transition_duration_s", "timeout_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise SchemaValidationError(
                    f"Order4FreeFlightWaypoint.{name} must be finite and positive"
                )
        if not math.isfinite(float(self.dwell_s)) or self.dwell_s < 0.0:
            raise SchemaValidationError(
                "Order4FreeFlightWaypoint.dwell_s must be finite and non-negative"
            )
        if self.timeout_s + 1.0e-12 < self.transition_duration_s + self.dwell_s:
            raise SchemaValidationError(
                "Order4FreeFlightWaypoint.timeout_s must cover transition_duration_s + dwell_s"
            )


@dataclass
class Order4FreeFlightMission(SchemaBase):
    mission_version: str
    mission_id: str
    floor_initialized: bool
    waypoints: list[Order4FreeFlightWaypoint]
    hover_height_delta_m: float = 0.5
    hover_acquisition_dwell_s: float = 0.5
    final_hover_hold_s: float = 5.0
    mission_timeout_s: float = 45.0
    mission_hash: str = ""

    def validate(self) -> None:
        if self.mission_version != ORDER4_FREE_FLIGHT_MISSION_VERSION:
            raise SchemaValidationError(
                "Order4FreeFlightMission.mission_version mismatch"
            )
        require_non_empty(self.mission_id, "Order4FreeFlightMission.mission_id")
        if self.floor_initialized is not True:
            raise SchemaValidationError(
                "Order4FreeFlightMission requires floor_initialized=true"
            )
        if len(self.waypoints) < 2:
            raise SchemaValidationError(
                "Order4FreeFlightMission requires at least two waypoints"
            )
        if len(self.waypoints) > 16:
            raise SchemaValidationError(
                "Order4FreeFlightMission supports at most 16 waypoints"
            )
        waypoint_ids = [waypoint.waypoint_id for waypoint in self.waypoints]
        if len(waypoint_ids) != len(set(waypoint_ids)):
            raise SchemaValidationError(
                "Order4FreeFlightMission waypoint ids must be unique"
            )
        for name in (
            "hover_height_delta_m",
            "hover_acquisition_dwell_s",
            "final_hover_hold_s",
            "mission_timeout_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise SchemaValidationError(
                    f"Order4FreeFlightMission.{name} must be finite and positive"
                )
        if not 0.1 <= self.hover_height_delta_m <= 2.0:
            raise SchemaValidationError(
                "Order4FreeFlightMission.hover_height_delta_m must be in [0.1, 2.0]"
            )
        minimum_duration = sum(
            waypoint.transition_duration_s + waypoint.dwell_s
            for waypoint in self.waypoints
        ) + self.hover_acquisition_dwell_s + self.final_hover_hold_s
        if self.mission_timeout_s + 1.0e-12 < minimum_duration:
            raise SchemaValidationError(
                "Order4FreeFlightMission.mission_timeout_s is shorter than the requested mission dwell"
            )
        expected_hash = order4_free_flight_mission_hash(self.to_dict())
        if self.mission_hash != expected_hash:
            raise SchemaValidationError(
                "Order4FreeFlightMission.mission_hash does not match its canonical payload"
            )

    def identity_payload(self) -> dict[str, Any]:
        return order4_free_flight_mission_payload(self.to_dict())

    def to_canonical_json(self) -> str:
        return canonical_json(self)


@dataclass
class Order4DeterministicPlannerConfig(SchemaBase):
    update_rate_hz: float = 2.0
    horizon_s: float = 2.0
    knot_dt_s: float = 0.25
    floor_settle_duration_s: float = 1.0
    floor_settle_dwell_s: float = 0.25
    takeoff_duration_s: float = 2.0
    hover_acquisition_timeout_s: float = 5.0
    position_tolerance_m: float = 0.20
    attitude_tolerance_rad: float = 0.25
    linear_speed_tolerance_mps: float = 0.15
    angular_speed_tolerance_rad_s: float = 0.25
    max_tilt_rad: float = 1.2
    trajectory_expiry_grace_s: float = 0.25

    def validate(self) -> None:
        for name in (
            "update_rate_hz",
            "horizon_s",
            "knot_dt_s",
            "floor_settle_duration_s",
            "floor_settle_dwell_s",
            "takeoff_duration_s",
            "hover_acquisition_timeout_s",
            "position_tolerance_m",
            "attitude_tolerance_rad",
            "linear_speed_tolerance_mps",
            "angular_speed_tolerance_rad_s",
            "max_tilt_rad",
            "trajectory_expiry_grace_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise SchemaValidationError(
                    f"Order4DeterministicPlannerConfig.{name} must be finite and positive"
                )
        if self.knot_dt_s > self.horizon_s:
            raise SchemaValidationError(
                "Order4DeterministicPlannerConfig.knot_dt_s must not exceed horizon_s"
            )
        if self.floor_settle_dwell_s > self.floor_settle_duration_s:
            raise SchemaValidationError(
                "Order4DeterministicPlannerConfig.floor_settle_dwell_s must not exceed floor_settle_duration_s"
            )
        if self.update_period_s + 1.0e-12 >= self.horizon_s:
            raise SchemaValidationError(
                "Order4 planner update period must be shorter than its trajectory horizon"
            )

    @property
    def update_period_s(self) -> float:
        return 1.0 / self.update_rate_hz


@dataclass
class Order4TrajectoryRuntimeStep(SchemaBase):
    runtime_version: str
    time_s: float
    mission_hash: str
    phase: Order4FreeFlightPhase
    waypoint_index: int | None
    mission_progress_ratio: float
    plan_sequence: int
    plan_start_time_s: float
    plan_elapsed_s: float
    active_knot_index: int
    next_knot_index: int
    interpolation_ratio: float
    replanned: bool
    safe_hold_active: bool
    failure_reason: str | None
    reachability_status: Order4ReachabilityStatus
    active_knot: InteractionKnot

    def validate(self) -> None:
        if self.runtime_version != ORDER4_FREE_FLIGHT_RUNTIME_VERSION:
            raise SchemaValidationError(
                "Order4TrajectoryRuntimeStep.runtime_version mismatch"
            )
        if not math.isfinite(float(self.time_s)) or self.time_s < 0.0:
            raise SchemaValidationError(
                "Order4TrajectoryRuntimeStep.time_s must be finite and non-negative"
            )
        if not math.isfinite(float(self.plan_start_time_s)):
            raise SchemaValidationError(
                "Order4TrajectoryRuntimeStep.plan_start_time_s must be finite"
            )
        if not math.isfinite(float(self.plan_elapsed_s)) or self.plan_elapsed_s < 0.0:
            raise SchemaValidationError(
                "Order4TrajectoryRuntimeStep.plan_elapsed_s must be finite and non-negative"
            )
        if not 0.0 <= self.mission_progress_ratio <= 1.0:
            raise SchemaValidationError(
                "Order4TrajectoryRuntimeStep.mission_progress_ratio must be in [0, 1]"
            )
        if self.plan_sequence < 1:
            raise SchemaValidationError(
                "Order4TrajectoryRuntimeStep.plan_sequence must be positive"
            )
        if min(self.active_knot_index, self.next_knot_index) < 0:
            raise SchemaValidationError(
                "Order4TrajectoryRuntimeStep knot indices must be non-negative"
            )
        if not 0.0 <= self.interpolation_ratio <= 1.0:
            raise SchemaValidationError(
                "Order4TrajectoryRuntimeStep.interpolation_ratio must be in [0, 1]"
            )
        if self.waypoint_index is not None and self.waypoint_index < 0:
            raise SchemaValidationError(
                "Order4TrajectoryRuntimeStep.waypoint_index must be non-negative"
            )
        if self.safe_hold_active != (self.phase == "safe_hold"):
            raise SchemaValidationError(
                "Order4TrajectoryRuntimeStep safe-hold flag and phase disagree"
            )
        if self.reachability_status == "not_applicable_no_active_assignments":
            if self.active_knot.contact_assignments:
                raise SchemaValidationError(
                    "Order4 reachability cannot be not-applicable with active assignments"
                )


def order4_free_flight_mission_payload(
    value: Order4FreeFlightMission | Mapping[str, Any],
) -> dict[str, Any]:
    data = value.to_dict() if isinstance(value, Order4FreeFlightMission) else dict(value)
    data.pop("mission_hash", None)
    return data


def order4_free_flight_mission_hash(
    value: Order4FreeFlightMission | Mapping[str, Any],
) -> str:
    return stable_hash(order4_free_flight_mission_payload(value))


def build_order4_free_flight_mission(
    *,
    mission_id: str,
    waypoints: Sequence[Order4FreeFlightWaypoint | Mapping[str, Any]],
    hover_height_delta_m: float = 0.5,
    hover_acquisition_dwell_s: float = 0.5,
    final_hover_hold_s: float = 5.0,
    mission_timeout_s: float = 45.0,
) -> Order4FreeFlightMission:
    serialized_waypoints = [
        waypoint.to_dict()
        if isinstance(waypoint, Order4FreeFlightWaypoint)
        else dict(waypoint)
        for waypoint in waypoints
    ]
    payload: dict[str, Any] = {
        "mission_version": ORDER4_FREE_FLIGHT_MISSION_VERSION,
        "mission_id": mission_id,
        "floor_initialized": True,
        "waypoints": serialized_waypoints,
        "hover_height_delta_m": hover_height_delta_m,
        "hover_acquisition_dwell_s": hover_acquisition_dwell_s,
        "final_hover_hold_s": final_hover_hold_s,
        "mission_timeout_s": mission_timeout_s,
    }
    payload["mission_hash"] = order4_free_flight_mission_hash(payload)
    return Order4FreeFlightMission.from_dict(payload)


def default_order4_free_flight_mission(
    *,
    final_hover_hold_s: float = 5.0,
) -> Order4FreeFlightMission:
    return build_order4_free_flight_mission(
        mission_id="order4_default_multi_waypoint",
        waypoints=[
            Order4FreeFlightWaypoint(
                waypoint_id="translate_x",
                position_offset_world=[0.25, 0.0, 0.05],
                orientation_rpy_rad=[0.0, 0.0, 0.0],
                transition_duration_s=2.0,
                dwell_s=0.5,
                timeout_s=8.0,
            ),
            Order4FreeFlightWaypoint(
                waypoint_id="translate_y_attitude",
                position_offset_world=[0.25, 0.20, 0.05],
                orientation_rpy_rad=[0.08, -0.08, 0.20],
                transition_duration_s=2.5,
                dwell_s=0.5,
                timeout_s=9.0,
            ),
            Order4FreeFlightWaypoint(
                waypoint_id="return_hover",
                position_offset_world=[0.0, 0.0, 0.0],
                orientation_rpy_rad=[0.0, 0.0, 0.0],
                transition_duration_s=2.5,
                dwell_s=0.5,
                timeout_s=9.0,
            ),
        ],
        final_hover_hold_s=final_hover_hold_s,
        mission_timeout_s=max(45.0, final_hover_hold_s + 25.0),
    )
