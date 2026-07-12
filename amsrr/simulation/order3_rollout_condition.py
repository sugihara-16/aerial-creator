from __future__ import annotations

"""Deterministic target generation for hash-bound Order-3 rollout conditions."""

from dataclasses import dataclass
import math

from amsrr.schemas.common import Pose7D, SchemaBase, SchemaValidationError, require_len
from amsrr.schemas.order3_rollout_condition import Order3RolloutCondition


ORDER3_FREE_FLIGHT_REPORT_VERSION = "order3_free_flight_isaac_report_v1"


@dataclass
class Order3ConditionTarget(SchemaBase):
    desired_pose_world: Pose7D
    desired_twist_world: list[float]
    ramp_progress: float
    hold_active: bool

    def validate(self) -> None:
        require_len(self.desired_pose_world, 7, "Order3ConditionTarget.desired_pose_world")
        require_len(self.desired_twist_world, 6, "Order3ConditionTarget.desired_twist_world")
        values = [*self.desired_pose_world, *self.desired_twist_world]
        if not all(math.isfinite(float(value)) for value in values):
            raise SchemaValidationError("Order3ConditionTarget values must be finite")
        if not 0.0 <= float(self.ramp_progress) <= 1.0:
            raise SchemaValidationError("Order3ConditionTarget.ramp_progress must be in [0, 1]")
        if type(self.hold_active) is not bool:
            raise SchemaValidationError("Order3ConditionTarget.hold_active must be boolean")


@dataclass
class Order3ConditionRealization(SchemaBase):
    condition_hash: str
    task_mode: str
    requested_initial_root_pose_world: list[float]
    applied_initial_root_pose_world: list[float]
    requested_initial_twist_world: list[float]
    applied_initial_twist_world: list[float]
    requested_mass_scale: float
    applied_mass_scale: float
    requested_inertia_scale: float
    applied_inertia_scale: float
    requested_thrust_scale: float
    applied_thrust_scale: float
    mass_randomization_applied: bool
    inertia_randomization_applied: bool
    thrust_randomization_applied: bool
    initial_state_applied: bool
    final_target_pose_world: list[float]
    final_target_twist_world: list[float]

    def validate(self) -> None:
        if len(self.condition_hash) != 64 or any(
            character not in "0123456789abcdef" for character in self.condition_hash
        ):
            raise SchemaValidationError("Order3ConditionRealization.condition_hash must be sha256")
        if self.task_mode not in {"hover", "waypoint", "takeoff"}:
            raise SchemaValidationError("Order3ConditionRealization.task_mode is invalid")
        for name, width in (
            ("requested_initial_root_pose_world", 7),
            ("applied_initial_root_pose_world", 7),
            ("requested_initial_twist_world", 6),
            ("applied_initial_twist_world", 6),
            ("final_target_pose_world", 7),
            ("final_target_twist_world", 6),
        ):
            values = getattr(self, name)
            require_len(values, width, f"Order3ConditionRealization.{name}")
            if not all(math.isfinite(float(value)) for value in values):
                raise SchemaValidationError(f"Order3ConditionRealization.{name} must be finite")
        for name in (
            "requested_mass_scale",
            "applied_mass_scale",
            "requested_inertia_scale",
            "applied_inertia_scale",
            "requested_thrust_scale",
            "applied_thrust_scale",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise SchemaValidationError(
                    f"Order3ConditionRealization.{name} must be finite and positive"
                )
        for name in (
            "mass_randomization_applied",
            "inertia_randomization_applied",
            "thrust_randomization_applied",
            "initial_state_applied",
        ):
            if type(getattr(self, name)) is not bool:
                raise SchemaValidationError(
                    f"Order3ConditionRealization.{name} must be boolean"
                )


class Order3ConditionTargetScheduler:
    """Generate hover/waypoint targets from a nominal in-air centroidal pose.

    Initial-state perturbations are applied by the simulator reset path.  The
    scheduler deliberately targets the unperturbed nominal hover pose so that
    hover episodes exercise recovery.  Waypoint episodes then smoothly ramp
    from that pose to the requested translation and full RPY orientation.
    """

    def __init__(
        self,
        condition: Order3RolloutCondition,
        *,
        nominal_hover_pose_world: Pose7D,
    ) -> None:
        if condition.task_mode not in {"hover", "waypoint"}:
            raise SchemaValidationError(
                "Order3ConditionTargetScheduler only supports hover/waypoint"
            )
        require_len(nominal_hover_pose_world, 7, "nominal_hover_pose_world")
        self.condition = condition
        self.nominal_pose = tuple(float(value) for value in nominal_hover_pose_world)
        self.final_pose: Pose7D
        if condition.task_mode == "hover":
            self.final_pose = self.nominal_pose
        else:
            waypoint_rotation = rpy_to_quat_xyzw(
                condition.waypoint_orientation_rpy_rad
            )
            final_rotation = quaternion_multiply_xyzw(
                tuple(self.nominal_pose[3:7]), waypoint_rotation
            )
            self.final_pose = (
                self.nominal_pose[0]
                + float(condition.waypoint_position_offset_world[0]),
                self.nominal_pose[1]
                + float(condition.waypoint_position_offset_world[1]),
                self.nominal_pose[2]
                + float(condition.waypoint_position_offset_world[2]),
                *final_rotation,
            )

    def target_at(self, elapsed_s: float) -> Order3ConditionTarget:
        if not math.isfinite(float(elapsed_s)) or elapsed_s < 0.0:
            raise SchemaValidationError("Order3 target elapsed_s must be finite and non-negative")
        if self.condition.task_mode == "hover":
            return Order3ConditionTarget(
                desired_pose_world=self.nominal_pose,
                desired_twist_world=[0.0] * 6,
                ramp_progress=1.0,
                hold_active=True,
            )
        linear = min(max(float(elapsed_s) / self.condition.waypoint_ramp_s, 0.0), 1.0)
        smooth = linear * linear * (3.0 - 2.0 * linear)
        position = tuple(
            self.nominal_pose[index]
            + (self.final_pose[index] - self.nominal_pose[index]) * smooth
            for index in range(3)
        )
        orientation = quaternion_slerp_xyzw(
            tuple(self.nominal_pose[3:7]), tuple(self.final_pose[3:7]), smooth
        )
        return Order3ConditionTarget(
            desired_pose_world=(*position, *orientation),
            desired_twist_world=[0.0] * 6,
            ramp_progress=smooth,
            hold_active=linear >= 1.0,
        )

    @property
    def terminal_evidence_start_s(self) -> float:
        """Earliest time from which terminal dwell may be accumulated.

        A randomized episode must not pass on dwell accumulated before its
        disturbance.  Finite disturbances require recovery after the wrench
        ends; a persistent disturbance (duration zero) requires dwell under
        load after it begins.  Waypoint dwell likewise starts only after the
        target ramp is complete.
        """

        return order3_terminal_evidence_start_s(self.condition)


def order3_terminal_evidence_start_s(
    condition: Order3RolloutCondition,
) -> float:
    """Return the start of admissible terminal dwell for any Order-3 task."""

    target_ready_s = (
        float(condition.waypoint_ramp_s)
        if condition.task_mode == "waypoint"
        else 0.0
    )
    has_disturbance = any(
        abs(float(value)) > 1.0e-12 for value in condition.external_wrench_body
    )
    if not has_disturbance:
        return target_ready_s
    disturbance_ready_s = float(condition.disturbance_start_s)
    if condition.disturbance_duration_s > 0.0:
        disturbance_ready_s += float(condition.disturbance_duration_s)
    return max(target_ready_s, disturbance_ready_s)


def order3_tracking_window_start_s(
    condition: Order3RolloutCondition,
) -> float:
    """Start of the paired tracking-cost window for a rollout condition.

    Disturbed comparisons measure response and recovery beginning at wrench
    onset.  Nominal hover/waypoint/takeoff comparisons use the whole controlled
    horizon.  Learned and baseline rollouts share the exact hash-bound value.
    """

    if any(abs(float(value)) > 1.0e-12 for value in condition.external_wrench_body):
        return float(condition.disturbance_start_s)
    return 0.0


def rpy_to_quat_xyzw(values: list[float] | tuple[float, float, float]) -> tuple[float, float, float, float]:
    require_len(values, 3, "rpy")
    roll, pitch, yaw = (float(value) for value in values)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return normalize_quaternion_xyzw(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )
    )


def quaternion_multiply_xyzw(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = normalize_quaternion_xyzw(left)
    rx, ry, rz, rw = normalize_quaternion_xyzw(right)
    return normalize_quaternion_xyzw(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        )
    )


def quaternion_slerp_xyzw(
    start: tuple[float, float, float, float],
    end: tuple[float, float, float, float],
    ratio: float,
) -> tuple[float, float, float, float]:
    start_q = normalize_quaternion_xyzw(start)
    end_q = normalize_quaternion_xyzw(end)
    dot = sum(left * right for left, right in zip(start_q, end_q, strict=True))
    if dot < 0.0:
        end_q = tuple(-value for value in end_q)
        dot = -dot
    dot = min(max(dot, -1.0), 1.0)
    t = min(max(float(ratio), 0.0), 1.0)
    if dot > 0.9995:
        return normalize_quaternion_xyzw(
            tuple(start_q[index] + t * (end_q[index] - start_q[index]) for index in range(4))
        )
    angle = math.acos(dot)
    denominator = math.sin(angle)
    left_scale = math.sin((1.0 - t) * angle) / denominator
    right_scale = math.sin(t * angle) / denominator
    return normalize_quaternion_xyzw(
        tuple(
            left_scale * start_q[index] + right_scale * end_q[index]
            for index in range(4)
        )
    )


def normalize_quaternion_xyzw(
    value: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(component * component for component in value))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise SchemaValidationError("quaternion must be finite and non-zero")
    return tuple(component / norm for component in value)  # type: ignore[return-value]


__all__ = [
    "ORDER3_FREE_FLIGHT_REPORT_VERSION",
    "Order3ConditionRealization",
    "Order3ConditionTarget",
    "Order3ConditionTargetScheduler",
    "order3_terminal_evidence_start_s",
    "order3_tracking_window_start_s",
    "quaternion_multiply_xyzw",
    "quaternion_slerp_xyzw",
    "rpy_to_quat_xyzw",
]
