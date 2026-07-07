from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from amsrr.schemas.common import ContactMode, Condition, Pose7D, SchemaBase, Vector3, require_len, require_non_empty


@dataclass
class ContactAssignment(SchemaBase):
    slot_id: int
    anchor_id: int
    candidate_id: int
    contact_mode: ContactMode
    schedule_state: Literal["approach", "attach", "maintain", "slide", "release"]
    wrench_target: list[float] | None = None
    wrench_lower: list[float] | None = None
    wrench_upper: list[float] | None = None
    priority: float = 1.0

    def validate(self) -> None:
        if min(self.slot_id, self.anchor_id, self.candidate_id) < 0:
            from amsrr.schemas.common import SchemaValidationError

            raise SchemaValidationError("ContactAssignment ids must be non-negative")
        for name in ("wrench_target", "wrench_lower", "wrench_upper"):
            value = getattr(self, name)
            if value is not None:
                require_len(value, 6, f"ContactAssignment.{name}")


@dataclass
class CentroidalTarget(SchemaBase):
    com_pos_world: Vector3 | None = None
    com_vel_world: Vector3 | None = None
    body_orientation_world: tuple[float, float, float, float] | None = None
    centroidal_wrench_preference: list[float] | None = None

    def validate(self) -> None:
        if self.com_pos_world is not None:
            require_len(self.com_pos_world, 3, "CentroidalTarget.com_pos_world")
        if self.com_vel_world is not None:
            require_len(self.com_vel_world, 3, "CentroidalTarget.com_vel_world")
        if self.body_orientation_world is not None:
            require_len(self.body_orientation_world, 4, "CentroidalTarget.body_orientation_world")
        if self.centroidal_wrench_preference is not None:
            require_len(self.centroidal_wrench_preference, 6, "CentroidalTarget.centroidal_wrench_preference")


@dataclass
class PostureTarget(SchemaBase):
    joint_pos_target: dict[str, float] | None = None
    joint_vel_target: dict[str, float] | None = None
    free_anchor_pose_targets: dict[int, Pose7D] | None = None


@dataclass
class ObjectTarget(SchemaBase):
    object_id: str
    pose_target_world: Pose7D | None = None
    twist_target_world: list[float] | None = None
    generalized_q_target: list[float] | None = None
    generalized_qdot_target: list[float] | None = None

    def validate(self) -> None:
        require_non_empty(self.object_id, "ObjectTarget.object_id")
        if self.pose_target_world is not None:
            require_len(self.pose_target_world, 7, "ObjectTarget.pose_target_world")
        if self.twist_target_world is not None:
            require_len(self.twist_target_world, 6, "ObjectTarget.twist_target_world")


@dataclass
class InteractionKnot(SchemaBase):
    t_rel_s: float
    contact_assignments: list[ContactAssignment]
    centroidal_target: CentroidalTarget | None = None
    posture_target: PostureTarget | None = None
    object_targets: list[ObjectTarget] = field(default_factory=list)
    priority_weights: dict[str, float] = field(default_factory=dict)
    guard_conditions: list[Condition] = field(default_factory=list)


@dataclass
class ContactWrenchTrajectory(SchemaBase):
    horizon_s: float
    dt_s: float
    knots: list[InteractionKnot]
    derived_mode_label: str | None = None

    def validate(self) -> None:
        if self.horizon_s <= 0.0 or self.dt_s <= 0.0:
            from amsrr.schemas.common import SchemaValidationError

            raise SchemaValidationError("ContactWrenchTrajectory horizon_s and dt_s must be positive")


@dataclass
class PolicyCommand(SchemaBase):
    desired_body_twist: list[float] | None = None
    desired_body_pose: Pose7D | None = None
    desired_anchor_pose_offsets: dict[int, Pose7D] = field(default_factory=dict)
    joint_position_bias: dict[str, float] = field(default_factory=dict)
    joint_velocity_bias: dict[str, float] = field(default_factory=dict)
    residual_wrench_body: list[float] | None = None
    contact_tracking_bias: dict[int, list[float]] = field(default_factory=dict)
    priority_weights: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        if self.desired_body_twist is not None:
            require_len(self.desired_body_twist, 6, "PolicyCommand.desired_body_twist")
        if self.desired_body_pose is not None:
            require_len(self.desired_body_pose, 7, "PolicyCommand.desired_body_pose")
        if self.residual_wrench_body is not None:
            require_len(self.residual_wrench_body, 6, "PolicyCommand.residual_wrench_body")


@dataclass
class ControllerStatus(SchemaBase):
    status: Literal["ok", "warning", "infeasible", "fault"]
    qp_feasible: bool
    active_mode: str | None = None
    message: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class ControllerCommand(SchemaBase):
    rotor_thrusts_n: dict[str, float]
    vectoring_joint_targets: dict[str, float]
    joint_torque_commands: dict[str, float]
    dock_mechanism_commands: dict[str, float]
    controller_status: ControllerStatus

