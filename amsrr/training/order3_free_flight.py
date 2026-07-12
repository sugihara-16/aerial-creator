from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from amsrr.schemas.common import (
    Pose7D,
    SchemaBase,
    SchemaValidationError,
    StrEnum,
    require_len,
    require_non_empty,
)
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.policies import POLICY_COMMAND_CONTRACT_CENTROIDAL


ORDER3_FREE_FLIGHT_VERSION = "order3_free_flight_v1"
TRUE_CENTROIDAL_TRACKING_SOURCE = "true_morphology_centroidal_frame"
ORDER3_REQUIRED_MODULE_COUNTS: tuple[int, ...] = tuple(range(2, 9))

# There are only eight rooted, port-labelled two-module structures in the
# current Holon proposal distribution (two pitch and two yaw ports per module).
# A morphology-disjoint split must therefore not request 8/2/2 at N=2.
ORDER3_TWO_MODULE_STRUCTURAL_CAPACITY = 8

ORDER3_EXCLUDED_CLAIMS: tuple[str, ...] = (
    "object_task",
    "contact_assignment",
    "dock_attach_detach",
    "dock_joint_motion",
    "p4_full_completion",
)

SAFETY_QP_INFEASIBLE = "qp_infeasible"
SAFETY_HARD_COLLISION = "hard_collision"
SAFETY_NON_FINITE_STATE = "non_finite_state"
SAFETY_UNSUPPORTED_ACTUATOR = "unsupported_actuator"
SAFETY_TIMEOUT = "timeout"


class Order3TaskMode(StrEnum):
    HOVER = "hover"
    WAYPOINT = "waypoint"
    TAKEOFF = "takeoff"


class Order3LearningMode(StrEnum):
    BEHAVIOR_CLONING = "behavior_cloning"
    PPO = "ppo"
    EVALUATION = "evaluation"


@dataclass
class Order3CurriculumStage(SchemaBase):
    stage_id: str
    stage_index: int
    learning_mode: Order3LearningMode
    min_modules: int
    max_modules: int
    task_modes: list[Order3TaskMode]
    deterministic_teacher_required: bool
    initial_state_randomization: bool
    disturbance_randomization: bool
    model_randomization: bool
    floor_enabled: bool
    held_out_only: bool
    minimum_success_rate: float
    maximum_safety_failure_episodes: int = 0
    residual_policy: bool = True
    object_task_claim: bool = False
    contact_task_claim: bool = False
    dock_motion_claim: bool = False
    p4_full_completion_claim: bool = False

    def validate(self) -> None:
        require_non_empty(self.stage_id, "Order3CurriculumStage.stage_id")
        if self.stage_index < 0:
            raise SchemaValidationError(
                "Order3CurriculumStage.stage_index must be non-negative"
            )
        if not 2 <= self.min_modules <= self.max_modules <= 8:
            raise SchemaValidationError(
                "Order3CurriculumStage module range must be within [2, 8]"
            )
        if not self.task_modes or len(set(self.task_modes)) != len(self.task_modes):
            raise SchemaValidationError(
                "Order3CurriculumStage.task_modes must be non-empty and unique"
            )
        if Order3TaskMode.TAKEOFF in self.task_modes and not self.floor_enabled:
            raise SchemaValidationError(
                "Order3 takeoff curriculum stages must enable the floor"
            )
        if not 0.0 <= self.minimum_success_rate <= 1.0:
            raise SchemaValidationError(
                "Order3CurriculumStage.minimum_success_rate must be in [0, 1]"
            )
        if self.maximum_safety_failure_episodes < 0:
            raise SchemaValidationError(
                "Order3CurriculumStage.maximum_safety_failure_episodes must be non-negative"
            )
        if self.learning_mode == Order3LearningMode.BEHAVIOR_CLONING:
            if not self.deterministic_teacher_required:
                raise SchemaValidationError(
                    "Order3 behavior cloning requires the deterministic v2 teacher"
                )
            if self.held_out_only:
                raise SchemaValidationError(
                    "Order3 behavior cloning must not train on held-out morphologies"
                )
        if self.learning_mode == Order3LearningMode.EVALUATION and not self.held_out_only:
            raise SchemaValidationError(
                "Order3 evaluation must be held-out-only"
            )
        if any(
            (
                self.object_task_claim,
                self.contact_task_claim,
                self.dock_motion_claim,
                self.p4_full_completion_claim,
            )
        ):
            raise SchemaValidationError(
                "Order3 free-flight curriculum cannot claim object/contact/dock-motion/P4-full scope"
            )


@dataclass
class Order3CurriculumSchedule(SchemaBase):
    schedule_version: str = ORDER3_FREE_FLIGHT_VERSION
    stages: list[Order3CurriculumStage] = field(default_factory=list)
    control_contract_version: str = POLICY_COMMAND_CONTRACT_CENTROIDAL
    tracking_state_source: str = TRUE_CENTROIDAL_TRACKING_SOURCE
    object_task_claim: bool = False
    contact_task_claim: bool = False
    p4_full_completion_claim: bool = False

    def validate(self) -> None:
        require_non_empty(
            self.schedule_version, "Order3CurriculumSchedule.schedule_version"
        )
        if self.control_contract_version != POLICY_COMMAND_CONTRACT_CENTROIDAL:
            raise SchemaValidationError(
                "Order3 curriculum requires the centroidal_local_joint_v2 contract"
            )
        if self.tracking_state_source != TRUE_CENTROIDAL_TRACKING_SOURCE:
            raise SchemaValidationError(
                "Order3 curriculum must track the true morphology centroidal frame"
            )
        if any(
            (
                self.object_task_claim,
                self.contact_task_claim,
                self.p4_full_completion_claim,
            )
        ):
            raise SchemaValidationError(
                "Order3 schedule cannot claim object/contact/P4-full scope"
            )
        if not self.stages:
            raise SchemaValidationError("Order3CurriculumSchedule.stages must not be empty")
        if len({stage.stage_id for stage in self.stages}) != len(self.stages):
            raise SchemaValidationError("Order3 curriculum stage ids must be unique")
        expected_indices = list(range(len(self.stages)))
        if [stage.stage_index for stage in self.stages] != expected_indices:
            raise SchemaValidationError(
                "Order3 curriculum stage indices must be contiguous and ordered"
            )
        if self.stages[0].learning_mode != Order3LearningMode.BEHAVIOR_CLONING:
            raise SchemaValidationError("Order3 curriculum must begin with behavior cloning")
        if not any(stage.learning_mode == Order3LearningMode.PPO for stage in self.stages):
            raise SchemaValidationError("Order3 curriculum must contain a PPO stage")
        if self.stages[-1].learning_mode != Order3LearningMode.EVALUATION:
            raise SchemaValidationError("Order3 curriculum must end with held-out evaluation")
        # BC may cover the complete offline morphology pool before PPO starts
        # with an easier live-simulation subset.  Require monotonic expansion
        # only across the PPO stages themselves.
        max_modules = [
            stage.max_modules
            for stage in self.stages
            if stage.learning_mode == Order3LearningMode.PPO
        ]
        if any(right < left for left, right in zip(max_modules, max_modules[1:])):
            raise SchemaValidationError(
                "Order3 curriculum maximum module count must not decrease"
            )


def default_order3_curriculum_schedule() -> Order3CurriculumSchedule:
    """Return the approved BC -> residual PPO -> held-out curriculum."""

    return Order3CurriculumSchedule(
        stages=[
            Order3CurriculumStage(
                stage_id="3a_bc_v2_teacher",
                stage_index=0,
                learning_mode=Order3LearningMode.BEHAVIOR_CLONING,
                min_modules=2,
                max_modules=8,
                task_modes=[
                    Order3TaskMode.HOVER,
                    Order3TaskMode.WAYPOINT,
                    Order3TaskMode.TAKEOFF,
                ],
                deterministic_teacher_required=True,
                initial_state_randomization=False,
                disturbance_randomization=False,
                model_randomization=False,
                floor_enabled=True,
                held_out_only=False,
                minimum_success_rate=0.95,
            ),
            Order3CurriculumStage(
                stage_id="3b_ppo_hover_waypoint",
                stage_index=1,
                learning_mode=Order3LearningMode.PPO,
                min_modules=2,
                max_modules=4,
                task_modes=[Order3TaskMode.HOVER, Order3TaskMode.WAYPOINT],
                deterministic_teacher_required=False,
                initial_state_randomization=True,
                disturbance_randomization=False,
                model_randomization=False,
                floor_enabled=False,
                held_out_only=False,
                minimum_success_rate=0.90,
            ),
            Order3CurriculumStage(
                stage_id="3c_ppo_morphology_randomized",
                stage_index=2,
                learning_mode=Order3LearningMode.PPO,
                min_modules=2,
                max_modules=8,
                task_modes=[Order3TaskMode.HOVER, Order3TaskMode.WAYPOINT],
                deterministic_teacher_required=False,
                initial_state_randomization=True,
                disturbance_randomization=True,
                model_randomization=True,
                floor_enabled=False,
                held_out_only=False,
                minimum_success_rate=0.90,
            ),
            Order3CurriculumStage(
                stage_id="3d_ppo_takeoff_hover",
                stage_index=3,
                learning_mode=Order3LearningMode.PPO,
                min_modules=2,
                max_modules=8,
                task_modes=[Order3TaskMode.TAKEOFF, Order3TaskMode.HOVER],
                deterministic_teacher_required=False,
                initial_state_randomization=True,
                disturbance_randomization=True,
                model_randomization=True,
                floor_enabled=True,
                held_out_only=False,
                minimum_success_rate=0.90,
            ),
            Order3CurriculumStage(
                stage_id="3e_held_out_evaluation",
                stage_index=4,
                learning_mode=Order3LearningMode.EVALUATION,
                min_modules=2,
                max_modules=8,
                task_modes=[
                    Order3TaskMode.HOVER,
                    Order3TaskMode.WAYPOINT,
                    Order3TaskMode.TAKEOFF,
                ],
                deterministic_teacher_required=False,
                initial_state_randomization=True,
                disturbance_randomization=True,
                model_randomization=True,
                floor_enabled=True,
                held_out_only=True,
                minimum_success_rate=0.90,
            ),
        ]
    )


@dataclass
class Order3FreeFlightRewardConfig(SchemaBase):
    position_scale_m: float = 0.20
    attitude_scale_rad: float = 0.25
    linear_velocity_scale_mps: float = 0.15
    angular_velocity_scale_rad_s: float = 0.25
    progress_scale: float = 0.25
    disturbance_force_scale_n: float = 20.0
    disturbance_torque_scale_nm: float = 2.0
    wind_speed_scale_mps: float = 5.0
    model_scale_deviation: float = 0.20
    success_position_threshold_m: float = 0.20
    success_attitude_threshold_rad: float = 0.25
    success_linear_velocity_threshold_mps: float = 0.15
    success_angular_velocity_threshold_rad_s: float = 0.25
    success_hold_duration_s: float = 1.0
    takeoff_min_height_gain_ratio: float = 0.80
    w_position: float = 1.0
    w_attitude: float = 1.0
    w_linear_velocity: float = 0.5
    w_angular_velocity: float = 0.5
    w_progress: float = 0.5
    w_energy: float = 0.01
    w_action_delta: float = 0.02
    w_saturation: float = 0.25
    w_fallback: float = 0.25
    w_privileged_disturbance_improvement: float = 0.5
    success_bonus: float = 10.0
    failure_penalty: float = 10.0

    def validate(self) -> None:
        positive = (
            "position_scale_m",
            "attitude_scale_rad",
            "linear_velocity_scale_mps",
            "angular_velocity_scale_rad_s",
            "progress_scale",
            "disturbance_force_scale_n",
            "disturbance_torque_scale_nm",
            "wind_speed_scale_mps",
            "model_scale_deviation",
            "success_position_threshold_m",
            "success_attitude_threshold_rad",
            "success_linear_velocity_threshold_mps",
            "success_angular_velocity_threshold_rad_s",
            "success_hold_duration_s",
        )
        non_negative = (
            "w_position",
            "w_attitude",
            "w_linear_velocity",
            "w_angular_velocity",
            "w_progress",
            "w_energy",
            "w_action_delta",
            "w_saturation",
            "w_fallback",
            "w_privileged_disturbance_improvement",
            "success_bonus",
            "failure_penalty",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise SchemaValidationError(
                    f"Order3FreeFlightRewardConfig.{name} must be finite and positive"
                )
        for name in non_negative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise SchemaValidationError(
                    f"Order3FreeFlightRewardConfig.{name} must be finite and non-negative"
                )
        if not 0.0 < self.takeoff_min_height_gain_ratio <= 1.0:
            raise SchemaValidationError(
                "Order3FreeFlightRewardConfig.takeoff_min_height_gain_ratio must be in (0, 1]"
            )


@dataclass
class Order3PrivilegedRewardSignals(SchemaBase):
    """Simulator-only signals for the reward/critic, never the actor input."""

    applied_external_wrench_body: list[float] = field(default_factory=lambda: [0.0] * 6)
    wind_velocity_world: list[float] = field(default_factory=lambda: [0.0] * 3)
    mass_scale: float = 1.0
    thrust_scale: float = 1.0
    deterministic_baseline_tracking_cost: float | None = None
    actor_observation_allowed: bool = False

    def validate(self) -> None:
        require_len(
            self.applied_external_wrench_body,
            6,
            "Order3PrivilegedRewardSignals.applied_external_wrench_body",
        )
        require_len(
            self.wind_velocity_world,
            3,
            "Order3PrivilegedRewardSignals.wind_velocity_world",
        )
        _require_finite_vector(
            self.applied_external_wrench_body,
            "Order3PrivilegedRewardSignals.applied_external_wrench_body",
        )
        _require_finite_vector(
            self.wind_velocity_world,
            "Order3PrivilegedRewardSignals.wind_velocity_world",
        )
        for name in ("mass_scale", "thrust_scale"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise SchemaValidationError(
                    f"Order3PrivilegedRewardSignals.{name} must be finite and positive"
                )
        if self.deterministic_baseline_tracking_cost is not None:
            value = float(self.deterministic_baseline_tracking_cost)
            if not math.isfinite(value) or value < 0.0:
                raise SchemaValidationError(
                    "Order3 privileged baseline tracking cost must be finite and non-negative"
                )
        if self.actor_observation_allowed:
            raise SchemaValidationError(
                "Order3 privileged reward signals must not be exposed to the actor"
            )


@dataclass
class Order3FreeFlightStep(SchemaBase):
    module_count: int
    task_mode: Order3TaskMode
    centroidal_pose_world: Pose7D
    centroidal_twist_world: list[float]
    target_pose_world: Pose7D
    target_twist_world: list[float]
    tracking_state_source: str = TRUE_CENTROIDAL_TRACKING_SOURCE
    control_contract_version: str = POLICY_COMMAND_CONTRACT_CENTROIDAL
    previous_tracking_cost: float | None = None
    within_tolerance_duration_s: float = 0.0
    takeoff_height_gain_ratio: float | None = None
    normalized_energy: float = 0.0
    normalized_action_delta: float = 0.0
    qp_feasible: bool = True
    hard_collision: bool = False
    non_finite_state: bool = False
    unsupported_actuator: bool = False
    actuator_saturated: bool = False
    fallback_active: bool = False
    timed_out: bool = False
    terminal: bool = False
    privileged: Order3PrivilegedRewardSignals | None = None
    object_task_active: bool = False
    contact_assignment_count: int = 0
    dock_motion_commanded: bool = False

    def validate(self) -> None:
        if not 2 <= self.module_count <= 8:
            raise SchemaValidationError(
                "Order3FreeFlightStep.module_count must be in [2, 8]"
            )
        if self.tracking_state_source != TRUE_CENTROIDAL_TRACKING_SOURCE:
            raise SchemaValidationError(
                "Order3 reward requires true morphology centroidal tracking"
            )
        if self.control_contract_version != POLICY_COMMAND_CONTRACT_CENTROIDAL:
            raise SchemaValidationError(
                "Order3 reward requires centroidal_local_joint_v2"
            )
        _require_pose(self.centroidal_pose_world, "Order3FreeFlightStep.centroidal_pose_world")
        _require_pose(self.target_pose_world, "Order3FreeFlightStep.target_pose_world")
        require_len(
            self.centroidal_twist_world,
            6,
            "Order3FreeFlightStep.centroidal_twist_world",
        )
        require_len(
            self.target_twist_world,
            6,
            "Order3FreeFlightStep.target_twist_world",
        )
        _require_finite_vector(
            self.centroidal_twist_world,
            "Order3FreeFlightStep.centroidal_twist_world",
        )
        _require_finite_vector(
            self.target_twist_world,
            "Order3FreeFlightStep.target_twist_world",
        )
        for name in (
            "within_tolerance_duration_s",
            "normalized_energy",
            "normalized_action_delta",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise SchemaValidationError(
                    f"Order3FreeFlightStep.{name} must be finite and non-negative"
                )
        for name in ("previous_tracking_cost", "takeoff_height_gain_ratio"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0.0):
                raise SchemaValidationError(
                    f"Order3FreeFlightStep.{name} must be finite and non-negative when present"
                )
        if self.task_mode == Order3TaskMode.TAKEOFF and self.takeoff_height_gain_ratio is None:
            raise SchemaValidationError(
                "Order3 takeoff reward requires takeoff_height_gain_ratio"
            )
        if self.contact_assignment_count < 0:
            raise SchemaValidationError(
                "Order3FreeFlightStep.contact_assignment_count must be non-negative"
            )
        if self.object_task_active or self.contact_assignment_count or self.dock_motion_commanded:
            raise SchemaValidationError(
                "Order3 free-flight steps exclude object tasks, contact assignments, and dock motion"
            )


@dataclass
class Order3FreeFlightRewardResult(SchemaBase):
    reward: float
    terms: dict[str, float]
    tracking_cost: float
    success: bool
    failure: bool
    terminal: bool
    failure_reasons: list[str]
    privileged_reward_used: bool
    privileged_actor_observation_allowed: bool = False
    tracking_state_source: str = TRUE_CENTROIDAL_TRACKING_SOURCE
    control_contract_version: str = POLICY_COMMAND_CONTRACT_CENTROIDAL
    object_task_claim: bool = False
    contact_task_claim: bool = False
    p4_full_completion_claim: bool = False

    def validate(self) -> None:
        if not math.isfinite(self.reward) or not math.isfinite(self.tracking_cost):
            raise SchemaValidationError("Order3 reward result must be finite")
        if any(not math.isfinite(value) for value in self.terms.values()):
            raise SchemaValidationError("Order3 reward terms must be finite")
        if self.success and self.failure:
            raise SchemaValidationError("Order3 reward cannot be both success and failure")
        if (self.success or self.failure) and not self.terminal:
            raise SchemaValidationError("Order3 terminal outcome must terminate the episode")
        if self.failure != bool(self.failure_reasons):
            raise SchemaValidationError(
                "Order3 failure flag must match failure_reasons"
            )
        if self.privileged_actor_observation_allowed:
            raise SchemaValidationError(
                "Order3 privileged reward must not enter actor observations"
            )
        if self.tracking_state_source != TRUE_CENTROIDAL_TRACKING_SOURCE:
            raise SchemaValidationError("Order3 reward result source is not true centroidal")
        if self.control_contract_version != POLICY_COMMAND_CONTRACT_CENTROIDAL:
            raise SchemaValidationError("Order3 reward result contract is not v2")
        if any(
            (
                self.object_task_claim,
                self.contact_task_claim,
                self.p4_full_completion_claim,
            )
        ):
            raise SchemaValidationError(
                "Order3 reward cannot claim object/contact/P4-full completion"
            )


def compute_order3_free_flight_reward(
    step: Order3FreeFlightStep,
    *,
    config: Order3FreeFlightRewardConfig | None = None,
) -> Order3FreeFlightRewardResult:
    """Compute bounded free-flight reward from a true centroidal state.

    ``step.privileged`` may improve training credit assignment and compare the
    learned policy with a deterministic baseline under the same disturbance.
    Its contract explicitly forbids inclusion in actor observations.
    """

    cfg = config or Order3FreeFlightRewardConfig()
    position_error = _distance3(
        step.centroidal_pose_world[:3], step.target_pose_world[:3]
    )
    attitude_error = _quaternion_angle(
        step.centroidal_pose_world[3:7], step.target_pose_world[3:7]
    )
    linear_velocity_error = _distance3(
        step.centroidal_twist_world[:3], step.target_twist_world[:3]
    )
    angular_velocity_error = _distance3(
        step.centroidal_twist_world[3:6], step.target_twist_world[3:6]
    )
    normalized_errors = (
        position_error / cfg.position_scale_m,
        attitude_error / cfg.attitude_scale_rad,
        linear_velocity_error / cfg.linear_velocity_scale_mps,
        angular_velocity_error / cfg.angular_velocity_scale_rad_s,
    )
    tracking_cost = sum(normalized_errors) / len(normalized_errors)
    r_position = 1.0 / (1.0 + normalized_errors[0])
    r_attitude = 1.0 / (1.0 + normalized_errors[1])
    r_linear_velocity = 1.0 / (1.0 + normalized_errors[2])
    r_angular_velocity = 1.0 / (1.0 + normalized_errors[3])
    r_progress = (
        0.0
        if step.previous_tracking_cost is None
        else _clamp(
            (float(step.previous_tracking_cost) - tracking_cost) / cfg.progress_scale,
            -1.0,
            1.0,
        )
    )
    p_energy = _clamp01(step.normalized_energy)
    p_action_delta = _clamp01(step.normalized_action_delta)
    p_saturation = 1.0 if step.actuator_saturated else 0.0
    p_fallback = 1.0 if step.fallback_active else 0.0

    privileged_improvement = 0.0
    disturbance_severity = 0.0
    privileged_reward_used = False
    if step.privileged is not None:
        disturbance_severity = _disturbance_severity(step.privileged, cfg)
        baseline_cost = step.privileged.deterministic_baseline_tracking_cost
        if baseline_cost is not None and disturbance_severity > 0.0:
            privileged_improvement = disturbance_severity * _clamp(
                (float(baseline_cost) - tracking_cost)
                / max(float(baseline_cost), 1.0e-6),
                -1.0,
                1.0,
            )
            privileged_reward_used = True

    failure_reasons = _safety_failure_reasons(step)
    failure = bool(failure_reasons)
    within_tracking_tolerance = (
        position_error <= cfg.success_position_threshold_m
        and attitude_error <= cfg.success_attitude_threshold_rad
        and linear_velocity_error <= cfg.success_linear_velocity_threshold_mps
        and angular_velocity_error <= cfg.success_angular_velocity_threshold_rad_s
    )
    takeoff_gate = (
        step.task_mode != Order3TaskMode.TAKEOFF
        or float(step.takeoff_height_gain_ratio or 0.0)
        >= cfg.takeoff_min_height_gain_ratio
    )
    success = (
        not failure
        and within_tracking_tolerance
        and takeoff_gate
        and step.within_tolerance_duration_s + 1.0e-12
        >= cfg.success_hold_duration_s
    )
    terminal = bool(step.terminal or success or failure)

    reward = (
        cfg.w_position * r_position
        + cfg.w_attitude * r_attitude
        + cfg.w_linear_velocity * r_linear_velocity
        + cfg.w_angular_velocity * r_angular_velocity
        + cfg.w_progress * r_progress
        - cfg.w_energy * p_energy
        - cfg.w_action_delta * p_action_delta
        - cfg.w_saturation * p_saturation
        - cfg.w_fallback * p_fallback
        + cfg.w_privileged_disturbance_improvement * privileged_improvement
    )
    if success:
        reward += cfg.success_bonus
    elif failure:
        reward -= cfg.failure_penalty

    return Order3FreeFlightRewardResult(
        reward=reward,
        terms={
            "position_error_m": position_error,
            "attitude_error_rad": attitude_error,
            "linear_velocity_error_mps": linear_velocity_error,
            "angular_velocity_error_rad_s": angular_velocity_error,
            "tracking_cost": tracking_cost,
            "r_position": r_position,
            "r_attitude": r_attitude,
            "r_linear_velocity": r_linear_velocity,
            "r_angular_velocity": r_angular_velocity,
            "r_progress": r_progress,
            "p_energy": p_energy,
            "p_action_delta": p_action_delta,
            "p_saturation": p_saturation,
            "p_fallback": p_fallback,
            "privileged_disturbance_severity": disturbance_severity,
            "privileged_baseline_improvement": privileged_improvement,
            "terminal_success": 1.0 if success else 0.0,
            "terminal_failure": 1.0 if failure else 0.0,
        },
        tracking_cost=tracking_cost,
        success=success,
        failure=failure,
        terminal=terminal,
        failure_reasons=failure_reasons,
        privileged_reward_used=privileged_reward_used,
    )


@dataclass
class Order3TerminalMetrics(SchemaBase):
    position_error_m: float
    attitude_error_rad: float
    linear_velocity_error_mps: float
    angular_velocity_error_rad_s: float
    within_tolerance_duration_s: float
    takeoff_height_gain_ratio: float | None = None

    def validate(self) -> None:
        for name in (
            "position_error_m",
            "attitude_error_rad",
            "linear_velocity_error_mps",
            "angular_velocity_error_rad_s",
            "within_tolerance_duration_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise SchemaValidationError(
                    f"Order3TerminalMetrics.{name} must be finite and non-negative"
                )
        if self.takeoff_height_gain_ratio is not None:
            value = float(self.takeoff_height_gain_ratio)
            if not math.isfinite(value) or value < 0.0:
                raise SchemaValidationError(
                    "Order3TerminalMetrics.takeoff_height_gain_ratio must be finite and non-negative"
                )


def order3_terminal_metrics_success(
    metrics: Order3TerminalMetrics,
    *,
    task_mode: Order3TaskMode,
    config: Order3FreeFlightRewardConfig | None = None,
) -> bool:
    cfg = config or Order3FreeFlightRewardConfig()
    if task_mode == Order3TaskMode.TAKEOFF and metrics.takeoff_height_gain_ratio is None:
        raise SchemaValidationError(
            "Order3 takeoff terminal metrics require takeoff_height_gain_ratio"
        )
    return bool(
        metrics.position_error_m <= cfg.success_position_threshold_m
        and metrics.attitude_error_rad <= cfg.success_attitude_threshold_rad
        and metrics.linear_velocity_error_mps
        <= cfg.success_linear_velocity_threshold_mps
        and metrics.angular_velocity_error_rad_s
        <= cfg.success_angular_velocity_threshold_rad_s
        and metrics.within_tolerance_duration_s + 1.0e-12
        >= cfg.success_hold_duration_s
        and (
            task_mode != Order3TaskMode.TAKEOFF
            or float(metrics.takeoff_height_gain_ratio or 0.0)
            >= cfg.takeoff_min_height_gain_ratio
        )
    )


@dataclass
class Order3EvaluationEpisode(SchemaBase):
    episode_id: str
    structural_hash: str
    module_count: int
    split: DatasetSplit
    success: bool
    tracking_cost: float
    deterministic_baseline_tracking_cost: float | None = None
    randomized: bool = False
    fallback_used: bool = False
    qp_infeasible: bool = False
    hard_collision: bool = False
    non_finite_state: bool = False
    unsupported_actuator: bool = False
    task_mode: Order3TaskMode = Order3TaskMode.HOVER
    terminal_metrics: Order3TerminalMetrics | None = None
    deterministic_baseline_terminal_metrics: Order3TerminalMetrics | None = None
    condition_hash: str | None = None
    condition_seed: int | None = None
    checkpoint_sha256: str | None = None
    learned_report_path: str | None = None
    learned_report_sha256: str | None = None
    deterministic_baseline_report_path: str | None = None
    deterministic_baseline_report_sha256: str | None = None
    fallback_reason: str | None = None
    isaac_backed: bool = False
    object_task_claim: bool = False
    contact_task_claim: bool = False
    p4_full_completion_claim: bool = False

    def validate(self) -> None:
        require_non_empty(self.episode_id, "Order3EvaluationEpisode.episode_id")
        require_non_empty(
            self.structural_hash, "Order3EvaluationEpisode.structural_hash"
        )
        if not 2 <= self.module_count <= 8:
            raise SchemaValidationError(
                "Order3EvaluationEpisode.module_count must be in [2, 8]"
            )
        if not math.isfinite(self.tracking_cost) or self.tracking_cost < 0.0:
            raise SchemaValidationError(
                "Order3EvaluationEpisode.tracking_cost must be finite and non-negative"
            )
        if self.deterministic_baseline_tracking_cost is not None:
            value = float(self.deterministic_baseline_tracking_cost)
            if not math.isfinite(value) or value < 0.0:
                raise SchemaValidationError(
                    "Order3 evaluation baseline cost must be finite and non-negative"
                )
        for name in (
            "condition_hash",
            "checkpoint_sha256",
            "learned_report_sha256",
            "deterministic_baseline_report_sha256",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise SchemaValidationError(
                    f"Order3EvaluationEpisode.{name} must be a lowercase SHA-256 digest"
                )
        if self.condition_seed is not None and self.condition_seed < 0:
            raise SchemaValidationError(
                "Order3EvaluationEpisode.condition_seed must be non-negative"
            )
        for name in (
            "learned_report_path",
            "deterministic_baseline_report_path",
        ):
            value = getattr(self, name)
            if value is not None and not value:
                raise SchemaValidationError(
                    f"Order3EvaluationEpisode.{name} must be non-empty when present"
                )
        if self.fallback_reason is not None:
            require_non_empty(
                self.fallback_reason,
                "Order3EvaluationEpisode.fallback_reason",
            )
        if self.terminal_metrics is not None:
            recomputed_success = order3_terminal_metrics_success(
                self.terminal_metrics,
                task_mode=self.task_mode,
            )
            if self.success != recomputed_success:
                raise SchemaValidationError(
                    "Order3EvaluationEpisode.success must match terminal metrics"
                )
        if any(
            (
                self.object_task_claim,
                self.contact_task_claim,
                self.p4_full_completion_claim,
            )
        ):
            raise SchemaValidationError(
                "Order3 evaluation cannot claim object/contact/P4-full completion"
            )

    @property
    def safety_failure(self) -> bool:
        return any(
            (
                self.qp_infeasible,
                self.hard_collision,
                self.non_finite_state,
                self.unsupported_actuator,
            )
        )


@dataclass
class Order3ModuleCoverage(SchemaBase):
    module_count: int
    episode_count: int
    unique_structural_hash_count: int
    success_rate: float
    mean_tracking_cost: float
    mean_baseline_tracking_cost: float | None
    randomized_mean_relative_improvement: float | None
    safety_failure_episode_count: int
    fallback_rate: float

    def validate(self) -> None:
        if not 2 <= self.module_count <= 8:
            raise SchemaValidationError("Order3ModuleCoverage.module_count must be in [2, 8]")
        if self.episode_count < 0 or self.unique_structural_hash_count < 0:
            raise SchemaValidationError("Order3 module coverage counts must be non-negative")
        if self.safety_failure_episode_count < 0:
            raise SchemaValidationError("Order3 safety failure count must be non-negative")
        for name in ("success_rate", "fallback_rate"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise SchemaValidationError(
                    f"Order3ModuleCoverage.{name} must be finite and in [0, 1]"
                )
        if not math.isfinite(self.mean_tracking_cost) or self.mean_tracking_cost < 0.0:
            raise SchemaValidationError(
                "Order3ModuleCoverage.mean_tracking_cost must be finite and non-negative"
            )
        for name in (
            "mean_baseline_tracking_cost",
            "randomized_mean_relative_improvement",
        ):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise SchemaValidationError(
                    f"Order3ModuleCoverage.{name} must be finite when present"
                )


@dataclass
class Order3CoverageSummary(SchemaBase):
    required_module_counts: list[int]
    covered_module_counts: list[int]
    missing_module_counts: list[int]
    per_module_count: dict[str, Order3ModuleCoverage]
    episode_count: int
    aggregate_success_rate: float
    safety_failure_episode_count: int
    fallback_rate: float
    randomized_episode_count: int
    randomized_mean_relative_improvement: float | None
    split_episode_counts: dict[str, int]
    minimum_episodes_per_module_count: int
    coverage_complete: bool
    safety_passed: bool
    object_task_claim: bool = False
    contact_task_claim: bool = False
    p4_full_completion_claim: bool = False

    def validate(self) -> None:
        if sorted(set(self.required_module_counts)) != sorted(self.required_module_counts):
            raise SchemaValidationError(
                "Order3CoverageSummary.required_module_counts must be unique and sorted"
            )
        if any(count not in ORDER3_REQUIRED_MODULE_COUNTS for count in self.required_module_counts):
            raise SchemaValidationError(
                "Order3 coverage required module counts must be within [2, 8]"
            )
        if self.minimum_episodes_per_module_count < 1:
            raise SchemaValidationError(
                "Order3 minimum episodes per module count must be positive"
            )
        for name in (
            "episode_count",
            "safety_failure_episode_count",
            "randomized_episode_count",
        ):
            if int(getattr(self, name)) < 0:
                raise SchemaValidationError(f"Order3CoverageSummary.{name} must be non-negative")
        for name in ("aggregate_success_rate", "fallback_rate"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise SchemaValidationError(
                    f"Order3CoverageSummary.{name} must be finite and in [0, 1]"
                )
        if (
            self.randomized_mean_relative_improvement is not None
            and not math.isfinite(self.randomized_mean_relative_improvement)
        ):
            raise SchemaValidationError(
                "Order3 randomized improvement must be finite when present"
            )
        if self.coverage_complete != (not self.missing_module_counts):
            raise SchemaValidationError(
                "Order3 coverage_complete must match missing_module_counts"
            )
        if self.safety_passed != (self.safety_failure_episode_count == 0):
            raise SchemaValidationError(
                "Order3 safety_passed must match safety failure count"
            )
        if any(
            (
                self.object_task_claim,
                self.contact_task_claim,
                self.p4_full_completion_claim,
            )
        ):
            raise SchemaValidationError(
                "Order3 coverage cannot claim object/contact/P4-full completion"
            )


def summarize_order3_module_coverage(
    episodes: Iterable[Order3EvaluationEpisode],
    *,
    required_module_counts: Sequence[int] = ORDER3_REQUIRED_MODULE_COUNTS,
    minimum_episodes_per_module_count: int = 1,
) -> Order3CoverageSummary:
    values = list(episodes)
    required = sorted(set(int(count) for count in required_module_counts))
    if len(required) != len(required_module_counts):
        raise ValueError("required_module_counts must not contain duplicates")
    if any(count not in ORDER3_REQUIRED_MODULE_COUNTS for count in required):
        raise ValueError("required_module_counts must be within [2, 8]")
    if minimum_episodes_per_module_count < 1:
        raise ValueError("minimum_episodes_per_module_count must be positive")

    grouped = {
        module_count: [item for item in values if item.module_count == module_count]
        for module_count in required
    }
    per_module = {
        str(module_count): _module_coverage(module_count, grouped[module_count])
        for module_count in required
    }
    covered = [
        count
        for count in required
        if len(grouped[count]) >= minimum_episodes_per_module_count
    ]
    missing = [count for count in required if count not in covered]
    randomized_comparisons = _relative_improvements(
        item
        for item in values
        if item.randomized and item.deterministic_baseline_tracking_cost is not None
    )
    safety_failures = sum(item.safety_failure for item in values)
    split_counts = {
        split.value: sum(item.split == split for item in values)
        for split in DatasetSplit
    }
    return Order3CoverageSummary(
        required_module_counts=required,
        covered_module_counts=covered,
        missing_module_counts=missing,
        per_module_count=per_module,
        episode_count=len(values),
        aggregate_success_rate=_ratio(sum(item.success for item in values), len(values)),
        safety_failure_episode_count=safety_failures,
        fallback_rate=_ratio(sum(item.fallback_used for item in values), len(values)),
        randomized_episode_count=sum(item.randomized for item in values),
        randomized_mean_relative_improvement=(
            _mean(randomized_comparisons) if randomized_comparisons else None
        ),
        split_episode_counts=split_counts,
        minimum_episodes_per_module_count=minimum_episodes_per_module_count,
        coverage_complete=not missing,
        safety_passed=safety_failures == 0,
    )


def recommended_order3_morphology_split_counts(
    module_count: int,
) -> dict[DatasetSplit, int]:
    """Return morphology-disjoint split counts respecting N=2 capacity."""

    if module_count not in ORDER3_REQUIRED_MODULE_COUNTS:
        raise ValueError("Order3 module_count must be in [2, 8]")
    if module_count == 2:
        return {
            DatasetSplit.TRAIN: 4,
            DatasetSplit.VALIDATION: 2,
            DatasetSplit.HELD_OUT: 2,
        }
    return {
        DatasetSplit.TRAIN: 8,
        DatasetSplit.VALIDATION: 2,
        DatasetSplit.HELD_OUT: 2,
    }


def order3_scope_metadata() -> dict[str, object]:
    return {
        "version": ORDER3_FREE_FLIGHT_VERSION,
        "control_contract_version": POLICY_COMMAND_CONTRACT_CENTROIDAL,
        "tracking_state_source": TRUE_CENTROIDAL_TRACKING_SOURCE,
        "module_count_min": 2,
        "module_count_max": 8,
        "excluded_claims": list(ORDER3_EXCLUDED_CLAIMS),
        "object_task_claim": False,
        "contact_task_claim": False,
        "p4_full_completion_claim": False,
    }


def _module_coverage(
    module_count: int,
    episodes: Sequence[Order3EvaluationEpisode],
) -> Order3ModuleCoverage:
    baseline_values = [
        float(item.deterministic_baseline_tracking_cost)
        for item in episodes
        if item.deterministic_baseline_tracking_cost is not None
    ]
    randomized_improvements = _relative_improvements(
        item
        for item in episodes
        if item.randomized and item.deterministic_baseline_tracking_cost is not None
    )
    return Order3ModuleCoverage(
        module_count=module_count,
        episode_count=len(episodes),
        unique_structural_hash_count=len({item.structural_hash for item in episodes}),
        success_rate=_ratio(sum(item.success for item in episodes), len(episodes)),
        mean_tracking_cost=_mean([item.tracking_cost for item in episodes]),
        mean_baseline_tracking_cost=(
            _mean(baseline_values) if baseline_values else None
        ),
        randomized_mean_relative_improvement=(
            _mean(randomized_improvements) if randomized_improvements else None
        ),
        safety_failure_episode_count=sum(item.safety_failure for item in episodes),
        fallback_rate=_ratio(sum(item.fallback_used for item in episodes), len(episodes)),
    )


def _relative_improvements(
    episodes: Iterable[Order3EvaluationEpisode],
) -> list[float]:
    output: list[float] = []
    for item in episodes:
        baseline = float(item.deterministic_baseline_tracking_cost or 0.0)
        output.append(
            _clamp((baseline - item.tracking_cost) / max(baseline, 1.0e-6), -1.0, 1.0)
        )
    return output


def _safety_failure_reasons(step: Order3FreeFlightStep) -> list[str]:
    reasons: list[str] = []
    if not step.qp_feasible:
        reasons.append(SAFETY_QP_INFEASIBLE)
    if step.hard_collision:
        reasons.append(SAFETY_HARD_COLLISION)
    if step.non_finite_state:
        reasons.append(SAFETY_NON_FINITE_STATE)
    if step.unsupported_actuator:
        reasons.append(SAFETY_UNSUPPORTED_ACTUATOR)
    if step.timed_out:
        reasons.append(SAFETY_TIMEOUT)
    return reasons


def _disturbance_severity(
    signals: Order3PrivilegedRewardSignals,
    config: Order3FreeFlightRewardConfig,
) -> float:
    force = _norm(signals.applied_external_wrench_body[:3]) / config.disturbance_force_scale_n
    torque = _norm(signals.applied_external_wrench_body[3:6]) / config.disturbance_torque_scale_nm
    wind = _norm(signals.wind_velocity_world) / config.wind_speed_scale_mps
    mass = abs(signals.mass_scale - 1.0) / config.model_scale_deviation
    thrust = abs(signals.thrust_scale - 1.0) / config.model_scale_deviation
    return _clamp01(max(force, torque, wind, mass, thrust))


def _require_pose(pose: Sequence[float], path: str) -> None:
    require_len(pose, 7, path)
    _require_finite_vector(pose, path)
    if _norm(pose[3:7]) <= 1.0e-12:
        raise SchemaValidationError(f"{path} quaternion must have non-zero norm")


def _require_finite_vector(values: Sequence[float], path: str) -> None:
    if any(not math.isfinite(float(value)) for value in values):
        raise SchemaValidationError(f"{path} must contain finite values")


def _quaternion_angle(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = _norm(left)
    right_norm = _norm(right)
    dot = sum(
        float(lhs) * float(rhs) / (left_norm * right_norm)
        for lhs, rhs in zip(left, right, strict=True)
    )
    return 2.0 * math.acos(_clamp(abs(dot), 0.0, 1.0))


def _distance3(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(
        sum(
            (float(lhs) - float(rhs)) ** 2
            for lhs, rhs in zip(left, right, strict=True)
        )
    )


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(float(value) for value in values) / len(values) if values else 0.0


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)
