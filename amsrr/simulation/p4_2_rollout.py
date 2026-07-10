from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from amsrr.schemas.common import (
    Pose7D,
    SchemaBase,
    SchemaValidationError,
    StrEnum,
    require_len,
    require_non_empty,
)
from amsrr.schemas.policies import ControllerCommand, ControllerStatus, PolicyCommand
from amsrr.schemas.runtime import RuntimeObservation
from amsrr.utils.config import load_config


P4_2_ROLLOUT_VERSION = "p4_2_deterministic_rollout_v1"
P4_2_CONTACT_MODEL = "kinematic_payload_coupled_attach_v1"
P4_2_REQUIRED_REAL_ROLLOUTS = ("p2_p3_deterministic_grasp_carry",)
P4_2_SUCCESS_SCOPE_NOTE = (
    "P4.2 success_rate is deterministic payload-carry rollout success under "
    "contact_model=kinematic_payload_coupled_attach_v1. It is not high-fidelity natural "
    "grasp success, true fixed-joint dynamics success, learned policy success, "
    "P4.3 learning bootstrap, or P4 full completion."
)


class P4_2RolloutPhase(StrEnum):
    RESET = "reset"
    APPROACH = "approach"
    PREGRASP_ALIGN = "pregrasp_align"
    ATTACH_ATTEMPT = "attach_attempt"
    ATTACHED_MAINTAIN = "attached_maintain"
    TRANSPORT = "transport"
    RELEASE = "release"
    SUCCESS = "success"
    DROP_FAILURE = "drop_failure"
    COLLISION_FAILURE = "collision_failure"
    CONTROLLER_FAILURE = "controller_failure"
    TIMEOUT_FAILURE = "timeout_failure"


P4_2_TERMINAL_PHASES = {
    P4_2RolloutPhase.SUCCESS,
    P4_2RolloutPhase.DROP_FAILURE,
    P4_2RolloutPhase.COLLISION_FAILURE,
    P4_2RolloutPhase.CONTROLLER_FAILURE,
    P4_2RolloutPhase.TIMEOUT_FAILURE,
}


@dataclass
class P4_2PhaseDefinition(SchemaBase):
    phase: P4_2RolloutPhase
    terminal: bool
    entry_conditions: list[str]
    exit_conditions: list[str]
    timeout_s: float | None = None
    timeout_transition: P4_2RolloutPhase | None = None

    def validate(self) -> None:
        if not self.entry_conditions:
            raise SchemaValidationError("P4_2PhaseDefinition.entry_conditions must be non-empty")
        if not self.terminal and not self.exit_conditions:
            raise SchemaValidationError("P4_2PhaseDefinition.exit_conditions must be non-empty for non-terminal phases")
        if self.timeout_s is not None and self.timeout_s <= 0.0:
            raise SchemaValidationError("P4_2PhaseDefinition.timeout_s must be positive when set")
        if self.terminal and self.timeout_transition is not None:
            raise SchemaValidationError("P4_2PhaseDefinition terminal phases cannot have timeout_transition")


@dataclass
class P4_2PhaseTransitionRecord(SchemaBase):
    from_phase: P4_2RolloutPhase
    to_phase: P4_2RolloutPhase
    time_s: float
    phase_elapsed_s: float
    reason: str
    entry_condition_results: dict[str, bool] = field(default_factory=dict)
    exit_condition_results: dict[str, bool] = field(default_factory=dict)
    timeout_s: float | None = None

    def validate(self) -> None:
        if self.time_s < 0.0:
            raise SchemaValidationError("P4_2PhaseTransitionRecord.time_s must be non-negative")
        if self.phase_elapsed_s < 0.0:
            raise SchemaValidationError("P4_2PhaseTransitionRecord.phase_elapsed_s must be non-negative")
        require_non_empty(self.reason, "P4_2PhaseTransitionRecord.reason")
        if self.timeout_s is not None and self.timeout_s <= 0.0:
            raise SchemaValidationError("P4_2PhaseTransitionRecord.timeout_s must be positive when set")


@dataclass
class P4_2DeterministicRolloutConfig(SchemaBase):
    config_path: str = "configs/env/isaac_lab.yaml"
    robot_model_config_path: str = "configs/robot/robot_model.yaml"
    p3_config_path: str = "configs/training/p3_assembly_grasp_carry.yaml"
    control_dt_s: float = 0.005
    max_episode_steps: int = 1200
    rollout_name: str = "p2_p3_deterministic_grasp_carry"
    object_id: str = "box_01"
    object_size_m: tuple[float, float, float] = (0.30, 0.20, 0.15)
    object_mass_kg: float = 1.0
    object_initial_pose_world: Pose7D = (0.8, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0)
    contact_model: str = P4_2_CONTACT_MODEL
    attach_distance_threshold_m: float = 0.06
    attach_relative_velocity_threshold_mps: float = 0.20
    attach_snap_distance_threshold_m: float = 0.03
    pregrasp_alignment_distance_m: float = 0.12
    transport_min_displacement_m: float = 0.25
    object_drop_distance_threshold_m: float = 0.35
    controller_failure_consecutive_steps: int = 20
    phase_timeouts_s: dict[str, float] = field(
        default_factory=lambda: {
            "reset": 0.5,
            "approach": 4.0,
            "pregrasp_align": 3.0,
            "attach_attempt": 2.0,
            "attached_maintain": 1.0,
            "transport": 6.0,
            "release": 2.0,
        }
    )

    def validate(self) -> None:
        for name in (
            "config_path",
            "robot_model_config_path",
            "p3_config_path",
            "rollout_name",
            "object_id",
            "contact_model",
        ):
            require_non_empty(getattr(self, name), f"P4_2DeterministicRolloutConfig.{name}")
        if self.contact_model != P4_2_CONTACT_MODEL:
            raise SchemaValidationError(
                f"P4_2DeterministicRolloutConfig.contact_model must be {P4_2_CONTACT_MODEL!r}"
            )
        if self.control_dt_s <= 0.0:
            raise SchemaValidationError("P4_2DeterministicRolloutConfig.control_dt_s must be positive")
        if self.max_episode_steps <= 0:
            raise SchemaValidationError("P4_2DeterministicRolloutConfig.max_episode_steps must be positive")
        if self.object_mass_kg <= 0.0:
            raise SchemaValidationError("P4_2DeterministicRolloutConfig.object_mass_kg must be positive")
        require_len(self.object_size_m, 3, "P4_2DeterministicRolloutConfig.object_size_m")
        require_len(self.object_initial_pose_world, 7, "P4_2DeterministicRolloutConfig.object_initial_pose_world")
        for name in (
            "attach_distance_threshold_m",
            "attach_relative_velocity_threshold_mps",
            "attach_snap_distance_threshold_m",
            "pregrasp_alignment_distance_m",
            "transport_min_displacement_m",
            "object_drop_distance_threshold_m",
        ):
            if getattr(self, name) <= 0.0:
                raise SchemaValidationError(f"P4_2DeterministicRolloutConfig.{name} must be positive")
        if self.controller_failure_consecutive_steps <= 0:
            raise SchemaValidationError(
                "P4_2DeterministicRolloutConfig.controller_failure_consecutive_steps must be positive"
            )
        for phase in (
            P4_2RolloutPhase.RESET,
            P4_2RolloutPhase.APPROACH,
            P4_2RolloutPhase.PREGRASP_ALIGN,
            P4_2RolloutPhase.ATTACH_ATTEMPT,
            P4_2RolloutPhase.ATTACHED_MAINTAIN,
            P4_2RolloutPhase.TRANSPORT,
            P4_2RolloutPhase.RELEASE,
        ):
            timeout = self.phase_timeouts_s.get(phase.value)
            if timeout is None or timeout <= 0.0:
                raise SchemaValidationError(f"P4_2 phase timeout is missing or invalid for {phase.value!r}")


@dataclass
class P4_2AttachConditionReport(SchemaBase):
    candidate_id: int
    anchor_id: int
    slot_id: int
    object_id: str
    distance_m: float
    relative_velocity_mps: float
    attach_snap_distance_m: float
    relative_pose_error_m: float
    assignment_feasible: bool
    controller_ok: bool
    within_distance: bool
    within_relative_velocity: bool
    within_attach_snap_distance: bool
    within_attach_phase_timeout: bool
    passed: bool
    attach_phase_elapsed_s: float = 0.0
    attach_phase_timeout_s: float | None = None
    failure_reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        require_non_empty(self.object_id, "P4_2AttachConditionReport.object_id")
        if self.distance_m < 0.0:
            raise SchemaValidationError("P4_2AttachConditionReport.distance_m must be non-negative")
        if self.relative_velocity_mps < 0.0:
            raise SchemaValidationError("P4_2AttachConditionReport.relative_velocity_mps must be non-negative")
        if self.attach_snap_distance_m < 0.0:
            raise SchemaValidationError("P4_2AttachConditionReport.attach_snap_distance_m must be non-negative")
        if self.relative_pose_error_m < 0.0:
            raise SchemaValidationError("P4_2AttachConditionReport.relative_pose_error_m must be non-negative")
        if self.attach_phase_elapsed_s < 0.0:
            raise SchemaValidationError("P4_2AttachConditionReport.attach_phase_elapsed_s must be non-negative")
        if self.attach_phase_timeout_s is not None and self.attach_phase_timeout_s <= 0.0:
            raise SchemaValidationError("P4_2AttachConditionReport.attach_phase_timeout_s must be positive")
        expected = (
            self.within_distance
            and self.within_relative_velocity
            and self.within_attach_snap_distance
            and self.within_attach_phase_timeout
            and self.assignment_feasible
            and self.controller_ok
        )
        if self.passed != expected:
            raise SchemaValidationError("P4_2AttachConditionReport.passed does not match condition flags")
        if not self.passed and not self.failure_reasons:
            raise SchemaValidationError("P4_2AttachConditionReport.failure_reasons required when attach fails")


@dataclass
class P4_2AttachEvent(SchemaBase):
    time_s: float
    phase: P4_2RolloutPhase
    event_type: str
    contact_model: str
    object_id: str
    candidate_id: int
    anchor_id: int
    slot_id: int
    contact_pose_world: Pose7D
    anchor_pose_world: Pose7D
    object_pose_world: Pose7D
    distance_m: float
    relative_velocity_mps: float
    attach_snap_distance_m: float
    relative_pose_error_m: float
    assignment_feasible: bool
    controller_ok: bool
    condition_report: P4_2AttachConditionReport
    candidate_ids: list[int] = field(default_factory=list)
    anchor_ids: list[int] = field(default_factory=list)
    slot_ids: list[int] = field(default_factory=list)
    contact_region_ids: list[str] = field(default_factory=list)
    distance_margins: dict[str, float] = field(default_factory=dict)
    assignment_feasibility: dict[str, Any] = field(default_factory=dict)
    anchor_link_id: str | None = None
    anchor_resolved_body_name: str | None = None
    anchor_pose_source: str = "module_state_fallback"
    anchor_link_pose_world: Pose7D | None = None
    anchor_local_pose_in_link: Pose7D | None = None
    anchor_link_twist_world: list[float] = field(default_factory=list)
    anchor_link_resolution: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.time_s < 0.0:
            raise SchemaValidationError("P4_2AttachEvent.time_s must be non-negative")
        if self.phase != P4_2RolloutPhase.ATTACH_ATTEMPT:
            raise SchemaValidationError("P4_2AttachEvent.phase must be attach_attempt")
        if self.event_type != "attach":
            raise SchemaValidationError("P4_2AttachEvent.event_type must be 'attach'")
        if self.contact_model != P4_2_CONTACT_MODEL:
            raise SchemaValidationError(f"P4_2AttachEvent.contact_model must be {P4_2_CONTACT_MODEL!r}")
        require_non_empty(self.object_id, "P4_2AttachEvent.object_id")
        require_len(self.contact_pose_world, 7, "P4_2AttachEvent.contact_pose_world")
        require_len(self.anchor_pose_world, 7, "P4_2AttachEvent.anchor_pose_world")
        require_len(self.object_pose_world, 7, "P4_2AttachEvent.object_pose_world")
        if not self.candidate_ids:
            raise SchemaValidationError("P4_2AttachEvent.candidate_ids must be non-empty")
        if not self.anchor_ids:
            raise SchemaValidationError("P4_2AttachEvent.anchor_ids must be non-empty")
        if not self.slot_ids:
            raise SchemaValidationError("P4_2AttachEvent.slot_ids must be non-empty")
        if not self.contact_region_ids:
            raise SchemaValidationError("P4_2AttachEvent.contact_region_ids must be non-empty")
        if self.attach_snap_distance_m < 0.0:
            raise SchemaValidationError("P4_2AttachEvent.attach_snap_distance_m must be non-negative")
        if self.relative_pose_error_m < 0.0:
            raise SchemaValidationError("P4_2AttachEvent.relative_pose_error_m must be non-negative")
        if not self.condition_report.passed:
            raise SchemaValidationError("P4_2AttachEvent requires a passed condition_report")
        require_non_empty(self.anchor_pose_source, "P4_2AttachEvent.anchor_pose_source")
        if self.anchor_pose_source == "isaac_link":
            require_non_empty(self.anchor_link_id or "", "P4_2AttachEvent.anchor_link_id")
            require_non_empty(self.anchor_resolved_body_name or "", "P4_2AttachEvent.anchor_resolved_body_name")
            if self.anchor_link_pose_world is None:
                raise SchemaValidationError("P4_2AttachEvent.anchor_link_pose_world is required for isaac_link anchors")
            if self.anchor_local_pose_in_link is None:
                raise SchemaValidationError("P4_2AttachEvent.anchor_local_pose_in_link is required for isaac_link anchors")
            require_len(self.anchor_link_pose_world, 7, "P4_2AttachEvent.anchor_link_pose_world")
            require_len(self.anchor_local_pose_in_link, 7, "P4_2AttachEvent.anchor_local_pose_in_link")
            if self.anchor_link_twist_world and len(self.anchor_link_twist_world) != 6:
                raise SchemaValidationError("P4_2AttachEvent.anchor_link_twist_world must have length 6")


@dataclass
class P4_2ReleaseEvent(SchemaBase):
    release_time_s: float
    phase: P4_2RolloutPhase
    event_type: str
    contact_model: str
    object_id: str
    object_pose_world: Pose7D
    robot_pose_world: Pose7D
    intended_release: bool
    post_release_object_pose_error_m: float

    def validate(self) -> None:
        if self.release_time_s < 0.0:
            raise SchemaValidationError("P4_2ReleaseEvent.release_time_s must be non-negative")
        if self.phase != P4_2RolloutPhase.RELEASE:
            raise SchemaValidationError("P4_2ReleaseEvent.phase must be release")
        if self.event_type != "release":
            raise SchemaValidationError("P4_2ReleaseEvent.event_type must be 'release'")
        if self.contact_model != P4_2_CONTACT_MODEL:
            raise SchemaValidationError(f"P4_2ReleaseEvent.contact_model must be {P4_2_CONTACT_MODEL!r}")
        require_non_empty(self.object_id, "P4_2ReleaseEvent.object_id")
        require_len(self.object_pose_world, 7, "P4_2ReleaseEvent.object_pose_world")
        require_len(self.robot_pose_world, 7, "P4_2ReleaseEvent.robot_pose_world")
        if self.post_release_object_pose_error_m < 0.0:
            raise SchemaValidationError("P4_2ReleaseEvent.post_release_object_pose_error_m must be non-negative")


@dataclass
class P4_2MetricDefinitions(SchemaBase):
    contact_model: str = P4_2_CONTACT_MODEL
    success_rate_definition: str = P4_2_SUCCESS_SCOPE_NOTE
    object_drop_definition: str = (
        "object_drop is true when an attached object separates before release or when object-to-grasp "
        "distance exceeds object_drop_distance_threshold_m during attached_maintain/transport."
    )
    hard_collision_definition: str = (
        "hard_collision is true for non-intended robot/object/environment collision events above the configured "
        "hard-contact threshold. Intended grasp contacts and kinematic attach contacts are excluded."
    )
    controller_qp_infeasible_terminal_definition: str = (
        "controller_qp_infeasible_terminal is true when rollout terminates in controller_failure because the "
        "controller or QP is infeasible for the configured consecutive-step threshold, or when the terminal "
        "ControllerStatus is infeasible/fault."
    )
    intended_contact_exclusion: str = (
        "selected grasp contacts, selected support contacts, and kinematic_payload_coupled_attach_v1 contacts are not "
        "counted as hard_collision."
    )

    def validate(self) -> None:
        if self.contact_model != P4_2_CONTACT_MODEL:
            raise SchemaValidationError(f"P4_2MetricDefinitions.contact_model must be {P4_2_CONTACT_MODEL!r}")
        for name in (
            "success_rate_definition",
            "object_drop_definition",
            "hard_collision_definition",
            "controller_qp_infeasible_terminal_definition",
            "intended_contact_exclusion",
        ):
            require_non_empty(getattr(self, name), f"P4_2MetricDefinitions.{name}")


@dataclass
class P4_2DeterministicRolloutResult(SchemaBase):
    rollout_name: str
    attempted: bool
    passed: bool
    skipped: bool
    isaac_backed: bool
    contact_model: str = P4_2_CONTACT_MODEL
    backend: str = "isaac_lab"
    uses_p2_selected_design: bool = False
    uses_p3_assembled_morphology: bool = False
    morphology_asset_reflected: bool = False
    module_placement_reflected: bool = False
    actuator_mapping_reflected: bool = False
    final_phase: P4_2RolloutPhase = P4_2RolloutPhase.RESET
    skip_reason: str | None = None
    phase_transitions: list[P4_2PhaseTransitionRecord] = field(default_factory=list)
    attach_events: list[P4_2AttachEvent] = field(default_factory=list)
    release_events: list[P4_2ReleaseEvent] = field(default_factory=list)
    runtime_observations: list[RuntimeObservation] = field(default_factory=list)
    policy_commands: list[PolicyCommand] = field(default_factory=list)
    controller_commands: list[ControllerCommand] = field(default_factory=list)
    actuator_target_records: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    rollout_artifacts: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        require_non_empty(self.rollout_name, "P4_2DeterministicRolloutResult.rollout_name")
        if self.contact_model != P4_2_CONTACT_MODEL:
            raise SchemaValidationError(
                f"P4_2DeterministicRolloutResult.contact_model must be {P4_2_CONTACT_MODEL!r}"
            )
        if self.skipped and self.skip_reason is None:
            raise SchemaValidationError("P4_2DeterministicRolloutResult.skip_reason is required when skipped")
        if self.passed and self.final_phase != P4_2RolloutPhase.SUCCESS:
            raise SchemaValidationError("P4_2DeterministicRolloutResult.passed requires final_phase=success")
        if self.passed and not self.attach_events:
            raise SchemaValidationError("P4.2 successful rollout requires at least one attach event")
        if self.passed and not self.release_events:
            raise SchemaValidationError("P4.2 successful rollout requires at least one release event")
        if self.passed and not (
            self.uses_p2_selected_design
            and self.uses_p3_assembled_morphology
            and self.morphology_asset_reflected
            and self.module_placement_reflected
            and self.actuator_mapping_reflected
        ):
            raise SchemaValidationError("P4.2 successful rollout must reflect the P2/P3 morphology in Isaac")


def load_p4_2_deterministic_rollout_config(path: str | Path) -> P4_2DeterministicRolloutConfig:
    data = load_config(path)
    return P4_2DeterministicRolloutConfig.from_dict(data.get("env", data))


def default_p4_2_phase_definitions(
    config: P4_2DeterministicRolloutConfig | None = None,
) -> list[P4_2PhaseDefinition]:
    config = config or P4_2DeterministicRolloutConfig()
    timeout = config.phase_timeouts_s
    return [
        P4_2PhaseDefinition(
            phase=P4_2RolloutPhase.RESET,
            terminal=False,
            entry_conditions=["Isaac scene reset requested with TaskSpec and P3 assembled MorphologyGraph"],
            exit_conditions=["robot, object, floor, controller, and actuator mapping initialized"],
            timeout_s=timeout["reset"],
            timeout_transition=P4_2RolloutPhase.TIMEOUT_FAILURE,
        ),
        P4_2PhaseDefinition(
            phase=P4_2RolloutPhase.APPROACH,
            terminal=False,
            entry_conditions=["reset completed and selected ContactWrenchTrajectory is available"],
            exit_conditions=["selected RobotAnchors move toward selected object contact candidates"],
            timeout_s=timeout["approach"],
            timeout_transition=P4_2RolloutPhase.TIMEOUT_FAILURE,
        ),
        P4_2PhaseDefinition(
            phase=P4_2RolloutPhase.PREGRASP_ALIGN,
            terminal=False,
            entry_conditions=["anchor-to-candidate distance is within pregrasp_alignment_distance_m"],
            exit_conditions=["relative pose and velocity are within attach-attempt gate"],
            timeout_s=timeout["pregrasp_align"],
            timeout_transition=P4_2RolloutPhase.TIMEOUT_FAILURE,
        ),
        P4_2PhaseDefinition(
            phase=P4_2RolloutPhase.ATTACH_ATTEMPT,
            terminal=False,
            entry_conditions=["selected assignment feasibility and controller status are acceptable"],
            exit_conditions=[
                "attach condition report passes distance, relative velocity, snap distance, timeout, feasibility, and controller gates"
            ],
            timeout_s=timeout["attach_attempt"],
            timeout_transition=P4_2RolloutPhase.DROP_FAILURE,
        ),
        P4_2PhaseDefinition(
            phase=P4_2RolloutPhase.ATTACHED_MAINTAIN,
            terminal=False,
            entry_conditions=["kinematic payload-coupled attach event was recorded"],
            exit_conditions=["attached object remains stable for the maintain dwell time"],
            timeout_s=timeout["attached_maintain"],
            timeout_transition=P4_2RolloutPhase.DROP_FAILURE,
        ),
        P4_2PhaseDefinition(
            phase=P4_2RolloutPhase.TRANSPORT,
            terminal=False,
            entry_conditions=["attached maintain completed without object drop or hard collision"],
            exit_conditions=[
                "attached object reaches the target release region or the bounded P4.2 payload-carry displacement gate"
            ],
            timeout_s=timeout["transport"],
            timeout_transition=P4_2RolloutPhase.TIMEOUT_FAILURE,
        ),
        P4_2PhaseDefinition(
            phase=P4_2RolloutPhase.RELEASE,
            terminal=False,
            entry_conditions=["object is attached and within release tolerance of the goal pose"],
            exit_conditions=["object is released and remains within goal tolerance"],
            timeout_s=timeout["release"],
            timeout_transition=P4_2RolloutPhase.TIMEOUT_FAILURE,
        ),
        P4_2PhaseDefinition(
            phase=P4_2RolloutPhase.SUCCESS,
            terminal=True,
            entry_conditions=["release completed and object remains inside goal tolerance"],
            exit_conditions=[],
        ),
        P4_2PhaseDefinition(
            phase=P4_2RolloutPhase.DROP_FAILURE,
            terminal=True,
            entry_conditions=["object_drop condition became true before successful release"],
            exit_conditions=[],
        ),
        P4_2PhaseDefinition(
            phase=P4_2RolloutPhase.COLLISION_FAILURE,
            terminal=True,
            entry_conditions=["hard_collision condition became true for a non-intended contact"],
            exit_conditions=[],
        ),
        P4_2PhaseDefinition(
            phase=P4_2RolloutPhase.CONTROLLER_FAILURE,
            terminal=True,
            entry_conditions=["controller/QP infeasible terminal condition became true"],
            exit_conditions=[],
        ),
        P4_2PhaseDefinition(
            phase=P4_2RolloutPhase.TIMEOUT_FAILURE,
            terminal=True,
            entry_conditions=["phase or episode timeout expired before a successful release"],
            exit_conditions=[],
        ),
    ]


def evaluate_p4_2_attach_conditions(
    *,
    candidate_id: int,
    anchor_id: int,
    slot_id: int,
    object_id: str,
    distance_m: float,
    relative_velocity_mps: float,
    assignment_feasible: bool,
    controller_status: ControllerStatus,
    attach_snap_distance_m: float | None = None,
    relative_pose_error_m: float | None = None,
    attach_phase_elapsed_s: float = 0.0,
    attach_phase_timeout_s: float | None = None,
    config: P4_2DeterministicRolloutConfig | None = None,
) -> P4_2AttachConditionReport:
    config = config or P4_2DeterministicRolloutConfig()
    controller_ok = not p4_2_controller_status_is_fatal(controller_status)
    within_distance = distance_m <= config.attach_distance_threshold_m
    within_relative_velocity = relative_velocity_mps <= config.attach_relative_velocity_threshold_mps
    snap_distance = distance_m if attach_snap_distance_m is None else float(attach_snap_distance_m)
    pose_error = snap_distance if relative_pose_error_m is None else float(relative_pose_error_m)
    within_attach_snap_distance = snap_distance <= config.attach_snap_distance_threshold_m
    timeout_s = (
        config.phase_timeouts_s.get(P4_2RolloutPhase.ATTACH_ATTEMPT.value)
        if attach_phase_timeout_s is None
        else float(attach_phase_timeout_s)
    )
    within_attach_phase_timeout = timeout_s is None or float(attach_phase_elapsed_s) <= timeout_s
    failure_reasons: list[str] = []
    if not within_distance:
        failure_reasons.append("anchor_candidate_distance_above_threshold")
    if not within_relative_velocity:
        failure_reasons.append("relative_velocity_above_threshold")
    if not within_attach_snap_distance:
        failure_reasons.append("attach_snap_distance_above_threshold")
    if not within_attach_phase_timeout:
        failure_reasons.append("attach_phase_timeout_exceeded")
    if not assignment_feasible:
        failure_reasons.append("assignment_feasibility_failed")
    if not controller_ok:
        failure_reasons.append("controller_status_not_attach_safe")
    passed = (
        within_distance
        and within_relative_velocity
        and within_attach_snap_distance
        and within_attach_phase_timeout
        and assignment_feasible
        and controller_ok
    )
    return P4_2AttachConditionReport(
        candidate_id=candidate_id,
        anchor_id=anchor_id,
        slot_id=slot_id,
        object_id=object_id,
        distance_m=float(distance_m),
        relative_velocity_mps=float(relative_velocity_mps),
        attach_snap_distance_m=snap_distance,
        relative_pose_error_m=pose_error,
        assignment_feasible=assignment_feasible,
        controller_ok=controller_ok,
        within_distance=within_distance,
        within_relative_velocity=within_relative_velocity,
        within_attach_snap_distance=within_attach_snap_distance,
        within_attach_phase_timeout=within_attach_phase_timeout,
        passed=passed,
        attach_phase_elapsed_s=float(attach_phase_elapsed_s),
        attach_phase_timeout_s=timeout_s,
        failure_reasons=failure_reasons,
        metrics={
            "attach_distance_m": float(distance_m),
            "attach_relative_velocity_mps": float(relative_velocity_mps),
            "attach_snap_distance_m": snap_distance,
            "attach_relative_pose_error_m": pose_error,
            "attach_phase_elapsed_s": float(attach_phase_elapsed_s),
            "attach_phase_timeout_s": float(timeout_s or 0.0),
            "attach_assignment_feasible": 1.0 if assignment_feasible else 0.0,
            "attach_controller_ok": 1.0 if controller_ok else 0.0,
            "attach_condition_passed": 1.0 if passed else 0.0,
        },
    )


def p4_2_controller_status_is_fatal(controller_status: ControllerStatus) -> bool:
    if controller_status.status == "fault":
        return True
    if controller_status.qp_feasible:
        return False
    qp_solver_success = controller_status.metrics.get("qp_solver_success")
    if qp_solver_success is not None:
        return float(qp_solver_success) <= 0.0
    return controller_status.status == "infeasible"


def p4_2_metric_definitions() -> P4_2MetricDefinitions:
    return P4_2MetricDefinitions()


def p4_2_no_mislabeling_artifacts(*, backend: str = "isaac_lab") -> dict[str, Any]:
    return {
        "phase": "P4.2",
        "backend": backend,
        "contact_model": P4_2_CONTACT_MODEL,
        "success_rate_scope_note": P4_2_SUCCESS_SCOPE_NOTE,
        "is_p4_full_completion": False,
        "p4_3_learning_bootstrap": False,
        "learning_claim": False,
        "learned_policy_success_claim": False,
        "high_fidelity_natural_grasp_success_claim": False,
        "true_fixed_joint_dynamics_success_claim": False,
        "checkpoint_claim": False,
        "reward_curve_training_claim": False,
        "p4_4_natural_contact_grasp_remaining": True,
    }


def p4_2_failure_metrics(
    *,
    final_phase: P4_2RolloutPhase,
    object_drop: bool = False,
    hard_collision: bool = False,
    controller_qp_infeasible_terminal: bool = False,
) -> dict[str, float]:
    return {
        "success": 1.0 if final_phase == P4_2RolloutPhase.SUCCESS else 0.0,
        "object_drop": 1.0 if object_drop or final_phase == P4_2RolloutPhase.DROP_FAILURE else 0.0,
        "hard_collision": 1.0 if hard_collision or final_phase == P4_2RolloutPhase.COLLISION_FAILURE else 0.0,
        "controller_qp_infeasible_terminal": 1.0
        if controller_qp_infeasible_terminal or final_phase == P4_2RolloutPhase.CONTROLLER_FAILURE
        else 0.0,
        "timeout_failure": 1.0 if final_phase == P4_2RolloutPhase.TIMEOUT_FAILURE else 0.0,
        "isaac_backed": 1.0,
        "p4_full_completion": 0.0,
        "p4_3_learning_bootstrap": 0.0,
        "learned_policy_success_claim": 0.0,
        "high_fidelity_natural_grasp_success_claim": 0.0,
        "true_fixed_joint_dynamics_success_claim": 0.0,
    }
