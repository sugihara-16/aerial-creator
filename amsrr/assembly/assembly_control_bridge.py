from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from amsrr.geometry.pose_math import (
    FACE_TO_FACE_DOCK_RELATION,
    Transform3D,
    compose_transform,
    inverse_transform,
    matmul,
    pose_from_transform,
    transform_from_pose,
    transpose,
)
from amsrr.schemas.common import (
    Pose7D,
    SchemaBase,
    SchemaValidationError,
    Vector3,
    require_len,
    require_non_empty,
)
from amsrr.schemas.morphology import MorphologyGraph, PortNode
from amsrr.schemas.physical_model import JointModel, PhysicalModel
from amsrr.schemas.policies import (
    POLICY_COMMAND_CONTRACT_CENTROIDAL,
    PolicyCommand,
)


ASSEMBLY_CONTROL_BRIDGE_CONTRACT_VERSION = "assembly_component_policy_v1"
AssemblyControlPhase = Literal[
    "staging",
    "prealign_dwell",
    "axial_approach",
    "fix_ready",
    "verify",
    "safe_hold",
]
AssemblyConstraintAction = Literal["none", "create", "verify"]
ComponentRole = Literal["leader", "follower"]


@dataclass(frozen=True)
class AssemblyControlBridgeConfig:
    staging_offset_m: float = 0.15
    staging_axial_tolerance_m: float = 0.01
    transverse_tolerance_m: float = 0.01
    attitude_tolerance_rad: float = math.radians(3.0)
    relative_linear_speed_tolerance_mps: float = 0.05
    relative_angular_speed_tolerance_radps: float = 0.10
    prealign_dwell_s: float = 0.30
    approach_speed_mps: float = 0.02
    fix_axial_tolerance_m: float = 0.003
    selected_contact_dwell_s: float = 0.10
    max_selected_contact_force_n: float = 30.0
    max_selected_contact_penetration_m: float = 0.002
    max_joint_correction_rad: float = math.radians(5.0)
    step_timeout_s: float = 30.0
    require_selected_pair_contact: bool = True

    def __post_init__(self) -> None:
        positive = {
            "staging_offset_m": self.staging_offset_m,
            "staging_axial_tolerance_m": self.staging_axial_tolerance_m,
            "transverse_tolerance_m": self.transverse_tolerance_m,
            "attitude_tolerance_rad": self.attitude_tolerance_rad,
            "relative_linear_speed_tolerance_mps": self.relative_linear_speed_tolerance_mps,
            "relative_angular_speed_tolerance_radps": self.relative_angular_speed_tolerance_radps,
            "prealign_dwell_s": self.prealign_dwell_s,
            "approach_speed_mps": self.approach_speed_mps,
            "fix_axial_tolerance_m": self.fix_axial_tolerance_m,
            "selected_contact_dwell_s": self.selected_contact_dwell_s,
            "max_selected_contact_force_n": self.max_selected_contact_force_n,
            "max_selected_contact_penetration_m": self.max_selected_contact_penetration_m,
            "max_joint_correction_rad": self.max_joint_correction_rad,
            "step_timeout_s": self.step_timeout_s,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise SchemaValidationError(f"AssemblyControlBridgeConfig.{name} must be finite and positive")
        if self.fix_axial_tolerance_m > self.staging_offset_m:
            raise SchemaValidationError(
                "AssemblyControlBridgeConfig.fix_axial_tolerance_m cannot exceed staging_offset_m"
            )


@dataclass
class AssemblyComponentSpec(SchemaBase):
    component_id: str
    module_ids: list[int]

    def validate(self) -> None:
        require_non_empty(self.component_id, "AssemblyComponentSpec.component_id")
        if not self.module_ids:
            raise SchemaValidationError("AssemblyComponentSpec.module_ids must be non-empty")
        if any(module_id < 0 for module_id in self.module_ids):
            raise SchemaValidationError("AssemblyComponentSpec.module_ids must be non-negative")
        if self.module_ids != sorted(set(self.module_ids)):
            raise SchemaValidationError("AssemblyComponentSpec.module_ids must be sorted and unique")


@dataclass
class AssemblyControlRequest(SchemaBase):
    step_id: int
    leader: AssemblyComponentSpec
    follower: AssemblyComponentSpec
    leader_port_id: int
    follower_port_id: int
    leader_joint_corrections_rad: dict[str, float] = field(default_factory=dict)
    follower_joint_corrections_rad: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        if min(self.step_id, self.leader_port_id, self.follower_port_id) < 0:
            raise SchemaValidationError("AssemblyControlRequest ids must be non-negative")
        if self.leader.component_id == self.follower.component_id:
            raise SchemaValidationError("AssemblyControlRequest components must be distinct")
        if set(self.leader.module_ids) & set(self.follower.module_ids):
            raise SchemaValidationError("AssemblyControlRequest components must be disjoint")
        for path, values in (
            ("leader_joint_corrections_rad", self.leader_joint_corrections_rad),
            ("follower_joint_corrections_rad", self.follower_joint_corrections_rad),
        ):
            for joint_id, value in values.items():
                require_non_empty(joint_id, f"AssemblyControlRequest.{path}.key")
                if not math.isfinite(float(value)):
                    raise SchemaValidationError(
                        f"AssemblyControlRequest.{path}[{joint_id!r}] must be finite"
                    )


@dataclass
class AssemblyComponentObservation(SchemaBase):
    component_id: str
    module_ids: list[int]
    body_pose_world: Pose7D
    selected_connect_pose_world: Pose7D
    selected_connect_linear_velocity_world: Vector3
    selected_connect_angular_velocity_world: Vector3
    qp_feasible: bool

    def validate(self) -> None:
        require_non_empty(self.component_id, "AssemblyComponentObservation.component_id")
        if self.module_ids != sorted(set(self.module_ids)) or not self.module_ids:
            raise SchemaValidationError(
                "AssemblyComponentObservation.module_ids must be sorted, unique, and non-empty"
            )
        require_len(self.body_pose_world, 7, "AssemblyComponentObservation.body_pose_world")
        require_len(
            self.selected_connect_pose_world,
            7,
            "AssemblyComponentObservation.selected_connect_pose_world",
        )
        require_len(
            self.selected_connect_linear_velocity_world,
            3,
            "AssemblyComponentObservation.selected_connect_linear_velocity_world",
        )
        require_len(
            self.selected_connect_angular_velocity_world,
            3,
            "AssemblyComponentObservation.selected_connect_angular_velocity_world",
        )
        _require_finite_sequence(self.body_pose_world, "AssemblyComponentObservation.body_pose_world")
        _require_finite_sequence(
            self.selected_connect_pose_world,
            "AssemblyComponentObservation.selected_connect_pose_world",
        )
        _require_finite_sequence(
            self.selected_connect_linear_velocity_world,
            "AssemblyComponentObservation.selected_connect_linear_velocity_world",
        )
        _require_finite_sequence(
            self.selected_connect_angular_velocity_world,
            "AssemblyComponentObservation.selected_connect_angular_velocity_world",
        )
        _require_nonzero_pose_quaternion(
            self.body_pose_world,
            "AssemblyComponentObservation.body_pose_world",
        )
        _require_nonzero_pose_quaternion(
            self.selected_connect_pose_world,
            "AssemblyComponentObservation.selected_connect_pose_world",
        )


@dataclass
class AssemblyControlObservation(SchemaBase):
    time_s: float
    components: list[AssemblyComponentObservation]
    selected_pair_contact: bool = False
    selected_pair_contact_evidence_valid: bool = False
    selected_pair_contact_force_n: float = 0.0
    selected_pair_penetration_m: float = 0.0
    unintended_contact: bool = False
    constraint_present: bool = False
    constraint_verified: bool = False

    def validate(self) -> None:
        if not math.isfinite(self.time_s) or self.time_s < 0.0:
            raise SchemaValidationError("AssemblyControlObservation.time_s must be finite and non-negative")
        component_ids = [component.component_id for component in self.components]
        if len(component_ids) != len(set(component_ids)):
            raise SchemaValidationError("AssemblyControlObservation.components has duplicate component_id values")
        if self.constraint_verified and not self.constraint_present:
            raise SchemaValidationError(
                "AssemblyControlObservation.constraint_verified requires constraint_present"
            )
        for name in ("selected_pair_contact_force_n", "selected_pair_penetration_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise SchemaValidationError(
                    f"AssemblyControlObservation.{name} must be finite and non-negative"
                )
        if self.selected_pair_contact_evidence_valid and not self.selected_pair_contact:
            raise SchemaValidationError(
                "Valid selected-pair contact evidence requires selected_pair_contact"
            )


@dataclass
class AssemblyAlignmentError(SchemaBase):
    axial_gap_m: float
    axial_target_error_m: float
    transverse_error_m: float
    attitude_error_rad: float
    relative_linear_speed_mps: float
    relative_angular_speed_radps: float

    def validate(self) -> None:
        for name in (
            "axial_gap_m",
            "axial_target_error_m",
            "transverse_error_m",
            "attitude_error_rad",
            "relative_linear_speed_mps",
            "relative_angular_speed_radps",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise SchemaValidationError(f"AssemblyAlignmentError.{name} must be finite")
        if min(
            self.axial_target_error_m,
            self.transverse_error_m,
            self.attitude_error_rad,
            self.relative_linear_speed_mps,
            self.relative_angular_speed_radps,
        ) < 0.0:
            raise SchemaValidationError("AssemblyAlignmentError magnitudes must be non-negative")


@dataclass
class AssemblyComponentPolicyTarget(SchemaBase):
    component_id: str
    role: ComponentRole
    module_ids: list[int]
    policy_command: PolicyCommand

    def validate(self) -> None:
        require_non_empty(self.component_id, "AssemblyComponentPolicyTarget.component_id")
        if not self.module_ids:
            raise SchemaValidationError("AssemblyComponentPolicyTarget.module_ids must be non-empty")
        if self.policy_command.control_contract_version != POLICY_COMMAND_CONTRACT_CENTROIDAL:
            raise SchemaValidationError(
                "AssemblyComponentPolicyTarget requires centroidal_local_joint_v2"
            )
        if self.policy_command.contact_tracking_bias:
            raise SchemaValidationError(
                "AssemblyComponentPolicyTarget cannot emit contact wrench bias"
            )


@dataclass
class AssemblyConstraintIntent(SchemaBase):
    action: AssemblyConstraintAction
    leader_port_id: int
    follower_port_id: int
    required_relative_pose: Pose7D

    def validate(self) -> None:
        if min(self.leader_port_id, self.follower_port_id) < 0:
            raise SchemaValidationError("AssemblyConstraintIntent port ids must be non-negative")
        require_len(
            self.required_relative_pose,
            7,
            "AssemblyConstraintIntent.required_relative_pose",
        )
        if not _poses_equal(self.required_relative_pose, FACE_TO_FACE_DOCK_RELATION):
            raise SchemaValidationError(
                "AssemblyConstraintIntent must use the canonical face-to-face relation"
            )


@dataclass
class AssemblyComponentCommandBundle(SchemaBase):
    contract_version: str
    step_id: int
    phase: AssemblyControlPhase
    component_targets: list[AssemblyComponentPolicyTarget]
    constraint_intent: AssemblyConstraintIntent

    def validate(self) -> None:
        if self.contract_version != ASSEMBLY_CONTROL_BRIDGE_CONTRACT_VERSION:
            raise SchemaValidationError(
                "AssemblyComponentCommandBundle has an unsupported contract_version"
            )
        if self.step_id < 0:
            raise SchemaValidationError("AssemblyComponentCommandBundle.step_id must be non-negative")
        component_ids = [target.component_id for target in self.component_targets]
        if len(component_ids) != 2 or len(set(component_ids)) != 2:
            raise SchemaValidationError(
                "AssemblyComponentCommandBundle requires exactly two distinct components"
            )
        if {target.role for target in self.component_targets} != {"leader", "follower"}:
            raise SchemaValidationError(
                "AssemblyComponentCommandBundle requires leader and follower targets"
            )


@dataclass
class AssemblyControlStepProgress(SchemaBase):
    step_id: int
    phase: AssemblyControlPhase
    phase_elapsed_s: float
    selected_contact_dwell_elapsed_s: float
    alignment_error: AssemblyAlignmentError
    gate_results: dict[str, bool]
    completed: bool
    failed: bool
    failure_reason: str | None = None

    def validate(self) -> None:
        if self.step_id < 0:
            raise SchemaValidationError("AssemblyControlStepProgress.step_id must be non-negative")
        if not math.isfinite(self.phase_elapsed_s) or self.phase_elapsed_s < 0.0:
            raise SchemaValidationError(
                "AssemblyControlStepProgress.phase_elapsed_s must be finite and non-negative"
            )
        if (
            not math.isfinite(self.selected_contact_dwell_elapsed_s)
            or self.selected_contact_dwell_elapsed_s < 0.0
        ):
            raise SchemaValidationError(
                "AssemblyControlStepProgress.selected_contact_dwell_elapsed_s must be finite and non-negative"
            )
        if self.completed and self.failed:
            raise SchemaValidationError(
                "AssemblyControlStepProgress cannot be both completed and failed"
            )
        if self.failed != (self.failure_reason is not None):
            raise SchemaValidationError(
                "AssemblyControlStepProgress failed and failure_reason must agree"
            )


@dataclass
class AssemblyControlStepOutput(SchemaBase):
    commands: AssemblyComponentCommandBundle
    progress: AssemblyControlStepProgress

    def validate(self) -> None:
        if self.commands.step_id != self.progress.step_id:
            raise SchemaValidationError("AssemblyControlStepOutput step ids must agree")
        if self.commands.phase != self.progress.phase:
            raise SchemaValidationError("AssemblyControlStepOutput phases must agree")


class AssemblyControlBridge:
    """Deterministic component-level assembly targets; never final actuators.

    The bridge is intentionally simulator-agnostic.  It converts observed
    component/connect-frame state into two v2 ``PolicyCommand`` objects and an
    exact-frame constraint intent that a later Isaac-backed executor may honor.
    """

    def __init__(
        self,
        morphology: MorphologyGraph,
        physical_models_by_module_id: dict[int, PhysicalModel],
        *,
        config: AssemblyControlBridgeConfig | None = None,
    ) -> None:
        self.morphology = morphology
        self.physical_models_by_module_id = dict(physical_models_by_module_id)
        self.config = config or AssemblyControlBridgeConfig()
        self._request: AssemblyControlRequest | None = None
        self._phase: AssemblyControlPhase | None = None
        self._started_s = 0.0
        self._phase_started_s = 0.0
        self._last_time_s = 0.0
        self._leader_hold_pose: Pose7D | None = None
        self._safe_hold_poses: dict[str, Pose7D] = {}
        self._failure_reason: str | None = None
        self._completed = False
        self._selected_contact_started_s: float | None = None

    def begin(
        self,
        request: AssemblyControlRequest,
        observation: AssemblyControlObservation,
    ) -> AssemblyControlStepOutput:
        self._validate_request(request)
        components = self._require_components(request, observation)
        self._request = request
        self._phase = "staging"
        self._started_s = observation.time_s
        self._phase_started_s = observation.time_s
        self._last_time_s = observation.time_s
        self._leader_hold_pose = components[request.leader.component_id].body_pose_world
        self._safe_hold_poses = {
            component_id: component.body_pose_world
            for component_id, component in components.items()
        }
        self._failure_reason = None
        self._completed = False
        self._selected_contact_started_s = None
        return self._build_output(observation, components)

    def tick(self, observation: AssemblyControlObservation) -> AssemblyControlStepOutput:
        request = self._require_session()
        if observation.time_s < self._last_time_s:
            raise SchemaValidationError(
                "AssemblyControlObservation.time_s must be monotonic within a bridge session"
            )
        components = self._require_components(request, observation)
        self._last_time_s = observation.time_s

        if self._phase != "safe_hold" and not self._completed:
            if observation.unintended_contact:
                self._enter_safe_hold("unintended_contact", components, observation.time_s)
            elif not all(component.qp_feasible for component in components.values()):
                self._enter_safe_hold("component_qp_infeasible", components, observation.time_s)
            elif observation.constraint_present and self._phase in {
                "staging",
                "prealign_dwell",
                "axial_approach",
            }:
                self._enter_safe_hold(
                    "constraint_present_before_fix_gate",
                    components,
                    observation.time_s,
                )
            elif (
                observation.selected_pair_contact
                and not observation.selected_pair_contact_evidence_valid
            ):
                self._enter_safe_hold(
                    "selected_pair_contact_evidence_invalid",
                    components,
                    observation.time_s,
                )
            elif (
                observation.selected_pair_contact_force_n
                > self.config.max_selected_contact_force_n
            ):
                self._enter_safe_hold(
                    "selected_pair_contact_force_exceeded",
                    components,
                    observation.time_s,
                )
            elif (
                observation.selected_pair_penetration_m
                > self.config.max_selected_contact_penetration_m
            ):
                self._enter_safe_hold(
                    "selected_pair_contact_penetration_exceeded",
                    components,
                    observation.time_s,
                )
            elif observation.selected_pair_contact and self._phase in {
                "staging",
                "prealign_dwell",
            }:
                self._enter_safe_hold(
                    "selected_pair_contact_before_approach",
                    components,
                    observation.time_s,
                )
            elif observation.time_s - self._started_s > self.config.step_timeout_s:
                self._enter_safe_hold("assembly_step_timeout", components, observation.time_s)
            else:
                self._advance_phase(observation, components)
        return self._build_output(observation, components)

    def enter_safe_hold(
        self,
        observation: AssemblyControlObservation,
        *,
        reason: str,
    ) -> AssemblyControlStepOutput:
        require_non_empty(reason, "AssemblyControlBridge.enter_safe_hold.reason")
        request = self._require_session()
        components = self._require_components(request, observation)
        self._last_time_s = observation.time_s
        self._enter_safe_hold(reason, components, observation.time_s)
        return self._build_output(observation, components)

    @property
    def phase(self) -> AssemblyControlPhase | None:
        return self._phase

    def _advance_phase(
        self,
        observation: AssemblyControlObservation,
        components: dict[str, AssemblyComponentObservation],
    ) -> None:
        phase = self._require_phase()
        desired_gap = self._desired_axial_gap(observation.time_s)
        errors = self._alignment_error(components, desired_gap)
        alignment_ok = self._alignment_gate(errors)
        twist_ok = self._twist_gate(errors)

        if phase == "staging":
            if alignment_ok and twist_ok:
                self._transition("prealign_dwell", observation.time_s)
            return

        if phase == "prealign_dwell":
            if not (alignment_ok and twist_ok):
                self._transition("staging", observation.time_s)
            elif observation.time_s - self._phase_started_s >= self.config.prealign_dwell_s:
                self._transition("axial_approach", observation.time_s)
            return

        if phase == "axial_approach":
            if observation.selected_pair_contact:
                if self._fix_gate(errors, observation):
                    if self._selected_contact_started_s is None:
                        self._selected_contact_started_s = observation.time_s
                    elif (
                        observation.time_s - self._selected_contact_started_s
                        >= self.config.selected_contact_dwell_s
                    ):
                        self._transition("fix_ready", observation.time_s)
                else:
                    self._enter_safe_hold(
                        "selected_pair_contact_before_fix_gate",
                        components,
                        observation.time_s,
                    )
                return
            self._selected_contact_started_s = None
            if not self.config.require_selected_pair_contact and self._fix_gate(errors, observation):
                self._transition("fix_ready", observation.time_s)
            return

        if phase == "fix_ready":
            if observation.constraint_present:
                self._transition("verify", observation.time_s)
            return

        if phase == "verify":
            if not observation.constraint_present:
                self._enter_safe_hold("constraint_lost_during_verify", components, observation.time_s)
            elif observation.constraint_verified:
                self._completed = True

    def _build_output(
        self,
        observation: AssemblyControlObservation,
        components: dict[str, AssemblyComponentObservation],
    ) -> AssemblyControlStepOutput:
        request = self._require_session()
        phase = self._require_phase()
        desired_gap = self._desired_axial_gap(observation.time_s)
        errors = self._alignment_error(components, desired_gap)

        if phase == "safe_hold":
            leader_pose = self._safe_hold_poses[request.leader.component_id]
            follower_pose = self._safe_hold_poses[request.follower.component_id]
        else:
            leader_pose = self._require_leader_hold_pose()
            follower_pose = self._follower_body_target(components, desired_gap)

        leader_command = self._policy_command(
            request.leader,
            leader_pose,
            request.leader_joint_corrections_rad,
            phase,
        )
        follower_command = self._policy_command(
            request.follower,
            follower_pose,
            request.follower_joint_corrections_rad,
            phase,
        )
        constraint_action: AssemblyConstraintAction = "none"
        if phase == "fix_ready":
            constraint_action = "create"
        elif phase == "verify":
            constraint_action = "verify"

        gate_results = {
            "component_qp_feasible": all(component.qp_feasible for component in components.values()),
            "no_unintended_contact": not observation.unintended_contact,
            "axial_target": errors.axial_target_error_m <= self._axial_tolerance_for_phase(phase),
            "transverse": errors.transverse_error_m <= self.config.transverse_tolerance_m,
            "attitude": errors.attitude_error_rad <= self.config.attitude_tolerance_rad,
            "relative_linear_speed": (
                errors.relative_linear_speed_mps
                <= self.config.relative_linear_speed_tolerance_mps
            ),
            "relative_angular_speed": (
                errors.relative_angular_speed_radps
                <= self.config.relative_angular_speed_tolerance_radps
            ),
            "selected_pair_contact": observation.selected_pair_contact,
            "selected_pair_contact_evidence_valid": (
                observation.selected_pair_contact_evidence_valid
            ),
            "selected_pair_contact_force_safe": (
                observation.selected_pair_contact_force_n
                <= self.config.max_selected_contact_force_n
            ),
            "selected_pair_contact_penetration_safe": (
                observation.selected_pair_penetration_m
                <= self.config.max_selected_contact_penetration_m
            ),
            "selected_pair_contact_dwell": (
                self._selected_contact_dwell_elapsed(observation.time_s)
                >= self.config.selected_contact_dwell_s
            ),
            "constraint_present": observation.constraint_present,
            "constraint_verified": observation.constraint_verified,
        }
        commands = AssemblyComponentCommandBundle(
            contract_version=ASSEMBLY_CONTROL_BRIDGE_CONTRACT_VERSION,
            step_id=request.step_id,
            phase=phase,
            component_targets=[
                AssemblyComponentPolicyTarget(
                    component_id=request.leader.component_id,
                    role="leader",
                    module_ids=list(request.leader.module_ids),
                    policy_command=leader_command,
                ),
                AssemblyComponentPolicyTarget(
                    component_id=request.follower.component_id,
                    role="follower",
                    module_ids=list(request.follower.module_ids),
                    policy_command=follower_command,
                ),
            ],
            constraint_intent=AssemblyConstraintIntent(
                action=constraint_action,
                leader_port_id=request.leader_port_id,
                follower_port_id=request.follower_port_id,
                required_relative_pose=FACE_TO_FACE_DOCK_RELATION,
            ),
        )
        progress = AssemblyControlStepProgress(
            step_id=request.step_id,
            phase=phase,
            phase_elapsed_s=max(0.0, observation.time_s - self._phase_started_s),
            selected_contact_dwell_elapsed_s=self._selected_contact_dwell_elapsed(
                observation.time_s
            ),
            alignment_error=errors,
            gate_results=gate_results,
            completed=self._completed,
            failed=self._failure_reason is not None,
            failure_reason=self._failure_reason,
        )
        return AssemblyControlStepOutput(commands=commands, progress=progress)

    def _policy_command(
        self,
        component: AssemblyComponentSpec,
        body_pose_world: Pose7D,
        corrections: dict[str, float],
        phase: AssemblyControlPhase,
    ) -> PolicyCommand:
        joint_targets = self._canonical_joint_targets(component, corrections)
        return PolicyCommand(
            desired_body_twist=[0.0] * 6,
            desired_body_pose=body_pose_world,
            residual_wrench_body=[0.0] * 6,
            contact_tracking_bias={},
            priority_weights={"assembly_pose": 1.0, f"assembly_phase:{phase}": 1.0},
            control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
            joint_position_targets=joint_targets,
            joint_velocity_targets={joint_id: 0.0 for joint_id in joint_targets},
            joint_torque_bias={joint_id: 0.0 for joint_id in joint_targets},
        )

    def _canonical_joint_targets(
        self,
        component: AssemblyComponentSpec,
        corrections: dict[str, float],
    ) -> dict[str, float]:
        targets: dict[str, float] = {}
        for module_id in component.module_ids:
            model = self.physical_models_by_module_id[module_id]
            for local_joint_id in _dock_mechanism_joint_ids(model):
                global_joint_id = f"module_{module_id}:{local_joint_id}"
                targets[global_joint_id] = float(corrections.get(global_joint_id, 0.0))
        return dict(sorted(targets.items()))

    def _alignment_error(
        self,
        components: dict[str, AssemblyComponentObservation],
        desired_gap_m: float,
    ) -> AssemblyAlignmentError:
        request = self._require_session()
        leader = components[request.leader.component_id]
        follower = components[request.follower.component_id]
        leader_connect = transform_from_pose(leader.selected_connect_pose_world)
        follower_connect = transform_from_pose(follower.selected_connect_pose_world)
        relative = compose_transform(inverse_transform(leader_connect), follower_connect)
        axial_gap = float(relative.translation[0])
        transverse_error = math.hypot(relative.translation[1], relative.translation[2])
        desired_rotation = transform_from_pose(FACE_TO_FACE_DOCK_RELATION).rotation
        rotation_error = matmul(transpose(desired_rotation), relative.rotation)
        attitude_error = _rotation_angle(rotation_error)
        relative_linear = _norm3(
            _subtract3(
                follower.selected_connect_linear_velocity_world,
                leader.selected_connect_linear_velocity_world,
            )
        )
        relative_angular = _norm3(
            _subtract3(
                follower.selected_connect_angular_velocity_world,
                leader.selected_connect_angular_velocity_world,
            )
        )
        return AssemblyAlignmentError(
            axial_gap_m=axial_gap,
            axial_target_error_m=abs(axial_gap - desired_gap_m),
            transverse_error_m=transverse_error,
            attitude_error_rad=attitude_error,
            relative_linear_speed_mps=relative_linear,
            relative_angular_speed_radps=relative_angular,
        )

    def _follower_body_target(
        self,
        components: dict[str, AssemblyComponentObservation],
        desired_gap_m: float,
    ) -> Pose7D:
        request = self._require_session()
        leader = components[request.leader.component_id]
        follower = components[request.follower.component_id]
        leader_connect_world = transform_from_pose(leader.selected_connect_pose_world)
        offset = Transform3D(
            rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            translation=(desired_gap_m, 0.0, 0.0),
        )
        desired_follower_connect_world = compose_transform(
            compose_transform(leader_connect_world, offset),
            transform_from_pose(FACE_TO_FACE_DOCK_RELATION),
        )
        follower_body_world = transform_from_pose(follower.body_pose_world)
        follower_connect_world = transform_from_pose(follower.selected_connect_pose_world)
        follower_connect_in_body = compose_transform(
            inverse_transform(follower_body_world),
            follower_connect_world,
        )
        desired_follower_body_world = compose_transform(
            desired_follower_connect_world,
            inverse_transform(follower_connect_in_body),
        )
        return pose_from_transform(desired_follower_body_world)

    def _desired_axial_gap(self, time_s: float) -> float:
        phase = self._require_phase()
        if phase in {"staging", "prealign_dwell"}:
            return self.config.staging_offset_m
        if phase == "axial_approach":
            elapsed = max(0.0, time_s - self._phase_started_s)
            return max(0.0, self.config.staging_offset_m - self.config.approach_speed_mps * elapsed)
        return 0.0

    def _alignment_gate(self, errors: AssemblyAlignmentError) -> bool:
        return (
            errors.axial_target_error_m <= self.config.staging_axial_tolerance_m
            and errors.transverse_error_m <= self.config.transverse_tolerance_m
            and errors.attitude_error_rad <= self.config.attitude_tolerance_rad
        )

    def _twist_gate(self, errors: AssemblyAlignmentError) -> bool:
        return (
            errors.relative_linear_speed_mps
            <= self.config.relative_linear_speed_tolerance_mps
            and errors.relative_angular_speed_radps
            <= self.config.relative_angular_speed_tolerance_radps
        )

    def _fix_gate(
        self,
        errors: AssemblyAlignmentError,
        observation: AssemblyControlObservation,
    ) -> bool:
        contact_ok = observation.selected_pair_contact or not self.config.require_selected_pair_contact
        return (
            contact_ok
            and (
                observation.selected_pair_contact_evidence_valid
                or not self.config.require_selected_pair_contact
            )
            and observation.selected_pair_contact_force_n <= self.config.max_selected_contact_force_n
            and observation.selected_pair_penetration_m <= self.config.max_selected_contact_penetration_m
            and abs(errors.axial_gap_m) <= self.config.fix_axial_tolerance_m
            and errors.transverse_error_m <= self.config.transverse_tolerance_m
            and errors.attitude_error_rad <= self.config.attitude_tolerance_rad
            and self._twist_gate(errors)
        )

    def _axial_tolerance_for_phase(self, phase: AssemblyControlPhase) -> float:
        if phase in {"fix_ready", "verify"}:
            return self.config.fix_axial_tolerance_m
        return self.config.staging_axial_tolerance_m

    def _transition(self, phase: AssemblyControlPhase, time_s: float) -> None:
        if phase in {"staging", "prealign_dwell", "axial_approach"}:
            self._selected_contact_started_s = None
        self._phase = phase
        self._phase_started_s = time_s

    def _selected_contact_dwell_elapsed(self, time_s: float) -> float:
        if self._selected_contact_started_s is None:
            return 0.0
        return max(0.0, time_s - self._selected_contact_started_s)

    def _enter_safe_hold(
        self,
        reason: str,
        components: dict[str, AssemblyComponentObservation],
        time_s: float,
    ) -> None:
        self._safe_hold_poses = {
            component_id: component.body_pose_world
            for component_id, component in components.items()
        }
        self._failure_reason = reason
        self._completed = False
        self._transition("safe_hold", time_s)

    def _validate_request(self, request: AssemblyControlRequest) -> None:
        graph_module_ids = {module.module_id for module in self.morphology.modules}
        requested_module_ids = set(request.leader.module_ids) | set(request.follower.module_ids)
        if not requested_module_ids <= graph_module_ids:
            unknown = sorted(requested_module_ids - graph_module_ids)
            raise SchemaValidationError(
                f"AssemblyControlRequest references unknown module ids: {unknown}"
            )
        missing_models = sorted(requested_module_ids - set(self.physical_models_by_module_id))
        if missing_models:
            raise SchemaValidationError(
                f"AssemblyControlBridge is missing PhysicalModel entries for modules: {missing_models}"
            )
        leader_port = self._port(request.leader_port_id)
        follower_port = self._port(request.follower_port_id)
        if leader_port.module_id not in request.leader.module_ids:
            raise SchemaValidationError("leader_port_id is not part of the leader component")
        if follower_port.module_id not in request.follower.module_ids:
            raise SchemaValidationError("follower_port_id is not part of the follower component")
        self._validate_port_against_physical_model(leader_port)
        self._validate_port_against_physical_model(follower_port)
        self._validate_selected_edge(request)
        leader_port_spec = self._physical_port_spec(leader_port)
        follower_port_spec = self._physical_port_spec(follower_port)
        if (
            follower_port_spec.port_type not in leader_port_spec.compatible_port_types
            or leader_port_spec.port_type not in follower_port_spec.compatible_port_types
        ):
            raise SchemaValidationError("Selected assembly ports are not mutually compatible")
        self._validate_joint_corrections(
            request.leader,
            request.leader_joint_corrections_rad,
        )
        self._validate_joint_corrections(
            request.follower,
            request.follower_joint_corrections_rad,
        )

    def _validate_port_against_physical_model(self, port: PortNode) -> None:
        self._physical_port_spec(port)

    def _physical_port_spec(self, port: PortNode):
        model = self.physical_models_by_module_id[port.module_id]
        matches = [candidate for candidate in model.dock_ports if candidate.port_id == port.port_local_id]
        if len(matches) != 1:
            raise SchemaValidationError(
                f"PhysicalModel for module {port.module_id} does not uniquely define port "
                f"{port.port_local_id!r}"
            )
        return matches[0]

    def _validate_selected_edge(self, request: AssemblyControlRequest) -> None:
        selected = {request.leader_port_id, request.follower_port_id}
        matching_edges = [
            edge
            for edge in self.morphology.dock_edges
            if {edge.src_port_id, edge.dst_port_id} == selected
        ]
        if len(matching_edges) != 1:
            raise SchemaValidationError(
                "AssemblyControlRequest ports must identify exactly one MorphologyGraph DockEdge"
            )

    def _validate_joint_corrections(
        self,
        component: AssemblyComponentSpec,
        corrections: dict[str, float],
    ) -> None:
        allowed: dict[str, JointModel] = {}
        for module_id in component.module_ids:
            model = self.physical_models_by_module_id[module_id]
            joint_by_id = {joint.joint_id: joint for joint in model.joints}
            for local_joint_id in _dock_mechanism_joint_ids(model):
                allowed[f"module_{module_id}:{local_joint_id}"] = joint_by_id[local_joint_id]
        for global_joint_id, correction in corrections.items():
            if global_joint_id not in allowed:
                raise SchemaValidationError(
                    f"Joint correction {global_joint_id!r} is not a dock articulation joint "
                    f"of component {component.component_id!r}"
                )
            value = float(correction)
            if abs(value) > self.config.max_joint_correction_rad:
                raise SchemaValidationError(
                    f"Joint correction {global_joint_id!r} exceeds the canonical-q=0 correction bound"
                )
            joint = allowed[global_joint_id]
            if joint.limit_lower is not None and value < float(joint.limit_lower):
                raise SchemaValidationError(f"Joint correction {global_joint_id!r} is below its lower limit")
            if joint.limit_upper is not None and value > float(joint.limit_upper):
                raise SchemaValidationError(f"Joint correction {global_joint_id!r} is above its upper limit")

    def _require_components(
        self,
        request: AssemblyControlRequest,
        observation: AssemblyControlObservation,
    ) -> dict[str, AssemblyComponentObservation]:
        by_id = {component.component_id: component for component in observation.components}
        expected = {request.leader.component_id, request.follower.component_id}
        if set(by_id) != expected:
            raise SchemaValidationError(
                "AssemblyControlObservation must contain exactly the active leader and follower components"
            )
        for spec in (request.leader, request.follower):
            if by_id[spec.component_id].module_ids != spec.module_ids:
                raise SchemaValidationError(
                    f"Observed module membership changed for component {spec.component_id!r}"
                )
        return by_id

    def _port(self, port_id: int) -> PortNode:
        matches = [port for port in self.morphology.ports if port.port_global_id == port_id]
        if len(matches) != 1:
            raise SchemaValidationError(
                f"MorphologyGraph does not uniquely define port_global_id {port_id}"
            )
        return matches[0]

    def _require_session(self) -> AssemblyControlRequest:
        if self._request is None:
            raise RuntimeError("AssemblyControlBridge.begin must be called before tick")
        return self._request

    def _require_phase(self) -> AssemblyControlPhase:
        if self._phase is None:
            raise RuntimeError("AssemblyControlBridge has no active phase")
        return self._phase

    def _require_leader_hold_pose(self) -> Pose7D:
        if self._leader_hold_pose is None:
            raise RuntimeError("AssemblyControlBridge has no leader hold pose")
        return self._leader_hold_pose


def _dock_mechanism_joint_ids(model: PhysicalModel) -> list[str]:
    joint_by_id = {joint.joint_id: joint for joint in model.joints}
    joint_ids = sorted(
        {
            str(port.mechanical_limits["mechanism_joint_id"])
            for port in model.dock_ports
            if port.mechanical_limits.get("mechanism_joint_id")
        }
    )
    missing = [joint_id for joint_id in joint_ids if joint_id not in joint_by_id]
    if missing:
        raise SchemaValidationError(
            f"PhysicalModel dock ports reference unknown mechanism joints: {missing}"
        )
    return joint_ids


def _rotation_angle(rotation) -> float:
    cosine = max(-1.0, min(1.0, (rotation[0][0] + rotation[1][1] + rotation[2][2] - 1.0) * 0.5))
    return math.acos(cosine)


def _subtract3(left: Vector3, right: Vector3) -> Vector3:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _norm3(value: Vector3) -> float:
    return math.sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2])


def _require_finite_sequence(values, path: str) -> None:
    if not all(math.isfinite(float(value)) for value in values):
        raise SchemaValidationError(f"{path} must contain finite values")


def _require_nonzero_pose_quaternion(pose: Pose7D, path: str) -> None:
    norm_squared = sum(float(value) * float(value) for value in pose[3:7])
    if norm_squared <= 0.0:
        raise SchemaValidationError(f"{path} quaternion must have non-zero norm")


def _poses_equal(left: Pose7D, right: Pose7D, *, tolerance: float = 1.0e-12) -> bool:
    return all(abs(float(a) - float(b)) <= tolerance for a, b in zip(left, right, strict=True))


__all__ = [
    "ASSEMBLY_CONTROL_BRIDGE_CONTRACT_VERSION",
    "AssemblyAlignmentError",
    "AssemblyComponentCommandBundle",
    "AssemblyComponentObservation",
    "AssemblyComponentPolicyTarget",
    "AssemblyComponentSpec",
    "AssemblyConstraintIntent",
    "AssemblyControlBridge",
    "AssemblyControlBridgeConfig",
    "AssemblyControlObservation",
    "AssemblyControlRequest",
    "AssemblyControlStepOutput",
    "AssemblyControlStepProgress",
]
