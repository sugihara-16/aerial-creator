from __future__ import annotations

"""Tensorized phase-aware reward and success gates for Order 9 Isaac PPO."""

from dataclasses import dataclass

import torch

from amsrr.simulation.order9_object_task_runtime import (
    ORDER9_OBJECT_TASK_PHASES,
    Order9ObjectTaskPhase,
)
from amsrr.training.p4_3_reward import P4_3RewardConfig


ORDER9_TENSOR_REWARD_TERM_NAMES = (
    "weighted_object_goal_progress",
    "weighted_object_pose_accuracy",
    "weighted_grasp_maintenance",
    "weighted_wrench_range_violation_penalty",
    "weighted_normal_contact_quality_penalty",
    "weighted_centroidal_stability",
    "weighted_energy_penalty",
    "weighted_qp_residual_penalty",
    "weighted_slip_penalty",
    "weighted_collision_penalty",
    "weighted_actuator_saturation_penalty",
    "terminal_success_bonus",
    "terminal_failure_penalty",
)
ORDER9_DEPLOYABLE_NORMAL_CONTACT_QUALITY_REWARD_CONTRACT = (
    "order9_fk_motor_load_normal_contact_quality_v1"
)
ORDER9_RELEASE_SUCCESS_CONTRACT = (
    "order9_release_success_contact_free_object_pose_v2"
)
ORDER9_OBJECT_POSE_SUCCESS_CONTRACT = (
    "order9_object_pose_success_50mm_plus_2mm_numeric_margin_v1"
)
ORDER9_CONTACT_ACQUISITION_SUCCESS_CONTRACT = (
    "order9_c3_privileged_two_physx_contacts_qp_dwell_v7"
)
ORDER9_C3_PRIVILEGED_PHASE_SUPERVISION_CONTRACT = (
    "order9_c3_privileged_physx_contact_outcome_phase_supervisor_v2"
)
ORDER9_RELEASE_JOINT_POSITION_TOLERANCE_RAD = 0.05
ORDER9_OBJECT_POSITION_TOLERANCE_M = 0.052


@dataclass(frozen=True)
class Order9TensorRewardGateConfig:
    required_contact_count: int = 2
    contact_force_threshold_n: float = 0.5
    contact_dwell_s: float = 0.25
    contact_break_grace_s: float = 0.05
    # Training-only continuation curriculum.  The reward penalty always uses
    # the exact teacher box; deterministic evaluation must set this to 1.0.
    wrench_range_gate_scale: float = 1.0
    release_contact_free_dwell_s: float = 0.10
    # Retained in artifact metadata for lineage compatibility; not a success
    # gate under ORDER9_RELEASE_SUCCESS_CONTRACT v2.
    release_joint_position_tolerance_rad: float = (
        ORDER9_RELEASE_JOINT_POSITION_TOLERANCE_RAD
    )
    approach_position_tolerance_m: float = 0.08
    approach_orientation_tolerance_rad: float = 0.20
    approach_linear_speed_tolerance_mps: float = 0.02
    # The task-level precision remains 50 mm.  The additional 2 mm is a
    # deterministic comparison margin for PhysX/contact-equilibrium variation;
    # without it, otherwise identical rollouts toggle at 49.997--51.8 mm.
    object_position_tolerance_m: float = (
        ORDER9_OBJECT_POSITION_TOLERANCE_M
    )
    object_orientation_tolerance_rad: float = 0.20
    lift_off_clearance_m: float = 0.001
    downward_drop_velocity_threshold_mps: float = 0.25
    retreat_position_tolerance_m: float = 0.05
    settle_linear_speed_mps: float = 0.05
    settle_angular_speed_radps: float = 0.10
    settle_dwell_s: float = 1.0
    qp_infeasible_grace_s: float = 0.10

    def validate(self) -> None:
        if self.required_contact_count < 1:
            raise ValueError("required_contact_count must be positive")
        for name, value in self.__dict__.items():
            if name == "required_contact_count":
                continue
            if not float(value) > 0.0:
                raise ValueError(f"{name} must be positive")
        if self.wrench_range_gate_scale < 1.0:
            raise ValueError("wrench_range_gate_scale must be at least one")


@dataclass(frozen=True)
class Order9TensorRewardState:
    contact_dwell_s: torch.Tensor
    contact_break_s: torch.Tensor
    contact_free_dwell_s: torch.Tensor
    settle_dwell_s: torch.Tensor
    qp_infeasible_dwell_s: torch.Tensor
    previous_object_goal_distance_m: torch.Tensor
    grasp_acquired: torch.Tensor


@dataclass(frozen=True)
class Order9TensorRewardInput:
    phase_index: torch.Tensor
    phase_elapsed_s: torch.Tensor
    phase_duration_s: torch.Tensor
    robot_body_pose_world: torch.Tensor
    robot_body_twist_world: torch.Tensor
    module_twist_world: torch.Tensor
    object_pose_world: torch.Tensor
    object_twist_world: torch.Tensor
    desired_robot_pose_world: torch.Tensor
    desired_object_pose_world: torch.Tensor
    local_joint_positions_rad: torch.Tensor
    phase_goal_joint_positions_rad: torch.Tensor
    selected_contact_forces_world: torch.Tensor
    selected_contact_wrenches_contact: torch.Tensor
    wrench_lower_contact: torch.Tensor
    wrench_upper_contact: torch.Tensor
    wrench_bound_mask: torch.Tensor
    selected_link_twist_world: torch.Tensor
    selected_contact_mask: torch.Tensor
    # Deployable evidence only: FK grasp-frame error, relative normal speed,
    # and motor-current-equivalent joint-load projection.  No PhysX contact
    # force or F/T sensor value enters this shaping term.
    selected_grasp_frame_normal_error_m: torch.Tensor
    selected_relative_normal_velocity_mps: torch.Tensor
    selected_motor_load_proxy: torch.Tensor
    target_motor_load_proxy: torch.Tensor
    contact_preload_complete: torch.Tensor
    prohibited_collision: torch.Tensor
    support_top_z_m: torch.Tensor
    object_half_height_m: torch.Tensor
    qp_feasible: torch.Tensor
    allocation_residual_norm: torch.Tensor
    rotor_thrusts_n: torch.Tensor
    rotor_saturation: torch.Tensor
    joint_torque_bias_nm: torch.Tensor
    # Privileged outcome evidence, never an actor/controller input. Omitted by
    # historical callers: the legacy grasp-loss semantics remain unchanged.
    supported_placement: torch.Tensor | None = None


@dataclass(frozen=True)
class Order9TensorRewardResult:
    reward: torch.Tensor
    phase_success: torch.Tensor
    terminal_failure: torch.Tensor
    timeout: torch.Tensor
    object_dropped: torch.Tensor
    hard_collision: torch.Tensor
    qp_infeasible_terminal: torch.Tensor
    release_valid: torch.Tensor
    active_contact_count: torch.Tensor
    wrench_range_satisfied: torch.Tensor
    slip_speed_mps: torch.Tensor
    wrench_range_violation: torch.Tensor
    terms: dict[str, torch.Tensor]
    next_state: Order9TensorRewardState


class Order9TensorRewardEngine:
    """Exact phase masking plus privileged reward/safety evidence."""

    def __init__(
        self,
        *,
        reward_config: P4_3RewardConfig | None = None,
        gate_config: Order9TensorRewardGateConfig | None = None,
        control_dt_s: float = 0.02,
    ) -> None:
        self.reward_config = reward_config or P4_3RewardConfig()
        self.gate_config = gate_config or Order9TensorRewardGateConfig()
        self.gate_config.validate()
        if control_dt_s <= 0.0:
            raise ValueError("Order9 tensor reward dt must be positive")
        self.control_dt_s = float(control_dt_s)
        self._use_cuda_graph = False
        self._cuda_call = None

    def initial_state(
        self,
        *,
        object_pose_world: torch.Tensor,
        desired_object_pose_world: torch.Tensor,
    ) -> Order9TensorRewardState:
        if object_pose_world.shape != desired_object_pose_world.shape or (
            object_pose_world.ndim != 2 or object_pose_world.shape[-1] != 7
        ):
            raise ValueError("Order9 tensor reward initial poses must be [B, 7]")
        distance = torch.linalg.vector_norm(
            object_pose_world[:, :3] - desired_object_pose_world[:, :3], dim=-1
        )
        zero = torch.zeros_like(distance)
        return Order9TensorRewardState(
            contact_dwell_s=zero.clone(),
            contact_break_s=zero.clone(),
            contact_free_dwell_s=zero.clone(),
            settle_dwell_s=zero.clone(),
            qp_infeasible_dwell_s=zero.clone(),
            previous_object_goal_distance_m=distance,
            grasp_acquired=torch.zeros_like(distance, dtype=torch.bool),
        )

    def step(
        self,
        evidence: Order9TensorRewardInput,
        state: Order9TensorRewardState,
    ) -> Order9TensorRewardResult:
        self._validate(evidence, state)
        if self._use_cuda_graph:
            from amsrr.utils.tensor_dataclass_graph import TensorDataclassGraph
            if self._cuda_call is None:
                self._cuda_call = TensorDataclassGraph()
            return self._cuda_call.call(self._compute, (evidence, state),
                configuration=(repr(self.reward_config), repr(self.gate_config), self.control_dt_s))
        return self._compute(evidence, state)

    def _compute(self, evidence, state):
        cfg = self.reward_config
        gate = self.gate_config
        dt = self.control_dt_s
        force_norm = torch.linalg.vector_norm(
            evidence.selected_contact_forces_world, dim=-1
        )
        active_contact = (
            force_norm >= gate.contact_force_threshold_n
        ) & evidence.selected_contact_mask
        active_count = active_contact.sum(dim=-1)
        enough_contact = active_count >= gate.required_contact_count
        wrench_bound = evidence.wrench_bound_mask
        lower_violation = torch.relu(
            evidence.wrench_lower_contact
            - evidence.selected_contact_wrenches_contact
        )
        upper_violation = torch.relu(
            evidence.selected_contact_wrenches_contact
            - evidence.wrench_upper_contact
        )
        # Normalize by the teacher-proposed interval itself.  This introduces
        # no task-specific force/torque threshold and does not prefer an
        # arbitrary pointwise decomposition while the realized wrench remains
        # in range.
        wrench_width = (
            evidence.wrench_upper_contact - evidence.wrench_lower_contact
        ).clamp_min(1.0e-6)
        normalized_axis_violation = (
            (lower_violation + upper_violation) / wrench_width
        )
        # A 6D wrench belongs to the proposed box only when every component is
        # inside.  L-infinity aggregation preserves that feasibility meaning;
        # averaging six axes would dilute a missing grasp-normal force.
        per_assignment_normalized_wrench_violation = (
            normalized_axis_violation.amax(dim=-1)
        )
        # Preserve the former linear penalty exactly through one interval
        # width, but retain ordering beyond it.  A hard clamp made 79--98% of
        # measured C3 contact samples indistinguishable once the wrench was
        # more than one interval width outside the teacher box.  The
        # logarithmic tail keeps severe PhysX impulses bounded in influence
        # without creating that reward plateau.
        per_assignment_wrench_penalty = torch.where(
            per_assignment_normalized_wrench_violation <= 1.0,
            per_assignment_normalized_wrench_violation,
            1.0
            + torch.log(
                per_assignment_normalized_wrench_violation.clamp_min(1.0)
            ),
        )
        bound_count = wrench_bound.sum(dim=-1)
        wrench_range_violation = (
            (
                per_assignment_wrench_penalty
                * wrench_bound.to(per_assignment_wrench_penalty.dtype)
            ).sum(dim=-1)
            / bound_count.clamp_min(1).to(per_assignment_wrench_penalty.dtype)
        )
        wrench_range_violation = torch.where(
            bound_count > 0,
            wrench_range_violation,
            torch.zeros_like(wrench_range_violation),
        )
        wrench_midpoint = 0.5 * (
            evidence.wrench_lower_contact + evidence.wrench_upper_contact
        )
        gate_half_width = (
            0.5
            * gate.wrench_range_gate_scale
            * (
                evidence.wrench_upper_contact
                - evidence.wrench_lower_contact
            )
        )
        gate_lower = wrench_midpoint - gate_half_width
        gate_upper = wrench_midpoint + gate_half_width
        gate_axis_satisfied = (
            evidence.selected_contact_wrenches_contact >= gate_lower
        ) & (
            evidence.selected_contact_wrenches_contact <= gate_upper
        )
        wrench_range_satisfied = (
            bound_count >= gate.required_contact_count
        ) & torch.all(
            torch.all(gate_axis_satisfied, dim=-1) | ~wrench_bound,
            dim=-1,
        )
        # C3 uses PhysX contact truth only as privileged training/evaluation
        # supervision.  Exact membership in the teacher's per-contact 6D
        # wrench box remains a dense reward/diagnostic signal, but is not a
        # grasp-success gate: stable grasp outcome and controller constraints
        # are the task authority.  Neither raw contact truth nor the retired
        # controller-side preload flag enters the actor/controller path.
        grasp_ready = (
            enough_contact
            & evidence.qp_feasible
        )
        contact_dwell = torch.where(
            grasp_ready,
            state.contact_dwell_s + dt,
            torch.zeros_like(state.contact_dwell_s),
        )
        grasp_acquired = state.grasp_acquired | (
            contact_dwell >= gate.contact_dwell_s
        )
        contact_break = torch.where(
            grasp_acquired & ~enough_contact,
            state.contact_break_s + dt,
            torch.zeros_like(state.contact_break_s),
        )
        no_contact = active_count == 0
        contact_free_dwell = torch.where(
            no_contact,
            state.contact_free_dwell_s + dt,
            torch.zeros_like(state.contact_free_dwell_s),
        )
        qp_dwell = torch.where(
            ~evidence.qp_feasible,
            state.qp_infeasible_dwell_s + dt,
            torch.zeros_like(state.qp_infeasible_dwell_s),
        )

        object_position_error = torch.linalg.vector_norm(
            evidence.object_pose_world[:, :3]
            - evidence.desired_object_pose_world[:, :3],
            dim=-1,
        )
        object_orientation_error = _quaternion_distance(
            evidence.object_pose_world[:, 3:7],
            evidence.desired_object_pose_world[:, 3:7],
        )
        object_pose_ok = (
            object_position_error <= gate.object_position_tolerance_m
        ) & (object_orientation_error <= gate.object_orientation_tolerance_rad)
        robot_position_error = torch.linalg.vector_norm(
            evidence.robot_body_pose_world[:, :3]
            - evidence.desired_robot_pose_world[:, :3],
            dim=-1,
        )
        robot_orientation_error = _quaternion_distance(
            evidence.robot_body_pose_world[:, 3:7],
            evidence.desired_robot_pose_world[:, 3:7],
        )
        robot_speed = torch.linalg.vector_norm(
            evidence.robot_body_twist_world[:, :3], dim=-1
        )
        object_linear_speed = torch.linalg.vector_norm(
            evidence.object_twist_world[:, :3], dim=-1
        )
        object_angular_speed = torch.linalg.vector_norm(
            evidence.object_twist_world[:, 3:6], dim=-1
        )
        settled = (
            object_pose_ok
            & (object_linear_speed <= gate.settle_linear_speed_mps)
            & (object_angular_speed <= gate.settle_angular_speed_radps)
        )
        settle_dwell = torch.where(
            settled,
            state.settle_dwell_s + dt,
            torch.zeros_like(state.settle_dwell_s),
        )
        release_valid = (
            contact_free_dwell >= gate.release_contact_free_dwell_s
        ) & object_pose_ok

        phase = evidence.phase_index.long()
        approach = phase == _phase_index(Order9ObjectTaskPhase.APPROACH)
        contact = phase == _phase_index(Order9ObjectTaskPhase.CONTACT_ACQUISITION)
        lift = phase == _phase_index(Order9ObjectTaskPhase.LIFT)
        transport = phase == _phase_index(Order9ObjectTaskPhase.TRANSPORT)
        place = phase == _phase_index(Order9ObjectTaskPhase.PLACE)
        release = phase == _phase_index(Order9ObjectTaskPhase.RELEASE)
        retreat = phase == _phase_index(Order9ObjectTaskPhase.RETREAT)
        settle = phase == _phase_index(Order9ObjectTaskPhase.SETTLE)
        support_clearance = (
            evidence.object_pose_world[:, 2]
            - evidence.object_half_height_m
            - evidence.support_top_z_m
        )
        phase_success = (
            approach
            & (robot_position_error <= gate.approach_position_tolerance_m)
            & (robot_orientation_error <= gate.approach_orientation_tolerance_rad)
            & (robot_speed <= gate.approach_linear_speed_tolerance_mps)
            & evidence.qp_feasible
        )
        phase_success |= contact & (contact_dwell >= gate.contact_dwell_s)
        phase_success |= (
            lift
            & object_pose_ok
            & (support_clearance >= gate.lift_off_clearance_m)
            & grasp_ready
        )
        phase_success |= (transport | place) & object_pose_ok & grasp_ready
        # Release is a physical task outcome: the payload is contact-free and
        # remains at its commanded pose.  Exact agreement with one nominal IK
        # posture is neither necessary for release nor morphology invariant;
        # joint tracking remains part of the controller/reward diagnostics.
        phase_success |= release & release_valid
        phase_success |= (
            retreat
            & (robot_position_error <= gate.retreat_position_tolerance_m)
            & no_contact
            & evidence.qp_feasible
        )
        phase_success |= settle & (settle_dwell >= gate.settle_dwell_s)

        maintain_phase = lift | transport | place
        supported_place = (torch.zeros_like(place) if evidence.supported_placement is None
                           else place & evidence.supported_placement)
        object_dropped = maintain_phase & (
            ((contact_break >= gate.contact_break_grace_s) & ~supported_place)
            | (
                evidence.object_twist_world[:, 2]
                < -gate.downward_drop_velocity_threshold_mps
            )
            | (support_clearance < -gate.object_position_tolerance_m)
        )
        hard_collision = evidence.prohibited_collision
        qp_terminal = qp_dwell >= gate.qp_infeasible_grace_s
        timeout = (evidence.phase_elapsed_s >= evidence.phase_duration_s) & (
            ~phase_success
        )
        terminal_failure = hard_collision | object_dropped | qp_terminal | timeout
        phase_success &= ~terminal_failure

        current_distance = object_position_error
        progress = (
            state.previous_object_goal_distance_m - current_distance
        ) / cfg.progress_scale_m
        progress = progress.clamp(min=-1.0, max=1.0)
        pose_accuracy = 0.5 * (
            1.0 / (1.0 + object_position_error / cfg.pose_position_scale_m)
            + 1.0
            / (1.0 + object_orientation_error / cfg.pose_rotation_scale_rad)
        )
        grasp_score = (
            active_count.to(evidence.object_pose_world.dtype)
            / float(gate.required_contact_count)
        ).clamp(max=1.0)
        mean_module_twist = evidence.module_twist_world.mean(dim=1, keepdim=True)
        deviation = evidence.module_twist_world - mean_module_twist
        linear_deviation = torch.sqrt(
            deviation[:, :, :3].square().sum(dim=-1).mean(dim=-1)
        )
        angular_deviation = torch.sqrt(
            deviation[:, :, 3:6].square().sum(dim=-1).mean(dim=-1)
        )
        stability = 1.0 / (
            1.0
            + linear_deviation / cfg.centroidal_linear_speed_scale_mps
            + angular_deviation / cfg.centroidal_angular_speed_scale_radps
        )
        rotor_energy = (
            evidence.rotor_thrusts_n / cfg.rotor_thrust_scale_n
        ).square().mean(dim=-1)
        joint_energy = (
            evidence.joint_torque_bias_nm / cfg.joint_torque_scale_nm
        ).square().mean(dim=(-1, -2))
        energy = (0.5 * (rotor_energy + joint_energy)).clamp(max=1.0)
        qp_penalty = (
            evidence.allocation_residual_norm.abs() / cfg.qp_residual_scale
        ).clamp(max=1.0)
        relative_linear_speed = torch.linalg.vector_norm(
            evidence.selected_link_twist_world[:, :, :3]
            - evidence.object_twist_world[:, None, :3],
            dim=-1,
        )
        slip = torch.where(
            active_contact,
            relative_linear_speed,
            torch.zeros_like(relative_linear_speed),
        ).max(dim=-1).values
        slip_penalty = (slip / cfg.slip_speed_scale_mps).clamp(max=1.0)
        # Dense, direction-specific contact credit.  The old contact-count
        # reward saturates as soon as both anchors touch and the generic slip
        # norm cannot say whether the normal, tangent, or rotation coordinate
        # caused the motion.  These quantities are available on hardware from
        # FK, object-state estimation, and motor current.  The target load is
        # resolved once from the same actuator/leverage model that generates
        # nominal preload.
        selected = evidence.selected_contact_mask
        selected_count = selected.sum(dim=-1).clamp_min(1).to(
            evidence.object_pose_world.dtype
        )
        gap_penalty = (
            torch.relu(evidence.selected_grasp_frame_normal_error_m)
            / cfg.normal_contact_gap_scale_m
        ).clamp(max=1.0)
        separating_penalty = (
            torch.relu(evidence.selected_relative_normal_velocity_mps)
            / cfg.normal_contact_separation_speed_scale_mps
        ).clamp(max=1.0)
        target_load = evidence.target_motor_load_proxy.clamp_min(1.0e-6)
        load_ratio = evidence.selected_motor_load_proxy / target_load
        load_target_available = evidence.target_motor_load_proxy > 1.0e-6
        load_deficit_penalty = torch.where(
            load_target_available,
            (
                torch.relu(cfg.normal_contact_load_lower_ratio - load_ratio)
                / cfg.normal_contact_load_lower_ratio
            ).clamp(max=1.0),
            torch.zeros_like(load_ratio),
        )
        load_excess_penalty = torch.where(
            load_target_available,
            (
                torch.relu(load_ratio - cfg.normal_contact_load_upper_ratio)
                / cfg.normal_contact_load_upper_ratio
            ).clamp(max=1.0),
            torch.zeros_like(load_ratio),
        )
        per_contact_normal_quality_penalty = (
            0.50 * gap_penalty
            + 0.35 * load_deficit_penalty
            + 0.10 * load_excess_penalty
            + 0.05 * separating_penalty
        )
        normal_contact_quality_penalty = (
            per_contact_normal_quality_penalty * selected.to(
                per_contact_normal_quality_penalty.dtype
            )
        ).sum(dim=-1) / selected_count
        saturation = evidence.rotor_saturation.to(
            evidence.object_pose_world.dtype
        ).mean(dim=-1)
        collision_penalty = hard_collision.to(evidence.object_pose_world.dtype)
        active_progress = lift | transport | place
        active_pose = transport | place | release | retreat | settle
        active_grasp = contact | lift | transport | place
        # Centroidal stability is task-independent and remains active in every
        # phase; keep the all-true mask explicit instead of a tautology.
        active_stability = torch.ones_like(settle)
        weighted_progress = cfg.w_progress * progress * active_progress
        weighted_pose = cfg.w_pose * pose_accuracy * active_pose
        weighted_grasp = cfg.w_grasp * grasp_score * active_grasp
        weighted_wrench_range = (
            -cfg.w_wrench_range * wrench_range_violation * active_grasp
        )
        weighted_normal_contact_quality = (
            -cfg.w_normal_contact_quality
            * normal_contact_quality_penalty
            * active_grasp
        )
        weighted_stability = cfg.w_stable * stability * active_stability
        weighted_energy = -cfg.w_energy * energy
        weighted_qp = -cfg.w_qp * qp_penalty
        weighted_slip = -cfg.w_slip * slip_penalty
        weighted_collision = -cfg.w_collision * collision_penalty
        weighted_saturation = -cfg.w_saturation * saturation
        reward = (
            weighted_progress
            + weighted_pose
            + weighted_grasp
            + weighted_wrench_range
            + weighted_normal_contact_quality
            + weighted_stability
            + weighted_energy
            + weighted_qp
            + weighted_slip
            + weighted_collision
            + weighted_saturation
            + cfg.success_bonus * phase_success
            - cfg.failure_penalty * terminal_failure
        )
        terms = {
            "weighted_object_goal_progress": weighted_progress,
            "weighted_object_pose_accuracy": weighted_pose,
            "weighted_grasp_maintenance": weighted_grasp,
            "weighted_wrench_range_violation_penalty": weighted_wrench_range,
            "weighted_normal_contact_quality_penalty": (
                weighted_normal_contact_quality
            ),
            "weighted_centroidal_stability": weighted_stability,
            "weighted_energy_penalty": weighted_energy,
            "weighted_qp_residual_penalty": weighted_qp,
            "weighted_slip_penalty": weighted_slip,
            "weighted_collision_penalty": weighted_collision,
            "weighted_actuator_saturation_penalty": weighted_saturation,
            "terminal_success_bonus": cfg.success_bonus * phase_success,
            "terminal_failure_penalty": -cfg.failure_penalty * terminal_failure,
        }
        return Order9TensorRewardResult(
            reward=reward,
            phase_success=phase_success,
            terminal_failure=terminal_failure,
            timeout=timeout,
            object_dropped=object_dropped,
            hard_collision=hard_collision,
            qp_infeasible_terminal=qp_terminal,
            release_valid=release_valid,
            active_contact_count=active_count,
            wrench_range_satisfied=wrench_range_satisfied,
            slip_speed_mps=slip,
            wrench_range_violation=wrench_range_violation,
            terms=terms,
            next_state=Order9TensorRewardState(
                contact_dwell_s=contact_dwell,
                contact_break_s=contact_break,
                contact_free_dwell_s=contact_free_dwell,
                settle_dwell_s=settle_dwell,
                qp_infeasible_dwell_s=qp_dwell,
                previous_object_goal_distance_m=current_distance,
                grasp_acquired=grasp_acquired,
            ),
        )

    @staticmethod
    def reset_state_subset(
        state: Order9TensorRewardState,
        env_ids: torch.Tensor,
        *,
        object_pose_world: torch.Tensor,
        desired_object_pose_world: torch.Tensor,
    ) -> Order9TensorRewardState:
        values = {
            name: value.clone() for name, value in state.__dict__.items()
        }
        for name, value in values.items():
            if name == "previous_object_goal_distance_m":
                value[env_ids] = torch.linalg.vector_norm(
                    object_pose_world[env_ids, :3]
                    - desired_object_pose_world[env_ids, :3],
                    dim=-1,
                )
            elif value.dtype == torch.bool:
                value[env_ids] = False
            else:
                value[env_ids] = 0.0
        return Order9TensorRewardState(**values)

    @staticmethod
    def _validate(
        evidence: Order9TensorRewardInput, state: Order9TensorRewardState
    ) -> None:
        batch = evidence.phase_index.shape[0]
        if evidence.supported_placement is not None and (
            evidence.supported_placement.shape != (batch,)
            or evidence.supported_placement.dtype != torch.bool
        ):
            raise ValueError("supported placement evidence must be a batch boolean")
        if evidence.phase_index.shape != (batch,):
            raise ValueError("Order9 tensor reward phase shape differs")
        if evidence.object_pose_world.shape != (batch, 7):
            raise ValueError("Order9 tensor reward object pose shape differs")
        if (
            evidence.local_joint_positions_rad.ndim != 3
            or evidence.local_joint_positions_rad.shape
            != evidence.phase_goal_joint_positions_rad.shape
            or evidence.local_joint_positions_rad.shape[0] != batch
        ):
            raise ValueError("Order9 release joint posture shape differs")
        finite_values = [evidence.local_joint_positions_rad, evidence.phase_goal_joint_positions_rad]
        finite_messages = ["Order9 release joint posture is non-finite"] * 2
        if evidence.selected_contact_forces_world.ndim != 3 or (
            evidence.selected_contact_forces_world.shape[0] != batch
            or evidence.selected_contact_forces_world.shape[-1] != 3
        ):
            raise ValueError("Order9 selected contact force shape differs")
        selected_shape = evidence.selected_contact_forces_world.shape[:2]
        if evidence.selected_contact_mask.shape != selected_shape or (
            evidence.selected_link_twist_world.shape != (*selected_shape, 6)
        ):
            raise ValueError("Order9 selected contact evidence shape differs")
        for name in (
            "selected_grasp_frame_normal_error_m",
            "selected_relative_normal_velocity_mps",
            "selected_motor_load_proxy",
            "target_motor_load_proxy",
        ):
            value = getattr(evidence, name)
            if value.shape != selected_shape:
                raise ValueError(f"Order9 {name} shape or finiteness differs")
            finite_values.append(value)
            finite_messages.append(f"Order9 {name} shape or finiteness differs")
        if evidence.contact_preload_complete.shape != (batch,):
            raise ValueError("Order9 contact preload completion shape differs")
        wrench_shape = (*selected_shape, 6)
        if (
            evidence.selected_contact_wrenches_contact.shape != wrench_shape
            or evidence.wrench_lower_contact.shape != wrench_shape
            or evidence.wrench_upper_contact.shape != wrench_shape
            or evidence.wrench_bound_mask.shape != selected_shape
        ):
            raise ValueError("Order9 selected contact-wrench range shape differs")
        finite_values.append(evidence.selected_contact_wrenches_contact)
        finite_messages.append("Order9 selected contact wrench is non-finite")
        # Masked reductions keep unused NaN bounds legal without dynamic-size
        # advanced indexing (which synchronizes CUDA to discover its length).
        active = evidence.wrench_bound_mask.unsqueeze(-1)
        device = evidence.phase_index.device
        finite_flat = torch.cat([value.reshape(-1).to(device=device) for value in finite_values])
        invalid = [~torch.isfinite(finite_flat).all()]
        invalid += [((~torch.isfinite(evidence.wrench_lower_contact)
                      | ~torch.isfinite(evidence.wrench_upper_contact)) & active).any().to(device=device),
                    ((evidence.wrench_lower_contact > evidence.wrench_upper_contact) & active).any().to(device=device)]
        flags = torch.stack(invalid).tolist()
        if flags[0]:
            # Preserve the specific error without many tiny GPU reductions on
            # the usual all-finite path.
            for value, message in zip(finite_values, finite_messages):
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(message)
        messages = ["Order9 contact-wrench bound is non-finite",
                    "Order9 contact-wrench lower bound exceeds upper"]
        for failed, message in zip(flags[1:], messages):
            if failed:
                raise ValueError(message)
        if state.contact_dwell_s.shape != (batch,):
            raise ValueError("Order9 tensor reward state batch differs")


def _phase_index(phase: Order9ObjectTaskPhase) -> int:
    return ORDER9_OBJECT_TASK_PHASES.index(phase)


def _quaternion_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left / left.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    right = right / right.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    dot = (left * right).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


__all__ = [
    "ORDER9_DEPLOYABLE_NORMAL_CONTACT_QUALITY_REWARD_CONTRACT",
    "Order9TensorRewardEngine",
    "Order9TensorRewardGateConfig",
    "Order9TensorRewardInput",
    "Order9TensorRewardResult",
    "Order9TensorRewardState",
]
