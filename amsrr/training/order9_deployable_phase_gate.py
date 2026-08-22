from __future__ import annotations

"""Deployable Order 9 task-phase transitions.

This gate is deliberately separate from the privileged reward engine.  It may
use estimated poses/velocities, a Jacobian-based normal-force estimate derived
from motor-current-equivalent joint torque, and QPID status, but it never
consumes simulator contact force, penetration truth, collision labels, or
reward success.
"""

from dataclasses import dataclass

import torch

from amsrr.simulation.order9_object_task_runtime import (
    ORDER9_OBJECT_TASK_PHASES,
    Order9ObjectTaskPhase,
)


ORDER9_DEPLOYABLE_PHASE_GATE_VERSION = (
    "order9_kinematic_estimated_normal_force_phase_gate_v4_physical_release"
)


@dataclass(frozen=True)
class Order9DeployablePhaseGateConfig:
    required_anchor_count: int = 2
    approach_position_tolerance_m: float = 0.08
    approach_orientation_tolerance_rad: float = 0.20
    approach_linear_speed_tolerance_mps: float = 0.02
    object_position_tolerance_m: float = 0.052
    object_orientation_tolerance_rad: float = 0.20
    contact_surface_distance_tolerance_m: float = 0.004
    contact_surface_inward_limit_m: float = 0.006
    contact_relative_speed_tolerance_mps: float = 0.05
    contact_force_estimator_confidence_threshold: float = 0.25
    contact_dwell_s: float = 0.25
    release_surface_clearance_m: float = 0.005
    # Diagnostic/lineage compatibility only; physical release does not require
    # equality to one nominal IK posture.
    release_joint_position_tolerance_rad: float = 0.05
    retreat_position_tolerance_m: float = 0.05
    settle_linear_speed_mps: float = 0.05
    settle_angular_speed_radps: float = 0.10
    settle_dwell_s: float = 1.0

    def validate(self) -> None:
        if self.required_anchor_count < 1:
            raise ValueError("deployable phase gate anchor count must be positive")
        for name, value in self.__dict__.items():
            if name == "required_anchor_count":
                continue
            if not float(value) > 0.0:
                raise ValueError(f"deployable phase gate {name} must be positive")
        if self.contact_force_estimator_confidence_threshold > 1.0:
            raise ValueError(
                "deployable phase gate estimator confidence must not exceed one"
            )


@dataclass(frozen=True)
class Order9DeployablePhaseGateState:
    contact_dwell_s: torch.Tensor
    settle_dwell_s: torch.Tensor
    release_latched: torch.Tensor


@dataclass(frozen=True)
class Order9DeployablePhaseGateInput:
    phase_index: torch.Tensor
    robot_body_pose_world: torch.Tensor
    robot_body_twist_world: torch.Tensor
    object_pose_world: torch.Tensor
    object_twist_world: torch.Tensor
    desired_robot_pose_world: torch.Tensor
    desired_object_pose_world: torch.Tensor
    local_joint_positions_rad: torch.Tensor
    phase_goal_joint_positions_rad: torch.Tensor
    selected_surface_distance_m: torch.Tensor
    selected_relative_speed_mps: torch.Tensor
    selected_estimated_normal_force_n: torch.Tensor
    selected_required_normal_force_n: torch.Tensor
    selected_force_estimator_confidence: torch.Tensor
    selected_anchor_mask: torch.Tensor
    qp_feasible: torch.Tensor


@dataclass(frozen=True)
class Order9DeployablePhaseGateResult:
    phase_success: torch.Tensor
    grasp_ready: torch.Tensor
    released: torch.Tensor
    next_state: Order9DeployablePhaseGateState


class Order9DeployablePhaseGate:
    """Kinematic/proprioceptive task gate suitable for simulator and hardware."""

    version = ORDER9_DEPLOYABLE_PHASE_GATE_VERSION

    def __init__(
        self,
        *,
        config: Order9DeployablePhaseGateConfig | None = None,
        control_dt_s: float = 0.02,
    ) -> None:
        self.config = config or Order9DeployablePhaseGateConfig()
        self.config.validate()
        if control_dt_s <= 0.0:
            raise ValueError("deployable phase gate dt must be positive")
        self.control_dt_s = float(control_dt_s)

    def initial_state(
        self, *, batch_size: int, device: torch.device | str
    ) -> Order9DeployablePhaseGateState:
        if batch_size < 1:
            raise ValueError("deployable phase gate batch must be positive")
        zero = torch.zeros(batch_size, device=device, dtype=torch.float32)
        return Order9DeployablePhaseGateState(
            contact_dwell_s=zero.clone(),
            settle_dwell_s=zero.clone(),
            release_latched=torch.zeros(
                batch_size, device=device, dtype=torch.bool
            ),
        )

    def step(
        self,
        evidence: Order9DeployablePhaseGateInput,
        state: Order9DeployablePhaseGateState,
    ) -> Order9DeployablePhaseGateResult:
        self._validate(evidence, state)
        cfg = self.config
        phase = evidence.phase_index.long()
        selected = evidence.selected_anchor_mask
        geometric_contact = (
            (evidence.selected_surface_distance_m
             <= cfg.contact_surface_distance_tolerance_m)
            & (evidence.selected_surface_distance_m
               >= -cfg.contact_surface_inward_limit_m)
            & (evidence.selected_relative_speed_mps
               <= cfg.contact_relative_speed_tolerance_mps)
        )
        force_evidence = (
            evidence.selected_estimated_normal_force_n
            >= evidence.selected_required_normal_force_n
        ) & (
            evidence.selected_force_estimator_confidence
            >= cfg.contact_force_estimator_confidence_threshold
        )
        ready_by_anchor = selected & geometric_contact & force_evidence
        assigned_count = selected.sum(dim=-1)
        grasp_ready = (
            (assigned_count >= cfg.required_anchor_count)
            & ((ready_by_anchor | ~selected).all(dim=-1))
            & evidence.qp_feasible
        )
        contact_phase = phase == _phase(Order9ObjectTaskPhase.CONTACT_ACQUISITION)
        contact_dwell = torch.where(
            contact_phase & grasp_ready,
            state.contact_dwell_s + self.control_dt_s,
            torch.zeros_like(state.contact_dwell_s),
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
            object_position_error <= cfg.object_position_tolerance_m
        ) & (
            object_orientation_error <= cfg.object_orientation_tolerance_rad
        )
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
        released = (
            (~selected)
            | (evidence.selected_surface_distance_m
               >= cfg.release_surface_clearance_m)
        ).all(dim=-1)
        settled = (
            object_pose_ok
            & (object_linear_speed <= cfg.settle_linear_speed_mps)
            & (object_angular_speed <= cfg.settle_angular_speed_radps)
        )
        settle_phase = phase == _phase(Order9ObjectTaskPhase.SETTLE)
        settle_dwell = torch.where(
            settle_phase & settled,
            state.settle_dwell_s + self.control_dt_s,
            torch.zeros_like(state.settle_dwell_s),
        )

        approach = phase == _phase(Order9ObjectTaskPhase.APPROACH)
        contact = contact_phase
        lift = phase == _phase(Order9ObjectTaskPhase.LIFT)
        transport = phase == _phase(Order9ObjectTaskPhase.TRANSPORT)
        place = phase == _phase(Order9ObjectTaskPhase.PLACE)
        release = phase == _phase(Order9ObjectTaskPhase.RELEASE)
        retreat = phase == _phase(Order9ObjectTaskPhase.RETREAT)
        settle = settle_phase
        release_latched = state.release_latched | (release & released)
        success = (
            approach
            & (robot_position_error <= cfg.approach_position_tolerance_m)
            & (robot_orientation_error <= cfg.approach_orientation_tolerance_rad)
            & (robot_speed <= cfg.approach_linear_speed_tolerance_mps)
            & evidence.qp_feasible
        )
        success |= contact & (contact_dwell >= cfg.contact_dwell_s)
        success |= (lift | transport | place) & object_pose_ok & grasp_ready
        success |= (
            release
            & released
            & object_pose_ok
        )
        success |= (
            retreat
            & release_latched
            & (robot_position_error <= cfg.retreat_position_tolerance_m)
            & evidence.qp_feasible
        )
        success |= settle & (settle_dwell >= cfg.settle_dwell_s)
        return Order9DeployablePhaseGateResult(
            phase_success=success,
            grasp_ready=grasp_ready,
            released=released,
            next_state=Order9DeployablePhaseGateState(
                contact_dwell_s=contact_dwell,
                settle_dwell_s=settle_dwell,
                release_latched=release_latched,
            ),
        )

    @staticmethod
    def reset_state_subset(
        state: Order9DeployablePhaseGateState,
        env_ids: torch.Tensor,
        *,
        clear_release_latch: bool = True,
    ) -> Order9DeployablePhaseGateState:
        contact = state.contact_dwell_s.clone()
        settle = state.settle_dwell_s.clone()
        release_latched = state.release_latched.clone()
        contact[env_ids] = 0.0
        settle[env_ids] = 0.0
        if clear_release_latch:
            release_latched[env_ids] = False
        return Order9DeployablePhaseGateState(
            contact, settle, release_latched
        )

    @staticmethod
    def assume_release_complete_subset(
        state: Order9DeployablePhaseGateState, env_ids: torch.Tensor
    ) -> Order9DeployablePhaseGateState:
        """Restore predecessor completion for retreat/settle phase resets."""

        release_latched = state.release_latched.clone()
        release_latched[env_ids] = True
        return Order9DeployablePhaseGateState(
            state.contact_dwell_s,
            state.settle_dwell_s,
            release_latched,
        )

    @staticmethod
    def _validate(
        evidence: Order9DeployablePhaseGateInput,
        state: Order9DeployablePhaseGateState,
    ) -> None:
        batch = evidence.phase_index.shape[0]
        anchor_shape = evidence.selected_surface_distance_m.shape
        if evidence.phase_index.shape != (batch,) or len(anchor_shape) != 2:
            raise ValueError("deployable phase gate primary shape differs")
        for value in (
            evidence.selected_relative_speed_mps,
            evidence.selected_estimated_normal_force_n,
            evidence.selected_required_normal_force_n,
            evidence.selected_force_estimator_confidence,
            evidence.selected_anchor_mask,
        ):
            if value.shape != anchor_shape:
                raise ValueError("deployable phase gate anchor shape differs")
        expected = {
            "robot_body_pose_world": (batch, 7),
            "robot_body_twist_world": (batch, 6),
            "object_pose_world": (batch, 7),
            "object_twist_world": (batch, 6),
            "desired_robot_pose_world": (batch, 7),
            "desired_object_pose_world": (batch, 7),
            "qp_feasible": (batch,),
        }
        for name, shape in expected.items():
            if getattr(evidence, name).shape != shape:
                raise ValueError(f"deployable phase gate {name} shape differs")
        if (
            evidence.local_joint_positions_rad.ndim != 3
            or evidence.local_joint_positions_rad.shape
            != evidence.phase_goal_joint_positions_rad.shape
            or evidence.local_joint_positions_rad.shape[0] != batch
            or state.contact_dwell_s.shape != (batch,)
            or state.settle_dwell_s.shape != (batch,)
            or state.release_latched.shape != (batch,)
            or state.release_latched.dtype != torch.bool
        ):
            raise ValueError("deployable phase gate posture/state shape differs")
        if bool((evidence.selected_required_normal_force_n < 0.0).any()):
            raise ValueError("deployable phase gate required force must be non-negative")
        if bool(
            ((evidence.selected_force_estimator_confidence < 0.0)
             | (evidence.selected_force_estimator_confidence > 1.0)).any()
        ):
            raise ValueError("deployable phase gate estimator confidence is invalid")
        for value in evidence.__dict__.values():
            if isinstance(value, torch.Tensor) and value.dtype != torch.bool:
                if not bool(torch.isfinite(value).all()):
                    raise ValueError("deployable phase gate input is non-finite")


def _phase(value: Order9ObjectTaskPhase) -> int:
    return ORDER9_OBJECT_TASK_PHASES.index(value)


def _quaternion_distance(current: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
    current = current / torch.linalg.vector_norm(current, dim=-1, keepdim=True).clamp_min(1.0e-8)
    desired = desired / torch.linalg.vector_norm(desired, dim=-1, keepdim=True).clamp_min(1.0e-8)
    dot = (current * desired).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


__all__ = [
    "ORDER9_DEPLOYABLE_PHASE_GATE_VERSION",
    "Order9DeployablePhaseGate",
    "Order9DeployablePhaseGateConfig",
    "Order9DeployablePhaseGateInput",
    "Order9DeployablePhaseGateResult",
    "Order9DeployablePhaseGateState",
]
