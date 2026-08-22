from __future__ import annotations

import torch

from amsrr.simulation.order9_object_task_runtime import (
    ORDER9_OBJECT_TASK_PHASES,
    Order9ObjectTaskPhase,
)
from amsrr.training.order9_deployable_phase_gate import (
    Order9DeployablePhaseGate,
    Order9DeployablePhaseGateInput,
)


def _evidence(phase: Order9ObjectTaskPhase) -> Order9DeployablePhaseGateInput:
    pose = torch.tensor([[0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]])
    return Order9DeployablePhaseGateInput(
        phase_index=torch.tensor([ORDER9_OBJECT_TASK_PHASES.index(phase)]),
        robot_body_pose_world=pose.clone(),
        robot_body_twist_world=torch.zeros(1, 6),
        object_pose_world=pose.clone(),
        object_twist_world=torch.zeros(1, 6),
        desired_robot_pose_world=pose.clone(),
        desired_object_pose_world=pose.clone(),
        local_joint_positions_rad=torch.zeros(1, 2, 4),
        phase_goal_joint_positions_rad=torch.zeros(1, 2, 4),
        selected_surface_distance_m=torch.full((1, 2), 0.002),
        selected_relative_speed_mps=torch.zeros(1, 2),
        selected_estimated_normal_force_n=torch.full((1, 2), 2.0),
        selected_required_normal_force_n=torch.full((1, 2), 1.0),
        selected_force_estimator_confidence=torch.full((1, 2), 0.9),
        selected_anchor_mask=torch.ones(1, 2, dtype=torch.bool),
        qp_feasible=torch.ones(1, dtype=torch.bool),
    )


def test_contact_gate_uses_deployable_dwell() -> None:
    gate = Order9DeployablePhaseGate(control_dt_s=0.05)
    state = gate.initial_state(batch_size=1, device="cpu")
    result = None
    for _ in range(5):
        result = gate.step(
            _evidence(Order9ObjectTaskPhase.CONTACT_ACQUISITION), state
        )
        state = result.next_state
    assert result is not None
    assert result.grasp_ready.item()
    assert result.phase_success.item()


def test_contact_gate_rejects_proximity_without_estimated_force() -> None:
    gate = Order9DeployablePhaseGate(control_dt_s=0.25)
    state = gate.initial_state(batch_size=1, device="cpu")
    evidence = _evidence(Order9ObjectTaskPhase.CONTACT_ACQUISITION)
    evidence = Order9DeployablePhaseGateInput(
        **{
            **evidence.__dict__,
            "selected_estimated_normal_force_n": torch.zeros(1, 2),
        }
    )
    result = gate.step(evidence, state)
    assert not result.grasp_ready.item()
    assert not result.phase_success.item()


def test_contact_gate_rejects_force_below_physical_requirement() -> None:
    gate = Order9DeployablePhaseGate(control_dt_s=0.25)
    state = gate.initial_state(batch_size=1, device="cpu")
    evidence = _evidence(Order9ObjectTaskPhase.CONTACT_ACQUISITION)
    evidence = Order9DeployablePhaseGateInput(
        **{
            **evidence.__dict__,
            "selected_estimated_normal_force_n": torch.full((1, 2), 0.9),
            "selected_required_normal_force_n": torch.full((1, 2), 1.0),
        }
    )
    result = gate.step(evidence, state)
    assert not result.grasp_ready.item()


def test_contact_gate_rejects_low_confidence_force_estimate() -> None:
    gate = Order9DeployablePhaseGate(control_dt_s=0.25)
    state = gate.initial_state(batch_size=1, device="cpu")
    evidence = _evidence(Order9ObjectTaskPhase.CONTACT_ACQUISITION)
    evidence = Order9DeployablePhaseGateInput(
        **{
            **evidence.__dict__,
            "selected_force_estimator_confidence": torch.full((1, 2), 0.1),
        }
    )
    result = gate.step(evidence, state)
    assert not result.grasp_ready.item()


def test_release_gate_uses_geometry_not_contact_force() -> None:
    gate = Order9DeployablePhaseGate(control_dt_s=0.02)
    state = gate.initial_state(batch_size=1, device="cpu")
    evidence = _evidence(Order9ObjectTaskPhase.RELEASE)
    evidence = Order9DeployablePhaseGateInput(
        **{
            **evidence.__dict__,
            "selected_surface_distance_m": torch.full((1, 2), 0.01),
            "local_joint_positions_rad": torch.full((1, 2, 4), -0.5),
            "phase_goal_joint_positions_rad": torch.full((1, 2, 4), 0.5),
        }
    )
    result = gate.step(evidence, state)
    assert result.released.item()
    assert result.phase_success.item()
    assert result.next_state.release_latched.item()


def test_retreat_uses_latched_release_completion_not_contact_half_space() -> None:
    gate = Order9DeployablePhaseGate(control_dt_s=0.02)
    state = gate.initial_state(batch_size=1, device="cpu")
    released = _evidence(Order9ObjectTaskPhase.RELEASE)
    released = Order9DeployablePhaseGateInput(
        **{
            **released.__dict__,
            "selected_surface_distance_m": torch.full((1, 2), 0.01),
        }
    )
    release_result = gate.step(released, state)
    transitioned = gate.reset_state_subset(
        release_result.next_state,
        torch.tensor([0]),
        clear_release_latch=False,
    )
    retreat = _evidence(Order9ObjectTaskPhase.RETREAT)
    retreat = Order9DeployablePhaseGateInput(
        **{
            **retreat.__dict__,
            # A collision-free retreat may cross the original contact-plane
            # half space after geometric release has already completed.
            "selected_surface_distance_m": torch.full((1, 2), -0.50),
        }
    )

    result = gate.step(retreat, transitioned)

    assert not result.released.item()
    assert result.phase_success.item()
    assert result.next_state.release_latched.item()


def test_retreat_phase_reset_restores_release_completion_contract() -> None:
    gate = Order9DeployablePhaseGate(control_dt_s=0.02)
    state = gate.initial_state(batch_size=2, device="cpu")
    state = gate.assume_release_complete_subset(state, torch.tensor([1]))

    assert state.release_latched.tolist() == [False, True]
    cleared = gate.reset_state_subset(state, torch.tensor([1]))
    assert cleared.release_latched.tolist() == [False, False]
