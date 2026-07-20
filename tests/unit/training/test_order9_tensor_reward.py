from __future__ import annotations

import torch

from amsrr.simulation.order9_object_task_runtime import (
    ORDER9_OBJECT_TASK_PHASES,
    Order9ObjectTaskPhase,
)
from amsrr.training.order9_tensor_reward import (
    Order9TensorRewardEngine,
    Order9TensorRewardInput,
)


def _phase(phase: Order9ObjectTaskPhase) -> int:
    return ORDER9_OBJECT_TASK_PHASES.index(phase)


def _evidence(
    phase: Order9ObjectTaskPhase,
    *,
    contact: bool = True,
    collision: bool = False,
    qp_feasible: bool = True,
) -> Order9TensorRewardInput:
    pose = torch.tensor([[0.0, 0.0, 0.30, 0.0, 0.0, 0.0, 1.0]])
    forces = torch.full((1, 2, 3), 0.0)
    if contact:
        forces[:, :, 2] = 1.0
    return Order9TensorRewardInput(
        phase_index=torch.tensor([_phase(phase)]),
        phase_elapsed_s=torch.tensor([0.2]),
        phase_duration_s=torch.tensor([2.0]),
        robot_body_pose_world=pose.clone(),
        robot_body_twist_world=torch.zeros((1, 6)),
        module_twist_world=torch.zeros((1, 3, 6)),
        object_pose_world=pose.clone(),
        object_twist_world=torch.zeros((1, 6)),
        desired_robot_pose_world=pose.clone(),
        desired_object_pose_world=pose.clone(),
        selected_contact_forces_world=forces,
        selected_link_twist_world=torch.zeros((1, 2, 6)),
        selected_contact_mask=torch.ones((1, 2), dtype=torch.bool),
        prohibited_collision=torch.tensor([collision]),
        support_top_z_m=torch.tensor([0.15]),
        object_half_height_m=torch.tensor([0.075]),
        qp_feasible=torch.tensor([qp_feasible]),
        allocation_residual_norm=torch.zeros((1,)),
        rotor_thrusts_n=torch.ones((1, 12)),
        rotor_saturation=torch.zeros((1, 12), dtype=torch.bool),
        joint_torque_bias_nm=torch.zeros((1, 3, 4)),
    )


def test_contact_phase_requires_order8_contact_dwell() -> None:
    engine = Order9TensorRewardEngine(control_dt_s=0.05)
    evidence = _evidence(Order9ObjectTaskPhase.CONTACT_ACQUISITION)
    state = engine.initial_state(
        object_pose_world=evidence.object_pose_world,
        desired_object_pose_world=evidence.desired_object_pose_world,
    )
    results = []
    for _ in range(5):
        result = engine.step(evidence, state)
        results.append(result)
        state = result.next_state

    assert not results[3].phase_success.item()
    assert results[4].phase_success.item()
    assert results[4].active_contact_count.item() == 2
    assert results[4].reward.item() > results[3].reward.item()


def test_grasp_loss_after_acquisition_is_terminal_drop() -> None:
    engine = Order9TensorRewardEngine(control_dt_s=0.05)
    contact = _evidence(Order9ObjectTaskPhase.CONTACT_ACQUISITION)
    state = engine.initial_state(
        object_pose_world=contact.object_pose_world,
        desired_object_pose_world=contact.desired_object_pose_world,
    )
    for _ in range(5):
        state = engine.step(contact, state).next_state
    lost = _evidence(Order9ObjectTaskPhase.TRANSPORT, contact=False)

    result = engine.step(lost, state)

    assert result.object_dropped.item()
    assert result.terminal_failure.item()
    assert result.reward.item() < 0.0


def test_hard_collision_dominates_simultaneous_phase_success() -> None:
    engine = Order9TensorRewardEngine(control_dt_s=0.05)
    evidence = _evidence(
        Order9ObjectTaskPhase.APPROACH, collision=True, contact=False
    )
    state = engine.initial_state(
        object_pose_world=evidence.object_pose_world,
        desired_object_pose_world=evidence.desired_object_pose_world,
    )

    result = engine.step(evidence, state)

    assert result.hard_collision.item()
    assert result.terminal_failure.item()
    assert not result.phase_success.item()
    assert result.terms["weighted_object_goal_progress"].item() == 0.0
