from __future__ import annotations

import math

import torch

from amsrr.simulation.order9_object_task_runtime import (
    ORDER9_OBJECT_TASK_PHASES,
    Order9ObjectTaskPhase,
)
from amsrr.training.order9_tensor_reward import (
    ORDER9_OBJECT_POSITION_TOLERANCE_M,
    Order9TensorRewardEngine,
    Order9TensorRewardGateConfig,
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
    wrench_force_n: float = 1.0,
    contact_preload_complete: bool = True,
    joint_position_rad: float = 0.0,
    phase_goal_joint_position_rad: float = 0.0,
    object_position_error_m: float = 0.0,
    normal_error_m: float = 0.0,
    relative_normal_velocity_mps: float = 0.0,
    motor_load_proxy: float = 1.0,
    target_motor_load_proxy: float = 1.0,
) -> Order9TensorRewardInput:
    pose = torch.tensor([[0.0, 0.0, 0.30, 0.0, 0.0, 0.0, 1.0]])
    forces = torch.full((1, 2, 3), 0.0)
    if contact:
        forces[:, :, 2] = 1.0
    wrenches = torch.zeros((1, 2, 6))
    if contact:
        wrenches[:, 0, 2] = wrench_force_n
        wrenches[:, 1, 2] = -wrench_force_n
    lower = torch.full((1, 2, 6), -2.0)
    upper = torch.full((1, 2, 6), 2.0)
    lower[:, 0, 2] = 1.0
    upper[:, 0, 2] = 3.0
    lower[:, 1, 2] = -3.0
    upper[:, 1, 2] = -1.0
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
        desired_object_pose_world=pose.clone().index_put(
            (torch.tensor([0]), torch.tensor([0])),
            torch.tensor([object_position_error_m]),
        ),
        local_joint_positions_rad=torch.full(
            (1, 3, 4), joint_position_rad
        ),
        phase_goal_joint_positions_rad=torch.full(
            (1, 3, 4), phase_goal_joint_position_rad
        ),
        selected_contact_forces_world=forces,
        selected_contact_wrenches_contact=wrenches,
        wrench_lower_contact=lower,
        wrench_upper_contact=upper,
        wrench_bound_mask=torch.ones((1, 2), dtype=torch.bool),
        selected_link_twist_world=torch.zeros((1, 2, 6)),
        selected_contact_mask=torch.ones((1, 2), dtype=torch.bool),
        selected_grasp_frame_normal_error_m=torch.full(
            (1, 2), normal_error_m
        ),
        selected_relative_normal_velocity_mps=torch.full(
            (1, 2), relative_normal_velocity_mps
        ),
        selected_motor_load_proxy=torch.full((1, 2), motor_load_proxy),
        target_motor_load_proxy=torch.full(
            (1, 2), target_motor_load_proxy
        ),
        contact_preload_complete=torch.tensor([contact_preload_complete]),
        prohibited_collision=torch.tensor([collision]),
        support_top_z_m=torch.tensor([0.15]),
        object_half_height_m=torch.tensor([0.075]),
        qp_feasible=torch.tensor([qp_feasible]),
        allocation_residual_norm=torch.zeros((1,)),
        rotor_thrusts_n=torch.ones((1, 12)),
        rotor_saturation=torch.zeros((1, 12), dtype=torch.bool),
        joint_torque_bias_nm=torch.zeros((1, 3, 4)),
    )


def test_deployable_normal_contact_quality_penalizes_gap_and_low_load() -> None:
    engine = Order9TensorRewardEngine(control_dt_s=0.05)
    healthy = _evidence(Order9ObjectTaskPhase.CONTACT_ACQUISITION)
    weak = _evidence(
        Order9ObjectTaskPhase.CONTACT_ACQUISITION,
        normal_error_m=0.004,
        relative_normal_velocity_mps=0.02,
        motor_load_proxy=0.0,
    )
    healthy_state = engine.initial_state(
        object_pose_world=healthy.object_pose_world,
        desired_object_pose_world=healthy.desired_object_pose_world,
    )
    weak_state = engine.initial_state(
        object_pose_world=weak.object_pose_world,
        desired_object_pose_world=weak.desired_object_pose_world,
    )

    healthy_result = engine.step(healthy, healthy_state)
    weak_result = engine.step(weak, weak_state)

    assert (
        healthy_result.terms["weighted_normal_contact_quality_penalty"].item()
        == 0.0
    )
    assert (
        weak_result.terms["weighted_normal_contact_quality_penalty"].item()
        < 0.0
    )
    assert weak_result.reward.item() < healthy_result.reward.item()


def test_normal_contact_quality_is_inactive_during_approach() -> None:
    engine = Order9TensorRewardEngine(control_dt_s=0.05)
    evidence = _evidence(
        Order9ObjectTaskPhase.APPROACH,
        contact=False,
        normal_error_m=0.004,
        relative_normal_velocity_mps=0.02,
        motor_load_proxy=0.0,
    )
    state = engine.initial_state(
        object_pose_world=evidence.object_pose_world,
        desired_object_pose_world=evidence.desired_object_pose_world,
    )

    result = engine.step(evidence, state)

    assert (
        result.terms["weighted_normal_contact_quality_penalty"].item() == 0.0
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
    assert results[4].wrench_range_satisfied.item()
    assert results[4].reward.item() > results[3].reward.item()


def test_selected_three_contact_group_requires_third_contact():
    from dataclasses import fields, replace
    evidence=_evidence(Order9ObjectTaskPhase.CONTACT_ACQUISITION)
    expanded={}
    for field in fields(evidence):
        value=getattr(evidence,field.name)
        if isinstance(value,torch.Tensor) and value.ndim>=2 and value.shape[1]==2:
            expanded[field.name]=torch.cat((value,value[:,:1].clone()),dim=1)
    evidence=replace(evidence,**expanded)
    engine=Order9TensorRewardEngine(control_dt_s=.05,
        gate_config=Order9TensorRewardGateConfig(required_contact_count=3))
    state=engine.initial_state(object_pose_world=evidence.object_pose_world,
        desired_object_pose_world=evidence.desired_object_pose_world)
    evidence.selected_contact_forces_world[:,2]=0.
    for _ in range(6):
        result=engine.step(evidence,state);state=result.next_state
        assert result.active_contact_count.item()==2
        assert not result.phase_success.item()
    evidence.selected_contact_forces_world[:,2,2]=1.
    for _ in range(6):
        result=engine.step(evidence,state);state=result.next_state
    assert result.active_contact_count.item()==3
    assert result.phase_success.item()


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


def test_wrench_range_penalty_is_zero_inside_and_negative_outside() -> None:
    engine = Order9TensorRewardEngine(control_dt_s=0.05)
    inside = _evidence(
        Order9ObjectTaskPhase.CONTACT_ACQUISITION,
        wrench_force_n=1.0,
    )
    outside = _evidence(
        Order9ObjectTaskPhase.CONTACT_ACQUISITION,
        wrench_force_n=4.0,
    )
    inside_state = engine.initial_state(
        object_pose_world=inside.object_pose_world,
        desired_object_pose_world=inside.desired_object_pose_world,
    )
    outside_state = engine.initial_state(
        object_pose_world=outside.object_pose_world,
        desired_object_pose_world=outside.desired_object_pose_world,
    )

    inside_result = engine.step(inside, inside_state)
    outside_result = engine.step(outside, outside_state)

    assert inside_result.wrench_range_violation.item() == 0.0
    assert (
        inside_result.terms["weighted_wrench_range_violation_penalty"].item()
        == 0.0
    )
    assert outside_result.active_contact_count.item() == 2
    assert outside_result.wrench_range_violation.item() == 0.5
    assert (
        outside_result.terms["weighted_wrench_range_violation_penalty"].item()
        < 0.0
    )
    assert outside_result.reward.item() < inside_result.reward.item()


def test_wrench_range_penalty_retains_ordering_beyond_one_interval_width() -> None:
    engine = Order9TensorRewardEngine(control_dt_s=0.05)
    one_width_outside = _evidence(
        Order9ObjectTaskPhase.CONTACT_ACQUISITION,
        wrench_force_n=5.0,
    )
    two_widths_outside = _evidence(
        Order9ObjectTaskPhase.CONTACT_ACQUISITION,
        wrench_force_n=7.0,
    )
    one_state = engine.initial_state(
        object_pose_world=one_width_outside.object_pose_world,
        desired_object_pose_world=one_width_outside.desired_object_pose_world,
    )
    two_state = engine.initial_state(
        object_pose_world=two_widths_outside.object_pose_world,
        desired_object_pose_world=two_widths_outside.desired_object_pose_world,
    )

    one_result = engine.step(one_width_outside, one_state)
    two_result = engine.step(two_widths_outside, two_state)

    assert math.isclose(one_result.wrench_range_violation.item(), 1.0)
    assert math.isclose(
        two_result.wrench_range_violation.item(),
        1.0 + math.log(2.0),
        rel_tol=1.0e-6,
    )
    assert (
        two_result.terms["weighted_wrench_range_violation_penalty"].item()
        < one_result.terms["weighted_wrench_range_violation_penalty"].item()
    )
    assert not one_result.wrench_range_satisfied.item()
    assert not two_result.wrench_range_satisfied.item()


def test_contact_phase_allows_upper_overshoot_with_range_penalty() -> None:
    engine = Order9TensorRewardEngine(control_dt_s=0.05)
    evidence = _evidence(
        Order9ObjectTaskPhase.CONTACT_ACQUISITION,
        wrench_force_n=4.0,
    )
    state = engine.initial_state(
        object_pose_world=evidence.object_pose_world,
        desired_object_pose_world=evidence.desired_object_pose_world,
    )

    for _ in range(10):
        result = engine.step(evidence, state)
        state = result.next_state

    assert result.active_contact_count.item() == 2
    assert result.wrench_range_violation.item() == 0.5
    assert not result.wrench_range_satisfied.item()
    assert result.phase_success.item()
    assert result.next_state.grasp_acquired.item()


def test_range_diagnostic_scale_preserves_reward_and_not_phase_gate() -> None:
    evidence = _evidence(
        Order9ObjectTaskPhase.CONTACT_ACQUISITION,
        wrench_force_n=4.0,
    )
    exact = Order9TensorRewardEngine(control_dt_s=0.05)
    relaxed = Order9TensorRewardEngine(
        gate_config=Order9TensorRewardGateConfig(
            wrench_range_gate_scale=2.0
        ),
        control_dt_s=0.05,
    )
    exact_state = exact.initial_state(
        object_pose_world=evidence.object_pose_world,
        desired_object_pose_world=evidence.desired_object_pose_world,
    )
    relaxed_state = relaxed.initial_state(
        object_pose_world=evidence.object_pose_world,
        desired_object_pose_world=evidence.desired_object_pose_world,
    )

    for _ in range(5):
        exact_result = exact.step(evidence, exact_state)
        relaxed_result = relaxed.step(evidence, relaxed_state)
        exact_state = exact_result.next_state
        relaxed_state = relaxed_result.next_state

    assert exact_result.wrench_range_violation.item() == 0.5
    assert relaxed_result.wrench_range_violation.item() == 0.5
    assert (
        exact_result.terms["weighted_wrench_range_violation_penalty"].item()
        == relaxed_result.terms[
            "weighted_wrench_range_violation_penalty"
        ].item()
    )
    assert not exact_result.wrench_range_satisfied.item()
    assert exact_result.phase_success.item()
    assert relaxed_result.wrench_range_satisfied.item()
    assert relaxed_result.phase_success.item()


def test_contact_phase_ignores_retired_controller_preload_flag() -> None:
    engine = Order9TensorRewardEngine(control_dt_s=0.05)
    evidence = _evidence(
        Order9ObjectTaskPhase.CONTACT_ACQUISITION,
        wrench_force_n=1.0,
        contact_preload_complete=False,
    )
    state = engine.initial_state(
        object_pose_world=evidence.object_pose_world,
        desired_object_pose_world=evidence.desired_object_pose_world,
    )

    for _ in range(10):
        result = engine.step(evidence, state)
        state = result.next_state

    assert result.active_contact_count.item() == 2
    assert result.wrench_range_violation.item() == 0.0
    assert result.phase_success.item()
    assert result.next_state.grasp_acquired.item()


def test_contact_phase_requires_qp_feasible_during_wrench_dwell() -> None:
    engine = Order9TensorRewardEngine(control_dt_s=0.05)
    evidence = _evidence(
        Order9ObjectTaskPhase.CONTACT_ACQUISITION,
        qp_feasible=False,
    )
    state = engine.initial_state(
        object_pose_world=evidence.object_pose_world,
        desired_object_pose_world=evidence.desired_object_pose_world,
    )

    for _ in range(4):
        result = engine.step(evidence, state)
        state = result.next_state

    assert result.wrench_range_satisfied.item()
    assert not result.phase_success.item()
    assert not result.next_state.grasp_acquired.item()


def test_release_requires_contact_free_dwell_not_nominal_joint_posture() -> None:
    engine = Order9TensorRewardEngine(control_dt_s=0.05)
    incomplete = _evidence(
        Order9ObjectTaskPhase.RELEASE,
        contact=False,
        joint_position_rad=0.0,
        phase_goal_joint_position_rad=0.20,
    )
    state = engine.initial_state(
        object_pose_world=incomplete.object_pose_world,
        desired_object_pose_world=incomplete.desired_object_pose_world,
    )
    first = engine.step(incomplete, state)
    second = engine.step(incomplete, first.next_state)

    assert not first.release_valid.item()
    assert second.release_valid.item()
    assert second.phase_success.item()


def test_transport_uses_explicit_two_mm_numeric_pose_margin() -> None:
    assert ORDER9_OBJECT_POSITION_TOLERANCE_M == 0.052
    engine = Order9TensorRewardEngine(control_dt_s=0.05)
    inside = _evidence(
        Order9ObjectTaskPhase.TRANSPORT,
        object_position_error_m=0.0518,
    )
    outside = _evidence(
        Order9ObjectTaskPhase.TRANSPORT,
        object_position_error_m=0.0521,
    )
    inside_state = engine.initial_state(
        object_pose_world=inside.object_pose_world,
        desired_object_pose_world=inside.desired_object_pose_world,
    )
    outside_state = engine.initial_state(
        object_pose_world=outside.object_pose_world,
        desired_object_pose_world=outside.desired_object_pose_world,
    )

    assert engine.step(inside, inside_state).phase_success.item()
    assert not engine.step(outside, outside_state).phase_success.item()


def test_transport_penalizes_but_does_not_gate_wrench_range_membership() -> None:
    engine = Order9TensorRewardEngine(control_dt_s=0.05)
    outside = _evidence(
        Order9ObjectTaskPhase.TRANSPORT,
        wrench_force_n=4.0,
    )
    state = engine.initial_state(
        object_pose_world=outside.object_pose_world,
        desired_object_pose_world=outside.desired_object_pose_world,
    )

    result = engine.step(outside, state)

    assert result.active_contact_count.item() == 2
    assert not result.wrench_range_satisfied.item()
    assert result.phase_success.item()
    assert result.terms[
        "weighted_wrench_range_violation_penalty"
    ].item() < 0.0


def test_supported_place_only_exempts_contact_break_and_is_not_latched():
    from dataclasses import replace
    engine = Order9TensorRewardEngine(control_dt_s=.02)
    evidence = _evidence(Order9ObjectTaskPhase.PLACE, contact=False)
    state = engine.initial_state(object_pose_world=evidence.object_pose_world,
                                 desired_object_pose_world=evidence.desired_object_pose_world)
    state = replace(state, grasp_acquired=torch.tensor([True]), contact_break_s=torch.tensor([.06]))
    supported = replace(evidence, supported_placement=torch.tensor([True]))
    outcome = engine.step(supported, state)
    assert not outcome.object_dropped.item()
    assert not outcome.phase_success.item()  # not a fabricated grasp or task success
    assert engine.step(evidence, outcome.next_state).object_dropped.item()
    assert engine.step(replace(supported, phase_index=torch.tensor([3])), state).object_dropped.item()
    falling = supported.object_twist_world.clone(); falling[:, 2] = -.251
    assert engine.step(replace(supported, object_twist_world=falling), state).object_dropped.item()
    below = supported.object_pose_world.clone(); below[:, 2] = .15
    assert engine.step(replace(supported, object_pose_world=below), state).object_dropped.item()
    assert engine.step(replace(supported, prohibited_collision=torch.tensor([True])), state).hard_collision.item()
