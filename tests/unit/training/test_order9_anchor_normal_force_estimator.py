from __future__ import annotations

import torch

from amsrr.training.order9_anchor_normal_force_estimator import (
    Order9AnchorNormalForceEstimator,
    Order9AnchorNormalForceEstimatorConfig,
    order9_anchor_reaction_normal_world,
    order9_branch_free_motion_baseline_update_mask,
    order9_required_anchor_normal_force_n,
    shift_link_origin_linear_jacobian_to_point,
)


def _inputs(batch: int = 1):
    # Two independently actuated contact normals: tau = -J^T n f.
    jacobian = torch.zeros(batch, 2, 3, 4)
    jacobian[:, 0, 0, 0] = 0.10
    jacobian[:, 0, 0, 1] = 0.05
    jacobian[:, 1, 0, 2] = -0.08
    jacobian[:, 1, 0, 3] = -0.04
    normals = torch.zeros(batch, 2, 3)
    normals[..., 0] = 1.0
    owner = torch.tensor(
        [[True, True, False, False], [False, False, True, True]]
    )
    return jacobian, normals, owner


def test_estimator_removes_free_motion_baseline_and_recovers_normal_force() -> None:
    estimator = Order9AnchorNormalForceEstimator(
        Order9AnchorNormalForceEstimatorConfig(
            force_filter_alpha=1.0,
            ridge_damping_m2=1.0e-8,
            projected_iterations=64,
            fit_residual_scale_nm=1.0,
        )
    )
    state = estimator.initial_state(
        batch_size=1, anchor_count=2, joint_count=4, device="cpu"
    )
    jacobian, normals, owner = _inputs()
    baseline = torch.tensor([[0.20, -0.10, 0.15, -0.05]])
    gravity = torch.tensor([[0.05, 0.02, -0.03, 0.01]])
    free = estimator.step(
        applied_joint_torque_nm=gravity + baseline,
        gravity_joint_torque_nm=gravity,
        grasp_point_linear_jacobian_world=jacobian,
        contact_normal_world=normals,
        anchor_joint_owner_mask=owner,
        selected_anchor_mask=torch.ones(1, 2, dtype=torch.bool),
        baseline_update_mask=torch.ones(1, dtype=torch.bool),
        estimation_active_mask=torch.zeros(1, dtype=torch.bool),
        state=state,
    )
    # Contact force on the robot is +normal, hence actuator residual is
    # -J^T normal * force.
    force = torch.tensor([[3.0, 5.0]])
    normal_jacobian = -torch.einsum("bax,baxj->baj", normals, jacobian)
    contact_torque = torch.einsum("baj,ba->bj", normal_jacobian, force)
    estimate = estimator.step(
        applied_joint_torque_nm=gravity + baseline + contact_torque,
        gravity_joint_torque_nm=gravity,
        grasp_point_linear_jacobian_world=jacobian,
        contact_normal_world=normals,
        anchor_joint_owner_mask=owner,
        selected_anchor_mask=torch.ones(1, 2, dtype=torch.bool),
        baseline_update_mask=torch.zeros(1, dtype=torch.bool),
        estimation_active_mask=torch.ones(1, dtype=torch.bool),
        state=free.next_state,
    )
    torch.testing.assert_close(
        estimate.normal_force_n, force, atol=2.0e-3, rtol=2.0e-3
    )
    assert bool((estimate.confidence > 0.8).all())


def test_estimator_does_not_turn_internal_baseline_into_contact() -> None:
    estimator = Order9AnchorNormalForceEstimator(
        Order9AnchorNormalForceEstimatorConfig(force_filter_alpha=1.0)
    )
    state = estimator.initial_state(
        batch_size=1, anchor_count=2, joint_count=4, device="cpu"
    )
    jacobian, normals, owner = _inputs()
    load = torch.tensor([[0.7, -0.4, 0.5, -0.2]])
    initialized = estimator.step(
        applied_joint_torque_nm=load,
        gravity_joint_torque_nm=torch.zeros_like(load),
        grasp_point_linear_jacobian_world=jacobian,
        contact_normal_world=normals,
        anchor_joint_owner_mask=owner,
        selected_anchor_mask=torch.ones(1, 2, dtype=torch.bool),
        baseline_update_mask=torch.ones(1, dtype=torch.bool),
        estimation_active_mask=torch.zeros(1, dtype=torch.bool),
        state=state,
    )
    estimate = estimator.step(
        applied_joint_torque_nm=load,
        gravity_joint_torque_nm=torch.zeros_like(load),
        grasp_point_linear_jacobian_world=jacobian,
        contact_normal_world=normals,
        anchor_joint_owner_mask=owner,
        selected_anchor_mask=torch.ones(1, 2, dtype=torch.bool),
        baseline_update_mask=torch.zeros(1, dtype=torch.bool),
        estimation_active_mask=torch.ones(1, dtype=torch.bool),
        state=initialized.next_state,
    )
    torch.testing.assert_close(
        estimate.normal_force_n, torch.zeros(1, 2), atol=1.0e-7, rtol=0.0
    )


def test_branch_baseline_updates_only_clearly_free_anchor_joints() -> None:
    owner = torch.tensor(
        [[True, True, False, False], [False, False, True, True]]
    )
    mask = order9_branch_free_motion_baseline_update_mask(
        approach_mask=torch.tensor([False, False, True]),
        contact_acquisition_mask=torch.tensor([True, True, False]),
        selected_surface_distance_m=torch.tensor(
            [[0.010, 0.001], [0.001, 0.010], [0.0, 0.0]]
        ),
        selected_anchor_mask=torch.ones(3, 2, dtype=torch.bool),
        anchor_joint_owner_mask=owner,
        baseline_initialized=torch.tensor([True, True, False]),
        contact_clearance_m=0.004,
    )
    assert mask.tolist() == [
        [True, True, False, False],
        [False, False, True, True],
        [True, True, True, True],
    ]


def test_estimator_accepts_per_joint_baseline_update_mask() -> None:
    estimator = Order9AnchorNormalForceEstimator(
        Order9AnchorNormalForceEstimatorConfig(
            baseline_update_alpha=1.0,
            force_filter_alpha=1.0,
        )
    )
    state = estimator.initial_state(
        batch_size=1, anchor_count=2, joint_count=4, device="cpu"
    )
    jacobian, normals, owner = _inputs()
    initialized = estimator.step(
        applied_joint_torque_nm=torch.zeros(1, 4),
        gravity_joint_torque_nm=torch.zeros(1, 4),
        grasp_point_linear_jacobian_world=jacobian,
        contact_normal_world=normals,
        anchor_joint_owner_mask=owner,
        selected_anchor_mask=torch.ones(1, 2, dtype=torch.bool),
        baseline_update_mask=torch.ones(1, 4, dtype=torch.bool),
        estimation_active_mask=torch.zeros(1, dtype=torch.bool),
        state=state,
    )
    updated = estimator.step(
        applied_joint_torque_nm=torch.tensor([[0.2, 0.1, 0.8, 0.4]]),
        gravity_joint_torque_nm=torch.zeros(1, 4),
        grasp_point_linear_jacobian_world=jacobian,
        contact_normal_world=normals,
        anchor_joint_owner_mask=owner,
        selected_anchor_mask=torch.ones(1, 2, dtype=torch.bool),
        baseline_update_mask=torch.tensor([[True, True, False, False]]),
        estimation_active_mask=torch.ones(1, dtype=torch.bool),
        state=initialized.next_state,
    )
    torch.testing.assert_close(
        updated.next_state.free_motion_torque_baseline_nm,
        torch.tensor([[0.2, 0.1, 0.0, 0.0]]),
    )


def test_estimator_fit_confidence_ignores_unowned_joint_torque() -> None:
    estimator = Order9AnchorNormalForceEstimator(
        Order9AnchorNormalForceEstimatorConfig(
            force_filter_alpha=1.0,
            ridge_damping_m2=1.0e-8,
            projected_iterations=64,
            fit_residual_scale_nm=0.1,
        )
    )
    state = estimator.initial_state(
        batch_size=1, anchor_count=2, joint_count=5, device="cpu"
    )
    jacobian, normals, owner = _inputs()
    jacobian = torch.nn.functional.pad(jacobian, (0, 1))
    owner = torch.nn.functional.pad(owner, (0, 1), value=False)
    free = estimator.step(
        applied_joint_torque_nm=torch.zeros(1, 5),
        gravity_joint_torque_nm=torch.zeros(1, 5),
        grasp_point_linear_jacobian_world=jacobian,
        contact_normal_world=normals,
        anchor_joint_owner_mask=owner,
        selected_anchor_mask=torch.ones(1, 2, dtype=torch.bool),
        baseline_update_mask=torch.ones(1, dtype=torch.bool),
        estimation_active_mask=torch.zeros(1, dtype=torch.bool),
        state=state,
    )
    force = torch.tensor([[3.0, 5.0]])
    normal_jacobian = -torch.einsum("bax,baxj->baj", normals, jacobian)
    contact_torque = torch.einsum("baj,ba->bj", normal_jacobian, force)
    contact_torque[:, 4] = 50.0
    estimate = estimator.step(
        applied_joint_torque_nm=contact_torque,
        gravity_joint_torque_nm=torch.zeros(1, 5),
        grasp_point_linear_jacobian_world=jacobian,
        contact_normal_world=normals,
        anchor_joint_owner_mask=owner,
        selected_anchor_mask=torch.ones(1, 2, dtype=torch.bool),
        baseline_update_mask=torch.zeros(1, dtype=torch.bool),
        estimation_active_mask=torch.ones(1, dtype=torch.bool),
        state=free.next_state,
    )
    torch.testing.assert_close(
        estimate.normal_force_n, force, atol=2.0e-3, rtol=2.0e-3
    )
    assert float(estimate.fit_residual_nm.item()) < 1.0e-4
    assert bool((estimate.confidence > 0.8).all())


def test_estimator_coordinate_nnls_recovers_weak_moment_arm() -> None:
    estimator = Order9AnchorNormalForceEstimator(
        Order9AnchorNormalForceEstimatorConfig(
            force_filter_alpha=1.0,
            ridge_damping_m2=1.0e-10,
            projected_iterations=12,
            fit_residual_scale_nm=1.0,
        )
    )
    state = estimator.initial_state(
        batch_size=1, anchor_count=2, joint_count=2, device="cpu"
    )
    jacobian = torch.zeros(1, 2, 3, 2)
    jacobian[0, 0, 0, 0] = 0.0066
    jacobian[0, 1, 0, 1] = 0.409
    normals = torch.tensor([[[1.0, 0.0, 0.0]] * 2])
    expected = torch.tensor([[12.0, 13.0]])
    normal_jacobian = -torch.einsum("bax,baxj->baj", normals, jacobian)
    torque = torch.einsum("baj,ba->bj", normal_jacobian, expected)
    estimate = estimator.step(
        applied_joint_torque_nm=torque,
        gravity_joint_torque_nm=torch.zeros_like(torque),
        grasp_point_linear_jacobian_world=jacobian,
        contact_normal_world=normals,
        anchor_joint_owner_mask=torch.eye(2, dtype=torch.bool),
        selected_anchor_mask=torch.ones(1, 2, dtype=torch.bool),
        baseline_update_mask=torch.zeros(1, dtype=torch.bool),
        estimation_active_mask=torch.ones(1, dtype=torch.bool),
        state=state,
    )
    torch.testing.assert_close(
        estimate.normal_force_n, expected, atol=5.0e-4, rtol=5.0e-4
    )


def test_estimator_fails_closed_for_unobservable_anchor() -> None:
    estimator = Order9AnchorNormalForceEstimator(
        Order9AnchorNormalForceEstimatorConfig(force_filter_alpha=1.0)
    )
    state = estimator.initial_state(
        batch_size=1, anchor_count=2, joint_count=4, device="cpu"
    )
    estimate = estimator.step(
        applied_joint_torque_nm=torch.ones(1, 4),
        gravity_joint_torque_nm=torch.zeros(1, 4),
        grasp_point_linear_jacobian_world=torch.zeros(1, 2, 3, 4),
        contact_normal_world=torch.tensor([[[1.0, 0.0, 0.0]] * 2]),
        anchor_joint_owner_mask=torch.ones(2, 4, dtype=torch.bool),
        selected_anchor_mask=torch.ones(1, 2, dtype=torch.bool),
        baseline_update_mask=torch.zeros(1, dtype=torch.bool),
        estimation_active_mask=torch.ones(1, dtype=torch.bool),
        state=state,
    )
    assert estimate.normal_force_n.tolist() == [[0.0, 0.0]]
    assert estimate.confidence.tolist() == [[0.0, 0.0]]


def test_required_force_combines_payload_support_and_pi_h_interval() -> None:
    lower = torch.zeros(1, 2, 6)
    upper = torch.zeros_like(lower)
    lower[0, 0, 0], upper[0, 0, 0] = -12.0, 12.0
    lower[0, 1, 0], upper[0, 1, 0] = 4.0, 6.0
    required = order9_required_anchor_normal_force_n(
        wrench_lower_contact=lower,
        wrench_upper_contact=upper,
        contact_normal_contact=torch.tensor(
            [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
        ),
        wrench_bound_mask=torch.ones(1, 2, dtype=torch.bool),
        selected_anchor_mask=torch.ones(1, 2, dtype=torch.bool),
        estimated_payload_mass_kg=torch.ones(1),
        contact_friction=4.5,
    )
    torch.testing.assert_close(required[0, 0], torch.tensor(9.81 / 9.0))
    torch.testing.assert_close(required[0, 1], torch.tensor(4.0))


def test_reaction_normal_reverses_signed_pi_h_action_wrench() -> None:
    normal = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    lower = torch.zeros(1, 2, 6)
    upper = torch.zeros_like(lower)
    lower[0, 0, 0], upper[0, 0, 0] = 4.0, 6.0
    lower[0, 1, 1], upper[0, 1, 1] = -6.0, -4.0
    frame = torch.zeros(1, 2, 7)
    frame[..., 6] = 1.0
    reaction = order9_anchor_reaction_normal_world(
        contact_normal_world=normal,
        contact_frame_pose_world=frame,
        wrench_lower_contact=lower,
        wrench_upper_contact=upper,
    )
    torch.testing.assert_close(
        reaction, torch.tensor([[[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    )


def test_link_jacobian_is_shifted_to_grasp_frame_origin() -> None:
    jacobian = torch.zeros(1, 1, 6, 2)
    jacobian[0, 0, 3:, 0] = torch.tensor([0.0, 0.0, 1.0])
    jacobian[0, 0, :3, 1] = torch.tensor([1.0, 2.0, 3.0])
    shifted = shift_link_origin_linear_jacobian_to_point(
        body_link_jacobian_world=jacobian,
        body_position_world=torch.zeros(1, 1, 3),
        point_position_world=torch.tensor([[[1.0, 0.0, 0.0]]]),
        joint_columns=torch.tensor([0, 1]),
    )
    torch.testing.assert_close(
        shifted[0, 0, :, 0], torch.tensor([0.0, 1.0, 0.0])
    )
    torch.testing.assert_close(
        shifted[0, 0, :, 1], torch.tensor([1.0, 2.0, 3.0])
    )
