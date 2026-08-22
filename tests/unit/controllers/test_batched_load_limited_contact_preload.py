from __future__ import annotations

import torch

from amsrr.controllers.batched_load_limited_contact_preload import (
    BatchedLoadLimitedContactPreload,
)
from amsrr.policies.order9_tensor_command_decoder import (
    Order9TensorPolicyCommand,
)


def _command(batch: int = 2) -> Order9TensorPolicyCommand:
    return Order9TensorPolicyCommand(
        desired_body_pose_world=torch.zeros((batch, 7)),
        desired_body_twist=torch.zeros((batch, 6)),
        residual_wrench_body=torch.zeros((batch, 6)),
        joint_position_targets_rad=torch.tensor(
            [[[0.10, -0.10]], [[0.20, -0.20]]]
        ),
        joint_velocity_targets_radps=torch.ones((batch, 1, 2)),
        joint_torque_bias_nm=torch.ones((batch, 1, 2)),
        joint_target_mask=torch.ones((batch, 1, 2), dtype=torch.bool),
        module_ids=(0,),
        local_joint_ids=("left", "right"),
        contact_compression_legacy_joint_action=torch.zeros((batch,)),
        contact_compression_residual_action=torch.zeros((batch,)),
        contact_compression_action=torch.zeros((batch,)),
        contact_compression_joint_delta_rad=torch.zeros((batch, 1, 2)),
    )


def _preload() -> BatchedLoadLimitedContactPreload:
    return BatchedLoadLimitedContactPreload(
        batch_size=2,
        closure_direction_rad=torch.tensor([[1.0, -1.0]]),
        joint_owner_mask_by_anchor=torch.tensor(
            [[[True, False]], [[False, True]]]
        ),
        device="cpu",
    )


def _step(
    preload: BatchedLoadLimitedContactPreload,
    *,
    phase: tuple[int, int] = (1, 1),
    force_n: float = 1.0,
    signed_normal_force_n: float = 0.0,
    required_normal_force_n: float = 4.0,
    phase_progress: tuple[float, float] = (0.5, 0.5),
    loads: tuple[tuple[float, float], tuple[float, float]] = (
        (0.2, 0.2),
        (0.2, 0.2),
    ),
) -> Order9TensorPolicyCommand:
    forces = torch.zeros((2, 2, 3))
    forces[:, :, 2] = force_n
    return preload.apply(
        policy_command=_command(),
        runtime_phase_index=torch.tensor(phase),
        phase_progress=torch.tensor(phase_progress),
        selected_contact_forces_world=forces,
        selected_contact_mask=torch.ones((2, 2), dtype=torch.bool),
        selected_signed_normal_force_n=torch.full(
            (2, 2), signed_normal_force_n
        ),
        required_signed_normal_force_n=torch.full(
            (2, 2), required_normal_force_n
        ),
        damping_compensated_joint_load_nm=torch.tensor(loads).reshape(2, 1, 2),
        dt_s=0.02,
        contact_phase_index=1,
        lift_phase_index=2,
        transport_phase_index=3,
        place_phase_index=4,
        release_phase_index=5,
    )


def test_batched_preload_arms_after_two_contact_dwell_and_advances_targets() -> None:
    preload = _preload()
    for _ in range(4):
        output = _step(preload)
        assert not bool(preload.initialized.any())
        assert torch.equal(output.joint_torque_bias_nm, torch.ones((2, 1, 2)))

    output = _step(preload)

    assert bool(preload.initialized.all())
    assert not bool(preload.complete.any())
    torch.testing.assert_close(
        output.joint_velocity_targets_radps[0, 0],
        torch.tensor([0.002, -0.002]),
    )
    torch.testing.assert_close(
        output.joint_position_targets_rad[0, 0],
        torch.tensor([0.10004, -0.10004]),
    )
    assert torch.equal(output.joint_torque_bias_nm, torch.zeros((2, 1, 2)))


def test_batched_preload_arms_at_nominal_end_without_raw_contact() -> None:
    preload = _preload()

    output = _step(
        preload,
        force_n=0.0,
        phase_progress=(1.0, 0.75),
    )

    assert preload.initialized.tolist() == [True, False]
    assert not bool(preload.complete.any())
    torch.testing.assert_close(
        output.joint_velocity_targets_radps[0, 0],
        torch.tensor([0.002, -0.002]),
    )
    assert torch.equal(
        output.joint_torque_bias_nm[1],
        torch.ones((1, 2)),
    )


def test_batched_preload_freezes_branches_and_holds_through_lift() -> None:
    preload = _preload()
    for _ in range(5):
        _step(preload)
    for _ in range(5):
        output = _step(
            preload,
            loads=((1.3, 0.2), (1.3, 1.3)),
        )

    assert preload.frozen_anchor[0].tolist() == [True, False]
    assert preload.frozen_anchor[1].tolist() == [True, True]
    assert preload.complete.tolist() == [False, True]
    torch.testing.assert_close(
        output.joint_velocity_targets_radps[0, 0],
        torch.tensor([0.0, -0.002]),
    )
    # The ordinary phase gate cannot enter lift before completion.  Exercise
    # the defensive carried-phase continuation with a full second dwell.
    for _ in range(5):
        held = _step(
            preload,
            phase=(2, 2),
            loads=((1.3, 1.3), (1.3, 1.3)),
        )
    assert bool(preload.complete.all())
    assert torch.equal(held.joint_velocity_targets_radps, torch.zeros((2, 1, 2)))


def test_phase_specific_lift_reset_is_restored_as_completed_hold() -> None:
    preload = _preload()
    output = _step(preload, phase=(2, 2))

    assert bool(preload.initialized.all())
    assert bool(preload.complete.all())
    assert torch.equal(
        output.joint_position_targets_rad,
        _command().joint_position_targets_rad,
    )
    assert torch.equal(output.joint_velocity_targets_radps, torch.zeros((2, 1, 2)))


def test_pi_h_normal_wrench_floor_can_complete_morphology_coupled_branch() -> None:
    preload = _preload()
    for _ in range(5):
        _step(preload)
    for _ in range(5):
        output = _step(
            preload,
            signed_normal_force_n=4.5,
            required_normal_force_n=4.0,
        )

    assert bool(preload.complete.all())
    assert bool(preload.frozen_anchor.all())
    assert torch.equal(
        output.joint_velocity_targets_radps,
        torch.zeros((2, 1, 2)),
    )
