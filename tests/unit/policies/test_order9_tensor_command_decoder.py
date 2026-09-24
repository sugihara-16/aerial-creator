from __future__ import annotations

import pytest
import torch

from amsrr.policies.order9_low_level_policy import (
    ORDER9_GLOBAL_ACTION_SIZE,
    Order9LowLevelPolicyConfig,
)
from amsrr.policies.order9_tensor_command_decoder import (
    ORDER9_INDEPENDENT_COMPRESSION_ACTION_ADAPTER_VERSION,
    Order9TensorPolicyCommandDecoder,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config


def _decoder():
    physical = build_physical_model_from_config("configs/robot/robot_model.yaml")
    config = Order9LowLevelPolicyConfig()
    return (
        Order9TensorPolicyCommandDecoder(
            module_ids=(0, 2), physical_model=physical, config=config
        ),
        config,
    )


def test_nominal_specialization_matches_zero_residual_general_decoder():
    decoder, config = _decoder()
    torch.manual_seed(87)
    batch = 8
    q = torch.randn(batch, 2, len(decoder.local_joint_ids)) * 10
    inputs = dict(reference_body_pose_world=torch.randn(batch, 7),
                  reference_body_twist=torch.randn(batch, 6),
                  reference_local_joint_positions_rad=q,
                  reference_local_joint_velocities_radps=torch.randn_like(q),
                  total_mass_kg=torch.rand(batch) * 10)
    # Includes negative-scalar and non-unit quaternions, and joint limit clipping.
    reference = decoder.decode(**inputs,
        normalized_global_action=torch.zeros(batch, ORDER9_GLOBAL_ACTION_SIZE),
        normalized_joint_action=torch.zeros(batch, 2, 3 * config.max_local_joint_slots),
        policy_module_ids=torch.tensor([[0, 2]] * batch),
        reference_local_joint_mask=torch.ones_like(q, dtype=torch.bool))
    nominal = decoder.decode_nominal(**inputs)
    for name in reference.__dataclass_fields__:
        a, b = getattr(reference, name), getattr(nominal, name)
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        else:
            assert a == b
    inputs['reference_body_pose_world'][0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        decoder.decode_nominal(**inputs)


def test_tensor_command_decoder_preserves_policy_controller_boundary() -> None:
    decoder, config = _decoder()
    batch_size = 2
    module_count = 2
    slot_count = len(decoder.local_joint_ids)
    pose = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]] * batch_size
    )
    twist = torch.zeros((batch_size, 6))
    global_action = torch.ones((batch_size, ORDER9_GLOBAL_ACTION_SIZE))
    joint_action = torch.ones(
        (batch_size, module_count, 3 * config.max_local_joint_slots)
    )
    current_q = torch.full((batch_size, module_count, slot_count), 0.2)
    mask = torch.ones_like(current_q, dtype=torch.bool)
    mass = torch.tensor([4.0, 6.0])

    command = decoder.decode(
        reference_body_pose_world=pose,
        reference_body_twist=twist,
        normalized_global_action=global_action,
        normalized_joint_action=joint_action,
        policy_module_ids=torch.tensor([[0, 2], [0, 2]]),
        reference_local_joint_positions_rad=current_q,
        reference_local_joint_velocities_radps=torch.zeros_like(current_q),
        reference_local_joint_mask=mask,
        total_mass_kg=mass,
    )

    assert torch.allclose(
        command.desired_body_pose_world[:, :3],
        torch.tensor([[0.05, 0.05, 1.05]] * batch_size),
        atol=1.0e-6,
    )
    assert torch.linalg.vector_norm(
        command.desired_body_pose_world[:, 3:7], dim=-1
    ).tolist() == pytest.approx([1.0] * batch_size)
    assert command.desired_body_twist[0, :3].tolist() == pytest.approx(
        [config.linear_twist_correction_limit_mps] * 3
    )
    assert command.desired_body_twist[0, 3:].tolist() == pytest.approx(
        [config.angular_twist_correction_limit_radps] * 3
    )
    assert command.residual_wrench_body[0, 0].item() == pytest.approx(
        4.0 * 9.81 * config.residual_force_weight_fraction
    )
    assert command.residual_wrench_body[0, 3].item() == pytest.approx(
        module_count * config.residual_torque_per_module_nm
    )
    assert command.joint_position_targets_rad[0, 0, 0].item() == pytest.approx(
        0.2 + config.joint_position_delta_limit_rad
    )
    assert command.joint_velocity_targets_radps[0, 0, 0].item() == pytest.approx(
        config.joint_velocity_limit_rad_s
    )
    assert command.joint_target_mask.all()
    assert command.module_ids == (0, 2)
    assert "pitch_dock_mech_joint1" in command.local_joint_ids


def test_tensor_command_decoder_rejects_policy_module_identity_mismatch() -> None:
    decoder, config = _decoder()
    slot_count = len(decoder.local_joint_ids)
    with pytest.raises(ValueError, match="module-id tensor differs"):
        decoder.decode(
            reference_body_pose_world=torch.tensor(
                [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]
            ),
            reference_body_twist=torch.zeros((1, 6)),
            normalized_global_action=torch.zeros((1, ORDER9_GLOBAL_ACTION_SIZE)),
            normalized_joint_action=torch.zeros(
                (1, 2, 3 * config.max_local_joint_slots)
            ),
            policy_module_ids=torch.tensor([[2, 0]]),
            reference_local_joint_positions_rad=torch.zeros((1, 2, slot_count)),
            reference_local_joint_velocities_radps=torch.zeros(
                (1, 2, slot_count)
            ),
            reference_local_joint_mask=torch.ones(
                (1, 2, slot_count), dtype=torch.bool
            ),
            total_mass_kg=torch.ones((1,)),
        )


def test_tensor_command_decoder_hard_clamps_joint_position_to_physical_limits() -> None:
    decoder, config = _decoder()
    slot_count = len(decoder.local_joint_ids)
    upper = torch.tensor(decoder._position_upper_limits).reshape(1, 1, -1)
    reference = upper.expand(1, 2, -1) - 0.01
    joint_action = torch.zeros((1, 2, 3 * config.max_local_joint_slots))
    joint_action[..., :slot_count] = 1.0
    command = decoder.decode(
        reference_body_pose_world=torch.tensor(
            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]
        ),
        reference_body_twist=torch.zeros((1, 6)),
        normalized_global_action=torch.zeros((1, ORDER9_GLOBAL_ACTION_SIZE)),
        normalized_joint_action=joint_action,
        policy_module_ids=torch.tensor([[0, 2]]),
        reference_local_joint_positions_rad=reference,
        reference_local_joint_velocities_radps=torch.zeros_like(reference),
        reference_local_joint_mask=torch.ones_like(reference, dtype=torch.bool),
        total_mass_kg=torch.ones((1,)),
    )
    assert torch.equal(
        command.joint_position_targets_rad,
        upper.expand_as(command.joint_position_targets_rad),
    )


def test_tensor_command_decoder_maps_one_existing_action_to_compression() -> None:
    decoder, config = _decoder()
    batch_size = 2
    module_count = 2
    slot_count = len(decoder.local_joint_ids)
    reference = torch.zeros((batch_size, module_count, slot_count))
    joint_action = torch.zeros(
        (batch_size, module_count, 3 * config.max_local_joint_slots)
    )
    direction = torch.zeros((module_count, slot_count))
    direction[0, 0] = 0.10
    direction[1, 0] = -0.20
    # The largest direction coordinate is repurposed as the scalar action.
    joint_action[:, 1, 0] = 0.5
    joint_action[:, 0, 1] = 0.25
    command = decoder.decode(
        reference_body_pose_world=torch.tensor(
            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]] * batch_size
        ),
        reference_body_twist=torch.zeros((batch_size, 6)),
        normalized_global_action=torch.zeros(
            (batch_size, ORDER9_GLOBAL_ACTION_SIZE)
        ),
        normalized_joint_action=joint_action,
        policy_module_ids=torch.tensor([[0, 2], [0, 2]]),
        reference_local_joint_positions_rad=reference,
        reference_local_joint_velocities_radps=torch.zeros_like(reference),
        reference_local_joint_mask=torch.ones_like(reference, dtype=torch.bool),
        total_mass_kg=torch.ones((batch_size,)),
        contact_compression_joint_direction_rad=direction,
        contact_compression_action_mask=torch.tensor([True, False]),
    )

    assert command.contact_compression_action.tolist() == pytest.approx(
        [0.5, 0.0]
    )
    assert command.joint_position_targets_rad[0, 0, 0].item() == pytest.approx(
        0.05
    )
    assert command.joint_position_targets_rad[0, 1, 0].item() == pytest.approx(
        -0.10
    )
    assert command.joint_position_targets_rad[0, 0, 1].item() == pytest.approx(
        0.25 * config.joint_position_delta_limit_rad
    )
    # Outside attach/maintain, the same coordinate retains its original
    # per-joint action meaning.
    assert command.joint_position_targets_rad[1, 1, 0].item() == pytest.approx(
        0.5 * config.joint_position_delta_limit_rad
    )
    assert torch.count_nonzero(
        command.contact_compression_joint_delta_rad[1]
    ).item() == 0


def test_tensor_command_decoder_adds_common_compression_residual() -> None:
    decoder, config = _decoder()
    batch_size = 2
    module_count = 2
    slot_count = len(decoder.local_joint_ids)
    reference = torch.zeros((batch_size, module_count, slot_count))
    joint_action = torch.zeros(
        (batch_size, module_count, 3 * config.max_local_joint_slots)
    )
    direction = torch.zeros((module_count, slot_count))
    direction[0, 0] = 0.10
    direction[1, 0] = -0.20
    joint_action[:, 1, 0] = 0.2
    command = decoder.decode(
        reference_body_pose_world=torch.tensor(
            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]] * batch_size
        ),
        reference_body_twist=torch.zeros((batch_size, 6)),
        normalized_global_action=torch.zeros(
            (batch_size, ORDER9_GLOBAL_ACTION_SIZE)
        ),
        normalized_joint_action=joint_action,
        policy_module_ids=torch.tensor([[0, 2], [0, 2]]),
        reference_local_joint_positions_rad=reference,
        reference_local_joint_velocities_radps=torch.zeros_like(reference),
        reference_local_joint_mask=torch.ones_like(reference, dtype=torch.bool),
        total_mass_kg=torch.ones((batch_size,)),
        contact_compression_joint_direction_rad=direction,
        contact_compression_action_mask=torch.tensor([True, False]),
        normalized_contact_compression_residual_action=torch.tensor([0.3, 0.7]),
    )

    assert command.contact_compression_legacy_joint_action.tolist() == pytest.approx(
        [0.2, 0.0]
    )
    assert command.contact_compression_residual_action.tolist() == pytest.approx(
        [0.3, 0.0]
    )
    assert command.contact_compression_action.tolist() == pytest.approx([0.5, 0.0])
    assert command.joint_position_targets_rad[0, 0, 0].item() == pytest.approx(
        0.05
    )
    assert command.joint_position_targets_rad[0, 1, 0].item() == pytest.approx(
        -0.10
    )
    assert command.joint_position_targets_rad[1, 1, 0].item() == pytest.approx(
        0.2 * config.joint_position_delta_limit_rad
    )


def test_tensor_command_decoder_keeps_joint_and_compression_actions_independent() -> None:
    decoder, config = _decoder()
    slot_count = len(decoder.local_joint_ids)
    reference = torch.zeros((1, 2, slot_count))
    joint_action = torch.zeros((1, 2, 3 * config.max_local_joint_slots))
    direction = torch.zeros((2, slot_count))
    direction[0, 0] = 0.10
    direction[1, 0] = -0.20
    # This is the largest-direction coordinate historically repurposed by v2.
    # In v3 it remains an ordinary, independent per-joint correction.
    joint_action[:, 1, 0] = 0.2

    command = decoder.decode(
        reference_body_pose_world=torch.tensor(
            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]
        ),
        reference_body_twist=torch.zeros((1, 6)),
        normalized_global_action=torch.zeros((1, ORDER9_GLOBAL_ACTION_SIZE)),
        normalized_joint_action=joint_action,
        policy_module_ids=torch.tensor([[0, 2]]),
        reference_local_joint_positions_rad=reference,
        reference_local_joint_velocities_radps=torch.zeros_like(reference),
        reference_local_joint_mask=torch.ones_like(reference, dtype=torch.bool),
        total_mass_kg=torch.ones((1,)),
        contact_compression_joint_direction_rad=direction,
        contact_compression_action_mask=torch.tensor([True]),
        normalized_contact_compression_residual_action=torch.tensor([0.3]),
        contact_compression_action_adapter_version=(
            ORDER9_INDEPENDENT_COMPRESSION_ACTION_ADAPTER_VERSION
        ),
    )

    assert command.contact_compression_legacy_joint_action.item() == 0.0
    assert command.contact_compression_action.item() == pytest.approx(0.3)
    assert command.joint_position_targets_rad[0, 0, 0].item() == pytest.approx(
        0.03
    )
    assert command.joint_position_targets_rad[0, 1, 0].item() == pytest.approx(
        0.2 * config.joint_position_delta_limit_rad - 0.06
    )


def test_tensor_command_decoder_clamps_complete_compression_action() -> None:
    decoder, config = _decoder()
    batch_size = 2
    module_count = 2
    slot_count = len(decoder.local_joint_ids)
    reference = torch.zeros((batch_size, module_count, slot_count))
    joint_action = torch.zeros(
        (batch_size, module_count, 3 * config.max_local_joint_slots)
    )
    direction = torch.zeros((module_count, slot_count))
    direction[0, 0] = 0.10
    joint_action[:, 0, 0] = torch.tensor([0.8, -0.8])
    command = decoder.decode(
        reference_body_pose_world=torch.tensor(
            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]] * batch_size
        ),
        reference_body_twist=torch.zeros((batch_size, 6)),
        normalized_global_action=torch.zeros(
            (batch_size, ORDER9_GLOBAL_ACTION_SIZE)
        ),
        normalized_joint_action=joint_action,
        policy_module_ids=torch.tensor([[0, 2], [0, 2]]),
        reference_local_joint_positions_rad=reference,
        reference_local_joint_velocities_radps=torch.zeros_like(reference),
        reference_local_joint_mask=torch.ones_like(reference, dtype=torch.bool),
        total_mass_kg=torch.ones((batch_size,)),
        contact_compression_joint_direction_rad=direction,
        contact_compression_action_mask=torch.tensor([True, True]),
        normalized_contact_compression_residual_action=torch.tensor([0.3, -0.3]),
        contact_compression_action_limit=0.35,
    )

    assert command.contact_compression_action.tolist() == pytest.approx(
        [0.35, -0.35]
    )
    assert command.contact_compression_joint_delta_rad[:, 0, 0].tolist() == (
        pytest.approx([0.035, -0.035])
    )
