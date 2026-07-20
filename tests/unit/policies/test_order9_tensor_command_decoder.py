from __future__ import annotations

import pytest
import torch

from amsrr.policies.order9_low_level_policy import Order9LowLevelPolicyConfig
from amsrr.policies.order9_tensor_command_decoder import (
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


def test_tensor_command_decoder_preserves_policy_controller_boundary() -> None:
    decoder, config = _decoder()
    batch_size = 2
    module_count = 2
    slot_count = len(decoder.local_joint_ids)
    pose = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]] * batch_size
    )
    twist = torch.zeros((batch_size, 6))
    baseline_wrench = torch.ones((batch_size, 6))
    global_action = torch.ones((batch_size, 12))
    joint_action = torch.ones(
        (batch_size, module_count, 3 * config.max_local_joint_slots)
    )
    current_q = torch.full((batch_size, module_count, slot_count), 0.2)
    mask = torch.ones_like(current_q, dtype=torch.bool)
    mass = torch.tensor([4.0, 6.0])

    command = decoder.decode(
        baseline_body_pose_world=pose,
        baseline_body_twist=twist,
        baseline_residual_wrench_body=baseline_wrench,
        normalized_global_action=global_action,
        normalized_joint_action=joint_action,
        policy_module_ids=torch.tensor([[0, 2], [0, 2]]),
        current_local_joint_positions_rad=current_q,
        current_local_joint_mask=mask,
        total_mass_kg=mass,
    )

    blend = config.trust_region_blend
    assert command.desired_body_pose_world is pose
    assert command.desired_body_twist[0, :3].tolist() == pytest.approx(
        [blend * config.linear_twist_correction_limit_mps] * 3
    )
    assert command.desired_body_twist[0, 3:].tolist() == pytest.approx(
        [blend * config.angular_twist_correction_limit_radps] * 3
    )
    assert command.residual_wrench_body[0, 0].item() == pytest.approx(
        1.0
        + blend
        * 4.0
        * 9.81
        * config.residual_force_weight_fraction
    )
    assert command.residual_wrench_body[0, 3].item() == pytest.approx(
        1.0
        + blend
        * module_count
        * config.residual_torque_per_module_nm
    )
    assert command.joint_position_targets_rad[0, 0, 0].item() == pytest.approx(
        0.2 + blend * config.joint_position_delta_limit_rad
    )
    assert command.joint_velocity_targets_radps[0, 0, 0].item() == pytest.approx(
        blend * config.joint_velocity_limit_rad_s
    )
    assert command.joint_target_mask.all()
    assert command.module_ids == (0, 2)
    assert "pitch_dock_mech_joint1" in command.local_joint_ids


def test_tensor_command_decoder_rejects_policy_module_identity_mismatch() -> None:
    decoder, config = _decoder()
    slot_count = len(decoder.local_joint_ids)
    with pytest.raises(ValueError, match="module-id tensor differs"):
        decoder.decode(
            baseline_body_pose_world=torch.tensor(
                [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]
            ),
            baseline_body_twist=torch.zeros((1, 6)),
            baseline_residual_wrench_body=torch.zeros((1, 6)),
            normalized_global_action=torch.zeros((1, 12)),
            normalized_joint_action=torch.zeros(
                (1, 2, 3 * config.max_local_joint_slots)
            ),
            policy_module_ids=torch.tensor([[2, 0]]),
            current_local_joint_positions_rad=torch.zeros((1, 2, slot_count)),
            current_local_joint_mask=torch.ones(
                (1, 2, slot_count), dtype=torch.bool
            ),
            total_mass_kg=torch.ones((1,)),
        )
