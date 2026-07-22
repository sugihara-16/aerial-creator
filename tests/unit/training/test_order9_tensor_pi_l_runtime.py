from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from amsrr.encoders.morphology_graph_encoder import MORPHOLOGY_NODE_FEATURE_NAMES
from amsrr.policies.order9_low_level_policy import (
    ORDER9_GLOBAL_ACTION_SIZE,
    Order9LowLevelPolicyConfig,
    Order9PhaseConditionedActorCritic,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.simulation.order8_natural_contact import (
    build_representative_order8_morphology,
)
from amsrr.simulation.order9_tensor_isaac_io import Order9TensorIsaacState
from amsrr.simulation.order9_tensor_object_task import Order9TensorObjectTaskTarget
from amsrr.training.order9_tensor_pi_l_runtime import Order9TensorPiLRuntime


def _runtime_fixture(
    batch_size: int = 2,
    *,
    policy_frame_origins_world: torch.Tensor | None = None,
):
    physical = build_physical_model_from_config("configs/robot/robot_model.yaml")
    morphology = build_representative_order8_morphology(physical)
    policy = Order9PhaseConditionedActorCritic(Order9LowLevelPolicyConfig())
    runtime = Order9TensorPiLRuntime(
        morphology_graph=morphology,
        physical_model=physical,
        policy=policy,
        batch_size=batch_size,
        device="cpu",
        policy_frame_origins_world=policy_frame_origins_world,
    )
    module_count = runtime.builder.module_count
    joint_count = runtime.builder.local_joint_count
    pose = torch.zeros((batch_size, module_count, 7))
    pose[..., 6] = 1.0
    for module in range(module_count):
        pose[:, module, 0] = 0.16 * module
        pose[:, module, 2] = 0.8
    state = Order9TensorIsaacState(
        module_pose_world=pose,
        module_twist_world=torch.zeros((batch_size, module_count, 6)),
        local_joint_positions_rad=torch.zeros(
            (batch_size, module_count, joint_count)
        ),
        local_joint_velocities_radps=torch.zeros(
            (batch_size, module_count, joint_count)
        ),
        robot_root_pose_world=torch.tensor(
            [[0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0]] * batch_size
        ),
        robot_root_twist_world=torch.zeros((batch_size, 6)),
        object_pose_world=torch.tensor(
            [[0.3, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0]] * batch_size
        ),
        object_twist_world=torch.zeros((batch_size, 6)),
    )
    target = Order9TensorObjectTaskTarget(
        desired_robot_root_pose_world=torch.tensor(
            [[0.0, 0.0, 0.82, 0.0, 0.0, 0.0, 1.0]] * batch_size
        ),
        desired_robot_root_twist_world=torch.zeros((batch_size, 6)),
        nominal_joint_positions_rad=torch.zeros(
            (batch_size, module_count, len(runtime.decoder.local_joint_ids))
        ),
        nominal_joint_velocities_radps=torch.zeros(
            (batch_size, module_count, len(runtime.decoder.local_joint_ids))
        ),
        desired_object_pose_world=torch.tensor(
            [[0.3, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]] * batch_size
        ),
        phase_goal_robot_root_pose_world=torch.tensor(
            [[0.0, 0.0, 0.82, 0.0, 0.0, 0.0, 1.0]] * batch_size
        ),
        phase_goal_object_pose_world=torch.tensor(
            [[0.3, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]] * batch_size
        ),
        phase_progress=torch.tensor([0.1, 0.6]),
        contact_schedule_index=torch.tensor([1, 3]),
    )
    return runtime, state, target


def test_tensor_pi_l_runtime_matches_c0_graph_observation_contract() -> None:
    origins = torch.tensor([[1.5, -1.5, 0.0], [-1.5, 1.5, 0.0]])
    runtime, state, target = _runtime_fixture(
        policy_frame_origins_world=origins
    )
    local_module_pose = state.module_pose_world.clone()
    shifted_pose = local_module_pose.clone()
    shifted_pose[..., :3] += origins[:, None, :]
    shifted_pose[0, 2, 3:7] = -shifted_pose[0, 2, 3:7]
    state = replace(
        state,
        module_pose_world=shifted_pose,
        robot_root_pose_world=torch.cat(
            (
                state.robot_root_pose_world[:, :3] + origins,
                state.robot_root_pose_world[:, 3:7],
            ),
            dim=-1,
        ),
        object_pose_world=torch.cat(
            (
                state.object_pose_world[:, :3] + origins,
                state.object_pose_world[:, 3:7],
            ),
            dim=-1,
        ),
    )
    target = replace(
        target,
        desired_robot_root_pose_world=torch.cat(
            (
                target.desired_robot_root_pose_world[:, :3] + origins,
                target.desired_robot_root_pose_world[:, 3:7],
            ),
            dim=-1,
        ),
    )

    result = runtime.compute(
        time_s=torch.zeros(2),
        phase_index=torch.zeros(2, dtype=torch.long),
        task_target=target,
        state=state,
        estimated_payload_mass_kg=torch.zeros(2),
        estimated_payload_inertia_body=torch.zeros((2, 6)),
        payload_active=torch.zeros(2, dtype=torch.bool),
        deterministic=True,
    )

    features = runtime.bucket.batch.node_features
    pose_start = MORPHOLOGY_NODE_FEATURE_NAMES.index("runtime.pose.x")
    pose_end = MORPHOLOGY_NODE_FEATURE_NAMES.index("runtime.pose.qw") + 1
    joint_count_index = MORPHOLOGY_NODE_FEATURE_NAMES.index(
        "runtime.joint_position.count"
    )
    encoded_pose = features[..., pose_start:pose_end]
    assert torch.allclose(encoded_pose, local_module_pose, atol=1.0e-6)
    assert torch.equal(
        features[..., joint_count_index],
        torch.full_like(features[..., joint_count_index], 12.0),
    )
    assert torch.allclose(
        result.control_model.body_pose_world[:, :2]
        - origins[:, :2],
        result.control_model.body_pose_world[0:1, :2] - origins[0:1, :2],
        atol=1.0e-6,
    )
    assert runtime.bucket.batch.metadata["runtime_pose_translation_frame"] == (
        "world_minus_policy_frame_origin"
    )
    assert runtime.bucket.batch.metadata["runtime_active_local_joint_count"] == 12


def test_tensor_pi_l_runtime_rejects_invalid_policy_frame_origins() -> None:
    with pytest.raises(ValueError, match="origins"):
        _runtime_fixture(
            policy_frame_origins_world=torch.zeros((2, 2))
        )


def test_tensor_pi_l_runtime_preserves_policy_command_controller_boundary() -> None:
    torch.manual_seed(7)
    runtime, state, target = _runtime_fixture()
    phase = torch.tensor([0, 3], dtype=torch.long)

    result = runtime.compute(
        time_s=torch.tensor([0.0, 0.2]),
        phase_index=phase,
        task_target=target,
        state=state,
        estimated_payload_mass_kg=torch.tensor([0.1, 0.2]),
        estimated_payload_inertia_body=torch.tensor(
            [[0.001, 0.001, 0.001, 0.0, 0.0, 0.0]] * 2
        ),
        payload_active=torch.tensor([False, True]),
        deterministic=True,
    )

    assert result.policy_step.action.shape == (2, ORDER9_GLOBAL_ACTION_SIZE)
    assert result.policy_step.joint_action.shape[:2] == (
        2,
        runtime.builder.module_count,
    )
    assert torch.isfinite(result.policy_command.desired_body_pose_world).all()
    assert torch.all(
        (
            result.policy_command.desired_body_pose_world[:, :3]
            - target.desired_robot_root_pose_world[:, :3]
        ).abs()
        <= runtime.config.centroidal_position_correction_limit_m + 1.0e-6
    )
    assert torch.allclose(
        torch.linalg.vector_norm(
            result.policy_command.desired_body_pose_world[:, 3:7], dim=-1
        ),
        torch.ones(2),
        atol=1.0e-6,
    )
    assert result.controller_result.allocation.rotor_thrusts_n.shape == (
        2,
        runtime.builder.rotor_count,
    )
    phase_offset = len(runtime._phase_feature_template[0]) - (
        runtime.config.max_phase_count + 3
    )
    assert result.phase_features[0, phase_offset].item() == 1.0
    assert result.phase_features[1, phase_offset + 3].item() == 1.0
    assert torch.isfinite(result.controller_result.desired_wrench_body).all()
    assert torch.equal(runtime.previous_action, result.policy_step.action)


def test_tensor_pi_l_runtime_accepts_canonical_actor_phase_indices() -> None:
    runtime, state, target = _runtime_fixture()
    result = runtime.compute(
        time_s=torch.zeros(2),
        phase_index=torch.tensor([8, 10], dtype=torch.long),
        task_target=target,
        state=state,
        estimated_payload_mass_kg=torch.zeros(2),
        estimated_payload_inertia_body=torch.zeros((2, 6)),
        payload_active=torch.zeros(2, dtype=torch.bool),
        deterministic=True,
    )
    phase_offset = len(runtime._phase_feature_template[0]) - (
        runtime.config.max_phase_count + 3
    )
    assert result.phase_features[0, phase_offset + 8].item() == 1.0
    assert result.phase_features[1, phase_offset + 10].item() == 1.0

    with pytest.raises(ValueError, match="phase index"):
        runtime.compute(
            time_s=torch.zeros(2),
            phase_index=torch.tensor([0, 11], dtype=torch.long),
            task_target=target,
            state=state,
            estimated_payload_mass_kg=torch.zeros(2),
            estimated_payload_inertia_body=torch.zeros((2, 6)),
            payload_active=torch.zeros(2, dtype=torch.bool),
            deterministic=True,
        )


def test_tensor_pi_l_runtime_resets_only_terminal_recurrent_rows() -> None:
    runtime, state, target = _runtime_fixture()
    result = runtime.compute(
        time_s=torch.zeros(2),
        phase_index=torch.zeros(2, dtype=torch.long),
        task_target=target,
        state=state,
        estimated_payload_mass_kg=torch.zeros(2),
        estimated_payload_inertia_body=torch.zeros((2, 6)),
        payload_active=torch.zeros(2, dtype=torch.bool),
        deterministic=True,
    )
    second_state = runtime.recurrent_state[1].clone()

    runtime.finish_transition(
        phase_success=torch.tensor([False, True]),
        terminal_or_reset=torch.tensor([True, False]),
        current_vectoring_angles_rad=(
            result.control_model.current_vectoring_angles_rad
        ),
    )

    assert torch.equal(runtime.recurrent_state[0], torch.zeros_like(second_state))
    assert torch.equal(runtime.recurrent_state[1], second_state)
    assert runtime.task_success.tolist() == [False, True]
    assert runtime.controller_qp_feasible.tolist()[0] is True


def test_tensor_pi_l_bootstrap_value_does_not_advance_runtime_state() -> None:
    runtime, state, target = _runtime_fixture()
    runtime.compute(
        time_s=torch.zeros(2),
        phase_index=torch.zeros(2, dtype=torch.long),
        task_target=target,
        state=state,
        estimated_payload_mass_kg=torch.zeros(2),
        estimated_payload_inertia_body=torch.zeros((2, 6)),
        payload_active=torch.zeros(2, dtype=torch.bool),
        deterministic=True,
    )
    before = (
        runtime.previous_action.clone(),
        runtime.recurrent_state.clone(),
        runtime.controller_state.previous_rotor_thrusts_n.clone(),
        runtime.controller_status_one_hot.clone(),
    )

    value = runtime.evaluate_bootstrap_value(
        time_s=torch.full((2,), 0.02),
        phase_index=torch.zeros(2, dtype=torch.long),
        task_target=target,
        state=state,
        estimated_payload_mass_kg=torch.zeros(2),
        estimated_payload_inertia_body=torch.zeros((2, 6)),
        payload_active=torch.zeros(2, dtype=torch.bool),
    )

    assert value.shape == (2,)
    assert torch.isfinite(value).all()
    for current, expected in zip(
        (
            runtime.previous_action,
            runtime.recurrent_state,
            runtime.controller_state.previous_rotor_thrusts_n,
            runtime.controller_status_one_hot,
        ),
        before,
        strict=True,
    ):
        assert torch.equal(current, expected)


def test_tensor_pi_l_runtime_applies_estimated_object_frame_com_error() -> None:
    runtime, state, target = _runtime_fixture()
    target = replace(
        target,
        phase_progress=torch.full((2,), 0.5),
        contact_schedule_index=torch.full((2,), 3),
    )
    result = runtime.compute(
        time_s=torch.zeros(2),
        phase_index=torch.full((2,), 3, dtype=torch.long),
        task_target=target,
        state=state,
        estimated_payload_mass_kg=torch.ones(2),
        estimated_payload_inertia_body=torch.tensor(
            [[0.01, 0.0, 0.0, 0.01, 0.0, 0.01]] * 2
        ),
        payload_active=torch.ones(2, dtype=torch.bool),
        estimated_payload_com_object=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.05, 0.0]]
        ),
        deterministic=True,
    )

    torque_delta = (
        result.controller_result.desired_wrench_body[1, 3:]
        - result.controller_result.desired_wrench_body[0, 3:]
    )
    assert torque_delta[0].abs() > 0.1
    assert torque_delta[1].abs() < 1.0e-5
