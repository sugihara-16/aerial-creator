from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import pytest
import torch

from amsrr.controllers.rigid_body_model import RigidBodyControlModelBuilder
from amsrr.morphology.random_connected import (
    RandomConnectedMorphologyDistribution,
    morphology_structural_hash,
)
from amsrr.policies.low_level_policy_base import (
    BaselineLowLevelPolicy,
    BaselineLowLevelPolicyConfig,
    LowLevelPolicyContext,
)
from amsrr.policies.morphology_conditioned_low_level_policy import (
    ORDER3_ACTOR_FEATURE_NAMES,
    MorphologyConditionedActorCritic,
    MorphologyConditionedLowLevelPolicy,
    Order3MorphologyConditionedPolicyConfig,
    load_order3_policy_checkpoint,
    order3_actor_feature_schema_hash,
    order3_actor_feature_vector,
    order3_graph_feature_schema_hash,
    save_order3_policy_checkpoint,
)
from amsrr.policies.order9_low_level_policy import (
    ORDER9_GLOBAL_ACTION_SIZE,
    ORDER9_PI_L_POLICY_VERSION,
    Order9LowLevelPolicyConfig,
    Order9PhaseConditionedActorCritic,
    order9_phase_actor_feature_vector,
)
from amsrr.policies.order9_low_level_runtime import (
    Order9CompletePolicyCommandRuntime,
    Order9LowLevelRuntimePolicy,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.datasets import (
    DatasetSplit,
    LowLevelControlRecord,
    PolicyBehaviorTrace,
    StageDecisionMasks,
)
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.order3 import (
    ORDER3_ACTION_NAMES,
    ORDER3_ACTION_SIZE,
    ORDER3_CHECKPOINT_VERSION,
    ORDER3_ENCODER_VERSION,
    ORDER3_FALLBACK_VERSION,
    ORDER3_POLICY_ARCHITECTURE_VERSION,
    ORDER3_POLICY_FAMILY,
    ORDER3_TENSORIZER_VERSION,
    Order3PolicyCheckpointMetadata,
)
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import (
    POLICY_COMMAND_CONTRACT_CENTROIDAL,
    CentroidalTarget,
    ContactWrenchTrajectory,
    ControllerCommand,
    ControllerStatus,
    InteractionKnot,
)
from amsrr.schemas.runtime import (
    ContactState,
    ModuleRuntimeState,
    RuntimeObservation,
    TaskProgressState,
)
from amsrr.training.order9_pi_l_learning import (
    compute_order9_pi_l_behavior_cloning_loss,
    compute_order9_pi_l_ppo_loss,
    encode_order9_pi_l_teacher_action,
)
from amsrr.training.order9_curriculum import Order9PPOOptimizationConfig
from amsrr.training.order9_checkpoints import save_order9_policy_checkpoint
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_offline_training import build_order9_checkpoint_metadata
from amsrr.training.order9_pipeline import order9_schedule_hash, order9_stage_by_id
from amsrr.training.order9_ppo import (
    ORDER9_PI_L_ACTION_SEMANTICS,
    ORDER9_PI_L_GRAPH_JOINT_SUMMARY_NON_FIXED,
    _pi_l_actor_graph_observation,
    order9_pi_l_behavior_trace_from_inference,
    update_order9_pi_l_ppo,
)
from amsrr.training.order9_tensor_runtime import (
    Order9CentroidalTensorObservation,
    Order9TensorizedTopologyBucket,
    order9_low_level_actor_features_from_tensors,
)
from amsrr.utils.hashing import hash_file, stable_hash


@pytest.fixture(scope="module")
def physical_model() -> PhysicalModel:
    return build_physical_model_from_config("configs/robot/robot_model.yaml")


@pytest.fixture(scope="module")
def morphology_distribution(
    physical_model: PhysicalModel,
) -> RandomConnectedMorphologyDistribution:
    return RandomConnectedMorphologyDistribution(physical_model)


def test_actor_critic_shapes_masks_and_backpropagate_through_graph_and_gru(
    physical_model: PhysicalModel,
    morphology_distribution: RandomConnectedMorphologyDistribution,
) -> None:
    graphs = [
        morphology_distribution.sample(seed=101, module_count=2),
        morphology_distribution.sample(seed=102, module_count=5),
    ]
    observations = [_runtime(graph, physical_model) for graph in graphs]
    config = _small_config()
    torch.manual_seed(123)
    model = MorphologyConditionedActorCritic(config)
    actor_features = torch.zeros((2, len(ORDER3_ACTOR_FEATURE_NAMES)))
    previous_action = torch.zeros((2, ORDER3_ACTION_SIZE))
    recurrent_state = model.initial_state(2)

    step = model.step(
        graphs,
        observations,
        actor_features,
        previous_action,
        recurrent_state,
        privileged_disturbance_body=torch.ones((2, 6)),
        deterministic=False,
    )

    assert step.action.shape == (2, ORDER3_ACTION_SIZE)
    assert step.action_mean.shape == (2, ORDER3_ACTION_SIZE)
    assert step.log_prob.shape == (2,)
    assert step.entropy.shape == (2,)
    assert step.value.shape == (2,)
    assert step.recurrent_state.shape == (2, config.recurrent_hidden_dim)
    assert step.graph_encoding.tokens.shape == (2, 5, config.graph_hidden_dim)
    assert step.graph_encoding.mask.sum(dim=1).tolist() == [2, 5]
    assert step.joint_residuals.shape == (2, 5, 3 * config.max_local_joint_slots)
    assert torch.count_nonzero(step.joint_residuals[0, 2:]).item() == 0
    assert bool(torch.isfinite(step.log_prob).all().item())
    assert bool((step.action.abs() <= 1.0).all().item())

    loss = (
        step.action.square().mean()
        + step.value.square().mean()
        + step.recurrent_state.square().mean()
        + step.joint_residuals.square().mean()
    )
    loss.backward()
    assert model.graph_encoder.node_projection[0].weight.grad is not None
    assert torch.count_nonzero(model.graph_encoder.node_projection[0].weight.grad).item() > 0
    assert model.recurrent.weight_ih.grad is not None
    assert torch.count_nonzero(model.recurrent.weight_ih.grad).item() > 0
    assert model.actor_mean.weight.grad is not None
    assert torch.count_nonzero(model.actor_mean.weight.grad).item() > 0


def test_policy_runtime_state_round_trip_restores_recurrent_behavior(
    physical_model: PhysicalModel,
) -> None:
    config = _small_config()
    runtime = MorphologyConditionedLowLevelPolicy(
        model=MorphologyConditionedActorCritic(config),
        physical_model=physical_model,
        config=config,
    )
    runtime.checkpoint_sha256 = "a" * 64
    runtime._hidden.fill_(0.25)
    runtime._previous_action.fill_(-0.5)
    runtime._last_graph_id = "runtime-graph"
    runtime._last_time_s = 1.25
    exported = runtime.export_runtime_state()

    runtime.reset()
    runtime.restore_runtime_state(exported)

    assert torch.allclose(runtime._hidden, torch.full_like(runtime._hidden, 0.25))
    assert torch.allclose(
        runtime._previous_action,
        torch.full_like(runtime._previous_action, -0.5),
    )
    assert runtime._last_graph_id == "runtime-graph"
    assert runtime._last_time_s == pytest.approx(1.25)


def test_policy_runtime_state_fails_closed_on_nonfinite_hidden(
    physical_model: PhysicalModel,
) -> None:
    config = _small_config()
    runtime = MorphologyConditionedLowLevelPolicy(
        model=MorphologyConditionedActorCritic(config),
        physical_model=physical_model,
        config=config,
    )
    runtime.checkpoint_sha256 = "b" * 64
    exported = runtime.export_runtime_state()
    exported["hidden"][0][0] = float("nan")  # type: ignore[index]

    with pytest.raises(SchemaValidationError, match="must be finite"):
        runtime.restore_runtime_state(exported)


def test_order9_actor_conditions_on_phase_and_includes_joint_action_in_log_prob(
    physical_model: PhysicalModel,
    morphology_distribution: RandomConnectedMorphologyDistribution,
) -> None:
    graphs = [
        morphology_distribution.sample(seed=911, module_count=2),
        morphology_distribution.sample(seed=912, module_count=5),
    ]
    contexts = [
        replace(
            _context(graph, physical_model),
            task_type="object_grasp_carry",
            task_adapter_id="object_grasp_carry_v1",
            phase_index=index,
            phase_count=11,
        )
        for index, graph in enumerate(graphs)
    ]
    config = Order9LowLevelPolicyConfig(
        graph_hidden_dim=16,
        graph_message_layers=1,
        recurrent_hidden_dim=24,
        max_local_joint_slots=4,
    )
    model = Order9PhaseConditionedActorCritic(config)
    phase_features = torch.tensor(
        [order9_phase_actor_feature_vector(context, config) for context in contexts]
    )
    step = model.step(
        graphs,
        [context.runtime_observation for context in contexts],
        torch.zeros((2, len(ORDER3_ACTOR_FEATURE_NAMES))),
        torch.zeros((2, ORDER9_GLOBAL_ACTION_SIZE)),
        model.initial_state(2),
        phase_features=phase_features,
        deterministic=False,
    )

    assert step.action.shape == (2, ORDER9_GLOBAL_ACTION_SIZE)
    assert step.joint_action.shape == (2, 5, 3 * config.max_local_joint_slots)
    assert torch.count_nonzero(step.joint_action[0, 2:]).item() == 0
    assert torch.isfinite(step.log_prob).all()
    assert torch.isfinite(step.entropy).all()
    loss = step.log_prob.mean() + step.value.mean() + step.joint_action.square().mean()
    loss.backward()
    assert model.joint_decoder[0].weight.grad is not None
    assert model.joint_actor_log_std.grad is not None


def test_order9_model_warm_starts_order3_trunk_and_deploys_with_phase_context(
    physical_model: PhysicalModel,
    morphology_distribution: RandomConnectedMorphologyDistribution,
) -> None:
    graph = morphology_distribution.sample(seed=913, module_count=3)
    config = Order9LowLevelPolicyConfig(
        graph_hidden_dim=16,
        graph_message_layers=1,
        recurrent_hidden_dim=24,
        max_local_joint_slots=4,
    )
    order3 = MorphologyConditionedActorCritic(
        Order3MorphologyConditionedPolicyConfig(
            graph_hidden_dim=config.graph_hidden_dim,
            graph_message_layers=config.graph_message_layers,
            recurrent_hidden_dim=config.recurrent_hidden_dim,
            max_local_joint_slots=config.max_local_joint_slots,
        )
    )
    model = Order9PhaseConditionedActorCritic(config)
    missing, unexpected = model.initialize_from_order3(order3)
    context = replace(
        _context(graph, physical_model),
        task_type="object_grasp_carry",
        task_adapter_id="object_grasp_carry_v1",
        phase_index=3,
        phase_count=11,
    )
    wrapper = Order9CompletePolicyCommandRuntime(
        model=model,
        physical_model=physical_model,
        config=config,
        deterministic=True,
    )

    inference = wrapper.command_with_trace(context)

    assert missing
    assert unexpected == []
    assert inference.learned_policy_applied
    assert inference.normalized_joint_action
    assert inference.joint_action_mean
    assert inference.command.control_contract_version == POLICY_COMMAND_CONTRACT_CENTROIDAL
    assert inference.command.joint_position_targets


def test_order9_pi_l_teacher_codec_and_losses_cover_both_action_heads(
    physical_model: PhysicalModel,
    morphology_distribution: RandomConnectedMorphologyDistribution,
) -> None:
    graph = morphology_distribution.sample(seed=914, module_count=3)
    context = replace(
        _context(graph, physical_model),
        task_type="object_grasp_carry",
        task_adapter_id="object_grasp_carry_v1",
        phase_index=2,
        phase_count=11,
    )
    config = Order9LowLevelPolicyConfig(
        graph_hidden_dim=16,
        graph_message_layers=1,
        recurrent_hidden_dim=24,
        max_local_joint_slots=4,
    )
    teacher_model = Order9PhaseConditionedActorCritic(config)
    teacher_policy = Order9CompletePolicyCommandRuntime(
        model=teacher_model,
        physical_model=physical_model,
        config=config,
        deterministic=True,
    )
    teacher_trace = teacher_policy.command_with_trace(context)
    control_model = RigidBodyControlModelBuilder().build(
        graph, physical_model, context.runtime_observation
    )

    encoded = encode_order9_pi_l_teacher_action(
        context=context,
        teacher_command=teacher_trace.command,
        control_model=control_model,
        config=config,
    )

    assert encoded.global_action == pytest.approx(teacher_trace.normalized_action)
    assert torch.allclose(
        torch.tensor(encoded.joint_action),
        torch.tensor(teacher_trace.normalized_joint_action),
        atol=1.0e-6,
    )
    student = Order9PhaseConditionedActorCritic(config)
    loss = compute_order9_pi_l_behavior_cloning_loss(
        student,
        [context],
        [teacher_trace.command],
        decision_returns=[1.5],
    )
    assert loss.active_joint_coordinate_count > 0
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert student.actor_mean.weight.grad is not None
    assert student.joint_decoder[0].weight.grad is not None

    ppo = compute_order9_pi_l_ppo_loss(
        new_log_prob=torch.tensor([-0.2], requires_grad=True),
        old_log_prob=torch.tensor([-0.3]),
        advantages=torch.tensor([1.0]),
        new_values=torch.tensor([0.4], requires_grad=True),
        returns=torch.tensor([0.8]),
        entropy=torch.tensor([0.5]),
    )
    assert torch.isfinite(ppo)

    excessive_twist = list(teacher_trace.command.desired_body_twist or [0.0] * 6)
    excessive_twist[0] += 10.0
    with pytest.raises(SchemaValidationError, match="exceeds the actor authority"):
        encode_order9_pi_l_teacher_action(
            context=context,
            teacher_command=replace(
                teacher_trace.command, desired_body_twist=excessive_twist
            ),
            control_model=control_model,
            config=config,
        )


def test_order9_pi_l_recurrent_behavior_trace_replays_through_ppo(
    physical_model: PhysicalModel,
    morphology_distribution: RandomConnectedMorphologyDistribution,
) -> None:
    graph = morphology_distribution.sample(seed=916, module_count=3)
    context = replace(
        _context(graph, physical_model),
        task_type="object_grasp_carry",
        task_adapter_id="object_grasp_carry_v1",
        phase_index=2,
        phase_count=11,
    )
    config = Order9LowLevelPolicyConfig(
        graph_hidden_dim=16,
        graph_message_layers=1,
        recurrent_hidden_dim=24,
        max_local_joint_slots=4,
    )
    model = Order9PhaseConditionedActorCritic(config)
    wrapper = Order9CompletePolicyCommandRuntime(
        model=model,
        physical_model=physical_model,
        config=config,
        deterministic=False,
    )
    inference = wrapper.command_with_trace(context)
    checkpoint_sha = "c" * 64
    module_ids = sorted(module.module_id for module in graph.modules)
    behavior = PolicyBehaviorTrace(
        policy_family="pi_l",
        policy_version=ORDER9_PI_L_POLICY_VERSION,
        action_semantics=ORDER9_PI_L_ACTION_SEMANTICS,
        action_payload={
            "global_action": inference.normalized_action,
            "module_ids": module_ids,
            "joint_action": inference.normalized_joint_action,
            "previous_global_action": inference.previous_action,
            "privileged_disturbance_body": [0.0] * 6,
            "actor_graph_frame_origin_world": [0.0, 0.0, 0.0],
            "actor_graph_joint_summary_semantics": "all_present_joints",
        },
        stochastic=True,
        policy_checkpoint_sha256=checkpoint_sha,
        old_log_prob=inference.log_prob,
        old_value=inference.value,
        recurrent_state_in=inference.recurrent_state_in,
        recurrent_state_out=inference.recurrent_state_out,
    )
    record = LowLevelControlRecord(
        record_id="order9-pi-l-ppo-0",
        episode_id="order9-pi-l-ppo-episode",
        task_id="order9-pi-l-ppo-task",
        split=DatasetSplit.TRAIN,
        step_index=0,
        time_s=context.runtime_observation.time_s,
        trajectory_record_id="order9-pi-l-trajectory-0",
        active_trajectory_index=0,
        active_knot_index=0,
        runtime_observation=context.runtime_observation,
        active_knot=context.active_knot,
        policy_command=inference.command,
        controller_command=ControllerCommand(
            rotor_thrusts_n={},
            vectoring_joint_targets={},
            joint_torque_commands={},
            dock_mechanism_commands={},
            controller_status=context.runtime_observation.controller_status,
            control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
        ),
        actuator_target_record={},
        reward_terms={"task": 1.0},
        reward=1.0,
        terminal=True,
        stage_masks=StageDecisionMasks(low_level_control_mask=True),
        task_type="object_grasp_carry",
        task_adapter_id="object_grasp_carry_v1",
        phase_index=2,
        phase_count=11,
        behavior_trace=behavior,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    progress = []

    result = update_order9_pi_l_ppo(
        model,
        [record],
        physical_model=physical_model,
        optimizer=optimizer,
        config=Order9PPOOptimizationConfig(
            rollout_steps_per_environment=1,
            epochs_per_update=1,
            minibatch_size=1,
        ),
        behavior_checkpoint_sha256=checkpoint_sha,
        seed=9,
        sequence_length=1,
        progress_callback=lambda step, metrics: progress.append((step, metrics)),
    )

    assert result.policy_family == "pi_l"
    assert result.sample_count == 1
    assert result.optimizer_step_count == 1
    assert progress[0][0] == 1
    assert progress[0][1]["optimizer_step"] == 1.0
    assert "actor_loss" in progress[0][1]
    assert result.metadata["exact_behavior_replay_validated"] is True
    assert result.metadata["exact_replay_record_count"] == 1
    assert result.metadata["timestep_batched_active_sequences"] is True
    assert (
        result.metadata["exact_replay_timestep_batched_active_sequences"] is True
    )


def test_order9_tensor_hot_path_matches_schema_features_and_graph_encoding(
    physical_model: PhysicalModel,
    morphology_distribution: RandomConnectedMorphologyDistribution,
) -> None:
    graph = morphology_distribution.sample(seed=915, module_count=3)
    context = replace(
        _context(graph, physical_model, time_s=0.7),
        task_type="object_grasp_carry",
        task_adapter_id="object_grasp_carry_v1",
        phase_index=2,
        phase_count=11,
    )
    control_model = RigidBodyControlModelBuilder().build(
        graph, physical_model, context.runtime_observation
    )
    target_pose = (0.4, -0.2, 1.8, 0.0, 0.0, 0.0, 1.0)
    target_twist = [0.1, -0.1, 0.0, 0.02, 0.0, -0.03]
    expected_features = order3_actor_feature_vector(
        context.runtime_observation,
        control_model,
        target_pose_world=target_pose,
        target_twist=target_twist,
    )
    status_order = ("ok", "warning", "infeasible", "fault")
    tensor_features = order9_low_level_actor_features_from_tensors(
        Order9CentroidalTensorObservation(
            time_s=torch.tensor([context.runtime_observation.time_s]),
            module_count=torch.tensor([float(len(graph.modules))]),
            total_mass_kg=torch.tensor([control_model.total_mass_kg]),
            inertia_body=torch.tensor([control_model.inertia_body]),
            body_pose_world=torch.tensor([control_model.body_pose_world]),
            body_twist_world=torch.tensor([control_model.body_twist_world]),
            target_pose_world=torch.tensor([target_pose]),
            target_twist=torch.tensor([target_twist]),
            controller_qp_feasible=torch.tensor([1.0]),
            controller_status_one_hot=torch.tensor(
                [[1.0 if context.runtime_observation.controller_status.status == item else 0.0 for item in status_order]]
            ),
            allocation_residual_norm=torch.tensor(
                [
                    context.runtime_observation.controller_status.metrics.get(
                        "allocation_residual_norm",
                        context.runtime_observation.controller_status.metrics.get(
                            "residual_norm", 0.0
                        ),
                    )
                ]
            ),
            task_progress_ratio=torch.tensor(
                [context.runtime_observation.task_progress.progress_ratio]
            ),
            task_success=torch.tensor(
                [float(context.runtime_observation.task_progress.success)]
            ),
        )
    )
    assert torch.allclose(
        tensor_features,
        torch.tensor([expected_features]),
        atol=1.0e-6,
    )

    states = sorted(context.runtime_observation.module_states, key=lambda item: item.module_id)
    joint_ids = sorted({joint_id for state in states for joint_id in state.joint_positions})
    positions = torch.zeros((1, len(states), len(joint_ids)))
    velocities = torch.zeros_like(positions)
    joint_mask = torch.zeros_like(positions, dtype=torch.bool)
    for module_index, state in enumerate(states):
        for joint_index, joint_id in enumerate(joint_ids):
            if joint_id in state.joint_positions:
                positions[0, module_index, joint_index] = state.joint_positions[joint_id]
                velocities[0, module_index, joint_index] = state.joint_velocities[joint_id]
                joint_mask[0, module_index, joint_index] = True
    bucket = Order9TensorizedTopologyBucket(graph, batch_size=1, device="cpu")
    graph_batch = bucket.update_runtime_(
        module_pose_world=torch.tensor([[state.pose_world for state in states]]),
        module_twist_world=torch.tensor([[state.twist_world for state in states]]),
        module_health=torch.tensor([[state.health for state in states]]),
        joint_positions=positions,
        joint_velocities=velocities,
        joint_mask=joint_mask,
        strict=True,
    )
    config = Order9LowLevelPolicyConfig(
        graph_hidden_dim=16,
        graph_message_layers=1,
        recurrent_hidden_dim=24,
        max_local_joint_slots=4,
    )
    model = Order9PhaseConditionedActorCritic(config)
    phase = torch.tensor([order9_phase_actor_feature_vector(context, config)])
    previous = torch.zeros((1, ORDER9_GLOBAL_ACTION_SIZE))
    hidden = model.initial_state(1)
    schema_step = model.step(
        [graph],
        [context.runtime_observation],
        tensor_features,
        previous,
        hidden,
        phase_features=phase,
        deterministic=True,
    )
    tensor_step = model.step(
        graph_batch,
        None,
        tensor_features,
        previous,
        hidden,
        phase_features=phase,
        deterministic=True,
    )
    assert torch.allclose(
        schema_step.graph_encoding.tokens,
        tensor_step.graph_encoding.tokens,
        atol=1.0e-6,
    )
    assert torch.allclose(schema_step.action, tensor_step.action, atol=1.0e-6)
    assert torch.allclose(schema_step.joint_action, tensor_step.joint_action, atol=1.0e-6)


def test_order9_ppo_replay_restores_copied_environment_graph_frame(
    physical_model: PhysicalModel,
    morphology_distribution: RandomConnectedMorphologyDistribution,
) -> None:
    graph = morphology_distribution.sample(seed=916, module_count=3)
    local = _runtime(graph, physical_model)
    origin = (12.0, -8.0, 0.5)
    copied = RuntimeObservation.from_dict(local.to_dict())
    for state in copied.module_states:
        state.pose_world = tuple(
            [state.pose_world[index] + origin[index] for index in range(3)]
            + list(state.pose_world[3:])
        )
    trace = PolicyBehaviorTrace(
        policy_family="pi_l",
        policy_version=ORDER9_PI_L_POLICY_VERSION,
        action_semantics=ORDER9_PI_L_ACTION_SEMANTICS,
        action_payload={
            "actor_graph_frame_origin_world": list(origin),
            "actor_graph_joint_summary_semantics": (
                ORDER9_PI_L_GRAPH_JOINT_SUMMARY_NON_FIXED
            ),
        },
    )

    replay = _pi_l_actor_graph_observation(copied, trace, physical_model)
    active_joint_ids = {
        joint.joint_id
        for joint in physical_model.joints
        if joint.joint_type != "fixed"
    }

    for expected, actual in zip(local.module_states, replay.module_states):
        assert actual.pose_world == pytest.approx(expected.pose_world)
        assert set(actual.joint_positions) == active_joint_ids
        assert set(actual.joint_velocities) == active_joint_ids


def test_deterministic_and_stochastic_actions_have_consistent_log_prob_contract(
    physical_model: PhysicalModel,
    morphology_distribution: RandomConnectedMorphologyDistribution,
) -> None:
    graph = morphology_distribution.sample(seed=110, module_count=3)
    observation = _runtime(graph, physical_model)
    config = _small_config()
    torch.manual_seed(19)
    model = MorphologyConditionedActorCritic(config)
    features = torch.zeros((1, len(ORDER3_ACTOR_FEATURE_NAMES)))
    previous = torch.zeros((1, ORDER3_ACTION_SIZE))
    hidden = model.initial_state(1)

    first = model.step(
        [graph], [observation], features, previous, hidden, deterministic=True
    )
    second = model.step(
        [graph], [observation], features, previous, hidden, deterministic=True
    )
    assert torch.equal(first.action, first.action_mean)
    assert torch.equal(first.action, second.action)
    assert torch.equal(first.log_prob, second.log_prob)

    torch.manual_seed(20)
    stochastic_first = model.step(
        [graph], [observation], features, previous, hidden, deterministic=False
    )
    torch.manual_seed(21)
    stochastic_second = model.step(
        [graph], [observation], features, previous, hidden, deterministic=False
    )
    assert not torch.equal(stochastic_first.action, stochastic_second.action)
    assert bool(torch.isfinite(stochastic_first.log_prob).all().item())
    assert bool((stochastic_first.action.abs() <= 1.0).all().item())

    evaluated = model.step(
        [graph],
        [observation],
        features,
        previous,
        hidden,
        action=stochastic_first.action.detach(),
    )
    assert torch.allclose(evaluated.action, stochastic_first.action)
    assert torch.allclose(evaluated.log_prob, stochastic_first.log_prob, atol=1.0e-5)


def test_deployable_policy_passes_centroidal_pose_exactly_and_bounds_learned_fields(
    physical_model: PhysicalModel,
    morphology_distribution: RandomConnectedMorphologyDistribution,
) -> None:
    graph = morphology_distribution.sample(seed=120, module_count=2)
    context = _context(graph, physical_model, time_s=0.4)
    config = _small_config(
        trust_region_blend=0.25,
        linear_twist_correction_limit_mps=0.4,
        angular_twist_correction_limit_radps=0.6,
        residual_force_weight_fraction=0.1,
        residual_torque_per_module_nm=0.2,
    )
    model = _constant_action_model(config, twist_sign=1.0, wrench_sign=-1.0)
    policy = MorphologyConditionedLowLevelPolicy(
        model=model,
        physical_model=physical_model,
        config=config,
        deterministic=True,
    )
    control_model = RigidBodyControlModelBuilder().build(
        graph,
        physical_model,
        context.runtime_observation,
    )

    trace = policy.command_with_trace(context)
    command = trace.command

    assert trace.learned_policy_applied is True
    assert command.control_contract_version == POLICY_COMMAND_CONTRACT_CENTROIDAL
    assert context.active_knot is not None
    assert context.active_knot.centroidal_target is not None
    target = context.active_knot.centroidal_target
    assert target.com_pos_world is not None
    assert target.body_orientation_world is not None
    expected_pose = (*target.com_pos_world, *target.body_orientation_world)
    assert command.desired_body_pose == expected_pose

    baseline_twist = [*target.com_vel_world, 0.0, 0.0, 0.0]  # type: ignore[misc]
    assert command.desired_body_twist is not None
    twist_delta = [
        command.desired_body_twist[index] - baseline_twist[index]
        for index in range(6)
    ]
    twist_bounds = [
        *([config.trust_region_blend * config.linear_twist_correction_limit_mps] * 3),
        *([config.trust_region_blend * config.angular_twist_correction_limit_radps] * 3),
    ]
    assert all(abs(delta) <= bound + 1.0e-7 for delta, bound in zip(twist_delta, twist_bounds))
    assert all(delta > 0.99 * bound for delta, bound in zip(twist_delta, twist_bounds))

    assert command.residual_wrench_body is not None
    wrench_bounds = [
        *(
            [
                config.trust_region_blend
                * control_model.total_mass_kg
                * 9.81
                * config.residual_force_weight_fraction
            ]
            * 3
        ),
        *(
            [
                config.trust_region_blend
                * len(graph.modules)
                * config.residual_torque_per_module_nm
            ]
            * 3
        ),
    ]
    assert all(
        abs(value) <= bound + 1.0e-7
        for value, bound in zip(command.residual_wrench_body, wrench_bounds)
    )
    assert all(value < -0.99 * bound for value, bound in zip(command.residual_wrench_body, wrench_bounds))


def test_free_flight_outputs_absolute_dock_hold_only_and_clears_deprecated_contact_fields(
    physical_model: PhysicalModel,
    morphology_distribution: RandomConnectedMorphologyDistribution,
) -> None:
    graph = morphology_distribution.sample(seed=130, module_count=3)
    context = _context(graph, physical_model)
    config = _small_config(free_flight_joint_residual_enabled=False)
    policy = MorphologyConditionedLowLevelPolicy(
        model=_constant_action_model(config, twist_sign=0.0, wrench_sign=0.0),
        physical_model=physical_model,
        config=config,
    )

    command = policy.command(context)

    dock_joint_ids = _dock_joint_ids(physical_model)
    vectoring_joint_ids = {
        joint_id for rotor in physical_model.rotors for joint_id in rotor.vectoring_joint_ids
    }
    expected_positions = {
        f"module_{state.module_id}:{joint_id}": 0.0
        for state in context.runtime_observation.module_states
        for joint_id in dock_joint_ids
    }
    assert command.joint_position_targets == expected_positions
    assert command.joint_velocity_targets == {key: 0.0 for key in expected_positions}
    assert command.joint_torque_bias == {key: 0.0 for key in expected_positions}
    assert not any(
        global_id.partition(":")[2] in vectoring_joint_ids
        for global_id in command.joint_position_targets
    )
    assert command.desired_anchor_pose_offsets == {}
    assert command.joint_position_bias == {}
    assert command.joint_velocity_bias == {}
    assert command.contact_tracking_bias == {}
    assert not hasattr(command, "internal_wrench_bias")


def test_contact_wrench_is_private_to_actor_but_privileged_critic_input_does_not_change_action(
    physical_model: PhysicalModel,
    morphology_distribution: RandomConnectedMorphologyDistribution,
) -> None:
    graph = morphology_distribution.sample(seed=140, module_count=4)
    first = _runtime(graph, physical_model, contact_wrench_scale=1.0)
    second = _runtime(graph, physical_model, contact_wrench_scale=1_000.0)
    control_builder = RigidBodyControlModelBuilder()
    first_control = control_builder.build(graph, physical_model, first)
    second_control = control_builder.build(graph, physical_model, second)
    target_pose = (0.4, -0.2, 1.8, 0.0, 0.0, 0.0, 1.0)
    target_twist = [0.0] * 6

    first_features = order3_actor_feature_vector(
        first,
        first_control,
        target_pose_world=target_pose,
        target_twist=target_twist,
    )
    second_features = order3_actor_feature_vector(
        second,
        second_control,
        target_pose_world=target_pose,
        target_twist=target_twist,
    )
    assert first_features == second_features

    config = _small_config()
    torch.manual_seed(44)
    model = MorphologyConditionedActorCritic(config)
    hidden = model.initial_state(1)
    previous = torch.zeros((1, ORDER3_ACTION_SIZE))
    first_step = model.step(
        [graph],
        [first],
        torch.tensor([first_features]),
        previous,
        hidden,
        privileged_disturbance_body=torch.zeros((1, 6)),
        deterministic=True,
    )
    second_step = model.step(
        [graph],
        [second],
        torch.tensor([second_features]),
        previous,
        hidden,
        privileged_disturbance_body=torch.full((1, 6), 100.0),
        deterministic=True,
    )
    assert torch.equal(first_step.action, second_step.action)
    assert torch.equal(first_step.action_mean, second_step.action_mean)
    assert torch.equal(first_step.recurrent_state, second_step.recurrent_state)


def test_gru_state_carries_between_steps_and_resets_when_episode_time_rewinds(
    physical_model: PhysicalModel,
    morphology_distribution: RandomConnectedMorphologyDistribution,
) -> None:
    graph = morphology_distribution.sample(seed=150, module_count=2)
    config = _small_config()
    torch.manual_seed(51)
    policy = MorphologyConditionedLowLevelPolicy(
        model=MorphologyConditionedActorCritic(config),
        physical_model=physical_model,
        config=config,
    )

    first = policy.command_with_trace(_context(graph, physical_model, time_s=0.5))
    second = policy.command_with_trace(_context(graph, physical_model, time_s=0.6))
    rewind = policy.command_with_trace(_context(graph, physical_model, time_s=0.1))

    assert first.learned_policy_applied and second.learned_policy_applied and rewind.learned_policy_applied
    assert second.recurrent_state_in == pytest.approx(first.recurrent_state_out)
    assert any(abs(value) > 1.0e-8 for value in second.recurrent_state_in)
    assert rewind.recurrent_state_in == pytest.approx([0.0] * config.recurrent_hidden_dim)

    policy.reset()
    after_explicit_reset = policy.command_with_trace(
        _context(graph, physical_model, time_s=0.7)
    )
    assert after_explicit_reset.recurrent_state_in == pytest.approx(
        [0.0] * config.recurrent_hidden_dim
    )


def test_controller_and_actor_feature_ood_use_strict_v2_fallback(
    physical_model: PhysicalModel,
    morphology_distribution: RandomConnectedMorphologyDistribution,
) -> None:
    graph = morphology_distribution.sample(seed=160, module_count=2)
    config = _small_config()
    policy = MorphologyConditionedLowLevelPolicy(
        model=MorphologyConditionedActorCritic(config),
        physical_model=physical_model,
        config=config,
    )
    infeasible_status = ControllerStatus(status="infeasible", qp_feasible=False)
    infeasible = policy.command_with_trace(
        _context(
            graph,
            physical_model,
            controller_status=infeasible_status,
        )
    )

    assert infeasible.learned_policy_applied is False
    assert infeasible.fallback_reason == "controller_infeasible"
    assert infeasible.command.control_contract_version == POLICY_COMMAND_CONTRACT_CENTROIDAL
    assert infeasible.command.joint_position_targets
    assert set(infeasible.command.joint_position_targets.values()) == {0.0}
    assert infeasible.command.contact_tracking_bias == {}

    ood_config = _small_config(ood_absolute_feature_limit=1.0e-5)
    ood_policy = MorphologyConditionedLowLevelPolicy(
        model=MorphologyConditionedActorCritic(ood_config),
        physical_model=physical_model,
        config=ood_config,
    )
    ood = ood_policy.command_with_trace(_context(graph, physical_model, time_s=0.4))
    assert ood.learned_policy_applied is False
    assert ood.fallback_reason == "actor_feature_ood"
    assert ood.command.control_contract_version == POLICY_COMMAND_CONTRACT_CENTROIDAL
    assert ood.command.joint_position_targets
    assert set(ood.command.joint_position_targets.values()) == {0.0}


def test_checkpoint_rejects_legacy_roundtrips_v2_and_rejects_contract_tampering(
    tmp_path: Path,
    physical_model: PhysicalModel,
) -> None:
    legacy_path = tmp_path / "legacy_v1.pt"
    torch.save(
        {
            "checkpoint_version": "p4_3_pi_l_checkpoint_v1",
            "state_dict": {},
        },
        legacy_path,
    )
    with pytest.raises(SchemaValidationError, match="not the centroidal morphology-conditioned"):
        load_order3_policy_checkpoint(legacy_path)

    config = _small_config()
    torch.manual_seed(71)
    model = MorphologyConditionedActorCritic(config)
    metadata = _checkpoint_metadata(config, physical_model)
    checkpoint_path = tmp_path / "order3_v2.pt"
    saved_hash = save_order3_policy_checkpoint(
        checkpoint_path,
        model=model,
        metadata=metadata,
    )
    loaded = load_order3_policy_checkpoint(checkpoint_path)
    assert saved_hash == hash_file(checkpoint_path)
    assert loaded.sha256 == saved_hash
    assert loaded.config.to_dict() == config.to_dict()
    assert loaded.metadata.to_dict() == metadata.to_dict()
    for name, expected in model.state_dict().items():
        assert torch.equal(loaded.model.state_dict()[name], expected)

    verified_policy = MorphologyConditionedLowLevelPolicy.from_checkpoint(
        checkpoint_path,
        physical_model=physical_model,
        expected_sha256=saved_hash,
    )
    assert verified_policy.config.to_dict() == config.to_dict()
    mismatched_fallback = BaselineLowLevelPolicy(
        BaselineLowLevelPolicyConfig(
            control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
            residual_force_limit_n=3.5,
        )
    )
    with pytest.raises(SchemaValidationError, match="fallback config hash"):
        MorphologyConditionedLowLevelPolicy.from_checkpoint(
            checkpoint_path,
            physical_model=physical_model,
            expected_sha256=saved_hash,
            baseline_policy=mismatched_fallback,
        )
    with pytest.raises(SchemaValidationError, match="sha256 mismatch"):
        load_order3_policy_checkpoint(
            checkpoint_path,
            expected_sha256="0" * 64,
        )

    weight_tamper = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    weight_name = next(iter(weight_tamper["state_dict"]))
    weight_tamper["state_dict"][weight_name] = (
        weight_tamper["state_dict"][weight_name].clone() + 1.0e-3
    )
    weight_tamper_path = tmp_path / "weight-tampered.pt"
    torch.save(weight_tamper, weight_tamper_path)
    with pytest.raises(SchemaValidationError, match="sha256 mismatch"):
        load_order3_policy_checkpoint(
            weight_tamper_path,
            expected_sha256=saved_hash,
        )

    mismatched_model = replace(
        physical_model,
        aggregate_mass_kg=physical_model.aggregate_mass_kg + 0.01,
    )
    with pytest.raises(SchemaValidationError, match="PhysicalModel hash"):
        MorphologyConditionedLowLevelPolicy.from_checkpoint(
            checkpoint_path,
            physical_model=mismatched_model,
            expected_sha256=saved_hash,
        )

    alternate_urdf_path = tmp_path / "alternate.urdf"
    alternate_urdf_path.write_text(
        Path(physical_model.urdf_path).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    alternate_model = replace(physical_model, urdf_path=str(alternate_urdf_path))
    alternate_metadata = replace(
        metadata,
        physical_model_hash=alternate_model.stable_hash(),
        urdf_hash="intentionally-wrong-urdf-hash",
    )
    alternate_checkpoint_path = tmp_path / "alternate-model.pt"
    alternate_hash = save_order3_policy_checkpoint(
        alternate_checkpoint_path,
        model=model,
        metadata=alternate_metadata,
    )
    with pytest.raises(SchemaValidationError, match="URDF hash"):
        MorphologyConditionedLowLevelPolicy.from_checkpoint(
            alternate_checkpoint_path,
            physical_model=alternate_model,
            expected_sha256=alternate_hash,
        )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload["metadata"]["encoder_version"] = "tampered_encoder"
    tampered_path = tmp_path / "tampered.pt"
    torch.save(payload, tampered_path)
    with pytest.raises(SchemaValidationError, match="encoder_version"):
        load_order3_policy_checkpoint(tampered_path)

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload["unexpected"] = True
    extra_key_path = tmp_path / "extra-key.pt"
    torch.save(payload, extra_key_path)
    with pytest.raises(SchemaValidationError, match="keys do not match"):
        load_order3_policy_checkpoint(extra_key_path)


def test_checkpoint_structural_allowlist_falls_back_for_valid_unseen_graph(
    tmp_path: Path,
    physical_model: PhysicalModel,
    morphology_distribution: RandomConnectedMorphologyDistribution,
) -> None:
    allowed_graph = morphology_distribution.sample(seed=810, module_count=3)
    unseen_graph = morphology_distribution.sample(seed=811, module_count=3)
    assert morphology_structural_hash(allowed_graph) != morphology_structural_hash(
        unseen_graph
    )
    config = _small_config()
    model = _constant_action_model(config, twist_sign=1.0, wrench_sign=1.0)
    metadata = _checkpoint_metadata(
        config,
        physical_model,
        morphology_hashes=[morphology_structural_hash(allowed_graph)],
    )
    path = tmp_path / "allowlisted.pt"
    expected_hash = save_order3_policy_checkpoint(path, model=model, metadata=metadata)
    policy = MorphologyConditionedLowLevelPolicy.from_checkpoint(
        path,
        physical_model=physical_model,
        expected_sha256=expected_hash,
    )

    allowed = policy.command_with_trace(_context(allowed_graph, physical_model))
    assert allowed.learned_policy_applied is True
    policy.reset()
    unseen = policy.command_with_trace(_context(unseen_graph, physical_model))
    assert unseen.learned_policy_applied is False
    assert unseen.fallback_reason == "structural_hash_ood"


def test_privileged_critic_input_changes_value_without_changing_actor_outputs(
    physical_model: PhysicalModel,
    morphology_distribution: RandomConnectedMorphologyDistribution,
) -> None:
    graph = morphology_distribution.sample(seed=910, module_count=3)
    observation = _runtime(graph, physical_model)
    config = _small_config()
    model = MorphologyConditionedActorCritic(config)
    with torch.no_grad():
        model.critic[0].weight.zero_()
        model.critic[0].bias.zero_()
        model.critic[0].weight[:, -6:] = 0.1
        model.critic[2].weight.fill_(0.1)
        model.critic[2].bias.zero_()
    actor_features = torch.zeros((1, len(ORDER3_ACTOR_FEATURE_NAMES)))
    previous_action = torch.zeros((1, ORDER3_ACTION_SIZE))
    recurrent = model.initial_state(1)
    nominal = model.step(
        [graph],
        [observation],
        actor_features,
        previous_action,
        recurrent,
        privileged_disturbance_body=torch.zeros((1, 6)),
        deterministic=True,
    )
    disturbed = model.step(
        [graph],
        [observation],
        actor_features,
        previous_action,
        recurrent,
        privileged_disturbance_body=torch.ones((1, 6)),
        deterministic=True,
    )

    assert torch.equal(nominal.action, disturbed.action)
    assert torch.equal(nominal.action_mean, disturbed.action_mean)
    assert torch.equal(nominal.recurrent_state, disturbed.recurrent_state)
    assert torch.equal(nominal.log_prob, disturbed.log_prob)
    assert not torch.equal(nominal.value, disturbed.value)


def test_order9_runtime_loads_strict_checkpoint_and_preserves_exact_action_trace(
    tmp_path: Path,
    physical_model: PhysicalModel,
    morphology_distribution: RandomConnectedMorphologyDistribution,
) -> None:
    config = load_order9_learning_config()
    stage = order9_stage_by_id(config, "c1_pi_l_bc_fixed_nominal")
    policy = Order9PhaseConditionedActorCritic(
        Order9LowLevelPolicyConfig(
            graph_hidden_dim=16,
            graph_message_layers=1,
            recurrent_hidden_dim=24,
            max_local_joint_slots=4,
        )
    )
    metadata = build_order9_checkpoint_metadata(
        policy,
        stage=stage,
        schedule_hash=order9_schedule_hash(config),
        physical_model_hash=physical_model.stable_hash(),
        git_revision="unit-test",
        random_seed=19,
        input_artifact_hashes={"unit": "a" * 64},
        parent_checkpoint_sha256=None,
        source_order3_checkpoint_sha256=None,
        metrics={"loss": 1.0},
        trainer_version="unit_test",
    )
    checkpoint = tmp_path / "order9-pi-l.pt"
    checkpoint_sha = save_order9_policy_checkpoint(
        checkpoint,
        model=policy,
        metadata=metadata,
    )
    runtime = Order9LowLevelRuntimePolicy.from_checkpoint(
        checkpoint,
        physical_model=physical_model,
        expected_sha256=checkpoint_sha,
        expected_schedule_hash=order9_schedule_hash(config),
        deterministic=False,
    )
    graph = morphology_distribution.sample(seed=919, module_count=3)
    context = replace(
        _context(graph, physical_model),
        task_type="object_grasp_carry",
        task_adapter_id="object_grasp_carry_v1",
        phase_index=2,
        phase_count=11,
    )

    inference = runtime.command_with_trace(context)
    trace = order9_pi_l_behavior_trace_from_inference(
        inference,
        checkpoint_sha256=checkpoint_sha,
    )

    assert inference.learned_policy_applied
    assert inference.module_ids == sorted(module.module_id for module in graph.modules)
    assert trace.stochastic
    assert trace.policy_checkpoint_sha256 == checkpoint_sha
    assert trace.action_payload["module_ids"] == inference.module_ids
    assert trace.action_payload["joint_action"] == inference.normalized_joint_action


def _small_config(**overrides) -> Order3MorphologyConditionedPolicyConfig:
    config = Order3MorphologyConditionedPolicyConfig(
        graph_hidden_dim=16,
        graph_message_layers=1,
        recurrent_hidden_dim=24,
        max_local_joint_slots=4,
    )
    return replace(config, **overrides)


def _constant_action_model(
    config: Order3MorphologyConditionedPolicyConfig,
    *,
    twist_sign: float,
    wrench_sign: float,
) -> MorphologyConditionedActorCritic:
    model = MorphologyConditionedActorCritic(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.actor_mean.bias[:6].fill_(20.0 * twist_sign)
        model.actor_mean.bias[6:].fill_(20.0 * wrench_sign)
    return model


def _context(
    graph: MorphologyGraph,
    physical_model: PhysicalModel,
    *,
    time_s: float = 0.25,
    controller_status: ControllerStatus | None = None,
) -> LowLevelPolicyContext:
    status = controller_status or ControllerStatus(
        status="ok",
        qp_feasible=True,
        metrics={"allocation_residual_norm": 0.01},
    )
    observation = _runtime(
        graph,
        physical_model,
        time_s=time_s,
        controller_status=status,
    )
    knot = InteractionKnot(
        t_rel_s=0.0,
        contact_assignments=[],
        centroidal_target=CentroidalTarget(
            com_pos_world=(0.35, -0.15, 1.75),
            com_vel_world=(0.12, -0.08, 0.05),
            body_orientation_world=(0.0, 0.0, math.sin(0.1), math.cos(0.1)),
        ),
    )
    trajectory = ContactWrenchTrajectory(
        horizon_s=1.0,
        dt_s=0.02,
        knots=[knot],
        derived_mode_label="order3_unit_free_flight",
    )
    return LowLevelPolicyContext(
        runtime_observation=observation,
        morphology_graph=graph,
        physical_model=physical_model,
        contact_wrench_trajectory=trajectory,
        active_knot=knot,
        controller_status=status,
    )


def _runtime(
    graph: MorphologyGraph,
    physical_model: PhysicalModel,
    *,
    time_s: float = 0.25,
    controller_status: ControllerStatus | None = None,
    contact_wrench_scale: float | None = None,
) -> RuntimeObservation:
    joint_ids = [joint.joint_id for joint in physical_model.joints]
    dock_ids = set(_dock_joint_ids(physical_model))
    states: list[ModuleRuntimeState] = []
    for module in graph.modules:
        design_pose = module.pose_in_design_frame
        positions = {
            joint_id: (
                0.01 * (module.module_id + 1) * (index + 1)
                if joint_id in dock_ids
                else 0.0
            )
            for index, joint_id in enumerate(joint_ids)
        }
        states.append(
            ModuleRuntimeState(
                module_id=module.module_id,
                pose_world=(
                    0.2 + float(design_pose[0]),
                    -0.1 + float(design_pose[1]),
                    1.3 + float(design_pose[2]),
                    *design_pose[3:],
                ),
                twist_world=[0.01, -0.02, 0.03, 0.01, 0.02, -0.01],
                joint_positions=positions,
                joint_velocities={joint_id: 0.0 for joint_id in joint_ids},
                health=0.98,
            )
        )
    contacts = []
    if contact_wrench_scale is not None:
        contacts.append(
            ContactState(
                contact_id="privileged-contact",
                entity_a="robot",
                entity_b="floor",
                wrench_world=[
                    contact_wrench_scale,
                    -2.0 * contact_wrench_scale,
                    3.0 * contact_wrench_scale,
                    0.1 * contact_wrench_scale,
                    -0.2 * contact_wrench_scale,
                    0.3 * contact_wrench_scale,
                ],
            )
        )
    return RuntimeObservation(
        time_s=time_s,
        morphology_graph=graph,
        module_states=states,
        object_states=[],
        contact_states=contacts,
        controller_status=controller_status
        or ControllerStatus(
            status="ok",
            qp_feasible=True,
            metrics={"allocation_residual_norm": 0.01},
        ),
        task_progress=TaskProgressState(progress_ratio=0.2),
    )


def _dock_joint_ids(physical_model: PhysicalModel) -> list[str]:
    return sorted(
        {
            str(port.mechanical_limits["mechanism_joint_id"])
            for port in physical_model.dock_ports
            if port.mechanical_limits.get("mechanism_joint_id")
        }
    )


def _checkpoint_metadata(
    config: Order3MorphologyConditionedPolicyConfig,
    physical_model: PhysicalModel,
    *,
    morphology_hashes: list[str] | None = None,
) -> Order3PolicyCheckpointMetadata:
    return Order3PolicyCheckpointMetadata(
        checkpoint_version=ORDER3_CHECKPOINT_VERSION,
        policy_family=ORDER3_POLICY_FAMILY,
        policy_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
        architecture_version=ORDER3_POLICY_ARCHITECTURE_VERSION,
        tensorizer_version=ORDER3_TENSORIZER_VERSION,
        encoder_version=ORDER3_ENCODER_VERSION,
        training_stage="bc",
        action_names=list(ORDER3_ACTION_NAMES),
        actor_feature_schema_hash=order3_actor_feature_schema_hash(),
        graph_feature_schema_hash=order3_graph_feature_schema_hash(),
        config_hash=config.stable_hash(),
        pool_hash="unit-pool-hash",
        dataset_hash="unit-dataset-hash",
        physical_model_hash=physical_model.stable_hash(),
        urdf_hash=hash_file(physical_model.urdf_path),
        controller_contract_hash="unit-controller-contract-hash",
        fallback_version=ORDER3_FALLBACK_VERSION,
        fallback_config_hash=stable_hash(
            BaselineLowLevelPolicyConfig(
                control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
            )
        ),
        seed=71,
        git_revision="unit-test-revision",
        metadata={
            "morphology_hashes": {
                "train": list(morphology_hashes or ["a" * 64]),
                "validation": [],
                "held_out": [],
            }
        },
    )
