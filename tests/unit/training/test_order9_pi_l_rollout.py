from __future__ import annotations

from dataclasses import replace

import torch

from amsrr.morphology.random_connected import RandomConnectedMorphologyDistribution
from amsrr.policies.low_level_policy_base import LowLevelPolicyContext
from amsrr.policies.morphology_conditioned_low_level_policy import (
    MorphologyConditionedLowLevelPolicy,
)
from amsrr.policies.order9_low_level_policy import (
    Order9LowLevelPolicyConfig,
    Order9PhaseConditionedActorCritic,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.policies import (
    POLICY_COMMAND_CONTRACT_CENTROIDAL,
    CentroidalTarget,
    ContactWrenchTrajectory,
    ControllerCommand,
    ControllerStatus,
    InteractionKnot,
)
from amsrr.schemas.runtime import (
    ModuleRuntimeState,
    ObjectRuntimeState,
    RuntimeObservation,
    TaskProgressState,
)
from amsrr.training.order9_pi_l_rollout import Order9PiLEpisodeCollector
from amsrr.training.order9_teacher import build_order8_grasp_carry_task_spec


CHECKPOINT_SHA = "7" * 64


def test_pi_l_collector_splits_on_fallback_and_never_credits_it() -> None:
    task, model, graph, policy, trajectory = _runtime_components()
    collector = Order9PiLEpisodeCollector(
        physical_episode_id="physical-episode-0",
        task_spec=task,
        split=DatasetSplit.TRAIN,
        checkpoint_sha256=CHECKPOINT_SHA,
    )

    actor0, reward0 = _observation_pair(graph, model, time_s=0.00, x=0.50)
    collector.observe_state(actor_observation=actor0, reward_observation=reward0)
    context0 = _context(actor0, graph, model, trajectory)
    inference0 = policy.command_with_trace(context0)
    assert collector.record_action(
        context=context0,
        inference=inference0,
        trajectory_record_id="pi-h-trajectory-0",
        active_trajectory_index=0,
        active_knot_index=0,
        controller_command=_controller(actor0.controller_status),
        actuator_target_record={},
    )

    actor1, reward1 = _observation_pair(graph, model, time_s=0.02, x=0.51)
    first = collector.observe_state(
        actor_observation=actor1,
        reward_observation=reward1,
    )
    assert first is not None and not first.terminal

    fallback = replace(
        policy.command_with_trace(_context(actor1, graph, model, trajectory)),
        learned_policy_applied=False,
        fallback_reason="synthetic_safety_intervention",
    )
    assert not collector.record_action(
        context=_context(actor1, graph, model, trajectory),
        inference=fallback,
        trajectory_record_id="pi-h-fallback",
        active_trajectory_index=0,
        active_knot_index=0,
        controller_command=_controller(actor1.controller_status),
        actuator_target_record={},
    )
    assert first.terminal
    assert first.reward_terms["segment_ended_by_pi_l_fallback"] == 1.0

    actor2, reward2 = _observation_pair(graph, model, time_s=0.04, x=0.52)
    collector.observe_state(actor_observation=actor2, reward_observation=reward2)
    context2 = _context(actor2, graph, model, trajectory)
    inference2 = policy.command_with_trace(context2)
    assert collector.record_action(
        context=context2,
        inference=inference2,
        trajectory_record_id="pi-h-trajectory-1",
        active_trajectory_index=1,
        active_knot_index=0,
        controller_command=_controller(actor2.controller_status),
        actuator_target_record={},
    )
    actor3, reward3 = _observation_pair(graph, model, time_s=0.06, x=0.53)
    collector.observe_state(actor_observation=actor3, reward_observation=reward3)
    result = collector.finalize(
        terminal=True,
        release_valid=True,
        object_dropped=False,
        hard_collision=False,
        timeout=False,
        qp_infeasible_terminal=False,
    )

    assert result.learned_action_count == 2
    assert result.fallback_action_count == 1
    assert result.gae_segment_count == 2
    assert len(result.records) == 2
    assert result.records[0].episode_id != result.records[1].episode_id
    assert all(record.terminal for record in result.records)
    assert all(record.behavior_trace.stochastic for record in result.records)
    assert all(
        record.behavior_trace.policy_checkpoint_sha256 == CHECKPOINT_SHA
        for record in result.records
    )
    assert result.terminal_reward_credited_to_actor


def test_pi_l_terminal_reward_is_not_credited_when_fallback_owns_last_action() -> None:
    task, model, graph, policy, trajectory = _runtime_components()
    collector = Order9PiLEpisodeCollector(
        physical_episode_id="physical-episode-fallback-tail",
        task_spec=task,
        split=DatasetSplit.TRAIN,
        checkpoint_sha256=CHECKPOINT_SHA,
    )
    actor0, reward0 = _observation_pair(graph, model, time_s=0.00, x=0.50)
    collector.observe_state(actor_observation=actor0, reward_observation=reward0)
    context0 = _context(actor0, graph, model, trajectory)
    inference0 = policy.command_with_trace(context0)
    collector.record_action(
        context=context0,
        inference=inference0,
        trajectory_record_id="pi-h-trajectory",
        active_trajectory_index=0,
        active_knot_index=0,
        controller_command=_controller(actor0.controller_status),
        actuator_target_record={},
    )
    actor1, reward1 = _observation_pair(graph, model, time_s=0.02, x=0.51)
    learned = collector.observe_state(
        actor_observation=actor1,
        reward_observation=reward1,
    )
    fallback = replace(
        policy.command_with_trace(_context(actor1, graph, model, trajectory)),
        learned_policy_applied=False,
        fallback_reason="synthetic_terminal_fallback",
    )
    collector.record_action(
        context=_context(actor1, graph, model, trajectory),
        inference=fallback,
        trajectory_record_id="pi-h-fallback",
        active_trajectory_index=0,
        active_knot_index=0,
        controller_command=_controller(actor1.controller_status),
        actuator_target_record={},
    )
    actor2, reward2 = _observation_pair(graph, model, time_s=0.04, x=0.52)
    collector.observe_state(actor_observation=actor2, reward_observation=reward2)
    before_finalize = learned.reward
    result = collector.finalize(
        terminal=True,
        release_valid=True,
        object_dropped=False,
        hard_collision=False,
        timeout=False,
        qp_infeasible_terminal=False,
    )

    assert result.records == (learned,)
    assert learned.terminal
    assert learned.reward == before_finalize
    assert result.terminal_reward_credited_to_actor is False


def _runtime_components():
    physical_model = build_physical_model_from_config(
        "configs/robot/robot_model.yaml"
    )
    graph = RandomConnectedMorphologyDistribution(physical_model).sample(
        seed=9019,
        module_count=2,
    )
    task = build_order8_grasp_carry_task_spec(
        object_pose_world=(0.5, 0.0, 0.225, 0.0, 0.0, 0.0, 1.0),
        object_size_m=(0.30, 0.40, 0.15),
        object_mass_kg=1.0,
        object_friction=0.6,
        required_transport_distance_m=0.20,
        support_height_m=0.15,
        max_contact_force_n=30.0,
        max_contact_torque_nm=5.0,
    )
    knot = InteractionKnot(
        t_rel_s=0.0,
        contact_assignments=[],
        centroidal_target=CentroidalTarget(
            com_pos_world=(0.5, 0.0, 1.0),
            com_vel_world=(0.0, 0.0, 0.0),
            body_orientation_world=(0.0, 0.0, 0.0, 1.0),
        ),
    )
    trajectory = ContactWrenchTrajectory(
        horizon_s=0.1,
        dt_s=0.1,
        knots=[knot],
    )
    config = Order9LowLevelPolicyConfig(
        graph_hidden_dim=16,
        graph_message_layers=1,
        recurrent_hidden_dim=24,
        max_local_joint_slots=4,
    )
    torch.manual_seed(9019)
    policy = MorphologyConditionedLowLevelPolicy(
        model=Order9PhaseConditionedActorCritic(config),
        physical_model=physical_model,
        config=config,
        deterministic=False,
    )
    return task, physical_model, graph, policy, trajectory


def _context(observation, graph, physical_model, trajectory):
    return LowLevelPolicyContext(
        runtime_observation=observation,
        morphology_graph=graph,
        physical_model=physical_model,
        contact_wrench_trajectory=trajectory,
        active_knot=trajectory.knots[0],
        controller_status=observation.controller_status,
        task_type="object_grasp_carry",
        task_adapter_id="object_grasp_carry_v1",
        phase_index=4,
        phase_count=11,
    )


def _observation_pair(graph, physical_model, *, time_s: float, x: float):
    status = ControllerStatus(
        status="ok",
        qp_feasible=True,
        metrics={"allocation_residual_norm": 0.0},
    )
    joint_ids = [joint.joint_id for joint in physical_model.joints]
    modules = [
        ModuleRuntimeState(
            module_id=module.module_id,
            pose_world=module.pose_in_design_frame,
            twist_world=[0.0] * 6,
            joint_positions={joint_id: 0.0 for joint_id in joint_ids},
            joint_velocities={joint_id: 0.0 for joint_id in joint_ids},
            health=1.0,
        )
        for module in graph.modules
    ]
    objects = [
        ObjectRuntimeState(
            object_id="order8_object",
            pose_world=(x, 0.0, 0.225, 0.0, 0.0, 0.0, 1.0),
            twist_world=[0.0] * 6,
        )
    ]
    actor = RuntimeObservation(
        time_s=time_s,
        morphology_graph=graph,
        module_states=modules,
        object_states=objects,
        contact_states=[],
        controller_status=status,
        task_progress=TaskProgressState(
            phase_label="transport",
            progress_ratio=0.5,
            metrics={},
        ),
    )
    reward = replace(
        actor,
        task_progress=TaskProgressState(
            phase_label="transport",
            progress_ratio=0.5,
            metrics={
                "grasp_data_available": 1.0,
                "grasp_maintenance": 1.0,
                "collision_data_available": 1.0,
                "hard_collision": 0.0,
                "slip_data_available": 1.0,
                "slip": 0.0,
            },
        ),
    )
    return actor, reward


def _controller(status: ControllerStatus) -> ControllerCommand:
    return ControllerCommand(
        rotor_thrusts_n={},
        vectoring_joint_targets={},
        joint_torque_commands={},
        dock_mechanism_commands={},
        controller_status=status,
        control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
    )
