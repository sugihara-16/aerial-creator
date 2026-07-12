from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from amsrr.controllers.isaac_controller_bridge import IsaacActuatorTargetRecord
from amsrr.controllers.rigid_body_model import RigidBodyControlModelBuilder
from amsrr.morphology.random_connected import (
    RandomConnectedMorphologyDistribution,
    morphology_structural_hash,
)
from amsrr.policies.morphology_conditioned_low_level_policy import (
    MorphologyConditionedActorCritic,
    Order3MorphologyConditionedPolicyConfig,
    order3_actor_feature_schema_hash,
    order3_actor_feature_vector,
    order3_graph_feature_schema_hash,
    save_order3_policy_checkpoint,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.order3 import (
    ORDER3_ACTION_NAMES,
    ORDER3_CHECKPOINT_VERSION,
    ORDER3_ENCODER_VERSION,
    ORDER3_FALLBACK_VERSION,
    ORDER3_POLICY_ARCHITECTURE_VERSION,
    ORDER3_POLICY_FAMILY,
    ORDER3_TENSORIZER_VERSION,
    Order3PolicyCheckpointMetadata,
)
from amsrr.schemas.order3_rollout_condition import build_order3_rollout_condition
from amsrr.schemas.policies import (
    POLICY_COMMAND_CONTRACT_CENTROIDAL,
    ControllerCommand,
    ControllerStatus,
    PolicyCommand,
)
from amsrr.schemas.runtime import (
    ContactState,
    ModuleRuntimeState,
    RuntimeObservation,
    TaskProgressState,
)
from amsrr.training.order3_free_flight import (
    Order3FreeFlightRewardConfig,
    Order3TaskMode,
)
from amsrr.training.order3_online_collector import (
    Order3OnlineCollectorConfig,
    collect_order3_online_transitions,
)

CHECKPOINT_HASH = "a" * 64
_FINAL_BOOTSTRAP_KEY = "order3_pi_l_final_bootstrap_value"


def _source(*, contact_wrench: bool = False):
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    morphology = RandomConnectedMorphologyDistribution(physical_model).sample(
        seed=17,
        module_count=2,
    )
    times = [0.0, 0.1, 0.2, 0.3]
    observations: list[RuntimeObservation] = []
    for time_s in [*times, 0.4]:
        module_states = [
            ModuleRuntimeState(
                module_id=module.module_id,
                pose_world=(
                    module.pose_in_design_frame[0],
                    module.pose_in_design_frame[1],
                    module.pose_in_design_frame[2] + 1.0,
                    *module.pose_in_design_frame[3:7],
                ),
                twist_world=[0.0] * 6,
            )
            for module in morphology.modules
        ]
        contacts = []
        if contact_wrench and time_s in {0.0, 0.4}:
            contacts = [
                ContactState(
                    contact_id=f"floor-{time_s}",
                    entity_a=morphology.graph_id,
                    entity_b="floor",
                    wrench_world=[0.0, 0.0, 8.0, 0.0, 0.0, 0.0],
                    metadata={"source": "isaac_lab_contact_sensor"},
                )
            ]
        observations.append(
            RuntimeObservation(
                time_s=time_s,
                morphology_graph=morphology,
                module_states=module_states,
                object_states=[],
                contact_states=contacts,
                controller_status=ControllerStatus(
                    status="ok",
                    qp_feasible=True,
                    active_mode="rigid_body_qp",
                ),
                task_progress=TaskProgressState(),
            )
        )
    target_pose = (
        RigidBodyControlModelBuilder()
        .build(
            morphology,
            physical_model,
            observations[0],
        )
        .body_pose_world
    )
    command_a = PolicyCommand(
        desired_body_pose=target_pose,
        desired_body_twist=[0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
        residual_wrench_body=[0.0] * 6,
        control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
    )
    command_b = PolicyCommand(
        desired_body_pose=target_pose,
        desired_body_twist=[-0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
        residual_wrench_body=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
    )
    policy_commands = [command_a, command_a, command_b, command_b]
    controller_commands = [
        ControllerCommand(
            rotor_thrusts_n={"module_0:rotor_1": 5.0},
            vectoring_joint_targets={},
            joint_torque_commands={},
            dock_mechanism_commands={},
            controller_status=ControllerStatus(
                status="ok",
                qp_feasible=True,
                active_mode="rigid_body_qp",
            ),
            control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
        )
        for _ in times
    ]
    actuator_records = [
        IsaacActuatorTargetRecord(
            time_s=time_s,
            backend="isaac_lab",
            morphology_graph_id=morphology.graph_id,
            command_index=index,
            actuator_targets=[],
            qp_status="ok",
        )
        for index, time_s in enumerate(times)
    ]
    action_a = [0.1] * len(ORDER3_ACTION_NAMES)
    action_b = [-0.2] * len(ORDER3_ACTION_NAMES)
    metadata = Order3PolicyCheckpointMetadata(
        checkpoint_version=ORDER3_CHECKPOINT_VERSION,
        policy_family=ORDER3_POLICY_FAMILY,
        policy_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
        architecture_version=ORDER3_POLICY_ARCHITECTURE_VERSION,
        tensorizer_version=ORDER3_TENSORIZER_VERSION,
        encoder_version=ORDER3_ENCODER_VERSION,
        training_stage="bc",
        action_names=list(ORDER3_ACTION_NAMES),
        actor_feature_schema_hash="actor",
        graph_feature_schema_hash="graph",
        config_hash="config",
        pool_hash="pool",
        dataset_hash="dataset",
        physical_model_hash=physical_model.stable_hash(),
        urdf_hash="urdf",
        controller_contract_hash="controller",
        fallback_version=ORDER3_FALLBACK_VERSION,
        fallback_config_hash="fallback",
        seed=1,
        git_revision="revision",
    )
    report = {
        "spawn_passed": True,
        "isaac_backed": True,
        "command_returncode": 0,
        "random_morphology_takeoff_smoke": True,
        "random_morphology_takeoff_smoke_passed": True,
        "random_morphology_takeoff_graph_id": morphology.graph_id,
        "random_morphology_takeoff_morphology_hash": morphology.stable_hash(),
        "random_morphology_takeoff_module_count": 2,
        "random_morphology_takeoff_physical_model_hash": physical_model.stable_hash(),
        "random_morphology_takeoff_learned_policy_used": True,
        "random_morphology_takeoff_controller": (
            "order3_morphology_conditioned_pi_l_plus_deterministic_qpid"
        ),
        "random_morphology_takeoff_control_contract_version": (
            POLICY_COMMAND_CONTRACT_CENTROIDAL
        ),
        "random_morphology_takeoff_tracking_state_source": (
            "true_morphology_centroidal_frame"
        ),
        "random_morphology_takeoff_true_centroidal_tracking": True,
        "random_morphology_takeoff_contact_wrench_tracking_claim": False,
        "random_morphology_takeoff_internal_wrench_tracking_claim": False,
        "random_morphology_takeoff_qp_actuator_variable_scope": (
            "rotor_thrust_vectoring_and_slack_only"
        ),
        "random_morphology_takeoff_sim_dt_s": 0.1,
        "random_morphology_takeoff_steps": len(times),
        "random_morphology_takeoff_settled_pose_world": [
            target_pose[0],
            target_pose[1],
            target_pose[2] - 0.5,
            *target_pose[3:7],
        ],
        "random_morphology_takeoff_hover_target_pose_world": list(target_pose),
        "random_morphology_takeoff_hover_height_delta_m": 0.5,
        "random_morphology_takeoff_hover_hold_time_s": 0.4,
        "random_morphology_takeoff_hover_hold_required_s": 0.35,
        "random_morphology_takeoff_position_error_threshold_m": 0.2,
        "random_morphology_takeoff_attitude_error_threshold_rad": 0.25,
        "random_morphology_takeoff_hover_linear_speed_threshold_mps": 0.15,
        "random_morphology_takeoff_hover_angular_speed_threshold_rad_s": 0.25,
        "random_morphology_takeoff_min_height_gain_ratio": 0.8,
        "random_morphology_takeoff_exact_cross_module_collision_passed": True,
        "random_morphology_takeoff_finite_state": True,
        "random_morphology_takeoff_qp_infeasible_count": 0,
        "random_morphology_takeoff_missing_actuator_count": 0,
        "random_morphology_takeoff_unsupported_actuator_count": 0,
        "random_morphology_takeoff_clipped_target_count": 0,
        "random_morphology_takeoff_runtime_observations": [
            observation.to_dict() for observation in observations[:-1]
        ],
        "random_morphology_takeoff_policy_commands": [
            command.to_dict() for command in policy_commands
        ],
        "random_morphology_takeoff_controller_commands": [
            command.to_dict() for command in controller_commands
        ],
        "random_morphology_takeoff_actuator_target_records": [
            record.to_dict() for record in actuator_records
        ],
        "order3_pi_l_rollout": True,
        "order3_pi_l_checkpoint_sha256": CHECKPOINT_HASH,
        "order3_pi_l_checkpoint_metadata": metadata.to_dict(),
        "order3_pi_l_stochastic": True,
        "order3_pi_l_policy_decision_count": 2,
        "order3_pi_l_policy_applied_count": 2,
        "order3_pi_l_fallback_count": 0,
        "order3_pi_l_transition_traces": [
            {
                "step_index": 0,
                "time_s": 0.0,
                "target_pose_world": list(target_pose),
                "target_twist": [0.0] * 6,
                "previous_action": [0.0] * len(ORDER3_ACTION_NAMES),
                "action": action_a,
                "action_mean": action_a,
                "recurrent_state_in": [0.0] * 8,
                "recurrent_state_out": [0.2] * 8,
                "old_log_prob": -0.5,
                "old_value": 0.25,
                "policy_applied": True,
                "fallback_reason": None,
                "privileged_disturbance_body": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            {
                "step_index": 2,
                "time_s": 0.2,
                "target_pose_world": list(target_pose),
                "target_twist": [0.0] * 6,
                "previous_action": action_a,
                "action": action_b,
                "action_mean": action_b,
                "recurrent_state_in": [0.2] * 8,
                "recurrent_state_out": [0.3] * 8,
                "old_log_prob": -0.75,
                "old_value": 0.5,
                "policy_applied": True,
                "fallback_reason": None,
                "privileged_disturbance_body": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
        ],
        "order3_pi_l_final_runtime_observation": observations[-1].to_dict(),
        "order3_pi_l_final_bootstrap_value": 0.75,
        "order3_privileged_external_wrench_body": [
            2.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        "order3_disturbance_start_s": 0.0,
        "order3_disturbance_duration_s": 0.0,
        "random_morphology_takeoff_artifacts": {
            "backend": "isaac_lab",
            "isaac_backed": True,
            "dry_run": False,
            "is_p4_full_completion": False,
            "object_task_claim": False,
            "learned_policy_claim": True,
            "learned_policy_scope": "order3_free_flight_takeoff_hover",
        },
    }
    config = Order3OnlineCollectorConfig(
        reward_config=Order3FreeFlightRewardConfig(
            success_hold_duration_s=0.35,
        )
    )
    return report, morphology, physical_model, config


def test_collects_decision_aligned_online_transitions_and_sanitizes_wrench() -> None:
    report, _, physical_model, config = _source(contact_wrench=True)

    result = collect_order3_online_transitions(
        report,
        split=DatasetSplit.TRAIN,
        physical_model=physical_model,
        expected_checkpoint_sha256=CHECKPOINT_HASH,
        config=config,
    )

    assert result.structural_hash
    assert result.checkpoint_sha256 == CHECKPOINT_HASH
    assert result.removed_privileged_contact_wrench_count == 2
    assert [item.step_index for item in result.transitions] == [0, 2]
    assert [item.terminal for item in result.transitions] == [False, True]
    assert all(not item.truncated for item in result.transitions)
    assert result.transitions[0].action == [0.1] * 12
    assert result.transitions[1].previous_action == [0.1] * 12
    assert result.transitions[1].old_log_prob == -0.75
    assert result.transitions[1].old_value == 0.5
    assert result.transitions[1].recurrent_state_in == [0.2] * 8
    assert result.transitions[0].privileged_disturbance_body[0] == 2.0
    assert result.transitions[0].behavior_policy_kind == "order3_checkpoint"
    assert result.transitions[0].behavior_policy_version == ORDER3_CHECKPOINT_VERSION
    assert result.transitions[0].behavior_checkpoint_hash == CHECKPOINT_HASH
    assert result.transitions[0].action_semantics == "learned_residual"
    assert result.transitions[0].metrics["outcome_step_index"] == 2.0
    assert result.transitions[1].metrics["outcome_step_index"] == 4.0
    assert (
        result.transitions[0].runtime_observation.contact_states[0].wrench_world is None
    )
    assert report["random_morphology_takeoff_runtime_observations"][0][
        "contact_states"
    ][0]["wrench_world"] == [0.0, 0.0, 8.0, 0.0, 0.0, 0.0]
    assert result.object_task_claim is False
    assert result.contact_task_claim is False
    assert result.dock_motion_claim is False
    assert result.p4_full_completion_claim is False


def test_online_collector_requires_absolute_zero_dock_hold_targets() -> None:
    report, morphology, physical_model, config = _source()
    joint_id = "pitch_dock_mech_joint1"
    key = f"module_{morphology.modules[0].module_id}:{joint_id}"
    for observation in report["random_morphology_takeoff_runtime_observations"]:
        observation["module_states"][0]["joint_positions"][joint_id] = 0.002
    for command in report["random_morphology_takeoff_policy_commands"]:
        command["joint_position_targets"] = {key: 0.0}
        command["joint_velocity_targets"] = {key: 0.0}
        command["joint_torque_bias"] = {key: 0.0}

    collected = collect_order3_online_transitions(
        report,
        split=DatasetSplit.TRAIN,
        physical_model=physical_model,
        expected_checkpoint_sha256=CHECKPOINT_HASH,
        config=config,
    )
    assert collected.transitions

    report["random_morphology_takeoff_policy_commands"][0][
        "joint_position_targets"
    ][key] = 0.002
    with pytest.raises(SchemaValidationError, match="commanded dock joint motion"):
        collect_order3_online_transitions(
            report,
            split=DatasetSplit.TRAIN,
            physical_model=physical_model,
            expected_checkpoint_sha256=CHECKPOINT_HASH,
            config=config,
        )


def test_behavior_checkpoint_replay_binds_actor_rows_and_log_prob(
    tmp_path,
) -> None:
    report, morphology, physical_model, config = _source()
    policy_config = Order3MorphologyConditionedPolicyConfig(
        graph_hidden_dim=8,
        graph_message_layers=1,
        recurrent_hidden_dim=8,
    )
    torch.manual_seed(42)
    model = MorphologyConditionedActorCritic(policy_config)
    metadata = replace(
        Order3PolicyCheckpointMetadata.from_dict(
            report["order3_pi_l_checkpoint_metadata"]
        ),
        config_hash=policy_config.stable_hash(),
        actor_feature_schema_hash=order3_actor_feature_schema_hash(),
        graph_feature_schema_hash=order3_graph_feature_schema_hash(),
        metadata={
            "morphology_hashes": {
                DatasetSplit.TRAIN.value: [morphology_structural_hash(morphology)]
            }
        },
    )
    checkpoint_path = tmp_path / "behavior.pt"
    checkpoint_hash = save_order3_policy_checkpoint(
        checkpoint_path,
        model=model,
        metadata=metadata,
    )
    report["order3_pi_l_checkpoint_sha256"] = checkpoint_hash
    report["order3_pi_l_checkpoint_metadata"] = metadata.to_dict()

    observations = [
        RuntimeObservation.from_dict(value)
        for value in report["random_morphology_takeoff_runtime_observations"]
    ]
    hidden = model.initial_state(1)
    builder = RigidBodyControlModelBuilder()
    with torch.no_grad():
        for trace in report["order3_pi_l_transition_traces"]:
            observation = observations[trace["step_index"]]
            control_model = builder.build(morphology, physical_model, observation)
            features = order3_actor_feature_vector(
                observation,
                control_model,
                target_pose_world=trace["target_pose_world"],
                target_twist=trace["target_twist"],
                max_modules=policy_config.max_modules,
            )
            trace["recurrent_state_in"] = hidden[0].tolist()
            step = model.step(
                [morphology],
                [observation],
                torch.tensor([features], dtype=torch.float32),
                torch.tensor([trace["previous_action"]], dtype=torch.float32),
                hidden,
                privileged_disturbance_body=torch.tensor(
                    [trace["privileged_disturbance_body"]], dtype=torch.float32
                ),
                action=torch.tensor([trace["action"]], dtype=torch.float32),
            )
            trace["old_log_prob"] = float(step.log_prob[0].item())
            trace["old_value"] = float(step.value[0].item())
            trace["recurrent_state_out"] = step.recurrent_state[0].tolist()
            hidden = step.recurrent_state

    result = collect_order3_online_transitions(
        report,
        split=DatasetSplit.TRAIN,
        physical_model=physical_model,
        expected_checkpoint_sha256=checkpoint_hash,
        behavior_checkpoint_path=checkpoint_path,
        config=config,
    )
    assert result.metadata["behavior_replay_verified"] is True

    tampered = copy.deepcopy(report)
    tampered["order3_pi_l_transition_traces"][0]["old_log_prob"] += 0.01
    with pytest.raises(SchemaValidationError, match="old_log_prob"):
        collect_order3_online_transitions(
            tampered,
            split=DatasetSplit.TRAIN,
            physical_model=physical_model,
            expected_checkpoint_sha256=checkpoint_hash,
            behavior_checkpoint_path=checkpoint_path,
            config=config,
        )


@pytest.mark.parametrize("task_mode", ["hover", "waypoint"])
def test_collects_hash_bound_in_air_curriculum_modes(task_mode: str) -> None:
    report, _, physical_model, legacy_config = _source()
    target = list(report["random_morphology_takeoff_hover_target_pose_world"])
    if task_mode == "waypoint":
        target[0] += 0.1
    condition = build_order3_rollout_condition(
        stage_id=f"3c_{task_mode}",
        task_mode=task_mode,
        seed=404,
        waypoint_position_offset_world=(
            (0.1, 0.0, 0.0) if task_mode == "waypoint" else (0.0, 0.0, 0.0)
        ),
        hold_s=legacy_config.reward_config.success_hold_duration_s,
        external_wrench_body=(2.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        disturbance_start_s=0.0,
        disturbance_duration_s=0.0,
    )
    report.update(
        {
            "random_morphology_takeoff_smoke": False,
            "order3_rollout_condition": condition.to_dict(),
            "order3_rollout_condition_hash": condition.condition_hash,
            "order3_rollout_seed_applied": condition.seed,
            "order3_task_mode": task_mode,
            "order3_free_flight_success": True,
            "order3_free_flight_terminal_metrics": {
                "position_error_m": 0.1 if task_mode == "waypoint" else 0.0,
                "attitude_error_rad": 0.0,
                "linear_velocity_error_mps": 0.0,
                "angular_velocity_error_rad_s": 0.0,
                "within_tolerance_duration_s": 0.4,
                "takeoff_height_gain_ratio": None,
            },
            "order3_condition_realization": {
                "final_target_pose_world": target,
            },
        }
    )
    report["random_morphology_takeoff_artifacts"]["learned_policy_scope"] = (
        f"order3_free_flight_{task_mode}"
    )
    for trace in report["order3_pi_l_transition_traces"]:
        trace["target_pose_world"] = list(target)
    for command in report["random_morphology_takeoff_policy_commands"]:
        command["desired_body_pose"] = list(target)
    config = Order3OnlineCollectorConfig(
        reward_config=legacy_config.reward_config,
        task_mode=Order3TaskMode(task_mode),
    )

    result = collect_order3_online_transitions(
        report,
        split=DatasetSplit.TRAIN,
        physical_model=physical_model,
        config=config,
    )

    assert result.metadata["task_mode"] == task_mode
    assert result.metadata["curriculum_stage_id"] == f"3c_{task_mode}"
    assert result.metadata["rollout_condition_hash"] == condition.condition_hash
    assert all(
        transition.metrics["task_mode_code"]
        == float(list(Order3TaskMode).index(Order3TaskMode(task_mode)))
        for transition in result.transitions
    )
    assert all(
        transition.metrics.get("reward.takeoff_height", 0.0) == 0.0
        for transition in result.transitions
    )

    tampered = copy.deepcopy(report)
    tampered["order3_rollout_condition"]["seed"] += 1
    with pytest.raises(SchemaValidationError, match="rollout condition"):
        collect_order3_online_transitions(
            tampered,
            split=DatasetSplit.TRAIN,
            physical_model=physical_model,
            config=config,
        )


def test_time_limit_truncation_requires_and_preserves_final_bootstrap() -> None:
    report, _, physical_model, _ = _source()
    report["random_morphology_takeoff_smoke_passed"] = False
    report["random_morphology_takeoff_hover_hold_required_s"] = 1.0
    config = Order3OnlineCollectorConfig(
        reward_config=Order3FreeFlightRewardConfig(success_hold_duration_s=1.0)
    )

    result = collect_order3_online_transitions(
        report,
        split=DatasetSplit.TRAIN,
        physical_model=physical_model,
        config=config,
    )

    assert result.transitions[-1].terminal is False
    assert result.transitions[-1].truncated is True
    assert result.transitions[-1].bootstrap_value == 0.75

    report[_FINAL_BOOTSTRAP_KEY] = None
    with pytest.raises(SchemaValidationError, match="bootstrap"):
        collect_order3_online_transitions(
            report,
            split=DatasetSplit.TRAIN,
            physical_model=physical_model,
            config=config,
        )


def test_safety_failure_is_terminal_and_does_not_bootstrap() -> None:
    report, _, physical_model, config = _source()
    report["random_morphology_takeoff_smoke_passed"] = False
    report["random_morphology_takeoff_qp_infeasible_count"] = 1
    report["random_morphology_takeoff_controller_commands"][3]["controller_status"][
        "status"
    ] = "infeasible"
    report["random_morphology_takeoff_controller_commands"][3]["controller_status"][
        "qp_feasible"
    ] = False
    report["order3_pi_l_final_bootstrap_value"] = None

    result = collect_order3_online_transitions(
        report,
        split=DatasetSplit.TRAIN,
        physical_model=physical_model,
        config=config,
    )

    assert result.transitions[-1].terminal is True
    assert result.transitions[-1].truncated is False
    assert result.transitions[-1].bootstrap_value is None
    assert result.transitions[-1].metrics["reward.terminal_failure"] == 1.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report["order3_pi_l_transition_traces"][1].__setitem__(
                "previous_action", [0.0] * 12
            ),
            "previous action chain",
        ),
        (
            lambda report: report["random_morphology_takeoff_policy_commands"][
                2
            ].__setitem__("desired_body_pose", [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0]),
            "passed through",
        ),
        (
            lambda report: report["order3_pi_l_transition_traces"][0].__setitem__(
                "policy_applied", False
            ),
            "fallback",
        ),
    ],
)
def test_rejects_broken_policy_decision_sequence(mutation, message: str) -> None:
    report, _, physical_model, config = _source()
    mutation(report)

    with pytest.raises(SchemaValidationError, match=message):
        collect_order3_online_transitions(
            report,
            split=DatasetSplit.TRAIN,
            physical_model=physical_model,
            config=config,
        )


def test_rejects_privileged_actor_authority_and_wrong_split_hash() -> None:
    report, _, physical_model, config = _source()
    privileged = copy.deepcopy(report)
    privileged["order3_pi_l_checkpoint_metadata"]["actor_uses_privileged_wrench"] = True
    with pytest.raises(SchemaValidationError, match="metadata"):
        collect_order3_online_transitions(
            privileged,
            split=DatasetSplit.TRAIN,
            physical_model=physical_model,
            config=config,
        )

    with pytest.raises(SchemaValidationError, match="structural hash"):
        collect_order3_online_transitions(
            report,
            split=DatasetSplit.HELD_OUT,
            physical_model=physical_model,
            expected_structural_hash="wrong-assigned-structure",
            config=config,
        )


def test_strict_contact_wrench_mode_rejects_unsanitized_actor_input() -> None:
    report, _, physical_model, config = _source(contact_wrench=True)
    strict = Order3OnlineCollectorConfig(
        reward_config=config.reward_config,
        sanitize_privileged_contact_wrench=False,
    )

    with pytest.raises(SchemaValidationError, match="privileged contact wrench"):
        collect_order3_online_transitions(
            report,
            split=DatasetSplit.TRAIN,
            physical_model=physical_model,
            config=strict,
        )
