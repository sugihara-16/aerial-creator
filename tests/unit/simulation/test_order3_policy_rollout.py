from __future__ import annotations

import pytest

from amsrr.morphology.random_connected import RandomConnectedMorphologyDistribution
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaValidationError
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
from amsrr.schemas.policies import PolicyCommand
from amsrr.schemas.order3_rollout_condition import build_order3_rollout_condition
from amsrr.simulation.order3_policy_rollout import (
    Order3DeterministicBaselineRolloutConfig,
    Order3DeterministicBaselineRolloutEnv,
    Order3IsaacPolicyRolloutConfig,
    Order3IsaacPolicyRolloutEnv,
    Order3IsaacPolicyRolloutResult,
    _order3_condition_report_failures,
    _order3_in_air_report_failures,
    _order3_report_failures,
)
from amsrr.simulation.order3_rollout_condition import (
    ORDER3_FREE_FLIGHT_REPORT_VERSION,
    Order3ConditionRealization,
)
from amsrr.simulation.random_morphology_takeoff import (
    RandomMorphologyTakeoffConfig,
    RandomMorphologyTakeoffEnv,
)


CHECKPOINT_HASH = "a" * 64


def _takeoff_env() -> RandomMorphologyTakeoffEnv:
    return RandomMorphologyTakeoffEnv(
        config=RandomMorphologyTakeoffConfig(
            control_contract_version="centroidal_local_joint_v2",
        )
    )


def _metadata() -> Order3PolicyCheckpointMetadata:
    return Order3PolicyCheckpointMetadata(
        checkpoint_version=ORDER3_CHECKPOINT_VERSION,
        policy_family=ORDER3_POLICY_FAMILY,
        policy_contract_version="centroidal_local_joint_v2",
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
        physical_model_hash="physical",
        urdf_hash="urdf",
        controller_contract_hash="controller",
        fallback_version=ORDER3_FALLBACK_VERSION,
        fallback_config_hash="fallback",
        seed=1,
        git_revision="revision",
    )


def test_order3_rollout_dry_command_binds_checkpoint_and_disturbance() -> None:
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    morphology = RandomConnectedMorphologyDistribution(physical_model).sample(
        seed=4,
        module_count=2,
    )
    config = Order3IsaacPolicyRolloutConfig(
        checkpoint_path="/tmp/order3-checkpoint.pt",
        expected_checkpoint_sha256=CHECKPOINT_HASH,
        external_wrench_body=[1.0, 0.0, 0.0, 0.0, 0.1, 0.0],
    )

    result = Order3IsaacPolicyRolloutEnv(
        config=config,
        takeoff_env=_takeoff_env(),
    ).run(morphology, dry_run=True)

    command = result.takeoff_result.report["probe_command"]
    assert "--order3-pi-l-checkpoint-path" in command
    assert command[command.index("--order3-pi-l-checkpoint-path") + 1] == config.checkpoint_path
    assert "--order3-rollout-condition-json" in command
    condition_json = command[command.index("--order3-rollout-condition-json") + 1]
    assert '"external_wrench_body":[1.0,0.0,0.0,0.0,0.1,0.0]' in condition_json
    assert "--control-contract-version" in command
    assert result.takeoff_result.real_isaac_passed is False
    assert result.p4_full_completion_claim is False


def test_order3_rollout_dry_command_propagates_kit_viewer_options() -> None:
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    morphology = RandomConnectedMorphologyDistribution(physical_model).sample(
        seed=5,
        module_count=2,
    )
    env = Order3IsaacPolicyRolloutEnv(
        config=Order3IsaacPolicyRolloutConfig(
            checkpoint_path="/tmp/order3-checkpoint.pt",
            expected_checkpoint_sha256=CHECKPOINT_HASH,
        ),
        takeoff_env=_takeoff_env(),
        viewer="kit",
        realtime_playback=True,
        keep_open_after_rollout_s=12.5,
    )

    command = env.build_probe_command(morphology)

    assert command[command.index("--viz") + 1] == "kit"
    assert "--realtime-playback" in command
    assert command[command.index("--keep-open-after-smoke-s") + 1] == "12.5"

    with pytest.raises(ValueError, match="require viewer='kit'"):
        Order3IsaacPolicyRolloutEnv(
            config=env.config,
            takeoff_env=_takeoff_env(),
            realtime_playback=True,
        )


def test_order3_in_air_condition_report_contract_is_strict() -> None:
    condition = build_order3_rollout_condition(
        stage_id="3c_randomized",
        task_mode="waypoint",
        seed=12,
        waypoint_position_offset_world=(0.2, 0.0, 0.1),
        waypoint_orientation_rpy_rad=(0.1, -0.2, 0.3),
        mass_scale=1.05,
        inertia_scale=0.95,
        thrust_scale=0.98,
    )
    realization = Order3ConditionRealization(
        condition_hash=condition.condition_hash,
        task_mode="waypoint",
        requested_initial_root_pose_world=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        applied_initial_root_pose_world=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        requested_initial_twist_world=[0.0] * 6,
        applied_initial_twist_world=[0.0] * 6,
        requested_mass_scale=1.05,
        applied_mass_scale=1.05,
        requested_inertia_scale=0.95,
        applied_inertia_scale=0.95,
        requested_thrust_scale=0.98,
        applied_thrust_scale=0.98,
        mass_randomization_applied=True,
        inertia_randomization_applied=True,
        thrust_randomization_applied=True,
        initial_state_applied=True,
        final_target_pose_world=[0.2, 0.0, 1.1, 0.0, 0.0, 0.0, 1.0],
        final_target_twist_world=[0.0] * 6,
    )
    terminal_metrics = {
        "position_error_m": 0.01,
        "attitude_error_rad": 0.02,
        "linear_velocity_error_mps": 0.01,
        "angular_velocity_error_rad_s": 0.01,
        "within_tolerance_duration_s": 1.0,
        "takeoff_height_gain_ratio": None,
    }
    report = {
        "order3_rollout_condition": condition.to_dict(),
        "order3_rollout_condition_hash": condition.condition_hash,
        "order3_rollout_task_mode": "waypoint",
        "order3_task_mode": "waypoint",
        "order3_report_validation_failures": [],
        "order3_rollout_seed_applied": {
            "seed": condition.seed,
            "python_random": True,
            "torch": True,
        },
        "order3_privileged_external_wrench_body": list(condition.external_wrench_body),
        "order3_disturbance_start_s": condition.disturbance_start_s,
        "order3_disturbance_duration_s": condition.disturbance_duration_s,
        "order3_condition_realization": realization.to_dict(),
        "order3_terminal_evidence_start_s": 1.0,
        "order3_terminal_evidence_completed": True,
        "order3_tracking_window_start_s": 0.0,
        "order3_tracking_window_end_s": 5.0,
        "order3_tracking_window_sample_count": 200,
        "order3_free_flight_report_version": ORDER3_FREE_FLIGHT_REPORT_VERSION,
        "random_morphology_takeoff_fixed_dock_neutral_hold_passed": True,
        "random_morphology_takeoff_fixed_dock_joint_count": 12,
        "random_morphology_takeoff_dock_joint_position_tolerance_rad": 0.0053,
        "random_morphology_takeoff_max_abs_dock_joint_position_rad": 0.001,
        "random_morphology_takeoff_final_max_abs_dock_joint_position_rad": 0.001,
        "random_morphology_takeoff_max_abs_dock_position_target_rad": 0.0,
        "random_morphology_takeoff_max_abs_dock_velocity_target_rad_s": 0.0,
        "random_morphology_takeoff_max_abs_dock_torque_bias_nm": 0.0,
        "order3_free_flight_floor_initialization": False,
        "order3_free_flight_floor_evidence_claim": False,
        "isaac_backed": True,
        "order3_free_flight_passed": True,
        "order3_free_flight_success": True,
        "order3_free_flight_tracking_cost": 0.1,
        "order3_structural_hash": "d" * 64,
        "order3_terminal_metrics": terminal_metrics,
        "order3_free_flight_terminal_metrics": terminal_metrics,
        "order3_free_flight_qp_infeasible_count": 0,
        "order3_free_flight_hard_collision_count": 0,
        "order3_free_flight_non_finite_state_count": 0,
        "order3_free_flight_unsupported_actuator_count": 0,
        "order3_qp_infeasible": False,
        "order3_hard_collision": False,
        "order3_non_finite_state": False,
        "order3_unsupported_actuator": False,
    }
    assert _order3_condition_report_failures(
        report, expected_condition=condition
    ) == []
    assert _order3_in_air_report_failures(report) == []

    report["order3_free_flight_floor_evidence_claim"] = True
    assert "in_air_floor_evidence_claim" in _order3_in_air_report_failures(report)


def test_order3_dry_in_air_rollout_binds_task_condition() -> None:
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    morphology = RandomConnectedMorphologyDistribution(physical_model).sample(
        seed=19,
        module_count=3,
    )
    condition = build_order3_rollout_condition(
        stage_id="3b_hover",
        task_mode="hover",
        seed=81,
        initial_position_offset_world=(0.05, -0.02, 0.03),
    )
    result = Order3IsaacPolicyRolloutEnv(
        config=Order3IsaacPolicyRolloutConfig(
            checkpoint_path="/tmp/order3-checkpoint.pt",
            expected_checkpoint_sha256=CHECKPOINT_HASH,
            rollout_condition=condition,
        ),
        takeoff_env=_takeoff_env(),
    ).run(morphology, dry_run=True)

    assert result.task_mode == "hover"
    assert result.rollout_condition == condition
    assert result.terminal_metrics is not None
    command = result.takeoff_result.report["probe_command"]
    assert condition.to_canonical_json() in command
    assert Order3IsaacPolicyRolloutResult.from_json(result.to_json()) == result

    baseline_command = Order3DeterministicBaselineRolloutEnv(
        config=Order3DeterministicBaselineRolloutConfig(
            rollout_condition=condition,
        ),
        takeoff_env=_takeoff_env(),
    ).build_probe_command(morphology)
    assert baseline_command[
        baseline_command.index("--order3-rollout-condition-json") + 1
    ] == condition.to_canonical_json()


def test_order3_report_trace_contract_validates_pose_passthrough() -> None:
    pose = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    command = PolicyCommand(
        desired_body_pose=pose,
        desired_body_twist=[0.0] * 6,
        residual_wrench_body=[0.0] * 6,
        control_contract_version="centroidal_local_joint_v2",
    )
    report = {
        "asset_cache_reuse_enabled": True,
        "asset_cache_key": "b" * 64,
        "generated_urdf_sha256": "c" * 64,
        "order3_pi_l_rollout": True,
        "order3_pi_l_checkpoint_sha256": CHECKPOINT_HASH,
        "order3_pi_l_checkpoint_metadata": _metadata().to_dict(),
        "order3_pi_l_policy_decision_count": 1,
        "order3_pi_l_policy_applied_count": 1,
        "order3_pi_l_fallback_count": 0,
        "order3_pi_l_transition_traces": [
            {
                "step_index": 0,
                "target_pose_world": list(pose),
                "target_twist": [0.0] * 6,
                "previous_action": [0.0] * 12,
                "action": [0.0] * 12,
                "privileged_disturbance_body": [0.0] * 6,
                "recurrent_state_in": [0.0] * 8,
            }
        ],
        "random_morphology_takeoff_runtime_observations": [{}],
        "random_morphology_takeoff_policy_commands": [command.to_dict()],
    }

    assert _order3_report_failures(
        report,
        expected_checkpoint_sha256=CHECKPOINT_HASH,
        record_transitions=True,
    ) == []

    report["random_morphology_takeoff_policy_commands"][0]["desired_body_pose"][2] = 2.0
    failures = _order3_report_failures(
        report,
        expected_checkpoint_sha256=CHECKPOINT_HASH,
        record_transitions=True,
    )
    assert "centroidal_pose_not_passed_through" in failures


def test_order3_rollout_config_and_contract_fail_closed() -> None:
    with pytest.raises(SchemaValidationError, match="sha256"):
        Order3IsaacPolicyRolloutConfig(
            checkpoint_path="checkpoint.pt",
            expected_checkpoint_sha256="bad",
        )
    with pytest.raises(SchemaValidationError, match="centroidal_local_joint_v2"):
        Order3IsaacPolicyRolloutEnv(
            config=Order3IsaacPolicyRolloutConfig(
                checkpoint_path="checkpoint.pt",
                expected_checkpoint_sha256=CHECKPOINT_HASH,
            ),
            takeoff_env=RandomMorphologyTakeoffEnv(),
        )
