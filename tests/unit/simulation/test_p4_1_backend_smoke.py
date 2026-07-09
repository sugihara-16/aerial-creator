from __future__ import annotations

from amsrr.schemas.policies import ControllerStatus
from amsrr.schemas.runtime import ModuleRuntimeState, RuntimeObservation, TaskProgressState
from amsrr.simulation import (
    P4_1FullSceneBackendConfig,
    evaluate_runtime_observation_joint_state,
)
from amsrr.simulation.p4_control_controller_smoke import build_fixed_morphology
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.utils.config import load_config


def test_p4_1_backend_smoke_config_loader_contract() -> None:
    data = load_config("configs/training/p4_1_backend_smoke.yaml")

    config = P4_1FullSceneBackendConfig.from_dict(data["env"])

    assert config.config_path == "configs/env/isaac_lab.yaml"
    assert config.robot_model_config_path == "configs/robot/robot_model.yaml"
    assert config.p3_config_path == "configs/training/p3_assembly_grasp_carry.yaml"
    assert config.smoke_name == "p2_p3_full_scene_backend"
    assert config.require_p2_p3_design is True
    assert config.object_id == "box_01"
    assert config.object_initial_pose_world == (0.8, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0)


def test_p4_1_runtime_observation_joint_state_requires_vectoring_and_dock_joints() -> None:
    observation = _runtime_observation_with_joints(
        {
            "gimbal1": 0.0,
            "gimbal2": 0.0,
            "pitch_dock_mech_joint1": 0.0,
            "yaw_dock_mech_joint1": 0.0,
        },
        {
            "gimbal1": 0.0,
            "gimbal2": 0.0,
            "pitch_dock_mech_joint1": 0.0,
            "yaw_dock_mech_joint1": 0.0,
        },
    )

    result = evaluate_runtime_observation_joint_state([observation])

    assert result.passed is True
    assert result.vectoring_joint_key_count == 2
    assert result.dock_joint_key_count == 2
    assert result.modules_with_joint_positions == 1
    assert result.modules_with_joint_velocities == 1
    assert result.metrics["articulated_model_update_checked"] == 0.0


def test_p4_1_runtime_observation_joint_state_rejects_empty_joint_positions() -> None:
    observation = _runtime_observation_with_joints({}, {})

    result = evaluate_runtime_observation_joint_state([observation])

    assert result.passed is False
    assert "P4.1 module joint_positions are not populated for every module" in result.failure_reasons
    assert "P4.1 vectoring/gimbal joint positions are missing" in result.failure_reasons
    assert "P4.1 dock mechanism joint positions are missing" in result.failure_reasons


def test_p4_1_articulated_joint_state_requires_model_update_metric() -> None:
    observation = _runtime_observation_with_joints(
        {"gimbal1": 0.05, "pitch_dock_mech_joint1": 0.1},
        {"gimbal1": 0.0, "pitch_dock_mech_joint1": 0.0},
    )

    failed = evaluate_runtime_observation_joint_state(
        [observation],
        articulated_morphology=True,
        articulated_model_update_metrics={
            "max_model_rotor_origin_change_m": 0.0,
            "max_model_allocation_change": 0.0,
        },
    )
    passed = evaluate_runtime_observation_joint_state(
        [observation],
        articulated_morphology=True,
        articulated_model_update_metrics={
            "max_model_rotor_origin_change_m": 0.01,
            "max_model_allocation_change": 0.0,
        },
    )

    assert failed.passed is False
    assert "P4.1 articulated observation did not prove B(q) model update" in failed.failure_reasons
    assert passed.passed is True
    assert passed.articulated_model_update_checked is True
    assert passed.articulated_model_update_passed is True


def _runtime_observation_with_joints(
    joint_positions: dict[str, float],
    joint_velocities: dict[str, float],
) -> RuntimeObservation:
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    morphology = build_fixed_morphology(physical_model, module_count=1, module_spacing_m=0.45)
    return RuntimeObservation(
        time_s=0.0,
        morphology_graph=morphology,
        module_states=[
            ModuleRuntimeState(
                module_id=0,
                pose_world=(0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0),
                twist_world=[0.0] * 6,
                joint_positions=joint_positions,
                joint_velocities=joint_velocities,
            )
        ],
        object_states=[],
        contact_states=[],
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
        task_progress=TaskProgressState(),
    )
