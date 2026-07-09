from __future__ import annotations

from typing import Any

from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.policies import ControllerCommand, ControllerStatus
from amsrr.schemas.runtime import ModuleRuntimeState, ObjectRuntimeState, RuntimeObservation, TaskProgressState
from amsrr.simulation import (
    IsaacLabAvailability,
    IsaacLabBackend,
    IsaacLabBackendConfig,
    P4_1FullSceneBackendConfig,
    P4_1IsaacBackendEnv,
    load_p4_1_full_scene_backend_config,
    p4_1_result_from_report,
)
from amsrr.simulation.p4_control_controller_smoke import build_fixed_morphology


def test_p4_1_full_scene_backend_config_loader() -> None:
    backend_config, env_config = load_p4_1_full_scene_backend_config("configs/training/p4_1_backend_smoke.yaml")

    assert backend_config.micromamba_env == "isaaclab3"
    assert backend_config.holon_urdf_path == "assets/robots/holon/holon.urdf"
    assert env_config.smoke_name == "p2_p3_full_scene_backend"
    assert env_config.object_size_m == (0.30, 0.20, 0.15)
    assert env_config.object_initial_pose_world == (0.8, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0)
    assert env_config.require_p2_p3_design is True


def test_p4_1_backend_command_marks_full_scene_object_and_p2_p3() -> None:
    backend = IsaacLabBackend(IsaacLabBackendConfig())

    command = backend.p4_1_full_scene_backend_smoke_command(
        config_path="configs/env/isaac_lab.yaml",
        convert_if_missing=False,
        force_convert=True,
        steps=12,
        module_count=3,
        module_spacing_m=0.5,
        object_size_m=(0.4, 0.2, 0.1),
        object_mass_kg=1.5,
        object_pose_world=(0.7, 0.1, 0.35, 0.0, 0.0, 0.0, 1.0),
        uses_p2_p3_design=True,
    )

    assert "--p4-1-full-scene-backend-smoke" in command
    assert "--p4-1-uses-p2-p3" in command
    assert "--fixed-module-count" in command
    assert "3" in command
    assert "--fixed-module-spacing-m" in command
    assert "0.5" in command
    assert "--p4-1-object-size-m" in command
    assert "0.4" in command
    assert "--p4-1-object-mass-kg" in command
    assert "1.5" in command
    assert "--p4-1-object-pose-world" in command
    assert "0.35" in command
    assert "--force-convert" in command
    assert "--convert-if-missing" not in command


def test_p4_1_dry_run_and_missing_backend_do_not_claim_completion() -> None:
    dry_env = P4_1IsaacBackendEnv(config=P4_1FullSceneBackendConfig())

    dry_result = dry_env.run_smoke(dry_run=True)

    assert dry_result.skipped is True
    assert dry_result.attempted is False
    assert dry_result.passed is False
    assert dry_result.skip_reason == "dry_run"

    missing_backend = IsaacLabBackend(
        IsaacLabBackendConfig(
            isaaclab_path="/tmp/amsrr_missing_isaaclab",
            holon_urdf_path="/tmp/amsrr_missing_holon.urdf",
        )
    )
    missing_env = P4_1IsaacBackendEnv(config=P4_1FullSceneBackendConfig(), backend=missing_backend)

    missing_result = missing_env.run_smoke(dry_run=False)

    assert missing_result.skipped is True
    assert missing_result.attempted is False
    assert missing_result.passed is False
    assert "isaaclab_path_missing" in str(missing_result.skip_reason)


def test_p4_1_fake_backend_report_parses_per_step_logs_and_joint_state() -> None:
    report = _p4_1_report(step_count=2, module_count=3)
    backend = _P4_1Backend(report=report)
    env = P4_1IsaacBackendEnv(
        config=P4_1FullSceneBackendConfig(max_episode_steps=2),
        backend=backend,  # type: ignore[arg-type]
    )

    result = env.run_smoke(
        dry_run=False,
        module_count=3,
        module_spacing_m=0.5,
        uses_p2_selected_design=True,
        uses_p3_assembled_morphology=True,
    )

    assert result.attempted is True
    assert result.skipped is False
    assert result.passed is True
    assert result.isaac_backed is True
    assert result.full_scene_spawned is True
    assert result.robot_spawned is True
    assert result.object_spawned is True
    assert result.floor_spawned is True
    assert result.uses_p2_selected_design is True
    assert result.uses_p3_assembled_morphology is True
    assert len(result.runtime_observations) == 2
    assert len(result.controller_commands) == 2
    assert len(result.actuator_target_records) == 2
    assert len(result.object_pose_history) == 2
    assert result.runtime_observations[0].object_states[0].object_id == "box_01"
    assert result.joint_state_metrics is not None
    assert result.joint_state_metrics.passed is True
    assert result.joint_state_metrics.vectoring_joint_key_count >= 6
    assert result.joint_state_metrics.dock_joint_key_count >= 6
    assert backend.calls[0]["module_count"] == 3
    assert backend.calls[0]["module_spacing_m"] == 0.5
    assert backend.calls[0]["uses_p2_p3_design"] is True


def test_p4_1_report_without_joint_positions_fails_joint_state_gate() -> None:
    report = _p4_1_report(step_count=1, module_count=2, include_joint_positions=False)

    result = p4_1_result_from_report(
        report,
        smoke_name="p2_p3_full_scene_backend",
        uses_p2_selected_design=True,
        uses_p3_assembled_morphology=True,
    )

    assert result.passed is False
    assert result.joint_state_metrics is not None
    assert "P4.1 module joint_positions are not populated for every module" in result.joint_state_metrics.failure_reasons


class _P4_1Backend:
    def __init__(self, *, report: dict[str, Any]) -> None:
        self.report = report
        self.calls: list[dict[str, Any]] = []

    def availability(self) -> IsaacLabAvailability:
        return IsaacLabAvailability(
            available=True,
            isaaclab_path_exists=True,
            launch_script_exists=True,
            urdf_exists=True,
            generated_usd_exists=False,
            python_modules_available=True,
            missing_reasons=[],
        )

    def run_p4_1_full_scene_backend_smoke(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        return self.report


def _p4_1_report(
    *,
    step_count: int,
    module_count: int,
    include_joint_positions: bool = True,
) -> dict[str, Any]:
    runtime_observations = [
        _runtime_observation(step_idx=step_idx, module_count=module_count, include_joint_positions=include_joint_positions)
        for step_idx in range(step_count)
    ]
    controller_commands = [
        ControllerCommand(
            rotor_thrusts_n={f"module_{module_id}:thruster{module_id}": 0.1 for module_id in range(module_count)},
            vectoring_joint_targets={f"module_{module_id}:gimbal1": 0.0 for module_id in range(module_count)},
            joint_torque_commands={},
            dock_mechanism_commands={
                f"module_{module_id}:pitch_dock_mech_joint1": 0.0 for module_id in range(module_count)
            },
            controller_status=ControllerStatus(status="ok", qp_feasible=True),
        ).to_dict()
        for _ in range(step_count)
    ]
    actuator_records = [
        {
            "time_s": float(step_idx) * 0.005,
            "backend": "isaac_lab",
            "morphology_graph_id": "p4-1-full-scene-backend-smoke",
            "command_index": step_idx,
            "actuator_targets": [],
            "clipped_targets": [],
            "missing_actuators": [],
            "unsupported_actuators": [],
            "allocation_residual_norm": 0.0,
            "qp_status": "ok",
            "metrics": {},
            "metadata": {},
        }
        for step_idx in range(step_count)
    ]
    object_pose_history = [[0.8, 0.0, 0.4 + 0.001 * step_idx, 0.0, 0.0, 0.0, 1.0] for step_idx in range(step_count)]
    return {
        "isaac_backed": True,
        "p4_1_full_scene_backend_smoke": True,
        "p4_1_full_scene_backend_smoke_passed": True,
        "p4_1_full_scene_spawned": True,
        "p4_1_robot_spawned": True,
        "p4_1_object_spawned": True,
        "p4_1_floor_spawned": True,
        "p4_1_uses_p2_p3": True,
        "p4_1_articulated_morphology": False,
        "p4_1_runtime_observations": [observation.to_dict() for observation in runtime_observations],
        "p4_1_controller_commands": controller_commands,
        "p4_1_actuator_target_records": actuator_records,
        "p4_1_object_pose_history": object_pose_history,
        "p4_1_runtime_observation_count": step_count,
        "p4_1_controller_command_count": step_count,
        "p4_1_actuator_target_record_count": step_count,
        "p4_1_object_pose_count": step_count,
        "p4_1_module_count": module_count,
        "p4_1_steps": step_count,
    }


def _runtime_observation(
    *,
    step_idx: int,
    module_count: int,
    include_joint_positions: bool,
) -> RuntimeObservation:
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    morphology = build_fixed_morphology(physical_model, module_count=module_count, module_spacing_m=0.45)
    module_states = []
    for module_id in range(module_count):
        joint_positions = (
            {
                "gimbal1": 0.01 * step_idx,
                "gimbal2": -0.01 * step_idx,
                "pitch_dock_mech_joint1": 0.0,
                "yaw_dock_mech_joint1": 0.0,
            }
            if include_joint_positions
            else {}
        )
        module_states.append(
            ModuleRuntimeState(
                module_id=module_id,
                pose_world=(0.45 * module_id, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0),
                twist_world=[0.0] * 6,
                joint_positions=joint_positions,
                joint_velocities={key: 0.0 for key in joint_positions},
            )
        )
    return RuntimeObservation(
        time_s=0.005 * step_idx,
        morphology_graph=morphology,
        module_states=module_states,
        object_states=[
            ObjectRuntimeState(
                object_id="box_01",
                pose_world=(0.8, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0),
                twist_world=[0.0] * 6,
            )
        ],
        contact_states=[],
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
        task_progress=TaskProgressState(),
    )
