from __future__ import annotations

from pathlib import Path
from typing import Any

from amsrr.logging import read_episode_archives_jsonl
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.policies import ControllerCommand, ControllerStatus
from amsrr.schemas.runtime import ModuleRuntimeState, ObjectRuntimeState, RuntimeObservation, TaskProgressState
from amsrr.simulation import IsaacLabAvailability, P4_1FullSceneBackendConfig, P4_1IsaacBackendEnv
from amsrr.simulation.p4_control_controller_smoke import build_fixed_morphology
from amsrr.training import (
    P4_1BackendSmokeRunner,
    P4_1BackendSmokeRunnerConfig,
    load_p4_1_backend_smoke_runner_config,
)


def test_p4_1_backend_smoke_runner_config_loader() -> None:
    runner_config, env_config = load_p4_1_backend_smoke_runner_config("configs/training/p4_1_backend_smoke.yaml")

    assert runner_config.runner_version == "p4_1_backend_smoke_runner_v1"
    assert runner_config.dry_run is True
    assert runner_config.archive_path == "artifacts/p4_1/p4_1_backend_smoke.jsonl"
    assert runner_config.p3_config_path == "configs/training/p3_assembly_grasp_carry.yaml"
    assert env_config.smoke_name == "p2_p3_full_scene_backend"


def test_p4_1_runner_builds_p2_selected_and_p3_assembled_case() -> None:
    runner = P4_1BackendSmokeRunner(
        runner_config=P4_1BackendSmokeRunnerConfig(dry_run=True, archive_path=None, seed=0)
    )

    case = runner.build_p2_p3_case()

    assert case.selection.selected_candidate.feasibility_result.feasible is True
    assert case.assembly_report.success is True
    assert case.module_count == 3
    assert case.module_count != 2
    assert case.assembled_morphology.graph_id.endswith(":construction:2")


def test_p4_1_runner_archives_fake_backend_per_step_logs(tmp_path: Path) -> None:
    backend = _FakeP4_1Backend(step_count=2)
    env = P4_1IsaacBackendEnv(
        config=P4_1FullSceneBackendConfig(max_episode_steps=2),
        backend=backend,  # type: ignore[arg-type]
    )
    runner = P4_1BackendSmokeRunner(
        runner_config=P4_1BackendSmokeRunnerConfig(dry_run=False, archive_path=None, seed=0),
        env_config=P4_1FullSceneBackendConfig(max_episode_steps=2),
        env=env,
    )
    archive_path = tmp_path / "p4_1_backend_smoke.jsonl"

    result = runner.run(archive_path=archive_path)

    assert result.dry_run is False
    assert result.smoke_result.passed is True
    assert result.smoke_result.isaac_backed is False
    assert result.metrics["p2_selected_design_used"] == 1.0
    assert result.metrics["p3_assembly_result_used"] == 1.0
    assert result.metrics["fixed_two_module_only"] == 0.0
    assert backend.calls[0]["module_count"] == 3
    assert backend.calls[0]["uses_p2_p3_design"] is True

    assert len(result.archives) == 1
    archive = result.archives[0]
    assert archive.design_output is not None
    assert archive.feasibility_result is not None
    assert archive.assembly_plan is not None
    assert len(archive.runtime_observations) == 2
    assert len(archive.controller_commands) == 2
    assert len(archive.actuator_target_records) == 2
    assert archive.rollout_artifacts["archive_type"] == "p4_1_backend_smoke_per_step"
    assert len(archive.rollout_artifacts["p4_1_object_pose_history"]) == 2
    assert archive.rollout_artifacts["p2_selected_design_used"] is True
    assert archive.rollout_artifacts["p3_assembled_morphology_used"] is True
    assert archive.rollout_artifacts["is_p4_full_completion"] is False
    assert archive.rollout_artifacts["object_grasp_carry_claim"] is False
    assert archive.rollout_artifacts["learning_claim"] is False
    assert archive.metrics["object_grasp_carry_success_claim"] == 0.0
    assert archive.metrics["p4_2_rollout_claim"] == 0.0

    loaded = read_episode_archives_jsonl(archive_path)

    assert len(loaded) == 1
    assert len(loaded[0].runtime_observations) == 2
    assert len(loaded[0].controller_commands) == 2
    assert len(loaded[0].actuator_target_records) == 2


class _FakeP4_1Backend:
    def __init__(self, *, step_count: int) -> None:
        self.step_count = step_count
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
        return _p4_1_report(step_count=self.step_count, module_count=int(kwargs["module_count"]))


def _p4_1_report(*, step_count: int, module_count: int) -> dict[str, Any]:
    runtime_observations = [
        _runtime_observation(step_idx=step_idx, module_count=module_count)
        for step_idx in range(step_count)
    ]
    controller_commands = [
        ControllerCommand(
            rotor_thrusts_n={f"module_{module_id}:thrust_1": 0.1 for module_id in range(module_count)},
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
            "metrics": {"missing_actuator_count": 0.0, "unsupported_actuator_count": 0.0},
            "metadata": {},
        }
        for step_idx in range(step_count)
    ]
    object_pose_history = [[0.8, 0.0, 0.4 + 0.001 * step_idx, 0.0, 0.0, 0.0, 1.0] for step_idx in range(step_count)]
    return {
        "isaac_backed": False,
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


def _runtime_observation(*, step_idx: int, module_count: int) -> RuntimeObservation:
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    morphology = build_fixed_morphology(physical_model, module_count=module_count, module_spacing_m=0.45)
    module_states = [
        ModuleRuntimeState(
            module_id=module_id,
            pose_world=(0.45 * module_id, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0),
            twist_world=[0.0] * 6,
            joint_positions={
                "gimbal1": 0.01 * step_idx,
                "gimbal2": -0.01 * step_idx,
                "pitch_dock_mech_joint1": 0.0,
                "yaw_dock_mech_joint1": 0.0,
            },
            joint_velocities={
                "gimbal1": 0.0,
                "gimbal2": 0.0,
                "pitch_dock_mech_joint1": 0.0,
                "yaw_dock_mech_joint1": 0.0,
            },
        )
        for module_id in range(module_count)
    ]
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
