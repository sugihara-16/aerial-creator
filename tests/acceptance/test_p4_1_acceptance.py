from __future__ import annotations

from dataclasses import replace
from typing import Any

from amsrr.acceptance import run_p4_1_acceptance
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.policies import ControllerCommand, ControllerStatus
from amsrr.schemas.runtime import ModuleRuntimeState, ObjectRuntimeState, RuntimeObservation, TaskProgressState
from amsrr.simulation import IsaacLabAvailability, P4_1FullSceneBackendConfig, P4_1IsaacBackendEnv
from amsrr.simulation.p4_control_controller_smoke import build_fixed_morphology
from amsrr.training import P4_1BackendSmokeRunner, P4_1BackendSmokeRunnerConfig


def test_p4_1_fast_gate_does_not_complete_without_real_isaac_smoke() -> None:
    runner_result = _fake_runner_result()

    report = run_p4_1_acceptance(runner_result.archives, smoke_results=[runner_result.smoke_result])

    assert report.fast_gate_passed is True
    assert report.real_isaac_smoke_passed is False
    assert report.completion_passed is False
    assert report.p2_selected_design_used is True
    assert report.p3_assembly_result_used is True
    assert report.not_fixed_two_module_only is True
    assert report.per_step_runtime_observations_saved is True
    assert report.per_step_controller_commands_saved is True
    assert report.per_step_actuator_target_records_saved is True
    assert report.object_pose_history_saved is True
    assert report.joint_state_preserved is True
    assert "P4.1 real Isaac smoke gate has not passed" in report.failure_reasons
    assert "P4.1 smoke was not Isaac-backed: p2_p3_full_scene_backend" in report.failure_reasons
    assert report.metrics["completion_passed"] == 0.0


def test_p4_1_completion_requires_real_isaac_full_scene_smoke() -> None:
    runner_result = _fake_runner_result()
    real_smoke = replace(runner_result.smoke_result, isaac_backed=True)

    report = run_p4_1_acceptance(runner_result.archives, smoke_results=[real_smoke])

    assert report.fast_gate_passed is True
    assert report.real_isaac_smoke_passed is True
    assert report.completion_passed is True
    assert report.failure_reasons == []
    assert report.passed_smoke_names == ["p2_p3_full_scene_backend"]
    assert type(report).from_json(report.to_json()).to_dict() == report.to_dict()


def test_p4_1_fast_gate_rejects_missing_joint_positions() -> None:
    runner_result = _fake_runner_result()
    archive = runner_result.archives[0]
    archive.runtime_observations[0].module_states[0].joint_positions = {}

    report = run_p4_1_acceptance([archive], smoke_results=[runner_result.smoke_result])

    assert report.fast_gate_passed is False
    assert report.completion_passed is False
    assert "P4.1 RuntimeObservation joint-state preservation failed" in report.failure_reasons


def _fake_runner_result():
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
    return runner.run(archive_path=None)


class _FakeP4_1Backend:
    def __init__(self, *, step_count: int) -> None:
        self.step_count = step_count

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
    return RuntimeObservation(
        time_s=0.005 * step_idx,
        morphology_graph=morphology,
        module_states=[
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
        ],
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
