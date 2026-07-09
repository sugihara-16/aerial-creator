from __future__ import annotations

from dataclasses import replace
from typing import Any

from amsrr.acceptance import run_p4_2_acceptance
from amsrr.schemas.policies import ControllerCommand, ControllerStatus, PolicyCommand
from amsrr.schemas.runtime import ModuleRuntimeState, ObjectRuntimeState, RuntimeObservation, TaskProgressState
from amsrr.simulation import (
    IsaacLabAvailability,
    P4_2AttachConditionReport,
    P4_2AttachEvent,
    P4_2DeterministicRolloutConfig,
    P4_2IsaacEnv,
    P4_2PhaseTransitionRecord,
    P4_2RolloutPhase,
)
from amsrr.training import P4_2DeterministicRolloutRunner, P4_2DeterministicRolloutRunnerConfig


def test_p4_2_fast_gate_does_not_complete_without_real_isaac_rollout() -> None:
    runner_result = _fake_runner_result()

    report = run_p4_2_acceptance(runner_result.archives, rollout_results=[runner_result.rollout_result])

    assert report.fast_gate_passed is True
    assert report.real_isaac_rollout_passed is False
    assert report.completion_passed is False
    assert report.p2_selected_design_used is True
    assert report.p3_assembly_result_used is True
    assert report.trajectory_records_saved is True
    assert report.selected_contact_candidates_saved is True
    assert report.phase_sequence_saved is True
    assert report.per_step_runtime_observations_saved is True
    assert report.per_step_policy_commands_saved is True
    assert report.per_step_controller_commands_saved is True
    assert report.per_step_actuator_target_records_saved is True
    assert report.attach_events_saved is True
    assert report.morphology_reflection_saved is True
    assert "P4.2 real Isaac rollout gate has not passed" in report.failure_reasons
    assert "P4.2 rollout was not Isaac-backed: p2_p3_deterministic_grasp_carry" in report.failure_reasons
    assert report.metrics["completion_passed"] == 0.0


def test_p4_2_completion_requires_real_isaac_rollout() -> None:
    runner_result = _fake_runner_result()
    real_rollout = replace(runner_result.rollout_result, isaac_backed=True)

    report = run_p4_2_acceptance(runner_result.archives, rollout_results=[real_rollout])

    assert report.fast_gate_passed is True
    assert report.real_isaac_rollout_passed is True
    assert report.completion_passed is True
    assert report.failure_reasons == []
    assert report.passed_rollout_names == ["p2_p3_deterministic_grasp_carry"]
    assert type(report).from_json(report.to_json()).to_dict() == report.to_dict()


def test_p4_2_fast_gate_rejects_missing_attach_event() -> None:
    runner_result = _fake_runner_result()
    archive = runner_result.archives[0]
    archive.rollout_artifacts["p4_2_attach_events"] = []
    archive.metrics["attach_event_count"] = 0.0

    report = run_p4_2_acceptance([archive], rollout_results=[runner_result.rollout_result])

    assert report.fast_gate_passed is False
    assert report.completion_passed is False
    assert "P4.2 archives are missing gated object attach events" in report.failure_reasons


def _fake_runner_result():
    backend = _FakeP4_2Backend(step_count=2)
    env_config = P4_2DeterministicRolloutConfig(max_episode_steps=2)
    env = P4_2IsaacEnv(config=env_config, backend=backend)  # type: ignore[arg-type]
    runner = P4_2DeterministicRolloutRunner(
        runner_config=P4_2DeterministicRolloutRunnerConfig(dry_run=False, archive_path=None, seed=0),
        env_config=env_config,
        env=env,
    )
    return runner.run(archive_path=None)


class _FakeP4_2Backend:
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

    def run_p4_2_deterministic_rollout(self, **kwargs: Any) -> dict[str, Any]:
        return _p4_2_report(morphology=kwargs["morphology_graph"], step_count=self.step_count)


def _p4_2_report(*, morphology, step_count: int) -> dict[str, Any]:
    runtime_observations = [
        _runtime_observation(morphology=morphology, step_idx=step_idx)
        for step_idx in range(step_count)
    ]
    policy_commands = [
        PolicyCommand(
            desired_body_pose=(0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0),
            desired_body_twist=[0.0] * 6,
            priority_weights={"p4_2_phase_transport": 1.0},
        ).to_dict()
        for _ in range(step_count)
    ]
    controller_commands = [
        ControllerCommand(
            rotor_thrusts_n={f"module_{module.module_id}:thrust_1": 0.2 for module in morphology.modules},
            vectoring_joint_targets={f"module_{module.module_id}:gimbal1": 0.0 for module in morphology.modules},
            joint_torque_commands={},
            dock_mechanism_commands={
                f"module_{module.module_id}:pitch_dock_mech_joint1": 0.0 for module in morphology.modules
            },
            controller_status=ControllerStatus(status="ok", qp_feasible=True),
        ).to_dict()
        for _ in range(step_count)
    ]
    actuator_records = [
        {
            "time_s": float(step_idx) * 0.005,
            "backend": "isaac_lab",
            "morphology_graph_id": morphology.graph_id,
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
    return {
        "isaac_backed": False,
        "p4_2_deterministic_rollout": True,
        "p4_2_deterministic_rollout_passed": True,
        "p4_2_contact_model": "kinematic_fixed_joint_attach_v1",
        "p4_2_final_phase": "success",
        "p4_2_uses_p2_p3": True,
        "p4_2_morphology_asset_reflected": True,
        "p4_2_module_placement_reflected": True,
        "p4_2_actuator_mapping_reflected": True,
        "p4_2_object_attach_release_only": True,
        "p4_2_module_attach_detach_claim": False,
        "p4_2_dynamic_morphology_update_claim": False,
        "p4_2_asset_generation_semantics": "reset_time_fixed_morphology_not_pi_a_dynamic_construction",
        "p4_2_runtime_observations": [observation.to_dict() for observation in runtime_observations],
        "p4_2_policy_commands": policy_commands,
        "p4_2_controller_commands": controller_commands,
        "p4_2_actuator_target_records": actuator_records,
        "p4_2_phase_transitions": [
            P4_2PhaseTransitionRecord(
                from_phase=P4_2RolloutPhase.RESET,
                to_phase=P4_2RolloutPhase.APPROACH,
                time_s=0.0,
                phase_elapsed_s=0.0,
                reason="reset_complete",
            ).to_dict(),
            P4_2PhaseTransitionRecord(
                from_phase=P4_2RolloutPhase.RELEASE,
                to_phase=P4_2RolloutPhase.SUCCESS,
                time_s=float(step_count) * 0.005,
                phase_elapsed_s=0.005,
                reason="released_at_goal",
            ).to_dict(),
        ],
        "p4_2_attach_events": [_attach_event().to_dict()],
        "p4_2_runtime_observation_count": step_count,
        "p4_2_policy_command_count": step_count,
        "p4_2_controller_command_count": step_count,
        "p4_2_actuator_target_record_count": step_count,
        "success": 1.0,
        "object_drop": 0.0,
        "hard_collision": 0.0,
        "controller_qp_infeasible_terminal": 0.0,
        "p4_full_completion": 0.0,
        "p4_3_learning_bootstrap": 0.0,
        "learned_policy_success_claim": 0.0,
    }


def _runtime_observation(*, morphology, step_idx: int) -> RuntimeObservation:
    return RuntimeObservation(
        time_s=float(step_idx) * 0.005,
        morphology_graph=morphology,
        module_states=[
            ModuleRuntimeState(
                module_id=module.module_id,
                pose_world=module.pose_in_design_frame,
                twist_world=[0.0] * 6,
                joint_positions={"gimbal1": 0.0, "pitch_dock_mech_joint1": 0.0},
                joint_velocities={"gimbal1": 0.0, "pitch_dock_mech_joint1": 0.0},
            )
            for module in morphology.modules
        ],
        object_states=[
            ObjectRuntimeState(
                object_id="box_01",
                pose_world=(0.8, 0.0, 0.4 + 0.01 * step_idx, 0.0, 0.0, 0.0, 1.0),
                twist_world=[0.0] * 6,
            )
        ],
        contact_states=[],
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
        task_progress=TaskProgressState(phase_label="transport", progress_ratio=0.5),
    )


def _attach_event() -> P4_2AttachEvent:
    report = P4_2AttachConditionReport(
        candidate_id=1,
        anchor_id=2,
        slot_id=0,
        object_id="box_01",
        distance_m=0.02,
        relative_velocity_mps=0.01,
        assignment_feasible=True,
        controller_ok=True,
        within_distance=True,
        within_relative_velocity=True,
        passed=True,
    )
    return P4_2AttachEvent(
        time_s=0.01,
        phase=P4_2RolloutPhase.ATTACH_ATTEMPT,
        event_type="attach",
        contact_model="kinematic_fixed_joint_attach_v1",
        object_id="box_01",
        candidate_id=1,
        anchor_id=2,
        slot_id=0,
        contact_pose_world=(0.8, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0),
        anchor_pose_world=(0.79, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0),
        object_pose_world=(0.8, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0),
        distance_m=0.02,
        relative_velocity_mps=0.01,
        assignment_feasible=True,
        controller_ok=True,
        condition_report=report,
    )
