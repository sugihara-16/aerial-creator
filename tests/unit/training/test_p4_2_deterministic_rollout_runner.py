from __future__ import annotations

from pathlib import Path
from typing import Any

from amsrr.logging import read_episode_archives_jsonl
from amsrr.schemas.policies import ControllerCommand, ControllerStatus, PolicyCommand
from amsrr.schemas.runtime import ModuleRuntimeState, ObjectRuntimeState, RuntimeObservation, TaskProgressState
from amsrr.simulation import (
    IsaacLabAvailability,
    P4_2AttachConditionReport,
    P4_2AttachEvent,
    P4_2DeterministicRolloutConfig,
    P4_2IsaacEnv,
    P4_2PhaseTransitionRecord,
    P4_2ReleaseEvent,
    P4_2RolloutPhase,
)
from amsrr.training import (
    P4_2DeterministicRolloutRunner,
    P4_2DeterministicRolloutRunnerConfig,
    load_p4_2_deterministic_rollout_runner_config,
)


def test_p4_2_deterministic_rollout_runner_config_loader() -> None:
    runner_config, env_config = load_p4_2_deterministic_rollout_runner_config(
        "configs/training/p4_2_deterministic_rollout.yaml"
    )

    assert runner_config.runner_version == "p4_2_deterministic_rollout_runner_v1"
    assert runner_config.dry_run is True
    assert runner_config.archive_path == "artifacts/p4_2/p4_2_deterministic_rollout.jsonl"
    assert runner_config.p3_config_path == "configs/training/p3_assembly_grasp_carry.yaml"
    assert env_config.rollout_name == "p2_p3_deterministic_grasp_carry"
    assert env_config.contact_model == "kinematic_payload_coupled_attach_v1"


def test_p4_2_runner_builds_p2_p3_case_candidates_and_trajectory() -> None:
    runner = P4_2DeterministicRolloutRunner(
        runner_config=P4_2DeterministicRolloutRunnerConfig(dry_run=True, archive_path=None, seed=0)
    )

    case = runner.build_p2_p3_rollout_case()

    assert case.selection.selected_candidate.feasibility_result.feasible is True
    assert case.assembly_report.success is True
    assert case.module_count == 3
    assert case.contact_candidate_set.morphology_graph_id == case.assembled_morphology.graph_id
    assert len(case.contact_candidate_set.candidates) > 0
    assert case.trajectory.derived_mode_label == "p4_2_deterministic_grasp_carry"
    assert [guard["phase"] for guard in (knot.guard_conditions[0] for knot in case.trajectory.knots)] == [
        "approach",
        "pregrasp_align",
        "attach_attempt",
        "attached_maintain",
        "transport",
        "release",
    ]


def test_p4_2_runner_archives_fake_backend_rollout_without_real_completion_claim(tmp_path: Path) -> None:
    backend = _FakeP4_2Backend(step_count=2)
    env_config = P4_2DeterministicRolloutConfig(max_episode_steps=2)
    env = P4_2IsaacEnv(config=env_config, backend=backend)  # type: ignore[arg-type]
    runner = P4_2DeterministicRolloutRunner(
        runner_config=P4_2DeterministicRolloutRunnerConfig(dry_run=False, archive_path=None, seed=0),
        env_config=env_config,
        env=env,
    )
    archive_path = tmp_path / "p4_2_deterministic_rollout.jsonl"

    result = runner.run(archive_path=archive_path)

    assert result.dry_run is False
    assert result.rollout_result.passed is True
    assert result.rollout_result.isaac_backed is False
    assert result.acceptance_report.fast_gate_passed is True
    assert result.acceptance_report.real_isaac_rollout_passed is False
    assert result.acceptance_report.completion_passed is False
    assert result.metrics["p2_selected_design_used"] == 1.0
    assert result.metrics["p3_assembly_result_used"] == 1.0
    assert result.metrics["fast_gate_passed"] == 1.0
    assert result.metrics["completion_passed"] == 0.0
    assert result.metrics["p4_full_completion"] == 0.0
    assert result.metrics["p4_3_learning_bootstrap"] == 0.0
    assert result.metrics["real_isaac_completion_claim"] == 0.0
    assert backend.calls[0]["morphology_graph"].graph_id == result.archives[0].rollout_artifacts[
        "assembled_morphology_graph_id"
    ]
    assert backend.calls[0]["uses_p2_p3_design"] is True

    assert len(result.archives) == 1
    archive = result.archives[0]
    assert archive.design_output is not None
    assert archive.feasibility_result is not None
    assert archive.assembly_plan is not None
    assert len(archive.trajectory_records) == 1
    assert len(archive.runtime_observations) == 2
    assert len(archive.policy_commands) == 2
    assert len(archive.controller_commands) == 2
    assert len(archive.actuator_target_records) == 2
    assert archive.rollout_artifacts["archive_type"] == "p4_2_deterministic_rollout_per_step"
    assert archive.rollout_artifacts["p2_selected_design_used"] is True
    assert archive.rollout_artifacts["p3_assembled_morphology_used"] is True
    assert archive.rollout_artifacts["object_attach_release_only"] is True
    assert len(archive.rollout_artifacts["p4_2_release_events"]) == 1
    assert archive.rollout_artifacts["module_attach_detach_claim"] is False
    assert archive.rollout_artifacts["dynamic_morphology_update_claim"] is False
    assert archive.rollout_artifacts["real_isaac_completion_claim"] is False
    assert archive.rollout_artifacts["p4_3_learning_bootstrap"] is False
    assert archive.rollout_artifacts["true_fixed_joint_dynamics_success_claim"] is False
    assert archive.rollout_artifacts["p4_4_natural_contact_grasp_remaining"] is True
    assert archive.rollout_artifacts["checkpoint_claim"] is False
    assert archive.rollout_artifacts["reward_curve_training_claim"] is False
    assert archive.metrics["isaac_backed"] == 0.0
    assert archive.metrics["p4_2_deterministic_rollout_passed"] == 1.0
    assert archive.metrics["learned_policy_success_claim"] == 0.0
    assert archive.metrics["high_fidelity_natural_grasp_success_claim"] == 0.0
    assert archive.metrics["true_fixed_joint_dynamics_success_claim"] == 0.0
    assert archive.metrics["p4_4_natural_contact_grasp_remaining"] == 1.0

    loaded = read_episode_archives_jsonl(archive_path)

    assert len(loaded) == 1
    assert len(loaded[0].trajectory_records) == 1
    assert len(loaded[0].runtime_observations) == 2
    assert loaded[0].rollout_artifacts["real_isaac_completion_claim"] is False


class _FakeP4_2Backend:
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

    def run_p4_2_deterministic_rollout(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
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
            controller_status=ControllerStatus(
                status="ok",
                qp_feasible=True,
                metrics=_payload_controller_metrics(),
            ),
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
            "metrics": {
                "allocation_residual_norm": 0.0,
                "missing_actuator_count": 0.0,
                "unsupported_actuator_count": 0.0,
                "clipped_target_count": 0.0,
            },
            "metadata": {},
        }
        for step_idx in range(step_count)
    ]
    return {
        "isaac_backed": False,
        "p4_2_deterministic_rollout": True,
        "p4_2_deterministic_rollout_passed": True,
        "p4_2_contact_model": "kinematic_payload_coupled_attach_v1",
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
        "p4_2_release_events": [_release_event().to_dict()],
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
        attach_snap_distance_m=0.01,
        relative_pose_error_m=0.01,
        assignment_feasible=True,
        controller_ok=True,
        within_distance=True,
        within_relative_velocity=True,
        within_attach_snap_distance=True,
        within_attach_phase_timeout=True,
        passed=True,
    )
    return P4_2AttachEvent(
        time_s=0.01,
        phase=P4_2RolloutPhase.ATTACH_ATTEMPT,
        event_type="attach",
        contact_model="kinematic_payload_coupled_attach_v1",
        object_id="box_01",
        candidate_id=1,
        anchor_id=2,
        slot_id=0,
        contact_pose_world=(0.8, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0),
        anchor_pose_world=(0.79, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0),
        object_pose_world=(0.8, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0),
        distance_m=0.02,
        relative_velocity_mps=0.01,
        attach_snap_distance_m=0.01,
        relative_pose_error_m=0.01,
        assignment_feasible=True,
        controller_ok=True,
        condition_report=report,
        candidate_ids=[1],
        anchor_ids=[2],
        slot_ids=[0],
        contact_region_ids=["box_01:region_0"],
        distance_margins={"anchor_candidate_distance_margin_m": 0.04},
        assignment_feasibility={"feasible": True},
    )


def _release_event() -> P4_2ReleaseEvent:
    return P4_2ReleaseEvent(
        release_time_s=0.02,
        phase=P4_2RolloutPhase.RELEASE,
        event_type="release",
        contact_model="kinematic_payload_coupled_attach_v1",
        object_id="box_01",
        object_pose_world=(2.0, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0),
        robot_pose_world=(2.0, 0.0, 0.6, 0.0, 0.0, 0.0, 1.0),
        intended_release=True,
        post_release_object_pose_error_m=0.02,
    )


def _payload_controller_metrics() -> dict[str, float]:
    return {
        "payload_coupled": 1.0,
        "payload_mass_kg": 1.0,
        "payload_inertia_body_ixx": 0.01,
        "payload_inertia_body_ixy": 0.0,
        "payload_inertia_body_ixz": 0.0,
        "payload_inertia_body_iyy": 0.02,
        "payload_inertia_body_iyz": 0.0,
        "payload_inertia_body_izz": 0.03,
        "payload_com_offset_body_x": 0.1,
        "payload_com_offset_body_y": 0.0,
        "payload_com_offset_body_z": 0.0,
        "payload_gravity_wrench_body_fx": 0.0,
        "payload_gravity_wrench_body_fy": 0.0,
        "payload_gravity_wrench_body_fz": 9.80665,
        "payload_gravity_wrench_body_tx": 0.0,
        "payload_gravity_wrench_body_ty": -0.980665,
        "payload_gravity_wrench_body_tz": 0.0,
        "target_wrench_body_before_payload_fz": 100.0,
        "target_wrench_body_after_payload_fz": 109.80665,
        "achieved_wrench_body_fz": 109.80665,
        "allocation_residual_norm": 0.0,
        "clipped": 0.0,
        "clipped_target_count": 0.0,
    }
