from __future__ import annotations

from typing import Any

from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.policies import ControllerCommand, ControllerStatus, PolicyCommand
from amsrr.schemas.runtime import ModuleRuntimeState, ObjectRuntimeState, RuntimeObservation, TaskProgressState
from amsrr.simulation import (
    IsaacLabAvailability,
    IsaacLabBackend,
    IsaacLabBackendConfig,
    P4_2AttachConditionReport,
    P4_2AttachEvent,
    P4_2DeterministicRolloutConfig,
    P4_2IsaacEnv,
    P4_2PhaseTransitionRecord,
    P4_2RolloutPhase,
    load_p4_2_deterministic_rollout_env_config,
    p4_2_result_from_report,
)
from amsrr.simulation.p4_control_controller_smoke import build_fixed_morphology


def test_p4_2_env_config_loader() -> None:
    backend_config, env_config = load_p4_2_deterministic_rollout_env_config(
        "configs/training/p4_2_deterministic_rollout.yaml"
    )

    assert backend_config.micromamba_env == "isaaclab3"
    assert env_config.rollout_name == "p2_p3_deterministic_grasp_carry"
    assert env_config.contact_model == "kinematic_fixed_joint_attach_v1"
    assert env_config.phase_timeouts_s["attach_attempt"] == 2.0


def test_p4_2_backend_command_uses_morphology_graph_json_not_module_count_only() -> None:
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    morphology = build_fixed_morphology(physical_model, module_count=3, module_spacing_m=0.45)
    backend = IsaacLabBackend(IsaacLabBackendConfig())

    command = backend.p4_2_deterministic_rollout_command(
        config_path="configs/env/isaac_lab.yaml",
        convert_if_missing=False,
        force_convert=True,
        steps=20,
        morphology_graph=morphology,
        object_size_m=(0.4, 0.2, 0.1),
        object_mass_kg=1.5,
        object_pose_world=(0.7, 0.1, 0.35, 0.0, 0.0, 0.0, 1.0),
        uses_p2_p3_design=True,
    )

    assert "--p4-2-deterministic-rollout" in command
    assert "--p4-2-morphology-graph-json" in command
    graph_json = command[command.index("--p4-2-morphology-graph-json") + 1]
    assert "fixed-morphology-controller-command-smoke" in graph_json
    assert "--fixed-module-count" not in command
    assert "--p4-2-uses-p2-p3" in command
    assert "--p4-2-object-size-m" in command
    assert "0.4" in command
    assert "--force-convert" in command
    assert "--convert-if-missing" not in command


def test_p4_2_dry_run_and_missing_morphology_do_not_attempt_rollout() -> None:
    env = P4_2IsaacEnv(config=P4_2DeterministicRolloutConfig())

    dry = env.run_rollout(dry_run=True)
    missing = env.run_rollout(dry_run=False, morphology_graph=None)

    assert dry.skipped is True
    assert dry.attempted is False
    assert dry.passed is False
    assert dry.rollout_artifacts["object_attach_release_only"] is True
    assert missing.skipped is True
    assert missing.skip_reason == "missing_morphology_graph"


def test_p4_2_fake_backend_report_parses_rollout_logs_and_attach_event() -> None:
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    morphology = build_fixed_morphology(physical_model, module_count=3, module_spacing_m=0.45)
    report = _p4_2_report(morphology=morphology, step_count=2)
    backend = _P4_2Backend(report=report)
    env = P4_2IsaacEnv(
        config=P4_2DeterministicRolloutConfig(max_episode_steps=2),
        backend=backend,  # type: ignore[arg-type]
    )

    result = env.run_rollout(
        dry_run=False,
        morphology_graph=morphology,
        uses_p2_selected_design=True,
        uses_p3_assembled_morphology=True,
    )

    assert result.attempted is True
    assert result.skipped is False
    assert result.passed is True
    assert result.final_phase == P4_2RolloutPhase.SUCCESS
    assert result.uses_p2_selected_design is True
    assert result.uses_p3_assembled_morphology is True
    assert result.morphology_asset_reflected is True
    assert result.module_placement_reflected is True
    assert result.actuator_mapping_reflected is True
    assert len(result.runtime_observations) == 2
    assert len(result.policy_commands) == 2
    assert len(result.controller_commands) == 2
    assert len(result.actuator_target_records) == 2
    assert len(result.phase_transitions) >= 2
    assert len(result.attach_events) == 1
    assert result.attach_events[0].event_type == "attach"
    assert result.rollout_artifacts["object_attach_release_only"] is True
    assert result.rollout_artifacts["module_attach_detach_claim"] is False
    assert result.rollout_artifacts["dynamic_morphology_update_claim"] is False
    assert (
        result.rollout_artifacts["asset_generation_semantics"]
        == "reset_time_fixed_morphology_not_pi_a_dynamic_construction"
    )
    assert backend.calls[0]["morphology_graph"].graph_id == morphology.graph_id
    assert backend.calls[0]["uses_p2_p3_design"] is True


def test_p4_2_report_with_reflected_graph_but_no_attach_event_cannot_pass() -> None:
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    morphology = build_fixed_morphology(physical_model, module_count=3, module_spacing_m=0.45)
    report = _p4_2_report(morphology=morphology, step_count=2)
    report["p4_2_deterministic_rollout_passed"] = False
    report["p4_2_final_phase"] = "timeout_failure"
    report["p4_2_attach_events"] = []
    report["p4_2_attach_gate_input_available"] = False
    report["p4_2_unconditional_attach_allowed"] = False
    report["success"] = 0.0
    report["timeout_failure"] = 1.0

    result = p4_2_result_from_report(
        report,
        rollout_name="p2_p3_deterministic_grasp_carry",
        uses_p2_selected_design=True,
        uses_p3_assembled_morphology=True,
    )

    assert result.passed is False
    assert result.final_phase == P4_2RolloutPhase.TIMEOUT_FAILURE
    assert result.morphology_asset_reflected is True
    assert result.module_placement_reflected is True
    assert result.actuator_mapping_reflected is True
    assert result.attach_events == []
    assert result.metrics["p4_2_unconditional_attach_allowed"] == 0.0


def test_p4_2_report_without_morphology_reflection_cannot_pass() -> None:
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    morphology = build_fixed_morphology(physical_model, module_count=2, module_spacing_m=0.45)
    report = _p4_2_report(morphology=morphology, step_count=1)
    report["p4_2_morphology_asset_reflected"] = False

    result = p4_2_result_from_report(
        report,
        rollout_name="p2_p3_deterministic_grasp_carry",
        uses_p2_selected_design=True,
        uses_p3_assembled_morphology=True,
    )

    assert result.passed is False
    assert result.skip_reason == "p4_2_rollout_failed"


class _P4_2Backend:
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

    def run_p4_2_deterministic_rollout(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        return self.report


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
            rotor_thrusts_n={f"module_{module_id}:thrust1": 0.2 for module_id in range(len(morphology.modules))},
            vectoring_joint_targets={f"module_{module_id}:gimbal1": 0.0 for module_id in range(len(morphology.modules))},
            joint_torque_commands={},
            dock_mechanism_commands={
                f"module_{module_id}:pitch_dock_mech_joint1": 0.0 for module_id in range(len(morphology.modules))
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
        "isaac_backed": True,
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
