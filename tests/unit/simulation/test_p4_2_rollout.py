from __future__ import annotations

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.policies import ControllerStatus
from amsrr.simulation import (
    P4_2_CONTACT_MODEL,
    P4_2_SUCCESS_SCOPE_NOTE,
    P4_2AttachConditionReport,
    P4_2AttachEvent,
    P4_2DeterministicRolloutConfig,
    P4_2DeterministicRolloutResult,
    P4_2RolloutPhase,
    default_p4_2_phase_definitions,
    evaluate_p4_2_attach_conditions,
    load_p4_2_deterministic_rollout_config,
    p4_2_controller_status_is_fatal,
    p4_2_failure_metrics,
    p4_2_metric_definitions,
    p4_2_no_mislabeling_artifacts,
)


def test_p4_2_config_loader_and_phase_state_machine_contract() -> None:
    config = load_p4_2_deterministic_rollout_config("configs/training/p4_2_deterministic_rollout.yaml")
    phases = default_p4_2_phase_definitions(config)
    by_phase = {definition.phase: definition for definition in phases}

    assert config.contact_model == P4_2_CONTACT_MODEL
    assert config.rollout_name == "p2_p3_deterministic_grasp_carry"
    assert [phase.value for phase in P4_2RolloutPhase] == [
        "reset",
        "approach",
        "pregrasp_align",
        "attach_attempt",
        "attached_maintain",
        "transport",
        "release",
        "success",
        "drop_failure",
        "collision_failure",
        "controller_failure",
        "timeout_failure",
    ]
    assert set(by_phase) == set(P4_2RolloutPhase)
    assert by_phase[P4_2RolloutPhase.RESET].timeout_s == 0.5
    assert by_phase[P4_2RolloutPhase.ATTACH_ATTEMPT].timeout_transition == P4_2RolloutPhase.DROP_FAILURE
    assert by_phase[P4_2RolloutPhase.SUCCESS].terminal is True
    assert by_phase[P4_2RolloutPhase.COLLISION_FAILURE].terminal is True
    assert "snap distance" in by_phase[P4_2RolloutPhase.ATTACH_ATTEMPT].exit_conditions[0]


def test_p4_2_attach_conditions_require_distance_velocity_feasibility_and_controller_status() -> None:
    config = P4_2DeterministicRolloutConfig(
        attach_distance_threshold_m=0.05,
        attach_relative_velocity_threshold_mps=0.10,
    )

    passed = evaluate_p4_2_attach_conditions(
        candidate_id=3,
        anchor_id=2,
        slot_id=1,
        object_id="box_01",
        distance_m=0.03,
        relative_velocity_mps=0.04,
        assignment_feasible=True,
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
        config=config,
    )
    failed = evaluate_p4_2_attach_conditions(
        candidate_id=3,
        anchor_id=2,
        slot_id=1,
        object_id="box_01",
        distance_m=0.08,
        relative_velocity_mps=0.20,
        assignment_feasible=False,
        controller_status=ControllerStatus(status="infeasible", qp_feasible=False),
        config=config,
    )
    degraded = evaluate_p4_2_attach_conditions(
        candidate_id=3,
        anchor_id=2,
        slot_id=1,
        object_id="box_01",
        distance_m=0.03,
        relative_velocity_mps=0.04,
        assignment_feasible=True,
        controller_status=ControllerStatus(
            status="infeasible",
            qp_feasible=False,
            metrics={"qp_solver_success": 1.0, "clipped": 1.0},
        ),
        config=config,
    )

    assert passed.passed is True
    assert passed.failure_reasons == []
    assert passed.metrics["attach_condition_passed"] == 1.0
    assert failed.passed is False
    assert failed.failure_reasons == [
        "anchor_candidate_distance_above_threshold",
        "relative_velocity_above_threshold",
        "attach_snap_distance_above_threshold",
        "assignment_feasibility_failed",
        "controller_status_not_attach_safe",
    ]
    assert failed.metrics["attach_condition_passed"] == 0.0
    assert p4_2_controller_status_is_fatal(ControllerStatus(status="infeasible", qp_feasible=False))
    assert degraded.passed is True
    assert degraded.metrics["attach_controller_ok"] == 1.0


def test_p4_2_metric_contract_excludes_intended_grasp_contacts_from_hard_collision() -> None:
    definitions = p4_2_metric_definitions()

    assert definitions.contact_model == P4_2_CONTACT_MODEL
    assert P4_2_SUCCESS_SCOPE_NOTE == definitions.success_rate_definition
    assert "not high-fidelity natural grasp success" in definitions.success_rate_definition
    assert "not counted as hard_collision" in definitions.intended_contact_exclusion
    assert "Intended grasp contacts and kinematic attach contacts are excluded" in definitions.hard_collision_definition


def test_p4_2_no_mislabeling_contract_rejects_learning_and_full_completion_claims() -> None:
    artifacts = p4_2_no_mislabeling_artifacts()

    assert artifacts["phase"] == "P4.2"
    assert artifacts["contact_model"] == P4_2_CONTACT_MODEL
    assert artifacts["is_p4_full_completion"] is False
    assert artifacts["p4_3_learning_bootstrap"] is False
    assert artifacts["learned_policy_success_claim"] is False
    assert artifacts["true_fixed_joint_dynamics_success_claim"] is False
    assert artifacts["checkpoint_claim"] is False
    assert artifacts["reward_curve_training_claim"] is False
    assert artifacts["p4_4_natural_contact_grasp_remaining"] is True


def test_p4_2_success_result_requires_attach_event_and_reflected_p2_p3_morphology() -> None:
    with pytest.raises(SchemaValidationError, match="successful rollout requires at least one attach event"):
        P4_2DeterministicRolloutResult(
            rollout_name="p2_p3_deterministic_grasp_carry",
            attempted=True,
            passed=True,
            skipped=False,
            isaac_backed=True,
            final_phase=P4_2RolloutPhase.SUCCESS,
        )


def test_p4_2_link_backed_attach_event_requires_resolved_isaac_body_evidence() -> None:
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

    with pytest.raises(SchemaValidationError, match="anchor_link_pose_world is required"):
        P4_2AttachEvent(
            time_s=0.01,
            phase=P4_2RolloutPhase.ATTACH_ATTEMPT,
            event_type="attach",
            contact_model=P4_2_CONTACT_MODEL,
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
            anchor_link_id="pitch_dock_mech1",
            anchor_resolved_body_name="module_2__pitch_dock_mech1",
            anchor_pose_source="isaac_link",
        )

    event = P4_2AttachEvent(
        time_s=0.01,
        phase=P4_2RolloutPhase.ATTACH_ATTEMPT,
        event_type="attach",
        contact_model=P4_2_CONTACT_MODEL,
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
        anchor_link_id="pitch_dock_mech1",
        anchor_resolved_body_name="module_2__pitch_dock_mech1",
        anchor_pose_source="isaac_link",
        anchor_link_pose_world=(0.79, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0),
        anchor_local_pose_in_link=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        anchor_link_twist_world=[0.0] * 6,
    )

    assert event.anchor_pose_source == "isaac_link"
    assert event.anchor_resolved_body_name == "module_2__pitch_dock_mech1"


def test_p4_2_failure_metrics_define_terminal_rates_without_learning_claims() -> None:
    metrics = p4_2_failure_metrics(final_phase=P4_2RolloutPhase.CONTROLLER_FAILURE)

    assert metrics["success"] == 0.0
    assert metrics["controller_qp_infeasible_terminal"] == 1.0
    assert metrics["object_drop"] == 0.0
    assert metrics["hard_collision"] == 0.0
    assert metrics["p4_full_completion"] == 0.0
    assert metrics["p4_3_learning_bootstrap"] == 0.0
    assert metrics["learned_policy_success_claim"] == 0.0
    assert metrics["high_fidelity_natural_grasp_success_claim"] == 0.0
    assert metrics["true_fixed_joint_dynamics_success_claim"] == 0.0
