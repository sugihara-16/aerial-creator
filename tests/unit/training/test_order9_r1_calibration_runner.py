from __future__ import annotations

from amsrr.training import order9_r1_calibration_runner as runner
from amsrr.training.order9_r1_calibration_runner import (
    ORDER9_R1_C3_CHECKPOINT_SHA256,
    ORDER9_R1_FULL_LAYER_EXECUTION_CHAIN,
    Order9R1CalibrationLevelResult,
    Order9R1DeterministicTeacherScreenPipeline,
    Order9R1FullLayerReplayResult,
    Order9R1PreparedCandidate,
    enumerate_order9_r1_calibration_level_cases,
    run_order9_r1_calibration,
)
from amsrr.training.order9_r1_fast_screen import (
    ORDER9_R1_FAST_SCREEN_EXECUTION_MODEL,
    ORDER9_R1_FAST_SCREEN_VERSION,
    Order9R1FastScreenResult,
)

_PROTOCOL = "configs/training/order9_r1_calibration_protocol_v1.yaml"
_BUCKETS = (
    "artifacts/p4_full/order9/releases/c3_pi_l_promoted_update18_v1/runtime/"
    "rollout_buckets_current_lineage_v5/"
    "manifest_c3_release_smooth_clear300_current_config_min2_preload_screened_v2.json"
)


def _level_result(level_id: str, split: str, bucket_count: int, passed: bool):
    lattice = 27 * bucket_count
    interior = 4 * bucket_count
    cases = lattice + interior
    replays = 2 * cases
    return Order9R1CalibrationLevelResult(
        level_id=level_id,
        split=split,
        bucket_count=bucket_count,
        case_count=cases,
        lattice_case_count=lattice,
        interior_case_count=interior,
        support_gate_success_count=cases if passed else cases - 1,
        teacher_success_count=cases if passed else cases - 1,
        fast_screen_attempt_count=cases if passed else cases - 1,
        fast_screen_success_count=cases if passed else cases - 1,
        isaac_attempt_count=replays if passed else replays - 2,
        isaac_success_count=replays if passed else replays - 2,
        lattice_pass_count=lattice if passed else lattice - 1,
        interior_teacher_success_count=interior,
        interior_fast_screen_attempt_count=interior,
        interior_fast_screen_success_count=interior,
        interior_isaac_attempt_count=2 * interior,
        interior_isaac_success_count=2 * interior,
        safety_failure_count=0,
        fallback_count=0,
        minimum_support_projected_com_margin_m=0.01,
        passed=passed,
        failed_gates=() if passed else ("lattice_pass_rate",),
    )


def test_runner_stops_ladder_and_confirms_largest_pass_once(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_run_level(**kwargs):
        level_id = kwargs["level"].level_id
        split = kwargs["split"]
        calls.append((split, level_id))
        passed = split == "validation" or "l3_" not in level_id
        result = _level_result(level_id, split, len(kwargs["buckets"]), passed)
        return result, result.case_count

    monkeypatch.setattr(runner, "_run_level", fake_run_level)
    result = run_order9_r1_calibration(
        protocol_path=_PROTOCOL,
        repository_root=".",
        prepare_candidate=lambda _case: None,  # not reached by the patched level
        fast_screen=lambda _case, _prepared: None,
        evaluate_full_layer=lambda _case, _prepared, _index: None,
    )

    assert calls == [
        ("train", "r1_l1_10mm_5deg"),
        ("train", "r1_l2_20mm_10deg"),
        ("train", "r1_l3_30mm_15deg"),
        ("validation", "r1_l2_20mm_10deg"),
    ]
    assert result.status == "accepted"
    assert result.selected_level_id == "r1_l2_20mm_10deg"
    assert result.accepted_level_id == "r1_l2_20mm_10deg"
    assert result.confirmation_attempt_count == 1


def test_confirmation_failure_rejects_v1_without_stepback(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_run_level(**kwargs):
        level_id = kwargs["level"].level_id
        split = kwargs["split"]
        calls.append((split, level_id))
        passed = split == "train" and "l3_" not in level_id
        result = _level_result(level_id, split, len(kwargs["buckets"]), passed)
        return result, result.case_count

    monkeypatch.setattr(runner, "_run_level", fake_run_level)
    result = run_order9_r1_calibration(
        protocol_path=_PROTOCOL,
        repository_root=".",
        prepare_candidate=lambda _case: None,
        fast_screen=lambda _case, _prepared: None,
        evaluate_full_layer=lambda _case, _prepared, _index: None,
    )

    assert calls[-1] == ("validation", "r1_l2_20mm_10deg")
    assert sum(split == "validation" for split, _level in calls) == 1
    assert result.status == "rejected"
    assert result.selected_level_id == "r1_l2_20mm_10deg"
    assert result.accepted_level_id is None


def test_fast_screen_rejection_never_invokes_full_control() -> None:
    full_control_calls = 0

    def prepare(case):
        return Order9R1PreparedCandidate(
            candidate_id=case.candidate_id,
            candidate_task_hash=case.task_spec.stable_hash(),
            morphology_hash=case.source_bucket.morphology_hash,
            physical_model_hash=case.physical_model_hash,
            teacher_trajectory_complete=True,
            failure_reason=None,
        )

    def reject_screen(case, _prepared):
        return Order9R1FastScreenResult(
            screen_version=ORDER9_R1_FAST_SCREEN_VERSION,
            execution_model=ORDER9_R1_FAST_SCREEN_EXECUTION_MODEL,
            candidate_id=case.candidate_id,
            candidate_task_hash=case.task_spec.stable_hash(),
            morphology_hash=case.source_bucket.morphology_hash,
            physical_model_hash=case.physical_model_hash,
            resolved_path_hash="1" * 64,
            accepted=False,
            eligible_for_full_control_test=False,
            violation_codes=("E_TEST_REJECT",),
            checked_phases=("approach", "contact_acquisition"),
            window_count=2,
            resolved_knot_count=2,
            collision_evidence_window_count=2,
            configuration_plan_collision_check_count=2,
            grasp_pose_reached=False,
            final_maintained_contact_count=0,
            minimum_joint_limit_margin_rad=0.1,
            minimum_normalized_joint_limit_reserve=0.1,
            maximum_body_tilt_rad=0.1,
            maximum_joint_rate_rad_s=0.1,
            minimum_joint_rate_margin_rad_s=0.1,
            minimum_collision_clearance_m=0.1,
            maximum_collision_violating_pair_count=0,
            screen_wall_time_s=0.001,
        )

    def full_control(case, _prepared, replay_index):
        nonlocal full_control_calls
        full_control_calls += 1
        return Order9R1FullLayerReplayResult(
            candidate_id=case.candidate_id,
            replay_index=replay_index,
            success=True,
            safety_failure=False,
            fallback_used=False,
            failure_reason=None,
            execution_chain=ORDER9_R1_FULL_LAYER_EXECUTION_CHAIN,
            c3_checkpoint_sha256=ORDER9_R1_C3_CHECKPOINT_SHA256,
        )

    result = run_order9_r1_calibration(
        protocol_path=_PROTOCOL,
        repository_root=".",
        prepare_candidate=prepare,
        fast_screen=reject_screen,
        evaluate_full_layer=full_control,
    )

    assert result.status == "rejected"
    assert len(result.selection_results) == 1
    assert result.selection_results[0].case_count == 22 * 31
    assert result.selection_results[0].fast_screen_attempt_count == 22 * 31
    assert result.selection_results[0].isaac_attempt_count == 0
    assert result.confirmation_attempt_count == 0
    assert full_control_calls == 0


def test_production_teacher_screen_adapter_constructs_without_running_solver() -> None:
    pipeline = Order9R1DeterministicTeacherScreenPipeline(
        repository_root=".",
        source_bucket_manifest_path=_BUCKETS,
    )

    assert pipeline.physical_model.stable_hash() == (
        "23dfd5a15d170b356e00a188f952afe98b5646b728a245493c36de36f789591a"
    )
    assert pipeline.screen_config.minimum_normalized_joint_limit_reserve == 0.01
    assert pipeline.anchor_position_tolerance_m == 0.03
    assert pipeline.enforce_joint_limit_reserve_during_ik is False


def test_r1_pipeline_applies_reserve_after_nominal_generation(monkeypatch) -> None:
    pipeline = Order9R1DeterministicTeacherScreenPipeline(
        repository_root=".",
        source_bucket_manifest_path=_BUCKETS,
        minimum_normalized_joint_limit_reserve=0.01,
        enforce_joint_limit_reserve_during_ik=True,
    )
    case = enumerate_order9_r1_calibration_level_cases(
        protocol_path=_PROTOCOL,
        repository_root=".",
        level_id="r1_l1_10mm_5deg",
        split="train",
    )[0]
    generated = {}
    projected = {}
    nominal_sentinel = object()
    projected_sentinel = object()

    def fake_generate(**kwargs):
        generated.update(kwargs)
        return nominal_sentinel

    def fake_project(nominal, **kwargs):
        projected["nominal"] = nominal
        projected.update(kwargs)
        return projected_sentinel

    monkeypatch.setattr(
        runner,
        "generate_order9_c3_nominal_grasp_trajectory",
        fake_generate,
    )
    monkeypatch.setattr(
        runner,
        "project_order9_r1_nominal_joint_reserve",
        fake_project,
    )

    prepared = pipeline.prepare(case)

    assert prepared.teacher_trajectory_complete is True
    assert prepared.payload is projected_sentinel
    assert "minimum_normalized_joint_limit_reserve" not in generated
    assert projected["nominal"] is nominal_sentinel
    assert projected["minimum_normalized_joint_limit_reserve"] == 0.01
    assert projected["anchor_position_tolerance_m"] == 0.03


def test_selection_case_keeps_source_split_separate_from_c3_execution_alias() -> None:
    case = enumerate_order9_r1_calibration_level_cases(
        protocol_path=_PROTOCOL,
        repository_root=".",
        level_id="r1_l1_10mm_5deg",
        split="train",
    )[0]

    assert case.split == "train"
    assert case.task_spec.metadata["dataset_split"] == "validation"
    assert case.task_spec.metadata["r1_calibration_source_split"] == "train"
    assert case.task_spec.metadata["r1_calibration_execution_split"] == "validation"
    assert (
        case.task_spec.metadata["r1_calibration_execution_split_alias_contract"]
        == "protected_c3_formal_runner_validation_alias_v1"
    )
