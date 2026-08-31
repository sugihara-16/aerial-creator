from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.training.order9_r1_range_selection_v10 import (
    ORDER9_R1_RANGE_LEVEL_IDS,
    Order9R1RangeTeacherScreenPipelineV10,
    chunk_candidate_ids,
    corresponding_minimum_level_candidate_id,
    evaluate_range_level_gate,
    load_minimum_level_teacher_hints,
    load_range_protocol,
    range_case_priority,
)
from amsrr.training.order9_r1_rotational_teacher_pipeline_v7 import (
    Order9R1RotationalTeacherScreenPipelineV7,
)
from scripts.order9_run_r1_range_corner_probe_v10 import _corner_cases
from scripts.order9_run_r1_range_selection_v10 import (
    _ensure_python_hash_seed_zero,
    _has_decisive_failure,
)


def test_corresponding_minimum_level_candidate_preserves_case_identity() -> None:
    assert corresponding_minimum_level_candidate_id(
        "r1_l3_30mm_15deg__train__train-000004-190f3a425b3e__lattice_26"
    ) == ("r1_l1_10mm_5deg__train__train-000004-190f3a425b3e__lattice_26")
    with pytest.raises(SchemaValidationError):
        corresponding_minimum_level_candidate_id(
            "r1_l1_10mm_5deg__train__train-000004-190f3a425b3e__lattice_26"
        )


def test_range_priority_runs_corners_before_interior_and_chunks_at_four() -> None:
    assert range_case_priority("corner", "lattice", 0) < range_case_priority(
        "center", "lattice", 13
    )
    assert range_case_priority("center", "lattice", 13) < range_case_priority(
        "interior", "interior", 0
    )
    assert chunk_candidate_ids(tuple(str(index) for index in range(9))) == (
        ("0", "1", "2", "3"),
        ("4", "5", "6", "7"),
        ("8",),
    )
    with pytest.raises(ValueError):
        chunk_candidate_ids(("a",), maximum_size=5)


def test_corner_probe_selects_eight_lattice_extremes_for_each_source() -> None:
    cases = _corner_cases(ORDER9_R1_RANGE_LEVEL_IDS[0])
    assert len(cases) == 176
    assert {case.sample_index for case in cases} == {0, 2, 6, 8, 18, 20, 24, 26}
    assert {case.sample_kind for case in cases} == {"lattice"}


def test_range_hints_bind_the_minimum_level_selected_contact() -> None:
    hints = load_minimum_level_teacher_hints(
        "configs/training/order9_r1_range_teacher_hints_v10.json",
        repository_root=".",
    )
    assert hints["r1_l1_10mm_5deg__train__train-000027-7fe33c662d36__lattice_20"][
        0
    ] == ((19, 22), "slot_0:grasp_pair:0", 0.12, 0.007)
    assert hints["r1_l2_20mm_10deg__train__train-000027-7fe33c662d36__lattice_08"][
        0
    ] == (
        (19, 22),
        "slot_0:grasp_pair:0",
        0.12,
        0.007,
        0.02,
        (0.015, -0.015, 0.0),
    )
    assert (
        len(hints["r1_l2_20mm_10deg__train__train-000027-7fe33c662d36__lattice_08"])
        == 1
    )


def test_range_hint_rejects_combined_contact_offset_over_30mm(tmp_path) -> None:
    payload = json.loads(
        Path("configs/training/order9_r1_range_teacher_hints_v10.json").read_text(
            encoding="utf-8"
        )
    )
    candidate = payload["candidate_fast_paths"][
        "r1_l2_20mm_10deg__train__train-000027-7fe33c662d36__lattice_08"
    ]
    candidate["grasp_contact_height_offset_m"] = 0.03
    candidate["grasp_contact_tangent_offset_world_m"] = [0.02, -0.02, 0.0]
    path = tmp_path / "over_30mm.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SchemaValidationError):
        load_minimum_level_teacher_hints(path, repository_root=".")


def test_range_pipeline_retries_after_post_selection_rejection(monkeypatch) -> None:
    candidate_id = "r1_l2_20mm_10deg__train__train-000027-7fe33c662d36__lattice_08"
    minimum_id = corresponding_minimum_level_candidate_id(candidate_id)
    first = ((19, 22), "slot_0:grasp_pair:0", 0.12, 0.012)
    second = ((21, 29), "slot_0:grasp_pair:3", 0.12, 0.007)
    pipeline = object.__new__(Order9R1RangeTeacherScreenPipelineV10)
    pipeline.teacher_candidate_overrides = {}
    pipeline.teacher_fast_paths = {minimum_id: (first, second)}
    pipeline.strict_preferred_candidate_options = False
    calls = []

    def prepare(_self, case):
        calls.append(
            (
                _self.teacher_candidate_overrides[case.candidate_id],
                _self.strict_preferred_candidate_options,
            )
        )
        return SimpleNamespace(teacher_trajectory_complete=len(calls) == 2)

    monkeypatch.setattr(Order9R1RotationalTeacherScreenPipelineV7, "prepare", prepare)
    result = pipeline.prepare(SimpleNamespace(candidate_id=candidate_id))
    assert result.teacher_trajectory_complete
    assert calls == [((first,), True), ((second,), True)]
    assert candidate_id not in pipeline.teacher_candidate_overrides
    assert not pipeline.strict_preferred_candidate_options


def test_range_specific_fast_paths_do_not_fall_back_to_unbounded_enumeration(
    monkeypatch,
) -> None:
    candidate_id = "r1_l2_20mm_10deg__train__train-000027-7fe33c662d36__lattice_08"
    options = (
        ((19, 22), "slot_0:grasp_pair:0", 0.12, 0.012),
        ((21, 29), "slot_0:grasp_pair:2", 0.12, 0.007),
    )
    pipeline = object.__new__(Order9R1RangeTeacherScreenPipelineV10)
    pipeline.teacher_candidate_overrides = {}
    pipeline.teacher_fast_paths = {candidate_id: options}
    pipeline.strict_preferred_candidate_options = False
    calls = []

    def prepare(_self, case):
        calls.append(_self.teacher_candidate_overrides[case.candidate_id])
        return SimpleNamespace(teacher_trajectory_complete=False)

    monkeypatch.setattr(Order9R1RotationalTeacherScreenPipelineV7, "prepare", prepare)
    result = pipeline.prepare(SimpleNamespace(candidate_id=candidate_id))
    assert not result.teacher_trajectory_complete
    assert calls == [(options[0],), (options[1],)]


def test_valid_single_lattice_failure_stops_range_execution() -> None:
    cases = {"failed": SimpleNamespace(sample_kind="lattice")}
    assert _has_decisive_failure(
        {
            "failed": {
                "valid_evidence": True,
                "candidate_passed": False,
                "safety_failure_count": 2,
                "fallback_count": 0,
            }
        },
        cases,
    )
    assert not _has_decisive_failure(
        {
            "failed": {
                "valid_evidence": False,
                "candidate_passed": False,
                "safety_failure_count": 0,
                "fallback_count": 0,
            }
        },
        cases,
    )


def test_hash_seed_guard_does_not_restart_when_seed_is_zero(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr("os.execve", lambda *_args: pytest.fail("unexpected restart"))
    _ensure_python_hash_seed_zero()


def _entry(*, sample_kind: str, passed: bool = True) -> dict[str, object]:
    return {
        "sample_kind": sample_kind,
        "teacher_feasible": True,
        "screen_passed": True,
        "candidate_passed": passed,
        "episode_count": 2,
        "success_count": 2 if passed else 1,
        "safety_failure_count": 0,
        "fallback_count": 0,
    }


def _evaluate(entries):
    return evaluate_range_level_gate(
        level_id=ORDER9_R1_RANGE_LEVEL_IDS[0],
        entries=entries,
        minimum_lattice_pass_rate=1.0,
        minimum_teacher_feasibility_rate=0.95,
        minimum_fast_screen_pass_rate=0.95,
        minimum_isaac_success_rate=0.95,
        minimum_interior_teacher_feasibility_rate=0.95,
        minimum_interior_fast_screen_pass_rate=0.95,
        minimum_interior_isaac_success_rate=0.95,
        maximum_safety_failure_count=0,
        maximum_fallback_rate=0.0,
    )


def test_range_gate_accepts_complete_success_and_rejects_lattice_failure() -> None:
    accepted = _evaluate(
        [_entry(sample_kind="lattice"), _entry(sample_kind="interior")]
    )
    assert accepted.passed
    assert accepted.total_success_count == 4

    rejected = _evaluate(
        [_entry(sample_kind="lattice", passed=False), _entry(sample_kind="interior")]
    )
    assert not rejected.passed
    assert "lattice_pass_rate" in rejected.failure_reasons


def test_range_protocol_requires_bounded_restartable_control_contract(tmp_path) -> None:
    payload = {
        "protocol_version": "order9_r1_range_selection_protocol_v10",
        "status": "approved",
        "selection_level_ids": list(ORDER9_R1_RANGE_LEVEL_IDS),
        "maximum_candidates_per_isaac_scene": 4,
        "isaac_environment_spacing_m": 3.0,
        "preparation_numeric_library_threads_per_worker": 2,
        "python_hash_seed": 0,
        "pi_l_actor_command_applied": False,
        "qpid_qp_applied": True,
        "local_servo_applied": True,
        "stop_on_first_failed_level": True,
        "accept_largest_consecutive_passing_level": True,
        "restart_reuses_validated_evidence": True,
        "single_candidate_confirmation_on_bounded_failure": True,
        "calibration_evidence_training_eligible": False,
        "held_out_confirmation_in_scope": False,
        "formal_teacher_collection_authorized": False,
    }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_range_protocol(path) == payload

    payload["maximum_candidates_per_isaac_scene"] = 5
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SchemaValidationError):
        load_range_protocol(path)
