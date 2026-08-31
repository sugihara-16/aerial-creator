from __future__ import annotations

import json
from pathlib import Path

from amsrr.training.order9_r1_calibration import (
    load_order9_r1_calibration_protocol,
)
from amsrr.training.order9_r1_calibration_runner import (
    enumerate_order9_r1_calibration_level_cases,
)
from amsrr.training.order9_r1_complete_task_geometry_v12 import (
    _complete_exact_result,
    _convex_proxy_proves_admission,
)
from amsrr.utils.hashing import hash_file
from scripts import order9_run_r1_range_selection_v11 as runner

ROOT = Path(__file__).resolve().parents[3]


def test_v11_protocol_and_user_approval_are_hash_bound() -> None:
    protocol, approval = runner._load_contract()
    assert approval["approved_protocol"]["sha256"] == hash_file(runner.PROTOCOL)
    assert protocol["selection_bucket_count"] == 22
    assert protocol["confirmation_bucket_count"] == 14
    assert protocol["selection_level_ids"][0] == "r1_l2_20mm_10deg"
    assert protocol["inherited_minimum_level_id"] == "r1_l1_10mm_5deg"
    assert protocol["full_eight_phase_geometry_admission_required"] is True
    assert protocol["anchor_position_tolerance_m"] == 0.03
    assert protocol["anchor_attitude_tolerance_rad"] < 0.053
    assert protocol["pi_l_actor_command_applied"] is False
    assert protocol["formal_teacher_collection_authorized"] is False


def test_v11_rebuilds_level_one_with_corrected_nominal_geometry() -> None:
    cases = runner._corrected_cases("r1_l1_10mm_5deg", "train")
    source_cases = enumerate_order9_r1_calibration_level_cases(
        protocol_path=runner.BASE_PROTOCOL,
        repository_root=ROOT,
        level_id="r1_l1_10mm_5deg",
        split="train",
    )
    assert len(cases) == 22 * 31
    case = cases[0]
    source_case = source_cases[0]
    movable = next(value for value in case.task_spec.scene.objects if value.movable)
    source_movable = next(
        value for value in source_case.task_spec.scene.objects if value.movable
    )
    geometry = next(
        value
        for value in case.task_spec.scene.geometry_library
        if value.geometry_id == movable.geometry_id
    )
    source_geometry = next(
        value
        for value in source_case.task_spec.scene.geometry_library
        if value.geometry_id == source_movable.geometry_id
    )
    assert geometry.primitive_params["size_m"][2] == (
        source_geometry.primitive_params["size_m"][2] + 0.04
    )
    assert case.task_spec.metadata["r1_formal_range_selection_eligible"] is True


def _records(*, lattice_passed: bool = True):
    values = []
    for index in range(31):
        sample_kind = "lattice" if index < 27 else "interior"
        passed = lattice_passed or sample_kind == "interior"
        values.append(
            {
                "sample_kind": sample_kind,
                "teacher_feasible": passed,
                "screen_passed": passed,
                "prepared": passed,
                "candidate_passed": passed,
                "episode_count": 2 if passed else 0,
                "success_count": 2 if passed else 0,
                "safety_failure_count": 0,
                "fallback_count": 0,
            }
        )
    return values


def test_v11_skips_isaac_after_decisive_pre_isaac_lattice_rejection() -> None:
    assert runner._pre_isaac_gate_failed(_records(lattice_passed=False)) is True
    assert runner._pre_isaac_gate_failed(_records()) is False


def test_v11_level_one_uses_the_same_numeric_gate_without_aliasing_result() -> None:
    base = load_order9_r1_calibration_protocol(
        runner.BASE_PROTOCOL,
        repository_root=ROOT,
    )
    gate = runner._gate(base, "r1_l1_10mm_5deg", _records())
    assert gate.passed is True
    assert gate.level_id == "r1_l1_10mm_5deg"


def test_v12_convex_proof_is_conservative_and_exact_fallback_is_complete() -> None:
    separated = {
        "minimum_robot_clearance_m": 0.001,
        "minimum_object_clearance_m": 0.020,
        "minimum_ground_clearance_m": 0.0,
        "selected_contact_penetration_violating_pair_count": 0,
    }
    assert _convex_proxy_proves_admission(
        separated,
        required_clearance_m=0.0195,
    )
    assert not _convex_proxy_proves_admission(
        {**separated, "minimum_robot_clearance_m": -0.001},
        required_clearance_m=0.0195,
    )
    saturated = _complete_exact_result(
        {
            "minimum_robot_clearance_m": -0.001,
            "minimum_object_clearance_m": 0.003,
            "robot_colliding_pair_count": 1,
            "object_colliding_pair_count": 0,
            "worst_pairs": [
                {
                    "second_kind": "robot",
                    "clearance_m": 0.0001,
                    "colliding": False,
                }
            ]
            * 32,
        },
        required_clearance_m=0.0195,
    )
    assert any(value.get("colliding") is True for value in saturated["worst_pairs"])
    assert any(
        value.get("second_kind") == "object" and value["clearance_m"] == 0.003
        for value in saturated["worst_pairs"]
    )


def test_v11_final_rejection_ledger_binds_all_evidence() -> None:
    path = ROOT / "for_codex/R1_RANGE_SELECTION_V11_RESULT_LEDGER.json"
    ledger = json.loads(path.read_text(encoding="utf-8"))
    assert ledger["decision"] == "rejected_before_teacher_collection"
    assert ledger["accepted_range"] is None
    assert ledger["selection"]["passed"] is False
    assert ledger["confirmation"]["passed"] is False
    assert ledger["selection"]["isaac_invoked"] is False
    assert ledger["confirmation"]["isaac_invoked"] is False
    assert ledger["formal_teacher_collection_authorized"] is False
    assert ledger["pi_h_imitation_learning_authorized"] is False
    for binding in ledger["bindings"].values():
        source = ROOT / binding["path"]
        assert source.is_file()
        assert hash_file(source) == binding["sha256"]
