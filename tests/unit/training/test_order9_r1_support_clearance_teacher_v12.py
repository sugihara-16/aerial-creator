from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.training.order9_r1_complete_task_materialization_v13 import (
    finalize_order9_r1_v13_materialized_case,
)
from amsrr.training.order9_r1_support_clearance_teacher_v12 import (
    E_R1_SUPPORT_CLEARANCE_BUFFER,
    E_R1_UNCERTIFIED_RIGID_POSE_COPY,
    ORDER9_R1_SUPPORT_CLEARANCE_TEACHER_V12_RELATIVE,
    build_order9_r1_support_clearance_generation_certificate,
    load_order9_r1_support_clearance_teacher_v12_contract,
    order9_r1_vertical_clearance_shift_candidates,
    require_order9_r1_support_clearance_generation_certificate,
    select_order9_r1_maximin_clearance_candidate,
)

ROOT = Path(__file__).resolve().parents[3]


def _contract():
    return load_order9_r1_support_clearance_teacher_v12_contract(
        ORDER9_R1_SUPPORT_CLEARANCE_TEACHER_V12_RELATIVE,
        repository_root=ROOT,
    )


def _accepted_audit(clearance: float = 0.030) -> dict:
    return {
        "accepted": True,
        "status": "accepted",
        "checked_phases": [
            "approach",
            "contact_acquisition",
            "lift",
            "transport",
            "place",
            "release",
            "retreat",
            "settle",
        ],
        "minimum_robot_support_clearance_lower_bound_m": clearance,
        "generation_method": "complete_path_hard_constraint_candidate_selection_v1",
        "vertical_contact_frame_shift_m": 0.0,
        "original_deterministic_ik_solution_rigidly_preserved": True,
        "isaac_invoked": False,
        "controller_layers_invoked": False,
    }


def test_v12_separates_acceptance_and_planning_clearance() -> None:
    contract = _contract()
    assert contract.acceptance_clearance_m == pytest.approx(0.0195)
    assert contract.planning_clearance_m == pytest.approx(0.0300)
    assert contract.actual_planning_buffer_m == pytest.approx(0.0105)
    assert contract.solver_requested_clearance_m == pytest.approx(0.0305)


def test_v12_rejects_configuration_without_the_required_buffer(tmp_path) -> None:
    payload = json.loads(
        (ROOT / ORDER9_R1_SUPPORT_CLEARANCE_TEACHER_V12_RELATIVE).read_text(
            encoding="utf-8"
        )
    )
    payload["planning_clearance_m"] = payload["acceptance_clearance_m"]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SchemaValidationError, match=E_R1_SUPPORT_CLEARANCE_BUFFER):
        load_order9_r1_support_clearance_teacher_v12_contract(
            path,
            repository_root=tmp_path,
        )


def test_v12_maximin_selection_is_deterministic() -> None:
    candidates = (
        ("first", _accepted_audit(0.031)),
        ("rejected", {"accepted": False, "status": "rejected"}),
        ("best", _accepted_audit(0.036)),
        ("same-best-later", _accepted_audit(0.036)),
    )
    index, candidate, audit = select_order9_r1_maximin_clearance_candidate(candidates)
    assert index == 2
    assert candidate == "best"
    assert audit["minimum_robot_support_clearance_lower_bound_m"] == 0.036


def test_v12_vertical_shift_starts_at_smallest_quantized_buffer() -> None:
    audit = {"minimum_robot_support_clearance_lower_bound_m": 0.018085638181918967}
    shifts = order9_r1_vertical_clearance_shift_candidates(
        audit,
        clearance_contract=_contract(),
    )
    assert shifts == pytest.approx((0.015, 0.020, 0.025, 0.030))


def test_v12_certificate_binds_all_phases_and_both_clearances() -> None:
    contract = _contract()
    certificate = build_order9_r1_support_clearance_generation_certificate(
        _accepted_audit(),
        clearance_contract=contract,
        candidate_id="candidate",
        candidate_rank=0,
        evaluated_candidate_count=1,
    )
    require_order9_r1_support_clearance_generation_certificate(
        certificate,
        clearance_contract=contract,
        candidate_id="candidate",
    )
    assert certificate["unsupported_rigid_pose_copy"] is False
    assert certificate["planning_buffer_m"] == pytest.approx(0.0105)


@pytest.mark.parametrize(
    "method",
    (
        "analytic_vertical_contact_frame_fine_shift_v13",
        "analytic_planar_contact_frame_shift_v13",
    ),
)
def test_v13_certificate_accepts_bounded_exact_margin_repairs(method: str) -> None:
    contract = _contract()
    audit = _accepted_audit()
    audit["generation_method"] = method
    certificate = build_order9_r1_support_clearance_generation_certificate(
        audit,
        clearance_contract=contract,
        candidate_id="candidate",
        candidate_rank=0,
        evaluated_candidate_count=1,
    )

    require_order9_r1_support_clearance_generation_certificate(
        certificate,
        clearance_contract=contract,
        candidate_id="candidate",
    )


def test_v13_rejects_old_rigid_copy_before_materialization(monkeypatch) -> None:
    invoked = False

    def forbidden(*args, **kwargs):
        nonlocal invoked
        invoked = True
        raise AssertionError("v12 materializer must not run")

    monkeypatch.setattr(
        "amsrr.training.order9_r1_complete_task_materialization_v13."
        "finalize_order9_r1_v12_materialized_case",
        forbidden,
    )
    materialized = SimpleNamespace(
        manifest=SimpleNamespace(candidate_id="old-rigid-copy")
    )
    with pytest.raises(
        SchemaValidationError,
        match=E_R1_UNCERTIFIED_RIGID_POSE_COPY,
    ):
        finalize_order9_r1_v13_materialized_case(
            materialized,
            task_spec=None,
            phase_time_scales={},
            joint_rate_limit_rad_s=1.0,
            repository_root=ROOT,
            geometry_contract=None,
            clearance_contract=_contract(),
            clearance_generation_certificate=None,
        )
    assert invoked is False
