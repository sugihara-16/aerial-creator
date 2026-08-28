from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from amsrr.schemas.contact_candidates import ContactCandidateSet
from amsrr.schemas.policies import ContactWrenchTrajectory
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.order9_r1_grasp_rotation import (
    audit_order9_r1_grasp_rotation_margin,
)

_FAILED_CASE = (
    "artifacts/p4_full/order9/r1_teacher/"
    "calibration_v8_short_retreat_safe_timing/selection/"
    "r1_l1_10mm_5deg/"
    "r1_l1_10mm_5deg__train__train-000004-190f3a425b3e__lattice_02"
)
_NOMINAL = _FAILED_CASE + "/nominal_set/buckets/" + _FAILED_CASE.rsplit("/", 1)[-1]


def _load_failed_case_geometry():
    task_spec = TaskSpec.from_json(
        Path(_FAILED_CASE + "/task_spec.json").read_text(encoding="utf-8")
    )
    candidates = ContactCandidateSet.from_json(
        Path(_NOMINAL + "/contact_candidate_set.json").read_text(encoding="utf-8")
    )
    trajectory = ContactWrenchTrajectory.from_json(
        Path(_NOMINAL + "/phases/lift.json").read_text(encoding="utf-8")
    )
    return task_spec, candidates, trajectory


def test_r1_rotation_gate_rejects_known_y_face_pitch_slip_geometry() -> None:
    task_spec, candidates, trajectory = _load_failed_case_geometry()

    result = audit_order9_r1_grasp_rotation_margin(
        task_spec=task_spec,
        contact_candidate_set=candidates,
        trajectory=trajectory,
        selected_candidate_group_id="slot_0:grasp_pair:3",
        required_rotation_axis_object=(0.0, 1.0, 0.0),
        minimum_force_moment_arm_m=0.10,
    )

    assert result["accepted"] is False
    assert result["minimum_force_moment_arm_observed_m"] < 1.0e-9
    assert result["violation_codes"] == ["E_R1_GRASP_ROTATION_FORCE_MOMENT_ARM"]
    assert result["isaac_invoked"] is False


def test_r1_rotation_gate_accepts_x_face_replacement_geometry() -> None:
    task_spec, candidates, trajectory = _load_failed_case_geometry()
    replacement = deepcopy(trajectory)
    replacement_by_anchor = {0: 0, 1: 3}
    for knot in replacement.knots:
        for assignment in knot.contact_assignments:
            if assignment.anchor_id in replacement_by_anchor:
                assignment.candidate_id = replacement_by_anchor[assignment.anchor_id]

    result = audit_order9_r1_grasp_rotation_margin(
        task_spec=task_spec,
        contact_candidate_set=candidates,
        trajectory=replacement,
        selected_candidate_group_id="slot_0:grasp_pair:0",
        required_rotation_axis_object=(0.0, 1.0, 0.0),
        minimum_force_moment_arm_m=0.10,
    )

    assert result["accepted"] is True
    assert result["minimum_force_moment_arm_observed_m"] > 0.14
    assert result["violation_codes"] == []
    assert result["dynamic_success_claimed"] is False
