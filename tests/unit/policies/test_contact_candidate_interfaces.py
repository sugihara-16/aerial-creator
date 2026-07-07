from __future__ import annotations

from amsrr.policies.assignment_feasibility import ASSIGNMENT_QP_INFEASIBLE_CODE, evaluate_assignment_level_qp
from amsrr.policies.contact_candidate_set import build_contact_candidate_set
from amsrr.schemas.common import ContactMode
from amsrr.schemas.contact_candidates import ContactCandidate
from amsrr.schemas.policies import ContactAssignment


def _candidate(candidate_id: int, *, slot_id: int, anchor_id: int) -> ContactCandidate:
    return ContactCandidate(
        candidate_id=candidate_id,
        slot_id=slot_id,
        anchor_id=anchor_id,
        target_entity_id="box_01",
        region_id=f"box_01_face_{candidate_id}",
        contact_pose_world=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        contact_frame_world=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        normal_world=(0.0, 0.0, 1.0),
        tangent_basis_world=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        contact_mode=ContactMode.GRASP,
        friction=0.6,
        patch_area_m2=0.01,
        candidate_scores={"mode_match": 1.0},
        unary_valid=True,
    )


def test_contact_candidate_pairwise_conflict_matrix() -> None:
    candidate_set = build_contact_candidate_set(
        set_id="set_001",
        task_id="task_001",
        morphology_graph_id="morph_001",
        candidates=[
            _candidate(0, slot_id=0, anchor_id=0),
            _candidate(1, slot_id=0, anchor_id=1),
            _candidate(2, slot_id=1, anchor_id=0),
        ],
    )

    assert candidate_set.pairwise_conflict_matrix == [
        [False, False, True],
        [False, False, False],
        [True, False, False],
    ]
    assert candidate_set.pairwise_compatibility_score[0][1] == 0.75
    assert candidate_set.pairwise_compatibility_score[0][2] == 0.0
    assert candidate_set.slot_coverage == {0: [0, 1], 1: [2]}


def test_assignment_level_qp_infeasible_case() -> None:
    candidate_set = build_contact_candidate_set(
        set_id="set_001",
        task_id="task_001",
        morphology_graph_id="morph_001",
        candidates=[
            _candidate(0, slot_id=0, anchor_id=0),
            _candidate(1, slot_id=0, anchor_id=1),
        ],
    )
    assignments = [
        ContactAssignment(
            slot_id=0,
            anchor_id=0,
            candidate_id=0,
            contact_mode=ContactMode.GRASP,
            schedule_state="maintain",
            wrench_target=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ),
        ContactAssignment(
            slot_id=0,
            anchor_id=1,
            candidate_id=1,
            contact_mode=ContactMode.GRASP,
            schedule_state="maintain",
            wrench_target=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ),
    ]

    result = evaluate_assignment_level_qp(
        assignments,
        candidate_set,
        qp_residual=0.25,
        qp_residual_threshold=0.01,
        active_phase_id=3,
    )

    assert result.feasible is False
    assert result.qp_residual == 0.25
    assert result.violation_codes == [ASSIGNMENT_QP_INFEASIBLE_CODE]
    assert result.assignment_key in candidate_set.assignment_feasibility_cache
