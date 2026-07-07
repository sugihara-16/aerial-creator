from __future__ import annotations

from amsrr.schemas.contact_candidates import AssignmentFeasibilityResult, ContactCandidateSet
from amsrr.schemas.policies import ContactAssignment


ASSIGNMENT_QP_INFEASIBLE_CODE = "E_ASSIGNMENT_QP_INFEASIBLE"


def assignment_key_from_assignments(
    assignments: list[ContactAssignment],
    *,
    active_phase_id: int | None = None,
) -> str:
    parts = [
        f"{assignment.candidate_id}:{assignment.slot_id}:{assignment.anchor_id}:{assignment.contact_mode.value}:{assignment.schedule_state}"
        for assignment in sorted(assignments, key=lambda item: (item.candidate_id, item.slot_id, item.anchor_id, item.schedule_state))
    ]
    phase = "none" if active_phase_id is None else str(active_phase_id)
    return f"phase={phase}|" + "|".join(parts)


def evaluate_assignment_level_qp(
    assignments: list[ContactAssignment],
    candidate_set: ContactCandidateSet,
    *,
    qp_residual: float,
    qp_residual_threshold: float = 1.0e-6,
    wrench_residual: float | None = None,
    min_friction_margin: float | None = None,
    min_collision_margin_m: float | None = None,
    active_phase_id: int | None = None,
    update_cache: bool = True,
) -> AssignmentFeasibilityResult:
    """Evaluate selected assignments, not every candidate subset."""

    key = assignment_key_from_assignments(assignments, active_phase_id=active_phase_id)
    violation_codes: list[str] = []
    feasible = qp_residual <= qp_residual_threshold
    if not feasible:
        violation_codes.append(ASSIGNMENT_QP_INFEASIBLE_CODE)
    candidate_ids = sorted({assignment.candidate_id for assignment in assignments})
    result = AssignmentFeasibilityResult(
        assignment_key=key,
        candidate_ids=candidate_ids,
        feasible=feasible,
        violation_codes=violation_codes,
        wrench_residual=wrench_residual,
        qp_residual=qp_residual,
        min_friction_margin=min_friction_margin,
        min_collision_margin_m=min_collision_margin_m,
    )
    if update_cache:
        candidate_set.assignment_feasibility_cache[key] = result
    return result
