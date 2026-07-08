from __future__ import annotations

from amsrr.schemas.contact_candidates import AssignmentFeasibilityResult, ContactCandidateSet
from amsrr.schemas.policies import ContactAssignment


ASSIGNMENT_QP_INFEASIBLE_CODE = "E_ASSIGNMENT_QP_INFEASIBLE"
ASSIGNMENT_WRENCH_INFEASIBLE_CODE = "E_ASSIGNMENT_WRENCH_INFEASIBLE"
CONTACT_CANDIDATE_PAIR_CONFLICT_CODE = "E_CONTACT_CANDIDATE_PAIR_CONFLICT"
CONTACT_CANDIDATE_UNARY_INVALID_CODE = "E_CONTACT_CANDIDATE_UNARY_INVALID"
CONTACT_GROUP_INSUFFICIENT_CODE = "E_CONTACT_GROUP_INSUFFICIENT"
COLLISION_MARGIN_FAIL_CODE = "E_COLLISION_MARGIN_FAIL"


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


def evaluate_selected_assignment_feasibility(
    assignments: list[ContactAssignment],
    candidate_set: ContactCandidateSet,
    *,
    slot_min_counts: dict[int, int] | None = None,
    slot_max_counts: dict[int, int] | None = None,
    qp_residual: float | None = None,
    qp_residual_threshold: float = 1.0e-6,
    wrench_residual: float | None = None,
    wrench_residual_threshold: float = 1.0e-6,
    min_required_friction: float = 0.05,
    min_collision_margin_m: float | None = None,
    collision_margin_threshold_m: float = 0.0,
    opposing_normal_dot_threshold: float = -0.25,
    active_phase_id: int | None = None,
    update_cache: bool = True,
) -> AssignmentFeasibilityResult:
    """Evaluate a selected assignment set without enumerating candidate subsets."""

    key = assignment_key_from_assignments(assignments, active_phase_id=active_phase_id)
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidate_set.candidates}
    candidate_index_by_id = {candidate.candidate_id: idx for idx, candidate in enumerate(candidate_set.candidates)}
    violation_codes: list[str] = []

    selected_candidate_ids: list[int] = []
    for assignment in assignments:
        candidate = candidate_by_id.get(assignment.candidate_id)
        if candidate is None:
            _append_unique(violation_codes, CONTACT_CANDIDATE_UNARY_INVALID_CODE)
            continue
        selected_candidate_ids.append(candidate.candidate_id)
        if not candidate.unary_valid:
            _append_unique(violation_codes, CONTACT_CANDIDATE_UNARY_INVALID_CODE)
        if (
            candidate.slot_id != assignment.slot_id
            or candidate.anchor_id != assignment.anchor_id
            or candidate.contact_mode != assignment.contact_mode
        ):
            _append_unique(violation_codes, CONTACT_CANDIDATE_UNARY_INVALID_CODE)

    _check_slot_cardinality(assignments, slot_min_counts or {}, slot_max_counts or {}, violation_codes)
    _check_pairwise_conflicts(selected_candidate_ids, candidate_index_by_id, candidate_set, violation_codes)

    selected_candidates = [candidate_by_id[candidate_id] for candidate_id in sorted(set(selected_candidate_ids)) if candidate_id in candidate_by_id]
    friction_margin = _minimum_friction_margin(selected_candidates, min_required_friction)
    if friction_margin is not None and friction_margin < 0.0:
        _append_unique(violation_codes, ASSIGNMENT_WRENCH_INFEASIBLE_CODE)

    computed_wrench_residual = (
        float(wrench_residual)
        if wrench_residual is not None
        else _grasp_opposition_residual(assignments, candidate_by_id, opposing_normal_dot_threshold)
    )
    if computed_wrench_residual > wrench_residual_threshold:
        _append_unique(violation_codes, ASSIGNMENT_WRENCH_INFEASIBLE_CODE)

    if qp_residual is not None and qp_residual > qp_residual_threshold:
        _append_unique(violation_codes, ASSIGNMENT_QP_INFEASIBLE_CODE)

    if min_collision_margin_m is not None and min_collision_margin_m < collision_margin_threshold_m:
        _append_unique(violation_codes, COLLISION_MARGIN_FAIL_CODE)

    result = AssignmentFeasibilityResult(
        assignment_key=key,
        candidate_ids=sorted(set(selected_candidate_ids)),
        feasible=not violation_codes,
        violation_codes=violation_codes,
        wrench_residual=computed_wrench_residual,
        qp_residual=qp_residual,
        min_friction_margin=friction_margin,
        min_collision_margin_m=min_collision_margin_m,
    )
    if update_cache:
        candidate_set.assignment_feasibility_cache[key] = result
    return result


def _check_slot_cardinality(
    assignments: list[ContactAssignment],
    slot_min_counts: dict[int, int],
    slot_max_counts: dict[int, int],
    violation_codes: list[str],
) -> None:
    counts: dict[int, int] = {}
    for assignment in assignments:
        counts[assignment.slot_id] = counts.get(assignment.slot_id, 0) + 1
    for slot_id, min_count in sorted(slot_min_counts.items()):
        if counts.get(slot_id, 0) < min_count:
            _append_unique(violation_codes, CONTACT_GROUP_INSUFFICIENT_CODE)
    for slot_id, max_count in sorted(slot_max_counts.items()):
        if counts.get(slot_id, 0) > max_count:
            _append_unique(violation_codes, CONTACT_GROUP_INSUFFICIENT_CODE)


def _check_pairwise_conflicts(
    selected_candidate_ids: list[int],
    candidate_index_by_id: dict[int, int],
    candidate_set: ContactCandidateSet,
    violation_codes: list[str],
) -> None:
    unique_ids = sorted(set(selected_candidate_ids))
    if len(unique_ids) != len(selected_candidate_ids):
        _append_unique(violation_codes, CONTACT_CANDIDATE_PAIR_CONFLICT_CODE)
    for left_index, left_id in enumerate(unique_ids):
        if left_id not in candidate_index_by_id:
            continue
        for right_id in unique_ids[left_index + 1 :]:
            if right_id not in candidate_index_by_id:
                continue
            left_matrix_idx = candidate_index_by_id[left_id]
            right_matrix_idx = candidate_index_by_id[right_id]
            if candidate_set.pairwise_conflict_matrix[left_matrix_idx][right_matrix_idx]:
                _append_unique(violation_codes, CONTACT_CANDIDATE_PAIR_CONFLICT_CODE)


def _minimum_friction_margin(candidates: list, min_required_friction: float) -> float | None:
    frictions = [candidate.friction for candidate in candidates if candidate.friction is not None]
    if not frictions:
        return None
    return min(frictions) - min_required_friction


def _grasp_opposition_residual(
    assignments: list[ContactAssignment],
    candidate_by_id: dict[int, object],
    opposing_normal_dot_threshold: float,
) -> float:
    residual = 0.0
    assignments_by_slot: dict[int, list[ContactAssignment]] = {}
    for assignment in assignments:
        if assignment.contact_mode.value != "grasp":
            continue
        assignments_by_slot.setdefault(assignment.slot_id, []).append(assignment)
    for slot_assignments in assignments_by_slot.values():
        if len(slot_assignments) < 2:
            continue
        best_dot: float | None = None
        for left_idx, left_assignment in enumerate(slot_assignments):
            left_candidate = candidate_by_id.get(left_assignment.candidate_id)
            if left_candidate is None:
                continue
            for right_assignment in slot_assignments[left_idx + 1 :]:
                right_candidate = candidate_by_id.get(right_assignment.candidate_id)
                if right_candidate is None:
                    continue
                dot_value = _dot(left_candidate.normal_world, right_candidate.normal_world)  # type: ignore[attr-defined]
                best_dot = dot_value if best_dot is None else min(best_dot, dot_value)
        if best_dot is None:
            continue
        residual = max(residual, max(0.0, best_dot - opposing_normal_dot_threshold))
    return residual


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return float(left[0] * right[0] + left[1] * right[1] + left[2] * right[2])


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)
