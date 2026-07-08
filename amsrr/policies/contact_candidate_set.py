from __future__ import annotations

from collections import defaultdict

from amsrr.schemas.contact_candidates import ContactCandidate, ContactCandidateGroupProposal, ContactCandidateSet


CONTACT_CANDIDATE_SET_VERSION = "p0_agent_hi_v1"


def build_pairwise_conflict_matrix(candidates: list[ContactCandidate]) -> list[list[bool]]:
    """Build a symmetric pairwise conflict matrix without subset enumeration."""

    matrix = [[False for _ in candidates] for _ in candidates]
    for i, left in enumerate(candidates):
        for j, right in enumerate(candidates):
            if i >= j:
                continue
            conflict = _candidates_conflict(left, right)
            matrix[i][j] = conflict
            matrix[j][i] = conflict
    return matrix


def build_pairwise_compatibility_score(candidates: list[ContactCandidate]) -> list[list[float]]:
    conflict_matrix = build_pairwise_conflict_matrix(candidates)
    scores = [[1.0 for _ in candidates] for _ in candidates]
    for i, left in enumerate(candidates):
        for j, right in enumerate(candidates):
            if i == j:
                scores[i][j] = 1.0
            elif conflict_matrix[i][j]:
                scores[i][j] = 0.0
            elif left.slot_id == right.slot_id and left.anchor_id != right.anchor_id:
                scores[i][j] = 0.75
            else:
                scores[i][j] = 0.5
    return scores


def build_slot_coverage(candidates: list[ContactCandidate]) -> dict[int, list[int]]:
    slot_coverage: dict[int, list[int]] = defaultdict(list)
    for candidate in candidates:
        if candidate.unary_valid:
            slot_coverage[candidate.slot_id].append(candidate.candidate_id)
    return {slot_id: sorted(candidate_ids) for slot_id, candidate_ids in sorted(slot_coverage.items())}


def build_contact_candidate_set(
    *,
    set_id: str,
    task_id: str,
    morphology_graph_id: str,
    candidates: list[ContactCandidate],
    group_proposals: list[ContactCandidateGroupProposal] | None = None,
    sampler_version: str = CONTACT_CANDIDATE_SET_VERSION,
) -> ContactCandidateSet:
    sorted_candidates = sorted(candidates, key=lambda candidate: candidate.candidate_id)
    return ContactCandidateSet(
        set_id=set_id,
        task_id=task_id,
        morphology_graph_id=morphology_graph_id,
        candidates=sorted_candidates,
        candidate_mask=[candidate.unary_valid for candidate in sorted_candidates],
        slot_coverage=build_slot_coverage(sorted_candidates),
        pairwise_conflict_matrix=build_pairwise_conflict_matrix(sorted_candidates),
        pairwise_compatibility_score=build_pairwise_compatibility_score(sorted_candidates),
        group_proposals=sorted(group_proposals or [], key=lambda proposal: proposal.group_id),
        assignment_feasibility_cache={},
        sampler_version=sampler_version,
    )


def _candidates_conflict(left: ContactCandidate, right: ContactCandidate) -> bool:
    if left.candidate_id == right.candidate_id:
        return True
    if left.anchor_id == right.anchor_id:
        return True
    if left.slot_id == right.slot_id and left.anchor_id == right.anchor_id:
        return True
    return False
