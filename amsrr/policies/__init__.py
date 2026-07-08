"""Policy-side interface helpers for A-MSRR."""

from amsrr.policies.assignment_feasibility import (
    ASSIGNMENT_QP_INFEASIBLE_CODE,
    ASSIGNMENT_WRENCH_INFEASIBLE_CODE,
    COLLISION_MARGIN_FAIL_CODE,
    CONTACT_CANDIDATE_PAIR_CONFLICT_CODE,
    CONTACT_CANDIDATE_UNARY_INVALID_CODE,
    CONTACT_GROUP_INSUFFICIENT_CODE,
    assignment_key_from_assignments,
    evaluate_assignment_level_qp,
    evaluate_selected_assignment_feasibility,
)
from amsrr.policies.contact_candidate_set import (
    CONTACT_CANDIDATE_SET_VERSION,
    build_contact_candidate_set,
    build_pairwise_compatibility_score,
    build_pairwise_conflict_matrix,
)
from amsrr.policies.contact_candidate_sampler import (
    CONTACT_CANDIDATE_SAMPLER_VERSION,
    ContactCandidateSampler,
    ContactCandidateSamplerConfig,
    build_group_proposals,
)
from amsrr.policies.design_candidate_generator import DesignActionCandidate, DesignCandidateGenerator, DesignCandidateStep
from amsrr.policies.design_policy_base import DesignPolicyBase, DesignPolicyContext, FixedSimpleDesignPolicy
from amsrr.policies.design_teacher import DesignTeacherExample, DesignTeacherVariant, DeterministicDesignTeacher

__all__ = [
    "ASSIGNMENT_QP_INFEASIBLE_CODE",
    "ASSIGNMENT_WRENCH_INFEASIBLE_CODE",
    "COLLISION_MARGIN_FAIL_CODE",
    "CONTACT_CANDIDATE_PAIR_CONFLICT_CODE",
    "CONTACT_CANDIDATE_UNARY_INVALID_CODE",
    "CONTACT_CANDIDATE_SET_VERSION",
    "CONTACT_CANDIDATE_SAMPLER_VERSION",
    "CONTACT_GROUP_INSUFFICIENT_CODE",
    "ContactCandidateSampler",
    "ContactCandidateSamplerConfig",
    "DesignActionCandidate",
    "DesignCandidateGenerator",
    "DesignCandidateStep",
    "DesignPolicyBase",
    "DesignPolicyContext",
    "DesignTeacherExample",
    "DesignTeacherVariant",
    "DeterministicDesignTeacher",
    "FixedSimpleDesignPolicy",
    "assignment_key_from_assignments",
    "build_contact_candidate_set",
    "build_group_proposals",
    "build_pairwise_compatibility_score",
    "build_pairwise_conflict_matrix",
    "evaluate_assignment_level_qp",
    "evaluate_selected_assignment_feasibility",
]
