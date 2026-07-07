from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from amsrr.schemas.common import ContactMode, Pose7D, SchemaBase, Vector3, require_len, require_non_empty


@dataclass
class ContactCandidate(SchemaBase):
    candidate_id: int
    slot_id: int
    anchor_id: int
    target_entity_id: str
    region_id: str
    contact_pose_world: Pose7D
    contact_frame_world: Pose7D
    normal_world: Vector3
    tangent_basis_world: list[float]
    contact_mode: ContactMode
    friction: float | None
    patch_area_m2: float
    candidate_scores: dict[str, float]
    unary_valid: bool
    unary_violation_codes: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if min(self.candidate_id, self.slot_id, self.anchor_id) < 0:
            from amsrr.schemas.common import SchemaValidationError

            raise SchemaValidationError("ContactCandidate ids must be non-negative")
        require_non_empty(self.target_entity_id, "ContactCandidate.target_entity_id")
        require_non_empty(self.region_id, "ContactCandidate.region_id")
        require_len(self.contact_pose_world, 7, "ContactCandidate.contact_pose_world")
        require_len(self.contact_frame_world, 7, "ContactCandidate.contact_frame_world")
        require_len(self.normal_world, 3, "ContactCandidate.normal_world")
        require_len(self.tangent_basis_world, 6, "ContactCandidate.tangent_basis_world")
        if self.patch_area_m2 < 0.0:
            from amsrr.schemas.common import SchemaValidationError

            raise SchemaValidationError("ContactCandidate.patch_area_m2 must be non-negative")


@dataclass
class ContactCandidateGroupProposal(SchemaBase):
    group_id: str
    candidate_ids: list[int]
    group_type: Literal["grasp_pair", "multi_grasp", "perch_set", "support_set", "locomotion_stance"]
    group_score: float
    group_violation_codes: list[str] = field(default_factory=list)

    def validate(self) -> None:
        require_non_empty(self.group_id, "ContactCandidateGroupProposal.group_id")


@dataclass
class AssignmentFeasibilityResult(SchemaBase):
    assignment_key: str
    candidate_ids: list[int]
    feasible: bool
    violation_codes: list[str]
    wrench_residual: float | None = None
    qp_residual: float | None = None
    min_friction_margin: float | None = None
    min_collision_margin_m: float | None = None

    def validate(self) -> None:
        require_non_empty(self.assignment_key, "AssignmentFeasibilityResult.assignment_key")


@dataclass
class ContactCandidateSet(SchemaBase):
    set_id: str
    task_id: str
    morphology_graph_id: str
    candidates: list[ContactCandidate]
    candidate_mask: list[bool]
    slot_coverage: dict[int, list[int]]
    pairwise_conflict_matrix: list[list[bool]]
    pairwise_compatibility_score: list[list[float]]
    group_proposals: list[ContactCandidateGroupProposal]
    assignment_feasibility_cache: dict[str, AssignmentFeasibilityResult]
    sampler_version: str

    def validate(self) -> None:
        require_non_empty(self.set_id, "ContactCandidateSet.set_id")
        require_non_empty(self.task_id, "ContactCandidateSet.task_id")
        require_non_empty(self.morphology_graph_id, "ContactCandidateSet.morphology_graph_id")
        require_non_empty(self.sampler_version, "ContactCandidateSet.sampler_version")
        n = len(self.candidates)
        if len(self.candidate_mask) != n:
            from amsrr.schemas.common import SchemaValidationError

            raise SchemaValidationError("ContactCandidateSet.candidate_mask length must match candidates")
        for matrix_name in ("pairwise_conflict_matrix", "pairwise_compatibility_score"):
            matrix = getattr(self, matrix_name)
            if len(matrix) != n or any(len(row) != n for row in matrix):
                from amsrr.schemas.common import SchemaValidationError

                raise SchemaValidationError(f"ContactCandidateSet.{matrix_name} must be square with candidate count")

