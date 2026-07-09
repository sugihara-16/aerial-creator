from __future__ import annotations

from dataclasses import dataclass

from amsrr.policies.assignment_feasibility import evaluate_selected_assignment_feasibility
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.schemas.common import ContactMode, Pose7D, SchemaValidationError
from amsrr.schemas.contact_candidates import ContactCandidate, ContactCandidateSet
from amsrr.schemas.irg import IRGNodeType
from amsrr.schemas.policies import (
    CentroidalTarget,
    ContactAssignment,
    ContactWrenchTrajectory,
    InteractionKnot,
    ObjectTarget,
    PostureTarget,
)


@dataclass(frozen=True)
class BaselineTrajectoryPlannerConfig:
    horizon_s: float = 2.0
    dt_s: float = 0.25
    grasp_force_n: float = 5.0
    support_force_n: float = 5.0
    max_group_attempts: int = 8


@dataclass(frozen=True)
class P4_2DeterministicPlannerConfig:
    horizon_s: float = 2.0
    dt_s: float = 0.25
    grasp_force_n: float = 5.0
    support_force_n: float = 5.0
    max_group_attempts: int = 8
    approach_height_offset_m: float = 0.15
    min_body_target_height_m: float = 0.35


class GraspCarryBaselinePlanner:
    """Deterministic P1 pi_H baseline for fixed/simple grasp-carry experiments."""

    def __init__(self, config: BaselineTrajectoryPlannerConfig | None = None) -> None:
        self.config = config or BaselineTrajectoryPlannerConfig()

    def plan(self, context: HighLevelPolicyContext) -> ContactWrenchTrajectory:
        slot_min_counts, slot_max_counts = _slot_count_requirements(context)
        maintain_assignments = select_feasible_assignments(
            context.contact_candidate_set,
            slot_min_counts=slot_min_counts,
            slot_max_counts=slot_max_counts,
            max_group_attempts=self.config.max_group_attempts,
        )
        goal_pose = _object_goal_pose(context)
        object_id = _object_id_for_assignments(maintain_assignments, context.contact_candidate_set)
        lift_pose = _lift_pose(context, object_id)

        approach = _assignments_with_state(maintain_assignments, context.contact_candidate_set, "approach", self.config)
        attach = _assignments_with_state(maintain_assignments, context.contact_candidate_set, "attach", self.config)
        maintain = _assignments_with_state(maintain_assignments, context.contact_candidate_set, "maintain", self.config)
        release = _assignments_with_state(maintain_assignments, context.contact_candidate_set, "release", self.config)

        knots = [
            InteractionKnot(
                t_rel_s=0.0,
                contact_assignments=approach,
                centroidal_target=CentroidalTarget(centroidal_wrench_preference=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                posture_target=PostureTarget(free_anchor_pose_targets=_anchor_pose_targets(context.contact_candidate_set, approach)),
                object_targets=[],
                priority_weights={"approach": 1.0, "contact": 0.2},
                guard_conditions=[{"type": "candidate_approach_started"}],
            ),
            InteractionKnot(
                t_rel_s=self.config.dt_s,
                contact_assignments=attach,
                centroidal_target=CentroidalTarget(centroidal_wrench_preference=[0.0, 0.0, self.config.support_force_n, 0.0, 0.0, 0.0]),
                posture_target=PostureTarget(free_anchor_pose_targets=_anchor_pose_targets(context.contact_candidate_set, attach)),
                object_targets=[],
                priority_weights={"contact": 1.0, "posture": 0.5},
                guard_conditions=[{"type": "contact_attach_window"}],
            ),
            InteractionKnot(
                t_rel_s=2.0 * self.config.dt_s,
                contact_assignments=maintain,
                centroidal_target=CentroidalTarget(centroidal_wrench_preference=[0.0, 0.0, self.config.support_force_n, 0.0, 0.0, 0.0]),
                posture_target=PostureTarget(free_anchor_pose_targets=_anchor_pose_targets(context.contact_candidate_set, maintain)),
                object_targets=[ObjectTarget(object_id=object_id, pose_target_world=lift_pose)] if lift_pose is not None else [],
                priority_weights={"contact": 1.0, "object_lift": 1.0, "stability": 0.8},
                guard_conditions=[{"type": "object_lift_active"}],
            ),
            InteractionKnot(
                t_rel_s=max(3.0 * self.config.dt_s, self.config.horizon_s - self.config.dt_s),
                contact_assignments=maintain,
                centroidal_target=CentroidalTarget(centroidal_wrench_preference=[0.0, 0.0, self.config.support_force_n, 0.0, 0.0, 0.0]),
                posture_target=PostureTarget(free_anchor_pose_targets=_anchor_pose_targets(context.contact_candidate_set, maintain)),
                object_targets=[ObjectTarget(object_id=object_id, pose_target_world=goal_pose)] if goal_pose is not None else [],
                priority_weights={"contact": 1.0, "object_goal": 1.0, "stability": 0.8},
                guard_conditions=[{"type": "transport_to_goal"}],
            ),
            InteractionKnot(
                t_rel_s=self.config.horizon_s,
                contact_assignments=release,
                centroidal_target=CentroidalTarget(centroidal_wrench_preference=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                posture_target=PostureTarget(free_anchor_pose_targets={}),
                object_targets=[ObjectTarget(object_id=object_id, pose_target_world=goal_pose)] if goal_pose is not None else [],
                priority_weights={"release": 1.0, "object_goal": 0.8},
                guard_conditions=[{"type": "release_after_place"}],
            ),
        ]
        return ContactWrenchTrajectory(
            horizon_s=self.config.horizon_s,
            dt_s=self.config.dt_s,
            knots=knots,
            derived_mode_label="grasp_carry_baseline",
        )


class P4_2DeterministicGraspCarryPlanner:
    """P4.2 deterministic pi_H plan with explicit rollout-phase guard labels."""

    def __init__(self, config: P4_2DeterministicPlannerConfig | None = None) -> None:
        self.config = config or P4_2DeterministicPlannerConfig()

    def plan(self, context: HighLevelPolicyContext) -> ContactWrenchTrajectory:
        slot_min_counts, slot_max_counts = _slot_count_requirements(context)
        maintain_assignments = select_feasible_assignments(
            context.contact_candidate_set,
            slot_min_counts=slot_min_counts,
            slot_max_counts=slot_max_counts,
            max_group_attempts=self.config.max_group_attempts,
        )
        baseline_config = BaselineTrajectoryPlannerConfig(
            horizon_s=self.config.horizon_s,
            dt_s=self.config.dt_s,
            grasp_force_n=self.config.grasp_force_n,
            support_force_n=self.config.support_force_n,
            max_group_attempts=self.config.max_group_attempts,
        )
        approach = _assignments_with_state(maintain_assignments, context.contact_candidate_set, "approach", baseline_config)
        attach = _assignments_with_state(maintain_assignments, context.contact_candidate_set, "attach", baseline_config)
        maintain = _assignments_with_state(maintain_assignments, context.contact_candidate_set, "maintain", baseline_config)
        release = _assignments_with_state(maintain_assignments, context.contact_candidate_set, "release", baseline_config)
        goal_pose = _object_goal_pose(context)
        object_id = _object_id_for_assignments(maintain_assignments, context.contact_candidate_set)
        lift_pose = _lift_pose(context, object_id)
        contact_centroid = _selected_contact_centroid(context.contact_candidate_set, maintain_assignments)
        approach_body_pose = _p4_2_body_target_pose(
            context,
            contact_centroid=contact_centroid,
            object_goal_pose=goal_pose,
            phase="approach",
            config=self.config,
        )
        pregrasp_body_pose = _p4_2_body_target_pose(
            context,
            contact_centroid=contact_centroid,
            object_goal_pose=goal_pose,
            phase="pregrasp_align",
            config=self.config,
        )
        transport_body_pose = _p4_2_body_target_pose(
            context,
            contact_centroid=contact_centroid,
            object_goal_pose=goal_pose,
            phase="transport",
            config=self.config,
        )
        knots = [
            InteractionKnot(
                t_rel_s=0.0,
                contact_assignments=approach,
                centroidal_target=CentroidalTarget(
                    com_pos_world=approach_body_pose[:3],
                    body_orientation_world=approach_body_pose[3:7],
                    centroidal_wrench_preference=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ),
                posture_target=PostureTarget(free_anchor_pose_targets=_anchor_pose_targets(context.contact_candidate_set, approach)),
                object_targets=[],
                priority_weights={"approach": 1.0, "contact": 0.2, "p4_2_deterministic": 1.0},
                guard_conditions=_p4_2_guards("approach"),
            ),
            InteractionKnot(
                t_rel_s=self.config.dt_s,
                contact_assignments=approach,
                centroidal_target=CentroidalTarget(
                    com_pos_world=pregrasp_body_pose[:3],
                    body_orientation_world=pregrasp_body_pose[3:7],
                    centroidal_wrench_preference=[0.0, 0.0, self.config.support_force_n, 0.0, 0.0, 0.0],
                ),
                posture_target=PostureTarget(free_anchor_pose_targets=_anchor_pose_targets(context.contact_candidate_set, approach)),
                object_targets=[],
                priority_weights={"pregrasp_align": 1.0, "contact": 0.6, "p4_2_deterministic": 1.0},
                guard_conditions=_p4_2_guards("pregrasp_align"),
            ),
            InteractionKnot(
                t_rel_s=2.0 * self.config.dt_s,
                contact_assignments=attach,
                centroidal_target=CentroidalTarget(
                    com_pos_world=pregrasp_body_pose[:3],
                    body_orientation_world=pregrasp_body_pose[3:7],
                    centroidal_wrench_preference=[0.0, 0.0, self.config.support_force_n, 0.0, 0.0, 0.0],
                ),
                posture_target=PostureTarget(free_anchor_pose_targets=_anchor_pose_targets(context.contact_candidate_set, attach)),
                object_targets=[],
                priority_weights={"attach_attempt": 1.0, "contact": 1.0, "p4_2_deterministic": 1.0},
                guard_conditions=_p4_2_guards("attach_attempt"),
            ),
            InteractionKnot(
                t_rel_s=3.0 * self.config.dt_s,
                contact_assignments=maintain,
                centroidal_target=CentroidalTarget(
                    com_pos_world=pregrasp_body_pose[:3],
                    body_orientation_world=pregrasp_body_pose[3:7],
                    centroidal_wrench_preference=[0.0, 0.0, self.config.support_force_n, 0.0, 0.0, 0.0],
                ),
                posture_target=PostureTarget(free_anchor_pose_targets=_anchor_pose_targets(context.contact_candidate_set, maintain)),
                object_targets=[ObjectTarget(object_id=object_id, pose_target_world=lift_pose)] if lift_pose is not None else [],
                priority_weights={"attached_maintain": 1.0, "object_lift": 1.0, "p4_2_deterministic": 1.0},
                guard_conditions=_p4_2_guards("attached_maintain"),
            ),
            InteractionKnot(
                t_rel_s=4.0 * self.config.dt_s,
                contact_assignments=maintain,
                centroidal_target=CentroidalTarget(
                    com_pos_world=transport_body_pose[:3],
                    body_orientation_world=transport_body_pose[3:7],
                    centroidal_wrench_preference=[0.0, 0.0, self.config.support_force_n, 0.0, 0.0, 0.0],
                ),
                posture_target=PostureTarget(free_anchor_pose_targets=_anchor_pose_targets(context.contact_candidate_set, maintain)),
                object_targets=[ObjectTarget(object_id=object_id, pose_target_world=goal_pose)] if goal_pose is not None else [],
                priority_weights={"transport": 1.0, "object_goal": 1.0, "p4_2_deterministic": 1.0},
                guard_conditions=_p4_2_guards("transport"),
            ),
            InteractionKnot(
                t_rel_s=max(5.0 * self.config.dt_s, self.config.horizon_s),
                contact_assignments=release,
                centroidal_target=CentroidalTarget(
                    com_pos_world=transport_body_pose[:3],
                    body_orientation_world=transport_body_pose[3:7],
                    centroidal_wrench_preference=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ),
                posture_target=PostureTarget(free_anchor_pose_targets={}),
                object_targets=[ObjectTarget(object_id=object_id, pose_target_world=goal_pose)] if goal_pose is not None else [],
                priority_weights={"release": 1.0, "object_goal": 0.8, "p4_2_deterministic": 1.0},
                guard_conditions=_p4_2_guards("release"),
            ),
        ]
        return ContactWrenchTrajectory(
            horizon_s=max(self.config.horizon_s, knots[-1].t_rel_s),
            dt_s=self.config.dt_s,
            knots=knots,
            derived_mode_label="p4_2_deterministic_grasp_carry",
        )


def select_feasible_assignments(
    candidate_set: ContactCandidateSet,
    *,
    slot_min_counts: dict[int, int],
    slot_max_counts: dict[int, int],
    max_group_attempts: int = 8,
) -> list[ContactAssignment]:
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidate_set.candidates}
    for candidate_ids in _candidate_id_groups(candidate_set, max_group_attempts=max_group_attempts):
        assignments = [_assignment_from_candidate(candidate_by_id[candidate_id], "maintain", BaselineTrajectoryPlannerConfig()) for candidate_id in candidate_ids if candidate_id in candidate_by_id]
        if not assignments:
            continue
        result = evaluate_selected_assignment_feasibility(
            assignments,
            candidate_set,
            slot_min_counts=slot_min_counts,
            slot_max_counts=slot_max_counts,
        )
        if result.feasible:
            return assignments
    raise SchemaValidationError("No feasible selected ContactAssignment set found for baseline pi_H")


def _candidate_id_groups(candidate_set: ContactCandidateSet, *, max_group_attempts: int) -> list[list[int]]:
    groups = []
    proposals = sorted(
        candidate_set.group_proposals,
        key=lambda proposal: (_group_priority(proposal.group_type), -proposal.group_score, proposal.group_id),
    )
    for proposal in proposals[:max_group_attempts]:
        groups.append(list(proposal.candidate_ids))
    if groups:
        return groups
    fallback: list[int] = []
    for _, candidate_ids in sorted(candidate_set.slot_coverage.items()):
        fallback.extend(candidate_ids)
    return [fallback] if fallback else []


def _group_priority(group_type: str) -> int:
    if group_type == "grasp_pair":
        return 0
    if group_type == "multi_grasp":
        return 1
    if group_type == "support_set":
        return 2
    return 3


def _assignments_with_state(
    assignments: list[ContactAssignment],
    candidate_set: ContactCandidateSet,
    schedule_state: str,
    config: BaselineTrajectoryPlannerConfig,
) -> list[ContactAssignment]:
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidate_set.candidates}
    return [
        _assignment_from_candidate(candidate_by_id[assignment.candidate_id], schedule_state, config)
        for assignment in assignments
    ]


def _assignment_from_candidate(
    candidate: ContactCandidate,
    schedule_state: str,
    config: BaselineTrajectoryPlannerConfig,
) -> ContactAssignment:
    force = _wrench_for_candidate(candidate, schedule_state, config)
    lower = None if schedule_state in {"approach", "release"} else [min(0.0, value) for value in force]
    upper = None if schedule_state in {"approach", "release"} else [max(0.0, value) for value in force]
    return ContactAssignment(
        slot_id=candidate.slot_id,
        anchor_id=candidate.anchor_id,
        candidate_id=candidate.candidate_id,
        contact_mode=candidate.contact_mode,
        schedule_state=schedule_state,  # type: ignore[arg-type]
        wrench_target=force if schedule_state not in {"approach", "release"} else None,
        wrench_lower=lower,
        wrench_upper=upper,
        priority=1.0 if schedule_state in {"attach", "maintain"} else 0.5,
    )


def _wrench_for_candidate(
    candidate: ContactCandidate,
    schedule_state: str,
    config: BaselineTrajectoryPlannerConfig,
) -> list[float]:
    if schedule_state in {"approach", "release"}:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if candidate.contact_mode == ContactMode.GRASP:
        magnitude = config.grasp_force_n
        return [
            -candidate.normal_world[0] * magnitude,
            -candidate.normal_world[1] * magnitude,
            -candidate.normal_world[2] * magnitude,
            0.0,
            0.0,
            0.0,
        ]
    if candidate.contact_mode == ContactMode.SUPPORT:
        return [0.0, 0.0, config.support_force_n, 0.0, 0.0, 0.0]
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _slot_count_requirements(context: HighLevelPolicyContext) -> tuple[dict[int, int], dict[int, int]]:
    mins: dict[int, int] = {}
    maxes: dict[int, int] = {}
    for node in context.irg.nodes:
        if node.node_type != IRGNodeType.CONTACT_SLOT:
            continue
        slot_id = int(node.feature.get("slot_id", node.node_id))
        if node.feature.get("required", True):
            mins[slot_id] = int(node.feature.get("min_count_group", 1))
        maxes[slot_id] = int(node.feature.get("max_count_group", 1))
    return mins, maxes


def _object_goal_pose(context: HighLevelPolicyContext) -> Pose7D | None:
    for node in context.irg.nodes:
        if node.node_type != IRGNodeType.STATE_TARGET:
            continue
        if node.ref_id == "object_goal_pose":
            pose = node.feature.get("pose_target_world")
            if pose is None:
                return None
            if len(pose) != 7:
                raise SchemaValidationError("object_goal_pose must have length 7")
            return tuple(float(value) for value in pose)  # type: ignore[return-value]
    return None


def _object_id_for_assignments(assignments: list[ContactAssignment], candidate_set: ContactCandidateSet) -> str:
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidate_set.candidates}
    for assignment in assignments:
        candidate = candidate_by_id.get(assignment.candidate_id)
        if candidate is not None:
            return candidate.target_entity_id
    raise SchemaValidationError("Baseline trajectory requires at least one selected candidate")


def _lift_pose(context: HighLevelPolicyContext, object_id: str) -> Pose7D | None:
    if context.runtime_observation is None:
        return None
    for obj_state in context.runtime_observation.object_states:
        if obj_state.object_id == object_id:
            pose = obj_state.pose_world
            return (pose[0], pose[1], pose[2] + 0.05, pose[3], pose[4], pose[5], pose[6])
    return None


def _anchor_pose_targets(candidate_set: ContactCandidateSet, assignments: list[ContactAssignment]) -> dict[int, Pose7D]:
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidate_set.candidates}
    refs: dict[int, Pose7D] = {}
    for assignment in assignments:
        candidate = candidate_by_id.get(assignment.candidate_id)
        if candidate is not None:
            refs[assignment.anchor_id] = candidate.contact_pose_world
    return refs


def _p4_2_guards(phase: str) -> list[dict[str, str]]:
    return [
        {
            "type": "p4_2_phase",
            "phase": phase,
            "contact_model": "kinematic_payload_coupled_attach_v1",
        }
    ]


def p4_2_phase_from_knot(knot: InteractionKnot) -> str | None:
    for guard in knot.guard_conditions:
        if guard.get("type") == "p4_2_phase":
            phase = guard.get("phase")
            return str(phase) if phase is not None else None
    return None


def _selected_contact_centroid(
    candidate_set: ContactCandidateSet,
    assignments: list[ContactAssignment],
) -> tuple[float, float, float]:
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidate_set.candidates}
    poses = [
        candidate_by_id[assignment.candidate_id].contact_pose_world
        for assignment in assignments
        if assignment.candidate_id in candidate_by_id
    ]
    if not poses:
        return (0.0, 0.0, 0.0)
    inv = 1.0 / float(len(poses))
    return (
        sum(pose[0] for pose in poses) * inv,
        sum(pose[1] for pose in poses) * inv,
        sum(pose[2] for pose in poses) * inv,
    )


def _p4_2_body_target_pose(
    context: HighLevelPolicyContext,
    *,
    contact_centroid: tuple[float, float, float],
    object_goal_pose: Pose7D | None,
    phase: str,
    config: P4_2DeterministicPlannerConfig,
) -> Pose7D:
    current_body_pose = _runtime_body_pose(context)
    orientation = current_body_pose[3:7] if current_body_pose is not None else (0.0, 0.0, 0.0, 1.0)
    if phase in {"transport", "release"} and object_goal_pose is not None:
        offset = _runtime_body_object_offset(context)
        return (
            object_goal_pose[0] + offset[0],
            object_goal_pose[1] + offset[1],
            max(object_goal_pose[2] + offset[2], config.min_body_target_height_m),
            orientation[0],
            orientation[1],
            orientation[2],
            orientation[3],
        )
    return (
        contact_centroid[0],
        contact_centroid[1],
        max(contact_centroid[2] + config.approach_height_offset_m, config.min_body_target_height_m),
        orientation[0],
        orientation[1],
        orientation[2],
        orientation[3],
    )


def _runtime_body_pose(context: HighLevelPolicyContext) -> Pose7D | None:
    if context.runtime_observation is None or not context.runtime_observation.module_states:
        return None
    return context.runtime_observation.module_states[0].pose_world


def _runtime_body_object_offset(context: HighLevelPolicyContext) -> tuple[float, float, float]:
    body_pose = _runtime_body_pose(context)
    if body_pose is None or context.runtime_observation is None or not context.runtime_observation.object_states:
        return (0.0, 0.0, 0.0)
    object_pose = context.runtime_observation.object_states[0].pose_world
    return (
        body_pose[0] - object_pose[0],
        body_pose[1] - object_pose[1],
        body_pose[2] - object_pose[2],
    )
