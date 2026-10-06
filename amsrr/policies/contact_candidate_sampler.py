from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from amsrr.geometry.contact_material import (
    ContactFrictionResolution,
    resolve_contact_friction,
)
from amsrr.geometry.surface_patch_graph import dot, normalize, orthonormal_basis
from amsrr.policies.assignment_feasibility import DEFAULT_OPPOSING_NORMAL_DOT_THRESHOLD
from amsrr.policies.contact_candidate_set import _candidates_conflict, build_contact_candidate_set
from amsrr.schemas.common import ContactMode, Pose7D, SchemaValidationError, Vector3
from amsrr.schemas.contact_candidates import ContactCandidate, ContactCandidateGroupProposal, ContactCandidateSet
from amsrr.schemas.geometry import ContactRegion, GeometryDescriptor, SurfacePatchToken
from amsrr.schemas.interaction_envelope import InteractionEnvelope
from amsrr.schemas.irg import IRGNode, IRGNodeType, InteractionRequirementGraph
from amsrr.schemas.morphology import MorphologyGraph, RobotAnchor
from amsrr.schemas.runtime import RuntimeObservation
from amsrr.schemas.task_spec import TaskSpec


CONTACT_CANDIDATE_SAMPLER_VERSION = "contact_sampler_v4_directional_diverse_grasp"
CONTACT_MODE_TO_ANCHOR_TYPE = {
    ContactMode.GRASP: "grasp",
    ContactMode.SUPPORT: "support",
    ContactMode.PUSH: "push",
    ContactMode.LATCH: "latch",
    ContactMode.PERCH: "perch",
    ContactMode.TOOL: "tool",
    ContactMode.BODY_CONTACT: "body_contact",
}


@dataclass(frozen=True)
class ContactCandidateSamplerConfig:
    samples_per_slot_region_anchor: int = 1
    max_group_proposals_per_slot: int = 8
    min_friction_for_grasp: float = 0.05
    max_grasp_contacts: int = 3
    grasp_neighborhoods: bool = False


@dataclass(frozen=True)
class _SlotSpec:
    slot_id: int
    target_entity_id: str
    allowed_region_ids: list[str]
    contact_mode: ContactMode
    required: bool
    min_count: int
    max_count: int
    required_anchor_capability: dict[str, Any]


@dataclass(frozen=True)
class _RegionRecord:
    region: ContactRegion
    patch: SurfacePatchToken | None


class ContactCandidateSampler:
    """Morphology-conditioned deterministic contact proposal sampler."""

    def __init__(self, config: ContactCandidateSamplerConfig | None = None) -> None:
        self.config = config or ContactCandidateSamplerConfig()

    def sample(
        self,
        *,
        task_spec: TaskSpec,
        irg: InteractionRequirementGraph,
        interaction_envelope: InteractionEnvelope,
        morphology_graph: MorphologyGraph,
        geometry_descriptors: dict[str, GeometryDescriptor],
        runtime_observation: RuntimeObservation | None = None,
        anchor_poses_world: dict[int, Pose7D] | None = None,
    ) -> ContactCandidateSet:
        if self.config.samples_per_slot_region_anchor <= 0:
            raise SchemaValidationError("ContactCandidateSamplerConfig.samples_per_slot_region_anchor must be positive")
        slots = _contact_slots(irg)
        regions = _region_records(geometry_descriptors)
        candidates: list[ContactCandidate] = []
        candidate_id = 0
        for slot in slots:
            for region_id in slot.allowed_region_ids:
                region_record = regions.get(region_id)
                if region_record is None:
                    continue
                for anchor in _compatible_anchors(morphology_graph.robot_anchors, slot):
                    mode = slot.contact_mode
                    if mode not in region_record.region.allowed_contact_modes:
                        continue
                    for sample_index in range(self.config.samples_per_slot_region_anchor):
                        candidate = self._candidate(
                            candidate_id,
                            slot,
                            region_record,
                            anchor,
                            task_spec=task_spec,
                            sample_index=sample_index,
                            runtime_observation=runtime_observation,
                        )
                        candidates.append(candidate)
                        candidate_id += 1
        group_proposals = build_group_proposals(
            candidates, max_per_slot=self.config.max_group_proposals_per_slot,
            max_grasp_contacts=self.config.max_grasp_contacts,
            slot_cardinalities={s.slot_id: (s.min_count, s.max_count) for s in slots},
            anchor_poses_world=anchor_poses_world,
            grasp_neighborhoods=self.config.grasp_neighborhoods,
        )
        return build_contact_candidate_set(
            set_id=f"candidates:{task_spec.task_id}:{morphology_graph.graph_id}:{interaction_envelope.envelope_id}",
            task_id=task_spec.task_id,
            morphology_graph_id=morphology_graph.graph_id,
            candidates=candidates,
            group_proposals=group_proposals,
            sampler_version=CONTACT_CANDIDATE_SAMPLER_VERSION,
        )

    def _candidate(
        self,
        candidate_id: int,
        slot: _SlotSpec,
        region_record: _RegionRecord,
        anchor: RobotAnchor,
        *,
        task_spec: TaskSpec,
        sample_index: int,
        runtime_observation: RuntimeObservation | None,
    ) -> ContactCandidate:
        region = region_record.region
        patch = region_record.patch
        pose_world = _contact_pose_world(task_spec, slot.target_entity_id, region, patch, sample_index)
        normal_world = _normal_world(task_spec, slot.target_entity_id, region.normal_summary_object)
        tangent_u, tangent_v = orthonormal_basis(normal_world)
        friction = resolve_contact_friction(
            task_spec.metadata,
            target_entity_id=slot.target_entity_id,
            contact_mode=slot.contact_mode,
            target_surface_friction=region.friction,
        )
        violations = _unary_violations(slot, region, anchor, normal_world, runtime_observation)
        scores = _candidate_scores(
            slot,
            region,
            anchor,
            normal_world,
            violations,
            friction=friction,
        )
        return ContactCandidate(
            candidate_id=candidate_id,
            slot_id=slot.slot_id,
            anchor_id=anchor.anchor_id,
            target_entity_id=slot.target_entity_id,
            region_id=region.region_id,
            contact_pose_world=pose_world,
            contact_frame_world=pose_world,
            normal_world=normal_world,
            tangent_basis_world=[*tangent_u, *tangent_v],
            contact_mode=slot.contact_mode,
            friction=friction.effective_friction,
            patch_area_m2=region.area_m2,
            candidate_scores=scores,
            unary_valid=not violations,
            unary_violation_codes=violations,
        )


def build_group_proposals(
    candidates: list[ContactCandidate],
    *,
    max_per_slot: int = 8,
    max_grasp_contacts: int = 3,
    slot_cardinalities: dict[int, tuple[int, int]] | None = None,
    anchor_poses_world: dict[int, Pose7D] | None = None,
    grasp_neighborhoods: bool = False,
) -> list[ContactCandidateGroupProposal]:
    if max_per_slot < 1 or max_grasp_contacts < 2:
        raise SchemaValidationError("invalid contact group search budget")
    proposals: list[ContactCandidateGroupProposal] = []
    valid_candidates = [candidate for candidate in candidates if candidate.unary_valid]
    by_slot: dict[int, list[ContactCandidate]] = {}
    for candidate in valid_candidates:
        by_slot.setdefault(candidate.slot_id, []).append(candidate)
    for slot_id, slot_candidates in sorted(by_slot.items()):
        minimum, maximum = (slot_cardinalities or {}).get(slot_id, (2, max_grasp_contacts))
        if grasp_neighborhoods and any(c.contact_mode == ContactMode.GRASP for c in slot_candidates):
            if max_grasp_contacts != 2 or minimum > 2 or maximum < 2:
                raise SchemaValidationError('anchor neighborhoods require two-contact groups')
            # Rank within each physical pair; a global cap would remove entire
            # pairs from the two five-anchor neighborhoods before policy input.
            grasp = [c for c in slot_candidates if c.contact_mode == ContactMode.GRASP]
            owners = sorted({c.anchor_id for c in grasp})
            for i, left in enumerate(owners):
                for right in owners[i+1:]:
                    members = [c for c in grasp if c.anchor_id in (left, right)]
                    masks = {c.anchor_id: int(c.candidate_scores.get('grasp_neighborhood_id', 0)) for c in members}
                    if not ((masks[left] & 1 and masks[right] & 2) or
                            (masks[left] & 2 and masks[right] & 1)):
                        continue
                    groups = _bounded_grasp_proposals(slot_id, members, minimum=2, maximum=2,
                        budget=max_per_slot, anchor_poses_world=anchor_poses_world)
                    for group in groups:
                        group.group_id = f'slot_{slot_id}:grasp_pair:{left}:{right}:{group.group_id.rsplit(":", 1)[-1]}'
                    proposals.extend(groups)
            continue
        groups = _bounded_grasp_proposals(slot_id, slot_candidates,
            minimum=max(2, minimum), maximum=min(maximum, max_grasp_contacts),
            budget=max_per_slot, anchor_poses_world=anchor_poses_world)
        if groups:
            proposals.extend(groups)
            continue
        support_set = _support_set_proposal(slot_id, slot_candidates)
        if support_set is not None:
            proposals.append(support_set)
    return proposals


def _bounded_grasp_proposals(slot_id, candidates, *, minimum, maximum, budget, anchor_poses_world):
    """Beam extension, with cardinality quotas; never enumerate N-way subsets.

    This is a geometric proposal, not a force-closure or IK certificate. The
    selected group's wrench allocation and trajectory are checked downstream.
    """
    grasp = sorted((c for c in candidates if c.contact_mode == ContactMode.GRASP),
                   key=lambda c: c.candidate_id)
    maximum = min(maximum, len({c.anchor_id for c in grasp}))
    if maximum < minimum:
        return []
    by_id = {c.candidate_id: c for c in grasp}
    beam_width = 4 * budget
    pairs = _grasp_pair_proposals(slot_id, grasp, max_per_slot=beam_width,
                                  anchor_poses_world=anchor_poses_world)
    by_count = {2: pairs}
    beam = [tuple(g.candidate_ids) for g in pairs]
    for count in range(3, maximum + 1):
        scored = {}
        for ids in beam:
            anchors = {by_id[i].anchor_id for i in ids}
            entity = by_id[ids[0]].target_entity_id
            for candidate in grasp:
                if (candidate.anchor_id in anchors or candidate.target_entity_id != entity
                        or any(_candidates_conflict(candidate, by_id[i]) for i in ids)):
                    continue
                extended = tuple(sorted((*ids, candidate.candidate_id)))
                if extended in scored:
                    continue
                members = [by_id[i] for i in extended]
                points = np.asarray([c.contact_pose_world[:3] for c in members])
                distances = np.linalg.norm(points[:, None] - points[None], axis=-1)
                if np.min(distances[np.triu_indices(count, 1)]) < 1e-4:
                    continue
                singular = np.linalg.svd(points - points.mean(0), compute_uv=False)
                # A third point on the same line adds no lever arm about it.
                if singular[1] < max(1e-4, .01 * singular[0]):
                    continue
                normals = np.asarray([c.normal_world for c in members])
                opposition = float(np.max(-(normals @ normals.T)))
                if opposition < -DEFAULT_OPPOSING_NORMAL_DOT_THRESHOLD:
                    continue
                owners = {c.candidate_scores.get('anchor_module_id', c.anchor_id) for c in members}
                score = (.5 * min(1., opposition) + .4 * float(singular[1] / singular[0])
                         + .1 * len(owners) / count)
                score *= _relative_reach_score(members, anchor_poses_world)
                scored[extended] = score
        ranked = sorted(scored, key=lambda ids: (
            -len({by_id[i].candidate_scores.get('anchor_module_id', by_id[i].anchor_id) for i in ids}),
            -scored[ids], ids))
        # Keep alternative physical anchor sets in the finite beam, rather
        # than spending every slot on face permutations of the same fingers.
        seen, diverse, remainder = set(), [], []
        for ids in ranked:
            signature=tuple(sorted(by_id[i].anchor_id for i in ids))
            (remainder if signature in seen else diverse).append(ids)
            seen.add(signature)
        ranked=(diverse+remainder)[:beam_width]
        by_count[count] = [ContactCandidateGroupProposal(
            group_id=f"slot_{slot_id}:multi_grasp:{count}:{index}",
            candidate_ids=list(ids), group_type="multi_grasp",
            group_score=scored[ids], group_violation_codes=[],
        ) for index, ids in enumerate(ranked)]
        beam = ranked
    # Interleave rather than allowing pair scores to consume the entire budget.
    result = []
    for index in range(budget):
        for count in range(minimum, maximum + 1):
            groups = by_count.get(count, [])
            if index < len(groups):
                result.append(groups[index])
                if len(result) == budget:
                    return result
    return result


def _relative_reach_score(members, anchor_poses_world):
    """Soft cost of deformation from observed geometry, not an IK certificate.

    Relative vectors preserve the direction of each anchor-to-target binding.
    Distances and normal Gram matrices alone cannot distinguish swapped
    bindings. Common translation/rotation of the world frame cancels here;
    a rotation of the robot relative to the object is a real motion cost.
    """
    if anchor_poses_world is None:
        return 1.
    actual = np.asarray([anchor_poses_world[c.anchor_id][:3] for c in members])
    target = np.asarray([c.contact_pose_world[:3] for c in members])
    pairs = np.triu_indices(len(members), 1)
    a = (actual[:, None] - actual[None])[pairs]
    b = (target[:, None] - target[None])[pairs]
    error = float(np.mean(np.linalg.norm(a - b, axis=-1)))
    observed_normals = np.asarray([_quat_rotate(anchor_poses_world[c.anchor_id][3:], (1.,0.,0.)) for c in members])
    target_normals = -np.asarray([c.normal_world for c in members])
    angular_error = float(np.mean(1. - np.clip(np.sum(observed_normals * target_normals, axis=-1), -1., 1.)))
    return .1 + .9 * math.exp(-error / max(.05, float(np.mean(np.linalg.norm(b, axis=-1)))) - angular_error)


def _grasp_pair_proposals(
    slot_id: int,
    candidates: list[ContactCandidate],
    *,
    max_per_slot: int,
    anchor_poses_world=None,
) -> list[ContactCandidateGroupProposal]:
    grasp_candidates = [candidate for candidate in candidates if candidate.contact_mode == ContactMode.GRASP]
    scored_pairs: list[tuple[float, tuple[int, int]]] = []
    for left_idx, left in enumerate(grasp_candidates):
        for right in grasp_candidates[left_idx + 1 :]:
            if _candidates_conflict(left, right) or left.target_entity_id != right.target_entity_id:
                continue
            normal_dot = dot(left.normal_world, right.normal_world)
            # Do not spend the finite proposal budget on pairs the default
            # assignment guard already rejects. Any admissible larger group
            # contains an opposing pair and can still extend from these seeds.
            if normal_dot > DEFAULT_OPPOSING_NORMAL_DOT_THRESHOLD:
                continue
            normal_opposition = max(0.0, -normal_dot)
            same_region_penalty = 0.25 if left.region_id == right.region_id else 0.0
            score = min(1.0, 0.5 + 0.5 * normal_opposition - same_region_penalty)
            if left.candidate_scores.get('anchor_module_id', left.anchor_id) == right.candidate_scores.get('anchor_module_id', right.anchor_id):
                score -= .15
            score *= _relative_reach_score([left, right], anchor_poses_world)
            if score <= 0.0:
                continue
            scored_pairs.append((score, tuple(sorted((left.candidate_id, right.candidate_id)))))
    owners = {c.candidate_id: c.candidate_scores.get('anchor_module_id', c.anchor_id)
              for c in grasp_candidates}
    scored_pairs.sort(key=lambda item: (-len({owners[i] for i in item[1]}), -item[0], item[1]))
    by_id = {c.candidate_id: c for c in grasp_candidates}
    seen, diverse, remainder = set(), [], []
    for item in scored_pairs:
        members = [by_id[i] for i in item[1]]
        # Cover distinct physical fingers AND object region pairs before
        # spending more slots on reversed assignments or repeated samples.
        surfaces = tuple(sorted(
            ('surface', c.candidate_scores['surface_port_id'])
            if c.candidate_scores.get('surface_port_id', -1) >= 0 else ('anchor', c.anchor_id)
            for c in members))
        signature = (members[0].target_entity_id, surfaces, tuple(sorted(c.region_id for c in members)))
        (remainder if signature in seen else diverse).append(item)
        seen.add(signature)
    scored_pairs = diverse + remainder
    proposals: list[ContactCandidateGroupProposal] = []
    for idx, (score, candidate_ids) in enumerate(scored_pairs[:max_per_slot]):
        proposals.append(
            ContactCandidateGroupProposal(
                group_id=f"slot_{slot_id}:grasp_pair:{idx}",
                candidate_ids=list(candidate_ids),
                group_type="grasp_pair",
                group_score=score,
                group_violation_codes=[],
            )
        )
    return proposals


def _support_set_proposal(slot_id: int, candidates: list[ContactCandidate]) -> ContactCandidateGroupProposal | None:
    support_ids = [candidate.candidate_id for candidate in candidates if candidate.contact_mode == ContactMode.SUPPORT]
    if not support_ids:
        return None
    return ContactCandidateGroupProposal(
        group_id=f"slot_{slot_id}:support_set:0",
        candidate_ids=sorted(support_ids),
        group_type="support_set",
        group_score=1.0 / max(1, len(support_ids)),
        group_violation_codes=[],
    )


def _contact_slots(irg: InteractionRequirementGraph) -> list[_SlotSpec]:
    slots: list[_SlotSpec] = []
    for node in sorted(irg.nodes, key=lambda item: item.node_id):
        if node.node_type != IRGNodeType.CONTACT_SLOT:
            continue
        slots.append(_slot_from_node(node))
    return slots


def _slot_from_node(node: IRGNode) -> _SlotSpec:
    try:
        mode = ContactMode(node.feature.get("contact_mode"))
    except ValueError as exc:
        raise SchemaValidationError(f"ContactSlot {node.node_id} has unsupported contact_mode") from exc
    return _SlotSpec(
        slot_id=int(node.feature.get("slot_id", node.node_id)),
        target_entity_id=str(node.feature.get("target_entity_id", "")),
        allowed_region_ids=list(node.feature.get("allowed_region_ids", []) or []),
        contact_mode=mode,
        required=bool(node.feature.get("required", True)),
        min_count=int(node.feature.get("min_count_group", 1)),
        max_count=int(node.feature.get("max_count_group", 1)),
        required_anchor_capability=dict(node.feature.get("required_anchor_capability", {}) or {}),
    )


def _region_records(geometry_descriptors: dict[str, GeometryDescriptor]) -> dict[str, _RegionRecord]:
    records: dict[str, _RegionRecord] = {}
    for descriptor in geometry_descriptors.values():
        patches = {patch.patch_id: patch for patch in descriptor.surface_patch_graph.nodes}
        for region in descriptor.contact_region_graph.nodes:
            patch = patches.get(region.patch_ids[0]) if region.patch_ids else None
            records[region.region_id] = _RegionRecord(region=region, patch=patch)
    return records


def _compatible_anchors(anchors: list[RobotAnchor], slot: _SlotSpec) -> list[RobotAnchor]:
    required_anchor_type = CONTACT_MODE_TO_ANCHOR_TYPE.get(slot.contact_mode)
    if required_anchor_type is None:
        return []
    compatible: list[RobotAnchor] = []
    for anchor in sorted(anchors, key=lambda item: item.anchor_id):
        if anchor.anchor_type != required_anchor_type:
            continue
        if slot.slot_id not in anchor.associated_contact_slot_ids:
            continue
        capability_type = slot.required_anchor_capability.get("capability_type")
        if capability_type is not None and capability_type != anchor.anchor_type:
            continue
        compatible.append(anchor)
    return compatible


def _contact_pose_world(
    task_spec: TaskSpec,
    entity_id: str,
    region: ContactRegion,
    patch: SurfacePatchToken | None,
    sample_index: int,
) -> Pose7D:
    entity_pose = _entity_pose_world(task_spec, entity_id)
    local_position = patch.position_object if patch is not None else (0.0, 0.0, 0.0)
    if sample_index > 0:
        tangent_u, _ = orthonormal_basis(region.normal_summary_object)
        offset_scale = min(0.02, math.sqrt(max(region.area_m2, 0.0)) * 0.1)
        local_position = (
            local_position[0] + tangent_u[0] * offset_scale * float(sample_index),
            local_position[1] + tangent_u[1] * offset_scale * float(sample_index),
            local_position[2] + tangent_u[2] * offset_scale * float(sample_index),
        )
    world_position = _transform_point(entity_pose, local_position)
    return (*world_position, entity_pose[3], entity_pose[4], entity_pose[5], entity_pose[6])


def _normal_world(task_spec: TaskSpec, entity_id: str, normal_object: Vector3) -> Vector3:
    entity_pose = _entity_pose_world(task_spec, entity_id)
    quat = (entity_pose[3], entity_pose[4], entity_pose[5], entity_pose[6])
    return normalize(_quat_rotate(quat, normalize(normal_object)))


def _entity_pose_world(task_spec: TaskSpec, entity_id: str) -> Pose7D:
    for obj in task_spec.scene.objects:
        if obj.object_id == entity_id:
            return obj.pose_world
    for surface in task_spec.scene.environment.support_surfaces:
        if surface.surface_id == entity_id:
            return surface.pose_world
    for obstacle in task_spec.scene.environment.obstacles:
        if obstacle.obstacle_id == entity_id:
            return obstacle.pose_world
    return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def _transform_point(pose: Pose7D, local: Vector3) -> Vector3:
    quat = (pose[3], pose[4], pose[5], pose[6])
    rotated = _quat_rotate(quat, local)
    return (pose[0] + rotated[0], pose[1] + rotated[1], pose[2] + rotated[2])


def _quat_rotate(quat: tuple[float, float, float, float], vector: Vector3) -> Vector3:
    qx, qy, qz, qw = quat
    x, y, z = vector
    tx = 2.0 * (qy * z - qz * y)
    ty = 2.0 * (qz * x - qx * z)
    tz = 2.0 * (qx * y - qy * x)
    return (
        x + qw * tx + (qy * tz - qz * ty),
        y + qw * ty + (qz * tx - qx * tz),
        z + qw * tz + (qx * ty - qy * tx),
    )


def _unary_violations(
    slot: _SlotSpec,
    region: ContactRegion,
    anchor: RobotAnchor,
    normal_world: Vector3,
    runtime_observation: RuntimeObservation | None,
) -> list[str]:
    violations: list[str] = []
    if slot.contact_mode not in region.allowed_contact_modes:
        violations.append("E_CONTACT_CANDIDATE_UNARY_INVALID")
    if not any(abs(value) > 1.0e-9 for value in normal_world):
        violations.append("E_CONTACT_CANDIDATE_UNARY_INVALID")
    min_force = _optional_float(slot.required_anchor_capability.get("min_force_n"))
    max_force = _optional_float(anchor.capability.get("max_force_n"))
    if min_force is not None and max_force is not None and max_force < min_force:
        violations.append("E_ANCHOR_CAPABILITY_INSUFFICIENT")
    if slot.contact_mode == ContactMode.GRASP and region.friction is not None and region.friction < 0.0:
        violations.append("E_CONTACT_CANDIDATE_UNARY_INVALID")
    if runtime_observation is not None:
        module_ids = {state.module_id for state in runtime_observation.module_states}
        if anchor.module_id not in module_ids:
            violations.append("E_COARSE_REACHABILITY_FAIL")
    return sorted(set(violations))


def _candidate_scores(
    slot: _SlotSpec,
    region: ContactRegion,
    anchor: RobotAnchor,
    normal_world: Vector3,
    violations: list[str],
    *,
    friction: ContactFrictionResolution,
) -> dict[str, float]:
    effective_friction = (
        friction.effective_friction
        if friction.effective_friction is not None
        else 0.5
    )
    min_force = _optional_float(slot.required_anchor_capability.get("min_force_n")) or 0.0
    max_force = _optional_float(anchor.capability.get("max_force_n")) or max(min_force, 1.0)
    return {
        "anchor_module_id": float(anchor.module_id),
        **({'grasp_neighborhood_id': float(anchor.capability['grasp_neighborhood_mask'])}
           if 'grasp_neighborhood_mask' in anchor.capability else {}),
        "surface_port_id": float(anchor.capability.get('surface_port_id', -1)),
        "mode_match": 1.0 if not violations else 0.0,
        "normal_alignment": 1.0 if any(abs(value) > 1.0e-9 for value in normal_world) else 0.0,
        "local_reachability": 1.0 if slot.slot_id in anchor.associated_contact_slot_ids else 0.0,
        "surface_quality": min(1.0, max(region.area_m2, 0.0) / 0.01),
        "moment_arm_quality": _moment_arm_quality(region.normal_summary_object),
        "support_quality": 1.0 if slot.contact_mode == ContactMode.SUPPORT and normal_world[2] > 0.0 else 0.5,
        "friction_plausibility": min(1.0, max(effective_friction, 0.0)),
        "anchor_capability": 1.0 if min_force <= 0.0 else min(1.0, max_force / max(min_force, 1.0e-9)),
        "material_contract_applied": 1.0 if friction.task_material_applied else 0.0,
        "material_target_surface_friction": (
            0.0
            if friction.target_surface_friction is None
            else friction.target_surface_friction
        ),
        "material_robot_surface_friction": (
            0.0
            if friction.robot_surface_friction is None
            else friction.robot_surface_friction
        ),
        "material_effective_friction": (
            0.0
            if friction.effective_friction is None
            else friction.effective_friction
        ),
        "material_friction_combine_mode_code": friction.combine_mode_code,
    }


def _moment_arm_quality(normal: Vector3) -> float:
    lateral = math.sqrt(normal[0] * normal[0] + normal[1] * normal[1])
    return min(1.0, max(0.0, lateral))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
