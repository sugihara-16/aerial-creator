from __future__ import annotations

from copy import deepcopy

import pytest

from amsrr.geometry.contact_material import with_selected_robot_contact_material
from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.policies.contact_candidate_sampler import CONTACT_CANDIDATE_SAMPLER_VERSION, ContactCandidateSampler
from amsrr.policies.design_policy_base import DesignPolicyContext, FixedSimpleDesignPolicy
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import ContactMode
from amsrr.schemas.irg import IRGNodeType
from amsrr.schemas.task_spec import TaskSpec


def _pipeline(grasp_carry_dict: dict):
    task = TaskSpec.from_dict(grasp_carry_dict)
    builder_result = IRGBuilder().build_with_scene_graph(task)
    irg = builder_result.irg
    envelope = InteractionEnvelopeExtractor().extract(irg)
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    design = FixedSimpleDesignPolicy().design(
        DesignPolicyContext(
            task_spec=task,
            irg=irg,
            interaction_envelope=envelope,
            physical_model=physical_model,
        )
    )
    return task, irg, envelope, builder_result.scene_graph.geometry_descriptors, design


def test_contact_candidate_sampler_returns_non_empty_grasp_carry_candidates(grasp_carry_dict: dict) -> None:
    task, irg, envelope, descriptors, design = _pipeline(grasp_carry_dict)

    candidate_set = ContactCandidateSampler().sample(
        task_spec=task,
        irg=irg,
        interaction_envelope=envelope,
        morphology_graph=design.target_morphology,
        geometry_descriptors=descriptors,
    )

    required_slot_ids = {
        int(node.feature["slot_id"])
        for node in irg.nodes
        if node.node_type == IRGNodeType.CONTACT_SLOT and node.feature["required"]
    }
    assert candidate_set.sampler_version == CONTACT_CANDIDATE_SAMPLER_VERSION
    assert candidate_set.candidates
    assert candidate_set.morphology_graph_id == design.target_morphology.graph_id
    assert required_slot_ids <= set(candidate_set.slot_coverage)
    assert all(candidate.unary_valid for candidate in candidate_set.candidates)
    assert all(candidate.target_entity_id == "box_01" for candidate in candidate_set.candidates)
    assert all(candidate.region_id.startswith("box_01_face_") for candidate in candidate_set.candidates)
    assert {candidate.contact_mode for candidate in candidate_set.candidates} >= {ContactMode.GRASP, ContactMode.SUPPORT}
    assert any(candidate.contact_pose_world[0] != 0.0 for candidate in candidate_set.candidates)


def test_contact_candidate_sampler_builds_grasp_pair_group_proposals(grasp_carry_dict: dict) -> None:
    task, irg, envelope, descriptors, design = _pipeline(grasp_carry_dict)

    candidate_set = ContactCandidateSampler().sample(
        task_spec=task,
        irg=irg,
        interaction_envelope=envelope,
        morphology_graph=design.target_morphology,
        geometry_descriptors=descriptors,
    )

    grasp_pairs = [proposal for proposal in candidate_set.group_proposals if proposal.group_type == "grasp_pair"]
    assert grasp_pairs
    assert all(len(proposal.candidate_ids) == 2 for proposal in grasp_pairs)
    by_id = {candidate.candidate_id: candidate for candidate in candidate_set.candidates}
    for proposal in grasp_pairs:
        left, right = [by_id[candidate_id] for candidate_id in proposal.candidate_ids]
        assert left.slot_id == right.slot_id
        assert left.anchor_id != right.anchor_id
        assert left.contact_mode == ContactMode.GRASP
        assert right.contact_mode == ContactMode.GRASP
        assert 0.0 < proposal.group_score <= 1.0


def test_contact_candidate_sampler_uses_robot_anchor_associations(grasp_carry_dict: dict) -> None:
    task, irg, envelope, descriptors, design = _pipeline(grasp_carry_dict)

    candidate_set = ContactCandidateSampler().sample(
        task_spec=task,
        irg=irg,
        interaction_envelope=envelope,
        morphology_graph=design.target_morphology,
        geometry_descriptors=descriptors,
    )

    anchor_slot_pairs = {
        (anchor.anchor_id, slot_id)
        for anchor in design.target_morphology.robot_anchors
        for slot_id in anchor.associated_contact_slot_ids
    }
    assert {
        (candidate.anchor_id, candidate.slot_id)
        for candidate in candidate_set.candidates
    } <= anchor_slot_pairs
    assert type(candidate_set).from_json(candidate_set.to_json()).to_dict() == candidate_set.to_dict()


def test_contact_candidate_sampler_archives_effective_selected_surface_friction(
    grasp_carry_dict: dict,
) -> None:
    data = deepcopy(grasp_carry_dict)
    target_id = str(data["scene"]["objects"][0]["object_id"])
    data["metadata"] = with_selected_robot_contact_material(
        data.get("metadata", {}),
        target_entity_ids=[target_id],
        contact_modes=[ContactMode.GRASP],
        robot_static_friction=4.5,
        friction_combine_mode="max",
    )
    task, irg, envelope, descriptors, design = _pipeline(data)

    candidate_set = ContactCandidateSampler().sample(
        task_spec=task,
        irg=irg,
        interaction_envelope=envelope,
        morphology_graph=design.target_morphology,
        geometry_descriptors=descriptors,
    )

    grasp = [
        item for item in candidate_set.candidates
        if item.contact_mode == ContactMode.GRASP
    ]
    support = [
        item for item in candidate_set.candidates
        if item.contact_mode == ContactMode.SUPPORT
    ]
    assert grasp
    assert support
    assert all(item.friction == pytest.approx(4.5) for item in grasp)
    assert all(item.candidate_scores["material_contract_applied"] == 1.0 for item in grasp)
    assert all(item.candidate_scores["material_effective_friction"] == pytest.approx(4.5) for item in grasp)
    assert all(item.candidate_scores["material_contract_applied"] == 0.0 for item in support)


def test_reach_order_distinguishes_swapped_bindings_and_is_world_frame_invariant():
    import numpy as np
    from scipy.spatial.transform import Rotation
    from amsrr.policies.contact_candidate_sampler import _relative_reach_score
    from tests.unit.policies.test_contact_candidate_interfaces import _candidate

    left = _candidate(0, slot_id=0, anchor_id=10, normal_world=(-1., 0., 0.))
    right = _candidate(1, slot_id=0, anchor_id=20, normal_world=(1., 0., 0.))
    left.contact_pose_world = (-.2, 0., 0., 0., 0., 0., 1.)
    right.contact_pose_world = (.2, 0., 0., 0., 0., 0., 1.)
    # The robot already matches the target relative positions and inward normals.
    poses = {10: (-.2, 0., 0., 0., 0., 0., 1.), 20: (.2, 0., 0., 0., 0., 1., 0.)}
    matching = [left, right]
    swapped = deepcopy(matching)
    swapped[0].anchor_id, swapped[1].anchor_id = 20, 10
    baseline = _relative_reach_score(matching, poses)
    assert baseline == pytest.approx(1.)
    assert _relative_reach_score(swapped, poses) < baseline
    # Transform both scene and robot, including roll/pitch and a large translation.
    rotation = Rotation.from_euler('xyz', [.4, -.8, 1.1])
    offset = np.array([8., -5., 3.])
    def transform_pose(p):
        return tuple(rotation.apply(p[:3]) + offset) + tuple((rotation * Rotation.from_quat(p[3:])).as_quat())
    moved = deepcopy(matching)
    for c in moved:
        c.contact_pose_world = transform_pose(c.contact_pose_world)
        c.normal_world = tuple(rotation.apply(c.normal_world))
    assert _relative_reach_score(moved, {k: transform_pose(v) for k, v in poses.items()}) == pytest.approx(baseline)


def test_pair_budget_covers_other_surfaces_before_repeated_samples():
    from amsrr.policies.contact_candidate_sampler import build_group_proposals
    from tests.unit.policies.test_contact_candidate_interfaces import _candidate

    candidates = []
    # Many samples of one finger pair must not starve a second available pair.
    for i in range(5):
        left = _candidate(i, slot_id=0, anchor_id=0, normal_world=(-1., 0., 0.))
        left.region_id = 'left'
        candidates.append(left)
    right = _candidate(5, slot_id=0, anchor_id=1, normal_world=(1., 0., 0.))
    right.region_id = 'right'
    other = deepcopy(right)
    other.candidate_id, other.anchor_id = 6, 2
    candidates.extend([right, other])
    groups = build_group_proposals(candidates, max_per_slot=2, max_grasp_contacts=2)
    by_id = {c.candidate_id: c for c in candidates}
    assert {frozenset(by_id[i].anchor_id for i in g.candidate_ids) for g in groups} == {
        frozenset((0, 1)), frozenset((0, 2))}
    assert [g.to_dict() for g in groups] == [g.to_dict() for g in build_group_proposals(
        list(reversed(candidates)), max_per_slot=2, max_grasp_contacts=2)]


def test_pair_budget_excludes_nonopposing_normals_before_truncation():
    from amsrr.policies.contact_candidate_sampler import build_group_proposals
    from tests.unit.policies.test_contact_candidate_interfaces import _candidate

    candidates = [
        _candidate(0, slot_id=0, anchor_id=0, normal_world=(1., 0., 0.)),
        _candidate(1, slot_id=0, anchor_id=1, normal_world=(0., 1., 0.)),
        _candidate(2, slot_id=0, anchor_id=2, normal_world=(-1., 0., 0.)),
    ]
    # The valid pair is on one module; old owner-first ranking put the invalid
    # cross-module pairs ahead of it and consumed the only proposal slot.
    for c, owner in zip(candidates, [0, 1, 0]):
        c.candidate_scores['anchor_module_id'] = owner
    groups = build_group_proposals(candidates, max_per_slot=1, max_grasp_contacts=2)
    assert len(groups) == 1
    assert set(groups[0].candidate_ids) == {0, 2}


@pytest.mark.parametrize('maximum', [2, 3, 4])
def test_proposals_never_reuse_a_physical_surface_with_different_anchor_ids(maximum):
    from amsrr.policies.contact_candidate_sampler import build_group_proposals
    from tests.unit.policies.test_contact_candidate_interfaces import _candidate

    candidates = []
    for i, (position, normal, surface) in enumerate([
        ((-.2, 0., 0.), (-1., 0., 0.), 10),
        ((.2, 0., 0.), (1., 0., 0.), 20),
        ((0., .2, 0.), (0., 1., 0.), 10),  # alias of the first physical surface
        ((0., -.2, 0.), (0., -1., 0.), 30),
        ((0., 0., .2), (0., 0., 1.), 40),
    ]):
        c = _candidate(i, slot_id=0, anchor_id=i, normal_world=normal)
        c.contact_pose_world = (*position, 0., 0., 0., 1.)
        c.candidate_scores['surface_port_id'] = surface
        candidates.append(c)
    groups = build_group_proposals(candidates, max_per_slot=32, max_grasp_contacts=maximum)
    assert {len(g.candidate_ids) for g in groups} == set(range(2, maximum + 1))
    for group in groups:
        surfaces = [candidates[i].candidate_scores['surface_port_id'] for i in group.candidate_ids]
        assert len(set(surfaces)) == len(surfaces)
