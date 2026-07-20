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
