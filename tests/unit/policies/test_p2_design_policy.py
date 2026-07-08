from __future__ import annotations

from dataclasses import replace

from amsrr.feasibility.violation_codes import F_CLOSED_LOOP_REJECT_V1
from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.morphology.grasp_carry_designs import (
    GRASP_CARRY_VARIANT_ORDER,
    GraspCarryMorphologyVariant,
    build_grasp_carry_variant_design_output,
)
from amsrr.policies.design_policy_base import DesignPolicyContext
from amsrr.policies.design_policy_p2 import P2DesignPolicy
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.task_spec import TaskSpec


def _context(grasp_carry_dict: dict) -> DesignPolicyContext:
    task = TaskSpec.from_dict(grasp_carry_dict)
    irg = IRGBuilder().build(task)
    envelope = InteractionEnvelopeExtractor().extract(irg)
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    return DesignPolicyContext(
        task_spec=task,
        irg=irg,
        physical_model=physical_model,
        interaction_envelope=envelope,
    )


def test_p2_design_policy_enumerates_variants_and_selects_best_accepted(grasp_carry_dict: dict) -> None:
    context = _context(grasp_carry_dict)
    policy = P2DesignPolicy()

    selection = policy.evaluate_candidates(context)
    design = policy.design(context)

    assert len(selection.candidates) == len(GRASP_CARRY_VARIANT_ORDER)
    assert len(selection.accepted_candidates) == len(GRASP_CARRY_VARIANT_ORDER)
    assert selection.rejected_candidates == []
    assert selection.selected_candidate.variant == GraspCarryMorphologyVariant.TRI_ANCHOR_SUPPORT_GRASP.value
    assert design.target_morphology.graph_id == selection.selected_candidate.design_output.target_morphology.graph_id
    assert design.design_scores["p2_design_policy_selected"] == 1.0
    assert design.design_scores["p2_design_policy_candidate_count"] == float(len(GRASP_CARRY_VARIANT_ORDER))
    assert design.design_scores["p2_design_policy_accepted_count"] == float(len(GRASP_CARRY_VARIANT_ORDER))
    assert design.design_scores["p2_design_policy_rejected_count"] == 0.0

    accepted_scores = [candidate.soft_score for candidate in selection.accepted_candidates]
    assert selection.selected_candidate.soft_score == max(accepted_scores)


def test_p2_design_policy_splits_rejected_candidates_with_feasibility_checker(grasp_carry_dict: dict) -> None:
    context = _context(grasp_carry_dict)
    good_design = build_grasp_carry_variant_design_output(
        context.task_spec,
        context.irg,
        context.physical_model,
        variant=GraspCarryMorphologyVariant.TRI_ANCHOR_SUPPORT_GRASP,
    )
    bad_design = replace(
        good_design,
        target_morphology=replace(good_design.target_morphology, is_closed_loop=True),
    )
    policy = P2DesignPolicy()

    selection = policy.evaluate_design_outputs(
        context,
        [
            (GraspCarryMorphologyVariant.CHAIN_GRASP, bad_design),
            (GraspCarryMorphologyVariant.TRI_ANCHOR_SUPPORT_GRASP, good_design),
        ],
    )

    assert len(selection.accepted_candidates) == 1
    assert len(selection.rejected_candidates) == 1
    assert selection.selected_candidate.accepted is True
    assert selection.selected_candidate.variant == GraspCarryMorphologyVariant.TRI_ANCHOR_SUPPORT_GRASP.value
    rejected = selection.rejected_candidates[0]
    assert F_CLOSED_LOOP_REJECT_V1 in rejected.rejection_reason
    assert rejected.feasibility_result.proxy_scores[f"L_{F_CLOSED_LOOP_REJECT_V1}"] == 1.0
    assert rejected.design_output.design_scores["p2_design_policy_accepted"] == 0.0
    assert selection.selected_candidate.design_output.design_scores["p2_design_policy_accepted_count"] == 1.0
    assert selection.selected_candidate.design_output.design_scores["p2_design_policy_rejected_count"] == 1.0


def test_p2_design_policy_falls_back_to_best_rejected_when_none_accepted(grasp_carry_dict: dict) -> None:
    context = _context(grasp_carry_dict)
    design_a = build_grasp_carry_variant_design_output(
        context.task_spec,
        context.irg,
        context.physical_model,
        variant=GraspCarryMorphologyVariant.CHAIN_GRASP,
    )
    design_b = build_grasp_carry_variant_design_output(
        context.task_spec,
        context.irg,
        context.physical_model,
        variant=GraspCarryMorphologyVariant.SYMMETRIC_TWO_ANCHOR_GRASP,
    )
    bad_a = replace(design_a, target_morphology=replace(design_a.target_morphology, is_closed_loop=True))
    bad_b = replace(design_b, target_morphology=replace(design_b.target_morphology, is_closed_loop=True))

    selection = P2DesignPolicy().evaluate_design_outputs(
        context,
        [
            (GraspCarryMorphologyVariant.CHAIN_GRASP, bad_a),
            (GraspCarryMorphologyVariant.SYMMETRIC_TWO_ANCHOR_GRASP, bad_b),
        ],
    )

    assert selection.accepted_candidates == []
    assert len(selection.rejected_candidates) == 2
    assert selection.selected_candidate.accepted is False
    assert selection.selected_candidate.soft_score == max(candidate.soft_score for candidate in selection.rejected_candidates)
    assert selection.selected_candidate.design_output.design_scores["p2_design_policy_selected"] == 1.0
