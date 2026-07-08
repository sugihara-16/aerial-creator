from __future__ import annotations

from amsrr.feasibility.checker import FeasibilityChecker
from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.policies.design_candidate_generator import DesignCandidateGenerator
from amsrr.policies.design_policy_base import DesignPolicyContext, FixedSimpleDesignPolicy
from amsrr.policies.design_teacher import DesignTeacherVariant, DeterministicDesignTeacher
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.morphology import DesignActionType
from amsrr.schemas.task_spec import TaskSpec


def _context(grasp_carry_dict: dict) -> DesignPolicyContext:
    task = TaskSpec.from_dict(grasp_carry_dict)
    irg = IRGBuilder().build(task)
    envelope = InteractionEnvelopeExtractor().extract(irg)
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    return DesignPolicyContext(
        task_spec=task,
        irg=irg,
        interaction_envelope=envelope,
        physical_model=physical_model,
    )


def test_design_teacher_selects_p1_grasp_support_variant(grasp_carry_dict: dict) -> None:
    context = _context(grasp_carry_dict)
    example = DeterministicDesignTeacher().generate(context)
    design = example.design_output

    assert example.variant == DesignTeacherVariant.TRI_ANCHOR_SUPPORT_GRASP
    assert design.task_id == context.task_spec.task_id
    assert design.irg_id == context.irg.irg_id
    assert design.target_morphology.graph_id.startswith("morphology:grasp_carry_box_001:")
    assert design.target_morphology.graph_id.endswith("tri_anchor_support_grasp")
    assert design.design_scores["fixed_simple_p1"] == 1.0
    assert design.design_scores["p2_grasp_carry_variant_builder"] == 1.0
    assert design.design_scores["teacher_action_count"] == float(len(design.design_actions))
    assert {anchor.anchor_type for anchor in design.target_morphology.robot_anchors} == {"grasp", "support"}
    assert design.design_actions[-1].action_type == DesignActionType.STOP


def test_design_candidate_trace_masks_stop_until_final_step(grasp_carry_dict: dict) -> None:
    example = DeterministicDesignTeacher().generate(_context(grasp_carry_dict))

    assert len(example.candidate_trace) == len(example.design_output.design_actions)
    for step in example.candidate_trace[:-1]:
        stop_candidates = [candidate for candidate in step.candidates if candidate.action.action_type == DesignActionType.STOP]
        assert stop_candidates
        assert all(not candidate.valid for candidate in stop_candidates)

    final_step = example.candidate_trace[-1]
    assert final_step.selected_action.action_type == DesignActionType.STOP
    assert final_step.candidates == [
        candidate for candidate in final_step.candidates if candidate.valid and candidate.action.action_type == DesignActionType.STOP
    ]


def test_fixed_simple_design_policy_outputs_feasible_stop(grasp_carry_dict: dict) -> None:
    context = _context(grasp_carry_dict)
    design = FixedSimpleDesignPolicy().design(context)
    feasibility = FeasibilityChecker().check_design(
        design,
        task_spec=context.task_spec,
        irg=context.irg,
        physical_model=context.physical_model,
    )
    stop_candidate = DesignCandidateGenerator().final_stop_candidate(
        design,
        task_spec=context.task_spec,
        irg=context.irg,
        feasibility_result=feasibility,
    )

    assert feasibility.feasible
    assert stop_candidate.valid
    assert stop_candidate.reason_code == "stop_valid"
    assert not hasattr(design, "rotor_thrusts_n")
