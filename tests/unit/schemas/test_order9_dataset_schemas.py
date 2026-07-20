from __future__ import annotations

import pytest

from amsrr.feasibility.checker import FeasibilityChecker
from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.policies.design_policy_base import DesignPolicyContext
from amsrr.policies.design_teacher import (
    DesignTeacherVariant,
    DeterministicDesignTeacher,
)
from amsrr.policies.order9_design_grammar import Order9DesignGrammar
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.datasets import (
    DatasetSplit,
    DesignActionCandidateRecord,
    PolicyBehaviorTrace,
    SequentialDesignStepRecord,
    SequentialDesignTrajectoryRecord,
    StageDecisionMasks,
    TrajectoryProvenance,
    TrajectorySourceKind,
)
from amsrr.schemas.task_spec import TaskSpec


def test_sequential_design_trajectory_roundtrips_exact_masks_and_history(
    grasp_carry_dict: dict,
) -> None:
    task = TaskSpec.from_dict(grasp_carry_dict)
    irg = IRGBuilder().build(task)
    envelope = InteractionEnvelopeExtractor().extract(irg)
    physical_model = build_physical_model_from_config(
        "configs/robot/robot_model.yaml"
    )
    context = DesignPolicyContext(task, irg, physical_model, envelope)
    teacher = DeterministicDesignTeacher().generate(
        context,
        variant=DesignTeacherVariant.SYMMETRIC_TWO_ANCHOR_GRASP,
    ).design_output
    grammar = Order9DesignGrammar(context)
    trace = grammar.teacher_trace(teacher)
    final_state = grammar.initial_state()
    for item in trace:
        selected = next(
            candidate
            for candidate in item.candidate_step.candidates
            if candidate.action.to_dict()
            == item.candidate_step.selected_action.to_dict()
        )
        final_state = grammar.apply(final_state, selected)
    final_design = grammar.build_design_output(final_state)
    feasibility = FeasibilityChecker().check_design(
        final_design,
        task_spec=task,
        irg=irg,
        physical_model=physical_model,
    )
    steps = []
    for index, item in enumerate(trace):
        selected_index = next(
            candidate_index
            for candidate_index, candidate in enumerate(item.candidate_step.candidates)
            if candidate.action.to_dict()
            == item.candidate_step.selected_action.to_dict()
        )
        steps.append(
            SequentialDesignStepRecord(
                step_index=index,
                partial_action_history=list(item.state.action_history),
                candidates=[
                    DesignActionCandidateRecord(
                        candidate_index=candidate_index,
                        action=candidate.action,
                        valid=candidate.valid,
                        reason_code=candidate.reason_code,
                        score_prior=candidate.score_prior,
                    )
                    for candidate_index, candidate in enumerate(
                        item.candidate_step.candidates
                    )
                ],
                selected_candidate_index=selected_index,
                reward=1.0 if index == len(trace) - 1 else 0.0,
                terminal=index == len(trace) - 1,
                truncated=False,
                behavior_trace=PolicyBehaviorTrace(
                    policy_family="pi_d",
                    policy_version="order9_teacher_v1",
                    action_semantics="masked_grammar_candidate_index",
                    action_payload={
                        "selected_action": item.candidate_step.selected_action.to_dict()
                    },
                ),
            )
        )
    record = SequentialDesignTrajectoryRecord(
        record_id="order9-pi-d-trace-1",
        episode_id="order9-design-episode-1",
        task_id=task.task_id,
        split=DatasetSplit.TRAIN,
        task_spec=task,
        irg=irg,
        interaction_envelope=envelope,
        physical_model_hash=physical_model.stable_hash(),
        steps=steps,
        design_output=final_design,
        feasibility_result=feasibility,
        episode_return=1.0,
        task_success=True,
        failure_reason=None,
        stage_masks=StageDecisionMasks(design_decision_mask=True),
        trajectory_provenance=TrajectoryProvenance(
            source_kind=TrajectorySourceKind.DETERMINISTIC_TEACHER,
            source_version="order9_design_teacher_v1",
        ),
    )

    roundtrip = SequentialDesignTrajectoryRecord.from_json(record.to_json())

    assert roundtrip.to_dict() == record.to_dict()
    assert roundtrip.feasibility_result is not None
    assert roundtrip.feasibility_result.feasible


def test_stochastic_behavior_trace_requires_replay_contract() -> None:
    with pytest.raises(SchemaValidationError, match="requires checkpoint hash"):
        PolicyBehaviorTrace.from_dict(
            {
                "policy_family": "pi_h",
                "policy_version": "pi_h_v1",
                "action_semantics": "full_tensor_action",
                "action_payload": {"assignment": [1, 0]},
                "stochastic": True,
            }
        )
