from __future__ import annotations

import torch
from amsrr.policies.order9_design_runtime import (
    Order9DesignRuntime,
    Order9DesignRuntimeConfig,
)

from amsrr.feasibility.checker import FeasibilityChecker
from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.policies.design_policy_base import DesignPolicyContext
from amsrr.policies.design_teacher import (
    DesignTeacherVariant,
    DeterministicDesignTeacher,
)
from amsrr.policies.order9_design_grammar import (
    ORDER9_DESIGN_GRAMMAR_VERSION,
    Order9DesignGrammar,
)
from amsrr.policies.order9_design_policy import (
    Order9AutoregressiveDesignPolicy,
    Order9DesignPolicyConfig,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.morphology import DesignActionType
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.order9_pi_d_learning import (
    compute_order9_pi_d_behavior_cloning_loss,
    compute_order9_pi_d_masked_ppo_loss,
)


def test_order9_grammar_replays_teacher_through_real_masks_and_feasible_stop(
    grasp_carry_dict: dict,
) -> None:
    context = _context(grasp_carry_dict)
    target = DeterministicDesignTeacher().generate(
        context,
        variant=DesignTeacherVariant.SYMMETRIC_TWO_ANCHOR_GRASP,
    ).design_output
    grammar = Order9DesignGrammar(context)

    trace = grammar.teacher_trace(target)
    state = grammar.initial_state()
    for step in trace:
        assert step.state == state
        assert any(
            candidate.action.action_type == DesignActionType.STOP
            for candidate in step.candidate_step.candidates
        )
        selected = next(
            candidate
            for candidate in step.candidate_step.candidates
            if candidate.action.to_dict() == step.candidate_step.selected_action.to_dict()
        )
        assert selected.valid
        state = grammar.apply(state, selected)

    assert state.stopped
    design = grammar.build_design_output(state)
    result = FeasibilityChecker().check_design(
        design,
        task_spec=context.task_spec,
        irg=context.irg,
        physical_model=context.physical_model,
    )
    assert result.feasible
    assert design.design_scores["order9_sequential_pi_d"] == 1.0
    assert design.target_morphology.control_groups[0].metadata["grammar_version"] == (
        ORDER9_DESIGN_GRAMMAR_VERSION
    )
    assert design.design_actions[-1].action_type == DesignActionType.STOP
    assert len(design.target_morphology.modules) == len(target.target_morphology.modules)
    assert {
        frozenset((edge.src_port_id, edge.dst_port_id))
        for edge in design.target_morphology.dock_edges
    } == {
        frozenset((edge.src_port_id, edge.dst_port_id))
        for edge in target.target_morphology.dock_edges
    }


def test_stop_is_masked_before_hard_feasible_design(grasp_carry_dict: dict) -> None:
    grammar = Order9DesignGrammar(_context(grasp_carry_dict))

    candidates = grammar.candidates(grammar.initial_state())
    stop = next(
        candidate
        for candidate in candidates
        if candidate.action.action_type == DesignActionType.STOP
    )

    assert stop.valid is False
    assert stop.reason_code == "stop_masked_base_unassigned"


def test_sequential_pi_d_bc_and_masked_ppo_contract_backpropagate(
    grasp_carry_dict: dict,
) -> None:
    context = _context(grasp_carry_dict)
    target = DeterministicDesignTeacher().generate(
        context,
        variant=DesignTeacherVariant.SYMMETRIC_TWO_ANCHOR_GRASP,
    ).design_output
    trace = Order9DesignGrammar(context).teacher_trace(target)
    policy = Order9AutoregressiveDesignPolicy(
        Order9DesignPolicyConfig(d_model=48)
    )

    losses = compute_order9_pi_d_behavior_cloning_loss(
        policy,
        context,
        trace,
        design_return=2.0,
        entropy_bonus_weight=0.01,
    )
    losses.total.backward()

    assert losses.step_count == len(trace)
    assert torch.isfinite(losses.total)
    assert torch.isfinite(losses.policy)
    assert torch.isfinite(losses.value)
    assert torch.isfinite(losses.entropy)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in policy.parameters()
    )

    ppo = compute_order9_pi_d_masked_ppo_loss(
        new_log_prob=torch.tensor([-0.9, -1.1], requires_grad=True),
        old_log_prob=torch.tensor([-1.0, -1.0]),
        advantages=torch.tensor([1.0, -0.5]),
        new_values=torch.tensor([0.2, 0.4], requires_grad=True),
        returns=torch.tensor([0.5, 0.0]),
        entropy=torch.tensor([0.4, 0.3]),
    )
    assert torch.isfinite(ppo)


def test_order9_design_runtime_retries_then_hard_checks_without_projection(
    grasp_carry_dict: dict,
) -> None:
    context = _context(grasp_carry_dict)
    target = DeterministicDesignTeacher().generate(
        context,
        variant=DesignTeacherVariant.SYMMETRIC_TWO_ANCHOR_GRASP,
    ).design_output
    proposal = _SequenceProposal(target, failures=1)
    runtime = Order9DesignRuntime(
        proposal=proposal,
        checker=FeasibilityChecker(),
        fallback=_Fallback(target),
        config=Order9DesignRuntimeConfig(
            maximum_proposal_attempts=3,
            deterministic_proposal=True,
        ),
    )

    decision = runtime.design(context)

    assert not decision.used_fallback
    assert decision.proposal_attempt_count == 2
    assert len(decision.rejected_proposal_reasons) == 1
    assert decision.design_output.to_dict() == target.to_dict()
    assert decision.feasibility_result.feasible


def test_order9_design_runtime_uses_separately_checked_fallback(
    grasp_carry_dict: dict,
) -> None:
    context = _context(grasp_carry_dict)
    target = DeterministicDesignTeacher().generate(
        context,
        variant=DesignTeacherVariant.SYMMETRIC_TWO_ANCHOR_GRASP,
    ).design_output
    runtime = Order9DesignRuntime(
        proposal=_SequenceProposal(target, failures=99),
        checker=FeasibilityChecker(),
        fallback=_Fallback(target),
        config=Order9DesignRuntimeConfig(maximum_proposal_attempts=2),
    )

    decision = runtime.design(context)

    assert decision.used_fallback
    assert decision.proposal_attempt_count == 2
    assert decision.fallback_version == "test_deterministic_fallback_v1"
    assert decision.feasibility_result.feasible


def _context(grasp_carry_dict: dict) -> DesignPolicyContext:
    task = TaskSpec.from_dict(grasp_carry_dict)
    irg = IRGBuilder().build(task)
    return DesignPolicyContext(
        task_spec=task,
        irg=irg,
        physical_model=build_physical_model_from_config("configs/robot/robot_model.yaml"),
        interaction_envelope=InteractionEnvelopeExtractor().extract(irg),
    )


class _SequenceProposal:
    def __init__(self, design, *, failures: int) -> None:
        self.design = design
        self.failures = failures
        self.calls = 0

    def propose(self, context, *, deterministic, checker=None):
        del context, deterministic, checker
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("synthetic learned proposal rejection")
        return self.design


class _Fallback:
    fallback_version = "test_deterministic_fallback_v1"

    def __init__(self, design) -> None:
        self._design = design

    def design(self, context):
        del context
        return self._design
