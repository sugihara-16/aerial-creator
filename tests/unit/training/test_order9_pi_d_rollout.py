from __future__ import annotations

import torch

from amsrr.feasibility.checker import FeasibilityChecker
from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.policies.design_policy_base import DesignPolicyContext
from amsrr.policies.design_teacher import (
    DesignTeacherVariant,
    DeterministicDesignTeacher,
)
from amsrr.policies.order9_design_policy import (
    Order9AutoregressiveDesignPolicy,
    Order9DesignPolicyConfig,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.order9_curriculum import Order9PPOOptimizationConfig
from amsrr.training.order9_pi_d_rollout import (
    order9_pi_d_executed_record,
    order9_pi_d_rejection_record,
    sample_order9_pi_d_proposal,
    sample_order9_pi_d_with_hard_checker,
)
from amsrr.training.order9_ppo import update_order9_pi_d_ppo


CHECKPOINT_SHA = "4" * 64


def test_pi_d_stochastic_proposal_replays_exact_masks_into_ppo(
    grasp_carry_dict: dict,
) -> None:
    torch.manual_seed(9009)
    context = _context(grasp_carry_dict)
    policy = Order9AutoregressiveDesignPolicy(
        Order9DesignPolicyConfig(d_model=32, maximum_design_steps=64)
    )

    sample = sample_order9_pi_d_proposal(
        policy,
        context,
        checker=FeasibilityChecker(),
        checkpoint_sha256=CHECKPOINT_SHA,
    )
    record = order9_pi_d_executed_record(
        sample,
        record_id="pi-d-executed",
        episode_id="pi-d-executed-episode",
        split=DatasetSplit.TRAIN,
        context=context,
        episode_return=2.5,
        task_success=True,
        failure_reason=None,
    )

    assert sample.accepted
    assert all(step.behavior_trace.stochastic for step in sample.steps)
    assert all(
        step.candidates[step.selected_candidate_index].valid
        for step in sample.steps
    )
    assert sum(step.reward for step in record.steps) == 2.5
    assert record.steps[-1].terminal
    assert record.trajectory_provenance.metadata["fallback_reward_credited"] is False

    result = update_order9_pi_d_ppo(
        policy,
        [record],
        physical_model=context.physical_model,
        optimizer=torch.optim.Adam(policy.parameters(), lr=1.0e-4),
        config=Order9PPOOptimizationConfig(
            epochs_per_update=1,
            minibatch_size=64,
        ),
        behavior_checkpoint_sha256=CHECKPOINT_SHA,
        seed=9009,
    )
    assert result.sample_count == len(record.steps)
    assert result.optimizer_step_count == 1


def test_pi_d_incomplete_proposals_are_negative_and_fallback_is_not_credited(
    grasp_carry_dict: dict,
) -> None:
    context = _context(grasp_carry_dict)
    policy = Order9AutoregressiveDesignPolicy(
        Order9DesignPolicyConfig(d_model=32, maximum_design_steps=1)
    )
    fallback_design = DeterministicDesignTeacher().generate(
        context,
        variant=DesignTeacherVariant.SYMMETRIC_TWO_ANCHOR_GRASP,
    ).design_output
    decision = sample_order9_pi_d_with_hard_checker(
        policy,
        context,
        checker=FeasibilityChecker(),
        fallback=_Fallback(fallback_design),
        checkpoint_sha256=CHECKPOINT_SHA,
        max_proposal_attempts=2,
    )

    assert decision.used_fallback
    assert decision.accepted_sample is None
    assert len(decision.rejected_samples) == 2
    rejection = order9_pi_d_rejection_record(
        decision.rejected_samples[0],
        record_id="pi-d-rejected",
        episode_id="pi-d-rejected-segment",
        split=DatasetSplit.TRAIN,
        context=context,
        rejection_penalty=1.0,
    )
    assert rejection.episode_return == -1.0
    assert rejection.steps[-1].reward == -1.0
    assert rejection.steps[-1].terminal
    assert rejection.design_output is None
    assert rejection.trajectory_provenance.metadata["fallback_reward_credited"] is False


def _context(grasp_carry_dict: dict) -> DesignPolicyContext:
    task = TaskSpec.from_dict(grasp_carry_dict)
    irg = IRGBuilder().build(task)
    return DesignPolicyContext(
        task_spec=task,
        irg=irg,
        physical_model=build_physical_model_from_config(
            "configs/robot/robot_model.yaml"
        ),
        interaction_envelope=InteractionEnvelopeExtractor().extract(irg),
    )


class _Fallback:
    fallback_version = "test-pi-d-fallback-v1"

    def __init__(self, design) -> None:
        self._design = design

    def design(self, context):
        del context
        return self._design
