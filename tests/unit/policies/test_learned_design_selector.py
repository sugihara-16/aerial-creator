from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from amsrr.feasibility.violation_codes import F_CLOSED_LOOP_REJECT_V1
from amsrr.morphology.grasp_carry_designs import GraspCarryMorphologyVariant
from amsrr.policies.design_policy_p2 import P2DesignPolicy
from amsrr.policies.learned_design_selector import (
    LearnedDesignSelector,
    LearnedDesignSelectorConfig,
)
from amsrr.schemas.feasibility import FeasibilityResult, Violation, ViolationSeverity
from amsrr.training.p2_inspection_context import build_p2_inspection_context
from amsrr.training.p2_learned_scorer import TinyP2MLP
from amsrr.training.p2_learning_dataset import P2_LEARNING_FEATURE_NAMES
from amsrr.training.p4_3_pi_d_training import P4_3_PI_D_CHECKPOINT_TASK


def test_learned_design_selector_hard_filters_ranks_and_rechecks(tmp_path: Path) -> None:
    checkpoint_path = _checkpoint(tmp_path / "checkpoint.pt")
    inspection = build_p2_inspection_context(seed=3, sample_index=0)
    selector = LearnedDesignSelector(checkpoint_path)

    selection = selector.evaluate_candidates(inspection.design_context)

    assert selection.fallback_used is False
    assert selection.used_learned_ranking is True
    assert selection.selected_recheck is not None
    assert selection.selected_recheck.feasible is True
    assert selection.selected_candidate.candidate_id == min(selection.hard_feasible_candidate_ids)
    assert set(selection.learned_scores) == set(selection.hard_feasible_candidate_ids)
    assert selection.selected_candidate.design_output.design_scores[
        "p4_3_pi_d_learned_selected"
    ] == 1.0


def test_learned_design_selector_falls_back_for_invalid_checkpoint(tmp_path: Path) -> None:
    inspection = build_p2_inspection_context(seed=5, sample_index=0)
    invalid_path = tmp_path / "invalid.pt"
    torch.save({"task": "wrong"}, invalid_path)
    deterministic = P2DesignPolicy().evaluate_candidates(inspection.design_context)

    selection = LearnedDesignSelector(invalid_path).evaluate_candidates(inspection.design_context)

    assert selection.fallback_used is True
    assert selection.used_learned_ranking is False
    assert selection.fallback_reason is not None
    assert selection.fallback_reason.startswith("checkpoint_invalid:")
    assert selection.selected_candidate.candidate_id == deterministic.selected_candidate.candidate_id
    assert selection.selected_candidate.design_output.design_scores[
        "p4_3_pi_d_fallback_used"
    ] == 1.0


def test_learned_design_selector_never_scores_hard_infeasible_candidate(
    tmp_path: Path,
) -> None:
    inspection = build_p2_inspection_context(seed=7, sample_index=0)
    original = P2DesignPolicy().enumerate_candidate_designs(inspection.design_context)
    invalid = replace(
        original[0][1],
        target_morphology=replace(original[0][1].target_morphology, is_closed_loop=True),
    )
    good = original[1][1]
    fallback = _TwoCandidatePolicy(invalid, good)
    selector = LearnedDesignSelector(_checkpoint(tmp_path / "checkpoint.pt"), fallback_policy=fallback)

    selection = selector.evaluate_candidates(inspection.design_context)

    assert selection.fallback_used is False
    assert selection.hard_feasible_candidate_ids == [1]
    assert set(selection.learned_scores) == {1}
    assert selection.selected_candidate.candidate_id == 1
    assert F_CLOSED_LOOP_REJECT_V1 in selection.rejected_candidates[0].rejection_reason


def test_learned_design_selector_falls_back_when_selected_recheck_fails(
    tmp_path: Path,
) -> None:
    inspection = build_p2_inspection_context(seed=9, sample_index=0)
    selector = LearnedDesignSelector(
        _checkpoint(tmp_path / "checkpoint.pt"),
        feasibility_checker=_RejectingRecheck(),  # type: ignore[arg-type]
    )

    selection = selector.evaluate_candidates(inspection.design_context)

    assert selection.fallback_used is True
    assert selection.used_learned_ranking is False
    assert selection.fallback_reason == "selected_candidate_recheck_failed"
    assert selection.selected_recheck is not None
    assert selection.selected_recheck.feasible is False


def test_learned_design_selector_falls_back_for_ood_features(tmp_path: Path) -> None:
    inspection = build_p2_inspection_context(seed=13, sample_index=0)
    checkpoint_path = _checkpoint(tmp_path / "checkpoint.pt")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    payload["feature_min"] = [0.0] * len(P2_LEARNING_FEATURE_NAMES)
    payload["feature_max"] = [0.0] * len(P2_LEARNING_FEATURE_NAMES)
    torch.save(payload, checkpoint_path)
    selector = LearnedDesignSelector(
        checkpoint_path,
        config=LearnedDesignSelectorConfig(
            ood_absolute_margin=0.0,
            ood_relative_margin=0.0,
        ),
    )

    selection = selector.evaluate_candidates(inspection.design_context)

    assert selection.fallback_used is True
    assert selection.fallback_reason is not None
    assert selection.fallback_reason.startswith("out_of_distribution_feature:")


def test_learned_design_selector_falls_back_when_no_candidate_passes_hard_gate(
    tmp_path: Path,
) -> None:
    inspection = build_p2_inspection_context(seed=15, sample_index=0)
    original = P2DesignPolicy().enumerate_candidate_designs(inspection.design_context)
    invalid_a = replace(
        original[0][1],
        target_morphology=replace(original[0][1].target_morphology, is_closed_loop=True),
    )
    invalid_b = replace(
        original[1][1],
        target_morphology=replace(original[1][1].target_morphology, is_closed_loop=True),
    )
    selector = LearnedDesignSelector(
        _checkpoint(tmp_path / "checkpoint.pt"),
        fallback_policy=_TwoCandidatePolicy(invalid_a, invalid_b),
    )

    selection = selector.evaluate_candidates(inspection.design_context)

    assert selection.fallback_used is True
    assert selection.fallback_reason == "no_hard_feasible_candidate"
    assert selection.hard_feasible_candidate_ids == []
    assert selection.learned_scores == {}


class _TwoCandidatePolicy(P2DesignPolicy):
    def __init__(self, invalid, good) -> None:
        super().__init__()
        self.invalid = invalid
        self.good = good

    def enumerate_candidate_designs(self, context):
        del context
        return [
            (GraspCarryMorphologyVariant.CHAIN_GRASP, self.invalid),
            (GraspCarryMorphologyVariant.SYMMETRIC_TWO_ANCHOR_GRASP, self.good),
        ]


class _RejectingRecheck:
    def check_design(self, design_output, **kwargs) -> FeasibilityResult:
        del design_output, kwargs
        return FeasibilityResult(
            feasible=False,
            hard_violations=[
                Violation(
                    code="F_RECHECK_TEST",
                    severity=ViolationSeverity.HARD,
                    message="unit-test deterministic recheck rejection",
                )
            ],
            soft_violations=[],
            margins={},
            proxy_scores={},
            checker_version="unit_recheck",
        )


def _checkpoint(path: Path) -> Path:
    model = TinyP2MLP(input_dim=len(P2_LEARNING_FEATURE_NAMES), hidden_dim=8)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    torch.save(
        {
            "model_type": "TinyP2MLP",
            "task": P4_3_PI_D_CHECKPOINT_TASK,
            "state_dict": model.state_dict(),
            "feature_names": list(P2_LEARNING_FEATURE_NAMES),
            "feature_min": [-100.0] * len(P2_LEARNING_FEATURE_NAMES),
            "feature_max": [100.0] * len(P2_LEARNING_FEATURE_NAMES),
            "target_mean": 0.0,
            "target_scale": 1.0,
        },
        path,
    )
    return path
