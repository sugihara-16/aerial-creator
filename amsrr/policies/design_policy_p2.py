from __future__ import annotations

from dataclasses import dataclass, replace

from amsrr.feasibility.checker import FeasibilityChecker
from amsrr.morphology.grasp_carry_designs import (
    GRASP_CARRY_VARIANT_ORDER,
    GraspCarryMorphologyVariant,
    build_grasp_carry_variant_design_output,
)
from amsrr.policies.design_policy_base import DesignPolicyContext
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.feasibility import FeasibilityResult
from amsrr.schemas.morphology import DesignOutput
from amsrr.schemas.task_spec import TaskType


P2_DESIGN_POLICY_VERSION = "p2_agent_e_design_policy_v1"


@dataclass(frozen=True)
class P2DesignPolicyConfig:
    variants: tuple[GraspCarryMorphologyVariant, ...] = GRASP_CARRY_VARIANT_ORDER
    prefer_accepted: bool = True
    coverage_weight: float = 4.0
    reachability_weight: float = 2.0
    slot_weight: float = 1.0
    thrust_weight: float = 0.6
    payload_weight: float = 0.4
    optional_support_bonus: float = 0.5
    module_penalty: float = 0.05
    dock_edge_penalty: float = 0.025
    hard_rejection_penalty: float = 100.0


@dataclass(frozen=True)
class P2DesignCandidateEvaluation:
    candidate_id: int
    variant: str
    design_output: DesignOutput
    feasibility_result: FeasibilityResult
    accepted: bool
    soft_score: float
    rejection_reason: str | None = None


@dataclass(frozen=True)
class P2DesignSelection:
    candidates: list[P2DesignCandidateEvaluation]
    accepted_candidates: list[P2DesignCandidateEvaluation]
    rejected_candidates: list[P2DesignCandidateEvaluation]
    selected_candidate: P2DesignCandidateEvaluation
    policy_version: str = P2_DESIGN_POLICY_VERSION


class P2DesignPolicy:
    """Deterministic P2 π_D scaffold over grasp/carry morphology candidates."""

    def __init__(
        self,
        config: P2DesignPolicyConfig | None = None,
        feasibility_checker: FeasibilityChecker | None = None,
    ) -> None:
        self.config = config or P2DesignPolicyConfig()
        self.feasibility_checker = feasibility_checker or FeasibilityChecker()

    def design(self, context: DesignPolicyContext) -> DesignOutput:
        return self.evaluate_candidates(context).selected_candidate.design_output

    def enumerate_candidate_designs(
        self,
        context: DesignPolicyContext,
    ) -> list[tuple[str, DesignOutput]]:
        if context.task_spec.task_type != TaskType.OBJECT_GRASP_CARRY:
            raise SchemaValidationError("P2DesignPolicy currently supports object_grasp_carry only")
        candidates: list[tuple[str, DesignOutput]] = []
        for variant in self.config.variants:
            candidates.append(
                (
                    variant.value,
                    build_grasp_carry_variant_design_output(
                        context.task_spec,
                        context.irg,
                        context.physical_model,
                        variant=variant,
                    ),
                )
            )
        return candidates

    def evaluate_candidates(self, context: DesignPolicyContext) -> P2DesignSelection:
        return self.evaluate_design_outputs(context, self.enumerate_candidate_designs(context))

    def evaluate_design_outputs(
        self,
        context: DesignPolicyContext,
        candidate_designs: list[tuple[str | GraspCarryMorphologyVariant, DesignOutput]],
    ) -> P2DesignSelection:
        if not candidate_designs:
            raise SchemaValidationError("P2DesignPolicy requires at least one candidate design")
        evaluated: list[P2DesignCandidateEvaluation] = []
        for candidate_id, (variant, design_output) in enumerate(candidate_designs):
            variant_value = variant.value if isinstance(variant, GraspCarryMorphologyVariant) else str(variant)
            feasibility = self.feasibility_checker.check_design(
                design_output,
                task_spec=context.task_spec,
                irg=context.irg,
                physical_model=context.physical_model,
            )
            accepted = feasibility.feasible
            soft_score = self._soft_score(design_output, feasibility, variant_value=variant_value)
            rejection_reason = None if accepted else _rejection_reason(feasibility)
            annotated_design = _annotate_candidate_design(
                design_output,
                candidate_id=candidate_id,
                variant_value=variant_value,
                accepted=accepted,
                soft_score=soft_score,
                selected=False,
                accepted_count=0,
                rejected_count=0,
                candidate_count=0,
            )
            evaluated.append(
                P2DesignCandidateEvaluation(
                    candidate_id=candidate_id,
                    variant=variant_value,
                    design_output=annotated_design,
                    feasibility_result=feasibility,
                    accepted=accepted,
                    soft_score=soft_score,
                    rejection_reason=rejection_reason,
                )
            )
        return self._selection(evaluated)

    def _selection(self, candidates: list[P2DesignCandidateEvaluation]) -> P2DesignSelection:
        accepted = [candidate for candidate in candidates if candidate.accepted]
        rejected = [candidate for candidate in candidates if not candidate.accepted]
        selection_pool = accepted if (self.config.prefer_accepted and accepted) else candidates
        selected = sorted(selection_pool, key=lambda item: (-item.soft_score, item.candidate_id))[0]
        annotated_candidates = [
            _replace_candidate_design(
                candidate,
                selected=candidate.candidate_id == selected.candidate_id,
                accepted_count=len(accepted),
                rejected_count=len(rejected),
                candidate_count=len(candidates),
            )
            for candidate in candidates
        ]
        selected_candidate = next(candidate for candidate in annotated_candidates if candidate.candidate_id == selected.candidate_id)
        return P2DesignSelection(
            candidates=annotated_candidates,
            accepted_candidates=[candidate for candidate in annotated_candidates if candidate.accepted],
            rejected_candidates=[candidate for candidate in annotated_candidates if not candidate.accepted],
            selected_candidate=selected_candidate,
        )

    def _soft_score(
        self,
        design_output: DesignOutput,
        feasibility_result: FeasibilityResult,
        *,
        variant_value: str,
    ) -> float:
        margins = feasibility_result.margins
        score = 0.0
        score += self.config.coverage_weight * margins.get("required_slot_anchor_capability_coverage_ratio", 0.0)
        score += self.config.reachability_weight * margins.get("coarse_reachability_ratio", 0.0)
        score += self.config.slot_weight * margins.get("required_slot_coverage_ratio", 0.0)
        score += self.config.thrust_weight * _clamp(margins.get("thrust_margin_ratio", 0.0), -1.0, 4.0)
        score += self.config.payload_weight * _clamp(margins.get("payload_margin_ratio", 0.0), -1.0, 4.0)
        score += self.config.optional_support_bonus * min(1, _anchor_count(design_output, "support"))
        score += _variant_prior(variant_value)
        score -= self.config.module_penalty * len(design_output.target_morphology.modules)
        score -= self.config.dock_edge_penalty * len(design_output.target_morphology.dock_edges)
        if not feasibility_result.feasible:
            score -= self.config.hard_rejection_penalty
            score -= float(len(feasibility_result.hard_violations))
        return score


def _replace_candidate_design(
    candidate: P2DesignCandidateEvaluation,
    *,
    selected: bool,
    accepted_count: int,
    rejected_count: int,
    candidate_count: int,
) -> P2DesignCandidateEvaluation:
    return replace(
        candidate,
        design_output=_annotate_candidate_design(
            candidate.design_output,
            candidate_id=candidate.candidate_id,
            variant_value=candidate.variant,
            accepted=candidate.accepted,
            soft_score=candidate.soft_score,
            selected=selected,
            accepted_count=accepted_count,
            rejected_count=rejected_count,
            candidate_count=candidate_count,
        ),
    )


def _annotate_candidate_design(
    design_output: DesignOutput,
    *,
    candidate_id: int,
    variant_value: str,
    accepted: bool,
    soft_score: float,
    selected: bool,
    accepted_count: int,
    rejected_count: int,
    candidate_count: int,
) -> DesignOutput:
    variant_id = _variant_id(variant_value)
    scores = {
        **design_output.design_scores,
        "p2_design_policy_candidate_id": float(candidate_id),
        "p2_design_policy_variant_id": variant_id,
        "p2_design_policy_soft_score": soft_score,
        "p2_design_policy_accepted": 1.0 if accepted else 0.0,
        "p2_design_policy_selected": 1.0 if selected else 0.0,
        "p2_design_policy_candidate_count": float(candidate_count),
        "p2_design_policy_accepted_count": float(accepted_count),
        "p2_design_policy_rejected_count": float(rejected_count),
        "p2_design_policy_version_id": 1.0,
    }
    return replace(design_output, design_scores=scores)


def _rejection_reason(feasibility_result: FeasibilityResult) -> str:
    if not feasibility_result.hard_violations:
        return "none"
    return ",".join(sorted({violation.code for violation in feasibility_result.hard_violations}))


def _variant_id(variant_value: str) -> float:
    for idx, variant in enumerate(GRASP_CARRY_VARIANT_ORDER):
        if variant.value == variant_value:
            return float(idx)
    return -1.0


def _variant_prior(variant_value: str) -> float:
    return {
        GraspCarryMorphologyVariant.CHAIN_GRASP.value: 0.0,
        GraspCarryMorphologyVariant.SYMMETRIC_TWO_ANCHOR_GRASP.value: 0.25,
        GraspCarryMorphologyVariant.TRI_ANCHOR_SUPPORT_GRASP.value: 0.6,
        GraspCarryMorphologyVariant.CENTRAL_BASE_PLUS_TWO_GRASP_ARMS.value: 0.1,
    }.get(variant_value, 0.0)


def _anchor_count(design_output: DesignOutput, anchor_type: str) -> int:
    return sum(1 for anchor in design_output.target_morphology.robot_anchors if anchor.anchor_type == anchor_type)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
