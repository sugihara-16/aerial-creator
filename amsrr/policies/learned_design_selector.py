from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from amsrr.feasibility.checker import FeasibilityChecker
from amsrr.policies.design_policy_base import DesignPolicyContext
from amsrr.policies.design_policy_p2 import (
    P2DesignCandidateEvaluation,
    P2DesignPolicy,
    P2DesignSelection,
)
from amsrr.schemas.feasibility import FeasibilityResult
from amsrr.schemas.morphology import DesignOutput
from amsrr.training.p2_learned_scorer import TinyP2MLP
from amsrr.training.p2_learning_dataset import P2_LEARNING_FEATURE_NAMES
from amsrr.training.p4_3_pi_d_training import (
    P4_3_PI_D_CHECKPOINT_TASK,
    p2_candidate_feature_vector,
)
from amsrr.utils.hashing import hash_file


P4_3_LEARNED_DESIGN_SELECTOR_VERSION = "p4_3_learned_design_selector_v1"


@dataclass(frozen=True)
class LearnedDesignSelectorConfig:
    ood_absolute_margin: float = 0.5
    ood_relative_margin: float = 0.25

    def __post_init__(self) -> None:
        if self.ood_absolute_margin < 0.0 or self.ood_relative_margin < 0.0:
            raise ValueError("OOD margins must be non-negative")


@dataclass(frozen=True)
class LearnedDesignSelection:
    candidates: list[P2DesignCandidateEvaluation]
    accepted_candidates: list[P2DesignCandidateEvaluation]
    rejected_candidates: list[P2DesignCandidateEvaluation]
    selected_candidate: P2DesignCandidateEvaluation
    learned_scores: dict[int, float]
    learned_ranking_candidate_ids: list[int]
    hard_feasible_candidate_ids: list[int]
    selected_recheck: FeasibilityResult | None
    used_learned_ranking: bool
    fallback_used: bool
    fallback_reason: str | None
    checkpoint_hash: str | None
    policy_version: str = P4_3_LEARNED_DESIGN_SELECTOR_VERSION


class LearnedDesignSelector:
    """Outcome scorer wrapped by deterministic pre/post feasibility gates."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        fallback_policy: P2DesignPolicy | None = None,
        feasibility_checker: FeasibilityChecker | None = None,
        config: LearnedDesignSelectorConfig | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.fallback_policy = fallback_policy or P2DesignPolicy()
        self.feasibility_checker = feasibility_checker or FeasibilityChecker()
        self.config = config or LearnedDesignSelectorConfig()
        self._model: TinyP2MLP | None = None
        self._feature_min: list[float] = []
        self._feature_max: list[float] = []
        self._target_mean = 0.0
        self._target_scale = 1.0
        self._checkpoint_hash: str | None = None
        self._checkpoint_error: str | None = None
        self._load_runtime_checkpoint()

    @property
    def checkpoint_usable(self) -> bool:
        return self._model is not None and self._checkpoint_error is None

    @property
    def checkpoint_error(self) -> str | None:
        return self._checkpoint_error

    def design(self, context: DesignPolicyContext) -> DesignOutput:
        return self.evaluate_candidates(context).selected_candidate.design_output

    def evaluate_candidates(self, context: DesignPolicyContext) -> LearnedDesignSelection:
        deterministic = self.fallback_policy.evaluate_candidates(context)
        hard_feasible = list(deterministic.accepted_candidates)
        if not self.checkpoint_usable:
            return self._fallback(
                deterministic,
                reason=f"checkpoint_invalid:{self._checkpoint_error or 'unavailable'}",
                hard_feasible=hard_feasible,
            )
        if not hard_feasible:
            return self._fallback(
                deterministic,
                reason="no_hard_feasible_candidate",
                hard_feasible=hard_feasible,
            )

        feature_rows: list[list[float]] = []
        for candidate in hard_feasible:
            try:
                features = p2_candidate_feature_vector(candidate)
            except Exception as exc:
                return self._fallback(
                    deterministic,
                    reason=f"feature_invalid:{type(exc).__name__}:{exc}",
                    hard_feasible=hard_feasible,
                )
            if self._is_ood(features):
                return self._fallback(
                    deterministic,
                    reason=f"out_of_distribution_feature:candidate:{candidate.candidate_id}",
                    hard_feasible=hard_feasible,
                )
            feature_rows.append(features)

        assert self._model is not None
        tensor = torch.tensor(feature_rows, dtype=torch.float32)
        self._model.eval()
        with torch.no_grad():
            normalized_scores = self._model(tensor)
            raw_scores = normalized_scores * self._target_scale + self._target_mean
        values = [float(value) for value in raw_scores.tolist()]
        if not all(math.isfinite(value) for value in values):
            return self._fallback(
                deterministic,
                reason="non_finite_prediction",
                hard_feasible=hard_feasible,
            )
        learned_scores = {
            candidate.candidate_id: score
            for candidate, score in zip(hard_feasible, values, strict=True)
        }
        ranking = sorted(
            hard_feasible,
            key=lambda candidate: (-learned_scores[candidate.candidate_id], candidate.candidate_id),
        )
        selected = ranking[0]

        recheck = self.feasibility_checker.check_design(
            selected.design_output,
            task_spec=context.task_spec,
            irg=context.irg,
            physical_model=context.physical_model,
        )
        if not recheck.feasible:
            return self._fallback(
                deterministic,
                reason="selected_candidate_recheck_failed",
                hard_feasible=hard_feasible,
                learned_scores=learned_scores,
                ranking=[candidate.candidate_id for candidate in ranking],
                selected_recheck=recheck,
            )

        annotated = [
            _annotate_candidate(
                candidate,
                learned_score=learned_scores.get(candidate.candidate_id),
                learned_selected=candidate.candidate_id == selected.candidate_id,
                fallback_used=False,
            )
            for candidate in deterministic.candidates
        ]
        selected_annotated = next(
            candidate for candidate in annotated if candidate.candidate_id == selected.candidate_id
        )
        return LearnedDesignSelection(
            candidates=annotated,
            accepted_candidates=[candidate for candidate in annotated if candidate.accepted],
            rejected_candidates=[candidate for candidate in annotated if not candidate.accepted],
            selected_candidate=selected_annotated,
            learned_scores=learned_scores,
            learned_ranking_candidate_ids=[candidate.candidate_id for candidate in ranking],
            hard_feasible_candidate_ids=sorted(
                candidate.candidate_id for candidate in hard_feasible
            ),
            selected_recheck=recheck,
            used_learned_ranking=True,
            fallback_used=False,
            fallback_reason=None,
            checkpoint_hash=self._checkpoint_hash,
        )

    def _fallback(
        self,
        deterministic: P2DesignSelection,
        *,
        reason: str,
        hard_feasible: list[P2DesignCandidateEvaluation],
        learned_scores: dict[int, float] | None = None,
        ranking: list[int] | None = None,
        selected_recheck: FeasibilityResult | None = None,
    ) -> LearnedDesignSelection:
        learned_scores = learned_scores or {}
        ranking = ranking or []
        selected_id = deterministic.selected_candidate.candidate_id
        annotated = [
            _annotate_candidate(
                candidate,
                learned_score=learned_scores.get(candidate.candidate_id),
                learned_selected=False,
                fallback_used=True,
            )
            for candidate in deterministic.candidates
        ]
        return LearnedDesignSelection(
            candidates=annotated,
            accepted_candidates=[candidate for candidate in annotated if candidate.accepted],
            rejected_candidates=[candidate for candidate in annotated if not candidate.accepted],
            selected_candidate=next(
                candidate for candidate in annotated if candidate.candidate_id == selected_id
            ),
            learned_scores=learned_scores,
            learned_ranking_candidate_ids=ranking,
            hard_feasible_candidate_ids=sorted(
                candidate.candidate_id for candidate in hard_feasible
            ),
            selected_recheck=selected_recheck,
            used_learned_ranking=False,
            fallback_used=True,
            fallback_reason=reason,
            checkpoint_hash=self._checkpoint_hash,
        )

    def _load_runtime_checkpoint(self) -> None:
        try:
            payload = _load_checkpoint(self.checkpoint_path)
            if payload.get("task") != P4_3_PI_D_CHECKPOINT_TASK:
                raise ValueError("checkpoint task mismatch")
            if payload.get("model_type") != "TinyP2MLP":
                raise ValueError("checkpoint model_type mismatch")
            if payload.get("feature_names") != P2_LEARNING_FEATURE_NAMES:
                raise ValueError("checkpoint feature layout mismatch")
            state_dict = payload["state_dict"]
            first_weight = state_dict["net.0.weight"]
            if int(first_weight.shape[1]) != len(P2_LEARNING_FEATURE_NAMES):
                raise ValueError("checkpoint input dimension mismatch")
            hidden_dim = int(first_weight.shape[0])
            model = TinyP2MLP(
                input_dim=len(P2_LEARNING_FEATURE_NAMES),
                hidden_dim=hidden_dim,
            )
            model.load_state_dict(state_dict, strict=True)
            feature_min = [float(value) for value in payload["feature_min"]]
            feature_max = [float(value) for value in payload["feature_max"]]
            if len(feature_min) != len(P2_LEARNING_FEATURE_NAMES) or len(feature_max) != len(
                P2_LEARNING_FEATURE_NAMES
            ):
                raise ValueError("checkpoint feature bounds mismatch")
            if not all(
                math.isfinite(lower) and math.isfinite(upper) and lower <= upper
                for lower, upper in zip(feature_min, feature_max, strict=True)
            ):
                raise ValueError("checkpoint feature bounds are invalid")
            target_mean = float(payload["target_mean"])
            target_scale = float(payload["target_scale"])
            if not math.isfinite(target_mean) or not math.isfinite(target_scale) or target_scale <= 0.0:
                raise ValueError("checkpoint target normalization is invalid")
            checkpoint_hash = hash_file(self.checkpoint_path)
        except Exception as exc:
            self._checkpoint_error = f"{type(exc).__name__}:{exc}"
            return
        self._model = model
        self._feature_min = feature_min
        self._feature_max = feature_max
        self._target_mean = target_mean
        self._target_scale = target_scale
        self._checkpoint_hash = checkpoint_hash

    def _is_ood(self, features: list[float]) -> bool:
        if len(features) != len(self._feature_min) or not all(math.isfinite(value) for value in features):
            return True
        for value, lower, upper in zip(
            features,
            self._feature_min,
            self._feature_max,
            strict=True,
        ):
            span = upper - lower
            margin = self.config.ood_absolute_margin + self.config.ood_relative_margin * span
            if value < lower - margin or value > upper + margin:
                return True
        return False


OutcomeConditionedDesignSelector = LearnedDesignSelector


def _annotate_candidate(
    candidate: P2DesignCandidateEvaluation,
    *,
    learned_score: float | None,
    learned_selected: bool,
    fallback_used: bool,
) -> P2DesignCandidateEvaluation:
    scores = {
        **candidate.design_output.design_scores,
        "p4_3_pi_d_learned_selected": 1.0 if learned_selected else 0.0,
        "p4_3_pi_d_fallback_used": 1.0 if fallback_used else 0.0,
        "p4_3_pi_d_hard_feasible": 1.0 if candidate.accepted else 0.0,
    }
    if learned_score is not None:
        scores["p4_3_pi_d_outcome_score"] = learned_score
    return replace(
        candidate,
        design_output=replace(candidate.design_output, design_scores=scores),
    )


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    return payload
