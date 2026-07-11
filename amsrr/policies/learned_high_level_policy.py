from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn

from amsrr.policies.assignment_feasibility import (
    evaluate_selected_assignment_feasibility,
)
from amsrr.policies.contact_candidate_encoder import (
    DEFAULT_CONTACT_CANDIDATE_D_MODEL,
    ContactCandidateEncoder,
    ContactCandidateEncoderOutput,
)
from amsrr.policies.contact_wrench_trajectory import (
    P4_2DeterministicGraspCarryPlanner,
)
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.schemas.common import SchemaBase, SchemaValidationError
from amsrr.schemas.contact_candidates import ContactCandidateGroupProposal, ContactCandidateSet
from amsrr.schemas.irg import IRGNodeType
from amsrr.schemas.policies import ContactAssignment, ContactWrenchTrajectory


@dataclass(frozen=True)
class LearnedHighLevelPolicyConfig:
    encoder_d_model: int = DEFAULT_CONTACT_CANDIDATE_D_MODEL
    hidden_dim: int = 32
    max_timing_residual_s: float = 0.05
    timing_residual_enabled: bool = True

    def validate(self) -> None:
        if self.encoder_d_model <= 0:
            raise SchemaValidationError(
                "LearnedHighLevelPolicyConfig.encoder_d_model must be positive"
            )
        if self.hidden_dim <= 0:
            raise SchemaValidationError(
                "LearnedHighLevelPolicyConfig.hidden_dim must be positive"
            )
        if self.max_timing_residual_s < 0.0 or not math.isfinite(
            self.max_timing_residual_s
        ):
            raise SchemaValidationError(
                "LearnedHighLevelPolicyConfig.max_timing_residual_s must be finite and non-negative"
            )


@dataclass(frozen=True)
class HighLevelPolicyScores:
    candidate_scores: dict[int, float]
    group_scores: dict[str, float]
    timing_residual_s: float = 0.0


@dataclass(frozen=True)
class HighLevelPolicyDecision:
    used_fallback: bool
    fallback_reason: str | None
    selected_candidate_ids: tuple[int, ...]
    selected_group_id: str | None
    timing_residual_s: float
    assignment_feasible: bool


class HighLevelScoreProvider(Protocol):
    def predict(self, encoding: ContactCandidateEncoderOutput) -> HighLevelPolicyScores:
        ...


class P4_3HighLevelRanker(nn.Module):
    """Tiny candidate/group ranker used by the minimum P4.3c bootstrap."""

    def __init__(
        self,
        *,
        d_model: int = DEFAULT_CONTACT_CANDIDATE_D_MODEL,
        hidden_dim: int = 32,
        max_timing_residual_s: float = 0.05,
    ) -> None:
        super().__init__()
        if d_model <= 0 or hidden_dim <= 0:
            raise ValueError("P4_3HighLevelRanker dimensions must be positive")
        if max_timing_residual_s < 0.0 or not math.isfinite(max_timing_residual_s):
            raise ValueError(
                "P4_3HighLevelRanker.max_timing_residual_s must be finite and non-negative"
            )
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.max_timing_residual_s = max_timing_residual_s
        self.candidate_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.group_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.timing_head = nn.Sequential(
            nn.Linear(2 * d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        candidate_features: torch.Tensor,
        group_features: torch.Tensor,
        *,
        candidate_mask: torch.Tensor | None = None,
        group_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if candidate_features.ndim != 2 or candidate_features.shape[1] != self.d_model:
            raise ValueError(
                "candidate_features must have shape [num_candidates, d_model]"
            )
        if group_features.ndim != 2 or group_features.shape[1] != self.d_model:
            raise ValueError("group_features must have shape [num_groups, d_model]")
        candidate_logits = self.candidate_head(candidate_features).squeeze(-1)
        group_logits = self.group_head(group_features).squeeze(-1)
        if candidate_mask is None:
            candidate_mask = torch.ones(
                candidate_features.shape[0],
                dtype=torch.bool,
                device=candidate_features.device,
            )
        if group_mask is None:
            group_mask = torch.ones(
                group_features.shape[0],
                dtype=torch.bool,
                device=group_features.device,
            )
        candidate_pool = _masked_mean(candidate_features, candidate_mask)
        group_pool = _masked_mean(group_features, group_mask)
        timing_residual = self.timing_head(
            torch.cat((candidate_pool, group_pool), dim=0)
        ).squeeze(-1)
        timing_residual = timing_residual * self.max_timing_residual_s
        return candidate_logits, group_logits, timing_residual

    def predict(self, encoding: ContactCandidateEncoderOutput) -> HighLevelPolicyScores:
        device = next(self.parameters()).device
        candidate_features = torch.tensor(
            encoding.candidate_tokens(), dtype=torch.float32, device=device
        )
        group_rows = encoding.group_tokens()
        group_features = torch.tensor(
            group_rows,
            dtype=torch.float32,
            device=device,
        ).reshape((-1, self.d_model))
        candidate_mask = torch.tensor(
            encoding.candidate_valid_mask(), dtype=torch.bool, device=device
        )
        group_mask = torch.tensor(
            encoding.group_valid_mask(), dtype=torch.bool, device=device
        )
        self.eval()
        with torch.no_grad():
            candidate_logits, group_logits, timing_residual = self(
                candidate_features,
                group_features,
                candidate_mask=candidate_mask,
                group_mask=group_mask,
            )
        candidate_count = encoding.candidate_counts[0]
        group_count = encoding.group_counts[0]
        candidate_scores = {
            encoding.candidate_ids[0][index]: float(candidate_logits[index].item())
            for index in range(candidate_count)
            if candidate_mask[index].item()
        }
        group_scores = {
            str(encoding.group_ids[0][index]): float(group_logits[index].item())
            for index in range(group_count)
            if group_mask[index].item()
        }
        return HighLevelPolicyScores(
            candidate_scores=candidate_scores,
            group_scores=group_scores,
            timing_residual_s=float(timing_residual.item()),
        )


class LearnedHighLevelPolicy:
    """Safe learned pi_H selector with a deterministic P4.2 planner fallback.

    The learned model only ranks existing candidate/group IDs and may shift
    interior knot times by a bounded residual. Deterministic code constructs and
    validates the final ``ContactWrenchTrajectory``. It never emits controller or
    actuator commands.
    """

    def __init__(
        self,
        score_provider: HighLevelScoreProvider,
        *,
        config: LearnedHighLevelPolicyConfig | None = None,
        encoder: ContactCandidateEncoder | None = None,
        fallback_planner: P4_2DeterministicGraspCarryPlanner | None = None,
    ) -> None:
        self.config = config or LearnedHighLevelPolicyConfig()
        self.config.validate()
        self.score_provider = score_provider
        self.encoder = encoder or ContactCandidateEncoder(
            d_model=self.config.encoder_d_model
        )
        self.fallback_planner = fallback_planner or P4_2DeterministicGraspCarryPlanner()
        self.last_decision: HighLevelPolicyDecision | None = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        fallback_planner: P4_2DeterministicGraspCarryPlanner | None = None,
    ) -> "LearnedHighLevelPolicy":
        checkpoint = _load_torch_checkpoint(checkpoint_path)
        if checkpoint.get("model_type") != "P4_3HighLevelRanker":
            raise ValueError("checkpoint is not a P4_3HighLevelRanker")
        config = LearnedHighLevelPolicyConfig(**checkpoint["policy_config"])
        config.validate()
        model = P4_3HighLevelRanker(
            d_model=config.encoder_d_model,
            hidden_dim=config.hidden_dim,
            max_timing_residual_s=config.max_timing_residual_s,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return cls(
            model,
            config=config,
            fallback_planner=fallback_planner,
        )

    def plan(self, context: HighLevelPolicyContext) -> ContactWrenchTrajectory:
        _validate_high_level_context(context)
        try:
            encoding = self.encoder.encode(context.contact_candidate_set)
            scores = self.score_provider.predict(encoding)
            selected_ids, selected_group_id = self._select_candidate_ids(
                context.contact_candidate_set,
                encoding,
                scores,
                context=context,
            )
            timing_residual = self._validated_timing_residual(scores.timing_residual_s)
            feasibility = evaluate_selected_candidate_ids(
                context,
                selected_ids,
                update_cache=True,
            )
            if not feasibility.feasible:
                raise SchemaValidationError(
                    "learned pi_H selection failed deterministic assignment feasibility: "
                    + ",".join(feasibility.violation_codes)
                )
            trajectory = self._decode_trajectory(
                context,
                selected_ids=selected_ids,
                selected_group_id=selected_group_id,
                timing_residual_s=timing_residual,
            )
            _validate_decoded_trajectory(
                trajectory,
                context.contact_candidate_set,
                expected_candidate_ids=set(selected_ids),
            )
            self.last_decision = HighLevelPolicyDecision(
                used_fallback=False,
                fallback_reason=None,
                selected_candidate_ids=tuple(sorted(selected_ids)),
                selected_group_id=selected_group_id,
                timing_residual_s=timing_residual,
                assignment_feasible=True,
            )
            return trajectory
        except (SchemaValidationError, ValueError, KeyError, RuntimeError) as exc:
            return self._fallback(context, reason=f"{type(exc).__name__}: {exc}")

    def _select_candidate_ids(
        self,
        candidate_set: ContactCandidateSet,
        encoding: ContactCandidateEncoderOutput,
        scores: HighLevelPolicyScores,
        *,
        context: HighLevelPolicyContext,
    ) -> tuple[list[int], str | None]:
        valid_candidate_ids = {
            encoding.candidate_ids[0][index]
            for index, is_valid in enumerate(encoding.candidate_valid_mask())
            if is_valid
        }
        valid_group_ids = {
            str(encoding.group_ids[0][index])
            for index, is_valid in enumerate(encoding.group_valid_mask())
            if is_valid
        }
        candidate_score_ids = set(scores.candidate_scores)
        group_score_ids = set(scores.group_scores)
        unknown_candidates = candidate_score_ids - valid_candidate_ids
        unknown_groups = group_score_ids - valid_group_ids
        if unknown_candidates or unknown_groups:
            raise SchemaValidationError(
                "learned pi_H output referenced unknown or invalid IDs: "
                f"candidates={sorted(unknown_candidates)}, groups={sorted(unknown_groups)}"
            )
        if candidate_score_ids != valid_candidate_ids:
            missing = sorted(valid_candidate_ids - candidate_score_ids)
            raise SchemaValidationError(
                f"learned pi_H output omitted valid candidate scores: {missing}"
            )
        if group_score_ids != valid_group_ids:
            missing = sorted(valid_group_ids - group_score_ids)
            raise SchemaValidationError(
                f"learned pi_H output omitted valid group scores: {missing}"
            )
        if not all(math.isfinite(value) for value in scores.candidate_scores.values()):
            raise SchemaValidationError("learned pi_H candidate scores must be finite")
        if not all(math.isfinite(value) for value in scores.group_scores.values()):
            raise SchemaValidationError("learned pi_H group scores must be finite")
        if not valid_candidate_ids:
            raise SchemaValidationError("learned pi_H has no valid candidate to select")

        group_by_id = {
            proposal.group_id: proposal
            for proposal in candidate_set.group_proposals
            if proposal.group_id in valid_group_ids
        }
        if group_by_id:
            def combined_group_score(group_id: str) -> tuple[float, str]:
                proposal = group_by_id[group_id]
                member_score = sum(
                    scores.candidate_scores[candidate_id]
                    for candidate_id in proposal.candidate_ids
                ) / float(len(proposal.candidate_ids))
                return scores.group_scores[group_id] + member_score, group_id

            selected_group_id = max(
                sorted(group_by_id), key=combined_group_score
            )
            return list(group_by_id[selected_group_id].candidate_ids), selected_group_id

        return (
            _greedy_candidate_selection(
                context,
                scores.candidate_scores,
                valid_candidate_ids=valid_candidate_ids,
            ),
            None,
        )

    def _validated_timing_residual(self, value: float) -> float:
        if not self.config.timing_residual_enabled:
            return 0.0
        if not math.isfinite(value):
            raise SchemaValidationError("learned pi_H timing residual must be finite")
        if abs(value) > self.config.max_timing_residual_s + 1.0e-9:
            raise SchemaValidationError(
                "learned pi_H timing residual exceeds the configured bound"
            )
        return float(value)

    def _decode_trajectory(
        self,
        context: HighLevelPolicyContext,
        *,
        selected_ids: list[int],
        selected_group_id: str | None,
        timing_residual_s: float,
    ) -> ContactWrenchTrajectory:
        source_group = next(
            (
                group
                for group in context.contact_candidate_set.group_proposals
                if group.group_id == selected_group_id
            ),
            None,
        )
        selected_set = _candidate_set_for_selected_ids(
            context.contact_candidate_set,
            selected_ids,
            source_group=source_group,
        )
        selected_context = HighLevelPolicyContext(
            irg=context.irg,
            interaction_envelope=context.interaction_envelope,
            morphology_graph=context.morphology_graph,
            contact_candidate_set=selected_set,
            runtime_observation=context.runtime_observation,
        )
        trajectory = self.fallback_planner.plan(selected_context)
        decoded = ContactWrenchTrajectory.from_dict(trajectory.to_dict())
        if timing_residual_s != 0.0 and len(decoded.knots) > 2:
            maximum_safe_residual = _maximum_safe_timing_residual(decoded)
            if abs(timing_residual_s) > maximum_safe_residual + 1.0e-9:
                raise SchemaValidationError(
                    "learned pi_H timing residual would violate monotonic knot timing"
                )
            for knot in decoded.knots[1:-1]:
                knot.t_rel_s += timing_residual_s
        decoded.derived_mode_label = "p4_3_learned_pi_h"
        for knot in decoded.knots:
            knot.priority_weights.pop("p4_2_deterministic", None)
            knot.priority_weights["p4_3_learned_pi_h"] = 1.0
            knot.guard_conditions.append(
                {
                    "type": "p4_3_learned_pi_h",
                    "selection_source": "learned_ranker_deterministic_decoder",
                }
            )
        return ContactWrenchTrajectory.from_dict(decoded.to_dict())

    def _fallback(
        self,
        context: HighLevelPolicyContext,
        *,
        reason: str,
    ) -> ContactWrenchTrajectory:
        trajectory = self.fallback_planner.plan(context)
        _validate_decoded_trajectory(
            trajectory,
            context.contact_candidate_set,
            expected_candidate_ids=None,
        )
        selected_ids = sorted(
            {
                assignment.candidate_id
                for knot in trajectory.knots
                for assignment in knot.contact_assignments
            }
        )
        feasibility = evaluate_selected_candidate_ids(
            context,
            selected_ids,
            update_cache=True,
        )
        if not feasibility.feasible:
            raise SchemaValidationError(
                "deterministic pi_H fallback failed assignment feasibility"
            )
        self.last_decision = HighLevelPolicyDecision(
            used_fallback=True,
            fallback_reason=reason,
            selected_candidate_ids=tuple(selected_ids),
            selected_group_id=None,
            timing_residual_s=0.0,
            assignment_feasible=True,
        )
        return trajectory


def policy_config_dict(config: LearnedHighLevelPolicyConfig) -> dict[str, Any]:
    return asdict(config)


def slot_count_requirements(
    context: HighLevelPolicyContext,
) -> tuple[dict[int, int], dict[int, int]]:
    minimums: dict[int, int] = {}
    maximums: dict[int, int] = {}
    for node in context.irg.nodes:
        if node.node_type != IRGNodeType.CONTACT_SLOT:
            continue
        slot_id = int(node.feature.get("slot_id", node.node_id))
        if node.feature.get("required", True):
            minimums[slot_id] = int(node.feature.get("min_count_group", 1))
        maximums[slot_id] = int(node.feature.get("max_count_group", 1))
    return minimums, maximums


def _validate_high_level_context(context: HighLevelPolicyContext) -> None:
    for value in (
        context.irg,
        context.interaction_envelope,
        context.morphology_graph,
        context.contact_candidate_set,
    ):
        type(value).from_dict(value.to_dict())
    if context.runtime_observation is not None:
        type(context.runtime_observation).from_dict(context.runtime_observation.to_dict())
    if context.contact_candidate_set.task_id != context.irg.task_id:
        raise SchemaValidationError(
            "HighLevelPolicyContext candidate set and IRG task IDs must match"
        )
    if context.interaction_envelope.task_id != context.irg.task_id:
        raise SchemaValidationError(
            "HighLevelPolicyContext envelope and IRG task IDs must match"
        )
    if (
        context.contact_candidate_set.morphology_graph_id
        != context.morphology_graph.graph_id
    ):
        raise SchemaValidationError(
            "HighLevelPolicyContext candidate set must reference the morphology graph"
        )


def evaluate_selected_candidate_ids(
    context: HighLevelPolicyContext,
    candidate_ids: list[int],
    *,
    update_cache: bool,
):
    candidate_by_id = {
        candidate.candidate_id: candidate
        for candidate in context.contact_candidate_set.candidates
    }
    if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
        raise SchemaValidationError(
            "learned pi_H selected candidate IDs must be non-empty and unique"
        )
    unknown = sorted(set(candidate_ids) - set(candidate_by_id))
    if unknown:
        raise SchemaValidationError(
            f"learned pi_H selected unknown candidate IDs: {unknown}"
        )
    assignments = [
        ContactAssignment(
            slot_id=candidate_by_id[candidate_id].slot_id,
            anchor_id=candidate_by_id[candidate_id].anchor_id,
            candidate_id=candidate_id,
            contact_mode=candidate_by_id[candidate_id].contact_mode,
            schedule_state="maintain",
        )
        for candidate_id in candidate_ids
    ]
    minimums, maximums = slot_count_requirements(context)
    return evaluate_selected_assignment_feasibility(
        assignments,
        context.contact_candidate_set,
        slot_min_counts=minimums,
        slot_max_counts=maximums,
        update_cache=update_cache,
    )


def _greedy_candidate_selection(
    context: HighLevelPolicyContext,
    scores: dict[int, float],
    *,
    valid_candidate_ids: set[int],
) -> list[int]:
    candidate_by_id = {
        candidate.candidate_id: candidate
        for candidate in context.contact_candidate_set.candidates
    }
    minimums, maximums = slot_count_requirements(context)
    if not minimums:
        required_total = max(1, context.interaction_envelope.required_contact_count_range[0])
        slot_ids = sorted(
            {candidate_by_id[candidate_id].slot_id for candidate_id in valid_candidate_ids}
        )
        if slot_ids:
            minimums[slot_ids[0]] = required_total
            maximums.setdefault(
                slot_ids[0],
                max(required_total, context.interaction_envelope.required_contact_count_range[1]),
            )

    selected: list[int] = []
    for slot_id, minimum_count in sorted(minimums.items()):
        ranked = sorted(
            (
                candidate_id
                for candidate_id in valid_candidate_ids
                if candidate_by_id[candidate_id].slot_id == slot_id
            ),
            key=lambda candidate_id: (-scores[candidate_id], candidate_id),
        )
        slot_limit = maximums.get(slot_id, minimum_count)
        for candidate_id in ranked:
            if len(
                [
                    selected_id
                    for selected_id in selected
                    if candidate_by_id[selected_id].slot_id == slot_id
                ]
            ) >= min(minimum_count, slot_limit):
                break
            selected.append(candidate_id)
        selected_in_slot = sum(
            candidate_by_id[candidate_id].slot_id == slot_id
            for candidate_id in selected
        )
        if selected_in_slot < minimum_count:
            raise SchemaValidationError(
                f"learned pi_H candidate ranking cannot satisfy slot {slot_id} cardinality"
            )
    if not selected:
        selected.append(
            min(valid_candidate_ids, key=lambda candidate_id: (-scores[candidate_id], candidate_id))
        )
    return selected


def _candidate_set_for_selected_ids(
    candidate_set: ContactCandidateSet,
    selected_ids: list[int],
    *,
    source_group: ContactCandidateGroupProposal | None,
) -> ContactCandidateSet:
    data = copy.deepcopy(candidate_set.to_dict())
    if source_group is None:
        selected_modes = {
            candidate.contact_mode.value
            for candidate in candidate_set.candidates
            if candidate.candidate_id in set(selected_ids)
        }
        if selected_modes == {"support"}:
            group_type = "support_set"
        elif selected_modes == {"perch"}:
            group_type = "perch_set"
        else:
            group_type = "grasp_pair" if len(selected_ids) <= 2 else "multi_grasp"
        group_id = "p4_3_learned_candidate_selection"
    else:
        group_type = source_group.group_type
        group_id = source_group.group_id
    data["group_proposals"] = [
        {
            "group_id": group_id,
            "candidate_ids": list(selected_ids),
            "group_type": group_type,
            "group_score": 1.0,
            "group_violation_codes": [],
        }
    ]
    data["assignment_feasibility_cache"] = {}
    return ContactCandidateSet.from_dict(data)


def _maximum_safe_timing_residual(trajectory: ContactWrenchTrajectory) -> float:
    gaps = [
        right.t_rel_s - left.t_rel_s
        for left, right in zip(trajectory.knots, trajectory.knots[1:])
    ]
    if not gaps or min(gaps) <= 0.0:
        return 0.0
    return 0.45 * min(gaps)


def _validate_decoded_trajectory(
    trajectory: ContactWrenchTrajectory,
    candidate_set: ContactCandidateSet,
    *,
    expected_candidate_ids: set[int] | None,
) -> None:
    validated = ContactWrenchTrajectory.from_dict(trajectory.to_dict())
    if not validated.knots:
        raise SchemaValidationError("decoded pi_H trajectory must contain knots")
    times = [knot.t_rel_s for knot in validated.knots]
    if not all(math.isfinite(value) for value in times):
        raise SchemaValidationError("decoded pi_H knot times must be finite")
    if times[0] < 0.0 or any(right <= left for left, right in zip(times, times[1:])):
        raise SchemaValidationError("decoded pi_H knot times must be strictly increasing")
    if times[-1] > validated.horizon_s + 1.0e-9:
        raise SchemaValidationError("decoded pi_H knot exceeds trajectory horizon")
    _require_finite_payload(validated.to_dict(), "ContactWrenchTrajectory")
    candidate_by_id = {
        candidate.candidate_id: candidate for candidate in candidate_set.candidates
    }
    output_ids: set[int] = set()
    for knot in validated.knots:
        for assignment in knot.contact_assignments:
            candidate = candidate_by_id.get(assignment.candidate_id)
            if candidate is None:
                raise SchemaValidationError(
                    "decoded pi_H trajectory references an unknown candidate ID"
                )
            if (
                assignment.slot_id != candidate.slot_id
                or assignment.anchor_id != candidate.anchor_id
                or assignment.contact_mode != candidate.contact_mode
            ):
                raise SchemaValidationError(
                    "decoded pi_H trajectory changed source slot/anchor/contact-mode IDs"
                )
            output_ids.add(assignment.candidate_id)
    if expected_candidate_ids is not None and output_ids != expected_candidate_ids:
        raise SchemaValidationError(
            "decoded pi_H trajectory did not preserve the selected source candidate IDs"
        )


def _require_finite_payload(value: Any, path: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise SchemaValidationError(f"{path} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_payload(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_payload(item, f"{path}.{key}")


def _masked_mean(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if features.shape[0] == 0 or not bool(mask.any().item()):
        return torch.zeros(
            features.shape[1], dtype=features.dtype, device=features.device
        )
    return features[mask].mean(dim=0)


def _load_torch_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(Path(path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("pi_H checkpoint payload must be a mapping")
    return checkpoint
