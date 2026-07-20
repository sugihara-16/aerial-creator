from __future__ import annotations

r"""Full-trajectory learned :math:`\pi_H` used by the Order 9 curriculum.

The policy is deliberately a proposal generator.  It does not call a teacher,
feasibility checker, deterministic planner, or fallback.  The separate Order 9
runtime passes the returned trajectory to ``C_H`` unchanged.
"""

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical, Normal

from amsrr.encoders.interaction_envelope_encoder import InteractionEnvelopeEncoder
from amsrr.encoders.morphology_graph_encoder import MorphologyGraphEncoder
from amsrr.geometry.pose_math import compose_pose
from amsrr.policies.contact_candidate_encoder import (
    DEFAULT_CONTACT_CANDIDATE_D_MODEL,
    ContactCandidateEncoder,
)
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.schemas.common import Pose7D, SchemaValidationError
from amsrr.schemas.policies import (
    CONTACT_WRENCH_CONTRACT_CONTACT_FRAME,
    CentroidalTarget,
    ContactAssignment,
    ContactWrenchTrajectory,
    InteractionKnot,
    ObjectTarget,
    PostureTarget,
)


ORDER9_FULL_PI_H_VERSION = "order9_autoregressive_full_trajectory_pi_h_v1"
SCHEDULE_STATES = ("approach", "attach", "maintain", "slide", "release")
PRIORITY_KEYS = ("contact", "centroidal", "posture", "object", "safety")
GUARD_TYPES = (
    "elapsed_fraction",
    "pose_error_below",
    "velocity_below",
    "contact_estimate_valid",
    "controller_feasible",
    "task_phase_complete",
)


@dataclass(frozen=True)
class Order9HighLevelPolicyConfig:
    d_model: int = 96
    candidate_d_model: int = DEFAULT_CONTACT_CANDIDATE_D_MODEL
    envelope_d_model: int = 32
    runtime_feature_dim: int = 20
    object_feature_dim: int = 14
    num_knots: int = 9
    horizon_s: float = 2.0
    max_com_offset_m: float = 0.75
    max_com_velocity_mps: float = 1.5
    max_anchor_offset_m: float = 0.20
    max_object_offset_m: float = 1.0
    max_object_twist: float = 2.0
    force_limit_n: float = 30.0
    torque_limit_nm: float = 5.0
    minimum_interval_fraction: float = 1.0e-3
    continuous_log_std_init: float = -1.5

    def validate(self) -> None:
        integer_fields = (
            "d_model",
            "candidate_d_model",
            "envelope_d_model",
            "runtime_feature_dim",
            "object_feature_dim",
            "num_knots",
        )
        if any(getattr(self, name) <= 0 for name in integer_fields):
            raise ValueError("Order 9 pi_H dimensions must be positive")
        if self.num_knots < 2:
            raise ValueError("Order 9 pi_H requires at least two knots")
        positive_fields = (
            "horizon_s",
            "max_com_offset_m",
            "max_com_velocity_mps",
            "max_anchor_offset_m",
            "max_object_offset_m",
            "max_object_twist",
            "force_limit_n",
            "torque_limit_nm",
            "minimum_interval_fraction",
        )
        if any(
            not math.isfinite(float(getattr(self, name)))
            or float(getattr(self, name)) <= 0.0
            for name in positive_fields
        ):
            raise ValueError("Order 9 pi_H limits must be finite and positive")
        if not math.isfinite(self.continuous_log_std_init):
            raise ValueError("continuous_log_std_init must be finite")


@dataclass
class Order9PiHNetworkOutput:
    assignment_logits: torch.Tensor
    schedule_logits: torch.Tensor
    wrench_raw_mean: torch.Tensor
    anchor_pose_raw_mean: torch.Tensor
    knot_target_raw_mean: torch.Tensor
    priority_raw_mean: torch.Tensor
    guard_logits: torch.Tensor
    guard_threshold_raw_mean: torch.Tensor
    interval_raw_mean: torch.Tensor
    object_target_raw_mean: torch.Tensor
    candidate_mask: torch.Tensor
    object_mask: torch.Tensor
    value: torch.Tensor


@dataclass
class Order9PiHAction:
    assignment_active: torch.Tensor
    schedule_index: torch.Tensor
    wrench_raw: torch.Tensor
    anchor_pose_raw: torch.Tensor
    knot_target_raw: torch.Tensor
    priority_raw: torch.Tensor
    guard_active: torch.Tensor
    guard_threshold_raw: torch.Tensor
    interval_raw: torch.Tensor
    object_target_raw: torch.Tensor


@dataclass
class Order9PiHActionEvaluation:
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


class Order9AutoregressiveHighLevelPolicy(nn.Module):
    """Morphology-conditioned actor-critic that emits the complete pi_H schema."""

    def __init__(self, config: Order9HighLevelPolicyConfig | None = None) -> None:
        super().__init__()
        self.config = config or Order9HighLevelPolicyConfig()
        self.config.validate()
        cfg = self.config
        self.candidate_encoder = ContactCandidateEncoder(d_model=cfg.candidate_d_model)
        self.envelope_encoder = InteractionEnvelopeEncoder(d_model=cfg.envelope_d_model)
        self.morphology_encoder = MorphologyGraphEncoder(d_model=cfg.d_model)
        self.candidate_projection = nn.Sequential(
            nn.Linear(cfg.candidate_d_model, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.SiLU(),
        )
        self.envelope_projection = nn.Sequential(
            nn.Linear(cfg.envelope_d_model, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.SiLU(),
        )
        self.runtime_projection = nn.Sequential(
            nn.Linear(cfg.runtime_feature_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.SiLU(),
        )
        self.object_projection = nn.Sequential(
            nn.Linear(cfg.object_feature_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.SiLU(),
        )
        self.context_fusion = nn.Sequential(
            nn.Linear(3 * cfg.d_model, 2 * cfg.d_model),
            nn.SiLU(),
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )
        self.knot_embeddings = nn.Parameter(torch.empty(cfg.num_knots, cfg.d_model))
        nn.init.normal_(self.knot_embeddings, std=0.02)
        self.start_summary = nn.Parameter(torch.zeros(cfg.d_model))
        self.autoregressive_cell = nn.GRUCell(2 * cfg.d_model, cfg.d_model)
        self.candidate_query = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        pair_width = 2 * cfg.d_model
        self.assignment_head = _head(pair_width, cfg.d_model, 1)
        self.schedule_head = _head(pair_width, cfg.d_model, len(SCHEDULE_STATES))
        self.wrench_head = _head(pair_width, cfg.d_model, 18)
        self.anchor_pose_head = _head(pair_width, cfg.d_model, 7)
        self.knot_target_head = _head(cfg.d_model, cfg.d_model, 10)
        self.priority_head = _head(cfg.d_model, cfg.d_model, len(PRIORITY_KEYS))
        self.guard_head = _head(cfg.d_model, cfg.d_model, len(GUARD_TYPES))
        self.guard_threshold_head = _head(
            cfg.d_model, cfg.d_model, len(GUARD_TYPES)
        )
        self.interval_head = _head(cfg.d_model, cfg.d_model, 1)
        self.object_target_head = _head(pair_width, cfg.d_model, 13)
        self.value_head = _head(cfg.d_model, cfg.d_model, 1)
        init = cfg.continuous_log_std_init
        self.wrench_log_std = nn.Parameter(torch.full((18,), init))
        self.anchor_pose_log_std = nn.Parameter(torch.full((7,), init))
        self.knot_target_log_std = nn.Parameter(torch.full((10,), init))
        self.priority_log_std = nn.Parameter(torch.full((len(PRIORITY_KEYS),), init))
        self.guard_threshold_log_std = nn.Parameter(
            torch.full((len(GUARD_TYPES),), init)
        )
        self.interval_log_std = nn.Parameter(torch.full((1,), init))
        self.object_target_log_std = nn.Parameter(torch.full((13,), init))

    def forward_contexts(
        self,
        contexts: Sequence[HighLevelPolicyContext],
        *,
        teacher_assignment_mask: torch.Tensor | None = None,
    ) -> Order9PiHNetworkOutput:
        if not contexts:
            raise ValueError("Order 9 pi_H requires at least one context")
        device = self.knot_embeddings.device
        dtype = self.knot_embeddings.dtype
        morphologies = [context.morphology_graph for context in contexts]
        observations = [context.runtime_observation for context in contexts]
        morphology = self.morphology_encoder(
            morphologies,
            runtime_observations=observations,
        ).graph_embeddings
        candidate_raw, candidate_mask = self._candidate_batch(contexts, device, dtype)
        candidates = self.candidate_projection(candidate_raw)
        envelope_raw, envelope_mask = self._envelope_batch(contexts, device, dtype)
        envelope_tokens = self.envelope_projection(envelope_raw)
        envelope = _masked_mean(envelope_tokens, envelope_mask)
        runtime = self.runtime_projection(
            torch.tensor(
                [_runtime_features(context, self.config.runtime_feature_dim) for context in contexts],
                dtype=dtype,
                device=device,
            )
        )
        context_embedding = self.context_fusion(
            torch.cat((morphology, envelope, runtime), dim=-1)
        )
        object_raw, object_mask = self._object_batch(contexts, device, dtype)
        objects = self.object_projection(object_raw)

        if teacher_assignment_mask is not None:
            expected = (
                len(contexts),
                self.config.num_knots,
                candidates.shape[1],
            )
            if tuple(teacher_assignment_mask.shape) != expected:
                raise ValueError(
                    f"teacher_assignment_mask must have shape {expected}"
                )
            teacher_assignment_mask = teacher_assignment_mask.to(
                device=device, dtype=dtype
            )

        hidden = context_embedding
        previous_summary = self.start_summary.expand(len(contexts), -1)
        assignment_logits: list[torch.Tensor] = []
        schedule_logits: list[torch.Tensor] = []
        wrench_means: list[torch.Tensor] = []
        anchor_means: list[torch.Tensor] = []
        knot_means: list[torch.Tensor] = []
        priority_means: list[torch.Tensor] = []
        guard_logits: list[torch.Tensor] = []
        guard_thresholds: list[torch.Tensor] = []
        interval_means: list[torch.Tensor] = []
        object_means: list[torch.Tensor] = []
        hidden_states: list[torch.Tensor] = []
        scale = math.sqrt(float(self.config.d_model))

        for knot_index in range(self.config.num_knots):
            knot_embedding = self.knot_embeddings[knot_index].expand(len(contexts), -1)
            hidden = self.autoregressive_cell(
                torch.cat((knot_embedding, previous_summary), dim=-1), hidden
            )
            hidden_states.append(hidden)
            hidden_expanded = hidden.unsqueeze(1).expand(-1, candidates.shape[1], -1)
            pair = torch.cat((hidden_expanded, candidates), dim=-1)
            logits = self.assignment_head(pair).squeeze(-1)
            logits = logits + (
                self.candidate_query(hidden).unsqueeze(1) * candidates
            ).sum(dim=-1) / scale
            logits = logits.masked_fill(~candidate_mask, -30.0)
            assignment_logits.append(logits)
            schedule_logits.append(self.schedule_head(pair))
            wrench_means.append(self.wrench_head(pair))
            anchor_means.append(self.anchor_pose_head(pair))
            knot_means.append(self.knot_target_head(hidden))
            priority_means.append(self.priority_head(hidden))
            guard_logits.append(self.guard_head(hidden))
            guard_thresholds.append(self.guard_threshold_head(hidden))
            interval_means.append(self.interval_head(hidden))
            object_hidden = hidden.unsqueeze(1).expand(-1, objects.shape[1], -1)
            object_means.append(
                self.object_target_head(torch.cat((object_hidden, objects), dim=-1))
            )
            if teacher_assignment_mask is None:
                weights = torch.sigmoid(logits) * candidate_mask.to(dtype)
            else:
                weights = teacher_assignment_mask[:, knot_index] * candidate_mask.to(dtype)
            denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            selected_summary = (candidates * weights.unsqueeze(-1)).sum(dim=1) / denominator
            previous_summary = selected_summary

        final_hidden = torch.stack(hidden_states, dim=1).mean(dim=1)
        return Order9PiHNetworkOutput(
            assignment_logits=torch.stack(assignment_logits, dim=1),
            schedule_logits=torch.stack(schedule_logits, dim=1),
            wrench_raw_mean=torch.stack(wrench_means, dim=1),
            anchor_pose_raw_mean=torch.stack(anchor_means, dim=1),
            knot_target_raw_mean=torch.stack(knot_means, dim=1),
            priority_raw_mean=torch.stack(priority_means, dim=1),
            guard_logits=torch.stack(guard_logits, dim=1),
            guard_threshold_raw_mean=torch.stack(guard_thresholds, dim=1),
            interval_raw_mean=torch.stack(interval_means, dim=1).squeeze(-1),
            object_target_raw_mean=torch.stack(object_means, dim=1),
            candidate_mask=candidate_mask,
            object_mask=object_mask,
            value=self.value_head(final_hidden).squeeze(-1),
        )

    def sample_action(
        self,
        output: Order9PiHNetworkOutput,
        *,
        deterministic: bool = False,
    ) -> Order9PiHAction:
        assignment = Bernoulli(logits=output.assignment_logits)
        schedule = Categorical(logits=output.schedule_logits)
        guard = Bernoulli(logits=output.guard_logits)
        choose = lambda distribution: distribution.mean if deterministic else distribution.sample()
        assignment_active = choose(assignment).bool() & output.candidate_mask.unsqueeze(1)
        schedule_index = (
            output.schedule_logits.argmax(dim=-1)
            if deterministic
            else schedule.sample()
        )
        guard_active = choose(guard).bool()
        return Order9PiHAction(
            assignment_active=assignment_active,
            schedule_index=schedule_index,
            wrench_raw=self._normal_action(
                output.wrench_raw_mean, self.wrench_log_std, deterministic
            ),
            anchor_pose_raw=self._normal_action(
                output.anchor_pose_raw_mean, self.anchor_pose_log_std, deterministic
            ),
            knot_target_raw=self._normal_action(
                output.knot_target_raw_mean, self.knot_target_log_std, deterministic
            ),
            priority_raw=self._normal_action(
                output.priority_raw_mean, self.priority_log_std, deterministic
            ),
            guard_active=guard_active,
            guard_threshold_raw=self._normal_action(
                output.guard_threshold_raw_mean,
                self.guard_threshold_log_std,
                deterministic,
            ),
            interval_raw=self._normal_action(
                output.interval_raw_mean.unsqueeze(-1),
                self.interval_log_std,
                deterministic,
            ).squeeze(-1),
            object_target_raw=self._normal_action(
                output.object_target_raw_mean,
                self.object_target_log_std,
                deterministic,
            ),
        )

    def evaluate_action(
        self,
        output: Order9PiHNetworkOutput,
        action: Order9PiHAction,
    ) -> Order9PiHActionEvaluation:
        candidate_mask = output.candidate_mask.unsqueeze(1).to(output.assignment_logits.dtype)
        object_mask = output.object_mask[:, None, :, None].to(
            output.assignment_logits.dtype
        )
        selected_mask = action.assignment_active.to(output.assignment_logits.dtype)
        assignment_dist = Bernoulli(logits=output.assignment_logits)
        schedule_dist = Categorical(logits=output.schedule_logits)
        guard_dist = Bernoulli(logits=output.guard_logits)
        log_prob = (
            assignment_dist.log_prob(action.assignment_active.to(output.assignment_logits.dtype))
            * candidate_mask
        ).sum(dim=(1, 2))
        log_prob = log_prob + (
            schedule_dist.log_prob(action.schedule_index) * selected_mask
        ).sum(dim=(1, 2))
        entropy = (assignment_dist.entropy() * candidate_mask).sum(dim=(1, 2))
        entropy = entropy + (schedule_dist.entropy() * selected_mask).sum(dim=(1, 2))
        for mean, value, log_std, mask in (
            (output.wrench_raw_mean, action.wrench_raw, self.wrench_log_std, selected_mask.unsqueeze(-1)),
            (output.anchor_pose_raw_mean, action.anchor_pose_raw, self.anchor_pose_log_std, selected_mask.unsqueeze(-1)),
            (output.knot_target_raw_mean, action.knot_target_raw, self.knot_target_log_std, 1.0),
            (output.priority_raw_mean, action.priority_raw, self.priority_log_std, 1.0),
            (output.guard_threshold_raw_mean, action.guard_threshold_raw, self.guard_threshold_log_std, 1.0),
            (output.interval_raw_mean.unsqueeze(-1), action.interval_raw.unsqueeze(-1), self.interval_log_std, 1.0),
            (output.object_target_raw_mean, action.object_target_raw, self.object_target_log_std, object_mask),
        ):
            distribution = Normal(mean, log_std.exp())
            term_log_prob = distribution.log_prob(value) * mask
            term_entropy = distribution.entropy() * mask
            reduce_dims = tuple(range(1, term_log_prob.ndim))
            log_prob = log_prob + term_log_prob.sum(dim=reduce_dims)
            entropy = entropy + term_entropy.sum(dim=reduce_dims)
        log_prob = log_prob + guard_dist.log_prob(
            action.guard_active.to(output.guard_logits.dtype)
        ).sum(dim=(1, 2))
        entropy = entropy + guard_dist.entropy().sum(dim=(1, 2))
        return Order9PiHActionEvaluation(
            log_prob=log_prob,
            entropy=entropy,
            value=output.value,
        )

    def propose(
        self,
        context: HighLevelPolicyContext,
        *,
        deterministic: bool = True,
    ) -> ContactWrenchTrajectory:
        self.eval()
        with torch.no_grad():
            output = self.forward_contexts([context])
            action = self.sample_action(output, deterministic=deterministic)
        return self.decode_action(context, action, batch_index=0)

    def decode_action(
        self,
        context: HighLevelPolicyContext,
        action: Order9PiHAction,
        *,
        batch_index: int,
    ) -> ContactWrenchTrajectory:
        cfg = self.config
        candidates = context.contact_candidate_set.candidates
        if action.assignment_active.shape[-1] < len(candidates):
            raise ValueError("pi_H action candidate width is smaller than the context")
        times = _decode_times(
            action.interval_raw[batch_index],
            horizon_s=cfg.horizon_s,
            minimum_fraction=cfg.minimum_interval_fraction,
        )
        com_pose = _centroidal_reference_pose(context)
        objects = (
            []
            if context.runtime_observation is None
            else context.runtime_observation.object_states
        )
        knots: list[InteractionKnot] = []
        for knot_index in range(cfg.num_knots):
            assignments: list[ContactAssignment] = []
            free_anchor_targets: dict[int, Pose7D] = {}
            for candidate_index, candidate in enumerate(candidates):
                if not bool(
                    action.assignment_active[
                        batch_index, knot_index, candidate_index
                    ].item()
                ):
                    continue
                schedule_index = int(
                    action.schedule_index[
                        batch_index, knot_index, candidate_index
                    ].item()
                )
                schedule_state = SCHEDULE_STATES[schedule_index]
                wrench = _decode_wrench(
                    action.wrench_raw[
                        batch_index, knot_index, candidate_index
                    ],
                    force_limit_n=cfg.force_limit_n,
                    torque_limit_nm=cfg.torque_limit_nm,
                )
                active_wrench = schedule_state in {"attach", "maintain", "slide"}
                assignments.append(
                    ContactAssignment(
                        slot_id=candidate.slot_id,
                        anchor_id=candidate.anchor_id,
                        candidate_id=candidate.candidate_id,
                        contact_mode=candidate.contact_mode,
                        schedule_state=schedule_state,  # type: ignore[arg-type]
                        wrench_target=wrench[0] if active_wrench else None,
                        wrench_lower=wrench[1] if active_wrench else None,
                        wrench_upper=wrench[2] if active_wrench else None,
                        priority=float(
                            torch.nn.functional.softplus(
                                action.priority_raw[batch_index, knot_index, 0]
                            ).item()
                        ),
                        wrench_frame="contact",
                    )
                )
                anchor_raw = action.anchor_pose_raw[
                    batch_index, knot_index, candidate_index
                ]
                anchor_delta = _decode_pose_delta(
                    anchor_raw,
                    max_offset_m=cfg.max_anchor_offset_m,
                )
                free_anchor_targets[candidate.anchor_id] = compose_pose(
                    candidate.contact_pose_world, anchor_delta
                )

            knot_raw = action.knot_target_raw[batch_index, knot_index]
            centroidal_delta = _decode_pose_delta(
                torch.cat((knot_raw[:3], knot_raw[6:10])),
                max_offset_m=cfg.max_com_offset_m,
            )
            centroidal_pose = compose_pose(com_pose, centroidal_delta)
            centroidal_target = CentroidalTarget(
                com_pos_world=tuple(float(value) for value in centroidal_pose[:3]),
                com_vel_world=tuple(
                    float(value)
                    for value in (
                        torch.tanh(knot_raw[3:6]) * cfg.max_com_velocity_mps
                    ).tolist()
                ),
                body_orientation_world=tuple(
                    float(value) for value in centroidal_pose[3:7]
                ),
            )
            object_targets: list[ObjectTarget] = []
            for object_index, object_state in enumerate(objects):
                object_raw = action.object_target_raw[
                    batch_index, knot_index, object_index
                ]
                object_delta = _decode_pose_delta(
                    object_raw[:7], max_offset_m=cfg.max_object_offset_m
                )
                object_pose = compose_pose(object_state.pose_world, object_delta)
                object_targets.append(
                    ObjectTarget(
                        object_id=object_state.object_id,
                        pose_target_world=object_pose,
                        twist_target_world=[
                            float(value)
                            for value in (
                                torch.tanh(object_raw[7:13])
                                * cfg.max_object_twist
                            ).tolist()
                        ],
                    )
                )
            priorities = {
                key: float(
                    torch.nn.functional.softplus(
                        action.priority_raw[batch_index, knot_index, index]
                    ).item()
                )
                for index, key in enumerate(PRIORITY_KEYS)
            }
            guards = []
            for guard_index, guard_type in enumerate(GUARD_TYPES):
                if bool(
                    action.guard_active[batch_index, knot_index, guard_index].item()
                ):
                    guards.append(
                        {
                            "type": guard_type,
                            "threshold": float(
                                torch.sigmoid(
                                    action.guard_threshold_raw[
                                        batch_index, knot_index, guard_index
                                    ]
                                ).item()
                            ),
                            "source": ORDER9_FULL_PI_H_VERSION,
                        }
                    )
            if context.runtime_observation is not None:
                phase_label = context.runtime_observation.task_progress.phase_label
                if phase_label:
                    # A proposal is a receding-horizon plan for the currently
                    # active task phase.  Phase completion triggers a fresh
                    # high-level decision; recording the phase on every knot
                    # lets C_H apply the correct IRG requirements without
                    # guessing future task-state-machine transitions.
                    guards.append(
                        {
                            "type": "order9_task_phase",
                            "phase_label": str(phase_label),
                            "source": "runtime_task_progress_latched_at_proposal",
                        }
                    )
            knots.append(
                InteractionKnot(
                    t_rel_s=times[knot_index],
                    contact_assignments=assignments,
                    centroidal_target=centroidal_target,
                    posture_target=PostureTarget(
                        free_anchor_pose_targets=free_anchor_targets
                    ),
                    object_targets=object_targets,
                    priority_weights=priorities,
                    guard_conditions=guards,
                )
            )
        trajectory = ContactWrenchTrajectory(
            horizon_s=cfg.horizon_s,
            dt_s=cfg.horizon_s / float(cfg.num_knots - 1),
            knots=knots,
            derived_mode_label=ORDER9_FULL_PI_H_VERSION,
            contract_version=CONTACT_WRENCH_CONTRACT_CONTACT_FRAME,
        )
        trajectory.validate()
        return trajectory

    def _candidate_batch(
        self,
        contexts: Sequence[HighLevelPolicyContext],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded: list[list[list[float]]] = []
        masks: list[list[bool]] = []
        maximum = max(1, max(len(item.contact_candidate_set.candidates) for item in contexts))
        for context in contexts:
            candidate_set = context.contact_candidate_set
            if candidate_set.candidates:
                output = self.candidate_encoder.encode(candidate_set)
                rows = output.candidate_tokens()
                valid = output.candidate_valid_mask()
            else:
                rows = []
                valid = []
            rows = [list(row) for row in rows]
            valid = [bool(item) for item in valid]
            while len(rows) < maximum:
                rows.append([0.0] * self.config.candidate_d_model)
                valid.append(False)
            encoded.append(rows)
            masks.append(valid)
        return (
            torch.tensor(encoded, dtype=dtype, device=device),
            torch.tensor(masks, dtype=torch.bool, device=device),
        )

    def _envelope_batch(
        self,
        contexts: Sequence[HighLevelPolicyContext],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.envelope_encoder.encode_batch(
            [context.interaction_envelope for context in contexts]
        )
        return (
            torch.tensor(output.tokens, dtype=dtype, device=device),
            torch.tensor(output.mask, dtype=torch.bool, device=device),
        )

    def _object_batch(
        self,
        contexts: Sequence[HighLevelPolicyContext],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = [
            []
            if context.runtime_observation is None
            else [
                _object_features(item, self.config.object_feature_dim)
                for item in context.runtime_observation.object_states
            ]
            for context in contexts
        ]
        maximum = max(1, max(len(item) for item in rows))
        masks: list[list[bool]] = []
        for item in rows:
            masks.append([True] * len(item) + [False] * (maximum - len(item)))
            item.extend(
                [[0.0] * self.config.object_feature_dim for _ in range(maximum - len(item))]
            )
        return (
            torch.tensor(rows, dtype=dtype, device=device),
            torch.tensor(masks, dtype=torch.bool, device=device),
        )

    @staticmethod
    def _normal_action(
        mean: torch.Tensor,
        log_std: torch.Tensor,
        deterministic: bool,
    ) -> torch.Tensor:
        return mean if deterministic else Normal(mean, log_std.exp()).sample()


def decode_wrench_tensors(
    raw: torch.Tensor,
    *,
    force_limit_n: float,
    torque_limit_nm: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable ordered-bound parameterization used by BC and decoding."""

    limits = raw.new_tensor(
        [force_limit_n] * 3 + [torque_limit_nm] * 3
    )
    center = torch.tanh(raw[..., :6]) * limits
    half_span = torch.sigmoid(raw[..., 6:12]) * limits
    target = center + torch.tanh(raw[..., 12:18]) * half_span
    return target, center - half_span, center + half_span


def _decode_wrench(
    raw: torch.Tensor,
    *,
    force_limit_n: float,
    torque_limit_nm: float,
) -> tuple[list[float], list[float], list[float]]:
    target, lower, upper = decode_wrench_tensors(
        raw,
        force_limit_n=force_limit_n,
        torque_limit_nm=torque_limit_nm,
    )
    return (
        [float(value) for value in target.tolist()],
        [float(value) for value in lower.tolist()],
        [float(value) for value in upper.tolist()],
    )


def _decode_times(
    raw: torch.Tensor,
    *,
    horizon_s: float,
    minimum_fraction: float,
) -> list[float]:
    intervals = torch.nn.functional.softplus(raw[1:]) + minimum_fraction
    normalized = intervals / intervals.sum().clamp_min(1.0e-12)
    times = torch.cat((raw.new_zeros(1), torch.cumsum(normalized, dim=0)))
    times[-1] = 1.0
    return [float(value) * horizon_s for value in times.tolist()]


def _decode_pose_delta(raw: torch.Tensor, *, max_offset_m: float) -> Pose7D:
    if raw.numel() != 7:
        raise ValueError("pose delta requires seven raw values")
    xyz = torch.tanh(raw[:3]) * max_offset_m
    quaternion = raw[3:7]
    norm = torch.linalg.vector_norm(quaternion)
    if float(norm.item()) < 1.0e-8:
        quaternion = quaternion.new_tensor([0.0, 0.0, 0.0, 1.0])
    else:
        quaternion = quaternion / norm
    return tuple(float(value) for value in torch.cat((xyz, quaternion)).tolist())  # type: ignore[return-value]


def _centroidal_reference_pose(context: HighLevelPolicyContext) -> Pose7D:
    observation = context.runtime_observation
    if observation is not None and observation.module_states:
        count = float(len(observation.module_states))
        xyz = tuple(
            sum(float(module.pose_world[axis]) for module in observation.module_states)
            / count
            for axis in range(3)
        )
        quaternion = observation.module_states[0].pose_world[3:7]
        return (*xyz, *quaternion)
    modules = context.morphology_graph.modules
    if not modules:
        raise SchemaValidationError("pi_H context morphology has no modules")
    count = float(len(modules))
    xyz = tuple(
        sum(float(module.pose_in_design_frame[axis]) for module in modules) / count
        for axis in range(3)
    )
    return (*xyz, *modules[0].pose_in_design_frame[3:7])


def _runtime_features(
    context: HighLevelPolicyContext,
    width: int,
) -> list[float]:
    observation = context.runtime_observation
    if observation is None:
        base = [0.0] * 18
    else:
        status_order = ("ok", "warning", "infeasible", "fault")
        mean_linear_speed = _mean_norm(
            [state.twist_world[:3] for state in observation.module_states]
        )
        mean_angular_speed = _mean_norm(
            [state.twist_world[3:6] for state in observation.module_states]
        )
        phase = observation.task_progress.phase_label or "unspecified"
        phase_scalar = _stable_scalar(phase)
        active_contact_count = sum(
            1 for contact in observation.contact_states if contact.active
        )
        base = [
            float(observation.time_s),
            float(observation.task_progress.progress_ratio),
            1.0 if observation.task_progress.success else 0.0,
            float(len(observation.module_states)),
            float(len(observation.object_states)),
            float(active_contact_count),
            mean_linear_speed,
            mean_angular_speed,
            math.sin(2.0 * math.pi * phase_scalar),
            math.cos(2.0 * math.pi * phase_scalar),
            *[
                1.0 if observation.controller_status.status == status else 0.0
                for status in status_order
            ],
            1.0 if observation.controller_status.qp_feasible else 0.0,
            float(len(context.morphology_graph.robot_anchors)),
            float(len(context.contact_candidate_set.candidates)),
            float(len(context.contact_candidate_set.group_proposals)),
        ]
    return _pad_or_trim(base, width)


def _object_features(object_state: object, width: int) -> list[float]:
    pose = list(getattr(object_state, "pose_world"))
    twist = list(getattr(object_state, "twist_world"))
    object_id = str(getattr(object_state, "object_id"))
    return _pad_or_trim([*pose, *twist, _stable_scalar(object_id)], width)


def _mean_norm(rows: list[list[float]]) -> float:
    if not rows:
        return 0.0
    return sum(math.sqrt(sum(float(value) ** 2 for value in row)) for row in rows) / len(rows)


def _stable_scalar(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def _pad_or_trim(values: list[float], width: int) -> list[float]:
    result = [float(value) for value in values[:width]]
    result.extend([0.0] * (width - len(result)))
    return result


def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(tokens.dtype)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _head(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )
