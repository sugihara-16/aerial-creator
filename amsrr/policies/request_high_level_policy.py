"""Learned pi_H ranks complete request tuples; it has no continuous CWT heads."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from scipy.spatial.transform import Rotation

import torch
from torch import nn
from torch.nn import functional as F

from amsrr.encoders.morphology_graph_encoder import MorphologyGraphEncoder
from amsrr.encoders.interaction_envelope_encoder import InteractionEnvelopeEncoder
from amsrr.policies.contact_candidate_encoder import ContactCandidateEncoder
from amsrr.policies.high_level_requests import HighLevelDecisionContext
from amsrr.schemas.common import SchemaBase, SchemaValidationError
from amsrr.schemas.high_level import (
    HIGH_LEVEL_REQUEST_CONTRACT,
    HighLevelRequest,
    REQUEST_OUTCOME_STATUSES,
    RequestRankingLabel,
)
from amsrr.schemas.irg import IRGNodeType, PhaseType

REQUEST_FEATURE_VERSION = "request_group_phase_goal_observation_v1"
PHASE_TYPES = tuple(PhaseType)
REQUEST_FEATURE_NAMES = (
    *(f"contact_group.{i}" for i in range(48)),
    *(f"interaction_envelope.{i}" for i in range(32)),
    *(f"current_phase.{x.value}" for x in PHASE_TYPES),
    *(f"requested_phase.{x.value}" for x in PHASE_TYPES),
    "continuation",
    "same_contact_group",
    "elapsed_phase_s",
    "target_dx_m",
    "target_dy_m",
    "target_dz_m",
    "target_position_known",
    "target_rx_rad",
    "target_ry_rad",
    "target_rz_rad",
    "target_pos_tolerance_m",
    "target_rot_tolerance_rad",
    "contact_distance_m",
    "normal_load_n",
    "slip_speed_mps",
    "confidence",
    "contact_feedback_known",
    "controller_qp_feasible",
    "observed_guards_ready",
    *(f"previous_outcome.{status}" for status in REQUEST_OUTCOME_STATUSES),
)


@dataclass
class RequestHighLevelPolicyConfig(SchemaBase):
    d_model: int = 64
    hidden_dim: int = 64
    feature_version: str = REQUEST_FEATURE_VERSION
    contract_version: str = HIGH_LEVEL_REQUEST_CONTRACT

    def validate(self) -> None:
        if self.d_model < 1 or self.hidden_dim < 1:
            raise SchemaValidationError("request network dimensions must be positive")
        if (
            self.feature_version != REQUEST_FEATURE_VERSION
            or self.contract_version != HIGH_LEVEL_REQUEST_CONTRACT
        ):
            raise SchemaValidationError("unsupported request policy/feature contract")


class RequestHighLevelPolicy(nn.Module):
    policy_version = HIGH_LEVEL_REQUEST_CONTRACT

    def __init__(self, config: RequestHighLevelPolicyConfig | None = None) -> None:
        super().__init__()
        self.config = config or RequestHighLevelPolicyConfig()
        self.config.validate()
        self.candidate_encoder = ContactCandidateEncoder(d_model=48)
        self.envelope_encoder = InteractionEnvelopeEncoder(d_model=32)
        self.morphology_encoder = MorphologyGraphEncoder(d_model=self.config.d_model)
        self.request_head = nn.Sequential(
            nn.Linear(
                len(REQUEST_FEATURE_NAMES) + self.config.d_model, self.config.hidden_dim
            ),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, 1),
        )

    def request_features(self, context: HighLevelDecisionContext) -> torch.Tensor:
        context.validate_snapshot()
        encoding = self.candidate_encoder.encode(context.scene.contact_candidate_set)
        group_tokens = dict(zip(encoding.group_ids[0], encoding.group_tokens()))
        envelope = deepcopy(context.scene.interaction_envelope)
        # Scene IDs are provenance, not a feature for memorizing training cases.
        envelope.task_id = "request_task"
        encoded_envelope = self.envelope_encoder.encode(envelope)
        valid_tokens = [
            row
            for row, active in zip(encoded_envelope.tokens[0], encoded_envelope.mask[0])
            if active
        ]
        envelope_features = [
            sum(row[i] for row in valid_tokens) / len(valid_tokens) for i in range(32)
        ]
        phases = {
            n.node_id: PhaseType(n.feature["phase_type"])
            for n in context.scene.irg.nodes
            if n.node_type == IRGNodeType.PHASE
        }
        objects = {o.object_id: o for o in context.observation.object_states}
        estimates = {e.candidate_id: e for e in context.observation.contact_estimates}
        guards = {g.guard_id: g for g in context.observation.guard_samples}
        rows = []
        for entry in context.catalog.entries:
            current_phase = phases[context.execution_state.phase_id]
            target_phase = PhaseType(entry.phase_type)
            relative = [0.0, 0.0, 0.0]
            rotation_error = [0.0, 0.0, 0.0]
            tolerance = [0.0, 0.0]
            known_target = False
            for target in entry.target_nodes:
                f = target.feature
                entity = f.get("target_entity_id")
                pose = f.get("pose_target_world")
                if entity in objects and pose is not None:
                    relative = [
                        float(pose[i]) - objects[entity].pose_world[i] for i in range(3)
                    ]
                    known_target = True
                    rotation_error = (
                        (
                            Rotation.from_quat(objects[entity].pose_world[3:]).inv()
                            * Rotation.from_quat(pose[3:])
                        )
                        .as_rotvec()
                        .tolist()
                    )
                    limits = f.get("tolerance", {})
                    tolerance = [
                        float(limits.get("pos_m", 0.0)),
                        float(limits.get("rot_rad", 0.0)),
                    ]
                    break
            feedback = [
                estimates[cid] for cid in entry.candidate_ids if cid in estimates
            ]
            if feedback:
                contact = [
                    sum(getattr(e, name) for e in feedback) / len(feedback)
                    for name in (
                        "signed_distance_m",
                        "normal_load_n",
                        "slip_speed_mps",
                        "confidence",
                    )
                ]
            else:
                contact = [0.0] * 4
            row = [
                *group_tokens.get(entry.request.contact_group_id, [0.0] * 48),
                *envelope_features,
                *(float(current_phase == phase) for phase in PHASE_TYPES),
                *(float(target_phase == phase) for phase in PHASE_TYPES),
                float(entry.transition is None),
                float(
                    entry.request.contact_group_id
                    == context.execution_state.contact_group_id
                ),
                context.observation.time_s - context.execution_state.phase_started_s,
                *relative,
                float(known_target),
                *rotation_error,
                *tolerance,
                *contact,
                float(bool(feedback)),
                float(context.observation.controller_status.qp_feasible),
                float(
                    bool(entry.required_guard_ids)
                    and all(
                        gid in guards and guards[gid].satisfied
                        for gid in entry.required_guard_ids
                    )
                ),
                *(
                    float(
                        context.previous_outcome is not None
                        and context.previous_outcome.status == status
                    )
                    for status in REQUEST_OUTCOME_STATUSES
                ),
            ]
            if len(row) != len(REQUEST_FEATURE_NAMES):
                raise SchemaValidationError("request feature layout mismatch")
            rows.append(row)
        parameter = next(self.parameters())
        values = torch.tensor(
            rows, device=parameter.device, dtype=parameter.dtype
        ).reshape(-1, len(REQUEST_FEATURE_NAMES))
        if not torch.isfinite(values).all():
            raise SchemaValidationError("nonfinite request features")
        return values

    def forward(self, context: HighLevelDecisionContext) -> torch.Tensor:
        features = self.request_features(context)
        if not len(features):
            return features.new_empty((0,))
        morphology = self.morphology_encoder(
            [context.scene.morphology_graph],
            runtime_observations=[context.scene.runtime_observation],
        ).graph_embeddings[0]
        return self.request_head(
            torch.cat((features, morphology.expand(len(features), -1)), dim=1)
        ).squeeze(-1)

    @torch.no_grad()
    def rank_with_scores(
        self, context: HighLevelDecisionContext
    ) -> tuple[list[HighLevelRequest], list[float]]:
        # Inference uses eval mode even if the last operation was an optimizer step.
        was_training = self.training
        self.eval()
        try:
            scores = self(context)
            if not torch.isfinite(scores).all():
                raise SchemaValidationError("nonfinite request ranking")
            indices = torch.argsort(scores, descending=True, stable=True).tolist()
            return (
                [deepcopy(context.catalog.entries[i].request) for i in indices],
                [float(scores[i]) for i in indices],
            )
        finally:
            self.train(was_training)

    def rank(self, context: HighLevelDecisionContext) -> list[HighLevelRequest]:
        return self.rank_with_scores(context)[0]

    def checkpoint(self) -> dict:
        return {
            "contract_version": HIGH_LEVEL_REQUEST_CONTRACT,
            "feature_version": REQUEST_FEATURE_VERSION,
            "config": self.config.to_dict(),
            "state_dict": {
                k: v.detach().cpu().clone() for k, v in self.state_dict().items()
            },
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: dict) -> "RequestHighLevelPolicy":
        if (
            checkpoint.get("contract_version") != HIGH_LEVEL_REQUEST_CONTRACT
            or checkpoint.get("feature_version") != REQUEST_FEATURE_VERSION
        ):
            raise SchemaValidationError(
                "checkpoint is not a compatible request-ranking policy"
            )
        model = cls(RequestHighLevelPolicyConfig.from_dict(checkpoint["config"]))
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise SchemaValidationError("nonfinite request checkpoint")
        return model

    @classmethod
    def load(cls, path: str | Path) -> "RequestHighLevelPolicy":
        return cls.from_checkpoint(
            torch.load(path, map_location="cpu", weights_only=True)
        )


def request_ranking_loss(
    policy: RequestHighLevelPolicy,
    context: HighLevelDecisionContext,
    label: RequestRankingLabel,
) -> torch.Tensor:
    """Supervised tuple selection; labels never participate in feature construction.

    The dataset must trace this label to a checked plan's observed execution
    outcome. Candidate-search success trains only the accepted request; emergency
    fallback success must never reward the rejected request.
    """
    label.validate()
    if label.decision_snapshot_hash != context.catalog.snapshot_hash:
        raise SchemaValidationError("label belongs to a different decision snapshot")
    if label.execution_outcome not in {
        "first_choice_success",
        "searched_candidate_success",
    }:
        raise SchemaValidationError(
            "fallback/failure cannot supervise successful request ranking"
        )
    context.catalog.resolve(label.request)
    scores = policy(context)
    key = label.request.stable_hash()
    index = next(
        i
        for i, e in enumerate(context.catalog.entries)
        if e.request.stable_hash() == key
    )
    return F.cross_entropy(scores[None], torch.tensor([index], device=scores.device))
