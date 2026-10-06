"""Learned pi_H ranks complete request tuples; it has no continuous CWT heads."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

import torch
from torch import nn
from torch.nn import functional as F

from amsrr.encoders.morphology_graph_encoder import (
    MorphologyGraphEncoder,
    MorphologyGraphBatch,
)
from amsrr.encoders.interaction_envelope_encoder import InteractionEnvelopeEncoder
from amsrr.policies.contact_candidate_encoder import ContactCandidateEncoder
from amsrr.policies.contact_group_geometry import (
    contact_geometry, CONTACT_FEATURE_DIM, normalize_contact_features,
)
from amsrr.policies.high_level_requests import HighLevelDecisionContext
from amsrr.schemas.common import SchemaBase, SchemaValidationError
from amsrr.schemas.high_level import (
    HIGH_LEVEL_REQUEST_CONTRACT,
    HighLevelRequest,
    REQUEST_OUTCOME_STATUSES,
    RequestRankingLabel,
)
from amsrr.schemas.irg import IRGNodeType, PhaseType

REQUEST_FEATURE_VERSION = "request_module_frame_gripper_member_geometry_v5"
REQUEST_SELECTION_SCOPES = ("requests", "contact_group")
PHASE_TYPES = tuple(PhaseType)
OWNER_GEOMETRY_NAMES = ("dx_m", "dy_m", "dz_m", "nx", "ny", "nz", "distance_m")
REQUEST_FEATURE_NAMES = (
    *(f"contact_group.{i}" for i in range(48)),
    *(
        f"contact_owner.{name}.{stat}"
        for name in OWNER_GEOMETRY_NAMES
        for stat in ("mean", "min", "max")
    ),
    *(f"interaction_envelope.{i}" for i in range(32)),
    *(f"current_phase.{x.value}" for x in PHASE_TYPES),
    *(f"requested_phase.{x.value}" for x in PHASE_TYPES),
    "continuation",
    "same_contact_group",
    "elapsed_phase_s",
    "nominal_phase_duration_s",
    "nominal_phase_progress",
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
    selection_scope: str = "requests"
    member_width: int = 96

    def validate(self) -> None:
        if min(self.d_model, self.hidden_dim, self.member_width) < 1:
            raise SchemaValidationError("request network dimensions must be positive")
        if self.selection_scope not in REQUEST_SELECTION_SCOPES:
            raise SchemaValidationError("unsupported request selection scope")
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
        self.register_buffer("feature_mean", torch.zeros(len(REQUEST_FEATURE_NAMES)))
        self.register_buffer("feature_scale", torch.ones(len(REQUEST_FEATURE_NAMES)))
        self.register_buffer("contact_mean", torch.zeros(CONTACT_FEATURE_DIM))
        self.register_buffer("contact_scale", torch.ones(CONTACT_FEATURE_DIM))
        w = self.config.member_width
        self.contact_member = nn.Sequential(
            nn.Linear(CONTACT_FEATURE_DIM, w), nn.SiLU(), nn.Linear(w, w), nn.SiLU()
        )
        self.contact_head = nn.Sequential(
            nn.Linear(2 * w + 1, w), nn.SiLU(), nn.Dropout(0.1), nn.Linear(w, 1)
        )
        self.request_head = nn.Sequential(
            nn.Linear(
                len(REQUEST_FEATURE_NAMES) + self.config.d_model + 2 * w + 1,
                self.config.hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, 1),
        )

    def request_features(self, context: HighLevelDecisionContext) -> torch.Tensor:
        context.validate_snapshot()
        return self._request_features(context)

    def _request_features(self, context: HighLevelDecisionContext) -> torch.Tensor:
        """Pure feature calculation after the caller validates this snapshot."""
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
        phase_nodes = {
            n.node_id: n
            for n in context.scene.irg.nodes
            if n.node_type == IRGNodeType.PHASE
        }
        duration = float(
            phase_nodes[context.execution_state.phase_id].feature.get(
                "nominal_duration_s"
            )
            or 0.0
        )
        elapsed = context.observation.time_s - context.execution_state.phase_started_s
        candidates = {
            c.candidate_id: c for c in context.scene.contact_candidate_set.candidates
        }
        anchors = {a.anchor_id: a for a in context.scene.morphology_graph.robot_anchors}
        modules = {m.module_id: m for m in context.observation.module_states}
        objects = {o.object_id: o for o in context.observation.object_states}
        estimates = {e.candidate_id: e for e in context.observation.contact_estimates}
        guards = {g.guard_id: g for g in context.observation.guard_samples}
        rows = []
        for entry in context.catalog.entries:
            # Preserve candidate -> owner correspondence lost by group averaging.
            # These are module-origin distances, not an IK/reachability claim.
            geometry = []
            for cid in entry.candidate_ids:
                candidate = candidates[cid]
                owner = modules[anchors[candidate.anchor_id].module_id]
                rotation = Rotation.from_quat(owner.pose_world[3:]).inv()
                delta = (
                    np.asarray(candidate.contact_pose_world[:3]) - owner.pose_world[:3]
                )
                geometry.append(
                    [
                        *rotation.apply(delta),
                        *rotation.apply(candidate.normal_world),
                        float(np.linalg.norm(delta)),
                    ]
                )
            geometry_features = (
                [
                    float(value)
                    for column in np.asarray(geometry).T
                    for value in (column.mean(), column.min(), column.max())
                ]
                if geometry
                else [0.0] * (3 * len(OWNER_GEOMETRY_NAMES))
            )
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
                *geometry_features,
                *envelope_features,
                *(float(current_phase == phase) for phase in PHASE_TYPES),
                *(float(target_phase == phase) for phase in PHASE_TYPES),
                float(entry.transition is None),
                float(
                    entry.request.contact_group_id
                    == context.execution_state.contact_group_id
                ),
                elapsed,
                duration,
                min(elapsed / duration, 2.0) if duration > 0 else 0.0,
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

    def selection_mask(self, context: HighLevelDecisionContext) -> torch.Tensor:
        """Restrict the contact warmup to an uncommitted, current-phase choice.

        No teacher label is needed to fix the temporal part of this decision.
        A contact-only checkpoint cannot be used as a learned transition policy.
        """
        context.validate_snapshot()
        state = context.execution_state
        if self.config.selection_scope == "contact_group":
            if (
                state.plan_id is not None
                or state.contact_group_id is not None
                or state.contact_bindings
                or context.previous_outcome is not None
            ):
                raise SchemaValidationError(
                    "initial-contact policy requires uncommitted execution state"
                )
            allowed = [
                e.request.contact_group_id is not None
                and e.request.transition_id is None
                and e.phase_id == state.phase_id
                for e in context.catalog.entries
            ]
            entries = [
                e for e, active in zip(context.catalog.entries, allowed) if active
            ]
            groups = [e.request.contact_group_id for e in entries]
            if (
                len(groups) != len(set(groups))
                or len({e.request.subgoal_id for e in entries}) > 1
            ):
                raise SchemaValidationError(
                    "contact-only choice has ambiguous subgoals"
                )
        else:
            allowed = [True] * len(context.catalog.entries)
        return torch.tensor(
            allowed, dtype=torch.bool, device=next(self.parameters()).device
        )

    def forward(self, context: HighLevelDecisionContext) -> torch.Tensor:
        mask = self.selection_mask(context)
        features = self.request_features(context)
        if not len(features):
            return features.new_empty((0,))
        morphology = self.morphology_encoder.tensorizer.tensorize(
            [context.scene.morphology_graph],
            runtime_observations=[context.scene.runtime_observation],
        )
        candidates, membership = contact_geometry(context)
        return self.forward_encoded(
            features[None],
            morphology,
            mask[None],
            candidates=candidates[None].to(features.device),
            membership=membership[None].to(features.device),
        )[0]

    def forward_encoded(
        self,
        features: torch.Tensor,
        morphology: MorphologyGraphBatch,
        mask: torch.Tensor,
        *,
        candidates: torch.Tensor,
        membership: torch.Tensor,
    ) -> torch.Tensor:
        """Shared inference/training path after deterministic tensorization."""
        if (
            features.ndim != 3
            or features.shape[-1] != len(REQUEST_FEATURE_NAMES)
            or mask.shape != features.shape[:2]
        ):
            raise SchemaValidationError("invalid encoded request batch")
        if not bool((self.feature_scale > 0).all()):
            raise SchemaValidationError("invalid request feature normalization")
        if (
            candidates.ndim != 3
            or candidates.shape[-1] != CONTACT_FEATURE_DIM
            or membership.shape != (*features.shape[:2], candidates.shape[1])
            or not bool((self.contact_scale > 0).all())
        ):
            raise SchemaValidationError("invalid contact-member batch or normalization")
        member = self.contact_member(
            normalize_contact_features(candidates, self.contact_mean, self.contact_scale)
        )
        weights = membership.to(member.dtype)
        count = weights.sum(-1, keepdim=True)
        mean = torch.bmm(weights, member) / count.clamp_min(1)
        second = torch.bmm(weights, member.square()) / count.clamp_min(1)
        group = torch.cat((mean, second, count / 12.0), dim=-1)
        if self.config.selection_scope == "contact_group":
            # Legacy v32 group identity consumes only measured candidate geometry.
            # Phase, plan, world-frame graph embeddings and old group summaries
            # cannot supply shortcuts to the contact-only decision.
            return self.contact_head(group).squeeze(-1).masked_fill(~mask, -torch.inf)
        encoded = self.morphology_encoder(morphology).graph_embeddings
        normalized = (features - self.feature_mean) / self.feature_scale
        scores = self.request_head(
            torch.cat(
                (normalized, encoded[:, None].expand(-1, features.shape[1], -1), group),
                dim=-1,
            )
        ).squeeze(-1)
        return scores.masked_fill(~mask, -torch.inf)

    @torch.no_grad()
    def rank_with_scores(
        self, context: HighLevelDecisionContext
    ) -> tuple[list[HighLevelRequest], list[float]]:
        # Inference uses eval mode even if the last operation was an optimizer step.
        was_training = self.training
        self.eval()
        try:
            allowed = self.selection_mask(context)
            scores = self(context)
            if not torch.isfinite(scores[allowed]).all():
                raise SchemaValidationError("nonfinite request ranking")
            indices = [
                i
                for i in torch.argsort(scores, descending=True, stable=True).tolist()
                if allowed[i]
            ]
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
        if any(not torch.isfinite(p).all() for p in model.state_dict().values()):
            raise SchemaValidationError("nonfinite request checkpoint")
        if not bool((model.feature_scale > 0).all()) or not bool(
            (model.contact_scale > 0).all()
        ):
            raise SchemaValidationError("invalid request feature normalization")
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
    if not bool(policy.selection_mask(context)[index]):
        raise SchemaValidationError("label excluded by request selection scope")
    return F.cross_entropy(scores[None], torch.tensor([index], device=scores.device))
