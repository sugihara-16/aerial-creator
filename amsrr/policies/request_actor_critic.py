"""Categorical event-driven request actor shared by imitation and PPO.

Initial selection is supervised only from uncommitted pre-action states. Once
committed, the actor requests continuation or an IRG successor; the executor
retains the contact binding and owns deterministic completion guards.
"""

from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical

from amsrr.policies.request_high_level_policy import (
    RequestHighLevelPolicy,
    RequestHighLevelPolicyConfig,
    REQUEST_FEATURE_NAMES,
    REQUEST_FEATURE_VERSION,
)
from amsrr.policies.contact_group_geometry import contact_geometry
from amsrr.encoders.morphology_graph_encoder import MorphologyGraphBatch
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.high_level import HIGH_LEVEL_REQUEST_CONTRACT

ACTOR_CRITIC_VERSION = "causal_event_request_actor_critic_v1"
MORPHOLOGY_ACTOR_CRITIC_VERSION = "causal_morphology_event_request_actor_critic_v2"
SAMPLING_ORDER_VERSION = "semantic_request_sampling_order_v1"


def sample_request_index(entries, scores, *, generator=None):
    """Make a seeded categorical draw independent of opaque catalog hashes."""

    def key(index):
        entry = entries[index]
        return (
            entry.phase_id,
            entry.request.contact_group_id or "",
            tuple(sorted(entry.candidate_ids)),
            tuple(sorted(n.node_id for n in entry.target_nodes)),
            (
                ()
                if entry.transition is None
                else (entry.transition.src_id, entry.transition.dst_id)
            ),
        )

    ordered = sorted(torch.nonzero(torch.isfinite(scores)).flatten().tolist(), key=key)
    if not ordered or len({key(i) for i in ordered}) != len(ordered):
        raise ValueError("categorical requests require unique semantic sampling keys")
    indices = torch.tensor(ordered, device=scores.device)
    probabilities = scores[indices].softmax(-1)
    return ordered[int(torch.multinomial(probabilities, 1, generator=generator))]


class RequestActorCritic(nn.Module):
    def __init__(self, config=None, *, morphology_aware=False):
        super().__init__()
        self.ranker = RequestHighLevelPolicy(config or RequestHighLevelPolicyConfig())
        self.morphology_aware = morphology_aware
        self.version = MORPHOLOGY_ACTOR_CRITIC_VERSION if morphology_aware else ACTOR_CRITIC_VERSION
        if morphology_aware:
            old = self.ranker.contact_member[0]
            self.ranker.contact_member[0] = nn.Linear(old.in_features + self.ranker.config.d_model, old.out_features)
        if self.ranker.config.selection_scope != "requests":
            raise ValueError("event policy requires full request scope")
        width = len(REQUEST_FEATURE_NAMES) + self.ranker.config.d_model
        self.value_head = nn.Sequential(
            nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 1)
        )
        # Dropout would invalidate old/new PPO probabilities. BC and PPO use
        # the exact same deterministic logits (categorical sampling is external).
        self.ranker.contact_head[2] = nn.Identity()

    def encode(self, context):
        context.validate_snapshot()
        initial = context.execution_state.plan_id is None
        if initial and (
            context.execution_state.contact_group_id is not None
            or context.execution_state.contact_bindings
        ):
            raise ValueError("uncommitted request has an existing binding")
        features = self.ranker.request_features(context)
        mask = torch.ones(len(features), dtype=torch.bool)
        if initial:
            mask = torch.tensor(
                [
                    e.request.contact_group_id is not None
                    and e.request.transition_id is None
                    and e.phase_id == context.execution_state.phase_id
                    for e in context.catalog.entries
                ]
            )
        elif context.execution_state.contact_group_id is not None:
            mask = torch.tensor(
                [
                    e.request.contact_group_id
                    in (None, context.execution_state.contact_group_id)
                    for e in context.catalog.entries
                ]
            )
        candidates, membership = contact_geometry(context)
        graph = self.ranker.morphology_encoder.tensorizer.tensorize(
            [context.scene.morphology_graph],
            runtime_observations=[context.scene.runtime_observation],
        )
        if not bool(mask.any()):
            raise ValueError("no eligible request")
        extra = {}
        if self.morphology_aware:
            from amsrr.policies.request_morphology_features import canonical_morphology_batch, candidate_owner_indices
            graph = canonical_morphology_batch(graph)
            extra["owners"] = candidate_owner_indices(context, graph)
        return dict(
            features=features.detach().cpu(),
            mask=mask,
            candidates=candidates,
            membership=membership,
            initial=torch.tensor(initial),
            graph=graph,
            **extra,
        )

    def forward_encoded(
        self, features, morphology, mask, *, candidates, membership, initial, owners=None
    ):
        r = self.ranker
        graph_encoding = r.morphology_encoder(morphology)
        local = (candidates - r.contact_mean) / r.contact_scale
        if self.morphology_aware:
            if owners is None or owners.shape != candidates.shape[:2]:
                raise ValueError("morphology-aware requests require candidate owners")
            nodes = graph_encoding.node_embeddings
            if bool((owners < 0).any()) or bool((owners >= nodes.shape[1]).any()):
                raise ValueError("candidate owner outside graph")
            owned = nodes.gather(1, owners[:, :, None].expand(-1, -1, nodes.shape[-1]))
            local = torch.cat((local, owned), -1)
        member = r.contact_member(local)
        weights = membership.to(member.dtype)
        count = weights.sum(-1, keepdim=True)
        group = torch.cat(
            (
                torch.bmm(weights, member) / count.clamp_min(1),
                torch.bmm(weights, member.square()) / count.clamp_min(1),
                count / 12.0,
            ),
            dim=-1,
        )
        graph = graph_encoding.graph_embeddings
        contact = r.contact_head(group).squeeze(-1)
        normalized = (features - r.feature_mean) / r.feature_scale
        temporal = r.request_head(
            torch.cat(
                (normalized, graph[:, None].expand(-1, features.shape[1], -1), group),
                dim=-1,
            )
        ).squeeze(-1)
        scores = torch.where(initial[:, None], contact, temporal).masked_fill(
            ~mask, -torch.inf
        )
        if not bool(mask.any(-1).all()) or not torch.isfinite(scores[mask]).all():
            raise ValueError("invalid categorical support/logits")
        mean = (normalized * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True)
        value = self.value_head(torch.cat((mean, graph), -1)).squeeze(-1)
        return scores, value

    def evaluate_encoding(self, encoded):
        device = next(self.parameters()).device
        graph = encoded["graph"].to(device=device)
        return self.forward_encoded(
            encoded["features"][None].to(device),
            graph,
            encoded["mask"][None].to(device),
            candidates=encoded["candidates"][None].to(device),
            membership=encoded["membership"][None].to(device),
            initial=encoded["initial"].reshape(1).to(device),
            owners=None if "owners" not in encoded else encoded["owners"][None].to(device),
        )

    @torch.no_grad()
    def decide(self, context, *, deterministic=False, generator=None):
        self.eval()
        encoded = self.encode(context)
        scores, value = self.evaluate_encoding(encoded)
        dist = Categorical(logits=scores[0])
        index = (
            int(scores[0].argmax())
            if deterministic
            else sample_request_index(
                context.catalog.entries, scores[0], generator=generator
            )
        )
        return dict(
            request=deepcopy(context.catalog.entries[index].request),
            action=index,
            log_prob=float(dist.log_prob(torch.tensor(index, device=scores.device))),
            value=float(value[0]),
            encoded=encoded,
            snapshot_hash=context.catalog.snapshot_hash,
            requests=[e.request.to_dict() for e in context.catalog.entries],
            logits=scores[0].cpu(),
            sampling_order=SAMPLING_ORDER_VERSION,
        )

    def rank_with_scores(self, context):
        with torch.no_grad():
            scores, _ = self.evaluate_encoding(self.encode(context))
            ids = [
                i
                for i in scores[0].argsort(descending=True, stable=True).tolist()
                if torch.isfinite(scores[0, i])
            ]
            return (
                [deepcopy(context.catalog.entries[i].request) for i in ids],
                [float(scores[0, i]) for i in ids],
            )

    def checkpoint(self):
        return dict(
            version=self.version,
            contract_version=HIGH_LEVEL_REQUEST_CONTRACT,
            feature_version=REQUEST_FEATURE_VERSION,
            config=self.ranker.config.to_dict(),
            state_dict={
                k: v.detach().cpu().clone() for k, v in self.state_dict().items()
            },
        )

    @classmethod
    def from_checkpoint(cls, checkpoint):
        if (
            checkpoint.get("version") not in {ACTOR_CRITIC_VERSION, MORPHOLOGY_ACTOR_CRITIC_VERSION}
            or checkpoint.get("contract_version") != HIGH_LEVEL_REQUEST_CONTRACT
            or checkpoint.get("feature_version") != REQUEST_FEATURE_VERSION
        ):
            raise SchemaValidationError("incompatible event actor/critic checkpoint")
        model = cls(RequestHighLevelPolicyConfig.from_dict(checkpoint["config"]),
                    morphology_aware=checkpoint["version"] == MORPHOLOGY_ACTOR_CRITIC_VERSION)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
            raise ValueError("nonfinite event checkpoint")
        if not bool(
            (model.ranker.feature_scale > 0).all()
            and (model.ranker.contact_scale > 0).all()
        ):
            raise ValueError("invalid normalization")
        return model

    @classmethod
    def load(cls, path):
        return cls.from_checkpoint(
            torch.load(Path(path), map_location="cpu", weights_only=True)
        )
