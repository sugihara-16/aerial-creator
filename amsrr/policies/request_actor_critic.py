"""Categorical event-driven request actor shared by imitation and PPO.

Initial selection is supervised only from uncommitted pre-action states. Once
committed, the actor requests continuation or an IRG successor; the executor
retains the contact binding and owns deterministic completion guards.
"""

from copy import deepcopy
import math
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
from amsrr.policies.contact_group_geometry import _contact_geometry, normalize_contact_features
from amsrr.encoders.morphology_graph_encoder import MorphologyGraphBatch
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.high_level import HIGH_LEVEL_REQUEST_CONTRACT
from amsrr.policies.anchor_preference import AnchorPreference, request_anchor_distances

ACTOR_CRITIC_VERSION = "causal_event_request_actor_critic_v1"
MORPHOLOGY_ACTOR_CRITIC_VERSION = "causal_morphology_event_request_actor_critic_v2"
OBJECT_ACTOR_CRITIC_VERSION = "causal_object_condition_event_request_actor_critic_v3"
ANCHOR_PRIOR_ACTOR_CRITIC_VERSION = "causal_anchor_preference_event_actor_critic_v4"
OBJECT_FEATURE_VERSION = "estimated_mass_kg_com_object_0p1m_v1"
SAMPLING_ORDER_VERSION = "semantic_request_sampling_order_v1"
SPLIT_ACTOR_VERSION = "independent_fixed_branch_request_actor_v1"


def object_condition_features(task):
    """Same pre-action estimates as the planner; no true mass or future state fallback."""
    if len(task.scene.objects) != 1:
        raise ValueError("object-conditioned contact profile requires one rigid object")
    metadata=task.metadata
    if 'estimated_mass_kg' not in metadata or 'estimated_com_object' not in metadata:
        raise ValueError("object-conditioned policy requires declared mass/COM estimates")
    com=metadata['estimated_com_object']
    if len(com)!=3:
        raise ValueError("estimated object COM must have three coordinates")
    features=torch.tensor([metadata['estimated_mass_kg'],*[float(v)/.1 for v in com]],dtype=torch.float32)
    if not torch.isfinite(features).all() or features[0]<=0:
        raise ValueError("invalid estimated object mass/COM")
    return features


def require_training_object_inputs(model, task):
    """Do not silently restart varied-property training with a legacy actor."""
    condition=task.metadata.get('object_condition',{})
    if any(k in condition for k in ('mass_factor','com_fraction')) and not model.object_condition_aware:
        raise ValueError('object-condition training requires mass/COM inputs; run configure-learning-signals first')
    if model.temporal_options is not None and not model.motion_feedback:
        raise ValueError('duration training requires observed motion feedback; run configure-motion-feedback first')


class InitialContactTeacher(nn.Module):
    """Explicit demonstration contact choice; later requests use the expert.

    Callers must mark its rollouts as teacher data and exclude them from PPO.
    The encoded initial observation is built before the chosen action is bound.
    """
    def __init__(self, model, group_id):
        super().__init__()
        self.model, self.group_id = model, group_id
        self.version = model.version

    @torch.no_grad()
    def decide(self, context, **kwargs):
        result = self.model.decide(context, **kwargs)
        if context.execution_state.plan_id is not None:
            return result
        choices = [i for i,e in enumerate(context.catalog.entries)
                   if e.request.contact_group_id == self.group_id
                   and e.request.transition_id is None and result['encoded']['mask'][i]]
        if len(choices) != 1:
            raise ValueError('teacher contact group is not uniquely admissible')
        index=choices[0]
        result.update(request=deepcopy(context.catalog.entries[index].request), action=index,
            log_prob=float(Categorical(logits=result['logits']).log_prob(torch.tensor(index))))
        return result


def sample_request_index(entries, scores, *, generator=None, hold_indices=None):
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
            0 if hold_indices is None else int(hold_indices[index]),
        )

    ordered = sorted(torch.nonzero(torch.isfinite(scores)).flatten().tolist(), key=key)
    if not ordered or len({key(i) for i in ordered}) != len(ordered):
        raise ValueError("categorical requests require unique semantic sampling keys")
    indices = torch.tensor(ordered, device=scores.device)
    probabilities = scores[indices].softmax(-1)
    return ordered[int(torch.multinomial(probabilities, 1, generator=generator))]


class RequestActorCritic(nn.Module):
    def __init__(self, config=None, *, morphology_aware=False, object_condition_aware=False):
        super().__init__()
        self.anchor_preference = None
        self.ranked_contact = None
        self.ranking_prefix_length = 1
        self.temporal_options = None
        self.contact_return_ranking = None
        self.motion_feedback = False
        self.temporal_object_conditions = False
        self.fixed_actor = None
        self.trainable_branch = None
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
        self.object_condition_aware=False
        if object_condition_aware:
            self.enable_object_conditions()

    def enable_fixed_branch(self, reference, *, trainable_branch):
        """Use an independent immutable actor for the other decision branch.

        Values always come from the learner, which predicts the return of the
        combined policy. Freezing a head on a shared encoder is insufficient.
        """
        if (trainable_branch not in {'contact', 'temporal'} or self.fixed_actor is not None
                or reference.fixed_actor is not None or reference is self):
            raise ValueError('invalid independent fixed actor')
        own, fixed = self.checkpoint(), reference.checkpoint()
        if {k:v for k,v in own.items() if k != 'state_dict'} != {
                k:v for k,v in fixed.items() if k != 'state_dict'}:
            raise ValueError('fixed actor observation/action contracts differ')
        self.fixed_actor = deepcopy(reference).to(next(self.parameters()).device).eval()
        self.fixed_actor.requires_grad_(False)
        self.trainable_branch = trainable_branch

    def enable_object_conditions(self):
        """Explicit v2 -> v3 migration with exactly zero initial policy residual."""
        if not self.morphology_aware:
            raise ValueError("object conditioning requires morphology-aware contacts")
        if self.object_condition_aware:
            return
        width=self.ranker.contact_head[0].in_features
        self.ranker.object_condition_head=nn.Sequential(nn.Linear(width+4,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.ranker.object_condition_head[-1].weight)
        nn.init.zeros_(self.ranker.object_condition_head[-1].bias)
        self.object_condition_aware=True
        self.version=(ANCHOR_PRIOR_ACTOR_CRITIC_VERSION if self.anchor_preference is not None
                      else OBJECT_ACTOR_CRITIC_VERSION)
        self.ranker.object_condition_head.to(next(self.parameters()).device)

    def enable_motion_feedback(self):
        """Append nine observed motion features without changing old outputs."""
        if self.motion_feedback or self.temporal_options is None:
            raise ValueError('motion feedback requires an unmigrated duration actor')
        for head in (self.ranker.request_head, self.ranker.hold_head, self.value_head):
            old = head[0]
            new = nn.Linear(old.in_features + 9, old.out_features).to(old.weight)
            with torch.no_grad():
                new.weight.zero_()
                new.weight[:, :old.in_features].copy_(old.weight)
                new.bias.copy_(old.bias)
            head[0] = new
        self.motion_feedback = True

    @staticmethod
    def observed_motion_features(context):
        """Distances, slip, motor torque and object speed; no force/truth input."""
        committed = context.execution_state.contact_group_id
        ids = {cid for e in context.catalog.entries
               if committed is not None and e.request.contact_group_id == committed
               for cid in e.candidate_ids}
        observed = {x.candidate_id: x for x in context.observation.contact_motion}
        selected = [observed[cid] for cid in sorted(ids) if cid in observed]
        complete = bool(ids) and len(selected) == len(ids)
        values = [0.] * 7
        if complete:
            values = [min(x.signed_distance_m for x in selected) / .01,
                      max(x.signed_distance_m for x in selected) / .01,
                      max(x.slip_speed_mps for x in selected) / .05,
                      min(x.motor_load_nm for x in selected) / .1,
                      max(x.motor_load_nm for x in selected) / .1,
                      1., float(all(x.velocity_valid for x in selected))]
        objects = context.observation.object_states
        # These are observation velocities already admitted by the public input.
        linear = max((sum(v*v for v in o.twist_world[:3]) ** .5 for o in objects), default=0.)
        angular = max((sum(v*v for v in o.twist_world[3:]) ** .5 for o in objects), default=0.)
        return torch.tensor([*values, linear / .05, angular / .2], dtype=torch.float32)

    def enable_temporal_object_conditions(self):
        """Expose the existing estimated mass/COM to timing and state value."""
        if (not self.motion_feedback or not self.object_condition_aware
                or self.temporal_object_conditions):
            raise ValueError('temporal object inputs require unmigrated motion/object actor')
        for head in (self.ranker.request_head, self.ranker.hold_head, self.value_head):
            old = head[0]
            new = nn.Linear(old.in_features + 4, old.out_features).to(old.weight)
            with torch.no_grad():
                new.weight.zero_()
                new.weight[:, :old.in_features].copy_(old.weight)
                new.bias.copy_(old.bias)
            head[0] = new
        self.temporal_object_conditions = True

    def enable_temporal_options(self, config=None):
        """One sampled action chooses a request AND bounded keep duration."""
        config = dict(config or dict(version='request_keep_duration_v1',
            durations_s=[.5, 1., 2., 4., 8.], initial_wait_probability=.2))
        durations = config.get('durations_s', [])
        probability = config.get('initial_wait_probability', 0.)
        if (self.temporal_options is not None
                or set(config) != {'version', 'durations_s', 'initial_wait_probability'}
                or config['version'] != 'request_keep_duration_v1'
                or not durations or len(set(durations)) != len(durations)
                or any(not math.isfinite(t) or t < .5 or t > 8. for t in durations)
                or durations != sorted(durations)
                or not math.isfinite(probability) or not 0 < probability < 1):
            raise ValueError('invalid temporal duration contract')
        self.temporal_options = config
        width = self.ranker.request_head[0].in_features
        self.ranker.hold_head = nn.Sequential(nn.Linear(width,64),nn.SiLU(),nn.Linear(64,len(durations)))
        nn.init.zeros_(self.ranker.hold_head[-1].weight)
        nn.init.zeros_(self.ranker.hold_head[-1].bias)
        self.ranker.hold_head.to(next(self.parameters()).device)

    def enable_anchor_preference(self, preference=None):
        if self.anchor_preference is not None:
            raise ValueError("anchor preference is already configured")
        preference = AnchorPreference() if preference is None else preference
        if not isinstance(preference, AnchorPreference):
            raise TypeError("expected AnchorPreference")
        self.anchor_preference = preference
        self.version = ANCHOR_PRIOR_ACTOR_CRITIC_VERSION

    def encode(self, context):
        context.validate_snapshot()
        initial = context.execution_state.plan_id is None
        if initial and (
            context.execution_state.contact_group_id is not None
            or context.execution_state.contact_bindings
        ):
            raise ValueError("uncommitted request has an existing binding")
        # This call owns one complete snapshot validation. These pure helpers
        # consume it synchronously without rebuilding the same large hash twice.
        features = self.ranker._request_features(context)
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
        candidates, membership = _contact_geometry(context)
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
        if self.object_condition_aware:
            extra['object_features']=object_condition_features(context.task_spec)
        if self.anchor_preference is not None:
            extra['anchor_distances'] = request_anchor_distances(context, mask)
        if self.temporal_options is not None:
            indices, holds = [], []
            for i, entry in enumerate(context.catalog.entries):
                options = (range(1, len(self.temporal_options['durations_s'])+1)
                    if not initial and mask[i] and entry.request.transition_id is None else [0])
                for h in options:
                    indices.append(i); holds.append(h)
            indices = torch.tensor(indices, dtype=torch.long)
            features, mask, membership = features[indices], mask[indices], membership[indices]
            if 'anchor_distances' in extra:
                extra['anchor_distances'] = extra['anchor_distances'][indices]
            extra.update(request_indices=indices, hold_indices=torch.tensor(holds,dtype=torch.long))
        return dict(
            features=features.detach().cpu(),
            mask=mask,
            candidates=candidates,
            membership=membership,
            initial=torch.tensor(initial),
            graph=graph,
            **({'motion_features': self.observed_motion_features(context)} if self.motion_feedback else {}),
            **extra,
        )

    def action_features(self, features, morphology, *, candidates, membership,
                        owners=None, object_features=None, motion_features=None,
                        normalization_clip=None):
        """Shared causal state/action representation, independent of the learner."""
        r = self.ranker
        graph_encoding = r.morphology_encoder(morphology)
        local = normalize_contact_features(candidates, r.contact_mean, r.contact_scale)
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
        normalized = (features - r.feature_mean) / r.feature_scale
        if normalization_clip is not None:
            normalized = normalized.clamp(-normalization_clip, normalization_clip)
        temporal_input = torch.cat(
            (normalized, graph[:, None].expand(-1, features.shape[1], -1), group), dim=-1)
        if self.motion_feedback:
            if (motion_features is None or motion_features.shape != (len(features), 9)
                    or not torch.isfinite(motion_features).all()):
                raise ValueError('motion actor requires recorded finite observation features')
            temporal_input = torch.cat((temporal_input,
                motion_features[:, None].expand(-1, features.shape[1], -1)), -1)
        if self.temporal_object_conditions:
            temporal_input = torch.cat((temporal_input,
                object_features[:, None].expand(-1, features.shape[1], -1)), -1)
        return group, graph, normalized, temporal_input

    def forward_encoded(
        self, features, morphology, mask, *, candidates, membership, initial, owners=None, object_features=None,
        anchor_distances=None, hold_indices=None, motion_features=None,
    ):
        r = self.ranker
        group, graph, normalized, temporal_input = self.action_features(
            features, morphology, candidates=candidates, membership=membership,
            owners=owners, object_features=object_features, motion_features=motion_features)
        contact = r.contact_head(group).squeeze(-1)
        if self.anchor_preference is not None:
            if anchor_distances is None or anchor_distances.shape != mask.shape:
                raise ValueError("anchor-preference actor requires recorded request distances")
            prior = self.anchor_preference.logits(anchor_distances, mask).masked_fill(~mask, 0)
            contact = contact + prior
        if self.object_condition_aware:
            if (object_features is None or object_features.shape!=(len(features),4)
                    or not torch.isfinite(object_features).all() or (object_features[:,0]<=0).any()):
                raise ValueError("v3 actor requires valid estimated object features")
            condition=object_features[:,None].expand(-1,group.shape[1],-1)
            contact=contact+r.object_condition_head(torch.cat((group,condition),-1)).squeeze(-1)
        temporal = r.request_head(temporal_input).squeeze(-1)
        if self.temporal_options is not None:
            count = len(self.temporal_options['durations_s'])
            if (hold_indices is None or hold_indices.shape != mask.shape
                    or (hold_indices < 0).any() or (hold_indices > count).any()
                    or (hold_indices[initial] != 0).any()):
                raise ValueError('duration policy requires recorded hold options')
            hold = (hold_indices > 0) & mask
            moving = mask & ~hold
            # Relative transition ranking is preserved; keep duration gets its
            # own learned, state-dependent probability instead of ten rare draws.
            reference_scores = temporal.masked_fill(~moving, -torch.inf)
            reference_scores = torch.where(moving.any(-1)[:,None],reference_scores,torch.zeros_like(reference_scores))
            reference = reference_scores.logsumexp(-1)
            reference = torch.where(moving.any(-1), reference, torch.zeros_like(reference))
            residual = r.hold_head(temporal_input).gather(-1,(hold_indices-1).clamp_min(0)[...,None]).squeeze(-1)
            prior = self.temporal_options['initial_wait_probability']
            option_count = hold.sum(-1,keepdim=True).clamp_min(1).to(temporal.dtype)
            keep_scores = reference[:,None] + math.log(prior/(1-prior)) - option_count.log() + residual
            temporal = torch.where(hold, keep_scores, temporal)
        scores = torch.where(initial[:, None], contact, temporal).masked_fill(
            ~mask, -torch.inf
        )
        if not bool(mask.any(-1).all()) or not torch.isfinite(scores[mask]).all():
            raise ValueError("invalid categorical support/logits")
        value_features = self._value_features(normalized, mask, graph, hold_indices)
        if self.motion_feedback:
            value_features = torch.cat((value_features, motion_features), -1)
        if self.temporal_object_conditions:
            value_features = torch.cat((value_features, object_features), -1)
        value = self.value_head(value_features).squeeze(-1)
        if self.fixed_actor is not None:
            learning = initial if self.trainable_branch == 'contact' else ~initial
            if bool((~learning).any()):
                self.fixed_actor.eval()
                with torch.no_grad():
                    fixed_scores, _ = self.fixed_actor.forward_encoded(
                        features, morphology, mask, candidates=candidates,
                        membership=membership, initial=initial, owners=owners,
                        object_features=object_features, anchor_distances=anchor_distances,
                        hold_indices=hold_indices, motion_features=motion_features)
                scores = torch.where(learning[:, None], scores, fixed_scores)
        return scores, value

    @staticmethod
    def _value_features(normalized, mask, graph, hold_indices=None):
        weights = mask.to(normalized.dtype)
        if hold_indices is not None:
            # Repeated duration options share one observation: do not count the
            # same request five times in the critic's pooled state.
            holds = hold_indices > 0
            count = hold_indices.amax(-1,keepdim=True).clamp_min(1)
            weights = torch.where(holds, weights/count, weights)
        mean = (normalized * weights[:, :, None]).sum(1) / weights.sum(1, keepdim=True)
        return torch.cat((mean, graph), -1)

    @torch.no_grad()
    def frozen_value_features(self, features, morphology, mask, hold_indices=None, motion_features=None, object_features=None):
        """Inputs reusable only while the deterministic shared encoder is frozen."""
        if self.training or any(p.requires_grad for p in self.ranker.parameters()):
            raise ValueError("value feature reuse requires an eval-mode frozen ranker")
        graph = self.ranker.morphology_encoder(morphology).graph_embeddings
        normalized = (features - self.ranker.feature_mean) / self.ranker.feature_scale
        result = self._value_features(normalized, mask, graph, hold_indices)
        if self.motion_feedback:
            result = torch.cat((result, motion_features), -1)
        if self.temporal_object_conditions:
            if (object_features is None or object_features.shape != (len(features), 4)
                    or not torch.isfinite(object_features).all() or (object_features[:, 0] <= 0).any()):
                raise ValueError('temporal value requires estimated object features')
            result = torch.cat((result, object_features), -1)
        return result

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
            object_features=None if "object_features" not in encoded else encoded["object_features"][None].to(device),
            anchor_distances=None if "anchor_distances" not in encoded else encoded["anchor_distances"][None].to(device),
            hold_indices=None if "hold_indices" not in encoded else encoded["hold_indices"][None].to(device),
            motion_features=None if "motion_features" not in encoded else encoded["motion_features"][None].to(device),
            **({'clock_features': encoded['clock_features'][None].to(device)}
               if 'clock_features' in encoded else {}),
        )

    @torch.no_grad()
    def decide(self, context, *, deterministic=False, generator=None):
        return self.decide_many(
            context, generators=[generator], deterministic=deterministic
        )[0]

    @torch.no_grad()
    def decide_many(self, context, *, generators, deterministic=False):
        """Independent draws from one current observation, without an inference cache.

        The collector requests several replicas of the same initial condition.
        Encode that immutable snapshot once; each draw still consumes its own
        generator through the canonical semantic sampling order.
        """
        self.eval()
        encoded = self.encode(context)
        scores, value = self.evaluate_encoding(encoded)
        dist = Categorical(logits=scores[0])
        return [self._decision(context, encoded, scores, value, dist,
                               deterministic=deterministic, generator=generator)
                for generator in generators]

    def _decision(self, context, encoded, scores, value, dist, *, deterministic, generator):
        ranking = None
        indices = encoded.get('request_indices', torch.arange(len(context.catalog.entries)))
        entries = [context.catalog.entries[int(i)] for i in indices]
        if self.ranked_contact is not None and bool(encoded['initial']):
            remaining = scores[0].clone()
            ranking = []
            while torch.isfinite(remaining).any():
                index = (int(remaining.argmax()) if deterministic else
                         sample_request_index(entries, remaining, generator=generator))
                ranking.append(index)
                remaining[index] = -torch.inf
            prefix = ranking[:self.ranking_prefix_length]
            if len(prefix) != self.ranking_prefix_length:
                raise ValueError('ranking prefix exceeds legal candidates')
            from amsrr.policies.request_ranking import prefix_log_probability
            index = prefix[-1]
            log_prob = float(prefix_log_probability(scores[0], prefix))
        else:
            index = (int(scores[0].argmax()) if deterministic else
                     sample_request_index(entries, scores[0], generator=generator, hold_indices=encoded.get('hold_indices')))
            log_prob = float(dist.log_prob(torch.tensor(index, device=scores.device)))
        result = dict(
            request=deepcopy(entries[index].request), action=index,
            log_prob=log_prob, value=float(value[0]), encoded=encoded,
            snapshot_hash=context.catalog.snapshot_hash,
            requests=[e.request.to_dict() for e in entries],
            logits=scores[0].cpu(), sampling_order=SAMPLING_ORDER_VERSION)
        if self.temporal_options is not None:
            hold = int(encoded['hold_indices'][index])
            result['hold_duration_s'] = (0. if hold == 0 else self.temporal_options['durations_s'][hold-1])
        if ranking is not None:
            result.update(ranking_order=ranking, ranking_prefix=prefix,
                          planning_attempt_cost=self.ranked_contact['attempt_cost'])
        return result

    def rank_with_scores(self, context):
        with torch.no_grad():
            encoded = self.encode(context)
            scores, _ = self.evaluate_encoding(encoded)
            indices = encoded.get('request_indices', torch.arange(len(context.catalog.entries)))
            ids = [
                i
                for i in scores[0].argsort(descending=True, stable=True).tolist()
                if torch.isfinite(scores[0, i])
            ]
            return (
                [deepcopy(context.catalog.entries[int(indices[i])].request) for i in ids],
                [float(scores[0, i]) for i in ids],
            )

    def checkpoint(self):
        payload = dict(
            version=self.version,
            contract_version=HIGH_LEVEL_REQUEST_CONTRACT,
            feature_version=REQUEST_FEATURE_VERSION,
            config=self.ranker.config.to_dict(),
            **({'object_feature_version':OBJECT_FEATURE_VERSION} if self.object_condition_aware else {}),
            **(dict(anchor_preference=self.anchor_preference.to_dict(),
                    base_actor_version=(OBJECT_ACTOR_CRITIC_VERSION if self.object_condition_aware else
                        MORPHOLOGY_ACTOR_CRITIC_VERSION if self.morphology_aware else ACTOR_CRITIC_VERSION))
               if self.anchor_preference is not None else {}),
            **({'ranked_contact': self.ranked_contact} if self.ranked_contact is not None else {}),
            **({'temporal_options': self.temporal_options} if self.temporal_options is not None else {}),
            **({'contact_return_ranking': self.contact_return_ranking} if self.contact_return_ranking is not None else {}),
            **({'motion_feedback': 'observed_contact_motion_v1'} if self.motion_feedback else {}),
            **({'temporal_object_conditions': OBJECT_FEATURE_VERSION} if self.temporal_object_conditions else {}),
            state_dict={
                k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                if not k.startswith('fixed_actor.')
            },
        )
        if self.fixed_actor is None:
            return payload
        return dict(version=SPLIT_ACTOR_VERSION, contract_version=HIGH_LEVEL_REQUEST_CONTRACT,
                    feature_version=REQUEST_FEATURE_VERSION, trainable_branch=self.trainable_branch,
                    trainable_actor=payload, fixed_actor=self.fixed_actor.checkpoint())

    @classmethod
    def from_checkpoint(cls, checkpoint):
        if checkpoint.get('version') in ('causal_request_double_q_v1', 'causal_request_set_conditioned_double_q_v2'):
            from amsrr.policies.request_q_policy import RequestQPolicy
            return RequestQPolicy.from_checkpoint(checkpoint)
        if checkpoint.get('version') == SPLIT_ACTOR_VERSION:
            if (checkpoint.get('contract_version') != HIGH_LEVEL_REQUEST_CONTRACT
                    or checkpoint.get('feature_version') != REQUEST_FEATURE_VERSION
                    or checkpoint['trainable_actor'].get('version') == SPLIT_ACTOR_VERSION
                    or checkpoint['fixed_actor'].get('version') == SPLIT_ACTOR_VERSION):
                raise SchemaValidationError('incompatible independent fixed actor checkpoint')
            model = cls.from_checkpoint(checkpoint['trainable_actor'])
            model.enable_fixed_branch(cls.from_checkpoint(checkpoint['fixed_actor']),
                                      trainable_branch=checkpoint['trainable_branch'])
            return model
        if (
            checkpoint.get("version") not in {ACTOR_CRITIC_VERSION, MORPHOLOGY_ACTOR_CRITIC_VERSION, OBJECT_ACTOR_CRITIC_VERSION, ANCHOR_PRIOR_ACTOR_CRITIC_VERSION}
            or checkpoint.get("contract_version") != HIGH_LEVEL_REQUEST_CONTRACT
            or checkpoint.get("feature_version") != REQUEST_FEATURE_VERSION
        ):
            raise SchemaValidationError("incompatible event actor/critic checkpoint")
        preference = checkpoint['version'] == ANCHOR_PRIOR_ACTOR_CRITIC_VERSION
        base = checkpoint.get('base_actor_version') if preference else checkpoint['version']
        if base not in {ACTOR_CRITIC_VERSION, MORPHOLOGY_ACTOR_CRITIC_VERSION, OBJECT_ACTOR_CRITIC_VERSION}:
            raise SchemaValidationError('incompatible anchor-preference base actor')
        if not preference and 'anchor_preference' in checkpoint:
            raise SchemaValidationError('anchor preference requires its versioned checkpoint')
        object_aware=base==OBJECT_ACTOR_CRITIC_VERSION
        if object_aware and checkpoint.get('object_feature_version')!=OBJECT_FEATURE_VERSION:
            raise SchemaValidationError('incompatible object estimate feature contract')
        model = cls(RequestHighLevelPolicyConfig.from_dict(checkpoint["config"]),
                    morphology_aware=base != ACTOR_CRITIC_VERSION,
                    object_condition_aware=object_aware)
        if preference:
            if 'anchor_preference' not in checkpoint:
                raise SchemaValidationError('missing checkpoint anchor preference')
            model.enable_anchor_preference(AnchorPreference(**checkpoint['anchor_preference']))
        if 'ranked_contact' in checkpoint:
            from amsrr.policies.request_ranking import validate_ranking_config
            model.ranked_contact = validate_ranking_config(checkpoint['ranked_contact'])
        if 'temporal_options' in checkpoint:
            model.enable_temporal_options(checkpoint['temporal_options'])
        if 'motion_feedback' in checkpoint:
            if checkpoint['motion_feedback'] != 'observed_contact_motion_v1':
                raise ValueError('unsupported motion feedback contract')
            model.enable_motion_feedback()
        if 'temporal_object_conditions' in checkpoint:
            if checkpoint['temporal_object_conditions'] != OBJECT_FEATURE_VERSION:
                raise ValueError('unsupported temporal object feature contract')
            model.enable_temporal_object_conditions()
        if 'contact_return_ranking' in checkpoint:
            config=checkpoint['contact_return_ranking']
            if (set(config)!={'version','weight','min_samples'}
                    or config['version']!='condition_success_return_pairs_v1'
                    or not math.isfinite(config['weight']) or config['weight']<=0
                    or not isinstance(config['min_samples'],int) or config['min_samples']<2):
                raise ValueError('invalid contact return ranking contract')
            model.contact_return_ranking=dict(config)
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
