"""Variable-action Double-DQN policy using the request actor's causal encoder.

The downstream planner, contact binding and transition guards are unchanged.
Q(s,a) predicts discounted physical reward. Training explores categorically;
deployment orders candidates by Q, with no case-specific action corrections.
"""
from copy import deepcopy
import math

import torch
from torch import nn

from amsrr.policies.request_actor_critic import RequestActorCritic

Q_POLICY_VERSION = 'causal_request_double_q_v1'
CONTEXTUAL_Q_POLICY_VERSION = 'causal_request_set_conditioned_double_q_v2'
CONTEXTUAL_RANKING_VERSION = 'first_feasible_autoregressive_q_v2'


class ActionSetQHead(nn.Sequential):
    """Deep Sets context for the remaining legal actions, without slot IDs.

    The zero-initialized context projection preserves an existing local scorer.
    Mean pooled nonlinear action features plus cardinality distinguish states
    after a rejected planner attempt, including the last remaining candidate.
    """
    def __init__(self, local_head):
        super().__init__(*list(local_head.children()))
        hidden = self[1].out_features
        self.context = nn.Linear(hidden + 1, hidden, bias=False)
        self.context.to(device=self[1].weight.device, dtype=self[1].weight.dtype)
        nn.init.zeros_(self.context.weight)

    def forward(self, features, mask):
        local = self[2](self[1](self[0](features)))
        count = mask.sum(-1, keepdim=True).to(local.dtype)
        pooled = local.masked_fill(~mask[..., None], 0).sum(1) / count.clamp_min(1)
        context = torch.cat((pooled, count.log1p()), -1)
        conditioned = local + self.context(context)[:, None]
        return self[5](self[4](self[3](conditioned)))


def clock_features(time_s, deadlines):
    deadlines = [float(x) for x in deadlines if x is not None]
    if (not math.isfinite(time_s) or time_s < 0
            or any(not math.isfinite(x) or x <= 0 for x in deadlines)):
        raise ValueError('invalid observed task clock or authored deadlines')
    return torch.tensor([time_s / 100., min(deadlines, default=0.) / 100.,
                         max(deadlines, default=0.) / 100., float(bool(deadlines))])


class RequestQPolicy(RequestActorCritic):
    def __init__(self, actor, *, epsilon=.2, temperature=.25, candidate_context=False):
        if (actor.fixed_actor is not None or actor.anchor_preference is not None
                or not actor.temporal_object_conditions or actor.temporal_options is None):
            raise ValueError('Q migration requires a causal joint actor with object and duration inputs')
        super().__init__(deepcopy(actor.ranker.config), morphology_aware=actor.morphology_aware,
                         object_condition_aware=actor.object_condition_aware)
        self.ranker = deepcopy(actor.ranker)
        self.value_head = deepcopy(actor.value_head)
        for name in ('version', 'ranked_contact', 'temporal_options', 'motion_feedback',
                     'temporal_object_conditions'):
            setattr(self, name, deepcopy(getattr(actor, name)))
        self.q_config = dict(epsilon=float(epsilon), temperature=float(temperature))
        if not 0 < epsilon < 1 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('invalid Q exploration configuration')
        width = self.ranker.request_head[0].in_features + 5
        # Same scorer for every member of each variable-size candidate set.
        def head():
            return nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 192), nn.SiLU(),
                                 nn.Linear(192, 96), nn.SiLU(), nn.Linear(96, 1))
        self.q_contact = head()
        self.q_temporal = head()
        if candidate_context:
            self.enable_candidate_context()

    def enable_candidate_context(self):
        if self.q_config.get('candidate_context', False):
            raise ValueError('candidate-set context is already enabled')
        self.q_contact = ActionSetQHead(self.q_contact)
        self.q_temporal = ActionSetQHead(self.q_temporal)
        self.q_config['candidate_context'] = True
        if self.ranked_contact is not None:
            self.ranked_contact = dict(self.ranked_contact, version=CONTEXTUAL_RANKING_VERSION)

    def encode(self, context):
        result = super().encode(context)
        result['clock_features'] = clock_features(
            context.observation.time_s, [g.time_limit_s for g in context.task_spec.goals])
        return result

    def q_values(self, features, morphology, mask, *, candidates, membership, initial,
                 owners=None, object_features=None, anchor_distances=None,
                 hold_indices=None, motion_features=None, clock_features=None):
        if (clock_features is None or clock_features.shape != (len(features), 4)
                or not torch.isfinite(clock_features).all()):
            raise ValueError('Q policy requires the observed task clock and authored deadlines')
        if object_features is None or not torch.isfinite(object_features).all():
            raise ValueError('Q policy requires causal estimated object properties')
        _, _, _, x = self.action_features(features, morphology, candidates=candidates,
            membership=membership, owners=owners, object_features=object_features,
            motion_features=motion_features, normalization_clip=10.)
        if hold_indices is None or hold_indices.shape != mask.shape:
            raise ValueError('Q policy requires duration action identities')
        duration = x.new_tensor([0., *self.temporal_options['durations_s']])[hold_indices]
        x = torch.cat((x, clock_features[:, None].expand(-1, x.shape[1], -1),
                       torch.log1p(duration)[..., None]), -1)
        if self.q_config.get('candidate_context', False):
            contact = self.q_contact(x, mask).squeeze(-1)
            temporal = self.q_temporal(x, mask).squeeze(-1)
        else:
            contact = self.q_contact(x).squeeze(-1)
            temporal = self.q_temporal(x).squeeze(-1)
        values = torch.where(initial[:, None], contact, temporal).masked_fill(~mask, -torch.inf)
        if not mask.any(-1).all() or not torch.isfinite(values[mask]).all():
            raise ValueError('invalid Q values or empty action support')
        return values, values.max(-1).values

    def forward_encoded(self, *args, **kwargs):
        values, state_value = self.q_values(*args, **kwargs)
        valid = torch.isfinite(values)
        probability = (1 - self.q_config['epsilon']) * (
            values / self.q_config['temperature']).softmax(-1)
        probability = probability + self.q_config['epsilon'] * valid / valid.sum(-1, keepdim=True)
        return probability.log().masked_fill(~valid, -torch.inf), state_value

    def _decision(self, context, encoded, scores, value, dist, *, deterministic, generator):
        if not (self.q_config.get('candidate_context', False)
                and self.ranked_contact is not None and bool(encoded['initial'])):
            return super()._decision(context, encoded, scores, value, dist,
                                    deterministic=deterministic, generator=generator)
        from amsrr.policies.request_actor_critic import sample_request_index, SAMPLING_ORDER_VERSION
        indices = encoded.get('request_indices', torch.arange(len(context.catalog.entries)))
        entries = [context.catalog.entries[int(i)] for i in indices]
        available = encoded['mask'].clone()
        ranking, terms = [], []
        while bool(available.any()):
            conditional, _ = self.evaluate_encoding(dict(encoded, mask=available))
            logits = conditional[0]
            index = (int(logits.argmax()) if deterministic else
                     sample_request_index(entries, logits, generator=generator))
            ranking.append(index)
            terms.append(logits[index] - logits.logsumexp(-1))
            available = available.clone(); available[index] = False
        prefix = ranking[:self.ranking_prefix_length]
        if not prefix or len(prefix) != self.ranking_prefix_length:
            raise ValueError('ranking prefix exceeds legal candidates')
        index = prefix[-1]
        return dict(request=deepcopy(entries[index].request), action=index,
                    log_prob=float(torch.stack(terms[:len(prefix)]).sum()), value=float(value[0]),
                    encoded=encoded, snapshot_hash=context.catalog.snapshot_hash,
                    requests=[e.request.to_dict() for e in entries], logits=scores[0].cpu(),
                    sampling_order=SAMPLING_ORDER_VERSION, hold_duration_s=0.,
                    ranking_order=ranking, ranking_prefix=prefix,
                    ranking_conditional_log_probs=[float(term) for term in terms],
                    ranking_policy_version=CONTEXTUAL_RANKING_VERSION,
                    planning_attempt_cost=self.ranked_contact['attempt_cost'])

    def rank_with_scores(self, context):
        if not (self.q_config.get('candidate_context', False) and self.ranked_contact is not None):
            return super().rank_with_scores(context)
        result = self.decide(context, deterministic=True)
        if 'ranking_order' not in result:
            return super().rank_with_scores(context)
        from amsrr.schemas.high_level import HighLevelRequest
        return ([HighLevelRequest.from_dict(result['requests'][i]) for i in result['ranking_order']],
                result['ranking_conditional_log_probs'])

    def ranking_log_probability(self, encoding, prefix, *, batched=True):
        """Replay the conditional policy at every actually attempted action."""
        from amsrr.training.request_ppo import collate_transitions, evaluate_batch
        if not prefix or len(prefix) != len(set(prefix)):
            raise ValueError('empty or repeated ranking prefix')
        available = encoding['mask'].clone(); rows = []
        for action in prefix:
            if action < 0 or action >= len(available) or not bool(available[action]):
                raise ValueError('illegal ranking action')
            rows.append(dict(encoding=dict(encoding, mask=available)))
            available = available.clone(); available[action] = False
        device = next(self.parameters()).device
        if batched:
            logits, _ = evaluate_batch(self, collate_transitions(rows), torch.arange(len(rows)), device)
        else:
            logits = torch.cat([evaluate_batch(self, collate_transitions([row]),
                                torch.arange(1), device)[0] for row in rows])
        actions = torch.tensor(prefix, device=device)
        return (logits[torch.arange(len(rows), device=device), actions] - logits.logsumexp(-1)).sum()

    def checkpoint(self):
        actor = super().checkpoint()
        actor['state_dict'] = {k: v for k, v in actor['state_dict'].items()
                               if not k.startswith(('q_contact.', 'q_temporal.'))}
        version = CONTEXTUAL_Q_POLICY_VERSION if self.q_config.get('candidate_context', False) else Q_POLICY_VERSION
        return dict(version=version, encoder_actor=actor, q_config=self.q_config,
                    q_state={k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                             if k.startswith(('q_contact.', 'q_temporal.'))})

    @classmethod
    def from_checkpoint(cls, payload):
        if payload.get('version') not in (Q_POLICY_VERSION, CONTEXTUAL_Q_POLICY_VERSION):
            raise ValueError('incompatible Q policy checkpoint')
        if bool(payload['q_config'].get('candidate_context', False)) != (payload['version'] == CONTEXTUAL_Q_POLICY_VERSION):
            raise ValueError('Q policy version and candidate context disagree')
        actor = RequestActorCritic.from_checkpoint(payload['encoder_actor'])
        model = cls(actor, **payload['q_config'])
        expected = {k for k in model.state_dict() if k.startswith(('q_contact.', 'q_temporal.'))}
        if set(payload['q_state']) != expected:
            raise ValueError('incomplete Q policy heads')
        model.load_state_dict({**model.state_dict(), **payload['q_state']}, strict=True)
        if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
            raise ValueError('nonfinite Q policy checkpoint')
        return model
