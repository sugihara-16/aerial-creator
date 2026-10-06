"""Sampled without-replacement ranking, stopped at the first accepted plan.

The action is the tried prefix, not just its last (executed) contact assignment.
Untried suffixes consume RNG for reproducibility but never enter PPO credit.
"""
import math
import torch
from torch.distributions import Categorical


def validate_ranking_config(config):
    if (set(config) != {'version', 'attempt_cost'}
            or config['version'] not in ('first_feasible_plackett_luce_v1', 'first_feasible_autoregressive_q_v2')
            or not math.isfinite(config['attempt_cost']) or config['attempt_cost'] <= 0):
        raise ValueError('invalid contact ranking contract')
    return dict(config)


def prefix_log_probability(logits, prefix):
    if not prefix or len(prefix) != len(set(prefix)):
        raise ValueError('empty or repeated ranking prefix')
    available = torch.isfinite(logits)
    terms = []
    for index in prefix:
        if index < 0 or index >= len(logits) or not bool(available[index]):
            raise ValueError('illegal ranking action')
        masked = logits.masked_fill(~available, -torch.inf)
        terms.append(masked[index] - torch.logsumexp(masked, -1))
        available = available.clone()
        available[index] = False
    return torch.stack(terms).sum()


def action_statistics(logits, events, reference=None):
    """Joint prefix probability and conditional entropy/KL at visited choices.

    KL is the sum of conditional categorical KL along the sampled old-policy
    prefix. Its rollout average estimates the stopped-ranking policy KL.
    """
    dist = Categorical(logits=logits)
    actions = torch.tensor([e['action'] for e in events], device=logits.device)
    log_probs = dist.log_prob(actions)
    entropies = dist.entropy()
    from amsrr.training.request_ppo import categorical_kl_from_logits
    divergences = (None if reference is None else
                   categorical_kl_from_logits(reference, logits))
    for i, event in enumerate(events):
        prefix = event.get('ranking_prefix')
        if prefix is None:
            continue
        if not bool(event['encoding']['initial']) or prefix[-1] != event['action']:
            raise ValueError('ranking must describe the initial executed assignment')
        log_probs[i] = prefix_log_probability(logits[i], prefix)
        available = torch.isfinite(logits[i])
        entropy, divergence = [], []
        for index in prefix:
            current = Categorical(logits=logits[i].masked_fill(~available, -torch.inf))
            entropy.append(current.entropy())
            if reference is not None:
                divergence.append(categorical_kl_from_logits(
                    reference[i].masked_fill(~available, -torch.inf),
                    logits[i].masked_fill(~available, -torch.inf)))
            available = available.clone(); available[index] = False
        entropies[i] = torch.stack(entropy).sum()
        if reference is not None:
            divergences[i] = torch.stack(divergence).sum()
    return log_probs, entropies, divergences


def record_ranking(event, decision):
    if 'ranking_prefix' not in decision:
        return
    prefix = decision['ranking_prefix']
    cost = decision['planning_attempt_cost'] * len(prefix)
    event.update(ranking_prefix=list(prefix), planning_attempts=len(prefix),
                 planning_cost=cost)
    if 'ranking_policy_version' in decision:
        event['ranking_policy_version'] = decision['ranking_policy_version']
    event['reward'] -= cost
    event['reward_events'].insert(0, (event['time_s'], -cost))
