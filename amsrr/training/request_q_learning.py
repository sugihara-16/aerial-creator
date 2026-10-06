"""Experience replay and physical-time Double DQN for variable request actions."""
from copy import deepcopy
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from amsrr.policies.request_actor_critic import RequestActorCritic
from amsrr.policies.request_q_policy import RequestQPolicy, clock_features
from amsrr.training.request_ppo import collate_transitions, evaluate_batch, discounted_event_returns
from amsrr.utils.hashing import hash_file

TRAINABLE_PREFIXES = ('q_contact.', 'q_temporal.', 'ranker.morphology_encoder.', 'ranker.contact_member.')


def migrate_action_context(checkpoint, output):
    """Add remaining-action context once, retaining every old weight and moment.

    New projections start at zero, so the Q function initially equals the old
    scorer. Adam state is remapped by parameter name, not shifted numeric IDs.
    The target network receives the same zero context; its old lag is retained.
    """
    checkpoint, output = Path(checkpoint), Path(output)
    model = RequestActorCritic.load(checkpoint)
    if not isinstance(model, RequestQPolicy) or model.q_config.get('candidate_context', False):
        raise ValueError('context migration requires a legacy Q checkpoint')
    saved = torch.load(checkpoint.parent/'q_training_state.pt', map_location='cpu', weights_only=True)
    if saved['checkpoint_sha256'] != hash_file(checkpoint):
        raise ValueError('Q optimizer state checkpoint mismatch')
    old_names = [n for n, _ in model.named_parameters() if n.startswith(TRAINABLE_PREFIXES)]
    optimizer = deepcopy(saved['optimizer'])
    if len(optimizer['param_groups']) != 1 or len(optimizer['param_groups'][0]['params']) != len(old_names):
        raise ValueError('unsupported Q optimizer parameter layout')
    old_ids = dict(zip(old_names, optimizer['param_groups'][0]['params'], strict=True))
    if set(optimizer['state']) - set(old_ids.values()):
        raise ValueError('unbound Q optimizer state')
    target = deepcopy(model)
    target.load_state_dict(saved['target'], strict=True)
    old_weights = compact(model.state_dict())
    model.enable_candidate_context(); target.enable_candidate_context()
    new_names = [n for n, _ in model.named_parameters() if n.startswith(TRAINABLE_PREFIXES)]
    optimizer['state'] = {i: optimizer['state'][old_ids[name]]
        for i, name in enumerate(new_names)
        if name in old_ids and old_ids[name] in optimizer['state']}
    optimizer['param_groups'][0]['params'] = list(range(len(new_names)))
    for name, value in old_weights.items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
        torch.testing.assert_close(saved['target'][name], target.state_dict()[name], rtol=0, atol=0)
    output.mkdir(parents=True, exist_ok=False)
    torch.save(model.checkpoint(), output/'checkpoint.pt')
    torch.save(dict(saved, checkpoint_sha256=hash_file(output/'checkpoint.pt'),
                    optimizer=optimizer, target=target.state_dict()), output/'q_training_state.pt')
    report = dict(source=str(checkpoint), source_sha256=hash_file(checkpoint),
        checkpoint_sha256=hash_file(output/'checkpoint.pt'),
        old_weights_and_target_exact=True, old_optimizer_moments_preserved=True,
        added_parameters=[n for n in new_names if n not in old_ids],
        prior_optimizer_steps=saved['steps'],
        policy_change='Conditional ranking recomputes Q and exploration on the remaining action set.')
    (output/'migration.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def compact(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(compact(v) for v in value)
    return value


def episode_transitions(episode, *, deadlines, gamma=.99, attempt_cost=.25):
    """Expand actual planner attempts into zero-time MDP steps, never invent trials.

    Only failed prefix members are followed by the same state with that action
    removed. The final member receives the physical outcome. Their sum equals
    the recorded episode reward, including the original planning charge.
    """
    if not episode or not episode[-1]['done'] or any(e['done'] for e in episode[:-1]):
        raise ValueError('replay requires one complete physical episode')
    # Also validates causal physical reward timestamps and their recorded sum.
    original_returns = discounted_event_returns(episode, gamma)
    states = []
    for event in episode:
        encoding = compact(event['encoding'])
        expected = clock_features(float(event['time_s']), deadlines)
        if 'clock_features' in encoding:
            torch.testing.assert_close(encoding['clock_features'], expected, rtol=0, atol=1e-7)
        encoding['clock_features'] = expected
        states.append(encoding)
    rows = []
    for index, event in enumerate(episode):
        reward = sum(gamma ** (float(t) - float(event['time_s'])) * float(r)
                     for t, r in event['reward_events'])
        discount = 0. if event['done'] else gamma ** (float(event['end_time_s']) - float(event['time_s']))
        next_state = None if event['done'] else states[index + 1]
        prefix = event.get('ranking_prefix', [event['action']])
        if (not prefix or len(set(prefix)) != len(prefix) or prefix[-1] != event['action']
                or not states[index]['mask'][prefix].all()):
            raise ValueError('invalid executed action prefix')
        if 'ranking_prefix' in event:
            if (event.get('planning_attempts') != len(prefix)
                    or not math.isclose(event.get('planning_cost', -1), attempt_cost * len(prefix))):
                raise ValueError('recorded planner prefix charge mismatch')
        elif len(prefix) != 1:
            raise ValueError('non-planning action has a prefix')
        state = states[index]
        for j, action in enumerate(prefix):
            if j + 1 < len(prefix):
                following = dict(state, mask=state['mask'].clone())
                following['mask'][action] = False
                rows.append(dict(encoding=state, action=action, reward=-attempt_cost,
                                 discount=1., next_encoding=following))
                state = following
            else:
                rows.append(dict(encoding=state, action=action,
                    reward=reward + attempt_cost * (len(prefix) - 1),
                    discount=discount, next_encoding=next_state))
    value = 0.
    for row in reversed(rows):
        value = row['reward'] + row['discount'] * value
        row['mc_return'] = value
    if not math.isclose(rows[0]['mc_return'], original_returns[0], abs_tol=1e-6):
        raise ValueError('planner expansion changed discounted episode return')
    return rows


def double_q_target(reward, discount, online_next, target_next):
    """Select with the current net, evaluate with the slowly moving target net."""
    actions = online_next.argmax(-1)
    value = target_next.gather(-1, actions[:, None]).squeeze(-1)
    value = torch.where(discount > 0, value, torch.zeros_like(value))
    return reward + discount * value


def conservative_q_loss(values, actions):
    """Discrete CQL(H): penalize optimistic alternatives absent from replay.

    Kumar et al., NeurIPS 2020, equation 4 (uniform action prior). Invalid
    actions have -inf values and are excluded from the log-sum-exp integral.
    """
    selected = values.gather(-1, actions[:, None]).squeeze(-1)
    return (values.logsumexp(-1) - selected).mean()


def q_regression_loss(predicted, expected, mode):
    # With CQL's behavior-frequency regularizer, clipping the TD derivative
    # can prevent a rare, high-return action from outranking a frequent failure
    # even in a two-action tabular problem. Use the squared Bellman error from
    # CQL's objective; gradient-norm clipping still bounds optimizer steps.
    if mode == 'double_q':
        return .5 * F.mse_loss(predicted, expected)
    if mode == 'monte_carlo':
        return F.smooth_l1_loss(predicted, expected)
    raise ValueError('unknown Q regression stage')


def load_compatible_replay(replay_paths):
    """Keep one physical execution contract across the entire replay buffer."""
    rows, provenance, runtime = [], [], None
    for path in replay_paths:
        replay = torch.load(path, map_location='cpu', weights_only=True)
        if replay['split'] != 'train' or replay['version'] != 'request_physical_q_replay_v1':
            raise ValueError('Q replay must be explicitly bound training experience')
        contract = replay.get('runtime_contracts')
        if not isinstance(contract, dict) or not contract:
            raise ValueError('Q replay lacks physical execution contracts')
        if runtime is not None and contract != runtime:
            raise ValueError('Q replay mixes physical execution contracts')
        runtime = contract
        rows.extend(replay['transitions'])
        provenance.append(dict(path=str(path), sha256=hash_file(path), count=len(replay['transitions'])))
    if not rows:
        raise ValueError('empty Q replay')
    return rows, provenance, runtime


def train_q(checkpoint, replay_paths, output, *, steps=2000, batch_size=128,
            learning_rate=1e-4, target_tau=.005, mode='double_q', seed=4105, device='cpu',
            conservative_weight=0.):
    """One reproducible optimizer stage. MC warmup precedes online Double DQN."""
    if mode not in {'monte_carlo', 'double_q'} or steps < 1 or batch_size < 2 or batch_size % 2:
        raise ValueError('invalid Q training stage')
    if not math.isfinite(conservative_weight) or conservative_weight < 0:
        raise ValueError('invalid conservative Q weight')
    torch.manual_seed(seed)
    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    actor = RequestActorCritic.load(checkpoint)
    model = actor if isinstance(actor, RequestQPolicy) else RequestQPolicy(actor)
    model.to(device).eval()
    # Old policy/value readouts are retained for checkpoint provenance only.
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(TRAINABLE_PREFIXES))
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    target = deepcopy(model).eval()
    for p in target.parameters():
        p.requires_grad_(False)
    state_path = Path(checkpoint).parent / 'q_training_state.pt'
    prior_steps = 0
    regression_contract = 'squared_bellman_huber_initialization_v1'
    if state_path.exists():
        saved = torch.load(state_path, map_location=device, weights_only=True)
        if saved['checkpoint_sha256'] != hash_file(checkpoint):
            raise ValueError('Q optimizer state checkpoint mismatch')
        if saved['learning_rate'] != learning_rate or saved['target_tau'] != target_tau:
            raise ValueError('unannounced Q optimizer setting change')
        if saved.get('mode') == mode and mode == 'double_q' and saved.get('regression_contract') != regression_contract:
            raise ValueError('Q regression objective changed; start a declared comparison from the MC initializer')
        if ('conservative_weight' in saved and saved.get('mode') == mode
                and saved['conservative_weight'] != conservative_weight):
            raise ValueError('unannounced conservative Q setting change')
        optimizer.load_state_dict(saved['optimizer'])
        target.load_state_dict(saved['target'])
        prior_steps = saved['steps']
    rows, provenance, runtime_contracts = load_compatible_replay(replay_paths)
    initial = torch.tensor([bool(r['encoding']['initial']) for r in rows])
    groups = [initial.nonzero().flatten(), (~initial).nonzero().flatten()]
    if any(len(g) == 0 for g in groups):
        raise ValueError('Q replay requires both contact and temporal decisions')
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for step in range(steps):
        # Stratified experience replay prevents repeated temporal decisions
        # from making initial contact learning vanish as tasks get longer.
        ids = torch.cat([g[torch.randint(len(g), (batch_size//2,), generator=generator)] for g in groups])
        batch = [rows[int(i)] for i in ids]
        data = collate_transitions(batch)
        local = torch.arange(len(batch))
        values, _ = evaluate_batch(model, data, local, device, action_values=True)
        actions = torch.tensor([r['action'] for r in batch], device=device)
        predicted = values.gather(-1, actions[:, None]).squeeze(-1)
        if mode == 'monte_carlo':
            expected = torch.tensor([r['mc_return'] for r in batch], device=device)
        else:
            next_data = collate_transitions([dict(encoding=r['next_encoding'] or r['encoding']) for r in batch])
            with torch.no_grad():
                online_next, _ = evaluate_batch(model, next_data, local, device, action_values=True)
                target_next, _ = evaluate_batch(target, next_data, local, device, action_values=True)
                expected = double_q_target(torch.tensor([r['reward'] for r in batch], device=device),
                    torch.tensor([r['discount'] for r in batch], device=device), online_next, target_next)
        td_loss = q_regression_loss(predicted, expected, mode)
        conservative_loss = conservative_q_loss(values, actions)
        loss = td_loss + conservative_weight * conservative_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 10., error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for a, b in zip(target.parameters(), model.parameters()):
                a.lerp_(b, target_tau)
        if step % 100 == 0 or step + 1 == steps:
            row = dict(step=step+1, loss=float(loss.detach()), prediction_mean=float(predicted.detach().mean()),
                       target_mean=float(expected.mean()), td_loss=float(td_loss.detach()),
                       conservative_loss=float(conservative_loss.detach()), seconds=time.monotonic()-started)
            losses.append(row)
            print(json.dumps(row), flush=True)
    torch.save(model.checkpoint(), output/'checkpoint.pt')
    torch.save(dict(checkpoint_sha256=hash_file(output/'checkpoint.pt'), optimizer=optimizer.state_dict(),
                    target=target.state_dict(), learning_rate=learning_rate, target_tau=target_tau,
                    conservative_weight=conservative_weight, mode=mode,
                    regression_contract=regression_contract,
                    steps=prior_steps+steps), output/'q_training_state.pt')
    restored = RequestActorCritic.load(output/'checkpoint.pt').eval()
    # CUDA graph reductions may differ by a few ulps between identical calls.
    # Require exact persisted weights and exact deterministic CPU replay.
    model.to('cpu')
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(evaluate_batch(model, data, local, 'cpu'),
                                   evaluate_batch(restored, data, local, 'cpu'), rtol=0, atol=0)
    report = dict(mode=mode, steps=steps, prior_steps=prior_steps, seed=seed, learning_rate=learning_rate,
        target_tau=target_tau, conservative_weight=conservative_weight,
        regression_contract=regression_contract,
        episodes_source='verified training-only replay manifests', replay=provenance,
        runtime_contracts=runtime_contracts,
        transitions=len(rows), contact_transitions=int(initial.sum()), temporal_transitions=int((~initial).sum()),
        losses=losses, seconds=time.monotonic()-started, reload_exact=True)
    (output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
