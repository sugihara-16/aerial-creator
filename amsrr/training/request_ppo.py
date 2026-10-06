"""The same categorical event actor and observation tensors for BC and PPO."""

from dataclasses import fields
from copy import deepcopy
from pathlib import Path
import json
import math
import time

import torch
from amsrr.policies.request_ranking import action_statistics
from torch.distributions import Categorical
from torch.nn import functional as F

from amsrr.policies.request_actor_critic import RequestActorCritic, ACTOR_CRITIC_VERSION
from amsrr.training.request_imitation import (
    _pad_graphs,
    _graph_dict,
    graph_batch,
    write_json,
)
from amsrr.utils.hashing import hash_file
from amsrr.policies.anchor_preference import AnchorPreference, ANCHOR_PREFERENCE_CONTRACT


def request_runtime_contracts():
    """Action support and physical transition semantics of newly collected data."""
    # Local import avoids the executor's import of serialize_encoding at startup.
    from amsrr.policies.high_level_requests import REQUEST_ADMISSION_CONTRACT
    from amsrr.simulation.request_event_execution import ContactPointVelocityObserver, RequestEventSupervisor
    from amsrr.controllers.grasp_slip_compensation import GRASP_SLIP_CONTRACT, GRASP_POSE_CONTRACT
    from amsrr.simulation.placement_support import PLACEMENT_SUPPORT_CONTRACT
    from amsrr.policies.naive_contact_planner import NAIVE_CONTACT_PLAN_VERSION, NAIVE_CONTACT_INTERPOLATION
    from amsrr.policies.contact_candidate_sampler import CONTACT_CANDIDATE_SAMPLER_VERSION
    from amsrr.policies.contact_group_geometry import CONTACT_NUMERIC_FEATURE_CONTRACT
    from amsrr.training.order9_actuator_aware_nominal_preload import ORDER9_ACTUATOR_AWARE_NOMINAL_PRELOAD_VERSION
    from amsrr.simulation.teacher_contact_execution import NOMINAL_BODY_TRACKING_CONTRACT

    return {
        "admission": REQUEST_ADMISSION_CONTRACT,
        "contact_velocity": ContactPointVelocityObserver.contract_version,
        "contact_maintenance": RequestEventSupervisor.contact_maintenance_contract,
        "grasp_slip": GRASP_SLIP_CONTRACT,
        "grasp_pose": GRASP_POSE_CONTRACT,
        "grasp_release_reference": "accepted_feedback_clear_before_nominal_retreat_v1",
        "placement_outcome": PLACEMENT_SUPPORT_CONTRACT,
        "goal_completion": RequestEventSupervisor.goal_completion_contract,
        "nominal_plan": NAIVE_CONTACT_PLAN_VERSION,
        "contact_sampler": CONTACT_CANDIDATE_SAMPLER_VERSION,
        "contact_numeric_features": CONTACT_NUMERIC_FEATURE_CONTRACT,
        "nominal_contact_load": ORDER9_ACTUATOR_AWARE_NOMINAL_PRELOAD_VERSION,
        "nominal_body_tracking": NOMINAL_BODY_TRACKING_CONTRACT,
        "nominal_reference": NAIVE_CONTACT_INTERPOLATION,
        "nominal_twist": "world_reference_to_measured_body_qpid_v1",
        "centroidal_velocity": "all_module_link_mass_weighted_joint_jacobian_v1",
        "internal_joint_load": "tree_virtual_work_rigid_inertia_one_sided_opening_v9",
        "release_separation": "observed_finite_primitive_signed_distance_5mm_v1",
        "release_payload_handoff": "stationary_real_clock_handoff_before_opening_v1",
        "single_environment_joint_load_solver": "native_cpu_admm_roundtrip_same_iterations_v1",
        "vectoring_rate_limit": "physical_and_authored_operating_drive_minimum_v1",
        "sliding_contact_correction": "loaded_near_geometry_independent_of_slip_speed_v1",
        "contact_goal": "collision_clear_contact_before_pregrasp_v1",
        "anchor_distances": ANCHOR_PREFERENCE_CONTRACT,
        "decision_motion_feedback": "observed_contact_motion_v1",
    }


def dataset_inputs(data):
    mask = data["mask"].clone()
    for i, row in enumerate(data["rows"]):
        initial = row["initial"]
        for j, r in enumerate(row["requests"]):
            if initial:
                mask[i, j] &= (
                    r["contact_group_id"] is not None and r["transition_id"] is None
                )
            elif row["committed_group_id"] is not None:
                mask[i, j] &= r["contact_group_id"] in (None, row["committed_group_id"])
        if not mask[i, row["label"]]:
            raise ValueError("causal action support excluded demonstration")
        if initial != (row["raw_index"] == 0):
            raise ValueError("post-choice observation disguised as initial decision")
    result = dict(
        features=data["features"],
        mask=mask,
        graphs=data["graphs"],
        candidates=data["candidate_features"],
        membership=data["membership"],
        initial=torch.tensor([r["initial"] for r in data["rows"]]),
    )
    if "owners" in data:
        result["owners"] = data["owners"]
    if "object_features" in data:
        result["object_features"] = data["object_features"]
    return result


def evaluate_batch(model, data, ids, device, *, action_values=False):
    forward = model.q_values if action_values else model.forward_encoded
    return forward(
        data["features"][ids].to(device),
        graph_batch(data["graphs"], ids, device),
        data["mask"][ids].to(device),
        candidates=data["candidates"][ids].to(device),
        membership=data["membership"][ids].to(device),
        initial=data["initial"][ids].to(device),
        owners=None if "owners" not in data else data["owners"][ids].to(device),
        object_features=None if "object_features" not in data else data["object_features"][ids].to(device),
        anchor_distances=None if "anchor_distances" not in data else data["anchor_distances"][ids].to(device),
        hold_indices=None if "hold_indices" not in data else data["hold_indices"][ids].to(device),
        motion_features=None if "motion_features" not in data else data["motion_features"][ids].to(device),
        **({'clock_features': data['clock_features'][ids].to(device)}
           if 'clock_features' in data else {}),
    )


def metrics(model, data, rows, ids, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in ids.split(128):
            scores.append(evaluate_batch(model, data, batch, device)[0].cpu())
    scores = torch.cat(scores)
    result = {}
    for category in ("initial_group", "transition", "continuation", "forced"):
        active = [
            j for j, i in enumerate(ids.tolist()) if rows[i]["category"] == category
        ]
        if not active:
            continue
        selected = scores[active]
        target = torch.tensor([rows[int(ids[j])]["label"] for j in active])
        probs = selected.softmax(-1)
        result[category] = dict(
            count=len(active),
            correct=int((selected.argmax(-1) == target).sum()),
            accuracy=float((selected.argmax(-1) == target).float().mean()),
            mean_teacher_probability=float(
                probs[torch.arange(len(active)), target].mean()
            ),
        )
    return result


def train_imitation(
    manifest_path, output, *, epochs=1200, timeout_s=580, device="cuda", seed=17,
    morphology_aware=False,
):
    output = Path(output)
    output.mkdir(exist_ok=False, parents=True)
    started = time.monotonic()
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    manifest = json.loads(Path(manifest_path).read_text())
    path = Path(manifest["dataset_path"])
    if hash_file(path) != manifest["dataset_sha256"]:
        raise ValueError("dataset bytes changed")
    source = torch.load(path, map_location="cpu", weights_only=True)
    data = dataset_inputs(source)
    rows = source["rows"]
    labels = torch.tensor([r["label"] for r in rows])
    split = {
        s: torch.tensor([i for i, r in enumerate(rows) if r["split"] == s])
        for s in ("train", "validation", "held_out")
    }
    model = RequestActorCritic(morphology_aware=morphology_aware).to(device)
    if morphology_aware != ("owners" in data):
        raise ValueError("dataset/model morphology contracts differ")
    train = split["train"]
    initial = train[data["initial"][train]]
    temporal = train[~data["initial"][train]]
    # Initial statistics and contact gradients come ONLY from initial states.
    used = (data["membership"][initial] & data["mask"][initial, :, None]).any(1)
    local = data["candidates"][initial][used]
    raw = data["features"][temporal][data["mask"][temporal]]
    with torch.no_grad():
        r = model.ranker
        r.contact_mean.copy_(local.mean(0))
        r.contact_scale.copy_(local.std(0).clamp_min(0.05))
        r.feature_mean.copy_(raw.mean(0))
        scale = raw.std(0)
        r.feature_scale.copy_(torch.where(scale > 1e-5, scale, 1.0))
    history = []
    auxiliary = None
    if morphology_aware and "kinematic_targets" in source:
        valid = source["kinematic_target_mask"].clone()
        valid &= torch.tensor([r["split"] == "train" and r["initial"] for r in rows])[:, None]
        values = source["kinematic_targets"][valid]
        if not len(values):
            raise ValueError("no training-only kinematic targets")
        target_mean, target_scale = values.mean(0), values.std(0).clamp_min(.05)
        targets = (source["kinematic_targets"] - target_mean) / target_scale
        auxiliary = torch.nn.Sequential(torch.nn.Linear(model.ranker.contact_head[0].in_features, 64),
                                        torch.nn.SiLU(), torch.nn.Linear(64, values.shape[-1])).to(device)
    # Two bounded fits, not repeated architecture/hyperparameter searching.
    for stage, indices, max_epochs in (
        ("initial", initial, epochs),
        ("execution", temporal, epochs),
    ):
        for p in model.parameters():
            p.requires_grad_(False)
        modules = (
            [model.ranker.contact_member, model.ranker.contact_head]
            if stage == "initial"
            else [model.ranker.request_head]
        )
        if (stage == "initial") == morphology_aware:
            modules.append(model.ranker.morphology_encoder)
        for module in modules:
            for p in module.parameters():
                p.requires_grad_(True)
        parameters = [p for p in model.parameters() if p.requires_grad]
        captured = []
        handle = None
        if stage == "initial" and auxiliary is not None:
            parameters += list(auxiliary.parameters())
            handle = model.ranker.contact_head.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        optimizer = torch.optim.AdamW(
            parameters,
            lr=0.002,
            weight_decay=0.0001,
        )
        best_loss = float("inf")
        best_state = None
        for epoch in range(1, max_epochs + 1):
            if time.monotonic() - started > timeout_s:
                raise TimeoutError("BC training deadline")
            model.train()
            losses = []
            for batch in indices[torch.randperm(len(indices))].split(128):
                optimizer.zero_grad(set_to_none=True)
                captured.clear()
                logits, _ = evaluate_batch(model, data, batch, device)
                loss = F.cross_entropy(logits, labels[batch].to(device))
                if morphology_aware and stage == "initial":
                    # Uniform mass is assigned ONLY to eligible requests. This
                    # avoids saturated BC initialization and invalid-mask loss.
                    log_probs = logits.log_softmax(-1)
                    eligible = data["mask"][batch].to(device)
                    uniform_loss = -(log_probs.masked_fill(~eligible, 0).sum(-1) / eligible.sum(-1)).mean()
                    loss = .95 * loss + .05 * uniform_loss
                auxiliary_loss = None
                if handle is not None:
                    active = valid[batch].to(device)
                    auxiliary_loss = F.smooth_l1_loss(auxiliary(captured[-1])[active], targets[batch].to(device)[active])
                    loss = loss + .2 * auxiliary_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 5.0, error_if_nonfinite=True
                )
                optimizer.step()
                losses.append(float(loss.detach()))
            if epoch % 10 == 0 or epoch == 1:
                m = metrics(model, data, rows, indices, device)
                total = sum(v["count"] for v in m.values())
                correct = sum(v["correct"] for v in m.values())
                loss = sum(losses) / len(losses)
                item = dict(
                    stage=stage,
                    epoch=epoch,
                    loss=loss,
                    train=m,
                    seconds=time.monotonic() - started,
                    auxiliary_loss=None if auxiliary_loss is None else float(auxiliary_loss.detach()),
                )
                history.append(item)
                print(json.dumps(item), flush=True)
                with (output / "epochs.jsonl").open("a") as f:
                    f.write(json.dumps(item) + "\n")
                if loss < best_loss:
                    best_loss = loss
                    best_state = model.checkpoint()
                if (
                    correct == total
                    and min(v["mean_teacher_probability"] for v in m.values()) > (.94 if morphology_aware and stage == "initial" else .985)
                    and (auxiliary_loss is None or float(auxiliary_loss.detach()) < .05)
                ):
                    # The stopping state satisfies the train-fit contract. A
                    # lower minibatch loss from an earlier epoch may not.
                    best_state = model.checkpoint()
                    break
        if handle is not None:
            handle.remove()
        model = RequestActorCritic.from_checkpoint(best_state).to(device)
    for p in model.parameters():
        p.requires_grad_(True)
    torch.save(model.checkpoint(), output / "checkpoint.pt")
    report = dict(
        version=model.version,
        selection="train fit; fixed architecture; no held-out selection",
        metrics={
            s: metrics(model, data, rows, ids, device)
            for s, ids in split.items()
            if s != "held_out"
        },
        initial_training_examples=len(initial),
        execution_training_examples=len(temporal),
        actor_critic_sha256=hash_file(output / "checkpoint.pt"),
        dataset_sha256=hash_file(path),
        seconds=time.monotonic() - started,
        morphology_aware=morphology_aware,
        kinematic_auxiliary_training_only=auxiliary is not None,
        initial_label_smoothing=.05 if morphology_aware else 0.,
    )
    write_json(output / "result.json", report)
    return report


def serialize_encoding(encoded):
    return {
        **{k: v.detach().cpu() for k, v in encoded.items() if k != "graph"},
        "graphs": _graph_dict(encoded["graph"]),
    }


def collate_transitions(transitions):
    items = [r["encoding"] for r in transitions]
    n = len(items)
    width = max(len(i["mask"]) for i in items)
    cw = max(len(i["candidates"]) for i in items)
    fw = items[0]["features"].shape[-1]
    cf = items[0]["candidates"].shape[-1]
    data = dict(
        features=torch.zeros(n, width, fw),
        mask=torch.zeros(n, width, dtype=torch.bool),
        candidates=torch.zeros(n, cw, cf),
        membership=torch.zeros(n, width, cw, dtype=torch.bool),
        initial=torch.stack([i["initial"] for i in items]),
        graphs=_pad_graphs([i["graphs"] for i in items]),
    )
    if any("owners" in i for i in items):
        if not all("owners" in i for i in items):
            raise ValueError("mixed request morphology encoding contracts")
        data["owners"] = torch.zeros(n, cw, dtype=torch.long)
    if any('object_features' in i for i in items):
        if not all('object_features' in i for i in items):
            raise ValueError('mixed object estimate encoding contracts')
        data['object_features']=torch.stack([i['object_features'] for i in items])
    if any('motion_features' in i for i in items):
        if not all('motion_features' in i for i in items):
            raise ValueError('mixed motion feedback encoding contracts')
        data['motion_features'] = torch.stack([i['motion_features'] for i in items])
    if any('clock_features' in i for i in items):
        if not all('clock_features' in i for i in items):
            raise ValueError('mixed task clock encoding contracts')
        data['clock_features'] = torch.stack([i['clock_features'] for i in items])
    if any('anchor_distances' in i for i in items):
        if not all('anchor_distances' in i for i in items):
            raise ValueError('mixed anchor-preference encoding contracts')
        data['anchor_distances'] = torch.zeros(n, width)
    for key in ('hold_indices', 'request_indices'):
        if any(key in i for i in items):
            if not all(key in i for i in items):
                raise ValueError('mixed duration encoding contracts')
            data[key] = torch.zeros(n, width, dtype=torch.long)
    for j, i in enumerate(items):
        w = len(i["mask"])
        for key in ('hold_indices', 'request_indices'):
            if key in data:
                if i[key].shape != i['mask'].shape:
                    raise ValueError('duration options do not match archived requests')
                data[key][j,:w] = i[key]
        c = len(i["candidates"])
        data["features"][j, :w] = i["features"]
        data["mask"][j, :w] = i["mask"]
        data["candidates"][j, :c] = i["candidates"]
        data["membership"][j, :w, :c] = i["membership"]
        if "owners" in data:
            data["owners"][j, :c] = i["owners"]
        if 'anchor_distances' in data:
            if i['anchor_distances'].shape != i['mask'].shape:
                raise ValueError('anchor distances do not match archived requests')
            data['anchor_distances'][j, :w] = i['anchor_distances']
    return data


def discounted_event_returns(trajectory, gamma_per_second):
    """Discount physical reward occurrences, retaining explicit legacy replay.

    Old archives have decision-time aggregate rewards. New timed archives also
    retain that raw sum for reporting, but learning uses each occurrence time.
    Mixing the two interpretations within an episode is rejected.
    """
    if not math.isfinite(gamma_per_second) or not 0 < gamma_per_second <= 1:
        raise ValueError("discount must be in (0, 1] per simulation second")
    timed = ["reward_timing" in e for e in trajectory]
    if any(timed) and not all(timed):
        raise ValueError("mixed reward timing contracts")
    result = []
    value = 0.0
    for i in range(len(trajectory) - 1, -1, -1):
        reward = float(trajectory[i]["reward"])
        if not math.isfinite(reward):
            raise ValueError("nonfinite event reward")
        if all(timed):
            event = trajectory[i]
            if event["reward_timing"] != "observed_reward_time_v1":
                raise ValueError("unsupported reward timing contract")
            start = float(event["time_s"])
            end = float(event["end_time_s"])
            if not math.isfinite(start) or not math.isfinite(end) or end < start:
                raise ValueError("invalid reward interval")
            if i + 1 < len(trajectory) and abs(end - float(trajectory[i + 1]["time_s"])) > 1e-7:
                raise ValueError("reward interval does not end at next decision")
            raw, reward, previous = 0.0, 0.0, start
            for timestamp, amount in event["reward_events"]:
                timestamp, amount = float(timestamp), float(amount)
                if (not math.isfinite(timestamp) or not math.isfinite(amount)
                        or timestamp < previous - 1e-9 or timestamp > end + 1e-9):
                    raise ValueError("reward occurrence outside causal interval")
                raw += amount
                reward += gamma_per_second ** max(0.0, timestamp - start) * amount
                previous = timestamp
            if not math.isclose(raw, float(event["reward"]), abs_tol=1e-6, rel_tol=1e-7):
                raise ValueError("timed reward does not match raw sum")
        discount = 0.0
        if i + 1 < len(trajectory):
            if "time_s" not in trajectory[i] or "time_s" not in trajectory[i + 1]:
                raise ValueError("event return requires observed decision timestamps")
            start = float(trajectory[i]["time_s"])
            end = float(trajectory[i + 1]["time_s"])
            if not math.isfinite(start) or not math.isfinite(end) or end < start:
                raise ValueError("decision timestamps must be finite and monotonic")
            discount = gamma_per_second ** (end - start)
        value = reward + discount * value
        result.append(value)
    return list(reversed(result))


def categorical_kl_from_logits(reference_logits, updated_logits):
    """KL on a fixed action mask without treating softmax underflow as support loss."""
    if reference_logits.shape != updated_logits.shape:
        raise ValueError("categorical KL requires matching shapes")
    support = torch.isfinite(reference_logits)
    if (
        not torch.equal(support, torch.isfinite(updated_logits))
        or not support.any(-1).all()
        or not (support | torch.isneginf(reference_logits)).all()
        or not (support | torch.isneginf(updated_logits)).all()
    ):
        raise ValueError("categorical KL requires the same valid action support")
    reference_log = reference_logits.double().log_softmax(-1)
    updated_log = updated_logits.double().log_softmax(-1)
    difference = reference_log.masked_fill(~support, 0) - updated_log.masked_fill(~support, 0)
    return (reference_log.exp() * difference).sum(-1)


def generalized_event_advantages(trajectory, gamma_per_second, gae_lambda):
    """GAE for complete episodes, with physical-time discount and per-decision lambda.

    Values are the predictions recorded before acting, never fitted outcomes.
    A terminal episode bootstraps with zero; a truncated collection is rejected.
    Lambda=1 exactly recovers Monte Carlo return minus collected value.
    """
    if not math.isfinite(gae_lambda) or not 0 <= gae_lambda <= 1:
        raise ValueError('GAE lambda must be in [0, 1]')
    if not trajectory or not trajectory[-1]['done'] or any(e['done'] for e in trajectory[:-1]):
        raise ValueError('GAE requires a complete episode with one terminal event')
    returns = discounted_event_returns(trajectory, gamma_per_second)
    values = [float(e['value']) for e in trajectory]
    if not all(math.isfinite(v) for v in values):
        raise ValueError('nonfinite collected value')
    advantages = [0.] * len(trajectory)
    carry = 0.
    for i in range(len(trajectory)-1, -1, -1):
        discount = (gamma_per_second ** (float(trajectory[i+1]['time_s']) - float(trajectory[i]['time_s']))
                    if i+1 < len(trajectory) else 0.)
        next_return = returns[i+1] if i+1 < len(trajectory) else 0.
        next_value = values[i+1] if i+1 < len(trajectory) else 0.
        interval_reward = returns[i] - discount * next_return
        delta = interval_reward + discount * next_value - values[i]
        carry = delta + discount * gae_lambda * carry
        advantages[i] = carry
    return advantages


def anchor_preference_loss(model, logits, data, initial):
    """KL(policy || fixed upstream prior), averaged over contact choices only.

    Unlike a one-time logit offset, this remains in every PPO update. The
    ordinary task returns/value targets are unchanged and reported separately.
    """
    if model.anchor_preference is None or not initial.any():
        return logits.masked_fill(~torch.isfinite(logits), 0).sum() * 0.
    if 'anchor_distances' not in data:
        raise ValueError('anchor preference requires collected distances')
    ids = torch.nonzero(initial.detach().cpu()).flatten()
    prior = model.anchor_preference.logits(
        data['anchor_distances'][ids].to(logits.device), data['mask'][ids].to(logits.device))
    return categorical_kl_from_logits(logits[initial], prior).mean()


@torch.no_grad()
def anchor_preference_metrics(logits, data):
    if 'anchor_distances' not in data or not data['initial'].any():
        return None
    ids = torch.nonzero(data['initial']).flatten()
    distances = data['anchor_distances'][ids].to(logits.device)
    probability = logits[ids.to(logits.device)].softmax(-1)
    return dict(expected_total_steps=float((probability*distances).sum(-1).mean()),
                center_probability=float((probability*(distances==0)).sum(-1).mean()))


def configure_anchor_preference(checkpoint, output, *, distance_decay=math.log(2.), kl_coefficient=.05):
    """Explicit, non-training migration; old rollouts cannot match the new SHA."""
    model = RequestActorCritic.load(checkpoint)
    model.enable_anchor_preference(AnchorPreference(distance_decay, kl_coefficient))
    state_path = Path(checkpoint).parent / 'training_state.pt'
    saved = torch.load(state_path, weights_only=True, map_location='cpu') if state_path.exists() else None
    if saved is not None and saved['checkpoint_sha256'] != hash_file(checkpoint):
        raise ValueError('optimizer does not belong to the migration checkpoint')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.save(model.checkpoint(), output/'checkpoint.pt')
    if saved is not None:
        saved.update(checkpoint_sha256=hash_file(output/'checkpoint.pt'),
                     anchor_preference=model.anchor_preference.to_dict())
        torch.save(saved, output/'training_state.pt')
    result = dict(source_checkpoint=str(Path(checkpoint).resolve()),
        source_checkpoint_sha256=hash_file(checkpoint), checkpoint_sha256=hash_file(output/'checkpoint.pt'),
        anchor_preference=model.anchor_preference.to_dict(), optimizer_preserved=saved is not None,
        completed_updates=None if saved is None else saved['completed_updates'],
        parameters_changed=False, ppo_update=False, fresh_rollouts_required=True)
    write_json(output/'anchor_preference_migration.json', result)
    return result


def configure_learning_signals(checkpoint, output, *, seed=17, enable_return_correction=True):
    """Enable deployable inputs/duration actions; migrate Adam states by name.

    This is a policy boundary, not a PPO update. Original weights and their
    optimizer histories survive; only new branches start fresh.
    """
    model = RequestActorCritic.load(checkpoint)
    old_names = [name for name, _ in model.ranker.named_parameters()]
    old_weights = deepcopy(model.state_dict())
    state_path = Path(checkpoint).parent / 'training_state.pt'
    saved = torch.load(state_path, weights_only=True, map_location='cpu') if state_path.exists() else None
    if saved is not None and saved['checkpoint_sha256'] != hash_file(checkpoint):
        raise ValueError('optimizer does not belong to migration checkpoint')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model.enable_object_conditions()
        model.enable_temporal_options()
    model.contact_return_ranking = (dict(version='condition_success_return_pairs_v1', weight=1., min_samples=2)
                                    if enable_return_correction else None)
    for name, value in old_weights.items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0,atol=0)
    if saved is not None:
        old = saved['actor_optimizer']
        if len(old['param_groups']) != 1 or len(old['param_groups'][0]['params']) != len(old_names):
            raise ValueError('unsupported optimizer parameter mapping')
        optimizer = torch.optim.Adam(model.ranker.parameters(),lr=saved['learning_rate'])
        updated = optimizer.state_dict()
        new_ids = updated['param_groups'][0]['params']
        updated['param_groups'][0] = {**deepcopy(old['param_groups'][0]), 'params':new_ids}
        old_ids = dict(zip(old_names,old['param_groups'][0]['params']))
        for (name, _), new_id in zip(model.ranker.named_parameters(),new_ids):
            if name in old_ids and old_ids[name] in old['state']:
                updated['state'][new_id] = deepcopy(old['state'][old_ids[name]])
        saved['actor_optimizer'] = updated
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.save(model.checkpoint(),output/'checkpoint.pt')
    if saved is not None:
        saved['checkpoint_sha256']=hash_file(output/'checkpoint.pt')
        torch.save(saved,output/'training_state.pt')
    result=dict(source_checkpoint=str(Path(checkpoint).resolve()),source_sha256=hash_file(checkpoint),
        checkpoint_sha256=hash_file(output/'checkpoint.pt'),seed=seed,
        original_parameters_preserved=True,existing_optimizer_moments_preserved=saved is not None,
        completed_updates=None if saved is None else saved['completed_updates'],
        object_inputs_enabled=True,temporal_options=model.temporal_options,
        contact_return_ranking=model.contact_return_ranking,ppo_update=False,fresh_rollouts_required=True)
    write_json(output/'learning_signal_migration.json',result)
    return result


def configure_motion_feedback(checkpoint, output, *, include_object_conditions=False):
    """Preserve old policy/value outputs and Adam history while adding inputs."""
    model = RequestActorCritic.load(checkpoint)
    before = deepcopy(model.state_dict())
    state_path = Path(checkpoint).parent / 'training_state.pt'
    saved = torch.load(state_path, weights_only=True, map_location='cpu') if state_path.exists() else None
    if saved is not None and saved['checkpoint_sha256'] != hash_file(checkpoint):
        raise ValueError('optimizer does not belong to motion migration checkpoint')
    added = 0
    if not model.motion_feedback:
        model.enable_motion_feedback()
        added += 9
    if include_object_conditions:
        model.enable_temporal_object_conditions()
        added += 4
    if not added:
        raise ValueError('motion feedback is already configured')
    for name, value in before.items():
        migrated = model.state_dict()[name]
        if value.shape != migrated.shape:
            assert value.ndim == 2 and migrated.shape == (value.shape[0], value.shape[1] + added)
            torch.testing.assert_close(value, migrated[:, :value.shape[1]], rtol=0, atol=0)
            assert not migrated[:, value.shape[1]:].count_nonzero()
        else:
            torch.testing.assert_close(value, migrated, rtol=0, atol=0)
    if saved is not None:
        for key, parameters in [('actor_optimizer', list(model.ranker.parameters())),
                                ('value_optimizer', list(model.value_head.parameters()))]:
            optimizer = saved[key]
            if len(optimizer['param_groups']) != 1 or len(optimizer['param_groups'][0]['params']) != len(parameters):
                raise ValueError('unsupported motion optimizer mapping')
            for pid, p in zip(optimizer['param_groups'][0]['params'], parameters):
                for name, value in list(optimizer['state'].get(pid, {}).items()):
                    if torch.is_tensor(value) and value.ndim and value.shape != p.shape:
                        if value.ndim != 2 or p.shape != (value.shape[0], value.shape[1] + added):
                            raise ValueError('unexpected motion optimizer shape')
                        padded = torch.zeros_like(p)
                        padded[:, :value.shape[1]].copy_(value)
                        optimizer['state'][pid][name] = padded
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.save(model.checkpoint(), output / 'checkpoint.pt')
    if saved is not None:
        saved['checkpoint_sha256'] = hash_file(output / 'checkpoint.pt')
        torch.save(saved, output / 'training_state.pt')
    result = dict(source_checkpoint=str(Path(checkpoint).resolve()), source_sha256=hash_file(checkpoint),
                  checkpoint_sha256=hash_file(output / 'checkpoint.pt'),
                  motion_feedback='observed_contact_motion_v1', original_weights_and_moments_preserved=True,
                  temporal_object_conditions=model.temporal_object_conditions,
                  fresh_rollouts_required=True, ppo_update=False,
                  completed_updates=None if saved is None else saved['completed_updates'])
    write_json(output / 'motion_feedback_migration.json', result)
    return result


def observed_contact_return_pairs(transitions, returns, baseline, *, min_samples=2):
    """Compare repeatedly successful/failed assignments at the same initial state.

    Planning costs incurred before the accepted assignment are removed here;
    PPO still receives their unchanged total reward. Unexecuted candidates and
    exhausted rankings are not physical outcome labels. Every condition has
    equal weight, regardless of how often its favored assignment was sampled.
    Mixed outcomes and small differences among successes remain PPO targets,
    not hard preference labels; this avoids amplifying timing noise.
    """
    conditions = {}
    for index, condition in zip(baseline['event_indices'],baseline['conditions']):
        event=transitions[index]
        if event.get('planning_rejected',False):
            continue
        row=conditions.setdefault(condition,dict(index=index,groups={}))
        value=float(returns[index])+float(event.get('planning_cost',0.))
        if not isinstance(event.get('episode_success'),bool):
            raise ValueError('return ranking requires recorded task outcomes')
        row['groups'].setdefault(event['action'],[]).append((value,event['episode_success']))
    pairs=[]
    scale=max(float(returns.std()),1e-6)
    for condition,row in conditions.items():
        groups={k:sum(r for r,_ in v)/len(v) for k,v in row['groups'].items() if len(v)>=min_samples}
        local=[]
        for winner, high in groups.items():
            for loser, low in groups.items():
                if (all(s for _,s in row['groups'][winner])
                        and not any(s for _,s in row['groups'][loser]) and high>low):
                    local.append(dict(index=row['index'],winner=winner,loser=loser,
                        margin=min(1.,(high-low)/scale),return_gap=high-low,
                        winner_samples=len(row['groups'][winner]),loser_samples=len(row['groups'][loser]),condition=condition))
        for pair in local:
            pair['weight']=1./len(local)
        pairs.extend(local)
    count=len({p['condition'] for p in pairs})
    for pair in pairs:pair['weight']/=count
    return pairs


def contact_return_ranking_loss(logits, reference, pairs):
    """Bounded relative-odds improvement, with unknown assignments unlabeled."""
    if not pairs:
        return logits[torch.isfinite(logits)].sum()*0.
    rows=torch.tensor([p['index'] for p in pairs],device=logits.device)
    win=torch.tensor([p['winner'] for p in pairs],device=logits.device)
    lose=torch.tensor([p['loser'] for p in pairs],device=logits.device)
    gap=logits[rows,win]-logits[rows,lose]
    previous=reference[rows,win]-reference[rows,lose]
    margin=torch.tensor([p['margin'] for p in pairs],device=logits.device)
    weight=torch.tensor([p['weight'] for p in pairs],device=logits.device)
    return (F.relu(margin-(gap-previous))*weight).sum()


def contact_return_ranking_metrics(logits, reference, pairs):
    changes=[float((logits[p['index'],p['winner']].double()-logits[p['index'],p['loser']].double()
        -reference[p['index'],p['winner']].double()+reference[p['index'],p['loser']].double()).detach()) for p in pairs]
    return dict(pairs=len(pairs),improved=sum(x>1e-6 for x in changes),
        regressed=sum(x< -1e-6 for x in changes),minimum_log_odds_change=min(changes,default=0.),
        mean_log_odds_change=sum(changes)/max(1,len(changes)))


def preserve_observed_contact_preferences(model, data, reference, pairs, previous_state, device):
    """Keep observed preferences by relinearizing the actual nonlinear actor.

    Project toward the proposed PPO step, not toward a globally shrunken step.
    Re-evaluate all constraints after each small correction. The caller still
    checks both PPO KL limits and restores optimizer state if this fails.
    """
    if not pairs:
        return dict(projected=False, accepted=True)
    probe_ids = sorted({p['index'] for p in pairs})
    ids = torch.tensor(probe_ids)
    local = {index: j for j, index in enumerate(probe_ids)}
    local_pairs = [{**p, 'index': local[p['index']]} for p in pairs]
    ref = reference[ids.to(reference.device)]
    with torch.no_grad():
        scores = evaluate_batch(model, data, ids, device)[0]
        diagnostics = contact_return_ranking_metrics(scores, ref, local_pairs)
    if diagnostics['minimum_log_odds_change'] >= -1e-6:
        return dict(projected=False, accepted=True, metrics=diagnostics)
    import numpy as np
    from scipy.optimize import minimize
    started = time.monotonic()
    params = list(model.ranker.parameters())
    proposal = torch.nn.utils.parameters_to_vector(params).detach().clone()
    original = torch.cat([previous_state['ranker.' + name].reshape(-1)
                          for name, _ in model.ranker.named_parameters()])
    temporal_only = torch.cat([
        torch.full((p.numel(),), name.startswith(('request_head.', 'hold_head.')),
                   dtype=torch.bool, device=p.device)
        for name, p in model.ranker.named_parameters()
    ])

    def assign(vector):
        offset = 0
        with torch.no_grad():
            for p in params:
                p.copy_(vector[offset:offset + p.numel()].view_as(p))
                offset += p.numel()

    current = proposal.clone()
    step_scale = max(float((proposal - original).norm()), 1e-12)
    iterations = []
    accepted = False
    # Bound optimizer overhead; inability to satisfy these observed constraints
    # rejects the proposal instead of silently weakening their tolerance.
    for iteration in range(16):
        logits = evaluate_batch(model, data, ids, device)[0]
        rows, bounds = [], []
        for pair in local_pairs:
            i, winner, loser = pair['index'], pair['winner'], pair['loser']
            gap = logits[i, winner] - logits[i, loser]
            gradient = torch.autograd.grad(gap, params, retain_graph=True, allow_unused=True)
            flat = torch.cat([
                torch.zeros_like(p).reshape(-1) if g is None else g.detach().reshape(-1)
                for p, g in zip(params, gradient)
            ]).double().cpu().numpy()
            norm = max(float(np.linalg.norm(flat)), 1e-12)
            rows.append(flat / norm)
            deficit = ref[i, winner].double() - ref[i, loser].double() - gap.double()
            bounds.append((float(deficit.detach()) + 1e-5) / norm)
        matrix = np.stack(rows)
        toward_proposal = (proposal - current).double().cpu().numpy()
        bound = (np.asarray(bounds) - matrix @ toward_proposal) / step_scale
        gram = matrix @ matrix.T
        solved = minimize(
            lambda x: (.5 * x @ gram @ x - bound @ x, gram @ x - bound),
            np.zeros(len(local_pairs)), jac=True, bounds=[(0., None)] * len(local_pairs),
            method='L-BFGS-B', options=dict(maxiter=500, ftol=1e-15, gtol=1e-12),
        )
        correction = matrix.T @ solved.x * step_scale
        current = torch.where(
            temporal_only, proposal,
            proposal + torch.as_tensor(correction, device=proposal.device, dtype=proposal.dtype),
        )
        assign(current)
        with torch.no_grad():
            scores = evaluate_batch(model, data, ids, device)[0]
            diagnostics = contact_return_ranking_metrics(scores, ref, local_pairs)
        iterations.append(dict(
            iteration=iteration + 1, minimum=diagnostics['minimum_log_odds_change'],
            correction_norm=float(np.linalg.norm(correction)),
            solver_success=bool(solved.success),
        ))
        if diagnostics['minimum_log_odds_change'] >= -1e-6:
            accepted = True
            break
    if not accepted:
        assign(original)
    return dict(
        projected=True, accepted=accepted, scale=1., metrics=diagnostics,
        independent_temporal_step_preserved=accepted, nonlinear_iterations=iterations,
        requested_step_norm=step_scale, correction_norm=iterations[-1]['correction_norm'],
        seconds=time.monotonic() - started,
    )


def branch_weighted_mean(values, initial, contact_weight):
    """Average each populated decision branch before assigning its weight."""
    if values.ndim != 1 or initial.shape != values.shape or initial.dtype != torch.bool:
        raise ValueError("branch reduction requires aligned one-dimensional tensors")
    if not math.isfinite(contact_weight) or not 0 < contact_weight < 1:
        raise ValueError("contact loss weight must be between zero and one")
    if not initial.any() or initial.all():
        raise ValueError("balanced PPO requires contact and temporal experiences")
    return contact_weight * values[initial].mean() + (1 - contact_weight) * values[~initial].mean()


def leave_one_out_contact_returns(values, conditions):
    """Other independent draws at the same initial state form the baseline."""
    if values.ndim != 1 or len(values) != len(conditions) or not torch.isfinite(values).all():
        raise ValueError("invalid grouped contact returns")
    baseline = torch.empty_like(values)
    for condition in sorted(set(conditions)):
        ids = [i for i, label in enumerate(conditions) if label == condition]
        if len(ids) < 2:
            raise ValueError("contact baseline requires other independent draws")
        for i in ids:
            baseline[i] = values[[j for j in ids if j != i]].mean()
    return values - baseline, baseline


def grouped_contact_baseline(transitions, returns, rollout_paths, manifest):
    """Validate condition identity, complete groups, seeds and actual initial inputs."""
    manifest_path = Path(manifest)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != "same_initial_condition_rollout_groups_v1" or manifest.get("group_size", 0) < 2:
        raise ValueError("invalid contact group manifest")
    canonical_sources = None
    canonical = manifest.get("canonical_dataset")
    if canonical is not None:
        if hash_file(canonical["path"]) != canonical["sha256"]:
            raise ValueError("canonical split dataset identity changed")
        sources = torch.load(canonical["path"], weights_only=True, map_location="cpu")["sources"]
        canonical_sources = {s["episode_id"]: s for s in sources}
        if len(canonical_sources) != len(sources):
            raise ValueError("duplicate canonical split episode")
    entries = {str(Path(r["path"]).resolve()): r for r in manifest["rollouts"]}
    if len(entries) != len(manifest["rollouts"]) or set(entries) != {str(Path(p).resolve()) for p in rollout_paths}:
        raise ValueError("contact group manifest must bind every rollout exactly once")
    indices, conditions, seeds, reference = [], [], {}, {}
    offset = 0
    for path in rollout_paths:
        entry = entries[str(Path(path).resolve())]
        if hash_file(path) != entry["sha256"]:
            raise ValueError("contact group rollout identity changed")
        from amsrr.utils.hashing import stable_hash
        record = json.loads(Path(entry["record_path"]).read_text())
        effective_split = record["dataset_split"]
        if canonical_sources is not None:
            source = canonical_sources.get(record.get("episode_id"))
            if (source is None
                    or source["binding"]["sha256"] != entry["record_sha256"]
                    or source.get("original_split", source["split"]) != effective_split):
                raise ValueError("canonical split does not bind original episode")
            effective_split = source["split"]
        if (hash_file(entry["record_path"]) != entry["record_sha256"]
                or effective_split != "train"
                or stable_hash(dict(record_sha256=entry["record_sha256"], snapshot_hash=entry["snapshot_hash"])) != entry["condition_hash"]):
            raise ValueError("contact baseline requires bound training initial conditions")
        condition = entry["condition_hash"]
        rollout = torch.load(path, weights_only=True, map_location="cpu")
        recorded_seeds = rollout["environment_seeds"]
        if len(recorded_seeds) != len(rollout["episodes"]):
            raise ValueError("contact group requires one independent seed per episode")
        for trajectory, seed in zip(rollout["episodes"], recorded_seeds):
            if not trajectory or not bool(trajectory[0]["encoding"]["initial"]) or any(bool(e["encoding"]["initial"]) for e in trajectory[1:]):
                raise ValueError("contact baseline requires exactly one initial selection")
            if seed in seeds.setdefault(condition, set()):
                raise ValueError("duplicate seed in contact baseline group")
            seeds[condition].add(seed)
            encoded = trajectory[0]["encoding"]
            if condition in reference:
                torch.testing.assert_close(encoded, reference[condition], rtol=0., atol=2e-5)
            else:
                reference[condition] = encoded
            indices.append(offset); conditions.append(condition)
            offset += len(trajectory)
    if offset != len(transitions) or any(len(s) != manifest["group_size"] for s in seeds.values()):
        raise ValueError("incomplete contact baseline groups")
    if len(indices) != sum(bool(t["encoding"]["initial"]) for t in transitions):
        raise ValueError("contact grouping omitted an initial decision")
    raw, baseline = leave_one_out_contact_returns(returns[indices], conditions)
    audit = dict(manifest_sha256=hash_file(manifest_path), group_size=manifest["group_size"],
                 canonical_dataset=canonical,
                 condition_count=len(seeds), event_indices=indices, conditions=conditions,
                 initial_returns=returns[indices].tolist(), other_draw_baselines=baseline.tolist(),
                 raw_contact_advantages=raw.tolist(), temporal_advantage_unchanged=True,
                 normalization="same pooled return standard deviation as timed_mc_v1")
    return indices, raw, audit


def ppo_update(
    checkpoint,
    rollout_paths,
    output,
    *,
    epochs=4,
    learning_rate=1e-4,
    gamma=0.99,
    device="cpu",
    actor_scope="all",
    entropy_coefficient=0.001,
    target_kl=None,
    contact_loss_weight=None,
    contact_target_kl=None,
    temporal_target_kl=None,
    temporal_epochs=0,
    training_profile="legacy_event_v1",
    value_epochs=80,
    contact_group_manifest=None,
    contact_baseline_transition_from=None,
    gae_lambda=None,
    contact_return_correction="checkpoint",
    previous_learning_rate=None,
):
    """Episodic event PPO. Reject greedy, stale or teacher-forced trajectories."""
    if (
        epochs < 1 or not math.isfinite(learning_rate) or learning_rate <= 0
        or actor_scope not in {"all", "temporal"}
        or not math.isfinite(entropy_coefficient) or entropy_coefficient < 0
        or (target_kl is not None and (not math.isfinite(target_kl) or target_kl <= 0))
    ):
        raise ValueError("invalid PPO optimization settings")
    if training_profile not in {"legacy_event_v1", "timed_mc_v1", "timed_value_v1"} or value_epochs < 1:
        raise ValueError("invalid PPO training profile")
    if contact_group_manifest is not None and training_profile != "timed_value_v1":
        raise ValueError("grouped contact baseline requires timed_value_v1")
    contact_baseline_kind = "condition_loo_v1" if contact_group_manifest is not None else "value_v1"
    contact_baseline_transition = None
    modern = training_profile in {"timed_mc_v1", "timed_value_v1"}
    if gae_lambda is not None and (training_profile != 'timed_value_v1'
            or not math.isfinite(gae_lambda) or not 0 <= gae_lambda <= 1):
        raise ValueError('GAE requires timed_value_v1 and lambda in [0, 1]')
    if contact_return_correction not in {'checkpoint', 'disabled'}:
        raise ValueError('unknown contact return correction')
    if modern and actor_scope != "all":
        raise ValueError("timed MC profile updates all actor branches")
    balanced = contact_loss_weight is not None
    if balanced:
        if (actor_scope != "all" or not math.isfinite(contact_loss_weight)
                or not 0 < contact_loss_weight < 1
                or any(v is None or not math.isfinite(v) or v <= 0
                       for v in (contact_target_kl, temporal_target_kl))):
            raise ValueError("balanced PPO requires all actors, a valid weight and both branch KL limits")
    elif contact_target_kl is not None or temporal_target_kl is not None:
        raise ValueError("branch KL limits require contact_loss_weight")
    if (not isinstance(temporal_epochs, int) or temporal_epochs < 0
            or (temporal_epochs and (not modern or not balanced))):
        raise ValueError("extra temporal epochs require modern balanced PPO")
    model = RequestActorCritic.load(checkpoint).to(device).eval()
    from amsrr.policies.request_q_policy import RequestQPolicy
    if isinstance(model, RequestQPolicy):
        raise ValueError('Q policy checkpoints require train_request_q.py, not the PPO optimizer')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if model.fixed_actor is not None and (not modern or actor_scope != 'all'
            or balanced or temporal_epochs or model.contact_return_ranking is not None
            or model.anchor_preference is not None):
        raise ValueError('independent actors require ordinary timed PPO without branch corrections')
    original_return_correction = model.contact_return_ranking
    if contact_return_correction == 'disabled':
        model.contact_return_ranking = None
    preference_config = None if model.anchor_preference is None else model.anchor_preference.to_dict()
    torch.set_num_threads(4)
    checkpoint_hash = hash_file(checkpoint)
    transitions = []
    returns = []
    gae_advantages = []
    for path in rollout_paths:
        rollout = torch.load(path, map_location="cpu", weights_only=True)
        if rollout.get("runtime_contracts") != request_runtime_contracts():
            raise ValueError(
                "PPO requires on-policy runtime contracts matching current admission and contact velocity; "
                "use the archived source to reproduce older data"
            )
        if (
            rollout["checkpoint_sha256"] != checkpoint_hash
            or rollout["sampling"] != "categorical"
            or rollout["teacher_phase_supervision"]
            or rollout.get("teacher_contact_supervision", False)
            or not rollout["complete"]
            or rollout.get("diagnostic_only", False)
            or rollout.get("evaluation_only", False)
        ):
            raise ValueError(
                "PPO requires complete on-policy causal categorical rollouts"
            )
        for trajectory in rollout["episodes"]:
            if not trajectory or not trajectory[-1]["done"]:
                raise ValueError("missing terminal event")
            if modern and any(e.get("reward_timing") != "observed_reward_time_v1" for e in trajectory):
                raise ValueError("timed MC requires physical reward timestamps")
            if model.contact_return_ranking is not None:
                success=trajectory[-1].get('task_success')
                if not isinstance(success,bool):
                    raise ValueError('return ranking requires fresh recorded task outcomes')
                trajectory=[dict(e) for e in trajectory]
                trajectory[0]['episode_success']=success
            transitions.extend(trajectory)
            returns.extend(discounted_event_returns(trajectory, gamma))
            if gae_lambda is not None:
                gae_advantages.extend(generalized_event_advantages(trajectory, gamma, gae_lambda))
    if len(transitions) < 2:
        raise ValueError("insufficient PPO events")
    for event in transitions:
        if model.temporal_options is not None:
            holds = event['encoding'].get('hold_indices')
            if holds is None:
                raise ValueError('duration policy requires fresh option rollouts')
            h = int(holds[event['action']])
            duration = 0. if h == 0 else model.temporal_options['durations_s'][h-1]
            if event.get('hold_duration_s') != duration:
                raise ValueError('sampled hold duration differs from executed action')
        elif 'hold_indices' in event['encoding']:
            raise ValueError('legacy policy cannot consume duration rollouts')
        ranked = model.ranked_contact is not None and bool(event['encoding']['initial'])
        if ranked != ('ranking_prefix' in event):
            raise ValueError('checkpoint and collected ranking action contracts differ')
        if ranked:
            expected_cost = model.ranked_contact['attempt_cost'] * len(event['ranking_prefix'])
            if (event.get('planning_attempts') != len(event['ranking_prefix'])
                    or event.get('planning_cost') != expected_cost
                    or event['reward_events'][0] != (event['time_s'], -expected_cost)):
                raise ValueError('ranking planning cost is missing or inconsistent')
    data = collate_transitions(transitions)
    policy_mask = (
        ~data["initial"].to(device)
        if actor_scope == "temporal"
        else torch.ones(len(transitions), dtype=torch.bool, device=device)
    )
    if model.fixed_actor is not None:
        policy_mask = (data['initial'].to(device) if model.trainable_branch == 'contact'
                       else ~data['initial'].to(device))
    if int(policy_mask.sum()) < 2:
        raise ValueError("insufficient eligible PPO actor events")
    initial = data["initial"].to(device)
    if balanced and (not initial.any() or initial.all()):
        raise ValueError("balanced PPO requires contact and temporal experiences")
    if actor_scope == "temporal":
        # Preserve contact selection and representation while fitting the
        # continuation/transition head. Value gradients cannot move the actor
        # through the shared graph encoder in this optimization stage.
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(("ranker.request_head.", "value_head.")))
    ids = torch.arange(len(transitions))
    actions = torch.tensor([r["action"] for r in transitions], device=device)
    old_log = torch.tensor([r["log_prob"] for r in transitions], device=device)
    target = torch.tensor(returns, device=device)
    old_value = torch.tensor([r["value"] for r in transitions], device=device)
    with torch.no_grad():
        logits, value = evaluate_batch(model, data, ids, device)
        replay = action_statistics(logits, transitions)[0]
        reference_logits = logits.detach().clone()
        error = float((replay - old_log).abs().max())
        replay_numerical_rechecks = []
        if error > 2e-5:
            # A long ranking sums several float32 log probabilities. Batched
            # matrix kernels can cross the absolute tolerance through rounding.
            # Check flagged observations using the actual single-observation
            # inference shape; never accept a changed distribution or old log.
            for index in torch.nonzero((replay-old_log).abs() > 2e-5).flatten().tolist():
                item = collate_transitions([transitions[index]])
                single_logits, _ = evaluate_batch(model, item, torch.tensor([0]), device)
                width = single_logits.shape[1]
                try:
                    torch.testing.assert_close(logits[index, :width], single_logits[0],
                                               rtol=1.3e-6, atol=1e-5)
                    assert not torch.isfinite(logits[index, width:]).any()
                except AssertionError as exc:
                    raise ValueError('batched and individual policy distributions differ') from exc
                single_log = action_statistics(single_logits, [transitions[index]])[0][0]
                single_error = float((single_log-old_log[index]).abs())
                if single_error > 2e-5:
                    raise ValueError(f"on-policy replay probabilities differ: {single_error}")
                replay_numerical_rechecks.append(dict(event_index=index,
                    batched_log_probability_error=float((replay[index]-old_log[index]).abs()),
                    individual_log_probability_error=single_error))
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    actor_parameters = list(model.ranker.parameters()) if modern else list(model.parameters())
    optimizer = torch.optim.Adam(actor_parameters, lr=learning_rate)
    value_optimizer = torch.optim.Adam(model.value_head.parameters(), lr=learning_rate) if modern else None
    state_path = Path(checkpoint).parent / "training_state.pt"
    prior_updates = 0
    learning_rate_transition = None
    value_history = []
    if modern:
        if state_path.exists():
            saved = torch.load(state_path, weights_only=True, map_location=device)
            if (saved["profile"] != training_profile or saved["checkpoint_sha256"] != checkpoint_hash):
                raise ValueError("optimizer state/checkpoint/profile identity mismatch")
            if saved['learning_rate'] != learning_rate:
                if previous_learning_rate != saved['learning_rate']:
                    raise ValueError('learning rate change requires its explicit previous value')
                learning_rate_transition = dict(previous=previous_learning_rate, current=learning_rate,
                                                after_completed_updates=saved['completed_updates'])
            elif previous_learning_rate is not None:
                raise ValueError('learning rate transition was already applied')
            if saved.get('anchor_preference') != preference_config:
                raise ValueError('optimizer anchor preference differs from checkpoint')
            prior_baseline = saved.get("contact_baseline_kind", "value_v1")
            if prior_baseline != contact_baseline_kind:
                if contact_baseline_transition_from != prior_baseline:
                    raise ValueError("contact baseline change requires its explicit previous contract")
                contact_baseline_transition = dict(previous=prior_baseline, current=contact_baseline_kind,
                                                   after_completed_updates=saved["completed_updates"])
            elif contact_baseline_transition_from is not None:
                raise ValueError("contact baseline transition was already applied")
            optimizer.load_state_dict(saved["actor_optimizer"])
            value_optimizer.load_state_dict(saved["value_optimizer"])
            if learning_rate_transition is not None:
                for opt in (optimizer, value_optimizer):
                    for group in opt.param_groups:
                        group['lr'] = learning_rate
            prior_updates = saved["completed_updates"]
    if previous_learning_rate is not None and learning_rate_transition is None:
        raise ValueError('learning rate transition requires a matching saved optimizer')
    if contact_baseline_transition_from is not None and contact_baseline_transition is None:
        raise ValueError("contact baseline transition requires a matching saved optimizer")
    # BC does not fit the value head. The first timed-value update therefore
    # uses MC returns; later updates use the value recorded BEFORE each action,
    # never the value fitted to this same batch's outcomes below.
    use_collected_value = not modern or (training_profile == "timed_value_v1" and prior_updates > 0)
    advantage = target - old_value if use_collected_value else target.clone()
    if gae_lambda is not None:
        if not use_collected_value:
            raise ValueError('GAE requires a previously trained value baseline')
        advantage = torch.tensor(gae_advantages, device=device)
        # Lambda returns train the critic; grouped initial-contact advantages
        # retain their independent same-condition Monte Carlo baseline below.
        value_target = advantage + old_value
    else:
        value_target = target
    advantage = (advantage - advantage[policy_mask].mean()) / advantage[policy_mask].std().clamp_min(1e-6)
    contact_baseline_audit = None
    if contact_group_manifest is not None:
        contact_indices, contact_advantages, contact_baseline_audit = grouped_contact_baseline(
            transitions, target, rollout_paths, contact_group_manifest)
        advantage[contact_indices] = contact_advantages / target[policy_mask].std().clamp_min(1e-6)
    return_pairs=[]
    if model.contact_return_ranking is not None:
        if contact_baseline_audit is None or actor_scope != 'all' or not modern:
            raise ValueError('return ranking requires grouped on-policy all-actor PPO')
        return_pairs=observed_contact_return_pairs(transitions,target,contact_baseline_audit,
            min_samples=model.contact_return_ranking['min_samples'])
    target_variance = float(target.var(unbiased=False))
    value_diagnostics = dict(
        baseline="collected_value" if use_collected_value else "batch_constant",
        collected_value_mse=float((target - old_value).square().mean()),
        collected_value_explained_variance=(None if target_variance < 1e-12 else
            1. - float((target - old_value).var(unbiased=False)) / target_variance),
    )
    history = []
    rejected_step = None
    initial_step_backtracks = []
    epoch = 0
    while epoch < epochs:
        saved_model = deepcopy(model.state_dict()) if balanced or return_pairs else None
        saved_optimizer = deepcopy(optimizer.state_dict()) if balanced or return_pairs else None
        optimizer.zero_grad(set_to_none=True)
        logits, value = evaluate_batch(model, data, ids, device)
        dist = Categorical(logits=logits)
        action_log, action_entropy, _ = action_statistics(logits, transitions)
        ratio = (action_log - old_log).exp()
        actor_terms = -torch.minimum(
            ratio * advantage, ratio.clamp(0.8, 1.2) * advantage
        )
        actor = (branch_weighted_mean(actor_terms, initial, contact_loss_weight)
                 if balanced else actor_terms[policy_mask].mean())
        critic = 0.5 * F.mse_loss(value, target)
        entropy_terms = action_entropy
        entropy = (branch_weighted_mean(entropy_terms, initial, contact_loss_weight)
                   if balanced else entropy_terms[policy_mask].mean())
        preference_kl = anchor_preference_loss(model, logits, data, initial)
        preference_loss = preference_kl * (
            0. if model.anchor_preference is None or actor_scope == 'temporal'
            else model.anchor_preference.kl_coefficient)
        return_ranking_loss=contact_return_ranking_loss(logits,reference_logits,return_pairs)
        return_ranking_weight=0. if model.contact_return_ranking is None else model.contact_return_ranking['weight']
        loss = (actor + (0.0 if modern else critic) - entropy_coefficient * entropy + preference_loss
                + return_ranking_weight*return_ranking_loss)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite PPO loss")
        loss.backward()
        if actor_scope == "temporal":
            grad = torch.nn.utils.clip_grad_norm_(
                model.ranker.request_head.parameters(), 1.0, error_if_nonfinite=True
            )
            critic_grad = torch.nn.utils.clip_grad_norm_(
                model.value_head.parameters(), 1.0, error_if_nonfinite=True
            )
        else:
            grad = torch.nn.utils.clip_grad_norm_(
                actor_parameters, 1.0, error_if_nonfinite=True
            )
            critic_grad = None
        optimizer.step()
        preference_step=preserve_observed_contact_preferences(
            model,data,reference_logits,return_pairs,saved_model,device)
        if not preference_step['accepted']:
            model.load_state_dict(saved_model);optimizer.load_state_dict(saved_optimizer)
            rejected_step=dict(epoch=epoch+1,reason='observed contact preference reversal',
                               preference_step=preference_step,restored_previous_parameters=True)
            break
        with torch.no_grad():
            updated_logits, _ = evaluate_batch(model, data, ids, device)
            divergences = action_statistics(updated_logits, transitions, reference_logits)[2]
            kl = float(divergences[policy_mask].mean())
            branch_kl = {
                "contact": float(divergences[initial].mean()) if initial.any() else None,
                "temporal": float(divergences[~initial].mean()) if (~initial).any() else None,
            }
        if balanced and (branch_kl["contact"] > contact_target_kl
                         or branch_kl["temporal"] > temporal_target_kl
                         or (target_kl is not None and kl > target_kl)):
            # Restore shared encoders and critic too: freezing just one head
            # cannot bound its distribution when shared features change.
            model.load_state_dict(saved_model)
            optimizer.load_state_dict(saved_optimizer)
            if modern and not history and len(initial_step_backtracks) < 8:
                # A saturated warmup can exceed the trust region on its very
                # first Adam step. Retry that same gradient from the exact
                # model/moment state with a bounded, smaller step; never relax
                # the behavioral KL limits or count a rollback as an update.
                old_rates = [group["lr"] for group in optimizer.param_groups]
                for group in optimizer.param_groups:
                    group["lr"] *= 0.5
                initial_step_backtracks.append(dict(
                    attempted_learning_rates=old_rates,
                    retry_learning_rates=[group["lr"] for group in optimizer.param_groups],
                    kl_from_collection=kl, branch_kl=branch_kl,
                    model_and_optimizer_restored=True,
                ))
                continue
            rejected_step = dict(epoch=epoch + 1, kl_from_collection=kl,
                                 branch_kl=branch_kl, restored_previous_parameters=True)
            break
        history.append(
            dict(
                loss=float(loss.detach()),
                actor=float(actor.detach()),
                contact_return_ranking_loss=float(return_ranking_loss.detach()),
                contact_preference_step=preference_step,
                critic=float(critic.detach()),
                entropy=float(entropy.detach()),
                anchor_preference_kl=float(preference_kl.detach()),
                anchor_preference_loss=float(preference_loss.detach()),
                gradient_norm=float(grad),
                critic_gradient_norm=None if critic_grad is None else float(critic_grad),
                kl_from_collection=kl,
                branch_kl=branch_kl,
            )
        )
        epoch += 1
        if target_kl is not None and kl >= target_kl:
            break
    # The contact trust region must not consume the independent temporal
    # heads' optimization budget. Keep all representations/contact scores
    # fixed while using the SAME collected actions, advantages and reference.
    temporal_history = []
    temporal_rejected_step = None
    if temporal_epochs:
        temporal_ids = ids[~initial.cpu()]
        temporal_events = [transitions[int(i)] for i in temporal_ids]
        temporal_parameters = []
        for name, parameter in model.ranker.named_parameters():
            active = name.startswith(("request_head.", "hold_head."))
            parameter.requires_grad_(active)
            if active:
                temporal_parameters.append(parameter)
        with torch.no_grad():
            contact_logits_before = evaluate_batch(model, data, ids[initial.cpu()], device)[0]
        for step in range(temporal_epochs):
            saved_model = deepcopy(model.state_dict())
            saved_optimizer = deepcopy(optimizer.state_dict())
            optimizer.zero_grad(set_to_none=True)
            logits, _ = evaluate_batch(model, data, temporal_ids, device)
            action_log, entropy_terms, _ = action_statistics(logits, temporal_events)
            ratio = (action_log - old_log[~initial]).exp()
            actor = -torch.minimum(ratio * advantage[~initial],
                                   ratio.clamp(.8, 1.2) * advantage[~initial]).mean()
            loss = actor - entropy_coefficient * entropy_terms.mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite temporal PPO loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(temporal_parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                updated_logits, _ = evaluate_batch(model, data, ids, device)
                divergences = action_statistics(updated_logits, transitions, reference_logits)[2]
                temporal_kl = float(divergences[~initial].mean())
                pooled_kl = float(divergences.mean())
                torch.testing.assert_close(updated_logits[initial], contact_logits_before, rtol=0., atol=2e-5)
            if (temporal_kl > temporal_target_kl
                    or (target_kl is not None and pooled_kl > target_kl)):
                model.load_state_dict(saved_model)
                optimizer.load_state_dict(saved_optimizer)
                temporal_rejected_step = dict(epoch=step+1, temporal_kl=temporal_kl,
                    kl_from_collection=pooled_kl, restored_previous_parameters=True)
                break
            temporal_history.append(dict(loss=float(loss.detach()), actor=float(actor.detach()),
                entropy=float(entropy_terms.detach().mean()), gradient_norm=float(grad),
                temporal_kl=temporal_kl, kl_from_collection=pooled_kl))
        for parameter in model.ranker.parameters():
            parameter.requires_grad_(True)
    if modern:
        # Fit the critic on the FINAL shared representation. Fitting before
        # actor updates makes the saved critic stale when those encoders move.
        # Advantages above still use values recorded before data collection;
        # this fit cannot leak these outcomes into their own actor update.
        for p in model.ranker.parameters():
            p.requires_grad_(False)
        # This stage changes only value_head. Reuse the exact inputs to that
        # head; no actor update follows this fit.
        value_features = model.frozen_value_features(
            data["features"][ids].to(device),
            graph_batch(data["graphs"], ids, device),
            data["mask"][ids].to(device),
            hold_indices=None if "hold_indices" not in data else data["hold_indices"][ids].to(device),
            motion_features=None if "motion_features" not in data else data["motion_features"][ids].to(device),
            object_features=None if "object_features" not in data else data["object_features"][ids].to(device),
        )
        for _ in range(value_epochs):
            value_optimizer.zero_grad(set_to_none=True)
            prediction = model.value_head(value_features).squeeze(-1)
            value_loss = F.mse_loss(prediction, value_target)
            if not torch.isfinite(value_loss):
                raise ValueError("nonfinite value fit")
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.value_head.parameters(), 1., error_if_nonfinite=True)
            value_optimizer.step()
            value_history.append(float(value_loss.detach()))
        del value_features
        for p in model.ranker.parameters():
            p.requires_grad_(True)
    delta = {
        k: float((v.detach() - before[k]).abs().max())
        for k, v in model.named_parameters()
    }
    if model.fixed_actor is not None:
        if any(v != 0 for k, v in delta.items() if k.startswith('fixed_actor.')):
            raise ValueError('immutable actor changed during PPO')
        with torch.no_grad():
            final_logits, _ = evaluate_batch(model, data, ids, device)
            torch.testing.assert_close(final_logits[~policy_mask], reference_logits[~policy_mask],
                                       rtol=0, atol=0)
    if max(v for k, v in delta.items() if k.startswith("ranker.")) <= 0:
        write_json(output / "failure.json", dict(
            reason="PPO actor did not update", initial_step_backtracks=initial_step_backtracks,
            rejected_step=rejected_step, checkpoint_sha256=checkpoint_hash,
        ))
        raise ValueError("PPO actor did not update")
    torch.save(model.checkpoint(), output / "checkpoint.pt")
    if modern:
        torch.save(dict(profile=training_profile, checkpoint_sha256=hash_file(output / "checkpoint.pt"),
                        learning_rate=learning_rate, completed_updates=prior_updates + 1,
                        contact_baseline_kind=contact_baseline_kind,
                        anchor_preference=preference_config,
                        actor_optimizer=optimizer.state_dict(), value_optimizer=value_optimizer.state_dict()),
                   output / "training_state.pt")
    restored = RequestActorCritic.load(output / "checkpoint.pt").to(device)
    with torch.no_grad():
        a = evaluate_batch(model, data, ids, device)
        b = evaluate_batch(restored, data, ids, device)
        torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
        torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)
    report = dict(
        optimization=dict(actor_scope=actor_scope, requested_epochs=epochs,
                          independent_trainable_branch=model.trainable_branch,
                          eligible_actor_events=int(policy_mask.sum()),
                          immutable_actor_unchanged=model.fixed_actor is not None,
                          training_profile=training_profile, optimizer_updates_resumed=prior_updates,
                          gae_lambda_per_decision=gae_lambda,
                          contact_return_correction=contact_return_correction,
                          original_contact_return_correction=original_return_correction,
                          learning_rate_transition=learning_rate_transition,
                          value_epochs=len(value_history), value_mse_history=value_history,
                          value_fit_order="after_actor" if modern else "joint",
                          completed_epochs=len(history), learning_rate=learning_rate,
                          completed_actor_steps=len(history) + len(temporal_history),
                          effective_actor_learning_rates=[group["lr"] for group in optimizer.param_groups],
                          initial_step_backtracks=initial_step_backtracks,
                          entropy_coefficient=entropy_coefficient, target_kl=target_kl,
                          anchor_preference=preference_config,
                          contact_loss_weight=contact_loss_weight,
                          contact_target_kl=contact_target_kl,
                          temporal_target_kl=temporal_target_kl,
                          temporal_epochs=temporal_epochs,
                          completed_temporal_epochs=len(temporal_history),
                          temporal_rejected_step=temporal_rejected_step,
                          branch_balanced_actor_and_entropy=balanced,
                          rejected_step=rejected_step,
                          separate_actor_critic_gradient_clipping=actor_scope == "temporal"),
        eligible_actor_events=int(policy_mask.sum()),
        discount=dict(gamma_per_simulation_second=gamma, clock="observed_time_s"),
        events=len(transitions),
        episodes=sum(
            len(torch.load(p, weights_only=True)["episodes"]) for p in rollout_paths
        ),
        replay_log_probability_max_error=error,
        replay_numerical_rechecks=replay_numerical_rechecks,
        value_diagnostics=value_diagnostics,
        contact_baseline_kind=contact_baseline_kind,
        contact_baseline_transition=contact_baseline_transition,
        contact_baseline=contact_baseline_audit,
        contact_return_ranking=dict(config=model.contact_return_ranking,
            observations=return_pairs,metrics=contact_return_ranking_metrics(b[0],reference_logits,return_pairs)),
        anchor_preference_diagnostics=dict(
            before=anchor_preference_metrics(reference_logits, data),
            after=anchor_preference_metrics(b[0], data)),
        updates=history,
        temporal_updates=temporal_history,
        parameter_max_changes=delta,
        checkpoint_sha256=hash_file(output / "checkpoint.pt"),
        reload_exact=True,
    )
    write_json(output / "result.json", report)
    return report
