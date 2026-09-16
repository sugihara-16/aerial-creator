"""The same categorical event actor and observation tensors for BC and PPO."""

from dataclasses import fields
from pathlib import Path
import json
import math
import time

import torch
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
    return dict(
        features=data["features"],
        mask=mask,
        graphs=data["graphs"],
        candidates=data["candidate_features"],
        membership=data["membership"],
        initial=torch.tensor([r["initial"] for r in data["rows"]]),
    )


def evaluate_batch(model, data, ids, device):
    return model.forward_encoded(
        data["features"][ids].to(device),
        graph_batch(data["graphs"], ids, device),
        data["mask"][ids].to(device),
        candidates=data["candidates"][ids].to(device),
        membership=data["membership"][ids].to(device),
        initial=data["initial"][ids].to(device),
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
    manifest_path, output, *, epochs=1200, timeout_s=580, device="cuda", seed=17
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
    model = RequestActorCritic().to(device)
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
            else [model.ranker.request_head, model.ranker.morphology_encoder]
        )
        for module in modules:
            for p in module.parameters():
                p.requires_grad_(True)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
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
                logits, _ = evaluate_batch(model, data, batch, device)
                loss = F.cross_entropy(logits, labels[batch].to(device))
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
                    and min(v["mean_teacher_probability"] for v in m.values()) > 0.985
                ):
                    # The stopping state satisfies the train-fit contract. A
                    # lower minibatch loss from an earlier epoch may not.
                    best_state = model.checkpoint()
                    break
        model = RequestActorCritic.from_checkpoint(best_state).to(device)
    for p in model.parameters():
        p.requires_grad_(True)
    torch.save(model.checkpoint(), output / "checkpoint.pt")
    report = dict(
        version=ACTOR_CRITIC_VERSION,
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
    for j, i in enumerate(items):
        w = len(i["mask"])
        c = len(i["candidates"])
        data["features"][j, :w] = i["features"]
        data["mask"][j, :w] = i["mask"]
        data["candidates"][j, :c] = i["candidates"]
        data["membership"][j, :w, :c] = i["membership"]
    return data


def discounted_event_returns(trajectory, gamma_per_second):
    """Discount by observed simulation time, independent of decision frequency.

    The terminal action has no successor value, so its terminal timestamp is
    unnecessary. A one-event planning rejection is therefore also supported.
    """
    if not math.isfinite(gamma_per_second) or not 0 < gamma_per_second <= 1:
        raise ValueError("discount must be in (0, 1] per simulation second")
    result = []
    value = 0.0
    for i in range(len(trajectory) - 1, -1, -1):
        reward = float(trajectory[i]["reward"])
        if not math.isfinite(reward):
            raise ValueError("nonfinite event reward")
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


def ppo_update(
    checkpoint,
    rollout_paths,
    output,
    *,
    epochs=4,
    learning_rate=1e-4,
    gamma=0.99,
    device="cpu",
):
    """Episodic event PPO. Reject greedy, stale or teacher-forced trajectories."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model = RequestActorCritic.load(checkpoint).to(device).eval()
    torch.set_num_threads(4)
    checkpoint_hash = hash_file(checkpoint)
    transitions = []
    returns = []
    for path in rollout_paths:
        rollout = torch.load(path, map_location="cpu", weights_only=True)
        if (
            rollout["checkpoint_sha256"] != checkpoint_hash
            or rollout["sampling"] != "categorical"
            or rollout["teacher_phase_supervision"]
            or not rollout["complete"]
            or rollout.get("diagnostic_only", False)
        ):
            raise ValueError(
                "PPO requires complete on-policy causal categorical rollouts"
            )
        for trajectory in rollout["episodes"]:
            if not trajectory or not trajectory[-1]["done"]:
                raise ValueError("missing terminal event")
            transitions.extend(trajectory)
            returns.extend(discounted_event_returns(trajectory, gamma))
    if len(transitions) < 2:
        raise ValueError("insufficient PPO events")
    data = collate_transitions(transitions)
    ids = torch.arange(len(transitions))
    actions = torch.tensor([r["action"] for r in transitions], device=device)
    old_log = torch.tensor([r["log_prob"] for r in transitions], device=device)
    target = torch.tensor(returns, device=device)
    old_value = torch.tensor([r["value"] for r in transitions], device=device)
    with torch.no_grad():
        logits, value = evaluate_batch(model, data, ids, device)
        replay = Categorical(logits=logits).log_prob(actions)
        error = float((replay - old_log).abs().max())
        if error > 2e-5:
            raise ValueError(f"on-policy replay probabilities differ: {error}")
    advantage = target - old_value
    advantage = (advantage - advantage.mean()) / advantage.std().clamp_min(1e-6)
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits, value = evaluate_batch(model, data, ids, device)
        dist = Categorical(logits=logits)
        ratio = (dist.log_prob(actions) - old_log).exp()
        actor = -torch.minimum(
            ratio * advantage, ratio.clamp(0.8, 1.2) * advantage
        ).mean()
        critic = 0.5 * F.mse_loss(value, target)
        entropy = dist.entropy().mean()
        loss = actor + critic - 0.001 * entropy
        if not torch.isfinite(loss):
            raise ValueError("nonfinite PPO loss")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        history.append(
            dict(
                loss=float(loss.detach()),
                actor=float(actor.detach()),
                critic=float(critic.detach()),
                entropy=float(entropy.detach()),
                gradient_norm=float(grad),
            )
        )
    delta = {
        k: float((v.detach() - before[k]).abs().max())
        for k, v in model.named_parameters()
    }
    if max(v for k, v in delta.items() if k.startswith("ranker.")) <= 0:
        raise ValueError("PPO actor did not update")
    torch.save(model.checkpoint(), output / "checkpoint.pt")
    restored = RequestActorCritic.load(output / "checkpoint.pt").to(device)
    with torch.no_grad():
        a = evaluate_batch(model, data, ids, device)
        b = evaluate_batch(restored, data, ids, device)
        torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
        torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)
    report = dict(
        discount=dict(gamma_per_simulation_second=gamma, clock="observed_time_s"),
        events=len(transitions),
        episodes=sum(
            len(torch.load(p, weights_only=True)["episodes"]) for p in rollout_paths
        ),
        replay_log_probability_max_error=error,
        updates=history,
        parameter_max_changes=delta,
        checkpoint_sha256=hash_file(output / "checkpoint.pt"),
        reload_exact=True,
    )
    write_json(output / "result.json", report)
    return report
