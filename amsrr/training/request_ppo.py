"""The same categorical event actor and observation tensors for BC and PPO."""

from dataclasses import fields
from copy import deepcopy
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


def request_runtime_contracts():
    """Action support and physical transition semantics of newly collected data."""
    # Local import avoids the executor's import of serialize_encoding at startup.
    from amsrr.policies.high_level_requests import REQUEST_ADMISSION_CONTRACT
    from amsrr.simulation.request_event_execution import ContactPointVelocityObserver, RequestEventSupervisor
    from amsrr.controllers.grasp_slip_compensation import GRASP_SLIP_CONTRACT, GRASP_POSE_CONTRACT
    from amsrr.simulation.placement_support import PLACEMENT_SUPPORT_CONTRACT

    return {
        "admission": REQUEST_ADMISSION_CONTRACT,
        "contact_velocity": ContactPointVelocityObserver.contract_version,
        "grasp_slip": GRASP_SLIP_CONTRACT,
        "grasp_pose": GRASP_POSE_CONTRACT,
        "placement_outcome": PLACEMENT_SUPPORT_CONTRACT,
        "goal_completion": RequestEventSupervisor.goal_completion_contract,
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
    return result


def evaluate_batch(model, data, ids, device):
    return model.forward_encoded(
        data["features"][ids].to(device),
        graph_batch(data["graphs"], ids, device),
        data["mask"][ids].to(device),
        candidates=data["candidates"][ids].to(device),
        membership=data["membership"][ids].to(device),
        initial=data["initial"][ids].to(device),
        owners=None if "owners" not in data else data["owners"][ids].to(device),
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
    for j, i in enumerate(items):
        w = len(i["mask"])
        c = len(i["candidates"])
        data["features"][j, :w] = i["features"]
        data["mask"][j, :w] = i["mask"]
        data["candidates"][j, :c] = i["candidates"]
        data["membership"][j, :w, :c] = i["membership"]
        if "owners" in data:
            data["owners"][j, :c] = i["owners"]
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
    training_profile="legacy_event_v1",
    value_epochs=80,
    contact_group_manifest=None,
    contact_baseline_transition_from=None,
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
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model = RequestActorCritic.load(checkpoint).to(device).eval()
    torch.set_num_threads(4)
    checkpoint_hash = hash_file(checkpoint)
    transitions = []
    returns = []
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
            transitions.extend(trajectory)
            returns.extend(discounted_event_returns(trajectory, gamma))
    if len(transitions) < 2:
        raise ValueError("insufficient PPO events")
    data = collate_transitions(transitions)
    policy_mask = (
        ~data["initial"].to(device)
        if actor_scope == "temporal"
        else torch.ones(len(transitions), dtype=torch.bool, device=device)
    )
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
        replay = Categorical(logits=logits).log_prob(actions)
        reference_logits = logits.detach().clone()
        error = float((replay - old_log).abs().max())
        if error > 2e-5:
            raise ValueError(f"on-policy replay probabilities differ: {error}")
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    actor_parameters = list(model.ranker.parameters()) if modern else list(model.parameters())
    optimizer = torch.optim.Adam(actor_parameters, lr=learning_rate)
    value_optimizer = torch.optim.Adam(model.value_head.parameters(), lr=learning_rate) if modern else None
    state_path = Path(checkpoint).parent / "training_state.pt"
    prior_updates = 0
    value_history = []
    if modern:
        if state_path.exists():
            saved = torch.load(state_path, weights_only=True, map_location=device)
            if (saved["profile"] != training_profile or saved["checkpoint_sha256"] != checkpoint_hash
                    or saved["learning_rate"] != learning_rate):
                raise ValueError("optimizer state/checkpoint/profile identity mismatch")
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
            prior_updates = saved["completed_updates"]
    if contact_baseline_transition_from is not None and contact_baseline_transition is None:
        raise ValueError("contact baseline transition requires a matching saved optimizer")
    # BC does not fit the value head. The first timed-value update therefore
    # uses MC returns; later updates use the value recorded BEFORE each action,
    # never the value fitted to this same batch's outcomes below.
    use_collected_value = not modern or (training_profile == "timed_value_v1" and prior_updates > 0)
    advantage = target - old_value if use_collected_value else target.clone()
    advantage = (advantage - advantage[policy_mask].mean()) / advantage[policy_mask].std().clamp_min(1e-6)
    contact_baseline_audit = None
    if contact_group_manifest is not None:
        contact_indices, contact_advantages, contact_baseline_audit = grouped_contact_baseline(
            transitions, target, rollout_paths, contact_group_manifest)
        advantage[contact_indices] = contact_advantages / target[policy_mask].std().clamp_min(1e-6)
    target_variance = float(target.var(unbiased=False))
    value_diagnostics = dict(
        baseline="collected_value" if use_collected_value else "batch_constant",
        collected_value_mse=float((target - old_value).square().mean()),
        collected_value_explained_variance=(None if target_variance < 1e-12 else
            1. - float((target - old_value).var(unbiased=False)) / target_variance),
    )
    if modern:
        # No actor parameter, including shared encoders, may move during this
        # value fit. Value updates run even when actor KL later stops the actor.
        for p in model.ranker.parameters():
            p.requires_grad_(False)
        for _ in range(value_epochs):
            value_optimizer.zero_grad(set_to_none=True)
            _, prediction = evaluate_batch(model, data, ids, device)
            value_loss = F.mse_loss(prediction, target)
            if not torch.isfinite(value_loss):
                raise ValueError("nonfinite value fit")
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.value_head.parameters(), 1., error_if_nonfinite=True)
            value_optimizer.step()
            value_history.append(float(value_loss.detach()))
        for p in model.ranker.parameters():
            p.requires_grad_(True)
    history = []
    rejected_step = None
    initial_step_backtracks = []
    epoch = 0
    while epoch < epochs:
        saved_model = deepcopy(model.state_dict()) if balanced else None
        saved_optimizer = deepcopy(optimizer.state_dict()) if balanced else None
        optimizer.zero_grad(set_to_none=True)
        logits, value = evaluate_batch(model, data, ids, device)
        dist = Categorical(logits=logits)
        ratio = (dist.log_prob(actions) - old_log).exp()
        actor_terms = -torch.minimum(
            ratio * advantage, ratio.clamp(0.8, 1.2) * advantage
        )
        actor = (branch_weighted_mean(actor_terms, initial, contact_loss_weight)
                 if balanced else actor_terms[policy_mask].mean())
        critic = 0.5 * F.mse_loss(value, target)
        entropy_terms = dist.entropy()
        entropy = (branch_weighted_mean(entropy_terms, initial, contact_loss_weight)
                   if balanced else entropy_terms[policy_mask].mean())
        loss = actor + (0.0 if modern else critic) - entropy_coefficient * entropy
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
        with torch.no_grad():
            updated_logits, _ = evaluate_batch(model, data, ids, device)
            divergences = categorical_kl_from_logits(reference_logits, updated_logits)
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
                critic=float(critic.detach()),
                entropy=float(entropy.detach()),
                gradient_norm=float(grad),
                critic_gradient_norm=None if critic_grad is None else float(critic_grad),
                kl_from_collection=kl,
                branch_kl=branch_kl,
            )
        )
        epoch += 1
        if target_kl is not None and kl >= target_kl:
            break
    delta = {
        k: float((v.detach() - before[k]).abs().max())
        for k, v in model.named_parameters()
    }
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
                          training_profile=training_profile, optimizer_updates_resumed=prior_updates,
                          value_epochs=len(value_history), value_mse_history=value_history,
                          completed_epochs=len(history), learning_rate=learning_rate,
                          effective_actor_learning_rates=[group["lr"] for group in optimizer.param_groups],
                          initial_step_backtracks=initial_step_backtracks,
                          entropy_coefficient=entropy_coefficient, target_kl=target_kl,
                          contact_loss_weight=contact_loss_weight,
                          contact_target_kl=contact_target_kl,
                          temporal_target_kl=temporal_target_kl,
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
        value_diagnostics=value_diagnostics,
        contact_baseline_kind=contact_baseline_kind,
        contact_baseline_transition=contact_baseline_transition,
        contact_baseline=contact_baseline_audit,
        updates=history,
        parameter_max_changes=delta,
        checkpoint_sha256=hash_file(output / "checkpoint.pt"),
        reload_exact=True,
    )
    write_json(output / "result.json", report)
    return report
