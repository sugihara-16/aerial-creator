"""Train-only temporal warmup with phase retiming and explicit feature scope.

Contact selection and the shared encoders remain unchanged. The temporal head
starts from phase, timing, target-error and feedback features; all its columns
remain ordinary trainable parameters when subsequently loaded for PPO.
"""
from copy import deepcopy
from pathlib import Path
import math
import time

import torch
from torch.nn import functional as F

from amsrr.policies.request_actor_critic import RequestActorCritic
from amsrr.policies.request_high_level_policy import REQUEST_FEATURE_NAMES
from amsrr.training.request_ppo import dataset_inputs, evaluate_batch, write_json
from amsrr.utils.hashing import hash_file

PROFILE = "task_state_retimed_temporal_warmup_v1"
STATE_START = next(i for i, name in enumerate(REQUEST_FEATURE_NAMES)
                   if name.startswith("current_phase."))
STATE_END = len(REQUEST_FEATURE_NAMES)
TIME_COLUMNS = tuple(REQUEST_FEATURE_NAMES.index(name) for name in
                     ("elapsed_phase_s", "nominal_phase_duration_s"))


def retimed_head_inputs(encoded, raw_features, factors, mean, scale):
    """Preserve progress and observed task state while varying the plan clock."""
    if (factors.shape != (len(encoded),) or not torch.isfinite(factors).all()
            or bool((factors <= 0).any())):
        raise ValueError("retiming factors must be finite positive per-row values")
    result = encoded.clone()
    columns = list(TIME_COLUMNS)
    result[:, :, columns] = (
        raw_features[:, :, columns] * factors[:, None, None] - mean[columns]
    ) / scale[columns]
    result[:, :, :STATE_START] = 0.
    result[:, :, STATE_END:] = 0.
    return result


def warmup_temporal(checkpoint, dataset_path, output, *, epochs=400,
                    seed=17, device="cuda", timeout_s=240):
    if epochs < 1 or timeout_s <= 0:
        raise ValueError("warmup budgets must be positive")
    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = torch.load(dataset_path, map_location="cpu", weights_only=True)
    data = dataset_inputs(source)
    indices = torch.tensor([i for i, row in enumerate(source["rows"])
                            if row["split"] == "train" and not row["initial"]])
    if not len(indices):
        raise ValueError("no training temporal decisions")
    model = RequestActorCritic.load(checkpoint).to(device).eval()
    parent = model.checkpoint()["state_dict"]
    captured = []
    handle = model.ranker.request_head.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach()))
    try:
        with torch.no_grad():
            for batch in indices.split(128):
                evaluate_batch(model, data, batch, device)
    finally:
        handle.remove()
    inputs = torch.cat(captured)
    raw = data["features"][indices].to(device)
    mask = data["mask"][indices].to(device)
    labels = torch.tensor([source["rows"][i]["label"] for i in indices], device=device)
    ids = torch.arange(len(indices), device=device)
    if not bool(mask[ids, labels].all()):
        raise ValueError("teacher label outside eligible requests")
    torch.manual_seed(seed)
    head = deepcopy(model.ranker.request_head)
    for module in head.modules():
        if isinstance(module, torch.nn.Linear):
            module.reset_parameters()
    with torch.no_grad():
        head[0].weight[:, :STATE_START] = 0.
        head[0].weight[:, STATE_END:] = 0.
    optimizer = torch.optim.AdamW(head.parameters(), lr=.002, weight_decay=.0001)

    def logits(batch, factors):
        values = retimed_head_inputs(inputs[batch], raw[batch], factors,
            model.ranker.feature_mean, model.ranker.feature_scale)
        return head(values).squeeze(-1).masked_fill(~mask[batch], -torch.inf)

    def metrics():
        with torch.no_grad():
            return {str(factor): dict(
                correct=int((logits(ids, torch.full((len(ids),), factor, device=device)).argmax(-1) == labels).sum()),
                count=len(ids), loss=float(F.cross_entropy(
                    logits(ids, torch.full((len(ids),), factor, device=device)), labels)))
                for factor in (1., 4., 10.)}

    best_loss, best = float("inf"), None
    for epoch in range(1, epochs + 1):
        total = 0.
        for batch in ids[torch.randperm(len(ids), device=device)].split(128):
            if time.monotonic() - started > timeout_s:
                raise TimeoutError("temporal warmup deadline")
            factors = torch.exp(torch.empty(len(batch), device=device).uniform_(math.log(.25), math.log(16.)))
            loss = F.cross_entropy(logits(batch, factors), labels[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        if total < best_loss:
            best_loss, best = total, deepcopy(head.state_dict())
        if epoch % 20 == 0 and all(v["correct"] == v["count"] and v["loss"] < .025
                                   for v in metrics().values()):
            best = deepcopy(head.state_dict())
            break
    head.load_state_dict(best)
    model.ranker.request_head.load_state_dict(head.state_dict())
    for name, value in model.state_dict().items():
        if not name.startswith("ranker.request_head."):
            assert torch.equal(value.cpu(), parent[name]), name
    assert not head[0].weight[:, :STATE_START].count_nonzero()
    assert not head[0].weight[:, STATE_END:].count_nonzero()
    torch.save(model.checkpoint(), output / "checkpoint.pt")
    report = dict(profile=PROFILE, parent_sha256=hash_file(checkpoint),
        dataset_sha256=hash_file(dataset_path), seed=seed, epochs=epoch,
        time_scale_range=[.25, 16.], original_teacher_labels_preserved=True,
        validation_used=False, train=metrics(), seconds=time.monotonic() - started,
        input_feature_names=list(REQUEST_FEATURE_NAMES[STATE_START:STATE_END]),
        contact_encoder_value_unchanged=True, ppo_columns_remain_trainable=True,
        checkpoint_sha256=hash_file(output / "checkpoint.pt"))
    write_json(output / "training.json", report)
    return report
