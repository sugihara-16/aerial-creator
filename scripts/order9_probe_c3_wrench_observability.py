#!/usr/bin/env python3
from __future__ import annotations

"""Probe privileged C3 wrench observability from deployable, sensorless inputs."""

import argparse
import json
from pathlib import Path
import random
import sys
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch
from torch import nn

from amsrr.training.order9_tensor_on_policy_dataset import (
    load_order9_tensor_pi_l_dataset,
)


PROBE_VERSION = "order9_c3_sensorless_wrench_observability_probe_v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--maximum-samples-per-topology", type=int, default=30000)
    return parser


class _Probe(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(width, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 2),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.model(value)


def _joint_action_summary(value: torch.Tensor) -> torch.Tensor:
    return _module_summary(value)


def _module_summary(value: torch.Tensor) -> torch.Tensor:
    if value.ndim < 4:
        raise ValueError("module summary requires [T, B, M, ...]")
    flat = value.flatten(start_dim=3)
    return torch.cat(
        (
            flat.mean(dim=2),
            flat.square().mean(dim=2).sqrt(),
            flat.amin(dim=2),
            flat.amax(dim=2),
        ),
        dim=-1,
    )


def _selected_assignment_summary(
    value: torch.Tensor, selected: torch.Tensor
) -> torch.Tensor:
    weight = selected.float().unsqueeze(-1)
    count = weight.sum(dim=-2).clamp_min(1.0)
    mean = (value * weight).sum(dim=-2) / count
    rms = (value.square() * weight).sum(dim=-2).div(count).sqrt()
    lower = torch.where(selected.unsqueeze(-1), value, torch.inf).amin(dim=-2)
    upper = torch.where(selected.unsqueeze(-1), value, -torch.inf).amax(dim=-2)
    return torch.cat((mean, rms, lower, upper), dim=-1)


def _relative_module_pose(value: torch.Tensor) -> torch.Tensor:
    result = value.clone()
    result[..., :3] -= result[..., :3].mean(dim=2, keepdim=True)
    return result


def _samples(artifact, *, include_recurrent: bool) -> tuple[torch.Tensor, ...]:
    tensors = artifact.tensors
    valid = tensors["valid"] & (tensors["phase_index"] == 1)
    selected = tensors["selected_assignment_mask"]
    assignment_features = tensors["actor_active_assignment_features"]
    anchor_indices = torch.tensor(
        [
            [int(assignment["anchor_id"]) for assignment in assignments]
            for assignments in artifact.metadata["assignment_templates_by_environment"]
        ],
        dtype=torch.long,
    )
    if anchor_indices.shape != selected.shape[1:]:
        raise ValueError("assignment template count differs from selected wrench slots")
    selected_features = assignment_features.gather(
        2,
        anchor_indices.unsqueeze(0)
        .unsqueeze(-1)
        .expand(
            assignment_features.shape[0],
            -1,
            -1,
            assignment_features.shape[-1],
        ),
    )
    features = [
        tensors["actor_active_knot_features"],
        _selected_assignment_summary(selected_features, selected),
        tensors["previous_global_action"],
        tensors["global_action"],
        _joint_action_summary(tensors["joint_action"]),
        _module_summary(tensors["local_joint_positions_rad"]),
        _module_summary(tensors["local_joint_velocities_radps"]),
        _module_summary(_relative_module_pose(tensors["module_pose_world"])),
        _module_summary(tensors["module_twist_world"]),
        tensors["object_twist_world"],
        tensors["controller_desired_wrench_body"],
        tensors["actor_allocation_residual_norm"].unsqueeze(-1),
        tensors["actor_controller_qp_feasible"].float().unsqueeze(-1),
    ]
    if include_recurrent:
        # The actor's recurrent output already incorporates the current graph
        # observation and remains deployable; it is the appropriate feature
        # for an auxiliary wrench-observer head evaluated after encoding.
        features.append(tensors["recurrent_state_out"])
    x = torch.cat(features, dim=-1)[valid]
    lower = tensors["wrench_lower_contact"][valid]
    upper = tensors["wrench_upper_contact"][valid]
    measured = tensors["selected_contact_wrenches_contact"][valid]
    midpoint = 0.5 * (lower + upper)
    half_width = (0.5 * (upper - lower).abs()).clamp_min(5.0e-4)
    normalized = ((measured - midpoint) / half_width).abs()
    assignment = (
        selected[valid]
        & tensors["wrench_bound_mask"][valid]
    )
    violation = torch.where(
        assignment.unsqueeze(-1),
        (normalized - 1.0).clamp_min(0.0),
        torch.zeros_like(normalized),
    ).amax(dim=(1, 2))
    target = torch.stack(
        (violation.clamp_max(20.0).log1p(), (violation == 0.0).float()),
        dim=-1,
    )
    return x, target


def _subsample(
    values: tuple[torch.Tensor, ...], maximum: int, seed: int
) -> tuple[torch.Tensor, ...]:
    count = values[0].shape[0]
    if count <= maximum:
        return values
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(count, generator=generator)[:maximum]
    return tuple(value.index_select(0, indices) for value in values)


def _metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    residual = prediction[:, 0] - target[:, 0]
    total = (target[:, 0] - target[:, 0].mean()).square().sum().clamp_min(1.0e-9)
    r2 = 1.0 - residual.square().sum() / total
    actual_range = target[:, 1] > 0.5
    predicted_range = prediction[:, 1].sigmoid() >= 0.5
    true_positive = (actual_range & predicted_range).sum().float()
    precision = true_positive / predicted_range.sum().clamp_min(1)
    recall = true_positive / actual_range.sum().clamp_min(1)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1.0e-9)
    actual_violation = target[:, 0].expm1()
    predicted_violation = prediction[:, 0].clamp_min(0).expm1()
    centered_actual = actual_violation - actual_violation.mean()
    centered_prediction = predicted_violation - predicted_violation.mean()
    correlation = (centered_actual * centered_prediction).sum() / (
        centered_actual.square().sum().sqrt()
        * centered_prediction.square().sum().sqrt()
    ).clamp_min(1.0e-9)
    return {
        "log1p_violation_mae": float(residual.abs().mean().cpu()),
        "log1p_violation_rmse": float(residual.square().mean().sqrt().cpu()),
        "log1p_violation_r2": float(r2.cpu()),
        "range_accuracy": float((actual_range == predicted_range).float().mean().cpu()),
        "range_precision": float(precision.cpu()),
        "range_recall": float(recall.cpu()),
        "range_f1": float(f1.cpu()),
        "violation_magnitude_correlation": float(correlation.cpu()),
        "actual_in_range_fraction": float(actual_range.float().mean().cpu()),
        "predicted_in_range_fraction": float(predicted_range.float().mean().cpu()),
        "sample_count": int(target.shape[0]),
    }


def _train(
    train: tuple[torch.Tensor, ...],
    tests: dict[str, tuple[torch.Tensor, ...]],
    *,
    device: torch.device,
    epochs: int,
) -> dict[str, object]:
    x_train, y_train = train
    mean = x_train.mean(dim=0)
    scale = x_train.std(dim=0).clamp_min(1.0e-4)
    model = _Probe(x_train.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    generator = torch.Generator().manual_seed(9009)
    losses = []
    for _ in range(epochs):
        order = torch.randperm(x_train.shape[0], generator=generator)
        total_loss = 0.0
        count = 0
        model.train()
        for start in range(0, len(order), 2048):
            selected = order[start : start + 2048]
            x = ((x_train[selected] - mean) / scale).to(device)
            y = y_train[selected].to(device)
            prediction = model(x)
            regression = (prediction[:, 0] - y[:, 0]).square().mean()
            positive = y[:, 1].sum().clamp_min(1.0)
            negative = (1.0 - y[:, 1]).sum().clamp_min(1.0)
            positive_weight = (negative / positive).clamp_max(100.0)
            classification = nn.functional.binary_cross_entropy_with_logits(
                prediction[:, 1], y[:, 1], pos_weight=positive_weight
            )
            loss = regression + 0.25 * classification
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(selected)
            count += len(selected)
        losses.append(total_loss / count)
    evaluations = {}
    model.eval()
    with torch.no_grad():
        for name, (x, y) in tests.items():
            rows = []
            for start in range(0, x.shape[0], 4096):
                rows.append(model(((x[start : start + 4096] - mean) / scale).to(device)).cpu())
            evaluations[name] = _metrics(torch.cat(rows), y)
    return {
        "training_loss_by_epoch": losses,
        "input_width": x_train.shape[-1],
        "evaluations": evaluations,
    }


def main() -> int:
    args = _parser().parse_args()
    random.seed(9009)
    torch.manual_seed(9009)
    device = torch.device(args.device)
    bundle = load_order9_tensor_pi_l_dataset(
        (REPOSITORY_ROOT / args.rollout_dataset).resolve()
    )
    report = {
        "probe_version": PROBE_VERSION,
        "raw_physx_actor_input": False,
        "force_torque_sensor_used": False,
        "target": "privileged_contact_frame_wrench_normalized_by_commanded_range",
        "split_contract": "train_m02_to_m06_test_m07_to_m08",
        "probes": {},
    }
    started = time.perf_counter()
    for include_recurrent, label in ((False, "current_step"), (True, "recurrent_history")):
        by_module = {}
        for artifact in bundle.train_artifacts:
            module_count = len(artifact.metadata["morphology_graph"]["modules"])
            by_module[module_count] = _subsample(
                _samples(artifact, include_recurrent=include_recurrent),
                args.maximum_samples_per_topology,
                seed=9009 + module_count,
            )
        training_parts = {
            module_count: tuple(
                value[: -max(1, value.shape[0] // 5)]
                for value in by_module[module_count]
            )
            for module_count in range(2, 7)
        }
        holdout_parts = {
            module_count: tuple(
                value[-max(1, value.shape[0] // 5) :]
                for value in by_module[module_count]
            )
            for module_count in range(2, 7)
        }
        training = tuple(
            torch.cat([training_parts[m][i] for m in range(2, 7)])
            for i in range(2)
        )
        tests = {
            "seen_topology_environment_holdout": tuple(
                torch.cat([holdout_parts[m][i] for m in range(2, 7)])
                for i in range(2)
            ),
            "unseen_m07": by_module[7],
            "unseen_m08": by_module[8],
        }
        report["probes"][label] = _train(
            training, tests, device=device, epochs=args.epochs
        )
        print(label, json.dumps(report["probes"][label]["evaluations"], sort_keys=True), flush=True)
    report["wall_elapsed_s"] = time.perf_counter() - started
    output = (REPOSITORY_ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"REPORT {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
