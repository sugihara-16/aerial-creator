from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from amsrr.training.p2_learning_dataset import (
    P2_LEARNING_FEATURE_NAMES,
    load_id_split,
    load_p2_learning_dataset,
)


@dataclass(frozen=True)
class P2ScorerTrainingManifest:
    output_dir: str
    checkpoint_path: str
    metrics_path: str
    loss_curve_path: str
    metrics: dict[str, float]


class TinyP2MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 24) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def train_p2_learned_scorer(
    *,
    dataset_path: str | Path = "outputs/p2_5/datasets/p2_candidate_dataset.jsonl",
    train_ids_path: str | Path = "outputs/p2_5/datasets/train_ids.json",
    val_ids_path: str | Path = "outputs/p2_5/datasets/val_ids.json",
    output_dir: str | Path = "outputs/p2_5/training/pi_d_scorer",
    epochs: int = 40,
    lr: float = 0.03,
    seed: int = 0,
) -> P2ScorerTrainingManifest:
    torch.manual_seed(seed)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    records = load_p2_learning_dataset(dataset_path)
    train_ids = set(load_id_split(train_ids_path))
    val_ids = set(load_id_split(val_ids_path))
    train_records = [record for record in records if record["record_id"] in train_ids]
    val_records = [record for record in records if record["record_id"] in val_ids]
    x_train, y_train = _tensors(train_records, "selected_label")
    x_val, y_val = _tensors(val_records, "selected_label")
    model = TinyP2MLP(input_dim=x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    curve: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train)
        train_loss = loss_fn(logits, y_train)
        train_loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val), y_val)
        curve.append({"epoch": float(epoch + 1), "train_loss": float(train_loss.item()), "val_loss": float(val_loss.item())})
    metrics = _selected_metrics(model, x_train, y_train, x_val, y_val)
    metrics["num_train_samples"] = float(len(train_records))
    metrics["num_val_samples"] = float(len(val_records))
    checkpoint_path = target_dir / "checkpoint.pt"
    metrics_path = target_dir / "metrics.json"
    loss_curve_path = target_dir / "loss_curve.csv"
    torch.save(
        {
            "model_type": "TinyP2MLP",
            "task": "pi_d_selected_candidate_binary_classification",
            "state_dict": model.state_dict(),
            "feature_names": P2_LEARNING_FEATURE_NAMES,
            "metrics": metrics,
            "source_of_truth": "P2DesignPolicy teacher selected label",
            "production_path": "not used in production path",
        },
        checkpoint_path,
    )
    _write_json(metrics_path, metrics)
    _write_curve(loss_curve_path, curve)
    return P2ScorerTrainingManifest(
        output_dir=str(target_dir),
        checkpoint_path=str(checkpoint_path),
        metrics_path=str(metrics_path),
        loss_curve_path=str(loss_curve_path),
        metrics=metrics,
    )


def _tensors(records: list[dict[str, Any]], target_key: str) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.tensor([record["features"] for record in records], dtype=torch.float32)
    targets = torch.tensor([float(record[target_key]) for record in records], dtype=torch.float32)
    return features, targets


def _selected_metrics(
    model: TinyP2MLP,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
) -> dict[str, float]:
    loss_fn = nn.BCEWithLogitsLoss()
    model.eval()
    with torch.no_grad():
        train_logits = model(x_train)
        val_logits = model(x_val)
        train_loss = loss_fn(train_logits, y_train)
        val_loss = loss_fn(val_logits, y_val)
        val_probs = torch.sigmoid(val_logits)
        val_pred = (val_probs >= 0.5).float()
        selected_accuracy = (val_pred == y_val).float().mean()
    return {
        "train_loss": float(train_loss.item()),
        "val_loss": float(val_loss.item()),
        "selected_accuracy": float(selected_accuracy.item()),
    }


def _write_json(path: Path, data: dict[str, float]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_curve(path: Path, curve: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(curve)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the P2.5 learned π_D scorer bootstrap.")
    parser.add_argument("--dataset", default="outputs/p2_5/datasets/p2_candidate_dataset.jsonl")
    parser.add_argument("--train-ids", default="outputs/p2_5/datasets/train_ids.json")
    parser.add_argument("--val-ids", default="outputs/p2_5/datasets/val_ids.json")
    parser.add_argument("--output-dir", default="outputs/p2_5/training/pi_d_scorer")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    manifest = train_p2_learned_scorer(
        dataset_path=args.dataset,
        train_ids_path=args.train_ids,
        val_ids_path=args.val_ids,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )
    print(f"checkpoint: {manifest.checkpoint_path}")
    print(f"metrics: {manifest.metrics_path}")
    print(f"loss curve: {manifest.loss_curve_path}")
    print(json.dumps(manifest.metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
