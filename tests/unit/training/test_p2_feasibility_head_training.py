from __future__ import annotations

import json
from pathlib import Path

from amsrr.training.p2_feasibility_head_training import train_p2_feasibility_head
from amsrr.training.p2_learning_dataset import build_p2_learning_dataset


def test_p2_feasibility_head_training_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    dataset = build_p2_learning_dataset(output_dir=tmp_path / "datasets", sample_count=8, seed=3)

    manifest = train_p2_feasibility_head(
        dataset_path=dataset.dataset_path,
        train_ids_path=dataset.train_ids_path,
        val_ids_path=dataset.val_ids_path,
        output_dir=tmp_path / "feasibility_head",
        epochs=5,
        seed=4,
    )

    assert Path(manifest.checkpoint_path).exists()
    assert Path(manifest.metrics_path).exists()
    assert Path(manifest.loss_curve_path).exists()
    with Path(manifest.metrics_path).open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    for key in ("train_loss", "val_loss", "binary_accuracy", "precision", "recall"):
        assert key in metrics
    assert metrics["num_train_samples"] == dataset.train_count
    assert metrics["num_val_samples"] == dataset.val_count
    assert 0.0 <= metrics["binary_accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
