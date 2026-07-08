from __future__ import annotations

import json
from pathlib import Path

from amsrr.training.p2_learned_scorer import train_p2_learned_scorer
from amsrr.training.p2_learning_dataset import build_p2_learning_dataset


def test_p2_learned_scorer_training_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    dataset = build_p2_learning_dataset(output_dir=tmp_path / "datasets", sample_count=8, seed=1)

    manifest = train_p2_learned_scorer(
        dataset_path=dataset.dataset_path,
        train_ids_path=dataset.train_ids_path,
        val_ids_path=dataset.val_ids_path,
        output_dir=tmp_path / "pi_d_scorer",
        epochs=5,
        seed=2,
    )

    assert Path(manifest.checkpoint_path).exists()
    assert Path(manifest.metrics_path).exists()
    assert Path(manifest.loss_curve_path).exists()
    with Path(manifest.metrics_path).open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    assert "train_loss" in metrics
    assert "val_loss" in metrics
    assert "selected_accuracy" in metrics
    assert metrics["num_train_samples"] == dataset.train_count
    assert metrics["num_val_samples"] == dataset.val_count
    assert 0.0 <= metrics["selected_accuracy"] <= 1.0
