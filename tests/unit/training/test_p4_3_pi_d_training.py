from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch

from amsrr.schemas.datasets import DatasetSplit, DesignOutcomeRecord, StageDecisionMasks
from amsrr.training.p2_inspection_context import build_p2_inspection_context
from amsrr.training.p2_learned_scorer import TinyP2MLP
from amsrr.training.p2_learning_dataset import P2_LEARNING_FEATURE_NAMES
from amsrr.training.p4_3_pi_d_training import (
    design_outcome_feature_vector,
    outcome_safety_target,
    train_p4_3_pi_d,
)


def test_p4_3_pi_d_trains_direct_design_outcome_shards_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    records: list[DesignOutcomeRecord] = []
    for sample_index, split in enumerate(
        (DatasetSplit.TRAIN, DatasetSplit.VALIDATION, DatasetSplit.HELD_OUT)
    ):
        context = build_p2_inspection_context(seed=11, sample_index=sample_index)
        for offset, candidate in enumerate(context.selection.accepted_candidates[:2]):
            safe = offset == 0
            records.append(
                _outcome_record(
                    candidate,
                    split=split,
                    suffix=f"{sample_index}-{offset}",
                    safe=safe,
                )
            )
    dataset_path = tmp_path / "design_outcome.jsonl"
    _write_records(dataset_path, records)

    p2_checkpoint = tmp_path / "p2_checkpoint.pt"
    p2_model = TinyP2MLP(input_dim=len(P2_LEARNING_FEATURE_NAMES), hidden_dim=24)
    torch.save(
        {
            "model_type": "TinyP2MLP",
            "task": "pi_d_selected_candidate_binary_classification",
            "state_dict": p2_model.state_dict(),
            "feature_names": list(P2_LEARNING_FEATURE_NAMES),
        },
        p2_checkpoint,
    )

    manifest = train_p4_3_pi_d(
        dataset_paths=dataset_path,
        output_dir=tmp_path / "training",
        p2_checkpoint_path=p2_checkpoint,
        epochs=3,
        seed=3,
    )

    for path in (
        manifest.checkpoint_path,
        manifest.metrics_path,
        manifest.loss_curve_path,
        manifest.rollout_outcome_evaluation_path,
        manifest.fallback_metadata_path,
    ):
        assert Path(path).is_file()
    checkpoint = torch.load(manifest.checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["feature_names"] == P2_LEARNING_FEATURE_NAMES
    assert checkpoint["p2_initialization"]["used"] is True
    assert checkpoint["outcome_target_is_inference_feature"] is False
    assert len(checkpoint["training_config_hash"]) == 64
    assert checkpoint["training_config"]["epochs"] == 3
    assert len(checkpoint["dataset_hash"]) == 64
    assert manifest.metrics["num_train_samples"] == 2.0
    assert manifest.metrics["num_validation_samples"] == 2.0
    assert manifest.metrics["num_held_out_samples"] == 2.0
    with Path(manifest.rollout_outcome_evaluation_path).open("r", encoding="utf-8") as handle:
        evaluation = json.load(handle)
    assert evaluation["record_count"] == 2
    assert evaluation["outcome_fields_are_targets_not_inference_features"] is True
    assert evaluation["training_config_hash"] == checkpoint["training_config_hash"]
    assert evaluation["dataset_hash"] == checkpoint["dataset_hash"]
    assert evaluation["p4_full_completion_claim"] is False
    with Path(manifest.fallback_metadata_path).open("r", encoding="utf-8") as handle:
        fallback = json.load(handle)
    assert fallback["deterministic_fallback"] == "P2DesignPolicy"
    assert fallback["deterministic_fallback_available"] is True
    assert fallback["learned_feasibility_replaces_hard_gate"] is False


def test_p4_3_pi_d_outcome_changes_target_but_never_feature_vector() -> None:
    context = build_p2_inspection_context(seed=4, sample_index=0)
    candidate = context.selection.accepted_candidates[0]
    safe = _outcome_record(candidate, split=DatasetSplit.TRAIN, suffix="safe", safe=True)
    unsafe = replace(
        safe,
        record_id=f"{safe.record_id}:unsafe",
        task_success=False,
        object_dropped=True,
        hard_collision=True,
        controller_infeasible_terminal=True,
        episode_return=-1.0,
    )

    assert design_outcome_feature_vector(safe) == design_outcome_feature_vector(unsafe)
    assert outcome_safety_target(safe) > outcome_safety_target(unsafe)


def _outcome_record(candidate, *, split: DatasetSplit, suffix: str, safe: bool) -> DesignOutcomeRecord:
    return DesignOutcomeRecord(
        record_id=f"{candidate.design_output.task_id}:outcome:{suffix}",
        episode_id=f"episode:{suffix}",
        task_id=candidate.design_output.task_id,
        split=split,
        candidate_id=candidate.candidate_id,
        selected_for_rollout=True,
        design_output=candidate.design_output,
        feasibility_result=candidate.feasibility_result,
        rollout_executed=True,
        task_success=safe,
        object_dropped=not safe,
        hard_collision=False,
        controller_infeasible_terminal=False,
        episode_return=1.0 if safe else -1.0,
        rollout_metrics={"success": 1.0 if safe else 0.0},
        failure_reason=None if safe else "object_drop",
        stage_masks=StageDecisionMasks(design_decision_mask=True),
    )


def _write_records(path: Path, records: list[DesignOutcomeRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_json())
            handle.write("\n")
