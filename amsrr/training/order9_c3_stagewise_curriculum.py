from __future__ import annotations

"""Evidence helpers for the common-action, stagewise Order 9 C3 curriculum."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

from amsrr.schemas.datasets import DatasetSplit
from amsrr.training.order9_c3_promotion import (
    load_order9_evaluation_episode_jsonl,
)
from amsrr.training.order9_rollout_buckets import (
    Order9PiLRolloutBucketManifest,
)
from amsrr.utils.hashing import hash_file


ORDER9_C3_STAGEWISE_CURRICULUM_VERSION = (
    "order9_c3_incremental_reward_phase_success_new_module_validation_v2"
)
ORDER9_C3_STAGEWISE_MODULE_MAXIMA = (3, 4, 5, 6, 7, 8)
ORDER9_C3_QUICK_VALIDATION_CONTRACT = (
    "order9_c3_phase_reset_quick_validation_v1_parent_checkpoint"
)
ORDER9_C3_CONTINUOUS_VALIDATION_SCHEDULE = (
    "order9_c3_continuous_validation_every_n_updates_and_stage_end_v1"
)
ORDER9_C3_INCREMENTAL_FORGETTING_CONTRACT = (
    "order9_c3_incremental_reward_phase_success_forgetting_v1"
)


@dataclass(frozen=True)
class Order9C3StagewiseBucketValidation:
    bucket_id: str
    module_count: int
    episode_count: int
    success_count: int
    safety_failure_count: int

    @property
    def success_rate(self) -> float:
        return self.success_count / self.episode_count

    def to_dict(self) -> dict[str, object]:
        return {
            "bucket_id": self.bucket_id,
            "module_count": self.module_count,
            "episode_count": self.episode_count,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "safety_failure_count": self.safety_failure_count,
        }


@dataclass(frozen=True)
class Order9C3StagewiseValidation:
    maximum_module_count: int
    expected_episode_count_per_bucket: int
    minimum_success_rate: float
    expected_bucket_ids: tuple[str, ...]
    buckets: tuple[Order9C3StagewiseBucketValidation, ...]

    @property
    def episode_count(self) -> int:
        return sum(value.episode_count for value in self.buckets)

    @property
    def success_count(self) -> int:
        return sum(value.success_count for value in self.buckets)

    @property
    def safety_failure_count(self) -> int:
        return sum(value.safety_failure_count for value in self.buckets)

    @property
    def complete(self) -> bool:
        return (
            {value.bucket_id for value in self.buckets}
            == set(self.expected_bucket_ids)
            and all(
                value.episode_count == self.expected_episode_count_per_bucket
                for value in self.buckets
            )
        )

    @property
    def total_expected_episode_count(self) -> int:
        return len(self.expected_bucket_ids) * self.expected_episode_count_per_bucket

    @property
    def maximum_possible_success_rate(self) -> float:
        return (
            self.success_count
            + self.total_expected_episode_count
            - self.episode_count
        ) / self.total_expected_episode_count

    @property
    def mathematically_failed(self) -> bool:
        return (
            self.safety_failure_count > 0
            or self.maximum_possible_success_rate < self.minimum_success_rate
        )

    @property
    def passed(self) -> bool:
        return (
            self.complete
            and self.safety_failure_count == 0
            and self.success_count / self.episode_count >= self.minimum_success_rate
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "contract": ORDER9_C3_STAGEWISE_CURRICULUM_VERSION,
            "maximum_module_count": self.maximum_module_count,
            "expected_episode_count_per_bucket": (
                self.expected_episode_count_per_bucket
            ),
            "minimum_success_rate": self.minimum_success_rate,
            "expected_bucket_count": len(self.expected_bucket_ids),
            "completed_bucket_count": len(self.buckets),
            "complete": self.complete,
            "episode_count": self.episode_count,
            "total_expected_episode_count": self.total_expected_episode_count,
            "success_count": self.success_count,
            "success_rate": (
                self.success_count / self.episode_count
                if self.episode_count
                else 0.0
            ),
            "maximum_possible_success_rate": self.maximum_possible_success_rate,
            "safety_failure_count": self.safety_failure_count,
            "mathematically_failed": self.mathematically_failed,
            "passed": self.passed,
            "expected_bucket_ids": list(self.expected_bucket_ids),
            "missing_bucket_ids": sorted(
                set(self.expected_bucket_ids)
                - {value.bucket_id for value in self.buckets}
            ),
            "buckets": [value.to_dict() for value in self.buckets],
        }


def select_order9_c3_stagewise_validation_bucket_ids(
    manifest: Order9PiLRolloutBucketManifest,
    *,
    maximum_module_count: int,
) -> tuple[str, ...]:
    """Select exactly two fixed held-out buckets for every seen morphology size."""

    if maximum_module_count not in ORDER9_C3_STAGEWISE_MODULE_MAXIMA:
        raise ValueError("C3 stagewise maximum module count must lie in [3, 8]")
    selected: list[str] = []
    for module_count in range(2, maximum_module_count + 1):
        candidates = sorted(
            (
                bucket
                for bucket in manifest.buckets
                if bucket.split == DatasetSplit.VALIDATION
                and bucket.module_count == module_count
            ),
            key=lambda bucket: (bucket.sample_index, bucket.bucket_id),
        )
        if len(candidates) != 2:
            raise ValueError(
                "C3 stagewise validation requires exactly two held-out buckets "
                f"for module_count={module_count}; observed={len(candidates)}"
            )
        selected.extend(bucket.bucket_id for bucket in candidates)
    return tuple(selected)


def select_order9_c3_new_module_validation_bucket_ids(
    manifest: Order9PiLRolloutBucketManifest,
    *,
    module_count: int,
) -> tuple[str, ...]:
    """Select only the two fixed held-out buckets of the newly added size."""

    if module_count not in ORDER9_C3_STAGEWISE_MODULE_MAXIMA:
        raise ValueError("C3 new-module count must lie in [3, 8]")
    candidates = sorted(
        (
            bucket
            for bucket in manifest.buckets
            if bucket.split == DatasetSplit.VALIDATION
            and bucket.module_count == module_count
        ),
        key=lambda bucket: (bucket.sample_index, bucket.bucket_id),
    )
    if len(candidates) != 2:
        raise ValueError(
            "C3 new-module validation requires exactly two held-out buckets "
            f"for module_count={module_count}; observed={len(candidates)}"
        )
    return tuple(bucket.bucket_id for bucket in candidates)


def compare_order9_c3_incremental_forgetting(
    *,
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    existing_maximum_module_count: int,
    reward_relative_tolerance: float = 0.05,
    reward_absolute_tolerance: float = 0.02,
    phase_success_absolute_tolerance: float = 0.03,
) -> dict[str, object]:
    """Check old morphologies using only reward and phase-success rollouts.

    The two summaries must use the same alternating bucket cycle. This avoids
    mistaking a different topology sample for catastrophic forgetting.
    """

    if (
        existing_maximum_module_count < 2
        or not 0.0 <= reward_relative_tolerance < 1.0
        or reward_absolute_tolerance < 0.0
        or not 0.0 <= phase_success_absolute_tolerance <= 1.0
    ):
        raise ValueError("invalid C3 incremental forgetting threshold")
    baseline_cycle = int(baseline_summary["bucket_cycle_index"])
    candidate_cycle = int(candidate_summary["bucket_cycle_index"])
    if baseline_cycle != candidate_cycle:
        raise ValueError("C3 forgetting summaries use different bucket cycles")
    rows: list[dict[str, object]] = []
    for module_count in range(2, existing_maximum_module_count + 1):
        aggregate_name = f"module_{module_count:02d}_train"
        baseline = baseline_summary["aggregates"].get(aggregate_name)
        candidate = candidate_summary["aggregates"].get(aggregate_name)
        if not isinstance(baseline, dict) or not isinstance(candidate, dict):
            raise ValueError(
                f"C3 forgetting summary is missing {aggregate_name}"
            )
        baseline_reward = float(baseline["reward_mean"])
        candidate_reward = float(candidate["reward_mean"])
        reward_tolerance = max(
            reward_absolute_tolerance,
            abs(baseline_reward) * reward_relative_tolerance,
        )
        baseline_phase_success = float(baseline["phase_success_rate"])
        candidate_phase_success = float(candidate["phase_success_rate"])
        reward_passed = candidate_reward >= baseline_reward - reward_tolerance
        phase_success_passed = (
            candidate_phase_success
            >= baseline_phase_success - phase_success_absolute_tolerance
        )
        rows.append(
            {
                "module_count": module_count,
                "baseline_reward_mean": baseline_reward,
                "candidate_reward_mean": candidate_reward,
                "reward_delta": candidate_reward - baseline_reward,
                "reward_tolerance": reward_tolerance,
                "reward_passed": reward_passed,
                "baseline_phase_success_rate": baseline_phase_success,
                "candidate_phase_success_rate": candidate_phase_success,
                "phase_success_delta": (
                    candidate_phase_success - baseline_phase_success
                ),
                "phase_success_tolerance": phase_success_absolute_tolerance,
                "phase_success_passed": phase_success_passed,
                "passed": reward_passed and phase_success_passed,
            }
        )
    return {
        "contract": ORDER9_C3_INCREMENTAL_FORGETTING_CONTRACT,
        "bucket_cycle_index": candidate_cycle,
        "baseline_checkpoint_sha256": baseline_summary[
            "evaluated_behavior_checkpoint_sha256"
        ],
        "candidate_checkpoint_sha256": candidate_summary[
            "evaluated_behavior_checkpoint_sha256"
        ],
        "existing_maximum_module_count": existing_maximum_module_count,
        "reward_relative_tolerance": reward_relative_tolerance,
        "reward_absolute_tolerance": reward_absolute_tolerance,
        "phase_success_absolute_tolerance": phase_success_absolute_tolerance,
        "passed": all(bool(row["passed"]) for row in rows),
        "modules": rows,
    }


def summarize_order9_c3_stagewise_validation(
    *,
    manifest: Order9PiLRolloutBucketManifest,
    maximum_module_count: int,
    expected_episode_count_per_bucket: int,
    evaluation_root: str | Path,
    minimum_success_rate: float = 0.80,
    allow_mathematically_failed_partial: bool = False,
    new_module_only: bool = False,
) -> Order9C3StagewiseValidation:
    if (
        expected_episode_count_per_bucket < 1
        or not 0.0 <= minimum_success_rate <= 1.0
    ):
        raise ValueError("stagewise validation episode count must be positive")
    root = Path(evaluation_root).resolve()
    module_by_bucket = {
        bucket.bucket_id: bucket.module_count for bucket in manifest.buckets
    }
    summaries: list[Order9C3StagewiseBucketValidation] = []
    expected_bucket_ids = (
        select_order9_c3_new_module_validation_bucket_ids(
            manifest,
            module_count=maximum_module_count,
        )
        if new_module_only
        else select_order9_c3_stagewise_validation_bucket_ids(
            manifest,
            maximum_module_count=maximum_module_count,
        )
    )
    for bucket_id in expected_bucket_ids:
        episode_path = (
            root / "buckets" / bucket_id / "evaluation_episodes.jsonl"
        )
        if not episode_path.is_file():
            continue
        episodes = load_order9_evaluation_episode_jsonl(
            episode_path
        )
        summaries.append(
            Order9C3StagewiseBucketValidation(
                bucket_id=bucket_id,
                module_count=module_by_bucket[bucket_id],
                episode_count=len(episodes),
                success_count=sum(int(value.task_success) for value in episodes),
                safety_failure_count=sum(
                    int(value.safety_failure) for value in episodes
                ),
            )
        )
    result = Order9C3StagewiseValidation(
        maximum_module_count=maximum_module_count,
        expected_episode_count_per_bucket=expected_episode_count_per_bucket,
        minimum_success_rate=minimum_success_rate,
        expected_bucket_ids=expected_bucket_ids,
        buckets=tuple(summaries),
    )
    if not result.complete and not (
        allow_mathematically_failed_partial and result.mathematically_failed
    ):
        raise ValueError("C3 stagewise validation evidence is incomplete")
    return result


def should_run_order9_c3_continuous_validation(
    *,
    attempt_ordinal: int,
    interval_updates: int,
    maximum_attempt_count: int,
) -> bool:
    """Select periodic candidate checkpoints and always include stage end."""

    if (
        attempt_ordinal < 1
        or interval_updates < 1
        or maximum_attempt_count < 1
        or attempt_ordinal > maximum_attempt_count
    ):
        raise ValueError("C3 continuous-validation schedule is invalid")
    return (
        attempt_ordinal % interval_updates == 0
        or attempt_ordinal == maximum_attempt_count
    )


def summarize_order9_c3_phase_reset_validation(
    *,
    dataset_manifest_path: str | Path,
    rollout_log_path: str | Path,
) -> dict[str, object]:
    """Archive the already-collected, cheap phase-reset validation signal.

    The validation shard belongs to the behavior checkpoint used to collect
    the generation, i.e. the parent of the PPO child produced from that same
    generation.  The explicit checkpoint binding below prevents this signal
    from being misreported as post-update continuous validation.
    """

    manifest_path = Path(dataset_manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation_shards = [
        value
        for value in manifest.get("shards", [])
        if value.get("split") == DatasetSplit.VALIDATION.value
    ]
    if len(validation_shards) != 1:
        raise ValueError("C3 quick validation requires exactly one validation shard")
    shard = validation_shards[0]
    rollout_result = _load_order9_rollout_json(rollout_log_path)
    expected_checkpoint = str(manifest["behavior_checkpoint_sha256"])
    if (
        rollout_result.get("split") != DatasetSplit.VALIDATION.value
        or rollout_result.get("raw_artifact_sha256") != shard.get("sha256")
        or rollout_result.get("environment_steps")
        != shard.get("environment_step_count")
        or rollout_result.get("promotion_evidence_eligible") is not False
    ):
        raise ValueError("C3 quick validation rollout binding differs")

    raw_path = Path(str(shard["path"])).resolve()
    payload: dict[str, object] = {
        "contract": ORDER9_C3_QUICK_VALIDATION_CONTRACT,
        "promotion_evidence_eligible": False,
        "validation_semantics": "phase_reset_short_horizon_stochastic",
        "evaluated_checkpoint_role": "pre_update_behavior_parent",
        "evaluated_checkpoint_sha256": expected_checkpoint,
        "generation_id": str(manifest["generation_id"]),
        "dataset_manifest_path": str(manifest_path),
        "dataset_manifest_sha256": hash_file(manifest_path),
        "raw_artifact_path": str(raw_path),
        "raw_artifact_sha256": str(shard["sha256"]),
        "raw_tensor_metrics_available": raw_path.is_file(),
        "module_count": int(shard["module_count"]),
        "environment_count": int(shard["environment_count"]),
        "environment_step_count": int(shard["environment_step_count"]),
        "terminal_count": int(rollout_result["terminal_count"]),
        "successful_terminal_count": int(
            rollout_result["successful_terminal_count"]
        ),
        "terminal_success_rate": _safe_ratio(
            int(rollout_result["successful_terminal_count"]),
            int(rollout_result["terminal_count"]),
        ),
        "phase_transition_counts": dict(
            rollout_result.get("phase_transition_counts", {})
        ),
        "aggregate_env_steps_per_s": float(
            rollout_result["aggregate_env_steps_per_s"]
        ),
        "collection_wall_elapsed_s": float(
            rollout_result["collection_wall_elapsed_s"]
        ),
    }
    if not raw_path.is_file():
        return payload

    # This import is intentionally lazy: curriculum selection and state
    # inspection remain lightweight when reproducible raw tensors were already
    # cleaned after their hash-bound summary was written.
    import torch

    from amsrr.training.order9_tensor_rollout_artifact import (
        load_order9_tensor_rollout_artifact,
    )

    artifact = load_order9_tensor_rollout_artifact(
        raw_path, expected_sha256=str(shard["sha256"])
    )
    if artifact.metadata.get("pi_l_checkpoint_sha256") != expected_checkpoint:
        raise ValueError("C3 quick validation checkpoint binding differs")
    tensors = artifact.tensors
    valid = tensors["valid"]
    valid_count = int(valid.sum().item())
    if valid_count != int(shard["environment_step_count"]):
        raise ValueError("C3 quick validation valid-step count differs")

    def rate(name: str) -> float:
        value = tensors[name][valid]
        return float(value.to(dtype=torch.float64).mean().item())

    reward = tensors["reward"][valid].to(dtype=torch.float64)
    reward_terms = tensors["reward_terms"][valid].to(dtype=torch.float64)
    reward_names = tuple(
        str(value) for value in artifact.metadata["reward_term_names"]
    )
    if reward_terms.shape[-1] != len(reward_names):
        raise ValueError("C3 quick validation reward layout differs")
    phase = tensors["phase_index"][valid]
    runtime_labels = tuple(
        str(value) for value in artifact.metadata["runtime_phase_labels"]
    )
    actor_phase_by_runtime = tuple(
        int(value)
        for value in artifact.metadata["actor_phase_index_by_runtime"]
    )
    if len(actor_phase_by_runtime) != len(runtime_labels):
        raise ValueError("C3 quick validation phase mapping differs")
    label_by_actor_phase = dict(zip(actor_phase_by_runtime, runtime_labels))
    phase_rows: dict[str, dict[str, object]] = {}
    for phase_index in sorted(int(value) for value in phase.unique().tolist()):
        if phase_index not in label_by_actor_phase:
            raise ValueError("C3 quick validation phase index differs")
        mask = phase == phase_index
        phase_rows[label_by_actor_phase[phase_index]] = {
            "phase_index": phase_index,
            "environment_step_count": int(mask.sum().item()),
            "reward_mean": float(reward[mask].mean().item()),
            "phase_success_rate": float(
                tensors["phase_success"][valid][mask]
                .to(dtype=torch.float64)
                .mean()
                .item()
            ),
            "qp_feasible_rate": float(
                tensors["qp_feasible"][valid][mask]
                .to(dtype=torch.float64)
                .mean()
                .item()
            ),
            "prohibited_collision_rate": float(
                tensors["prohibited_collision"][valid][mask]
                .to(dtype=torch.float64)
                .mean()
                .item()
            ),
        }
    payload.update(
        {
            "reward_mean": float(reward.mean().item()),
            "reward_std": float(reward.std(unbiased=False).item()),
            "reward_term_means": {
                name: float(reward_terms[:, index].mean().item())
                for index, name in enumerate(reward_names)
            },
            "phase_success_rate": rate("phase_success"),
            "qp_feasible_rate": rate("qp_feasible"),
            "prohibited_collision_rate": rate("prohibited_collision"),
            "rotor_saturation_rate": rate("rotor_saturation"),
            "terminal_step_rate": rate("terminal"),
            "truncated_step_rate": rate("truncated"),
            "actor_task_success_rate": rate("actor_task_success"),
            "phases": phase_rows,
        }
    )
    return payload


def _load_order9_rollout_json(path: str | Path) -> dict[str, Any]:
    prefix = "ORDER9_ROLLOUT_JSON="
    payload: dict[str, Any] | None = None
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(prefix):
                value = json.loads(line[len(prefix) :])
                if not isinstance(value, dict):
                    raise ValueError("C3 quick validation rollout result is invalid")
                payload = value
    if payload is None:
        raise ValueError("C3 quick validation rollout result is missing")
    return payload


def _safe_ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def write_order9_c3_stagewise_json(
    path: str | Path,
    payload: dict[str, object],
) -> None:
    """Atomically persist resumable curriculum state or validation evidence."""

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def cleanup_order9_intermediate_raw_artifacts(
    *,
    roots: Sequence[str | Path],
    allowed_ancestor: str | Path,
    evidence_path: str | Path,
) -> dict[str, object]:
    """Delete only explicitly scoped, reproducible intermediate ``.pt`` files."""

    ancestor = Path(allowed_ancestor).resolve()
    candidates: list[Path] = []
    for raw_root in roots:
        root = Path(raw_root).resolve()
        if ancestor not in root.parents or root.name not in {"raw", "buckets"}:
            raise ValueError(f"unsafe C3 intermediate cleanup root: {root}")
        if not root.exists():
            continue
        candidates.extend(
            path
            for path in root.rglob("*.pt")
            if path.is_file() and ancestor in path.resolve().parents
        )
    records = [
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": hash_file(path),
        }
        for path in sorted(set(candidates))
    ]
    payload: dict[str, object] = {
        "contract": "order9_reproducible_intermediate_raw_cleanup_v1",
        "allowed_ancestor": str(ancestor),
        "file_count": len(records),
        "size_bytes": sum(int(value["size_bytes"]) for value in records),
        "files": records,
    }
    write_order9_c3_stagewise_json(evidence_path, payload)
    for path in candidates:
        path.unlink()
    return payload


__all__ = [
    "ORDER9_C3_CONTINUOUS_VALIDATION_SCHEDULE",
    "ORDER9_C3_INCREMENTAL_FORGETTING_CONTRACT",
    "ORDER9_C3_QUICK_VALIDATION_CONTRACT",
    "ORDER9_C3_STAGEWISE_CURRICULUM_VERSION",
    "ORDER9_C3_STAGEWISE_MODULE_MAXIMA",
    "Order9C3StagewiseBucketValidation",
    "Order9C3StagewiseValidation",
    "cleanup_order9_intermediate_raw_artifacts",
    "compare_order9_c3_incremental_forgetting",
    "select_order9_c3_new_module_validation_bucket_ids",
    "select_order9_c3_stagewise_validation_bucket_ids",
    "should_run_order9_c3_continuous_validation",
    "summarize_order9_c3_phase_reset_validation",
    "summarize_order9_c3_stagewise_validation",
    "write_order9_c3_stagewise_json",
]
