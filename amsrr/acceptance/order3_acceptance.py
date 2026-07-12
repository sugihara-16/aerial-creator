from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from amsrr.morphology.random_connected import morphology_structural_hash
from amsrr.policies.morphology_conditioned_low_level_policy import (
    load_order3_policy_checkpoint,
)
from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_non_empty
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.order3 import (
    ORDER3_ENCODER_VERSION,
    ORDER3_POLICY_ARCHITECTURE_VERSION,
    ORDER3_POLICY_FAMILY,
    ORDER3_TENSORIZER_VERSION,
    Order3DatasetManifest,
    Order3MorphologyPoolManifest,
)
from amsrr.schemas.order3_rollout_condition import (
    ORDER3_ROLLOUT_CONDITION_VERSION,
    Order3RolloutCondition,
)
from amsrr.schemas.policies import POLICY_COMMAND_CONTRACT_CENTROIDAL
from amsrr.simulation.order3_policy_rollout import (
    order3_condition_report_failures,
)
from amsrr.training.order3_free_flight import (
    ORDER3_FREE_FLIGHT_VERSION,
    ORDER3_REQUIRED_MODULE_COUNTS,
    Order3EvaluationEpisode,
    Order3TaskMode,
    Order3TerminalMetrics,
    order3_terminal_metrics_success,
    recommended_order3_morphology_split_counts,
)
from amsrr.utils.hashing import hash_file, stable_hash


ORDER3_ACCEPTANCE_VERSION = "order3_free_flight_acceptance_v1"
ORDER3_ACCEPTANCE_ARTIFACT_VERSION = "order3_free_flight_evaluation_artifact_v1"
ORDER3_AGGREGATE_SUCCESS_THRESHOLD = 0.95
ORDER3_PER_MODULE_SUCCESS_THRESHOLD = 0.90
ORDER3_NOMINAL_MAX_RELATIVE_DEGRADATION = 0.05
ORDER3_RANDOMIZED_MIN_RELATIVE_IMPROVEMENT = 0.15
ORDER3_RANDOMIZED_MIN_SUCCESS_GAIN = 0.10
ORDER3_ID_MAX_FALLBACK_RATE = 0.01

_SHA256_LENGTH = 64
_FORBIDDEN_TRUE_CLAIM_KEYS = frozenset(
    {
        "object_task_claim",
        "contact_task_claim",
        "natural_contact_success_claim",
        "p4_full_completion_claim",
        "p4_full_completion",
        "is_p4_full_completion",
    }
)


@dataclass
class Order3AcceptanceArtifactMetadata(SchemaBase):
    """Immutable binding/provenance fields emitted by the evaluation runner.

    Acceptance never consumes producer-computed pass/fail booleans.  The
    booleans here describe provenance or authority boundaries and are checked
    against the checkpoint and raw episode collection.
    """

    artifact_version: str
    evaluation_scope_version: str
    evaluation_source: str
    checkpoint_sha256: str
    dataset_manifest_sha256: str
    policy_family: str
    policy_contract_version: str
    architecture_version: str
    tensorizer_version: str
    encoder_version: str
    graph_encoder_used: bool
    recurrent_gru_used: bool
    actor_uses_privileged_wrench: bool
    deterministic_fallback_available: bool
    pool_hash: str
    evaluation_episode_set_hash: str
    rollout_condition_version: str
    raw_report_hashes_bound: bool
    paired_deterministic_baseline: bool
    required_task_modes: list[str]
    object_task_claim: bool
    contact_task_claim: bool
    p4_full_completion_claim: bool

    def validate(self) -> None:
        for name in (
            "artifact_version",
            "evaluation_scope_version",
            "evaluation_source",
            "policy_family",
            "policy_contract_version",
            "architecture_version",
            "tensorizer_version",
            "encoder_version",
        ):
            require_non_empty(
                str(getattr(self, name)),
                f"Order3AcceptanceArtifactMetadata.{name}",
            )
        for name in (
            "checkpoint_sha256",
            "dataset_manifest_sha256",
            "pool_hash",
            "evaluation_episode_set_hash",
        ):
            if not _is_sha256(str(getattr(self, name))):
                raise SchemaValidationError(
                    f"Order3AcceptanceArtifactMetadata.{name} must be a lowercase SHA-256 digest"
                )
        if self.rollout_condition_version != ORDER3_ROLLOUT_CONDITION_VERSION:
            raise SchemaValidationError(
                "Order3 acceptance rollout condition version mismatch"
            )
        if set(self.required_task_modes) != {
            mode.value for mode in Order3TaskMode
        } or len(self.required_task_modes) != len(Order3TaskMode):
            raise SchemaValidationError(
                "Order3 acceptance requires hover/waypoint/takeoff task modes"
            )


@dataclass
class Order3AcceptancePassSummary(SchemaBase):
    pool_passed: bool
    dataset_passed: bool
    checkpoint_passed: bool
    artifact_metadata_passed: bool
    held_out_coverage_passed: bool
    held_out_performance_passed: bool
    safety_passed: bool
    nominal_baseline_passed: bool
    randomized_robustness_passed: bool
    id_fallback_passed: bool
    ood_fallback_passed: bool
    no_scope_mislabeling_passed: bool
    completion_passed: bool

    def validate(self) -> None:
        components_passed = all(
            value
            for name, value in self.to_dict().items()
            if name != "completion_passed"
        )
        if self.completion_passed and not components_passed:
            raise SchemaValidationError(
                "Order3AcceptancePassSummary cannot pass with a failed component gate"
            )


@dataclass
class Order3AcceptanceReport(SchemaBase):
    acceptance_version: str
    pass_summary: Order3AcceptancePassSummary
    completion_passed: bool
    pool_split_module_counts: dict[str, dict[str, int]]
    held_out_episode_count: int
    held_out_unique_morphology_count: int
    ood_episode_count: int
    aggregate_held_out_success_rate: float
    per_module_success_rates: dict[str, float]
    safety_failure_episode_count: int
    nominal_relative_degradation: float | None
    randomized_relative_improvement: float | None
    randomized_success_gain: float | None
    id_fallback_rate: float
    ood_fallback_rate: float
    failures: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    object_task_claim: bool = False
    contact_task_claim: bool = False
    p4_full_completion_claim: bool = False

    def validate(self) -> None:
        if self.acceptance_version != ORDER3_ACCEPTANCE_VERSION:
            raise SchemaValidationError(
                f"Order3AcceptanceReport.acceptance_version must be {ORDER3_ACCEPTANCE_VERSION!r}"
            )
        for name in (
            "held_out_episode_count",
            "held_out_unique_morphology_count",
            "ood_episode_count",
            "safety_failure_episode_count",
        ):
            if int(getattr(self, name)) < 0:
                raise SchemaValidationError(
                    f"Order3AcceptanceReport.{name} must be non-negative"
                )
        for name in (
            "aggregate_held_out_success_rate",
            "id_fallback_rate",
            "ood_fallback_rate",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise SchemaValidationError(
                    f"Order3AcceptanceReport.{name} must be finite and in [0, 1]"
                )
        for module_count, value in self.per_module_success_rates.items():
            if module_count not in {str(count) for count in ORDER3_REQUIRED_MODULE_COUNTS}:
                raise SchemaValidationError(
                    "Order3AcceptanceReport.per_module_success_rates has an invalid module count"
                )
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise SchemaValidationError(
                    "Order3AcceptanceReport per-module success rates must be finite and in [0, 1]"
                )
        for name in (
            "nominal_relative_degradation",
            "randomized_relative_improvement",
            "randomized_success_gain",
        ):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise SchemaValidationError(
                    f"Order3AcceptanceReport.{name} must be finite when present"
                )
        if any(
            (
                self.object_task_claim,
                self.contact_task_claim,
                self.p4_full_completion_claim,
            )
        ):
            raise SchemaValidationError(
                "Order3 acceptance cannot claim object/contact/P4-full completion"
            )
        if self.completion_passed != self.pass_summary.completion_passed:
            raise SchemaValidationError(
                "Order3 acceptance completion must match the pass summary"
            )
        if self.completion_passed != (not self.failures):
            raise SchemaValidationError(
                "Order3 acceptance completion must match the absence of failures"
            )


def run_order3_acceptance(
    *,
    pool_manifest: Order3MorphologyPoolManifest | str | Path,
    dataset_manifest: Order3DatasetManifest | str | Path,
    checkpoint_path: str | Path,
    expected_checkpoint_sha256: str,
    episodes: Iterable[Order3EvaluationEpisode],
    artifact_metadata: Order3AcceptanceArtifactMetadata | Mapping[str, Any] | str | Path,
) -> Order3AcceptanceReport:
    """Recompute the complete Order-3 free-flight gate from bound evidence.

    Passing file paths is recommended: it lets the gate bind the exact dataset
    bytes to checkpoint metadata.  Typed manifest inputs use their canonical
    schema hash for the same binding.
    """

    pool, pool_source_hash, pool_source = _resolve_manifest(
        pool_manifest, Order3MorphologyPoolManifest
    )
    dataset, dataset_source_hash, dataset_source = _resolve_manifest(
        dataset_manifest, Order3DatasetManifest
    )
    artifact = _resolve_artifact_metadata(artifact_metadata)
    checkpoint = Path(checkpoint_path)
    values = list(episodes)
    failures: list[str] = []

    pool_counts, pool_hashes, pool_module_by_hash = _evaluate_pool(pool, failures)
    pool_hash = pool.stable_hash()
    pool_passed = not any(reason.startswith("pool_") for reason in failures)

    _evaluate_dataset(
        dataset,
        pool=pool,
        pool_hash=pool_hash,
        pool_hashes=pool_hashes,
        failures=failures,
    )
    dataset_passed = not any(reason.startswith("dataset_") for reason in failures)

    loaded_checkpoint = None
    if not _is_sha256(expected_checkpoint_sha256):
        failures.append("checkpoint_expected_sha256_invalid")
    elif not checkpoint.is_file():
        failures.append("checkpoint_missing")
    else:
        try:
            loaded_checkpoint = load_order3_policy_checkpoint(
                checkpoint,
                expected_sha256=expected_checkpoint_sha256,
            )
        except (OSError, RuntimeError, SchemaValidationError, TypeError, ValueError):
            failures.append("checkpoint_invalid_or_sha256_mismatch")
    if loaded_checkpoint is not None:
        metadata = loaded_checkpoint.metadata
        if metadata.training_stage not in {"ppo", "evaluation"}:
            failures.append("checkpoint_not_ppo_or_evaluation_stage")
        if metadata.pool_hash != dataset.pool_hash:
            failures.append("checkpoint_pool_hash_mismatch")
        if metadata.dataset_hash != dataset_source_hash:
            failures.append("checkpoint_dataset_hash_mismatch")
        if metadata.physical_model_hash != dataset.physical_model_hash:
            failures.append("checkpoint_physical_model_hash_mismatch")
        if metadata.actor_uses_privileged_wrench:
            failures.append("checkpoint_actor_privileged_wrench_enabled")
        if _truthy_claim_paths(metadata.to_dict()):
            failures.append("checkpoint_scope_claim_invalid")
    checkpoint_passed = not any(
        reason.startswith("checkpoint_") for reason in failures
    )

    _evaluate_artifact_metadata(
        artifact,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        dataset_source_hash=dataset_source_hash,
        pool_hash=pool_hash,
        evaluation_episode_set_hash=stable_hash(
            [episode.to_dict() for episode in values]
        ),
        loaded_checkpoint=loaded_checkpoint,
        failures=failures,
    )
    artifact_metadata_passed = not any(
        reason.startswith("artifact_") for reason in failures
    )

    evaluation = _evaluate_episodes(
        values,
        pool_hashes=pool_hashes,
        pool_module_by_hash=pool_module_by_hash,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_physical_model_hash=pool.physical_model_hash,
        failures=failures,
    )
    no_scope_mislabeling = not (
        _truthy_claim_paths(pool.metadata)
        or _truthy_claim_paths(dataset.metadata)
        or _truthy_claim_paths(artifact.to_dict())
        or (
            loaded_checkpoint is not None
            and _truthy_claim_paths(loaded_checkpoint.metadata.to_dict())
        )
        or any(
            episode.object_task_claim
            or episode.contact_task_claim
            or episode.p4_full_completion_claim
            for episode in values
        )
    )
    if not no_scope_mislabeling:
        failures.append("scope_claim_invalid")

    component_gates = {
        "pool_passed": pool_passed,
        "dataset_passed": dataset_passed,
        "checkpoint_passed": checkpoint_passed,
        "artifact_metadata_passed": artifact_metadata_passed,
        "held_out_coverage_passed": evaluation["coverage_passed"],
        "held_out_performance_passed": evaluation["performance_passed"],
        "safety_passed": evaluation["safety_passed"],
        "nominal_baseline_passed": evaluation["nominal_passed"],
        "randomized_robustness_passed": evaluation["randomized_passed"],
        "id_fallback_passed": evaluation["id_fallback_passed"],
        "ood_fallback_passed": evaluation["ood_fallback_passed"],
        "no_scope_mislabeling_passed": no_scope_mislabeling,
    }
    completion_passed = all(component_gates.values()) and not failures
    pass_summary = Order3AcceptancePassSummary(
        **component_gates,
        completion_passed=completion_passed,
    )
    return Order3AcceptanceReport(
        acceptance_version=ORDER3_ACCEPTANCE_VERSION,
        pass_summary=pass_summary,
        completion_passed=completion_passed,
        pool_split_module_counts=pool_counts,
        held_out_episode_count=evaluation["held_out_episode_count"],
        held_out_unique_morphology_count=evaluation[
            "held_out_unique_morphology_count"
        ],
        ood_episode_count=evaluation["ood_episode_count"],
        aggregate_held_out_success_rate=evaluation["aggregate_success_rate"],
        per_module_success_rates=evaluation["per_module_success_rates"],
        safety_failure_episode_count=evaluation["safety_failure_count"],
        nominal_relative_degradation=evaluation["nominal_degradation"],
        randomized_relative_improvement=evaluation["randomized_improvement"],
        randomized_success_gain=evaluation["randomized_success_gain"],
        id_fallback_rate=evaluation["id_fallback_rate"],
        ood_fallback_rate=evaluation["ood_fallback_rate"],
        failures=failures,
        artifacts={
            "pool_manifest": pool_source,
            "pool_manifest_sha256_or_canonical_hash": pool_source_hash,
            "dataset_manifest": dataset_source,
            "dataset_manifest_sha256_or_canonical_hash": dataset_source_hash,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": expected_checkpoint_sha256,
        },
    )


def run_order3_acceptance_from_paths(
    *,
    pool_manifest_path: str | Path,
    dataset_manifest_path: str | Path,
    checkpoint_path: str | Path,
    expected_checkpoint_sha256: str,
    episodes_path: str | Path,
    artifact_metadata_path: str | Path,
) -> Order3AcceptanceReport:
    """CLI-friendly path-only adapter for persisted JSON evidence."""

    episode_payload = json.loads(Path(episodes_path).read_text(encoding="utf-8"))
    if isinstance(episode_payload, dict):
        episode_payload = episode_payload.get("episodes")
    if not isinstance(episode_payload, list):
        raise SchemaValidationError(
            "Order3 evaluation episode artifact must be a list or {'episodes': [...]}"
        )
    episodes = [Order3EvaluationEpisode.from_dict(item) for item in episode_payload]
    return run_order3_acceptance(
        pool_manifest=pool_manifest_path,
        dataset_manifest=dataset_manifest_path,
        checkpoint_path=checkpoint_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        episodes=episodes,
        artifact_metadata=artifact_metadata_path,
    )


def _evaluate_pool(
    manifest: Order3MorphologyPoolManifest,
    failures: list[str],
) -> tuple[
    dict[str, dict[str, int]],
    dict[DatasetSplit, set[str]],
    dict[str, int],
]:
    counts = {
        split.value: {str(module_count): 0 for module_count in ORDER3_REQUIRED_MODULE_COUNTS}
        for split in DatasetSplit
    }
    hashes = {split: set() for split in DatasetSplit}
    module_by_hash: dict[str, int] = {}
    for entry in manifest.entries:
        try:
            computed = morphology_structural_hash(entry.morphology_graph)
        except (SchemaValidationError, TypeError, ValueError):
            failures.append("pool_morphology_graph_invalid")
            continue
        if computed != entry.structural_hash:
            failures.append("pool_structural_hash_mismatch")
        if entry.module_count != len(entry.morphology_graph.modules):
            failures.append("pool_module_count_mismatch")
        if entry.structural_hash in module_by_hash:
            failures.append("pool_structural_hash_not_globally_disjoint")
        module_by_hash[entry.structural_hash] = entry.module_count
        hashes[entry.split].add(entry.structural_hash)
        if entry.module_count in ORDER3_REQUIRED_MODULE_COUNTS:
            counts[entry.split.value][str(entry.module_count)] += 1

    for module_count in ORDER3_REQUIRED_MODULE_COUNTS:
        required = recommended_order3_morphology_split_counts(module_count)
        for split in DatasetSplit:
            if counts[split.value][str(module_count)] != required[split]:
                failures.append(
                    f"pool_quota_mismatch:{split.value}:n{module_count}"
                )
    for index, left in enumerate(DatasetSplit):
        for right in list(DatasetSplit)[index + 1 :]:
            if hashes[left].intersection(hashes[right]):
                failures.append(
                    f"pool_split_hash_overlap:{left.value}:{right.value}"
                )
    return counts, hashes, module_by_hash


def _evaluate_dataset(
    manifest: Order3DatasetManifest,
    *,
    pool: Order3MorphologyPoolManifest,
    pool_hash: str,
    pool_hashes: dict[DatasetSplit, set[str]],
    failures: list[str],
) -> None:
    dataset_hashes = {
        split: set(manifest.morphology_hashes.get(split.value, []))
        for split in DatasetSplit
    }
    for index, left in enumerate(DatasetSplit):
        for right in list(DatasetSplit)[index + 1 :]:
            if dataset_hashes[left].intersection(dataset_hashes[right]):
                failures.append(
                    f"dataset_split_hash_overlap:{left.value}:{right.value}"
                )
    for split in DatasetSplit:
        if dataset_hashes[split] != pool_hashes[split]:
            failures.append(f"dataset_pool_hash_set_mismatch:{split.value}")
    if manifest.pool_hash != pool_hash:
        failures.append("dataset_pool_manifest_hash_mismatch")
    if manifest.physical_model_hash != pool.physical_model_hash:
        failures.append("dataset_pool_physical_model_hash_mismatch")
    if manifest.actor_privileged_wrench_inputs:
        failures.append("dataset_actor_privileged_wrench_enabled")
    if _truthy_claim_paths(manifest.metadata):
        failures.append("dataset_scope_claim_invalid")


def _evaluate_artifact_metadata(
    artifact: Order3AcceptanceArtifactMetadata,
    *,
    expected_checkpoint_sha256: str,
    dataset_source_hash: str,
    pool_hash: str,
    evaluation_episode_set_hash: str,
    loaded_checkpoint,
    failures: list[str],
) -> None:
    expected = {
        "artifact_version": ORDER3_ACCEPTANCE_ARTIFACT_VERSION,
        "evaluation_scope_version": ORDER3_FREE_FLIGHT_VERSION,
        "evaluation_source": "real_isaac_paired_learned_and_deterministic_v2",
        "checkpoint_sha256": expected_checkpoint_sha256,
        "dataset_manifest_sha256": dataset_source_hash,
        "policy_family": ORDER3_POLICY_FAMILY,
        "policy_contract_version": POLICY_COMMAND_CONTRACT_CENTROIDAL,
        "architecture_version": ORDER3_POLICY_ARCHITECTURE_VERSION,
        "tensorizer_version": ORDER3_TENSORIZER_VERSION,
        "encoder_version": ORDER3_ENCODER_VERSION,
        "graph_encoder_used": True,
        "recurrent_gru_used": True,
        "actor_uses_privileged_wrench": False,
        "deterministic_fallback_available": True,
        "pool_hash": pool_hash,
        "evaluation_episode_set_hash": evaluation_episode_set_hash,
        "rollout_condition_version": ORDER3_ROLLOUT_CONDITION_VERSION,
        "raw_report_hashes_bound": True,
        "paired_deterministic_baseline": True,
        "required_task_modes": sorted(mode.value for mode in Order3TaskMode),
        "object_task_claim": False,
        "contact_task_claim": False,
        "p4_full_completion_claim": False,
    }
    for name, expected_value in expected.items():
        if getattr(artifact, name) != expected_value:
            failures.append(f"artifact_metadata_mismatch:{name}")
    if loaded_checkpoint is not None:
        checkpoint_metadata = loaded_checkpoint.metadata
        for name in (
            "policy_family",
            "policy_contract_version",
            "architecture_version",
            "tensorizer_version",
            "encoder_version",
            "actor_uses_privileged_wrench",
        ):
            if getattr(artifact, name) != getattr(checkpoint_metadata, name):
                failures.append(f"artifact_checkpoint_metadata_mismatch:{name}")


def _evaluate_episodes(
    episodes: list[Order3EvaluationEpisode],
    *,
    pool_hashes: dict[DatasetSplit, set[str]],
    pool_module_by_hash: dict[str, int],
    expected_checkpoint_sha256: str,
    expected_physical_model_hash: str,
    failures: list[str],
) -> dict[str, Any]:
    all_pool_hashes = set().union(*pool_hashes.values())
    held_out_hashes = pool_hashes[DatasetSplit.HELD_OUT]
    integrity_passed = True
    if len({episode.episode_id for episode in episodes}) != len(episodes):
        failures.append("evaluation_duplicate_episode_id")
        integrity_passed = False

    id_episodes: list[Order3EvaluationEpisode] = []
    held_out: list[Order3EvaluationEpisode] = []
    ood: list[Order3EvaluationEpisode] = []
    for episode in episodes:
        if not _is_sha256(episode.structural_hash):
            failures.append("evaluation_structural_hash_invalid")
            integrity_passed = False
        is_ood = episode.structural_hash not in all_pool_hashes
        if not _validate_bound_episode_reports(
            episode,
            is_ood=is_ood,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            expected_physical_model_hash=expected_physical_model_hash,
            failures=failures,
        ):
            integrity_passed = False
        if not is_ood:
            id_episodes.append(episode)
            expected_count = pool_module_by_hash[episode.structural_hash]
            if episode.module_count != expected_count:
                failures.append("evaluation_module_count_mismatch")
                integrity_passed = False
            expected_split = next(
                split
                for split, values in pool_hashes.items()
                if episode.structural_hash in values
            )
            if episode.split != expected_split:
                failures.append("evaluation_pool_split_mismatch")
                integrity_passed = False
            if episode.structural_hash in held_out_hashes:
                held_out.append(episode)
        else:
            ood.append(episode)

    covered_hashes = {episode.structural_hash for episode in held_out}
    required_matrix = {
        (structural_hash, task_mode, randomized)
        for structural_hash in held_out_hashes
        for task_mode in Order3TaskMode
        for randomized in (False, True)
    }
    observed_matrix = {
        (episode.structural_hash, episode.task_mode, episode.randomized)
        for episode in held_out
    }
    coverage_passed = (
        integrity_passed
        and covered_hashes == held_out_hashes
        and required_matrix.issubset(observed_matrix)
    )
    if covered_hashes != held_out_hashes:
        failures.append("evaluation_held_out_morphology_coverage_incomplete")
    if not required_matrix.issubset(observed_matrix):
        failures.append("evaluation_task_mode_randomization_matrix_incomplete")

    recomputed_success: dict[str, bool] = {}
    for episode in held_out:
        if episode.terminal_metrics is None:
            failures.append("evaluation_terminal_metrics_missing")
            recomputed_success[episode.episode_id] = False
            continue
        value = order3_terminal_metrics_success(
            episode.terminal_metrics,
            task_mode=episode.task_mode,
        )
        if value != episode.success:
            failures.append("evaluation_producer_success_mismatch")
        recomputed_success[episode.episode_id] = value

    aggregate_success = _ratio(
        sum(recomputed_success.get(episode.episode_id, False) for episode in held_out),
        len(held_out),
    )
    per_module_rates = {
        str(module_count): _ratio(
            sum(
                recomputed_success.get(episode.episode_id, False)
                for episode in held_out
                if episode.module_count == module_count
            ),
            sum(episode.module_count == module_count for episode in held_out),
        )
        for module_count in ORDER3_REQUIRED_MODULE_COUNTS
    }
    performance_passed = (
        bool(held_out)
        and all(episode.terminal_metrics is not None for episode in held_out)
        and aggregate_success + 1.0e-12 >= ORDER3_AGGREGATE_SUCCESS_THRESHOLD
        and all(
            rate + 1.0e-12 >= ORDER3_PER_MODULE_SUCCESS_THRESHOLD
            for rate in per_module_rates.values()
        )
    )
    if not performance_passed:
        failures.append("evaluation_held_out_success_threshold_failed")

    safety_failure_count = sum(episode.safety_failure for episode in episodes)
    safety_passed = safety_failure_count == 0
    if not safety_passed:
        failures.append("evaluation_safety_terminal_present")

    nominal = [episode for episode in held_out if not episode.randomized]
    nominal_degradation = _mean_relative_change(
        nominal,
        numerator=lambda episode: episode.tracking_cost
        - float(episode.deterministic_baseline_tracking_cost or 0.0),
    )
    nominal_has_evidence = bool(nominal) and all(
        episode.deterministic_baseline_tracking_cost is not None
        for episode in nominal
    )
    nominal_passed = (
        nominal_has_evidence
        and nominal_degradation is not None
        and nominal_degradation <= ORDER3_NOMINAL_MAX_RELATIVE_DEGRADATION + 1.0e-12
    )
    if not nominal_passed:
        failures.append("evaluation_nominal_baseline_degradation_failed")

    randomized = [episode for episode in held_out if episode.randomized]
    randomized_has_tracking_evidence = bool(randomized) and all(
        episode.deterministic_baseline_tracking_cost is not None
        for episode in randomized
    )
    randomized_improvement = _mean_relative_change(
        randomized,
        numerator=lambda episode: float(
            episode.deterministic_baseline_tracking_cost or 0.0
        )
        - episode.tracking_cost,
    )
    randomized_success_gain: float | None = None
    if randomized and all(
        episode.deterministic_baseline_terminal_metrics is not None
        for episode in randomized
    ):
        baseline_successes = [
            order3_terminal_metrics_success(
                episode.deterministic_baseline_terminal_metrics,
                task_mode=episode.task_mode,
            )
            for episode in randomized
        ]
        learned_successes = [
            recomputed_success.get(episode.episode_id, False)
            for episode in randomized
        ]
        randomized_success_gain = _ratio(
            sum(learned_successes), len(learned_successes)
        ) - _ratio(sum(baseline_successes), len(baseline_successes))
    randomized_tracking_passed = (
        randomized_has_tracking_evidence
        and randomized_improvement is not None
        and randomized_improvement + 1.0e-12
        >= ORDER3_RANDOMIZED_MIN_RELATIVE_IMPROVEMENT
    )
    randomized_success_passed = (
        randomized_success_gain is not None
        and randomized_success_gain + 1.0e-12
        >= ORDER3_RANDOMIZED_MIN_SUCCESS_GAIN
    )
    randomized_passed = randomized_tracking_passed or randomized_success_passed
    if not randomized_passed:
        failures.append("evaluation_randomized_robustness_failed")

    id_fallback_rate = _ratio(
        sum(episode.fallback_used for episode in id_episodes), len(id_episodes)
    )
    id_fallback_passed = bool(id_episodes) and (
        id_fallback_rate <= ORDER3_ID_MAX_FALLBACK_RATE + 1.0e-12
    )
    if not id_fallback_passed:
        failures.append("evaluation_id_fallback_rate_failed")

    ood_fallback_rate = _ratio(sum(episode.fallback_used for episode in ood), len(ood))
    ood_fallback_passed = bool(ood) and all(
        episode.fallback_used and episode.fallback_reason == "structural_hash_ood"
        for episode in ood
    )
    if not ood_fallback_passed:
        failures.append("evaluation_ood_fallback_evidence_missing")

    return {
        "coverage_passed": coverage_passed,
        "performance_passed": performance_passed,
        "safety_passed": safety_passed,
        "nominal_passed": nominal_passed,
        "randomized_passed": randomized_passed,
        "id_fallback_passed": id_fallback_passed,
        "ood_fallback_passed": ood_fallback_passed,
        "held_out_episode_count": len(held_out),
        "held_out_unique_morphology_count": len(covered_hashes),
        "ood_episode_count": len(ood),
        "aggregate_success_rate": aggregate_success,
        "per_module_success_rates": per_module_rates,
        "safety_failure_count": safety_failure_count,
        "nominal_degradation": nominal_degradation,
        "randomized_improvement": randomized_improvement,
        "randomized_success_gain": randomized_success_gain,
        "id_fallback_rate": id_fallback_rate,
        "ood_fallback_rate": ood_fallback_rate,
    }


def _validate_bound_episode_reports(
    episode: Order3EvaluationEpisode,
    *,
    is_ood: bool,
    expected_checkpoint_sha256: str,
    expected_physical_model_hash: str,
    failures: list[str],
) -> bool:
    prefix = f"evaluation_report:{episode.episode_id}"
    required_values = {
        "condition_hash": episode.condition_hash,
        "condition_seed": episode.condition_seed,
        "checkpoint_sha256": episode.checkpoint_sha256,
        "learned_report_path": episode.learned_report_path,
        "learned_report_sha256": episode.learned_report_sha256,
        "deterministic_baseline_report_path": (
            episode.deterministic_baseline_report_path
        ),
        "deterministic_baseline_report_sha256": (
            episode.deterministic_baseline_report_sha256
        ),
    }
    missing = [name for name, value in required_values.items() if value is None]
    if missing or not episode.isaac_backed:
        failures.append(f"{prefix}:provenance_missing")
        return False
    if episode.checkpoint_sha256 != expected_checkpoint_sha256:
        failures.append(f"{prefix}:checkpoint_mismatch")
        return False
    learned = _load_bound_report(
        str(episode.learned_report_path),
        str(episode.learned_report_sha256),
        prefix=f"{prefix}:learned",
        failures=failures,
    )
    baseline = _load_bound_report(
        str(episode.deterministic_baseline_report_path),
        str(episode.deterministic_baseline_report_sha256),
        prefix=f"{prefix}:baseline",
        failures=failures,
    )
    if learned is None or baseline is None:
        return False
    valid = True

    def mismatch(name: str) -> None:
        nonlocal valid
        failures.append(f"{prefix}:{name}")
        valid = False

    for report, policy_kind in ((learned, "learned"), (baseline, "baseline")):
        if report.get("isaac_backed") is not True:
            mismatch(f"{policy_kind}_not_isaac_backed")
        if report.get("order3_report_validation_failures", []) != []:
            mismatch(f"{policy_kind}_validation_failures")
        if report.get("order3_task_mode") != episode.task_mode.value:
            mismatch(f"{policy_kind}_task_mode_mismatch")
        if report.get("order3_structural_hash") != episode.structural_hash:
            mismatch(f"{policy_kind}_structural_hash_mismatch")
        if report.get("order3_rollout_condition_hash") != episode.condition_hash:
            mismatch(f"{policy_kind}_condition_hash_mismatch")
        condition_value = report.get("order3_rollout_condition")
        try:
            condition = (
                Order3RolloutCondition.from_json(condition_value)
                if isinstance(condition_value, str)
                else Order3RolloutCondition.from_dict(condition_value)
            )
        except (SchemaValidationError, TypeError, ValueError, json.JSONDecodeError):
            mismatch(f"{policy_kind}_condition_invalid")
            continue
        if condition.condition_hash != episode.condition_hash:
            mismatch(f"{policy_kind}_condition_payload_mismatch")
        if condition.seed != episode.condition_seed:
            mismatch(f"{policy_kind}_condition_seed_mismatch")
        if condition.task_mode != episode.task_mode.value:
            mismatch(f"{policy_kind}_condition_task_mode_mismatch")
        if _condition_is_randomized(condition) != episode.randomized:
            mismatch(f"{policy_kind}_randomized_label_mismatch")
        condition_failures = order3_condition_report_failures(
            report,
            expected_condition=condition,
        )
        for condition_failure in condition_failures:
            mismatch(
                f"{policy_kind}_condition_realization_invalid:"
                f"{condition_failure}"
            )

    provenance_keys = (
        "random_morphology_takeoff_backend_config_hash",
        "random_morphology_takeoff_physical_model_hash",
        "random_morphology_takeoff_collision_geometry_hash",
    )
    for key in provenance_keys:
        learned_value = learned.get(key)
        baseline_value = baseline.get(key)
        if (
            not isinstance(learned_value, str)
            or not _is_sha256(learned_value)
            or learned_value != baseline_value
        ):
            mismatch(f"paired_{key}_mismatch")
    if (
        learned.get("random_morphology_takeoff_physical_model_hash")
        != expected_physical_model_hash
    ):
        mismatch("physical_model_hash_mismatch")

    if learned.get("order3_pi_l_rollout") is not True:
        mismatch("learned_policy_marker_missing")
    if learned.get("order3_pi_l_checkpoint_sha256") != expected_checkpoint_sha256:
        mismatch("learned_checkpoint_mismatch")
    if baseline.get("order3_deterministic_baseline_rollout") is not True:
        mismatch("baseline_marker_missing")

    fallback_used, fallback_reason = _report_fallback(learned)
    if fallback_used != episode.fallback_used or fallback_reason != episode.fallback_reason:
        mismatch("fallback_evidence_mismatch")
    safety = {
        "qp_infeasible": episode.qp_infeasible,
        "hard_collision": episode.hard_collision,
        "non_finite_state": episode.non_finite_state,
        "unsupported_actuator": episode.unsupported_actuator,
    }
    for name, expected in safety.items():
        if learned.get(f"order3_{name}") is not expected:
            mismatch(f"learned_{name}_mismatch")
        if baseline.get(f"order3_{name}") is not False:
            mismatch(f"baseline_{name}_present")
    if is_ood:
        if not fallback_used or fallback_reason != "structural_hash_ood":
            mismatch("ood_fallback_reason_mismatch")
        return valid

    learned_metrics = _report_terminal_metrics(learned)
    baseline_metrics = _report_terminal_metrics(baseline)
    if learned_metrics != episode.terminal_metrics:
        mismatch("learned_terminal_metrics_mismatch")
    if baseline_metrics != episode.deterministic_baseline_terminal_metrics:
        mismatch("baseline_terminal_metrics_mismatch")
    if not _float_equal(learned.get("order3_free_flight_tracking_cost"), episode.tracking_cost):
        mismatch("learned_tracking_cost_mismatch")
    if not _float_equal(
        baseline.get("order3_free_flight_tracking_cost"),
        episode.deterministic_baseline_tracking_cost,
    ):
        mismatch("baseline_tracking_cost_mismatch")
    if learned.get("order3_free_flight_success") is not episode.success:
        mismatch("learned_success_mismatch")
    return valid


def _load_bound_report(
    path_value: str,
    expected_sha256: str,
    *,
    prefix: str,
    failures: list[str],
) -> dict[str, Any] | None:
    path = Path(path_value)
    if not path.is_file() or hash_file(path) != expected_sha256:
        failures.append(f"{prefix}:missing_or_hash_mismatch")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        failures.append(f"{prefix}:invalid_json")
        return None
    if not isinstance(payload, dict):
        failures.append(f"{prefix}:invalid_payload")
        return None
    if isinstance(payload.get("takeoff_result"), dict):
        candidate = payload["takeoff_result"].get("report")
        if isinstance(candidate, dict):
            return candidate
    if isinstance(payload.get("free_flight_result"), dict):
        candidate = payload["free_flight_result"].get("report")
        if isinstance(candidate, dict):
            return candidate
    if isinstance(payload.get("report"), dict):
        return payload["report"]
    return payload


def _condition_is_randomized(condition: Order3RolloutCondition) -> bool:
    vectors = (
        condition.initial_position_offset_world,
        condition.initial_orientation_rpy_rad,
        condition.initial_linear_velocity_world,
        condition.initial_angular_velocity_body,
        condition.external_wrench_body,
    )
    return bool(
        any(abs(float(value)) > 1.0e-12 for vector in vectors for value in vector)
        or any(
            abs(float(value) - 1.0) > 1.0e-12
            for value in (
                condition.mass_scale,
                condition.inertia_scale,
                condition.thrust_scale,
            )
        )
    )


def _report_fallback(report: Mapping[str, Any]) -> tuple[bool, str | None]:
    direct = report.get("order3_fallback_used")
    reason = report.get("order3_fallback_reason")
    if isinstance(direct, bool):
        return direct, str(reason) if reason is not None else None
    count = report.get("order3_pi_l_fallback_count")
    used = isinstance(count, int) and not isinstance(count, bool) and count > 0
    reasons = {
        trace.get("fallback_reason")
        for trace in report.get("order3_pi_l_transition_traces", [])
        if isinstance(trace, dict) and trace.get("fallback_reason") is not None
    }
    resolved = next(iter(reasons)) if len(reasons) == 1 else None
    return used, str(resolved) if resolved is not None else None


def _report_terminal_metrics(report: Mapping[str, Any]) -> Order3TerminalMetrics | None:
    value = report.get("order3_free_flight_terminal_metrics")
    if not isinstance(value, dict):
        return None
    try:
        return Order3TerminalMetrics.from_dict(value)
    except (SchemaValidationError, TypeError, ValueError):
        return None


def _float_equal(left: Any, right: Any) -> bool:
    if right is None:
        return left is None
    return (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1.0e-9)
    )


def _mean_relative_change(
    episodes: list[Order3EvaluationEpisode],
    *,
    numerator,
) -> float | None:
    if not episodes or any(
        episode.deterministic_baseline_tracking_cost is None for episode in episodes
    ):
        return None
    values = [
        float(numerator(episode))
        / max(float(episode.deterministic_baseline_tracking_cost or 0.0), 1.0e-6)
        for episode in episodes
    ]
    return sum(values) / len(values)


def _resolve_manifest(value, manifest_type):
    if isinstance(value, manifest_type):
        return value, value.stable_hash(), "<in-memory-canonical-manifest>"
    path = Path(value)
    manifest = manifest_type.from_json(path.read_text(encoding="utf-8"))
    return manifest, hash_file(path), str(path)


def _resolve_artifact_metadata(
    value: Order3AcceptanceArtifactMetadata | Mapping[str, Any] | str | Path,
) -> Order3AcceptanceArtifactMetadata:
    if isinstance(value, Order3AcceptanceArtifactMetadata):
        return value
    if isinstance(value, Mapping):
        return Order3AcceptanceArtifactMetadata.from_dict(dict(value))
    return Order3AcceptanceArtifactMetadata.from_json(
        Path(value).read_text(encoding="utf-8")
    )


def _truthy_claim_paths(value: Any, path: str = "") -> list[str]:
    output: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            if str(key) in _FORBIDDEN_TRUE_CLAIM_KEYS and bool(item):
                output.append(child)
            output.extend(_truthy_claim_paths(item, child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            output.extend(_truthy_claim_paths(item, f"{path}[{index}]"))
    return output


def _is_sha256(value: str) -> bool:
    return len(value) == _SHA256_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Order-3 free-flight acceptance gate")
    parser.add_argument("--pool-manifest", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--artifact-metadata", required=True)
    arguments = parser.parse_args(argv)
    report = run_order3_acceptance_from_paths(
        pool_manifest_path=arguments.pool_manifest,
        dataset_manifest_path=arguments.dataset_manifest,
        checkpoint_path=arguments.checkpoint,
        expected_checkpoint_sha256=arguments.checkpoint_sha256,
        episodes_path=arguments.episodes,
        artifact_metadata_path=arguments.artifact_metadata,
    )
    print(report.to_json(indent=2))
    return 0 if report.pass_summary.completion_passed else 1


if __name__ == "__main__":  # pragma: no cover - exercised by the path adapter tests.
    raise SystemExit(_main())
