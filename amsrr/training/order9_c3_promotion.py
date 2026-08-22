from __future__ import annotations

"""Resumable, hash-bound helpers for the formal C3 promotion evaluation."""

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.order9 import Order9StageRunManifest
from amsrr.simulation.order9_tensor_object_task import (
    ORDER9_PAYLOAD_FEEDFORWARD_PHASE_CONTRACT,
)
from amsrr.training.order9_evaluation import (
    ORDER9_EVALUATION_HORIZON_TIMEOUT_CONTRACT,
    Order9EvaluationEpisode,
    order9_evaluation_horizon_timeout_outcome,
    write_order9_evaluation_episodes_jsonl,
)
from amsrr.training.order9_c3_boundary_sampling import (
    ORDER9_C3_BOUNDARY_TAIL_SAMPLING_CONTRACT,
)
from amsrr.training.order9_c3_nominal_runtime import (
    ORDER9_C3_FORMAL_PHASE_ZERO_STATE_SOURCE,
)
from amsrr.training.order9_pi_l_stage_runner import (
    ORDER9_ROLLOUT_RESULT_PREFIX,
    load_order9_rollout_result,
)
from amsrr.training.order9_tensor_reward import (
    ORDER9_C3_PRIVILEGED_PHASE_SUPERVISION_CONTRACT,
    ORDER9_CONTACT_ACQUISITION_SUCCESS_CONTRACT,
    ORDER9_OBJECT_POSITION_TOLERANCE_M,
    ORDER9_OBJECT_POSE_SUCCESS_CONTRACT,
    ORDER9_RELEASE_JOINT_POSITION_TOLERANCE_RAD,
    ORDER9_RELEASE_SUCCESS_CONTRACT,
)
from amsrr.training.order9_tensor_rollout_artifact import (
    ORDER9_PRODUCTION_COLLECTOR_VERSION,
    load_order9_tensor_rollout_artifact,
)
from amsrr.training.order9_rollout_buckets import (
    Order9PiLRolloutBucket,
    Order9PiLRolloutBucketManifest,
)
from amsrr.utils.hashing import hash_file


ORDER9_C3_PROMOTION_RUNNER_VERSION = (
    "order9_c3_promotion_runner_v19_persistent_same_morphology_process"
)
ORDER9_C3_HORIZON_TIMEOUT_RECOVERY_VERSION = (
    "order9_c3_horizon_timeout_recovery_v1_hash_bound_raw"
)


def order9_c3_promotion_success_upper_bound(
    *,
    completed_success_count: int,
    completed_episode_count: int,
    total_episode_count: int,
    minimum_success_rate: float,
) -> dict[str, object] | None:
    """Return fail-fast evidence when the success-rate gate is unreachable."""

    if not 0 <= completed_success_count <= completed_episode_count:
        raise ValueError("completed C3 promotion counts are inconsistent")
    if not 0 <= completed_episode_count <= total_episode_count:
        raise ValueError("completed C3 episodes exceed the formal total")
    if not math.isfinite(minimum_success_rate) or not 0.0 <= minimum_success_rate <= 1.0:
        raise ValueError("C3 minimum success rate must be in [0, 1]")
    if total_episode_count < 1:
        raise ValueError("C3 formal episode total must be positive")
    remaining_episode_count = total_episode_count - completed_episode_count
    maximum_possible_success_count = (
        completed_success_count + remaining_episode_count
    )
    maximum_possible_success_rate = (
        maximum_possible_success_count / total_episode_count
    )
    if maximum_possible_success_rate >= minimum_success_rate:
        return None
    return {
        "contract": "order9_c3_promotion_success_upper_bound_v1",
        "minimum_success_rate": float(minimum_success_rate),
        "total_episode_count": total_episode_count,
        "completed_episode_count": completed_episode_count,
        "completed_success_count": completed_success_count,
        "remaining_episode_count": remaining_episode_count,
        "maximum_possible_success_count": maximum_possible_success_count,
        "maximum_possible_success_rate": maximum_possible_success_rate,
        "failed_gate": "minimum_success_rate",
    }


def _horizon_timeout_recovery_path(episode_jsonl_path: str | Path) -> Path:
    return Path(episode_jsonl_path).with_name(
        "evaluation_horizon_timeout_recovery.json"
    )


def recover_order9_c3_horizon_timeout_evidence(
    *,
    log_path: str | Path,
    raw_artifact_path: str | Path,
    episode_jsonl_path: str | Path,
    stage_id: str,
    generation_id: str,
    checkpoint_sha256: str,
    expected_episode_count: int,
) -> tuple[Order9EvaluationEpisode, ...]:
    """Recover fail-closed timeout rows from a complete hash-bound rollout.

    Collector v32 wrote the full raw tensor artifact before rejecting a run
    with no first terminal.  Such a run is complete physical evidence of an
    evaluation-horizon task failure, so derive the missing JSONL rows without
    repeating Isaac.  Every byte that determines the rows remains bound by the
    raw-artifact SHA and a separate recovery manifest.
    """

    log = Path(log_path).resolve()
    raw = Path(raw_artifact_path).resolve()
    episode_jsonl = Path(episode_jsonl_path).resolve()
    if expected_episode_count < 1:
        raise ValueError("expected_episode_count must be positive")
    if not log.is_file() or not raw.is_file():
        raise SchemaValidationError("C3 timeout recovery source is incomplete")
    log_text = log.read_text(encoding="utf-8", errors="replace")
    expected_error = (
        "Order9 evaluation rollout produced 0 first-terminal episodes; "
        f"{expected_episode_count} required"
    )
    if expected_error not in log_text or checkpoint_sha256 not in log_text:
        raise SchemaValidationError("C3 timeout recovery log contract differs")

    raw_sha256 = hash_file(raw)
    artifact = load_order9_tensor_rollout_artifact(
        raw, expected_sha256=raw_sha256
    )
    metadata = artifact.metadata
    expected_metadata = {
        "stage_id": stage_id,
        "generation_id": generation_id,
        "pi_l_checkpoint_sha256": checkpoint_sha256,
        "evaluation_mode": True,
        "deterministic_policy": True,
        "initial_phase_zero": True,
        "formal_phase_zero_start": True,
        "formal_phase_zero_state_source": (
            ORDER9_C3_FORMAL_PHASE_ZERO_STATE_SOURCE
        ),
        "initial_phase_elapsed_s": 0.0,
        "terminal_count": 0,
        "successful_terminal_count": 0,
        "raw_contact_actor_input": False,
        "controller_side_contact_preload_version": None,
        "collector_version": ORDER9_PRODUCTION_COLLECTOR_VERSION,
        "phase_supervision_source": "privileged_physx_contact_outcome",
        "c3_privileged_phase_supervision_contract": (
            ORDER9_C3_PRIVILEGED_PHASE_SUPERVISION_CONTRACT
        ),
        "contact_acquisition_success_contract": (
            ORDER9_CONTACT_ACQUISITION_SUCCESS_CONTRACT
        ),
        "wrench_range_gate_scale": 1.0,
        "wrench_range_hard_gate_enabled": False,
        "exact_wrench_reward_bounds_preserved": True,
    }
    for name, value in expected_metadata.items():
        if metadata.get(name) != value:
            raise SchemaValidationError(
                f"C3 timeout recovery raw metadata differs at {name}"
            )
    if (
        artifact.environment_count < expected_episode_count
        or artifact.step_count < 1
        or int(metadata.get("requested_rollout_steps", -1))
        != artifact.step_count
        or int(metadata.get("rollout_steps", -1)) != artifact.step_count
        or bool(artifact.tensors["terminal"][:, :expected_episode_count].any())
        or not bool(
            artifact.tensors["valid"][:, :expected_episode_count].all()
        )
        or not bool(
            artifact.tensors["truncated"][-1, :expected_episode_count].all()
        )
        or bool(
            artifact.tensors[
                "prohibited_collision"
            ][:, :expected_episode_count].any()
        )
    ):
        raise SchemaValidationError(
            "C3 timeout recovery raw rollout is not a complete horizon"
        )
    task_specs = metadata.get("task_specs")
    environment_splits = metadata.get("environment_splits")
    if (
        not isinstance(task_specs, list)
        or not isinstance(environment_splits, list)
        or len(task_specs) < expected_episode_count
        or len(environment_splits) < expected_episode_count
    ):
        raise SchemaValidationError("C3 timeout recovery task metadata differs")

    episodes: list[Order9EvaluationEpisode] = []
    random_seed = int(metadata.get("random_seed", -1))
    if random_seed < 0:
        raise SchemaValidationError("C3 timeout recovery seed is invalid")
    for environment in range(expected_episode_count):
        valid = artifact.tensors["valid"][:, environment]
        step_count = int(valid.sum().item())
        final_step = int(valid.nonzero(as_tuple=False)[-1].item())
        outcome = order9_evaluation_horizon_timeout_outcome(
            environment=environment,
            environment_step_count=step_count,
            episode_return=float(
                artifact.tensors["reward"][:, environment][valid].sum().item()
            ),
            terminal_phase_index=int(
                artifact.tensors["phase_index"][final_step, environment].item()
            ),
        )
        task_spec = task_specs[environment]
        if not isinstance(task_spec, dict) or not str(task_spec.get("task_id", "")):
            raise SchemaValidationError("C3 timeout recovery task id is invalid")
        episodes.append(
            Order9EvaluationEpisode(
                episode_id=(
                    f"{generation_id}:evaluation:env:{environment:04d}"
                ),
                task_id=str(task_spec["task_id"]),
                split=DatasetSplit(environment_splits[environment]),
                random_seed=random_seed + environment,
                task_success=False,
                no_fallback_success=False,
                safety_failure=False,
                high_level_decision_count=0,
                fallback_decision_count=0,
                environment_step_count=step_count,
                isaac_backed=True,
                full_mesh_evaluation=True,
                source_artifact_path=str(raw),
                source_artifact_sha256=raw_sha256,
                failure_reason=str(outcome["failure_reason"]),
                metrics={
                    "episode_return": float(outcome["episode_return"]),
                    "terminal_phase_index": float(
                        outcome["terminal_phase_index"]
                    ),
                    "hard_collision": 0.0,
                    "object_dropped": 0.0,
                    "qp_infeasible_terminal": 0.0,
                    "timeout": 1.0,
                    "evaluation_horizon_timeout": 1.0,
                },
                metadata={
                    "environment_index": environment,
                    "generation_id": generation_id,
                    "deterministic_policy": True,
                    "initial_phase_index": 0,
                    "initial_phase_stratum_index": -1,
                    "formal_phase_zero_start": True,
                    "formal_phase_zero_state_source": (
                        ORDER9_C3_FORMAL_PHASE_ZERO_STATE_SOURCE
                    ),
                    "initial_phase_elapsed_s": 0.0,
                    "first_terminal_only": True,
                    "first_terminal_or_horizon_timeout": True,
                    "evaluation_horizon_timeout_contract": (
                        ORDER9_EVALUATION_HORIZON_TIMEOUT_CONTRACT
                    ),
                    "evaluation_horizon_timeout": True,
                    "recovered_from_complete_raw_rollout": True,
                    "raw_contact_actor_input": False,
                    "payload_feedforward_phase_contract": (
                        ORDER9_PAYLOAD_FEEDFORWARD_PHASE_CONTRACT
                    ),
                    "contact_acquisition_success_contract": (
                        ORDER9_CONTACT_ACQUISITION_SUCCESS_CONTRACT
                    ),
                    "wrench_range_hard_gate_enabled": False,
                    "c3_privileged_phase_supervision_contract": (
                        ORDER9_C3_PRIVILEGED_PHASE_SUPERVISION_CONTRACT
                    ),
                    "release_success_contract": ORDER9_RELEASE_SUCCESS_CONTRACT,
                    "release_joint_position_tolerance_rad": float(
                        ORDER9_RELEASE_JOINT_POSITION_TOLERANCE_RAD
                    ),
                    "object_pose_success_contract": (
                        ORDER9_OBJECT_POSE_SUCCESS_CONTRACT
                    ),
                    "object_position_tolerance_m": float(
                        ORDER9_OBJECT_POSITION_TOLERANCE_M
                    ),
                    "c3_boundary_tail_sampling_contract": (
                        ORDER9_C3_BOUNDARY_TAIL_SAMPLING_CONTRACT
                    ),
                    "controller_side_contact_preload_version": None,
                },
            )
        )
    write_order9_evaluation_episodes_jsonl(episode_jsonl, episodes)
    recovery_path = _horizon_timeout_recovery_path(episode_jsonl)
    write_order9_c3_promotion_state(
        recovery_path,
        {
            "recovery_version": ORDER9_C3_HORIZON_TIMEOUT_RECOVERY_VERSION,
            "evaluation_horizon_timeout_contract": (
                ORDER9_EVALUATION_HORIZON_TIMEOUT_CONTRACT
            ),
            "stage_id": stage_id,
            "generation_id": generation_id,
            "checkpoint_sha256": checkpoint_sha256,
            "raw_artifact_path": str(raw),
            "raw_artifact_sha256": raw_sha256,
            "source_log_path": str(log),
            "source_log_sha256": hash_file(log),
            "episode_jsonl_path": str(episode_jsonl),
            "episode_jsonl_sha256": hash_file(episode_jsonl),
            "episode_count": len(episodes),
        },
    )
    return tuple(episodes)


def select_order9_c3_promotion_buckets(
    manifest: Order9PiLRolloutBucketManifest,
) -> tuple[Order9PiLRolloutBucket, ...]:
    """Return the fixed 2--8 module validation matrix (two buckets each)."""

    buckets = tuple(
        sorted(
            (
                bucket
                for bucket in manifest.buckets
                if bucket.split == DatasetSplit.VALIDATION
            ),
            key=lambda bucket: (bucket.module_count, bucket.sample_index, bucket.bucket_id),
        )
    )
    counts = {
        module_count: sum(bucket.module_count == module_count for bucket in buckets)
        for module_count in range(2, 9)
    }
    if len(buckets) != 14 or counts != {module_count: 2 for module_count in range(2, 9)}:
        raise SchemaValidationError(
            "C3 promotion requires exactly two validation buckets for every "
            "module count from 2 through 8"
        )
    return buckets


def load_order9_evaluation_episode_jsonl(
    path: str | Path,
) -> tuple[Order9EvaluationEpisode, ...]:
    rows: list[Order9EvaluationEpisode] = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise SchemaValidationError(
                    f"{source}:{line_number} is not a JSON object"
                )
            rows.append(Order9EvaluationEpisode.from_dict(payload))
    if not rows:
        raise SchemaValidationError(f"evaluation JSONL is empty: {source}")
    return tuple(rows)


def validate_order9_c3_promotion_bucket_evidence(
    *,
    log_path: str | Path,
    raw_artifact_path: str | Path,
    episode_jsonl_path: str | Path,
    stage_id: str,
    generation_id: str,
    checkpoint_sha256: str,
    expected_episode_count: int,
) -> tuple[Order9EvaluationEpisode, ...]:
    """Validate one completed deterministic phase-zero Isaac evaluation."""

    if expected_episode_count < 1:
        raise ValueError("expected episode count must be positive")
    log = Path(log_path).resolve()
    raw = Path(raw_artifact_path).resolve()
    episode_jsonl = Path(episode_jsonl_path).resolve()
    if not log.is_file() or not raw.is_file() or not episode_jsonl.is_file():
        raise SchemaValidationError("C3 promotion bucket evidence is incomplete")
    recovery_path = _horizon_timeout_recovery_path(episode_jsonl)
    recovery: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    if recovery_path.is_file():
        payload = json.loads(recovery_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise SchemaValidationError("C3 timeout recovery manifest is invalid")
        recovery = payload
    else:
        result = load_order9_rollout_result(log)

    if result is not None:
        expected = {
            "passed": True,
            "stage_id": stage_id,
            "generation_id": generation_id,
            "split": DatasetSplit.VALIDATION.value,
            "evaluation_mode": True,
            "evaluation_episode_count": expected_episode_count,
            "deterministic_policy": True,
            "initial_phase_zero": True,
            "formal_phase_zero_start": True,
            "formal_phase_zero_state_source": (
                ORDER9_C3_FORMAL_PHASE_ZERO_STATE_SOURCE
            ),
            "initial_phase_elapsed_s": 0.0,
            "payload_feedforward_phase_contract": (
                ORDER9_PAYLOAD_FEEDFORWARD_PHASE_CONTRACT
            ),
            "contact_acquisition_success_contract": (
                ORDER9_CONTACT_ACQUISITION_SUCCESS_CONTRACT
            ),
            "wrench_range_hard_gate_enabled": False,
            "c3_privileged_phase_supervision_contract": (
                ORDER9_C3_PRIVILEGED_PHASE_SUPERVISION_CONTRACT
            ),
            "release_success_contract": ORDER9_RELEASE_SUCCESS_CONTRACT,
            "release_joint_position_tolerance_rad": (
                ORDER9_RELEASE_JOINT_POSITION_TOLERANCE_RAD
            ),
            "object_pose_success_contract": ORDER9_OBJECT_POSE_SUCCESS_CONTRACT,
            "object_position_tolerance_m": ORDER9_OBJECT_POSITION_TOLERANCE_M,
            "c3_boundary_tail_sampling_contract": (
                ORDER9_C3_BOUNDARY_TAIL_SAMPLING_CONTRACT
            ),
            "controller_side_contact_preload_version": None,
            "evaluation_horizon_timeout_contract": (
                ORDER9_EVALUATION_HORIZON_TIMEOUT_CONTRACT
            ),
        }
        for name, value in expected.items():
            if result.get(name) != value:
                raise SchemaValidationError(
                    f"C3 promotion rollout differs at {name}: "
                    f"{result.get(name)!r} != {value!r}"
                )
        if Path(str(result.get("raw_artifact_path", ""))).resolve() != raw:
            raise SchemaValidationError("C3 promotion raw artifact path differs")
        if Path(str(result.get("evaluation_jsonl", ""))).resolve() != episode_jsonl:
            raise SchemaValidationError("C3 promotion episode JSONL path differs")

    raw_sha256 = hash_file(raw)
    raw_artifact = load_order9_tensor_rollout_artifact(
        raw,
        expected_sha256=raw_sha256,
    )
    if (
        raw_artifact.metadata.get("collector_version")
        != ORDER9_PRODUCTION_COLLECTOR_VERSION
    ):
        raise SchemaValidationError(
            "C3 promotion raw artifact collector version differs"
        )
    if (
        raw_artifact.metadata.get("controller_side_contact_preload_version")
        is not None
    ):
        raise SchemaValidationError(
            "C3 promotion raw artifact contact preload version differs"
        )
    raw_expected = {
        "stage_id": stage_id,
        "generation_id": generation_id,
        "pi_l_checkpoint_sha256": checkpoint_sha256,
        "evaluation_mode": True,
        "deterministic_policy": True,
        "initial_phase_zero": True,
        "formal_phase_zero_start": True,
        "formal_phase_zero_state_source": (
            ORDER9_C3_FORMAL_PHASE_ZERO_STATE_SOURCE
        ),
        "initial_phase_elapsed_s": 0.0,
        "raw_contact_actor_input": False,
        "phase_supervision_source": "privileged_physx_contact_outcome",
        "c3_privileged_phase_supervision_contract": (
            ORDER9_C3_PRIVILEGED_PHASE_SUPERVISION_CONTRACT
        ),
        "wrench_range_gate_scale": 1.0,
        "wrench_range_hard_gate_enabled": False,
        "exact_wrench_reward_bounds_preserved": True,
    }
    for name, value in raw_expected.items():
        if raw_artifact.metadata.get(name) != value:
            raise SchemaValidationError(
                f"C3 promotion raw artifact differs at {name}"
            )
    if result is not None:
        if result.get("raw_artifact_sha256") != raw_sha256:
            raise SchemaValidationError("C3 promotion raw artifact hash differs")
        if (
            raw_artifact.metadata.get("evaluation_horizon_timeout_contract")
            != ORDER9_EVALUATION_HORIZON_TIMEOUT_CONTRACT
        ):
            raise SchemaValidationError(
                "C3 promotion raw timeout contract differs"
            )
    else:
        assert recovery is not None
        recovery_expected = {
            "recovery_version": ORDER9_C3_HORIZON_TIMEOUT_RECOVERY_VERSION,
            "evaluation_horizon_timeout_contract": (
                ORDER9_EVALUATION_HORIZON_TIMEOUT_CONTRACT
            ),
            "stage_id": stage_id,
            "generation_id": generation_id,
            "checkpoint_sha256": checkpoint_sha256,
            "raw_artifact_path": str(raw),
            "raw_artifact_sha256": raw_sha256,
            "source_log_path": str(log),
            "source_log_sha256": hash_file(log),
            "episode_jsonl_path": str(episode_jsonl),
            "episode_jsonl_sha256": hash_file(episode_jsonl),
            "episode_count": expected_episode_count,
        }
        for name, value in recovery_expected.items():
            if recovery.get(name) != value:
                raise SchemaValidationError(
                    f"C3 timeout recovery differs at {name}"
                )

    episodes = load_order9_evaluation_episode_jsonl(episode_jsonl)
    if len(episodes) != expected_episode_count:
        raise SchemaValidationError("C3 promotion episode count differs")
    expected_ids = {
        f"{generation_id}:evaluation:env:{index:04d}"
        for index in range(expected_episode_count)
    }
    if {episode.episode_id for episode in episodes} != expected_ids:
        raise SchemaValidationError("C3 promotion episode identities differ")
    for episode in episodes:
        metadata = episode.metadata
        if (
            episode.split != DatasetSplit.VALIDATION
            or not episode.isaac_backed
            or not episode.full_mesh_evaluation
            or Path(episode.source_artifact_path).resolve() != raw
            or episode.source_artifact_sha256 != raw_sha256
            or metadata.get("generation_id") != generation_id
            or metadata.get("deterministic_policy") is not True
            or metadata.get("initial_phase_index") != 0
            or metadata.get("initial_phase_stratum_index") != -1
            or metadata.get("formal_phase_zero_start") is not True
            or metadata.get("formal_phase_zero_state_source")
            != ORDER9_C3_FORMAL_PHASE_ZERO_STATE_SOURCE
            or metadata.get("initial_phase_elapsed_s") != 0.0
            or metadata.get("first_terminal_only") is not True
            or metadata.get("first_terminal_or_horizon_timeout") is not True
            or metadata.get("evaluation_horizon_timeout_contract")
            != ORDER9_EVALUATION_HORIZON_TIMEOUT_CONTRACT
            or metadata.get("raw_contact_actor_input") is not False
            or metadata.get("payload_feedforward_phase_contract")
            != ORDER9_PAYLOAD_FEEDFORWARD_PHASE_CONTRACT
            or metadata.get("release_success_contract")
            != ORDER9_RELEASE_SUCCESS_CONTRACT
            or metadata.get("contact_acquisition_success_contract")
            != ORDER9_CONTACT_ACQUISITION_SUCCESS_CONTRACT
            or metadata.get("wrench_range_hard_gate_enabled") is not False
            or metadata.get("c3_privileged_phase_supervision_contract")
            != ORDER9_C3_PRIVILEGED_PHASE_SUPERVISION_CONTRACT
            or metadata.get("release_joint_position_tolerance_rad")
            != ORDER9_RELEASE_JOINT_POSITION_TOLERANCE_RAD
            or metadata.get("object_pose_success_contract")
            != ORDER9_OBJECT_POSE_SUCCESS_CONTRACT
            or metadata.get("object_position_tolerance_m")
            != ORDER9_OBJECT_POSITION_TOLERANCE_M
            or metadata.get("c3_boundary_tail_sampling_contract")
            != ORDER9_C3_BOUNDARY_TAIL_SAMPLING_CONTRACT
            or metadata.get("controller_side_contact_preload_version")
            is not None
        ):
            raise SchemaValidationError(
                f"C3 promotion episode contract differs: {episode.episode_id}"
            )
    if recovery is not None and any(
        episode.task_success
        or episode.safety_failure
        or episode.failure_reason != "evaluation_horizon_timeout"
        or episode.metadata.get("evaluation_horizon_timeout") is not True
        or episode.metadata.get("recovered_from_complete_raw_rollout") is not True
        for episode in episodes
    ):
        raise SchemaValidationError("C3 timeout recovery episode differs")
    if checkpoint_sha256 not in log.read_text(encoding="utf-8", errors="replace"):
        raise SchemaValidationError("C3 promotion log lacks checkpoint binding")
    return episodes


def measured_order9_training_rollout_throughput(
    generation_roots: Iterable[str | Path],
    *,
    required_environment_count: int,
) -> tuple[int, float]:
    """Aggregate production-width tensorized rollout work and wall time.

    A C3 generation also contains deliberately narrow continuous-state shards
    (currently twelve environments) whose purpose is distribution coverage,
    not runtime benchmarking.  Mixing those shards into the production
    throughput metric makes the result depend on the curriculum mixture and
    can reject a runtime that independently exceeds the configured benchmark.

    Only collectors run at the benchmark environment count are therefore
    admitted.  Their full collection wall times are summed even when runs may
    have overlapped, so the resulting measurement remains conservative.
    """

    if required_environment_count < 1:
        raise ValueError("required environment count must be positive")

    environment_steps = 0
    wall_elapsed_s = 0.0
    seen_logs = 0
    for raw_root in generation_roots:
        root = Path(raw_root)
        logs = sorted((root / "logs").glob("*.log"))
        if not logs:
            raise SchemaValidationError(f"training generation has no logs: {root}")
        for log in logs:
            if ORDER9_ROLLOUT_RESULT_PREFIX not in log.read_text(
                encoding="utf-8", errors="replace"
            ):
                continue
            payload = load_order9_rollout_result(log)
            if int(payload.get("environment_count", 0)) != required_environment_count:
                continue
            steps = int(payload.get("environment_steps", 0))
            wall = float(payload.get("collection_wall_elapsed_s", 0.0))
            if steps < 1 or not math.isfinite(wall) or wall <= 0.0:
                raise SchemaValidationError(
                    f"training rollout measurement is invalid: {log}"
                )
            environment_steps += steps
            wall_elapsed_s += wall
            seen_logs += 1
    if seen_logs < 1:
        raise SchemaValidationError(
            "no production-width training rollout logs were found for "
            f"environment_count={required_environment_count}"
        )
    return environment_steps, wall_elapsed_s


def resolve_order9_training_generation_lineage(
    active_manifest_path: str | Path,
    *,
    through_update: int,
) -> tuple[Path, ...]:
    """Resolve the exact generation roots by following checkpoint provenance.

    C3 may branch into a new training-lineage directory after a diagnostic
    update.  Consequently, ``<current lineage>/generations/0..N`` is not a
    valid provenance model.  Every training manifest already binds both its
    own dataset and its parent checkpoint, so follow those immutable bindings
    instead and return the generation roots in update order.
    """

    if through_update < 0:
        raise ValueError("through_update must be non-negative")
    manifest_path = Path(active_manifest_path).resolve()
    generation_roots: list[Path] = []
    for expected_update in range(through_update, -1, -1):
        if not manifest_path.is_file():
            raise SchemaValidationError(
                f"training lineage manifest is missing: {manifest_path}"
            )
        manifest = Order9StageRunManifest.from_json(
            manifest_path.read_text(encoding="utf-8")
        )
        dataset_bindings = [
            item
            for item in manifest.input_artifacts
            if item.artifact_kind == "dataset_manifest"
        ]
        if len(dataset_bindings) != 1:
            raise SchemaValidationError(
                f"training manifest must bind one dataset: {manifest_path}"
            )
        dataset_path = Path(dataset_bindings[0].path).resolve()
        if not dataset_path.is_file() or hash_file(dataset_path) != dataset_bindings[0].sha256:
            raise SchemaValidationError(
                f"training dataset binding differs: {dataset_path}"
            )
        generation_root = dataset_path.parent.parent
        if generation_root.name != f"generation_{expected_update:06d}":
            raise SchemaValidationError(
                "training generation index differs: "
                f"{generation_root.name!r} != generation_{expected_update:06d!s}"
            )
        generation_roots.append(generation_root)
        if expected_update == 0:
            break

        parent_bindings = [
            item
            for item in manifest.input_artifacts
            if item.artifact_kind == "parent_checkpoint"
        ]
        if len(parent_bindings) != 1:
            raise SchemaValidationError(
                f"training manifest must bind one parent checkpoint: {manifest_path}"
            )
        parent_checkpoint = Path(parent_bindings[0].path).resolve()
        if (
            not parent_checkpoint.is_file()
            or hash_file(parent_checkpoint) != parent_bindings[0].sha256
        ):
            raise SchemaValidationError(
                f"parent checkpoint binding differs: {parent_checkpoint}"
            )
        manifest_path = _resolve_training_manifest_for_checkpoint(
            parent_checkpoint,
            expected_sha256=parent_bindings[0].sha256,
        )

    generation_roots.reverse()
    return tuple(generation_roots)


def _resolve_training_manifest_for_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str,
) -> Path:
    """Follow hash-bound checkpoint transforms to a PPO training manifest.

    A head-only teacher pass may sit between two PPO updates.  Such a pass has
    no rollout generation of its own, so it must not consume an update index.
    Its report binds the transformed checkpoint to its parent checkpoint.  We
    validate those bindings until reaching the regular PPO manifest that owns
    the parent checkpoint.
    """

    checkpoint = Path(checkpoint_path).resolve()
    checkpoint_sha256 = str(expected_sha256)
    seen: set[tuple[Path, str]] = set()
    while True:
        identity = (checkpoint, checkpoint_sha256)
        if identity in seen:
            raise SchemaValidationError(
                f"checkpoint transform lineage contains a cycle: {checkpoint}"
            )
        seen.add(identity)
        if not checkpoint.is_file() or hash_file(checkpoint) != checkpoint_sha256:
            raise SchemaValidationError(
                f"checkpoint transform binding differs: {checkpoint}"
            )

        manifest_path = checkpoint.parent / "stage_training_complete.json"
        if manifest_path.is_file():
            manifest = Order9StageRunManifest.from_json(
                manifest_path.read_text(encoding="utf-8")
            )
            outputs = [
                item
                for item in manifest.output_artifacts
                if item.artifact_kind == "policy_checkpoint"
                and Path(item.path).resolve() == checkpoint
                and item.sha256 == checkpoint_sha256
            ]
            if len(outputs) != 1:
                raise SchemaValidationError(
                    "training manifest does not own transformed parent "
                    f"checkpoint: {manifest_path}"
                )
            return manifest_path

        report_bindings: set[tuple[Path, str]] = set()
        for report_path in sorted(checkpoint.parent.glob("*report*.json")):
            try:
                payload = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, Mapping):
                continue
            output_raw = payload.get("output_checkpoint")
            output_sha256 = payload.get("output_checkpoint_sha256")
            parent_raw = payload.get("parent_checkpoint")
            parent_sha256 = payload.get("parent_checkpoint_sha256")
            if not all(
                isinstance(value, str)
                for value in (
                    output_raw,
                    output_sha256,
                    parent_raw,
                    parent_sha256,
                )
            ):
                continue
            output = Path(output_raw)
            if not output.is_absolute():
                output = report_path.parent / output
            if output.resolve() != checkpoint or output_sha256 != checkpoint_sha256:
                continue
            parent = Path(parent_raw)
            if not parent.is_absolute():
                parent = report_path.parent / parent
            parent = parent.resolve()
            if not parent.is_file() or hash_file(parent) != parent_sha256:
                raise SchemaValidationError(
                    f"checkpoint transform report parent differs: {report_path}"
                )
            report_bindings.add((parent, parent_sha256))

        if len(report_bindings) != 1:
            raise SchemaValidationError(
                "checkpoint has no unique hash-bound training or transform "
                f"provenance: {checkpoint}"
            )
        checkpoint, checkpoint_sha256 = report_bindings.pop()


def write_order9_c3_promotion_state(
    path: str | Path,
    payload: Mapping[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def log_contains_rollout_result(path: str | Path) -> bool:
    source = Path(path)
    if not source.is_file():
        return False
    return any(
        line.startswith(ORDER9_ROLLOUT_RESULT_PREFIX)
        for line in source.read_text(encoding="utf-8", errors="replace").splitlines()
    )
