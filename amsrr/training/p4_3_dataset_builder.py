from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from amsrr.logging.episode_archive import EpisodeArchive, read_episode_archives_jsonl
from amsrr.schemas.contact_candidates import ContactCandidateSet
from amsrr.schemas.datasets import (
    P4_3_DATASET_SCHEMA_VERSION,
    DatasetKind,
    DatasetShard,
    DatasetSplit,
    DesignOutcomeRecord,
    InteractionTrajectoryRecord,
    LowLevelControlRecord,
    P4_3DatasetManifest,
    P4_3_DATASET_KINDS,
    StageDecisionMasks,
    TrajectoryProvenance,
    TrajectorySourceKind,
)
from amsrr.schemas.policies import ContactWrenchTrajectory, InteractionKnot
from amsrr.training.p4_3_reward import P4_3RewardConfig, compute_p4_3_archive_rewards
from amsrr.utils.hashing import stable_hash


@dataclass(frozen=True)
class P4_3DatasetBuildResult:
    manifest: P4_3DatasetManifest
    manifest_path: str
    low_level_records: list[LowLevelControlRecord]
    trajectory_records: list[InteractionTrajectoryRecord]
    design_outcome_records: list[DesignOutcomeRecord]


def build_p4_3_dataset(
    *,
    archive_paths: Iterable[str | Path],
    output_dir: str | Path,
    reward_config: P4_3RewardConfig | None = None,
    low_level_stride: int = 4,
    split_fractions: dict[str, float] | None = None,
) -> P4_3DatasetBuildResult:
    paths = [Path(path) for path in archive_paths]
    if not paths:
        raise ValueError("P4.3 dataset builder requires at least one archive path")
    if low_level_stride < 1:
        raise ValueError("P4.3 low_level_stride must be positive")
    archives = [archive for path in paths for archive in read_episode_archives_jsonl(path)]
    if not archives:
        raise ValueError("P4.3 source archives contain no episodes")
    episode_ids = [archive.episode_id for archive in archives]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("P4.3 source episode ids must be unique")

    fractions = split_fractions or {
        "train": 2.0 / 3.0,
        "validation": 1.0 / 6.0,
        "held_out": 1.0 / 6.0,
    }
    if set(fractions) != {"train", "validation", "held_out"}:
        raise ValueError("P4.3 split_fractions must define train, validation, held_out")
    if any(value <= 0.0 for value in fractions.values()) or abs(sum(fractions.values()) - 1.0) > 1.0e-6:
        raise ValueError("P4.3 split_fractions must be positive and sum to one")
    split_by_task = _task_splits(
        (archive.task_spec.task_id for archive in archives),
        fractions=fractions,
    )
    low_level: list[LowLevelControlRecord] = []
    trajectories: list[InteractionTrajectoryRecord] = []
    design_outcomes: list[DesignOutcomeRecord] = []
    rewards_by_episode: dict[str, list[dict[str, float]]] = {}
    for archive in archives:
        split = split_by_task[archive.task_spec.task_id]
        rewards = compute_p4_3_archive_rewards(archive, config=reward_config)
        rewards_by_episode[archive.episode_id] = rewards
        low_level.extend(_low_level_records(archive, split, rewards, stride=low_level_stride))
        trajectories.append(_trajectory_record(archive, split, rewards))
        design_outcomes.append(_design_outcome_record(archive, split, rewards))

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    shards: list[DatasetShard] = []
    for split in DatasetSplit:
        split_archives = [a for a in archives if split_by_task[a.task_spec.task_id] == split]
        split_low = [record for record in low_level if record.split == split]
        split_high = [record for record in trajectories if record.split == split]
        split_design = [record for record in design_outcomes if record.split == split]
        shards.extend(
            _write_split_shards(
                target_dir,
                split,
                split_archives,
                split_low,
                split_high,
                split_design,
            )
        )

    manifest = _manifest(
        paths,
        archives,
        split_by_task,
        shards,
        low_level_stride=low_level_stride,
        reward_config=reward_config or P4_3RewardConfig(),
        split_fractions=fractions,
    )
    manifest_path = target_dir / "manifest.json"
    manifest_path.write_text(manifest.to_json(indent=2) + "\n", encoding="utf-8")
    _write_json(target_dir / "train_ids.json", manifest.train_task_ids)
    _write_json(target_dir / "validation_ids.json", manifest.validation_task_ids)
    _write_json(target_dir / "held_out_ids.json", manifest.held_out_task_ids)
    return P4_3DatasetBuildResult(
        manifest=manifest,
        manifest_path=str(manifest_path),
        low_level_records=low_level,
        trajectory_records=trajectories,
        design_outcome_records=design_outcomes,
    )


def _task_splits(
    task_ids: Iterable[str],
    *,
    fractions: dict[str, float],
) -> dict[str, DatasetSplit]:
    ordered = sorted(set(task_ids), key=lambda value: (stable_hash(value), value))
    if not ordered:
        raise ValueError("P4.3 task split requires task ids")
    mapping = {task_id: DatasetSplit.TRAIN for task_id in ordered}
    if len(ordered) >= 3:
        validation_count = max(1, round(len(ordered) * fractions["validation"]))
        held_out_count = max(1, round(len(ordered) * fractions["held_out"]))
        if validation_count + held_out_count >= len(ordered):
            held_out_count = 1
            validation_count = 1
        for task_id in ordered[-held_out_count:]:
            mapping[task_id] = DatasetSplit.HELD_OUT
        for task_id in ordered[-(held_out_count + validation_count) : -held_out_count]:
            mapping[task_id] = DatasetSplit.VALIDATION
    elif len(ordered) == 2:
        mapping[ordered[-1]] = DatasetSplit.HELD_OUT
    return mapping


def _low_level_records(
    archive: EpisodeArchive,
    split: DatasetSplit,
    rewards: list[dict[str, float]],
    *,
    stride: int,
) -> list[LowLevelControlRecord]:
    lengths = {
        len(archive.runtime_observations),
        len(archive.policy_commands),
        len(archive.controller_commands),
        len(archive.actuator_target_records),
    }
    if len(lengths) != 1:
        raise ValueError(
            f"P4.3 per-step archive fields must align for {archive.episode_id}: {sorted(lengths)}"
        )
    count = next(iter(lengths))
    if count == 0 or len(rewards) != count:
        raise ValueError(f"P4.3 archive {archive.episode_id} has no aligned reward/control steps")
    trajectory_index, trajectory = _primary_trajectory(archive)
    output: list[LowLevelControlRecord] = []
    # The probe logs observation[i] before command[i].  The terminal transition
    # reward therefore belongs to command N-2 (obs[N-2] -> obs[N-1]), while the
    # final command row N-1 has command-only reward because no post-observation
    # is logged.  Preserve both rows even when temporal striding would skip the
    # causal terminal transition.
    required_tail_indices = {count - 1}
    if count >= 2:
        required_tail_indices.add(count - 2)
    indices = sorted(set(range(0, count, stride)) | required_tail_indices)
    previous_index = -1
    for index in indices:
        observation = archive.runtime_observations[index]
        knot_index, knot = _active_knot(
            trajectory,
            observation.time_s,
            phase_label=observation.task_progress.phase_label,
        )
        reward_record = dict(rewards[index])
        interval_reward = float(
            sum(item["reward"] for item in rewards[previous_index + 1 : index + 1])
        )
        reward_record["sampled_step_reward"] = float(rewards[index]["reward"])
        reward_record["interval_aggregated_reward"] = interval_reward
        reward_record["interval_start_step"] = float(previous_index + 1)
        reward_record["interval_end_step"] = float(index)
        output.append(
            LowLevelControlRecord(
                record_id=f"{archive.episode_id}:low:{index:06d}",
                episode_id=archive.episode_id,
                task_id=archive.task_spec.task_id,
                split=split,
                step_index=index,
                time_s=observation.time_s,
                trajectory_record_id=f"{archive.episode_id}:trajectory:{trajectory_index}",
                active_trajectory_index=trajectory_index,
                active_knot_index=knot_index,
                runtime_observation=observation,
                active_knot=knot,
                policy_command=archive.policy_commands[index],
                controller_command=archive.controller_commands[index],
                actuator_target_record=archive.actuator_target_records[index],
                reward_terms=reward_record,
                reward=interval_reward,
                terminal=reward_record.get("terminal_reward_data_available", 0.0) > 0.5,
                stage_masks=StageDecisionMasks(low_level_control_mask=True),
            )
        )
        previous_index = index
    return output


def _trajectory_record(
    archive: EpisodeArchive,
    split: DatasetSplit,
    rewards: list[dict[str, float]],
) -> InteractionTrajectoryRecord:
    _, trajectory = _primary_trajectory(archive)
    if not archive.runtime_observations:
        raise ValueError("P4.3 trajectory record requires a runtime observation")
    raw_candidates = archive.rollout_artifacts.get("contact_candidate_set")
    if not isinstance(raw_candidates, dict):
        raise ValueError("P4.3 archive is missing contact_candidate_set")
    candidate_set = ContactCandidateSet.from_dict(raw_candidates)
    selected_ids = sorted(
        {
            assignment.candidate_id
            for knot in trajectory.knots
            for assignment in knot.contact_assignments
        }
    )
    feasibility = list(candidate_set.assignment_feasibility_cache.values())
    return InteractionTrajectoryRecord(
        record_id=f"{archive.episode_id}:trajectory:0",
        episode_id=archive.episode_id,
        task_id=archive.task_spec.task_id,
        split=split,
        decision_index=0,
        decision_time_s=archive.runtime_observations[0].time_s,
        irg=archive.irg,
        interaction_envelope=archive.interaction_envelope,
        morphology_graph=archive.runtime_observations[0].morphology_graph,
        contact_candidate_set=candidate_set,
        runtime_observation=archive.runtime_observations[0],
        trajectory=trajectory,
        selected_candidate_ids=selected_ids,
        assignment_feasibility_results=feasibility,
        decision_return=float(sum(record["reward"] for record in rewards)),
        stage_masks=StageDecisionMasks(high_level_decision_mask=True),
        trajectory_provenance=TrajectoryProvenance(
            source_kind=TrajectorySourceKind.IMPORTED_LEGACY,
            source_version=str(
                archive.rollout_artifacts.get(
                    "high_level_policy_version",
                    trajectory.derived_mode_label or "legacy_archive_unknown",
                )
            ),
            metadata={
                "source_archive_episode_id": archive.episode_id,
                "trajectory_feasibility_not_reconstructed": True,
            },
        ),
        trajectory_feasibility_result=None,
    )


def _design_outcome_record(
    archive: EpisodeArchive,
    split: DatasetSplit,
    rewards: list[dict[str, float]],
) -> DesignOutcomeRecord:
    if archive.design_output is None or archive.feasibility_result is None:
        raise ValueError("P4.3 design outcome requires DesignOutput and FeasibilityResult")
    candidate_id = int(archive.rollout_artifacts.get("p4_3_candidate_id", -1))
    if candidate_id < 0:
        candidate_id = int(archive.design_output.design_scores.get("p2_design_policy_candidate_id", 0.0))
    metrics = {key: float(value) for key, value in archive.metrics.items()}
    metrics["p4_2_bounded_carry_success"] = 1.0 if archive.success else 0.0
    terminal_task_success = any(
        record.get("terminal_reward_data_available", 0.0) > 0.5
        and record.get("terminal_success", 0.0) > 0.5
        for record in rewards
    )
    return DesignOutcomeRecord(
        record_id=f"{archive.episode_id}:design:{candidate_id}",
        episode_id=archive.episode_id,
        task_id=archive.task_spec.task_id,
        split=split,
        candidate_id=candidate_id,
        selected_for_rollout=True,
        design_output=archive.design_output,
        feasibility_result=archive.feasibility_result,
        rollout_executed=True,
        task_success=terminal_task_success,
        object_dropped=_metric_bool(metrics, "object_drop"),
        hard_collision=_metric_bool(metrics, "hard_collision"),
        controller_infeasible_terminal=_metric_bool(metrics, "controller_qp_infeasible_terminal"),
        episode_return=float(sum(record["reward"] for record in rewards)),
        rollout_metrics=metrics,
        failure_reason=archive.failure_reason,
        stage_masks=StageDecisionMasks(design_decision_mask=True),
    )


def _primary_trajectory(archive: EpisodeArchive) -> tuple[int, ContactWrenchTrajectory]:
    if not archive.trajectory_records:
        raise ValueError(f"P4.3 archive {archive.episode_id} has no trajectory")
    return 0, archive.trajectory_records[0]


def _active_knot(
    trajectory: ContactWrenchTrajectory,
    time_s: float,
    *,
    phase_label: str | None,
) -> tuple[int, InteractionKnot]:
    ordered = sorted(enumerate(trajectory.knots), key=lambda item: item[1].t_rel_s)
    if phase_label:
        for index, knot in ordered:
            if any(
                guard.get("type") == "p4_2_phase" and guard.get("phase") == phase_label
                for guard in knot.guard_conditions
            ):
                return index, knot
    active_index, active = ordered[0]
    relative_time = time_s % max(trajectory.horizon_s, trajectory.dt_s)
    for index, knot in ordered:
        if knot.t_rel_s <= relative_time:
            active_index, active = index, knot
        else:
            break
    return active_index, active


def _write_split_shards(
    output_dir: Path,
    split: DatasetSplit,
    archives: list[EpisodeArchive],
    low_level: list[LowLevelControlRecord],
    trajectories: list[InteractionTrajectoryRecord],
    design_outcomes: list[DesignOutcomeRecord],
) -> list[DatasetShard]:
    values: list[tuple[DatasetKind, list[Any]]] = [
        (DatasetKind.ISAAC_ROLLOUT, archives),
        (DatasetKind.LOW_LEVEL_CONTROL, low_level),
        (DatasetKind.INTERACTION_TRAJECTORY, trajectories),
        (DatasetKind.DESIGN_OUTCOME, design_outcomes),
    ]
    shards: list[DatasetShard] = []
    for kind, records in values:
        path = output_dir / f"{kind.value}_{split.value}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(record.to_json())
                handle.write("\n")
        shards.append(
            DatasetShard(
                dataset_kind=kind,
                split=split,
                path=str(path),
                record_count=len(records),
                sha256=_file_sha256(path),
            )
        )
    return shards


def _manifest(
    paths: list[Path],
    archives: list[EpisodeArchive],
    split_by_task: dict[str, DatasetSplit],
    shards: list[DatasetShard],
    *,
    low_level_stride: int,
    reward_config: P4_3RewardConfig,
    split_fractions: dict[str, float],
) -> P4_3DatasetManifest:
    task_hashes = {archive.task_spec.task_id: archive.task_hash for archive in archives}
    geometry_hashes = {
        f"{archive.task_spec.task_id}:{geometry_id}": value
        for archive in archives
        for geometry_id, value in archive.geometry_hashes.items()
    }
    seeds = sorted(
        {
            int(archive.reproducibility.get("random_seed", 0))
            for archive in archives
        }
    )
    backend_names = sorted(
        {str(archive.rollout_artifacts.get("backend", "unknown")) for archive in archives}
    )
    record_counts = {
        kind.value: sum(shard.record_count for shard in shards if shard.dataset_kind == kind)
        for kind in P4_3_DATASET_KINDS
    }
    split_tasks = {
        split: sorted(task_id for task_id, value in split_by_task.items() if value == split)
        for split in DatasetSplit
    }
    seed_data = {
        "episodes": [archive.episode_id for archive in archives],
        "tasks": task_hashes,
        "shards": [shard.sha256 for shard in shards],
    }
    return P4_3DatasetManifest(
        dataset_id=f"p4-3-{stable_hash(seed_data)[:16]}",
        schema_version=P4_3_DATASET_SCHEMA_VERSION,
        source_archive_paths=[str(path) for path in paths],
        source_episode_ids=[archive.episode_id for archive in archives],
        train_task_ids=split_tasks[DatasetSplit.TRAIN],
        validation_task_ids=split_tasks[DatasetSplit.VALIDATION],
        held_out_task_ids=split_tasks[DatasetSplit.HELD_OUT],
        shards=shards,
        record_counts=record_counts,
        source_hash=stable_hash([str(path) for path in paths]),
        config_hash=stable_hash(
            {
                "source_config_hashes": sorted({archive.config_hash for archive in archives}),
                "reward_config": reward_config,
                "low_level_stride": low_level_stride,
                "split_fractions": split_fractions,
            }
        ),
        robot_model_hash=stable_hash(sorted({archive.robot_model_hash for archive in archives})),
        urdf_hash=_combined_reproducibility_hash(archives, "urdf_hash"),
        thrust_model_hash=_combined_reproducibility_hash(archives, "thrust_model_hash"),
        task_hashes=task_hashes,
        geometry_hashes=geometry_hashes,
        random_seeds=seeds,
        simulator_version="+".join(backend_names),
        simulator_hash=stable_hash(backend_names),
        metadata={
            "phase": "P4.3a",
            "task_disjoint_splits": True,
            "low_level_stride": low_level_stride,
            "low_level_effective_rate_hz": 200.0 / float(low_level_stride),
            "reward_config_hash": stable_hash(reward_config),
            "reward_alignment": "command_i_with_obs_i_to_obs_i_plus_1",
            "final_command_reward": "command_only_post_observation_unavailable",
            "terminal_reward_alignment": "command_n_minus_2_transition_to_obs_n_minus_1",
            "split_fractions": split_fractions,
            "isaac_backed_episode_count": sum(
                1
                for archive in archives
                if archive.metrics.get("isaac_backed", 0.0) > 0.5
                or archive.rollout_artifacts.get("backend") == "isaac_lab"
            ),
            "source_episode_count": len(archives),
            "contact_models": sorted(
                {str(archive.rollout_artifacts.get("contact_model", "unknown")) for archive in archives}
            ),
            "natural_contact_success_claim": False,
            "p4_full_completion_claim": False,
        },
    )


def _combined_reproducibility_hash(archives: list[EpisodeArchive], key: str) -> str:
    values = sorted({str(archive.reproducibility.get(key, "")) for archive in archives})
    values = [value for value in values if value]
    return values[0] if len(values) == 1 else stable_hash(values or ["missing"])


def _metric_bool(metrics: dict[str, float], key: str) -> bool:
    return bool(metrics.get(key, 0.0) > 0.5)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
