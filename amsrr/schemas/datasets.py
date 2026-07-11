from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from amsrr.schemas.common import SchemaBase, SchemaValidationError, StrEnum, require_non_empty
from amsrr.schemas.contact_candidates import AssignmentFeasibilityResult, ContactCandidateSet
from amsrr.schemas.feasibility import FeasibilityResult
from amsrr.schemas.interaction_envelope import InteractionEnvelope
from amsrr.schemas.irg import InteractionRequirementGraph
from amsrr.schemas.morphology import DesignOutput, MorphologyGraph
from amsrr.schemas.policies import (
    ContactWrenchTrajectory,
    ControllerCommand,
    InteractionKnot,
    PolicyCommand,
)
from amsrr.schemas.runtime import RuntimeObservation


P4_3_DATASET_SCHEMA_VERSION = "p4_3_dataset_v1"


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    HELD_OUT = "held_out"


class DatasetKind(StrEnum):
    ISAAC_ROLLOUT = "isaac_rollout"
    LOW_LEVEL_CONTROL = "low_level_control"
    INTERACTION_TRAJECTORY = "interaction_trajectory"
    DESIGN_OUTCOME = "design_outcome"


@dataclass
class StageDecisionMasks(SchemaBase):
    """Credit-assignment masks required by Section 22.3."""

    design_decision_mask: bool = False
    high_level_decision_mask: bool = False
    low_level_control_mask: bool = False
    assembly_execution_mask: bool = False


@dataclass
class LowLevelControlRecord(SchemaBase):
    """One index-aligned observation-to-actuator control step.

    Reward fields are nullable because legacy rollout archives may contain only a
    terminal reward. Dataset builders must not broadcast that terminal value over
    otherwise unrelated control steps.
    """

    record_id: str
    episode_id: str
    task_id: str
    split: DatasetSplit
    step_index: int
    time_s: float
    trajectory_record_id: str
    active_trajectory_index: int
    active_knot_index: int
    runtime_observation: RuntimeObservation
    active_knot: InteractionKnot
    policy_command: PolicyCommand
    controller_command: ControllerCommand
    actuator_target_record: dict[str, Any]
    reward_terms: dict[str, float] | None
    reward: float | None
    terminal: bool
    stage_masks: StageDecisionMasks

    def validate(self) -> None:
        _require_record_identity(self.record_id, self.episode_id, self.task_id, type(self).__name__)
        require_non_empty(self.trajectory_record_id, "LowLevelControlRecord.trajectory_record_id")
        _require_non_negative_int(self.step_index, "LowLevelControlRecord.step_index")
        _require_non_negative_int(
            self.active_trajectory_index,
            "LowLevelControlRecord.active_trajectory_index",
        )
        _require_non_negative_int(self.active_knot_index, "LowLevelControlRecord.active_knot_index")
        _require_finite_non_negative(self.time_s, "LowLevelControlRecord.time_s")
        if not math.isclose(self.time_s, self.runtime_observation.time_s, rel_tol=0.0, abs_tol=1.0e-9):
            raise SchemaValidationError(
                "LowLevelControlRecord.time_s must match runtime_observation.time_s"
            )
        if not self.stage_masks.low_level_control_mask:
            raise SchemaValidationError(
                "LowLevelControlRecord.stage_masks.low_level_control_mask must be true"
            )
        if (self.reward_terms is None) != (self.reward is None):
            raise SchemaValidationError(
                "LowLevelControlRecord.reward_terms and reward must both be present or both be null"
            )
        if self.reward_terms is not None:
            _require_finite_mapping(self.reward_terms, "LowLevelControlRecord.reward_terms")
            _require_finite(self.reward, "LowLevelControlRecord.reward")


@dataclass
class InteractionTrajectoryRecord(SchemaBase):
    """A self-contained high-level decision and its rollout return."""

    record_id: str
    episode_id: str
    task_id: str
    split: DatasetSplit
    decision_index: int
    decision_time_s: float
    irg: InteractionRequirementGraph
    interaction_envelope: InteractionEnvelope
    morphology_graph: MorphologyGraph
    contact_candidate_set: ContactCandidateSet
    runtime_observation: RuntimeObservation
    trajectory: ContactWrenchTrajectory
    selected_candidate_ids: list[int]
    assignment_feasibility_results: list[AssignmentFeasibilityResult]
    decision_return: float
    stage_masks: StageDecisionMasks

    def validate(self) -> None:
        _require_record_identity(self.record_id, self.episode_id, self.task_id, type(self).__name__)
        _require_non_negative_int(self.decision_index, "InteractionTrajectoryRecord.decision_index")
        _require_finite_non_negative(
            self.decision_time_s,
            "InteractionTrajectoryRecord.decision_time_s",
        )
        _require_finite(self.decision_return, "InteractionTrajectoryRecord.decision_return")
        if not math.isclose(
            self.decision_time_s,
            self.runtime_observation.time_s,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise SchemaValidationError(
                "InteractionTrajectoryRecord.decision_time_s must match runtime_observation.time_s"
            )
        if not self.stage_masks.high_level_decision_mask:
            raise SchemaValidationError(
                "InteractionTrajectoryRecord.stage_masks.high_level_decision_mask must be true"
            )
        task_refs = {
            "irg": self.irg.task_id,
            "interaction_envelope": self.interaction_envelope.task_id,
            "contact_candidate_set": self.contact_candidate_set.task_id,
        }
        mismatches = sorted(name for name, task_id in task_refs.items() if task_id != self.task_id)
        if mismatches:
            raise SchemaValidationError(
                "InteractionTrajectoryRecord.task_id must match " + ", ".join(mismatches)
            )
        graph_id = self.morphology_graph.graph_id
        if self.contact_candidate_set.morphology_graph_id != graph_id:
            raise SchemaValidationError(
                "InteractionTrajectoryRecord.contact_candidate_set must reference morphology_graph"
            )
        if self.runtime_observation.morphology_graph.graph_id != graph_id:
            raise SchemaValidationError(
                "InteractionTrajectoryRecord.runtime_observation must reference morphology_graph"
            )
        if not self.trajectory.knots:
            raise SchemaValidationError("InteractionTrajectoryRecord.trajectory.knots must be non-empty")

        selected = self.selected_candidate_ids
        if any(candidate_id < 0 for candidate_id in selected):
            raise SchemaValidationError(
                "InteractionTrajectoryRecord.selected_candidate_ids must be non-negative"
            )
        if len(selected) != len(set(selected)):
            raise SchemaValidationError(
                "InteractionTrajectoryRecord.selected_candidate_ids must be unique"
            )
        available = {candidate.candidate_id for candidate in self.contact_candidate_set.candidates}
        unknown = sorted(set(selected) - available)
        if unknown:
            raise SchemaValidationError(
                f"InteractionTrajectoryRecord.selected_candidate_ids are unknown: {unknown}"
            )
        trajectory_ids = {
            assignment.candidate_id
            for knot in self.trajectory.knots
            for assignment in knot.contact_assignments
        }
        if not trajectory_ids.issubset(set(selected)):
            missing = sorted(trajectory_ids - set(selected))
            raise SchemaValidationError(
                "InteractionTrajectoryRecord.selected_candidate_ids omit trajectory assignments: "
                f"{missing}"
            )
        for result in self.assignment_feasibility_results:
            unknown_result_ids = sorted(set(result.candidate_ids) - available)
            if unknown_result_ids:
                raise SchemaValidationError(
                    "InteractionTrajectoryRecord.assignment_feasibility_results reference unknown "
                    f"candidate ids: {unknown_result_ids}"
                )


@dataclass
class DesignOutcomeRecord(SchemaBase):
    """One P2 design candidate paired with deterministic feasibility and outcome labels."""

    record_id: str
    episode_id: str | None
    task_id: str
    split: DatasetSplit
    candidate_id: int
    selected_for_rollout: bool
    design_output: DesignOutput
    feasibility_result: FeasibilityResult
    rollout_executed: bool
    task_success: bool | None
    object_dropped: bool | None
    hard_collision: bool | None
    controller_infeasible_terminal: bool | None
    episode_return: float | None
    rollout_metrics: dict[str, float]
    failure_reason: str | None
    stage_masks: StageDecisionMasks

    def validate(self) -> None:
        require_non_empty(self.record_id, "DesignOutcomeRecord.record_id")
        require_non_empty(self.task_id, "DesignOutcomeRecord.task_id")
        if self.episode_id is not None:
            require_non_empty(self.episode_id, "DesignOutcomeRecord.episode_id")
        _require_non_negative_int(self.candidate_id, "DesignOutcomeRecord.candidate_id")
        if self.design_output.task_id != self.task_id:
            raise SchemaValidationError(
                "DesignOutcomeRecord.task_id must match design_output.task_id"
            )
        if not self.stage_masks.design_decision_mask:
            raise SchemaValidationError(
                "DesignOutcomeRecord.stage_masks.design_decision_mask must be true"
            )
        if self.rollout_executed and not self.selected_for_rollout:
            raise SchemaValidationError(
                "DesignOutcomeRecord.rollout_executed requires selected_for_rollout"
            )
        if self.rollout_executed and not self.feasibility_result.feasible:
            raise SchemaValidationError(
                "DesignOutcomeRecord cannot execute a deterministically infeasible design"
            )

        outcome_values = (
            self.task_success,
            self.object_dropped,
            self.hard_collision,
            self.controller_infeasible_terminal,
            self.episode_return,
        )
        if self.rollout_executed:
            if self.episode_id is None:
                raise SchemaValidationError(
                    "DesignOutcomeRecord.rollout_executed requires episode_id"
                )
            if any(value is None for value in outcome_values):
                raise SchemaValidationError(
                    "DesignOutcomeRecord.rollout_executed requires all rollout outcome labels"
                )
        elif any(value is not None for value in outcome_values):
            raise SchemaValidationError(
                "DesignOutcomeRecord without rollout execution cannot contain rollout outcome labels"
            )
        if self.episode_return is not None:
            _require_finite(self.episode_return, "DesignOutcomeRecord.episode_return")
        _require_finite_mapping(self.rollout_metrics, "DesignOutcomeRecord.rollout_metrics")


@dataclass
class DatasetShard(SchemaBase):
    dataset_kind: DatasetKind
    split: DatasetSplit | None
    path: str
    record_count: int
    sha256: str

    def validate(self) -> None:
        require_non_empty(self.path, "DatasetShard.path")
        require_non_empty(self.sha256, "DatasetShard.sha256")
        _require_non_negative_int(self.record_count, "DatasetShard.record_count")


@dataclass
class P4_3DatasetManifest(SchemaBase):
    """Task-disjoint split and provenance contract for P4.3a-d artifacts."""

    dataset_id: str
    schema_version: Literal["p4_3_dataset_v1"]
    source_archive_paths: list[str]
    source_episode_ids: list[str]
    train_task_ids: list[str]
    validation_task_ids: list[str]
    held_out_task_ids: list[str]
    shards: list[DatasetShard]
    record_counts: dict[str, int]
    source_hash: str
    config_hash: str
    robot_model_hash: str
    urdf_hash: str
    thrust_model_hash: str
    task_hashes: dict[str, str]
    geometry_hashes: dict[str, str]
    random_seeds: list[int]
    simulator_version: str
    simulator_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        require_non_empty(self.dataset_id, "P4_3DatasetManifest.dataset_id")
        for path in self.source_archive_paths:
            require_non_empty(path, "P4_3DatasetManifest.source_archive_paths[]")
        for episode_id in self.source_episode_ids:
            require_non_empty(episode_id, "P4_3DatasetManifest.source_episode_ids[]")
        if len(self.source_episode_ids) != len(set(self.source_episode_ids)):
            raise SchemaValidationError(
                "P4_3DatasetManifest.source_episode_ids must be unique"
            )

        split_tasks = {
            DatasetSplit.TRAIN: self.train_task_ids,
            DatasetSplit.VALIDATION: self.validation_task_ids,
            DatasetSplit.HELD_OUT: self.held_out_task_ids,
        }
        for split, task_ids in split_tasks.items():
            for task_id in task_ids:
                require_non_empty(task_id, f"P4_3DatasetManifest.{split.value}_task_ids[]")
            if len(task_ids) != len(set(task_ids)):
                raise SchemaValidationError(
                    f"P4_3DatasetManifest.{split.value}_task_ids must be unique"
                )
        split_names = list(split_tasks)
        for index, left in enumerate(split_names):
            for right in split_names[index + 1 :]:
                overlap = sorted(set(split_tasks[left]).intersection(split_tasks[right]))
                if overlap:
                    raise SchemaValidationError(
                        "P4_3DatasetManifest task splits must be disjoint; "
                        f"{left.value}/{right.value} overlap: {overlap}"
                    )

        all_task_ids = set().union(*(set(task_ids) for task_ids in split_tasks.values()))
        missing_task_hashes = sorted(all_task_ids - set(self.task_hashes))
        if missing_task_hashes:
            raise SchemaValidationError(
                "P4_3DatasetManifest.task_hashes missing split tasks: "
                f"{missing_task_hashes}"
            )
        _require_non_empty_mapping(self.task_hashes, "P4_3DatasetManifest.task_hashes")
        _require_non_empty_mapping(self.geometry_hashes, "P4_3DatasetManifest.geometry_hashes")

        allowed_kinds = set(DatasetKind.values())
        unknown_count_kinds = sorted(set(self.record_counts) - allowed_kinds)
        if unknown_count_kinds:
            raise SchemaValidationError(
                f"P4_3DatasetManifest.record_counts has unknown dataset kinds: {unknown_count_kinds}"
            )
        for kind, count in self.record_counts.items():
            _require_non_negative_int(count, f"P4_3DatasetManifest.record_counts[{kind!r}]")
        shard_paths = [shard.path for shard in self.shards]
        if len(shard_paths) != len(set(shard_paths)):
            raise SchemaValidationError("P4_3DatasetManifest.shards paths must be unique")
        for kind in DatasetKind:
            shard_count = sum(
                shard.record_count for shard in self.shards if shard.dataset_kind == kind
            )
            declared_count = self.record_counts.get(kind.value)
            if declared_count is not None and shard_count != declared_count:
                raise SchemaValidationError(
                    "P4_3DatasetManifest.record_counts must match shard counts for "
                    f"{kind.value}: {declared_count} != {shard_count}"
                )

        for name in (
            "source_hash",
            "config_hash",
            "robot_model_hash",
            "urdf_hash",
            "thrust_model_hash",
            "simulator_version",
            "simulator_hash",
        ):
            require_non_empty(getattr(self, name), f"P4_3DatasetManifest.{name}")
        if not self.random_seeds:
            raise SchemaValidationError("P4_3DatasetManifest.random_seeds must be non-empty")
        if any(seed < 0 for seed in self.random_seeds):
            raise SchemaValidationError(
                "P4_3DatasetManifest.random_seeds must be non-negative"
            )
        if len(self.random_seeds) != len(set(self.random_seeds)):
            raise SchemaValidationError("P4_3DatasetManifest.random_seeds must be unique")


def _require_record_identity(record_id: str, episode_id: str, task_id: str, schema_name: str) -> None:
    require_non_empty(record_id, f"{schema_name}.record_id")
    require_non_empty(episode_id, f"{schema_name}.episode_id")
    require_non_empty(task_id, f"{schema_name}.task_id")


def _require_non_negative_int(value: int, path: str) -> None:
    if value < 0:
        raise SchemaValidationError(f"{path} must be non-negative")


def _require_finite(value: float | None, path: str) -> None:
    if value is None or not math.isfinite(value):
        raise SchemaValidationError(f"{path} must be finite")


def _require_finite_non_negative(value: float, path: str) -> None:
    _require_finite(value, path)
    if value < 0.0:
        raise SchemaValidationError(f"{path} must be non-negative")


def _require_finite_mapping(values: dict[str, float], path: str) -> None:
    for key, value in values.items():
        require_non_empty(key, f"{path}.key")
        _require_finite(value, f"{path}[{key!r}]")


def _require_non_empty_mapping(values: dict[str, str], path: str) -> None:
    if not values:
        raise SchemaValidationError(f"{path} must be non-empty")
    for key, value in values.items():
        require_non_empty(key, f"{path}.key")
        require_non_empty(value, f"{path}[{key!r}]")
