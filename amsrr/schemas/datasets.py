from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from amsrr.schemas.common import SchemaBase, SchemaValidationError, StrEnum, require_non_empty
from amsrr.schemas.contact_candidates import AssignmentFeasibilityResult, ContactCandidateSet
from amsrr.schemas.feasibility import FeasibilityResult, TrajectoryFeasibilityResult
from amsrr.schemas.interaction_envelope import InteractionEnvelope
from amsrr.schemas.irg import InteractionRequirementGraph
from amsrr.schemas.morphology import DesignAction, DesignOutput, MorphologyGraph
from amsrr.schemas.policies import (
    ContactWrenchTrajectory,
    ControllerCommand,
    InteractionKnot,
    PolicyCommand,
)
from amsrr.schemas.runtime import RuntimeObservation
from amsrr.schemas.task_spec import TaskSpec


P4_3_DATASET_SCHEMA_LEGACY_VERSION = "p4_3_dataset_v1"
P4_3_DATASET_SCHEMA_VERSION = "p4_3_dataset_v2"


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    HELD_OUT = "held_out"


class DatasetKind(StrEnum):
    ISAAC_ROLLOUT = "isaac_rollout"
    LOW_LEVEL_CONTROL = "low_level_control"
    INTERACTION_TRAJECTORY = "interaction_trajectory"
    DESIGN_OUTCOME = "design_outcome"
    DESIGN_ACTION_TRAJECTORY = "design_action_trajectory"


# P4.3 v1/v2 predates the sequential design-action records introduced for
# Order 9.  Keep its required shard set explicit so extending DatasetKind does
# not retroactively change the acceptance contract of archived P4.3 data.
P4_3_DATASET_KINDS = (
    DatasetKind.ISAAC_ROLLOUT,
    DatasetKind.LOW_LEVEL_CONTROL,
    DatasetKind.INTERACTION_TRAJECTORY,
    DatasetKind.DESIGN_OUTCOME,
)


class TrajectorySourceKind(StrEnum):
    DETERMINISTIC_TEACHER = "deterministic_teacher"
    LEARNED_POLICY = "learned_policy"
    DETERMINISTIC_FALLBACK = "deterministic_fallback"
    IMPORTED_LEGACY = "imported_legacy"


class HighLevelTransitionKind(StrEnum):
    """How one learned high-level proposal entered the rollout dataset."""

    EXECUTED_TRAJECTORY = "executed_trajectory"
    CHECKER_REJECTION = "checker_rejection"


@dataclass
class TrajectoryProvenance(SchemaBase):
    source_kind: TrajectorySourceKind
    source_version: str
    policy_checkpoint_sha256: str | None = None
    parent_record_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        require_non_empty(self.source_version, "TrajectoryProvenance.source_version")
        if self.policy_checkpoint_sha256 is not None:
            require_non_empty(
                self.policy_checkpoint_sha256,
                "TrajectoryProvenance.policy_checkpoint_sha256",
            )
        if self.parent_record_id is not None:
            require_non_empty(
                self.parent_record_id,
                "TrajectoryProvenance.parent_record_id",
            )


@dataclass
class StageDecisionMasks(SchemaBase):
    """Credit-assignment masks required by Section 22.3."""

    design_decision_mask: bool = False
    high_level_decision_mask: bool = False
    low_level_control_mask: bool = False
    assembly_execution_mask: bool = False


@dataclass
class PolicyBehaviorTrace(SchemaBase):
    """Exact behavior-policy action needed for BC/PPO replay."""

    policy_family: Literal["pi_l", "pi_h", "pi_d"]
    policy_version: str
    action_semantics: str
    action_payload: dict[str, Any]
    stochastic: bool = False
    policy_checkpoint_sha256: str | None = None
    old_log_prob: float | None = None
    old_value: float | None = None
    recurrent_state_in: list[float] = field(default_factory=list)
    recurrent_state_out: list[float] = field(default_factory=list)

    def validate(self) -> None:
        require_non_empty(self.policy_version, "PolicyBehaviorTrace.policy_version")
        require_non_empty(self.action_semantics, "PolicyBehaviorTrace.action_semantics")
        _require_json_finite(self.action_payload, "PolicyBehaviorTrace.action_payload")
        if (self.old_log_prob is None) != (self.old_value is None):
            raise SchemaValidationError(
                "PolicyBehaviorTrace.old_log_prob and old_value must both be present or absent"
            )
        if self.old_log_prob is not None:
            _require_finite(self.old_log_prob, "PolicyBehaviorTrace.old_log_prob")
            _require_finite(self.old_value, "PolicyBehaviorTrace.old_value")
        for name in ("recurrent_state_in", "recurrent_state_out"):
            values = getattr(self, name)
            if not all(math.isfinite(float(value)) for value in values):
                raise SchemaValidationError(f"PolicyBehaviorTrace.{name} must be finite")
        if bool(self.recurrent_state_in) != bool(self.recurrent_state_out):
            raise SchemaValidationError(
                "PolicyBehaviorTrace recurrent states must both be present or absent"
            )
        if self.recurrent_state_in and len(self.recurrent_state_in) != len(
            self.recurrent_state_out
        ):
            raise SchemaValidationError(
                "PolicyBehaviorTrace recurrent state widths must match"
            )
        if self.policy_checkpoint_sha256 is not None:
            _require_sha256(
                self.policy_checkpoint_sha256,
                "PolicyBehaviorTrace.policy_checkpoint_sha256",
            )
        if self.stochastic:
            if self.policy_checkpoint_sha256 is None or self.old_log_prob is None:
                raise SchemaValidationError(
                    "stochastic PolicyBehaviorTrace requires checkpoint hash and old policy values"
                )


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
    task_type: str | None = None
    task_adapter_id: str | None = None
    phase_index: int | None = None
    phase_count: int | None = None
    truncated: bool = False
    bootstrap_value: float = 0.0
    behavior_trace: PolicyBehaviorTrace | None = None

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
        if self.terminal and self.truncated:
            raise SchemaValidationError(
                "LowLevelControlRecord cannot be terminal and truncated simultaneously"
            )
        _require_finite(self.bootstrap_value, "LowLevelControlRecord.bootstrap_value")
        if not self.truncated and not math.isclose(self.bootstrap_value, 0.0, abs_tol=1.0e-12):
            raise SchemaValidationError(
                "LowLevelControlRecord bootstrap_value is restricted to truncation boundaries"
            )
        phase_fields = (
            self.task_type,
            self.task_adapter_id,
            self.phase_index,
            self.phase_count,
        )
        if any(value is not None for value in phase_fields):
            if any(value is None for value in phase_fields):
                raise SchemaValidationError(
                    "LowLevelControlRecord Order9 task/phase fields must be supplied together"
                )
            require_non_empty(
                self.task_adapter_id or "", "LowLevelControlRecord.task_adapter_id"
            )
            require_non_empty(
                self.task_type or "", "LowLevelControlRecord.task_type"
            )
            if self.phase_count is None or self.phase_count < 1:
                raise SchemaValidationError("LowLevelControlRecord.phase_count must be positive")
            if self.phase_index is None or not 0 <= self.phase_index < self.phase_count:
                raise SchemaValidationError(
                    "LowLevelControlRecord.phase_index must lie within phase_count"
                )
        if self.behavior_trace is not None and self.behavior_trace.policy_family != "pi_l":
            raise SchemaValidationError(
                "LowLevelControlRecord behavior trace must belong to pi_l"
            )


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
    decision_reward: float | None = None
    trajectory_provenance: TrajectoryProvenance | None = None
    trajectory_feasibility_result: TrajectoryFeasibilityResult | None = None
    terminal: bool = False
    truncated: bool = False
    bootstrap_value: float = 0.0
    behavior_trace: PolicyBehaviorTrace | None = None
    transition_kind: HighLevelTransitionKind = (
        HighLevelTransitionKind.EXECUTED_TRAJECTORY
    )
    proposal_attempt_index: int = 0
    fallback_reward_credited: bool = False

    def validate(self) -> None:
        _require_record_identity(self.record_id, self.episode_id, self.task_id, type(self).__name__)
        _require_non_negative_int(self.decision_index, "InteractionTrajectoryRecord.decision_index")
        _require_finite_non_negative(
            self.decision_time_s,
            "InteractionTrajectoryRecord.decision_time_s",
        )
        _require_finite(self.decision_return, "InteractionTrajectoryRecord.decision_return")
        if self.decision_reward is not None:
            _require_finite(
                self.decision_reward,
                "InteractionTrajectoryRecord.decision_reward",
            )
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
        if self.trajectory_feasibility_result is not None:
            result = self.trajectory_feasibility_result
            if result.contract_version != self.trajectory.contract_version:
                raise SchemaValidationError(
                    "InteractionTrajectoryRecord trajectory feasibility contract must "
                    "match trajectory.contract_version"
                )
            if len(result.knot_results) != len(self.trajectory.knots):
                raise SchemaValidationError(
                    "InteractionTrajectoryRecord trajectory feasibility result must "
                    "cover every trajectory knot"
                )
        if self.terminal and self.truncated:
            raise SchemaValidationError(
                "InteractionTrajectoryRecord cannot be terminal and truncated simultaneously"
            )
        _require_finite(
            self.bootstrap_value, "InteractionTrajectoryRecord.bootstrap_value"
        )
        if not self.truncated and not math.isclose(self.bootstrap_value, 0.0, abs_tol=1.0e-12):
            raise SchemaValidationError(
                "InteractionTrajectoryRecord bootstrap_value is restricted to truncation boundaries"
            )
        if self.behavior_trace is not None and self.behavior_trace.policy_family != "pi_h":
            raise SchemaValidationError(
                "InteractionTrajectoryRecord behavior trace must belong to pi_h"
            )
        if (
            self.behavior_trace is not None
            and self.behavior_trace.stochastic
            and self.decision_reward is None
        ):
            raise SchemaValidationError(
                "stochastic InteractionTrajectoryRecord requires decision_reward for PPO"
            )
        _require_non_negative_int(
            self.proposal_attempt_index,
            "InteractionTrajectoryRecord.proposal_attempt_index",
        )
        if self.fallback_reward_credited:
            raise SchemaValidationError(
                "deterministic fallback reward cannot be credited to pi_h"
            )
        if self.transition_kind == HighLevelTransitionKind.CHECKER_REJECTION:
            if (
                self.trajectory_feasibility_result is None
                or self.trajectory_feasibility_result.feasible
            ):
                raise SchemaValidationError(
                    "checker-rejection transition requires an infeasible C_H result"
                )
            if not self.terminal or self.truncated:
                raise SchemaValidationError(
                    "checker-rejection transition must be an independent terminal GAE boundary"
                )
            if self.decision_reward is None or self.decision_reward >= 0.0:
                raise SchemaValidationError(
                    "checker-rejection transition requires a negative decision reward"
                )
            if self.behavior_trace is None or not self.behavior_trace.stochastic:
                raise SchemaValidationError(
                    "checker-rejection transition requires its stochastic pi_H behavior"
                )
        elif (
            self.trajectory_feasibility_result is not None
            and not self.trajectory_feasibility_result.feasible
        ):
            raise SchemaValidationError(
                "an infeasible pi_H trajectory cannot be marked as executed"
            )


@dataclass
class DesignActionCandidateRecord(SchemaBase):
    candidate_index: int
    action: DesignAction
    valid: bool
    reason_code: str
    score_prior: float = 0.0

    def validate(self) -> None:
        _require_non_negative_int(
            self.candidate_index, "DesignActionCandidateRecord.candidate_index"
        )
        require_non_empty(self.reason_code, "DesignActionCandidateRecord.reason_code")
        _require_finite(self.score_prior, "DesignActionCandidateRecord.score_prior")


@dataclass
class SequentialDesignStepRecord(SchemaBase):
    step_index: int
    partial_action_history: list[DesignAction]
    candidates: list[DesignActionCandidateRecord]
    selected_candidate_index: int
    reward: float
    terminal: bool
    truncated: bool
    bootstrap_value: float = 0.0
    behavior_trace: PolicyBehaviorTrace | None = None

    def validate(self) -> None:
        _require_non_negative_int(self.step_index, "SequentialDesignStepRecord.step_index")
        if not self.candidates:
            raise SchemaValidationError("SequentialDesignStepRecord.candidates must be non-empty")
        if [candidate.candidate_index for candidate in self.candidates] != list(
            range(len(self.candidates))
        ):
            raise SchemaValidationError(
                "SequentialDesignStepRecord candidate indices must be contiguous and ordered"
            )
        if not 0 <= self.selected_candidate_index < len(self.candidates):
            raise SchemaValidationError(
                "SequentialDesignStepRecord selected candidate index is out of range"
            )
        if not self.candidates[self.selected_candidate_index].valid:
            raise SchemaValidationError(
                "SequentialDesignStepRecord selected candidate must pass the deterministic mask"
            )
        _require_finite(self.reward, "SequentialDesignStepRecord.reward")
        _require_finite(self.bootstrap_value, "SequentialDesignStepRecord.bootstrap_value")
        if self.terminal and self.truncated:
            raise SchemaValidationError(
                "SequentialDesignStepRecord cannot be terminal and truncated simultaneously"
            )
        if not self.truncated and not math.isclose(self.bootstrap_value, 0.0, abs_tol=1.0e-12):
            raise SchemaValidationError(
                "SequentialDesignStepRecord bootstrap_value is restricted to truncation boundaries"
            )
        if self.behavior_trace is not None:
            if self.behavior_trace.policy_family != "pi_d":
                raise SchemaValidationError(
                    "SequentialDesignStepRecord behavior trace must belong to pi_d"
                )
            selected = self.candidates[self.selected_candidate_index].action.to_dict()
            payload_selected = self.behavior_trace.action_payload.get("selected_action")
            if payload_selected is not None and payload_selected != selected:
                raise SchemaValidationError(
                    "SequentialDesignStepRecord selected action conflicts with behavior trace"
                )


@dataclass
class SequentialDesignTrajectoryRecord(SchemaBase):
    """One complete sequential pi_D decision with exact runtime masks."""

    record_id: str
    episode_id: str
    task_id: str
    split: DatasetSplit
    task_spec: TaskSpec
    irg: InteractionRequirementGraph
    interaction_envelope: InteractionEnvelope
    physical_model_hash: str
    steps: list[SequentialDesignStepRecord]
    design_output: DesignOutput | None
    feasibility_result: FeasibilityResult | None
    episode_return: float
    task_success: bool
    failure_reason: str | None
    stage_masks: StageDecisionMasks
    trajectory_provenance: TrajectoryProvenance

    def validate(self) -> None:
        _require_record_identity(self.record_id, self.episode_id, self.task_id, type(self).__name__)
        require_non_empty(
            self.physical_model_hash,
            "SequentialDesignTrajectoryRecord.physical_model_hash",
        )
        if self.task_spec.task_id != self.task_id or self.irg.task_id != self.task_id:
            raise SchemaValidationError(
                "SequentialDesignTrajectoryRecord task identity mismatch"
            )
        if self.interaction_envelope.task_id != self.task_id:
            raise SchemaValidationError(
                "SequentialDesignTrajectoryRecord envelope task identity mismatch"
            )
        if not self.stage_masks.design_decision_mask:
            raise SchemaValidationError(
                "SequentialDesignTrajectoryRecord requires design_decision_mask"
            )
        if not self.steps:
            raise SchemaValidationError(
                "SequentialDesignTrajectoryRecord.steps must be non-empty"
            )
        if [step.step_index for step in self.steps] != list(range(len(self.steps))):
            raise SchemaValidationError(
                "SequentialDesignTrajectoryRecord step indices must be contiguous and ordered"
            )
        selected_history: list[DesignAction] = []
        for step in self.steps:
            if [action.to_dict() for action in step.partial_action_history] != [
                action.to_dict() for action in selected_history
            ]:
                raise SchemaValidationError(
                    "SequentialDesignTrajectoryRecord partial action history is not replayable"
                )
            selected_history.append(
                step.candidates[step.selected_candidate_index].action
            )
        boundaries = [
            index
            for index, step in enumerate(self.steps)
            if step.terminal or step.truncated
        ]
        if boundaries != [len(self.steps) - 1]:
            raise SchemaValidationError(
                "SequentialDesignTrajectoryRecord requires exactly one final boundary"
            )
        if (self.design_output is None) != (self.feasibility_result is None):
            raise SchemaValidationError(
                "SequentialDesignTrajectoryRecord design and feasibility must both be present or absent"
            )
        if self.design_output is not None:
            if self.design_output.task_id != self.task_id:
                raise SchemaValidationError(
                    "SequentialDesignTrajectoryRecord design task identity mismatch"
                )
            if [action.to_dict() for action in self.design_output.design_actions] != [
                action.to_dict() for action in selected_history
            ]:
                raise SchemaValidationError(
                    "SequentialDesignTrajectoryRecord design action trace mismatch"
                )
        if self.task_success:
            if self.design_output is None or not self.feasibility_result.feasible:
                raise SchemaValidationError(
                    "successful SequentialDesignTrajectoryRecord requires a feasible design"
                )
            if self.failure_reason is not None:
                raise SchemaValidationError(
                    "successful SequentialDesignTrajectoryRecord cannot have failure_reason"
                )
        elif self.failure_reason is None:
            raise SchemaValidationError(
                "failed SequentialDesignTrajectoryRecord requires failure_reason"
            )
        _require_finite(
            self.episode_return, "SequentialDesignTrajectoryRecord.episode_return"
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
    schema_version: Literal["p4_3_dataset_v1", "p4_3_dataset_v2"]
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


def _require_sha256(value: str, path: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise SchemaValidationError(f"{path} must be a lowercase SHA-256 digest")


def _require_json_finite(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise SchemaValidationError(f"{path} must not contain non-finite numbers")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_json_finite(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise SchemaValidationError(f"{path} keys must be non-empty strings")
            _require_json_finite(item, f"{path}[{key!r}]")
        return
    raise SchemaValidationError(f"{path} contains a non-JSON value")
