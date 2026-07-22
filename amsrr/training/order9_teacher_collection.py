from __future__ import annotations

"""Production C0 teacher capture, episode shards, and verified dataset assembly."""

import gzip
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from amsrr.feasibility.contact_wrench_trajectory import (
    ContactWrenchTrajectoryCheckerConfig,
    ContactWrenchTrajectoryFeasibilityChecker,
)
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_non_empty
from amsrr.schemas.contact_candidates import ContactCandidateSet
from amsrr.schemas.datasets import (
    P4_3_DATASET_SCHEMA_VERSION,
    DatasetKind,
    DatasetShard,
    DatasetSplit,
    InteractionTrajectoryRecord,
    LowLevelControlRecord,
    P4_3DatasetManifest,
    PolicyBehaviorTrace,
    StageDecisionMasks,
)
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.policies import (
    CONTACT_WRENCH_CONTRACT_CONTACT_FRAME,
    CentroidalTarget,
    ContactWrenchTrajectory,
    ControllerCommand,
    PolicyCommand,
    PostureTarget,
)
from amsrr.schemas.runtime import RuntimeObservation
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.order9_reward import ActorPhaseContext, Order9RewardEngine
from amsrr.training.order9_curriculum import ORDER9_C0_COLLECTION_PROFILE_VERSION
from amsrr.training.order9_teacher import (
    ORDER9_NATURAL_CONTACT_TEACHER_VERSION,
    rolling_teacher_snapshot_to_v2,
    teacher_interaction_record,
)
from amsrr.training.order9_teacher_windows import (
    Order9TeacherWindowConfig,
    compose_order9_teacher_windows,
)
from amsrr.utils.hashing import hash_file, stable_hash


ORDER9_TEACHER_COLLECTION_VERSION = "order9_c0_centroidal_teacher_collection_v4"
ORDER9_TEACHER_EPISODE_MANIFEST_VERSION = "order9_teacher_episode_manifest_v1"
ORDER9_TEACHER_DATASET_BUILDER_VERSION = "order9_centroidal_teacher_dataset_builder_v3"
ORDER9_LOW_LEVEL_TEACHER_VERSION = (
    "order8_deterministic_complete_policy_command_teacher_v2"
)


@dataclass(frozen=True)
class Order9TeacherCollectionConfig:
    episode_id: str
    split: DatasetSplit
    low_level_stride: int = 1
    high_level_stride: int = 5
    discount_gamma: float = 0.99
    window_horizon_s: float = 2.0
    window_knot_dt_s: float = 0.10

    def validate(self) -> None:
        if not self.episode_id:
            raise SchemaValidationError("Order9 teacher episode_id must be non-empty")
        if self.low_level_stride < 1 or self.high_level_stride < 1:
            raise SchemaValidationError("Order9 teacher strides must be positive")
        if not math.isfinite(self.discount_gamma) or not 0.0 <= self.discount_gamma <= 1.0:
            raise SchemaValidationError("Order9 teacher discount_gamma must be in [0, 1]")
        if (
            not math.isfinite(self.window_horizon_s)
            or not math.isfinite(self.window_knot_dt_s)
            or self.window_horizon_s <= 0.0
            or self.window_knot_dt_s <= 0.0
        ):
            raise SchemaValidationError("Order9 teacher window values must be positive")


@dataclass
class Order9TeacherEpisodeResult:
    episode_id: str
    task_spec: TaskSpec
    split: DatasetSplit
    success: bool
    failure_reason: str | None
    low_level_records: list[LowLevelControlRecord]
    trajectory_records: list[InteractionTrajectoryRecord]
    source_trajectory_records: list[InteractionTrajectoryRecord]
    metrics: dict[str, float]


@dataclass
class Order9TeacherEpisodeManifest(SchemaBase):
    manifest_version: str
    collection_version: str
    episode_id: str
    task_spec: TaskSpec
    split: DatasetSplit
    random_seed: int
    success: bool
    failure_reason: str | None
    low_level_shard_path: str
    low_level_record_count: int
    low_level_shard_sha256: str
    trajectory_shard_path: str
    trajectory_record_count: int
    trajectory_shard_sha256: str
    robot_model_hash: str
    urdf_hash: str
    thrust_model_hash: str
    config_hash: str
    simulator_version: str
    simulator_hash: str
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.manifest_version != ORDER9_TEACHER_EPISODE_MANIFEST_VERSION:
            raise SchemaValidationError("Order9 teacher episode manifest version mismatch")
        if self.collection_version != ORDER9_TEACHER_COLLECTION_VERSION:
            raise SchemaValidationError("Order9 teacher collection version mismatch")
        for name in (
            "episode_id",
            "low_level_shard_path",
            "trajectory_shard_path",
            "robot_model_hash",
            "urdf_hash",
            "thrust_model_hash",
            "config_hash",
            "simulator_version",
            "simulator_hash",
        ):
            require_non_empty(str(getattr(self, name)), f"Order9TeacherEpisodeManifest.{name}")
        if self.task_spec.task_id == "":
            raise SchemaValidationError("Order9 teacher task_id must be non-empty")
        if self.random_seed < 0:
            raise SchemaValidationError("Order9 teacher random_seed must be non-negative")
        if self.low_level_record_count < 1 or self.trajectory_record_count < 1:
            raise SchemaValidationError("Order9 teacher episode shards must be non-empty")
        for name in ("low_level_shard_sha256", "trajectory_shard_sha256"):
            value = str(getattr(self, name))
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise SchemaValidationError(f"Order9TeacherEpisodeManifest.{name} must be SHA-256")
        if self.success == (self.failure_reason is not None):
            raise SchemaValidationError(
                "successful teacher episodes cannot have failure_reason and failed episodes require it"
            )
        if any(not math.isfinite(float(value)) for value in self.metrics.values()):
            raise SchemaValidationError("Order9 teacher episode metrics must be finite")


@dataclass
class _PendingCommand:
    frame_index: int
    actor_observation: RuntimeObservation
    reward_observation: RuntimeObservation
    trajectory: ContactWrenchTrajectory
    policy_command: PolicyCommand
    controller_command: ControllerCommand
    actuator_target_record: dict[str, Any]


@dataclass
class _CompletedStep:
    frame_index: int
    actor_observation: RuntimeObservation
    post_reward_observation: RuntimeObservation
    trajectory: ContactWrenchTrajectory
    policy_command: PolicyCommand
    controller_command: ControllerCommand
    actuator_target_record: dict[str, Any]
    reward: float
    reward_terms: dict[str, float]
    phase_context: ActorPhaseContext


class Order9TeacherEpisodeCollector:
    """Capture exact Order 8 commands while keeping privileged truth actor-hidden."""

    def __init__(
        self,
        *,
        task_spec: TaskSpec,
        morphology_graph: MorphologyGraph,
        contact_candidate_set: ContactCandidateSet,
        config: Order9TeacherCollectionConfig,
        reward_engine: Order9RewardEngine | None = None,
        checker: ContactWrenchTrajectoryFeasibilityChecker | None = None,
    ) -> None:
        config.validate()
        task_spec.validate()
        morphology_graph.validate()
        contact_candidate_set.validate()
        if contact_candidate_set.task_id != task_spec.task_id:
            raise SchemaValidationError("teacher candidate/task identity mismatch")
        if contact_candidate_set.morphology_graph_id != morphology_graph.graph_id:
            raise SchemaValidationError("teacher candidate/morphology identity mismatch")
        self.task_spec = task_spec
        self.morphology_graph = morphology_graph
        self.contact_candidate_set = contact_candidate_set
        self.config = config
        self.reward_engine = reward_engine or Order9RewardEngine()
        self.checker = checker or ContactWrenchTrajectoryFeasibilityChecker(
            config=ContactWrenchTrajectoryCheckerConfig.warmup_proxy()
        )
        self._base_context = _compile_context(
            task_spec,
            morphology_graph,
            contact_candidate_set,
            runtime_observation=None,
        )
        self._latest_actor_observation: RuntimeObservation | None = None
        self._latest_reward_observation: RuntimeObservation | None = None
        self._pending: _PendingCommand | None = None
        self._completed: list[_CompletedStep] = []
        self._command_count = 0
        self._finalized = False

    @property
    def pending_command(self) -> bool:
        return self._pending is not None

    def observe_state(
        self,
        *,
        actor_observation: RuntimeObservation,
        reward_observation: RuntimeObservation,
    ) -> None:
        """Install a state and causally close the preceding command transition."""

        self._require_open()
        actor_observation.validate()
        reward_observation.validate()
        _require_actor_safe(actor_observation)
        _require_observation_pair(actor_observation, reward_observation)
        if self._latest_actor_observation is not None and (
            actor_observation.time_s <= self._latest_actor_observation.time_s
        ):
            raise SchemaValidationError("teacher observation times must increase strictly")
        if self._pending is not None:
            output = self.reward_engine.step(
                task_spec=self.task_spec,
                observation=reward_observation,
                previous_observation=self._pending.reward_observation,
                controller_command=self._pending.controller_command,
                actuator_target_record=self._pending.actuator_target_record,
                state_transition_available=True,
            )
            terms = dict(output.terms)
            terms.update(
                {
                    "transition_start_time_s": float(
                        self._pending.actor_observation.time_s
                    ),
                    "transition_end_time_s": float(actor_observation.time_s),
                    "transition_dt_s": float(
                        actor_observation.time_s
                        - self._pending.actor_observation.time_s
                    ),
                    "privileged_reward_observation_only": 1.0,
                }
            )
            self._completed.append(
                _CompletedStep(
                    frame_index=self._pending.frame_index,
                    actor_observation=self._pending.actor_observation,
                    post_reward_observation=reward_observation,
                    trajectory=self._pending.trajectory,
                    policy_command=self._pending.policy_command,
                    controller_command=self._pending.controller_command,
                    actuator_target_record=self._pending.actuator_target_record,
                    reward=float(output.reward),
                    reward_terms=terms,
                    phase_context=output.phase_context,
                )
            )
            self._pending = None
        self._latest_actor_observation = actor_observation
        self._latest_reward_observation = reward_observation

    def record_command(
        self,
        *,
        trajectory: ContactWrenchTrajectory,
        centroidal_reference_pose_world: Sequence[float],
        centroidal_reference_twist: Sequence[float],
        policy_command: PolicyCommand,
        controller_command: ControllerCommand,
        actuator_target_record: Mapping[str, Any],
        decision_dt_s: float,
    ) -> None:
        """Bind the exact applied teacher command to the latest pre-state."""

        self._require_open()
        if self._pending is not None:
            raise SchemaValidationError("teacher command recorded before prior transition closed")
        if self._latest_actor_observation is None or self._latest_reward_observation is None:
            raise SchemaValidationError("teacher command requires an observed pre-state")
        phase_label = self._latest_actor_observation.task_progress.phase_label
        if not phase_label:
            raise SchemaValidationError("teacher command requires actor-visible phase")
        context = self._context(self._latest_actor_observation)
        snapshot = rolling_teacher_snapshot_to_v2(
            trajectory,
            context,
            decision_dt_s=decision_dt_s,
            phase_label=phase_label,
        )
        _install_order9_policy_references(
            snapshot,
            centroidal_reference_pose_world=centroidal_reference_pose_world,
            centroidal_reference_twist=centroidal_reference_twist,
            policy_command=policy_command,
        )
        self._pending = _PendingCommand(
            frame_index=self._command_count,
            actor_observation=self._latest_actor_observation,
            reward_observation=self._latest_reward_observation,
            trajectory=snapshot,
            policy_command=PolicyCommand.from_dict(policy_command.to_dict()),
            controller_command=ControllerCommand.from_dict(
                controller_command.to_dict()
            ),
            actuator_target_record=json.loads(
                json.dumps(dict(actuator_target_record), sort_keys=True)
            ),
        )
        self._command_count += 1

    def finalize(
        self,
        *,
        success: bool,
        failure_reason: str | None,
        release_valid: bool | None,
        object_dropped: bool | None,
        hard_collision: bool | None,
        timeout: bool | None,
        qp_infeasible_terminal: bool | None,
    ) -> Order9TeacherEpisodeResult:
        self._require_open()
        if self._pending is not None:
            raise SchemaValidationError(
                "teacher collector has an unclosed final command; observe its post-state first"
            )
        if not self._completed:
            raise SchemaValidationError("teacher collector captured no complete transitions")
        if success == (failure_reason is not None):
            raise SchemaValidationError(
                "successful teacher episode cannot have failure_reason and failure requires one"
            )
        final_observation = self._completed[-1].post_reward_observation
        terminal_terms = self.reward_engine.terminal(
            task_spec=self.task_spec,
            observation=final_observation,
            release_valid=release_valid,
            object_dropped=object_dropped,
            hard_collision=hard_collision,
            timeout=timeout,
            qp_infeasible_terminal=qp_infeasible_terminal,
        )
        terminal_reward = float(terminal_terms["terminal_reward"])
        self._completed[-1].reward += terminal_reward
        self._completed[-1].reward_terms.update(terminal_terms)
        self._completed[-1].reward_terms["terminal_reward_data_available"] = 1.0
        self._completed[-1].reward_terms["reward"] = self._completed[-1].reward

        source_records, selected_source_positions = self._source_records()
        full_records = compose_order9_teacher_windows(
            source_records,
            checker=self.checker,
            config=Order9TeacherWindowConfig(
                horizon_s=self.config.window_horizon_s,
                knot_dt_s=self.config.window_knot_dt_s,
            ),
        )
        if len(full_records) != len(source_records):
            raise SchemaValidationError(
                "terminal teacher episode must yield a full window for every source decision"
            )
        for source, full in zip(source_records, full_records, strict=True):
            full.behavior_trace = _pi_h_teacher_trace(full.trajectory)
            full.terminal = source.terminal
            full.truncated = source.truncated
            full.bootstrap_value = source.bootstrap_value
            full.validate()
        low_records = self._low_level_records(
            full_records,
            selected_source_positions,
        )
        self._finalized = True
        rewards = [step.reward for step in self._completed]
        return Order9TeacherEpisodeResult(
            episode_id=self.config.episode_id,
            task_spec=self.task_spec,
            split=self.config.split,
            success=bool(success),
            failure_reason=failure_reason,
            low_level_records=low_records,
            trajectory_records=full_records,
            source_trajectory_records=source_records,
            metrics={
                "success": 1.0 if success else 0.0,
                "control_transition_count": float(len(self._completed)),
                "low_level_record_count": float(len(low_records)),
                "high_level_record_count": float(len(full_records)),
                "episode_return": float(sum(rewards)),
                "terminal_reward": terminal_reward,
                "raw_contact_actor_input": 0.0,
                "warmup_proxy_c_h": 1.0,
            },
        )

    def _source_records(
        self,
    ) -> tuple[list[InteractionTrajectoryRecord], list[int]]:
        selected = _decision_positions(
            self._completed,
            stride=self.config.high_level_stride,
        )
        returns = _discounted_step_returns(
            self._completed,
            gamma=self.config.discount_gamma,
        )
        records: list[InteractionTrajectoryRecord] = []
        for decision_index, position in enumerate(selected):
            step = self._completed[position]
            next_position = (
                selected[decision_index + 1]
                if decision_index + 1 < len(selected)
                else len(self._completed)
            )
            decision_reward = sum(
                value.reward for value in self._completed[position:next_position]
            )
            context = self._context(step.actor_observation)
            record = teacher_interaction_record(
                record_id=(
                    f"{self.config.episode_id}:teacher-source:{decision_index:06d}"
                ),
                episode_id=self.config.episode_id,
                split=self.config.split,
                decision_index=decision_index,
                context=context,
                trajectory=step.trajectory,
                checker=self.checker,
                decision_return=returns[position],
            )
            record.decision_reward = float(decision_reward)
            record.behavior_trace = _pi_h_teacher_trace(step.trajectory)
            record.terminal = decision_index == len(selected) - 1
            record.validate()
            records.append(record)
        return records, selected

    def _low_level_records(
        self,
        trajectory_records: Sequence[InteractionTrajectoryRecord],
        source_positions: Sequence[int],
    ) -> list[LowLevelControlRecord]:
        selected_steps = _stride_positions(
            len(self._completed), self.config.low_level_stride
        )
        records: list[LowLevelControlRecord] = []
        previous_position = -1
        source_cursor = 0
        for output_index, position in enumerate(selected_steps):
            while (
                source_cursor + 1 < len(source_positions)
                and source_positions[source_cursor + 1] <= position
            ):
                source_cursor += 1
            step = self._completed[position]
            trajectory_record = trajectory_records[source_cursor]
            interval = self._completed[previous_position + 1 : position + 1]
            reward = float(sum(item.reward for item in interval))
            terms = dict(step.reward_terms)
            terms.update(
                {
                    "sampled_step_reward": float(step.reward),
                    "interval_aggregated_reward": reward,
                    "interval_start_frame": float(previous_position + 1),
                    "interval_end_frame": float(position),
                }
            )
            terminal = output_index == len(selected_steps) - 1
            record = LowLevelControlRecord(
                record_id=f"{self.config.episode_id}:low:{output_index:06d}",
                episode_id=self.config.episode_id,
                task_id=self.task_spec.task_id,
                split=self.config.split,
                step_index=output_index,
                time_s=step.actor_observation.time_s,
                trajectory_record_id=trajectory_record.record_id,
                active_trajectory_index=source_cursor,
                active_knot_index=0,
                runtime_observation=step.actor_observation,
                active_knot=step.trajectory.knots[0],
                policy_command=step.policy_command,
                controller_command=step.controller_command,
                actuator_target_record=step.actuator_target_record,
                reward_terms=terms,
                reward=reward,
                terminal=terminal,
                stage_masks=StageDecisionMasks(low_level_control_mask=True),
                task_type=self.task_spec.task_type.value,
                task_adapter_id=step.phase_context.task_adapter_id,
                phase_index=step.phase_context.phase_index,
                phase_count=step.phase_context.phase_count,
                behavior_trace=_pi_l_teacher_trace(step.policy_command),
            )
            record.validate()
            records.append(record)
            previous_position = position
        return records

    def _context(self, observation: RuntimeObservation) -> HighLevelPolicyContext:
        return HighLevelPolicyContext(
            irg=self._base_context.irg,
            interaction_envelope=self._base_context.interaction_envelope,
            morphology_graph=self.morphology_graph,
            contact_candidate_set=self.contact_candidate_set,
            runtime_observation=observation,
        )

    def _require_open(self) -> None:
        if self._finalized:
            raise SchemaValidationError("teacher collector is already finalized")


def write_order9_teacher_episode(
    result: Order9TeacherEpisodeResult,
    output_dir: str | Path,
    *,
    random_seed: int,
    robot_model_hash: str,
    urdf_hash: str,
    thrust_model_hash: str,
    config_hash: str,
    simulator_version: str,
    simulator_hash: str,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write one relocatable, independently hash-verified teacher episode."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    low_path = target / "low_level_control.jsonl.gz"
    high_path = target / "interaction_trajectory.jsonl.gz"
    _atomic_write_jsonl_gzip(low_path, result.low_level_records)
    _atomic_write_jsonl_gzip(high_path, result.trajectory_records)
    manifest = Order9TeacherEpisodeManifest(
        manifest_version=ORDER9_TEACHER_EPISODE_MANIFEST_VERSION,
        collection_version=ORDER9_TEACHER_COLLECTION_VERSION,
        episode_id=result.episode_id,
        task_spec=result.task_spec,
        split=result.split,
        random_seed=int(random_seed),
        success=result.success,
        failure_reason=result.failure_reason,
        low_level_shard_path=low_path.name,
        low_level_record_count=len(result.low_level_records),
        low_level_shard_sha256=hash_file(low_path),
        trajectory_shard_path=high_path.name,
        trajectory_record_count=len(result.trajectory_records),
        trajectory_shard_sha256=hash_file(high_path),
        robot_model_hash=robot_model_hash,
        urdf_hash=urdf_hash,
        thrust_model_hash=thrust_model_hash,
        config_hash=config_hash,
        simulator_version=simulator_version,
        simulator_hash=simulator_hash,
        metrics=dict(result.metrics),
        metadata=dict(metadata or {}),
    )
    path = target / "episode_manifest.json"
    _atomic_write_text(path, manifest.to_json(indent=2) + "\n")
    return path


def load_order9_teacher_episode(
    path: str | Path,
) -> tuple[
    Order9TeacherEpisodeManifest,
    list[LowLevelControlRecord],
    list[InteractionTrajectoryRecord],
]:
    manifest_path, manifest = load_order9_teacher_episode_manifest(path)
    low_path = manifest_path.parent / manifest.low_level_shard_path
    high_path = manifest_path.parent / manifest.trajectory_shard_path
    low = [LowLevelControlRecord.from_dict(row) for row in _iter_jsonl(low_path)]
    high = [InteractionTrajectoryRecord.from_dict(row) for row in _iter_jsonl(high_path)]
    if len(low) != manifest.low_level_record_count:
        raise SchemaValidationError("Order9 teacher low-level shard count mismatch")
    if len(high) != manifest.trajectory_record_count:
        raise SchemaValidationError("Order9 teacher trajectory shard count mismatch")
    for record in [*low, *high]:
        if (
            record.episode_id != manifest.episode_id
            or record.task_id != manifest.task_spec.task_id
            or record.split != manifest.split
        ):
            raise SchemaValidationError("Order9 teacher episode record identity mismatch")
    return manifest, low, high


def load_order9_teacher_low_level_records(
    path: str | Path,
) -> tuple[Order9TeacherEpisodeManifest, list[LowLevelControlRecord]]:
    """Load one verified episode's pi_L records without its trajectory shard."""

    manifest_path = _episode_manifest_path(Path(path))
    manifest = Order9TeacherEpisodeManifest.from_json(
        manifest_path.read_text(encoding="utf-8")
    )
    low_path = manifest_path.parent / manifest.low_level_shard_path
    if hash_file(low_path) != manifest.low_level_shard_sha256:
        raise SchemaValidationError("Order9 teacher low-level shard hash mismatch")
    low = [LowLevelControlRecord.from_dict(row) for row in _iter_jsonl(low_path)]
    if len(low) != manifest.low_level_record_count:
        raise SchemaValidationError("Order9 teacher low-level shard count mismatch")
    for record in low:
        if (
            record.episode_id != manifest.episode_id
            or record.task_id != manifest.task_spec.task_id
            or record.split != manifest.split
        ):
            raise SchemaValidationError("Order9 teacher episode record identity mismatch")
    return manifest, low


def validate_order9_teacher_pi_l_representability(
    path: str | Path,
    *,
    physical_model,
    policy_config=None,
) -> dict[str, float]:
    """Fail closed unless every C0 command fits the production pi_L decoder."""

    from amsrr.controllers.rigid_body_model import RigidBodyControlModelBuilder
    from amsrr.policies.low_level_policy_base import LowLevelPolicyContext
    from amsrr.policies.order9_low_level_policy import Order9LowLevelPolicyConfig
    from amsrr.training.order9_pi_l_learning import (
        encode_order9_pi_l_teacher_action,
    )

    config = policy_config or Order9LowLevelPolicyConfig()
    if not isinstance(config, Order9LowLevelPolicyConfig):
        raise TypeError("Order9 teacher validation requires an Order9 pi_L config")
    config.validate()
    manifest, records = load_order9_teacher_low_level_records(path)
    if manifest.robot_model_hash != physical_model.stable_hash():
        raise SchemaValidationError(
            "Order9 teacher PhysicalModel hash differs during representability validation"
        )
    builder = RigidBodyControlModelBuilder()
    maximum = 0.0
    for record in records:
        active_knot = type(record.active_knot).from_dict(
            record.active_knot.to_dict()
        )
        trajectory = ContactWrenchTrajectory(
            horizon_s=0.02,
            dt_s=0.02,
            knots=[active_knot],
            contract_version=CONTACT_WRENCH_CONTRACT_CONTACT_FRAME,
        )
        context = LowLevelPolicyContext(
            runtime_observation=record.runtime_observation,
            morphology_graph=record.runtime_observation.morphology_graph,
            physical_model=physical_model,
            contact_wrench_trajectory=trajectory,
            active_knot=active_knot,
            controller_status=record.runtime_observation.controller_status,
            task_type=record.task_type,
            task_adapter_id=record.task_adapter_id,
            phase_index=record.phase_index,
            phase_count=record.phase_count,
        )
        control_model = builder.build(
            context.morphology_graph,
            physical_model,
            context.runtime_observation,
        )
        encoded = encode_order9_pi_l_teacher_action(
            context=context,
            teacher_command=record.policy_command,
            control_model=control_model,
            config=config,
        )
        maximum = max(maximum, encoded.maximum_absolute_unclipped_action)
    return {
        "record_count": float(len(records)),
        "maximum_absolute_unclipped_action": float(maximum),
        "global_action_size": float(config.action_size),
    }


def load_order9_teacher_episode_manifest(
    path: str | Path,
) -> tuple[Path, Order9TeacherEpisodeManifest]:
    """Verify one episode manifest and its shard bytes without materializing rows."""

    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "episode_manifest.json"
    manifest = Order9TeacherEpisodeManifest.from_json(
        manifest_path.read_text(encoding="utf-8")
    )
    low_path = manifest_path.parent / manifest.low_level_shard_path
    high_path = manifest_path.parent / manifest.trajectory_shard_path
    if hash_file(low_path) != manifest.low_level_shard_sha256:
        raise SchemaValidationError("Order9 teacher low-level shard hash mismatch")
    if hash_file(high_path) != manifest.trajectory_shard_sha256:
        raise SchemaValidationError("Order9 teacher trajectory shard hash mismatch")
    return manifest_path, manifest


def build_order9_teacher_dataset(
    episode_manifest_paths: Iterable[str | Path],
    output_dir: str | Path,
) -> P4_3DatasetManifest:
    """Merge successful C0 episode bundles into task-disjoint verified shards."""

    paths = [Path(path) for path in episode_manifest_paths]
    if not paths:
        raise SchemaValidationError("Order9 teacher dataset requires episode manifests")
    loaded = [load_order9_teacher_episode_manifest(path) for path in paths]
    manifest_paths = [item[0] for item in loaded]
    manifests = [item[1] for item in loaded]
    if len({item.episode_id for item in manifests}) != len(manifests):
        raise SchemaValidationError("Order9 teacher episode IDs must be unique")
    failed = [item.episode_id for item in manifests if not item.success]
    if failed:
        raise SchemaValidationError(
            "failed Order9 teacher episodes cannot enter BC dataset: " + ",".join(failed)
        )
    profile_versions = {
        str(item.metadata.get("c0_collection_profile_version", ""))
        for item in manifests
    }
    if profile_versions != {ORDER9_C0_COLLECTION_PROFILE_VERSION}:
        raise SchemaValidationError(
            "Order9 C0 dataset requires one current bounded-diversity profile"
        )
    condition_ids = [str(item.metadata.get("c0_condition_id", "")) for item in manifests]
    if any(not value for value in condition_ids) or len(set(condition_ids)) != len(
        condition_ids
    ):
        raise SchemaValidationError(
            "Order9 C0 dataset requires unique non-empty condition identities"
        )
    low_level_strides = {
        int(item.metadata.get("teacher_low_level_stride", -1)) for item in manifests
    }
    high_level_strides = {
        int(item.metadata.get("teacher_high_level_stride", -1)) for item in manifests
    }
    if min(*low_level_strides, *high_level_strides) < 1 or (
        len(low_level_strides) != 1 or len(high_level_strides) != 1
    ):
        raise SchemaValidationError(
            "Order9 C0 dataset teacher strides must be positive and uniform"
        )
    split_by_task: dict[str, DatasetSplit] = {}
    for item in manifests:
        previous = split_by_task.setdefault(item.task_spec.task_id, item.split)
        if previous != item.split:
            raise SchemaValidationError("Order9 teacher task crosses dataset splits")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    shards: list[DatasetShard] = []
    for split in DatasetSplit:
        for kind, shard_name, count_name in (
            (
                DatasetKind.LOW_LEVEL_CONTROL,
                "low_level_shard_path",
                "low_level_record_count",
            ),
            (
                DatasetKind.INTERACTION_TRAJECTORY,
                "trajectory_shard_path",
                "trajectory_record_count",
            ),
        ):
            shard_path = target / f"{kind.value}_{split.value}.jsonl.gz"
            selected = [
                manifest_path.parent / str(getattr(manifest, shard_name))
                for manifest_path, manifest in zip(
                    manifest_paths, manifests, strict=True
                )
                if manifest.split == split
            ]
            record_count = sum(
                int(getattr(manifest, count_name))
                for manifest in manifests
                if manifest.split == split
            )
            # Concatenated gzip members are part of the gzip format and are read
            # transparently by Python's gzip module.  Copying the already
            # hash-verified episode members keeps C0 assembly O(1) in RAM while
            # preserving every serialized training record byte-for-byte.
            _atomic_concatenate_files(shard_path, selected)
            shards.append(
                DatasetShard(
                    dataset_kind=kind,
                    split=split,
                    path=shard_path.name,
                    record_count=record_count,
                    sha256=hash_file(shard_path),
                )
            )
    split_tasks = {
        split: sorted(task for task, value in split_by_task.items() if value == split)
        for split in DatasetSplit
    }
    task_hashes = {item.task_spec.task_id: item.task_spec.stable_hash() for item in manifests}
    geometry_hashes = {
        f"{item.task_spec.task_id}:{geometry.geometry_id}": geometry.stable_hash()
        for item in manifests
        for geometry in item.task_spec.scene.geometry_library
    }
    record_counts = {
        kind.value: sum(
            shard.record_count for shard in shards if shard.dataset_kind == kind
        )
        for kind in DatasetKind
    }
    source_digests = [hash_file(path) for path in manifest_paths]
    manifest = P4_3DatasetManifest(
        dataset_id=(
            "order9-c0-teacher-"
            + stable_hash(
                {
                    "episodes": sorted(item.episode_id for item in manifests),
                    "source_digests": sorted(source_digests),
                    "shards": [shard.sha256 for shard in shards],
                }
            )[:16]
        ),
        schema_version=P4_3_DATASET_SCHEMA_VERSION,
        source_archive_paths=[str(path) for path in manifest_paths],
        source_episode_ids=[item.episode_id for item in manifests],
        train_task_ids=split_tasks[DatasetSplit.TRAIN],
        validation_task_ids=split_tasks[DatasetSplit.VALIDATION],
        held_out_task_ids=split_tasks[DatasetSplit.HELD_OUT],
        shards=shards,
        record_counts=record_counts,
        source_hash=stable_hash(sorted(source_digests)),
        config_hash=stable_hash(sorted({item.config_hash for item in manifests})),
        robot_model_hash=_single_or_combined(item.robot_model_hash for item in manifests),
        urdf_hash=_single_or_combined(item.urdf_hash for item in manifests),
        thrust_model_hash=_single_or_combined(item.thrust_model_hash for item in manifests),
        task_hashes=task_hashes,
        geometry_hashes=geometry_hashes,
        random_seeds=sorted({item.random_seed for item in manifests}),
        simulator_version="+".join(sorted({item.simulator_version for item in manifests})),
        simulator_hash=_single_or_combined(item.simulator_hash for item in manifests),
        metadata={
            "builder_version": ORDER9_TEACHER_DATASET_BUILDER_VERSION,
            "collection_version": ORDER9_TEACHER_COLLECTION_VERSION,
            "c0_collection_profile_version": ORDER9_C0_COLLECTION_PROFILE_VERSION,
            "c0_condition_ids": condition_ids,
            "teacher_low_level_stride": next(iter(low_level_strides)),
            "teacher_high_level_stride": next(iter(high_level_strides)),
            "phase_balanced_pi_l_sampling_required": True,
            "task_disjoint_splits": True,
            "source_episode_count": len(manifests),
            "successful_episode_count": len(manifests),
            "raw_contact_actor_input": False,
            "privileged_contact_role": "reward_and_safety_only",
            "teacher_c_h_mode": "warmup_proxy",
            "full_trajectory_semantics": "rolling_snapshots_zero_order_hold",
            "gzip_shards": True,
        },
    )
    _atomic_write_text(target / "manifest.json", manifest.to_json(indent=2) + "\n")
    return manifest


def _compile_context(
    task_spec: TaskSpec,
    morphology_graph: MorphologyGraph,
    contact_candidate_set: ContactCandidateSet,
    *,
    runtime_observation: RuntimeObservation | None,
) -> HighLevelPolicyContext:
    from amsrr.training.order9_teacher import compile_high_level_context

    return compile_high_level_context(
        task_spec,
        morphology_graph,
        contact_candidate_set,
        runtime_observation=runtime_observation,
    )


def _require_actor_safe(observation: RuntimeObservation) -> None:
    if observation.contact_states:
        raise SchemaValidationError("Order9 teacher actor observation contains raw contact states")
    forbidden = {
        "raw_contact",
        "contact_force",
        "contact_wrench",
        "penetration",
        "grasp_acquired",
        "hard_collision",
        "slip",
    }
    leaked = sorted(
        key
        for key in observation.task_progress.metrics
        if any(token in key.lower() for token in forbidden)
    )
    if leaked:
        raise SchemaValidationError(
            "Order9 teacher actor observation leaked privileged metrics: "
            + ",".join(leaked)
        )


def _require_observation_pair(
    actor: RuntimeObservation,
    reward: RuntimeObservation,
) -> None:
    if not math.isclose(actor.time_s, reward.time_s, abs_tol=1.0e-9):
        raise SchemaValidationError("teacher actor/reward observation times differ")
    if actor.morphology_graph.stable_hash() != reward.morphology_graph.stable_hash():
        raise SchemaValidationError("teacher actor/reward morphologies differ")
    if [state.to_dict() for state in actor.module_states] != [
        state.to_dict() for state in reward.module_states
    ]:
        raise SchemaValidationError("teacher actor/reward module states differ")
    if [state.to_dict() for state in actor.object_states] != [
        state.to_dict() for state in reward.object_states
    ]:
        raise SchemaValidationError("teacher actor/reward object states differ")


def _install_order9_policy_references(
    snapshot: ContactWrenchTrajectory,
    *,
    centroidal_reference_pose_world: Sequence[float],
    centroidal_reference_twist: Sequence[float],
    policy_command: PolicyCommand,
) -> None:
    """Replace legacy root-frame labels with the actual pi_H references.

    The Order 8 planner historically stored its base-root waypoint in the
    ``CentroidalTarget`` slot.  The controller first transformed that waypoint
    and the articulated posture into an assembled-morphology centroidal target.
    C0 must archive that transformed target so both pi_H and pi_L supervision
    use the schema's world-centroidal meaning.
    """

    pose = tuple(float(value) for value in centroidal_reference_pose_world)
    twist = tuple(float(value) for value in centroidal_reference_twist)
    if len(pose) != 7 or len(twist) != 6 or not all(
        math.isfinite(value) for value in (*pose, *twist)
    ):
        raise SchemaValidationError(
            "Order9 teacher centroidal reference pose/twist must be finite"
        )
    if not snapshot.knots:
        raise SchemaValidationError("Order9 teacher snapshot has no active knot")
    knot = snapshot.knots[0]
    previous_wrench = (
        None
        if knot.centroidal_target is None
        else knot.centroidal_target.centroidal_wrench_preference
    )
    knot.centroidal_target = CentroidalTarget(
        com_pos_world=(pose[0], pose[1], pose[2]),
        com_vel_world=(twist[0], twist[1], twist[2]),
        body_orientation_world=(pose[3], pose[4], pose[5], pose[6]),
        centroidal_wrench_preference=(
            None if previous_wrench is None else list(previous_wrench)
        ),
    )
    free_anchor_targets = (
        None
        if knot.posture_target is None
        else knot.posture_target.free_anchor_pose_targets
    )
    knot.posture_target = PostureTarget(
        joint_pos_target=dict(policy_command.joint_position_targets),
        joint_vel_target=dict(policy_command.joint_velocity_targets),
        free_anchor_pose_targets=(
            None if free_anchor_targets is None else dict(free_anchor_targets)
        ),
    )
    snapshot.validate()


def _pi_l_teacher_trace(command: PolicyCommand) -> PolicyBehaviorTrace:
    return PolicyBehaviorTrace(
        policy_family="pi_l",
        policy_version=ORDER9_LOW_LEVEL_TEACHER_VERSION,
        action_semantics="deterministic_complete_policy_command_v3",
        action_payload={"policy_command": command.to_dict()},
        stochastic=False,
    )


def _pi_h_teacher_trace(trajectory: ContactWrenchTrajectory) -> PolicyBehaviorTrace:
    return PolicyBehaviorTrace(
        policy_family="pi_h",
        policy_version=ORDER9_NATURAL_CONTACT_TEACHER_VERSION,
        action_semantics="full_contact_wrench_trajectory_v2",
        action_payload={"trajectory": trajectory.to_dict()},
        stochastic=False,
    )


def _decision_positions(
    steps: Sequence[_CompletedStep], *, stride: int
) -> list[int]:
    selected = set(range(0, len(steps), stride))
    selected.add(len(steps) - 1)
    for index in range(1, len(steps)):
        previous = steps[index - 1].actor_observation.task_progress.phase_label
        current = steps[index].actor_observation.task_progress.phase_label
        if current != previous:
            selected.add(index)
    return sorted(selected)


def _stride_positions(count: int, stride: int) -> list[int]:
    return sorted(set(range(0, count, stride)) | {count - 1})


def _discounted_step_returns(
    steps: Sequence[_CompletedStep], *, gamma: float
) -> list[float]:
    values = [0.0] * len(steps)
    running = 0.0
    for index in reversed(range(len(steps))):
        running = float(steps[index].reward) + gamma * running
        values[index] = running
    return values


def _episode_manifest_path(path: Path) -> Path:
    return path / "episode_manifest.json" if path.is_dir() else path


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise SchemaValidationError(f"{path}:{line_number} is not an object")
            yield value


def _atomic_write_jsonl_gzip(path: Path, records: Sequence[SchemaBase]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        with gzip.open(temporary_name, "wt", encoding="utf-8") as handle:
            for record in records:
                record.validate()
                handle.write(record.to_json())
                handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_concatenate_files(path: Path, sources: Sequence[Path]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as target:
            for source in sources:
                with source.open("rb") as handle:
                    shutil.copyfileobj(handle, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_text(path: Path, value: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _single_or_combined(values: Iterable[str]) -> str:
    unique = sorted(set(values))
    if not unique:
        raise SchemaValidationError("Order9 teacher provenance set is empty")
    return unique[0] if len(unique) == 1 else stable_hash(unique)


__all__ = [
    "ORDER9_TEACHER_COLLECTION_VERSION",
    "Order9TeacherCollectionConfig",
    "Order9TeacherEpisodeCollector",
    "Order9TeacherEpisodeManifest",
    "Order9TeacherEpisodeResult",
    "build_order9_teacher_dataset",
    "load_order9_teacher_episode",
    "load_order9_teacher_episode_manifest",
    "load_order9_teacher_low_level_records",
    "validate_order9_teacher_pi_l_representability",
    "write_order9_teacher_episode",
]
