from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from amsrr.schemas.common import Pose7D, SchemaBase, SchemaValidationError, require_len, require_non_empty
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.feasibility import FeasibilityResult
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.policies import POLICY_COMMAND_CONTRACT_CENTROIDAL
from amsrr.schemas.runtime import RuntimeObservation


ORDER3_DATASET_VERSION = "order3_morphology_pi_l_dataset_v1"
ORDER3_CHECKPOINT_VERSION = "order3_morphology_pi_l_checkpoint_v1"
ORDER3_POOL_VERSION = "order3_morphology_pool_v1"
ORDER3_POLICY_FAMILY = "morphology_conditioned_pi_l"
ORDER3_ACTION_SIZE = 12
ORDER3_TENSORIZER_VERSION = "order3_homogeneous_module_graph_tensor_v1"
ORDER3_ENCODER_VERSION = "order3_module_graph_message_passing_v1"
ORDER3_POLICY_ARCHITECTURE_VERSION = "order3_module_graph_gru_actor_critic_v1"
ORDER3_FALLBACK_VERSION = "order3_centroidal_v2_deterministic_hold_v1"
ORDER3_ACTION_NAMES: tuple[str, ...] = (
    "twist_correction.vx",
    "twist_correction.vy",
    "twist_correction.vz",
    "twist_correction.wx",
    "twist_correction.wy",
    "twist_correction.wz",
    "residual_wrench.fx",
    "residual_wrench.fy",
    "residual_wrench.fz",
    "residual_wrench.tx",
    "residual_wrench.ty",
    "residual_wrench.tz",
)


@dataclass
class Order3MorphologyPoolEntry(SchemaBase):
    split: DatasetSplit
    module_count: int
    structural_hash: str
    requested_seed: int
    accepted_proposal_seed: int
    morphology_graph: MorphologyGraph
    feasibility_result: FeasibilityResult
    sampling_metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        require_non_empty(self.structural_hash, "Order3MorphologyPoolEntry.structural_hash")
        if not 2 <= self.module_count <= 8:
            raise SchemaValidationError("Order3MorphologyPoolEntry.module_count must be in [2, 8]")
        if self.module_count != len(self.morphology_graph.modules):
            raise SchemaValidationError(
                "Order3MorphologyPoolEntry.module_count must match morphology_graph"
            )
        if self.requested_seed < 0 or self.accepted_proposal_seed < 0:
            raise SchemaValidationError("Order3MorphologyPoolEntry seeds must be non-negative")
        if not self.feasibility_result.feasible:
            raise SchemaValidationError("Order3MorphologyPoolEntry must be deterministically feasible")


@dataclass
class Order3MorphologyPoolManifest(SchemaBase):
    pool_version: str
    master_seed: int
    physical_model_hash: str
    config_hash: str
    entries: list[Order3MorphologyPoolEntry]
    split_counts: dict[str, int]
    module_count_counts: dict[str, int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.pool_version != ORDER3_POOL_VERSION:
            raise SchemaValidationError(
                f"Order3MorphologyPoolManifest.pool_version must be {ORDER3_POOL_VERSION!r}"
            )
        if self.master_seed < 0:
            raise SchemaValidationError("Order3MorphologyPoolManifest.master_seed must be non-negative")
        require_non_empty(self.physical_model_hash, "Order3MorphologyPoolManifest.physical_model_hash")
        require_non_empty(self.config_hash, "Order3MorphologyPoolManifest.config_hash")
        if not self.entries:
            raise SchemaValidationError("Order3MorphologyPoolManifest.entries must not be empty")
        hashes = [entry.structural_hash for entry in self.entries]
        if len(hashes) != len(set(hashes)):
            raise SchemaValidationError(
                "Order3MorphologyPoolManifest structural hashes must be globally disjoint"
            )
        expected_split_counts = {
            split.value: sum(entry.split == split for entry in self.entries)
            for split in DatasetSplit
        }
        if self.split_counts != expected_split_counts:
            raise SchemaValidationError("Order3MorphologyPoolManifest.split_counts mismatch")
        expected_module_counts = {
            str(module_count): sum(entry.module_count == module_count for entry in self.entries)
            for module_count in range(2, 9)
        }
        if self.module_count_counts != expected_module_counts:
            raise SchemaValidationError("Order3MorphologyPoolManifest.module_count_counts mismatch")


@dataclass
class Order3PolicyTransition(SchemaBase):
    episode_id: str
    split: DatasetSplit
    graph_id: str
    structural_hash: str
    step_index: int
    time_s: float
    runtime_observation: RuntimeObservation
    target_pose_world: Pose7D
    target_twist: list[float]
    previous_action: list[float]
    action: list[float]
    recurrent_state_in: list[float]
    old_log_prob: float
    old_value: float
    reward: float
    terminal: bool
    policy_applied: bool
    privileged_disturbance_body: list[float] = field(default_factory=lambda: [0.0] * 6)
    metrics: dict[str, float] = field(default_factory=dict)
    truncated: bool = False
    bootstrap_value: float | None = None
    behavior_policy_kind: Literal["deterministic_v2_teacher", "order3_checkpoint"] = (
        "deterministic_v2_teacher"
    )
    behavior_policy_version: str = ORDER3_FALLBACK_VERSION
    behavior_checkpoint_hash: str | None = None
    action_semantics: Literal["reference_hold", "learned_residual"] = "reference_hold"
    policy_contract_version: str = POLICY_COMMAND_CONTRACT_CENTROIDAL
    dataset_version: str = ORDER3_DATASET_VERSION

    def validate(self) -> None:
        require_non_empty(self.episode_id, "Order3PolicyTransition.episode_id")
        require_non_empty(self.graph_id, "Order3PolicyTransition.graph_id")
        require_non_empty(self.structural_hash, "Order3PolicyTransition.structural_hash")
        if self.step_index < 0 or self.time_s < 0.0:
            raise SchemaValidationError("Order3PolicyTransition index/time must be non-negative")
        require_len(self.target_pose_world, 7, "Order3PolicyTransition.target_pose_world")
        require_len(self.target_twist, 6, "Order3PolicyTransition.target_twist")
        require_len(self.previous_action, ORDER3_ACTION_SIZE, "Order3PolicyTransition.previous_action")
        require_len(self.action, ORDER3_ACTION_SIZE, "Order3PolicyTransition.action")
        if any(abs(float(value)) > 1.0 for value in self.action):
            raise SchemaValidationError("Order3PolicyTransition.action must be normalized to [-1, 1]")
        if any(abs(float(value)) > 1.0 for value in self.previous_action):
            raise SchemaValidationError(
                "Order3PolicyTransition.previous_action must be normalized to [-1, 1]"
            )
        require_len(
            self.privileged_disturbance_body,
            6,
            "Order3PolicyTransition.privileged_disturbance_body",
        )
        if not self.recurrent_state_in:
            raise SchemaValidationError("Order3PolicyTransition.recurrent_state_in must not be empty")
        values = [
            self.time_s,
            self.old_log_prob,
            self.old_value,
            self.reward,
            *self.target_pose_world,
            *self.target_twist,
            *self.previous_action,
            *self.action,
            *self.recurrent_state_in,
            *self.privileged_disturbance_body,
            *self.metrics.values(),
        ]
        if not all(math.isfinite(float(value)) for value in values):
            raise SchemaValidationError("Order3PolicyTransition numeric values must be finite")
        if self.terminal and self.truncated:
            raise SchemaValidationError(
                "Order3PolicyTransition cannot be both terminal and truncated"
            )
        if self.truncated:
            if self.bootstrap_value is None or not math.isfinite(float(self.bootstrap_value)):
                raise SchemaValidationError(
                    "Order3PolicyTransition truncated rows require a finite bootstrap_value"
                )
        elif self.bootstrap_value is not None:
            raise SchemaValidationError(
                "Order3PolicyTransition bootstrap_value is only valid for truncation"
            )
        require_non_empty(
            self.behavior_policy_version,
            "Order3PolicyTransition.behavior_policy_version",
        )
        if self.behavior_policy_kind == "deterministic_v2_teacher":
            if self.behavior_checkpoint_hash is not None:
                raise SchemaValidationError(
                    "deterministic Order3 teacher rows must not claim a checkpoint hash"
                )
            if self.action_semantics != "reference_hold":
                raise SchemaValidationError(
                    "deterministic Order3 teacher rows require reference_hold actions"
                )
        else:
            require_non_empty(
                self.behavior_checkpoint_hash or "",
                "Order3PolicyTransition.behavior_checkpoint_hash",
            )
            if self.action_semantics != "learned_residual":
                raise SchemaValidationError(
                    "Order3 checkpoint rows require learned_residual actions"
                )
        if self.runtime_observation.morphology_graph.graph_id != self.graph_id:
            raise SchemaValidationError(
                "Order3PolicyTransition graph_id must match RuntimeObservation morphology"
            )
        if self.policy_contract_version != POLICY_COMMAND_CONTRACT_CENTROIDAL:
            raise SchemaValidationError(
                "Order3PolicyTransition requires centroidal_local_joint_v2"
            )
        if self.dataset_version != ORDER3_DATASET_VERSION:
            raise SchemaValidationError(
                f"Order3PolicyTransition.dataset_version must be {ORDER3_DATASET_VERSION!r}"
            )


@dataclass
class Order3DatasetManifest(SchemaBase):
    dataset_version: str
    policy_contract_version: str
    policy_family: str
    pool_hash: str
    physical_model_hash: str
    config_hash: str
    transition_shards: dict[str, list[str]]
    transition_shard_hashes: dict[str, str]
    transition_counts: dict[str, int]
    morphology_hashes: dict[str, list[str]]
    real_isaac_episode_counts: dict[str, int]
    actor_privileged_wrench_inputs: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.dataset_version != ORDER3_DATASET_VERSION:
            raise SchemaValidationError(
                f"Order3DatasetManifest.dataset_version must be {ORDER3_DATASET_VERSION!r}"
            )
        if self.policy_contract_version != POLICY_COMMAND_CONTRACT_CENTROIDAL:
            raise SchemaValidationError(
                "Order3DatasetManifest requires centroidal_local_joint_v2"
            )
        if self.policy_family != ORDER3_POLICY_FAMILY:
            raise SchemaValidationError(
                f"Order3DatasetManifest.policy_family must be {ORDER3_POLICY_FAMILY!r}"
            )
        for name in ("pool_hash", "physical_model_hash", "config_hash"):
            require_non_empty(getattr(self, name), f"Order3DatasetManifest.{name}")
        expected_splits = {split.value for split in DatasetSplit}
        for name, values in (
            ("transition_shards", set(self.transition_shards)),
            ("transition_counts", set(self.transition_counts)),
            ("morphology_hashes", set(self.morphology_hashes)),
            ("real_isaac_episode_counts", set(self.real_isaac_episode_counts)),
        ):
            if values != expected_splits:
                raise SchemaValidationError(f"Order3DatasetManifest.{name} split keys mismatch")
        all_hashes = [
            item
            for split_hashes in self.morphology_hashes.values()
            for item in split_hashes
        ]
        if len(all_hashes) != len(set(all_hashes)):
            raise SchemaValidationError(
                "Order3DatasetManifest morphology hashes must be split-disjoint"
            )
        known_shards = {
            path
            for split_paths in self.transition_shards.values()
            for path in split_paths
        }
        if set(self.transition_shard_hashes) != known_shards:
            raise SchemaValidationError(
                "Order3DatasetManifest.transition_shard_hashes must cover every shard"
            )
        if self.actor_privileged_wrench_inputs:
            raise SchemaValidationError(
                "Order3DatasetManifest forbids privileged wrench inputs to the actor"
            )


@dataclass
class Order3PolicyCheckpointMetadata(SchemaBase):
    checkpoint_version: str
    policy_family: str
    policy_contract_version: str
    architecture_version: str
    tensorizer_version: str
    encoder_version: str
    training_stage: Literal["bc", "ppo", "evaluation"]
    action_names: list[str]
    actor_feature_schema_hash: str
    graph_feature_schema_hash: str
    config_hash: str
    pool_hash: str
    dataset_hash: str
    physical_model_hash: str
    urdf_hash: str
    controller_contract_hash: str
    fallback_version: str
    fallback_config_hash: str
    seed: int
    git_revision: str
    actor_uses_privileged_wrench: bool = False
    outputs_contact_wrench: bool = False
    outputs_internal_wrench: bool = False
    outputs_vectoring_joint_targets: bool = False
    parent_bc_checkpoint_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        expected = {
            "checkpoint_version": ORDER3_CHECKPOINT_VERSION,
            "policy_family": ORDER3_POLICY_FAMILY,
            "policy_contract_version": POLICY_COMMAND_CONTRACT_CENTROIDAL,
            "architecture_version": ORDER3_POLICY_ARCHITECTURE_VERSION,
            "tensorizer_version": ORDER3_TENSORIZER_VERSION,
            "encoder_version": ORDER3_ENCODER_VERSION,
            "fallback_version": ORDER3_FALLBACK_VERSION,
        }
        for field_name, expected_value in expected.items():
            if getattr(self, field_name) != expected_value:
                raise SchemaValidationError(
                    f"Order3PolicyCheckpointMetadata.{field_name} must be {expected_value!r}"
                )
        if tuple(self.action_names) != ORDER3_ACTION_NAMES:
            raise SchemaValidationError(
                "Order3PolicyCheckpointMetadata.action_names do not match the v1 action contract"
            )
        for field_name in (
            "actor_feature_schema_hash",
            "graph_feature_schema_hash",
            "config_hash",
            "pool_hash",
            "dataset_hash",
            "physical_model_hash",
            "urdf_hash",
            "controller_contract_hash",
            "fallback_config_hash",
            "git_revision",
        ):
            require_non_empty(
                getattr(self, field_name),
                f"Order3PolicyCheckpointMetadata.{field_name}",
            )
        if self.seed < 0:
            raise SchemaValidationError(
                "Order3PolicyCheckpointMetadata.seed must be non-negative"
            )
        if any(
            (
                self.actor_uses_privileged_wrench,
                self.outputs_contact_wrench,
                self.outputs_internal_wrench,
                self.outputs_vectoring_joint_targets,
            )
        ):
            raise SchemaValidationError(
                "Order3 checkpoint violates the actor/controller authority boundary"
            )
        if self.training_stage == "ppo" and not self.parent_bc_checkpoint_hash:
            raise SchemaValidationError(
                "Order3 PPO checkpoint requires parent_bc_checkpoint_hash"
            )
        if self.parent_bc_checkpoint_hash is not None:
            require_non_empty(
                self.parent_bc_checkpoint_hash,
                "Order3PolicyCheckpointMetadata.parent_bc_checkpoint_hash",
            )
