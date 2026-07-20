from __future__ import annotations

"""Persisted lineage and stage-run contracts for Order 9 learning."""

import math
from dataclasses import dataclass, field
from typing import Any

from amsrr.schemas.common import (
    SchemaBase,
    SchemaValidationError,
    StrEnum,
    require_non_empty,
)
from amsrr.schemas.policies import CONTACT_WRENCH_CONTRACT_CONTACT_FRAME


ORDER9_POLICY_CHECKPOINT_VERSION = "order9_policy_checkpoint_v1"
ORDER9_STAGE_RUN_VERSION = "order9_stage_run_v1"


class Order9PolicyFamily(StrEnum):
    PI_L = "pi_l"
    PI_H = "pi_h"
    PI_D = "pi_d"


class Order9StageRunStatus(StrEnum):
    PREPARED = "prepared"
    RUNNING = "running"
    EVALUATED = "evaluated"
    PROMOTED = "promoted"
    REJECTED = "rejected"


@dataclass
class Order9PolicyCheckpointMetadata(SchemaBase):
    checkpoint_version: str
    policy_family: Order9PolicyFamily
    policy_version: str
    curriculum_schedule_hash: str
    curriculum_stage_id: str
    curriculum_stage_index: int
    learning_mode: str
    model_config_hash: str
    state_dict_hash: str
    physical_model_hash: str
    actor_observation_contract: str
    critic_observation_contract: str
    action_contract: str
    git_revision: str
    random_seed: int
    input_artifact_hashes: dict[str, str]
    parent_checkpoint_sha256: str | None = None
    source_order3_checkpoint_sha256: str | None = None
    contact_wrench_contract_version: str = CONTACT_WRENCH_CONTRACT_CONTACT_FRAME
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.checkpoint_version != ORDER9_POLICY_CHECKPOINT_VERSION:
            raise SchemaValidationError(
                "Order9PolicyCheckpointMetadata checkpoint version mismatch"
            )
        for name in (
            "policy_version",
            "curriculum_stage_id",
            "learning_mode",
            "actor_observation_contract",
            "critic_observation_contract",
            "action_contract",
            "git_revision",
        ):
            require_non_empty(
                str(getattr(self, name)), f"Order9PolicyCheckpointMetadata.{name}"
            )
        if self.curriculum_stage_index < 0 or self.random_seed < 0:
            raise SchemaValidationError(
                "Order9 checkpoint stage index and random seed must be non-negative"
            )
        for name in (
            "curriculum_schedule_hash",
            "model_config_hash",
            "state_dict_hash",
            "physical_model_hash",
        ):
            _require_sha256(
                str(getattr(self, name)), f"Order9PolicyCheckpointMetadata.{name}"
            )
        for name in (
            "parent_checkpoint_sha256",
            "source_order3_checkpoint_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_sha256(value, f"Order9PolicyCheckpointMetadata.{name}")
        if self.contact_wrench_contract_version != CONTACT_WRENCH_CONTRACT_CONTACT_FRAME:
            raise SchemaValidationError(
                "Order9 checkpoints require contact_frame_robot_on_target_v2"
            )
        if not self.input_artifact_hashes:
            raise SchemaValidationError(
                "Order9 checkpoints must bind at least one input artifact"
            )
        for key, value in self.input_artifact_hashes.items():
            require_non_empty(key, "Order9PolicyCheckpointMetadata.input_artifact_hashes.key")
            _require_sha256(
                value,
                f"Order9PolicyCheckpointMetadata.input_artifact_hashes[{key!r}]",
            )
        for key, value in self.metrics.items():
            require_non_empty(key, "Order9PolicyCheckpointMetadata.metrics.key")
            if not math.isfinite(float(value)):
                raise SchemaValidationError(
                    f"Order9 checkpoint metric {key!r} must be finite"
                )


@dataclass
class Order9ArtifactBinding(SchemaBase):
    artifact_kind: str
    path: str
    sha256: str

    def validate(self) -> None:
        require_non_empty(self.artifact_kind, "Order9ArtifactBinding.artifact_kind")
        require_non_empty(self.path, "Order9ArtifactBinding.path")
        _require_sha256(self.sha256, "Order9ArtifactBinding.sha256")


@dataclass
class Order9StageRunManifest(SchemaBase):
    run_version: str
    run_id: str
    stage_id: str
    stage_index: int
    status: Order9StageRunStatus
    schedule_hash: str
    stage_config_hash: str
    runtime_config_hash: str
    random_seed: int
    device: str
    environment_count: int
    input_artifacts: list[Order9ArtifactBinding]
    output_artifacts: list[Order9ArtifactBinding] = field(default_factory=list)
    policy_checkpoint_sha256_by_family: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    promotion_failed_gates: list[str] = field(default_factory=list)
    promoted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.run_version != ORDER9_STAGE_RUN_VERSION:
            raise SchemaValidationError("Order9 stage-run version mismatch")
        for name in ("run_id", "stage_id", "device"):
            require_non_empty(str(getattr(self, name)), f"Order9StageRunManifest.{name}")
        if self.stage_index < 0 or self.random_seed < 0:
            raise SchemaValidationError(
                "Order9 stage index and random seed must be non-negative"
            )
        if self.environment_count < 1:
            raise SchemaValidationError(
                "Order9 stage environment_count must be positive"
            )
        for name in ("schedule_hash", "stage_config_hash", "runtime_config_hash"):
            _require_sha256(str(getattr(self, name)), f"Order9StageRunManifest.{name}")
        paths = [item.path for item in (*self.input_artifacts, *self.output_artifacts)]
        if len(paths) != len(set(paths)):
            raise SchemaValidationError(
                "Order9 stage-run artifact paths must be unique"
            )
        for family, digest in self.policy_checkpoint_sha256_by_family.items():
            if family not in Order9PolicyFamily.values():
                raise SchemaValidationError(
                    f"Order9 stage-run has unknown policy family {family!r}"
                )
            _require_sha256(
                digest,
                f"Order9StageRunManifest.policy_checkpoint_sha256_by_family[{family!r}]",
            )
        if len(self.promotion_failed_gates) != len(set(self.promotion_failed_gates)):
            raise SchemaValidationError(
                "Order9 promotion failed-gate names must be unique"
            )
        if self.promoted != (self.status == Order9StageRunStatus.PROMOTED):
            raise SchemaValidationError(
                "Order9 stage-run promoted flag must match promoted status"
            )
        if self.promoted and self.promotion_failed_gates:
            raise SchemaValidationError(
                "A promoted Order9 stage cannot retain failed gates"
            )
        if self.status == Order9StageRunStatus.REJECTED and not self.promotion_failed_gates:
            raise SchemaValidationError(
                "A rejected Order9 stage must record failed gates"
            )
        for key, value in self.metrics.items():
            require_non_empty(key, "Order9StageRunManifest.metrics.key")
            if not math.isfinite(float(value)):
                raise SchemaValidationError(
                    f"Order9 stage metric {key!r} must be finite"
                )


def _require_sha256(value: str, path: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise SchemaValidationError(f"{path} must be a lowercase SHA-256 digest")
