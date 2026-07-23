from __future__ import annotations

"""Explicit, hash-bound C2-v2 to C3 active-knot ``pi_L`` transformation."""

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from amsrr.policies.order9_active_knot_features import (
    ORDER9_ACTIVE_ASSIGNMENT_FEATURE_NAMES,
    ORDER9_ACTIVE_KNOT_FEATURE_CONTRACT_VERSION,
    ORDER9_ACTIVE_KNOT_GLOBAL_FEATURE_NAMES,
)
from amsrr.policies.order9_low_level_policy import (
    ORDER9_ACTIVE_KNOT_PI_L_POLICY_VERSION,
    ORDER9_PI_L_POLICY_VERSION,
    Order9ActiveKnotLowLevelPolicyConfig,
    Order9ActiveKnotPhaseConditionedActorCritic,
    Order9PhaseConditionedActorCritic,
)
from amsrr.schemas.common import (
    SchemaBase,
    SchemaValidationError,
    require_non_empty,
)
from amsrr.schemas.order9 import Order9PolicyFamily
from amsrr.training.order9_checkpoints import (
    load_order9_policy_checkpoint,
    save_order9_policy_checkpoint,
)
from amsrr.training.order9_curriculum import (
    ORDER9_CURRICULUM_VERSION,
    load_order9_learning_config,
)
from amsrr.training.order9_curriculum_lineage import (
    DEFAULT_ORDER9_CURRICULUM_LINEAGE_IMPORT_PATH,
    validate_order9_curriculum_lineage_import,
)
from amsrr.training.order9_offline_training import (
    build_order9_checkpoint_metadata,
)
from amsrr.training.order9_pipeline import (
    order9_schedule_hash,
    order9_stage_by_id,
)
from amsrr.utils.hashing import hash_file


ORDER9_PI_L_ACTIVE_KNOT_MIGRATION_VERSION = (
    "order9_pi_l_c2_v2_to_c3_active_knot_v1"
)
ORDER9_C3_STAGE_ID = "c3_pi_l_ppo_arbitrary_morphology"
_ZERO_RESIDUAL_PARAMETERS = (
    "active_knot_encoder.2.weight",
    "active_knot_encoder.2.bias",
    "active_assignment_encoder.2.weight",
    "active_assignment_encoder.2.bias",
)


@dataclass
class Order9PiLActiveKnotMigrationManifest(SchemaBase):
    migration_version: str
    source_checkpoint_path: str
    source_checkpoint_sha256: str
    source_policy_version: str
    source_schedule_hash: str
    lineage_import_path: str
    lineage_import_sha256: str
    source_promotion_manifest_path: str
    source_promotion_manifest_sha256: str
    target_checkpoint_path: str
    target_checkpoint_sha256: str
    target_policy_version: str
    target_schedule_hash: str
    target_stage_id: str
    target_stage_index: int
    active_knot_feature_contract_version: str
    active_knot_global_feature_names: list[str]
    active_assignment_feature_names: list[str]
    copied_parameter_names: list[str]
    fresh_parameter_names: list[str]
    zero_residual_parameter_names: list[str]
    maximum_copied_parameter_error: float
    behavior_preserving_initialization: bool
    promoted_checkpoint: bool
    metadata: dict[str, Any]

    def validate(self) -> None:
        if self.migration_version != ORDER9_PI_L_ACTIVE_KNOT_MIGRATION_VERSION:
            raise SchemaValidationError("Order9 active-knot migration version mismatch")
        for name in (
            "source_checkpoint_path",
            "lineage_import_path",
            "source_promotion_manifest_path",
            "target_checkpoint_path",
            "target_stage_id",
        ):
            require_non_empty(
                str(getattr(self, name)),
                f"Order9PiLActiveKnotMigrationManifest.{name}",
            )
        for name in (
            "source_checkpoint_sha256",
            "source_schedule_hash",
            "lineage_import_sha256",
            "source_promotion_manifest_sha256",
            "target_checkpoint_sha256",
            "target_schedule_hash",
        ):
            _require_sha256(
                str(getattr(self, name)),
                f"Order9PiLActiveKnotMigrationManifest.{name}",
            )
        if (
            self.source_policy_version != ORDER9_PI_L_POLICY_VERSION
            or self.target_policy_version
            != ORDER9_ACTIVE_KNOT_PI_L_POLICY_VERSION
            or self.target_stage_id != ORDER9_C3_STAGE_ID
            or self.target_stage_index != 3
        ):
            raise SchemaValidationError(
                "Order9 active-knot migration policy/stage boundary differs"
            )
        if (
            self.active_knot_feature_contract_version
            != ORDER9_ACTIVE_KNOT_FEATURE_CONTRACT_VERSION
            or tuple(self.active_knot_global_feature_names)
            != ORDER9_ACTIVE_KNOT_GLOBAL_FEATURE_NAMES
            or tuple(self.active_assignment_feature_names)
            != ORDER9_ACTIVE_ASSIGNMENT_FEATURE_NAMES
        ):
            raise SchemaValidationError(
                "Order9 active-knot migration feature contract differs"
            )
        if (
            not self.copied_parameter_names
            or not self.fresh_parameter_names
            or len(self.copied_parameter_names)
            != len(set(self.copied_parameter_names))
            or len(self.fresh_parameter_names)
            != len(set(self.fresh_parameter_names))
            or set(self.copied_parameter_names) & set(self.fresh_parameter_names)
        ):
            raise SchemaValidationError(
                "Order9 active-knot migration parameter partition is invalid"
            )
        if tuple(self.zero_residual_parameter_names) != _ZERO_RESIDUAL_PARAMETERS:
            raise SchemaValidationError(
                "Order9 active-knot migration zero-residual boundary differs"
            )
        if (
            not math.isfinite(float(self.maximum_copied_parameter_error))
            or self.maximum_copied_parameter_error != 0.0
            or not self.behavior_preserving_initialization
            or self.promoted_checkpoint
        ):
            raise SchemaValidationError(
                "Order9 active-knot initializer must be exact and unpromoted"
            )


@dataclass(frozen=True)
class PreparedOrder9PiLActiveKnotInitializer:
    checkpoint_path: Path
    checkpoint_sha256: str
    manifest_path: Path
    manifest_sha256: str
    manifest: Order9PiLActiveKnotMigrationManifest


def prepare_order9_pi_l_active_knot_initializer(
    *,
    config_path: str | Path,
    output_checkpoint_path: str | Path,
    output_manifest_path: str | Path,
    git_revision: str,
    lineage_import_path: str | Path = (
        DEFAULT_ORDER9_CURRICULUM_LINEAGE_IMPORT_PATH
    ),
    device: str | torch.device = "cpu",
) -> PreparedOrder9PiLActiveKnotInitializer:
    require_non_empty(git_revision, "git_revision")
    config_path = Path(config_path).resolve()
    lineage_path = Path(lineage_import_path).resolve()
    checkpoint_path = Path(output_checkpoint_path).resolve()
    manifest_path = Path(output_manifest_path).resolve()
    for path in (checkpoint_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"Order9 C3 initializer output exists: {path}")
    config = load_order9_learning_config(config_path)
    if config.curriculum.schedule_version != ORDER9_CURRICULUM_VERSION:
        raise SchemaValidationError(
            "Order9 active-knot initializer requires the v3 curriculum"
        )
    stage = order9_stage_by_id(config, ORDER9_C3_STAGE_ID)
    lineage = validate_order9_curriculum_lineage_import(
        lineage_path,
        successor_config=config,
        load_checkpoint=True,
        device=device,
    )
    source = load_order9_policy_checkpoint(
        lineage.policy_checkpoint_path,
        device=device,
        expected_sha256=lineage.specification.policy_checkpoint_sha256,
        expected_family=Order9PolicyFamily.PI_L,
        expected_schedule_hash=lineage.specification.source_schedule_hash,
    )
    if (
        source.metadata.policy_version != ORDER9_PI_L_POLICY_VERSION
        or not isinstance(source.model, Order9PhaseConditionedActorCritic)
        or isinstance(
            source.model, Order9ActiveKnotPhaseConditionedActorCritic
        )
    ):
        raise SchemaValidationError(
            "Order9 active-knot migration source is not the promoted legacy C2 actor"
        )
    config_values = source.model_config.to_dict()
    target_config = Order9ActiveKnotLowLevelPolicyConfig.from_dict(
        config_values
    )
    target = Order9ActiveKnotPhaseConditionedActorCritic(target_config).to(
        device
    )
    fresh_names, unexpected = target.initialize_from_legacy_order9(source.model)
    if unexpected:
        raise AssertionError("validated active-knot migration has unexpected keys")
    source_state = source.model.state_dict()
    target_state = target.state_dict()
    copied_names = sorted(source_state)
    maximum_error = max(
        (
            float(
                (
                    target_state[name].detach().to("cpu")
                    - source_state[name].detach().to("cpu")
                )
                .abs()
                .max()
                .item()
            )
            for name in copied_names
        ),
        default=0.0,
    )
    if maximum_error != 0.0:
        raise SchemaValidationError(
            "Order9 active-knot migration did not copy legacy parameters exactly"
        )
    for name in _ZERO_RESIDUAL_PARAMETERS:
        if name not in target_state or bool((target_state[name] != 0.0).any()):
            raise SchemaValidationError(
                "Order9 active-knot migration residual branch is not zero"
            )
    schedule_hash = order9_schedule_hash(config)
    metadata = build_order9_checkpoint_metadata(
        target,
        stage=stage,
        schedule_hash=schedule_hash,
        physical_model_hash=source.metadata.physical_model_hash,
        git_revision=git_revision,
        random_seed=config.production_runtime.seed + stage.stage_index * 100_000,
        input_artifact_hashes={
            "legacy_c2_checkpoint": source.sha256,
            "curriculum_lineage_import": hash_file(lineage_path),
            "legacy_c2_promotion_manifest": (
                lineage.specification.promoted_stage_manifest_sha256
            ),
        },
        parent_checkpoint_sha256=source.sha256,
        source_order3_checkpoint_sha256=(
            source.metadata.source_order3_checkpoint_sha256
        ),
        metrics={
            "maximum_copied_parameter_error": maximum_error,
            "copied_parameter_count": float(len(copied_names)),
            "fresh_parameter_count": float(len(fresh_names)),
        },
        trainer_version=ORDER9_PI_L_ACTIVE_KNOT_MIGRATION_VERSION,
        extra_metadata={
            "ppo_update_index": -1,
            "initializer_only": True,
            "promoted_checkpoint": False,
            "source_policy_version": source.metadata.policy_version,
            "source_schedule_hash": source.metadata.curriculum_schedule_hash,
            "active_knot_feature_contract_version": (
                ORDER9_ACTIVE_KNOT_FEATURE_CONTRACT_VERSION
            ),
            "active_knot_global_feature_names": list(
                ORDER9_ACTIVE_KNOT_GLOBAL_FEATURE_NAMES
            ),
            "active_assignment_feature_names": list(
                ORDER9_ACTIVE_ASSIGNMENT_FEATURE_NAMES
            ),
            "zero_residual_parameter_names": list(
                _ZERO_RESIDUAL_PARAMETERS
            ),
        },
    )
    target_sha = save_order9_policy_checkpoint(
        checkpoint_path, model=target, metadata=metadata
    )
    loaded_target = load_order9_policy_checkpoint(
        checkpoint_path,
        device=device,
        expected_sha256=target_sha,
        expected_family=Order9PolicyFamily.PI_L,
        expected_schedule_hash=schedule_hash,
    )
    if not isinstance(
        loaded_target.model, Order9ActiveKnotPhaseConditionedActorCritic
    ):
        raise SchemaValidationError(
            "Order9 active-knot target checkpoint reconstructed wrong model"
        )
    manifest = Order9PiLActiveKnotMigrationManifest(
        migration_version=ORDER9_PI_L_ACTIVE_KNOT_MIGRATION_VERSION,
        source_checkpoint_path=str(lineage.policy_checkpoint_path),
        source_checkpoint_sha256=source.sha256,
        source_policy_version=source.metadata.policy_version,
        source_schedule_hash=source.metadata.curriculum_schedule_hash,
        lineage_import_path=str(lineage_path),
        lineage_import_sha256=hash_file(lineage_path),
        source_promotion_manifest_path=str(
            lineage.promoted_stage_manifest_path
        ),
        source_promotion_manifest_sha256=(
            lineage.specification.promoted_stage_manifest_sha256
        ),
        target_checkpoint_path=str(checkpoint_path),
        target_checkpoint_sha256=target_sha,
        target_policy_version=loaded_target.metadata.policy_version,
        target_schedule_hash=schedule_hash,
        target_stage_id=stage.stage_id,
        target_stage_index=stage.stage_index,
        active_knot_feature_contract_version=(
            ORDER9_ACTIVE_KNOT_FEATURE_CONTRACT_VERSION
        ),
        active_knot_global_feature_names=list(
            ORDER9_ACTIVE_KNOT_GLOBAL_FEATURE_NAMES
        ),
        active_assignment_feature_names=list(
            ORDER9_ACTIVE_ASSIGNMENT_FEATURE_NAMES
        ),
        copied_parameter_names=copied_names,
        fresh_parameter_names=sorted(fresh_names),
        zero_residual_parameter_names=list(_ZERO_RESIDUAL_PARAMETERS),
        maximum_copied_parameter_error=maximum_error,
        behavior_preserving_initialization=True,
        promoted_checkpoint=False,
        metadata={
            "source_state_dict_hash": source.metadata.state_dict_hash,
            "target_state_dict_hash": loaded_target.metadata.state_dict_hash,
            "transformation": (
                "copy every legacy parameter exactly; append zero-output "
                "active-knot global/node residual encoders"
            ),
        },
    )
    _atomic_write_json(manifest_path, manifest.to_dict())
    validate_order9_pi_l_active_knot_migration(
        manifest_path,
        expected_config_path=config_path,
        device=device,
    )
    return PreparedOrder9PiLActiveKnotInitializer(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=target_sha,
        manifest_path=manifest_path,
        manifest_sha256=hash_file(manifest_path),
        manifest=manifest,
    )


def validate_order9_pi_l_active_knot_migration(
    manifest_path: str | Path,
    *,
    expected_config_path: str | Path,
    device: str | torch.device = "cpu",
) -> Order9PiLActiveKnotMigrationManifest:
    path = Path(manifest_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = Order9PiLActiveKnotMigrationManifest.from_dict(payload)
    config = load_order9_learning_config(expected_config_path)
    if manifest.target_schedule_hash != order9_schedule_hash(config):
        raise SchemaValidationError(
            "Order9 active-knot migration target schedule hash differs"
        )
    if hash_file(manifest.lineage_import_path) != manifest.lineage_import_sha256:
        raise SchemaValidationError(
            "Order9 active-knot migration lineage bytes differ"
        )
    if (
        hash_file(manifest.source_promotion_manifest_path)
        != manifest.source_promotion_manifest_sha256
    ):
        raise SchemaValidationError(
            "Order9 active-knot migration source promotion bytes differ"
        )
    target = load_order9_policy_checkpoint(
        manifest.target_checkpoint_path,
        device=device,
        expected_sha256=manifest.target_checkpoint_sha256,
        expected_family=Order9PolicyFamily.PI_L,
        expected_schedule_hash=manifest.target_schedule_hash,
    )
    if (
        target.metadata.curriculum_stage_id != manifest.target_stage_id
        or target.metadata.curriculum_stage_index
        != manifest.target_stage_index
        or target.metadata.parent_checkpoint_sha256
        != manifest.source_checkpoint_sha256
        or target.metadata.metadata.get("initializer_only") is not True
        or target.metadata.metadata.get("promoted_checkpoint") is not False
    ):
        raise SchemaValidationError(
            "Order9 active-knot initializer checkpoint lineage differs"
        )
    return manifest


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _require_sha256(value: str, path: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise SchemaValidationError(f"{path} must be a lowercase SHA-256")


__all__ = [
    "ORDER9_C3_STAGE_ID",
    "ORDER9_PI_L_ACTIVE_KNOT_MIGRATION_VERSION",
    "Order9PiLActiveKnotMigrationManifest",
    "PreparedOrder9PiLActiveKnotInitializer",
    "prepare_order9_pi_l_active_knot_initializer",
    "validate_order9_pi_l_active_knot_migration",
]
