from __future__ import annotations

"""Hash-bound v4-to-v5 C3 contact-residual policy migration."""

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

from amsrr.policies.order9_low_level_policy import (
    ORDER9_ACTIVE_KNOT_PI_L_POLICY_VERSION,
    ORDER9_CONTACT_RESIDUAL_PI_L_POLICY_VERSION,
    Order9ActiveKnotLowLevelPolicyConfig,
    Order9ActiveKnotPhaseConditionedActorCritic,
    Order9ContactResidualPhaseConditionedActorCritic,
)
from amsrr.schemas.common import SchemaBase, SchemaValidationError
from amsrr.schemas.order9 import Order9PolicyFamily
from amsrr.training.order9_checkpoints import (
    load_order9_policy_checkpoint,
    save_order9_policy_checkpoint,
)
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_offline_training import (
    build_order9_checkpoint_metadata,
)
from amsrr.training.order9_pipeline import (
    order9_schedule_hash,
    order9_stage_by_id,
)
from amsrr.utils.hashing import hash_file


ORDER9_PI_L_CONTACT_RESIDUAL_MIGRATION_VERSION = (
    "order9_pi_l_active_knot_v4_to_contact_residual_v5_initializer_aware_v2"
)
ORDER9_C3_STAGE_ID = "c3_pi_l_ppo_arbitrary_morphology"
_ZERO_OUTPUT_PARAMETERS = (
    "contact_residual_decoder.2.weight",
    "contact_residual_decoder.2.bias",
)


@dataclass
class Order9PiLContactResidualMigrationManifest(SchemaBase):
    migration_version: str
    source_checkpoint_path: str
    source_checkpoint_sha256: str
    source_policy_version: str
    source_update_index: int
    target_checkpoint_path: str
    target_checkpoint_sha256: str
    target_policy_version: str
    target_update_index: int
    schedule_hash: str
    copied_parameter_names: list[str]
    fresh_parameter_names: list[str]
    zero_output_parameter_names: list[str]
    maximum_copied_parameter_error: float
    behavior_preserving_initialization: bool

    def validate(self) -> None:
        if (
            self.migration_version
            != ORDER9_PI_L_CONTACT_RESIDUAL_MIGRATION_VERSION
            or self.source_policy_version
            != ORDER9_ACTIVE_KNOT_PI_L_POLICY_VERSION
            or self.target_policy_version
            != ORDER9_CONTACT_RESIDUAL_PI_L_POLICY_VERSION
            or self.source_update_index != self.target_update_index
        ):
            raise SchemaValidationError(
                "Order9 contact-residual migration identity differs"
            )
        for value in (
            self.source_checkpoint_sha256,
            self.target_checkpoint_sha256,
            self.schedule_hash,
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise SchemaValidationError(
                    "Order9 contact-residual migration SHA-256 is invalid"
                )
        if (
            not self.copied_parameter_names
            or not self.fresh_parameter_names
            or set(self.copied_parameter_names) & set(self.fresh_parameter_names)
            or tuple(self.zero_output_parameter_names)
            != _ZERO_OUTPUT_PARAMETERS
            or not math.isfinite(self.maximum_copied_parameter_error)
            or self.maximum_copied_parameter_error != 0.0
            or not self.behavior_preserving_initialization
        ):
            raise SchemaValidationError(
                "Order9 contact-residual migration is not behavior preserving"
            )


def prepare_order9_pi_l_contact_residual_initializer(
    *,
    config_path: str | Path,
    source_checkpoint_path: str | Path,
    output_checkpoint_path: str | Path,
    output_manifest_path: str | Path,
    git_revision: str,
    device: str | torch.device = "cpu",
) -> Order9PiLContactResidualMigrationManifest:
    config = load_order9_learning_config(config_path)
    stage = order9_stage_by_id(config, ORDER9_C3_STAGE_ID)
    schedule_hash = order9_schedule_hash(config)
    source = load_order9_policy_checkpoint(
        source_checkpoint_path,
        device=device,
        expected_family=Order9PolicyFamily.PI_L,
        expected_schedule_hash=schedule_hash,
    )
    if (
        source.metadata.policy_version
        != ORDER9_ACTIVE_KNOT_PI_L_POLICY_VERSION
        or type(source.model)
        is not Order9ActiveKnotPhaseConditionedActorCritic
    ):
        raise SchemaValidationError(
            "Order9 contact-residual source must be the exact v4 active-knot actor"
        )
    source_update = source.metadata.metadata.get("ppo_update_index")
    if not isinstance(source_update, int) or source_update < -1:
        raise SchemaValidationError(
            "Order9 contact-residual source lacks a PPO update index"
        )
    if source_update == -1 and not bool(
        source.metadata.metadata.get("initializer_only", False)
    ):
        raise SchemaValidationError(
            "Order9 contact-residual update -1 source is not an initializer"
        )
    checkpoint_path = Path(output_checkpoint_path).resolve()
    manifest_path = Path(output_manifest_path).resolve()
    for path in (checkpoint_path, manifest_path):
        if path.exists():
            raise FileExistsError(path)

    target_config = Order9ActiveKnotLowLevelPolicyConfig.from_dict(
        source.model_config.to_dict()
    )
    target = Order9ContactResidualPhaseConditionedActorCritic(
        target_config
    ).to(device)
    fresh, unexpected = target.initialize_from_active_knot(source.model)
    if unexpected:
        raise AssertionError("validated contact-residual migration has extras")
    source_state = source.model.state_dict()
    target_state = target.state_dict()
    copied = sorted(source_state)
    maximum_error = max(
        (
            float(
                (
                    target_state[name].detach().cpu()
                    - source_state[name].detach().cpu()
                )
                .abs()
                .max()
                .item()
            )
            for name in copied
        ),
        default=0.0,
    )
    for name in _ZERO_OUTPUT_PARAMETERS:
        if name not in target_state or bool((target_state[name] != 0.0).any()):
            raise SchemaValidationError(
                "Order9 contact-residual branch is not zero-output initialized"
            )
    prior_rollouts = source.metadata.metadata.get(
        "consumed_rollout_manifest_sha256s", []
    )
    if not isinstance(prior_rollouts, list):
        raise SchemaValidationError(
            "Order9 contact-residual source rollout lineage is invalid"
        )
    metadata = build_order9_checkpoint_metadata(
        target,
        stage=stage,
        schedule_hash=schedule_hash,
        physical_model_hash=source.metadata.physical_model_hash,
        git_revision=git_revision,
        random_seed=config.production_runtime.seed + stage.stage_index * 100_000,
        input_artifact_hashes={"source_v4_checkpoint": source.sha256},
        parent_checkpoint_sha256=source.sha256,
        source_order3_checkpoint_sha256=(
            source.metadata.source_order3_checkpoint_sha256
        ),
        metrics={
            "maximum_copied_parameter_error": maximum_error,
            "copied_parameter_count": float(len(copied)),
            "fresh_parameter_count": float(len(fresh)),
        },
        trainer_version=ORDER9_PI_L_CONTACT_RESIDUAL_MIGRATION_VERSION,
        extra_metadata={
            "ppo_update_index": source_update,
            "consumed_rollout_manifest_sha256s": list(prior_rollouts),
            "behavior_preserving_migration": True,
            "source_policy_version": source.metadata.policy_version,
            "zero_output_parameter_names": list(_ZERO_OUTPUT_PARAMETERS),
        },
    )
    target_sha = save_order9_policy_checkpoint(
        checkpoint_path, model=target, metadata=metadata
    )
    loaded = load_order9_policy_checkpoint(
        checkpoint_path,
        device=device,
        expected_sha256=target_sha,
        expected_family=Order9PolicyFamily.PI_L,
        expected_schedule_hash=schedule_hash,
    )
    if type(loaded.model) is not Order9ContactResidualPhaseConditionedActorCritic:
        raise SchemaValidationError(
            "Order9 contact-residual checkpoint reconstructed wrong model"
        )
    manifest = Order9PiLContactResidualMigrationManifest(
        migration_version=ORDER9_PI_L_CONTACT_RESIDUAL_MIGRATION_VERSION,
        source_checkpoint_path=str(Path(source_checkpoint_path).resolve()),
        source_checkpoint_sha256=source.sha256,
        source_policy_version=source.metadata.policy_version,
        source_update_index=source_update,
        target_checkpoint_path=str(checkpoint_path),
        target_checkpoint_sha256=target_sha,
        target_policy_version=loaded.metadata.policy_version,
        target_update_index=source_update,
        schedule_hash=schedule_hash,
        copied_parameter_names=copied,
        fresh_parameter_names=sorted(fresh),
        zero_output_parameter_names=list(_ZERO_OUTPUT_PARAMETERS),
        maximum_copied_parameter_error=maximum_error,
        behavior_preserving_initialization=True,
    )
    manifest.validate()
    _atomic_write_json(manifest_path, manifest.to_dict())
    if hash_file(checkpoint_path) != manifest.target_checkpoint_sha256:
        raise SchemaValidationError(
            "Order9 contact-residual checkpoint bytes changed after migration"
        )
    return manifest


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
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


__all__ = [
    "ORDER9_PI_L_CONTACT_RESIDUAL_MIGRATION_VERSION",
    "Order9PiLContactResidualMigrationManifest",
    "prepare_order9_pi_l_contact_residual_initializer",
]
