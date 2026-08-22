from __future__ import annotations

"""Hash-bound v6-to-v7 task-generic contact-space initializer migration."""

import json
import os
import tempfile
from pathlib import Path

import torch

from amsrr.policies.order9_low_level_policy import (
    ORDER9_CONTACT_SPACE_PI_L_POLICY_VERSION,
    ORDER9_MORPHOLOGY_INVARIANT_COMPRESSION_PI_L_POLICY_VERSION,
    Order9ActiveKnotLowLevelPolicyConfig,
    Order9ContactSpacePhaseConditionedActorCritic,
    Order9MorphologyInvariantCompressionActorCritic,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.order9 import Order9PolicyFamily
from amsrr.training.order9_checkpoints import (
    load_order9_policy_checkpoint,
    save_order9_policy_checkpoint,
)
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_offline_training import build_order9_checkpoint_metadata
from amsrr.training.order9_pipeline import order9_schedule_hash, order9_stage_by_id


ORDER9_PI_L_CONTACT_SPACE_MIGRATION_VERSION = (
    "order9_pi_l_v6_to_contact_space_projected_v7_v2_zero_command_boundary"
)
ORDER9_C3_STAGE_ID = "c3_pi_l_ppo_arbitrary_morphology"


# These v6 mean layers existed in the checkpoint, but their commands were
# masked by the compression-only execution contract.  Copying their non-zero
# values into v7 would therefore activate unvalidated CoM and per-joint
# commands at the migration boundary.  They are representation-compatible,
# but not command-compatible, and must start as fresh residual heads.
_NEWLY_ACTIVATED_MEAN_PARAMETER_NAMES = (
    "actor_mean.weight",
    "actor_mean.bias",
    "joint_decoder.2.weight",
    "joint_decoder.2.bias",
)
_NEWLY_ACTIVATED_LOG_STD_PARAMETER_NAMES = (
    "actor_log_std",
    "joint_actor_log_std",
)


def _reset_newly_activated_v7_action_outputs(
    target: Order9ContactSpacePhaseConditionedActorCritic,
) -> tuple[str, ...]:
    """Make v6-to-v7 migration command-preserving at zero residual action."""

    state = target.state_dict()
    required = (
        *_NEWLY_ACTIVATED_MEAN_PARAMETER_NAMES,
        *_NEWLY_ACTIVATED_LOG_STD_PARAMETER_NAMES,
    )
    missing = sorted(set(required) - set(state))
    if missing:
        raise SchemaValidationError(
            f"Order9 v7 newly activated action parameters are missing: {missing}"
        )
    with torch.no_grad():
        for name in _NEWLY_ACTIVATED_MEAN_PARAMETER_NAMES:
            state[name].zero_()
        state["actor_log_std"].fill_(float(target.config.joint_action_log_std_init))
        state["joint_actor_log_std"].fill_(
            float(target.config.joint_action_log_std_init)
        )
    return required


def prepare_order9_pi_l_contact_space_initializer(
    *,
    config_path: str | Path,
    source_checkpoint_path: str | Path,
    output_checkpoint_path: str | Path,
    output_manifest_path: str | Path,
    git_revision: str,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    config = load_order9_learning_config(config_path)
    stage = order9_stage_by_id(config, ORDER9_C3_STAGE_ID)
    schedule_hash = order9_schedule_hash(config)
    source = load_order9_policy_checkpoint(
        source_checkpoint_path,
        device=device,
        expected_family=Order9PolicyFamily.PI_L,
        expected_schedule_hash=schedule_hash,
    )
    if type(source.model) is not Order9MorphologyInvariantCompressionActorCritic:
        raise SchemaValidationError("Order9 v7 initializer source must be exact v6")
    source_update = source.metadata.metadata.get("ppo_update_index")
    if not isinstance(source_update, int) or source_update < -1:
        raise SchemaValidationError("Order9 v7 source update index is invalid")
    checkpoint_path = Path(output_checkpoint_path).resolve()
    manifest_path = Path(output_manifest_path).resolve()
    for path in (checkpoint_path, manifest_path):
        if path.exists():
            raise FileExistsError(path)

    target_config = Order9ActiveKnotLowLevelPolicyConfig.from_dict(
        source.model_config.to_dict()
    )
    target = Order9ContactSpacePhaseConditionedActorCritic(target_config).to(device)
    fresh, unexpected = target.initialize_from_morphology_invariant_compression(
        source.model
    )
    if unexpected:
        raise AssertionError("validated v6-to-v7 migration has unexpected parameters")
    reset_names = _reset_newly_activated_v7_action_outputs(target)
    target_state = target.state_dict()
    copied = sorted(set(target_state) - set(fresh))
    copied_without_reset = sorted(set(copied) - set(reset_names))
    maximum_error = max(
        (
            float(
                (
                    target_state[name].detach().cpu()
                    - source.model.state_dict()[name].detach().cpu()
                )
                .abs()
                .max()
                .item()
            )
            for name in copied_without_reset
        ),
        default=0.0,
    )
    zero_names = (
        "contact_space_actor_mean.2.weight",
        "contact_space_actor_mean.2.bias",
    )
    if any(bool((target_state[name] != 0.0).any()) for name in zero_names):
        raise SchemaValidationError("Order9 v7 contact action mean is not zero")
    if any(
        bool((target_state[name] != 0.0).any())
        for name in _NEWLY_ACTIVATED_MEAN_PARAMETER_NAMES
    ):
        raise SchemaValidationError(
            "Order9 v7 newly activated action mean is not zero"
        )
    metadata = build_order9_checkpoint_metadata(
        target,
        stage=stage,
        schedule_hash=schedule_hash,
        physical_model_hash=source.metadata.physical_model_hash,
        git_revision=git_revision,
        random_seed=config.production_runtime.seed + stage.stage_index * 100_000,
        input_artifact_hashes={"source_v6_checkpoint": source.sha256},
        parent_checkpoint_sha256=source.sha256,
        source_order3_checkpoint_sha256=(
            source.metadata.source_order3_checkpoint_sha256
        ),
        metrics={
            "maximum_copied_parameter_error": maximum_error,
            "copied_parameter_count": float(len(copied_without_reset)),
            "fresh_parameter_count": float(len(fresh)),
            "reset_parameter_count": float(len(reset_names)),
        },
        trainer_version=ORDER9_PI_L_CONTACT_SPACE_MIGRATION_VERSION,
        extra_metadata={
            "ppo_update_index": source_update,
            "consumed_rollout_manifest_sha256s": [],
            "source_policy_version": source.metadata.policy_version,
            "dropped_v6_grasp_specific_heads": True,
            "residual_wrench_actor_coordinate_applied": False,
            "joint_torque_actor_coordinate_applied": False,
            "raw_contact_actor_input": False,
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
    if (
        type(loaded.model) is not Order9ContactSpacePhaseConditionedActorCritic
        or loaded.metadata.policy_version != ORDER9_CONTACT_SPACE_PI_L_POLICY_VERSION
    ):
        raise SchemaValidationError("Order9 v7 checkpoint reconstructed incorrectly")
    manifest: dict[str, object] = {
        "migration_version": ORDER9_PI_L_CONTACT_SPACE_MIGRATION_VERSION,
        "source_checkpoint_path": str(Path(source_checkpoint_path).resolve()),
        "source_checkpoint_sha256": source.sha256,
        "source_policy_version": (
            ORDER9_MORPHOLOGY_INVARIANT_COMPRESSION_PI_L_POLICY_VERSION
        ),
        "target_checkpoint_path": str(checkpoint_path),
        "target_checkpoint_sha256": target_sha,
        "target_policy_version": ORDER9_CONTACT_SPACE_PI_L_POLICY_VERSION,
        "target_update_index": source_update,
        "schedule_hash": schedule_hash,
        "copied_parameter_names": copied,
        "copied_parameter_names_excluding_reset": copied_without_reset,
        "fresh_parameter_names": sorted(fresh),
        "zero_output_parameter_names": sorted(
            set(zero_names) | set(_NEWLY_ACTIVATED_MEAN_PARAMETER_NAMES)
        ),
        "reset_parameter_names": list(reset_names),
        "maximum_copied_parameter_error": maximum_error,
    }
    _atomic_write_json(manifest_path, manifest)
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
    "ORDER9_PI_L_CONTACT_SPACE_MIGRATION_VERSION",
    "prepare_order9_pi_l_contact_space_initializer",
]
