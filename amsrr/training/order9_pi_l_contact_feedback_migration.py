from __future__ import annotations

"""Hash-bound v7-to-v8 deployable contact-feedback initializer migration."""

import json
import math
import os
import tempfile
from pathlib import Path

import torch

from amsrr.policies.order9_low_level_policy import (
    ORDER9_CONTACT_FEEDBACK_PI_L_POLICY_VERSION,
    ORDER9_CONTACT_FEEDBACK_SPACE_FEATURE_NAMES,
    ORDER9_CONTACT_SPACE_FEATURE_NAMES,
    ORDER9_CONTACT_SPACE_PI_L_POLICY_VERSION,
    Order9ContactFeedbackLowLevelPolicyConfig,
    Order9ContactFeedbackPhaseConditionedActorCritic,
    Order9ContactSpacePhaseConditionedActorCritic,
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


ORDER9_PI_L_CONTACT_FEEDBACK_MIGRATION_VERSION = (
    "order9_pi_l_v7_to_deployable_contact_feedback_v8_v1"
)
ORDER9_C3_STAGE_ID = "c3_pi_l_ppo_arbitrary_morphology"


def prepare_order9_pi_l_contact_feedback_initializer(
    *,
    config_path: str | Path,
    source_checkpoint_path: str | Path,
    output_checkpoint_path: str | Path,
    output_manifest_path: str | Path,
    git_revision: str,
    initial_inward_residual_m: float = 0.006,
    initial_normal_exploration_std_m: float = 0.001,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    if not 0.0 <= float(initial_inward_residual_m) < 0.010:
        raise SchemaValidationError(
            "Order9 v8 initial inward residual must be inside the non-negative "
            "10 mm span"
        )
    if not 0.0 < float(initial_normal_exploration_std_m) < 0.010:
        raise SchemaValidationError(
            "Order9 v8 normal exploration std must be inside the 10 mm span"
        )
    config = load_order9_learning_config(config_path)
    stage = order9_stage_by_id(config, ORDER9_C3_STAGE_ID)
    schedule_hash = order9_schedule_hash(config)
    source = load_order9_policy_checkpoint(
        source_checkpoint_path,
        device=device,
        expected_family=Order9PolicyFamily.PI_L,
        expected_schedule_hash=schedule_hash,
    )
    if type(source.model) is not Order9ContactSpacePhaseConditionedActorCritic:
        raise SchemaValidationError("Order9 v8 initializer source must be exact v7")
    if source.metadata.policy_version != ORDER9_CONTACT_SPACE_PI_L_POLICY_VERSION:
        raise SchemaValidationError("Order9 v8 initializer source version differs")
    source_update = source.metadata.metadata.get("ppo_update_index")
    if not isinstance(source_update, int) or source_update < -1:
        raise SchemaValidationError("Order9 v8 source update index is invalid")

    checkpoint_path = Path(output_checkpoint_path).resolve()
    manifest_path = Path(output_manifest_path).resolve()
    for path in (checkpoint_path, manifest_path):
        if path.exists():
            raise FileExistsError(path)
    target_config_values = source.model_config.to_dict()
    target_config_values["contact_space_feature_dim"] = len(
        ORDER9_CONTACT_FEEDBACK_SPACE_FEATURE_NAMES
    )
    target_config = Order9ContactFeedbackLowLevelPolicyConfig.from_dict(
        target_config_values
    )
    target = Order9ContactFeedbackPhaseConditionedActorCritic(target_config).to(
        device
    )
    fresh, unexpected = target.initialize_from_contact_space(source.model)
    if unexpected:
        raise AssertionError("validated v7-to-v8 migration has unexpected parameters")

    normal_span_m = 0.010
    normalized_mean = float(initial_inward_residual_m) / normal_span_m
    normalized_std = float(initial_normal_exploration_std_m) / normal_span_m
    final = target.contact_space_actor_mean[-1]
    if not isinstance(final, torch.nn.Linear):  # pragma: no cover
        raise RuntimeError("Order9 v8 contact actor final layer is not linear")
    with torch.no_grad():
        final.bias[0] = math.atanh(normalized_mean)
        target.contact_space_actor_log_std[0] = math.log(normalized_std)

    expanded_name = "contact_space_feature_encoder.0.weight"
    source_width = len(ORDER9_CONTACT_SPACE_FEATURE_NAMES)
    target_state = target.state_dict()
    if bool((target_state[expanded_name][:, source_width:] != 0.0).any()):
        raise SchemaValidationError("Order9 v8 fresh feedback columns are non-zero")
    copied_error = max(
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
            for name in source.model.state_dict()
            if name not in {
                expanded_name,
                "contact_space_actor_mean.2.bias",
                "contact_space_actor_log_std",
            }
        ),
        default=0.0,
    )
    if copied_error != 0.0:
        raise SchemaValidationError("Order9 v8 copied parameter differs")

    metadata = build_order9_checkpoint_metadata(
        target,
        stage=stage,
        schedule_hash=schedule_hash,
        physical_model_hash=source.metadata.physical_model_hash,
        git_revision=git_revision,
        random_seed=config.production_runtime.seed + stage.stage_index * 100_000,
        input_artifact_hashes={"source_v7_checkpoint": source.sha256},
        parent_checkpoint_sha256=source.sha256,
        source_order3_checkpoint_sha256=(
            source.metadata.source_order3_checkpoint_sha256
        ),
        metrics={
            "maximum_copied_parameter_error": copied_error,
            "initial_inward_residual_m": float(initial_inward_residual_m),
            "initial_normal_exploration_std_m": float(
                initial_normal_exploration_std_m
            ),
        },
        trainer_version=ORDER9_PI_L_CONTACT_FEEDBACK_MIGRATION_VERSION,
        extra_metadata={
            "acceptance_eligible": False,
            "ppo_update_index": -1,
            "warm_start_source_ppo_update_index": source_update,
            "warm_start_only": True,
            "warm_start_requires_fresh_on_policy_ppo": True,
            "source_policy_version": source.metadata.policy_version,
            "deployable_contact_feedback": True,
            "raw_contact_actor_input": False,
            "contact_feedback_inputs": [
                "kinematic_grasp_frame_surface_distance",
                "kinematic_anchor_object_relative_twist",
                "signed_motor_load_compression_proxy",
            ],
            "diagnostic_informed_contact_normal_initializer": True,
            "initial_inward_residual_m": float(initial_inward_residual_m),
            "initial_normal_exploration_std_m": float(
                initial_normal_exploration_std_m
            ),
            "c3_action_contract": "contact_space_projected_policy_command",
        },
    )
    checkpoint_sha = save_order9_policy_checkpoint(
        checkpoint_path, model=target, metadata=metadata
    )
    loaded = load_order9_policy_checkpoint(
        checkpoint_path,
        device=device,
        expected_sha256=checkpoint_sha,
        expected_family=Order9PolicyFamily.PI_L,
        expected_schedule_hash=schedule_hash,
    )
    if (
        type(loaded.model) is not Order9ContactFeedbackPhaseConditionedActorCritic
        or loaded.metadata.policy_version
        != ORDER9_CONTACT_FEEDBACK_PI_L_POLICY_VERSION
    ):
        raise SchemaValidationError("Order9 v8 checkpoint reconstructed incorrectly")

    manifest: dict[str, object] = {
        "migration_version": ORDER9_PI_L_CONTACT_FEEDBACK_MIGRATION_VERSION,
        "source_checkpoint_path": str(Path(source_checkpoint_path).resolve()),
        "source_checkpoint_sha256": source.sha256,
        "source_policy_version": source.metadata.policy_version,
        "source_update_index": source_update,
        "target_checkpoint_path": str(checkpoint_path),
        "target_checkpoint_sha256": checkpoint_sha,
        "target_policy_version": ORDER9_CONTACT_FEEDBACK_PI_L_POLICY_VERSION,
        "target_update_index": -1,
        "schedule_hash": schedule_hash,
        "fresh_parameter_slices": fresh,
        "initial_inward_residual_m": float(initial_inward_residual_m),
        "initial_normal_exploration_std_m": float(
            initial_normal_exploration_std_m
        ),
        "maximum_copied_parameter_error": copied_error,
        "raw_contact_actor_input": False,
        "requires_fresh_on_policy_ppo": True,
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
    "ORDER9_PI_L_CONTACT_FEEDBACK_MIGRATION_VERSION",
    "prepare_order9_pi_l_contact_feedback_initializer",
]
