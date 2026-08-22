from __future__ import annotations

"""Hash-bound v8-to-v9 categorical contact-normal policy migration."""

import json
import os
import tempfile
from pathlib import Path

import torch

from amsrr.policies.order9_low_level_policy import (
    ORDER9_CATEGORICAL_CONTACT_NORMAL_PI_L_POLICY_VERSION,
    ORDER9_CONTACT_FEEDBACK_PI_L_POLICY_VERSION,
    Order9CategoricalContactNormalLowLevelPolicyConfig,
    Order9CategoricalContactNormalPhaseConditionedActorCritic,
    Order9ContactFeedbackPhaseConditionedActorCritic,
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


ORDER9_PI_L_CATEGORICAL_CONTACT_NORMAL_MIGRATION_VERSION = (
    "order9_pi_l_v8_to_categorical_contact_normal_v9_v2"
)
ORDER9_C3_STAGE_ID = "c3_pi_l_ppo_arbitrary_morphology"


def prepare_order9_pi_l_categorical_contact_normal_initializer(
    *,
    config_path: str | Path,
    source_checkpoint_path: str | Path,
    output_checkpoint_path: str | Path,
    output_manifest_path: str | Path,
    git_revision: str,
    initial_std_normalized: float = 0.30,
    prior_mode: str = "gaussian_centered",
    uniform_window_min_normalized: float = 0.50,
    uniform_window_max_normalized: float = 1.00,
    uniform_outside_logit_penalty: float = 4.0,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    if not 0.0 < float(initial_std_normalized) <= 1.0:
        raise SchemaValidationError(
            "Order9 categorical contact-normal initial std is invalid"
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
    if type(source.model) is not Order9ContactFeedbackPhaseConditionedActorCritic:
        raise SchemaValidationError("Order9 v9 initializer source must be exact v8")
    if source.metadata.policy_version != ORDER9_CONTACT_FEEDBACK_PI_L_POLICY_VERSION:
        raise SchemaValidationError("Order9 v9 initializer source version differs")
    source_update = source.metadata.metadata.get("ppo_update_index")
    if not isinstance(source_update, int) or source_update < -1:
        raise SchemaValidationError("Order9 v9 source update index is invalid")

    checkpoint_path = Path(output_checkpoint_path).resolve()
    manifest_path = Path(output_manifest_path).resolve()
    for path in (checkpoint_path, manifest_path):
        if path.exists():
            raise FileExistsError(path)

    target_config_values = source.model_config.to_dict()
    target_config_values.update(
        {
            "contact_normal_category_count": 81,
            "contact_normal_category_min_normalized": -1.0,
            "contact_normal_category_max_normalized": 1.0,
            "contact_normal_category_initial_std_normalized": float(
                initial_std_normalized
            ),
            "contact_normal_category_prior_mode": str(prior_mode),
            "contact_normal_category_uniform_window_min_normalized": float(
                uniform_window_min_normalized
            ),
            "contact_normal_category_uniform_window_max_normalized": float(
                uniform_window_max_normalized
            ),
            "contact_normal_category_uniform_outside_logit_penalty": float(
                uniform_outside_logit_penalty
            ),
        }
    )
    target_config = Order9CategoricalContactNormalLowLevelPolicyConfig.from_dict(
        target_config_values
    )
    target = Order9CategoricalContactNormalPhaseConditionedActorCritic(
        target_config
    ).to(device)
    fresh, unexpected = target.initialize_from_contact_feedback(source.model)
    if unexpected:
        raise AssertionError("validated v8-to-v9 migration has unexpected parameters")

    source_state = source.model.state_dict()
    target_state = target.state_dict()
    copied_error = max(
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
            for name in source_state
        ),
        default=0.0,
    )
    if copied_error != 0.0:
        raise SchemaValidationError("Order9 v9 copied parameter differs")

    metadata = build_order9_checkpoint_metadata(
        target,
        stage=stage,
        schedule_hash=schedule_hash,
        physical_model_hash=source.metadata.physical_model_hash,
        git_revision=git_revision,
        random_seed=config.production_runtime.seed + stage.stage_index * 100_000,
        input_artifact_hashes={"source_v8_checkpoint": source.sha256},
        parent_checkpoint_sha256=source.sha256,
        source_order3_checkpoint_sha256=(
            source.metadata.source_order3_checkpoint_sha256
        ),
        metrics={
            "maximum_copied_parameter_error": copied_error,
            "contact_normal_category_count": 81.0,
            "contact_normal_category_step_m": 0.0005,
            "initial_std_normalized": float(initial_std_normalized),
            "uniform_window_min_normalized": float(
                uniform_window_min_normalized
            ),
            "uniform_window_max_normalized": float(
                uniform_window_max_normalized
            ),
            "uniform_outside_logit_penalty": float(
                uniform_outside_logit_penalty
            ),
        },
        trainer_version=ORDER9_PI_L_CATEGORICAL_CONTACT_NORMAL_MIGRATION_VERSION,
        extra_metadata={
            "acceptance_eligible": False,
            "ppo_update_index": source_update,
            "categorical_distribution_migration_only": True,
            "source_policy_version": source.metadata.policy_version,
            "contact_normal_action_distribution": (
                "categorical_81_bins_-20mm_to_20mm"
            ),
            "contact_normal_action_category_count": 81,
            "contact_normal_action_category_step_m": 0.0005,
            "contact_normal_action_min_m": -0.020,
            "contact_normal_action_max_m": 0.020,
            "initial_std_normalized": float(initial_std_normalized),
            "contact_normal_prior_mode": str(prior_mode),
            "uniform_window_min_normalized": float(
                uniform_window_min_normalized
            ),
            "uniform_window_max_normalized": float(
                uniform_window_max_normalized
            ),
            "uniform_outside_logit_penalty": float(
                uniform_outside_logit_penalty
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
        type(loaded.model)
        is not Order9CategoricalContactNormalPhaseConditionedActorCritic
        or loaded.metadata.policy_version
        != ORDER9_CATEGORICAL_CONTACT_NORMAL_PI_L_POLICY_VERSION
    ):
        raise SchemaValidationError("Order9 v9 checkpoint reconstructed incorrectly")

    manifest: dict[str, object] = {
        "migration_version": ORDER9_PI_L_CATEGORICAL_CONTACT_NORMAL_MIGRATION_VERSION,
        "source_checkpoint_path": str(Path(source_checkpoint_path).resolve()),
        "source_checkpoint_sha256": source.sha256,
        "source_policy_version": source.metadata.policy_version,
        "source_update_index": source_update,
        "target_checkpoint_path": str(checkpoint_path),
        "target_checkpoint_sha256": checkpoint_sha,
        "target_policy_version": ORDER9_CATEGORICAL_CONTACT_NORMAL_PI_L_POLICY_VERSION,
        "target_update_index": source_update,
        "schedule_hash": schedule_hash,
        "fresh_parameters": fresh,
        "contact_normal_action_category_count": 81,
        "contact_normal_action_category_step_m": 0.0005,
        "initial_std_normalized": float(initial_std_normalized),
        "contact_normal_prior_mode": str(prior_mode),
        "uniform_window_min_normalized": float(uniform_window_min_normalized),
        "uniform_window_max_normalized": float(uniform_window_max_normalized),
        "uniform_outside_logit_penalty": float(uniform_outside_logit_penalty),
        "maximum_copied_parameter_error": copied_error,
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
    "ORDER9_PI_L_CATEGORICAL_CONTACT_NORMAL_MIGRATION_VERSION",
    "prepare_order9_pi_l_categorical_contact_normal_initializer",
]
