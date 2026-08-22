from __future__ import annotations

"""Neutralize only the learned contact preload residual after a nominal rebase."""

import json
import os
import tempfile
from pathlib import Path

import torch

from amsrr.policies.order9_low_level_policy import (
    ORDER9_CATEGORICAL_CONTACT_NORMAL_PI_L_POLICY_VERSION,
    Order9CategoricalContactNormalPhaseConditionedActorCritic,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.order9 import Order9PolicyFamily
from amsrr.training.order9_checkpoints import (
    load_order9_policy_checkpoint,
    save_order9_policy_checkpoint,
)
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_offline_training import build_order9_checkpoint_metadata
from amsrr.training.order9_pipeline import order9_schedule_hash, order9_stage_by_id
from amsrr.utils.hashing import hash_file


ORDER9_ACTUATOR_AWARE_PRELOAD_MIGRATION_VERSION = (
    "order9_actuator_aware_nominal_preload_residual_rebase_v1"
)
ORDER9_C3_STAGE_ID = "c3_pi_l_ppo_arbitrary_morphology"


def prepare_order9_actuator_aware_preload_initializer(
    *,
    config_path: str | Path,
    source_checkpoint_path: str | Path,
    output_checkpoint_path: str | Path,
    output_manifest_path: str | Path,
    git_revision: str,
    neutral_residual_std_normalized: float = 0.15,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    """Rebase a v9 contact residual to zero without changing other heads.

    The actuator-aware nominal resolver now supplies the morphology-dependent
    preload.  The categorical normal coordinate is therefore a residual around
    zero.  Existing checkpoints were trained with a positive-preload prior, so
    only the final categorical layer is reset.  A subsequent fresh on-policy
    update is mandatory.
    """

    sigma = float(neutral_residual_std_normalized)
    if not 0.0 < sigma <= 1.0:
        raise SchemaValidationError("neutral contact residual std is invalid")
    config_path = Path(config_path).resolve()
    source_checkpoint_path = Path(source_checkpoint_path).resolve()
    checkpoint_path = Path(output_checkpoint_path).resolve()
    manifest_path = Path(output_manifest_path).resolve()
    for path in (checkpoint_path, manifest_path):
        if path.exists():
            raise FileExistsError(path)

    learning = load_order9_learning_config(config_path)
    stage = order9_stage_by_id(learning, ORDER9_C3_STAGE_ID)
    schedule_hash = order9_schedule_hash(learning)
    physical = build_physical_model_from_config(
        Path(learning.production_runtime.robot_model_config_path).resolve()
    )
    source = load_order9_policy_checkpoint(
        source_checkpoint_path,
        device=device,
        expected_family=Order9PolicyFamily.PI_L,
    )
    if type(source.model) is not Order9CategoricalContactNormalPhaseConditionedActorCritic:
        raise SchemaValidationError("preload residual rebase requires exact v9 pi_L")
    if source.metadata.policy_version != ORDER9_CATEGORICAL_CONTACT_NORMAL_PI_L_POLICY_VERSION:
        raise SchemaValidationError("preload residual rebase policy version differs")
    if source.metadata.physical_model_hash != physical.stable_hash():
        raise SchemaValidationError("preload residual rebase physical model differs")
    source_update = source.metadata.metadata.get("ppo_update_index")
    if not isinstance(source_update, int) or source_update < 0:
        raise SchemaValidationError("preload residual rebase source update is invalid")
    if source.model.config.contact_normal_category_prior_mode != "uniform_window":
        raise SchemaValidationError(
            "preload residual rebase currently requires the reviewed positive-window prior"
        )

    target = Order9CategoricalContactNormalPhaseConditionedActorCritic(
        source.model_config
    ).to(device)
    target.load_state_dict(source.model.state_dict(), strict=True)
    final = target.contact_normal_category_logits[-1]
    if not isinstance(final, torch.nn.Linear):  # pragma: no cover
        raise RuntimeError("categorical contact-normal final layer differs")
    categories = torch.linspace(
        target.config.contact_normal_category_min_normalized,
        target.config.contact_normal_category_max_normalized,
        target.config.contact_normal_category_count,
        device=final.bias.device,
        dtype=final.bias.dtype,
    )
    desired_logits = -0.5 * (categories / sigma).square()
    in_positive_window = (
        categories
        >= target.config.contact_normal_category_uniform_window_min_normalized
    ) & (
        categories
        <= target.config.contact_normal_category_uniform_window_max_normalized
    )
    built_in_prior = torch.where(
        in_positive_window,
        torch.zeros_like(categories),
        torch.full_like(
            categories,
            -float(target.config.contact_normal_category_uniform_outside_logit_penalty),
        ),
    )
    with torch.no_grad():
        final.weight.zero_()
        final.bias.copy_(desired_logits - built_in_prior)

    source_state = source.model.state_dict()
    target_state = target.state_dict()
    changed = sorted(
        name
        for name in target_state
        if not torch.equal(target_state[name].detach().cpu(), source_state[name].detach().cpu())
    )
    expected_changed = sorted(
        (
            "contact_normal_category_logits.2.bias",
            "contact_normal_category_logits.2.weight",
        )
    )
    if changed != expected_changed:
        raise SchemaValidationError(
            f"preload residual rebase changed unexpected parameters: {changed}"
        )

    metadata = build_order9_checkpoint_metadata(
        target,
        stage=stage,
        schedule_hash=schedule_hash,
        physical_model_hash=physical.stable_hash(),
        git_revision=git_revision,
        random_seed=learning.production_runtime.seed + stage.stage_index * 100_000,
        input_artifact_hashes={
            "source_checkpoint": source.sha256,
            "training_config": hash_file(config_path),
        },
        parent_checkpoint_sha256=source.sha256,
        source_order3_checkpoint_sha256=source.metadata.source_order3_checkpoint_sha256,
        metrics={
            "changed_parameter_tensor_count": float(len(changed)),
            "neutral_residual_std_normalized": sigma,
            "neutral_residual_std_m": 0.020 * sigma,
        },
        trainer_version=ORDER9_ACTUATOR_AWARE_PRELOAD_MIGRATION_VERSION,
        extra_metadata={
            "acceptance_eligible": False,
            "ppo_update_index": -1,
            "warm_start_source_ppo_update_index": source_update,
            "warm_start_only": True,
            "warm_start_requires_fresh_on_policy_ppo": True,
            "actuator_aware_nominal_preload_residual_rebase": True,
            "changed_parameter_names": changed,
            "contact_normal_residual_center_m": 0.0,
            "c3_action_contract": "contact_space_projected_policy_command",
        },
    )
    checkpoint_sha256 = save_order9_policy_checkpoint(
        checkpoint_path, model=target, metadata=metadata
    )
    loaded = load_order9_policy_checkpoint(
        checkpoint_path,
        device=device,
        expected_sha256=checkpoint_sha256,
        expected_family=Order9PolicyFamily.PI_L,
        expected_schedule_hash=schedule_hash,
    )
    if loaded.metadata.metadata.get("acceptance_eligible") is not False:
        raise SchemaValidationError("preload residual rebase became acceptance eligible")

    manifest: dict[str, object] = {
        "migration_version": ORDER9_ACTUATOR_AWARE_PRELOAD_MIGRATION_VERSION,
        "source_checkpoint_path": str(source_checkpoint_path),
        "source_checkpoint_sha256": source.sha256,
        "source_update_index": source_update,
        "target_checkpoint_path": str(checkpoint_path),
        "target_checkpoint_sha256": checkpoint_sha256,
        "target_update_index": -1,
        "schedule_hash": schedule_hash,
        "changed_parameter_names": changed,
        "neutral_residual_std_normalized": sigma,
        "neutral_residual_std_m": 0.020 * sigma,
        "acceptance_eligible": False,
        "requires_fresh_on_policy_ppo": True,
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


__all__ = [
    "ORDER9_ACTUATOR_AWARE_PRELOAD_MIGRATION_VERSION",
    "prepare_order9_actuator_aware_preload_initializer",
]
