from __future__ import annotations

"""Exact, provenance-bound physical-model rebind for a C3 ``pi_L`` initializer."""

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from amsrr.policies.order9_low_level_policy import (
    ORDER9_ACTIVE_KNOT_PI_L_POLICY_VERSION,
    Order9ActiveKnotPhaseConditionedActorCritic,
)
from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_non_empty
from amsrr.schemas.order9 import Order9PolicyCheckpointMetadata, Order9PolicyFamily
from amsrr.training.order9_checkpoints import (
    load_order9_policy_checkpoint,
    save_order9_policy_checkpoint,
)
from amsrr.utils.hashing import hash_file


ORDER9_PI_L_INITIALIZER_PHYSICAL_REBIND_VERSION = (
    "order9_pi_l_initializer_physical_rebind_v1"
)


@dataclass
class Order9PiLInitializerPhysicalRebindManifest(SchemaBase):
    rebind_version: str
    source_checkpoint_path: str
    source_checkpoint_sha256: str
    target_checkpoint_path: str
    target_checkpoint_sha256: str
    curriculum_schedule_hash: str
    source_physical_model_hash: str
    target_physical_model_hash: str
    source_state_dict_hash: str
    target_state_dict_hash: str
    exact_parameter_copy: bool
    metadata: dict[str, Any]

    def validate(self) -> None:
        if self.rebind_version != ORDER9_PI_L_INITIALIZER_PHYSICAL_REBIND_VERSION:
            raise SchemaValidationError("Order9 initializer rebind version mismatch")
        for name in ("source_checkpoint_path", "target_checkpoint_path"):
            require_non_empty(str(getattr(self, name)), name)
        for name in (
            "source_checkpoint_sha256",
            "target_checkpoint_sha256",
            "curriculum_schedule_hash",
            "source_physical_model_hash",
            "target_physical_model_hash",
            "source_state_dict_hash",
            "target_state_dict_hash",
        ):
            _require_sha256(str(getattr(self, name)), name)
        if not self.exact_parameter_copy:
            raise SchemaValidationError("Order9 initializer rebind must copy parameters exactly")
        if self.source_state_dict_hash != self.target_state_dict_hash:
            raise SchemaValidationError("Order9 initializer rebind changed policy parameters")


@dataclass(frozen=True)
class PreparedOrder9PiLInitializerPhysicalRebind:
    checkpoint_path: Path
    checkpoint_sha256: str
    manifest_path: Path
    manifest_sha256: str
    manifest: Order9PiLInitializerPhysicalRebindManifest


def prepare_order9_pi_l_initializer_physical_rebind(
    *,
    source_checkpoint_path: str | Path,
    output_checkpoint_path: str | Path,
    output_manifest_path: str | Path,
    target_physical_model_hash: str,
    git_revision: str,
    device: str | torch.device = "cpu",
) -> PreparedOrder9PiLInitializerPhysicalRebind:
    """Re-save an untrained initializer with current physical provenance only.

    No tensor, model configuration, policy contract, stage identity, schedule,
    random seed, or C2 parent is changed.  The source initializer remains an
    immutable input artifact and the new checkpoint records its SHA-256.
    """

    _require_sha256(target_physical_model_hash, "target_physical_model_hash")
    require_non_empty(git_revision, "git_revision")
    source_path = Path(source_checkpoint_path).resolve()
    checkpoint_path = Path(output_checkpoint_path).resolve()
    manifest_path = Path(output_manifest_path).resolve()
    for path in (checkpoint_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"Order9 initializer rebind output exists: {path}")
    source = load_order9_policy_checkpoint(
        source_path,
        device=device,
        expected_family=Order9PolicyFamily.PI_L,
    )
    if (
        source.metadata.policy_version != ORDER9_ACTIVE_KNOT_PI_L_POLICY_VERSION
        or not isinstance(source.model, Order9ActiveKnotPhaseConditionedActorCritic)
        or source.metadata.metadata.get("initializer_only") is not True
        or source.metadata.metadata.get("ppo_update_index") != -1
        or source.metadata.metadata.get("promoted_checkpoint") is not False
    ):
        raise SchemaValidationError(
            "Order9 physical rebind source is not the unpromoted C3 initializer"
        )
    metadata_values = source.metadata.to_dict()
    metadata_values["physical_model_hash"] = target_physical_model_hash
    input_hashes = dict(source.metadata.input_artifact_hashes)
    input_hashes["physical_rebind_source_initializer"] = source.sha256
    input_hashes["target_physical_model"] = target_physical_model_hash
    metadata_values["input_artifact_hashes"] = input_hashes
    extra = dict(source.metadata.metadata)
    extra.update(
        {
            "trainer_version": ORDER9_PI_L_INITIALIZER_PHYSICAL_REBIND_VERSION,
            "physical_model_rebind": True,
            "physical_model_rebind_source_checkpoint_sha256": source.sha256,
            "source_physical_model_hash": source.metadata.physical_model_hash,
            "target_physical_model_hash": target_physical_model_hash,
            "rebind_git_revision": git_revision,
        }
    )
    metadata_values["metadata"] = extra
    metadata = Order9PolicyCheckpointMetadata.from_dict(metadata_values)
    target_sha = save_order9_policy_checkpoint(
        checkpoint_path,
        model=source.model,
        metadata=metadata,
    )
    target = load_order9_policy_checkpoint(
        checkpoint_path,
        device=device,
        expected_sha256=target_sha,
        expected_family=Order9PolicyFamily.PI_L,
        expected_schedule_hash=source.metadata.curriculum_schedule_hash,
    )
    manifest = Order9PiLInitializerPhysicalRebindManifest(
        rebind_version=ORDER9_PI_L_INITIALIZER_PHYSICAL_REBIND_VERSION,
        source_checkpoint_path=str(source_path),
        source_checkpoint_sha256=source.sha256,
        target_checkpoint_path=str(checkpoint_path),
        target_checkpoint_sha256=target.sha256,
        curriculum_schedule_hash=source.metadata.curriculum_schedule_hash,
        source_physical_model_hash=source.metadata.physical_model_hash,
        target_physical_model_hash=target_physical_model_hash,
        source_state_dict_hash=source.metadata.state_dict_hash,
        target_state_dict_hash=target.metadata.state_dict_hash,
        exact_parameter_copy=(
            source.metadata.state_dict_hash == target.metadata.state_dict_hash
        ),
        metadata={
            "policy_version": target.metadata.policy_version,
            "curriculum_stage_id": target.metadata.curriculum_stage_id,
            "curriculum_stage_index": target.metadata.curriculum_stage_index,
            "parent_checkpoint_sha256": target.metadata.parent_checkpoint_sha256,
            "semantic_scope": (
                "metadata-only rebind after URDF grasp-frame and collision-mesh "
                "update; policy tensors and interfaces are unchanged"
            ),
        },
    )
    manifest.validate()
    _atomic_write_json(manifest_path, manifest.to_dict())
    validate_order9_pi_l_initializer_physical_rebind(
        manifest_path,
        expected_target_physical_model_hash=target_physical_model_hash,
        device=device,
    )
    return PreparedOrder9PiLInitializerPhysicalRebind(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=target.sha256,
        manifest_path=manifest_path,
        manifest_sha256=hash_file(manifest_path),
        manifest=manifest,
    )


def validate_order9_pi_l_initializer_physical_rebind(
    manifest_path: str | Path,
    *,
    expected_target_physical_model_hash: str | None = None,
    device: str | torch.device = "cpu",
) -> Order9PiLInitializerPhysicalRebindManifest:
    path = Path(manifest_path).resolve()
    manifest = Order9PiLInitializerPhysicalRebindManifest.from_json(
        path.read_text(encoding="utf-8")
    )
    source = load_order9_policy_checkpoint(
        manifest.source_checkpoint_path,
        device=device,
        expected_sha256=manifest.source_checkpoint_sha256,
        expected_family=Order9PolicyFamily.PI_L,
        expected_schedule_hash=manifest.curriculum_schedule_hash,
    )
    target = load_order9_policy_checkpoint(
        manifest.target_checkpoint_path,
        device=device,
        expected_sha256=manifest.target_checkpoint_sha256,
        expected_family=Order9PolicyFamily.PI_L,
        expected_schedule_hash=manifest.curriculum_schedule_hash,
    )
    if (
        source.metadata.state_dict_hash != manifest.source_state_dict_hash
        or target.metadata.state_dict_hash != manifest.target_state_dict_hash
        or target.metadata.physical_model_hash != manifest.target_physical_model_hash
        or target.metadata.metadata.get("physical_model_rebind") is not True
        or target.metadata.metadata.get(
            "physical_model_rebind_source_checkpoint_sha256"
        )
        != source.sha256
        or target.metadata.parent_checkpoint_sha256
        != source.metadata.parent_checkpoint_sha256
    ):
        raise SchemaValidationError("Order9 initializer physical rebind lineage differs")
    if (
        expected_target_physical_model_hash is not None
        and manifest.target_physical_model_hash
        != expected_target_physical_model_hash
    ):
        raise SchemaValidationError("Order9 initializer target physical hash differs")
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
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise SchemaValidationError(f"{path} must be a lowercase SHA-256")


__all__ = [
    "ORDER9_PI_L_INITIALIZER_PHYSICAL_REBIND_VERSION",
    "Order9PiLInitializerPhysicalRebindManifest",
    "PreparedOrder9PiLInitializerPhysicalRebind",
    "prepare_order9_pi_l_initializer_physical_rebind",
    "validate_order9_pi_l_initializer_physical_rebind",
]
