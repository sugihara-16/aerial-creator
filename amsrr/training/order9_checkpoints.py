from __future__ import annotations

"""Strict, hash-bound checkpoints shared by the three Order 9 policies."""

import hashlib
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from amsrr.policies.order9_design_policy import (
    ORDER9_AUTOREGRESSIVE_PI_D_VERSION,
    Order9AutoregressiveDesignPolicy,
    Order9DesignPolicyConfig,
)
from amsrr.policies.order9_high_level_policy import (
    ORDER9_FULL_PI_H_VERSION,
    Order9AutoregressiveHighLevelPolicy,
    Order9HighLevelPolicyConfig,
)
from amsrr.policies.order9_low_level_policy import (
    ORDER9_PI_L_POLICY_VERSION,
    Order9LowLevelPolicyConfig,
    Order9PhaseConditionedActorCritic,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.order9 import (
    ORDER9_POLICY_CHECKPOINT_VERSION,
    Order9PolicyCheckpointMetadata,
    Order9PolicyFamily,
)
from amsrr.utils.hashing import hash_file, stable_hash


@dataclass(frozen=True)
class LoadedOrder9PolicyCheckpoint:
    model: nn.Module
    model_config: object
    metadata: Order9PolicyCheckpointMetadata
    path: str
    sha256: str


def order9_model_config_dict(model: nn.Module) -> dict[str, Any]:
    config = getattr(model, "config", None)
    if config is None:
        raise SchemaValidationError("Order9 model has no serializable config")
    if hasattr(config, "to_dict"):
        value = config.to_dict()
    else:
        value = asdict(config)
    if not isinstance(value, dict):
        raise SchemaValidationError("Order9 model config must serialize to a mapping")
    return value


def order9_policy_identity(model: nn.Module) -> tuple[Order9PolicyFamily, str]:
    if isinstance(model, Order9PhaseConditionedActorCritic):
        return Order9PolicyFamily.PI_L, ORDER9_PI_L_POLICY_VERSION
    if isinstance(model, Order9AutoregressiveHighLevelPolicy):
        return Order9PolicyFamily.PI_H, ORDER9_FULL_PI_H_VERSION
    if isinstance(model, Order9AutoregressiveDesignPolicy):
        return Order9PolicyFamily.PI_D, ORDER9_AUTOREGRESSIVE_PI_D_VERSION
    raise SchemaValidationError(
        f"unsupported Order9 policy model type {type(model).__name__!r}"
    )


def order9_state_dict_hash(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise SchemaValidationError("Order9 state_dict values must be tensors")
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def save_order9_policy_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    metadata: Order9PolicyCheckpointMetadata,
) -> str:
    family, policy_version = order9_policy_identity(model)
    config = order9_model_config_dict(model)
    _validate_runtime_contract(
        family=family,
        policy_version=policy_version,
        config=config,
        state_dict=model.state_dict(),
        metadata=metadata,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": ORDER9_POLICY_CHECKPOINT_VERSION,
        "metadata": metadata.to_dict(),
        "model_config": config,
        "state_dict": model.state_dict(),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return hash_file(destination)


def load_order9_policy_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
    expected_sha256: str | None = None,
    expected_family: Order9PolicyFamily | str | None = None,
    expected_schedule_hash: str | None = None,
) -> LoadedOrder9PolicyCheckpoint:
    source = Path(path)
    actual_sha256 = hash_file(source)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise SchemaValidationError("Order9 policy checkpoint SHA-256 mismatch")
    try:
        payload = torch.load(source, map_location=device, weights_only=False)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SchemaValidationError(
            f"failed to load Order9 policy checkpoint: {exc}"
        ) from exc
    required = {"checkpoint_version", "metadata", "model_config", "state_dict"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise SchemaValidationError("Order9 policy checkpoint keys do not match v1")
    if payload["checkpoint_version"] != ORDER9_POLICY_CHECKPOINT_VERSION:
        raise SchemaValidationError("Order9 policy checkpoint version mismatch")
    metadata = Order9PolicyCheckpointMetadata.from_dict(payload["metadata"])
    family = metadata.policy_family
    if expected_family is not None and family != Order9PolicyFamily(expected_family):
        raise SchemaValidationError("Order9 policy checkpoint family mismatch")
    if (
        expected_schedule_hash is not None
        and metadata.curriculum_schedule_hash != expected_schedule_hash
    ):
        raise SchemaValidationError("Order9 policy checkpoint curriculum hash mismatch")
    model, config = _construct_model(family, payload["model_config"], device)
    expected_identity = order9_policy_identity(model)
    _validate_runtime_contract(
        family=expected_identity[0],
        policy_version=expected_identity[1],
        config=order9_model_config_dict(model),
        state_dict=payload["state_dict"],
        metadata=metadata,
    )
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise SchemaValidationError(
            f"Order9 policy state_dict is incompatible: {exc}"
        ) from exc
    model.eval()
    return LoadedOrder9PolicyCheckpoint(
        model=model,
        model_config=config,
        metadata=metadata,
        path=str(source),
        sha256=actual_sha256,
    )


def _construct_model(
    family: Order9PolicyFamily,
    raw_config: object,
    device: torch.device | str,
) -> tuple[nn.Module, object]:
    if not isinstance(raw_config, dict):
        raise SchemaValidationError("Order9 checkpoint model_config must be a mapping")
    try:
        if family == Order9PolicyFamily.PI_L:
            config = Order9LowLevelPolicyConfig.from_dict(raw_config)
            model: nn.Module = Order9PhaseConditionedActorCritic(config)
        elif family == Order9PolicyFamily.PI_H:
            config = Order9HighLevelPolicyConfig(**raw_config)
            model = Order9AutoregressiveHighLevelPolicy(config)
        elif family == Order9PolicyFamily.PI_D:
            config = Order9DesignPolicyConfig(**raw_config)
            model = Order9AutoregressiveDesignPolicy(config)
        else:  # pragma: no cover - enum construction prevents this.
            raise SchemaValidationError(f"unsupported Order9 family {family.value!r}")
    except (TypeError, ValueError, SchemaValidationError) as exc:
        raise SchemaValidationError(f"invalid Order9 model config: {exc}") from exc
    return model.to(device), config


def _validate_runtime_contract(
    *,
    family: Order9PolicyFamily,
    policy_version: str,
    config: dict[str, Any],
    state_dict: Mapping[str, torch.Tensor],
    metadata: Order9PolicyCheckpointMetadata,
) -> None:
    metadata.validate()
    if metadata.policy_family != family or metadata.policy_version != policy_version:
        raise SchemaValidationError("Order9 checkpoint policy identity mismatch")
    if metadata.model_config_hash != stable_hash(config):
        raise SchemaValidationError("Order9 checkpoint model config hash mismatch")
    if metadata.state_dict_hash != order9_state_dict_hash(state_dict):
        raise SchemaValidationError("Order9 checkpoint state_dict hash mismatch")
