from __future__ import annotations

"""R1-only diagnostic mask retaining joint and normal-compression actions."""

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

import torch

from amsrr.schemas.common import SchemaValidationError
from amsrr.utils.hashing import hash_file

ORDER9_R1_JOINT_COMPRESSION_DIAGNOSTIC_VERSION = (
    "order9_r1_pi_l_joint_and_contact_normal_only_v1"
)


def mask_order9_r1_joint_and_compression_inputs(
    *,
    normalized_global_action: torch.Tensor,
    normalized_joint_action: torch.Tensor,
    normalized_contact_action: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep direct joint action and contact-normal compression only."""

    if normalized_global_action.ndim != 2:
        raise ValueError("R1 diagnostic global action shape is invalid")
    if normalized_joint_action.ndim != 3:
        raise ValueError("R1 diagnostic joint action shape is invalid")
    if (
        normalized_contact_action.ndim != 3
        or normalized_contact_action.shape[-1] < 1
    ):
        raise ValueError("R1 diagnostic contact action shape is invalid")
    batch = normalized_global_action.shape[0]
    if (
        normalized_joint_action.shape[0] != batch
        or normalized_contact_action.shape[0] != batch
    ):
        raise ValueError("R1 diagnostic action batches differ")
    for value in (
        normalized_global_action,
        normalized_joint_action,
        normalized_contact_action,
    ):
        if not bool(torch.isfinite(value).all()):
            raise ValueError("R1 diagnostic action contains non-finite values")
    global_action = torch.zeros_like(normalized_global_action)
    joint_action = normalized_joint_action
    contact_action = torch.zeros_like(normalized_contact_action)
    contact_action[:, :, 0] = normalized_contact_action[:, :, 0]
    return global_action, joint_action, contact_action


class Order9R1JointCompressionProjectionRecorder:
    """Mask contact-space inputs and retain the exact applied tensors."""

    def __init__(self, original: Callable[..., Any]) -> None:
        self.original = original
        self.applied_global_actions: list[torch.Tensor] = []
        self.applied_joint_actions: list[torch.Tensor] = []
        self.applied_contact_actions: list[torch.Tensor] = []

    def __call__(self, **values: Any) -> Any:
        global_action, joint_action, contact_action = (
            mask_order9_r1_joint_and_compression_inputs(
                normalized_global_action=values["normalized_global_action"],
                normalized_joint_action=values["normalized_joint_action"],
                normalized_contact_action=values["normalized_contact_action"],
            )
        )
        masked = dict(values)
        masked["normalized_global_action"] = global_action
        masked["normalized_joint_action"] = joint_action
        masked["normalized_contact_action"] = contact_action
        result = self.original(**masked)
        if not bool((result.global_action == 0.0).all()):
            raise SchemaValidationError(
                "R1 joint/compression diagnostic leaked a global action"
            )
        self.applied_global_actions.append(
            result.global_action.detach().to(device="cpu").clone()
        )
        self.applied_joint_actions.append(
            result.joint_action.detach().to(device="cpu").clone()
        )
        self.applied_contact_actions.append(
            contact_action.detach().to(device="cpu").clone()
        )
        return result

    def stacked(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.applied_global_actions:
            raise SchemaValidationError(
                "R1 joint/compression diagnostic recorded no policy steps"
            )
        return (
            torch.stack(self.applied_global_actions),
            torch.stack(self.applied_joint_actions),
            torch.stack(self.applied_contact_actions),
        )


def annotate_order9_r1_joint_compression_rollout(
    *,
    raw_path: str | Path,
    episode_path: str | Path,
    recorder: Order9R1JointCompressionProjectionRecorder,
    wrapper_path: str | Path,
    protected_rollout_path: str | Path,
    protected_rollout_sha256: str,
) -> None:
    """Bind the diagnostic mask and refresh episode-to-raw hashes."""

    raw_source = Path(raw_path).resolve()
    episode_source = Path(episode_path).resolve()
    wrapper_source = Path(wrapper_path).resolve()
    protected_source = Path(protected_rollout_path).resolve()
    if hash_file(protected_source) != protected_rollout_sha256:
        raise SchemaValidationError("protected C3 rollout bytes changed")
    payload = torch.load(raw_source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise SchemaValidationError("R1 diagnostic raw artifact is invalid")
    tensors = payload.get("tensors")
    metadata = payload.get("metadata")
    if not isinstance(tensors, dict) or not isinstance(metadata, dict):
        raise SchemaValidationError("R1 diagnostic raw artifact is incomplete")
    global_action, joint_action, contact_action = recorder.stacked()
    step_count = int(tensors["time_s"].shape[0])
    if (
        global_action.shape[0] != step_count
        or joint_action.shape[0] != step_count
        or contact_action.shape[0] != step_count
        or not bool((global_action == 0.0).all())
        or not bool((contact_action[:, :, :, 1:] == 0.0).all())
    ):
        raise SchemaValidationError(
            "R1 diagnostic applied-action evidence is inconsistent"
        )
    tensors["diagnostic_applied_global_action"] = global_action
    tensors["diagnostic_applied_joint_action"] = joint_action
    tensors["diagnostic_applied_contact_space_action"] = contact_action
    metadata.update(
        {
            "diagnostic_action_ablation": (
                ORDER9_R1_JOINT_COMPRESSION_DIAGNOSTIC_VERSION
            ),
            "diagnostic_joint_action_retained": True,
            "diagnostic_contact_normal_compression_retained": True,
            "diagnostic_centroidal_action_retained": False,
            "diagnostic_contact_tangent_rotation_action_retained": False,
            "diagnostic_mask_wrapper_path": str(wrapper_source),
            "diagnostic_mask_wrapper_sha256": hash_file(wrapper_source),
            "protected_rollout_path": str(protected_source),
            "protected_rollout_sha256": protected_rollout_sha256,
            "promotion_evidence_eligible": False,
            "training_eligible": False,
        }
    )
    _atomic_torch_save(payload, raw_source)
    raw_sha256 = hash_file(raw_source)
    records = [
        json.loads(line)
        for line in episode_source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise SchemaValidationError("R1 diagnostic episode evidence is empty")
    for record in records:
        record["source_artifact_path"] = str(raw_source)
        record["source_artifact_sha256"] = raw_sha256
        episode_metadata = record.setdefault("metadata", {})
        episode_metadata.update(
            {
                "diagnostic_action_ablation": (
                    ORDER9_R1_JOINT_COMPRESSION_DIAGNOSTIC_VERSION
                ),
                "promotion_evidence_eligible": False,
                "training_eligible": False,
            }
        )
    _atomic_text_write(
        episode_source,
        "".join(
            json.dumps(record, sort_keys=True) + "\n" for record in records
        ),
    )


def _atomic_torch_save(payload: object, destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text_write(destination: Path, text: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "ORDER9_R1_JOINT_COMPRESSION_DIAGNOSTIC_VERSION",
    "Order9R1JointCompressionProjectionRecorder",
    "annotate_order9_r1_joint_compression_rollout",
    "mask_order9_r1_joint_and_compression_inputs",
]
