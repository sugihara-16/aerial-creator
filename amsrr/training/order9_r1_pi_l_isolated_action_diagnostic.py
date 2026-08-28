from __future__ import annotations

"""R1 diagnostics that retain exactly one pi_L correction path."""

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Literal

import torch

from amsrr.schemas.common import SchemaValidationError
from amsrr.utils.hashing import hash_file

Order9R1IsolatedActionMode = Literal["joint_only", "contact_normal_only"]

ORDER9_R1_ISOLATED_ACTION_MODES: tuple[Order9R1IsolatedActionMode, ...] = (
    "joint_only",
    "contact_normal_only",
)
ORDER9_R1_ISOLATED_ACTION_DIAGNOSTIC_VERSION = (
    "order9_r1_pi_l_isolated_action_v1"
)


def mask_order9_r1_isolated_action_inputs(
    *,
    mode: Order9R1IsolatedActionMode,
    normalized_global_action: torch.Tensor,
    normalized_joint_action: torch.Tensor,
    normalized_contact_action: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Retain direct joint action or contact-normal action, never both."""

    if mode not in ORDER9_R1_ISOLATED_ACTION_MODES:
        raise ValueError(f"unsupported R1 isolated action mode: {mode}")
    if normalized_global_action.ndim != 2:
        raise ValueError(
            "R1 isolated diagnostic global action shape is invalid"
        )
    if normalized_joint_action.ndim != 3:
        raise ValueError(
            "R1 isolated diagnostic joint action shape is invalid"
        )
    if (
        normalized_contact_action.ndim != 3
        or normalized_contact_action.shape[-1] < 1
    ):
        raise ValueError(
            "R1 isolated diagnostic contact action shape is invalid"
        )
    batch = normalized_global_action.shape[0]
    if (
        normalized_joint_action.shape[0] != batch
        or normalized_contact_action.shape[0] != batch
    ):
        raise ValueError("R1 isolated diagnostic action batches differ")
    for value in (
        normalized_global_action,
        normalized_joint_action,
        normalized_contact_action,
    ):
        if not bool(torch.isfinite(value).all()):
            raise ValueError("R1 isolated diagnostic action is non-finite")

    global_action = torch.zeros_like(normalized_global_action)
    joint_action = torch.zeros_like(normalized_joint_action)
    contact_action = torch.zeros_like(normalized_contact_action)
    if mode == "joint_only":
        joint_action = normalized_joint_action
    else:
        contact_action[:, :, 0] = normalized_contact_action[:, :, 0]
    return global_action, joint_action, contact_action


class Order9R1IsolatedProjectionRecorder:
    """Mask projection inputs and retain exact input/output evidence."""

    def __init__(
        self,
        original: Callable[..., Any],
        *,
        mode: Order9R1IsolatedActionMode,
    ) -> None:
        if mode not in ORDER9_R1_ISOLATED_ACTION_MODES:
            raise ValueError(f"unsupported R1 isolated action mode: {mode}")
        self.original = original
        self.mode = mode
        self.masked_direct_joint_actions: list[torch.Tensor] = []
        self.applied_global_actions: list[torch.Tensor] = []
        self.applied_joint_actions: list[torch.Tensor] = []
        self.applied_contact_actions: list[torch.Tensor] = []

    def __call__(self, **values: Any) -> Any:
        global_action, joint_action, contact_action = (
            mask_order9_r1_isolated_action_inputs(
                mode=self.mode,
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
                "R1 isolated diagnostic leaked a global action"
            )
        self.masked_direct_joint_actions.append(_cpu_clone(joint_action))
        self.applied_global_actions.append(_cpu_clone(result.global_action))
        self.applied_joint_actions.append(_cpu_clone(result.joint_action))
        self.applied_contact_actions.append(_cpu_clone(contact_action))
        return result

    def stacked(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.applied_global_actions:
            raise SchemaValidationError(
                "R1 isolated diagnostic recorded no policy steps"
            )
        return (
            torch.stack(self.masked_direct_joint_actions),
            torch.stack(self.applied_global_actions),
            torch.stack(self.applied_joint_actions),
            torch.stack(self.applied_contact_actions),
        )


def annotate_order9_r1_isolated_action_rollout(
    *,
    raw_path: str | Path,
    episode_path: str | Path,
    recorder: Order9R1IsolatedProjectionRecorder,
    wrapper_path: str | Path,
    protected_rollout_path: str | Path,
    protected_rollout_sha256: str,
) -> None:
    """Bind an isolated action mask to raw and episode evidence."""

    raw_source = Path(raw_path).resolve()
    episode_source = Path(episode_path).resolve()
    wrapper_source = Path(wrapper_path).resolve()
    protected_source = Path(protected_rollout_path).resolve()
    if hash_file(protected_source) != protected_rollout_sha256:
        raise SchemaValidationError("protected C3 rollout bytes changed")
    payload = torch.load(raw_source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise SchemaValidationError("R1 isolated raw artifact is invalid")
    tensors = payload.get("tensors")
    metadata = payload.get("metadata")
    if not isinstance(tensors, dict) or not isinstance(metadata, dict):
        raise SchemaValidationError("R1 isolated raw artifact is incomplete")
    direct_joint, global_action, joint_action, contact_action = (
        recorder.stacked()
    )
    step_count = int(tensors["time_s"].shape[0])
    if any(
        value.shape[0] != step_count
        for value in (
            direct_joint,
            global_action,
            joint_action,
            contact_action,
        )
    ):
        raise SchemaValidationError(
            "R1 isolated applied-action evidence length differs"
        )
    if not bool((global_action == 0.0).all()) or not bool(
        (contact_action[:, :, :, 1:] == 0.0).all()
    ):
        raise SchemaValidationError(
            "R1 isolated action evidence contains a disabled action"
        )
    if recorder.mode == "joint_only":
        if not bool((contact_action == 0.0).all()):
            raise SchemaValidationError(
                "joint-only diagnostic retained a contact action"
            )
    elif not bool((direct_joint == 0.0).all()):
        raise SchemaValidationError(
            "contact-normal-only diagnostic retained a direct joint action"
        )

    tensors["diagnostic_masked_direct_joint_action"] = direct_joint
    tensors["diagnostic_applied_global_action"] = global_action
    tensors["diagnostic_applied_joint_action"] = joint_action
    tensors["diagnostic_applied_contact_space_action"] = contact_action
    diagnostic_contract = (
        f"{ORDER9_R1_ISOLATED_ACTION_DIAGNOSTIC_VERSION}:{recorder.mode}"
    )
    metadata.update(
        {
            "diagnostic_action_ablation": diagnostic_contract,
            "diagnostic_isolated_action_mode": recorder.mode,
            "diagnostic_joint_action_retained": recorder.mode == "joint_only",
            "diagnostic_contact_normal_compression_retained": (
                recorder.mode == "contact_normal_only"
            ),
            "diagnostic_centroidal_action_retained": False,
            "diagnostic_contact_tangent_rotation_action_retained": False,
            "diagnostic_direct_torque_action_retained": False,
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
        raise SchemaValidationError("R1 isolated episode evidence is empty")
    for record in records:
        record["source_artifact_path"] = str(raw_source)
        record["source_artifact_sha256"] = raw_sha256
        episode_metadata = record.setdefault("metadata", {})
        episode_metadata.update(
            {
                "diagnostic_action_ablation": diagnostic_contract,
                "diagnostic_isolated_action_mode": recorder.mode,
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


def protected_rollout_source_without_process_exit(source: str) -> str:
    """Delay the protected rollout's application close without editing it."""

    original_tail = (
        "finally:\n"
        "    simulation_app.close()\n"
        "raise SystemExit(_exit_code)\n"
    )
    replacement_tail = "finally:\n    pass\n"
    if source.count(original_tail) != 1 or not source.endswith(original_tail):
        raise SchemaValidationError(
            "protected C3 rollout finalizer no longer matches the bound form"
        )
    return source[: -len(original_tail)] + replacement_tail


def _cpu_clone(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu").clone()


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
    "ORDER9_R1_ISOLATED_ACTION_DIAGNOSTIC_VERSION",
    "ORDER9_R1_ISOLATED_ACTION_MODES",
    "Order9R1IsolatedActionMode",
    "Order9R1IsolatedProjectionRecorder",
    "annotate_order9_r1_isolated_action_rollout",
    "mask_order9_r1_isolated_action_inputs",
    "protected_rollout_source_without_process_exit",
]
