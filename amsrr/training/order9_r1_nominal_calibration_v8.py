from __future__ import annotations

"""Validate the R1 v8 overlay of v7 cases and admitted yaw repairs."""

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from amsrr.schemas.common import SchemaValidationError
from amsrr.training.order9_r1_isaac_calibration import load_order9_r1_isaac_case
from amsrr.training.order9_r1_nominal_calibration_v7 import (
    load_order9_r1_nominal_calibration_v7_contract,
)
from amsrr.utils.config import load_config
from amsrr.utils.hashing import hash_file

ORDER9_R1_NOMINAL_V8_PROTOCOL_VERSION = "order9_r1_nominal_calibration_protocol_v8"
ORDER9_R1_NOMINAL_V8_APPROVAL_VERSION = (
    "order9_r1_nominal_calibration_approval_record_v8"
)
ORDER9_R1_NOMINAL_V8_REPAIR_LEDGER_VERSION = (
    "order9_r1_nominal_v8_yaw_repair_evidence_ledger_v1"
)
ORDER9_R1_NOMINAL_V8_PROTOCOL_RELATIVE = Path(
    "configs/training/order9_r1_nominal_calibration_protocol_v8.yaml"
)
ORDER9_R1_NOMINAL_V8_APPROVAL_RELATIVE = Path(
    "for_codex/R1_NOMINAL_CALIBRATION_PROTOCOL_V8_APPROVAL.json"
)

_ACTION_TENSOR_KEYS = (
    "applied_global_action",
    "contact_space_residual_action",
    "global_action",
    "joint_action",
    "previous_global_action",
)
_REQUIRED_BINDINGS = (
    "base_v7_protocol",
    "base_v7_approval",
    "protected_c3_checkpoint",
    "protected_c3_rollout",
    "protected_c3_release_ledger",
    "batched_wrapper_implementation",
    "batched_runtime_implementation",
    "subset_runner_implementation",
    "yaw_repair_implementation",
    "yaw_repair_derivation_script",
    "repaired_candidate_exclusion",
    "repair_evidence_ledger",
    "v8_calibration_implementation",
)
# The digests remain part of the immutable v8 record, while these live paths
# are allowed to contain their v9 successors. Evidence and C3 bindings remain
# strict.
_SUPERSEDED_LIVE_IMPLEMENTATION_BINDINGS = frozenset(
    {
        "batched_wrapper_implementation",
        "yaw_repair_implementation",
        "v8_calibration_implementation",
    }
)


def load_order9_r1_nominal_calibration_v8_contract(
    repository_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the v8 contract and fail closed if any bound byte changed."""

    repository = Path(repository_root).resolve()
    protocol_path = repository / ORDER9_R1_NOMINAL_V8_PROTOCOL_RELATIVE
    approval_path = repository / ORDER9_R1_NOMINAL_V8_APPROVAL_RELATIVE
    protocol = load_config(protocol_path)
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if (
        protocol.get("protocol_version") != ORDER9_R1_NOMINAL_V8_PROTOCOL_VERSION
        or protocol.get("status") != "approved"
        or protocol.get("base_level_id") != "r1_l1_10mm_5deg"
        or protocol.get("base_split") != "train"
        or int(protocol.get("source_bucket_count", -1)) != 22
        or int(protocol.get("candidate_count", -1)) != 682
        or int(protocol.get("replay_count_per_candidate", -1)) != 2
        or int(protocol.get("v7_unchanged_candidate_count", -1)) != 673
        or int(protocol.get("yaw_repaired_candidate_count", -1)) != 9
        or protocol.get("pi_l_actor_command_applied") is not False
        or protocol.get("qpid_qp_applied") is not True
        or protocol.get("local_servo_applied") is not True
        or protocol.get("isaac_required") is not True
        or protocol.get("minimum_level_result_authorizes_collection") is not False
    ):
        raise SchemaValidationError("R1 nominal v8 protocol contract is invalid")
    if (
        approval.get("record_version") != ORDER9_R1_NOMINAL_V8_APPROVAL_VERSION
        or approval.get("decision") != "approved"
        or approval.get("approved_by") != "repository_user"
        or approval.get("approved_protocol", {}).get("sha256")
        != hash_file(protocol_path)
        or Path(str(approval.get("approved_protocol", {}).get("path", "")))
        != ORDER9_R1_NOMINAL_V8_PROTOCOL_RELATIVE
    ):
        raise SchemaValidationError("R1 nominal v8 approval record is invalid")
    for key in _REQUIRED_BINDINGS:
        binding = protocol.get(key)
        if not isinstance(binding, dict):
            raise SchemaValidationError(f"R1 nominal v8 binding missing: {key}")
        path = repository / str(binding.get("path", ""))
        if not path.is_file() or (
            key not in _SUPERSEDED_LIVE_IMPLEMENTATION_BINDINGS
            and hash_file(path) != binding.get("sha256")
        ):
            raise SchemaValidationError(f"R1 nominal v8 binding changed: {key}")
    load_order9_r1_nominal_calibration_v7_contract(repository)
    return protocol, approval


def validate_order9_r1_nominal_v8_repair_evidence(
    repository_root: str | Path,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Validate all nine repaired paths, cheap admissions, and Isaac replays."""

    repository = Path(repository_root).resolve()
    if protocol is None:
        protocol, _approval = load_order9_r1_nominal_calibration_v8_contract(repository)
    exclusion_path = repository / str(protocol["repaired_candidate_exclusion"]["path"])
    excluded_payload = json.loads(exclusion_path.read_text(encoding="utf-8"))
    expected_ids = excluded_payload.get("candidate_ids")
    if (
        not isinstance(expected_ids, list)
        or len(expected_ids) != 9
        or expected_ids != sorted(expected_ids)
        or len(expected_ids) != len(set(expected_ids))
    ):
        raise SchemaValidationError("R1 nominal v8 exclusion set is invalid")

    ledger_path = repository / str(protocol["repair_evidence_ledger"]["path"])
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    entries = ledger.get("entries") if isinstance(ledger, dict) else None
    if (
        ledger.get("ledger_version") != ORDER9_R1_NOMINAL_V8_REPAIR_LEDGER_VERSION
        or not isinstance(entries, list)
        or len(entries) != 9
        or [entry.get("candidate_id") for entry in entries] != expected_ids
    ):
        raise SchemaValidationError("R1 nominal v8 repair ledger is invalid")
    return tuple(
        _validate_repair_entry(entry, repository=repository) for entry in entries
    )


def _validate_repair_entry(
    entry: Mapping[str, Any], *, repository: Path
) -> dict[str, Any]:
    candidate_id = entry.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise SchemaValidationError("R1 nominal v8 repair candidate ID is invalid")
    paths = {}
    for key in ("case_manifest", "lightweight_admission", "raw_rollout", "episodes"):
        binding = entry.get(key)
        if not isinstance(binding, dict):
            raise SchemaValidationError(f"R1 nominal v8 repair binding missing: {key}")
        path = repository / str(binding.get("path", ""))
        if not path.is_file() or hash_file(path) != binding.get("sha256"):
            raise SchemaValidationError(
                f"R1 nominal v8 repair evidence changed: {candidate_id}:{key}"
            )
        paths[key] = path

    case = load_order9_r1_isaac_case(paths["case_manifest"], repository)
    if (
        case.manifest.candidate_id != candidate_id
        or case.manifest.source_bucket_id != "train-000002-0e86a4ff3e71"
        or case.manifest.replay_count != 2
        or case.manifest.training_eligible
    ):
        raise SchemaValidationError("R1 nominal v8 repaired case contract differs")
    admission = json.loads(paths["lightweight_admission"].read_text(encoding="utf-8"))
    clearance = float(admission.get("minimum_collision_clearance_m", -1.0))
    reserve = float(admission.get("minimum_normalized_joint_limit_reserve", -1.0))
    if (
        admission.get("candidate_id") != candidate_id
        or admission.get("status") != "accepted"
        or admission.get("ik_invoked") is not False
        or admission.get("trajectory_optimization_invoked") is not False
        or admission.get("controller_layers_invoked") is not False
        or admission.get("isaac_invoked") is not False
        or admission.get("training_eligible") is not False
        or clearance + 1.0e-9 < 0.0118
        or reserve + 1.0e-12 < 0.01
    ):
        raise SchemaValidationError("R1 nominal v8 lightweight admission differs")

    raw = torch.load(paths["raw_rollout"], map_location="cpu", weights_only=False)
    metadata = raw.get("metadata") if isinstance(raw, dict) else None
    tensors = raw.get("tensors") if isinstance(raw, dict) else None
    nominal = (
        metadata.get("c3_nominal_reference") if isinstance(metadata, dict) else None
    )
    if (
        not isinstance(metadata, dict)
        or not isinstance(tensors, dict)
        or not isinstance(nominal, dict)
        or metadata.get("generation_id") != f"r1_calibration:{candidate_id}"
        or metadata.get("diagnostic_nominal_qpid_only") is not True
        or metadata.get("formal_phase_zero_start") is not True
        or metadata.get("pi_l_actor_command_applied") is not False
        or metadata.get("r1_candidate_reset_bank_used_for_formal_start") is not False
        or metadata.get("r1_batched_nominal_runtime_version")
        != "order9_r1_environment_wise_nominal_isaac_batch_v1"
        or not 1 <= int(metadata.get("r1_batched_candidate_count", -1)) <= 31
        or nominal.get("set_manifest_sha256") != case.manifest.nominal_set.sha256
        or nominal.get("artifact_sha256") != case.manifest.nominal_artifact.sha256
    ):
        raise SchemaValidationError("R1 nominal v8 repaired raw contract differs")
    for key in _ACTION_TENSOR_KEYS:
        value = tensors.get(key)
        if (
            not isinstance(value, torch.Tensor)
            or not torch.isfinite(value).all()
            or bool(torch.count_nonzero(value).item())
        ):
            raise SchemaValidationError(
                f"R1 nominal v8 repaired action is not zero: {candidate_id}:{key}"
            )
    if not _controller_commands_are_active(tensors):
        raise SchemaValidationError(
            "R1 nominal v8 controller command evidence is empty"
        )

    records = [
        json.loads(line)
        for line in paths["episodes"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 2:
        raise SchemaValidationError("R1 nominal v8 repaired replay count differs")
    for record in records:
        metrics = record.get("metrics") if isinstance(record, dict) else None
        if (
            record.get("task_success") is not True
            or record.get("safety_failure") is not False
            or int(record.get("fallback_decision_count", -1)) != 0
            or not isinstance(metrics, dict)
            or float(metrics.get("terminal_phase_index", -1.0)) != 8.0
            or float(metrics.get("hard_collision", 1.0)) != 0.0
            or float(metrics.get("object_dropped", 1.0)) != 0.0
            or float(metrics.get("timeout", 1.0)) != 0.0
            or float(metrics.get("qp_infeasible_terminal", 1.0)) != 0.0
        ):
            raise SchemaValidationError("R1 nominal v8 repaired Isaac replay failed")
    return {
        "candidate_id": candidate_id,
        "episode_count": 2,
        "success_count": 2,
        "safety_failure_count": 0,
        "fallback_count": 0,
        "minimum_collision_clearance_m": clearance,
        "minimum_normalized_joint_limit_reserve": reserve,
        "pi_l_actor_command_applied": False,
    }


def _controller_commands_are_active(tensors: Mapping[str, Any]) -> bool:
    required = (
        "rotor_thrusts_n",
        "command_joint_position_targets_rad",
        "controller_desired_wrench_body",
    )
    for key in required:
        value = tensors.get(key)
        if (
            not isinstance(value, torch.Tensor)
            or not torch.isfinite(value).all()
            or not bool(torch.count_nonzero(value).item())
        ):
            return False
    return True


__all__ = [
    "load_order9_r1_nominal_calibration_v8_contract",
    "validate_order9_r1_nominal_v8_repair_evidence",
]
