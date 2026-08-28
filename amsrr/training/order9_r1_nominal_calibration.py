from __future__ import annotations

"""Formal R1 calibration through deterministic nominal control without pi_L."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import torch

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.datasets import DatasetSplit
from amsrr.training.order9_c3_promotion import (
    load_order9_evaluation_episode_jsonl,
)
from amsrr.training.order9_pi_l_stage_runner import (
    coalesce_order9_same_morphology_collectors,
    run_parallel_order9_collectors,
)
from amsrr.training.order9_r1_clearance_diagnostic import (
    write_order9_r1_clearance_diagnostic,
)
from amsrr.training.order9_r1_isaac_calibration import (
    MaterializedOrder9R1IsaacCase,
    order9_r1_isaac_case_command,
)
from amsrr.utils.config import load_config
from amsrr.utils.hashing import hash_file

ORDER9_R1_NOMINAL_CALIBRATION_PROTOCOL_VERSION = (
    "order9_r1_nominal_calibration_protocol_v2"
)
ORDER9_R1_NOMINAL_CALIBRATION_APPROVAL_VERSION = (
    "order9_r1_nominal_calibration_approval_record_v2"
)
ORDER9_R1_NOMINAL_EXECUTION_CONTRACT = (
    "deterministic_teacher_ik_nominal_qpid_qp_local_servo_isaac_v1"
)
ORDER9_R1_NOMINAL_GENERATION_PREFIX = "r1_nominal_calibration:"
ORDER9_R1_NOMINAL_CASE_CONTRACT_VERSION = (
    "order9_r1_nominal_calibration_case_contract_v2"
)
ORDER9_R1_NOMINAL_CASE_RESULT_VERSION = (
    "order9_r1_nominal_calibration_case_result_v2"
)
ORDER9_R1_NOMINAL_PROTOCOL_RELATIVE = Path(
    "configs/training/order9_r1_nominal_calibration_protocol_v2.yaml"
)
ORDER9_R1_NOMINAL_APPROVAL_RELATIVE = Path(
    "for_codex/R1_NOMINAL_CALIBRATION_PROTOCOL_V2_APPROVAL.json"
)
_EXPECTED_PROTOCOL_SHA256 = (
    "f6f4a42bc0a3f6297162f25c3a321aa30b2e74f3f389ca7b058a1d209ae158c3"
)
_EXPECTED_APPROVAL_SHA256 = (
    "b6d34d7c6c5814efe86f1f9daeafe5a1cd20a1985bcb90a81c3c9167f5d4515d"
)
_REQUIRED_APPROVAL_SCOPE = {
    "reuse_four_level_numeric_ladder",
    "selection_on_22_train_buckets",
    "single_confirmation_on_14_validation_buckets",
    "fast_screen_before_isaac",
    "nominal_qpid_qp_local_servo_standard_path",
    "pi_l_retained_but_not_applied",
    "two_isaac_replays_per_screen_accepted_candidate",
    "stop_after_first_failed_level",
    "no_teacher_collection_or_learning_by_protocol_alone",
}
_ZERO_ACTION_TENSORS = (
    "applied_global_action",
    "global_action",
    "joint_action",
    "contact_space_residual_action",
    "command_residual_wrench_body",
    "command_joint_torque_bias_nm",
)


@dataclass(frozen=True)
class Order9R1NominalReplayResult:
    candidate_id: str
    replay_index: int
    success: bool
    safety_failure: bool
    fallback_used: bool
    failure_reason: str | None
    execution_chain: str = ORDER9_R1_NOMINAL_EXECUTION_CONTRACT
    deterministic_teacher_used: bool = True
    deterministic_ik_used: bool = True
    nominal_preload_used: bool = True
    pi_l_system_retained: bool = True
    pi_l_actor_command_applied: bool = False
    qpid_qp_used: bool = True
    local_servo_used: bool = True
    isaac_used: bool = True

    def validate(self) -> None:
        if (
            not self.candidate_id
            or self.replay_index < 0
            or self.execution_chain != ORDER9_R1_NOMINAL_EXECUTION_CONTRACT
            or not all(
                (
                    self.deterministic_teacher_used,
                    self.deterministic_ik_used,
                    self.nominal_preload_used,
                    self.pi_l_system_retained,
                    self.qpid_qp_used,
                    self.local_servo_used,
                    self.isaac_used,
                )
            )
            or self.pi_l_actor_command_applied
        ):
            raise SchemaValidationError(
                "Order9 R1 nominal replay contract is invalid"
            )
        if self.success and (
            self.safety_failure or self.fallback_used or self.failure_reason
        ):
            raise SchemaValidationError(
                "successful R1 nominal replay has failure evidence"
            )
        if not self.success and self.failure_reason is None:
            raise SchemaValidationError(
                "failed R1 nominal replay lacks a failure reason"
            )


def load_order9_r1_nominal_calibration_contract(
    repository_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and byte-validate the approved additive nominal protocol."""

    repository = Path(repository_root).resolve()
    protocol_path = repository / ORDER9_R1_NOMINAL_PROTOCOL_RELATIVE
    approval_path = repository / ORDER9_R1_NOMINAL_APPROVAL_RELATIVE
    if hash_file(protocol_path) != _EXPECTED_PROTOCOL_SHA256:
        raise SchemaValidationError("R1 nominal protocol bytes changed")
    if hash_file(approval_path) != _EXPECTED_APPROVAL_SHA256:
        raise SchemaValidationError("R1 nominal approval bytes changed")
    protocol = load_config(protocol_path)
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if (
        protocol.get("protocol_version")
        != ORDER9_R1_NOMINAL_CALIBRATION_PROTOCOL_VERSION
        or protocol.get("calibration_gate_id")
        != "r1-nominal-calibration-gate-v2"
        or protocol.get("status") != "approved"
        or protocol.get("execution_contract")
        != ORDER9_R1_NOMINAL_EXECUTION_CONTRACT
        or protocol.get("pi_l_system_retained") is not True
        or protocol.get("pi_l_actor_command_applied") is not False
        or protocol.get("isaac_required_after_fast_screen") is not True
    ):
        raise SchemaValidationError("R1 nominal protocol contract is invalid")
    if (
        approval.get("record_version")
        != ORDER9_R1_NOMINAL_CALIBRATION_APPROVAL_VERSION
        or approval.get("decision") != "approved"
        or approval.get("calibration_gate_id")
        != protocol["calibration_gate_id"]
        or set(approval.get("approval_scope", ())) != _REQUIRED_APPROVAL_SCOPE
        or approval.get("approved_protocol", {}).get("sha256")
        != _EXPECTED_PROTOCOL_SHA256
    ):
        raise SchemaValidationError("R1 nominal approval record is invalid")
    for key in (
        "base_numeric_protocol",
        "base_numeric_approval",
        "protected_c3_curriculum",
        "protected_c3_checkpoint",
        "protected_c3_rollout",
        "protected_c3_release_ledger",
    ):
        binding = protocol.get(key)
        if not isinstance(binding, dict):
            raise SchemaValidationError(
                f"R1 nominal binding is missing: {key}"
            )
        source = repository / str(binding.get("path", ""))
        if not source.is_file() or hash_file(source) != binding.get("sha256"):
            raise SchemaValidationError(f"R1 nominal binding changed: {key}")
    if hash_file(approval_path) != hash_file(
        repository / str(protocol["approval_record"])
    ):
        raise SchemaValidationError("R1 nominal approval path differs")
    return protocol, approval


def order9_r1_nominal_case_command(
    materialized: MaterializedOrder9R1IsaacCase,
    *,
    repository_root: str | Path,
    python_executable: str | Path,
    rollout_steps: int = 15000,
) -> tuple[list[str], Path]:
    """Build the protected rollout command with all pi_L commands bypassed."""

    command, log_path = order9_r1_isaac_case_command(
        materialized,
        repository_root=repository_root,
        python_executable=python_executable,
        rollout_steps=rollout_steps,
    )
    generation_id = (
        ORDER9_R1_NOMINAL_GENERATION_PREFIX
        + materialized.manifest.candidate_id
    )
    _replace_option(command, "--generation-id", generation_id)
    if "--diagnostic-nominal-qpid-only" in command:
        raise SchemaValidationError("R1 nominal command duplicated its bypass")
    command.append("--diagnostic-nominal-qpid-only")
    return command, log_path


def run_order9_r1_nominal_isaac_cases(
    cases: Sequence[MaterializedOrder9R1IsaacCase],
    *,
    repository_root: str | Path,
    python_executable: str | Path = (
        "/home/leus/.local/share/mamba/envs/isaaclab3/bin/python"
    ),
    rollout_steps: int = 15000,
    maximum_parallel_process_count: int = 2,
    persistent_morphology_coalescing: bool = True,
) -> dict[str, tuple[Order9R1NominalReplayResult, ...]]:
    """Run or resume nominal cases, then annotate and validate their evidence."""

    if not cases:
        return {}
    repository = Path(repository_root).resolve()
    load_order9_r1_nominal_calibration_contract(repository)
    commands: dict[str, list[str]] = {}
    logs: dict[str, Path] = {}
    morphology: dict[str, str] = {}
    complete: dict[str, tuple[Order9R1NominalReplayResult, ...]] = {}
    for case in cases:
        try:
            complete[case.manifest.candidate_id] = (
                validate_order9_r1_nominal_isaac_result(
                    case,
                    repository_root=repository,
                )
            )
            continue
        except (OSError, ValueError, SchemaValidationError):
            pass
        command, log = order9_r1_nominal_case_command(
            case,
            repository_root=repository,
            python_executable=python_executable,
            rollout_steps=rollout_steps,
        )
        name = case.manifest.candidate_id
        commands[name] = command
        logs[name] = log
        morphology[name] = case.manifest.morphology_hash
    if commands:
        if persistent_morphology_coalescing:
            commands, logs, _members = (
                coalesce_order9_same_morphology_collectors(
                    commands,
                    log_paths=logs,
                    morphology_hash_by_name=morphology,
                    batch_root=cases[0].manifest_path.parents[2]
                    / "persistent_process_batches",
                )
            )
        run_parallel_order9_collectors(
            commands,
            repository_root=repository,
            log_paths=logs,
            maximum_parallel_process_count=maximum_parallel_process_count,
            process_start_stagger_s=3.0,
        )
    for case in cases:
        annotate_order9_r1_nominal_isaac_result(
            case,
            repository_root=repository,
        )
        complete[case.manifest.candidate_id] = (
            validate_order9_r1_nominal_isaac_result(
                case,
                repository_root=repository,
            )
        )
        write_order9_r1_nominal_case_result(
            case,
            complete[case.manifest.candidate_id],
            repository_root=repository,
        )
    return complete


def write_order9_r1_nominal_case_contract(
    materialized: MaterializedOrder9R1IsaacCase,
    *,
    repository_root: str | Path,
) -> Path:
    repository = Path(repository_root).resolve()
    protocol, _approval = load_order9_r1_nominal_calibration_contract(
        repository
    )
    destination = materialized.manifest_path.parent / "nominal_contract.json"
    payload = {
        "contract_version": ORDER9_R1_NOMINAL_CASE_CONTRACT_VERSION,
        "candidate_id": materialized.manifest.candidate_id,
        "execution_contract": ORDER9_R1_NOMINAL_EXECUTION_CONTRACT,
        "pi_l_system_retained": True,
        "pi_l_actor_command_applied": False,
        "nominal_preload_applied": True,
        "qpid_qp_applied": True,
        "local_servo_applied": True,
        "isaac_required": True,
        "c3_promotion_evidence_eligible": False,
        "training_eligible": False,
        "formal_teacher_collection_authorized": False,
        "bindings": {
            "nominal_protocol": _binding(
                repository / ORDER9_R1_NOMINAL_PROTOCOL_RELATIVE
            ),
            "nominal_approval": _binding(
                repository / ORDER9_R1_NOMINAL_APPROVAL_RELATIVE
            ),
            "base_numeric_protocol": _binding(
                repository / protocol["base_numeric_protocol"]["path"]
            ),
            "case_manifest": _binding(materialized.manifest_path),
            "nominal_runtime_implementation": _binding(Path(__file__)),
        },
    }
    return write_order9_r1_clearance_diagnostic(payload, destination)


def annotate_order9_r1_nominal_isaac_result(
    materialized: MaterializedOrder9R1IsaacCase,
    *,
    repository_root: str | Path,
) -> None:
    """Mark protected diagnostic output as formal R1 nominal evidence."""

    repository = Path(repository_root).resolve()
    protocol, _approval = load_order9_r1_nominal_calibration_contract(
        repository
    )
    root = materialized.manifest_path.parent / "isaac"
    raw_path = root / "evaluation_rollout.pt"
    episodes_path = root / "evaluation_episodes.jsonl"
    payload = torch.load(raw_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise SchemaValidationError("R1 nominal raw artifact is invalid")
    metadata = payload.get("metadata")
    tensors = payload.get("tensors")
    if not isinstance(metadata, dict) or not isinstance(tensors, dict):
        raise SchemaValidationError("R1 nominal raw artifact is incomplete")
    _require_zero_policy_actions(tensors)
    expected_generation = (
        ORDER9_R1_NOMINAL_GENERATION_PREFIX
        + materialized.manifest.candidate_id
    )
    if (
        metadata.get("diagnostic_nominal_qpid_only") is not True
        or metadata.get("generation_id") != expected_generation
        or metadata.get("formal_phase_zero_start") is not True
    ):
        raise SchemaValidationError("R1 nominal raw source contract differs")
    annotation = {
        "r1_nominal_calibration_contract": (
            ORDER9_R1_NOMINAL_EXECUTION_CONTRACT
        ),
        "r1_nominal_calibration_protocol_sha256": _EXPECTED_PROTOCOL_SHA256,
        "r1_nominal_calibration_approval_sha256": hash_file(
            repository / ORDER9_R1_NOMINAL_APPROVAL_RELATIVE
        ),
        "r1_nominal_runtime_implementation_sha256": hash_file(Path(__file__)),
        "pi_l_system_retained": True,
        "pi_l_actor_command_applied": False,
        "nominal_preload_applied": True,
        "promotion_evidence_eligible": False,
        "training_eligible": False,
        "formal_teacher_collection_authorized": False,
    }
    already_annotated = all(
        metadata.get(key) == value for key, value in annotation.items()
    )
    records = [
        json.loads(line)
        for line in episodes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if already_annotated:
        raw_sha = hash_file(raw_path)
        if all(
            record.get("source_artifact_sha256") == raw_sha
            and all(
                record.get("metadata", {}).get(key) == value
                for key, value in annotation.items()
            )
            for record in records
        ):
            return
    metadata.update(annotation)
    _atomic_torch_save(payload, raw_path)
    raw_sha = hash_file(raw_path)
    if not records:
        raise SchemaValidationError("R1 nominal episode evidence is empty")
    for record in records:
        record["source_artifact_path"] = str(raw_path.resolve())
        record["source_artifact_sha256"] = raw_sha
        record.setdefault("metadata", {}).update(annotation)
    _atomic_text_write(
        episodes_path,
        "".join(
            json.dumps(record, sort_keys=True) + "\n" for record in records
        ),
    )


def validate_order9_r1_nominal_isaac_result(
    materialized: MaterializedOrder9R1IsaacCase,
    *,
    repository_root: str | Path,
) -> tuple[Order9R1NominalReplayResult, ...]:
    repository = Path(repository_root).resolve()
    load_order9_r1_nominal_calibration_contract(repository)
    root = materialized.manifest_path.parent / "isaac"
    raw_path = root / "evaluation_rollout.pt"
    episodes_path = root / "evaluation_episodes.jsonl"
    payload = torch.load(raw_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise SchemaValidationError("R1 nominal raw artifact is invalid")
    metadata = payload.get("metadata")
    tensors = payload.get("tensors")
    expected_generation = (
        ORDER9_R1_NOMINAL_GENERATION_PREFIX
        + materialized.manifest.candidate_id
    )
    if not isinstance(metadata, dict) or not isinstance(tensors, dict):
        raise SchemaValidationError("R1 nominal raw artifact is incomplete")
    _require_zero_policy_actions(tensors)
    if (
        metadata.get("diagnostic_nominal_qpid_only") is not True
        or metadata.get("r1_nominal_calibration_contract")
        != ORDER9_R1_NOMINAL_EXECUTION_CONTRACT
        or metadata.get("pi_l_actor_command_applied") is not False
        or metadata.get("promotion_evidence_eligible") is not False
        or metadata.get("training_eligible") is not False
        or metadata.get("generation_id") != expected_generation
        or metadata.get("formal_phase_zero_start") is not True
    ):
        raise SchemaValidationError("R1 nominal annotated contract differs")
    episodes = load_order9_evaluation_episode_jsonl(episodes_path)
    if len(episodes) != materialized.manifest.replay_count:
        raise SchemaValidationError("R1 nominal replay count differs")
    raw_sha = hash_file(raw_path)
    results = []
    for replay_index, episode in enumerate(episodes):
        if (
            episode.split != DatasetSplit.VALIDATION
            or episode.metadata.get("generation_id") != expected_generation
            or episode.metadata.get("pi_l_actor_command_applied") is not False
            or episode.metadata.get("r1_nominal_calibration_contract")
            != ORDER9_R1_NOMINAL_EXECUTION_CONTRACT
            or episode.source_artifact_sha256 != raw_sha
            or Path(episode.source_artifact_path).resolve()
            != raw_path.resolve()
        ):
            raise SchemaValidationError("R1 nominal episode binding differs")
        success = bool(
            episode.task_success
            and episode.no_fallback_success
            and not episode.safety_failure
            and episode.fallback_decision_count == 0
            and episode.isaac_backed
            and episode.full_mesh_evaluation
        )
        result = Order9R1NominalReplayResult(
            candidate_id=materialized.manifest.candidate_id,
            replay_index=replay_index,
            success=success,
            safety_failure=episode.safety_failure,
            fallback_used=episode.fallback_decision_count > 0,
            failure_reason=(
                None if success else (episode.failure_reason or "task failed")
            ),
        )
        result.validate()
        results.append(result)
    return tuple(results)


def write_order9_r1_nominal_case_result(
    materialized: MaterializedOrder9R1IsaacCase,
    results: Sequence[Order9R1NominalReplayResult],
    *,
    repository_root: str | Path,
) -> Path:
    repository = Path(repository_root).resolve()
    root = materialized.manifest_path.parent
    raw_path = root / "isaac/evaluation_rollout.pt"
    episodes_path = root / "isaac/evaluation_episodes.jsonl"
    log_path = root / "isaac/evaluation.log"
    payload = {
        "result_version": ORDER9_R1_NOMINAL_CASE_RESULT_VERSION,
        "candidate_id": materialized.manifest.candidate_id,
        "execution_contract": ORDER9_R1_NOMINAL_EXECUTION_CONTRACT,
        "status": (
            "accepted"
            if all(value.success for value in results)
            else "rejected"
        ),
        "episode_count": len(results),
        "success_count": sum(value.success for value in results),
        "safety_failure_count": sum(value.safety_failure for value in results),
        "fallback_count": sum(value.fallback_used for value in results),
        "failure_reasons": [value.failure_reason for value in results],
        "pi_l_system_retained": True,
        "pi_l_actor_command_applied": False,
        "c3_promotion_evidence_eligible": False,
        "training_eligible": False,
        "formal_teacher_collection_authorized": False,
        "bindings": {
            "nominal_protocol": _binding(
                repository / ORDER9_R1_NOMINAL_PROTOCOL_RELATIVE
            ),
            "nominal_approval": _binding(
                repository / ORDER9_R1_NOMINAL_APPROVAL_RELATIVE
            ),
            "case_manifest": _binding(materialized.manifest_path),
            "nominal_contract": _binding(root / "nominal_contract.json"),
            "raw_rollout": _binding(raw_path),
            "episodes": _binding(episodes_path),
            "log": _binding(log_path),
            "nominal_runtime_implementation": _binding(Path(__file__)),
        },
    }
    return write_order9_r1_clearance_diagnostic(
        payload, root / "nominal_result.json"
    )


def _require_zero_policy_actions(tensors: Mapping[str, Any]) -> None:
    for name in _ZERO_ACTION_TENSORS:
        value = tensors.get(name)
        if not isinstance(value, torch.Tensor) or not bool(
            (value == 0.0).all()
        ):
            raise SchemaValidationError(
                f"R1 nominal execution applied a pi_L action: {name}"
            )


def _replace_option(command: list[str], option: str, value: str) -> None:
    matches = [index for index, item in enumerate(command) if item == option]
    if len(matches) != 1 or matches[0] + 1 >= len(command):
        raise SchemaValidationError(
            f"R1 nominal command option differs: {option}"
        )
    command[matches[0] + 1] = value


def _binding(path: Path) -> dict[str, str]:
    source = path.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {"path": str(source), "sha256": hash_file(source)}


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
    "ORDER9_R1_NOMINAL_EXECUTION_CONTRACT",
    "ORDER9_R1_NOMINAL_GENERATION_PREFIX",
    "Order9R1NominalReplayResult",
    "annotate_order9_r1_nominal_isaac_result",
    "load_order9_r1_nominal_calibration_contract",
    "order9_r1_nominal_case_command",
    "run_order9_r1_nominal_isaac_cases",
    "validate_order9_r1_nominal_isaac_result",
    "write_order9_r1_nominal_case_contract",
    "write_order9_r1_nominal_case_result",
]
