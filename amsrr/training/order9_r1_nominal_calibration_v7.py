from __future__ import annotations

"""Formal R1 minimum-level replay with a bucket-local nominal compression fix."""

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from amsrr.schemas.common import SchemaValidationError
from amsrr.training.order9_pi_l_stage_runner import (
    ORDER9_PERSISTENT_BUCKET_JOBS_VERSION,
    run_parallel_order9_collectors,
)
from amsrr.training.order9_r1_clearance_diagnostic import (
    write_order9_r1_clearance_diagnostic,
)
from amsrr.training.order9_r1_isaac_calibration import MaterializedOrder9R1IsaacCase
from amsrr.training.order9_r1_nominal_calibration import (
    Order9R1NominalReplayResult,
    annotate_order9_r1_nominal_isaac_result,
    order9_r1_nominal_case_command,
    validate_order9_r1_nominal_isaac_result,
    write_order9_r1_nominal_case_contract,
    write_order9_r1_nominal_case_result,
)
from amsrr.training.order9_r1_nominal_calibration_v6 import (
    load_order9_r1_nominal_calibration_v6_contract,
)
from amsrr.utils.config import load_config
from amsrr.utils.hashing import hash_file

ORDER9_R1_NOMINAL_V7_PROTOCOL_VERSION = "order9_r1_nominal_calibration_protocol_v7"
ORDER9_R1_NOMINAL_V7_APPROVAL_VERSION = (
    "order9_r1_nominal_calibration_approval_record_v7"
)
ORDER9_R1_NOMINAL_V7_EXECUTION_CONTRACT = (
    "deterministic_teacher_v6_path_bucket_local_5mm_nominal_compression_"
    "qpid_qp_local_servo_isaac_v7"
)
ORDER9_R1_NOMINAL_V7_CASE_CONTRACT_VERSION = (
    "order9_r1_nominal_calibration_case_contract_v7"
)
ORDER9_R1_NOMINAL_V7_CASE_RESULT_VERSION = (
    "order9_r1_nominal_calibration_case_result_v7"
)
ORDER9_R1_NOMINAL_V7_PROTOCOL_RELATIVE = Path(
    "configs/training/order9_r1_nominal_calibration_protocol_v7.yaml"
)
ORDER9_R1_NOMINAL_V7_APPROVAL_RELATIVE = Path(
    "for_codex/R1_NOMINAL_CALIBRATION_PROTOCOL_V7_APPROVAL.json"
)
ORDER9_R1_NOMINAL_COMPRESSION_WRAPPER_RELATIVE = Path(
    "scripts/order9_r1_nominal_compression_sweep_rollout.py"
)
ORDER9_R1_NOMINAL_COMPRESSION_OPTION = "--r1-nominal-additional-compression-sweep-mm"
ORDER9_R1_NOMINAL_COMPRESSION_WRAPPER_VERSION = (
    "order9_r1_deterministic_nominal_compression_sweep_v1"
)

_BINDING_KEYS = (
    "base_v6_protocol",
    "base_v6_approval",
    "base_v6_result_ledger",
    "base_numeric_protocol",
    "protected_c3_curriculum",
    "protected_c3_checkpoint",
    "protected_c3_rollout",
    "protected_c3_release_ledger",
    "compression_wrapper_implementation",
    "v7_calibration_implementation",
    "v7_runner_implementation",
    "selected_repair_raw_evidence",
    "selected_repair_episode_evidence",
)
_REQUIRED_APPROVAL_SCOPE = {
    "preserve_v6_teacher_and_ik_trajectory_bytes",
    "preserve_v6_contact_assignments",
    "apply_five_mm_nominal_compression_only_to_train_000004",
    "default_zero_additional_compression",
    "nominal_qpid_qp_local_servo_standard_path",
    "pi_l_retained_but_not_applied",
    "reuse_all_682_v6_fast_screen_admissions",
    "rerun_all_682_minimum_level_cases_in_isaac",
    "two_isaac_replays_per_candidate",
    "quarantine_invalid_interrupted_outputs_before_retry",
    "no_learning_or_teacher_collection_by_minimum_level_result_alone",
}


def load_order9_r1_nominal_calibration_v7_contract(
    repository_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load v7 and fail closed on every bound input byte."""

    repository = Path(repository_root).resolve()
    protocol_path = repository / ORDER9_R1_NOMINAL_V7_PROTOCOL_RELATIVE
    approval_path = repository / ORDER9_R1_NOMINAL_V7_APPROVAL_RELATIVE
    protocol = load_config(protocol_path)
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    overrides = protocol.get("additional_compression_mm_by_source_bucket")
    if (
        protocol.get("protocol_version") != ORDER9_R1_NOMINAL_V7_PROTOCOL_VERSION
        or protocol.get("calibration_gate_id")
        != "r1-nominal-bucket-local-compression-gate-v7"
        or protocol.get("status") != "approved"
        or protocol.get("execution_contract") != ORDER9_R1_NOMINAL_V7_EXECUTION_CONTRACT
        or protocol.get("base_level_id") != "r1_l1_10mm_5deg"
        or protocol.get("base_split") != "train"
        or int(protocol.get("source_bucket_count", -1)) != 22
        or int(protocol.get("candidate_count_per_bucket", -1)) != 31
        or int(protocol.get("replay_count_per_candidate", -1)) != 2
        or int(protocol.get("maximum_jobs_per_process", -1)) != 31
        or protocol.get("interrupted_attempt_quarantine_root")
        != "artifacts/p4_full/order9/r1_teacher/diagnostics/"
        "r1_v7_interrupted_formal_attempts"
        or float(protocol.get("default_additional_compression_mm", -1.0)) != 0.0
        or overrides != {"train-000004-190f3a425b3e": 5.0}
        or protocol.get("v6_teacher_trajectory_bytes_reused") is not True
        or protocol.get("contact_assignments_changed") is not False
        or protocol.get("pi_l_actor_command_applied") is not False
        or protocol.get("isaac_required") is not True
        or protocol.get("minimum_level_result_authorizes_collection") is not False
    ):
        raise SchemaValidationError("R1 nominal v7 protocol contract is invalid")
    protocol_sha = hash_file(protocol_path)
    if (
        approval.get("record_version") != ORDER9_R1_NOMINAL_V7_APPROVAL_VERSION
        or approval.get("decision") != "approved"
        or approval.get("calibration_gate_id") != protocol["calibration_gate_id"]
        or set(approval.get("approval_scope", ())) != _REQUIRED_APPROVAL_SCOPE
        or approval.get("approved_protocol", {}).get("sha256") != protocol_sha
        or Path(str(approval.get("approved_protocol", {}).get("path", "")))
        != ORDER9_R1_NOMINAL_V7_PROTOCOL_RELATIVE
    ):
        raise SchemaValidationError("R1 nominal v7 approval record is invalid")
    for key in _BINDING_KEYS:
        binding = protocol.get(key)
        if not isinstance(binding, dict):
            raise SchemaValidationError(f"R1 nominal v7 binding missing: {key}")
        source = repository / str(binding.get("path", ""))
        if not source.is_file() or hash_file(source) != binding.get("sha256"):
            raise SchemaValidationError(f"R1 nominal v7 binding changed: {key}")
    load_order9_r1_nominal_calibration_v6_contract(repository)
    _validate_selected_repair_evidence(protocol, repository)
    return protocol, approval


def order9_r1_v7_additional_compression_mm(
    materialized: MaterializedOrder9R1IsaacCase,
    protocol: Mapping[str, Any],
) -> float:
    overrides = dict(protocol["additional_compression_mm_by_source_bucket"])
    value = float(
        overrides.get(
            materialized.manifest.source_bucket_id,
            protocol["default_additional_compression_mm"],
        )
    )
    if not 0.0 <= value <= 10.0:
        raise SchemaValidationError("R1 nominal v7 compression is outside [0, 10] mm")
    return value


def order9_r1_nominal_v7_case_command(
    materialized: MaterializedOrder9R1IsaacCase,
    *,
    repository_root: str | Path,
    python_executable: str | Path,
    rollout_steps: int,
    protocol: Mapping[str, Any],
) -> tuple[list[str], Path]:
    repository = Path(repository_root).resolve()
    command, log_path = order9_r1_nominal_case_command(
        materialized,
        repository_root=repository,
        python_executable=python_executable,
        rollout_steps=rollout_steps,
    )
    script_indices = [
        index
        for index, value in enumerate(command)
        if Path(value).name == "order9_vectorized_isaac_rollout.py"
    ]
    if len(script_indices) != 1:
        raise SchemaValidationError("R1 nominal v7 protected rollout command differs")
    command[script_indices[0]] = str(
        repository / ORDER9_R1_NOMINAL_COMPRESSION_WRAPPER_RELATIVE
    )
    value = order9_r1_v7_additional_compression_mm(materialized, protocol)
    command.extend([ORDER9_R1_NOMINAL_COMPRESSION_OPTION, f"{value:.1f},{value:.1f}"])
    return command, log_path


def coalesce_order9_r1_v7_same_morphology_collectors(
    commands: Mapping[str, Sequence[str]],
    *,
    log_paths: Mapping[str, str | Path],
    morphology_hash_by_name: Mapping[str, str],
    batch_root: str | Path,
    maximum_jobs_per_process: int = 31,
) -> tuple[dict[str, list[str]], dict[str, Path], dict[str, tuple[str, ...]]]:
    """Coalesce wrapper commands while keeping its option out of child argv."""

    names = tuple(commands)
    if (
        not names
        or set(names) != set(log_paths)
        or set(names) != set(morphology_hash_by_name)
    ):
        raise ValueError("R1 nominal v7 collector inputs must be aligned")
    if maximum_jobs_per_process < 2:
        raise ValueError("R1 nominal v7 persistent batch size must be at least two")
    groups: dict[str, list[str]] = {}
    for name in names:
        morphology_hash = morphology_hash_by_name[name]
        if not isinstance(morphology_hash, str) or not morphology_hash:
            raise ValueError("R1 nominal v7 morphology hash is invalid")
        groups.setdefault(morphology_hash, []).append(name)

    root = Path(batch_root).resolve()
    output_commands: dict[str, list[str]] = {}
    output_logs: dict[str, Path] = {}
    members: dict[str, tuple[str, ...]] = {}
    for morphology_hash, grouped_names in groups.items():
        chunks = [
            grouped_names[start : start + maximum_jobs_per_process]
            for start in range(0, len(grouped_names), maximum_jobs_per_process)
        ]
        for chunk_index, chunk_names in enumerate(chunks):
            if len(chunk_names) == 1:
                name = chunk_names[0]
                output_commands[name] = list(commands[name])
                output_logs[name] = Path(log_paths[name]).resolve()
                members[name] = (name,)
                continue
            batch_name = f"morphology_{morphology_hash[:16]}_chunk_{chunk_index:02d}"
            first = list(commands[chunk_names[0]])
            script_index = _single_wrapper_index(first)
            expected_sweep = _single_option_value(
                first, ORDER9_R1_NOMINAL_COMPRESSION_OPTION
            )
            jobs = []
            for name in chunk_names:
                command = list(commands[name])
                if (
                    _single_wrapper_index(command) != script_index
                    or command[: script_index + 1] != first[: script_index + 1]
                    or _single_option_value(
                        command, ORDER9_R1_NOMINAL_COMPRESSION_OPTION
                    )
                    != expected_sweep
                ):
                    raise ValueError(
                        "same-morphology R1 v7 collectors differ in executable or repair"
                    )
                jobs.append(
                    {
                        "name": name,
                        "argv": _remove_option_pair(
                            command[script_index + 1 :],
                            ORDER9_R1_NOMINAL_COMPRESSION_OPTION,
                        ),
                        "log_path": str(Path(log_paths[name]).resolve()),
                    }
                )
            batch_dir = root / batch_name
            batch_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = batch_dir / "persistent_bucket_jobs.json"
            write_order9_r1_clearance_diagnostic(
                {
                    "version": ORDER9_PERSISTENT_BUCKET_JOBS_VERSION,
                    "morphology_hash": morphology_hash,
                    "chunk_index": chunk_index,
                    "maximum_jobs_per_process": maximum_jobs_per_process,
                    "r1_nominal_additional_compression_sweep_mm": expected_sweep,
                    "jobs": jobs,
                },
                manifest_path,
            )
            output_commands[batch_name] = [
                *first,
                "--persistent-bucket-jobs",
                str(manifest_path),
            ]
            output_logs[batch_name] = batch_dir / "persistent_process.log"
            members[batch_name] = tuple(chunk_names)
    return output_commands, output_logs, members


def run_order9_r1_nominal_v7_isaac_cases(
    cases: Sequence[MaterializedOrder9R1IsaacCase],
    *,
    repository_root: str | Path,
    python_executable: str | Path,
    rollout_steps: int = 15000,
    maximum_parallel_process_count: int = 4,
) -> dict[str, tuple[Order9R1NominalReplayResult, ...]]:
    if not cases:
        return {}
    repository = Path(repository_root).resolve()
    protocol, _approval = load_order9_r1_nominal_calibration_v7_contract(repository)
    commands: dict[str, list[str]] = {}
    logs: dict[str, Path] = {}
    morphology: dict[str, str] = {}
    complete: dict[str, tuple[Order9R1NominalReplayResult, ...]] = {}
    for case in cases:
        write_order9_r1_nominal_v7_case_contract(
            case, repository_root=repository, protocol=protocol
        )
        try:
            complete[case.manifest.candidate_id] = (
                validate_order9_r1_nominal_v7_isaac_result(
                    case, repository_root=repository, protocol=protocol
                )
            )
            continue
        except (OSError, ValueError, SchemaValidationError) as error:
            _quarantine_incomplete_v7_isaac_output(
                case,
                repository_root=repository,
                protocol=protocol,
                reason=f"{type(error).__name__}: {error}",
            )
        command, log = order9_r1_nominal_v7_case_command(
            case,
            repository_root=repository,
            python_executable=python_executable,
            rollout_steps=rollout_steps,
            protocol=protocol,
        )
        name = case.manifest.candidate_id
        commands[name] = command
        logs[name] = log
        morphology[name] = case.manifest.morphology_hash
    if commands:
        commands, logs, _members = coalesce_order9_r1_v7_same_morphology_collectors(
            commands,
            log_paths=logs,
            morphology_hash_by_name=morphology,
            batch_root=(
                Path(protocol["output_root"])
                if Path(protocol["output_root"]).is_absolute()
                else repository / protocol["output_root"] / "persistent_process_batches"
            ),
            maximum_jobs_per_process=int(protocol["maximum_jobs_per_process"]),
        )
        run_parallel_order9_collectors(
            commands,
            repository_root=repository,
            log_paths=logs,
            maximum_parallel_process_count=maximum_parallel_process_count,
            process_start_stagger_s=3.0,
        )
    for case in cases:
        annotate_order9_r1_nominal_isaac_result(case, repository_root=repository)
        results = validate_order9_r1_nominal_v7_isaac_result(
            case, repository_root=repository, protocol=protocol
        )
        complete[case.manifest.candidate_id] = results
        write_order9_r1_nominal_case_result(case, results, repository_root=repository)
        write_order9_r1_nominal_v7_case_result(
            case, results, repository_root=repository, protocol=protocol
        )
    return complete


def write_order9_r1_nominal_v7_case_contract(
    materialized: MaterializedOrder9R1IsaacCase,
    *,
    repository_root: str | Path,
    protocol: Mapping[str, Any] | None = None,
) -> Path:
    repository = Path(repository_root).resolve()
    if protocol is None:
        protocol, _approval = load_order9_r1_nominal_calibration_v7_contract(repository)
    base_contract = write_order9_r1_nominal_case_contract(
        materialized, repository_root=repository
    )
    root = materialized.manifest_path.parent
    source_root = _source_v6_case_root(materialized, protocol, repository)
    compression_mm = order9_r1_v7_additional_compression_mm(materialized, protocol)
    payload = {
        "contract_version": ORDER9_R1_NOMINAL_V7_CASE_CONTRACT_VERSION,
        "candidate_id": materialized.manifest.candidate_id,
        "execution_contract": ORDER9_R1_NOMINAL_V7_EXECUTION_CONTRACT,
        "source_bucket_id": materialized.manifest.source_bucket_id,
        "additional_compression_mm": compression_mm,
        "v6_teacher_trajectory_bytes_reused": True,
        "contact_assignments_changed": False,
        "pi_l_system_retained": True,
        "pi_l_actor_command_applied": False,
        "nominal_preload_applied": True,
        "qpid_qp_applied": True,
        "local_servo_applied": True,
        "isaac_required": True,
        "training_eligible": False,
        "formal_teacher_collection_authorized": False,
        "bindings": {
            "v7_protocol": _binding(
                repository / ORDER9_R1_NOMINAL_V7_PROTOCOL_RELATIVE
            ),
            "v7_approval": _binding(
                repository / ORDER9_R1_NOMINAL_V7_APPROVAL_RELATIVE
            ),
            "base_nominal_contract": _binding(base_contract),
            "v7_case_manifest_copy": _binding(materialized.manifest_path),
            "source_v6_case_manifest": _binding(source_root / "case_manifest.json"),
            "source_v6_case_contract": _binding(
                source_root / "nominal_contract_v6.json"
            ),
            "compression_wrapper": _binding(
                repository / ORDER9_R1_NOMINAL_COMPRESSION_WRAPPER_RELATIVE
            ),
        },
    }
    return write_order9_r1_clearance_diagnostic(
        payload, root / "nominal_contract_v7.json"
    )


def validate_order9_r1_nominal_v7_isaac_result(
    materialized: MaterializedOrder9R1IsaacCase,
    *,
    repository_root: str | Path,
    protocol: Mapping[str, Any] | None = None,
) -> tuple[Order9R1NominalReplayResult, ...]:
    repository = Path(repository_root).resolve()
    if protocol is None:
        protocol, _approval = load_order9_r1_nominal_calibration_v7_contract(repository)
    results = validate_order9_r1_nominal_isaac_result(
        materialized, repository_root=repository
    )
    root = materialized.manifest_path.parent / "isaac"
    raw_path = root / "evaluation_rollout.pt"
    episodes_path = root / "evaluation_episodes.jsonl"
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    metadata = raw.get("metadata") if isinstance(raw, dict) else None
    expected_mm = order9_r1_v7_additional_compression_mm(materialized, protocol)
    expected_values = [expected_mm] * materialized.manifest.replay_count
    wrapper_sha = hash_file(repository / ORDER9_R1_NOMINAL_COMPRESSION_WRAPPER_RELATIVE)
    if (
        not isinstance(metadata, dict)
        or metadata.get("r1_nominal_compression_diagnostic_version")
        != ORDER9_R1_NOMINAL_COMPRESSION_WRAPPER_VERSION
        or metadata.get("r1_nominal_additional_compression_mm_by_environment")
        != expected_values
        or metadata.get("r1_nominal_compression_wrapper_sha256") != wrapper_sha
        or metadata.get("pi_l_actor_command_applied") is not False
    ):
        raise SchemaValidationError("R1 nominal v7 raw repair contract differs")
    records = [
        json.loads(line)
        for line in episodes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != len(results) or any(
        record.get("metadata", {}).get("r1_nominal_compression_diagnostic_version")
        != ORDER9_R1_NOMINAL_COMPRESSION_WRAPPER_VERSION
        or float(
            record.get("metadata", {}).get("r1_nominal_additional_compression_mm", -1.0)
        )
        != expected_mm
        or record.get("metadata", {}).get("pi_l_actor_command_applied") is not False
        for record in records
    ):
        raise SchemaValidationError("R1 nominal v7 episode repair contract differs")
    return results


def write_order9_r1_nominal_v7_case_result(
    materialized: MaterializedOrder9R1IsaacCase,
    results: Sequence[Order9R1NominalReplayResult],
    *,
    repository_root: str | Path,
    protocol: Mapping[str, Any] | None = None,
) -> Path:
    repository = Path(repository_root).resolve()
    if protocol is None:
        protocol, _approval = load_order9_r1_nominal_calibration_v7_contract(repository)
    root = materialized.manifest_path.parent
    payload = {
        "result_version": ORDER9_R1_NOMINAL_V7_CASE_RESULT_VERSION,
        "candidate_id": materialized.manifest.candidate_id,
        "execution_contract": ORDER9_R1_NOMINAL_V7_EXECUTION_CONTRACT,
        "source_bucket_id": materialized.manifest.source_bucket_id,
        "additional_compression_mm": order9_r1_v7_additional_compression_mm(
            materialized, protocol
        ),
        "status": "accepted" if all(value.success for value in results) else "rejected",
        "episode_count": len(results),
        "success_count": sum(value.success for value in results),
        "safety_failure_count": sum(value.safety_failure for value in results),
        "fallback_count": sum(value.fallback_used for value in results),
        "pi_l_actor_command_applied": False,
        "training_eligible": False,
        "formal_teacher_collection_authorized": False,
        "bindings": {
            "v7_protocol": _binding(
                repository / ORDER9_R1_NOMINAL_V7_PROTOCOL_RELATIVE
            ),
            "v7_approval": _binding(
                repository / ORDER9_R1_NOMINAL_V7_APPROVAL_RELATIVE
            ),
            "v7_case_contract": _binding(root / "nominal_contract_v7.json"),
            "base_nominal_result": _binding(root / "nominal_result.json"),
            "raw_rollout": _binding(root / "isaac/evaluation_rollout.pt"),
            "episodes": _binding(root / "isaac/evaluation_episodes.jsonl"),
        },
    }
    return write_order9_r1_clearance_diagnostic(
        payload, root / "nominal_result_v7.json"
    )


def _validate_selected_repair_evidence(
    protocol: Mapping[str, Any], repository: Path
) -> None:
    raw_path = repository / protocol["selected_repair_raw_evidence"]["path"]
    episodes_path = repository / protocol["selected_repair_episode_evidence"]["path"]
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    metadata = raw.get("metadata") if isinstance(raw, dict) else None
    episodes = [
        json.loads(line)
        for line in episodes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if (
        not isinstance(metadata, dict)
        or metadata.get("r1_nominal_compression_diagnostic_version")
        != ORDER9_R1_NOMINAL_COMPRESSION_WRAPPER_VERSION
        or metadata.get("r1_nominal_additional_compression_mm_by_environment")
        != [5.0, 5.0]
        or metadata.get("pi_l_actor_command_applied") is not False
        or len(episodes) != 2
        or not all(record.get("task_success") is True for record in episodes)
        or any(record.get("safety_failure") is True for record in episodes)
        or any(
            int(record.get("fallback_decision_count", -1)) != 0 for record in episodes
        )
        or any(
            float(
                record.get("metadata", {}).get(
                    "r1_nominal_additional_compression_mm", -1.0
                )
            )
            != 5.0
            for record in episodes
        )
    ):
        raise SchemaValidationError("R1 nominal v7 selected repair evidence failed")


def _source_v6_case_root(
    materialized: MaterializedOrder9R1IsaacCase,
    protocol: Mapping[str, Any],
    repository: Path,
) -> Path:
    return (
        repository
        / str(protocol["source_v6_level_root"])
        / materialized.manifest.candidate_id
    )


def _quarantine_incomplete_v7_isaac_output(
    materialized: MaterializedOrder9R1IsaacCase,
    *,
    repository_root: Path,
    protocol: Mapping[str, Any],
    reason: str,
) -> Path | None:
    """Move invalid interrupted output aside before a fail-closed retry."""

    source = materialized.manifest_path.parent / "isaac"
    if not source.is_dir() or not any(source.iterdir()):
        return None
    quarantine_root = (
        repository_root
        / str(protocol["interrupted_attempt_quarantine_root"])
        / materialized.manifest.candidate_id
    )
    attempt_index = 0
    while (quarantine_root / f"attempt_{attempt_index:03d}").exists():
        attempt_index += 1
    destination = quarantine_root / f"attempt_{attempt_index:03d}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.rename(destination)
    write_order9_r1_clearance_diagnostic(
        {
            "record_version": "order9_r1_nominal_v7_interrupted_attempt_v1",
            "candidate_id": materialized.manifest.candidate_id,
            "reason": reason,
            "source_directory": str(source),
            "quarantined_directory": str(destination),
            "retry_allowed": True,
            "accepted_evidence": False,
            "training_eligible": False,
            "formal_teacher_collection_authorized": False,
        },
        destination.parent / f"attempt_{attempt_index:03d}.json",
    )
    return destination


def _single_wrapper_index(command: Sequence[str]) -> int:
    indices = [
        index
        for index, value in enumerate(command)
        if Path(value).name == ORDER9_R1_NOMINAL_COMPRESSION_WRAPPER_RELATIVE.name
    ]
    if len(indices) != 1:
        raise ValueError("R1 nominal v7 command lacks exactly one wrapper")
    return indices[0]


def _single_option_value(command: Sequence[str], option: str) -> str:
    indices = [index for index, value in enumerate(command) if value == option]
    if len(indices) != 1 or indices[0] + 1 >= len(command):
        raise ValueError(f"R1 nominal v7 command lacks exactly one {option}")
    return command[indices[0] + 1]


def _remove_option_pair(command: Sequence[str], option: str) -> list[str]:
    index = list(command).index(option)
    result = list(command)
    del result[index : index + 2]
    return result


def _binding(path: Path) -> dict[str, str]:
    source = path.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {"path": str(source), "sha256": hash_file(source)}


__all__ = [
    "ORDER9_R1_NOMINAL_V7_EXECUTION_CONTRACT",
    "coalesce_order9_r1_v7_same_morphology_collectors",
    "load_order9_r1_nominal_calibration_v7_contract",
    "order9_r1_nominal_v7_case_command",
    "order9_r1_v7_additional_compression_mm",
    "run_order9_r1_nominal_v7_isaac_cases",
    "validate_order9_r1_nominal_v7_isaac_result",
    "write_order9_r1_nominal_v7_case_contract",
    "write_order9_r1_nominal_v7_case_result",
]
