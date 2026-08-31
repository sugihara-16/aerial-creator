#!/usr/bin/env python3
from __future__ import annotations

"""Select the R1 train-side range after the accepted 10 mm / 5 degree level."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

_PREPARATION_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_PREPARATION_THREADS_PER_WORKER = 2
_ISAAC_THREAD_ENV = {key: os.environ.get(key) for key in _PREPARATION_THREAD_ENV_KEYS}
for _thread_environment_key in _PREPARATION_THREAD_ENV_KEYS:
    os.environ[_thread_environment_key] = str(_PREPARATION_THREADS_PER_WORKER)

import torch

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.schemas.common import SchemaValidationError  # noqa: E402
from amsrr.training.order9_r1_calibration import (  # noqa: E402
    load_order9_r1_calibration_protocol,
)
from amsrr.training.order9_r1_calibration_runner import (  # noqa: E402
    enumerate_order9_r1_calibration_level_cases,
)
from amsrr.training.order9_r1_complete_task_materialization_v7 import (  # noqa: E402
    finalize_order9_r1_v7_materialized_case,
)
from amsrr.training.order9_r1_isaac_calibration import (  # noqa: E402
    load_order9_r1_isaac_case,
    materialize_order9_r1_isaac_case,
)
from amsrr.training.order9_r1_nominal_calibration import (  # noqa: E402
    order9_r1_nominal_case_command,
)
from amsrr.training.order9_r1_range_selection_v10 import (  # noqa: E402
    ORDER9_R1_MINIMUM_LEVEL_ID,
    ORDER9_R1_RANGE_LEVEL_IDS,
    Order9R1RangeTeacherScreenPipelineV10,
    chunk_candidate_ids,
    evaluate_range_level_gate,
    load_range_protocol,
    range_case_priority,
)
from amsrr.training.order9_r1_safe_timing import (  # noqa: E402
    order9_r1_safe_phase_time_scales,
)
from amsrr.utils.hashing import hash_file  # noqa: E402

RUNNER_VERSION = "order9_r1_train_range_selection_v10_bounded_four"
PROTOCOL = REPOSITORY / "configs/training/order9_r1_range_selection_protocol_v10.json"
APPROVAL = REPOSITORY / "for_codex/R1_RANGE_SELECTION_PROTOCOL_V10_APPROVAL.json"
BASE_PROTOCOL = REPOSITORY / "configs/training/order9_r1_calibration_protocol_v1.yaml"
PARENT_RESULT = REPOSITORY / "for_codex/R1_NOMINAL_CALIBRATION_V9_RESULT_LEDGER.json"
WRAPPER = REPOSITORY / "scripts/order9_r1_batched_nominal_compression_rollout.py"
RUNTIME = REPOSITORY / "amsrr/training/order9_r1_batched_nominal_runtime.py"
OUTPUT_ROOT = REPOSITORY / "artifacts/p4_full/order9/r1_teacher/range_selection_v10"
RESULT_LEDGER = OUTPUT_ROOT / "range_selection_result.json"

REPLAY_COUNT = 2
EXPECTED_SOURCE_COUNT = 22
EXPECTED_CANDIDATE_COUNT = 682
TRAIN003_SOURCE_ID = "train-000003-3b95da871f01"
TRAIN004_SOURCE_ID = "train-000004-190f3a425b3e"
_ACTION_TENSORS = (
    "applied_global_action",
    "contact_space_residual_action",
    "global_action",
    "joint_action",
    "previous_global_action",
)
_ACTIVE_CONTROLLER_TENSORS = (
    "rotor_thrusts_n",
    "command_joint_position_targets_rad",
    "controller_desired_wrench_body",
)
_PREPARATION_PIPELINE = None
_PREPARATION_PIPELINE_KEY = None


def _ensure_python_hash_seed_zero() -> None:
    """Restart the CLI before imports can observe a randomized hash order."""

    if os.environ.get("PYTHONHASHSEED") == "0":
        return
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    os.execve(sys.executable, [sys.executable, *sys.argv], environment)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--isaac-python",
        default="/home/leus/.local/share/mamba/envs/isaaclab3/bin/python",
    )
    parser.add_argument("--maximum-preparation-process-count", type=int, default=24)
    parser.add_argument("--maximum-parallel-batches", type=int, default=4)
    parser.add_argument("--maximum-wall-time-s", type=float, default=36000.0)
    parser.add_argument("--rollout-steps", type=int, default=15000)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument(
        "--preparation-only-level",
        choices=ORDER9_R1_RANGE_LEVEL_IDS,
        help="Prepare and screen one level without invoking Isaac.",
    )
    return parser


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _binding(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "path": str(resolved.relative_to(REPOSITORY)),
        "sha256": hash_file(resolved),
    }


def _validate_binding(value: object, *, label: str) -> Path:
    if not isinstance(value, dict):
        raise SchemaValidationError(f"R1 range binding is missing: {label}")
    path = REPOSITORY / str(value.get("path", ""))
    if not path.is_file() or value.get("sha256") != hash_file(path):
        raise SchemaValidationError(f"R1 range binding changed: {label}")
    return path


def _load_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = load_range_protocol(PROTOCOL)
    approval = json.loads(APPROVAL.read_text(encoding="utf-8"))
    if (
        approval.get("record_version")
        != "order9_r1_range_selection_approval_record_v10"
        or approval.get("decision") != "approved"
        or approval.get("approved_by") != "repository_user"
        or approval.get("approved_protocol", {}).get("path")
        != str(PROTOCOL.relative_to(REPOSITORY))
        or approval.get("approved_protocol", {}).get("sha256") != hash_file(PROTOCOL)
    ):
        raise SchemaValidationError("R1 range approval differs")
    required = (
        "base_numeric_protocol",
        "parent_minimum_level_result",
        "protected_c3_checkpoint",
        "protected_c3_rollout",
        "protected_c3_curriculum",
        "protected_c3_release_ledger",
        "range_helper_implementation",
        "range_teacher_hints",
        "lightweight_planner_implementation",
        "runner_implementation",
        "batched_wrapper_implementation",
        "batched_runtime_implementation",
    )
    for key in required:
        _validate_binding(protocol.get(key), label=key)
    parent = json.loads(PARENT_RESULT.read_text(encoding="utf-8"))
    if (
        parent.get("decision") != "accepted_minimum_level"
        or parent.get("level_id") != ORDER9_R1_MINIMUM_LEVEL_ID
        or int(parent.get("candidate_count", -1)) != EXPECTED_CANDIDATE_COUNT
        or int(parent.get("success_count", -1)) != 2 * EXPECTED_CANDIDATE_COUNT
    ):
        raise SchemaValidationError("R1 range parent result differs")
    return protocol, approval


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SchemaValidationError(f"R1 range JSON object expected: {path}")
    return payload


def _validate_prepared_case(case, destination: Path) -> dict[str, Any]:
    manifest_path = destination / "case_manifest.json"
    screen_path = destination / "fast_screen.json"
    semantic_path = destination / "complete_task_semantic_audit.json"
    parameter_path = destination / "complete_task_parameter_effect_audit.json"
    materialized = load_order9_r1_isaac_case(manifest_path, REPOSITORY)
    screen = _read_json(screen_path)
    semantic = _read_json(semantic_path)
    parameter = _read_json(parameter_path)
    if (
        materialized.manifest.candidate_id != case.candidate_id
        or materialized.manifest.source_bucket_id != case.source_bucket.bucket_id
        or materialized.manifest.morphology_hash != case.source_bucket.morphology_hash
        or screen.get("candidate_id") != case.candidate_id
        or screen.get("accepted") is not True
        or screen.get("eligible_for_full_control_test") is not True
        or screen.get("isaac_invoked") is not False
        or screen.get("controller_layers_invoked") is not False
        or semantic.get("status") != "accepted"
        or semantic.get("isaac_invoked") is not False
        or semantic.get("controller_layers_invoked") is not False
        or parameter.get("status") != "accepted"
        or parameter.get("isaac_invoked") is not False
        or parameter.get("controller_layers_invoked") is not False
    ):
        raise SchemaValidationError(
            f"R1 range prepared case differs: {case.candidate_id}"
        )
    evidence = {
        "case_manifest": _binding(manifest_path),
        "fast_screen": _binding(screen_path),
        "complete_task_semantic_audit": _binding(semantic_path),
        "complete_task_parameter_effect_audit": _binding(parameter_path),
    }
    rotation_path = destination / "grasp_rotation_stability_audit.json"
    if rotation_path.is_file():
        rotation = _read_json(rotation_path)
        if rotation.get("accepted") is not True:
            raise SchemaValidationError("R1 range rotation admission differs")
        evidence["grasp_rotation_stability_audit"] = _binding(rotation_path)
    return {
        "candidate_id": case.candidate_id,
        "source_bucket_id": case.source_bucket.bucket_id,
        "module_count": case.source_bucket.module_count,
        "sample_kind": case.sample_kind,
        "sample_index": case.sample_index,
        "teacher_feasible": True,
        "screen_passed": True,
        "prepared": True,
        "preparation_failure_reason": None,
        "preparation_evidence": evidence,
    }


def _prepare_case(arguments) -> dict[str, Any]:
    case, base_protocol_text, output_root_text = arguments
    base_protocol_path = Path(base_protocol_text)
    output_root = Path(output_root_text)
    destination = output_root / case.candidate_id
    if (destination / "case_manifest.json").is_file():
        try:
            return _validate_prepared_case(case, destination)
        except (OSError, RuntimeError, ValueError):
            # An interrupted run may leave the manifest before every bound
            # audit.  Incomplete cache state is never accepted as evidence.
            shutil.rmtree(destination)
    if destination.exists():
        shutil.rmtree(destination)

    base = load_order9_r1_calibration_protocol(
        base_protocol_path, repository_root=REPOSITORY
    )
    global _PREPARATION_PIPELINE, _PREPARATION_PIPELINE_KEY
    pipeline_key = (
        str(base_protocol_path),
        base.source_bucket_manifest.sha256,
        base.minimum_normalized_joint_limit_reserve,
        base.maximum_body_tilt_rad,
    )
    if _PREPARATION_PIPELINE_KEY != pipeline_key:
        _PREPARATION_PIPELINE = Order9R1RangeTeacherScreenPipelineV10(
            repository_root=REPOSITORY,
            source_bucket_manifest_path=(REPOSITORY / base.source_bucket_manifest.path),
            minimum_normalized_joint_limit_reserve=(
                base.minimum_normalized_joint_limit_reserve
            ),
            maximum_body_tilt_rad=base.maximum_body_tilt_rad,
            anchor_position_tolerance_m=0.030,
            enforce_joint_limit_reserve_during_ik=True,
        )
        _PREPARATION_PIPELINE_KEY = pipeline_key

    record = {
        "candidate_id": case.candidate_id,
        "source_bucket_id": case.source_bucket.bucket_id,
        "module_count": case.source_bucket.module_count,
        "sample_kind": case.sample_kind,
        "sample_index": case.sample_index,
        "teacher_feasible": False,
        "screen_passed": False,
        "prepared": False,
        "preparation_failure_reason": None,
        "preparation_evidence": {},
    }
    try:
        prepared = _PREPARATION_PIPELINE.prepare(case)
        prepared.validate_for(case)
        if not prepared.teacher_trajectory_complete:
            record["preparation_failure_reason"] = prepared.failure_reason
            return record
        record["teacher_feasible"] = True
        screen = _PREPARATION_PIPELINE.screen(case, prepared)
        if not screen.accepted or not screen.eligible_for_full_control_test:
            record["preparation_failure_reason"] = ",".join(screen.violation_codes)
            return record
        record["screen_passed"] = True
        materialized = materialize_order9_r1_isaac_case(
            case=case,
            prepared=prepared,
            screen=screen,
            output_dir=destination,
            repository_root=REPOSITORY,
            approved_protocol_path=base_protocol_path,
        )
        finalize_order9_r1_v7_materialized_case(
            materialized,
            task_spec=case.task_spec,
            phase_time_scales=order9_r1_safe_phase_time_scales(
                case.source_bucket.module_count
            ),
            joint_rate_limit_rad_s=(
                float(screen.maximum_joint_rate_rad_s)
                + float(screen.minimum_joint_rate_margin_rad_s)
            ),
            repository_root=REPOSITORY,
        )
        return _validate_prepared_case(case, destination)
    except (OSError, RuntimeError, ValueError, SchemaValidationError) as error:
        if destination.exists():
            shutil.rmtree(destination)
        record["preparation_failure_reason"] = f"{type(error).__name__}: {error}"
        return record


def _prepare_level(
    level_id: str,
    *,
    output_root: Path,
    maximum_process_count: int,
) -> tuple[list[Any], list[dict[str, Any]]]:
    cases = list(
        enumerate_order9_r1_calibration_level_cases(
            protocol_path=BASE_PROTOCOL,
            repository_root=REPOSITORY,
            level_id=level_id,
            split="train",
        )
    )
    if len(cases) != EXPECTED_CANDIDATE_COUNT:
        raise SchemaValidationError("R1 range candidate count differs")
    cases.sort(
        key=lambda case: range_case_priority(
            case.candidate_id, case.sample_kind, case.sample_index
        )
    )
    preparation_root = output_root / "prepared" / level_id
    arguments = [(case, str(BASE_PROTOCOL), str(preparation_root)) for case in cases]
    records: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=maximum_process_count) as executor:
        futures = {
            executor.submit(_prepare_case, value): value[0] for value in arguments
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            case = futures[future]
            records[case.candidate_id] = future.result()
            if completed % 25 == 0 or completed == len(cases):
                print(
                    "ORDER9_R1_RANGE_PREPARATION_PROGRESS="
                    + json.dumps(
                        {
                            "level_id": level_id,
                            "completed": completed,
                            "total": len(cases),
                            "admitted": sum(
                                bool(record["prepared"]) for record in records.values()
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    ordered = [records[case.candidate_id] for case in cases]
    _atomic_json(
        output_root / "preparation" / f"{level_id}.json",
        {
            "record_version": "order9_r1_range_preparation_v10",
            "level_id": level_id,
            "candidate_count": len(ordered),
            "prepared_count": sum(bool(record["prepared"]) for record in ordered),
            "isaac_invoked": False,
            "controller_layers_invoked": False,
            "entries": ordered,
        },
    )
    return cases, ordered


def _job_for_case(
    case, *, preparation_root: Path, isaac_python: str, rollout_steps: int
):
    materialized = load_order9_r1_isaac_case(
        preparation_root / case.candidate_id / "case_manifest.json", REPOSITORY
    )
    command, log_path = order9_r1_nominal_case_command(
        materialized,
        repository_root=REPOSITORY,
        python_executable=isaac_python,
        rollout_steps=rollout_steps,
    )
    indices = [
        index
        for index, value in enumerate(command)
        if Path(value).name == "order9_vectorized_isaac_rollout.py"
    ]
    if len(indices) != 1:
        raise SchemaValidationError("R1 range rollout command differs")
    return {
        "name": case.candidate_id,
        "argv": command[indices[0] + 1 :],
        "log_path": str(log_path),
    }


def _source_adjustments(source_id: str) -> tuple[float, float]:
    compression_mm = (
        5.0 if source_id in {TRAIN003_SOURCE_ID, TRAIN004_SOURCE_ID} else 0.0
    )
    release_height_m = 0.03 if source_id == TRAIN003_SOURCE_ID else 0.0
    return compression_mm, release_height_m


def _write_batch_manifest(
    *,
    path: Path,
    source_id: str,
    jobs_by_id: Mapping[str, dict[str, Any]],
    candidate_ids: Sequence[str],
    execution_kind: str,
) -> Path:
    compression_mm, release_height_m = _source_adjustments(source_id)
    values = tuple(str(value) for value in candidate_ids)
    if (
        not values
        or len(values) > 4
        or any(value not in jobs_by_id for value in values)
    ):
        raise SchemaValidationError("R1 range batch candidate set differs")
    payload = {
        "version": "order9_r1_range_bounded_batch_manifest_v10",
        "runner_version": RUNNER_VERSION,
        "source_bucket_id": source_id,
        "execution_kind": execution_kind,
        "candidate_count": len(values),
        "r1_nominal_additional_compression_sweep_mm": (
            f"{compression_mm:.1f},{compression_mm:.1f}"
        ),
        "r1_release_height_offset_m": release_height_m,
        "jobs": [jobs_by_id[value] for value in values],
    }
    if path.is_file() and _read_json(path) == payload:
        return path
    return _atomic_json(path, payload)


def _resolved_reference_path(value: object) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (REPOSITORY / path).resolve()


def _require_current_nominal_reference(
    *,
    candidate_id: str,
    metadata: Mapping[str, Any],
    preparation_root: Path,
) -> dict[str, dict[str, str]]:
    """Bind cached Isaac evidence to the teacher currently being admitted."""

    reference = metadata.get("c3_nominal_reference")
    if not isinstance(reference, dict):
        raise SchemaValidationError("Isaac nominal reference is missing")
    expected_bucket_root = (
        preparation_root
        / candidate_id
        / "nominal_set"
        / "buckets"
        / candidate_id
    ).resolve()
    expected = {
        "set_manifest": (
            (preparation_root / candidate_id / "nominal_set" / "manifest.json").resolve(),
            "set_manifest_path",
            "set_manifest_sha256",
        ),
        "artifact_manifest": (
            (expected_bucket_root / "manifest.json").resolve(),
            "artifact_path",
            "artifact_sha256",
        ),
        "nominal_timeline": (
            (expected_bucket_root / "nominal_timeline.json").resolve(),
            None,
            "timeline_sha256",
        ),
    }
    bindings: dict[str, dict[str, str]] = {}
    for label, (expected_path, path_key, hash_key) in expected.items():
        if not expected_path.is_file():
            raise SchemaValidationError(
                f"current teacher artifact is missing: {label}"
            )
        if path_key is not None and _resolved_reference_path(
            reference.get(path_key)
        ) != expected_path:
            raise SchemaValidationError(
                f"Isaac nominal reference path differs: {label}"
            )
        current_hash = hash_file(expected_path)
        if reference.get(hash_key) != current_hash:
            raise SchemaValidationError(
                f"Isaac nominal reference hash differs: {label}"
            )
        bindings[label] = {
            "path": expected_path.relative_to(REPOSITORY).as_posix(),
            "sha256": current_hash,
        }
    return bindings


def _quarantine_invalid_candidate_output(
    *,
    candidate_id: str,
    evidence_root: Path,
    preparation_root: Path,
    failure_reason: str,
) -> dict[str, str] | None:
    """Move stale or partial Isaac evidence aside without deleting it."""

    source = (evidence_root / candidate_id).resolve()
    if not source.exists():
        return None
    output_root = preparation_root.resolve().parents[1]
    isaac_root = (output_root / "isaac").resolve()
    if not source.is_relative_to(isaac_root):
        raise SchemaValidationError("Isaac quarantine source is outside output root")
    quarantine_root = (
        output_root
        / "diagnostics"
        / "isaac_evidence_quarantine"
        / preparation_root.name
        / candidate_id
    )
    quarantine_root.mkdir(parents=True, exist_ok=True)
    reason_digest = hashlib.sha256(failure_reason.encode("utf-8")).hexdigest()[:12]
    attempt = 1
    while True:
        destination = quarantine_root / (
            f"{reason_digest}__attempt_{attempt:03d}"
        )
        if not destination.exists():
            break
        attempt += 1
    source.rename(destination)
    manifest = _atomic_json(
        quarantine_root / f"{destination.name}.json",
        {
            "record_version": "order9_r1_stale_isaac_evidence_quarantine_v1",
            "candidate_id": candidate_id,
            "failure_reason": failure_reason,
            "source_path": source.relative_to(REPOSITORY).as_posix(),
            "quarantine_path": destination.relative_to(REPOSITORY).as_posix(),
            "destructive_deletion_used": False,
        },
    )
    return _binding(manifest)


def _inspect_candidate_output(
    candidate_id: str,
    evidence_root: Path,
    *,
    preparation_root: Path | None = None,
    quarantine_invalid: bool = False,
) -> dict[str, Any]:
    raw_path = evidence_root / candidate_id / "isaac/evaluation_rollout.pt"
    episodes_path = evidence_root / candidate_id / "isaac/evaluation_episodes.jsonl"
    if not raw_path.is_file() or not episodes_path.is_file():
        result = {
            "candidate_id": candidate_id,
            "valid_evidence": False,
            "candidate_passed": False,
            "episode_count": 0,
            "success_count": 0,
            "safety_failure_count": 0,
            "fallback_count": 0,
            "failure_reason": "missing Isaac output",
        }
        if quarantine_invalid and preparation_root is not None:
            quarantine = _quarantine_invalid_candidate_output(
                candidate_id=candidate_id,
                evidence_root=evidence_root,
                preparation_root=preparation_root,
                failure_reason=result["failure_reason"],
            )
            if quarantine is not None:
                result["quarantined_evidence_manifest"] = quarantine
        return result
    try:
        payload = torch.load(raw_path, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata") if isinstance(payload, dict) else None
        tensors = payload.get("tensors") if isinstance(payload, dict) else None
        if (
            not isinstance(metadata, dict)
            or not isinstance(tensors, dict)
            or metadata.get("r1_batched_nominal_runtime_version")
            != "order9_r1_environment_wise_nominal_isaac_batch_v3_bounded_four"
            or metadata.get("r1_batched_nominal_wrapper_sha256") != hash_file(WRAPPER)
            or metadata.get("r1_batched_nominal_runtime_sha256") != hash_file(RUNTIME)
            or metadata.get("r1_batched_environment_origin_packed") is not False
            or abs(float(metadata.get("r1_batched_environment_spacing_m", -1.0)) - 3.0)
            > 1.0e-12
            or not 1 <= int(metadata.get("r1_batched_candidate_count", -1)) <= 4
            or int(metadata.get("r1_batched_maximum_candidates_per_scene", -1)) != 4
            or metadata.get("pi_l_actor_command_applied") is not False
            or metadata.get("diagnostic_nominal_qpid_only") is not True
            or metadata.get("formal_phase_zero_start") is not True
        ):
            raise SchemaValidationError("raw control contract differs")
        nominal_reference_bindings = None
        if preparation_root is not None:
            nominal_reference_bindings = _require_current_nominal_reference(
                candidate_id=candidate_id,
                metadata=metadata,
                preparation_root=preparation_root,
            )
        for key in _ACTION_TENSORS:
            value = tensors.get(key)
            if (
                not isinstance(value, torch.Tensor)
                or not torch.isfinite(value).all()
                or bool(torch.count_nonzero(value).item())
            ):
                raise SchemaValidationError(f"pi_L action differs: {key}")
        for key in _ACTIVE_CONTROLLER_TENSORS:
            value = tensors.get(key)
            if (
                not isinstance(value, torch.Tensor)
                or not torch.isfinite(value).all()
                or not bool(torch.count_nonzero(value).item())
            ):
                raise SchemaValidationError(f"controller evidence is empty: {key}")
        records = [
            json.loads(line)
            for line in episodes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(records) != REPLAY_COUNT:
            raise SchemaValidationError("replay count differs")
        success = sum(record.get("task_success") is True for record in records)
        safety = sum(record.get("safety_failure") is True for record in records)
        fallback = sum(
            int(record.get("fallback_decision_count", 0)) > 0 for record in records
        )
        terminal = []
        for record in records:
            metrics = record.get("metrics") if isinstance(record, dict) else None
            if not isinstance(metrics, dict):
                raise SchemaValidationError("episode metrics are missing")
            terminal.append(
                bool(
                    float(metrics.get("terminal_phase_index", -1.0)) == 8.0
                    and float(metrics.get("hard_collision", 1.0)) == 0.0
                    and float(metrics.get("object_dropped", 1.0)) == 0.0
                    and float(metrics.get("timeout", 1.0)) == 0.0
                    and float(metrics.get("qp_infeasible_terminal", 1.0)) == 0.0
                )
            )
        passed = (
            success == REPLAY_COUNT and safety == 0 and fallback == 0 and all(terminal)
        )
        result = {
            "candidate_id": candidate_id,
            "valid_evidence": True,
            "candidate_passed": passed,
            "episode_count": REPLAY_COUNT,
            "success_count": success,
            "safety_failure_count": safety,
            "fallback_count": fallback,
            "failure_reason": None if passed else "Isaac task gate failed",
            "raw_rollout": _binding(raw_path),
            "episodes": _binding(episodes_path),
        }
        if nominal_reference_bindings is not None:
            result["current_nominal_reference"] = nominal_reference_bindings
        return result
    except (OSError, RuntimeError, ValueError, SchemaValidationError) as error:
        result = {
            "candidate_id": candidate_id,
            "valid_evidence": False,
            "candidate_passed": False,
            "episode_count": 0,
            "success_count": 0,
            "safety_failure_count": 0,
            "fallback_count": 0,
            "failure_reason": f"{type(error).__name__}: {error}",
        }
        if quarantine_invalid and preparation_root is not None:
            quarantine = _quarantine_invalid_candidate_output(
                candidate_id=candidate_id,
                evidence_root=evidence_root,
                preparation_root=preparation_root,
                failure_reason=result["failure_reason"],
            )
            if quarantine is not None:
                result["quarantined_evidence_manifest"] = quarantine
        return result


def _run_processes(
    work: Sequence[tuple[str, Path, Path]],
    *,
    isaac_python: str,
    maximum_parallel: int,
    deadline: float,
) -> dict[str, int]:
    queue = list(work)
    active: dict[str, tuple[subprocess.Popen[str], Any]] = {}
    results: dict[str, int] = {}
    isaac_environment = dict(os.environ)
    for key, value in _ISAAC_THREAD_ENV.items():
        if value is None:
            isaac_environment.pop(key, None)
        else:
            isaac_environment[key] = value
    try:
        while queue or active:
            if time.time() >= deadline:
                raise TimeoutError("R1 range selection exceeded its wall-time bound")
            while queue and len(active) < maximum_parallel:
                work_id, manifest, output_root = queue.pop(0)
                output_root.mkdir(parents=True, exist_ok=True)
                log_path = manifest.with_suffix(".log")
                stream = log_path.open("w", encoding="utf-8")
                command = [
                    str(Path(isaac_python).resolve()),
                    str(WRAPPER),
                    "--r1-batched-jobs",
                    str(manifest),
                    "--r1-batched-output-root",
                    str(output_root),
                    "--no-formal-annotation",
                ]
                active[work_id] = (
                    subprocess.Popen(
                        command,
                        cwd=REPOSITORY,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=isaac_environment,
                    ),
                    stream,
                )
            time.sleep(1.0)
            for work_id, (process, stream) in tuple(active.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                results[work_id] = int(code)
                del active[work_id]
    except BaseException:
        for process, _stream in active.values():
            if process.poll() is None:
                process.terminate()
        for process, stream in active.values():
            try:
                process.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            stream.close()
        raise
    return results


def _has_decisive_failure(
    evidence: Mapping[str, Mapping[str, Any]], case_by_id: Mapping[str, Any]
) -> bool:
    for candidate_id, result in evidence.items():
        if result.get("valid_evidence") is not True:
            continue
        case = case_by_id[candidate_id]
        if result.get("safety_failure_count", 0) or result.get("fallback_count", 0):
            return True
        if case.sample_kind == "lattice" and not result.get("candidate_passed"):
            return True
    return False


def _execute_level(
    level_id: str,
    *,
    cases: Sequence[Any],
    preparation_records: Sequence[Mapping[str, Any]],
    output_root: Path,
    isaac_python: str,
    rollout_steps: int,
    maximum_parallel_batches: int,
    deadline: float,
) -> list[dict[str, Any]]:
    records = {
        str(record["candidate_id"]): dict(record) for record in preparation_records
    }
    case_by_id = {case.candidate_id: case for case in cases}
    admitted = [case for case in cases if records[case.candidate_id]["prepared"]]
    preparation_root = output_root / "prepared" / level_id
    jobs_by_id = {
        case.candidate_id: _job_for_case(
            case,
            preparation_root=preparation_root,
            isaac_python=isaac_python,
            rollout_steps=rollout_steps,
        )
        for case in admitted
    }
    by_source: dict[str, list[Any]] = {}
    for case in admitted:
        by_source.setdefault(case.source_bucket.bucket_id, []).append(case)
    batch_specs = []
    for source_id, source_cases in sorted(by_source.items()):
        source_cases.sort(
            key=lambda case: range_case_priority(
                case.candidate_id, case.sample_kind, case.sample_index
            )
        )
        for chunk_index, candidate_ids in enumerate(
            chunk_candidate_ids(tuple(case.candidate_id for case in source_cases))
        ):
            priority = min(
                range_case_priority(
                    case_by_id[value].candidate_id,
                    case_by_id[value].sample_kind,
                    case_by_id[value].sample_index,
                )
                for value in candidate_ids
            )
            batch_specs.append((priority, source_id, chunk_index, candidate_ids))
    batch_specs.sort(key=lambda value: (value[0], value[1], value[2]))

    evidence: dict[str, dict[str, Any]] = {}
    isaac_root = output_root / "isaac" / level_id
    for wave_start in range(0, len(batch_specs), maximum_parallel_batches):
        if time.time() >= deadline:
            raise TimeoutError("R1 range selection exceeded its wall-time bound")
        wave = batch_specs[wave_start : wave_start + maximum_parallel_batches]
        work = []
        wave_roots: dict[str, Path] = {}
        for _priority, source_id, chunk_index, candidate_ids in wave:
            digest = hashlib.sha256("\n".join(candidate_ids).encode()).hexdigest()[:12]
            work_id = f"{source_id}:chunk_{chunk_index:03d}_{digest}"
            candidate_root = (
                isaac_root / "batches" / source_id / f"chunk_{chunk_index:03d}_{digest}"
            )
            wave_roots[work_id] = candidate_root
            pending = []
            for candidate_id in candidate_ids:
                single_root = isaac_root / "single" / candidate_id
                single = _inspect_candidate_output(
                    candidate_id,
                    single_root,
                    preparation_root=preparation_root,
                    quarantine_invalid=True,
                )
                if single.get("valid_evidence"):
                    evidence[candidate_id] = {
                        **single,
                        "evidence_group": "single_confirmation",
                    }
                    continue
                bounded = _inspect_candidate_output(
                    candidate_id,
                    candidate_root,
                    preparation_root=preparation_root,
                    quarantine_invalid=True,
                )
                if bounded.get("valid_evidence"):
                    if bounded.get("candidate_passed"):
                        evidence[candidate_id] = {
                            **bounded,
                            "evidence_group": "bounded_four",
                        }
                    # A valid bounded failure needs a single confirmation, but
                    # the bounded rollout itself must not be repeated.
                    continue
                pending.append(candidate_id)
            if not pending:
                continue
            manifest = _write_batch_manifest(
                path=(
                    isaac_root
                    / "manifests"
                    / source_id
                    / f"chunk_{chunk_index:03d}_{digest}.json"
                ),
                source_id=source_id,
                jobs_by_id=jobs_by_id,
                candidate_ids=pending,
                execution_kind="bounded_four_range_selection",
            )
            work.append((work_id, manifest, candidate_root))
        _run_processes(
            work,
            isaac_python=isaac_python,
            maximum_parallel=maximum_parallel_batches,
            deadline=deadline,
        )

        # On restart, a previously completed single confirmation can already
        # make this lattice level decisively fail.  Do not launch any remaining
        # bounded or single jobs once that evidence is available.
        if _has_decisive_failure(evidence, case_by_id):
            print(
                "ORDER9_R1_RANGE_ISAAC_PROGRESS="
                + json.dumps(
                    {
                        "level_id": level_id,
                        "completed_candidates": len(evidence),
                        "total_candidates": len(admitted),
                        "decisive_failure": True,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            break

        retry_work = []
        retry_roots: dict[str, Path] = {}
        for _priority, source_id, _chunk_index, candidate_ids in wave:
            for candidate_id in candidate_ids:
                if candidate_id in evidence:
                    continue
                matching = [
                    (work_id, root)
                    for work_id, root in wave_roots.items()
                    if work_id.startswith(source_id + ":")
                    and (root / candidate_id).is_dir()
                ]
                current = (
                    _inspect_candidate_output(
                        candidate_id,
                        matching[0][1],
                        preparation_root=preparation_root,
                        quarantine_invalid=True,
                    )
                    if len(matching) == 1
                    else {"candidate_passed": False}
                )
                if current.get("candidate_passed"):
                    evidence[candidate_id] = {
                        **current,
                        "evidence_group": "bounded_four",
                    }
                    continue
                retry_root = isaac_root / "single" / candidate_id
                cached = _inspect_candidate_output(
                    candidate_id,
                    retry_root,
                    preparation_root=preparation_root,
                    quarantine_invalid=True,
                )
                if cached.get("valid_evidence"):
                    evidence[candidate_id] = {
                        **cached,
                        "evidence_group": "single_confirmation",
                    }
                    continue
                manifest = _write_batch_manifest(
                    path=isaac_root / "single_manifests" / f"{candidate_id}.json",
                    source_id=source_id,
                    jobs_by_id=jobs_by_id,
                    candidate_ids=(candidate_id,),
                    execution_kind="single_candidate_confirmation",
                )
                retry_roots[candidate_id] = retry_root
                retry_work.append((candidate_id, manifest, retry_root))
        # Single confirmations are intentionally sequential.  One confirmed
        # lattice/safety/fallback failure is enough to reject the level, so
        # launching every failed bounded candidate in parallel wastes time.
        for candidate_id, manifest, retry_root in retry_work:
            _run_processes(
                ((candidate_id, manifest, retry_root),),
                isaac_python=isaac_python,
                maximum_parallel=1,
                deadline=deadline,
            )
            evidence[candidate_id] = {
                **_inspect_candidate_output(
                    candidate_id,
                    retry_root,
                    preparation_root=preparation_root,
                    quarantine_invalid=True,
                ),
                "evidence_group": "single_confirmation",
            }
            if _has_decisive_failure(evidence, case_by_id):
                break

        decisive = _has_decisive_failure(evidence, case_by_id)
        print(
            "ORDER9_R1_RANGE_ISAAC_PROGRESS="
            + json.dumps(
                {
                    "level_id": level_id,
                    "completed_candidates": len(evidence),
                    "total_candidates": len(admitted),
                    "decisive_failure": decisive,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if decisive:
            break

    entries = []
    for case in cases:
        base = records[case.candidate_id]
        runtime = evidence.get(
            case.candidate_id,
            {
                "candidate_passed": False,
                "episode_count": 0,
                "success_count": 0,
                "safety_failure_count": 0,
                "fallback_count": 0,
                "valid_evidence": False,
                "failure_reason": (
                    base.get("preparation_failure_reason")
                    or "not attempted after decisive failure"
                ),
            },
        )
        entries.append({**base, **runtime})
    return entries


def _gate(base, level_id: str, entries: Sequence[Mapping[str, Any]]):
    return evaluate_range_level_gate(
        level_id=level_id,
        entries=entries,
        minimum_lattice_pass_rate=base.minimum_lattice_pass_rate,
        minimum_teacher_feasibility_rate=base.minimum_teacher_feasibility_rate,
        minimum_fast_screen_pass_rate=base.minimum_fast_screen_pass_rate,
        minimum_isaac_success_rate=base.minimum_isaac_success_rate,
        minimum_interior_teacher_feasibility_rate=(
            base.minimum_interior_teacher_feasibility_rate
        ),
        minimum_interior_fast_screen_pass_rate=(
            base.minimum_interior_fast_screen_pass_rate
        ),
        minimum_interior_isaac_success_rate=(base.minimum_interior_isaac_success_rate),
        maximum_safety_failure_count=base.maximum_safety_failure_count,
        maximum_fallback_rate=base.maximum_fallback_rate,
    )


def main() -> int:
    _ensure_python_hash_seed_zero()
    args = _parser().parse_args()
    if (
        not 1 <= args.maximum_preparation_process_count <= 24
        or not 1 <= args.maximum_parallel_batches <= 4
        or args.maximum_wall_time_s <= 0.0
        or args.rollout_steps < 1
    ):
        raise ValueError("R1 range runtime limits are invalid")
    protocol, approval = _load_contract()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    base = load_order9_r1_calibration_protocol(
        BASE_PROTOCOL, repository_root=REPOSITORY
    )
    started = time.time()
    deadline = started + min(
        float(args.maximum_wall_time_s), float(protocol["maximum_wall_time_s"])
    )
    if args.preparation_only_level:
        _cases, preparation = _prepare_level(
            args.preparation_only_level,
            output_root=output_root,
            maximum_process_count=args.maximum_preparation_process_count,
        )
        print(
            "ORDER9_R1_RANGE_PREPARATION_COMPLETE="
            + json.dumps(
                {
                    "level_id": args.preparation_only_level,
                    "candidate_count": len(preparation),
                    "prepared_count": sum(
                        bool(record["prepared"]) for record in preparation
                    ),
                    "isaac_invoked": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    selected_level = ORDER9_R1_MINIMUM_LEVEL_ID
    level_results = []
    try:
        for level_id in ORDER9_R1_RANGE_LEVEL_IDS:
            if time.time() >= deadline:
                raise TimeoutError("R1 range selection exceeded its wall-time bound")
            cases, preparation = _prepare_level(
                level_id,
                output_root=output_root,
                maximum_process_count=args.maximum_preparation_process_count,
            )
            entries = _execute_level(
                level_id,
                cases=cases,
                preparation_records=preparation,
                output_root=output_root,
                isaac_python=args.isaac_python,
                rollout_steps=args.rollout_steps,
                maximum_parallel_batches=args.maximum_parallel_batches,
                deadline=deadline,
            )
            gate = _gate(base, level_id, entries)
            level_path = _atomic_json(
                output_root / "results" / f"{level_id}.json",
                {
                    "result_version": "order9_r1_range_level_result_v10",
                    "runner_version": RUNNER_VERSION,
                    "level_id": level_id,
                    "gate": gate.to_dict(),
                    "candidate_count": len(entries),
                    "pi_l_actor_command_applied": False,
                    "qpid_qp_applied": True,
                    "local_servo_applied": True,
                    "training_eligible": False,
                    "entries": entries,
                },
            )
            level_results.append(
                {
                    "level_id": level_id,
                    "passed": gate.passed,
                    "result": _binding(level_path),
                    "gate": gate.to_dict(),
                }
            )
            if not gate.passed:
                break
            selected_level = level_id
    except TimeoutError as error:
        _atomic_json(
            output_root / "incomplete_timeout.json",
            {
                "runner_version": RUNNER_VERSION,
                "status": "incomplete_timeout",
                "selected_level_before_timeout": selected_level,
                "completed_levels": level_results,
                "elapsed_s": time.time() - started,
                "error": str(error),
            },
        )
        return 2

    selection_complete = bool(
        level_results
        and (
            not level_results[-1]["passed"]
            or level_results[-1]["level_id"] == ORDER9_R1_RANGE_LEVEL_IDS[-1]
        )
    )
    ledger = {
        "ledger_version": "order9_r1_range_selection_result_ledger_v10",
        "runner_version": RUNNER_VERSION,
        "status": "selection_complete" if selection_complete else "incomplete",
        "selected_level_id": selected_level,
        "selection_level_ids": list(ORDER9_R1_RANGE_LEVEL_IDS),
        "completed_level_count": len(level_results),
        "selection_complete": selection_complete,
        "held_out_confirmation_executed": False,
        "formal_teacher_collection_authorized": False,
        "training_eligible": False,
        "pi_l_actor_command_applied": False,
        "qpid_qp_applied": True,
        "local_servo_applied": True,
        "maximum_candidates_per_isaac_scene": 4,
        "isaac_environment_spacing_m": 3.0,
        "elapsed_s": time.time() - started,
        "bindings": {
            "protocol": _binding(PROTOCOL),
            "approval": _binding(APPROVAL),
            "base_numeric_protocol": _binding(BASE_PROTOCOL),
            "parent_minimum_level_result": _binding(PARENT_RESULT),
            "range_helper_implementation": _binding(
                REPOSITORY / "amsrr/training/order9_r1_range_selection_v10.py"
            ),
            "lightweight_planner_implementation": _binding(
                REPOSITORY / "amsrr/training/order9_configuration_space_planner.py"
            ),
            "runner_implementation": _binding(Path(__file__)),
            "batched_wrapper_implementation": _binding(WRAPPER),
            "batched_runtime_implementation": _binding(RUNTIME),
        },
        "level_results": level_results,
        "approval_record_version": approval["record_version"],
    }
    result_path = _atomic_json(output_root / RESULT_LEDGER.name, ledger)
    print(
        "ORDER9_R1_RANGE_SELECTION_COMPLETE="
        + json.dumps(
            {
                "selected_level_id": selected_level,
                "completed_level_count": len(level_results),
                "result_ledger": str(result_path),
                "result_sha256": hash_file(result_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if selection_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
