#!/usr/bin/env python3
from __future__ import annotations

"""Run 673 R1 cases in verified batches of at most four Isaac candidates."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping

import torch

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.schemas.common import SchemaValidationError  # noqa: E402
from amsrr.utils.hashing import hash_file  # noqa: E402
from scripts.order9_r1_batched_nominal_compression_rollout import (  # noqa: E402
    BATCH_ENV_SPACING_M,
    BATCH_VERSION,
)
from scripts.order9_run_r1_batched_nominal_calibration_v7 import (  # noqa: E402
    _select_source_bucket_manifests,
    _source_bucket_id,
)

RUNNER_VERSION = (
    "order9_r1_bounded_four_minimum_level_runner_v3_existing_evidence_first"
)
WRAPPER = REPOSITORY / "scripts/order9_r1_batched_nominal_compression_rollout.py"
RUNTIME = REPOSITORY / "amsrr/training/order9_r1_batched_nominal_runtime.py"
LEGACY_V7_OUTPUT_ROOT = REPOSITORY / (
    "artifacts/p4_full/order9/r1_teacher/calibration_v9_nominal_compression_v7/"
    "selection/r1_l1_10mm_5deg"
)
LEGACY_BATCH_RUNTIME_SHA256 = (
    "422172ac78410f4f897b389ca337dd1ae46aebdcf5ebd4fca546bf8e33e49eae"
)
LEGACY_BATCH_WRAPPER_SHA256S = frozenset(
    {
        "f6ddc4f4c61be1146e65a392039d4081b93b0e9f4e3c8d06e22b19186a409518",
        "e8347a86d11d6ac3b10e705c62c529dd6caa01ae7dee7ee276d001de4a5bfa98",
    }
)
SOURCE_ROOT = REPOSITORY / (
    "artifacts/p4_full/order9/r1_teacher/calibration_v9_nominal_compression_v7/"
    "selection/r1_l1_10mm_5deg/persistent_process_batches"
)
OUTPUT_ROOT = REPOSITORY / (
    "artifacts/p4_full/order9/r1_teacher/validation_v10_bounded_four/"
    "r1_l1_10mm_5deg"
)
TRAIN003_SOURCE_ID = "train-000003-3b95da871f01"
TRAIN003_REPAIR_MANIFEST = REPOSITORY / (
    "artifacts/p4_full/order9/r1_teacher/diagnostics/"
    "r1_v8_train000003_original_grasp_compression5_release30_full_v1/all_jobs.json"
)
REPAIRED_EXCLUSION = REPOSITORY / (
    "configs/training/order9_r1_nominal_v8_repaired_candidates.json"
)
EXPECTED_SOURCE_BUCKET_COUNT = 22
EXPECTED_TOTAL_CANDIDATE_COUNT = 673
REPLAY_COUNT = 2
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--isaac-python",
        default="/home/leus/.local/share/mamba/envs/isaaclab3/bin/python",
    )
    parser.add_argument("--maximum-parallel-batches", type=int, default=4)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    return parser


def _atomic_json(destination: Path, payload: Mapping[str, Any]) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _ledger_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPOSITORY))
    except ValueError:
        return str(resolved)


def _excluded_candidate_ids() -> frozenset[str]:
    payload = json.loads(REPAIRED_EXCLUSION.read_text(encoding="utf-8"))
    values = payload.get("candidate_ids") if isinstance(payload, dict) else None
    if (
        not isinstance(values, list)
        or len(values) != 9
        or values != sorted(values)
        or len(values) != len(set(values))
    ):
        raise SchemaValidationError("R1 v9 repaired-candidate exclusion differs")
    return frozenset(values)


def _manifest_jobs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    jobs = payload.get("jobs") if isinstance(payload, dict) else None
    if not isinstance(jobs, list) or not jobs:
        raise SchemaValidationError(f"R1 v9 job manifest is invalid: {path}")
    return jobs


def _materialize_run_manifest(
    *,
    source_id: str,
    source_manifest: Path,
    destination: Path,
    excluded_ids: frozenset[str],
) -> tuple[Path, tuple[str, ...]]:
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    jobs = payload.get("jobs") if isinstance(payload, dict) else None
    if not isinstance(jobs, list) or not jobs:
        raise SchemaValidationError("R1 v9 source manifest is invalid")
    selected = [job for job in jobs if job.get("name") not in excluded_ids]
    candidate_ids = tuple(str(job.get("name")) for job in selected)
    if (
        not candidate_ids
        or len(candidate_ids) != len(set(candidate_ids))
        or any(_source_bucket_id(job) != source_id for job in selected)
    ):
        raise SchemaValidationError("R1 v9 selected source jobs differ")
    value = {
        **{key: item for key, item in payload.items() if key != "jobs"},
        "runner_version": RUNNER_VERSION,
        "source_bucket_id": source_id,
        "source_manifest_path": str(source_manifest),
        "source_manifest_sha256": hash_file(source_manifest),
        "excluded_candidate_ids": sorted(
            set(job.get("name") for job in jobs) & excluded_ids
        ),
        "candidate_count": len(candidate_ids),
        "jobs": selected,
    }
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != value:
            comparable_existing = {
                key: item for key, item in existing.items() if key != "runner_version"
            }
            comparable_value = {
                key: item for key, item in value.items() if key != "runner_version"
            }
            if comparable_existing != comparable_value:
                raise SchemaValidationError(
                    "R1 v9 run manifest already exists with other bytes: "
                    f"{destination}"
                )
            _atomic_json(destination, value)
    else:
        _atomic_json(destination, value)
    return destination, candidate_ids


def _materialize_single_retry_manifest(
    *, source_manifest: Path, candidate_id: str, destination: Path
) -> Path:
    return _materialize_subset_retry_manifest(
        source_manifest=source_manifest,
        candidate_ids=(candidate_id,),
        destination=destination,
        execution_kind="single_candidate_confirmation",
    )


def _materialize_subset_retry_manifest(
    *,
    source_manifest: Path,
    candidate_ids: tuple[str, ...],
    destination: Path,
    execution_kind: str,
) -> Path:
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    jobs = payload.get("jobs") if isinstance(payload, dict) else None
    selected_by_id = (
        {str(job.get("name")): job for job in jobs}
        if isinstance(jobs, list)
        else {}
    )
    if (
        not candidate_ids
        or len(candidate_ids) != len(set(candidate_ids))
        or any(candidate_id not in selected_by_id for candidate_id in candidate_ids)
    ):
        raise SchemaValidationError(
            "R1 v9 retry candidate set is missing or not unique"
        )
    selected = [selected_by_id[candidate_id] for candidate_id in candidate_ids]
    value = {
        **{key: item for key, item in payload.items() if key != "jobs"},
        "execution_kind": execution_kind,
        "parent_run_manifest_path": _ledger_path(source_manifest),
        "parent_run_manifest_sha256": hash_file(source_manifest),
        "candidate_count": len(selected),
        "jobs": selected,
    }
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != value:
            raise SchemaValidationError(
                f"R1 v9 retry manifest already exists with other bytes: {destination}"
            )
    else:
        _atomic_json(destination, value)
    return destination


def _validate_candidate_output(
    candidate_id: str,
    *,
    output_root: Path,
    wrapper_sha256: str,
    runtime_sha256: str,
) -> dict[str, Any]:
    case_root = output_root / candidate_id / "isaac"
    raw_path = case_root / "evaluation_rollout.pt"
    episode_path = case_root / "evaluation_episodes.jsonl"
    if not raw_path.is_file() or not episode_path.is_file():
        raise SchemaValidationError(f"R1 v9 output is missing: {candidate_id}")
    payload = torch.load(raw_path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    tensors = payload.get("tensors") if isinstance(payload, dict) else None
    if (
        not isinstance(metadata, dict)
        or not isinstance(tensors, dict)
        or metadata.get("r1_batched_nominal_runtime_version") != BATCH_VERSION
        or metadata.get("r1_batched_nominal_wrapper_sha256") != wrapper_sha256
        or metadata.get("r1_batched_nominal_runtime_sha256") != runtime_sha256
        or metadata.get("r1_batched_environment_origin_packed") is not False
        or abs(
            float(metadata.get("r1_batched_environment_spacing_m", -1.0))
            - BATCH_ENV_SPACING_M
        )
        > 1.0e-12
        or not 1 <= int(metadata.get("r1_batched_candidate_count", -1)) <= 4
        or int(metadata.get("r1_batched_maximum_candidates_per_scene", -1)) != 4
        or metadata.get("pi_l_actor_command_applied") is not False
        or metadata.get("diagnostic_nominal_qpid_only") is not True
        or metadata.get("formal_phase_zero_start") is not True
    ):
        raise SchemaValidationError(f"R1 v9 raw contract differs: {candidate_id}")
    for key in _ACTION_TENSORS:
        value = tensors.get(key)
        if (
            not isinstance(value, torch.Tensor)
            or not torch.isfinite(value).all()
            or bool(torch.count_nonzero(value).item())
        ):
            raise SchemaValidationError(f"R1 v9 pi_L action differs: {candidate_id}:{key}")
    for key in _ACTIVE_CONTROLLER_TENSORS:
        value = tensors.get(key)
        if (
            not isinstance(value, torch.Tensor)
            or not torch.isfinite(value).all()
            or not bool(torch.count_nonzero(value).item())
        ):
            raise SchemaValidationError(
                f"R1 v9 controller evidence is empty: {candidate_id}:{key}"
            )
    records = [
        json.loads(line)
        for line in episode_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != REPLAY_COUNT:
        raise SchemaValidationError(f"R1 v9 replay count differs: {candidate_id}")
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
            raise SchemaValidationError(f"R1 v9 Isaac replay failed: {candidate_id}")
    return {
        "candidate_id": candidate_id,
        "raw_rollout": {
            "path": _ledger_path(raw_path),
            "sha256": hash_file(raw_path),
        },
        "episodes": {
            "path": _ledger_path(episode_path),
            "sha256": hash_file(episode_path),
        },
        "episode_count": REPLAY_COUNT,
        "success_count": REPLAY_COUNT,
        "safety_failure_count": 0,
        "fallback_count": 0,
    }


def _validate_output_root(
    candidate_ids: tuple[str, ...], *, output_root: Path
) -> tuple[dict[str, Any], ...]:
    wrapper_sha = hash_file(WRAPPER)
    runtime_sha = hash_file(RUNTIME)
    return tuple(
        _validate_candidate_output(
            candidate_id,
            output_root=output_root,
            wrapper_sha256=wrapper_sha,
            runtime_sha256=runtime_sha,
        )
        for candidate_id in candidate_ids
    )


def _validate_legacy_candidate_output(candidate_id: str) -> dict[str, Any]:
    case_root = LEGACY_V7_OUTPUT_ROOT / candidate_id
    manifest_path = case_root / "case_manifest.json"
    raw_path = case_root / "isaac/evaluation_rollout.pt"
    episode_path = case_root / "isaac/evaluation_episodes.jsonl"
    if (
        not manifest_path.is_file()
        or not raw_path.is_file()
        or not episode_path.is_file()
    ):
        raise SchemaValidationError(f"R1 v9 legacy output is missing: {candidate_id}")
    case = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = torch.load(raw_path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    tensors = payload.get("tensors") if isinstance(payload, dict) else None
    nominal = (
        metadata.get("c3_nominal_reference") if isinstance(metadata, dict) else None
    )
    if (
        case.get("candidate_id") != candidate_id
        or int(case.get("replay_count", -1)) != REPLAY_COUNT
        or not isinstance(metadata, dict)
        or not isinstance(tensors, dict)
        or not isinstance(nominal, dict)
        or metadata.get("r1_batched_nominal_runtime_version")
        != "order9_r1_environment_wise_nominal_isaac_batch_v1"
        or metadata.get("r1_batched_nominal_wrapper_sha256")
        not in LEGACY_BATCH_WRAPPER_SHA256S
        or metadata.get("r1_batched_nominal_runtime_sha256")
        != LEGACY_BATCH_RUNTIME_SHA256
        or metadata.get("pi_l_actor_command_applied") is not False
        or metadata.get("diagnostic_nominal_qpid_only") is not True
        or metadata.get("formal_phase_zero_start") is not True
        or nominal.get("set_manifest_sha256")
        != case.get("nominal_set", {}).get("sha256")
        or nominal.get("artifact_sha256")
        != case.get("nominal_artifact", {}).get("sha256")
    ):
        raise SchemaValidationError(
            f"R1 v9 legacy raw contract differs: {candidate_id}"
        )
    for key in _ACTION_TENSORS:
        value = tensors.get(key)
        if (
            not isinstance(value, torch.Tensor)
            or not torch.isfinite(value).all()
            or bool(torch.count_nonzero(value).item())
        ):
            raise SchemaValidationError(
                f"R1 v9 legacy pi_L action differs: {candidate_id}:{key}"
            )
    for key in _ACTIVE_CONTROLLER_TENSORS:
        value = tensors.get(key)
        if (
            not isinstance(value, torch.Tensor)
            or not torch.isfinite(value).all()
            or not bool(torch.count_nonzero(value).item())
        ):
            raise SchemaValidationError(
                f"R1 v9 legacy controller evidence is empty: {candidate_id}:{key}"
            )
    records = [
        json.loads(line)
        for line in episode_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != REPLAY_COUNT:
        raise SchemaValidationError(
            f"R1 v9 legacy replay count differs: {candidate_id}"
        )
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
            raise SchemaValidationError(
                f"R1 v9 legacy Isaac replay failed: {candidate_id}"
            )
    return {
        "candidate_id": candidate_id,
        "case_manifest": {
            "path": _ledger_path(manifest_path),
            "sha256": hash_file(manifest_path),
        },
        "raw_rollout": {
            "path": _ledger_path(raw_path),
            "sha256": hash_file(raw_path),
        },
        "episodes": {
            "path": _ledger_path(episode_path),
            "sha256": hash_file(episode_path),
        },
        "episode_count": REPLAY_COUNT,
        "success_count": REPLAY_COUNT,
        "safety_failure_count": 0,
        "fallback_count": 0,
        "legacy_batch_wrapper_sha256": metadata[
            "r1_batched_nominal_wrapper_sha256"
        ],
    }


def _legacy_is_complete(candidate_id: str) -> bool:
    try:
        _validate_legacy_candidate_output(candidate_id)
    except (OSError, RuntimeError, ValueError, SchemaValidationError):
        return False
    return True


def _is_complete(candidate_ids: tuple[str, ...], *, output_root: Path) -> bool:
    try:
        _validate_output_root(candidate_ids, output_root=output_root)
    except (OSError, RuntimeError, ValueError, SchemaValidationError):
        return False
    return True


def _existing_retry_evidence_root(
    candidate_id: str,
    *,
    source_id: str,
    output_root: Path,
) -> Path | None:
    """Return already validated retry evidence before scheduling new Isaac work."""

    single_root = output_root / "single_retries" / candidate_id
    if _is_complete((candidate_id,), output_root=single_root):
        return single_root
    bounded_root = output_root / "bounded_retries" / source_id
    if bounded_root.is_dir():
        for candidate_root in sorted(bounded_root.glob(f"*/{candidate_id}")):
            retry_root = candidate_root.parent
            if _is_complete((candidate_id,), output_root=retry_root):
                return retry_root
    return None


def _any_candidate_output_materialized(
    candidate_ids: tuple[str, ...], *, output_root: Path
) -> bool:
    return any(
        (output_root / candidate_id / "isaac/evaluation_rollout.pt").is_file()
        or (output_root / candidate_id / "isaac/evaluation_episodes.jsonl").is_file()
        for candidate_id in candidate_ids
    )


def main() -> int:
    args = _parser().parse_args()
    if not 1 <= args.maximum_parallel_batches <= 4:
        raise ValueError("R1 v9 parallel batch count must be in [1, 4]")
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifests = _select_source_bucket_manifests(SOURCE_ROOT)
    if len(manifests) != EXPECTED_SOURCE_BUCKET_COUNT:
        raise SchemaValidationError("R1 v9 source bucket count differs")
    exclusions = _excluded_candidate_ids()
    queue: list[tuple[str, Path, Path, tuple[str, ...]]] = []
    source_evidence_roots: dict[str, Path] = {}
    candidates_by_source: dict[str, tuple[str, ...]] = {}
    run_manifests: dict[str, Path] = {}
    for source_id, source_manifest in sorted(manifests.items()):
        selected_source = (
            TRAIN003_REPAIR_MANIFEST
            if source_id == TRAIN003_SOURCE_ID
            else source_manifest
        )
        source_exclusions = frozenset(
            value
            for value in exclusions
            if any(job.get("name") == value for job in _manifest_jobs(selected_source))
        )
        manifest, candidate_ids = _materialize_run_manifest(
            source_id=source_id,
            source_manifest=selected_source,
            destination=output_root / "run_manifests" / f"{source_id}.json",
            excluded_ids=source_exclusions,
        )
        candidates_by_source[source_id] = candidate_ids
        run_manifests[source_id] = manifest
        candidate_output_root = output_root / "source_buckets" / source_id
        source_evidence_roots[source_id] = candidate_output_root

    if sum(map(len, candidates_by_source.values())) != EXPECTED_TOTAL_CANDIDATE_COUNT:
        raise SchemaValidationError("R1 v9 total selected candidate count differs")

    active: dict[str, tuple[subprocess.Popen[str], Any, Path, tuple[str, ...], Path]] = {}
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()
    while queue or active:
        while queue and len(active) < args.maximum_parallel_batches:
            source_id, manifest, candidate_output_root, candidate_ids = queue.pop(0)
            candidate_output_root.mkdir(parents=True, exist_ok=True)
            log_path = output_root / "logs" / f"{source_id}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            stream = log_path.open("w", encoding="utf-8")
            command = [
                str(Path(args.isaac_python).resolve()),
                str(WRAPPER),
                "--r1-batched-jobs",
                str(manifest),
                "--r1-batched-output-root",
                str(candidate_output_root),
                "--no-formal-annotation",
            ]
            process = subprocess.Popen(
                command,
                cwd=REPOSITORY,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active[source_id] = (
                process,
                stream,
                log_path,
                candidate_ids,
                candidate_output_root,
            )
            print(
                "ORDER9_R1_V9_STARTED="
                + json.dumps(
                    {
                        "source_bucket_id": source_id,
                        "candidate_count": len(candidate_ids),
                        "active_batch_count": len(active),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        time.sleep(1.0)
        for source_id, entry in tuple(active.items()):
            process, stream, log_path, candidate_ids, candidate_output_root = entry
            return_code = process.poll()
            if return_code is None:
                continue
            stream.close()
            del active[source_id]
            if return_code != 0:
                failures.append(
                    {
                        "source_bucket_id": source_id,
                        "error": f"child return code {return_code}",
                        "log_path": str(log_path),
                    }
                )
            print(
                "ORDER9_R1_V9_FINISHED="
                + json.dumps(
                    {
                        "source_bucket_id": source_id,
                        "return_code": return_code,
                        "remaining_batch_count": len(queue) + len(active),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if failures:
        _atomic_json(output_root / "failed_batches.json", {"failures": failures})
        raise RuntimeError(f"R1 v9 bounded validation failed: {failures}")

    candidate_evidence_roots: dict[str, Path] = {}
    invalid_by_source: dict[str, list[str]] = {}
    for source_id in sorted(candidates_by_source):
        source_root = source_evidence_roots[source_id]
        for candidate_id in candidates_by_source[source_id]:
            if _is_complete((candidate_id,), output_root=source_root):
                candidate_evidence_roots[candidate_id] = source_root
            elif (
                source_id != TRAIN003_SOURCE_ID
                and _legacy_is_complete(candidate_id)
            ):
                candidate_evidence_roots[candidate_id] = LEGACY_V7_OUTPUT_ROOT
            else:
                retry_root = _existing_retry_evidence_root(
                    candidate_id,
                    source_id=source_id,
                    output_root=output_root,
                )
                if retry_root is not None:
                    candidate_evidence_roots[candidate_id] = retry_root
                else:
                    invalid_by_source.setdefault(source_id, []).append(candidate_id)

    bounded_queue: list[tuple[str, str, Path, Path, tuple[str, ...]]] = []
    for source_id in sorted(invalid_by_source):
        values = invalid_by_source[source_id]
        for chunk_index, start in enumerate(range(0, len(values), 4)):
            candidate_ids = tuple(values[start : start + 4])
            candidate_digest = hashlib.sha256(
                "\n".join(candidate_ids).encode("utf-8")
            ).hexdigest()[:12]
            chunk_name = f"chunk_{chunk_index:03d}_{candidate_digest}"
            retry_id = f"{source_id}:{chunk_name}"
            retry_root = (
                output_root
                / "bounded_retries"
                / source_id
                / chunk_name
            )
            if _is_complete(candidate_ids, output_root=retry_root):
                candidate_evidence_roots.update(
                    {candidate_id: retry_root for candidate_id in candidate_ids}
                )
                continue
            retry_manifest = _materialize_subset_retry_manifest(
                source_manifest=run_manifests[source_id],
                candidate_ids=candidate_ids,
                destination=(
                    output_root
                    / "bounded_retry_manifests"
                    / source_id
                    / f"{chunk_name}.json"
                ),
                execution_kind="bounded_four_candidate_confirmation",
            )
            bounded_queue.append(
                (retry_id, source_id, retry_manifest, retry_root, candidate_ids)
            )

    bounded_active: dict[
        str, tuple[
            subprocess.Popen[str], Any, Path, str, Path, tuple[str, ...]
        ]
    ] = {}
    bounded_retry_count = len(bounded_queue)
    bounded_process_failures: list[dict[str, Any]] = []
    while bounded_queue or bounded_active:
        while bounded_queue and len(bounded_active) < args.maximum_parallel_batches:
            retry_id, source_id, manifest, retry_root, candidate_ids = (
                bounded_queue.pop(0)
            )
            retry_root.mkdir(parents=True, exist_ok=True)
            log_path = output_root / "bounded_retry_logs" / f"{retry_id}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            stream = log_path.open("w", encoding="utf-8")
            command = [
                str(Path(args.isaac_python).resolve()),
                str(WRAPPER),
                "--r1-batched-jobs",
                str(manifest),
                "--r1-batched-output-root",
                str(retry_root),
                "--no-formal-annotation",
            ]
            process = subprocess.Popen(
                command,
                cwd=REPOSITORY,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
            bounded_active[retry_id] = (
                process,
                stream,
                log_path,
                source_id,
                retry_root,
                candidate_ids,
            )
            print(
                "ORDER9_R1_V9_BOUNDED_RETRY_STARTED="
                + json.dumps(
                    {
                        "retry_id": retry_id,
                        "source_bucket_id": source_id,
                        "candidate_count": len(candidate_ids),
                        "active_retry_count": len(bounded_active),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        time.sleep(1.0)
        for retry_id, entry in tuple(bounded_active.items()):
            process, stream, log_path, source_id, retry_root, candidate_ids = entry
            return_code = process.poll()
            if return_code is None:
                continue
            stream.close()
            del bounded_active[retry_id]
            if return_code != 0:
                bounded_process_failures.append(
                    {
                        "retry_id": retry_id,
                        "source_bucket_id": source_id,
                        "return_code": return_code,
                        "log_path": str(log_path),
                    }
                )
            for candidate_id in candidate_ids:
                if _is_complete((candidate_id,), output_root=retry_root):
                    candidate_evidence_roots[candidate_id] = retry_root
            print(
                "ORDER9_R1_V9_BOUNDED_RETRY_FINISHED="
                + json.dumps(
                    {
                        "retry_id": retry_id,
                        "return_code": return_code,
                        "remaining_retry_count": (
                            len(bounded_queue) + len(bounded_active)
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if bounded_process_failures:
        _atomic_json(
            output_root / "bounded_retry_process_failures.json",
            {"failures": bounded_process_failures},
        )

    retry_queue: list[tuple[str, str, Path, Path]] = []
    for source_id in sorted(candidates_by_source):
        for candidate_id in candidates_by_source[source_id]:
            if candidate_id in candidate_evidence_roots:
                continue
            retry_root = output_root / "single_retries" / candidate_id
            if _is_complete((candidate_id,), output_root=retry_root):
                candidate_evidence_roots[candidate_id] = retry_root
                continue
            retry_manifest = _materialize_single_retry_manifest(
                source_manifest=run_manifests[source_id],
                candidate_id=candidate_id,
                destination=(
                    output_root / "retry_manifests" / f"{candidate_id}.json"
                ),
            )
            retry_queue.append(
                (candidate_id, source_id, retry_manifest, retry_root)
            )

    active_retries: dict[
        str, tuple[subprocess.Popen[str], Any, Path, str, Path]
    ] = {}
    single_retry_count = len(retry_queue)
    retry_failures: list[dict[str, Any]] = []
    while retry_queue or active_retries:
        while retry_queue and len(active_retries) < args.maximum_parallel_batches:
            candidate_id, source_id, manifest, retry_root = retry_queue.pop(0)
            retry_root.mkdir(parents=True, exist_ok=True)
            log_path = output_root / "retry_logs" / f"{candidate_id}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            stream = log_path.open("w", encoding="utf-8")
            command = [
                str(Path(args.isaac_python).resolve()),
                str(WRAPPER),
                "--r1-batched-jobs",
                str(manifest),
                "--r1-batched-output-root",
                str(retry_root),
                "--no-formal-annotation",
            ]
            process = subprocess.Popen(
                command,
                cwd=REPOSITORY,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active_retries[candidate_id] = (
                process,
                stream,
                log_path,
                source_id,
                retry_root,
            )
            print(
                "ORDER9_R1_V9_SINGLE_RETRY_STARTED="
                + json.dumps(
                    {
                        "candidate_id": candidate_id,
                        "source_bucket_id": source_id,
                        "active_retry_count": len(active_retries),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        time.sleep(1.0)
        for candidate_id, entry in tuple(active_retries.items()):
            process, stream, log_path, source_id, retry_root = entry
            return_code = process.poll()
            if return_code is None:
                continue
            stream.close()
            del active_retries[candidate_id]
            try:
                if return_code != 0:
                    raise RuntimeError(f"child return code {return_code}")
                _validate_output_root((candidate_id,), output_root=retry_root)
            except (OSError, RuntimeError, ValueError, SchemaValidationError) as error:
                retry_failures.append(
                    {
                        "candidate_id": candidate_id,
                        "source_bucket_id": source_id,
                        "error": str(error),
                        "log_path": str(log_path),
                    }
                )
            else:
                candidate_evidence_roots[candidate_id] = retry_root
            print(
                "ORDER9_R1_V9_SINGLE_RETRY_FINISHED="
                + json.dumps(
                    {
                        "candidate_id": candidate_id,
                        "return_code": return_code,
                        "remaining_retry_count": (
                            len(retry_queue) + len(active_retries)
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if retry_failures:
        _atomic_json(
            output_root / "failed_single_retries.json",
            {"failures": retry_failures},
        )
        raise RuntimeError(f"R1 v9 single confirmations failed: {retry_failures}")

    entries = []
    for source_id in sorted(candidates_by_source):
        for candidate_id in candidates_by_source[source_id]:
            evidence_root = candidate_evidence_roots.get(candidate_id)
            if evidence_root is None:
                raise SchemaValidationError(
                    f"R1 v9 accepted evidence is missing: {candidate_id}"
                )
            entry = (
                _validate_legacy_candidate_output(candidate_id)
                if evidence_root == LEGACY_V7_OUTPUT_ROOT
                else _validate_candidate_output(
                    candidate_id,
                    output_root=evidence_root,
                    wrapper_sha256=hash_file(WRAPPER),
                    runtime_sha256=hash_file(RUNTIME),
                )
            )
            entries.append(
                {
                    **entry,
                    "source_bucket_id": source_id,
                    "single_candidate_confirmation": (
                        evidence_root.parent.name == "single_retries"
                    ),
                    "bounded_four_candidate_confirmation": (
                        "bounded_retries" in evidence_root.parts
                    ),
                    "legacy_v7_evidence_reused": (
                        evidence_root == LEGACY_V7_OUTPUT_ROOT
                    ),
                }
            )
    summary = {
        "runner_version": RUNNER_VERSION,
        "status": "passed",
        "source_bucket_count": len(candidates_by_source),
        "candidate_count": len(entries),
        "episode_count": sum(entry["episode_count"] for entry in entries),
        "success_count": sum(entry["success_count"] for entry in entries),
        "safety_failure_count": sum(
            entry["safety_failure_count"] for entry in entries
        ),
        "fallback_count": sum(entry["fallback_count"] for entry in entries),
        "pi_l_actor_command_applied": False,
        "qpid_qp_local_servo_applied": True,
        "environment_spacing_m": BATCH_ENV_SPACING_M,
        "wrapper_sha256": hash_file(WRAPPER),
        "runtime_sha256": hash_file(RUNTIME),
        "repaired_candidate_count_validated_separately": len(exclusions),
        "bounded_four_candidate_confirmation_batch_count": bounded_retry_count,
        "single_candidate_confirmation_count": single_retry_count,
        "legacy_v7_candidate_count": sum(
            entry["legacy_v7_evidence_reused"] for entry in entries
        ),
        "legacy_batch_wrapper_sha256s": sorted(LEGACY_BATCH_WRAPPER_SHA256S),
        "legacy_batch_runtime_sha256": LEGACY_BATCH_RUNTIME_SHA256,
        "source_manifests": {
            source_id: {
                "path": _ledger_path(run_manifests[source_id]),
                "sha256": hash_file(run_manifests[source_id]),
                "candidate_count": len(candidates_by_source[source_id]),
                "batch_output_root": _ledger_path(source_evidence_roots[source_id]),
            }
            for source_id in sorted(run_manifests)
        },
        "elapsed_s": time.perf_counter() - started,
        "entries": entries,
    }
    summary_path = _atomic_json(output_root / "result_ledger.json", summary)
    print(
        "ORDER9_R1_V9_COMPLETE="
        + json.dumps(
            {
                "candidate_count": len(entries),
                "episode_count": summary["episode_count"],
                "success_count": summary["success_count"],
                "result_ledger": str(summary_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
