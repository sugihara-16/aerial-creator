#!/usr/bin/env python3
from __future__ import annotations

"""Complete R1 v7 with bounded batches and single-candidate failure replay."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.schemas.common import SchemaValidationError  # noqa: E402
from amsrr.training.order9_r1_clearance_diagnostic import (  # noqa: E402
    write_order9_r1_clearance_diagnostic,
)
from amsrr.training.order9_r1_isaac_calibration import (  # noqa: E402
    load_order9_r1_isaac_case,
)
from amsrr.training.order9_r1_nominal_calibration_v7 import (  # noqa: E402
    _quarantine_incomplete_v7_isaac_output,
    load_order9_r1_nominal_calibration_v7_contract,
    validate_order9_r1_nominal_v7_isaac_result,
)
from amsrr.utils.hashing import hash_file  # noqa: E402

BATCH_WRAPPER = REPOSITORY / "scripts/order9_r1_batched_nominal_compression_rollout.py"
BATCH_RUNTIME = REPOSITORY / "amsrr/training/order9_r1_batched_nominal_runtime.py"
RUNNER_VERSION = "order9_r1_batched_minimum_level_runner_v3_tiered_retry"
MAXIMUM_CANDIDATES_PER_SCENE = 31
MAXIMUM_BOUNDED_RETRY_CANDIDATES_PER_SCENE = 4
BOUNDED_RETRY_REASON_TOKEN = "requires_bounded_batch_replay"
SINGLE_RETRY_REASON_TOKEN = "requires_single_candidate_replay"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--isaac-python",
        default="/home/leus/.local/share/mamba/envs/isaaclab3/bin/python",
    )
    parser.add_argument("--maximum-parallel-batches", type=int, default=2)
    parser.add_argument(
        "--excluded-candidates-json",
        help=(
            "Optional hash-bound v8 overlay input. Candidate IDs listed here are "
            "left untouched by this v7 executor and must be validated separately."
        ),
    )
    return parser


def _load_excluded_candidate_ids(path: str | Path | None) -> frozenset[str]:
    if path is None:
        return frozenset()
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    values = payload.get("candidate_ids") if isinstance(payload, dict) else None
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, str) or not value for value in values)
        or len(values) != len(set(values))
        or values != sorted(values)
    ):
        raise SchemaValidationError("R1 excluded candidate manifest is invalid")
    return frozenset(values)


def _candidate_manifest(job: dict[str, Any]) -> Path:
    argv = job["argv"]
    index = argv.index("--output-raw")
    return Path(argv[index + 1]).resolve().parents[1] / "case_manifest.json"


def _source_bucket_id(job: dict[str, Any]) -> str:
    payload = json.loads(_candidate_manifest(job).read_text(encoding="utf-8"))
    value = payload.get("source_bucket_id")
    if not isinstance(value, str) or not value:
        raise SchemaValidationError("R1 batched source bucket identity is missing")
    return value


def _select_source_bucket_manifests(batch_root: Path) -> dict[str, Path]:
    options: dict[str, list[Path]] = {}
    for path in sorted(batch_root.glob("*/persistent_bucket_jobs.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        jobs = payload.get("jobs") if isinstance(payload, dict) else None
        if not isinstance(jobs, list) or len(jobs) != 31:
            continue
        source_ids = {_source_bucket_id(job) for job in jobs}
        if len(source_ids) != 1:
            continue
        options.setdefault(next(iter(source_ids)), []).append(path)
    selected = {}
    for source_id, paths in options.items():
        preferred = [path for path in paths if "_chunk_" in path.parent.name]
        selected[source_id] = sorted(preferred or paths)[0]
    if len(selected) != 22:
        raise SchemaValidationError(
            f"R1 batched source-bucket manifest count differs: {len(selected)}"
        )
    return selected


def _valid_case(manifest_path: Path, protocol: dict[str, Any]) -> bool:
    materialized = load_order9_r1_isaac_case(manifest_path, REPOSITORY)
    try:
        results = validate_order9_r1_nominal_v7_isaac_result(
            materialized, repository_root=REPOSITORY, protocol=protocol
        )
    except (OSError, ValueError, SchemaValidationError):
        return False
    raw = __import__("torch").load(
        manifest_path.parent / "isaac/evaluation_rollout.pt",
        map_location="cpu",
        weights_only=False,
    )
    metadata = raw.get("metadata") if isinstance(raw, dict) else None
    return bool(
        isinstance(metadata, dict)
        and metadata.get("r1_batched_nominal_runtime_version")
        == "order9_r1_environment_wise_nominal_isaac_batch_v1"
        and metadata.get("r1_batched_nominal_wrapper_sha256")
        == hash_file(BATCH_WRAPPER)
        and metadata.get("r1_batched_nominal_runtime_sha256")
        == hash_file(BATCH_RUNTIME)
        and 1
        <= int(metadata.get("r1_batched_candidate_count", -1))
        <= MAXIMUM_CANDIDATES_PER_SCENE
        and all(result.success for result in results)
    )


def _existing_batch_candidate_count(manifest_path: Path) -> int | None:
    raw_path = manifest_path.parent / "isaac/evaluation_rollout.pt"
    if not raw_path.is_file():
        return None
    try:
        raw = __import__("torch").load(
            raw_path,
            map_location="cpu",
            weights_only=False,
        )
    except (OSError, RuntimeError, ValueError):
        return None
    metadata = raw.get("metadata") if isinstance(raw, dict) else None
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("r1_batched_candidate_count")
    return int(value) if isinstance(value, int) else None


def _quarantine_job(
    job: dict[str, Any],
    *,
    protocol: dict[str, Any],
    reason: str,
) -> None:
    manifest_path = _candidate_manifest(job)
    isaac = manifest_path.parent / "isaac"
    if not isaac.is_dir() or not any(isaac.iterdir()):
        return
    materialized = load_order9_r1_isaac_case(manifest_path, REPOSITORY)
    _quarantine_incomplete_v7_isaac_output(
        materialized,
        repository_root=REPOSITORY,
        protocol=protocol,
        reason=reason,
    )


def _prior_quarantine_has_reason_token(
    manifest_path: Path,
    protocol: dict[str, Any],
    token: str,
) -> bool:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate_id = manifest.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        return False
    quarantine_root = (
        REPOSITORY / str(protocol["interrupted_attempt_quarantine_root"]) / candidate_id
    )
    for record_path in quarantine_root.glob("attempt_*.json"):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if token in str(record.get("reason", "")):
            return True
    return False


def _prior_quarantine_requires_single_retry(
    manifest_path: Path,
    protocol: dict[str, Any],
) -> bool:
    return _prior_quarantine_has_reason_token(
        manifest_path,
        protocol,
        SINGLE_RETRY_REASON_TOKEN,
    )


def _write_pending_manifest(
    *,
    payload: dict[str, Any],
    source_manifest: Path,
    destination: Path,
    jobs: list[dict[str, Any]],
    source_pending_job_count: int,
    execution_kind: str,
    sequence_index: int,
) -> Path:
    return write_order9_r1_clearance_diagnostic(
        {
            **{key: value for key, value in payload.items() if key != "jobs"},
            "source_manifest_path": str(source_manifest),
            "source_manifest_sha256": hash_file(source_manifest),
            "source_pending_job_count": source_pending_job_count,
            "maximum_candidates_per_scene": MAXIMUM_CANDIDATES_PER_SCENE,
            "execution_kind": execution_kind,
            "execution_sequence_index": sequence_index,
            "pending_job_count": len(jobs),
            "jobs": jobs,
        },
        destination,
    )


def _prepare_pending_manifests(
    source_id: str,
    source_manifest: Path,
    *,
    protocol: dict[str, Any],
    output_root: Path,
    excluded_candidate_ids: frozenset[str] = frozenset(),
) -> tuple[tuple[Path, int], ...]:
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    pending = []
    bounded_retries = []
    single_retries = []
    for job in payload["jobs"]:
        name = job.get("name") if isinstance(job, dict) else None
        if name in excluded_candidate_ids:
            continue
        manifest_path = _candidate_manifest(job)
        if _valid_case(manifest_path, protocol):
            continue
        existing_batch_count = _existing_batch_candidate_count(manifest_path)
        requires_single_retry = _prior_quarantine_requires_single_retry(
            manifest_path, protocol
        ) or (
            existing_batch_count is not None
            and 1 < existing_batch_count <= MAXIMUM_BOUNDED_RETRY_CANDIDATES_PER_SCENE
        )
        requires_bounded_retry = _prior_quarantine_has_reason_token(
            manifest_path,
            protocol,
            BOUNDED_RETRY_REASON_TOKEN,
        ) or (
            existing_batch_count is not None
            and existing_batch_count > MAXIMUM_BOUNDED_RETRY_CANDIDATES_PER_SCENE
        )
        isaac = manifest_path.parent / "isaac"
        if isaac.is_dir() and any(isaac.iterdir()):
            _quarantine_job(
                job,
                protocol=protocol,
                reason=(
                    "batched_minimum_level_retry: existing evidence is invalid; "
                    + (
                        SINGLE_RETRY_REASON_TOKEN
                        if requires_single_retry
                        else (
                            BOUNDED_RETRY_REASON_TOKEN
                            if requires_bounded_retry
                            else "initial_batch_retry_allowed"
                        )
                    )
                ),
            )
        if requires_single_retry:
            single_retries.append(job)
        elif requires_bounded_retry:
            bounded_retries.append(job)
        else:
            pending.append(job)
    if not pending and not bounded_retries and not single_retries:
        return ()
    batch_dir = output_root / source_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    result = []
    source_pending_job_count = len(pending) + len(bounded_retries) + len(single_retries)
    for chunk_index, start in enumerate(
        range(0, len(pending), MAXIMUM_CANDIDATES_PER_SCENE)
    ):
        jobs = pending[start : start + MAXIMUM_CANDIDATES_PER_SCENE]
        chunk_dir = batch_dir / f"chunk_{chunk_index:03d}"
        pending_path = _write_pending_manifest(
            payload=payload,
            source_manifest=source_manifest,
            destination=chunk_dir / "pending_bucket_jobs.json",
            jobs=jobs,
            source_pending_job_count=source_pending_job_count,
            execution_kind="bounded_batch",
            sequence_index=chunk_index,
        )
        result.append((pending_path, len(jobs)))
    for retry_index, start in enumerate(
        range(0, len(bounded_retries), MAXIMUM_BOUNDED_RETRY_CANDIDATES_PER_SCENE)
    ):
        jobs = bounded_retries[
            start : start + MAXIMUM_BOUNDED_RETRY_CANDIDATES_PER_SCENE
        ]
        retry_dir = batch_dir / f"bounded_retry_{retry_index:03d}"
        pending_path = _write_pending_manifest(
            payload=payload,
            source_manifest=source_manifest,
            destination=retry_dir / "pending_bucket_jobs.json",
            jobs=jobs,
            source_pending_job_count=source_pending_job_count,
            execution_kind="bounded_batch_retry",
            sequence_index=retry_index,
        )
        result.append((pending_path, len(jobs)))
    for retry_index, job in enumerate(single_retries):
        retry_dir = batch_dir / f"single_retry_{retry_index:03d}"
        pending_path = _write_pending_manifest(
            payload=payload,
            source_manifest=source_manifest,
            destination=retry_dir / "pending_bucket_jobs.json",
            jobs=[job],
            source_pending_job_count=source_pending_job_count,
            execution_kind="single_candidate_retry",
            sequence_index=retry_index,
        )
        result.append((pending_path, 1))
    return tuple(result)


def _tail(path: Path, line_count: int = 30) -> str:
    if not path.is_file():
        return ""
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace").splitlines()[-line_count:]
    )


def main() -> int:
    args = _parser().parse_args()
    if not 1 <= args.maximum_parallel_batches <= 4:
        raise ValueError("R1 batched parallel count must be in [1, 4]")
    protocol, _approval = load_order9_r1_nominal_calibration_v7_contract(REPOSITORY)
    excluded_candidate_ids = _load_excluded_candidate_ids(args.excluded_candidates_json)
    output_root = REPOSITORY / str(protocol["output_root"])
    source_manifests = _select_source_bucket_manifests(
        output_root / "persistent_process_batches"
    )
    run_root = output_root / "batched_process_batches"
    run_root.mkdir(parents=True, exist_ok=True)
    queue = []
    total_pending = 0
    for source_id, source_manifest in sorted(source_manifests.items()):
        pending_chunks = _prepare_pending_manifests(
            source_id,
            source_manifest,
            protocol=protocol,
            output_root=run_root,
            excluded_candidate_ids=excluded_candidate_ids,
        )
        for pending, count in pending_chunks:
            batch_id = f"{source_id}:{pending.parent.name}"
            queue.append((batch_id, source_id, pending, count))
            total_pending += count
    single_retry_count = sum(
        count
        for _batch_id, _source_id, pending, count in queue
        if "single_retry" in pending.parent.name
    )
    bounded_retry_count = sum(
        count
        for _batch_id, _source_id, pending, count in queue
        if "bounded_retry" in pending.parent.name
    )
    started = time.perf_counter()
    active: dict[str, tuple[subprocess.Popen, Any, Path, Path, str, int]] = {}
    completed = 0
    failures = []
    runtime_bounded_retry_index = 0
    runtime_single_retry_index = 0
    while queue or active:
        while queue and len(active) < args.maximum_parallel_batches:
            batch_id, source_id, manifest, count = queue.pop(0)
            log_path = manifest.parent / "batch_process.log"
            log_stream = log_path.open("w", encoding="utf-8")
            command = [
                str(Path(args.isaac_python).resolve()),
                str(BATCH_WRAPPER),
                "--r1-batched-jobs",
                str(manifest),
            ]
            process = subprocess.Popen(
                command,
                cwd=REPOSITORY,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active[batch_id] = (
                process,
                log_stream,
                log_path,
                manifest,
                source_id,
                count,
            )
            print(
                "ORDER9_R1_BATCH_STARTED="
                + json.dumps(
                    {
                        "source_bucket_id": source_id,
                        "batch_id": batch_id,
                        "candidate_count": count,
                        "active_batch_count": len(active),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        time.sleep(1.0)
        for batch_id, active_entry in list(active.items()):
            process, stream, log_path, manifest, source_id, count = active_entry
            return_code = process.poll()
            if return_code is None:
                continue
            stream.close()
            del active[batch_id]
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            completed_marker = "ORDER9_R1_BATCH_COMPLETE=" in log_text
            traceback_present = "Traceback (most recent call last):" in log_text
            if return_code != 0 or not completed_marker or traceback_present:
                failures.append(
                    {
                        "source_bucket_id": source_id,
                        "batch_id": batch_id,
                        "return_code": return_code,
                        "completion_marker_present": completed_marker,
                        "traceback_present": traceback_present,
                        "log_path": str(log_path),
                        "log_tail": _tail(log_path),
                    }
                )
            else:
                manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
                jobs = manifest_payload.get("jobs")
                if not isinstance(jobs, list) or len(jobs) != count:
                    raise SchemaValidationError(
                        "R1 completed batch manifest job count differs"
                    )
                valid_jobs = [
                    job
                    for job in jobs
                    if _valid_case(_candidate_manifest(job), protocol)
                ]
                completed += len(valid_jobs)
                invalid_jobs = [job for job in jobs if job not in valid_jobs]
                if invalid_jobs and count == 1:
                    failures.append(
                        {
                            "source_bucket_id": source_id,
                            "batch_id": batch_id,
                            "return_code": return_code,
                            "completion_marker_present": completed_marker,
                            "traceback_present": traceback_present,
                            "log_path": str(log_path),
                            "log_tail": _tail(log_path),
                            "failure_reason": (
                                "single-candidate replay did not satisfy the v7 gate"
                            ),
                        }
                    )
                elif invalid_jobs:
                    source_manifest = Path(
                        str(manifest_payload["source_manifest_path"])
                    ).resolve()
                    bounded_retry = count > MAXIMUM_BOUNDED_RETRY_CANDIDATES_PER_SCENE
                    retry_limit = (
                        MAXIMUM_BOUNDED_RETRY_CANDIDATES_PER_SCENE
                        if bounded_retry
                        else 1
                    )
                    reason_token = (
                        BOUNDED_RETRY_REASON_TOKEN
                        if bounded_retry
                        else SINGLE_RETRY_REASON_TOKEN
                    )
                    for job in invalid_jobs:
                        _quarantine_job(
                            job,
                            protocol=protocol,
                            reason=(
                                "batched_minimum_level_retry: bounded-batch result "
                                "failed; " + reason_token
                            ),
                        )
                    retry_entries = []
                    for start in range(0, len(invalid_jobs), retry_limit):
                        retry_jobs = invalid_jobs[start : start + retry_limit]
                        if bounded_retry:
                            retry_index = runtime_bounded_retry_index
                            retry_name = f"runtime_bounded_retry_{retry_index:04d}"
                            execution_kind = "bounded_batch_runtime_retry"
                            runtime_bounded_retry_index += 1
                            bounded_retry_count += len(retry_jobs)
                        else:
                            retry_index = runtime_single_retry_index
                            retry_name = f"runtime_single_retry_{retry_index:04d}"
                            execution_kind = "single_candidate_runtime_retry"
                            runtime_single_retry_index += 1
                            single_retry_count += len(retry_jobs)
                        retry_dir = run_root / source_id / retry_name
                        retry_manifest = _write_pending_manifest(
                            payload=manifest_payload,
                            source_manifest=source_manifest,
                            destination=retry_dir / "pending_bucket_jobs.json",
                            jobs=retry_jobs,
                            source_pending_job_count=total_pending,
                            execution_kind=execution_kind,
                            sequence_index=retry_index,
                        )
                        retry_entries.append(
                            (
                                f"{source_id}:{retry_name}",
                                source_id,
                                retry_manifest,
                                len(retry_jobs),
                            )
                        )
                    queue[0:0] = retry_entries
            print(
                "ORDER9_R1_BATCH_FINISHED="
                + json.dumps(
                    {
                        "source_bucket_id": source_id,
                        "batch_id": batch_id,
                        "candidate_count": count,
                        "return_code": return_code,
                        "completed_candidate_count": completed,
                        "pending_candidate_count": total_pending - completed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if failures:
        raise RuntimeError("R1 batched Isaac failures: " + json.dumps(failures))
    if not excluded_candidate_ids:
        final_command = [
            str(Path(args.isaac_python).resolve()),
            str(REPOSITORY / "scripts/order9_run_r1_nominal_calibration_v7.py"),
            "--maximum-parallel-process-count",
            "1",
        ]
        final = subprocess.run(
            final_command,
            cwd=REPOSITORY,
            text=True,
            capture_output=True,
        )
        if final.returncode != 0:
            raise RuntimeError(
                "R1 v7 final aggregation failed:\n" + final.stdout + "\n" + final.stderr
            )
    elapsed = time.perf_counter() - started
    retained_candidate_count = 682 - len(excluded_candidate_ids)
    bindings = {
        "batch_wrapper": {
            "path": str(BATCH_WRAPPER),
            "sha256": hash_file(BATCH_WRAPPER),
        },
        "batch_runtime": {
            "path": str(BATCH_RUNTIME),
            "sha256": hash_file(BATCH_RUNTIME),
        },
    }
    if excluded_candidate_ids:
        excluded_path = Path(args.excluded_candidates_json).resolve()
        bindings["v8_excluded_candidates"] = {
            "path": str(excluded_path),
            "sha256": hash_file(excluded_path),
        }
    else:
        bindings.update(
            {
                "v7_level_result": {
                    "path": str(output_root / "level_result_v7.json"),
                    "sha256": hash_file(output_root / "level_result_v7.json"),
                },
                "v7_run_result": {
                    "path": str(output_root / "run_result_v7.json"),
                    "sha256": hash_file(output_root / "run_result_v7.json"),
                },
            }
        )
    result_path = write_order9_r1_clearance_diagnostic(
        {
            "run_version": RUNNER_VERSION,
            "status": (
                "completed_v7"
                if not excluded_candidate_ids
                else "completed_v7_subset_for_v8_overlay"
            ),
            "source_bucket_count": len(source_manifests),
            "candidate_count": retained_candidate_count,
            "episode_count": retained_candidate_count * 2,
            "excluded_candidate_count": len(excluded_candidate_ids),
            "excluded_candidate_ids": sorted(excluded_candidate_ids),
            "newly_executed_candidate_count": total_pending,
            "maximum_parallel_batches": args.maximum_parallel_batches,
            "maximum_candidates_per_scene": MAXIMUM_CANDIDATES_PER_SCENE,
            "maximum_bounded_retry_candidates_per_scene": (
                MAXIMUM_BOUNDED_RETRY_CANDIDATES_PER_SCENE
            ),
            "bounded_batch_retry_count": bounded_retry_count,
            "single_candidate_retry_count": single_retry_count,
            "batch_failure_replay_contract": (
                "order9_r1_large_batch_bounded_batch_single_candidate_replay_v1"
            ),
            "measured_wall_time_s": elapsed,
            "pi_l_actor_command_applied": False,
            "bindings": bindings,
        },
        output_root
        / (
            "batched_run_result_v7.json"
            if not excluded_candidate_ids
            else "batched_run_result_v7_subset_for_v8.json"
        ),
    )
    print(
        "ORDER9_R1_BATCHED_MINIMUM_LEVEL_COMPLETE="
        + json.dumps(
            {
                "candidate_count": retained_candidate_count,
                "episode_count": retained_candidate_count * 2,
                "excluded_candidate_count": len(excluded_candidate_ids),
                "measured_wall_time_s": elapsed,
                "result_path": str(result_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
