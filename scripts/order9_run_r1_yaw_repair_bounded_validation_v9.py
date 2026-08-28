#!/usr/bin/env python3
from __future__ import annotations

"""Revalidate the nine R1 yaw-repaired paths in Isaac batches of at most four."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.schemas.common import SchemaValidationError  # noqa: E402
from amsrr.training.order9_r1_nominal_calibration_v8 import (  # noqa: E402
    validate_order9_r1_nominal_v8_repair_evidence,
)
from amsrr.utils.config import load_config  # noqa: E402
from amsrr.utils.hashing import hash_file  # noqa: E402
from scripts.order9_r1_batched_nominal_compression_rollout import (  # noqa: E402
    BATCH_ENV_SPACING_M,
)
from scripts.order9_run_r1_origin_packed_validation_v9 import (  # noqa: E402
    REPAIRED_EXCLUSION,
    RUNTIME,
    WRAPPER,
    _atomic_json,
    _ledger_path,
    _validate_candidate_output,
)

RUNNER_VERSION = "order9_r1_yaw_repair_bounded_revalidation_v1"
SOURCE_JOBS = REPOSITORY / (
    "artifacts/p4_full/order9/r1_teacher/diagnostics/"
    "r1_v7_train000002_yaw_branch_repair_v3/"
    "all_yaw_repaired_candidate_jobs.json"
)
V8_PROTOCOL = REPOSITORY / (
    "configs/training/order9_r1_nominal_calibration_protocol_v8.yaml"
)
V8_REPAIR_LEDGER = REPOSITORY / (
    "for_codex/R1_NOMINAL_CALIBRATION_V8_REPAIR_EVIDENCE_LEDGER.json"
)
OUTPUT_ROOT = REPOSITORY / (
    "artifacts/p4_full/order9/r1_teacher/validation_v10_bounded_four/"
    "r1_l1_10mm_5deg/yaw_repaired_bounded_v1"
)
RESULT_LEDGER = OUTPUT_ROOT / "result_ledger.json"
EXPECTED_CANDIDATE_COUNT = 9
REPLAY_COUNT = 2
MAXIMUM_CANDIDATES_PER_SCENE = 4
SOURCE_BUCKET_ID = "train-000002-0e86a4ff3e71"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--isaac-python",
        default="/home/leus/.local/share/mamba/envs/isaaclab3/bin/python",
    )
    parser.add_argument("--maximum-parallel-batches", type=int, default=3)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    return parser


def _binding(path: Path) -> dict[str, str]:
    return {"path": _ledger_path(path), "sha256": hash_file(path)}


def _expected_candidate_ids() -> tuple[str, ...]:
    payload = json.loads(REPAIRED_EXCLUSION.read_text(encoding="utf-8"))
    values = payload.get("candidate_ids") if isinstance(payload, dict) else None
    if (
        not isinstance(values, list)
        or len(values) != EXPECTED_CANDIDATE_COUNT
        or values != sorted(values)
        or len(values) != len(set(values))
    ):
        raise SchemaValidationError("R1 yaw-repair exclusion set differs")
    return tuple(str(value) for value in values)


def _source_payload() -> dict[str, Any]:
    payload = json.loads(SOURCE_JOBS.read_text(encoding="utf-8"))
    jobs = payload.get("jobs") if isinstance(payload, dict) else None
    expected = _expected_candidate_ids()
    if (
        not isinstance(jobs, list)
        or len(jobs) != EXPECTED_CANDIDATE_COUNT
        or tuple(sorted(str(job.get("name")) for job in jobs)) != expected
        or any(
            "--diagnostic-nominal-qpid-only" not in job.get("argv", [])
            for job in jobs
        )
    ):
        raise SchemaValidationError("R1 yaw-repair source jobs differ")
    return payload


def _job_chunks(payload: Mapping[str, Any]) -> tuple[tuple[dict[str, Any], ...], ...]:
    jobs = sorted(payload["jobs"], key=lambda value: str(value["name"]))
    chunks = tuple(
        tuple(jobs[start : start + MAXIMUM_CANDIDATES_PER_SCENE])
        for start in range(0, len(jobs), MAXIMUM_CANDIDATES_PER_SCENE)
    )
    if (
        tuple(len(chunk) for chunk in chunks) != (4, 4, 1)
        or tuple(
            sorted(str(job["name"]) for chunk in chunks for job in chunk)
        )
        != _expected_candidate_ids()
    ):
        raise SchemaValidationError("R1 yaw-repair bounded chunks differ")
    return chunks


def _materialize_chunk_manifest(
    payload: Mapping[str, Any],
    *,
    chunk_index: int,
    jobs: tuple[dict[str, Any], ...],
    destination: Path,
) -> Path:
    value = {
        **{key: item for key, item in payload.items() if key != "jobs"},
        "runner_version": RUNNER_VERSION,
        "execution_kind": "bounded_four_yaw_repair_revalidation",
        "source_jobs": _binding(SOURCE_JOBS),
        "chunk_index": chunk_index,
        "candidate_count": len(jobs),
        "maximum_candidates_per_scene": MAXIMUM_CANDIDATES_PER_SCENE,
        "environment_spacing_m": BATCH_ENV_SPACING_M,
        "jobs": list(jobs),
    }
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != value:
            raise SchemaValidationError(
                f"R1 yaw-repair chunk manifest differs: {destination}"
            )
        return destination
    return _atomic_json(destination, value)


def _entry_for_candidate(
    candidate_id: str,
    *,
    evidence_root: Path,
    batch_id: str,
    batch_manifest: Path,
    repair_binding: Mapping[str, Any],
) -> dict[str, Any]:
    result = _validate_candidate_output(
        candidate_id,
        output_root=evidence_root,
        wrapper_sha256=hash_file(WRAPPER),
        runtime_sha256=hash_file(RUNTIME),
    )
    return {
        **result,
        "source_bucket_id": SOURCE_BUCKET_ID,
        "evidence_batch_id": batch_id,
        "batch_manifest": _binding(batch_manifest),
        "case_manifest": dict(repair_binding["case_manifest"]),
        "lightweight_admission": dict(repair_binding["lightweight_admission"]),
        "pi_l_actor_command_applied": False,
        "qpid_qp_applied": True,
        "local_servo_applied": True,
    }


def validate_result_ledger(
    path: str | Path = RESULT_LEDGER,
) -> tuple[dict[str, Any], ...]:
    ledger_path = Path(path).resolve()
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    entries = ledger.get("entries") if isinstance(ledger, dict) else None
    if (
        ledger.get("runner_version") != RUNNER_VERSION
        or ledger.get("status") != "passed"
        or int(ledger.get("candidate_count", -1)) != EXPECTED_CANDIDATE_COUNT
        or int(ledger.get("episode_count", -1))
        != EXPECTED_CANDIDATE_COUNT * REPLAY_COUNT
        or int(ledger.get("success_count", -1))
        != EXPECTED_CANDIDATE_COUNT * REPLAY_COUNT
        or int(ledger.get("safety_failure_count", -1)) != 0
        or int(ledger.get("fallback_count", -1)) != 0
        or int(ledger.get("maximum_candidates_per_scene", -1))
        != MAXIMUM_CANDIDATES_PER_SCENE
        or abs(
            float(ledger.get("environment_spacing_m", -1.0))
            - BATCH_ENV_SPACING_M
        )
        > 1.0e-12
        or ledger.get("pi_l_actor_command_applied") is not False
        or ledger.get("qpid_qp_applied") is not True
        or ledger.get("local_servo_applied") is not True
        or not isinstance(entries, list)
        or len(entries) != EXPECTED_CANDIDATE_COUNT
    ):
        raise SchemaValidationError("R1 yaw-repair bounded result differs")
    for key, expected_path in (
        ("source_jobs", SOURCE_JOBS),
        ("repair_evidence_ledger", V8_REPAIR_LEDGER),
        ("wrapper_implementation", WRAPPER),
        ("runtime_implementation", RUNTIME),
        ("runner_implementation", Path(__file__).resolve()),
    ):
        binding = ledger.get("bindings", {}).get(key)
        if (
            not isinstance(binding, dict)
            or Path(str(binding.get("path", ""))) != expected_path.relative_to(REPOSITORY)
            or binding.get("sha256") != hash_file(expected_path)
        ):
            raise SchemaValidationError(f"R1 yaw-repair binding differs: {key}")

    repair_ledger = json.loads(V8_REPAIR_LEDGER.read_text(encoding="utf-8"))
    repairs = {entry["candidate_id"]: entry for entry in repair_ledger["entries"]}
    validated = []
    for entry in entries:
        candidate_id = entry.get("candidate_id") if isinstance(entry, dict) else None
        if candidate_id not in repairs:
            raise SchemaValidationError("R1 yaw-repair candidate differs")
        for key in ("batch_manifest", "case_manifest", "lightweight_admission"):
            binding = entry.get(key)
            source = REPOSITORY / str(binding.get("path", "")) if isinstance(binding, dict) else None
            if (
                source is None
                or not source.is_file()
                or binding.get("sha256") != hash_file(source)
            ):
                raise SchemaValidationError(
                    f"R1 yaw-repair entry binding differs: {candidate_id}:{key}"
                )
        raw_path = REPOSITORY / str(entry["raw_rollout"]["path"])
        evidence_root = raw_path.parents[2]
        batch_manifest = REPOSITORY / str(entry["batch_manifest"]["path"])
        batch = json.loads(batch_manifest.read_text(encoding="utf-8"))
        if (
            int(batch.get("candidate_count", -1)) > MAXIMUM_CANDIDATES_PER_SCENE
            or candidate_id not in {str(job.get("name")) for job in batch["jobs"]}
        ):
            raise SchemaValidationError(
                f"R1 yaw-repair batch membership differs: {candidate_id}"
            )
        current = _entry_for_candidate(
            candidate_id,
            evidence_root=evidence_root,
            batch_id=str(entry["evidence_batch_id"]),
            batch_manifest=batch_manifest,
            repair_binding=repairs[candidate_id],
        )
        if current != entry:
            raise SchemaValidationError(
                f"R1 yaw-repair entry bytes differ: {candidate_id}"
            )
        validated.append(current)
    if tuple(sorted(entry["candidate_id"] for entry in validated)) != _expected_candidate_ids():
        raise SchemaValidationError("R1 yaw-repair result set differs")
    return tuple(validated)


def main() -> int:
    args = _parser().parse_args()
    if not 1 <= args.maximum_parallel_batches <= 3:
        raise ValueError("R1 yaw-repair parallel batch count must be in [1, 3]")
    output_root = Path(args.output_root).resolve()
    protocol = load_config(V8_PROTOCOL)
    validate_order9_r1_nominal_v8_repair_evidence(REPOSITORY, protocol=protocol)
    repair_ledger = json.loads(V8_REPAIR_LEDGER.read_text(encoding="utf-8"))
    repair_bindings = {
        entry["candidate_id"]: entry for entry in repair_ledger["entries"]
    }
    source = _source_payload()
    chunks = _job_chunks(source)
    queued: list[tuple[str, Path, Path, tuple[str, ...]]] = []
    batch_records: list[tuple[str, Path, Path, tuple[str, ...]]] = []
    for chunk_index, jobs in enumerate(chunks):
        batch_id = f"chunk_{chunk_index:02d}"
        manifest = _materialize_chunk_manifest(
            source,
            chunk_index=chunk_index,
            jobs=jobs,
            destination=output_root / "manifests" / f"{batch_id}.json",
        )
        evidence_root = output_root / "batches" / batch_id
        candidate_ids = tuple(str(job["name"]) for job in jobs)
        batch_records.append((batch_id, manifest, evidence_root, candidate_ids))
        try:
            for candidate_id in candidate_ids:
                _validate_candidate_output(
                    candidate_id,
                    output_root=evidence_root,
                    wrapper_sha256=hash_file(WRAPPER),
                    runtime_sha256=hash_file(RUNTIME),
                )
        except (OSError, RuntimeError, ValueError, SchemaValidationError):
            queued.append((batch_id, manifest, evidence_root, candidate_ids))

    active: dict[str, tuple[subprocess.Popen[str], Any, Path]] = {}
    failures = []
    started = time.perf_counter()
    while queued or active:
        while queued and len(active) < args.maximum_parallel_batches:
            batch_id, manifest, evidence_root, candidate_ids = queued.pop(0)
            evidence_root.mkdir(parents=True, exist_ok=True)
            log_path = output_root / "logs" / f"{batch_id}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            stream = log_path.open("w", encoding="utf-8")
            command = [
                str(Path(args.isaac_python).resolve()),
                str(WRAPPER),
                "--r1-batched-jobs",
                str(manifest),
                "--r1-batched-output-root",
                str(evidence_root),
                "--no-formal-annotation",
            ]
            process = subprocess.Popen(
                command,
                cwd=REPOSITORY,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active[batch_id] = (process, stream, log_path)
            print(
                "ORDER9_R1_YAW_REPAIR_BOUNDED_STARTED="
                + json.dumps(
                    {"batch_id": batch_id, "candidate_count": len(candidate_ids)},
                    sort_keys=True,
                ),
                flush=True,
            )
        time.sleep(1.0)
        for batch_id, (process, stream, log_path) in tuple(active.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            stream.close()
            del active[batch_id]
            if return_code != 0:
                failures.append(
                    {
                        "batch_id": batch_id,
                        "return_code": return_code,
                        "log_path": _ledger_path(log_path),
                    }
                )
            print(
                "ORDER9_R1_YAW_REPAIR_BOUNDED_FINISHED="
                + json.dumps(
                    {"batch_id": batch_id, "return_code": return_code},
                    sort_keys=True,
                ),
                flush=True,
            )
    if failures:
        _atomic_json(output_root / "failed_batches.json", {"failures": failures})
        raise RuntimeError(f"R1 yaw-repair bounded execution failed: {failures}")

    entries = []
    for batch_id, manifest, evidence_root, candidate_ids in batch_records:
        for candidate_id in candidate_ids:
            entries.append(
                _entry_for_candidate(
                    candidate_id,
                    evidence_root=evidence_root,
                    batch_id=batch_id,
                    batch_manifest=manifest,
                    repair_binding=repair_bindings[candidate_id],
                )
            )
    entries.sort(key=lambda value: value["candidate_id"])
    ledger = {
        "runner_version": RUNNER_VERSION,
        "status": "passed",
        "source_bucket_id": SOURCE_BUCKET_ID,
        "candidate_count": len(entries),
        "episode_count": sum(entry["episode_count"] for entry in entries),
        "success_count": sum(entry["success_count"] for entry in entries),
        "safety_failure_count": sum(
            entry["safety_failure_count"] for entry in entries
        ),
        "fallback_count": sum(entry["fallback_count"] for entry in entries),
        "maximum_candidates_per_scene": MAXIMUM_CANDIDATES_PER_SCENE,
        "environment_spacing_m": BATCH_ENV_SPACING_M,
        "pi_l_actor_command_applied": False,
        "qpid_qp_applied": True,
        "local_servo_applied": True,
        "elapsed_s": time.perf_counter() - started,
        "bindings": {
            "source_jobs": _binding(SOURCE_JOBS),
            "repair_evidence_ledger": _binding(V8_REPAIR_LEDGER),
            "wrapper_implementation": _binding(WRAPPER),
            "runtime_implementation": _binding(RUNTIME),
            "runner_implementation": _binding(Path(__file__).resolve()),
        },
        "entries": entries,
    }
    result_path = _atomic_json(output_root / "result_ledger.json", ledger)
    validate_result_ledger(result_path)
    print(
        "ORDER9_R1_YAW_REPAIR_BOUNDED_COMPLETE="
        + json.dumps(
            {
                "candidate_count": len(entries),
                "episode_count": ledger["episode_count"],
                "success_count": ledger["success_count"],
                "result_ledger": str(result_path),
                "result_ledger_sha256": hash_file(result_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
