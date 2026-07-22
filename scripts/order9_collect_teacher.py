#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.order8 import load_order8_natural_contact_config
from amsrr.training.order9_c0_curriculum import (
    Order9C0TeacherCondition,
    build_order9_c0_teacher_conditions,
)
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_runtime_load import Order9RuntimeLoadMonitor
from amsrr.training.order9_teacher_collection import (
    build_order9_teacher_dataset,
    load_order9_teacher_episode_manifest,
    validate_order9_teacher_pi_l_representability,
)
from amsrr.utils.hashing import hash_file

@dataclass(frozen=True)
class _CollectionJob:
    index: int
    split: str
    episode_id: str
    task_id: str
    condition: Order9C0TeacherCondition
    condition_config_path: Path
    episode_dir: Path
    report_path: Path
    log_path: Path


@dataclass(frozen=True)
class _CollectionOutcome:
    job: _CollectionJob
    returncode: int
    wall_time_s: float
    manifest_path: Path | None
    generated_usd_path: Path | None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run resumable bounded-diversity real-Isaac C0 Order-8 teacher "
            "collection and assemble its verified dataset."
        )
    )
    parser.add_argument(
        "--curriculum-config",
        default="configs/training/order9_learning_curriculum.yaml",
    )
    parser.add_argument("--episode-count", type=int)
    parser.add_argument("--validation-count", type=int)
    parser.add_argument("--held-out-count", type=int)
    parser.add_argument("--parallel-processes", type=int)
    parser.add_argument("--seed-start", type=int, default=9009)
    parser.add_argument(
        "--output-root", default="artifacts/p4_full/order9/c0_teacher"
    )
    parser.add_argument("--config", default="configs/training/order8_natural_contact.yaml")
    parser.add_argument("--backend-config", default="configs/env/isaac_lab.yaml")
    parser.add_argument("--robot-config", default="configs/robot/robot_model.yaml")
    parser.add_argument("--low-level-stride", type=int)
    parser.add_argument("--high-level-stride", type=int)
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--force-recollect", action="store_true")
    parser.add_argument(
        "--stream-child-output",
        action="store_true",
        help="Stream verbose Isaac telemetry instead of writing per-episode logs.",
    )
    args = parser.parse_args()

    learning = load_order9_learning_config(args.curriculum_config)
    learning.validate()
    profile = learning.teacher_collection
    episode_count = (
        profile.episode_count if args.episode_count is None else args.episode_count
    )
    validation_count = (
        profile.validation_episode_count
        if args.validation_count is None
        else args.validation_count
    )
    held_out_count = (
        profile.held_out_episode_count
        if args.held_out_count is None
        else args.held_out_count
    )
    parallel_processes = (
        profile.parallel_process_count
        if args.parallel_processes is None
        else args.parallel_processes
    )
    low_level_stride = (
        profile.low_level_stride
        if args.low_level_stride is None
        else args.low_level_stride
    )
    high_level_stride = (
        profile.high_level_stride
        if args.high_level_stride is None
        else args.high_level_stride
    )
    if episode_count < 5:
        parser.error("--episode-count must cover nominal plus four boundary conditions")
    if min(validation_count, held_out_count) < 1:
        parser.error("validation and held-out counts must be positive")
    if validation_count + held_out_count >= episode_count:
        parser.error("validation + held-out counts must leave training episodes")
    if min(low_level_stride, high_level_stride, parallel_processes) < 1:
        parser.error("teacher strides and parallel process count must be positive")
    if args.seed_start < 0:
        parser.error("--seed-start must be non-negative")

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        collection_lock = _acquire_collection_lock(root)
    except RuntimeError as exc:
        parser.error(str(exc))
    episodes_root = root / "episodes"
    reports_root = root / "reports"
    logs_root = root / "logs"
    conditions_root = root / "conditions"
    for directory in (episodes_root, reports_root, logs_root, conditions_root):
        directory.mkdir(parents=True, exist_ok=True)

    base_config = load_order8_natural_contact_config(args.config)
    conditions = build_order9_c0_teacher_conditions(
        base_config,
        episode_count=episode_count,
        seed_start=args.seed_start,
        randomization=learning.randomization,
    )
    _write_conditions_manifest(conditions_root / "manifest.json", conditions)
    jobs = [
        _job(
            condition,
            count=episode_count,
            validation_count=validation_count,
            held_out_count=held_out_count,
            episodes_root=episodes_root,
            reports_root=reports_root,
            logs_root=logs_root,
            conditions_root=conditions_root,
        )
        for condition in conditions
    ]
    physical_model = build_physical_model_from_config(args.robot_config)
    successful: list[Path] = []
    representability_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    pending: list[_CollectionJob] = []

    for job in jobs:
        manifest_path = job.episode_dir / "episode_manifest.json"
        if not args.force_recollect and _manifest_is_compatible(
            manifest_path,
            job=job,
            low_level_stride=low_level_stride,
            high_level_stride=high_level_stride,
        ):
            representability = validate_order9_teacher_pi_l_representability(
                manifest_path, physical_model=physical_model
            )
            successful.append(manifest_path)
            representability_rows.append(
                {"episode_id": job.episode_id, **representability}
            )
            print(
                f"reuse {job.index + 1}/{episode_count}: {job.episode_id} "
                f"condition={job.condition.sample_kind} pi_l_max="
                f"{representability['maximum_absolute_unclipped_action']:.4f}",
                flush=True,
            )
        else:
            pending.append(job)

    collection_started = time.monotonic()
    load_monitor = Order9RuntimeLoadMonitor(
        sample_interval_s=learning.production_runtime.runtime_load_sample_interval_s,
        device=learning.production_runtime.device,
    )
    load_monitor.start()
    outcomes: list[_CollectionOutcome] = []
    prepared_generated_usd_path: Path | None = None
    try:
        # Convert and hash-audit the current robot asset once before concurrent
        # readers enter Isaac.  Do this on every resumed invocation as well:
        # compatible episode manifests prove their own bytes, but they cannot
        # prove that the mutable generated-asset cache was not later damaged.
        if pending:
            first = pending.pop(0)
            outcome = _run_job(
                first,
                args=args,
                low_level_stride=low_level_stride,
                high_level_stride=high_level_stride,
                reuse_generated_asset=False,
                generated_usd_path=None,
            )
            outcomes.append(outcome)
            _print_outcome(outcome, episode_count)
            if outcome.returncode == 0:
                prepared_generated_usd_path = outcome.generated_usd_path
            elif args.stop_on_failure:
                pending.clear()

        if pending and prepared_generated_usd_path is None:
            # It is unsafe to start concurrent readers without a successfully
            # generated and hash-audited immutable USD path.  If fail-fast was
            # not requested, retain best-effort semantics by running each
            # remaining conversion serially.
            for job in pending:
                outcome = _run_job(
                    job,
                    args=args,
                    low_level_stride=low_level_stride,
                    high_level_stride=high_level_stride,
                    reuse_generated_asset=False,
                    generated_usd_path=None,
                )
                outcomes.append(outcome)
                _print_outcome(outcome, episode_count)
                if outcome.returncode == 0:
                    prepared_generated_usd_path = outcome.generated_usd_path
                    break
            pending = [
                job
                for job in pending
                if all(outcome.job != job for outcome in outcomes)
            ]

        if pending and prepared_generated_usd_path is not None:
            with ThreadPoolExecutor(max_workers=parallel_processes) as executor:
                futures = {
                    executor.submit(
                        _run_job,
                        job,
                        args=args,
                        low_level_stride=low_level_stride,
                        high_level_stride=high_level_stride,
                        reuse_generated_asset=True,
                        generated_usd_path=prepared_generated_usd_path,
                    ): job
                    for job in pending
                }
                stop_requested = False
                for future in as_completed(futures):
                    try:
                        outcome = future.result()
                    except CancelledError:
                        continue
                    outcomes.append(outcome)
                    _print_outcome(outcome, episode_count)
                    if (
                        outcome.returncode != 0
                        and args.stop_on_failure
                        and not stop_requested
                    ):
                        stop_requested = True
                        for other in futures:
                            other.cancel()
    finally:
        runtime_load = load_monitor.stop()

    for outcome in sorted(outcomes, key=lambda value: value.job.index):
        job = outcome.job
        manifest_path = outcome.manifest_path
        if (
            outcome.returncode == 0
            and manifest_path is not None
            and _manifest_is_compatible(
                manifest_path,
                job=job,
                low_level_stride=low_level_stride,
                high_level_stride=high_level_stride,
            )
        ):
            representability = validate_order9_teacher_pi_l_representability(
                manifest_path, physical_model=physical_model
            )
            successful.append(manifest_path)
            representability_rows.append(
                {"episode_id": job.episode_id, **representability}
            )
            continue
        failures.append(
            {
                "episode_id": job.episode_id,
                "seed": job.condition.random_seed,
                "split": job.split,
                "condition_id": job.condition.condition_id,
                "sample_kind": job.condition.sample_kind,
                "returncode": outcome.returncode,
                "wall_time_s": outcome.wall_time_s,
                "report_path": str(job.report_path),
                "log_path": str(job.log_path),
            }
        )

    successful = sorted(set(successful), key=lambda path: str(path))
    wall_time_s = time.monotonic() - collection_started
    total_records = sum(
        int(float(row["record_count"])) for row in representability_rows
    )
    summary = {
        "collection_profile": profile.to_dict(),
        "requested_episode_count": episode_count,
        "successful_episode_count": len(successful),
        "failure_count": len(failures),
        "failures": failures,
        "parallel_process_count": parallel_processes,
        "low_level_stride": low_level_stride,
        "high_level_stride": high_level_stride,
        "condition_manifest_path": str(conditions_root / "manifest.json"),
        "conditions": [condition.to_dict() for condition in conditions],
        "pi_l_representability": sorted(
            representability_rows, key=lambda row: str(row["episode_id"])
        ),
        "pi_l_maximum_absolute_unclipped_action": max(
            (
                float(row["maximum_absolute_unclipped_action"])
                for row in representability_rows
            ),
            default=0.0,
        ),
        "low_level_record_count": total_records,
        "low_level_records_per_wall_s": (
            float(total_records) / wall_time_s if wall_time_s > 0.0 else 0.0
        ),
        "wall_time_s": wall_time_s,
        "runtime_load": runtime_load,
        "prepared_generated_usd_path": (
            None
            if prepared_generated_usd_path is None
            else str(prepared_generated_usd_path)
        ),
        "prepared_generated_usd_sha256": (
            None
            if prepared_generated_usd_path is None
            else hash_file(prepared_generated_usd_path)
        ),
    }
    summary_path = root / "collection_summary.json"
    _atomic_write_text(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if len(successful) != episode_count:
        print(f"summary: {summary_path}", flush=True)
        return 1
    dataset_dir = root / "dataset"
    manifest = build_order9_teacher_dataset(successful, dataset_dir)
    print(f"dataset: {dataset_dir / 'manifest.json'}")
    print(f"dataset_id: {manifest.dataset_id}")
    print(f"summary: {summary_path}")
    return 0


def _acquire_collection_lock(root: Path):
    """Hold an exclusive lease before any episode artifact can be mutated."""

    lock_path = root / ".collection.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(
            f"another Order9 teacher collector owns {lock_path}"
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(
        json.dumps(
            {
                "pid": os.getpid(),
                "started_unix_s": time.time(),
            },
            sort_keys=True,
        )
        + "\n"
    )
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def _job(
    condition: Order9C0TeacherCondition,
    *,
    count: int,
    validation_count: int,
    held_out_count: int,
    episodes_root: Path,
    reports_root: Path,
    logs_root: Path,
    conditions_root: Path,
) -> _CollectionJob:
    index = condition.episode_index
    seed = condition.random_seed
    episode_id = f"order9-c0-episode-{seed:06d}"
    config_path = conditions_root / f"{episode_id}.json"
    _atomic_write_text(
        config_path,
        json.dumps(
            {"order8": condition.order8_config.to_dict()},
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return _CollectionJob(
        index=index,
        split=_split(
            index,
            count=count,
            validation_count=validation_count,
            held_out_count=held_out_count,
        ),
        episode_id=episode_id,
        task_id=f"order9-c0-task-{seed:06d}",
        condition=condition,
        condition_config_path=config_path,
        episode_dir=episodes_root / episode_id,
        report_path=reports_root / f"{episode_id}.json",
        log_path=logs_root / f"{episode_id}.log",
    )


def _run_job(
    job: _CollectionJob,
    *,
    args: argparse.Namespace,
    low_level_stride: int,
    high_level_stride: int,
    reuse_generated_asset: bool,
    generated_usd_path: Path | None,
) -> _CollectionOutcome:
    job.episode_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/order8_natural_contact.py"),
        "--real",
        "--config",
        str(job.condition_config_path),
        "--backend-config",
        args.backend_config,
        "--seed",
        str(job.condition.random_seed),
        "--report-path",
        str(job.report_path),
        "--order9-teacher-output",
        str(job.episode_dir),
        "--order9-teacher-episode-id",
        job.episode_id,
        "--order9-teacher-task-id",
        job.task_id,
        "--order9-teacher-split",
        job.split,
        "--order9-teacher-low-level-stride",
        str(low_level_stride),
        "--order9-teacher-high-level-stride",
        str(high_level_stride),
    ]
    if reuse_generated_asset:
        if generated_usd_path is None:
            raise ValueError("reused Order9 teacher jobs require a generated USD path")
        command.extend(
            [
                "--reuse-generated-asset",
                "--generated-usd-path",
                str(generated_usd_path.resolve()),
            ]
        )
    started = time.monotonic()
    if args.stream_child_output:
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    else:
        with job.log_path.open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                check=False,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
    elapsed = time.monotonic() - started
    manifest_path = job.episode_dir / "episode_manifest.json"
    validated_generated_usd_path = _validated_generated_usd_path(job.report_path)
    returncode = int(completed.returncode)
    if returncode == 0 and validated_generated_usd_path is None:
        returncode = 1
    if (
        returncode == 0
        and generated_usd_path is not None
        and validated_generated_usd_path != generated_usd_path.resolve()
    ):
        returncode = 1
    return _CollectionOutcome(
        job=job,
        returncode=returncode,
        wall_time_s=elapsed,
        manifest_path=manifest_path if manifest_path.is_file() else None,
        generated_usd_path=validated_generated_usd_path,
    )


def _validated_generated_usd_path(report_path: Path) -> Path | None:
    """Return the report-bound USD only when its current bytes still match."""

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        report = payload["report"]
        usd_path = Path(report["usd_path"]).resolve()
        expected_sha256 = str(report["generated_usd_sha256"])
        if len(expected_sha256) != 64 or not usd_path.is_file():
            return None
        if hash_file(usd_path) != expected_sha256:
            return None
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return usd_path


def _manifest_is_compatible(
    path: Path,
    *,
    job: _CollectionJob,
    low_level_stride: int,
    high_level_stride: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        _, manifest = load_order9_teacher_episode_manifest(path)
    except (OSError, ValueError, SchemaValidationError):
        return False
    metadata = manifest.metadata
    return bool(
        manifest.success
        and manifest.episode_id == job.episode_id
        and manifest.task_spec.task_id == job.task_id
        and manifest.split.value == job.split
        and manifest.random_seed == job.condition.random_seed
        and manifest.config_hash == job.condition.order8_config.stable_hash()
        and metadata.get("c0_collection_profile_version")
        == job.condition.profile_version
        and metadata.get("c0_condition_id") == job.condition.condition_id
        and int(metadata.get("teacher_low_level_stride", -1))
        == low_level_stride
        and int(metadata.get("teacher_high_level_stride", -1))
        == high_level_stride
    )


def _print_outcome(outcome: _CollectionOutcome, count: int) -> None:
    status = "complete" if outcome.returncode == 0 else "failed"
    print(
        f"{status} {outcome.job.index + 1}/{count}: {outcome.job.episode_id} "
        f"condition={outcome.job.condition.sample_kind} "
        f"wall_s={outcome.wall_time_s:.1f}",
        flush=True,
    )


def _write_conditions_manifest(
    path: Path,
    conditions: list[Order9C0TeacherCondition],
) -> None:
    _atomic_write_text(
        path,
        json.dumps(
            {
                "profile_version": conditions[0].profile_version,
                "condition_count": len(conditions),
                "conditions": [condition.to_dict() for condition in conditions],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _split(
    index: int,
    *,
    count: int,
    validation_count: int,
    held_out_count: int,
) -> str:
    if index >= count - held_out_count:
        return "held_out"
    if index >= count - held_out_count - validation_count:
        return "validation"
    return "train"


if __name__ == "__main__":
    raise SystemExit(main())
