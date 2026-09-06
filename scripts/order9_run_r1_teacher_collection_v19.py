#!/usr/bin/env python3
from __future__ import annotations

"""Run or resume the authorized 140-episode R1 teacher collection."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.geometry.pose_math import (  # noqa: E402
    pose_from_transform,
    pose_to_xyz_rpy,
    transform_from_xyz_rpy,
)
from amsrr.schemas.common import SchemaValidationError  # noqa: E402
from amsrr.schemas.task_spec import TaskSpec  # noqa: E402
from amsrr.training.order9_r1_isaac_calibration import (  # noqa: E402
    load_order9_r1_isaac_case,
)
from amsrr.training.order9_r1_nominal_calibration import (  # noqa: E402
    order9_r1_nominal_case_command,
)
from amsrr.training.order9_r1_teacher_collection_preparation import (  # noqa: E402
    read_json_object,
    validate_binding,
    validate_order9_r1_teacher_collection_plan,
)
from amsrr.training import order9_r1_yaw_branch_repair as transfer_runtime  # noqa: E402
from amsrr.training.order9_r1_yaw_branch_repair import (  # noqa: E402
    derive_order9_r1_planar_nominal_transfer_case,
)
from amsrr.utils.hashing import hash_file  # noqa: E402

DEFAULT_ROOT = REPOSITORY / (
    "artifacts/p4_full/order9/r1_teacher/formal_collection_v19"
)
DEFAULT_ISAAC_PYTHON = Path("/home/leus/.local/share/mamba/envs/isaaclab3/bin/python")
WRAPPER = REPOSITORY / "scripts/order9_r1_batched_nominal_compression_rollout.py"
EXECUTION_REVISION = "identity_v2"


def _revision_path(output_root: Path, name: str) -> Path:
    """Keep an invalidated execution attempt immutable and start a clean revision."""

    return output_root / f"{name}_{EXECUTION_REVISION}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(DEFAULT_ROOT / "collection_plan.json"))
    parser.add_argument("--preflight", default=str(DEFAULT_ROOT / "preflight.json"))
    parser.add_argument("--launch-authorization", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--isaac-python", default=str(DEFAULT_ISAAC_PYTHON))
    parser.add_argument("--rollout-steps", type=int, default=15000)
    parser.add_argument("--parallel-processes", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    return parser


def _atomic_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
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
    source = path.resolve()
    return {
        "path": source.relative_to(REPOSITORY).as_posix(),
        "sha256": hash_file(source),
    }


def _load_authorized_inputs(args: argparse.Namespace):
    plan_path = Path(args.plan).resolve()
    preflight_path = Path(args.preflight).resolve()
    authorization_path = Path(args.launch_authorization).resolve()
    plan = read_json_object(plan_path)
    preflight = read_json_object(preflight_path)
    authorization = read_json_object(authorization_path)
    validate_order9_r1_teacher_collection_plan(plan, repository_root=REPOSITORY)
    if (
        preflight.get("status") != "ready_awaiting_explicit_launch"
        or preflight.get("ready_for_explicit_launch") is not True
        or preflight.get("collection_started") is not False
        or preflight.get("plan") != _binding(plan_path)
    ):
        raise SchemaValidationError("R1 teacher collection preflight differs")
    if (
        authorization.get("authorization_version")
        != "order9_r1_teacher_collection_launch_v19"
        or authorization.get("status") != "approved"
        or authorization.get("collection_launch_authorized") is not True
        or authorization.get("training_authorized") is not False
        or authorization.get("execution_revision") != EXECUTION_REVISION
        or authorization.get("plan") != _binding(plan_path)
        or authorization.get("preflight") != _binding(preflight_path)
        or authorization.get("runner") != _binding(Path(__file__))
        or authorization.get("wrapper") != _binding(WRAPPER)
    ):
        raise SchemaValidationError(
            "R1 teacher collection lacks an explicit hash-bound start authorization"
        )
    return plan, authorization_path


def build_target_identity_task(
    reference: TaskSpec, entry: Mapping[str, Any], *, fraction: float, candidate_id: str
) -> TaskSpec:
    """Create one identity-only target for an incremental planar scene move."""

    reference.validate()
    if not 0.0 < fraction <= 1.0:
        raise ValueError("R1 scene-transfer fraction must be in (0, 1]")
    object_goal = next(
        goal
        for goal in reference.goals
        if goal.goal_type == "object_pose" and goal.target_pose_world is not None
    )
    target_object_id = object_goal.target_entity_id
    source_object = next(
        value
        for value in reference.scene.objects
        if value.object_id == target_object_id
    )
    xyz, rpy = pose_to_xyz_rpy(source_object.pose_world)
    requested = pose_from_transform(
        transform_from_xyz_rpy(
            (
                xyz[0] + fraction * float(entry["x_offset_m"]),
                xyz[1] + fraction * float(entry["y_offset_m"]),
                xyz[2],
            ),
            (rpy[0], rpy[1], rpy[2] + fraction * float(entry["yaw_offset_rad"])),
        )
    )
    payload = reference.to_dict()
    payload["task_id"] = (
        str(entry["task_id"])
        if math.isclose(fraction, 1.0)
        else f"{entry['task_id']}:{candidate_id}"
    )
    for value in payload["scene"]["objects"]:
        if value["object_id"] == target_object_id:
            value["pose_world"] = list(requested)
    metadata = deepcopy(payload.get("metadata", {}))
    metadata.update(
        {
            "order9_rollout_bucket_id": candidate_id,
            "dataset_split": entry["split"],
            "r1_calibration_candidate_id": candidate_id,
            "r1_teacher_collection_episode_id": entry["episode_id"],
            "r1_teacher_collection_seed": int(entry["seed"]),
            "r1_teacher_collection_pose_condition_hash": entry["pose_condition_hash"],
            "r1_formal_teacher_collection_eligible": False,
        }
    )
    payload["metadata"] = metadata
    task = TaskSpec.from_dict(payload)
    task.validate()
    return task


def _local_case_task_path(case_root: Path, repository: Path) -> Path:
    root = Path(case_root).resolve()
    path = root / "task_spec.json"
    if not path.is_file() or Path(repository).resolve() != REPOSITORY:
        raise FileNotFoundError(path)
    case = read_json_object(root / "case_manifest.json")
    if case.get("task_spec", {}).get("sha256") != hash_file(path):
        raise SchemaValidationError("R1 local case task binding differs")
    return path


def _materialize(entry: Mapping[str, Any], output_root: Path) -> Path:
    candidate_id = str(entry["episode_id"])
    destination = _revision_path(output_root, "cases") / candidate_id
    if destination.is_dir():
        load_order9_r1_isaac_case(destination / "case_manifest.json", REPOSITORY)
        return destination
    source_manifest = validate_binding(
        entry["source_case_manifest"],
        repository_root=REPOSITORY,
        label=f"{candidate_id}_source_case",
    )
    source_root = source_manifest.parent
    reference_task = TaskSpec.from_json(
        (source_root / "task_spec.json").read_text(encoding="utf-8")
    )
    step_count = max(
        2, math.ceil(abs(float(entry["yaw_offset_rad"])) / math.radians(10))
    )
    previous = source_root
    original_resolver = transfer_runtime._source_bucket_task_path
    transfer_runtime._source_bucket_task_path = _local_case_task_path
    try:
        for step in range(1, step_count + 1):
            final = step == step_count
            step_candidate_id = (
                candidate_id if final else f"{candidate_id}-transfer-{step:02d}"
            )
            target = build_target_identity_task(
                reference_task,
                entry,
                fraction=step / step_count,
                candidate_id=step_candidate_id,
            )
            step_root = (
                destination
                if final
                else output_root
                / f"transfer_intermediates_{EXECUTION_REVISION}"
                / candidate_id
                / step_candidate_id
            )
            if not step_root.is_dir():
                derive_order9_r1_planar_nominal_transfer_case(
                    reference_case_root=previous,
                    target_task_spec=target,
                    target_candidate_id=step_candidate_id,
                    target_level_id="r1_teacher_collection_v19",
                    target_seed=int(entry["seed"]),
                    destination_case_root=step_root,
                    repository_root=REPOSITORY,
                )
            previous = step_root
    finally:
        transfer_runtime._source_bucket_task_path = original_resolver
    load_order9_r1_isaac_case(destination / "case_manifest.json", REPOSITORY)
    return destination


def _replace(argv: list[str], option: str, value: str) -> None:
    indices = [index for index, item in enumerate(argv) if item == option]
    if len(indices) != 1 or indices[0] + 1 >= len(argv):
        raise SchemaValidationError(f"R1 collection command lacks one {option}")
    argv[indices[0] + 1] = value


def _prepare_job(
    entry: Mapping[str, Any], *, output_root: Path, authorization_path: Path, args
) -> dict[str, Any]:
    case_root = _materialize(entry, output_root)
    materialized = load_order9_r1_isaac_case(
        case_root / "case_manifest.json", REPOSITORY
    )
    task = TaskSpec.from_json(
        (case_root / "task_spec.json").read_text(encoding="utf-8")
    )
    if (
        materialized.manifest.candidate_id != entry["episode_id"]
        or task.metadata.get("order9_rollout_bucket_id")
        != materialized.manifest.candidate_id
        or task.metadata.get("r1_calibration_candidate_id")
        != materialized.manifest.candidate_id
    ):
        raise SchemaValidationError(
            "R1 materialized task, rollout bucket, and nominal candidate identities differ"
        )
    command, _log = order9_r1_nominal_case_command(
        materialized,
        repository_root=REPOSITORY,
        python_executable=args.isaac_python,
        rollout_steps=args.rollout_steps,
    )
    script_indices = [
        index
        for index, value in enumerate(command)
        if Path(value).name == "order9_vectorized_isaac_rollout.py"
    ]
    if len(script_indices) != 1:
        raise SchemaValidationError("R1 collection protected command differs")
    argv = list(command[script_indices[0] + 1 :])
    _replace(argv, "--num-envs", "1")
    _replace(argv, "--evaluation-episode-count", "1")
    episode_root = _revision_path(output_root, "episodes")
    manifest_path = (
        _revision_path(output_root, "jobs") / f"{entry['episode_id']}.json"
    )
    payload = {
        "version": "order9_r1_teacher_collection_job_v19",
        "execution_revision": EXECUTION_REVISION,
        "r1_formal_teacher_collection": True,
        "r1_teacher_collection_launch_authorization": _binding(authorization_path),
        "r1_scene_support_from_task": True,
        "r1_nominal_additional_compression_sweep_mm": "5.0",
        "r1_release_height_offset_m": 0.0,
        "jobs": [
            {
                "name": materialized.manifest.candidate_id,
                "argv": argv,
                "log_path": str(case_root / "isaac" / "evaluation.log"),
            }
        ],
    }
    if manifest_path.is_file() and read_json_object(manifest_path) != payload:
        raise SchemaValidationError("R1 cached collection job differs")
    if not manifest_path.is_file():
        _atomic_json(manifest_path, payload)
    return {
        "entry": entry,
        "case_root": case_root,
        "job_manifest": manifest_path,
        "episode_root": episode_root,
        "command": [
            str(Path(args.isaac_python).resolve()),
            str(WRAPPER),
            "--r1-batched-jobs",
            str(manifest_path),
            "--r1-batched-output-root",
            str(episode_root),
            "--no-formal-annotation",
        ],
    }


def _episode_paths(job: Mapping[str, Any]) -> tuple[Path, Path]:
    candidate = str(job["entry"]["episode_id"])
    root = Path(job["episode_root"]) / candidate / "isaac"
    return root / "evaluation_rollout.pt", root / "evaluation_episodes.jsonl"


def _validate_episode(job: Mapping[str, Any]) -> dict[str, Any]:
    raw_path, episode_path = _episode_paths(job)
    records = [
        json.loads(line)
        for line in episode_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 1 or not raw_path.is_file():
        raise SchemaValidationError("R1 collection episode output is incomplete")
    record = records[0]
    metadata = record.get("metadata", {})
    metrics = record.get("metrics", {})
    entry = job["entry"]
    task = TaskSpec.from_json(
        (Path(job["case_root"]) / "task_spec.json").read_text(encoding="utf-8")
    )
    if (
        record.get("task_success") is not True
        or record.get("safety_failure") is not False
        or record.get("no_fallback_success") is not True
        or int(record.get("fallback_decision_count", -1)) != 0
        or metadata.get("r1_scene_support_from_task") is not True
        or metadata.get("formal_teacher_collection_authorized") is not True
        or metadata.get("r1_formal_teacher_collection_candidate") is not True
        or any(
            float(metrics.get(name, 1.0)) != 0.0
            for name in (
                "hard_collision",
                "object_dropped",
                "qp_infeasible_terminal",
                "timeout",
            )
        )
        or Path(record.get("source_artifact_path", "")).resolve() != raw_path.resolve()
        or record.get("source_artifact_sha256") != hash_file(raw_path)
        or record.get("split") != "validation"
        or task.task_id != entry["task_id"]
        or task.metadata.get("dataset_split") != entry["split"]
        or task.metadata.get("r1_teacher_collection_pose_condition_hash")
        != entry["pose_condition_hash"]
    ):
        raise SchemaValidationError("R1 collection episode failed its admission gate")
    return {
        "record_version": "order9_r1_teacher_collection_episode_v19",
        "status": "accepted",
        "training_eligible": True,
        "episode_id": entry["episode_id"],
        "task_id": entry["task_id"],
        "dataset_split": entry["split"],
        "execution_split": "validation",
        "execution_split_alias_contract": (
            "protected_c3_formal_runner_validation_alias_v1"
        ),
        "module_count": entry["module_count"],
        "seed": entry["seed"],
        "pose_condition_hash": entry["pose_condition_hash"],
        "case_manifest": _binding(Path(job["case_root"]) / "case_manifest.json"),
        "raw_rollout": _binding(raw_path),
        "evaluation_episode": _binding(episode_path),
        "task_success": True,
        "safety_failure": False,
        "fallback_decision_count": 0,
        "pi_l_actor_command_applied": False,
        "qpid_qp_applied": True,
        "local_servo_applied": True,
    }


def _run_job(job: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    result_path = (
        _revision_path(output_root, "records")
        / f"{job['entry']['episode_id']}.json"
    )
    if result_path.is_file():
        record = read_json_object(result_path)
        if record.get("status") != "accepted":
            raise SchemaValidationError("R1 cached collection result is not accepted")
        for label in ("case_manifest", "raw_rollout", "evaluation_episode"):
            validate_binding(record[label], repository_root=REPOSITORY, label=label)
        return record
    raw_path, episode_path = _episode_paths(job)
    if not (raw_path.is_file() and episode_path.is_file()):
        log_path = (
            _revision_path(output_root, "logs")
            / f"{job['entry']['episode_id']}.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                job["command"],
                cwd=REPOSITORY,
                env={**os.environ, "PYTHONHASHSEED": "0"},
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"R1 teacher episode {job['entry']['episode_id']} failed; see {log_path}"
            )
    return read_json_object(_atomic_json(result_path, _validate_episode(job)))


def main() -> int:
    args = _parser().parse_args()
    if (
        args.parallel_processes < 1
        or args.offset < 0
        or (args.limit is not None and args.limit < 1)
    ):
        raise ValueError("R1 collection selection arguments are invalid")
    plan, authorization_path = _load_authorized_inputs(args)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    lock_stream = (output_root / ".collection.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another R1 teacher collection process is active") from error
    selected = plan["entries"][args.offset :]
    if args.limit is not None:
        selected = selected[: args.limit]
    jobs = [
        _prepare_job(
            entry,
            output_root=output_root,
            authorization_path=authorization_path,
            args=args,
        )
        for entry in selected
    ]
    results = []
    executor = ThreadPoolExecutor(max_workers=args.parallel_processes)
    pending: dict[Any, Mapping[str, Any]] = {}
    job_iterator = iter(jobs)
    try:
        for job in job_iterator:
            pending[executor.submit(_run_job, job, output_root)] = job
            if len(pending) == args.parallel_processes:
                break
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                pending.pop(future)
                result = future.result()
                results.append(result)
                print(
                    f"ORDER9_R1_TEACHER_COLLECTION={len(results)}/{len(jobs)} "
                    f"{result['episode_id']} accepted",
                    flush=True,
                )
                next_job = next(job_iterator, None)
                if next_job is not None:
                    pending[executor.submit(_run_job, next_job, output_root)] = next_job
    except BaseException:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    all_records = sorted(_revision_path(output_root, "records").glob("*.json"))
    summary = {
        "summary_version": "order9_r1_teacher_collection_summary_v19",
        "execution_revision": EXECUTION_REVISION,
        "status": "complete" if len(all_records) == 140 else "in_progress",
        "planned_episode_count": 140,
        "accepted_episode_count": len(all_records),
        "collection_records": [_binding(path) for path in all_records],
        "training_authorized": False,
    }
    _atomic_json(
        output_root / f"collection_summary_{EXECUTION_REVISION}.json", summary
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
