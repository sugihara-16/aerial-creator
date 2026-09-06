#!/usr/bin/env python3
from __future__ import annotations

"""Reselect the R1 pose range by moving the complete task scene together."""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import math
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

import torch

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.schemas.common import SchemaValidationError  # noqa: E402
from amsrr.geometry.pose_math import (
    compose_pose,
    inverse_pose,
    transform_from_pose,
)  # noqa: E402
from amsrr.schemas.task_spec import TaskSpec  # noqa: E402
from amsrr.training.order9_r1_isaac_calibration import (  # noqa: E402
    load_order9_r1_isaac_case,
)
from amsrr.training.order9_r1_nominal_calibration import (  # noqa: E402
    order9_r1_nominal_case_command,
)
from amsrr.training.order9_r1_yaw_branch_repair import (  # noqa: E402
    ORDER9_R1_PLANAR_SCENE_TRANSFER_VERSION,
    audit_order9_r1_scene_transfer_case,
    derive_order9_r1_planar_nominal_transfer_case,
)
from amsrr.utils.hashing import hash_file  # noqa: E402
from scripts import order9_run_r1_range_selection_v11 as runtime_v11  # noqa: E402

RUNNER_VERSION = "order9_r1_scene_invariant_range_selection_v15"
PROTOCOL = REPOSITORY / (
    "configs/training/order9_r1_scene_invariant_range_selection_protocol_v15.json"
)
APPROVAL = REPOSITORY / (
    "for_codex/R1_SCENE_INVARIANT_RANGE_SELECTION_PROTOCOL_V15_APPROVAL.json"
)
OUTPUT_ROOT = REPOSITORY / (
    "artifacts/p4_full/order9/r1_teacher/range_selection_scene_transfer_v15"
)
SOURCE_PREPARED = REPOSITORY / (
    "artifacts/p4_full/order9/r1_teacher/range_selection_v13/prepared/"
    "r1_l2_20mm_10deg"
)
SOURCE_ISAAC = REPOSITORY / (
    "artifacts/p4_full/order9/r1_teacher/range_selection_v13/isaac/" "r1_l2_20mm_10deg"
)
LEVELS = (
    "r1_l2_20mm_10deg",
    "r1_l3_30mm_15deg",
    "r1_l4_40mm_20deg",
)
SOURCE_LEVEL = LEVELS[0]
ISAAC_PYTHON = "/home/leus/.local/share/mamba/envs/isaaclab3/bin/python"
WRAPPER = REPOSITORY / "scripts/order9_r1_batched_nominal_compression_rollout.py"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--isaac-python", default=ISAAC_PYTHON)
    parser.add_argument("--proof-workers", type=int, default=8)
    parser.add_argument("--maximum-parallel-isaac", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=15000)
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SchemaValidationError(f"R1 v15 JSON object expected: {path}")
    return value


def _atomic_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_text = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_text)
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


def _portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY))
    except ValueError:
        return str(path.resolve())


def _binding(path: Path) -> dict[str, str]:
    return {"path": _portable(path), "sha256": hash_file(path)}


def _validate_binding(value: object, *, label: str) -> Path:
    if not isinstance(value, dict):
        raise SchemaValidationError(f"R1 v15 {label} binding is missing")
    path = REPOSITORY / str(value.get("path", ""))
    if not path.is_file() or hash_file(path) != value.get("sha256"):
        raise SchemaValidationError(f"R1 v15 {label} binding changed")
    return path


def _load_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = _read_json(PROTOCOL)
    approval = _read_json(APPROVAL)
    if (
        protocol.get("protocol_version")
        != "order9_r1_scene_invariant_range_selection_protocol_v15"
        or protocol.get("status") != "approved"
        or tuple(protocol.get("level_ids", ())) != LEVELS
        or protocol.get("selection_split") != "train"
        or int(protocol.get("selection_bucket_count", -1)) != 22
        or int(protocol.get("candidate_count_per_level", -1)) != 682
        or int(protocol.get("source_replay_count_per_candidate", -1)) != 2
        or protocol.get("complete_scene_transform_required") is not True
        or protocol.get("support_transformed_with_scene") is not True
        or protocol.get("joint_targets_exactly_preserved") is not True
        or protocol.get("contact_assignments_exactly_preserved") is not True
        or protocol.get("all_eight_phases_required") is not True
        or int(protocol.get("representative_isaac_module_count", -1)) != 7
        or protocol.get("representative_isaac_level_id") != LEVELS[-1]
        or protocol.get("pi_l_actor_command_applied") is not False
        or protocol.get("qpid_qp_enabled_for_isaac") is not True
        or protocol.get("local_servo_enabled_for_isaac") is not True
        or protocol.get("formal_teacher_collection_authorized") is not False
    ):
        raise SchemaValidationError("R1 v15 range-selection protocol differs")
    if (
        approval.get("record_version")
        != "order9_r1_scene_invariant_range_selection_approval_v15"
        or approval.get("decision") != "approved"
        or approval.get("approved_by") != "repository_user"
        or approval.get("approved_protocol", {}).get("path") != _portable(PROTOCOL)
        or approval.get("approved_protocol", {}).get("sha256") != hash_file(PROTOCOL)
        or approval.get("learning_authorized") is not False
        or approval.get("teacher_collection_authorized") is not False
    ):
        raise SchemaValidationError("R1 v15 approval differs")
    for label in (
        "scene_transfer_implementation",
        "runner_implementation",
        "isaac_wrapper_implementation",
        "protected_c3_checkpoint",
        "protected_c3_rollout",
        "protected_c3_curriculum",
        "protected_c3_release_ledger",
    ):
        _validate_binding(protocol.get(label), label=label)
    return protocol, approval


def _cases(level_id: str) -> list[Any]:
    cases = list(runtime_v11._corrected_cases(level_id, "train"))
    if (
        len(cases) != 682
        or len({case.candidate_id for case in cases}) != 682
        or len({case.source_bucket.bucket_id for case in cases}) != 22
    ):
        raise SchemaValidationError("R1 v15 train case population differs")
    return sorted(cases, key=lambda case: case.candidate_id)


def _match_source_cases(
    source_cases: list[Any], target_cases: list[Any]
) -> dict[str, Any]:
    """Choose the nearest successful pose from the same morphology.

    Lattice indices correspond across levels, but the four interior points are
    independently sampled at every level.  Matching interior points by index
    would therefore create an unrelated, unnecessarily large transform.
    """

    by_bucket: dict[str, list[tuple[Any, tuple[float, ...]]]] = {}
    for source in source_cases:
        task = TaskSpec.from_json(
            (SOURCE_PREPARED / source.candidate_id / "task_spec.json").read_text(
                encoding="utf-8"
            )
        )
        pose = tuple(float(value) for value in task.scene.objects[0].pose_world)
        by_bucket.setdefault(source.source_bucket.bucket_id, []).append((source, pose))
    result = {}
    for target in target_cases:
        target_pose = tuple(
            float(value) for value in target.task_spec.scene.objects[0].pose_world
        )
        ranked = []
        for source, source_pose in by_bucket.get(target.source_bucket.bucket_id, ()):
            delta = compose_pose(target_pose, inverse_pose(source_pose))
            rotation = transform_from_pose(delta).rotation
            yaw = abs(math.atan2(rotation[1][0], rotation[0][0]))
            distance = math.dist(target_pose[:2], source_pose[:2])
            score = max(
                yaw / math.radians(10.0),
                distance / (math.sqrt(2.0) * 0.040),
            )
            ranked.append((score, yaw, distance, source.candidate_id, source))
        if not ranked:
            raise SchemaValidationError("R1 v15 target lacks a source morphology")
        score, yaw, distance, _candidate_id, source = min(ranked)
        if (
            score > 1.0 + 1.0e-8
            or yaw > math.radians(10.0) + 1.0e-8
            or distance > math.sqrt(2.0) * 0.040 + 1.0e-8
        ):
            raise SchemaValidationError(
                f"R1 v15 target lacks a bounded nearby source: {target.candidate_id}"
            )
        result[target.candidate_id] = source
    if len(result) != len(target_cases):
        raise SchemaValidationError("R1 v15 source matching is incomplete")
    return result


def _load_clean_source_evidence(
    expected_candidate_ids: set[str],
) -> dict[str, Path]:
    paths: dict[str, list[Path]] = {}
    for path in sorted(SOURCE_ISAAC.rglob("evaluation_episodes.jsonl")):
        candidate_id = path.parents[1].name
        if candidate_id in expected_candidate_ids:
            paths.setdefault(candidate_id, []).append(path)
    accepted: dict[str, Path] = {}
    for candidate_id in sorted(expected_candidate_ids):
        for path in paths.get(candidate_id, ()):
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(records) == 2 and all(
                record.get("task_success") is True
                and record.get("safety_failure") is False
                and int(record.get("fallback_decision_count", -1)) == 0
                for record in records
            ):
                accepted[candidate_id] = path
                break
    if set(accepted) != expected_candidate_ids:
        missing = sorted(expected_candidate_ids - set(accepted))
        raise SchemaValidationError(
            f"R1 v15 lacks clean L2 source evidence: {missing[:4]}"
        )
    return accepted


def _write_source_level_result(
    *, output_root: Path, cases: list[Any], evidence: dict[str, Path]
) -> Path:
    path = output_root / "results" / f"{SOURCE_LEVEL}.json"
    if path.is_file():
        payload = _read_json(path)
        if (
            payload.get("result_version") != "order9_r1_scene_source_result_v15"
            or payload.get("status") != "accepted"
            or int(payload.get("candidate_count", -1)) != 682
            or int(payload.get("episode_count", -1)) != 1364
            or int(payload.get("success_count", -1)) != 1364
        ):
            raise SchemaValidationError("R1 v15 cached source result differs")
        return path
    entries = [
        {
            "candidate_id": case.candidate_id,
            "source_bucket_id": case.source_bucket.bucket_id,
            "module_count": case.source_bucket.module_count,
            "sample_kind": case.sample_kind,
            "sample_index": case.sample_index,
            "episodes": _binding(evidence[case.candidate_id]),
            "episode_count": 2,
            "success_count": 2,
            "safety_failure_count": 0,
            "fallback_count": 0,
        }
        for case in cases
    ]
    return _atomic_json(
        path,
        {
            "result_version": "order9_r1_scene_source_result_v15",
            "status": "accepted",
            "level_id": SOURCE_LEVEL,
            "bucket_count": 22,
            "candidate_count": 682,
            "episode_count": 1364,
            "success_count": 1364,
            "safety_failure_count": 0,
            "fallback_count": 0,
            "pi_l_actor_command_applied": False,
            "entries": entries,
        },
    )


def _prove_target_case(
    *,
    target_case: Any,
    source_case: Any,
    source_evidence: Path,
    output_root: Path,
) -> Path:
    proof_path = (
        output_root
        / "proofs"
        / target_case.level_id
        / f"{target_case.candidate_id}.json"
    )
    if proof_path.is_file():
        proof = _read_json(proof_path)
        if (
            proof.get("proof_version") != "order9_r1_scene_equivalence_proof_v15"
            or proof.get("status") != "accepted"
            or proof.get("target_candidate_id") != target_case.candidate_id
            or proof.get("source_candidate_id") != source_case.candidate_id
            or proof.get("source_isaac_evidence", {}).get("sha256")
            != hash_file(source_evidence)
        ):
            raise SchemaValidationError("R1 v15 cached scene proof differs")
        return proof_path
    source_root = SOURCE_PREPARED / source_case.candidate_id
    audit = audit_order9_r1_scene_transfer_case(
        reference_case_root=source_root,
        target_task_spec=target_case.task_spec,
        repository_root=REPOSITORY,
    )
    return _atomic_json(
        proof_path,
        {
            "proof_version": "order9_r1_scene_equivalence_proof_v15",
            "status": "accepted",
            "source_candidate_id": source_case.candidate_id,
            "target_candidate_id": target_case.candidate_id,
            "source_bucket_id": target_case.source_bucket.bucket_id,
            "module_count": target_case.source_bucket.module_count,
            "sample_kind": target_case.sample_kind,
            "sample_index": target_case.sample_index,
            "target_offsets": [
                target_case.x_offset_m,
                target_case.y_offset_m,
                target_case.yaw_offset_rad,
            ],
            "source_isaac_evidence": {
                **_binding(source_evidence),
                "episode_count": 2,
                "success_count": 2,
                "safety_failure_count": 0,
                "fallback_count": 0,
            },
            "scene_transfer_audit": audit,
            "support_transformed_with_scene": True,
            "pi_l_actor_command_applied": False,
            "controller_layers_invoked": False,
            "isaac_invoked": False,
            "training_eligible": False,
        },
    )


def _prove_level(
    *,
    level_id: str,
    matched_sources: dict[str, Any],
    target_cases: list[Any],
    evidence: dict[str, Path],
    output_root: Path,
    workers: int,
) -> Path:
    paths: list[Path] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        futures = {}
        for target in target_cases:
            source = matched_sources[target.candidate_id]
            future = executor.submit(
                _prove_target_case,
                target_case=target,
                source_case=source,
                source_evidence=evidence[source.candidate_id],
                output_root=output_root,
            )
            futures[future] = target.candidate_id
        for index, future in enumerate(as_completed(futures), start=1):
            paths.append(future.result())
            if index % 50 == 0 or index == len(futures):
                print(
                    f"ORDER9_R1_V15_PROOF_PROGRESS={level_id}:{index}/{len(futures)}",
                    flush=True,
                )
    summaries = [_read_json(path) for path in sorted(paths)]
    maximum_position_error = max(
        float(
            item["scene_transfer_audit"]["invariant_audit"][
                "maximum_object_relative_position_error_m"
            ]
        )
        for item in summaries
    )
    maximum_attitude_error = max(
        float(
            item["scene_transfer_audit"]["invariant_audit"][
                "maximum_object_relative_attitude_error_rad"
            ]
        )
        for item in summaries
    )
    minimum_clearance = min(
        float(
            item["scene_transfer_audit"]["collision_certificate"]["minimum_clearance_m"]
        )
        for item in summaries
    )
    result_path = output_root / "results" / f"{level_id}.json"
    if result_path.is_file():
        return result_path
    return _atomic_json(
        result_path,
        {
            "result_version": "order9_r1_scene_equivalence_level_result_v15",
            "status": "accepted",
            "level_id": level_id,
            "bucket_count": 22,
            "candidate_count": len(summaries),
            "accepted_proof_count": sum(
                item.get("status") == "accepted" for item in summaries
            ),
            "source_success_episode_count": 2 * len(summaries),
            "maximum_object_relative_position_error_m": maximum_position_error,
            "maximum_object_relative_attitude_error_rad": maximum_attitude_error,
            "minimum_collision_clearance_m": minimum_clearance,
            "support_transformed_with_scene": True,
            "joint_targets_exactly_preserved": True,
            "contact_assignments_exactly_preserved": True,
            "controller_layers_invoked": False,
            "isaac_invoked": False,
            "training_eligible": False,
            "proofs": [_binding(path) for path in sorted(paths)],
        },
    )


def _replace(arguments: list[str], option: str, value: str) -> None:
    indices = [index for index, item in enumerate(arguments) if item == option]
    if len(indices) != 1 or indices[0] + 1 >= len(arguments):
        raise SchemaValidationError(f"R1 v15 command lacks one {option}")
    arguments[indices[0] + 1] = value


def _prepare_representative(
    *,
    target_case: Any,
    source_case: Any,
    output_root: Path,
    isaac_python: str,
    rollout_steps: int,
) -> dict[str, Any]:
    destination = output_root / "representatives" / target_case.candidate_id
    if not destination.is_dir():
        derive_order9_r1_planar_nominal_transfer_case(
            reference_case_root=SOURCE_PREPARED / source_case.candidate_id,
            target_task_spec=target_case.task_spec,
            target_candidate_id=target_case.candidate_id,
            target_level_id=target_case.level_id,
            target_seed=target_case.seed,
            destination_case_root=destination,
            repository_root=REPOSITORY,
        )
    materialized = load_order9_r1_isaac_case(
        destination / "case_manifest.json", REPOSITORY
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
        raise SchemaValidationError("R1 v15 protected rollout command differs")
    argv = list(command[indices[0] + 1 :])
    _replace(argv, "--num-envs", "1")
    _replace(argv, "--evaluation-episode-count", "1")
    job_path = destination / "isaac_job.json"
    output = destination / "isaac_replay"
    if not job_path.is_file():
        _atomic_json(
            job_path,
            {
                "version": "order9_r1_scene_range_representative_job_v15",
                "candidate_id": target_case.candidate_id,
                "source_reference_candidate_id": source_case.candidate_id,
                "r1_diagnostic_replay_count": 1,
                "r1_nominal_additional_compression_sweep_mm": "5.0",
                "r1_release_height_offset_m": 0.0,
                "r1_scene_support_from_task": True,
                "training_eligible": False,
                "formal_teacher_collection_authorized": False,
                "jobs": [
                    {
                        "name": target_case.candidate_id,
                        "argv": argv,
                        "log_path": str(log_path),
                    }
                ],
            },
        )
    return {
        "candidate_id": target_case.candidate_id,
        "module_count": target_case.source_bucket.module_count,
        "source_candidate_id": source_case.candidate_id,
        "case_root": destination,
        "job_path": job_path,
        "output_root": output,
        "command": [
            isaac_python,
            str(WRAPPER),
            "--r1-batched-jobs",
            str(job_path),
            "--r1-batched-output-root",
            str(output),
            "--no-formal-annotation",
        ],
    }


def _representative_episode_path(item: dict[str, Any]) -> Path:
    return (
        item["output_root"]
        / item["candidate_id"]
        / "isaac"
        / "evaluation_episodes.jsonl"
    )


def _run_one_representative(item: dict[str, Any]) -> tuple[str, int]:
    episode_path = _representative_episode_path(item)
    if episode_path.is_file():
        return item["candidate_id"], 0
    log_path = item["case_root"] / "isaac_launcher.log"
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    environment["TORCHINDUCTOR_COMPILE_THREADS"] = "8"
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            item["command"],
            cwd=REPOSITORY,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return item["candidate_id"], completed.returncode


def _validate_representative(item: dict[str, Any]) -> dict[str, Any]:
    episode_path = _representative_episode_path(item)
    records = [
        json.loads(line)
        for line in episode_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    raw_path = episode_path.with_name("evaluation_rollout.pt")
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    target_task = _read_json(item["case_root"] / "task_spec.json")
    expected_support = target_task["scene"]["environment"]["support_surfaces"][0][
        "pose_world"
    ]
    actual_support = raw["metadata"]["task_specs"][0]["scene"]["environment"][
        "support_surfaces"
    ][0]["pose_world"]
    metadata = raw.get("metadata", {})
    accepted = (
        len(records) == 1
        and records[0].get("task_success") is True
        and records[0].get("safety_failure") is False
        and int(records[0].get("fallback_decision_count", -1)) == 0
        and float(records[0].get("metrics", {}).get("hard_collision", -1.0)) == 0.0
        and float(records[0].get("metrics", {}).get("object_dropped", -1.0)) == 0.0
        and float(records[0].get("metrics", {}).get("timeout", -1.0)) == 0.0
        and metadata.get("r1_scene_support_from_task") is True
        and metadata.get("pi_l_actor_command_applied") is False
        and metadata.get("promotion_evidence_eligible") is False
        and metadata.get("training_eligible") is False
        and actual_support == expected_support
    )
    return {
        "candidate_id": item["candidate_id"],
        "module_count": item["module_count"],
        "status": "accepted" if accepted else "rejected",
        "episode_count": len(records),
        "success_count": sum(record.get("task_success") is True for record in records),
        "safety_failure_count": sum(
            record.get("safety_failure") is True for record in records
        ),
        "fallback_count": sum(
            int(record.get("fallback_decision_count", 0)) for record in records
        ),
        "task_support_pose_world": expected_support,
        "isaac_support_pose_world": actual_support,
        "episodes": _binding(episode_path),
        "raw_rollout": _binding(raw_path),
    }


def _select_representatives(
    target_cases: list[Any], matched_sources: dict[str, Any]
) -> list[tuple[Any, Any]]:
    selected = []
    seen: set[int] = set()
    for target in target_cases:
        if target.sample_kind != "lattice" or target.sample_index != 8:
            continue
        module_count = int(target.source_bucket.module_count)
        if module_count in seen:
            continue
        selected.append((target, matched_sources[target.candidate_id]))
        seen.add(module_count)
    if seen != set(range(2, 9)):
        raise SchemaValidationError("R1 v15 representatives lack a module stratum")
    return selected


def main() -> int:
    args = _parser().parse_args()
    if (
        not 1 <= args.proof_workers <= 24
        or not 1 <= args.maximum_parallel_isaac <= 4
        or args.rollout_steps < 1
        or os.environ.get("PYTHONHASHSEED") != "0"
    ):
        raise ValueError("R1 v15 runtime contract differs")
    protocol, approval = _load_contract()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    source_case_list = _cases(SOURCE_LEVEL)
    source_cases = {case.candidate_id: case for case in source_case_list}
    source_evidence = _load_clean_source_evidence(set(source_cases))
    source_result = _write_source_level_result(
        output_root=output_root,
        cases=source_case_list,
        evidence=source_evidence,
    )
    level_results = {SOURCE_LEVEL: source_result}
    target_case_sets = {level: _cases(level) for level in LEVELS[1:]}
    matched_sources = {
        level: _match_source_cases(source_case_list, target_case_sets[level])
        for level in LEVELS[1:]
    }
    for level in LEVELS[1:]:
        level_results[level] = _prove_level(
            level_id=level,
            matched_sources=matched_sources[level],
            target_cases=target_case_sets[level],
            evidence=source_evidence,
            output_root=output_root,
            workers=args.proof_workers,
        )

    representatives = [
        _prepare_representative(
            target_case=target,
            source_case=source,
            output_root=output_root,
            isaac_python=args.isaac_python,
            rollout_steps=args.rollout_steps,
        )
        for target, source in _select_representatives(
            target_case_sets[LEVELS[-1]], matched_sources[LEVELS[-1]]
        )
    ]
    if args.prepare_only:
        print(
            "ORDER9_R1_V15_PREPARED="
            + json.dumps(
                {
                    "output_root": str(output_root),
                    "proof_counts": {LEVELS[1]: 682, LEVELS[2]: 682},
                    "representative_count": len(representatives),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    with ThreadPoolExecutor(max_workers=args.maximum_parallel_isaac) as executor:
        futures = {
            executor.submit(_run_one_representative, item): item
            for item in representatives
        }
        for future in as_completed(futures):
            candidate_id, returncode = future.result()
            print(
                "ORDER9_R1_V15_ISAAC="
                + json.dumps(
                    {"candidate_id": candidate_id, "returncode": returncode},
                    sort_keys=True,
                ),
                flush=True,
            )
    representative_results = [
        _validate_representative(item) for item in representatives
    ]
    representative_path = output_root / "results" / "l4_representative_isaac.json"
    if not representative_path.is_file():
        _atomic_json(
            representative_path,
            {
                "result_version": "order9_r1_scene_representative_isaac_v15",
                "status": (
                    "accepted"
                    if all(
                        item["status"] == "accepted" for item in representative_results
                    )
                    else "rejected"
                ),
                "level_id": LEVELS[-1],
                "module_counts": list(range(2, 9)),
                "candidate_count": len(representative_results),
                "episode_count": sum(
                    item["episode_count"] for item in representative_results
                ),
                "success_count": sum(
                    item["success_count"] for item in representative_results
                ),
                "safety_failure_count": sum(
                    item["safety_failure_count"] for item in representative_results
                ),
                "fallback_count": sum(
                    item["fallback_count"] for item in representative_results
                ),
                "entries": representative_results,
            },
        )
    representative_summary = _read_json(representative_path)
    status = (
        "accepted"
        if representative_summary.get("status") == "accepted"
        and all(
            _read_json(path).get("status") == "accepted"
            for path in level_results.values()
        )
        else "rejected"
    )
    ledger_path = output_root / "range_selection_result.json"
    if ledger_path.is_file():
        raise FileExistsError(ledger_path)
    ledger = {
        "ledger_version": "order9_r1_scene_invariant_range_selection_result_v15",
        "runner_version": RUNNER_VERSION,
        "status": status,
        "selected_level_id": LEVELS[-1] if status == "accepted" else None,
        "accepted_level_id": LEVELS[-1] if status == "accepted" else None,
        "selection_split": "train",
        "selection_bucket_count": 22,
        "levels": {level: _binding(path) for level, path in level_results.items()},
        "representative_isaac": _binding(representative_path),
        "complete_scene_transform_required": True,
        "support_transformed_with_scene": True,
        "pi_l_actor_command_applied": False,
        "qpid_qp_applied_for_representative_isaac": True,
        "local_servo_applied_for_representative_isaac": True,
        "held_out_confirmation_executed": False,
        "formal_teacher_collection_authorized": False,
        "training_eligible": False,
        "elapsed_s": time.perf_counter() - started,
        "bindings": {
            "protocol": _binding(PROTOCOL),
            "approval": _binding(APPROVAL),
            "scene_transfer_implementation": protocol["scene_transfer_implementation"],
            "runner_implementation": protocol["runner_implementation"],
            "isaac_wrapper_implementation": protocol["isaac_wrapper_implementation"],
            "protected_c3_checkpoint": protocol["protected_c3_checkpoint"],
            "protected_c3_rollout": protocol["protected_c3_rollout"],
            "protected_c3_curriculum": protocol["protected_c3_curriculum"],
            "protected_c3_release_ledger": protocol["protected_c3_release_ledger"],
        },
        "approval_record_version": approval["record_version"],
    }
    _atomic_json(ledger_path, ledger)
    print(
        "ORDER9_R1_V15_COMPLETE="
        + json.dumps(
            {
                "status": status,
                "accepted_level_id": ledger["accepted_level_id"],
                "ledger": str(ledger_path),
                "ledger_sha256": hash_file(ledger_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if status == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
