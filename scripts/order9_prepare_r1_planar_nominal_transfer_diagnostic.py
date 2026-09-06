#!/usr/bin/env python3
from __future__ import annotations

"""Prepare one l3 diagnostic by moving one successful nearby complete scene."""

import argparse
import json
import os
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.schemas.common import SchemaValidationError  # noqa: E402
from amsrr.schemas.policies import ContactWrenchTrajectory  # noqa: E402
from amsrr.schemas.task_spec import TaskSpec  # noqa: E402
from amsrr.training.order9_c3_nominal_trajectory import (  # noqa: E402
    validate_order9_c3_nominal_trajectory_artifact_bytes,
    validate_order9_c3_nominal_trajectory_set_bytes,
)
from amsrr.training.order9_r1_isaac_calibration import (  # noqa: E402
    load_order9_r1_isaac_case,
)
from amsrr.training.order9_r1_nominal_calibration import (  # noqa: E402
    order9_r1_nominal_case_command,
)
from amsrr.training.order9_r1_yaw_branch_repair import (  # noqa: E402
    _object_start_and_goal,
    _validated_task_scene_rigid_transform,
    audit_order9_r1_scene_transfer_invariants,
    derive_order9_r1_planar_nominal_transfer_case,
    transform_order9_r1_planar_task_scene,
    transform_order9_r1_scene_trajectory,
)
from amsrr.utils.hashing import hash_file  # noqa: E402
from scripts import order9_run_r1_range_selection_v11 as runtime_v11  # noqa: E402

DEFAULT_REFERENCE = REPOSITORY / (
    "artifacts/p4_full/order9/r1_teacher/range_selection_v13/prepared/"
    "r1_l2_20mm_10deg/"
    "r1_l2_20mm_10deg__train__train-000004-190f3a425b3e__lattice_08"
)
DEFAULT_REFERENCE_EVIDENCE = REPOSITORY / (
    "artifacts/p4_full/order9/r1_teacher/range_selection_v13/isaac/"
    "r1_l2_20mm_10deg/batches/train-000004-190f3a425b3e/"
    "chunk_000_59c8d12532cb/"
    "r1_l2_20mm_10deg__train__train-000004-190f3a425b3e__lattice_08/"
    "isaac/evaluation_episodes.jsonl"
)
DEFAULT_CANDIDATE = "r1_l3_30mm_15deg__train__train-000004-190f3a425b3e__lattice_08"
DEFAULT_OUTPUT = REPOSITORY / (
    "artifacts/p4_full/order9/r1_teacher/diagnostics/"
    "r1_l3_planar_scene_transfer_v4/" + DEFAULT_CANDIDATE
)
ISAAC_PYTHON = "/home/leus/.local/share/mamba/envs/isaaclab3/bin/python"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-case-root", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--candidate-id", default=DEFAULT_CANDIDATE)
    parser.add_argument("--output-case-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--isaac-python", default=ISAAC_PYTHON)
    parser.add_argument("--rollout-steps", type=int, default=15000)
    return parser


def _replace(arguments: list[str], option: str, value: str) -> None:
    indices = [index for index, item in enumerate(arguments) if item == option]
    if len(indices) != 1 or indices[0] + 1 >= len(arguments):
        raise SchemaValidationError(f"R1 planar diagnostic lacks one {option}")
    arguments[indices[0] + 1] = value


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _invariant_preflight(reference_root: Path, target_task: TaskSpec) -> dict:
    set_path = reference_root / "nominal_set" / "manifest.json"
    nominal_set = validate_order9_c3_nominal_trajectory_set_bytes(
        set_path, repository_root=REPOSITORY
    )
    if len(nominal_set.entries) != 1:
        raise SchemaValidationError("R1 planar preflight source set is not singular")
    artifact_path = set_path.parent / nominal_set.entries[0].artifact_path
    artifact = validate_order9_c3_nominal_trajectory_artifact_bytes(artifact_path)
    reference_task = TaskSpec.from_json(
        (reference_root / "task_spec.json").read_text(encoding="utf-8")
    )
    scene_task = transform_order9_r1_planar_task_scene(
        reference_task,
        target_identity_task=target_task,
    )
    delta = _validated_task_scene_rigid_transform(reference_task, scene_task)
    source_phases = {
        entry.phase: ContactWrenchTrajectory.from_json(
            (artifact_path.parent / entry.trajectory_path).read_text(encoding="utf-8")
        )
        for entry in artifact.phase_trajectories
    }
    target_phases = {
        phase: transform_order9_r1_scene_trajectory(
            trajectory,
            delta_pose_world=delta,
        )
        for phase, trajectory in source_phases.items()
    }
    reference_start, reference_goal = _object_start_and_goal(reference_task)
    target_start, target_goal = _object_start_and_goal(scene_task)
    audit = audit_order9_r1_scene_transfer_invariants(
        source_phases=source_phases,
        target_phases=target_phases,
        delta_pose_world=delta,
    )
    return {
        **audit,
        "source_task_start_pose_world": list(reference_start),
        "source_task_goal_pose_world": list(reference_goal),
        "target_task_start_pose_world": list(target_start),
        "target_task_goal_pose_world": list(target_goal),
        "fixed_scene_delta_pose_world": list(delta),
        "source_support_pose_world": list(
            reference_task.scene.environment.support_surfaces[0].pose_world
        ),
        "target_support_pose_world": list(
            scene_task.scene.environment.support_surfaces[0].pose_world
        ),
        "support_transformed_with_same_fixed_delta": True,
    }


def main() -> int:
    args = _parser().parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("R1 planar diagnostic requires PYTHONHASHSEED=0")
    destination = Path(args.output_case_root).resolve()
    diagnostic_root = (
        REPOSITORY / "artifacts/p4_full/order9/r1_teacher/diagnostics"
    ).resolve()
    try:
        destination.relative_to(diagnostic_root)
    except ValueError as error:
        raise ValueError(
            "R1 planar diagnostic output must remain diagnostic"
        ) from error

    matching = [
        case
        for case in runtime_v11._corrected_cases("r1_l3_30mm_15deg", "train")
        if case.candidate_id == args.candidate_id
    ]
    if len(matching) != 1:
        raise SchemaValidationError("R1 planar diagnostic target is not one l3 case")
    case = matching[0]
    if (
        case.sample_kind != "lattice"
        or case.x_offset_m != -0.030
        or case.y_offset_m != 0.030
        or abs(case.yaw_offset_rad - 0.2617993877991494) > 1.0e-12
        or case.source_bucket.bucket_id != "train-000004-190f3a425b3e"
    ):
        raise SchemaValidationError("R1 planar diagnostic target offsets differ")
    invariant_preflight = _invariant_preflight(
        Path(args.reference_case_root).resolve(), case.task_spec
    )
    reference_task = TaskSpec.from_json(
        (Path(args.reference_case_root) / "task_spec.json").read_text(encoding="utf-8")
    )
    expected_reference_offsets = {
        "r1_initial_x_offset_m": -0.020,
        "r1_initial_y_offset_m": 0.020,
        "r1_initial_yaw_offset_rad": 0.17453292519943295,
    }
    if any(
        abs(float(reference_task.metadata[key]) - expected) > 1.0e-12
        for key, expected in expected_reference_offsets.items()
    ):
        raise SchemaValidationError("R1 scene reference is not the nearby l2 case")
    reference_records = [
        json.loads(line)
        for line in DEFAULT_REFERENCE_EVIDENCE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if (
        len(reference_records) != 2
        or any(record.get("task_success") is not True for record in reference_records)
        or any(
            record.get("safety_failure") is not False for record in reference_records
        )
        or any(
            int(record.get("fallback_decision_count", -1)) != 0
            for record in reference_records
        )
    ):
        raise SchemaValidationError("R1 planar nominal reference success differs")

    try:
        admission = derive_order9_r1_planar_nominal_transfer_case(
            reference_case_root=Path(args.reference_case_root),
            target_task_spec=case.task_spec,
            target_candidate_id=case.candidate_id,
            target_level_id=case.level_id,
            target_seed=case.seed,
            destination_case_root=destination,
            repository_root=REPOSITORY,
        )
    except SchemaValidationError as error:
        rejection_path = destination.parent / (
            destination.name + "__lightweight_rejection.json"
        )
        rejection = {
            "record_version": "order9_r1_planar_scene_transfer_rejection_v4",
            "status": "rejected_before_isaac",
            "candidate_id": case.candidate_id,
            "source_reference_candidate_id": json.loads(
                (Path(args.reference_case_root) / "case_manifest.json").read_text(
                    encoding="utf-8"
                )
            )["candidate_id"],
            "target_offsets": [
                case.x_offset_m,
                case.y_offset_m,
                case.yaw_offset_rad,
            ],
            "failure_reason": str(error),
            "planar_transfer_invariant_audit": invariant_preflight,
            "nominal_reference_isaac_evidence": {
                "path": str(DEFAULT_REFERENCE_EVIDENCE.relative_to(REPOSITORY)),
                "sha256": hash_file(DEFAULT_REFERENCE_EVIDENCE),
                "success_count": 2,
                "episode_count": 2,
                "safety_failure_count": 0,
                "fallback_count": 0,
            },
            "controller_layers_invoked": False,
            "isaac_invoked": False,
            "training_eligible": False,
            "formal_range_evidence_eligible": False,
        }
        _atomic_json(rejection_path, rejection)
        print(
            "ORDER9_R1_PLANAR_NOMINAL_TRANSFER_REJECTED="
            + json.dumps(rejection, sort_keys=True)
        )
        return 2
    materialized = load_order9_r1_isaac_case(
        destination / "case_manifest.json", REPOSITORY
    )
    command, log_path = order9_r1_nominal_case_command(
        materialized,
        repository_root=REPOSITORY,
        python_executable=args.isaac_python,
        rollout_steps=args.rollout_steps,
    )
    rollout_indices = [
        index
        for index, value in enumerate(command)
        if Path(value).name == "order9_vectorized_isaac_rollout.py"
    ]
    if len(rollout_indices) != 1:
        raise SchemaValidationError("R1 planar protected rollout command differs")
    argv = list(command[rollout_indices[0] + 1 :])
    _replace(argv, "--num-envs", "1")
    _replace(argv, "--evaluation-episode-count", "1")
    job_manifest = destination / "isaac_one_replay_job.json"
    output_root = destination / "isaac_one_replay"
    payload = {
        "version": "order9_r1_planar_nominal_transfer_diagnostic_job_v1",
        "candidate_id": case.candidate_id,
        "source_reference_candidate_id": admission["reference_candidate_id"],
        "r1_diagnostic_replay_count": 1,
        "r1_nominal_additional_compression_sweep_mm": "5.0",
        "r1_release_height_offset_m": 0.0,
        "r1_scene_support_from_task": True,
        "training_eligible": False,
        "formal_range_evidence_eligible": False,
        "jobs": [
            {
                "name": case.candidate_id,
                "argv": argv,
                "log_path": str(log_path),
            }
        ],
    }
    _atomic_json(job_manifest, payload)
    result = {
        **admission,
        "target_offsets": [case.x_offset_m, case.y_offset_m, case.yaw_offset_rad],
        "job_manifest_path": str(job_manifest.relative_to(REPOSITORY)),
        "job_manifest_sha256": hash_file(job_manifest),
        "isaac_output_root": str(output_root.relative_to(REPOSITORY)),
        "isaac_command": [
            args.isaac_python,
            str(
                REPOSITORY / "scripts/order9_r1_batched_nominal_compression_rollout.py"
            ),
            "--r1-batched-jobs",
            str(job_manifest),
            "--r1-batched-output-root",
            str(output_root),
            "--no-formal-annotation",
        ],
    }
    _atomic_json(destination / "planar_nominal_transfer_result.json", result)
    print("ORDER9_R1_PLANAR_NOMINAL_TRANSFER=" + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
