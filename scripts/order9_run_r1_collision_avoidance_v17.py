#!/usr/bin/env python3
from __future__ import annotations

"""Prepare and replay the four diagnosed R1 v16 collision repairs."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
from unittest.mock import patch

for _name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import torch

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.schemas.common import SchemaValidationError  # noqa: E402
from amsrr.schemas.datasets import DatasetSplit  # noqa: E402
from amsrr.schemas.morphology import MorphologyGraph  # noqa: E402
from amsrr.schemas.task_spec import TaskSpec  # noqa: E402
from amsrr.feasibility.articulated_reachability import (  # noqa: E402
    ArticulatedContactIKSolver,
)
from amsrr.robot_model.whole_structure_kinematics import (  # noqa: E402
    ordered_global_dock_joint_ids,
)
from amsrr.training.order9_c3_nominal_trajectory import (  # noqa: E402
    write_order9_c3_nominal_trajectory_artifact,
)
from amsrr.training.order9_r1_calibration import (  # noqa: E402
    load_order9_r1_calibration_protocol,
)
from amsrr.training.order9_r1_collision_avoidance_v17 import (  # noqa: E402
    ORDER9_R1_COLLISION_AVOIDANCE_V17_VERSION,
    Order9R1CollisionAvoidanceHeuristicV17,
    order9_r1_collision_avoidance_heuristics_v17,
)
from amsrr.training.order9_r1_complete_task_materialization_v14 import (  # noqa: E402
    finalize_order9_r1_v14_materialized_case,
)
from amsrr.training.order9_r1_isaac_calibration import (  # noqa: E402
    load_order9_r1_isaac_case,
    materialize_order9_r1_isaac_case,
)
from amsrr.training.order9_r1_nominal_calibration import (  # noqa: E402
    order9_r1_nominal_case_command,
)
from amsrr.training.order9_r1_nominal_geometry_v11 import (  # noqa: E402
    ORDER9_R1_NOMINAL_GEOMETRY_V11_RELATIVE,
    load_order9_r1_nominal_geometry_v11_contract,
)
from amsrr.training.order9_r1_range_selection_v13 import (  # noqa: E402
    Order9R1ExactMarginTeacherScreenPipelineV13,
)
from amsrr.training.order9_r1_safe_timing import (  # noqa: E402
    order9_r1_safe_phase_time_scales,
)
from amsrr.training.order9_r1_support_clearance_teacher_v12 import (  # noqa: E402
    ORDER9_R1_SUPPORT_CLEARANCE_TEACHER_V12_RELATIVE,
    load_order9_r1_support_clearance_teacher_v12_contract,
)
from amsrr.training import (
    order9_r1_early_tilt_pipeline as early_tilt_runtime,
)  # noqa: E402
from amsrr.training import order9_r1_yaw_branch_repair as yaw_runtime  # noqa: E402
from amsrr.training.order9_r1_yaw_branch_repair import (  # noqa: E402
    derive_order9_r1_planar_nominal_transfer_case,
)
from amsrr.utils.hashing import hash_file  # noqa: E402
from scripts import order9_run_r1_held_out_scene_confirmation_v16 as v16  # noqa: E402

RUNNER_VERSION = "order9_r1_collision_avoidance_runner_v17"
OUTPUT_ROOT = REPOSITORY / (
    "artifacts/p4_full/order9/r1_teacher/diagnostics/"
    "held_out_collision_avoidance_v17"
)
WRAPPER = REPOSITORY / "scripts/order9_r1_batched_nominal_compression_rollout.py"
ISAAC_PYTHON = "/home/leus/.local/share/mamba/envs/isaaclab3/bin/python"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--isaac-python", default=ISAAC_PYTHON)
    parser.add_argument("--rollout-steps", type=int, default=15000)
    parser.add_argument("--maximum-parallel-isaac", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_text = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
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


def _binding(path: Path) -> dict[str, str]:
    return {
        "path": path.resolve().relative_to(REPOSITORY).as_posix(),
        "sha256": hash_file(path),
    }


def _replace_argument(arguments: list[str], option: str, value: str) -> None:
    matches = [index for index, item in enumerate(arguments) if item == option]
    if len(matches) != 1 or matches[0] + 1 >= len(arguments):
        raise SchemaValidationError(f"R1 v17 command lacks exactly one {option}")
    arguments[matches[0] + 1] = value


def _diagnostic_case_task_path(case_root: Path, repository: Path) -> Path:
    root = Path(case_root).resolve()
    task_path = root / "task_spec.json"
    case = json.loads((root / "case_manifest.json").read_text(encoding="utf-8"))
    task = TaskSpec.from_json(task_path.read_text(encoding="utf-8"))
    binding = case.get("task_spec", {})
    bound_path = (REPOSITORY / str(binding.get("path", ""))).resolve()
    if (
        task.metadata.get("dataset_split") != "held_out"
        or task.metadata.get("r1_calibration_candidate_id") != case.get("candidate_id")
        or bound_path.name != "task_spec.json"
        or OUTPUT_ROOT not in bound_path.parents
        or binding.get("sha256") != hash_file(task_path)
        or Path(repository).resolve() != REPOSITORY
    ):
        raise SchemaValidationError("R1 v17 diagnostic task binding differs")
    return task_path


def _renamed_task(task: TaskSpec, *, candidate_id: str) -> TaskSpec:
    payload = task.to_dict()
    payload["task_id"] = f"{task.task_id}:{ORDER9_R1_COLLISION_AVOIDANCE_V17_VERSION}"
    metadata = deepcopy(payload.get("metadata", {}))
    metadata.update(
        {
            "order9_rollout_bucket_id": candidate_id,
            "r1_calibration_candidate_id": candidate_id,
            "r1_collision_avoidance_version": (
                ORDER9_R1_COLLISION_AVOIDANCE_V17_VERSION
            ),
            "r1_formal_teacher_collection_eligible": False,
        }
    )
    payload["metadata"] = metadata
    value = TaskSpec.from_dict(payload)
    value.validate()
    return value


def _pipeline() -> Order9R1ExactMarginTeacherScreenPipelineV13:
    base = load_order9_r1_calibration_protocol(
        v16.BASE_PROTOCOL, repository_root=REPOSITORY
    )
    geometry = load_order9_r1_nominal_geometry_v11_contract(
        ORDER9_R1_NOMINAL_GEOMETRY_V11_RELATIVE,
        repository_root=REPOSITORY,
    )
    clearance = load_order9_r1_support_clearance_teacher_v12_contract(
        ORDER9_R1_SUPPORT_CLEARANCE_TEACHER_V12_RELATIVE,
        repository_root=REPOSITORY,
    )
    return Order9R1ExactMarginTeacherScreenPipelineV13(
        repository_root=REPOSITORY,
        source_bucket_manifest_path=REPOSITORY / base.source_bucket_manifest.path,
        minimum_normalized_joint_limit_reserve=(
            base.minimum_normalized_joint_limit_reserve
        ),
        maximum_body_tilt_rad=base.maximum_body_tilt_rad,
        anchor_position_tolerance_m=0.030,
        enforce_joint_limit_reserve_during_ik=True,
        geometry_contract=geometry,
        clearance_contract=clearance,
        physical_model_config_path=REPOSITORY / "configs/robot/robot_model.yaml",
    )


def _prepare_reference(
    *,
    pair,
    source_case,
    heuristic: Order9R1CollisionAvoidanceHeuristicV17,
    rule_root: Path,
) -> Path:
    destination = rule_root / "reference_case"
    if destination.is_dir():
        load_order9_r1_isaac_case(destination / "case_manifest.json", REPOSITORY)
        return destination
    case = v16._reference_case(pair, source_case)
    # Keep the calibrated candidate grammar intact.  The heuristic identity is
    # carried by the isolated artifact root and task metadata instead.
    task = _renamed_task(case.task_spec, candidate_id=case.candidate_id)
    case = replace(case, task_spec=task)
    pipeline = _pipeline()
    option = heuristic.teacher_option
    pipeline.teacher_fast_paths[case.candidate_id] = (option,)
    pipeline.teacher_candidate_overrides[case.candidate_id] = (option,)
    pipeline.strict_preferred_candidate_options = True
    pipeline.bounded_local_contact_repairs = frozenset(
        value
        for value in pipeline.bounded_local_contact_repairs
        if value != case.candidate_id
    )
    free_joint_seed = None
    forced_free_joint_targets = None
    if heuristic.free_joint_targets_rad:
        morphology = MorphologyGraph.from_json(
            (REPOSITORY / pair.asset_entry.morphology_graph_path).read_text(
                encoding="utf-8"
            )
        )
        ordered_ids = ordered_global_dock_joint_ids(morphology, pipeline.physical_model)
        requested = dict(heuristic.free_joint_targets_rad)
        if not set(requested).issubset(ordered_ids):
            raise SchemaValidationError("R1 v17 free-joint target is absent")
        free_joint_seed = {
            joint_id: float(requested.get(joint_id, 0.0)) for joint_id in ordered_ids
        }
        forced_free_joint_targets = requested
        pipeline.configuration_goal_seeds[case.candidate_id] = free_joint_seed
    if free_joint_seed is None:
        prepared = pipeline.prepare(case)
    else:
        original_generate = (
            early_tilt_runtime.generate_order9_c3_nominal_grasp_trajectory
        )
        original_contact_solve = ArticulatedContactIKSolver.solve

        def generate_with_free_joint_seed(*values, **keywords):
            keywords["contact_goal_joint_seed_positions_rad"] = free_joint_seed
            keywords["configuration_goal_joint_seed_positions_rad"] = free_joint_seed
            return original_generate(*values, **keywords)

        def solve_with_free_joint_target(solver, *values, **keywords):
            solution = original_contact_solve(solver, *values, **keywords)
            if not solution.feasible:
                return solution
            q = dict(solution.joint_positions_rad)
            assert forced_free_joint_targets is not None
            if not set(forced_free_joint_targets).issubset(q):
                raise SchemaValidationError("R1 v17 IK solution lacks free joint")
            q.update(forced_free_joint_targets)
            return replace(solution, joint_positions_rad=q)

        with patch.object(
            early_tilt_runtime,
            "generate_order9_c3_nominal_grasp_trajectory",
            side_effect=generate_with_free_joint_seed,
        ), patch.object(
            ArticulatedContactIKSolver,
            "solve",
            new=solve_with_free_joint_target,
        ):
            prepared = pipeline.prepare(case)
    prepared.validate_for(case)
    if not prepared.teacher_trajectory_complete:
        raise SchemaValidationError(str(prepared.failure_reason))
    screen = pipeline.screen(case, prepared)
    if not screen.accepted or not screen.eligible_for_full_control_test:
        raise SchemaValidationError(
            "R1 v17 heuristic failed lightweight screen: "
            + ",".join(screen.violation_codes)
        )

    source_root = rule_root / "reference_source"
    source_root.mkdir(parents=True, exist_ok=False)
    task_path = _write_json(source_root / "task_spec.json", task.to_dict())
    source_artifact = source_root / "artifact"
    write_order9_c3_nominal_trajectory_artifact(
        prepared.payload,
        output_dir=source_artifact,
        bucket_id=case.candidate_id,
        split=DatasetSplit.VALIDATION,
        task_spec=task,
        lift_clearance_m=0.3,
        task_spec_sha256=hash_file(task_path),
        structural_hash=pair.held_out_entry.structural_hash,
        physical_model_hash=case.physical_model_hash,
        robot_urdf_path=(REPOSITORY / pair.asset_entry.urdf_path).resolve(),
    )
    source_manifest = source_artifact / "manifest.json"
    case.source_bucket.metadata["accepted_nominal_trajectory"] = {
        "artifact_path": source_manifest.relative_to(REPOSITORY).as_posix(),
        "artifact_sha256": hash_file(source_manifest),
        "r1_collision_avoidance_heuristic": heuristic.heuristic_id,
    }
    materialized = materialize_order9_r1_isaac_case(
        case=case,
        prepared=prepared,
        screen=screen,
        output_dir=destination,
        repository_root=REPOSITORY,
        approved_protocol_path=v16.BASE_PROTOCOL,
    )
    geometry = load_order9_r1_nominal_geometry_v11_contract(
        ORDER9_R1_NOMINAL_GEOMETRY_V11_RELATIVE,
        repository_root=REPOSITORY,
    )
    clearance = load_order9_r1_support_clearance_teacher_v12_contract(
        ORDER9_R1_SUPPORT_CLEARANCE_TEACHER_V12_RELATIVE,
        repository_root=REPOSITORY,
    )
    phase_time_scales = order9_r1_safe_phase_time_scales(
        case.source_bucket.module_count
    )
    phase_time_scales["approach"] = heuristic.approach_time_scale
    finalize_order9_r1_v14_materialized_case(
        materialized,
        task_spec=task,
        phase_time_scales=phase_time_scales,
        joint_rate_limit_rad_s=(
            float(screen.maximum_joint_rate_rad_s)
            + float(screen.minimum_joint_rate_margin_rad_s)
        ),
        repository_root=REPOSITORY,
        geometry_contract=geometry,
        clearance_contract=clearance,
        clearance_generation_certificate=(
            pipeline.clearance_generation_certificates.get(case.candidate_id)
        ),
    )
    load_order9_r1_isaac_case(destination / "case_manifest.json", REPOSITORY)
    return destination


def _prepare_target(
    *,
    pair,
    source_case,
    heuristic: Order9R1CollisionAvoidanceHeuristicV17,
    reference: Path,
    rule_root: Path,
) -> Path:
    destination = rule_root / "target_case"
    if destination.is_dir():
        load_order9_r1_isaac_case(destination / "case_manifest.json", REPOSITORY)
        return destination
    original_candidate_id, identity = v16._target_identity_task(pair, source_case)
    candidate_id = original_candidate_id
    identity = _renamed_task(identity, candidate_id=candidate_id)
    reference_task = TaskSpec.from_json(
        (reference / "task_spec.json").read_text(encoding="utf-8")
    )
    midpoint, final_task, _delta_one, _delta_two, _overall = (
        v16.build_order9_r1_held_out_composed_target_v16(
            reference_task,
            target_identity_task=identity,
        )
    )
    midpoint_id = str(midpoint.metadata["r1_calibration_candidate_id"])
    intermediate = rule_root / "intermediate_case"
    original_resolver = yaw_runtime._source_bucket_task_path
    yaw_runtime._source_bucket_task_path = _diagnostic_case_task_path
    try:
        if not intermediate.is_dir():
            derive_order9_r1_planar_nominal_transfer_case(
                reference_case_root=reference,
                target_task_spec=midpoint,
                target_candidate_id=midpoint_id,
                target_level_id=f"{v16.TARGET_LEVEL}_v17_midpoint",
                target_seed=int(
                    pair.held_out_entry.requested_seed + source_case.sample_index
                ),
                destination_case_root=intermediate,
                repository_root=REPOSITORY,
            )
        derive_order9_r1_planar_nominal_transfer_case(
            reference_case_root=intermediate,
            target_task_spec=final_task,
            target_candidate_id=candidate_id,
            target_level_id=f"{v16.TARGET_LEVEL}_v17_collision_avoidance",
            target_seed=int(
                pair.held_out_entry.requested_seed + source_case.sample_index
            ),
            destination_case_root=destination,
            repository_root=REPOSITORY,
        )
    finally:
        yaw_runtime._source_bucket_task_path = original_resolver
    load_order9_r1_isaac_case(destination / "case_manifest.json", REPOSITORY)
    return destination


def _job(
    *,
    root: Path,
    heuristic: Order9R1CollisionAvoidanceHeuristicV17,
    isaac_python: str,
    rollout_steps: int,
) -> dict[str, Any]:
    materialized = load_order9_r1_isaac_case(root / "case_manifest.json", REPOSITORY)
    command, log_path = order9_r1_nominal_case_command(
        materialized,
        repository_root=REPOSITORY,
        python_executable=isaac_python,
        rollout_steps=rollout_steps,
    )
    runner_indices = [
        index
        for index, value in enumerate(command)
        if Path(value).name == "order9_vectorized_isaac_rollout.py"
    ]
    if len(runner_indices) != 1:
        raise SchemaValidationError("R1 v17 rollout command differs")
    argv = list(command[runner_indices[0] + 1 :])
    _replace_argument(argv, "--num-envs", "1")
    _replace_argument(argv, "--evaluation-episode-count", "1")
    if "--diagnostic-collision-evidence" not in argv:
        argv.append("--diagnostic-collision-evidence")
    job_path = root.parent / "isaac_job.json"
    replay_root = root.parent / "isaac_replay"
    two_stage = None
    _write_json(
        job_path,
        {
            "version": RUNNER_VERSION,
            "heuristic": heuristic.__dict__,
            "formal_teacher_collection_authorized": False,
            "training_eligible": False,
            "r1_diagnostic_replay_count": 1,
            "r1_nominal_additional_compression_sweep_mm": "5.0",
            "r1_release_height_offset_m": 0.0,
            "r1_scene_support_from_task": True,
            "r1_diagnostic_free_joint_unfold": None,
            "r1_diagnostic_phase_time_dilation": (
                None
            ),
            "r1_diagnostic_late_approach_clearance": None,
            "r1_diagnostic_two_stage_approach": two_stage,
            "jobs": [
                {
                    "name": materialized.manifest.candidate_id,
                    "argv": argv,
                    "log_path": str(log_path),
                }
            ],
        },
    )
    return {
        "heuristic": heuristic,
        "candidate_id": materialized.manifest.candidate_id,
        "case_root": root,
        "job_path": job_path,
        "replay_root": replay_root,
        "command": [
            isaac_python,
            str(WRAPPER),
            "--r1-batched-jobs",
            str(job_path),
            "--r1-batched-output-root",
            str(replay_root),
            "--no-formal-annotation",
        ],
    }


def _run_job(job: dict[str, Any]) -> int:
    episode = (
        job["replay_root"] / job["candidate_id"] / "isaac/evaluation_episodes.jsonl"
    )
    if episode.is_file():
        return 0
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    environment["TORCHINDUCTOR_COMPILE_THREADS"] = "8"
    log = job["job_path"].with_suffix(".launcher.log")
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            job["command"],
            cwd=REPOSITORY,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(completed.returncode)


def _result(job: dict[str, Any]) -> dict[str, Any]:
    episode_path = (
        job["replay_root"] / job["candidate_id"] / "isaac/evaluation_episodes.jsonl"
    )
    raw_path = episode_path.with_name("evaluation_rollout.pt")
    records = [
        json.loads(line)
        for line in episode_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    metadata = raw.get("metadata", {})
    accepted = (
        len(records) == 1
        and records[0].get("task_success") is True
        and records[0].get("safety_failure") is False
        and int(records[0].get("fallback_decision_count", -1)) == 0
        and float(records[0].get("metrics", {}).get("hard_collision", -1.0)) == 0.0
        and metadata.get("pi_l_actor_command_applied") is False
        and metadata.get("r1_scene_support_from_task") is True
    )
    return {
        "heuristic_id": job["heuristic"].heuristic_id,
        "structural_hash_prefix": job["heuristic"].structural_hash_prefix,
        "sample_index": job["heuristic"].sample_index,
        "candidate_id": job["candidate_id"],
        "status": "accepted" if accepted else "rejected",
        "episode": records[0] if len(records) == 1 else None,
        "episodes": _binding(episode_path),
        "raw_rollout": _binding(raw_path),
        "pi_l_actor_command_applied": metadata.get("pi_l_actor_command_applied"),
        "r1_scene_support_from_task": metadata.get("r1_scene_support_from_task"),
    }


def main() -> int:
    arguments = _parser().parse_args()
    if (
        os.environ.get("PYTHONHASHSEED") != "0"
        or arguments.rollout_steps < 1
        or not 1 <= arguments.maximum_parallel_isaac <= 4
    ):
        raise ValueError("R1 v17 runtime contract differs")
    v16._load_contract()
    v16.runtime_v13._install_v13_preparation_hooks()
    output_root = Path(arguments.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    heuristics = order9_r1_collision_avoidance_heuristics_v17()
    pairs, references, targets = v16._pairs_and_cases()
    pair_by_prefix = {pair.held_out_entry.structural_hash[:12]: pair for pair in pairs}
    jobs = []
    reference_cache: dict[tuple[object, ...], Path] = {}
    for key, heuristic in heuristics.items():
        pair = pair_by_prefix.get(heuristic.structural_hash_prefix)
        if pair is None:
            raise SchemaValidationError("R1 v17 held-out morphology is absent")
        reference_source = references[pair.bucket_id]
        target_matches = [
            value
            for value in targets[pair.bucket_id]
            if value.sample_kind == "lattice"
            and value.sample_index == heuristic.sample_index
        ]
        if len(target_matches) != 1:
            raise SchemaValidationError("R1 v17 target condition is absent")
        rule_root = output_root / heuristic.heuristic_id
        reference_key = (
            heuristic.structural_hash_prefix,
            *heuristic.teacher_option,
            heuristic.approach_time_scale,
            heuristic.free_joint_targets_rad,
        )
        reference = reference_cache.get(reference_key)
        if reference is None:
            reference = _prepare_reference(
                pair=pair,
                source_case=reference_source,
                heuristic=heuristic,
                rule_root=rule_root,
            )
            reference_cache[reference_key] = reference
        target = _prepare_target(
            pair=pair,
            source_case=target_matches[0],
            heuristic=heuristic,
            reference=reference,
            rule_root=rule_root,
        )
        jobs.append(
            _job(
                root=target,
                heuristic=heuristic,
                isaac_python=arguments.isaac_python,
                rollout_steps=arguments.rollout_steps,
            )
        )
        print(
            "ORDER9_R1_COLLISION_HEURISTIC_PREPARED="
            + json.dumps({"key": key, "heuristic_id": heuristic.heuristic_id}),
            flush=True,
        )
    preparation = {
        "result_version": ORDER9_R1_COLLISION_AVOIDANCE_V17_VERSION,
        "status": "accepted",
        "condition_count": len(jobs),
        "heuristics": [job["heuristic"].__dict__ for job in jobs],
        "lightweight_screen_passed_count": len(jobs),
        "acceptance_or_safety_gate_changed": False,
        "pi_l_actor_command_applied": False,
        "training_eligible": False,
        "formal_teacher_collection_authorized": False,
    }
    _write_json(output_root / "preparation_result.json", preparation)
    if arguments.prepare_only:
        print(json.dumps(preparation, sort_keys=True), flush=True)
        return 0

    with ThreadPoolExecutor(max_workers=arguments.maximum_parallel_isaac) as executor:
        futures = {executor.submit(_run_job, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            returncode = future.result()
            print(
                "ORDER9_R1_COLLISION_HEURISTIC_ISAAC="
                + json.dumps(
                    {
                        "heuristic_id": job["heuristic"].heuristic_id,
                        "returncode": returncode,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if returncode != 0:
                raise RuntimeError("R1 v17 Isaac subprocess failed")
    results = [_result(job) for job in jobs]
    payload = {
        "result_version": ORDER9_R1_COLLISION_AVOIDANCE_V17_VERSION,
        "status": (
            "accepted"
            if all(value["status"] == "accepted" for value in results)
            else "rejected"
        ),
        "condition_count": len(results),
        "success_count": sum(value["status"] == "accepted" for value in results),
        "hard_collision_count": sum(
            float(
                (value.get("episode") or {})
                .get("metrics", {})
                .get("hard_collision", 0.0)
            )
            for value in results
        ),
        "safety_failure_count": sum(
            (value.get("episode") or {}).get("safety_failure") is True
            for value in results
        ),
        "results": results,
        "acceptance_or_safety_gate_changed": False,
        "training_eligible": False,
        "formal_teacher_collection_authorized": False,
    }
    _write_json(output_root / "result.json", payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0 if payload["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
