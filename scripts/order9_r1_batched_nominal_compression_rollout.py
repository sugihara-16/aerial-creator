#!/usr/bin/env python3
from __future__ import annotations

"""Replay one R1 source bucket as environment-wise nominal trajectories."""

from copy import deepcopy
from dataclasses import replace
import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any, Sequence

import torch

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.schemas.common import SchemaValidationError  # noqa: E402
from amsrr.schemas.task_spec import TaskSpec  # noqa: E402
from amsrr.simulation.order9_object_task_runtime import (  # noqa: E402
    Order9ObjectTaskPhase,
)
from amsrr.simulation.order9_tensor_object_task import (  # noqa: E402
    ORDER9_CONTACT_SCHEDULE_APPROACH,
    ORDER9_CONTACT_SCHEDULE_ATTACH,
    ORDER9_CONTACT_SCHEDULE_MAINTAIN,
    ORDER9_CONTACT_SCHEDULE_RELEASE,
)
from amsrr.training.order9_r1_batched_nominal_runtime import (  # noqa: E402
    Order9R1BatchedNominalTensorReference,
    Order9R1BatchedWrenchRangeReference,
    configure_order9_r1_batched_nominal_runtime,
    order9_r1_batched_task_for_environment,
)
from amsrr.training.order9_r1_isaac_calibration import (  # noqa: E402
    load_order9_r1_isaac_case,
)
from amsrr.training.order9_r1_nominal_calibration import (  # noqa: E402
    annotate_order9_r1_nominal_isaac_result,
)
from amsrr.training.order9_r1_nominal_calibration_v7 import (  # noqa: E402
    validate_order9_r1_nominal_v7_isaac_result,
)
from amsrr.training.order9_r1_nominal_retime import _retime_trajectory  # noqa: E402
from amsrr.training.order9_r1_release_clearance_repair import (  # noqa: E402
    apply_order9_r1_release_clearance_repair,
)
from scripts.order9_r1_nominal_compression_sweep_rollout import (  # noqa: E402
    _annotate as annotate_compression,
    _compression_scale,
)
from amsrr.training.order9_tensor_pi_l_runtime import (  # noqa: E402
    Order9TensorPiLRuntime,
)
from amsrr.utils.hashing import hash_file  # noqa: E402

PROTECTED_ROLLOUT = REPOSITORY / "scripts/order9_vectorized_isaac_rollout.py"
PROTECTED_ROLLOUT_SHA256 = (
    "e813f950dde32818b7375949bfb58baa983dde760e58da22cfbb79cc33b764c2"
)
BATCH_VERSION = "order9_r1_environment_wise_nominal_isaac_batch_v3_bounded_four"
REPLAY_COUNT = 2
# Same-scene replays are limited to four candidates. Larger batches changed
# contact outcomes even when collision-filtered clones were packed near the
# origin. Four-candidate batches retain the protected rollout's normal 3 m
# separation and were checked against isolated two-replay execution.
MAXIMUM_CANDIDATES_PER_SCENE = 4
BATCH_ENV_SPACING_M = 3.0
BATCH_RUNTIME = REPOSITORY / "amsrr/training/order9_r1_batched_nominal_runtime.py"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1-batched-jobs", required=True)
    parser.add_argument("--r1-batched-output-root")
    parser.add_argument("--r1-batched-offset", type=int, default=0)
    parser.add_argument("--r1-batched-limit", type=int)
    parser.add_argument("--no-formal-annotation", action="store_true")
    return parser


def _value(arguments: Sequence[str], option: str) -> str:
    indices = [index for index, value in enumerate(arguments) if value == option]
    if len(indices) != 1 or indices[0] + 1 >= len(arguments):
        raise SchemaValidationError(f"R1 batched job lacks exactly one {option}")
    return str(arguments[indices[0] + 1])


def _replace_value(arguments: list[str], option: str, value: str) -> None:
    index = arguments.index(option)
    arguments[index + 1] = value


def _configure_batched_environment_spacing(arguments: list[str]) -> None:
    value = str(BATCH_ENV_SPACING_M)
    if "--env-spacing" in arguments:
        _replace_value(arguments, "--env-spacing", value)
    else:
        arguments.extend(("--env-spacing", value))


def _load_jobs(
    path: Path, *, offset: int, limit: int | None
) -> tuple[dict[str, Any], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    jobs = payload.get("jobs") if isinstance(payload, dict) else None
    if not isinstance(jobs, list) or not jobs:
        raise SchemaValidationError("R1 batched job manifest is invalid")
    if not 0 <= offset < len(jobs):
        raise ValueError("R1 batched diagnostic offset is invalid")
    jobs = jobs[offset:]
    if limit is not None:
        if not 1 <= limit <= len(jobs):
            raise ValueError("R1 batched diagnostic limit is invalid")
        jobs = jobs[:limit]
    normalized = []
    names: set[str] = set()
    for job in jobs:
        argv = job.get("argv") if isinstance(job, dict) else None
        name = job.get("name") if isinstance(job, dict) else None
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or not isinstance(argv, list)
            or not argv
            or any(not isinstance(value, str) for value in argv)
        ):
            raise SchemaValidationError("R1 batched job entry is invalid")
        names.add(name)
        normalized.append({**job, "name": name, "argv": list(argv)})
    if len(normalized) > MAXIMUM_CANDIDATES_PER_SCENE:
        raise SchemaValidationError(
            "R1 batched job count exceeds the verified four-candidate limit"
        )
    return tuple(normalized)


def _validate_common_job_contract(jobs: Sequence[dict[str, Any]]) -> None:
    common = (
        "--config",
        "--stage",
        "--pi-l-checkpoint",
        "--pi-l-checkpoint-sha256",
        "--morphology-graph-json",
        "--robot-usd",
        "--selected-gripper-friction",
        "--contact-stiffness",
        "--contact-damping",
        "--c3-action-contract",
        "--rollout-steps",
    )
    first = jobs[0]["argv"]
    expected = {option: _value(first, option) for option in common}
    expected_vectors = {
        "--estimated-inertia-body": tuple(
            first[
                first.index("--estimated-inertia-body")
                + 1 : first.index("--estimated-inertia-body")
                + 7
            ]
        ),
        "--estimated-com-object": tuple(
            first[
                first.index("--estimated-com-object")
                + 1 : first.index("--estimated-com-object")
                + 4
            ]
        ),
    }
    for job in jobs:
        argv = job["argv"]
        if any(_value(argv, option) != value for option, value in expected.items()):
            raise SchemaValidationError("R1 batched jobs do not share one morphology")
        if _value(argv, "--num-envs") != str(REPLAY_COUNT) or _value(
            argv, "--evaluation-episode-count"
        ) != str(REPLAY_COUNT):
            raise SchemaValidationError("R1 batched job replay count differs")
        for option, values in expected_vectors.items():
            start = argv.index(option) + 1
            if tuple(argv[start : start + len(values)]) != values:
                raise SchemaValidationError("R1 batched physical estimates differ")
        required_flags = {
            "--formal-phase-zero-start",
            "--diagnostic-nominal-qpid-only",
            "--no-tensorboard",
        }
        if not required_flags.issubset(argv):
            raise SchemaValidationError("R1 batched formal flags differ")


def _load_inputs(
    namespace: dict[str, Any],
    jobs: Sequence[dict[str, Any]],
    *,
    release_height_offset_m: float,
):
    config_path = (REPOSITORY / namespace["args_cli"].config).resolve()
    config = namespace["load_order9_learning_config"](config_path)
    physical = namespace["build_physical_model_from_config"](
        REPOSITORY / config.production_runtime.robot_model_config_path
    )
    tasks = []
    bundles = []
    wrench_trajectories = []
    source_bucket_ids = set()
    for job in jobs:
        argv = job["argv"]
        task_path = Path(_value(argv, "--task-spec-json")).resolve()
        task = TaskSpec.from_json(task_path.read_text(encoding="utf-8"))
        task.validate()
        bucket_id = task.metadata.get("order9_rollout_bucket_id")
        if not isinstance(bucket_id, str) or not bucket_id:
            raise SchemaValidationError("R1 batched task lacks bucket identity")
        source_bucket_id = json.loads(
            (
                Path(_value(argv, "--output-raw")).parents[1] / "case_manifest.json"
            ).read_text(encoding="utf-8")
        ).get("source_bucket_id")
        if not isinstance(source_bucket_id, str):
            raise SchemaValidationError("R1 batched case lacks source bucket")
        source_bucket_ids.add(source_bucket_id)
        bundle = namespace["load_order9_c3_accepted_nominal_bundle"](
            Path(_value(argv, "--c3-nominal-set-manifest")),
            bucket_id=bucket_id,
            repository_root=REPOSITORY,
            expected_set_sha256=_value(argv, "--c3-nominal-set-sha256"),
            expected_artifact_sha256=_value(argv, "--c3-nominal-artifact-sha256"),
            expected_task_spec_sha256=hash_file(task_path),
            expected_physical_model_hash=physical.stable_hash(),
        )
        if release_height_offset_m > 0.0:
            bundle = replace(
                bundle,
                phase_trajectories=apply_order9_r1_release_clearance_repair(
                    bundle.phase_trajectories,
                    height_offset_m=release_height_offset_m,
                ),
            )
        tasks.append(task)
        bundles.append(bundle)
        wrench_trajectories.append(
            namespace["calibrate_order9_contact_wrench_moment_envelopes"](
                bundle.trajectory, bundle.contact_candidate_set
            )
        )
    if len(source_bucket_ids) != 1:
        raise SchemaValidationError("R1 batched jobs span multiple source buckets")
    return (
        config,
        physical,
        tuple(tasks),
        tuple(bundles),
        tuple(wrench_trajectories),
        next(iter(source_bucket_ids)),
    )


def _candidate_virtual_contact_delta(
    namespace: dict[str, Any],
    *,
    config,
    physical,
    task: TaskSpec,
    bundle,
    argv: Sequence[str],
    module_ids: Sequence[int],
    joint_ids: Sequence[str],
) -> torch.Tensor:
    target, _geometry = namespace["_target_object_and_geometry"](task)
    contact = bundle.phase_trajectories[Order9ObjectTaskPhase.CONTACT_ACQUISITION.value]
    base_lead = float(config.production_runtime.c3_virtual_contact_inward_lead_m)
    actuator_solution = None
    if config.production_runtime.c3_actuator_aware_nominal_preload_enabled:
        actuator_solution = namespace["solve_order9_actuator_aware_nominal_preload"](
            morphology=bundle.morphology,
            physical_model=physical,
            contact_knot=contact.knots[-1],
            candidate_set=bundle.contact_candidate_set,
            object_mass_kg=float(target.mass_kg),
            contact_friction=float(_value(argv, "--selected-gripper-friction")),
            contact_stiffness_n_per_m=float(_value(argv, "--contact-stiffness")),
            config=namespace["Order9ActuatorAwareNominalPreloadConfig"](
                minimum_inward_lead_m=base_lead,
                maximum_inward_lead_m=float(
                    config.production_runtime.c3_actuator_aware_nominal_preload_maximum_m
                ),
                inward_lead_quantization_m=float(
                    config.production_runtime.c3_actuator_aware_nominal_preload_quantization_m
                ),
                support_safety_factor=float(
                    config.production_runtime.c3_actuator_aware_nominal_preload_support_safety_factor
                ),
                maximum_peak_effort_utilization=float(
                    config.production_runtime.c3_actuator_aware_nominal_preload_maximum_peak_utilization
                ),
                minimum_anchor_force_fraction=float(
                    config.production_runtime.c3_actuator_aware_nominal_preload_minimum_anchor_fraction
                ),
                model_error_margin_m=float(
                    config.production_runtime.c3_actuator_aware_nominal_preload_model_error_margin_m
                ),
            ),
        )
        if not actuator_solution.feasible:
            raise RuntimeError("R1 batched actuator-aware nominal preload failed")
        base_lead = float(actuator_solution.inward_lead_m)
    if actuator_solution is None:
        solution = namespace["solve_order9_virtual_contact_compression"](
            morphology=bundle.morphology,
            physical_model=physical,
            contact_knot=contact.knots[-1],
            candidate_set=bundle.contact_candidate_set,
            config=namespace["Order9VirtualContactCompressionConfig"](
                inward_lead_m=base_lead
            ),
        )
    else:
        solution = namespace[
            "solve_order9_virtual_contact_compression_for_achieved_lead"
        ](
            morphology=bundle.morphology,
            physical_model=physical,
            contact_knot=contact.knots[-1],
            candidate_set=bundle.contact_candidate_set,
            minimum_achieved_inward_lead_m=base_lead,
            maximum_requested_inward_lead_m=float(
                config.production_runtime.c3_actuator_aware_nominal_preload_maximum_m
            ),
            requested_lead_quantization_m=float(
                config.production_runtime.c3_actuator_aware_nominal_preload_quantization_m
            ),
        )
    delta = namespace["compression_joint_delta_tensor"](
        solution,
        module_ids=tuple(module_ids),
        local_joint_ids=tuple(joint_ids),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    if actuator_solution is not None:
        collision_object = namespace["build_order9_c3_posture_collision_object"](task)
        resolver = namespace["Order9PostureTrajectoryResolver"](
            physical,
            config=namespace["Order9PostureResolverConfig"](collision_margin_m=0.001),
            collision_object=collision_object,
            prefer_native_solver=True,
            require_native_solver=True,
        )
        collision_limit = namespace[
            "limit_order9_virtual_contact_compression_by_collision"
        ](
            morphology=bundle.morphology,
            physical_model=physical,
            contact_knot=contact.knots[-1],
            solution=solution,
            collision_object=collision_object,
            collision_solver=resolver.ik_solver,
            collision_margin_m=0.001,
            maximum_action_joint_delta_rad=0.15,
        )
        scale = float(collision_limit.maximum_action_scale)
        applied = scale * min(
            float(value) for value in solution.achieved_inward_displacement_m.values()
        )
        minimum = max(
            float(config.production_runtime.c3_virtual_contact_inward_lead_m),
            float(actuator_solution.compliance_predicted_inward_lead_m),
        )
        if applied + 1.0e-9 < minimum:
            raise RuntimeError("R1 batched nominal preload collision admission failed")
        delta = delta * scale
    return delta


def _candidate_contact_basis(
    namespace: dict[str, Any],
    *,
    config,
    physical,
    bundle,
    module_ids: Sequence[int],
    joint_ids: Sequence[str],
):
    contact = bundle.phase_trajectories[Order9ObjectTaskPhase.CONTACT_ACQUISITION.value]
    return namespace["build_order9_contact_space_action_basis"](
        morphology=bundle.morphology,
        physical_model=physical,
        contact_knot=contact.knots[-1],
        candidate_set=bundle.contact_candidate_set,
        module_ids=tuple(module_ids),
        local_joint_ids=tuple(joint_ids),
        config=namespace["Order9ContactSpaceActionConfig"](
            normal_translation_quantization_step_m=(
                config.optimization.c3_boundary_fine_tune.contact_normal_action_quantization_step_m
            )
        ),
    )


def _install_batched_formal_start(namespace: dict[str, Any]) -> None:
    def restore(scene, *, sim, io, qpid_runtime, nominal_reference):
        robot = scene["robot"]
        obj = scene["object"]
        env_ids = torch.arange(scene.num_envs, device=scene.device, dtype=torch.long)
        reference = nominal_reference.batched_phase_start_reference(
            Order9ObjectTaskPhase.APPROACH
        )
        if tuple(nominal_reference.module_ids) != tuple(io.module_ids):
            raise RuntimeError("R1 batched formal module identity differs")
        scene.reset(env_ids)
        q = namespace["_torch"](robot.data.default_joint_pos).clone()
        qdot = torch.zeros_like(q)
        reference_q = reference["joint_positions_rad"].to(
            device=scene.device, dtype=q.dtype
        )
        reference_qdot = reference["joint_velocities_radps"].to(
            device=scene.device, dtype=q.dtype
        )
        local_index = {
            joint_id: index for index, joint_id in enumerate(io.local_joint_ids)
        }
        for module_row, _module_id in enumerate(io.module_ids):
            for reference_joint, joint_id in enumerate(nominal_reference.joint_ids):
                robot_index = io.local_joint_indices[module_row][local_index[joint_id]]
                if robot_index < 0:
                    raise RuntimeError("R1 batched formal joint was merged")
                q[:, robot_index] = reference_q[:, module_row, reference_joint]
                qdot[:, robot_index] = reference_qdot[:, module_row, reference_joint]
        robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
        robot.write_joint_velocity_to_sim_index(velocity=qdot, env_ids=env_ids)
        sim.forward()
        scene.update(0.0)
        state = io.gather_state(robot=robot, object_asset=obj)
        control = qpid_runtime.builder.build(
            module_pose_world=state.module_pose_world,
            module_twist_world=state.module_twist_world,
            local_joint_positions_rad=state.local_joint_positions_rad,
        )
        current_root = namespace["_torch"](robot.data.root_pose_w)
        root_pose_world = current_root.clone()
        desired_body_world = reference["body_pose_local"].to(
            device=scene.device, dtype=q.dtype
        )
        desired_body_world[:, :3] += scene.env_origins
        for env_id in range(scene.num_envs):
            root_to_body = namespace["compose_pose"](
                namespace["inverse_pose"](
                    tuple(float(value) for value in current_root[env_id].tolist())
                ),
                tuple(
                    float(value) for value in control.body_pose_world[env_id].tolist()
                ),
            )
            root_pose_world[env_id] = torch.tensor(
                namespace["compose_pose"](
                    tuple(
                        float(value) for value in desired_body_world[env_id].tolist()
                    ),
                    namespace["inverse_pose"](root_to_body),
                ),
                device=scene.device,
                dtype=q.dtype,
            )
        body_twist = reference["body_twist"].to(device=scene.device, dtype=q.dtype)
        object_pose_world = reference["object_pose_local"].to(
            device=scene.device, dtype=q.dtype
        )
        object_pose_world[:, :3] += scene.env_origins
        object_twist = reference["object_twist"].to(device=scene.device, dtype=q.dtype)
        robot.write_root_pose_to_sim_index(root_pose=root_pose_world, env_ids=env_ids)
        robot.write_root_velocity_to_sim_index(
            root_velocity=body_twist, env_ids=env_ids
        )
        robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
        robot.write_joint_velocity_to_sim_index(velocity=qdot, env_ids=env_ids)
        robot.set_joint_position_target_index(target=q, env_ids=env_ids)
        robot.set_joint_velocity_target_index(
            target=torch.zeros_like(qdot), env_ids=env_ids
        )
        robot.set_joint_effort_target_index(
            target=torch.zeros_like(qdot), env_ids=env_ids
        )
        obj.write_root_pose_to_sim_index(root_pose=object_pose_world, env_ids=env_ids)
        obj.write_root_velocity_to_sim_index(
            root_velocity=object_twist, env_ids=env_ids
        )
        expected = {
            "body_pose_world": desired_body_world,
            "object_pose_world": object_pose_world,
            "joint_positions_rad": reference_q,
        }
        return (
            torch.zeros(scene.num_envs, device=scene.device, dtype=q.dtype),
            desired_body_world,
            object_pose_world,
            expected,
        )

    def validate(state, *, control, expected, scene_origins, io, reference_joint_ids):
        del scene_origins
        indices = torch.tensor(
            [io.local_joint_ids.index(value) for value in reference_joint_ids],
            device=state.local_joint_positions_rad.device,
            dtype=torch.long,
        )
        actual_joint = state.local_joint_positions_rad.index_select(2, indices)
        maxima = {
            "body_position_m": float(
                torch.linalg.vector_norm(
                    control.body_pose_world[:, :3] - expected["body_pose_world"][:, :3],
                    dim=-1,
                )
                .max()
                .item()
            ),
            "body_orientation_rad": float(
                namespace["_quaternion_distance_rad"](
                    control.body_pose_world[:, 3:7],
                    expected["body_pose_world"][:, 3:7],
                )
                .max()
                .item()
            ),
            "object_position_m": float(
                torch.linalg.vector_norm(
                    state.object_pose_world[:, :3]
                    - expected["object_pose_world"][:, :3],
                    dim=-1,
                )
                .max()
                .item()
            ),
            "object_orientation_rad": float(
                namespace["_quaternion_distance_rad"](
                    state.object_pose_world[:, 3:7],
                    expected["object_pose_world"][:, 3:7],
                )
                .max()
                .item()
            ),
            "joint_position_rad": float(
                (actual_joint - expected["joint_positions_rad"]).abs().max().item()
            ),
        }
        if (
            maxima["body_position_m"] > 2.0e-4
            or maxima["body_orientation_rad"] > 2.0e-4
            or maxima["object_position_m"] > 1.0e-4
            or maxima["object_orientation_rad"] > 1.0e-4
            or maxima["joint_position_rad"] > 1.0e-5
        ):
            raise RuntimeError(f"R1 batched formal installation differs: {maxima}")

    namespace["_restore_c3_formal_phase_zero"] = restore
    namespace["_validate_c3_formal_phase_zero_alignment"] = validate


def _atomic_torch_save(payload: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text_write(destination: Path, value: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _split_outputs(
    *,
    combined_raw: Path,
    combined_episodes: Path,
    jobs: Sequence[dict[str, Any]],
    tasks: Sequence[TaskSpec],
    bundles: Sequence[Any],
    wrench_trajectories: Sequence[Any],
    output_root: Path | None,
    additional_compression_mm: float,
    release_height_offset_m: float,
    scene_support_from_task: bool,
    lift_time_dilation: float | None,
    grasp_centering: dict[str, Any] | None,
    grasp_torque_bias_fraction: float | None,
    contact_compression_by_slot_mm: tuple[float, ...] | None,
    maintain_vertical_scale: float | None,
    formal_teacher_collection: bool = False,
) -> tuple[tuple[Path, Path], ...]:
    payload = torch.load(combined_raw, map_location="cpu", weights_only=False)
    tensors = payload.get("tensors") if isinstance(payload, dict) else None
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    records = [
        json.loads(line)
        for line in combined_episodes.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    environment_count = len(jobs) * REPLAY_COUNT
    if (
        not isinstance(tensors, dict)
        or not isinstance(metadata, dict)
        or len(records) != environment_count
        or any(
            not isinstance(value, torch.Tensor)
            or value.ndim < 2
            or value.shape[1] != environment_count
            for value in tensors.values()
        )
    ):
        raise SchemaValidationError("R1 batched combined output is invalid")
    records_by_environment = {
        int(record.get("metadata", {}).get("environment_index", -1)): record
        for record in records
    }
    if set(records_by_environment) != set(range(environment_count)):
        raise SchemaValidationError("R1 batched episode environments differ")
    outputs = []
    for case_index, (job, task, bundle, trajectory) in enumerate(
        zip(jobs, tasks, bundles, wrench_trajectories)
    ):
        argv = job["argv"]
        generation = _value(argv, "--generation-id")
        if output_root is None:
            raw_path = Path(_value(argv, "--output-raw")).resolve()
            episode_path = Path(_value(argv, "--evaluation-jsonl")).resolve()
            log_path = Path(str(job["log_path"])).resolve()
        else:
            case_root = output_root / str(job["name"]) / "isaac"
            raw_path = case_root / "evaluation_rollout.pt"
            episode_path = case_root / "evaluation_episodes.jsonl"
            log_path = case_root / "evaluation.log"
        start = case_index * REPLAY_COUNT
        stop = start + REPLAY_COUNT
        case_metadata = deepcopy(metadata)
        case_metadata["generation_id"] = generation
        case_metadata["random_seed"] = int(_value(argv, "--seed"))
        for key in (
            "task_specs",
            "environment_splits",
            "assignment_templates_by_environment",
        ):
            values = metadata.get(key)
            if isinstance(values, list) and len(values) == environment_count:
                case_metadata[key] = deepcopy(values[start:stop])
        assignments = namespace_assignments(trajectory)
        case_metadata["assignment_templates_by_environment"] = [
            [assignment.to_dict() for assignment in assignments]
            for _ in range(REPLAY_COUNT)
        ]
        case_metadata["active_knot_trajectory_template"] = trajectory.to_dict()
        case_metadata["c3_nominal_reference"] = {
            **dict(bundle.provenance),
            "runtime_version": Order9R1BatchedNominalTensorReference.runtime_version,
        }
        candidate_reset_bank = Path(_value(argv, "--c3-reset-bank")).resolve()
        case_metadata.update(
            {
                "r1_batched_nominal_runtime_version": BATCH_VERSION,
                "r1_batched_nominal_wrapper_sha256": hash_file(Path(__file__)),
                "r1_batched_nominal_runtime_sha256": hash_file(BATCH_RUNTIME),
                "r1_batched_candidate_count": len(jobs),
                "r1_batched_replay_count": REPLAY_COUNT,
                "r1_batched_environment_range": [start, stop],
                "r1_batched_physics_translation_invariant": False,
                "r1_batched_environment_origin_packed": False,
                "r1_batched_environment_spacing_m": BATCH_ENV_SPACING_M,
                "r1_batched_maximum_candidates_per_scene": (
                    MAXIMUM_CANDIDATES_PER_SCENE
                ),
                "r1_candidate_reset_bank_path": str(candidate_reset_bank),
                "r1_candidate_reset_bank_available": candidate_reset_bank.is_file(),
                "r1_candidate_reset_bank_used_for_formal_start": False,
                "pi_l_actor_command_applied": False,
                "promotion_evidence_eligible": False,
                "training_eligible": False,
                "formal_teacher_collection_authorized": formal_teacher_collection,
                "r1_formal_teacher_collection_candidate": formal_teacher_collection,
                "r1_release_height_offset_m": release_height_offset_m,
                "r1_scene_support_from_task": scene_support_from_task,
                "r1_diagnostic_lift_time_dilation": lift_time_dilation,
                "r1_diagnostic_grasp_centering": grasp_centering,
                "r1_diagnostic_grasp_torque_bias_fraction": (
                    grasp_torque_bias_fraction
                ),
                "r1_diagnostic_contact_compression_by_slot_mm": (
                    None
                    if contact_compression_by_slot_mm is None
                    else list(contact_compression_by_slot_mm)
                ),
                "r1_diagnostic_maintain_vertical_scale": maintain_vertical_scale,
            }
        )
        case_payload = {
            **{
                key: deepcopy(value)
                for key, value in payload.items()
                if key not in {"metadata", "tensors"}
            },
            "metadata": case_metadata,
            "tensors": {
                key: value[:, start:stop].clone() for key, value in tensors.items()
            },
        }
        _atomic_torch_save(case_payload, raw_path)
        raw_sha = hash_file(raw_path)
        case_records = []
        for local_index, environment in enumerate(range(start, stop)):
            record = deepcopy(records_by_environment[environment])
            record["episode_id"] = f"{generation}:evaluation:env:{local_index:04d}"
            record["random_seed"] = int(_value(argv, "--seed")) + local_index
            record["source_artifact_path"] = str(raw_path)
            record["source_artifact_sha256"] = raw_sha
            record.setdefault("metadata", {})["generation_id"] = generation
            record["metadata"]["environment_index"] = local_index
            record["metadata"]["r1_batched_source_environment_index"] = environment
            record["metadata"]["r1_batched_nominal_runtime_version"] = BATCH_VERSION
            record["metadata"]["r1_batched_nominal_wrapper_sha256"] = hash_file(
                Path(__file__)
            )
            record["metadata"]["r1_batched_nominal_runtime_sha256"] = hash_file(
                BATCH_RUNTIME
            )
            record["metadata"]["r1_scene_support_from_task"] = scene_support_from_task
            record["metadata"][
                "formal_teacher_collection_authorized"
            ] = formal_teacher_collection
            record["metadata"][
                "r1_formal_teacher_collection_candidate"
            ] = formal_teacher_collection
            record["metadata"]["r1_diagnostic_lift_time_dilation"] = lift_time_dilation
            record["metadata"]["r1_diagnostic_grasp_centering"] = grasp_centering
            record["metadata"][
                "r1_diagnostic_grasp_torque_bias_fraction"
            ] = grasp_torque_bias_fraction
            record["metadata"]["r1_diagnostic_contact_compression_by_slot_mm"] = (
                None
                if contact_compression_by_slot_mm is None
                else list(contact_compression_by_slot_mm)
            )
            record["metadata"][
                "r1_diagnostic_maintain_vertical_scale"
            ] = maintain_vertical_scale
            case_records.append(record)
        _atomic_text_write(
            episode_path,
            "".join(
                json.dumps(record, sort_keys=True) + "\n" for record in case_records
            ),
        )
        _atomic_text_write(
            log_path,
            json.dumps(
                {
                    "runtime_version": BATCH_VERSION,
                    "candidate_id": job["name"],
                    "combined_raw_sha256": hash_file(combined_raw),
                    "combined_episode_sha256": hash_file(combined_episodes),
                    "environment_range": [start, stop],
                    "pi_l_actor_command_applied": False,
                },
                sort_keys=True,
            )
            + "\n",
        )
        outputs.append((raw_path, episode_path))
    return tuple(outputs)


def namespace_assignments(trajectory):
    maintained = []
    for knot in trajectory.knots:
        for assignment in knot.contact_assignments:
            schedule = assignment.schedule_state
            if getattr(schedule, "value", schedule) == "maintain":
                maintained.append(assignment)
    by_slot = {assignment.slot_id: assignment for assignment in maintained}
    return tuple(by_slot[index] for index in sorted(by_slot))


def _diagnostic_grasp_centering_target(
    target,
    *,
    profile: dict[str, Any] | None,
):
    """Apply a bounded whole-grasp offset while object contact is active."""

    if profile is None:
        return target
    if set(profile) != {"translation_world_m"}:
        raise SchemaValidationError("R1 grasp-centering profile keys differ")
    raw_translation = profile["translation_world_m"]
    if not isinstance(raw_translation, list) or len(raw_translation) != 3:
        raise SchemaValidationError("R1 grasp-centering translation is invalid")
    translation = torch.tensor(
        [float(value) for value in raw_translation],
        device=target.desired_robot_root_pose_world.device,
        dtype=target.desired_robot_root_pose_world.dtype,
    )
    if (
        not bool(torch.isfinite(translation).all())
        or not 0.0 < float(torch.linalg.vector_norm(translation)) <= 0.030 + 1.0e-7
    ):
        raise SchemaValidationError("R1 grasp-centering translation is out of range")
    progress = target.phase_progress.clamp(0.0, 1.0)
    smooth = progress.square() * (3.0 - 2.0 * progress)
    schedule = target.contact_schedule_index
    weight = torch.where(
        schedule == ORDER9_CONTACT_SCHEDULE_ATTACH,
        smooth,
        torch.where(
            schedule == ORDER9_CONTACT_SCHEDULE_MAINTAIN,
            torch.ones_like(progress),
            torch.where(
                schedule == ORDER9_CONTACT_SCHEDULE_RELEASE,
                1.0 - smooth,
                torch.zeros_like(progress),
            ),
        ),
    )
    offset = weight.unsqueeze(-1) * translation.reshape(1, 3)
    pose = target.desired_robot_root_pose_world.clone()
    goal = target.phase_goal_robot_root_pose_world.clone()
    pose[:, :3] += offset
    goal[:, :3] += offset
    return replace(
        target,
        desired_robot_root_pose_world=pose,
        phase_goal_robot_root_pose_world=goal,
    )


def _diagnostic_grasp_torque_bias(
    result,
    *,
    target,
    compression_direction: torch.Tensor,
    effort_limits_nm: torch.Tensor,
    fraction: float | None,
):
    """Hold bounded actuator effort in the same direction as grasp closure."""

    if fraction is None:
        return result
    command = result.policy_command
    if (
        not 0.0 < fraction <= 0.20
        or compression_direction.shape != command.joint_torque_bias_nm.shape
        or effort_limits_nm.shape != (command.joint_torque_bias_nm.shape[-1],)
    ):
        raise SchemaValidationError("R1 grasp torque-bias inputs differ")
    maximum = compression_direction.abs().amax(dim=(1, 2), keepdim=True)
    direction = compression_direction / maximum.clamp_min(1.0e-9)
    active = _compression_scale(target).reshape(-1, 1, 1)
    limit = effort_limits_nm.reshape(1, 1, -1) * fraction
    bias = direction * limit * active
    return replace(
        result,
        policy_command=replace(command, joint_torque_bias_nm=bias),
    )


def _contact_compression_values_by_active_slot(
    contact_slot_mask: torch.Tensor,
    active_values_mm: tuple[float, ...],
) -> torch.Tensor:
    """Expand values ordered by active contact into the fixed contact slots."""
    if contact_slot_mask.ndim != 2 or contact_slot_mask.dtype != torch.bool:
        raise SchemaValidationError("R1 contact compression mask is invalid")
    if not active_values_mm or any(
        not math.isfinite(value) or not 0.0 <= value <= 80.0
        for value in active_values_mm
    ):
        raise SchemaValidationError("R1 contact compression values are invalid")
    active_counts = contact_slot_mask.sum(dim=1)
    if torch.any(active_counts != len(active_values_mm)):
        raise SchemaValidationError(
            "R1 contact compression count differs from active contacts"
        )
    ranks = contact_slot_mask.to(dtype=torch.long).cumsum(dim=1) - 1
    ordered = torch.tensor(
        active_values_mm,
        device=contact_slot_mask.device,
        dtype=torch.float32,
    )
    expanded = ordered[ranks.clamp(min=0)]
    return torch.where(contact_slot_mask, expanded, torch.zeros_like(expanded))


def _diagnostic_free_joint_unfold_target(
    target,
    *,
    profile: dict[str, Any] | None,
    module_ids: Sequence[int],
    joint_ids: Sequence[str],
):
    """Return an R1-only target that unfolds one unused Dock after clearance.

    The override is intentionally restricted by ``main`` to isolated,
    one-replay diagnostics.  It changes neither the protected C3 runner nor
    any selected grasp/structural joint.
    """

    if profile is None:
        return target
    required = {
        "global_joint_id",
        "approach_unfold_start_fraction",
        "approach_unfold_end_fraction",
        "final_target_rad",
    }
    if set(profile) != required:
        raise SchemaValidationError("R1 free-joint unfold profile keys differ")
    global_id = profile["global_joint_id"]
    if not isinstance(global_id, str) or ":" not in global_id:
        raise SchemaValidationError("R1 free-joint unfold joint id is invalid")
    module_text, local_joint_id = global_id.split(":", 1)
    if not module_text.startswith("module_"):
        raise SchemaValidationError("R1 free-joint unfold module id is invalid")
    try:
        module_index = tuple(module_ids).index(int(module_text[7:]))
        joint_index = tuple(joint_ids).index(local_joint_id)
    except (ValueError, TypeError) as error:
        raise SchemaValidationError(
            "R1 free-joint unfold target is absent from the runtime"
        ) from error
    start = float(profile["approach_unfold_start_fraction"])
    end = float(profile["approach_unfold_end_fraction"])
    final = float(profile["final_target_rad"])
    if not (0.0 < start < end < 1.0) or not torch.isfinite(torch.tensor(final)):
        raise SchemaValidationError("R1 free-joint unfold interval is invalid")

    progress = target.phase_progress.clamp(0.0, 1.0)
    normalized = ((progress - start) / (end - start)).clamp(0.0, 1.0)
    smooth = normalized.square() * (3.0 - 2.0 * normalized)
    approach = target.contact_schedule_index == ORDER9_CONTACT_SCHEDULE_APPROACH
    weight = torch.where(approach, smooth, torch.ones_like(smooth))
    positions = target.nominal_joint_positions_rad.clone()
    current = positions[:, module_index, joint_index]
    positions[:, module_index, joint_index] = current + weight * (final - current)
    velocities = target.nominal_joint_velocities_radps.clone()
    # QPID nominal hold deliberately commands zero reference velocity.  Keep
    # this component zero as well so the override cannot inject a feedforward
    # speed outside the local-servo contract.
    velocities[:, module_index, joint_index] = 0.0
    return replace(
        target,
        nominal_joint_positions_rad=positions,
        nominal_joint_velocities_radps=velocities,
    )


def _diagnostic_late_approach_clearance_target(
    target,
    *,
    profile: dict[str, Any] | None,
):
    """Offset only the late approach, then smoothly return to its endpoint."""

    if profile is None:
        return target
    required = {
        "translation_world_m",
        "rise_start_fraction",
        "rise_end_fraction",
        "fall_start_fraction",
        "fall_end_fraction",
    }
    if set(profile) != required:
        raise SchemaValidationError("R1 late-approach clearance keys differ")
    raw_translation = profile["translation_world_m"]
    rise_start = float(profile["rise_start_fraction"])
    rise_end = float(profile["rise_end_fraction"])
    fall_start = float(profile["fall_start_fraction"])
    fall_end = float(profile["fall_end_fraction"])
    if not isinstance(raw_translation, list) or len(raw_translation) != 3:
        raise SchemaValidationError("R1 late-approach translation is invalid")
    translation = tuple(float(value) for value in raw_translation)
    if not (
        0.0 < float(torch.linalg.vector_norm(torch.tensor(translation))) <= 0.20
        and 0.0 < rise_start < rise_end < fall_start < fall_end <= 1.0
    ):
        raise SchemaValidationError("R1 late-approach clearance profile is invalid")
    progress = target.phase_progress.clamp(0.0, 1.0)
    rise_u = ((progress - rise_start) / (rise_end - rise_start)).clamp(0.0, 1.0)
    fall_u = ((progress - fall_start) / (fall_end - fall_start)).clamp(0.0, 1.0)
    rise = rise_u.square() * (3.0 - 2.0 * rise_u)
    fall = fall_u.square() * (3.0 - 2.0 * fall_u)
    approach = target.contact_schedule_index == ORDER9_CONTACT_SCHEDULE_APPROACH
    weight = torch.where(approach, rise * (1.0 - fall), torch.zeros_like(rise))
    pose = target.desired_robot_root_pose_world.clone()
    pose[:, :3] += weight.unsqueeze(-1) * torch.tensor(
        translation, device=pose.device, dtype=pose.dtype
    ).reshape(1, 3)
    return replace(target, desired_robot_root_pose_world=pose)


def _diagnostic_joint_clearance_pulse_target(
    target,
    *,
    profile: dict[str, Any] | None,
    module_ids: Sequence[int],
    joint_ids: Sequence[str],
):
    """Temporarily move one non-grasp joint during the approach."""

    if profile is None:
        return target
    required = {
        "global_joint_id",
        "offset_rad",
        "rise_start_fraction",
        "rise_end_fraction",
        "fall_start_fraction",
        "fall_end_fraction",
    }
    if set(profile) != required:
        raise SchemaValidationError("R1 joint-clearance pulse keys differ")
    global_id = profile["global_joint_id"]
    if not isinstance(global_id, str) or ":" not in global_id:
        raise SchemaValidationError("R1 joint-clearance pulse id is invalid")
    module_text, local_joint_id = global_id.split(":", 1)
    try:
        module_index = tuple(module_ids).index(int(module_text.removeprefix("module_")))
        joint_index = tuple(joint_ids).index(local_joint_id)
    except (ValueError, TypeError) as error:
        raise SchemaValidationError(
            "R1 joint-clearance pulse target is absent"
        ) from error
    offset = float(profile["offset_rad"])
    rise_start = float(profile["rise_start_fraction"])
    rise_end = float(profile["rise_end_fraction"])
    fall_start = float(profile["fall_start_fraction"])
    fall_end = float(profile["fall_end_fraction"])
    if not (
        0.0 < abs(offset) <= 0.35
        and 0.0 < rise_start < rise_end < fall_start < fall_end <= 1.0
    ):
        raise SchemaValidationError("R1 joint-clearance pulse profile is invalid")
    progress = target.phase_progress.clamp(0.0, 1.0)
    rise_u = ((progress - rise_start) / (rise_end - rise_start)).clamp(0.0, 1.0)
    fall_u = ((progress - fall_start) / (fall_end - fall_start)).clamp(0.0, 1.0)
    rise = rise_u.square() * (3.0 - 2.0 * rise_u)
    fall = fall_u.square() * (3.0 - 2.0 * fall_u)
    approach = target.contact_schedule_index == ORDER9_CONTACT_SCHEDULE_APPROACH
    weight = torch.where(approach, rise * (1.0 - fall), torch.zeros_like(rise))
    positions = target.nominal_joint_positions_rad.clone()
    positions[:, module_index, joint_index] += weight * offset
    return replace(target, nominal_joint_positions_rad=positions)


def _diagnostic_two_stage_approach_target(
    target,
    *,
    profile: dict[str, Any] | None,
    start_joint_positions_rad: torch.Tensor | None = None,
    goal_joint_positions_rad: torch.Tensor | None = None,
):
    """Replace the approach translation by a clearance transit and descent.

    This is an isolated R1 diagnostic override.  The start is represented
    relative to the phase goal, so the same construction remains invariant to
    Isaac environment origins and planar scene translations.
    """

    if profile is None:
        return target
    required = {
        "start_position_offset_from_goal_m",
        "clearance_height_above_goal_m",
        "rise_end_fraction",
        "transit_end_fraction",
        "descent_start_fraction",
        "phase_duration_s",
        "joint_motion_start_fraction",
        "joint_motion_end_fraction",
    }
    if set(profile) != required:
        raise SchemaValidationError("R1 two-stage approach profile keys differ")
    raw_offset = profile["start_position_offset_from_goal_m"]
    clearance = float(profile["clearance_height_above_goal_m"])
    rise_end = float(profile["rise_end_fraction"])
    transit_end = float(profile["transit_end_fraction"])
    descent_start = float(profile["descent_start_fraction"])
    duration = float(profile["phase_duration_s"])
    joint_start = float(profile["joint_motion_start_fraction"])
    joint_end = float(profile["joint_motion_end_fraction"])
    if (
        not isinstance(raw_offset, list)
        or len(raw_offset) != 3
        or not all(torch.isfinite(torch.tensor(float(value))) for value in raw_offset)
        or not 0.25 <= clearance <= 0.60
        or not 0.10 <= rise_end < transit_end <= descent_start <= 0.90
        or not 10.0 <= duration <= 120.0
        or not 0.0 <= joint_start < joint_end < descent_start
        or start_joint_positions_rad is None
        or goal_joint_positions_rad is None
        or start_joint_positions_rad.shape != target.nominal_joint_positions_rad.shape
        or goal_joint_positions_rad.shape != target.nominal_joint_positions_rad.shape
    ):
        raise SchemaValidationError("R1 two-stage approach profile is invalid")

    pose = target.desired_robot_root_pose_world.clone()
    twist = target.desired_robot_root_twist_world.clone()
    goal = target.phase_goal_robot_root_pose_world[:, :3]
    offset = torch.tensor(
        tuple(float(value) for value in raw_offset),
        device=pose.device,
        dtype=pose.dtype,
    ).reshape(1, 3)
    start = goal + offset
    overhead_start = start.clone()
    overhead_goal = goal.clone()
    overhead_start[:, 2] = goal[:, 2] + clearance
    overhead_goal[:, 2] = goal[:, 2] + clearance
    progress = target.phase_progress.clamp(0.0, 1.0)

    def interpolate(
        first: torch.Tensor,
        second: torch.Tensor,
        begin: float,
        end: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u = ((progress - begin) / (end - begin)).clamp(0.0, 1.0)
        weight = u.square() * (3.0 - 2.0 * u)
        position = first + weight.unsqueeze(-1) * (second - first)
        derivative = ((6.0 * u * (1.0 - u)) / ((end - begin) * duration)).unsqueeze(
            -1
        ) * (second - first)
        return position, derivative

    rise_position, rise_velocity = interpolate(start, overhead_start, 0.0, rise_end)
    transit_position, transit_velocity = interpolate(
        overhead_start, overhead_goal, rise_end, transit_end
    )
    descent_position, descent_velocity = interpolate(
        overhead_goal, goal, descent_start, 1.0
    )
    rising = progress < rise_end
    transiting = (progress >= rise_end) & (progress < transit_end)
    waiting = (progress >= transit_end) & (progress < descent_start)
    position = torch.where(
        rising.unsqueeze(-1),
        rise_position,
        torch.where(
            transiting.unsqueeze(-1),
            transit_position,
            torch.where(waiting.unsqueeze(-1), overhead_goal, descent_position),
        ),
    )
    velocity = torch.where(
        rising.unsqueeze(-1),
        rise_velocity,
        torch.where(
            transiting.unsqueeze(-1),
            transit_velocity,
            torch.where(
                waiting.unsqueeze(-1),
                torch.zeros_like(transit_velocity),
                descent_velocity,
            ),
        ),
    )
    joint_u = ((progress - joint_start) / (joint_end - joint_start)).clamp(0.0, 1.0)
    joint_weight = joint_u.square() * (3.0 - 2.0 * joint_u)
    joint_positions = start_joint_positions_rad + joint_weight.reshape(-1, 1, 1) * (
        goal_joint_positions_rad - start_joint_positions_rad
    )
    zero_joint_velocities = torch.zeros_like(target.nominal_joint_velocities_radps)
    approach = target.contact_schedule_index == ORDER9_CONTACT_SCHEDULE_APPROACH
    pose[:, :3] = torch.where(approach.unsqueeze(-1), position, pose[:, :3])
    twist[:, :3] = torch.where(approach.unsqueeze(-1), velocity, twist[:, :3])
    positions = torch.where(
        approach.reshape(-1, 1, 1),
        joint_positions,
        target.nominal_joint_positions_rad,
    )
    joint_velocities = torch.where(
        approach.reshape(-1, 1, 1),
        zero_joint_velocities,
        target.nominal_joint_velocities_radps,
    )
    return replace(
        target,
        desired_robot_root_pose_world=pose,
        desired_robot_root_twist_world=twist,
        nominal_joint_positions_rad=positions,
        nominal_joint_velocities_radps=joint_velocities,
    )


def main() -> int:
    global REPLAY_COUNT
    args = _parser().parse_args()
    if hash_file(PROTECTED_ROLLOUT) != PROTECTED_ROLLOUT_SHA256:
        raise SchemaValidationError("protected C3 rollout bytes changed")
    manifest_payload = json.loads(
        Path(args.r1_batched_jobs).read_text(encoding="utf-8")
    )
    formal_teacher_collection = bool(
        manifest_payload.get("r1_formal_teacher_collection", False)
    )
    if formal_teacher_collection:
        authorization = manifest_payload.get(
            "r1_teacher_collection_launch_authorization"
        )
        if (
            not isinstance(authorization, dict)
            or not isinstance(authorization.get("path"), str)
            or not isinstance(authorization.get("sha256"), str)
        ):
            raise SchemaValidationError(
                "R1 formal teacher collection lacks launch authorization"
            )
        authorization_path = (REPOSITORY / authorization["path"]).resolve()
        if (
            (
                REPOSITORY != authorization_path
                and REPOSITORY not in authorization_path.parents
            )
            or not authorization_path.is_file()
            or hash_file(authorization_path) != authorization.get("sha256")
        ):
            raise SchemaValidationError(
                "R1 formal teacher collection launch authorization changed"
            )
        authorization_payload = json.loads(
            authorization_path.read_text(encoding="utf-8")
        )
        if (
            authorization_payload.get("authorization_version")
            != "order9_r1_teacher_collection_launch_v19"
            or authorization_payload.get("status") != "approved"
            or authorization_payload.get("collection_launch_authorized") is not True
            or authorization_payload.get("training_authorized") is not False
        ):
            raise SchemaValidationError(
                "R1 formal teacher collection is not explicitly authorized"
            )
        if not args.no_formal_annotation or args.r1_batched_output_root is None:
            raise SchemaValidationError(
                "R1 formal teacher collection requires isolated raw output"
            )
        REPLAY_COUNT = 1
    diagnostic_replay_count = manifest_payload.get("r1_diagnostic_replay_count")
    if diagnostic_replay_count is not None:
        if (
            diagnostic_replay_count != 1
            or not args.no_formal_annotation
            or args.r1_batched_output_root is None
        ):
            raise SchemaValidationError(
                "R1 one-replay mode is restricted to isolated diagnostic output"
            )
        REPLAY_COUNT = 1
    free_joint_unfold = manifest_payload.get("r1_diagnostic_free_joint_unfold")
    if free_joint_unfold is not None and (
        diagnostic_replay_count != 1
        or not args.no_formal_annotation
        or args.r1_batched_output_root is None
    ):
        raise SchemaValidationError(
            "R1 free-joint unfold is restricted to isolated one-replay diagnostics"
        )
    raw_time_dilation = manifest_payload.get("r1_diagnostic_phase_time_dilation")
    phase_time_dilation = None
    if raw_time_dilation is not None:
        if (
            diagnostic_replay_count != 1
            or not args.no_formal_annotation
            or args.r1_batched_output_root is None
            or not isinstance(raw_time_dilation, dict)
            or set(raw_time_dilation) != {"approach"}
        ):
            raise SchemaValidationError(
                "R1 phase-time dilation is restricted to isolated approach diagnostics"
            )
        phase_time_dilation = float(raw_time_dilation["approach"])
        if not 1.0 < phase_time_dilation <= 2.0:
            raise SchemaValidationError("R1 approach dilation is outside (1, 2]")
    raw_lift_time_dilation = manifest_payload.get("r1_diagnostic_lift_time_dilation")
    lift_time_dilation = (
        None if raw_lift_time_dilation is None else float(raw_lift_time_dilation)
    )
    if lift_time_dilation is not None and (
        diagnostic_replay_count != 1
        or not args.no_formal_annotation
        or args.r1_batched_output_root is None
        or not 1.0 < lift_time_dilation <= 10.0
    ):
        raise SchemaValidationError(
            "R1 lift dilation is restricted to isolated diagnostics in (1, 10]"
        )
    grasp_centering = manifest_payload.get("r1_diagnostic_grasp_centering")
    if grasp_centering is not None and (
        diagnostic_replay_count != 1
        or not args.no_formal_annotation
        or args.r1_batched_output_root is None
        or not isinstance(grasp_centering, dict)
    ):
        raise SchemaValidationError(
            "R1 grasp centering is restricted to isolated diagnostics"
        )
    raw_torque_fraction = manifest_payload.get(
        "r1_diagnostic_grasp_torque_bias_fraction"
    )
    grasp_torque_bias_fraction = (
        None if raw_torque_fraction is None else float(raw_torque_fraction)
    )
    if grasp_torque_bias_fraction is not None and (
        diagnostic_replay_count != 1
        or not args.no_formal_annotation
        or args.r1_batched_output_root is None
        or not 0.0 < grasp_torque_bias_fraction <= 0.20
    ):
        raise SchemaValidationError(
            "R1 grasp torque bias is restricted to isolated diagnostics"
        )
    raw_contact_compression = manifest_payload.get(
        "r1_diagnostic_contact_compression_by_slot_mm"
    )
    contact_compression_by_slot_mm = (
        None
        if raw_contact_compression is None
        else tuple(float(value) for value in raw_contact_compression)
    )
    if contact_compression_by_slot_mm is not None and (
        diagnostic_replay_count != 1
        or not args.no_formal_annotation
        or args.r1_batched_output_root is None
        or not isinstance(raw_contact_compression, list)
        or not contact_compression_by_slot_mm
        or any(
            not math.isfinite(value) or not 0.0 <= value <= 80.0
            for value in contact_compression_by_slot_mm
        )
    ):
        raise SchemaValidationError(
            "R1 contact-wise compression is restricted to isolated diagnostics"
        )
    raw_vertical_scale = manifest_payload.get("r1_diagnostic_maintain_vertical_scale")
    maintain_vertical_scale = (
        None if raw_vertical_scale is None else float(raw_vertical_scale)
    )
    if maintain_vertical_scale is not None and (
        diagnostic_replay_count != 1
        or not args.no_formal_annotation
        or args.r1_batched_output_root is None
        or not math.isfinite(maintain_vertical_scale)
        or not 0.10 <= maintain_vertical_scale < 1.0
    ):
        raise SchemaValidationError(
            "R1 maintain vertical scaling is restricted to isolated diagnostics"
        )
    late_approach_clearance = manifest_payload.get(
        "r1_diagnostic_late_approach_clearance"
    )
    if late_approach_clearance is not None and (
        diagnostic_replay_count != 1
        or not args.no_formal_annotation
        or args.r1_batched_output_root is None
        or not isinstance(late_approach_clearance, dict)
    ):
        raise SchemaValidationError(
            "R1 late-approach clearance is restricted to isolated diagnostics"
        )
    joint_clearance_pulse = manifest_payload.get("r1_diagnostic_joint_clearance_pulse")
    if joint_clearance_pulse is not None and (
        diagnostic_replay_count != 1
        or not args.no_formal_annotation
        or args.r1_batched_output_root is None
        or not isinstance(joint_clearance_pulse, dict)
    ):
        raise SchemaValidationError(
            "R1 joint-clearance pulse is restricted to isolated diagnostics"
        )
    two_stage_approach = manifest_payload.get("r1_diagnostic_two_stage_approach")
    if two_stage_approach is not None and (
        diagnostic_replay_count != 1
        or not args.no_formal_annotation
        or args.r1_batched_output_root is None
        or not isinstance(two_stage_approach, dict)
        or free_joint_unfold is not None
        or late_approach_clearance is not None
    ):
        raise SchemaValidationError(
            "R1 two-stage approach is restricted to an isolated, uncombined diagnostic"
        )
    scene_support_from_task = manifest_payload.get("r1_scene_support_from_task", False)
    if not isinstance(scene_support_from_task, bool):
        raise SchemaValidationError("R1 scene-support switch is not boolean")
    if (
        scene_support_from_task
        and not formal_teacher_collection
        and (
            diagnostic_replay_count != 1
            or not args.no_formal_annotation
            or args.r1_batched_output_root is None
        )
    ):
        raise SchemaValidationError(
            "R1 task-scene support is restricted to isolated one-replay diagnostics"
        )
    raw_release_height = manifest_payload.get("r1_release_height_offset_m", 0.0)
    if isinstance(raw_release_height, bool) or not isinstance(
        raw_release_height, (int, float)
    ):
        raise SchemaValidationError("R1 release height offset is invalid")
    release_height_offset_m = float(raw_release_height)
    if release_height_offset_m < 0.0:
        raise SchemaValidationError("R1 release height offset is negative")
    jobs = _load_jobs(
        Path(args.r1_batched_jobs).resolve(),
        offset=args.r1_batched_offset,
        limit=args.r1_batched_limit,
    )
    _validate_common_job_contract(jobs)
    first_arguments = list(jobs[0]["argv"])
    environment_count = len(jobs) * REPLAY_COUNT
    with tempfile.TemporaryDirectory(prefix="order9-r1-batched-") as temporary_text:
        temporary = Path(temporary_text)
        combined_raw = temporary / "evaluation_rollout.pt"
        combined_episodes = temporary / "evaluation_episodes.jsonl"
        _replace_value(first_arguments, "--num-envs", str(environment_count))
        _replace_value(
            first_arguments, "--evaluation-episode-count", str(environment_count)
        )
        _replace_value(first_arguments, "--output-raw", str(combined_raw))
        _replace_value(first_arguments, "--evaluation-jsonl", str(combined_episodes))
        _replace_value(
            first_arguments,
            "--generation-id",
            f"r1_nominal_batched:{jobs[0]['name']}",
        )
        _configure_batched_environment_spacing(first_arguments)
        previous_argv = sys.argv
        namespace: dict[str, Any] = {
            "__file__": str(PROTECTED_ROLLOUT),
            "__name__": "order9_r1_batched_protected_runtime",
            "__package__": None,
            "__cached__": None,
        }
        original_compute = Order9TensorPiLRuntime.compute_nominal_qpid_hold
        try:
            sys.argv = [str(PROTECTED_ROLLOUT), *first_arguments]
            source = PROTECTED_ROLLOUT.read_text(encoding="utf-8")
            marker = "\n_exit_code = 1\n"
            if source.count(marker) != 1:
                raise SchemaValidationError("protected C3 rollout entry marker changed")
            exec(
                compile(source.split(marker, 1)[0], str(PROTECTED_ROLLOUT), "exec"),
                namespace,
            )
            (
                config,
                physical,
                tasks,
                bundles,
                wrench_trajectories,
                source_bucket_id,
            ) = _load_inputs(
                namespace,
                jobs,
                release_height_offset_m=release_height_offset_m,
            )
            if phase_time_dilation is not None:
                bundles = tuple(
                    replace(
                        bundle,
                        phase_trajectories={
                            **bundle.phase_trajectories,
                            "approach": _retime_trajectory(
                                bundle.phase_trajectories["approach"],
                                scale=phase_time_dilation,
                            ),
                        },
                    )
                    for bundle in bundles
                )
            if scene_support_from_task:
                if len(tasks) != 1:
                    raise SchemaValidationError(
                        "R1 task-scene support requires exactly one task"
                    )
                surfaces = tasks[0].scene.environment.support_surfaces
                if len(surfaces) != 1:
                    raise SchemaValidationError(
                        "R1 task-scene support requires exactly one support"
                    )
                surface = surfaces[0]
                geometries = {
                    value.geometry_id: value
                    for value in tasks[0].scene.geometry_library
                }
                geometry = geometries.get(surface.geometry_id)
                raw_size = (
                    None
                    if geometry is None
                    else (geometry.primitive_params or {}).get("size_m")
                )
                if not isinstance(raw_size, list) or len(raw_size) != 3:
                    raise SchemaValidationError(
                        "R1 task-scene support lacks box dimensions"
                    )
                task_support_size = tuple(float(value) for value in raw_size)
                task_support_pose = tuple(float(value) for value in surface.pose_world)
                original_scene_cfg = namespace["_scene_cfg"]

                def scene_cfg_with_task_support(**kwargs):
                    kwargs["support_size"] = task_support_size
                    kwargs["support_pose"] = task_support_pose
                    return original_scene_cfg(**kwargs)

                namespace["_scene_cfg"] = scene_cfg_with_task_support
            first_bundle = bundles[0]
            module_ids = tuple(
                sorted(module.module_id for module in first_bundle.morphology.modules)
            )
            joint_ids = tuple(
                sorted(
                    {
                        str(port.mechanical_limits["mechanism_joint_id"])
                        for port in physical.dock_ports
                        if port.mechanical_limits.get("mechanism_joint_id")
                    }
                )
            )
            approach_start_joint_bank = torch.tensor(
                [
                    [
                        [
                            float(
                                bundle.phase_trajectories["approach"]
                                .knots[0]
                                .posture_target.joint_pos_target[
                                    f"module_{module_id}:{joint_id}"
                                ]
                            )
                            for joint_id in joint_ids
                        ]
                        for module_id in module_ids
                    ]
                    for bundle in bundles
                ],
                dtype=torch.float32,
            ).repeat_interleave(REPLAY_COUNT, dim=0)
            approach_goal_joint_bank = torch.tensor(
                [
                    [
                        [
                            float(
                                bundle.phase_trajectories["approach"]
                                .knots[-1]
                                .posture_target.joint_pos_target[
                                    f"module_{module_id}:{joint_id}"
                                ]
                            )
                            for joint_id in joint_ids
                        ]
                        for module_id in module_ids
                    ]
                    for bundle in bundles
                ],
                dtype=torch.float32,
            ).repeat_interleave(REPLAY_COUNT, dim=0)
            base_deltas = []
            bases = []
            for index, (task, bundle, job) in enumerate(zip(tasks, bundles, jobs)):
                base_deltas.append(
                    _candidate_virtual_contact_delta(
                        namespace,
                        config=config,
                        physical=physical,
                        task=task,
                        bundle=bundle,
                        argv=job["argv"],
                        module_ids=module_ids,
                        joint_ids=joint_ids,
                    )
                )
                bases.append(
                    _candidate_contact_basis(
                        namespace,
                        config=config,
                        physical=physical,
                        bundle=bundle,
                        module_ids=module_ids,
                        joint_ids=joint_ids,
                    )
                )
                print(
                    f"ORDER9_R1_BATCH_PREPARED={index + 1}/{len(jobs)}",
                    flush=True,
                )
            selected_ids = []
            for trajectory in wrench_trajectories:
                selected_ids.append(
                    tuple(
                        value.anchor_id
                        for value in namespace["_maintain_assignments"](trajectory)
                    )
                )
            if any(value != selected_ids[0] for value in selected_ids[1:]):
                raise SchemaValidationError("R1 batched selected anchors differ")
            configure_order9_r1_batched_nominal_runtime(
                bundles=bundles,
                tasks=tasks,
                wrench_trajectories=wrench_trajectories,
                replay_count=REPLAY_COUNT,
                phase_time_dilations=(
                    {} if lift_time_dilation is None else {"lift": lift_time_dilation}
                ),
                maintain_vertical_scale=maintain_vertical_scale,
            )
            namespace["Order9C3NominalTensorReference"] = (
                Order9R1BatchedNominalTensorReference
            )
            namespace["Order9TensorWrenchRangeReference"] = (
                Order9R1BatchedWrenchRangeReference
            )
            original_translated_task = namespace["_translated_task"]

            def translated_task(_task, origin, index):
                return original_translated_task(
                    order9_r1_batched_task_for_environment(index), origin, index
                )

            namespace["_translated_task"] = translated_task
            _install_batched_formal_start(namespace)
            base_delta_bank = torch.stack(base_deltas).repeat_interleave(
                REPLAY_COUNT, dim=0
            )
            default_phase_durations = namespace[
                "Order9ObjectTaskRuntimeConfig"
            ]().phase_duration_s
            phase_duration_bank = torch.tensor(
                [
                    [
                        max(
                            float(default_phase_durations[phase.value]),
                            float(
                                bundle.phase_trajectories[phase.value].knots[-1].t_rel_s
                            )
                            + 5.0,
                        )
                        for phase in namespace["ORDER9_OBJECT_TASK_PHASES"]
                    ]
                    for bundle in bundles
                ],
                dtype=torch.float32,
            ).repeat_interleave(REPLAY_COUNT, dim=0)
            original_apply = namespace["apply_order9_virtual_contact_compression"]

            def apply_batched_virtual_contact(task_target, **kwargs):
                kwargs["joint_delta_rad"] = base_delta_bank.to(
                    device=task_target.nominal_joint_positions_rad.device,
                    dtype=task_target.nominal_joint_positions_rad.dtype,
                )
                phase_indices = kwargs["phase_index"]
                durations = phase_duration_bank.to(
                    device=phase_indices.device,
                    dtype=task_target.nominal_joint_positions_rad.dtype,
                )
                kwargs["phase_duration_s"] = durations[
                    torch.arange(environment_count, device=phase_indices.device),
                    phase_indices,
                ]
                return original_apply(task_target, **kwargs)

            namespace["apply_order9_virtual_contact_compression"] = (
                apply_batched_virtual_contact
            )
            raw_sweep = str(
                manifest_payload.get("r1_nominal_additional_compression_sweep_mm", "")
            )
            sweep_values = [float(value) for value in raw_sweep.split(",") if value]
            if len(sweep_values) != REPLAY_COUNT or len(set(sweep_values)) != 1:
                raise SchemaValidationError("R1 batched compression value differs")
            additional_mm = sweep_values[0]
            values_m = torch.tensor(
                [additional_mm * 1.0e-3] * environment_count, dtype=torch.float32
            )
            matrix_bank = torch.stack(
                [basis.contact_to_joint_position for basis in bases]
            ).repeat_interleave(REPLAY_COUNT, dim=0)
            mask_bank = torch.stack(
                [basis.contact_slot_mask for basis in bases]
            ).repeat_interleave(REPLAY_COUNT, dim=0)
            contact_values_m = None
            if contact_compression_by_slot_mm is not None:
                contact_values_m = (
                    _contact_compression_values_by_active_slot(
                        mask_bank,
                        contact_compression_by_slot_mm,
                    )
                    * 1.0e-3
                )

            def compute_batched(runtime, **kwargs):
                target = kwargs["task_target"]
                compression_delta = torch.zeros_like(target.nominal_joint_positions_rad)
                if additional_mm > 0.0 or contact_values_m is not None:
                    matrices = matrix_bank.to(
                        device=target.nominal_joint_positions_rad.device,
                        dtype=target.nominal_joint_positions_rad.dtype,
                    )
                    masks = mask_bank.to(device=matrices.device)
                    values = values_m.to(device=matrices.device, dtype=matrices.dtype)
                    physical_action = torch.zeros(
                        (environment_count, masks.shape[1], 6),
                        device=matrices.device,
                        dtype=matrices.dtype,
                    )
                    if contact_values_m is None:
                        normal_values = values.unsqueeze(1).expand(-1, masks.shape[1])
                    else:
                        normal_values = contact_values_m.to(
                            device=matrices.device,
                            dtype=matrices.dtype,
                        )
                    physical_action[:, :, 0] = torch.where(
                        masks, normal_values, physical_action[:, :, 0]
                    )
                    flat = torch.einsum(
                        "bij,bj->bi",
                        matrices,
                        physical_action.reshape(environment_count, -1),
                    )
                    delta = flat.reshape_as(target.nominal_joint_positions_rad)
                    delta *= _compression_scale(target).reshape(environment_count, 1, 1)
                    compression_delta = delta
                    target = replace(
                        target,
                        nominal_joint_positions_rad=(
                            target.nominal_joint_positions_rad + delta
                        ),
                    )
                target = _diagnostic_free_joint_unfold_target(
                    target,
                    profile=free_joint_unfold,
                    module_ids=module_ids,
                    joint_ids=joint_ids,
                )
                target = _diagnostic_late_approach_clearance_target(
                    target,
                    profile=late_approach_clearance,
                )
                target = _diagnostic_joint_clearance_pulse_target(
                    target,
                    profile=joint_clearance_pulse,
                    module_ids=module_ids,
                    joint_ids=joint_ids,
                )
                target = _diagnostic_two_stage_approach_target(
                    target,
                    profile=two_stage_approach,
                    start_joint_positions_rad=approach_start_joint_bank.to(
                        device=target.nominal_joint_positions_rad.device,
                        dtype=target.nominal_joint_positions_rad.dtype,
                    ),
                    goal_joint_positions_rad=approach_goal_joint_bank.to(
                        device=target.nominal_joint_positions_rad.device,
                        dtype=target.nominal_joint_positions_rad.dtype,
                    ),
                )
                target = _diagnostic_grasp_centering_target(
                    target,
                    profile=grasp_centering,
                )
                kwargs["task_target"] = target
                result = original_compute(runtime, **kwargs)
                return _diagnostic_grasp_torque_bias(
                    result,
                    target=target,
                    compression_direction=compression_delta,
                    effort_limits_nm=torch.tensor(
                        tuple(float(value) for value in runtime.decoder._effort_limits),
                        device=target.nominal_joint_positions_rad.device,
                        dtype=target.nominal_joint_positions_rad.dtype,
                    ),
                    fraction=grasp_torque_bias_fraction,
                )

            Order9TensorPiLRuntime.compute_nominal_qpid_hold = compute_batched
            print(
                "ORDER9_R1_BATCH_ISAAC_START="
                + json.dumps(
                    {
                        "source_bucket_id": source_bucket_id,
                        "candidate_count": len(jobs),
                        "environment_count": environment_count,
                        "additional_compression_mm": additional_mm,
                        "contact_compression_by_slot_mm": (
                            None
                            if contact_compression_by_slot_mm is None
                            else list(contact_compression_by_slot_mm)
                        ),
                        "maintain_vertical_scale": maintain_vertical_scale,
                        "release_height_offset_m": release_height_offset_m,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            result = namespace["main"]()
            outputs = _split_outputs(
                combined_raw=combined_raw,
                combined_episodes=combined_episodes,
                jobs=jobs,
                tasks=tasks,
                bundles=bundles,
                wrench_trajectories=wrench_trajectories,
                output_root=(
                    None
                    if args.r1_batched_output_root is None
                    else Path(args.r1_batched_output_root).resolve()
                ),
                additional_compression_mm=additional_mm,
                release_height_offset_m=release_height_offset_m,
                scene_support_from_task=scene_support_from_task,
                lift_time_dilation=lift_time_dilation,
                grasp_centering=grasp_centering,
                grasp_torque_bias_fraction=grasp_torque_bias_fraction,
                contact_compression_by_slot_mm=contact_compression_by_slot_mm,
                maintain_vertical_scale=maintain_vertical_scale,
                formal_teacher_collection=formal_teacher_collection,
            )
            if not args.no_formal_annotation and args.r1_batched_output_root is None:
                for job, (raw_path, episode_path) in zip(jobs, outputs):
                    manifest_path = raw_path.parents[1] / "case_manifest.json"
                    materialized = load_order9_r1_isaac_case(manifest_path, REPOSITORY)
                    annotate_order9_r1_nominal_isaac_result(
                        materialized, repository_root=REPOSITORY
                    )
                    annotate_compression(
                        raw_path=raw_path,
                        episode_path=episode_path,
                        values_mm=(additional_mm,) * REPLAY_COUNT,
                        call_count=int(result.get("environment_step_count", 1)),
                    )
                    validate_order9_r1_nominal_v7_isaac_result(
                        materialized, repository_root=REPOSITORY
                    )
                    validated = torch.load(
                        raw_path, map_location="cpu", weights_only=False
                    )["metadata"]
                    if (
                        validated.get("r1_batched_nominal_runtime_version")
                        != BATCH_VERSION
                        or validated.get("r1_batched_nominal_wrapper_sha256")
                        != hash_file(Path(__file__))
                        or validated.get("r1_batched_nominal_runtime_sha256")
                        != hash_file(BATCH_RUNTIME)
                    ):
                        raise SchemaValidationError(
                            "R1 batched output provenance differs"
                        )
            print(
                "ORDER9_R1_BATCH_COMPLETE="
                + json.dumps(
                    {
                        "candidate_count": len(outputs),
                        "episode_count": len(outputs) * REPLAY_COUNT,
                        "outputs": [str(raw) for raw, _episodes in outputs],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except BaseException:
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            raise
        finally:
            sys.argv = previous_argv
            Order9TensorPiLRuntime.compute_nominal_qpid_hold = original_compute
            simulation_app = namespace.get("simulation_app")
            if simulation_app is not None:
                simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
