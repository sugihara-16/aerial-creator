#!/usr/bin/env python3
from __future__ import annotations

"""Audit Order 9 rolling-teacher frame reconstruction without Isaac/PhysX."""

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
from typing import Mapping

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.controllers.rigid_body_model import RigidBodyControlModelBuilder
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.robot_model.physical_model_builder import (
    build_physical_model_from_config,
)
from amsrr.robot_model.urdf_loader import URDFModel, load_urdf
from amsrr.robot_model.urdf_transforms import (
    link_poses_at_joint_positions,
)
from amsrr.robot_model.whole_structure_kinematics import (
    WholeStructureKinematics,
    ordered_global_dock_joint_ids,
)
from amsrr.schemas.common import Pose7D
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import ControllerStatus
from amsrr.schemas.runtime import (
    ModuleRuntimeState,
    ObjectRuntimeState,
    RuntimeObservation,
    TaskProgressState,
)
from amsrr.schemas.task_spec import TaskSpec
from amsrr.simulation.order9_fixed_nominal_asset import (
    load_order9_fixed_nominal_asset_manifest,
    validate_order9_fixed_nominal_asset_manifest_bytes,
)
from amsrr.simulation.order9_object_task_state import (
    load_order9_canonical_reset,
)
from amsrr.training.order9_articulated_teacher import (
    Order9ArticulatedTrajectoryTeacher,
    _baseline_object_goal,
    _phase_target_configuration,
    _quaternion_distance,
)
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.utils.hashing import hash_file
from scripts.order9_isaac_shadow_smoke import _c2_bucket_inputs


DEFAULT_ROLLING_REPORT = (
    "artifacts/p4_full/order9/posture_resolver/"
    "c2_rolling_approach_v5_bucket8_real_isaac.json"
)
DEFAULT_FIXED_ASSET_MANIFEST = (
    "artifacts/p4_full/order9/fixed_nominal_asset/manifest.json"
)
DEFAULT_C2_TENSOR_ROLLOUT = (
    "artifacts/p4_full/order9/stages/"
    "c2_pi_l_ppo_fixed_conservative/evaluations/"
    "extension_update_000049_final_200/bucket8/evaluation_rollout.pt"
)
DEFAULT_OUTPUT = (
    "artifacts/p4_full/order9/posture_resolver/"
    "c2_rolling_approach_v5_offline_frame_audit.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rolling-report",
        default=DEFAULT_ROLLING_REPORT,
    )
    parser.add_argument(
        "--fixed-asset-manifest",
        default=DEFAULT_FIXED_ASSET_MANIFEST,
    )
    parser.add_argument(
        "--c2-tensor-rollout",
        default=DEFAULT_C2_TENSOR_ROLLOUT,
    )
    parser.add_argument(
        "--config",
        default="configs/training/order9_learning_curriculum.yaml",
    )
    parser.add_argument("--tensor-time-index", type=int, default=0)
    parser.add_argument("--tensor-environment-index", type=int, default=0)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return (
        (REPOSITORY_ROOT / value).resolve()
        if not value.is_absolute()
        else value.resolve()
    )


def _pose_position_error(left: Pose7D, right: Pose7D) -> float:
    return math.dist(left[:3], right[:3])


def _pose_attitude_error(left: Pose7D, right: Pose7D) -> float:
    return _quaternion_distance(left[3:7], right[3:7])


def _local_joint_maps(
    global_positions: Mapping[str, float],
    global_velocities: Mapping[str, float],
    *,
    module_id: int,
) -> tuple[dict[str, float], dict[str, float]]:
    prefix = f"module_{module_id}:"
    generated_prefix = f"module_{module_id}__"

    def local(
        values: Mapping[str, float],
    ) -> dict[str, float]:
        result = {}
        for key, value in values.items():
            if key.startswith(prefix):
                result[key[len(prefix) :]] = float(value)
            elif key.startswith(generated_prefix):
                result[key[len(generated_prefix) :]] = float(value)
        return result

    return local(global_positions), local(global_velocities)


def _generated_joint_positions(
    global_positions: Mapping[str, float],
) -> dict[str, float]:
    return {
        (
            key.replace(":", "__", 1)
            if ":" in key
            else key
        ): float(value)
        for key, value in global_positions.items()
    }


def _observation_from_generated_urdf(
    *,
    morphology: MorphologyGraph,
    urdf_model: URDFModel,
    root_pose_world: Pose7D,
    root_twist_world: tuple[float, ...],
    global_positions: Mapping[str, float],
    global_velocities: Mapping[str, float],
    object_id: str,
    object_pose_world: Pose7D,
    object_twist_world: tuple[float, ...],
    time_s: float,
) -> RuntimeObservation:
    poses = link_poses_at_joint_positions(
        urdf_model,
        _generated_joint_positions(global_positions),
        root_pose_world=root_pose_world,
    )
    module_states = []
    for module in sorted(morphology.modules, key=lambda value: value.module_id):
        local_q, local_qdot = _local_joint_maps(
            global_positions,
            global_velocities,
            module_id=module.module_id,
        )
        module_states.append(
            ModuleRuntimeState(
                module_id=module.module_id,
                pose_world=poses[f"module_{module.module_id}__fc"],
                twist_world=list(root_twist_world),
                joint_positions=local_q,
                joint_velocities=local_qdot,
            )
        )
    observation = RuntimeObservation(
        time_s=float(time_s),
        morphology_graph=morphology,
        module_states=module_states,
        object_states=[
            ObjectRuntimeState(
                object_id=object_id,
                pose_world=object_pose_world,
                twist_world=list(object_twist_world),
            )
        ],
        contact_states=[],
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
        task_progress=TaskProgressState(
            phase_label="approach",
            progress_ratio=0.0,
        ),
    )
    observation.validate()
    return observation


def _observation_from_legacy_reconstruction(
    *,
    morphology: MorphologyGraph,
    physical_model: PhysicalModel,
    root_pose_world: Pose7D,
    root_twist_world: tuple[float, ...],
    global_positions: Mapping[str, float],
    global_velocities: Mapping[str, float],
    object_id: str,
    object_pose_world: Pose7D,
    object_twist_world: tuple[float, ...],
    time_s: float,
) -> RuntimeObservation:
    dock_ids = ordered_global_dock_joint_ids(morphology, physical_model)

    def dock_position(joint_id: str) -> float:
        generated_id = joint_id.replace(":", "__", 1)
        if joint_id in global_positions:
            return float(global_positions[joint_id])
        if generated_id in global_positions:
            return float(global_positions[generated_id])
        raise RuntimeError(
            f"rolling endpoint lacks structural Dock joint {joint_id}"
        )

    fk = WholeStructureKinematics().forward(
        morphology,
        physical_model,
        {joint_id: dock_position(joint_id) for joint_id in dock_ids},
        root_pose_world,
        (),
    )
    module_states = []
    for module in sorted(morphology.modules, key=lambda value: value.module_id):
        local_q, local_qdot = _local_joint_maps(
            global_positions,
            global_velocities,
            module_id=module.module_id,
        )
        module_states.append(
            ModuleRuntimeState(
                module_id=module.module_id,
                pose_world=fk.module_root_poses_world[module.module_id],
                twist_world=list(root_twist_world),
                joint_positions=local_q,
                joint_velocities=local_qdot,
            )
        )
    observation = RuntimeObservation(
        time_s=float(time_s),
        morphology_graph=morphology,
        module_states=module_states,
        object_states=[
            ObjectRuntimeState(
                object_id=object_id,
                pose_world=object_pose_world,
                twist_world=list(object_twist_world),
            )
        ],
        contact_states=[],
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
        task_progress=TaskProgressState(
            phase_label="approach",
            progress_ratio=0.0,
        ),
    )
    observation.validate()
    return observation


def _metric_vector(
    metrics: Mapping[str, object],
    prefix: str,
    size: int,
) -> tuple[float, ...]:
    values = []
    for index in range(size):
        key = f"endpoint.{prefix}_{index}"
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise RuntimeError(f"rolling endpoint lacks {key}")
        values.append(float(value))
    return tuple(values)


def _endpoint_state(
    endpoint: Mapping[str, object],
) -> tuple[
    Pose7D,
    tuple[float, ...],
    dict[str, float],
    dict[str, float],
    Pose7D,
    tuple[float, ...],
    float,
]:
    metrics = endpoint.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError("rolling endpoint lacks metrics")
    positions = {
        key.removeprefix("endpoint.joint_position_rad."): float(value)
        for key, value in metrics.items()
        if key.startswith("endpoint.joint_position_rad.")
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }
    velocities = {
        key.removeprefix("endpoint.joint_velocity_radps."): float(value)
        for key, value in metrics.items()
        if key.startswith("endpoint.joint_velocity_radps.")
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }
    return (
        _metric_vector(metrics, "robot_root_pose_world", 7),  # type: ignore[return-value]
        _metric_vector(metrics, "robot_root_twist_world", 6),
        positions,
        velocities,
        _metric_vector(metrics, "object_pose_world", 7),  # type: ignore[return-value]
        _metric_vector(metrics, "object_twist_world", 6),
        float(metrics.get("endpoint.elapsed_s", 0.0)),
    )


def _phase_reset_joints(
    values: Mapping[str, float],
) -> dict[str, float]:
    return {key: float(value) for key, value in values.items()}


def _centroidal_pose(
    *,
    morphology: MorphologyGraph,
    physical_model: PhysicalModel,
    observation: RuntimeObservation,
) -> Pose7D:
    return RigidBodyControlModelBuilder().build(
        morphology,
        physical_model,
        observation,
    ).body_pose_world


def _teacher_stage(
    *,
    context: HighLevelPolicyContext,
    task: TaskSpec,
    physical_model: PhysicalModel,
) -> dict[str, object]:
    teacher = Order9ArticulatedTrajectoryTeacher(physical_model)
    plan = teacher.plan(
        context,
        initial_object_poses_world={
            value.object_id: value.pose_world for value in task.scene.objects
        },
    )
    observation = context.runtime_observation
    if observation is None:
        raise RuntimeError("teacher audit context has no runtime observation")
    module_states = {
        state.module_id: state for state in observation.module_states
    }
    base_pose = tuple(
        module_states[context.morphology_graph.base_module_id].pose_world
    )
    dock_ids = ordered_global_dock_joint_ids(
        context.morphology_graph,
        physical_model,
    )
    start_q = {}
    for joint_id in dock_ids:
        module_prefix, local_id = joint_id.split(":", 1)
        module_id = int(module_prefix.removeprefix("module_"))
        start_q[joint_id] = float(
            module_states[module_id].joint_positions.get(local_id, 0.0)
        )
    object_id = task.scene.objects[0].object_id
    current_object_pose = tuple(
        next(
            state.pose_world
            for state in observation.object_states
            if state.object_id == object_id
        )
    )
    target_q, target_base_pose, _, terminal = _phase_target_configuration(
        phase=plan.task_phase,
        start_q=start_q,
        start_base_pose=base_pose,
        current_object_pose=current_object_pose,
        initial_object_pose=task.scene.objects[0].pose_world,
        object_goal_pose=_baseline_object_goal(
            plan.raw_trajectory,
            object_id=object_id,
        ),
        solution=plan.ik_solution,
        config=teacher.config,
    )
    maximum_joint_error = max(
        (
            abs(
                float(plan.ik_solution.joint_positions_rad[joint_id])
                - float(start_q[joint_id])
            )
            for joint_id in start_q
        ),
        default=0.0,
    )
    if terminal:
        stage = "final_pregrasp"
    elif all(abs(float(value)) <= 1.0e-12 for value in target_q.values()):
        planar_delta = max(
            abs(float(target_base_pose[index]) - float(base_pose[index]))
            for index in (0, 1)
        )
        stage = "vertical" if planar_delta <= 1.0e-9 else "lateral"
    elif (
        maximum_joint_error
        <= teacher.config.approach_staging_joint_tolerance_rad
    ):
        stage = "attitude_convergence"
    else:
        stage = "articulation"
    return {
        "raw_pi_h_trajectory_hash": plan.raw_trajectory.stable_hash(),
        "resolved_trajectory_hash": plan.trajectory.stable_hash(),
        "stage": stage,
        "phase_target_reached": bool(plan.phase_target_reached),
        "start_base_pose_world": list(base_pose),
        "target_base_pose_world": list(target_base_pose),
        "target_translation_error_m": _pose_position_error(
            base_pose,
            target_base_pose,
        ),
        "vertical_target_error_m": (
            float(target_base_pose[2]) - float(base_pose[2])
        ),
        "maximum_contact_joint_error_rad": maximum_joint_error,
        "contact_attitude_error_rad": _pose_attitude_error(
            base_pose,
            plan.ik_solution.base_pose_world,
        ),
        "position_gate_tolerance_m": (
            teacher.config.approach_staging_position_tolerance_m
        ),
    }


def _phase_zero_tensor_validation(
    *,
    tensor_rollout_path: Path,
    urdf_model: URDFModel,
    physical_model: PhysicalModel,
    expected_morphology: MorphologyGraph,
    time_index: int,
    environment_index: int,
) -> dict[str, object]:
    artifact = torch.load(
        tensor_rollout_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    metadata = artifact["metadata"]
    tensors = artifact["tensors"]
    morphology = MorphologyGraph.from_dict(metadata["morphology_graph"])
    if morphology.stable_hash() != expected_morphology.stable_hash():
        raise RuntimeError("C2 tensor rollout morphology differs from audit graph")
    module_ids = [int(value) for value in metadata["module_ids"]]
    local_joint_ids = [str(value) for value in metadata["local_joint_ids"]]
    root_pose = tuple(
        float(value)
        for value in tensors["robot_root_pose_world"][
            time_index,
            environment_index,
        ]
    )
    positions = {
        f"module_{module_id}__{joint_id}": float(
            tensors["local_joint_positions_rad"][
                time_index,
                environment_index,
                module_index,
                joint_index,
            ]
        )
        for module_index, module_id in enumerate(module_ids)
        for joint_index, joint_id in enumerate(local_joint_ids)
    }
    predicted = link_poses_at_joint_positions(
        urdf_model,
        positions,
        root_pose_world=root_pose,  # type: ignore[arg-type]
    )
    actual_module_poses = {
        module_id: tuple(
            float(value)
            for value in tensors["module_pose_world"][
                time_index,
                environment_index,
                module_index,
            ]
        )
        for module_index, module_id in enumerate(module_ids)
    }
    rows = []
    predicted_states = []
    actual_states = []
    for module_index, module_id in enumerate(module_ids):
        local_q = {
            joint_id: float(
                tensors["local_joint_positions_rad"][
                    time_index,
                    environment_index,
                    module_index,
                    joint_index,
                ]
            )
            for joint_index, joint_id in enumerate(local_joint_ids)
        }
        predicted_pose = predicted[f"module_{module_id}__fc"]
        actual_pose = actual_module_poses[module_id]
        rows.append(
            {
                "module_id": module_id,
                "position_error_m": _pose_position_error(
                    predicted_pose,
                    actual_pose,
                ),
                "attitude_error_rad": _pose_attitude_error(
                    predicted_pose,
                    actual_pose,
                ),
                "predicted_module_pose_world": list(predicted_pose),
                "actual_module_pose_world": list(actual_pose),
            }
        )
        common = {
            "module_id": module_id,
            "twist_world": [0.0] * 6,
            "joint_positions": local_q,
            "joint_velocities": {},
        }
        predicted_states.append(
            ModuleRuntimeState(
                pose_world=predicted_pose,
                **common,
            )
        )
        actual_states.append(
            ModuleRuntimeState(
                pose_world=actual_pose,
                **common,
            )
        )

    def observation(states: list[ModuleRuntimeState]) -> RuntimeObservation:
        return RuntimeObservation(
            time_s=0.0,
            morphology_graph=morphology,
            module_states=states,
            object_states=[],
            contact_states=[],
            controller_status=ControllerStatus(
                status="ok",
                qp_feasible=True,
            ),
            task_progress=TaskProgressState(
                phase_label="approach",
                progress_ratio=0.0,
            ),
        )

    predicted_centroidal = _centroidal_pose(
        morphology=morphology,
        physical_model=physical_model,
        observation=observation(predicted_states),
    )
    actual_centroidal = _centroidal_pose(
        morphology=morphology,
        physical_model=physical_model,
        observation=observation(actual_states),
    )
    return {
        "artifact_version": artifact["artifact_version"],
        "tensor_rollout_path": str(tensor_rollout_path),
        "tensor_rollout_sha256": hash_file(tensor_rollout_path),
        "time_index": time_index,
        "environment_index": environment_index,
        "root_pose_world": list(root_pose),
        "module_comparisons": rows,
        "maximum_module_position_error_m": max(
            row["position_error_m"] for row in rows
        ),
        "maximum_module_attitude_error_rad": max(
            row["attitude_error_rad"] for row in rows
        ),
        "predicted_centroidal_pose_world": list(predicted_centroidal),
        "actual_centroidal_pose_world": list(actual_centroidal),
        "centroidal_position_error_m": _pose_position_error(
            predicted_centroidal,
            actual_centroidal,
        ),
        "centroidal_attitude_error_rad": _pose_attitude_error(
            predicted_centroidal,
            actual_centroidal,
        ),
    }


def main() -> int:
    args = _parser().parse_args()
    rolling_report_path = _resolve(args.rolling_report)
    fixed_manifest_path = _resolve(args.fixed_asset_manifest)
    tensor_rollout_path = _resolve(args.c2_tensor_rollout)
    output_path = _resolve(args.output)

    rolling_report = json.loads(
        rolling_report_path.read_text(encoding="utf-8")
    )
    config = load_order9_learning_config(_resolve(args.config))
    physical_model = build_physical_model_from_config(
        _resolve(config.production_runtime.robot_model_config_path)
    )
    reset = load_order9_canonical_reset(
        _resolve(config.production_runtime.canonical_order8_report_path),
        expected_sha256=(
            config.production_runtime.canonical_order8_report_sha256
        ),
    )
    fixed_manifest = load_order9_fixed_nominal_asset_manifest(
        fixed_manifest_path
    )
    validate_order9_fixed_nominal_asset_manifest_bytes(
        fixed_manifest,
        repository_root=REPOSITORY_ROOT,
        expected_physical_model_hash=physical_model.stable_hash(),
    )
    generated_urdf_path = _resolve(fixed_manifest.generated_urdf_path)
    urdf_model = load_urdf(generated_urdf_path)

    bucket_evidence = rolling_report["c2_bucket"]
    bucket_manifest_path = _resolve(bucket_evidence["manifest_path"])
    (
        task,
        morphology,
        base_context,
        _,
        _,
        _,
        phase_reset,
        _,
        _,
        _,
    ) = _c2_bucket_inputs(
        bucket_manifest_path,
        bucket_id=str(bucket_evidence["bucket_id"]),
        repository=REPOSITORY_ROOT,
        physical_model=physical_model,
        reset=reset,
        articulated_urdf_path=generated_urdf_path,
    )
    if morphology.stable_hash() != fixed_manifest.source_morphology_hash:
        raise RuntimeError("fixed asset and rolling bucket morphology differ")

    tensor_validation = _phase_zero_tensor_validation(
        tensor_rollout_path=tensor_rollout_path,
        urdf_model=urdf_model,
        physical_model=physical_model,
        expected_morphology=morphology,
        time_index=int(args.tensor_time_index),
        environment_index=int(args.tensor_environment_index),
    )

    phase_zero_positions = _phase_reset_joints(
        phase_reset.joint_positions_rad
    )
    phase_zero_velocities = _phase_reset_joints(
        phase_reset.joint_velocities_radps
    )
    initial_state = (
        tuple(phase_reset.robot_root_pose_world),
        tuple(phase_reset.robot_root_twist_world),
        phase_zero_positions,
        phase_zero_velocities,
        tuple(phase_reset.object_pose_world),
        tuple(phase_reset.object_twist_world),
        0.0,
    )

    execution_windows = rolling_report["execution_windows"]
    observations = rolling_report["observations"]
    if not isinstance(execution_windows, list) or not isinstance(
        observations,
        list,
    ):
        raise RuntimeError("rolling report has malformed window evidence")
    endpoint_offsets = []
    running = 0
    for window in execution_windows:
        running += int(window["trajectory_knot_count"])
        endpoint_offsets.append(running - 1)
    if running != len(observations):
        raise RuntimeError(
            "rolling report observation count differs from window knots"
        )

    builder = RigidBodyControlModelBuilder()
    window_rows = []
    for window_index, window in enumerate(execution_windows):
        state = (
            initial_state
            if window_index == 0
            else _endpoint_state(
                observations[endpoint_offsets[window_index - 1]]
            )
        )
        (
            root_pose,
            root_twist,
            positions,
            velocities,
            object_pose,
            object_twist,
            time_s,
        ) = state
        legacy_observation = _observation_from_legacy_reconstruction(
            morphology=morphology,
            physical_model=physical_model,
            root_pose_world=root_pose,
            root_twist_world=root_twist,
            global_positions=positions,
            global_velocities=velocities,
            object_id=task.scene.objects[0].object_id,
            object_pose_world=object_pose,
            object_twist_world=object_twist,
            time_s=time_s,
        )
        corrected_observation = _observation_from_generated_urdf(
            morphology=morphology,
            urdf_model=urdf_model,
            root_pose_world=root_pose,
            root_twist_world=root_twist,
            global_positions=positions,
            global_velocities=velocities,
            object_id=task.scene.objects[0].object_id,
            object_pose_world=object_pose,
            object_twist_world=object_twist,
            time_s=time_s,
        )
        legacy_context = replace(
            base_context,
            runtime_observation=legacy_observation,
        )
        corrected_context = replace(
            base_context,
            runtime_observation=corrected_observation,
        )
        legacy_stage = _teacher_stage(
            context=legacy_context,
            task=task,
            physical_model=physical_model,
        )
        corrected_stage = _teacher_stage(
            context=corrected_context,
            task=task,
            physical_model=physical_model,
        )
        legacy_centroidal = builder.build(
            morphology,
            physical_model,
            legacy_observation,
        ).body_pose_world
        corrected_centroidal = builder.build(
            morphology,
            physical_model,
            corrected_observation,
        ).body_pose_world
        archived_hash = window["teacher"]["raw_pi_h_trajectory_hash"]
        window_rows.append(
            {
                "window_index": window_index,
                "state_source": (
                    "canonical_phase_zero"
                    if window_index == 0
                    else f"window_{window_index - 1}_endpoint"
                ),
                "archived_raw_pi_h_trajectory_hash": archived_hash,
                "legacy_replay_hash_matches_archive": (
                    legacy_stage["raw_pi_h_trajectory_hash"]
                    == archived_hash
                ),
                "legacy": {
                    **legacy_stage,
                    "centroidal_pose_world": list(legacy_centroidal),
                },
                "urdf_fk_corrected": {
                    **corrected_stage,
                    "centroidal_pose_world": list(corrected_centroidal),
                },
                "frame_delta": {
                    "base_module_position_m": _pose_position_error(
                        tuple(
                            legacy_stage["start_base_pose_world"]
                        ),
                        tuple(
                            corrected_stage["start_base_pose_world"]
                        ),
                    ),
                    "base_module_attitude_rad": _pose_attitude_error(
                        tuple(
                            legacy_stage["start_base_pose_world"]
                        ),
                        tuple(
                            corrected_stage["start_base_pose_world"]
                        ),
                    ),
                    "centroidal_position_m": _pose_position_error(
                        legacy_centroidal,
                        corrected_centroidal,
                    ),
                    "centroidal_attitude_rad": _pose_attitude_error(
                        legacy_centroidal,
                        corrected_centroidal,
                    ),
                },
                "gate_decision_changed": (
                    legacy_stage["stage"] != corrected_stage["stage"]
                    or legacy_stage["phase_target_reached"]
                    != corrected_stage["phase_target_reached"]
                ),
            }
        )

    hash_matches = sum(
        bool(row["legacy_replay_hash_matches_archive"])
        for row in window_rows
    )
    changed = [
        int(row["window_index"])
        for row in window_rows
        if row["gate_decision_changed"]
    ]
    maximum_centroidal_delta = max(
        float(row["frame_delta"]["centroidal_position_m"])
        for row in window_rows
    )
    minimum_centroidal_delta = min(
        float(row["frame_delta"]["centroidal_position_m"])
        for row in window_rows
    )
    report = {
        "audit_version": "order9_rolling_teacher_frame_audit_v1",
        "isaac_or_physx_started": False,
        "passed": bool(
            tensor_validation["maximum_module_position_error_m"] <= 0.002
            and tensor_validation["maximum_module_attitude_error_rad"] <= 0.003
            and len(window_rows) == len(execution_windows)
        ),
        "finding": (
            "legacy_articulation_root_was_treated_as_module_fc"
        ),
        "rolling_report_path": str(rolling_report_path),
        "rolling_report_sha256": hash_file(rolling_report_path),
        "fixed_asset_manifest_path": str(fixed_manifest_path),
        "fixed_asset_manifest_sha256": hash_file(fixed_manifest_path),
        "generated_urdf_path": str(generated_urdf_path),
        "generated_urdf_sha256": hash_file(generated_urdf_path),
        "phase_zero_tensor_validation": tensor_validation,
        "summary": {
            "window_count": len(window_rows),
            "legacy_hash_match_count": hash_matches,
            "archived_trajectory_bytes_reproduced": (
                hash_matches == len(window_rows)
            ),
            "gate_decision_changed_window_indices": changed,
            "minimum_legacy_to_corrected_centroidal_delta_m": (
                minimum_centroidal_delta
            ),
            "maximum_legacy_to_corrected_centroidal_delta_m": (
                maximum_centroidal_delta
            ),
            "legacy_stage_sequence": [
                row["legacy"]["stage"] for row in window_rows
            ],
            "corrected_stage_sequence": [
                row["urdf_fk_corrected"]["stage"] for row in window_rows
            ],
            "legacy_phase_target_reached_count": sum(
                bool(row["legacy"]["phase_target_reached"])
                for row in window_rows
            ),
            "corrected_phase_target_reached_count": sum(
                bool(row["urdf_fk_corrected"]["phase_target_reached"])
                for row in window_rows
            ),
        },
        "windows": window_rows,
        "limitations": [
            "No dynamics, contact, collision, or controller tracking was evaluated.",
            "Module twists reuse the articulation-root twist; stage selection depends on pose and Dock joint position.",
            "Current source does not reproduce the archived trajectory hashes; the audit compares both frame paths under the same current teacher implementation at the archived measured states.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"OUTPUT={output_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
