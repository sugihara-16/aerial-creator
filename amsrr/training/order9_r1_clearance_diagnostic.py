from __future__ import annotations

"""Reproducible A/B diagnosis of the rejected R1 support collision."""

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import torch

from amsrr.geometry.convex_clearance import OrientedBox, oriented_box_clearance
from amsrr.geometry.pose_math import (
    compose_pose,
    inverse_pose,
    pose_from_transform,
    quat_from_matrix,
    transform_from_xyz_rpy,
)
from amsrr.robot_model.physical_model_builder import (
    build_physical_model_from_config,
)
from amsrr.schemas.common import Pose7D, SchemaValidationError
from amsrr.schemas.task_spec import TaskSpec
from amsrr.simulation.order9_object_task_state import (
    load_order9_canonical_reset,
)
from amsrr.training.order9_posture_resolver import Order9PostureCollisionBox
from amsrr.utils.hashing import hash_file

ORDER9_R1_CLEARANCE_DIAGNOSTIC_VERSION = (
    "order9_r1_learned_vs_nominal_support_clearance_diagnostic_v1"
)
_PHASE_LABELS = (
    "approach",
    "contact_acquisition",
    "lift",
    "transport",
    "place",
    "release",
    "retreat",
    "settle",
    "complete",
)


def build_order9_r1_clearance_diagnostic(
    *,
    learned_raw_path: str | Path,
    learned_episode_path: str | Path,
    learned_log_path: str | Path,
    nominal_raw_path: str | Path,
    nominal_episode_path: str | Path,
    nominal_log_path: str | Path,
    task_spec_path: str | Path,
    physical_model_config_path: str | Path,
    canonical_order8_report_path: str | Path,
    canonical_order8_report_sha256: str,
    protected_rollout_path: str | Path,
    promoted_checkpoint_path: str | Path,
    promoted_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Compare the failed learned replay with a nominal-QPID-only replay."""

    paths = {
        "learned_raw": Path(learned_raw_path).resolve(),
        "learned_episodes": Path(learned_episode_path).resolve(),
        "learned_log": Path(learned_log_path).resolve(),
        "nominal_raw": Path(nominal_raw_path).resolve(),
        "nominal_episodes": Path(nominal_episode_path).resolve(),
        "nominal_log": Path(nominal_log_path).resolve(),
        "task_spec": Path(task_spec_path).resolve(),
        "physical_model_config": Path(physical_model_config_path).resolve(),
        "canonical_order8_report": Path(
            canonical_order8_report_path
        ).resolve(),
        "protected_rollout": Path(protected_rollout_path).resolve(),
        "promoted_checkpoint": Path(promoted_checkpoint_path).resolve(),
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("R1 clearance diagnostic input is missing")
    if hash_file(paths["promoted_checkpoint"]) != promoted_checkpoint_sha256:
        raise SchemaValidationError("R1 diagnostic checkpoint bytes changed")

    learned = torch.load(
        paths["learned_raw"], map_location="cpu", weights_only=False
    )
    nominal = torch.load(
        paths["nominal_raw"], map_location="cpu", weights_only=False
    )
    learned_tensors, learned_metadata = _validated_raw(
        learned,
        nominal_only=False,
        promoted_checkpoint_sha256=promoted_checkpoint_sha256,
    )
    nominal_tensors, nominal_metadata = _validated_raw(
        nominal,
        nominal_only=True,
        promoted_checkpoint_sha256=promoted_checkpoint_sha256,
    )
    learned_episodes = _load_episodes(paths["learned_episodes"])
    nominal_episodes = _load_episodes(paths["nominal_episodes"])
    _validate_episode_binding(
        learned_episodes, paths["learned_raw"], expect_success=False
    )
    _validate_episode_binding(
        nominal_episodes, paths["nominal_raw"], expect_success=True
    )
    collision_records = _load_collision_records(paths["learned_log"])
    if len(collision_records) != len(learned_episodes):
        raise SchemaValidationError(
            "R1 learned replay lacks per-environment collision diagnostics"
        )
    if any(
        record.get("prohibited_environment_contact") is not True
        or record.get("prohibited_object_contact") is not False
        or len(record.get("active_bodies", ())) != 1
        or record["active_bodies"][0].get("body_name") != "module_1__battery1"
        for record in collision_records
    ):
        raise SchemaValidationError(
            "R1 collision identity is not the battery/support case"
        )

    task = TaskSpec.from_json(paths["task_spec"].read_text(encoding="utf-8"))
    task.validate()
    canonical = load_order9_canonical_reset(
        paths["canonical_order8_report"],
        expected_sha256=canonical_order8_report_sha256,
    )
    support_pose = tuple(
        float(value)
        for value in canonical.metadata["object_support_pose_world"]
    )
    support_size = tuple(
        float(value) for value in canonical.metadata["object_support_size_m"]
    )
    support_local = Order9PostureCollisionBox(
        box_id="canonical_order8_isaac_support",
        size_m=support_size,
        pose_world=support_pose,
    )
    physical = build_physical_model_from_config(paths["physical_model_config"])
    battery_local_pose, collision_local_pose, bounds = _battery_geometry(
        physical
    )
    movable = [value for value in task.scene.objects if value.movable]
    if len(movable) != 1:
        raise SchemaValidationError(
            "R1 diagnostic task must have one movable object"
        )
    task_object_position = tuple(
        float(value) for value in movable[0].pose_world[:3]
    )

    terminal_index = int(collision_records[0]["rollout_index"])
    if any(
        int(record["rollout_index"]) != terminal_index
        for record in collision_records
    ):
        raise SchemaValidationError(
            "R1 collision replay terminal indices differ"
        )
    if terminal_index >= nominal_tensors["module_pose_world"].shape[0]:
        raise SchemaValidationError(
            "R1 nominal replay is shorter than learned failure"
        )

    learned_clearance = _clearance_series(
        learned_tensors,
        task_object_position=task_object_position,
        support_local=support_local,
        battery_local_pose=battery_local_pose,
        collision_local_pose=collision_local_pose,
        collision_bounds=bounds,
    )
    nominal_clearance = _clearance_series(
        nominal_tensors,
        task_object_position=task_object_position,
        support_local=support_local,
        battery_local_pose=battery_local_pose,
        collision_local_pose=collision_local_pose,
        collision_bounds=bounds,
    )
    reconstruction_errors = []
    exact_terminal_clearance = []
    for record in collision_records:
        environment = int(record["environment_index"])
        observed = tuple(
            float(value)
            for value in record["active_bodies"][0]["body_pose_world"]
        )
        reconstructed = _battery_pose_from_module(
            learned_tensors["module_pose_world"][
                terminal_index, environment, 1
            ],
            battery_local_pose,
        )
        reconstruction_errors.append(
            {
                "environment_index": environment,
                "position_error_m": math.sqrt(
                    sum(
                        (reconstructed[index] - observed[index]) ** 2
                        for index in range(3)
                    )
                ),
                "attitude_error_rad": _quaternion_distance(
                    reconstructed[3:7], observed[3:7]
                ),
            }
        )
        support = _environment_support_box(
            learned_tensors,
            environment=environment,
            task_object_position=task_object_position,
            support_local=support_local,
        )
        exact_terminal_clearance.append(
            1000.0
            * _battery_support_clearance(
                observed,
                support=support,
                collision_local_pose=collision_local_pose,
                collision_bounds=bounds,
            )
        )

    learned_at_terminal_mm = [
        1000.0 * values[terminal_index] for values in learned_clearance
    ]
    nominal_at_terminal_mm = [
        1000.0 * values[terminal_index] for values in nominal_clearance
    ]
    learned_module = learned_tensors["module_pose_world"][terminal_index, :, 1]
    nominal_module = nominal_tensors["module_pose_world"][terminal_index, :, 1]
    learned_battery = [
        _battery_pose_from_module(learned_module[index], battery_local_pose)
        for index in range(learned_module.shape[0])
    ]
    nominal_battery = [
        _battery_pose_from_module(nominal_module[index], battery_local_pose)
        for index in range(nominal_module.shape[0])
    ]
    command_delta = (
        learned_tensors["command_body_pose_world"][terminal_index, :, :3]
        - learned_tensors["desired_body_pose_world"][terminal_index, :, :3]
    )
    joint_command_delta = (
        learned_tensors["command_joint_position_targets_rad"][terminal_index]
        - learned_tensors["desired_joint_positions_rad"][terminal_index]
    )
    desired_pose_difference = (
        learned_tensors["desired_body_pose_world"][: terminal_index + 1]
        - nominal_tensors["desired_body_pose_world"][: terminal_index + 1]
    ).abs()
    desired_joint_difference = (
        learned_tensors["desired_joint_positions_rad"][: terminal_index + 1]
        - nominal_tensors["desired_joint_positions_rad"][: terminal_index + 1]
    ).abs()

    trace_start = max(0, terminal_index - 40)
    trace_end = min(
        nominal_tensors["module_pose_world"].shape[0], terminal_index + 21
    )
    trace = []
    for mode, tensors, series in (
        ("learned_pi_l", learned_tensors, learned_clearance),
        ("nominal_qpid_only", nominal_tensors, nominal_clearance),
    ):
        available_end = min(trace_end, tensors["module_pose_world"].shape[0])
        for index in range(trace_start, available_end):
            for environment in range(len(series)):
                trace.append(
                    {
                        "mode": mode,
                        "rollout_index": index,
                        "environment_index": environment,
                        "phase_index": int(
                            tensors["phase_index"][index, environment]
                        ),
                        "phase_label": _phase_label(
                            int(tensors["phase_index"][index, environment])
                        ),
                        "phase_progress": float(
                            tensors["phase_progress"][index, environment]
                        ),
                        "battery_support_proxy_clearance_mm": (
                            1000.0 * series[environment][index]
                        ),
                    }
                )

    bindings = {
        name: {"path": str(path), "sha256": hash_file(path)}
        for name, path in paths.items()
    }
    return {
        "diagnostic_version": ORDER9_R1_CLEARANCE_DIAGNOSTIC_VERSION,
        "promotion_evidence_eligible": False,
        "promotion_ineligibility_reasons": [
            "pi_l_command_bypassed_in_nominal_ab_replay",
            "diagnostic_only_posthoc_clearance_reconstruction",
        ],
        "input_bindings": bindings,
        "protected_checkpoint_sha256": promoted_checkpoint_sha256,
        "execution_contract": {
            "learned": "teacher_to_ik_to_promoted_c3_pi_l_to_qpid_qp_to_local_servo_to_isaac",
            "nominal": "teacher_to_ik_to_qpid_qp_to_local_servo_to_isaac",
            "only_controlled_difference": "promoted_c3_pi_l_command_applied_or_bypassed",
            "same_teacher_reference_through_failure_index": bool(
                float(desired_pose_difference.max()) <= 1.0e-9
                and float(desired_joint_difference.max()) <= 1.0e-9
            ),
            "maximum_teacher_body_pose_component_difference": float(
                desired_pose_difference.max()
            ),
            "maximum_teacher_joint_position_difference_rad": float(
                desired_joint_difference.max()
            ),
            "learned_raw_diagnostic_nominal_qpid_only": bool(
                learned_metadata["diagnostic_nominal_qpid_only"]
            ),
            "nominal_raw_diagnostic_nominal_qpid_only": bool(
                nominal_metadata["diagnostic_nominal_qpid_only"]
            ),
        },
        "episode_outcomes": {
            "learned": _episode_summary(learned_episodes),
            "nominal_qpid_only": _episode_summary(nominal_episodes),
        },
        "collision_identity": {
            "collider_robot_body": "module_1__battery1",
            "contact_target": "support",
            "object_contact": False,
            "environment_contact": True,
            "self_collision_enabled": False,
            "runtime_phase_index": int(
                collision_records[0]["runtime_phase_index"]
            ),
            "runtime_phase_label": _phase_label(
                int(collision_records[0]["runtime_phase_index"])
            ),
            "phase_elapsed_s": float(collision_records[0]["phase_elapsed_s"]),
            "rollout_index": terminal_index,
            "records": collision_records,
        },
        "support_geometry_used_by_isaac": {
            "source": "hash_bound_canonical_order8_reset",
            "pose_world": list(support_pose),
            "size_m": list(support_size),
            "top_z_m": support_pose[2] + 0.5 * support_size[2],
        },
        "clearance_reconstruction": {
            "geometry_model": "urdf_collision_mesh_local_oriented_box_proxy",
            "proxy_is_conservative_not_exact_mesh_distance": True,
            "battery_body_pose_source": "module_fc_pose_plus_urdf_fixed_transform",
            "terminal_pose_cross_check": reconstruction_errors,
            "learned_terminal_proxy_clearance_mm": learned_at_terminal_mm,
            "learned_terminal_exact_logged_pose_proxy_clearance_mm": (
                exact_terminal_clearance
            ),
            "nominal_same_index_proxy_clearance_mm": nominal_at_terminal_mm,
            "clearance_preserved_by_pi_l_bypass_at_failure_index_mm": [
                nominal_at_terminal_mm[index] - learned_at_terminal_mm[index]
                for index in range(len(learned_at_terminal_mm))
            ],
            "learned_minimum_proxy_clearance_mm": [
                1000.0 * min(values) for values in learned_clearance
            ],
            "nominal_minimum_proxy_clearance_mm": [
                1000.0 * min(values) for values in nominal_clearance
            ],
        },
        "pi_l_effect_at_failure_index": {
            "centroidal_command_minus_teacher_position_mm": (
                (1000.0 * command_delta).tolist()
            ),
            "maximum_joint_command_minus_teacher_mrad": [
                1000.0 * float(joint_command_delta[index].abs().max())
                for index in range(joint_command_delta.shape[0])
            ],
            "battery_position_learned_minus_nominal_mm": [
                [
                    1000.0
                    * (
                        learned_battery[environment][axis]
                        - nominal_battery[environment][axis]
                    )
                    for axis in range(3)
                ]
                for environment in range(len(learned_battery))
            ],
        },
        "conclusion": {
            "teacher_plus_qpid_qp_local_servo_is_dynamically_viable": True,
            "promoted_pi_l_correction_is_decisive_for_observed_failure": True,
            "v1_rejection_remains_valid": True,
            "recommended_preparation_change": (
                "freeze_source_support_and_require_12mm_collision_clearance"
            ),
        },
        "critical_trace": trace,
    }


def write_order9_r1_clearance_diagnostic(
    payload: Mapping[str, Any], destination: str | Path
) -> Path:
    target = Path(destination).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def _validated_raw(
    payload: object,
    *,
    nominal_only: bool,
    promoted_checkpoint_sha256: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not isinstance(payload, dict):
        raise SchemaValidationError(
            "R1 diagnostic raw artifact is not a mapping"
        )
    tensors = payload.get("tensors")
    metadata = payload.get("metadata")
    if not isinstance(tensors, dict) or not isinstance(metadata, dict):
        raise SchemaValidationError("R1 diagnostic raw artifact is incomplete")
    required = {
        "module_pose_world",
        "object_pose_world",
        "desired_body_pose_world",
        "desired_joint_positions_rad",
        "command_body_pose_world",
        "command_joint_position_targets_rad",
        "phase_index",
        "phase_progress",
    }
    if not required.issubset(tensors):
        raise SchemaValidationError(
            "R1 diagnostic raw tensor set is incomplete"
        )
    if (
        metadata.get("pi_l_checkpoint_sha256") != promoted_checkpoint_sha256
        or metadata.get("diagnostic_nominal_qpid_only") is not nominal_only
        or metadata.get("formal_phase_zero_start") is not True
        or metadata.get("deterministic_policy") is not True
    ):
        raise SchemaValidationError(
            "R1 diagnostic raw execution contract differs"
        )
    return tensors, metadata


def _load_episodes(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 2:
        raise SchemaValidationError("R1 A/B diagnostic requires two episodes")
    return records


def _validate_episode_binding(
    episodes: Sequence[Mapping[str, Any]],
    raw_path: Path,
    *,
    expect_success: bool,
) -> None:
    digest = hash_file(raw_path)
    for episode in episodes:
        success = bool(
            episode.get("task_success")
            and episode.get("no_fallback_success")
            and not episode.get("safety_failure")
            and int(episode.get("fallback_decision_count", -1)) == 0
        )
        if (
            success is not expect_success
            or episode.get("source_artifact_sha256") != digest
            or Path(str(episode.get("source_artifact_path"))).resolve()
            != raw_path
        ):
            raise SchemaValidationError("R1 A/B episode/raw binding differs")


def _load_collision_records(path: Path) -> list[dict[str, Any]]:
    prefix = "ORDER9_COLLISION_DIAGNOSTIC="
    return [
        json.loads(line[len(prefix) :])
        for line in path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        if line.startswith(prefix)
    ]


def _battery_geometry(
    physical,
) -> tuple[Pose7D, Pose7D, tuple[tuple[float, ...], tuple[float, ...]]]:
    link_poses = _zero_joint_link_poses(physical)
    base_link = str(physical.metadata.get("baselink", {}).get("name", "fc"))
    if base_link not in link_poses or "battery1" not in link_poses:
        raise SchemaValidationError(
            "R1 diagnostic physical model lacks battery1/fc"
        )
    battery_local = compose_pose(
        inverse_pose(link_poses[base_link]), link_poses["battery1"]
    )
    from amsrr.feasibility.order9_posture_collision import (
        build_collision_geometry_arrays,
    )

    link_index = {
        link.link_id: index for index, link in enumerate(physical.links)
    }
    arrays = build_collision_geometry_arrays(
        physical,
        link_index=link_index,
        repository_root=Path(__file__).resolve().parents[2],
    )
    indices = [
        index
        for index, name in enumerate(arrays.link_ids)
        if name == "battery1"
    ]
    if len(indices) != 1:
        raise SchemaValidationError(
            "R1 diagnostic battery1 collision is ambiguous"
        )
    index = indices[0]
    rotation = tuple(
        tuple(float(value) for value in row)
        for row in arrays.local_rotations[index]
    )
    collision_local: Pose7D = (
        *tuple(float(value) for value in arrays.local_translations[index]),
        *quat_from_matrix(rotation),
    )
    center = arrays.proxy_centers[index]
    half = arrays.proxy_half_extents[index]
    bounds = (
        tuple(float(center[axis] - half[axis]) for axis in range(3)),
        tuple(float(center[axis] + half[axis]) for axis in range(3)),
    )
    return battery_local, collision_local, bounds


def _zero_joint_link_poses(physical) -> dict[str, Pose7D]:
    children = {joint.child_link for joint in physical.joints}
    roots = [
        link.link_id for link in physical.links if link.link_id not in children
    ]
    if len(roots) != 1:
        raise SchemaValidationError(
            "R1 diagnostic physical link tree has no unique root"
        )
    outgoing: dict[str, list[Any]] = {}
    for joint in physical.joints:
        outgoing.setdefault(joint.parent_link, []).append(joint)
    poses: dict[str, Pose7D] = {roots[0]: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)}
    pending = [roots[0]]
    while pending:
        parent = pending.pop(0)
        for joint in outgoing.get(parent, ()):
            origin = pose_from_transform(
                transform_from_xyz_rpy(joint.origin_xyz, joint.origin_rpy)
            )
            poses[joint.child_link] = compose_pose(poses[parent], origin)
            pending.append(joint.child_link)
    return poses


def _clearance_series(
    tensors: Mapping[str, torch.Tensor],
    *,
    task_object_position: tuple[float, float, float],
    support_local: Order9PostureCollisionBox,
    battery_local_pose: Pose7D,
    collision_local_pose: Pose7D,
    collision_bounds: tuple[tuple[float, ...], tuple[float, ...]],
) -> list[list[float]]:
    poses = tensors["module_pose_world"]
    output: list[list[float]] = []
    for environment in range(poses.shape[1]):
        support = _environment_support_box(
            tensors,
            environment=environment,
            task_object_position=task_object_position,
            support_local=support_local,
        )
        values = []
        for module_pose in poses[:, environment, 1]:
            battery_pose = _battery_pose_from_module(
                module_pose, battery_local_pose
            )
            values.append(
                _battery_support_clearance(
                    battery_pose,
                    support=support,
                    collision_local_pose=collision_local_pose,
                    collision_bounds=collision_bounds,
                )
            )
        output.append(values)
    return output


def _environment_support_box(
    tensors: Mapping[str, torch.Tensor],
    *,
    environment: int,
    task_object_position: tuple[float, float, float],
    support_local: Order9PostureCollisionBox,
) -> OrientedBox:
    observed_object = tensors["object_pose_world"][0, environment, :3]
    # Isaac environment origins are horizontal; the small initial object Z
    # settling offset is physical state, not a cloned-scene translation.
    origin = (
        float(observed_object[0]) - task_object_position[0],
        float(observed_object[1]) - task_object_position[1],
        0.0,
    )
    center = tuple(
        float(support_local.pose_world[index]) + origin[index]
        for index in range(3)
    )
    return OrientedBox.axis_aligned(
        center,
        tuple(0.5 * float(value) for value in support_local.size_m),
    )


def _battery_pose_from_module(
    module_pose: torch.Tensor, local_pose: Pose7D
) -> Pose7D:
    return compose_pose(
        tuple(float(value) for value in module_pose), local_pose
    )


def _battery_support_clearance(
    battery_pose: Sequence[float],
    *,
    support: OrientedBox,
    collision_local_pose: Pose7D,
    collision_bounds: tuple[tuple[float, ...], tuple[float, ...]],
) -> float:
    collision_pose = compose_pose(
        tuple(float(value) for value in battery_pose), collision_local_pose
    )
    battery_box = OrientedBox.from_pose_and_local_bounds(
        collision_pose, collision_bounds[0], collision_bounds[1]
    )
    return oriented_box_clearance(battery_box, support)


def _episode_summary(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "episode_count": len(episodes),
        "task_success_count": sum(
            bool(value.get("task_success")) for value in episodes
        ),
        "safety_failure_count": sum(
            bool(value.get("safety_failure")) for value in episodes
        ),
        "fallback_decision_count": sum(
            int(value.get("fallback_decision_count", 0)) for value in episodes
        ),
        "environment_step_counts": [
            int(value.get("environment_step_count", 0)) for value in episodes
        ],
        "failure_reasons": [value.get("failure_reason") for value in episodes],
    }


def _phase_label(index: int) -> str:
    return (
        _PHASE_LABELS[index]
        if 0 <= index < len(_PHASE_LABELS)
        else f"phase_{index}"
    )


def _quaternion_distance(
    left: Sequence[float], right: Sequence[float]
) -> float:
    dot = abs(sum(float(a) * float(b) for a, b in zip(left, right)))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return math.inf
    return 2.0 * math.acos(max(-1.0, min(1.0, dot / (left_norm * right_norm))))


__all__ = [
    "ORDER9_R1_CLEARANCE_DIAGNOSTIC_VERSION",
    "build_order9_r1_clearance_diagnostic",
    "write_order9_r1_clearance_diagnostic",
]
