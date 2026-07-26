from __future__ import annotations

"""Exact-visual-mesh browser animation for saved C3 nominal trajectories."""

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from amsrr.feasibility.articulated_reachability import (
    base_pose_for_centroidal_target,
)
from amsrr.robot_model.whole_structure_kinematics import (
    WholeStructureKinematics,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.contact_candidates import ContactCandidateSet
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import InteractionKnot
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.order9_c3_nominal_trajectory import (
    Order9C3NominalTrajectory,
)
from amsrr.training.order9_c3_teacher import (
    build_order9_c3_posture_collision_object,
)
from amsrr.visualization.order9_c3_curation import (
    Order9C3MeshViewerArtifacts,
    Order9C3ViewerBox,
    Order9C3ViewerMarker,
    build_order9_c3_urdf_mesh_scene,
    order9_c3_urdf_root_pose_from_baselink_pose,
    render_order9_c3_mesh_viewer,
)


ORDER9_C3_NOMINAL_MESH_ANIMATION_VERSION = (
    "order9_c3_nominal_exact_visual_mesh_animation_v2_support_ground"
)
_GROUND_DISPLAY_SIZE_XY_M = 2.0
_GROUND_DISPLAY_THICKNESS_M = 0.01


def render_order9_c3_nominal_mesh_animation(
    *,
    result: Order9C3NominalTrajectory,
    task_spec: TaskSpec,
    physical_model: PhysicalModel,
    robot_urdf_path: str | Path,
    html_path: str | Path,
    mesh_library_path: str | Path,
    viewer_javascript_path: str | Path,
    bucket_id: str,
    frames_per_second: int = 10,
    source_metadata: Mapping[str, object] | None = None,
) -> Order9C3MeshViewerArtifacts:
    """Render an ideal-tracking mesh animation without Isaac/PhysX."""

    if not 1 <= frames_per_second <= 20:
        raise ValueError("frames_per_second must be in [1, 20]")
    morphology = result.selection_bundle.design_output.target_morphology
    candidates = result.selection_bundle.contact_candidate_set
    selected_candidates = _selected_candidates(result, candidates)
    object_ids = {
        value.target_entity_id for value in selected_candidates
    }
    if len(object_ids) != 1:
        raise SchemaValidationError(
            "C3 nominal animation requires one selected object"
        )
    object_id = next(iter(object_ids))
    object_spec = next(
        (value for value in task_spec.scene.objects if value.object_id == object_id),
        None,
    )
    if object_spec is None:
        raise SchemaValidationError(
            "C3 nominal animation selected object is absent from TaskSpec"
        )
    geometry = next(
        (
            value
            for value in task_spec.scene.geometry_library
            if value.geometry_id == object_spec.geometry_id
        ),
        None,
    )
    if geometry is None or geometry.geometry_type.value != "box":
        raise SchemaValidationError(
            "C3 nominal animation currently requires a box object"
        )
    raw_size = (geometry.primitive_params or {}).get("size_m")
    if not isinstance(raw_size, (list, tuple)) or len(raw_size) != 3:
        raise SchemaValidationError(
            "C3 nominal animation box lacks size_m"
        )
    object_size = tuple(
        float(raw_size[index]) * float(geometry.scale[index])
        for index in range(3)
    )
    collision_scene = build_order9_c3_posture_collision_object(task_spec)
    if len(collision_scene.environment_boxes) != 1:
        raise SchemaValidationError(
            "C3 nominal animation requires one bucket support"
        )
    support = collision_scene.environment_boxes[0]
    if collision_scene.ground_plane_z_m is None:
        raise SchemaValidationError(
            "C3 nominal animation requires a ground plane"
        )
    ground_pose = (
        float(support.pose_world[0]),
        float(support.pose_world[1]),
        float(collision_scene.ground_plane_z_m)
        - 0.5 * _GROUND_DISPLAY_THICKNESS_M,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    timeline = _sample_timeline(
        result.timeline,
        frames_per_second=frames_per_second,
    )
    scenes = []
    frame_records = []
    current_object_pose = tuple(object_spec.pose_world)
    kinematics = WholeStructureKinematics()
    for record in timeline:
        knot = InteractionKnot.from_dict(record["knot"])
        q = _joint_positions(knot)
        centroidal = knot.centroidal_target
        if (
            centroidal is None
            or centroidal.com_pos_world is None
            or centroidal.body_orientation_world is None
        ):
            raise SchemaValidationError(
                "C3 nominal animation frame lacks centroidal pose"
            )
        baselink_pose = base_pose_for_centroidal_target(
            morphology,
            physical_model,
            q,
            tuple(float(value) for value in centroidal.com_pos_world),
            tuple(
                float(value)
                for value in centroidal.body_orientation_world
            ),
            kinematics=kinematics,
        )
        root_pose = order9_c3_urdf_root_pose_from_baselink_pose(
            robot_urdf_path,
            baselink_pose,
        )
        scene = build_order9_c3_urdf_mesh_scene(
            robot_urdf_path,
            joint_positions_rad=q,
            root_pose_world=root_pose,
        )
        if scenes:
            _require_same_instance_identity(scenes[0], scene)
        scenes.append(scene)
        targets = [
            value
            for value in knot.object_targets
            if value.object_id == object_id
            and value.pose_target_world is not None
        ]
        if len(targets) > 1:
            raise SchemaValidationError(
                "C3 nominal animation frame has duplicate object targets"
            )
        if targets:
            current_object_pose = tuple(targets[0].pose_target_world)
        frame_records.append(
            {
                "frame_index": len(frame_records),
                "time_s": float(record["global_time_s"]),
                "phase": str(record["phase"]),
                "window_index": int(record["window_index"]),
                "window_local_time_s": float(
                    record["window_local_time_s"]
                ),
                "model_matrices": [
                    list(value["model_matrix"])
                    for value in scene.instances
                ],
                "label_positions": [
                    list(value["position_world"])
                    for value in scene.module_labels
                ],
                "box_poses": [
                    list(current_object_pose),
                    list(support.pose_world),
                    list(ground_pose),
                ],
            }
        )
    first_scene = scenes[0]
    markers = tuple(
        Order9C3ViewerMarker(
            marker_id=f"candidate-{candidate.candidate_id}",
            label=f"P{candidate.candidate_id}",
            position_world=tuple(candidate.contact_pose_world[:3]),
            direction_world=tuple(candidate.normal_world),
            selected=True,
            kind="contact",
        )
        for candidate in selected_candidates
    )
    animation = {
        "animation_version": ORDER9_C3_NOMINAL_MESH_ANIMATION_VERSION,
        "frames_per_second": frames_per_second,
        "frame_count": len(frame_records),
        "duration_s": result.duration_s,
        "frames": frame_records,
    }
    return render_order9_c3_mesh_viewer(
        urdf_scene=first_scene,
        html_path=html_path,
        mesh_library_path=mesh_library_path,
        viewer_javascript_path=viewer_javascript_path,
        title=f"Order 9 C3 nominal trajectory: {bucket_id}",
        subtitle=(
            f"{len(morphology.modules)} modules | "
            f"{len(result.windows)} rolling windows | "
            f"{result.duration_s:.2f} s"
        ),
        markers=markers,
        boxes=(
            Order9C3ViewerBox(
                object_id=object_id,
                pose_world=tuple(object_spec.pose_world),
                size_m=object_size,
                color_rgba=(0.95, 0.66, 0.12, 0.45),
            ),
            Order9C3ViewerBox(
                object_id=support.box_id,
                pose_world=support.pose_world,
                size_m=support.size_m,
                color_rgba=(0.34, 0.34, 0.38, 0.45),
            ),
            Order9C3ViewerBox(
                object_id="ground_halfspace_display_patch",
                pose_world=ground_pose,
                size_m=(
                    _GROUND_DISPLAY_SIZE_XY_M,
                    _GROUND_DISPLAY_SIZE_XY_M,
                    _GROUND_DISPLAY_THICKNESS_M,
                ),
                color_rgba=(0.20, 0.48, 0.80, 0.12),
            ),
        ),
        metadata={
            **dict(source_metadata or {}),
            "bucket_id": bucket_id,
            "animation_version": (
                ORDER9_C3_NOMINAL_MESH_ANIMATION_VERSION
            ),
            "ideal_tracking": True,
            "isaac_dynamics": False,
            "proxy_collision_validation_status": (
                result.proxy_collision_validation_status
            ),
            "collision_obstacles": [
                collision_scene.object_id,
                support.box_id,
            ],
            "ground_halfspace_minimum_z_m": (
                collision_scene.ground_plane_z_m
            ),
            "ground_display_is_finite_patch_only": True,
        },
        animation=animation,
        semantic_scope=(
            "exact URDF visual meshes replaying the saved ideal-tracking "
            "pi_H-teacher plus deterministic-IK nominal trajectory, generated "
            "with convex self/object/bucket-support and ground-halfspace "
            "collision constraints; not Isaac dynamics, exact visual-mesh "
            "collision admission, or learned-policy success"
        ),
    )


def _selected_candidates(
    result: Order9C3NominalTrajectory,
    candidates: ContactCandidateSet,
) -> tuple:
    final = result.windows[-1].plan.trajectory.knots[-1]
    candidate_by_id = {
        value.candidate_id: value for value in candidates.candidates
    }
    selected = []
    for assignment in final.contact_assignments:
        if assignment.schedule_state not in {
            "attach",
            "maintain",
            "slide",
        }:
            continue
        candidate = candidate_by_id.get(assignment.candidate_id)
        if candidate is None:
            raise SchemaValidationError(
                "C3 nominal animation assignment references unknown candidate"
            )
        selected.append(candidate)
    if len(selected) != 2:
        raise SchemaValidationError(
            "C3 nominal animation requires two final grasp candidates"
        )
    return tuple(selected)


def _joint_positions(knot: InteractionKnot) -> dict[str, float]:
    posture = knot.posture_target
    if posture is None or posture.joint_pos_target is None:
        raise SchemaValidationError(
            "C3 nominal animation frame lacks resolved joint positions"
        )
    return {
        str(key): float(value)
        for key, value in posture.joint_pos_target.items()
    }


def _sample_timeline(
    records: Sequence[dict[str, Any]],
    *,
    frames_per_second: int,
) -> tuple[dict[str, Any], ...]:
    if not records:
        raise SchemaValidationError("C3 nominal animation timeline is empty")
    period = 1.0 / float(frames_per_second)
    selected = [records[0]]
    next_time = float(records[0]["global_time_s"]) + period
    for record in records[1:-1]:
        time_s = float(record["global_time_s"])
        if time_s + 1.0e-9 < next_time:
            continue
        selected.append(record)
        next_time = time_s + period
    if records[-1] is not selected[-1]:
        selected.append(records[-1])
    return tuple(selected)


def _require_same_instance_identity(first, current) -> None:
    first_identity = [
        (
            value["instance_id"],
            value["mesh_key"],
            value["layer"],
        )
        for value in first.instances
    ]
    current_identity = [
        (
            value["instance_id"],
            value["mesh_key"],
            value["layer"],
        )
        for value in current.instances
    ]
    if current_identity != first_identity:
        raise SchemaValidationError(
            "C3 nominal animation mesh instance identity changed by frame"
        )
    if [value["label_id"] for value in current.module_labels] != [
        value["label_id"] for value in first.module_labels
    ]:
        raise SchemaValidationError(
            "C3 nominal animation module-label identity changed by frame"
        )


__all__ = [
    "ORDER9_C3_NOMINAL_MESH_ANIMATION_VERSION",
    "render_order9_c3_nominal_mesh_animation",
]
