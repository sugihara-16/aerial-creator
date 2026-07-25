from __future__ import annotations

"""Offline animation for the deterministic Order 9 articulated teacher."""

import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from amsrr.feasibility.articulated_reachability import (
    base_pose_for_centroidal_target,
    resolve_mesh_backed_anchor_references,
)
from amsrr.geometry.pose_math import (
    compose_pose,
    inverse_pose,
    matvec,
    transform_from_pose,
    transpose,
)
from amsrr.robot_model.whole_structure_kinematics import (
    MeshBackedAnchorReference,
    WholeStructureKinematics,
)
from amsrr.schemas.common import Pose7D, SchemaValidationError
from amsrr.schemas.contact_candidates import (
    ContactCandidate,
    ContactCandidateSet,
)
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import (
    ContactAssignment,
    ContactWrenchTrajectory,
    InteractionKnot,
)
from amsrr.schemas.task_spec import GeometrySpec, ObjectSpec, TaskSpec
from amsrr.training.order9_articulated_teacher import (
    Order9ArticulatedTeacherPlan,
)
from amsrr.training.order9_c3_teacher import Order9C3TeacherBundle
from amsrr.utils.hashing import stable_hash


ORDER9_ARTICULATED_TEACHER_ANIMATION_VERSION = (
    "order9_articulated_teacher_animation_v1"
)
_ACTIVE_CONTACT_STATES = frozenset({"attach", "maintain", "slide"})


@dataclass(frozen=True)
class Order9TeacherAnimationArtifacts:
    html_path: Path
    summary_path: Path
    frame_count: int
    horizon_s: float
    grasp_knot_index: int
    animation_version: str = ORDER9_ARTICULATED_TEACHER_ANIMATION_VERSION


def build_order9_articulated_teacher_animation_payload(
    *,
    task_spec: TaskSpec,
    morphology: MorphologyGraph,
    contact_candidate_set: ContactCandidateSet,
    trajectory_plan: Order9ArticulatedTeacherPlan,
    physical_model: PhysicalModel,
    selected_surface_port_ids: Sequence[int] | None = None,
    frames_per_second: int = 20,
    source_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Sample the teacher command trajectory through whole-structure FK."""

    if not 1 <= frames_per_second <= 60:
        raise ValueError("frames_per_second must be in [1, 60]")
    task_spec.validate()
    morphology.validate()
    contact_candidate_set.validate()
    physical_model.validate()
    trajectory = trajectory_plan.trajectory
    trajectory.validate()
    if len(trajectory.knots) < 2:
        raise SchemaValidationError(
            "teacher animation requires at least two trajectory knots"
        )

    grasp_knot_index, selected_assignments = _selected_grasp_assignments(
        trajectory
    )
    candidate_by_id = {
        candidate.candidate_id: candidate
        for candidate in contact_candidate_set.candidates
    }
    selected_candidates = []
    for assignment in selected_assignments:
        candidate = candidate_by_id.get(assignment.candidate_id)
        if candidate is None:
            raise SchemaValidationError(
                "teacher animation assignment references an unknown candidate"
            )
        selected_candidates.append(candidate)
    target_ids = {
        candidate.target_entity_id for candidate in selected_candidates
    }
    if len(target_ids) != 1:
        raise SchemaValidationError(
            "teacher animation currently requires one grasped object"
        )
    object_id = next(iter(target_ids))
    object_spec = _object_by_id(task_spec, object_id)
    geometry = _geometry_by_id(task_spec, object_spec.geometry_id)
    selected_anchor_ids = tuple(
        assignment.anchor_id for assignment in selected_assignments
    )
    anchor_references = resolve_mesh_backed_anchor_references(
        morphology,
        physical_model,
        selected_anchor_ids,
    )
    reference_by_anchor_id = {
        reference.anchor.anchor_id: reference
        for reference in anchor_references
    }
    surface_ids = tuple(
        reference_by_anchor_id[
            assignment.anchor_id
        ].surface.port_global_id
        for assignment in selected_assignments
    )
    if (
        selected_surface_port_ids is not None
        and sorted(surface_ids)
        != sorted(int(value) for value in selected_surface_port_ids)
    ):
        raise SchemaValidationError(
            "teacher animation selected surfaces differ from resolved anchors"
        )

    times = _sample_times(trajectory, frames_per_second)
    kinematics = WholeStructureKinematics()
    frames = [
        _sample_frame(
            time_s=time_s,
            trajectory=trajectory,
            morphology=morphology,
            physical_model=physical_model,
            kinematics=kinematics,
            anchor_references=anchor_references,
            selected_assignments=selected_assignments,
            candidate_by_id=candidate_by_id,
            object_spec=object_spec,
        )
        for time_s in times
    ]
    grasp_knot = trajectory.knots[grasp_knot_index]
    grasp_q = _joint_positions(grasp_knot)
    scene_center, scene_radius = _scene_bounds(frames, geometry)
    summary = {
        "animation_version": ORDER9_ARTICULATED_TEACHER_ANIMATION_VERSION,
        "semantic_scope": (
            "deterministic teacher command and kinematic FK only; "
            "not Isaac dynamics or grasp-success evidence"
        ),
        "source_metadata": dict(source_metadata or {}),
        "task": {
            "task_id": task_spec.task_id,
            "task_type": task_spec.task_type.value,
        },
        "morphology": {
            "graph_id": morphology.graph_id,
            "morphology_hash": morphology.stable_hash(),
            "module_count": len(morphology.modules),
            "base_module_id": morphology.base_module_id,
            "modules": [
                {
                    "module_id": module.module_id,
                    "role_id": module.role_id,
                    "is_base": module.is_base,
                }
                for module in sorted(
                    morphology.modules, key=lambda value: value.module_id
                )
            ],
            "dock_edges": [
                {
                    "edge_id": edge.edge_id,
                    "src_module_id": edge.src_module_id,
                    "src_port_id": edge.src_port_id,
                    "dst_module_id": edge.dst_module_id,
                    "dst_port_id": edge.dst_port_id,
                }
                for edge in sorted(
                    morphology.dock_edges, key=lambda value: value.edge_id
                )
            ],
        },
        "object": {
            "object_id": object_spec.object_id,
            "geometry_id": object_spec.geometry_id,
            "geometry_type": geometry.geometry_type.value,
            "primitive_params": dict(geometry.primitive_params or {}),
            "scale": list(geometry.scale),
            "mass_kg": object_spec.mass_kg,
            "density_kg_m3": object_spec.density_kg_m3,
            "initial_pose_world": list(object_spec.pose_world),
            "goal_pose_world": _goal_pose(task_spec, object_spec.object_id),
        },
        "contacts": [
            {
                "anchor_id": assignment.anchor_id,
                "candidate_id": assignment.candidate_id,
                "slot_id": assignment.slot_id,
                "surface_port_id": surface_ids[index],
                "module_id": reference_by_anchor_id[
                    assignment.anchor_id
                ].anchor.module_id,
                "candidate_contact_pose_world": list(
                    selected_candidates[index].contact_pose_world
                ),
                "candidate_normal_world": list(
                    selected_candidates[index].normal_world
                ),
            }
            for index, assignment in enumerate(selected_assignments)
        ],
        "ik": {
            "solver_version": trajectory_plan.ik_solution.solver_version,
            "feasible": trajectory_plan.ik_solution.feasible,
            "iterations": trajectory_plan.ik_solution.iterations,
            "maximum_position_error_m": (
                trajectory_plan.ik_solution.maximum_position_error_m
            ),
            "maximum_normal_error_rad": (
                trajectory_plan.ik_solution.maximum_normal_error_rad
            ),
            "candidate_group_id": trajectory_plan.candidate_group_id,
        },
        "trajectory": {
            "teacher_version": trajectory_plan.teacher_version,
            "trajectory_hash": stable_hash(trajectory.to_dict()),
            "horizon_s": trajectory.horizon_s,
            "dt_s": trajectory.dt_s,
            "grasp_knot_index": grasp_knot_index,
            "grasp_time_s": grasp_knot.t_rel_s,
            "knots": [
                {
                    "knot_index": index,
                    "time_s": knot.t_rel_s,
                    "phase_label": _phase_label(knot, index),
                    "schedule_states": sorted(
                        {
                            assignment.schedule_state
                            for assignment in knot.contact_assignments
                        }
                    ),
                }
                for index, knot in enumerate(trajectory.knots)
            ],
        },
        "grasp_joint_positions": {
            joint_id: {
                "rad": float(value),
                "deg": math.degrees(float(value)),
            }
            for joint_id, value in sorted(grasp_q.items())
        },
        "render": {
            "frames_per_second": frames_per_second,
            "frame_count": len(frames),
            "scene_center_world": list(scene_center),
            "scene_radius_m": scene_radius,
            "module_arm_radius_m": _module_render_radius(physical_model),
            "geometry_fidelity": (
                "module root frames and schematic Holon arms; "
                "object primitive uses TaskSpec dimensions"
            ),
        },
    }
    return {
        "animation_version": ORDER9_ARTICULATED_TEACHER_ANIMATION_VERSION,
        "summary": summary,
        "frames": frames,
    }


def write_order9_articulated_teacher_animation(
    payload: Mapping[str, Any],
    *,
    html_path: str | Path,
    summary_path: str | Path | None = None,
) -> Order9TeacherAnimationArtifacts:
    """Write one self-contained interactive HTML and its numeric summary."""

    if payload.get("animation_version") != (
        ORDER9_ARTICULATED_TEACHER_ANIMATION_VERSION
    ):
        raise SchemaValidationError("teacher animation payload version mismatch")
    summary = payload.get("summary")
    frames = payload.get("frames")
    if not isinstance(summary, dict) or not isinstance(frames, list) or not frames:
        raise SchemaValidationError("teacher animation payload is incomplete")
    output_html = Path(html_path)
    output_summary = (
        Path(summary_path)
        if summary_path is not None
        else output_html.with_suffix(".summary.json")
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    task = summary["task"]
    morphology = summary["morphology"]
    title = (
        f"Order 9 articulated teacher: {task['task_id']} "
        f"({morphology['module_count']} modules)"
    )
    embedded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).replace("</", "<\\/")
    page = (
        _HTML_TEMPLATE.replace("__DOCUMENT_TITLE__", html.escape(title))
        .replace("__PAYLOAD_JSON__", embedded)
    )
    output_html.write_text(page, encoding="utf-8")
    output_summary.write_text(
        json.dumps(summary, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return Order9TeacherAnimationArtifacts(
        html_path=output_html,
        summary_path=output_summary,
        frame_count=len(frames),
        horizon_s=float(summary["trajectory"]["horizon_s"]),
        grasp_knot_index=int(summary["trajectory"]["grasp_knot_index"]),
    )


def render_order9_c3_teacher_animation(
    *,
    task_spec: TaskSpec,
    bundle: Order9C3TeacherBundle,
    physical_model: PhysicalModel,
    html_path: str | Path,
    summary_path: str | Path | None = None,
    frames_per_second: int = 20,
    source_metadata: Mapping[str, Any] | None = None,
) -> Order9TeacherAnimationArtifacts:
    payload = build_order9_articulated_teacher_animation_payload(
        task_spec=task_spec,
        morphology=bundle.design_output.target_morphology,
        contact_candidate_set=bundle.contact_candidate_set,
        trajectory_plan=bundle.trajectory_plan,
        physical_model=physical_model,
        selected_surface_port_ids=bundle.selected_surface_port_ids,
        frames_per_second=frames_per_second,
        source_metadata=source_metadata,
    )
    return write_order9_articulated_teacher_animation(
        payload,
        html_path=html_path,
        summary_path=summary_path,
    )


def _selected_grasp_assignments(
    trajectory: ContactWrenchTrajectory,
) -> tuple[int, tuple[ContactAssignment, ...]]:
    for index in range(len(trajectory.knots) - 1, -1, -1):
        knot = trajectory.knots[index]
        maintained = tuple(
            assignment
            for assignment in knot.contact_assignments
            if assignment.schedule_state == "maintain"
        )
        if maintained:
            return index, maintained
    fallback: tuple[int, tuple[ContactAssignment, ...]] | None = None
    for index, knot in enumerate(trajectory.knots):
        active = tuple(
            assignment
            for assignment in knot.contact_assignments
            if assignment.schedule_state in _ACTIVE_CONTACT_STATES
        )
        if active:
            fallback = (index, active)
    if fallback is not None:
        return fallback
    raise SchemaValidationError(
        "teacher animation found no active grasp assignment"
    )


def _object_by_id(task_spec: TaskSpec, object_id: str) -> ObjectSpec:
    try:
        return next(
            value
            for value in task_spec.scene.objects
            if value.object_id == object_id
        )
    except StopIteration as error:
        raise SchemaValidationError(
            f"teacher animation object {object_id!r} is absent"
        ) from error


def _geometry_by_id(task_spec: TaskSpec, geometry_id: str) -> GeometrySpec:
    try:
        return next(
            value
            for value in task_spec.scene.geometry_library
            if value.geometry_id == geometry_id
        )
    except StopIteration as error:
        raise SchemaValidationError(
            f"teacher animation geometry {geometry_id!r} is absent"
        ) from error


def _sample_times(
    trajectory: ContactWrenchTrajectory,
    frames_per_second: int,
) -> tuple[float, ...]:
    horizon = float(trajectory.horizon_s)
    count = max(2, int(math.ceil(horizon * frames_per_second)) + 1)
    values = {
        min(horizon, float(index) / float(frames_per_second))
        for index in range(count)
    }
    values.add(horizon)
    values.update(float(knot.t_rel_s) for knot in trajectory.knots)
    return tuple(sorted(values))


def _sample_frame(
    *,
    time_s: float,
    trajectory: ContactWrenchTrajectory,
    morphology: MorphologyGraph,
    physical_model: PhysicalModel,
    kinematics: WholeStructureKinematics,
    anchor_references: Sequence[MeshBackedAnchorReference],
    selected_assignments: Sequence[ContactAssignment],
    candidate_by_id: Mapping[int, ContactCandidate],
    object_spec: ObjectSpec,
) -> dict[str, Any]:
    left_index, right_index, alpha = _trajectory_segment(
        trajectory, time_s
    )
    left = trajectory.knots[left_index]
    right = trajectory.knots[right_index]
    q = _interpolate_joint_maps(
        _joint_positions(left),
        _joint_positions(right),
        alpha,
    )
    centroidal_pose = _interpolate_pose(
        _centroidal_pose(left),
        _centroidal_pose(right),
        alpha,
    )
    object_pose = _interpolate_pose(
        _object_pose(left, object_spec),
        _object_pose(right, object_spec),
        alpha,
    )
    base_pose = base_pose_for_centroidal_target(
        morphology,
        physical_model,
        q,
        centroidal_pose[:3],
        centroidal_pose[3:],
        kinematics=kinematics,
    )
    fk = kinematics.forward(
        morphology,
        physical_model,
        q,
        base_pose,
        anchor_references,
    )
    schedule_by_key = {
        (assignment.anchor_id, assignment.candidate_id): (
            assignment.schedule_state
        )
        for assignment in left.contact_assignments
    }
    contacts = []
    for assignment in selected_assignments:
        candidate = candidate_by_id[assignment.candidate_id]
        actual_pose = fk.anchor_poses_world[assignment.anchor_id]
        target_pose, target_normal = _contact_target_at_object_pose(
            candidate.contact_pose_world,
            candidate.normal_world,
            initial_object_pose=object_spec.pose_world,
            object_pose=object_pose,
        )
        actual_axis = _pose_x_axis(actual_pose)
        position_error = math.sqrt(
            sum(
                (
                    float(target_pose[axis])
                    - float(actual_pose[axis])
                )
                ** 2
                for axis in range(3)
            )
        )
        schedule_state = schedule_by_key.get(
            (assignment.anchor_id, assignment.candidate_id),
            "inactive",
        )
        contacts.append(
            {
                "anchor_id": assignment.anchor_id,
                "candidate_id": assignment.candidate_id,
                "schedule_state": schedule_state,
                "active": schedule_state in _ACTIVE_CONTACT_STATES,
                "actual_pose_world": list(actual_pose),
                "actual_outward_axis_world": list(actual_axis),
                "target_pose_world": list(target_pose),
                "target_normal_world": list(target_normal),
                "position_error_m": position_error,
            }
        )
    return {
        "time_s": time_s,
        "left_knot_index": left_index,
        "right_knot_index": right_index,
        "phase_label": _phase_label(left, left_index),
        "segment_alpha": alpha,
        "joint_positions_rad": {
            key: float(value) for key, value in sorted(q.items())
        },
        "base_pose_world": list(base_pose),
        "centroidal_pose_world": list(centroidal_pose),
        "object_pose_world": list(object_pose),
        "modules": [
            {
                "module_id": module_id,
                "pose_world": list(fk.module_root_poses_world[module_id]),
            }
            for module_id in sorted(fk.module_root_poses_world)
        ],
        "contacts": contacts,
    }


def _trajectory_segment(
    trajectory: ContactWrenchTrajectory,
    time_s: float,
) -> tuple[int, int, float]:
    times = [float(knot.t_rel_s) for knot in trajectory.knots]
    left = 0
    for index, value in enumerate(times):
        if value <= time_s + 1.0e-12:
            left = index
        else:
            break
    right = min(left + 1, len(times) - 1)
    elapsed = times[right] - times[left]
    alpha = (
        0.0
        if right == left or elapsed <= 1.0e-12
        else min(1.0, max(0.0, (time_s - times[left]) / elapsed))
    )
    return left, right, alpha


def _joint_positions(knot: InteractionKnot) -> dict[str, float]:
    posture = knot.posture_target
    if posture is None or posture.joint_pos_target is None:
        raise SchemaValidationError(
            "teacher animation requires complete joint positions at every knot"
        )
    return {
        str(key): float(value)
        for key, value in posture.joint_pos_target.items()
    }


def _centroidal_pose(knot: InteractionKnot) -> Pose7D:
    target = knot.centroidal_target
    if (
        target is None
        or target.com_pos_world is None
        or target.body_orientation_world is None
    ):
        raise SchemaValidationError(
            "teacher animation requires a complete centroidal pose at every knot"
        )
    return (
        *tuple(float(value) for value in target.com_pos_world),
        *tuple(float(value) for value in target.body_orientation_world),
    )


def _object_pose(knot: InteractionKnot, object_spec: ObjectSpec) -> Pose7D:
    for target in knot.object_targets:
        if (
            target.object_id == object_spec.object_id
            and target.pose_target_world is not None
        ):
            return tuple(float(value) for value in target.pose_target_world)
    return object_spec.pose_world


def _interpolate_joint_maps(
    left: Mapping[str, float],
    right: Mapping[str, float],
    alpha: float,
) -> dict[str, float]:
    if set(left) != set(right):
        raise SchemaValidationError(
            "teacher animation joint set changes between knots"
        )
    return {
        key: (1.0 - alpha) * float(left[key]) + alpha * float(right[key])
        for key in sorted(left)
    }


def _interpolate_pose(left: Pose7D, right: Pose7D, alpha: float) -> Pose7D:
    position = tuple(
        (1.0 - alpha) * float(left[index])
        + alpha * float(right[index])
        for index in range(3)
    )
    quaternion = _slerp(left[3:], right[3:], alpha)
    return (*position, *quaternion)


def _slerp(
    left: Sequence[float],
    right: Sequence[float],
    alpha: float,
) -> tuple[float, float, float, float]:
    first = _normalized_quaternion(left)
    second = _normalized_quaternion(right)
    dot = sum(first[index] * second[index] for index in range(4))
    if dot < 0.0:
        second = tuple(-value for value in second)
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return _normalized_quaternion(
            tuple(
                (1.0 - alpha) * first[index] + alpha * second[index]
                for index in range(4)
            )
        )
    angle = math.acos(dot)
    sine = math.sin(angle)
    left_scale = math.sin((1.0 - alpha) * angle) / sine
    right_scale = math.sin(alpha * angle) / sine
    return tuple(
        left_scale * first[index] + right_scale * second[index]
        for index in range(4)
    )


def _normalized_quaternion(
    values: Sequence[float],
) -> tuple[float, float, float, float]:
    parsed = tuple(float(value) for value in values)
    norm = math.sqrt(sum(value * value for value in parsed))
    if norm <= 1.0e-12:
        raise SchemaValidationError("teacher animation received a zero quaternion")
    return tuple(value / norm for value in parsed)


def _contact_target_at_object_pose(
    contact_pose_world: Pose7D,
    normal_world: Sequence[float],
    *,
    initial_object_pose: Pose7D,
    object_pose: Pose7D,
) -> tuple[Pose7D, tuple[float, float, float]]:
    object_to_contact = compose_pose(
        inverse_pose(initial_object_pose),
        contact_pose_world,
    )
    target_pose = compose_pose(object_pose, object_to_contact)
    initial_rotation = transform_from_pose(initial_object_pose).rotation
    object_rotation = transform_from_pose(object_pose).rotation
    local_normal = matvec(transpose(initial_rotation), normal_world)
    target_normal = matvec(object_rotation, local_normal)
    return target_pose, tuple(float(value) for value in target_normal)


def _pose_x_axis(pose: Pose7D) -> tuple[float, float, float]:
    rotation = transform_from_pose(pose).rotation
    return (
        float(rotation[0][0]),
        float(rotation[1][0]),
        float(rotation[2][0]),
    )


def _phase_label(knot: InteractionKnot, index: int) -> str:
    for guard in knot.guard_conditions:
        value = guard.get("phase_label")
        if isinstance(value, str) and value:
            return value
    states = sorted(
        {
            assignment.schedule_state
            for assignment in knot.contact_assignments
        }
    )
    return "+".join(states) if states else f"knot_{index}"


def _goal_pose(task_spec: TaskSpec, object_id: str) -> list[float] | None:
    for goal in task_spec.goals:
        if (
            goal.target_entity_id == object_id
            and goal.target_pose_world is not None
        ):
            return list(goal.target_pose_world)
    return None


def _module_render_radius(physical_model: PhysicalModel) -> float:
    distances = [
        math.sqrt(sum(float(value) ** 2 for value in port.local_pose[:3]))
        for port in physical_model.dock_ports
    ]
    maximum = max(distances, default=0.20)
    return min(0.32, max(0.16, maximum))


def _scene_bounds(
    frames: Sequence[Mapping[str, Any]],
    geometry: GeometrySpec,
) -> tuple[tuple[float, float, float], float]:
    points = [
        tuple(float(value) for value in module["pose_world"][:3])
        for frame in frames
        for module in frame["modules"]
    ]
    points.extend(
        tuple(float(value) for value in frame["object_pose_world"][:3])
        for frame in frames
    )
    points.extend(
        tuple(float(value) for value in contact["target_pose_world"][:3])
        for frame in frames
        for contact in frame["contacts"]
    )
    center = tuple(
        (min(point[axis] for point in points)
         + max(point[axis] for point in points))
        / 2.0
        for axis in range(3)
    )
    object_extent = _geometry_extent(geometry)
    radius = max(
        (
            math.sqrt(
                sum((point[axis] - center[axis]) ** 2 for axis in range(3))
            )
            for point in points
        ),
        default=1.0,
    )
    return center, max(0.5, radius + object_extent)


def _geometry_extent(geometry: GeometrySpec) -> float:
    params = geometry.primitive_params or {}
    if geometry.geometry_type.value == "box":
        values = params.get("size_m", (0.2, 0.2, 0.2))
        return 0.5 * math.sqrt(sum(float(value) ** 2 for value in values))
    if geometry.geometry_type.value == "sphere":
        return float(params.get("radius_m", 0.1))
    return max(
        float(params.get("radius_m", 0.1)),
        0.5 * float(params.get("height_m", params.get("length_m", 0.2))),
    )


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__DOCUMENT_TITLE__</title>
<style>
:root{color-scheme:dark;--bg:#0d1117;--panel:#161b22;--line:#30363d;--text:#e6edf3;--muted:#8b949e;--accent:#58a6ff;--target:#ff7b72;--ok:#3fb950}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif}
header{padding:12px 18px;border-bottom:1px solid var(--line);display:flex;gap:20px;align-items:center;flex-wrap:wrap}
h1{font-size:17px;margin:0}.scope{color:#f2cc60;font-size:12px}
main{display:grid;grid-template-columns:minmax(560px,1fr) 390px;height:calc(100vh - 61px)}
.viewer{position:relative;min-height:520px}canvas{width:100%;height:100%;display:block;background:radial-gradient(circle at 50% 45%,#182233,#080b10 72%)}
.overlay{position:absolute;left:14px;top:12px;padding:8px 10px;background:#0d1117d9;border:1px solid var(--line);border-radius:6px;pointer-events:none}
.overlay b{color:var(--accent)}.legend{position:absolute;left:14px;bottom:68px;padding:7px 10px;background:#0d1117d9;border:1px solid var(--line);border-radius:6px;color:var(--muted)}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin:0 4px 0 10px}.dot:first-child{margin-left:0}
.controls{position:absolute;left:14px;right:14px;bottom:12px;background:#0d1117e8;border:1px solid var(--line);border-radius:7px;padding:8px;display:grid;grid-template-columns:auto 1fr auto auto;gap:9px;align-items:center}
button,select{background:#21262d;color:var(--text);border:1px solid #484f58;border-radius:5px;padding:5px 9px}input[type=range]{width:100%}
aside{overflow:auto;border-left:1px solid var(--line);background:var(--panel);padding:14px}
h2{font-size:14px;margin:13px 0 7px;color:#79c0ff}h2:first-child{margin-top:0}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:7px}.card{border:1px solid var(--line);border-radius:6px;padding:7px;background:#0d1117}.card span{display:block;color:var(--muted);font-size:11px}.card strong{font-size:13px}
table{border-collapse:collapse;width:100%;font:11px ui-monospace,monospace}th,td{padding:4px 5px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left;word-break:break-all}
.current{color:#ffa657}.small{font-size:11px;color:var(--muted);word-break:break-word}.topology{font:12px ui-monospace,monospace}
@media(max-width:950px){main{grid-template-columns:1fr;height:auto}.viewer{height:70vh}aside{border-left:0;border-top:1px solid var(--line)}}
</style>
</head>
<body>
<header><h1>__DOCUMENT_TITLE__</h1><div class="scope">運動学的teacher指令の可視化です。Isaac物理・把持成功の証拠ではありません。</div></header>
<main>
<section class="viewer">
<canvas id="view"></canvas>
<div class="overlay"><div><b id="phase"></b></div><div id="time"></div><div id="contactError"></div></div>
<div class="legend"><span class="dot" style="background:#58a6ff"></span>module <span class="dot" style="background:#ff7b72"></span>contact target <span class="dot" style="background:#3fb950"></span>active anchor</div>
<div class="controls"><button id="play">▶ Play</button><input id="timeline" type="range" min="0" value="0" step="1"><select id="speed"><option value=".25">0.25×</option><option value=".5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select><button id="reset">Reset view</button></div>
</section>
<aside>
<h2>Condition</h2><div class="cards" id="cards"></div>
<h2>Morphology</h2><div class="topology" id="topology"></div>
<h2>Selected contacts</h2><div class="small" id="contacts"></div>
<h2>Joint targets</h2><div class="small">current command / grasp command</div><table><thead><tr><th>joint</th><th>now [deg]</th><th>grasp [deg]</th></tr></thead><tbody id="joints"></tbody></table>
<h2>Interaction</h2><div class="small">Drag: orbit camera · Wheel: zoom · Slider: teacher time</div>
</aside>
</main>
<script>
"use strict";
const payload=__PAYLOAD_JSON__;
const summary=payload.summary,frames=payload.frames;
const canvas=document.getElementById("view"),ctx=canvas.getContext("2d");
const timeline=document.getElementById("timeline"),playButton=document.getElementById("play");
timeline.max=String(frames.length-1);
let index=0,playing=false,lastStamp=null,playTime=0;
let yaw=-0.72,pitch=0.48,zoom=1,dragging=false,lastPointer=[0,0];
const center=summary.render.scene_center_world,radius=summary.render.scene_radius_m;
const moduleRadius=summary.render.module_arm_radius_m;
const edgeList=summary.morphology.dock_edges;
const graspJoints=summary.grasp_joint_positions;
const jointIds=Object.keys(graspJoints);

function card(label,value){return `<div class="card"><span>${label}</span><strong>${value}</strong></div>`}
document.getElementById("cards").innerHTML=
 card("Modules",summary.morphology.module_count)+card("Object",summary.object.geometry_type)+
 card("Size",objectSizeLabel())+card("Mass",`${Number(summary.object.mass_kg).toFixed(3)} kg`)+
 card("IK iterations",summary.ik.iterations)+card("IK pos. error",`${(1000*summary.ik.maximum_position_error_m).toFixed(3)} mm`);
document.getElementById("topology").innerHTML=edgeList.map(e=>`M${e.src_module_id}:P${e.src_port_id} → M${e.dst_module_id}:P${e.dst_port_id}`).join("<br>");
document.getElementById("contacts").innerHTML=summary.contacts.map(c=>`A${c.anchor_id} / M${c.module_id} / surface P${c.surface_port_id} / candidate ${c.candidate_id}`).join("<br>");
document.getElementById("joints").innerHTML=jointIds.map(id=>`<tr><td>${id}</td><td class="current" data-joint="${id}"></td><td>${graspJoints[id].deg.toFixed(2)}</td></tr>`).join("");

function objectSizeLabel(){
 const p=summary.object.primitive_params;
 if(Array.isArray(p.size_m))return p.size_m.map(v=>Number(v).toFixed(3)).join(" × ")+" m";
 if(p.radius_m!==undefined)return `r=${Number(p.radius_m).toFixed(3)} m`;
 return "see TaskSpec";
}
function resize(){
 const ratio=window.devicePixelRatio||1,box=canvas.getBoundingClientRect();
 canvas.width=Math.max(1,Math.floor(box.width*ratio));canvas.height=Math.max(1,Math.floor(box.height*ratio));
 ctx.setTransform(ratio,0,0,ratio,0,0);draw();
}
function project(point){
 const box=canvas.getBoundingClientRect(),dx=point[0]-center[0],dy=point[1]-center[1],dz=point[2]-center[2];
 const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
 const x1=cy*dx-sy*dy,y1=sy*dx+cy*dy,z1=dz;
 const depth=cp*y1-sp*z1,z2=sp*y1+cp*z1;
 const scale=Math.min(box.width,box.height)*0.40*zoom/radius;
 return [box.width/2+x1*scale,box.height/2-z2*scale,depth];
}
function qrotate(q,v){
 const [x,y,z,w]=q,tx=2*(y*v[2]-z*v[1]),ty=2*(z*v[0]-x*v[2]),tz=2*(x*v[1]-y*v[0]);
 return [v[0]+w*tx+(y*tz-z*ty),v[1]+w*ty+(z*tx-x*tz),v[2]+w*tz+(x*ty-y*tx)];
}
function add(a,b){return [a[0]+b[0],a[1]+b[1],a[2]+b[2]]}
function scale(v,s){return [v[0]*s,v[1]*s,v[2]*s]}
function line3(a,b,color,width=1,dash=[]){
 const p=project(a),q=project(b);ctx.save();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke();ctx.restore();
}
function point3(p,color,r,label){
 const s=project(p);ctx.fillStyle=color;ctx.beginPath();ctx.arc(s[0],s[1],r,0,2*Math.PI);ctx.fill();
 if(label){ctx.font="12px ui-monospace,monospace";ctx.fillStyle="#e6edf3";ctx.fillText(label,s[0]+7,s[1]-7)}
}
function drawGrid(){
 const span=Math.ceil(radius),z=center[2]-radius*.45;
 for(let i=-span;i<=span;i+=.5){line3([center[0]+i,center[1]-span,z],[center[0]+i,center[1]+span,z],"#26303b",1);line3([center[0]-span,center[1]+i,z],[center[0]+span,center[1]+i,z],"#26303b",1)}
}
function drawBox(pose,size){
 const half=size.map(v=>v/2),corners=[];
 for(const x of [-half[0],half[0]])for(const y of [-half[1],half[1]])for(const z of [-half[2],half[2]])corners.push(add(pose.slice(0,3),qrotate(pose.slice(3),[x,y,z])));
 const edges=[[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]];
 edges.forEach(e=>line3(corners[e[0]],corners[e[1]],"#d29922",2));
}
function drawObject(frame){
 const type=summary.object.geometry_type,p=summary.object.primitive_params,pose=frame.object_pose_world;
 if(type==="box"&&Array.isArray(p.size_m)){drawBox(pose,p.size_m.map(Number));return}
 const c=project(pose.slice(0,3)),r=Math.max(8,(Number(p.radius_m||.1)/radius)*canvas.getBoundingClientRect().height*.4*zoom);
 ctx.strokeStyle="#d29922";ctx.lineWidth=2;ctx.beginPath();ctx.arc(c[0],c[1],r,0,2*Math.PI);ctx.stroke();
}
function drawModule(module,isBase){
 const pose=module.pose_world,c=pose.slice(0,3),q=pose.slice(3),x=qrotate(q,[moduleRadius,0,0]),y=qrotate(q,[0,moduleRadius,0]),z=qrotate(q,[0,0,moduleRadius*.45]);
 line3(add(c,scale(x,-1)),add(c,x),"#58a6ff",4);line3(add(c,scale(y,-1)),add(c,y),"#79c0ff",4);line3(c,add(c,z),"#a5d6ff",2);
 point3(c,isBase?"#f2cc60":"#58a6ff",isBase?6:5,`M${module.module_id}`);
}
function draw(){
 const box=canvas.getBoundingClientRect();ctx.clearRect(0,0,box.width,box.height);
 const frame=frames[index],byId=new Map(frame.modules.map(m=>[m.module_id,m]));
 drawGrid();drawObject(frame);
 edgeList.forEach(e=>line3(byId.get(e.src_module_id).pose_world.slice(0,3),byId.get(e.dst_module_id).pose_world.slice(0,3),"#8b949e",2));
 frame.modules.forEach(m=>drawModule(m,m.module_id===summary.morphology.base_module_id));
 frame.contacts.forEach(c=>{
   const actual=c.actual_pose_world.slice(0,3),target=c.target_pose_world.slice(0,3);
   line3(actual,target,c.active?"#3fb950":"#6e7681",1,[4,4]);
   line3(target,add(target,scale(c.target_normal_world,.13)),"#ff7b72",3);
   line3(actual,add(actual,scale(c.actual_outward_axis_world,.13)),"#3fb950",2);
   point3(target,"#ff7b72",5,`target A${c.anchor_id}`);point3(actual,c.active?"#3fb950":"#8b949e",4,`A${c.anchor_id}`);
 });
 document.getElementById("phase").textContent=frame.phase_label;
 document.getElementById("time").textContent=`t = ${frame.time_s.toFixed(3)} / ${summary.trajectory.horizon_s.toFixed(3)} s · knots ${frame.left_knot_index}→${frame.right_knot_index}`;
 const maxError=Math.max(...frame.contacts.map(c=>c.position_error_m));
 document.getElementById("contactError").textContent=`kinematic contact error: ${(1000*maxError).toFixed(3)} mm`;
 jointIds.forEach(id=>{const cell=document.querySelector(`[data-joint="${CSS.escape(id)}"]`);cell.textContent=(frame.joint_positions_rad[id]*180/Math.PI).toFixed(2)});
 timeline.value=String(index);
}
function indexForTime(time){
 let best=0;for(let i=1;i<frames.length;i++){if(Math.abs(frames[i].time_s-time)<Math.abs(frames[best].time_s-time))best=i}return best;
}
function animate(stamp){
 if(playing){
   if(lastStamp!==null)playTime+=(stamp-lastStamp)/1000*Number(document.getElementById("speed").value);
   if(playTime>summary.trajectory.horizon_s)playTime=0;
   index=indexForTime(playTime);draw();
 }
 lastStamp=stamp;requestAnimationFrame(animate);
}
playButton.onclick=()=>{playing=!playing;playButton.textContent=playing?"⏸ Pause":"▶ Play";playTime=frames[index].time_s;lastStamp=null};
timeline.oninput=()=>{index=Number(timeline.value);playTime=frames[index].time_s;draw()};
document.getElementById("reset").onclick=()=>{yaw=-.72;pitch=.48;zoom=1;draw()};
canvas.addEventListener("pointerdown",e=>{dragging=true;lastPointer=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)});
canvas.addEventListener("pointermove",e=>{if(!dragging)return;yaw+=(e.clientX-lastPointer[0])*.008;pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-lastPointer[1])*.008));lastPointer=[e.clientX,e.clientY];draw()});
canvas.addEventListener("pointerup",()=>dragging=false);
canvas.addEventListener("wheel",e=>{e.preventDefault();zoom=Math.max(.35,Math.min(4,zoom*Math.exp(-e.deltaY*.001)));draw()},{passive:false});
window.addEventListener("resize",resize);resize();requestAnimationFrame(animate);
</script>
</body>
</html>
"""


__all__ = [
    "ORDER9_ARTICULATED_TEACHER_ANIMATION_VERSION",
    "Order9TeacherAnimationArtifacts",
    "build_order9_articulated_teacher_animation_payload",
    "render_order9_c3_teacher_animation",
    "write_order9_articulated_teacher_animation",
]
