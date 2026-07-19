from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Iterable, TypeVar

from amsrr.geometry.pose_math import (
    compose_pose,
    inverse_pose,
    matvec,
    pose_from_transform,
    transform_from_pose,
    transform_from_xyz_rpy,
)
from amsrr.schemas.common import Pose7D, SchemaValidationError, Vector3
from amsrr.schemas.morphology import MorphologyGraph, PortNode
from amsrr.schemas.physical_model import (
    CollisionPrimitive,
    DockPortSpec,
    JointModel,
    PhysicalModel,
)

_MESH_BACKED_COLLISION_TYPES = {"mesh", "convex"}
_CONVEX_DECOMPOSITION_MESH_SUFFIXES = (
    ".dae",
    ".obj",
    ".ply",
    ".stl",
    ".usd",
    ".usda",
    ".usdc",
)
_ItemT = TypeVar("_ItemT")


class GripperSurfaceResolutionError(SchemaValidationError):
    """Raised when a graph Dock port cannot resolve to usable collision geometry."""


@dataclass(frozen=True)
class GripperCollisionPrimitive:
    primitive_id: str
    link_id: str
    primitive_type: str
    geometry_ref: str | None
    local_pose: Pose7D | None
    convex_decomposition_compatible: bool
    requires_convex_decomposition: bool


@dataclass(frozen=True)
class GripperSurface:
    module_id: int
    port_global_id: int
    port_local_id: str
    port_type: str
    mechanism_link_id: str
    mechanism_joint_id: str
    mechanism_joint_type: str
    mechanism_joint_axis_local: Vector3
    mechanism_joint_axis_design: Vector3
    mechanism_joint_limit_lower: float | None
    mechanism_joint_limit_upper: float | None
    connect_frame_module: Pose7D
    connect_frame_design: Pose7D
    neutral_outward_axis_design: Vector3
    collision_primitives: tuple[GripperCollisionPrimitive, ...]


@dataclass(frozen=True)
class GripperSurfacePair:
    first: GripperSurface
    second: GripperSurface
    surface_separation_m: float
    first_inward_axis_design: Vector3
    second_inward_axis_design: Vector3
    first_mechanism_position_target: float
    second_mechanism_position_target: float
    first_inward_alignment: float
    second_inward_alignment: float
    opposition_alignment: float
    grasp_anchor_module_ids: tuple[int, ...]


def resolve_unoccupied_gripper_surfaces(
    morphology: MorphologyGraph,
    physical_model: PhysicalModel,
) -> tuple[GripperSurface, ...]:
    """Resolve free graph Dock ports to mesh-backed mechanism collision bodies.

    Every DockEdge endpoint is treated as occupied even if a stale ``PortNode``
    flag says otherwise.  The returned connect-frame and collision metadata are
    sourced from ``PhysicalModel`` after their correspondence with the graph is
    checked.
    """

    modules_by_id = {module.module_id: module for module in morphology.modules}
    if len(modules_by_id) != len(morphology.modules):
        raise GripperSurfaceResolutionError("morphology has duplicate module ids")
    physical_ports_by_id = _unique_by_id(
        physical_model.dock_ports,
        key=lambda port: port.port_id,
        label="PhysicalModel Dock port",
    )
    physical_joints_by_id = _unique_by_id(
        physical_model.joints,
        key=lambda joint: joint.joint_id,
        label="PhysicalModel joint",
    )
    edge_port_ids = {
        port_id
        for edge in morphology.dock_edges
        for port_id in (edge.src_port_id, edge.dst_port_id)
    }
    collisions_by_link: dict[str, list[CollisionPrimitive]] = {}
    for primitive in physical_model.collision_primitives:
        collisions_by_link.setdefault(primitive.link_id, []).append(primitive)

    surfaces: list[GripperSurface] = []
    for graph_port in sorted(morphology.ports, key=_graph_port_sort_key):
        if graph_port.occupied or graph_port.port_global_id in edge_port_ids:
            continue
        module = modules_by_id.get(graph_port.module_id)
        if module is None:
            raise GripperSurfaceResolutionError(
                f"port {graph_port.port_global_id} references missing module "
                f"{graph_port.module_id}"
            )
        physical_port = physical_ports_by_id.get(graph_port.port_local_id)
        if physical_port is None:
            raise GripperSurfaceResolutionError(
                f"port {graph_port.port_global_id} has no PhysicalModel DockPortSpec "
                f"{graph_port.port_local_id!r}"
            )
        _validate_graph_port_matches_physical(graph_port, physical_port)
        connect_joint = physical_joints_by_id.get(physical_port.port_id)
        if connect_joint is None:
            raise GripperSurfaceResolutionError(
                f"DockPortSpec {physical_port.port_id!r} has no connect-frame joint"
            )
        if connect_joint.parent_link != physical_port.parent_link:
            raise GripperSurfaceResolutionError(
                f"DockPortSpec {physical_port.port_id!r} parent link "
                f"{physical_port.parent_link!r} does not match connect joint parent "
                f"{connect_joint.parent_link!r}"
            )
        mechanism_joint = _mechanism_joint_for_port(
            physical_port,
            physical_model.joints,
        )
        collision_primitives = _mesh_collision_primitives(
            physical_port.parent_link,
            collisions_by_link.get(physical_port.parent_link, []),
        )
        connect_frame_design = compose_pose(
            module.pose_in_design_frame,
            physical_port.local_pose,
        )
        mechanism_axis_design = _mechanism_axis_in_design(
            module_pose=module.pose_in_design_frame,
            connect_frame_module=physical_port.local_pose,
            connect_joint=connect_joint,
            mechanism_joint=mechanism_joint,
        )
        neutral_outward_axis = _unit(
            matvec(
                transform_from_pose(connect_frame_design).rotation,
                (1.0, 0.0, 0.0),
            ),
            label=f"DockPortSpec {physical_port.port_id!r} outward axis",
        )
        surfaces.append(
            GripperSurface(
                module_id=graph_port.module_id,
                port_global_id=graph_port.port_global_id,
                port_local_id=graph_port.port_local_id,
                port_type=physical_port.port_type,
                mechanism_link_id=physical_port.parent_link,
                mechanism_joint_id=mechanism_joint.joint_id,
                mechanism_joint_type=mechanism_joint.joint_type,
                mechanism_joint_axis_local=mechanism_joint.axis_xyz,
                mechanism_joint_axis_design=mechanism_axis_design,
                mechanism_joint_limit_lower=mechanism_joint.limit_lower,
                mechanism_joint_limit_upper=mechanism_joint.limit_upper,
                connect_frame_module=physical_port.local_pose,
                connect_frame_design=connect_frame_design,
                neutral_outward_axis_design=neutral_outward_axis,
                collision_primitives=collision_primitives,
            )
        )
    return tuple(surfaces)


def select_opposing_gripper_surface_pair(
    morphology: MorphologyGraph,
    physical_model: PhysicalModel,
    *,
    minimum_inward_alignment: float = math.sqrt(0.5),
) -> GripperSurfacePair:
    """Select two independently actuated surfaces that can face one another.

    When the morphology already contains grasp anchors, candidates are restricted
    to their modules so the two selected contacts are simultaneously usable by
    that design.  Joint targets are neutral-frame geometric targets only.  The
    default gate requires each surface normal to remain within 45 degrees of the
    inter-surface gap; runtime control remains responsible for collision-aware
    whole-structure motion and final contact alignment.
    """

    if (
        not isinstance(minimum_inward_alignment, (int, float))
        or isinstance(minimum_inward_alignment, bool)
        or not math.isfinite(float(minimum_inward_alignment))
        or not 0.0 < float(minimum_inward_alignment) <= 1.0
    ):
        raise GripperSurfaceResolutionError(
            "minimum_inward_alignment must be finite and in (0, 1]"
        )
    surfaces = resolve_unoccupied_gripper_surfaces(morphology, physical_model)
    grasp_anchor_module_ids = tuple(
        sorted(
            {
                anchor.module_id
                for anchor in morphology.robot_anchors
                if anchor.anchor_type == "grasp"
            }
        )
    )
    if len(grasp_anchor_module_ids) == 1:
        raise GripperSurfaceResolutionError(
            "opposing grasp requires grasp anchors on at least two modules"
        )
    permitted_modules = (
        set(grasp_anchor_module_ids) if len(grasp_anchor_module_ids) >= 2 else None
    )

    ranked_pairs: list[tuple[tuple[float, ...], GripperSurfacePair]] = []
    for first, second in combinations(surfaces, 2):
        if first.module_id == second.module_id:
            continue
        if permitted_modules is not None and (
            first.module_id not in permitted_modules
            or second.module_id not in permitted_modules
        ):
            continue
        delta = _subtract(
            second.connect_frame_design[:3],
            first.connect_frame_design[:3],
        )
        separation = _norm(delta)
        if separation <= 1.0e-12:
            continue
        first_inward = _scale(delta, 1.0 / separation)
        second_inward = _scale(first_inward, -1.0)
        first_alignment, first_target, first_normal = _best_inward_alignment(
            first,
            first_inward,
        )
        second_alignment, second_target, second_normal = _best_inward_alignment(
            second,
            second_inward,
        )
        minimum_alignment = min(first_alignment, second_alignment)
        if minimum_alignment + 1.0e-12 < minimum_inward_alignment:
            continue
        opposition = -_dot(first_normal, second_normal)
        pair = GripperSurfacePair(
            first=first,
            second=second,
            surface_separation_m=separation,
            first_inward_axis_design=first_inward,
            second_inward_axis_design=second_inward,
            first_mechanism_position_target=first_target,
            second_mechanism_position_target=second_target,
            first_inward_alignment=first_alignment,
            second_inward_alignment=second_alignment,
            opposition_alignment=opposition,
            grasp_anchor_module_ids=grasp_anchor_module_ids,
        )
        rank = (
            -round(minimum_alignment, 12),
            -round(opposition, 12),
            round(separation, 12),
            float(first.module_id),
            float(first.port_global_id),
            float(second.module_id),
            float(second.port_global_id),
        )
        ranked_pairs.append((rank, pair))
    if not ranked_pairs:
        scope = (
            f"grasp-anchor modules {list(grasp_anchor_module_ids)}"
            if grasp_anchor_module_ids
            else "distinct morphology modules"
        )
        raise GripperSurfaceResolutionError(
            "no two unoccupied mesh-backed Dock surfaces on "
            f"{scope} can achieve inward alignment >= {minimum_inward_alignment}"
        )
    ranked_pairs.sort(key=lambda item: item[0])
    return ranked_pairs[0][1]


def _graph_port_sort_key(port: PortNode) -> tuple[int, int, str]:
    return port.module_id, port.port_global_id, port.port_local_id


def _unique_by_id(
    items: Iterable[_ItemT],
    *,
    key: Callable[[_ItemT], str],
    label: str,
) -> dict[str, _ItemT]:
    by_id: dict[str, _ItemT] = {}
    for item in items:
        item_id = key(item)
        if item_id in by_id:
            raise GripperSurfaceResolutionError(f"{label} id {item_id!r} is not unique")
        by_id[item_id] = item
    return by_id


def _validate_graph_port_matches_physical(
    graph_port: PortNode,
    physical_port: DockPortSpec,
) -> None:
    if graph_port.port_type != physical_port.port_type:
        raise GripperSurfaceResolutionError(
            f"graph port {graph_port.port_global_id} type {graph_port.port_type!r} "
            f"does not match PhysicalModel type {physical_port.port_type!r}"
        )
    if any(
        not math.isclose(float(graph_value), float(model_value), abs_tol=1.0e-9)
        for graph_value, model_value in zip(
            graph_port.local_pose,
            physical_port.local_pose,
        )
    ):
        raise GripperSurfaceResolutionError(
            f"graph port {graph_port.port_global_id} connect frame is stale relative "
            f"to PhysicalModel DockPortSpec {physical_port.port_id!r}"
        )


def _mechanism_joint_for_port(
    physical_port: DockPortSpec,
    joints: list[JointModel],
) -> JointModel:
    metadata_joint_id = physical_port.mechanical_limits.get("mechanism_joint_id")
    if metadata_joint_id is not None and (
        not isinstance(metadata_joint_id, str) or not metadata_joint_id
    ):
        raise GripperSurfaceResolutionError(
            f"DockPortSpec {physical_port.port_id!r} has invalid mechanism_joint_id metadata"
        )
    candidates = [
        joint for joint in joints if joint.child_link == physical_port.parent_link
    ]
    if isinstance(metadata_joint_id, str) and metadata_joint_id:
        candidates = [
            joint for joint in candidates if joint.joint_id == metadata_joint_id
        ]
    if len(candidates) != 1:
        raise GripperSurfaceResolutionError(
            f"DockPortSpec {physical_port.port_id!r} does not resolve to exactly one "
            f"mechanism joint for parent link {physical_port.parent_link!r}"
        )
    return candidates[0]


def _mesh_collision_primitives(
    mechanism_link_id: str,
    primitives: list[CollisionPrimitive],
) -> tuple[GripperCollisionPrimitive, ...]:
    if not primitives:
        raise GripperSurfaceResolutionError(
            f"Dock mechanism link {mechanism_link_id!r} has no collision primitives"
        )
    resolved: list[GripperCollisionPrimitive] = []
    for primitive in sorted(primitives, key=lambda item: item.primitive_id):
        if primitive.primitive_type not in _MESH_BACKED_COLLISION_TYPES:
            raise GripperSurfaceResolutionError(
                f"Dock mechanism collision {primitive.primitive_id!r} must be mesh "
                f"or convex, got {primitive.primitive_type!r}"
            )
        if primitive.primitive_type == "mesh" and not primitive.geometry_ref:
            raise GripperSurfaceResolutionError(
                f"Dock mechanism mesh collision {primitive.primitive_id!r} has no geometry_ref"
            )
        if (
            primitive.primitive_type == "mesh"
            and primitive.geometry_ref is not None
            and not primitive.geometry_ref.lower()
            .split("?", 1)[0]
            .endswith(_CONVEX_DECOMPOSITION_MESH_SUFFIXES)
        ):
            raise GripperSurfaceResolutionError(
                f"Dock mechanism mesh collision {primitive.primitive_id!r} geometry "
                f"{primitive.geometry_ref!r} is not a supported convex-decomposition mesh"
            )
        resolved.append(
            GripperCollisionPrimitive(
                primitive_id=primitive.primitive_id,
                link_id=primitive.link_id,
                primitive_type=primitive.primitive_type,
                geometry_ref=primitive.geometry_ref,
                local_pose=primitive.local_pose,
                convex_decomposition_compatible=True,
                requires_convex_decomposition=primitive.primitive_type == "mesh",
            )
        )
    return tuple(resolved)


def _mechanism_axis_in_design(
    *,
    module_pose: Pose7D,
    connect_frame_module: Pose7D,
    connect_joint: JointModel,
    mechanism_joint: JointModel,
) -> Vector3:
    connect_origin = pose_from_transform(
        transform_from_xyz_rpy(connect_joint.origin_xyz, connect_joint.origin_rpy)
    )
    mechanism_link_module = compose_pose(
        connect_frame_module,
        inverse_pose(connect_origin),
    )
    mechanism_link_design = compose_pose(module_pose, mechanism_link_module)
    return _unit(
        matvec(
            transform_from_pose(mechanism_link_design).rotation,
            mechanism_joint.axis_xyz,
        ),
        label=f"mechanism joint {mechanism_joint.joint_id!r} axis",
    )


def _best_inward_alignment(
    surface: GripperSurface,
    inward_axis: Vector3,
) -> tuple[float, float, Vector3]:
    normal = surface.neutral_outward_axis_design
    axis = surface.mechanism_joint_axis_design
    joint_type = surface.mechanism_joint_type
    if joint_type not in {"revolute", "continuous"}:
        return _dot(normal, inward_axis), 0.0, normal

    axial_normal = _dot(normal, axis)
    normal_perpendicular = _subtract(normal, _scale(axis, axial_normal))
    target_perpendicular = _subtract(
        inward_axis,
        _scale(axis, _dot(inward_axis, axis)),
    )
    cosine_coefficient = _dot(normal_perpendicular, target_perpendicular)
    sine_coefficient = _dot(
        _cross(axis, normal_perpendicular),
        target_perpendicular,
    )
    optimum = math.atan2(sine_coefficient, cosine_coefficient)
    if joint_type == "continuous":
        candidates = [optimum]
    else:
        lower = surface.mechanism_joint_limit_lower
        upper = surface.mechanism_joint_limit_upper
        if lower is None or upper is None or lower > upper:
            raise GripperSurfaceResolutionError(
                f"revolute mechanism joint {surface.mechanism_joint_id!r} has invalid limits"
            )
        candidates = [float(lower), float(upper)]
        two_pi = 2.0 * math.pi
        first_turn = math.ceil((lower - optimum) / two_pi)
        last_turn = math.floor((upper - optimum) / two_pi)
        candidates.extend(
            optimum + two_pi * float(turn) for turn in range(first_turn, last_turn + 1)
        )
    evaluated = []
    for position in candidates:
        rotated_normal = _rotate_about_axis(normal, axis, position)
        evaluated.append(
            (
                _dot(rotated_normal, inward_axis),
                abs(position),
                position,
                rotated_normal,
            )
        )
    evaluated.sort(key=lambda item: (-round(item[0], 12), item[1], item[2]))
    alignment, _absolute_position, target, rotated_normal = evaluated[0]
    return alignment, target, rotated_normal


def _rotate_about_axis(vector: Vector3, axis: Vector3, angle: float) -> Vector3:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return _unit(
        _add(
            _add(
                _scale(vector, cosine),
                _scale(_cross(axis, vector), sine),
            ),
            _scale(axis, _dot(axis, vector) * (1.0 - cosine)),
        ),
        label="rotated gripper normal",
    )


def _unit(vector: Vector3, *, label: str) -> Vector3:
    magnitude = _norm(vector)
    if not math.isfinite(magnitude) or magnitude <= 1.0e-12:
        raise GripperSurfaceResolutionError(f"{label} must be finite and non-zero")
    return _scale(vector, 1.0 / magnitude)


def _norm(vector: Vector3) -> float:
    return math.sqrt(_dot(vector, vector))


def _dot(left: Vector3, right: Vector3) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _add(left: Vector3, right: Vector3) -> Vector3:
    return (
        float(left[0]) + float(right[0]),
        float(left[1]) + float(right[1]),
        float(left[2]) + float(right[2]),
    )


def _subtract(left, right) -> Vector3:
    return (
        float(left[0]) - float(right[0]),
        float(left[1]) - float(right[1]),
        float(left[2]) - float(right[2]),
    )


def _scale(vector: Vector3, scalar: float) -> Vector3:
    return (
        float(vector[0]) * scalar,
        float(vector[1]) * scalar,
        float(vector[2]) * scalar,
    )


def _cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


__all__ = [
    "GripperCollisionPrimitive",
    "GripperSurface",
    "GripperSurfacePair",
    "GripperSurfaceResolutionError",
    "resolve_unoccupied_gripper_surfaces",
    "select_opposing_gripper_surface_pair",
]
