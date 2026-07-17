from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from amsrr.geometry.pose_math import (
    FACE_TO_FACE_DOCK_RELATION,
    Matrix3,
    compose_pose,
    dock_module_relative_pose,
    inverse_pose,
    matmul,
    pose_from_transform,
    quat_from_matrix,
    transform_from_pose,
    transform_from_xyz_rpy,
    transpose,
)
from amsrr.robot_model.gripper_surfaces import GripperSurface
from amsrr.schemas.common import Pose7D, SchemaValidationError
from amsrr.schemas.morphology import DockEdge, MorphologyGraph, PortNode, RobotAnchor
from amsrr.schemas.physical_model import DockPortSpec, JointModel, PhysicalModel

_IDENTITY_POSE: Pose7D = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class MeshBackedAnchorReference:
    anchor: RobotAnchor
    surface: GripperSurface


@dataclass(frozen=True)
class EdgeConstraintResidual:
    edge_id: int
    position_error_m: float
    attitude_error_rad: float


@dataclass(frozen=True)
class WholeStructureKinematicsConfig:
    finite_difference_step: float = 1.0e-5
    authored_pose_tolerance: float = 1.0e-7
    edge_position_tolerance_m: float = 1.0e-9
    edge_attitude_tolerance_rad: float = 1.0e-8


@dataclass(frozen=True)
class WholeStructureKinematicsResult:
    module_root_poses_world: dict[int, Pose7D]
    anchor_poses_world: dict[int, Pose7D]
    ordered_global_dock_joint_ids: tuple[str, ...]
    anchor_jacobians: dict[int, tuple[tuple[float, ...], ...]]
    edge_constraint_residuals: dict[int, EdgeConstraintResidual]
    finite_difference_modes: dict[str, str]


@dataclass(frozen=True)
class _ModelContext:
    root_link_id: str
    module_base_link_id: str
    outgoing_joints: dict[str, tuple[JointModel, ...]]
    kinematic_outgoing_joints: dict[str, tuple[JointModel, ...]]
    joint_origin_poses: dict[str, Pose7D]
    joint_motion_axes: dict[str, tuple[float, float, float]]
    port_specs_by_id: dict[str, DockPortSpec]
    connect_link_by_port_id: dict[str, str]
    dock_joint_ids: tuple[str, ...]
    dock_limits: dict[str, tuple[float | None, float | None]]


@dataclass(frozen=True)
class _GraphContext:
    modules: tuple[int, ...]
    base_module_id: int
    ports_by_global_id: dict[int, PortNode]
    adjacency: dict[int, tuple[tuple[int, DockEdge], ...]]


@dataclass(frozen=True)
class _ForwardSnapshot:
    module_roots: dict[int, Pose7D]
    anchor_poses: dict[int, Pose7D]
    edge_residuals: dict[int, EdgeConstraintResidual]


class WholeStructureKinematics:
    """Graph-constrained FK and full-Dock-column finite-difference Jacobians."""

    def __init__(self, config: WholeStructureKinematicsConfig | None = None) -> None:
        self.config = config or WholeStructureKinematicsConfig()
        _validate_config(self.config)
        self._cached_model_source: PhysicalModel | None = None
        self._cached_model_context: _ModelContext | None = None
        self._cached_graph_source: MorphologyGraph | None = None
        self._cached_graph_context: _GraphContext | None = None

    def compute(
        self,
        morphology: MorphologyGraph,
        physical_model: PhysicalModel,
        global_dock_joint_positions: Mapping[str, float],
        base_pose_world: Pose7D,
        selected_anchors: Sequence[MeshBackedAnchorReference],
    ) -> WholeStructureKinematicsResult:
        model, graph = self._contexts(morphology, physical_model)
        anchor_refs = _validate_anchor_references(
            morphology,
            graph,
            model,
            selected_anchors,
            self.config,
        )
        ordered_joint_ids = _ordered_global_dock_joint_ids(
            graph.modules, model.dock_joint_ids
        )
        q = _validate_global_q(
            global_dock_joint_positions, ordered_joint_ids, graph, model
        )
        _validate_pose(base_pose_world, "base_pose_world")

        module_link_pose_cache: dict[
            tuple[int, tuple[float, ...]], dict[str, Pose7D]
        ] = {}
        nominal = self._forward(
            graph,
            model,
            q,
            base_pose_world,
            anchor_refs,
            module_link_pose_cache=module_link_pose_cache,
        )
        jacobian_rows = {
            reference.anchor.anchor_id: [[] for _ in range(6)]
            for reference in anchor_refs
        }
        difference_modes: dict[str, str] = {}
        for global_joint_id in ordered_joint_ids:
            local_joint_id = _split_global_joint_id(global_joint_id)[1]
            lower, upper = model.dock_limits[local_joint_id]
            minus_q, plus_q, denominator, mode = _finite_difference_samples(
                q,
                global_joint_id,
                lower=lower,
                upper=upper,
                requested_step=self.config.finite_difference_step,
            )
            difference_modes[global_joint_id] = mode
            before = (
                nominal
                if minus_q is None
                else self._forward(
                    graph,
                    model,
                    minus_q,
                    base_pose_world,
                    anchor_refs,
                    module_link_pose_cache=module_link_pose_cache,
                )
            )
            after = (
                nominal
                if plus_q is None
                else self._forward(
                    graph,
                    model,
                    plus_q,
                    base_pose_world,
                    anchor_refs,
                    module_link_pose_cache=module_link_pose_cache,
                )
            )
            for reference in anchor_refs:
                anchor_id = reference.anchor.anchor_id
                derivative = _pose_finite_difference(
                    before.anchor_poses[anchor_id],
                    after.anchor_poses[anchor_id],
                    denominator,
                )
                for row, value in enumerate(derivative):
                    jacobian_rows[anchor_id][row].append(value)

        jacobians = {
            anchor_id: tuple(tuple(float(value) for value in row) for row in rows)
            for anchor_id, rows in jacobian_rows.items()
        }
        return WholeStructureKinematicsResult(
            module_root_poses_world=dict(nominal.module_roots),
            anchor_poses_world=dict(nominal.anchor_poses),
            ordered_global_dock_joint_ids=ordered_joint_ids,
            anchor_jacobians=jacobians,
            edge_constraint_residuals=dict(nominal.edge_residuals),
            finite_difference_modes=difference_modes,
        )

    def forward(
        self,
        morphology: MorphologyGraph,
        physical_model: PhysicalModel,
        global_dock_joint_positions: Mapping[str, float],
        base_pose_world: Pose7D,
        selected_anchors: Sequence[MeshBackedAnchorReference],
    ) -> WholeStructureKinematicsResult:
        """Return FK in the same result shape, with zero-valued full-column Jacobians."""

        model, graph = self._contexts(morphology, physical_model)
        anchor_refs = _validate_anchor_references(
            morphology,
            graph,
            model,
            selected_anchors,
            self.config,
        )
        ordered_joint_ids = _ordered_global_dock_joint_ids(
            graph.modules, model.dock_joint_ids
        )
        q = _validate_global_q(
            global_dock_joint_positions, ordered_joint_ids, graph, model
        )
        _validate_pose(base_pose_world, "base_pose_world")
        snapshot = self._forward(graph, model, q, base_pose_world, anchor_refs)
        zeros = tuple(0.0 for _ in ordered_joint_ids)
        return WholeStructureKinematicsResult(
            module_root_poses_world=dict(snapshot.module_roots),
            anchor_poses_world=dict(snapshot.anchor_poses),
            ordered_global_dock_joint_ids=ordered_joint_ids,
            anchor_jacobians={
                reference.anchor.anchor_id: (zeros, zeros, zeros, zeros, zeros, zeros)
                for reference in anchor_refs
            },
            edge_constraint_residuals=dict(snapshot.edge_residuals),
            finite_difference_modes={},
        )

    def _forward(
        self,
        graph: _GraphContext,
        model: _ModelContext,
        q: Mapping[str, float],
        base_pose_world: Pose7D,
        anchor_refs: Sequence[MeshBackedAnchorReference],
        *,
        module_link_pose_cache: (
            dict[tuple[int, tuple[float, ...]], dict[str, Pose7D]] | None
        ) = None,
    ) -> _ForwardSnapshot:
        module_link_poses: dict[int, dict[str, Pose7D]] = {}
        port_local_poses: dict[int, dict[int, Pose7D]] = {}
        for module_id in graph.modules:
            local_positions = tuple(
                float(q[_global_dock_joint_id(module_id, local_joint_id)])
                for local_joint_id in model.dock_joint_ids
            )
            cache_key = (module_id, local_positions)
            link_poses = (
                None
                if module_link_pose_cache is None
                else module_link_pose_cache.get(cache_key)
            )
            if link_poses is None:
                local_q = dict(zip(model.dock_joint_ids, local_positions, strict=True))
                link_poses = _module_link_poses(model, local_q)
                if module_link_pose_cache is not None:
                    module_link_pose_cache[cache_key] = link_poses
            module_link_poses[module_id] = link_poses
            port_local_poses[module_id] = {
                port_global_id: link_poses[
                    model.connect_link_by_port_id[port.port_local_id]
                ]
                for port_global_id, port in graph.ports_by_global_id.items()
                if port.module_id == module_id
            }

        roots: dict[int, Pose7D] = {graph.base_module_id: base_pose_world}
        frontier = [graph.base_module_id]
        while frontier:
            known_module = frontier.pop(0)
            for neighbor, edge in graph.adjacency[known_module]:
                if neighbor in roots:
                    continue
                if edge.src_module_id == known_module:
                    known_port_id = edge.src_port_id
                    child_port_id = edge.dst_port_id
                    relation = FACE_TO_FACE_DOCK_RELATION
                else:
                    known_port_id = edge.dst_port_id
                    child_port_id = edge.src_port_id
                    relation = inverse_pose(FACE_TO_FACE_DOCK_RELATION)
                known_connect_world = compose_pose(
                    roots[known_module],
                    port_local_poses[known_module][known_port_id],
                )
                desired_child_connect_world = compose_pose(
                    known_connect_world, relation
                )
                roots[neighbor] = compose_pose(
                    desired_child_connect_world,
                    inverse_pose(port_local_poses[neighbor][child_port_id]),
                )
                frontier.append(neighbor)

        if set(roots) != set(graph.modules):
            raise SchemaValidationError(
                "Whole-structure FK traversal did not reach every module"
            )
        anchor_poses = {
            reference.anchor.anchor_id: compose_pose(
                compose_pose(
                    roots[reference.anchor.module_id],
                    module_link_poses[reference.anchor.module_id][
                        reference.surface.mechanism_link_id
                    ],
                ),
                reference.anchor.local_pose,
            )
            for reference in anchor_refs
        }
        edge_residuals = _edge_constraint_residuals(graph, roots, port_local_poses)
        for residual in edge_residuals.values():
            if (
                residual.position_error_m > self.config.edge_position_tolerance_m
                or residual.attitude_error_rad > self.config.edge_attitude_tolerance_rad
            ):
                raise SchemaValidationError(
                    f"DockEdge {residual.edge_id} exact connect-frame constraint was not satisfied"
                )
        return _ForwardSnapshot(
            module_roots=roots,
            anchor_poses=anchor_poses,
            edge_residuals=edge_residuals,
        )

    def _contexts(
        self,
        morphology: MorphologyGraph,
        physical_model: PhysicalModel,
    ) -> tuple[_ModelContext, _GraphContext]:
        if self._cached_model_source is not physical_model:
            self._cached_model_context = _build_model_context(
                physical_model,
                self.config,
            )
            self._cached_model_source = physical_model
            self._cached_graph_source = None
            self._cached_graph_context = None
        if self._cached_model_context is None:
            raise RuntimeError("Whole-structure model-context cache is empty")
        if self._cached_graph_source is not morphology:
            self._cached_graph_context = _build_graph_context(
                morphology,
                physical_model,
                self.config,
            )
            self._cached_graph_source = morphology
        if self._cached_graph_context is None:
            raise RuntimeError("Whole-structure graph-context cache is empty")
        return self._cached_model_context, self._cached_graph_context


def ordered_global_dock_joint_ids(
    morphology: MorphologyGraph,
    physical_model: PhysicalModel,
) -> tuple[str, ...]:
    """Return the deterministic full global Dock-joint column ordering."""

    config = WholeStructureKinematicsConfig()
    model = _build_model_context(physical_model, config)
    graph = _build_graph_context(morphology, physical_model, config)
    return _ordered_global_dock_joint_ids(graph.modules, model.dock_joint_ids)


def _build_model_context(
    physical_model: PhysicalModel,
    config: WholeStructureKinematicsConfig,
) -> _ModelContext:
    link_ids = [link.link_id for link in physical_model.links]
    if len(link_ids) != len(set(link_ids)) or not link_ids:
        raise SchemaValidationError("PhysicalModel must contain unique links")
    joints_by_id = {joint.joint_id: joint for joint in physical_model.joints}
    if len(joints_by_id) != len(physical_model.joints):
        raise SchemaValidationError("PhysicalModel joints must have unique ids")
    child_joint: dict[str, JointModel] = {}
    outgoing: dict[str, list[JointModel]] = {link_id: [] for link_id in link_ids}
    for joint in physical_model.joints:
        if joint.parent_link not in outgoing or joint.child_link not in outgoing:
            raise SchemaValidationError(
                f"Joint {joint.joint_id!r} references an unknown link"
            )
        if joint.child_link in child_joint:
            raise SchemaValidationError(
                f"Link {joint.child_link!r} has multiple parent joints"
            )
        child_joint[joint.child_link] = joint
        outgoing[joint.parent_link].append(joint)
    roots = sorted(set(link_ids) - set(child_joint))
    if len(roots) != 1:
        raise SchemaValidationError(
            "PhysicalModel joint graph must have exactly one root link"
        )

    visited = {roots[0]}
    frontier = [roots[0]]
    while frontier:
        parent = frontier.pop(0)
        for joint in sorted(outgoing[parent], key=lambda item: item.joint_id):
            if joint.child_link in visited:
                raise SchemaValidationError(
                    "PhysicalModel joint graph contains a cycle"
                )
            visited.add(joint.child_link)
            frontier.append(joint.child_link)
    if visited != set(link_ids):
        raise SchemaValidationError("PhysicalModel joint graph is disconnected")

    vectoring_ids = {
        joint_id
        for rotor in physical_model.rotors
        for joint_id in rotor.vectoring_joint_ids
    }
    dock_joint_ids: set[str] = set()
    port_specs: dict[str, DockPortSpec] = {}
    connect_links: dict[str, str] = {}
    dock_limits: dict[str, tuple[float | None, float | None]] = {}
    for port in physical_model.dock_ports:
        if port.port_id in port_specs:
            raise SchemaValidationError(f"Duplicate DockPortSpec id {port.port_id!r}")
        port_specs[port.port_id] = port
        mechanism_joint_id = port.mechanical_limits.get("mechanism_joint_id")
        if not isinstance(mechanism_joint_id, str) or not mechanism_joint_id:
            raise SchemaValidationError(
                f"DockPortSpec {port.port_id!r} lacks a valid mechanism_joint_id"
            )
        mechanism_joint = joints_by_id.get(mechanism_joint_id)
        if mechanism_joint is None:
            raise SchemaValidationError(
                f"DockPortSpec {port.port_id!r} references unknown mechanism joint"
            )
        if mechanism_joint_id in vectoring_ids:
            raise SchemaValidationError(
                "Vectoring joints cannot be Dock kinematic variables"
            )
        if mechanism_joint.joint_type not in {"revolute", "continuous", "prismatic"}:
            raise SchemaValidationError("Dock mechanism joints must remain articulated")
        if mechanism_joint.child_link != port.parent_link:
            raise SchemaValidationError(
                f"DockPortSpec {port.port_id!r} mechanism link/joint mismatch"
            )
        connect_joint = joints_by_id.get(port.port_id)
        if connect_joint is None or connect_joint.parent_link != port.parent_link:
            raise SchemaValidationError(
                f"DockPortSpec {port.port_id!r} has no matching connect-frame joint"
            )
        dock_joint_ids.add(mechanism_joint_id)
        connect_links[port.port_id] = connect_joint.child_link
        limits = _validated_joint_limits(mechanism_joint)
        _validate_port_mechanical_limits(port, mechanism_joint, limits)
        previous_limits = dock_limits.get(mechanism_joint_id)
        if previous_limits is not None and previous_limits != limits:
            raise SchemaValidationError(
                f"Dock mechanism joint {mechanism_joint_id!r} has inconsistent port limits"
            )
        dock_limits[mechanism_joint_id] = limits

    if not dock_joint_ids:
        raise SchemaValidationError("PhysicalModel contains no Dock mechanism joints")
    baselink_metadata = physical_model.metadata.get("baselink")
    metadata_name = (
        baselink_metadata.get("name") if isinstance(baselink_metadata, dict) else None
    )
    if isinstance(metadata_name, str) and metadata_name in outgoing:
        module_base_link_id = metadata_name
    elif "fc" in outgoing:
        module_base_link_id = "fc"
    else:
        module_base_link_id = roots[0]
    required_link_ids = {
        roots[0],
        module_base_link_id,
        *connect_links.values(),
        *(port.parent_link for port in physical_model.dock_ports),
    }
    active_link_ids = set(required_link_ids)
    for link_id in tuple(required_link_ids):
        current = link_id
        while current in child_joint:
            parent = child_joint[current].parent_link
            active_link_ids.add(parent)
            current = parent
    kinematic_outgoing = {
        link_id: tuple(
            sorted(
                (
                    joint
                    for joint in outgoing[link_id]
                    if joint.child_link in active_link_ids
                ),
                key=lambda item: item.joint_id,
            )
        )
        for link_id in active_link_ids
    }
    active_joints = tuple(
        joint
        for link_id in sorted(kinematic_outgoing)
        for joint in kinematic_outgoing[link_id]
    )
    context = _ModelContext(
        root_link_id=roots[0],
        module_base_link_id=module_base_link_id,
        outgoing_joints={
            link_id: tuple(sorted(items, key=lambda item: item.joint_id))
            for link_id, items in outgoing.items()
        },
        kinematic_outgoing_joints=kinematic_outgoing,
        joint_origin_poses={
            joint.joint_id: pose_from_transform(
                transform_from_xyz_rpy(joint.origin_xyz, joint.origin_rpy)
            )
            for joint in active_joints
        },
        joint_motion_axes={
            joint.joint_id: _unit_axis(joint.axis_xyz, joint.joint_id)
            for joint in active_joints
            if joint.joint_type != "fixed"
        },
        port_specs_by_id=port_specs,
        connect_link_by_port_id=connect_links,
        dock_joint_ids=tuple(sorted(dock_joint_ids)),
        dock_limits=dock_limits,
    )
    neutral_links = _module_link_poses(
        context,
        {joint_id: 0.0 for joint_id in context.dock_joint_ids},
    )
    for port in physical_model.dock_ports:
        actual = neutral_links[context.connect_link_by_port_id[port.port_id]]
        if not _poses_close(actual, port.local_pose, config.authored_pose_tolerance):
            raise SchemaValidationError(
                f"DockPortSpec {port.port_id!r} local pose is stale relative to JointModel tree"
            )
    return context


def _build_graph_context(
    morphology: MorphologyGraph,
    physical_model: PhysicalModel,
    config: WholeStructureKinematicsConfig,
) -> _GraphContext:
    module_ids = tuple(sorted(module.module_id for module in morphology.modules))
    if not module_ids or len(module_ids) != len(set(module_ids)):
        raise SchemaValidationError("MorphologyGraph must contain unique modules")
    if morphology.base_module_id not in module_ids:
        raise SchemaValidationError("MorphologyGraph base module is unknown")
    declared_bases = [
        module.module_id for module in morphology.modules if module.is_base
    ]
    if declared_bases != [morphology.base_module_id]:
        raise SchemaValidationError(
            "MorphologyGraph must declare exactly its base_module_id as base"
        )
    if morphology.is_closed_loop:
        raise SchemaValidationError(
            "Whole-structure kinematics requires an acyclic morphology"
        )
    ports = {port.port_global_id: port for port in morphology.ports}
    if len(ports) != len(morphology.ports):
        raise SchemaValidationError("MorphologyGraph ports must have unique global ids")
    physical_ports = {port.port_id: port for port in physical_model.dock_ports}
    for port in morphology.ports:
        if port.module_id not in module_ids:
            raise SchemaValidationError(
                "MorphologyGraph port references an unknown module"
            )
        spec = physical_ports.get(port.port_local_id)
        if spec is None:
            raise SchemaValidationError(
                "MorphologyGraph port references an unknown PhysicalModel port"
            )
        if port.port_type != spec.port_type:
            raise SchemaValidationError(
                "MorphologyGraph and PhysicalModel port types disagree"
            )
        if not _poses_close(
            port.local_pose, spec.local_pose, config.authored_pose_tolerance
        ):
            raise SchemaValidationError(
                "MorphologyGraph port pose is stale relative to PhysicalModel"
            )

    if len(morphology.dock_edges) != len(module_ids) - 1:
        raise SchemaValidationError(
            "MorphologyGraph DockEdges must form exactly one tree"
        )
    adjacency_lists: dict[int, list[tuple[int, DockEdge]]] = {
        morphology.base_module_id: []
    }
    for module_id in module_ids:
        adjacency_lists.setdefault(module_id, [])
    used_ports: set[int] = set()
    edge_ids: set[int] = set()
    for edge in morphology.dock_edges:
        if edge.edge_id in edge_ids:
            raise SchemaValidationError("MorphologyGraph DockEdge ids must be unique")
        edge_ids.add(edge.edge_id)
        if edge.src_module_id == edge.dst_module_id:
            raise SchemaValidationError("DockEdge cannot connect a module to itself")
        if (
            edge.src_module_id not in adjacency_lists
            or edge.dst_module_id not in adjacency_lists
        ):
            raise SchemaValidationError("DockEdge references an unknown module")
        src_port = ports.get(edge.src_port_id)
        dst_port = ports.get(edge.dst_port_id)
        if src_port is None or dst_port is None:
            raise SchemaValidationError("DockEdge references an unknown port")
        if (
            src_port.module_id != edge.src_module_id
            or dst_port.module_id != edge.dst_module_id
        ):
            raise SchemaValidationError("DockEdge port/module ownership mismatch")
        src_spec = physical_ports[src_port.port_local_id]
        dst_spec = physical_ports[dst_port.port_local_id]
        if (
            dst_spec.port_type not in src_spec.compatible_port_types
            or src_spec.port_type not in dst_spec.compatible_port_types
        ):
            raise SchemaValidationError("DockEdge connects incompatible port types")
        if edge.src_port_id in used_ports or edge.dst_port_id in used_ports:
            raise SchemaValidationError(
                "A Dock port cannot belong to multiple exact constraints"
            )
        used_ports.update((edge.src_port_id, edge.dst_port_id))
        if edge.latch_state == "detached":
            raise SchemaValidationError(
                "Detached DockEdges do not constrain whole-structure FK"
            )
        expected_module_relation = dock_module_relative_pose(
            src_port.local_pose,
            dst_port.local_pose,
            port_relation=FACE_TO_FACE_DOCK_RELATION,
        )
        if not _poses_close(
            edge.relative_pose_src_to_dst,
            expected_module_relation,
            config.authored_pose_tolerance,
        ):
            raise SchemaValidationError(
                "DockEdge module relation is stale relative to exact connect frames"
            )
        adjacency_lists[edge.src_module_id].append((edge.dst_module_id, edge))
        adjacency_lists[edge.dst_module_id].append((edge.src_module_id, edge))

    visited = {morphology.base_module_id}
    frontier = [morphology.base_module_id]
    while frontier:
        module_id = frontier.pop(0)
        for neighbor, _edge in sorted(
            adjacency_lists[module_id],
            key=lambda item: (item[0], item[1].edge_id),
        ):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            frontier.append(neighbor)
    if visited != set(module_ids):
        raise SchemaValidationError("MorphologyGraph is disconnected")
    adjacency = {
        module_id: tuple(sorted(items, key=lambda item: (item[0], item[1].edge_id)))
        for module_id, items in adjacency_lists.items()
    }
    # Insertion order deliberately starts with the requested traversal base.
    adjacency = {
        morphology.base_module_id: adjacency[morphology.base_module_id],
        **{
            module_id: adjacency[module_id]
            for module_id in module_ids
            if module_id != morphology.base_module_id
        },
    }
    return _GraphContext(
        modules=module_ids,
        base_module_id=morphology.base_module_id,
        ports_by_global_id=ports,
        adjacency=adjacency,
    )


def _validate_anchor_references(
    morphology: MorphologyGraph,
    graph: _GraphContext,
    model: _ModelContext,
    selected_anchors: Sequence[MeshBackedAnchorReference],
    config: WholeStructureKinematicsConfig,
) -> tuple[MeshBackedAnchorReference, ...]:
    references = tuple(sorted(selected_anchors, key=lambda item: item.anchor.anchor_id))
    anchor_by_id = {anchor.anchor_id: anchor for anchor in morphology.robot_anchors}
    if len(anchor_by_id) != len(morphology.robot_anchors):
        raise SchemaValidationError("MorphologyGraph RobotAnchor ids must be unique")
    selected_ids: set[int] = set()
    edge_port_ids = {
        port_id
        for edge in morphology.dock_edges
        for port_id in (edge.src_port_id, edge.dst_port_id)
    }
    for reference in references:
        anchor = reference.anchor
        surface = reference.surface
        if anchor.anchor_id in selected_ids:
            raise SchemaValidationError("Selected anchor ids must be unique")
        selected_ids.add(anchor.anchor_id)
        graph_anchor = anchor_by_id.get(anchor.anchor_id)
        if graph_anchor != anchor:
            raise SchemaValidationError(
                "Selected anchor does not match MorphologyGraph"
            )
        if (
            anchor.module_id not in graph.modules
            or anchor.module_id != surface.module_id
        ):
            raise SchemaValidationError("Selected anchor/surface module mismatch")
        if anchor.link_id != surface.mechanism_link_id:
            raise SchemaValidationError(
                "Selected anchor must be local to its Dock mechanism link"
            )
        if surface.mechanism_link_id not in model.outgoing_joints:
            raise SchemaValidationError(
                "Selected surface references an unknown mechanism link"
            )
        if surface.mechanism_joint_id not in model.dock_joint_ids:
            raise SchemaValidationError(
                "Selected surface mechanism joint is not a Dock joint"
            )
        if surface.port_global_id in edge_port_ids:
            raise SchemaValidationError(
                "A structural DockEdge port cannot also be a gripper surface"
            )
        graph_port = graph.ports_by_global_id.get(surface.port_global_id)
        if graph_port is None or graph_port.module_id != surface.module_id:
            raise SchemaValidationError(
                "Selected surface references an unknown graph port"
            )
        if graph_port.occupied:
            raise SchemaValidationError(
                "Selected gripper surface port must be unoccupied"
            )
        spec = model.port_specs_by_id.get(surface.port_local_id)
        if spec is None or graph_port.port_local_id != spec.port_id:
            raise SchemaValidationError(
                "Selected surface references an unknown PhysicalModel port"
            )
        if spec.parent_link != surface.mechanism_link_id:
            raise SchemaValidationError(
                "Selected surface mechanism link does not match DockPortSpec"
            )
        if (
            spec.mechanical_limits.get("mechanism_joint_id")
            != surface.mechanism_joint_id
        ):
            raise SchemaValidationError(
                "Selected surface mechanism joint does not match DockPortSpec"
            )
        if not _poses_close(
            surface.connect_frame_module,
            spec.local_pose,
            config.authored_pose_tolerance,
        ):
            raise SchemaValidationError("Selected surface connect frame is stale")
        if not surface.collision_primitives:
            raise SchemaValidationError("Selected gripper surface must be mesh-backed")
        for primitive in surface.collision_primitives:
            if (
                primitive.link_id != surface.mechanism_link_id
                or primitive.primitive_type not in {"mesh", "convex"}
            ):
                raise SchemaValidationError(
                    "Selected gripper surface collision is not mesh-backed"
                )
        _validate_pose(anchor.local_pose, f"anchor[{anchor.anchor_id}].local_pose")
    return references


def _module_link_poses(
    model: _ModelContext,
    dock_q: Mapping[str, float],
) -> dict[str, Pose7D]:
    if set(dock_q) != set(model.dock_joint_ids):
        raise SchemaValidationError(
            "Module FK requires every local Dock joint and no extras"
        )
    poses_root: dict[str, Pose7D] = {model.root_link_id: _IDENTITY_POSE}
    frontier = [model.root_link_id]
    while frontier:
        parent = frontier.pop(0)
        for joint in model.kinematic_outgoing_joints.get(parent, ()):
            q = float(dock_q.get(joint.joint_id, 0.0))
            origin = model.joint_origin_poses[joint.joint_id]
            motion = _joint_motion_pose(
                joint,
                q,
                axis=model.joint_motion_axes.get(joint.joint_id),
            )
            poses_root[joint.child_link] = compose_pose(
                poses_root[parent],
                compose_pose(origin, motion),
            )
            frontier.append(joint.child_link)
    root_from_module = inverse_pose(poses_root[model.module_base_link_id])
    return {
        link_id: compose_pose(root_from_module, pose)
        for link_id, pose in poses_root.items()
    }


def _joint_motion_pose(
    joint: JointModel,
    q: float,
    *,
    axis: tuple[float, float, float] | None = None,
) -> Pose7D:
    if joint.joint_type == "fixed":
        return _IDENTITY_POSE
    if not math.isfinite(q):
        raise SchemaValidationError(f"Joint {joint.joint_id!r} position must be finite")
    axis = axis or _unit_axis(joint.axis_xyz, joint.joint_id)
    if joint.joint_type in {"revolute", "continuous"}:
        half = 0.5 * q
        sine = math.sin(half)
        return (
            0.0,
            0.0,
            0.0,
            axis[0] * sine,
            axis[1] * sine,
            axis[2] * sine,
            math.cos(half),
        )
    if joint.joint_type == "prismatic":
        return (
            axis[0] * q,
            axis[1] * q,
            axis[2] * q,
            0.0,
            0.0,
            0.0,
            1.0,
        )
    raise SchemaValidationError(f"Unsupported JointModel type {joint.joint_type!r}")


def _edge_constraint_residuals(
    graph: _GraphContext,
    roots: Mapping[int, Pose7D],
    port_local_poses: Mapping[int, Mapping[int, Pose7D]],
) -> dict[int, EdgeConstraintResidual]:
    unique_edges = {
        edge.edge_id: edge
        for items in graph.adjacency.values()
        for _neighbor, edge in items
    }
    residuals: dict[int, EdgeConstraintResidual] = {}
    for edge_id, edge in sorted(unique_edges.items()):
        src_world = compose_pose(
            roots[edge.src_module_id],
            port_local_poses[edge.src_module_id][edge.src_port_id],
        )
        dst_world = compose_pose(
            roots[edge.dst_module_id],
            port_local_poses[edge.dst_module_id][edge.dst_port_id],
        )
        expected_dst = compose_pose(src_world, FACE_TO_FACE_DOCK_RELATION)
        error = compose_pose(inverse_pose(expected_dst), dst_world)
        residuals[edge_id] = EdgeConstraintResidual(
            edge_id=edge_id,
            position_error_m=_norm3(error[:3]),
            attitude_error_rad=_quaternion_angle(error[3:]),
        )
    return residuals


def _finite_difference_samples(
    q: Mapping[str, float],
    joint_id: str,
    *,
    lower: float | None,
    upper: float | None,
    requested_step: float,
) -> tuple[dict[str, float] | None, dict[str, float] | None, float, str]:
    value = float(q[joint_id])
    minus_room = math.inf if lower is None else value - lower
    plus_room = math.inf if upper is None else upper - value
    if minus_room >= requested_step and plus_room >= requested_step:
        minus = dict(q)
        plus = dict(q)
        minus[joint_id] = value - requested_step
        plus[joint_id] = value + requested_step
        return minus, plus, 2.0 * requested_step, "central"
    if plus_room >= requested_step:
        plus = dict(q)
        plus[joint_id] = value + requested_step
        return None, plus, requested_step, "forward"
    if minus_room >= requested_step:
        minus = dict(q)
        minus[joint_id] = value - requested_step
        return minus, None, requested_step, "backward"
    if plus_room > 1.0e-12:
        step = min(requested_step, plus_room)
        plus = dict(q)
        plus[joint_id] = value + step
        return None, plus, step, "forward_reduced"
    if minus_room > 1.0e-12:
        step = min(requested_step, minus_room)
        minus = dict(q)
        minus[joint_id] = value - step
        return minus, None, step, "backward_reduced"
    raise SchemaValidationError(
        f"Dock joint {joint_id!r} has no finite-difference room"
    )


def _pose_finite_difference(
    before: Pose7D,
    after: Pose7D,
    denominator: float,
) -> tuple[float, float, float, float, float, float]:
    translation = tuple(
        (float(after[index]) - float(before[index])) / denominator for index in range(3)
    )
    before_rotation = transform_from_pose(before).rotation
    after_rotation = transform_from_pose(after).rotation
    delta_world = matmul(after_rotation, transpose(before_rotation))
    rotation_log = _rotation_log_vector(delta_world)
    angular = tuple(value / denominator for value in rotation_log)
    return (*translation, *angular)


def _rotation_log_vector(rotation: Matrix3) -> tuple[float, float, float]:
    qx, qy, qz, qw = quat_from_matrix(rotation)
    vector_norm = math.sqrt(qx * qx + qy * qy + qz * qz)
    if vector_norm <= 1.0e-15:
        return (0.0, 0.0, 0.0)
    angle = 2.0 * math.atan2(vector_norm, max(0.0, qw))
    scale = angle / vector_norm
    return (qx * scale, qy * scale, qz * scale)


def _validate_global_q(
    values: Mapping[str, float],
    ordered_ids: tuple[str, ...],
    graph: _GraphContext,
    model: _ModelContext,
) -> dict[str, float]:
    expected = set(ordered_ids)
    supplied = set(values)
    if supplied != expected:
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        vectoring_local_ids = {
            local_id
            for extra_id in extra
            for _module_id, local_id in [_split_global_joint_id(extra_id)]
            if "gimbal" in local_id.lower() or "vectoring" in local_id.lower()
        }
        if vectoring_local_ids:
            raise SchemaValidationError(
                "Vectoring joints are excluded from the Dock q map"
            )
        raise SchemaValidationError(
            f"Global Dock q map mismatch; missing={missing}, extra={extra}"
        )
    result: dict[str, float] = {}
    for joint_id in ordered_ids:
        value = float(values[joint_id])
        if not math.isfinite(value):
            raise SchemaValidationError(
                f"Dock joint {joint_id!r} position must be finite"
            )
        module_id, local_id = _split_global_joint_id(joint_id)
        if module_id not in graph.modules or local_id not in model.dock_joint_ids:
            raise SchemaValidationError(f"Unknown global Dock joint {joint_id!r}")
        lower, upper = model.dock_limits[local_id]
        if lower is not None and value < lower - 1.0e-12:
            raise SchemaValidationError(
                f"Dock joint {joint_id!r} is below its lower limit"
            )
        if upper is not None and value > upper + 1.0e-12:
            raise SchemaValidationError(
                f"Dock joint {joint_id!r} is above its upper limit"
            )
        if lower is not None:
            value = max(value, lower)
        if upper is not None:
            value = min(value, upper)
        result[joint_id] = value
    return result


def _validated_joint_limits(joint: JointModel) -> tuple[float | None, float | None]:
    if joint.joint_type == "continuous":
        return None, None
    lower = joint.limit_lower
    upper = joint.limit_upper
    if lower is None or upper is None:
        raise SchemaValidationError(f"Dock joint {joint.joint_id!r} has missing limits")
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise SchemaValidationError(f"Dock joint {joint.joint_id!r} has invalid limits")
    return float(lower), float(upper)


def _validate_port_mechanical_limits(
    port: DockPortSpec,
    joint: JointModel,
    limits: tuple[float | None, float | None],
) -> None:
    metadata = port.mechanical_limits
    for key, expected in (("limit_lower", limits[0]), ("limit_upper", limits[1])):
        value = metadata.get(key)
        if expected is None:
            if value is not None:
                raise SchemaValidationError(
                    f"DockPortSpec {port.port_id!r} {key} disagrees with continuous joint"
                )
        elif (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1.0e-12)
        ):
            raise SchemaValidationError(
                f"DockPortSpec {port.port_id!r} {key} disagrees with JointModel"
            )
    for key, expected in (
        ("effort_limit", joint.effort_limit),
        ("velocity_limit", joint.velocity_limit),
    ):
        value = metadata.get(key)
        if (
            expected is None
            or not math.isfinite(float(expected))
            or float(expected) <= 0.0
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            or not math.isclose(
                float(value), float(expected), rel_tol=0.0, abs_tol=1.0e-12
            )
        ):
            raise SchemaValidationError(
                f"DockPortSpec {port.port_id!r} {key} must match a positive JointModel limit"
            )


def _ordered_global_dock_joint_ids(
    module_ids: Sequence[int],
    local_joint_ids: Sequence[str],
) -> tuple[str, ...]:
    return tuple(
        _global_dock_joint_id(module_id, local_joint_id)
        for module_id in sorted(module_ids)
        for local_joint_id in sorted(local_joint_ids)
    )


def _global_dock_joint_id(module_id: int, local_joint_id: str) -> str:
    return f"module_{module_id}:{local_joint_id}"


def _split_global_joint_id(global_joint_id: str) -> tuple[int, str]:
    if ":" not in global_joint_id:
        return -1, global_joint_id
    module_label, local_id = global_joint_id.split(":", 1)
    if (
        not module_label.startswith("module_")
        or not module_label[7:].isdigit()
        or not local_id
    ):
        return -1, local_id
    return int(module_label[7:]), local_id


def _unit_axis(axis: Sequence[float], joint_id: str) -> tuple[float, float, float]:
    norm = _norm3(axis)
    if norm <= 1.0e-12:
        raise SchemaValidationError(f"Joint {joint_id!r} has a zero motion axis")
    return (float(axis[0]) / norm, float(axis[1]) / norm, float(axis[2]) / norm)


def _norm3(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values[:3]))


def _quaternion_angle(values: Sequence[float]) -> float:
    norm = math.sqrt(sum(float(value) ** 2 for value in values))
    if norm <= 0.0:
        raise SchemaValidationError("Quaternion norm must be positive")
    w = abs(float(values[3]) / norm)
    return 2.0 * math.acos(min(1.0, max(-1.0, w)))


def _poses_close(left: Pose7D, right: Pose7D, tolerance: float) -> bool:
    error = compose_pose(inverse_pose(left), right)
    return _norm3(error[:3]) <= tolerance and _quaternion_angle(error[3:]) <= tolerance


def _validate_pose(pose: Pose7D, name: str) -> None:
    if len(pose) != 7 or not all(math.isfinite(float(value)) for value in pose):
        raise SchemaValidationError(f"{name} must be a finite Pose7D")
    if math.sqrt(sum(float(value) ** 2 for value in pose[3:])) <= 1.0e-12:
        raise SchemaValidationError(f"{name} quaternion must be non-zero")


def _validate_config(config: WholeStructureKinematicsConfig) -> None:
    for name in (
        "finite_difference_step",
        "authored_pose_tolerance",
        "edge_position_tolerance_m",
        "edge_attitude_tolerance_rad",
    ):
        value = getattr(config, name)
        if not math.isfinite(value) or value <= 0.0:
            raise SchemaValidationError(f"{name} must be finite and positive")
