from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from amsrr.geometry.pose_math import (
    FACE_TO_FACE_DOCK_RELATION,
    compose_pose,
    dock_module_relative_pose,
    inverse_pose,
    pose_from_transform,
    pose_to_xyz_rpy,
    transform_from_xyz_rpy,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.robot_model.urdf_loader import URDFModel, load_urdf
from amsrr.robot_model.urdf_transforms import link_poses_in_root_frame


FIXED_MODULE_PREFIX_SEPARATOR = "__"


def write_fixed_morphology_urdf(
    source_urdf_path: str | Path,
    output_urdf_path: str | Path,
    *,
    module_count: int = 2,
    module_spacing_m: float = 0.45,
    prefix_separator: str = FIXED_MODULE_PREFIX_SEPARATOR,
    mesh_search_dirs: list[str | Path] | None = None,
) -> Path:
    """Write a deterministic rigid multi-module URDF for P4-control smoke tests."""

    if module_count < 1:
        raise SchemaValidationError("fixed morphology module_count must be >= 1")
    if module_spacing_m <= 0.0:
        raise SchemaValidationError("fixed morphology module_spacing_m must be positive")
    source_path = Path(source_urdf_path).resolve()
    output_path = Path(output_urdf_path)
    source_root = ET.parse(source_path).getroot()
    if _tag_name(source_root) != "robot":
        raise SchemaValidationError(f"Expected <robot> root in {source_path}")
    search_dirs = _normalise_mesh_search_dirs(mesh_search_dirs)

    source_links = _named_children(source_root, "link")
    if "root" not in source_links:
        raise SchemaValidationError("source Holon URDF must contain root link")
    module_root_poses = fixed_morphology_module_poses(
        source_path,
        module_count=module_count,
        module_spacing_m=module_spacing_m,
    )
    parent_module_ids = _fixed_morphology_parent_module_ids(module_count)

    output_root = ET.Element("robot", {"name": f"holon_fixed_morphology_{module_count}"})
    ET.SubElement(output_root, "baselink", {"name": _prefixed_name(0, "fc", prefix_separator)})
    ET.SubElement(output_root, "thrust_link", {"name": "thrust"})

    for module_id in range(module_count):
        prefix = f"module_{module_id}{prefix_separator}"
        for child in list(source_root):
            tag = _tag_name(child)
            if tag in {"baselink", "thrust_link", "m_f_rate"}:
                continue
            copied = copy.deepcopy(child)
            _prefix_element(copied, prefix=prefix, source_dir=source_path.parent, mesh_search_dirs=search_dirs)
            output_root.append(copied)
        if module_id > 0:
            parent_module_id = parent_module_ids[module_id]
            parent_pose = module_root_poses[parent_module_id]
            child_pose = module_root_poses[module_id]
            relative_pose = compose_pose(inverse_pose(parent_pose), child_pose)
            xyz, rpy = pose_to_xyz_rpy(relative_pose)
            joint = ET.SubElement(
                output_root,
                "joint",
                {
                    "name": f"fixed_module_{module_id}_to_module_{parent_module_id}",
                    "type": "fixed",
                },
            )
            ET.SubElement(joint, "origin", {"xyz": _format_vec(xyz), "rpy": _format_vec(rpy)})
            ET.SubElement(joint, "parent", {"link": _prefixed_name(parent_module_id, "root", prefix_separator)})
            ET.SubElement(joint, "child", {"link": _prefixed_name(module_id, "root", prefix_separator)})
            ET.SubElement(joint, "axis", {"xyz": "0 0 0"})

    _indent(output_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(output_root).write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def write_fixed_morphology_graph_urdf(
    source_urdf_path: str | Path,
    output_urdf_path: str | Path,
    *,
    morphology_graph: MorphologyGraph,
    prefix_separator: str = FIXED_MODULE_PREFIX_SEPARATOR,
    mesh_search_dirs: list[str | Path] | None = None,
) -> Path:
    """Write a rigid URDF from an assembled MorphologyGraph.

    This reflects a P3 assembled graph at reset time. It is not a dynamic
    module attach/detach operation and does not edit morphology during rollout.
    """

    if not morphology_graph.modules:
        raise SchemaValidationError("fixed morphology graph URDF requires at least one module")
    if morphology_graph.is_closed_loop:
        raise SchemaValidationError("fixed morphology graph URDF requires a tree morphology")
    source_path = Path(source_urdf_path).resolve()
    output_path = Path(output_urdf_path)
    source_root = ET.parse(source_path).getroot()
    if _tag_name(source_root) != "robot":
        raise SchemaValidationError(f"Expected <robot> root in {source_path}")
    source_links = _named_children(source_root, "link")
    if "root" not in source_links:
        raise SchemaValidationError("source Holon URDF must contain root link")
    search_dirs = _normalise_mesh_search_dirs(mesh_search_dirs)
    module_ids = sorted(module.module_id for module in morphology_graph.modules)
    if morphology_graph.base_module_id not in module_ids:
        raise SchemaValidationError("morphology graph base module is missing")
    tree_edges = _morphology_graph_tree_edges(morphology_graph)
    module_poses = morphology_graph_module_root_poses(
        morphology_graph,
        source_urdf_path=source_path,
    )

    output_root = ET.Element("robot", {"name": f"holon_graph_morphology_{_safe_graph_name(morphology_graph.graph_id)}"})
    ET.SubElement(output_root, "baselink", {"name": _prefixed_name(morphology_graph.base_module_id, "fc", prefix_separator)})
    ET.SubElement(output_root, "thrust_link", {"name": "thrust"})

    for module_id in module_ids:
        prefix = f"module_{module_id}{prefix_separator}"
        for child in list(source_root):
            tag = _tag_name(child)
            if tag in {"baselink", "thrust_link", "m_f_rate"}:
                continue
            copied = copy.deepcopy(child)
            _prefix_element(copied, prefix=prefix, source_dir=source_path.parent, mesh_search_dirs=search_dirs)
            output_root.append(copied)

    for edge_index, (parent_module_id, child_module_id) in enumerate(tree_edges):
        parent_pose = module_poses[parent_module_id]
        child_pose = module_poses[child_module_id]
        relative_pose = compose_pose(inverse_pose(parent_pose), child_pose)
        xyz, rpy = pose_to_xyz_rpy(relative_pose)
        joint = ET.SubElement(
            output_root,
            "joint",
            {
                "name": f"graph_edge_{edge_index}_module_{child_module_id}_to_module_{parent_module_id}",
                "type": "fixed",
            },
        )
        ET.SubElement(joint, "origin", {"xyz": _format_vec(xyz), "rpy": _format_vec(rpy)})
        ET.SubElement(joint, "parent", {"link": _prefixed_name(parent_module_id, "root", prefix_separator)})
        ET.SubElement(joint, "child", {"link": _prefixed_name(child_module_id, "root", prefix_separator)})
        ET.SubElement(joint, "axis", {"xyz": "0 0 0"})

    _indent(output_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(output_root).write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def morphology_graph_module_poses(
    morphology_graph: MorphologyGraph,
) -> dict[int, tuple[float, float, float, float, float, float, float]]:
    modules_by_id = {module.module_id: module for module in morphology_graph.modules}
    base = modules_by_id.get(morphology_graph.base_module_id)
    if base is None:
        raise SchemaValidationError("morphology graph base module is missing")
    base_inv = inverse_pose(base.pose_in_design_frame)
    return {
        module_id: compose_pose(base_inv, module.pose_in_design_frame)
        for module_id, module in modules_by_id.items()
    }


def morphology_graph_module_root_poses(
    morphology_graph: MorphologyGraph,
    *,
    source_urdf_path: str | Path,
) -> dict[int, tuple[float, float, float, float, float, float, float]]:
    """Convert graph module-frame (``fc``) poses to generated-URDF root poses.

    ``MorphologyGraph`` dock geometry is expressed in the PhysicalModel module
    frame, which is Holon's ``fc`` link.  A generated multi-module URDF is
    connected at each copied ``root`` link.  The two frames coincide only for
    translations without rotation, so apply the conjugation
    ``root_T_fc * fc0_T_fci * fc_T_root`` before writing fixed root joints.
    """

    source_path = Path(source_urdf_path).resolve()
    urdf_model = load_urdf(source_path)
    link_poses_root = link_poses_in_root_frame(urdf_model)
    base_link = urdf_model.metadata.get("baselink", {}).get("name", "fc")
    if base_link not in link_poses_root:
        raise SchemaValidationError(f"source Holon module frame {base_link!r} is missing")
    root_to_module_frame = link_poses_root[base_link]
    module_frame_to_root = inverse_pose(root_to_module_frame)
    return {
        module_id: compose_pose(
            compose_pose(root_to_module_frame, module_pose),
            module_frame_to_root,
        )
        for module_id, module_pose in morphology_graph_module_poses(morphology_graph).items()
    }


def fixed_morphology_module_poses(
    source_urdf_path: str | Path,
    *,
    module_count: int = 2,
    module_spacing_m: float = 0.45,
) -> dict[int, tuple[float, float, float, float, float, float, float]]:
    if module_count < 1:
        raise SchemaValidationError("fixed morphology module_count must be >= 1")
    source_path = Path(source_urdf_path).resolve()
    urdf_model = load_urdf(source_path)
    port_records = _connect_port_records(urdf_model)
    if len(port_records) < 2:
        return {
            module_id: (module_spacing_m * module_id, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
            for module_id in range(module_count)
        }
    poses = {0: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)}
    used_ports: set[tuple[int, str]] = set()
    for module_id in range(1, module_count):
        parent_module_id = module_id - 1
        src, dst = _first_compatible_free_pair(
            parent_module_id,
            module_id,
            port_records,
            used_ports,
        )
        used_ports.update({(parent_module_id, src.port_id), (module_id, dst.port_id)})
        parent_to_child = dock_module_relative_pose(src.pose_root, dst.pose_root)
        poses[module_id] = compose_pose(poses[parent_module_id], parent_to_child)
    return poses


@dataclass(frozen=True)
class _ConnectPortRecord:
    port_id: str
    port_type: str
    connect_link: str
    mechanism_joint_id: str | None
    pose_root: tuple[float, float, float, float, float, float, float]


@dataclass(frozen=True)
class ArticulatedDockConnection:
    parent_module_id: int
    child_module_id: int
    parent_port_id: str
    child_port_id: str
    parent_connect_link: str
    child_connect_link: str
    parent_mechanism_joint_id: str | None
    child_mechanism_joint_id: str | None
    parent_connect_to_child_root_pose: tuple[float, float, float, float, float, float, float]


@dataclass(frozen=True)
class ArticulatedGraphDockConnection:
    edge_id: int
    parent_module_id: int
    child_module_id: int
    parent_port_id: str
    child_port_id: str
    parent_connect_link: str
    child_connect_link: str
    parent_mechanism_joint_id: str | None
    child_mechanism_joint_id: str | None


def _fixed_morphology_parent_module_ids(module_count: int) -> dict[int, int]:
    return {module_id: module_id - 1 for module_id in range(1, module_count)}


def _morphology_graph_tree_edges(morphology_graph: MorphologyGraph) -> list[tuple[int, int]]:
    module_ids = {module.module_id for module in morphology_graph.modules}
    if len(module_ids) <= 1:
        return []
    adjacency: dict[int, list[int]] = {module_id: [] for module_id in module_ids}
    for edge in morphology_graph.dock_edges:
        if edge.src_module_id not in module_ids or edge.dst_module_id not in module_ids:
            raise SchemaValidationError("dock edge references a missing module")
        adjacency[edge.src_module_id].append(edge.dst_module_id)
        adjacency[edge.dst_module_id].append(edge.src_module_id)
    if not any(adjacency.values()):
        raise SchemaValidationError("multi-module graph URDF requires dock edges")

    visited = {morphology_graph.base_module_id}
    frontier = [morphology_graph.base_module_id]
    tree_edges: list[tuple[int, int]] = []
    while frontier:
        parent = frontier.pop(0)
        for neighbor in sorted(adjacency[parent]):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            frontier.append(neighbor)
            tree_edges.append((parent, neighbor))
    if visited != module_ids:
        raise SchemaValidationError("morphology graph is disconnected")
    if len(tree_edges) != len(module_ids) - 1:
        raise SchemaValidationError("morphology graph must be a tree for fixed graph URDF generation")
    return tree_edges


def _oriented_morphology_graph_edges(morphology_graph: MorphologyGraph):
    module_ids = {module.module_id for module in morphology_graph.modules}
    if morphology_graph.base_module_id not in module_ids:
        raise SchemaValidationError("morphology graph base module is missing")
    if len(morphology_graph.dock_edges) != max(0, len(module_ids) - 1):
        raise SchemaValidationError(
            "articulated morphology graph must contain exactly one tree edge per non-base module"
        )
    edge_ids = [edge.edge_id for edge in morphology_graph.dock_edges]
    if len(edge_ids) != len(set(edge_ids)):
        raise SchemaValidationError("morphology graph repeats a dock edge id")
    adjacency = {module_id: [] for module_id in module_ids}
    for edge in morphology_graph.dock_edges:
        if edge.src_module_id not in module_ids or edge.dst_module_id not in module_ids:
            raise SchemaValidationError("dock edge references a missing module")
        if edge.src_module_id == edge.dst_module_id:
            raise SchemaValidationError("dock edge cannot connect a module to itself")
        adjacency[edge.src_module_id].append((edge.dst_module_id, edge))
        adjacency[edge.dst_module_id].append((edge.src_module_id, edge))

    visited = {morphology_graph.base_module_id}
    frontier = [morphology_graph.base_module_id]
    oriented = []
    while frontier:
        parent = frontier.pop(0)
        for child, edge in sorted(
            adjacency[parent], key=lambda item: (item[0], item[1].edge_id)
        ):
            if child in visited:
                continue
            visited.add(child)
            frontier.append(child)
            oriented.append((parent, child, edge))
    if visited != module_ids or len(oriented) != max(0, len(module_ids) - 1):
        raise SchemaValidationError("morphology graph is disconnected")
    return oriented


def _reroot_module_tree(
    module_root: ET.Element,
    *,
    new_root_link: str,
    old_root_link: str,
) -> None:
    """Reverse the unique joint path from a Dock link to the module root.

    For an active original joint ``T0 * motion(q)``, the reversed transform is
    ``motion(-q) * inverse(T0)``.  A zero-origin active joint followed by a
    fixed offset represents that transform while retaining the original joint
    name and generalized-coordinate sign.
    """

    links = _named_children(module_root, "link")
    if new_root_link not in links or old_root_link not in links:
        raise SchemaValidationError("module re-root link is missing")
    joints = _iter_named_joints(module_root)
    joint_by_child: dict[str, ET.Element] = {}
    for joint in joints:
        child = _child(joint, "child")
        if child is None or not child.attrib.get("link"):
            raise SchemaValidationError("source URDF joint has no child link")
        child_link = child.attrib["link"]
        if child_link in joint_by_child:
            raise SchemaValidationError("source URDF link has multiple parent joints")
        joint_by_child[child_link] = joint

    path: list[ET.Element] = []
    current = new_root_link
    visited: set[str] = set()
    while current != old_root_link:
        if current in visited:
            raise SchemaValidationError("source URDF joint path contains a cycle")
        visited.add(current)
        joint = joint_by_child.get(current)
        if joint is None:
            raise SchemaValidationError(
                f"Dock link {new_root_link!r} does not descend from {old_root_link!r}"
            )
        parent = _child(joint, "parent")
        if parent is None or not parent.attrib.get("link"):
            raise SchemaValidationError("source URDF joint has no parent link")
        path.append(joint)
        current = parent.attrib["link"]

    existing_link_names = set(links)
    existing_joint_names = {joint.attrib["name"] for joint in joints}
    for joint in path:
        joint_name = joint.attrib["name"]
        joint_type = joint.attrib.get("type", "fixed")
        parent = _child(joint, "parent")
        child = _child(joint, "child")
        if parent is None or child is None:
            raise SchemaValidationError("source URDF joint endpoints are incomplete")
        old_parent = parent.attrib["link"]
        old_child = child.attrib["link"]
        origin_pose = _joint_origin_pose(joint)
        inverse_origin = inverse_pose(origin_pose)

        if joint_type == "fixed":
            parent.attrib["link"] = old_child
            child.attrib["link"] = old_parent
            _set_joint_origin_pose(joint, inverse_origin)
            continue
        if joint_type not in {"revolute", "continuous", "prismatic"}:
            raise SchemaValidationError(
                f"Cannot re-root unsupported URDF joint type {joint_type!r}"
            )
        axis = _child(joint, "axis")
        if axis is None or not axis.attrib.get("xyz"):
            raise SchemaValidationError(
                f"Active URDF joint {joint_name!r} has no axis"
            )
        axis_values = _parse_vec3(axis.attrib["xyz"], f"joint {joint_name} axis")
        dummy_link = f"{joint_name}__reroot_offset_link"
        offset_joint_name = f"{joint_name}__reroot_offset"
        if dummy_link in existing_link_names or offset_joint_name in existing_joint_names:
            raise SchemaValidationError("URDF re-root helper name collision")
        existing_link_names.add(dummy_link)
        existing_joint_names.add(offset_joint_name)

        parent.attrib["link"] = old_child
        child.attrib["link"] = dummy_link
        _set_joint_origin_pose(
            joint, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        )
        axis.attrib["xyz"] = _format_vec(tuple(-value for value in axis_values))
        helper_link = ET.Element("link", {"name": dummy_link})
        inertial = ET.SubElement(helper_link, "inertial")
        ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        ET.SubElement(inertial, "mass", {"value": "1e-6"})
        ET.SubElement(
            inertial,
            "inertia",
            {
                "ixx": "1e-12",
                "ixy": "0",
                "ixz": "0",
                "iyy": "1e-12",
                "iyz": "0",
                "izz": "1e-12",
            },
        )
        module_root.append(helper_link)
        offset = ET.Element(
            "joint", {"name": offset_joint_name, "type": "fixed"}
        )
        xyz, rpy = pose_to_xyz_rpy(inverse_origin)
        ET.SubElement(
            offset, "origin", {"xyz": _format_vec(xyz), "rpy": _format_vec(rpy)}
        )
        ET.SubElement(offset, "parent", {"link": dummy_link})
        ET.SubElement(offset, "child", {"link": old_parent})
        ET.SubElement(offset, "axis", {"xyz": "0 0 0"})
        module_root.append(offset)


def _joint_origin_pose(
    joint: ET.Element,
) -> tuple[float, float, float, float, float, float, float]:
    origin = _child(joint, "origin")
    xyz = (
        _parse_vec3(origin.attrib.get("xyz", "0 0 0"), "joint origin xyz")
        if origin is not None
        else (0.0, 0.0, 0.0)
    )
    rpy = (
        _parse_vec3(origin.attrib.get("rpy", "0 0 0"), "joint origin rpy")
        if origin is not None
        else (0.0, 0.0, 0.0)
    )
    return pose_from_transform(transform_from_xyz_rpy(xyz, rpy))


def _set_joint_origin_pose(
    joint: ET.Element,
    pose: tuple[float, float, float, float, float, float, float],
) -> None:
    origin = _child(joint, "origin")
    if origin is None:
        origin = ET.Element("origin")
        joint.insert(0, origin)
    xyz, rpy = pose_to_xyz_rpy(pose)
    origin.attrib["xyz"] = _format_vec(xyz)
    origin.attrib["rpy"] = _format_vec(rpy)


def _parse_vec3(value: str, label: str) -> tuple[float, float, float]:
    try:
        parsed = tuple(float(item) for item in value.split())
    except ValueError as exc:
        raise SchemaValidationError(f"{label} is not numeric") from exc
    if len(parsed) != 3:
        raise SchemaValidationError(f"{label} must contain three values")
    return parsed


def _safe_graph_name(graph_id: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in graph_id)[:80] or "graph"


def articulated_morphology_connections(
    source_urdf_path: str | Path,
    *,
    module_count: int = 2,
) -> list[ArticulatedDockConnection]:
    if module_count < 1:
        raise SchemaValidationError("articulated morphology module_count must be >= 1")
    urdf_model = load_urdf(Path(source_urdf_path).resolve())
    port_records = _connect_port_records(urdf_model)
    if len(port_records) < 2 and module_count > 1:
        raise SchemaValidationError("articulated morphology requires at least two connect ports")
    connections: list[ArticulatedDockConnection] = []
    used_ports: set[tuple[int, str]] = set()
    for module_id in range(1, module_count):
        parent_module_id = module_id - 1
        src, dst = _first_compatible_free_pair(parent_module_id, module_id, port_records, used_ports)
        used_ports.update({(parent_module_id, src.port_id), (module_id, dst.port_id)})
        parent_connect_to_child_root = compose_pose(FACE_TO_FACE_DOCK_RELATION, inverse_pose(dst.pose_root))
        connections.append(
            ArticulatedDockConnection(
                parent_module_id=parent_module_id,
                child_module_id=module_id,
                parent_port_id=src.port_id,
                child_port_id=dst.port_id,
                parent_connect_link=src.connect_link,
                child_connect_link=dst.connect_link,
                parent_mechanism_joint_id=src.mechanism_joint_id,
                child_mechanism_joint_id=dst.mechanism_joint_id,
                parent_connect_to_child_root_pose=parent_connect_to_child_root,
            )
        )
    return connections


def articulated_morphology_graph_connections(
    source_urdf_path: str | Path,
    *,
    morphology_graph: MorphologyGraph,
) -> list[ArticulatedGraphDockConnection]:
    """Resolve graph edges to exact parent/child Dock links.

    Edges are oriented away from ``base_module_id``.  The child module will be
    re-rooted at the selected Dock connect link by
    :func:`write_articulated_morphology_graph_urdf`.
    """

    morphology_graph.validate()
    if not morphology_graph.modules:
        raise SchemaValidationError(
            "articulated morphology graph requires at least one module"
        )
    if morphology_graph.is_closed_loop:
        raise SchemaValidationError(
            "articulated morphology graph requires a tree morphology"
        )
    port_by_global_id = {
        port.port_global_id: port for port in morphology_graph.ports
    }
    records = _connect_port_records(load_urdf(Path(source_urdf_path).resolve()))
    record_by_port_id = {record.port_id: record for record in records}
    if len(record_by_port_id) != len(records):
        raise SchemaValidationError("source URDF repeats a Dock port id")

    connections: list[ArticulatedGraphDockConnection] = []
    for parent_module_id, child_module_id, edge in _oriented_morphology_graph_edges(
        morphology_graph
    ):
        if edge.src_module_id == parent_module_id:
            parent_global_port_id = edge.src_port_id
            child_global_port_id = edge.dst_port_id
        else:
            parent_global_port_id = edge.dst_port_id
            child_global_port_id = edge.src_port_id
        parent_port = port_by_global_id.get(parent_global_port_id)
        child_port = port_by_global_id.get(child_global_port_id)
        if parent_port is None or child_port is None:
            raise SchemaValidationError("dock edge references a missing graph port")
        if (
            parent_port.module_id != parent_module_id
            or child_port.module_id != child_module_id
        ):
            raise SchemaValidationError(
                "dock edge port ownership differs from its module ids"
            )
        parent_record = record_by_port_id.get(parent_port.port_local_id)
        child_record = record_by_port_id.get(child_port.port_local_id)
        if parent_record is None or child_record is None:
            raise SchemaValidationError(
                "graph Dock port is absent from the source Holon URDF"
            )
        if not _ports_compatible(parent_record.port_type, child_record.port_type):
            raise SchemaValidationError("graph Dock edge connects incompatible ports")
        connections.append(
            ArticulatedGraphDockConnection(
                edge_id=edge.edge_id,
                parent_module_id=parent_module_id,
                child_module_id=child_module_id,
                parent_port_id=parent_record.port_id,
                child_port_id=child_record.port_id,
                parent_connect_link=parent_record.connect_link,
                child_connect_link=child_record.connect_link,
                parent_mechanism_joint_id=parent_record.mechanism_joint_id,
                child_mechanism_joint_id=child_record.mechanism_joint_id,
            )
        )
    return connections


def write_articulated_morphology_graph_urdf(
    source_urdf_path: str | Path,
    output_urdf_path: str | Path,
    *,
    morphology_graph: MorphologyGraph,
    prefix_separator: str = FIXED_MODULE_PREFIX_SEPARATOR,
    mesh_search_dirs: list[str | Path] | None = None,
) -> Path:
    """Write a graph-exact articulated multi-module URDF.

    A dynamic Order-8 Dock constraint joins both selected mechanism links.  A
    single URDF can represent the same open-tree kinematics only when every
    non-base module is re-rooted at its selected connect frame.  Reversing the
    child-side path preserves both Dock mechanism DOFs in the structural chain;
    a root-to-root fixed joint would incorrectly remove them.
    """

    source_path = Path(source_urdf_path).resolve()
    output_path = Path(output_urdf_path)
    source_root = ET.parse(source_path).getroot()
    if _tag_name(source_root) != "robot":
        raise SchemaValidationError(f"Expected <robot> root in {source_path}")
    source_links = _named_children(source_root, "link")
    if "root" not in source_links:
        raise SchemaValidationError("source Holon URDF must contain root link")
    search_dirs = _normalise_mesh_search_dirs(mesh_search_dirs)
    connections = articulated_morphology_graph_connections(
        source_path,
        morphology_graph=morphology_graph,
    )
    connection_by_child = {
        connection.child_module_id: connection for connection in connections
    }
    module_ids = sorted(module.module_id for module in morphology_graph.modules)

    output_root = ET.Element(
        "robot",
        {
            "name": (
                "holon_articulated_graph_"
                f"{_safe_graph_name(morphology_graph.graph_id)}"
            )
        },
    )
    ET.SubElement(
        output_root,
        "baselink",
        {
            "name": _prefixed_name(
                morphology_graph.base_module_id, "fc", prefix_separator
            )
        },
    )
    ET.SubElement(output_root, "thrust_link", {"name": "thrust"})

    for module_id in module_ids:
        module_root = ET.Element("robot", {"name": f"module_{module_id}"})
        for child in list(source_root):
            if _tag_name(child) in {"baselink", "thrust_link", "m_f_rate"}:
                continue
            module_root.append(copy.deepcopy(child))
        connection = connection_by_child.get(module_id)
        if connection is not None:
            _reroot_module_tree(
                module_root,
                new_root_link=connection.child_connect_link,
                old_root_link="root",
            )
        prefix = f"module_{module_id}{prefix_separator}"
        for child in list(module_root):
            _prefix_element(
                child,
                prefix=prefix,
                source_dir=source_path.parent,
                mesh_search_dirs=search_dirs,
            )
            output_root.append(child)

    for connection in connections:
        xyz, rpy = pose_to_xyz_rpy(FACE_TO_FACE_DOCK_RELATION)
        joint = ET.SubElement(
            output_root,
            "joint",
            {
                "name": (
                    f"articulated_graph_edge_{connection.edge_id}_module_"
                    f"{connection.child_module_id}_to_module_"
                    f"{connection.parent_module_id}"
                ),
                "type": "fixed",
            },
        )
        ET.SubElement(
            joint, "origin", {"xyz": _format_vec(xyz), "rpy": _format_vec(rpy)}
        )
        ET.SubElement(
            joint,
            "parent",
            {
                "link": _prefixed_name(
                    connection.parent_module_id,
                    connection.parent_connect_link,
                    prefix_separator,
                )
            },
        )
        ET.SubElement(
            joint,
            "child",
            {
                "link": _prefixed_name(
                    connection.child_module_id,
                    connection.child_connect_link,
                    prefix_separator,
                )
            },
        )
        ET.SubElement(joint, "axis", {"xyz": "0 0 0"})

    _indent(output_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(output_root).write(
        output_path, encoding="utf-8", xml_declaration=True
    )
    return output_path


def write_articulated_morphology_urdf(
    source_urdf_path: str | Path,
    output_urdf_path: str | Path,
    *,
    module_count: int = 2,
    prefix_separator: str = FIXED_MODULE_PREFIX_SEPARATOR,
    mesh_search_dirs: list[str | Path] | None = None,
) -> Path:
    """Write a tree-structured articulated multi-module URDF.

    Child module roots are attached to the selected parent connect dummy link, so
    the parent dock mechanism joint moves the whole child module subtree.
    """

    if module_count < 1:
        raise SchemaValidationError("articulated morphology module_count must be >= 1")
    source_path = Path(source_urdf_path).resolve()
    output_path = Path(output_urdf_path)
    source_root = ET.parse(source_path).getroot()
    if _tag_name(source_root) != "robot":
        raise SchemaValidationError(f"Expected <robot> root in {source_path}")
    source_links = _named_children(source_root, "link")
    if "root" not in source_links:
        raise SchemaValidationError("source Holon URDF must contain root link")
    search_dirs = _normalise_mesh_search_dirs(mesh_search_dirs)
    connections = articulated_morphology_connections(source_path, module_count=module_count)

    output_root = ET.Element("robot", {"name": f"holon_articulated_morphology_{module_count}"})
    ET.SubElement(output_root, "baselink", {"name": _prefixed_name(0, "fc", prefix_separator)})
    ET.SubElement(output_root, "thrust_link", {"name": "thrust"})

    for module_id in range(module_count):
        prefix = f"module_{module_id}{prefix_separator}"
        for child in list(source_root):
            tag = _tag_name(child)
            if tag in {"baselink", "thrust_link", "m_f_rate"}:
                continue
            copied = copy.deepcopy(child)
            _prefix_element(copied, prefix=prefix, source_dir=source_path.parent, mesh_search_dirs=search_dirs)
            output_root.append(copied)

    for connection in connections:
        xyz, rpy = pose_to_xyz_rpy(connection.parent_connect_to_child_root_pose)
        joint = ET.SubElement(
            output_root,
            "joint",
            {
                "name": f"articulated_module_{connection.child_module_id}_to_module_{connection.parent_module_id}",
                "type": "fixed",
            },
        )
        ET.SubElement(joint, "origin", {"xyz": _format_vec(xyz), "rpy": _format_vec(rpy)})
        ET.SubElement(
            joint,
            "parent",
            {"link": _prefixed_name(connection.parent_module_id, connection.parent_connect_link, prefix_separator)},
        )
        ET.SubElement(joint, "child", {"link": _prefixed_name(connection.child_module_id, "root", prefix_separator)})
        ET.SubElement(joint, "axis", {"xyz": "0 0 0"})

    _indent(output_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(output_root).write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def _connect_port_records(urdf_model: URDFModel) -> list[_ConnectPortRecord]:
    link_poses_root = link_poses_in_root_frame(urdf_model)
    joints_by_name = {joint.name: joint for joint in urdf_model.joints}
    mechanism_joint_by_child = {
        joint.child_link: joint
        for joint in urdf_model.joints
        if "dock_mech_joint" in joint.name
    }
    records: list[_ConnectPortRecord] = []
    for joint_name in urdf_model.candidate_connect_joints:
        joint = joints_by_name[joint_name]
        mechanism_joint = mechanism_joint_by_child.get(joint.parent_link)
        records.append(
            _ConnectPortRecord(
                port_id=joint_name,
                port_type=_port_type_from_name(joint_name + "_" + joint.parent_link),
                connect_link=joint.child_link,
                mechanism_joint_id=mechanism_joint.name if mechanism_joint is not None else None,
                pose_root=link_poses_root[joint.child_link],
            )
        )
    return sorted(records, key=lambda item: item.port_id)


def _port_type_from_name(name: str) -> str:
    if "pitch" in name:
        return "pitch_dock"
    if "yaw" in name:
        return "yaw_dock"
    return "generic_dock"


def _ports_compatible(src_type: str, dst_type: str) -> bool:
    if src_type == "pitch_dock":
        return dst_type == "yaw_dock"
    if src_type == "yaw_dock":
        return dst_type == "pitch_dock"
    return src_type == "generic_dock" or dst_type == "generic_dock"


def _first_compatible_free_pair(
    src_module_id: int,
    dst_module_id: int,
    records: list[_ConnectPortRecord],
    used_ports: set[tuple[int, str]],
) -> tuple[_ConnectPortRecord, _ConnectPortRecord]:
    for src in records:
        if (src_module_id, src.port_id) in used_ports:
            continue
        for dst in records:
            if (dst_module_id, dst.port_id) in used_ports:
                continue
            if _ports_compatible(src.port_type, dst.port_type):
                return src, dst
    raise SchemaValidationError("No compatible free dock port pair available for fixed morphology")


def write_resolved_mesh_urdf(
    source_urdf_path: str | Path,
    output_urdf_path: str | Path,
    *,
    mesh_search_dirs: list[str | Path] | None = None,
) -> Path:
    """Write a URDF copy with relative mesh references resolved to existing absolute paths."""

    source_path = Path(source_urdf_path).resolve()
    output_path = Path(output_urdf_path)
    source_root = ET.parse(source_path).getroot()
    if _tag_name(source_root) != "robot":
        raise SchemaValidationError(f"Expected <robot> root in {source_path}")
    _resolve_mesh_refs(source_root, source_dir=source_path.parent, mesh_search_dirs=_normalise_mesh_search_dirs(mesh_search_dirs))
    _indent(source_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(source_root).write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def write_joint_velocity_override_urdf(
    source_urdf_path: str | Path,
    output_urdf_path: str | Path,
    *,
    joint_velocity_overrides: dict[str, float],
    prefix_separator: str = FIXED_MODULE_PREFIX_SEPARATOR,
) -> Path:
    """Write a URDF copy with selected joint velocity limits overridden."""

    source_path = Path(source_urdf_path).resolve()
    output_path = Path(output_urdf_path)
    source_root = ET.parse(source_path).getroot()
    if _tag_name(source_root) != "robot":
        raise SchemaValidationError(f"Expected <robot> root in {source_path}")
    normalised = {name: float(value) for name, value in joint_velocity_overrides.items()}
    for joint_name, velocity in normalised.items():
        if velocity < 0.0:
            raise SchemaValidationError(f"Joint velocity override for {joint_name!r} must be non-negative")
    for joint in _iter_named_joints(source_root):
        joint_name = joint.attrib["name"]
        override = _matching_joint_velocity_override(
            joint_name,
            normalised,
            prefix_separator=prefix_separator,
        )
        if override is None:
            continue
        limit = _child(joint, "limit")
        if limit is None:
            limit = ET.SubElement(joint, "limit")
        limit.attrib["velocity"] = f"{override:.9g}"
    _indent(source_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(source_root).write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def fixed_module_link_name(module_id: int, local_link_name: str) -> str:
    return _prefixed_name(module_id, local_link_name, FIXED_MODULE_PREFIX_SEPARATOR)


def fixed_module_joint_name(module_id: int, local_joint_name: str) -> str:
    return _prefixed_name(module_id, local_joint_name, FIXED_MODULE_PREFIX_SEPARATOR)


def split_fixed_module_name(name: str, *, prefix_separator: str = FIXED_MODULE_PREFIX_SEPARATOR) -> tuple[int, str] | None:
    prefix = "module_"
    if not name.startswith(prefix):
        return None
    module_part, separator, local_name = name.partition(prefix_separator)
    if separator == "" or not local_name:
        return None
    module_id_text = module_part[len(prefix) :]
    if not module_id_text.isdigit():
        return None
    return int(module_id_text), local_name


def _prefix_element(
    element: ET.Element,
    *,
    prefix: str,
    source_dir: Path,
    mesh_search_dirs: list[Path],
) -> None:
    tag = _tag_name(element)
    if tag in {"link", "joint", "transmission"} and "name" in element.attrib:
        element.attrib["name"] = prefix + element.attrib["name"]
    if tag == "parent" and "link" in element.attrib:
        element.attrib["link"] = prefix + element.attrib["link"]
    if tag == "child" and "link" in element.attrib:
        element.attrib["link"] = prefix + element.attrib["link"]
    if tag == "gazebo" and "reference" in element.attrib:
        element.attrib["reference"] = prefix + element.attrib["reference"]
    if tag == "joint" and "name" in element.attrib and element.attrib["name"] and element.text is None:
        pass
    if tag == "actuator" and "name" in element.attrib:
        element.attrib["name"] = prefix + element.attrib["name"]
    if tag == "mesh" and "filename" in element.attrib:
        element.attrib["filename"] = _resolve_mesh_filename(
            element.attrib["filename"],
            source_dir=source_dir,
            mesh_search_dirs=mesh_search_dirs,
        )

    for child in list(element):
        _prefix_element(child, prefix=prefix, source_dir=source_dir, mesh_search_dirs=mesh_search_dirs)


def _resolve_mesh_refs(element: ET.Element, *, source_dir: Path, mesh_search_dirs: list[Path]) -> None:
    if _tag_name(element) == "mesh" and "filename" in element.attrib:
        element.attrib["filename"] = _resolve_mesh_filename(
            element.attrib["filename"],
            source_dir=source_dir,
            mesh_search_dirs=mesh_search_dirs,
        )
    for child in list(element):
        _resolve_mesh_refs(child, source_dir=source_dir, mesh_search_dirs=mesh_search_dirs)


def _resolve_mesh_filename(filename: str, *, source_dir: Path, mesh_search_dirs: list[Path]) -> str:
    if _is_absolute_mesh_ref(filename):
        return filename

    relative_path = Path(filename)
    candidates = [source_dir / relative_path]
    for search_dir in mesh_search_dirs:
        candidates.append(search_dir / relative_path)
        candidates.append(search_dir / relative_path.name)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str((source_dir / relative_path).resolve())


def _normalise_mesh_search_dirs(mesh_search_dirs: list[str | Path] | None) -> list[Path]:
    return [Path(search_dir).resolve() for search_dir in mesh_search_dirs or []]


def _prefixed_name(module_id: int, local_name: str, separator: str) -> str:
    return f"module_{module_id}{separator}{local_name}"


def _format_vec(vector) -> str:
    return " ".join(f"{float(value):.9g}" for value in vector)


def _named_children(root: ET.Element, tag: str) -> dict[str, ET.Element]:
    return {
        child.attrib["name"]: child
        for child in list(root)
        if _tag_name(child) == tag and child.attrib.get("name")
    }


def _iter_named_joints(root: ET.Element) -> list[ET.Element]:
    return [
        child
        for child in list(root)
        if _tag_name(child) == "joint" and child.attrib.get("name")
    ]


def _child(element: ET.Element, tag: str) -> ET.Element | None:
    for child in list(element):
        if _tag_name(child) == tag:
            return child
    return None


def _matching_joint_velocity_override(
    joint_name: str,
    overrides: dict[str, float],
    *,
    prefix_separator: str,
) -> float | None:
    if joint_name in overrides:
        return overrides[joint_name]
    parsed = split_fixed_module_name(joint_name, prefix_separator=prefix_separator)
    if parsed is not None:
        return overrides.get(parsed[1])
    return None


def _tag_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _is_absolute_mesh_ref(filename: str) -> bool:
    return filename.startswith("package://") or filename.startswith("file://") or Path(filename).is_absolute()


def _indent(element: ET.Element, level: int = 0) -> None:
    indent_text = "\n" + level * "  "
    child_indent = "\n" + (level + 1) * "  "
    children = list(element)
    if children:
        if not element.text or not element.text.strip():
            element.text = child_indent
        for child in children:
            _indent(child, level + 1)
        if not children[-1].tail or not children[-1].tail.strip():
            children[-1].tail = indent_text
    if level and (not element.tail or not element.tail.strip()):
        element.tail = indent_text
