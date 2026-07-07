from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from amsrr.schemas.common import SchemaValidationError, Vector3


SUPPORTED_JOINT_TYPES = {"fixed", "revolute", "continuous", "prismatic"}


@dataclass(frozen=True)
class URDFLink:
    name: str
    mass_kg: float
    inertia_kgm2: list[float]
    local_com: Vector3
    visual_mesh_refs: list[str] = field(default_factory=list)
    collision_mesh_refs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class URDFJoint:
    name: str
    joint_type: str
    parent_link: str
    child_link: str
    origin_xyz: Vector3
    origin_rpy: Vector3
    axis_xyz: Vector3
    limit_lower: float | None
    limit_upper: float | None
    effort_limit: float | None
    velocity_limit: float | None


@dataclass(frozen=True)
class URDFModel:
    robot_name: str
    source_path: str
    links: list[URDFLink]
    joints: list[URDFJoint]
    root_links: list[str]
    child_to_joint: dict[str, str]
    joint_type_counts: dict[str, int]
    total_mass_kg: float
    frame_tree_valid: bool
    frame_tree_errors: list[str]
    candidate_rotor_links: list[str]
    candidate_rotor_joints: list[str]
    candidate_dock_links: list[str]
    candidate_dock_joints: list[str]
    candidate_connect_joints: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


def _tag_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _children(element: ET.Element, tag: str) -> list[ET.Element]:
    return [child for child in list(element) if _tag_name(child) == tag]


def _child(element: ET.Element, tag: str) -> ET.Element | None:
    matches = _children(element, tag)
    return matches[0] if matches else None


def _parse_vector(text: str | None, *, default: Vector3) -> Vector3:
    if text is None:
        return default
    parts = text.split()
    if len(parts) != 3:
        raise SchemaValidationError(f"Expected 3-vector, got {text!r}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _parse_float_attr(element: ET.Element | None, name: str, default: float | None = None) -> float | None:
    if element is None:
        return default
    value = element.attrib.get(name)
    if value is None:
        return default
    return float(value)


def _parse_limit_float(element: ET.Element | None, name: str, *, zero_is_unknown: bool = False) -> float | None:
    value = _parse_float_attr(element, name)
    if value is None:
        return None
    if zero_is_unknown and value == 0.0:
        return None
    return value


def _mesh_refs(link_element: ET.Element, tag: str) -> list[str]:
    refs: list[str] = []
    for visual_or_collision in _children(link_element, tag):
        geometry = _child(visual_or_collision, "geometry")
        if geometry is None:
            continue
        mesh = _child(geometry, "mesh")
        if mesh is not None and mesh.attrib.get("filename"):
            refs.append(mesh.attrib["filename"])
    return refs


def _parse_link(link_element: ET.Element) -> URDFLink:
    name = link_element.attrib.get("name")
    if not name:
        raise SchemaValidationError("URDF link is missing name")

    inertial = _child(link_element, "inertial")
    mass_kg = 0.0
    inertia_kgm2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    local_com: Vector3 = (0.0, 0.0, 0.0)
    if inertial is not None:
        origin = _child(inertial, "origin")
        if origin is not None:
            local_com = _parse_vector(origin.attrib.get("xyz"), default=(0.0, 0.0, 0.0))
        mass = _child(inertial, "mass")
        mass_kg = float(mass.attrib.get("value", 0.0)) if mass is not None else 0.0
        inertia = _child(inertial, "inertia")
        if inertia is not None:
            inertia_kgm2 = [
                float(inertia.attrib.get("ixx", 0.0)),
                float(inertia.attrib.get("ixy", 0.0)),
                float(inertia.attrib.get("ixz", 0.0)),
                float(inertia.attrib.get("iyy", 0.0)),
                float(inertia.attrib.get("iyz", 0.0)),
                float(inertia.attrib.get("izz", 0.0)),
            ]

    return URDFLink(
        name=name,
        mass_kg=mass_kg,
        inertia_kgm2=inertia_kgm2,
        local_com=local_com,
        visual_mesh_refs=_mesh_refs(link_element, "visual"),
        collision_mesh_refs=_mesh_refs(link_element, "collision"),
    )


def _parse_joint(joint_element: ET.Element) -> URDFJoint:
    name = joint_element.attrib.get("name")
    joint_type = joint_element.attrib.get("type")
    if not name or not joint_type:
        raise SchemaValidationError("URDF joint is missing name or type")
    if joint_type not in SUPPORTED_JOINT_TYPES:
        raise SchemaValidationError(f"Unsupported URDF joint type {joint_type!r} in joint {name!r}")

    parent = _child(joint_element, "parent")
    child = _child(joint_element, "child")
    if parent is None or child is None:
        raise SchemaValidationError(f"URDF joint {name!r} is missing parent or child")
    parent_link = parent.attrib.get("link")
    child_link = child.attrib.get("link")
    if not parent_link or not child_link:
        raise SchemaValidationError(f"URDF joint {name!r} has invalid parent or child link")

    origin = _child(joint_element, "origin")
    axis = _child(joint_element, "axis")
    limit = _child(joint_element, "limit")
    return URDFJoint(
        name=name,
        joint_type=joint_type,
        parent_link=parent_link,
        child_link=child_link,
        origin_xyz=_parse_vector(origin.attrib.get("xyz") if origin is not None else None, default=(0.0, 0.0, 0.0)),
        origin_rpy=_parse_vector(origin.attrib.get("rpy") if origin is not None else None, default=(0.0, 0.0, 0.0)),
        axis_xyz=_parse_vector(axis.attrib.get("xyz") if axis is not None else None, default=(0.0, 0.0, 1.0)),
        limit_lower=_parse_limit_float(limit, "lower"),
        limit_upper=_parse_limit_float(limit, "upper"),
        effort_limit=_parse_limit_float(limit, "effort", zero_is_unknown=True),
        velocity_limit=_parse_limit_float(limit, "velocity", zero_is_unknown=True),
    )


def _validate_frame_tree(links: list[URDFLink], joints: list[URDFJoint]) -> tuple[list[str], bool, list[str]]:
    link_names = {link.name for link in links}
    errors: list[str] = []
    child_to_joint: dict[str, str] = {}
    for joint in joints:
        if joint.parent_link not in link_names:
            errors.append(f"joint {joint.name!r} parent link {joint.parent_link!r} is missing")
        if joint.child_link not in link_names:
            errors.append(f"joint {joint.name!r} child link {joint.child_link!r} is missing")
        if joint.child_link in child_to_joint:
            errors.append(f"link {joint.child_link!r} has multiple parent joints")
        child_to_joint[joint.child_link] = joint.name

    root_links = sorted(link_names - set(child_to_joint))
    if len(root_links) != 1:
        errors.append(f"URDF frame tree should have exactly one root link, got {root_links}")

    children_by_parent: dict[str, list[str]] = {}
    for joint in joints:
        children_by_parent.setdefault(joint.parent_link, []).append(joint.child_link)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(link_name: str) -> None:
        if link_name in visiting:
            errors.append(f"cycle detected at link {link_name!r}")
            return
        if link_name in visited:
            return
        visiting.add(link_name)
        for child in children_by_parent.get(link_name, []):
            visit(child)
        visiting.remove(link_name)
        visited.add(link_name)

    for root in root_links:
        visit(root)
    if link_names and visited != link_names:
        errors.append(f"unreachable links: {sorted(link_names - visited)}")

    return root_links, not errors, errors


def _name_contains_any(name: str, patterns: list[str]) -> bool:
    normalized_name = name.replace("_", "")
    return any(
        pattern and (pattern in name or pattern.replace("_", "") in normalized_name)
        for pattern in patterns
    )


def _metadata_from_root(root: ET.Element) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for child in list(root):
        tag = _tag_name(child)
        if tag in {"baselink", "thrust_link", "m_f_rate"}:
            metadata[tag] = dict(child.attrib)
    return metadata


def load_urdf(
    path: str | Path,
    *,
    rotor_link_patterns: list[str] | None = None,
    rotor_joint_patterns: list[str] | None = None,
    dock_link_patterns: list[str] | None = None,
    dock_joint_patterns: list[str] | None = None,
) -> URDFModel:
    """Parse a URDF or xacro-derived XML file without ROS/xacro dependencies."""

    urdf_path = Path(path)
    root = ET.parse(urdf_path).getroot()
    if _tag_name(root) != "robot":
        raise SchemaValidationError(f"Expected <robot> root in {urdf_path}")

    links = [_parse_link(element) for element in _children(root, "link")]
    joints = [_parse_joint(element) for element in _children(root, "joint")]
    root_links, frame_tree_valid, frame_tree_errors = _validate_frame_tree(links, joints)
    link_names = [link.name for link in links]
    joint_names = [joint.name for joint in joints]

    rotor_link_patterns = rotor_link_patterns or ["thrust_"]
    rotor_joint_patterns = rotor_joint_patterns or ["rotor"]
    dock_link_patterns = dock_link_patterns or ["pitch_dock", "yaw_dock", "dock_mech"]
    dock_joint_patterns = dock_joint_patterns or ["dock_mech_joint", "connect_point"]

    metadata = _metadata_from_root(root)
    metadata["link_count"] = len(links)
    metadata["joint_count"] = len(joints)
    metadata["joint_type_counts"] = dict(Counter(joint.joint_type for joint in joints))
    metadata["source_suffix"] = urdf_path.suffix
    metadata["contains_xacro_extension_tags"] = any(
        key in metadata for key in ("baselink", "thrust_link", "m_f_rate")
    )

    return URDFModel(
        robot_name=root.attrib.get("name", urdf_path.stem),
        source_path=str(urdf_path),
        links=links,
        joints=joints,
        root_links=root_links,
        child_to_joint={joint.child_link: joint.name for joint in joints},
        joint_type_counts=dict(Counter(joint.joint_type for joint in joints)),
        total_mass_kg=sum(link.mass_kg for link in links),
        frame_tree_valid=frame_tree_valid,
        frame_tree_errors=frame_tree_errors,
        candidate_rotor_links=sorted(name for name in link_names if _name_contains_any(name, rotor_link_patterns)),
        candidate_rotor_joints=sorted(
            joint.name
            for joint in joints
            if joint.joint_type == "continuous" and _name_contains_any(joint.name, rotor_joint_patterns)
        ),
        candidate_dock_links=sorted(name for name in link_names if _name_contains_any(name, dock_link_patterns)),
        candidate_dock_joints=sorted(name for name in joint_names if _name_contains_any(name, dock_joint_patterns)),
        candidate_connect_joints=sorted(name for name in joint_names if "connect_point" in name),
        metadata=metadata,
    )


def rpy_to_quat(rpy: Vector3) -> tuple[float, float, float, float]:
    """Convert URDF RPY to xyzw quaternion."""

    roll, pitch, yaw = rpy
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def pose_from_xyz_rpy(xyz: Vector3, rpy: Vector3) -> tuple[float, float, float, float, float, float, float]:
    qx, qy, qz, qw = rpy_to_quat(rpy)
    return (xyz[0], xyz[1], xyz[2], qx, qy, qz, qw)
