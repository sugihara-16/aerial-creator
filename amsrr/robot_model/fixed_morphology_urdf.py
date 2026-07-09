from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from amsrr.geometry.pose_math import (
    compose_pose,
    dock_module_relative_pose,
    inverse_pose,
    pose_to_xyz_rpy,
)
from amsrr.schemas.common import SchemaValidationError
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
    pose_root: tuple[float, float, float, float, float, float, float]


def _fixed_morphology_parent_module_ids(module_count: int) -> dict[int, int]:
    return {module_id: module_id - 1 for module_id in range(1, module_count)}


def _connect_port_records(urdf_model: URDFModel) -> list[_ConnectPortRecord]:
    link_poses_root = link_poses_in_root_frame(urdf_model)
    joints_by_name = {joint.name: joint for joint in urdf_model.joints}
    records: list[_ConnectPortRecord] = []
    for joint_name in urdf_model.candidate_connect_joints:
        joint = joints_by_name[joint_name]
        records.append(
            _ConnectPortRecord(
                port_id=joint_name,
                port_type=_port_type_from_name(joint_name + "_" + joint.parent_link),
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
