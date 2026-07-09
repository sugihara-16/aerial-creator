from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

from amsrr.schemas.common import SchemaValidationError


FIXED_MODULE_PREFIX_SEPARATOR = "__"


def write_fixed_morphology_urdf(
    source_urdf_path: str | Path,
    output_urdf_path: str | Path,
    *,
    module_count: int = 2,
    module_spacing_m: float = 0.45,
    prefix_separator: str = FIXED_MODULE_PREFIX_SEPARATOR,
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

    source_links = _named_children(source_root, "link")
    if "root" not in source_links:
        raise SchemaValidationError("source Holon URDF must contain root link")

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
            _prefix_element(copied, prefix=prefix, source_dir=source_path.parent)
            output_root.append(copied)
        if module_id > 0:
            joint = ET.SubElement(
                output_root,
                "joint",
                {
                    "name": f"fixed_module_{module_id}_to_module_0",
                    "type": "fixed",
                },
            )
            ET.SubElement(joint, "origin", {"xyz": f"{module_spacing_m * module_id:.9g} 0 0", "rpy": "0 0 0"})
            ET.SubElement(joint, "parent", {"link": _prefixed_name(0, "root", prefix_separator)})
            ET.SubElement(joint, "child", {"link": _prefixed_name(module_id, "root", prefix_separator)})
            ET.SubElement(joint, "axis", {"xyz": "0 0 0"})

    _indent(output_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(output_root).write(output_path, encoding="utf-8", xml_declaration=True)
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


def _prefix_element(element: ET.Element, *, prefix: str, source_dir: Path) -> None:
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
        filename = element.attrib["filename"]
        if not _is_absolute_mesh_ref(filename):
            element.attrib["filename"] = str((source_dir / filename).resolve())

    for child in list(element):
        _prefix_element(child, prefix=prefix, source_dir=source_dir)


def _prefixed_name(module_id: int, local_name: str, separator: str) -> str:
    return f"module_{module_id}{separator}{local_name}"


def _named_children(root: ET.Element, tag: str) -> dict[str, ET.Element]:
    return {
        child.attrib["name"]: child
        for child in list(root)
        if _tag_name(child) == tag and child.attrib.get("name")
    }


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
