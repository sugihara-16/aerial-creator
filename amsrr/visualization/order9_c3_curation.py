from __future__ import annotations

"""Actual-mesh inspection artifacts for the Order 9 C3 curation pilot.

This module deliberately stays outside production bucket admission.  It turns a
graph-specific URDF and an optional articulated-teacher solution into a
browser-viewable scene whose geometry comes directly from the URDF mesh files.
The resulting images and HTML are inspection aids, not collision or Isaac
evidence.
"""

import base64
import json
import math
import re
import shutil
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from amsrr.geometry.pose_math import (
    Transform3D,
    compose_pose,
    compose_transform,
    inverse_pose,
    pose_from_transform,
    transform_from_pose,
    transform_from_xyz_rpy,
)
from amsrr.robot_model.urdf_loader import load_urdf
from amsrr.robot_model.urdf_transforms import link_poses_in_root_frame
from amsrr.schemas.common import Pose7D, SchemaValidationError
from amsrr.utils.hashing import hash_file


ORDER9_C3_CURATION_VIEWER_VERSION = (
    "order9_c3_curation_mesh_viewer_v3_baselink_frame_review"
)
_IDENTITY_POSE: Pose7D = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
_MODULE_LINK_PATTERN = re.compile(r"^module_(\d+)__(.+)$")


@dataclass(frozen=True)
class Order9C3ViewerMarker:
    marker_id: str
    label: str
    position_world: tuple[float, float, float]
    direction_world: tuple[float, float, float] | None = None
    color_rgba: tuple[float, float, float, float] = (1.0, 0.35, 0.05, 1.0)
    selected: bool = False
    kind: str = "anchor"

    def __post_init__(self) -> None:
        if not self.marker_id or not self.label:
            raise ValueError("viewer marker id and label must be non-empty")
        _finite_vector(self.position_world, 3, "marker position")
        if self.direction_world is not None:
            _finite_vector(self.direction_world, 3, "marker direction")
        _finite_vector(self.color_rgba, 4, "marker color")


@dataclass(frozen=True)
class Order9C3ViewerBox:
    object_id: str
    pose_world: Pose7D
    size_m: tuple[float, float, float]
    color_rgba: tuple[float, float, float, float] = (0.15, 0.68, 0.95, 0.28)

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("viewer box object_id must be non-empty")
        _finite_vector(self.pose_world, 7, "box pose")
        _finite_vector(self.size_m, 3, "box size")
        if any(float(value) <= 0.0 for value in self.size_m):
            raise ValueError("viewer box dimensions must be positive")
        _finite_vector(self.color_rgba, 4, "box color")


@dataclass(frozen=True)
class Order9C3UrdfMeshScene:
    urdf_path: Path
    root_link_id: str
    link_poses_world: dict[str, Pose7D]
    instances: tuple[dict[str, object], ...]
    module_labels: tuple[dict[str, object], ...]
    mesh_paths: tuple[Path, ...]


@dataclass(frozen=True)
class Order9C3MeshViewerArtifacts:
    html_path: Path
    scene_path: Path
    mesh_library_path: Path
    viewer_javascript_path: Path
    scene_sha256: str
    mesh_library_sha256: str
    viewer_version: str = ORDER9_C3_CURATION_VIEWER_VERSION


def build_order9_c3_urdf_mesh_scene(
    urdf_path: str | Path,
    *,
    joint_positions_rad: Mapping[str, float] | None = None,
    root_pose_world: Pose7D = _IDENTITY_POSE,
) -> Order9C3UrdfMeshScene:
    """Resolve all URDF visual/collision mesh instances at one joint pose."""

    source = Path(urdf_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    _finite_vector(root_pose_world, 7, "root_pose_world")
    root = ET.parse(source).getroot()
    if _tag(root) != "robot":
        raise SchemaValidationError(f"Expected <robot> root in {source}")
    link_elements = {
        element.attrib["name"]: element
        for element in _children(root, "link")
        if element.attrib.get("name")
    }
    if not link_elements:
        raise SchemaValidationError("mesh viewer URDF contains no links")
    joints = [_parse_joint(element) for element in _children(root, "joint")]
    child_links = [joint["child_link"] for joint in joints]
    if len(child_links) != len(set(child_links)):
        raise SchemaValidationError("mesh viewer URDF is not a link tree")
    root_links = sorted(set(link_elements) - set(child_links))
    if len(root_links) != 1:
        raise SchemaValidationError(
            f"mesh viewer URDF requires one root link, got {root_links}"
        )
    root_link = root_links[0]
    q = _normalize_urdf_joint_positions(joint_positions_rad or {})
    outgoing: dict[str, list[dict[str, object]]] = {}
    for joint in joints:
        outgoing.setdefault(str(joint["parent_link"]), []).append(joint)
    for values in outgoing.values():
        values.sort(key=lambda value: str(value["joint_id"]))

    link_transforms: dict[str, Transform3D] = {
        root_link: transform_from_pose(root_pose_world)
    }
    pending = [root_link]
    while pending:
        parent = pending.pop(0)
        for joint in outgoing.get(parent, []):
            joint_id = str(joint["joint_id"])
            joint_type = str(joint["joint_type"])
            origin = transform_from_xyz_rpy(
                joint["origin_xyz"],  # type: ignore[arg-type]
                joint["origin_rpy"],  # type: ignore[arg-type]
            )
            motion = _joint_motion_transform(
                joint_type,
                float(q.get(joint_id, 0.0)),
                joint["axis_xyz"],  # type: ignore[arg-type]
                joint_id=joint_id,
            )
            child = str(joint["child_link"])
            link_transforms[child] = compose_transform(
                link_transforms[parent],
                compose_transform(origin, motion),
            )
            pending.append(child)
    if set(link_transforms) != set(link_elements):
        missing = sorted(set(link_elements) - set(link_transforms))
        raise SchemaValidationError(
            f"mesh viewer URDF has unreachable links: {missing}"
        )

    instances: list[dict[str, object]] = []
    mesh_paths: set[Path] = set()
    for link_id, element in sorted(link_elements.items()):
        module_id, local_link_id = _module_link_identity(link_id)
        for layer in ("visual", "collision"):
            for index, item in enumerate(_children(element, layer)):
                geometry = _child(item, "geometry")
                mesh = None if geometry is None else _child(geometry, "mesh")
                if mesh is None or not mesh.attrib.get("filename"):
                    continue
                mesh_path = _resolve_mesh_path(
                    mesh.attrib["filename"],
                    urdf_directory=source.parent,
                )
                _validate_binary_stl(mesh_path)
                mesh_paths.add(mesh_path)
                origin_element = _child(item, "origin")
                local = transform_from_xyz_rpy(
                    _vector_attr(origin_element, "xyz", (0.0, 0.0, 0.0)),
                    _vector_attr(origin_element, "rpy", (0.0, 0.0, 0.0)),
                )
                transform = compose_transform(link_transforms[link_id], local)
                scale = _vector_text(
                    mesh.attrib.get("scale"),
                    default=(1.0, 1.0, 1.0),
                    label=f"{link_id} mesh scale",
                )
                rgba = (
                    _material_rgba(item)
                    if layer == "visual"
                    else (0.95, 0.18, 0.16, 0.38)
                )
                instances.append(
                    {
                        "instance_id": f"{layer}:{link_id}:{index}",
                        "layer": layer,
                        "module_id": module_id,
                        "link_id": link_id,
                        "local_link_id": local_link_id,
                        "detail_class": _detail_class(local_link_id),
                        "mesh_path": str(mesh_path),
                        "mesh_key": hash_file(mesh_path),
                        "model_matrix": _column_major_matrix(transform, scale),
                        "color_rgba": list(
                            _module_color(module_id, rgba, layer=layer)
                        ),
                    }
                )

    module_labels = []
    module_ids = sorted(
        {
            int(instance["module_id"])
            for instance in instances
            if instance["module_id"] is not None
        }
    )
    for module_id in module_ids:
        preferred = f"module_{module_id}__main_body"
        fallback = f"module_{module_id}__root"
        link_id = preferred if preferred in link_transforms else fallback
        pose = pose_from_transform(link_transforms[link_id])
        module_labels.append(
            {
                "label_id": f"module-{module_id}",
                "label": f"M{module_id}",
                "position_world": list(pose[:3]),
                "kind": "module",
                "color_rgba": list(_module_color(module_id, (1.0,) * 4)),
            }
        )
    return Order9C3UrdfMeshScene(
        urdf_path=source,
        root_link_id=root_link,
        link_poses_world={
            link_id: pose_from_transform(transform)
            for link_id, transform in link_transforms.items()
        },
        instances=tuple(instances),
        module_labels=tuple(module_labels),
        mesh_paths=tuple(sorted(mesh_paths)),
    )


def order9_c3_urdf_root_pose_from_baselink_pose(
    urdf_path: str | Path,
    baselink_pose_world: Pose7D,
) -> Pose7D:
    """Convert the PhysicalModel module-frame pose to the URDF root pose."""

    source = Path(urdf_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    _finite_vector(baselink_pose_world, 7, "baselink_pose_world")
    model = load_urdf(source)
    if len(model.root_links) != 1:
        raise SchemaValidationError(
            f"C3 viewer requires one URDF root, got {model.root_links}"
        )
    metadata = model.metadata.get("baselink")
    baselink_id = (
        metadata.get("name") if isinstance(metadata, dict) else None
    )
    if not isinstance(baselink_id, str) or not baselink_id:
        raise SchemaValidationError(
            "C3 viewer URDF lacks a baselink frame identity"
        )
    poses_root = link_poses_in_root_frame(model)
    root_to_baselink = poses_root.get(baselink_id)
    if root_to_baselink is None:
        raise SchemaValidationError(
            f"C3 viewer baselink {baselink_id!r} is absent from the URDF tree"
        )
    return compose_pose(
        tuple(float(value) for value in baselink_pose_world),
        inverse_pose(root_to_baselink),
    )


def write_order9_c3_stl_mesh_library(
    mesh_paths: Iterable[str | Path],
    output_path: str | Path,
) -> Path:
    """Write one shared browser-side library of exact binary STL bytes."""

    destination = Path(output_path).resolve()
    unique: dict[str, Path] = {}
    for value in mesh_paths:
        path = Path(value).resolve()
        _validate_binary_stl(path)
        digest = hash_file(path)
        previous = unique.get(digest)
        if previous is not None and previous.read_bytes() != path.read_bytes():
            raise SchemaValidationError("STL SHA-256 collision")
        unique[digest] = path
    if not unique:
        raise ValueError("mesh library requires at least one STL")
    payload = {
        "version": ORDER9_C3_CURATION_VIEWER_VERSION,
        "meshes": {
            digest: {
                "filename": path.name,
                "sha256": digest,
                "byte_length": path.stat().st_size,
                "base64": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
            for digest, path in sorted(unique.items())
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "window.AMSRR_ORDER9_MESH_LIBRARY="
        + json.dumps(payload, separators=(",", ":"), sort_keys=True)
        + ";\n",
        encoding="utf-8",
    )
    return destination


def write_order9_c3_viewer_javascript(output_path: str | Path) -> Path:
    source = Path(__file__).resolve().parent / "static" / "order9_c3_mesh_viewer.js"
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination


def render_order9_c3_mesh_viewer(
    *,
    urdf_scene: Order9C3UrdfMeshScene,
    html_path: str | Path,
    mesh_library_path: str | Path,
    viewer_javascript_path: str | Path,
    title: str,
    subtitle: str,
    markers: Sequence[Order9C3ViewerMarker] = (),
    boxes: Sequence[Order9C3ViewerBox] = (),
    metadata: Mapping[str, object] | None = None,
    default_view: str = "iso",
    animation: Mapping[str, object] | None = None,
    semantic_scope: str | None = None,
) -> Order9C3MeshViewerArtifacts:
    """Write a small HTML/JSON scene referencing shared exact STL bytes."""

    if default_view not in {"iso", "top", "front", "side"}:
        raise ValueError("default_view must be iso/top/front/side")
    destination = Path(html_path).resolve()
    library = Path(mesh_library_path).resolve()
    viewer_js = Path(viewer_javascript_path).resolve()
    for required in (library, viewer_js):
        if not required.is_file():
            raise FileNotFoundError(required)
    if animation is not None:
        frames = animation.get("frames")
        if not isinstance(frames, list) or not frames:
            raise SchemaValidationError(
                "mesh viewer animation requires non-empty frames"
            )
        frame_encoding = str(animation.get("frame_encoding", "model_matrices_v1"))
        if frame_encoding not in {"model_matrices_v1", "urdf_fk_v1"}:
            raise SchemaValidationError(
                "mesh viewer animation frame encoding is unsupported"
            )
        urdf_fk = animation.get("urdf_fk")
        coordinate_count = 0
        if frame_encoding == "urdf_fk_v1":
            if not isinstance(urdf_fk, dict):
                raise SchemaValidationError(
                    "URDF-FK animation requires a kinematic payload"
                )
            links = urdf_fk.get("links")
            joints = urdf_fk.get("joints")
            coordinate_joint_ids = urdf_fk.get("coordinate_joint_ids")
            if (
                not isinstance(links, list)
                or not links
                or not isinstance(joints, list)
                or not isinstance(coordinate_joint_ids, list)
            ):
                raise SchemaValidationError(
                    "URDF-FK animation kinematic payload is invalid"
                )
            coordinate_count = len(coordinate_joint_ids)
            for instance in urdf_scene.instances:
                if (
                    not isinstance(instance.get("link_index"), int)
                    or not isinstance(instance.get("local_matrix"), list)
                    or len(instance["local_matrix"]) != 16
                ):
                    raise SchemaValidationError(
                        "URDF-FK animation mesh binding is incomplete"
                    )
        for index, frame in enumerate(frames):
            if not isinstance(frame, dict):
                raise SchemaValidationError(
                    f"mesh viewer animation frame {index} is invalid"
                )
            if frame_encoding == "urdf_fk_v1":
                root_pose = frame.get("root_pose_world")
                joint_positions = frame.get("joint_positions")
                if (
                    not isinstance(root_pose, list)
                    or len(root_pose) != 7
                    or not isinstance(joint_positions, list)
                    or len(joint_positions) != coordinate_count
                ):
                    raise SchemaValidationError(
                        "URDF-FK animation frame state shape differs"
                    )
            else:
                matrices = frame.get("model_matrices")
                if not isinstance(matrices, list) or len(matrices) != len(
                    urdf_scene.instances
                ):
                    raise SchemaValidationError(
                        "mesh viewer animation frame instance identity differs"
                    )
    scene_path = destination.with_suffix(".scene.json")
    payload = {
        "viewer_version": ORDER9_C3_CURATION_VIEWER_VERSION,
        "semantic_scope": (
            semantic_scope
            or (
                "human inspection of URDF-referenced mesh geometry; not Isaac "
                "collision, dynamics, path-feasibility, or grasp-success "
                "evidence"
            )
        ),
        "title": str(title),
        "subtitle": str(subtitle),
        "default_view": default_view,
        "source_urdf_path": str(urdf_scene.urdf_path),
        "source_urdf_sha256": hash_file(urdf_scene.urdf_path),
        "root_link_id": urdf_scene.root_link_id,
        "instances": list(urdf_scene.instances),
        "labels": list(urdf_scene.module_labels),
        "markers": [
            {
                "marker_id": marker.marker_id,
                "label": marker.label,
                "position_world": list(marker.position_world),
                "direction_world": (
                    None
                    if marker.direction_world is None
                    else list(marker.direction_world)
                ),
                "color_rgba": list(marker.color_rgba),
                "selected": marker.selected,
                "kind": marker.kind,
            }
            for marker in markers
        ],
        "boxes": [
            {
                "object_id": box.object_id,
                "pose_world": list(box.pose_world),
                "size_m": list(box.size_m),
                "color_rgba": list(box.color_rgba),
            }
            for box in boxes
        ],
        "metadata": dict(metadata or {}),
        "animation": None if animation is None else dict(animation),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    scene_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    relative_library = _relative_web_path(library, destination.parent)
    relative_viewer = _relative_web_path(viewer_js, destination.parent)
    html_payload = _viewer_html(
        title=title,
        scene_payload=payload,
        mesh_library_source=relative_library,
        viewer_javascript_source=relative_viewer,
    )
    destination.write_text(html_payload, encoding="utf-8")
    return Order9C3MeshViewerArtifacts(
        html_path=destination,
        scene_path=scene_path,
        mesh_library_path=library,
        viewer_javascript_path=viewer_js,
        scene_sha256=hash_file(scene_path),
        mesh_library_sha256=hash_file(library),
    )


def transformed_link_axis(
    pose_world: Pose7D,
    axis_local: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> tuple[float, float, float]:
    rotation = transform_from_pose(pose_world).rotation
    value = (
        rotation[0][0] * axis_local[0]
        + rotation[0][1] * axis_local[1]
        + rotation[0][2] * axis_local[2],
        rotation[1][0] * axis_local[0]
        + rotation[1][1] * axis_local[1]
        + rotation[1][2] * axis_local[2],
        rotation[2][0] * axis_local[0]
        + rotation[2][1] * axis_local[1]
        + rotation[2][2] * axis_local[2],
    )
    norm = math.sqrt(sum(float(component) ** 2 for component in value))
    if norm <= 1.0e-12:
        raise ValueError("transformed axis has zero length")
    return tuple(float(component) / norm for component in value)  # type: ignore[return-value]


def _parse_joint(element: ET.Element) -> dict[str, object]:
    joint_id = element.attrib.get("name")
    joint_type = element.attrib.get("type")
    parent = _child(element, "parent")
    child = _child(element, "child")
    if (
        not joint_id
        or joint_type not in {"fixed", "revolute", "continuous", "prismatic"}
        or parent is None
        or child is None
        or not parent.attrib.get("link")
        or not child.attrib.get("link")
    ):
        raise SchemaValidationError("mesh viewer encountered an invalid URDF joint")
    origin = _child(element, "origin")
    axis = _child(element, "axis")
    return {
        "joint_id": joint_id,
        "joint_type": joint_type,
        "parent_link": parent.attrib["link"],
        "child_link": child.attrib["link"],
        "origin_xyz": _vector_attr(origin, "xyz", (0.0, 0.0, 0.0)),
        "origin_rpy": _vector_attr(origin, "rpy", (0.0, 0.0, 0.0)),
        "axis_xyz": _vector_attr(axis, "xyz", (0.0, 0.0, 1.0)),
    }


def _normalize_urdf_joint_positions(
    values: Mapping[str, float],
) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for source_id, source_value in values.items():
        value = float(source_value)
        if not math.isfinite(value):
            raise ValueError(f"joint {source_id!r} position must be finite")
        joint_id = str(source_id)
        if ":" in joint_id and joint_id.startswith("module_"):
            module_label, local_id = joint_id.split(":", 1)
            joint_id = f"{module_label}__{local_id}"
        normalized[joint_id] = value
    return normalized


def _joint_motion_transform(
    joint_type: str,
    value: float,
    axis_xyz: tuple[float, float, float],
    *,
    joint_id: str,
) -> Transform3D:
    if joint_type == "fixed":
        return transform_from_pose(_IDENTITY_POSE)
    norm = math.sqrt(sum(float(component) ** 2 for component in axis_xyz))
    if norm <= 1.0e-12:
        raise SchemaValidationError(f"joint {joint_id!r} has zero axis")
    axis = tuple(float(component) / norm for component in axis_xyz)
    if joint_type == "prismatic":
        return Transform3D(
            rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            translation=tuple(value * component for component in axis),  # type: ignore[arg-type]
        )
    half = 0.5 * value
    sine = math.sin(half)
    pose: Pose7D = (
        0.0,
        0.0,
        0.0,
        axis[0] * sine,
        axis[1] * sine,
        axis[2] * sine,
        math.cos(half),
    )
    return transform_from_pose(pose)


def _module_link_identity(link_id: str) -> tuple[int | None, str]:
    match = _MODULE_LINK_PATTERN.match(link_id)
    if match is None:
        return None, link_id
    return int(match.group(1)), match.group(2)


def _detail_class(local_link_id: str) -> str:
    if "thrust_" in local_link_id or "gimbal_link" in local_link_id:
        return "full"
    return "shape"


def _module_color(
    module_id: int | None,
    source_rgba: Sequence[float],
    *,
    layer: str = "visual",
) -> tuple[float, float, float, float]:
    if layer == "collision":
        return (0.95, 0.16, 0.12, 0.38)
    palette = (
        (0.12, 0.47, 0.71),
        (1.00, 0.50, 0.05),
        (0.17, 0.63, 0.17),
        (0.84, 0.15, 0.16),
        (0.58, 0.40, 0.74),
        (0.55, 0.34, 0.29),
        (0.89, 0.47, 0.76),
        (0.50, 0.50, 0.50),
    )
    if module_id is None:
        rgb = tuple(float(value) for value in source_rgba[:3])
    else:
        rgb = palette[module_id % len(palette)]
    alpha = min(1.0, max(0.15, float(source_rgba[3])))
    return (rgb[0], rgb[1], rgb[2], alpha)


def _column_major_matrix(
    transform: Transform3D,
    scale: tuple[float, float, float],
) -> list[float]:
    rotation = transform.rotation
    return [
        rotation[0][0] * scale[0],
        rotation[1][0] * scale[0],
        rotation[2][0] * scale[0],
        0.0,
        rotation[0][1] * scale[1],
        rotation[1][1] * scale[1],
        rotation[2][1] * scale[1],
        0.0,
        rotation[0][2] * scale[2],
        rotation[1][2] * scale[2],
        rotation[2][2] * scale[2],
        0.0,
        transform.translation[0],
        transform.translation[1],
        transform.translation[2],
        1.0,
    ]


def _material_rgba(element: ET.Element) -> tuple[float, float, float, float]:
    material = _child(element, "material")
    color = None if material is None else _child(material, "color")
    if color is None:
        return (0.72, 0.74, 0.78, 1.0)
    value = _vector_text(
        color.attrib.get("rgba"),
        default=(0.72, 0.74, 0.78, 1.0),
        label="material rgba",
        expected_length=4,
    )
    return value  # type: ignore[return-value]


def _resolve_mesh_path(filename: str, *, urdf_directory: Path) -> Path:
    if filename.startswith("file://"):
        value = Path(filename[7:])
    elif "://" in filename:
        raise SchemaValidationError(
            f"mesh viewer does not resolve URI {filename!r}"
        )
    else:
        value = Path(filename)
    path = value.resolve() if value.is_absolute() else (urdf_directory / value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _validate_binary_stl(path: Path) -> None:
    if path.suffix.lower() != ".stl":
        raise SchemaValidationError(
            f"curation viewer currently requires STL meshes, got {path}"
        )
    size = path.stat().st_size
    if size < 84:
        raise SchemaValidationError(f"STL is truncated: {path}")
    with path.open("rb") as handle:
        handle.seek(80)
        triangle_count = struct.unpack("<I", handle.read(4))[0]
    if 84 + 50 * triangle_count != size:
        raise SchemaValidationError(
            f"curation viewer requires binary STL bytes: {path}"
        )


def _viewer_html(
    *,
    title: str,
    scene_payload: Mapping[str, object],
    mesh_library_source: str,
    viewer_javascript_source: str,
) -> str:
    encoded_scene = json.dumps(
        scene_payload,
        separators=(",", ":"),
        sort_keys=True,
    ).replace("</", "<\\/")
    escaped_title = (
        str(title).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    html,body{{width:100%;height:100%;margin:0;overflow:hidden;background:#f5f7fa;color:#17202a;font-family:system-ui,sans-serif}}
    #gl{{position:absolute;inset:0;width:100%;height:100%;touch-action:none}}
    #labels{{position:absolute;inset:0;pointer-events:none;overflow:hidden}}
    .label{{position:absolute;transform:translate(-50%,-50%);padding:2px 5px;border-radius:4px;background:rgba(255,255,255,.88);border:1px solid currentColor;font:600 12px/1.15 ui-monospace,monospace;white-space:nowrap;text-shadow:0 1px white}}
    .label.module{{font-size:14px;background:rgba(255,255,255,.72)}}
    .label.selected{{font-size:13px;border-width:2px;background:#fff4bf}}
    #toolbar{{position:absolute;left:12px;top:12px;display:flex;gap:6px;flex-wrap:wrap;max-width:70%;padding:8px;border-radius:8px;background:rgba(255,255,255,.9);box-shadow:0 2px 10px #0002}}
    button,label.control{{font:13px system-ui,sans-serif;border:1px solid #aeb6bf;border-radius:5px;background:white;padding:5px 8px}}
    button.review-accept{{background:#e7f8ec;border-color:#53a96b}} button.review-reject{{background:#fdeaea;border-color:#c85b5b}} button.review-recompute{{background:#fff4d6;border-color:#c79a2b}}
    button:disabled{{opacity:.5;cursor:wait}}
    label.control{{display:flex;align-items:center;gap:4px}}
    #info{{position:absolute;right:12px;top:12px;max-width:34em;padding:10px 12px;border-radius:8px;background:rgba(255,255,255,.91);box-shadow:0 2px 10px #0002}}
    #info h1{{font-size:16px;margin:0 0 4px}} #info p{{font-size:12px;margin:0;color:#46515c}}
    #review-controls{{position:absolute;right:12px;bottom:12px;max-width:38em;padding:9px;border-radius:8px;background:rgba(255,255,255,.94);box-shadow:0 2px 10px #0002}}
    #review-controls .row{{display:flex;gap:6px;align-items:center;flex-wrap:wrap}} #review-note{{min-width:18em;flex:1;padding:5px 7px;border:1px solid #aeb6bf;border-radius:5px}} #review-status{{font:12px ui-monospace,monospace;margin-top:6px;color:#39434d}}
    #animation-controls{{position:absolute;left:50%;bottom:12px;transform:translateX(-50%);display:flex;gap:7px;align-items:center;min-width:min(60vw,760px);padding:8px 10px;border-radius:8px;background:rgba(255,255,255,.94);box-shadow:0 2px 10px #0002}}
    #animation-controls[hidden]{{display:none}} #animation-slider{{flex:1;min-width:180px}} #animation-time{{font:12px ui-monospace,monospace;min-width:13em;text-align:right}} #animation-speed{{font:13px system-ui,sans-serif;padding:4px}}
    #diagnostics{{position:absolute;left:12px;bottom:64px;width:min(520px,calc(100vw - 24px));padding:8px 10px;border-radius:8px;background:rgba(255,255,255,.93);box-shadow:0 2px 10px #0002}}
    #diagnostics[hidden]{{display:none}} #diagnostic-chart{{display:block;width:100%;height:210px}} #diagnostic-live{{font:12px ui-monospace,monospace;color:#263441;margin-top:4px;white-space:pre-wrap}}
    #status{{position:absolute;left:12px;bottom:12px;padding:6px 9px;border-radius:5px;background:rgba(0,0,0,.66);color:white;font:12px ui-monospace,monospace}}
    body.capture #toolbar,body.capture #review-controls,body.capture #animation-controls,body.capture #diagnostics{{display:none!important}} body.capture #info{{max-width:42em}}
  </style>
</head>
<body>
  <canvas id="gl"></canvas><div id="labels"></div>
  <div id="toolbar">
    <button data-view="top">Top</button><button data-view="iso">ISO</button>
    <button data-view="front">Front</button><button data-view="side">Side</button>
    <button id="fit">Fit</button>
    <label class="control"><input id="labels-toggle" type="checkbox" checked>labels</label>
    <label class="control"><input id="detail-toggle" type="checkbox">full detail</label>
    <label class="control"><input id="collision-toggle" type="checkbox">collision mesh</label>
    <label class="control"><input id="object-toggle" type="checkbox" checked>object</label>
  </div>
  <div id="info"><h1>{escaped_title}</h1><p id="subtitle"></p></div>
  <div id="review-controls" hidden>
    <div class="row">
      <button class="review-accept" data-review-action="accept">Accept</button>
      <button class="review-reject" data-review-action="reject">Reject</button>
      <button class="review-recompute" data-review-action="recalculate">再計算</button>
      <input id="review-note" type="text" placeholder="任意メモ">
    </div>
    <div id="review-status">判定待ち</div>
  </div>
  <div id="animation-controls" hidden>
    <button id="animation-play" type="button">Play</button>
    <input id="animation-slider" type="range" min="0" max="0" step="1" value="0">
    <select id="animation-speed" aria-label="playback speed">
      <option value="0.25">0.25×</option><option value="0.5">0.5×</option>
      <option value="1" selected>1×</option><option value="2">2×</option>
      <option value="4">4×</option>
    </select>
    <span id="animation-time">t=0.00 s</span>
  </div>
  <div id="diagnostics" hidden>
    <canvas id="diagnostic-chart"></canvas>
    <div id="diagnostic-live"></div>
  </div>
  <div id="status">loading exact STL geometry…</div>
  <script>window.AMSRR_ORDER9_SCENE={encoded_scene};</script>
  <script src="{mesh_library_source}"></script>
  <script src="{viewer_javascript_source}"></script>
</body>
</html>
"""


def _relative_web_path(target: Path, origin: Path) -> str:
    import os

    return Path(os.path.relpath(target, origin)).as_posix()


def _vector_attr(
    element: ET.Element | None,
    attribute: str,
    default: tuple[float, float, float],
) -> tuple[float, float, float]:
    if element is None:
        return default
    value = _vector_text(
        element.attrib.get(attribute),
        default=default,
        label=attribute,
    )
    return value  # type: ignore[return-value]


def _vector_text(
    text: str | None,
    *,
    default: tuple[float, ...],
    label: str,
    expected_length: int | None = None,
) -> tuple[float, ...]:
    if text is None:
        return default
    values = tuple(float(value) for value in text.split())
    length = len(default) if expected_length is None else expected_length
    if len(values) != length or not all(math.isfinite(value) for value in values):
        raise SchemaValidationError(f"{label} must contain {length} finite values")
    return values


def _finite_vector(values: Sequence[float], length: int, label: str) -> None:
    if len(values) != length or not all(math.isfinite(float(value)) for value in values):
        raise ValueError(f"{label} must contain {length} finite values")


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _children(element: ET.Element, tag: str) -> list[ET.Element]:
    return [child for child in list(element) if _tag(child) == tag]


def _child(element: ET.Element, tag: str) -> ET.Element | None:
    values = _children(element, tag)
    return values[0] if values else None


__all__ = [
    "ORDER9_C3_CURATION_VIEWER_VERSION",
    "Order9C3MeshViewerArtifacts",
    "Order9C3UrdfMeshScene",
    "Order9C3ViewerBox",
    "Order9C3ViewerMarker",
    "build_order9_c3_urdf_mesh_scene",
    "render_order9_c3_mesh_viewer",
    "transformed_link_axis",
    "write_order9_c3_stl_mesh_library",
    "write_order9_c3_viewer_javascript",
]
