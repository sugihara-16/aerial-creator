#!/usr/bin/env python3
from __future__ import annotations

"""Run one hash-bound C3a final-pose admission in real Isaac/PhysX.

The curation artifact owns a static final pose, not a bound approach path.
This runner therefore admits only that exact pose: it validates all source
hashes, restores the authored articulation/object state, and observes exact
PhysX raw contacts from the first step through a short unforced settle.  It
never synthesizes or claims trajectory evidence.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
SCRIPT_ROOT = Path(__file__).resolve().parent


RUNNER_VERSION = "order9_c3a_real_isaac_static_admission_v1"
PRODUCTION_COLLISION_FILTER_SEMANTICS = (
    "production_random_morphology_takeoff_all_same_module_"
    "plus_intended_dock_v1"
)
DEFAULT_BATCH_MANIFEST = (
    "artifacts/p4_full/order9/c3a_contact_penetration_batch_001/manifest.json"
)
DEFAULT_REVIEW_DECISIONS = (
    "artifacts/p4_full/order9/c3a_contact_penetration_batch_001/"
    "review_decisions.json"
)
DEFAULT_OUTPUT_ROOT = (
    "artifacts/p4_full/order9/c3a_isaac_admission_batch_001"
)
DEFAULT_PHYSICAL_CONTACT_CONFIG = (
    "configs/training/order8_natural_contact.yaml"
)
STATIC_STATE_POSITION_TOLERANCE_M = 5.0e-6
STATIC_STATE_ATTITUDE_TOLERANCE_RAD = 1.0e-5
STATIC_STATE_JOINT_TOLERANCE_RAD = 5.0e-6


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_BATCH_MANIFEST)
    parser.add_argument("--review-decisions", default=DEFAULT_REVIEW_DECISIONS)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--robot-model-config",
        default="configs/robot/robot_model.yaml",
    )
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--settle-steps", type=int, default=8)
    parser.add_argument(
        "--max-contact-patches-per-body-pair",
        type=int,
        default=32,
    )
    return parser


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value.resolve()
    return (REPOSITORY_ROOT / value).resolve()


def _portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path.resolve())


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_tree(root: Path) -> tuple[str, list[dict[str, object]]]:
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        entries.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": _hash_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    payload = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), entries


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _require_vector(
    value: object,
    *,
    size: int,
    label: str,
) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{label} must have {size} values")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in result):
        raise ValueError(f"{label} contains a non-finite value")
    return result


def _locate_usd(urdf_path: Path) -> Path:
    usd_directory = urdf_path.parent / "usd_articulated_v2"
    candidates = sorted(
        path
        for path in usd_directory.glob(f"*/{urdf_path.stem}.usd*")
        if path.is_file()
    )
    if len(candidates) != 1:
        raise ValueError(
            f"expected one converted USD for {urdf_path}, found {len(candidates)}"
        )
    return candidates[0].resolve()


def _load_bound_case(
    *,
    manifest_path: Path,
    review_path: Path,
    case_id: str,
    robot_model_config_path: Path,
) -> dict[str, Any]:
    from amsrr.morphology.random_connected import morphology_structural_hash
    from amsrr.robot_model.physical_model_builder import (
        build_physical_model_from_config,
    )
    from amsrr.robot_model.urdf_loader import load_urdf
    from amsrr.robot_model.whole_structure_kinematics import (
        ordered_global_dock_joint_ids,
    )
    from amsrr.schemas.morphology import MorphologyGraph
    from amsrr.schemas.order8 import load_order8_natural_contact_config
    from amsrr.utils.hashing import stable_hash
    from amsrr.visualization.order9_c3_curation import (
        order9_c3_urdf_root_pose_from_baselink_pose,
    )

    manifest = _read_json(manifest_path)
    review = _read_json(review_path)
    matching = [
        item
        for item in manifest.get("cases", [])
        if isinstance(item, dict) and item.get("case_id") == case_id
    ]
    if len(matching) != 1:
        raise ValueError(f"manifest must contain exactly one case {case_id!r}")
    case = matching[0]
    decision = review.get("decisions", {}).get(case_id)
    if not isinstance(decision, dict):
        raise ValueError(f"review decision is missing for {case_id!r}")
    if (
        decision.get("action") != "accept"
        or decision.get("review_status") != "accepted_by_user"
        or case.get("review_status") != "accepted_by_user"
        or case.get("final_pose", {}).get("review_status")
        != "accepted_by_user"
    ):
        raise ValueError(f"{case_id!r} is not accepted by the user")

    scene_path = _resolve(case["final_pose"]["scene_path"])
    graph_path = _resolve(case["graph_path"])
    urdf_path = _resolve(case["urdf_path"])
    for label, path in (
        ("manifest", manifest_path),
        ("review", review_path),
        ("scene", scene_path),
        ("graph", graph_path),
        ("URDF", urdf_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} input is missing: {path}")
    actual_scene_sha256 = _hash_file(scene_path)
    if decision.get("scene_sha256") != actual_scene_sha256:
        raise ValueError("accepted review scene hash is stale")
    if case.get("graph_sha256") != _hash_file(graph_path):
        raise ValueError("manifest graph hash is stale")
    if case.get("urdf_sha256") != _hash_file(urdf_path):
        raise ValueError("manifest URDF hash is stale")

    morphology = MorphologyGraph.from_json(
        graph_path.read_text(encoding="utf-8")
    )
    morphology.validate()
    actual_structural_hash = morphology_structural_hash(morphology)
    if case.get("structural_hash") != actual_structural_hash:
        raise ValueError("manifest structural hash is stale")

    final_pose = case.get("final_pose")
    if not isinstance(final_pose, dict):
        raise ValueError("case final_pose is absent")
    collision_aware = final_pose.get("collision_aware")
    if (
        not isinstance(collision_aware, dict)
        or collision_aware.get("admission_status") != "accepted"
    ):
        raise ValueError("case lacks accepted collision-aware evidence")
    selected_pairs = collision_aware.get("selected_contact_pairs")
    if not isinstance(selected_pairs, list) or len(selected_pairs) != 2:
        raise ValueError("case must have exactly two selected contact pairs")
    selected_body_names = []
    for item in selected_pairs:
        if not isinstance(item, dict):
            raise ValueError("selected contact pair is malformed")
        module_id = int(item["first_module_index"])
        link_id = str(item["first_link_id"])
        selected_body_names.append(f"module_{module_id}__{link_id}")
    if len(set(selected_body_names)) != 2:
        raise ValueError("selected contact bodies must be distinct")

    ik = final_pose.get("ik")
    if not isinstance(ik, dict):
        raise ValueError("case final IK is absent")
    root_pose = _require_vector(
        ik.get("base_pose_world"),
        size=7,
        label="base_pose_world",
    )
    joint_positions = ik.get("joint_positions_rad")
    if not isinstance(joint_positions, dict) or not joint_positions:
        raise ValueError("final joint positions are absent")
    authored_joint_positions = {
        str(name): float(value) for name, value in joint_positions.items()
    }
    if (
        stable_hash(authored_joint_positions)
        != collision_aware.get("joint_solution_hash")
    ):
        raise ValueError("final joint solution hash is stale")
    physical_model = build_physical_model_from_config(
        robot_model_config_path
    )
    physical_contact_config_path = _resolve(
        DEFAULT_PHYSICAL_CONTACT_CONFIG
    )
    physical_contact_config = load_order8_natural_contact_config(
        physical_contact_config_path
    )
    expected_dock_joint_names = set(
        ordered_global_dock_joint_ids(morphology, physical_model)
    )
    if set(authored_joint_positions) != expected_dock_joint_names:
        raise ValueError(
            "final joint pose does not exactly cover the morphology Dock "
            "joint identity"
        )
    requested_joint_positions = {
        str(name).replace(":", "__", 1): float(value)
        for name, value in authored_joint_positions.items()
    }
    if any(
        not math.isfinite(value)
        for value in requested_joint_positions.values()
    ):
        raise ValueError("final joint positions contain a non-finite value")
    urdf_model = load_urdf(urdf_path)
    movable_joint_names = tuple(
        sorted(
            joint.name
            for joint in urdf_model.joints
            if joint.joint_type != "fixed"
        )
    )
    if not set(requested_joint_positions).issubset(movable_joint_names):
        raise ValueError("final Dock joint identity differs from the URDF")
    isaac_root_pose = order9_c3_urdf_root_pose_from_baselink_pose(
        urdf_path,
        root_pose,
    )
    root_conversion_position_error = _position_error(
        isaac_root_pose,
        root_pose,
    )
    root_conversion_attitude_error = _attitude_error(
        isaac_root_pose[3:7],
        root_pose[3:7],
    )
    if (
        not all(math.isfinite(value) for value in isaac_root_pose)
        or root_conversion_attitude_error > 1.0e-12
        or not 0.0608 < root_conversion_position_error < 0.0611
    ):
        raise ValueError(
            "generated articulated URDF root conversion differs from the "
            "fixed Holon fc-baselink contract"
        )

    scene = _read_json(scene_path)
    boxes = scene.get("boxes")
    if not isinstance(boxes, list) or len(boxes) != 1:
        raise ValueError("final scene must contain exactly one object box")
    object_box = boxes[0]
    if not isinstance(object_box, dict):
        raise ValueError("final scene object box is malformed")
    object_pose = _require_vector(
        object_box.get("pose_world"),
        size=7,
        label="object pose",
    )
    object_size = _require_vector(
        object_box.get("size_m"),
        size=3,
        label="object size",
    )
    if any(value <= 0.0 for value in object_size):
        raise ValueError("object size must be positive")

    usd_path = _locate_usd(urdf_path)
    usd_tree_hash, usd_files = _hash_tree(
        urdf_path.parent / "usd_articulated_v2"
    )
    return {
        "case_id": case_id,
        "manifest_path": manifest_path,
        "manifest_sha256": _hash_file(manifest_path),
        "review_path": review_path,
        "review_sha256": _hash_file(review_path),
        "scene_path": scene_path,
        "scene_sha256": actual_scene_sha256,
        "graph_path": graph_path,
        "graph_sha256": case["graph_sha256"],
        "urdf_path": urdf_path,
        "urdf_sha256": case["urdf_sha256"],
        "usd_path": usd_path,
        "usd_root_sha256": _hash_file(usd_path),
        "usd_tree_sha256": usd_tree_hash,
        "usd_files": usd_files,
        "structural_hash": actual_structural_hash,
        "morphology": morphology,
        "module_count": len(morphology.modules),
        "root_pose_world": tuple(isaac_root_pose),
        "ik_base_pose_world": root_pose,
        "root_pose_conversion": {
            "method": "generated_urdf_root_from_baselink_pose",
            "ik_baselink_pose_world": list(root_pose),
            "isaac_articulation_root_pose_world": list(isaac_root_pose),
            "position_error_m": root_conversion_position_error,
            "attitude_error_rad": root_conversion_attitude_error,
            "identity_required": False,
            "translation_contract_m": "approximately_0.060935",
            "passed": True,
        },
        "ik_baselink_body_name": "module_0__baselink",
        "joint_positions": requested_joint_positions,
        "expected_movable_joint_names": movable_joint_names,
        "expected_dock_joint_names": tuple(
            sorted(requested_joint_positions)
        ),
        "object_id": str(object_box["object_id"]),
        "object_pose_world": object_pose,
        "object_size_m": object_size,
        "selected_body_names": tuple(sorted(selected_body_names)),
        "selected_contact_penetration_limit_m": float(
            collision_aware["selected_contact_penetration_limit_m"]
        ),
        "physical_contact_config_path": physical_contact_config_path,
        "physical_contact_config_sha256": _hash_file(
            physical_contact_config_path
        ),
        "physical_contact_force_threshold_n": float(
            physical_contact_config.contact_normal_force_threshold_n
        ),
        "order8_max_force_per_contact_n": float(
            physical_contact_config.max_force_per_contact_n
        ),
        "physical_contact_penetration_noise_floor_m": float(
            physical_contact_config.contact_penetration_noise_floor_m
        ),
        "joint_solution_hash": collision_aware.get("joint_solution_hash"),
        "trajectory_path_status": "not_run_missing_bound_path",
    }


def _tensor(value: Any) -> Any:
    return value.torch if hasattr(value, "torch") else value


def _tensor_row(value: Any) -> list[float]:
    tensor = _tensor(value)
    row = tensor[0] if tensor.ndim > 1 else tensor
    return [float(item) for item in row.detach().cpu().tolist()]


def _infer_fixed_child_pose_world(
    *,
    requested_parent_pose_world: Sequence[float],
    requested_child_pose_world: Sequence[float],
    observed_parent_pose_world: Sequence[float],
) -> list[float]:
    """Compose an observed parent with its hash-bound fixed child transform."""

    from amsrr.geometry.pose_math import compose_pose, inverse_pose

    requested_parent = tuple(
        float(value) for value in requested_parent_pose_world
    )
    requested_child = tuple(
        float(value) for value in requested_child_pose_world
    )
    observed_parent = tuple(
        float(value) for value in observed_parent_pose_world
    )
    if any(
        len(pose) != 7
        for pose in (requested_parent, requested_child, observed_parent)
    ):
        raise ValueError("fixed child pose inference requires three Pose7D values")
    parent_to_child = compose_pose(
        inverse_pose(requested_parent),
        requested_child,
    )
    return list(compose_pose(observed_parent, parent_to_child))


def _position_error(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(
        sum((float(left[index]) - float(right[index])) ** 2 for index in range(3))
    )


def _attitude_error(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise ValueError("attitude error requires non-zero quaternions")
    dot = abs(
        sum(float(left[index]) * float(right[index]) for index in range(4))
        / (left_norm * right_norm)
    )
    dot = min(1.0, max(-1.0, dot))
    return 2.0 * math.acos(dot)


def _body_paths_by_name(stage: Any, root_path: str) -> dict[str, str]:
    from pxr import Sdf, Usd, UsdPhysics

    root = stage.GetPrimAtPath(Sdf.Path(root_path))
    if not root.IsValid():
        raise RuntimeError(f"robot prim is invalid: {root_path}")
    result = {}
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            result[str(prim.GetName())] = prim.GetPath().pathString
    if not result:
        raise RuntimeError("spawned robot has no rigid bodies")
    return result


def _module_id_from_prim_path(path: str) -> int | None:
    matches = re.findall(r"(?:^|/)module_(\d+)__", str(path))
    return int(matches[-1]) if matches else None


def _configure_collision_filters(
    stage: Any,
    *,
    morphology: Any,
    physical_model: Any,
    root_path: str,
) -> dict[str, object]:
    """Reuse the production random-morphology Isaac collision scope."""

    from pxr import UsdPhysics

    from amsrr.simulation.random_morphology_takeoff import (
        intended_dock_body_link_pairs,
    )

    root_prefix = root_path.rstrip("/") + "/"
    bodies_by_module = {
        int(module.module_id): [] for module in morphology.modules
    }
    for prim in stage.Traverse():
        prim_path = prim.GetPath().pathString
        if not prim_path.startswith(root_prefix):
            continue
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        module_id = _module_id_from_prim_path(prim_path)
        if module_id in bodies_by_module:
            bodies_by_module[module_id].append(prim)
    missing = sorted(
        module_id
        for module_id, prims in bodies_by_module.items()
        if not prims
    )
    if missing:
        raise RuntimeError(
            f"collision filtering found no rigid bodies for modules {missing}"
        )

    filtered_count = 0

    def filter_pair(source: Any, target: Any) -> None:
        nonlocal filtered_count
        api = UsdPhysics.FilteredPairsAPI.Apply(source)
        api.CreateFilteredPairsRel().AddTarget(target.GetPath())
        filtered_count += 1

    direct_body_by_module_link = {
        (
            module_id,
            str(prim.GetName()).removeprefix(f"module_{module_id}__"),
        ): prim
        for module_id, prims in bodies_by_module.items()
        for prim in prims
        if str(prim.GetName()).startswith(f"module_{module_id}__")
    }
    same_module_count = 0
    for module_id in sorted(bodies_by_module):
        module_prims = sorted(
            bodies_by_module[module_id],
            key=lambda prim: prim.GetPath().pathString,
        )
        for source_index, source in enumerate(module_prims):
            for target in module_prims[source_index + 1 :]:
                filter_pair(source, target)
                same_module_count += 1

    intended_paths = []
    intended_links = intended_dock_body_link_pairs(
        morphology,
        physical_model,
    )
    for source_module, source_link, target_module, target_link in intended_links:
        source = direct_body_by_module_link.get(
            (source_module, source_link)
        )
        target = direct_body_by_module_link.get(
            (target_module, target_link)
        )
        if source is None or target is None:
            raise RuntimeError(
                "intended dock body did not resolve to USD rigid bodies: "
                f"{source_module}:{source_link} -> "
                f"{target_module}:{target_link}"
            )
        filter_pair(source, target)
        intended_paths.append(
            tuple(
                sorted(
                    (
                        source.GetPath().pathString,
                        target.GetPath().pathString,
                    )
                )
            )
        )
    if filtered_count != same_module_count + len(intended_paths):
        raise RuntimeError(
            "collision-filter scope differs from production random "
            "morphology takeoff"
        )
    module_ids = sorted(bodies_by_module)
    adjacent_module_pairs = {
        tuple(sorted((edge.src_module_id, edge.dst_module_id)))
        for edge in morphology.dock_edges
    }
    module_pair_count = len(module_ids) * (len(module_ids) - 1) // 2
    return {
        "semantics": PRODUCTION_COLLISION_FILTER_SEMANTICS,
        "isaac_monitored_contact_scope": (
            "cross_module_non_dock_plus_robot_object"
        ),
        "same_module_contact_owner": (
            "upstream_exact_mesh_penetration_gate"
        ),
        "rigid_body_count": sum(
            len(prims) for prims in bodies_by_module.values()
        ),
        "filtered_body_pair_count": filtered_count,
        "same_module_filtered_body_pair_count": same_module_count,
        "intended_dock_filtered_body_pair_count": len(intended_paths),
        "intended_dock_body_link_pairs": intended_links,
        "intended_dock_body_path_pairs": sorted(intended_paths),
        "adjacent_module_pair_count": len(adjacent_module_pairs),
        "cross_module_pair_count": module_pair_count,
        "nonadjacent_module_pair_count": (
            module_pair_count - len(adjacent_module_pairs)
        ),
        "body_paths_by_module": {
            module_id: sorted(
                prim.GetPath().pathString for prim in prims
            )
            for module_id, prims in bodies_by_module.items()
        },
}


def _activate_contact_reports(stage: Any, *, root_path: str) -> int:
    from pxr import PhysxSchema, UsdPhysics

    prefix = root_path.rstrip("/") + "/"
    applied = 0
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if path != root_path and not path.startswith(prefix):
            continue
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        PhysxSchema.PhysxContactReportAPI.Apply(
            prim
        ).CreateThresholdAttr().Set(0.0)
        applied += 1
    if applied == 0:
        raise RuntimeError("robot has no rigid bodies for contact reporting")
    return applied


def _create_cross_module_contact_views(
    physics_view: Any,
    *,
    body_paths_by_module: Mapping[int, Sequence[str]],
    max_patches_per_body_pair: int,
) -> list[dict[str, object]]:
    views = []
    module_ids = sorted(int(value) for value in body_paths_by_module)
    for source_index, source_module in enumerate(module_ids):
        for target_module in module_ids[source_index + 1 :]:
            sensors = sorted(body_paths_by_module[source_module])
            filters = sorted(body_paths_by_module[target_module])
            capacity = (
                len(sensors)
                * len(filters)
                * int(max_patches_per_body_pair)
            )
            view = physics_view.create_rigid_contact_view(
                sensors,
                filter_patterns=[list(filters) for _ in sensors],
                max_contact_data_count=capacity,
            )
            if (
                int(view.sensor_count) != len(sensors)
                or int(view.filter_count) != len(filters)
                or int(view.max_contact_data_count) != capacity
            ):
                raise RuntimeError(
                    "cross-module contact view layout mismatch for "
                    f"{source_module}-{target_module}"
                )
            views.append(
                {
                    "module_pair": (source_module, target_module),
                    "view": view,
                    "raw_contact_capacity": capacity,
                    "sensor_paths": sensors,
                    "filter_paths": filters,
                }
            )
    return views


def _included_contact_matrix_indices(
    *,
    module_pair: tuple[int, int],
    sensor_paths: Sequence[str],
    filter_paths: Sequence[str],
) -> tuple[int, ...]:
    """Select unique, non-diagonal rigid-body pairs from a contact matrix."""

    indices = []
    for sensor_index, sensor_path in enumerate(sensor_paths):
        for filter_index, filter_path in enumerate(filter_paths):
            if sensor_path == filter_path:
                continue
            if (
                module_pair[0] == module_pair[1]
                and sensor_path > filter_path
            ):
                continue
            indices.append(sensor_index * len(filter_paths) + filter_index)
    return tuple(indices)


def _measure_cross_module_contacts(
    views: Sequence[Mapping[str, object]],
    *,
    dt: float,
    torch_module: Any,
    warp_module: Any,
    force_threshold_n: float,
    penetration_noise_floor_m: float,
) -> dict[str, object]:
    raw_count = 0
    physical_count = 0
    physical_pair_count = 0
    maximum_force = 0.0
    minimum_separation: float | None = None
    saturated = False
    nonfinite = False
    counts_by_pair = {}
    physical_counts_by_pair = {}
    body_pair_counts = {}
    body_pair_evidence = {}
    for entry in views:
        view = entry["view"]
        (
            force_buffer,
            _point_buffer,
            _normal_buffer,
            separation_buffer,
            count_buffer,
            start_buffer,
        ) = view.get_contact_data(dt)
        counts = warp_module.to_torch(count_buffer).reshape(-1).to(
            torch_module.int64
        )
        starts = warp_module.to_torch(start_buffer).reshape(-1).to(
            torch_module.int64
        )
        module_pair = entry["module_pair"]
        sensor_paths = list(entry["sensor_paths"])
        filter_paths = list(entry["filter_paths"])
        included_indices = _included_contact_matrix_indices(
            module_pair=module_pair,
            sensor_paths=sensor_paths,
            filter_paths=filter_paths,
        )
        counts_host = [int(value) for value in counts.detach().cpu().tolist()]
        starts_host = [int(value) for value in starts.detach().cpu().tolist()]
        included_counts = [counts_host[index] for index in included_indices]
        capacity = int(entry["raw_contact_capacity"])
        forces = warp_module.to_torch(force_buffer).reshape(-1)
        separations = warp_module.to_torch(separation_buffer).reshape(-1)
        pair_physical_count = 0
        for index, count in zip(
            included_indices,
            included_counts,
            strict=True,
        ):
            if count <= 0:
                continue
            start = starts_host[index]
            if start < 0 or start + count > capacity:
                saturated = True
            stop = min(max(start, 0) + count, capacity)
            active_indices = torch_module.arange(
                max(start, 0),
                stop,
                dtype=torch_module.long,
                device=forces.device,
            )
            pair_forces = forces.index_select(0, active_indices)
            pair_separations = separations.index_select(
                0, active_indices
            )
            evidence = _physical_contact_patch_evidence(
                patch_forces_n=pair_forces.detach().cpu().tolist(),
                patch_separations_m=(
                    pair_separations.detach().cpu().tolist()
                ),
                force_threshold_n=force_threshold_n,
                penetration_noise_floor_m=penetration_noise_floor_m,
            )
            sensor_index = index // len(filter_paths)
            filter_index = index % len(filter_paths)
            body_pair = tuple(
                sorted(
                    (
                        sensor_paths[sensor_index].rsplit("/", 1)[-1],
                        filter_paths[filter_index].rsplit("/", 1)[-1],
                    )
                )
            )
            body_pair_key = "|".join(body_pair)
            body_pair_counts[body_pair_key] = (
                body_pair_counts.get(body_pair_key, 0) + count
            )
            body_pair_evidence[body_pair_key] = evidence
            nonfinite = bool(nonfinite or not evidence["finite"])
            pair_physical_count += int(evidence["physical_patch_count"])
            if bool(evidence["physical_contact"]):
                physical_pair_count += 1
            maximum_force = max(
                maximum_force,
                float(evidence["raw_contact_max_force_n"]),
            )
            pair_minimum = evidence["raw_contact_min_separation_m"]
            if pair_minimum is not None:
                minimum_separation = (
                    float(pair_minimum)
                    if minimum_separation is None
                    else min(minimum_separation, float(pair_minimum))
                )
        pair_count = sum(included_counts)
        pair_key = f"{module_pair[0]}-{module_pair[1]}"
        counts_by_pair[pair_key] = pair_count
        physical_counts_by_pair[pair_key] = pair_physical_count
        raw_count += pair_count
        physical_count += pair_physical_count
        if pair_count >= capacity or any(
            starts_host[index] + counts_host[index] > capacity
            for index in included_indices
        ):
            saturated = True
    return {
        "raw_contact_count": raw_count,
        "physical_contact_count": physical_count,
        "physical_contact_pair_count": physical_pair_count,
        "raw_contact_max_force_n": maximum_force,
        "raw_contact_min_separation_m": minimum_separation,
        "raw_contact_saturated": saturated,
        "raw_contact_nonfinite": nonfinite,
        "force_threshold_n": float(force_threshold_n),
        "penetration_noise_floor_m": float(
            penetration_noise_floor_m
        ),
        "pair_raw_contact_counts": counts_by_pair,
        "pair_physical_contact_counts": physical_counts_by_pair,
        "body_pair_raw_contact_counts": dict(
            sorted(body_pair_counts.items())
        ),
        "body_pair_contact_evidence": dict(
            sorted(body_pair_evidence.items())
        ),
    }


def _rigid_body_path(stage: Any, root_path: str) -> str:
    from pxr import Sdf, Usd, UsdPhysics

    root = stage.GetPrimAtPath(Sdf.Path(root_path))
    if not root.IsValid():
        raise RuntimeError(f"rigid object prim is invalid: {root_path}")
    bodies = [
        prim.GetPath().pathString
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if len(bodies) != 1:
        raise RuntimeError(
            f"expected one rigid body below {root_path}, found {len(bodies)}"
        )
    return bodies[0]


def _owning_body(
    path: str,
    *,
    robot_body_paths: Mapping[str, str],
    object_body_path: str,
) -> tuple[str, str] | None:
    candidates = [
        ("robot", name, body_path)
        for name, body_path in robot_body_paths.items()
        if path == body_path or path.startswith(body_path.rstrip("/") + "/")
    ]
    if path == object_body_path or path.startswith(
        object_body_path.rstrip("/") + "/"
    ):
        candidates.append(("object", "object", object_body_path))
    if not candidates:
        return None
    kind, name, _ = max(candidates, key=lambda item: len(item[2]))
    return kind, name


def _classify_initial_pairs(
    raw_pairs: Sequence[Sequence[str]],
    *,
    robot_body_paths: Mapping[str, str],
    object_body_path: str,
    selected_body_names: Sequence[str],
) -> dict[str, object]:
    selected = set(selected_body_names)
    robot_robot = set()
    selected_robot_object = set()
    prohibited_robot_object = set()
    unclassified_robot_related = set()
    external = set()
    for raw in raw_pairs:
        path0, path1 = str(raw[0]), str(raw[1])
        pair = tuple(sorted((path0, path1)))
        owner0 = _owning_body(
            path0,
            robot_body_paths=robot_body_paths,
            object_body_path=object_body_path,
        )
        owner1 = _owning_body(
            path1,
            robot_body_paths=robot_body_paths,
            object_body_path=object_body_path,
        )
        owners = (owner0, owner1)
        if owner0 is None and owner1 is None:
            external.add(pair)
            continue
        if owner0 is None or owner1 is None:
            unclassified_robot_related.add(pair)
            continue
        kinds = {owner0[0], owner1[0]}
        if kinds == {"robot"}:
            robot_robot.add(pair)
            continue
        if kinds == {"robot", "object"}:
            robot_name = owner0[1] if owner0[0] == "robot" else owner1[1]
            if robot_name in selected:
                selected_robot_object.add(robot_name)
            else:
                prohibited_robot_object.add(pair)
            continue
        external.add(pair)
    return {
        "raw_pair_count": len(raw_pairs),
        "robot_robot_pairs": [list(item) for item in sorted(robot_robot)],
        "selected_robot_object_body_names": sorted(selected_robot_object),
        "prohibited_robot_object_pairs": [
            list(item) for item in sorted(prohibited_robot_object)
        ],
        "unclassified_robot_related_pairs": [
            list(item) for item in sorted(unclassified_robot_related)
        ],
        "external_pairs": [list(item) for item in sorted(external)],
    }


def _physical_contact_patch_evidence(
    *,
    patch_forces_n: Sequence[float],
    patch_separations_m: Sequence[float],
    force_threshold_n: float,
    penetration_noise_floor_m: float,
) -> dict[str, object]:
    """Classify raw PhysX patches with the production Order 8 noise floors."""

    if len(patch_forces_n) != len(patch_separations_m):
        raise ValueError("contact patch force/separation lengths differ")
    force_threshold = float(force_threshold_n)
    penetration_floor = float(penetration_noise_floor_m)
    if (
        not math.isfinite(force_threshold)
        or force_threshold <= 0.0
        or not math.isfinite(penetration_floor)
        or penetration_floor <= 0.0
    ):
        raise ValueError("physical-contact thresholds must be finite and positive")
    forces = [float(value) for value in patch_forces_n]
    separations = [float(value) for value in patch_separations_m]
    finite = all(
        math.isfinite(force) and math.isfinite(separation)
        for force, separation in zip(forces, separations, strict=True)
    )
    physical_indices = [
        index
        for index, (force, separation) in enumerate(
            zip(forces, separations, strict=True)
        )
        if math.isfinite(force)
        and math.isfinite(separation)
        and (
            abs(force) >= force_threshold
            or separation <= -penetration_floor
        )
    ]
    finite_forces = [abs(force) for force in forces if math.isfinite(force)]
    finite_separations = [
        separation for separation in separations if math.isfinite(separation)
    ]
    return {
        "raw_contact_count": len(forces),
        "finite": finite,
        "physical_contact": bool(physical_indices),
        "physical_patch_count": len(physical_indices),
        "physical_patch_indices": physical_indices,
        "raw_contact_max_force_n": max(finite_forces, default=0.0),
        "raw_contact_min_separation_m": (
            min(finite_separations) if finite_separations else None
        ),
        "force_threshold_n": force_threshold,
        "penetration_noise_floor_m": penetration_floor,
        "force_threshold_met": any(
            force >= force_threshold for force in finite_forces
        ),
        "penetration_noise_floor_exceeded": any(
            separation <= -penetration_floor
            for separation in finite_separations
        ),
    }


def _measure_object_contacts(
    view: Any,
    *,
    body_names: Sequence[str],
    dt: float,
    torch_module: Any,
    warp_module: Any,
    force_threshold_n: float,
    penetration_noise_floor_m: float,
) -> dict[str, object]:
    (
        force_buffer,
        _point_buffer,
        _normal_buffer,
        separation_buffer,
        count_buffer,
        start_buffer,
    ) = view.get_contact_data(dt)
    counts = warp_module.to_torch(count_buffer).reshape(-1).to(
        torch_module.int64
    )
    starts = warp_module.to_torch(start_buffer).reshape(-1).to(
        torch_module.int64
    )
    force_values = warp_module.to_torch(force_buffer).reshape(-1)
    separations = warp_module.to_torch(separation_buffer).reshape(-1)
    if counts.numel() != len(body_names):
        raise RuntimeError("robot-object contact count layout mismatch")
    capacity = int(view.max_contact_data_count)
    active_by_body: dict[str, dict[str, object]] = {}
    physical_by_body: dict[str, dict[str, object]] = {}
    saturated = False
    nonfinite = False
    physical_patch_count = 0
    for index, body_name in enumerate(body_names):
        count = int(counts[index].detach().cpu())
        start = int(starts[index].detach().cpu())
        if count <= 0:
            continue
        if start < 0:
            saturated = True
            continue
        stop = min(start + count, capacity)
        if start + count > capacity:
            saturated = True
        active_indices = torch_module.arange(
            start,
            stop,
            dtype=torch_module.long,
            device=force_values.device,
        )
        body_forces = force_values.index_select(0, active_indices)
        body_separations = separations.index_select(0, active_indices)
        evidence = _physical_contact_patch_evidence(
            patch_forces_n=body_forces.detach().cpu().tolist(),
            patch_separations_m=body_separations.detach().cpu().tolist(),
            force_threshold_n=force_threshold_n,
            penetration_noise_floor_m=penetration_noise_floor_m,
        )
        active_by_body[str(body_name)] = evidence
        nonfinite = bool(nonfinite or not evidence["finite"])
        physical_patch_count += int(evidence["physical_patch_count"])
        if bool(evidence["physical_contact"]):
            physical_by_body[str(body_name)] = evidence
    return {
        "active_by_body": active_by_body,
        "physical_by_body": physical_by_body,
        "raw_contact_count": int(counts.sum().detach().cpu()),
        "physical_contact_count": physical_patch_count,
        "raw_contact_capacity": capacity,
        "raw_contact_saturated": saturated,
        "raw_contact_nonfinite": nonfinite,
        "force_threshold_n": float(force_threshold_n),
        "penetration_noise_floor_m": float(
            penetration_noise_floor_m
        ),
    }


def _gpu_snapshot() -> dict[str, object]:
    query = (
        "index,name,memory.total,memory.used,utilization.gpu,"
        "utilization.memory,temperature.gpu,power.draw"
    )
    try:
        process = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        return {"available": False, "error": str(error)}
    rows = []
    for line in process.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 8:
            continue
        rows.append(
            {
                "index": int(values[0]),
                "name": values[1],
                "memory_total_mib": float(values[2]),
                "memory_used_mib": float(values[3]),
                "gpu_utilization_percent": float(values[4]),
                "memory_utilization_percent": float(values[5]),
                "temperature_c": float(values[6]),
                "power_draw_w": float(values[7]),
            }
        )
    return {"available": bool(rows), "devices": rows}


def _state_evidence(
    robot: Any,
    object_asset: Any,
    *,
    requested_root_pose: Sequence[float],
    requested_joint_by_name: Mapping[str, float],
    requested_object_pose: Sequence[float],
) -> dict[str, object]:
    root_pose = _tensor_row(robot.data.root_pose_w)
    object_pose = _tensor_row(object_asset.data.root_pose_w)
    joint_values = _tensor_row(robot.data.joint_pos)
    joint_by_name = dict(zip(robot.joint_names, joint_values, strict=True))
    missing = sorted(set(requested_joint_by_name) - set(joint_by_name))
    if missing:
        raise RuntimeError(f"spawned articulation lacks requested joints: {missing}")
    joint_errors = {
        name: abs(float(joint_by_name[name]) - float(target))
        for name, target in requested_joint_by_name.items()
    }
    evidence = {
        "requested_root_pose_world": list(requested_root_pose),
        "observed_root_pose_world": root_pose,
        "root_position_error_m": _position_error(
            root_pose, requested_root_pose
        ),
        "root_attitude_error_rad": _attitude_error(
            root_pose[3:7], requested_root_pose[3:7]
        ),
        "requested_object_pose_world": list(requested_object_pose),
        "observed_object_pose_world": object_pose,
        "object_position_error_m": _position_error(
            object_pose, requested_object_pose
        ),
        "object_attitude_error_rad": _attitude_error(
            object_pose[3:7], requested_object_pose[3:7]
        ),
        "requested_joint_count": len(requested_joint_by_name),
        "spawned_joint_count": len(robot.joint_names),
        "maximum_requested_joint_error_rad": max(
            joint_errors.values(), default=0.0
        ),
        "joint_errors_rad": dict(sorted(joint_errors.items())),
    }
    evidence["passed"] = bool(
        evidence["root_position_error_m"]
        <= STATIC_STATE_POSITION_TOLERANCE_M
        and evidence["root_attitude_error_rad"]
        <= STATIC_STATE_ATTITUDE_TOLERANCE_RAD
        and evidence["object_position_error_m"]
        <= STATIC_STATE_POSITION_TOLERANCE_M
        and evidence["object_attitude_error_rad"]
        <= STATIC_STATE_ATTITUDE_TOLERANCE_RAD
        and evidence["maximum_requested_joint_error_rad"]
        <= STATIC_STATE_JOINT_TOLERANCE_RAD
    )
    return evidence


def _write_report(path: Path, report: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_batch_summary(
    *,
    output_root: Path,
    manifest_path: Path,
    review_path: Path,
) -> Path:
    batch = _read_json(manifest_path)
    review = _read_json(review_path)
    decisions = review.get("decisions", {})
    rows = []
    for case in batch.get("cases", []):
        if not isinstance(case, dict):
            continue
        case_id = str(case["case_id"])
        decision = (
            decisions.get(case_id) if isinstance(decisions, dict) else None
        )
        report_path = output_root / "cases" / case_id / "admission.json"
        row: dict[str, object] = {
            "case_id": case_id,
            "structural_hash": case["structural_hash"],
            "accepted_scene_sha256": (
                decision.get("scene_sha256")
                if isinstance(decision, dict)
                else None
            ),
            "report_path": _portable(report_path),
        }
        if report_path.is_file():
            report = _read_json(report_path)
            settle = report.get("settle", {})
            if not isinstance(settle, dict):
                settle = {}
            selected_forces = settle.get(
                "maximum_selected_force_n", {}
            )
            if not isinstance(selected_forces, dict):
                selected_forces = {}
            selected_force_limit = settle.get(
                "order8_selected_contact_force_limit_n"
            )
            maximum_selected_force = (
                max(float(value) for value in selected_forces.values())
                if selected_forces
                else None
            )
            force_limit_passed = (
                maximum_selected_force is not None
                and selected_force_limit is not None
                and maximum_selected_force <= float(selected_force_limit)
            )
            static_passed = bool(report.get("passed", False))
            row.update(
                {
                    "status": report.get("status"),
                    "passed": static_passed,
                    "result_semantics": (
                        "static_pose_collision_contact_pass"
                        if static_passed
                        else "static_pose_collision_contact_fail"
                    ),
                    "failure_reasons": list(
                        report.get("failure_reasons", [])
                    ),
                    "report_sha256": _hash_file(report_path),
                    "wall_time_s": (
                        report.get("resource_usage", {}).get("wall_time_s")
                        if isinstance(report.get("resource_usage"), dict)
                        else None
                    ),
                    "trajectory_path_status": (
                        report.get("method", {}).get(
                            "trajectory_path_status"
                        )
                        if isinstance(report.get("method"), dict)
                        else None
                    ),
                    "maximum_selected_force_n": maximum_selected_force,
                    "order8_selected_contact_force_limit_n": (
                        selected_force_limit
                    ),
                    "order8_selected_contact_force_limit_passed": (
                        force_limit_passed
                    ),
                    "production_isaac_admission_eligible": False,
                    "production_eligibility_limitations": [
                        "trajectory_path_not_run_missing_bound_path",
                        *(
                            []
                            if force_limit_passed
                            else [
                                "order8_selected_contact_force_limit_"
                                "not_demonstrated_or_exceeded"
                            ]
                        ),
                    ],
                }
            )
        else:
            row.update(
                {
                    "status": "not_run",
                    "passed": False,
                    "result_semantics": "not_run",
                    "failure_reasons": ["not_run"],
                    "production_isaac_admission_eligible": False,
                    "production_eligibility_limitations": ["not_run"],
                }
            )
        rows.append(row)
    summary = {
        "report_version": "order9_c3a_real_isaac_admission_batch_v1",
        "semantic_scope": (
            "exact accepted final pose only; no trajectory/path claim"
        ),
        "passed_field_semantics": "static_pose_collision_contact_pass",
        "method": {
            "admission_gates": [
                "exact_static_spawn_readback",
                "selected_robot_object_contact",
                "selected_contact_penetration_limit",
                "no_cross_module_non_dock_contact",
                "no_nonselected_robot_object_contact",
            ],
            "final_state_drift_gate_role": (
                "diagnostic_only_not_an_admission_gate"
            ),
            "trajectory_reachability": (
                "out_of_scope_missing_hash_bound_collision_admitted_path"
            ),
            "order8_selected_contact_force_limit_role": (
                "reported_limitation_not_static_pose_gate"
            ),
        },
        "source_binding": {
            "batch_manifest_path": _portable(manifest_path),
            "batch_manifest_sha256": _hash_file(manifest_path),
            "review_decisions_path": _portable(review_path),
            "review_decisions_sha256": _hash_file(review_path),
            "runner_path": _portable(Path(__file__).resolve()),
            "runner_sha256": _hash_file(Path(__file__).resolve()),
        },
        "case_count": len(rows),
        "passed_count": sum(bool(row["passed"]) for row in rows),
        "static_pose_collision_contact_pass_count": sum(
            bool(row["passed"]) for row in rows
        ),
        "production_isaac_admission_eligible_count": sum(
            bool(row["production_isaac_admission_eligible"])
            for row in rows
        ),
        "failed_count": sum(
            row["status"] == "fail" for row in rows
        ),
        "not_run_count": sum(
            row["status"] == "not_run" for row in rows
        ),
        "production_bucket_promoted": False,
        "learning_performed": False,
        "checkpoint_updated": False,
        "cases": rows,
    }
    summary_path = output_root / "manifest.json"
    _write_report(summary_path, summary)
    return summary_path


def _run_isaac(
    args: argparse.Namespace,
    bound: Mapping[str, Any],
    *,
    output_path: Path,
) -> dict[str, object]:
    started_wall = time.perf_counter()
    started_usage = resource.getrusage(resource.RUSAGE_SELF)
    runner_path = Path(__file__).resolve()
    runner_sha256_at_start = _hash_file(runner_path)
    from isaaclab.app import AppLauncher

    parser = _parser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    try:
        import torch
        import warp as wp

        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import (
            Articulation,
            ArticulationCfg,
            RigidObject,
            RigidObjectCfg,
        )
        from isaaclab.sim import SimulationContext
        from amsrr.robot_model.physical_model_builder import (
            build_physical_model_from_config,
        )
        gpu_samples = [_gpu_snapshot()]
        device = str(args_cli.device)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        morphology = bound["morphology"]
        physical_model = build_physical_model_from_config(
            _resolve(args.robot_model_config)
        )
        initial_joint_map = {
            name: float(value)
            for name, value in bound["joint_positions"].items()
        }

        def create_runtime() -> tuple[
            Any,
            Any,
            Any,
            dict[str, object],
            int,
            dict[str, str],
        ]:
            sim_utils.create_new_stage()
            runtime_sim = SimulationContext(
                sim_utils.SimulationCfg(
                    dt=float(args.dt),
                    device=device,
                    use_fabric=True,
                )
            )
            runtime_robot = Articulation(
                ArticulationCfg(
                    prim_path="/World/Robot",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=str(bound["usd_path"]),
                        activate_contact_sensors=True,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            disable_gravity=True,
                            max_depenetration_velocity=2.0,
                            enable_gyroscopic_forces=True,
                        ),
                        articulation_props=(
                            sim_utils.ArticulationRootPropertiesCfg(
                                enabled_self_collisions=True,
                                solver_position_iteration_count=8,
                                solver_velocity_iteration_count=8,
                                sleep_threshold=0.0,
                                stabilization_threshold=0.001,
                            )
                        ),
                        copy_from_source=False,
                    ),
                    init_state=ArticulationCfg.InitialStateCfg(
                        pos=tuple(bound["root_pose_world"][:3]),
                        rot=tuple(bound["root_pose_world"][3:7]),
                        joint_pos=initial_joint_map,
                        joint_vel={".*": 0.0},
                    ),
                    actuators={
                        "gimbal_joints": ImplicitActuatorCfg(
                            joint_names_expr=[".*gimbal.*"],
                            stiffness=20.0,
                            damping=1.0,
                        ),
                        "dock_joints": ImplicitActuatorCfg(
                            joint_names_expr=[".*dock_mech.*"],
                            stiffness=20.0,
                            damping=1.0,
                        ),
                        "rotor_spinner_joints": ImplicitActuatorCfg(
                            joint_names_expr=[".*rotor.*"],
                            stiffness=0.0,
                            damping=0.0,
                        ),
                    },
                )
            )
            runtime_object = RigidObject(
                RigidObjectCfg(
                    prim_path="/World/Object",
                    spawn=sim_utils.CuboidCfg(
                        size=tuple(bound["object_size_m"]),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            disable_gravity=True,
                            max_depenetration_velocity=2.0,
                        ),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        activate_contact_sensors=True,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.72, 0.38, 0.12)
                        ),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=tuple(bound["object_pose_world"][:3]),
                        rot=tuple(bound["object_pose_world"][3:7]),
                    ),
                )
            )
            runtime_filter_evidence = _configure_collision_filters(
                runtime_sim.stage,
                morphology=morphology,
                physical_model=physical_model,
                root_path="/World/Robot",
            )
            runtime_body_paths = _body_paths_by_name(
                runtime_sim.stage, "/World/Robot"
            )
            runtime_contact_report_body_count = _activate_contact_reports(
                runtime_sim.stage,
                root_path="/World/Robot",
            )
            runtime_sim.reset()
            return (
                runtime_sim,
                runtime_robot,
                runtime_object,
                runtime_filter_evidence,
                runtime_contact_report_body_count,
                runtime_body_paths,
            )

        (
            sim,
            robot,
            object_asset,
            filter_evidence,
            contact_report_body_count,
            body_paths,
        ) = create_runtime()

        if set(robot.joint_names) != set(
            bound["expected_movable_joint_names"]
        ):
            raise RuntimeError(
                "spawned Isaac articulation joint identity differs from the "
                "generated URDF movable-joint identity"
            )
        requested_joint = torch.zeros(
            (1, len(robot.joint_names)),
            dtype=torch.float32,
            device=device,
        )
        unknown_joints = sorted(
            set(initial_joint_map) - set(robot.joint_names)
        )
        if unknown_joints:
            raise RuntimeError(
                f"requested joints are absent from USD: {unknown_joints}"
            )
        for name, value in initial_joint_map.items():
            requested_joint[0, robot.joint_names.index(name)] = float(value)
        root_pose_tensor = torch.tensor(
            [bound["root_pose_world"]],
            dtype=torch.float32,
            device=device,
        )
        object_pose_tensor = torch.tensor(
            [bound["object_pose_world"]],
            dtype=torch.float32,
            device=device,
        )
        zero_root_velocity = torch.zeros(
            (1, 6), dtype=torch.float32, device=device
        )
        zero_joint_velocity = torch.zeros_like(requested_joint)

        def restore_state(
            joint_position: Any,
            object_pose: Any,
        ) -> None:
            robot.write_root_pose_to_sim_index(root_pose=root_pose_tensor)
            robot.write_root_velocity_to_sim_index(
                root_velocity=zero_root_velocity
            )
            robot.write_joint_position_to_sim_index(
                position=joint_position
            )
            robot.write_joint_velocity_to_sim_index(
                velocity=zero_joint_velocity
            )
            robot.set_joint_position_target_index(target=joint_position)
            robot.set_joint_velocity_target_index(
                target=zero_joint_velocity
            )
            robot.set_joint_effort_target_index(
                target=zero_joint_velocity
            )
            object_asset.write_root_pose_to_sim_index(
                root_pose=object_pose
            )
            object_asset.write_root_velocity_to_sim_index(
                root_velocity=zero_root_velocity
            )
            sim.forward()
            robot.update(0.0)
            object_asset.update(0.0)

        body_paths = _body_paths_by_name(sim.stage, "/World/Robot")
        body_names = list(robot.body_names)
        if set(body_names) != set(body_paths):
            raise RuntimeError("Isaac body tensors and USD rigid bodies differ")

        restore_state(requested_joint, object_pose_tensor)
        spawn_evidence = _state_evidence(
            robot,
            object_asset,
            requested_root_pose=bound["root_pose_world"],
            requested_joint_by_name=initial_joint_map,
            requested_object_pose=bound["object_pose_world"],
        )
        observed_ik_baselink_pose = _infer_fixed_child_pose_world(
            requested_parent_pose_world=bound["root_pose_world"],
            requested_child_pose_world=bound["ik_base_pose_world"],
            observed_parent_pose_world=spawn_evidence[
                "observed_root_pose_world"
            ],
        )
        baselink_position_error = _position_error(
            observed_ik_baselink_pose,
            bound["ik_base_pose_world"],
        )
        baselink_attitude_error = _attitude_error(
            observed_ik_baselink_pose[3:7],
            bound["ik_base_pose_world"][3:7],
        )
        spawn_evidence["ik_baselink_pose"] = {
            "body_name": bound["ik_baselink_body_name"],
            "observation_method": (
                "observed_articulation_root_composed_with_hash_bound_"
                "generated_urdf_fixed_root_to_baselink_transform_v1"
            ),
            "direct_body_tensor_read": False,
            "requested_pose_world": list(bound["ik_base_pose_world"]),
            "observed_pose_world": observed_ik_baselink_pose,
            "position_error_m": baselink_position_error,
            "attitude_error_rad": baselink_attitude_error,
        }
        spawn_evidence["passed"] = bool(
            spawn_evidence["passed"]
            and baselink_position_error
            <= STATIC_STATE_POSITION_TOLERANCE_M
            and baselink_attitude_error
            <= STATIC_STATE_ATTITUDE_TOLERANCE_RAD
        )
        missing_selected_bodies = sorted(
            set(bound["selected_body_names"]) - set(body_paths)
        )
        if missing_selected_bodies:
            raise RuntimeError(
                f"selected bodies are absent from USD: {missing_selected_bodies}"
            )
        object_body_path = _rigid_body_path(sim.stage, "/World/Object")

        physics_view = sim.physics_manager.get_physics_sim_view()
        ordered_body_paths = [body_paths[name] for name in body_names]
        object_contact_capacity = (
            len(body_names) * int(args.max_contact_patches_per_body_pair)
        )
        object_contact_view = physics_view.create_rigid_contact_view(
            ordered_body_paths,
            filter_patterns=[
                [object_body_path] for _ in ordered_body_paths
            ],
            max_contact_data_count=object_contact_capacity,
        )
        if (
            int(object_contact_view.sensor_count) != len(body_names)
            or int(object_contact_view.filter_count) != 1
            or int(object_contact_view.max_contact_data_count)
            != object_contact_capacity
        ):
            raise RuntimeError("robot-object contact view layout mismatch")
        cross_module_views = _create_cross_module_contact_views(
            physics_view,
            body_paths_by_module=filter_evidence["body_paths_by_module"],
            max_patches_per_body_pair=int(
                args.max_contact_patches_per_body_pair
            ),
        )

        selected_contact_steps = {
            name: [] for name in bound["selected_body_names"]
        }
        prohibited_object_contact_steps = []
        self_contact_steps = []
        object_raw_contact_steps = []
        self_raw_contact_steps = []
        object_contact_saturated_steps = []
        self_contact_saturated_steps = []
        object_contact_nonfinite_steps = []
        self_contact_nonfinite_steps = []
        maximum_selected_force_n = {
            name: 0.0 for name in bound["selected_body_names"]
        }
        minimum_selected_separation_m = {
            name: math.inf for name in bound["selected_body_names"]
        }
        maximum_self_contact_force_n = 0.0
        minimum_self_contact_separation_m: float | None = None
        first_step_object_measurement = None
        first_step_self_measurement = None
        for step in range(int(args.settle_steps)):
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(float(args.dt))
            object_asset.update(float(args.dt))
            object_measurement = _measure_object_contacts(
                object_contact_view,
                body_names=body_names,
                dt=float(args.dt),
                torch_module=torch,
                warp_module=wp,
                force_threshold_n=float(
                    bound["physical_contact_force_threshold_n"]
                ),
                penetration_noise_floor_m=float(
                    bound[
                        "physical_contact_penetration_noise_floor_m"
                    ]
                ),
            )
            if step == 0:
                first_step_object_measurement = object_measurement
            active_by_body = object_measurement["active_by_body"]
            physical_by_body = object_measurement["physical_by_body"]
            if int(object_measurement["raw_contact_count"]) > 0:
                object_raw_contact_steps.append(
                    {
                        "step": step + 1,
                        "raw_contact_count": int(
                            object_measurement["raw_contact_count"]
                        ),
                        "physical_contact_count": int(
                            object_measurement["physical_contact_count"]
                        ),
                        "body_contact_evidence": dict(active_by_body),
                    }
                )
            prohibited = sorted(
                name
                for name in physical_by_body
                if name not in set(bound["selected_body_names"])
            )
            if prohibited:
                prohibited_object_contact_steps.append(
                    {
                        "step": step + 1,
                        "body_names": prohibited,
                        "body_contact_evidence": {
                            name: physical_by_body[name]
                            for name in prohibited
                        },
                    }
                )
            if object_measurement["raw_contact_saturated"]:
                object_contact_saturated_steps.append(step + 1)
            if object_measurement["raw_contact_nonfinite"]:
                object_contact_nonfinite_steps.append(step + 1)
            for name in bound["selected_body_names"]:
                contact = physical_by_body.get(name)
                if contact is None:
                    continue
                selected_contact_steps[name].append(step + 1)
                maximum_selected_force_n[name] = max(
                    maximum_selected_force_n[name],
                    float(contact["raw_contact_max_force_n"]),
                )
                minimum_selected_separation_m[name] = min(
                    minimum_selected_separation_m[name],
                    float(contact["raw_contact_min_separation_m"]),
                )

            self_measurement = _measure_cross_module_contacts(
                cross_module_views,
                dt=float(args.dt),
                torch_module=torch,
                warp_module=wp,
                force_threshold_n=float(
                    bound["physical_contact_force_threshold_n"]
                ),
                penetration_noise_floor_m=float(
                    bound[
                        "physical_contact_penetration_noise_floor_m"
                    ]
                ),
            )
            if step == 0:
                first_step_self_measurement = self_measurement
            maximum_self_contact_force_n = max(
                maximum_self_contact_force_n,
                float(self_measurement["raw_contact_max_force_n"]),
            )
            step_minimum_self = self_measurement[
                "raw_contact_min_separation_m"
            ]
            if step_minimum_self is not None:
                minimum_self_contact_separation_m = (
                    float(step_minimum_self)
                    if minimum_self_contact_separation_m is None
                    else min(
                        minimum_self_contact_separation_m,
                        float(step_minimum_self),
                    )
                )
            raw_self_evidence = {
                "step": step + 1,
                "raw_contact_count": int(
                    self_measurement["raw_contact_count"]
                ),
                "physical_contact_count": int(
                    self_measurement["physical_contact_count"]
                ),
                "physical_contact_pair_count": int(
                    self_measurement["physical_contact_pair_count"]
                ),
                "pair_raw_contact_counts": dict(
                    self_measurement["pair_raw_contact_counts"]
                ),
                "pair_physical_contact_counts": dict(
                    self_measurement["pair_physical_contact_counts"]
                ),
                "body_pair_raw_contact_counts": dict(
                    self_measurement["body_pair_raw_contact_counts"]
                ),
                "body_pair_contact_evidence": dict(
                    self_measurement["body_pair_contact_evidence"]
                ),
            }
            if int(self_measurement["raw_contact_count"]) > 0:
                self_raw_contact_steps.append(raw_self_evidence)
            if int(self_measurement["physical_contact_count"]) > 0:
                self_contact_steps.append(
                    raw_self_evidence
                )
            if bool(self_measurement["raw_contact_saturated"]):
                self_contact_saturated_steps.append(step + 1)
            if bool(self_measurement["raw_contact_nonfinite"]):
                self_contact_nonfinite_steps.append(step + 1)
            gpu_samples.append(_gpu_snapshot())

        settle_state = _state_evidence(
            robot,
            object_asset,
            requested_root_pose=bound["root_pose_world"],
            requested_joint_by_name=initial_joint_map,
            requested_object_pose=bound["object_pose_world"],
        )
        if (
            first_step_object_measurement is None
            or first_step_self_measurement is None
        ):
            raise RuntimeError("settle produced no first-step contact evidence")
        first_step_physical_by_body = first_step_object_measurement[
            "physical_by_body"
        ]
        initial_selected = {
            name
            for name in first_step_physical_by_body
            if name in set(bound["selected_body_names"])
        }
        initial_prohibited_object_bodies = sorted(
            name
            for name in first_step_physical_by_body
            if name not in set(bound["selected_body_names"])
        )
        initial_collision = {
            "method": (
                "isaac_physx_exact_restore_first_step_raw_contact_views_"
                "order8_physical_patch_classification_v2"
            ),
            "selected_robot_object_body_names": sorted(initial_selected),
            "prohibited_robot_object_body_names": (
                initial_prohibited_object_bodies
            ),
            "prohibited_self_raw_contact_count": int(
                first_step_self_measurement["raw_contact_count"]
            ),
            "prohibited_self_physical_contact_count": int(
                first_step_self_measurement["physical_contact_count"]
            ),
            "prohibited_self_physical_contact_pair_count": int(
                first_step_self_measurement[
                    "physical_contact_pair_count"
                ]
            ),
            "prohibited_self_pair_raw_contact_counts": dict(
                first_step_self_measurement["pair_raw_contact_counts"]
            ),
            "prohibited_self_pair_physical_contact_counts": dict(
                first_step_self_measurement[
                    "pair_physical_contact_counts"
                ]
            ),
            "prohibited_self_body_pair_raw_contact_counts": dict(
                first_step_self_measurement[
                    "body_pair_raw_contact_counts"
                ]
            ),
            "prohibited_self_body_pair_contact_evidence": dict(
                first_step_self_measurement[
                    "body_pair_contact_evidence"
                ]
            ),
            "robot_object_raw_contact_count": int(
                first_step_object_measurement["raw_contact_count"]
            ),
            "robot_object_physical_contact_count": int(
                first_step_object_measurement["physical_contact_count"]
            ),
            "robot_object_body_contact_evidence": dict(
                first_step_object_measurement["active_by_body"]
            ),
            "robot_object_raw_contact_saturated": bool(
                first_step_object_measurement["raw_contact_saturated"]
            ),
            "robot_object_raw_contact_nonfinite": bool(
                first_step_object_measurement["raw_contact_nonfinite"]
            ),
            "self_raw_contact_saturated": bool(
                first_step_self_measurement["raw_contact_saturated"]
            ),
            "self_raw_contact_nonfinite": bool(
                first_step_self_measurement["raw_contact_nonfinite"]
            ),
        }
        selected_contact_passed = all(
            name in initial_selected and bool(selected_contact_steps[name])
            for name in bound["selected_body_names"]
        )
        selected_penetration_passed = all(
            not math.isinf(minimum_selected_separation_m[name])
            and minimum_selected_separation_m[name]
            >= (
                -float(bound["selected_contact_penetration_limit_m"])
                - 1.0e-6
            )
            for name in bound["selected_body_names"]
        )
        initial_forbidden_passed = bool(
            not initial_prohibited_object_bodies
            and int(
                first_step_self_measurement["physical_contact_count"]
            )
            == 0
            and not first_step_object_measurement["raw_contact_saturated"]
            and not first_step_self_measurement["raw_contact_saturated"]
            and not first_step_object_measurement["raw_contact_nonfinite"]
            and not first_step_self_measurement["raw_contact_nonfinite"]
        )
        settle_forbidden_passed = bool(
            not prohibited_object_contact_steps
            and not self_contact_steps
            and not object_contact_saturated_steps
            and not self_contact_saturated_steps
            and not object_contact_nonfinite_steps
            and not self_contact_nonfinite_steps
        )
        passed = bool(
            spawn_evidence["passed"]
            and initial_forbidden_passed
            and settle_forbidden_passed
            and selected_contact_passed
            and selected_penetration_passed
        )
        failure_reasons = []
        if not spawn_evidence["passed"]:
            failure_reasons.append("exact_static_spawn_mismatch")
        if int(first_step_self_measurement["physical_contact_count"]) > 0:
            failure_reasons.append("initial_prohibited_self_contact")
        if initial_prohibited_object_bodies:
            failure_reasons.append(
                "initial_prohibited_robot_object_contact"
            )
        if prohibited_object_contact_steps:
            failure_reasons.append("settle_prohibited_robot_object_contact")
        if self_contact_steps:
            failure_reasons.append("settle_prohibited_self_contact")
        if object_contact_saturated_steps or self_contact_saturated_steps:
            failure_reasons.append("contact_buffer_saturated")
        if object_contact_nonfinite_steps or self_contact_nonfinite_steps:
            failure_reasons.append("contact_data_nonfinite")
        if not selected_contact_passed:
            failure_reasons.append("selected_contact_not_physically_observed")
        if not selected_penetration_passed:
            failure_reasons.append(
                "selected_contact_penetration_limit_exceeded"
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        finished_usage = resource.getrusage(resource.RUSAGE_SELF)
        wall_s = time.perf_counter() - started_wall
        cpu_s = (
            finished_usage.ru_utime
            - started_usage.ru_utime
            + finished_usage.ru_stime
            - started_usage.ru_stime
        )
        gpu_samples.append(_gpu_snapshot())
        runner_sha256_at_end = _hash_file(runner_path)
        if runner_sha256_at_end != runner_sha256_at_start:
            raise RuntimeError(
                "admission runner changed during real-Isaac execution"
            )
        report = {
            "report_version": RUNNER_VERSION,
            "status": "pass" if passed else "fail",
            "passed": passed,
            "passed_field_semantics": (
                "static_pose_collision_contact_pass"
            ),
            "failure_reasons": failure_reasons,
            "case_id": bound["case_id"],
            "production_isaac_admission_eligible": False,
            "production_eligibility_limitations": [
                "trajectory_path_not_run_missing_bound_path",
                *(
                    []
                    if all(
                        force
                        <= float(bound["order8_max_force_per_contact_n"])
                        for force in maximum_selected_force_n.values()
                    )
                    else ["order8_selected_contact_force_limit_exceeded"]
                ),
            ],
            "method": {
                "static_spawn": (
                    "isaaclab_articulation_exact_root_joint_object_restore_v1"
                ),
                "initial_collision": (
                    "isaac_physx_exact_restore_first_step_raw_contact_views_"
                    "order8_physical_patch_classification_v2"
                ),
                "self_collision_filter": (
                    PRODUCTION_COLLISION_FILTER_SEMANTICS
                ),
                "settle_contacts": (
                    "isaac_physx_raw_rigid_contact_views_order8_"
                    "physical_patch_classification_v2"
                ),
                "dt_s": float(args.dt),
                "settle_steps": int(args.settle_steps),
                "trajectory_path_status": bound["trajectory_path_status"],
                "trajectory_claimed": False,
                "final_state_drift_gate_role": (
                    "diagnostic_only_not_an_admission_gate"
                ),
            },
            "source_binding": {
                "runner_path": _portable(runner_path),
                "runner_sha256": runner_sha256_at_start,
                "runner_sha256_at_start": runner_sha256_at_start,
                "runner_sha256_at_end": runner_sha256_at_end,
                "runner_sha256_stable_during_execution": True,
                "manifest_path": _portable(bound["manifest_path"]),
                "manifest_sha256": bound["manifest_sha256"],
                "review_decisions_path": _portable(bound["review_path"]),
                "review_decisions_sha256": bound["review_sha256"],
                "accepted_scene_path": _portable(bound["scene_path"]),
                "accepted_scene_sha256": bound["scene_sha256"],
                "morphology_graph_path": _portable(bound["graph_path"]),
                "morphology_graph_sha256": bound["graph_sha256"],
                "structural_hash": bound["structural_hash"],
                "urdf_path": _portable(bound["urdf_path"]),
                "urdf_sha256": bound["urdf_sha256"],
                "usd_path": _portable(bound["usd_path"]),
                "usd_root_sha256": bound["usd_root_sha256"],
                "usd_tree_sha256": bound["usd_tree_sha256"],
                "usd_file_count": len(bound["usd_files"]),
                "joint_solution_hash": bound["joint_solution_hash"],
                "root_pose_conversion": bound[
                    "root_pose_conversion"
                ],
                "physical_contact_config_path": _portable(
                    bound["physical_contact_config_path"]
                ),
                "physical_contact_config_sha256": bound[
                    "physical_contact_config_sha256"
                ],
                "order8_max_force_per_contact_n": float(
                    bound["order8_max_force_per_contact_n"]
                ),
            },
            "scene": {
                "module_count": bound["module_count"],
                "object_id": bound["object_id"],
                "object_size_m": list(bound["object_size_m"]),
                "selected_body_names": list(bound["selected_body_names"]),
                "spawned_body_count": len(robot.body_names),
                "spawned_joint_count": len(robot.joint_names),
                "expected_movable_joint_count": len(
                    bound["expected_movable_joint_names"]
                ),
                "expected_dock_joint_count": len(
                    bound["expected_dock_joint_names"]
                ),
                "movable_joint_identity_exact_match": True,
            },
            "exact_static_spawn": spawn_evidence,
            "collision_filter": {
                key: value
                for key, value in filter_evidence.items()
                if key != "body_paths_by_module"
            },
            "contact_report_body_count": contact_report_body_count,
            "physical_contact_classification": {
                "semantics": (
                    "finite_unsaturated_raw_patch_with_abs_solver_force_"
                    "at_or_above_threshold_or_negative_separation_beyond_"
                    "penetration_noise_floor_v1"
                ),
                "threshold_source": "order8_natural_contact_config",
                "force_threshold_n": float(
                    bound["physical_contact_force_threshold_n"]
                ),
                "penetration_noise_floor_m": float(
                    bound[
                        "physical_contact_penetration_noise_floor_m"
                    ]
                ),
                "raw_speculative_patches_are_physical_contact": False,
                "order8_force_limit_role": (
                    "diagnostic_limitation_not_static_pose_admission_gate"
                ),
            },
            "initial_collision": initial_collision,
            "settle": {
                "selected_contact_steps": selected_contact_steps,
                "maximum_selected_force_n": maximum_selected_force_n,
                "order8_selected_contact_force_limit_n": float(
                    bound["order8_max_force_per_contact_n"]
                ),
                "order8_selected_contact_force_limit_passed": all(
                    force
                    <= float(bound["order8_max_force_per_contact_n"])
                    for force in maximum_selected_force_n.values()
                ),
                "minimum_selected_separation_m": {
                    name: (
                        None
                        if math.isinf(value)
                        else float(value)
                    )
                    for name, value in minimum_selected_separation_m.items()
                },
                "selected_contact_passed": selected_contact_passed,
                "selected_contact_penetration_limit_m": float(
                    bound["selected_contact_penetration_limit_m"]
                ),
                "selected_contact_penetration_passed": (
                    selected_penetration_passed
                ),
                "prohibited_robot_object_contact_steps": (
                    prohibited_object_contact_steps
                ),
                "prohibited_self_contact_steps": self_contact_steps,
                "robot_object_raw_contact_steps": (
                    object_raw_contact_steps
                ),
                "self_raw_contact_steps": self_raw_contact_steps,
                "object_contact_saturated_steps": (
                    object_contact_saturated_steps
                ),
                "self_contact_saturated_steps": self_contact_saturated_steps,
                "object_contact_nonfinite_steps": (
                    object_contact_nonfinite_steps
                ),
                "self_contact_nonfinite_steps": (
                    self_contact_nonfinite_steps
                ),
                "maximum_prohibited_self_contact_force_n": (
                    maximum_self_contact_force_n
                ),
                "minimum_prohibited_self_contact_separation_m": (
                    minimum_self_contact_separation_m
                ),
                "final_state_drift": settle_state,
                "final_state_drift_gate_role": (
                    "diagnostic_only_not_an_admission_gate; exact spawn is "
                    "gated before stepping and contact gates use raw PhysX "
                    "evidence during the configured minimum hold"
                ),
            },
            "resource_usage": {
                "wall_time_s": wall_s,
                "process_cpu_time_s": cpu_s,
                "process_cpu_percent_of_one_core": (
                    100.0 * cpu_s / wall_s if wall_s > 0.0 else None
                ),
                "maximum_resident_set_kib": int(
                    finished_usage.ru_maxrss
                ),
                "torch_cuda_max_memory_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated(device))
                    if torch.cuda.is_available()
                    else 0
                ),
                "torch_cuda_max_memory_reserved_bytes": (
                    int(torch.cuda.max_memory_reserved(device))
                    if torch.cuda.is_available()
                    else 0
                ),
                "gpu_device_global_samples": gpu_samples,
                "gpu_sample_scope": (
                    "device-global nvidia-smi snapshots; torch memory is "
                    "process-specific"
                ),
            },
        }
        _write_report(output_path, report)
        print(
            "ORDER9_C3A_ISAAC_ADMISSION="
            + json.dumps(
                {
                    "case_id": bound["case_id"],
                    "status": report["status"],
                    "report": _portable(output_path),
                    "report_sha256": _hash_file(output_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        sim.stop()
        sim.clear_instance()
        return report
    except BaseException as error:
        print(
            "ORDER9_C3A_ISAAC_INTERNAL_ERROR="
            + json.dumps(
                {
                    "case_id": bound["case_id"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        raise
    finally:
        simulation_app.close()


def main() -> int:
    preliminary_args, _ = _parser().parse_known_args()
    if preliminary_args.dt <= 0.0 or not math.isfinite(preliminary_args.dt):
        raise ValueError("--dt must be finite and positive")
    if preliminary_args.settle_steps < 1:
        raise ValueError("--settle-steps must be positive")
    if preliminary_args.max_contact_patches_per_body_pair < 1:
        raise ValueError(
            "--max-contact-patches-per-body-pair must be positive"
        )
    output_path = (
        _resolve(preliminary_args.output_root)
        / "cases"
        / preliminary_args.case_id
        / "admission.json"
    )
    bound = _load_bound_case(
        manifest_path=_resolve(preliminary_args.manifest),
        review_path=_resolve(preliminary_args.review_decisions),
        case_id=preliminary_args.case_id,
        robot_model_config_path=_resolve(
            preliminary_args.robot_model_config
        ),
    )
    try:
        report = _run_isaac(
            preliminary_args,
            bound,
            output_path=output_path,
        )
    except BaseException as error:
        report = {
            "report_version": RUNNER_VERSION,
            "status": "fail",
            "passed": False,
            "failure_reasons": ["isaac_execution_error"],
            "failure": {
                "error_type": type(error).__name__,
                "error": str(error),
            },
            "case_id": preliminary_args.case_id,
            "source_binding": {
                "runner_path": _portable(Path(__file__)),
                "runner_sha256": _hash_file(Path(__file__).resolve()),
                "manifest_path": _portable(bound["manifest_path"]),
                "manifest_sha256": bound["manifest_sha256"],
                "review_decisions_path": _portable(bound["review_path"]),
                "review_decisions_sha256": bound["review_sha256"],
                "accepted_scene_path": _portable(bound["scene_path"]),
                "accepted_scene_sha256": bound["scene_sha256"],
                "morphology_graph_path": _portable(bound["graph_path"]),
                "morphology_graph_sha256": bound["graph_sha256"],
                "structural_hash": bound["structural_hash"],
                "urdf_path": _portable(bound["urdf_path"]),
                "urdf_sha256": bound["urdf_sha256"],
                "usd_path": _portable(bound["usd_path"]),
                "usd_root_sha256": bound["usd_root_sha256"],
                "usd_tree_sha256": bound["usd_tree_sha256"],
                "joint_solution_hash": bound["joint_solution_hash"],
            },
            "method": {
                "trajectory_path_status": (
                    bound["trajectory_path_status"]
                ),
                "trajectory_claimed": False,
            },
        }
        _write_report(output_path, report)
        _write_batch_summary(
            output_root=_resolve(preliminary_args.output_root),
            manifest_path=_resolve(preliminary_args.manifest),
            review_path=_resolve(preliminary_args.review_decisions),
        )
        print(
            "ORDER9_C3A_ISAAC_ADMISSION="
            + json.dumps(
                {
                    "case_id": preliminary_args.case_id,
                    "status": "fail",
                    "report": _portable(output_path),
                    "report_sha256": _hash_file(output_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        raise
    _write_report(output_path, report)
    summary_path = _write_batch_summary(
        output_root=_resolve(preliminary_args.output_root),
        manifest_path=_resolve(preliminary_args.manifest),
        review_path=_resolve(preliminary_args.review_decisions),
    )
    print(
        "ORDER9_C3A_ISAAC_ADMISSION="
        + json.dumps(
            {
                "case_id": preliminary_args.case_id,
                "status": report["status"],
                "report": _portable(output_path),
                "report_sha256": _hash_file(output_path),
                "batch_summary": _portable(summary_path),
                "batch_summary_sha256": _hash_file(summary_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if bool(report["passed"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
