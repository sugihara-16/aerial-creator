from __future__ import annotations

"""R1 nominal payload height and single-source support contract.

The historical C3/R1 payload remains available as a difficult condition.  A
new nominal R1 view raises the payload centre and its side grasp points while
keeping the payload bottom, mass, planar pose, and task displacement fixed.
The support box is copied from the hash-bound Order-8 report so teacher
generation, lightweight admission, and Isaac consume one geometry.
"""

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.task_spec import GeometryType, TaskSpec
from amsrr.simulation.order9_object_task_state import load_order9_canonical_reset
from amsrr.training.order9_posture_resolver import (
    Order9PostureCollisionBox,
    Order9PostureCollisionObject,
)
from amsrr.training.order9_r1_calibration_runner import Order9R1CalibrationCase
from amsrr.utils.hashing import hash_file, stable_hash

ORDER9_R1_NOMINAL_GEOMETRY_V11_VERSION = "order9_r1_nominal_geometry_v11"
ORDER9_R1_NOMINAL_GEOMETRY_V11_RELATIVE = Path(
    "configs/training/order9_r1_nominal_geometry_v11.json"
)


@dataclass(frozen=True)
class Order9R1NominalGeometryV11Contract:
    config_path: Path
    config_sha256: str
    canonical_report_path: Path
    canonical_report_sha256: str
    support_size_m: tuple[float, float, float]
    support_pose_world: tuple[float, float, float, float, float, float, float]
    object_height_increment_m: float
    grasp_contact_height_offset_m: float
    required_robot_support_clearance_m: float
    other_collision_clearance_m: float


@dataclass(frozen=True)
class Order9R1GeometryVariantsV11:
    nominal_case: Order9R1CalibrationCase
    small_object_task_spec: TaskSpec
    contract: Order9R1NominalGeometryV11Contract


def load_order9_r1_nominal_geometry_v11_contract(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> Order9R1NominalGeometryV11Contract:
    repository = Path(repository_root).resolve()
    source = Path(path)
    if not source.is_absolute():
        source = repository / source
    source = source.resolve()
    if repository not in source.parents or not source.is_file():
        raise SchemaValidationError("R1 v11 nominal-geometry config is invalid")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if (
        set(payload)
        != {
            "config_version",
            "canonical_order8_report",
            "nominal_object_height_increment_m",
            "nominal_grasp_contact_height_offset_m",
            "required_robot_support_clearance_m",
            "other_collision_clearance_m",
            "small_object_variant",
        }
        or payload.get("config_version") != ORDER9_R1_NOMINAL_GEOMETRY_V11_VERSION
    ):
        raise SchemaValidationError("R1 v11 nominal-geometry fields changed")
    binding = payload.get("canonical_order8_report")
    small = payload.get("small_object_variant")
    if (
        not isinstance(binding, dict)
        or set(binding) != {"path", "sha256"}
        or not isinstance(small, dict)
        or small
        != {
            "role": "difficult_condition_only",
            "formal_range_selection_eligible": False,
            "formal_teacher_collection_eligible": False,
        }
    ):
        raise SchemaValidationError("R1 v11 geometry provenance changed")
    report = (repository / str(binding["path"])).resolve()
    expected = str(binding["sha256"])
    if repository not in report.parents or hash_file(report) != expected:
        raise SchemaValidationError("R1 v11 canonical support bytes changed")
    reset = load_order9_canonical_reset(report, expected_sha256=expected)
    values = tuple(
        float(payload[name])
        for name in (
            "nominal_object_height_increment_m",
            "nominal_grasp_contact_height_offset_m",
            "required_robot_support_clearance_m",
            "other_collision_clearance_m",
        )
    )
    if (
        any(not math.isfinite(value) for value in values)
        or any(value <= 0.0 for value in values[:3])
        or values[3] < 0.0
        or values[0] < 0.04 - 1.0e-12
        or values[1] < 0.02 - 1.0e-12
        or values[2] < 0.019 - 1.0e-12
        or values[3] > values[2]
    ):
        raise SchemaValidationError("R1 v11 clearance values changed")
    return Order9R1NominalGeometryV11Contract(
        config_path=source,
        config_sha256=hash_file(source),
        canonical_report_path=report,
        canonical_report_sha256=expected,
        support_size_m=tuple(
            float(value) for value in reset.metadata["object_support_size_m"]
        ),
        support_pose_world=tuple(
            float(value) for value in reset.metadata["object_support_pose_world"]
        ),
        object_height_increment_m=values[0],
        grasp_contact_height_offset_m=values[1],
        required_robot_support_clearance_m=values[2],
        other_collision_clearance_m=values[3],
    )


def build_order9_r1_geometry_variants_v11(
    case: Order9R1CalibrationCase,
    *,
    contract: Order9R1NominalGeometryV11Contract,
) -> Order9R1GeometryVariantsV11:
    """Return a corrected nominal case and the untouched-height difficult case."""

    small = _rewrite_support(
        case.task_spec,
        contract=contract,
        geometry_role="small_object_difficult_condition",
    )
    nominal = _raise_object_geometry(small, contract=contract)
    object_spec, size = _target_object_and_size(nominal)
    estimated_mass = float(case.source_bucket.estimated_mass_kg)
    estimated_inertia = _cuboid_inertia(estimated_mass, size)
    source_bucket = replace(
        case.source_bucket,
        estimated_inertia_body=list(estimated_inertia),
        metadata={
            **case.source_bucket.metadata,
            "r1_nominal_geometry_v11": True,
            "r1_nominal_geometry_v11_config_sha256": contract.config_sha256,
            "r1_estimated_inertia_recomputed_for_nominal_height": True,
        },
    )
    source_bucket.validate()
    payload = nominal.to_dict()
    metadata = dict(payload.get("metadata", {}) or {})
    metadata.update(
        {
            "estimated_inertia_body": list(estimated_inertia),
            "estimated_mass_kg": estimated_mass,
            "r1_nominal_object_mass_kg": float(object_spec.mass_kg),
        }
    )
    payload["metadata"] = metadata
    nominal = TaskSpec.from_dict(payload)
    nominal.validate()
    nominal_case = replace(case, task_spec=nominal, source_bucket=source_bucket)
    return Order9R1GeometryVariantsV11(
        nominal_case=nominal_case,
        small_object_task_spec=small,
        contract=contract,
    )


def build_order9_r1_explicit_support_collision_object_v11(
    task_spec: TaskSpec,
    *,
    contract: Order9R1NominalGeometryV11Contract,
) -> Order9PostureCollisionObject:
    """Build collision geometry only from the TaskSpec after identity audit."""

    require_order9_r1_support_identity_v11(task_spec, contract=contract)
    object_spec, size = _target_object_and_size(task_spec)
    support = task_spec.scene.environment.support_surfaces[0]
    support_geometry = next(
        value
        for value in task_spec.scene.geometry_library
        if value.geometry_id == support.geometry_id
    )
    support_size = tuple(
        float(value) for value in support_geometry.primitive_params["size_m"]
    )
    return Order9PostureCollisionObject(
        object_id=object_spec.object_id,
        size_m=size,
        initial_pose_world=tuple(float(value) for value in object_spec.pose_world),
        environment_boxes=(
            Order9PostureCollisionBox(
                box_id="canonical_order8_isaac_support",
                size_m=support_size,
                pose_world=tuple(float(value) for value in support.pose_world),
            ),
        ),
        ground_plane_z_m=0.0,
    )


def require_order9_r1_support_identity_v11(
    task_spec: TaskSpec,
    *,
    contract: Order9R1NominalGeometryV11Contract,
) -> None:
    surfaces = task_spec.scene.environment.support_surfaces
    if len(surfaces) != 1:
        raise SchemaValidationError("R1 v11 requires one canonical support")
    surface = surfaces[0]
    matches = [
        value
        for value in task_spec.scene.geometry_library
        if value.geometry_id == surface.geometry_id
    ]
    if len(matches) != 1:
        raise SchemaValidationError("R1 v11 support geometry identity differs")
    geometry = matches[0]
    params = geometry.primitive_params or {}
    raw_size = params.get("size_m")
    if (
        geometry.geometry_type != GeometryType.BOX
        or not isinstance(raw_size, list)
        or len(raw_size) != 3
        or any(
            abs(float(raw_size[index]) - contract.support_size_m[index]) > 1.0e-12
            for index in range(3)
        )
        or any(
            abs(float(surface.pose_world[index]) - contract.support_pose_world[index])
            > 1.0e-12
            for index in range(7)
        )
    ):
        raise SchemaValidationError(
            "R1 v11 TaskSpec support differs from the Isaac canonical support"
        )
    metadata = task_spec.metadata
    if (
        metadata.get("r1_support_single_source_version")
        != ORDER9_R1_NOMINAL_GEOMETRY_V11_VERSION
        or metadata.get("r1_support_source_report_sha256")
        != contract.canonical_report_sha256
        or metadata.get("r1_support_geometry_hash")
        != stable_hash(
            {
                "size_m": list(contract.support_size_m),
                "pose_world": list(contract.support_pose_world),
            }
        )
    ):
        raise SchemaValidationError("R1 v11 support provenance is incomplete")


def _rewrite_support(
    task_spec: TaskSpec,
    *,
    contract: Order9R1NominalGeometryV11Contract,
    geometry_role: str,
) -> TaskSpec:
    payload = task_spec.to_dict()
    surfaces = payload["scene"]["environment"]["support_surfaces"]
    if len(surfaces) != 1:
        raise SchemaValidationError("R1 v11 requires one support surface")
    surface = surfaces[0]
    geometries = [
        value
        for value in payload["scene"]["geometry_library"]
        if value["geometry_id"] == surface["geometry_id"]
    ]
    if len(geometries) != 1:
        raise SchemaValidationError("R1 v11 support geometry is missing")
    geometry = geometries[0]
    if geometry.get("geometry_type") != GeometryType.BOX.value:
        raise SchemaValidationError("R1 v11 support must be a box")
    geometry["primitive_params"] = {"size_m": list(contract.support_size_m)}
    surface["pose_world"] = list(contract.support_pose_world)
    metadata = dict(payload.get("metadata", {}) or {})
    metadata.update(
        {
            "r1_support_single_source_version": (
                ORDER9_R1_NOMINAL_GEOMETRY_V11_VERSION
            ),
            "r1_support_source_report_path": str(contract.canonical_report_path),
            "r1_support_source_report_sha256": contract.canonical_report_sha256,
            "r1_support_geometry_hash": stable_hash(
                {
                    "size_m": list(contract.support_size_m),
                    "pose_world": list(contract.support_pose_world),
                }
            ),
            "r1_geometry_role": geometry_role,
            "r1_formal_range_selection_eligible": False,
            "r1_formal_teacher_collection_eligible": False,
        }
    )
    payload["metadata"] = metadata
    result = TaskSpec.from_dict(payload)
    require_order9_r1_support_identity_v11(result, contract=contract)
    return result


def _raise_object_geometry(
    task_spec: TaskSpec,
    *,
    contract: Order9R1NominalGeometryV11Contract,
) -> TaskSpec:
    payload = task_spec.to_dict()
    object_data, geometry_data = _target_object_and_geometry_data(payload)
    raw_size = geometry_data.get("primitive_params", {}).get("size_m")
    if (
        geometry_data.get("geometry_type") != GeometryType.BOX.value
        or not isinstance(raw_size, list)
        or len(raw_size) != 3
    ):
        raise SchemaValidationError("R1 v11 target object must be a box")
    original_size = tuple(float(value) for value in raw_size)
    new_size = (
        original_size[0],
        original_size[1],
        original_size[2] + contract.object_height_increment_m,
    )
    centre_raise = 0.5 * contract.object_height_increment_m
    geometry_data["primitive_params"]["size_m"] = list(new_size)
    object_data["pose_world"][2] = float(object_data["pose_world"][2]) + centre_raise
    mass = float(object_data["mass_kg"])
    object_data["density_kg_m3"] = mass / math.prod(new_size)
    object_data["inertia_kgm2"] = list(_cuboid_inertia(mass, new_size))
    for goal in payload["goals"]:
        if goal.get("target_entity_id") == object_data["object_id"] and isinstance(
            goal.get("target_pose_world"), list
        ):
            goal["target_pose_world"][2] = (
                float(goal["target_pose_world"][2]) + centre_raise
            )
    payload["task_id"] = str(payload["task_id"]) + "_nominal_geometry_v11"
    metadata = dict(payload.get("metadata", {}) or {})
    metadata.update(
        {
            "r1_geometry_role": "nominal_object_height_corrected",
            "r1_nominal_geometry_version": ORDER9_R1_NOMINAL_GEOMETRY_V11_VERSION,
            "r1_nominal_geometry_config_path": str(contract.config_path),
            "r1_nominal_geometry_config_sha256": contract.config_sha256,
            "r1_original_object_size_m": list(original_size),
            "r1_nominal_object_size_m": list(new_size),
            "r1_nominal_object_height_increment_m": (
                contract.object_height_increment_m
            ),
            "r1_object_centre_raise_m": centre_raise,
            "r1_object_bottom_height_preserved": True,
            "r1_object_mass_preserved": True,
            "r1_nominal_grasp_contact_height_offset_m": (
                contract.grasp_contact_height_offset_m
            ),
            "r1_required_robot_support_clearance_m": (
                contract.required_robot_support_clearance_m
            ),
            "r1_formal_range_selection_eligible": True,
            "r1_formal_teacher_collection_eligible": False,
            "r1_small_object_parent_task_hash": task_spec.stable_hash(),
        }
    )
    payload["metadata"] = metadata
    result = TaskSpec.from_dict(payload)
    result.validate()
    require_order9_r1_support_identity_v11(result, contract=contract)
    return result


def _target_object_and_geometry_data(payload: dict[str, Any]):
    target_ids = {
        value.get("target_entity_id")
        for value in payload["goals"]
        if value.get("goal_type") == "object_pose"
    }
    objects = [
        value
        for value in payload["scene"]["objects"]
        if value.get("object_id") in target_ids and value.get("movable") is True
    ]
    if len(objects) != 1:
        raise SchemaValidationError("R1 v11 requires one movable target object")
    object_data = objects[0]
    geometries = [
        value
        for value in payload["scene"]["geometry_library"]
        if value.get("geometry_id") == object_data.get("geometry_id")
    ]
    if len(geometries) != 1:
        raise SchemaValidationError("R1 v11 target geometry identity differs")
    return object_data, geometries[0]


def _target_object_and_size(task_spec: TaskSpec):
    payload = task_spec.to_dict()
    object_data, geometry_data = _target_object_and_geometry_data(payload)
    raw_size = geometry_data.get("primitive_params", {}).get("size_m")
    if not isinstance(raw_size, list) or len(raw_size) != 3:
        raise SchemaValidationError("R1 v11 target box size is invalid")
    object_spec = next(
        value
        for value in task_spec.scene.objects
        if value.object_id == object_data["object_id"]
    )
    return object_spec, tuple(float(value) for value in raw_size)


def _cuboid_inertia(
    mass_kg: float, size_m: tuple[float, float, float]
) -> tuple[float, float, float, float, float, float]:
    x, y, z = (float(value) for value in size_m)
    factor = float(mass_kg) / 12.0
    return (
        factor * (y * y + z * z),
        0.0,
        0.0,
        factor * (x * x + z * z),
        0.0,
        factor * (x * x + y * y),
    )


__all__ = [
    "ORDER9_R1_NOMINAL_GEOMETRY_V11_RELATIVE",
    "ORDER9_R1_NOMINAL_GEOMETRY_V11_VERSION",
    "Order9R1GeometryVariantsV11",
    "Order9R1NominalGeometryV11Contract",
    "build_order9_r1_explicit_support_collision_object_v11",
    "build_order9_r1_geometry_variants_v11",
    "load_order9_r1_nominal_geometry_v11_contract",
    "require_order9_r1_support_identity_v11",
]
