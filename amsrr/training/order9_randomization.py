from __future__ import annotations

"""Conservative, physics-consistent Order 9 randomization around Order 8."""

import math
import random
from dataclasses import dataclass
from typing import Any

from amsrr.geometry.mass_properties import (
    RigidBodyMassProperties,
    cuboid_mass_properties,
    mass_properties_from_geometry,
)
from amsrr.geometry.contact_material import with_selected_robot_contact_material
from amsrr.schemas.common import ContactMode, SchemaBase, SchemaValidationError, require_len
from amsrr.schemas.task_spec import CollisionModel, GeometrySpec, GeometryType, TaskSpec


ORDER9_RANDOMIZATION_VERSION = "order9_conservative_randomization_v2_material_combine"
ORDER9_EXPANDED_RANDOMIZATION_VERSION = (
    "order9_expanded_shape_inertia_randomization_v2_material_combine"
)


@dataclass
class Order9ConservativeRandomizationConfig(SchemaBase):
    dimension_scale: tuple[float, float] = (0.975, 1.025)
    mass_scale: tuple[float, float] = (0.95, 1.05)
    initial_xy_offset_m: tuple[float, float] = (-0.005, 0.005)
    support_gap_m: tuple[float, float] = (0.0, 0.005)
    initial_yaw_rad: tuple[float, float] = (
        -2.0 * math.pi / 180.0,
        2.0 * math.pi / 180.0,
    )
    selected_gripper_friction: tuple[float, float] = (4.275, 4.725)
    contact_stiffness_scale: tuple[float, float] = (0.95, 1.05)
    estimated_mass_scale: tuple[float, float] = (0.98, 1.02)
    estimated_com_error_m: tuple[float, float] = (-0.001, 0.001)
    nominal_selected_gripper_friction: float = 4.5
    nominal_contact_stiffness_n_per_m: float = 7500.0
    nominal_contact_damping_n_s_per_m: float = 75.0
    support_top_z_m: float = 0.15

    def validate(self) -> None:
        positive_ranges = (
            "dimension_scale",
            "mass_scale",
            "selected_gripper_friction",
            "contact_stiffness_scale",
            "estimated_mass_scale",
        )
        signed_ranges = (
            "initial_xy_offset_m",
            "support_gap_m",
            "initial_yaw_rad",
            "estimated_com_error_m",
        )
        for name in positive_ranges:
            _validate_range(getattr(self, name), name, positive=True)
        for name in signed_ranges:
            _validate_range(getattr(self, name), name, positive=False)
        if self.support_gap_m[0] < 0.0:
            raise SchemaValidationError(
                "Order9 support_gap_m must not place the object below its support"
            )
        for name in (
            "nominal_selected_gripper_friction",
            "nominal_contact_stiffness_n_per_m",
            "nominal_contact_damping_n_s_per_m",
            "support_top_z_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise SchemaValidationError(
                    f"Order9ConservativeRandomizationConfig.{name} must be finite and positive"
                )


@dataclass
class Order9RandomizationSample(SchemaBase):
    seed: int
    sample_index: int
    task_spec: TaskSpec
    true_mass_properties: RigidBodyMassProperties
    estimated_mass_properties: RigidBodyMassProperties
    selected_gripper_friction: float
    contact_stiffness_n_per_m: float
    contact_damping_n_s_per_m: float
    sampled_values: dict[str, float]
    randomization_version: str = ORDER9_RANDOMIZATION_VERSION

    def validate(self) -> None:
        if self.seed < 0 or self.sample_index < 0:
            raise SchemaValidationError(
                "Order9RandomizationSample seed and sample_index must be non-negative"
            )
        if not all(math.isfinite(float(value)) for value in self.sampled_values.values()):
            raise SchemaValidationError(
                "Order9RandomizationSample.sampled_values must be finite"
            )


class Order9ConservativeRandomizer:
    def __init__(
        self,
        config: Order9ConservativeRandomizationConfig | None = None,
    ) -> None:
        self.config = config or Order9ConservativeRandomizationConfig()

    def sample(
        self,
        base_task_spec: TaskSpec,
        *,
        seed: int,
        sample_index: int = 0,
    ) -> Order9RandomizationSample:
        if seed < 0 or sample_index < 0:
            raise ValueError("Order 9 randomization seed and sample_index must be non-negative")
        rng = random.Random(_derived_seed(seed, sample_index))
        values = {
            "dimension_scale_x": _uniform(self.config.dimension_scale, rng),
            "dimension_scale_y": _uniform(self.config.dimension_scale, rng),
            "dimension_scale_z": _uniform(self.config.dimension_scale, rng),
            "mass_scale": _uniform(self.config.mass_scale, rng),
            "initial_offset_x_m": _uniform(self.config.initial_xy_offset_m, rng),
            "initial_offset_y_m": _uniform(self.config.initial_xy_offset_m, rng),
            "support_gap_m": _uniform(self.config.support_gap_m, rng),
            "initial_yaw_rad": _uniform(self.config.initial_yaw_rad, rng),
            "selected_gripper_friction": _uniform(
                self.config.selected_gripper_friction,
                rng,
            ),
            "contact_stiffness_scale": _uniform(
                self.config.contact_stiffness_scale,
                rng,
            ),
            "estimated_mass_scale": _uniform(
                self.config.estimated_mass_scale,
                rng,
            ),
            "estimated_com_error_x_m": _uniform(
                self.config.estimated_com_error_m,
                rng,
            ),
            "estimated_com_error_y_m": _uniform(
                self.config.estimated_com_error_m,
                rng,
            ),
            "estimated_com_error_z_m": _uniform(
                self.config.estimated_com_error_m,
                rng,
            ),
        }
        return self._apply_values(
            base_task_spec,
            seed=seed,
            sample_index=sample_index,
            values=values,
        )

    def boundary_preflight_samples(
        self,
        base_task_spec: TaskSpec,
    ) -> list[Order9RandomizationSample]:
        """Return nominal plus paired worst-case corners for fail-fast Isaac checks."""

        low_high = (
            (0, "small_light_low_contact"),
            (1, "large_heavy_high_contact"),
            (2, "small_heavy_low_friction"),
            (3, "large_light_high_friction"),
        )
        output = [self._nominal_sample(base_task_spec)]
        for index, label in low_high:
            even = index % 2 == 0
            cross = index >= 2
            values = self._corner_values(
                dimensions_high=not even,
                mass_high=(not even) ^ cross,
                contact_high=not even,
                positive_pose=not even,
            )
            values["preflight_corner_id"] = float(index)
            sample = self._apply_values(
                base_task_spec,
                seed=index + 1,
                sample_index=index + 1,
                values=values,
            )
            metadata = sample.task_spec.metadata
            metadata["preflight_corner_label"] = label
            output.append(sample)
        return output

    def _nominal_sample(self, base_task_spec: TaskSpec) -> Order9RandomizationSample:
        values = self._corner_values(
            dimensions_high=None,
            mass_high=None,
            contact_high=None,
            positive_pose=None,
        )
        values["preflight_corner_id"] = -1.0
        return self._apply_values(
            base_task_spec,
            seed=0,
            sample_index=0,
            values=values,
        )

    def _corner_values(
        self,
        *,
        dimensions_high: bool | None,
        mass_high: bool | None,
        contact_high: bool | None,
        positive_pose: bool | None,
    ) -> dict[str, float]:
        choose = _range_choice
        dimension = choose(self.config.dimension_scale, dimensions_high)
        pose = choose(self.config.initial_xy_offset_m, positive_pose)
        com = choose(self.config.estimated_com_error_m, positive_pose)
        return {
            "dimension_scale_x": dimension,
            "dimension_scale_y": dimension,
            "dimension_scale_z": dimension,
            "mass_scale": choose(self.config.mass_scale, mass_high),
            "initial_offset_x_m": pose,
            "initial_offset_y_m": -pose,
            "support_gap_m": choose(self.config.support_gap_m, positive_pose),
            "initial_yaw_rad": choose(self.config.initial_yaw_rad, positive_pose),
            "selected_gripper_friction": choose(
                self.config.selected_gripper_friction,
                contact_high,
            ),
            "contact_stiffness_scale": choose(
                self.config.contact_stiffness_scale,
                contact_high,
            ),
            "estimated_mass_scale": choose(
                self.config.estimated_mass_scale,
                mass_high,
            ),
            "estimated_com_error_x_m": com,
            "estimated_com_error_y_m": -com,
            "estimated_com_error_z_m": com,
        }

    def _apply_values(
        self,
        base_task_spec: TaskSpec,
        *,
        seed: int,
        sample_index: int,
        values: dict[str, float],
    ) -> Order9RandomizationSample:
        task_data = base_task_spec.to_dict()
        object_data = _target_object_data(task_data)
        geometry_data = _geometry_data(task_data, str(object_data["geometry_id"]))
        if geometry_data.get("geometry_type") != GeometryType.BOX.value:
            raise SchemaValidationError(
                "conservative Order 9 anchor randomization currently requires a box; "
                "later curriculum shape families use geometry mass-properties adapters"
            )
        params = dict(geometry_data.get("primitive_params") or {})
        nominal_size_raw = params.get("size_m")
        if not isinstance(nominal_size_raw, list) or len(nominal_size_raw) != 3:
            raise SchemaValidationError("Order 9 box geometry requires primitive_params.size_m")
        nominal_size = tuple(float(value) for value in nominal_size_raw)
        size = (
            nominal_size[0] * values["dimension_scale_x"],
            nominal_size[1] * values["dimension_scale_y"],
            nominal_size[2] * values["dimension_scale_z"],
        )
        nominal_mass = float(object_data["mass_kg"])
        target_mass = nominal_mass * values["mass_scale"]
        density = target_mass / (size[0] * size[1] * size[2])
        true_properties = cuboid_mass_properties(size, density_kg_m3=density)
        estimated_density = density * values["estimated_mass_scale"]
        estimated_base = cuboid_mass_properties(
            size,
            density_kg_m3=estimated_density,
        )
        estimated_properties = RigidBodyMassProperties(
            mass_kg=estimated_base.mass_kg,
            center_of_mass_object=(
                values["estimated_com_error_x_m"],
                values["estimated_com_error_y_m"],
                values["estimated_com_error_z_m"],
            ),
            inertia_kgm2=estimated_base.inertia_kgm2,
            volume_m3=estimated_base.volume_m3,
            density_kg_m3=estimated_base.density_kg_m3,
            source=estimated_base.source + ":estimator_model",
        )
        params["size_m"] = list(size)
        geometry_data["primitive_params"] = params
        object_data["mass_kg"] = true_properties.mass_kg
        object_data["inertia_kgm2"] = true_properties.inertia_kgm2
        object_data["center_of_mass_object"] = list(
            true_properties.center_of_mass_object
        )
        object_data["density_kg_m3"] = true_properties.density_kg_m3
        nominal_pose = list(object_data["pose_world"])
        yaw_quaternion = _yaw_quaternion(values["initial_yaw_rad"])
        object_data["pose_world"] = [
            float(nominal_pose[0]) + values["initial_offset_x_m"],
            float(nominal_pose[1]) + values["initial_offset_y_m"],
            self.config.support_top_z_m + size[2] / 2.0 + values["support_gap_m"],
            *yaw_quaternion,
        ]
        task_data["task_id"] = (
            f"{base_task_spec.task_id}_order9_{sample_index:06d}_{seed:08d}"
        )
        metadata = dict(task_data.get("metadata", {}) or {})
        metadata.update(
            {
                "randomization_family": "order9_conservative_order8_anchor",
                "randomization_version": ORDER9_RANDOMIZATION_VERSION,
                "randomization_seed": seed,
                "randomization_sample_index": sample_index,
                "true_mass_properties_source": true_properties.source,
                "estimated_mass_properties_source": estimated_properties.source,
            }
        )
        task_data["metadata"] = with_selected_robot_contact_material(
            metadata,
            target_entity_ids=[str(object_data["object_id"])],
            contact_modes=[ContactMode.GRASP],
            robot_static_friction=values["selected_gripper_friction"],
            robot_dynamic_friction=values["selected_gripper_friction"],
            friction_combine_mode="max",
            robot_surface_scope="selected_grasp_anchor_surfaces",
        )

        stiffness = (
            self.config.nominal_contact_stiffness_n_per_m
            * values["contact_stiffness_scale"]
        )
        # c = 2*zeta*sqrt(k*m_eff); scale c by sqrt(k*m) to preserve the
        # nominal damping ratio without guessing a new absolute zeta.
        damping = self.config.nominal_contact_damping_n_s_per_m * math.sqrt(
            values["contact_stiffness_scale"] * values["mass_scale"]
        )
        return Order9RandomizationSample(
            seed=seed,
            sample_index=sample_index,
            task_spec=TaskSpec.from_dict(task_data),
            true_mass_properties=true_properties,
            estimated_mass_properties=estimated_properties,
            selected_gripper_friction=values["selected_gripper_friction"],
            contact_stiffness_n_per_m=stiffness,
            contact_damping_n_s_per_m=damping,
            sampled_values=dict(values),
        )


@dataclass
class Order9ExpandedObjectRandomizationConfig(SchemaBase):
    equivalent_volume_scale: tuple[float, float] = (0.80, 1.20)
    mass_scale: tuple[float, float] = (0.80, 1.20)
    train_box_xy_ratio: tuple[float, float] = (0.65, 0.90)
    train_box_zy_ratio: tuple[float, float] = (0.30, 0.50)
    train_cylinder_height_diameter_ratio: tuple[float, float] = (0.60, 1.20)
    held_out_box_xy_ratio: tuple[float, float] = (0.95, 1.25)
    held_out_box_zy_ratio: tuple[float, float] = (0.55, 0.80)
    held_out_cylinder_height_diameter_ratio: tuple[float, float] = (1.40, 2.00)
    held_out_capsule_cylinder_height_diameter_ratio: tuple[float, float] = (0.75, 1.50)
    initial_xy_offset_m: tuple[float, float] = (-0.015, 0.015)
    support_gap_m: tuple[float, float] = (0.0, 0.008)
    initial_yaw_rad: tuple[float, float] = (-10.0 * math.pi / 180.0, 10.0 * math.pi / 180.0)
    object_friction: tuple[float, float] = (0.54, 0.72)
    selected_gripper_friction: tuple[float, float] = (4.0, 5.0)
    contact_stiffness_scale: tuple[float, float] = (0.90, 1.10)
    estimated_mass_scale: tuple[float, float] = (0.95, 1.05)
    estimated_com_error_m: tuple[float, float] = (-0.003, 0.003)
    nominal_contact_stiffness_n_per_m: float = 7500.0
    nominal_contact_damping_n_s_per_m: float = 75.0
    support_top_z_m: float = 0.15

    def validate(self) -> None:
        for name in (
            "equivalent_volume_scale",
            "mass_scale",
            "train_box_xy_ratio",
            "train_box_zy_ratio",
            "train_cylinder_height_diameter_ratio",
            "held_out_box_xy_ratio",
            "held_out_box_zy_ratio",
            "held_out_cylinder_height_diameter_ratio",
            "held_out_capsule_cylinder_height_diameter_ratio",
            "object_friction",
            "selected_gripper_friction",
            "contact_stiffness_scale",
            "estimated_mass_scale",
        ):
            _validate_range(getattr(self, name), name, positive=True)
        for name in (
            "initial_xy_offset_m",
            "support_gap_m",
            "initial_yaw_rad",
            "estimated_com_error_m",
        ):
            _validate_range(getattr(self, name), name, positive=False)
        if self.support_gap_m[0] < 0.0:
            raise SchemaValidationError("expanded support gap must be non-negative")
        for name in (
            "nominal_contact_stiffness_n_per_m",
            "nominal_contact_damping_n_s_per_m",
            "support_top_z_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise SchemaValidationError(f"expanded randomization {name} must be positive")


class Order9ExpandedObjectRandomizer:
    """Disjoint expanded-train and held-out primitive shape distributions."""

    TRAIN_FAMILIES = ("box_train", "sphere_train", "cylinder_train")
    HELD_OUT_FAMILIES = (
        "capsule_held_out",
        "box_aspect_held_out",
        "cylinder_aspect_held_out",
    )

    def __init__(
        self,
        config: Order9ExpandedObjectRandomizationConfig | None = None,
    ) -> None:
        self.config = config or Order9ExpandedObjectRandomizationConfig()
        self.config.validate()

    def sample(
        self,
        base_task_spec: TaskSpec,
        *,
        seed: int,
        sample_index: int = 0,
        held_out: bool = False,
    ) -> Order9RandomizationSample:
        if seed < 0 or sample_index < 0:
            raise ValueError("expanded randomization seed and sample_index must be non-negative")
        rng = random.Random(_derived_seed(seed, sample_index))
        task_data = base_task_spec.to_dict()
        object_data = _target_object_data(task_data)
        geometry_data = _geometry_data(task_data, str(object_data["geometry_id"]))
        nominal_geometry = GeometrySpec.from_dict(geometry_data)
        nominal_mass = float(object_data["mass_kg"])
        nominal_density = (
            float(object_data["density_kg_m3"])
            if object_data.get("density_kg_m3") is not None
            else nominal_mass
            / mass_properties_from_geometry(
                nominal_geometry, density_kg_m3=1.0
            ).volume_m3
        )
        nominal_properties = mass_properties_from_geometry(
            nominal_geometry, density_kg_m3=nominal_density
        )
        volume_scale = _uniform(self.config.equivalent_volume_scale, rng)
        desired_volume = nominal_properties.volume_m3 * volume_scale
        family_pool = self.HELD_OUT_FAMILIES if held_out else self.TRAIN_FAMILIES
        family = family_pool[rng.randrange(len(family_pool))]
        geometry_type, primitive_params, half_height = self._sample_geometry(
            family,
            desired_volume=desired_volume,
            rng=rng,
        )
        geometry_data.update(
            {
                "geometry_type": geometry_type.value,
                "primitive_params": primitive_params,
                "asset_path": None,
                "scale": [1.0, 1.0, 1.0],
                "collision_model": CollisionModel.PRIMITIVE.value,
            }
        )
        sampled_geometry = GeometrySpec.from_dict(geometry_data)
        unit_properties = mass_properties_from_geometry(
            sampled_geometry, density_kg_m3=1.0
        )
        target_mass = nominal_mass * _uniform(self.config.mass_scale, rng)
        density = target_mass / unit_properties.volume_m3
        true_properties = mass_properties_from_geometry(
            sampled_geometry, density_kg_m3=density
        )
        estimated_density = density * _uniform(self.config.estimated_mass_scale, rng)
        estimated_base = mass_properties_from_geometry(
            sampled_geometry, density_kg_m3=estimated_density
        )
        com_error = tuple(
            _uniform(self.config.estimated_com_error_m, rng) for _ in range(3)
        )
        estimated_properties = RigidBodyMassProperties(
            mass_kg=estimated_base.mass_kg,
            center_of_mass_object=com_error,  # type: ignore[arg-type]
            inertia_kgm2=estimated_base.inertia_kgm2,
            volume_m3=estimated_base.volume_m3,
            density_kg_m3=estimated_base.density_kg_m3,
            source=estimated_base.source + ":estimator_model",
        )
        object_data["mass_kg"] = true_properties.mass_kg
        object_data["inertia_kgm2"] = true_properties.inertia_kgm2
        object_data["center_of_mass_object"] = list(true_properties.center_of_mass_object)
        object_data["density_kg_m3"] = true_properties.density_kg_m3
        object_data["friction"] = _uniform(self.config.object_friction, rng)
        nominal_pose = list(object_data["pose_world"])
        new_center_z = (
            self.config.support_top_z_m
            + half_height
            + _uniform(self.config.support_gap_m, rng)
        )
        yaw = _uniform(self.config.initial_yaw_rad, rng)
        object_data["pose_world"] = [
            float(nominal_pose[0]) + _uniform(self.config.initial_xy_offset_m, rng),
            float(nominal_pose[1]) + _uniform(self.config.initial_xy_offset_m, rng),
            new_center_z,
            *_yaw_quaternion(yaw),
        ]
        for goal in task_data["goals"]:
            if goal.get("target_entity_id") != object_data["object_id"]:
                continue
            target_pose = goal.get("target_pose_world")
            if isinstance(target_pose, list) and len(target_pose) == 7:
                vertical_offset = float(target_pose[2]) - float(nominal_pose[2])
                target_pose[2] = new_center_z + vertical_offset
        task_data["task_id"] = (
            f"{base_task_spec.task_id}_order9_expanded_{sample_index:06d}_{seed:08d}"
        )
        selected_friction = _uniform(self.config.selected_gripper_friction, rng)
        metadata = dict(task_data.get("metadata", {}) or {})
        metadata.update(
            {
                "randomization_family": family,
                "randomization_split": "held_out" if held_out else "train",
                "randomization_version": ORDER9_EXPANDED_RANDOMIZATION_VERSION,
                "randomization_seed": seed,
                "randomization_sample_index": sample_index,
                "true_mass_properties_source": true_properties.source,
                "estimated_mass_properties_source": estimated_properties.source,
                "train_held_out_family_disjoint": True,
            }
        )
        task_data["metadata"] = with_selected_robot_contact_material(
            metadata,
            target_entity_ids=[str(object_data["object_id"])],
            contact_modes=[ContactMode.GRASP],
            robot_static_friction=selected_friction,
            robot_dynamic_friction=selected_friction,
            friction_combine_mode="max",
            robot_surface_scope="selected_grasp_anchor_surfaces",
        )
        stiffness_scale = _uniform(self.config.contact_stiffness_scale, rng)
        damping = self.config.nominal_contact_damping_n_s_per_m * math.sqrt(
            stiffness_scale * (true_properties.mass_kg / nominal_mass)
        )
        family_id = float(
            (self.TRAIN_FAMILIES + self.HELD_OUT_FAMILIES).index(family)
        )
        return Order9RandomizationSample(
            seed=seed,
            sample_index=sample_index,
            task_spec=TaskSpec.from_dict(task_data),
            true_mass_properties=true_properties,
            estimated_mass_properties=estimated_properties,
            selected_gripper_friction=selected_friction,
            contact_stiffness_n_per_m=(
                self.config.nominal_contact_stiffness_n_per_m * stiffness_scale
            ),
            contact_damping_n_s_per_m=damping,
            sampled_values={
                "family_id": family_id,
                "held_out": 1.0 if held_out else 0.0,
                "equivalent_volume_scale": volume_scale,
                "true_mass_kg": true_properties.mass_kg,
                "true_density_kg_m3": true_properties.density_kg_m3,
                "estimated_mass_kg": estimated_properties.mass_kg,
                "object_friction": float(object_data["friction"]),
                "selected_gripper_friction": selected_friction,
                "contact_stiffness_scale": stiffness_scale,
                "initial_yaw_rad": yaw,
            },
            randomization_version=ORDER9_EXPANDED_RANDOMIZATION_VERSION,
        )

    def _sample_geometry(
        self,
        family: str,
        *,
        desired_volume: float,
        rng: random.Random,
    ) -> tuple[GeometryType, dict[str, float | list[float]], float]:
        if family in {"box_train", "box_aspect_held_out"}:
            xy_bounds = (
                self.config.train_box_xy_ratio
                if family == "box_train"
                else self.config.held_out_box_xy_ratio
            )
            zy_bounds = (
                self.config.train_box_zy_ratio
                if family == "box_train"
                else self.config.held_out_box_zy_ratio
            )
            xy = _uniform(xy_bounds, rng)
            zy = _uniform(zy_bounds, rng)
            y = (desired_volume / (xy * zy)) ** (1.0 / 3.0)
            size = [xy * y, y, zy * y]
            return GeometryType.BOX, {"size_m": size}, size[2] / 2.0
        if family == "sphere_train":
            radius = (3.0 * desired_volume / (4.0 * math.pi)) ** (1.0 / 3.0)
            return GeometryType.SPHERE, {"radius_m": radius}, radius
        if family in {"cylinder_train", "cylinder_aspect_held_out"}:
            bounds = (
                self.config.train_cylinder_height_diameter_ratio
                if family == "cylinder_train"
                else self.config.held_out_cylinder_height_diameter_ratio
            )
            ratio = _uniform(bounds, rng)
            radius = (desired_volume / (2.0 * math.pi * ratio)) ** (1.0 / 3.0)
            height = 2.0 * radius * ratio
            return (
                GeometryType.CYLINDER,
                {"radius_m": radius, "height_m": height},
                height / 2.0,
            )
        if family == "capsule_held_out":
            ratio = _uniform(
                self.config.held_out_capsule_cylinder_height_diameter_ratio,
                rng,
            )
            coefficient = math.pi * (2.0 * ratio + 4.0 / 3.0)
            radius = (desired_volume / coefficient) ** (1.0 / 3.0)
            cylinder_height = 2.0 * radius * ratio
            return (
                GeometryType.CAPSULE,
                {"radius_m": radius, "height_m": cylinder_height},
                cylinder_height / 2.0 + radius,
            )
        raise AssertionError(family)


def _target_object_data(task_data: dict[str, Any]) -> dict[str, Any]:
    target_id = next(
        (
            goal.get("target_entity_id")
            for goal in task_data["goals"]
            if goal.get("goal_type") == "object_pose"
        ),
        None,
    )
    for obj in task_data["scene"]["objects"]:
        if obj.get("object_id") == target_id:
            return obj
    raise SchemaValidationError("Order 9 randomization requires an object_pose target")


def _geometry_data(task_data: dict[str, Any], geometry_id: str) -> dict[str, Any]:
    for geometry in task_data["scene"]["geometry_library"]:
        if geometry.get("geometry_id") == geometry_id:
            return geometry
    raise SchemaValidationError(
        f"Order 9 randomization target geometry {geometry_id!r} is missing"
    )


def _yaw_quaternion(yaw_rad: float) -> list[float]:
    return [0.0, 0.0, math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0)]


def _uniform(bounds: tuple[float, float], rng: random.Random) -> float:
    return rng.uniform(float(bounds[0]), float(bounds[1]))


def _derived_seed(seed: int, sample_index: int) -> int:
    # Keep sampling independent of Python's process-randomized hash while
    # making sample_index an actual part of the episode randomization key.
    return (int(seed) * 0x9E3779B185EBCA87 + int(sample_index)) & ((1 << 64) - 1)


def _range_choice(bounds: tuple[float, float], high: bool | None) -> float:
    if high is None:
        return (float(bounds[0]) + float(bounds[1])) / 2.0
    return float(bounds[1 if high else 0])


def _validate_range(
    bounds: tuple[float, float],
    path: str,
    *,
    positive: bool,
) -> None:
    require_len(bounds, 2, path)
    lower, upper = (float(value) for value in bounds)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise SchemaValidationError(f"{path} must be a finite ordered range")
    if positive and lower <= 0.0:
        raise SchemaValidationError(f"{path} must be positive")
