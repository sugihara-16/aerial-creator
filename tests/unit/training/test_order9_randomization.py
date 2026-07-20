from __future__ import annotations

import pytest

from amsrr.geometry.contact_material import resolve_contact_friction
from amsrr.schemas.common import ContactMode
from amsrr.geometry.mass_properties import (
    cuboid_mass_properties,
    mass_properties_from_geometry,
)
from amsrr.schemas.task_spec import GeometryType
from amsrr.training.order9_randomization import (
    Order9ConservativeRandomizer,
    Order9ExpandedObjectRandomizer,
)
from amsrr.training.order9_teacher import build_order8_grasp_carry_task_spec


def test_conservative_randomization_is_deterministic_and_order8_bounded() -> None:
    base = _task()
    randomizer = Order9ConservativeRandomizer()

    first = randomizer.sample(base, seed=17, sample_index=3)
    second = randomizer.sample(base, seed=17, sample_index=3)

    assert first.to_dict() == second.to_dict()
    obj = first.task_spec.scene.objects[0]
    geometry = first.task_spec.scene.geometry_library[0]
    size = tuple(geometry.primitive_params["size_m"])
    assert all(
        nominal * 0.975 <= value <= nominal * 1.025
        for nominal, value in zip((0.30, 0.40, 0.15), size)
    )
    assert 0.95 <= obj.mass_kg <= 1.05
    assert obj.pose_world[2] >= 0.15 + size[2] / 2.0
    assert 4.275 <= first.selected_gripper_friction <= 4.725
    assert 7125.0 <= first.contact_stiffness_n_per_m <= 7875.0
    resolved = resolve_contact_friction(
        first.task_spec.metadata,
        target_entity_id=obj.object_id,
        contact_mode=ContactMode.GRASP,
        target_surface_friction=obj.friction,
    )
    assert resolved.effective_friction == pytest.approx(
        first.selected_gripper_friction
    )


def test_true_inertia_is_recomputed_from_geometry_and_uniform_density() -> None:
    sample = Order9ConservativeRandomizer().sample(_task(), seed=23)
    obj = sample.task_spec.scene.objects[0]
    geometry = sample.task_spec.scene.geometry_library[0]
    size = tuple(geometry.primitive_params["size_m"])
    expected = cuboid_mass_properties(
        size,
        density_kg_m3=obj.density_kg_m3,
    )

    assert obj.mass_kg == pytest.approx(expected.mass_kg)
    assert obj.center_of_mass_object == expected.center_of_mass_object
    assert obj.inertia_kgm2 == pytest.approx(expected.inertia_kgm2)
    assert sample.estimated_mass_properties.mass_kg != pytest.approx(
        sample.true_mass_properties.mass_kg,
        abs=1.0e-12,
    )


def test_boundary_preflight_includes_nominal_and_physical_corners() -> None:
    samples = Order9ConservativeRandomizer().boundary_preflight_samples(_task())

    assert len(samples) == 5
    assert samples[0].selected_gripper_friction == pytest.approx(4.5)
    assert samples[0].contact_stiffness_n_per_m == pytest.approx(7500.0)
    assert samples[0].contact_damping_n_s_per_m == pytest.approx(75.0)
    assert all(
        sample.task_spec.scene.objects[0].pose_world[2]
        >= 0.15
        + sample.task_spec.scene.geometry_library[0].primitive_params["size_m"][2]
        / 2.0
        for sample in samples
    )


def test_expanded_train_and_held_out_shapes_are_disjoint_and_physics_consistent() -> None:
    randomizer = Order9ExpandedObjectRandomizer()
    train_samples = [
        randomizer.sample(_task(), seed=41, sample_index=index, held_out=False)
        for index in range(24)
    ]
    held_out_samples = [
        randomizer.sample(_task(), seed=43, sample_index=index, held_out=True)
        for index in range(24)
    ]

    train_families = {
        sample.task_spec.metadata["randomization_family"] for sample in train_samples
    }
    held_out_families = {
        sample.task_spec.metadata["randomization_family"] for sample in held_out_samples
    }
    assert train_families <= set(randomizer.TRAIN_FAMILIES)
    assert held_out_families <= set(randomizer.HELD_OUT_FAMILIES)
    assert train_families
    assert held_out_families
    assert train_families.isdisjoint(held_out_families)
    assert all(
        sample.task_spec.metadata["randomization_split"] == "train"
        for sample in train_samples
    )
    assert all(
        sample.task_spec.metadata["randomization_split"] == "held_out"
        for sample in held_out_samples
    )

    for sample in [*train_samples, *held_out_samples]:
        task = sample.task_spec
        obj = task.scene.objects[0]
        geometry = next(
            item for item in task.scene.geometry_library if item.geometry_id == obj.geometry_id
        )
        expected = mass_properties_from_geometry(
            geometry, density_kg_m3=obj.density_kg_m3
        )
        assert obj.mass_kg == pytest.approx(expected.mass_kg)
        assert obj.inertia_kgm2 == pytest.approx(expected.inertia_kgm2)
        assert obj.center_of_mass_object == pytest.approx(
            expected.center_of_mass_object
        )
        assert obj.pose_world[2] >= 0.15
    assert GeometryType.CAPSULE not in {
        next(
            item
            for item in sample.task_spec.scene.geometry_library
            if item.geometry_id == sample.task_spec.scene.objects[0].geometry_id
        ).geometry_type
        for sample in train_samples
    }


def _task():
    return build_order8_grasp_carry_task_spec(
        object_pose_world=(0.5, 0.0, 0.225, 0.0, 0.0, 0.0, 1.0),
        object_size_m=(0.30, 0.40, 0.15),
        object_mass_kg=1.0,
        object_friction=0.6,
        required_transport_distance_m=0.20,
        support_height_m=0.15,
        max_contact_force_n=30.0,
        max_contact_torque_nm=5.0,
    )
