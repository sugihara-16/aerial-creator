from __future__ import annotations

import pytest

from amsrr.geometry.geometry_processor import GeometryProcessor
from amsrr.schemas.common import ContactMode
from amsrr.schemas.task_spec import GeometrySpec


def test_geometry_processor_box_regions() -> None:
    spec = GeometrySpec.from_dict(
        {
            "geometry_id": "box_geom",
            "geometry_type": "box",
            "primitive_params": {"size_m": [0.30, 0.20, 0.15]},
            "asset_path": None,
            "scale": [1.0, 1.0, 1.0],
            "collision_model": "primitive",
        }
    )
    descriptor = GeometryProcessor().process_geometry(
        spec,
        entity_id="box_01",
        friction=0.6,
        allowed_contact_modes=[ContactMode.GRASP, ContactMode.SUPPORT],
    )

    assert descriptor.geometry_id == "box_geom"
    assert descriptor.global_shape_features.bbox_m == pytest.approx((0.30, 0.20, 0.15))
    assert descriptor.global_shape_features.volume_m3 == pytest.approx(0.009)
    assert descriptor.global_shape_features.surface_area_m2 == pytest.approx(0.27)
    assert len(descriptor.surface_patch_graph.nodes) == 6
    assert len(descriptor.contact_region_graph.nodes) == 6
    assert {region.region_type for region in descriptor.contact_region_graph.nodes} == {"face"}
    assert {region.entity_id for region in descriptor.contact_region_graph.nodes} == {"box_01"}
    assert {tuple(region.normal_summary_object) for region in descriptor.contact_region_graph.nodes} == {
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    }
    assert sum(edge.edge_type == "opposite_patch" for edge in descriptor.surface_patch_graph.edges) == 3
    assert descriptor.collision_ref.startswith("primitive://sha256:")
    assert descriptor.exact_geometry_ref.startswith("primitive://sha256:")
    assert "assets/" not in descriptor.collision_ref
    assert "assets/" not in descriptor.exact_geometry_ref


def test_geometry_processor_mesh_smoke() -> None:
    spec = GeometrySpec.from_dict(
        {
            "geometry_id": "battery_mesh",
            "geometry_type": "mesh",
            "primitive_params": None,
            "asset_path": "module_urdf/mesh/battery_1.STL",
            "scale": [1.0, 1.0, 1.0],
            "collision_model": "mesh",
        }
    )
    descriptor = GeometryProcessor().process_geometry(
        spec,
        entity_id="battery_1",
        allowed_contact_modes=[ContactMode.SUPPORT],
    )

    assert descriptor.geometry_id == "battery_mesh"
    assert descriptor.global_shape_features.surface_area_m2 > 0.0
    assert descriptor.global_shape_features.volume_m3 > 0.0
    assert all(value > 0.0 for value in descriptor.global_shape_features.bbox_m)
    assert 1 <= len(descriptor.surface_patch_graph.nodes) <= 6
    assert len(descriptor.surface_patch_graph.nodes) == len(descriptor.contact_region_graph.nodes)
    assert {region.region_type for region in descriptor.contact_region_graph.nodes} == {"mesh_patch_cluster"}
    assert {patch.entity_id for patch in descriptor.surface_patch_graph.nodes} == {"battery_1"}
    assert all(patch.allowed_contact_modes == [ContactMode.SUPPORT] for patch in descriptor.surface_patch_graph.nodes)
    assert descriptor.collision_ref.startswith("mesh://sha256:")
    assert descriptor.exact_geometry_ref.startswith("mesh://sha256:")
    assert "module_urdf" not in descriptor.collision_ref
    assert "module_urdf" not in descriptor.exact_geometry_ref

