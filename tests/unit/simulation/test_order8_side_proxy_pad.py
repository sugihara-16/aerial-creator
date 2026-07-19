from __future__ import annotations

from collections import Counter
from dataclasses import replace
import math

import numpy as np
import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.simulation.order8_side_proxy_pad import (
    ORDER8_SIDE_PROXY_PAD_PREVIEW_VERSION,
    build_order8_side_proxy_pad_specs,
    load_order8_side_proxy_pad_preview_config,
)


CONFIG_PATH = "configs/training/order8_side_proxy_pad_preview.yaml"
URDF_PATH = "assets/robots/holon/holon.urdf"


def test_real_holon_side_proxy_micro_pads_follow_only_conical_mesh() -> None:
    config = load_order8_side_proxy_pad_preview_config(CONFIG_PATH)
    specs = build_order8_side_proxy_pad_specs(
        urdf_path=URDF_PATH,
        config=config,
    )

    assert config.version == ORDER8_SIDE_PROXY_PAD_PREVIEW_VERSION
    assert config.acceptance_eligible is False
    assert config.visual_approval_recorded is True
    assert config.contact_runtime_enabled is True
    counts = Counter(spec.link_id for spec in specs)
    assert set(counts) == {"yaw_dock_mech1", "yaw_dock_mech2"}
    # Neighbouring cells are merged, but the result remains a local surface
    # tiling rather than one large plate per link.
    assert all(60 <= count <= 90 for count in counts.values())
    assert counts["yaw_dock_mech1"] == counts["yaw_dock_mech2"]

    for link_id in config.link_ids:
        link_specs = [spec for spec in specs if spec.link_id == link_id]
        assert {spec.axial_band_index for spec in link_specs} == set(
            range(config.axial_band_count)
        )
        assert {spec.circumferential_segment_index for spec in link_specs} == set(
            range(config.circumferential_segment_count)
        )
        # Normals cover the full existing circumference instead of one
        # object-facing hard-coded plane.
        normals = np.asarray(
            [spec.outward_normal_local for spec in link_specs], dtype=float
        )
        surface_points = np.asarray(
            [spec.representative_surface_point_local for spec in link_specs],
            dtype=float,
        )
        assert np.min(surface_points[:, 0]) >= config.cone_axial_min_m
        assert np.max(surface_points[:, 0]) <= config.cone_axial_max_m
        assert np.min(normals[:, 0]) >= config.cone_normal_axial_min
        assert np.max(normals[:, 0]) <= config.cone_normal_axial_max
        # The cone normal also has a positive axial component, so its radial
        # Y/Z components cannot individually reach unit magnitude.
        assert np.min(normals[:, 1]) < -0.75
        assert np.max(normals[:, 1]) > 0.75
        assert np.min(normals[:, 2]) < -0.75
        assert np.max(normals[:, 2]) > 0.75

        for spec in link_specs:
            assert config.tile_size_min_m <= spec.size_m[0] <= config.tile_size_max_m
            assert config.tile_size_min_m <= spec.size_m[1] <= config.tile_size_max_m
            assert spec.size_m[2] == pytest.approx(0.0008)
            assert spec.inner_face_surface_gap_m == pytest.approx(0.0002)
            assert spec.surface_fit_max_gap_m <= 0.0015
            assert spec.candidate_triangle_count >= spec.surface_triangle_count >= 1

            rotation = np.column_stack(
                (
                    spec.axial_axis_local,
                    spec.circumferential_axis_local,
                    spec.outward_normal_local,
                )
            )
            assert rotation.T @ rotation == pytest.approx(np.eye(3), abs=1.0e-8)
            assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1.0e-8)
            assert math.sqrt(
                sum(value * value for value in spec.orientation_local_xyzw)
            ) == pytest.approx(1.0)

            normal = np.asarray(spec.outward_normal_local)
            center_offset = np.asarray(spec.center_local) - np.asarray(
                spec.representative_surface_point_local
            )
            assert center_offset == pytest.approx(
                normal * (config.mesh_clearance_m + 0.5 * config.thickness_m),
                abs=1.0e-10,
            )


def test_side_proxy_preview_requires_visual_approval_and_rejects_acceptance() -> None:
    config = load_order8_side_proxy_pad_preview_config(CONFIG_PATH)

    with pytest.raises(SchemaValidationError, match="acceptance-ineligible"):
        replace(config, acceptance_eligible=True).validate()
    with pytest.raises(SchemaValidationError, match="visual approval"):
        replace(config, visual_approval_recorded=False).validate()
