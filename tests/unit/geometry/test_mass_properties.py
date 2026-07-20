from __future__ import annotations

import math

import pytest

from amsrr.geometry.mass_properties import (
    cuboid_mass_properties,
    cylinder_mass_properties,
    sphere_mass_properties,
)


def test_uniform_cuboid_mass_com_and_inertia_are_exact() -> None:
    properties = cuboid_mass_properties(
        (0.30, 0.40, 0.15),
        density_kg_m3=1.0 / (0.30 * 0.40 * 0.15),
    )

    assert properties.mass_kg == pytest.approx(1.0)
    assert properties.center_of_mass_object == (0.0, 0.0, 0.0)
    assert properties.inertia_kgm2 == pytest.approx(
        [
            (0.40**2 + 0.15**2) / 12.0,
            0.0,
            0.0,
            (0.30**2 + 0.15**2) / 12.0,
            0.0,
            (0.30**2 + 0.40**2) / 12.0,
        ]
    )


def test_sphere_and_cylinder_use_uniform_density_geometry() -> None:
    sphere = sphere_mass_properties(0.1, density_kg_m3=500.0)
    cylinder = cylinder_mass_properties(0.1, 0.3, density_kg_m3=500.0)

    assert sphere.mass_kg == pytest.approx(500.0 * 4.0 * math.pi * 0.1**3 / 3.0)
    assert sphere.inertia_kgm2[0] == pytest.approx(
        2.0 * sphere.mass_kg * 0.1**2 / 5.0
    )
    assert cylinder.mass_kg == pytest.approx(500.0 * math.pi * 0.1**2 * 0.3)
    assert cylinder.inertia_kgm2[5] == pytest.approx(
        cylinder.mass_kg * 0.1**2 / 2.0
    )
