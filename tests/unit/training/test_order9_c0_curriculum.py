from __future__ import annotations

from amsrr.schemas.order8 import load_order8_natural_contact_config
from amsrr.training.order9_c0_curriculum import (
    build_order9_c0_teacher_conditions,
)
from amsrr.training.order9_curriculum import load_order9_learning_config


def test_c0_conditions_cover_nominal_boundaries_and_seeded_diversity() -> None:
    base = load_order8_natural_contact_config(
        "configs/training/order8_natural_contact.yaml"
    )
    learning = load_order9_learning_config()
    first = build_order9_c0_teacher_conditions(
        base,
        episode_count=20,
        seed_start=9009,
        randomization=learning.randomization,
    )
    second = build_order9_c0_teacher_conditions(
        base,
        episode_count=20,
        seed_start=9009,
        randomization=learning.randomization,
    )
    assert [value.to_dict() for value in first] == [
        value.to_dict() for value in second
    ]
    assert len({value.condition_id for value in first}) == 20
    assert first[0].sample_kind == "nominal"
    assert first[0].order8_config.stable_hash() == base.stable_hash()
    assert [value.sample_kind for value in first[1:5]] == [
        "small_light_low_contact",
        "large_heavy_high_contact",
        "small_heavy_low_contact",
        "large_light_high_contact",
    ]
    assert all(value.sample_kind == "seeded_random" for value in first[5:])
    assert all(
        0.95 * base.object_mass_kg
        <= value.order8_config.object_mass_kg
        <= 1.05 * base.object_mass_kg
        for value in first
    )
    assert all(
        abs(
            value.order8_config.initial_object_standoff_m
            - base.initial_object_standoff_m
        )
        <= 0.005 + 1.0e-12
        for value in first
    )
