from __future__ import annotations

import pytest

from amsrr.training.order9_training_nominal_preload_deficit import (
    order9_training_preload_deficit_category_indices,
    parse_order9_training_nominal_preload_deficit_mm,
    plan_order9_training_nominal_preload_deficit,
)


def test_parse_requires_baseline_and_unique_bounded_values() -> None:
    assert parse_order9_training_nominal_preload_deficit_mm("0,2,4,6,8") == (
        0.0,
        2.0,
        4.0,
        6.0,
        8.0,
    )
    with pytest.raises(ValueError, match="include the unmodified"):
        parse_order9_training_nominal_preload_deficit_mm("2,4")
    with pytest.raises(ValueError, match="unique"):
        parse_order9_training_nominal_preload_deficit_mm("0,2,2")


def test_plan_is_balanced_seeded_and_never_reverses_compression() -> None:
    plan = plan_order9_training_nominal_preload_deficit(
        nominal_lead_m=0.006,
        requested_deficit_mm=(0.0, 2.0, 4.0, 6.0, 8.0),
        environment_count=12,
        seed=41,
    )
    assert plan == plan_order9_training_nominal_preload_deficit(
        nominal_lead_m=0.006,
        requested_deficit_mm=(0.0, 2.0, 4.0, 6.0, 8.0),
        environment_count=12,
        seed=41,
    )
    counts = {
        value: plan.deficit_m_by_environment.count(value)
        for value in set(plan.deficit_m_by_environment)
    }
    assert max(counts.values()) - min(counts.values()) <= 1
    assert min(plan.total_lead_m_by_environment) == 0.0
    assert max(plan.total_lead_m_by_environment) == pytest.approx(0.006)
    assert all(0.0 <= value <= 1.0 for value in plan.scale_by_environment)


def test_effective_preload_deficit_maps_to_positive_categorical_bin() -> None:
    assert order9_training_preload_deficit_category_indices(
        nominal_lead_m=0.006,
        total_lead_m_by_environment=(0.006, 0.004, 0.0),
        category_step_m=0.0005,
        category_count=81,
    ) == (40, 44, 52)

    with pytest.raises(ValueError, match="off category grid"):
        order9_training_preload_deficit_category_indices(
            nominal_lead_m=0.0061,
            total_lead_m_by_environment=(0.004,),
            category_step_m=0.0005,
            category_count=81,
        )
