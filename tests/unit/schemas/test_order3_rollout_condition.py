from __future__ import annotations

import json

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.order3_rollout_condition import (
    Order3RolloutCondition,
    build_order3_rollout_condition,
    order3_rollout_condition_hash,
)


def test_order3_rollout_condition_is_canonical_and_hash_bound() -> None:
    condition = build_order3_rollout_condition(
        stage_id="disturbed_waypoints",
        task_mode="waypoint",
        seed=42,
        waypoint_position_offset_world=(0.25, 0.0, 0.1),
        external_wrench_body=(1.0, 0.0, 0.0, 0.0, 0.1, 0.0),
        mass_scale=1.05,
        inertia_scale=0.97,
        thrust_scale=0.95,
    )

    assert condition.condition_hash == order3_rollout_condition_hash(condition)
    assert condition.to_canonical_json() == condition.to_canonical_json()
    assert Order3RolloutCondition.from_json(condition.to_canonical_json()) == condition

    tampered = json.loads(condition.to_canonical_json())
    tampered["mass_scale"] = 1.10
    with pytest.raises(SchemaValidationError, match="condition_hash"):
        Order3RolloutCondition.from_dict(tampered)


def test_order3_rollout_condition_rejects_invalid_dynamics_or_timing() -> None:
    with pytest.raises(SchemaValidationError, match="mass_scale"):
        build_order3_rollout_condition(
            stage_id="invalid",
            task_mode="hover",
            seed=0,
            mass_scale=0.0,
        )
    with pytest.raises(SchemaValidationError, match="disturbance_duration_s"):
        build_order3_rollout_condition(
            stage_id="invalid",
            task_mode="takeoff",
            seed=0,
            disturbance_duration_s=-1.0,
        )
    with pytest.raises(SchemaValidationError, match="task_mode"):
        build_order3_rollout_condition(
            stage_id="invalid",
            task_mode="landing",  # type: ignore[arg-type]
            seed=0,
        )
