from __future__ import annotations

from pathlib import Path

from amsrr.training.order9_r1_rotation_stability_repair import (
    ORDER9_R1_ROTATION_STABILITY_OVERRIDES_RELATIVE,
    load_order9_r1_rotation_stability_repairs,
)

REPOSITORY = Path(__file__).resolve().parents[3]


def test_rotation_repair_is_additive_and_bound_to_failure_evidence() -> None:
    rules = load_order9_r1_rotation_stability_repairs(
        REPOSITORY / ORDER9_R1_ROTATION_STABILITY_OVERRIDES_RELATIVE,
        repository_root=REPOSITORY,
    )
    assert set(rules) == {"train-000003-3b95da871f01"}
    rule = rules["train-000003-3b95da871f01"]
    assert rule.selected_surface_port_ids == (4, 16)
    assert rule.preferred_candidate_group_ids == ("slot_0:grasp_pair:0",)
    assert rule.required_rotation_axis_object == (0.0, 1.0, 0.0)
    assert rule.minimum_force_moment_arm_m == 0.10
    assert rule.grasp_contact_height_offset_m == 0.0
