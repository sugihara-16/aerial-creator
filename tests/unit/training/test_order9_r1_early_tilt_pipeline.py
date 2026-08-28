from __future__ import annotations

import json

import pytest

from amsrr.training import order9_r1_early_tilt_pipeline as early
from amsrr.training.order9_r1_calibration_runner import (
    enumerate_order9_r1_calibration_level_cases,
)

_PROTOCOL = "configs/training/order9_r1_calibration_protocol_v1.yaml"
_BUCKETS = (
    "artifacts/p4_full/order9/releases/c3_pi_l_promoted_update18_v1/runtime/"
    "rollout_buckets_current_lineage_v5/"
    "manifest_c3_release_smooth_clear300_current_config_min2_preload_screened_v2.json"
)


def test_r1_early_tilt_pipeline_forwards_limit_before_dense_generation(
    monkeypatch,
) -> None:
    pipeline = early.Order9R1EarlyTiltTeacherScreenPipeline(
        repository_root=".",
        source_bucket_manifest_path=_BUCKETS,
        minimum_normalized_joint_limit_reserve=0.01,
        anchor_position_tolerance_m=0.03,
        enforce_joint_limit_reserve_during_ik=True,
    )
    case = enumerate_order9_r1_calibration_level_cases(
        protocol_path=_PROTOCOL,
        repository_root=".",
        level_id="r1_l1_10mm_5deg",
        split="train",
    )[0]
    generated = {}
    projected = {}
    nominal_sentinel = object()
    projected_sentinel = object()

    def fake_generate(**kwargs):
        generated.update(kwargs)
        return nominal_sentinel

    def fake_project(nominal, **kwargs):
        projected["nominal"] = nominal
        projected.update(kwargs)
        return projected_sentinel

    monkeypatch.setattr(
        early,
        "generate_order9_c3_nominal_grasp_trajectory",
        fake_generate,
    )
    monkeypatch.setattr(
        early,
        "project_order9_r1_nominal_joint_reserve",
        fake_project,
    )

    prepared = pipeline.prepare(case)

    assert prepared.teacher_trajectory_complete is True
    assert prepared.payload is projected_sentinel
    assert generated["maximum_contact_solution_body_tilt_rad"] == (
        pipeline.screen_config.maximum_body_tilt_rad
    )
    assert projected["nominal"] is nominal_sentinel
    assert projected["minimum_normalized_joint_limit_reserve"] == 0.01
    assert projected["anchor_position_tolerance_m"] == 0.03


def test_r1_early_tilt_pipeline_retries_without_c3_surface_pin(
    monkeypatch,
) -> None:
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        if "preferred_surface_port_ids" in kwargs:
            raise early.SchemaValidationError("preferred teacher is extreme")
        return "fallback-nominal"

    monkeypatch.setattr(
        early,
        "generate_order9_c3_nominal_grasp_trajectory",
        fake_generate,
    )

    result = early._generate_with_safe_surface_fallback(
        task_spec=object(),
        structural_target=object(),
        physical_model=object(),
        preferred_surface_port_id_options=((8, 13),),
        preferred_candidate_options=(),
        maximum_body_tilt_rad=1.0,
    )

    assert result == "fallback-nominal"
    assert calls[0]["preferred_surface_port_ids"] == (8, 13)
    assert "preferred_surface_port_ids" not in calls[1]
    assert all(
        value["maximum_contact_solution_body_tilt_rad"] == 1.0 for value in calls
    )


def test_r1_teacher_override_pins_safe_surface_pairs() -> None:
    pipeline = early.Order9R1EarlyTiltTeacherScreenPipeline(
        repository_root=".",
        source_bucket_manifest_path=_BUCKETS,
    )

    assert pipeline.teacher_surface_overrides["train-000016-a9b26f370bba"] == ((2, 6),)
    assert pipeline.teacher_surface_overrides["train-000027-7fe33c662d36"] == (
        (21, 29),
        (19, 22),
    )


def test_r1_early_tilt_pipeline_tries_configured_pairs_in_order(
    monkeypatch,
) -> None:
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        if kwargs.get("preferred_surface_port_ids") == (21, 29):
            raise early.SchemaValidationError("first pair is unsuitable")
        return "second-pair-nominal"

    monkeypatch.setattr(
        early,
        "generate_order9_c3_nominal_grasp_trajectory",
        fake_generate,
    )

    result = early._generate_with_safe_surface_fallback(
        task_spec=object(),
        structural_target=object(),
        physical_model=object(),
        preferred_surface_port_id_options=((21, 29), (19, 22)),
        preferred_candidate_options=(),
        maximum_body_tilt_rad=1.0,
    )

    assert result == "second-pair-nominal"
    assert [value["preferred_surface_port_ids"] for value in calls] == [
        (21, 29),
        (19, 22),
    ]


def test_r1_teacher_override_rejects_duplicate_ports(tmp_path) -> None:
    source = tmp_path / "invalid.json"
    source.write_text(
        json.dumps(
            {
                "config_version": "order9_r1_teacher_overrides_v1",
                "overrides": {"bucket": {"selected_surface_port_ids": [2, 2]}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(early.SchemaValidationError, match="ports"):
        early._load_teacher_surface_overrides(source)


def test_r1_early_tilt_pipeline_tries_exact_contact_option_first(
    monkeypatch,
) -> None:
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return "exact-option-nominal"

    monkeypatch.setattr(
        early,
        "generate_order9_c3_nominal_grasp_trajectory",
        fake_generate,
    )

    result = early._generate_with_safe_surface_fallback(
        task_spec=object(),
        structural_target=object(),
        physical_model=object(),
        preferred_surface_port_id_options=((21, 29),),
        preferred_candidate_options=(
            ((19, 22), "slot_0:grasp_pair:0", 0.12, 0.007, 0.03),
        ),
        maximum_body_tilt_rad=1.0,
    )

    assert result == "exact-option-nominal"
    assert calls == [
        {
            "task_spec": calls[0]["task_spec"],
            "structural_target": calls[0]["structural_target"],
            "physical_model": calls[0]["physical_model"],
            "maximum_contact_solution_body_tilt_rad": 1.0,
            "preferred_surface_port_ids": (19, 22),
            "preferred_candidate_group_id": "slot_0:grasp_pair:0",
            "pregrasp_clearance_m": 0.12,
            "collision_margin_m": 0.007,
            "grasp_contact_height_offset_m": 0.03,
        }
    ]


def test_r1_early_tilt_pipeline_strict_contact_does_not_fallback(
    monkeypatch,
) -> None:
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        raise early.SchemaValidationError("bound option is unsuitable")

    monkeypatch.setattr(
        early,
        "generate_order9_c3_nominal_grasp_trajectory",
        fake_generate,
    )

    with pytest.raises(
        early.SchemaValidationError,
        match="bound preferred contact options are infeasible",
    ):
        early._generate_with_safe_surface_fallback(
            task_spec=object(),
            structural_target=object(),
            physical_model=object(),
            preferred_surface_port_id_options=((21, 29), (19, 22)),
            preferred_candidate_options=(
                ((19, 22), "slot_0:grasp_pair:0", 0.12, 0.007),
            ),
            maximum_body_tilt_rad=1.0,
            strict_preferred_candidate_options=True,
        )

    assert len(calls) == 1
