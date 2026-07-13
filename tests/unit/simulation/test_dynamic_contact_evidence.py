from __future__ import annotations

from amsrr.simulation.dynamic_contact_evidence import (
    classify_raw_contact_patches,
    evaluate_final_seated_alignment,
    evaluate_final_seated_contact,
    evaluate_funnel_guidance_contact,
)


def test_selected_pair_is_exempt_but_penetrating_other_pair_is_unintended() -> None:
    evidence = classify_raw_contact_patches(
        contact_counts=[1, 1],
        start_indices=[0, 1],
        patch_forces_n=[4.0, 0.0, 0.0, 0.0],
        patch_separations_m=[-0.001, -0.0002, 1.0, 1.0],
        raw_capacity=4,
        force_threshold_n=0.001,
        selected_pair_index=0,
        patch_points_world=[(0.0, 0.0, 0.0)] * 4,
        patch_normals_world=[(1.0, 0.0, 0.0)] * 4,
    )

    assert evidence["valid"] is True
    assert evidence["selected_physical_contact"] is True
    assert evidence["unintended_physical_contact"] is True
    assert evidence["unintended_max_penetration_m"] == 0.0002


def test_unload_scope_treats_every_raw_patch_as_external_contact() -> None:
    evidence = classify_raw_contact_patches(
        contact_counts=[1],
        start_indices=[0],
        patch_forces_n=[0.02, 0.0],
        patch_separations_m=[0.001, 1.0],
        raw_capacity=2,
        force_threshold_n=0.01,
    )

    assert evidence["selected_raw_contact_count"] == 0
    assert evidence["monitored_physical_contact"] is True
    assert evidence["monitored_max_patch_force_n"] == 0.02


def test_saturation_and_nonfinite_data_fail_closed() -> None:
    saturated = classify_raw_contact_patches(
        contact_counts=[2],
        start_indices=[0],
        patch_forces_n=[0.0, 0.0],
        patch_separations_m=[1.0, 1.0],
        raw_capacity=2,
        force_threshold_n=0.01,
    )
    nonfinite = classify_raw_contact_patches(
        contact_counts=[1],
        start_indices=[0],
        patch_forces_n=[float("nan"), 0.0],
        patch_separations_m=[1.0, 1.0],
        raw_capacity=2,
        force_threshold_n=0.01,
    )

    assert saturated["valid"] is False
    assert saturated["raw_contact_saturated"] is True
    assert nonfinite["valid"] is False
    assert nonfinite["nonfinite"] is True


def _selected_contact_measurement() -> dict[str, object]:
    return {
        "selected_contact": True,
        "selected_raw_contact_count": 2,
        "selected_force_n": 3.443,
        "selected_penetration_m": 0.0004,
        "raw_contact_valid": True,
        "raw_contact_saturated": False,
        "raw_contact_nonfinite": False,
        "raw_contact_layout_invalid": False,
        "unintended_contact": False,
    }


def test_funnel_guidance_allows_contact_before_final_connect_frame_gate() -> None:
    evidence = evaluate_funnel_guidance_contact(
        _selected_contact_measurement(),
        axial_gap_m=0.046536,
        transverse_error_m=0.004458,
        attitude_error_rad=0.02,
        min_axial_gap_m=-0.003,
        max_axial_gap_m=0.060,
        max_transverse_error_m=0.010,
        max_attitude_error_rad=0.0524,
        max_force_n=30.0,
        max_penetration_m=0.002,
    )

    assert evidence["guidance_contact_valid"] is True
    assert evidence["guidance_axial_gap_m"] == 0.046536


def test_funnel_guidance_fails_closed_on_raw_or_safety_violation() -> None:
    measurement = _selected_contact_measurement()
    measurement["raw_contact_saturated"] = True
    measurement["selected_force_n"] = 31.0

    evidence = evaluate_funnel_guidance_contact(
        measurement,
        axial_gap_m=0.046,
        transverse_error_m=0.001,
        attitude_error_rad=0.01,
        min_axial_gap_m=-0.003,
        max_axial_gap_m=0.060,
        max_transverse_error_m=0.010,
        max_attitude_error_rad=0.0524,
        max_force_n=30.0,
        max_penetration_m=0.002,
    )

    assert evidence["guidance_contact_valid"] is False
    assert "raw_contact_not_saturated" in evidence["guidance_failure_reasons"]
    assert "force_safe" in evidence["guidance_failure_reasons"]


def test_final_seated_gate_is_distinct_and_requires_continuous_dwell() -> None:
    guidance = evaluate_funnel_guidance_contact(
        _selected_contact_measurement(),
        axial_gap_m=0.001,
        transverse_error_m=0.001,
        attitude_error_rad=0.004,
        min_axial_gap_m=-0.003,
        max_axial_gap_m=0.060,
        max_transverse_error_m=0.010,
        max_attitude_error_rad=0.0524,
        max_force_n=30.0,
        max_penetration_m=0.002,
    )
    too_short = evaluate_final_seated_contact(
        guidance,
        axial_error_m=0.001,
        transverse_error_m=0.001,
        position_error_m=0.0015,
        attitude_error_rad=0.004,
        relative_linear_speed_mps=0.005,
        relative_angular_speed_radps=0.01,
        continuous_strict_dwell_s=0.099,
        required_strict_dwell_s=0.10,
        max_axial_error_m=0.003,
        max_transverse_error_m=0.002,
        max_position_error_m=0.003,
        max_attitude_error_rad=0.00873,
        max_relative_linear_speed_mps=0.01,
        max_relative_angular_speed_radps=0.03,
        both_component_qps_feasible=True,
    )
    seated = evaluate_final_seated_contact(
        guidance,
        axial_error_m=0.001,
        transverse_error_m=0.001,
        position_error_m=0.0015,
        attitude_error_rad=0.004,
        relative_linear_speed_mps=0.005,
        relative_angular_speed_radps=0.01,
        continuous_strict_dwell_s=0.10,
        required_strict_dwell_s=0.10,
        max_axial_error_m=0.003,
        max_transverse_error_m=0.002,
        max_position_error_m=0.003,
        max_attitude_error_rad=0.00873,
        max_relative_linear_speed_mps=0.01,
        max_relative_angular_speed_radps=0.03,
        both_component_qps_feasible=True,
    )

    assert too_short["final_seated_valid"] is False
    assert "continuous_strict_dwell" in too_short["final_seated_failure_reasons"]
    assert seated["final_seated_valid"] is True


def test_contactless_final_alignment_keeps_fallback_evidence_distinct() -> None:
    kwargs = {
        "axial_error_m": 0.001,
        "transverse_error_m": 0.001,
        "position_error_m": 0.0015,
        "attitude_error_rad": 0.004,
        "relative_linear_speed_mps": 0.005,
        "relative_angular_speed_radps": 0.01,
        "continuous_strict_dwell_s": 0.10,
        "required_strict_dwell_s": 0.10,
        "max_axial_error_m": 0.003,
        "max_transverse_error_m": 0.002,
        "max_position_error_m": 0.003,
        "max_attitude_error_rad": 0.00873,
        "max_relative_linear_speed_mps": 0.01,
        "max_relative_angular_speed_radps": 0.03,
        "both_component_qps_feasible": True,
    }

    seated = evaluate_final_seated_alignment(**kwargs)
    too_short = evaluate_final_seated_alignment(
        **{**kwargs, "continuous_strict_dwell_s": 0.099}
    )
    too_fast = evaluate_final_seated_alignment(
        **{**kwargs, "relative_linear_speed_mps": 0.011}
    )

    assert seated["evidence_version"] == "final_seated_alignment_v1"
    assert seated["selected_pair_contact_required"] is False
    assert seated["selected_pair_contact_observed"] is False
    assert seated["final_seated_valid"] is True
    assert "selected_contact_points_world" not in seated
    assert too_short["final_seated_valid"] is False
    assert too_fast["final_seated_valid"] is False
