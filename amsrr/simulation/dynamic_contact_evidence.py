from __future__ import annotations

import math
from collections.abc import Sequence


FUNNEL_GUIDANCE_CONTACT_EVIDENCE_VERSION = "funnel_guidance_contact_v1"
FINAL_SEATED_CONTACT_EVIDENCE_VERSION = "final_seated_contact_v1"
FINAL_SEATED_ALIGNMENT_EVIDENCE_VERSION = "final_seated_alignment_v1"


def classify_raw_contact_patches(
    *,
    contact_counts: Sequence[int],
    start_indices: Sequence[int],
    patch_forces_n: Sequence[float],
    patch_separations_m: Sequence[float],
    raw_capacity: int,
    force_threshold_n: float,
    selected_pair_index: int | None = None,
    patch_points_world: Sequence[Sequence[float]] | None = None,
    patch_normals_world: Sequence[Sequence[float]] | None = None,
) -> dict[str, object]:
    """Classify non-aggregated PhysX contact patches without force cancellation.

    ``selected_pair_index`` identifies the one intentional mating body pair.
    With ``None``, every physical patch is treated as monitored external
    contact (used by the follower-subtree unload gate).  Invalid indices,
    buffer saturation, and non-finite active data are returned as fail-closed
    evidence instead of being silently discarded.
    """

    counts = [int(value) for value in contact_counts]
    starts = [int(value) for value in start_indices]
    capacity = int(raw_capacity)
    threshold = float(force_threshold_n)
    invalid_layout = bool(
        capacity <= 0
        or not math.isfinite(threshold)
        or threshold < 0.0
        or len(counts) != len(starts)
        or len(patch_forces_n) < max(capacity, 0)
        or len(patch_separations_m) < max(capacity, 0)
        or (
            patch_points_world is not None
            and len(patch_points_world) < max(capacity, 0)
        )
        or (
            patch_normals_world is not None
            and len(patch_normals_world) < max(capacity, 0)
        )
        or (
            selected_pair_index is not None
            and not 0 <= int(selected_pair_index) < len(counts)
        )
    )
    effective_capacity = max(
        0,
        min(
            capacity,
            len(patch_forces_n),
            len(patch_separations_m),
            (
                len(patch_points_world)
                if patch_points_world is not None
                else capacity
            ),
            (
                len(patch_normals_world)
                if patch_normals_world is not None
                else capacity
            ),
        ),
    )
    active_by_pair: list[list[int]] = [[] for _ in counts]
    out_of_bounds = False
    total_count = 0
    for pair_index, (start, count) in enumerate(zip(starts, counts)):
        if start < 0 or count < 0:
            out_of_bounds = True
            continue
        total_count += count
        stop = start + count
        if stop > effective_capacity:
            out_of_bounds = True
            stop = min(stop, effective_capacity)
        if start < effective_capacity:
            active_by_pair[pair_index] = list(range(start, max(start, stop)))

    saturated = bool(
        capacity <= 0
        or total_count >= capacity
        or out_of_bounds
    )
    nonfinite = False
    selected_indices: list[int] = []
    monitored_indices: list[int] = []
    physical_monitored_indices: list[int] = []
    physical_unintended_indices: list[int] = []
    max_monitored_force = 0.0
    max_unintended_force = 0.0
    max_unintended_penetration = 0.0
    min_monitored_separation: float | None = None

    for pair_index, indices in enumerate(active_by_pair):
        is_selected = selected_pair_index is not None and pair_index == selected_pair_index
        if is_selected:
            selected_indices.extend(indices)
        else:
            monitored_indices.extend(indices)
        for patch_index in indices:
            force = float(patch_forces_n[patch_index])
            separation = float(patch_separations_m[patch_index])
            vectors = []
            if patch_points_world is not None:
                vectors.append(patch_points_world[patch_index])
            if patch_normals_world is not None:
                vectors.append(patch_normals_world[patch_index])
            finite = bool(
                math.isfinite(force)
                and math.isfinite(separation)
                and all(
                    len(vector) == 3
                    and all(math.isfinite(float(value)) for value in vector)
                    for vector in vectors
                )
            )
            if not finite:
                nonfinite = True
                continue
            physical = abs(force) > threshold or separation <= 0.0
            if not is_selected:
                max_monitored_force = max(max_monitored_force, abs(force))
                min_monitored_separation = (
                    separation
                    if min_monitored_separation is None
                    else min(min_monitored_separation, separation)
                )
                if physical:
                    physical_monitored_indices.append(patch_index)
                    if selected_pair_index is not None:
                        physical_unintended_indices.append(patch_index)
                        max_unintended_force = max(max_unintended_force, abs(force))
                        max_unintended_penetration = max(
                            max_unintended_penetration,
                            max(0.0, -separation),
                        )

    selected_forces = [
        abs(float(patch_forces_n[index]))
        for index in selected_indices
        if math.isfinite(float(patch_forces_n[index]))
    ]
    selected_separations = [
        float(patch_separations_m[index])
        for index in selected_indices
        if math.isfinite(float(patch_separations_m[index]))
    ]
    selected_physical = any(
        abs(float(patch_forces_n[index])) > threshold
        or float(patch_separations_m[index]) <= 0.0
        for index in selected_indices
        if math.isfinite(float(patch_forces_n[index]))
        and math.isfinite(float(patch_separations_m[index]))
    )
    valid = not (invalid_layout or saturated or nonfinite)
    return {
        "valid": valid,
        "invalid_layout": invalid_layout,
        "nonfinite": nonfinite,
        "out_of_bounds": out_of_bounds,
        "raw_contact_saturated": saturated,
        "raw_contact_count": total_count,
        "raw_contact_capacity": capacity,
        "selected_raw_contact_count": len(selected_indices),
        "selected_patch_indices": selected_indices,
        "selected_physical_contact": selected_physical,
        "selected_max_patch_force_n": max(selected_forces, default=0.0),
        "selected_min_separation_m": min(selected_separations, default=0.0),
        "monitored_raw_contact_count": len(monitored_indices),
        "monitored_physical_contact_count": len(physical_monitored_indices),
        "monitored_physical_contact": bool(physical_monitored_indices),
        "monitored_max_patch_force_n": max_monitored_force,
        "monitored_min_separation_m": (
            0.0 if min_monitored_separation is None else min_monitored_separation
        ),
        "unintended_physical_contact_count": len(physical_unintended_indices),
        "unintended_physical_contact": bool(physical_unintended_indices),
        "unintended_max_patch_force_n": max_unintended_force,
        "unintended_max_penetration_m": max_unintended_penetration,
    }


def evaluate_funnel_guidance_contact(
    measurement: dict[str, object],
    *,
    axial_gap_m: float,
    transverse_error_m: float,
    attitude_error_rad: float,
    min_axial_gap_m: float,
    max_axial_gap_m: float,
    max_transverse_error_m: float,
    max_attitude_error_rad: float,
    max_force_n: float,
    max_penetration_m: float,
) -> dict[str, object]:
    """Validate a selected Dock-body contact as bounded funnel guidance.

    The pitch-side funnel is expected to contact the yaw-side insert before
    the final connect frames coincide.  This gate therefore validates the raw
    exact-body-pair evidence and a coarse approach envelope; it deliberately
    does *not* require contact points to lie on either final connect plane.
    """

    numeric = {
        "guidance_axial_gap_m": float(axial_gap_m),
        "guidance_transverse_error_m": float(transverse_error_m),
        "guidance_attitude_error_rad": float(attitude_error_rad),
        "selected_force_n": float(measurement.get("selected_force_n", math.inf)),
        "selected_penetration_m": float(
            measurement.get("selected_penetration_m", math.inf)
        ),
    }
    finite = all(math.isfinite(value) for value in numeric.values())
    selected_raw_contact_count = int(
        measurement.get("selected_raw_contact_count", 0)
    )
    failures: list[str] = []
    checks = {
        "selected_contact": measurement.get("selected_contact") is True,
        "selected_raw_contact_present": selected_raw_contact_count > 0,
        "raw_contact_valid": measurement.get("raw_contact_valid") is True,
        "raw_contact_not_saturated": (
            measurement.get("raw_contact_saturated") is False
        ),
        "raw_contact_finite": measurement.get("raw_contact_nonfinite") is False,
        "raw_contact_layout_valid": (
            measurement.get("raw_contact_layout_invalid") is False
        ),
        "no_unintended_contact": measurement.get("unintended_contact") is False,
        "finite_metrics": finite,
        "axial_envelope": (
            finite
            and float(min_axial_gap_m)
            <= numeric["guidance_axial_gap_m"]
            <= float(max_axial_gap_m)
        ),
        "transverse_envelope": (
            finite
            and numeric["guidance_transverse_error_m"]
            <= float(max_transverse_error_m)
        ),
        "attitude_envelope": (
            finite
            and numeric["guidance_attitude_error_rad"]
            <= float(max_attitude_error_rad)
        ),
        "force_safe": (
            finite and numeric["selected_force_n"] <= float(max_force_n)
        ),
        "penetration_safe": (
            finite
            and numeric["selected_penetration_m"] <= float(max_penetration_m)
        ),
    }
    failures.extend(name for name, passed in checks.items() if not passed)
    return {
        "evidence_version": FUNNEL_GUIDANCE_CONTACT_EVIDENCE_VERSION,
        "selected_pair_scope": "selected_dock_body_pair",
        "selected_pair_exact_body_match": True,
        "guidance_contact_valid": not failures,
        "guidance_failure_reasons": failures,
        "selected_raw_contact_count": selected_raw_contact_count,
        **numeric,
        "guidance_min_axial_gap_m": float(min_axial_gap_m),
        "guidance_contact_max_axial_gap_m": float(max_axial_gap_m),
        "guidance_contact_max_transverse_error_m": float(
            max_transverse_error_m
        ),
        "guidance_contact_max_attitude_error_rad": float(
            max_attitude_error_rad
        ),
        "guidance_max_force_n": float(max_force_n),
        "guidance_max_penetration_m": float(max_penetration_m),
    }


def evaluate_final_seated_contact(
    guidance_evidence: dict[str, object],
    *,
    axial_error_m: float,
    transverse_error_m: float,
    position_error_m: float,
    attitude_error_rad: float,
    relative_linear_speed_mps: float,
    relative_angular_speed_radps: float,
    continuous_strict_dwell_s: float,
    required_strict_dwell_s: float,
    max_axial_error_m: float,
    max_transverse_error_m: float,
    max_position_error_m: float,
    max_attitude_error_rad: float,
    max_relative_linear_speed_mps: float,
    max_relative_angular_speed_radps: float,
    both_component_qps_feasible: bool,
) -> dict[str, object]:
    """Validate final seated state independently from first funnel contact."""

    numeric = {
        "axial_error_m": float(axial_error_m),
        "transverse_error_m": float(transverse_error_m),
        "position_error_m": float(position_error_m),
        "attitude_error_rad": float(attitude_error_rad),
        "relative_linear_speed_mps": float(relative_linear_speed_mps),
        "relative_angular_speed_radps": float(relative_angular_speed_radps),
        "continuous_strict_dwell_s": float(continuous_strict_dwell_s),
        "required_strict_dwell_s": float(required_strict_dwell_s),
    }
    finite = all(math.isfinite(value) for value in numeric.values())
    checks = {
        "guidance_contact_valid": (
            guidance_evidence.get("guidance_contact_valid") is True
        ),
        "both_component_qps_feasible": bool(both_component_qps_feasible),
        "finite_metrics": finite,
        "axial_strict": finite
        and numeric["axial_error_m"] <= float(max_axial_error_m),
        "transverse_strict": finite
        and numeric["transverse_error_m"] <= float(max_transverse_error_m),
        "position_strict": finite
        and numeric["position_error_m"] <= float(max_position_error_m),
        "attitude_strict": finite
        and numeric["attitude_error_rad"] <= float(max_attitude_error_rad),
        "relative_linear_speed_strict": finite
        and numeric["relative_linear_speed_mps"]
        <= float(max_relative_linear_speed_mps),
        "relative_angular_speed_strict": finite
        and numeric["relative_angular_speed_radps"]
        <= float(max_relative_angular_speed_radps),
        "continuous_strict_dwell": finite
        and numeric["continuous_strict_dwell_s"]
        >= numeric["required_strict_dwell_s"],
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "evidence_version": FINAL_SEATED_CONTACT_EVIDENCE_VERSION,
        "selected_pair_scope": "selected_dock_body_pair",
        "selected_pair_exact_body_match": True,
        "selected_pair_contact_required": True,
        "selected_pair_contact_observed": True,
        "final_seated_valid": not failures,
        "final_seated_failure_reasons": failures,
        "both_component_qps_feasible": bool(both_component_qps_feasible),
        **numeric,
        "max_axial_error_m": float(max_axial_error_m),
        "max_transverse_error_m": float(max_transverse_error_m),
        "max_position_error_m": float(max_position_error_m),
        "max_attitude_error_rad": float(max_attitude_error_rad),
        "max_relative_linear_speed_mps": float(
            max_relative_linear_speed_mps
        ),
        "max_relative_angular_speed_radps": float(
            max_relative_angular_speed_radps
        ),
    }


def evaluate_final_seated_alignment(
    *,
    axial_error_m: float,
    transverse_error_m: float,
    position_error_m: float,
    attitude_error_rad: float,
    relative_linear_speed_mps: float,
    relative_angular_speed_radps: float,
    continuous_strict_dwell_s: float,
    required_strict_dwell_s: float,
    max_axial_error_m: float,
    max_transverse_error_m: float,
    max_position_error_m: float,
    max_attitude_error_rad: float,
    max_relative_linear_speed_mps: float,
    max_relative_angular_speed_radps: float,
    both_component_qps_feasible: bool,
) -> dict[str, object]:
    """Validate a contactless final alignment after selected-pair filtering.

    This evidence is intentionally distinct from physical funnel-contact
    evidence.  It proves only the strict connect-frame pose/twist dwell and QP
    feasibility; the separately reported filter evidence proves why selected
    Dock-body contact is absent.
    """

    numeric = {
        "axial_error_m": float(axial_error_m),
        "transverse_error_m": float(transverse_error_m),
        "position_error_m": float(position_error_m),
        "attitude_error_rad": float(attitude_error_rad),
        "relative_linear_speed_mps": float(relative_linear_speed_mps),
        "relative_angular_speed_radps": float(relative_angular_speed_radps),
        "continuous_strict_dwell_s": float(continuous_strict_dwell_s),
        "required_strict_dwell_s": float(required_strict_dwell_s),
    }
    finite = all(math.isfinite(value) for value in numeric.values())
    checks = {
        "both_component_qps_feasible": bool(both_component_qps_feasible),
        "finite_metrics": finite,
        "axial_strict": finite
        and numeric["axial_error_m"] <= float(max_axial_error_m),
        "transverse_strict": finite
        and numeric["transverse_error_m"] <= float(max_transverse_error_m),
        "position_strict": finite
        and numeric["position_error_m"] <= float(max_position_error_m),
        "attitude_strict": finite
        and numeric["attitude_error_rad"] <= float(max_attitude_error_rad),
        "relative_linear_speed_strict": finite
        and numeric["relative_linear_speed_mps"]
        <= float(max_relative_linear_speed_mps),
        "relative_angular_speed_strict": finite
        and numeric["relative_angular_speed_radps"]
        <= float(max_relative_angular_speed_radps),
        "continuous_strict_dwell": finite
        and numeric["continuous_strict_dwell_s"]
        >= numeric["required_strict_dwell_s"],
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "evidence_version": FINAL_SEATED_ALIGNMENT_EVIDENCE_VERSION,
        "selected_pair_scope": "selected_dock_body_pair",
        "selected_pair_contact_required": False,
        "selected_pair_contact_observed": False,
        "final_seated_valid": not failures,
        "final_seated_failure_reasons": failures,
        "both_component_qps_feasible": bool(both_component_qps_feasible),
        **numeric,
        "max_axial_error_m": float(max_axial_error_m),
        "max_transverse_error_m": float(max_transverse_error_m),
        "max_position_error_m": float(max_position_error_m),
        "max_attitude_error_rad": float(max_attitude_error_rad),
        "max_relative_linear_speed_mps": float(
            max_relative_linear_speed_mps
        ),
        "max_relative_angular_speed_radps": float(
            max_relative_angular_speed_radps
        ),
    }


__all__ = [
    "FINAL_SEATED_ALIGNMENT_EVIDENCE_VERSION",
    "FINAL_SEATED_CONTACT_EVIDENCE_VERSION",
    "FUNNEL_GUIDANCE_CONTACT_EVIDENCE_VERSION",
    "classify_raw_contact_patches",
    "evaluate_final_seated_alignment",
    "evaluate_final_seated_contact",
    "evaluate_funnel_guidance_contact",
]
