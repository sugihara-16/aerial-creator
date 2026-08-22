from __future__ import annotations

import pytest

from amsrr.training.order9_virtual_contact_compression import (
    ORDER9_CONTACT_COMPRESSION_MAXIMUM_ACTION_JOINT_DELTA_RAD,
    ORDER9_VIRTUAL_CONTACT_COMPRESSION_VERSION,
    solve_order9_collision_safe_compression_scale,
)


def test_collision_safe_compression_scale_accepts_full_direction() -> None:
    result = solve_order9_collision_safe_compression_scale(
        lambda scale: (True, None, None)
    )

    assert result.maximum_action_scale == 1.0
    assert result.collision_check_count == 1
    assert result.limiting_scene_id is None


def test_collision_safe_compression_scale_bisects_limiting_scene() -> None:
    threshold = 0.625

    def evaluator(scale: float):
        if scale <= threshold:
            return True, None, None
        return False, "support", {"accepted": False, "scale": scale}

    result = solve_order9_collision_safe_compression_scale(
        evaluator, bisection_iterations=14
    )

    assert result.maximum_action_scale == pytest.approx(threshold, abs=1.0e-4)
    assert result.limiting_scene_id == "support"
    assert result.collision_check_count == 16


def test_collision_safe_compression_scale_respects_joint_trust_region() -> None:
    result = solve_order9_collision_safe_compression_scale(
        lambda scale: (True, None, None), maximum_scale=0.625
    )

    assert result.maximum_action_scale == 0.625
    assert result.limiting_scene_id == "joint_trust_region"
    assert result.maximum_action_joint_delta_rad == (
        ORDER9_CONTACT_COMPRESSION_MAXIMUM_ACTION_JOINT_DELTA_RAD
    )


def test_collision_safe_compression_scale_rejects_invalid_nominal() -> None:
    with pytest.raises(ValueError, match="accepted nominal posture"):
        solve_order9_collision_safe_compression_scale(
            lambda scale: (False, "support", {"accepted": False})
        )


def test_virtual_contact_solver_version_requires_monotone_continuation() -> None:
    assert "bounded_refinement" in ORDER9_VIRTUAL_CONTACT_COMPRESSION_VERSION
