from __future__ import annotations

import math

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.simulation.dynamic_assembly import DynamicSeparationLifecycle


def _lifecycle(
    *,
    nominal_separation_steps: int = 4,
    max_separation_steps: int = 8,
    required_post_release_stable_steps: int = 5,
    max_post_release_steps: int = 10,
) -> DynamicSeparationLifecycle:
    return DynamicSeparationLifecycle(
        nominal_separation_steps=nominal_separation_steps,
        max_separation_steps=max_separation_steps,
        minimum_gap_m=0.16,
        minimum_clearance_m=0.03,
        required_post_release_stable_steps=(
            required_post_release_stable_steps
        ),
        max_post_release_steps=max_post_release_steps,
    )


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("nominal_separation_steps", 0),
        ("nominal_separation_steps", True),
        ("max_separation_steps", 3),
        ("max_separation_steps", 9),
        ("minimum_gap_m", 0.0),
        ("minimum_gap_m", math.inf),
        ("minimum_clearance_m", -0.01),
        ("required_post_release_stable_steps", 0),
        ("max_post_release_steps", 4),
        ("max_post_release_steps", 11),
    ],
)
def test_lifecycle_rejects_invalid_configuration(
    override: str,
    value: object,
) -> None:
    values: dict[str, object] = {
        "nominal_separation_steps": 4,
        "max_separation_steps": 8,
        "minimum_gap_m": 0.16,
        "minimum_clearance_m": 0.03,
        "required_post_release_stable_steps": 5,
        "max_post_release_steps": 10,
    }
    values[override] = value

    with pytest.raises(SchemaValidationError):
        DynamicSeparationLifecycle(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("gap_m", "clearance_m"),
    [
        (0.16 - 1.0e-12, 0.03),
        (0.16, 0.03 - 1.0e-12),
    ],
)
def test_separation_requires_nominal_duration_and_both_measured_gates(
    gap_m: float,
    clearance_m: float,
) -> None:
    lifecycle = _lifecycle()
    for _ in range(3):
        assert (
            lifecycle.observe_separation(gap_m=0.16, clearance_m=0.03)
            == "continue"
        )

    assert (
        lifecycle.observe_separation(gap_m=gap_m, clearance_m=clearance_m)
        == "continue"
    )
    assert (
        lifecycle.observe_separation(gap_m=0.16, clearance_m=0.03)
        == "request_filter_removal"
    )


def test_real_lag_trace_waits_for_clearance_after_nominal_separation() -> None:
    lifecycle = DynamicSeparationLifecycle(
        nominal_separation_steps=800,
        max_separation_steps=1600,
        minimum_gap_m=0.16,
        minimum_clearance_m=0.03,
        required_post_release_stable_steps=200,
        max_post_release_steps=400,
    )

    for _ in range(800):
        assert (
            lifecycle.observe_separation(
                gap_m=0.169685806,
                clearance_m=0.0061395,
            )
            == "continue"
        )
    for _ in range(152):
        assert (
            lifecycle.observe_separation(
                gap_m=0.199,
                clearance_m=0.029,
            )
            == "continue"
        )

    assert lifecycle.separation_steps == 952
    assert (
        lifecycle.observe_separation(gap_m=0.205467, clearance_m=0.034332)
        == "request_filter_removal"
    )
    assert lifecycle.confirm_filter_removal(verified=True) == "post_release"
    assert lifecycle.phase == "post_release"


def test_separation_times_out_when_measured_gate_never_qualifies() -> None:
    lifecycle = _lifecycle()

    for _ in range(7):
        assert (
            lifecycle.observe_separation(gap_m=0.20, clearance_m=0.029)
            == "continue"
        )
    assert (
        lifecycle.observe_separation(gap_m=0.20, clearance_m=0.029)
        == "timeout"
    )
    assert lifecycle.phase == "timed_out"
    assert lifecycle.failure_reason == "separation_timeout"


def test_filter_removal_verification_failure_blocks_post_release() -> None:
    lifecycle = _lifecycle()
    for _ in range(3):
        assert (
            lifecycle.observe_separation(gap_m=0.16, clearance_m=0.03)
            == "continue"
        )
    assert (
        lifecycle.observe_separation(gap_m=0.16, clearance_m=0.03)
        == "request_filter_removal"
    )

    assert (
        lifecycle.confirm_filter_removal(verified=False)
        == "verification_failed"
    )
    assert lifecycle.phase == "failed"
    assert lifecycle.failure_reason == "filter_removal_verification_failed"
    with pytest.raises(SchemaValidationError):
        lifecycle.observe_post_release(stable=True)


def test_post_release_completes_after_existing_47_plus_153_stable_steps() -> None:
    lifecycle = DynamicSeparationLifecycle(
        nominal_separation_steps=1,
        max_separation_steps=2,
        minimum_gap_m=0.16,
        minimum_clearance_m=0.03,
        required_post_release_stable_steps=200,
        max_post_release_steps=400,
    )
    assert (
        lifecycle.observe_separation(gap_m=0.16, clearance_m=0.03)
        == "request_filter_removal"
    )
    assert lifecycle.confirm_filter_removal(verified=True) == "post_release"

    for _ in range(47):
        assert lifecycle.observe_post_release(stable=True) == "continue"
    for _ in range(152):
        assert lifecycle.observe_post_release(stable=True) == "continue"
    assert lifecycle.post_release_stable_steps == 199
    assert lifecycle.observe_post_release(stable=True) == "complete"
    assert lifecycle.post_release_steps == 200
    assert lifecycle.post_release_stable_steps == 200


def test_post_release_stable_dwell_is_continuous_and_resettable() -> None:
    lifecycle = _lifecycle(
        nominal_separation_steps=1,
        max_separation_steps=2,
        required_post_release_stable_steps=3,
        max_post_release_steps=6,
    )
    assert (
        lifecycle.observe_separation(gap_m=0.16, clearance_m=0.03)
        == "request_filter_removal"
    )
    assert lifecycle.confirm_filter_removal(verified=True) == "post_release"

    assert lifecycle.observe_post_release(stable=True) == "continue"
    assert lifecycle.observe_post_release(stable=True) == "continue"
    assert lifecycle.post_release_stable_steps == 2
    assert lifecycle.observe_post_release(stable=False) == "continue"
    assert lifecycle.post_release_stable_steps == 0
    assert lifecycle.observe_post_release(stable=True) == "continue"
    assert lifecycle.observe_post_release(stable=True) == "continue"
    assert lifecycle.observe_post_release(stable=True) == "complete"


def test_post_release_times_out_when_continuous_dwell_never_qualifies() -> None:
    lifecycle = _lifecycle(
        nominal_separation_steps=1,
        max_separation_steps=2,
        required_post_release_stable_steps=3,
        max_post_release_steps=6,
    )
    assert (
        lifecycle.observe_separation(gap_m=0.16, clearance_m=0.03)
        == "request_filter_removal"
    )
    assert lifecycle.confirm_filter_removal(verified=True) == "post_release"

    for stable in (True, True, False, True, True):
        assert lifecycle.observe_post_release(stable=stable) == "continue"
    assert lifecycle.observe_post_release(stable=False) == "timeout"
    assert lifecycle.phase == "timed_out"
    assert lifecycle.failure_reason == "post_release_timeout"


def test_gate_success_wins_on_exact_final_budget_steps() -> None:
    lifecycle = _lifecycle(
        nominal_separation_steps=2,
        max_separation_steps=4,
        required_post_release_stable_steps=3,
        max_post_release_steps=6,
    )
    for _ in range(3):
        assert (
            lifecycle.observe_separation(gap_m=0.20, clearance_m=0.029)
            == "continue"
        )
    assert (
        lifecycle.observe_separation(gap_m=0.20, clearance_m=0.03)
        == "request_filter_removal"
    )
    assert lifecycle.confirm_filter_removal(verified=True) == "post_release"

    for _ in range(3):
        assert lifecycle.observe_post_release(stable=False) == "continue"
    for _ in range(2):
        assert lifecycle.observe_post_release(stable=True) == "continue"
    assert lifecycle.observe_post_release(stable=True) == "complete"
    assert lifecycle.post_release_steps == lifecycle.max_post_release_steps
