from __future__ import annotations

import pytest

from amsrr.feasibility.contact_wrench_shadow_metrics import (
    MeasuredCandidateWrench,
    evaluate_shadow_contact_wrench_residual,
)
from tests.unit.feasibility.test_contact_wrench_hybrid import (
    _context,
    _trajectory,
)


def test_measured_wrenches_inside_ranges_support_payload() -> None:
    trajectory = _trajectory(tangential_half_width_n=12.0)
    metric = evaluate_shadow_contact_wrench_residual(
        _context(),
        trajectory.knots[0],
        [
            MeasuredCandidateWrench(
                candidate_id=0,
                wrench_contact=(-5.0, 0.0, 5.0, 0.0, 0.0, 0.0),
            ),
            MeasuredCandidateWrench(
                candidate_id=1,
                wrench_contact=(5.0, 0.0, 5.0, 0.0, 0.0, 0.0),
            ),
        ],
    )

    assert metric.residual == pytest.approx(0.0)
    assert metric.margins["measured_net_force_z_n"] == pytest.approx(10.0)
    assert metric.margins["active_numeric_wrench_requirement_count"] == 1.0


def test_measured_wrench_residual_reports_payload_deficit_dimensionlessly() -> None:
    trajectory = _trajectory(tangential_half_width_n=12.0)
    metric = evaluate_shadow_contact_wrench_residual(
        _context(),
        trajectory.knots[0],
        [
            MeasuredCandidateWrench(
                candidate_id=0,
                wrench_contact=(-5.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            ),
            MeasuredCandidateWrench(
                candidate_id=1,
                wrench_contact=(5.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            ),
        ],
    )

    assert metric.residual == pytest.approx(9.80665 / 30.0)


def test_missing_active_contact_measurement_fails_closed() -> None:
    trajectory = _trajectory(tangential_half_width_n=12.0)
    metric = evaluate_shadow_contact_wrench_residual(
        _context(),
        trajectory.knots[0],
        [
            MeasuredCandidateWrench(
                candidate_id=0,
                wrench_contact=(-5.0, 0.0, 5.0, 0.0, 0.0, 0.0),
            )
        ],
    )

    assert metric.residual == 1.0
    assert metric.margins["shadow_wrench_metric_failed"] == 1.0


def test_measured_force_outside_policy_range_is_rejected() -> None:
    trajectory = _trajectory(tangential_half_width_n=12.0)
    metric = evaluate_shadow_contact_wrench_residual(
        _context(),
        trajectory.knots[0],
        [
            MeasuredCandidateWrench(
                candidate_id=0,
                wrench_contact=(-8.5, 0.0, 5.0, 0.0, 0.0, 0.0),
            ),
            MeasuredCandidateWrench(
                candidate_id=1,
                wrench_contact=(8.5, 0.0, 5.0, 0.0, 0.0, 0.0),
            ),
        ],
    )

    assert metric.residual == pytest.approx(3.0 / 30.0)
