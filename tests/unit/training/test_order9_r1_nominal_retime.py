from __future__ import annotations

from dataclasses import replace

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.policies import (
    CentroidalTarget,
    ContactWrenchTrajectory,
    InteractionKnot,
    PostureTarget,
)
from amsrr.training.order9_r1_nominal_retime import (
    Order9R1NominalRetimeAdmission,
    _configuration_endpoint,
    _retime_trajectory,
    _semantically_close,
)


def test_r1_retime_preserves_path_endpoint_and_scales_velocity() -> None:
    source = _trajectory()

    result = _retime_trajectory(source, scale=0.5)

    assert result.horizon_s == pytest.approx(1.0)
    assert [value.t_rel_s for value in result.knots] == pytest.approx([0.0, 0.5, 1.0])
    assert [
        value.posture_target.joint_pos_target["module_0:q"]
        for value in result.knots
    ] == pytest.approx([0.0, 0.5, 1.0])
    assert all(
        value.posture_target.joint_vel_target["module_0:q"]
        == pytest.approx(1.0)
        for value in result.knots
    )
    assert _semantically_close(
        _configuration_endpoint(source),
        _configuration_endpoint(result),
    )


def test_r1_retime_admission_fails_closed_on_negative_rate_margin() -> None:
    value = Order9R1NominalRetimeAdmission(
        admission_version="order9_r1_uniform_nominal_retime_v1",
        time_scale=0.1,
        phase_time_scales={
            "approach": 0.5,
            "contact_acquisition": 0.5,
            "lift": 0.1,
            "transport": 0.1,
            "place": 0.1,
            "release": 0.1,
            "retreat": 0.1,
            "settle": 0.1,
        },
        source_artifact_sha256="a" * 64,
        retimed_artifact_sha256="b" * 64,
        source_duration_s=252.0,
        retimed_duration_s=54.0,
        joint_rate_limit_rad_s=2.7,
        maximum_commanded_joint_rate_rad_s=2.6,
        minimum_joint_rate_margin_rad_s=0.1,
        maximum_centroidal_speed_m_s=0.2,
        maximum_object_linear_speed_m_s=0.1,
        phase_order_unchanged=True,
        geometric_path_unchanged=True,
        endpoint_configuration_unchanged=True,
        contact_assignments_unchanged=True,
        collision_admission_inherited_from_identical_geometry=True,
        accepted=True,
    )
    value.validate()

    with pytest.raises(SchemaValidationError, match="joint-rate"):
        replace(value, minimum_joint_rate_margin_rad_s=-0.01).validate()


def _trajectory() -> ContactWrenchTrajectory:
    knots = []
    for time_s, position in ((0.0, 0.0), (1.0, 0.5), (2.0, 1.0)):
        knots.append(
            InteractionKnot(
                t_rel_s=time_s,
                contact_assignments=[],
                centroidal_target=CentroidalTarget(
                    com_pos_world=(position, 0.0, 0.5),
                    com_vel_world=(0.5, 0.0, 0.0),
                    body_orientation_world=(0.0, 0.0, 0.0, 1.0),
                ),
                posture_target=PostureTarget(
                    joint_pos_target={"module_0:q": position},
                    joint_vel_target={"module_0:q": 0.5},
                ),
            )
        )
    return ContactWrenchTrajectory(horizon_s=2.0, dt_s=0.5, knots=knots)
