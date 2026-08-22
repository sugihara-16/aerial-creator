from __future__ import annotations

import numpy as np
import pytest

from amsrr.training.order9_actuator_aware_nominal_preload import (
    Order9ActuatorAwareNominalPreloadConfig,
    solve_order9_actuator_aware_nominal_preload_from_jacobian,
)


def test_preload_resolver_accounts_for_joint_and_contact_compliance() -> None:
    result = solve_order9_actuator_aware_nominal_preload_from_jacobian(
        inward_normal_joint_jacobian_m=np.asarray(
            [[1.0, 0.0], [0.0, 1.0]], dtype=float
        ),
        joint_ids=("left", "right"),
        joint_stiffness_nm_per_rad=(200.0, 200.0),
        joint_peak_effort_limit_nm=(4.1, 4.1),
        joint_continuous_effort_limit_nm=(1.3, 1.3),
        object_mass_kg=1.0,
        contact_friction=4.5,
        contact_stiffness_n_per_m=8000.0,
    )

    # 1 kg * g * 1.25, shared by two mu=4.5 contacts.
    assert result.target_normal_force_n_by_anchor == pytest.approx(
        (1.3620347, 1.3620347), rel=1.0e-5
    )
    assert result.inward_lead_m == pytest.approx(0.007)
    assert result.feasible
    assert result.peak_effort_utilization < 0.8
    assert result.continuous_effort_utilization > 1.0


def test_preload_resolver_adds_common_calibrated_model_error_margin() -> None:
    result = solve_order9_actuator_aware_nominal_preload_from_jacobian(
        inward_normal_joint_jacobian_m=np.asarray(
            [[1.0, 0.0], [0.0, 1.0]], dtype=float
        ),
        joint_ids=("left", "right"),
        joint_stiffness_nm_per_rad=(200.0, 200.0),
        joint_peak_effort_limit_nm=(4.1, 4.1),
        joint_continuous_effort_limit_nm=(1.3, 1.3),
        object_mass_kg=1.0,
        contact_friction=4.5,
        contact_stiffness_n_per_m=8000.0,
        config=Order9ActuatorAwareNominalPreloadConfig(
            model_error_margin_m=0.007
        ),
    )

    assert result.compliance_predicted_inward_lead_m == pytest.approx(
        0.0069804, rel=1.0e-4
    )
    assert result.model_error_margin_m == pytest.approx(0.007)
    assert result.inward_lead_m == pytest.approx(0.014)
    assert result.feasible


def test_preload_resolver_rejects_peak_effort_infeasibility() -> None:
    result = solve_order9_actuator_aware_nominal_preload_from_jacobian(
        inward_normal_joint_jacobian_m=np.asarray(
            [[2.0], [2.0]], dtype=float
        ),
        joint_ids=("shared",),
        joint_stiffness_nm_per_rad=(200.0,),
        joint_peak_effort_limit_nm=(4.1,),
        joint_continuous_effort_limit_nm=(1.3,),
        object_mass_kg=1.0,
        contact_friction=1.0,
        contact_stiffness_n_per_m=8000.0,
    )

    assert not result.feasible
    assert result.rejection_reason == "peak_effort_utilization"


def test_preload_resolver_rejects_excessive_servo_lead() -> None:
    result = solve_order9_actuator_aware_nominal_preload_from_jacobian(
        inward_normal_joint_jacobian_m=np.asarray(
            [[1.8, 0.0], [0.0, 1.8]], dtype=float
        ),
        joint_ids=("left", "right"),
        joint_stiffness_nm_per_rad=(50.0, 50.0),
        joint_peak_effort_limit_nm=(20.0, 20.0),
        joint_continuous_effort_limit_nm=(10.0, 10.0),
        object_mass_kg=1.0,
        contact_friction=4.5,
        contact_stiffness_n_per_m=8000.0,
        config=Order9ActuatorAwareNominalPreloadConfig(
            maximum_inward_lead_m=0.040
        ),
    )

    assert not result.feasible
    assert result.rejection_reason == "nominal_inward_lead"
    assert result.inward_lead_m == pytest.approx(0.040)
