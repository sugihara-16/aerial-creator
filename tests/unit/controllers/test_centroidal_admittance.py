from __future__ import annotations

from dataclasses import replace

import pytest

from amsrr.controllers.centroidal_admittance import (
    CentroidalAdmittanceConfig,
    CentroidalAdmittanceController,
    CentroidalExternalWrenchEstimate,
    CentroidalExternalWrenchEstimator,
    CentroidalExternalWrenchEstimatorConfig,
)
from amsrr.controllers.rigid_body_model import (
    RigidBodyControlModel,
    RotorControlElement,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.policies import ControllerCommand, ControllerStatus


def _model(*, linear_velocity_x_mps: float) -> RigidBodyControlModel:
    rotor = RotorControlElement(
        global_rotor_id="module_0:thrust_1",
        module_id=0,
        rotor_id="thrust_1",
        thrust_frame_link="thrust_1",
        origin_body=(0.0, 0.0, 0.0),
        axis_body=(0.0, 0.0, 1.0),
        thrust_min_n=0.0,
        thrust_max_n=20.0,
        reaction_torque_coeff_nm_per_n=0.0,
        reaction_torque_axis_body=(0.0, 0.0, 1.0),
        vectoring_joint_ids=[],
        virtual_x_axis_body=(1.0, 0.0, 0.0),
        virtual_z_axis_body=(0.0, 0.0, 1.0),
        allocation_column_body=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    )
    return RigidBodyControlModel(
        model_id="centroidal-admittance-test",
        graph_id="graph",
        base_module_id=0,
        body_pose_world=(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        body_twist_world=[
            linear_velocity_x_mps,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        total_mass_kg=1.0,
        center_of_mass_body=(0.0, 0.0, 0.0),
        inertia_body=[1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        rotor_elements=[rotor],
        rotor_origins_body={"module_0:thrust_1": rotor.origin_body},
        rotor_axes_body={"module_0:thrust_1": rotor.axis_body},
        allocation_matrix_body=[[0.0], [0.0], [1.0], [0.0], [0.0], [0.0]],
        vectoring_joint_axes_body={},
        dock_actuator_ids=[],
        active_actuator_limits={
            "module_0:thrust_1": {
                "lower": 0.0,
                "upper": 20.0,
                "velocity": None,
                "effort": None,
            }
        },
        current_joint_positions={},
    )


def _hover_command() -> ControllerCommand:
    return ControllerCommand(
        rotor_thrusts_n={"module_0:thrust_1": 9.80665},
        vectoring_joint_targets={},
        joint_torque_commands={},
        dock_mechanism_commands={},
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
    )


def test_centroidal_external_wrench_estimator_recovers_net_force() -> None:
    estimator = CentroidalExternalWrenchEstimator(
        CentroidalExternalWrenchEstimatorConfig(
            wrench_filter_time_constant_s=1.0e-6,
            bias_filter_time_constant_s=1.0,
        )
    )

    estimate = estimator.estimate(
        previous_model=_model(linear_velocity_x_mps=0.0),
        current_model=_model(linear_velocity_x_mps=0.1),
        applied_controller_command=_hover_command(),
        dt_s=0.1,
        calibrate_bias=False,
    )

    assert estimate.valid
    assert estimate.wrench_body == pytest.approx((1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert estimate.force_norm_n == pytest.approx(1.0)
    assert estimate.torque_norm_nm == pytest.approx(0.0)


def test_centroidal_external_wrench_estimator_calibrates_static_bias() -> None:
    estimator = CentroidalExternalWrenchEstimator()
    biased_hover = replace(
        _hover_command(),
        rotor_thrusts_n={"module_0:thrust_1": 9.0},
    )

    estimate = estimator.estimate(
        previous_model=_model(linear_velocity_x_mps=0.0),
        current_model=_model(linear_velocity_x_mps=0.0),
        applied_controller_command=biased_hover,
        dt_s=0.01,
        calibrate_bias=True,
    )

    assert estimate.valid
    assert estimate.wrench_body == pytest.approx((0.0,) * 6)
    assert estimate.raw_wrench_body[2] == pytest.approx(0.80665)


def test_centroidal_admittance_yields_with_force_and_bounds_offset() -> None:
    controller = CentroidalAdmittanceController(
        CentroidalAdmittanceConfig(
            force_deadband_n=0.5,
            linear_admittance_mps_per_n=0.01,
            maximum_linear_speed_mps=0.02,
            maximum_translation_offset_m=0.003,
        )
    )
    estimate = CentroidalExternalWrenchEstimate(
        valid=True,
        wrench_body=(10.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        raw_wrench_body=(10.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        bias_wrench_body=(0.0,) * 6,
        force_norm_n=10.0,
        torque_norm_nm=0.0,
    )
    pose = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    first = controller.update(
        nominal_pose_world=pose,
        current_pose_world=pose,
        estimate=estimate,
        dt_s=0.1,
        active=True,
    )
    second = controller.update(
        nominal_pose_world=pose,
        current_pose_world=pose,
        estimate=estimate,
        dt_s=0.1,
        active=True,
    )

    assert first.desired_body_twist[:3] == pytest.approx((0.02, 0.0, 0.0))
    assert first.translation_offset_world == pytest.approx((0.002, 0.0, 0.0))
    assert second.translation_offset_world == pytest.approx((0.003, 0.0, 0.0))
    assert second.desired_body_pose[:3] == pytest.approx((0.003, 0.0, 1.0))

    inactive = controller.update(
        nominal_pose_world=pose,
        current_pose_world=pose,
        estimate=estimate,
        dt_s=0.1,
        active=False,
    )
    assert not inactive.active
    assert inactive.translation_offset_world == (0.0, 0.0, 0.0)
    assert inactive.desired_body_pose == pose


def test_centroidal_admittance_can_yield_only_along_contact_axis() -> None:
    controller = CentroidalAdmittanceController(
        CentroidalAdmittanceConfig(
            force_deadband_n=0.1,
            torque_deadband_nm=0.01,
            linear_admittance_mps_per_n=0.001,
            angular_admittance_radps_per_nm=0.1,
            maximum_linear_speed_mps=1.0,
            maximum_angular_speed_radps=1.0,
            maximum_translation_offset_m=1.0,
        )
    )
    estimate = CentroidalExternalWrenchEstimate(
        valid=True,
        wrench_body=(2.0, 3.0, 4.0, 1.0, 2.0, 3.0),
        raw_wrench_body=(2.0, 3.0, 4.0, 1.0, 2.0, 3.0),
        bias_wrench_body=(0.0,) * 6,
        force_norm_n=5.0,
        torque_norm_nm=3.0,
    )
    pose = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    command = controller.update(
        nominal_pose_world=pose,
        current_pose_world=pose,
        estimate=estimate,
        dt_s=0.1,
        active=True,
        linear_projection_axis_world=(1.0, 0.0, 0.0),
        angular_admittance_enabled=False,
    )

    assert command.desired_body_twist[0] > 0.0
    assert command.desired_body_twist[1:] == pytest.approx((0.0,) * 5)
    assert command.translation_offset_world == pytest.approx(
        (0.1 * command.desired_body_twist[0], 0.0, 0.0)
    )
    assert command.desired_body_pose[2] == pytest.approx(1.0)


def test_centroidal_admittance_configs_fail_closed() -> None:
    with pytest.raises(SchemaValidationError, match="positive"):
        CentroidalAdmittanceConfig(maximum_linear_speed_mps=0.0).validate()
    with pytest.raises(SchemaValidationError, match="finite and positive"):
        CentroidalExternalWrenchEstimatorConfig(
            wrench_filter_time_constant_s=0.0
        ).validate()
