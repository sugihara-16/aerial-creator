from __future__ import annotations

import pytest

from amsrr.robot_model.joint_actuator_model import load_joint_actuator_model
from amsrr.robot_model.urdf_loader import load_urdf


def test_joint_actuator_model_loads_official_motor_limits() -> None:
    model = load_joint_actuator_model("configs/robot/joint_actuators.yaml")

    vectoring = model.actuator_roles["vectoring"]
    assert vectoring.model == "XC330-T181-T"
    assert vectoring.user_reported_model == "SC330-T181"
    assert vectoring.peak_torque_nm == pytest.approx(0.76)
    assert vectoring.no_load_speed_rad_s == pytest.approx(10.890854)
    assert vectoring.continuous_torque_limit_nm == pytest.approx(0.152)
    assert vectoring.simulation_drive.safe_velocity_limit_rad_s == pytest.approx(3.0)
    assert vectoring.simulation_drive.armature_kg_m2 == pytest.approx(0.0)

    dock = model.actuator_roles["dock"]
    assert dock.model == "AK40-10 KV170"
    assert dock.continuous_torque_limit_nm == pytest.approx(1.3)
    assert dock.peak_torque_nm == pytest.approx(4.1)
    assert dock.rated_speed_rad_s == pytest.approx(38.746309)
    assert dock.no_load_speed_rad_s == pytest.approx(45.553093)
    assert dock.protocol_torque_limit_nm == pytest.approx(5.0)
    assert dock.backlash_rad == pytest.approx(0.005236)
    assert dock.simulation_drive.armature_kg_m2 == pytest.approx(0.01)
    assert dock.simulation_drive.damping == pytest.approx(5.0)


def test_joint_actuator_model_matches_joint_roles() -> None:
    model = load_joint_actuator_model("configs/robot/joint_actuators.yaml")

    assert model.spec_for_joint("gimbal1").role == "vectoring"  # type: ignore[union-attr]
    assert model.spec_for_joint("pitch_dock_mech_joint1").role == "dock"  # type: ignore[union-attr]
    assert model.spec_for_joint("yaw_dock_mech_joint2").role == "dock"  # type: ignore[union-attr]
    assert model.spec_for_joint("rotor1") is None


def test_source_and_runtime_urdf_share_actuator_hard_limits() -> None:
    actuator_model = load_joint_actuator_model("configs/robot/joint_actuators.yaml")
    source = load_urdf("module_urdf/holon.urdf.xacro")
    runtime = load_urdf("assets/robots/holon/holon.urdf")
    source_joints = {joint.name: joint for joint in source.joints}
    runtime_joints = {joint.name: joint for joint in runtime.joints}

    for joint_id, source_joint in source_joints.items():
        spec = actuator_model.spec_for_joint(joint_id)
        if spec is None:
            continue
        runtime_joint = runtime_joints[joint_id]
        assert source_joint.effort_limit == pytest.approx(spec.peak_torque_nm)
        assert runtime_joint.effort_limit == pytest.approx(spec.peak_torque_nm)
        assert source_joint.velocity_limit == pytest.approx(spec.no_load_speed_rad_s)
        assert runtime_joint.velocity_limit == pytest.approx(spec.no_load_speed_rad_s)
