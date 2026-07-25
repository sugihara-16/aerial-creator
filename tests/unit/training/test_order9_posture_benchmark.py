from __future__ import annotations

from amsrr.robot_model.physical_model_builder import (
    build_physical_model_from_config,
)
from amsrr.schemas.physical_model import PhysicalModel
from scripts.order9_benchmark_posture_resolver import (
    _maximum_float_difference,
    _physical_model_content_hash,
    _quantized_stable_hash,
)


def test_physical_model_content_hash_ignores_project_root_paths() -> None:
    physical = build_physical_model_from_config(
        "configs/robot/robot_model.yaml"
    )
    relocated_payload = physical.to_dict()
    relocated_payload["urdf_path"] = (
        "/opt/amsrr/assets/robots/holon/holon.urdf"
    )
    relocated_payload["metadata"]["joint_actuator_model_path"] = (
        "/opt/amsrr/configs/robot/joint_actuators.yaml"
    )
    relocated = PhysicalModel.from_dict(relocated_payload)

    assert relocated.stable_hash() != physical.stable_hash()
    assert _physical_model_content_hash(
        relocated
    ) == _physical_model_content_hash(physical)


def test_quantized_trajectory_hash_ignores_subnanounit_roundoff() -> None:
    baseline = {
        "joint_pos_target": [1.348193204628398, -2.149512671952793e-8],
        "time_s": 0.05,
    }
    cross_platform = {
        "joint_pos_target": [1.3481932046284004, -2.1495094629120846e-8],
        "time_s": 0.05,
    }
    materially_different = {
        "joint_pos_target": [1.348193404628398, -2.149512671952793e-8],
        "time_s": 0.05,
    }

    assert _quantized_stable_hash(
        cross_platform
    ) == _quantized_stable_hash(baseline)
    assert _quantized_stable_hash(
        materially_different
    ) != _quantized_stable_hash(baseline)


def test_trajectory_difference_preserves_structure_and_measures_floats() -> None:
    expected = {
        "knots": [
            {
                "joint_pos_target": [1.0, -0.25],
                "schedule_state": "maintain",
            }
        ]
    }
    actual = {
        "knots": [
            {
                "joint_pos_target": [1.0 + 2.6e-11, -0.25],
                "schedule_state": "maintain",
            }
        ]
    }

    assert _maximum_float_difference(expected, actual) < 3.0e-11
