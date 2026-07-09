from __future__ import annotations

import math

import pytest

from amsrr.controllers.rigid_body_model import RigidBodyControlModelBuilder
from amsrr.robot_model.physical_model_builder import build_module_capability_token, build_physical_model_from_config
from amsrr.schemas.morphology import ControlGroup, ModuleNode, MorphologyGraph
from amsrr.schemas.policies import ControllerStatus
from amsrr.schemas.runtime import ModuleRuntimeState, RuntimeObservation, TaskProgressState


def _physical_model():
    return build_physical_model_from_config("configs/robot/robot_model.yaml")


def _morphology(module_count: int = 1) -> MorphologyGraph:
    physical_model = _physical_model()
    capability = build_module_capability_token(physical_model)
    return MorphologyGraph(
        graph_id=f"rigid-body-test-{module_count}",
        modules=[
            ModuleNode(
                module_id=module_id,
                module_type="holon",
                pose_in_design_frame=(0.25 * module_id, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                role_id="base" if module_id == 0 else "attached",
                is_base=module_id == 0,
                capability_token=capability,
            )
            for module_id in range(module_count)
        ],
        ports=[],
        dock_edges=[],
        robot_anchors=[],
        control_groups=[ControlGroup(group_id="all", module_ids=list(range(module_count)), role="whole_body")],
        base_module_id=0,
        is_closed_loop=False,
    )


def _runtime(module_count: int = 1, *, gimbal1: float = 0.0) -> RuntimeObservation:
    return RuntimeObservation(
        time_s=0.0,
        morphology_graph=_morphology(module_count),
        module_states=[
            ModuleRuntimeState(
                module_id=module_id,
                pose_world=(0.4 * module_id, 0.0, 0.1 * module_id, 0.0, 0.0, 0.0, 1.0),
                twist_world=[0.0] * 6,
                joint_positions={
                    "gimbal1": gimbal1,
                    "gimbal2": 0.0,
                    "gimbal3": 0.0,
                    "gimbal4": 0.0,
                    "pitch_dock_mech_joint1": 0.0,
                    "pitch_dock_mech_joint2": 0.0,
                    "yaw_dock_mech_joint1": 0.0,
                    "yaw_dock_mech_joint2": 0.0,
                },
                joint_velocities={},
            )
            for module_id in range(module_count)
        ],
        object_states=[],
        contact_states=[],
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
        task_progress=TaskProgressState(),
    )


def test_rigid_body_model_builds_single_module_allocation_matrix() -> None:
    physical_model = _physical_model()
    model = RigidBodyControlModelBuilder().build(
        _morphology(),
        physical_model,
        _runtime(),
    )

    assert model.total_mass_kg == pytest.approx(physical_model.aggregate_mass_kg)
    assert model.center_of_mass_body == pytest.approx((0.0, 0.0, 0.0))
    assert model.metadata["body_frame_origin"] == "com"
    assert model.metadata["body_frame_orientation_source"] == "module:0"
    assert len(model.rotor_elements) == 4
    assert len(model.allocation_matrix_body) == 6
    assert all(len(row) == 4 for row in model.allocation_matrix_body)
    assert sorted(model.rotor_origins_body) == [
        "module_0:thrust_1",
        "module_0:thrust_2",
        "module_0:thrust_3",
        "module_0:thrust_4",
    ]
    assert set(model.vectoring_joint_axes_body) == {
        "module_0:gimbal1",
        "module_0:gimbal2",
        "module_0:gimbal3",
        "module_0:gimbal4",
    }
    assert model.current_joint_positions["module_0:gimbal1"] == pytest.approx(0.0)
    assert model.rotor_elements[0].virtual_x_axis_body is not None
    assert model.rotor_elements[0].virtual_z_axis_body is not None
    assert model.to_dict()["rotor_elements"][0]["global_rotor_id"].startswith("module_0:")


def test_rigid_body_model_updates_rotor_axis_from_joint_position() -> None:
    physical_model = _physical_model()
    builder = RigidBodyControlModelBuilder()

    model_zero = builder.build(_morphology(), physical_model, _runtime(gimbal1=0.0))
    model_tilted = builder.build(_morphology(), physical_model, _runtime(gimbal1=0.5))

    axis_zero = model_zero.rotor_axes_body["module_0:thrust_1"]
    axis_tilted = model_tilted.rotor_axes_body["module_0:thrust_1"]
    assert axis_tilted != pytest.approx(axis_zero)
    assert math.sqrt(sum(value * value for value in axis_tilted)) == pytest.approx(1.0)
    assert model_tilted.allocation_matrix_body[1][0] != pytest.approx(model_zero.allocation_matrix_body[1][0])


def test_rigid_body_model_handles_multiple_modules_with_unique_actuator_ids() -> None:
    physical_model = _physical_model()
    model = RigidBodyControlModelBuilder().build(
        _morphology(module_count=2),
        physical_model,
        _runtime(module_count=2),
    )

    assert model.total_mass_kg == pytest.approx(2.0 * physical_model.aggregate_mass_kg)
    assert len(model.rotor_elements) == 8
    assert "module_0:thrust_1" in model.rotor_origins_body
    assert "module_1:thrust_1" in model.rotor_origins_body
    assert len(set(model.rotor_origins_body)) == 8
    assert model.metadata["active_module_count"] == 2
    assert any(key.startswith("module_1:gimbal") for key in model.active_actuator_limits)
