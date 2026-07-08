from __future__ import annotations

import pytest

from amsrr.controllers.controller_base import ControllerContext
from amsrr.controllers.qp_allocator_interface import (
    BoundedVerticalRotorAllocator,
    QPAllocationProblem,
    QP_INFEASIBLE_CODE,
    QP_THRUST_CLIPPED_CODE,
    QP_UNSUPPORTED_WRENCH_CODE,
    RotorAllocationSpec,
)
from amsrr.controllers.qpid_controller import QPIDController
from amsrr.robot_model.physical_model_builder import build_module_capability_token, build_physical_model_from_config
from amsrr.schemas.morphology import ModuleNode, MorphologyGraph
from amsrr.schemas.policies import CentroidalTarget, ControllerStatus, InteractionKnot, PolicyCommand, PostureTarget
from amsrr.schemas.runtime import ModuleRuntimeState, RuntimeObservation, TaskProgressState


def _physical_model():
    return build_physical_model_from_config("configs/robot/robot_model.yaml")


def _morphology_graph():
    physical_model = _physical_model()
    capability = build_module_capability_token(physical_model)
    return MorphologyGraph(
        graph_id="controller-test",
        modules=[
            ModuleNode(
                module_id=0,
                module_type="holon",
                pose_in_design_frame=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                role_id="base",
                is_base=True,
                capability_token=capability,
            )
        ],
        ports=[],
        dock_edges=[],
        robot_anchors=[],
        control_groups=[],
        base_module_id=0,
        is_closed_loop=False,
    )


def _runtime_observation():
    return RuntimeObservation(
        time_s=0.0,
        morphology_graph=_morphology_graph(),
        module_states=[
            ModuleRuntimeState(
                module_id=0,
                pose_world=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                twist_world=[0.0] * 6,
                joint_positions={"gimbal1": 0.0, "payload_joint": 0.0, "pitch_dock_mech_joint1": 0.1},
                joint_velocities={"gimbal1": 0.0, "payload_joint": 0.0},
            )
        ],
        object_states=[],
        contact_states=[],
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
        task_progress=TaskProgressState(),
    )


def _active_knot(wrench_z: float = 10.0) -> InteractionKnot:
    return InteractionKnot(
        t_rel_s=0.0,
        contact_assignments=[],
        centroidal_target=CentroidalTarget(
            centroidal_wrench_preference=[0.0, 0.0, wrench_z, 0.0, 0.0, 0.0],
        ),
        posture_target=PostureTarget(
            joint_pos_target={"gimbal1": 0.0, "payload_joint": 0.0},
            joint_vel_target={"payload_joint": 0.0},
        ),
        priority_weights={"controller": 1.0},
    )


def test_bounded_vertical_rotor_allocator_feasible_and_unsupported_residual() -> None:
    problem = QPAllocationProblem(
        desired_wrench_body=[1.0, 0.0, 10.0, 0.0, 0.0, 0.0],
        rotors=[
            RotorAllocationSpec("r1", (0.0, 0.0, 1.0), 0.0, 10.0),
            RotorAllocationSpec("r2", (0.0, 0.0, -1.0), 0.0, 10.0),
        ],
    )

    result = BoundedVerticalRotorAllocator().allocate(problem)

    assert result.feasible is True
    assert result.rotor_thrusts_n == {"r1": 5.0, "r2": 5.0}
    assert result.residual_wrench_body == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert result.metrics["unsupported_wrench_norm"] == pytest.approx(1.0)
    assert QP_UNSUPPORTED_WRENCH_CODE in result.violation_codes


def test_bounded_vertical_rotor_allocator_reports_infeasible_clip() -> None:
    problem = QPAllocationProblem(
        desired_wrench_body=[0.0, 0.0, 25.0, 0.0, 0.0, 0.0],
        rotors=[
            RotorAllocationSpec("r1", (0.0, 0.0, 1.0), 0.0, 10.0),
            RotorAllocationSpec("r2", (0.0, 0.0, 1.0), 0.0, 10.0),
        ],
    )

    result = BoundedVerticalRotorAllocator().allocate(problem)

    assert result.feasible is False
    assert result.clipped is True
    assert result.residual_wrench_body[2] == pytest.approx(5.0)
    assert QP_INFEASIBLE_CODE in result.violation_codes
    assert QP_THRUST_CLIPPED_CODE in result.violation_codes


def test_qpid_controller_outputs_controller_command() -> None:
    physical_model = _physical_model()
    runtime = _runtime_observation()
    active_knot = _active_knot()
    command = PolicyCommand(
        joint_position_bias={"gimbal1": 3.0, "payload_joint": 2.0},
        joint_velocity_bias={"payload_joint": 1.0},
    )

    controller_command = QPIDController().compute(
        ControllerContext(
            runtime_observation=runtime,
            morphology_graph=runtime.morphology_graph,
            physical_model=physical_model,
            active_knot=active_knot,
            policy_command=command,
        )
    )

    assert controller_command.controller_status.status == "ok"
    assert controller_command.controller_status.qp_feasible is True
    assert sum(controller_command.rotor_thrusts_n.values()) == pytest.approx(10.0)
    assert all(0.0 <= value <= 20.0 for value in controller_command.rotor_thrusts_n.values())
    assert controller_command.vectoring_joint_targets["gimbal1"] == pytest.approx(2.0)
    assert controller_command.joint_torque_commands["payload_joint"] == pytest.approx(8.4)
    assert controller_command.dock_mechanism_commands["pitch_dock_mech_joint1"] == pytest.approx(0.1)
    assert type(controller_command).from_json(controller_command.to_json()).to_dict() == controller_command.to_dict()


def test_qpid_controller_reports_infeasible_vertical_wrench() -> None:
    physical_model = _physical_model()
    runtime = _runtime_observation()

    controller_command = QPIDController().compute(
        ControllerContext(
            runtime_observation=runtime,
            morphology_graph=runtime.morphology_graph,
            physical_model=physical_model,
            active_knot=_active_knot(wrench_z=1000.0),
            policy_command=PolicyCommand(),
        )
    )

    assert controller_command.controller_status.status == "infeasible"
    assert controller_command.controller_status.qp_feasible is False
    assert controller_command.controller_status.metrics["clipped"] == 1.0
    assert sum(controller_command.rotor_thrusts_n.values()) == pytest.approx(80.0)
