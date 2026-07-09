from __future__ import annotations

import math

import pytest

from amsrr.controllers.controller_base import ControllerContext
from amsrr.controllers.qp_allocator_interface import (
    BoundedVerticalRotorAllocator,
    QPAllocationResult,
    QPAllocationProblem,
    QP_INFEASIBLE_CODE,
    QP_THRUST_CLIPPED_CODE,
    QP_UNSUPPORTED_WRENCH_CODE,
    QP_VECTORING_CLIPPED_CODE,
    RigidBodyPseudoinverseAllocator,
    RotorAllocationSpec,
    VirtualThrustQPAllocator,
)
from amsrr.controllers.qpid_controller import QPIDController, QPIDControllerConfig
from amsrr.controllers.rigid_body_model import RigidBodyControlModel, RotorControlElement
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


def _multi_module_morphology(module_count: int = 2) -> MorphologyGraph:
    physical_model = _physical_model()
    capability = build_module_capability_token(physical_model)
    return MorphologyGraph(
        graph_id=f"controller-test-{module_count}-module",
        modules=[
            ModuleNode(
                module_id=module_id,
                module_type="holon",
                pose_in_design_frame=(0.45 * module_id, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                role_id="base" if module_id == 0 else "fixed_attached",
                is_base=module_id == 0,
                capability_token=capability,
            )
            for module_id in range(module_count)
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


def _multi_module_runtime_observation(module_count: int = 2) -> RuntimeObservation:
    morphology = _multi_module_morphology(module_count)
    return RuntimeObservation(
        time_s=0.0,
        morphology_graph=morphology,
        module_states=[
            ModuleRuntimeState(
                module_id=module_id,
                pose_world=(0.45 * module_id, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                twist_world=[0.0] * 6,
                joint_positions={
                    "gimbal1": 0.0,
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


def _single_vectoring_rigid_body_model(
    *,
    thrust_max_n: float = 10.0,
    joint_lower: float = -1.0,
    joint_upper: float = 1.0,
    joint_velocity: float | None = None,
) -> RigidBodyControlModel:
    rotor = RotorControlElement(
        global_rotor_id="module_0:thrust_1",
        module_id=0,
        rotor_id="thrust_1",
        thrust_frame_link="thrust_1",
        origin_body=(0.0, 0.0, 0.0),
        axis_body=(0.0, 0.0, 1.0),
        thrust_min_n=0.0,
        thrust_max_n=thrust_max_n,
        reaction_torque_coeff_nm_per_n=0.0,
        reaction_torque_axis_body=(0.0, 0.0, 1.0),
        vectoring_joint_ids=["module_0:gimbal1"],
        virtual_x_axis_body=(1.0, 0.0, 0.0),
        virtual_z_axis_body=(0.0, 0.0, 1.0),
        allocation_column_body=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    )
    return RigidBodyControlModel(
        model_id="test-rigid-body",
        graph_id="test-graph",
        base_module_id=0,
        body_pose_world=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        total_mass_kg=1.0,
        center_of_mass_body=(0.0, 0.0, 0.0),
        inertia_body=[1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        rotor_elements=[rotor],
        rotor_origins_body={"module_0:thrust_1": rotor.origin_body},
        rotor_axes_body={"module_0:thrust_1": rotor.axis_body},
        allocation_matrix_body=[[0.0], [0.0], [1.0], [0.0], [0.0], [0.0]],
        vectoring_joint_axes_body={"module_0:gimbal1": (1.0, 0.0, 0.0)},
        dock_actuator_ids=[],
        active_actuator_limits={
            "module_0:thrust_1": {"lower": 0.0, "upper": thrust_max_n, "velocity": None, "effort": None},
            "module_0:gimbal1": {
                "lower": joint_lower,
                "upper": joint_upper,
                "velocity": joint_velocity,
                "effort": 6.6,
            },
        },
        current_joint_positions={"module_0:gimbal1": 0.0},
    )


class _RecordingAllocator:
    def __init__(self) -> None:
        self.problem: QPAllocationProblem | None = None

    def allocate(self, problem: QPAllocationProblem):
        self.problem = problem
        return QPAllocationResult(
            rotor_thrusts_n={},
            feasible=True,
            residual_wrench_body=[0.0] * 6,
            residual_norm=0.0,
            achieved_wrench_body=list(problem.desired_wrench_body or [0.0] * 6),
            metrics={"qp_primary_path": 1.0},
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


def test_virtual_thrust_qp_allocator_back_converts_vectoring_channel() -> None:
    allocator = VirtualThrustQPAllocator()
    allocator.regularization_weight = 0.0
    allocator.previous_command_weight = 0.0
    problem = QPAllocationProblem(
        desired_wrench_body=[2.0, 0.0, 5.0, 0.0, 0.0, 0.0],
        rotors=[],
        rigid_body_model=_single_vectoring_rigid_body_model(),
        unsupported_wrench_tolerance=1.0e-6,
    )

    result = allocator.allocate(problem)

    assert result.feasible is True
    assert result.metrics["qp_primary_path"] == 1.0
    assert result.metrics["degraded_fallback"] == 0.0
    assert result.metrics["virtual_channel_count"] == 2.0
    assert result.rotor_thrusts_n["module_0:thrust_1"] == pytest.approx((2.0**2 + 5.0**2) ** 0.5)
    assert result.vectoring_joint_targets["module_0:gimbal1"] == pytest.approx(0.3805063771)
    assert result.achieved_wrench_body[:3] == pytest.approx([2.0, 0.0, 5.0], abs=1.0e-6)


def test_virtual_thrust_qp_allocator_applies_limits_and_hard_clamp() -> None:
    allocator = VirtualThrustQPAllocator()
    allocator.regularization_weight = 0.0
    allocator.previous_command_weight = 0.0
    problem = QPAllocationProblem(
        desired_wrench_body=[10.0, 0.0, 10.0, 0.0, 0.0, 0.0],
        rotors=[],
        rigid_body_model=_single_vectoring_rigid_body_model(thrust_max_n=10.0, joint_velocity=0.5),
        control_dt_s=0.1,
        unsupported_wrench_tolerance=100.0,
    )

    result = allocator.allocate(problem)

    assert result.rotor_thrusts_n["module_0:thrust_1"] <= 10.0
    assert result.vectoring_joint_targets["module_0:gimbal1"] <= 0.05
    assert result.clipped is True
    assert QP_THRUST_CLIPPED_CODE in result.violation_codes or QP_VECTORING_CLIPPED_CODE in result.violation_codes


def test_rigid_body_pseudoinverse_allocator_back_converts_vectoring_channel() -> None:
    problem = QPAllocationProblem(
        desired_wrench_body=[2.0, 0.0, 5.0, 0.0, 0.0, 0.0],
        rotors=[],
        rigid_body_model=_single_vectoring_rigid_body_model(),
        unsupported_wrench_tolerance=1.0e-6,
    )

    result = RigidBodyPseudoinverseAllocator().allocate(problem)

    assert result.feasible is True
    assert result.metrics["pseudoinverse_path"] == 1.0
    assert result.metrics["qp_primary_path"] == 0.0
    assert result.rotor_thrusts_n["module_0:thrust_1"] == pytest.approx((2.0**2 + 5.0**2) ** 0.5)
    assert result.vectoring_joint_targets["module_0:gimbal1"] == pytest.approx(0.3805063771)
    assert result.achieved_wrench_body[:3] == pytest.approx([2.0, 0.0, 5.0], abs=1.0e-6)


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
    assert controller_command.dock_mechanism_commands["pitch_dock_mech_joint1"] == pytest.approx(0.0)
    assert type(controller_command).from_json(controller_command.to_json()).to_dict() == controller_command.to_dict()


def test_qpid_controller_uses_posture_target_for_dock_mechanism_commands() -> None:
    physical_model = _physical_model()
    runtime = _runtime_observation()

    controller_command = QPIDController().compute(
        ControllerContext(
            runtime_observation=runtime,
            morphology_graph=runtime.morphology_graph,
            physical_model=physical_model,
            active_knot=InteractionKnot(
                t_rel_s=0.0,
                contact_assignments=[],
                posture_target=PostureTarget(
                    joint_pos_target={
                        "pitch_dock_mech_joint1": 0.25,
                        "yaw_dock_mech_joint1": 99.0,
                    }
                ),
            ),
            policy_command=PolicyCommand(),
        )
    )

    assert controller_command.dock_mechanism_commands["pitch_dock_mech_joint1"] == pytest.approx(0.25)
    assert controller_command.dock_mechanism_commands["yaw_dock_mech_joint1"] == pytest.approx(1.5708)


def test_qpid_controller_default_hover_uses_rigid_body_total_mass_for_multi_module() -> None:
    physical_model = _physical_model()
    runtime = _multi_module_runtime_observation(module_count=2)
    allocator = _RecordingAllocator()

    QPIDController(
        allocator=allocator,
        config=QPIDControllerConfig(allocation_mode="rigid_body_qp"),
    ).compute(
        ControllerContext(
            runtime_observation=runtime,
            morphology_graph=runtime.morphology_graph,
            physical_model=physical_model,
            active_knot=InteractionKnot(t_rel_s=0.0, contact_assignments=[]),
            policy_command=PolicyCommand(),
        )
    )

    assert allocator.problem is not None
    assert allocator.problem.rigid_body_model is not None
    assert allocator.problem.rigid_body_model.total_mass_kg == pytest.approx(2.0 * physical_model.aggregate_mass_kg)
    assert allocator.problem.desired_wrench_body[2] == pytest.approx(
        2.0 * physical_model.aggregate_mass_kg * QPIDControllerConfig().gravity_mps2
    )


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


def test_qpid_controller_can_select_rigid_body_qp_primary_path() -> None:
    physical_model = _physical_model()
    runtime = _runtime_observation()

    controller_command = QPIDController(
        config=QPIDControllerConfig(
            allocation_mode="rigid_body_qp",
            unsupported_wrench_tolerance=1000.0,
        )
    ).compute(
        ControllerContext(
            runtime_observation=runtime,
            morphology_graph=runtime.morphology_graph,
            physical_model=physical_model,
            active_knot=_active_knot(wrench_z=10.0),
            policy_command=PolicyCommand(),
        )
    )

    assert controller_command.controller_status.active_mode == "qpid_rigid_body_qp"
    assert controller_command.controller_status.metrics["qp_primary_path"] == 1.0
    assert any(key.startswith("module_0:thrust_") for key in controller_command.rotor_thrusts_n)
    assert any(key.startswith("module_0:gimbal") for key in controller_command.vectoring_joint_targets)


def test_qpid_controller_rigid_body_qp_hover_is_feasible_with_default_tolerance() -> None:
    physical_model = _physical_model()
    runtime = _runtime_observation()

    controller_command = QPIDController(
        config=QPIDControllerConfig(allocation_mode="rigid_body_qp")
    ).compute(
        ControllerContext(
            runtime_observation=runtime,
            morphology_graph=runtime.morphology_graph,
            physical_model=physical_model,
            active_knot=_active_knot(wrench_z=physical_model.aggregate_mass_kg * 9.80665),
            policy_command=PolicyCommand(),
        )
    )

    assert controller_command.controller_status.status == "ok"
    assert controller_command.controller_status.qp_feasible is True
    assert controller_command.controller_status.metrics["allocation_residual_norm"] < 1.0e-4
    assert controller_command.controller_status.metrics["clipped"] == 0.0


def test_qpid_controller_can_select_rigid_body_pseudoinverse_debug_path() -> None:
    physical_model = _physical_model()
    runtime = _runtime_observation()

    controller_command = QPIDController(
        config=QPIDControllerConfig(
            allocation_mode="rigid_body_pseudoinverse",
            unsupported_wrench_tolerance=1000.0,
        )
    ).compute(
        ControllerContext(
            runtime_observation=runtime,
            morphology_graph=runtime.morphology_graph,
            physical_model=physical_model,
            active_knot=_active_knot(wrench_z=10.0),
            policy_command=PolicyCommand(),
        )
    )

    assert controller_command.controller_status.active_mode == "qpid_rigid_body_pseudoinverse"
    assert controller_command.controller_status.metrics["pseudoinverse_path"] == 1.0
    assert any(key.startswith("module_0:thrust_") for key in controller_command.rotor_thrusts_n)
    assert any(key.startswith("module_0:gimbal") for key in controller_command.vectoring_joint_targets)


def test_qpid_controller_builds_pid_wrench_from_policy_body_target_and_feedforward() -> None:
    physical_model = _physical_model()
    runtime = _runtime_observation()
    yaw_target_rad = 0.1
    allocator = _RecordingAllocator()
    controller = QPIDController(
        allocator=allocator,
        config=QPIDControllerConfig(control_dt_s=0.005),
    )

    controller_command = controller.compute(
        ControllerContext(
            runtime_observation=runtime,
            morphology_graph=runtime.morphology_graph,
            physical_model=physical_model,
            active_knot=InteractionKnot(t_rel_s=0.0, contact_assignments=[]),
            policy_command=PolicyCommand(
                desired_body_pose=(
                    0.0,
                    0.0,
                    0.2,
                    0.0,
                    0.0,
                    math.sin(0.5 * yaw_target_rad),
                    math.cos(0.5 * yaw_target_rad),
                ),
                desired_body_twist=[0.0] * 6,
                residual_wrench_body=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ),
        )
    )

    assert allocator.problem is not None
    desired = allocator.problem.desired_wrench_body
    assert desired is not None
    rigid_model = controller.rigid_body_model_builder.build(runtime.morphology_graph, physical_model, runtime)
    expected_z_acc = 5.0 * 0.2 + 1.0 * (0.2 * 0.005)
    expected_yaw_ang_acc = 5.0 * yaw_target_rad + 1.0 * (yaw_target_rad * 0.005)
    expected_torque = (
        rigid_model.inertia_body[2] * expected_yaw_ang_acc,
        rigid_model.inertia_body[4] * expected_yaw_ang_acc,
        rigid_model.inertia_body[5] * expected_yaw_ang_acc,
    )
    assert desired[0] == pytest.approx(1.0)
    assert desired[1] == pytest.approx(2.0)
    assert desired[2] == pytest.approx(physical_model.aggregate_mass_kg * (9.80665 + expected_z_acc) + 3.0)
    assert desired[3] == pytest.approx(expected_torque[0] + 4.0)
    assert desired[4] == pytest.approx(expected_torque[1] + 5.0)
    assert desired[5] == pytest.approx(expected_torque[2] + 6.0)
    assert controller_command.controller_status.metrics["pid_target_builder_active"] == 1.0
    assert controller_command.controller_status.metrics["target_pos_error_m"] == pytest.approx(0.2)
    assert controller_command.controller_status.metrics["target_rot_error_rad"] == pytest.approx(yaw_target_rad)
