from __future__ import annotations

import torch

from amsrr.controllers.batched_qpid_controller import BatchedQPIDController
from amsrr.controllers.batched_rigid_body_model import (
    BatchedRigidBodyControlModelBuilder,
)
from amsrr.controllers.controller_base import ControllerContext
from amsrr.controllers.qp_allocator_interface import QPAllocationResult
from amsrr.controllers.qpid_controller import QPIDController, QPIDControllerConfig
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.policies import (
    POLICY_COMMAND_CONTRACT_CENTROIDAL,
    ControllerStatus,
    InteractionKnot,
    PolicyCommand,
)
from amsrr.schemas.runtime import ModuleRuntimeState, RuntimeObservation, TaskProgressState
from amsrr.simulation.order8_natural_contact import (
    build_representative_order8_morphology,
)


class _RecordingAllocator:
    def __init__(self) -> None:
        self.problem = None

    def allocate(self, problem):
        self.problem = problem
        desired = list(problem.desired_wrench_body or [0.0] * 6)
        return QPAllocationResult(
            rotor_thrusts_n={},
            feasible=True,
            residual_wrench_body=[0.0] * 6,
            residual_norm=0.0,
            achieved_wrench_body=desired,
        )


def _fixture(dtype=torch.float64):
    physical = build_physical_model_from_config("configs/robot/robot_model.yaml")
    graph = build_representative_order8_morphology(physical)
    builder = BatchedRigidBodyControlModelBuilder(graph, physical)
    module_ids = list(builder.module_ids)
    pose = torch.tensor(
        [
            [
                [0.20 + 0.31 * index, 0.0, 0.80, 0.0, 0.0, 0.0, 1.0]
                for index in range(len(module_ids))
            ]
        ],
        dtype=dtype,
    )
    twist = torch.tensor(
        [
            [
                [0.01, -0.02, 0.005, 0.02, -0.01, 0.03]
                for _ in module_ids
            ]
        ],
        dtype=dtype,
    )
    joints = torch.zeros(
        (1, len(module_ids), builder.local_joint_count), dtype=dtype
    )
    for module_index in range(len(module_ids)):
        for joint_index, joint in enumerate(physical.joints):
            if joint.joint_type == "revolute":
                joints[0, module_index, joint_index] = 0.01 * (
                    module_index + joint_index % 2
                )
    control_model = builder.build(
        module_pose_world=pose,
        module_twist_world=twist,
        local_joint_positions_rad=joints,
    )
    states = [
        ModuleRuntimeState(
            module_id=module_id,
            pose_world=tuple(pose[0, module_index].tolist()),
            twist_world=twist[0, module_index].tolist(),
            joint_positions={
                joint_id: float(joints[0, module_index, joint_index])
                for joint_index, joint_id in enumerate(builder.local_joint_ids)
            },
            joint_velocities={joint_id: 0.0 for joint_id in builder.local_joint_ids},
        )
        for module_index, module_id in enumerate(module_ids)
    ]
    observation = RuntimeObservation(
        time_s=0.0,
        morphology_graph=graph,
        module_states=states,
        object_states=[],
        contact_states=[],
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
        task_progress=TaskProgressState(),
    )
    return physical, graph, observation, control_model


def test_batched_qpid_desired_wrench_matches_scalar_controller() -> None:
    physical, graph, observation, control_model = _fixture()
    config = QPIDControllerConfig(
        allocation_mode="rigid_body_qp",
        control_dt_s=0.02,
        unsupported_wrench_tolerance=1000.0,
    )
    target_pose = control_model.body_pose_world.clone()
    target_pose[:, :3] += torch.tensor([[0.02, -0.01, 0.03]], dtype=torch.float64)
    target_twist = torch.tensor(
        [[0.04, -0.02, 0.01, 0.01, -0.01, 0.02]], dtype=torch.float64
    )
    residual = torch.tensor(
        [[0.1, -0.2, 0.3, 0.01, -0.02, 0.03]], dtype=torch.float64
    )
    recorder = _RecordingAllocator()
    scalar = QPIDController(allocator=recorder, config=config)
    command = PolicyCommand(
        desired_body_pose=tuple(target_pose[0].tolist()),
        desired_body_twist=target_twist[0].tolist(),
        residual_wrench_body=residual[0].tolist(),
        control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
    )
    scalar.compute(
        ControllerContext(
            runtime_observation=observation,
            morphology_graph=graph,
            physical_model=physical,
            active_knot=InteractionKnot(t_rel_s=0.0, contact_assignments=[]),
            policy_command=command,
            control_dt_s=0.02,
        )
    )
    assert recorder.problem is not None
    batched = BatchedQPIDController(config=config)
    state = batched.initial_state(
        1,
        control_model.current_vectoring_angles_rad.shape[1],
        device=target_pose.device,
        dtype=target_pose.dtype,
    )
    result = batched.compute(
        control_model=control_model,
        desired_body_pose_world=target_pose,
        desired_body_twist=target_twist,
        residual_wrench_body=residual,
        state=state,
    )

    torch.testing.assert_close(
        result.desired_wrench_body[0],
        torch.tensor(recorder.problem.desired_wrench_body, dtype=torch.float64),
        rtol=5.0e-10,
        atol=5.0e-10,
    )


def test_batched_qpid_freezes_integrator_when_allocation_clips() -> None:
    _, _, _, control_model = _fixture(dtype=torch.float32)
    config = QPIDControllerConfig(
        allocation_mode="rigid_body_qp",
        control_dt_s=0.02,
        unsupported_wrench_tolerance=1000.0,
    )
    controller = BatchedQPIDController(config=config)
    state = controller.initial_state(
        1,
        control_model.current_vectoring_angles_rad.shape[1],
        device=control_model.body_pose_world.device,
        dtype=control_model.body_pose_world.dtype,
    )
    target = control_model.body_pose_world.clone()
    target[:, 2] += 100.0
    result = controller.compute(
        control_model=control_model,
        desired_body_pose_world=target,
        desired_body_twist=torch.zeros((1, 6)),
        residual_wrench_body=torch.zeros((1, 6)),
        state=state,
    )

    assert not result.integrator_committed.item()
    torch.testing.assert_close(
        result.next_state.position_error_integral_world,
        state.position_error_integral_world,
    )
    assert result.allocation.thrust_clipped.any().item()
