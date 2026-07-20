from types import SimpleNamespace

import torch

from amsrr.policies.order9_tensor_command_decoder import Order9TensorPolicyCommand
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.simulation.order8_natural_contact import (
    build_representative_order8_morphology,
)
from amsrr.simulation.order9_tensor_isaac_io import Order9TensorIsaacIO


def _fixture():
    physical = build_physical_model_from_config("configs/robot/robot_model.yaml")
    morphology = build_representative_order8_morphology(physical)
    module_ids = tuple(sorted(module.module_id for module in morphology.modules))
    body_names = tuple(
        f"module_{module_id}__{link.link_id}"
        for module_id in module_ids
        for link in physical.links
    )
    joint_names = tuple(
        f"module_{module_id}__{joint.joint_id}"
        for module_id in module_ids
        for joint in physical.joints
    )
    io = Order9TensorIsaacIO(
        morphology_graph=morphology,
        physical_model=physical,
        robot_body_names=body_names,
        robot_joint_names=joint_names,
    )
    return physical, morphology, io


def test_tensor_isaac_io_gathers_exact_module_and_joint_layout() -> None:
    _, _, io = _fixture()
    batch = 2
    bodies = len(io.robot_body_names)
    joints = len(io.robot_joint_names)
    body_pose = torch.zeros((batch, bodies, 7))
    body_pose[..., 6] = 1.0
    body_linear = torch.arange(batch * bodies * 3, dtype=torch.float32).reshape(
        batch, bodies, 3
    )
    body_angular = -body_linear
    joint_position = torch.arange(batch * joints, dtype=torch.float32).reshape(
        batch, joints
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_pose_w=body_pose,
            body_lin_vel_w=body_linear,
            body_ang_vel_w=body_angular,
            joint_pos=joint_position,
            joint_vel=-joint_position,
            root_pose_w=torch.tensor(
                [[0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]] * batch
            ),
            root_lin_vel_w=torch.zeros((batch, 3)),
            root_ang_vel_w=torch.ones((batch, 3)),
        )
    )
    object_asset = SimpleNamespace(
        data=SimpleNamespace(
            root_pose_w=torch.tensor(
                [[1.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]] * batch
            ),
            root_lin_vel_w=torch.zeros((batch, 3)),
            root_ang_vel_w=torch.zeros((batch, 3)),
        )
    )

    state = io.gather_state(robot=robot, object_asset=object_asset)

    assert state.module_pose_world.shape == (batch, io.module_count, 7)
    assert state.module_twist_world.shape == (batch, io.module_count, 6)
    assert state.local_joint_positions_rad.shape == (
        batch,
        io.module_count,
        len(io.local_joint_ids),
    )
    expected = joint_position[0, io.local_joint_indices[1][3]]
    assert state.local_joint_positions_rad[0, 1, 3] == expected
    assert torch.equal(
        state.module_twist_world[0, 0, :3],
        body_linear[0, io.module_body_indices[0]],
    )
    assert state.robot_root_twist_world.shape == (batch, 6)
    assert state.object_pose_world.shape == (batch, 7)


def test_tensor_contact_reducer_separates_allowed_object_and_environment_contact() -> None:
    _, _, io = _fixture()
    batch = 2
    bodies = len(io.robot_body_names)
    anchors = io.selected_anchor_count
    object_matrix = torch.zeros((batch, 1, bodies, 3))
    for filter_index in io.selected_anchor_filter_indices:
        object_matrix[:, 0, filter_index, 2] = 2.0
    # Equal-and-opposite object contact is the only robot contact initially.
    robot_net = -object_matrix[:, 0].clone()
    assignment = torch.ones((batch, anchors), dtype=torch.bool)
    allow = torch.ones((batch,), dtype=torch.bool)

    evidence = io.reduce_contacts(
        robot_net_contact_forces_world=robot_net,
        object_force_matrix_world=object_matrix,
        robot_body_linear_velocity_world=torch.zeros((batch, bodies, 3)),
        robot_body_angular_velocity_world=torch.zeros((batch, bodies, 3)),
        selected_assignment_mask=assignment,
        allow_selected_object_contact=allow,
    )

    assert evidence.selected_contact_mask.all()
    assert not evidence.prohibited_collision.any()
    unselected = next(
        index
        for index in range(bodies)
        if index not in io.selected_anchor_body_indices
    )
    object_matrix[0, 0, unselected, 0] = 1.0
    robot_net[0, unselected, 0] = -1.0
    robot_net[1, unselected, 1] = 1.0
    evidence = io.reduce_contacts(
        robot_net_contact_forces_world=robot_net,
        object_force_matrix_world=object_matrix,
        robot_body_linear_velocity_world=torch.zeros((batch, bodies, 3)),
        robot_body_angular_velocity_world=torch.zeros((batch, bodies, 3)),
        selected_assignment_mask=assignment,
        allow_selected_object_contact=allow,
    )
    assert evidence.prohibited_object_contact.tolist() == [True, False]
    assert evidence.prohibited_environment_contact.tolist() == [False, True]
    assert evidence.prohibited_collision.tolist() == [True, True]


class _Composer:
    def __init__(self) -> None:
        self.kwargs = None

    def set_forces_and_torques_index(self, **kwargs) -> None:
        self.kwargs = kwargs


class _Robot:
    def __init__(self) -> None:
        self.permanent_wrench_composer = _Composer()
        self.calls: list[tuple[str, torch.Tensor, torch.Tensor]] = []

    def _record(self, label: str, *, target, joint_ids) -> None:
        self.calls.append((label, target.clone(), joint_ids.clone()))

    def set_joint_position_target_index(self, *, target, joint_ids) -> None:
        self._record("position", target=target, joint_ids=joint_ids)

    def set_joint_velocity_target_index(self, *, target, joint_ids) -> None:
        self._record("velocity", target=target, joint_ids=joint_ids)

    def set_joint_effort_target_index(self, *, target, joint_ids) -> None:
        self._record("effort", target=target, joint_ids=joint_ids)


def test_tensor_isaac_io_applies_qp_rotors_gimbals_and_policy_dock_intent() -> None:
    physical, _, io = _fixture()
    batch = 2
    rotors = io.rigid_body_builder.rotor_count
    dock_ids = tuple(
        sorted(
            {
                str(port.mechanical_limits["mechanism_joint_id"])
                for port in physical.dock_ports
            }
        )
    )
    slots = len(dock_ids)
    shape = (batch, io.module_count, slots)
    command = Order9TensorPolicyCommand(
        desired_body_pose_world=torch.zeros((batch, 7)),
        desired_body_twist=torch.zeros((batch, 6)),
        residual_wrench_body=torch.zeros((batch, 6)),
        joint_position_targets_rad=torch.full(shape, 0.1),
        joint_velocity_targets_radps=torch.full(shape, 0.2),
        joint_torque_bias_nm=torch.full(shape, 0.3),
        joint_target_mask=torch.ones(shape, dtype=torch.bool),
        module_ids=io.module_ids,
        local_joint_ids=dock_ids,
    )
    allocation = SimpleNamespace(
        rotor_thrusts_n=torch.full((batch, rotors), 2.0),
        vectoring_joint_targets_rad=torch.full((batch, rotors), 0.05),
    )
    robot = _Robot()

    io.apply(
        robot=robot,
        policy_command=command,
        controller_result=SimpleNamespace(allocation=allocation),
    )

    wrench = robot.permanent_wrench_composer.kwargs
    assert wrench is not None
    assert wrench["forces"].shape == (batch, rotors, 3)
    assert wrench["torques"].shape == (batch, rotors, 3)
    assert len(robot.calls) == 4
    assert robot.calls[0][0] == "position"  # QP gimbal target
    assert [item[0] for item in robot.calls[1:]] == [
        "position",
        "velocity",
        "effort",
    ]
    assert robot.calls[-1][1].shape == (batch, io.module_count * slots)


def test_tensor_isaac_io_tracks_only_planner_selected_anchor_pairs() -> None:
    physical, morphology, full = _fixture()
    selected = (full.selected_anchor_ids[-1], full.selected_anchor_ids[0])
    io = Order9TensorIsaacIO(
        morphology_graph=morphology,
        physical_model=physical,
        robot_body_names=full.robot_body_names,
        robot_joint_names=full.robot_joint_names,
        selected_anchor_ids=selected,
    )

    assert io.selected_anchor_ids == selected
    assert io.selected_anchor_count == 2
    assert io.selected_anchor_body_names == (
        full.selected_anchor_body_names[-1],
        full.selected_anchor_body_names[0],
    )


def test_tensor_isaac_io_zero_fills_physx_merged_fixed_joints() -> None:
    physical, morphology, full = _fixture()
    fixed = {joint.joint_id for joint in physical.joints if joint.joint_type == "fixed"}
    joint_names = tuple(
        name
        for name in full.robot_joint_names
        if name.split("__", 1)[1] not in fixed
    )
    io = Order9TensorIsaacIO(
        morphology_graph=morphology,
        physical_model=physical,
        robot_body_names=full.robot_body_names,
        robot_joint_names=joint_names,
    )
    batch = 1
    body_pose = torch.zeros((batch, len(io.robot_body_names), 7))
    body_pose[..., 6] = 1.0
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_pose_w=body_pose,
            body_lin_vel_w=torch.zeros((batch, len(io.robot_body_names), 3)),
            body_ang_vel_w=torch.zeros((batch, len(io.robot_body_names), 3)),
            joint_pos=torch.ones((batch, len(joint_names))),
            joint_vel=torch.ones((batch, len(joint_names))),
            root_pose_w=body_pose[:, 0],
            root_lin_vel_w=torch.zeros((batch, 3)),
            root_ang_vel_w=torch.zeros((batch, 3)),
        )
    )
    obj = SimpleNamespace(
        data=SimpleNamespace(
            root_pose_w=body_pose[:, 0],
            root_lin_vel_w=torch.zeros((batch, 3)),
            root_ang_vel_w=torch.zeros((batch, 3)),
        )
    )

    state = io.gather_state(robot=robot, object_asset=obj)

    fixed_indices = [
        index for index, name in enumerate(io.local_joint_ids) if name in fixed
    ]
    assert fixed_indices
    assert not state.local_joint_positions_rad[:, :, fixed_indices].any()
    assert not state.local_joint_velocities_radps[:, :, fixed_indices].any()
