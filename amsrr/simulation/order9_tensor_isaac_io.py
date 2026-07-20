from __future__ import annotations

"""Tensor-only Isaac I/O for one topology-bucketed Order 9 rollout.

The class deliberately owns only name/index resolution, state gathering,
actuator application, and privileged contact reduction.  Policy evaluation,
QPID/QP allocation, reward, and phase transitions remain separate so raw
contact truth cannot accidentally enter the actor feature path.
"""

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from amsrr.controllers.batched_qpid_controller import BatchedQPIDResult
from amsrr.controllers.batched_rigid_body_model import (
    BatchedRigidBodyControlModelBuilder,
)
from amsrr.policies.order9_tensor_command_decoder import (
    Order9TensorPolicyCommand,
)
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.physical_model import PhysicalModel


ORDER9_TENSOR_ISAAC_IO_VERSION = "order9_tensor_isaac_io_v1"


@dataclass(frozen=True)
class Order9TensorIsaacState:
    module_pose_world: torch.Tensor
    module_twist_world: torch.Tensor
    local_joint_positions_rad: torch.Tensor
    local_joint_velocities_radps: torch.Tensor
    robot_root_pose_world: torch.Tensor
    robot_root_twist_world: torch.Tensor
    object_pose_world: torch.Tensor
    object_twist_world: torch.Tensor


@dataclass(frozen=True)
class Order9TensorContactEvidence:
    selected_contact_forces_world: torch.Tensor
    selected_link_twist_world: torch.Tensor
    selected_contact_mask: torch.Tensor
    object_contact_forces_by_robot_body_world: torch.Tensor
    prohibited_object_contact: torch.Tensor
    prohibited_environment_contact: torch.Tensor
    prohibited_collision: torch.Tensor


class Order9TensorIsaacIO:
    """Resolve one copied articulation layout and keep every hot-path op batched."""

    io_version = ORDER9_TENSOR_ISAAC_IO_VERSION

    def __init__(
        self,
        *,
        morphology_graph: MorphologyGraph,
        physical_model: PhysicalModel,
        robot_body_names: Sequence[str],
        robot_joint_names: Sequence[str],
        object_filter_body_names: Sequence[str] | None = None,
        selected_anchor_ids: Sequence[int] | None = None,
        contact_force_threshold_n: float = 0.5,
    ) -> None:
        morphology_graph.validate()
        physical_model.validate()
        if contact_force_threshold_n <= 0.0:
            raise ValueError("Order9 contact threshold must be positive")
        self.morphology_graph = MorphologyGraph.from_dict(
            morphology_graph.to_dict()
        )
        self.physical_model = PhysicalModel.from_dict(physical_model.to_dict())
        self.robot_body_names = tuple(str(value) for value in robot_body_names)
        self.robot_joint_names = tuple(str(value) for value in robot_joint_names)
        if (
            not self.robot_body_names
            or len(set(self.robot_body_names)) != len(self.robot_body_names)
            or not self.robot_joint_names
            or len(set(self.robot_joint_names)) != len(self.robot_joint_names)
        ):
            raise ValueError("Order9 Isaac body/joint names must be unique")
        self.object_filter_body_names = tuple(
            self.robot_body_names
            if object_filter_body_names is None
            else (str(value) for value in object_filter_body_names)
        )
        if set(self.object_filter_body_names) != set(self.robot_body_names):
            raise ValueError(
                "Order9 object contact filters must cover every robot body exactly"
            )
        self.contact_force_threshold_n = float(contact_force_threshold_n)
        self.rigid_body_builder = BatchedRigidBodyControlModelBuilder(
            self.morphology_graph, self.physical_model
        )
        self.module_ids = self.rigid_body_builder.module_ids
        self.local_joint_ids = self.rigid_body_builder.local_joint_ids
        body_index = {name: index for index, name in enumerate(self.robot_body_names)}
        joint_index = {
            name: index for index, name in enumerate(self.robot_joint_names)
        }
        base_link = _module_frame_link_id(self.physical_model)
        self.module_body_indices = tuple(
            _exact_index(body_index, _combined_name(module_id, base_link), "module body")
            for module_id in self.module_ids
        )
        self.local_joint_indices = tuple(
            tuple(
                _joint_index_or_fixed_zero(
                    joint_index,
                    _combined_name(module_id, local_joint_id),
                    joint_type=next(
                        joint.joint_type
                        for joint in self.physical_model.joints
                        if joint.joint_id == local_joint_id
                    ),
                )
                for local_joint_id in self.local_joint_ids
            )
            for module_id in self.module_ids
        )
        all_anchor_rows = sorted(
            self.morphology_graph.robot_anchors,
            key=lambda anchor: anchor.anchor_id,
        )
        if not all_anchor_rows:
            raise ValueError("Order9 topology bucket requires robot anchors")
        anchor_by_id = {anchor.anchor_id: anchor for anchor in all_anchor_rows}
        requested_anchor_ids = (
            tuple(anchor.anchor_id for anchor in all_anchor_rows)
            if selected_anchor_ids is None
            else tuple(int(value) for value in selected_anchor_ids)
        )
        if not requested_anchor_ids or len(set(requested_anchor_ids)) != len(
            requested_anchor_ids
        ):
            raise ValueError("Order9 selected anchor ids must be non-empty and unique")
        unknown_anchor_ids = sorted(set(requested_anchor_ids) - set(anchor_by_id))
        if unknown_anchor_ids:
            raise ValueError(
                f"Order9 selected anchor ids are not in morphology: {unknown_anchor_ids}"
            )
        anchor_rows = tuple(anchor_by_id[value] for value in requested_anchor_ids)
        self.selected_anchor_ids = tuple(anchor.anchor_id for anchor in anchor_rows)
        self.selected_anchor_body_names = tuple(
            _combined_name(anchor.module_id, _require_link(anchor.link_id))
            for anchor in anchor_rows
        )
        self.selected_anchor_body_indices = tuple(
            _exact_index(body_index, name, "anchor body")
            for name in self.selected_anchor_body_names
        )
        filter_index = {
            name: index for index, name in enumerate(self.object_filter_body_names)
        }
        self.selected_anchor_filter_indices = tuple(
            _exact_index(filter_index, name, "anchor contact filter")
            for name in self.selected_anchor_body_names
        )
        rotor_specs = self.rigid_body_builder._rotor_specs
        self.rotor_body_indices = tuple(
            _exact_index(
                body_index,
                _combined_name(module_id, rotor.rotor_id),
                "rotor body",
            )
            for module_id, rotor in rotor_specs
        )
        self.rotor_thrust_axes_local = tuple(
            tuple(float(value) for value in rotor.thrust_axis_local)
            for _, rotor in rotor_specs
        )
        self.rotor_reaction_coefficients = tuple(
            float(rotor.reaction_torque_coeff_nm_per_n)
            for _, rotor in rotor_specs
        )
        self.vectoring_joint_indices = tuple(
            _exact_index(
                joint_index,
                _combined_name(module_id, rotor.vectoring_joint_ids[0]),
                "vectoring joint",
            )
            for module_id, rotor in rotor_specs
        )

    @property
    def module_count(self) -> int:
        return len(self.module_ids)

    @property
    def selected_anchor_count(self) -> int:
        return len(self.selected_anchor_ids)

    def gather_state(self, *, robot: Any, object_asset: Any) -> Order9TensorIsaacState:
        body_pose = _torch(robot.data.body_pose_w)
        body_linear = _torch(robot.data.body_lin_vel_w)
        body_angular = _torch(robot.data.body_ang_vel_w)
        joint_position = _torch(robot.data.joint_pos)
        joint_velocity = _torch(robot.data.joint_vel)
        module_indices = torch.tensor(
            self.module_body_indices, device=body_pose.device, dtype=torch.long
        )
        local_indices = torch.tensor(
            self.local_joint_indices,
            device=joint_position.device,
            dtype=torch.long,
        )
        local_present = local_indices >= 0
        safe_local_indices = local_indices.clamp_min(0)
        module_pose = body_pose.index_select(1, module_indices)
        module_twist = torch.cat(
            (
                body_linear.index_select(1, module_indices),
                body_angular.index_select(1, module_indices),
            ),
            dim=-1,
        )
        local_q = joint_position[:, safe_local_indices]
        local_qdot = joint_velocity[:, safe_local_indices]
        local_q = torch.where(local_present.unsqueeze(0), local_q, 0.0)
        local_qdot = torch.where(local_present.unsqueeze(0), local_qdot, 0.0)
        root_pose = _torch(robot.data.root_pose_w)
        root_twist = torch.cat(
            (
                _torch(robot.data.root_lin_vel_w),
                _torch(robot.data.root_ang_vel_w),
            ),
            dim=-1,
        )
        object_pose_source = getattr(
            object_asset.data, "root_com_pose_w", object_asset.data.root_pose_w
        )
        object_velocity_source = getattr(
            object_asset.data, "root_com_vel_w", None
        )
        object_pose = _torch(object_pose_source)
        object_twist = (
            _torch(object_velocity_source)
            if object_velocity_source is not None
            else torch.cat(
                (
                    _torch(object_asset.data.root_lin_vel_w),
                    _torch(object_asset.data.root_ang_vel_w),
                ),
                dim=-1,
            )
        )
        return Order9TensorIsaacState(
            module_pose_world=module_pose,
            module_twist_world=module_twist,
            local_joint_positions_rad=local_q,
            local_joint_velocities_radps=local_qdot,
            robot_root_pose_world=root_pose,
            robot_root_twist_world=root_twist,
            object_pose_world=object_pose,
            object_twist_world=object_twist,
        )

    def reduce_contacts(
        self,
        *,
        robot_net_contact_forces_world: torch.Tensor,
        object_force_matrix_world: torch.Tensor,
        robot_body_linear_velocity_world: torch.Tensor,
        robot_body_angular_velocity_world: torch.Tensor,
        selected_assignment_mask: torch.Tensor,
        allow_selected_object_contact: torch.Tensor,
    ) -> Order9TensorContactEvidence:
        """Separate intended robot-object pairs from every other collision.

        ``object_force_matrix_world`` is force on the object, indexed by the
        exact robot-body filter order.  Adding it to the corresponding net
        force on each robot body cancels object contact and leaves support or
        other-environment contact.  Self collision is disabled by the scene.
        """

        batch_size = robot_net_contact_forces_world.shape[0]
        body_count = len(self.robot_body_names)
        if robot_net_contact_forces_world.shape != (batch_size, body_count, 3):
            raise ValueError("Order9 robot net contact force shape differs")
        matrix = object_force_matrix_world
        if matrix.ndim == 4 and matrix.shape[1] == 1:
            matrix = matrix[:, 0]
        if matrix.shape != (batch_size, body_count, 3):
            raise ValueError("Order9 object contact force matrix shape differs")
        if selected_assignment_mask.shape != (
            batch_size,
            self.selected_anchor_count,
        ):
            raise ValueError("Order9 selected assignment mask shape differs")
        if allow_selected_object_contact.shape != (batch_size,):
            raise ValueError("Order9 allowed-contact phase mask shape differs")
        if robot_body_linear_velocity_world.shape != (
            batch_size,
            body_count,
            3,
        ) or robot_body_angular_velocity_world.shape != (
            batch_size,
            body_count,
            3,
        ):
            raise ValueError("Order9 robot body velocity shape differs")
        # Convert the object's filter order to robot body order once per call.
        body_by_filter = torch.tensor(
            [self.object_filter_body_names.index(name) for name in self.robot_body_names],
            device=matrix.device,
            dtype=torch.long,
        )
        object_by_body = matrix.index_select(1, body_by_filter)
        selected_filter = torch.tensor(
            self.selected_anchor_filter_indices,
            device=matrix.device,
            dtype=torch.long,
        )
        selected_body = torch.tensor(
            self.selected_anchor_body_indices,
            device=matrix.device,
            dtype=torch.long,
        )
        selected_forces = -matrix.index_select(1, selected_filter)
        selected_twist = torch.cat(
            (
                robot_body_linear_velocity_world.index_select(1, selected_body),
                robot_body_angular_velocity_world.index_select(1, selected_body),
            ),
            dim=-1,
        )
        threshold = self.contact_force_threshold_n
        active_selected = selected_assignment_mask & (
            torch.linalg.vector_norm(selected_forces, dim=-1) >= threshold
        )
        allowed_body = torch.zeros(
            (batch_size, body_count),
            device=matrix.device,
            dtype=torch.bool,
        )
        allowed_selected = selected_assignment_mask & allow_selected_object_contact[:, None]
        allowed_body.scatter_(1, selected_body[None].expand(batch_size, -1), allowed_selected)
        object_active = torch.linalg.vector_norm(object_by_body, dim=-1) >= threshold
        prohibited_object = (object_active & ~allowed_body).any(dim=-1)
        environment_force = robot_net_contact_forces_world + object_by_body
        prohibited_environment = (
            torch.linalg.vector_norm(environment_force, dim=-1) >= threshold
        ).any(dim=-1)
        return Order9TensorContactEvidence(
            selected_contact_forces_world=selected_forces,
            selected_link_twist_world=selected_twist,
            selected_contact_mask=active_selected,
            object_contact_forces_by_robot_body_world=object_by_body,
            prohibited_object_contact=prohibited_object,
            prohibited_environment_contact=prohibited_environment,
            prohibited_collision=prohibited_object | prohibited_environment,
        )

    def apply(
        self,
        *,
        robot: Any,
        policy_command: Order9TensorPolicyCommand,
        controller_result: BatchedQPIDResult,
    ) -> None:
        allocation = controller_result.allocation
        thrust = allocation.rotor_thrusts_n
        batch_size, rotor_count = thrust.shape
        if rotor_count != len(self.rotor_body_indices):
            raise ValueError("Order9 allocation rotor count differs from Isaac layout")
        device, dtype = thrust.device, thrust.dtype
        axes = torch.tensor(
            self.rotor_thrust_axes_local, device=device, dtype=dtype
        )
        coefficients = torch.tensor(
            self.rotor_reaction_coefficients, device=device, dtype=dtype
        )
        forces = thrust.unsqueeze(-1) * axes.unsqueeze(0)
        torques = thrust.unsqueeze(-1) * coefficients.reshape(1, -1, 1) * axes.unsqueeze(0)
        robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=forces,
            torques=torques,
            body_ids=torch.tensor(
                self.rotor_body_indices, device=device, dtype=torch.int32
            ),
            is_global=False,
        )
        robot.set_joint_position_target_index(
            target=allocation.vectoring_joint_targets_rad,
            joint_ids=torch.tensor(
                self.vectoring_joint_indices, device=device, dtype=torch.int32
            ),
        )
        self._apply_policy_joints(
            robot=robot,
            command=policy_command,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

    def _apply_policy_joints(
        self,
        *,
        robot: Any,
        command: Order9TensorPolicyCommand,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if command.module_ids != self.module_ids:
            raise ValueError("Order9 policy command module order differs")
        if not command.local_joint_ids:
            return
        local_lookup = {
            joint_id: index for index, joint_id in enumerate(self.local_joint_ids)
        }
        indices: list[int] = []
        command_slots: list[tuple[int, int]] = []
        for module_index, module_id in enumerate(self.module_ids):
            for command_index, joint_id in enumerate(command.local_joint_ids):
                if joint_id not in local_lookup:
                    raise ValueError("Order9 policy command references unknown local joint")
                indices.append(self.local_joint_indices[module_index][local_lookup[joint_id]])
                command_slots.append((module_index, command_index))
        q = torch.stack(
            [command.joint_position_targets_rad[:, module, slot] for module, slot in command_slots],
            dim=1,
        ).to(device=device, dtype=dtype)
        qdot = torch.stack(
            [command.joint_velocity_targets_radps[:, module, slot] for module, slot in command_slots],
            dim=1,
        ).to(device=device, dtype=dtype)
        effort = torch.stack(
            [command.joint_torque_bias_nm[:, module, slot] for module, slot in command_slots],
            dim=1,
        ).to(device=device, dtype=dtype)
        mask = torch.stack(
            [command.joint_target_mask[:, module, slot] for module, slot in command_slots],
            dim=1,
        )
        if mask.shape != (batch_size, len(indices)) or not bool(mask.all()):
            raise ValueError("Order9 policy joint target mask is incomplete")
        joint_ids = torch.tensor(indices, device=device, dtype=torch.int32)
        robot.set_joint_position_target_index(target=q, joint_ids=joint_ids)
        robot.set_joint_velocity_target_index(target=qdot, joint_ids=joint_ids)
        robot.set_joint_effort_target_index(target=effort, joint_ids=joint_ids)


def _torch(value: Any) -> torch.Tensor:
    return value.torch if hasattr(value, "torch") else value


def _combined_name(module_id: int, local_id: str) -> str:
    return f"module_{int(module_id)}__{local_id}"


def _module_frame_link_id(physical_model: PhysicalModel) -> str:
    raw = physical_model.metadata.get("baselink", {})
    if isinstance(raw, dict):
        value = str(raw.get("name", "fc"))
    else:
        value = "fc"
    if not value:
        raise ValueError("Order9 PhysicalModel module frame is empty")
    return value


def _exact_index(values: dict[str, int], name: str, label: str) -> int:
    if name not in values:
        raise ValueError(f"Order9 cannot resolve {label} {name!r}")
    return values[name]


def _joint_index_or_fixed_zero(
    values: dict[str, int], name: str, *, joint_type: str
) -> int:
    if name in values:
        return values[name]
    if joint_type == "fixed":
        # Isaac/PhysX may merge massless fixed-joint children.  The rigid-body
        # model still retains that schema slot, whose generalized coordinate is
        # identically zero.
        return -1
    raise ValueError(f"Order9 cannot resolve module-local joint {name!r}")


def _require_link(value: str | None) -> str:
    if not value:
        raise ValueError("Order9 robot anchor lacks a rigid-body link id")
    return value


__all__ = [
    "ORDER9_TENSOR_ISAAC_IO_VERSION",
    "Order9TensorContactEvidence",
    "Order9TensorIsaacIO",
    "Order9TensorIsaacState",
]
