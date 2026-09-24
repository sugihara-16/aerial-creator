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
        self._constant_cache = {}
        self._command_index_cache = {}
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

    def _constants(self, device, dtype):
        """Immutable topology tensors; observed values are never cached."""
        key = (torch.device(device), dtype)
        if key not in self._constant_cache:
            index = lambda values: torch.tensor(values, device=device, dtype=torch.long)
            local = index(self.local_joint_indices)
            filter_index = {name: i for i, name in enumerate(self.object_filter_body_names)}
            self._constant_cache[key] = dict(
                module=index(self.module_body_indices), local=local.clamp_min(0),
                local_present=(local >= 0).unsqueeze(0),
                body_by_filter=index([filter_index[name] for name in self.robot_body_names]),
                selected_filter=index(self.selected_anchor_filter_indices),
                selected_body=index(self.selected_anchor_body_indices),
                axes=torch.tensor(self.rotor_thrust_axes_local, device=device, dtype=dtype),
                coefficients=torch.tensor(self.rotor_reaction_coefficients, device=device, dtype=dtype),
                rotor=index(self.rotor_body_indices).to(torch.int32),
                vectoring=index(self.vectoring_joint_indices).to(torch.int32),
            )
        return self._constant_cache[key]

    def gather_state(self, *, robot: Any, object_asset: Any) -> Order9TensorIsaacState:
        body_pose = _torch(robot.data.body_pose_w)
        body_linear = _torch(robot.data.body_lin_vel_w)
        body_angular = _torch(robot.data.body_ang_vel_w)
        joint_position = _torch(robot.data.joint_pos)
        joint_velocity = _torch(robot.data.joint_vel)
        constants = self._constants(body_pose.device, body_pose.dtype)
        module_indices = constants["module"]
        joint_constants = self._constants(joint_position.device, joint_position.dtype)
        local_present = joint_constants["local_present"]
        safe_local_indices = joint_constants["local"]
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
        local_q = torch.where(local_present, local_q, 0.0)
        local_qdot = torch.where(local_present, local_qdot, 0.0)
        root_pose = _torch(robot.data.root_pose_w)
        root_twist = torch.cat(
            (
                _torch(robot.data.root_lin_vel_w),
                _torch(robot.data.root_ang_vel_w),
            ),
            dim=-1,
        )
        object_pose_source = getattr(object_asset.data, "root_com_pose_w", None)
        if object_pose_source is None:
            object_pose_source = object_asset.data.root_pose_w
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
        constants = self._constants(matrix.device, matrix.dtype)
        body_by_filter = constants["body_by_filter"]
        object_by_body = matrix.index_select(1, body_by_filter)
        selected_filter = constants["selected_filter"]
        selected_body = constants["selected_body"]
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

    def reduce_contact_wrenches(
        self,
        *,
        normal_force_magnitudes_n: torch.Tensor,
        normal_points_world: torch.Tensor,
        normal_vectors_world: torch.Tensor,
        normal_contact_counts: torch.Tensor,
        normal_contact_starts: torch.Tensor,
        friction_forces_world: torch.Tensor,
        friction_points_world: torch.Tensor,
        friction_contact_counts: torch.Tensor,
        friction_contact_starts: torch.Tensor,
        contact_frame_pose_world: torch.Tensor,
        selected_assignment_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate raw contact patches into assignment-frame 6D wrenches.

        The result is the wrench exerted by each selected robot anchor on the
        object, expressed about the corresponding ``ContactCandidate`` frame.
        This raw PhysX path is privileged reward/critic evidence only.
        """

        batch_size = contact_frame_pose_world.shape[0]
        selected_count = len(self.selected_anchor_filter_indices)
        filter_count = len(self.object_filter_body_names)
        if contact_frame_pose_world.shape != (batch_size, selected_count, 7):
            raise ValueError("Order9 contact-frame pose shape differs")
        if selected_assignment_mask.shape != (batch_size, selected_count):
            raise ValueError("Order9 selected wrench mask shape differs")
        for name, value in (
            ("normal_contact_counts", normal_contact_counts),
            ("normal_contact_starts", normal_contact_starts),
            ("friction_contact_counts", friction_contact_counts),
            ("friction_contact_starts", friction_contact_starts),
        ):
            if value.shape != (batch_size, filter_count):
                raise ValueError(f"Order9 {name} shape differs")
        normal_force = normal_force_magnitudes_n.reshape(-1)
        if normal_points_world.shape != (normal_force.shape[0], 3) or (
            normal_vectors_world.shape != normal_points_world.shape
        ):
            raise ValueError("Order9 raw normal-contact buffer shape differs")
        if friction_forces_world.ndim != 2 or friction_forces_world.shape[-1] != 3 or (
            friction_points_world.shape != friction_forces_world.shape
        ):
            raise ValueError("Order9 raw friction-contact buffer shape differs")
        # PhysX may leave unused raw-buffer rows as NaN.  Finiteness is checked
        # after the active start/count ranges have been gathered below.
        if not bool(torch.isfinite(contact_frame_pose_world).all()):
            raise ValueError("Order9 contact-frame evidence is non-finite")
        device = contact_frame_pose_world.device
        dtype = contact_frame_pose_world.dtype
        selected_filter = self._constants(device, dtype)["selected_filter"]
        origins = contact_frame_pose_world[..., :3]
        normal_world = _aggregate_selected_patch_wrenches(
            forces_world=(
                normal_force.to(device=device, dtype=dtype).unsqueeze(-1)
                * normal_vectors_world.to(device=device, dtype=dtype)
            ),
            points_world=normal_points_world.to(device=device, dtype=dtype),
            counts=normal_contact_counts.to(device=device),
            starts=normal_contact_starts.to(device=device),
            selected_filter_indices=selected_filter,
            reference_origins_world=origins,
        )
        friction_world = _aggregate_selected_patch_wrenches(
            forces_world=friction_forces_world.to(device=device, dtype=dtype),
            points_world=friction_points_world.to(device=device, dtype=dtype),
            counts=friction_contact_counts.to(device=device),
            starts=friction_contact_starts.to(device=device),
            selected_filter_indices=selected_filter,
            reference_origins_world=origins,
        )
        wrench_world = normal_world + friction_world
        world_from_contact = _normalized_quaternion(
            contact_frame_pose_world[..., 3:7]
        )
        contact_from_world = world_from_contact.clone()
        contact_from_world[..., :3].neg_()
        wrench_contact = torch.cat(
            (
                _quaternion_rotate(contact_from_world, wrench_world[..., :3]),
                _quaternion_rotate(contact_from_world, wrench_world[..., 3:6]),
            ),
            dim=-1,
        )
        wrench_contact = torch.where(
            selected_assignment_mask.unsqueeze(-1),
            wrench_contact,
            torch.zeros_like(wrench_contact),
        )
        if not bool(torch.isfinite(wrench_contact).all()):
            raise ValueError("Order9 reduced contact wrench is non-finite")
        return wrench_contact

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
        constants = self._constants(device, dtype)
        axes = constants["axes"]
        coefficients = constants["coefficients"]
        forces = thrust.unsqueeze(-1) * axes.unsqueeze(0)
        torques = thrust.unsqueeze(-1) * coefficients.reshape(1, -1, 1) * axes.unsqueeze(0)
        robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=forces,
            torques=torques,
            body_ids=constants["rotor"],
            is_global=False,
        )
        robot.set_joint_position_target_index(
            target=allocation.vectoring_joint_targets_rad,
            joint_ids=constants["vectoring"],
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
        key = (tuple(command.local_joint_ids), torch.device(device))
        if key not in self._command_index_cache:
            local_lookup = {joint_id: i for i, joint_id in enumerate(self.local_joint_ids)}
            if any(joint_id not in local_lookup for joint_id in command.local_joint_ids):
                raise ValueError("Order9 policy command references unknown local joint")
            indices = [self.local_joint_indices[module][local_lookup[joint_id]]
                       for module in range(len(self.module_ids)) for joint_id in command.local_joint_ids]
            self._command_index_cache[key] = torch.tensor(indices, device=device, dtype=torch.int32)
        joint_ids = self._command_index_cache[key]
        expected_shape = (batch_size, len(self.module_ids), len(command.local_joint_ids))
        values = (command.joint_position_targets_rad, command.joint_velocity_targets_radps,
                  command.joint_torque_bias_nm, command.joint_target_mask)
        if any(value.shape != expected_shape for value in values):
            raise ValueError("Order9 policy joint target shape differs")
        q, qdot, effort = [value.reshape(batch_size, -1).to(device=device, dtype=dtype)
                          for value in values[:3]]
        mask = command.joint_target_mask.reshape(batch_size, -1)
        if not bool(mask.all()):
            raise ValueError("Order9 policy joint target mask is incomplete")
        robot.set_joint_position_target_index(target=q, joint_ids=joint_ids)
        robot.set_joint_velocity_target_index(target=qdot, joint_ids=joint_ids)
        robot.set_joint_effort_target_index(target=effort, joint_ids=joint_ids)


def _torch(value: Any) -> torch.Tensor:
    return value.torch if hasattr(value, "torch") else value


def _aggregate_selected_patch_wrenches(
    *,
    forces_world: torch.Tensor,
    points_world: torch.Tensor,
    counts: torch.Tensor,
    starts: torch.Tensor,
    selected_filter_indices: torch.Tensor,
    reference_origins_world: torch.Tensor,
) -> torch.Tensor:
    batch_size, selected_count, _ = reference_origins_world.shape
    selected_counts = counts.index_select(1, selected_filter_indices).reshape(-1).long()
    selected_starts = starts.index_select(1, selected_filter_indices).reshape(-1).long()
    invalid_offsets, invalid_range, patch_count = torch.stack((
        ((selected_counts < 0) | (selected_starts < 0)).any().long(),
        ((selected_counts > 0) & (selected_starts + selected_counts > forces_world.shape[0])).any().long(),
        selected_counts.sum(),
    )).tolist()
    if invalid_offsets:
        raise ValueError("Order9 raw contact offsets must be non-negative")
    if invalid_range:
        raise ValueError("Order9 raw contact buffer range is invalid")
    pair_count = batch_size * selected_count
    output = torch.zeros(
        (pair_count, 6), device=forces_world.device, dtype=forces_world.dtype
    )
    if patch_count == 0:
        return output.reshape(batch_size, selected_count, 6)
    pair_ids = torch.repeat_interleave(
        torch.arange(pair_count, device=counts.device, dtype=torch.long),
        selected_counts, output_size=patch_count,
    )
    repeated_starts = torch.repeat_interleave(selected_starts, selected_counts, output_size=patch_count)
    repeated_offsets = torch.repeat_interleave(
        torch.cumsum(selected_counts, dim=0) - selected_counts,
        selected_counts, output_size=patch_count,
    )
    patch_indices = repeated_starts + (
        torch.arange(pair_ids.numel(), device=counts.device, dtype=torch.long)
        - repeated_offsets
    )
    force = forces_world.index_select(0, patch_indices)
    point = points_world.index_select(0, patch_indices)
    if not bool(torch.isfinite(force).all()) or not bool(torch.isfinite(point).all()):
        raise ValueError("Order9 active raw contact patch is non-finite")
    origin = reference_origins_world.reshape(pair_count, 3).index_select(0, pair_ids)
    moment = torch.cross(point - origin, force, dim=-1)
    output.index_add_(0, pair_ids, torch.cat((force, moment), dim=-1))
    return output.reshape(batch_size, selected_count, 6)


def _quaternion_rotate(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    q_xyz = quaternion[..., :3]
    q_w = quaternion[..., 3:4]
    first = torch.cross(q_xyz, vector, dim=-1)
    return vector + 2.0 * torch.cross(q_xyz, first + q_w * vector, dim=-1)


def _normalized_quaternion(value: torch.Tensor) -> torch.Tensor:
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)


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
