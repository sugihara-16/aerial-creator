from __future__ import annotations

"""Isaac Lab tensor/contact-view adapter for the Order 9 copied runtime.

The module imports no Isaac package at import time.  The persistent worker
constructs it only after ``AppLauncher`` has started Kit and injects the live
scene objects plus Torch/Warp modules.  This keeps ordinary unit tests and
training-data tools usable without Isaac while retaining real-PhysX evidence
in production.
"""

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from amsrr.controllers.controller_base import PayloadCoupling
from amsrr.controllers.isaac_controller_bridge import IsaacActuatorTargetRecord
from amsrr.feasibility.contact_wrench_hybrid import ShadowCollisionSample
from amsrr.feasibility.contact_wrench_shadow_metrics import MeasuredCandidateWrench
from amsrr.geometry.convex_clearance import (
    ConvexPolytope,
    OrientedBox,
    capsule_oriented_box_clearance,
    circumscribed_cylinder_polytope,
    oriented_box_clearance,
    oriented_box_polytope_clearance,
    sphere_oriented_box_clearance,
)
from amsrr.geometry.pose_math import inverse_pose, matvec, transform_from_pose
from amsrr.geometry.wrench import world_wrench_to_contact
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import ControllerStatus, InteractionKnot
from amsrr.schemas.runtime import (
    ModuleRuntimeState,
    ObjectRuntimeState,
    RuntimeObservation,
    TaskProgressState,
)
from amsrr.simulation.order9_object_task_runtime import (
    ORDER9_OBJECT_TASK_PHASES,
    Order9ObjectTaskRuntimeConfig,
)
from amsrr.simulation.order9_object_task_state import Order9IsaacStateSnapshot


ORDER9_ISAAC_SCENE_ADAPTER_VERSION = "order9_isaaclab_scene_adapter_v3"
_ACTIVE_CONTACT_STATES = frozenset({"attach", "maintain", "slide"})


@dataclass(frozen=True)
class Order9IsaacContactViewLayout:
    selected_sensor_body_names: tuple[str, ...]
    selected_anchor_ids: tuple[int, ...]
    all_sensor_body_names: tuple[str, ...]
    all_sensor_entity_ids: tuple[str, ...]
    all_filter_entity_ids: tuple[str, ...] = ("object", "support")

    def __post_init__(self) -> None:
        if (
            not self.selected_sensor_body_names
            or len(self.selected_sensor_body_names) != len(self.selected_anchor_ids)
            or len(set(self.selected_sensor_body_names))
            != len(self.selected_sensor_body_names)
            or len(set(self.selected_anchor_ids)) != len(self.selected_anchor_ids)
        ):
            raise ValueError("Order9 selected contact-view layout is invalid")
        if (
            not self.all_sensor_body_names
            or len(self.all_sensor_body_names) != len(self.all_sensor_entity_ids)
            or len(set(self.all_sensor_body_names)) != len(self.all_sensor_body_names)
            or len(self.all_filter_entity_ids) != 2
        ):
            raise ValueError("Order9 all-contact view layout is invalid")


class IsaacLabOrder9SceneAdapter:
    """Read/write one copied Isaac environment and reduce privileged evidence."""

    adapter_version = ORDER9_ISAAC_SCENE_ADAPTER_VERSION

    def __init__(
        self,
        *,
        sim: Any,
        robot: Any,
        object_asset: Any,
        morphology_graph: MorphologyGraph,
        physical_model: PhysicalModel,
        selected_contact_view: Any,
        all_contact_view: Any,
        contact_layout: Order9IsaacContactViewLayout,
        torch_module: Any,
        warp_module: Any,
        device: str,
        object_id: str,
        object_mass_kg: float,
        object_inertia_body: Sequence[float],
        object_friction: float,
        selected_gripper_friction: float,
        contact_stiffness_n_per_m: float,
        contact_damping_n_s_per_m: float,
        object_geometry_type: str,
        object_size_m: Sequence[float],
        support_top_z_m: float,
        support_center_world_m: Sequence[float],
        support_half_extents_m: Sequence[float],
        body_local_aabb_m: Mapping[str, Sequence[Sequence[float]]],
        actuator_readback: Mapping[str, object] | None = None,
        phase_config: Order9ObjectTaskRuntimeConfig | None = None,
        robot_on_target_force_sign: float = -1.0,
        contact_force_threshold_n: float = 1.0e-4,
    ) -> None:
        morphology_graph.validate()
        physical_model.validate()
        self.sim = sim
        self.robot = robot
        self.object_asset = object_asset
        self.morphology_graph = morphology_graph
        self.physical_model = physical_model
        self.selected_contact_view = selected_contact_view
        self.all_contact_view = all_contact_view
        self.contact_layout = contact_layout
        self.torch = torch_module
        self.wp = warp_module
        self.device = str(device)
        self.object_id = str(object_id)
        self.object_mass_kg = float(object_mass_kg)
        self.object_inertia_body = tuple(float(value) for value in object_inertia_body)
        self.object_friction = float(object_friction)
        self.selected_gripper_friction = float(selected_gripper_friction)
        self.contact_stiffness_n_per_m = float(contact_stiffness_n_per_m)
        self.contact_damping_n_s_per_m = float(contact_damping_n_s_per_m)
        self.object_geometry_type = str(object_geometry_type)
        self.object_size_m = tuple(float(value) for value in object_size_m)
        self.support_top_z_m = float(support_top_z_m)
        self.support_center_world_m = tuple(
            float(value) for value in support_center_world_m
        )
        self.support_half_extents_m = tuple(
            float(value) for value in support_half_extents_m
        )
        self.body_local_aabb_m = {
            str(name): (
                tuple(float(value) for value in bounds[0]),
                tuple(float(value) for value in bounds[1]),
            )
            for name, bounds in body_local_aabb_m.items()
        }
        self.actuator_readback = (
            None if actuator_readback is None else dict(actuator_readback)
        )
        self.phase_config = phase_config or Order9ObjectTaskRuntimeConfig()
        self.phase_config.validate()
        self.robot_on_target_force_sign = float(robot_on_target_force_sign)
        self.contact_force_threshold_n = float(contact_force_threshold_n)
        self._simulation_time_s = 0.0
        self._phase_index = 0
        self._phase_elapsed_s = 0.0
        self._command_index = 0
        self._restored = False
        self._closed = False
        self._validate_configuration()

    def describe(self) -> dict[str, object]:
        return {
            "adapter_version": self.adapter_version,
            "joint_names": list(self.robot.joint_names),
            "body_names": list(self.robot.body_names),
            "object_id": self.object_id,
            "object_mass_kg": self.object_mass_kg,
            "object_inertia_body": list(self.object_inertia_body),
            "object_friction": self.object_friction,
            "selected_gripper_friction": self.selected_gripper_friction,
            "contact_stiffness_n_per_m": self.contact_stiffness_n_per_m,
            "contact_damping_n_s_per_m": self.contact_damping_n_s_per_m,
            "object_geometry_type": self.object_geometry_type,
            "object_size_m": list(self.object_size_m),
            "support_center_world_m": list(self.support_center_world_m),
            "support_half_extents_m": list(self.support_half_extents_m),
            "phase_count": len(ORDER9_OBJECT_TASK_PHASES),
            "selected_anchor_ids": list(self.contact_layout.selected_anchor_ids),
            "selected_sensor_body_names": list(
                self.contact_layout.selected_sensor_body_names
            ),
            "actuator_readback": self.actuator_readback,
        }

    def restore_snapshot(self, snapshot: Order9IsaacStateSnapshot) -> None:
        self._require_open()
        snapshot.validate()
        if tuple(snapshot.joint_names) != tuple(self.robot.joint_names):
            raise SchemaValidationError(
                "Order9 Isaac snapshot joint order differs from spawned articulation"
            )
        torch = self.torch
        root_pose = torch.tensor(
            [snapshot.robot_root_pose_world], dtype=torch.float32, device=self.device
        )
        root_twist = torch.tensor(
            [snapshot.robot_root_twist_world], dtype=torch.float32, device=self.device
        )
        joint_position = torch.tensor(
            [snapshot.joint_positions_rad], dtype=torch.float32, device=self.device
        )
        joint_velocity = torch.tensor(
            [snapshot.joint_velocities_radps], dtype=torch.float32, device=self.device
        )
        object_pose = torch.tensor(
            [snapshot.object_pose_world], dtype=torch.float32, device=self.device
        )
        object_twist = torch.tensor(
            [snapshot.object_twist_world], dtype=torch.float32, device=self.device
        )
        self.robot.write_root_pose_to_sim_index(root_pose=root_pose)
        self.robot.write_root_velocity_to_sim_index(root_velocity=root_twist)
        self.robot.write_joint_position_to_sim_index(position=joint_position)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_velocity)
        self.robot.set_joint_position_target_index(target=joint_position)
        self.robot.set_joint_velocity_target_index(
            target=torch.zeros_like(joint_velocity)
        )
        self.robot.set_joint_effort_target_index(
            target=torch.zeros_like(joint_velocity)
        )
        self.object_asset.write_root_pose_to_sim_index(root_pose=object_pose)
        self.object_asset.write_root_velocity_to_sim_index(root_velocity=object_twist)
        self.sim.forward()
        self.robot.update(0.0)
        self.object_asset.update(0.0)
        self._simulation_time_s = float(snapshot.simulation_time_s)
        self._phase_index = int(snapshot.phase_index)
        self._phase_elapsed_s = float(snapshot.phase_elapsed_s)
        self._command_index = int(snapshot.command_index)
        self._restored = True

    def capture_snapshot(self) -> Order9IsaacStateSnapshot:
        self._require_restored()
        snapshot = Order9IsaacStateSnapshot(
            simulation_time_s=float(self._simulation_time_s),
            robot_root_pose_world=_tensor_row(self.robot.data.root_pose_w),
            robot_root_twist_world=_root_twist(self.robot),
            joint_names=list(self.robot.joint_names),
            joint_positions_rad=_tensor_row(self.robot.data.joint_pos),
            joint_velocities_radps=_tensor_row(self.robot.data.joint_vel),
            object_id=self.object_id,
            object_pose_world=_object_pose(self.object_asset),
            object_twist_world=_object_twist(self.object_asset),
            phase_index=int(self._phase_index),
            phase_elapsed_s=float(self._phase_elapsed_s),
            command_index=int(self._command_index),
            metadata={
                "scene_adapter_version": self.adapter_version,
                "raw_contact_actor_input": False,
            },
        )
        snapshot.validate()
        return snapshot

    def actor_observation(
        self,
        *,
        morphology_graph: MorphologyGraph,
        controller_status: ControllerStatus,
        elapsed_s: float,
    ) -> RuntimeObservation:
        self._require_restored()
        del elapsed_s
        if morphology_graph.stable_hash() != self.morphology_graph.stable_hash():
            raise SchemaValidationError("Order9 scene observation morphology mismatch")
        module_states: list[ModuleRuntimeState] = []
        module_frame_link_id = _module_frame_link_id(self.physical_model)
        for module in sorted(morphology_graph.modules, key=lambda item: item.module_id):
            module_id = int(module.module_id)
            frame_name = _combined_name(
                module_id,
                module_frame_link_id,
            )
            body_index = _exact_name_index(self.robot.body_names, frame_name, "body")
            pose = _tensor_body_row(
                self.robot.data.body_pos_w, body_index
            ) + _tensor_body_row(self.robot.data.body_quat_w, body_index)
            linear_tensor = getattr(
                self.robot.data,
                "body_link_lin_vel_w",
                self.robot.data.body_lin_vel_w,
            )
            angular_tensor = getattr(
                self.robot.data,
                "body_link_ang_vel_w",
                self.robot.data.body_ang_vel_w,
            )
            local_q: dict[str, float] = {}
            local_qdot: dict[str, float] = {}
            for joint in self.physical_model.joints:
                name = _combined_name(module_id, joint.joint_id)
                if name not in self.robot.joint_names:
                    continue
                index = self.robot.joint_names.index(name)
                local_q[joint.joint_id] = _tensor_scalar(
                    self.robot.data.joint_pos, index
                )
                local_qdot[joint.joint_id] = _tensor_scalar(
                    self.robot.data.joint_vel, index
                )
            module_states.append(
                ModuleRuntimeState(
                    module_id=module_id,
                    pose_world=tuple(float(value) for value in pose),
                    twist_world=(
                        _tensor_body_row(linear_tensor, body_index)
                        + _tensor_body_row(angular_tensor, body_index)
                    ),
                    joint_positions=local_q,
                    joint_velocities=local_qdot,
                    health=1.0,
                )
            )
        phase = ORDER9_OBJECT_TASK_PHASES[self._phase_index]
        duration = float(self.phase_config.phase_duration_s[phase.value])
        observation = RuntimeObservation(
            time_s=float(self._simulation_time_s),
            morphology_graph=morphology_graph,
            module_states=module_states,
            object_states=[
                ObjectRuntimeState(
                    object_id=self.object_id,
                    pose_world=tuple(_object_pose(self.object_asset)),
                    twist_world=_object_twist(self.object_asset),
                )
            ],
            # Raw PhysX contact is intentionally returned only through the
            # privileged evidence methods below.
            contact_states=[],
            controller_status=ControllerStatus.from_dict(
                controller_status.to_dict()
            ),
            task_progress=TaskProgressState(
                phase_label=phase.value,
                progress_ratio=min(max(self._phase_elapsed_s / duration, 0.0), 1.0),
                success=False,
                failure_reason=None,
                metrics={
                    "phase_index": float(self._phase_index),
                    "phase_count": float(len(ORDER9_OBJECT_TASK_PHASES)),
                },
            ),
        )
        observation.validate()
        return observation

    def apply_actuator_targets(self, record: IsaacActuatorTargetRecord) -> int:
        self._require_restored()
        torch = self.torch
        rotor_by_id = {rotor.rotor_id: rotor for rotor in self.physical_model.rotors}
        forces: dict[int, list[float]] = {}
        torques: dict[int, list[float]] = {}
        position: dict[int, float] = {}
        velocity: dict[int, float] = {}
        effort: dict[int, float] = {}
        unresolved = 0
        for target in record.actuator_targets:
            module_id = int(target.metadata.get("module_id", -1))
            local_id = str(target.metadata.get("local_id", target.command_key))
            if module_id < 0:
                unresolved += 1
                continue
            if target.actuator_type == "rotor_thrust":
                rotor = rotor_by_id.get(local_id)
                name = _combined_name(module_id, local_id)
                if rotor is None or name not in self.robot.body_names:
                    unresolved += 1
                    continue
                index = self.robot.body_names.index(name)
                forces[index] = [
                    float(axis) * float(target.target_value)
                    for axis in rotor.thrust_axis_local
                ]
                torques[index] = [
                    float(axis)
                    * float(rotor.reaction_torque_coeff_nm_per_n)
                    * float(target.target_value)
                    for axis in rotor.thrust_axis_local
                ]
                continue
            name = _combined_name(module_id, local_id)
            if name not in self.robot.joint_names:
                unresolved += 1
                continue
            index = self.robot.joint_names.index(name)
            if target.actuator_type in {
                "vectoring_joint_position",
                "dock_joint_position",
                "joint_position",
            }:
                position[index] = float(target.target_value)
            elif target.actuator_type == "joint_velocity":
                velocity[index] = float(target.target_value)
            elif target.actuator_type in {"joint_effort", "joint_effort_bias"}:
                effort[index] = effort.get(index, 0.0) + float(target.target_value)
            else:
                unresolved += 1
        self.robot.permanent_wrench_composer.reset()
        if forces:
            body_ids = sorted(forces)
            self.robot.permanent_wrench_composer.set_forces_and_torques_index(
                forces=torch.tensor(
                    [[forces[index] for index in body_ids]],
                    dtype=torch.float32,
                    device=self.device,
                ),
                torques=torch.tensor(
                    [[torques[index] for index in body_ids]],
                    dtype=torch.float32,
                    device=self.device,
                ),
                body_ids=torch.tensor(body_ids, dtype=torch.int32, device=self.device),
                is_global=False,
            )
        _set_sparse_joint_target(
            self.robot,
            position,
            setter="set_joint_position_target_index",
            torch_module=torch,
            device=self.device,
        )
        _set_sparse_joint_target(
            self.robot,
            velocity,
            setter="set_joint_velocity_target_index",
            torch_module=torch,
            device=self.device,
        )
        _set_sparse_joint_target(
            self.robot,
            effort,
            setter="set_joint_effort_target_index",
            torch_module=torch,
            device=self.device,
        )
        self._command_index = int(record.command_index) + 1
        return unresolved

    def step(self, dt_s: float) -> None:
        self._require_restored()
        self.robot.write_data_to_sim()
        self.sim.step(render=False)
        self.robot.update(float(dt_s))
        self.object_asset.update(float(dt_s))
        self._simulation_time_s += float(dt_s)
        self._phase_elapsed_s += float(dt_s)

    def measured_candidate_wrenches(
        self,
        *,
        context: HighLevelPolicyContext,
        active_knot: InteractionKnot,
    ) -> Sequence[MeasuredCandidateWrench]:
        self._require_restored()
        candidate_by_id = {
            candidate.candidate_id: candidate
            for candidate in context.contact_candidate_set.candidates
        }
        sensor_by_anchor = {
            anchor_id: index
            for index, anchor_id in enumerate(self.contact_layout.selected_anchor_ids)
        }
        matrix = self._contact_force_matrix(self.selected_contact_view)
        counts, starts, force_magnitudes, points = self._contact_points(
            self.selected_contact_view
        )
        result: list[MeasuredCandidateWrench] = []
        for assignment in active_knot.contact_assignments:
            if assignment.schedule_state not in _ACTIVE_CONTACT_STATES:
                continue
            candidate = candidate_by_id.get(assignment.candidate_id)
            sensor_index = sensor_by_anchor.get(assignment.anchor_id)
            if candidate is None or sensor_index is None or sensor_index >= len(matrix):
                result.append(
                    MeasuredCandidateWrench(
                        candidate_id=assignment.candidate_id,
                        wrench_contact=(0.0,) * 6,
                        evidence_valid=False,
                        sample_count=0,
                    )
                )
                continue
            force_world = tuple(
                self.robot_on_target_force_sign * float(value)
                for value in matrix[sensor_index]
            )
            count = counts[sensor_index] if sensor_index < len(counts) else 0
            start = starts[sensor_index] if sensor_index < len(starts) else 0
            point = _weighted_contact_point(
                count=count,
                start=start,
                force_magnitudes=force_magnitudes,
                points=points,
                fallback=tuple(float(value) for value in candidate.contact_pose_world[:3]),
            )
            arm = tuple(
                float(point[index]) - float(candidate.contact_pose_world[index])
                for index in range(3)
            )
            torque_world = _cross(arm, force_world)
            wrench_contact = world_wrench_to_contact(
                [*force_world, *torque_world], candidate
            )
            result.append(
                MeasuredCandidateWrench(
                    candidate_id=assignment.candidate_id,
                    wrench_contact=tuple(float(value) for value in wrench_contact),
                    evidence_valid=bool(count > 0 and _finite_vector(force_world)),
                    sample_count=max(0, int(count)),
                )
            )
        return result

    def collision_evidence(
        self,
        *,
        context: HighLevelPolicyContext,
        active_knot: InteractionKnot,
    ) -> tuple[Sequence[ShadowCollisionSample], float]:
        self._require_restored()
        active_by_anchor = {
            assignment.anchor_id: assignment
            for assignment in active_knot.contact_assignments
            if assignment.schedule_state in _ACTIVE_CONTACT_STATES
        }
        candidate_by_id = {
            candidate.candidate_id: candidate
            for candidate in context.contact_candidate_set.candidates
        }
        anchor_by_body: dict[str, int] = {}
        for anchor in context.morphology_graph.robot_anchors:
            if anchor.link_id is None:
                continue
            anchor_by_body[_combined_name(anchor.module_id, anchor.link_id)] = anchor.anchor_id
        matrix = self._contact_force_matrix(self.all_contact_view)
        filter_count = len(self.contact_layout.all_filter_entity_ids)
        expected = len(self.contact_layout.all_sensor_body_names) * filter_count
        if len(matrix) != expected:
            raise RuntimeError("Order9 all-contact force matrix layout mismatch")
        samples: list[ShadowCollisionSample] = []
        for sensor_index, (body_name, entity_id) in enumerate(
            zip(
                self.contact_layout.all_sensor_body_names,
                self.contact_layout.all_sensor_entity_ids,
            )
        ):
            for filter_index, target_entity in enumerate(
                self.contact_layout.all_filter_entity_ids
            ):
                force = matrix[sensor_index * filter_count + filter_index]
                if _norm(force) < self.contact_force_threshold_n:
                    continue
                anchor_id = anchor_by_body.get(body_name)
                assignment = active_by_anchor.get(anchor_id) if anchor_id is not None else None
                candidate = (
                    None
                    if assignment is None
                    else candidate_by_id.get(assignment.candidate_id)
                )
                intended = bool(
                    target_entity == self.object_id
                    and assignment is not None
                    and candidate is not None
                    and candidate.target_entity_id == self.object_id
                )
                samples.append(
                    ShadowCollisionSample(
                        entity_a=entity_id,
                        entity_b=target_entity,
                        signed_distance_m=-1.0e-6,
                        candidate_id=(
                            assignment.candidate_id if intended and assignment is not None else None
                        ),
                        anchor_id=anchor_id,
                        target_entity_id=target_entity,
                        task_allowed=False,
                        allowance_reason=None,
                    )
                )
        clearance, coarse_sample = self._prohibited_clearance(
            context=context,
            active_knot=active_knot,
        )
        if coarse_sample is not None:
            samples.append(coarse_sample)
        return tuple(samples), float(max(clearance, 0.0))

    def payload_coupling(
        self,
        *,
        active_knot: InteractionKnot,
    ) -> PayloadCoupling | None:
        if not any(
            assignment.schedule_state in _ACTIVE_CONTACT_STATES
            for assignment in active_knot.contact_assignments
        ):
            return None
        body_pose = _centroidal_pose(self.robot, self.physical_model, self.morphology_graph)
        object_pose = _object_pose(self.object_asset)
        body_from_world = transform_from_pose(inverse_pose(tuple(body_pose))).rotation
        offset_world = tuple(
            float(object_pose[index]) - float(body_pose[index]) for index in range(3)
        )
        offset_body = matvec(body_from_world, offset_world)
        payload = PayloadCoupling(
            payload_id=self.object_id,
            contact_model="natural_contact_grasp_v1",
            mass_kg=self.object_mass_kg,
            inertia_body=list(self.object_inertia_body),
            com_offset_body=tuple(float(value) for value in offset_body),
            coupling_mode="order9_observed_natural_contact_payload_v1",
        )
        payload.validate()
        return payload

    def finite_state(self) -> bool:
        tensors = (
            self.robot.data.root_pose_w,
            self.robot.data.root_lin_vel_w,
            self.robot.data.root_ang_vel_w,
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            self.object_asset.data.root_pose_w,
            self.object_asset.data.root_lin_vel_w,
            self.object_asset.data.root_ang_vel_w,
        )
        return all(bool(self.torch.isfinite(_as_torch(item)).all().item()) for item in tensors)

    def reset(self) -> None:
        if self._closed:
            return
        self.robot.permanent_wrench_composer.reset()
        self._restored = False
        self._simulation_time_s = 0.0
        self._phase_elapsed_s = 0.0
        self._command_index = 0

    def close(self) -> None:
        if self._closed:
            return
        self.reset()
        self._closed = True

    def _contact_force_matrix(self, view: Any) -> list[tuple[float, float, float]]:
        tensor = self.wp.to_torch(
            view.get_contact_force_matrix(float(self.sim.get_physics_dt()))
        ).reshape(-1, 3)
        return [
            tuple(float(value) for value in row)
            for row in tensor.detach().cpu().tolist()
        ]

    def _contact_points(
        self,
        view: Any,
    ) -> tuple[list[int], list[int], list[float], list[tuple[float, float, float]]]:
        (
            force_buffer,
            point_buffer,
            _normal_buffer,
            _separation_buffer,
            count_buffer,
            start_buffer,
        ) = view.get_contact_data(float(self.sim.get_physics_dt()))
        counts = self.wp.to_torch(count_buffer).reshape(-1).to(self.torch.int64)
        starts = self.wp.to_torch(start_buffer).reshape(-1).to(self.torch.int64)
        forces = self.wp.to_torch(force_buffer).reshape(-1)
        points = self.wp.to_torch(point_buffer).reshape(-1, 3)
        return (
            [int(value) for value in counts.detach().cpu().tolist()],
            [int(value) for value in starts.detach().cpu().tolist()],
            [float(value) for value in forces.detach().cpu().tolist()],
            [
                tuple(float(value) for value in row)
                for row in points.detach().cpu().tolist()
            ],
        )

    def _prohibited_clearance(
        self,
        *,
        context: HighLevelPolicyContext,
        active_knot: InteractionKnot,
    ) -> tuple[float, ShadowCollisionSample | None]:
        candidate_by_id = {
            candidate.candidate_id: candidate
            for candidate in context.contact_candidate_set.candidates
        }
        active_object_anchor_ids = {
            assignment.anchor_id
            for assignment in active_knot.contact_assignments
            if assignment.schedule_state in _ACTIVE_CONTACT_STATES
            and assignment.candidate_id in candidate_by_id
            and candidate_by_id[assignment.candidate_id].target_entity_id
            == self.object_id
        }
        anchor_by_body = {
            _combined_name(anchor.module_id, anchor.link_id): anchor.anchor_id
            for anchor in context.morphology_graph.robot_anchors
            if anchor.link_id is not None
        }
        object_pose = _object_pose(self.object_asset)
        object_geometry = _primitive_object_clearance_geometry(
            self.object_geometry_type,
            self.object_size_m,
            object_pose,
        )
        support_box = OrientedBox.axis_aligned(
            self.support_center_world_m,
            self.support_half_extents_m,
        )
        minimum = math.inf
        closest: ShadowCollisionSample | None = None
        entity_by_body = dict(
            zip(
                self.contact_layout.all_sensor_body_names,
                self.contact_layout.all_sensor_entity_ids,
            )
        )
        for body_name in self.contact_layout.all_sensor_body_names:
            if body_name not in self.robot.body_names:
                continue
            bounds = self.body_local_aabb_m.get(body_name)
            if bounds is None:
                continue
            body_index = self.robot.body_names.index(body_name)
            body_pose = tuple(
                _tensor_body_row(self.robot.data.body_pos_w, body_index)
                + _tensor_body_row(self.robot.data.body_quat_w, body_index)
            )
            body_box = OrientedBox.from_pose_and_local_bounds(
                body_pose,
                bounds[0],
                bounds[1],
            )
            anchor_id = anchor_by_body.get(body_name)
            if anchor_id not in active_object_anchor_ids:
                object_clearance = _primitive_object_box_clearance(
                    body_box,
                    geometry_type=self.object_geometry_type,
                    geometry=object_geometry,
                )
                if object_clearance < minimum:
                    minimum = object_clearance
                    closest = ShadowCollisionSample(
                        entity_a=entity_by_body[body_name],
                        entity_b=self.object_id,
                        signed_distance_m=float(object_clearance),
                        target_entity_id=self.object_id,
                    )
            support_clearance = oriented_box_clearance(body_box, support_box)
            if support_clearance < minimum:
                minimum = support_clearance
                closest = ShadowCollisionSample(
                    entity_a=entity_by_body[body_name],
                    entity_b="support",
                    signed_distance_m=float(support_clearance),
                    target_entity_id="support",
                )
        if math.isinf(minimum):
            return 0.0, None
        return float(minimum), closest

    def _validate_configuration(self) -> None:
        if not self.object_id:
            raise ValueError("Order9 Isaac object id must be non-empty")
        if self.object_mass_kg <= 0.0 or not math.isfinite(self.object_mass_kg):
            raise ValueError("Order9 Isaac object mass must be positive")
        for name in (
            "object_friction",
            "selected_gripper_friction",
            "contact_stiffness_n_per_m",
            "contact_damping_n_s_per_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"Order9 Isaac {name} must be positive")
        if len(self.object_inertia_body) != 6 or not _finite_vector(
            self.object_inertia_body
        ):
            raise ValueError("Order9 Isaac object inertia must have six finite values")
        if self.object_geometry_type not in {"box", "sphere", "cylinder", "capsule"}:
            raise ValueError("Order9 Isaac object geometry type is unsupported")
        if len(self.object_size_m) != 3 or any(
            not math.isfinite(value) or value <= 0.0
            for value in self.object_size_m
        ):
            raise ValueError("Order9 Isaac object size must be positive")
        if self.object_geometry_type in {"sphere", "cylinder", "capsule"} and not math.isclose(
            self.object_size_m[0],
            self.object_size_m[1],
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError("Order9 round object x/y size must be a common diameter")
        if self.object_geometry_type == "sphere" and not math.isclose(
            self.object_size_m[1],
            self.object_size_m[2],
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError("Order9 sphere size must be an equal-diameter vector")
        if not math.isfinite(self.support_top_z_m):
            raise ValueError("Order9 Isaac support top must be finite")
        if (
            len(self.support_center_world_m) != 3
            or len(self.support_half_extents_m) != 3
            or not _finite_vector(self.support_center_world_m)
            or any(
                not math.isfinite(value) or value <= 0.0
                for value in self.support_half_extents_m
            )
            or not math.isclose(
                self.support_center_world_m[2] + self.support_half_extents_m[2],
                self.support_top_z_m,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
        ):
            raise ValueError("Order9 Isaac support bounds are invalid")
        if self.robot_on_target_force_sign not in {-1.0, 1.0}:
            raise ValueError("Order9 contact force sign must be +/-1")
        if self.contact_force_threshold_n <= 0.0:
            raise ValueError("Order9 contact threshold must be positive")
        if self.actuator_readback is not None and (
            self.actuator_readback.get("matches_physical_model") is not True
        ):
            raise ValueError("Order9 actuator readback is not PhysicalModel-bound")
        missing = set(self.contact_layout.all_sensor_body_names) - set(
            self.robot.body_names
        )
        if missing:
            raise ValueError(f"Order9 contact bodies are absent: {sorted(missing)}")
        for minimum, maximum in self.body_local_aabb_m.values():
            if (
                len(minimum) != 3
                or len(maximum) != 3
                or not _finite_vector((*minimum, *maximum))
                or any(left > right for left, right in zip(minimum, maximum))
            ):
                raise ValueError("Order9 body-local AABBs are invalid")

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Order9 Isaac scene adapter is closed")

    def _require_restored(self) -> None:
        self._require_open()
        if not self._restored:
            raise RuntimeError("Order9 Isaac scene adapter has no restored snapshot")


def _set_sparse_joint_target(
    robot: Any,
    values: Mapping[int, float],
    *,
    setter: str,
    torch_module: Any,
    device: str,
) -> None:
    if not values:
        return
    indices = sorted(values)
    getattr(robot, setter)(
        target=torch_module.tensor(
            [[float(values[index]) for index in indices]],
            dtype=torch_module.float32,
            device=device,
        ),
        joint_ids=torch_module.tensor(
            indices,
            dtype=torch_module.int32,
            device=device,
        ),
    )


def _combined_name(module_id: int, local_id: str | None) -> str:
    if local_id is None:
        raise SchemaValidationError("Order9 module-local body/joint id is absent")
    return f"module_{int(module_id)}__{local_id}"


def _exact_name_index(names: Sequence[str], value: str, label: str) -> int:
    if value not in names:
        raise RuntimeError(f"Order9 cannot resolve {label} {value!r}")
    return list(names).index(value)


def _as_torch(value: Any) -> Any:
    return value.torch if hasattr(value, "torch") else value


def _tensor_row(value: Any) -> list[float]:
    tensor = _as_torch(value)
    row = tensor[0] if getattr(tensor, "ndim", 1) > 1 else tensor
    return [float(item) for item in row.detach().cpu().tolist()]


def _tensor_body_row(value: Any, body_index: int) -> list[float]:
    tensor = _as_torch(value)
    return [float(item) for item in tensor[0, body_index].detach().cpu().tolist()]


def _tensor_scalar(value: Any, index: int) -> float:
    tensor = _as_torch(value)
    return float(tensor[0, index].detach().cpu().item())


def _root_twist(robot: Any) -> list[float]:
    return _tensor_row(robot.data.root_lin_vel_w) + _tensor_row(
        robot.data.root_ang_vel_w
    )


def _object_pose(object_asset: Any) -> list[float]:
    tensor = getattr(object_asset.data, "root_com_pose_w", object_asset.data.root_pose_w)
    return _tensor_row(tensor)


def _object_twist(object_asset: Any) -> list[float]:
    tensor = getattr(object_asset.data, "root_com_vel_w", None)
    if tensor is not None:
        return _tensor_row(tensor)
    return _tensor_row(object_asset.data.root_lin_vel_w) + _tensor_row(
        object_asset.data.root_ang_vel_w
    )


def _weighted_contact_point(
    *,
    count: int,
    start: int,
    force_magnitudes: Sequence[float],
    points: Sequence[Sequence[float]],
    fallback: tuple[float, float, float],
) -> tuple[float, float, float]:
    stop = int(start) + int(count)
    if count <= 0 or start < 0 or stop > min(len(force_magnitudes), len(points)):
        return fallback
    weights = [abs(float(force_magnitudes[index])) for index in range(start, stop)]
    total = sum(weights)
    if total <= 0.0:
        return fallback
    return tuple(
        sum(weights[offset] * float(points[start + offset][axis]) for offset in range(count))
        / total
        for axis in range(3)
    )


def _cross(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float]:
    return (
        float(left[1]) * float(right[2]) - float(left[2]) * float(right[1]),
        float(left[2]) * float(right[0]) - float(left[0]) * float(right[2]),
        float(left[0]) * float(right[1]) - float(left[1]) * float(right[0]),
    )


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(sum(float(item) ** 2 for item in value))


def _finite_vector(value: Sequence[float]) -> bool:
    return all(math.isfinite(float(item)) for item in value)


def _primitive_object_clearance_geometry(
    geometry_type: str,
    size_m: Sequence[float],
    pose_world: Sequence[float],
) -> object:
    if geometry_type == "box":
        half = tuple(0.5 * float(value) for value in size_m)
        return OrientedBox.from_pose_and_local_bounds(
            pose_world,
            tuple(-value for value in half),
            half,
        )
    transform = transform_from_pose(tuple(float(value) for value in pose_world))
    radius = 0.5 * float(size_m[0])
    if geometry_type == "sphere":
        return (tuple(transform.translation), radius)
    if geometry_type == "cylinder":
        return circumscribed_cylinder_polytope(
            pose_world,
            radius=radius,
            height=float(size_m[2]),
        )
    if geometry_type == "capsule":
        half_segment = 0.5 * float(size_m[2])
        start = tuple(
            float(transform.translation[axis]) + value
            for axis, value in enumerate(
                matvec(transform.rotation, (0.0, 0.0, -half_segment))
            )
        )
        end = tuple(
            float(transform.translation[axis]) + value
            for axis, value in enumerate(
                matvec(transform.rotation, (0.0, 0.0, half_segment))
            )
        )
        return (start, end, radius)
    raise ValueError(f"unsupported Order9 object geometry {geometry_type!r}")


def _primitive_object_box_clearance(
    body_box: OrientedBox,
    *,
    geometry_type: str,
    geometry: object,
) -> float:
    if geometry_type == "box":
        if not isinstance(geometry, OrientedBox):
            raise TypeError("Order9 box clearance geometry is invalid")
        return oriented_box_clearance(body_box, geometry)
    if geometry_type == "cylinder":
        if not isinstance(geometry, ConvexPolytope):
            raise TypeError("Order9 cylinder clearance geometry is invalid")
        return oriented_box_polytope_clearance(body_box, geometry)
    if not isinstance(geometry, tuple):
        raise TypeError("Order9 round-object clearance geometry is invalid")
    if geometry_type == "sphere":
        center, radius = geometry
        return sphere_oriented_box_clearance(center, float(radius), body_box)
    if geometry_type == "capsule":
        start, end, radius = geometry
        return capsule_oriented_box_clearance(
            start,
            end,
            float(radius),
            body_box,
        )
    raise ValueError(f"unsupported Order9 object geometry {geometry_type!r}")


def _centroidal_pose(
    robot: Any,
    physical_model: PhysicalModel,
    morphology_graph: MorphologyGraph,
) -> list[float]:
    module_frame_link_id = _module_frame_link_id(physical_model)
    masses = []
    positions = []
    orientation: list[float] | None = None
    for module in sorted(morphology_graph.modules, key=lambda item: item.module_id):
        name = _combined_name(module.module_id, module_frame_link_id)
        index = _exact_name_index(robot.body_names, name, "module frame")
        masses.append(float(physical_model.aggregate_mass_kg))
        positions.append(_tensor_body_row(robot.data.body_pos_w, index))
        if module.module_id == morphology_graph.base_module_id:
            orientation = _tensor_body_row(robot.data.body_quat_w, index)
    total = sum(masses)
    if total <= 0.0 or orientation is None:
        raise RuntimeError("Order9 cannot build centroidal pose")
    center = [
        sum(mass * position[axis] for mass, position in zip(masses, positions)) / total
        for axis in range(3)
    ]
    return [*center, *orientation]


def _module_frame_link_id(physical_model: PhysicalModel) -> str:
    raw = physical_model.metadata.get("baselink", {})
    value = str(raw.get("name", "fc")) if isinstance(raw, Mapping) else "fc"
    if not value:
        raise SchemaValidationError("Order9 PhysicalModel module frame is empty")
    return value


__all__ = [
    "ORDER9_ISAAC_SCENE_ADAPTER_VERSION",
    "IsaacLabOrder9SceneAdapter",
    "Order9IsaacContactViewLayout",
]
