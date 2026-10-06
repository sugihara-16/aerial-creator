from __future__ import annotations

"""Nominal execution of the provisional contact plan on the existing QPID bridge."""

import json
import time
from dataclasses import replace

import torch
from amsrr.controllers.batched_rigid_body_model import _quaternion_to_matrix
from amsrr.simulation.order9_tensor_object_task import (
    ORDER9_CONTACT_SCHEDULE_ATTACH,
    ORDER9_CONTACT_SCHEDULE_MAINTAIN,
    ORDER9_CONTACT_SCHEDULE_RELEASE,
)

from amsrr.training.order9_tensor_pi_l_runtime import (
    Order9TensorNominalQPIDStep,
    Order9TensorPiLRuntime,
)

NOMINAL_BODY_TRACKING_CONTRACT = "nominal_contact_xy_pid_12_1_4_v1"


def qpid_reference_twist(reference_twist_world, measured_body_pose_world):
    """QPID takes linear world velocity and angular measured-body velocity."""
    result = reference_twist_world.clone()
    body_from_world = _quaternion_to_matrix(measured_body_pose_world[:, 3:]).transpose(-1, -2)
    result[:, 3:] = (body_from_world @ reference_twist_world[:, 3:].unsqueeze(-1)).squeeze(-1)
    return result


class TeacherContactController(Order9TensorPiLRuntime):
    """Use planned q/qdot and body twist, with zero learned corrections.

    The inherited constructor loads the legacy decoder/model configuration for
    the isolated teacher harness. Neither actor inference nor learning is used.
    Unlike the legacy reset HOLD method, execution retains trajectory velocity.
    """

    execution_calls = 0

    def __init__(self, *args, **kwargs):
        joint_load_contacts = kwargs.pop("joint_load_contacts", None)
        super().__init__(*args, **kwargs)
        # Mass-normalized horizontal tracking; keep vertical/attitude gains,
        # actuator bounds, allocation constraints and controller state intact.
        self.controller.config = replace(self.controller.config,
            xy_p_gain=12.0, xy_i_gain=1.0, xy_d_gain=4.0)
        self._joint_load_contacts = joint_load_contacts
        from amsrr.utils.tensor_dataclass_graph import TensorDataclassGraph
        self._load_graph = TensorDataclassGraph()
        self._joint_load_tensors = {k: torch.as_tensor(v, device=self.device, dtype=self.dtype)
            for k, v in (joint_load_contacts or {}).items() if k != "references"}
        if joint_load_contacts is not None:
            from amsrr.controllers.articulated_joint_load import ArticulatedJointLoadModel
            self.builder._internal_load_model = ArticulatedJointLoadModel(
                self.morphology_graph, self.physical_model, joint_load_contacts["references"])
        self.builder._use_cuda_graph = True
        self.builder._use_cpu_compile = True
        self.builder._use_cpu_native = True
        self.builder._reuse_post_step_model = True
        self._warm_control_kernels()

    @torch.no_grad()
    def _warm_control_kernels(self):
        """Compile/capture before control starts, without advancing PID state."""
        poses = torch.zeros((self.batch_size, self.builder.module_count, 7), device=self.device, dtype=self.dtype)
        poses[..., 6] = 1.
        twists = torch.zeros((*poses.shape[:2], 6), device=self.device, dtype=self.dtype)
        joints = torch.zeros((*poses.shape[:2], self.builder.local_joint_count), device=self.device, dtype=self.dtype)
        model = self.builder.build(module_pose_world=poses, module_twist_world=twists,
            local_joint_positions_rad=joints, local_joint_velocities_rad=torch.zeros_like(joints))
        self.builder.build_kinematics(module_pose_world=poses, module_twist_world=twists,
            local_joint_positions_rad=joints, local_joint_velocities_rad=torch.zeros_like(joints))
        self.controller.compute(control_model=model, desired_body_pose_world=model.body_pose_world,
            desired_body_twist=model.body_twist_world, residual_wrench_body=torch.zeros_like(model.body_twist_world),
            state=self.controller_state, payload_active=torch.zeros(self.batch_size, device=self.device, dtype=torch.bool),
            payload_mass_kg=torch.zeros(self.batch_size, device=self.device, dtype=self.dtype),
            payload_inertia_body=torch.zeros((self.batch_size, 6), device=self.device, dtype=self.dtype),
            payload_com_offset_body=torch.zeros((self.batch_size, 3), device=self.device, dtype=self.dtype))
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def compute_nominal_qpid_hold(
        self,
        *,
        task_target,
        state,
        estimated_payload_mass_kg,
        estimated_payload_inertia_body,
        payload_active,
        estimated_payload_com_object=None,
    ):
        self._validate_controller_inputs(
            task_target=task_target,
            state=state,
            estimated_payload_mass_kg=estimated_payload_mass_kg,
            estimated_payload_inertia_body=estimated_payload_inertia_body,
            estimated_payload_com_object=estimated_payload_com_object,
            payload_active=payload_active,
        )
        model = self.builder.build(
            module_pose_world=state.module_pose_world,
            module_twist_world=state.module_twist_world,
            local_joint_positions_rad=state.local_joint_positions_rad,
            local_joint_velocities_rad=state.local_joint_velocities_radps,
        )
        command = self.decoder.decode_nominal(
            reference_body_pose_world=task_target.desired_robot_root_pose_world,
            reference_body_twist=task_target.desired_robot_root_twist_world,
            reference_local_joint_positions_rad=task_target.nominal_joint_positions_rad,
            reference_local_joint_velocities_radps=task_target.nominal_joint_velocities_radps,
            total_mass_kg=model.total_mass_kg,
        )
        offset = self._payload_offset_body(
            model.body_pose_world,
            state.object_pose_world,
            estimated_payload_com_object=(
                torch.zeros((self.batch_size, 3), device=self.device, dtype=self.dtype)
                if estimated_payload_com_object is None
                else estimated_payload_com_object
            ),
        )
        inputs = dict(model=model, object_pose_world=state.object_pose_world,
            phase_progress=task_target.phase_progress, contact_schedule_index=task_target.contact_schedule_index,
            payload_active=payload_active, estimated_payload_mass_kg=estimated_payload_mass_kg)
        internal_constraints = self._load_graph.call(self._internal_constraints,
            kwargs=inputs, configuration=None) if self._joint_load_contacts is not None else None
        result = self.controller.compute(
            control_model=model,
            internal_joint_constraints=internal_constraints,
            desired_body_pose_world=command.desired_body_pose_world,
            desired_body_twist=qpid_reference_twist(command.desired_body_twist, model.body_pose_world),
            residual_wrench_body=command.residual_wrench_body,
            state=self.controller_state,
            payload_active=payload_active,
            payload_mass_kg=estimated_payload_mass_kg,
            payload_inertia_body=estimated_payload_inertia_body,
            payload_com_offset_body=offset,
        )
        self.controller_state = result.next_state
        self.controller_qp_feasible.copy_(result.allocation.feasible)
        self.allocation_residual_norm.copy_(result.allocation.residual_norm)
        self.controller_status_one_hot.zero_()
        status = torch.where(result.allocation.feasible, 0, 2).long()
        self.controller_status_one_hot.scatter_(1, status.unsqueeze(1), 1.0)
        type(self).execution_calls += 1
        error = (
            (
                command.joint_position_targets_rad
                - task_target.nominal_joint_positions_rad
            )
            .abs()
            .amax()
        )
        if float(error) > 1e-6:
            raise RuntimeError("decoder changed a checked joint position reference")
        if not torch.allclose(
            command.joint_velocity_targets_radps,
            task_target.nominal_joint_velocities_radps,
            atol=1e-6,
            rtol=0,
        ):
            raise RuntimeError("decoder changed a checked joint velocity reference")
        if type(self).execution_calls % 250 == 0:
            print(
                "TEACHER_CONTACT_PROGRESS="
                + json.dumps(
                    {
                        "controller_calls": type(self).execution_calls,
                        "monotonic_s": time.monotonic(),
                        "schedule": task_target.contact_schedule_index.tolist(),
                        "phase_progress": task_target.phase_progress.tolist(),
                        "object_z_m": state.object_pose_world[:, 2].tolist(),
                        "qp_feasible": result.allocation.feasible.tolist(),
                    }
                ),
                flush=True,
            )
        return Order9TensorNominalQPIDStep(
            control_model=model,
            policy_command=command,
            controller_result=result,
        )

    def _internal_constraints(self, *, model, object_pose_world, phase_progress,
            contact_schedule_index, payload_active, estimated_payload_mass_kg):
        """Pure load geometry; fixed plan parameters are owned by this controller."""
        internal_constraints = None
        if self._joint_load_contacts is not None:
            settings = self._joint_load_contacts
            normals = self._joint_load_tensors["normal_object"]
            object_rotation = _quaternion_to_matrix(object_pose_world[:, 3:])
            normal_world = (object_rotation[:, None] @ normals[None, :, :, None]).squeeze(-1)
            fn = self._joint_load_tensors["normal_force_n"]
            support = fn / fn.sum().clamp_min(1.e-9)
            active = payload_active.to(self.dtype)
            progress = phase_progress.clamp(0., 1.)
            normal_weight = torch.where(contact_schedule_index == ORDER9_CONTACT_SCHEDULE_RELEASE,
                1. - progress * progress * (3. - 2. * progress), active)
            normal_bias = (model.joint_contact_jacobian *
                (normal_world * fn[None, :, None])[:, None]).sum(dim=(-1, -2)) * normal_weight[:, None]
            support_force = -estimated_payload_mass_kg[:, None] * 9.81 * support[None]
            bias = model.joint_gravity_load_nm + normal_bias + (
                model.joint_contact_jacobian[..., 2] * support_force[:, None]).sum(-1) * active[:, None]
            if "contact_force_world_n" in settings:
                # Same nominal force-on-object allocation as the planner. Its
                # opposite acts on the robot; do not add a second mg support
                # distribution. This is a nominal load estimate, not sensing.
                reaction = -self._joint_load_tensors["contact_force_world_n"][None]
                normal_reaction = normal_world * fn[None, :, None]
                remaining = reaction - normal_reaction
                bias = model.joint_gravity_load_nm + normal_bias + (
                    model.joint_contact_jacobian * remaining[:, None]).sum(dim=(-1, -2)) * active[:, None]
            # Quasistatic extra opening at fixed whole-body CoM. Nominal normal
            # compression is already represented by the grasp preload.
            fixed_com_jacobian = model.joint_contact_jacobian - (
                model.joint_mass_jacobian[:, :, None] / model.total_mass_kg[:, None, None, None])
            opening_map = torch.einsum("bac,bjac->baj", normal_world, fixed_com_jacobian)
            opening_map = opening_map / self._joint_load_tensors["stiffness_nm_per_rad"][None, None]
            acquiring_or_holding = ((contact_schedule_index == ORDER9_CONTACT_SCHEDULE_ATTACH) |
                                    (contact_schedule_index == ORDER9_CONTACT_SCHEDULE_MAINTAIN))
            opening_map = opening_map * acquiring_or_holding[:, None, None]
            internal_constraints = dict(joint_load_matrix=model.joint_load_matrix,
                joint_load_bias_nm=bias,
                joint_load_limit_nm=self._joint_load_tensors["limits_nm"][None].expand(self.batch_size, -1),
                joint_opening_map=opening_map, joint_normal_bias_nm=normal_bias)
        return internal_constraints
