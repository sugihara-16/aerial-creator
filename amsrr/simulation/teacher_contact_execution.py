from __future__ import annotations

"""Nominal execution of the provisional contact plan on the existing QPID bridge."""

import json
import time

import torch

from amsrr.training.order9_tensor_pi_l_runtime import (
    Order9TensorNominalQPIDStep,
    Order9TensorPiLRuntime,
)


class TeacherContactController(Order9TensorPiLRuntime):
    """Use planned q/qdot and body twist, with zero learned corrections.

    The inherited constructor loads the legacy decoder/model configuration for
    the isolated teacher harness. Neither actor inference nor learning is used.
    Unlike the legacy reset HOLD method, execution retains trajectory velocity.
    """

    execution_calls = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.builder._use_cuda_graph = True
        self.builder._use_cpu_compile = True
        self._warm_control_kernels()

    @torch.no_grad()
    def _warm_control_kernels(self):
        """Compile/capture before control starts, without advancing PID state."""
        poses = torch.zeros((self.batch_size, self.builder.module_count, 7), device=self.device, dtype=self.dtype)
        poses[..., 6] = 1.
        twists = torch.zeros((*poses.shape[:2], 6), device=self.device, dtype=self.dtype)
        joints = torch.zeros((*poses.shape[:2], self.builder.local_joint_count), device=self.device, dtype=self.dtype)
        model = self.builder.build(module_pose_world=poses, module_twist_world=twists, local_joint_positions_rad=joints)
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
        result = self.controller.compute(
            control_model=model,
            desired_body_pose_world=command.desired_body_pose_world,
            desired_body_twist=command.desired_body_twist,
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
