from __future__ import annotations

"""GPU-resident ``pi_L -> PolicyCommand -> QPID/QP`` Order 9 hot path."""

import hashlib
import math
from dataclasses import dataclass

import torch

from amsrr.controllers.batched_qpid_controller import (
    BatchedQPIDController,
    BatchedQPIDResult,
    BatchedQPIDState,
)
from amsrr.controllers.batched_rigid_body_model import (
    BatchedRigidBodyControlModel,
    BatchedRigidBodyControlModelBuilder,
)
from amsrr.policies.order9_low_level_policy import (
    ORDER9_GLOBAL_ACTION_SIZE,
    Order9LowLevelActorCriticStep,
    Order9LowLevelPolicyConfig,
    Order9PhaseConditionedActorCritic,
)
from amsrr.policies.order9_tensor_command_decoder import (
    Order9TensorPolicyCommand,
    Order9TensorPolicyCommandDecoder,
)
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.task_spec import TaskType
from amsrr.simulation.order9_object_task_runtime import (
    ORDER9_OBJECT_TASK_ADAPTER_ID,
    ORDER9_OBJECT_TASK_ACTOR_PHASE_COUNT,
)
from amsrr.simulation.order9_tensor_isaac_io import Order9TensorIsaacState
from amsrr.simulation.order9_tensor_object_task import Order9TensorObjectTaskTarget
from amsrr.training.order9_tensor_runtime import (
    Order9CentroidalTensorObservation,
    Order9TensorizedTopologyBucket,
    order9_low_level_actor_features_from_tensors,
)


ORDER9_TENSOR_PI_L_RUNTIME_VERSION = "order9_tensor_complete_pi_l_qpid_runtime_v3"


@dataclass(frozen=True)
class Order9TensorPiLStep:
    control_model: BatchedRigidBodyControlModel
    actor_features: torch.Tensor
    phase_features: torch.Tensor
    previous_global_action: torch.Tensor
    recurrent_state_in: torch.Tensor
    actor_controller_qp_feasible: torch.Tensor
    actor_controller_status_one_hot: torch.Tensor
    actor_allocation_residual_norm: torch.Tensor
    actor_task_success: torch.Tensor
    policy_step: Order9LowLevelActorCriticStep
    policy_command: Order9TensorPolicyCommand
    controller_result: BatchedQPIDResult
    privileged_disturbance_body: torch.Tensor


class Order9TensorPiLRuntime:
    """Stateful recurrent policy and controller for a fixed topology bucket."""

    runtime_version = ORDER9_TENSOR_PI_L_RUNTIME_VERSION

    def __init__(
        self,
        *,
        morphology_graph: MorphologyGraph,
        physical_model: PhysicalModel,
        policy: Order9PhaseConditionedActorCritic,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        controller: BatchedQPIDController | None = None,
        policy_frame_origins_world: torch.Tensor | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("Order9 tensor pi_L batch size must be positive")
        morphology_graph.validate()
        physical_model.validate()
        self.device = torch.device(device)
        self.dtype = dtype
        self.batch_size = int(batch_size)
        self.morphology_graph = MorphologyGraph.from_dict(
            morphology_graph.to_dict()
        )
        self.physical_model = PhysicalModel.from_dict(physical_model.to_dict())
        self.policy = policy.to(device=self.device, dtype=self.dtype)
        self.policy.eval()
        if not isinstance(self.policy.config, Order9LowLevelPolicyConfig):
            raise TypeError("Order9 tensor runtime requires Order9 pi_L config")
        self.config = self.policy.config
        self.builder = BatchedRigidBodyControlModelBuilder(
            self.morphology_graph, self.physical_model
        )
        self.bucket = Order9TensorizedTopologyBucket(
            self.morphology_graph,
            batch_size=self.batch_size,
            device=self.device,
            dtype=self.dtype,
        )
        self.decoder = Order9TensorPolicyCommandDecoder(
            module_ids=self.builder.module_ids,
            physical_model=self.physical_model,
            config=self.config,
        )
        self.controller = controller or BatchedQPIDController()
        self.policy_frame_origins_world = self._prepare_policy_frame_origins(
            policy_frame_origins_world
        )
        self.previous_action = torch.zeros(
            (self.batch_size, ORDER9_GLOBAL_ACTION_SIZE),
            device=self.device,
            dtype=self.dtype,
        )
        self.recurrent_state = self.policy.initial_state(
            self.batch_size, device=self.device, dtype=self.dtype
        )
        self.controller_state = self.controller.initial_state(
            self.batch_size,
            self.builder.rotor_count,
            device=self.device,
            dtype=self.dtype,
        )
        self.controller_qp_feasible = torch.ones(
            (self.batch_size,), device=self.device, dtype=torch.bool
        )
        self.controller_status_one_hot = torch.zeros(
            (self.batch_size, 4), device=self.device, dtype=self.dtype
        )
        self.controller_status_one_hot[:, 0] = 1.0
        self.allocation_residual_norm = torch.zeros(
            (self.batch_size,), device=self.device, dtype=self.dtype
        )
        self.task_success = torch.zeros(
            (self.batch_size,), device=self.device, dtype=torch.bool
        )
        self.module_health = torch.ones(
            (self.batch_size, self.builder.module_count),
            device=self.device,
            dtype=self.dtype,
        )
        joint_type_by_id = {
            joint.joint_id: joint.joint_type
            for joint in self.physical_model.joints
        }
        active_joint_mask = torch.tensor(
            [
                joint_type_by_id[joint_id] != "fixed"
                for joint_id in self.builder.local_joint_ids
            ],
            device=self.device,
            dtype=torch.bool,
        )
        self.joint_mask = active_joint_mask.reshape(1, 1, -1).expand(
            self.batch_size,
            self.builder.module_count,
            self.builder.local_joint_count,
        )
        self.bucket.batch.metadata.update(
            {
                "runtime_pose_translation_frame": (
                    "world"
                    if self.policy_frame_origins_world is None
                    else "world_minus_policy_frame_origin"
                ),
                "runtime_joint_summary_semantics": "non_fixed_joints_only",
                "runtime_active_local_joint_count": int(
                    active_joint_mask.sum().item()
                ),
            }
        )
        self._command_joint_indices = tuple(
            self.builder.local_joint_ids.index(joint_id)
            for joint_id in self.decoder.local_joint_ids
        )
        self._phase_feature_template = self._build_phase_feature_template()

    @torch.no_grad()
    def compute(
        self,
        *,
        time_s: torch.Tensor,
        phase_index: torch.Tensor,
        task_target: Order9TensorObjectTaskTarget,
        state: Order9TensorIsaacState,
        estimated_payload_mass_kg: torch.Tensor,
        estimated_payload_inertia_body: torch.Tensor,
        payload_active: torch.Tensor,
        estimated_payload_com_object: torch.Tensor | None = None,
        privileged_disturbance_body: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> Order9TensorPiLStep:
        self._validate_step_inputs(
            time_s=time_s,
            phase_index=phase_index,
            task_target=task_target,
            state=state,
            estimated_payload_mass_kg=estimated_payload_mass_kg,
            estimated_payload_inertia_body=estimated_payload_inertia_body,
            estimated_payload_com_object=estimated_payload_com_object,
            payload_active=payload_active,
            privileged_disturbance_body=privileged_disturbance_body,
        )
        control_model = self.builder.build(
            module_pose_world=state.module_pose_world,
            module_twist_world=state.module_twist_world,
            local_joint_positions_rad=state.local_joint_positions_rad,
        )
        policy_module_pose_world = state.module_pose_world
        if self.policy_frame_origins_world is not None:
            policy_module_pose_world = state.module_pose_world.clone()
            policy_module_pose_world[..., :3].sub_(
                self.policy_frame_origins_world[:, None, :]
            )
        graph_batch = self.bucket.update_runtime_(
            module_pose_world=policy_module_pose_world,
            module_twist_world=state.module_twist_world,
            module_health=self.module_health,
            joint_positions=state.local_joint_positions_rad,
            joint_velocities=state.local_joint_velocities_radps,
            joint_mask=self.joint_mask,
        )
        actor_features = order9_low_level_actor_features_from_tensors(
            Order9CentroidalTensorObservation(
                time_s=time_s,
                module_count=torch.full_like(
                    time_s, float(self.builder.module_count)
                ),
                total_mass_kg=control_model.total_mass_kg,
                inertia_body=control_model.inertia_body,
                body_pose_world=control_model.body_pose_world,
                body_twist_world=control_model.body_twist_world,
                target_pose_world=task_target.desired_robot_root_pose_world,
                target_twist=task_target.desired_robot_root_twist_world,
                controller_qp_feasible=self.controller_qp_feasible,
                controller_status_one_hot=self.controller_status_one_hot,
                allocation_residual_norm=self.allocation_residual_norm,
                task_progress_ratio=task_target.phase_progress,
                task_success=self.task_success,
            ),
            max_modules=self.config.max_modules,
        )
        phase_features = self._phase_feature_template.clone()
        phase_offset = len(TaskType)
        progress_offset = phase_offset + self.config.max_phase_count
        phase_features[:, phase_offset:progress_offset].zero_()
        phase_features.scatter_(
            1,
            (phase_offset + phase_index.long()).unsqueeze(1),
            1.0,
        )
        phase_features[:, progress_offset] = task_target.phase_progress
        previous = self.previous_action.clone()
        recurrent_in = self.recurrent_state.clone()
        actor_qp = self.controller_qp_feasible.clone()
        actor_status = self.controller_status_one_hot.clone()
        actor_residual = self.allocation_residual_norm.clone()
        actor_success = self.task_success.clone()
        privileged = (
            torch.zeros(
                (self.batch_size, 6), device=self.device, dtype=self.dtype
            )
            if privileged_disturbance_body is None
            else privileged_disturbance_body
        )
        policy_step = self.policy.step(
            graph_batch,
            None,
            actor_features,
            previous,
            recurrent_in,
            phase_features=phase_features,
            privileged_disturbance_body=privileged,
            deterministic=deterministic,
        )
        command_reference_q = task_target.nominal_joint_positions_rad
        command_reference_qdot = task_target.nominal_joint_velocities_radps
        command_mask = torch.ones_like(command_reference_q, dtype=torch.bool)
        command = self.decoder.decode(
            reference_body_pose_world=task_target.desired_robot_root_pose_world,
            reference_body_twist=task_target.desired_robot_root_twist_world,
            normalized_global_action=policy_step.action,
            normalized_joint_action=policy_step.joint_action,
            policy_module_ids=policy_step.graph_encoding.module_ids,
            reference_local_joint_positions_rad=command_reference_q,
            reference_local_joint_velocities_radps=command_reference_qdot,
            reference_local_joint_mask=command_mask,
            total_mass_kg=control_model.total_mass_kg,
        )
        payload_offset_body = self._payload_offset_body(
            control_model.body_pose_world,
            state.object_pose_world,
            estimated_payload_com_object=(
                torch.zeros(
                    (self.batch_size, 3),
                    device=self.device,
                    dtype=self.dtype,
                )
                if estimated_payload_com_object is None
                else estimated_payload_com_object
            ),
        )
        controller_result = self.controller.compute(
            control_model=control_model,
            desired_body_pose_world=command.desired_body_pose_world,
            desired_body_twist=command.desired_body_twist,
            residual_wrench_body=command.residual_wrench_body,
            state=self.controller_state,
            payload_active=payload_active,
            payload_mass_kg=estimated_payload_mass_kg,
            payload_inertia_body=estimated_payload_inertia_body,
            payload_com_offset_body=payload_offset_body,
        )
        self.previous_action.copy_(policy_step.action)
        self.recurrent_state.copy_(policy_step.recurrent_state)
        self.controller_state = controller_result.next_state
        self.controller_qp_feasible.copy_(controller_result.allocation.feasible)
        self.allocation_residual_norm.copy_(
            controller_result.allocation.residual_norm
        )
        self.controller_status_one_hot.zero_()
        status_index = torch.where(
            controller_result.allocation.feasible,
            torch.zeros_like(phase_index),
            torch.full_like(phase_index, 2),
        )
        self.controller_status_one_hot.scatter_(
            1, status_index.long().unsqueeze(1), 1.0
        )
        return Order9TensorPiLStep(
            control_model=control_model,
            actor_features=actor_features,
            phase_features=phase_features,
            previous_global_action=previous,
            recurrent_state_in=recurrent_in,
            actor_controller_qp_feasible=actor_qp,
            actor_controller_status_one_hot=actor_status,
            actor_allocation_residual_norm=actor_residual,
            actor_task_success=actor_success,
            policy_step=policy_step,
            policy_command=command,
            controller_result=controller_result,
            privileged_disturbance_body=privileged,
        )

    @torch.no_grad()
    def evaluate_bootstrap_value(
        self,
        *,
        time_s: torch.Tensor,
        phase_index: torch.Tensor,
        task_target: Order9TensorObjectTaskTarget,
        state: Order9TensorIsaacState,
        estimated_payload_mass_kg: torch.Tensor,
        estimated_payload_inertia_body: torch.Tensor,
        payload_active: torch.Tensor,
        estimated_payload_com_object: torch.Tensor | None = None,
        privileged_disturbance_body: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate the next-state critic without advancing runtime state.

        A rollout truncation bootstraps from the post-transition observation and
        the recurrent/controller memory produced by the applied action.  Calling
        :meth:`compute` is the exact policy path, but its state changes must not
        leak into the next real action.
        """

        snapshot = (
            self.previous_action.clone(),
            self.recurrent_state.clone(),
            _clone_controller_state(self.controller_state),
            self.controller_qp_feasible.clone(),
            self.controller_status_one_hot.clone(),
            self.allocation_residual_norm.clone(),
            self.task_success.clone(),
        )
        try:
            result = self.compute(
                time_s=time_s,
                phase_index=phase_index,
                task_target=task_target,
                state=state,
                estimated_payload_mass_kg=estimated_payload_mass_kg,
                estimated_payload_inertia_body=estimated_payload_inertia_body,
                estimated_payload_com_object=estimated_payload_com_object,
                payload_active=payload_active,
                privileged_disturbance_body=privileged_disturbance_body,
                deterministic=True,
            )
            return result.policy_step.value.clone()
        finally:
            (
                previous,
                recurrent,
                controller_state,
                qp_feasible,
                status,
                residual,
                task_success,
            ) = snapshot
            self.previous_action.copy_(previous)
            self.recurrent_state.copy_(recurrent)
            self.controller_state = controller_state
            self.controller_qp_feasible.copy_(qp_feasible)
            self.controller_status_one_hot.copy_(status)
            self.allocation_residual_norm.copy_(residual)
            self.task_success.copy_(task_success)

    def finish_transition(
        self,
        *,
        phase_success: torch.Tensor,
        terminal_or_reset: torch.Tensor,
        current_vectoring_angles_rad: torch.Tensor,
    ) -> None:
        if phase_success.shape != (self.batch_size,) or terminal_or_reset.shape != (
            self.batch_size,
        ):
            raise ValueError("Order9 transition result shape differs")
        self.task_success.copy_(phase_success)
        env_ids = torch.nonzero(terminal_or_reset, as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return
        self.previous_action[env_ids] = 0.0
        self.recurrent_state[env_ids] = 0.0
        self.controller_state = self.controller.reset_state_subset(
            self.controller_state,
            env_ids,
            current_vectoring_angles_rad=current_vectoring_angles_rad,
        )
        self.controller_qp_feasible[env_ids] = True
        self.controller_status_one_hot[env_ids] = 0.0
        self.controller_status_one_hot[env_ids, 0] = 1.0
        self.allocation_residual_norm[env_ids] = 0.0
        self.task_success[env_ids] = False

    def _build_phase_feature_template(self) -> torch.Tensor:
        values = torch.zeros(
            (self.batch_size, self.config.phase_feature_dim),
            device=self.device,
            dtype=self.dtype,
        )
        values[:, list(TaskType).index(TaskType.OBJECT_GRASP_CARRY)] = 1.0
        progress_offset = len(TaskType) + self.config.max_phase_count
        adapter = int.from_bytes(
            hashlib.sha256(ORDER9_OBJECT_TASK_ADAPTER_ID.encode("utf-8")).digest()[:8],
            "big",
        ) / float(2**64 - 1)
        values[:, progress_offset + 1] = math.sin(2.0 * math.pi * adapter)
        values[:, progress_offset + 2] = math.cos(2.0 * math.pi * adapter)
        return values

    def _prepare_policy_frame_origins(
        self, origins_world: torch.Tensor | None
    ) -> torch.Tensor | None:
        if origins_world is None:
            return None
        if tuple(origins_world.shape) != (self.batch_size, 3):
            raise ValueError(
                "Order9 policy-frame origins must have shape [batch_size, 3]"
            )
        origins = origins_world.to(device=self.device, dtype=self.dtype).clone()
        if not bool(torch.isfinite(origins).all().item()):
            raise ValueError("Order9 policy-frame origins must be finite")
        return origins

    @staticmethod
    def _payload_offset_body(
        body_pose_world: torch.Tensor,
        object_pose_world: torch.Tensor,
        *,
        estimated_payload_com_object: torch.Tensor,
    ) -> torch.Tensor:
        body_rotation = _quaternion_to_matrix(body_pose_world[:, 3:7])
        object_rotation = _quaternion_to_matrix(object_pose_world[:, 3:7])
        estimated_com_world = object_pose_world[:, :3] + (
            object_rotation @ estimated_payload_com_object.unsqueeze(-1)
        ).squeeze(-1)
        offset_world = estimated_com_world - body_pose_world[:, :3]
        return (
            body_rotation.transpose(-1, -2) @ offset_world.unsqueeze(-1)
        ).squeeze(-1)

    def _validate_step_inputs(self, **values) -> None:
        batch = self.batch_size
        expected = {
            "time_s": (batch,),
            "phase_index": (batch,),
            "estimated_payload_mass_kg": (batch,),
            "estimated_payload_inertia_body": (batch, 6),
            "payload_active": (batch,),
        }
        for name, shape in expected.items():
            if tuple(values[name].shape) != shape:
                raise ValueError(f"Order9 tensor runtime {name} shape differs")
        estimated_com = values["estimated_payload_com_object"]
        if estimated_com is not None and tuple(estimated_com.shape) != (batch, 3):
            raise ValueError(
                "Order9 tensor runtime estimated_payload_com_object shape differs"
            )
        if bool((values["phase_index"] < 0).any()) or bool(
            (
                values["phase_index"]
                >= min(
                    self.config.max_phase_count,
                    ORDER9_OBJECT_TASK_ACTOR_PHASE_COUNT,
                )
            ).any()
        ):
            raise ValueError("Order9 tensor runtime phase index is invalid")
        state = values["state"]
        if state.module_pose_world.shape != (batch, self.builder.module_count, 7):
            raise ValueError("Order9 tensor runtime module pose shape differs")
        target = values["task_target"]
        if target.desired_robot_root_pose_world.shape != (batch, 7):
            raise ValueError("Order9 tensor runtime target pose shape differs")
        if (
            target.desired_robot_root_twist_world.shape != (batch, 6)
            or target.desired_object_pose_world.shape != (batch, 7)
            or target.phase_goal_robot_root_pose_world.shape != (batch, 7)
            or target.phase_goal_object_pose_world.shape != (batch, 7)
        ):
            raise ValueError("Order9 tensor runtime task-target shape differs")
        expected_joint_shape = (
            batch,
            self.builder.module_count,
            len(self._command_joint_indices),
        )
        if (
            target.nominal_joint_positions_rad.shape != expected_joint_shape
            or target.nominal_joint_velocities_radps.shape
            != expected_joint_shape
        ):
            raise ValueError(
                "Order9 tensor runtime target joint posture shape differs"
            )
        privileged = values["privileged_disturbance_body"]
        if privileged is not None and privileged.shape != (batch, 6):
            raise ValueError("Order9 privileged disturbance shape differs")


def _quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    q = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    x, y, z, w = q.unbind(dim=-1)
    return torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)


def _clone_controller_state(state: BatchedQPIDState) -> BatchedQPIDState:
    return BatchedQPIDState(
        position_error_integral_world=(
            state.position_error_integral_world.clone()
        ),
        attitude_error_integral_body=(
            state.attitude_error_integral_body.clone()
        ),
        previous_rotor_thrusts_n=state.previous_rotor_thrusts_n.clone(),
        previous_vectoring_targets_rad=(
            state.previous_vectoring_targets_rad.clone()
        ),
    )


__all__ = [
    "ORDER9_TENSOR_PI_L_RUNTIME_VERSION",
    "Order9TensorPiLRuntime",
    "Order9TensorPiLStep",
]
