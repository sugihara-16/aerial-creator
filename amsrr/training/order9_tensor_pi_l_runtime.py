from __future__ import annotations

"""GPU-resident ``pi_L -> PolicyCommand -> QPID/QP`` Order 9 hot path."""

import hashlib
import math
from dataclasses import dataclass, replace

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
    ORDER9_CONTACT_FEEDBACK_FEATURE_NAMES,
    ORDER9_GLOBAL_ACTION_SIZE,
    Order9ActiveKnotPhaseConditionedActorCritic,
    Order9ContactSpacePhaseConditionedActorCritic,
    Order9ContactFeedbackPhaseConditionedActorCritic,
    Order9LowLevelActorCriticStep,
    Order9LowLevelPolicyConfig,
    Order9PhaseConditionedActorCritic,
)
from amsrr.policies.order9_active_knot_features import (
    Order9ActiveKnotTensorTemplate,
)
from amsrr.policies.order9_tensor_command_decoder import (
    ORDER9_INDEPENDENT_COMPRESSION_ACTION_ADAPTER_VERSION,
    ORDER9_MORPHOLOGY_INVARIANT_COMPRESSION_ACTION_ADAPTER_VERSION,
    Order9TensorPolicyCommand,
    Order9TensorPolicyCommandDecoder,
)
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import ContactWrenchTrajectory
from amsrr.schemas.task_spec import TaskType
from amsrr.simulation.order9_object_task_runtime import (
    ORDER9_OBJECT_TASK_ADAPTER_ID,
    ORDER9_OBJECT_TASK_ACTOR_PHASE_COUNT,
)
from amsrr.simulation.order9_tensor_isaac_io import Order9TensorIsaacState
from amsrr.simulation.order9_tensor_object_task import (
    ORDER9_CONTACT_SCHEDULE_ATTACH,
    ORDER9_CONTACT_SCHEDULE_MAINTAIN,
    Order9TensorObjectTaskTarget,
)
from amsrr.training.order9_tensor_runtime import (
    Order9CentroidalTensorObservation,
    Order9TensorizedTopologyBucket,
    order9_low_level_actor_features_from_tensors,
)
from amsrr.training.order9_c3_action_contract import (
    ORDER9_C3_ACTION_CONTRACT_CONTACT_SPACE_PROJECTED,
    ORDER9_C3_ACTION_CONTRACTS,
    order9_c3_action_contract_global_dimension,
    order9_c3_action_contract_uses_full_policy,
    order9_c3_action_contract_uses_contact_space,
)
from amsrr.training.order9_contact_space_action import (
    Order9ContactSpaceActionBasis,
    apply_order9_diagnostic_contact_normal_residual,
    apply_order9_contact_space_action,
)


ORDER9_TENSOR_PI_L_RUNTIME_VERSION = "order9_tensor_contact_space_pi_l_qpid_runtime_v11"
ORDER9_NOMINAL_ONLY_ACTION_MASK_CONTRACT = (
    "order9_place_complete_release_retreat_nominal_only_safety_mask_v2"
)
ORDER9_DIAGNOSTIC_ACTION_ABLATION_MODES = frozenset(
    {
        "zero_global",
        "zero_joint",
        "contact_compression_only",
        "contact_compression_plus_centroidal",
        "contact_compression_plus_global",
    }
)


@dataclass(frozen=True)
class Order9TensorPiLStep:
    control_model: BatchedRigidBodyControlModel
    actor_features: torch.Tensor
    phase_features: torch.Tensor
    active_knot_features: torch.Tensor | None
    active_assignment_features: torch.Tensor | None
    contact_slot_features: torch.Tensor | None
    contact_slot_owner_module_indices: torch.Tensor | None
    contact_slot_mask: torch.Tensor | None
    contact_constraint_weight: torch.Tensor | None
    previous_global_action: torch.Tensor
    applied_global_action: torch.Tensor
    recurrent_state_in: torch.Tensor
    actor_controller_qp_feasible: torch.Tensor
    actor_controller_status_one_hot: torch.Tensor
    actor_allocation_residual_norm: torch.Tensor
    actor_task_success: torch.Tensor
    policy_step: Order9LowLevelActorCriticStep
    policy_command: Order9TensorPolicyCommand
    controller_result: BatchedQPIDResult
    privileged_disturbance_body: torch.Tensor


@dataclass(frozen=True)
class Order9TensorNominalQPIDStep:
    """Actor-free nominal hold command evaluated by the production QPID/QP."""

    control_model: BatchedRigidBodyControlModel
    policy_command: Order9TensorPolicyCommand
    controller_result: BatchedQPIDResult


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
        active_knot_trajectory: ContactWrenchTrajectory | None = None,
        contact_space_action_basis: Order9ContactSpaceActionBasis | None = None,
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
        self._command_joint_index_tensor = torch.tensor(
            self._command_joint_indices,
            device=self.device,
            dtype=torch.long,
        )
        self._decoder_module_ids = torch.tensor(
            self.builder.module_ids,
            device=self.device,
            dtype=torch.long,
        ).reshape(1, -1).expand(self.batch_size, -1)
        self._phase_feature_template = self._build_phase_feature_template()
        self._active_knot_template = None
        if isinstance(
            self.policy, Order9ActiveKnotPhaseConditionedActorCritic
        ):
            if active_knot_trajectory is None:
                raise ValueError(
                    "Order9 active-knot tensor policy requires its trajectory"
                )
            self._active_knot_template = Order9ActiveKnotTensorTemplate(
                trajectory=active_knot_trajectory,
                morphology_graph=self.morphology_graph,
                physical_model=self.physical_model,
                module_ids=self.builder.module_ids,
                batch_size=self.batch_size,
                device=self.device,
                dtype=self.dtype,
            )
        self.contact_space_action_basis = None
        if isinstance(self.policy, Order9ContactSpacePhaseConditionedActorCritic):
            if contact_space_action_basis is None:
                raise ValueError(
                    "Order9 v7 tensor policy requires its contact-space basis"
                )
            if contact_space_action_basis.module_ids != self.builder.module_ids:
                raise ValueError("Order9 contact-space basis module ids differ")
            if contact_space_action_basis.local_joint_ids != self.decoder.local_joint_ids:
                raise ValueError("Order9 contact-space basis joint ids differ")
            self.contact_space_action_basis = contact_space_action_basis.to(
                device=self.device, dtype=self.dtype
            )
        elif contact_space_action_basis is not None:
            raise ValueError("Order9 legacy tensor policy cannot use a v7 basis")
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
        hardware_joint_load_nm: torch.Tensor | None = None,
        estimated_payload_com_object: torch.Tensor | None = None,
        privileged_disturbance_body: torch.Tensor | None = None,
        contact_compression_joint_direction_rad: torch.Tensor | None = None,
        contact_compression_action_limit: float | torch.Tensor | None = None,
        contact_compression_only_action: bool = False,
        joint_only_action: bool = False,
        nominal_only_action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
        diagnostic_action_ablation: str | None = None,
        diagnostic_contact_normal_residual_m: torch.Tensor | None = None,
        contact_feedback_features: torch.Tensor | None = None,
        c3_action_contract: str | None = None,
    ) -> Order9TensorPiLStep:
        if c3_action_contract is not None and c3_action_contract not in (
            ORDER9_C3_ACTION_CONTRACTS
        ):
            raise ValueError("Order9 C3 action contract is invalid")
        if (
            diagnostic_action_ablation is not None
            and diagnostic_action_ablation
            not in ORDER9_DIAGNOSTIC_ACTION_ABLATION_MODES
        ):
            raise ValueError("Order9 diagnostic action ablation mode is invalid")
        if diagnostic_action_ablation is not None and not deterministic:
            raise ValueError(
                "Order9 diagnostic action ablation requires deterministic inference"
            )
        if diagnostic_contact_normal_residual_m is not None and not deterministic:
            raise ValueError(
                "Order9 diagnostic contact-normal override requires deterministic inference"
            )
        if diagnostic_contact_normal_residual_m is not None and (
            diagnostic_action_ablation is not None
            or c3_action_contract
            != ORDER9_C3_ACTION_CONTRACT_CONTACT_SPACE_PROJECTED
        ):
            raise ValueError(
                "Order9 diagnostic contact-normal override requires only the "
                "contact-space action contract"
            )
        if diagnostic_action_ablation is not None and (
            contact_compression_only_action
            or joint_only_action
            or c3_action_contract is not None
        ):
            raise ValueError(
                "Order9 production action restriction and diagnostic ablation "
                "are mutually exclusive"
            )
        if contact_compression_only_action and joint_only_action:
            raise ValueError(
                "Order9 compression-only and joint-only production actions "
                "are mutually exclusive"
            )
        if c3_action_contract is not None and (
            contact_compression_only_action or joint_only_action
        ):
            raise ValueError(
                "Order9 explicit C3 action contract and production module-count "
                "restriction are mutually exclusive"
            )
        if nominal_only_action_mask is not None and (
            nominal_only_action_mask.dtype != torch.bool
            or tuple(nominal_only_action_mask.shape) != (self.batch_size,)
        ):
            raise ValueError("Order9 nominal-only action mask shape or dtype differs")
        contact_space_contract = (
            c3_action_contract is not None
            and order9_c3_action_contract_uses_contact_space(c3_action_contract)
        )
        if (
            (contact_compression_only_action or (
                c3_action_contract is not None and not contact_space_contract
            ))
            and contact_compression_joint_direction_rad is None
        ):
            raise ValueError(
                "Order9 contact-compression action contract requires its IK direction"
            )
        if hardware_joint_load_nm is not None and tuple(
            hardware_joint_load_nm.shape
        ) != (
            self.batch_size,
            self.builder.module_count,
            len(self.decoder.local_joint_ids),
        ):
            raise ValueError(
                "Order9 deployable hardware joint load shape differs"
            )
        if hardware_joint_load_nm is not None and not bool(
            torch.isfinite(hardware_joint_load_nm).all()
        ):
            raise ValueError(
                "Order9 deployable hardware joint load is non-finite"
            )
        feedback_policy = isinstance(
            self.policy, Order9ContactFeedbackPhaseConditionedActorCritic
        )
        if feedback_policy:
            expected_feedback_shape = (
                self.batch_size,
                int(self.config.max_contact_slots),
                len(ORDER9_CONTACT_FEEDBACK_FEATURE_NAMES),
            )
            if contact_feedback_features is None or tuple(
                contact_feedback_features.shape
            ) != expected_feedback_shape:
                raise ValueError(
                    "Order9 v8 pi_L requires complete deployable contact feedback"
                )
            if not bool(torch.isfinite(contact_feedback_features).all()):
                raise ValueError("Order9 v8 contact feedback is non-finite")
        elif contact_feedback_features is not None:
            raise ValueError("Order9 v7-or-earlier pi_L cannot consume v8 feedback")
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
        active_knot_features = None
        active_assignment_features = None
        contact_slot_features = None
        contact_slot_owner_module_indices = None
        contact_slot_mask = None
        if self._active_knot_template is not None:
            deployable_joint_load = (
                torch.zeros(
                    (
                        self.batch_size,
                        self.builder.module_count,
                        len(self.decoder.local_joint_ids),
                    ),
                    device=self.device,
                    dtype=self.dtype,
                )
                if hardware_joint_load_nm is None
                else hardware_joint_load_nm
            )
            (
                active_knot_features,
                active_assignment_features,
            ) = self._active_knot_template.features(
                time_s=time_s,
                phase_progress=task_target.phase_progress,
                contact_schedule_index=task_target.contact_schedule_index,
                body_pose_world=control_model.body_pose_world,
                body_twist_world=control_model.body_twist_world,
                object_pose_world=state.object_pose_world,
                object_twist_world=state.object_twist_world,
                desired_body_pose_world=(
                    task_target.desired_robot_root_pose_world
                ),
                desired_body_twist_world=(
                    task_target.desired_robot_root_twist_world
                ),
                desired_object_pose_world=task_target.desired_object_pose_world,
                current_joint_positions_rad=(
                    state.local_joint_positions_rad.index_select(
                        -1, self._command_joint_index_tensor
                    )
                ),
                current_joint_velocities_radps=(
                    state.local_joint_velocities_radps.index_select(
                        -1, self._command_joint_index_tensor
                    )
                ),
                desired_joint_positions_rad=(
                    task_target.nominal_joint_positions_rad
                ),
                desired_joint_velocities_radps=(
                    task_target.nominal_joint_velocities_radps
                ),
                controller_qp_feasible=actor_qp,
                controller_status_one_hot=actor_status,
                allocation_residual_norm=actor_residual,
                hardware_joint_load_nm=deployable_joint_load,
            )
        if self.contact_space_action_basis is not None:
            contact_slot_features = (
                self.contact_space_action_basis.contact_slot_features.unsqueeze(0).expand(
                    self.batch_size, -1, -1
                )
            )
            if feedback_policy:
                contact_slot_features = torch.cat(
                    (
                        contact_slot_features,
                        contact_feedback_features.to(
                            device=self.device, dtype=self.dtype
                        ),
                    ),
                    dim=-1,
                )
            contact_slot_owner_module_indices = (
                self.contact_space_action_basis.contact_slot_owner_module_indices.unsqueeze(0).expand(
                    self.batch_size, -1
                )
            )
            contact_slot_mask = (
                self.contact_space_action_basis.contact_slot_mask.unsqueeze(0).expand(
                    self.batch_size, -1
                )
            )
        active_kwargs = (
            {}
            if active_knot_features is None
            else {
                "active_knot_features": active_knot_features,
                "active_assignment_features": active_assignment_features,
            }
        )
        if contact_slot_features is not None:
            active_kwargs.update(
                {
                    "contact_slot_features": contact_slot_features,
                    "contact_slot_owner_module_indices": (
                        contact_slot_owner_module_indices
                    ),
                    "contact_slot_mask": contact_slot_mask,
                }
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
            **active_kwargs,
        )
        if diagnostic_contact_normal_residual_m is not None:
            if self.contact_space_action_basis is None:
                raise ValueError(
                    "Order9 diagnostic contact-normal override lacks its basis"
                )
            overridden_contact_action = apply_order9_diagnostic_contact_normal_residual(
                normalized_contact_action=policy_step.contact_space_residual_action,
                inward_residual_m=diagnostic_contact_normal_residual_m,
                basis=self.contact_space_action_basis,
            )
            policy_step = replace(
                policy_step,
                contact_space_residual_action=overridden_contact_action,
                contact_space_residual_action_mean=overridden_contact_action,
            )
        applied_global_action = policy_step.action
        applied_joint_action = policy_step.joint_action
        contact_projection = None
        if c3_action_contract is not None:
            if contact_space_contract:
                if self.contact_space_action_basis is None:
                    raise ValueError(
                        "Order9 contact-space contract lacks its basis"
                    )
                contact_projection = apply_order9_contact_space_action(
                    normalized_global_action=policy_step.action,
                    normalized_joint_action=policy_step.joint_action,
                    normalized_contact_action=(
                        policy_step.contact_space_residual_action
                    ),
                    phase_index=phase_index,
                    phase_progress=task_target.phase_progress,
                    basis=self.contact_space_action_basis,
                    policy_config=self.config,
                    maximum_combined_joint_delta_rad=float(
                        self.config.joint_position_delta_limit_rad
                    ),
                )
                applied_global_action = contact_projection.global_action
                applied_joint_action = contact_projection.joint_action
            elif not order9_c3_action_contract_uses_full_policy(
                c3_action_contract
            ):
                _, applied_joint_action = _contact_compression_only_actions(
                    policy_step,
                    contact_compression_joint_direction_rad=(
                        contact_compression_joint_direction_rad
                    ),
                    contact_schedule_index=task_target.contact_schedule_index,
                    local_joint_slot_count=len(self.decoder.local_joint_ids),
                )
            if not contact_space_contract:
                global_dimension = order9_c3_action_contract_global_dimension(
                    c3_action_contract
                )
                applied_global_action = torch.zeros_like(policy_step.action)
                applied_global_action[:, :global_dimension] = policy_step.action[
                    :, :global_dimension
                ]
        elif contact_compression_only_action:
            applied_global_action, applied_joint_action = (
                _contact_compression_only_actions(
                    policy_step,
                    contact_compression_joint_direction_rad=(
                        contact_compression_joint_direction_rad
                    ),
                    contact_schedule_index=task_target.contact_schedule_index,
                    local_joint_slot_count=len(self.decoder.local_joint_ids),
                )
            )
        elif joint_only_action:
            applied_global_action = torch.zeros_like(applied_global_action)
        if diagnostic_action_ablation is not None:
            policy_step = _diagnostic_action_ablation(
                policy_step,
                mode=diagnostic_action_ablation,
                contact_compression_joint_direction_rad=(
                    contact_compression_joint_direction_rad
                ),
                contact_schedule_index=task_target.contact_schedule_index,
                local_joint_slot_count=len(self.decoder.local_joint_ids),
            )
            applied_global_action = policy_step.action
            applied_joint_action = policy_step.joint_action
        if nominal_only_action_mask is not None:
            applied_global_action = torch.where(
                nominal_only_action_mask[:, None],
                torch.zeros_like(applied_global_action),
                applied_global_action,
            )
            applied_joint_action = torch.where(
                nominal_only_action_mask[:, None, None],
                torch.zeros_like(applied_joint_action),
                applied_joint_action,
            )
        command_reference_q = task_target.nominal_joint_positions_rad
        command_reference_qdot = task_target.nominal_joint_velocities_radps
        command_mask = torch.ones_like(command_reference_q, dtype=torch.bool)
        command = self.decoder.decode(
            reference_body_pose_world=task_target.desired_robot_root_pose_world,
            reference_body_twist=task_target.desired_robot_root_twist_world,
            normalized_global_action=applied_global_action,
            normalized_joint_action=applied_joint_action,
            policy_module_ids=policy_step.graph_encoding.module_ids,
            reference_local_joint_positions_rad=command_reference_q,
            reference_local_joint_velocities_radps=command_reference_qdot,
            reference_local_joint_mask=command_mask,
            total_mass_kg=control_model.total_mass_kg,
            contact_compression_joint_direction_rad=(
                None
                if contact_space_contract
                else contact_compression_joint_direction_rad
            ),
            contact_compression_action_mask=(
                None
                if (
                    contact_compression_joint_direction_rad is None
                    or contact_space_contract
                )
                else (
                    (
                        task_target.contact_schedule_index
                        == ORDER9_CONTACT_SCHEDULE_ATTACH
                    )
                    | (
                        task_target.contact_schedule_index
                        == ORDER9_CONTACT_SCHEDULE_MAINTAIN
                    )
                )
            ),
            normalized_contact_compression_residual_action=(
                None
                if contact_space_contract
                else policy_step.contact_compression_residual_action
            ),
            contact_compression_action_limit=contact_compression_action_limit,
            contact_compression_action_adapter_version=(
                ORDER9_INDEPENDENT_COMPRESSION_ACTION_ADAPTER_VERSION
                if (
                    c3_action_contract is not None
                    and order9_c3_action_contract_uses_full_policy(
                        c3_action_contract
                    )
                )
                else (
                    ORDER9_MORPHOLOGY_INVARIANT_COMPRESSION_ACTION_ADAPTER_VERSION
                )
            ),
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
        self.previous_action.copy_(
            applied_global_action
            if c3_action_contract is not None
            else policy_step.action
        )
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
            active_knot_features=active_knot_features,
            active_assignment_features=active_assignment_features,
            contact_slot_features=contact_slot_features,
            contact_slot_owner_module_indices=(
                contact_slot_owner_module_indices
            ),
            contact_slot_mask=contact_slot_mask,
            contact_constraint_weight=(
                None
                if contact_projection is None
                else contact_projection.contact_constraint_weight
            ),
            previous_global_action=previous,
            applied_global_action=applied_global_action,
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
    def compute_nominal_qpid_hold(
        self,
        *,
        task_target: Order9TensorObjectTaskTarget,
        state: Order9TensorIsaacState,
        estimated_payload_mass_kg: torch.Tensor,
        estimated_payload_inertia_body: torch.Tensor,
        payload_active: torch.Tensor,
        estimated_payload_com_object: torch.Tensor | None = None,
    ) -> Order9TensorNominalQPIDStep:
        """Hold one nominal target with QPID/QP while bypassing ``pi_L``.

        Reset construction needs a policy-independent physical settling path.
        The command therefore uses the exact nominal centroidal pose and joint
        posture, zero desired velocities, zero residual wrench, and zero joint
        torque bias.  Payload gravity/inertia feedforward and the production
        QP allocation remain active.  Actor actions and recurrent state are not
        evaluated or advanced.
        """

        self._validate_controller_inputs(
            task_target=task_target,
            state=state,
            estimated_payload_mass_kg=estimated_payload_mass_kg,
            estimated_payload_inertia_body=estimated_payload_inertia_body,
            estimated_payload_com_object=estimated_payload_com_object,
            payload_active=payload_active,
        )
        control_model = self.builder.build(
            module_pose_world=state.module_pose_world,
            module_twist_world=state.module_twist_world,
            local_joint_positions_rad=state.local_joint_positions_rad,
        )
        command_reference_q = task_target.nominal_joint_positions_rad
        command = self.decoder.decode(
            reference_body_pose_world=task_target.desired_robot_root_pose_world,
            reference_body_twist=torch.zeros_like(
                task_target.desired_robot_root_twist_world
            ),
            normalized_global_action=torch.zeros(
                (self.batch_size, ORDER9_GLOBAL_ACTION_SIZE),
                device=self.device,
                dtype=self.dtype,
            ),
            normalized_joint_action=torch.zeros(
                (
                    self.batch_size,
                    self.builder.module_count,
                    3 * self.config.max_local_joint_slots,
                ),
                device=self.device,
                dtype=self.dtype,
            ),
            policy_module_ids=self._decoder_module_ids,
            reference_local_joint_positions_rad=command_reference_q,
            reference_local_joint_velocities_radps=torch.zeros_like(
                task_target.nominal_joint_velocities_radps
            ),
            reference_local_joint_mask=torch.ones_like(
                command_reference_q, dtype=torch.bool
            ),
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
        self.controller_state = controller_result.next_state
        self.controller_qp_feasible.copy_(controller_result.allocation.feasible)
        self.allocation_residual_norm.copy_(
            controller_result.allocation.residual_norm
        )
        self.controller_status_one_hot.zero_()
        status_index = torch.where(
            controller_result.allocation.feasible,
            torch.zeros(
                self.batch_size, device=self.device, dtype=torch.long
            ),
            torch.full(
                (self.batch_size,), 2, device=self.device, dtype=torch.long
            ),
        )
        self.controller_status_one_hot.scatter_(
            1, status_index.unsqueeze(1), 1.0
        )
        return Order9TensorNominalQPIDStep(
            control_model=control_model,
            policy_command=command,
            controller_result=controller_result,
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
        hardware_joint_load_nm: torch.Tensor | None = None,
        contact_feedback_features: torch.Tensor | None = None,
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
                hardware_joint_load_nm=hardware_joint_load_nm,
                contact_feedback_features=contact_feedback_features,
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
        }
        for name, shape in expected.items():
            if tuple(values[name].shape) != shape:
                raise ValueError(f"Order9 tensor runtime {name} shape differs")
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
        self._validate_controller_inputs(**values)
        privileged = values["privileged_disturbance_body"]
        if privileged is not None and privileged.shape != (batch, 6):
            raise ValueError("Order9 privileged disturbance shape differs")

    def _validate_controller_inputs(self, **values) -> None:
        batch = self.batch_size
        expected = {
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


def _diagnostic_action_ablation(
    policy_step: Order9LowLevelActorCriticStep,
    *,
    mode: str,
    contact_compression_joint_direction_rad: torch.Tensor | None,
    contact_schedule_index: torch.Tensor,
    local_joint_slot_count: int,
) -> Order9LowLevelActorCriticStep:
    """Ablate applied deterministic actions without bypassing actor state.

    This path is diagnostic-only.  It preserves the policy's recurrent-state
    evolution while making the command actually applied to QPID explicit in
    the persisted rollout artifact.
    """

    global_action = policy_step.action
    global_mean = policy_step.action_mean
    joint_action = policy_step.joint_action
    joint_mean = policy_step.joint_action_mean
    if mode == "zero_global":
        global_action = torch.zeros_like(global_action)
        global_mean = torch.zeros_like(global_mean)
    elif mode == "zero_joint":
        joint_action = torch.zeros_like(joint_action)
        joint_mean = torch.zeros_like(joint_mean)
    elif mode in {
        "contact_compression_only",
        "contact_compression_plus_centroidal",
        "contact_compression_plus_global",
    }:
        _, joint_action = _contact_compression_only_actions(
            policy_step,
            contact_compression_joint_direction_rad=(
                contact_compression_joint_direction_rad
            ),
            contact_schedule_index=contact_schedule_index,
            local_joint_slot_count=local_joint_slot_count,
        )
        _, joint_mean = _contact_compression_only_actions(
            replace(
                policy_step,
                action=policy_step.action_mean,
                joint_action=policy_step.joint_action_mean,
            ),
            contact_compression_joint_direction_rad=(
                contact_compression_joint_direction_rad
            ),
            contact_schedule_index=contact_schedule_index,
            local_joint_slot_count=local_joint_slot_count,
        )
        if mode == "contact_compression_only":
            global_action = torch.zeros_like(policy_step.action)
            global_mean = torch.zeros_like(policy_step.action_mean)
        elif mode == "contact_compression_plus_centroidal":
            global_action = policy_step.action.clone()
            global_mean = policy_step.action_mean.clone()
            # Preserve centroidal pose/twist correction while isolating the
            # separately questioned residual-wrench feed-forward channels.
            global_action[:, 12:18] = 0.0
            global_mean[:, 12:18] = 0.0
        else:
            global_action = policy_step.action
            global_mean = policy_step.action_mean
    else:  # pragma: no cover - validated at the public boundary.
        raise ValueError("Order9 diagnostic action ablation mode is invalid")
    return replace(
        policy_step,
        action=global_action,
        action_mean=global_mean,
        joint_action=joint_action,
        joint_action_mean=joint_mean,
    )


def _contact_compression_only_actions(
    policy_step: Order9LowLevelActorCriticStep,
    *,
    contact_compression_joint_direction_rad: torch.Tensor | None,
    contact_schedule_index: torch.Tensor,
    local_joint_slot_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the command-space subset used by conservative morphology entry.

    The sampled actor action and its log probability remain untouched for
    exact on-policy replay.  Only the command passed to the decoder is masked.
    """

    global_action = torch.zeros_like(policy_step.action)
    joint_action = torch.zeros_like(policy_step.joint_action)
    if contact_compression_joint_direction_rad is None:
        return global_action, joint_action
    direction = contact_compression_joint_direction_rad
    if direction.ndim == 2:
        direction = direction.unsqueeze(0).expand(
            policy_step.joint_action.shape[0], -1, -1
        )
    expected = (
        policy_step.joint_action.shape[0],
        policy_step.joint_action.shape[1],
        local_joint_slot_count,
    )
    if tuple(direction.shape) != expected:
        raise ValueError(
            "Order9 contact-compression-only direction shape differs"
        )
    flat_index = direction[0].reshape(-1).abs().argmax()
    control_module = int(flat_index.item()) // local_joint_slot_count
    control_joint = int(flat_index.item()) % local_joint_slot_count
    active = (
        (contact_schedule_index == ORDER9_CONTACT_SCHEDULE_ATTACH)
        | (contact_schedule_index == ORDER9_CONTACT_SCHEDULE_MAINTAIN)
    )
    joint_action[:, control_module, control_joint] = torch.where(
        active,
        policy_step.joint_action[:, control_module, control_joint],
        torch.zeros_like(
            policy_step.joint_action[:, control_module, control_joint]
        ),
    )
    return global_action, joint_action


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
    "Order9TensorNominalQPIDStep",
    "Order9TensorPiLRuntime",
    "Order9TensorPiLStep",
]
