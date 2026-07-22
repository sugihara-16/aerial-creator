from __future__ import annotations

"""Teacher-action encoding and BC/PPO losses for the Order 9 ``pi_L`` actor.

The conversion in this module is deliberately the inverse of the bounded
decoder owned by :mod:`amsrr.policies.morphology_conditioned_low_level_policy`.
It therefore detects teacher commands that the actor cannot represent instead
of silently training against clipped labels.
"""

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch.nn import functional as F

from amsrr.controllers.rigid_body_model import (
    RigidBodyControlModel,
    RigidBodyControlModelBuilder,
)
from amsrr.policies.low_level_policy_base import LowLevelPolicyContext
from amsrr.policies.morphology_conditioned_low_level_policy import (
    ORDER3_ACTOR_FEATURE_NAMES,
    order3_actor_feature_vector,
)
from amsrr.policies.order9_low_level_policy import (
    ORDER9_GLOBAL_ACTION_SIZE,
    Order9LowLevelPolicyConfig,
    Order9PhaseConditionedActorCritic,
    order9_phase_actor_feature_vector,
)
from amsrr.policies.order9_policy_command import (
    encode_order9_centroidal_pose_action,
    order9_joint_reference,
    order9_pi_l_reference_command,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.policies import (
    POLICY_COMMAND_CONTRACT_CENTROIDAL,
    PolicyCommand,
)
from amsrr.training.order9_pi_h_learning import clipped_ppo_surrogate_loss


ORDER9_PI_L_LEARNING_VERSION = "order9_complete_policy_command_pi_l_bc_ppo_v2"
_GRAVITY_MPS2 = 9.81
_REPRESENTABILITY_TOLERANCE = 1.0e-6


@dataclass(frozen=True)
class Order9PiLTeacherAction:
    """One teacher command expressed in the actor's normalized coordinates."""

    global_action: tuple[float, ...]
    joint_action: tuple[tuple[float, ...], ...]
    module_ids: tuple[int, ...]
    maximum_absolute_unclipped_action: float


@dataclass
class Order9PiLBehaviorCloningLoss:
    total: torch.Tensor
    global_action: torch.Tensor
    joint_action: torch.Tensor
    value: torch.Tensor
    active_joint_coordinate_count: int
    recurrent_state_out: torch.Tensor
    teacher_global_action: torch.Tensor


def encode_order9_pi_l_teacher_action(
    *,
    context: LowLevelPolicyContext,
    teacher_command: PolicyCommand,
    control_model: RigidBodyControlModel,
    config: Order9LowLevelPolicyConfig,
) -> Order9PiLTeacherAction:
    """Invert the complete deployed decoder and reject clipped demonstrations."""

    if teacher_command.control_contract_version != POLICY_COMMAND_CONTRACT_CENTROIDAL:
        raise SchemaValidationError("Order9 pi_L teacher must use the centroidal-v2 contract")
    config.validate()
    reference_command = order9_pi_l_reference_command(context)
    reference_pose = _required_pose(reference_command)
    teacher_pose = _required_pose(teacher_command)
    reference_twist = _six_vector(
        reference_command.desired_body_twist, "reference twist"
    )
    teacher_twist = _six_vector(teacher_command.desired_body_twist, "teacher twist")
    twist_limits = (
        *([config.linear_twist_correction_limit_mps] * 3),
        *([config.angular_twist_correction_limit_radps] * 3),
    )
    teacher_wrench = _six_vector(teacher_command.residual_wrench_body, "teacher wrench")
    wrench_limits = (
        *(
            [
                control_model.total_mass_kg
                * _GRAVITY_MPS2
                * config.residual_force_weight_fraction
            ]
            * 3
        ),
        *(
            [
                len(context.runtime_observation.morphology_graph.modules)
                * config.residual_torque_per_module_nm
            ]
            * 3
        ),
    )
    global_values = list(
        encode_order9_centroidal_pose_action(
            reference_pose, teacher_pose, config
        )
    )
    global_values.extend(
        (teacher_twist[index] - reference_twist[index]) / twist_limits[index]
        for index in range(6)
    )
    global_values.extend(
        teacher_wrench[index] / wrench_limits[index]
        for index in range(6)
    )
    if len(global_values) != ORDER9_GLOBAL_ACTION_SIZE:
        raise RuntimeError("Order9 pi_L global action layout drifted")

    joint_ids = _dock_mechanism_joint_ids(context.physical_model)
    if len(joint_ids) > config.max_local_joint_slots:
        raise SchemaValidationError("Order9 pi_L joint decoder has too few local slots")
    state_by_id = {
        state.module_id: state for state in context.runtime_observation.module_states
    }
    module_ids = tuple(sorted(module.module_id for module in context.morphology_graph.modules))
    joint_by_id = {joint.joint_id: joint for joint in context.physical_model.joints}
    width = 3 * config.max_local_joint_slots
    joint_values: list[tuple[float, ...]] = []
    for module_id in module_ids:
        if module_id not in state_by_id:
            raise SchemaValidationError("Order9 pi_L teacher is missing a module runtime state")
        state = state_by_id[module_id]
        row = [0.0] * width
        for slot, joint_id in enumerate(joint_ids):
            if joint_id not in state.joint_positions:
                continue
            global_id = f"module_{module_id}:{joint_id}"
            current = float(state.joint_positions[joint_id])
            q_reference, qdot_reference = order9_joint_reference(
                reference_command,
                global_joint_id=global_id,
                local_joint_id=joint_id,
                current_position_rad=current,
            )
            q_target = float(teacher_command.joint_position_targets.get(global_id, current))
            qdot_target = float(teacher_command.joint_velocity_targets.get(global_id, 0.0))
            torque_target = float(teacher_command.joint_torque_bias.get(global_id, 0.0))
            row[slot] = (
                q_target - q_reference
            ) / config.joint_position_delta_limit_rad
            row[config.max_local_joint_slots + slot] = (
                qdot_target - qdot_reference
            ) / (
                config.joint_velocity_limit_rad_s
            )
            effort_limit = float(joint_by_id[joint_id].effort_limit or 0.0)
            torque_scale = effort_limit * config.joint_torque_fraction
            if torque_scale <= 0.0:
                if not math.isclose(torque_target, 0.0, abs_tol=1.0e-12):
                    raise SchemaValidationError(
                        f"Order9 pi_L teacher torque for {global_id!r} is not representable"
                    )
                row[2 * config.max_local_joint_slots + slot] = 0.0
            else:
                row[2 * config.max_local_joint_slots + slot] = torque_target / torque_scale
        joint_values.append(tuple(row))

    all_values = [*global_values, *(value for row in joint_values for value in row)]
    if not all(math.isfinite(value) for value in all_values):
        raise SchemaValidationError("Order9 pi_L teacher action contains non-finite values")
    maximum = max((abs(value) for value in all_values), default=0.0)
    if maximum > 1.0 + _REPRESENTABILITY_TOLERANCE:
        raise SchemaValidationError(
            "Order9 pi_L teacher command exceeds the actor authority "
            f"(maximum normalized magnitude {maximum:.6g})"
        )
    return Order9PiLTeacherAction(
        global_action=tuple(_unit_clip(value) for value in global_values),
        joint_action=tuple(
            tuple(_unit_clip(value) for value in row) for row in joint_values
        ),
        module_ids=module_ids,
        maximum_absolute_unclipped_action=maximum,
    )


def compute_order9_pi_l_behavior_cloning_loss(
    policy: Order9PhaseConditionedActorCritic,
    contexts: Sequence[LowLevelPolicyContext],
    teacher_commands: Sequence[PolicyCommand],
    *,
    previous_actions: torch.Tensor | None = None,
    recurrent_state: torch.Tensor | None = None,
    privileged_disturbance_body: torch.Tensor | None = None,
    decision_returns: Sequence[float] | None = None,
    joint_loss_weight: float = 1.0,
    value_loss_weight: float = 0.5,
) -> Order9PiLBehaviorCloningLoss:
    """Compute one recurrent-step BC loss over global and local-joint actions."""

    if not contexts or len(contexts) != len(teacher_commands):
        raise ValueError("Order9 pi_L BC requires equally sized non-empty batches")
    if decision_returns is not None and len(decision_returns) != len(contexts):
        raise ValueError("Order9 pi_L return batch size mismatch")
    if joint_loss_weight < 0.0 or value_loss_weight < 0.0:
        raise ValueError("Order9 pi_L BC loss weights must be non-negative")
    config = policy.config
    parameter = next(policy.parameters())
    device, dtype = parameter.device, parameter.dtype
    references = [order9_pi_l_reference_command(context) for context in contexts]
    builder = RigidBodyControlModelBuilder()
    control_models = [
        builder.build(
            context.morphology_graph,
            context.physical_model,
            context.runtime_observation,
        )
        for context in contexts
    ]
    encoded = [
        encode_order9_pi_l_teacher_action(
            context=context,
            teacher_command=teacher,
            control_model=control_model,
            config=config,
        )
        for context, teacher, control_model in zip(
            contexts, teacher_commands, control_models
        )
    ]
    actor_features = torch.tensor(
        [
            order3_actor_feature_vector(
                context.runtime_observation,
                control_model,
                target_pose_world=_required_pose(reference),
                target_twist=_six_vector(
                    reference.desired_body_twist, "reference twist"
                ),
                max_modules=config.max_modules,
            )
            for context, control_model, reference in zip(
                contexts, control_models, references
            )
        ],
        dtype=dtype,
        device=device,
    )
    phase_features = torch.tensor(
        [order9_phase_actor_feature_vector(context, config) for context in contexts],
        dtype=dtype,
        device=device,
    )
    batch_size = len(contexts)
    previous = (
        torch.zeros(
            (batch_size, ORDER9_GLOBAL_ACTION_SIZE), dtype=dtype, device=device
        )
        if previous_actions is None
        else previous_actions.to(device=device, dtype=dtype)
    )
    hidden = (
        policy.initial_state(batch_size, device=device, dtype=dtype)
        if recurrent_state is None
        else recurrent_state.to(device=device, dtype=dtype)
    )
    output = policy.step(
        [context.morphology_graph for context in contexts],
        [context.runtime_observation for context in contexts],
        actor_features,
        previous,
        hidden,
        phase_features=phase_features,
        privileged_disturbance_body=privileged_disturbance_body,
        deterministic=True,
    )
    global_target = torch.tensor(
        [item.global_action for item in encoded], dtype=dtype, device=device
    )
    joint_target = torch.zeros_like(output.joint_action_mean)
    joint_coordinate_mask = torch.zeros_like(output.joint_action_mean, dtype=torch.bool)
    for batch_index, item in enumerate(encoded):
        module_index = {
            int(module_id): index
            for index, module_id in enumerate(
                output.graph_encoding.module_ids[batch_index].detach().cpu().tolist()
            )
            if int(module_id) >= 0
        }
        for source_index, module_id in enumerate(item.module_ids):
            target_index = module_index[module_id]
            joint_target[batch_index, target_index] = torch.tensor(
                item.joint_action[source_index], dtype=dtype, device=device
            )
            joint_coordinate_mask[batch_index, target_index] = _joint_coordinate_mask(
                contexts[batch_index], module_id, config, device=device
            )
    global_loss = F.smooth_l1_loss(output.action_mean, global_target)
    joint_loss = _masked_mean(
        F.smooth_l1_loss(output.joint_action_mean, joint_target, reduction="none"),
        joint_coordinate_mask,
    )
    if decision_returns is None:
        value_loss = output.value.sum() * 0.0
    else:
        value_target = torch.tensor(decision_returns, dtype=dtype, device=device)
        value_loss = F.mse_loss(output.value, value_target)
    total = global_loss + joint_loss_weight * joint_loss + value_loss_weight * value_loss
    return Order9PiLBehaviorCloningLoss(
        total=total,
        global_action=global_loss,
        joint_action=joint_loss,
        value=value_loss,
        active_joint_coordinate_count=int(joint_coordinate_mask.sum().item()),
        recurrent_state_out=output.recurrent_state,
        teacher_global_action=global_target,
    )


def compute_order9_pi_l_ppo_loss(
    *,
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    new_values: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    clip_ratio: float = 0.2,
    value_loss_weight: float = 0.5,
    entropy_bonus_weight: float = 0.001,
) -> torch.Tensor:
    """PPO objective whose log probability covers both actor action heads."""

    if value_loss_weight < 0.0 or entropy_bonus_weight < 0.0:
        raise ValueError("Order9 pi_L PPO loss weights must be non-negative")
    actor = clipped_ppo_surrogate_loss(
        new_log_prob=new_log_prob,
        old_log_prob=old_log_prob,
        advantages=advantages,
        clip_ratio=clip_ratio,
    )
    critic = F.mse_loss(new_values, returns)
    return actor + value_loss_weight * critic - entropy_bonus_weight * entropy.mean()


def _joint_coordinate_mask(
    context: LowLevelPolicyContext,
    module_id: int,
    config: Order9LowLevelPolicyConfig,
    *,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(3 * config.max_local_joint_slots, dtype=torch.bool, device=device)
    state = next(
        state
        for state in context.runtime_observation.module_states
        if state.module_id == module_id
    )
    for slot, joint_id in enumerate(_dock_mechanism_joint_ids(context.physical_model)):
        if joint_id in state.joint_positions:
            mask[slot] = True
            mask[config.max_local_joint_slots + slot] = True
            mask[2 * config.max_local_joint_slots + slot] = True
    return mask


def _dock_mechanism_joint_ids(physical_model) -> list[str]:
    return sorted(
        {
            str(port.mechanical_limits["mechanism_joint_id"])
            for port in physical_model.dock_ports
            if port.mechanical_limits.get("mechanism_joint_id")
        }
    )


def _required_pose(command: PolicyCommand):
    if command.desired_body_pose is None:
        raise SchemaValidationError("Order9 pi_L requires a centroidal pose")
    return command.desired_body_pose


def _six_vector(values: Sequence[float] | None, label: str) -> tuple[float, ...]:
    resolved = [0.0] * 6 if values is None else [float(value) for value in values]
    if len(resolved) != 6 or not all(math.isfinite(value) for value in resolved):
        raise SchemaValidationError(f"Order9 pi_L {label} must contain six finite values")
    return tuple(resolved)


def _unit_clip(value: float) -> float:
    return min(max(float(value), -1.0), 1.0)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype)
    denominator = weights.sum()
    if float(denominator.detach().cpu().item()) <= 0.0:
        return values.sum() * 0.0
    return (values * weights).sum() / denominator
