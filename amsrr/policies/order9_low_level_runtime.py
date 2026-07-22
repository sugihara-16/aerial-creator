from __future__ import annotations

"""Strict deployment wrapper for a phase-conditioned Order 9 ``pi_L``.

The actor remains a ``PolicyCommand`` producer.  Controller/QP and actuator
ownership stay downstream, and any unsafe runtime state uses the existing
deterministic centroidal fallback without manufacturing an actor transition.
"""

from pathlib import Path
from typing import Sequence

import torch

from amsrr.policies.low_level_policy_base import (
    BaselineLowLevelPolicy,
    BaselineLowLevelPolicyConfig,
    LowLevelPolicyContext,
)
from amsrr.policies.morphology_conditioned_low_level_policy import (
    MorphologyConditionedLowLevelPolicy,
    Order3PolicyDiagnostics,
    Order3PolicyInference,
)
from amsrr.policies.order9_low_level_policy import (
    ORDER9_GLOBAL_ACTION_NAMES,
    Order9PhaseConditionedActorCritic,
)
from amsrr.policies.order9_policy_command import (
    decode_order9_centroidal_pose_action,
    order9_joint_reference,
    order9_pi_l_reference_command,
)
from amsrr.controllers.rigid_body_model import RigidBodyControlModel
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.order9 import (
    Order9PolicyCheckpointMetadata,
    Order9PolicyFamily,
)
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import (
    POLICY_COMMAND_CONTRACT_CENTROIDAL,
    PolicyCommand,
)
from amsrr.schemas.runtime import RuntimeObservation
from amsrr.training.order9_checkpoints import load_order9_policy_checkpoint


ORDER9_PI_L_RUNTIME_VERSION = "order9_complete_policy_command_pi_l_runtime_v2"
ORDER9_PI_L_ACTOR_OBSERVATION_CONTRACT = (
    "task_phase_morphology_centroidal_no_raw_contact_v1"
)
ORDER9_PI_L_CRITIC_OBSERVATION_CONTRACT = (
    "actor_plus_privileged_disturbance_v1"
)
ORDER9_PI_L_ACTION_CONTRACT = (
    "bounded_complete_centroidal_and_absolute_local_joint_command_v2"
)


class Order9CompletePolicyCommandRuntime(MorphologyConditionedLowLevelPolicy):
    """Order 9 learned-command path with a substitution-only fallback."""

    def _learned_reference_command(
        self,
        context: LowLevelPolicyContext,
        deterministic_fallback: PolicyCommand,
    ) -> PolicyCommand:
        del deterministic_fallback
        return order9_pi_l_reference_command(context)

    def _global_action_names(self) -> Sequence[str]:
        return ORDER9_GLOBAL_ACTION_NAMES

    def _decode_command(
        self,
        reference: PolicyCommand,
        observation: RuntimeObservation,
        control_model: RigidBodyControlModel,
        action: list[float],
        step,
    ) -> PolicyCommand:
        if reference.desired_body_pose is None:
            raise SchemaValidationError("Order9 pi_L reference pose is missing")
        pose = decode_order9_centroidal_pose_action(
            reference.desired_body_pose,
            action[:6],
            self.config,
        )
        reference_twist = list(reference.desired_body_twist or [0.0] * 6)
        twist_limits = [
            *([self.config.linear_twist_correction_limit_mps] * 3),
            *([self.config.angular_twist_correction_limit_radps] * 3),
        ]
        desired_twist = [
            float(reference_twist[index]) + action[6 + index] * twist_limits[index]
            for index in range(6)
        ]
        wrench_scales = [
            *(
                [
                    control_model.total_mass_kg
                    * 9.81
                    * self.config.residual_force_weight_fraction
                ]
                * 3
            ),
            *(
                [
                    len(observation.morphology_graph.modules)
                    * self.config.residual_torque_per_module_nm
                ]
                * 3
            ),
        ]
        residual_wrench = [
            action[12 + index] * wrench_scales[index] for index in range(6)
        ]
        joint_positions, joint_velocities, joint_torque_bias = (
            self._decode_order9_joint_targets(reference, observation, step)
        )
        return PolicyCommand(
            desired_body_pose=pose,
            desired_body_twist=desired_twist,
            residual_wrench_body=residual_wrench,
            priority_weights=dict(reference.priority_weights),
            control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
            joint_position_targets=joint_positions,
            joint_velocity_targets=joint_velocities,
            joint_torque_bias=joint_torque_bias,
        )

    def _decode_order9_joint_targets(
        self,
        reference: PolicyCommand,
        observation: RuntimeObservation,
        step,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        joint_ids = sorted(
            {
                str(port.mechanical_limits["mechanism_joint_id"])
                for port in self.physical_model.dock_ports
                if port.mechanical_limits.get("mechanism_joint_id")
            }
        )
        if len(joint_ids) > self.config.max_local_joint_slots:
            raise SchemaValidationError("Order9 joint decoder has too few local slots")
        state_by_id = {
            state.module_id: state for state in observation.module_states
        }
        module_ids = step.graph_encoding.module_ids[0].detach().cpu().tolist()
        raw = step.joint_action[0].detach().cpu()
        joint_by_id = {
            joint.joint_id: joint for joint in self.physical_model.joints
        }
        q_targets: dict[str, float] = {}
        qdot_targets: dict[str, float] = {}
        torque_bias: dict[str, float] = {}
        for node_index, module_id_value in enumerate(module_ids):
            module_id = int(module_id_value)
            if module_id < 0:
                continue
            state = state_by_id[module_id]
            for slot, joint_id in enumerate(joint_ids):
                if joint_id not in state.joint_positions:
                    continue
                global_id = f"module_{module_id}:{joint_id}"
                q_reference, qdot_reference = order9_joint_reference(
                    reference,
                    global_joint_id=global_id,
                    local_joint_id=joint_id,
                    current_position_rad=float(state.joint_positions[joint_id]),
                )
                q_targets[global_id] = q_reference + (
                    float(raw[node_index, slot].item())
                    * self.config.joint_position_delta_limit_rad
                )
                qdot_targets[global_id] = qdot_reference + (
                    float(
                        raw[
                            node_index,
                            self.config.max_local_joint_slots + slot,
                        ].item()
                    )
                    * self.config.joint_velocity_limit_rad_s
                )
                effort_limit = float(joint_by_id[joint_id].effort_limit or 0.0)
                torque_bias[global_id] = (
                    float(
                        raw[
                            node_index,
                            2 * self.config.max_local_joint_slots + slot,
                        ].item()
                    )
                    * effort_limit
                    * self.config.joint_torque_fraction
                )
        return q_targets, qdot_targets, torque_bias


class Order9LowLevelRuntimePolicy:
    """Load and run one immutable Order 9 ``pi_L`` checkpoint."""

    def __init__(
        self,
        *,
        runtime_policy: MorphologyConditionedLowLevelPolicy,
        checkpoint_metadata: Order9PolicyCheckpointMetadata,
        checkpoint_sha256: str,
    ) -> None:
        if not isinstance(
            runtime_policy.model, Order9PhaseConditionedActorCritic
        ):
            raise TypeError("Order9 pi_L runtime requires the phase-conditioned model")
        self.runtime_policy = runtime_policy
        self.checkpoint_metadata = checkpoint_metadata
        self.checkpoint_sha256 = checkpoint_sha256

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        physical_model: PhysicalModel,
        expected_sha256: str | None = None,
        expected_schedule_hash: str | None = None,
        deterministic: bool = True,
        device: torch.device | str = "cpu",
        baseline_policy: BaselineLowLevelPolicy | None = None,
    ) -> "Order9LowLevelRuntimePolicy":
        loaded = load_order9_policy_checkpoint(
            checkpoint_path,
            device=device,
            expected_sha256=expected_sha256,
            expected_family=Order9PolicyFamily.PI_L,
            expected_schedule_hash=expected_schedule_hash,
        )
        if loaded.metadata.physical_model_hash != physical_model.stable_hash():
            raise SchemaValidationError(
                "Order9 pi_L checkpoint PhysicalModel hash does not match runtime"
            )
        _validate_checkpoint_contracts(loaded.metadata)
        model = loaded.model
        if not isinstance(model, Order9PhaseConditionedActorCritic):
            raise SchemaValidationError("Order9 pi_L checkpoint constructed wrong model")
        resolved_baseline = baseline_policy or BaselineLowLevelPolicy(
            BaselineLowLevelPolicyConfig(
                control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
            )
        )
        if (
            resolved_baseline.config.control_contract_version
            != POLICY_COMMAND_CONTRACT_CENTROIDAL
        ):
            raise SchemaValidationError(
                "Order9 pi_L fallback must use the centroidal-v2 command contract"
            )
        runtime = Order9CompletePolicyCommandRuntime(
            model=model,
            physical_model=physical_model,
            config=model.config,
            deterministic=deterministic,
            baseline_policy=resolved_baseline,
            # Order 9 intentionally generalizes to any grammar-valid 2--8
            # module topology; the deterministic runtime checks identities and
            # module bounds on every call instead of using an Order-3 allowlist.
            allowed_structural_hashes=None,
            device=device,
        )
        runtime.checkpoint_sha256 = loaded.sha256
        return cls(
            runtime_policy=runtime,
            checkpoint_metadata=loaded.metadata,
            checkpoint_sha256=loaded.sha256,
        )

    @property
    def deterministic(self) -> bool:
        return self.runtime_policy.deterministic

    @property
    def last_diagnostics(self) -> Order3PolicyDiagnostics:
        return self.runtime_policy.last_diagnostics

    def reset(self) -> None:
        self.runtime_policy.reset()

    def export_runtime_state(self) -> dict[str, object]:
        return self.runtime_policy.export_runtime_state()

    def restore_runtime_state(self, payload: dict[str, object]) -> None:
        self.runtime_policy.restore_runtime_state(payload)

    def command(self, context: LowLevelPolicyContext) -> PolicyCommand:
        return self.runtime_policy.command(context)

    def command_with_trace(
        self,
        context: LowLevelPolicyContext,
        *,
        privileged_disturbance_body: Sequence[float] | None = None,
    ) -> Order3PolicyInference:
        return self.runtime_policy.command_with_trace(
            context,
            privileged_disturbance_body=privileged_disturbance_body,
        )

    def bootstrap_value(
        self,
        context: LowLevelPolicyContext,
        *,
        privileged_disturbance_body: Sequence[float] | None = None,
    ) -> float:
        return self.runtime_policy.bootstrap_value(
            context,
            privileged_disturbance_body=privileged_disturbance_body,
        )


def _validate_checkpoint_contracts(
    metadata: Order9PolicyCheckpointMetadata,
) -> None:
    expected = {
        "actor_observation_contract": ORDER9_PI_L_ACTOR_OBSERVATION_CONTRACT,
        "critic_observation_contract": ORDER9_PI_L_CRITIC_OBSERVATION_CONTRACT,
        "action_contract": ORDER9_PI_L_ACTION_CONTRACT,
    }
    mismatches = [
        name
        for name, value in expected.items()
        if str(getattr(metadata, name)) != value
    ]
    if mismatches:
        raise SchemaValidationError(
            "Order9 pi_L checkpoint runtime contract mismatch: "
            + ",".join(sorted(mismatches))
        )


__all__ = [
    "ORDER9_PI_L_ACTION_CONTRACT",
    "ORDER9_PI_L_ACTOR_OBSERVATION_CONTRACT",
    "ORDER9_PI_L_CRITIC_OBSERVATION_CONTRACT",
    "ORDER9_PI_L_RUNTIME_VERSION",
    "Order9CompletePolicyCommandRuntime",
    "Order9LowLevelRuntimePolicy",
]
