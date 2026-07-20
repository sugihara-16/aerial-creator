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
    Order9PhaseConditionedActorCritic,
)
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
from amsrr.training.order9_checkpoints import load_order9_policy_checkpoint


ORDER9_PI_L_RUNTIME_VERSION = "order9_phase_conditioned_pi_l_runtime_v1"
ORDER9_PI_L_ACTOR_OBSERVATION_CONTRACT = (
    "task_phase_morphology_centroidal_no_raw_contact_v1"
)
ORDER9_PI_L_CRITIC_OBSERVATION_CONTRACT = (
    "actor_plus_privileged_disturbance_v1"
)
ORDER9_PI_L_ACTION_CONTRACT = (
    "bounded_global_and_masked_local_joint_residual_v1"
)


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
        runtime = MorphologyConditionedLowLevelPolicy(
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
    "Order9LowLevelRuntimePolicy",
]
