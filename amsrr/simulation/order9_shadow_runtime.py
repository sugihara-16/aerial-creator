from __future__ import annotations

"""Immutable-main-state coordinator for the production Order 9 C_H shadow.

The concrete Isaac driver is topology-bucket-specific and may live in a
persistent worker process.  This coordinator owns the invariant shared by all
drivers: the proposal is executed only in copied state, the main environment
digest must be identical before and after, and malformed/partial evidence is
rejected by the outer fail-closed hybrid checker.
"""

from dataclasses import replace
from threading import RLock
from typing import Protocol, Sequence

from amsrr.feasibility.contact_wrench_hybrid import ShadowKnotObservation
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.schemas.policies import ContactWrenchTrajectory


ORDER9_IMMUTABLE_SHADOW_BACKEND_VERSION = "order9_immutable_shadow_backend_v1"


class Order9ShadowRolloutDriver(Protocol):
    """Simulator-specific operations, normally backed by a persistent worker."""

    @property
    def driver_version(self) -> str:
        ...

    def main_state_digest(self, context: HighLevelPolicyContext) -> str:
        """Digest the complete live state that must remain unchanged."""

    def synchronize_shadow(self, context: HighLevelPolicyContext) -> None:
        """Copy live physical/controller/policy state into the shadow worker."""

    def execute_shadow(
        self,
        *,
        context: HighLevelPolicyContext,
        trajectory: ContactWrenchTrajectory,
    ) -> Sequence[ShadowKnotObservation]:
        """Execute one proposal and return exactly one observation per knot."""

    def reset_shadow(self) -> None:
        """Discard transient shadow state without touching the main environment."""


class ImmutableMainStateShadowBackend:
    """Serialize access to a shadow driver and audit all immutable inputs."""

    def __init__(self, driver: Order9ShadowRolloutDriver) -> None:
        if not driver.driver_version:
            raise ValueError("Order9 shadow driver_version must be non-empty")
        self.driver = driver
        self._lock = RLock()

    @property
    def backend_version(self) -> str:
        return (
            f"{ORDER9_IMMUTABLE_SHADOW_BACKEND_VERSION}:"
            f"{self.driver.driver_version}"
        )

    def rollout(
        self,
        *,
        context: HighLevelPolicyContext,
        trajectory: ContactWrenchTrajectory,
    ) -> tuple[ShadowKnotObservation, ...]:
        with self._lock:
            proposal_hash = trajectory.stable_hash()
            candidate_hash = context.contact_candidate_set.stable_hash()
            morphology_hash = context.morphology_graph.stable_hash()
            before = self.driver.main_state_digest(context)
            if not before:
                raise RuntimeError("Order9 main-state digest must be non-empty")
            observations: tuple[ShadowKnotObservation, ...] | None = None
            execution_error: Exception | None = None
            try:
                self.driver.synchronize_shadow(context)
                observations = tuple(
                    self.driver.execute_shadow(
                        context=context,
                        trajectory=trajectory,
                    )
                )
            except Exception as exc:  # propagated to hybrid fail-closed path.
                execution_error = exc
            finally:
                reset_error: Exception | None = None
                try:
                    self.driver.reset_shadow()
                except Exception as exc:
                    reset_error = exc
                after = self.driver.main_state_digest(context)
                if reset_error is not None and execution_error is None:
                    execution_error = reset_error
            if execution_error is not None:
                raise RuntimeError("Order9 shadow driver failed") from execution_error
            if observations is None:
                raise RuntimeError("Order9 shadow driver returned no observations")
            if len(observations) != len(trajectory.knots):
                raise RuntimeError(
                    "Order9 shadow driver must return one observation per knot"
                )
            if any(
                not isinstance(item, ShadowKnotObservation)
                for item in observations
            ):
                raise TypeError(
                    "Order9 shadow driver returned an invalid observation"
                )
            if trajectory.stable_hash() != proposal_hash:
                raise RuntimeError("Order9 shadow driver mutated the proposal")
            if context.contact_candidate_set.stable_hash() != candidate_hash:
                raise RuntimeError(
                    "Order9 shadow driver mutated the contact candidate set"
                )
            if context.morphology_graph.stable_hash() != morphology_hash:
                raise RuntimeError("Order9 shadow driver mutated the morphology")
            main_unchanged = bool(after == before)
            return tuple(
                replace(
                    item,
                    main_state_unchanged=(
                        item.main_state_unchanged and main_unchanged
                    ),
                )
                for item in observations
            )


__all__ = [
    "ORDER9_IMMUTABLE_SHADOW_BACKEND_VERSION",
    "ImmutableMainStateShadowBackend",
    "Order9ShadowRolloutDriver",
]
