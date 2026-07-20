from __future__ import annotations

"""Physics-evidence reduction for the isolated Order 9 Isaac shadow worker.

This module owns the time semantics of a counterfactual trajectory.  A
simulator-specific runtime restores the copied state and performs one frozen
``pi_L``/controller step at a time.  The executor samples contact-wrench
evidence at each original knot and conservatively retains the minimum
collision clearance and maximum controller residual along the interval that
led to that knot.
"""

import math
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from amsrr.feasibility.contact_wrench_hybrid import (
    ShadowCollisionSample,
    ShadowKnotObservation,
)
from amsrr.feasibility.contact_wrench_shadow_metrics import (
    MeasuredCandidateWrench,
    evaluate_shadow_contact_wrench_residual,
)
from amsrr.morphology.random_connected import morphology_structural_hash
from amsrr.policies.contact_wrench_trajectory_runtime import (
    ContactWrenchTrajectoryExecutor,
)
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.schemas.policies import ContactWrenchTrajectory, InteractionKnot
from amsrr.simulation.order9_shadow_worker import Order9ShadowStateExport


ORDER9_ISAAC_SHADOW_EXECUTOR_VERSION = "order9_isaac_shadow_executor_v1"


@dataclass(frozen=True)
class Order9IsaacShadowExecutorConfig:
    control_dt_s: float = 0.02
    maximum_horizon_s: float = 2.0
    force_scale_n: float = 30.0
    torque_scale_nm: float = 5.0
    fail_closed_residual: float = 1.0
    maximum_control_steps: int = 200

    def __post_init__(self) -> None:
        for name in (
            "control_dt_s",
            "maximum_horizon_s",
            "force_scale_n",
            "torque_scale_nm",
            "fail_closed_residual",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"Order9 shadow executor {name} must be positive")
        if self.maximum_control_steps < 1:
            raise ValueError("Order9 shadow maximum_control_steps must be positive")
        if self.maximum_horizon_s / self.control_dt_s > self.maximum_control_steps:
            raise ValueError(
                "Order9 shadow maximum_control_steps cannot cover maximum_horizon_s"
            )


@dataclass(frozen=True)
class Order9IsaacControlStepEvidence:
    """One post-physics observation from the copied Isaac runtime.

    ``controller_qp_residual`` is already dimensionless.  Contact wrenches are
    net measured wrenches per selected patch expressed in each candidate's
    schema contact frame; they are not controller targets or estimates.
    """

    controller_qp_residual: float
    measured_candidate_wrenches: tuple[MeasuredCandidateWrench, ...] = ()
    collision_samples: tuple[ShadowCollisionSample, ...] = ()
    collision_free_clearance_m: float = 0.0
    finite_state: bool = True
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("controller_qp_residual", "collision_free_clearance_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"Order9IsaacControlStepEvidence.{name} must be non-negative"
                )
        if any(not math.isfinite(float(value)) for value in self.metrics.values()):
            raise ValueError("Order9 Isaac shadow step metrics must be finite")


class Order9CopiedIsaacRuntime(Protocol):
    """Concrete Isaac scene/controller binding owned only by the worker."""

    @property
    def runtime_version(self) -> str:
        ...

    @property
    def pi_l_checkpoint_sha256(self) -> str:
        ...

    @property
    def topology_structural_hash(self) -> str:
        ...

    def describe(self) -> dict[str, object]:
        ...

    def restore_copied_state(self, state: Order9ShadowStateExport) -> None:
        ...

    def begin_trajectory(
        self,
        *,
        context: HighLevelPolicyContext,
        trajectory: ContactWrenchTrajectory,
    ) -> None:
        ...

    def observe(
        self,
        *,
        context: HighLevelPolicyContext,
        trajectory: ContactWrenchTrajectory,
        active_knot: InteractionKnot,
        elapsed_s: float,
    ) -> Order9IsaacControlStepEvidence:
        ...

    def advance(
        self,
        *,
        context: HighLevelPolicyContext,
        trajectory: ContactWrenchTrajectory,
        active_knot: InteractionKnot,
        elapsed_s: float,
        dt_s: float,
    ) -> Order9IsaacControlStepEvidence:
        ...

    def reset_copied_state(self) -> None:
        ...

    def close(self) -> None:
        ...


class Order9IsaacShadowExecutor:
    """Execute a proposal in copied state and emit one evidence row per knot."""

    def __init__(
        self,
        runtime: Order9CopiedIsaacRuntime,
        *,
        config: Order9IsaacShadowExecutorConfig | None = None,
    ) -> None:
        if not runtime.runtime_version:
            raise ValueError("Order9 copied Isaac runtime version must be non-empty")
        _require_sha256(runtime.pi_l_checkpoint_sha256)
        _require_sha256(runtime.topology_structural_hash)
        self.runtime = runtime
        self.config = config or Order9IsaacShadowExecutorConfig()
        self._state_digest: str | None = None
        self._closed = False

    @property
    def worker_version(self) -> str:
        return (
            f"{ORDER9_ISAAC_SHADOW_EXECUTOR_VERSION}:"
            f"{self.runtime.runtime_version}"
        )

    @property
    def pi_l_checkpoint_sha256(self) -> str:
        return self.runtime.pi_l_checkpoint_sha256

    def describe(self) -> dict[str, object]:
        descriptor = dict(self.runtime.describe())
        descriptor.update(
            {
                "executor_version": ORDER9_ISAAC_SHADOW_EXECUTOR_VERSION,
                "control_dt_s": float(self.config.control_dt_s),
                "maximum_horizon_s": float(self.config.maximum_horizon_s),
            }
        )
        return descriptor

    def synchronize(self, state: Order9ShadowStateExport) -> None:
        self._require_open()
        state.validate()
        if self._state_digest is not None:
            raise RuntimeError("Order9 shadow executor requires reset before synchronize")
        if state.pi_l_checkpoint_sha256 != self.pi_l_checkpoint_sha256:
            raise RuntimeError("Order9 shadow executor pi_L checkpoint mismatch")
        if state.topology_structural_hash != self.runtime.topology_structural_hash:
            raise RuntimeError("Order9 shadow executor topology bucket mismatch")
        before = state.state_digest
        detached = Order9ShadowStateExport.from_dict(state.to_dict())
        self.runtime.restore_copied_state(detached)
        if state.state_digest != before or detached.state_digest != before:
            raise RuntimeError("Order9 copied runtime mutated the exported state")
        self._state_digest = before

    def execute(
        self,
        *,
        state_digest: str,
        context: HighLevelPolicyContext,
        trajectory: ContactWrenchTrajectory,
    ) -> tuple[ShadowKnotObservation, ...]:
        self._require_open()
        if self._state_digest is None or state_digest != self._state_digest:
            raise RuntimeError("Order9 shadow executor has no matching copied state")
        trajectory.validate()
        if trajectory.horizon_s > self.config.maximum_horizon_s + 1.0e-9:
            raise RuntimeError("Order9 shadow proposal exceeds configured horizon")
        if (
            morphology_structural_hash(context.morphology_graph)
            != self.runtime.topology_structural_hash
        ):
            raise RuntimeError("Order9 shadow context uses the wrong topology bucket")
        proposal_hash = trajectory.stable_hash()
        candidate_hash = context.contact_candidate_set.stable_hash()
        self.runtime.begin_trajectory(context=context, trajectory=trajectory)
        sampler = ContactWrenchTrajectoryExecutor(expiry_grace_s=0.0)
        sampler.install(trajectory, plan_start_time_s=0.0)
        elapsed_s = 0.0
        control_step_count = 0
        interval_evidence: list[Order9IsaacControlStepEvidence] = []
        observations: list[ShadowKnotObservation] = []

        for knot_index, knot in enumerate(trajectory.knots):
            target_s = float(knot.t_rel_s)
            while elapsed_s + 1.0e-10 < target_s:
                dt_s = min(self.config.control_dt_s, target_s - elapsed_s)
                sample = sampler.sample(time_s=elapsed_s)
                evidence = self.runtime.advance(
                    context=context,
                    trajectory=trajectory,
                    active_knot=sample.active_knot,
                    elapsed_s=elapsed_s,
                    dt_s=dt_s,
                )
                interval_evidence.append(evidence)
                elapsed_s += dt_s
                control_step_count += 1
                if control_step_count > self.config.maximum_control_steps:
                    raise RuntimeError("Order9 shadow proposal exceeded control-step limit")

            endpoint = self.runtime.observe(
                context=context,
                trajectory=trajectory,
                active_knot=knot,
                elapsed_s=target_s,
            )
            interval_evidence.append(endpoint)
            observations.append(
                self._reduce_knot(
                    context=context,
                    knot=knot,
                    knot_index=knot_index,
                    evidence=interval_evidence,
                    endpoint=endpoint,
                    elapsed_s=target_s,
                )
            )
            interval_evidence = []

        if trajectory.stable_hash() != proposal_hash:
            raise RuntimeError("Order9 copied runtime mutated the proposal")
        if context.contact_candidate_set.stable_hash() != candidate_hash:
            raise RuntimeError("Order9 copied runtime mutated contact candidates")
        return tuple(observations)

    def reset(self, *, state_digest: str) -> None:
        self._require_open()
        if self._state_digest is None or state_digest != self._state_digest:
            raise RuntimeError("Order9 shadow executor reset digest mismatch")
        try:
            self.runtime.reset_copied_state()
        finally:
            self._state_digest = None

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._state_digest is not None:
                self.runtime.reset_copied_state()
        finally:
            self._state_digest = None
            self.runtime.close()
            self._closed = True

    def _reduce_knot(
        self,
        *,
        context: HighLevelPolicyContext,
        knot: InteractionKnot,
        knot_index: int,
        evidence: Sequence[Order9IsaacControlStepEvidence],
        endpoint: Order9IsaacControlStepEvidence,
        elapsed_s: float,
    ) -> ShadowKnotObservation:
        if not evidence:
            raise RuntimeError("Order9 shadow knot has no simulator evidence")
        metric = evaluate_shadow_contact_wrench_residual(
            context,
            knot,
            endpoint.measured_candidate_wrenches,
            force_scale_n=self.config.force_scale_n,
            torque_scale_nm=self.config.torque_scale_nm,
            missing_or_invalid_residual=self.config.fail_closed_residual,
        )
        collision_samples, clearance = _reduce_collision_evidence(evidence)
        finite = all(item.finite_state for item in evidence)
        controller_residual = max(
            float(item.controller_qp_residual) for item in evidence
        )
        metrics = {
            "knot_index": float(knot_index),
            "knot_time_s": float(elapsed_s),
            "interval_evidence_count": float(len(evidence)),
            "endpoint_measured_contact_count": float(
                len(endpoint.measured_candidate_wrenches)
            ),
            **{
                f"wrench.{key}": float(value)
                for key, value in metric.margins.items()
            },
            **{
                f"endpoint.{key}": float(value)
                for key, value in endpoint.metrics.items()
            },
        }
        return ShadowKnotObservation(
            controller_qp_residual=controller_residual,
            contact_wrench_residual=float(metric.residual),
            collision_samples=collision_samples,
            collision_free_clearance_m=clearance,
            finite_state=finite,
            # This worker is process-isolated; the outer coordinator still
            # compares exported main-state digests before accepting evidence.
            main_state_unchanged=True,
            metrics=metrics,
        )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Order9 shadow executor is closed")


def _reduce_collision_evidence(
    rows: Sequence[Order9IsaacControlStepEvidence],
) -> tuple[tuple[ShadowCollisionSample, ...], float]:
    selected: dict[
        tuple[str, str, int | None, int | None, str | None, bool, str | None],
        ShadowCollisionSample,
    ] = {}
    clearance = math.inf
    for row in rows:
        clearance = min(clearance, float(row.collision_free_clearance_m))
        for sample in row.collision_samples:
            key = (
                sample.entity_a,
                sample.entity_b,
                sample.candidate_id,
                sample.anchor_id,
                sample.target_entity_id,
                sample.task_allowed,
                sample.allowance_reason,
            )
            previous = selected.get(key)
            if previous is None or sample.signed_distance_m < previous.signed_distance_m:
                selected[key] = sample
    if math.isinf(clearance):
        clearance = 0.0
    return (
        tuple(selected[key] for key in sorted(selected, key=repr)),
        float(clearance),
    )


def _require_sha256(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("Order9 shadow runtime identity must be a SHA-256 digest")


__all__ = [
    "ORDER9_ISAAC_SHADOW_EXECUTOR_VERSION",
    "Order9CopiedIsaacRuntime",
    "Order9IsaacControlStepEvidence",
    "Order9IsaacShadowExecutor",
    "Order9IsaacShadowExecutorConfig",
]
