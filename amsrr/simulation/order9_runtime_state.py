from __future__ import annotations

"""Atomic main-state export used by the isolated Order 9 shadow worker."""

import math
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Mapping, Protocol

from amsrr.morphology.random_connected import morphology_structural_hash
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.schemas.common import SchemaValidationError
from amsrr.simulation.order9_shadow_worker import Order9ShadowStateExport


ORDER9_LIVE_MAIN_STATE_EXPORTER_VERSION = "order9_live_main_state_exporter_v1"


@dataclass(frozen=True)
class Order9SimulationSnapshot:
    state_id: str
    topology_structural_hash: str
    simulation_time_s: float
    simulation_state: dict[str, Any]
    execution_state: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.state_id:
            raise ValueError("Order9 simulation snapshot state_id must be non-empty")
        _require_sha256(self.topology_structural_hash, "topology_structural_hash")
        if (
            not math.isfinite(float(self.simulation_time_s))
            or self.simulation_time_s < 0.0
        ):
            raise ValueError("Order9 simulation snapshot time must be non-negative")


class Order9SimulationSnapshotSource(Protocol):
    """Environment-owned atomic simulation/executor snapshot boundary."""

    @property
    def snapshot_source_version(self) -> str:
        ...

    def capture_order9_simulation_snapshot(
        self,
        context: HighLevelPolicyContext,
    ) -> Order9SimulationSnapshot:
        ...


class Order9ControllerStateSource(Protocol):
    def export_runtime_state(self) -> dict[str, object]:
        ...

    def restore_runtime_state(self, payload: dict[str, object]) -> None:
        ...


class Order9PolicyStateSource(Protocol):
    checkpoint_sha256: str

    def export_runtime_state(self) -> dict[str, object]:
        ...

    def restore_runtime_state(self, payload: dict[str, object]) -> None:
        ...


class Order9LiveMainStateExporter:
    """Capture simulator, executor, QPID, and recurrent pi_L state together."""

    def __init__(
        self,
        *,
        simulation_source: Order9SimulationSnapshotSource,
        controller: Order9ControllerStateSource,
        pi_l_policy: Order9PolicyStateSource,
        checkpoint_sha256: str,
        state_lock: RLock | None = None,
    ) -> None:
        _require_sha256(checkpoint_sha256, "checkpoint_sha256")
        if pi_l_policy.checkpoint_sha256 != checkpoint_sha256:
            raise ValueError(
                "Order9 main-state exporter policy/checkpoint identity mismatch"
            )
        if not simulation_source.snapshot_source_version:
            raise ValueError("Order9 simulation snapshot source version is empty")
        self.simulation_source = simulation_source
        self.controller = controller
        self.pi_l_policy = pi_l_policy
        self.checkpoint_sha256 = checkpoint_sha256
        self.state_lock = state_lock or RLock()

    def export_shadow_state(
        self,
        context: HighLevelPolicyContext,
    ) -> Order9ShadowStateExport:
        with self.state_lock:
            snapshot = self.simulation_source.capture_order9_simulation_snapshot(
                context
            )
            expected_topology = morphology_structural_hash(context.morphology_graph)
            if snapshot.topology_structural_hash != expected_topology:
                raise SchemaValidationError(
                    "Order9 main snapshot topology differs from policy context"
                )
            if self.pi_l_policy.checkpoint_sha256 != self.checkpoint_sha256:
                raise SchemaValidationError(
                    "Order9 live pi_L checkpoint changed after exporter construction"
                )
            controller_state = {
                "qpid": self.controller.export_runtime_state(),
                "trajectory_execution": dict(snapshot.execution_state),
            }
            state = Order9ShadowStateExport(
                state_id=snapshot.state_id,
                topology_structural_hash=snapshot.topology_structural_hash,
                simulation_time_s=float(snapshot.simulation_time_s),
                simulation_state=dict(snapshot.simulation_state),
                controller_state=controller_state,
                pi_l_state=self.pi_l_policy.export_runtime_state(),
                pi_l_checkpoint_sha256=self.checkpoint_sha256,
                metadata={
                    **snapshot.metadata,
                    "main_state_exporter_version": (
                        ORDER9_LIVE_MAIN_STATE_EXPORTER_VERSION
                    ),
                    "simulation_snapshot_source_version": (
                        self.simulation_source.snapshot_source_version
                    ),
                    "atomic_state_lock_held": True,
                },
            )
            state.validate()
            return state


def restore_order9_controller_and_policy_state(
    state: Order9ShadowStateExport,
    *,
    controller: Order9ControllerStateSource,
    pi_l_policy: Order9PolicyStateSource,
) -> dict[str, Any]:
    """Validate both payloads before restoring copied controller/policy state."""

    if pi_l_policy.checkpoint_sha256 != state.pi_l_checkpoint_sha256:
        raise SchemaValidationError("Order9 copied pi_L checkpoint mismatch")
    raw_qpid = state.controller_state.get("qpid")
    raw_execution = state.controller_state.get("trajectory_execution")
    if not isinstance(raw_qpid, Mapping) or not isinstance(raw_execution, Mapping):
        raise SchemaValidationError(
            "Order9 copied controller state requires qpid and trajectory_execution"
        )
    if not isinstance(state.pi_l_state, Mapping):
        raise SchemaValidationError("Order9 copied pi_L state must be a mapping")
    # Each restore method validates fully before assigning its own fields.  The
    # simulator runtime calls this only after its copied physics state has been
    # staged, and discards the whole shadow copy on any exception.
    controller.restore_runtime_state(dict(raw_qpid))
    pi_l_policy.restore_runtime_state(dict(state.pi_l_state))
    return dict(raw_execution)


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"Order9 {name} must be a SHA-256 digest")


__all__ = [
    "ORDER9_LIVE_MAIN_STATE_EXPORTER_VERSION",
    "Order9ControllerStateSource",
    "Order9LiveMainStateExporter",
    "Order9PolicyStateSource",
    "Order9SimulationSnapshot",
    "Order9SimulationSnapshotSource",
    "restore_order9_controller_and_policy_state",
]
