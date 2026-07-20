from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from amsrr.simulation.order9_shadow_runtime import ImmutableMainStateShadowBackend
from amsrr.simulation.order9_shadow_worker import (
    Order9ShadowStateExport,
    Order9ShadowWorkerService,
    PersistentIsaacShadowDriver,
)
from amsrr.feasibility.contact_wrench_hybrid import ShadowKnotObservation
from amsrr.morphology.random_connected import morphology_structural_hash
from tests.unit.simulation.test_order9_shadow_executor import (
    _state as valid_state,
    _valid_context,
)
from tests.unit.feasibility.test_contact_wrench_hybrid import (
    _context,
    _trajectory,
)


@dataclass
class _Exporter:
    checkpoint_sha256: str
    position_x: float = 0.0

    def export_shadow_state(self, _context) -> Order9ShadowStateExport:
        return Order9ShadowStateExport(
            state_id="unit-main-state",
            topology_structural_hash="a" * 64,
            simulation_time_s=0.5,
            simulation_state={"root_state": [self.position_x, 0.0, 0.5]},
            controller_state={"integrator": [0.0] * 6},
            pi_l_state={"recurrent": [0.0] * 4},
            pi_l_checkpoint_sha256=self.checkpoint_sha256,
        )


@dataclass
class _Transport:
    checkpoint_sha256: str
    transport_version: str = "unit-transport-v1"
    calls: list[str] = field(default_factory=list)
    synchronized_digest: str | None = None

    def request(self, operation, payload):
        self.calls.append(operation)
        if operation == "synchronize":
            self.synchronized_digest = payload["state_digest"]
            return {
                "operation": operation,
                "accepted": True,
                "state_digest": payload["state_digest"],
                "pi_l_checkpoint_sha256": self.checkpoint_sha256,
            }
        if operation == "execute":
            return {
                "operation": operation,
                "accepted": True,
                "state_digest": payload["state_digest"],
                "pi_l_checkpoint_sha256": self.checkpoint_sha256,
                "proposal_hash": payload["proposal_hash"],
                "observations": [
                    {
                        "controller_qp_residual": 0.0,
                        "contact_wrench_residual": 0.0,
                        "collision_free_clearance_m": 0.05,
                        "finite_state": True,
                        "metrics": {"executed_steps": 5.0},
                    }
                ],
            }
        if operation == "reset":
            return {
                "operation": operation,
                "accepted": True,
                "state_digest": payload["state_digest"],
            }
        raise AssertionError(operation)


def test_persistent_driver_binds_state_proposal_and_checkpoint() -> None:
    checkpoint = "b" * 64
    exporter = _Exporter(checkpoint)
    transport = _Transport(checkpoint)
    driver = PersistentIsaacShadowDriver(
        state_exporter=exporter,
        transport=transport,
        pi_l_checkpoint_sha256=checkpoint,
        worker_version="unit-isaac-worker-v1",
    )

    observations = ImmutableMainStateShadowBackend(driver).rollout(
        context=_context(),
        trajectory=_trajectory(tangential_half_width_n=12.0),
    )

    assert transport.calls == ["synchronize", "execute", "reset"]
    assert observations[0].finite_state is True
    assert observations[0].main_state_unchanged is True
    assert observations[0].metrics["executed_steps"] == 5.0


def test_persistent_driver_detects_main_state_change() -> None:
    checkpoint = "c" * 64
    exporter = _Exporter(checkpoint)
    transport = _Transport(checkpoint)
    driver = PersistentIsaacShadowDriver(
        state_exporter=exporter,
        transport=transport,
        pi_l_checkpoint_sha256=checkpoint,
        worker_version="unit-isaac-worker-v1",
    )

    original_execute = driver.execute_shadow

    def execute_and_mutate(**kwargs):
        result = original_execute(**kwargs)
        exporter.position_x = 0.1
        return result

    driver.execute_shadow = execute_and_mutate  # type: ignore[method-assign]
    observations = ImmutableMainStateShadowBackend(driver).rollout(
        context=_context(),
        trajectory=_trajectory(tangential_half_width_n=12.0),
    )

    assert observations[0].main_state_unchanged is False


def test_shadow_state_rejects_non_finite_simulator_payload() -> None:
    with pytest.raises(ValueError, match="finite"):
        Order9ShadowStateExport(
            state_id="bad",
            topology_structural_hash="d" * 64,
            simulation_time_s=0.0,
            simulation_state={"bad": float("nan")},
            controller_state={},
            pi_l_state={},
            pi_l_checkpoint_sha256="e" * 64,
        )


class _Executor:
    worker_version = "unit-service-executor-v1"
    pi_l_checkpoint_sha256 = "f" * 64

    def __init__(self) -> None:
        self.calls: list[str] = []

    def synchronize(self, _state) -> None:
        self.calls.append("synchronize")

    def execute(self, **_) -> tuple[ShadowKnotObservation, ...]:
        self.calls.append("execute")
        return (
            ShadowKnotObservation(
                controller_qp_residual=0.0,
                contact_wrench_residual=0.0,
                collision_free_clearance_m=0.05,
            ),
        )

    def reset(self, **_) -> None:
        self.calls.append("reset")

    def close(self) -> None:
        self.calls.append("close")


def test_worker_service_enforces_synchronize_execute_reset_identity_chain() -> None:
    context = _valid_context()
    trajectory = _trajectory(tangential_half_width_n=12.0)
    topology_hash = morphology_structural_hash(context.morphology_graph)
    state = valid_state(topology_hash)
    state.pi_l_checkpoint_sha256 = "f" * 64
    executor = _Executor()
    service = Order9ShadowWorkerService(executor)

    synchronized, stop = service.handle(
        "synchronize",
        {
            "state": state.to_dict(),
            "state_digest": state.state_digest,
            "pi_l_checkpoint_sha256": executor.pi_l_checkpoint_sha256,
        },
    )
    executed, stop_execute = service.handle(
        "execute",
        {
            "state_digest": state.state_digest,
            "pi_l_checkpoint_sha256": executor.pi_l_checkpoint_sha256,
            "proposal_hash": trajectory.stable_hash(),
            "trajectory": trajectory.to_dict(),
            "context": {
                "irg": context.irg.to_dict(),
                "interaction_envelope": context.interaction_envelope.to_dict(),
                "morphology_graph": context.morphology_graph.to_dict(),
                "contact_candidate_set": context.contact_candidate_set.to_dict(),
                "runtime_observation": None,
            },
        },
    )
    reset, stop_reset = service.handle(
        "reset", {"state_digest": state.state_digest}
    )

    assert synchronized["accepted"] is True and stop is False
    assert executed["accepted"] is True and stop_execute is False
    assert len(executed["observations"]) == 1
    assert reset["accepted"] is True and stop_reset is False
    assert executor.calls == ["synchronize", "execute", "reset"]
    service.close()
    assert executor.calls[-1] == "close"


def test_worker_service_rejects_wrong_proposal_hash_without_execution() -> None:
    context = _valid_context()
    trajectory = _trajectory(tangential_half_width_n=12.0)
    topology_hash = morphology_structural_hash(context.morphology_graph)
    state = valid_state(topology_hash)
    state.pi_l_checkpoint_sha256 = "f" * 64
    executor = _Executor()
    service = Order9ShadowWorkerService(executor)
    service.handle(
        "synchronize",
        {
            "state": state.to_dict(),
            "state_digest": state.state_digest,
            "pi_l_checkpoint_sha256": executor.pi_l_checkpoint_sha256,
        },
    )

    result, _ = service.handle(
        "execute",
        {
            "state_digest": state.state_digest,
            "pi_l_checkpoint_sha256": executor.pi_l_checkpoint_sha256,
            "proposal_hash": "0" * 64,
            "trajectory": trajectory.to_dict(),
            "context": {
                "irg": context.irg.to_dict(),
                "interaction_envelope": context.interaction_envelope.to_dict(),
                "morphology_graph": context.morphology_graph.to_dict(),
                "contact_candidate_set": context.contact_candidate_set.to_dict(),
                "runtime_observation": None,
            },
        },
    )

    assert result["accepted"] is False
    assert executor.calls == ["synchronize"]
    service.close()
