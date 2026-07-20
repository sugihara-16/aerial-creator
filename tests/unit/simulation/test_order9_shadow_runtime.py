from __future__ import annotations

from dataclasses import dataclass

import pytest

from amsrr.feasibility.contact_wrench_hybrid import ShadowKnotObservation
from amsrr.simulation.order9_shadow_runtime import (
    ImmutableMainStateShadowBackend,
)
from tests.unit.feasibility.test_contact_wrench_hybrid import (
    _context,
    _trajectory,
)


@dataclass
class _Driver:
    before_digest: str = "main-state-a"
    after_digest: str = "main-state-a"
    mutate_proposal: bool = False
    execute_error: bool = False
    synchronized: int = 0
    reset_count: int = 0
    digest_calls: int = 0
    driver_version: str = "unit-driver-v1"

    def main_state_digest(self, _context) -> str:
        self.digest_calls += 1
        return self.before_digest if self.digest_calls == 1 else self.after_digest

    def synchronize_shadow(self, _context) -> None:
        self.synchronized += 1

    def execute_shadow(self, *, context, trajectory):
        del context
        if self.execute_error:
            raise RuntimeError("simulated worker failure")
        if self.mutate_proposal:
            trajectory.derived_mode_label = "mutated"
        return (
            ShadowKnotObservation(
                controller_qp_residual=0.0,
                contact_wrench_residual=0.0,
                collision_free_clearance_m=0.05,
            ),
        )

    def reset_shadow(self) -> None:
        self.reset_count += 1


def test_shadow_coordinator_copies_executes_resets_and_preserves_main() -> None:
    driver = _Driver()
    backend = ImmutableMainStateShadowBackend(driver)

    observations = backend.rollout(
        context=_context(),
        trajectory=_trajectory(tangential_half_width_n=12.0),
    )

    assert observations[0].main_state_unchanged is True
    assert driver.synchronized == 1
    assert driver.reset_count == 1
    assert driver.digest_calls == 2
    assert "unit-driver-v1" in backend.backend_version


def test_shadow_coordinator_marks_digest_change_without_hiding_evidence() -> None:
    driver = _Driver(after_digest="main-state-b")

    observations = ImmutableMainStateShadowBackend(driver).rollout(
        context=_context(),
        trajectory=_trajectory(tangential_half_width_n=12.0),
    )

    assert observations[0].main_state_unchanged is False


def test_shadow_coordinator_resets_and_raises_on_worker_failure() -> None:
    driver = _Driver(execute_error=True)

    with pytest.raises(RuntimeError, match="shadow driver failed"):
        ImmutableMainStateShadowBackend(driver).rollout(
            context=_context(),
            trajectory=_trajectory(tangential_half_width_n=12.0),
        )

    assert driver.reset_count == 1
    assert driver.digest_calls == 2


def test_shadow_coordinator_rejects_proposal_mutation() -> None:
    driver = _Driver(mutate_proposal=True)

    with pytest.raises(RuntimeError, match="mutated the proposal"):
        ImmutableMainStateShadowBackend(driver).rollout(
            context=_context(),
            trajectory=_trajectory(tangential_half_width_n=12.0),
        )
