from __future__ import annotations

from dataclasses import replace

import pytest

from amsrr.feasibility.contact_wrench_hybrid import ShadowCollisionSample
from amsrr.feasibility.contact_wrench_shadow_metrics import MeasuredCandidateWrench
from amsrr.morphology.random_connected import morphology_structural_hash
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.contact_candidates import ContactCandidateSet
from amsrr.schemas.policies import ContactWrenchTrajectory, InteractionKnot
from amsrr.simulation.order8_natural_contact import (
    build_representative_order8_morphology,
)
from amsrr.simulation.order9_shadow_executor import (
    Order9IsaacControlStepEvidence,
    Order9IsaacShadowExecutor,
)
from amsrr.simulation.order9_shadow_worker import Order9ShadowStateExport
from tests.unit.feasibility.test_contact_wrench_hybrid import (
    _context as contact_context,
    _trajectory as contact_trajectory,
)


_CHECKPOINT = "a" * 64


class _Runtime:
    runtime_version = "unit-copied-isaac-v1"
    pi_l_checkpoint_sha256 = _CHECKPOINT

    def __init__(self, topology_hash: str) -> None:
        self.topology_structural_hash = topology_hash
        self.restore_count = 0
        self.advance_count = 0
        self.reset_count = 0
        self.closed = False

    def restore_copied_state(self, _state) -> None:
        self.restore_count += 1

    def begin_trajectory(self, **_) -> None:
        pass

    def observe(self, **_) -> Order9IsaacControlStepEvidence:
        return Order9IsaacControlStepEvidence(
            controller_qp_residual=0.0,
            measured_candidate_wrenches=_payload_wrenches(),
            collision_samples=(
                ShadowCollisionSample(
                    entity_a="robot",
                    entity_b="support",
                    signed_distance_m=0.05,
                ),
            ),
            collision_free_clearance_m=0.05,
        )

    def advance(self, **_) -> Order9IsaacControlStepEvidence:
        self.advance_count += 1
        return Order9IsaacControlStepEvidence(
            controller_qp_residual=3.0e-4,
            measured_candidate_wrenches=_payload_wrenches(),
            collision_samples=(
                ShadowCollisionSample(
                    entity_a="robot",
                    entity_b="support",
                    signed_distance_m=0.02,
                ),
            ),
            collision_free_clearance_m=0.02,
        )

    def reset_copied_state(self) -> None:
        self.reset_count += 1

    def close(self) -> None:
        self.closed = True


def test_shadow_executor_samples_wrench_at_knots_and_path_collision_minimum() -> None:
    context = _valid_context()
    trajectory = _two_knot_trajectory()
    topology_hash = morphology_structural_hash(context.morphology_graph)
    runtime = _Runtime(topology_hash)
    executor = Order9IsaacShadowExecutor(runtime)
    state = _state(topology_hash)

    executor.synchronize(state)
    observations = executor.execute(
        state_digest=state.state_digest,
        context=context,
        trajectory=trajectory,
    )

    assert len(observations) == 2
    assert observations[0].contact_wrench_residual == pytest.approx(0.0)
    assert observations[0].collision_free_clearance_m == pytest.approx(0.05)
    assert observations[1].contact_wrench_residual == pytest.approx(0.0)
    assert observations[1].controller_qp_residual == pytest.approx(3.0e-4)
    assert observations[1].collision_free_clearance_m == pytest.approx(0.02)
    assert observations[1].collision_samples[0].signed_distance_m == pytest.approx(
        0.02
    )
    assert runtime.advance_count == 5

    executor.reset(state_digest=state.state_digest)
    assert runtime.reset_count == 1


def test_shadow_executor_rejects_wrong_topology_bucket_before_physics() -> None:
    context = _valid_context()
    runtime = _Runtime("b" * 64)
    executor = Order9IsaacShadowExecutor(runtime)

    with pytest.raises(RuntimeError, match="topology bucket mismatch"):
        executor.synchronize(_state(morphology_structural_hash(context.morphology_graph)))

    assert runtime.restore_count == 0


def _payload_wrenches() -> tuple[MeasuredCandidateWrench, ...]:
    return (
        MeasuredCandidateWrench(
            candidate_id=0,
            wrench_contact=(-5.0, 0.0, 5.0, 0.0, 0.0, 0.0),
        ),
        MeasuredCandidateWrench(
            candidate_id=1,
            wrench_contact=(5.0, 0.0, 5.0, 0.0, 0.0, 0.0),
        ),
    )


def _valid_context() -> HighLevelPolicyContext:
    original = contact_context()
    physical_model = build_physical_model_from_config(
        "configs/robot/robot_model.yaml"
    )
    morphology = build_representative_order8_morphology(physical_model)
    candidate_set = ContactCandidateSet.from_dict(
        original.contact_candidate_set.to_dict()
    )
    candidate_set.morphology_graph_id = morphology.graph_id
    candidate_set.validate()
    return HighLevelPolicyContext(
        irg=original.irg,
        interaction_envelope=original.interaction_envelope,
        morphology_graph=morphology,
        contact_candidate_set=candidate_set,
        runtime_observation=original.runtime_observation,
    )


def _two_knot_trajectory() -> ContactWrenchTrajectory:
    original = contact_trajectory(tangential_half_width_n=12.0)
    first = original.knots[0]
    second = InteractionKnot.from_dict(first.to_dict())
    second.t_rel_s = 0.1
    return replace(original, knots=[first, second])


def _state(topology_hash: str) -> Order9ShadowStateExport:
    state = Order9ShadowStateExport(
        state_id="shadow-state-0",
        topology_structural_hash=topology_hash,
        simulation_time_s=1.5,
        simulation_state={"root_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]},
        controller_state={"integral": [0.0] * 6},
        pi_l_state={"hidden": [0.0], "previous_action": [0.0]},
        pi_l_checkpoint_sha256=_CHECKPOINT,
    )
    state.validate()
    return state
