from __future__ import annotations

from dataclasses import dataclass

from amsrr.feasibility.contact_wrench_trajectory import (
    TRAJECTORY_COLLISION_NOT_EVALUATED_CODE,
    TRAJECTORY_QP_NOT_EVALUATED_CODE,
    TRAJECTORY_WRENCH_CONE_FAIL_CODE,
    TRAJECTORY_WRENCH_NOT_EVALUATED_CODE,
    ContactWrenchTrajectoryCheckerConfig,
    ContactWrenchTrajectoryFeasibilityChecker,
    KnotPhysicsEvaluation,
)
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.policies.high_level_runtime import HighLevelTrajectoryRuntime
from amsrr.schemas.common import ContactMode
from amsrr.schemas.contact_candidates import ContactCandidate, ContactCandidateSet
from amsrr.schemas.feasibility import TrajectoryFeasibilityResult
from amsrr.schemas.interaction_envelope import InteractionEnvelope
from amsrr.schemas.irg import IRGNode, IRGNodeType, InteractionRequirementGraph
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.policies import (
    CONTACT_WRENCH_CONTRACT_CONTACT_FRAME,
    CONTACT_WRENCH_CONTRACT_IMPLICIT_WORLD,
    ContactAssignment,
    ContactWrenchTrajectory,
    InteractionKnot,
)


@dataclass(frozen=True)
class _PassingPhysicsEvaluator:
    def evaluate(self, **_) -> KnotPhysicsEvaluation:
        return KnotPhysicsEvaluation(
            qp_residual=0.0,
            wrench_residual=0.0,
            min_collision_margin_m=0.02,
            margins={"actuator_margin": 0.5},
            evaluator_version="unit-test-physics-v1",
        )


@dataclass
class _SequenceProposalPolicy:
    proposals: list[ContactWrenchTrajectory]
    calls: int = 0

    def propose(self, _) -> ContactWrenchTrajectory:
        proposal = self.proposals[min(self.calls, len(self.proposals) - 1)]
        self.calls += 1
        return ContactWrenchTrajectory.from_dict(proposal.to_dict())


@dataclass(frozen=True)
class _FixedFallback:
    trajectory: ContactWrenchTrajectory
    fallback_version: str = "unit-test-fallback-v1"

    def fallback(self, _) -> ContactWrenchTrajectory:
        return ContactWrenchTrajectory.from_dict(self.trajectory.to_dict())


def test_v2_production_checker_accepts_without_mutating_proposal() -> None:
    context = _context()
    trajectory = _trajectory()
    trajectory_hash = trajectory.stable_hash()
    cache_hash = context.contact_candidate_set.stable_hash()
    checker = ContactWrenchTrajectoryFeasibilityChecker(
        physics_evaluator=_PassingPhysicsEvaluator()
    )

    result = checker.check(trajectory, context)

    assert result.feasible is True
    assert result.hard_violations == []
    assert result.knot_results[0].qp_evaluated is True
    assert result.knot_results[0].collision_evaluated is True
    assert result.knot_results[0].wrench_evaluated is True
    assert result.metadata["proposal_mutated"] is False
    assert trajectory.stable_hash() == trajectory_hash
    assert context.contact_candidate_set.stable_hash() == cache_hash
    assert TrajectoryFeasibilityResult.from_json(result.to_json()).to_dict() == result.to_dict()


def test_production_checker_fails_closed_when_backend_checks_are_absent() -> None:
    result = ContactWrenchTrajectoryFeasibilityChecker().check(
        _trajectory(),
        _context(),
    )

    codes = {violation.code for violation in result.hard_violations}
    assert result.feasible is False
    assert TRAJECTORY_QP_NOT_EVALUATED_CODE in codes
    assert TRAJECTORY_COLLISION_NOT_EVALUATED_CODE in codes
    assert TRAJECTORY_WRENCH_NOT_EVALUATED_CODE in codes


def test_warmup_proxy_is_explicit_and_reports_unevaluated_checks() -> None:
    checker = ContactWrenchTrajectoryFeasibilityChecker(
        config=ContactWrenchTrajectoryCheckerConfig.warmup_proxy()
    )

    result = checker.check(_trajectory(), _context())

    assert result.feasible is True
    assert result.metadata["evaluation_mode"] == "warmup_proxy"
    assert {
        TRAJECTORY_QP_NOT_EVALUATED_CODE,
        TRAJECTORY_COLLISION_NOT_EVALUATED_CODE,
        TRAJECTORY_WRENCH_NOT_EVALUATED_CODE,
    }.issubset({warning.code for warning in result.warnings})


def test_checker_rejects_target_outside_friction_cone() -> None:
    trajectory = _trajectory()
    assignment = trajectory.knots[0].contact_assignments[0]
    assignment.wrench_target = [0.0, 10.0, 0.0, 0.0, 0.0, 0.0]
    assignment.wrench_lower = [-1.0, 9.0, -1.0, -1.0, -1.0, -1.0]
    assignment.wrench_upper = [1.0, 11.0, 1.0, 1.0, 1.0, 1.0]
    checker = ContactWrenchTrajectoryFeasibilityChecker(
        physics_evaluator=_PassingPhysicsEvaluator()
    )

    result = checker.check(trajectory, _context())

    assert result.feasible is False
    assert TRAJECTORY_WRENCH_CONE_FAIL_CODE in {
        violation.code for violation in result.hard_violations
    }


def test_legacy_json_defaults_to_explicit_world_v1_contract() -> None:
    data = _trajectory().to_dict()
    data.pop("contract_version")
    data["knots"][0]["contact_assignments"][0].pop("wrench_frame")

    trajectory = ContactWrenchTrajectory.from_dict(data)

    assert trajectory.contract_version == CONTACT_WRENCH_CONTRACT_IMPLICIT_WORLD
    assert trajectory.knots[0].contact_assignments[0].wrench_frame == "world"


def test_runtime_rejects_without_projection_then_accepts_second_proposal() -> None:
    invalid = _trajectory()
    assignment = invalid.knots[0].contact_assignments[0]
    assignment.wrench_target = [0.0, 10.0, 0.0, 0.0, 0.0, 0.0]
    assignment.wrench_lower = [-1.0, 9.0, -1.0, -1.0, -1.0, -1.0]
    assignment.wrench_upper = [1.0, 11.0, 1.0, 1.0, 1.0, 1.0]
    proposal_policy = _SequenceProposalPolicy([invalid, _trajectory()])
    runtime = HighLevelTrajectoryRuntime(
        proposal_policy=proposal_policy,
        checker=ContactWrenchTrajectoryFeasibilityChecker(
            physics_evaluator=_PassingPhysicsEvaluator()
        ),
        fallback=_FixedFallback(_trajectory()),
        max_proposal_attempts=2,
    )

    decision = runtime.plan_and_install(_context(), plan_start_time_s=0.0)

    assert decision.used_fallback is False
    assert decision.accepted_proposal_attempt == 1
    assert len(decision.rejected_proposals) == 1
    assert proposal_policy.calls == 2
    assert runtime.executor.trajectory is not None
    assert runtime.executor.trajectory.stable_hash() == decision.trajectory.stable_hash()


def test_runtime_uses_separately_checked_fallback_after_retries() -> None:
    invalid = _trajectory()
    assignment = invalid.knots[0].contact_assignments[0]
    assignment.wrench_target = [0.0, 10.0, 0.0, 0.0, 0.0, 0.0]
    assignment.wrench_lower = [-1.0, 9.0, -1.0, -1.0, -1.0, -1.0]
    assignment.wrench_upper = [1.0, 11.0, 1.0, 1.0, 1.0, 1.0]
    runtime = HighLevelTrajectoryRuntime(
        proposal_policy=_SequenceProposalPolicy([invalid]),
        checker=ContactWrenchTrajectoryFeasibilityChecker(
            physics_evaluator=_PassingPhysicsEvaluator()
        ),
        fallback=_FixedFallback(_trajectory()),
        max_proposal_attempts=2,
    )

    decision = runtime.plan_and_install(_context(), plan_start_time_s=0.0)

    assert decision.used_fallback is True
    assert decision.fallback_version == "unit-test-fallback-v1"
    assert len(decision.rejected_proposals) == 2


def _trajectory() -> ContactWrenchTrajectory:
    assignment = ContactAssignment(
        slot_id=0,
        anchor_id=0,
        candidate_id=0,
        contact_mode=ContactMode.GRASP,
        schedule_state="maintain",
        wrench_target=[-5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        wrench_lower=[-8.0, -0.5, -0.5, -0.1, -0.1, -0.1],
        wrench_upper=[-2.0, 0.5, 0.5, 0.1, 0.1, 0.1],
        wrench_frame="contact",
    )
    return ContactWrenchTrajectory(
        horizon_s=0.1,
        dt_s=0.1,
        knots=[InteractionKnot(t_rel_s=0.0, contact_assignments=[assignment])],
        contract_version=CONTACT_WRENCH_CONTRACT_CONTACT_FRAME,
    )


def _context() -> HighLevelPolicyContext:
    task_id = "checker-task"
    morphology = MorphologyGraph(
        graph_id="checker-morphology",
        modules=[],
        ports=[],
        dock_edges=[],
        robot_anchors=[],
        control_groups=[],
        base_module_id=0,
        is_closed_loop=False,
    )
    candidate = ContactCandidate(
        candidate_id=0,
        slot_id=0,
        anchor_id=0,
        target_entity_id="box",
        region_id="side",
        contact_pose_world=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        contact_frame_world=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        normal_world=(1.0, 0.0, 0.0),
        tangent_basis_world=[0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        contact_mode=ContactMode.GRASP,
        friction=0.6,
        patch_area_m2=0.01,
        candidate_scores={},
        unary_valid=True,
    )
    candidate_set = ContactCandidateSet(
        set_id="checker-candidates",
        task_id=task_id,
        morphology_graph_id=morphology.graph_id,
        candidates=[candidate],
        candidate_mask=[True],
        slot_coverage={0: [0]},
        pairwise_conflict_matrix=[[False]],
        pairwise_compatibility_score=[[1.0]],
        group_proposals=[],
        assignment_feasibility_cache={},
        sampler_version="unit-test-v1",
    )
    irg = InteractionRequirementGraph(
        irg_id="checker-irg",
        task_id=task_id,
        nodes=[
            IRGNode(
                node_id=0,
                node_type=IRGNodeType.TASK,
                ref_id=task_id,
                priority=1.0,
                is_hard=True,
                active_phase_id=None,
            ),
            IRGNode(
                node_id=1,
                node_type=IRGNodeType.CONTACT_SLOT,
                ref_id="slot-0",
                priority=1.0,
                is_hard=True,
                active_phase_id=None,
                feature={
                    "slot_id": 0,
                    "required": True,
                    "min_count_group": 1,
                    "max_count_group": 1,
                },
            )
        ],
        edges=[],
    )
    envelope = InteractionEnvelope(
        envelope_id="checker-envelope",
        task_id=task_id,
        required_contact_count_range=(1, 1),
        required_contact_modes=[ContactMode.GRASP],
        target_region_sets=[],
        wrench_space_requirements=[],
    )
    return HighLevelPolicyContext(
        irg=irg,
        interaction_envelope=envelope,
        morphology_graph=morphology,
        contact_candidate_set=candidate_set,
    )
