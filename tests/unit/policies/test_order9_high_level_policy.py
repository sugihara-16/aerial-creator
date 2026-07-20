from __future__ import annotations

import torch

from amsrr.schemas.contact_candidates import AssignmentFeasibilityResult
from amsrr.policies.order9_high_level_policy import (
    GUARD_TYPES,
    ORDER9_FULL_PI_H_VERSION,
    Order9AutoregressiveHighLevelPolicy,
    Order9HighLevelPolicyConfig,
)
from amsrr.schemas.common import ContactMode
from amsrr.schemas.contact_candidates import ContactCandidate, ContactCandidateSet
from amsrr.schemas.morphology import ModuleNode, MorphologyGraph, RobotAnchor
from amsrr.schemas.physical_model import ModuleCapabilityToken
from amsrr.schemas.policies import (
    CONTACT_WRENCH_CONTRACT_CONTACT_FRAME,
    ControllerStatus,
)
from amsrr.schemas.datasets import (
    DatasetSplit,
    HighLevelTransitionKind,
    InteractionTrajectoryRecord,
    StageDecisionMasks,
    TrajectoryProvenance,
    TrajectorySourceKind,
)
from amsrr.schemas.feasibility import (
    TrajectoryFeasibilityResult,
    TrajectoryKnotFeasibilityResult,
    Violation,
    ViolationSeverity,
)
from amsrr.schemas.runtime import (
    ModuleRuntimeState,
    ObjectRuntimeState,
    RuntimeObservation,
    TaskProgressState,
)
from amsrr.training.order9_teacher import (
    build_order8_grasp_carry_task_spec,
    compile_high_level_context,
)
from amsrr.training.order9_pi_h_learning import (
    compute_order9_pi_h_behavior_cloning_loss,
)
from amsrr.training.order9_curriculum import Order9PPOOptimizationConfig
from amsrr.training.order9_ppo import (
    order9_pi_h_behavior_trace,
    update_order9_pi_h_ppo,
)
from amsrr.training.order9_dataset import _validate_episode_boundaries
from amsrr.training.order9_pi_h_rollout import (
    Order9PiHEpisodeCollector,
    order9_pi_h_executed_record,
    order9_pi_h_rejection_record,
    sample_order9_pi_h_with_hard_checker,
)


def test_full_pi_h_emits_complete_v2_trajectory_and_ppo_terms() -> None:
    context = _context()
    config = Order9HighLevelPolicyConfig(d_model=48, num_knots=5)
    policy = Order9AutoregressiveHighLevelPolicy(config)

    output = policy.forward_contexts([context])
    action = policy.sample_action(output, deterministic=True)
    # Make the schema coverage deterministic independently of random init.
    action.assignment_active[:] = True
    action.schedule_index[:] = 2  # maintain
    action.guard_active[:] = False
    action.guard_active[:, :, GUARD_TYPES.index("controller_feasible")] = True
    trajectory = policy.decode_action(context, action, batch_index=0)
    evaluation = policy.evaluate_action(output, action)

    assert trajectory.contract_version == CONTACT_WRENCH_CONTRACT_CONTACT_FRAME
    assert trajectory.derived_mode_label == ORDER9_FULL_PI_H_VERSION
    assert len(trajectory.knots) == 5
    assert trajectory.knots[0].t_rel_s == 0.0
    assert trajectory.knots[-1].t_rel_s == config.horizon_s
    assert all(
        right.t_rel_s > left.t_rel_s
        for left, right in zip(trajectory.knots, trajectory.knots[1:])
    )
    assert all(knot.contact_assignments for knot in trajectory.knots)
    assignment = trajectory.knots[0].contact_assignments[0]
    assert assignment.wrench_frame == "contact"
    assert assignment.wrench_lower is not None
    assert assignment.wrench_target is not None
    assert assignment.wrench_upper is not None
    assert all(
        lower <= target <= upper
        for lower, target, upper in zip(
            assignment.wrench_lower,
            assignment.wrench_target,
            assignment.wrench_upper,
        )
    )
    assert trajectory.knots[0].centroidal_target is not None
    assert trajectory.knots[0].posture_target is not None
    assert trajectory.knots[0].object_targets[0].object_id == "order8_object"
    assert trajectory.knots[0].priority_weights.keys() == {
        "contact",
        "centroidal",
        "posture",
        "object",
        "safety",
    }
    assert trajectory.knots[0].guard_conditions[0]["type"] == "controller_feasible"
    assert all(
        any(
            guard.get("type") == "order9_task_phase"
            and guard.get("phase_label") == "apply_grasp_wrench"
            for guard in knot.guard_conditions
        )
        for knot in trajectory.knots
    )
    assert evaluation.log_prob.shape == (1,)
    assert evaluation.entropy.shape == (1,)
    assert evaluation.value.shape == (1,)
    assert torch.isfinite(evaluation.log_prob).all()
    assert torch.isfinite(evaluation.entropy).all()
    assert torch.isfinite(evaluation.value).all()


def test_full_pi_h_is_a_pure_proposal_and_supports_empty_contact_set() -> None:
    context = _context(empty_candidates=True)
    policy = Order9AutoregressiveHighLevelPolicy(
        Order9HighLevelPolicyConfig(d_model=48, num_knots=3)
    )

    trajectory = policy.propose(context)

    assert len(trajectory.knots) == 3
    assert all(not knot.contact_assignments for knot in trajectory.knots)
    assert trajectory.contract_version == CONTACT_WRENCH_CONTRACT_CONTACT_FRAME


def test_full_pi_h_behavior_cloning_covers_trajectory_fields_and_backpropagates() -> None:
    context = _context()
    policy = Order9AutoregressiveHighLevelPolicy(
        Order9HighLevelPolicyConfig(d_model=48, num_knots=5)
    )
    with torch.no_grad():
        output = policy.forward_contexts([context])
        action = policy.sample_action(output, deterministic=True)
        action.assignment_active[:] = True
        action.schedule_index[:] = 2
        teacher = policy.decode_action(context, action, batch_index=0)

    losses = compute_order9_pi_h_behavior_cloning_loss(
        policy,
        [context],
        [teacher],
        decision_returns=[1.5],
    )
    losses.total.backward()

    assert losses.selected_assignment_count == 5
    assert losses.active_wrench_count == 5
    assert torch.isfinite(losses.total)
    assert all(
        torch.isfinite(component)
        for component in (
            losses.assignment,
            losses.schedule,
            losses.wrench,
            losses.timing,
            losses.centroidal,
            losses.posture,
            losses.object_target,
            losses.priority,
            losses.guard,
            losses.value,
        )
    )
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in policy.parameters()
    )


def test_full_pi_h_stochastic_trace_replays_through_ppo_update() -> None:
    context = _context()
    policy = Order9AutoregressiveHighLevelPolicy(
        Order9HighLevelPolicyConfig(d_model=32, num_knots=3)
    )
    checkpoint_sha = "a" * 64
    output = policy.forward_contexts([context])
    action = policy.sample_action(output, deterministic=False)
    evaluation = policy.evaluate_action(output, action)
    trajectory = policy.decode_action(context, action, batch_index=0)
    selected = sorted(
        {
            assignment.candidate_id
            for knot in trajectory.knots
            for assignment in knot.contact_assignments
        }
    )
    record = InteractionTrajectoryRecord(
        record_id="order9-pi-h-ppo-0",
        episode_id="order9-pi-h-ppo-episode",
        task_id=context.irg.task_id,
        split=DatasetSplit.TRAIN,
        decision_index=0,
        decision_time_s=context.runtime_observation.time_s,
        irg=context.irg,
        interaction_envelope=context.interaction_envelope,
        morphology_graph=context.morphology_graph,
        contact_candidate_set=context.contact_candidate_set,
        runtime_observation=context.runtime_observation,
        trajectory=trajectory,
        selected_candidate_ids=selected,
        assignment_feasibility_results=[],
        decision_return=1.0,
        decision_reward=1.0,
        stage_masks=StageDecisionMasks(high_level_decision_mask=True),
        trajectory_provenance=TrajectoryProvenance(
            source_kind=TrajectorySourceKind.LEARNED_POLICY,
            source_version=ORDER9_FULL_PI_H_VERSION,
            policy_checkpoint_sha256=checkpoint_sha,
        ),
        terminal=True,
        behavior_trace=order9_pi_h_behavior_trace(
            action,
            evaluation,
            batch_index=0,
            candidate_count=len(context.contact_candidate_set.candidates),
            object_count=len(context.runtime_observation.object_states),
            checkpoint_sha256=checkpoint_sha,
        ),
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-4)

    result = update_order9_pi_h_ppo(
        policy,
        [record],
        optimizer=optimizer,
        config=Order9PPOOptimizationConfig(
            rollout_steps_per_environment=1,
            epochs_per_update=1,
            minibatch_size=1,
        ),
        behavior_checkpoint_sha256=checkpoint_sha,
        seed=9,
    )

    assert result.policy_family == "pi_h"
    assert result.sample_count == 1
    assert result.optimizer_step_count == 1


def test_pi_h_hard_gate_records_rejection_as_independent_ppo_boundary() -> None:
    context = _context()
    policy = Order9AutoregressiveHighLevelPolicy(
        Order9HighLevelPolicyConfig(d_model=32, num_knots=3)
    )
    checkpoint_sha = "b" * 64
    fallback = _StaticFallback(policy.propose(context))
    decision = sample_order9_pi_h_with_hard_checker(
        policy,
        context,
        checker=_SequencedChecker([False, True]),
        fallback=fallback,
        checkpoint_sha256=checkpoint_sha,
        max_proposal_attempts=2,
    )

    assert len(decision.samples) == 2
    assert len(decision.rejected_samples) == 1
    assert decision.accepted_sample is decision.samples[1]
    assert not decision.used_fallback
    rejection = order9_pi_h_rejection_record(
        decision.samples[0],
        record_id="order9-pi-h-rejected-0",
        episode_id="order9-pi-h-hard-gate-episode",
        split=DatasetSplit.TRAIN,
        decision_index=0,
        context=context,
        rejection_penalty=1.0,
    )
    executed = order9_pi_h_executed_record(
        decision.samples[1],
        record_id="order9-pi-h-executed-1",
        episode_id="order9-pi-h-hard-gate-episode",
        split=DatasetSplit.TRAIN,
        decision_index=1,
        context=context,
        decision_reward=2.0,
        decision_return=2.0,
        terminal=True,
    )

    assert rejection.transition_kind == HighLevelTransitionKind.CHECKER_REJECTION
    assert rejection.decision_reward == -1.0
    assert rejection.terminal
    assert rejection.trajectory_provenance.metadata["executed_in_environment"] is False
    assert executed.transition_kind == HighLevelTransitionKind.EXECUTED_TRAJECTORY
    assert executed.decision_reward == 2.0
    failures: list[str] = []
    _validate_episode_boundaries([rejection, executed], failures)
    assert failures == []

    optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-4)
    update = update_order9_pi_h_ppo(
        policy,
        [rejection, executed],
        optimizer=optimizer,
        config=Order9PPOOptimizationConfig(
            rollout_steps_per_environment=1,
            epochs_per_update=1,
            minibatch_size=2,
            hard_checker_rejection_penalty=1.0,
        ),
        behavior_checkpoint_sha256=checkpoint_sha,
        seed=9,
    )

    assert update.sample_count == 2


def test_pi_h_fallback_is_executed_but_never_becomes_actor_transition() -> None:
    context = _context()
    policy = Order9AutoregressiveHighLevelPolicy(
        Order9HighLevelPolicyConfig(d_model=32, num_knots=3)
    )
    fallback_trajectory = policy.propose(context)
    decision = sample_order9_pi_h_with_hard_checker(
        policy,
        context,
        checker=_SequencedChecker([False, False, True]),
        fallback=_StaticFallback(fallback_trajectory),
        checkpoint_sha256="c" * 64,
        max_proposal_attempts=2,
    )

    assert decision.used_fallback
    assert decision.accepted_sample is None
    assert len(decision.samples) == 2
    assert decision.execution_trajectory is fallback_trajectory
    assert decision.fallback_version == "test_static_fallback_v1"
    assert all(not sample.accepted for sample in decision.samples)


def test_pi_h_episode_collector_keeps_rejections_out_of_executed_gae_segment() -> None:
    context = _context()
    policy = Order9AutoregressiveHighLevelPolicy(
        Order9HighLevelPolicyConfig(d_model=32, num_knots=3)
    )
    checkpoint_sha = "d" * 64
    fallback = _StaticFallback(policy.propose(context))
    collector = Order9PiHEpisodeCollector(
        physical_episode_id="pi-h-physical-episode",
        split=DatasetSplit.TRAIN,
        rejection_penalty=1.0,
        discount_gamma=0.5,
    )
    first = sample_order9_pi_h_with_hard_checker(
        policy,
        context,
        checker=_SequencedChecker([True]),
        fallback=fallback,
        checkpoint_sha256=checkpoint_sha,
    )
    collector.record_decision(first, context=context)
    assert collector.add_environment_reward(1.0)
    second = sample_order9_pi_h_with_hard_checker(
        policy,
        context,
        checker=_SequencedChecker([False, True]),
        fallback=fallback,
        checkpoint_sha256=checkpoint_sha,
    )
    collector.record_decision(second, context=context)
    assert collector.add_environment_reward(2.0)
    result = collector.finalize(terminal=True, terminal_reward=3.0)

    executed = [
        record
        for record in result.records
        if record.transition_kind == HighLevelTransitionKind.EXECUTED_TRAJECTORY
    ]
    rejected = [
        record
        for record in result.records
        if record.transition_kind == HighLevelTransitionKind.CHECKER_REJECTION
    ]
    assert result.learned_execution_count == 2
    assert result.rejected_proposal_count == 1
    assert len(executed) == 2 and len(rejected) == 1
    assert executed[0].episode_id == executed[1].episode_id
    assert rejected[0].episode_id != executed[0].episode_id
    assert executed[0].decision_return == 3.5
    assert executed[1].decision_return == 5.0
    assert rejected[0].decision_return == -1.0
    assert not executed[0].terminal and executed[1].terminal


def test_pi_h_episode_collector_ignores_fallback_and_terminal_rewards() -> None:
    context = _context()
    policy = Order9AutoregressiveHighLevelPolicy(
        Order9HighLevelPolicyConfig(d_model=32, num_knots=3)
    )
    fallback = _StaticFallback(policy.propose(context))
    collector = Order9PiHEpisodeCollector(
        physical_episode_id="pi-h-fallback-episode",
        split=DatasetSplit.TRAIN,
        rejection_penalty=1.0,
    )
    accepted = sample_order9_pi_h_with_hard_checker(
        policy,
        context,
        checker=_SequencedChecker([True]),
        fallback=fallback,
        checkpoint_sha256="e" * 64,
    )
    collector.record_decision(accepted, context=context)
    collector.add_environment_reward(1.0)
    fallback_decision = sample_order9_pi_h_with_hard_checker(
        policy,
        context,
        checker=_SequencedChecker([False, False, True]),
        fallback=fallback,
        checkpoint_sha256="e" * 64,
        max_proposal_attempts=2,
    )
    collector.record_decision(fallback_decision, context=context)
    assert collector.add_environment_reward(10.0) is False
    result = collector.finalize(terminal=True, terminal_reward=20.0)

    executed = [
        record
        for record in result.records
        if record.transition_kind == HighLevelTransitionKind.EXECUTED_TRAJECTORY
    ]
    assert len(executed) == 1
    assert executed[0].decision_reward == 1.0
    assert executed[0].decision_return == 1.0
    assert executed[0].terminal
    assert result.fallback_execution_count == 1
    assert result.ignored_fallback_reward == 30.0


class _StaticFallback:
    fallback_version = "test_static_fallback_v1"

    def __init__(self, trajectory) -> None:
        self.trajectory = trajectory

    def fallback(self, _context):
        return self.trajectory


class _SequencedChecker:
    def __init__(self, outcomes: list[bool]) -> None:
        self.outcomes = list(outcomes)

    def check(self, trajectory, _context) -> TrajectoryFeasibilityResult:
        if not self.outcomes:
            raise AssertionError("unexpected C_H check")
        feasible = self.outcomes.pop(0)
        code = "E_TEST_HARD_REJECTION"
        knot_results = []
        for knot_index, knot in enumerate(trajectory.knots):
            candidate_ids = sorted(
                {assignment.candidate_id for assignment in knot.contact_assignments}
            )
            assignment_result = AssignmentFeasibilityResult(
                assignment_key=f"test-knot-{knot_index}",
                candidate_ids=candidate_ids,
                feasible=feasible,
                violation_codes=[] if feasible else [code],
                qp_residual=0.0,
                wrench_residual=0.0,
                min_collision_margin_m=0.01,
            )
            knot_results.append(
                TrajectoryKnotFeasibilityResult(
                    knot_index=knot_index,
                    t_rel_s=knot.t_rel_s,
                    assignment_result=assignment_result,
                    qp_evaluated=True,
                    collision_evaluated=True,
                    wrench_evaluated=True,
                    violation_codes=[] if feasible else [code],
                )
            )
        hard = []
        if not feasible:
            hard.append(
                Violation(
                    code=code,
                    severity=ViolationSeverity.HARD,
                    message="unit-test hard rejection",
                )
            )
        result = TrajectoryFeasibilityResult(
            feasible=feasible,
            hard_violations=hard,
            warnings=[],
            knot_results=knot_results,
            margins={},
            checker_version="test_sequenced_checker_v1",
            contract_version=trajectory.contract_version,
            metadata={"proposal_mutated": False},
        )
        result.validate()
        return result


def _context(*, empty_candidates: bool = False):
    task = build_order8_grasp_carry_task_spec(
        object_pose_world=(0.5, 0.0, 0.225, 0.0, 0.0, 0.0, 1.0),
        object_size_m=(0.30, 0.40, 0.15),
        object_mass_kg=1.0,
        object_friction=0.6,
        required_transport_distance_m=0.20,
        support_height_m=0.15,
        max_contact_force_n=30.0,
        max_contact_torque_nm=5.0,
    )
    capability = ModuleCapabilityToken(
        module_type="holon",
        aggregate_mass_norm=1.0,
        aggregate_inertia_features=[1.0] * 6,
        rotor_count=4,
        port_count=4,
        thrust_min_features=[0.0] * 4,
        thrust_max_features=[10.0] * 4,
        thrust_to_weight_ratio_est=2.0,
        dock_port_type_counts=[2, 2, 0],
        has_vectoring=True,
        has_dock_mechanism=True,
    )
    modules = [
        ModuleNode(
            module_id=module_id,
            module_type="holon",
            pose_in_design_frame=(
                0.25 * module_id,
                0.0,
                0.5,
                0.0,
                0.0,
                0.0,
                1.0,
            ),
            role_id="base_grasp" if module_id == 0 else "grasp_arm",
            is_base=module_id == 0,
            capability_token=capability,
        )
        for module_id in range(2)
    ]
    anchor = RobotAnchor(
        anchor_id=0,
        module_id=0,
        link_id="Dock_mechanism",
        local_pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        anchor_type="grasp",
        capability={"max_force_n": 30.0, "max_torque_nm": 5.0},
        associated_contact_slot_ids=[0],
    )
    morphology = MorphologyGraph(
        graph_id="order9-full-pi-h-test-morphology",
        modules=modules,
        ports=[],
        dock_edges=[],
        robot_anchors=[anchor],
        control_groups=[],
        base_module_id=0,
        is_closed_loop=False,
    )
    candidates = [] if empty_candidates else [_candidate()]
    candidate_set = ContactCandidateSet(
        set_id="order9-full-pi-h-test-candidates",
        task_id=task.task_id,
        morphology_graph_id=morphology.graph_id,
        candidates=candidates,
        candidate_mask=[True] * len(candidates),
        slot_coverage={} if empty_candidates else {0: [17]},
        pairwise_conflict_matrix=[] if empty_candidates else [[False]],
        pairwise_compatibility_score=[] if empty_candidates else [[1.0]],
        group_proposals=[],
        assignment_feasibility_cache={},
        sampler_version="order9-full-pi-h-test-v1",
    )
    observation = RuntimeObservation(
        time_s=0.25,
        morphology_graph=morphology,
        module_states=[
            ModuleRuntimeState(
                module_id=module_id,
                pose_world=(
                    0.25 * module_id,
                    0.0,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ),
                twist_world=[0.0] * 6,
            )
            for module_id in range(2)
        ],
        object_states=[
            ObjectRuntimeState(
                object_id="order8_object",
                pose_world=(0.5, 0.0, 0.225, 0.0, 0.0, 0.0, 1.0),
                twist_world=[0.0] * 6,
            )
        ],
        contact_states=[],
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
        task_progress=TaskProgressState(
            phase_label="apply_grasp_wrench",
            progress_ratio=0.4,
        ),
    )
    return compile_high_level_context(
        task,
        morphology,
        candidate_set,
        runtime_observation=observation,
    )


def _candidate() -> ContactCandidate:
    return ContactCandidate(
        candidate_id=17,
        slot_id=0,
        anchor_id=0,
        target_entity_id="order8_object",
        region_id="positive-x",
        contact_pose_world=(0.5, 0.0, 0.225, 0.0, 0.0, 0.0, 1.0),
        contact_frame_world=(0.5, 0.0, 0.225, 0.0, 0.0, 0.0, 1.0),
        normal_world=(1.0, 0.0, 0.0),
        tangent_basis_world=[0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        contact_mode=ContactMode.GRASP,
        friction=4.5,
        patch_area_m2=0.01,
        candidate_scores={},
        unary_valid=True,
    )
