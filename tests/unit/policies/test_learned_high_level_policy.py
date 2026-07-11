from __future__ import annotations

from dataclasses import dataclass

from amsrr.policies.contact_candidate_encoder import ContactCandidateEncoder
from amsrr.policies.contact_wrench_trajectory import P4_2DeterministicGraspCarryPlanner
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.policies.learned_high_level_policy import (
    HighLevelPolicyScores,
    LearnedHighLevelPolicy,
)
from amsrr.schemas.common import ContactMode
from amsrr.schemas.contact_candidates import (
    ContactCandidate,
    ContactCandidateGroupProposal,
    ContactCandidateSet,
)
from amsrr.schemas.interaction_envelope import InteractionEnvelope
from amsrr.schemas.irg import IRGNode, IRGNodeType, InteractionRequirementGraph
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.policies import ControllerStatus, ContactWrenchTrajectory
from amsrr.schemas.runtime import (
    ObjectRuntimeState,
    RuntimeObservation,
    TaskProgressState,
)


@dataclass
class _FixedScoreProvider:
    scores: HighLevelPolicyScores

    def predict(self, encoding) -> HighLevelPolicyScores:
        return self.scores


def test_contact_candidate_encoder_preserves_non_contiguous_source_ids() -> None:
    context = _context("encoder-task")

    output = ContactCandidateEncoder().encode(context.contact_candidate_set)

    assert output.candidate_counts == [2]
    assert output.group_counts == [2]
    assert output.source_ids[0][:2] == [101, 305]
    assert output.candidate_ids[0][:2] == [101, 305]
    assert output.group_ids[0] == ["group-primary", "group-secondary"]
    assert output.group_candidate_ids[0] == [[101], [305]]
    assert output.candidate_valid_mask() == [True, True]
    assert output.group_valid_mask() == [True, True]


def test_learned_pi_h_decodes_ranked_existing_group_with_original_schema_ids() -> None:
    context = _context("learned-task")
    provider = _FixedScoreProvider(
        HighLevelPolicyScores(
            candidate_scores={101: -2.0, 305: 3.0},
            group_scores={"group-primary": -2.0, "group-secondary": 4.0},
            timing_residual_s=0.01,
        )
    )
    policy = LearnedHighLevelPolicy(provider)

    trajectory = policy.plan(context)

    assert ContactWrenchTrajectory.from_json(trajectory.to_json()).to_dict() == trajectory.to_dict()
    assert trajectory.derived_mode_label == "p4_3_learned_pi_h"
    assert {
        assignment.candidate_id
        for knot in trajectory.knots
        for assignment in knot.contact_assignments
    } == {305}
    assert {
        assignment.slot_id
        for knot in trajectory.knots
        for assignment in knot.contact_assignments
    } == {7}
    assert {
        assignment.anchor_id
        for knot in trajectory.knots
        for assignment in knot.contact_assignments
    } == {12}
    assert trajectory.knots[1].t_rel_s == 0.26
    assert context.contact_candidate_set.assignment_feasibility_cache
    assert policy.last_decision is not None
    assert policy.last_decision.used_fallback is False
    assert policy.last_decision.selected_group_id == "group-secondary"
    assert not hasattr(trajectory, "rotor_thrusts_n")
    assert not hasattr(trajectory, "vectoring_joint_targets")


def test_learned_pi_h_unknown_id_output_uses_p4_2_deterministic_fallback() -> None:
    context = _context("fallback-task")
    expected = P4_2DeterministicGraspCarryPlanner().plan(context)
    expected_ids = {
        assignment.candidate_id
        for knot in expected.knots
        for assignment in knot.contact_assignments
    }
    provider = _FixedScoreProvider(
        HighLevelPolicyScores(
            candidate_scores={101: 1.0, 305: 0.0, 999: 100.0},
            group_scores={"group-primary": 1.0, "group-secondary": 0.0},
            timing_residual_s=0.0,
        )
    )
    policy = LearnedHighLevelPolicy(provider)

    trajectory = policy.plan(context)

    output_ids = {
        assignment.candidate_id
        for knot in trajectory.knots
        for assignment in knot.contact_assignments
    }
    assert output_ids == expected_ids == {101}
    assert trajectory.derived_mode_label == "p4_2_deterministic_grasp_carry"
    assert policy.last_decision is not None
    assert policy.last_decision.used_fallback is True
    assert "unknown or invalid IDs" in str(policy.last_decision.fallback_reason)
    assert policy.last_decision.assignment_feasible is True


def test_learned_pi_h_nan_timing_output_uses_fallback() -> None:
    context = _context("nan-task")
    provider = _FixedScoreProvider(
        HighLevelPolicyScores(
            candidate_scores={101: 1.0, 305: 0.0},
            group_scores={"group-primary": 1.0, "group-secondary": 0.0},
            timing_residual_s=float("nan"),
        )
    )
    policy = LearnedHighLevelPolicy(provider)

    trajectory = policy.plan(context)

    assert trajectory.derived_mode_label == "p4_2_deterministic_grasp_carry"
    assert policy.last_decision is not None
    assert policy.last_decision.used_fallback is True
    assert "timing residual must be finite" in str(policy.last_decision.fallback_reason)


def test_learned_pi_h_conflicting_group_output_uses_fallback() -> None:
    context = _context("conflict-task")
    context.contact_candidate_set.group_proposals[1].candidate_ids = [101, 305]
    context.contact_candidate_set.pairwise_conflict_matrix[0][1] = True
    context.contact_candidate_set.pairwise_conflict_matrix[1][0] = True
    provider = _FixedScoreProvider(
        HighLevelPolicyScores(
            candidate_scores={101: 0.0, 305: 1.0},
            group_scores={"group-primary": 0.0, "group-secondary": 100.0},
            timing_residual_s=0.0,
        )
    )
    policy = LearnedHighLevelPolicy(provider)

    trajectory = policy.plan(context)

    output_ids = {
        assignment.candidate_id
        for knot in trajectory.knots
        for assignment in knot.contact_assignments
    }
    assert output_ids == {101}
    assert policy.last_decision is not None
    assert policy.last_decision.used_fallback is True
    assert "unknown or invalid IDs" in str(policy.last_decision.fallback_reason)


def _context(task_id: str) -> HighLevelPolicyContext:
    morphology = MorphologyGraph(
        graph_id=f"morphology-{task_id}",
        modules=[],
        ports=[],
        dock_edges=[],
        robot_anchors=[],
        control_groups=[],
        base_module_id=0,
        is_closed_loop=False,
    )
    candidates = [
        _candidate(101, anchor_id=11, normal=(1.0, 0.0, 0.0)),
        _candidate(305, anchor_id=12, normal=(-1.0, 0.0, 0.0)),
    ]
    candidate_set = ContactCandidateSet(
        set_id=f"candidate-set-{task_id}",
        task_id=task_id,
        morphology_graph_id=morphology.graph_id,
        candidates=candidates,
        candidate_mask=[True, True],
        slot_coverage={7: [101, 305]},
        pairwise_conflict_matrix=[[False, False], [False, False]],
        pairwise_compatibility_score=[[1.0, 0.8], [0.8, 1.0]],
        group_proposals=[
            ContactCandidateGroupProposal(
                group_id="group-primary",
                candidate_ids=[101],
                group_type="grasp_pair",
                group_score=2.0,
            ),
            ContactCandidateGroupProposal(
                group_id="group-secondary",
                candidate_ids=[305],
                group_type="grasp_pair",
                group_score=1.0,
            ),
        ],
        assignment_feasibility_cache={},
        sampler_version="test-sampler-v1",
    )
    irg = InteractionRequirementGraph(
        irg_id=f"irg-{task_id}",
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
                ref_id="slot-7",
                priority=1.0,
                is_hard=True,
                active_phase_id=None,
                feature={
                    "slot_id": 7,
                    "required": True,
                    "min_count_group": 1,
                    "max_count_group": 1,
                },
            ),
        ],
        edges=[],
    )
    envelope = InteractionEnvelope(
        envelope_id=f"envelope-{task_id}",
        task_id=task_id,
        required_contact_count_range=(1, 1),
        required_contact_modes=[ContactMode.GRASP],
        target_region_sets=[],
        wrench_space_requirements=[],
    )
    runtime = RuntimeObservation(
        time_s=0.0,
        morphology_graph=morphology,
        module_states=[],
        object_states=[
            ObjectRuntimeState(
                object_id="box-1",
                pose_world=(0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0),
                twist_world=[0.0] * 6,
            )
        ],
        contact_states=[],
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
        task_progress=TaskProgressState(
            phase_label="approach",
            progress_ratio=0.0,
        ),
    )
    return HighLevelPolicyContext(
        irg=irg,
        interaction_envelope=envelope,
        morphology_graph=morphology,
        contact_candidate_set=candidate_set,
        runtime_observation=runtime,
    )


def _candidate(
    candidate_id: int,
    *,
    anchor_id: int,
    normal: tuple[float, float, float],
) -> ContactCandidate:
    return ContactCandidate(
        candidate_id=candidate_id,
        slot_id=7,
        anchor_id=anchor_id,
        target_entity_id="box-1",
        region_id=f"region-{candidate_id}",
        contact_pose_world=(
            0.5,
            0.01 * float(candidate_id % 10),
            0.3,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        contact_frame_world=(0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0),
        normal_world=normal,
        tangent_basis_world=[0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        contact_mode=ContactMode.GRASP,
        friction=0.6,
        patch_area_m2=0.01,
        candidate_scores={"normal_alignment": 1.0},
        unary_valid=True,
    )
