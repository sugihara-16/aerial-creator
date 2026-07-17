from __future__ import annotations

from dataclasses import replace
import math

import pytest

from amsrr.policies.deterministic_natural_contact_planner import (
    DeterministicNaturalContactPlanner,
    NaturalContactAnchorSelection,
    NaturalContactPlannerConfig,
    NaturalContactPlannerFeedback,
    ORDER8_FREE_MORPH_ANCHOR_ORIENTATION_WEIGHT,
)
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.schemas.common import ContactMode, SchemaValidationError
from amsrr.schemas.contact_candidates import ContactCandidate, ContactCandidateSet
from amsrr.schemas.morphology import ModuleNode, MorphologyGraph, RobotAnchor
from amsrr.schemas.order8 import Order8NaturalContactPhase
from amsrr.schemas.physical_model import ModuleCapabilityToken


def _capability() -> ModuleCapabilityToken:
    return ModuleCapabilityToken(
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


def _context() -> HighLevelPolicyContext:
    capability = _capability()
    graph = MorphologyGraph(
        graph_id="order8-test-graph",
        modules=[
            ModuleNode(0, "holon", (0, 0, 0, 0, 0, 0, 1), "left", True, capability),
            ModuleNode(1, "holon", (0, 1, 0, 0, 0, 0, 1), "right", False, capability),
        ],
        ports=[],
        dock_edges=[],
        robot_anchors=[
            RobotAnchor(0, 0, "pitch_dock_mech1", (0, 0, 0, 0, 0, 0, 1), "grasp", {}, [0]),
            RobotAnchor(1, 1, "yaw_dock_mech1", (0, 0, 0, 0, 0, 0, 1), "grasp", {}, [1]),
        ],
        control_groups=[],
        base_module_id=0,
        is_closed_loop=False,
    )
    candidates = [
        ContactCandidate(
            candidate_id=index,
            slot_id=index,
            anchor_id=index,
            target_entity_id="object",
            region_id=f"face:{index}",
            contact_pose_world=(0, 0, 0, 0, 0, 0, 1),
            contact_frame_world=(0, 0, 0, 0, 0, 0, 1),
            normal_world=(1.0 if index == 0 else -1.0, 0.0, 0.0),
            tangent_basis_world=[0, 1, 0, 0, 0, 1],
            contact_mode=ContactMode.GRASP,
            friction=0.6,
            patch_area_m2=0.01,
            candidate_scores={},
            unary_valid=True,
        )
        for index in range(2)
    ]
    candidate_set = ContactCandidateSet(
        set_id="order8-test-candidates",
        task_id="order8-test",
        morphology_graph_id=graph.graph_id,
        candidates=candidates,
        candidate_mask=[True, True],
        slot_coverage={0: [0], 1: [1]},
        pairwise_conflict_matrix=[[False, False], [False, False]],
        pairwise_compatibility_score=[[1.0, 1.0], [1.0, 1.0]],
        group_proposals=[],
        assignment_feasibility_cache={},
        sampler_version="test",
    )
    return HighLevelPolicyContext(
        irg=None,  # type: ignore[arg-type]
        interaction_envelope=None,  # type: ignore[arg-type]
        morphology_graph=graph,
        contact_candidate_set=candidate_set,
    )


def _selections() -> list[NaturalContactAnchorSelection]:
    return [
        NaturalContactAnchorSelection(0, 0, 0, "pitch_dock_mech1", (1.0, 0.0, 0.0)),
        NaturalContactAnchorSelection(1, 1, 1, "yaw_dock_mech1", (-1.0, 0.0, 0.0)),
    ]


def _feedback(
    time_s: float,
    *,
    hover: bool = True,
    reachable: bool = True,
    aligned: bool = True,
    contact_command_complete: bool = False,
    lift: bool = False,
    transport: bool = False,
    place: bool = False,
    release_command_complete: bool = False,
    retreat: bool = False,
    settle: bool = False,
) -> NaturalContactPlannerFeedback:
    pose = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    return NaturalContactPlannerFeedback(
        time_s=time_s,
        hover_ready=hover,
        simultaneous_reachability_passed=reachable,
        pregrasp_aligned=aligned,
        contact_command_dwell_complete=contact_command_complete,
        lift_clearance_reached=lift,
        transport_distance_reached=transport,
        intended_place_pose_reached=place,
        release_command_dwell_complete=release_command_complete,
        retreat_clearance_reached=retreat,
        post_release_settle_complete=settle,
        desired_body_pose_by_phase={phase: pose for phase in Order8NaturalContactPhase},
        desired_body_linear_velocity_by_phase={
            phase: (0.2, 0.0, 0.0) for phase in Order8NaturalContactPhase
        },
        desired_anchor_pose_by_id={0: pose, 1: pose},
        desired_object_pose_by_phase={Order8NaturalContactPhase.TRANSPORT: pose},
    )


def test_contact_trajectory_contains_two_assignments_and_no_actuator_command() -> None:
    planner = DeterministicNaturalContactPlanner(_selections())
    planner.observe(_feedback(0.0))
    assert planner.phase == Order8NaturalContactPhase.APPROACH
    planner.observe(_feedback(0.1))
    assert planner.phase == Order8NaturalContactPhase.CONTACT_ACQUISITION
    trajectory = planner.plan(_context())
    assert len(trajectory.knots[0].contact_assignments) == 2
    assert {assignment.schedule_state for assignment in trajectory.knots[0].contact_assignments} == {"attach"}
    assert trajectory.knots[0].contact_assignments[0].wrench_target == [11.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert trajectory.knots[0].centroidal_target.com_vel_world == (0.2, 0.0, 0.0)
    assert trajectory.knots[0].priority_weights[
        "anchor_orientation"
    ] == ORDER8_FREE_MORPH_ANCHOR_ORIENTATION_WEIGHT
    assert "PolicyCommand" not in trajectory.to_json()


def test_contact_acquisition_wrench_is_scaled_before_full_force() -> None:
    planner = DeterministicNaturalContactPlanner(_selections())
    planner.observe(_feedback(0.0))
    planner.observe(_feedback(0.1, contact_command_complete=False))
    feedback = _feedback(0.2, contact_command_complete=False)
    planner.observe(replace(feedback, contact_force_scale=0.25))

    trajectory = planner.plan(_context())

    assert planner.phase == Order8NaturalContactPhase.CONTACT_ACQUISITION
    assert trajectory.knots[0].contact_assignments[0].wrench_target == [
        2.75,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]


def test_contact_acquisition_wrench_can_ramp_independently_per_anchor() -> None:
    planner = DeterministicNaturalContactPlanner(_selections())
    planner.observe(_feedback(0.0))
    planner.observe(_feedback(0.1, contact_command_complete=False))
    feedback = _feedback(0.2, contact_command_complete=False)
    planner.observe(
        replace(
            feedback,
            contact_force_scale=0.0,
            contact_force_scale_by_anchor_id={0: 0.25, 1: 0.0},
        )
    )

    assignments = planner.plan(_context()).knots[0].contact_assignments

    assert assignments[0].wrench_target == [
        2.75,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert assignments[1].wrench_target == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]


def test_per_anchor_force_scale_requires_exact_finite_selection_coverage() -> None:
    planner = DeterministicNaturalContactPlanner(_selections())
    feedback = _feedback(0.0)
    with pytest.raises(SchemaValidationError, match="cover exactly"):
        planner.observe(
            replace(feedback, contact_force_scale_by_anchor_id={0: 0.25})
        )
    with pytest.raises(SchemaValidationError, match=r"finite and in \[0, 1\]"):
        planner.observe(
            replace(
                feedback,
                contact_force_scale_by_anchor_id={0: 0.25, 1: math.inf},
            )
        )


def test_contact_assignment_priority_can_relax_one_anchor_pose_task() -> None:
    planner = DeterministicNaturalContactPlanner(_selections())
    planner.observe(_feedback(0.0))
    feedback = _feedback(0.1)
    planner.observe(
        replace(feedback, anchor_pose_priority_by_id={0: 0.05, 1: 1.0})
    )

    assignments = planner.plan(_context()).knots[0].contact_assignments

    assert [assignment.priority for assignment in assignments] == [0.05, 1.0]


def test_full_gate_sequence_reaches_complete_and_releases_assignments() -> None:
    planner = DeterministicNaturalContactPlanner(_selections())
    planner.observe(_feedback(0.0))
    planner.observe(_feedback(0.1))
    planner.observe(_feedback(0.2, contact_command_complete=True))
    planner.observe(_feedback(0.3, lift=True))
    planner.observe(_feedback(0.4, transport=True))
    planner.observe(_feedback(0.5, place=True))
    planner.observe(_feedback(0.6, release_command_complete=True))
    planner.observe(_feedback(0.7, retreat=True))
    planner.observe(_feedback(0.8, settle=True))
    assert planner.phase == Order8NaturalContactPhase.COMPLETE
    assert planner.plan(_context()).knots[0].contact_assignments == []


def test_simultaneous_reachability_is_required_before_contact_acquisition() -> None:
    planner = DeterministicNaturalContactPlanner(_selections())
    planner.observe(_feedback(0.0))
    planner.observe(_feedback(0.1, reachable=False))
    assert planner.phase == Order8NaturalContactPhase.APPROACH


def test_contact_acquisition_uses_its_longer_bounded_timeout() -> None:
    planner = DeterministicNaturalContactPlanner(
        _selections(),
        config=NaturalContactPlannerConfig(
            phase_timeout_s=0.2,
            contact_acquisition_timeout_s=1.0,
        ),
    )
    planner.observe(_feedback(0.0))
    planner.observe(_feedback(0.1))
    assert planner.phase == Order8NaturalContactPhase.CONTACT_ACQUISITION

    planner.observe(_feedback(0.4))
    assert planner.phase == Order8NaturalContactPhase.CONTACT_ACQUISITION
    planner.observe(_feedback(1.2))
    assert planner.phase == Order8NaturalContactPhase.SAFE_HOLD
    assert planner.failure_reason == "phase_timeout:contact_acquisition"


def test_external_safety_supervisor_can_abort_without_privileged_policy_feedback() -> None:
    planner = DeterministicNaturalContactPlanner(_selections())
    planner.observe(_feedback(0.0))
    planner.request_safe_hold(time_s=0.1, reason="injected_failure")
    assert planner.phase == Order8NaturalContactPhase.SAFE_HOLD
    assert planner.plan(_context()).knots[0].contact_assignments == []
    assert planner.failure_reason == "safety_supervisor:injected_failure"


def test_privileged_contact_evidence_is_not_a_planner_feedback_field() -> None:
    assert "evidence" not in NaturalContactPlannerFeedback.__dataclass_fields__
    assert "raw_contact" not in " ".join(
        NaturalContactPlannerFeedback.__dataclass_fields__
    )


def test_context_rejects_unavailable_selected_candidate() -> None:
    context = _context()
    context.contact_candidate_set.candidate_mask[1] = False
    planner = DeterministicNaturalContactPlanner(_selections())
    planner.observe(_feedback(0.0))
    with pytest.raises(SchemaValidationError, match="unavailable candidate 1"):
        planner.plan(context)


def test_selection_requires_distinct_links_and_unit_normals() -> None:
    duplicate = replace(_selections()[1], dock_link_id="pitch_dock_mech1")
    with pytest.raises(SchemaValidationError, match="distinct dock_link_id"):
        DeterministicNaturalContactPlanner([_selections()[0], duplicate])
    bad_normal = replace(_selections()[1], inward_normal_world=(2.0, 0.0, 0.0))
    with pytest.raises(SchemaValidationError, match="unit vectors"):
        DeterministicNaturalContactPlanner([_selections()[0], bad_normal])
