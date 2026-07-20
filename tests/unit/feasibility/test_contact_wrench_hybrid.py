from __future__ import annotations

from dataclasses import dataclass

from amsrr.feasibility.contact_wrench_hybrid import (
    HybridContactWrenchPhysicsEvaluator,
    LightweightContactWrenchQPEvaluator,
    ShadowCollisionSample,
    ShadowKnotObservation,
)
from amsrr.feasibility.contact_wrench_trajectory import (
    ContactWrenchTrajectoryFeasibilityChecker,
)
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.schemas.common import ContactMode
from amsrr.schemas.contact_candidates import ContactCandidate, ContactCandidateSet
from amsrr.schemas.morphology import MorphologyGraph, RobotAnchor
from amsrr.schemas.policies import (
    CONTACT_WRENCH_CONTRACT_CONTACT_FRAME,
    ContactAssignment,
    ContactWrenchTrajectory,
    InteractionKnot,
)
from amsrr.training.order9_teacher import (
    build_order8_grasp_carry_task_spec,
    compile_high_level_context,
)


@dataclass(frozen=True)
class _FixedShadowBackend:
    observations: tuple[ShadowKnotObservation, ...]
    backend_version: str = "unit-shadow-v1"

    def rollout(self, **_) -> tuple[ShadowKnotObservation, ...]:
        return self.observations


def test_lightweight_qp_finds_payload_witness_without_mutating_proposal() -> None:
    context = _context()
    trajectory = _trajectory(tangential_half_width_n=12.0)
    proposal_hash = trajectory.stable_hash()
    context_hash = context.contact_candidate_set.stable_hash()

    result = LightweightContactWrenchQPEvaluator().evaluate_trajectory(
        context=context,
        trajectory=trajectory,
    )

    assert len(result) == 1
    assert result[0].qp_residual is not None
    assert result[0].qp_residual < 1.0e-4
    assert result[0].wrench_residual is not None
    assert result[0].wrench_residual < 1.0e-3
    assert result[0].margins["contact_qp_solved"] == 1.0
    assert result[0].margins["active_numeric_wrench_requirement_count"] == 1.0
    assert trajectory.stable_hash() == proposal_hash
    assert context.contact_candidate_set.stable_hash() == context_hash


def test_lightweight_qp_rejects_range_without_payload_capacity() -> None:
    result = LightweightContactWrenchQPEvaluator().evaluate_trajectory(
        context=_context(),
        trajectory=_trajectory(tangential_half_width_n=0.1),
    )

    assert result[0].qp_residual == 1.0
    assert result[0].wrench_residual == 1.0
    assert result[0].margins["contact_qp_solved"] == 0.0


def test_hybrid_checker_ignores_only_assigned_contacts_and_accepts_clear_scene() -> None:
    context = _context()
    trajectory = _trajectory(tangential_half_width_n=12.0)
    observation = ShadowKnotObservation(
        controller_qp_residual=2.0e-6,
        contact_wrench_residual=3.0e-5,
        collision_samples=(
            ShadowCollisionSample(
                entity_a="anchor:0",
                entity_b="payload",
                candidate_id=0,
                signed_distance_m=-0.001,
            ),
            ShadowCollisionSample(
                entity_a="anchor:1",
                entity_b="payload",
                candidate_id=1,
                signed_distance_m=-0.001,
            ),
            ShadowCollisionSample(
                entity_a="robot",
                entity_b="obstacle",
                signed_distance_m=0.04,
            ),
        ),
        collision_free_clearance_m=0.05,
        metrics={"rollout_steps": 5.0},
    )
    evaluator = HybridContactWrenchPhysicsEvaluator(
        shadow_backend=_FixedShadowBackend((observation,))
    )

    result = ContactWrenchTrajectoryFeasibilityChecker(
        physics_evaluator=evaluator
    ).check(trajectory, context)

    assert result.feasible is True
    margins = result.knot_results[0].margins
    assert margins["shadow_allowed_contact_count"] == 2.0
    assert margins["shadow_prohibited_pair_sample_count"] == 1.0
    assert margins["shadow_prohibited_collision_count"] == 0.0
    assert margins["required_collision_margin_m"] == 0.03
    assert margins["collision_margin_m"] > 0.0


def test_hybrid_checker_fails_closed_on_main_state_mutation() -> None:
    observation = ShadowKnotObservation(
        controller_qp_residual=0.0,
        contact_wrench_residual=0.0,
        collision_free_clearance_m=0.05,
        main_state_unchanged=False,
    )
    evaluator = HybridContactWrenchPhysicsEvaluator(
        shadow_backend=_FixedShadowBackend((observation,))
    )

    result = ContactWrenchTrajectoryFeasibilityChecker(
        physics_evaluator=evaluator
    ).check(_trajectory(tangential_half_width_n=12.0), _context())

    assert result.feasible is False
    assert result.knot_results[0].margins["shadow_main_state_unchanged"] == 0.0


def test_hybrid_checker_rejects_unassigned_collision_inside_irg_margin() -> None:
    observation = ShadowKnotObservation(
        controller_qp_residual=0.0,
        contact_wrench_residual=0.0,
        collision_samples=(
            ShadowCollisionSample(
                entity_a="robot",
                entity_b="support",
                signed_distance_m=-0.002,
            ),
        ),
        collision_free_clearance_m=0.05,
    )
    evaluator = HybridContactWrenchPhysicsEvaluator(
        shadow_backend=_FixedShadowBackend((observation,))
    )

    result = ContactWrenchTrajectoryFeasibilityChecker(
        physics_evaluator=evaluator
    ).check(_trajectory(tangential_half_width_n=12.0), _context())

    assert result.feasible is False
    margins = result.knot_results[0].margins
    assert margins["shadow_prohibited_collision_count"] == 1.0
    assert margins["collision_margin_m"] < 0.0


def _trajectory(*, tangential_half_width_n: float) -> ContactWrenchTrajectory:
    assignments = [
        ContactAssignment(
            slot_id=0,
            anchor_id=0,
            candidate_id=0,
            contact_mode=ContactMode.GRASP,
            schedule_state="maintain",
            wrench_target=[-5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            wrench_lower=[
                -5.5,
                -tangential_half_width_n,
                -tangential_half_width_n,
                -0.1,
                -0.1,
                -0.1,
            ],
            wrench_upper=[
                -4.5,
                tangential_half_width_n,
                tangential_half_width_n,
                0.1,
                0.1,
                0.1,
            ],
            wrench_frame="contact",
        ),
        ContactAssignment(
            slot_id=0,
            anchor_id=1,
            candidate_id=1,
            contact_mode=ContactMode.GRASP,
            schedule_state="maintain",
            wrench_target=[5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            wrench_lower=[
                4.5,
                -tangential_half_width_n,
                -tangential_half_width_n,
                -0.1,
                -0.1,
                -0.1,
            ],
            wrench_upper=[
                5.5,
                tangential_half_width_n,
                tangential_half_width_n,
                0.1,
                0.1,
                0.1,
            ],
            wrench_frame="contact",
        ),
    ]
    return ContactWrenchTrajectory(
        horizon_s=0.1,
        dt_s=0.1,
        knots=[
            InteractionKnot(
                t_rel_s=0.0,
                contact_assignments=assignments,
                guard_conditions=[
                    {
                        "type": "order9_task_phase",
                        "phase_label": "lift_object",
                    }
                ],
            )
        ],
        contract_version=CONTACT_WRENCH_CONTRACT_CONTACT_FRAME,
    )


def _context() -> HighLevelPolicyContext:
    task = build_order8_grasp_carry_task_spec(
        object_pose_world=(0.5, 0.0, 0.225, 0.0, 0.0, 0.0, 1.0),
        object_size_m=(0.30, 0.40, 0.15),
        object_mass_kg=1.0,
        object_friction=0.6,
        required_transport_distance_m=0.20,
        support_height_m=0.15,
        max_contact_force_n=30.0,
        max_contact_torque_nm=5.0,
        task_id="hybrid-c-h-task",
        object_id="payload",
    )
    morphology = MorphologyGraph(
        graph_id="hybrid-c-h-morphology",
        modules=[],
        ports=[],
        dock_edges=[],
        robot_anchors=[
            RobotAnchor(
                anchor_id=0,
                module_id=0,
                link_id="left",
                local_pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                anchor_type="grasp",
                capability={"max_force_n": 30.0, "max_torque_nm": 5.0},
                associated_contact_slot_ids=[0],
            ),
            RobotAnchor(
                anchor_id=1,
                module_id=1,
                link_id="right",
                local_pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                anchor_type="grasp",
                capability={"max_force_n": 30.0, "max_torque_nm": 5.0},
                associated_contact_slot_ids=[0],
            ),
        ],
        control_groups=[],
        base_module_id=0,
        is_closed_loop=False,
    )
    candidates = [
        ContactCandidate(
            candidate_id=0,
            slot_id=0,
            anchor_id=0,
            target_entity_id="payload",
            region_id="positive-x",
            contact_pose_world=(0.5, 0.0, 0.225, 0.0, 0.0, 0.0, 1.0),
            contact_frame_world=(0.5, 0.0, 0.225, 0.0, 0.0, 0.0, 1.0),
            normal_world=(1.0, 0.0, 0.0),
            tangent_basis_world=[0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            contact_mode=ContactMode.GRASP,
            friction=4.5,
            patch_area_m2=0.01,
            candidate_scores={"material_effective_friction": 4.5},
            unary_valid=True,
        ),
        ContactCandidate(
            candidate_id=1,
            slot_id=0,
            anchor_id=1,
            target_entity_id="payload",
            region_id="negative-x",
            contact_pose_world=(0.5, 0.0, 0.225, 0.0, 0.0, 0.0, 1.0),
            contact_frame_world=(0.5, 0.0, 0.225, 0.0, 0.0, 0.0, 1.0),
            normal_world=(-1.0, 0.0, 0.0),
            tangent_basis_world=[0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            contact_mode=ContactMode.GRASP,
            friction=4.5,
            patch_area_m2=0.01,
            candidate_scores={"material_effective_friction": 4.5},
            unary_valid=True,
        ),
    ]
    candidate_set = ContactCandidateSet(
        set_id="hybrid-c-h-candidates",
        task_id=task.task_id,
        morphology_graph_id=morphology.graph_id,
        candidates=candidates,
        candidate_mask=[True, True],
        slot_coverage={0: [0, 1]},
        pairwise_conflict_matrix=[[False, False], [False, False]],
        pairwise_compatibility_score=[[1.0, 1.0], [1.0, 1.0]],
        group_proposals=[],
        assignment_feasibility_cache={},
        sampler_version="hybrid-c-h-unit-v1",
    )
    return compile_high_level_context(task, morphology, candidate_set)
