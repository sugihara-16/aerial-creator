from __future__ import annotations

from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.policies.contact_candidate_sampler import ContactCandidateSampler
from amsrr.policies.contact_wrench_trajectory import GraspCarryBaselinePlanner, select_feasible_assignments
from amsrr.policies.design_policy_base import DesignPolicyContext, FixedSimpleDesignPolicy
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import ContactMode
from amsrr.schemas.task_spec import TaskSpec


def _context(grasp_carry_dict: dict) -> HighLevelPolicyContext:
    task = TaskSpec.from_dict(grasp_carry_dict)
    builder_result = IRGBuilder().build_with_scene_graph(task)
    irg = builder_result.irg
    envelope = InteractionEnvelopeExtractor().extract(irg)
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    design = FixedSimpleDesignPolicy().design(
        DesignPolicyContext(
            task_spec=task,
            irg=irg,
            interaction_envelope=envelope,
            physical_model=physical_model,
        )
    )
    candidate_set = ContactCandidateSampler().sample(
        task_spec=task,
        irg=irg,
        interaction_envelope=envelope,
        morphology_graph=design.target_morphology,
        geometry_descriptors=builder_result.scene_graph.geometry_descriptors,
    )
    return HighLevelPolicyContext(
        irg=irg,
        interaction_envelope=envelope,
        morphology_graph=design.target_morphology,
        contact_candidate_set=candidate_set,
    )


def test_grasp_carry_baseline_planner_outputs_contact_wrench_trajectory(grasp_carry_dict: dict) -> None:
    context = _context(grasp_carry_dict)

    trajectory = GraspCarryBaselinePlanner().plan(context)

    assert trajectory.derived_mode_label == "grasp_carry_baseline"
    assert trajectory.horizon_s == 2.0
    assert trajectory.dt_s == 0.25
    assert len(trajectory.knots) == 5
    assert [knot.contact_assignments[0].schedule_state for knot in trajectory.knots] == [
        "approach",
        "attach",
        "maintain",
        "maintain",
        "release",
    ]
    assert all(
        assignment.contact_mode == ContactMode.GRASP
        for assignment in trajectory.knots[2].contact_assignments
    )
    assert all(assignment.wrench_target is not None for assignment in trajectory.knots[2].contact_assignments)
    assert trajectory.knots[3].object_targets[0].object_id == "box_01"
    assert trajectory.knots[3].object_targets[0].pose_target_world == (2.0, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0)
    assert context.contact_candidate_set.assignment_feasibility_cache
    assert not hasattr(trajectory, "rotor_thrusts_n")
    assert type(trajectory).from_json(trajectory.to_json()).to_dict() == trajectory.to_dict()


def test_select_feasible_assignments_uses_grasp_pair_group(grasp_carry_dict: dict) -> None:
    context = _context(grasp_carry_dict)

    assignments = select_feasible_assignments(
        context.contact_candidate_set,
        slot_min_counts={0: 2},
        slot_max_counts={0: 4},
    )

    assert len(assignments) == 2
    assert {assignment.slot_id for assignment in assignments} == {0}
    assert len({assignment.anchor_id for assignment in assignments}) == 2
    assert all(assignment.schedule_state == "maintain" for assignment in assignments)
    assert context.contact_candidate_set.assignment_feasibility_cache
