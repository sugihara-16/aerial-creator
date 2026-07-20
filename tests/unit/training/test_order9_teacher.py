from __future__ import annotations

from amsrr.feasibility.contact_wrench_trajectory import (
    ContactWrenchTrajectoryCheckerConfig,
    ContactWrenchTrajectoryFeasibilityChecker,
)
from amsrr.feasibility.contact_wrench_hybrid import (
    LightweightContactWrenchQPEvaluator,
)
from amsrr.schemas.common import ContactMode
from amsrr.schemas.contact_candidates import ContactCandidate, ContactCandidateSet
from amsrr.schemas.datasets import (
    DatasetSplit,
    InteractionTrajectoryRecord,
    TrajectorySourceKind,
)
from amsrr.schemas.morphology import MorphologyGraph, RobotAnchor
from amsrr.schemas.policies import (
    CONTACT_WRENCH_CONTRACT_CONTACT_FRAME,
    ContactAssignment,
    ContactWrenchTrajectory,
    ControllerStatus,
    InteractionKnot,
)
from amsrr.schemas.runtime import RuntimeObservation, TaskProgressState
from amsrr.training.order9_teacher import (
    ORDER9_NATURAL_CONTACT_FALLBACK_VERSION,
    ORDER9_NATURAL_CONTACT_TEACHER_VERSION,
    Order9NaturalContactFallback,
    Order9NaturalContactTeacher,
    build_order8_grasp_carry_task_spec,
    compile_high_level_context,
    cuboid_inertia_tensor6,
    teacher_interaction_record,
    upgrade_teacher_trajectory_to_v2,
)
from amsrr.training.order9_teacher_windows import (
    ORDER9_TEACHER_WINDOW_VERSION,
    Order9TeacherWindowConfig,
    compose_order9_teacher_windows,
)


def test_order8_task_uses_normal_taskspec_to_irg_envelope_path() -> None:
    task = _task()
    context = compile_high_level_context(task, _morphology(), _candidate_set())

    assert context.irg.task_id == task.task_id
    assert context.interaction_envelope.task_id == task.task_id
    assert context.interaction_envelope.required_contact_count_range == (2, 4)
    assert context.interaction_envelope.required_contact_modes == [
        ContactMode.GRASP,
        ContactMode.SUPPORT,
    ]
    assert task.scene.objects[0].inertia_kgm2 == cuboid_inertia_tensor6(
        1.0,
        (0.30, 0.40, 0.15),
    )


def test_teacher_upgrade_preserves_world_targets_and_adds_safe_v2_bounds() -> None:
    context = compile_high_level_context(_task(), _morphology(), _candidate_set())
    legacy = _legacy_trajectory()

    converted = upgrade_teacher_trajectory_to_v2(legacy, context)

    assert converted.contract_version == CONTACT_WRENCH_CONTRACT_CONTACT_FRAME
    assignments = converted.knots[0].contact_assignments
    assert [assignment.wrench_frame for assignment in assignments] == [
        "contact",
        "contact",
    ]
    assert assignments[0].wrench_target == [-5.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert assignments[1].wrench_target == [5.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert all(assignment.wrench_lower is not None for assignment in assignments)
    assert all(assignment.wrench_upper is not None for assignment in assignments)

    result = ContactWrenchTrajectoryFeasibilityChecker(
        config=ContactWrenchTrajectoryCheckerConfig.warmup_proxy()
    ).check(converted, context)
    assert result.feasible is True


def test_teacher_ranges_expose_a_payload_support_witness_at_order8_friction() -> None:
    context = compile_high_level_context(
        _task(),
        _morphology(),
        _candidate_set(friction=4.5),
    )
    converted = upgrade_teacher_trajectory_to_v2(_legacy_trajectory(), context)

    assignments = converted.knots[0].contact_assignments
    assert sum(float(item.wrench_upper[2]) for item in assignments) >= 9.80665
    evaluations = LightweightContactWrenchQPEvaluator().evaluate_trajectory(
        context=context,
        trajectory=converted,
    )

    assert len(evaluations) == 1
    assert evaluations[0].qp_residual is not None
    assert evaluations[0].qp_residual < 1.0e-4
    assert evaluations[0].wrench_residual is not None
    assert evaluations[0].wrench_residual < 1.0e-3


def test_teacher_record_archives_provenance_and_trajectory_check() -> None:
    morphology = _morphology()
    task = _task()
    observation = RuntimeObservation(
        time_s=0.25,
        morphology_graph=morphology,
        module_states=[],
        object_states=[],
        contact_states=[],
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
        task_progress=TaskProgressState(
            phase_label="apply_grasp_wrench",
            progress_ratio=0.4,
        ),
    )
    context = compile_high_level_context(
        task,
        morphology,
        _candidate_set(),
        runtime_observation=observation,
    )
    trajectory = upgrade_teacher_trajectory_to_v2(_legacy_trajectory(), context)

    record = teacher_interaction_record(
        record_id="teacher-record-0",
        episode_id="teacher-episode-0",
        split=DatasetSplit.TRAIN,
        decision_index=0,
        context=context,
        trajectory=trajectory,
        checker=ContactWrenchTrajectoryFeasibilityChecker(
            config=ContactWrenchTrajectoryCheckerConfig.warmup_proxy()
        ),
    )

    assert record.trajectory_provenance is not None
    assert (
        record.trajectory_provenance.source_kind
        == TrajectorySourceKind.DETERMINISTIC_TEACHER
    )
    assert record.trajectory_provenance.source_version == ORDER9_NATURAL_CONTACT_TEACHER_VERSION
    assert record.trajectory_feasibility_result is not None
    assert record.trajectory_feasibility_result.feasible is True
    assert InteractionTrajectoryRecord.from_json(record.to_json()).to_dict() == record.to_dict()


def test_order9_fallback_is_separate_and_emits_the_checked_v2_contract() -> None:
    class _Planner:
        def plan(self, _context):
            return _legacy_trajectory()

    context = compile_high_level_context(_task(), _morphology(), _candidate_set())
    teacher = Order9NaturalContactTeacher(_Planner())
    fallback = Order9NaturalContactFallback(teacher)

    trajectory = fallback.fallback(context)

    assert fallback.fallback_version == ORDER9_NATURAL_CONTACT_FALLBACK_VERSION
    assert trajectory.contract_version == CONTACT_WRENCH_CONTRACT_CONTACT_FRAME
    assert all(
        assignment.wrench_lower is not None
        for assignment in trajectory.knots[0].contact_assignments
    )


def test_rolling_teacher_records_are_composed_with_archived_window_semantics() -> None:
    morphology = _morphology()
    task = _task()
    checker = ContactWrenchTrajectoryFeasibilityChecker(
        config=ContactWrenchTrajectoryCheckerConfig.warmup_proxy()
    )
    records = []
    for index in range(5):
        observation = RuntimeObservation(
            time_s=0.25 * index,
            morphology_graph=morphology,
            module_states=[],
            object_states=[],
            contact_states=[],
            controller_status=ControllerStatus(status="ok", qp_feasible=True),
            task_progress=TaskProgressState(
                phase_label="complete" if index == 4 else "apply_grasp_wrench",
                progress_ratio=0.25 * index,
                success=index == 4,
            ),
        )
        context = compile_high_level_context(
            task,
            morphology,
            _candidate_set(),
            runtime_observation=observation,
        )
        trajectory = upgrade_teacher_trajectory_to_v2(_legacy_trajectory(), context)
        records.append(
            teacher_interaction_record(
                record_id=f"teacher-source-{index}",
                episode_id="teacher-window-episode",
                split=DatasetSplit.TRAIN,
                decision_index=index,
                context=context,
                trajectory=trajectory,
                checker=checker,
                decision_return=5.0 - index,
            )
        )

    windows = compose_order9_teacher_windows(
        records,
        checker=checker,
        config=Order9TeacherWindowConfig(horizon_s=1.0, knot_dt_s=0.25),
    )

    assert len(windows) == 5
    first = windows[0]
    assert first.trajectory_provenance is not None
    assert first.trajectory_provenance.source_version == ORDER9_TEACHER_WINDOW_VERSION
    assert first.trajectory_provenance.metadata["source_record_ids"] == [
        f"teacher-source-{index}" for index in range(5)
    ]
    assert first.trajectory_provenance.metadata["resampling_semantics"] == (
        "latest_decision_zero_order_hold_on_fixed_grid"
    )
    assert first.trajectory_provenance.metadata["decision_return_semantics"] == (
        "window_start_record_decision_return"
    )
    assert first.trajectory_provenance.metadata["terminal_tail_hold_used"] is False
    assert first.decision_return == 5.0
    assert [knot.t_rel_s for knot in first.trajectory.knots] == [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ]
    assert windows[-1].trajectory_provenance is not None
    assert windows[-1].trajectory_provenance.metadata["terminal_tail_hold_used"] is True


def _task():
    return build_order8_grasp_carry_task_spec(
        object_pose_world=(0.5, 0.0, 0.225, 0.0, 0.0, 0.0, 1.0),
        object_size_m=(0.30, 0.40, 0.15),
        object_mass_kg=1.0,
        object_friction=0.6,
        required_transport_distance_m=0.20,
        support_height_m=0.15,
        max_contact_force_n=30.0,
        max_contact_torque_nm=5.0,
    )


def _morphology() -> MorphologyGraph:
    return MorphologyGraph(
        graph_id="order9-teacher-morphology",
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


def _candidate_set(*, friction: float = 0.6) -> ContactCandidateSet:
    candidates = [
        ContactCandidate(
            candidate_id=0,
            slot_id=0,
            anchor_id=0,
            target_entity_id="order8_object",
            region_id="positive-x",
            contact_pose_world=(0.5, 0.0, 0.225, 0.0, 0.0, 0.0, 1.0),
            contact_frame_world=(0.5, 0.0, 0.225, 0.0, 0.0, 0.0, 1.0),
            normal_world=(1.0, 0.0, 0.0),
            tangent_basis_world=[0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            contact_mode=ContactMode.GRASP,
            friction=friction,
            patch_area_m2=0.01,
            candidate_scores={},
            unary_valid=True,
        ),
        ContactCandidate(
            candidate_id=1,
            slot_id=0,
            anchor_id=1,
            target_entity_id="order8_object",
            region_id="negative-x",
            contact_pose_world=(0.5, 0.0, 0.225, 0.0, 0.0, 0.0, 1.0),
            contact_frame_world=(0.5, 0.0, 0.225, 0.0, 0.0, 0.0, 1.0),
            normal_world=(-1.0, 0.0, 0.0),
            tangent_basis_world=[0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            contact_mode=ContactMode.GRASP,
            friction=friction,
            patch_area_m2=0.01,
            candidate_scores={},
            unary_valid=True,
        ),
    ]
    return ContactCandidateSet(
        set_id="order9-teacher-candidates",
        task_id="order8-natural-contact-smoke",
        morphology_graph_id="order9-teacher-morphology",
        candidates=candidates,
        candidate_mask=[True, True],
        slot_coverage={0: [0, 1]},
        pairwise_conflict_matrix=[[False, False], [False, False]],
        pairwise_compatibility_score=[[1.0, 1.0], [1.0, 1.0]],
        group_proposals=[],
        assignment_feasibility_cache={},
        sampler_version="order9-teacher-unit-test-v1",
    )


def _legacy_trajectory() -> ContactWrenchTrajectory:
    assignments = [
        ContactAssignment(
            slot_id=0,
            anchor_id=0,
            candidate_id=0,
            contact_mode=ContactMode.GRASP,
            schedule_state="maintain",
            wrench_target=[-5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        ContactAssignment(
            slot_id=0,
            anchor_id=1,
            candidate_id=1,
            contact_mode=ContactMode.GRASP,
            schedule_state="maintain",
            wrench_target=[5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
    ]
    return ContactWrenchTrajectory(
        horizon_s=0.1,
        dt_s=0.1,
        knots=[InteractionKnot(t_rel_s=0.0, contact_assignments=assignments)],
        derived_mode_label="order8-test-teacher",
    )
