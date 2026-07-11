from __future__ import annotations

import pytest

from amsrr.schemas import P4_3DatasetManifest
from amsrr.schemas.common import ContactMode, SchemaValidationError
from amsrr.schemas.contact_candidates import (
    AssignmentFeasibilityResult,
    ContactCandidate,
    ContactCandidateSet,
)
from amsrr.schemas.datasets import (
    P4_3_DATASET_SCHEMA_VERSION,
    DatasetKind,
    DatasetShard,
    DatasetSplit,
    DesignOutcomeRecord,
    InteractionTrajectoryRecord,
    LowLevelControlRecord,
    StageDecisionMasks,
)
from amsrr.schemas.feasibility import FeasibilityResult, Violation, ViolationSeverity
from amsrr.schemas.interaction_envelope import InteractionEnvelope
from amsrr.schemas.irg import IRGNode, IRGNodeType, InteractionRequirementGraph
from amsrr.schemas.morphology import DesignOutput, MorphologyGraph
from amsrr.schemas.policies import (
    ContactAssignment,
    ContactWrenchTrajectory,
    ControllerCommand,
    ControllerStatus,
    InteractionKnot,
    PolicyCommand,
)
from amsrr.schemas.runtime import RuntimeObservation, TaskProgressState


TASK_ID = "task-train"
EPISODE_ID = "episode-001"


def test_p4_3_dataset_records_roundtrip() -> None:
    context = _trajectory_context()
    low_level = _low_level_record(context)
    interaction = _interaction_record(context)
    outcome = _design_outcome_record(context["morphology"])

    assert LowLevelControlRecord.from_json(low_level.to_json()).to_dict() == low_level.to_dict()
    assert InteractionTrajectoryRecord.from_json(interaction.to_json()).to_dict() == interaction.to_dict()
    assert DesignOutcomeRecord.from_json(outcome.to_json()).to_dict() == outcome.to_dict()
    assert low_level.stage_masks == StageDecisionMasks(low_level_control_mask=True)
    assert interaction.stage_masks == StageDecisionMasks(high_level_decision_mask=True)
    assert outcome.stage_masks == StageDecisionMasks(design_decision_mask=True)


def test_low_level_record_requires_step_alignment_and_explicit_reward_pair() -> None:
    context = _trajectory_context()
    values = _low_level_record(context).to_dict()
    values["time_s"] = 0.5

    with pytest.raises(SchemaValidationError, match="runtime_observation.time_s"):
        LowLevelControlRecord.from_dict(values)

    values = _low_level_record(context).to_dict()
    values["reward"] = None
    with pytest.raises(SchemaValidationError, match="both be present or both be null"):
        LowLevelControlRecord.from_dict(values)

    values["reward_terms"] = None
    record_without_aligned_reward = LowLevelControlRecord.from_dict(values)
    assert record_without_aligned_reward.reward is None
    assert record_without_aligned_reward.reward_terms is None


def test_interaction_record_rejects_unknown_selected_candidate() -> None:
    values = _interaction_record(_trajectory_context()).to_dict()
    values["selected_candidate_ids"] = [999]

    with pytest.raises(SchemaValidationError, match="unknown"):
        InteractionTrajectoryRecord.from_dict(values)


def test_design_outcome_keeps_hard_infeasible_candidates_off_rollout() -> None:
    morphology = _morphology()
    values = _design_outcome_record(morphology).to_dict()
    values["feasibility_result"] = FeasibilityResult(
        feasible=False,
        hard_violations=[
            Violation(
                code="E_HARD",
                severity=ViolationSeverity.HARD,
                message="deterministic hard rejection",
            )
        ],
        soft_violations=[],
        margins={},
        proxy_scores={},
        checker_version="checker-v1",
    ).to_dict()

    with pytest.raises(SchemaValidationError, match="deterministically infeasible"):
        DesignOutcomeRecord.from_dict(values)

    values.update(
        {
            "episode_id": None,
            "selected_for_rollout": False,
            "rollout_executed": False,
            "task_success": None,
            "object_dropped": None,
            "hard_collision": None,
            "controller_infeasible_terminal": None,
            "episode_return": None,
            "failure_reason": "E_HARD",
        }
    )
    rejected = DesignOutcomeRecord.from_dict(values)
    assert rejected.feasibility_result.feasible is False
    assert rejected.rollout_executed is False


def test_p4_3_manifest_roundtrip_and_task_disjoint_validation() -> None:
    manifest = _manifest()

    assert P4_3DatasetManifest.from_json(manifest.to_json()).to_dict() == manifest.to_dict()
    assert manifest.schema_version == P4_3_DATASET_SCHEMA_VERSION

    values = manifest.to_dict()
    values["validation_task_ids"] = ["task-train"]
    with pytest.raises(SchemaValidationError, match="task splits must be disjoint"):
        P4_3DatasetManifest.from_dict(values)


def test_p4_3_manifest_rejects_shard_count_mismatch() -> None:
    values = _manifest().to_dict()
    values["record_counts"][DatasetKind.LOW_LEVEL_CONTROL.value] = 2

    with pytest.raises(SchemaValidationError, match="must match shard counts"):
        P4_3DatasetManifest.from_dict(values)


def _trajectory_context() -> dict[str, object]:
    morphology = _morphology()
    status = ControllerStatus(status="ok", qp_feasible=True)
    runtime = RuntimeObservation(
        time_s=0.25,
        morphology_graph=morphology,
        module_states=[],
        object_states=[],
        contact_states=[],
        controller_status=status,
        task_progress=TaskProgressState(phase_label="maintain", progress_ratio=0.25),
    )
    assignment = ContactAssignment(
        slot_id=2,
        anchor_id=3,
        candidate_id=7,
        contact_mode=ContactMode.GRASP,
        schedule_state="maintain",
    )
    knot = InteractionKnot(t_rel_s=0.25, contact_assignments=[assignment])
    trajectory = ContactWrenchTrajectory(horizon_s=1.0, dt_s=0.25, knots=[knot])
    candidate = ContactCandidate(
        candidate_id=7,
        slot_id=2,
        anchor_id=3,
        target_entity_id="box-1",
        region_id="box-side",
        contact_pose_world=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        contact_frame_world=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        normal_world=(1.0, 0.0, 0.0),
        tangent_basis_world=[0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        contact_mode=ContactMode.GRASP,
        friction=0.6,
        patch_area_m2=0.01,
        candidate_scores={"normal_alignment": 1.0},
        unary_valid=True,
    )
    candidate_set = ContactCandidateSet(
        set_id="candidate-set-1",
        task_id=TASK_ID,
        morphology_graph_id=morphology.graph_id,
        candidates=[candidate],
        candidate_mask=[True],
        slot_coverage={2: [7]},
        pairwise_conflict_matrix=[[False]],
        pairwise_compatibility_score=[[1.0]],
        group_proposals=[],
        assignment_feasibility_cache={
            "7": AssignmentFeasibilityResult(
                assignment_key="7",
                candidate_ids=[7],
                feasible=True,
                violation_codes=[],
            )
        },
        sampler_version="sampler-v1",
    )
    irg = InteractionRequirementGraph(
        irg_id="irg-1",
        task_id=TASK_ID,
        nodes=[IRGNode(0, IRGNodeType.TASK, TASK_ID, 1.0, True, None)],
        edges=[],
    )
    envelope = InteractionEnvelope(
        envelope_id="envelope-1",
        task_id=TASK_ID,
        required_contact_count_range=(1, 1),
        required_contact_modes=[ContactMode.GRASP],
        target_region_sets=[],
        wrench_space_requirements=[],
    )
    return {
        "morphology": morphology,
        "status": status,
        "runtime": runtime,
        "knot": knot,
        "trajectory": trajectory,
        "candidate_set": candidate_set,
        "irg": irg,
        "envelope": envelope,
    }


def _low_level_record(context: dict[str, object]) -> LowLevelControlRecord:
    return LowLevelControlRecord(
        record_id="low-level-0001",
        episode_id=EPISODE_ID,
        task_id=TASK_ID,
        split=DatasetSplit.TRAIN,
        step_index=1,
        time_s=0.25,
        trajectory_record_id="trajectory-0001",
        active_trajectory_index=0,
        active_knot_index=0,
        runtime_observation=context["runtime"],
        active_knot=context["knot"],
        policy_command=PolicyCommand(desired_body_twist=[0.0] * 6),
        controller_command=ControllerCommand(
            rotor_thrusts_n={"rotor-0": 1.0},
            vectoring_joint_targets={},
            joint_torque_commands={},
            dock_mechanism_commands={},
            controller_status=context["status"],
        ),
        actuator_target_record={"rotor-0": 1.0},
        reward_terms={"r_tracking": 0.5, "r_energy": -0.1},
        reward=0.4,
        terminal=False,
        stage_masks=StageDecisionMasks(low_level_control_mask=True),
    )


def _interaction_record(context: dict[str, object]) -> InteractionTrajectoryRecord:
    return InteractionTrajectoryRecord(
        record_id="trajectory-0001",
        episode_id=EPISODE_ID,
        task_id=TASK_ID,
        split=DatasetSplit.TRAIN,
        decision_index=0,
        decision_time_s=0.25,
        irg=context["irg"],
        interaction_envelope=context["envelope"],
        morphology_graph=context["morphology"],
        contact_candidate_set=context["candidate_set"],
        runtime_observation=context["runtime"],
        trajectory=context["trajectory"],
        selected_candidate_ids=[7],
        assignment_feasibility_results=[
            AssignmentFeasibilityResult(
                assignment_key="7",
                candidate_ids=[7],
                feasible=True,
                violation_codes=[],
            )
        ],
        decision_return=1.0,
        stage_masks=StageDecisionMasks(high_level_decision_mask=True),
    )


def _design_outcome_record(morphology: MorphologyGraph) -> DesignOutcomeRecord:
    return DesignOutcomeRecord(
        record_id="design-outcome-0001",
        episode_id=EPISODE_ID,
        task_id=TASK_ID,
        split=DatasetSplit.TRAIN,
        candidate_id=0,
        selected_for_rollout=True,
        design_output=DesignOutput(
            task_id=TASK_ID,
            irg_id="irg-1",
            target_morphology=morphology,
            module_roles={},
            slot_anchor_binding_prior=[],
            design_actions=[],
        ),
        feasibility_result=FeasibilityResult(
            feasible=True,
            hard_violations=[],
            soft_violations=[],
            margins={"payload_margin": 1.0},
            proxy_scores={"score": 1.0},
            checker_version="checker-v1",
        ),
        rollout_executed=True,
        task_success=True,
        object_dropped=False,
        hard_collision=False,
        controller_infeasible_terminal=False,
        episode_return=1.0,
        rollout_metrics={"success_rate": 1.0},
        failure_reason=None,
        stage_masks=StageDecisionMasks(design_decision_mask=True),
    )


def _morphology() -> MorphologyGraph:
    return MorphologyGraph(
        graph_id="morphology-1",
        modules=[],
        ports=[],
        dock_edges=[],
        robot_anchors=[],
        control_groups=[],
        base_module_id=0,
        is_closed_loop=False,
    )


def _manifest() -> P4_3DatasetManifest:
    shards = [
        DatasetShard(
            dataset_kind=kind,
            split=None,
            path=f"{kind.value}.jsonl",
            record_count=1,
            sha256=f"{kind.value}-hash",
        )
        for kind in (
            DatasetKind.LOW_LEVEL_CONTROL,
            DatasetKind.INTERACTION_TRAJECTORY,
            DatasetKind.DESIGN_OUTCOME,
        )
    ]
    return P4_3DatasetManifest(
        dataset_id="p4-3-dataset-001",
        schema_version=P4_3_DATASET_SCHEMA_VERSION,
        source_archive_paths=["rollouts/source.jsonl"],
        source_episode_ids=[EPISODE_ID],
        train_task_ids=["task-train"],
        validation_task_ids=["task-validation"],
        held_out_task_ids=["task-held-out"],
        shards=shards,
        record_counts={shard.dataset_kind.value: shard.record_count for shard in shards},
        source_hash="source-hash",
        config_hash="config-hash",
        robot_model_hash="robot-model-hash",
        urdf_hash="urdf-hash",
        thrust_model_hash="thrust-model-hash",
        task_hashes={
            "task-train": "task-train-hash",
            "task-validation": "task-validation-hash",
            "task-held-out": "task-held-out-hash",
        },
        geometry_hashes={"box-geometry": "box-geometry-hash"},
        random_seeds=[0, 1, 2],
        simulator_version="isaac-sim-test",
        simulator_hash="isaac-sim-hash",
    )
