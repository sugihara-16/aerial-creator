from __future__ import annotations

import json
from pathlib import Path

import pytest

from amsrr.policies.contact_wrench_trajectory import P4_2DeterministicGraspCarryPlanner
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.policies.learned_high_level_policy import LearnedHighLevelPolicy
from amsrr.schemas.common import ContactMode, SchemaValidationError
from amsrr.schemas.contact_candidates import (
    ContactCandidate,
    ContactCandidateGroupProposal,
    ContactCandidateSet,
)
from amsrr.schemas.datasets import (
    DatasetSplit,
    InteractionTrajectoryRecord,
    StageDecisionMasks,
)
from amsrr.schemas.interaction_envelope import InteractionEnvelope
from amsrr.schemas.irg import IRGNode, IRGNodeType, InteractionRequirementGraph
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.policies import ControllerStatus, ContactWrenchTrajectory
from amsrr.schemas.runtime import RuntimeObservation, TaskProgressState
from amsrr.training.p4_3_pi_h_training import (
    load_interaction_trajectory_records,
    load_p4_3_pi_h_training_config,
    train_p4_3_pi_h,
)


def test_p4_3_pi_h_config_matches_bootstrap_contract() -> None:
    config = load_p4_3_pi_h_training_config()

    assert config.epochs == 20
    assert config.learning_rate == 0.001
    assert config.hidden_dim == 64
    assert config.batch_size == 64
    assert config.seed == 13
    assert config.update_rate_hz == 2.0
    assert config.checkpoint_dir == "artifacts/p4_3/pi_h"


def test_p4_3_pi_h_training_consumes_direct_records_and_writes_required_artifacts(
    tmp_path: Path,
) -> None:
    records = [
        _record(
            task_id="pi-h-train-task",
            split=DatasetSplit.TRAIN,
            record_id="train-001",
            teacher_candidate_id=101,
        ),
        _record(
            task_id="pi-h-train-task",
            split=DatasetSplit.TRAIN,
            record_id="train-002",
            teacher_candidate_id=305,
        ),
        _record(
            task_id="pi-h-validation-task",
            split=DatasetSplit.VALIDATION,
            record_id="validation-001",
            teacher_candidate_id=101,
        ),
    ]
    shard_path = tmp_path / "interaction_trajectory.jsonl"
    shard_path.write_text(
        "".join(record.to_json() + "\n" for record in records),
        encoding="utf-8",
    )

    manifest = train_p4_3_pi_h(
        shard_paths=shard_path,
        output_dir=tmp_path / "training",
        epochs=3,
        learning_rate=0.01,
        seed=3,
        hidden_dim=12,
    )

    expected_names = {
        "checkpoint.pt",
        "metrics.json",
        "loss_curve.csv",
        "rollout_evaluation.json",
        "fallback_metadata.json",
    }
    assert {path.name for path in (tmp_path / "training").iterdir()} == expected_names
    assert Path(manifest.checkpoint_path).is_file()
    assert Path(manifest.metrics_path).is_file()
    assert Path(manifest.loss_curve_path).is_file()
    assert Path(manifest.rollout_evaluation_path).is_file()
    assert Path(manifest.fallback_metadata_path).is_file()

    metrics = json.loads(Path(manifest.metrics_path).read_text(encoding="utf-8"))
    assert metrics["training_stage"] == "P4.3c"
    assert metrics["output_contract"] == "ContactWrenchTrajectory"
    assert metrics["actuator_command_output"] is False
    assert metrics["num_train_records"] == 2.0
    assert metrics["num_validation_records"] == 1.0
    assert metrics["validation_schema_valid_rate"] == 1.0
    assert metrics["validation_assignment_feasible_rate"] == 1.0

    rollout = json.loads(
        Path(manifest.rollout_evaluation_path).read_text(encoding="utf-8")
    )
    assert rollout["evaluation_type"] == "offline_teacher_record_decode"
    assert rollout["deterministic_fallback_available"] is True
    assert rollout["deterministic_safety_gate_used"] is True
    assert rollout["actuator_command_output"] is False
    assert rollout["isaac_rollout_claim"] is False

    fallback = json.loads(
        Path(manifest.fallback_metadata_path).read_text(encoding="utf-8")
    )
    assert fallback["fallback_class"] == "P4_2DeterministicGraspCarryPlanner"
    assert fallback["hard_safety_source"] == "evaluate_selected_assignment_feasibility"

    loaded_policy = LearnedHighLevelPolicy.from_checkpoint(manifest.checkpoint_path)
    loaded_trajectory = loaded_policy.plan(_context("checkpoint-eval", 101))
    assert ContactWrenchTrajectory.from_json(loaded_trajectory.to_json())
    assert not hasattr(loaded_trajectory, "rotor_thrusts_n")


def test_pi_h_loader_rejects_task_leakage_between_splits(tmp_path: Path) -> None:
    records = [
        _record(
            task_id="leaked-task",
            split=DatasetSplit.TRAIN,
            record_id="leak-train",
            teacher_candidate_id=101,
        ),
        _record(
            task_id="leaked-task",
            split=DatasetSplit.VALIDATION,
            record_id="leak-validation",
            teacher_candidate_id=101,
        ),
    ]
    shard_path = tmp_path / "leaked.jsonl"
    shard_path.write_text(
        "".join(record.to_json() + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(SchemaValidationError, match="task splits must be disjoint"):
        load_interaction_trajectory_records(shard_path)


def _record(
    *,
    task_id: str,
    split: DatasetSplit,
    record_id: str,
    teacher_candidate_id: int,
) -> InteractionTrajectoryRecord:
    context = _context(task_id, teacher_candidate_id)
    trajectory = P4_2DeterministicGraspCarryPlanner().plan(context)
    selected_ids = sorted(
        {
            assignment.candidate_id
            for knot in trajectory.knots
            for assignment in knot.contact_assignments
        }
    )
    assert selected_ids == [teacher_candidate_id]
    runtime = context.runtime_observation
    assert runtime is not None
    return InteractionTrajectoryRecord(
        record_id=record_id,
        episode_id=f"episode-{record_id}",
        task_id=task_id,
        split=split,
        decision_index=0,
        decision_time_s=runtime.time_s,
        irg=context.irg,
        interaction_envelope=context.interaction_envelope,
        morphology_graph=context.morphology_graph,
        contact_candidate_set=context.contact_candidate_set,
        runtime_observation=runtime,
        trajectory=trajectory,
        selected_candidate_ids=selected_ids,
        assignment_feasibility_results=list(
            context.contact_candidate_set.assignment_feasibility_cache.values()
        ),
        decision_return=1.0,
        stage_masks=StageDecisionMasks(high_level_decision_mask=True),
    )


def _context(task_id: str, teacher_candidate_id: int) -> HighLevelPolicyContext:
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
        _candidate(101, anchor_id=11),
        _candidate(305, anchor_id=12),
    ]
    primary_score = 2.0 if teacher_candidate_id == 101 else 1.0
    secondary_score = 2.0 if teacher_candidate_id == 305 else 1.0
    candidate_set = ContactCandidateSet(
        set_id=f"candidates-{task_id}-{teacher_candidate_id}",
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
                group_score=primary_score,
            ),
            ContactCandidateGroupProposal(
                group_id="group-secondary",
                candidate_ids=[305],
                group_type="grasp_pair",
                group_score=secondary_score,
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
        object_states=[],
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


def _candidate(candidate_id: int, *, anchor_id: int) -> ContactCandidate:
    normal = (1.0, 0.0, 0.0) if candidate_id == 101 else (-1.0, 0.0, 0.0)
    return ContactCandidate(
        candidate_id=candidate_id,
        slot_id=7,
        anchor_id=anchor_id,
        target_entity_id="box-1",
        region_id=f"region-{candidate_id}",
        contact_pose_world=(0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0),
        contact_frame_world=(0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0),
        normal_world=normal,
        tangent_basis_world=[0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        contact_mode=ContactMode.GRASP,
        friction=0.6,
        patch_area_m2=0.01,
        candidate_scores={"normal_alignment": 1.0},
        unary_valid=True,
    )
