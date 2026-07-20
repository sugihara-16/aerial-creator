from __future__ import annotations

from pathlib import Path

import torch

from amsrr.feasibility.checker import FeasibilityChecker
from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.policies.design_policy_base import DesignPolicyContext
from amsrr.policies.design_teacher import (
    DesignTeacherVariant,
    DeterministicDesignTeacher,
)
from amsrr.policies.order9_design_grammar import Order9DesignGrammar
from amsrr.policies.order9_design_policy import Order9DesignPolicyConfig
from amsrr.policies.order9_design_policy import Order9AutoregressiveDesignPolicy
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.datasets import (
    P4_3_DATASET_SCHEMA_VERSION,
    DatasetKind,
    DatasetShard,
    DatasetSplit,
    DesignActionCandidateRecord,
    P4_3DatasetManifest,
    PolicyBehaviorTrace,
    SequentialDesignStepRecord,
    SequentialDesignTrajectoryRecord,
    StageDecisionMasks,
    TrajectoryProvenance,
    TrajectorySourceKind,
)
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.order9_curriculum import (
    Order9PPOOptimizationConfig,
    load_order9_learning_config,
)
from amsrr.training.order9_offline_training import (
    reconstruct_order9_pi_d_teacher_trace,
    train_order9_behavior_cloning,
)
from amsrr.training.order9_ppo import (
    order9_pi_d_behavior_trace,
    update_order9_pi_d_ppo,
)
from amsrr.utils.hashing import hash_file


def test_order9_pi_d_offline_bc_replays_masks_and_writes_strict_checkpoint(
    tmp_path: Path,
    grasp_carry_dict: dict,
) -> None:
    physical_model = build_physical_model_from_config(
        "configs/robot/robot_model.yaml"
    )
    train = _design_record(
        grasp_carry_dict,
        task_id="order9-bc-train",
        episode_id="order9-bc-train-episode",
        split=DatasetSplit.TRAIN,
        physical_model=physical_model,
    )
    validation = _design_record(
        grasp_carry_dict,
        task_id="order9-bc-validation",
        episode_id="order9-bc-validation-episode",
        split=DatasetSplit.VALIDATION,
        physical_model=physical_model,
    )
    manifest_path = _write_dataset(tmp_path / "dataset", train, validation)
    config = load_order9_learning_config()
    config.optimization.pi_d_bc.epochs = 1
    config.optimization.pi_d_bc.batch_size = 1

    result = train_order9_behavior_cloning(
        config,
        stage_id="c7_pi_d_structured_bc",
        dataset_manifest_path=manifest_path,
        physical_model=physical_model,
        output_dir=tmp_path / "training",
        git_revision="unit-test",
        device="cpu",
        model_config=Order9DesignPolicyConfig(d_model=16, maximum_design_steps=64),
    )

    assert result.policy_family.value == "pi_d"
    assert result.training_record_count == 1
    assert result.validation_record_count == 1
    assert Path(result.checkpoint_path).is_file()
    assert Path(result.metrics_path).is_file()
    assert Path(result.loss_curve_path).is_file()
    assert hash_file(result.checkpoint_path) == result.checkpoint_sha256
    context, trace = reconstruct_order9_pi_d_teacher_trace(train, physical_model)
    assert context.task_spec.task_id == train.task_id
    assert len(trace) == len(train.steps)


def test_order9_pi_d_stochastic_trace_replays_through_masked_ppo(
    grasp_carry_dict: dict,
) -> None:
    physical_model = build_physical_model_from_config(
        "configs/robot/robot_model.yaml"
    )
    record = _design_record(
        grasp_carry_dict,
        task_id="order9-ppo-train",
        episode_id="order9-ppo-train-episode",
        split=DatasetSplit.TRAIN,
        physical_model=physical_model,
    )
    policy = Order9AutoregressiveDesignPolicy(
        Order9DesignPolicyConfig(d_model=16, maximum_design_steps=64)
    )
    checkpoint_sha = "b" * 64
    context, trace = reconstruct_order9_pi_d_teacher_trace(record, physical_model)
    history = policy.initial_history()
    for index, teacher_step in enumerate(trace):
        output = policy.forward_step(
            context,
            teacher_step.state,
            teacher_step.candidate_step.candidates,
            history=history,
        )
        selected_index = record.steps[index].selected_candidate_index
        evaluation = policy.evaluate_selected_step(output, selected_index)
        record.steps[index].behavior_trace = order9_pi_d_behavior_trace(
            evaluation,
            selected_action=teacher_step.candidate_step.selected_action.to_dict(),
            checkpoint_sha256=checkpoint_sha,
        )
        history = policy.advance_history(output, selected_index)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-4)

    result = update_order9_pi_d_ppo(
        policy,
        [record],
        physical_model=physical_model,
        optimizer=optimizer,
        config=Order9PPOOptimizationConfig(
            rollout_steps_per_environment=1,
            epochs_per_update=1,
            minibatch_size=256,
        ),
        behavior_checkpoint_sha256=checkpoint_sha,
        seed=8,
    )

    assert result.policy_family == "pi_d"
    assert result.sample_count == len(record.steps)
    assert result.optimizer_step_count == 1


def _design_record(
    template: dict,
    *,
    task_id: str,
    episode_id: str,
    split: DatasetSplit,
    physical_model,
) -> SequentialDesignTrajectoryRecord:
    raw = dict(template)
    raw["task_id"] = task_id
    task = TaskSpec.from_dict(raw)
    irg = IRGBuilder().build(task)
    envelope = InteractionEnvelopeExtractor().extract(irg)
    context = DesignPolicyContext(task, irg, physical_model, envelope)
    teacher = DeterministicDesignTeacher().generate(
        context,
        variant=DesignTeacherVariant.SYMMETRIC_TWO_ANCHOR_GRASP,
    ).design_output
    grammar = Order9DesignGrammar(context)
    trace = grammar.teacher_trace(teacher)
    state = grammar.initial_state()
    steps = []
    for index, item in enumerate(trace):
        selected_index = next(
            candidate_index
            for candidate_index, candidate in enumerate(item.candidate_step.candidates)
            if candidate.action.to_dict()
            == item.candidate_step.selected_action.to_dict()
        )
        steps.append(
            SequentialDesignStepRecord(
                step_index=index,
                partial_action_history=list(item.state.action_history),
                candidates=[
                    DesignActionCandidateRecord(
                        candidate_index=candidate_index,
                        action=candidate.action,
                        valid=candidate.valid,
                        reason_code=candidate.reason_code,
                        score_prior=candidate.score_prior,
                    )
                    for candidate_index, candidate in enumerate(
                        item.candidate_step.candidates
                    )
                ],
                selected_candidate_index=selected_index,
                reward=1.0 if index == len(trace) - 1 else 0.0,
                terminal=index == len(trace) - 1,
                truncated=False,
                behavior_trace=PolicyBehaviorTrace(
                    policy_family="pi_d",
                    policy_version="order9_deterministic_design_teacher_v1",
                    action_semantics="masked_grammar_candidate_index",
                    action_payload={
                        "selected_action": item.candidate_step.selected_action.to_dict()
                    },
                ),
            )
        )
        state = grammar.apply(
            state, item.candidate_step.candidates[selected_index]
        )
    design = grammar.build_design_output(state)
    feasibility = FeasibilityChecker().check_design(
        design,
        task_spec=task,
        irg=irg,
        physical_model=physical_model,
    )
    return SequentialDesignTrajectoryRecord(
        record_id=f"{episode_id}:design",
        episode_id=episode_id,
        task_id=task_id,
        split=split,
        task_spec=task,
        irg=irg,
        interaction_envelope=envelope,
        physical_model_hash=physical_model.stable_hash(),
        steps=steps,
        design_output=design,
        feasibility_result=feasibility,
        episode_return=1.0,
        task_success=True,
        failure_reason=None,
        stage_masks=StageDecisionMasks(design_decision_mask=True),
        trajectory_provenance=TrajectoryProvenance(
            source_kind=TrajectorySourceKind.DETERMINISTIC_TEACHER,
            source_version="order9_deterministic_design_teacher_v1",
        ),
    )


def _write_dataset(
    directory: Path,
    train: SequentialDesignTrajectoryRecord,
    validation: SequentialDesignTrajectoryRecord,
) -> Path:
    directory.mkdir(parents=True)
    shards = []
    for split, record in (
        (DatasetSplit.TRAIN, train),
        (DatasetSplit.VALIDATION, validation),
    ):
        path = directory / f"design_action_trajectory_{split.value}.jsonl"
        path.write_text(record.to_json() + "\n", encoding="utf-8")
        shards.append(
            DatasetShard(
                dataset_kind=DatasetKind.DESIGN_ACTION_TRAJECTORY,
                split=split,
                path=path.name,
                record_count=1,
                sha256=hash_file(path),
            )
        )
    manifest = P4_3DatasetManifest(
        dataset_id="order9-pi-d-bc-unit",
        schema_version=P4_3_DATASET_SCHEMA_VERSION,
        source_archive_paths=["unit-generated"],
        source_episode_ids=[train.episode_id, validation.episode_id],
        train_task_ids=[train.task_id],
        validation_task_ids=[validation.task_id],
        held_out_task_ids=[],
        shards=shards,
        record_counts={DatasetKind.DESIGN_ACTION_TRAJECTORY.value: 2},
        source_hash="unit-source",
        config_hash="unit-config",
        robot_model_hash="unit-robot",
        urdf_hash="unit-urdf",
        thrust_model_hash="unit-thrust",
        task_hashes={train.task_id: "train-hash", validation.task_id: "validation-hash"},
        geometry_hashes={"unit-geometry": "unit-geometry-hash"},
        random_seeds=[1, 2],
        simulator_version="unit",
        simulator_hash="unit-simulator",
    )
    path = directory / "manifest.json"
    path.write_text(manifest.to_json(indent=2) + "\n", encoding="utf-8")
    return path
