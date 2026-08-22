from __future__ import annotations

import json
from pathlib import Path

import pytest

from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.morphology.random_connected import (
    RandomConnectedMorphologyDistribution,
    morphology_structural_hash,
)
from amsrr.policies.design_policy_base import DesignPolicyContext
from amsrr.policies.order9_design_policy import (
    Order9AutoregressiveDesignPolicy,
    Order9DesignPolicyConfig,
)
from amsrr.policies.order9_low_level_policy import (
    Order9ContactResidualPhaseConditionedActorCritic,
    Order9MorphologyInvariantCompressionActorCritic,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.datasets import (
    DatasetSplit,
    TrajectoryProvenance,
    TrajectorySourceKind,
)
from amsrr.schemas.order9 import Order9PolicyFamily
from amsrr.training.order9_checkpoints import (
    load_order9_policy_checkpoint,
    save_order9_policy_checkpoint,
)
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_design_teacher_dataset import (
    Order9PiDTeacherDatasetConfig,
    _order9_base_task,
    build_order9_pi_d_teacher_record,
)
from amsrr.training.order9_dataset import load_order9_dataset
from amsrr.training.order9_offline_training import (
    build_order9_checkpoint_metadata,
    reconstruct_order9_pi_d_teacher_trace,
)
from amsrr.training.order9_online_dataset import write_order9_on_policy_dataset
from amsrr.training.order9_online_training import (
    _order9_c3_action_contract_trainable_prefixes,
    _validate_parent,
    train_order9_ppo_update,
)
from amsrr.training.order9_c3_action_contract import (
    order9_c3_action_contract_global_dimension,
    order9_c3_action_contract_uses_full_policy,
)
from amsrr.training.order9_pipeline import order9_schedule_hash, order9_stage_by_id
from amsrr.training.order9_ppo import order9_pi_d_behavior_trace
from amsrr.training.order9_randomization import Order9ExpandedObjectRandomizer
from amsrr.training.p2_inspection_context import default_grasp_carry_task_spec


def test_explicit_c3_contract_trains_the_v6_common_compression_head() -> None:
    v5 = Order9ContactResidualPhaseConditionedActorCritic()
    v6 = Order9MorphologyInvariantCompressionActorCritic()

    assert _order9_c3_action_contract_trainable_prefixes(
        v5, global_dimension=0
    ) == ("contact_residual_decoder.", "critic.")
    assert _order9_c3_action_contract_trainable_prefixes(
        v6, global_dimension=0
    ) == (
        "contact_compression_actor_mean.",
        "contact_compression_actor_log_std",
        "contact_compression_module_count_bias",
        "critic.",
    )
    assert "contact_residual_decoder." not in (
        _order9_c3_action_contract_trainable_prefixes(
            v6, global_dimension=0
        )
    )


def test_v6_c3_contract_adds_only_the_requested_global_head() -> None:
    v6 = Order9MorphologyInvariantCompressionActorCritic()

    prefixes = _order9_c3_action_contract_trainable_prefixes(
        v6, global_dimension=12
    )

    assert prefixes[-2:] == ("actor_mean.", "actor_log_std")
    assert "contact_residual_decoder." not in prefixes


def test_full_c3_contract_declares_complete_policy_command() -> None:
    contract = "full_policy_command_plus_independent_compression"

    assert order9_c3_action_contract_global_dimension(contract) == 18
    assert order9_c3_action_contract_uses_full_policy(contract)


def test_same_stage_warm_start_initializer_must_restart_at_update_zero() -> None:
    config = load_order9_learning_config()
    stage = order9_stage_by_id(config, "c3_pi_l_ppo_arbitrary_morphology")
    physical_model = build_physical_model_from_config(
        "configs/robot/robot_model.yaml"
    )
    policy = Order9AutoregressiveDesignPolicy(
        Order9DesignPolicyConfig(d_model=16, maximum_design_steps=64)
    )
    metadata = build_order9_checkpoint_metadata(
        policy,
        stage=stage,
        schedule_hash=order9_schedule_hash(config),
        physical_model_hash=physical_model.stable_hash(),
        git_revision="unit-test",
        random_seed=1,
        input_artifact_hashes={"unit": "a" * 64},
        parent_checkpoint_sha256="b" * 64,
        source_order3_checkpoint_sha256=None,
        metrics={"loss": 1.0},
        trainer_version="unit_test_warm_start",
        extra_metadata={
            "ppo_update_index": 21,
            "warm_start_only": True,
            "warm_start_requires_fresh_on_policy_ppo": True,
        },
    )

    _validate_parent(metadata, stage, 0, physical_model)
    with pytest.raises(
        SchemaValidationError,
        match="warm-start initializer must begin at PPO update zero",
    ):
        _validate_parent(metadata, stage, 1, physical_model)


def test_one_generation_one_pi_d_ppo_update_is_hash_bound(
    tmp_path: Path,
) -> None:
    config = load_order9_learning_config()
    config.optimization.pi_d_ppo.epochs_per_update = 1
    config.optimization.pi_d_ppo.minibatch_size = 256
    model_physics = build_physical_model_from_config(
        "configs/robot/robot_model.yaml"
    )
    policy = Order9AutoregressiveDesignPolicy(
        Order9DesignPolicyConfig(d_model=16, maximum_design_steps=64)
    )
    parent_stage = order9_stage_by_id(config, "post_r4_pi_d_structured_bc")
    parent_metadata = build_order9_checkpoint_metadata(
        policy,
        stage=parent_stage,
        schedule_hash=order9_schedule_hash(config),
        physical_model_hash=model_physics.stable_hash(),
        git_revision="unit-test",
        random_seed=1,
        input_artifact_hashes={"unit": "a" * 64},
        parent_checkpoint_sha256=None,
        source_order3_checkpoint_sha256=None,
        metrics={"loss": 1.0},
        trainer_version="unit_test_parent",
    )
    parent_path = tmp_path / "parent.pt"
    parent_sha = save_order9_policy_checkpoint(
        parent_path, model=policy, metadata=parent_metadata
    )
    records = []
    task_specs = {}
    structural_hashes: set[str] = set()
    distribution = RandomConnectedMorphologyDistribution(model_physics)
    base = _order9_base_task(
        default_grasp_carry_task_spec(), Order9PiDTeacherDatasetConfig()
    )
    for index, split in enumerate(
        (DatasetSplit.TRAIN, DatasetSplit.VALIDATION)
    ):
        seed = 200 + index
        while True:
            target = distribution.sample(seed=seed, module_count=3)
            structural_hash = morphology_structural_hash(target)
            if structural_hash not in structural_hashes:
                structural_hashes.add(structural_hash)
                break
            seed += 1
        task = Order9ExpandedObjectRandomizer().sample(
            base, seed=9800 + index, sample_index=index
        ).task_spec
        irg = IRGBuilder().build(task)
        envelope = InteractionEnvelopeExtractor().extract(irg)
        record = build_order9_pi_d_teacher_record(
            DesignPolicyContext(task, irg, model_physics, envelope),
            target,
            episode_id=f"online-pi-d-{index}",
            split=split,
        )
        context, trace = reconstruct_order9_pi_d_teacher_trace(
            record, model_physics
        )
        history = policy.initial_history()
        for step_index, teacher_step in enumerate(trace):
            output = policy.forward_step(
                context,
                teacher_step.state,
                teacher_step.candidate_step.candidates,
                history=history,
            )
            selected = record.steps[step_index].selected_candidate_index
            evaluation = policy.evaluate_selected_step(output, selected)
            record.steps[step_index].behavior_trace = order9_pi_d_behavior_trace(
                evaluation,
                selected_action=(
                    teacher_step.candidate_step.selected_action.to_dict()
                ),
                checkpoint_sha256=parent_sha,
            )
            history = policy.advance_history(output, selected)
        record.trajectory_provenance = TrajectoryProvenance(
            source_kind=TrajectorySourceKind.LEARNED_POLICY,
            source_version="unit_test_online_pi_d",
            policy_checkpoint_sha256=parent_sha,
        )
        records.append(record)
        task_specs[task.task_id] = task
    raw_path = tmp_path / "raw_isaac.json"
    raw_path.write_text("{}\n", encoding="utf-8")
    dataset_dir = tmp_path / "rollout"
    manifest = write_order9_on_policy_dataset(
        dataset_dir,
        generation_id="unit-generation-0",
        design_records=records,
        task_specs=task_specs,
        behavior_checkpoint_sha256_by_family={"pi_d": parent_sha},
        source_isaac_artifact_paths=[raw_path],
        on_policy_environment_step_count=2,
        random_seeds=[9800, 9801],
        config_hash="b" * 64,
        robot_model_hash=model_physics.stable_hash(),
        urdf_hash="c" * 64,
        thrust_model_hash=str(
            model_physics.metadata["thrust_model_hash"]
        ),
        simulator_version="unit-isaac",
        simulator_hash="d" * 64,
    )

    result = train_order9_ppo_update(
        config,
        stage_id="post_r4_pi_d_masked_ppo",
        rollout_manifest_path=dataset_dir / "manifest.json",
        rollout_bundle=load_order9_dataset(dataset_dir / "manifest.json"),
        parent_checkpoint_path=parent_path,
        physical_model=model_physics,
        output_dir=tmp_path / "update0",
        git_revision="unit-test",
        update_index=0,
        device="cpu",
    )
    child = load_order9_policy_checkpoint(
        result.checkpoint_path,
        expected_family=Order9PolicyFamily.PI_D,
        expected_schedule_hash=order9_schedule_hash(config),
    )

    assert manifest.metadata["one_fresh_generation"] is True
    assert result.parent_checkpoint_sha256 == parent_sha
    assert result.consumed_environment_steps == 2
    metrics = json.loads(Path(result.metrics_path).read_text(encoding="utf-8"))
    assert metrics["update_wall_elapsed_s"] > 0.0
    assert metrics["consumed_environment_steps_per_s"] > 0.0
    assert metrics["runtime_load"]["device"] == "cpu"
    assert metrics["runtime_load"]["process_rss_mib_peak"] > 0.0
    assert child.metadata.curriculum_stage_id == "post_r4_pi_d_masked_ppo"
    assert child.metadata.parent_checkpoint_sha256 == parent_sha
    assert child.metadata.metadata["ppo_update_index"] == 0
    with pytest.raises(
        SchemaValidationError, match="behavior replay|already consumed"
    ):
        train_order9_ppo_update(
            config,
            stage_id="post_r4_pi_d_masked_ppo",
            rollout_manifest_path=dataset_dir / "manifest.json",
            parent_checkpoint_path=result.checkpoint_path,
            physical_model=model_physics,
            output_dir=tmp_path / "update1",
            git_revision="unit-test",
            update_index=1,
            device="cpu",
        )
