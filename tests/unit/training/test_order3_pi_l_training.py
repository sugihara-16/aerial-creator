from __future__ import annotations

import csv
from dataclasses import replace
import math
from pathlib import Path

import pytest
import torch

from amsrr.controllers.rigid_body_model import RigidBodyControlModelBuilder
from amsrr.morphology.random_connected import (
    RandomConnectedMorphologyDistribution,
    morphology_structural_hash,
)
from amsrr.policies.morphology_conditioned_low_level_policy import (
    MorphologyConditionedActorCritic,
    Order3MorphologyConditionedPolicyConfig,
    load_order3_policy_checkpoint,
    order3_actor_feature_vector,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.order3 import (
    ORDER3_ACTION_SIZE,
    ORDER3_CHECKPOINT_VERSION,
    ORDER3_FALLBACK_VERSION,
    Order3PolicyTransition,
)
from amsrr.schemas.policies import ControllerStatus
from amsrr.schemas.runtime import (
    ModuleRuntimeState,
    RuntimeObservation,
    TaskProgressState,
)
from amsrr.training.order3_dataset import write_order3_dataset
from amsrr.training.order3_pi_l_training import (
    Order3BCTrainingConfig,
    Order3PPOTrainingConfig,
    Order3PiLTrainingConfig,
    compute_order3_gae,
    _indexed_ppo_episodes,
    _prepare_transitions,
    _select_complete_ppo_episodes,
    load_order3_pi_l_training_config,
    train_order3_pi_l,
    train_order3_pi_l_bc,
    train_order3_pi_l_ppo,
)


def test_order3_pi_l_training_config_loads_approved_defaults() -> None:
    training, policy = load_order3_pi_l_training_config()

    assert training.seed == 3011
    assert training.bc.sequence_length == 16
    assert training.bc.burn_in_steps == 4
    assert training.ppo.clip_ratio == pytest.approx(0.20)
    assert training.ppo.gae_lambda == pytest.approx(0.95)
    assert policy.action_size == ORDER3_ACTION_SIZE
    assert policy.max_modules == 8


def test_order3_gae_stops_at_episode_terminal_and_preserves_input_order() -> None:
    physical_model, morphologies = _physical_model_and_morphologies()
    del physical_model
    first = _episode(
        DatasetSplit.TRAIN,
        morphologies[DatasetSplit.TRAIN],
        episode_id="first",
        recurrent_width=16,
        rewards=(0.0, 0.0, 1.0),
    )
    second = _episode(
        DatasetSplit.TRAIN,
        morphologies[DatasetSplit.TRAIN],
        episode_id="second",
        recurrent_width=16,
        rewards=(0.0, 2.0),
    )
    interleaved = [first[0], second[0], first[1], second[1], first[2]]

    result = compute_order3_gae(interleaved, gamma=1.0, gae_lambda=1.0)

    assert result.advantages == pytest.approx([1.0, 2.0, 1.0, 2.0, 1.0])
    assert result.returns == pytest.approx(result.advantages)


def test_order3_gae_bootstraps_truncation_without_advantage_leakage() -> None:
    physical_model, morphologies = _physical_model_and_morphologies()
    del physical_model
    truncated = _episode(
        DatasetSplit.TRAIN,
        morphologies[DatasetSplit.TRAIN],
        episode_id="time-limit",
        recurrent_width=16,
        rewards=(2.0,),
        truncate_final=True,
        bootstrap_value=3.0,
    )
    terminal = _episode(
        DatasetSplit.TRAIN,
        morphologies[DatasetSplit.TRAIN],
        episode_id="terminal",
        recurrent_width=16,
        rewards=(7.0,),
    )

    result = compute_order3_gae(
        [truncated[0], terminal[0]],
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert result.advantages == pytest.approx([5.0, 7.0])
    assert result.returns == pytest.approx([5.0, 7.0])


def test_order3_ppo_selection_is_seeded_and_module_balanced() -> None:
    physical_model = build_physical_model_from_config(
        "configs/robot/robot_model.yaml"
    )
    distribution = RandomConnectedMorphologyDistribution(physical_model)
    transitions: list[Order3PolicyTransition] = []
    for module_count in (2, 3, 4):
        morphology = distribution.sample(
            seed=500 + module_count,
            module_count=module_count,
        )
        for replicate in range(3):
            transitions.extend(
                _episode(
                    DatasetSplit.TRAIN,
                    morphology,
                    episode_id=f"n{module_count}-r{replicate}",
                    recurrent_width=16,
                    rewards=(0.0, 1.0),
                )
            )
    episodes = _indexed_ppo_episodes(
        _prepare_transitions(transitions, physical_model)
    )
    first = _select_complete_ppo_episodes(
        episodes,
        rollout_step_budget=6,
        seed=11,
    )
    second = _select_complete_ppo_episodes(
        episodes,
        rollout_step_budget=6,
        seed=12,
    )

    def selected_signature(selected):
        return [episode[0][1].record.episode_id for episode in selected]

    assert {
        len(episode[0][1].record.runtime_observation.morphology_graph.modules)
        for episode in first
    } == {2, 3, 4}
    assert selected_signature(first) != selected_signature(second)


def test_order3_training_bc_ppo_artifacts_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    physical_model, morphologies = _physical_model_and_morphologies()
    transitions: list[Order3PolicyTransition] = []
    behavior_hash = "a" * 64
    for split in DatasetSplit:
        transitions.extend(
            _episode(
                split,
                morphologies[split],
                episode_id=f"{split.value}-bc-episode",
                recurrent_width=16,
                rewards=(0.0, 0.1, 0.2, 1.0),
            )
        )
        transitions.extend(
            _episode(
                split,
                morphologies[split],
                episode_id=f"{split.value}-ppo-episode",
                recurrent_width=16,
                rewards=(0.0, 0.1, 0.2, 1.0),
                behavior_policy_kind="order3_checkpoint",
                behavior_policy_version=ORDER3_CHECKPOINT_VERSION,
                behavior_checkpoint_hash=behavior_hash,
                action_semantics="learned_residual",
            )
        )
    dataset = write_order3_dataset(
        transitions,
        output_dir=tmp_path / "dataset",
        pool_hash="unit-pool-hash",
        physical_model_hash=physical_model.stable_hash(),
        config_hash="unit-dataset-config-hash",
    )
    policy_config = Order3MorphologyConditionedPolicyConfig(
        graph_hidden_dim=16,
        graph_message_layers=1,
        recurrent_hidden_dim=16,
        max_modules=8,
        action_size=ORDER3_ACTION_SIZE,
        initial_log_std=-1.5,
    )
    training_config = Order3PiLTrainingConfig(
        seed=77,
        device="cpu",
        artifact_root=str(tmp_path / "order3-artifacts"),
        bc=Order3BCTrainingConfig(
            epochs=8,
            batch_size=3,
            learning_rate=0.01,
            sequence_length=4,
            burn_in_steps=0,
            value_loss_weight=0.0,
            max_grad_norm=0.5,
        ),
        ppo=Order3PPOTrainingConfig(
            updates=1,
            rollout_steps_per_update=64,
            epochs_per_update=1,
            minibatch_size=4,
            learning_rate=0.001,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            value_loss_weight=0.5,
            entropy_weight=0.001,
            max_grad_norm=0.5,
        ),
    )

    result = train_order3_pi_l(
        dataset_path=dataset.manifest_path,
        physical_model=physical_model,
        training_config=training_config,
        policy_config=policy_config,
        output_root=tmp_path / "training-output",
        git_revision="unit-test-revision",
    )

    initial = result.metrics["bc"]["initial"]["train_actor_mse"]
    final = result.metrics["bc"]["final"]["train_actor_mse"]
    assert final < initial
    assert result.metrics["bc"]["zero_teacher_action_count"] == 4
    assert result.metrics["bc"]["reference_teacher_action_count"] == 4
    assert result.metrics["bc"]["held_out_used_for_optimization"] is False
    assert result.metrics["ppo"]["all_losses_finite"] is True
    assert result.metrics["ppo"]["held_out_used_for_optimization"] is False
    assert result.metrics["ppo"]["behavior_policy_version"] == ORDER3_CHECKPOINT_VERSION
    assert result.metrics["ppo"]["behavior_checkpoint_hash"] == behavior_hash
    assert result.metrics["ppo"]["action_semantics"] == "learned_residual"
    assert result.metrics["summary"]["legacy_p4_3_artifact_reused"] is False

    for path in (
        result.bc_checkpoint_path,
        result.ppo_checkpoint_path,
        result.bc_loss_curve_path,
        result.ppo_loss_curve_path,
        result.reward_curve_path,
        result.bc_metrics_path,
        result.ppo_metrics_path,
        result.summary_path,
    ):
        assert Path(path).is_file() and Path(path).stat().st_size > 0
    ppo_rows = list(csv.DictReader(Path(result.ppo_loss_curve_path).open(encoding="utf-8")))
    assert len(ppo_rows) == 1
    assert all(
        math.isfinite(float(value))
        for row in ppo_rows
        for value in row.values()
    )

    bc_checkpoint = load_order3_policy_checkpoint(result.bc_checkpoint_path)
    ppo_checkpoint = load_order3_policy_checkpoint(result.ppo_checkpoint_path)
    assert bc_checkpoint.metadata.training_stage == "bc"
    assert ppo_checkpoint.metadata.training_stage == "ppo"
    assert ppo_checkpoint.metadata.parent_bc_checkpoint_hash == result.bc_checkpoint_sha256
    assert ppo_checkpoint.metadata.actor_uses_privileged_wrench is False
    assert ppo_checkpoint.metadata.metadata["critic_privileged_inputs"] == [
        "privileged_disturbance_body"
    ]


def test_order3_production_staged_bc_then_matching_checkpoint_ppo(tmp_path: Path) -> None:
    physical_model, morphologies = _physical_model_and_morphologies()
    teacher_transitions = [
        transition
        for split in DatasetSplit
        for transition in _episode(
            split,
            morphologies[split],
            episode_id=f"{split.value}-teacher",
            recurrent_width=16,
            rewards=(0.0, 1.0),
        )
    ]
    teacher_dataset = write_order3_dataset(
        teacher_transitions,
        output_dir=tmp_path / "teacher-dataset",
        pool_hash="production-pool-hash",
        physical_model_hash=physical_model.stable_hash(),
        config_hash="teacher-config-hash",
    )
    policy_config = Order3MorphologyConditionedPolicyConfig(
        graph_hidden_dim=16,
        graph_message_layers=1,
        recurrent_hidden_dim=16,
    )
    training_config = Order3PiLTrainingConfig(
        seed=91,
        artifact_root=str(tmp_path / "production-artifacts"),
        bc=Order3BCTrainingConfig(
            epochs=2,
            batch_size=2,
            learning_rate=0.01,
            sequence_length=2,
            burn_in_steps=0,
        ),
        ppo=Order3PPOTrainingConfig(
            updates=1,
            rollout_steps_per_update=8,
            epochs_per_update=1,
            minibatch_size=2,
        ),
    )
    bc_result = train_order3_pi_l_bc(
        dataset_path=teacher_dataset.manifest_path,
        physical_model=physical_model,
        training_config=training_config,
        policy_config=policy_config,
        output_root=tmp_path / "production-bc",
        git_revision="unit-test-revision",
    )
    assert bc_result.metrics["workflow"] == "production_staged"

    rollout_transitions = [
        transition
        for split in DatasetSplit
        for transition in _episode(
            split,
            morphologies[split],
            episode_id=f"{split.value}-rollout",
            recurrent_width=16,
            rewards=(0.0, 1.0),
            behavior_policy_kind="order3_checkpoint",
            behavior_policy_version=ORDER3_CHECKPOINT_VERSION,
            behavior_checkpoint_hash=bc_result.checkpoint_sha256,
            action_semantics="learned_residual",
        )
    ]
    rollout_dataset = write_order3_dataset(
        rollout_transitions,
        output_dir=tmp_path / "rollout-dataset",
        pool_hash="production-pool-hash",
        physical_model_hash=physical_model.stable_hash(),
        config_hash="rollout-config-hash",
    )
    ppo_result = train_order3_pi_l_ppo(
        dataset_path=rollout_dataset.manifest_path,
        parent_bc_checkpoint_path=bc_result.checkpoint_path,
        parent_bc_checkpoint_sha256=bc_result.checkpoint_sha256,
        physical_model=physical_model,
        training_config=training_config,
        output_root=tmp_path / "production-ppo",
        git_revision="unit-test-revision",
    )

    assert ppo_result.metrics["workflow"] == "production_staged"
    assert ppo_result.metrics["behavior_checkpoint_matches_parent_bc"] is True
    assert ppo_result.metrics["rollout_selection"][
        "advantage_normalization_scope"
    ] == "selected_transitions_only"
    assert ppo_result.metrics["parent_bc_checkpoint_sha256"] == bc_result.checkpoint_sha256
    loaded = load_order3_policy_checkpoint(ppo_result.checkpoint_path)
    assert loaded.metadata.parent_bc_checkpoint_hash == bc_result.checkpoint_sha256

    with pytest.raises(SchemaValidationError, match="updates=1"):
        train_order3_pi_l_ppo(
            dataset_path=rollout_dataset.manifest_path,
            parent_bc_checkpoint_path=bc_result.checkpoint_path,
            parent_bc_checkpoint_sha256=bc_result.checkpoint_sha256,
            physical_model=physical_model,
            training_config=replace(
                training_config,
                ppo=replace(training_config.ppo, updates=2),
            ),
            output_root=tmp_path / "invalid-stale-ppo",
            git_revision="unit-test-revision",
        )

    second_rollout_dataset = write_order3_dataset(
        [
            transition
            for split in DatasetSplit
            for transition in _episode(
                split,
                morphologies[split],
                episode_id=f"{split.value}-rollout-update-2",
                recurrent_width=16,
                rewards=(0.0, 1.0),
                behavior_policy_kind="order3_checkpoint",
                behavior_policy_version=ORDER3_CHECKPOINT_VERSION,
                behavior_checkpoint_hash=ppo_result.checkpoint_sha256,
                action_semantics="learned_residual",
            )
        ],
        output_dir=tmp_path / "rollout-dataset-update-2",
        pool_hash="production-pool-hash",
        physical_model_hash=physical_model.stable_hash(),
        config_hash="rollout-update-2-config-hash",
    )
    second_ppo = train_order3_pi_l_ppo(
        dataset_path=second_rollout_dataset.manifest_path,
        parent_bc_checkpoint_path=ppo_result.checkpoint_path,
        parent_bc_checkpoint_sha256=ppo_result.checkpoint_sha256,
        physical_model=physical_model,
        training_config=training_config,
        output_root=tmp_path / "production-ppo-update-2",
        git_revision="unit-test-revision",
    )
    second_loaded = load_order3_policy_checkpoint(second_ppo.checkpoint_path)
    assert second_ppo.metrics["parent_checkpoint_sha256"] == ppo_result.checkpoint_sha256
    assert second_ppo.metrics["parent_checkpoint_training_stage"] == "ppo"
    assert second_ppo.metrics["fresh_online_rollout_update"] is True
    assert second_loaded.metadata.parent_bc_checkpoint_hash == bc_result.checkpoint_sha256
    assert second_loaded.metadata.metadata[
        "immediate_parent_checkpoint_hash"
    ] == ppo_result.checkpoint_sha256


def test_order3_production_ppo_rejects_rollout_from_another_checkpoint(
    tmp_path: Path,
) -> None:
    physical_model, morphologies = _physical_model_and_morphologies()
    teacher_dataset = write_order3_dataset(
        [
            transition
            for split in DatasetSplit
            for transition in _episode(
                split,
                morphologies[split],
                episode_id=f"{split.value}-teacher",
                recurrent_width=16,
                rewards=(1.0,),
            )
        ],
        output_dir=tmp_path / "teacher-dataset",
        pool_hash="production-pool-hash",
        physical_model_hash=physical_model.stable_hash(),
        config_hash="teacher-config-hash",
    )
    training_config = Order3PiLTrainingConfig(
        seed=4,
        artifact_root=str(tmp_path / "artifacts"),
        bc=Order3BCTrainingConfig(epochs=1, sequence_length=1, burn_in_steps=0),
        ppo=Order3PPOTrainingConfig(
            updates=1,
            rollout_steps_per_update=1,
            epochs_per_update=1,
            minibatch_size=1,
        ),
    )
    bc_result = train_order3_pi_l_bc(
        dataset_path=teacher_dataset.manifest_path,
        physical_model=physical_model,
        training_config=training_config,
        policy_config=Order3MorphologyConditionedPolicyConfig(
            graph_hidden_dim=16,
            graph_message_layers=1,
            recurrent_hidden_dim=16,
        ),
        output_root=tmp_path / "bc",
        git_revision="unit-test-revision",
    )
    wrong_hash = "f" * 64
    rollout_dataset = write_order3_dataset(
        [
            transition
            for split in DatasetSplit
            for transition in _episode(
                split,
                morphologies[split],
                episode_id=f"{split.value}-rollout",
                recurrent_width=16,
                rewards=(1.0,),
                behavior_policy_kind="order3_checkpoint",
                behavior_policy_version=ORDER3_CHECKPOINT_VERSION,
                behavior_checkpoint_hash=wrong_hash,
                action_semantics="learned_residual",
            )
        ],
        output_dir=tmp_path / "rollout-dataset",
        pool_hash="production-pool-hash",
        physical_model_hash=physical_model.stable_hash(),
        config_hash="rollout-config-hash",
    )

    with pytest.raises(
        SchemaValidationError,
        match="behavior checkpoint hash does not match",
    ):
        train_order3_pi_l_ppo(
            dataset_path=rollout_dataset.manifest_path,
            parent_bc_checkpoint_path=bc_result.checkpoint_path,
            parent_bc_checkpoint_sha256=bc_result.checkpoint_sha256,
            physical_model=physical_model,
            training_config=training_config,
            output_root=tmp_path / "ppo",
            git_revision="unit-test-revision",
        )


def test_order3_training_rejects_mixed_ppo_behavior_checkpoint_hashes(
    tmp_path: Path,
) -> None:
    physical_model, morphologies = _physical_model_and_morphologies()
    transitions: list[Order3PolicyTransition] = []
    for split in DatasetSplit:
        transitions.extend(
            _episode(
                split,
                morphologies[split],
                episode_id=f"{split.value}-bc",
                recurrent_width=16,
                rewards=(1.0,),
            )
        )
        transitions.extend(
            _episode(
                split,
                morphologies[split],
                episode_id=f"{split.value}-ppo",
                recurrent_width=16,
                rewards=(1.0,),
                behavior_policy_kind="order3_checkpoint",
                behavior_policy_version=ORDER3_CHECKPOINT_VERSION,
                behavior_checkpoint_hash=("b" if split == DatasetSplit.HELD_OUT else "a")
                * 64,
                action_semantics="learned_residual",
            )
        )
    dataset = write_order3_dataset(
        transitions,
        output_dir=tmp_path / "mixed-dataset",
        pool_hash="unit-pool-hash",
        physical_model_hash=physical_model.stable_hash(),
        config_hash="unit-dataset-config-hash",
    )

    with pytest.raises(
        SchemaValidationError,
        match="share one behavior policy version/checkpoint hash",
    ):
        train_order3_pi_l(
            dataset_path=dataset.manifest_path,
            physical_model=physical_model,
            training_config=Order3PiLTrainingConfig(
                seed=2,
                artifact_root=str(tmp_path / "artifacts"),
                bc=Order3BCTrainingConfig(epochs=1, sequence_length=1, burn_in_steps=0),
                ppo=Order3PPOTrainingConfig(
                    updates=1,
                    rollout_steps_per_update=1,
                    epochs_per_update=1,
                    minibatch_size=1,
                ),
            ),
            policy_config=Order3MorphologyConditionedPolicyConfig(
                graph_hidden_dim=16,
                graph_message_layers=1,
                recurrent_hidden_dim=16,
            ),
            output_root=tmp_path / "training-output",
            git_revision="unit-test-revision",
        )


def test_order3_actor_is_independent_of_critic_privileged_disturbance() -> None:
    physical_model, morphologies = _physical_model_and_morphologies()
    record = _episode(
        DatasetSplit.TRAIN,
        morphologies[DatasetSplit.TRAIN],
        episode_id="privacy",
        recurrent_width=16,
        rewards=(1.0,),
    )[0]
    config = Order3MorphologyConditionedPolicyConfig(
        graph_hidden_dim=16,
        graph_message_layers=1,
        recurrent_hidden_dim=16,
    )
    torch.manual_seed(5)
    model = MorphologyConditionedActorCritic(config)
    control_model = RigidBodyControlModelBuilder().build(
        record.runtime_observation.morphology_graph,
        physical_model,
        record.runtime_observation,
    )
    features = torch.tensor(
        [
            order3_actor_feature_vector(
                record.runtime_observation,
                control_model,
                target_pose_world=record.target_pose_world,
                target_twist=record.target_twist,
            )
        ],
        dtype=torch.float32,
    )
    privileged = torch.tensor(
        [[1.0, -2.0, 3.0, -4.0, 5.0, -6.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    common = (
        [record.runtime_observation.morphology_graph],
        [record.runtime_observation],
        features,
        torch.tensor([record.previous_action], dtype=torch.float32),
        torch.tensor([record.recurrent_state_in], dtype=torch.float32),
    )

    step = model.step(
        *common,
        privileged_disturbance_body=privileged,
        deterministic=True,
    )
    actor_gradient = torch.autograd.grad(
        step.action_mean.sum(),
        privileged,
        retain_graph=True,
        allow_unused=True,
    )[0]
    critic_gradient = torch.autograd.grad(step.value.sum(), privileged)[0]
    zero_privileged = model.step(
        *common,
        privileged_disturbance_body=torch.zeros_like(privileged),
        deterministic=True,
    )

    assert actor_gradient is None or torch.count_nonzero(actor_gradient).item() == 0
    assert torch.count_nonzero(critic_gradient).item() > 0
    assert torch.equal(step.action_mean, zero_privileged.action_mean)
    assert torch.equal(step.recurrent_state, zero_privileged.recurrent_state)
    assert not torch.equal(step.value, zero_privileged.value)


def _physical_model_and_morphologies():
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    distribution = RandomConnectedMorphologyDistribution(physical_model)
    morphologies = {
        DatasetSplit.TRAIN: distribution.sample(seed=1202, module_count=2),
        DatasetSplit.VALIDATION: distribution.sample(seed=1303, module_count=3),
        DatasetSplit.HELD_OUT: distribution.sample(seed=1404, module_count=4),
    }
    return physical_model, morphologies


def _episode(
    split: DatasetSplit,
    morphology: MorphologyGraph,
    *,
    episode_id: str,
    recurrent_width: int,
    rewards: tuple[float, ...],
    truncate_final: bool = False,
    bootstrap_value: float | None = None,
    behavior_policy_kind: str = "deterministic_v2_teacher",
    behavior_policy_version: str = ORDER3_FALLBACK_VERSION,
    behavior_checkpoint_hash: str | None = None,
    action_semantics: str = "reference_hold",
) -> list[Order3PolicyTransition]:
    output: list[Order3PolicyTransition] = []
    for step_index, reward in enumerate(rewards):
        time_s = 0.02 * step_index
        status = ControllerStatus(
            status="ok",
            qp_feasible=True,
            active_mode="rigid_body_qp",
            metrics={"allocation_residual_norm": 0.0},
        )
        module_states = []
        for module in morphology.modules:
            pose = list(module.pose_in_design_frame)
            pose[2] += 1.0
            module_states.append(
                ModuleRuntimeState(
                    module_id=module.module_id,
                    pose_world=tuple(pose),
                    twist_world=[0.01 * step_index, 0.0, 0.0, 0.0, 0.0, 0.0],
                    joint_positions={},
                    joint_velocities={},
                )
            )
        is_final = step_index == len(rewards) - 1
        terminal = is_final and not truncate_final
        observation = RuntimeObservation(
            time_s=time_s,
            morphology_graph=morphology,
            module_states=module_states,
            object_states=[],
            contact_states=[],
            controller_status=status,
            task_progress=TaskProgressState(
                phase_label="hover",
                progress_ratio=step_index / max(1, len(rewards) - 1),
                success=terminal,
                metrics={"tracking_error_m": 0.05},
            ),
        )
        output.append(
            Order3PolicyTransition(
                episode_id=episode_id,
                split=split,
                graph_id=morphology.graph_id,
                structural_hash=morphology_structural_hash(morphology),
                step_index=step_index,
                time_s=time_s,
                runtime_observation=observation,
                target_pose_world=(0.1, 0.0, 1.2, 0.0, 0.0, 0.0, 1.0),
                target_twist=[0.0] * 6,
                previous_action=[0.0] * ORDER3_ACTION_SIZE,
                action=[0.0] * ORDER3_ACTION_SIZE,
                recurrent_state_in=[0.0] * recurrent_width,
                old_log_prob=-1.0,
                old_value=0.0,
                reward=reward,
                terminal=terminal,
                truncated=is_final and truncate_final,
                bootstrap_value=bootstrap_value if is_final and truncate_final else None,
                behavior_policy_kind=behavior_policy_kind,
                behavior_policy_version=behavior_policy_version,
                behavior_checkpoint_hash=behavior_checkpoint_hash,
                action_semantics=action_semantics,
                policy_applied=step_index > 0,
                privileged_disturbance_body=[0.1 * step_index, 0.0, 0.0, 0.0, 0.0, 0.0],
                metrics={"isaac_backed": 1.0, "position_error_m": 0.05},
            )
        )
    return output
