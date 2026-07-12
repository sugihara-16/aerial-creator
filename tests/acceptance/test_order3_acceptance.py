from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

import pytest
import torch

from amsrr.acceptance.order3_acceptance import (
    ORDER3_ACCEPTANCE_ARTIFACT_VERSION,
    Order3AcceptanceArtifactMetadata,
    run_order3_acceptance,
    run_order3_acceptance_from_paths,
)
from amsrr.morphology.random_connected import (
    RandomConnectedMorphologyDistribution,
    morphology_structural_hash,
)
from amsrr.policies.morphology_conditioned_low_level_policy import (
    MorphologyConditionedActorCritic,
    Order3MorphologyConditionedPolicyConfig,
    order3_actor_feature_schema_hash,
    order3_graph_feature_schema_hash,
    save_order3_policy_checkpoint,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.feasibility import FeasibilityResult
from amsrr.schemas.order3 import (
    ORDER3_ACTION_NAMES,
    ORDER3_CHECKPOINT_VERSION,
    ORDER3_DATASET_VERSION,
    ORDER3_ENCODER_VERSION,
    ORDER3_FALLBACK_VERSION,
    ORDER3_POLICY_ARCHITECTURE_VERSION,
    ORDER3_POLICY_FAMILY,
    ORDER3_POOL_VERSION,
    ORDER3_TENSORIZER_VERSION,
    Order3DatasetManifest,
    Order3MorphologyPoolEntry,
    Order3MorphologyPoolManifest,
    Order3PolicyCheckpointMetadata,
)
from amsrr.schemas.order3_rollout_condition import (
    ORDER3_ROLLOUT_CONDITION_VERSION,
    Order3RolloutCondition,
    build_order3_rollout_condition,
)
from amsrr.schemas.policies import POLICY_COMMAND_CONTRACT_CENTROIDAL
from amsrr.simulation.order3_rollout_condition import (
    order3_terminal_evidence_start_s,
    order3_tracking_window_start_s,
)
from amsrr.training.order3_free_flight import (
    ORDER3_FREE_FLIGHT_VERSION,
    Order3EvaluationEpisode,
    Order3TaskMode,
    Order3TerminalMetrics,
    order3_terminal_metrics_success,
    recommended_order3_morphology_split_counts,
)
from amsrr.utils.hashing import hash_file, stable_hash


_UNIT_PHYSICAL_MODEL_HASH = build_physical_model_from_config().stable_hash()


@dataclass(frozen=True)
class _Evidence:
    pool: Order3MorphologyPoolManifest
    pool_path: Path
    dataset: Order3DatasetManifest
    dataset_path: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    artifact: Order3AcceptanceArtifactMetadata
    artifact_path: Path
    episodes: list[Order3EvaluationEpisode]
    episodes_path: Path


@pytest.fixture(scope="module")
def passing_evidence(tmp_path_factory: pytest.TempPathFactory) -> _Evidence:
    return _write_passing_evidence(tmp_path_factory.mktemp("order3-acceptance"))


def test_order3_acceptance_passes_bound_free_flight_evidence(
    passing_evidence: _Evidence,
) -> None:
    report = run_order3_acceptance_from_paths(
        pool_manifest_path=passing_evidence.pool_path,
        dataset_manifest_path=passing_evidence.dataset_path,
        checkpoint_path=passing_evidence.checkpoint_path,
        expected_checkpoint_sha256=passing_evidence.checkpoint_sha256,
        episodes_path=passing_evidence.episodes_path,
        artifact_metadata_path=passing_evidence.artifact_path,
    )

    assert report.completion_passed is True
    assert report.pass_summary.completion_passed is True
    assert report.failures == []
    assert report.aggregate_held_out_success_rate == 1.0
    assert set(report.per_module_success_rates) == {
        str(module_count) for module_count in range(2, 9)
    }
    assert all(value == 1.0 for value in report.per_module_success_rates.values())
    assert report.nominal_relative_degradation == pytest.approx(0.0)
    assert report.randomized_relative_improvement == pytest.approx(0.20)
    assert report.id_fallback_rate == 0.0
    assert report.ood_episode_count == 1
    assert report.ood_fallback_rate == 1.0
    assert report.object_task_claim is False
    assert report.contact_task_claim is False
    assert report.p4_full_completion_claim is False


def test_order3_acceptance_recomputes_exact_pool_quota(
    passing_evidence: _Evidence,
) -> None:
    removed = next(
        entry
        for entry in passing_evidence.pool.entries
        if entry.module_count == 8 and entry.split == DatasetSplit.TRAIN
    )
    entries = [entry for entry in passing_evidence.pool.entries if entry is not removed]
    changed_pool = replace(
        passing_evidence.pool,
        entries=entries,
        split_counts={
            split.value: sum(entry.split == split for entry in entries)
            for split in DatasetSplit
        },
        module_count_counts={
            str(module_count): sum(entry.module_count == module_count for entry in entries)
            for module_count in range(2, 9)
        },
    )

    report = _run(passing_evidence, pool_manifest=changed_pool)

    assert report.pass_summary.pool_passed is False
    assert "pool_quota_mismatch:train:n8" in report.failures


def test_order3_acceptance_recomputes_pool_hash_from_graph(
    passing_evidence: _Evidence,
) -> None:
    entries = list(passing_evidence.pool.entries)
    entries[0] = replace(entries[0], structural_hash="0" * 64)
    changed_pool = replace(passing_evidence.pool, entries=entries)

    report = _run(passing_evidence, pool_manifest=changed_pool)

    assert report.pass_summary.pool_passed is False
    assert "pool_structural_hash_mismatch" in report.failures


def test_order3_acceptance_recomputes_dataset_pool_binding(
    passing_evidence: _Evidence,
) -> None:
    hashes = {
        split: list(values)
        for split, values in passing_evidence.dataset.morphology_hashes.items()
    }
    hashes[DatasetSplit.HELD_OUT.value] = hashes[DatasetSplit.HELD_OUT.value][1:]
    changed_dataset = replace(passing_evidence.dataset, morphology_hashes=hashes)

    report = _run(passing_evidence, dataset_manifest=changed_dataset)

    assert report.pass_summary.dataset_passed is False
    assert "dataset_pool_hash_set_mismatch:held_out" in report.failures


def test_order3_acceptance_rejects_dataset_scope_claim(
    passing_evidence: _Evidence,
) -> None:
    metadata = dict(passing_evidence.dataset.metadata)
    metadata["p4_full_completion_claim"] = True
    changed_dataset = replace(passing_evidence.dataset, metadata=metadata)

    report = _run(passing_evidence, dataset_manifest=changed_dataset)

    assert report.pass_summary.dataset_passed is False
    assert "dataset_scope_claim_invalid" in report.failures
    assert report.pass_summary.no_scope_mislabeling_passed is False


def test_order3_acceptance_rejects_tampered_checkpoint_even_with_updated_claim(
    passing_evidence: _Evidence,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "tampered.pt"
    payload = torch.load(
        passing_evidence.checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    payload["metadata"]["actor_uses_privileged_wrench"] = True
    torch.save(payload, destination)
    digest = hash_file(destination)
    artifact = replace(
        passing_evidence.artifact,
        checkpoint_sha256=digest,
        actor_uses_privileged_wrench=True,
    )

    report = _run(
        passing_evidence,
        checkpoint_path=destination,
        checkpoint_sha256=digest,
        artifact_metadata=artifact,
    )

    assert report.pass_summary.checkpoint_passed is False
    assert "checkpoint_invalid_or_sha256_mismatch" in report.failures
    assert report.pass_summary.artifact_metadata_passed is False


def test_order3_acceptance_requires_raw_terminal_metrics(
    passing_evidence: _Evidence,
) -> None:
    episodes = list(passing_evidence.episodes)
    episodes[0] = replace(episodes[0], terminal_metrics=None)

    report = _run(passing_evidence, episodes=episodes)

    assert report.pass_summary.held_out_performance_passed is False
    assert "evaluation_terminal_metrics_missing" in report.failures


def test_order3_acceptance_checks_per_module_success_not_only_aggregate(
    passing_evidence: _Evidence,
) -> None:
    episodes = list(passing_evidence.episodes)
    failing_metrics = _terminal_metrics(success=False)
    failing_indices = [
        index
        for index, episode in enumerate(episodes)
        if episode.module_count == 2
        and episode.structural_hash in _held_out_hashes(passing_evidence)
    ][:2]
    for index in failing_indices:
        episodes[index] = replace(
            episodes[index],
            success=False,
            terminal_metrics=failing_metrics,
        )

    report = _run(passing_evidence, episodes=episodes)

    assert report.aggregate_held_out_success_rate >= 0.95
    assert report.per_module_success_rates["2"] < 0.90
    assert report.pass_summary.held_out_performance_passed is False
    assert "evaluation_held_out_success_threshold_failed" in report.failures


def test_order3_acceptance_rejects_any_safety_terminal(
    passing_evidence: _Evidence,
) -> None:
    episodes = list(passing_evidence.episodes)
    episodes[0] = replace(episodes[0], qp_infeasible=True)

    report = _run(passing_evidence, episodes=episodes)

    assert report.safety_failure_episode_count == 1
    assert report.pass_summary.safety_passed is False
    assert "evaluation_safety_terminal_present" in report.failures


def test_order3_acceptance_enforces_nominal_baseline_degradation(
    passing_evidence: _Evidence,
) -> None:
    episodes = [
        replace(episode, tracking_cost=1.06)
        if episode.structural_hash in _held_out_hashes(passing_evidence)
        and not episode.randomized
        else episode
        for episode in passing_evidence.episodes
    ]

    report = _run(passing_evidence, episodes=episodes)

    assert report.nominal_relative_degradation == pytest.approx(0.06)
    assert report.pass_summary.nominal_baseline_passed is False


def test_order3_acceptance_allows_randomized_success_gain_alternative(
    passing_evidence: _Evidence,
) -> None:
    episodes = [
        _synchronize_episode_reports(
            replace(
                episode,
                tracking_cost=0.90,
                deterministic_baseline_terminal_metrics=_terminal_metrics(success=False),
            ),
            suffix="success-gain",
        )
        if episode.structural_hash in _held_out_hashes(passing_evidence)
        and episode.randomized
        else episode
        for episode in passing_evidence.episodes
    ]

    report = _run(passing_evidence, episodes=episodes)

    assert report.randomized_relative_improvement == pytest.approx(0.10)
    assert report.randomized_success_gain == pytest.approx(1.0)
    assert report.pass_summary.randomized_robustness_passed is True
    assert report.pass_summary.completion_passed is True


def test_order3_acceptance_randomized_gate_fails_closed_without_either_branch(
    passing_evidence: _Evidence,
) -> None:
    episodes = [
        replace(
            episode,
            tracking_cost=0.90,
            deterministic_baseline_terminal_metrics=None,
        )
        if episode.structural_hash in _held_out_hashes(passing_evidence)
        and episode.randomized
        else episode
        for episode in passing_evidence.episodes
    ]

    report = _run(passing_evidence, episodes=episodes)

    assert report.randomized_relative_improvement == pytest.approx(0.10)
    assert report.randomized_success_gain is None
    assert report.pass_summary.randomized_robustness_passed is False


def test_order3_acceptance_enforces_id_and_ood_fallback_separately(
    passing_evidence: _Evidence,
) -> None:
    id_episodes = list(passing_evidence.episodes)
    id_episodes[0] = replace(id_episodes[0], fallback_used=True)
    id_report = _run(passing_evidence, episodes=id_episodes)

    assert id_report.id_fallback_rate > 0.01
    assert id_report.pass_summary.id_fallback_passed is False

    no_ood = [
        episode
        for episode in passing_evidence.episodes
        if episode.structural_hash in _held_out_hashes(passing_evidence)
    ]
    ood_report = _run(passing_evidence, episodes=no_ood)

    assert ood_report.ood_episode_count == 0
    assert ood_report.pass_summary.ood_fallback_passed is False
    assert "evaluation_ood_fallback_evidence_missing" in ood_report.failures


def test_order3_acceptance_requires_full_task_condition_matrix(
    passing_evidence: _Evidence,
) -> None:
    episodes = list(passing_evidence.episodes)
    episodes.pop(0)

    report = _run(passing_evidence, episodes=episodes)

    assert report.pass_summary.held_out_coverage_passed is False
    assert "evaluation_task_mode_randomization_matrix_incomplete" in report.failures


def test_order3_acceptance_rejects_tampered_raw_report_binding(
    passing_evidence: _Evidence,
) -> None:
    episodes = list(passing_evidence.episodes)
    episodes[0] = replace(episodes[0], learned_report_sha256="0" * 64)

    report = _run(passing_evidence, episodes=episodes)

    assert report.pass_summary.held_out_coverage_passed is False
    assert any("learned:missing_or_hash_mismatch" in item for item in report.failures)


@pytest.mark.parametrize(
    ("field_name", "tampered_value"),
    (
        ("applied_mass_scale", 1.2),
        ("initial_state_applied", False),
    ),
)
def test_order3_acceptance_rejects_tampered_condition_realization(
    passing_evidence: _Evidence,
    field_name: str,
    tampered_value,
) -> None:
    episodes = list(passing_evidence.episodes)
    source_episode = episodes[0]
    source_path = Path(str(source_episode.learned_report_path))
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    payload["order3_condition_realization"][field_name] = tampered_value
    tampered_path = source_path.with_name(
        f"{source_path.stem}-tampered-{field_name}.json"
    )
    tampered_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    episodes[0] = replace(
        source_episode,
        learned_report_path=str(tampered_path),
        learned_report_sha256=hash_file(tampered_path),
    )

    report = _run(passing_evidence, episodes=episodes)

    assert report.pass_summary.held_out_coverage_passed is False
    assert any(
        "learned_condition_realization_invalid" in item
        for item in report.failures
    )


def test_order3_acceptance_binds_ood_safety_before_fallback_return(
    passing_evidence: _Evidence,
) -> None:
    episodes = list(passing_evidence.episodes)
    source_episode = episodes[-1]
    source_path = Path(str(source_episode.learned_report_path))
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    payload["order3_qp_infeasible"] = True
    tampered_path = source_path.with_name(f"{source_path.stem}-unsafe.json")
    tampered_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    episodes[-1] = replace(
        source_episode,
        learned_report_path=str(tampered_path),
        learned_report_sha256=hash_file(tampered_path),
    )

    report = _run(passing_evidence, episodes=episodes)

    assert report.pass_summary.held_out_coverage_passed is False
    assert any("learned_qp_infeasible_mismatch" in item for item in report.failures)


def test_order3_acceptance_requires_structural_hash_ood_reason(
    passing_evidence: _Evidence,
) -> None:
    episodes = list(passing_evidence.episodes)
    episodes[-1] = replace(episodes[-1], fallback_reason="actor_feature_ood")

    report = _run(passing_evidence, episodes=episodes)

    assert report.pass_summary.ood_fallback_passed is False
    assert "evaluation_ood_fallback_evidence_missing" in report.failures


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("architecture_version", "flat_mlp_v1"),
        ("graph_encoder_used", False),
        ("recurrent_gru_used", False),
        ("actor_uses_privileged_wrench", True),
        ("object_task_claim", True),
        ("contact_task_claim", True),
        ("p4_full_completion_claim", True),
    ),
)
def test_order3_acceptance_requires_explicit_graph_gru_and_scope_metadata(
    passing_evidence: _Evidence,
    field_name: str,
    value,
) -> None:
    artifact = replace(passing_evidence.artifact, **{field_name: value})

    report = _run(passing_evidence, artifact_metadata=artifact)

    assert report.pass_summary.artifact_metadata_passed is False
    assert f"artifact_metadata_mismatch:{field_name}" in report.failures
    if field_name.endswith("claim"):
        assert report.pass_summary.no_scope_mislabeling_passed is False


def _run(
    evidence: _Evidence,
    *,
    pool_manifest=None,
    dataset_manifest=None,
    checkpoint_path: Path | None = None,
    checkpoint_sha256: str | None = None,
    episodes: list[Order3EvaluationEpisode] | None = None,
    artifact_metadata: Order3AcceptanceArtifactMetadata | None = None,
):
    resolved_episodes = episodes or evidence.episodes
    resolved_artifact = artifact_metadata or evidence.artifact
    if episodes is not None and artifact_metadata is None:
        resolved_artifact = replace(
            resolved_artifact,
            evaluation_episode_set_hash=stable_hash(
                [episode.to_dict() for episode in resolved_episodes]
            ),
        )
    return run_order3_acceptance(
        pool_manifest=pool_manifest or evidence.pool_path,
        dataset_manifest=dataset_manifest or evidence.dataset_path,
        checkpoint_path=checkpoint_path or evidence.checkpoint_path,
        expected_checkpoint_sha256=checkpoint_sha256 or evidence.checkpoint_sha256,
        episodes=resolved_episodes,
        artifact_metadata=resolved_artifact,
    )


def _write_passing_evidence(root: Path) -> _Evidence:
    pool = _pool_manifest()
    pool_path = root / "pool.json"
    pool_path.write_text(pool.to_json(indent=2) + "\n", encoding="utf-8")
    dataset = _dataset_manifest(pool)
    dataset_path = root / "dataset_manifest.json"
    dataset_path.write_text(dataset.to_json(indent=2) + "\n", encoding="utf-8")
    checkpoint_path = root / "checkpoint.pt"
    checkpoint_sha256 = _write_checkpoint(checkpoint_path, dataset, dataset_path)
    episodes = _episodes(
        pool,
        report_root=root / "reports",
        checkpoint_sha256=checkpoint_sha256,
    )
    artifact = _artifact_metadata(
        checkpoint_sha256=checkpoint_sha256,
        dataset_sha256=hash_file(dataset_path),
        pool_hash=pool.stable_hash(),
        episodes=episodes,
    )
    artifact_path = root / "evaluation_metadata.json"
    artifact_path.write_text(artifact.to_json(indent=2) + "\n", encoding="utf-8")
    episodes_path = root / "episodes.json"
    episodes_path.write_text(
        json.dumps(
            {"episodes": [episode.to_dict() for episode in episodes]},
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return _Evidence(
        pool=pool,
        pool_path=pool_path,
        dataset=dataset,
        dataset_path=dataset_path,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        artifact=artifact,
        artifact_path=artifact_path,
        episodes=episodes,
        episodes_path=episodes_path,
    )


def _pool_manifest() -> Order3MorphologyPoolManifest:
    physical_model = build_physical_model_from_config()
    distribution = RandomConnectedMorphologyDistribution(physical_model)
    feasibility = FeasibilityResult(
        feasible=True,
        hard_violations=[],
        soft_violations=[],
        margins={},
        proxy_scores={},
        checker_version="order3-acceptance-unit-feasibility-v1",
    )
    entries: list[Order3MorphologyPoolEntry] = []
    seed = 1
    for module_count in range(2, 9):
        template = distribution.sample(seed=module_count * 100, module_count=module_count)
        for split in DatasetSplit:
            count = recommended_order3_morphology_split_counts(module_count)[split]
            for split_index in range(count):
                modules = list(template.modules)
                modules[0] = replace(
                    modules[0],
                    role_id=f"base-{module_count}-{split.value}-{split_index}",
                )
                graph = replace(
                    template,
                    graph_id=f"unit:{module_count}:{split.value}:{split_index}",
                    modules=modules,
                )
                structural_hash = morphology_structural_hash(graph)
                entries.append(
                    Order3MorphologyPoolEntry(
                        split=split,
                        module_count=module_count,
                        structural_hash=structural_hash,
                        requested_seed=seed,
                        accepted_proposal_seed=seed,
                        morphology_graph=graph,
                        feasibility_result=feasibility,
                    )
                )
                seed += 1
    entries.sort(key=lambda item: (item.module_count, item.split.value, item.structural_hash))
    return Order3MorphologyPoolManifest(
        pool_version=ORDER3_POOL_VERSION,
        master_seed=123,
        physical_model_hash=physical_model.stable_hash(),
        config_hash="1" * 64,
        entries=entries,
        split_counts={
            split.value: sum(entry.split == split for entry in entries)
            for split in DatasetSplit
        },
        module_count_counts={
            str(module_count): sum(entry.module_count == module_count for entry in entries)
            for module_count in range(2, 9)
        },
        metadata={
            "split_unit": "canonical_structural_hash",
            "object_task_claim": False,
            "contact_task_claim": False,
            "p4_full_completion_claim": False,
        },
    )


def _dataset_manifest(pool: Order3MorphologyPoolManifest) -> Order3DatasetManifest:
    morphology_hashes = {
        split.value: sorted(
            entry.structural_hash for entry in pool.entries if entry.split == split
        )
        for split in DatasetSplit
    }
    shard_paths = {
        split.value: [f"transitions_{split.value}_00000.jsonl"]
        for split in DatasetSplit
    }
    return Order3DatasetManifest(
        dataset_version=ORDER3_DATASET_VERSION,
        policy_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
        policy_family=ORDER3_POLICY_FAMILY,
        pool_hash=pool.stable_hash(),
        physical_model_hash=pool.physical_model_hash,
        config_hash="2" * 64,
        transition_shards=shard_paths,
        transition_shard_hashes={
            path: "3" * 64 for paths in shard_paths.values() for path in paths
        },
        transition_counts={split.value: 1 for split in DatasetSplit},
        morphology_hashes=morphology_hashes,
        real_isaac_episode_counts={split.value: 1 for split in DatasetSplit},
        actor_privileged_wrench_inputs=False,
        metadata={
            "actor_observation_contract": "deployable_only",
            "object_task_claim": False,
            "contact_task_claim": False,
            "p4_full_completion_claim": False,
        },
    )


def _write_checkpoint(
    path: Path,
    dataset: Order3DatasetManifest,
    dataset_path: Path,
) -> str:
    config = Order3MorphologyConditionedPolicyConfig(
        graph_hidden_dim=8,
        graph_message_layers=1,
        recurrent_hidden_dim=8,
    )
    model = MorphologyConditionedActorCritic(config)
    metadata = Order3PolicyCheckpointMetadata(
        checkpoint_version=ORDER3_CHECKPOINT_VERSION,
        policy_family=ORDER3_POLICY_FAMILY,
        policy_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
        architecture_version=ORDER3_POLICY_ARCHITECTURE_VERSION,
        tensorizer_version=ORDER3_TENSORIZER_VERSION,
        encoder_version=ORDER3_ENCODER_VERSION,
        training_stage="ppo",
        action_names=list(ORDER3_ACTION_NAMES),
        actor_feature_schema_hash=order3_actor_feature_schema_hash(),
        graph_feature_schema_hash=order3_graph_feature_schema_hash(),
        config_hash=config.stable_hash(),
        pool_hash=dataset.pool_hash,
        dataset_hash=hash_file(dataset_path),
        physical_model_hash=dataset.physical_model_hash,
        urdf_hash="4" * 64,
        controller_contract_hash="5" * 64,
        fallback_version=ORDER3_FALLBACK_VERSION,
        fallback_config_hash="6" * 64,
        seed=123,
        git_revision="order3-acceptance-unit",
        actor_uses_privileged_wrench=False,
        outputs_contact_wrench=False,
        outputs_internal_wrench=False,
        outputs_vectoring_joint_targets=False,
        parent_bc_checkpoint_hash="7" * 64,
        metadata={
            "actor_privileged_inputs": [],
            "critic_privileged_inputs": ["privileged_disturbance_body"],
            "morphology_hashes": dataset.morphology_hashes,
            "object_task_claim": False,
            "contact_task_claim": False,
            "p4_full_completion_claim": False,
        },
    )
    return save_order3_policy_checkpoint(path, model=model, metadata=metadata)


def _artifact_metadata(
    *,
    checkpoint_sha256: str,
    dataset_sha256: str,
    pool_hash: str,
    episodes: list[Order3EvaluationEpisode],
) -> Order3AcceptanceArtifactMetadata:
    return Order3AcceptanceArtifactMetadata(
        artifact_version=ORDER3_ACCEPTANCE_ARTIFACT_VERSION,
        evaluation_scope_version=ORDER3_FREE_FLIGHT_VERSION,
        evaluation_source="real_isaac_paired_learned_and_deterministic_v2",
        checkpoint_sha256=checkpoint_sha256,
        dataset_manifest_sha256=dataset_sha256,
        policy_family=ORDER3_POLICY_FAMILY,
        policy_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
        architecture_version=ORDER3_POLICY_ARCHITECTURE_VERSION,
        tensorizer_version=ORDER3_TENSORIZER_VERSION,
        encoder_version=ORDER3_ENCODER_VERSION,
        graph_encoder_used=True,
        recurrent_gru_used=True,
        actor_uses_privileged_wrench=False,
        deterministic_fallback_available=True,
        pool_hash=pool_hash,
        evaluation_episode_set_hash=stable_hash(
            [episode.to_dict() for episode in episodes]
        ),
        rollout_condition_version=ORDER3_ROLLOUT_CONDITION_VERSION,
        raw_report_hashes_bound=True,
        paired_deterministic_baseline=True,
        required_task_modes=sorted(mode.value for mode in Order3TaskMode),
        object_task_claim=False,
        contact_task_claim=False,
        p4_full_completion_claim=False,
    )


def _episodes(
    pool: Order3MorphologyPoolManifest,
    *,
    report_root: Path,
    checkpoint_sha256: str,
) -> list[Order3EvaluationEpisode]:
    passing = _terminal_metrics(success=True)
    output: list[Order3EvaluationEpisode] = []
    held_out = [entry for entry in pool.entries if entry.split == DatasetSplit.HELD_OUT]
    report_root.mkdir(parents=True, exist_ok=True)
    seed = 1000
    for entry in held_out:
        for task_mode in Order3TaskMode:
            for randomized in (False, True):
                seed += 1
                condition = build_order3_rollout_condition(
                    stage_id="3e_held_out_evaluation",
                    task_mode=task_mode.value,
                    seed=seed,
                    initial_position_offset_world=(
                        (0.02, -0.01, 0.01) if randomized else (0.0, 0.0, 0.0)
                    ),
                    initial_orientation_rpy_rad=(
                        (0.02, -0.01, 0.03) if randomized else (0.0, 0.0, 0.0)
                    ),
                    initial_linear_velocity_world=(
                        (0.01, 0.0, -0.01) if randomized else (0.0, 0.0, 0.0)
                    ),
                    waypoint_position_offset_world=(
                        (0.15, -0.05, 0.05)
                        if task_mode == Order3TaskMode.WAYPOINT
                        else (0.0, 0.0, 0.0)
                    ),
                    waypoint_orientation_rpy_rad=(
                        (0.05, -0.04, 0.10)
                        if task_mode == Order3TaskMode.WAYPOINT
                        else (0.0, 0.0, 0.0)
                    ),
                    external_wrench_body=(
                        (1.0, -0.5, 0.2, 0.05, 0.0, -0.02)
                        if randomized
                        else (0.0,) * 6
                    ),
                    disturbance_start_s=0.5,
                    disturbance_duration_s=1.0 if randomized else 0.0,
                    mass_scale=1.05 if randomized else 1.0,
                    inertia_scale=0.97 if randomized else 1.0,
                    thrust_scale=0.95 if randomized else 1.0,
                )
                label = "randomized" if randomized else "nominal"
                stem = f"{entry.structural_hash[:12]}-{task_mode.value}-{label}"
                learned_path = report_root / f"{stem}-learned.json"
                baseline_path = report_root / f"{stem}-baseline.json"
                learned_cost = 0.8 if randomized else 1.0
                _write_raw_evaluation_report(
                    learned_path,
                    condition=condition,
                    structural_hash=entry.structural_hash,
                    checkpoint_sha256=checkpoint_sha256,
                    learned=True,
                    terminal_metrics=passing,
                    tracking_cost=learned_cost,
                    success=True,
                )
                _write_raw_evaluation_report(
                    baseline_path,
                    condition=condition,
                    structural_hash=entry.structural_hash,
                    checkpoint_sha256=checkpoint_sha256,
                    learned=False,
                    terminal_metrics=passing,
                    tracking_cost=1.0,
                    success=True,
                )
                output.append(
                    Order3EvaluationEpisode(
                        episode_id=stem,
                        structural_hash=entry.structural_hash,
                        module_count=entry.module_count,
                        split=DatasetSplit.HELD_OUT,
                        success=True,
                        tracking_cost=learned_cost,
                        deterministic_baseline_tracking_cost=1.0,
                        randomized=randomized,
                        task_mode=task_mode,
                        terminal_metrics=passing,
                        deterministic_baseline_terminal_metrics=passing,
                        condition_hash=condition.condition_hash,
                        condition_seed=condition.seed,
                        checkpoint_sha256=checkpoint_sha256,
                        learned_report_path=str(learned_path),
                        learned_report_sha256=hash_file(learned_path),
                        deterministic_baseline_report_path=str(baseline_path),
                        deterministic_baseline_report_sha256=hash_file(baseline_path),
                        isaac_backed=True,
                    )
                )
    ood_condition = build_order3_rollout_condition(
        stage_id="3e_ood_fallback",
        task_mode="hover",
        seed=9999,
    )
    ood_learned_path = report_root / "ood-fallback-learned.json"
    ood_baseline_path = report_root / "ood-fallback-baseline.json"
    _write_raw_evaluation_report(
        ood_learned_path,
        condition=ood_condition,
        structural_hash="f" * 64,
        checkpoint_sha256=checkpoint_sha256,
        learned=True,
        terminal_metrics=_terminal_metrics(success=False),
        tracking_cost=2.0,
        success=False,
        fallback_used=True,
        fallback_reason="structural_hash_ood",
    )
    _write_raw_evaluation_report(
        ood_baseline_path,
        condition=ood_condition,
        structural_hash="f" * 64,
        checkpoint_sha256=checkpoint_sha256,
        learned=False,
        terminal_metrics=_terminal_metrics(success=False),
        tracking_cost=2.0,
        success=False,
    )
    output.append(
        Order3EvaluationEpisode(
            episode_id="ood-fallback",
            structural_hash="f" * 64,
            module_count=8,
            split=DatasetSplit.HELD_OUT,
            success=False,
            tracking_cost=2.0,
            deterministic_baseline_tracking_cost=2.0,
            fallback_used=True,
            fallback_reason="structural_hash_ood",
            condition_hash=ood_condition.condition_hash,
            condition_seed=ood_condition.seed,
            checkpoint_sha256=checkpoint_sha256,
            learned_report_path=str(ood_learned_path),
            learned_report_sha256=hash_file(ood_learned_path),
            deterministic_baseline_report_path=str(ood_baseline_path),
            deterministic_baseline_report_sha256=hash_file(ood_baseline_path),
            isaac_backed=True,
        )
    )
    return output


def _write_raw_evaluation_report(
    path: Path,
    *,
    condition: Order3RolloutCondition,
    structural_hash: str,
    checkpoint_sha256: str,
    learned: bool,
    terminal_metrics: Order3TerminalMetrics,
    tracking_cost: float,
    success: bool,
    fallback_used: bool = False,
    fallback_reason: str | None = None,
) -> None:
    terminal_evidence_start = order3_terminal_evidence_start_s(condition)
    tracking_window_start = order3_tracking_window_start_s(condition)
    report = {
        "isaac_backed": True,
        "order3_report_validation_failures": [],
        "order3_task_mode": condition.task_mode,
        "order3_rollout_task_mode": condition.task_mode,
        "order3_structural_hash": structural_hash,
        "order3_rollout_condition": condition.to_dict(),
        "order3_rollout_condition_hash": condition.condition_hash,
        "order3_rollout_seed_applied": {
            "seed": condition.seed,
            "python_random": True,
            "numpy": True,
            "torch": True,
            "torch_cuda": True,
        },
        "order3_privileged_external_wrench_body": list(
            condition.external_wrench_body
        ),
        "order3_disturbance_start_s": condition.disturbance_start_s,
        "order3_disturbance_duration_s": condition.disturbance_duration_s,
        "order3_condition_realization": {
            "condition_hash": condition.condition_hash,
            "task_mode": condition.task_mode,
            "requested_initial_root_pose_world": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "applied_initial_root_pose_world": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "requested_initial_twist_world": [0.0] * 6,
            "applied_initial_twist_world": [0.0] * 6,
            "requested_mass_scale": condition.mass_scale,
            "applied_mass_scale": condition.mass_scale,
            "requested_inertia_scale": condition.inertia_scale,
            "applied_inertia_scale": condition.inertia_scale,
            "requested_thrust_scale": condition.thrust_scale,
            "applied_thrust_scale": condition.thrust_scale,
            "mass_randomization_applied": True,
            "inertia_randomization_applied": True,
            "thrust_randomization_applied": True,
            "initial_state_applied": True,
            "final_target_pose_world": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "final_target_twist_world": [0.0] * 6,
        },
        "order3_terminal_evidence_start_s": terminal_evidence_start,
        "order3_terminal_evidence_completed": True,
        "order3_tracking_window_start_s": tracking_window_start,
        "order3_tracking_window_end_s": max(
            terminal_evidence_start + condition.hold_s,
            tracking_window_start + condition.hold_s,
        ),
        "order3_tracking_window_sample_count": 10,
        "random_morphology_takeoff_backend_config_hash": "8" * 64,
        "random_morphology_takeoff_physical_model_hash": (
            _UNIT_PHYSICAL_MODEL_HASH
        ),
        "random_morphology_takeoff_collision_geometry_hash": "9" * 64,
        "order3_terminal_metrics": terminal_metrics.to_dict(),
        "order3_free_flight_terminal_metrics": terminal_metrics.to_dict(),
        "order3_free_flight_tracking_cost": tracking_cost,
        "order3_free_flight_success": success,
        "order3_qp_infeasible": False,
        "order3_hard_collision": False,
        "order3_non_finite_state": False,
        "order3_unsupported_actuator": False,
        "order3_pi_l_rollout": learned,
        "order3_pi_l_checkpoint_sha256": checkpoint_sha256 if learned else None,
        "order3_deterministic_baseline_rollout": not learned,
        "order3_fallback_used": fallback_used,
        "order3_fallback_reason": fallback_reason,
    }
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _synchronize_episode_reports(
    episode: Order3EvaluationEpisode,
    *,
    suffix: str,
) -> Order3EvaluationEpisode:
    learned_source = Path(str(episode.learned_report_path))
    baseline_source = Path(str(episode.deterministic_baseline_report_path))
    learned = json.loads(learned_source.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_source.read_text(encoding="utf-8"))
    learned.update(
        {
            "order3_free_flight_terminal_metrics": (
                episode.terminal_metrics.to_dict()
                if episode.terminal_metrics is not None
                else None
            ),
            "order3_free_flight_tracking_cost": episode.tracking_cost,
            "order3_free_flight_success": episode.success,
            "order3_qp_infeasible": episode.qp_infeasible,
            "order3_hard_collision": episode.hard_collision,
            "order3_non_finite_state": episode.non_finite_state,
            "order3_unsupported_actuator": episode.unsupported_actuator,
            "order3_fallback_used": episode.fallback_used,
            "order3_fallback_reason": episode.fallback_reason,
        }
    )
    baseline.update(
        {
            "order3_free_flight_terminal_metrics": (
                episode.deterministic_baseline_terminal_metrics.to_dict()
                if episode.deterministic_baseline_terminal_metrics is not None
                else None
            ),
            "order3_free_flight_tracking_cost": (
                episode.deterministic_baseline_tracking_cost
            ),
            "order3_free_flight_success": (
                order3_terminal_metrics_success(
                    episode.deterministic_baseline_terminal_metrics,
                    task_mode=episode.task_mode,
                )
                if episode.deterministic_baseline_terminal_metrics is not None
                else False
            ),
        }
    )
    learned_path = learned_source.with_name(f"{learned_source.stem}-{suffix}.json")
    baseline_path = baseline_source.with_name(f"{baseline_source.stem}-{suffix}.json")
    learned_path.write_text(json.dumps(learned, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    baseline_path.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return replace(
        episode,
        learned_report_path=str(learned_path),
        learned_report_sha256=hash_file(learned_path),
        deterministic_baseline_report_path=str(baseline_path),
        deterministic_baseline_report_sha256=hash_file(baseline_path),
    )


def _terminal_metrics(*, success: bool) -> Order3TerminalMetrics:
    return Order3TerminalMetrics(
        position_error_m=0.0 if success else 1.0,
        attitude_error_rad=0.0,
        linear_velocity_error_mps=0.0,
        angular_velocity_error_rad_s=0.0,
        within_tolerance_duration_s=1.0,
        takeoff_height_gain_ratio=1.0 if success else 0.0,
    )


def _held_out_hashes(evidence: _Evidence) -> set[str]:
    return {
        entry.structural_hash
        for entry in evidence.pool.entries
        if entry.split == DatasetSplit.HELD_OUT
    }
