from __future__ import annotations

from pathlib import Path

import pytest

from amsrr.logging.episode_archive import write_episode_archives_jsonl
from amsrr.training.p4_0_full_pipeline_runner import (
    P4_0FullPipelineRunner,
    P4_0FullPipelineRunnerConfig,
)
from amsrr.training.p4_3_dataset_builder import build_p4_3_dataset
from amsrr.training.p4_3_reward import compute_p4_3_archive_rewards
from amsrr.training.p2_inspection_context import default_grasp_carry_task_spec
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.policies.contact_candidate_sampler import ContactCandidateSampler


def test_p4_3_dataset_builder_writes_aligned_task_disjoint_shards(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    result = P4_0FullPipelineRunner(
        default_grasp_carry_task_spec(),
        runner_config=P4_0FullPipelineRunnerConfig(
            episode_count=3,
            seed=5,
            source_hash="p4_3_dataset_unit",
        )
    ).run()
    assert len(result.archives) == 3
    for archive in result.archives:
        step_count = min(
            len(archive.runtime_observations),
            len(archive.policy_commands),
            len(archive.controller_commands),
        )
        archive.runtime_observations = archive.runtime_observations[:step_count]
        archive.policy_commands = archive.policy_commands[:step_count]
        archive.controller_commands = archive.controller_commands[:step_count]
        archive.actuator_target_records = [
            {
                "command_index": index,
                "metrics": {"clipped_target_count": 0.0},
            }
            for index in range(step_count)
        ]
        archive.reproducibility.setdefault("urdf_hash", "unit-urdf")
        archive.reproducibility.setdefault("thrust_model_hash", "unit-thrust")
        archive.rollout_artifacts.setdefault("backend", "unit_simplified")
        archive.rollout_artifacts.setdefault("contact_model", "unit_contact")
        builder_result = IRGBuilder().build_with_scene_graph(archive.task_spec)
        candidate_set = ContactCandidateSampler().sample(
            task_spec=archive.task_spec,
            irg=archive.irg,
            interaction_envelope=archive.interaction_envelope,
            morphology_graph=archive.runtime_observations[0].morphology_graph,
            geometry_descriptors=builder_result.scene_graph.geometry_descriptors,
        )
        archive.rollout_artifacts["contact_candidate_set"] = candidate_set.to_dict()
    write_episode_archives_jsonl(source_path, result.archives)

    built = build_p4_3_dataset(
        archive_paths=[source_path],
        output_dir=tmp_path / "dataset",
        low_level_stride=4,
    )

    manifest = built.manifest
    assert manifest.record_counts["isaac_rollout"] == 3
    assert manifest.record_counts["interaction_trajectory"] == 3
    assert manifest.record_counts["design_outcome"] == 3
    assert manifest.record_counts["low_level_control"] == sum(
        len(_sampled_indices(len(archive.runtime_observations), stride=4))
        for archive in result.archives
    )
    assert len(manifest.train_task_ids) == 1
    assert len(manifest.validation_task_ids) == 1
    assert len(manifest.held_out_task_ids) == 1
    assert not set(manifest.train_task_ids) & set(manifest.validation_task_ids)
    assert all(record.stage_masks.low_level_control_mask for record in built.low_level_records)
    assert manifest.metadata["reward_alignment"] == "command_i_with_obs_i_to_obs_i_plus_1"
    for archive in result.archives:
        episode_records = [
            record
            for record in built.low_level_records
            if record.episode_id == archive.episode_id
        ]
        count = len(archive.runtime_observations)
        assert [record.step_index for record in episode_records] == _sampled_indices(
            count,
            stride=4,
        )
        assert sum(float(record.reward or 0.0) for record in episode_records) == pytest.approx(
            sum(record["reward"] for record in compute_p4_3_archive_rewards(archive))
        )
        previous_end = -1
        for record in episode_records:
            assert record.runtime_observation == archive.runtime_observations[record.step_index]
            assert record.policy_command == archive.policy_commands[record.step_index]
            assert record.controller_command == archive.controller_commands[record.step_index]
            assert record.reward_terms is not None
            assert record.reward_terms["pre_observation_index"] == float(record.step_index)
            assert record.reward_terms["interval_start_step"] == float(previous_end + 1)
            assert record.reward_terms["interval_end_step"] == float(record.step_index)
            previous_end = record.step_index
        if count >= 2:
            assert [record.step_index for record in episode_records if record.terminal] == [
                count - 2
            ]
            final = episode_records[-1]
            assert final.step_index == count - 1
            assert final.terminal is False
            assert final.reward_terms is not None
            assert final.reward_terms["state_transition_data_available"] == 0.0
            assert final.reward_terms["terminal_reward_data_available"] == 0.0
    assert Path(built.manifest_path).is_file()


def _sampled_indices(count: int, *, stride: int) -> list[int]:
    required = {count - 1}
    if count >= 2:
        required.add(count - 2)
    return sorted(set(range(0, count, stride)) | required)
