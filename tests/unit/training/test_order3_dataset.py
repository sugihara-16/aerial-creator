from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from amsrr.morphology.random_connected import (
    RandomConnectedMorphologyDistribution,
    morphology_structural_hash,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.order3 import (
    ORDER3_DATASET_VERSION,
    ORDER3_POLICY_FAMILY,
    Order3PolicyTransition,
)
from amsrr.schemas.policies import (
    POLICY_COMMAND_CONTRACT_CENTROIDAL,
    ControllerStatus,
)
from amsrr.schemas.runtime import (
    ContactState,
    ModuleRuntimeState,
    RuntimeObservation,
    TaskProgressState,
)
from amsrr.training.order3_dataset import (
    DEFAULT_ORDER3_DATASET_DIR,
    load_order3_dataset,
    write_order3_dataset,
)
from amsrr.utils.hashing import hash_file


def test_order3_dataset_deterministic_atomic_roundtrip(tmp_path: Path) -> None:
    morphologies = _morphologies()
    transitions = [
        _transition(
            DatasetSplit.HELD_OUT,
            morphologies[DatasetSplit.HELD_OUT],
            episode_id="held-episode",
            step_index=0,
            terminal=True,
        ),
        _transition(
            DatasetSplit.TRAIN,
            morphologies[DatasetSplit.TRAIN],
            episode_id="train-episode",
            step_index=1,
            terminal=True,
        ),
        _transition(
            DatasetSplit.VALIDATION,
            morphologies[DatasetSplit.VALIDATION],
            episode_id="validation-episode",
            step_index=0,
            terminal=True,
        ),
        _transition(
            DatasetSplit.TRAIN,
            morphologies[DatasetSplit.TRAIN],
            episode_id="train-episode",
            step_index=0,
            terminal=False,
        ),
    ]
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = write_order3_dataset(
        reversed(transitions),
        output_dir=first_dir,
        pool_hash="pool-hash",
        physical_model_hash="physical-model-hash",
        config_hash="config-hash",
        shard_size=1,
        metadata={"curriculum_version": "unit-curriculum-v1"},
    )
    second = write_order3_dataset(
        transitions,
        output_dir=second_dir,
        pool_hash="pool-hash",
        physical_model_hash="physical-model-hash",
        config_hash="config-hash",
        shard_size=1,
        metadata={"curriculum_version": "unit-curriculum-v1"},
    )

    assert Path(first.manifest_path).read_bytes() == Path(second.manifest_path).read_bytes()
    assert first.manifest.dataset_version == ORDER3_DATASET_VERSION
    assert first.manifest.policy_contract_version == POLICY_COMMAND_CONTRACT_CENTROIDAL
    assert first.manifest.policy_family == ORDER3_POLICY_FAMILY
    assert first.manifest.actor_privileged_wrench_inputs is False
    assert first.manifest.transition_counts == {
        "train": 2,
        "validation": 1,
        "held_out": 1,
    }
    assert first.manifest.real_isaac_episode_counts == {
        "train": 1,
        "validation": 1,
        "held_out": 1,
    }
    assert first.manifest.metadata["legacy_p4_3_artifact_reused"] is False
    assert first.manifest.metadata["module_count_counts"]["train"]["2"] == 1
    assert first.manifest.metadata["transition_module_count_counts"]["train"]["2"] == 2
    assert [record.step_index for record in first.transitions if record.split == DatasetSplit.TRAIN] == [0, 1]
    assert all(
        record.policy_contract_version == POLICY_COMMAND_CONTRACT_CENTROIDAL
        for record in first.transitions
    )
    for relative_path, expected_hash in first.manifest.transition_shard_hashes.items():
        assert hash_file(first_dir / relative_path) == expected_hash
        assert (first_dir / relative_path).read_bytes() == (second_dir / relative_path).read_bytes()
    assert not list(first_dir.glob("*.tmp"))
    assert load_order3_dataset(first_dir).transitions == first.transitions

    old_paths = set(first.manifest.transition_shard_hashes)
    updated_records = [
        replace(record, reward=2.0)
        if record.split == DatasetSplit.TRAIN and record.step_index == 0
        else record
        for record in transitions
    ]
    updated = write_order3_dataset(
        updated_records,
        output_dir=first_dir,
        pool_hash="pool-hash",
        physical_model_hash="physical-model-hash",
        config_hash="config-hash",
        shard_size=1,
        metadata={"curriculum_version": "unit-curriculum-v1"},
        overwrite=True,
    )
    new_paths = set(updated.manifest.transition_shard_hashes)
    assert old_paths != new_paths
    assert all(not (first_dir / path).exists() for path in old_paths - new_paths)
    assert load_order3_dataset(first_dir).transitions == updated.transitions


def test_order3_dataset_rejects_structural_hash_mismatch_and_split_leakage(
    tmp_path: Path,
) -> None:
    morphologies = _morphologies()
    records = _one_transition_per_split(morphologies)
    bad_hash = replace(records[0], structural_hash="0" * 64)

    with pytest.raises(SchemaValidationError, match="structural_hash does not match"):
        write_order3_dataset(
            [bad_hash, *records[1:]],
            output_dir=tmp_path / "bad-hash",
            pool_hash="pool",
            physical_model_hash="physical",
            config_hash="config",
        )

    leaked_validation = _transition(
        DatasetSplit.VALIDATION,
        morphologies[DatasetSplit.TRAIN],
        episode_id="leaked-validation",
        step_index=0,
        terminal=True,
    )
    with pytest.raises(SchemaValidationError, match="split-disjoint"):
        write_order3_dataset(
            [records[0], leaked_validation, records[2]],
            output_dir=tmp_path / "leaked",
            pool_hash="pool",
            physical_model_hash="physical",
            config_hash="config",
        )


def test_order3_dataset_rejects_privileged_actor_wrench_and_legacy_artifact_root(
    tmp_path: Path,
) -> None:
    morphologies = _morphologies()
    records = _one_transition_per_split(morphologies)
    observation_data = records[0].runtime_observation.to_dict()
    observation_data["contact_states"] = [
        ContactState(
            contact_id="privileged-contact",
            entity_a="robot",
            entity_b="floor",
            wrench_world=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ).to_dict()
    ]
    leaked_observation = RuntimeObservation.from_dict(observation_data)
    leaked = replace(records[0], runtime_observation=leaked_observation)

    with pytest.raises(SchemaValidationError, match="must not contain measured contact wrench"):
        write_order3_dataset(
            [leaked, *records[1:]],
            output_dir=tmp_path / "privileged-leak",
            pool_hash="pool",
            physical_model_hash="physical",
            config_hash="config",
        )

    legacy_destination = "artifacts/p4_3/order3-unit-must-never-be-written"
    with pytest.raises(SchemaValidationError, match="legacy artifacts/p4_3"):
        write_order3_dataset(
            records,
            output_dir=legacy_destination,
            pool_hash="pool",
            physical_model_hash="physical",
            config_hash="config",
        )
    assert not Path(legacy_destination).exists()
    assert DEFAULT_ORDER3_DATASET_DIR.startswith("artifacts/p4_full/order3_")


def test_order3_dataset_load_rejects_tampered_shard(tmp_path: Path) -> None:
    output_dir = tmp_path / "dataset"
    result = write_order3_dataset(
        _one_transition_per_split(_morphologies()),
        output_dir=output_dir,
        pool_hash="pool",
        physical_model_hash="physical",
        config_hash="config",
    )
    shard_path = output_dir / result.manifest.transition_shards["train"][0]
    shard_path.write_bytes(shard_path.read_bytes() + b"\n")

    with pytest.raises(SchemaValidationError, match="integrity check failed"):
        load_order3_dataset(output_dir)


def test_order3_dataset_refuses_unowned_nonempty_output_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "not-owned"
    output_dir.mkdir()
    (output_dir / "user-file.txt").write_text("preserve me\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        write_order3_dataset(
            _one_transition_per_split(_morphologies()),
            output_dir=output_dir,
            pool_hash="pool",
            physical_model_hash="physical",
            config_hash="config",
        )

    assert (output_dir / "user-file.txt").read_text(encoding="utf-8") == "preserve me\n"


def _morphologies() -> dict[DatasetSplit, MorphologyGraph]:
    physical_model = build_physical_model_from_config("configs/robot/robot_model.yaml")
    distribution = RandomConnectedMorphologyDistribution(physical_model)
    return {
        DatasetSplit.TRAIN: distribution.sample(seed=102, module_count=2),
        DatasetSplit.VALIDATION: distribution.sample(seed=203, module_count=3),
        DatasetSplit.HELD_OUT: distribution.sample(seed=304, module_count=4),
    }


def _one_transition_per_split(
    morphologies: dict[DatasetSplit, MorphologyGraph],
) -> list[Order3PolicyTransition]:
    return [
        _transition(
            split,
            morphologies[split],
            episode_id=f"{split.value}-episode",
            step_index=0,
            terminal=True,
        )
        for split in DatasetSplit
    ]


def _transition(
    split: DatasetSplit,
    morphology: MorphologyGraph,
    *,
    episode_id: str,
    step_index: int,
    terminal: bool,
) -> Order3PolicyTransition:
    time_s = 0.01 * step_index
    status = ControllerStatus(
        status="ok",
        qp_feasible=True,
        active_mode="rigid_body_qp",
        metrics={"allocation_residual_norm": 0.0},
    )
    observation = RuntimeObservation(
        time_s=time_s,
        morphology_graph=morphology,
        module_states=[
            ModuleRuntimeState(
                module_id=module.module_id,
                pose_world=module.pose_in_design_frame,
                twist_world=[0.0] * 6,
                joint_positions={},
                joint_velocities={},
            )
            for module in morphology.modules
        ],
        object_states=[],
        contact_states=[],
        controller_status=status,
        task_progress=TaskProgressState(
            phase_label="hover",
            progress_ratio=float(step_index),
            success=terminal,
            metrics={"tracking_error_m": 0.01},
        ),
    )
    return Order3PolicyTransition(
        episode_id=episode_id,
        split=split,
        graph_id=morphology.graph_id,
        structural_hash=morphology_structural_hash(morphology),
        step_index=step_index,
        time_s=time_s,
        runtime_observation=observation,
        target_pose_world=(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        target_twist=[0.0] * 6,
        previous_action=[0.0] * 12,
        action=[0.01] * 12,
        recurrent_state_in=[0.0] * 8,
        old_log_prob=-0.5,
        old_value=0.25,
        reward=1.0,
        terminal=terminal,
        policy_applied=True,
        privileged_disturbance_body=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        metrics={"isaac_backed": 1.0, "position_error_m": 0.01},
    )
