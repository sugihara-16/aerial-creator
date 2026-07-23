from __future__ import annotations

from pathlib import Path

import pytest

from amsrr.policies.order9_low_level_policy import (
    Order9LowLevelPolicyConfig,
    Order9PhaseConditionedActorCritic,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.order9 import (
    ORDER9_POLICY_CHECKPOINT_VERSION,
    ORDER9_STAGE_RUN_VERSION,
    Order9ArtifactBinding,
    Order9PolicyCheckpointMetadata,
    Order9PolicyFamily,
    Order9StageRunManifest,
    Order9StageRunStatus,
)
from amsrr.training.order9_checkpoints import (
    order9_model_config_dict,
    order9_policy_identity,
    order9_state_dict_hash,
    save_order9_policy_checkpoint,
)
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_curriculum_lineage import (
    ORDER9_CURRICULUM_LINEAGE_IMPORT_VERSION,
    Order9CurriculumLineageImport,
    load_order9_stage_parent_checkpoint,
    validate_order9_curriculum_lineage_import,
)
from amsrr.training.order9_pi_l_active_knot_migration import (
    prepare_order9_pi_l_active_knot_initializer,
)
from amsrr.utils.hashing import hash_file, stable_hash


def test_v2_to_v3_lineage_requires_explicit_active_knot_c3_initializer(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "order9_v2.yaml"
    successor_path = tmp_path / "order9_v3.yaml"
    source_path.write_text(
        Path("configs/training/order9_learning_curriculum_v2.yaml").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    successor_path.write_text(
        Path("configs/training/order9_learning_curriculum.yaml").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    source = load_order9_learning_config(source_path)
    successor = load_order9_learning_config(successor_path)
    source_hash = stable_hash(source.curriculum.to_dict())
    successor_hash = stable_hash(successor.curriculum.to_dict())
    source_stage = source.curriculum.stages[2]
    successor_stage = successor.curriculum.stages[3]

    model = Order9PhaseConditionedActorCritic(
        Order9LowLevelPolicyConfig(
            graph_hidden_dim=16,
            graph_message_layers=1,
            recurrent_hidden_dim=24,
            max_local_joint_slots=4,
        )
    )
    family, policy_version = order9_policy_identity(model)
    metadata = Order9PolicyCheckpointMetadata(
        checkpoint_version=ORDER9_POLICY_CHECKPOINT_VERSION,
        policy_family=family,
        policy_version=policy_version,
        curriculum_schedule_hash=source_hash,
        curriculum_stage_id=source_stage.stage_id,
        curriculum_stage_index=source_stage.stage_index,
        learning_mode=source_stage.learning_mode.value,
        model_config_hash=stable_hash(order9_model_config_dict(model)),
        state_dict_hash=order9_state_dict_hash(model.state_dict()),
        physical_model_hash="1" * 64,
        actor_observation_contract=(
            "task_phase_morphology_centroidal_no_raw_contact_v1"
        ),
        critic_observation_contract="actor_plus_privileged_disturbance_v1",
        action_contract=(
            "bounded_complete_centroidal_and_absolute_local_joint_command_v2"
        ),
        git_revision="unit-test",
        random_seed=9,
        input_artifact_hashes={"dataset_manifest": "2" * 64},
    )
    checkpoint_path = tmp_path / "c2.pt"
    checkpoint_sha256 = save_order9_policy_checkpoint(
        checkpoint_path,
        model=model,
        metadata=metadata,
    )
    manifest_path = tmp_path / "c2-promoted.json"
    manifest = Order9StageRunManifest(
        run_version=ORDER9_STAGE_RUN_VERSION,
        run_id="unit-c2-promoted",
        stage_id=source_stage.stage_id,
        stage_index=source_stage.stage_index,
        status=Order9StageRunStatus.PROMOTED,
        schedule_hash=source_hash,
        stage_config_hash=stable_hash(source_stage.to_dict()),
        runtime_config_hash="3" * 64,
        random_seed=9,
        device="cpu",
        environment_count=1,
        input_artifacts=[
            Order9ArtifactBinding(
                artifact_kind="unit_source",
                path="unit-source",
                sha256="4" * 64,
            )
        ],
        policy_checkpoint_sha256_by_family={
            Order9PolicyFamily.PI_L.value: checkpoint_sha256
        },
        promoted=True,
    )
    manifest_path.write_text(
        manifest.to_json(indent=2) + "\n",
        encoding="utf-8",
    )
    lineage_path = tmp_path / "lineage.json"
    specification = Order9CurriculumLineageImport(
        import_version=ORDER9_CURRICULUM_LINEAGE_IMPORT_VERSION,
        source_config_path=str(source_path),
        successor_config_path=str(successor_path),
        source_schedule_version=source.curriculum.schedule_version,
        successor_schedule_version=successor.curriculum.schedule_version,
        source_schedule_hash=source_hash,
        successor_schedule_hash=successor_hash,
        imported_through_stage_id=source_stage.stage_id,
        imported_through_stage_index=source_stage.stage_index,
        successor_first_stage_id=successor_stage.stage_id,
        promoted_stage_manifest_path=str(manifest_path),
        promoted_stage_manifest_sha256=hash_file(manifest_path),
        policy_checkpoint_path=str(checkpoint_path),
        policy_checkpoint_sha256=checkpoint_sha256,
        policy_family=Order9PolicyFamily.PI_L,
    )
    lineage_path.write_text(
        specification.to_json(indent=2) + "\n",
        encoding="utf-8",
    )

    validated = validate_order9_curriculum_lineage_import(
        lineage_path,
        successor_config=successor,
    )
    assert validated.specification.source_schedule_hash == source_hash
    with pytest.raises(SchemaValidationError, match="cannot execute C3 directly"):
        load_order9_stage_parent_checkpoint(
            successor,
            successor_stage,
            checkpoint_path,
            expected_family=Order9PolicyFamily.PI_L,
            update_index=0,
            lineage_import_path=lineage_path,
        )
    prepared = prepare_order9_pi_l_active_knot_initializer(
        config_path=successor_path,
        lineage_import_path=lineage_path,
        output_checkpoint_path=tmp_path / "c3-initializer.pt",
        output_manifest_path=tmp_path / "c3-initializer-manifest.json",
        git_revision="unit-test",
    )
    loaded = load_order9_stage_parent_checkpoint(
        successor,
        successor_stage,
        prepared.checkpoint_path,
        expected_family=Order9PolicyFamily.PI_L,
        update_index=0,
        lineage_import_path=lineage_path,
    )
    assert loaded.sha256 == prepared.checkpoint_sha256
    assert loaded.metadata.metadata["initializer_only"] is True
    assert loaded.metadata.metadata["ppo_update_index"] == -1
