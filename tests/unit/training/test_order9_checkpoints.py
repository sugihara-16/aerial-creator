from __future__ import annotations

from pathlib import Path

import pytest
import torch

from amsrr.policies.order9_design_policy import Order9AutoregressiveDesignPolicy
from amsrr.policies.order9_high_level_policy import Order9AutoregressiveHighLevelPolicy
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
    Order9StageRunManifest,
    Order9StageRunStatus,
)
from amsrr.training.order9_checkpoints import (
    load_order9_policy_checkpoint,
    order9_model_config_dict,
    order9_policy_identity,
    order9_state_dict_hash,
    save_order9_policy_checkpoint,
)
from amsrr.utils.hashing import hash_file, stable_hash


@pytest.mark.parametrize(
    "model",
    [
        Order9PhaseConditionedActorCritic(
            Order9LowLevelPolicyConfig(
                graph_hidden_dim=16,
                graph_message_layers=1,
                recurrent_hidden_dim=24,
                max_local_joint_slots=4,
            )
        ),
        Order9AutoregressiveHighLevelPolicy(),
        Order9AutoregressiveDesignPolicy(),
    ],
)
def test_order9_checkpoint_roundtrips_each_policy_family(
    tmp_path: Path,
    model: torch.nn.Module,
) -> None:
    family, policy_version = order9_policy_identity(model)
    metadata = _metadata(model)
    path = tmp_path / f"{family.value}.pt"

    saved_hash = save_order9_policy_checkpoint(path, model=model, metadata=metadata)
    loaded = load_order9_policy_checkpoint(
        path,
        expected_sha256=saved_hash,
        expected_family=family,
        expected_schedule_hash=metadata.curriculum_schedule_hash,
    )

    assert saved_hash == hash_file(path)
    assert loaded.metadata.policy_version == policy_version
    assert type(loaded.model) is type(model)
    for name, expected in model.state_dict().items():
        assert torch.equal(loaded.model.state_dict()[name], expected)


def test_order9_checkpoint_rejects_state_tampering(tmp_path: Path) -> None:
    model = Order9AutoregressiveDesignPolicy()
    path = tmp_path / "source.pt"
    save_order9_policy_checkpoint(path, model=model, metadata=_metadata(model))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    first = next(iter(payload["state_dict"]))
    payload["state_dict"][first] = payload["state_dict"][first].clone() + 1.0
    tampered = tmp_path / "tampered.pt"
    torch.save(payload, tampered)

    with pytest.raises(SchemaValidationError, match="state_dict hash mismatch"):
        load_order9_policy_checkpoint(tampered)


def test_stage_run_manifest_keeps_rejection_distinct_from_promotion() -> None:
    digest = "a" * 64
    rejected = Order9StageRunManifest(
        run_version=ORDER9_STAGE_RUN_VERSION,
        run_id="order9-c2-seed-9",
        stage_id="c2_pi_l_ppo_fixed_conservative",
        stage_index=2,
        status=Order9StageRunStatus.REJECTED,
        schedule_hash=digest,
        stage_config_hash=digest,
        runtime_config_hash=digest,
        random_seed=9,
        device="cuda:0",
        environment_count=128,
        input_artifacts=[
            Order9ArtifactBinding(
                artifact_kind="parent_checkpoint",
                path="parent.pt",
                sha256=digest,
            )
        ],
        promotion_failed_gates=["minimum_success_rate"],
    )

    assert Order9StageRunManifest.from_json(rejected.to_json()).to_dict() == rejected.to_dict()
    invalid = rejected.to_dict()
    invalid["promoted"] = True
    with pytest.raises(SchemaValidationError, match="promoted flag"):
        Order9StageRunManifest.from_dict(invalid)


def _metadata(model: torch.nn.Module) -> Order9PolicyCheckpointMetadata:
    family, policy_version = order9_policy_identity(model)
    config = order9_model_config_dict(model)
    return Order9PolicyCheckpointMetadata(
        checkpoint_version=ORDER9_POLICY_CHECKPOINT_VERSION,
        policy_family=family,
        policy_version=policy_version,
        curriculum_schedule_hash="1" * 64,
        curriculum_stage_id="c1_pi_l_bc_fixed_nominal",
        curriculum_stage_index=1,
        learning_mode="behavior_cloning",
        model_config_hash=stable_hash(config),
        state_dict_hash=order9_state_dict_hash(model.state_dict()),
        physical_model_hash="2" * 64,
        actor_observation_contract="task_phase_morphology_centroidal_no_raw_contact_v1",
        critic_observation_contract="actor_plus_privileged_disturbance_v1",
        action_contract="policy_family_specific_v1",
        git_revision="unit-test",
        random_seed=9,
        input_artifact_hashes={"dataset_manifest": "3" * 64},
    )
