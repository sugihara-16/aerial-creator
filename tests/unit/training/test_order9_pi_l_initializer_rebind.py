from pathlib import Path

from amsrr.training.order9_checkpoints import load_order9_policy_checkpoint
from amsrr.training.order9_pi_l_initializer_rebind import (
    prepare_order9_pi_l_initializer_physical_rebind,
    validate_order9_pi_l_initializer_physical_rebind,
)
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_pi_l_stage_runner import resolve_order9_pi_l_stage_plan


REPOSITORY = Path(__file__).resolve().parents[3]
SOURCE = (
    REPOSITORY
    / "artifacts/p4_full/order9/c3_preparation/pi_l_active_knot_initializer.pt"
)


def test_initializer_physical_rebind_preserves_every_policy_tensor(tmp_path: Path) -> None:
    target_physical_hash = "f" * 64
    prepared = prepare_order9_pi_l_initializer_physical_rebind(
        source_checkpoint_path=SOURCE,
        output_checkpoint_path=tmp_path / "rebound.pt",
        output_manifest_path=tmp_path / "rebound.json",
        target_physical_model_hash=target_physical_hash,
        git_revision="unit-test",
    )
    manifest = validate_order9_pi_l_initializer_physical_rebind(
        prepared.manifest_path,
        expected_target_physical_model_hash=target_physical_hash,
    )
    source = load_order9_policy_checkpoint(SOURCE)
    target = load_order9_policy_checkpoint(prepared.checkpoint_path)
    assert manifest.exact_parameter_copy is True
    assert target.metadata.state_dict_hash == source.metadata.state_dict_hash
    assert target.metadata.physical_model_hash == target_physical_hash
    assert target.metadata.parent_checkpoint_sha256 == (
        source.metadata.parent_checkpoint_sha256
    )
    assert target.metadata.metadata["ppo_update_index"] == -1


def test_rebound_initializer_passes_current_physical_stage_preflight(
    tmp_path: Path,
) -> None:
    target_physical_hash = "e" * 64
    prepared = prepare_order9_pi_l_initializer_physical_rebind(
        source_checkpoint_path=SOURCE,
        output_checkpoint_path=tmp_path / "rebound.pt",
        output_manifest_path=tmp_path / "rebound.json",
        target_physical_model_hash=target_physical_hash,
        git_revision="unit-test",
    )
    config = load_order9_learning_config(
        REPOSITORY / "configs/training/order9_learning_curriculum.yaml"
    )
    plan = resolve_order9_pi_l_stage_plan(
        config,
        stage_id="c3_pi_l_ppo_arbitrary_morphology",
        stage_root=tmp_path / "stage",
        initial_checkpoint_path=prepared.checkpoint_path,
        repository_root=REPOSITORY,
        expected_physical_model_hash=target_physical_hash,
    )
    assert plan.next_update_index == 0
    assert plan.parent_checkpoint_sha256 == prepared.checkpoint_sha256
