from pathlib import Path

import pytest

from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_rollout_buckets import (
    load_order9_pi_l_rollout_bucket_manifest,
    order9_pi_l_collector_arguments,
    prepare_order9_pi_l_rollout_buckets,
    validate_order9_pi_l_rollout_bucket_bytes,
)
from amsrr.training.order9_teacher import build_order8_grasp_carry_task_spec


def _base_task() -> TaskSpec:
    return build_order8_grasp_carry_task_spec(
        object_pose_world=(0.55, 0.0, 0.225, 0.0, 0.0, 0.0, 1.0),
        object_size_m=(0.30, 0.40, 0.15),
        object_mass_kg=1.0,
        object_friction=0.6,
        required_transport_distance_m=0.2,
        support_height_m=0.15,
        max_contact_force_n=30.0,
        max_contact_torque_nm=5.0,
        selected_gripper_friction=4.5,
        task_id="order9-bucket-unit-base",
    )


def test_fixed_pi_l_rollout_buckets_bind_randomization_and_collector_args(
    tmp_path: Path,
) -> None:
    repository = Path.cwd()
    config = load_order9_learning_config(
        repository / "configs/training/order9_learning_curriculum.yaml"
    )
    physical = build_physical_model_from_config(
        repository / "configs/robot/robot_model.yaml"
    )
    output = tmp_path / "buckets"
    manifest = prepare_order9_pi_l_rollout_buckets(
        output,
        config=config,
        stage_id="c2_pi_l_ppo_fixed_conservative",
        physical_model=physical,
        base_task_spec=_base_task(),
        train_bucket_count=2,
        validation_bucket_count=1,
        repository_root=repository,
        fixed_robot_usd_path=(
            "artifacts/isaac/robots/holon/holon_p4_2_graph/"
            "holon_p4_2_graph.usda"
        ),
        seed=120,
    )
    validate_order9_pi_l_rollout_bucket_bytes(
        output / "manifest.json", repository_root=repository
    )
    loaded = load_order9_pi_l_rollout_bucket_manifest(output)

    assert loaded.to_dict() == manifest.to_dict()
    assert len(manifest.buckets) == 3
    assert {bucket.split for bucket in manifest.buckets} == {
        DatasetSplit.TRAIN,
        DatasetSplit.VALIDATION,
    }
    assert len({bucket.task_id for bucket in manifest.buckets}) == 3
    assert len({bucket.structural_hash for bucket in manifest.buckets}) == 1
    for bucket in manifest.buckets:
        task = TaskSpec.from_json(
            (output / bucket.task_spec_path).read_text(encoding="utf-8")
        )
        assert task.metadata["order9_rollout_bucket_id"] == bucket.bucket_id
        assert task.metadata["estimated_com_object"] == bucket.estimated_com_object
        arguments = order9_pi_l_collector_arguments(
            bucket,
            bucket_manifest_path=output / "manifest.json",
            repository_root=repository,
        )
        assert "--estimated-com-object" in arguments
        assert "--task-spec-json" in arguments
        assert "--morphology-graph-json" in arguments

    with pytest.raises(FileExistsError, match="output exists"):
        prepare_order9_pi_l_rollout_buckets(
            output,
            config=config,
            stage_id="c2_pi_l_ppo_fixed_conservative",
            physical_model=physical,
            base_task_spec=_base_task(),
            train_bucket_count=1,
            validation_bucket_count=1,
            repository_root=repository,
            fixed_robot_usd_path=(
                "artifacts/isaac/robots/holon/holon_p4_2_graph/"
                "holon_p4_2_graph.usda"
            ),
        )
