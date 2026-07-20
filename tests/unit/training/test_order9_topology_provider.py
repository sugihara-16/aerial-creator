from __future__ import annotations

from amsrr.morphology.random_connected import morphology_structural_hash
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.datasets import DatasetSplit
from amsrr.training.order9_teacher import build_order8_grasp_carry_task_spec
from amsrr.training.order9_topology_provider import Order9CurriculumTopologyProvider


def test_topology_provider_is_deterministic_split_safe_and_task_conditioned() -> None:
    physical_model = build_physical_model_from_config(
        "configs/robot/robot_model.yaml"
    )
    provider = Order9CurriculumTopologyProvider.from_path(
        "artifacts/p4_full/order9/morphology_pool.json",
        physical_model=physical_model,
    )
    first = provider.sample(
        _task(),
        split=DatasetSplit.TRAIN,
        seed=9009,
        sample_index=4,
        module_count=4,
    )
    second = provider.sample(
        _task(),
        split=DatasetSplit.TRAIN,
        seed=9009,
        sample_index=4,
        module_count=4,
    )

    assert first.to_dict() == second.to_dict()
    assert first.module_count == 4
    assert first.split == DatasetSplit.TRAIN
    assert first.morphology_graph.robot_anchors
    assert any(
        anchor.associated_contact_slot_ids
        for anchor in first.morphology_graph.robot_anchors
    )
    assert (
        morphology_structural_hash(first.morphology_graph)
        == first.source_structural_hash
    )
    assert first.metadata["learned_pi_d_used"] is False


def _task():
    return build_order8_grasp_carry_task_spec(
        object_pose_world=(0.5, 0.0, 0.225, 0.0, 0.0, 0.0, 1.0),
        object_size_m=(0.30, 0.40, 0.15),
        object_mass_kg=1.0,
        object_friction=0.6,
        required_transport_distance_m=0.20,
        support_height_m=0.15,
        max_contact_force_n=30.0,
        max_contact_torque_nm=5.0,
    )
