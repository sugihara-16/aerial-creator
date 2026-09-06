from __future__ import annotations

from collections import Counter
import math

import pytest

from amsrr.geometry.pose_math import compose_pose, inverse_pose
from amsrr.schemas.datasets import DatasetSplit
from amsrr.training import order9_r1_held_out_confirmation_v16 as held_out_v16
from amsrr.training.order9_r1_held_out_confirmation_v16 import (
    ORDER9_R1_HELD_OUT_BUCKET_COUNT,
    ORDER9_R1_HELD_OUT_SAMPLE_COUNT,
    build_order9_r1_held_out_composed_target_v16,
    build_order9_r1_held_out_source_bucket_v16,
)
from scripts import order9_run_r1_held_out_scene_confirmation_v16 as runtime_v16


def test_held_out_confirmation_uses_current_complete_task_phase_order() -> None:
    assert held_out_v16._PHASES == (
        "approach",
        "contact_acquisition",
        "lift",
        "transport",
        "place",
        "release",
        "retreat",
        "settle",
    )


def test_held_out_population_is_complete_stratified_and_never_relabels_split() -> None:
    pairs, references, targets = runtime_v16._pairs_and_cases()

    assert len(pairs) == ORDER9_R1_HELD_OUT_BUCKET_COUNT
    assert Counter(pair.held_out_entry.module_count for pair in pairs) == {
        module_count: 2 for module_count in range(2, 9)
    }
    assert len({pair.held_out_entry.structural_hash for pair in pairs}) == 14
    assert all(pair.held_out_entry.split == DatasetSplit.HELD_OUT for pair in pairs)
    assert all(pair.asset_entry.split == DatasetSplit.HELD_OUT for pair in pairs)
    assert all(
        len(targets[pair.bucket_id]) == ORDER9_R1_HELD_OUT_SAMPLE_COUNT
        for pair in pairs
    )
    assert set(references) == {pair.bucket_id for pair in pairs}


def test_held_out_source_bucket_binds_held_out_robot_assets() -> None:
    pairs, _references, _targets = runtime_v16._pairs_and_cases()

    for pair in pairs:
        bucket = build_order9_r1_held_out_source_bucket_v16(
            pair,
            repository_root=runtime_v16.REPOSITORY,
            source_manifest_path=runtime_v16.SOURCE_MANIFEST,
        )
        assert bucket.split == DatasetSplit.HELD_OUT
        assert bucket.structural_hash == pair.held_out_entry.structural_hash
        assert bucket.robot_usd_path == pair.asset_entry.usd_path
        assert bucket.metadata["r1_held_out_source_split"] == "held_out"
        assert "accepted_nominal_trajectory" not in bucket.metadata


def test_composed_target_moves_support_with_object_in_one_total_transform() -> None:
    pairs, references, targets = runtime_v16._pairs_and_cases()
    pair = pairs[0]
    reference = runtime_v16._reference_case(pair, references[pair.bucket_id])
    corner = next(
        case
        for case in targets[pair.bucket_id]
        if case.sample_kind == "lattice" and case.sample_index == 0
    )
    candidate_id, identity = runtime_v16._target_identity_task(pair, corner)

    midpoint, final, delta_one, delta_two, overall = (
        build_order9_r1_held_out_composed_target_v16(
            reference.task_spec,
            target_identity_task=identity,
        )
    )

    source_object = reference.task_spec.scene.objects[0].pose_world
    final_object = final.scene.objects[0].pose_world
    source_support = reference.task_spec.scene.environment.support_surfaces[
        0
    ].pose_world
    final_support = final.scene.environment.support_surfaces[0].pose_world
    midpoint_delta = compose_pose(
        midpoint.scene.objects[0].pose_world,
        inverse_pose(source_object),
    )

    assert final.metadata["r1_calibration_candidate_id"] == candidate_id
    assert final.metadata["dataset_split"] == "held_out"
    assert final_object == pytest.approx(identity.scene.objects[0].pose_world)
    assert final_support == pytest.approx(compose_pose(overall, source_support))
    assert delta_one == pytest.approx(midpoint_delta)
    assert compose_pose(delta_two, delta_one) == pytest.approx(overall)
    assert math.dist(source_object[:2], midpoint.scene.objects[0].pose_world[:2]) <= (
        math.sqrt(2.0) * 0.020 + 1.0e-12
    )
    assert (
        math.dist(midpoint.scene.objects[0].pose_world[:2], final_object[:2])
        <= math.sqrt(2.0) * 0.020 + 1.0e-12
    )
    assert final.metadata["r1_support_pose_transformed_with_scene"] is True
