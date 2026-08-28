from __future__ import annotations

from pathlib import Path

import pytest

from amsrr.training.order9_c3_teacher import (
    ORDER9_C3_OVERHEAD_START_CLEARANCE_M,
    Order9C3TeacherConfig,
    build_order9_c3_neutral_runtime_observation,
    build_order9_c3_posture_collision_object,
    _offset_horizontal_grasp_candidates,
)
from amsrr.training.order9_rollout_buckets import (
    load_order9_pi_l_rollout_bucket_manifest,
)
from scripts.order9_prepare_c3_nominal_trajectories import (
    _load_posture_rejection_overrides,
    _parse_bucket_indices,
    _parser as nominal_trajectory_parser,
)
from scripts.order9_prepare_c3a_curation_pilot import _load_config
from tests.unit.training.test_order9_articulated_teacher import _system


def test_c3_teacher_config_accepts_pinned_surface_pair_and_group() -> None:
    config = Order9C3TeacherConfig(
        maximum_surface_pair_attempts=1,
        preferred_surface_port_ids=(3, 7),
        preferred_candidate_group_id="slot_0:grasp_pair:0",
    )

    assert config.preferred_surface_port_ids == (3, 7)
    assert config.preferred_candidate_group_id == "slot_0:grasp_pair:0"


def test_nominal_trajectory_cli_accepts_pinned_candidate_group() -> None:
    arguments = nominal_trajectory_parser().parse_args(
        [
            "--preferred-candidate-group-id",
            "slot_0:grasp_pair:2",
        ]
    )

    assert arguments.preferred_candidate_group_id == "slot_0:grasp_pair:2"


def test_c3_teacher_config_accepts_unique_excluded_surface_pairs() -> None:
    config = Order9C3TeacherConfig(
        excluded_surface_port_id_pairs=((3, 7), (11, 15)),
    )

    assert config.excluded_surface_port_id_pairs == ((3, 7), (11, 15))


@pytest.mark.parametrize(
    "pairs",
    [
        ((3, 3),),
        ((-1, 7),),
        ((3, 7), (7, 3)),
    ],
)
def test_c3_teacher_config_rejects_invalid_excluded_surface_pairs(
    pairs: tuple[tuple[int, int], ...],
) -> None:
    with pytest.raises(ValueError, match="excluded_surface_port_id_pairs"):
        Order9C3TeacherConfig(excluded_surface_port_id_pairs=pairs)


def test_c3_teacher_config_can_skip_full_recheck_for_static_curation() -> None:
    config = Order9C3TeacherConfig(
        require_full_trajectory_recheck=False,
    )

    assert config.require_full_trajectory_recheck is False


def test_c3_teacher_config_accepts_bounded_grasp_height_offset() -> None:
    config = Order9C3TeacherConfig(grasp_contact_height_offset_m=0.04)

    assert config.grasp_contact_height_offset_m == pytest.approx(0.04)


def test_c3_teacher_config_accepts_bounded_grasp_tangent_offset() -> None:
    config = Order9C3TeacherConfig(
        grasp_contact_tangent_offset_world_m=(0.0, 0.02, 0.0)
    )

    assert config.grasp_contact_tangent_offset_world_m == (0.0, 0.02, 0.0)


def test_c3_teacher_config_accepts_mesh_clear_pregrasp() -> None:
    config = Order9C3TeacherConfig(pregrasp_clearance_m=0.15)

    assert config.pregrasp_clearance_m == pytest.approx(0.15)


def test_c3_teacher_config_accepts_optional_contact_solution_tilt() -> None:
    config = Order9C3TeacherConfig(maximum_contact_solution_body_tilt_rad=1.0)

    assert config.maximum_contact_solution_body_tilt_rad == pytest.approx(1.0)


@pytest.mark.parametrize("value", [0.0, 3.2, float("nan")])
def test_c3_teacher_config_rejects_invalid_contact_solution_tilt(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="maximum_contact_solution_body_tilt"):
        Order9C3TeacherConfig(maximum_contact_solution_body_tilt_rad=value)


@pytest.mark.parametrize("value", [-0.001, 0.101, float("nan")])
def test_c3_teacher_config_rejects_invalid_grasp_height_offset(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="grasp_contact_height_offset_m"):
        Order9C3TeacherConfig(grasp_contact_height_offset_m=value)


def test_c3_teacher_config_rejects_oversize_grasp_tangent_offset() -> None:
    with pytest.raises(ValueError, match="grasp_contact_tangent_offset_world_m"):
        Order9C3TeacherConfig(
            grasp_contact_tangent_offset_world_m=(0.0, 0.031, 0.0)
        )


@pytest.mark.parametrize("value", [0.049, 0.301, float("nan")])
def test_c3_teacher_config_rejects_invalid_pregrasp_clearance(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="pregrasp_clearance_m"):
        Order9C3TeacherConfig(pregrasp_clearance_m=value)


def test_c3_grasp_height_offset_moves_side_contacts_only(
    grasp_carry_dict: dict,
) -> None:
    task, physical, context = _system(grasp_carry_dict)
    del task, physical
    original = context.contact_candidate_set
    shifted = _offset_horizontal_grasp_candidates(
        original,
        height_offset_m=0.04,
    )

    original_by_id = {
        candidate.candidate_id: candidate for candidate in original.candidates
    }
    shifted_count = 0
    for candidate in shifted.candidates:
        source = original_by_id[candidate.candidate_id]
        expected = (
            0.04
            if candidate.contact_mode.value == "grasp"
            and abs(float(candidate.normal_world[2])) <= 0.25
            else 0.0
        )
        assert candidate.contact_pose_world[2] == pytest.approx(
            source.contact_pose_world[2] + expected
        )
        shifted_count += int(expected > 0.0)
    assert shifted_count > 0


def test_c3_grasp_tangent_offset_stays_on_each_side_face(
    grasp_carry_dict: dict,
) -> None:
    task, physical, context = _system(grasp_carry_dict)
    del task, physical
    original = context.contact_candidate_set
    shifted = _offset_horizontal_grasp_candidates(
        original,
        height_offset_m=0.0,
        tangent_offset_world_m=(0.0, 0.02, 0.0),
    )
    original_by_id = {
        candidate.candidate_id: candidate for candidate in original.candidates
    }
    shifted_count = 0
    for candidate in shifted.candidates:
        source = original_by_id[candidate.candidate_id]
        if (
            candidate.contact_mode.value != "grasp"
            or abs(float(candidate.normal_world[2])) > 0.25
        ):
            continue
        displacement = tuple(
            float(candidate.contact_pose_world[index])
            - float(source.contact_pose_world[index])
            for index in range(3)
        )
        assert sum(
            displacement[index] * float(candidate.normal_world[index])
            for index in range(3)
        ) == pytest.approx(0.0, abs=1.0e-9)
        shifted_count += int(any(abs(value) > 1.0e-9 for value in displacement))
    assert shifted_count > 0


@pytest.mark.parametrize(
    "surface_ids",
    [
        (3, 3),
        (-1, 7),
    ],
)
def test_c3_teacher_config_rejects_invalid_pinned_surface_pair(
    surface_ids: tuple[int, int],
) -> None:
    with pytest.raises(ValueError, match="two distinct"):
        Order9C3TeacherConfig(preferred_surface_port_ids=surface_ids)


def test_c3_teacher_config_rejects_empty_pinned_group() -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        Order9C3TeacherConfig(preferred_candidate_group_id="")


def test_c3_teacher_config_requires_boolean_recheck_flag() -> None:
    with pytest.raises(ValueError, match="must be boolean"):
        Order9C3TeacherConfig(  # type: ignore[arg-type]
            require_full_trajectory_recheck=1
        )


def test_human_posture_rejections_bind_three_current_lineage_buckets() -> None:
    manifest_path = Path(
        "artifacts/p4_full/order9/stages/"
        "c3_pi_l_ppo_arbitrary_morphology/"
        "rollout_buckets_current_lineage_v5/manifest.json"
    )
    manifest = load_order9_pi_l_rollout_bucket_manifest(manifest_path)
    selected = [manifest.buckets[index] for index in (19, 27, 41)]

    overrides = _load_posture_rejection_overrides(
        Path(
            "configs/training/" "order9_c3_human_posture_rejections_v1.json"
        ).resolve(),
        buckets=selected,
        bucket_manifest_path=manifest_path.resolve(),
    )

    assert set(overrides) == {bucket.bucket_id for bucket in selected}
    assert all(
        len(value["excluded_surface_port_id_pairs"]) == 1
        for value in overrides.values()
    )


def test_contact_penetration_review_batch_is_five_hash_pinned_cases() -> None:
    config = _load_config(
        Path("configs/training/" "order9_c3a_contact_penetration_batch_001.yaml")
    )

    assert config["version"] == ("order9_c3a_contact_penetration_review_batch_v1")
    assert len(config["cases"]) == 5
    assert len({str(case["structural_hash"]) for case in config["cases"]}) == 5
    assert all(
        len(case["surface_port_ids"]) == 2 and case["candidate_group_id"]
        for case in config["cases"]
    )


def test_c3_overhead_start_and_bucket_support_share_scene_height(
    grasp_carry_dict: dict,
) -> None:
    task, physical, context = _system(grasp_carry_dict)
    collision = build_order9_c3_posture_collision_object(task)
    observation = build_order9_c3_neutral_runtime_observation(
        context.morphology_graph,
        physical,
        task,
    )
    support = collision.environment_boxes[0]
    support_top_z = float(support.pose_world[2]) + 0.5 * float(support.size_m[2])
    movable = next(value for value in task.scene.objects if value.movable)
    geometry = next(
        value
        for value in task.scene.geometry_library
        if value.geometry_id == movable.geometry_id
    )
    object_top_z = float(movable.pose_world[2]) + 0.5 * (
        float(geometry.primitive_params["size_m"][2]) * float(geometry.scale[2])
    )

    movable_goal = next(
        goal for goal in task.goals if goal.target_entity_id == movable.object_id
    )
    assert support.box_id == "order9_bucket_object_support"
    assert support.size_m[0] == pytest.approx(
        abs(movable_goal.target_pose_world[0] - movable.pose_world[0])
        + float(geometry.primitive_params["size_m"][0])
        + 0.05
    )
    assert collision.ground_plane_z_m == pytest.approx(0.0)
    assert all(
        state.pose_world[2]
        == pytest.approx(
            module.pose_in_design_frame[2]
            + max(object_top_z, support_top_z)
            + ORDER9_C3_OVERHEAD_START_CLEARANCE_M
        )
        for state, module in zip(
            observation.module_states,
            context.morphology_graph.modules,
            strict=True,
        )
    )


def test_nominal_bucket_index_selection_is_ordered_and_validated() -> None:
    assert _parse_bucket_indices(
        "4,5,6,13,20",
        bucket_count=28,
    ) == [4, 5, 6, 13, 20]

    with pytest.raises(ValueError, match="duplicates"):
        _parse_bucket_indices("4,4", bucket_count=28)
    with pytest.raises(ValueError, match="out-of-range"):
        _parse_bucket_indices("28", bucket_count=28)
