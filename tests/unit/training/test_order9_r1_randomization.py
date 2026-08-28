from __future__ import annotations

import math

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.order9 import Order9ArtifactBinding
from amsrr.training.order9_r1_randomization import (
    ORDER9_R1_DISTRIBUTION_MANIFEST_VERSION,
    ORDER9_R1_OBJECT_DISTRIBUTION,
    Order9R1DistributionManifest,
    Order9R1ReachablePoseEnvelope,
    Order9R1ReachablePoseRandomizer,
    load_order9_r1_distribution_manifest,
)
from amsrr.training.order9_teacher import build_order8_grasp_carry_task_spec

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64


def test_r1_distribution_requires_expansion_and_calibration_evidence() -> None:
    with pytest.raises(SchemaValidationError, match="expand beyond"):
        Order9R1ReachablePoseEnvelope(
            initial_x_offset_m=(-0.005, 0.005),
            initial_y_offset_m=(-0.005, 0.005),
            initial_yaw_offset_rad=(-math.radians(2.0), math.radians(2.0)),
        )

    invalid = _manifest()
    invalid.isaac_success_count = 58
    with pytest.raises(SchemaValidationError, match="Isaac calibration success gate"):
        invalid.validate()

    invalid = _manifest()
    invalid.confirmation_lattice_pass_count = 26
    with pytest.raises(SchemaValidationError, match="confirmation gate failed"):
        invalid.validate()

    invalid = _manifest()
    invalid.minimum_support_projected_com_margin_m = -0.001
    with pytest.raises(SchemaValidationError, match="projected-CoM support gate"):
        invalid.validate()

    invalid = _manifest()
    invalid.fast_screen_success_count = 29
    invalid.isaac_attempt_count = 58
    invalid.isaac_success_count = 58
    with pytest.raises(SchemaValidationError, match="fast exact-tracking"):
        invalid.validate()


def test_r1_manifest_round_trip_binds_accepted_bytes(tmp_path) -> None:
    path = tmp_path / "r1_distribution.json"
    path.write_text(_manifest().to_json(indent=2) + "\n", encoding="utf-8")

    loaded = load_order9_r1_distribution_manifest(path)

    assert loaded.manifest.accepted
    assert loaded.manifest.teacher_feasibility_rate == 1.0
    assert loaded.manifest.isaac_success_rate == 1.0
    assert loaded.manifest.interior_teacher_feasibility_rate == 1.0
    assert loaded.manifest.confirmation_lattice_pass_rate == 1.0
    assert len(loaded.sha256) == 64
    assert loaded.path == str(path.resolve())


def test_r1_randomizer_is_deterministic_and_pose_only(tmp_path) -> None:
    path = tmp_path / "r1_distribution.json"
    path.write_text(_manifest().to_json(indent=2) + "\n", encoding="utf-8")
    randomizer = Order9R1ReachablePoseRandomizer(
        load_order9_r1_distribution_manifest(path)
    )
    base = _task()

    first = randomizer.sample(base, seed=71, sample_index=5)
    second = randomizer.sample(base, seed=71, sample_index=5)

    assert first.to_dict() == second.to_dict()
    before_object = base.scene.objects[0]
    after_object = first.task_spec.scene.objects[0]
    before_geometry = base.scene.geometry_library[0]
    after_geometry = first.task_spec.scene.geometry_library[0]
    assert after_geometry.to_dict() == before_geometry.to_dict()
    assert after_object.mass_kg == before_object.mass_kg
    assert after_object.inertia_kgm2 == before_object.inertia_kgm2
    assert after_object.center_of_mass_object == before_object.center_of_mass_object
    assert after_object.friction == before_object.friction
    assert after_object.pose_world[2] == before_object.pose_world[2]

    before_goal = base.goals[0].target_pose_world
    after_goal = first.task_spec.goals[0].target_pose_world
    assert before_goal is not None and after_goal is not None
    assert after_goal[0] - after_object.pose_world[0] == pytest.approx(
        before_goal[0] - before_object.pose_world[0]
    )
    assert after_goal[1] - after_object.pose_world[1] == pytest.approx(
        before_goal[1] - before_object.pose_world[1]
    )
    assert first.task_spec.metadata["r1_object_properties_preserved"] is True
    assert first.task_spec.metadata["r1_goal_relative_transform_preserved"] is True
    assert first.task_spec.metadata["r1_support_pose_preserved"] is True
    assert first.task_spec.metadata["r1_robot_reset_preserved"] is True
    assert first.support_audit.minimum_projected_com_margin_m > 0.0
    assert len(first.support_audit.support_geometry_hash) == 64
    assert (
        first.task_spec.metadata["r1_support_projected_com_margin_m"]
        == first.support_audit.minimum_projected_com_margin_m
    )


def test_r1_calibration_lattice_covers_all_3x3x3_points(tmp_path) -> None:
    path = tmp_path / "r1_distribution.json"
    path.write_text(_manifest().to_json(indent=2) + "\n", encoding="utf-8")
    randomizer = Order9R1ReachablePoseRandomizer(
        load_order9_r1_distribution_manifest(path)
    )

    samples = randomizer.calibration_lattice_samples(
        _task(),
        calibration_seed=19009,
    )

    assert len(samples) == 27
    assert (
        len(
            {
                (
                    sample.initial_x_offset_m,
                    sample.initial_y_offset_m,
                    sample.initial_yaw_offset_rad,
                )
                for sample in samples
            }
        )
        == 27
    )
    assert {sample.initial_x_offset_m for sample in samples} == {-0.02, 0.0, 0.03}
    assert {sample.initial_y_offset_m for sample in samples} == {-0.01, 0.0, 0.015}
    assert {sample.initial_yaw_offset_rad for sample in samples} == {
        -math.radians(8.0),
        0.0,
        math.radians(10.0),
    }


def _manifest() -> Order9R1DistributionManifest:
    return Order9R1DistributionManifest(
        manifest_version=ORDER9_R1_DISTRIBUTION_MANIFEST_VERSION,
        distribution_id=ORDER9_R1_OBJECT_DISTRIBUTION,
        accepted=True,
        envelope=Order9R1ReachablePoseEnvelope(
            initial_x_offset_m=(-0.02, 0.03),
            initial_y_offset_m=(-0.01, 0.015),
            initial_yaw_offset_rad=(-math.radians(8.0), math.radians(10.0)),
        ),
        coordinate_frame="world",
        envelope_center="base_task_object_pose_world",
        sampling_method="independent_uniform_v1",
        calibration_seed=9009,
        calibration_gate_id="r1-calibration-gate-v1",
        accepted_level_id="r1_l2_20mm_10deg",
        calibration_gate_approval=Order9ArtifactBinding(
            artifact_kind="calibration_gate_approval",
            path="r1/calibration_gate_approval.json",
            sha256="d" * 64,
        ),
        approved_calibration_protocol=Order9ArtifactBinding(
            artifact_kind="approved_calibration_protocol",
            path="r1/calibration_protocol.yaml",
            sha256="4" * 64,
        ),
        fast_screen_version="order9_r1_exact_tracking_fast_screen_v1",
        fast_screen_execution_model=(
            "resolved_pi_h_ik_exact_tracking_to_grasp_pose_v1"
        ),
        full_control_test_requires_fast_screen_pass=True,
        calibration_minimum_lattice_pass_rate=1.0,
        calibration_minimum_teacher_feasibility_rate=0.95,
        calibration_minimum_fast_screen_pass_rate=0.95,
        calibration_minimum_isaac_success_rate=0.95,
        calibration_minimum_interior_teacher_feasibility_rate=0.95,
        calibration_minimum_interior_fast_screen_pass_rate=0.95,
        calibration_minimum_interior_isaac_success_rate=0.95,
        calibration_minimum_normalized_joint_limit_reserve=0.01,
        calibration_maximum_body_tilt_rad=math.radians(60.0),
        calibration_minimum_support_projected_com_margin_m=0.0,
        source_c3_release_id="c3_pi_l_promoted_update18_v1",
        source_c3_checkpoint_sha256=_SHA_A,
        source_c3_promotion_manifest_sha256=_SHA_B,
        curriculum_schedule_hash=_SHA_C,
        covered_module_counts=tuple(range(2, 9)),
        covered_topology_ids=tuple(f"topology-{index}" for index in range(7)),
        selection_split="train",
        selection_bucket_count=22,
        confirmation_split="validation",
        confirmation_bucket_count=14,
        lattice_case_count=27,
        lattice_pass_count=27,
        teacher_feasibility_attempt_count=31,
        teacher_feasibility_success_count=31,
        fast_screen_attempt_count=31,
        fast_screen_success_count=31,
        lattice_fast_screen_pass_count=27,
        minimum_fast_screen_normalized_joint_limit_reserve=0.10,
        maximum_fast_screen_body_tilt_rad=math.radians(30.0),
        isaac_attempt_count=62,
        isaac_success_count=62,
        safety_failure_count=0,
        fallback_count=0,
        confirmation_lattice_case_count=27,
        confirmation_lattice_pass_count=27,
        confirmation_teacher_feasibility_attempt_count=31,
        confirmation_teacher_feasibility_success_count=31,
        confirmation_fast_screen_attempt_count=31,
        confirmation_fast_screen_success_count=31,
        confirmation_lattice_fast_screen_pass_count=27,
        confirmation_minimum_fast_screen_normalized_joint_limit_reserve=0.10,
        confirmation_maximum_fast_screen_body_tilt_rad=math.radians(30.0),
        confirmation_isaac_attempt_count=62,
        confirmation_isaac_success_count=62,
        confirmation_safety_failure_count=0,
        confirmation_fallback_count=0,
        support_audit_attempt_count=31,
        minimum_support_projected_com_margin_m=0.10,
        minimum_support_full_footprint_margin_m=-0.05,
        confirmation_support_audit_attempt_count=31,
        confirmation_minimum_support_projected_com_margin_m=0.10,
        confirmation_minimum_support_full_footprint_margin_m=-0.05,
        measured_wall_time_s=120.0,
        canonical_box_geometry_preserved=True,
        object_properties_preserved=True,
        goal_relative_transform_preserved=True,
        support_geometry_preserved=True,
        support_pose_preserved=True,
        robot_reset_preserved=True,
        evidence_artifacts=[
            Order9ArtifactBinding(
                artifact_kind=kind,
                path=f"r1/{kind}.json",
                sha256=digest * 64,
            )
            for kind, digest in (
                ("calibration_config", "e"),
                ("teacher_complete_trajectory", "f"),
                ("fast_kinematic_screen", "6"),
                ("production_checker", "1"),
                ("isaac_replay", "2"),
                ("support_audit", "5"),
                ("topology_coverage", "3"),
            )
        ],
    )


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
