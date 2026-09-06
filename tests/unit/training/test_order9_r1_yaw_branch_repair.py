from __future__ import annotations

import math

import pytest

from amsrr.geometry.pose_math import compose_pose
from amsrr.schemas.policies import (
    CentroidalTarget,
    ContactWrenchTrajectory,
    InteractionKnot,
    ObjectTarget,
    PostureTarget,
)
from amsrr.training.order9_r1_yaw_branch_repair import (
    ORDER9_R1_PLANAR_NOMINAL_TRANSFER_VERSION,
    ORDER9_R1_PLANAR_SCENE_TRANSFER_VERSION,
    ORDER9_R1_YAW_BRANCH_REPAIR_VERSION,
    transform_order9_r1_planar_task_scene,
    transform_order9_r1_planar_trajectory,
    transform_order9_r1_scene_trajectory,
    transform_order9_r1_trajectory,
)
from amsrr.training.order9_teacher import build_order8_grasp_carry_task_spec


def test_yaw_branch_transform_moves_world_targets_but_preserves_joints() -> None:
    source = ContactWrenchTrajectory(
        horizon_s=1.0,
        dt_s=1.0,
        knots=[
            InteractionKnot(
                t_rel_s=0.0,
                contact_assignments=[],
                centroidal_target=CentroidalTarget(
                    com_pos_world=(1.0, 0.0, 0.5),
                    com_vel_world=(1.0, 0.0, 0.0),
                    body_orientation_world=(0.0, 0.0, 0.0, 1.0),
                    centroidal_wrench_preference=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                ),
                posture_target=PostureTarget(
                    joint_pos_target={"module_0:joint": 0.25},
                    joint_vel_target={"module_0:joint": -0.5},
                    free_anchor_pose_targets={3: (2.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0)},
                ),
                object_targets=[
                    ObjectTarget(
                        object_id="object",
                        pose_target_world=(1.0, 1.0, 0.2, 0.0, 0.0, 0.0, 1.0),
                        twist_target_world=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    )
                ],
            ),
            InteractionKnot(
                t_rel_s=1.0,
                contact_assignments=[],
                centroidal_target=CentroidalTarget(
                    com_pos_world=(1.0, 0.0, 0.5),
                    body_orientation_world=(0.0, 0.0, 0.0, 1.0),
                ),
                posture_target=PostureTarget(
                    joint_pos_target={"module_0:joint": 0.25},
                    joint_vel_target={"module_0:joint": 0.0},
                ),
            ),
        ],
        derived_mode_label="source",
    )
    half = math.sqrt(0.5)
    transformed = transform_order9_r1_trajectory(
        source,
        delta_pose_world=(0.0, 0.0, 0.0, 0.0, 0.0, half, half),
    )

    knot = transformed.knots[0]
    assert knot.centroidal_target is not None
    assert knot.centroidal_target.com_pos_world == pytest.approx((0.0, 1.0, 0.5))
    assert knot.centroidal_target.com_vel_world == pytest.approx((0.0, 1.0, 0.0))
    assert knot.posture_target is not None
    assert knot.posture_target.joint_pos_target == {"module_0:joint": 0.25}
    assert knot.posture_target.joint_vel_target == {"module_0:joint": -0.5}
    assert knot.posture_target.free_anchor_pose_targets is not None
    assert knot.posture_target.free_anchor_pose_targets[3][:3] == pytest.approx(
        (0.0, 2.0, 0.5)
    )
    assert knot.object_targets[0].pose_target_world[:3] == pytest.approx(
        (-1.0, 1.0, 0.2)
    )
    assert knot.object_targets[0].twist_target_world == pytest.approx(
        (0.0, 1.0, 0.0, -1.0, 0.0, 0.0)
    )
    assert ORDER9_R1_YAW_BRANCH_REPAIR_VERSION in transformed.derived_mode_label
    assert source.knots[0].centroidal_target.com_pos_world == (1.0, 0.0, 0.5)


def test_planar_transform_adds_translation_and_preserves_object_relative_path() -> None:
    source = ContactWrenchTrajectory(
        horizon_s=1.0,
        dt_s=1.0,
        knots=[
            InteractionKnot(
                t_rel_s=0.0,
                contact_assignments=[],
                centroidal_target=CentroidalTarget(
                    com_pos_world=(2.0, 1.0, 0.5),
                    body_orientation_world=(0.0, 0.0, 0.0, 1.0),
                ),
                posture_target=PostureTarget(
                    joint_pos_target={"module_0:joint": 0.25},
                    joint_vel_target={"module_0:joint": -0.5},
                    free_anchor_pose_targets={3: (2.0, 2.0, 0.5, 0.0, 0.0, 0.0, 1.0)},
                ),
                object_targets=[
                    ObjectTarget(
                        object_id="object",
                        pose_target_world=(2.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0),
                    )
                ],
            ),
            InteractionKnot(
                t_rel_s=1.0,
                contact_assignments=[],
                centroidal_target=CentroidalTarget(
                    com_pos_world=(3.0, 1.0, 0.5),
                    body_orientation_world=(0.0, 0.0, 0.0, 1.0),
                ),
                posture_target=PostureTarget(
                    joint_pos_target={"module_0:joint": 0.25},
                    joint_vel_target={"module_0:joint": 0.0},
                ),
                object_targets=[
                    ObjectTarget(
                        object_id="object",
                        pose_target_world=(3.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0),
                    )
                ],
            ),
        ],
        derived_mode_label="source",
    )
    half = math.sqrt(0.5)
    transformed = transform_order9_r1_planar_trajectory(
        source,
        translation_world=(-0.03, 0.03, 0.0),
        yaw_rotation_world=(0.0, 0.0, 0.0, 0.0, 0.0, half, half),
    )

    for source_knot, target_knot in zip(source.knots, transformed.knots):
        assert target_knot.object_targets[0].pose_target_world[:3] == pytest.approx(
            (
                source_knot.object_targets[0].pose_target_world[0] - 0.03,
                source_knot.object_targets[0].pose_target_world[1] + 0.03,
                source_knot.object_targets[0].pose_target_world[2],
            )
        )
        assert target_knot.posture_target is not None
        assert source_knot.posture_target is not None
        assert (
            target_knot.posture_target.joint_pos_target
            == source_knot.posture_target.joint_pos_target
        )
        assert (
            target_knot.posture_target.joint_vel_target
            == source_knot.posture_target.joint_vel_target
        )
    first = transformed.knots[0]
    assert first.centroidal_target is not None
    assert first.centroidal_target.com_pos_world == pytest.approx((0.97, 0.03, 0.5))
    assert first.posture_target is not None
    assert first.posture_target.free_anchor_pose_targets is not None
    assert first.posture_target.free_anchor_pose_targets[3][:3] == pytest.approx(
        (-0.03, 0.03, 0.5)
    )
    assert ORDER9_R1_PLANAR_NOMINAL_TRANSFER_VERSION in transformed.derived_mode_label


def test_scene_transfer_moves_support_task_and_trajectory_together() -> None:
    reference_task = build_order8_grasp_carry_task_spec(
        object_pose_world=(0.0, 0.0, 0.225, 0.0, 0.0, 0.0, 1.0),
        object_size_m=(0.30, 0.40, 0.15),
        object_mass_kg=1.0,
        object_friction=0.6,
        required_transport_distance_m=0.20,
        support_height_m=0.15,
        max_contact_force_n=30.0,
        max_contact_torque_nm=5.0,
    )
    angle = math.radians(5.0)
    delta = (
        0.01,
        0.01,
        0.0,
        0.0,
        0.0,
        math.sin(0.5 * angle),
        math.cos(0.5 * angle),
    )
    target_payload = reference_task.to_dict()
    target_payload["task_id"] = "scene-transfer-target"
    target_payload["scene"]["objects"][0]["pose_world"] = list(
        compose_pose(
            delta,
            tuple(target_payload["scene"]["objects"][0]["pose_world"]),
        )
    )
    target_identity = type(reference_task).from_dict(target_payload)

    transformed_task = transform_order9_r1_planar_task_scene(
        reference_task,
        target_identity_task=target_identity,
    )
    expected_support = compose_pose(
        delta,
        reference_task.scene.environment.support_surfaces[0].pose_world,
    )
    assert transformed_task.scene.environment.support_surfaces[0].pose_world == (
        pytest.approx(expected_support)
    )
    assert transformed_task.metadata["r1_support_pose_preserved"] is False
    assert transformed_task.metadata["r1_support_pose_transformed_with_scene"] is True

    trajectory = ContactWrenchTrajectory(
        horizon_s=0.1,
        dt_s=0.1,
        knots=[
            InteractionKnot(
                t_rel_s=0.0,
                contact_assignments=[],
                centroidal_target=CentroidalTarget(
                    com_pos_world=(0.0, 0.0, 0.5),
                    body_orientation_world=(0.0, 0.0, 0.0, 1.0),
                ),
                posture_target=PostureTarget(
                    joint_pos_target={"module_0:joint": 0.25},
                    joint_vel_target={"module_0:joint": 0.0},
                ),
                object_targets=[
                    ObjectTarget(
                        object_id="object",
                        pose_target_world=(0.0, 0.0, 0.225, 0.0, 0.0, 0.0, 1.0),
                    )
                ],
            )
        ],
        derived_mode_label="source",
    )
    transformed_trajectory = transform_order9_r1_scene_trajectory(
        trajectory,
        delta_pose_world=delta,
    )
    assert transformed_trajectory.knots[0].centroidal_target.com_pos_world == (
        pytest.approx((0.01, 0.01, 0.5))
    )
    assert (
        ORDER9_R1_PLANAR_SCENE_TRANSFER_VERSION
        in transformed_trajectory.derived_mode_label
    )
