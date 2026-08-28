from __future__ import annotations

import math

import pytest

from amsrr.schemas.policies import (
    CentroidalTarget,
    ContactWrenchTrajectory,
    InteractionKnot,
    ObjectTarget,
    PostureTarget,
)
from amsrr.training.order9_r1_yaw_branch_repair import (
    ORDER9_R1_YAW_BRANCH_REPAIR_VERSION,
    transform_order9_r1_trajectory,
)


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
