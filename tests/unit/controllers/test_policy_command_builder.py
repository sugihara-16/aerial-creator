from __future__ import annotations

from amsrr.controllers.policy_command_builder import PolicyCommandBiasBuilder
from amsrr.schemas.common import ContactMode
from amsrr.schemas.policies import (
    CentroidalTarget,
    ContactAssignment,
    InteractionKnot,
    PolicyCommand,
)


def test_policy_command_bias_builder() -> None:
    active_knot = InteractionKnot(
        t_rel_s=0.0,
        contact_assignments=[
            ContactAssignment(
                slot_id=2,
                anchor_id=3,
                candidate_id=7,
                contact_mode=ContactMode.GRASP,
                schedule_state="maintain",
                wrench_target=[0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
            )
        ],
        centroidal_target=CentroidalTarget(
            centroidal_wrench_preference=[0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
        ),
        priority_weights={"trajectory": 1.0, "posture": 0.5},
    )
    command = PolicyCommand(
        desired_body_twist=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        joint_position_bias={"gimbal1": 0.3, "gimbal2": -0.2},
        joint_velocity_bias={"gimbal1": 0.4},
        residual_wrench_body=[0.0, 0.0, 1.5, 0.0, 0.0, 0.1],
        contact_tracking_bias={7: [0.0, 0.0, 0.2, 0.0, 0.0, 0.0]},
        priority_weights={"posture": 0.9, "contact": 2.0},
    )

    refs = PolicyCommandBiasBuilder().build(
        command,
        active_knot,
        nominal_joint_positions={"gimbal1": 0.8, "gimbal2": 0.0},
        nominal_joint_velocities={"gimbal1": 0.1},
        joint_limits={"gimbal1": (-1.0, 1.0), "gimbal2": (-1.0, 1.0)},
        velocity_limits={"gimbal1": (-0.2, 0.2)},
    )

    assert refs.joint_position_ref == {"gimbal1": 1.0, "gimbal2": -0.2}
    assert refs.joint_velocity_ref == {"gimbal1": 0.2}
    assert refs.desired_wrench_body == [0.0, 0.0, 11.5, 0.0, 0.0, 0.1]
    assert refs.desired_body_twist == command.desired_body_twist
    assert refs.priority_weights == {"trajectory": 1.0, "posture": 0.9, "contact": 2.0}
    assert refs.contact_tracking_refs[7]["slot_id"] == 2
    assert refs.contact_tracking_refs[7]["tracking_bias"] == [0.0, 0.0, 0.2, 0.0, 0.0, 0.0]
    assert not hasattr(refs, "rotor_thrusts_n")
