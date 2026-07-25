from __future__ import annotations

from types import SimpleNamespace

from amsrr.training.order9_articulated_teacher import (
    Order9ArticulatedTrajectoryTeacher,
)
from amsrr.training.order9_posture_resolver import (
    Order9PostureCollisionObject,
    Order9PostureTrajectoryResolver,
)
from tests.unit.training.test_order9_articulated_teacher import _system


class _AcceptedCollisionSolver:
    solver_version = "unit-native-convex-v1"
    collision_config = SimpleNamespace(collision_margin_m=0.005)

    def __init__(self, inner) -> None:
        self.inner = inner
        self.scene_calls: list[dict[str, object]] = []
        self.check_calls = 0

    def set_collision_scene(self, **values) -> None:
        self.scene_calls.append(dict(values))

    def solve(self, **values):
        return self.inner.solve(**values)

    def check_configuration(self, **_values):
        self.check_calls += 1
        return {
            "accepted": True,
            "minimum_clearance_m": 0.012,
            "violating_pair_count": 0,
        }


def test_resolver_installs_and_archives_convex_collision_gate(
    grasp_carry_dict: dict,
) -> None:
    task, physical, context = _system(grasp_carry_dict)
    plan = Order9ArticulatedTrajectoryTeacher(physical).plan(
        context,
        initial_object_poses_world={
            obj.object_id: obj.pose_world for obj in task.scene.objects
        },
    )
    target = task.scene.objects[0]
    geometry = next(
        item
        for item in task.scene.geometry_library
        if item.geometry_id == target.geometry_id
    )
    inner = Order9PostureTrajectoryResolver(physical).ik_solver
    solver = _AcceptedCollisionSolver(inner)
    resolver = Order9PostureTrajectoryResolver(
        physical,
        ik_solver=solver,
        collision_object=Order9PostureCollisionObject(
            object_id=target.object_id,
            size_m=tuple(geometry.primitive_params["size_m"]),
            initial_pose_world=tuple(target.pose_world),
        ),
    )
    initial_q = dict(
        plan.trajectory.knots[0].posture_target.joint_pos_target
    )

    result = resolver.resolve(
        context=context,
        raw_trajectory=plan.raw_trajectory,
        initial_joint_positions_rad=initial_q,
    )

    assert solver.scene_calls
    assert solver.check_calls == len(solver.scene_calls)
    assert result.evidence.collision_gate_status == "accepted"
    assert result.evidence.collision_gate_version == solver.solver_version
    assert result.evidence.minimum_collision_clearance_m == 0.012
    assert result.evidence.maximum_collision_violating_pair_count == 0
    assert all(
        call["object_size_m"]
        == tuple(geometry.primitive_params["size_m"])
        for call in solver.scene_calls
    )
    expected_allowed_anchor_ids = [
        tuple(
            sorted(
                assignment.anchor_id
                for assignment in knot.contact_assignments
                if assignment.schedule_state
                in {"attach", "maintain", "slide"}
            )
        )
        for knot in plan.raw_trajectory.knots
        if (
            knot.posture_target is not None
            and knot.posture_target.free_anchor_pose_targets
        )
    ]
    assert [
        call["allowed_anchor_ids"] for call in solver.scene_calls
    ] == expected_allowed_anchor_ids
