from __future__ import annotations

from amsrr.irg.templates.base import IRGBuilderContext, SceneGraph
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.irg import IRGEdgeType
from amsrr.schemas.task_spec import TaskSpec, TaskType


class FreeFlightNavigationTemplate:
    task_type = TaskType.FREE_FLIGHT_NAVIGATION

    def validate_required_fields(self, task_spec: TaskSpec, scene_graph: SceneGraph) -> None:
        if not any(goal.goal_type in {"robot_pose", "free_flight_pose"} for goal in task_spec.goals):
            raise SchemaValidationError("free_flight_navigation requires robot_pose or free_flight_pose goal")

    def build(self, context: IRGBuilderContext, task_node_id: int) -> None:
        phase_ids = context.add_phase_sequence(task_node_id, ["takeoff_or_stabilize", "navigate", "hold_or_land"])
        goal = next(goal for goal in context.task_spec.goals if goal.goal_type in {"robot_pose", "free_flight_pose"})
        wrench = context.add_wrench_requirement(
            "centroidal_wrench_for_free_flight",
            applies_to="centroidal",
            frame="com",
            required_effect="free_flight_tracking",
            hard_or_soft="soft",
        )
        robot_pose_target = context.add_state_target(
            "robot_pose_target",
            target_type="body_pose",
            target_entity_id=goal.target_entity_id,
            pose_target_world=goal.target_pose_world,
            tolerance={"pos_m": goal.tolerance_pos_m, "rot_rad": goal.tolerance_rot_rad},
        )
        body_orientation_target = context.add_state_target(
            "body_orientation_target",
            target_type="body_pose",
            target_entity_id=goal.target_entity_id,
            pose_target_world=goal.target_pose_world,
            tolerance={"rot_rad": goal.tolerance_rot_rad},
        )
        context.add_edge(phase_ids["navigate"], wrench, IRGEdgeType.REQUIRES)
        context.add_edge(phase_ids["navigate"], robot_pose_target, IRGEdgeType.REQUIRES)
        context.add_edge(wrench, robot_pose_target, IRGEdgeType.SUPPORTS)
        context.add_edge(wrench, body_orientation_target, IRGEdgeType.SUPPORTS)

        for constraint_id, constraint_type, params, target in [
            (
                "collision_margin",
                "collision_margin",
                {"margin_m": context.task_spec.safety.collision_margin_m},
                phase_ids["navigate"],
            ),
            (
                "thrust_margin",
                "thrust_margin",
                {"min_thrust_margin_ratio": context.task_spec.safety.min_thrust_margin_ratio},
                phase_ids["navigate"],
            ),
            (
                "max_tilt_as_workspace",
                "workspace",
                {"template_constraint": "max_tilt", "max_tilt_rad": context.task_spec.safety.max_tilt_rad},
                phase_ids["navigate"],
            ),
        ]:
            constraint = context.add_constraint(
                constraint_id,
                constraint_type=constraint_type,
                parameters=params,
                violation_code=f"E_{constraint_id.upper()}",
            )
            context.add_edge(constraint, target, IRGEdgeType.CONSTRAINS)

