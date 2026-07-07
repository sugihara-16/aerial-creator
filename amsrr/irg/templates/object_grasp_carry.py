from __future__ import annotations

from amsrr.irg.templates.base import IRGBuilderContext, SceneGraph
from amsrr.schemas.common import ContactMode, SchemaValidationError
from amsrr.schemas.irg import IRGEdgeType
from amsrr.schemas.task_spec import ObjectSpec, TaskSpec, TaskType


def _target_object(task_spec: TaskSpec) -> ObjectSpec:
    objects = {obj.object_id: obj for obj in task_spec.scene.objects}
    for goal in task_spec.goals:
        if goal.goal_type == "object_pose" and goal.target_entity_id in objects:
            return objects[goal.target_entity_id]
    for obj in task_spec.scene.objects:
        if obj.movable:
            return obj
    raise SchemaValidationError("object_grasp_carry requires a movable target object")


class ObjectGraspCarryTemplate:
    task_type = TaskType.OBJECT_GRASP_CARRY

    def validate_required_fields(self, task_spec: TaskSpec, scene_graph: SceneGraph) -> None:
        target = _target_object(task_spec)
        if target.mass_kg is None:
            raise SchemaValidationError("object_grasp_carry target object requires mass_kg")
        if target.friction is None:
            raise SchemaValidationError("object_grasp_carry target object requires friction or configured default")
        if not any(goal.goal_type == "object_pose" and goal.target_entity_id == target.object_id for goal in task_spec.goals):
            raise SchemaValidationError("object_grasp_carry requires object_pose goal")

    def build(self, context: IRGBuilderContext, task_node_id: int) -> None:
        target = _target_object(context.task_spec)
        phase_ids = context.add_phase_sequence(
            task_node_id,
            [
                "approach_object",
                "establish_object_contacts",
                "apply_grasp_wrench",
                "lift_object",
                "transport_object",
                "place_object",
                "release_contacts",
            ],
        )
        region_nodes, region_ids = context.require_regions(target.object_id)
        slot = context.add_contact_slot(
            target_entity_type="object",
            target_entity_id=target.object_id,
            allowed_region_ids=region_ids,
            contact_mode=ContactMode.GRASP,
            required=True,
            min_count_group=2,
            max_count_group=4,
            required_anchor_capability={"capability_type": "grasp"},
        )
        support_slot = context.add_contact_slot(
            target_entity_type="object",
            target_entity_id=target.object_id,
            allowed_region_ids=region_ids,
            contact_mode=ContactMode.SUPPORT,
            required=False,
            min_count_group=0,
            max_count_group=2,
            required_anchor_capability={"capability_type": "support"},
        )
        for region_node_id in region_nodes.values():
            context.add_edge(region_node_id, slot, IRGEdgeType.ALLOWS)
            context.add_edge(region_node_id, support_slot, IRGEdgeType.ALLOWS)
        context.add_edge(phase_ids["establish_object_contacts"], slot, IRGEdgeType.ACTIVATES)
        context.add_edge(phase_ids["establish_object_contacts"], support_slot, IRGEdgeType.ACTIVATES)

        mass = target.mass_kg or 0.0
        gravity = abs(context.task_spec.scene.environment.gravity[2])
        payload_force = mass * gravity
        inward = context.add_wrench_requirement(
            "inward_grasp_force",
            applies_to="contact_slot",
            frame="contact_region",
            required_effect="inward_grasp_force",
            hard_or_soft="hard",
        )
        no_slip = context.add_wrench_requirement(
            "no_slip_requirement",
            applies_to="object_effect",
            frame="object",
            required_effect="frictional_no_slip_proxy",
            hard_or_soft="hard",
        )
        payload = context.add_wrench_requirement(
            "payload_support_force",
            applies_to="object_effect",
            frame="world",
            required_effect="payload_support_force",
            wrench_lower=[0.0, 0.0, payload_force, 0.0, 0.0, 0.0],
            hard_or_soft="hard",
        )
        pose_tracking = context.add_wrench_requirement(
            "object_pose_tracking_effect",
            applies_to="object_effect",
            frame="object",
            required_effect="object_pose_tracking_effect",
            hard_or_soft="soft",
        )
        for wrench in [inward, no_slip]:
            context.add_edge(slot, wrench, IRGEdgeType.REQUIRES)
        context.add_edge(support_slot, payload, IRGEdgeType.REQUIRES)
        context.add_edge(phase_ids["apply_grasp_wrench"], inward, IRGEdgeType.REQUIRES)
        context.add_edge(phase_ids["lift_object"], payload, IRGEdgeType.REQUIRES)
        context.add_edge(phase_ids["transport_object"], pose_tracking, IRGEdgeType.REQUIRES)

        object_goal = next(goal for goal in context.task_spec.goals if goal.goal_type == "object_pose")
        object_lift = context.add_state_target(
            "object_lift_height",
            target_type="object_pose",
            target_entity_id=target.object_id,
            tolerance={"height_margin_m": 0.05},
        )
        object_goal_pose = context.add_state_target(
            "object_goal_pose",
            target_type="object_pose",
            target_entity_id=target.object_id,
            pose_target_world=object_goal.target_pose_world,
            tolerance={"pos_m": object_goal.tolerance_pos_m, "rot_rad": object_goal.tolerance_rot_rad},
        )
        centroidal = context.add_state_target("centroidal_stability", target_type="centroidal", tolerance={})
        release = context.add_state_target(
            "release_contact_state",
            target_type="contact_state",
            target_entity_id=target.object_id,
            tolerance={"allow_object_drop": context.task_spec.safety.allow_object_drop},
        )
        context.add_edge(payload, object_lift, IRGEdgeType.SUPPORTS)
        context.add_edge(pose_tracking, object_goal_pose, IRGEdgeType.SUPPORTS)
        context.add_edge(inward, centroidal, IRGEdgeType.SUPPORTS)
        context.add_edge(phase_ids["release_contacts"], release, IRGEdgeType.REQUIRES)

        constraint_specs = [
            ("friction_cone", "friction_cone", {"friction": target.friction}, slot),
            ("max_contact_force", "max_contact_force", {"max_n": context.task_spec.safety.max_contact_force_n}, slot),
            ("collision_margin", "collision_margin", {"margin_m": context.task_spec.safety.collision_margin_m}, phase_ids["approach_object"]),
            ("thrust_margin", "thrust_margin", {"min_ratio": context.task_spec.safety.min_thrust_margin_ratio}, phase_ids["lift_object"]),
            ("payload_margin", "payload_margin", {"payload_force_n": payload_force}, phase_ids["transport_object"]),
        ]
        for constraint_id, constraint_type, params, target_node in constraint_specs:
            constraint = context.add_constraint(
                constraint_id,
                constraint_type=constraint_type,
                parameters=params,
                violation_code=f"E_{constraint_id.upper()}",
            )
            context.add_edge(constraint, target_node, IRGEdgeType.CONSTRAINS)

        capability = context.add_capability_requirement("grasp_capability", capability_type="grasp", min_force_n=payload_force / 2.0)
        context.add_edge(capability, slot, IRGEdgeType.APPLIES_TO)
