from __future__ import annotations

from amsrr.irg.templates.base import IRGBuilderContext, SceneGraph
from amsrr.schemas.common import ContactMode, SchemaValidationError
from amsrr.schemas.irg import IRGEdgeType
from amsrr.schemas.task_spec import ObjectSpec, TaskSpec, TaskType


def _valve_object(task_spec: TaskSpec) -> ObjectSpec:
    for obj in task_spec.scene.objects:
        if obj.kinematic_model is not None or "valve" in obj.semantic_tags:
            return obj
    raise SchemaValidationError("valve_operation requires valve object with kinematic model")


class ValveOperationTemplate:
    task_type = TaskType.VALVE_OPERATION

    def validate_required_fields(self, task_spec: TaskSpec, scene_graph: SceneGraph) -> None:
        valve = _valve_object(task_spec)
        if valve.kinematic_model is None or valve.kinematic_model.axis_world is None:
            raise SchemaValidationError("valve_operation requires valve axis / kinematic joint model")
        if not any(goal.goal_type == "object_joint_state" for goal in task_spec.goals):
            raise SchemaValidationError("valve_operation requires object_joint_state goal")

    def build(self, context: IRGBuilderContext, task_node_id: int) -> None:
        valve = _valve_object(context.task_spec)
        phase_ids = context.add_phase_sequence(
            task_node_id,
            ["approach_valve", "establish_valve_contact", "apply_tangential_wrench", "rotate_valve", "release_contact"],
        )
        region_nodes, region_ids = context.require_regions(valve.object_id)
        slot = context.add_contact_slot(
            target_entity_type="object",
            target_entity_id=valve.object_id,
            allowed_region_ids=region_ids,
            contact_mode=ContactMode.PUSH,
            required=True,
            min_count_group=1,
            max_count_group=2,
            required_anchor_capability={"capability_type": "push"},
        )
        for region_node_id in region_nodes.values():
            context.add_edge(region_node_id, slot, IRGEdgeType.ALLOWS)
        context.add_edge(phase_ids["establish_valve_contact"], slot, IRGEdgeType.ACTIVATES)

        tangential = context.add_wrench_requirement(
            "tangential_force",
            applies_to="contact_slot",
            frame="contact_region",
            required_effect="tangential_force",
        )
        axis_torque = context.add_wrench_requirement(
            "valve_axis_torque",
            applies_to="object_effect",
            frame="joint_axis",
            required_effect="valve_axis_torque",
            wrench_lower=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        context.add_edge(slot, tangential, IRGEdgeType.REQUIRES)
        context.add_edge(slot, axis_torque, IRGEdgeType.REQUIRES)
        context.add_edge(phase_ids["apply_tangential_wrench"], tangential, IRGEdgeType.REQUIRES)
        context.add_edge(phase_ids["rotate_valve"], axis_torque, IRGEdgeType.REQUIRES)

        joint_goal = next(goal for goal in context.task_spec.goals if goal.goal_type == "object_joint_state")
        valve_q = context.add_state_target(
            "valve_q_target",
            target_type="object_joint_state",
            target_entity_id=valve.object_id,
            q_target=joint_goal.target_q,
            tolerance={"q": joint_goal.tolerance_q},
        )
        body_stabilization = context.add_state_target("body_stabilization", target_type="body_pose", tolerance={})
        context.add_edge(axis_torque, valve_q, IRGEdgeType.SUPPORTS)
        context.add_edge(tangential, body_stabilization, IRGEdgeType.SUPPORTS)

        constraint_specs = [
            ("friction_cone", "friction_cone", {"friction": valve.friction}, slot),
            ("maintain_contact", "no_slip", {"template_constraint": "maintain_contact"}, slot),
            ("collision_margin", "collision_margin", {"margin_m": context.task_spec.safety.collision_margin_m}, phase_ids["approach_valve"]),
            ("thrust_margin", "thrust_margin", {"min_ratio": context.task_spec.safety.min_thrust_margin_ratio}, phase_ids["rotate_valve"]),
        ]
        for constraint_id, constraint_type, params, target_node in constraint_specs:
            constraint = context.add_constraint(
                constraint_id,
                constraint_type=constraint_type,
                parameters=params,
                violation_code=f"E_{constraint_id.upper()}",
            )
            context.add_edge(constraint, target_node, IRGEdgeType.CONSTRAINS)

        capability = context.add_capability_requirement("push_capability", capability_type="push", min_force_n=1.0)
        context.add_edge(capability, slot, IRGEdgeType.APPLIES_TO)

