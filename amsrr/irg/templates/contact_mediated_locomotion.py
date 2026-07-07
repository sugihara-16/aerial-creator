from __future__ import annotations

from amsrr.irg.templates.base import IRGBuilderContext, SceneGraph
from amsrr.schemas.common import ContactMode, SchemaValidationError
from amsrr.schemas.irg import IRGEdgeType
from amsrr.schemas.task_spec import SurfaceSpec, TaskSpec, TaskType


def _support_surface(task_spec: TaskSpec) -> SurfaceSpec:
    for surface in task_spec.scene.environment.support_surfaces:
        if surface.contact_allowed:
            return surface
    raise SchemaValidationError("contact_mediated_locomotion requires support surface")


class ContactMediatedLocomotionTemplate:
    task_type = TaskType.CONTACT_MEDIATED_LOCOMOTION

    def validate_required_fields(self, task_spec: TaskSpec, scene_graph: SceneGraph) -> None:
        _support_surface(task_spec)
        if not any(goal.goal_type in {"robot_pose", "centroidal_state", "object_displacement", "free_flight_pose"} for goal in task_spec.goals):
            raise SchemaValidationError("contact_mediated_locomotion requires locomotion target or displacement")

    def build(self, context: IRGBuilderContext, task_node_id: int) -> None:
        surface = _support_surface(context.task_spec)
        phase_ids = context.add_phase_sequence(
            task_node_id,
            [
                "approach_support_region",
                "establish_support_contact",
                "maintain_support",
                "shift_centroidal_state",
                "reposition_free_anchor",
                "reanchor_support",
                "release_or_continue",
            ],
        )
        region_nodes, region_ids = context.require_regions(surface.surface_id)
        slot = context.add_contact_slot(
            target_entity_type="environment",
            target_entity_id=surface.surface_id,
            allowed_region_ids=region_ids,
            contact_mode=ContactMode.SUPPORT,
            required=True,
            min_count_group=1,
            max_count_group=context.task_spec.robot_constraints.max_robot_anchors,
            required_anchor_capability={"capability_type": "support"},
        )
        for region_node_id in region_nodes.values():
            context.add_edge(region_node_id, slot, IRGEdgeType.ALLOWS)
        context.add_edge(phase_ids["establish_support_contact"], slot, IRGEdgeType.ACTIVATES)
        context.add_edge(phase_ids["reanchor_support"], slot, IRGEdgeType.ACTIVATES)

        support_force = context.add_wrench_requirement("contact_support_force", applies_to="contact_slot", frame="world", required_effect="contact_support_force")
        tangent = context.add_wrench_requirement("friction_limited_tangential_force", applies_to="contact_slot", frame="contact_region", required_effect="friction_limited_tangential_force")
        vertical_ratio = context.add_wrench_requirement("vertical_thrust_ratio_limit", applies_to="centroidal", frame="com", required_effect="vertical_thrust_ratio")
        support_ratio = context.add_wrench_requirement("contact_support_ratio_requirement", applies_to="centroidal", frame="com", required_effect="contact_support_ratio")
        for wrench in [support_force, tangent]:
            context.add_edge(slot, wrench, IRGEdgeType.REQUIRES)
        for phase_label, wrench in [
            ("maintain_support", support_force),
            ("shift_centroidal_state", support_ratio),
            ("reposition_free_anchor", vertical_ratio),
        ]:
            context.add_edge(phase_ids[phase_label], wrench, IRGEdgeType.REQUIRES)

        com_shift = context.add_state_target("com_shift", target_type="centroidal", tolerance={})
        body_pose = context.add_state_target("body_pose_stabilization", target_type="body_pose", tolerance={})
        locomotion = context.add_state_target("locomotion_progress_target", target_type="body_pose", tolerance={})
        context.add_edge(support_force, com_shift, IRGEdgeType.SUPPORTS)
        context.add_edge(tangent, locomotion, IRGEdgeType.SUPPORTS)
        context.add_edge(vertical_ratio, body_pose, IRGEdgeType.SUPPORTS)
        context.add_edge(support_ratio, com_shift, IRGEdgeType.SUPPORTS)

        for constraint_id, constraint_type, params, target_node in [
            ("no_slip", "no_slip", {"friction": surface.friction}, slot),
            ("friction_cone", "friction_cone", {"friction": surface.friction}, slot),
            ("support_ratio", "support_ratio", {"template_constraint": "support_polygon_proxy"}, phase_ids["shift_centroidal_state"]),
            ("vertical_thrust_ratio", "vertical_thrust_ratio", {"template_constraint": "vertical_thrust_ratio"}, phase_ids["reposition_free_anchor"]),
            ("collision_margin", "collision_margin", {"margin_m": context.task_spec.safety.collision_margin_m}, phase_ids["approach_support_region"]),
        ]:
            constraint = context.add_constraint(
                constraint_id,
                constraint_type=constraint_type,
                parameters=params,
                violation_code=f"E_{constraint_id.upper()}",
            )
            context.add_edge(constraint, target_node, IRGEdgeType.CONSTRAINS)

        capability = context.add_capability_requirement("support_capability", capability_type="support")
        context.add_edge(capability, slot, IRGEdgeType.APPLIES_TO)

