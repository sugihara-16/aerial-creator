from __future__ import annotations

from amsrr.irg.templates.base import IRGBuilderContext, SceneGraph
from amsrr.schemas.common import ContactMode, SchemaValidationError
from amsrr.schemas.irg import IRGEdgeType
from amsrr.schemas.task_spec import SurfaceSpec, TaskSpec, TaskType


def _perch_surface(task_spec: TaskSpec) -> SurfaceSpec:
    for surface in task_spec.scene.environment.support_surfaces:
        modes = {mode.value for mode in surface.allowed_contact_modes}
        if modes.intersection({"perch", "latch", "support"}):
            return surface
    raise SchemaValidationError("perching_manipulation requires perch/latch/support surface")


def _preferred_perch_mode(surface: SurfaceSpec) -> ContactMode:
    modes = set(surface.allowed_contact_modes)
    for mode in [ContactMode.PERCH, ContactMode.LATCH, ContactMode.SUPPORT]:
        if mode in modes:
            return mode
    return ContactMode.SUPPORT


class PerchingManipulationTemplate:
    task_type = TaskType.PERCHING_MANIPULATION

    def validate_required_fields(self, task_spec: TaskSpec, scene_graph: SceneGraph) -> None:
        _perch_surface(task_spec)

    def build(self, context: IRGBuilderContext, task_node_id: int) -> None:
        surface = _perch_surface(context.task_spec)
        contact_mode = _preferred_perch_mode(surface)
        phase_ids = context.add_phase_sequence(
            task_node_id,
            ["navigate_to_perch_region", "establish_perch_contact", "hold_perch_wrench", "optional_manipulation", "release_perch"],
        )
        region_nodes, region_ids = context.require_regions(surface.surface_id)
        slot = context.add_contact_slot(
            target_entity_type="environment",
            target_entity_id=surface.surface_id,
            allowed_region_ids=region_ids,
            contact_mode=contact_mode,
            required=True,
            min_count_group=1,
            max_count_group=2,
            required_anchor_capability={"capability_type": contact_mode.value},
        )
        for region_node_id in region_nodes.values():
            context.add_edge(region_node_id, slot, IRGEdgeType.ALLOWS)
        context.add_edge(phase_ids["establish_perch_contact"], slot, IRGEdgeType.ACTIVATES)

        hold = context.add_wrench_requirement("hold_wrench", applies_to="contact_slot", frame="world", required_effect="hold_wrench")
        slip = context.add_wrench_requirement("slip_resistance", applies_to="contact_slot", frame="contact_region", required_effect="slip_resistance")
        thrust_pref = context.add_wrench_requirement(
            "thrust_reduction_preference",
            applies_to="centroidal",
            frame="com",
            required_effect="thrust_reduction_preference",
            hard_or_soft="soft",
        )
        for wrench in [hold, slip]:
            context.add_edge(slot, wrench, IRGEdgeType.REQUIRES)
        context.add_edge(phase_ids["hold_perch_wrench"], hold, IRGEdgeType.REQUIRES)
        context.add_edge(phase_ids["hold_perch_wrench"], thrust_pref, IRGEdgeType.REQUIRES)

        body_hold = context.add_state_target("body_pose_hold", target_type="body_pose", tolerance={})
        context.add_edge(hold, body_hold, IRGEdgeType.SUPPORTS)
        context.add_edge(slip, body_hold, IRGEdgeType.SUPPORTS)
        context.add_edge(thrust_pref, body_hold, IRGEdgeType.SUPPORTS)

        for constraint_id, constraint_type, params, target_node in [
            ("max_contact_force", "max_contact_force", {"max_n": context.task_spec.safety.max_contact_force_n}, slot),
            ("latch_feasibility", "workspace", {"template_constraint": "latch_feasibility"}, slot),
            ("no_slip", "no_slip", {"friction": surface.friction}, slot),
            ("collision_margin", "collision_margin", {"margin_m": context.task_spec.safety.collision_margin_m}, phase_ids["navigate_to_perch_region"]),
        ]:
            constraint = context.add_constraint(
                constraint_id,
                constraint_type=constraint_type,
                parameters=params,
                violation_code=f"E_{constraint_id.upper()}",
            )
            context.add_edge(constraint, target_node, IRGEdgeType.CONSTRAINS)

        capability = context.add_capability_requirement(f"{contact_mode.value}_capability", capability_type=contact_mode.value)
        context.add_edge(capability, slot, IRGEdgeType.APPLIES_TO)

