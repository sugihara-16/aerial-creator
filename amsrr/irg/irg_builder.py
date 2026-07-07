from __future__ import annotations

from dataclasses import dataclass

from amsrr.geometry.geometry_processor import GeometryProcessor
from amsrr.irg.templates.base import IRGBuilderContext, InteractionTemplate, SceneEntity, SceneGraph
from amsrr.irg.templates.contact_mediated_locomotion import ContactMediatedLocomotionTemplate
from amsrr.irg.templates.free_flight import FreeFlightNavigationTemplate
from amsrr.irg.templates.object_grasp_carry import ObjectGraspCarryTemplate
from amsrr.irg.templates.perching_manipulation import PerchingManipulationTemplate
from amsrr.irg.templates.valve_operation import ValveOperationTemplate
from amsrr.irg.validator import IRGValidator
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.geometry import GeometryDescriptor
from amsrr.schemas.irg import InteractionRequirementGraph
from amsrr.schemas.task_spec import GeometrySpec, TaskSpec, TaskType


@dataclass(frozen=True)
class IRGBuilderResult:
    irg: InteractionRequirementGraph
    scene_graph: SceneGraph


class IRGBuilder:
    """Deterministic compiler from TaskSpec + GeometryDescriptor to IRG."""

    def __init__(
        self,
        *,
        geometry_processor: GeometryProcessor | None = None,
        validator: IRGValidator | None = None,
    ) -> None:
        self.geometry_processor = geometry_processor or GeometryProcessor()
        self.validator = validator or IRGValidator()
        templates: list[InteractionTemplate] = [
            FreeFlightNavigationTemplate(),
            ObjectGraspCarryTemplate(),
            ValveOperationTemplate(),
            PerchingManipulationTemplate(),
            ContactMediatedLocomotionTemplate(),
        ]
        self.templates = {template.task_type: template for template in templates}

    def build(
        self,
        task_spec: TaskSpec,
        geometry_descriptors: dict[str, GeometryDescriptor] | None = None,
    ) -> InteractionRequirementGraph:
        return self.build_with_scene_graph(task_spec, geometry_descriptors).irg

    def build_with_scene_graph(
        self,
        task_spec: TaskSpec,
        geometry_descriptors: dict[str, GeometryDescriptor] | None = None,
    ) -> IRGBuilderResult:
        scene_graph = self._build_scene_graph(task_spec, geometry_descriptors or {})
        template = self._template_for(task_spec.task_type)
        template.validate_required_fields(task_spec, scene_graph)
        context = IRGBuilderContext(task_spec, scene_graph)
        task_node_id = context.add_task_node()
        template.build(context, task_node_id)
        irg = InteractionRequirementGraph(
            irg_id=f"irg:{task_spec.task_id}",
            task_id=task_spec.task_id,
            nodes=context.nodes,
            edges=context.edges,
            metadata={
                "irg_builder_version": "p0_agent_d_v1",
                "task_type": task_spec.task_type.value,
                "geometry_ids": sorted(scene_graph.geometry_descriptors),
            },
        )
        self.validator.validate(irg, task_spec)
        return IRGBuilderResult(irg=irg, scene_graph=scene_graph)

    def _template_for(self, task_type: TaskType) -> InteractionTemplate:
        try:
            return self.templates[task_type]
        except KeyError as exc:
            raise SchemaValidationError(f"No IRG template registered for task_type {task_type.value!r}") from exc

    def _build_scene_graph(
        self,
        task_spec: TaskSpec,
        geometry_descriptors: dict[str, GeometryDescriptor],
    ) -> SceneGraph:
        descriptors = dict(geometry_descriptors)
        geometry_specs = {spec.geometry_id: spec for spec in task_spec.scene.geometry_library}
        entities: list[SceneEntity] = []

        for obj in sorted(task_spec.scene.objects, key=lambda item: item.object_id):
            entities.append(
                SceneEntity(
                    entity_id=obj.object_id,
                    entity_type="object",
                    geometry_id=obj.geometry_id,
                    contact_allowed=obj.contact_allowed,
                    allowed_contact_modes=obj.allowed_contact_modes,
                    object_spec=obj,
                )
            )
            self._ensure_descriptor(descriptors, geometry_specs, obj.geometry_id, obj.object_id, obj.friction, obj.contact_allowed, obj.allowed_contact_modes)

        for surface in sorted(task_spec.scene.environment.support_surfaces, key=lambda item: item.surface_id):
            entity_type = self._surface_entity_type(surface.surface_id)
            entities.append(
                SceneEntity(
                    entity_id=surface.surface_id,
                    entity_type=entity_type,
                    geometry_id=surface.geometry_id,
                    contact_allowed=surface.contact_allowed,
                    allowed_contact_modes=surface.allowed_contact_modes,
                    surface_spec=surface,
                )
            )
            self._ensure_descriptor(
                descriptors,
                geometry_specs,
                surface.geometry_id,
                surface.surface_id,
                surface.friction,
                surface.contact_allowed,
                surface.allowed_contact_modes,
                required=False,
            )

        for obstacle in sorted(task_spec.scene.environment.obstacles, key=lambda item: item.obstacle_id):
            entities.append(
                SceneEntity(
                    entity_id=obstacle.obstacle_id,
                    entity_type="obstacle",
                    geometry_id=obstacle.geometry_id,
                    contact_allowed=False,
                    allowed_contact_modes=[],
                )
            )
            self._ensure_descriptor(
                descriptors,
                geometry_specs,
                obstacle.geometry_id,
                obstacle.obstacle_id,
                None,
                False,
                [],
                required=False,
            )

        return SceneGraph(entities=entities, geometry_descriptors=descriptors)

    def _ensure_descriptor(
        self,
        descriptors: dict[str, GeometryDescriptor],
        geometry_specs: dict[str, GeometrySpec],
        geometry_id: str,
        entity_id: str,
        friction: float | None,
        contact_allowed: bool,
        allowed_contact_modes: list,
        required: bool = True,
    ) -> None:
        if geometry_id in descriptors:
            return
        if geometry_id not in geometry_specs:
            if not required:
                return
            raise SchemaValidationError(f"Missing GeometrySpec for geometry_id {geometry_id!r}")
        descriptors[geometry_id] = self.geometry_processor.process_geometry(
            geometry_specs[geometry_id],
            entity_id=entity_id,
            friction=friction,
            contact_allowed=contact_allowed,
            allowed_contact_modes=allowed_contact_modes,
        )

    @staticmethod
    def _surface_entity_type(surface_id: str) -> str:
        lowered = surface_id.lower()
        if "floor" in lowered:
            return "floor"
        if "wall" in lowered:
            return "wall"
        return "support_surface"
