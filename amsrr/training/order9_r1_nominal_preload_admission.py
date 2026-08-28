from __future__ import annotations

"""Lightweight R1 admission for the execution-side nominal preload."""

from dataclasses import dataclass

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.order9_actuator_aware_nominal_preload import (
    Order9ActuatorAwareNominalPreloadConfig,
    solve_order9_actuator_aware_nominal_preload,
)
from amsrr.simulation.order9_object_task_runtime import Order9ObjectTaskPhase
from amsrr.training.order9_c3_nominal_trajectory import (
    Order9C3AcceptedNominalBundle,
    Order9C3NominalTrajectory,
)
from amsrr.training.order9_c3_teacher import (
    build_order9_c3_posture_collision_object,
)
from amsrr.training.order9_posture_resolver import (
    Order9PostureResolverConfig,
    Order9PostureTrajectoryResolver,
)
from amsrr.training.order9_virtual_contact_compression import (
    limit_order9_virtual_contact_compression_by_collision,
    solve_order9_virtual_contact_compression_for_achieved_lead,
)


ORDER9_R1_NOMINAL_PRELOAD_ADMISSION_VERSION = (
    "order9_r1_nominal_preload_collision_admission_v1"
)


@dataclass(frozen=True)
class Order9R1NominalPreloadAdmissionConfig:
    minimum_inward_lead_m: float
    maximum_inward_lead_m: float
    inward_lead_quantization_m: float
    support_safety_factor: float
    maximum_peak_effort_utilization: float
    minimum_anchor_force_fraction: float
    model_error_margin_m: float
    collision_margin_m: float = 0.001
    maximum_action_joint_delta_rad: float = 0.15


@dataclass(frozen=True)
class Order9R1NominalPreloadAdmission:
    requested_inward_lead_m: float
    compliance_predicted_inward_lead_m: float
    achieved_inward_lead_m: float
    collision_safe_scale: float
    collision_check_count: int
    limiting_scene_id: str | None
    version: str = ORDER9_R1_NOMINAL_PRELOAD_ADMISSION_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "requested_inward_lead_m": self.requested_inward_lead_m,
            "compliance_predicted_inward_lead_m": (
                self.compliance_predicted_inward_lead_m
            ),
            "achieved_inward_lead_m": self.achieved_inward_lead_m,
            "collision_safe_scale": self.collision_safe_scale,
            "collision_check_count": self.collision_check_count,
            "limiting_scene_id": self.limiting_scene_id,
        }


def admit_order9_r1_nominal_preload(
    *,
    nominal: Order9C3NominalTrajectory | Order9C3AcceptedNominalBundle,
    task_spec: TaskSpec,
    physical_model: PhysicalModel,
    contact_friction: float,
    contact_stiffness_n_per_m: float,
    config: Order9R1NominalPreloadAdmissionConfig,
) -> Order9R1NominalPreloadAdmission:
    """Reproduce the runtime preload collision gate without starting Isaac."""

    movable = [value for value in task_spec.scene.objects if value.movable]
    if len(movable) != 1:
        raise SchemaValidationError(
            "R1 nominal preload admission requires one movable object"
        )
    if isinstance(nominal, Order9C3AcceptedNominalBundle):
        morphology = nominal.morphology
        candidate_set = nominal.contact_candidate_set
        contact_knot = nominal.phase_trajectories[
            Order9ObjectTaskPhase.CONTACT_ACQUISITION.value
        ].knots[-1]
    else:
        selection = nominal.selection_bundle
        morphology = selection.design_output.target_morphology
        candidate_set = selection.contact_candidate_set
        contact_knot = selection.trajectory.knots[-1]
    actuator = solve_order9_actuator_aware_nominal_preload(
        morphology=morphology,
        physical_model=physical_model,
        contact_knot=contact_knot,
        candidate_set=candidate_set,
        object_mass_kg=float(movable[0].mass_kg),
        contact_friction=float(contact_friction),
        contact_stiffness_n_per_m=float(contact_stiffness_n_per_m),
        config=Order9ActuatorAwareNominalPreloadConfig(
            minimum_inward_lead_m=float(config.minimum_inward_lead_m),
            maximum_inward_lead_m=float(config.maximum_inward_lead_m),
            inward_lead_quantization_m=float(config.inward_lead_quantization_m),
            support_safety_factor=float(config.support_safety_factor),
            maximum_peak_effort_utilization=float(
                config.maximum_peak_effort_utilization
            ),
            minimum_anchor_force_fraction=float(
                config.minimum_anchor_force_fraction
            ),
            model_error_margin_m=float(config.model_error_margin_m),
        ),
    )
    if not actuator.feasible:
        raise SchemaValidationError(
            "R1 nominal preload actuator admission failed: "
            + str(actuator.rejection_reason)
        )
    solution = solve_order9_virtual_contact_compression_for_achieved_lead(
        morphology=morphology,
        physical_model=physical_model,
        contact_knot=contact_knot,
        candidate_set=candidate_set,
        minimum_achieved_inward_lead_m=float(actuator.inward_lead_m),
        maximum_requested_inward_lead_m=float(config.maximum_inward_lead_m),
        requested_lead_quantization_m=float(config.inward_lead_quantization_m),
    )
    collision_object = build_order9_c3_posture_collision_object(task_spec)
    resolver = Order9PostureTrajectoryResolver(
        physical_model,
        config=Order9PostureResolverConfig(
            collision_margin_m=float(config.collision_margin_m)
        ),
        collision_object=collision_object,
        prefer_native_solver=True,
        require_native_solver=True,
    )
    collision_limit = limit_order9_virtual_contact_compression_by_collision(
        morphology=morphology,
        physical_model=physical_model,
        contact_knot=contact_knot,
        solution=solution,
        collision_object=collision_object,
        collision_solver=resolver.ik_solver,
        collision_margin_m=float(config.collision_margin_m),
        maximum_action_joint_delta_rad=float(
            config.maximum_action_joint_delta_rad
        ),
    )
    scale = float(collision_limit.maximum_action_scale)
    achieved = scale * min(
        float(value) for value in solution.achieved_inward_displacement_m.values()
    )
    required = max(
        float(config.minimum_inward_lead_m),
        float(actuator.compliance_predicted_inward_lead_m),
    )
    if achieved + 1.0e-9 < required:
        raise SchemaValidationError(
            "R1 nominal preload collision admission failed"
        )
    return Order9R1NominalPreloadAdmission(
        requested_inward_lead_m=float(actuator.inward_lead_m),
        compliance_predicted_inward_lead_m=(
            float(actuator.compliance_predicted_inward_lead_m)
        ),
        achieved_inward_lead_m=achieved,
        collision_safe_scale=scale,
        collision_check_count=int(collision_limit.collision_check_count),
        limiting_scene_id=collision_limit.limiting_scene_id,
    )


__all__ = [
    "ORDER9_R1_NOMINAL_PRELOAD_ADMISSION_VERSION",
    "Order9R1NominalPreloadAdmission",
    "Order9R1NominalPreloadAdmissionConfig",
    "admit_order9_r1_nominal_preload",
]
