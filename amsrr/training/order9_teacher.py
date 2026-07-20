from __future__ import annotations

"""Order 8 natural-contact teacher adapter for Order 9 supervision."""

import itertools
import math
from dataclasses import dataclass

from amsrr.feasibility.contact_wrench_trajectory import (
    ContactWrenchTrajectoryFeasibilityChecker,
)
from amsrr.geometry.contact_material import with_selected_robot_contact_material
from amsrr.geometry.wrench import (
    assignment_wrench_target_world,
    contact_wrench_to_world,
    world_wrench_to_contact,
)
from amsrr.geometry.mass_properties import cuboid_mass_properties
from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.policies.deterministic_natural_contact_planner import (
    ORDER8_DETERMINISTIC_PI_H_VERSION,
    DeterministicNaturalContactPlanner,
    NaturalContactPlannerFeedback,
)
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.schemas.common import ContactMode, Pose7D, SchemaValidationError
from amsrr.schemas.contact_candidates import ContactCandidate, ContactCandidateSet
from amsrr.schemas.datasets import (
    DatasetSplit,
    InteractionTrajectoryRecord,
    StageDecisionMasks,
    TrajectoryProvenance,
    TrajectorySourceKind,
)
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.policies import (
    CONTACT_WRENCH_CONTRACT_CONTACT_FRAME,
    ContactAssignment,
    ContactWrenchTrajectory,
    InteractionKnot,
)
from amsrr.schemas.runtime import RuntimeObservation
from amsrr.schemas.task_spec import (
    CollisionModel,
    EnvironmentSpec,
    GeometrySpec,
    GeometryType,
    GoalSpec,
    ObjectSpec,
    RobotConstraints,
    SafetySpec,
    SceneSpec,
    SurfaceSpec,
    TaskSpec,
    TaskType,
)


ORDER9_NATURAL_CONTACT_TEACHER_VERSION = "order9_natural_contact_teacher_v3_qp_ranges"
ORDER9_NATURAL_CONTACT_FALLBACK_VERSION = (
    "order9_natural_contact_fallback_v1_checked_v2_contract"
)


@dataclass(frozen=True)
class TeacherWrenchEnvelopeConfig:
    force_window_fraction: float = 0.20
    minimum_force_window_n: float = 0.50
    friction_capacity_window_fraction: float = 0.80
    torque_window_fraction: float = 0.20
    minimum_torque_window_nm: float = 0.05
    default_max_force_n: float = 30.0
    default_max_torque_nm: float = 5.0
    cone_tolerance_n: float = 1.0e-9
    max_cone_shrink_iterations: int = 24

    def __post_init__(self) -> None:
        for name in (
            "force_window_fraction",
            "minimum_force_window_n",
            "friction_capacity_window_fraction",
            "torque_window_fraction",
            "minimum_torque_window_nm",
            "default_max_force_n",
            "default_max_torque_nm",
            "cone_tolerance_n",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"TeacherWrenchEnvelopeConfig.{name} must be finite and non-negative")
        if self.default_max_force_n <= 0.0 or self.default_max_torque_nm <= 0.0:
            raise ValueError("teacher wrench limits must be positive")
        if self.friction_capacity_window_fraction > 1.0:
            raise ValueError(
                "friction_capacity_window_fraction must not exceed one"
            )
        if self.max_cone_shrink_iterations < 1:
            raise ValueError("max_cone_shrink_iterations must be positive")


class Order9NaturalContactTeacher:
    """Reuse Order 8 behavior while emitting the production v2 policy contract."""

    teacher_version = ORDER9_NATURAL_CONTACT_TEACHER_VERSION

    def __init__(
        self,
        planner: DeterministicNaturalContactPlanner,
        *,
        wrench_config: TeacherWrenchEnvelopeConfig | None = None,
    ) -> None:
        self.planner = planner
        self.wrench_config = wrench_config or TeacherWrenchEnvelopeConfig()

    def observe(self, feedback: NaturalContactPlannerFeedback) -> None:
        self.planner.observe(feedback)

    def teach(self, context: HighLevelPolicyContext) -> ContactWrenchTrajectory:
        legacy = self.planner.plan(context)
        return upgrade_teacher_trajectory_to_v2(
            legacy,
            context,
            config=self.wrench_config,
        )


class Order9NaturalContactFallback:
    """Separate deterministic fallback using the same checked v2 adapter."""

    fallback_version = ORDER9_NATURAL_CONTACT_FALLBACK_VERSION

    def __init__(self, teacher: Order9NaturalContactTeacher) -> None:
        self.teacher = teacher

    def fallback(
        self,
        context: HighLevelPolicyContext,
    ) -> ContactWrenchTrajectory:
        return self.teacher.teach(context)


def upgrade_teacher_trajectory_to_v2(
    trajectory: ContactWrenchTrajectory,
    context: HighLevelPolicyContext,
    *,
    config: TeacherWrenchEnvelopeConfig | None = None,
) -> ContactWrenchTrajectory:
    """Convert world-frame Order 8 labels without changing their physical target."""

    envelope_config = config or TeacherWrenchEnvelopeConfig()
    converted = ContactWrenchTrajectory.from_dict(trajectory.to_dict())
    candidate_by_id = {
        candidate.candidate_id: candidate
        for candidate in context.contact_candidate_set.candidates
    }
    anchor_by_id = {
        anchor.anchor_id: anchor for anchor in context.morphology_graph.robot_anchors
    }
    for knot in converted.knots:
        for assignment in knot.contact_assignments:
            candidate = candidate_by_id.get(assignment.candidate_id)
            if candidate is None:
                raise SchemaValidationError(
                    "teacher trajectory references an unknown contact candidate"
                )
            if assignment.wrench_frame != "world":
                raise SchemaValidationError(
                    "Order 9 teacher upgrade expects an explicit legacy world wrench"
                )
            active = assignment.schedule_state in {"attach", "maintain", "slide"}
            if not active:
                assignment.wrench_target = None
                assignment.wrench_lower = None
                assignment.wrench_upper = None
                assignment.wrench_frame = "contact"
                continue
            world_target = assignment_wrench_target_world(assignment, candidate)
            if world_target is None:
                world_target = [0.0] * 6
            target = world_wrench_to_contact(world_target, candidate)
            anchor = anchor_by_id.get(assignment.anchor_id)
            capability = {} if anchor is None else anchor.capability
            max_force_n = _positive_capability_limit(
                capability.get("max_force_n"),
                envelope_config.default_max_force_n,
            )
            max_torque_nm = _positive_capability_limit(
                capability.get("max_torque_nm"),
                envelope_config.default_max_torque_nm,
            )
            force_lower, force_upper = _friction_safe_force_box(
                target[:3],
                candidate,
                max_force_n=max_force_n,
                config=envelope_config,
            )
            torque_lower, torque_upper = _bounded_axis_window(
                target[3:],
                maximum=max_torque_nm,
                fraction=envelope_config.torque_window_fraction,
                minimum_window=envelope_config.minimum_torque_window_nm,
            )
            assignment.wrench_target = target
            assignment.wrench_lower = [*force_lower, *torque_lower]
            assignment.wrench_upper = [*force_upper, *torque_upper]
            assignment.wrench_frame = "contact"
    converted.contract_version = CONTACT_WRENCH_CONTRACT_CONTACT_FRAME
    converted.derived_mode_label = (
        f"{ORDER9_NATURAL_CONTACT_TEACHER_VERSION}:"
        f"source={trajectory.derived_mode_label or ORDER8_DETERMINISTIC_PI_H_VERSION}"
    )
    return ContactWrenchTrajectory.from_dict(converted.to_dict())


def rolling_teacher_snapshot_to_v2(
    trajectory: ContactWrenchTrajectory,
    context: HighLevelPolicyContext,
    *,
    decision_dt_s: float,
    phase_label: str,
    config: TeacherWrenchEnvelopeConfig | None = None,
) -> ContactWrenchTrajectory:
    """Turn the Order 8 rolling first knot into one checked decision snapshot.

    Order 8 declares a receding horizon but emits only its current knot.  C_H's
    coverage rule intentionally rejects such a partial horizon.  A source
    snapshot therefore has a one-decision horizon; full Order 9 supervision is
    subsequently composed from consecutive snapshots by
    :func:`compose_order9_teacher_windows`.
    """

    dt_s = float(decision_dt_s)
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise SchemaValidationError("teacher decision_dt_s must be finite and positive")
    if not phase_label:
        raise SchemaValidationError("teacher snapshot phase_label must be non-empty")
    converted = upgrade_teacher_trajectory_to_v2(
        trajectory,
        context,
        config=config,
    )
    if not converted.knots:
        raise SchemaValidationError("rolling teacher snapshot requires a current knot")
    knot = InteractionKnot.from_dict(converted.knots[0].to_dict())
    knot.t_rel_s = 0.0
    knot.guard_conditions = [
        guard
        for guard in knot.guard_conditions
        if guard.get("type") != "order9_task_phase"
    ]
    knot.guard_conditions.append(
        {"type": "order9_task_phase", "phase_label": str(phase_label)}
    )
    snapshot = ContactWrenchTrajectory(
        horizon_s=dt_s,
        dt_s=dt_s,
        knots=[knot],
        derived_mode_label=(
            f"{converted.derived_mode_label}:rolling_snapshot_dt={dt_s:.9g}"
        ),
        contract_version=CONTACT_WRENCH_CONTRACT_CONTACT_FRAME,
    )
    snapshot.validate()
    return snapshot


def build_order8_grasp_carry_task_spec(
    *,
    object_pose_world: Pose7D,
    object_size_m: tuple[float, float, float],
    object_mass_kg: float,
    object_friction: float,
    required_transport_distance_m: float,
    support_height_m: float,
    max_contact_force_n: float,
    max_contact_torque_nm: float,
    task_id: str = "order8-natural-contact-smoke",
    object_id: str = "order8_object",
    selected_gripper_friction: float = 4.5,
    friction_combine_mode: str = "max",
) -> TaskSpec:
    """Represent the Order 8 smoke through the normal TaskSpec→IRG path."""

    size = tuple(float(value) for value in object_size_m)
    if len(size) != 3 or any(value <= 0.0 for value in size):
        raise SchemaValidationError("Order 8 learning object size must contain three positive values")
    density = float(object_mass_kg) / (size[0] * size[1] * size[2])
    mass_properties = cuboid_mass_properties(size, density_kg_m3=density)
    goal_pose: Pose7D = (
        float(object_pose_world[0]) + float(required_transport_distance_m),
        float(object_pose_world[1]),
        float(object_pose_world[2]),
        *tuple(float(value) for value in object_pose_world[3:7]),
    )
    object_geometry_id = f"{object_id}:geometry"
    support_geometry_id = "order8_support:geometry"
    support_thickness_m = 0.05
    return TaskSpec(
        task_id=task_id,
        task_type=TaskType.OBJECT_GRASP_CARRY,
        scene=SceneSpec(
            geometry_library=[
                GeometrySpec(
                    geometry_id=object_geometry_id,
                    geometry_type=GeometryType.BOX,
                    primitive_params={"size_m": list(size)},
                    asset_path=None,
                    collision_model=CollisionModel.PRIMITIVE,
                ),
                GeometrySpec(
                    geometry_id=support_geometry_id,
                    geometry_type=GeometryType.BOX,
                    primitive_params={"size_m": [2.0, 2.0, support_thickness_m]},
                    asset_path=None,
                    collision_model=CollisionModel.PRIMITIVE,
                ),
            ],
            objects=[
                ObjectSpec(
                    object_id=object_id,
                    geometry_id=object_geometry_id,
                    pose_world=object_pose_world,
                    movable=True,
                    mass_kg=float(object_mass_kg),
                    inertia_kgm2=mass_properties.inertia_kgm2,
                    friction=float(object_friction),
                    material_tag="order8_natural_contact_object",
                    allowed_contact_modes=[ContactMode.GRASP, ContactMode.SUPPORT],
                    center_of_mass_object=mass_properties.center_of_mass_object,
                    density_kg_m3=mass_properties.density_kg_m3,
                )
            ],
            environment=EnvironmentSpec(
                support_surfaces=[
                    SurfaceSpec(
                        surface_id="order8_support",
                        geometry_id=support_geometry_id,
                        pose_world=(
                            0.0,
                            0.0,
                            float(support_height_m) - support_thickness_m / 2.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                        ),
                        allowed_contact_modes=[ContactMode.SUPPORT],
                        friction=0.8,
                    )
                ],
                obstacles=[],
            ),
        ),
        goals=[
            GoalSpec(
                goal_id="order8_object_goal",
                goal_type="object_pose",
                time_limit_s=150.0,
                target_entity_id=object_id,
                target_pose_world=goal_pose,
                tolerance_pos_m=0.05,
                tolerance_rot_rad=0.2,
            )
        ],
        robot_constraints=RobotConstraints(
            min_modules=1,
            max_modules=8,
            allowed_module_types=["holon"],
            max_robot_anchors=16,
        ),
        safety=SafetySpec(
            collision_margin_m=0.03,
            max_contact_force_n=float(max_contact_force_n),
            max_contact_torque_nm=float(max_contact_torque_nm),
            allow_object_drop=False,
        ),
        curriculum_tags=["order8_anchor", "order9_teacher"],
        metadata=with_selected_robot_contact_material(
            {
                "source_phase": "P4-full-order8-natural-contact",
                "teacher_version": ORDER9_NATURAL_CONTACT_TEACHER_VERSION,
            },
            target_entity_ids=[object_id],
            contact_modes=[ContactMode.GRASP],
            robot_static_friction=float(selected_gripper_friction),
            robot_dynamic_friction=float(selected_gripper_friction),
            friction_combine_mode=friction_combine_mode,
            robot_surface_scope="selected_grasp_anchor_surfaces",
        ),
    )


def compile_high_level_context(
    task_spec: TaskSpec,
    morphology_graph: MorphologyGraph,
    contact_candidate_set: ContactCandidateSet,
    *,
    runtime_observation: RuntimeObservation | None = None,
) -> HighLevelPolicyContext:
    if contact_candidate_set.task_id != task_spec.task_id:
        raise SchemaValidationError(
            "contact candidate set task_id must match the compiled TaskSpec"
        )
    if contact_candidate_set.morphology_graph_id != morphology_graph.graph_id:
        raise SchemaValidationError(
            "contact candidate set morphology_graph_id must match morphology"
        )
    if (
        runtime_observation is not None
        and runtime_observation.morphology_graph.graph_id != morphology_graph.graph_id
    ):
        raise SchemaValidationError(
            "runtime observation morphology must match high-level context"
        )
    irg = IRGBuilder().build(task_spec)
    envelope = InteractionEnvelopeExtractor().extract(irg)
    return HighLevelPolicyContext(
        irg=irg,
        interaction_envelope=envelope,
        morphology_graph=morphology_graph,
        contact_candidate_set=contact_candidate_set,
        runtime_observation=runtime_observation,
    )


def teacher_interaction_record(
    *,
    record_id: str,
    episode_id: str,
    split: DatasetSplit,
    decision_index: int,
    context: HighLevelPolicyContext,
    trajectory: ContactWrenchTrajectory,
    checker: ContactWrenchTrajectoryFeasibilityChecker,
    decision_return: float = 0.0,
    teacher_version: str = ORDER9_NATURAL_CONTACT_TEACHER_VERSION,
) -> InteractionTrajectoryRecord:
    observation = context.runtime_observation
    if observation is None:
        raise SchemaValidationError(
            "teacher InteractionTrajectoryRecord requires a RuntimeObservation"
        )
    feasibility = checker.check(trajectory, context)
    if not feasibility.feasible:
        codes = sorted({item.code for item in feasibility.hard_violations})
        raise SchemaValidationError(
            "teacher trajectory failed C_H and cannot enter the accepted dataset: "
            + ",".join(codes)
        )
    selected_ids = sorted(
        {
            assignment.candidate_id
            for knot in trajectory.knots
            for assignment in knot.contact_assignments
        }
    )
    assignment_results = [
        knot_result.assignment_result for knot_result in feasibility.knot_results
    ]
    return InteractionTrajectoryRecord(
        record_id=record_id,
        episode_id=episode_id,
        task_id=context.irg.task_id,
        split=split,
        decision_index=decision_index,
        decision_time_s=observation.time_s,
        irg=context.irg,
        interaction_envelope=context.interaction_envelope,
        morphology_graph=context.morphology_graph,
        contact_candidate_set=context.contact_candidate_set,
        runtime_observation=observation,
        trajectory=trajectory,
        selected_candidate_ids=selected_ids,
        assignment_feasibility_results=assignment_results,
        decision_return=float(decision_return),
        stage_masks=StageDecisionMasks(high_level_decision_mask=True),
        trajectory_provenance=TrajectoryProvenance(
            source_kind=TrajectorySourceKind.DETERMINISTIC_TEACHER,
            source_version=teacher_version,
            metadata={"accepted_by_checker": feasibility.checker_version},
        ),
        trajectory_feasibility_result=feasibility,
    )


def cuboid_inertia_tensor6(
    mass_kg: float,
    size_m: tuple[float, float, float],
) -> list[float]:
    mass = float(mass_kg)
    x, y, z = (float(value) for value in size_m)
    if not math.isfinite(mass) or mass <= 0.0:
        raise SchemaValidationError("cuboid mass must be finite and positive")
    if any(not math.isfinite(value) or value <= 0.0 for value in (x, y, z)):
        raise SchemaValidationError("cuboid size must be finite and positive")
    density = mass / (x * y * z)
    return cuboid_mass_properties(
        (x, y, z),
        density_kg_m3=density,
    ).inertia_kgm2


def _friction_safe_force_box(
    target_contact: list[float],
    candidate: ContactCandidate,
    *,
    max_force_n: float,
    config: TeacherWrenchEnvelopeConfig,
) -> tuple[list[float], list[float]]:
    target_norm = math.sqrt(sum(value * value for value in target_contact))
    if target_norm > max_force_n + config.cone_tolerance_n:
        raise SchemaValidationError(
            "teacher contact-force target exceeds the anchor capability"
        )
    base_widths = [
        min(
            max(config.minimum_force_window_n, abs(value) * config.force_window_fraction),
            max_force_n - abs(value),
        )
        for value in target_contact
    ]
    if candidate.friction is None:
        raise SchemaValidationError(
            "teacher contact-force range requires a resolved friction coefficient"
        )
    inward_world = tuple(-float(value) for value in candidate.normal_world)
    target_world = contact_wrench_to_world(
        [*target_contact, 0.0, 0.0, 0.0],
        candidate,
    )[:3]
    normal_load = max(
        0.0,
        sum(target_world[index] * inward_world[index] for index in range(3)),
    )
    inward_contact = world_wrench_to_contact(
        [*inward_world, 0.0, 0.0, 0.0],
        candidate,
    )[:3]
    # Two independent tangential axes share the circular-cone budget.  The
    # initial box exposes most of that capacity to the QP witness while the
    # exhaustive corner test below shrinks it until the complete box remains
    # inside both the circular friction cone and the anchor force norm.
    shared_tangent_capacity = (
        float(candidate.friction)
        * normal_load
        * config.friction_capacity_window_fraction
        / math.sqrt(2.0)
    )
    initial_widths = [
        min(
            max(
                base_width,
                shared_tangent_capacity
                * math.sqrt(max(0.0, 1.0 - inward_contact[index] ** 2)),
            ),
            max_force_n - abs(target_contact[index]),
        )
        for index, base_width in enumerate(base_widths)
    ]
    widths = [max(0.0, value) for value in initial_widths]
    for _ in range(config.max_cone_shrink_iterations + 1):
        lower = [value - width for value, width in zip(target_contact, widths)]
        upper = [value + width for value, width in zip(target_contact, widths)]
        if _force_box_is_safe(
            lower,
            upper,
            candidate,
            max_force_n=max_force_n,
            tolerance=config.cone_tolerance_n,
        ):
            return lower, upper
        widths = [width * 0.5 for width in widths]
    return list(target_contact), list(target_contact)


def _force_box_is_safe(
    lower: list[float],
    upper: list[float],
    candidate: ContactCandidate,
    *,
    max_force_n: float,
    tolerance: float,
) -> bool:
    friction = candidate.friction
    if friction is None:
        return False
    outward = tuple(float(value) for value in candidate.normal_world)
    inward = tuple(-value for value in outward)
    for corner in itertools.product(*zip(lower, upper)):
        force_world = contact_wrench_to_world([*corner, 0.0, 0.0, 0.0], candidate)[:3]
        force_norm = math.sqrt(sum(value * value for value in force_world))
        normal = sum(force_world[index] * inward[index] for index in range(3))
        tangent_sq = max(0.0, force_norm * force_norm - normal * normal)
        tangent = math.sqrt(tangent_sq)
        if (
            force_norm > max_force_n + tolerance
            or normal < -tolerance
            or tangent > float(friction) * normal + tolerance
        ):
            return False
    return True


def _bounded_axis_window(
    target: list[float],
    *,
    maximum: float,
    fraction: float,
    minimum_window: float,
) -> tuple[list[float], list[float]]:
    lower: list[float] = []
    upper: list[float] = []
    for value in target:
        if abs(value) > maximum:
            raise SchemaValidationError("teacher wrench target exceeds capability")
        width = min(max(minimum_window, abs(value) * fraction), maximum - abs(value))
        lower.append(value - max(0.0, width))
        upper.append(value + max(0.0, width))
    return lower, upper


def _positive_capability_limit(value: object, default: float) -> float:
    if value is None:
        return float(default)
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise SchemaValidationError("anchor wrench capability must be finite and positive")
    return parsed
