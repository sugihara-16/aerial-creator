from __future__ import annotations

"""Task-conditioned articulated teacher preparation for Order 9 C3."""

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Mapping

from amsrr.feasibility.articulated_reachability import (
    ArticulatedTrajectoryReachabilityEvaluator,
)
from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.policies.contact_candidate_sampler import ContactCandidateSampler
from amsrr.policies.design_policy_base import DesignPolicyContext
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.robot_model.gripper_surfaces import (
    GripperSurface,
    resolve_unoccupied_gripper_surfaces,
)
from amsrr.schemas.common import ContactMode, SchemaValidationError
from amsrr.schemas.contact_candidates import ContactCandidateSet
from amsrr.schemas.feasibility import FeasibilityResult
from amsrr.schemas.morphology import DesignOutput, MorphologyGraph
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import ContactWrenchTrajectory
from amsrr.schemas.runtime import (
    ModuleRuntimeState,
    ObjectRuntimeState,
    RuntimeObservation,
    TaskProgressState,
)
from amsrr.schemas.policies import ControllerStatus
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.order9_articulated_teacher import (
    Order9ArticulatedTeacherConfig,
    Order9ArticulatedTeacherPlan,
    Order9ArticulatedTrajectoryTeacher,
)
from amsrr.training.order9_posture_resolver import (
    Order9PostureCollisionBox,
    Order9PostureCollisionObject,
)
from amsrr.training.order9_design_teacher_dataset import (
    build_order9_task_conditioned_design_teacher,
)
from amsrr.utils.hashing import stable_hash


ORDER9_C3_ARTICULATED_TEACHER_VERSION = (
    "order9_c3_articulated_teacher_v8_mesh_clear_pregrasp"
)
ORDER9_C3_OVERHEAD_START_CLEARANCE_M = 0.35
ORDER9_C3_GROUND_PLANE_Z_M = 0.0
ORDER9_C3_SUPPORT_LONGITUDINAL_MARGIN_M = 0.05
ORDER9_C3_SUPPORT_LATERAL_INSET_M = 0.04


@dataclass(frozen=True)
class Order9C3TeacherConfig:
    maximum_surface_pair_attempts: int = 64
    preferred_surface_port_ids: tuple[int, int] | None = None
    preferred_candidate_group_id: str | None = None
    excluded_surface_port_id_pairs: tuple[tuple[int, int], ...] = ()
    require_full_trajectory_recheck: bool = True
    collision_margin_m: float = 0.005
    grasp_contact_height_offset_m: float = 0.0
    pregrasp_clearance_m: float = 0.08

    def __post_init__(self) -> None:
        if self.maximum_surface_pair_attempts < 1:
            raise ValueError("maximum_surface_pair_attempts must be positive")
        if self.preferred_surface_port_ids is not None:
            values = tuple(int(value) for value in self.preferred_surface_port_ids)
            if (
                len(values) != 2
                or len(set(values)) != 2
                or min(values) < 0
            ):
                raise ValueError(
                    "preferred_surface_port_ids must contain two distinct "
                    "non-negative port ids"
                )
        if (
            self.preferred_candidate_group_id is not None
            and not self.preferred_candidate_group_id
        ):
            raise ValueError(
                "preferred_candidate_group_id must be non-empty when provided"
            )
        excluded_pairs: list[frozenset[int]] = []
        for pair in self.excluded_surface_port_id_pairs:
            values = tuple(int(value) for value in pair)
            if (
                len(values) != 2
                or len(set(values)) != 2
                or min(values) < 0
            ):
                raise ValueError(
                    "excluded_surface_port_id_pairs must contain pairs of "
                    "two distinct non-negative port ids"
                )
            excluded_pairs.append(frozenset(values))
        if len(excluded_pairs) != len(set(excluded_pairs)):
            raise ValueError(
                "excluded_surface_port_id_pairs must not repeat a pair"
            )
        if not isinstance(self.require_full_trajectory_recheck, bool):
            raise ValueError(
                "require_full_trajectory_recheck must be boolean"
            )
        if (
            not math.isfinite(float(self.collision_margin_m))
            or self.collision_margin_m <= 0.0
        ):
            raise ValueError("collision_margin_m must be finite and positive")
        if (
            not math.isfinite(float(self.grasp_contact_height_offset_m))
            or not 0.0 <= self.grasp_contact_height_offset_m <= 0.10
        ):
            raise ValueError(
                "grasp_contact_height_offset_m must be finite and in "
                "[0, 0.10]"
            )
        if (
            not math.isfinite(float(self.pregrasp_clearance_m))
            or not 0.05 <= self.pregrasp_clearance_m <= 0.30
        ):
            raise ValueError(
                "pregrasp_clearance_m must be finite and in [0.05, 0.30]"
            )


@dataclass(frozen=True)
class Order9C3TeacherBundle:
    design_output: DesignOutput
    design_feasibility: FeasibilityResult
    contact_candidate_set: ContactCandidateSet
    trajectory_plan: Order9ArticulatedTeacherPlan
    selected_surface_port_ids: tuple[int, int]
    reachability_margins: dict[str, float]
    teacher_version: str = ORDER9_C3_ARTICULATED_TEACHER_VERSION

    @property
    def trajectory(self) -> ContactWrenchTrajectory:
        return self.trajectory_plan.trajectory


def order9_c3_teacher_evidence(
    bundle: Order9C3TeacherBundle,
) -> dict[str, object]:
    """Return compact immutable evidence for bucket/runtime agreement."""

    configuration_plan = bundle.trajectory_plan.configuration_space_plan
    return {
        "teacher_version": bundle.teacher_version,
        "trajectory_teacher_version": bundle.trajectory_plan.teacher_version,
        "ik_solver_version": bundle.trajectory_plan.ik_solution.solver_version,
        "posture_resolver_version": (
            bundle.trajectory_plan.posture_resolution.evidence.resolver_version
        ),
        "posture_solver_version": (
            bundle.trajectory_plan.posture_resolution.evidence.solver_version
        ),
        "selected_surface_port_ids": list(bundle.selected_surface_port_ids),
        "task_conditioned_morphology_hash": (
            bundle.design_output.target_morphology.stable_hash()
        ),
        "contact_candidate_set_hash": stable_hash(
            bundle.contact_candidate_set.to_dict()
        ),
        "raw_pi_h_trajectory_hash": stable_hash(
            bundle.trajectory_plan.raw_trajectory.to_dict()
        ),
        "trajectory_hash": stable_hash(bundle.trajectory.to_dict()),
        "posture_resolution_evidence_hash": stable_hash(
            bundle.trajectory_plan.posture_resolution.evidence.identity_dict()
        ),
        "resolved_posture_knot_count": len(bundle.trajectory.knots),
        "posture_maximum_joint_rate_rad_s": (
            bundle.trajectory_plan.posture_resolution.evidence
            .maximum_joint_rate_rad_s
        ),
        "posture_minimum_joint_rate_margin_rad_s": (
            bundle.trajectory_plan.posture_resolution.evidence
            .minimum_joint_rate_margin_rad_s
        ),
        "posture_collision_gate_status": (
            bundle.trajectory_plan.posture_resolution.evidence
            .collision_gate_status
        ),
        "posture_collision_gate_version": (
            bundle.trajectory_plan.posture_resolution.evidence
            .collision_gate_version
        ),
        "posture_minimum_collision_clearance_m": (
            bundle.trajectory_plan.posture_resolution.evidence
            .minimum_collision_clearance_m
        ),
        "posture_maximum_collision_violating_pair_count": (
            bundle.trajectory_plan.posture_resolution.evidence
            .maximum_collision_violating_pair_count
        ),
        "candidate_group_id": bundle.trajectory_plan.candidate_group_id,
        "ik_iterations": bundle.trajectory_plan.ik_solution.iterations,
        "ik_maximum_position_error_m": (
            bundle.trajectory_plan.ik_solution.maximum_position_error_m
        ),
        "ik_maximum_normal_error_rad": (
            bundle.trajectory_plan.ik_solution.maximum_normal_error_rad
        ),
        "configuration_planner_version": (
            None
            if configuration_plan is None
            else configuration_plan.planner_version
        ),
        "configuration_plan_method": (
            None if configuration_plan is None else configuration_plan.method
        ),
        "configuration_plan_state_count": (
            0 if configuration_plan is None else len(configuration_plan.states)
        ),
        "configuration_plan_collision_check_count": (
            0
            if configuration_plan is None
            else configuration_plan.collision_check_count
        ),
        "configuration_plan_sampled_state_count": (
            0
            if configuration_plan is None
            else configuration_plan.sampled_state_count
        ),
        "reachability_margins_hash": stable_hash(bundle.reachability_margins),
        "minimum_reachability_margin": min(
            bundle.reachability_margins.values(),
            default=0.0,
        ),
    }


def build_order9_c3_articulated_teacher(
    *,
    task_spec: TaskSpec,
    structural_target: MorphologyGraph,
    physical_model: PhysicalModel,
    config: Order9C3TeacherConfig | None = None,
    runtime_observation: RuntimeObservation | None = None,
    collision_object: Order9PostureCollisionObject | None = None,
    contact_goal_joint_seed_positions_rad: (
        Mapping[str, float] | None
    ) = None,
) -> Order9C3TeacherBundle:
    """Search mesh-backed surface pairs, then emit one complete checked plan."""

    cfg = config or Order9C3TeacherConfig()
    built = IRGBuilder().build_with_scene_graph(task_spec)
    envelope = InteractionEnvelopeExtractor().extract(built.irg)
    design_context = DesignPolicyContext(
        task_spec,
        built.irg,
        physical_model,
        envelope,
    )
    surfaces = resolve_unoccupied_gripper_surfaces(
        structural_target,
        physical_model,
    )
    pairs = _rank_surface_pairs(structural_target, surfaces)
    excluded_pairs = {
        frozenset(int(value) for value in pair)
        for pair in cfg.excluded_surface_port_id_pairs
    }
    if excluded_pairs:
        pairs = [
            pair
            for pair in pairs
            if frozenset(
                (pair[0].port_global_id, pair[1].port_global_id)
            )
            not in excluded_pairs
        ]
    if not pairs:
        raise SchemaValidationError(
            "C3 structural morphology has no eligible pair of two free "
            "mesh-backed surfaces on distinct modules"
        )
    if cfg.preferred_surface_port_ids is not None:
        requested = frozenset(int(value) for value in cfg.preferred_surface_port_ids)
        pairs = [
            pair
            for pair in pairs
            if frozenset(
                (pair[0].port_global_id, pair[1].port_global_id)
            )
            == requested
        ]
        if not pairs:
            raise SchemaValidationError(
                "preferred C3 surface pair is unavailable, occupied, or lies "
                "on one module"
            )
    failures: list[str] = []
    for first, second in pairs[: cfg.maximum_surface_pair_attempts]:
        surface_ids = (first.port_global_id, second.port_global_id)
        try:
            _trace, design, design_feasibility = (
                build_order9_task_conditioned_design_teacher(
                    design_context,
                    structural_target,
                    preferred_anchor_surface_ids=surface_ids,
                )
            )
            candidates = ContactCandidateSampler().sample(
                task_spec=task_spec,
                irg=built.irg,
                interaction_envelope=envelope,
                morphology_graph=design.target_morphology,
                geometry_descriptors=built.scene_graph.geometry_descriptors,
            )
            candidates = _offset_horizontal_grasp_candidates(
                candidates,
                height_offset_m=cfg.grasp_contact_height_offset_m,
            )
            teacher_observation = (
                build_order9_c3_neutral_runtime_observation(
                    design.target_morphology,
                    physical_model,
                    task_spec,
                    # C3 selection starts at the same neutral observation as
                    # the real phase sequence.  The final-contact IK above
                    # already proves that the selected pair is reachable;
                    # beginning the rolling trajectory in contact acquisition
                    # would incorrectly interpolate one window from the
                    # neutral CoM directly to the already-approached CoM.
                    phase_label="approach",
                )
                if runtime_observation is None
                else RuntimeObservation.from_dict(
                    runtime_observation.to_dict()
                )
            )
            high_level_context = HighLevelPolicyContext(
                built.irg,
                envelope,
                design.target_morphology,
                candidates,
                runtime_observation=teacher_observation,
            )
            plan = Order9ArticulatedTrajectoryTeacher(
                physical_model,
                config=Order9ArticulatedTeacherConfig(
                    preferred_candidate_group_id=(
                        cfg.preferred_candidate_group_id
                    ),
                    collision_margin_m=cfg.collision_margin_m,
                    pregrasp_clearance_m=cfg.pregrasp_clearance_m,
                ),
                collision_object=collision_object,
            ).plan(
                high_level_context,
                initial_object_poses_world={
                    obj.object_id: obj.pose_world
                    for obj in task_spec.scene.objects
                },
                contact_goal_joint_seed_positions_rad=(
                    contact_goal_joint_seed_positions_rad
                ),
            )
            checked_context = HighLevelPolicyContext(
                built.irg,
                envelope,
                design.target_morphology,
                candidates,
                runtime_observation=teacher_observation,
            )
            evaluations = ()
            if cfg.require_full_trajectory_recheck:
                evaluations = (
                    ArticulatedTrajectoryReachabilityEvaluator(
                        physical_model
                    ).evaluate_trajectory(
                        context=checked_context,
                        trajectory=plan.trajectory,
                    )
                )
                codes = sorted(
                    {
                        code
                        for evaluation in evaluations
                        for code in evaluation.violation_codes
                    }
                )
                if codes:
                    raise SchemaValidationError(
                        "articulated teacher failed its independent "
                        "reachability recheck: " + ",".join(codes)
                    )
            return Order9C3TeacherBundle(
                design_output=design,
                design_feasibility=design_feasibility,
                contact_candidate_set=candidates,
                trajectory_plan=plan,
                selected_surface_port_ids=surface_ids,
                reachability_margins={
                    f"knot_{index}.{name}": value
                    for index, evaluation in enumerate(evaluations)
                    for name, value in evaluation.margins.items()
                },
            )
        except (SchemaValidationError, ValueError, KeyError) as error:
            failures.append(
                f"{first.port_global_id}:{second.port_global_id}:{error}"
            )
    raise SchemaValidationError(
        "C3 articulated teacher exhausted surface-pair search; "
        + "; ".join(failures[:8])
    )


def _offset_horizontal_grasp_candidates(
    candidate_set: ContactCandidateSet,
    *,
    height_offset_m: float,
) -> ContactCandidateSet:
    """Move supported-box side contacts upward without changing identity.

    C3 contact regions are sampled at each face centre.  A centre-height side
    contact can be kinematically valid while the real docking mesh below its
    contact frame intersects the payload support.  The offset remains tangent
    to a vertical face, so it changes neither its normal nor its wrench model.
    """

    if height_offset_m == 0.0:
        return candidate_set
    result = ContactCandidateSet.from_dict(candidate_set.to_dict())
    shifted = 0
    for candidate in result.candidates:
        if (
            candidate.contact_mode != ContactMode.GRASP
            or abs(float(candidate.normal_world[2])) > 0.25
        ):
            continue
        pose = list(candidate.contact_pose_world)
        frame = list(candidate.contact_frame_world)
        pose[2] += float(height_offset_m)
        frame[2] += float(height_offset_m)
        candidate.contact_pose_world = tuple(pose)  # type: ignore[assignment]
        candidate.contact_frame_world = tuple(frame)  # type: ignore[assignment]
        candidate.candidate_scores = {
            **candidate.candidate_scores,
            "c3_grasp_contact_height_offset_m": float(height_offset_m),
        }
        candidate.validate()
        shifted += 1
    if shifted == 0:
        raise SchemaValidationError(
            "C3 grasp contact height offset found no horizontal grasp face"
        )
    result.sampler_version = (
        f"{result.sampler_version}+c3_side_height_"
        f"{1.0e3 * float(height_offset_m):.3f}mm"
    )
    result.validate()
    return result


def _rank_surface_pairs(
    graph: MorphologyGraph,
    surfaces: tuple[GripperSurface, ...],
) -> list[tuple[GripperSurface, GripperSurface]]:
    """Rank search order only; no neutral-pose quantity is a hard gate."""

    distances = {
        module.module_id: _graph_distances(graph, module.module_id)
        for module in graph.modules
    }
    ranked = []
    for first, second in combinations(surfaces, 2):
        if first.module_id == second.module_id:
            continue
        graph_distance = distances[first.module_id].get(second.module_id, 0)
        ranked.append(
            (
                (
                    -graph_distance,
                    first.module_id,
                    second.module_id,
                    first.port_global_id,
                    second.port_global_id,
                ),
                (first, second),
            )
        )
    ranked.sort(key=lambda value: value[0])
    return [pair for _rank, pair in ranked]


def _graph_distances(
    graph: MorphologyGraph,
    source: int,
) -> dict[int, int]:
    adjacency = {module.module_id: set() for module in graph.modules}
    for edge in graph.dock_edges:
        adjacency[edge.src_module_id].add(edge.dst_module_id)
        adjacency[edge.dst_module_id].add(edge.src_module_id)
    distances = {source: 0}
    pending = [source]
    while pending:
        current = pending.pop(0)
        for neighbor in sorted(adjacency[current]):
            if neighbor in distances:
                continue
            distances[neighbor] = distances[current] + 1
            pending.append(neighbor)
    return distances


def build_order9_c3_posture_collision_object(
    task_spec: TaskSpec,
) -> Order9PostureCollisionObject:
    """Resolve the single movable C3 box object for convex posture checking."""

    movable = [value for value in task_spec.scene.objects if value.movable]
    if len(movable) != 1:
        raise SchemaValidationError(
            "C3 posture collision checking requires exactly one movable object"
        )
    object_spec = movable[0]
    geometries = {
        value.geometry_id: value
        for value in task_spec.scene.geometry_library
    }
    geometry = geometries.get(object_spec.geometry_id)
    if geometry is None:
        raise SchemaValidationError(
            "C3 movable object references an unknown geometry"
        )
    if geometry.geometry_type.value != "box":
        raise SchemaValidationError(
            "C3 posture collision checking currently requires a box object"
        )
    parameters = geometry.primitive_params or {}
    raw_size = parameters.get("size_m")
    if (
        not isinstance(raw_size, (list, tuple))
        or len(raw_size) != 3
    ):
        raise SchemaValidationError(
            "C3 box collision geometry lacks a three-value size_m"
        )
    size_m = tuple(
        float(raw_size[index]) * float(geometry.scale[index])
        for index in range(3)
    )
    support = _bucket_object_support_collision_box(
        task_spec=task_spec,
        object_spec=object_spec,
        object_size_m=size_m,
        geometries=geometries,
    )
    return Order9PostureCollisionObject(
        object_id=object_spec.object_id,
        size_m=size_m,
        initial_pose_world=tuple(object_spec.pose_world),
        environment_boxes=(support,),
        ground_plane_z_m=ORDER9_C3_GROUND_PLANE_Z_M,
    )


def _bucket_object_support_collision_box(
    *,
    task_spec: TaskSpec,
    object_spec,
    object_size_m: tuple[float, float, float],
    geometries: dict,
) -> Order9PostureCollisionBox:
    """Reconstruct the per-bucket Order 8 support from object start/goal."""

    targets = [
        goal.target_pose_world
        for goal in task_spec.goals
        if goal.target_entity_id == object_spec.object_id
        and goal.target_pose_world is not None
    ]
    if len(targets) != 1:
        raise SchemaValidationError(
            "C3 bucket support requires one object-pose goal"
        )
    target = targets[0]
    start = object_spec.pose_world
    support_top_z = _support_top_z(task_spec, geometries)
    if support_top_z <= ORDER9_C3_GROUND_PLANE_Z_M:
        # Generic object-task fixtures may omit an explicit support surface.
        # In that case reconstruct the platform top from the lower of the
        # archived start/goal object bottoms.
        support_top_z = min(
            float(start[2]) - 0.5 * float(object_size_m[2]),
            float(target[2]) - 0.5 * float(object_size_m[2]),
        )
    support_height = support_top_z - ORDER9_C3_GROUND_PLANE_Z_M
    if support_height <= 0.0:
        raise SchemaValidationError(
            "C3 bucket support top must be above the ground plane"
        )
    return Order9PostureCollisionBox(
        box_id="order9_bucket_object_support",
        size_m=(
            abs(float(target[0]) - float(start[0]))
            + float(object_size_m[0])
            + ORDER9_C3_SUPPORT_LONGITUDINAL_MARGIN_M,
            max(
                0.05,
                abs(float(target[1]) - float(start[1]))
                + float(object_size_m[1])
                - ORDER9_C3_SUPPORT_LATERAL_INSET_M,
            ),
            support_height,
        ),
        pose_world=(
            0.5 * (float(start[0]) + float(target[0])),
            0.5 * (float(start[1]) + float(target[1])),
            ORDER9_C3_GROUND_PLANE_Z_M + 0.5 * support_height,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
    )


def build_order9_c3_neutral_runtime_observation(
    morphology: MorphologyGraph,
    physical_model: PhysicalModel,
    task_spec: TaskSpec,
    *,
    phase_label: str | None = None,
) -> RuntimeObservation:
    dock_ids = sorted(
        {
            str(port.mechanical_limits["mechanism_joint_id"])
            for port in physical_model.dock_ports
        }
    )
    movable = [value for value in task_spec.scene.objects if value.movable]
    if len(movable) != 1:
        raise SchemaValidationError(
            "C3 overhead initialization requires one movable object"
        )
    object_spec = movable[0]
    geometries = {
        value.geometry_id: value
        for value in task_spec.scene.geometry_library
    }
    geometry = geometries.get(object_spec.geometry_id)
    if geometry is None or geometry.geometry_type.value != "box":
        raise SchemaValidationError(
            "C3 overhead initialization requires a box object"
        )
    raw_size = (geometry.primitive_params or {}).get("size_m")
    if not isinstance(raw_size, (list, tuple)) or len(raw_size) != 3:
        raise SchemaValidationError(
            "C3 overhead initialization lacks object size_m"
        )
    object_top_z = float(object_spec.pose_world[2]) + 0.5 * (
        float(raw_size[2]) * float(geometry.scale[2])
    )
    support_top_z = _support_top_z(task_spec, geometries)
    overhead_root_z = max(object_top_z, support_top_z) + (
        ORDER9_C3_OVERHEAD_START_CLEARANCE_M
    )
    return RuntimeObservation(
        time_s=0.0,
        morphology_graph=morphology,
        module_states=[
            ModuleRuntimeState(
                module_id=module.module_id,
                pose_world=(
                    float(module.pose_in_design_frame[0]),
                    float(module.pose_in_design_frame[1]),
                    float(module.pose_in_design_frame[2])
                    + overhead_root_z,
                    *module.pose_in_design_frame[3:],
                ),
                twist_world=[0.0] * 6,
                joint_positions={joint_id: 0.0 for joint_id in dock_ids},
                joint_velocities={joint_id: 0.0 for joint_id in dock_ids},
            )
            for module in morphology.modules
        ],
        object_states=[
            ObjectRuntimeState(
                object_id=obj.object_id,
                pose_world=obj.pose_world,
                twist_world=[0.0] * 6,
            )
            for obj in task_spec.scene.objects
        ],
        contact_states=[],
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
        task_progress=TaskProgressState(phase_label=phase_label),
    )


def _support_top_z(
    task_spec: TaskSpec,
    geometries: dict,
) -> float:
    top_values = []
    for surface in task_spec.scene.environment.support_surfaces:
        geometry = geometries.get(surface.geometry_id)
        if geometry is None or geometry.geometry_type.value != "box":
            continue
        raw_size = (geometry.primitive_params or {}).get("size_m")
        if not isinstance(raw_size, (list, tuple)) or len(raw_size) != 3:
            continue
        top_values.append(
            float(surface.pose_world[2])
            + 0.5 * float(raw_size[2]) * float(geometry.scale[2])
        )
    return max(top_values, default=0.0)


# Backward-compatible private name retained for existing tests and diagnostics.
_neutral_runtime_observation = build_order9_c3_neutral_runtime_observation


__all__ = [
    "ORDER9_C3_ARTICULATED_TEACHER_VERSION",
    "ORDER9_C3_OVERHEAD_START_CLEARANCE_M",
    "Order9C3TeacherBundle",
    "Order9C3TeacherConfig",
    "build_order9_c3_articulated_teacher",
    "build_order9_c3_neutral_runtime_observation",
    "build_order9_c3_posture_collision_object",
    "order9_c3_teacher_evidence",
]
