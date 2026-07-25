from __future__ import annotations

"""Task-conditioned articulated teacher preparation for Order 9 C3."""

from dataclasses import dataclass
from itertools import combinations

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
from amsrr.schemas.common import SchemaValidationError
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
from amsrr.training.order9_design_teacher_dataset import (
    build_order9_task_conditioned_design_teacher,
)
from amsrr.utils.hashing import stable_hash


ORDER9_C3_ARTICULATED_TEACHER_VERSION = (
    "order9_c3_articulated_teacher_v3_posture_resolver"
)


@dataclass(frozen=True)
class Order9C3TeacherConfig:
    maximum_surface_pair_attempts: int = 64
    preferred_surface_port_ids: tuple[int, int] | None = None
    preferred_candidate_group_id: str | None = None

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
        "candidate_group_id": bundle.trajectory_plan.candidate_group_id,
        "ik_iterations": bundle.trajectory_plan.ik_solution.iterations,
        "ik_maximum_position_error_m": (
            bundle.trajectory_plan.ik_solution.maximum_position_error_m
        ),
        "ik_maximum_normal_error_rad": (
            bundle.trajectory_plan.ik_solution.maximum_normal_error_rad
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
    if not pairs:
        raise SchemaValidationError(
            "C3 structural morphology has no two free mesh-backed surfaces "
            "on distinct modules"
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
            neutral_observation = _neutral_runtime_observation(
                design.target_morphology,
                physical_model,
                task_spec,
                phase_label="contact_acquisition",
            )
            high_level_context = HighLevelPolicyContext(
                built.irg,
                envelope,
                design.target_morphology,
                candidates,
                runtime_observation=neutral_observation,
            )
            plan = Order9ArticulatedTrajectoryTeacher(
                physical_model,
                config=Order9ArticulatedTeacherConfig(
                    preferred_candidate_group_id=(
                        cfg.preferred_candidate_group_id
                    )
                ),
            ).plan(
                high_level_context,
                initial_object_poses_world={
                    obj.object_id: obj.pose_world
                    for obj in task_spec.scene.objects
                },
            )
            checked_context = HighLevelPolicyContext(
                built.irg,
                envelope,
                design.target_morphology,
                candidates,
                runtime_observation=neutral_observation,
            )
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
                    "articulated teacher failed its independent reachability "
                    "recheck: " + ",".join(codes)
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


def _neutral_runtime_observation(
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
    return RuntimeObservation(
        time_s=0.0,
        morphology_graph=morphology,
        module_states=[
            ModuleRuntimeState(
                module_id=module.module_id,
                pose_world=module.pose_in_design_frame,
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


__all__ = [
    "ORDER9_C3_ARTICULATED_TEACHER_VERSION",
    "Order9C3TeacherBundle",
    "Order9C3TeacherConfig",
    "build_order9_c3_articulated_teacher",
    "order9_c3_teacher_evidence",
]
