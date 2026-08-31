from __future__ import annotations

"""Ideal-tracking nominal approach/grasp trajectories for Order 9 C3.

The persisted raw trajectories remain phase-local pi_H proposals.  This module
chains their resolved endpoints only for offline preparation and inspection;
it does not create a longer pi_H policy output or claim dynamic task success.
"""

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from amsrr.geometry.pose_math import compose_pose, inverse_pose
from amsrr.feasibility.articulated_reachability import (
    base_pose_for_centroidal_target,
)
from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.robot_model.whole_structure_kinematics import (
    WholeStructureKinematics,
)
from amsrr.schemas.common import (
    SchemaBase,
    SchemaValidationError,
    require_non_empty,
)
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.contact_candidates import ContactCandidateSet
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.runtime import (
    ModuleRuntimeState,
    ObjectRuntimeState,
    RuntimeObservation,
    TaskProgressState,
)
from amsrr.schemas.policies import (
    ContactWrenchTrajectory,
    ControllerStatus,
    InteractionKnot,
)
from amsrr.simulation.order9_object_task_runtime import (
    ORDER9_OBJECT_TASK_PHASES,
    Order9ObjectTaskPhase,
    Order9ObjectTaskRuntimeConfig,
)
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.order9_articulated_teacher import (
    Order9ArticulatedTeacherConfig,
    Order9ArticulatedTeacherPlan,
    Order9ArticulatedTrajectoryTeacher,
)
from amsrr.training.order9_c3_teacher import (
    Order9C3TeacherConfig,
    Order9C3TeacherBundle,
    build_order9_c3_articulated_teacher,
    build_order9_c3_neutral_runtime_observation,
    build_order9_c3_posture_collision_object,
    order9_c3_teacher_evidence,
)
from amsrr.utils.hashing import hash_file, stable_hash

ORDER9_C3_NOMINAL_TRAJECTORY_VERSION = (
    "order9_c3_nominal_configuration_space_v6_complete_task_phases"
)
ORDER9_C3_NOMINAL_TRAJECTORY_SET_VERSION = (
    "order9_c3_nominal_trajectory_set_v6_complete_task_phases"
)
_PHASES_TO_GRASP = ("approach", "contact_acquisition")
_COMPLETE_TASK_PHASES = tuple(phase.value for phase in ORDER9_OBJECT_TASK_PHASES)


@dataclass(frozen=True)
class Order9C3NominalWindow:
    window_index: int
    phase: str
    global_start_time_s: float
    plan: Order9ArticulatedTeacherPlan


@dataclass(frozen=True)
class Order9C3NominalTrajectory:
    selection_bundle: Order9C3TeacherBundle
    windows: tuple[Order9C3NominalWindow, ...]
    timeline: tuple[dict[str, Any], ...]
    final_observation: RuntimeObservation
    proxy_collision_validation_status: str

    @property
    def duration_s(self) -> float:
        if not self.windows:
            return 0.0
        last = self.windows[-1]
        return float(last.global_start_time_s + last.plan.trajectory.horizon_s)


@dataclass
class Order9C3NominalWindowArtifact(SchemaBase):
    window_index: int
    phase: str
    global_start_time_s: float
    horizon_s: float
    phase_target_reached: bool
    raw_trajectory_path: str
    raw_trajectory_sha256: str
    raw_trajectory_hash: str
    resolved_trajectory_path: str
    resolved_trajectory_sha256: str
    resolved_trajectory_hash: str
    resolver_evidence_path: str
    resolver_evidence_sha256: str
    raw_knot_count: int
    resolved_knot_count: int
    configuration_planner_version: str
    configuration_plan_method: str
    configuration_plan_state_count: int
    configuration_plan_collision_check_count: int
    configuration_plan_sampled_state_count: int

    def validate(self) -> None:
        if self.window_index < 0:
            raise SchemaValidationError("nominal window index must be non-negative")
        require_non_empty(self.phase, "Order9C3NominalWindowArtifact.phase")
        if (
            not math.isfinite(float(self.global_start_time_s))
            or self.global_start_time_s < 0.0
            or not math.isfinite(float(self.horizon_s))
            or self.horizon_s <= 0.0
        ):
            raise SchemaValidationError(
                "nominal window timing must be finite and non-negative"
            )
        if self.raw_knot_count < 2 or self.resolved_knot_count < 2:
            raise SchemaValidationError(
                "nominal window trajectories require at least two knots"
            )
        require_non_empty(
            self.configuration_planner_version,
            "configuration_planner_version",
        )
        require_non_empty(
            self.configuration_plan_method,
            "configuration_plan_method",
        )
        if (
            self.configuration_plan_state_count < 2
            or self.configuration_plan_collision_check_count < 1
            or self.configuration_plan_sampled_state_count < 0
        ):
            raise SchemaValidationError(
                "nominal window configuration-plan evidence is invalid"
            )
        for name in (
            "raw_trajectory_path",
            "resolved_trajectory_path",
            "resolver_evidence_path",
        ):
            require_non_empty(
                str(getattr(self, name)),
                f"Order9C3NominalWindowArtifact.{name}",
            )
        for name in (
            "raw_trajectory_sha256",
            "raw_trajectory_hash",
            "resolved_trajectory_sha256",
            "resolved_trajectory_hash",
            "resolver_evidence_sha256",
        ):
            _require_sha256(str(getattr(self, name)), name)


@dataclass
class Order9C3NominalPhaseArtifact(SchemaBase):
    phase: str
    trajectory_path: str
    trajectory_sha256: str
    trajectory_hash: str
    knot_count: int
    generation_method: str
    collision_validation_status: str = "pending_offline_admission"

    def validate(self) -> None:
        if self.phase not in _COMPLETE_TASK_PHASES:
            raise SchemaValidationError(
                f"unknown C3 nominal task phase: {self.phase!r}"
            )
        for name in (
            "trajectory_path",
            "generation_method",
            "collision_validation_status",
        ):
            require_non_empty(
                str(getattr(self, name)),
                f"Order9C3NominalPhaseArtifact.{name}",
            )
        for name in ("trajectory_sha256", "trajectory_hash"):
            _require_sha256(str(getattr(self, name)), name)
        if self.knot_count < 2:
            raise SchemaValidationError(
                "C3 nominal phase trajectory requires at least two knots"
            )
        if self.collision_validation_status not in {
            "pending_offline_admission",
            "accepted_offline_admission",
        }:
            raise SchemaValidationError("C3 nominal phase collision status is invalid")


@dataclass
class Order9C3NominalTrajectoryArtifact(SchemaBase):
    bucket_id: str
    split: DatasetSplit
    physical_model_hash: str
    task_spec_sha256: str
    structural_hash: str
    robot_urdf_path: str
    robot_urdf_sha256: str
    task_conditioned_morphology_path: str
    task_conditioned_morphology_sha256: str
    task_conditioned_morphology_hash: str
    contact_candidate_set_path: str
    contact_candidate_set_sha256: str
    contact_candidate_set_hash: str
    timeline_path: str
    timeline_sha256: str
    selected_surface_port_ids: list[int]
    selected_candidate_group_id: str | None
    duration_s: float
    final_phase: str
    final_phase_target_reached: bool
    proxy_collision_validation_status: str
    windows: list[Order9C3NominalWindowArtifact]
    phase_trajectories: list[Order9C3NominalPhaseArtifact]
    selection_evidence: dict[str, Any] = field(default_factory=dict)
    artifact_version: str = ORDER9_C3_NOMINAL_TRAJECTORY_VERSION

    def validate(self) -> None:
        if self.artifact_version != ORDER9_C3_NOMINAL_TRAJECTORY_VERSION:
            raise SchemaValidationError(
                "C3 nominal trajectory artifact version mismatch"
            )
        for name in (
            "bucket_id",
            "robot_urdf_path",
            "task_conditioned_morphology_path",
            "contact_candidate_set_path",
            "timeline_path",
            "final_phase",
            "proxy_collision_validation_status",
        ):
            require_non_empty(
                str(getattr(self, name)),
                f"Order9C3NominalTrajectoryArtifact.{name}",
            )
        for name in (
            "physical_model_hash",
            "task_spec_sha256",
            "structural_hash",
            "robot_urdf_sha256",
            "task_conditioned_morphology_sha256",
            "task_conditioned_morphology_hash",
            "contact_candidate_set_sha256",
            "contact_candidate_set_hash",
            "timeline_sha256",
        ):
            _require_sha256(str(getattr(self, name)), name)
        if (
            len(self.selected_surface_port_ids) != 2
            or len(set(self.selected_surface_port_ids)) != 2
        ):
            raise SchemaValidationError(
                "C3 nominal trajectory requires two selected surfaces"
            )
        if not math.isfinite(float(self.duration_s)) or self.duration_s <= 0.0:
            raise SchemaValidationError(
                "C3 nominal trajectory duration must be positive"
            )
        if not self.windows:
            raise SchemaValidationError("C3 nominal trajectory must contain windows")
        if [value.window_index for value in self.windows] != list(
            range(len(self.windows))
        ):
            raise SchemaValidationError(
                "C3 nominal trajectory window indices are not contiguous"
            )
        for window in self.windows:
            window.validate()
        phases = [value.phase for value in self.phase_trajectories]
        if phases != list(_COMPLETE_TASK_PHASES):
            raise SchemaValidationError(
                "C3 nominal artifact must persist all eight task phases in order"
            )
        for phase in self.phase_trajectories:
            phase.validate()
        if self.final_phase != Order9ObjectTaskPhase.SETTLE.value:
            raise SchemaValidationError(
                "C3 nominal complete trajectory must end in settle"
            )
        if not self.final_phase_target_reached:
            raise SchemaValidationError(
                "C3 nominal grasp trajectory did not reach its final target"
            )
        if self.proxy_collision_validation_status not in {
            "pending_offline_admission",
            "enforced_during_generation",
        }:
            raise SchemaValidationError(
                "C3 nominal proxy collision validation status is invalid"
            )


@dataclass
class Order9C3NominalTrajectorySetEntry(SchemaBase):
    bucket_id: str
    split: DatasetSplit
    module_count: int
    structural_hash: str
    artifact_path: str
    artifact_sha256: str
    animation_html_path: str | None = None
    animation_scene_path: str | None = None
    animation_scene_sha256: str | None = None

    def validate(self) -> None:
        require_non_empty(
            self.bucket_id,
            "Order9C3NominalTrajectorySetEntry.bucket_id",
        )
        if not 2 <= self.module_count <= 8:
            raise SchemaValidationError("C3 nominal set module count must be in [2, 8]")
        for name in ("structural_hash", "artifact_sha256"):
            _require_sha256(str(getattr(self, name)), name)
        require_non_empty(
            self.artifact_path,
            "Order9C3NominalTrajectorySetEntry.artifact_path",
        )
        animation_values = (
            self.animation_html_path,
            self.animation_scene_path,
            self.animation_scene_sha256,
        )
        if any(value is not None for value in animation_values):
            if any(value is None for value in animation_values):
                raise SchemaValidationError(
                    "C3 nominal animation provenance is incomplete"
                )
            _require_sha256(
                str(self.animation_scene_sha256),
                "animation_scene_sha256",
            )


@dataclass
class Order9C3NominalTrajectorySetManifest(SchemaBase):
    bucket_manifest_path: str
    bucket_manifest_sha256: str
    physical_model_hash: str
    entries: list[Order9C3NominalTrajectorySetEntry]
    index_html_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    manifest_version: str = ORDER9_C3_NOMINAL_TRAJECTORY_SET_VERSION

    def validate(self) -> None:
        if self.manifest_version != ORDER9_C3_NOMINAL_TRAJECTORY_SET_VERSION:
            raise SchemaValidationError("C3 nominal trajectory set version mismatch")
        require_non_empty(
            self.bucket_manifest_path,
            "Order9C3NominalTrajectorySetManifest.bucket_manifest_path",
        )
        _require_sha256(
            self.bucket_manifest_sha256,
            "bucket_manifest_sha256",
        )
        _require_sha256(self.physical_model_hash, "physical_model_hash")
        if not self.entries:
            raise SchemaValidationError("C3 nominal trajectory set cannot be empty")
        ids = [value.bucket_id for value in self.entries]
        if len(ids) != len(set(ids)):
            raise SchemaValidationError("C3 nominal trajectory set bucket ids repeat")


@dataclass(frozen=True)
class Order9C3AcceptedNominalBundle:
    """Hash-validated offline teacher data consumed by one C3 bucket.

    The configuration-space planner is deliberately absent from this runtime
    object.  It loads only the already accepted, dense 10 Hz IK result.
    """

    set_manifest_path: Path
    set_manifest_sha256: str
    entry: Order9C3NominalTrajectorySetEntry
    artifact_path: Path
    artifact: Order9C3NominalTrajectoryArtifact
    morphology: MorphologyGraph
    contact_candidate_set: ContactCandidateSet
    trajectory: ContactWrenchTrajectory
    phase_trajectories: dict[str, ContactWrenchTrajectory]
    provenance: dict[str, Any]


def generate_order9_c3_nominal_grasp_trajectory(
    *,
    task_spec: TaskSpec,
    structural_target: MorphologyGraph,
    physical_model: PhysicalModel,
    maximum_windows_per_phase: int = 32,
    enforce_proxy_collision_during_generation: bool = True,
    collision_margin_m: float = 0.005,
    grasp_contact_height_offset_m: float = 0.0,
    grasp_contact_tangent_offset_world_m: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    ),
    pregrasp_clearance_m: float = 0.08,
    minimum_normalized_joint_limit_reserve: float = 0.0,
    maximum_contact_solution_body_tilt_rad: float | None = None,
    preferred_surface_port_ids: tuple[int, int] | None = None,
    preferred_candidate_group_id: str | None = None,
    excluded_surface_port_id_pairs: tuple[tuple[int, int], ...] = (),
    contact_goal_joint_seed_positions_rad: Mapping[str, float] | None = None,
    configuration_goal_joint_seed_positions_rad: (
        Mapping[str, float] | None
    ) = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> Order9C3NominalTrajectory:
    """Search contact groups until the complete local grasp path succeeds."""

    try:
        return _generate_order9_c3_nominal_grasp_trajectory_once(
            task_spec=task_spec,
            structural_target=structural_target,
            physical_model=physical_model,
            maximum_windows_per_phase=maximum_windows_per_phase,
            enforce_proxy_collision_during_generation=(
                enforce_proxy_collision_during_generation
            ),
            collision_margin_m=collision_margin_m,
            grasp_contact_height_offset_m=grasp_contact_height_offset_m,
            grasp_contact_tangent_offset_world_m=(
                grasp_contact_tangent_offset_world_m
            ),
            pregrasp_clearance_m=pregrasp_clearance_m,
            minimum_normalized_joint_limit_reserve=(
                minimum_normalized_joint_limit_reserve
            ),
            maximum_contact_solution_body_tilt_rad=(
                maximum_contact_solution_body_tilt_rad
            ),
            preferred_surface_port_ids=preferred_surface_port_ids,
            preferred_candidate_group_id=preferred_candidate_group_id,
            excluded_surface_port_id_pairs=excluded_surface_port_id_pairs,
            contact_goal_joint_seed_positions_rad=(
                contact_goal_joint_seed_positions_rad
            ),
            configuration_goal_joint_seed_positions_rad=(
                configuration_goal_joint_seed_positions_rad
            ),
            progress_callback=progress_callback,
        )
    except SchemaValidationError as initial_error:
        if preferred_candidate_group_id is not None:
            raise
        initial_failure = str(initial_error)

    collision_object = build_order9_c3_posture_collision_object(task_spec)
    probe = build_order9_c3_articulated_teacher(
        task_spec=task_spec,
        structural_target=structural_target,
        physical_model=physical_model,
        config=Order9C3TeacherConfig(
            preferred_surface_port_ids=preferred_surface_port_ids,
            excluded_surface_port_id_pairs=excluded_surface_port_id_pairs,
            collision_margin_m=collision_margin_m,
            grasp_contact_height_offset_m=grasp_contact_height_offset_m,
            grasp_contact_tangent_offset_world_m=(
                grasp_contact_tangent_offset_world_m
            ),
            pregrasp_clearance_m=pregrasp_clearance_m,
            minimum_normalized_joint_limit_reserve=(
                minimum_normalized_joint_limit_reserve
            ),
            maximum_contact_solution_body_tilt_rad=(
                maximum_contact_solution_body_tilt_rad
            ),
        ),
        collision_object=(
            collision_object if enforce_proxy_collision_during_generation else None
        ),
        contact_goal_joint_seed_positions_rad=(contact_goal_joint_seed_positions_rad),
        configuration_goal_joint_seed_positions_rad=(
            configuration_goal_joint_seed_positions_rad
        ),
    )
    first_group_id = probe.trajectory_plan.candidate_group_id
    group_ids = tuple(
        proposal.group_id
        for proposal in probe.contact_candidate_set.group_proposals
        if proposal.group_id != first_group_id
    )
    failures = [f"{first_group_id}:{initial_failure}"]
    for group_id in group_ids:
        try:
            return _generate_order9_c3_nominal_grasp_trajectory_once(
                task_spec=task_spec,
                structural_target=structural_target,
                physical_model=physical_model,
                maximum_windows_per_phase=maximum_windows_per_phase,
                enforce_proxy_collision_during_generation=(
                    enforce_proxy_collision_during_generation
                ),
                collision_margin_m=collision_margin_m,
                grasp_contact_height_offset_m=grasp_contact_height_offset_m,
                grasp_contact_tangent_offset_world_m=(
                    grasp_contact_tangent_offset_world_m
                ),
                pregrasp_clearance_m=pregrasp_clearance_m,
                minimum_normalized_joint_limit_reserve=(
                    minimum_normalized_joint_limit_reserve
                ),
                maximum_contact_solution_body_tilt_rad=(
                    maximum_contact_solution_body_tilt_rad
                ),
                preferred_surface_port_ids=preferred_surface_port_ids,
                preferred_candidate_group_id=group_id,
                excluded_surface_port_id_pairs=(excluded_surface_port_id_pairs),
                contact_goal_joint_seed_positions_rad=(
                    contact_goal_joint_seed_positions_rad
                ),
                configuration_goal_joint_seed_positions_rad=(
                    configuration_goal_joint_seed_positions_rad
                ),
                progress_callback=progress_callback,
            )
        except SchemaValidationError as error:
            failures.append(f"{group_id}:{error}")
    raise SchemaValidationError(
        "C3 nominal full-path contact-group search exhausted; "
        + "; ".join(failures[:8])
    )


def _generate_order9_c3_nominal_grasp_trajectory_once(
    *,
    task_spec: TaskSpec,
    structural_target: MorphologyGraph,
    physical_model: PhysicalModel,
    maximum_windows_per_phase: int = 32,
    enforce_proxy_collision_during_generation: bool = True,
    collision_margin_m: float = 0.005,
    grasp_contact_height_offset_m: float = 0.0,
    grasp_contact_tangent_offset_world_m: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    ),
    pregrasp_clearance_m: float = 0.08,
    minimum_normalized_joint_limit_reserve: float = 0.0,
    maximum_contact_solution_body_tilt_rad: float | None = None,
    preferred_surface_port_ids: tuple[int, int] | None = None,
    preferred_candidate_group_id: str | None = None,
    excluded_surface_port_id_pairs: tuple[tuple[int, int], ...] = (),
    contact_goal_joint_seed_positions_rad: Mapping[str, float] | None = None,
    configuration_goal_joint_seed_positions_rad: (
        Mapping[str, float] | None
    ) = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> Order9C3NominalTrajectory:
    """Chain ideal resolved endpoints from approach through grasp contact."""

    if maximum_windows_per_phase < 1:
        raise ValueError("maximum_windows_per_phase must be positive")
    if not enforce_proxy_collision_during_generation:
        raise SchemaValidationError(
            "C3 nominal trajectory generation requires convex collision-aware "
            "configuration planning"
        )
    collision_object = build_order9_c3_posture_collision_object(task_spec)
    selection = build_order9_c3_articulated_teacher(
        task_spec=task_spec,
        structural_target=structural_target,
        physical_model=physical_model,
        config=Order9C3TeacherConfig(
            preferred_surface_port_ids=preferred_surface_port_ids,
            preferred_candidate_group_id=preferred_candidate_group_id,
            excluded_surface_port_id_pairs=excluded_surface_port_id_pairs,
            collision_margin_m=collision_margin_m,
            grasp_contact_height_offset_m=grasp_contact_height_offset_m,
            grasp_contact_tangent_offset_world_m=(
                grasp_contact_tangent_offset_world_m
            ),
            pregrasp_clearance_m=pregrasp_clearance_m,
            minimum_normalized_joint_limit_reserve=(
                minimum_normalized_joint_limit_reserve
            ),
            maximum_contact_solution_body_tilt_rad=(
                maximum_contact_solution_body_tilt_rad
            ),
        ),
        collision_object=(
            collision_object if enforce_proxy_collision_during_generation else None
        ),
        contact_goal_joint_seed_positions_rad=(contact_goal_joint_seed_positions_rad),
        configuration_goal_joint_seed_positions_rad=(
            configuration_goal_joint_seed_positions_rad
        ),
    )
    built = IRGBuilder().build_with_scene_graph(task_spec)
    envelope = InteractionEnvelopeExtractor().extract(built.irg)
    morphology = selection.design_output.target_morphology
    observation = build_order9_c3_neutral_runtime_observation(
        morphology,
        physical_model,
        task_spec,
        phase_label=_PHASES_TO_GRASP[0],
    )
    teacher = Order9ArticulatedTrajectoryTeacher(
        physical_model,
        config=Order9ArticulatedTeacherConfig(
            preferred_candidate_group_id=(selection.trajectory_plan.candidate_group_id),
            collision_margin_m=collision_margin_m,
            pregrasp_clearance_m=pregrasp_clearance_m,
            minimum_normalized_joint_limit_reserve=(
                minimum_normalized_joint_limit_reserve
            ),
            maximum_contact_solution_body_tilt_rad=(
                maximum_contact_solution_body_tilt_rad
            ),
        ),
        collision_object=(
            collision_object if enforce_proxy_collision_during_generation else None
        ),
    )
    initial_object_poses = {
        value.object_id: value.pose_world for value in task_spec.scene.objects
    }
    windows: list[Order9C3NominalWindow] = []
    global_time_s = 0.0
    for phase in _PHASES_TO_GRASP:
        observation = RuntimeObservation.from_dict(observation.to_dict())
        observation.task_progress = TaskProgressState(
            phase_label=phase,
            progress_ratio=0.0,
        )
        reached = False
        for _phase_window_index in range(maximum_windows_per_phase):
            context = HighLevelPolicyContext(
                built.irg,
                envelope,
                morphology,
                selection.contact_candidate_set,
                runtime_observation=observation,
            )
            initial_q = _global_joint_positions(observation)
            plan = teacher.plan(
                context,
                initial_object_poses_world=initial_object_poses,
                nominal_start_joint_positions_rad=initial_q,
                contact_goal_joint_seed_positions_rad=(
                    contact_goal_joint_seed_positions_rad
                ),
                configuration_goal_joint_seed_positions_rad=(
                    configuration_goal_joint_seed_positions_rad
                ),
            )
            if plan.task_phase != phase:
                raise SchemaValidationError(
                    "C3 nominal teacher returned an unexpected task phase"
                )
            if plan.candidate_group_id != selection.trajectory_plan.candidate_group_id:
                raise SchemaValidationError(
                    "C3 nominal rolling plan changed the selected contact group"
                )
            window = Order9C3NominalWindow(
                window_index=len(windows),
                phase=phase,
                global_start_time_s=global_time_s,
                plan=plan,
            )
            windows.append(window)
            if progress_callback is not None:
                configuration_plan = plan.configuration_space_plan
                if configuration_plan is None:
                    raise SchemaValidationError(
                        "C3 nominal progress lacks configuration-space plan"
                    )
                plan_start = configuration_plan.states[0]
                plan_next = configuration_plan.states[1]
                plan_goal = configuration_plan.states[-1]
                ordered_ids = tuple(sorted(plan_start.joint_positions_rad))
                progress_callback(
                    {
                        "window_index": window.window_index,
                        "phase_window_index": _phase_window_index,
                        "phase": phase,
                        "phase_target_reached": plan.phase_target_reached,
                        "configuration_plan_method": (configuration_plan.method),
                        "configuration_plan_state_count": len(
                            configuration_plan.states
                        ),
                        "configuration_plan_collision_check_count": (
                            configuration_plan.collision_check_count
                        ),
                        "start_base_pose_world": list(plan_start.base_pose_world),
                        "next_base_pose_world": list(plan_next.base_pose_world),
                        "goal_base_pose_world": list(plan_goal.base_pose_world),
                        "start_joint_hash": stable_hash(plan_start.joint_positions_rad),
                        "next_joint_hash": stable_hash(plan_next.joint_positions_rad),
                        "goal_joint_hash": stable_hash(plan_goal.joint_positions_rad),
                        "maximum_start_next_joint_delta_rad": max(
                            (
                                abs(
                                    float(plan_next.joint_positions_rad[joint_id])
                                    - float(plan_start.joint_positions_rad[joint_id])
                                )
                                for joint_id in ordered_ids
                            ),
                            default=0.0,
                        ),
                        "maximum_start_goal_joint_delta_rad": max(
                            (
                                abs(
                                    float(plan_goal.joint_positions_rad[joint_id])
                                    - float(plan_start.joint_positions_rad[joint_id])
                                )
                                for joint_id in ordered_ids
                            ),
                            default=0.0,
                        ),
                    }
                )
            observation = _ideal_endpoint_observation(
                context=context,
                plan=plan,
                physical_model=physical_model,
                next_phase=phase,
            )
            global_time_s += float(plan.trajectory.horizon_s)
            if plan.phase_target_reached:
                reached = True
                break
        if not reached:
            raise SchemaValidationError(
                "C3 nominal ideal-tracking rollout exceeded its bounded "
                f"window budget in phase {phase!r}"
            )
    timeline = _flatten_nominal_windows(windows)
    result = Order9C3NominalTrajectory(
        selection_bundle=selection,
        windows=tuple(windows),
        timeline=timeline,
        final_observation=observation,
        proxy_collision_validation_status=(
            "enforced_during_generation"
            if enforce_proxy_collision_during_generation
            else "pending_offline_admission"
        ),
    )
    if result.windows[-1].phase != "contact_acquisition":
        raise SchemaValidationError(
            "C3 nominal ideal-tracking rollout ended before grasp acquisition"
        )
    if not result.windows[-1].plan.phase_target_reached:
        raise SchemaValidationError(
            "C3 nominal ideal-tracking rollout did not reach the grasp target"
        )
    return result


def materialize_order9_c3_complete_task_phases(
    *,
    phase_trajectories: Mapping[str, ContactWrenchTrajectory],
    task_spec: TaskSpec,
    lift_clearance_m: float,
    retreat_offset_m: float,
    phase_duration_s: Mapping[str, float] | None = None,
    nominal_dt_s: float = 0.1,
) -> dict[str, ContactWrenchTrajectory]:
    """Freeze all C3 task phases as offline, collision-checkable references.

    Approach and contact acquisition are the accepted configuration-space
    planner output.  The remaining task edges are deterministic.  Release is
    deliberately performed while ascending by one lift clearance, followed by
    an elevated retreat, so opening a morphology cannot sweep through the
    payload support surface.
    """

    if set(phase_trajectories) != set(_PHASES_TO_GRASP):
        raise SchemaValidationError(
            "C3 complete-task materialization requires approach/contact input"
        )
    durations = dict(
        Order9ObjectTaskRuntimeConfig().phase_duration_s
        if phase_duration_s is None
        else phase_duration_s
    )
    if set(durations) != set(_COMPLETE_TASK_PHASES) or any(
        not math.isfinite(float(value)) or float(value) <= 0.0
        for value in durations.values()
    ):
        raise SchemaValidationError("C3 complete-task phase durations are invalid")
    for name, value in (
        ("lift_clearance_m", lift_clearance_m),
        ("retreat_offset_m", retreat_offset_m),
        ("nominal_dt_s", nominal_dt_s),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise SchemaValidationError(f"C3 complete-task {name} is invalid")
    approach = phase_trajectories[Order9ObjectTaskPhase.APPROACH.value]
    contact = phase_trajectories[Order9ObjectTaskPhase.CONTACT_ACQUISITION.value]
    approach.validate()
    contact.validate()
    contact_start = contact.knots[0]
    contact_end = contact.knots[-1]
    start_body, _, open_q, _ = _complete_task_knot_state(contact_start)
    grasp_body, _, grasp_q, _ = _complete_task_knot_state(contact_end)
    del start_body

    object_targets = [
        goal
        for goal in task_spec.goals
        if goal.goal_type == "object_pose"
        and goal.target_entity_id is not None
        and goal.target_pose_world is not None
    ]
    if len(object_targets) != 1:
        raise SchemaValidationError(
            "C3 complete-task materialization requires one object-pose goal"
        )
    goal = object_targets[0]
    scene_objects = [
        value
        for value in task_spec.scene.objects
        if value.object_id == goal.target_entity_id
    ]
    if len(scene_objects) != 1:
        raise SchemaValidationError(
            "C3 complete-task target object is missing from the scene"
        )
    object_id = scene_objects[0].object_id
    object_start = tuple(float(value) for value in scene_objects[0].pose_world)
    object_goal = tuple(float(value) for value in goal.target_pose_world)
    approach = _bind_static_object_target(
        approach,
        object_id=object_id,
        object_pose=object_start,
    )
    contact = _bind_static_object_target(
        contact,
        object_id=object_id,
        object_pose=object_start,
    )
    lift = float(lift_clearance_m)
    del retreat_offset_m
    object_to_grasp_body = compose_pose(inverse_pose(object_start), grasp_body)
    grasp_lifted = _translated_pose(grasp_body, z=lift)
    body_at_goal = compose_pose(
        object_goal,
        object_to_grasp_body,
    )
    body_at_goal_lifted = _translated_pose(body_at_goal, z=lift)
    object_lifted = _translated_pose(object_start, z=lift)
    object_goal_lifted = _translated_pose(object_goal, z=lift)

    result = {
        Order9ObjectTaskPhase.APPROACH.value: approach,
        Order9ObjectTaskPhase.CONTACT_ACQUISITION.value: contact,
    }
    generated = (
        (
            Order9ObjectTaskPhase.LIFT.value,
            grasp_body,
            grasp_lifted,
            grasp_q,
            grasp_q,
            object_start,
            object_lifted,
            "maintain",
        ),
        (
            Order9ObjectTaskPhase.TRANSPORT.value,
            grasp_lifted,
            body_at_goal_lifted,
            grasp_q,
            grasp_q,
            object_lifted,
            object_goal_lifted,
            "maintain",
        ),
        (
            Order9ObjectTaskPhase.PLACE.value,
            body_at_goal_lifted,
            body_at_goal,
            grasp_q,
            grasp_q,
            object_goal_lifted,
            object_goal,
            "maintain",
        ),
    )
    for (
        phase,
        body_start,
        body_end,
        q_start,
        q_end,
        object_phase_start,
        object_phase_end,
        schedule_state,
    ) in generated:
        result[phase] = _build_complete_task_phase_trajectory(
            template=contact_end,
            phase=phase,
            duration_s=float(durations[phase]),
            dt_s=float(nominal_dt_s),
            body_start=body_start,
            body_end=body_end,
            q_start=q_start,
            q_end=q_end,
            object_id=object_id,
            object_start=object_phase_start,
            object_end=object_phase_end,
            contact_schedule_state=schedule_state,
            anchor_pose_translation_origin=object_start,
            contract_version=contact.contract_version,
            preserve_body_object_relative=True,
        )
    release = _build_reversed_accepted_phase_trajectory(
        source=contact,
        phase=Order9ObjectTaskPhase.RELEASE.value,
        duration_s=float(durations[Order9ObjectTaskPhase.RELEASE.value]),
        dt_s=float(nominal_dt_s),
        source_object_pose=object_start,
        target_object_pose=object_goal,
        added_clearance_start_m=0.0,
        added_clearance_end_m=lift,
        contact_schedule_state="release",
    )
    retreat = _build_reversed_accepted_phase_trajectory(
        source=approach,
        phase=Order9ObjectTaskPhase.RETREAT.value,
        duration_s=float(durations[Order9ObjectTaskPhase.RETREAT.value]),
        dt_s=float(nominal_dt_s),
        source_object_pose=object_start,
        target_object_pose=object_goal,
        added_clearance_start_m=lift,
        added_clearance_end_m=lift,
        contact_schedule_state="inactive",
    )
    result[Order9ObjectTaskPhase.RELEASE.value] = release
    result[Order9ObjectTaskPhase.RETREAT.value] = retreat
    retreat_end_body, _, retreat_end_q, _ = _complete_task_knot_state(retreat.knots[-1])
    result[Order9ObjectTaskPhase.SETTLE.value] = _build_complete_task_phase_trajectory(
        template=retreat.knots[-1],
        phase=Order9ObjectTaskPhase.SETTLE.value,
        duration_s=float(durations[Order9ObjectTaskPhase.SETTLE.value]),
        dt_s=float(nominal_dt_s),
        body_start=retreat_end_body,
        body_end=retreat_end_body,
        q_start=retreat_end_q,
        q_end=retreat_end_q,
        object_id=object_id,
        object_start=object_goal,
        object_end=object_goal,
        contact_schedule_state="inactive",
        anchor_pose_translation_origin=object_goal,
        contract_version=contact.contract_version,
    )
    if tuple(result) != _COMPLETE_TASK_PHASES:
        raise SchemaValidationError("C3 complete-task trajectory phase order changed")
    return result


def _bind_static_object_target(
    trajectory: ContactWrenchTrajectory,
    *,
    object_id: str,
    object_pose: Sequence[float],
) -> ContactWrenchTrajectory:
    knots = []
    for knot in trajectory.knots:
        payload = knot.to_dict()
        payload["object_targets"] = [
            {
                "object_id": object_id,
                "pose_target_world": [float(value) for value in object_pose],
                "twist_target_world": [0.0] * 6,
            }
        ]
        knots.append(InteractionKnot.from_dict(payload))
    value = ContactWrenchTrajectory(
        horizon_s=float(trajectory.horizon_s),
        dt_s=float(trajectory.dt_s),
        knots=knots,
        derived_mode_label=trajectory.derived_mode_label,
        contract_version=trajectory.contract_version,
    )
    value.validate()
    return value


def _build_reversed_accepted_phase_trajectory(
    *,
    source: ContactWrenchTrajectory,
    phase: str,
    duration_s: float,
    dt_s: float,
    source_object_pose: Sequence[float],
    target_object_pose: Sequence[float],
    added_clearance_start_m: float,
    added_clearance_end_m: float,
    contact_schedule_state: str,
) -> ContactWrenchTrajectory:
    """Time-reverse one accepted path under a rigid object-frame transform."""

    knot_count = max(2, int(math.ceil(duration_s / dt_s)) + 1)
    actual_dt_s = duration_s / float(knot_count - 1)
    source_times = [float(knot.t_rel_s) for knot in source.knots]
    object_delta = compose_pose(
        tuple(float(value) for value in target_object_pose),
        inverse_pose(tuple(float(value) for value in source_object_pose)),
    )
    knots = []
    upper_index = len(source.knots) - 1
    for index in range(knot_count):
        progress = index / float(knot_count - 1)
        source_time = (1.0 - progress) * float(source.horizon_s)
        while upper_index > 0 and source_times[upper_index - 1] >= source_time:
            upper_index -= 1
        if upper_index == 0:
            lower_index = 0
            upper_index = 1
        else:
            lower_index = upper_index - 1
        lower = source.knots[lower_index]
        upper = source.knots[upper_index]
        span = max(float(upper.t_rel_s) - float(lower.t_rel_s), 1.0e-12)
        alpha = min(
            max((source_time - float(lower.t_rel_s)) / span, 0.0),
            1.0,
        )
        lower_pose, _, lower_q, _ = _complete_task_knot_state(lower)
        upper_pose, _, upper_q, _ = _complete_task_knot_state(upper)
        source_pose = (
            *[
                (1.0 - alpha) * float(lower_pose[axis])
                + alpha * float(upper_pose[axis])
                for axis in range(3)
            ],
            *_normalized_quaternion_lerp(lower_pose[3:7], upper_pose[3:7], alpha),
        )
        body_pose = list(compose_pose(object_delta, source_pose))
        clearance = (1.0 - progress) * float(
            added_clearance_start_m
        ) + progress * float(added_clearance_end_m)
        body_pose[2] += clearance
        q = {
            key: (1.0 - alpha) * float(value) + alpha * float(upper_q[key])
            for key, value in lower_q.items()
        }
        payload = lower.to_dict()
        payload["t_rel_s"] = index * actual_dt_s
        centroidal = dict(payload.get("centroidal_target") or {})
        centroidal["com_pos_world"] = body_pose[:3]
        centroidal["com_vel_world"] = [0.0, 0.0, 0.0]
        centroidal["body_orientation_world"] = body_pose[3:7]
        payload["centroidal_target"] = centroidal
        posture = dict(payload.get("posture_target") or {})
        posture["joint_pos_target"] = q
        posture["joint_vel_target"] = {key: 0.0 for key in q}
        free_anchors = posture.get("free_anchor_pose_targets")
        if isinstance(free_anchors, Mapping):
            transformed = {}
            for key, value in free_anchors.items():
                pose = list(
                    compose_pose(
                        object_delta,
                        tuple(float(item) for item in value),
                    )
                )
                pose[2] += clearance
                transformed[key] = pose
            posture["free_anchor_pose_targets"] = transformed
        payload["posture_target"] = posture
        assignments = []
        if contact_schedule_state != "inactive":
            for assignment in lower.contact_assignments:
                value = assignment.to_dict()
                value["schedule_state"] = contact_schedule_state
                assignments.append(value)
        payload["contact_assignments"] = assignments
        payload["object_targets"] = [
            {
                "object_id": source.knots[0].object_targets[0].object_id,
                "pose_target_world": [float(value) for value in target_object_pose],
                "twist_target_world": [0.0] * 6,
            }
        ]
        payload["guard_conditions"] = []
        knots.append(InteractionKnot.from_dict(payload))
    trajectory = ContactWrenchTrajectory(
        horizon_s=duration_s,
        dt_s=actual_dt_s,
        knots=knots,
        derived_mode_label=(f"order9_c3_reversed_accepted_clearance_phase:{phase}"),
        contract_version=source.contract_version,
    )
    trajectory.validate()
    return trajectory


def _build_complete_task_phase_trajectory(
    *,
    template: InteractionKnot,
    phase: str,
    duration_s: float,
    dt_s: float,
    body_start: Sequence[float],
    body_end: Sequence[float],
    q_start: Mapping[str, float],
    q_end: Mapping[str, float],
    object_id: str,
    object_start: Sequence[float],
    object_end: Sequence[float],
    contact_schedule_state: str,
    anchor_pose_translation_origin: Sequence[float],
    contract_version: str,
    preserve_body_object_relative: bool = False,
) -> ContactWrenchTrajectory:
    knot_count = max(2, int(math.ceil(duration_s / dt_s)) + 1)
    actual_dt_s = duration_s / float(knot_count - 1)
    body_delta = [
        float(body_end[index]) - float(body_start[index]) for index in range(3)
    ]
    object_delta = [
        float(object_end[index]) - float(object_start[index]) for index in range(3)
    ]
    q_delta = {key: float(q_end[key]) - float(value) for key, value in q_start.items()}
    if set(q_delta) != set(q_end):
        raise SchemaValidationError(
            "C3 complete-task joint identities changed within a phase"
        )
    body_relative_to_object = (
        compose_pose(inverse_pose(object_start), body_start)
        if preserve_body_object_relative
        else None
    )
    knots = []
    for index in range(knot_count):
        progress = index / float(knot_count - 1)
        smooth = progress * progress * (3.0 - 2.0 * progress)
        derivative = 6.0 * progress * (1.0 - progress)
        body_pose = [
            float(body_start[axis]) + smooth * body_delta[axis] for axis in range(3)
        ] + list(_normalized_quaternion_lerp(body_start[3:7], body_end[3:7], smooth))
        object_pose = [
            float(object_start[axis]) + smooth * object_delta[axis] for axis in range(3)
        ] + list(
            _normalized_quaternion_lerp(object_start[3:7], object_end[3:7], smooth)
        )
        if body_relative_to_object is not None:
            body_pose = list(compose_pose(tuple(object_pose), body_relative_to_object))
        q = {
            key: float(value) + smooth * q_delta[key] for key, value in q_start.items()
        }
        qdot = {key: q_delta[key] / duration_s * derivative for key in q_start}
        payload = template.to_dict()
        payload["t_rel_s"] = index * actual_dt_s
        centroidal = dict(payload.get("centroidal_target") or {})
        centroidal["com_pos_world"] = body_pose[:3]
        centroidal["com_vel_world"] = [
            value / duration_s * derivative for value in body_delta
        ]
        centroidal["body_orientation_world"] = body_pose[3:7]
        payload["centroidal_target"] = centroidal
        posture = dict(payload.get("posture_target") or {})
        posture["joint_pos_target"] = q
        posture["joint_vel_target"] = qdot
        free_anchors = posture.get("free_anchor_pose_targets")
        if isinstance(free_anchors, Mapping):
            posture["free_anchor_pose_targets"] = {
                key: list(
                    compose_pose(
                        tuple(object_pose),
                        compose_pose(
                            inverse_pose(
                                tuple(
                                    float(item)
                                    for item in anchor_pose_translation_origin
                                )
                            ),
                            tuple(float(item) for item in value),
                        ),
                    )
                )
                for key, value in free_anchors.items()
            }
        payload["posture_target"] = posture
        assignments = []
        if contact_schedule_state != "inactive":
            for assignment in template.contact_assignments:
                value = assignment.to_dict()
                value["schedule_state"] = contact_schedule_state
                assignments.append(value)
        payload["contact_assignments"] = assignments
        payload["object_targets"] = [
            {
                "object_id": object_id,
                "pose_target_world": object_pose,
                "twist_target_world": [
                    *[value / duration_s * derivative for value in object_delta],
                    0.0,
                    0.0,
                    0.0,
                ],
            }
        ]
        payload["guard_conditions"] = []
        knots.append(InteractionKnot.from_dict(payload))
    trajectory = ContactWrenchTrajectory(
        horizon_s=duration_s,
        dt_s=actual_dt_s,
        knots=knots,
        derived_mode_label=f"order9_c3_complete_task_phase:{phase}",
        contract_version=contract_version,
    )
    trajectory.validate()
    return trajectory


def _complete_task_knot_state(
    knot: InteractionKnot,
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    dict[str, float],
    dict[str, float],
]:
    centroidal = knot.centroidal_target
    posture = knot.posture_target
    if (
        centroidal is None
        or centroidal.com_pos_world is None
        or centroidal.body_orientation_world is None
        or posture is None
        or posture.joint_pos_target is None
        or posture.joint_vel_target is None
    ):
        raise SchemaValidationError(
            "C3 complete-task source knot lacks centroidal/joint targets"
        )
    return (
        tuple(float(value) for value in centroidal.com_pos_world)
        + tuple(float(value) for value in centroidal.body_orientation_world),
        tuple(float(value) for value in (centroidal.com_vel_world or (0.0, 0.0, 0.0)))
        + (0.0, 0.0, 0.0),
        {str(key): float(value) for key, value in posture.joint_pos_target.items()},
        {str(key): float(value) for key, value in posture.joint_vel_target.items()},
    )


def _translated_pose(
    pose: Sequence[float],
    *,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
) -> tuple[float, ...]:
    return (
        float(pose[0]) + float(x),
        float(pose[1]) + float(y),
        float(pose[2]) + float(z),
        *[float(value) for value in pose[3:7]],
    )


def _normalized_quaternion_lerp(
    start: Sequence[float],
    end: Sequence[float],
    progress: float,
) -> tuple[float, float, float, float]:
    left = [float(value) for value in start]
    right = [float(value) for value in end]
    if sum(a * b for a, b in zip(left, right)) < 0.0:
        right = [-value for value in right]
    value = [(1.0 - progress) * a + progress * b for a, b in zip(left, right)]
    norm = math.sqrt(sum(item * item for item in value))
    if norm <= 1.0e-12:
        raise SchemaValidationError("C3 complete-task quaternion is singular")
    return tuple(item / norm for item in value)


def _phase_trajectories_from_nominal_windows(
    windows: Sequence[Order9C3NominalWindow],
) -> dict[str, ContactWrenchTrajectory]:
    return _join_phase_windows(
        (
            (
                window.phase,
                float(window.global_start_time_s),
                window.plan.trajectory,
            )
            for window in windows
        )
    )


def _flatten_complete_phase_trajectories(
    phases: Mapping[str, ContactWrenchTrajectory],
) -> tuple[dict[str, Any], ...]:
    records = []
    offset = 0.0
    for phase in _COMPLETE_TASK_PHASES:
        trajectory = phases[phase]
        for knot_index, knot in enumerate(trajectory.knots):
            if records and knot_index == 0:
                continue
            records.append(
                {
                    "sample_index": len(records),
                    "global_time_s": offset + float(knot.t_rel_s),
                    "phase": phase,
                    "phase_local_time_s": float(knot.t_rel_s),
                    "phase_target_reached": True,
                    "knot": knot.to_dict(),
                }
            )
        offset += float(trajectory.horizon_s)
    return tuple(records)


def write_order9_c3_nominal_trajectory_artifact(
    result: Order9C3NominalTrajectory,
    *,
    output_dir: str | Path,
    bucket_id: str,
    split: DatasetSplit,
    task_spec: TaskSpec,
    lift_clearance_m: float,
    task_spec_sha256: str,
    structural_hash: str,
    physical_model_hash: str,
    robot_urdf_path: str | Path,
    retreat_offset_m: float | None = None,
    phase_duration_s: Mapping[str, float] | None = None,
) -> Order9C3NominalTrajectoryArtifact:
    """Persist raw windows plus the complete eight-phase nominal contract."""

    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"C3 nominal trajectory output exists: {destination}")
    destination.mkdir(parents=True)
    urdf_path = Path(robot_urdf_path).resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(urdf_path)
    morphology_path = destination / "task_conditioned_morphology.json"
    candidates_path = destination / "contact_candidate_set.json"
    timeline_path = destination / "nominal_timeline.json"
    grasp_phases = _phase_trajectories_from_nominal_windows(result.windows)
    complete_phases = materialize_order9_c3_complete_task_phases(
        phase_trajectories=grasp_phases,
        task_spec=task_spec,
        lift_clearance_m=lift_clearance_m,
        retreat_offset_m=(
            Order9ObjectTaskRuntimeConfig().retreat_offset_m
            if retreat_offset_m is None
            else float(retreat_offset_m)
        ),
        phase_duration_s=phase_duration_s,
    )
    complete_timeline = _flatten_complete_phase_trajectories(complete_phases)
    complete_duration_s = sum(
        float(complete_phases[phase].horizon_s) for phase in _COMPLETE_TASK_PHASES
    )
    _write_new_text(
        morphology_path,
        result.selection_bundle.design_output.target_morphology.to_json(indent=2)
        + "\n",
    )
    _write_new_text(
        candidates_path,
        result.selection_bundle.contact_candidate_set.to_json(indent=2) + "\n",
    )
    _write_new_text(
        timeline_path,
        _json_payload(
            {
                "timeline_version": ORDER9_C3_NOMINAL_TRAJECTORY_VERSION,
                "semantic_scope": (
                    "ideal exact tracking of chained phase-local raw pi_H "
                    "proposals and deterministic IK nominal trajectories; "
                    "not dynamics, learned-policy success, or offline "
                    "collision admission"
                ),
                "proxy_collision_validation_status": (
                    result.proxy_collision_validation_status
                ),
                "duration_s": complete_duration_s,
                "records": list(complete_timeline),
            }
        ),
    )
    window_artifacts = []
    for window in result.windows:
        configuration_plan = window.plan.configuration_space_plan
        if configuration_plan is None:
            raise SchemaValidationError(
                "C3 nominal window lacks configuration-space plan evidence"
            )
        window_dir = destination / "windows" / f"{window.window_index:03d}"
        window_dir.mkdir(parents=True)
        raw_path = window_dir / "raw_pi_h_trajectory.json"
        resolved_path = window_dir / "resolved_nominal_trajectory.json"
        evidence_path = window_dir / "posture_resolution_evidence.json"
        _write_new_text(
            raw_path,
            window.plan.raw_trajectory.to_json(indent=2) + "\n",
        )
        _write_new_text(
            resolved_path,
            window.plan.trajectory.to_json(indent=2) + "\n",
        )
        _write_new_text(
            evidence_path,
            _json_payload(window.plan.posture_resolution.evidence.to_dict()),
        )
        window_artifacts.append(
            Order9C3NominalWindowArtifact(
                window_index=window.window_index,
                phase=window.phase,
                global_start_time_s=window.global_start_time_s,
                horizon_s=float(window.plan.trajectory.horizon_s),
                phase_target_reached=window.plan.phase_target_reached,
                raw_trajectory_path=_relative(raw_path, destination),
                raw_trajectory_sha256=hash_file(raw_path),
                raw_trajectory_hash=stable_hash(window.plan.raw_trajectory.to_dict()),
                resolved_trajectory_path=_relative(resolved_path, destination),
                resolved_trajectory_sha256=hash_file(resolved_path),
                resolved_trajectory_hash=stable_hash(window.plan.trajectory.to_dict()),
                resolver_evidence_path=_relative(evidence_path, destination),
                resolver_evidence_sha256=hash_file(evidence_path),
                raw_knot_count=len(window.plan.raw_trajectory.knots),
                resolved_knot_count=len(window.plan.trajectory.knots),
                configuration_planner_version=(configuration_plan.planner_version),
                configuration_plan_method=configuration_plan.method,
                configuration_plan_state_count=len(configuration_plan.states),
                configuration_plan_collision_check_count=(
                    configuration_plan.collision_check_count
                ),
                configuration_plan_sampled_state_count=(
                    configuration_plan.sampled_state_count
                ),
            )
        )
    phase_artifacts = []
    for phase in _COMPLETE_TASK_PHASES:
        trajectory = complete_phases[phase]
        phase_path = destination / "phases" / f"{phase}.json"
        _write_new_text(phase_path, trajectory.to_json(indent=2) + "\n")
        phase_artifacts.append(
            Order9C3NominalPhaseArtifact(
                phase=phase,
                trajectory_path=_relative(phase_path, destination),
                trajectory_sha256=hash_file(phase_path),
                trajectory_hash=stable_hash(trajectory.to_dict()),
                knot_count=len(trajectory.knots),
                generation_method=(
                    "configuration_space_planner_resolved"
                    if phase in _PHASES_TO_GRASP
                    else "deterministic_complete_task_edge"
                ),
            )
        )
    manifest = Order9C3NominalTrajectoryArtifact(
        bucket_id=bucket_id,
        split=split,
        physical_model_hash=physical_model_hash,
        task_spec_sha256=task_spec_sha256,
        structural_hash=structural_hash,
        robot_urdf_path=str(urdf_path),
        robot_urdf_sha256=hash_file(urdf_path),
        task_conditioned_morphology_path=_relative(morphology_path, destination),
        task_conditioned_morphology_sha256=hash_file(morphology_path),
        task_conditioned_morphology_hash=(
            result.selection_bundle.design_output.target_morphology.stable_hash()
        ),
        contact_candidate_set_path=_relative(candidates_path, destination),
        contact_candidate_set_sha256=hash_file(candidates_path),
        contact_candidate_set_hash=stable_hash(
            result.selection_bundle.contact_candidate_set.to_dict()
        ),
        timeline_path=_relative(timeline_path, destination),
        timeline_sha256=hash_file(timeline_path),
        selected_surface_port_ids=list(
            result.selection_bundle.selected_surface_port_ids
        ),
        selected_candidate_group_id=(
            result.selection_bundle.trajectory_plan.candidate_group_id
        ),
        duration_s=complete_duration_s,
        final_phase=Order9ObjectTaskPhase.SETTLE.value,
        final_phase_target_reached=True,
        proxy_collision_validation_status=(result.proxy_collision_validation_status),
        windows=window_artifacts,
        phase_trajectories=phase_artifacts,
        selection_evidence=order9_c3_teacher_evidence(result.selection_bundle),
    )
    manifest.validate()
    _write_new_text(
        destination / "manifest.json",
        manifest.to_json(indent=2) + "\n",
    )
    validate_order9_c3_nominal_trajectory_artifact_bytes(destination / "manifest.json")
    return manifest


def validate_order9_c3_nominal_trajectory_artifact_bytes(
    manifest_path: str | Path,
) -> Order9C3NominalTrajectoryArtifact:
    source = Path(manifest_path).resolve()
    if source.is_dir():
        source = source / "manifest.json"
    artifact = Order9C3NominalTrajectoryArtifact.from_json(
        source.read_text(encoding="utf-8")
    )
    artifact.validate()
    root = source.parent
    checks = [
        (
            root / artifact.task_conditioned_morphology_path,
            artifact.task_conditioned_morphology_sha256,
        ),
        (
            root / artifact.contact_candidate_set_path,
            artifact.contact_candidate_set_sha256,
        ),
        (root / artifact.timeline_path, artifact.timeline_sha256),
        (Path(artifact.robot_urdf_path), artifact.robot_urdf_sha256),
    ]
    for window in artifact.windows:
        checks.extend(
            (
                (
                    root / window.raw_trajectory_path,
                    window.raw_trajectory_sha256,
                ),
                (
                    root / window.resolved_trajectory_path,
                    window.resolved_trajectory_sha256,
                ),
                (
                    root / window.resolver_evidence_path,
                    window.resolver_evidence_sha256,
                ),
            )
        )
    for phase in artifact.phase_trajectories:
        checks.append(
            (
                root / phase.trajectory_path,
                phase.trajectory_sha256,
            )
        )
    for path, expected in checks:
        if not path.is_file() or hash_file(path) != expected:
            raise SchemaValidationError(f"C3 nominal trajectory bytes changed: {path}")
    return artifact


def validate_order9_c3_nominal_trajectory_set_bytes(
    manifest_path: str | Path,
    *,
    repository_root: str | Path,
    expected_sha256: str | None = None,
) -> Order9C3NominalTrajectorySetManifest:
    """Validate a complete nominal set and every runtime-consumed byte."""

    source = Path(manifest_path).resolve()
    if source.is_dir():
        source = source / "manifest.json"
    if expected_sha256 is not None and hash_file(source) != expected_sha256:
        raise SchemaValidationError("C3 nominal trajectory set hash mismatch")
    manifest = Order9C3NominalTrajectorySetManifest.from_json(
        source.read_text(encoding="utf-8")
    )
    manifest.validate()
    repository = Path(repository_root).resolve()
    bucket_manifest_path = _resolve_repository_path(
        manifest.bucket_manifest_path,
        repository,
    )
    if hash_file(bucket_manifest_path) != manifest.bucket_manifest_sha256:
        raise SchemaValidationError("C3 nominal source bucket manifest changed")
    bucket_payload = json.loads(bucket_manifest_path.read_text(encoding="utf-8"))
    buckets = bucket_payload.get("buckets")
    if not isinstance(buckets, list) or len(buckets) != len(manifest.entries):
        raise SchemaValidationError(
            "C3 nominal set does not cover its source bucket manifest"
        )
    for index, (entry, bucket) in enumerate(zip(manifest.entries, buckets)):
        if (
            entry.bucket_id != bucket.get("bucket_id")
            or entry.split.value != bucket.get("split")
            or entry.module_count != int(bucket.get("module_count", -1))
            or entry.structural_hash != bucket.get("structural_hash")
        ):
            raise SchemaValidationError(
                f"C3 nominal entry identity differs at index {index}"
            )
        artifact_path = source.parent / entry.artifact_path
        if hash_file(artifact_path) != entry.artifact_sha256:
            raise SchemaValidationError(
                f"C3 nominal artifact hash changed: {entry.bucket_id}"
            )
        artifact = validate_order9_c3_nominal_trajectory_artifact_bytes(artifact_path)
        if (
            artifact.bucket_id != entry.bucket_id
            or artifact.split != entry.split
            or artifact.structural_hash != entry.structural_hash
            or artifact.physical_model_hash != manifest.physical_model_hash
            or artifact.task_spec_sha256 != bucket.get("task_spec_sha256")
        ):
            raise SchemaValidationError(
                f"C3 nominal artifact semantics differ: {entry.bucket_id}"
            )
        if entry.animation_scene_path is not None:
            scene_path = source.parent / entry.animation_scene_path
            if hash_file(scene_path) != entry.animation_scene_sha256:
                raise SchemaValidationError(
                    f"C3 nominal review scene changed: {entry.bucket_id}"
                )
            viewer_path = source.parent / str(entry.animation_html_path)
            if not viewer_path.is_file():
                raise SchemaValidationError(
                    f"C3 nominal review viewer is missing: {entry.bucket_id}"
                )
        _validate_component_collision_admission(
            artifact_path,
            bucket_id=entry.bucket_id,
        )
    return manifest


def load_order9_c3_accepted_nominal_bundle(
    manifest_path: str | Path,
    *,
    bucket_id: str,
    repository_root: str | Path,
    expected_set_sha256: str,
    expected_artifact_sha256: str | None = None,
    expected_structural_hash: str | None = None,
    expected_task_spec_sha256: str | None = None,
    expected_physical_model_hash: str | None = None,
) -> Order9C3AcceptedNominalBundle:
    """Load one accepted dense IK trajectory without invoking its planner."""

    source = Path(manifest_path).resolve()
    if source.is_dir():
        source = source / "manifest.json"
    manifest = validate_order9_c3_nominal_trajectory_set_bytes(
        source,
        repository_root=repository_root,
        expected_sha256=expected_set_sha256,
    )
    matches = [entry for entry in manifest.entries if entry.bucket_id == bucket_id]
    if len(matches) != 1:
        raise SchemaValidationError(
            f"C3 nominal set does not contain bucket exactly once: {bucket_id}"
        )
    entry = matches[0]
    if (
        expected_artifact_sha256 is not None
        and entry.artifact_sha256 != expected_artifact_sha256
    ):
        raise SchemaValidationError("C3 nominal bucket artifact binding changed")
    if (
        expected_structural_hash is not None
        and entry.structural_hash != expected_structural_hash
    ):
        raise SchemaValidationError("C3 nominal bucket structure changed")
    artifact_path = source.parent / entry.artifact_path
    artifact = validate_order9_c3_nominal_trajectory_artifact_bytes(artifact_path)
    if (
        expected_task_spec_sha256 is not None
        and artifact.task_spec_sha256 != expected_task_spec_sha256
    ):
        raise SchemaValidationError("C3 nominal bucket task changed")
    if (
        expected_physical_model_hash is not None
        and artifact.physical_model_hash != expected_physical_model_hash
    ):
        raise SchemaValidationError("C3 nominal physical model changed")
    root = artifact_path.parent
    morphology_path = root / artifact.task_conditioned_morphology_path
    candidates_path = root / artifact.contact_candidate_set_path
    morphology = MorphologyGraph.from_json(morphology_path.read_text(encoding="utf-8"))
    candidates = ContactCandidateSet.from_json(
        candidates_path.read_text(encoding="utf-8")
    )
    morphology.validate()
    candidates.validate()
    if (
        morphology.stable_hash() != artifact.task_conditioned_morphology_hash
        or stable_hash(candidates.to_dict()) != artifact.contact_candidate_set_hash
    ):
        raise SchemaValidationError("C3 nominal runtime semantics changed")
    phase_trajectories = _load_nominal_phase_trajectories(root, artifact)
    trajectory = _join_nominal_phase_trajectories(artifact, phase_trajectories)
    return Order9C3AcceptedNominalBundle(
        set_manifest_path=source,
        set_manifest_sha256=expected_set_sha256,
        entry=entry,
        artifact_path=artifact_path,
        artifact=artifact,
        morphology=morphology,
        contact_candidate_set=candidates,
        trajectory=trajectory,
        phase_trajectories=phase_trajectories,
        provenance={
            "execution_semantics": "offline_precomputed_nominal_replay",
            "configuration_space_planner_runtime_enabled": False,
            "set_manifest_path": _portable_repository_path(
                source, Path(repository_root).resolve()
            ),
            "set_manifest_sha256": expected_set_sha256,
            "artifact_path": _portable_repository_path(
                artifact_path, Path(repository_root).resolve()
            ),
            "artifact_sha256": entry.artifact_sha256,
            "timeline_sha256": artifact.timeline_sha256,
            "nominal_knot_dt_s": float(trajectory.dt_s),
            "selected_surface_port_ids": list(artifact.selected_surface_port_ids),
        },
    )


def _load_nominal_phase_trajectories(
    root: Path,
    artifact: Order9C3NominalTrajectoryArtifact,
) -> dict[str, ContactWrenchTrajectory]:
    phase_trajectories = {}
    for phase_artifact in artifact.phase_trajectories:
        path = root / phase_artifact.trajectory_path
        trajectory = ContactWrenchTrajectory.from_json(path.read_text(encoding="utf-8"))
        trajectory.validate()
        if (
            stable_hash(trajectory.to_dict()) != phase_artifact.trajectory_hash
            or len(trajectory.knots) != phase_artifact.knot_count
        ):
            raise SchemaValidationError("C3 nominal complete phase trajectory changed")
        phase_trajectories[phase_artifact.phase] = trajectory
    if tuple(phase_trajectories) != _COMPLETE_TASK_PHASES:
        raise SchemaValidationError(
            "C3 accepted nominal set must contain all task phases"
        )
    return phase_trajectories


def _join_phase_windows(
    windows: Iterable[tuple[str, float, ContactWrenchTrajectory]],
) -> dict[str, ContactWrenchTrajectory]:
    by_phase: dict[str, list[tuple[float, ContactWrenchTrajectory]]] = {}
    phase_start: dict[str, float] = {}
    for phase, global_start, trajectory in windows:
        trajectory.validate()
        by_phase.setdefault(phase, []).append((global_start, trajectory))
        phase_start.setdefault(phase, global_start)
    phase_trajectories: dict[str, ContactWrenchTrajectory] = {}
    for phase, windows in by_phase.items():
        origin = phase_start[phase]
        knots: list[InteractionKnot] = []
        for global_start, trajectory in windows:
            for knot_index, knot in enumerate(trajectory.knots):
                time_s = global_start - origin + float(knot.t_rel_s)
                if (
                    knots
                    and knot_index == 0
                    and abs(float(knots[-1].t_rel_s) - time_s) <= 1.0e-9
                ):
                    continue
                payload = knot.to_dict()
                payload["t_rel_s"] = time_s
                knots.append(InteractionKnot.from_dict(payload))
        if len(knots) < 2:
            raise SchemaValidationError(
                f"C3 nominal phase trajectory is incomplete: {phase}"
            )
        value = ContactWrenchTrajectory(
            horizon_s=float(knots[-1].t_rel_s),
            dt_s=min(float(trajectory.dt_s) for _, trajectory in windows),
            knots=knots,
            derived_mode_label=("order9_c3_accepted_nominal_phase_replay:" f"{phase}"),
            contract_version=windows[0][1].contract_version,
        )
        value.validate()
        phase_trajectories[phase] = value
    if set(phase_trajectories) != set(_PHASES_TO_GRASP):
        raise SchemaValidationError(
            "C3 accepted nominal set must contain approach and contact phases"
        )
    return phase_trajectories


def _join_nominal_phase_trajectories(
    artifact: Order9C3NominalTrajectoryArtifact,
    phases: Mapping[str, ContactWrenchTrajectory],
) -> ContactWrenchTrajectory:
    knots: list[InteractionKnot] = []
    offset = 0.0
    dt_s = math.inf
    contract_version = None
    # The deployable active-knot encoder consumes one representative schedule
    # template, not the phase clock.  Preserve its accepted approach/contact
    # scale while the complete per-phase tensors are replayed separately.
    for phase in _PHASES_TO_GRASP:
        trajectory = phases[phase]
        contract_version = contract_version or trajectory.contract_version
        if trajectory.contract_version != contract_version:
            raise SchemaValidationError("C3 nominal phase contracts differ")
        dt_s = min(dt_s, float(trajectory.dt_s))
        for knot_index, knot in enumerate(trajectory.knots):
            if knots and knot_index == 0:
                continue
            payload = knot.to_dict()
            payload["t_rel_s"] = offset + float(knot.t_rel_s)
            knots.append(InteractionKnot.from_dict(payload))
        offset += float(trajectory.horizon_s)
    value = ContactWrenchTrajectory(
        horizon_s=float(offset),
        dt_s=float(dt_s),
        knots=knots,
        derived_mode_label="order9_c3_accepted_active_knot_replay_v2",
        contract_version=contract_version,
    )
    value.validate()
    return value


def _validate_component_collision_admission(
    artifact_path: Path,
    *,
    bucket_id: str,
) -> None:
    component_root = artifact_path.parent.parent.parent
    validation_path = component_root / "collision_validation.json"
    if not validation_path.is_file():
        raise SchemaValidationError(
            f"C3 nominal collision admission is missing: {bucket_id}"
        )
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    records = payload.get("records")
    matches = (
        [
            record
            for record in records
            if isinstance(record, dict) and record.get("bucket_id") == bucket_id
        ]
        if isinstance(records, list)
        else []
    )
    if (
        payload.get("all_accepted") is not True
        or len(matches) != 1
        or matches[0].get("accepted") is not True
    ):
        raise SchemaValidationError(
            f"C3 nominal collision admission failed: {bucket_id}"
        )


def _resolve_repository_path(value: str | Path, repository: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repository / path).resolve()


def _portable_repository_path(path: Path, repository: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository.resolve()))
    except ValueError:
        return str(path.resolve())


def _ideal_endpoint_observation(
    *,
    context: HighLevelPolicyContext,
    plan: Order9ArticulatedTeacherPlan,
    physical_model: PhysicalModel,
    next_phase: str,
) -> RuntimeObservation:
    previous = context.runtime_observation
    if previous is None:
        raise SchemaValidationError(
            "C3 nominal endpoint reconstruction requires runtime state"
        )
    endpoint = plan.trajectory.knots[-1]
    posture = endpoint.posture_target
    centroidal = endpoint.centroidal_target
    if (
        posture is None
        or posture.joint_pos_target is None
        or posture.joint_vel_target is None
        or centroidal is None
        or centroidal.com_pos_world is None
        or centroidal.body_orientation_world is None
    ):
        raise SchemaValidationError(
            "C3 nominal resolved endpoint lacks joint/centroidal targets"
        )
    q = {str(key): float(value) for key, value in posture.joint_pos_target.items()}
    qdot = {str(key): float(value) for key, value in posture.joint_vel_target.items()}
    kinematics = WholeStructureKinematics()
    base_pose = base_pose_for_centroidal_target(
        context.morphology_graph,
        physical_model,
        q,
        tuple(float(value) for value in centroidal.com_pos_world),
        tuple(float(value) for value in centroidal.body_orientation_world),
        kinematics=kinematics,
    )
    fk = kinematics.forward(
        context.morphology_graph,
        physical_model,
        q,
        base_pose,
        (),
    )
    linear_velocity = tuple(
        float(value) for value in (centroidal.com_vel_world or (0.0, 0.0, 0.0))
    )
    object_targets = {target.object_id: target for target in endpoint.object_targets}
    object_states = []
    for state in previous.object_states:
        target = object_targets.get(state.object_id)
        pose = (
            tuple(state.pose_world)
            if target is None or target.pose_target_world is None
            else tuple(target.pose_target_world)
        )
        object_states.append(
            ObjectRuntimeState(
                object_id=state.object_id,
                pose_world=pose,
                twist_world=[0.0] * 6,
            )
        )
    module_states = []
    for module in sorted(
        context.morphology_graph.modules,
        key=lambda value: value.module_id,
    ):
        prefix = f"module_{module.module_id}:"
        module_states.append(
            ModuleRuntimeState(
                module_id=module.module_id,
                pose_world=fk.module_root_poses_world[module.module_id],
                twist_world=[*linear_velocity, 0.0, 0.0, 0.0],
                joint_positions={
                    key[len(prefix) :]: value
                    for key, value in q.items()
                    if key.startswith(prefix)
                },
                joint_velocities={
                    key[len(prefix) :]: value
                    for key, value in qdot.items()
                    if key.startswith(prefix)
                },
            )
        )
    observation = RuntimeObservation(
        time_s=float(previous.time_s + plan.trajectory.horizon_s),
        morphology_graph=context.morphology_graph,
        module_states=module_states,
        object_states=object_states,
        contact_states=[],
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
        task_progress=TaskProgressState(
            phase_label=next_phase,
            progress_ratio=1.0 if plan.phase_target_reached else 0.0,
        ),
    )
    observation.validate()
    return observation


def _global_joint_positions(
    observation: RuntimeObservation,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for state in observation.module_states:
        for local_id, value in state.joint_positions.items():
            values[f"module_{state.module_id}:{local_id}"] = float(value)
    return values


def _flatten_nominal_windows(
    windows: Sequence[Order9C3NominalWindow],
) -> tuple[dict[str, Any], ...]:
    records = []
    for window in windows:
        for knot_index, knot in enumerate(window.plan.trajectory.knots):
            if window.window_index > 0 and knot_index == 0:
                continue
            records.append(
                {
                    "sample_index": len(records),
                    "global_time_s": (
                        float(window.global_start_time_s) + float(knot.t_rel_s)
                    ),
                    "window_index": window.window_index,
                    "window_local_time_s": float(knot.t_rel_s),
                    "phase": window.phase,
                    "phase_target_reached": (window.plan.phase_target_reached),
                    "knot": knot.to_dict(),
                }
            )
    if len(records) < 2:
        raise SchemaValidationError("C3 nominal flattened timeline is incomplete")
    return tuple(records)


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _write_new_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _json_payload(value: Mapping[str, Any]) -> str:
    import json

    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n"
    )


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise SchemaValidationError(f"{name} is not SHA-256")


__all__ = [
    "ORDER9_C3_NOMINAL_TRAJECTORY_SET_VERSION",
    "ORDER9_C3_NOMINAL_TRAJECTORY_VERSION",
    "Order9C3AcceptedNominalBundle",
    "Order9C3NominalTrajectory",
    "Order9C3NominalTrajectoryArtifact",
    "Order9C3NominalPhaseArtifact",
    "Order9C3NominalTrajectorySetEntry",
    "Order9C3NominalTrajectorySetManifest",
    "Order9C3NominalWindow",
    "Order9C3NominalWindowArtifact",
    "generate_order9_c3_nominal_grasp_trajectory",
    "load_order9_c3_accepted_nominal_bundle",
    "materialize_order9_c3_complete_task_phases",
    "validate_order9_c3_nominal_trajectory_artifact_bytes",
    "validate_order9_c3_nominal_trajectory_set_bytes",
    "write_order9_c3_nominal_trajectory_artifact",
]
