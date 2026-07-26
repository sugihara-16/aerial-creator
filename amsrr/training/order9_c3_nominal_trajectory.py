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
from typing import Any, Callable, Mapping, Sequence

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
    "order9_c3_nominal_configuration_space_v5_local_contact_corridor"
)
ORDER9_C3_NOMINAL_TRAJECTORY_SET_VERSION = (
    "order9_c3_nominal_trajectory_set_v5_local_contact_corridor"
)
_PHASES_TO_GRASP = ("approach", "contact_acquisition")


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
            raise SchemaValidationError(
                "nominal window index must be non-negative"
            )
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
            raise SchemaValidationError(
                "C3 nominal trajectory must contain windows"
            )
        if [value.window_index for value in self.windows] != list(
            range(len(self.windows))
        ):
            raise SchemaValidationError(
                "C3 nominal trajectory window indices are not contiguous"
            )
        if self.final_phase != "contact_acquisition":
            raise SchemaValidationError(
                "C3 nominal grasp trajectory must end in contact acquisition"
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
            raise SchemaValidationError(
                "C3 nominal set module count must be in [2, 8]"
            )
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
            raise SchemaValidationError(
                "C3 nominal trajectory set version mismatch"
            )
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
            raise SchemaValidationError(
                "C3 nominal trajectory set cannot be empty"
            )
        ids = [value.bucket_id for value in self.entries]
        if len(ids) != len(set(ids)):
            raise SchemaValidationError(
                "C3 nominal trajectory set bucket ids repeat"
            )


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
    preferred_surface_port_ids: tuple[int, int] | None = None,
    preferred_candidate_group_id: str | None = None,
    excluded_surface_port_id_pairs: tuple[tuple[int, int], ...] = (),
    contact_goal_joint_seed_positions_rad: (
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
            preferred_surface_port_ids=preferred_surface_port_ids,
            preferred_candidate_group_id=preferred_candidate_group_id,
            excluded_surface_port_id_pairs=excluded_surface_port_id_pairs,
            contact_goal_joint_seed_positions_rad=(
                contact_goal_joint_seed_positions_rad
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
        ),
        collision_object=(
            collision_object
            if enforce_proxy_collision_during_generation
            else None
        ),
        contact_goal_joint_seed_positions_rad=(
            contact_goal_joint_seed_positions_rad
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
                preferred_surface_port_ids=preferred_surface_port_ids,
                preferred_candidate_group_id=group_id,
                excluded_surface_port_id_pairs=(
                    excluded_surface_port_id_pairs
                ),
                contact_goal_joint_seed_positions_rad=(
                    contact_goal_joint_seed_positions_rad
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
    preferred_surface_port_ids: tuple[int, int] | None = None,
    preferred_candidate_group_id: str | None = None,
    excluded_surface_port_id_pairs: tuple[tuple[int, int], ...] = (),
    contact_goal_joint_seed_positions_rad: (
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
        ),
        collision_object=(
            collision_object
            if enforce_proxy_collision_during_generation
            else None
        ),
        contact_goal_joint_seed_positions_rad=(
            contact_goal_joint_seed_positions_rad
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
            preferred_candidate_group_id=(
                selection.trajectory_plan.candidate_group_id
            )
        ),
        collision_object=(
            collision_object
            if enforce_proxy_collision_during_generation
            else None
        ),
    )
    initial_object_poses = {
        value.object_id: value.pose_world
        for value in task_spec.scene.objects
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
            )
            if plan.task_phase != phase:
                raise SchemaValidationError(
                    "C3 nominal teacher returned an unexpected task phase"
                )
            if (
                plan.candidate_group_id
                != selection.trajectory_plan.candidate_group_id
            ):
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
                        "configuration_plan_method": (
                            configuration_plan.method
                        ),
                        "configuration_plan_state_count": len(
                            configuration_plan.states
                        ),
                        "configuration_plan_collision_check_count": (
                            configuration_plan.collision_check_count
                        ),
                        "start_base_pose_world": list(
                            plan_start.base_pose_world
                        ),
                        "next_base_pose_world": list(
                            plan_next.base_pose_world
                        ),
                        "goal_base_pose_world": list(
                            plan_goal.base_pose_world
                        ),
                        "start_joint_hash": stable_hash(
                            plan_start.joint_positions_rad
                        ),
                        "next_joint_hash": stable_hash(
                            plan_next.joint_positions_rad
                        ),
                        "goal_joint_hash": stable_hash(
                            plan_goal.joint_positions_rad
                        ),
                        "maximum_start_next_joint_delta_rad": max(
                            (
                                abs(
                                    float(
                                        plan_next.joint_positions_rad[
                                            joint_id
                                        ]
                                    )
                                    - float(
                                        plan_start.joint_positions_rad[
                                            joint_id
                                        ]
                                    )
                                )
                                for joint_id in ordered_ids
                            ),
                            default=0.0,
                        ),
                        "maximum_start_goal_joint_delta_rad": max(
                            (
                                abs(
                                    float(
                                        plan_goal.joint_positions_rad[
                                            joint_id
                                        ]
                                    )
                                    - float(
                                        plan_start.joint_positions_rad[
                                            joint_id
                                        ]
                                    )
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


def write_order9_c3_nominal_trajectory_artifact(
    result: Order9C3NominalTrajectory,
    *,
    output_dir: str | Path,
    bucket_id: str,
    split: DatasetSplit,
    task_spec_sha256: str,
    structural_hash: str,
    physical_model_hash: str,
    robot_urdf_path: str | Path,
) -> Order9C3NominalTrajectoryArtifact:
    """Persist raw/resolved windows and one flattened inspection timeline."""

    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(
            f"C3 nominal trajectory output exists: {destination}"
        )
    destination.mkdir(parents=True)
    urdf_path = Path(robot_urdf_path).resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(urdf_path)
    morphology_path = destination / "task_conditioned_morphology.json"
    candidates_path = destination / "contact_candidate_set.json"
    timeline_path = destination / "nominal_timeline.json"
    _write_new_text(
        morphology_path,
        result.selection_bundle.design_output.target_morphology.to_json(
            indent=2
        )
        + "\n",
    )
    _write_new_text(
        candidates_path,
        result.selection_bundle.contact_candidate_set.to_json(indent=2)
        + "\n",
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
                "duration_s": result.duration_s,
                "records": list(result.timeline),
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
                raw_trajectory_hash=stable_hash(
                    window.plan.raw_trajectory.to_dict()
                ),
                resolved_trajectory_path=_relative(
                    resolved_path, destination
                ),
                resolved_trajectory_sha256=hash_file(resolved_path),
                resolved_trajectory_hash=stable_hash(
                    window.plan.trajectory.to_dict()
                ),
                resolver_evidence_path=_relative(
                    evidence_path, destination
                ),
                resolver_evidence_sha256=hash_file(evidence_path),
                raw_knot_count=len(window.plan.raw_trajectory.knots),
                resolved_knot_count=len(window.plan.trajectory.knots),
                configuration_planner_version=(
                    configuration_plan.planner_version
                ),
                configuration_plan_method=configuration_plan.method,
                configuration_plan_state_count=len(
                    configuration_plan.states
                ),
                configuration_plan_collision_check_count=(
                    configuration_plan.collision_check_count
                ),
                configuration_plan_sampled_state_count=(
                    configuration_plan.sampled_state_count
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
        task_conditioned_morphology_path=_relative(
            morphology_path, destination
        ),
        task_conditioned_morphology_sha256=hash_file(morphology_path),
        task_conditioned_morphology_hash=(
            result.selection_bundle.design_output.target_morphology.stable_hash()
        ),
        contact_candidate_set_path=_relative(
            candidates_path, destination
        ),
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
        duration_s=result.duration_s,
        final_phase=result.windows[-1].phase,
        final_phase_target_reached=(
            result.windows[-1].plan.phase_target_reached
        ),
        proxy_collision_validation_status=(
            result.proxy_collision_validation_status
        ),
        windows=window_artifacts,
        selection_evidence=order9_c3_teacher_evidence(
            result.selection_bundle
        ),
    )
    manifest.validate()
    _write_new_text(
        destination / "manifest.json",
        manifest.to_json(indent=2) + "\n",
    )
    validate_order9_c3_nominal_trajectory_artifact_bytes(
        destination / "manifest.json"
    )
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
    for path, expected in checks:
        if not path.is_file() or hash_file(path) != expected:
            raise SchemaValidationError(
                f"C3 nominal trajectory bytes changed: {path}"
            )
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
        artifact = validate_order9_c3_nominal_trajectory_artifact_bytes(
            artifact_path
        )
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
    morphology = MorphologyGraph.from_json(
        morphology_path.read_text(encoding="utf-8")
    )
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
            "selected_surface_port_ids": list(
                artifact.selected_surface_port_ids
            ),
        },
    )


def _load_nominal_phase_trajectories(
    root: Path,
    artifact: Order9C3NominalTrajectoryArtifact,
) -> dict[str, ContactWrenchTrajectory]:
    by_phase: dict[str, list[tuple[float, ContactWrenchTrajectory]]] = {}
    phase_start: dict[str, float] = {}
    for window in artifact.windows:
        path = root / window.resolved_trajectory_path
        trajectory = ContactWrenchTrajectory.from_json(
            path.read_text(encoding="utf-8")
        )
        trajectory.validate()
        if stable_hash(trajectory.to_dict()) != window.resolved_trajectory_hash:
            raise SchemaValidationError("C3 nominal resolved trajectory changed")
        by_phase.setdefault(window.phase, []).append(
            (float(window.global_start_time_s), trajectory)
        )
        phase_start.setdefault(window.phase, float(window.global_start_time_s))
    phase_trajectories: dict[str, ContactWrenchTrajectory] = {}
    for phase, windows in by_phase.items():
        origin = phase_start[phase]
        knots: list[InteractionKnot] = []
        for global_start, trajectory in windows:
            for knot_index, knot in enumerate(trajectory.knots):
                time_s = global_start - origin + float(knot.t_rel_s)
                if knots and knot_index == 0 and abs(
                    float(knots[-1].t_rel_s) - time_s
                ) <= 1.0e-9:
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
            dt_s=min(
                float(trajectory.dt_s) for _, trajectory in windows
            ),
            knots=knots,
            derived_mode_label=(
                "order9_c3_accepted_nominal_phase_replay:"
                f"{phase}"
            ),
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
        horizon_s=float(artifact.duration_s),
        dt_s=float(dt_s),
        knots=knots,
        derived_mode_label="order9_c3_accepted_nominal_replay_v1",
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
    q = {
        str(key): float(value)
        for key, value in posture.joint_pos_target.items()
    }
    qdot = {
        str(key): float(value)
        for key, value in posture.joint_vel_target.items()
    }
    kinematics = WholeStructureKinematics()
    base_pose = base_pose_for_centroidal_target(
        context.morphology_graph,
        physical_model,
        q,
        tuple(float(value) for value in centroidal.com_pos_world),
        tuple(
            float(value)
            for value in centroidal.body_orientation_world
        ),
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
        float(value)
        for value in (centroidal.com_vel_world or (0.0, 0.0, 0.0))
    )
    object_targets = {
        target.object_id: target
        for target in endpoint.object_targets
    }
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
                        float(window.global_start_time_s)
                        + float(knot.t_rel_s)
                    ),
                    "window_index": window.window_index,
                    "window_local_time_s": float(knot.t_rel_s),
                    "phase": window.phase,
                    "phase_target_reached": (
                        window.plan.phase_target_reached
                    ),
                    "knot": knot.to_dict(),
                }
            )
    if len(records) < 2:
        raise SchemaValidationError(
            "C3 nominal flattened timeline is incomplete"
        )
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

    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
    ) + "\n"


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
    "Order9C3NominalTrajectorySetEntry",
    "Order9C3NominalTrajectorySetManifest",
    "Order9C3NominalWindow",
    "Order9C3NominalWindowArtifact",
    "generate_order9_c3_nominal_grasp_trajectory",
    "load_order9_c3_accepted_nominal_bundle",
    "validate_order9_c3_nominal_trajectory_artifact_bytes",
    "validate_order9_c3_nominal_trajectory_set_bytes",
    "write_order9_c3_nominal_trajectory_artifact",
]
