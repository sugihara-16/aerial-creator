from __future__ import annotations

"""Joint-aware trajectory IK and hard kinematic reachability evidence.

The solver in this module is a deterministic teacher utility.  The evaluator
uses the same geometry but only checks a trajectory exactly as proposed; it
does not search, repair, rank, or project policy output.
"""

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from amsrr.controllers.rigid_body_model import RigidBodyControlModelBuilder
from amsrr.feasibility.contact_wrench_trajectory import (
    KnotReachabilityEvaluation,
)
from amsrr.geometry.pose_math import (
    Matrix3,
    compose_pose,
    inverse_pose,
    matmul,
    matvec,
    pose_from_transform,
    quat_from_matrix,
    transform_from_pose,
    transpose,
)
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.robot_model.gripper_surfaces import (
    GripperSurface,
    resolve_unoccupied_gripper_surfaces,
)
from amsrr.robot_model.whole_structure_kinematics import (
    MeshBackedAnchorReference,
    WholeStructureKinematics,
    ordered_global_dock_joint_ids,
)
from amsrr.schemas.common import Pose7D, SchemaValidationError
from amsrr.schemas.contact_candidates import ContactCandidate
from amsrr.schemas.morphology import MorphologyGraph, RobotAnchor
from amsrr.schemas.physical_model import JointModel, PhysicalModel
from amsrr.schemas.policies import (
    ContactAssignment,
    ContactWrenchTrajectory,
    ControllerStatus,
)
from amsrr.schemas.runtime import (
    ModuleRuntimeState,
    RuntimeObservation,
    TaskProgressState,
)


ARTICULATED_REACHABILITY_VERSION = "articulated_reachability_v1"
ARTICULATED_IK_TEACHER_VERSION = "articulated_contact_ik_teacher_v1"

REACHABILITY_POSTURE_MISSING_CODE = "E_REACHABILITY_POSTURE_MISSING"
REACHABILITY_JOINT_SET_CODE = "E_REACHABILITY_JOINT_SET"
REACHABILITY_JOINT_LIMIT_CODE = "E_REACHABILITY_JOINT_LIMIT"
REACHABILITY_JOINT_RATE_CODE = "E_REACHABILITY_JOINT_RATE"
REACHABILITY_CENTROIDAL_TARGET_CODE = "E_REACHABILITY_CENTROIDAL_TARGET"
REACHABILITY_ANCHOR_TARGET_CODE = "E_REACHABILITY_ANCHOR_TARGET"
REACHABILITY_CONTACT_POSITION_CODE = "E_REACHABILITY_CONTACT_POSITION"
REACHABILITY_CONTACT_NORMAL_CODE = "E_REACHABILITY_CONTACT_NORMAL"
REACHABILITY_GEOMETRY_CODE = "E_REACHABILITY_GEOMETRY"

_ACTIVE_CONTACT_STATES = frozenset({"attach", "maintain", "slide"})
_IDENTITY_POSE: Pose7D = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class ArticulatedReachabilityConfig:
    contact_position_tolerance_m: float = 0.010
    contact_normal_tolerance_rad: float = 0.20
    anchor_target_position_tolerance_m: float = 0.010
    anchor_target_attitude_tolerance_rad: float = 0.20
    joint_limit_tolerance_rad: float = 1.0e-6
    joint_rate_tolerance_radps: float = 1.0e-5

    def __post_init__(self) -> None:
        for name in (
            "contact_position_tolerance_m",
            "contact_normal_tolerance_rad",
            "anchor_target_position_tolerance_m",
            "anchor_target_attitude_tolerance_rad",
            "joint_limit_tolerance_rad",
            "joint_rate_tolerance_radps",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class ArticulatedIKConfig:
    maximum_iterations: int = 60
    damping: float = 2.0e-3
    position_weight: float = 1.0
    normal_weight: float = 0.35
    joint_regularization_weight: float = 1.0e-6
    maximum_joint_step_rad: float = 0.20
    maximum_base_translation_step_m: float = 0.20
    maximum_base_rotation_step_rad: float = 0.20
    contact_position_tolerance_m: float = 0.005
    contact_normal_tolerance_rad: float = 0.10

    def __post_init__(self) -> None:
        if self.maximum_iterations < 1:
            raise ValueError("maximum_iterations must be positive")
        for name in (
            "damping",
            "position_weight",
            "normal_weight",
            "joint_regularization_weight",
            "maximum_joint_step_rad",
            "maximum_base_translation_step_m",
            "maximum_base_rotation_step_rad",
            "contact_position_tolerance_m",
            "contact_normal_tolerance_rad",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class ArticulatedIKSolution:
    feasible: bool
    joint_positions_rad: dict[str, float]
    base_pose_world: Pose7D
    centroidal_pose_world: Pose7D
    anchor_poses_world: dict[int, Pose7D]
    maximum_position_error_m: float
    maximum_normal_error_rad: float
    iterations: int
    solver_version: str = ARTICULATED_IK_TEACHER_VERSION


def resolve_mesh_backed_anchor_references(
    morphology: MorphologyGraph,
    physical_model: PhysicalModel,
    anchor_ids: Sequence[int],
) -> tuple[MeshBackedAnchorReference, ...]:
    """Resolve graph anchors to their authoritative free Dock surfaces."""

    requested = tuple(sorted(set(int(value) for value in anchor_ids)))
    anchors = {anchor.anchor_id: anchor for anchor in morphology.robot_anchors}
    surfaces = {
        surface.port_global_id: surface
        for surface in resolve_unoccupied_gripper_surfaces(
            morphology, physical_model
        )
    }
    references: list[MeshBackedAnchorReference] = []
    for anchor_id in requested:
        anchor = anchors.get(anchor_id)
        if anchor is None:
            raise SchemaValidationError(
                f"reachability references unknown RobotAnchor {anchor_id}"
            )
        port_id = _anchor_surface_port_id(anchor)
        surface = surfaces.get(port_id)
        if surface is None:
            raise SchemaValidationError(
                f"RobotAnchor {anchor_id} does not resolve to a free mesh-backed surface"
            )
        references.append(
            MeshBackedAnchorReference(anchor=anchor, surface=surface)
        )
    return tuple(references)


class ArticulatedContactIKSolver:
    """Deterministic damped-least-squares teacher over Dock joints and free base."""

    def __init__(
        self,
        physical_model: PhysicalModel,
        *,
        config: ArticulatedIKConfig | None = None,
        kinematics: WholeStructureKinematics | None = None,
    ) -> None:
        self.physical_model = physical_model
        self.config = config or ArticulatedIKConfig()
        self.kinematics = kinematics or WholeStructureKinematics()
        self._rigid_body_builder = RigidBodyControlModelBuilder()

    def solve(
        self,
        *,
        morphology: MorphologyGraph,
        assignments: Sequence[ContactAssignment],
        candidates: Mapping[int, ContactCandidate],
        initial_joint_positions_rad: Mapping[str, float] | None = None,
        initial_base_pose_world: Pose7D | None = None,
    ) -> ArticulatedIKSolution:
        np = _numpy()
        active = tuple(
            assignment
            for assignment in assignments
            if assignment.schedule_state in _ACTIVE_CONTACT_STATES
        )
        if not active:
            raise SchemaValidationError(
                "articulated contact IK requires at least one active assignment"
            )
        if len({value.anchor_id for value in active}) != len(active):
            raise SchemaValidationError(
                "articulated contact IK requires unique selected anchors"
            )
        selected_candidates = []
        for assignment in active:
            candidate = candidates.get(assignment.candidate_id)
            if candidate is None:
                raise SchemaValidationError(
                    "articulated contact IK references an unknown candidate"
                )
            if candidate.anchor_id != assignment.anchor_id:
                raise SchemaValidationError(
                    "articulated contact IK assignment/candidate anchor mismatch"
                )
            selected_candidates.append(candidate)
        references = resolve_mesh_backed_anchor_references(
            morphology,
            self.physical_model,
            [value.anchor_id for value in active],
        )
        reference_by_id = {
            reference.anchor.anchor_id: reference for reference in references
        }
        ordered_ids = ordered_global_dock_joint_ids(
            morphology, self.physical_model
        )
        limits = _global_joint_limits(
            morphology, self.physical_model, ordered_ids
        )
        q = {
            joint_id: float(
                0.0
                if initial_joint_positions_rad is None
                else initial_joint_positions_rad.get(joint_id, 0.0)
            )
            for joint_id in ordered_ids
        }
        q = _clip_joint_map(q, limits)
        base_pose = (
            initial_base_pose_world
            if initial_base_pose_world is not None
            else _initial_base_pose(selected_candidates)
        )
        maximum_position_error = math.inf
        maximum_normal_error = math.inf
        anchor_poses: dict[int, Pose7D] = {}

        for iteration in range(1, self.config.maximum_iterations + 1):
            result = self.kinematics.compute(
                morphology,
                self.physical_model,
                q,
                base_pose,
                references,
            )
            anchor_poses = dict(result.anchor_poses_world)
            rows: list[list[float]] = []
            residuals: list[float] = []
            position_errors: list[float] = []
            normal_errors: list[float] = []
            joint_count = len(ordered_ids)
            for assignment, candidate in zip(
                active, selected_candidates, strict=True
            ):
                reference = reference_by_id[assignment.anchor_id]
                actual = result.anchor_poses_world[assignment.anchor_id]
                target_position = tuple(
                    float(value) for value in candidate.contact_pose_world[:3]
                )
                position_error = tuple(
                    target_position[index] - float(actual[index])
                    for index in range(3)
                )
                actual_outward = _pose_x_axis(actual)
                desired_outward = _unit(
                    tuple(-float(value) for value in candidate.normal_world)
                )
                normal_error = _axis_alignment_rotation(
                    actual_outward, desired_outward
                )
                position_errors.append(_norm(position_error))
                normal_errors.append(_norm(normal_error))
                joint_jacobian = result.anchor_jacobians[
                    reference.anchor.anchor_id
                ]
                base_position_jacobian = _base_position_jacobian(
                    actual[:3], base_pose[:3]
                )
                for axis in range(3):
                    rows.append(
                        [
                            *(
                                self.config.position_weight
                                * float(joint_jacobian[axis][column])
                                for column in range(joint_count)
                            ),
                            *(
                                self.config.position_weight
                                * float(value)
                                for value in base_position_jacobian[axis]
                            ),
                            *(
                                self.config.position_weight
                                * (1.0 if axis == column else 0.0)
                                for column in range(3)
                            ),
                        ]
                    )
                    residuals.append(
                        self.config.position_weight * position_error[axis]
                    )
                for axis in range(3):
                    rows.append(
                        [
                            *(
                                self.config.normal_weight
                                * float(joint_jacobian[axis + 3][column])
                                for column in range(joint_count)
                            ),
                            *(
                                self.config.normal_weight
                                * (1.0 if axis == column else 0.0)
                                for column in range(3)
                            ),
                            0.0,
                            0.0,
                            0.0,
                        ]
                    )
                    residuals.append(
                        self.config.normal_weight * normal_error[axis]
                    )
            maximum_position_error = max(position_errors, default=0.0)
            maximum_normal_error = max(normal_errors, default=0.0)
            if (
                maximum_position_error
                <= self.config.contact_position_tolerance_m
                and maximum_normal_error
                <= self.config.contact_normal_tolerance_rad
            ):
                centroidal = _centroidal_pose(
                    morphology,
                    self.physical_model,
                    q,
                    result.module_root_poses_world,
                    self._rigid_body_builder,
                )
                return ArticulatedIKSolution(
                    feasible=True,
                    joint_positions_rad=dict(q),
                    base_pose_world=base_pose,
                    centroidal_pose_world=centroidal,
                    anchor_poses_world=anchor_poses,
                    maximum_position_error_m=maximum_position_error,
                    maximum_normal_error_rad=maximum_normal_error,
                    iterations=iteration,
                )

            matrix = np.asarray(rows, dtype=float)
            residual = np.asarray(residuals, dtype=float)
            variable_count = matrix.shape[1]
            normal = matrix.T @ matrix
            normal += (self.config.damping**2) * np.eye(variable_count)
            right = matrix.T @ residual
            for index, joint_id in enumerate(ordered_ids):
                normal[index, index] += self.config.joint_regularization_weight
                right[index] -= (
                    self.config.joint_regularization_weight * q[joint_id]
                )
            try:
                delta = np.linalg.solve(normal, right)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(normal, right, rcond=None)[0]
            joint_delta = np.clip(
                delta[: len(ordered_ids)],
                -self.config.maximum_joint_step_rad,
                self.config.maximum_joint_step_rad,
            )
            rotation_delta = _bounded_vector(
                delta[len(ordered_ids) : len(ordered_ids) + 3],
                self.config.maximum_base_rotation_step_rad,
            )
            translation_delta = _bounded_vector(
                delta[len(ordered_ids) + 3 :],
                self.config.maximum_base_translation_step_m,
            )
            q = _clip_joint_map(
                {
                    joint_id: q[joint_id] + float(joint_delta[index])
                    for index, joint_id in enumerate(ordered_ids)
                },
                limits,
            )
            base_pose = _apply_base_delta(
                base_pose,
                rotation_delta,
                translation_delta,
            )

        final = self.kinematics.forward(
            morphology,
            self.physical_model,
            q,
            base_pose,
            references,
        )
        centroidal = _centroidal_pose(
            morphology,
            self.physical_model,
            q,
            final.module_root_poses_world,
            self._rigid_body_builder,
        )
        return ArticulatedIKSolution(
            feasible=False,
            joint_positions_rad=dict(q),
            base_pose_world=base_pose,
            centroidal_pose_world=centroidal,
            anchor_poses_world=dict(final.anchor_poses_world),
            maximum_position_error_m=maximum_position_error,
            maximum_normal_error_rad=maximum_normal_error,
            iterations=self.config.maximum_iterations,
        )


class ArticulatedTrajectoryReachabilityEvaluator:
    """Hard evaluator for complete joint/centroidal/contact consistency."""

    evaluator_version = ARTICULATED_REACHABILITY_VERSION

    def __init__(
        self,
        physical_model: PhysicalModel,
        *,
        config: ArticulatedReachabilityConfig | None = None,
        kinematics: WholeStructureKinematics | None = None,
    ) -> None:
        self.physical_model = physical_model
        self.config = config or ArticulatedReachabilityConfig()
        self.kinematics = kinematics or WholeStructureKinematics()

    def evaluate_trajectory(
        self,
        *,
        context: HighLevelPolicyContext,
        trajectory: ContactWrenchTrajectory,
    ) -> tuple[KnotReachabilityEvaluation, ...]:
        morphology = context.morphology_graph
        ordered_ids = ordered_global_dock_joint_ids(
            morphology, self.physical_model
        )
        limits = _global_joint_limits(
            morphology, self.physical_model, ordered_ids
        )
        velocity_limit = _dock_velocity_limit(self.physical_model)
        candidate_by_id = {
            candidate.candidate_id: candidate
            for candidate in context.contact_candidate_set.candidates
        }
        previous_q = _runtime_global_q(context, ordered_ids)
        previous_time = 0.0 if previous_q is not None else None
        results: list[KnotReachabilityEvaluation] = []
        for knot in trajectory.knots:
            codes: list[str] = []
            margins: dict[str, float] = {}
            posture = knot.posture_target
            q = None if posture is None else posture.joint_pos_target
            qdot = None if posture is None else posture.joint_vel_target
            if q is None or qdot is None:
                _append_code(codes, REACHABILITY_POSTURE_MISSING_CODE)
            elif set(q) != set(ordered_ids) or set(qdot) != set(ordered_ids):
                _append_code(codes, REACHABILITY_JOINT_SET_CODE)
            else:
                parsed_q = {key: float(q[key]) for key in ordered_ids}
                parsed_qdot = {key: float(qdot[key]) for key in ordered_ids}
                if not all(
                    math.isfinite(value)
                    for value in (*parsed_q.values(), *parsed_qdot.values())
                ):
                    _append_code(codes, REACHABILITY_JOINT_SET_CODE)
                else:
                    limit_margin = min(
                        min(
                            parsed_q[joint_id] - limits[joint_id][0],
                            limits[joint_id][1] - parsed_q[joint_id],
                        )
                        for joint_id in ordered_ids
                    )
                    margins["joint_limit_margin_rad"] = limit_margin
                    if limit_margin < -self.config.joint_limit_tolerance_rad:
                        _append_code(codes, REACHABILITY_JOINT_LIMIT_CODE)
                    velocity_margin = velocity_limit - max(
                        abs(value) for value in parsed_qdot.values()
                    )
                    margins["commanded_joint_rate_margin_radps"] = (
                        velocity_margin
                    )
                    if (
                        velocity_margin
                        < -self.config.joint_rate_tolerance_radps
                    ):
                        _append_code(codes, REACHABILITY_JOINT_RATE_CODE)
                    if previous_q is not None and previous_time is not None:
                        elapsed = float(knot.t_rel_s) - previous_time
                        if elapsed > 1.0e-12:
                            required_rate = max(
                                abs(
                                    parsed_q[joint_id]
                                    - previous_q[joint_id]
                                )
                                / elapsed
                                for joint_id in ordered_ids
                            )
                            margins["transition_joint_rate_margin_radps"] = (
                                velocity_limit - required_rate
                            )
                            if (
                                required_rate
                                > velocity_limit
                                + self.config.joint_rate_tolerance_radps
                            ):
                                _append_code(
                                    codes, REACHABILITY_JOINT_RATE_CODE
                                )
                    self._evaluate_geometry(
                        context=context,
                        knot=knot,
                        q=parsed_q,
                        candidates=candidate_by_id,
                        codes=codes,
                        margins=margins,
                    )
                    previous_q = parsed_q
                    previous_time = float(knot.t_rel_s)
            results.append(
                KnotReachabilityEvaluation(
                    feasible=not codes,
                    violation_codes=tuple(codes),
                    margins=margins,
                    evaluator_version=self.evaluator_version,
                )
            )
        return tuple(results)

    def _evaluate_geometry(
        self,
        *,
        context: HighLevelPolicyContext,
        knot: object,
        q: Mapping[str, float],
        candidates: Mapping[int, ContactCandidate],
        codes: list[str],
        margins: dict[str, float],
    ) -> None:
        assignments = tuple(getattr(knot, "contact_assignments"))
        active = tuple(
            value
            for value in assignments
            if value.schedule_state in _ACTIVE_CONTACT_STATES
        )
        centroidal = getattr(knot, "centroidal_target")
        posture = getattr(knot, "posture_target")
        if (
            centroidal is None
            or centroidal.com_pos_world is None
            or centroidal.body_orientation_world is None
        ):
            _append_code(codes, REACHABILITY_CENTROIDAL_TARGET_CODE)
            return
        try:
            base_pose = base_pose_for_centroidal_target(
                context.morphology_graph,
                self.physical_model,
                q,
                tuple(float(value) for value in centroidal.com_pos_world),
                tuple(
                    float(value)
                    for value in centroidal.body_orientation_world
                ),
                kinematics=self.kinematics,
            )
            references = resolve_mesh_backed_anchor_references(
                context.morphology_graph,
                self.physical_model,
                [value.anchor_id for value in active],
            )
            fk = self.kinematics.forward(
                context.morphology_graph,
                self.physical_model,
                q,
                base_pose,
                references,
            )
        except (SchemaValidationError, TypeError, ValueError, KeyError):
            _append_code(codes, REACHABILITY_GEOMETRY_CODE)
            return
        target_poses = (
            {}
            if posture is None or posture.free_anchor_pose_targets is None
            else posture.free_anchor_pose_targets
        )
        active_anchor_ids = {value.anchor_id for value in active}
        if set(target_poses) != active_anchor_ids:
            _append_code(codes, REACHABILITY_ANCHOR_TARGET_CODE)
        minimum_position_margin = math.inf
        minimum_normal_margin = math.inf
        minimum_anchor_position_margin = math.inf
        minimum_anchor_attitude_margin = math.inf
        for assignment in active:
            candidate = candidates.get(assignment.candidate_id)
            actual = fk.anchor_poses_world.get(assignment.anchor_id)
            if candidate is None or actual is None:
                _append_code(codes, REACHABILITY_GEOMETRY_CODE)
                continue
            expected_pose, expected_normal = _candidate_target_for_knot(
                context,
                knot,
                candidate,
            )
            position_error = _norm(
                tuple(
                    float(expected_pose[index])
                    - float(actual[index])
                    for index in range(3)
                )
            )
            desired_outward = _unit(
                tuple(-float(value) for value in expected_normal)
            )
            normal_error = _angle(
                _pose_x_axis(actual), desired_outward
            )
            minimum_position_margin = min(
                minimum_position_margin,
                self.config.contact_position_tolerance_m - position_error,
            )
            minimum_normal_margin = min(
                minimum_normal_margin,
                self.config.contact_normal_tolerance_rad - normal_error,
            )
            if position_error > self.config.contact_position_tolerance_m:
                _append_code(codes, REACHABILITY_CONTACT_POSITION_CODE)
            if normal_error > self.config.contact_normal_tolerance_rad:
                _append_code(codes, REACHABILITY_CONTACT_NORMAL_CODE)
            target = target_poses.get(assignment.anchor_id)
            if target is None:
                continue
            target_position_error = _norm(
                tuple(float(target[index]) - float(actual[index]) for index in range(3))
            )
            target_attitude_error = _pose_attitude_error(target, actual)
            minimum_anchor_position_margin = min(
                minimum_anchor_position_margin,
                self.config.anchor_target_position_tolerance_m
                - target_position_error,
            )
            minimum_anchor_attitude_margin = min(
                minimum_anchor_attitude_margin,
                self.config.anchor_target_attitude_tolerance_rad
                - target_attitude_error,
            )
            if (
                target_position_error
                > self.config.anchor_target_position_tolerance_m
                or target_attitude_error
                > self.config.anchor_target_attitude_tolerance_rad
            ):
                _append_code(codes, REACHABILITY_ANCHOR_TARGET_CODE)
        for name, value in (
            ("contact_position_margin_m", minimum_position_margin),
            ("contact_normal_margin_rad", minimum_normal_margin),
            ("anchor_target_position_margin_m", minimum_anchor_position_margin),
            ("anchor_target_attitude_margin_rad", minimum_anchor_attitude_margin),
        ):
            if math.isfinite(value):
                margins[name] = value


def base_pose_for_centroidal_target(
    morphology: MorphologyGraph,
    physical_model: PhysicalModel,
    joint_positions_rad: Mapping[str, float],
    com_pos_world: tuple[float, float, float],
    body_orientation_world: tuple[float, float, float, float],
    *,
    kinematics: WholeStructureKinematics | None = None,
) -> Pose7D:
    """Recover the base-root pose whose assembled CoM equals the target."""

    solver = kinematics or WholeStructureKinematics()
    orientation = _normalized_quaternion(body_orientation_world)
    zero_translation: Pose7D = (
        0.0,
        0.0,
        0.0,
        *orientation,
    )
    fk = solver.forward(
        morphology,
        physical_model,
        joint_positions_rad,
        zero_translation,
        (),
    )
    actual = _centroidal_pose(
        morphology,
        physical_model,
        joint_positions_rad,
        fk.module_root_poses_world,
        RigidBodyControlModelBuilder(),
    )
    return (
        float(com_pos_world[0]) - actual[0],
        float(com_pos_world[1]) - actual[1],
        float(com_pos_world[2]) - actual[2],
        *orientation,
    )


def _anchor_surface_port_id(anchor: RobotAnchor) -> int:
    values = [
        anchor.capability.get("dock_port_global_id"),
        anchor.capability.get("surface_port_id"),
    ]
    resolved = [
        int(value)
        for value in values
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
    ]
    if not resolved or len(set(resolved)) != 1:
        raise SchemaValidationError(
            f"RobotAnchor {anchor.anchor_id} lacks one unambiguous surface port id"
        )
    return resolved[0]


def _global_joint_limits(
    morphology: MorphologyGraph,
    physical_model: PhysicalModel,
    ordered_ids: Sequence[str],
) -> dict[str, tuple[float, float]]:
    joints = {joint.joint_id: joint for joint in physical_model.joints}
    values: dict[str, tuple[float, float]] = {}
    module_ids = {module.module_id for module in morphology.modules}
    for global_id in ordered_ids:
        module_label, local_id = global_id.split(":", 1)
        module_id = int(module_label.removeprefix("module_"))
        joint = joints.get(local_id)
        if module_id not in module_ids or joint is None:
            raise SchemaValidationError(f"unknown Dock joint {global_id!r}")
        if joint.limit_lower is None or joint.limit_upper is None:
            raise SchemaValidationError(
                f"Dock joint {global_id!r} requires finite limits"
            )
        values[global_id] = (
            float(joint.limit_lower),
            float(joint.limit_upper),
        )
    return values


def _clip_joint_map(
    q: Mapping[str, float],
    limits: Mapping[str, tuple[float, float]],
) -> dict[str, float]:
    return {
        joint_id: min(
            max(float(q[joint_id]), limits[joint_id][0]),
            limits[joint_id][1],
        )
        for joint_id in limits
    }


def _dock_velocity_limit(physical_model: PhysicalModel) -> float:
    candidates = [
        float(joint.velocity_limit)
        for joint in physical_model.joints
        if _is_dock_joint(joint, physical_model)
        and joint.velocity_limit is not None
        and math.isfinite(float(joint.velocity_limit))
        and float(joint.velocity_limit) > 0.0
    ]
    specs = physical_model.metadata.get("joint_actuator_specs")
    if isinstance(specs, dict):
        dock = specs.get("dock")
        drive = dock.get("simulation_drive") if isinstance(dock, dict) else None
        value = (
            drive.get("safe_velocity_limit_rad_s")
            if isinstance(drive, dict)
            else None
        )
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0.0
        ):
            candidates.append(float(value))
    if not candidates:
        raise SchemaValidationError(
            "PhysicalModel lacks a positive Dock velocity limit"
        )
    return min(candidates)


def _is_dock_joint(joint: JointModel, physical_model: PhysicalModel) -> bool:
    return joint.joint_id in {
        str(port.mechanical_limits.get("mechanism_joint_id"))
        for port in physical_model.dock_ports
    }


def _runtime_global_q(
    context: HighLevelPolicyContext,
    ordered_ids: Sequence[str],
) -> dict[str, float] | None:
    observation = context.runtime_observation
    if observation is None:
        return None
    module_states = {state.module_id: state for state in observation.module_states}
    result: dict[str, float] = {}
    for joint_id in ordered_ids:
        module_label, local_id = joint_id.split(":", 1)
        module_id = int(module_label.removeprefix("module_"))
        state = module_states.get(module_id)
        if state is None or local_id not in state.joint_positions:
            return None
        result[joint_id] = float(state.joint_positions[local_id])
    return result


def _candidate_target_for_knot(
    context: HighLevelPolicyContext,
    knot: object,
    candidate: ContactCandidate,
) -> tuple[Pose7D, tuple[float, float, float]]:
    """Move an object-local candidate with the knot's object pose target."""

    observation = context.runtime_observation
    if observation is None:
        return candidate.contact_pose_world, tuple(candidate.normal_world)
    current = next(
        (
            state.pose_world
            for state in observation.object_states
            if state.object_id == candidate.target_entity_id
        ),
        None,
    )
    target = next(
        (
            value.pose_target_world
            for value in getattr(knot, "object_targets")
            if value.object_id == candidate.target_entity_id
            and value.pose_target_world is not None
        ),
        None,
    )
    if current is None or target is None:
        return candidate.contact_pose_world, tuple(candidate.normal_world)
    delta = compose_pose(target, inverse_pose(current))
    expected_pose = compose_pose(delta, candidate.contact_pose_world)
    rotation = transform_from_pose(delta).rotation
    expected_normal = matvec(rotation, candidate.normal_world)
    return expected_pose, expected_normal


def _centroidal_pose(
    morphology: MorphologyGraph,
    physical_model: PhysicalModel,
    q: Mapping[str, float],
    module_root_poses: Mapping[int, Pose7D],
    builder: RigidBodyControlModelBuilder,
) -> Pose7D:
    local_positions: dict[int, dict[str, float]] = {
        module.module_id: {} for module in morphology.modules
    }
    for global_id, value in q.items():
        module_label, local_id = global_id.split(":", 1)
        module_id = int(module_label.removeprefix("module_"))
        local_positions[module_id][local_id] = float(value)
    observation = RuntimeObservation(
        time_s=0.0,
        morphology_graph=morphology,
        module_states=[
            ModuleRuntimeState(
                module_id=module_id,
                pose_world=module_root_poses[module_id],
                twist_world=[0.0] * 6,
                joint_positions=local_positions[module_id],
                joint_velocities={
                    local_id: 0.0
                    for local_id in local_positions[module_id]
                },
            )
            for module_id in sorted(module_root_poses)
        ],
        object_states=[],
        contact_states=[],
        controller_status=ControllerStatus(status="ok", qp_feasible=True),
        task_progress=TaskProgressState(),
    )
    return builder.build(morphology, physical_model, observation).body_pose_world


def _initial_base_pose(
    candidates: Sequence[ContactCandidate],
) -> Pose7D:
    count = float(len(candidates))
    centroid = tuple(
        sum(float(candidate.contact_pose_world[index]) for candidate in candidates)
        / count
        for index in range(3)
    )
    return (*centroid, 0.0, 0.0, 0.0, 1.0)


def _base_position_jacobian(
    point_world: Sequence[float],
    base_position_world: Sequence[float],
) -> tuple[tuple[float, float, float], ...]:
    x, y, z = (
        float(point_world[index]) - float(base_position_world[index])
        for index in range(3)
    )
    return (
        (0.0, z, -y),
        (-z, 0.0, x),
        (y, -x, 0.0),
    )


def _pose_x_axis(pose: Pose7D) -> tuple[float, float, float]:
    rotation = transform_from_pose(pose).rotation
    return (rotation[0][0], rotation[1][0], rotation[2][0])


def _axis_alignment_rotation(
    actual: Sequence[float],
    desired: Sequence[float],
) -> tuple[float, float, float]:
    left = _unit(actual)
    right = _unit(desired)
    cross = (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )
    sine = _norm(cross)
    cosine = max(-1.0, min(1.0, _dot(left, right)))
    if sine <= 1.0e-12:
        if cosine >= 0.0:
            return (0.0, 0.0, 0.0)
        orthogonal = _unit(
            (0.0, -left[2], left[1])
            if abs(left[0]) > 0.5
            else (-left[1], left[0], 0.0)
        )
        return tuple(math.pi * value for value in orthogonal)
    angle = math.atan2(sine, cosine)
    return tuple(angle * value / sine for value in cross)


def _apply_base_delta(
    pose: Pose7D,
    rotation_delta: Sequence[float],
    translation_delta: Sequence[float],
) -> Pose7D:
    current = transform_from_pose(pose)
    delta_rotation = _rotation_from_vector(rotation_delta)
    updated_rotation = matmul(delta_rotation, current.rotation)
    return pose_from_transform(
        type(current)(
            rotation=updated_rotation,
            translation=tuple(
                float(pose[index]) + float(translation_delta[index])
                for index in range(3)
            ),
        )
    )


def _rotation_from_vector(values: Sequence[float]) -> Matrix3:
    angle = _norm(values)
    if angle <= 1.0e-15:
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    x, y, z = (float(value) / angle for value in values)
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return (
        (c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s),
        (y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s),
        (z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c),
    )


def _pose_attitude_error(expected: Pose7D, actual: Pose7D) -> float:
    expected_rotation = transform_from_pose(expected).rotation
    actual_rotation = transform_from_pose(actual).rotation
    delta = matmul(expected_rotation, transpose(actual_rotation))
    quaternion = quat_from_matrix(delta)
    return 2.0 * math.acos(max(-1.0, min(1.0, abs(quaternion[3]))))


def _angle(left: Sequence[float], right: Sequence[float]) -> float:
    return math.acos(max(-1.0, min(1.0, _dot(_unit(left), _unit(right)))))


def _normalized_quaternion(
    values: Sequence[float],
) -> tuple[float, float, float, float]:
    if len(values) != 4:
        raise SchemaValidationError("body orientation must contain four values")
    norm = math.sqrt(sum(float(value) ** 2 for value in values))
    if not math.isfinite(norm) or norm <= 0.0:
        raise SchemaValidationError("body orientation quaternion is invalid")
    result = tuple(float(value) / norm for value in values)
    return result  # type: ignore[return-value]


def _bounded_vector(values: Sequence[float], maximum_norm: float) -> tuple[float, ...]:
    parsed = tuple(float(value) for value in values)
    norm = _norm(parsed)
    if norm <= maximum_norm or norm <= 1.0e-15:
        return parsed
    return tuple(value * maximum_norm / norm for value in parsed)


def _unit(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise SchemaValidationError("three-dimensional vector required")
    norm = _norm(values)
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise SchemaValidationError("vector norm must be positive")
    return tuple(float(value) / norm for value in values)  # type: ignore[return-value]


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right, strict=True))


def _append_code(codes: list[str], code: str) -> None:
    if code not in codes:
        codes.append(code)


def _numpy() -> object:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "ArticulatedContactIKSolver requires numpy at solve time"
        ) from exc
    return np


__all__ = [
    "ARTICULATED_IK_TEACHER_VERSION",
    "ARTICULATED_REACHABILITY_VERSION",
    "ArticulatedContactIKSolver",
    "ArticulatedIKConfig",
    "ArticulatedIKSolution",
    "ArticulatedReachabilityConfig",
    "ArticulatedTrajectoryReachabilityEvaluator",
    "base_pose_for_centroidal_target",
    "resolve_mesh_backed_anchor_references",
]
