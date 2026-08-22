from __future__ import annotations

"""Deterministic raw-pi_H to nominal-joint-trajectory resolution.

The raw high-level trajectory is policy-owned and must not contain joint
targets.  This module resolves a detached copy, leaving the proposal byte/hash
identity unchanged for archive, hard-check, and policy-credit purposes.
"""

from dataclasses import dataclass
import math
from pathlib import Path
from time import perf_counter
from typing import Mapping, Protocol, Sequence

from amsrr.feasibility.articulated_reachability import (
    CENTROIDAL_POSTURE_IK_VERSION,
    CentroidalPostureIKConfig,
    CentroidalPostureIKSolver,
)
from amsrr.policies.contact_wrench_trajectory_runtime import (
    ContactWrenchTrajectoryExecutor,
)
from amsrr.robot_model.whole_structure_kinematics import (
    ordered_global_dock_joint_ids,
)
from amsrr.schemas.common import Pose7D, SchemaValidationError
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import (
    ContactWrenchTrajectory,
    InteractionKnot,
    PostureTarget,
)
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.utils.hashing import stable_hash


ORDER9_POSTURE_TRAJECTORY_RESOLVER_VERSION = (
    "order9_posture_trajectory_resolver_v6_planner_seed_reference"
)


@dataclass(frozen=True)
class Order9PostureCollisionBox:
    """Static box obstacle checked by the convex posture collision path."""

    box_id: str
    size_m: tuple[float, float, float]
    pose_world: Pose7D

    def __post_init__(self) -> None:
        if not self.box_id:
            raise ValueError("collision box_id must be non-empty")
        if len(self.size_m) != 3 or any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in self.size_m
        ):
            raise ValueError(
                "collision box size_m must contain three positive values"
            )
        if len(self.pose_world) != 7 or any(
            not math.isfinite(float(value)) for value in self.pose_world
        ):
            raise ValueError("collision box pose_world must be a finite Pose7D")


@dataclass(frozen=True)
class Order9PostureCollisionObject:
    """Box-object identity needed by the interim convex online hard gate."""

    object_id: str
    size_m: tuple[float, float, float]
    initial_pose_world: Pose7D | None = None
    environment_boxes: tuple[Order9PostureCollisionBox, ...] = ()
    ground_plane_z_m: float | None = None

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("collision object_id must be non-empty")
        if len(self.size_m) != 3 or any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in self.size_m
        ):
            raise ValueError(
                "collision object size_m must contain three positive values"
            )
        if self.initial_pose_world is not None:
            if len(self.initial_pose_world) != 7 or any(
                not math.isfinite(float(value))
                for value in self.initial_pose_world
            ):
                raise ValueError(
                    "collision object initial pose must be a finite Pose7D"
                )
        if len({box.box_id for box in self.environment_boxes}) != len(
            self.environment_boxes
        ):
            raise ValueError(
                "collision environment box identities must be unique"
            )
        if self.ground_plane_z_m is not None and not math.isfinite(
            float(self.ground_plane_z_m)
        ):
            raise ValueError("collision ground plane z must be finite")


@dataclass(frozen=True)
class Order9PostureResolverConfig:
    output_rate_hz: float = 20.0
    joint_velocity_fraction: float = 0.90
    maximum_output_knots: int = 4096
    enforce_joint_rate: bool = True
    require_external_validation: bool = False
    anchor_position_tolerance_m: float = 0.005
    collision_margin_m: float = 0.005

    def __post_init__(self) -> None:
        if (
            not math.isfinite(float(self.output_rate_hz))
            or self.output_rate_hz <= 0.0
        ):
            raise ValueError("output_rate_hz must be finite and positive")
        if not 0.0 < float(self.joint_velocity_fraction) <= 1.0:
            raise ValueError("joint_velocity_fraction must be in (0, 1]")
        if self.maximum_output_knots < 2:
            raise ValueError("maximum_output_knots must be at least two")
        if (
            not math.isfinite(float(self.anchor_position_tolerance_m))
            or self.anchor_position_tolerance_m <= 0.0
        ):
            raise ValueError(
                "anchor_position_tolerance_m must be finite and positive"
            )
        if (
            not math.isfinite(float(self.collision_margin_m))
            or self.collision_margin_m <= 0.0
        ):
            raise ValueError("collision_margin_m must be finite and positive")
        if not isinstance(self.enforce_joint_rate, bool):
            raise ValueError("enforce_joint_rate must be boolean")


@dataclass(frozen=True)
class Order9PostureValidationResult:
    accepted: bool
    validator_version: str
    violation_codes: tuple[str, ...] = ()
    margins: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if not self.validator_version:
            raise ValueError("validator_version must be non-empty")
        if self.accepted and self.violation_codes:
            raise ValueError(
                "accepted validation cannot contain violation codes"
            )
        for key, value in (self.margins or {}).items():
            if not key or not math.isfinite(float(value)):
                raise ValueError(
                    "validation margins must have non-empty finite entries"
                )


class Order9PostureTrajectoryValidator(Protocol):
    def __call__(
        self,
        *,
        context: HighLevelPolicyContext,
        raw_trajectory: ContactWrenchTrajectory,
        resolved_trajectory: ContactWrenchTrajectory,
    ) -> Order9PostureValidationResult:
        """Validate a resolved path without modifying either trajectory."""


@dataclass(frozen=True)
class Order9PostureResolutionEvidence:
    resolver_version: str
    solver_version: str
    raw_trajectory_hash: str
    resolved_trajectory_hash: str
    output_rate_hz: float
    raw_knot_count: int
    resolved_knot_count: int
    solved_knot_count: int
    solver_iterations: tuple[int, ...]
    solver_cache_hits: tuple[bool, ...]
    maximum_anchor_position_error_m: float
    maximum_anchor_attitude_error_rad: float
    maximum_joint_rate_rad_s: float
    maximum_joint_rate_segment_index: int | None
    minimum_required_segment_duration_s: float
    joint_rate_limit_rad_s: float
    minimum_joint_rate_margin_rad_s: float
    solve_wall_time_s: float
    collision_gate_status: str
    collision_gate_version: str | None
    minimum_collision_clearance_m: float | None
    maximum_collision_violating_pair_count: int
    external_validation_status: str
    external_validator_version: str | None
    external_violation_codes: tuple[str, ...]
    external_margins: Mapping[str, float]
    initial_joint_state_hash: str
    nominal_joint_seed_trajectory_hash: str | None

    def identity_dict(self) -> dict[str, object]:
        """Deterministic resolver identity; excludes wall-clock telemetry."""

        return {
            "resolver_version": self.resolver_version,
            "solver_version": self.solver_version,
            "raw_trajectory_hash": self.raw_trajectory_hash,
            "resolved_trajectory_hash": self.resolved_trajectory_hash,
            "output_rate_hz": self.output_rate_hz,
            "raw_knot_count": self.raw_knot_count,
            "resolved_knot_count": self.resolved_knot_count,
            "solved_knot_count": self.solved_knot_count,
            "solver_iterations": list(self.solver_iterations),
            "maximum_anchor_position_error_m": (
                self.maximum_anchor_position_error_m
            ),
            "maximum_anchor_attitude_error_rad": (
                self.maximum_anchor_attitude_error_rad
            ),
            "maximum_joint_rate_rad_s": self.maximum_joint_rate_rad_s,
            "maximum_joint_rate_segment_index": (
                self.maximum_joint_rate_segment_index
            ),
            "minimum_required_segment_duration_s": (
                self.minimum_required_segment_duration_s
            ),
            "joint_rate_limit_rad_s": self.joint_rate_limit_rad_s,
            "minimum_joint_rate_margin_rad_s": (
                self.minimum_joint_rate_margin_rad_s
            ),
            "collision_gate_status": self.collision_gate_status,
            "collision_gate_version": self.collision_gate_version,
            "minimum_collision_clearance_m": (
                self.minimum_collision_clearance_m
            ),
            "maximum_collision_violating_pair_count": (
                self.maximum_collision_violating_pair_count
            ),
            "external_validation_status": self.external_validation_status,
            "external_validator_version": self.external_validator_version,
            "external_violation_codes": list(self.external_violation_codes),
            "external_margins": dict(self.external_margins),
            "initial_joint_state_hash": self.initial_joint_state_hash,
            "nominal_joint_seed_trajectory_hash": (
                self.nominal_joint_seed_trajectory_hash
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.identity_dict(),
            "solver_cache_hits": list(self.solver_cache_hits),
            "solve_wall_time_s": self.solve_wall_time_s,
        }


@dataclass(frozen=True)
class Order9ResolvedPostureTrajectory:
    raw_trajectory: ContactWrenchTrajectory
    trajectory: ContactWrenchTrajectory
    evidence: Order9PostureResolutionEvidence


class Order9PostureTrajectoryResolver:
    """Produce a dense nominal joint trajectory from an immutable raw plan."""

    resolver_version = ORDER9_POSTURE_TRAJECTORY_RESOLVER_VERSION

    def __init__(
        self,
        physical_model: PhysicalModel,
        *,
        config: Order9PostureResolverConfig | None = None,
        ik_solver: CentroidalPostureIKSolver | None = None,
        collision_object: Order9PostureCollisionObject | None = None,
        prefer_native_solver: bool = True,
        require_native_solver: bool = False,
        nominal_collision_pair_manifest: str | Path | None = None,
    ) -> None:
        if require_native_solver and not prefer_native_solver:
            raise ValueError(
                "require_native_solver cannot disable native preference"
            )
        self.physical_model = physical_model
        self.config = config or Order9PostureResolverConfig()
        self.collision_object = collision_object
        self.ik_solver = ik_solver or _default_posture_ik_solver(
            physical_model,
            collision_object=collision_object,
            prefer_native=prefer_native_solver,
            require_native=require_native_solver,
            nominal_collision_pair_manifest=(
                nominal_collision_pair_manifest
            ),
            ik_config=CentroidalPostureIKConfig(
                anchor_position_tolerance_m=(
                    self.config.anchor_position_tolerance_m
                ),
                continuity_regularization_weight=1.0e-5,
                # The resolver already supplies continuation seeds from the
                # preceding raw/dense trajectory.  A free-base relaxed seed
                # can jump branches and create a discontinuous nominal path.
                use_relaxed_seed=False,
            ),
            collision_margin_m=self.config.collision_margin_m,
        )
        collision_methods = (
            hasattr(self.ik_solver, "set_collision_scene"),
            hasattr(self.ik_solver, "check_configuration"),
        )
        if collision_object is not None and not all(collision_methods):
            raise ValueError(
                "a collision object requires a collision-aware IK solver"
            )
        if collision_object is None and any(collision_methods):
            raise ValueError(
                "a collision-aware IK solver requires collision object "
                "identity"
            )

    def resolve(
        self,
        *,
        context: HighLevelPolicyContext,
        raw_trajectory: ContactWrenchTrajectory,
        initial_joint_positions_rad: Mapping[str, float],
        nominal_joint_seed_positions_by_raw_knot: Sequence[
            Mapping[str, float]
        ]
        | None = None,
        external_validator: Order9PostureTrajectoryValidator | None = None,
    ) -> Order9ResolvedPostureTrajectory:
        start_time = perf_counter()
        source_hash_before = stable_hash(raw_trajectory.to_dict())
        raw = ContactWrenchTrajectory.from_dict(raw_trajectory.to_dict())
        _validate_raw_pi_h_authority(raw)
        executor = ContactWrenchTrajectoryExecutor()
        executor.install(raw, plan_start_time_s=0.0)

        ordered_ids = ordered_global_dock_joint_ids(
            context.morphology_graph, self.physical_model
        )
        initial_q = _validate_initial_joint_state(
            initial_joint_positions_rad,
            ordered_ids,
        )
        nominal_seed_positions = _validate_nominal_joint_seed_positions(
            nominal_joint_seed_positions_by_raw_knot,
            raw_knot_count=len(raw.knots),
            ordered_ids=ordered_ids,
        )
        iterations: list[int] = []
        cache_hits: list[bool] = []
        position_errors: list[float] = []
        attitude_errors: list[float] = []
        collision_clearances: list[float] = []
        collision_violating_pair_counts: list[int] = []

        def solve_knot(
            knot: InteractionKnot,
            *,
            seed_q: Mapping[str, float],
            label: str,
            record_evidence: bool,
        ) -> dict[str, float]:
            centroidal = knot.centroidal_target
            if (
                centroidal is None
                or centroidal.com_pos_world is None
                or centroidal.body_orientation_world is None
            ):
                raise SchemaValidationError(
                    "posture resolver requires a complete centroidal pose at "
                    f"{label}"
                )
            posture = knot.posture_target
            anchor_targets = (
                {}
                if posture is None
                or posture.free_anchor_pose_targets is None
                else dict(posture.free_anchor_pose_targets)
            )
            active_anchor_ids = {
                int(assignment.anchor_id)
                for assignment in knot.contact_assignments
                if assignment.schedule_state
                in {"attach", "maintain", "slide"}
            }
            if not active_anchor_ids.issubset(anchor_targets):
                missing = sorted(active_anchor_ids - set(anchor_targets))
                raise SchemaValidationError(
                    "posture resolver raw active knot lacks anchor pose "
                    f"targets for {missing} at {label}"
                )
            if not anchor_targets:
                return dict(seed_q)
            centroidal_pose_world = (
                *centroidal.com_pos_world,
                *centroidal.body_orientation_world,
            )
            if self.collision_object is not None:
                self.ik_solver.set_collision_scene(
                    morphology=context.morphology_graph,
                    object_pose_world=_collision_object_pose(
                        context=context,
                        knot=knot,
                        collision_object=self.collision_object,
                    ),
                    object_size_m=self.collision_object.size_m,
                    allowed_anchor_ids=_allowed_object_contact_anchors(
                        context=context,
                        knot=knot,
                        object_id=self.collision_object.object_id,
                    ),
                )
            solution = self.ik_solver.solve(
                morphology=context.morphology_graph,
                centroidal_pose_world=centroidal_pose_world,
                anchor_pose_targets_world=anchor_targets,
                initial_joint_positions_rad=seed_q,
            )
            if record_evidence:
                iterations.append(int(solution.iterations))
                cache_hits.append(bool(solution.cache_hit))
                position_errors.append(
                    float(solution.maximum_position_error_m)
                )
                attitude_errors.append(
                    float(solution.maximum_attitude_error_rad)
                )
            if not solution.feasible:
                collision_metrics = getattr(
                    self.ik_solver,
                    "last_native_collision_metrics",
                    None,
                )
                raise SchemaValidationError(
                    "posture resolver could not realize "
                    f"{label} without changing its CoM/anchor plan "
                    f"(position={solution.maximum_position_error_m:.6g}, "
                    "attitude="
                    f"{solution.maximum_attitude_error_rad:.6g}, "
                    f"collision={collision_metrics})"
                )
            if self.collision_object is not None and record_evidence:
                # The primary movable-object scene is already installed for
                # the IK solve above.  Validate that scene directly, then
                # switch only for the additional static environment boxes.
                # Sparse knots are included again in the dense output, so
                # checking the dense pass avoids redundant native calls.
                collision_scenes = [
                    (
                        self.collision_object.object_id,
                        None,
                        None,
                        None,
                    ),
                    *[
                        (
                            box.box_id,
                            box.pose_world,
                            box.size_m,
                            (),
                        )
                        for box in self.collision_object.environment_boxes
                    ],
                ]
                for (
                    obstacle_id,
                    obstacle_pose,
                    obstacle_size,
                    allowed_anchor_ids,
                ) in collision_scenes:
                    if obstacle_pose is not None:
                        self.ik_solver.set_collision_scene(
                            morphology=context.morphology_graph,
                            object_pose_world=obstacle_pose,
                            object_size_m=obstacle_size,
                            allowed_anchor_ids=allowed_anchor_ids,
                        )
                    collision = self.ik_solver.check_configuration(
                        morphology=context.morphology_graph,
                        centroidal_pose_world=centroidal_pose_world,
                        joint_positions_rad=solution.joint_positions_rad,
                        exact=False,
                        margin_m=float(
                            self.ik_solver.collision_config.collision_margin_m
                        ),
                        ground_plane_z_m=(
                            self.collision_object.ground_plane_z_m
                        ),
                    )
                    collision_clearances.append(
                        float(collision["minimum_clearance_m"])
                    )
                    collision_violating_pair_counts.append(
                        int(collision["violating_pair_count"])
                    )
                    if collision.get("accepted") is not True:
                        raise SchemaValidationError(
                            "posture resolver convex collision gate rejected "
                            f"{label} against {obstacle_id!r} "
                            "(minimum_clearance_m="
                            f"{collision['minimum_clearance_m']:.6g}, "
                            "violating_pair_count="
                            f"{collision['violating_pair_count']})"
                        )
            return dict(solution.joint_positions_rad)

        # First solve the raw 1--2 Hz waypoints.  Their q values are internal
        # IK seeds only; they never enter the raw pi_H contract.  Interpolating
        # these seeds for each dense target prevents a redundant articulated
        # mechanism from jumping to a different IK branch between otherwise
        # nearby task-space samples.
        raw_times = tuple(float(knot.t_rel_s) for knot in raw.knots)
        sparse_q: list[dict[str, float]] = []
        current_q = dict(initial_q)
        for raw_index, knot in enumerate(raw.knots):
            current_q = solve_knot(
                knot,
                seed_q=(
                    current_q
                    if nominal_seed_positions is None
                    else nominal_seed_positions[raw_index]
                ),
                label=f"raw knot {raw_index}",
                record_evidence=False,
            )
            sparse_q.append(dict(current_q))

        times = _dense_sample_times(
            raw,
            output_rate_hz=self.config.output_rate_hz,
            maximum_output_knots=self.config.maximum_output_knots,
        )
        sampled_knots = [
            executor.sample(time_s=time_s).active_knot
            for time_s in times
        ]
        resolved_q: list[dict[str, float]] = []
        for knot_index, (time_s, knot) in enumerate(
            zip(times, sampled_knots, strict=True)
        ):
            left, right, blend = _sparse_bracket(
                list(raw_times),
                float(time_s),
            )
            seed_q = {
                joint_id: (
                    (1.0 - blend) * sparse_q[left][joint_id]
                    + blend * sparse_q[right][joint_id]
                )
                for joint_id in ordered_ids
            }
            resolved_q.append(
                solve_knot(
                    knot,
                    seed_q=seed_q,
                    label=f"dense knot {knot_index}",
                    record_evidence=True,
                )
            )

        joint_rate_limit = (
            _dock_velocity_limit(self.physical_model)
            * self.config.joint_velocity_fraction
        )
        (
            maximum_joint_rate,
            maximum_rate_segment,
            minimum_required_segment_duration,
        ) = _maximum_timed_joint_rate(
            times,
            resolved_q,
            ordered_ids,
            joint_rate_limit_rad_s=joint_rate_limit,
        )
        if (
            self.config.enforce_joint_rate
            and maximum_joint_rate > joint_rate_limit + 1.0e-9
        ):
            raise SchemaValidationError(
                "posture resolver would exceed the unchanged raw timing's "
                "Dock joint-rate limit "
                f"({maximum_joint_rate:.6g} > {joint_rate_limit:.6g} rad/s)"
            )
        resolved = self._dense_trajectory(
            raw=raw,
            times=times,
            sampled_knots=sampled_knots,
            resolved_q=resolved_q,
            ordered_ids=ordered_ids,
        )
        validation = None
        if external_validator is not None:
            validation = external_validator(
                context=context,
                raw_trajectory=raw,
                resolved_trajectory=resolved,
            )
            if not validation.accepted:
                raise SchemaValidationError(
                    "posture resolver external validation rejected the "
                    "resolved path: "
                    + ",".join(validation.violation_codes)
                )
        elif self.config.require_external_validation:
            raise SchemaValidationError(
                "posture resolver requires an external trajectory validator"
            )

        source_hash_after = stable_hash(raw_trajectory.to_dict())
        if source_hash_after != source_hash_before:
            raise RuntimeError(
                "posture resolver mutated the policy-owned raw trajectory"
            )
        resolved_hash = stable_hash(resolved.to_dict())
        evidence = Order9PostureResolutionEvidence(
            resolver_version=self.resolver_version,
            solver_version=getattr(
                self.ik_solver,
                "solver_version",
                CENTROIDAL_POSTURE_IK_VERSION,
            ),
            raw_trajectory_hash=source_hash_before,
            resolved_trajectory_hash=resolved_hash,
            output_rate_hz=float(self.config.output_rate_hz),
            raw_knot_count=len(raw.knots),
            resolved_knot_count=len(resolved.knots),
            solved_knot_count=len(iterations),
            solver_iterations=tuple(iterations),
            solver_cache_hits=tuple(cache_hits),
            maximum_anchor_position_error_m=max(
                position_errors, default=0.0
            ),
            maximum_anchor_attitude_error_rad=max(
                attitude_errors, default=0.0
            ),
            maximum_joint_rate_rad_s=maximum_joint_rate,
            maximum_joint_rate_segment_index=maximum_rate_segment,
            minimum_required_segment_duration_s=(
                minimum_required_segment_duration
            ),
            joint_rate_limit_rad_s=joint_rate_limit,
            minimum_joint_rate_margin_rad_s=(
                joint_rate_limit - maximum_joint_rate
            ),
            solve_wall_time_s=perf_counter() - start_time,
            collision_gate_status=(
                "accepted"
                if self.collision_object is not None
                else "not_configured"
            ),
            collision_gate_version=(
                None
                if self.collision_object is None
                else getattr(self.ik_solver, "solver_version", None)
            ),
            minimum_collision_clearance_m=(
                None
                if not collision_clearances
                else min(collision_clearances)
            ),
            maximum_collision_violating_pair_count=max(
                collision_violating_pair_counts,
                default=0,
            ),
            external_validation_status=(
                "accepted" if validation is not None else "not_run"
            ),
            external_validator_version=(
                None if validation is None else validation.validator_version
            ),
            external_violation_codes=(
                ()
                if validation is None
                else tuple(validation.violation_codes)
            ),
            external_margins=(
                {}
                if validation is None
                else dict(validation.margins or {})
            ),
            initial_joint_state_hash=stable_hash(initial_q),
            nominal_joint_seed_trajectory_hash=(
                None
                if nominal_seed_positions is None
                else stable_hash(nominal_seed_positions)
            ),
        )
        return Order9ResolvedPostureTrajectory(
            raw_trajectory=raw,
            trajectory=resolved,
            evidence=evidence,
        )

    def _dense_trajectory(
        self,
        *,
        raw: ContactWrenchTrajectory,
        times: tuple[float, ...],
        sampled_knots: list[InteractionKnot],
        resolved_q: list[dict[str, float]],
        ordered_ids: tuple[str, ...],
    ) -> ContactWrenchTrajectory:
        if not (
            len(times) == len(sampled_knots) == len(resolved_q)
        ):
            raise SchemaValidationError(
                "posture resolver dense IK sample identity differs"
            )
        knots = []
        for index, (time_s, source_knot, q) in enumerate(
            zip(times, sampled_knots, resolved_q, strict=True)
        ):
            knot = InteractionKnot.from_dict(source_knot.to_dict())
            if len(times) == 1:
                qdot = {joint_id: 0.0 for joint_id in ordered_ids}
            else:
                left = max(0, index - 1)
                right = min(len(times) - 1, index + 1)
                duration = float(times[right]) - float(times[left])
                qdot = {
                    joint_id: (
                        resolved_q[right][joint_id]
                        - resolved_q[left][joint_id]
                    )
                    / duration
                    for joint_id in ordered_ids
                }
            free_anchor_targets = (
                None
                if knot.posture_target is None
                else knot.posture_target.free_anchor_pose_targets
            )
            knot.posture_target = PostureTarget(
                joint_pos_target=q,
                joint_vel_target=qdot,
                free_anchor_pose_targets=free_anchor_targets,
            )
            knots.append(knot)
        trajectory = ContactWrenchTrajectory(
            horizon_s=float(raw.horizon_s),
            dt_s=1.0 / float(self.config.output_rate_hz),
            knots=knots,
            derived_mode_label=(
                f"{self.resolver_version}:"
                f"source={raw.derived_mode_label or 'unspecified'}"
            ),
            contract_version=raw.contract_version,
        )
        trajectory.validate()
        # Apply runtime's stricter monotonic/horizon/finite validation too.
        verifier = ContactWrenchTrajectoryExecutor()
        verifier.install(trajectory, plan_start_time_s=0.0)
        return ContactWrenchTrajectory.from_dict(trajectory.to_dict())


def _default_posture_ik_solver(
    physical_model: PhysicalModel,
    *,
    collision_object: Order9PostureCollisionObject | None,
    prefer_native: bool,
    require_native: bool,
    nominal_collision_pair_manifest: str | Path | None,
    ik_config: CentroidalPostureIKConfig,
    collision_margin_m: float,
) -> CentroidalPostureIKSolver:
    if not prefer_native:
        return CentroidalPostureIKSolver(
            physical_model,
            config=ik_config,
        )
    from amsrr.feasibility.order9_native_loader import (
        Order9NativePostureIKUnavailable,
        load_order9_posture_native,
    )

    native_required = require_native or collision_object is not None
    try:
        # Load now so ``auto`` can choose the readable Python fallback before
        # the first morphology is compiled.
        load_order9_posture_native()
    except Order9NativePostureIKUnavailable:
        if native_required:
            raise
        return CentroidalPostureIKSolver(
            physical_model,
            config=ik_config,
        )

    from amsrr.feasibility.order9_native_posture_ik import (
        CppWholeStructureKinematics,
        NativeCentroidalPostureIKSolver,
    )

    if collision_object is None:
        kinematics = CppWholeStructureKinematics(physical_model)
        return NativeCentroidalPostureIKSolver(
            physical_model,
            kinematics=kinematics,
            config=ik_config,
        )

    from amsrr.feasibility.order9_posture_collision import (
        CollisionAwareIKConfig,
        CollisionAwareNativeCentroidalPostureIKSolver,
    )

    manifest = (
        Path(nominal_collision_pair_manifest).expanduser().resolve()
        if nominal_collision_pair_manifest is not None
        else (
            Path(__file__).resolve().parents[2]
            / "configs"
            / "robot"
            / "holon_nominal_collision_pairs_v1.json"
        )
    )
    if not manifest.is_file():
        raise FileNotFoundError(
            f"nominal collision-pair manifest is missing: {manifest}"
        )
    kinematics = CppWholeStructureKinematics(
        physical_model,
        enable_collision_geometry=True,
        enable_exact_collision_geometry=False,
        nominal_collision_pair_manifest=manifest,
    )
    return CollisionAwareNativeCentroidalPostureIKSolver(
        physical_model,
        kinematics=kinematics,
        config=ik_config,
        collision_config=CollisionAwareIKConfig(
            collision_margin_m=float(collision_margin_m),
            collision_activation_distance_m=max(
                0.020, float(collision_margin_m)
            ),
        ),
    )


def _collision_object_pose(
    *,
    context: HighLevelPolicyContext,
    knot: InteractionKnot,
    collision_object: Order9PostureCollisionObject,
) -> Pose7D:
    targets = [
        target
        for target in knot.object_targets
        if target.object_id == collision_object.object_id
        and target.pose_target_world is not None
    ]
    if len(targets) > 1:
        raise SchemaValidationError(
            "posture resolver received duplicate collision-object targets"
        )
    if targets:
        return tuple(
            float(value) for value in targets[0].pose_target_world
        )
    observation = context.runtime_observation
    if observation is not None:
        states = [
            state
            for state in observation.object_states
            if state.object_id == collision_object.object_id
        ]
        if len(states) > 1:
            raise SchemaValidationError(
                "runtime observation contains duplicate collision objects"
            )
        if states:
            return tuple(float(value) for value in states[0].pose_world)
    if collision_object.initial_pose_world is not None:
        return tuple(
            float(value) for value in collision_object.initial_pose_world
        )
    raise SchemaValidationError(
        "posture collision gate requires the current pose of object "
        f"{collision_object.object_id!r}"
    )


def _allowed_object_contact_anchors(
    *,
    context: HighLevelPolicyContext,
    knot: InteractionKnot,
    object_id: str,
) -> tuple[int, ...]:
    candidates = {
        int(candidate.candidate_id): candidate
        for candidate in context.contact_candidate_set.candidates
    }
    allowed: set[int] = set()
    for assignment in knot.contact_assignments:
        # Release begins from an intentionally contacting state.  Keep the
        # assigned anchor/object pairs task-allowed while the free-anchor
        # targets move them out; all other object collisions remain prohibited.
        if assignment.schedule_state not in {
            "attach",
            "maintain",
            "slide",
            "release",
        }:
            continue
        candidate = candidates.get(int(assignment.candidate_id))
        if candidate is None:
            raise SchemaValidationError(
                "posture collision gate cannot resolve assigned candidate "
                f"{assignment.candidate_id}"
            )
        if candidate.target_entity_id == object_id:
            allowed.add(int(assignment.anchor_id))
    return tuple(sorted(allowed))


def _validate_raw_pi_h_authority(
    trajectory: ContactWrenchTrajectory,
) -> None:
    for knot_index, knot in enumerate(trajectory.knots):
        posture = knot.posture_target
        if posture is None:
            continue
        if (
            posture.joint_pos_target is not None
            or posture.joint_vel_target is not None
        ):
            raise SchemaValidationError(
                "raw pi_H trajectory must not contain resolver-owned joint "
                f"targets (knot {knot_index})"
            )


def _validate_initial_joint_state(
    values: Mapping[str, float],
    ordered_ids: tuple[str, ...],
) -> dict[str, float]:
    actual = set(values)
    expected = set(ordered_ids)
    if actual != expected:
        raise SchemaValidationError(
            "posture resolver initial joint-state identity differs "
            f"(missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)})"
        )
    result = {joint_id: float(values[joint_id]) for joint_id in ordered_ids}
    if any(not math.isfinite(value) for value in result.values()):
        raise SchemaValidationError(
            "posture resolver initial joint state must be finite"
        )
    return result


def _validate_nominal_joint_seed_positions(
    values: Sequence[Mapping[str, float]] | None,
    *,
    raw_knot_count: int,
    ordered_ids: tuple[str, ...],
) -> tuple[dict[str, float], ...] | None:
    if values is None:
        return None
    if len(values) != raw_knot_count:
        raise SchemaValidationError(
            "posture resolver nominal seed count differs from raw knots"
        )
    result = tuple(
        _validate_initial_joint_state(value, ordered_ids)
        for value in values
    )
    return result


def _maximum_timed_joint_rate(
    times: tuple[float, ...],
    q_samples: list[dict[str, float]],
    ordered_ids: tuple[str, ...],
    *,
    joint_rate_limit_rad_s: float,
) -> tuple[float, int | None, float]:
    if len(times) != len(q_samples):
        raise SchemaValidationError(
            "posture resolver rate sample identity differs"
        )
    maximum = 0.0
    maximum_segment: int | None = None
    required_duration = 0.0
    for index in range(1, len(times)):
        duration = float(times[index]) - float(times[index - 1])
        if duration <= 0.0:
            raise SchemaValidationError(
                "posture resolver raw knot times must be strictly increasing"
            )
        maximum_delta = max(
            (
                abs(
                    q_samples[index][joint_id]
                    - q_samples[index - 1][joint_id]
                )
                for joint_id in ordered_ids
            ),
            default=0.0,
        )
        rate = maximum_delta / duration
        if rate > maximum:
            maximum = rate
            maximum_segment = index
            required_duration = maximum_delta / joint_rate_limit_rad_s
    return maximum, maximum_segment, required_duration


def _dense_sample_times(
    trajectory: ContactWrenchTrajectory,
    *,
    output_rate_hz: float,
    maximum_output_knots: int,
) -> tuple[float, ...]:
    period = 1.0 / float(output_rate_hz)
    count = int(math.floor(float(trajectory.horizon_s) / period))
    values = {
        round(index * period, 12)
        for index in range(count + 1)
    }
    values.update(
        round(float(knot.t_rel_s), 12)
        for knot in trajectory.knots
    )
    values.add(round(float(trajectory.horizon_s), 12))
    output = tuple(sorted(values))
    if len(output) > maximum_output_knots:
        raise SchemaValidationError(
            "posture resolver dense trajectory exceeds maximum_output_knots "
            f"(horizon_s={trajectory.horizon_s:.6g}, "
            f"knot_count={len(output)}, limit={maximum_output_knots})"
        )
    return output


def _sparse_bracket(
    times: list[float],
    sample_time_s: float,
) -> tuple[int, int, float]:
    left = 0
    for index, value in enumerate(times):
        if value <= sample_time_s + 1.0e-12:
            left = index
        else:
            break
    right = min(left + 1, len(times) - 1)
    if left == right:
        return left, right, 0.0
    duration = times[right] - times[left]
    ratio = min(
        max((sample_time_s - times[left]) / duration, 0.0),
        1.0,
    )
    return left, right, ratio


def _dock_velocity_limit(physical_model: PhysicalModel) -> float:
    mechanism_ids = {
        str(port.mechanical_limits.get("mechanism_joint_id"))
        for port in physical_model.dock_ports
    }
    values = [
        float(joint.velocity_limit)
        for joint in physical_model.joints
        if joint.joint_id in mechanism_ids
        and joint.velocity_limit is not None
        and math.isfinite(float(joint.velocity_limit))
        and float(joint.velocity_limit) > 0.0
    ]
    specs = physical_model.metadata.get("joint_actuator_specs")
    dock = specs.get("dock") if isinstance(specs, dict) else None
    drive = dock.get("simulation_drive") if isinstance(dock, dict) else None
    safe = (
        drive.get("safe_velocity_limit_rad_s")
        if isinstance(drive, dict)
        else None
    )
    if (
        isinstance(safe, (int, float))
        and not isinstance(safe, bool)
        and math.isfinite(float(safe))
        and float(safe) > 0.0
    ):
        values.append(float(safe))
    if not values:
        raise SchemaValidationError(
            "posture resolver requires a positive Dock velocity limit"
        )
    return min(values)


def _lerp(left: float, right: float, ratio: float) -> float:
    return float(left) + (float(right) - float(left)) * float(ratio)


__all__ = [
    "ORDER9_POSTURE_TRAJECTORY_RESOLVER_VERSION",
    "Order9PostureCollisionObject",
    "Order9PostureResolutionEvidence",
    "Order9PostureResolverConfig",
    "Order9PostureTrajectoryResolver",
    "Order9PostureTrajectoryValidator",
    "Order9PostureValidationResult",
    "Order9ResolvedPostureTrajectory",
]
