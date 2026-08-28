from __future__ import annotations

"""Fast, Isaac-free screening of an already resolved R1 teacher path.

The screen deliberately performs no IK solve, trajectory search, controller
step, or simulation.  It scans the dense trajectory that the deterministic
teacher and detached IK resolver already produced and verifies its bound
kinematic/collision evidence through grasp-pose acquisition.
"""

from dataclasses import dataclass
import math
from time import perf_counter
from typing import Mapping, Sequence

from amsrr.robot_model.whole_structure_kinematics import (
    ordered_global_dock_joint_ids,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import ContactWrenchTrajectory
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.order9_c3_nominal_trajectory import (
    Order9C3NominalTrajectory,
    Order9C3NominalWindow,
)
from amsrr.training.order9_posture_resolver import (
    Order9PostureResolutionEvidence,
)
from amsrr.utils.hashing import stable_hash

ORDER9_R1_FAST_SCREEN_VERSION = "order9_r1_exact_tracking_fast_screen_v1"
ORDER9_R1_FAST_SCREEN_EXECUTION_MODEL = (
    "resolved_pi_h_ik_exact_tracking_to_grasp_pose_v1"
)
ORDER9_R1_FAST_SCREEN_PHASES = ("approach", "contact_acquisition")

R1_FAST_SCREEN_PHASE_CODE = "E_R1_FAST_SCREEN_PHASE"
R1_FAST_SCREEN_GRASP_POSE_CODE = "E_R1_FAST_SCREEN_GRASP_POSE"
R1_FAST_SCREEN_TRAJECTORY_CODE = "E_R1_FAST_SCREEN_TRAJECTORY"
R1_FAST_SCREEN_CHAIN_CODE = "E_R1_FAST_SCREEN_CHAIN"
R1_FAST_SCREEN_JOINT_LIMIT_CODE = "E_R1_FAST_SCREEN_JOINT_LIMIT_RESERVE"
R1_FAST_SCREEN_JOINT_RATE_CODE = "E_R1_FAST_SCREEN_JOINT_RATE"
R1_FAST_SCREEN_BODY_TILT_CODE = "E_R1_FAST_SCREEN_BODY_TILT"
R1_FAST_SCREEN_COLLISION_CODE = "E_R1_FAST_SCREEN_COLLISION"
R1_FAST_SCREEN_PROVENANCE_CODE = "E_R1_FAST_SCREEN_PROVENANCE"


@dataclass(frozen=True)
class Order9R1FastScreenConfig:
    """Cheap post-resolution gates applied before any full-layer replay."""

    minimum_normalized_joint_limit_reserve: float = 0.01
    maximum_body_tilt_rad: float = math.radians(60.0)
    joint_limit_numerical_tolerance_rad: float = 1.0e-6
    chain_joint_position_tolerance_rad: float = 1.0e-6
    required_phases: tuple[str, ...] = ORDER9_R1_FAST_SCREEN_PHASES

    def __post_init__(self) -> None:
        reserve = float(self.minimum_normalized_joint_limit_reserve)
        if not math.isfinite(reserve) or not 0.0 < reserve < 0.5:
            raise ValueError(
                "minimum_normalized_joint_limit_reserve must be in (0, 0.5)"
            )
        tilt = float(self.maximum_body_tilt_rad)
        if not math.isfinite(tilt) or not 0.0 < tilt < math.pi / 2.0:
            raise ValueError("maximum_body_tilt_rad must be in (0, pi/2)")
        for name in (
            "joint_limit_numerical_tolerance_rad",
            "chain_joint_position_tolerance_rad",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if tuple(self.required_phases) != ORDER9_R1_FAST_SCREEN_PHASES:
            raise ValueError(
                "R1 fast screen must end at approach/contact-acquisition grasp pose"
            )


@dataclass(frozen=True)
class Order9R1FastScreenWindow:
    """One phase-local resolved teacher window and its existing evidence."""

    phase: str
    phase_target_reached: bool
    raw_trajectory: ContactWrenchTrajectory
    resolved_trajectory: ContactWrenchTrajectory
    resolution_evidence: Order9PostureResolutionEvidence
    configuration_plan_collision_check_count: int


@dataclass(frozen=True)
class Order9R1FastScreenResult:
    screen_version: str
    execution_model: str
    candidate_id: str
    candidate_task_hash: str
    morphology_hash: str
    physical_model_hash: str
    resolved_path_hash: str
    accepted: bool
    eligible_for_full_control_test: bool
    violation_codes: tuple[str, ...]
    checked_phases: tuple[str, ...]
    window_count: int
    resolved_knot_count: int
    collision_evidence_window_count: int
    configuration_plan_collision_check_count: int
    grasp_pose_reached: bool
    final_maintained_contact_count: int
    minimum_joint_limit_margin_rad: float
    minimum_normalized_joint_limit_reserve: float
    maximum_body_tilt_rad: float
    maximum_joint_rate_rad_s: float
    minimum_joint_rate_margin_rad_s: float
    minimum_collision_clearance_m: float | None
    maximum_collision_violating_pair_count: int
    screen_wall_time_s: float
    isaac_invoked: bool = False
    controller_layers_invoked: bool = False
    ik_resolve_invoked: bool = False
    trajectory_optimization_invoked: bool = False

    def __post_init__(self) -> None:
        if self.screen_version != ORDER9_R1_FAST_SCREEN_VERSION:
            raise ValueError("R1 fast-screen version mismatch")
        if self.execution_model != ORDER9_R1_FAST_SCREEN_EXECUTION_MODEL:
            raise ValueError("R1 fast-screen execution model mismatch")
        if not self.candidate_id:
            raise ValueError("R1 fast-screen candidate id is empty")
        for name in (
            "candidate_task_hash",
            "morphology_hash",
            "physical_model_hash",
            "resolved_path_hash",
        ):
            _require_sha256(str(getattr(self, name)), name)
        if self.accepted != self.eligible_for_full_control_test:
            raise ValueError("R1 fast-screen admission flags disagree")
        if self.accepted and self.violation_codes:
            raise ValueError("accepted R1 fast screen has violation codes")
        if not self.accepted and not self.violation_codes:
            raise ValueError("rejected R1 fast screen lacks a violation code")
        if (
            self.window_count < 1
            or self.resolved_knot_count < 1
            or self.collision_evidence_window_count < 0
            or self.configuration_plan_collision_check_count < 0
            or self.final_maintained_contact_count < 0
            or self.maximum_collision_violating_pair_count < 0
        ):
            raise ValueError("R1 fast-screen counts are invalid")
        for name in (
            "minimum_joint_limit_margin_rad",
            "minimum_normalized_joint_limit_reserve",
            "maximum_body_tilt_rad",
            "maximum_joint_rate_rad_s",
            "minimum_joint_rate_margin_rad_s",
            "screen_wall_time_s",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"R1 fast-screen {name} must be finite")
        if self.minimum_collision_clearance_m is not None and not math.isfinite(
            float(self.minimum_collision_clearance_m)
        ):
            raise ValueError("R1 fast-screen collision clearance must be finite")
        if any(
            (
                self.isaac_invoked,
                self.controller_layers_invoked,
                self.ik_resolve_invoked,
                self.trajectory_optimization_invoked,
            )
        ):
            raise ValueError("R1 fast screen cannot invoke heavy/full-layer work")

    def to_dict(self) -> dict[str, object]:
        return {
            "screen_version": self.screen_version,
            "execution_model": self.execution_model,
            "candidate_id": self.candidate_id,
            "candidate_task_hash": self.candidate_task_hash,
            "morphology_hash": self.morphology_hash,
            "physical_model_hash": self.physical_model_hash,
            "resolved_path_hash": self.resolved_path_hash,
            "accepted": self.accepted,
            "eligible_for_full_control_test": self.eligible_for_full_control_test,
            "violation_codes": list(self.violation_codes),
            "checked_phases": list(self.checked_phases),
            "window_count": self.window_count,
            "resolved_knot_count": self.resolved_knot_count,
            "collision_evidence_window_count": (self.collision_evidence_window_count),
            "configuration_plan_collision_check_count": (
                self.configuration_plan_collision_check_count
            ),
            "grasp_pose_reached": self.grasp_pose_reached,
            "final_maintained_contact_count": (self.final_maintained_contact_count),
            "minimum_joint_limit_margin_rad": (self.minimum_joint_limit_margin_rad),
            "minimum_normalized_joint_limit_reserve": (
                self.minimum_normalized_joint_limit_reserve
            ),
            "maximum_body_tilt_rad": self.maximum_body_tilt_rad,
            "maximum_joint_rate_rad_s": self.maximum_joint_rate_rad_s,
            "minimum_joint_rate_margin_rad_s": (self.minimum_joint_rate_margin_rad_s),
            "minimum_collision_clearance_m": self.minimum_collision_clearance_m,
            "maximum_collision_violating_pair_count": (
                self.maximum_collision_violating_pair_count
            ),
            "screen_wall_time_s": self.screen_wall_time_s,
            "isaac_invoked": self.isaac_invoked,
            "controller_layers_invoked": self.controller_layers_invoked,
            "ik_resolve_invoked": self.ik_resolve_invoked,
            "trajectory_optimization_invoked": (self.trajectory_optimization_invoked),
        }


def screen_order9_r1_nominal_candidate(
    *,
    candidate_id: str,
    task_spec: TaskSpec,
    physical_model: PhysicalModel,
    nominal: Order9C3NominalTrajectory,
    config: Order9R1FastScreenConfig | None = None,
) -> Order9R1FastScreenResult:
    """Adapt an in-memory teacher result to the cheap exact-tracking screen."""

    morphology = nominal.selection_bundle.design_output.target_morphology
    ordered_ids = ordered_global_dock_joint_ids(morphology, physical_model)
    limits = _global_joint_limits(
        ordered_ids=ordered_ids,
        physical_model=physical_model,
    )
    windows = tuple(_screen_window(window) for window in nominal.windows)
    return screen_order9_r1_resolved_grasp_path(
        candidate_id=candidate_id,
        candidate_task_hash=stable_hash(task_spec.to_dict()),
        morphology_hash=stable_hash(morphology.to_dict()),
        physical_model_hash=stable_hash(physical_model.to_dict()),
        ordered_joint_limits_rad=limits,
        windows=windows,
        proxy_collision_validation_status=(nominal.proxy_collision_validation_status),
        config=config,
    )


def screen_order9_r1_resolved_grasp_path(
    *,
    candidate_id: str,
    candidate_task_hash: str,
    morphology_hash: str,
    physical_model_hash: str,
    ordered_joint_limits_rad: Mapping[str, tuple[float, float]],
    windows: Sequence[Order9R1FastScreenWindow],
    proxy_collision_validation_status: str,
    config: Order9R1FastScreenConfig | None = None,
) -> Order9R1FastScreenResult:
    """Screen only persisted values; this function calls no solver or simulator."""

    started = perf_counter()
    settings = config or Order9R1FastScreenConfig()
    violations: list[str] = []
    ordered_ids = tuple(ordered_joint_limits_rad)
    if not ordered_ids:
        raise SchemaValidationError("R1 fast screen requires Dock joint limits")
    _validate_joint_limits(ordered_joint_limits_rad)
    if not candidate_id:
        raise SchemaValidationError("R1 fast-screen candidate id is empty")
    for name, value in (
        ("candidate_task_hash", candidate_task_hash),
        ("morphology_hash", morphology_hash),
        ("physical_model_hash", physical_model_hash),
    ):
        _require_sha256(value, name)

    checked_phases = _compressed_phases(windows)
    if checked_phases != tuple(settings.required_phases):
        _append(violations, R1_FAST_SCREEN_PHASE_CODE)
    if proxy_collision_validation_status != "enforced_during_generation":
        _append(violations, R1_FAST_SCREEN_COLLISION_CODE)

    minimum_limit_margin = math.inf
    minimum_limit_fraction = math.inf
    maximum_body_tilt = 0.0
    maximum_joint_rate = 0.0
    minimum_rate_margin = math.inf
    collision_clearances: list[float] = []
    maximum_collision_violations = 0
    collision_evidence_windows = 0
    collision_checks = 0
    resolved_knot_count = 0
    resolved_hashes: list[str] = []
    previous_final_q: dict[str, float] | None = None

    for window in windows:
        raw = window.raw_trajectory
        resolved = window.resolved_trajectory
        evidence = window.resolution_evidence
        try:
            raw.validate()
            resolved.validate()
        except (SchemaValidationError, TypeError, ValueError):
            _append(violations, R1_FAST_SCREEN_TRAJECTORY_CODE)
        raw_hash = stable_hash(raw.to_dict())
        resolved_hash = stable_hash(resolved.to_dict())
        resolved_hashes.append(resolved_hash)
        if (
            evidence.raw_trajectory_hash != raw_hash
            or evidence.resolved_trajectory_hash != resolved_hash
            or evidence.raw_knot_count != len(raw.knots)
            or evidence.resolved_knot_count != len(resolved.knots)
        ):
            _append(violations, R1_FAST_SCREEN_PROVENANCE_CODE)
        if (
            evidence.collision_gate_status != "accepted"
            or evidence.collision_gate_version is None
            or evidence.maximum_collision_violating_pair_count != 0
            or evidence.minimum_collision_clearance_m is None
        ):
            _append(violations, R1_FAST_SCREEN_COLLISION_CODE)
        else:
            collision_evidence_windows += 1
            collision_clearances.append(float(evidence.minimum_collision_clearance_m))
        maximum_collision_violations = max(
            maximum_collision_violations,
            int(evidence.maximum_collision_violating_pair_count),
        )
        collision_checks += int(window.configuration_plan_collision_check_count)
        if window.configuration_plan_collision_check_count < 1:
            _append(violations, R1_FAST_SCREEN_COLLISION_CODE)
        if not resolved.knots:
            _append(violations, R1_FAST_SCREEN_TRAJECTORY_CODE)
            continue
        resolved_knot_count += len(resolved.knots)
        first_q: dict[str, float] | None = None
        final_q: dict[str, float] | None = None
        window_commanded_joint_rate = 0.0
        for knot in resolved.knots:
            centroidal = knot.centroidal_target
            orientation = (
                None if centroidal is None else centroidal.body_orientation_world
            )
            if orientation is None:
                _append(violations, R1_FAST_SCREEN_TRAJECTORY_CODE)
            else:
                try:
                    body_tilt = _body_tilt_rad(orientation)
                except (TypeError, ValueError):
                    _append(violations, R1_FAST_SCREEN_TRAJECTORY_CODE)
                else:
                    maximum_body_tilt = max(maximum_body_tilt, body_tilt)
                    if body_tilt > settings.maximum_body_tilt_rad + 1.0e-9:
                        _append(violations, R1_FAST_SCREEN_BODY_TILT_CODE)
            posture = knot.posture_target
            q = None if posture is None else posture.joint_pos_target
            qdot = None if posture is None else posture.joint_vel_target
            if (
                q is None
                or qdot is None
                or set(q) != set(ordered_ids)
                or set(qdot) != set(ordered_ids)
            ):
                _append(violations, R1_FAST_SCREEN_TRAJECTORY_CODE)
                continue
            parsed_q = {joint_id: float(q[joint_id]) for joint_id in ordered_ids}
            parsed_qdot = {joint_id: float(qdot[joint_id]) for joint_id in ordered_ids}
            if not all(
                math.isfinite(value)
                for value in (*parsed_q.values(), *parsed_qdot.values())
            ):
                _append(violations, R1_FAST_SCREEN_TRAJECTORY_CODE)
                continue
            if first_q is None:
                first_q = dict(parsed_q)
            final_q = dict(parsed_q)
            for joint_id in ordered_ids:
                lower, upper = ordered_joint_limits_rad[joint_id]
                margin = min(
                    parsed_q[joint_id] - lower,
                    upper - parsed_q[joint_id],
                )
                fraction = margin / (upper - lower)
                minimum_limit_margin = min(minimum_limit_margin, margin)
                minimum_limit_fraction = min(minimum_limit_fraction, fraction)
                if (
                    margin < -settings.joint_limit_numerical_tolerance_rad
                    or fraction < settings.minimum_normalized_joint_limit_reserve
                ):
                    _append(violations, R1_FAST_SCREEN_JOINT_LIMIT_CODE)
            maximum_joint_rate = max(
                maximum_joint_rate,
                *(abs(value) for value in parsed_qdot.values()),
            )
            window_commanded_joint_rate = max(
                window_commanded_joint_rate,
                *(abs(value) for value in parsed_qdot.values()),
            )
        if previous_final_q is not None and first_q is not None:
            if (
                max(
                    abs(first_q[joint_id] - previous_final_q[joint_id])
                    for joint_id in ordered_ids
                )
                > settings.chain_joint_position_tolerance_rad
            ):
                _append(violations, R1_FAST_SCREEN_CHAIN_CODE)
        if final_q is not None:
            previous_final_q = final_q
        maximum_joint_rate = max(
            maximum_joint_rate,
            float(evidence.maximum_joint_rate_rad_s),
        )
        minimum_rate_margin = min(
            minimum_rate_margin,
            float(evidence.minimum_joint_rate_margin_rad_s),
            float(evidence.joint_rate_limit_rad_s) - window_commanded_joint_rate,
        )
        if (
            evidence.minimum_joint_rate_margin_rad_s < -1.0e-9
            or window_commanded_joint_rate > evidence.joint_rate_limit_rad_s + 1.0e-9
        ):
            _append(violations, R1_FAST_SCREEN_JOINT_RATE_CODE)

    last_window = None if not windows else windows[-1]
    final_maintained_contact_count = 0
    grasp_pose_reached = False
    if last_window is not None and last_window.resolved_trajectory.knots:
        last_knot = last_window.resolved_trajectory.knots[-1]
        maintained = {
            int(assignment.anchor_id)
            for assignment in last_knot.contact_assignments
            if assignment.schedule_state == "maintain"
        }
        final_maintained_contact_count = len(maintained)
        grasp_pose_reached = bool(
            last_window.phase == "contact_acquisition"
            and last_window.phase_target_reached
            and final_maintained_contact_count >= 2
        )
    if not grasp_pose_reached:
        _append(violations, R1_FAST_SCREEN_GRASP_POSE_CODE)

    if not math.isfinite(minimum_limit_margin):
        minimum_limit_margin = -1.0
        minimum_limit_fraction = -1.0
        _append(violations, R1_FAST_SCREEN_TRAJECTORY_CODE)
    if not math.isfinite(minimum_rate_margin):
        minimum_rate_margin = -1.0
        _append(violations, R1_FAST_SCREEN_TRAJECTORY_CODE)
    accepted = not violations
    result = Order9R1FastScreenResult(
        screen_version=ORDER9_R1_FAST_SCREEN_VERSION,
        execution_model=ORDER9_R1_FAST_SCREEN_EXECUTION_MODEL,
        candidate_id=candidate_id,
        candidate_task_hash=candidate_task_hash,
        morphology_hash=morphology_hash,
        physical_model_hash=physical_model_hash,
        resolved_path_hash=stable_hash(resolved_hashes),
        accepted=accepted,
        eligible_for_full_control_test=accepted,
        violation_codes=tuple(violations),
        checked_phases=checked_phases,
        window_count=len(windows),
        resolved_knot_count=resolved_knot_count,
        collision_evidence_window_count=collision_evidence_windows,
        configuration_plan_collision_check_count=collision_checks,
        grasp_pose_reached=grasp_pose_reached,
        final_maintained_contact_count=final_maintained_contact_count,
        minimum_joint_limit_margin_rad=minimum_limit_margin,
        minimum_normalized_joint_limit_reserve=minimum_limit_fraction,
        maximum_body_tilt_rad=maximum_body_tilt,
        maximum_joint_rate_rad_s=maximum_joint_rate,
        minimum_joint_rate_margin_rad_s=minimum_rate_margin,
        minimum_collision_clearance_m=(
            None if not collision_clearances else min(collision_clearances)
        ),
        maximum_collision_violating_pair_count=(maximum_collision_violations),
        screen_wall_time_s=perf_counter() - started,
    )
    return result


def require_order9_r1_fast_screen_pass(
    result: Order9R1FastScreenResult,
) -> None:
    """Fail closed before the caller can launch controller/Isaac replay."""

    if not result.accepted or not result.eligible_for_full_control_test:
        raise SchemaValidationError(
            "R1 candidate is not eligible for full-control testing: "
            + ",".join(result.violation_codes)
        )


def _screen_window(window: Order9C3NominalWindow) -> Order9R1FastScreenWindow:
    plan = window.plan
    configuration_plan = plan.configuration_space_plan
    return Order9R1FastScreenWindow(
        phase=str(window.phase),
        phase_target_reached=bool(plan.phase_target_reached),
        raw_trajectory=plan.raw_trajectory,
        resolved_trajectory=plan.trajectory,
        resolution_evidence=plan.posture_resolution.evidence,
        configuration_plan_collision_check_count=(
            0
            if configuration_plan is None
            else int(configuration_plan.collision_check_count)
        ),
    )


def _global_joint_limits(
    *,
    ordered_ids: Sequence[str],
    physical_model: PhysicalModel,
) -> dict[str, tuple[float, float]]:
    local = {joint.joint_id: joint for joint in physical_model.joints}
    limits: dict[str, tuple[float, float]] = {}
    for global_id in ordered_ids:
        joint = local.get(global_id.split(":", 1)[-1])
        if joint is None or joint.limit_lower is None or joint.limit_upper is None:
            raise SchemaValidationError(
                f"R1 fast screen lacks finite limits for {global_id!r}"
            )
        limits[global_id] = (
            float(joint.limit_lower),
            float(joint.limit_upper),
        )
    return limits


def _validate_joint_limits(
    limits: Mapping[str, tuple[float, float]],
) -> None:
    for joint_id, bounds in limits.items():
        if (
            not joint_id
            or len(bounds) != 2
            or not all(math.isfinite(float(value)) for value in bounds)
            or float(bounds[0]) >= float(bounds[1])
        ):
            raise SchemaValidationError(
                f"R1 fast-screen joint limits are invalid for {joint_id!r}"
            )


def _compressed_phases(
    windows: Sequence[Order9R1FastScreenWindow],
) -> tuple[str, ...]:
    result: list[str] = []
    for window in windows:
        if not result or result[-1] != window.phase:
            result.append(window.phase)
    return tuple(result)


def _body_tilt_rad(
    quaternion_xyzw: Sequence[float],
) -> float:
    if len(quaternion_xyzw) != 4:
        raise ValueError("body orientation must be a quaternion")
    x, y, z, w = (float(value) for value in quaternion_xyzw)
    norm_sq = x * x + y * y + z * z + w * w
    if not math.isfinite(norm_sq) or norm_sq <= 1.0e-12:
        raise ValueError("body orientation quaternion is invalid")
    body_up_world_z = 1.0 - 2.0 * (x * x + y * y) / norm_sq
    return math.acos(max(-1.0, min(1.0, body_up_world_z)))


def _append(values: list[str], code: str) -> None:
    if code not in values:
        values.append(code)


def _require_sha256(value: str, path: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise SchemaValidationError(f"{path} must be a lowercase SHA-256 digest")
