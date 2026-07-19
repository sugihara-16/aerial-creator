from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence


DOCK_JOINT_DRIVE_TUNING_VERSION = "dock_joint_drive_tuning_v1"
DOCK_JOINT_DRIVE_TUNING_METHOD = (
    "single_fixed_base_module_coarse_to_fine_time_domain_search_v1"
)
DOCK_JOINT_DRIVE_TUNING_SELECTION_SCOPE = "contact_free_bench_candidate_only"
DOCK_JOINT_DRIVE_TUNING_DEPLOYMENT_GATE = (
    "separate_representative_contact_task_validation_required"
)


@dataclass(frozen=True)
class DockJointDriveTuningConfig:
    simulation_dt_s: float = 0.02
    reset_settle_s: float = 0.10
    step_hold_s: float = 0.40
    return_hold_s: float = 0.40
    disturbance_hold_s: float = 0.40
    recovery_hold_s: float = 0.60
    step_amplitude_rad: float = 0.01
    disturbance_torque_nm: float = 1.20
    settling_position_tolerance_rad: float = 5.0e-4
    settling_velocity_tolerance_rad_s: float = 0.01
    effort_limit_nm: float = 4.10
    peak_current_a: float = 7.30
    velocity_limit_rad_s: float = 3.0
    coarse_kp_values: tuple[float, ...] = (
        75.0,
        100.0,
        150.0,
        200.0,
        250.0,
        300.0,
        400.0,
        500.0,
        650.0,
    )
    coarse_kd_values: tuple[float, ...] = (
        1.0,
        2.0,
        3.5,
        5.0,
        8.0,
        12.0,
        20.0,
        30.0,
    )
    fine_multipliers: tuple[float, ...] = (0.75, 0.875, 1.0, 1.125, 1.25)
    minimum_kp: float = 25.0
    maximum_kp: float = 800.0
    minimum_kd: float = 0.25
    maximum_kd: float = 40.0

    def validate(self) -> None:
        positive_scalars = {
            "simulation_dt_s": self.simulation_dt_s,
            "reset_settle_s": self.reset_settle_s,
            "step_hold_s": self.step_hold_s,
            "return_hold_s": self.return_hold_s,
            "disturbance_hold_s": self.disturbance_hold_s,
            "recovery_hold_s": self.recovery_hold_s,
            "step_amplitude_rad": self.step_amplitude_rad,
            "disturbance_torque_nm": self.disturbance_torque_nm,
            "settling_position_tolerance_rad": self.settling_position_tolerance_rad,
            "settling_velocity_tolerance_rad_s": self.settling_velocity_tolerance_rad_s,
            "effort_limit_nm": self.effort_limit_nm,
            "peak_current_a": self.peak_current_a,
            "velocity_limit_rad_s": self.velocity_limit_rad_s,
            "minimum_kp": self.minimum_kp,
            "maximum_kp": self.maximum_kp,
            "minimum_kd": self.minimum_kd,
            "maximum_kd": self.maximum_kd,
        }
        for name, value in positive_scalars.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.minimum_kp > self.maximum_kp:
            raise ValueError("minimum_kp must not exceed maximum_kp")
        if self.minimum_kd > self.maximum_kd:
            raise ValueError("minimum_kd must not exceed maximum_kd")
        for name, values in (
            ("coarse_kp_values", self.coarse_kp_values),
            ("coarse_kd_values", self.coarse_kd_values),
            ("fine_multipliers", self.fine_multipliers),
        ):
            if not values:
                raise ValueError(f"{name} must not be empty")
            if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in values):
                raise ValueError(f"{name} values must be finite and positive")
        if self.disturbance_torque_nm > self.effort_limit_nm:
            raise ValueError("disturbance torque must not exceed the actuator effort limit")

    def phase_steps(self) -> dict[str, int]:
        self.validate()
        return {
            "reset_settle": _duration_steps(self.reset_settle_s, self.simulation_dt_s),
            "step": _duration_steps(self.step_hold_s, self.simulation_dt_s),
            "return": _duration_steps(self.return_hold_s, self.simulation_dt_s),
            "disturbance": _duration_steps(
                self.disturbance_hold_s, self.simulation_dt_s
            ),
            "recovery": _duration_steps(self.recovery_hold_s, self.simulation_dt_s),
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "simulation_dt_s": self.simulation_dt_s,
            "reset_settle_s": self.reset_settle_s,
            "step_hold_s": self.step_hold_s,
            "return_hold_s": self.return_hold_s,
            "disturbance_hold_s": self.disturbance_hold_s,
            "recovery_hold_s": self.recovery_hold_s,
            "step_amplitude_rad": self.step_amplitude_rad,
            "disturbance_torque_nm": self.disturbance_torque_nm,
            "settling_position_tolerance_rad": self.settling_position_tolerance_rad,
            "settling_velocity_tolerance_rad_s": (
                self.settling_velocity_tolerance_rad_s
            ),
            "effort_limit_nm": self.effort_limit_nm,
            "peak_current_a": self.peak_current_a,
            "velocity_limit_rad_s": self.velocity_limit_rad_s,
            "coarse_kp_values": list(self.coarse_kp_values),
            "coarse_kd_values": list(self.coarse_kd_values),
            "fine_multipliers": list(self.fine_multipliers),
            "minimum_kp": self.minimum_kp,
            "maximum_kp": self.maximum_kp,
            "minimum_kd": self.minimum_kd,
            "maximum_kd": self.maximum_kd,
            "phase_steps": self.phase_steps(),
        }


@dataclass(frozen=True)
class DockJointDriveSample:
    phase: str
    phase_time_s: float
    position_rad_by_joint: Mapping[str, float]
    velocity_rad_s_by_joint: Mapping[str, float]
    target_rad_by_joint: Mapping[str, float]
    applied_torque_nm_by_joint: Mapping[str, float]


def coarse_gain_candidates(
    config: DockJointDriveTuningConfig,
) -> tuple[tuple[float, float], ...]:
    config.validate()
    return _unique_candidates(
        (kp, kd)
        for kp in config.coarse_kp_values
        for kd in config.coarse_kd_values
        if config.minimum_kp <= kp <= config.maximum_kp
        and config.minimum_kd <= kd <= config.maximum_kd
    )


def fine_gain_candidates(
    config: DockJointDriveTuningConfig,
    *,
    center_kp: float,
    center_kd: float,
    excluded: Iterable[tuple[float, float]] = (),
) -> tuple[tuple[float, float], ...]:
    config.validate()
    excluded_keys = {_candidate_key(kp, kd) for kp, kd in excluded}
    return tuple(
        candidate
        for candidate in _unique_candidates(
            (
                min(
                    max(float(center_kp) * kp_multiplier, config.minimum_kp),
                    config.maximum_kp,
                ),
                min(
                    max(float(center_kd) * kd_multiplier, config.minimum_kd),
                    config.maximum_kd,
                ),
            )
            for kp_multiplier in config.fine_multipliers
            for kd_multiplier in config.fine_multipliers
        )
        if _candidate_key(*candidate) not in excluded_keys
    )


def evaluate_gain_candidate(
    *,
    kp: float,
    kd: float,
    joint_names: Sequence[str],
    samples: Sequence[DockJointDriveSample],
    config: DockJointDriveTuningConfig,
) -> dict[str, object]:
    config.validate()
    if not joint_names:
        raise ValueError("joint_names must not be empty")
    if not samples:
        raise ValueError("samples must not be empty")
    expected = set(joint_names)
    for sample in samples:
        for field_name, values in (
            ("position", sample.position_rad_by_joint),
            ("velocity", sample.velocity_rad_s_by_joint),
            ("target", sample.target_rad_by_joint),
            ("torque", sample.applied_torque_nm_by_joint),
        ):
            if set(values) != expected:
                raise ValueError(
                    f"sample {field_name} keys must exactly match joint_names"
                )

    per_joint = {
        joint_name: _evaluate_joint_trace(
            joint_name=joint_name,
            samples=samples,
            config=config,
        )
        for joint_name in joint_names
    }
    maximum_speed = max(
        float(metrics["maximum_speed_rad_s"]) for metrics in per_joint.values()
    )
    maximum_torque = max(
        float(metrics["maximum_applied_torque_nm"])
        for metrics in per_joint.values()
    )
    maximum_current = maximum_torque / config.effort_limit_nm * config.peak_current_a
    finite = all(
        math.isfinite(float(value))
        for metrics in per_joint.values()
        for key, value in metrics.items()
        if key != "joint_name"
    )
    feasible = bool(
        finite
        and maximum_speed <= config.velocity_limit_rad_s + 1.0e-6
        and maximum_torque <= config.effort_limit_nm + 1.0e-5
        and maximum_current <= config.peak_current_a + 1.0e-5
    )
    joint_scores = [float(metrics["score"]) for metrics in per_joint.values()]
    aggregate_score = (
        0.65 * max(joint_scores) + 0.35 * sum(joint_scores) / len(joint_scores)
    )
    if not feasible:
        aggregate_score += 1.0e6
    return {
        "kp_nm_per_rad": float(kp),
        "kd_nms_per_rad": float(kd),
        "feasible": feasible,
        "score": aggregate_score,
        "maximum_speed_rad_s": maximum_speed,
        "maximum_applied_torque_nm": maximum_torque,
        "maximum_estimated_current_a": maximum_current,
        "per_joint": per_joint,
    }


def select_best_gain_candidate(
    results: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    feasible = [result for result in results if bool(result.get("feasible"))]
    if not feasible:
        raise ValueError("no feasible Dock joint drive gain candidate")
    return min(
        feasible,
        key=lambda result: (
            float(result["score"]),
            float(result["maximum_speed_rad_s"]),
            float(result["maximum_applied_torque_nm"]),
            float(result["kp_nm_per_rad"]),
            float(result["kd_nms_per_rad"]),
        ),
    )


def _evaluate_joint_trace(
    *,
    joint_name: str,
    samples: Sequence[DockJointDriveSample],
    config: DockJointDriveTuningConfig,
) -> dict[str, float | str]:
    by_phase = {
        phase: [sample for sample in samples if sample.phase == phase]
        for phase in ("step", "return", "disturbance", "recovery")
    }
    if any(not phase_samples for phase_samples in by_phase.values()):
        raise ValueError("trace must contain step, return, disturbance, and recovery")
    step_samples = by_phase["step"]
    tracking_samples = [*step_samples, *by_phase["return"], *by_phase["recovery"]]
    normalized_iae = (
        sum(
            abs(
                float(sample.target_rad_by_joint[joint_name])
                - float(sample.position_rad_by_joint[joint_name])
            )
            * config.simulation_dt_s
            for sample in tracking_samples
        )
        / (
            config.step_amplitude_rad
            * config.simulation_dt_s
            * len(tracking_samples)
        )
    )
    first_target = float(step_samples[0].target_rad_by_joint[joint_name])
    target_sign = 1.0 if first_target >= 0.0 else -1.0
    maximum_projected_step_position = max(
        target_sign * float(sample.position_rad_by_joint[joint_name])
        for sample in step_samples
    )
    overshoot_ratio = max(
        0.0,
        maximum_projected_step_position / config.step_amplitude_rad - 1.0,
    )
    step_settling_s = _settling_time(
        step_samples,
        joint_name=joint_name,
        position_tolerance=config.settling_position_tolerance_rad,
        velocity_tolerance=config.settling_velocity_tolerance_rad_s,
    )
    return_settling_s = _settling_time(
        by_phase["return"],
        joint_name=joint_name,
        position_tolerance=config.settling_position_tolerance_rad,
        velocity_tolerance=config.settling_velocity_tolerance_rad_s,
    )
    recovery_settling_s = _settling_time(
        by_phase["recovery"],
        joint_name=joint_name,
        position_tolerance=config.settling_position_tolerance_rad,
        velocity_tolerance=config.settling_velocity_tolerance_rad_s,
    )
    disturbance_peak = max(
        abs(float(sample.position_rad_by_joint[joint_name]))
        for sample in by_phase["disturbance"]
    )
    recovery_terminal_error = abs(
        float(by_phase["recovery"][-1].position_rad_by_joint[joint_name])
    )
    all_positions = [
        abs(float(sample.position_rad_by_joint[joint_name])) for sample in samples
    ]
    all_velocities = [
        abs(float(sample.velocity_rad_s_by_joint[joint_name])) for sample in samples
    ]
    all_torques = [
        abs(float(sample.applied_torque_nm_by_joint[joint_name]))
        for sample in samples
    ]
    saturation_fraction = sum(
        torque >= 0.98 * config.effort_limit_nm for torque in all_torques
    ) / len(all_torques)
    score = (
        2.0 * normalized_iae
        + 2.5 * overshoot_ratio
        + 0.75 * step_settling_s / config.step_hold_s
        + 0.50 * return_settling_s / config.return_hold_s
        + 0.75 * recovery_settling_s / config.recovery_hold_s
        + 1.5 * disturbance_peak / config.step_amplitude_rad
        + 1.5 * recovery_terminal_error / config.step_amplitude_rad
        + 0.25 * max(all_velocities) / config.velocity_limit_rad_s
        + 0.10 * max(all_torques) / config.effort_limit_nm
        + 1.0 * saturation_fraction
    )
    return {
        "joint_name": joint_name,
        "score": score,
        "normalized_integral_absolute_tracking_error": normalized_iae,
        "step_overshoot_ratio": overshoot_ratio,
        "step_settling_s": step_settling_s,
        "return_settling_s": return_settling_s,
        "disturbance_peak_deflection_rad": disturbance_peak,
        "recovery_settling_s": recovery_settling_s,
        "recovery_terminal_error_rad": recovery_terminal_error,
        "maximum_absolute_position_rad": max(all_positions),
        "maximum_speed_rad_s": max(all_velocities),
        "maximum_applied_torque_nm": max(all_torques),
        "torque_saturation_fraction": saturation_fraction,
    }


def _settling_time(
    samples: Sequence[DockJointDriveSample],
    *,
    joint_name: str,
    position_tolerance: float,
    velocity_tolerance: float,
) -> float:
    last_outside_index = -1
    for index, sample in enumerate(samples):
        error = abs(
            float(sample.target_rad_by_joint[joint_name])
            - float(sample.position_rad_by_joint[joint_name])
        )
        speed = abs(float(sample.velocity_rad_s_by_joint[joint_name]))
        if error > position_tolerance or speed > velocity_tolerance:
            last_outside_index = index
    if last_outside_index == len(samples) - 1:
        return float(samples[-1].phase_time_s)
    if last_outside_index < 0:
        return 0.0
    return float(samples[last_outside_index + 1].phase_time_s)


def _duration_steps(duration_s: float, dt_s: float) -> int:
    return max(1, int(math.ceil(float(duration_s) / float(dt_s) - 1.0e-12)))


def _candidate_key(kp: float, kd: float) -> tuple[float, float]:
    return (round(float(kp), 9), round(float(kd), 9))


def _unique_candidates(
    candidates: Iterable[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    result: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for kp, kd in candidates:
        key = _candidate_key(kp, kd)
        if key in seen:
            continue
        seen.add(key)
        result.append((float(kp), float(kd)))
    return tuple(result)
