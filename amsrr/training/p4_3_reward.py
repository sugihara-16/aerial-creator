from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Mapping, Sequence

from amsrr.logging.episode_archive import EpisodeArchive
from amsrr.schemas.policies import ControllerCommand
from amsrr.schemas.runtime import ObjectRuntimeState, RuntimeObservation
from amsrr.schemas.task_spec import GoalSpec, TaskSpec


@dataclass(frozen=True)
class P4_3RewardConfig:
    """Configurable deterministic bootstrap reward for v0.4 Section 22.4.

    Every per-step term is bounded before weighting.  Energy is a normalized
    command-effort proxy because the current archive does not contain measured
    electrical power.  A missing measurement contributes a neutral zero and is
    identified by its ``*_data_available`` field in the reward record.
    """

    w_progress: float = 1.0
    w_pose: float = 1.0
    w_grasp: float = 1.0
    w_stable: float = 1.0
    w_energy: float = 1.0
    w_qp: float = 1.0
    w_slip: float = 1.0
    w_collision: float = 1.0
    w_saturation: float = 1.0
    success_bonus: float = 10.0
    failure_penalty: float = 10.0
    progress_scale_m: float = 0.05
    pose_position_scale_m: float = 0.25
    pose_rotation_scale_rad: float = 0.50
    centroidal_linear_speed_scale_mps: float = 1.0
    centroidal_angular_speed_scale_radps: float = 1.0
    rotor_thrust_scale_n: float = 20.0
    joint_torque_scale_nm: float = 5.0
    qp_residual_scale: float = 10.0
    slip_speed_scale_mps: float = 0.10
    default_goal_tolerance_pos_m: float = 0.05
    default_goal_tolerance_rot_rad: float = 0.20

    def __post_init__(self) -> None:
        non_negative = {
            "w_progress",
            "w_pose",
            "w_grasp",
            "w_stable",
            "w_energy",
            "w_qp",
            "w_slip",
            "w_collision",
            "w_saturation",
            "success_bonus",
            "failure_penalty",
            "default_goal_tolerance_pos_m",
            "default_goal_tolerance_rot_rad",
        }
        positive = {item.name for item in fields(self)} - non_negative
        for name in non_negative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"P4_3RewardConfig.{name} must be finite and non-negative")
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"P4_3RewardConfig.{name} must be finite and positive")


def compute_p4_3_step_reward(
    *,
    task_spec: TaskSpec,
    observation: RuntimeObservation,
    previous_observation: RuntimeObservation | None = None,
    controller_command: ControllerCommand | None = None,
    actuator_target_record: Mapping[str, Any] | None = None,
    state_transition_available: bool = True,
    config: P4_3RewardConfig | None = None,
) -> dict[str, float]:
    """Compute one deterministic Section 22.4 per-step reward record.

    The function only consumes commands already present in the rollout.  It
    never creates, fills, or modifies a ``ControllerCommand`` or actuator target.
    ``previous_observation`` and ``observation`` are respectively the pre- and
    post-command states.  Set ``state_transition_available`` false when a
    command was logged without its post-command observation; in that case all
    state-derived terms are explicitly neutral and only command-derived energy,
    QP residual, and saturation terms are evaluated.
    """

    cfg = config or P4_3RewardConfig()
    if state_transition_available:
        progress, progress_available = _goal_progress_term(
            task_spec,
            previous_observation,
            observation,
            cfg,
        )
        pose, pose_available = _pose_accuracy_term(task_spec, observation, cfg)
        grasp, grasp_available = _grasp_maintenance_term(observation)
        stability, stability_available = _centroidal_stability_term(observation, cfg)
        slip, slip_available = _slip_term(observation, cfg)
        collision, collision_available = _collision_term(observation)
    else:
        progress, progress_available = 0.0, False
        pose, pose_available = 0.0, False
        grasp, grasp_available = 0.0, False
        stability, stability_available = 0.0, False
        slip, slip_available = 0.0, False
        collision, collision_available = 0.0, False
    energy, energy_available = _energy_term(controller_command, actuator_target_record, cfg)
    qp_residual, qp_available = _qp_residual_term(controller_command, cfg)
    saturation, saturation_available = _saturation_term(controller_command, actuator_target_record)

    weighted_progress = cfg.w_progress * progress
    weighted_pose = cfg.w_pose * pose
    weighted_grasp = cfg.w_grasp * grasp
    weighted_stability = cfg.w_stable * stability
    weighted_energy = -cfg.w_energy * energy
    weighted_qp = -cfg.w_qp * qp_residual
    weighted_slip = -cfg.w_slip * slip
    weighted_collision = -cfg.w_collision * collision
    weighted_saturation = -cfg.w_saturation * saturation
    per_step_reward = (
        weighted_progress
        + weighted_pose
        + weighted_grasp
        + weighted_stability
        + weighted_energy
        + weighted_qp
        + weighted_slip
        + weighted_collision
        + weighted_saturation
    )

    return {
        "step_index": 0.0,
        "time_s": float(observation.time_s),
        "r_object_goal_progress": progress,
        "r_object_pose_accuracy": pose,
        "r_grasp_maintenance": grasp,
        "r_centroidal_stability": stability,
        "r_energy": energy,
        "r_qp_residual": qp_residual,
        "r_slip": slip,
        "r_collision": collision,
        "r_actuator_saturation": saturation,
        "weighted_object_goal_progress": weighted_progress,
        "weighted_object_pose_accuracy": weighted_pose,
        "weighted_grasp_maintenance": weighted_grasp,
        "weighted_centroidal_stability": weighted_stability,
        "weighted_energy_penalty": weighted_energy,
        "weighted_qp_residual_penalty": weighted_qp,
        "weighted_slip_penalty": weighted_slip,
        "weighted_collision_penalty": weighted_collision,
        "weighted_actuator_saturation_penalty": weighted_saturation,
        "object_goal_progress_data_available": _flag(progress_available),
        "object_pose_accuracy_data_available": _flag(pose_available),
        "grasp_data_available": _flag(grasp_available),
        "contact_data_available": _flag(grasp_available),
        "centroidal_stability_data_available": _flag(stability_available),
        "energy_data_available": _flag(energy_available),
        "qp_residual_data_available": _flag(qp_available),
        "slip_data_available": _flag(slip_available),
        "collision_data_available": _flag(collision_available),
        "actuator_saturation_data_available": _flag(saturation_available),
        "missing_contact_data": _flag(not grasp_available),
        "missing_slip_data": _flag(not slip_available),
        "energy_is_command_effort_proxy": _flag(energy_available),
        "state_transition_data_available": _flag(state_transition_available),
        "per_step_reward": per_step_reward,
        "terminal_reward_data_available": 0.0,
        "terminal_reward": 0.0,
        "reward": per_step_reward,
    }


def compute_p4_3_terminal_reward(
    *,
    task_spec: TaskSpec,
    observation: RuntimeObservation,
    release_valid: bool | None = None,
    object_dropped: bool | None = None,
    hard_collision: bool | None = None,
    timeout: bool | None = None,
    qp_infeasible_terminal: bool | None = None,
    config: P4_3RewardConfig | None = None,
) -> dict[str, float]:
    """Compute the Section 22.4 terminal bonus or one failure penalty.

    Success requires both an object pose within the TaskSpec tolerance and an
    explicitly valid release.  Failure signals dominate a simultaneous success
    signal, and multiple failure causes still incur one terminal penalty.
    """

    cfg = config or P4_3RewardConfig()
    within_tolerance, pose_available = _goal_pose_within_tolerance(task_spec, observation, cfg)
    inferred = _failure_flags_from_reason(observation.task_progress.failure_reason)
    drop = _combine_optional_signal(object_dropped, inferred["object_dropped"])
    collision = _combine_optional_signal(hard_collision, inferred["hard_collision"])
    timed_out = _combine_optional_signal(timeout, inferred["timeout"])
    qp_failed = _combine_optional_signal(qp_infeasible_terminal, inferred["qp_infeasible_terminal"])
    failure = any(value is True for value in (drop, collision, timed_out, qp_failed))
    success = bool(pose_available and within_tolerance and release_valid is True and not failure)
    terminal_reward = cfg.success_bonus if success else (-cfg.failure_penalty if failure else 0.0)
    return {
        "terminal_goal_pose_within_tolerance": _flag(within_tolerance),
        "terminal_goal_pose_data_available": _flag(pose_available),
        "terminal_release_valid": _flag(release_valid is True),
        "terminal_release_data_available": _flag(release_valid is not None),
        "terminal_object_dropped": _flag(drop is True),
        "terminal_object_drop_data_available": _flag(drop is not None),
        "terminal_hard_collision": _flag(collision is True),
        "terminal_hard_collision_data_available": _flag(collision is not None),
        "terminal_timeout": _flag(timed_out is True),
        "terminal_timeout_data_available": _flag(timed_out is not None),
        "terminal_qp_infeasible": _flag(qp_failed is True),
        "terminal_qp_infeasible_data_available": _flag(qp_failed is not None),
        "terminal_success": _flag(success),
        "terminal_failure": _flag(failure),
        "terminal_reward": terminal_reward,
    }


def compute_p4_3_reward_records(
    *,
    task_spec: TaskSpec,
    runtime_observations: Sequence[RuntimeObservation],
    controller_commands: Sequence[ControllerCommand] = (),
    actuator_target_records: Sequence[Mapping[str, Any]] = (),
    release_valid: bool | None = None,
    object_dropped: bool | None = None,
    hard_collision: bool | None = None,
    timeout: bool | None = None,
    qp_infeasible_terminal: bool | None = None,
    config: P4_3RewardConfig | None = None,
) -> list[dict[str, float]]:
    """Create command-causal reward records without changing inputs.

    The Isaac probe logs ``observation[i]`` before applying ``command[i]``.
    Therefore row ``i`` keeps the imitation pair ``observation[i]`` /
    ``command[i]``, while its state-derived reward uses the forward transition
    ``observation[i] -> observation[i + 1]``.  Command-derived energy, QP, and
    saturation terms always use command/actuator row ``i``.

    The final command has no logged post-observation.  Its row is retained for
    imitation, but has neutral unavailable state terms and command-only reward.
    When at least two observations exist, the terminal reward is attached to
    row ``N - 2`` because command ``N - 2`` causally produced observation
    ``N - 1``.  With only one observation no terminal transition is inferred.
    """

    cfg = config or P4_3RewardConfig()
    records: list[dict[str, float]] = []
    observation_count = len(runtime_observations)
    for index, pre_observation in enumerate(runtime_observations):
        post_available = index + 1 < observation_count
        post_observation = (
            runtime_observations[index + 1]
            if post_available
            else pre_observation
        )
        record = compute_p4_3_step_reward(
            task_spec=task_spec,
            observation=post_observation,
            previous_observation=pre_observation if post_available else None,
            controller_command=_sequence_item(controller_commands, index),
            actuator_target_record=_sequence_item(actuator_target_records, index),
            state_transition_available=post_available,
            config=cfg,
        )
        record["step_index"] = float(index)
        record["command_index"] = float(index)
        record["time_s"] = float(pre_observation.time_s)
        record["pre_observation_index"] = float(index)
        record["post_observation_index"] = float(index + 1 if post_available else -1)
        record["post_observation_data_available"] = _flag(post_available)
        record["transition_dt_s"] = (
            float(post_observation.time_s - pre_observation.time_s)
            if post_available
            else 0.0
        )
        records.append(record)
    if observation_count < 2:
        return records
    terminal_index = observation_count - 2
    terminal = compute_p4_3_terminal_reward(
        task_spec=task_spec,
        observation=runtime_observations[-1],
        release_valid=release_valid,
        object_dropped=object_dropped,
        hard_collision=hard_collision,
        timeout=timeout,
        qp_infeasible_terminal=qp_infeasible_terminal,
        config=cfg,
    )
    records[terminal_index].update(terminal)
    records[terminal_index]["terminal_reward_data_available"] = 1.0
    records[terminal_index]["reward"] = (
        records[terminal_index]["per_step_reward"]
        + records[terminal_index]["terminal_reward"]
    )
    return records


def compute_p4_3_archive_rewards(
    archive: EpisodeArchive,
    *,
    config: P4_3RewardConfig | None = None,
) -> list[dict[str, float]]:
    """Derive aligned rewards from an existing EpisodeArchive without mutation."""

    return compute_p4_3_reward_records(
        task_spec=archive.task_spec,
        runtime_observations=archive.runtime_observations,
        controller_commands=archive.controller_commands,
        actuator_target_records=archive.actuator_target_records,
        release_valid=_archive_release_valid(archive),
        object_dropped=_archive_metric_bool(archive, "object_drop"),
        hard_collision=_archive_metric_bool(archive, "hard_collision"),
        timeout=_archive_metric_bool(archive, "timeout_failure"),
        qp_infeasible_terminal=_archive_metric_bool(archive, "controller_qp_infeasible_terminal"),
        config=config,
    )


def _goal_progress_term(
    task_spec: TaskSpec,
    previous: RuntimeObservation | None,
    current: RuntimeObservation,
    config: P4_3RewardConfig,
) -> tuple[float, bool]:
    goal = _object_pose_goal(task_spec)
    if goal is None or goal.target_pose_world is None:
        return 0.0, False
    current_state = _object_state(current, goal.target_entity_id)
    previous_pose = None
    if previous is not None:
        previous_state = _object_state(previous, goal.target_entity_id)
        previous_pose = previous_state.pose_world if previous_state is not None else None
    else:
        previous_pose = _initial_object_pose(task_spec, goal.target_entity_id)
    if current_state is None or previous_pose is None:
        return 0.0, False
    previous_distance = _position_distance(previous_pose, goal.target_pose_world)
    current_distance = _position_distance(current_state.pose_world, goal.target_pose_world)
    return _clamp((previous_distance - current_distance) / config.progress_scale_m, -1.0, 1.0), True


def _pose_accuracy_term(
    task_spec: TaskSpec,
    observation: RuntimeObservation,
    config: P4_3RewardConfig,
) -> tuple[float, bool]:
    errors = _goal_pose_errors(task_spec, observation)
    if errors is None:
        return 0.0, False
    position_error, rotation_error = errors
    position_score = 1.0 / (1.0 + position_error / config.pose_position_scale_m)
    rotation_score = 1.0 / (1.0 + rotation_error / config.pose_rotation_scale_rad)
    return 0.5 * (position_score + rotation_score), True


def _grasp_maintenance_term(observation: RuntimeObservation) -> tuple[float, bool]:
    metrics = observation.task_progress.metrics
    available = _availability(metrics, ("grasp_data_available", "contact_data_available"))
    if available is False:
        return 0.0, False
    explicit = _metric_value(metrics, ("grasp_maintenance", "grasp_maintained", "attached"))
    if explicit is not None:
        return _clamp01(explicit), True
    if observation.contact_states:
        active = sum(1 for contact in observation.contact_states if contact.active)
        return float(active) / float(len(observation.contact_states)), True
    if available is True:
        return 0.0, True
    return 0.0, False


def _centroidal_stability_term(
    observation: RuntimeObservation,
    config: P4_3RewardConfig,
) -> tuple[float, bool]:
    explicit = _metric_value(
        observation.task_progress.metrics,
        ("centroidal_stability", "centroidal_stability_score"),
    )
    if explicit is not None:
        return _clamp01(explicit), True
    twists = [state.twist_world for state in observation.module_states if _finite_vector(state.twist_world, 6)]
    if not twists:
        return 0.0, False
    mean = [sum(twist[index] for twist in twists) / len(twists) for index in range(6)]
    linear_deviation = math.sqrt(
        sum(sum((twist[index] - mean[index]) ** 2 for index in range(3)) for twist in twists) / len(twists)
    )
    angular_deviation = math.sqrt(
        sum(sum((twist[index] - mean[index]) ** 2 for index in range(3, 6)) for twist in twists) / len(twists)
    )
    score = 1.0 / (
        1.0
        + linear_deviation / config.centroidal_linear_speed_scale_mps
        + angular_deviation / config.centroidal_angular_speed_scale_radps
    )
    return score, True


def _energy_term(
    command: ControllerCommand | None,
    actuator_record: Mapping[str, Any] | None,
    config: P4_3RewardConfig,
) -> tuple[float, bool]:
    normalized_squared: list[float] = []
    if command is not None:
        normalized_squared.extend(
            (value / config.rotor_thrust_scale_n) ** 2
            for value in _finite_values(command.rotor_thrusts_n.values())
        )
        normalized_squared.extend(
            (value / config.joint_torque_scale_nm) ** 2
            for value in _finite_values(command.joint_torque_commands.values())
        )
        if not normalized_squared:
            return 0.0, True
    elif actuator_record is not None and isinstance(actuator_record.get("actuator_targets"), list):
        for target in actuator_record["actuator_targets"]:
            if not isinstance(target, Mapping):
                continue
            value = _finite_float(target.get("target_value"))
            if value is None:
                continue
            actuator_type = str(target.get("actuator_type", ""))
            if actuator_type == "rotor_thrust":
                normalized_squared.append((value / config.rotor_thrust_scale_n) ** 2)
            elif actuator_type == "joint_torque":
                normalized_squared.append((value / config.joint_torque_scale_nm) ** 2)
        if not normalized_squared:
            return 0.0, True
    else:
        return 0.0, False
    return _clamp01(sum(normalized_squared) / len(normalized_squared)), True


def _qp_residual_term(
    command: ControllerCommand | None,
    config: P4_3RewardConfig,
) -> tuple[float, bool]:
    if command is None:
        return 0.0, False
    status = command.controller_status
    residual = _metric_value(
        status.metrics,
        ("qp_residual", "allocation_residual_norm", "residual_norm", "force_residual_norm"),
    )
    if residual is None:
        return 0.0, False
    return _clamp01(abs(residual) / config.qp_residual_scale), True


def _slip_term(observation: RuntimeObservation, config: P4_3RewardConfig) -> tuple[float, bool]:
    metrics = observation.task_progress.metrics
    available = _availability(metrics, ("slip_data_available",))
    if available is False:
        return 0.0, False
    direct = _metric_item(metrics, ("slip", "slip_ratio", "slip_mps", "slip_speed_mps"))
    values: list[float] = []
    if direct is not None:
        values.append(_normalize_slip(*direct, config))
    metadata_available = False
    for contact in observation.contact_states:
        hint = _availability(contact.metadata, ("slip_data_available",))
        metadata_available = metadata_available or hint is True
        item = _metric_item(contact.metadata, ("slip", "slip_ratio", "slip_mps", "slip_speed_mps"))
        if item is not None:
            values.append(_normalize_slip(*item, config))
    if values:
        return max(values), True
    if available is True or metadata_available:
        return 0.0, True
    return 0.0, False


def _collision_term(observation: RuntimeObservation) -> tuple[float, bool]:
    metrics = observation.task_progress.metrics
    available = _availability(metrics, ("collision_data_available",))
    if available is False:
        return 0.0, False
    collision = _metric_value(metrics, ("hard_collision", "collision"))
    if collision is not None:
        return _clamp01(collision), True
    for contact in observation.contact_states:
        value = _metric_value(contact.metadata, ("hard_collision",))
        if value is not None:
            return _clamp01(value), True
    reason = (observation.task_progress.failure_reason or "").lower()
    if "collision" in reason:
        return 1.0, True
    if available is True:
        return 0.0, True
    return 0.0, False


def _saturation_term(
    command: ControllerCommand | None,
    actuator_record: Mapping[str, Any] | None,
) -> tuple[float, bool]:
    candidates: list[float] = []
    if command is not None:
        value = _metric_value(
            command.controller_status.metrics,
            ("actuator_saturation_ratio", "rotor_saturation_ratio", "saturation_ratio", "clipped"),
        )
        if value is not None:
            candidates.append(_clamp01(value))
    if actuator_record is not None:
        targets = actuator_record.get("actuator_targets")
        if isinstance(targets, list):
            valid_targets = [target for target in targets if isinstance(target, Mapping)]
            if valid_targets:
                candidates.append(
                    sum(1 for target in valid_targets if bool(target.get("clipped", False))) / len(valid_targets)
                )
            else:
                candidates.append(0.0)
        metrics = actuator_record.get("metrics")
        if isinstance(metrics, Mapping):
            clipped = _metric_value(metrics, ("clipped_target_count",))
            total = _metric_value(metrics, ("actuator_target_count",))
            if clipped is not None and total is not None and total > 0.0:
                candidates.append(_clamp01(clipped / total))
    if not candidates:
        return 0.0, False
    return max(candidates), True


def _goal_pose_within_tolerance(
    task_spec: TaskSpec,
    observation: RuntimeObservation,
    config: P4_3RewardConfig,
) -> tuple[bool, bool]:
    goal = _object_pose_goal(task_spec)
    errors = _goal_pose_errors(task_spec, observation)
    if goal is None or errors is None:
        return False, False
    position_error, rotation_error = errors
    position_tolerance = (
        config.default_goal_tolerance_pos_m if goal.tolerance_pos_m is None else goal.tolerance_pos_m
    )
    rotation_tolerance = (
        config.default_goal_tolerance_rot_rad if goal.tolerance_rot_rad is None else goal.tolerance_rot_rad
    )
    return position_error <= position_tolerance and rotation_error <= rotation_tolerance, True


def _goal_pose_errors(
    task_spec: TaskSpec,
    observation: RuntimeObservation,
) -> tuple[float, float] | None:
    goal = _object_pose_goal(task_spec)
    if goal is None or goal.target_pose_world is None:
        return None
    state = _object_state(observation, goal.target_entity_id)
    if state is None:
        return None
    return (
        _position_distance(state.pose_world, goal.target_pose_world),
        _quaternion_distance(state.pose_world[3:7], goal.target_pose_world[3:7]),
    )


def _object_pose_goal(task_spec: TaskSpec) -> GoalSpec | None:
    for goal in task_spec.goals:
        if goal.goal_type == "object_pose" and goal.target_pose_world is not None:
            return goal
    return None


def _object_state(observation: RuntimeObservation, object_id: str | None) -> ObjectRuntimeState | None:
    if object_id is not None:
        return next((state for state in observation.object_states if state.object_id == object_id), None)
    return observation.object_states[0] if len(observation.object_states) == 1 else None


def _initial_object_pose(task_spec: TaskSpec, object_id: str | None) -> Sequence[float] | None:
    if object_id is not None:
        return next((obj.pose_world for obj in task_spec.scene.objects if obj.object_id == object_id), None)
    return task_spec.scene.objects[0].pose_world if len(task_spec.scene.objects) == 1 else None


def _position_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(left[index]) - float(right[index])) ** 2 for index in range(3)))


def _quaternion_distance(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return math.pi
    dot = abs(sum(float(a) * float(b) for a, b in zip(left, right)) / (left_norm * right_norm))
    return 2.0 * math.acos(_clamp(dot, -1.0, 1.0))


def _normalize_slip(name: str, value: float, config: P4_3RewardConfig) -> float:
    if name in {"slip_mps", "slip_speed_mps"}:
        return _clamp01(abs(value) / config.slip_speed_scale_mps)
    return _clamp01(value)


def _failure_flags_from_reason(reason: str | None) -> dict[str, bool | None]:
    if not reason:
        return {
            "object_dropped": None,
            "hard_collision": None,
            "timeout": None,
            "qp_infeasible_terminal": None,
        }
    lowered = reason.lower()
    return {
        "object_dropped": True if "drop" in lowered else None,
        "hard_collision": True if "collision" in lowered else None,
        "timeout": True if "timeout" in lowered else None,
        "qp_infeasible_terminal": True
        if "controller" in lowered or "qp_infeasible" in lowered or "qp infeasible" in lowered
        else None,
    }


def _combine_optional_signal(explicit: bool | None, inferred: bool | None) -> bool | None:
    if explicit is True or inferred is True:
        return True
    if explicit is False:
        return False
    return inferred


def _archive_release_valid(archive: EpisodeArchive) -> bool | None:
    explicit = _metric_value(archive.metrics, ("release_valid",))
    if explicit is not None:
        return explicit > 0.0
    for key in ("p4_2_release_events", "release_events"):
        if key not in archive.rollout_artifacts:
            continue
        events = archive.rollout_artifacts[key]
        if not isinstance(events, list):
            return None
        return any(isinstance(event, Mapping) and bool(event.get("intended_release", False)) for event in events)
    return None


def _archive_metric_bool(archive: EpisodeArchive, name: str) -> bool | None:
    value = _finite_float(archive.metrics.get(name))
    return None if value is None else value > 0.0


def _sequence_item(items: Sequence[Any], index: int) -> Any | None:
    return items[index] if index < len(items) else None


def _availability(metrics: Mapping[str, Any], keys: Sequence[str]) -> bool | None:
    value = _metric_value(metrics, keys)
    return None if value is None else value > 0.0


def _metric_item(metrics: Mapping[str, Any], keys: Sequence[str]) -> tuple[str, float] | None:
    for key in keys:
        value = _finite_float(metrics.get(key))
        if value is not None:
            return key, value
    return None


def _metric_value(metrics: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    item = _metric_item(metrics, keys)
    return None if item is None else item[1]


def _finite_values(values: Sequence[float] | Any) -> list[float]:
    result: list[float] = []
    for value in values:
        converted = _finite_float(value)
        if converted is not None:
            result.append(abs(converted))
    return result


def _finite_vector(values: Sequence[float], expected_length: int) -> bool:
    return len(values) == expected_length and all(_finite_float(value) is not None for value in values)


def _finite_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


def _flag(value: bool) -> float:
    return 1.0 if value else 0.0
