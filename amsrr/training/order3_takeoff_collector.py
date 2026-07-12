from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from amsrr.controllers.isaac_controller_bridge import IsaacActuatorTargetRecord
from amsrr.controllers.rigid_body_model import RigidBodyControlModelBuilder
from amsrr.morphology.random_connected import morphology_structural_hash
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import (
    SchemaBase,
    SchemaValidationError,
    require_len,
    require_non_empty,
)
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.order3 import ORDER3_ACTION_SIZE, Order3PolicyTransition
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import (
    POLICY_COMMAND_CONTRACT_CENTROIDAL,
    ControllerCommand,
    PolicyCommand,
)
from amsrr.schemas.runtime import RuntimeObservation
from amsrr.simulation.random_morphology_takeoff import (
    RandomMorphologyTakeoffResult,
)
from amsrr.training.order3_free_flight import (
    TRUE_CENTROIDAL_TRACKING_SOURCE,
    Order3FreeFlightRewardConfig,
    Order3FreeFlightStep,
    Order3TaskMode,
    compute_order3_free_flight_reward,
)
from amsrr.utils.hashing import stable_hash


ORDER3_TAKEOFF_COLLECTOR_VERSION = "order3_takeoff_bc_collector_v1"
TAKEOFF_RUNTIME_OBSERVATIONS = "random_morphology_takeoff_runtime_observations"
TAKEOFF_POLICY_COMMANDS = "random_morphology_takeoff_policy_commands"
TAKEOFF_CONTROLLER_COMMANDS = "random_morphology_takeoff_controller_commands"
TAKEOFF_ACTUATOR_RECORDS = "random_morphology_takeoff_actuator_target_records"
TAKEOFF_CONTROL_POSE_HISTORY = "random_morphology_takeoff_control_pose_history"

_LEARNED_ACTION_TRACE_KEYS = (
    "random_morphology_takeoff_learned_action_trace",
    "random_morphology_takeoff_policy_action_trace",
    "order3_learned_action_trace",
    "order3_pi_l_transition_traces",
)
_FORBIDDEN_CONTACT_METADATA_KEYS = {
    "contact_wrench",
    "contact_wrench_world",
    "ground_truth_contact_wrench",
    "privileged_contact_wrench",
}


@dataclass
class Order3TakeoffBCCollectorConfig(SchemaBase):
    physical_model_config_path: str = "configs/robot/robot_model.yaml"
    recurrent_state_dim: int = 64
    sanitize_privileged_contact_wrench: bool = True
    control_pose_position_tolerance_m: float = 1.0e-6
    control_pose_attitude_tolerance_rad: float = 1.0e-6
    reward_config: Order3FreeFlightRewardConfig = field(
        default_factory=Order3FreeFlightRewardConfig
    )

    def validate(self) -> None:
        require_non_empty(
            self.physical_model_config_path,
            "Order3TakeoffBCCollectorConfig.physical_model_config_path",
        )
        if self.recurrent_state_dim < 1:
            raise SchemaValidationError(
                "Order3TakeoffBCCollectorConfig.recurrent_state_dim must be positive"
            )
        for name in (
            "control_pose_position_tolerance_m",
            "control_pose_attitude_tolerance_rad",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise SchemaValidationError(
                    f"Order3TakeoffBCCollectorConfig.{name} must be finite and positive"
                )


@dataclass
class Order3TakeoffBCCollectionResult(SchemaBase):
    collector_version: str
    episode_id: str
    split: DatasetSplit
    structural_hash: str
    transitions: list[Order3PolicyTransition]
    removed_privileged_contact_wrench_count: int
    source_is_real_isaac: bool
    learned_action_trace_available: bool
    online_ppo_rollout_eligible: bool
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def validate(self) -> None:
        if self.collector_version != ORDER3_TAKEOFF_COLLECTOR_VERSION:
            raise SchemaValidationError(
                "Order3TakeoffBCCollectionResult collector version mismatch"
            )
        require_non_empty(self.episode_id, "Order3TakeoffBCCollectionResult.episode_id")
        require_non_empty(
            self.structural_hash,
            "Order3TakeoffBCCollectionResult.structural_hash",
        )
        if not self.transitions:
            raise SchemaValidationError(
                "Order3TakeoffBCCollectionResult.transitions must not be empty"
            )
        if self.removed_privileged_contact_wrench_count < 0:
            raise SchemaValidationError(
                "removed privileged contact wrench count must be non-negative"
            )
        if not self.source_is_real_isaac:
            raise SchemaValidationError("Order3 takeoff BC source must be real Isaac")
        if self.learned_action_trace_available or self.online_ppo_rollout_eligible:
            raise SchemaValidationError(
                "Order3 takeoff BC collection cannot claim a learned online action trace"
            )
        for index, transition in enumerate(self.transitions):
            if transition.episode_id != self.episode_id:
                raise SchemaValidationError("Order3 takeoff transitions cross episode ids")
            if transition.split != self.split:
                raise SchemaValidationError("Order3 takeoff transitions cross splits")
            if transition.structural_hash != self.structural_hash:
                raise SchemaValidationError(
                    "Order3 takeoff transitions cross structural hashes"
                )
            if transition.step_index != index:
                raise SchemaValidationError(
                    "Order3 takeoff transition indices must be contiguous"
                )
        terminal_indices = [
            index for index, transition in enumerate(self.transitions) if transition.terminal
        ]
        if terminal_indices != [len(self.transitions) - 1]:
            raise SchemaValidationError(
                "Order3 takeoff BC terminal must be the final transition"
            )


def collect_order3_takeoff_bc_transitions(
    source: RandomMorphologyTakeoffResult | dict[str, Any],
    *,
    split: DatasetSplit,
    expected_structural_hash: str | None = None,
    episode_id: str | None = None,
    physical_model: PhysicalModel | None = None,
    config: Order3TakeoffBCCollectorConfig | None = None,
) -> Order3TakeoffBCCollectionResult:
    """Convert an aligned deterministic v2 takeoff report into BC records.

    The legacy takeoff probe has no learned-action/log-probability trace.  This
    collector can therefore create only deterministic residual-zero behavior
    cloning targets.  Its output is deliberately not evidence of online PPO or
    learned-policy execution in Isaac.

    Measured floor-contact wrenches are source-only privileged evidence.  With
    the default configuration an explicit actor view is reconstructed with
    ``ContactRuntimeState.wrench_world=None`` and the removal count is recorded.
    Disabling that reconstruction makes any such source wrench a hard error.
    """

    cfg = config or Order3TakeoffBCCollectorConfig()
    report = _source_report(source)
    _validate_source_provenance(source, report)
    _validate_v2_takeoff_contract(report)
    _reject_learned_action_trace(report)

    raw_sequences = _aligned_raw_sequences(report)
    observations = [
        RuntimeObservation.from_dict(item)
        for item in raw_sequences[TAKEOFF_RUNTIME_OBSERVATIONS]
    ]
    policy_commands = [
        PolicyCommand.from_dict(item)
        for item in raw_sequences[TAKEOFF_POLICY_COMMANDS]
    ]
    controller_commands = [
        ControllerCommand.from_dict(item)
        for item in raw_sequences[TAKEOFF_CONTROLLER_COMMANDS]
    ]
    actuator_records = [
        IsaacActuatorTargetRecord.from_dict(item)
        for item in raw_sequences[TAKEOFF_ACTUATOR_RECORDS]
    ]
    control_pose_history = raw_sequences[TAKEOFF_CONTROL_POSE_HISTORY]

    morphology = observations[0].morphology_graph
    structural_hash = morphology_structural_hash(morphology)
    if expected_structural_hash is not None and structural_hash != expected_structural_hash:
        raise SchemaValidationError(
            "takeoff report canonical structural hash does not match the assigned split"
        )
    _validate_aligned_typed_sequences(
        observations,
        policy_commands,
        controller_commands,
        actuator_records,
        graph_id=morphology.graph_id,
        structural_hash=structural_hash,
    )

    model = physical_model or build_physical_model_from_config(
        cfg.physical_model_config_path
    )
    reported_model_hash = report.get(
        "random_morphology_takeoff_physical_model_hash"
    )
    if reported_model_hash != model.stable_hash():
        raise SchemaValidationError(
            "takeoff report physical model hash does not match the collector model"
        )
    episode = episode_id or _default_episode_id(report, structural_hash, split)
    require_non_empty(episode, "order3 takeoff episode_id")

    phases = _phase_transitions(report)
    settled_pose = _finite_pose_report_value(
        report, "random_morphology_takeoff_settled_pose_world"
    )
    hover_target = _finite_pose_report_value(
        report, "random_morphology_takeoff_hover_target_pose_world"
    )
    hover_height_delta = float(
        report.get("random_morphology_takeoff_hover_height_delta_m", 0.0)
    )
    if not math.isfinite(hover_height_delta) or hover_height_delta <= 0.0:
        raise SchemaValidationError(
            "takeoff report hover height delta must be finite and positive"
        )
    simulation_dt = float(report["random_morphology_takeoff_sim_dt_s"])
    if not math.isfinite(simulation_dt) or simulation_dt <= 0.0:
        raise SchemaValidationError("takeoff report simulation dt must be positive")
    reported_hover_hold_s = _finite_non_negative_report_number(
        report, "random_morphology_takeoff_hover_hold_time_s"
    )
    reported_hover_hold_required_s = _finite_non_negative_report_number(
        report, "random_morphology_takeoff_hover_hold_required_s"
    )
    if not math.isclose(
        reported_hover_hold_required_s,
        cfg.reward_config.success_hold_duration_s,
        rel_tol=0.0,
        abs_tol=simulation_dt + 1.0e-12,
    ):
        raise SchemaValidationError(
            "Order3 reward hold duration does not match takeoff source contract"
        )

    builder = RigidBodyControlModelBuilder()
    transitions: list[Order3PolicyTransition] = []
    zero_action = [0.0] * ORDER3_ACTION_SIZE
    zero_recurrent_state = [0.0] * cfg.recurrent_state_dim
    previous_tracking_cost: float | None = None
    tolerance_dwell_s = 0.0
    removed_wrench_count = 0
    source_success = bool(
        report.get("random_morphology_takeoff_smoke_passed") is True
    )

    for index, (
        raw_observation,
        command,
        controller_command,
        actuator_record,
        reported_control_pose,
    ) in enumerate(
        zip(
            observations,
            policy_commands,
            controller_commands,
            actuator_records,
            control_pose_history,
            strict=True,
        )
    ):
        actor_observation, removed_for_step = _actor_view_observation(
            raw_observation,
            sanitize_privileged_contact_wrench=(
                cfg.sanitize_privileged_contact_wrench
            ),
        )
        removed_wrench_count += removed_for_step
        control_model = builder.build(morphology, model, actor_observation)
        # The probe logs observation[i] before command[i], while
        # control_pose_history[i] is measured after the corresponding physics
        # step.  Therefore observation[i] must match history[i - 1].  The
        # first observation has no predecessor and the final history row has
        # no aligned post-observation, but both are still shape/finite checked.
        _validate_control_pose_row(reported_control_pose)
        if index > 0:
            _validate_reported_control_pose(
                control_model.body_pose_world,
                control_pose_history[index - 1],
                position_tolerance_m=cfg.control_pose_position_tolerance_m,
                attitude_tolerance_rad=cfg.control_pose_attitude_tolerance_rad,
            )

        phase = _phase_at(actor_observation.time_s, phases)
        policy_applied = phase != "settle"
        if policy_applied:
            if command.desired_body_pose is None or command.desired_body_twist is None:
                raise SchemaValidationError(
                    "active takeoff teacher rows require desired body pose and twist"
                )
            target_pose = command.desired_body_pose
            target_twist = list(command.desired_body_twist)
        else:
            if command.desired_body_pose is not None or command.desired_body_twist is not None:
                raise SchemaValidationError(
                    "settle rows must not apply a desired body pose/twist policy"
                )
            target_pose = settled_pose
            target_twist = [0.0] * 6
        _validate_zero_residual_teacher(command)

        height_gain_ratio = max(
            0.0,
            (control_model.body_pose_world[2] - settled_pose[2])
            / hover_height_delta,
        )
        is_source_terminal = index == len(observations) - 1
        safety = _step_safety_flags(
            controller_command,
            actuator_record,
            report=report,
            source_terminal=is_source_terminal,
            source_success=source_success,
        )
        energy = _normalized_controller_energy(controller_command)

        probe_step = Order3FreeFlightStep(
            module_count=len(morphology.modules),
            task_mode=Order3TaskMode.TAKEOFF,
            centroidal_pose_world=control_model.body_pose_world,
            centroidal_twist_world=list(control_model.body_twist_world),
            target_pose_world=target_pose,
            target_twist_world=target_twist,
            previous_tracking_cost=previous_tracking_cost,
            within_tolerance_duration_s=0.0,
            takeoff_height_gain_ratio=height_gain_ratio,
            normalized_energy=energy,
            normalized_action_delta=0.0,
            qp_feasible=safety["qp_feasible"],
            hard_collision=safety["hard_collision"],
            non_finite_state=safety["non_finite_state"],
            unsupported_actuator=safety["unsupported_actuator"],
            actuator_saturated=safety["actuator_saturated"],
            fallback_active=safety["fallback_active"],
            timed_out=safety["timed_out"],
            terminal=is_source_terminal,
        )
        preliminary = compute_order3_free_flight_reward(
            probe_step, config=cfg.reward_config
        )
        if phase in {"hover_hold", "complete"} and _within_success_tolerance(
            preliminary.terms, cfg.reward_config
        ):
            tolerance_dwell_s += simulation_dt
        else:
            tolerance_dwell_s = 0.0
        reward_dwell = (
            max(tolerance_dwell_s, reported_hover_hold_s)
            if is_source_terminal
            else min(
                tolerance_dwell_s,
                max(cfg.reward_config.success_hold_duration_s - 1.0e-9, 0.0),
            )
        )
        reward_step = Order3FreeFlightStep.from_dict(
            {
                **probe_step.to_dict(),
                "within_tolerance_duration_s": reward_dwell,
            }
        )
        reward_result = compute_order3_free_flight_reward(
            reward_step, config=cfg.reward_config
        )
        if reward_result.terminal != is_source_terminal:
            raise SchemaValidationError(
                "takeoff source contains a terminal safety/success outcome before its final row"
            )
        if is_source_terminal and reward_result.success != source_success:
            raise SchemaValidationError(
                "recomputed Order3 takeoff success does not match source report success "
                f"(source={source_success}, recomputed={reward_result.success}, "
                f"dwell_s={reward_dwell:.9g}, tracking_cost={reward_result.tracking_cost:.9g}, "
                f"height_gain_ratio={height_gain_ratio:.9g})"
            )

        metrics = {
            "isaac_backed": 1.0,
            "deterministic_v2_teacher": 1.0,
            "teacher_normalized_action_zero": 1.0,
            "policy_applied": 1.0 if policy_applied else 0.0,
            "settle_phase": 1.0 if phase == "settle" else 0.0,
            "true_centroidal_tracking": 1.0,
            "removed_privileged_contact_wrench_count": float(removed_for_step),
            "learned_action_trace_available": 0.0,
            "online_ppo_rollout_eligible": 0.0,
            "object_task_claim": 0.0,
            "contact_task_claim": 0.0,
            "p4_full_completion_claim": 0.0,
            **{
                f"reward.{key}": float(value)
                for key, value in reward_result.terms.items()
            },
        }
        transitions.append(
            Order3PolicyTransition(
                episode_id=episode,
                split=split,
                graph_id=morphology.graph_id,
                structural_hash=structural_hash,
                step_index=index,
                time_s=actor_observation.time_s,
                runtime_observation=actor_observation,
                target_pose_world=target_pose,
                target_twist=target_twist,
                previous_action=list(zero_action),
                action=list(zero_action),
                recurrent_state_in=list(zero_recurrent_state),
                old_log_prob=0.0,
                old_value=0.0,
                reward=reward_result.reward,
                terminal=is_source_terminal,
                policy_applied=policy_applied,
                privileged_disturbance_body=[0.0] * 6,
                metrics=metrics,
            )
        )
        previous_tracking_cost = reward_result.tracking_cost

    return Order3TakeoffBCCollectionResult(
        collector_version=ORDER3_TAKEOFF_COLLECTOR_VERSION,
        episode_id=episode,
        split=split,
        structural_hash=structural_hash,
        transitions=transitions,
        removed_privileged_contact_wrench_count=removed_wrench_count,
        source_is_real_isaac=True,
        learned_action_trace_available=False,
        online_ppo_rollout_eligible=False,
        metadata={
            "source_type": type(source).__name__,
            "source_graph_id": morphology.graph_id,
            "source_transition_count": len(transitions),
            "source_control_contract_version": POLICY_COMMAND_CONTRACT_CENTROIDAL,
            "source_tracking_state": TRUE_CENTROIDAL_TRACKING_SOURCE,
            "source_is_real_isaac": True,
            "teacher_action_contract": "normalized_residual_zero",
            "teacher_target_contract": "active_deterministic_desired_pose_twist",
            "control_pose_history_semantics": "post_step_pose_i_matches_observation_i_plus_1",
            "settle_policy_applied": False,
            "sanitization_enabled": cfg.sanitize_privileged_contact_wrench,
            "removed_privileged_contact_wrench_count": removed_wrench_count,
            "learned_action_trace_available": False,
            "online_ppo_rollout_eligible": False,
            "boundary": "bc_only_source_report_has_no_learned_action_or_log_probability_trace",
            "object_task_claim": False,
            "contact_task_claim": False,
            "p4_full_completion_claim": False,
        },
    )


def _source_report(
    source: RandomMorphologyTakeoffResult | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(source, RandomMorphologyTakeoffResult):
        return source.report
    if not isinstance(source, dict):
        raise TypeError("Order3 takeoff source must be a result or report mapping")
    return source


def _validate_source_provenance(
    source: RandomMorphologyTakeoffResult | dict[str, Any],
    report: dict[str, Any],
) -> None:
    if isinstance(source, RandomMorphologyTakeoffResult):
        if not source.attempted or source.dry_run or not source.isaac_backed:
            raise SchemaValidationError(
                "Order3 takeoff BC requires an attempted non-dry real-Isaac result"
            )
    required = {
        "spawn_passed": True,
        "isaac_backed": True,
        "random_morphology_takeoff_smoke": True,
    }
    for key, expected in required.items():
        if report.get(key) is not expected:
            raise SchemaValidationError(
                f"Order3 takeoff report lacks real-Isaac provenance: {key}"
            )
    return_code = report.get("command_returncode", 0)
    if not isinstance(return_code, int) or isinstance(return_code, bool) or return_code != 0:
        raise SchemaValidationError("Order3 takeoff report command did not succeed")
    artifacts = report.get("random_morphology_takeoff_artifacts")
    if not isinstance(artifacts, dict):
        raise SchemaValidationError("Order3 takeoff report lacks provenance artifacts")
    artifact_expected = {
        "backend": "isaac_lab",
        "isaac_backed": True,
        "dry_run": False,
        "object_task_claim": False,
        "is_p4_full_completion": False,
    }
    for key, expected in artifact_expected.items():
        if artifacts.get(key) != expected:
            raise SchemaValidationError(
                f"Order3 takeoff artifact provenance mismatch: {key}"
            )


def _validate_v2_takeoff_contract(report: dict[str, Any]) -> None:
    expected = {
        "random_morphology_takeoff_control_contract_version": (
            POLICY_COMMAND_CONTRACT_CENTROIDAL
        ),
        "random_morphology_takeoff_tracking_state_source": (
            TRUE_CENTROIDAL_TRACKING_SOURCE
        ),
        "random_morphology_takeoff_true_centroidal_tracking": True,
        "random_morphology_takeoff_contact_wrench_tracking_claim": False,
        "random_morphology_takeoff_internal_wrench_tracking_claim": False,
        "random_morphology_takeoff_learned_policy_used": False,
        "random_morphology_takeoff_controller": "deterministic_qpid",
    }
    for key, value in expected.items():
        if key not in report or report[key] != value:
            raise SchemaValidationError(
                f"Order3 takeoff report v2/centroidal contract mismatch: {key}"
            )


def _reject_learned_action_trace(report: dict[str, Any]) -> None:
    for key in _LEARNED_ACTION_TRACE_KEYS:
        value = report.get(key)
        if value not in (None, []):
            raise SchemaValidationError(
                "Order3 takeoff BC collector does not accept a learned action trace"
            )


def _aligned_raw_sequences(report: dict[str, Any]) -> dict[str, list[Any]]:
    keys = (
        TAKEOFF_RUNTIME_OBSERVATIONS,
        TAKEOFF_POLICY_COMMANDS,
        TAKEOFF_CONTROLLER_COMMANDS,
        TAKEOFF_ACTUATOR_RECORDS,
        TAKEOFF_CONTROL_POSE_HISTORY,
    )
    sequences: dict[str, list[Any]] = {}
    for key in keys:
        value = report.get(key)
        if not isinstance(value, list):
            raise SchemaValidationError(
                f"Order3 takeoff report is missing aligned sequence {key!r}"
            )
        sequences[key] = value
    lengths = {len(value) for value in sequences.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise SchemaValidationError(
            f"Order3 takeoff report aligned sequence lengths mismatch: {sorted(lengths)}"
        )
    count = next(iter(lengths))
    reported_steps = report.get("random_morphology_takeoff_steps")
    if (
        not isinstance(reported_steps, int)
        or isinstance(reported_steps, bool)
        or reported_steps != count
    ):
        raise SchemaValidationError(
            "Order3 takeoff report step count does not match aligned sequences"
        )
    return sequences


def _validate_aligned_typed_sequences(
    observations: Sequence[RuntimeObservation],
    policy_commands: Sequence[PolicyCommand],
    controller_commands: Sequence[ControllerCommand],
    actuator_records: Sequence[IsaacActuatorTargetRecord],
    *,
    graph_id: str,
    structural_hash: str,
) -> None:
    previous_time = -1.0
    for index, (observation, policy, controller, actuator) in enumerate(
        zip(
            observations,
            policy_commands,
            controller_commands,
            actuator_records,
            strict=True,
        )
    ):
        if observation.time_s < previous_time:
            raise SchemaValidationError("Order3 takeoff observation time decreases")
        previous_time = observation.time_s
        if observation.morphology_graph.graph_id != graph_id:
            raise SchemaValidationError("Order3 takeoff observations cross graph ids")
        if morphology_structural_hash(observation.morphology_graph) != structural_hash:
            raise SchemaValidationError(
                "Order3 takeoff observations cross canonical morphologies"
            )
        if policy.control_contract_version != POLICY_COMMAND_CONTRACT_CENTROIDAL:
            raise SchemaValidationError("Order3 takeoff policy command is legacy")
        if controller.control_contract_version != POLICY_COMMAND_CONTRACT_CENTROIDAL:
            raise SchemaValidationError("Order3 takeoff controller command is legacy")
        if actuator.backend != "isaac_lab":
            raise SchemaValidationError("Order3 takeoff actuator provenance is not Isaac Lab")
        if actuator.morphology_graph_id != graph_id or actuator.command_index != index:
            raise SchemaValidationError("Order3 takeoff actuator alignment mismatch")
        if not math.isclose(
            actuator.time_s, observation.time_s, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise SchemaValidationError("Order3 takeoff actuator time mismatch")


def _actor_view_observation(
    observation: RuntimeObservation,
    *,
    sanitize_privileged_contact_wrench: bool,
) -> tuple[RuntimeObservation, int]:
    data = observation.to_dict()
    removed = 0
    for contact in data.get("contact_states", []):
        metadata = contact.get("metadata", {})
        forbidden = _find_forbidden_contact_metadata(metadata)
        if forbidden is not None:
            raise SchemaValidationError(
                f"Order3 takeoff contact metadata contains privileged field {forbidden!r}"
            )
        if contact.get("wrench_world") is None:
            continue
        if not sanitize_privileged_contact_wrench:
            raise SchemaValidationError(
                "Order3 actor observation contains privileged contact wrench"
            )
        contact["wrench_world"] = None
        removed += 1
    actor_view = RuntimeObservation.from_dict(data)
    if any(contact.wrench_world is not None for contact in actor_view.contact_states):
        raise SchemaValidationError(
            "Order3 contact-wrench sanitization did not produce an actor-safe view"
        )
    return actor_view, removed


def _find_forbidden_contact_metadata(value: Any) -> str | None:
    if isinstance(value, dict):
        for raw_key, item in value.items():
            key = str(raw_key).strip().lower()
            if (
                key in _FORBIDDEN_CONTACT_METADATA_KEYS
                or key.startswith("privileged_")
                or key.startswith("ground_truth_contact_wrench")
            ):
                return key
            nested = _find_forbidden_contact_metadata(item)
            if nested is not None:
                return nested
    elif isinstance(value, (list, tuple)):
        for item in value:
            nested = _find_forbidden_contact_metadata(item)
            if nested is not None:
                return nested
    return None


def _validate_reported_control_pose(
    computed_pose: Sequence[float],
    reported_pose: Any,
    *,
    position_tolerance_m: float,
    attitude_tolerance_rad: float,
) -> None:
    if not isinstance(reported_pose, (list, tuple)):
        raise SchemaValidationError("Order3 takeoff control pose history row is invalid")
    require_len(reported_pose, 7, "takeoff control pose history row")
    if any(not math.isfinite(float(value)) for value in reported_pose):
        raise SchemaValidationError("Order3 takeoff control pose history is non-finite")
    position_error = math.sqrt(
        sum(
            (float(computed_pose[index]) - float(reported_pose[index])) ** 2
            for index in range(3)
        )
    )
    attitude_error = _quaternion_angle(computed_pose[3:7], reported_pose[3:7])
    if position_error > position_tolerance_m or attitude_error > attitude_tolerance_rad:
        raise SchemaValidationError(
            "Order3 recomputed true-centroidal pose does not match report history"
        )


def _validate_control_pose_row(reported_pose: Any) -> None:
    if not isinstance(reported_pose, (list, tuple)):
        raise SchemaValidationError("Order3 takeoff control pose history row is invalid")
    require_len(reported_pose, 7, "takeoff control pose history row")
    if any(not math.isfinite(float(value)) for value in reported_pose):
        raise SchemaValidationError("Order3 takeoff control pose history is non-finite")


def _phase_transitions(report: dict[str, Any]) -> list[tuple[float, str]]:
    raw = report.get("random_morphology_takeoff_phase_transitions")
    if not isinstance(raw, list) or not raw:
        raise SchemaValidationError("Order3 takeoff report lacks phase transitions")
    phases: list[tuple[float, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise SchemaValidationError("Order3 takeoff phase transition is invalid")
        time_s = item.get("time_s")
        phase = item.get("to_phase")
        if (
            not isinstance(time_s, (int, float))
            or isinstance(time_s, bool)
            or not math.isfinite(float(time_s))
            or not isinstance(phase, str)
            or phase not in {"settle", "takeoff_ramp", "hover_hold", "complete"}
        ):
            raise SchemaValidationError("Order3 takeoff phase transition is invalid")
        phases.append((float(time_s), phase))
    phases.sort()
    if phases[0] != (0.0, "settle"):
        raise SchemaValidationError("Order3 takeoff phase sequence must begin at settle")
    return phases


def _phase_at(time_s: float, phases: Sequence[tuple[float, str]]) -> str:
    current = phases[0][1]
    for transition_time, phase in phases:
        if transition_time > time_s + 1.0e-12:
            break
        current = phase
    return current


def _finite_pose_report_value(report: dict[str, Any], key: str) -> tuple[float, ...]:
    value = report.get(key)
    if not isinstance(value, (list, tuple)):
        raise SchemaValidationError(f"Order3 takeoff report is missing {key}")
    require_len(value, 7, key)
    output = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in output):
        raise SchemaValidationError(f"Order3 takeoff report {key} is non-finite")
    return output


def _finite_non_negative_report_number(report: dict[str, Any], key: str) -> float:
    value = report.get(key)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise SchemaValidationError(
            f"Order3 takeoff report {key} must be finite and non-negative"
        )
    return float(value)


def _validate_zero_residual_teacher(command: PolicyCommand) -> None:
    residual = command.residual_wrench_body or [0.0] * 6
    if any(abs(float(value)) > 1.0e-12 for value in residual):
        raise SchemaValidationError(
            "Order3 takeoff BC zero-action teacher has non-zero residual wrench"
        )
    neutral_joint_fields = (
        command.joint_position_targets,
        command.joint_velocity_targets,
        command.joint_torque_bias,
    )
    if any(
        abs(float(value)) > 1.0e-12
        for field in neutral_joint_fields
        for value in field.values()
    ) or command.desired_anchor_pose_offsets:
        raise SchemaValidationError(
            "Order3 takeoff BC zero-action teacher contains an unsupported residual field"
        )


def _step_safety_flags(
    command: ControllerCommand,
    actuator: IsaacActuatorTargetRecord,
    *,
    report: dict[str, Any],
    source_terminal: bool,
    source_success: bool,
) -> dict[str, bool]:
    qp_feasible = bool(command.controller_status.qp_feasible)
    unsupported = bool(actuator.unsupported_actuators)
    clipped = bool(actuator.clipped_targets) or bool(
        command.controller_status.metrics.get("clipped", 0.0) > 0.0
    )
    fallback = bool(
        command.controller_status.metrics.get("degraded_fallback", 0.0) > 0.0
    )
    hard_collision = bool(
        source_terminal
        and report.get("random_morphology_takeoff_exact_cross_module_collision_passed")
        is not True
    )
    non_finite = bool(
        source_terminal
        and report.get("random_morphology_takeoff_finite_state") is not True
    )
    timed_out = bool(
        source_terminal
        and not source_success
        and qp_feasible
        and not unsupported
        and not hard_collision
        and not non_finite
    )
    return {
        "qp_feasible": qp_feasible,
        "hard_collision": hard_collision,
        "non_finite_state": non_finite,
        "unsupported_actuator": unsupported,
        "actuator_saturated": clipped,
        "fallback_active": fallback,
        "timed_out": timed_out,
    }


def _normalized_controller_energy(command: ControllerCommand) -> float:
    values = [
        (float(value) / 20.0) ** 2
        for value in command.rotor_thrusts_n.values()
        if math.isfinite(float(value))
    ]
    return min(sum(values) / len(values), 1.0) if values else 0.0


def _within_success_tolerance(
    terms: dict[str, float],
    config: Order3FreeFlightRewardConfig,
) -> bool:
    return (
        terms["position_error_m"] <= config.success_position_threshold_m
        and terms["attitude_error_rad"] <= config.success_attitude_threshold_rad
        and terms["linear_velocity_error_mps"]
        <= config.success_linear_velocity_threshold_mps
        and terms["angular_velocity_error_rad_s"]
        <= config.success_angular_velocity_threshold_rad_s
    )


def _default_episode_id(
    report: dict[str, Any], structural_hash: str, split: DatasetSplit
) -> str:
    seed = {
        "collector_version": ORDER3_TAKEOFF_COLLECTOR_VERSION,
        "structural_hash": structural_hash,
        "split": split.value,
        "backend_config_hash": report.get(
            "random_morphology_takeoff_backend_config_hash"
        ),
        "physical_model_hash": report.get(
            "random_morphology_takeoff_physical_model_hash"
        ),
        "steps": report.get("random_morphology_takeoff_steps"),
        "settled_pose": report.get("random_morphology_takeoff_settled_pose_world"),
        "hover_target": report.get(
            "random_morphology_takeoff_hover_target_pose_world"
        ),
    }
    return f"order3-takeoff-bc-{stable_hash(seed)[:16]}"


def _quaternion_angle(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise SchemaValidationError("Order3 takeoff control pose quaternion is invalid")
    dot = sum(
        float(lhs) * float(rhs) / (left_norm * right_norm)
        for lhs, rhs in zip(left, right, strict=True)
    )
    return 2.0 * math.acos(min(max(abs(dot), 0.0), 1.0))
