from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from amsrr.controllers.isaac_controller_bridge import IsaacActuatorTargetRecord
from amsrr.controllers.rigid_body_model import (
    RigidBodyControlModel,
    RigidBodyControlModelBuilder,
)
from amsrr.morphology.random_connected import morphology_structural_hash
from amsrr.policies.morphology_conditioned_low_level_policy import (
    load_order3_policy_checkpoint,
    order3_actor_feature_vector,
)
from amsrr.schemas.common import (
    SchemaBase,
    SchemaValidationError,
    require_len,
    require_non_empty,
)
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.order3 import (
    ORDER3_ACTION_SIZE,
    ORDER3_CHECKPOINT_VERSION,
    Order3PolicyCheckpointMetadata,
    Order3PolicyTransition,
)
from amsrr.schemas.order3_rollout_condition import Order3RolloutCondition
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import (
    POLICY_COMMAND_CONTRACT_CENTROIDAL,
    ControllerCommand,
    PolicyCommand,
)
from amsrr.schemas.runtime import RuntimeObservation
from amsrr.simulation.random_morphology_takeoff import RandomMorphologyTakeoffResult
from amsrr.training.order3_free_flight import (
    TRUE_CENTROIDAL_TRACKING_SOURCE,
    Order3FreeFlightRewardConfig,
    Order3FreeFlightStep,
    Order3PrivilegedRewardSignals,
    Order3TaskMode,
    compute_order3_free_flight_reward,
)
from amsrr.utils.hashing import stable_hash

ORDER3_ONLINE_COLLECTOR_VERSION = "order3_online_ppo_collector_v1"

_RUNTIME_ROWS = "random_morphology_takeoff_runtime_observations"
_POLICY_ROWS = "random_morphology_takeoff_policy_commands"
_CONTROLLER_ROWS = "random_morphology_takeoff_controller_commands"
_ACTUATOR_ROWS = "random_morphology_takeoff_actuator_target_records"
_TRACE_ROWS = "order3_pi_l_transition_traces"
_FINAL_OBSERVATION = "order3_pi_l_final_runtime_observation"
_FINAL_BOOTSTRAP = "order3_pi_l_final_bootstrap_value"

_FORBIDDEN_CONTACT_METADATA_KEYS = {
    "contact_wrench",
    "contact_wrench_world",
    "ground_truth_contact_wrench",
    "privileged_contact_wrench",
}


@dataclass
class Order3OnlineCollectorConfig(SchemaBase):
    """Fail-closed conversion contract for learned real-Isaac PPO rollouts."""

    reward_config: Order3FreeFlightRewardConfig = field(
        default_factory=Order3FreeFlightRewardConfig
    )
    task_mode: Order3TaskMode | None = None
    sanitize_privileged_contact_wrench: bool = True
    require_stochastic_behavior: bool = True
    sequence_tolerance: float = 1.0e-7
    joint_hold_tolerance: float = 1.0e-6
    behavior_replay_tolerance: float = 2.0e-4

    def validate(self) -> None:
        for name in (
            "sequence_tolerance",
            "joint_hold_tolerance",
            "behavior_replay_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise SchemaValidationError(
                    f"Order3OnlineCollectorConfig.{name} must be finite and positive"
                )


@dataclass
class Order3OnlineCollectionResult(SchemaBase):
    collector_version: str
    episode_id: str
    split: DatasetSplit
    structural_hash: str
    checkpoint_sha256: str
    transitions: list[Order3PolicyTransition]
    removed_privileged_contact_wrench_count: int
    source_is_real_isaac: bool = True
    online_ppo_rollout_eligible: bool = True
    object_task_claim: bool = False
    contact_task_claim: bool = False
    dock_motion_claim: bool = False
    p4_full_completion_claim: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.collector_version != ORDER3_ONLINE_COLLECTOR_VERSION:
            raise SchemaValidationError("Order3 online collector version mismatch")
        require_non_empty(self.episode_id, "Order3OnlineCollectionResult.episode_id")
        require_non_empty(
            self.structural_hash,
            "Order3OnlineCollectionResult.structural_hash",
        )
        if not _is_sha256(self.checkpoint_sha256):
            raise SchemaValidationError(
                "Order3OnlineCollectionResult.checkpoint_sha256 must be sha256"
            )
        if self.removed_privileged_contact_wrench_count < 0:
            raise SchemaValidationError(
                "removed privileged contact wrench count must be non-negative"
            )
        if not self.source_is_real_isaac or not self.online_ppo_rollout_eligible:
            raise SchemaValidationError(
                "Order3 online PPO collection requires an eligible real-Isaac source"
            )
        if any(
            (
                self.object_task_claim,
                self.contact_task_claim,
                self.dock_motion_claim,
                self.p4_full_completion_claim,
            )
        ):
            raise SchemaValidationError(
                "Order3 online free-flight collection cannot claim object/contact/dock/P4-full scope"
            )
        if not self.transitions:
            raise SchemaValidationError(
                "Order3OnlineCollectionResult.transitions must not be empty"
            )
        previous_step = -1
        for index, transition in enumerate(self.transitions):
            if transition.episode_id != self.episode_id:
                raise SchemaValidationError(
                    "Order3 online transitions cross episode ids"
                )
            if transition.split != self.split:
                raise SchemaValidationError("Order3 online transitions cross splits")
            if transition.structural_hash != self.structural_hash:
                raise SchemaValidationError(
                    "Order3 online transitions cross structural hashes"
                )
            if transition.step_index <= previous_step:
                raise SchemaValidationError(
                    "Order3 online transition source steps must be strictly increasing"
                )
            previous_step = transition.step_index
            if transition.behavior_policy_kind != "order3_checkpoint":
                raise SchemaValidationError(
                    "Order3 online transition behavior must be an Order3 checkpoint"
                )
            if transition.behavior_policy_version != ORDER3_CHECKPOINT_VERSION:
                raise SchemaValidationError(
                    "Order3 online transition checkpoint version mismatch"
                )
            if transition.behavior_checkpoint_hash != self.checkpoint_sha256:
                raise SchemaValidationError(
                    "Order3 online transition checkpoint hash mismatch"
                )
            if transition.action_semantics != "learned_residual":
                raise SchemaValidationError(
                    "Order3 online transition must preserve learned residual actions"
                )
            if not transition.policy_applied:
                raise SchemaValidationError(
                    "fallback rows are not eligible as on-policy Order3 transitions"
                )
            is_last = index == len(self.transitions) - 1
            if not is_last and (transition.terminal or transition.truncated):
                raise SchemaValidationError(
                    "Order3 online episode boundary must be its final transition"
                )
            if is_last and transition.terminal == transition.truncated:
                raise SchemaValidationError(
                    "Order3 online final transition must be terminal xor truncated"
                )


@dataclass(frozen=True)
class _Trace:
    source_step: int
    time_s: float
    target_pose_world: tuple[float, ...]
    target_twist: list[float]
    previous_action: list[float]
    action: list[float]
    recurrent_state_in: list[float]
    recurrent_state_out: list[float]
    old_log_prob: float
    old_value: float
    privileged_disturbance_body: list[float]


@dataclass(frozen=True)
class _AlignedRows:
    raw_observations: list[RuntimeObservation]
    actor_observations: list[RuntimeObservation]
    policy_commands: list[PolicyCommand]
    controller_commands: list[ControllerCommand]
    actuator_records: list[IsaacActuatorTargetRecord]
    final_raw_observation: RuntimeObservation
    final_actor_observation: RuntimeObservation
    removed_wrench_counts: list[int]
    final_removed_wrench_count: int


def collect_order3_online_transitions(
    source: RandomMorphologyTakeoffResult | dict[str, Any],
    *,
    split: DatasetSplit,
    physical_model: PhysicalModel,
    expected_structural_hash: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    behavior_checkpoint_path: str | Path | None = None,
    episode_id: str | None = None,
    config: Order3OnlineCollectorConfig | None = None,
) -> Order3OnlineCollectionResult:
    """Convert learned policy-decision traces into on-policy PPO transitions.

    Rewards are assigned to the action interval beginning at a policy decision.
    Its outcome is the next decision observation, or the separately logged final
    post-step observation for the last action.  Contact wrench truth is removed
    from every actor observation before any transition is constructed.
    """

    cfg = config or Order3OnlineCollectorConfig()
    report = _source_report(source)
    rollout_condition, task_mode = _resolve_rollout_condition(report, cfg)
    _validate_real_isaac_learned_provenance(
        source,
        report,
        cfg,
        task_mode=task_mode,
    )
    checkpoint_hash, checkpoint_metadata = _checkpoint_contract(
        report,
        physical_model=physical_model,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    aligned = _aligned_rows(
        report,
        sanitize_privileged_contact_wrench=(cfg.sanitize_privileged_contact_wrench),
    )
    morphology = aligned.raw_observations[0].morphology_graph
    structural_hash = morphology_structural_hash(morphology)
    if (
        expected_structural_hash is not None
        and structural_hash != expected_structural_hash
    ):
        raise SchemaValidationError(
            "Order3 online canonical structural hash does not match the assigned split"
        )
    _validate_graph_and_physical_identity(
        report,
        aligned,
        physical_model=physical_model,
        structural_hash=structural_hash,
    )
    traces = _parse_and_validate_traces(
        report,
        aligned,
        physical_model=physical_model,
        config=cfg,
    )
    behavior_replay_verified = False
    if behavior_checkpoint_path is not None:
        _validate_behavior_replay(
            behavior_checkpoint_path,
            expected_checkpoint_sha256=checkpoint_hash,
            traces=traces,
            aligned=aligned,
            physical_model=physical_model,
            tolerance=cfg.behavior_replay_tolerance,
        )
        behavior_replay_verified = True

    episode = episode_id or _default_episode_id(
        report,
        structural_hash=structural_hash,
        checkpoint_sha256=checkpoint_hash,
        split=split,
    )
    require_non_empty(episode, "Order3 online episode_id")

    if rollout_condition is None or task_mode == Order3TaskMode.TAKEOFF:
        final_target = _finite_pose(
            report.get("random_morphology_takeoff_hover_target_pose_world"),
            "random_morphology_takeoff_hover_target_pose_world",
        )
        settled_pose: tuple[float, ...] | None = _finite_pose(
            report.get("random_morphology_takeoff_settled_pose_world"),
            "random_morphology_takeoff_settled_pose_world",
        )
        hover_height_delta: float | None = _finite_positive(
            report,
            "random_morphology_takeoff_hover_height_delta_m",
        )
    else:
        realization = report.get("order3_condition_realization")
        if not isinstance(realization, dict):
            raise SchemaValidationError(
                "Order3 in-air rollout lacks its applied condition realization"
            )
        final_target = _finite_pose(
            realization.get("final_target_pose_world"),
            "order3_condition_realization.final_target_pose_world",
        )
        settled_pose = None
        hover_height_delta = None
    terminal_metrics = report.get("order3_free_flight_terminal_metrics")
    if rollout_condition is not None:
        if not isinstance(terminal_metrics, dict):
            raise SchemaValidationError("Order3 rollout terminal metrics are missing")
        reported_hold_s = _finite_number(
            terminal_metrics.get("within_tolerance_duration_s"),
            "order3_free_flight_terminal_metrics.within_tolerance_duration_s",
        )
        if reported_hold_s < 0.0:
            raise SchemaValidationError(
                "Order3 terminal tolerance dwell must be non-negative"
            )
    else:
        reported_hold_s = _finite_non_negative(
            report,
            "random_morphology_takeoff_hover_hold_time_s",
        )
    _validate_reward_contract(
        report,
        cfg.reward_config,
        rollout_condition=rollout_condition,
    )

    builder = RigidBodyControlModelBuilder()
    control_models = [
        builder.build(morphology, physical_model, observation)
        for observation in aligned.actor_observations
    ]
    final_control_model = builder.build(
        morphology,
        physical_model,
        aligned.final_actor_observation,
    )
    tolerance_dwell_s = 0.0
    source_success = (
        report.get("order3_free_flight_success") is True
        if rollout_condition is not None
        else report.get("random_morphology_takeoff_smoke_passed") is True
    )
    final_bootstrap = report.get(_FINAL_BOOTSTRAP)
    if final_bootstrap is not None:
        final_bootstrap = _finite_number(final_bootstrap, _FINAL_BOOTSTRAP)

    transitions: list[Order3PolicyTransition] = []
    for trace_index, trace in enumerate(traces):
        is_last = trace_index == len(traces) - 1
        outcome_step = (
            traces[trace_index + 1].source_step
            if not is_last
            else len(aligned.actor_observations)
        )
        outcome_observation = (
            aligned.actor_observations[outcome_step]
            if not is_last
            else aligned.final_actor_observation
        )
        outcome_model = (
            control_models[outcome_step] if not is_last else final_control_model
        )
        start_model = control_models[trace.source_step]
        interval_controllers = aligned.controller_commands[
            trace.source_step : outcome_step
        ]
        interval_actuators = aligned.actuator_records[trace.source_step : outcome_step]
        if not interval_controllers or not interval_actuators:
            raise SchemaValidationError(
                "Order3 policy decision has no causally aligned control interval"
            )

        start_tracking = _tracking_probe(
            start_model,
            trace,
            module_count=len(morphology.modules),
            task_mode=task_mode,
            settled_pose=settled_pose,
            hover_height_delta=hover_height_delta,
            config=cfg.reward_config,
        )
        target_is_final = _poses_close(
            trace.target_pose_world,
            final_target,
            position_tolerance=cfg.sequence_tolerance,
            attitude_tolerance=cfg.sequence_tolerance,
        )
        tolerance_dwell_s = _advance_tolerance_dwell(
            tolerance_dwell_s if target_is_final else 0.0,
            trace=trace,
            start_step=trace.source_step,
            outcome_step=outcome_step,
            control_models=control_models,
            observation_times=[
                observation.time_s for observation in aligned.actor_observations
            ],
            final_control_model=final_control_model,
            final_time_s=aligned.final_actor_observation.time_s,
            target_is_final=target_is_final,
            config=cfg.reward_config,
        )
        reward_dwell_s = tolerance_dwell_s
        reported_terminal_dwell_used = False
        if is_last and source_success:
            reward_dwell_s = max(reward_dwell_s, reported_hold_s)
            reported_terminal_dwell_used = reported_hold_s > tolerance_dwell_s
        elif not is_last:
            reward_dwell_s = min(
                reward_dwell_s,
                max(cfg.reward_config.success_hold_duration_s - 1.0e-12, 0.0),
            )

        safety = _interval_safety(
            interval_controllers,
            interval_actuators,
            report=report,
            is_last=is_last,
        )
        reward_step = Order3FreeFlightStep(
            module_count=len(morphology.modules),
            task_mode=task_mode,
            centroidal_pose_world=outcome_model.body_pose_world,
            centroidal_twist_world=list(outcome_model.body_twist_world),
            target_pose_world=trace.target_pose_world,
            target_twist_world=list(trace.target_twist),
            previous_tracking_cost=start_tracking,
            within_tolerance_duration_s=reward_dwell_s,
            takeoff_height_gain_ratio=(
                max(
                    0.0,
                    (
                        float(outcome_model.body_pose_world[2])
                        - float(settled_pose[2])
                    )
                    / float(hover_height_delta),
                )
                if task_mode == Order3TaskMode.TAKEOFF
                and settled_pose is not None
                and hover_height_delta is not None
                else None
            ),
            normalized_energy=_interval_energy(interval_controllers),
            normalized_action_delta=_normalized_action_delta(
                trace.previous_action,
                trace.action,
            ),
            qp_feasible=safety["qp_feasible"],
            hard_collision=safety["hard_collision"],
            non_finite_state=safety["non_finite_state"],
            unsupported_actuator=safety["unsupported_actuator"],
            actuator_saturated=safety["actuator_saturated"],
            fallback_active=False,
            timed_out=False,
            terminal=False,
            privileged=Order3PrivilegedRewardSignals(
                applied_external_wrench_body=list(trace.privileged_disturbance_body)
            ),
        )
        reward_result = compute_order3_free_flight_reward(
            reward_step,
            config=cfg.reward_config,
        )
        if not is_last and reward_result.terminal:
            raise SchemaValidationError(
                "Order3 online report contains policy decisions after a terminal outcome"
            )
        if is_last and source_success and not reward_result.success:
            raise SchemaValidationError(
                "Order3 online source success disagrees with recomputed true-centroidal reward"
            )
        if is_last and not source_success and reward_result.success:
            raise SchemaValidationError(
                "Order3 online source failure disagrees with recomputed terminal success"
            )

        terminal = bool(is_last and reward_result.terminal)
        truncated = bool(is_last and not terminal)
        bootstrap_value: float | None = None
        if truncated:
            if final_bootstrap is None:
                raise SchemaValidationError(
                    "Order3 time-limit truncation requires a finite final bootstrap value"
                )
            bootstrap_value = float(final_bootstrap)

        removed_for_actor_row = aligned.removed_wrench_counts[trace.source_step]
        metrics = {
            "isaac_backed": 1.0,
            "online_ppo_rollout_eligible": 1.0,
            "learned_residual_action": 1.0,
            "true_centroidal_tracking": 1.0,
            "source_policy_step_index": float(trace.source_step),
            "outcome_step_index": float(outcome_step),
            "decision_interval_step_count": float(outcome_step - trace.source_step),
            "decision_interval_duration_s": float(
                outcome_observation.time_s
                - aligned.actor_observations[trace.source_step].time_s
            ),
            "removed_privileged_contact_wrench_count": float(removed_for_actor_row),
            "reported_terminal_dwell_used": (
                1.0 if reported_terminal_dwell_used else 0.0
            ),
            "terminal": 1.0 if terminal else 0.0,
            "truncated": 1.0 if truncated else 0.0,
            "object_task_claim": 0.0,
            "contact_task_claim": 0.0,
            "dock_motion_claim": 0.0,
            "p4_full_completion_claim": 0.0,
            "task_mode_code": float(list(Order3TaskMode).index(task_mode)),
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
                step_index=trace.source_step,
                time_s=aligned.actor_observations[trace.source_step].time_s,
                runtime_observation=aligned.actor_observations[trace.source_step],
                target_pose_world=trace.target_pose_world,
                target_twist=list(trace.target_twist),
                previous_action=list(trace.previous_action),
                action=list(trace.action),
                recurrent_state_in=list(trace.recurrent_state_in),
                old_log_prob=trace.old_log_prob,
                old_value=trace.old_value,
                reward=reward_result.reward,
                terminal=terminal,
                truncated=truncated,
                bootstrap_value=bootstrap_value,
                policy_applied=True,
                privileged_disturbance_body=list(trace.privileged_disturbance_body),
                metrics=metrics,
                behavior_policy_kind="order3_checkpoint",
                behavior_policy_version=ORDER3_CHECKPOINT_VERSION,
                behavior_checkpoint_hash=checkpoint_hash,
                action_semantics="learned_residual",
            )
        )

    removed_total = (
        sum(aligned.removed_wrench_counts) + aligned.final_removed_wrench_count
    )
    return Order3OnlineCollectionResult(
        collector_version=ORDER3_ONLINE_COLLECTOR_VERSION,
        episode_id=episode,
        split=split,
        structural_hash=structural_hash,
        checkpoint_sha256=checkpoint_hash,
        transitions=transitions,
        removed_privileged_contact_wrench_count=removed_total,
        metadata={
            "source_type": type(source).__name__,
            "source_graph_id": morphology.graph_id,
            "source_policy_decision_count": len(traces),
            "source_simulation_step_count": len(aligned.actor_observations),
            "source_control_contract_version": POLICY_COMMAND_CONTRACT_CENTROIDAL,
            "source_tracking_state": TRUE_CENTROIDAL_TRACKING_SOURCE,
            "task_mode": task_mode.value,
            "curriculum_stage_id": (
                rollout_condition.stage_id if rollout_condition is not None else "legacy_takeoff"
            ),
            "rollout_condition_hash": (
                rollout_condition.condition_hash if rollout_condition is not None else "legacy"
            ),
            "checkpoint_version": checkpoint_metadata.checkpoint_version,
            "checkpoint_training_stage": checkpoint_metadata.training_stage,
            "behavior_policy_kind": "order3_checkpoint",
            "behavior_policy_version": ORDER3_CHECKPOINT_VERSION,
            "behavior_checkpoint_hash": checkpoint_hash,
            "behavior_replay_verified": behavior_replay_verified,
            "behavior_replay_tolerance": cfg.behavior_replay_tolerance,
            "action_semantics": "learned_residual",
            "source_is_real_isaac": True,
            "stochastic_behavior": bool(report["order3_pi_l_stochastic"]),
            "final_outcome_source": _FINAL_OBSERVATION,
            "final_bootstrap_available": final_bootstrap is not None,
            "sanitization_enabled": cfg.sanitize_privileged_contact_wrench,
            "removed_privileged_contact_wrench_count": removed_total,
            "object_task_claim": False,
            "contact_task_claim": False,
            "dock_motion_claim": False,
            "p4_full_completion_claim": False,
        },
    )


def _source_report(
    source: RandomMorphologyTakeoffResult | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(source, RandomMorphologyTakeoffResult):
        return source.report
    if not isinstance(source, dict):
        raise TypeError("Order3 online source must be a result or report mapping")
    return source


def _resolve_rollout_condition(
    report: dict[str, Any],
    config: Order3OnlineCollectorConfig,
) -> tuple[Order3RolloutCondition | None, Order3TaskMode]:
    raw = report.get("order3_rollout_condition")
    if raw is None:
        task_mode = config.task_mode or Order3TaskMode.TAKEOFF
        if task_mode != Order3TaskMode.TAKEOFF:
            raise SchemaValidationError(
                "Order3 hover/waypoint collection requires a hash-bound rollout condition"
            )
        return None, task_mode
    try:
        condition = (
            Order3RolloutCondition.from_json(raw)
            if isinstance(raw, str)
            else Order3RolloutCondition.from_dict(raw)
        )
    except (SchemaValidationError, TypeError, ValueError) as exc:
        raise SchemaValidationError(
            "Order3 learned report rollout condition is invalid"
        ) from exc
    if report.get("order3_rollout_condition_hash") != condition.condition_hash:
        raise SchemaValidationError(
            "Order3 learned report rollout condition hash mismatch"
        )
    if report.get("order3_task_mode") != condition.task_mode:
        raise SchemaValidationError(
            "Order3 learned report task mode differs from its rollout condition"
        )
    task_mode = Order3TaskMode(condition.task_mode)
    if config.task_mode is not None and config.task_mode != task_mode:
        raise SchemaValidationError(
            "Order3 collector task mode differs from the rollout condition"
        )
    return condition, task_mode


def _validate_real_isaac_learned_provenance(
    source: RandomMorphologyTakeoffResult | dict[str, Any],
    report: dict[str, Any],
    config: Order3OnlineCollectorConfig,
    *,
    task_mode: Order3TaskMode,
) -> None:
    if isinstance(source, RandomMorphologyTakeoffResult):
        if not source.attempted or source.dry_run or not source.isaac_backed:
            raise SchemaValidationError(
                "Order3 online collection requires an attempted non-dry real-Isaac result"
            )
    expected = {
        "spawn_passed": True,
        "isaac_backed": True,
        "random_morphology_takeoff_smoke": task_mode == Order3TaskMode.TAKEOFF,
        "random_morphology_takeoff_learned_policy_used": True,
        "order3_pi_l_rollout": True,
        "random_morphology_takeoff_control_contract_version": (
            POLICY_COMMAND_CONTRACT_CENTROIDAL
        ),
        "random_morphology_takeoff_tracking_state_source": (
            TRUE_CENTROIDAL_TRACKING_SOURCE
        ),
        "random_morphology_takeoff_true_centroidal_tracking": True,
        "random_morphology_takeoff_contact_wrench_tracking_claim": False,
        "random_morphology_takeoff_internal_wrench_tracking_claim": False,
        "random_morphology_takeoff_qp_actuator_variable_scope": (
            "rotor_thrust_vectoring_and_slack_only"
        ),
    }
    for key, value in expected.items():
        if key not in report or report[key] != value:
            raise SchemaValidationError(
                f"Order3 learned report provenance/contract mismatch: {key}"
            )
    if report.get("random_morphology_takeoff_controller") != (
        "order3_morphology_conditioned_pi_l_plus_deterministic_qpid"
    ):
        raise SchemaValidationError(
            "Order3 learned report controller identity mismatch"
        )
    if (
        config.require_stochastic_behavior
        and report.get("order3_pi_l_stochastic") is not True
    ):
        raise SchemaValidationError(
            "Order3 online PPO collection requires stochastic behavior"
        )
    return_code = report.get("command_returncode", 0)
    if (
        not isinstance(return_code, int)
        or isinstance(return_code, bool)
        or return_code != 0
    ):
        raise SchemaValidationError("Order3 learned report command did not succeed")
    decisions = _non_negative_int(report, "order3_pi_l_policy_decision_count")
    applied = _non_negative_int(report, "order3_pi_l_policy_applied_count")
    fallback = _non_negative_int(report, "order3_pi_l_fallback_count")
    traces = report.get(_TRACE_ROWS)
    if not isinstance(traces, list) or not traces:
        raise SchemaValidationError("Order3 learned report has no transition traces")
    if decisions != len(traces) or applied != decisions or fallback != 0:
        raise SchemaValidationError(
            "Order3 on-policy traces require every decision to use the learned checkpoint"
        )
    artifacts = report.get("random_morphology_takeoff_artifacts")
    if not isinstance(artifacts, dict):
        raise SchemaValidationError("Order3 learned report lacks Isaac artifacts")
    artifact_expected = {
        "backend": "isaac_lab",
        "isaac_backed": True,
        "dry_run": False,
        "is_p4_full_completion": False,
        "object_task_claim": False,
        "learned_policy_claim": True,
        "learned_policy_scope": (
            f"order3_free_flight_{task_mode.value}"
            if report.get("order3_rollout_condition") is not None
            else "order3_free_flight_takeoff_hover"
        ),
    }
    for key, value in artifact_expected.items():
        if artifacts.get(key) != value:
            raise SchemaValidationError(
                f"Order3 learned artifact provenance/scope mismatch: {key}"
            )
    for key in (
        "contact_task_claim",
        "dock_motion_claim",
        "p4_full_completion_claim",
    ):
        if (
            report.get(key, False) is not False
            or artifacts.get(key, False) is not False
        ):
            raise SchemaValidationError(
                f"Order3 learned report cannot claim out-of-scope {key}"
            )


def _checkpoint_contract(
    report: dict[str, Any],
    *,
    physical_model: PhysicalModel,
    expected_checkpoint_sha256: str | None,
) -> tuple[str, Order3PolicyCheckpointMetadata]:
    checkpoint_hash = report.get("order3_pi_l_checkpoint_sha256")
    if not isinstance(checkpoint_hash, str) or not _is_sha256(checkpoint_hash):
        raise SchemaValidationError("Order3 learned report checkpoint hash is invalid")
    if expected_checkpoint_sha256 is not None:
        if not _is_sha256(expected_checkpoint_sha256):
            raise SchemaValidationError(
                "expected Order3 checkpoint hash must be sha256"
            )
        if checkpoint_hash != expected_checkpoint_sha256:
            raise SchemaValidationError(
                "Order3 learned report checkpoint hash mismatch"
            )
    try:
        metadata = Order3PolicyCheckpointMetadata.from_dict(
            report.get("order3_pi_l_checkpoint_metadata")
        )
    except (SchemaValidationError, TypeError) as exc:
        raise SchemaValidationError(
            "Order3 learned report checkpoint metadata is invalid"
        ) from exc
    if metadata.checkpoint_version != ORDER3_CHECKPOINT_VERSION:
        raise SchemaValidationError(
            "Order3 learned behavior checkpoint version mismatch"
        )
    if metadata.physical_model_hash != physical_model.stable_hash():
        raise SchemaValidationError(
            "Order3 learned checkpoint PhysicalModel hash mismatch"
        )
    # Metadata.validate() rejects privileged actor wrench inputs, contact/internal
    # wrench outputs, and allocator-owned vectoring outputs independently.
    return checkpoint_hash, metadata


def _aligned_rows(
    report: dict[str, Any],
    *,
    sanitize_privileged_contact_wrench: bool,
) -> _AlignedRows:
    raw_sequences: dict[str, list[Any]] = {}
    for key in (_RUNTIME_ROWS, _POLICY_ROWS, _CONTROLLER_ROWS, _ACTUATOR_ROWS):
        value = report.get(key)
        if not isinstance(value, list):
            raise SchemaValidationError(
                f"Order3 learned report is missing aligned sequence {key!r}"
            )
        raw_sequences[key] = value
    lengths = {len(value) for value in raw_sequences.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise SchemaValidationError(
            f"Order3 learned aligned sequence lengths mismatch: {sorted(lengths)}"
        )
    count = next(iter(lengths))
    if _non_negative_int(report, "random_morphology_takeoff_steps") != count:
        raise SchemaValidationError(
            "Order3 learned report step count does not match aligned rows"
        )
    try:
        raw_observations = [
            RuntimeObservation.from_dict(item) for item in raw_sequences[_RUNTIME_ROWS]
        ]
        policy_commands = [
            PolicyCommand.from_dict(item) for item in raw_sequences[_POLICY_ROWS]
        ]
        controller_commands = [
            ControllerCommand.from_dict(item)
            for item in raw_sequences[_CONTROLLER_ROWS]
        ]
        actuator_records = [
            IsaacActuatorTargetRecord.from_dict(item)
            for item in raw_sequences[_ACTUATOR_ROWS]
        ]
        final_raw_observation = RuntimeObservation.from_dict(
            report.get(_FINAL_OBSERVATION)
        )
    except (SchemaValidationError, TypeError, ValueError) as exc:
        raise SchemaValidationError(
            "Order3 learned report contains an invalid typed aligned row"
        ) from exc

    simulation_dt = _finite_positive(
        report,
        "random_morphology_takeoff_sim_dt_s",
    )
    graph_id = raw_observations[0].morphology_graph.graph_id
    actor_observations: list[RuntimeObservation] = []
    removed_counts: list[int] = []
    for index, (observation, policy, controller, actuator) in enumerate(
        zip(
            raw_observations,
            policy_commands,
            controller_commands,
            actuator_records,
            strict=True,
        )
    ):
        expected_time = index * simulation_dt
        if not math.isclose(
            observation.time_s, expected_time, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise SchemaValidationError(
                "Order3 learned runtime observation time/index mismatch"
            )
        if observation.morphology_graph.graph_id != graph_id:
            raise SchemaValidationError("Order3 learned observations cross graph ids")
        if observation.object_states:
            raise SchemaValidationError(
                "Order3 free-flight online observations cannot contain object states"
            )
        if policy.control_contract_version != POLICY_COMMAND_CONTRACT_CENTROIDAL:
            raise SchemaValidationError("Order3 learned policy command is not v2")
        _validate_policy_authority(policy)
        if controller.control_contract_version != POLICY_COMMAND_CONTRACT_CENTROIDAL:
            raise SchemaValidationError("Order3 learned controller command is not v2")
        if controller.dock_mechanism_commands:
            raise SchemaValidationError(
                "Order3 free-flight controller cannot command dock mechanism motion"
            )
        if actuator.backend != "isaac_lab":
            raise SchemaValidationError(
                "Order3 online actuator provenance is not Isaac Lab"
            )
        if actuator.morphology_graph_id != graph_id or actuator.command_index != index:
            raise SchemaValidationError("Order3 actuator row alignment mismatch")
        if not math.isclose(
            actuator.time_s, observation.time_s, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise SchemaValidationError("Order3 actuator time alignment mismatch")
        actor_observation, removed = _actor_view_observation(
            observation,
            sanitize_privileged_contact_wrench=sanitize_privileged_contact_wrench,
        )
        actor_observations.append(actor_observation)
        removed_counts.append(removed)

    expected_final_time = count * simulation_dt
    if not math.isclose(
        final_raw_observation.time_s,
        expected_final_time,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise SchemaValidationError(
            "Order3 final runtime observation is not the post-step episode boundary"
        )
    if final_raw_observation.morphology_graph.graph_id != graph_id:
        raise SchemaValidationError("Order3 final observation graph id mismatch")
    if final_raw_observation.object_states:
        raise SchemaValidationError(
            "Order3 free-flight final observation cannot contain object states"
        )
    final_actor_observation, final_removed = _actor_view_observation(
        final_raw_observation,
        sanitize_privileged_contact_wrench=sanitize_privileged_contact_wrench,
    )
    _validate_counter_alignment(report, controller_commands, actuator_records)
    return _AlignedRows(
        raw_observations=raw_observations,
        actor_observations=actor_observations,
        policy_commands=policy_commands,
        controller_commands=controller_commands,
        actuator_records=actuator_records,
        final_raw_observation=final_raw_observation,
        final_actor_observation=final_actor_observation,
        removed_wrench_counts=removed_counts,
        final_removed_wrench_count=final_removed,
    )


def _validate_graph_and_physical_identity(
    report: dict[str, Any],
    aligned: _AlignedRows,
    *,
    physical_model: PhysicalModel,
    structural_hash: str,
) -> None:
    morphology = aligned.raw_observations[0].morphology_graph
    graph_stable_hash = morphology.stable_hash()
    for observation in [
        *aligned.raw_observations,
        aligned.final_raw_observation,
    ]:
        if observation.morphology_graph.stable_hash() != graph_stable_hash:
            raise SchemaValidationError(
                "Order3 learned report crosses morphology graphs"
            )
        if morphology_structural_hash(observation.morphology_graph) != structural_hash:
            raise SchemaValidationError(
                "Order3 learned report crosses canonical morphologies"
            )
    if report.get("random_morphology_takeoff_graph_id") != morphology.graph_id:
        raise SchemaValidationError("Order3 learned report graph id mismatch")
    if report.get("random_morphology_takeoff_morphology_hash") != graph_stable_hash:
        raise SchemaValidationError("Order3 learned report morphology hash mismatch")
    if report.get("random_morphology_takeoff_module_count") != len(morphology.modules):
        raise SchemaValidationError("Order3 learned report module count mismatch")
    if (
        report.get("random_morphology_takeoff_physical_model_hash")
        != physical_model.stable_hash()
    ):
        raise SchemaValidationError("Order3 learned report PhysicalModel hash mismatch")


def _parse_and_validate_traces(
    report: dict[str, Any],
    aligned: _AlignedRows,
    *,
    physical_model: PhysicalModel,
    config: Order3OnlineCollectorConfig,
) -> list[_Trace]:
    raw_traces = report[_TRACE_ROWS]
    traces: list[_Trace] = []
    recurrent_width: int | None = None
    previous_source_step = -1
    report_wrench = _finite_vector(
        report.get("order3_privileged_external_wrench_body"),
        6,
        "order3_privileged_external_wrench_body",
    )
    disturbance_start = _finite_non_negative(report, "order3_disturbance_start_s")
    disturbance_duration = _finite_non_negative(
        report,
        "order3_disturbance_duration_s",
    )
    for index, raw in enumerate(raw_traces):
        if not isinstance(raw, dict):
            raise SchemaValidationError(
                "Order3 learned transition trace must be a mapping"
            )
        source_step = _trace_non_negative_int(raw, "step_index")
        if source_step <= previous_source_step or source_step >= len(
            aligned.actor_observations
        ):
            raise SchemaValidationError(
                "Order3 learned trace source steps are not strictly aligned"
            )
        previous_source_step = source_step
        time_s = _finite_number(raw.get("time_s"), "trace.time_s")
        observation = aligned.actor_observations[source_step]
        if not math.isclose(time_s, observation.time_s, rel_tol=0.0, abs_tol=1.0e-9):
            raise SchemaValidationError("Order3 learned trace time is not step-aligned")
        target_pose = _finite_pose(
            raw.get("target_pose_world"), "trace.target_pose_world"
        )
        target_twist = _finite_vector(raw.get("target_twist"), 6, "trace.target_twist")
        previous_action = _finite_vector(
            raw.get("previous_action"), ORDER3_ACTION_SIZE, "trace.previous_action"
        )
        action = _finite_vector(raw.get("action"), ORDER3_ACTION_SIZE, "trace.action")
        if any(abs(value) > 1.0 for value in [*previous_action, *action]):
            raise SchemaValidationError("Order3 learned trace action is not normalized")
        recurrent_in = _finite_non_empty_vector(
            raw.get("recurrent_state_in"), "trace.recurrent_state_in"
        )
        recurrent_out = _finite_non_empty_vector(
            raw.get("recurrent_state_out"), "trace.recurrent_state_out"
        )
        if len(recurrent_in) != len(recurrent_out):
            raise SchemaValidationError("Order3 learned recurrent state width mismatch")
        if recurrent_width is None:
            recurrent_width = len(recurrent_in)
        elif len(recurrent_in) != recurrent_width:
            raise SchemaValidationError(
                "Order3 learned recurrent width changes in episode"
            )
        log_prob = _finite_number(raw.get("old_log_prob"), "trace.old_log_prob")
        old_value = _finite_number(raw.get("old_value"), "trace.old_value")
        disturbance = _finite_vector(
            raw.get("privileged_disturbance_body"),
            6,
            "trace.privileged_disturbance_body",
        )
        if (
            raw.get("policy_applied") is not True
            or raw.get("fallback_reason") is not None
        ):
            raise SchemaValidationError(
                "Order3 on-policy trace contains deterministic fallback behavior"
            )
        action_mean = raw.get("action_mean")
        if action_mean is not None:
            _finite_vector(action_mean, ORDER3_ACTION_SIZE, "trace.action_mean")
        expected_disturbance = _active_disturbance(
            time_s,
            wrench_body=report_wrench,
            start_s=disturbance_start,
            duration_s=disturbance_duration,
        )
        if not _vectors_close(
            disturbance, expected_disturbance, config.sequence_tolerance
        ):
            raise SchemaValidationError(
                "Order3 privileged disturbance trace does not match applied report schedule"
            )
        if index == 0:
            if not _vectors_close(
                previous_action,
                [0.0] * ORDER3_ACTION_SIZE,
                config.sequence_tolerance,
            ):
                raise SchemaValidationError(
                    "Order3 first policy decision previous action is not reset"
                )
            if not _vectors_close(
                recurrent_in,
                [0.0] * len(recurrent_in),
                config.sequence_tolerance,
            ):
                raise SchemaValidationError(
                    "Order3 first policy decision recurrent state is not reset"
                )
        else:
            previous = traces[-1]
            if not _vectors_close(
                previous_action, previous.action, config.sequence_tolerance
            ):
                raise SchemaValidationError(
                    "Order3 trace previous action chain is broken"
                )
            if not _vectors_close(
                recurrent_in,
                previous.recurrent_state_out,
                config.sequence_tolerance,
            ):
                raise SchemaValidationError(
                    "Order3 trace recurrent-state chain is broken"
                )
        command = aligned.policy_commands[source_step]
        if command.desired_body_pose != target_pose:
            raise SchemaValidationError(
                "Order3 trace target pose was not passed through completely"
            )
        _validate_free_flight_joint_hold(
            command,
            observation,
            physical_model=physical_model,
            tolerance=config.joint_hold_tolerance,
        )
        traces.append(
            _Trace(
                source_step=source_step,
                time_s=time_s,
                target_pose_world=target_pose,
                target_twist=target_twist,
                previous_action=previous_action,
                action=action,
                recurrent_state_in=recurrent_in,
                recurrent_state_out=recurrent_out,
                old_log_prob=log_prob,
                old_value=old_value,
                privileged_disturbance_body=disturbance,
            )
        )

    for trace_index, trace in enumerate(traces):
        interval_end = (
            traces[trace_index + 1].source_step
            if trace_index + 1 < len(traces)
            else len(aligned.policy_commands)
        )
        reference = aligned.policy_commands[trace.source_step].to_dict()
        for command in aligned.policy_commands[trace.source_step : interval_end]:
            if command.to_dict() != reference:
                raise SchemaValidationError(
                    "Order3 policy command changed between recorded policy decisions"
                )
    return traces


def _validate_behavior_replay(
    checkpoint_path: str | Path,
    *,
    expected_checkpoint_sha256: str,
    traces: Sequence[_Trace],
    aligned: _AlignedRows,
    physical_model: PhysicalModel,
    tolerance: float,
) -> None:
    """Replay the persisted actor inputs against the behavior checkpoint.

    This turns causal observation alignment into an artifact-level invariant:
    the stored action, log probability, value, and recurrent chain must be
    exactly reproducible from the observation that will enter PPO.
    """

    loaded = load_order3_policy_checkpoint(
        checkpoint_path,
        device="cpu",
        expected_sha256=expected_checkpoint_sha256,
    )
    if loaded.metadata.physical_model_hash != physical_model.stable_hash():
        raise SchemaValidationError(
            "Order3 behavior replay checkpoint PhysicalModel hash mismatch"
        )
    model = loaded.model.eval()
    hidden = model.initial_state(1, device=torch.device("cpu"))
    builder = RigidBodyControlModelBuilder()
    with torch.no_grad():
        for trace in traces:
            observation = aligned.actor_observations[trace.source_step]
            morphology = observation.morphology_graph
            control_model = builder.build(morphology, physical_model, observation)
            features = order3_actor_feature_vector(
                observation,
                control_model,
                target_pose_world=trace.target_pose_world,
                target_twist=trace.target_twist,
                max_modules=loaded.config.max_modules,
            )
            recorded_hidden = torch.tensor(
                [trace.recurrent_state_in], dtype=torch.float32
            )
            if not torch.allclose(
                hidden,
                recorded_hidden,
                rtol=0.0,
                atol=tolerance,
            ):
                raise SchemaValidationError(
                    "Order3 behavior replay recurrent input mismatch"
                )
            step = model.step(
                [morphology],
                [observation],
                torch.tensor([features], dtype=torch.float32),
                torch.tensor([trace.previous_action], dtype=torch.float32),
                hidden,
                privileged_disturbance_body=torch.tensor(
                    [trace.privileged_disturbance_body], dtype=torch.float32
                ),
                action=torch.tensor([trace.action], dtype=torch.float32),
            )
            if not math.isclose(
                float(step.log_prob[0].item()),
                trace.old_log_prob,
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                raise SchemaValidationError(
                    "Order3 behavior replay old_log_prob mismatch"
                )
            if not math.isclose(
                float(step.value[0].item()),
                trace.old_value,
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                raise SchemaValidationError(
                    "Order3 behavior replay old_value mismatch"
                )
            recorded_out = torch.tensor(
                [trace.recurrent_state_out], dtype=torch.float32
            )
            if not torch.allclose(
                step.recurrent_state,
                recorded_out,
                rtol=0.0,
                atol=tolerance,
            ):
                raise SchemaValidationError(
                    "Order3 behavior replay recurrent output mismatch"
                )
            hidden = step.recurrent_state


def _validate_policy_authority(command: PolicyCommand) -> None:
    if any(
        (
            command.contact_tracking_bias,
            command.desired_anchor_pose_offsets,
            command.joint_position_bias,
            command.joint_velocity_bias,
        )
    ):
        raise SchemaValidationError(
            "Order3 learned policy command violates the v2 actor authority boundary"
        )


def _validate_free_flight_joint_hold(
    command: PolicyCommand,
    observation: RuntimeObservation,
    *,
    physical_model: PhysicalModel,
    tolerance: float,
) -> None:
    dock_joint_ids = {
        joint.joint_id
        for joint in physical_model.joints
        if "dock_mech_joint" in joint.joint_id and joint.joint_type != "fixed"
    }
    expected_keys: set[str] = set()
    for state in observation.module_states:
        for joint_id in dock_joint_ids:
            if joint_id in state.joint_positions:
                expected_keys.add(f"module_{state.module_id}:{joint_id}")
    if set(command.joint_position_targets) != expected_keys:
        raise SchemaValidationError(
            "Order3 free-flight joint decoder does not expose the complete dock hold set"
        )
    if set(command.joint_velocity_targets) != expected_keys:
        raise SchemaValidationError(
            "Order3 free-flight joint velocity hold set is incomplete"
        )
    if set(command.joint_torque_bias) != expected_keys:
        raise SchemaValidationError(
            "Order3 free-flight joint torque-bias hold set is incomplete"
        )
    for key in expected_keys:
        if abs(float(command.joint_position_targets[key])) > tolerance:
            raise SchemaValidationError(
                "Order3 free-flight policy commanded dock joint motion"
            )
        if abs(float(command.joint_velocity_targets[key])) > tolerance:
            raise SchemaValidationError(
                "Order3 free-flight policy commanded nonzero dock joint velocity"
            )
        if abs(float(command.joint_torque_bias[key])) > tolerance:
            raise SchemaValidationError(
                "Order3 free-flight policy commanded nonzero dock joint torque bias"
            )


def _actor_view_observation(
    observation: RuntimeObservation,
    *,
    sanitize_privileged_contact_wrench: bool,
) -> tuple[RuntimeObservation, int]:
    data = observation.to_dict()
    removed = 0
    for contact in data.get("contact_states", []):
        forbidden = _find_forbidden_contact_metadata(contact.get("metadata", {}))
        if forbidden is not None:
            raise SchemaValidationError(
                f"Order3 contact metadata contains privileged field {forbidden!r}"
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
    if isinstance(value, Mapping):
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


def _validate_counter_alignment(
    report: dict[str, Any],
    controllers: Sequence[ControllerCommand],
    actuators: Sequence[IsaacActuatorTargetRecord],
) -> None:
    expected = {
        "random_morphology_takeoff_qp_infeasible_count": sum(
            not command.controller_status.qp_feasible for command in controllers
        ),
        "random_morphology_takeoff_missing_actuator_count": sum(
            len(record.missing_actuators) for record in actuators
        ),
        "random_morphology_takeoff_unsupported_actuator_count": sum(
            len(record.unsupported_actuators) for record in actuators
        ),
        "random_morphology_takeoff_clipped_target_count": sum(
            len(record.clipped_targets) for record in actuators
        ),
    }
    for key, value in expected.items():
        if _non_negative_int(report, key) != value:
            raise SchemaValidationError(
                f"Order3 learned report safety counter is not row-aligned: {key}"
            )


def _interval_safety(
    controllers: Sequence[ControllerCommand],
    actuators: Sequence[IsaacActuatorTargetRecord],
    *,
    report: dict[str, Any],
    is_last: bool,
) -> dict[str, bool]:
    qp_feasible = all(
        command.controller_status.qp_feasible
        and command.controller_status.status not in {"infeasible", "fault"}
        for command in controllers
    )
    unsupported = any(
        record.unsupported_actuators
        or record.missing_actuators
        or float(record.metrics.get("application_unresolved_target_count", 0.0)) > 0.0
        for record in actuators
    )
    saturated = any(record.clipped_targets for record in actuators) or any(
        command.controller_status.metrics.get("clipped", 0.0) > 0.0
        for command in controllers
    )
    return {
        "qp_feasible": qp_feasible,
        "hard_collision": bool(
            is_last
            and report.get(
                "random_morphology_takeoff_exact_cross_module_collision_passed"
            )
            is not True
        ),
        "non_finite_state": bool(
            is_last and report.get("random_morphology_takeoff_finite_state") is not True
        ),
        "unsupported_actuator": unsupported,
        "actuator_saturated": saturated,
    }


def _tracking_probe(
    control_model: RigidBodyControlModel,
    trace: _Trace,
    *,
    module_count: int,
    task_mode: Order3TaskMode,
    settled_pose: Sequence[float] | None,
    hover_height_delta: float | None,
    config: Order3FreeFlightRewardConfig,
) -> float:
    result = compute_order3_free_flight_reward(
        Order3FreeFlightStep(
            module_count=module_count,
            task_mode=task_mode,
            centroidal_pose_world=control_model.body_pose_world,
            centroidal_twist_world=list(control_model.body_twist_world),
            target_pose_world=trace.target_pose_world,
            target_twist_world=list(trace.target_twist),
            takeoff_height_gain_ratio=(
                max(
                    0.0,
                    (
                        float(control_model.body_pose_world[2])
                        - float(settled_pose[2])
                    )
                    / float(hover_height_delta),
                )
                if task_mode == Order3TaskMode.TAKEOFF
                and settled_pose is not None
                and hover_height_delta is not None
                else None
            ),
        ),
        config=config,
    )
    return result.tracking_cost


def _advance_tolerance_dwell(
    initial_dwell_s: float,
    *,
    trace: _Trace,
    start_step: int,
    outcome_step: int,
    control_models: Sequence[RigidBodyControlModel],
    observation_times: Sequence[float],
    final_control_model: RigidBodyControlModel,
    final_time_s: float,
    target_is_final: bool,
    config: Order3FreeFlightRewardConfig,
) -> float:
    dwell = initial_dwell_s
    previous_time = observation_times[start_step]
    samples: list[tuple[float, RigidBodyControlModel]] = [
        (observation_times[index], control_models[index])
        for index in range(start_step + 1, min(outcome_step + 1, len(control_models)))
    ]
    if outcome_step == len(control_models):
        samples.append((final_time_s, final_control_model))
    for sample_time, model in samples:
        delta = sample_time - previous_time
        if delta < -1.0e-12:
            raise SchemaValidationError("Order3 reward outcome time moves backwards")
        previous_time = sample_time
        if target_is_final and _within_tolerance(
            model,
            target_pose=trace.target_pose_world,
            target_twist=trace.target_twist,
            config=config,
        ):
            dwell += max(delta, 0.0)
        else:
            dwell = 0.0
    return dwell


def _within_tolerance(
    model: RigidBodyControlModel,
    *,
    target_pose: Sequence[float],
    target_twist: Sequence[float],
    config: Order3FreeFlightRewardConfig,
) -> bool:
    return bool(
        _norm_difference(model.body_pose_world[:3], target_pose[:3])
        <= config.success_position_threshold_m
        and _quaternion_angle(model.body_pose_world[3:7], target_pose[3:7])
        <= config.success_attitude_threshold_rad
        and _norm_difference(model.body_twist_world[:3], target_twist[:3])
        <= config.success_linear_velocity_threshold_mps
        and _norm_difference(model.body_twist_world[3:6], target_twist[3:6])
        <= config.success_angular_velocity_threshold_rad_s
    )


def _interval_energy(commands: Sequence[ControllerCommand]) -> float:
    values: list[float] = []
    for command in commands:
        normalized = [
            (float(value) / 20.0) ** 2 for value in command.rotor_thrusts_n.values()
        ]
        values.append(
            min(sum(normalized) / len(normalized), 1.0) if normalized else 0.0
        )
    return min(sum(values) / len(values), 1.0) if values else 0.0


def _normalized_action_delta(
    previous: Sequence[float], action: Sequence[float]
) -> float:
    return min(
        sum(
            abs(float(value) - float(old))
            for old, value in zip(previous, action, strict=True)
        )
        / (2.0 * len(action)),
        1.0,
    )


def _validate_reward_contract(
    report: dict[str, Any],
    config: Order3FreeFlightRewardConfig,
    *,
    rollout_condition: Order3RolloutCondition | None,
) -> None:
    expected = {
        "random_morphology_takeoff_position_error_threshold_m": config.success_position_threshold_m,
        "random_morphology_takeoff_attitude_error_threshold_rad": config.success_attitude_threshold_rad,
        "random_morphology_takeoff_hover_linear_speed_threshold_mps": config.success_linear_velocity_threshold_mps,
        "random_morphology_takeoff_hover_angular_speed_threshold_rad_s": config.success_angular_velocity_threshold_rad_s,
        "random_morphology_takeoff_hover_hold_required_s": config.success_hold_duration_s,
        "random_morphology_takeoff_min_height_gain_ratio": config.takeoff_min_height_gain_ratio,
    }
    for key, expected_value in expected.items():
        actual = _finite_number(report.get(key), key)
        if not math.isclose(actual, expected_value, rel_tol=0.0, abs_tol=1.0e-12):
            raise SchemaValidationError(
                f"Order3 online reward threshold does not match source report: {key}"
            )
    if rollout_condition is not None and not math.isclose(
        float(rollout_condition.hold_s),
        float(config.success_hold_duration_s),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise SchemaValidationError(
            "Order3 rollout condition hold duration differs from the reward contract"
        )


def _active_disturbance(
    time_s: float,
    *,
    wrench_body: Sequence[float],
    start_s: float,
    duration_s: float,
) -> list[float]:
    active = time_s + 1.0e-12 >= start_s and (
        duration_s <= 0.0 or time_s < start_s + duration_s - 1.0e-12
    )
    return [float(value) for value in wrench_body] if active else [0.0] * 6


def _default_episode_id(
    report: dict[str, Any],
    *,
    structural_hash: str,
    checkpoint_sha256: str,
    split: DatasetSplit,
) -> str:
    seed = {
        "collector_version": ORDER3_ONLINE_COLLECTOR_VERSION,
        "structural_hash": structural_hash,
        "checkpoint_sha256": checkpoint_sha256,
        "split": split.value,
        "backend_config_hash": report.get(
            "random_morphology_takeoff_backend_config_hash"
        ),
        "physical_model_hash": report.get(
            "random_morphology_takeoff_physical_model_hash"
        ),
        "step_count": report.get("random_morphology_takeoff_steps"),
        "trace_steps": [
            trace.get("step_index") for trace in report.get(_TRACE_ROWS, [])
        ],
        "rollout_condition_hash": report.get("order3_rollout_condition_hash"),
        "rollout_seed": report.get("order3_rollout_seed_applied"),
    }
    return f"order3-online-{stable_hash(seed)[:16]}"


def _finite_pose(value: Any, path: str) -> tuple[float, ...]:
    values = _finite_vector(value, 7, path)
    if math.sqrt(sum(item * item for item in values[3:7])) <= 1.0e-12:
        raise SchemaValidationError(f"{path} quaternion must have non-zero norm")
    return tuple(values)


def _finite_vector(value: Any, width: int, path: str) -> list[float]:
    if not isinstance(value, (list, tuple)):
        raise SchemaValidationError(f"{path} must be a vector")
    require_len(value, width, path)
    output = [_finite_number(item, path) for item in value]
    return output


def _finite_non_empty_vector(value: Any, path: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or not value:
        raise SchemaValidationError(f"{path} must be a non-empty vector")
    return [_finite_number(item, path) for item in value]


def _finite_number(value: Any, path: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise SchemaValidationError(f"{path} must be finite")
    return float(value)


def _finite_positive(report: Mapping[str, Any], key: str) -> float:
    value = _finite_number(report.get(key), key)
    if value <= 0.0:
        raise SchemaValidationError(f"{key} must be positive")
    return value


def _finite_non_negative(report: Mapping[str, Any], key: str) -> float:
    value = _finite_number(report.get(key), key)
    if value < 0.0:
        raise SchemaValidationError(f"{key} must be non-negative")
    return value


def _non_negative_int(report: Mapping[str, Any], key: str) -> int:
    value = report.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SchemaValidationError(f"{key} must be a non-negative integer")
    return value


def _trace_non_negative_int(trace: Mapping[str, Any], key: str) -> int:
    value = trace.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SchemaValidationError(f"trace.{key} must be a non-negative integer")
    return value


def _vectors_close(
    left: Sequence[float], right: Sequence[float], tolerance: float
) -> bool:
    return len(left) == len(right) and all(
        abs(float(lhs) - float(rhs)) <= tolerance
        for lhs, rhs in zip(left, right, strict=True)
    )


def _poses_close(
    left: Sequence[float],
    right: Sequence[float],
    *,
    position_tolerance: float,
    attitude_tolerance: float,
) -> bool:
    return bool(
        _norm_difference(left[:3], right[:3]) <= position_tolerance
        and _quaternion_angle(left[3:7], right[3:7]) <= attitude_tolerance
    )


def _norm_difference(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(
        sum(
            (float(lhs) - float(rhs)) ** 2 for lhs, rhs in zip(left, right, strict=True)
        )
    )


def _quaternion_angle(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise SchemaValidationError("Order3 quaternion must have non-zero norm")
    dot = sum(
        float(lhs) * float(rhs) / (left_norm * right_norm)
        for lhs, rhs in zip(left, right, strict=True)
    )
    return 2.0 * math.acos(min(max(abs(dot), 0.0), 1.0))


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = [
    "ORDER3_ONLINE_COLLECTOR_VERSION",
    "Order3OnlineCollectionResult",
    "Order3OnlineCollectorConfig",
    "collect_order3_online_transitions",
]
