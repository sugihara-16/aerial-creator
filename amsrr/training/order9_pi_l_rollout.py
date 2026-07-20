from __future__ import annotations

"""Causal online rollout collector for the learned Order 9 ``pi_L``.

The collector buffers records until a physical episode boundary so a safety
fallback can terminate the learned actor's GAE segment without inventing a
behavior log-probability or crediting fallback rewards to the actor.
"""

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from amsrr.policies.low_level_policy_base import LowLevelPolicyContext, select_active_knot
from amsrr.policies.morphology_conditioned_low_level_policy import Order3PolicyInference
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.datasets import (
    DatasetSplit,
    LowLevelControlRecord,
    PolicyBehaviorTrace,
    StageDecisionMasks,
)
from amsrr.schemas.policies import ControllerCommand, InteractionKnot, PolicyCommand
from amsrr.schemas.runtime import RuntimeObservation
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.order9_ppo import order9_pi_l_behavior_trace_from_inference
from amsrr.training.order9_reward import Order9RewardEngine


ORDER9_PI_L_ROLLOUT_VERSION = "order9_pi_l_causal_segment_rollout_v1"


@dataclass(frozen=True)
class Order9PiLRolloutResult:
    physical_episode_id: str
    records: tuple[LowLevelControlRecord, ...]
    learned_action_count: int
    fallback_action_count: int
    gae_segment_count: int
    terminal_reward_credited_to_actor: bool


@dataclass
class _PendingLearnedAction:
    segment_episode_id: str
    step_index: int
    actor_observation: RuntimeObservation
    reward_observation: RuntimeObservation
    trajectory_record_id: str
    active_trajectory_index: int
    active_knot_index: int
    active_knot: InteractionKnot
    task_type: str
    task_adapter_id: str
    phase_index: int
    phase_count: int
    behavior_trace: PolicyBehaviorTrace
    policy_command: PolicyCommand
    controller_command: ControllerCommand
    actuator_target_record: dict[str, Any]


class Order9PiLEpisodeCollector:
    """Collect exact learned actions and exclude deterministic fallback actions."""

    def __init__(
        self,
        *,
        physical_episode_id: str,
        task_spec: TaskSpec,
        split: DatasetSplit,
        checkpoint_sha256: str,
        reward_engine: Order9RewardEngine | None = None,
    ) -> None:
        if not physical_episode_id:
            raise SchemaValidationError(
                "Order9 pi_L physical episode id must be non-empty"
            )
        _require_sha256(checkpoint_sha256)
        task_spec.validate()
        self.physical_episode_id = physical_episode_id
        self.task_spec = task_spec
        self.split = split
        self.checkpoint_sha256 = checkpoint_sha256
        self.reward_engine = reward_engine or Order9RewardEngine()
        self._latest_actor_observation: RuntimeObservation | None = None
        self._latest_reward_observation: RuntimeObservation | None = None
        self._pending: _PendingLearnedAction | None = None
        self._records: list[LowLevelControlRecord] = []
        self._current_segment_id: str | None = None
        self._current_step_index = 0
        self._segment_count = 0
        self._learned_action_count = 0
        self._fallback_action_count = 0
        self._terminal_reward_credited = False
        self._finalized = False

    @property
    def pending_action(self) -> bool:
        return self._pending is not None

    @property
    def records(self) -> tuple[LowLevelControlRecord, ...]:
        return tuple(self._records)

    def observe_state(
        self,
        *,
        actor_observation: RuntimeObservation,
        reward_observation: RuntimeObservation,
    ) -> LowLevelControlRecord | None:
        """Install a state and causally close the preceding learned action."""

        self._require_open()
        actor_observation.validate()
        reward_observation.validate()
        _require_actor_safe(actor_observation)
        _require_observation_pair(actor_observation, reward_observation)
        if self._latest_actor_observation is not None and (
            actor_observation.time_s <= self._latest_actor_observation.time_s
        ):
            raise SchemaValidationError(
                "Order9 pi_L observation times must increase strictly"
            )
        completed: LowLevelControlRecord | None = None
        if self._pending is not None:
            pending = self._pending
            reward = self.reward_engine.step(
                task_spec=self.task_spec,
                observation=reward_observation,
                previous_observation=pending.reward_observation,
                controller_command=pending.controller_command,
                actuator_target_record=pending.actuator_target_record,
                state_transition_available=True,
            )
            terms = dict(reward.terms)
            terms.update(
                {
                    "transition_start_time_s": float(
                        pending.actor_observation.time_s
                    ),
                    "transition_end_time_s": float(actor_observation.time_s),
                    "transition_dt_s": float(
                        actor_observation.time_s
                        - pending.actor_observation.time_s
                    ),
                    "privileged_reward_observation_only": 1.0,
                    "deterministic_fallback_reward_credited": 0.0,
                }
            )
            completed = LowLevelControlRecord(
                record_id=(
                    f"{pending.segment_episode_id}:step:{pending.step_index:06d}"
                ),
                episode_id=pending.segment_episode_id,
                task_id=self.task_spec.task_id,
                split=self.split,
                step_index=pending.step_index,
                time_s=pending.actor_observation.time_s,
                trajectory_record_id=pending.trajectory_record_id,
                active_trajectory_index=pending.active_trajectory_index,
                active_knot_index=pending.active_knot_index,
                runtime_observation=pending.actor_observation,
                active_knot=pending.active_knot,
                policy_command=pending.policy_command,
                controller_command=pending.controller_command,
                actuator_target_record=pending.actuator_target_record,
                reward_terms=terms,
                reward=float(reward.reward),
                terminal=False,
                stage_masks=StageDecisionMasks(low_level_control_mask=True),
                task_type=pending.task_type,
                task_adapter_id=pending.task_adapter_id,
                phase_index=pending.phase_index,
                phase_count=pending.phase_count,
                behavior_trace=pending.behavior_trace,
            )
            completed.validate()
            self._records.append(completed)
            self._current_step_index += 1
            self._pending = None
        self._latest_actor_observation = actor_observation
        self._latest_reward_observation = reward_observation
        return completed

    def record_action(
        self,
        *,
        context: LowLevelPolicyContext,
        inference: Order3PolicyInference,
        trajectory_record_id: str,
        active_trajectory_index: int,
        active_knot_index: int,
        controller_command: ControllerCommand,
        actuator_target_record: Mapping[str, Any],
        privileged_disturbance_body: Sequence[float] | None = None,
    ) -> bool:
        """Bind an applied action; return false when it was deterministic fallback."""

        self._require_open()
        if self._pending is not None:
            raise SchemaValidationError(
                "Order9 pi_L action recorded before prior transition closed"
            )
        if self._latest_actor_observation is None or self._latest_reward_observation is None:
            raise SchemaValidationError("Order9 pi_L action requires an observed pre-state")
        _require_context_state(context, self._latest_actor_observation)
        _require_phase_context(context)
        if not inference.learned_policy_applied:
            self._fallback_action_count += 1
            self._close_current_segment_for_fallback()
            return False
        if inference.fallback_reason is not None:
            raise SchemaValidationError(
                "learned pi_L inference cannot also carry a fallback reason"
            )
        if not trajectory_record_id:
            raise SchemaValidationError(
                "Order9 pi_L action requires a trajectory record id"
            )
        if min(active_trajectory_index, active_knot_index) < 0:
            raise SchemaValidationError(
                "Order9 pi_L active trajectory/knot indices must be non-negative"
            )
        behavior = order9_pi_l_behavior_trace_from_inference(
            inference,
            checkpoint_sha256=self.checkpoint_sha256,
            privileged_disturbance_body=privileged_disturbance_body,
        )
        if self._current_segment_id is None:
            self._current_segment_id = (
                f"{self.physical_episode_id}:pi_l_segment:{self._segment_count:04d}"
            )
            self._segment_count += 1
            self._current_step_index = 0
        command = PolicyCommand.from_dict(inference.command.to_dict())
        controller = ControllerCommand.from_dict(controller_command.to_dict())
        controller.validate()
        targets = json.loads(json.dumps(dict(actuator_target_record), sort_keys=True))
        active_knot = InteractionKnot.from_dict(select_active_knot(context).to_dict())
        task_type = str(context.task_type)
        task_adapter_id = str(context.task_adapter_id)
        phase_index = int(context.phase_index)
        phase_count = int(context.phase_count)
        self._pending = _PendingLearnedAction(
            segment_episode_id=self._current_segment_id,
            step_index=self._current_step_index,
            actor_observation=RuntimeObservation.from_dict(
                self._latest_actor_observation.to_dict()
            ),
            reward_observation=RuntimeObservation.from_dict(
                self._latest_reward_observation.to_dict()
            ),
            trajectory_record_id=trajectory_record_id,
            active_trajectory_index=active_trajectory_index,
            active_knot_index=active_knot_index,
            active_knot=active_knot,
            task_type=task_type,
            task_adapter_id=task_adapter_id,
            phase_index=phase_index,
            phase_count=phase_count,
            behavior_trace=behavior,
            policy_command=command,
            controller_command=controller,
            actuator_target_record=targets,
        )
        self._learned_action_count += 1
        return True

    def finalize(
        self,
        *,
        terminal: bool,
        truncated: bool = False,
        bootstrap_value: float = 0.0,
        release_valid: bool | None = None,
        object_dropped: bool | None = None,
        hard_collision: bool | None = None,
        timeout: bool | None = None,
        qp_infeasible_terminal: bool | None = None,
    ) -> Order9PiLRolloutResult:
        """Close the last learned segment after its final post-state was observed."""

        self._require_open()
        if terminal == truncated:
            raise SchemaValidationError(
                "Order9 pi_L finalization requires exactly one terminal/truncated boundary"
            )
        if self._pending is not None:
            raise SchemaValidationError(
                "Order9 pi_L collector has an unclosed action; observe its post-state first"
            )
        if not math.isfinite(float(bootstrap_value)):
            raise SchemaValidationError("Order9 pi_L bootstrap value must be finite")
        if terminal and not math.isclose(float(bootstrap_value), 0.0, abs_tol=1.0e-12):
            raise SchemaValidationError("terminal pi_L boundary cannot bootstrap")
        last = self._last_current_segment_record()
        if last is not None:
            if terminal:
                if self._latest_reward_observation is None:
                    raise SchemaValidationError(
                        "Order9 pi_L terminal reward requires a final observation"
                    )
                terminal_terms = self.reward_engine.terminal(
                    task_spec=self.task_spec,
                    observation=self._latest_reward_observation,
                    release_valid=release_valid,
                    object_dropped=object_dropped,
                    hard_collision=hard_collision,
                    timeout=timeout,
                    qp_infeasible_terminal=qp_infeasible_terminal,
                )
                if last.reward is None or last.reward_terms is None:
                    raise AssertionError("validated pi_L reward disappeared")
                terminal_reward = float(terminal_terms["terminal_reward"])
                last.reward += terminal_reward
                last.reward_terms.update(terminal_terms)
                last.reward_terms["reward"] = last.reward
                last.terminal = True
                self._terminal_reward_credited = True
            else:
                last.truncated = True
                last.bootstrap_value = float(bootstrap_value)
            last.validate()
        self._finalized = True
        return Order9PiLRolloutResult(
            physical_episode_id=self.physical_episode_id,
            records=tuple(self._records),
            learned_action_count=self._learned_action_count,
            fallback_action_count=self._fallback_action_count,
            gae_segment_count=self._segment_count,
            terminal_reward_credited_to_actor=self._terminal_reward_credited,
        )

    def _close_current_segment_for_fallback(self) -> None:
        last = self._last_current_segment_record()
        if last is not None:
            last.terminal = True
            last.truncated = False
            last.bootstrap_value = 0.0
            if last.reward_terms is not None:
                last.reward_terms["segment_ended_by_pi_l_fallback"] = 1.0
            last.validate()
        self._current_segment_id = None
        self._current_step_index = 0

    def _last_current_segment_record(self) -> LowLevelControlRecord | None:
        if self._current_segment_id is None:
            return None
        for record in reversed(self._records):
            if record.episode_id == self._current_segment_id:
                return record
        return None

    def _require_open(self) -> None:
        if self._finalized:
            raise SchemaValidationError("Order9 pi_L collector is already finalized")


def _require_actor_safe(observation: RuntimeObservation) -> None:
    if observation.contact_states:
        raise SchemaValidationError(
            "Order9 pi_L actor observation contains raw contact states"
        )
    forbidden = {
        "raw_contact",
        "contact_force",
        "contact_wrench",
        "penetration",
        "grasp_acquired",
        "hard_collision",
        "slip",
    }
    leaked = sorted(
        key
        for key in observation.task_progress.metrics
        if any(token in key.lower() for token in forbidden)
    )
    if leaked:
        raise SchemaValidationError(
            "Order9 pi_L actor observation leaked privileged metrics: "
            + ",".join(leaked)
        )


def _require_observation_pair(
    actor: RuntimeObservation,
    reward: RuntimeObservation,
) -> None:
    if not math.isclose(actor.time_s, reward.time_s, abs_tol=1.0e-9):
        raise SchemaValidationError("Order9 pi_L actor/reward observation times differ")
    if actor.morphology_graph.stable_hash() != reward.morphology_graph.stable_hash():
        raise SchemaValidationError("Order9 pi_L actor/reward morphologies differ")
    if [state.to_dict() for state in actor.module_states] != [
        state.to_dict() for state in reward.module_states
    ]:
        raise SchemaValidationError("Order9 pi_L actor/reward module states differ")
    if [state.to_dict() for state in actor.object_states] != [
        state.to_dict() for state in reward.object_states
    ]:
        raise SchemaValidationError("Order9 pi_L actor/reward object states differ")


def _require_context_state(
    context: LowLevelPolicyContext,
    observation: RuntimeObservation,
) -> None:
    if context.runtime_observation.to_dict() != observation.to_dict():
        raise SchemaValidationError(
            "Order9 pi_L policy context is not the latest actor observation"
        )
    if context.morphology_graph.stable_hash() != observation.morphology_graph.stable_hash():
        raise SchemaValidationError("Order9 pi_L context morphology mismatch")


def _require_phase_context(context: LowLevelPolicyContext) -> None:
    values = (
        context.task_type,
        context.task_adapter_id,
        context.phase_index,
        context.phase_count,
    )
    if any(value is None for value in values):
        raise SchemaValidationError(
            "Order9 pi_L learned action requires task/adapter/phase actor context"
        )
    if context.phase_count is None or context.phase_index is None or not (
        0 <= context.phase_index < context.phase_count
    ):
        raise SchemaValidationError("Order9 pi_L phase context is invalid")


def _require_sha256(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise SchemaValidationError("Order9 pi_L checkpoint hash must be SHA-256")


__all__ = [
    "ORDER9_PI_L_ROLLOUT_VERSION",
    "Order9PiLEpisodeCollector",
    "Order9PiLRolloutResult",
]
