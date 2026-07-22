from __future__ import annotations

"""Task-adapter and phase-aware reward composition for Order 9."""

import math
from dataclasses import dataclass, field
from typing import Protocol

from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_non_empty
from amsrr.schemas.policies import ControllerCommand
from amsrr.schemas.runtime import RuntimeObservation
from amsrr.schemas.task_spec import TaskSpec, TaskType
from amsrr.simulation.order9_object_task_runtime import (
    ORDER9_OBJECT_TASK_ACTOR_PHASE_LABELS,
)
from amsrr.training.p4_3_reward import (
    P4_3RewardConfig,
    compute_p4_3_step_reward,
    compute_p4_3_terminal_reward,
)


ORDER9_REWARD_ENGINE_VERSION = "order9_phase_task_adapter_reward_v1"

COMMON_SAFETY_TERMS: tuple[str, ...] = (
    "energy",
    "qp_residual",
    "slip",
    "collision",
    "actuator_saturation",
)


@dataclass
class ActorPhaseContext(SchemaBase):
    task_adapter_id: str
    phase_label: str
    phase_index: int
    phase_count: int
    progress_ratio: float

    def validate(self) -> None:
        require_non_empty(self.task_adapter_id, "ActorPhaseContext.task_adapter_id")
        require_non_empty(self.phase_label, "ActorPhaseContext.phase_label")
        if self.phase_count < 1 or not 0 <= self.phase_index < self.phase_count:
            raise SchemaValidationError(
                "ActorPhaseContext phase index must be inside the declared phase count"
            )
        if not math.isfinite(self.progress_ratio) or not 0.0 <= self.progress_ratio <= 1.0:
            raise SchemaValidationError("ActorPhaseContext.progress_ratio must be in [0, 1]")

    def actor_features(self) -> list[float]:
        """Actor-visible phase one-hot plus progress; contains no raw contact truth."""

        one_hot = [0.0] * self.phase_count
        one_hot[self.phase_index] = 1.0
        return [*one_hot, float(self.progress_ratio)]


@dataclass
class Order9RewardOutput(SchemaBase):
    reward: float
    terms: dict[str, float]
    active_task_terms: list[str]
    phase_context: ActorPhaseContext
    raw_contact_used_as_actor_input: bool = False
    engine_version: str = ORDER9_REWARD_ENGINE_VERSION

    def validate(self) -> None:
        if not math.isfinite(self.reward):
            raise SchemaValidationError("Order9RewardOutput.reward must be finite")
        if any(not math.isfinite(float(value)) for value in self.terms.values()):
            raise SchemaValidationError("Order9RewardOutput.terms must be finite")
        if self.raw_contact_used_as_actor_input:
            raise SchemaValidationError(
                "Order9 actor observations must not contain privileged raw contact truth"
            )
        require_non_empty(self.engine_version, "Order9RewardOutput.engine_version")


class Order9TaskRewardAdapter(Protocol):
    adapter_id: str
    task_type: TaskType

    @property
    def phase_labels(self) -> tuple[str, ...]:
        ...

    def canonical_phase(self, phase_label: str) -> str:
        ...

    def active_task_terms(self, canonical_phase: str) -> tuple[str, ...]:
        ...

    def terminal_reward(
        self,
        *,
        task_spec: TaskSpec,
        observation: RuntimeObservation,
        config: P4_3RewardConfig,
        signals: dict[str, bool | None],
    ) -> dict[str, float]:
        ...


@dataclass(frozen=True)
class ObjectGraspCarryRewardAdapter:
    adapter_id: str = "object_grasp_carry_v1"
    task_type: TaskType = TaskType.OBJECT_GRASP_CARRY

    @property
    def phase_labels(self) -> tuple[str, ...]:
        return ORDER9_OBJECT_TASK_ACTOR_PHASE_LABELS

    def canonical_phase(self, phase_label: str) -> str:
        aliases = {
            "reset": "approach",
            "approach": "approach",
            "approach_object": "approach",
            "contact_acquisition": "establish_contact",
            "establish_object_contacts": "establish_contact",
            "apply_grasp_wrench": "apply_wrench",
            "lift": "lift",
            "lift_object": "lift",
            "transport": "transport",
            "transport_object": "transport",
            "place": "place",
            "place_object": "place",
            "release": "release",
            "release_contacts": "release",
            "retreat": "retreat",
            "settle": "settle",
            "complete": "complete",
            "safe_hold": "safe_hold",
            # Legacy simplified rollout labels.
            "grasp": "establish_contact",
            "carry": "transport",
            "done": "complete",
        }
        try:
            return aliases[phase_label]
        except KeyError as exc:
            raise SchemaValidationError(
                f"object_grasp_carry reward adapter received unknown phase {phase_label!r}"
            ) from exc

    def active_task_terms(self, canonical_phase: str) -> tuple[str, ...]:
        by_phase = {
            "approach": ("centroidal_stability",),
            "establish_contact": ("grasp_maintenance", "centroidal_stability"),
            "apply_wrench": ("grasp_maintenance", "centroidal_stability"),
            "lift": (
                "object_goal_progress",
                "grasp_maintenance",
                "centroidal_stability",
            ),
            "transport": (
                "object_goal_progress",
                "object_pose_accuracy",
                "grasp_maintenance",
                "centroidal_stability",
            ),
            "place": (
                "object_goal_progress",
                "object_pose_accuracy",
                "grasp_maintenance",
                "centroidal_stability",
            ),
            "release": ("object_pose_accuracy", "centroidal_stability"),
            "retreat": ("object_pose_accuracy", "centroidal_stability"),
            "settle": ("object_pose_accuracy", "centroidal_stability"),
            "complete": ("object_pose_accuracy",),
            "safe_hold": (),
        }
        return by_phase[canonical_phase]

    def terminal_reward(
        self,
        *,
        task_spec: TaskSpec,
        observation: RuntimeObservation,
        config: P4_3RewardConfig,
        signals: dict[str, bool | None],
    ) -> dict[str, float]:
        return compute_p4_3_terminal_reward(
            task_spec=task_spec,
            observation=observation,
            release_valid=signals.get("release_valid"),
            object_dropped=signals.get("object_dropped"),
            hard_collision=signals.get("hard_collision"),
            timeout=signals.get("timeout"),
            qp_infeasible_terminal=signals.get("qp_infeasible_terminal"),
            config=config,
        )


class Order9RewardEngine:
    """Compose invariant safety terms with an explicit task/phase adapter."""

    def __init__(
        self,
        *,
        config: P4_3RewardConfig | None = None,
        adapters: tuple[Order9TaskRewardAdapter, ...] | None = None,
    ) -> None:
        self.config = config or P4_3RewardConfig()
        configured = adapters or (ObjectGraspCarryRewardAdapter(),)
        self.adapters = {adapter.task_type: adapter for adapter in configured}
        if len(self.adapters) != len(configured):
            raise ValueError("Order9 reward adapters must have unique task types")

    def step(
        self,
        *,
        task_spec: TaskSpec,
        observation: RuntimeObservation,
        previous_observation: RuntimeObservation | None = None,
        controller_command: ControllerCommand | None = None,
        actuator_target_record: dict[str, object] | None = None,
        state_transition_available: bool = True,
    ) -> Order9RewardOutput:
        adapter = self._adapter(task_spec)
        raw_phase = observation.task_progress.phase_label or adapter.phase_labels[0]
        canonical_phase = adapter.canonical_phase(raw_phase)
        phase_index = adapter.phase_labels.index(canonical_phase)
        progress = observation.task_progress.progress_ratio
        progress_ratio = 0.0 if progress is None else min(max(float(progress), 0.0), 1.0)
        phase_context = ActorPhaseContext(
            task_adapter_id=adapter.adapter_id,
            phase_label=canonical_phase,
            phase_index=phase_index,
            phase_count=len(adapter.phase_labels),
            progress_ratio=progress_ratio,
        )
        base = compute_p4_3_step_reward(
            task_spec=task_spec,
            observation=observation,
            previous_observation=previous_observation,
            controller_command=controller_command,
            actuator_target_record=actuator_target_record,
            state_transition_available=state_transition_available,
            config=self.config,
        )
        task_terms = adapter.active_task_terms(canonical_phase)
        weighted = {
            "object_goal_progress": base["weighted_object_goal_progress"],
            "object_pose_accuracy": base["weighted_object_pose_accuracy"],
            "grasp_maintenance": base["weighted_grasp_maintenance"],
            "centroidal_stability": base["weighted_centroidal_stability"],
            "energy": base["weighted_energy_penalty"],
            "qp_residual": base["weighted_qp_residual_penalty"],
            "slip": base["weighted_slip_penalty"],
            "collision": base["weighted_collision_penalty"],
            "actuator_saturation": base["weighted_actuator_saturation_penalty"],
        }
        active = set(COMMON_SAFETY_TERMS) | set(task_terms)
        terms = {
            f"weighted_{name}": float(value if name in active else 0.0)
            for name, value in weighted.items()
        }
        terms.update(
            {
                "phase_index": float(phase_index),
                "phase_count": float(len(adapter.phase_labels)),
                "phase_progress_ratio": progress_ratio,
                "raw_contact_actor_input": 0.0,
                "raw_contact_reward_or_safety_only": 1.0,
            }
        )
        reward = sum(
            value for name, value in terms.items() if name.startswith("weighted_")
        )
        return Order9RewardOutput(
            reward=reward,
            terms=terms,
            active_task_terms=list(task_terms),
            phase_context=phase_context,
        )

    def terminal(
        self,
        *,
        task_spec: TaskSpec,
        observation: RuntimeObservation,
        **signals: bool | None,
    ) -> dict[str, float]:
        return self._adapter(task_spec).terminal_reward(
            task_spec=task_spec,
            observation=observation,
            config=self.config,
            signals=signals,
        )

    def _adapter(self, task_spec: TaskSpec) -> Order9TaskRewardAdapter:
        try:
            return self.adapters[task_spec.task_type]
        except KeyError as exc:
            raise SchemaValidationError(
                f"Order9 reward has no task adapter for {task_spec.task_type.value!r}"
            ) from exc
