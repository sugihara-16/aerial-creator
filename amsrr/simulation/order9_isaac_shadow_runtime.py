from __future__ import annotations

"""Concrete policy/controller binding for an isolated Order 9 Isaac scene.

Isaac-specific tensor and contact-view operations live behind
``Order9IsaacSceneAdapter``.  This class owns the production sequence that must
remain identical in the main task and counterfactual worker: restore exact
state, run the frozen checkpoint, run QPID, convert through the actuator
bridge, apply the converted record, advance physics, and reduce privileged
evidence.  No proposal projection occurs here.
"""

import math
from typing import Protocol, Sequence

from amsrr.controllers.actuator_mapping import ActuatorMapping
from amsrr.controllers.controller_base import ControllerContext, PayloadCoupling
from amsrr.controllers.isaac_controller_bridge import (
    IsaacActuatorTargetRecord,
    IsaacControllerBridge,
)
from amsrr.controllers.qpid_controller import QPIDController
from amsrr.feasibility.contact_wrench_hybrid import ShadowCollisionSample
from amsrr.feasibility.contact_wrench_shadow_metrics import MeasuredCandidateWrench
from amsrr.morphology.random_connected import morphology_structural_hash
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.policies.low_level_policy_base import LowLevelPolicyContext
from amsrr.policies.order9_low_level_runtime import Order9LowLevelRuntimePolicy
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import (
    ContactWrenchTrajectory,
    ControllerCommand,
    ControllerStatus,
    InteractionKnot,
)
from amsrr.schemas.runtime import RuntimeObservation
from amsrr.schemas.task_spec import TaskType
from amsrr.simulation.order9_object_task_runtime import (
    ORDER9_OBJECT_TASK_ADAPTER_ID,
    ORDER9_OBJECT_TASK_PHASES,
)
from amsrr.simulation.order9_object_task_state import (
    Order9IsaacStateSnapshot,
    order9_snapshot_from_mapping,
)
from amsrr.simulation.order9_runtime_state import (
    restore_order9_controller_and_policy_state,
)
from amsrr.simulation.order9_shadow_executor import (
    Order9IsaacControlStepEvidence,
)
from amsrr.simulation.order9_shadow_worker import Order9ShadowStateExport


ORDER9_ISAAC_COPIED_RUNTIME_VERSION = "order9_copied_isaac_policy_qpid_runtime_v1"


class Order9IsaacSceneAdapter(Protocol):
    """Small Isaac-only surface required by the policy/controller runtime."""

    @property
    def adapter_version(self) -> str:
        ...

    def describe(self) -> dict[str, object]:
        ...

    def restore_snapshot(self, snapshot: Order9IsaacStateSnapshot) -> None:
        ...

    def capture_snapshot(self) -> Order9IsaacStateSnapshot:
        ...

    def actor_observation(
        self,
        *,
        morphology_graph: MorphologyGraph,
        controller_status: ControllerStatus,
        elapsed_s: float,
    ) -> RuntimeObservation:
        ...

    def apply_actuator_targets(self, record: IsaacActuatorTargetRecord) -> int:
        """Apply all converted targets and return unresolved target count."""

    def step(self, dt_s: float) -> None:
        ...

    def measured_candidate_wrenches(
        self,
        *,
        context: HighLevelPolicyContext,
        active_knot: InteractionKnot,
    ) -> Sequence[MeasuredCandidateWrench]:
        ...

    def collision_evidence(
        self,
        *,
        context: HighLevelPolicyContext,
        active_knot: InteractionKnot,
    ) -> tuple[Sequence[ShadowCollisionSample], float]:
        ...

    def payload_coupling(
        self,
        *,
        active_knot: InteractionKnot,
    ) -> PayloadCoupling | None:
        ...

    def finite_state(self) -> bool:
        ...

    def reset(self) -> None:
        ...

    def close(self) -> None:
        ...


class Order9IsaacCopiedRuntime:
    """Execute one immutable ``pi_H`` proposal through the real control stack."""

    def __init__(
        self,
        *,
        scene_adapter: Order9IsaacSceneAdapter,
        morphology_graph: MorphologyGraph,
        physical_model: PhysicalModel,
        pi_l_policy: Order9LowLevelRuntimePolicy,
        controller: QPIDController,
        actuator_mapping: ActuatorMapping,
        bridge: IsaacControllerBridge | None = None,
        force_scale_n: float = 30.0,
        torque_scale_nm: float = 5.0,
    ) -> None:
        morphology_graph.validate()
        physical_model.validate()
        if actuator_mapping.graph_id != morphology_graph.graph_id:
            raise SchemaValidationError(
                "Order9 shadow actuator mapping graph identity mismatch"
            )
        if not scene_adapter.adapter_version:
            raise ValueError("Order9 Isaac scene adapter version must be non-empty")
        for name, value in (
            ("force_scale_n", force_scale_n),
            ("torque_scale_nm", torque_scale_nm),
        ):
            if not math.isfinite(float(value)) or value <= 0.0:
                raise ValueError(f"Order9 copied runtime {name} must be positive")
        self.scene_adapter = scene_adapter
        self.morphology_graph = morphology_graph
        self.physical_model = physical_model
        self.pi_l_policy = pi_l_policy
        self.controller = controller
        self.actuator_mapping = actuator_mapping
        self.bridge = bridge or IsaacControllerBridge()
        self.force_scale_n = float(force_scale_n)
        self.torque_scale_nm = float(torque_scale_nm)
        self._restored_state_digest: str | None = None
        self._restored_snapshot_hash: str | None = None
        self._previous_command: ControllerCommand | None = None
        self._command_index = 0
        self._last_status = ControllerStatus(status="ok", qp_feasible=True)
        self._trajectory: ContactWrenchTrajectory | None = None
        self._closed = False

    @property
    def runtime_version(self) -> str:
        return (
            f"{ORDER9_ISAAC_COPIED_RUNTIME_VERSION}:"
            f"{self.scene_adapter.adapter_version}"
        )

    @property
    def pi_l_checkpoint_sha256(self) -> str:
        return self.pi_l_policy.checkpoint_sha256

    @property
    def topology_structural_hash(self) -> str:
        return morphology_structural_hash(self.morphology_graph)

    def describe(self) -> dict[str, object]:
        return {
            "runtime_version": self.runtime_version,
            "topology_structural_hash": self.topology_structural_hash,
            "pi_l_checkpoint_sha256": self.pi_l_checkpoint_sha256,
            "scene": self.scene_adapter.describe(),
        }

    def restore_copied_state(self, state: Order9ShadowStateExport) -> None:
        self._require_open()
        if self._restored_state_digest is not None:
            raise RuntimeError("Order9 copied runtime requires reset before restore")
        if state.topology_structural_hash != self.topology_structural_hash:
            raise SchemaValidationError("Order9 copied runtime topology mismatch")
        snapshot = order9_snapshot_from_mapping(state.simulation_state)
        self.scene_adapter.restore_snapshot(snapshot)
        execution = restore_order9_controller_and_policy_state(
            state,
            controller=self.controller,
            pi_l_policy=self.pi_l_policy,
        )
        raw_previous = execution.get("previous_controller_command")
        self._previous_command = (
            None
            if raw_previous is None
            else ControllerCommand.from_dict(dict(raw_previous))
        )
        if self._previous_command is not None:
            self._previous_command.validate()
            self._last_status = ControllerStatus.from_dict(
                self._previous_command.controller_status.to_dict()
            )
        else:
            self._last_status = ControllerStatus(status="ok", qp_feasible=True)
        raw_index = execution.get("command_index", snapshot.command_index)
        if not isinstance(raw_index, int) or isinstance(raw_index, bool) or raw_index < 0:
            raise SchemaValidationError("Order9 copied command index is invalid")
        self._command_index = raw_index
        restored = self.scene_adapter.capture_snapshot()
        _require_same_physical_state(snapshot, restored)
        self._restored_snapshot_hash = restored.snapshot_hash
        self._restored_state_digest = state.state_digest

    def begin_trajectory(
        self,
        *,
        context: HighLevelPolicyContext,
        trajectory: ContactWrenchTrajectory,
    ) -> None:
        self._require_restored()
        if morphology_structural_hash(context.morphology_graph) != self.topology_structural_hash:
            raise SchemaValidationError("Order9 copied trajectory context topology mismatch")
        trajectory.validate()
        self._trajectory = ContactWrenchTrajectory.from_dict(trajectory.to_dict())

    def observe(
        self,
        *,
        context: HighLevelPolicyContext,
        trajectory: ContactWrenchTrajectory,
        active_knot: InteractionKnot,
        elapsed_s: float,
    ) -> Order9IsaacControlStepEvidence:
        del trajectory
        self._require_trajectory()
        return self._evidence(
            context=context,
            active_knot=active_knot,
            elapsed_s=elapsed_s,
            controller_residual=_normalized_controller_residual(
                self._last_status,
                force_scale_n=self.force_scale_n,
                fail_closed=False,
            ),
            metrics={"observation_only": 1.0},
        )

    def advance(
        self,
        *,
        context: HighLevelPolicyContext,
        trajectory: ContactWrenchTrajectory,
        active_knot: InteractionKnot,
        elapsed_s: float,
        dt_s: float,
    ) -> Order9IsaacControlStepEvidence:
        self._require_trajectory()
        if not math.isfinite(float(dt_s)) or dt_s <= 0.0:
            raise ValueError("Order9 copied runtime dt must be positive")
        observation = self.scene_adapter.actor_observation(
            morphology_graph=self.morphology_graph,
            controller_status=self._last_status,
            elapsed_s=elapsed_s,
        )
        observation.validate()
        phase_index = _phase_index(observation)
        low_context = LowLevelPolicyContext(
            runtime_observation=observation,
            morphology_graph=self.morphology_graph,
            physical_model=self.physical_model,
            contact_wrench_trajectory=trajectory,
            active_knot=active_knot,
            controller_status=self._last_status,
            task_type=TaskType.OBJECT_GRASP_CARRY.value,
            task_adapter_id=ORDER9_OBJECT_TASK_ADAPTER_ID,
            phase_index=phase_index,
            phase_count=len(ORDER9_OBJECT_TASK_PHASES),
        )
        inference = self.pi_l_policy.command_with_trace(low_context)
        command = self.controller.compute(
            ControllerContext(
                runtime_observation=observation,
                morphology_graph=self.morphology_graph,
                physical_model=self.physical_model,
                active_knot=active_knot,
                policy_command=inference.command,
                previous_command=self._previous_command,
                control_dt_s=float(dt_s),
                payload_coupling=self.scene_adapter.payload_coupling(
                    active_knot=active_knot
                ),
            )
        )
        command.validate()
        record = self.bridge.convert(
            command,
            self.actuator_mapping,
            time_s=float(observation.time_s),
            command_index=self._command_index,
        )
        record.validate()
        unresolved = self.scene_adapter.apply_actuator_targets(record)
        if not isinstance(unresolved, int) or isinstance(unresolved, bool) or unresolved < 0:
            raise RuntimeError("Order9 scene adapter returned invalid unresolved count")
        self.scene_adapter.step(float(dt_s))
        self._previous_command = ControllerCommand.from_dict(command.to_dict())
        self._last_status = ControllerStatus.from_dict(
            command.controller_status.to_dict()
        )
        self._command_index += 1
        failed = bool(
            not inference.learned_policy_applied
            or inference.fallback_reason is not None
            or unresolved
            or record.missing_actuators
            or record.unsupported_actuators
            or record.clipped_targets
        )
        residual = _normalized_controller_residual(
            self._last_status,
            force_scale_n=self.force_scale_n,
            fail_closed=failed,
        )
        return self._evidence(
            context=context,
            active_knot=active_knot,
            elapsed_s=elapsed_s + dt_s,
            controller_residual=residual,
            metrics={
                "learned_pi_l_applied": 1.0 if inference.learned_policy_applied else 0.0,
                "pi_l_fallback": 0.0 if inference.learned_policy_applied else 1.0,
                "unresolved_actuator_target_count": float(unresolved),
                "missing_actuator_count": float(len(record.missing_actuators)),
                "unsupported_actuator_count": float(
                    len(record.unsupported_actuators)
                ),
                "clipped_actuator_count": float(len(record.clipped_targets)),
            },
        )

    def reset_copied_state(self) -> None:
        self.scene_adapter.reset()
        self.controller.reset_integrators()
        self.pi_l_policy.reset()
        self._restored_state_digest = None
        self._restored_snapshot_hash = None
        self._previous_command = None
        self._command_index = 0
        self._last_status = ControllerStatus(status="ok", qp_feasible=True)
        self._trajectory = None

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._restored_state_digest is not None:
                self.reset_copied_state()
        finally:
            self.scene_adapter.close()
            self._closed = True

    def _evidence(
        self,
        *,
        context: HighLevelPolicyContext,
        active_knot: InteractionKnot,
        elapsed_s: float,
        controller_residual: float,
        metrics: dict[str, float],
    ) -> Order9IsaacControlStepEvidence:
        measured = tuple(
            self.scene_adapter.measured_candidate_wrenches(
                context=context,
                active_knot=active_knot,
            )
        )
        samples, clearance = self.scene_adapter.collision_evidence(
            context=context,
            active_knot=active_knot,
        )
        return Order9IsaacControlStepEvidence(
            controller_qp_residual=float(controller_residual),
            measured_candidate_wrenches=measured,
            collision_samples=tuple(samples),
            collision_free_clearance_m=float(clearance),
            finite_state=bool(self.scene_adapter.finite_state()),
            metrics={"elapsed_s": float(elapsed_s), **metrics},
        )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Order9 copied Isaac runtime is closed")

    def _require_restored(self) -> None:
        self._require_open()
        if self._restored_state_digest is None:
            raise RuntimeError("Order9 copied Isaac runtime has no restored state")

    def _require_trajectory(self) -> None:
        self._require_restored()
        if self._trajectory is None:
            raise RuntimeError("Order9 copied Isaac runtime has no active trajectory")


def _phase_index(observation: RuntimeObservation) -> int:
    label = observation.task_progress.phase_label
    try:
        return tuple(phase.value for phase in ORDER9_OBJECT_TASK_PHASES).index(
            str(label)
        )
    except ValueError as exc:
        raise SchemaValidationError(
            "Order9 copied observation has an unknown task phase"
        ) from exc


def _normalized_controller_residual(
    status: ControllerStatus,
    *,
    force_scale_n: float,
    fail_closed: bool,
) -> float:
    if fail_closed or not status.qp_feasible or status.status in {"infeasible", "fault"}:
        return 1.0
    raw = status.metrics.get(
        "allocation_residual_norm",
        status.metrics.get("residual_norm", 0.0),
    )
    value = float(raw)
    if not math.isfinite(value) or value < 0.0:
        return 1.0
    return value / float(force_scale_n)


def _require_same_physical_state(
    expected: Order9IsaacStateSnapshot,
    actual: Order9IsaacStateSnapshot,
    *,
    absolute_tolerance: float = 1.0e-6,
) -> None:
    if expected.joint_names != actual.joint_names or expected.object_id != actual.object_id:
        raise RuntimeError("Order9 copied Isaac restore changed state identity")
    fields = (
        (expected.robot_root_pose_world, actual.robot_root_pose_world),
        (expected.robot_root_twist_world, actual.robot_root_twist_world),
        (expected.joint_positions_rad, actual.joint_positions_rad),
        (expected.joint_velocities_radps, actual.joint_velocities_radps),
        (expected.object_pose_world, actual.object_pose_world),
        (expected.object_twist_world, actual.object_twist_world),
    )
    if any(
        len(left) != len(right)
        or any(
            not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=absolute_tolerance)
            for a, b in zip(left, right)
        )
        for left, right in fields
    ):
        raise RuntimeError("Order9 copied Isaac restore failed exact-state readback")


__all__ = [
    "ORDER9_ISAAC_COPIED_RUNTIME_VERSION",
    "Order9IsaacCopiedRuntime",
    "Order9IsaacSceneAdapter",
]
