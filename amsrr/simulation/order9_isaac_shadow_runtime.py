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
from typing import Mapping, Protocol, Sequence

from amsrr.controllers.actuator_mapping import ActuatorMapping
from amsrr.controllers.controller_base import ControllerContext, PayloadCoupling
from amsrr.controllers.isaac_controller_bridge import (
    IsaacActuatorTargetRecord,
    IsaacControllerBridge,
)
from amsrr.controllers.load_limited_contact_preload import (
    LOAD_LIMITED_CONTACT_PRELOAD_VERSION,
    LoadLimitedContactPreload,
    LoadLimitedContactPreloadConfig,
    LoadLimitedContactPreloadOutput,
)
from amsrr.controllers.qpid_controller import QPIDController
from amsrr.feasibility.articulated_reachability import (
    resolve_mesh_backed_anchor_references,
)
from amsrr.feasibility.contact_wrench_hybrid import ShadowCollisionSample
from amsrr.feasibility.contact_wrench_shadow_metrics import MeasuredCandidateWrench
from amsrr.morphology.random_connected import morphology_structural_hash
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.policies.low_level_policy_base import LowLevelPolicyContext
from amsrr.policies.order9_low_level_runtime import Order9LowLevelRuntimePolicy
from amsrr.policies.order9_policy_command import (
    order9_pi_l_reference_command,
)
from amsrr.robot_model.whole_structure_kinematics import (
    WholeStructureKinematics,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import (
    ContactWrenchTrajectory,
    ControllerCommand,
    ControllerStatus,
    InteractionKnot,
    PolicyCommand,
)
from amsrr.schemas.runtime import RuntimeObservation
from amsrr.schemas.task_spec import TaskType
from amsrr.simulation.order9_object_task_runtime import (
    ORDER9_OBJECT_TASK_ADAPTER_ID,
    ORDER9_OBJECT_TASK_PHASES,
    Order9ObjectTaskPhase,
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


ORDER9_ISAAC_COPIED_RUNTIME_VERSION = "order9_copied_isaac_policy_qpid_runtime_v2"


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

    def set_task_phase(self, phase_index: int) -> None:
        """Advance only task-phase identity while preserving physical state."""

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

    def damping_compensated_dock_load_nm(self) -> Mapping[str, float]:
        """Return actuator load with virtual-drive damping removed."""

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
        bypass_pi_l: bool = False,
        collect_contact_evidence: bool = True,
        collect_collision_evidence: bool = True,
        contact_preload_config: LoadLimitedContactPreloadConfig | None = None,
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
        self.bypass_pi_l = bool(bypass_pi_l)
        self.collect_contact_evidence = bool(collect_contact_evidence)
        self.collect_collision_evidence = bool(collect_collision_evidence)
        self.contact_preload = LoadLimitedContactPreload(
            contact_preload_config
        )
        self._whole_structure_kinematics = WholeStructureKinematics()
        self._last_nominal_joint_positions_rad: dict[str, float] = {}
        self._last_closure_direction_rad: dict[str, float] = {}
        self._contact_start_force_n_by_anchor: dict[int, float] = {}
        self._contact_start_dwell_s_by_anchor: dict[int, float] = {}
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
            "command_source": (
                "direct_qpid_reference"
                if self.bypass_pi_l
                else "learned_pi_l"
            ),
            "contact_evidence_enabled": self.collect_contact_evidence,
            "collision_evidence_enabled": self.collect_collision_evidence,
            "contact_preload": {
                "version": LOAD_LIMITED_CONTACT_PRELOAD_VERSION,
                "maximum_speed_rad_s": (
                    self.contact_preload.config.maximum_speed_rad_s
                ),
                "load_threshold_nm": (
                    self.contact_preload.config.load_threshold_nm
                ),
                "load_dwell_s": self.contact_preload.config.load_dwell_s,
                "contact_start_force_threshold_n": (
                    self.contact_preload.config.contact_start_force_threshold_n
                ),
                "contact_start_dwell_s": (
                    self.contact_preload.config.contact_start_dwell_s
                ),
                "placement": "after_pi_l_before_qpid",
            },
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
        observation = context.runtime_observation
        if observation is None:
            raise SchemaValidationError(
                "Order9 copied trajectory requires runtime phase identity"
            )
        self.scene_adapter.set_task_phase(_phase_index(observation))
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
            metrics={
                "observation_only": 1.0,
                **self.contact_preload.metrics(),
            },
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
        inference = None
        if self.bypass_pi_l:
            policy_command = order9_pi_l_reference_command(low_context)
        else:
            inference = self.pi_l_policy.command_with_trace(low_context)
            policy_command = inference.command
        policy_command, contact_preload_metrics = self._apply_contact_preload(
            context=context,
            observation=observation,
            active_knot=active_knot,
            policy_command=policy_command,
            dt_s=float(dt_s),
        )
        command = self.controller.compute(
            ControllerContext(
                runtime_observation=observation,
                morphology_graph=self.morphology_graph,
                physical_model=self.physical_model,
                active_knot=active_knot,
                policy_command=policy_command,
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
            (
                inference is not None
                and (
                    not inference.learned_policy_applied
                    or inference.fallback_reason is not None
                )
            )
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
                "learned_pi_l_applied": (
                    1.0
                    if inference is not None
                    and inference.learned_policy_applied
                    else 0.0
                ),
                "qpid_reference_applied": 1.0 if self.bypass_pi_l else 0.0,
                "pi_l_fallback": (
                    1.0
                    if inference is not None
                    and not inference.learned_policy_applied
                    else 0.0
                ),
                "unresolved_actuator_target_count": float(unresolved),
                "missing_actuator_count": float(len(record.missing_actuators)),
                "unsupported_actuator_count": float(
                    len(record.unsupported_actuators)
                ),
                "clipped_actuator_count": float(len(record.clipped_targets)),
                **contact_preload_metrics,
            },
        )

    def _apply_contact_preload(
        self,
        *,
        context: HighLevelPolicyContext,
        observation: RuntimeObservation,
        active_knot: InteractionKnot,
        policy_command: PolicyCommand,
        dt_s: float,
    ) -> tuple[PolicyCommand, dict[str, float]]:
        """Apply the proven Order 8 preload below pi_L and above QPID."""

        phase_label = str(observation.task_progress.phase_label)
        nominal = _dock_joint_mapping(
            (
                {}
                if active_knot.posture_target is None
                or active_knot.posture_target.joint_pos_target is None
                else active_knot.posture_target.joint_pos_target
            )
        )
        if phase_label == Order9ObjectTaskPhase.CONTACT_ACQUISITION.value:
            self._remember_closure_direction(nominal)
        elif phase_label in {
            Order9ObjectTaskPhase.RELEASE.value,
            Order9ObjectTaskPhase.RETREAT.value,
            Order9ObjectTaskPhase.SETTLE.value,
        }:
            self.contact_preload.reset()
            self._last_nominal_joint_positions_rad = {}
            self._last_closure_direction_rad = {}
            self._contact_start_force_n_by_anchor = {}
            self._contact_start_dwell_s_by_anchor = {}
            return policy_command, self.contact_preload.metrics()

        measured_contacts = ()
        valid_anchor_ids: set[int] = set()
        required_anchor_ids = {
            int(assignment.anchor_id)
            for assignment in active_knot.contact_assignments
            if assignment.schedule_state in {"attach", "maintain", "slide"}
        }
        if (
            phase_label == Order9ObjectTaskPhase.CONTACT_ACQUISITION.value
            and not self.contact_preload.initialized
            and len(required_anchor_ids) >= 2
        ):
            measured_contacts = tuple(
                self.scene_adapter.measured_candidate_wrenches(
                    context=context,
                    active_knot=active_knot,
                )
            )
            force_n_by_candidate = {
                value.candidate_id: (
                    math.sqrt(
                        sum(
                            float(component) ** 2
                            for component in value.wrench_contact[:3]
                        )
                    )
                    if value.evidence_valid
                    else 0.0
                )
                for value in measured_contacts
            }
            for assignment in active_knot.contact_assignments:
                anchor_id = int(assignment.anchor_id)
                if anchor_id not in required_anchor_ids:
                    continue
                force_n = float(
                    force_n_by_candidate.get(assignment.candidate_id, 0.0)
                )
                self._contact_start_force_n_by_anchor[anchor_id] = force_n
                if (
                    force_n + 1.0e-12
                    >= self.contact_preload.config.contact_start_force_threshold_n
                ):
                    self._contact_start_dwell_s_by_anchor[anchor_id] = (
                        self._contact_start_dwell_s_by_anchor.get(
                            anchor_id,
                            0.0,
                        )
                        + dt_s
                    )
                else:
                    self._contact_start_dwell_s_by_anchor[anchor_id] = 0.0
            valid_anchor_ids = {
                anchor_id
                for anchor_id in required_anchor_ids
                if (
                    self._contact_start_dwell_s_by_anchor.get(anchor_id, 0.0)
                    + 1.0e-12
                    >= self.contact_preload.config.contact_start_dwell_s
                )
            }
            if required_anchor_ids.issubset(valid_anchor_ids):
                loads = dict(
                    self.scene_adapter.damping_compensated_dock_load_nm()
                )
                ordered_joint_ids = tuple(sorted(loads))
                measured_positions = _measured_dock_joint_positions(observation)
                missing_measured = set(ordered_joint_ids).difference(
                    measured_positions
                )
                if missing_measured:
                    raise RuntimeError(
                        "Order9 contact preload lacks measured Dock joints: "
                        + ", ".join(sorted(missing_measured))
                    )
                closure_direction = {
                    joint_id: float(
                        self._last_closure_direction_rad.get(
                            joint_id,
                            nominal.get(
                                joint_id,
                                measured_positions[joint_id],
                            )
                            - measured_positions[joint_id],
                        )
                    )
                    for joint_id in ordered_joint_ids
                }
                joint_ids_by_anchor = self._preload_joint_ids_by_anchor(
                    anchor_ids=tuple(sorted(required_anchor_ids)),
                    ordered_joint_ids=ordered_joint_ids,
                    measured_positions_rad=measured_positions,
                    closure_direction_rad=closure_direction,
                )
                initial_targets = {
                    joint_id: float(
                        policy_command.joint_position_targets.get(
                            joint_id,
                            nominal.get(
                                joint_id,
                                measured_positions[joint_id],
                            ),
                        )
                    )
                    for joint_id in ordered_joint_ids
                }
                self.contact_preload.start(
                    ordered_joint_ids=ordered_joint_ids,
                    closure_velocity_targets_rad_s=closure_direction,
                    joint_ids_by_anchor=joint_ids_by_anchor,
                    initial_position_targets_rad=initial_targets,
                )

        output: LoadLimitedContactPreloadOutput | None = None
        if self.contact_preload.initialized:
            if phase_label in {
                Order9ObjectTaskPhase.CONTACT_ACQUISITION.value,
                Order9ObjectTaskPhase.LIFT.value,
                Order9ObjectTaskPhase.TRANSPORT.value,
                Order9ObjectTaskPhase.PLACE.value,
            }:
                output = (
                    self.contact_preload.hold()
                    if self.contact_preload.complete
                    else self.contact_preload.step(
                        applied_joint_load_nm=(
                            self.scene_adapter.damping_compensated_dock_load_nm()
                        ),
                        dt_s=dt_s,
                    )
                )

        valid_anchor_ids = {
            anchor_id
            for anchor_id in required_anchor_ids
            if (
                self._contact_start_dwell_s_by_anchor.get(anchor_id, 0.0)
                + 1.0e-12
                >= self.contact_preload.config.contact_start_dwell_s
            )
        }
        metrics = self.contact_preload.metrics()
        metrics.update(
            {
                "contact_preload_required_anchor_count": float(
                    len(required_anchor_ids)
                ),
                "contact_preload_valid_anchor_count": float(
                    len(valid_anchor_ids)
                ),
            }
        )
        metrics.update(
            {
                f"contact_preload_start_force_n.anchor_{anchor_id}": float(
                    value
                )
                for anchor_id, value in (
                    self._contact_start_force_n_by_anchor.items()
                )
            }
        )
        metrics.update(
            {
                f"contact_preload_start_dwell_s.anchor_{anchor_id}": float(
                    value
                )
                for anchor_id, value in (
                    self._contact_start_dwell_s_by_anchor.items()
                )
            }
        )
        if output is None:
            return policy_command, metrics
        updated = PolicyCommand.from_dict(policy_command.to_dict())
        updated.joint_position_targets.update(output.position_targets_rad)
        updated.joint_velocity_targets.update(
            output.velocity_targets_rad_s
        )
        updated.joint_torque_bias.update(
            {
                joint_id: 0.0
                for joint_id in output.position_targets_rad
            }
        )
        updated.validate()
        return updated, metrics

    def _remember_closure_direction(
        self,
        nominal_positions_rad: Mapping[str, float],
    ) -> None:
        if not nominal_positions_rad:
            return
        if self._last_nominal_joint_positions_rad:
            common = set(nominal_positions_rad).intersection(
                self._last_nominal_joint_positions_rad
            )
            delta = {
                joint_id: (
                    float(nominal_positions_rad[joint_id])
                    - self._last_nominal_joint_positions_rad[joint_id]
                )
                for joint_id in common
            }
            if delta and max(abs(value) for value in delta.values()) > 1.0e-9:
                self._last_closure_direction_rad = {
                    joint_id: float(delta.get(joint_id, 0.0))
                    for joint_id in nominal_positions_rad
                }
        self._last_nominal_joint_positions_rad = {
            joint_id: float(value)
            for joint_id, value in nominal_positions_rad.items()
        }

    def _preload_joint_ids_by_anchor(
        self,
        *,
        anchor_ids: Sequence[int],
        ordered_joint_ids: Sequence[str],
        measured_positions_rad: Mapping[str, float],
        closure_direction_rad: Mapping[str, float],
    ) -> dict[int, tuple[str, ...]]:
        references = resolve_mesh_backed_anchor_references(
            self.morphology_graph,
            self.physical_model,
            anchor_ids,
        )
        kinematics = self._whole_structure_kinematics.compute(
            self.morphology_graph,
            self.physical_model,
            {
                joint_id: float(measured_positions_rad[joint_id])
                for joint_id in ordered_joint_ids
            },
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            references,
        )
        if set(ordered_joint_ids) != set(
            kinematics.ordered_global_dock_joint_ids
        ):
            raise RuntimeError(
                "Order9 contact preload Dock set differs from kinematics"
            )
        anchors = {
            anchor.anchor_id: anchor
            for anchor in self.morphology_graph.robot_anchors
        }
        result: dict[int, tuple[str, ...]] = {}
        for anchor_id in anchor_ids:
            anchor = anchors[int(anchor_id)]
            required_local_id = str(
                anchor.capability.get("dock_mechanism_joint_id", "")
            )
            required_joint_id = (
                f"module_{anchor.module_id}:{required_local_id}"
            )
            jacobian = kinematics.anchor_jacobians[int(anchor_id)]
            influential = tuple(
                joint_id
                for column, joint_id in enumerate(
                    kinematics.ordered_global_dock_joint_ids
                )
                if (
                    joint_id == required_joint_id
                    or math.sqrt(
                        sum(float(row[column]) ** 2 for row in jacobian)
                    )
                    > 1.0e-8
                )
                and abs(float(closure_direction_rad[joint_id])) > 1.0e-9
            )
            if not influential:
                raise RuntimeError(
                    f"Order9 contact preload anchor {anchor_id} has no "
                    "moving influential Dock joint"
                )
            result[int(anchor_id)] = influential
        return result

    def reset_copied_state(self) -> None:
        self.scene_adapter.reset()
        self.controller.reset_integrators()
        self.pi_l_policy.reset()
        self.contact_preload.reset()
        self._last_nominal_joint_positions_rad = {}
        self._last_closure_direction_rad = {}
        self._contact_start_force_n_by_anchor = {}
        self._contact_start_dwell_s_by_anchor = {}
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
        if self.collect_contact_evidence:
            measured = tuple(
                self.scene_adapter.measured_candidate_wrenches(
                    context=context,
                    active_knot=active_knot,
                )
            )
        else:
            measured = ()
        if self.collect_collision_evidence:
            samples, clearance = self.scene_adapter.collision_evidence(
                context=context,
                active_knot=active_knot,
            )
        else:
            samples = ()
            clearance = 0.0
        endpoint_state_metrics: dict[str, float] = {}
        if metrics.get("observation_only") == 1.0:
            snapshot = self.scene_adapter.capture_snapshot()
            observation = self.scene_adapter.actor_observation(
                morphology_graph=self.morphology_graph,
                controller_status=self._last_status,
                elapsed_s=elapsed_s,
            )
            control_model = self.controller.rigid_body_model_builder.build(
                self.morphology_graph,
                self.physical_model,
                observation,
            )
            endpoint_state_metrics = {
                **{
                    f"centroidal_pose_world_{index}": float(value)
                    for index, value in enumerate(
                        control_model.body_pose_world
                    )
                },
                **{
                    f"centroidal_twist_world_{index}": float(value)
                    for index, value in enumerate(
                        control_model.body_twist_world
                    )
                },
                **{
                    (
                        "module_pose_world."
                        f"module_{state.module_id}_{index}"
                    ): float(value)
                    for state in observation.module_states
                    for index, value in enumerate(state.pose_world)
                },
                **{
                    (
                        "module_twist_world."
                        f"module_{state.module_id}_{index}"
                    ): float(value)
                    for state in observation.module_states
                    for index, value in enumerate(state.twist_world)
                },
                **{
                    f"robot_root_pose_world_{index}": float(value)
                    for index, value in enumerate(
                        snapshot.robot_root_pose_world
                    )
                },
                **{
                    f"robot_root_twist_world_{index}": float(value)
                    for index, value in enumerate(
                        snapshot.robot_root_twist_world
                    )
                },
                **{
                    f"object_pose_world_{index}": float(value)
                    for index, value in enumerate(snapshot.object_pose_world)
                },
                **{
                    f"object_twist_world_{index}": float(value)
                    for index, value in enumerate(snapshot.object_twist_world)
                },
                **{
                    f"joint_position_rad.{name}": float(value)
                    for name, value in zip(
                        snapshot.joint_names,
                        snapshot.joint_positions_rad,
                        strict=True,
                    )
                },
                **{
                    f"joint_velocity_radps.{name}": float(value)
                    for name, value in zip(
                        snapshot.joint_names,
                        snapshot.joint_velocities_radps,
                        strict=True,
                    )
                },
                "contact_evidence_enabled": (
                    1.0 if self.collect_contact_evidence else 0.0
                ),
                "collision_evidence_enabled": (
                    1.0 if self.collect_collision_evidence else 0.0
                ),
            }
        return Order9IsaacControlStepEvidence(
            controller_qp_residual=float(controller_residual),
            measured_candidate_wrenches=measured,
            collision_samples=tuple(samples),
            collision_free_clearance_m=float(clearance),
            finite_state=bool(self.scene_adapter.finite_state()),
            metrics={
                "elapsed_s": float(elapsed_s),
                **metrics,
                **endpoint_state_metrics,
            },
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


def _dock_joint_mapping(values: Mapping[str, float]) -> dict[str, float]:
    result = {
        str(joint_id): float(value)
        for joint_id, value in values.items()
        if "dock_mech_joint" in str(joint_id)
    }
    if any(not math.isfinite(value) for value in result.values()):
        raise SchemaValidationError(
            "Order9 contact preload nominal Dock targets must be finite"
        )
    return result


def _measured_dock_joint_positions(
    observation: RuntimeObservation,
) -> dict[str, float]:
    result = {
        f"module_{module.module_id}:{local_id}": float(value)
        for module in observation.module_states
        for local_id, value in module.joint_positions.items()
        if "dock_mech_joint" in str(local_id)
    }
    if any(not math.isfinite(value) for value in result.values()):
        raise SchemaValidationError(
            "Order9 contact preload measured Dock positions must be finite"
        )
    return result


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
