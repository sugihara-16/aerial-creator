from __future__ import annotations

"""Production deterministic free-flight pi_H fallback for P4-full Order 4."""

from dataclasses import dataclass
import math
from typing import Any, Sequence

from amsrr.controllers.rigid_body_model import RigidBodyControlModelBuilder
from amsrr.policies.contact_wrench_trajectory_runtime import (
    ContactWrenchTrajectoryExecutor,
    ContactWrenchTrajectoryRuntimeError,
)
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.schemas.common import Pose7D, SchemaValidationError
from amsrr.schemas.contact_candidates import ContactCandidateSet
from amsrr.schemas.interaction_envelope import (
    DurationRequirement,
    InteractionEnvelope,
    PrecisionRequirement,
)
from amsrr.schemas.irg import (
    IRGEdge,
    IRGEdgeType,
    IRGNode,
    IRGNodeType,
    InteractionRequirementGraph,
    PhaseType,
)
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.order4 import (
    ORDER4_FREE_FLIGHT_RUNTIME_VERSION,
    Order4DeterministicPlannerConfig,
    Order4FreeFlightMission,
    Order4FreeFlightPhase,
    Order4TrajectoryRuntimeStep,
)
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import (
    CentroidalTarget,
    ContactWrenchTrajectory,
    InteractionKnot,
    PostureTarget,
)
from amsrr.schemas.runtime import RuntimeObservation


ORDER4_MISSION_METADATA_KEY = "order4_free_flight_mission"
ORDER4_MISSION_HASH_METADATA_KEY = "order4_free_flight_mission_hash"
ORDER4_CONTEXT_VERSION = "order4_free_flight_high_level_context_v1"


@dataclass(frozen=True)
class Order4PhaseTransition:
    from_phase: str | None
    to_phase: str
    time_s: float
    reason: str
    waypoint_index: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "from_phase": self.from_phase,
            "to_phase": self.to_phase,
            "time_s": self.time_s,
            "reason": self.reason,
            "waypoint_index": self.waypoint_index,
        }


class Order4FreeFlightContextFactory:
    """Build the unchanged pi_H context around one versioned mission."""

    def __init__(
        self,
        *,
        mission: Order4FreeFlightMission,
        morphology_graph: MorphologyGraph,
        planner_config: Order4DeterministicPlannerConfig,
    ) -> None:
        mission.validate()
        planner_config.validate()
        self.mission = mission
        self.morphology_graph = morphology_graph
        task_id = f"order4:{mission.mission_id}"
        nodes: list[IRGNode] = [
            IRGNode(
                node_id=0,
                node_type=IRGNodeType.TASK,
                ref_id=task_id,
                priority=1.0,
                is_hard=True,
                active_phase_id=None,
                feature={"task_type": "free_flight_navigation"},
            )
        ]
        edges: list[IRGEdge] = []
        phase_labels = [
            "floor_settle",
            "takeoff",
            "hover_acquisition",
            *[
                f"waypoint:{waypoint.waypoint_id}"
                for waypoint in mission.waypoints
            ],
            "final_hover",
        ]
        previous_phase_id: int | None = None
        next_node_id = 1
        for phase_label in phase_labels:
            phase_id = next_node_id
            next_node_id += 1
            nodes.append(
                IRGNode(
                    node_id=phase_id,
                    node_type=IRGNodeType.PHASE,
                    ref_id=phase_label,
                    priority=1.0,
                    is_hard=True,
                    active_phase_id=phase_id,
                    feature={
                        "phase_type": PhaseType.FREE_MOTION.value,
                        "phase_label": phase_label,
                    },
                )
            )
            edges.append(
                IRGEdge(
                    src_id=0,
                    dst_id=phase_id,
                    edge_type=IRGEdgeType.CONTAINS,
                )
            )
            if previous_phase_id is not None:
                edges.append(
                    IRGEdge(
                        src_id=previous_phase_id,
                        dst_id=phase_id,
                        edge_type=IRGEdgeType.TEMPORAL_NEXT,
                    )
                )
            previous_phase_id = phase_id
        for waypoint_index, waypoint in enumerate(mission.waypoints):
            target_id = next_node_id
            next_node_id += 1
            nodes.append(
                IRGNode(
                    node_id=target_id,
                    node_type=IRGNodeType.STATE_TARGET,
                    ref_id=waypoint.waypoint_id,
                    priority=1.0,
                    is_hard=True,
                    active_phase_id=4 + waypoint_index,
                    feature={
                        "target_type": "body_pose_offset_from_initial_hover",
                        "position_offset_world": list(waypoint.position_offset_world),
                        "orientation_rpy_rad": list(waypoint.orientation_rpy_rad),
                    },
                )
            )
            edges.append(
                IRGEdge(
                    src_id=4 + waypoint_index,
                    dst_id=target_id,
                    edge_type=IRGEdgeType.REQUIRES,
                )
            )
        self.irg = InteractionRequirementGraph(
            irg_id=f"order4-free-flight-irg:{mission.mission_hash[:16]}",
            task_id=task_id,
            nodes=nodes,
            edges=edges,
            metadata={
                "context_version": ORDER4_CONTEXT_VERSION,
                ORDER4_MISSION_METADATA_KEY: mission.to_dict(),
                ORDER4_MISSION_HASH_METADATA_KEY: mission.mission_hash,
            },
        )
        self.envelope = InteractionEnvelope(
            envelope_id=f"order4-free-flight-envelope:{mission.mission_hash[:16]}",
            task_id=task_id,
            required_contact_count_range=(0, 0),
            required_contact_modes=[],
            target_region_sets=[],
            wrench_space_requirements=[],
            precision_requirements=[
                PrecisionRequirement(
                    target="centroidal_pose",
                    tolerance_pos_m=planner_config.position_tolerance_m,
                    tolerance_rot_rad=planner_config.attitude_tolerance_rad,
                )
            ],
            duration_requirements=[
                DurationRequirement(
                    phase_label="final_hover",
                    min_duration_s=mission.final_hover_hold_s,
                    max_duration_s=mission.mission_timeout_s,
                )
            ],
        )
        self.candidate_set = ContactCandidateSet(
            set_id=f"order4-empty-candidates:{mission.mission_hash[:16]}",
            task_id=task_id,
            morphology_graph_id=morphology_graph.graph_id,
            candidates=[],
            candidate_mask=[],
            slot_coverage={},
            pairwise_conflict_matrix=[],
            pairwise_compatibility_score=[],
            group_proposals=[],
            assignment_feasibility_cache={},
            sampler_version="order4_no_contact_candidates_v1",
        )
        for node in self.irg.nodes:
            node.validate()
        self.irg.validate()
        self.envelope.validate()
        self.candidate_set.validate()

    def context(
        self,
        runtime_observation: RuntimeObservation,
    ) -> HighLevelPolicyContext:
        if runtime_observation.morphology_graph.stable_hash() != self.morphology_graph.stable_hash():
            raise SchemaValidationError(
                "Order4 runtime observation morphology does not match its policy context"
            )
        return HighLevelPolicyContext(
            irg=self.irg,
            interaction_envelope=self.envelope,
            morphology_graph=self.morphology_graph,
            contact_candidate_set=self.candidate_set,
            runtime_observation=runtime_observation,
        )


class DeterministicFreeFlightPlanner:
    """State-dependent free-flight slice of the production deterministic pi_H."""

    def __init__(
        self,
        *,
        physical_model: PhysicalModel,
        config: Order4DeterministicPlannerConfig | None = None,
    ) -> None:
        self.physical_model = physical_model
        self.config = config or Order4DeterministicPlannerConfig()
        self.config.validate()
        self._model_builder = RigidBodyControlModelBuilder()
        self.reset()

    def reset(self) -> None:
        self._mission: Order4FreeFlightMission | None = None
        self._phase: Order4FreeFlightPhase | None = None
        self._waypoint_index: int | None = None
        self._mission_start_time_s: float | None = None
        self._phase_start_time_s: float | None = None
        self._guard_start_time_s: float | None = None
        self._settled_pose: Pose7D | None = None
        self._initial_hover_pose: Pose7D | None = None
        self._phase_start_pose: Pose7D | None = None
        self._safe_hold_pose: Pose7D | None = None
        self._failure_reason: str | None = None
        self._current_morphology: MorphologyGraph | None = None
        self._transitions: list[Order4PhaseTransition] = []

    @property
    def phase(self) -> Order4FreeFlightPhase:
        return self._phase or "floor_settle"

    @property
    def waypoint_index(self) -> int | None:
        return self._waypoint_index

    @property
    def failure_reason(self) -> str | None:
        return self._failure_reason

    @property
    def safe_hold_active(self) -> bool:
        return self.phase == "safe_hold"

    @property
    def transitions(self) -> list[Order4PhaseTransition]:
        return list(self._transitions)

    @property
    def final_target_pose(self) -> Pose7D | None:
        mission = self._mission
        if mission is None or self._initial_hover_pose is None:
            return None
        return self._waypoint_target_pose(len(mission.waypoints) - 1)

    def mission_progress_ratio(self, *, time_s: float) -> float:
        mission = self._mission
        if mission is None or self._phase_start_time_s is None:
            return 0.0
        phase_elapsed = max(0.0, float(time_s) - self._phase_start_time_s)
        if self.phase == "floor_settle":
            return 0.05 * min(
                phase_elapsed / self.config.floor_settle_duration_s,
                1.0,
            )
        if self.phase == "takeoff":
            return 0.05 + 0.10 * min(
                phase_elapsed / self.config.takeoff_duration_s,
                1.0,
            )
        if self.phase == "hover_acquisition":
            return 0.20
        if self.phase == "waypoint" and self._waypoint_index is not None:
            waypoint = mission.waypoints[self._waypoint_index]
            local = min(phase_elapsed / waypoint.transition_duration_s, 1.0)
            return min(
                0.90,
                0.20
                + 0.65
                * (self._waypoint_index + local)
                / max(len(mission.waypoints), 1),
            )
        if self.phase == "final_hover":
            return 0.90 + 0.10 * min(
                phase_elapsed / mission.final_hover_hold_s,
                1.0,
            )
        if self.phase == "complete":
            return 1.0
        mission_start_time_s = (
            self._mission_start_time_s
            if self._mission_start_time_s is not None
            else float(time_s)
        )
        return min(
            max(
                0.0,
                (float(time_s) - mission_start_time_s)
                / mission.mission_timeout_s,
            ),
            1.0,
        )

    def plan(self, context: HighLevelPolicyContext) -> ContactWrenchTrajectory:
        observation = _validated_runtime_observation(context)
        mission = _mission_from_context(context)
        if self._mission is None or self._mission.mission_hash != mission.mission_hash:
            self.reset()
            self._mission = mission
        self._current_morphology = context.morphology_graph
        pose, twist = self._centroidal_state(context, observation)
        now_s = float(observation.time_s)
        if self._phase is None:
            self._mission_start_time_s = now_s
            self._phase_start_time_s = now_s
            self._phase_start_pose = pose
            self._settled_pose = pose
            self._transition(
                "floor_settle",
                time_s=now_s,
                reason="mission_initialized",
                waypoint_index=None,
            )
        self._update_state(
            context=context,
            observation=observation,
            pose=pose,
            twist=twist,
        )
        return self._trajectory(now_s=now_s, current_pose=pose)

    def force_safe_hold(
        self,
        context: HighLevelPolicyContext,
        *,
        reason: str,
    ) -> ContactWrenchTrajectory:
        observation = _validated_runtime_observation(context)
        mission = _mission_from_context(context)
        if self._mission is None:
            self._mission = mission
        self._current_morphology = context.morphology_graph
        pose, _ = self._centroidal_state(context, observation)
        self._safe_hold_pose = pose
        self._failure_reason = reason
        self._guard_start_time_s = None
        self._transition(
            "safe_hold",
            time_s=observation.time_s,
            reason=reason,
            waypoint_index=self._waypoint_index,
        )
        return self._trajectory(now_s=observation.time_s, current_pose=pose)

    def _centroidal_state(
        self,
        context: HighLevelPolicyContext,
        observation: RuntimeObservation,
    ) -> tuple[Pose7D, list[float]]:
        model = self._model_builder.build(
            context.morphology_graph,
            self.physical_model,
            observation,
        )
        pose = tuple(float(value) for value in model.body_pose_world)
        twist = [float(value) for value in model.body_twist_world]
        if not all(math.isfinite(value) for value in (*pose, *twist)):
            raise SchemaValidationError("Order4 centroidal state must be finite")
        return pose, twist

    def _update_state(
        self,
        *,
        context: HighLevelPolicyContext,
        observation: RuntimeObservation,
        pose: Pose7D,
        twist: list[float],
    ) -> None:
        now_s = float(observation.time_s)
        mission = self._mission
        if mission is None or self._phase_start_time_s is None:
            raise RuntimeError("Order4 planner state was not initialized")
        if self.phase == "safe_hold":
            return
        if (
            observation.controller_status.status == "fault"
            or not observation.controller_status.qp_feasible
        ):
            self._enter_safe_hold(
                pose,
                time_s=now_s,
                reason="controller_not_feasible",
            )
            return
        if _body_tilt_rad(pose[3:7]) > self.config.max_tilt_rad:
            self._enter_safe_hold(pose, time_s=now_s, reason="tilt_limit_exceeded")
            return
        if (
            self._mission_start_time_s is not None
            and now_s - self._mission_start_time_s > mission.mission_timeout_s
        ):
            self._enter_safe_hold(pose, time_s=now_s, reason="mission_timeout")
            return
        phase_elapsed = now_s - self._phase_start_time_s
        speed_ok = (
            _norm(twist[:3]) <= self.config.linear_speed_tolerance_mps
            and _norm(twist[3:]) <= self.config.angular_speed_tolerance_rad_s
        )
        if self.phase == "floor_settle":
            floor_contact = any(
                contact.active and "floor" in contact.entity_b.lower()
                for contact in observation.contact_states
            )
            dwell_ok = self._dwell_satisfied(
                floor_contact and speed_ok,
                now_s=now_s,
                required_s=self.config.floor_settle_dwell_s,
            )
            if phase_elapsed >= self.config.floor_settle_duration_s and dwell_ok:
                self._settled_pose = pose
                self._initial_hover_pose = (
                    pose[0],
                    pose[1],
                    pose[2] + mission.hover_height_delta_m,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                )
                self._phase_start_pose = pose
                self._guard_start_time_s = None
                self._transition(
                    "takeoff",
                    time_s=now_s,
                    reason="floor_settle_guard_satisfied",
                    waypoint_index=None,
                )
            elif phase_elapsed > self.config.floor_settle_duration_s + 3.0:
                self._enter_safe_hold(
                    pose,
                    time_s=now_s,
                    reason="floor_settle_timeout",
                )
            return
        if self._initial_hover_pose is None:
            self._enter_safe_hold(
                pose,
                time_s=now_s,
                reason="missing_initial_hover_reference",
            )
            return
        if self.phase == "takeoff":
            if phase_elapsed >= self.config.takeoff_duration_s:
                self._phase_start_pose = pose
                self._guard_start_time_s = None
                self._transition(
                    "hover_acquisition",
                    time_s=now_s,
                    reason="takeoff_trajectory_elapsed",
                    waypoint_index=None,
                )
            elif phase_elapsed > self.config.takeoff_duration_s + 3.0:
                self._enter_safe_hold(pose, time_s=now_s, reason="takeoff_timeout")
            return
        if self.phase == "hover_acquisition":
            reached = _target_reached(
                pose,
                twist,
                self._initial_hover_pose,
                self.config,
            )
            if self._dwell_satisfied(
                reached,
                now_s=now_s,
                required_s=mission.hover_acquisition_dwell_s,
            ):
                self._waypoint_index = 0
                self._phase_start_pose = pose
                self._guard_start_time_s = None
                self._transition(
                    "waypoint",
                    time_s=now_s,
                    reason="hover_acquisition_guard_satisfied",
                    waypoint_index=0,
                )
            elif phase_elapsed > self.config.hover_acquisition_timeout_s:
                self._enter_safe_hold(
                    pose,
                    time_s=now_s,
                    reason="hover_acquisition_timeout",
                )
            return
        if self.phase == "waypoint":
            if self._waypoint_index is None:
                self._enter_safe_hold(
                    pose,
                    time_s=now_s,
                    reason="missing_waypoint_index",
                )
                return
            waypoint = mission.waypoints[self._waypoint_index]
            target_pose = self._waypoint_target_pose(self._waypoint_index)
            reached = (
                phase_elapsed >= waypoint.transition_duration_s
                and _target_reached(pose, twist, target_pose, self.config)
            )
            if self._dwell_satisfied(
                reached,
                now_s=now_s,
                required_s=waypoint.dwell_s,
            ):
                next_index = self._waypoint_index + 1
                self._phase_start_pose = pose
                self._guard_start_time_s = None
                if next_index < len(mission.waypoints):
                    self._waypoint_index = next_index
                    self._transition(
                        "waypoint",
                        time_s=now_s,
                        reason="waypoint_guard_satisfied",
                        waypoint_index=next_index,
                        force_record=True,
                    )
                else:
                    self._transition(
                        "final_hover",
                        time_s=now_s,
                        reason="all_waypoints_satisfied",
                        waypoint_index=self._waypoint_index,
                    )
            elif phase_elapsed > waypoint.timeout_s:
                self._enter_safe_hold(
                    pose,
                    time_s=now_s,
                    reason=f"waypoint_timeout:{waypoint.waypoint_id}",
                )
            return
        if self.phase == "final_hover":
            target_pose = self._waypoint_target_pose(len(mission.waypoints) - 1)
            reached = _target_reached(pose, twist, target_pose, self.config)
            if self._dwell_satisfied(
                reached,
                now_s=now_s,
                required_s=mission.final_hover_hold_s,
            ):
                self._guard_start_time_s = None
                self._transition(
                    "complete",
                    time_s=now_s,
                    reason="final_hover_dwell_satisfied",
                    waypoint_index=self._waypoint_index,
                )

    def _trajectory(
        self,
        *,
        now_s: float,
        current_pose: Pose7D,
    ) -> ContactWrenchTrajectory:
        phase = self.phase
        times = _knot_times(self.config.horizon_s, self.config.knot_dt_s)
        knots: list[InteractionKnot] = []
        for relative_s in times:
            pose = self._planned_pose_at(
                absolute_time_s=now_s + relative_s,
                current_pose=current_pose,
            )
            next_pose = self._planned_pose_at(
                absolute_time_s=now_s + relative_s + min(0.02, self.config.knot_dt_s),
                current_pose=current_pose,
            )
            velocity_dt = min(0.02, self.config.knot_dt_s)
            linear_velocity = tuple(
                (next_pose[index] - pose[index]) / velocity_dt
                for index in range(3)
            )
            knots.append(
                InteractionKnot(
                    t_rel_s=relative_s,
                    contact_assignments=[],
                    centroidal_target=CentroidalTarget(
                        com_pos_world=tuple(pose[:3]),
                        com_vel_world=linear_velocity,
                        body_orientation_world=tuple(pose[3:7]),
                        centroidal_wrench_preference=None,
                    ),
                    posture_target=self._neutral_dock_posture(),
                    object_targets=[],
                    priority_weights={
                        "centroidal_tracking": 1.0,
                        "order4_deterministic_fallback": 1.0,
                        f"order4_phase_{phase}": 1.0,
                    },
                    guard_conditions=self._guard_conditions(),
                )
            )
        trajectory = ContactWrenchTrajectory(
            horizon_s=self.config.horizon_s,
            dt_s=self.config.knot_dt_s,
            knots=knots,
            derived_mode_label=f"order4_free_flight_{phase}",
        )
        if any(knot.contact_assignments for knot in trajectory.knots):
            raise SchemaValidationError(
                "Order4 free-flight planner emitted a non-empty contact assignment"
            )
        return trajectory

    def _planned_pose_at(
        self,
        *,
        absolute_time_s: float,
        current_pose: Pose7D,
    ) -> Pose7D:
        if self._phase_start_time_s is None:
            return current_pose
        phase_elapsed = max(0.0, absolute_time_s - self._phase_start_time_s)
        if self.phase == "floor_settle":
            return self._settled_pose or current_pose
        if self.phase == "safe_hold":
            return self._safe_hold_pose or current_pose
        if self._initial_hover_pose is None:
            return current_pose
        if self.phase == "takeoff":
            start = self._settled_pose or self._phase_start_pose or current_pose
            ratio = _smoothstep(
                min(phase_elapsed / self.config.takeoff_duration_s, 1.0)
            )
            return _interpolate_pose(start, self._initial_hover_pose, ratio)
        if self.phase == "hover_acquisition":
            return self._initial_hover_pose
        if self.phase == "waypoint" and self._waypoint_index is not None:
            waypoint = self._mission.waypoints[self._waypoint_index]  # type: ignore[union-attr]
            start = self._phase_start_pose or current_pose
            target = self._waypoint_target_pose(self._waypoint_index)
            ratio = _smoothstep(
                min(phase_elapsed / waypoint.transition_duration_s, 1.0)
            )
            return _interpolate_pose(start, target, ratio)
        final_target = self.final_target_pose
        return final_target or self._initial_hover_pose

    def _waypoint_target_pose(self, index: int) -> Pose7D:
        mission = self._mission
        hover = self._initial_hover_pose
        if mission is None or hover is None:
            raise RuntimeError("Order4 waypoint target requested before hover reference")
        waypoint = mission.waypoints[index]
        quaternion = _rpy_to_quaternion(waypoint.orientation_rpy_rad)
        return (
            hover[0] + float(waypoint.position_offset_world[0]),
            hover[1] + float(waypoint.position_offset_world[1]),
            hover[2] + float(waypoint.position_offset_world[2]),
            *quaternion,
        )

    def _neutral_dock_posture(self) -> PostureTarget:
        joint_ids = sorted(
            {
                str(port.mechanical_limits["mechanism_joint_id"])
                for port in self.physical_model.dock_ports
                if port.mechanical_limits.get("mechanism_joint_id")
            }
        )
        module_ids = sorted(
            module.module_id
            for module in (
                self._current_morphology.modules
                if self._current_morphology is not None
                else []
            )
        )
        position_targets = {
            f"module_{module_id}:{joint_id}": 0.0
            for module_id in module_ids
            for joint_id in joint_ids
        }
        return PostureTarget(
            joint_pos_target=position_targets,
            joint_vel_target={key: 0.0 for key in position_targets},
            free_anchor_pose_targets={},
        )

    def _guard_conditions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "order4_phase_guard",
                "phase": self.phase,
                "waypoint_index": self._waypoint_index,
                "position_tolerance_m": self.config.position_tolerance_m,
                "attitude_tolerance_rad": self.config.attitude_tolerance_rad,
                "linear_speed_tolerance_mps": self.config.linear_speed_tolerance_mps,
                "angular_speed_tolerance_rad_s": self.config.angular_speed_tolerance_rad_s,
            },
            {
                "type": "simultaneous_reachability",
                "status": "not_applicable_no_active_assignments",
                "active_assignment_count": 0,
            },
        ]

    def _dwell_satisfied(
        self,
        condition: bool,
        *,
        now_s: float,
        required_s: float,
    ) -> bool:
        if not condition:
            self._guard_start_time_s = None
            return False
        if self._guard_start_time_s is None:
            self._guard_start_time_s = now_s
        return now_s - self._guard_start_time_s + 1.0e-12 >= required_s

    def _enter_safe_hold(self, pose: Pose7D, *, time_s: float, reason: str) -> None:
        self._safe_hold_pose = pose
        self._failure_reason = reason
        self._guard_start_time_s = None
        self._transition(
            "safe_hold",
            time_s=time_s,
            reason=reason,
            waypoint_index=self._waypoint_index,
        )

    def _transition(
        self,
        phase: Order4FreeFlightPhase,
        *,
        time_s: float,
        reason: str,
        waypoint_index: int | None,
        force_record: bool = False,
    ) -> None:
        previous = self._phase
        if previous == phase and not force_record:
            return
        self._phase = phase
        self._phase_start_time_s = float(time_s)
        self._transitions.append(
            Order4PhaseTransition(
                from_phase=previous,
                to_phase=phase,
                time_s=float(time_s),
                reason=reason,
                waypoint_index=waypoint_index,
            )
        )


class Order4FreeFlightTrajectoryRuntime:
    """Periodic pi_H replanning plus common relative-time trajectory execution."""

    def __init__(
        self,
        *,
        planner: DeterministicFreeFlightPlanner,
    ) -> None:
        self.planner = planner
        self.executor = ContactWrenchTrajectoryExecutor(
            expiry_grace_s=planner.config.trajectory_expiry_grace_s
        )
        self._last_replan_time_s: float | None = None
        self._plan_records: list[dict[str, object]] = []

    @property
    def active_trajectory(self) -> ContactWrenchTrajectory | None:
        return self.executor.trajectory

    @property
    def plan_records(self) -> list[dict[str, object]]:
        return [dict(record) for record in self._plan_records]

    def reset(self) -> None:
        self.planner.reset()
        self.executor.reset()
        self._last_replan_time_s = None
        self._plan_records = []

    def step(self, context: HighLevelPolicyContext) -> Order4TrajectoryRuntimeStep:
        observation = _validated_runtime_observation(context)
        mission = _mission_from_context(context)
        now_s = float(observation.time_s)
        replanned = False
        if (
            not self.executor.has_plan
            or self._last_replan_time_s is None
            or now_s - self._last_replan_time_s
            >= self.planner.config.update_period_s - 1.0e-9
        ):
            trajectory = self._plan_or_safe_hold(context)
            self.executor.install(trajectory, plan_start_time_s=now_s)
            self._last_replan_time_s = now_s
            replanned = True
            self._plan_records.append(
                {
                    "plan_sequence": self.executor.plan_sequence,
                    "plan_start_time_s": now_s,
                    "phase": self.planner.phase,
                    "waypoint_index": self.planner.waypoint_index,
                    "trajectory": trajectory.to_dict(),
                }
            )
        try:
            sample = self.executor.sample(time_s=now_s)
        except ContactWrenchTrajectoryRuntimeError as exc:
            trajectory = self.planner.force_safe_hold(
                context,
                reason=f"trajectory_runtime:{exc}",
            )
            self.executor.install(trajectory, plan_start_time_s=now_s)
            self._last_replan_time_s = now_s
            replanned = True
            sample = self.executor.sample(time_s=now_s)
        if sample.active_knot.contact_assignments:
            trajectory = self.planner.force_safe_hold(
                context,
                reason="order4_nonempty_contact_assignment",
            )
            self.executor.install(trajectory, plan_start_time_s=now_s)
            self._last_replan_time_s = now_s
            replanned = True
            sample = self.executor.sample(time_s=now_s)
        step = Order4TrajectoryRuntimeStep(
            runtime_version=ORDER4_FREE_FLIGHT_RUNTIME_VERSION,
            time_s=now_s,
            mission_hash=mission.mission_hash,
            phase=self.planner.phase,
            waypoint_index=self.planner.waypoint_index,
            mission_progress_ratio=self.planner.mission_progress_ratio(
                time_s=now_s
            ),
            plan_sequence=sample.plan_sequence,
            plan_start_time_s=sample.plan_start_time_s,
            plan_elapsed_s=sample.plan_elapsed_s,
            active_knot_index=sample.active_knot_index,
            next_knot_index=sample.next_knot_index,
            interpolation_ratio=sample.interpolation_ratio,
            replanned=replanned,
            safe_hold_active=self.planner.safe_hold_active,
            failure_reason=self.planner.failure_reason,
            reachability_status="not_applicable_no_active_assignments",
            active_knot=sample.active_knot,
        )
        step.validate()
        return step

    def _plan_or_safe_hold(
        self,
        context: HighLevelPolicyContext,
    ) -> ContactWrenchTrajectory:
        try:
            return self.planner.plan(context)
        except (RuntimeError, SchemaValidationError, ValueError) as exc:
            return self.planner.force_safe_hold(
                context,
                reason=f"planner_exception:{type(exc).__name__}:{exc}",
            )


def _mission_from_context(context: HighLevelPolicyContext) -> Order4FreeFlightMission:
    payload = context.irg.metadata.get(ORDER4_MISSION_METADATA_KEY)
    if not isinstance(payload, dict):
        raise SchemaValidationError(
            "Order4 HighLevelPolicyContext is missing its versioned mission"
        )
    mission = Order4FreeFlightMission.from_dict(payload)
    if context.irg.metadata.get(ORDER4_MISSION_HASH_METADATA_KEY) != mission.mission_hash:
        raise SchemaValidationError("Order4 IRG mission hash mismatch")
    if context.irg.task_id != context.interaction_envelope.task_id:
        raise SchemaValidationError("Order4 IRG/envelope task id mismatch")
    if context.contact_candidate_set.task_id != context.irg.task_id:
        raise SchemaValidationError("Order4 candidate/IRG task id mismatch")
    if context.contact_candidate_set.candidates:
        raise SchemaValidationError(
            "Order4 free-flight context must have no contact candidates"
        )
    if context.interaction_envelope.required_contact_count_range != (0, 0):
        raise SchemaValidationError(
            "Order4 free-flight envelope must require zero contacts"
        )
    return mission


def _validated_runtime_observation(
    context: HighLevelPolicyContext,
) -> RuntimeObservation:
    observation = context.runtime_observation
    if observation is None:
        raise SchemaValidationError(
            "Order4 deterministic planner requires RuntimeObservation"
        )
    observation.validate()
    if context.morphology_graph.graph_id != context.contact_candidate_set.morphology_graph_id:
        raise SchemaValidationError("Order4 candidate/morphology graph id mismatch")
    return observation


def _target_reached(
    pose: Pose7D,
    twist: Sequence[float],
    target: Pose7D,
    config: Order4DeterministicPlannerConfig,
) -> bool:
    return (
        _norm(
            [float(target[index]) - float(pose[index]) for index in range(3)]
        )
        <= config.position_tolerance_m
        and _quaternion_angle(pose[3:7], target[3:7])
        <= config.attitude_tolerance_rad
        and _norm(twist[:3]) <= config.linear_speed_tolerance_mps
        and _norm(twist[3:]) <= config.angular_speed_tolerance_rad_s
    )


def _knot_times(horizon_s: float, dt_s: float) -> list[float]:
    count = int(math.floor(horizon_s / dt_s + 1.0e-12))
    times = [index * dt_s for index in range(count + 1)]
    if horizon_s - times[-1] > 1.0e-9:
        times.append(horizon_s)
    else:
        times[-1] = horizon_s
    return times


def _interpolate_pose(start: Pose7D, end: Pose7D, ratio: float) -> Pose7D:
    position = tuple(
        float(start[index])
        + (float(end[index]) - float(start[index])) * ratio
        for index in range(3)
    )
    orientation = _quaternion_slerp(start[3:7], end[3:7], ratio)
    return (*position, *orientation)


def _rpy_to_quaternion(values: Sequence[float]) -> tuple[float, float, float, float]:
    roll, pitch, yaw = (float(value) for value in values)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return _normalize_quaternion(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )
    )


def _quaternion_slerp(
    start_values: Sequence[float],
    end_values: Sequence[float],
    ratio: float,
) -> tuple[float, float, float, float]:
    start = _normalize_quaternion(start_values)
    end = _normalize_quaternion(end_values)
    dot = sum(left * right for left, right in zip(start, end, strict=True))
    if dot < 0.0:
        end = tuple(-value for value in end)
        dot = -dot
    dot = min(max(dot, -1.0), 1.0)
    if dot > 0.9995:
        return _normalize_quaternion(
            tuple(
                left + (right - left) * ratio
                for left, right in zip(start, end, strict=True)
            )
        )
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    left_scale = math.sin((1.0 - ratio) * theta) / sin_theta
    right_scale = math.sin(ratio * theta) / sin_theta
    return _normalize_quaternion(
        tuple(
            left_scale * left + right_scale * right
            for left, right in zip(start, end, strict=True)
        )
    )


def _quaternion_angle(left: Sequence[float], right: Sequence[float]) -> float:
    left_q = _normalize_quaternion(left)
    right_q = _normalize_quaternion(right)
    dot = abs(sum(a * b for a, b in zip(left_q, right_q, strict=True)))
    return 2.0 * math.acos(min(max(dot, -1.0), 1.0))


def _body_tilt_rad(quaternion: Sequence[float]) -> float:
    x, y, _, w = _normalize_quaternion(quaternion)
    body_z_world_z = 1.0 - 2.0 * (x * x + y * y)
    return math.acos(min(max(body_z_world_z, -1.0), 1.0))


def _normalize_quaternion(
    values: Sequence[float],
) -> tuple[float, float, float, float]:
    if len(values) != 4:
        raise SchemaValidationError("Order4 quaternion must have four elements")
    data = tuple(float(value) for value in values)
    norm = math.sqrt(sum(value * value for value in data))
    if norm <= 1.0e-12:
        raise SchemaValidationError("Order4 quaternion must have non-zero norm")
    return tuple(value / norm for value in data)  # type: ignore[return-value]


def _smoothstep(value: float) -> float:
    clamped = min(max(float(value), 0.0), 1.0)
    return clamped * clamped * (3.0 - 2.0 * clamped)


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))
