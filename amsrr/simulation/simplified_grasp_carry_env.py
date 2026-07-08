from __future__ import annotations

import random
from dataclasses import dataclass, field
from math import sqrt

from amsrr.controllers.controller_base import ControllerContext
from amsrr.controllers.qpid_controller import QPIDController
from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.policies.contact_candidate_sampler import ContactCandidateSampler
from amsrr.policies.contact_wrench_trajectory import GraspCarryBaselinePlanner
from amsrr.policies.design_policy_base import DesignPolicyContext, FixedSimpleDesignPolicy
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.policies.low_level_policy_base import BaselineLowLevelPolicy, LowLevelPolicyContext, select_active_knot
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import Pose7D, SchemaBase, SchemaValidationError, Vector3
from amsrr.schemas.contact_candidates import ContactCandidateSet
from amsrr.schemas.interaction_envelope import InteractionEnvelope
from amsrr.schemas.irg import InteractionRequirementGraph
from amsrr.schemas.morphology import DesignOutput, MorphologyGraph
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import ContactAssignment, ContactWrenchTrajectory, ControllerCommand, ControllerStatus, PolicyCommand
from amsrr.schemas.runtime import ContactState, ModuleRuntimeState, ObjectRuntimeState, RuntimeObservation, TaskProgressState
from amsrr.schemas.task_spec import TaskSpec


@dataclass(frozen=True)
class SimplifiedGraspCarryEnvConfig:
    dt_s: float = 0.1
    max_episode_steps: int = 40
    goal_tolerance_m: float = 0.05
    object_tracking_speed_mps: float = 10.0
    initial_position_jitter_m: float = 0.02
    contact_break_force_n: float = 100.0
    robot_model_config_path: str = "configs/robot/robot_model.yaml"

    def __post_init__(self) -> None:
        if self.dt_s <= 0.0:
            raise SchemaValidationError("SimplifiedGraspCarryEnvConfig.dt_s must be positive")
        if self.max_episode_steps <= 0:
            raise SchemaValidationError("SimplifiedGraspCarryEnvConfig.max_episode_steps must be positive")
        if self.goal_tolerance_m < 0.0:
            raise SchemaValidationError("SimplifiedGraspCarryEnvConfig.goal_tolerance_m must be non-negative")
        if self.object_tracking_speed_mps < 0.0:
            raise SchemaValidationError("SimplifiedGraspCarryEnvConfig.object_tracking_speed_mps must be non-negative")
        if self.initial_position_jitter_m < 0.0:
            raise SchemaValidationError("SimplifiedGraspCarryEnvConfig.initial_position_jitter_m must be non-negative")


@dataclass
class SimplifiedGraspCarryBuildArtifacts:
    task_spec: TaskSpec
    irg: InteractionRequirementGraph
    interaction_envelope: InteractionEnvelope
    physical_model: PhysicalModel
    design_output: DesignOutput
    contact_candidate_set: ContactCandidateSet
    contact_wrench_trajectory: ContactWrenchTrajectory
    design_source: str


@dataclass
class SimplifiedEpisodeResult(SchemaBase):
    episode_id: str
    steps: int
    success: bool
    crashed: bool
    failure_reason: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class SimplifiedBatchRunResult(SchemaBase):
    episode_count: int
    success_count: int
    crash_count: int
    failure_count: int
    metrics: dict[str, float] = field(default_factory=dict)


class SimplifiedGraspCarryEnv:
    """Interface-backed P1 grasp/carry environment with simplified contact."""

    def __init__(
        self,
        task_spec: TaskSpec,
        *,
        config: SimplifiedGraspCarryEnvConfig | None = None,
        design_output: DesignOutput | None = None,
        assembled_morphology: MorphologyGraph | None = None,
        low_level_policy: BaselineLowLevelPolicy | None = None,
        controller: QPIDController | None = None,
    ) -> None:
        self.config = config or SimplifiedGraspCarryEnvConfig()
        self.low_level_policy = low_level_policy or BaselineLowLevelPolicy()
        self.controller = controller or QPIDController()
        self.artifacts = _build_artifacts(
            task_spec,
            self.config,
            design_output_override=design_output,
            morphology_override=assembled_morphology,
        )
        self._rng = random.Random(0)
        self._episode_id = "uninitialized"
        self._step_index = 0
        self._object_states: list[ObjectRuntimeState] = []
        self._attached = False
        self._ever_attached = False
        self._success = False
        self._failure_reason: str | None = None
        self._controller_status = ControllerStatus(status="ok", qp_feasible=True, active_mode="reset")
        self._last_policy_command: PolicyCommand | None = None
        self._last_controller_command: ControllerCommand | None = None
        self._runtime_observation = self.reset(seed=0, episode_id="initial")

    def reset(
        self,
        task_spec: TaskSpec | None = None,
        morphology: MorphologyGraph | None = None,
        *,
        design_output: DesignOutput | None = None,
        assembled_morphology: MorphologyGraph | None = None,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> RuntimeObservation:
        if morphology is not None and assembled_morphology is not None:
            raise SchemaValidationError(
                "SimplifiedGraspCarryEnv.reset accepts either morphology or assembled_morphology, not both"
            )
        morphology_override = assembled_morphology or morphology
        if task_spec is not None or design_output is not None or morphology_override is not None:
            rebuild_task = task_spec or self.artifacts.task_spec
            self.artifacts = _build_artifacts(
                rebuild_task,
                self.config,
                design_output_override=design_output,
                morphology_override=morphology_override,
            )
        if seed is not None:
            self._rng = random.Random(seed)
        self._episode_id = episode_id or f"episode_{seed if seed is not None else 0}"
        self._step_index = 0
        self._attached = False
        self._ever_attached = False
        self._success = False
        self._failure_reason = None
        self._controller_status = ControllerStatus(status="ok", qp_feasible=True, active_mode="reset")
        self._last_policy_command = None
        self._last_controller_command = None
        self._object_states = [
            ObjectRuntimeState(
                object_id=obj.object_id,
                pose_world=_jitter_pose_xy(obj.pose_world, self.config.initial_position_jitter_m, self._rng),
                twist_world=[0.0] * 6,
            )
            for obj in self.artifacts.task_spec.scene.objects
        ]
        self._runtime_observation = self._make_observation()
        return self._runtime_observation

    def step(self, controller_command: ControllerCommand) -> RuntimeObservation:
        if self._failure_reason is not None or self._success:
            self._last_controller_command = controller_command
            self._runtime_observation = self._make_observation()
            return self._runtime_observation

        self._last_controller_command = controller_command
        self._controller_status = controller_command.controller_status
        active_knot = select_active_knot(self._low_level_context())
        self._update_contact_state(active_knot.contact_assignments, controller_command)
        self._move_objects(active_knot)
        self._success = self._goal_distance() <= self.config.goal_tolerance_m and self._ever_attached
        self._step_index += 1
        if self._step_index >= self.config.max_episode_steps and not self._success:
            self._failure_reason = "timeout"
        self._runtime_observation = self._make_observation()
        return self._runtime_observation

    def get_runtime_observation(self) -> RuntimeObservation:
        return self._runtime_observation

    def run_episode(self, *, seed: int | None = None, episode_id: str | None = None) -> SimplifiedEpisodeResult:
        crashed = False
        failure_reason: str | None = None
        policy_count = 0
        controller_count = 0
        try:
            self.reset(seed=seed, episode_id=episode_id)
            while not self._success and self._failure_reason is None:
                active_knot = select_active_knot(self._low_level_context())
                policy_command = self.low_level_policy.command(self._low_level_context(active_knot=active_knot))
                controller_command = self.controller.compute(
                    ControllerContext(
                        runtime_observation=self.get_runtime_observation(),
                        morphology_graph=self.artifacts.design_output.target_morphology,
                        physical_model=self.artifacts.physical_model,
                        active_knot=active_knot,
                        policy_command=policy_command,
                        previous_command=self._last_controller_command,
                        control_dt_s=self.config.dt_s,
                    )
                )
                self._last_policy_command = policy_command
                policy_count += 1
                controller_count += 1
                self.step(controller_command)
        except Exception as exc:  # pragma: no cover - exercised only by regression failures.
            crashed = True
            failure_reason = f"{type(exc).__name__}: {exc}"
        if self._failure_reason is not None:
            failure_reason = self._failure_reason
        return SimplifiedEpisodeResult(
            episode_id=self._episode_id,
            steps=self._step_index,
            success=self._success and not crashed,
            crashed=crashed,
            failure_reason=failure_reason,
            metrics={
                "goal_distance_m": self._goal_distance(),
                "attached": 1.0 if self._attached else 0.0,
                "ever_attached": 1.0 if self._ever_attached else 0.0,
                "policy_command_count": float(policy_count),
                "controller_command_count": float(controller_count),
                "qp_feasible": 1.0 if self._controller_status.qp_feasible else 0.0,
            },
        )

    def _low_level_context(self, *, active_knot=None) -> LowLevelPolicyContext:
        return LowLevelPolicyContext(
            runtime_observation=self.get_runtime_observation(),
            morphology_graph=self.artifacts.design_output.target_morphology,
            physical_model=self.artifacts.physical_model,
            contact_wrench_trajectory=self.artifacts.contact_wrench_trajectory,
            active_knot=active_knot,
            controller_status=self._controller_status,
        )

    def _update_contact_state(
        self,
        assignments: list[ContactAssignment],
        controller_command: ControllerCommand,
    ) -> None:
        if not assignments:
            return
        schedule_states = {assignment.schedule_state for assignment in assignments}
        if controller_command.controller_status.status == "fault":
            self._attached = False
            self._failure_reason = "controller_fault"
            return
        if "release" in schedule_states and self._goal_distance() <= self.config.goal_tolerance_m:
            self._attached = False
            return
        if schedule_states.intersection({"attach", "maintain", "slide"}):
            if _max_assignment_force(assignments) > self.config.contact_break_force_n:
                self._attached = False
                self._failure_reason = "contact_break_force"
                return
            self._attached = True
            self._ever_attached = True

    def _move_objects(self, active_knot) -> None:
        if not self._attached and not self._ever_attached:
            return
        targets = [target for target in active_knot.object_targets if target.pose_target_world is not None]
        if not targets:
            return
        object_by_id = {state.object_id: state for state in self._object_states}
        updated: list[ObjectRuntimeState] = []
        for state in self._object_states:
            target = next((item for item in targets if item.object_id == state.object_id), None)
            if target is None or target.pose_target_world is None:
                updated.append(state)
                continue
            pose, twist = _step_pose_toward(
                state.pose_world,
                target.pose_target_world,
                self.config.object_tracking_speed_mps * self.config.dt_s,
                self.config.dt_s,
            )
            updated.append(
                ObjectRuntimeState(
                    object_id=state.object_id,
                    pose_world=pose,
                    twist_world=twist,
                    generalized_q=state.generalized_q,
                    generalized_qdot=state.generalized_qdot,
                )
            )
        if set(object_by_id) != {state.object_id for state in updated}:
            raise SchemaValidationError("internal object update changed object ids")
        self._object_states = updated

    def _make_observation(self) -> RuntimeObservation:
        time_s = float(self._step_index) * self.config.dt_s
        return RuntimeObservation(
            time_s=time_s,
            morphology_graph=self.artifacts.design_output.target_morphology,
            module_states=_module_states(self.artifacts.design_output.target_morphology),
            object_states=list(self._object_states),
            contact_states=self._contact_states(),
            controller_status=self._controller_status,
            task_progress=TaskProgressState(
                phase_label=_phase_label(self),
                progress_ratio=_clamp01(1.0 - self._goal_distance() / max(self._initial_goal_distance(), 1.0e-9)),
                success=self._success,
                failure_reason=self._failure_reason,
                metrics={
                    "goal_distance_m": self._goal_distance(),
                    "attached": 1.0 if self._attached else 0.0,
                    "ever_attached": 1.0 if self._ever_attached else 0.0,
                },
            ),
        )

    def _contact_states(self) -> list[ContactState]:
        if not self._attached:
            return []
        active_knot = select_active_knot(self._low_level_context())
        candidate_by_id = {
            candidate.candidate_id: candidate
            for candidate in self.artifacts.contact_candidate_set.candidates
        }
        states: list[ContactState] = []
        for assignment in active_knot.contact_assignments:
            if assignment.schedule_state not in {"attach", "maintain", "slide"}:
                continue
            candidate = candidate_by_id.get(assignment.candidate_id)
            if candidate is None:
                continue
            states.append(
                ContactState(
                    contact_id=f"assignment:{assignment.candidate_id}",
                    entity_a=f"anchor:{assignment.anchor_id}",
                    entity_b=candidate.target_entity_id,
                    contact_pose_world=candidate.contact_pose_world,
                    normal_world=candidate.normal_world,
                    wrench_world=assignment.wrench_target,
                    active=True,
                    metadata={
                        "slot_id": assignment.slot_id,
                        "contact_mode": assignment.contact_mode.value,
                        "schedule_state": assignment.schedule_state,
                    },
                )
            )
        return states

    def _goal_distance(self) -> float:
        goal = _object_goal_pose(self.artifacts.task_spec)
        if goal is None or not self._object_states:
            return 0.0
        target_object_id = _object_goal_id(self.artifacts.task_spec)
        for state in self._object_states:
            if state.object_id == target_object_id:
                return _position_distance(state.pose_world, goal)
        return 0.0

    def _initial_goal_distance(self) -> float:
        goal = _object_goal_pose(self.artifacts.task_spec)
        target_object_id = _object_goal_id(self.artifacts.task_spec)
        if goal is None:
            return 0.0
        for obj in self.artifacts.task_spec.scene.objects:
            if obj.object_id == target_object_id:
                return _position_distance(obj.pose_world, goal)
        return 0.0


def run_crash_free_episodes(
    env: SimplifiedGraspCarryEnv,
    *,
    episode_count: int = 1000,
    seed: int = 0,
) -> SimplifiedBatchRunResult:
    if episode_count <= 0:
        raise SchemaValidationError("episode_count must be positive")
    success_count = 0
    crash_count = 0
    failure_count = 0
    total_steps = 0
    final_goal_distance = 0.0
    for idx in range(episode_count):
        result = env.run_episode(seed=seed + idx, episode_id=f"p1_simplified_{idx:04d}")
        success_count += 1 if result.success else 0
        crash_count += 1 if result.crashed else 0
        failure_count += 1 if not result.success else 0
        total_steps += result.steps
        final_goal_distance = result.metrics.get("goal_distance_m", final_goal_distance)
    return SimplifiedBatchRunResult(
        episode_count=episode_count,
        success_count=success_count,
        crash_count=crash_count,
        failure_count=failure_count,
        metrics={
            "success_rate": float(success_count) / float(episode_count),
            "crash_rate": float(crash_count) / float(episode_count),
            "mean_steps": float(total_steps) / float(episode_count),
            "last_goal_distance_m": final_goal_distance,
        },
    )


def _build_artifacts(
    task_spec: TaskSpec,
    config: SimplifiedGraspCarryEnvConfig,
    *,
    design_output_override: DesignOutput | None = None,
    morphology_override: MorphologyGraph | None = None,
) -> SimplifiedGraspCarryBuildArtifacts:
    builder_result = IRGBuilder().build_with_scene_graph(task_spec)
    irg = builder_result.irg
    envelope = InteractionEnvelopeExtractor().extract(irg)
    physical_model = build_physical_model_from_config(config.robot_model_config_path)
    if design_output_override is None:
        design = FixedSimpleDesignPolicy().design(
            DesignPolicyContext(
                task_spec=task_spec,
                irg=irg,
                interaction_envelope=envelope,
                physical_model=physical_model,
            )
        )
        design_source = "fixed_simple"
    else:
        design = design_output_override
        design_source = "external_design_output"
    if morphology_override is not None:
        design = _design_with_target_morphology(design, morphology_override)
        design_source = (
            "external_design_output_with_assembled_morphology"
            if design_output_override is not None
            else "fixed_simple_with_morphology_override"
        )
    candidate_set = ContactCandidateSampler().sample(
        task_spec=task_spec,
        irg=irg,
        interaction_envelope=envelope,
        morphology_graph=design.target_morphology,
        geometry_descriptors=builder_result.scene_graph.geometry_descriptors,
    )
    high_context = HighLevelPolicyContext(
        irg=irg,
        interaction_envelope=envelope,
        morphology_graph=design.target_morphology,
        contact_candidate_set=candidate_set,
    )
    trajectory = GraspCarryBaselinePlanner().plan(high_context)
    return SimplifiedGraspCarryBuildArtifacts(
        task_spec=task_spec,
        irg=irg,
        interaction_envelope=envelope,
        physical_model=physical_model,
        design_output=design,
        contact_candidate_set=candidate_set,
        contact_wrench_trajectory=trajectory,
        design_source=design_source,
    )


def _design_with_target_morphology(design: DesignOutput, morphology_graph: MorphologyGraph) -> DesignOutput:
    return DesignOutput(
        task_id=design.task_id,
        irg_id=design.irg_id,
        target_morphology=morphology_graph,
        module_roles=dict(design.module_roles),
        slot_anchor_binding_prior=list(design.slot_anchor_binding_prior),
        design_actions=list(design.design_actions),
        design_logprobs=list(design.design_logprobs) if design.design_logprobs is not None else None,
        design_scores=dict(design.design_scores),
    )


def _module_states(morphology_graph: MorphologyGraph) -> list[ModuleRuntimeState]:
    return [
        ModuleRuntimeState(
            module_id=module.module_id,
            pose_world=module.pose_in_design_frame,
            twist_world=[0.0] * 6,
            joint_positions={},
            joint_velocities={},
            health=module.health,
        )
        for module in morphology_graph.modules
    ]


def _jitter_pose_xy(pose: Pose7D, jitter_m: float, rng: random.Random) -> Pose7D:
    if jitter_m == 0.0:
        return pose
    dx = rng.uniform(-jitter_m, jitter_m)
    dy = rng.uniform(-jitter_m, jitter_m)
    return (pose[0] + dx, pose[1] + dy, pose[2], pose[3], pose[4], pose[5], pose[6])


def _step_pose_toward(
    current: Pose7D,
    target: Pose7D,
    max_step_m: float,
    dt_s: float,
) -> tuple[Pose7D, list[float]]:
    delta = (target[0] - current[0], target[1] - current[1], target[2] - current[2])
    distance = _norm3(delta)
    if distance <= max(max_step_m, 1.0e-12):
        moved = (target[0], target[1], target[2])
        twist = [
            (target[0] - current[0]) / dt_s,
            (target[1] - current[1]) / dt_s,
            (target[2] - current[2]) / dt_s,
            0.0,
            0.0,
            0.0,
        ]
    else:
        scale = max_step_m / distance
        moved = (current[0] + delta[0] * scale, current[1] + delta[1] * scale, current[2] + delta[2] * scale)
        twist = [delta[0] * scale / dt_s, delta[1] * scale / dt_s, delta[2] * scale / dt_s, 0.0, 0.0, 0.0]
    return (moved[0], moved[1], moved[2], target[3], target[4], target[5], target[6]), twist


def _max_assignment_force(assignments: list[ContactAssignment]) -> float:
    max_force = 0.0
    for assignment in assignments:
        if assignment.wrench_target is None:
            continue
        max_force = max(max_force, _norm3((assignment.wrench_target[0], assignment.wrench_target[1], assignment.wrench_target[2])))
    return max_force


def _phase_label(env: SimplifiedGraspCarryEnv) -> str:
    if env._success:
        return "success"
    if env._failure_reason is not None:
        return "failure"
    if env._attached:
        return "transport_object"
    return "establish_contact"


def _object_goal_pose(task_spec: TaskSpec) -> Pose7D | None:
    for goal in task_spec.goals:
        if goal.goal_type == "object_pose" and goal.target_pose_world is not None:
            return goal.target_pose_world
    return None


def _object_goal_id(task_spec: TaskSpec) -> str | None:
    for goal in task_spec.goals:
        if goal.goal_type == "object_pose":
            return goal.target_entity_id
    return None


def _position_distance(left: Pose7D, right: Pose7D) -> float:
    return _norm3((right[0] - left[0], right[1] - left[1], right[2] - left[2]))


def _norm3(vector: Vector3) -> float:
    return sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])


def _clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)
