"""Causal pi_H execution on the provisional nominal controller.

The task-specific eight-phase adapter is declared by this grasp/carry IRG. Raw
PhysX contacts remain reward/evaluation evidence, never transition authority.
"""

from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
import json
import math
from types import SimpleNamespace

import torch

from amsrr.policies.high_level_requests import (
    RequestCatalogBuilder,
    contact_surface_hash,
)
from amsrr.policies.request_actor_critic import RequestActorCritic
from amsrr.schemas.high_level import ActiveExecutionState
from amsrr.schemas.runtime import (
    RuntimeObservation,
    ModuleRuntimeState,
    ObjectRuntimeState,
    TaskProgressState,
)
from amsrr.schemas.policies import ControllerStatus
from amsrr.training.request_imitation import teacher_request_scene, PHASES
from amsrr.training.request_ppo import serialize_encoding
from amsrr.training.order9_deployable_phase_gate import _quaternion_distance


class ContactPointVelocityObserver:
    """Causal sampled slip velocity from measured points in the object frame.

    The first observation after reset/binding change is invalid for a velocity
    guard. No target pose, contact force or simulator velocity enters this estimate.
    """

    contract_version = "observed_contact_point_difference_v1"

    def __init__(self):
        self.reset()

    def reset(self):
        self.previous = None
        self.previous_time = None
        self.binding = None

    def update(self, points_world, object_pose_world, *, time_s, binding):
        if (points_world.ndim != 3 or points_world.shape[-1] != 3
                or object_pose_world.shape != (points_world.shape[0], 7)
                or not math.isfinite(time_s)
                or not torch.isfinite(points_world).all()
                or not torch.isfinite(object_pose_world).all()):
            raise ValueError("invalid observed contact point sample")
        quaternion = object_pose_world[:, 3:]
        norm = quaternion.norm(dim=-1, keepdim=True)
        if bool((norm < 1e-8).any()):
            raise ValueError("invalid observed object orientation")
        quaternion = quaternion / norm
        offset = points_world - object_pose_world[:, None, :3]
        xyz = -quaternion[:, None, :3].expand_as(offset)
        cross = torch.cross(xyz, offset, dim=-1)
        relative = offset + 2 * (quaternion[:, None, 3:] * cross
                                + torch.cross(xyz, cross, dim=-1))
        valid = self.previous is not None and binding == self.binding
        if valid:
            dt = time_s - self.previous_time
            if dt <= 0 or relative.shape != self.previous.shape:
                raise ValueError("contact velocity requires monotonic samples and stable binding")
            speed = ((relative - self.previous) / dt).norm(dim=-1)
        else:
            speed = torch.zeros_like(relative[..., 0])
        self.previous = relative.detach().clone()
        self.previous_time = float(time_s)
        self.binding = binding
        return speed, valid


def object_pose_goal_deadlines(task, times_s, poses_world):
    """Evaluate authored deadlines from the completed physical trajectory.

    This is outcome/reward evidence only. Reaching the pose before the deadline
    may be followed by release and retreat after it. Final pose compliance is
    checked separately, so a transient hit cannot excuse drift after release.
    No tolerance is enlarged.
    The current grasp/carry collector observes one object per environment.
    """
    times = torch.as_tensor(times_s, dtype=torch.float64)
    poses = torch.as_tensor(poses_world, dtype=torch.float64)
    if (
        times.ndim != 1
        or not len(times)
        or poses.shape != (len(times), 7)
        or not torch.isfinite(times).all()
        or not torch.isfinite(poses).all()
        or (times < 0).any()
        or (times[1:] < times[:-1]).any()
        or (torch.linalg.vector_norm(poses[:, 3:], dim=-1) < 1e-8).any()
    ):
        raise ValueError("invalid physical object trajectory for deadline evaluation")
    goals = [g for g in task.goals if g.goal_type == "object_pose"]
    if not goals or len(task.scene.objects) != 1:
        raise ValueError("request deadline evaluation requires one observed object")
    outcomes = []
    for goal in goals:
        if (
            goal.target_entity_id != task.scene.objects[0].object_id
            or goal.target_pose_world is None
            or goal.tolerance_pos_m is None
            or goal.tolerance_rot_rad is None
        ):
            raise ValueError("object goal missing target or pose tolerances")
        target = poses.new_tensor(goal.target_pose_world)
        position_error = torch.linalg.vector_norm(poses[:, :3] - target[:3], dim=-1)
        rotation_error = _quaternion_distance(
            poses[:, 3:], target[None, 3:].expand(len(poses), -1)
        )
        reached = (position_error <= goal.tolerance_pos_m) & (
            rotation_error <= goal.tolerance_rot_rad
        )
        hits = torch.nonzero(reached).flatten()
        first = float(times[hits[0]]) if len(hits) else None
        outcomes.append(
            dict(
                goal_id=goal.goal_id,
                first_pose_goal_reach_s=first,
                time_limit_s=goal.time_limit_s,
                tolerance_pos_m=goal.tolerance_pos_m,
                tolerance_rot_rad=goal.tolerance_rot_rad,
                passed=first is not None and first <= goal.time_limit_s,
                final_pose_within_tolerance=bool(reached[-1]),
                final_position_error_m=float(position_error[-1]),
                final_orientation_error_rad=float(rotation_error[-1]),
            )
        )
    return outcomes


def apply_object_goal_outcomes(record, outcomes, *, deadline_terminated):
    """Adjudicate exact goal completion while retaining the physical root cause."""
    on_time = all(goal["passed"] for goal in outcomes)
    final_pose = all(goal["final_pose_within_tolerance"] for goal in outcomes)
    if on_time and final_pose:
        return
    record["task_success"] = False
    record["no_fallback_success"] = False
    if deadline_terminated:
        record["metrics"]["timeout"] = 1.0
        if not record["safety_failure"]:
            record["failure_reason"] = "object_pose_goal_deadline_missed"
    if not record.get("failure_reason"):
        record["failure_reason"] = (
            "object_pose_goal_not_maintained" if on_time and not final_pose
            else "object_pose_goal_not_reached"
        )


class ObjectPoseGoalDeadlineMonitor:
    """Terminate an impossible episode without changing actor or phase guards.

    The clock and post-step pose must be the same values archived in the physical
    trajectory. A goal reached on time remains satisfied during later release.
    """

    def __init__(self, task):
        self.task = task
        self.first_arrivals = {}
        self.last_time_s = None
        self.deadline_missed = False
        self.termination_time_s = None

    def apply(self, reward, *, time_s, object_pose_world):
        if self.last_time_s is not None and time_s < self.last_time_s:
            raise ValueError("deadline observation clock moved backwards")
        outcomes = object_pose_goal_deadlines(
            self.task,
            [time_s],
            torch.as_tensor(object_pose_world).detach().cpu().double()[None],
        )
        self.last_time_s = float(time_s)
        for goal in outcomes:
            identity = goal["goal_id"]
            if goal["passed"]:
                self.first_arrivals.setdefault(identity, float(time_s))
            if time_s > goal["time_limit_s"] and identity not in self.first_arrivals:
                self.deadline_missed = True
                if self.termination_time_s is None:
                    self.termination_time_s = float(time_s)
        if not self.deadline_missed:
            return reward
        return replace(
            reward,
            timeout=torch.ones_like(reward.timeout),
            terminal_failure=torch.ones_like(reward.terminal_failure),
            phase_success=torch.zeros_like(reward.phase_success),
        )


def validate_initial_encoding(expected, actual, *, tolerance=2e-5):
    """Audit every model input across preparation and the live reset boundary."""
    if expected.keys() != actual.keys():
        raise ValueError("initial encoding fields changed")
    for key, value in expected.items():
        other = actual[key]
        if isinstance(value, dict):
            validate_initial_encoding(value, other, tolerance=tolerance)
        elif value.shape != other.shape or (
            not torch.allclose(value, other, atol=tolerance, rtol=0.)
            if value.is_floating_point() else not torch.equal(value, other)
        ):
            raise ValueError(f"initial encoding changed: {key}")


class RequestEventSupervisor:
    """State for one environment; every stochastic choice is replayable by PPO."""

    goal_completion_contract = "authored_object_pose_goal_guard_v1"

    def __init__(
        self,
        *,
        model,
        bundle,
        task,
        physical,
        expected_group,
        seed=17,
        sample=True,
        dt=0.02,
        diagnostic_phase=None,
        initial_task=None,
        initial_candidates=None,
    ):
        self.model = model.cpu().eval()
        self.environment_origin = torch.zeros(3)
        self.bundle = bundle
        self.task = task
        tracked_goal = next(g for g in task.goals
            if g.goal_type == "object_pose"
            and g.target_entity_id == task.scene.objects[0].object_id)
        self.final_goal_position_tolerance_m = tracked_goal.tolerance_pos_m
        self.final_goal_orientation_tolerance_rad = tracked_goal.tolerance_rot_rad
        self.initial_task = task if initial_task is None else initial_task
        self.physical = physical
        initial_bundle = SimpleNamespace(
            morphology=bundle.morphology,
            contact_candidate_set=(bundle.contact_candidate_set
                if initial_candidates is None else initial_candidates),
        )
        self.initial_scene, self.initial_phase_ids = teacher_request_scene(
            initial_bundle, self.initial_task, plan_committed=False
        )
        self.scene, self.phase_ids = teacher_request_scene(bundle, task, plan_committed=True)
        self.group = next(
            g
            for g in bundle.contact_candidate_set.group_proposals
            if g.group_id == expected_group
        )
        self.expected_group = expected_group
        self.sample = sample
        self.dt = dt
        self.generator = torch.Generator().manual_seed(seed)
        self.events = []
        self.trace = []
        self.started = False
        self.time_s = 0.0
        self.phase_started_s = 0.0
        self.phase = -1
        self.next_decision_s = 0.0
        self.pending = None
        self.dwell = 0.0
        self.release_latched = False
        self.module_ids = None
        self.joint_ids = None
        self.plan_id = bundle.provenance["compiled_plan_id"]
        self.initial_object = None
        self.maximum_lift = 0.0
        self.maximum_transport = 0.0
        self.final_object = None
        self.initial_runtime_check = {}
        self.evaluator_phase_exits = []
        self.diagnostic_phase = diagnostic_phase
        self.contact_velocity_observer = ContactPointVelocityObserver()

    def observation(self, state, *, qp_feasible=True):
        if state.module_pose_world.shape[0] != 1:
            raise ValueError("request adapter supports exactly one environment")
        def local_pose(value):
            pose = value.detach().cpu().clone()
            pose[:3] -= self.environment_origin.to(dtype=pose.dtype)
            return tuple(pose.tolist())

        return RuntimeObservation(
            self.time_s,
            self.bundle.morphology,
            [
                ModuleRuntimeState(
                    mid,
                    local_pose(state.module_pose_world[0, i]),
                    state.module_twist_world[0, i].cpu().tolist(),
                    dict(
                        zip(
                            self.joint_ids,
                            state.local_joint_positions_rad[0, i].cpu().tolist(),
                        )
                    ),
                    dict(
                        zip(
                            self.joint_ids,
                            state.local_joint_velocities_radps[0, i].cpu().tolist(),
                        )
                    ),
                )
                for i, mid in enumerate(self.module_ids)
            ],
            [
                ObjectRuntimeState(
                    self.task.scene.objects[0].object_id,
                    local_pose(state.object_pose_world[0]),
                    state.object_twist_world[0].cpu().tolist(),
                )
            ],
            [],
            ControllerStatus("ok" if qp_feasible else "infeasible", bool(qp_feasible)),
            TaskProgressState(),
        )

    def context(self, observation, phase, *, initial=False):
        phase_name = PHASES[phase]
        state = ActiveExecutionState(
            (self.initial_phase_ids if initial else self.phase_ids)[phase_name],
            self.phase_started_s,
            contact_group_id=(
                None
                if initial or phase_name in ("retreat", "settle")
                else self.group.group_id
            ),
            plan_id=None if initial else self.plan_id,
        )
        if not initial and phase_name in (
            "contact_acquisition",
            "lift",
            "transport",
            "place",
            "release",
        ):
            authored = deepcopy(observation)
            for obj in authored.object_states:
                obj.pose_world = next(
                    o.pose_world
                    for o in self.task.scene.objects
                    if o.object_id == obj.object_id
                )
            for c in self.bundle.contact_candidate_set.candidates:
                if c.candidate_id in self.group.candidate_ids:
                    state.contact_bindings[c.candidate_id] = (
                        c.slot_id,
                        c.anchor_id,
                        c.target_entity_id,
                    )
                    state.contact_surface_hashes[c.candidate_id] = contact_surface_hash(
                        c, authored
                    )
        scene = replace(
            self.initial_scene if initial else self.scene,
            runtime_observation=observation,
        )
        return RequestCatalogBuilder().build(
            scene,
            task_spec=self.initial_task if initial else self.task,
            physical_model=self.physical,
            execution_state=state,
        )

    def choose(self, context):
        result = self.model.decide(
            context, deterministic=not self.sample, generator=self.generator
        )
        if self.events:
            self.events[-1]["end_time_s"] = self.time_s
        record = dict(
            encoding=serialize_encoding(result["encoded"]),
            action=result["action"],
            log_prob=result["log_prob"],
            value=result["value"],
            reward=0.0,
            reward_timing="observed_reward_time_v1",
            reward_events=[],
            done=False,
            request=result["request"].to_dict(),
            requests=result["requests"],
            snapshot_hash=result["snapshot_hash"],
            time_s=self.time_s,
            phase=self.phase,
            sampling="categorical" if self.sample else "greedy",
            sampling_order=result["sampling_order"],
        )
        self.events.append(record)
        return result["request"]

    def add_reward(self, amount):
        """Keep physical reward time distinct from the preceding decision time."""
        event = self.events[-1]
        event["reward"] += amount
        event["reward_events"].append((self.time_s, amount))

    def begin(self, controller, state, target=None):
        if self.started:
            return
        self.module_ids = controller.builder.module_ids
        self.joint_ids = controller.builder.local_joint_ids
        self.phase = 0 if self.diagnostic_phase is None else self.diagnostic_phase
        if self.diagnostic_phase is not None:
            self.time_s = (
                float(target.phase_progress[0])
                * self.bundle.phase_trajectories[PHASES[self.phase]].horizon_s
            )
            self.release_latched = self.phase > 5
        obs = self.observation(state)
        context = self.context(obs, self.phase, initial=self.diagnostic_phase is None)
        request = self.choose(context)
        self.initial_runtime_check = dict(
            selected_group=request.contact_group_id,
            expected_planned_group=self.expected_group,
            snapshot_hash=context.catalog.snapshot_hash,
            observation=obs.to_dict(),
        )
        if (
            self.diagnostic_phase is None
            and request.contact_group_id != self.expected_group
        ):
            raise ValueError(
                "live initial actor choice differs from prepared request; replan required"
            )
        self.started = True
        self.initial_object = state.object_pose_world[0, :3].detach().cpu().clone()
        print("REQUEST_INITIAL=" + json.dumps(self.initial_runtime_check), flush=True)

    def step(
        self,
        *,
        phase_index,
        phase_elapsed_s,
        target,
        state,
        control_model,
        surface_distance,
        relative_speed,
        motor_load,
        qp_feasible,
        privileged_reward,
        grasp_position_world=None,
    ):
        """Actor and guards see only the explicit observation/proprioception fields.

        privileged_reward is accumulated after observations and action selection;
        its phase-success bit is recorded only for independent outcome auditing.
        """
        if not self.started:
            raise RuntimeError("actor has not made the initial request")
        phase = int(phase_index[0])
        self.time_s += self.dt
        algebraic_relative_speed = relative_speed
        contact_velocity_valid = True
        if grasp_position_world is not None:
            relative_speed, contact_velocity_valid = self.contact_velocity_observer.update(
                grasp_position_world, state.object_pose_world, time_s=self.time_s,
                binding=(self.task.scene.objects[0].object_id, self.expected_group),
            )
        if phase != self.phase:
            if phase != self.phase + 1:
                raise ValueError("nonadjacent execution transition")
            self.phase = phase
            self.phase_started_s = self.time_s - float(phase_elapsed_s[0])
            self.pending = None
            self.dwell = 0.0
            self.next_decision_s = self.time_s
        obj = state.object_pose_world[0, :3].detach().cpu()
        self.final_object = obj
        self.maximum_lift = max(
            self.maximum_lift, float(obj[2] - self.initial_object[2])
        )
        self.maximum_transport = max(
            self.maximum_transport,
            float(torch.linalg.vector_norm(obj[:2] - self.initial_object[:2])),
        )
        robot_error = float(
            torch.linalg.vector_norm(
                control_model.body_pose_world[0, :3]
                - target.phase_goal_robot_root_pose_world[0, :3]
            )
        )
        object_error = float(
            torch.linalg.vector_norm(
                state.object_pose_world[0, :3]
                - target.phase_goal_object_pose_world[0, :3]
            )
        )
        robot_speed = float(
            torch.linalg.vector_norm(control_model.body_twist_world[0, :3])
        )
        object_speed = float(torch.linalg.vector_norm(state.object_twist_world[0, :3]))
        object_angular = float(
            torch.linalg.vector_norm(state.object_twist_world[0, 3:])
        )
        robot_orientation_error = float(
            _quaternion_distance(
                control_model.body_pose_world[:, 3:],
                target.phase_goal_robot_root_pose_world[:, 3:],
            )[0]
        )
        object_orientation_error = float(
            _quaternion_distance(
                state.object_pose_world[:, 3:],
                target.phase_goal_object_pose_world[:, 3:],
            )[0]
        )
        near = contact_velocity_valid and bool(
            (
                (surface_distance[0] <= 0.008)
                & (surface_distance[0] >= -0.008)
                & (relative_speed[0] <= 0.05)
            ).all()
        )
        loaded = bool((motor_load[0] >= 0.05).all())
        self.grip_contact_present = bool(
            ((surface_distance[0] <= 0.008) & (surface_distance[0] >= -0.008)).all()
        ) and loaded
        released = bool((surface_distance[0] >= 0.005).all())
        qp = bool(qp_feasible[0])
        progress = float(target.phase_progress[0])
        endpoint = progress >= 1.0 - 1e-6 or phase == 7
        if phase == 0:
            ready = robot_error <= 0.08 and robot_speed <= 0.02 and qp
        elif phase == 1:
            ready = near and loaded and qp
        elif phase in (2, 3):
            ready = object_error <= 0.052 and near and loaded and qp
        elif phase == 4:
            ready = object_error <= self.final_goal_position_tolerance_m and object_speed <= 0.05 and near and qp
        elif phase == 5:
            ready = released and object_error <= self.final_goal_position_tolerance_m and object_speed <= 0.05 and qp
        elif phase == 6:
            ready = self.release_latched and robot_error <= 0.05 and qp
        else:
            ready = (
                self.release_latched
                and object_error <= self.final_goal_position_tolerance_m
                and object_speed <= 0.05
                and object_angular <= 0.1
                and qp
            )
        ready = ready and (
            robot_orientation_error <= 0.2
            if phase in (0, 6)
            else object_orientation_error <= (
                self.final_goal_orientation_tolerance_rad if phase in (4, 5, 7) else 0.2
            )
        )
        self.dwell = self.dwell + self.dt if ready else 0.0
        minimum_dwell = 1.0 if phase == 7 else 0.25 if phase in (1, 5) else 0.0
        guarded = endpoint and ready and self.dwell >= minimum_dwell
        if phase == 5 and guarded:
            self.release_latched = True
        if (
            endpoint
            and self.pending is None
            and self.time_s + 1e-9 >= self.next_decision_s
            and phase < 7
        ):
            context = self.context(self.observation(state, qp_feasible=qp), phase)
            request = self.choose(context)
            entry = context.catalog.resolve(request)
            if request.transition_id is not None:
                if entry.phase_id != self.phase_ids[PHASES[phase + 1]]:
                    raise ValueError("unexpected IRG successor")
                self.pending = request
            self.next_decision_s = self.time_s + 0.5
        transition = guarded and (phase == 7 or self.pending is not None)
        # Dense privileged rewards are valid training targets, never actor input.
        # Use task progress/outcome, excluding exact wrench-box shaping.
        if self.events:
            self.add_reward(-0.001 * self.dt)
            if transition:
                self.add_reward(1.0)
        if transition:
            self.evaluator_phase_exits.append(
                dict(
                    phase=PHASES[phase],
                    privileged_phase_success=bool(privileged_reward.phase_success[0]),
                    time_s=self.time_s,
                )
            )
        if transition or int(self.time_s / self.dt) % 250 == 0:
            row = dict(
                time_s=self.time_s,
                phase=PHASES[phase],
                progress=progress,
                robot_error_m=robot_error,
                object_error_m=object_error,
                surface_distance_m=surface_distance[0].cpu().tolist(),
                relative_speed_mps=relative_speed[0].cpu().tolist(),
                algebraic_relative_speed_mps=algebraic_relative_speed[0].cpu().tolist(),
                contact_velocity_valid=contact_velocity_valid,
                motor_load_nm=motor_load[0].cpu().tolist(),
                near=near,
                loaded=loaded,
                guard=guarded,
                requested_transition=self.pending is not None,
                advance=transition,
            )
            self.trace.append(row)
            print("REQUEST_PROGRESS=" + json.dumps(row), flush=True)
        return torch.tensor([transition], device=phase_index.device, dtype=torch.bool)

    def finish(self, success):
        if not self.events:
            raise RuntimeError("no real actor events")
        self.add_reward(5.0 if success else -5.0)
        self.events[-1]["end_time_s"] = self.time_s
        self.events[-1]["done"] = True
        return dict(
            maximum_lift_m=self.maximum_lift,
            maximum_transport_m=self.maximum_transport,
            initial_runtime_check=self.initial_runtime_check,
            event_count=len(self.events),
            phase_exits=self.evaluator_phase_exits,
        )


def request_batch_row(value, index):
    """Keep a leading singleton environment dimension, including nested rewards."""
    if isinstance(value, torch.Tensor):
        return value[index:index + 1] if value.ndim else value
    if is_dataclass(value):
        return replace(value, **{
            f.name: request_batch_row(getattr(value, f.name), index) for f in fields(value)
        })
    if isinstance(value, SimpleNamespace):
        return SimpleNamespace(**{k: request_batch_row(v, index) for k, v in vars(value).items()})
    if isinstance(value, dict):
        return {k: request_batch_row(v, index) for k, v in value.items()}
    return value


class BatchedRequestEventSupervisor:
    """Independent high-level state over a shared batched physics/controller loop.

    The current execution entry groups replicas of one checked scene/plan.
    This adapter does not turn a single plan into multiple distinct task plans.
    """

    def __init__(self, supervisors):
        self.supervisors = tuple(supervisors)
        if not self.supervisors or len({id(s) for s in self.supervisors}) != len(self.supervisors):
            raise ValueError("batch requires distinct environment supervisors")
        self.deadlines = tuple(ObjectPoseGoalDeadlineMonitor(s.task) for s in self.supervisors)
        self.origins = None

    @property
    def episodes(self):
        return [s.events for s in self.supervisors]

    @property
    def trace(self):
        return [dict(environment=i, **row) for i, s in enumerate(self.supervisors) for row in s.trace]

    @property
    def initial_runtime_check(self):
        return [s.initial_runtime_check for s in self.supervisors]

    def begin(self, controller, state, target=None):
        count = len(self.supervisors)
        if state.module_pose_world.shape[0] != count:
            raise ValueError("supervisor/environment count differs")
        if self.origins is None:
            origins = controller.policy_frame_origins_world
            if origins is None:
                origins = torch.zeros(count, 3)
            if origins.shape != (count, 3):
                raise ValueError("environment origins must be [environment,3]")
            self.origins = origins.detach().cpu().clone()
            for index, supervisor in enumerate(self.supervisors):
                supervisor.environment_origin = self.origins[index].clone()
        for index, supervisor in enumerate(self.supervisors):
            if not supervisor.started:
                supervisor.begin(controller, request_batch_row(state, index),
                                 request_batch_row(target, index))

    def step(self, *, active, **kwargs):
        if active.shape != (len(self.supervisors),):
            raise ValueError("active environment mask differs")
        transitions = torch.zeros_like(active)
        for index, live in enumerate(active.detach().cpu().tolist()):
            if live:
                transitions[index] = self.supervisors[index].step(**{
                    name: request_batch_row(value, index) for name, value in kwargs.items()
                })[0]
        return transitions

    def apply_deadlines(self, reward, *, time_s, object_pose_world, active):
        if self.origins is None:
            raise RuntimeError("batch supervisor has not started")
        times = time_s.detach().cpu().double()
        poses = object_pose_world.detach().cpu().double().clone()
        poses[:, :3] -= self.origins.double()
        # Inactive rows stay terminal and cannot advance clocks/phases. The
        # harness records only their first terminal and never resets them.
        changes = {name: getattr(reward, name).clone()
                   for name in ("timeout", "terminal_failure", "phase_success")}
        for index, live in enumerate(active.detach().cpu().tolist()):
            if live:
                row = self.deadlines[index].apply(
                    request_batch_row(reward, index), time_s=float(times[index]),
                    object_pose_world=poses[index],
                )
                for name in changes:
                    changes[name][index] = getattr(row, name)[0]
            else:
                changes["terminal_failure"][index] = True
                changes["phase_success"][index] = False
        return replace(reward, **changes)
