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
from amsrr.schemas.high_level import ActiveExecutionState, ContactMotionEstimate
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


def observed_object_signed_distance(points_world, object_pose_world, geometry):
    """Distance to the finite observed primitive, positive only outside.

    Release may leave a selected face tangentially. Distance to that face's
    infinite plane is then not a distance to the object. Only measured pose/FK
    and authored geometry enter this calculation; no contact forces are used.
    Unsupported/nonuniform round primitives conservatively refuse release.
    """
    q = object_pose_world[:, 3:]
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1.e-12)
    offset = points_world - object_pose_world[:, None, :3]
    xyz = -q[:, None, :3].expand_as(offset)
    cross = torch.cross(xyz, offset, dim=-1)
    local = offset + 2 * (q[:, None, 3:] * cross + torch.cross(xyz, cross, dim=-1))
    kind = getattr(geometry.geometry_type, 'value', geometry.geometry_type)
    params, scale = geometry.primitive_params, geometry.scale
    if kind == 'box':
        size = params.get('size_m', [params.get(k, 0.) for k in ('x_m', 'y_m', 'z_m')])
        half = torch.as_tensor([s * a / 2 for s, a in zip(size, scale, strict=True)],
                               device=local.device, dtype=local.dtype)
        delta = local.abs() - half
        return delta.clamp_min(0.).norm(dim=-1) + delta.amax(dim=-1).clamp_max(0.)
    uniform_xy = math.isclose(scale[0], scale[1], rel_tol=0., abs_tol=1.e-9)
    if kind == 'sphere' and uniform_xy and math.isclose(scale[1], scale[2], rel_tol=0., abs_tol=1.e-9):
        radius = float(params.get('radius_m', params.get('radius'))) * scale[0]
        return local.norm(dim=-1) - radius
    if kind == 'cylinder' and uniform_xy:
        radius = float(params.get('radius_m', params.get('radius'))) * scale[0]
        half_height = float(params.get('height_m', params.get('height'))) * scale[2] / 2
        delta = torch.stack((local[..., :2].norm(dim=-1)-radius,
                             local[..., 2].abs()-half_height), dim=-1)
        return delta.clamp_min(0.).norm(dim=-1) + delta.amax(dim=-1).clamp_max(0.)
    return torch.full_like(local[..., 0], -float('inf'))


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
    contact_maintenance_contract = "acquired_contact_geometry_motor_continuity_v2"

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
        self.object_geometry = next(g for g in task.scene.geometry_library
            if g.geometry_id == task.scene.objects[0].geometry_id)
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
        self.contact_established = False
        self.module_ids = None
        self.joint_ids = None
        self.plan_id = bundle.provenance["compiled_plan_id"]
        self.initial_object = None
        self.maximum_lift = 0.0
        self.maximum_transport = 0.0
        self.maximum_estimated_joint_motor_load_nm = 0.0
        self.final_object = None
        self.initial_runtime_check = {}
        self.evaluator_phase_exits = []
        self.diagnostic_phase = diagnostic_phase
        self.contact_velocity_observer = ContactPointVelocityObserver()
        self.contact_motion = []

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
            contact_motion=[] if initial else self.contact_motion,
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
        if 'hold_duration_s' in result:
            duration = float(result['hold_duration_s'])
            if (not math.isfinite(duration) or duration < 0 or duration > 8.
                    or (result['request'].transition_id is not None and duration != 0)):
                raise ValueError('invalid sampled hold duration')
            record['hold_duration_s'] = duration
        self.chosen_hold_duration_s = float(result.get('hold_duration_s', .5))
        from amsrr.policies.request_ranking import record_ranking
        record_ranking(record, result)
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
        joint_motor_load=None,
        _metrics=None,
    ):
        """Actor and guards see only the explicit observation/proprioception fields.

        privileged_reward is accumulated after observations and action selection;
        its phase-success bit is recorded only for independent outcome auditing.
        """
        if not self.started:
            raise RuntimeError("actor has not made the initial request")
        phase = int(phase_index[0])
        self.time_s += self.dt
        if _metrics is None:
            if joint_motor_load is not None:
                self.maximum_estimated_joint_motor_load_nm = max(
                    self.maximum_estimated_joint_motor_load_nm,
                    float(joint_motor_load[0].abs().max()))
            algebraic_relative_speed = relative_speed
            contact_velocity_valid = True
            if grasp_position_world is not None:
                relative_speed, contact_velocity_valid = self.contact_velocity_observer.update(
                    grasp_position_world, state.object_pose_world, time_s=self.time_s,
                    binding=(self.task.scene.objects[0].object_id, self.expected_group),
                )
        else:
            if _metrics['maximum_joint_load'] is not None:
                self.maximum_estimated_joint_motor_load_nm = max(self.maximum_estimated_joint_motor_load_nm, _metrics['maximum_joint_load'])
            algebraic_relative_speed = relative_speed
            relative_speed = _metrics['relative_speed']
            contact_velocity_valid = _metrics['contact_velocity_valid']
        if phase != self.phase:
            if phase != self.phase + 1:
                raise ValueError("nonadjacent execution transition")
            self.phase = phase
            self.phase_started_s = self.time_s - float(phase_elapsed_s[0])
            self.pending = None
            self.dwell = 0.0
            self.next_decision_s = self.time_s
        if _metrics is None:
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
            release_distance = surface_distance
            if phase == 5 and grasp_position_world is not None:
                release_distance = observed_object_signed_distance(
                    grasp_position_world, state.object_pose_world, self.object_geometry)
            released = bool((release_distance[0] >= 0.005).all())
            qp = bool(qp_feasible[0])
        else:
            self.final_object = _metrics['object_position']
            self.maximum_lift = max(self.maximum_lift,_metrics['lift'])
            self.maximum_transport = max(self.maximum_transport,_metrics['travel'])
            robot_error, object_error, robot_speed, object_speed, object_angular = _metrics['errors']
            robot_orientation_error, object_orientation_error = _metrics['rotations']
            near, loaded, released, qp = (_metrics[k] for k in ('near','loaded','released','qp'))
            release_distance = _metrics['release_distance']
        # Motor torque is not an individual contact-force measurement. After
        # verified acquisition, quiet joints alone do not imply contact loss.
        # Retain that evidence only while every observed contact remains near
        # and nonslipping, and the group still has motor load. QP feasibility
        # gates action/compensation below; its temporary loss is not evidence
        # that a measured physical contact disappeared.
        group_loaded = bool((motor_load[0] >= 0.05).any()) if _metrics is None else _metrics['group_loaded']
        was_established = self.contact_established
        if not (near and group_loaded) or phase not in (1, 2, 3, 4):
            self.contact_established = False
        maintained = loaded or self.contact_established
        # Slip correction must remain available while contact is sliding.
        # Transition readiness still requires the nonslipping `near` condition.
        geometric_contact = (contact_velocity_valid and bool(
            ((surface_distance[0] <= 0.008) & (surface_distance[0] >= -0.008)).all())) if _metrics is None else _metrics['geometric_contact']
        self.grip_contact_present = geometric_contact and qp and maintained and phase in (1, 2, 3, 4)
        progress = float(target.phase_progress[0])
        endpoint = progress >= 1.0 - 1e-6 or phase == 7
        if phase == 0:
            ready = robot_error <= 0.08 and robot_speed <= 0.02 and qp
        elif phase == 1:
            ready = near and loaded and qp
        elif phase in (2, 3):
            ready = object_error <= 0.052 and near and maintained and qp
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
            # These are the same pose/FK and motor feedback values used by the
            # deployable guard. Do not convert motor torque into contact force.
            self.contact_motion = [
                ContactMotionEstimate(
                    cid, self.time_s, float(distance), float(speed),
                    abs(float(load)), bool(contact_velocity_valid),
                )
                for cid, distance, speed, load in zip(
                    self.group.candidate_ids, surface_distance[0].tolist(),
                    relative_speed[0].tolist(), motor_load[0].tolist(), strict=True,
                )
            ]
            context = self.context(self.observation(state() if callable(state) else state, qp_feasible=qp), phase)
            request = self.choose(context)
            entry = context.catalog.resolve(request)
            if request.transition_id is not None:
                if entry.phase_id != self.phase_ids[PHASES[phase + 1]]:
                    raise ValueError("unexpected IRG successor")
                self.pending = request
            self.next_decision_s = self.time_s + max(.5, self.chosen_hold_duration_s)
        transition = guarded and (phase == 7 or self.pending is not None)
        if transition and phase == 1:
            # Only an actual acquisition exit establishes continuity; neither
            # proximity alone nor an unexecuted transition request suffices.
            self.contact_established = True
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
        if transition or self.contact_established != was_established or int(self.time_s / self.dt) % 250 == 0:
            row = dict(
                time_s=self.time_s,
                phase=PHASES[phase],
                progress=progress,
                robot_error_m=robot_error,
                object_error_m=object_error,
                surface_distance_m=surface_distance[0].cpu().tolist(),
                release_surface_distance_m=release_distance[0].cpu().tolist(),
                relative_speed_mps=relative_speed[0].cpu().tolist(),
                algebraic_relative_speed_mps=algebraic_relative_speed[0].cpu().tolist(),
                contact_velocity_valid=contact_velocity_valid,
                motor_load_nm=motor_load[0].cpu().tolist(),
                near=near,
                loaded=loaded,
                qp_feasible=qp,
                contact_established=self.contact_established,
                maintained=maintained,
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
        self.events[-1]["task_success"] = bool(success)
        return dict(
            maximum_lift_m=self.maximum_lift,
            maximum_transport_m=self.maximum_transport,
            maximum_estimated_joint_motor_load_nm=self.maximum_estimated_joint_motor_load_nm,
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
        self._deadline_goals = None
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

    def _measurements(self, values, active):
        """Batch the same measured geometry; keep actor and guard state per row."""
        state, model, target = values['state'], values['control_model'], values['target']
        live = active.tolist()
        phases = values['phase_index'].tolist()
        ids = [i for i, enabled in enumerate(live) if enabled]
        relative_speed = values['relative_speed'].clone()
        valid = [True]*len(self.supervisors)
        points = values.get('grasp_position_world')
        if points is not None and ids:
            selected_points = points[ids]
            poses = state.object_pose_world[ids]
            if (points.ndim != 3 or points.shape[-1] != 3
                    or poses.shape != (len(ids), 7)
                    or not torch.isfinite(selected_points).all() or not torch.isfinite(poses).all()):
                raise ValueError('invalid observed contact point sample')
            quaternion = poses[:, 3:]
            norm = quaternion.norm(dim=-1,keepdim=True)
            if bool((norm < 1e-8).any()):
                raise ValueError('invalid observed object orientation')
            quaternion = quaternion/norm
            offset = selected_points-poses[:, None, :3]
            xyz = -quaternion[:, None, :3].expand_as(offset)
            cross = torch.cross(xyz, offset, dim=-1)
            current = offset + 2*(quaternion[:, None, 3:]*cross + torch.cross(xyz,cross,dim=-1))
            previous, intervals = [], []
            for row,index in enumerate(ids):
                supervisor = self.supervisors[index]
                observer = supervisor.contact_velocity_observer
                now = supervisor.time_s+supervisor.dt
                binding = (supervisor.task.scene.objects[0].object_id,supervisor.expected_group)
                valid[index] = observer.previous is not None and observer.binding == binding
                if not math.isfinite(now):
                    raise ValueError('invalid observed contact point sample')
                interval = now-observer.previous_time if valid[index] else 1.
                if valid[index] and (interval <= 0 or observer.previous.shape != current[row:row+1].shape):
                    raise ValueError('contact velocity requires monotonic samples and stable binding')
                previous.append(observer.previous if valid[index] else current[row:row+1])
                intervals.append(interval)
                # current owns this sample; rows are disjoint and no caller
                # mutates it. A view avoids one allocation per environment.
                observer.previous = current[row:row+1].detach()
                observer.previous_time, observer.binding = float(now), binding
            speeds = ((current-torch.cat(previous))/current.new_tensor(intervals)[:,None,None]).norm(dim=-1)
            relative_speed[ids] = speeds
        initial = torch.stack([s.initial_object for s in self.supervisors])
        obj = state.object_pose_world
        errors = torch.linalg.vector_norm(torch.stack((
            model.body_pose_world[:, :3]-target.phase_goal_robot_root_pose_world[:, :3],
            obj[:, :3]-target.phase_goal_object_pose_world[:, :3],
            model.body_twist_world[:, :3], state.object_twist_world[:, :3],
            state.object_twist_world[:, 3:]),dim=1),dim=-1).tolist()
        rotations = _quaternion_distance(torch.stack((model.body_pose_world[:,3:],obj[:,3:]),dim=1),
            torch.stack((target.phase_goal_robot_root_pose_world[:,3:],target.phase_goal_object_pose_world[:,3:]),dim=1)).tolist()
        lift = (obj[:,2]-initial[:,2]).tolist()
        travel = (obj[:,:2]-initial[:,:2]).norm(dim=-1).tolist()
        distance, loads = values['surface_distance'],values['motor_load']
        geometric = ((distance <= .008)&(distance >= -.008)).all(-1).tolist()
        slow = (relative_speed <= .05).all(-1).tolist()
        loaded = (loads >= .05).all(-1).tolist()
        group_loaded = (loads >= .05).any(-1).tolist()
        qp = values['qp_feasible'].tolist()
        motor = values.get('joint_motor_load')
        maxima = motor.abs().flatten(1).amax(-1).tolist() if motor is not None else [None]*len(ids)
        release_distance = distance.clone()
        if points is not None:
            release_groups = {}
            for index, supervisor in enumerate(self.supervisors):
                if live[index] and phases[index] == 5:
                    release_groups.setdefault(id(supervisor.object_geometry), []).append(index)
            for indices in release_groups.values():
                release_distance[indices] = observed_object_signed_distance(
                    points[indices], obj[indices], self.supervisors[indices[0]].object_geometry)
        released = (release_distance >= .005).all(-1).tolist()
        release_rows = release_distance.split(1, dim=0)
        speed_rows = relative_speed.split(1, dim=0)
        positions = obj[:, :3].unbind(0)
        metrics = []
        for index,s in enumerate(self.supervisors):
            metrics.append(dict(object_position=positions[index], lift=lift[index], travel=travel[index],
                errors=errors[index], rotations=rotations[index], relative_speed=speed_rows[index],
                contact_velocity_valid=valid[index], geometric_contact=valid[index] and geometric[index],
                near=valid[index] and geometric[index] and slow[index], loaded=loaded[index],
                group_loaded=group_loaded[index], release_distance=release_rows[index],
                released=released[index], qp=qp[index],
                maximum_joint_load=None if motor is None else maxima[index]))
        return metrics

    def step(self, *, active, **kwargs):
        if active.shape != (len(self.supervisors),):
            raise ValueError("active environment mask differs")
        from amsrr.utils.tensor_snapshot import cpu_snapshot
        # Only these fields are read by step; the full model contains large
        # dynamics tensors which are neither observations nor guard inputs.
        kwargs = dict(kwargs)
        for name, names in (
            ("control_model", ("body_pose_world", "body_twist_world")),
            ("target", ("phase_goal_robot_root_pose_world", "phase_goal_object_pose_world", "phase_progress")),
            ("privileged_reward", ("phase_success",)),
        ):
            if name in kwargs:
                kwargs[name] = SimpleNamespace(**{k: getattr(kwargs[name], k) for k in names})
        snapshot = cpu_snapshot(dict(active=active, inputs=kwargs))
        transitions = torch.zeros(active.shape, dtype=torch.bool)
        values = snapshot['inputs']
        use_metrics = all(isinstance(s, RequestEventSupervisor) for s in self.supervisors)
        metrics = self._measurements(values, snapshot['active']) if use_metrics else None
        if metrics is not None:
            elapsed = values['phase_elapsed_s'].tolist()
            progress = values['target'].phase_progress.tolist()
            phase_success = values['privileged_reward'].phase_success.tolist()
        for index, live in enumerate(snapshot['active'].tolist()):
            if live:
                if metrics is None:
                    row = {name: request_batch_row(value,index) for name,value in values.items()}
                else:
                    # All geometry/guard reductions already exist in metrics.
                    # Materialize full observations only for an actual choice;
                    # scalar clocks retain the same float(tensor) precision.
                    row = dict(phase_index=values['phase_index'][index:index+1], phase_elapsed_s=(elapsed[index],),
                        target=SimpleNamespace(phase_progress=(progress[index],)),
                        state=lambda index=index: request_batch_row(values['state'],index),
                        control_model=None, qp_feasible=None,
                        surface_distance=values['surface_distance'][index:index+1],
                        relative_speed=values['relative_speed'][index:index+1],
                        motor_load=values['motor_load'][index:index+1],
                        privileged_reward=SimpleNamespace(phase_success=(phase_success[index],)),
                        _metrics=metrics[index])
                transitions[index] = self.supervisors[index].step(**row)[0]
        return transitions.to(active.device)

    def apply_deadlines(self, reward, *, time_s, object_pose_world, active):
        if self.origins is None:
            raise RuntimeError("batch supervisor has not started")
        from amsrr.utils.tensor_snapshot import cpu_snapshot
        host = cpu_snapshot(dict(times=time_s, poses=object_pose_world, active=active))
        times = host['times'].double()
        poses = host['poses'].double()
        poses[:, :3] -= self.origins.double()
        live_mask = host['active'].bool()
        if self._deadline_goals is None:
            entries = []
            for index, monitor in enumerate(self.deadlines):
                task = monitor.task
                goals = [g for g in task.goals if g.goal_type == 'object_pose']
                if not goals or len(task.scene.objects) != 1:
                    raise ValueError('request deadline evaluation requires one observed object')
                for goal in goals:
                    if (goal.target_entity_id != task.scene.objects[0].object_id
                            or goal.target_pose_world is None or goal.tolerance_pos_m is None
                            or goal.tolerance_rot_rad is None):
                        raise ValueError('object goal missing target or pose tolerances')
                    entries.append((index, goal))
            self._deadline_goals = (entries,
                torch.tensor([i for i, _ in entries]),
                torch.tensor([g.target_pose_world for _, g in entries], dtype=torch.float64),
                torch.tensor([g.tolerance_pos_m for _, g in entries], dtype=torch.float64),
                torch.tensor([g.tolerance_rot_rad for _, g in entries], dtype=torch.float64))
        entries, owners, targets, pos_tolerance, rot_tolerance = self._deadline_goals
        observed = poses[owners]
        if (not torch.isfinite(times[live_mask]).all() or (times[live_mask] < 0).any()
                or not torch.isfinite(poses[live_mask]).all()
                or (poses[live_mask, 3:].norm(dim=-1) < 1e-8).any()):
            raise ValueError('invalid physical object trajectory for deadline evaluation')
        reached = ((observed[:, :3] - targets[:, :3]).norm(dim=-1) <= pos_tolerance) & (
            _quaternion_distance(observed[:, 3:], targets[:, 3:]) <= rot_tolerance)
        time_values = times.tolist()
        for index, live in enumerate(live_mask.tolist()):
            if live:
                monitor = self.deadlines[index]
                if monitor.last_time_s is not None and time_values[index] < monitor.last_time_s:
                    raise ValueError('deadline observation clock moved backwards')
                monitor.last_time_s = time_values[index]
        for hit, (index, goal) in zip(reached.tolist(), entries):
            if not bool(live_mask[index]):
                continue
            monitor = self.deadlines[index]
            now = time_values[index]
            if hit and now <= goal.time_limit_s:
                monitor.first_arrivals.setdefault(goal.goal_id, now)
            if now > goal.time_limit_s and goal.goal_id not in monitor.first_arrivals:
                monitor.deadline_missed = True
                if monitor.termination_time_s is None:
                    monitor.termination_time_s = now
        missed = (torch.tensor([m.deadline_missed for m in self.deadlines]) & live_mask).to(active.device)
        # Copy only the combined terminal mask, rather than three GPU scalar
        # assignments per environment. Inactive rows retain their timeout bit.
        return replace(reward, timeout=reward.timeout | missed,
            terminal_failure=reward.terminal_failure | missed | ~active,
            phase_success=reward.phase_success & ~missed & active)
