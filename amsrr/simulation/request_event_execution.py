"""Causal pi_H execution on the provisional nominal controller.

The task-specific eight-phase adapter is declared by this grasp/carry IRG. Raw
PhysX contacts remain reward/evaluation evidence, never transition authority.
"""

from copy import deepcopy
from dataclasses import replace
import json
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


class RequestEventSupervisor:
    """One environment per process; every stochastic choice is replayable by PPO."""

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
        diagnostic_phase=None
    ):
        self.model = model.cpu().eval()
        self.bundle = bundle
        self.task = task
        self.physical = physical
        self.initial_scene, self.phase_ids = teacher_request_scene(
            bundle, task, plan_committed=False
        )
        self.scene, _ = teacher_request_scene(bundle, task, plan_committed=True)
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

    def observation(self, state, *, qp_feasible=True):
        if state.module_pose_world.shape[0] != 1:
            raise ValueError("request adapter supports exactly one environment")
        return RuntimeObservation(
            self.time_s,
            self.bundle.morphology,
            [
                ModuleRuntimeState(
                    mid,
                    tuple(state.module_pose_world[0, i].cpu().tolist()),
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
                    tuple(state.object_pose_world[0].cpu().tolist()),
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
            self.phase_ids[phase_name],
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
            task_spec=self.task,
            physical_model=self.physical,
            execution_state=state,
        )

    def choose(self, context):
        result = self.model.decide(
            context, deterministic=not self.sample, generator=self.generator
        )
        record = dict(
            encoding=serialize_encoding(result["encoded"]),
            action=result["action"],
            log_prob=result["log_prob"],
            value=result["value"],
            reward=0.0,
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
        privileged_reward
    ):
        """Actor and guards see only the explicit observation/proprioception fields.

        privileged_reward is accumulated after observations and action selection;
        its phase-success bit is recorded only for independent outcome auditing.
        """
        if not self.started:
            raise RuntimeError("actor has not made the initial request")
        phase = int(phase_index[0])
        self.time_s += self.dt
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
        near = bool(
            (
                (surface_distance[0] <= 0.008)
                & (surface_distance[0] >= -0.008)
                & (relative_speed[0] <= 0.05)
            ).all()
        )
        loaded = bool((motor_load[0] >= 0.05).all())
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
            ready = object_error <= 0.052 and object_speed <= 0.05 and near and qp
        elif phase == 5:
            ready = released and object_error <= 0.052 and object_speed <= 0.05 and qp
        elif phase == 6:
            ready = self.release_latched and robot_error <= 0.05 and qp
        else:
            ready = (
                self.release_latched
                and object_error <= 0.052
                and object_speed <= 0.05
                and object_angular <= 0.1
                and qp
            )
        ready = ready and (
            robot_orientation_error <= 0.2
            if phase in (0, 6)
            else object_orientation_error <= 0.2
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
            self.events[-1]["reward"] -= 0.001 * self.dt
            if transition:
                self.events[-1]["reward"] += 1.0
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
        self.events[-1]["reward"] += 5.0 if success else -5.0
        self.events[-1]["done"] = True
        return dict(
            maximum_lift_m=self.maximum_lift,
            maximum_transport_m=self.maximum_transport,
            initial_runtime_check=self.initial_runtime_check,
            event_count=len(self.events),
            phase_exits=self.evaluator_phase_exits,
        )
