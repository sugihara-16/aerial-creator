from __future__ import annotations

"""Provisional force -> servo lead -> posture -> teacher-seeded joint plan.

This initial planner supports the existing quasi-static opposed grasp/carry
teachers. Contact forces are model estimates, never measured feedback or an
exact wrench-tracking contract. The archived plan includes preload BEFORE
validation/execution. It does not perform runtime IK or evaluate a learned actor.
"""

from dataclasses import dataclass, replace
import math
import time
from typing import Mapping

import numpy as np
import torch

from amsrr.feasibility.articulated_reachability import (
    base_pose_for_centroidal_target,
    resolve_mesh_backed_anchor_references,
)
from amsrr.robot_model.whole_structure_kinematics import ordered_global_dock_joint_ids
from amsrr.schemas.policies import ContactWrenchTrajectory, PostureTarget
from amsrr.training.order9_actuator_aware_nominal_preload import (
    Order9ActuatorAwareNominalPreloadConfig,
    solve_order9_actuator_aware_nominal_preload,
)
from amsrr.training.order9_c3_nominal_runtime import Order9C3NominalTensorReference
from amsrr.training.order9_c3_teacher import build_order9_c3_posture_collision_object
from amsrr.training.order9_posture_resolver import (
    Order9PostureResolverConfig,
    Order9PostureTrajectoryResolver,
)
from amsrr.training.order9_virtual_contact_compression import (
    Order9VirtualContactCompressionSolution,
    _compression_collision_result_accepted,
    _global_joint_limits,
)
from amsrr.utils.hashing import stable_hash

NAIVE_CONTACT_PLAN_VERSION = "naive_quasistatic_contact_plan_v10_per_anchor_preload"
NAIVE_CONTACT_INTERPOLATION = "cubic_centroidal_joint_exact_twist_v4"
# One nominal clearance for preload optimization and independent path checks.
# The checker's existing numerical tolerance remains owned by the checker.
NOMINAL_CONTACT_COLLISION_MARGIN_M = 0.001


def preload_weight(phase: str, progress: float) -> tuple[float, float]:
    """Compression and its derivative with respect to normalized phase time."""
    p = min(1.0, max(0.0, float(progress)))
    smooth, slope = p * p * (3.0 - 2.0 * p), 6.0 * p * (1.0 - p)
    if phase == "contact_acquisition":
        return smooth, slope
    if phase in {"lift", "transport", "place"}:
        return 1.0, 0.0
    if phase == "release":
        return 1.0 - smooth, -slope
    return 0.0, 0.0


def bounded_joint_curve(times, requested_q, lower, upper):
    """Choose bounded joint knots and monotonicity-preserving cubic slopes.

    Joint constraints belong to planning. Any difference from the initial
    additive guess is recorded, before FK, collision checks and execution.
    The terminal contact IK is separately required to achieve the chosen lead.
    """
    from scipy.interpolate import PchipInterpolator

    times = np.asarray(times, dtype=float)
    requested = np.asarray(requested_q, dtype=float)
    if (
        len(times) < 2
        or np.any(np.diff(times) <= 0)
        or not np.isfinite(requested).all()
    ):
        raise ValueError("invalid joint curve seed")
    q = np.clip(requested, np.asarray(lower), np.asarray(upper))
    qdot = PchipInterpolator(times, q, axis=0).derivative()(times)
    # Each phase may wait for its observed completion guard at the endpoint.
    qdot[0], qdot[-1] = 0.0, 0.0
    return q, qdot, float(np.max(np.abs(q - requested)))


def rest_to_rest_knot_times(times):
    """Spread endpoint acceleration over the phase, preserving geometric knots.

    Invert quintic progress 10u^3-15u^4+6u^5. Its peak slope is 15/8;
    multiplying the duration by that factor never shortens a source interval.
    The existing checked Hermite interpolation is rebuilt at these new times.
    This avoids accelerating to cruising speed within only the first dense IK
    interval. No endpoint, joint position, or collision criterion changes.
    """
    times = np.asarray(times, dtype=float)
    if len(times) < 2 or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError("invalid rest-to-rest knot times")
    duration = times[-1] - times[0]
    progress = (times - times[0]) / duration
    lower, upper = np.zeros_like(progress), np.ones_like(progress)
    for _ in range(50):
        midpoint = (lower + upper) / 2
        value = midpoint**3 * (10 + midpoint * (-15 + 6 * midpoint))
        lower = np.where(value < progress, midpoint, lower)
        upper = np.where(value >= progress, midpoint, upper)
    parameter = (lower + upper) / 2
    parameter[0], parameter[-1] = 0., 1.
    return times[0] + (15. / 8.) * duration * parameter


def retime_bounded_joint_curve(times, requested_q, lower, upper, speed_limit):
    """Enforce the authored planning rate on the executed cubic, locally.

    The rolling IK endpoint rate does not bound intermediate dense IK knots.
    Stretch only violating intervals, reconstruct monotone slopes, and check
    the exact quadratic velocity extrema. No geometric endpoint is changed.
    """
    times = np.asarray(times, dtype=float)
    intervals = np.diff(times).copy()
    limit = np.asarray(speed_limit, dtype=float)
    if not np.isfinite(limit).all() or (limit <= 0).any():
        raise ValueError("invalid executed joint speed limit")
    original_duration = float(times[-1] - times[0])
    original_intervals = intervals.copy()
    first_peak = None
    for iteration in range(24):
        current = np.r_[times[0], times[0] + np.cumsum(intervals)]
        q, v, correction = bounded_joint_curve(current, requested_q, lower, upper)
        h = intervals[:, None]
        a = 6 * (q[:-1] - q[1:]) / h + 3 * (v[:-1] + v[1:])
        b = -6 * (q[:-1] - q[1:]) / h - 4 * v[:-1] - 2 * v[1:]
        c = v[:-1]
        vertex = np.clip(np.divide(-b, 2 * a, out=np.zeros_like(a), where=np.abs(a) > 1e-14), 0., 1.)
        peak = np.maximum.reduce([np.abs(c), np.abs(a + b + c), np.abs(a * vertex**2 + b * vertex + c)])
        if first_peak is None:
            first_peak = float(peak.max())
        ratio = (peak / limit).max(axis=1)
        if ratio.max() <= 1. + 1e-8:
            return current, q, v, correction, dict(
                original_duration_s=original_duration, duration_s=float(current[-1] - current[0]),
                maximum_speed_before_radps=first_peak, maximum_speed_after_radps=float(peak.max()),
                speed_limit_radps=limit.tolist(), iterations=iteration,
                extended_interval_count=int((intervals > original_intervals + 1e-10).sum()),
            )
        intervals *= np.where(ratio > 1., ratio * 1.01, 1.)
    raise ValueError("executed joint-rate retiming did not converge")


def solve_nominal_contact_compression(*, morphology, physical_model, contact_knot,
                                     candidate_set, minimum_lead_m, maximum_lead_m,
                                     carried_knots=(), collision_object=None,
                                     collision_solver=None, lead_by_anchor=None):
    """Realize the same small normal lead using joints AND bounded body motion.

    Fixing the assembled CoM overconstrains some morphologies: reaching one
    contact can then grossly overcompress another. Translation is regularized
    together with joints. If the carried grasp intersects its environment at a
    later subgoal, a bounded body rotation is also allowed. No force observation
    is used. The resulting full path is checked separately before execution.
    """
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    if not 0 < minimum_lead_m <= maximum_lead_m:
        raise ValueError("invalid nominal compression interval")
    ids = ordered_global_dock_joint_ids(morphology, physical_model)
    limits = _global_joint_limits(ids, physical_model)
    assignments = sorted(contact_knot.contact_assignments, key=lambda a: a.anchor_id)
    anchors = [a.anchor_id for a in assignments]
    refs = resolve_mesh_backed_anchor_references(morphology, physical_model, anchors)
    kin = Order9PostureTrajectoryResolver(physical_model, prefer_native_solver=True,
        require_native_solver=True).ik_solver.kinematics
    fk = kin.bind_centroidal_anchor_fk(morphology, refs, ids)
    q0 = np.array([contact_knot.posture_target.joint_pos_target[j] for j in ids])
    rb = Rotation.from_quat(contact_knot.centroidal_target.body_orientation_world).as_matrix()
    origin = np.asarray(contact_knot.centroidal_target.com_pos_world)
    scales = np.r_[np.full(3, .030 / np.sqrt(3)), np.full(len(ids), .15)]
    lo = np.r_[-np.ones(3), np.maximum(-.15, [limits[j][0] - q0[i] for i, j in enumerate(ids)]) / .15]
    hi = np.r_[np.ones(3), np.minimum(.15, [limits[j][1] - q0[i] for i, j in enumerate(ids)]) / .15]

    def poses(x):
        d = x * scales
        local_p, local_r = fk((q0 + d[3:])[None, :])
        return (local_p[0] + d[:3]) @ rb.T + origin, rb @ local_r[0]

    p0, r0 = poses(np.zeros(len(scales)))
    candidates = {c.candidate_id: c for c in candidate_set.candidates}
    normals = np.array([candidates[a.candidate_id].normal_world for a in assignments])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    if not np.isfinite(lengths).all() or np.any(lengths <= 1e-9):
        raise ValueError("invalid nominal compression normals")
    normals /= lengths
    required = np.array([minimum_lead_m if lead_by_anchor is None else lead_by_anchor[a]
                         for a in anchors], dtype=float)
    if not np.isfinite(required).all() or (required <= 0).any() or (required > maximum_lead_m).any():
        raise ValueError("invalid per-contact compression lead")
    leads = np.minimum(maximum_lead_m, required + .0002)
    lead = float(leads.max())
    target = p0 - leads[:, None] * normals

    def residual(x):
        p, r = poses(x)
        return np.r_[(p - target).ravel() / .0001,
            Rotation.from_matrix(r @ r0.transpose(0, 2, 1)).as_rotvec().ravel() / .1,
            x * .03]

    fitted = least_squares(residual, np.clip(np.zeros(len(scales)), lo + 1e-10, hi - 1e-10),
        bounds=(lo, hi), max_nfev=100, ftol=1e-9, gtol=1e-9, xtol=1e-9)
    p, _ = poses(fitted.x)
    d = fitted.x * scales
    body_rotation = np.zeros(3)
    clearance_evidence = dict(required=False)

    def carried_clearances(translation, rotation, joint_delta):
        values = []
        for knot in carried_knots:
            r = Rotation.from_quat(knot.centroidal_target.body_orientation_world)
            pose = (*(
                np.asarray(knot.centroidal_target.com_pos_world) + r.apply(translation)
            ), *(r * Rotation.from_rotvec(rotation)).as_quat())
            q = {j: knot.posture_target.joint_pos_target[j] + joint_delta[i]
                 for i, j in enumerate(ids)}
            for box in collision_object.environment_boxes:
                collision_solver.set_collision_scene(morphology=morphology,
                    object_pose_world=box.pose_world, object_size_m=box.size_m,
                    allowed_anchor_ids=[])
                check = collision_solver.check_configuration(morphology=morphology,
                    centroidal_pose_world=pose, joint_positions_rad=q, exact=False,
                    margin_m=NOMINAL_CONTACT_COLLISION_MARGIN_M,
                    ground_plane_z_m=collision_object.ground_plane_z_m)
                values.append(check["minimum_clearance_m"])
        return np.asarray(values, dtype=float)

    if carried_knots and collision_object is not None and collision_object.environment_boxes:
        before = carried_clearances(d[:3], body_rotation, d[3:])
        # Use the unchanged final validator's required clearance. Safe nominal
        # solutions retain their exact prior commands; only infeasible goals
        # need the larger, bounded pose solve.
        if np.min(before) < .0008:
            pose_scales = np.r_[scales[:3], np.full(3, .1 / np.sqrt(3)), scales[3:]]
            pose_lo = np.r_[lo[:3], -np.ones(3), lo[3:]]
            pose_hi = np.r_[hi[:3], np.ones(3), hi[3:]]

            def pose_values(x):
                delta = x * pose_scales
                local_p, local_r = fk((q0 + delta[6:])[None, :])
                rotation = rb @ Rotation.from_rotvec(delta[3:6]).as_matrix()
                position = local_p[0] @ rotation.T + delta[:3] @ rb.T + origin
                return position, rotation @ local_r[0], delta

            def pose_residual(x):
                position, rotation, delta = pose_values(x)
                clearances = carried_clearances(delta[:3], delta[3:6], delta[6:])
                return np.r_[(position - target).ravel() / .0001,
                    Rotation.from_matrix(rotation @ r0.transpose(0, 2, 1)).as_rotvec().ravel() / .1,
                    x * .03,
                    np.minimum(clearances - NOMINAL_CONTACT_COLLISION_MARGIN_M, 0.) / .0001]

            # A deterministic neutral seed avoids inheriting a boundary-saturated
            # solution of the smaller, translation-only problem.
            fitted = least_squares(pose_residual,
                np.clip(np.zeros(len(pose_scales)), pose_lo + 1e-10, pose_hi - 1e-10),
                bounds=(pose_lo, pose_hi), max_nfev=100,
                ftol=1e-9, gtol=1e-9, xtol=1e-9)
            p, rotations, delta = pose_values(fitted.x)
            after = carried_clearances(delta[:3], delta[3:6], delta[6:])
            normal_error = np.arccos(np.clip(np.einsum('ai,ai->a',
                rotations[:, :, 0], -normals), -1., 1.))
            if np.min(after) < .0008 or np.max(normal_error) > .10:
                raise ValueError("contact posture has excessive carried-goal clearance or normal error")
            d = np.r_[delta[:3], delta[6:]]
            body_rotation = delta[3:6]
            clearance_evidence = dict(required=True, before_m=before.tolist(),
                after_m=after.tolist(), maximum_normal_error_rad=float(np.max(normal_error)),
                maximum_body_rotation_rad=.1, maximum_body_translation_m=.030,
                maximum_joint_change_rad=.15)
    inward = -np.einsum('ai,ai->a', p - p0, normals)
    tangent = np.linalg.norm(p - p0 + inward[:, None] * normals, axis=1)
    if (not np.isfinite(d).all() or np.any(inward < required - 1e-9)
            or inward.max() > maximum_lead_m + 1e-9 or tangent.max() > .002):
        raise ValueError("joint/centroidal preload IK did not realize bounded normal lead")
    result = Order9VirtualContactCompressionSolution(
        inward_lead_m=lead, joint_delta_rad=dict(zip(ids, d[3:].tolist())),
        achieved_inward_displacement_m=dict(zip(anchors, inward.tolist())),
        tangential_error_m=dict(zip(anchors, tangent.tolist())),
        maximum_joint_delta_rad=float(np.max(np.abs(d[3:]))), iterations=fitted.nfev,
        saturated=bool(np.any(np.abs(fitted.x) > .999)),
        version="joint_centroidal_nominal_compression_v2_carried_clearance")
    return result, d[:3], body_rotation, clearance_evidence


def preload_body_curve(phase, times, quaternions, translation_body):
    """Body-frame preload offset and its world velocity, including rotation."""
    from scipy.spatial.transform import Rotation
    t = np.asarray(times, dtype=float)
    rotations = Rotation.from_quat(quaternions)
    delta = rotations.apply(np.broadcast_to(translation_body, (len(t), 3)).copy())
    segment_omega = (rotations[1:] * rotations[:-1].inv()).as_rotvec() / np.diff(t)[:, None]
    omega = np.vstack((segment_omega[0], .5 * (segment_omega[:-1] + segment_omega[1:]), segment_omega[-1]))
    weights = np.array([preload_weight(phase, value / t[-1]) for value in t])
    position = weights[:, :1] * delta
    velocity = weights[:, 1:] / t[-1] * delta + weights[:, :1] * np.cross(omega, delta)
    # Phase guards may hold either endpoint indefinitely, just as for qdot.
    velocity[0], velocity[-1] = 0., 0.
    return position, velocity


def normalized_lerp_world_angular_velocity(start, end, alpha, duration):
    """Exact derivative of the existing shortest-arc normalized quaternion lerp.

    Quaternions are xyzw; omega_world = 2 * vector(qdot * conjugate(q)).
    This is the executed interpolation's derivative, not an Euler-angle rate.
    """
    end = torch.where((start * end).sum(-1, keepdim=True) < 0., -end, end)
    difference = end - start
    raw = start + alpha[:, None] * difference
    norm = raw.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    q = raw / norm
    qdot = (difference - q * (q * difference).sum(-1, keepdim=True)) / (norm * duration[:, None])
    return 2. * (q[:, 3:] * qdot[:, :3] - qdot[:, 3:] * q[:, :3]
                 + torch.linalg.cross(q[:, :3], qdot[:, :3], dim=-1))


class NaiveContactPlanReference(Order9C3NominalTensorReference):
    """The same continuous joint evaluator is used in checks and in Isaac."""

    runtime_version = NAIVE_CONTACT_INTERPOLATION

    def __init__(self, *, initial_observation=None, physical_model=None,
                 feedback_reference_clock=None, **kwargs):
        super().__init__(**kwargs)
        self.provenance["control_sampling_semantics"] = NAIVE_CONTACT_INTERPOLATION
        self.initial_observation = initial_observation
        self.physical_model = physical_model
        self.feedback_reference_clock = feedback_reference_clock

    def condition(self, target, *, phase_index, phase_elapsed_s, scene_origins):
        if self.feedback_reference_clock is not None:
            phase_elapsed_s = self.feedback_reference_clock(phase_index, phase_elapsed_s)
        batch = phase_index.shape[0]
        if phase_index.shape != (batch,) or phase_elapsed_s.shape != (batch,) or scene_origins.shape != (batch, 3):
            raise ValueError('C3 nominal replay batch shapes are invalid')
        if not hasattr(self, '_packed_references'):
            from amsrr.simulation.order9_object_task_runtime import ORDER9_OBJECT_TASK_PHASES
            from amsrr.utils.tensor_dataclass_graph import TensorDataclassGraph
            references = [self._references[phase.value] for phase in ORDER9_OBJECT_TASK_PHASES]
            counts = [len(ref.times_s) for ref in references]
            width = max(counts)
            names = ('times_s', 'body_pose_world', 'body_twist_world',
                     'joint_positions_rad', 'joint_velocities_radps', 'object_pose_world')
            self._packed_references = {}
            for name in names:
                values = []
                for ref, count in zip(references, counts):
                    value = getattr(ref, name)
                    padding = value[-1:].expand((width-count, *value.shape[1:])).clone()
                    if name == 'times_s':
                        padding.fill_(float('inf'))
                    values.append(torch.cat((value, padding)))
                self._packed_references[name] = torch.stack(values)
            self._reference_counts = torch.tensor(counts, device=self.device)
            self._reference_graph = TensorDataclassGraph()
        values = self._reference_graph.call(self._condition_packed,
            args=(target, phase_index, phase_elapsed_s, scene_origins),
            configuration='checked_hermite_reference_v1')
        return values

    def _condition_packed(self, target, phase_index, phase_elapsed_s, scene_origins):
        """Same knot curves and endpoint holds, without phase-wise GPU synchronization."""
        from amsrr.training.order9_c3_nominal_runtime import _normalized_lerp_quaternion
        tables = self._packed_references
        phase = phase_index.clamp(0, len(self._reference_counts)-1).long()
        valid = (phase_index >= 0) & (phase_index < len(self._reference_counts))
        count = self._reference_counts[phase]
        duration = tables['times_s'][phase, count-1]
        elapsed = phase_elapsed_s.to(device=self.device, dtype=self.dtype)
        t = torch.minimum(elapsed.clamp_min(0.), duration)
        upper = torch.searchsorted(tables['times_s'][phase].contiguous(),
            t[:, None].contiguous(), right=True).squeeze(-1).clamp_min(1)
        upper = torch.minimum(upper, count-1)
        lower = upper-1
        dt = tables['times_s'][phase, upper] - tables['times_s'][phase, lower]
        alpha = ((t-tables['times_s'][phase, lower])/dt).clamp(0., 1.)
        def endpoints(name):
            return tables[name][phase, lower], tables[name][phase, upper]
        b0, b1 = endpoints('body_pose_world')
        v0, v1 = endpoints('body_twist_world')
        u, h = alpha[:, None], dt[:, None]
        body = b0 + u*(b1-b0)
        body[:, 3:] = _normalized_lerp_quaternion(b0[:, 3:], b1[:, 3:], alpha)
        body[:, :3] = ((2*u**3-3*u**2+1)*b0[:, :3] + (u**3-2*u**2+u)*h*v0[:, :3]
            + (-2*u**3+3*u**2)*b1[:, :3] + (u**3-u**2)*h*v1[:, :3])
        linear = ((6*u**2-6*u)*b0[:, :3]/h + (3*u**2-4*u+1)*v0[:, :3]
            + (-6*u**2+6*u)*b1[:, :3]/h + (3*u**2-2*u)*v1[:, :3])
        angular = normalized_lerp_world_angular_velocity(b0[:, 3:], b1[:, 3:], alpha, dt)
        moving = ((elapsed > 0.) & (elapsed < duration))[:, None]
        twist = torch.where(moving, torch.cat((linear, angular), -1), 0.)
        q0, q1 = endpoints('joint_positions_rad')
        qv0, qv1 = endpoints('joint_velocities_radps')
        u, h = u[:, :, None], h[:, :, None]
        q = ((2*u**3-3*u**2+1)*q0 + (u**3-2*u**2+u)*h*qv0
            + (-2*u**3+3*u**2)*q1 + (u**3-u**2)*h*qv1)
        qdot = ((6*u**2-6*u)*q0/h + (3*u**2-4*u+1)*qv0
            + (-6*u**2+6*u)*q1/h + (3*u**2-2*u)*qv1)
        qdot = torch.where((elapsed > duration)[:, None, None], 0., qdot)
        o0, o1 = endpoints('object_pose_world')
        obj = o0 + alpha[:, None]*(o1-o0)
        obj[:, 3:] = _normalized_lerp_quaternion(o0[:, 3:], o1[:, 3:], alpha)
        goal = tables['body_pose_world'][phase, count-1].clone()
        object_goal = tables['object_pose_world'][phase, count-1].clone()
        origin = scene_origins.to(device=self.device, dtype=self.dtype)
        for pose in (body, obj, goal, object_goal):
            pose[:, :3] += origin
        changes = dict(desired_robot_root_pose_world=body, desired_robot_root_twist_world=twist,
            nominal_joint_positions_rad=q, nominal_joint_velocities_radps=qdot,
            desired_object_pose_world=obj, phase_goal_robot_root_pose_world=goal,
            phase_goal_object_pose_world=object_goal, phase_progress=(elapsed/duration.clamp_min(1e-6)).clamp(0., 1.))
        return replace(target, **{name: torch.where(valid.reshape((-1,)+(1,)*(value.ndim-1)),
            value, getattr(target, name)) for name, value in changes.items()})

    def phase_start_reference(self, phase=None):
        from amsrr.simulation.order9_object_task_runtime import Order9ObjectTaskPhase
        from amsrr.training.order9_c3_nominal_runtime import Order9C3PhaseStartReference
        from amsrr.controllers.rigid_body_model import RigidBodyControlModelBuilder

        phase = Order9ObjectTaskPhase.APPROACH if phase is None else phase
        if self.initial_observation is None or phase != Order9ObjectTaskPhase.APPROACH:
            return super().phase_start_reference(phase)
        # Reset belongs to the state before choosing a plan. In particular,
        # the selected path's first target velocity must never become state.
        obs = self.initial_observation
        measured = RigidBodyControlModelBuilder().build(
            obs.morphology_graph, self.physical_model, obs
        )
        modules = {m.module_id: m for m in obs.module_states}
        tensor = lambda value: torch.tensor(value, device=self.device, dtype=self.dtype)
        result = Order9C3PhaseStartReference(
            body_pose_local=tensor(measured.body_pose_world),
            body_twist=tensor(measured.body_twist_world),
            joint_positions_rad=tensor([[modules[mid].joint_positions[jid]
                for jid in self.joint_ids] for mid in self.module_ids]),
            joint_velocities_radps=tensor([[modules[mid].joint_velocities[jid]
                for jid in self.joint_ids] for mid in self.module_ids]),
            object_pose_local=tensor(obs.object_states[0].pose_world),
            object_twist=tensor(obs.object_states[0].twist_world),
        )
        result.validate()
        return result

    def _sample(self, reference, elapsed_s):
        values = list(super()._sample(reference, elapsed_s))
        t = elapsed_s.to(device=self.device, dtype=self.dtype).clamp(
            0.0, float(reference.times_s[-1])
        )
        upper = torch.searchsorted(reference.times_s, t, right=True).clamp(
            1, reference.times_s.numel() - 1
        )
        lower = upper - 1
        dt = reference.times_s[upper] - reference.times_s[lower]
        angular = normalized_lerp_world_angular_velocity(
            reference.body_pose_world[lower, 3:7], reference.body_pose_world[upper, 3:7],
            (t - reference.times_s[lower]) / dt, dt)
        # Both phase endpoints may be held while waiting for the event guard.
        moving = ((elapsed_s > 0.) & (elapsed_s < reference.times_s[-1])).reshape(-1, 1)
        # Continuous linear velocity avoids exciting a loaded grasp at every
        # knot. Use the SAME Hermite position curve in safety checks and control.
        body_u = ((t - reference.times_s[lower]) / dt)[:, None]
        body_h = dt[:, None]
        p0, p1 = reference.body_pose_world[lower, :3], reference.body_pose_world[upper, :3]
        b0, b1 = reference.body_twist_world[lower, :3], reference.body_twist_world[upper, :3]
        values[0][:, :3] = ((2*body_u**3 - 3*body_u**2 + 1)*p0
            + (body_u**3 - 2*body_u**2 + body_u)*body_h*b0
            + (-2*body_u**3 + 3*body_u**2)*p1
            + (body_u**3 - body_u**2)*body_h*b1)
        linear = ((6*body_u**2 - 6*body_u)*p0/body_h
            + (3*body_u**2 - 4*body_u + 1)*b0
            + (-6*body_u**2 + 6*body_u)*p1/body_h
            + (3*body_u**2 - 2*body_u)*b1)
        values[1][:, :3] = torch.where(moving, linear, torch.zeros_like(linear))
        values[1][:, 3:] = torch.where(moving, angular, torch.zeros_like(angular))
        u = ((t - reference.times_s[lower]) / dt).reshape(-1, 1, 1)
        h = dt.reshape(-1, 1, 1)
        q0, q1 = (
            reference.joint_positions_rad[lower],
            reference.joint_positions_rad[upper],
        )
        v0, v1 = (
            reference.joint_velocities_radps[lower],
            reference.joint_velocities_radps[upper],
        )
        values[2] = (
            (2 * u**3 - 3 * u**2 + 1) * q0
            + (u**3 - 2 * u**2 + u) * h * v0
            + (-2 * u**3 + 3 * u**2) * q1
            + (u**3 - u**2) * h * v1
        )
        values[3] = (
            (6 * u**2 - 6 * u) * q0 / h
            + (3 * u**2 - 4 * u + 1) * v0
            + (-6 * u**2 + 6 * u) * q1 / h
            + (3 * u**2 - 2 * u) * v1
        )
        outside = (elapsed_s.to(device=self.device) > reference.times_s[-1]).reshape(
            -1, 1, 1
        )
        values[3] = torch.where(outside, torch.zeros_like(values[3]), values[3])
        values[1] = torch.where(outside.reshape(-1, 1), torch.zeros_like(values[1]), values[1])
        return tuple(values)


@dataclass(frozen=True)
class NaiveContactPlan:
    phase_trajectories: Mapping[str, ContactWrenchTrajectory]
    evidence: dict

    def to_dict(self) -> dict:
        return {
            "version": NAIVE_CONTACT_PLAN_VERSION,
            "interpolation": NAIVE_CONTACT_INTERPOLATION,
            "phase_trajectories": {
                k: v.to_dict() for k, v in self.phase_trajectories.items()
            },
            "evidence": self.evidence,
        }

    @property
    def plan_id(self) -> str:
        return stable_hash(self.to_dict())


def plan_teacher_contact_trajectory(
    *,
    bundle,
    physical_model,
    task,
    object_mass_kg: float,
    contact_friction: float,
    contact_stiffness_n_per_m: float,
    preload_config: Order9ActuatorAwareNominalPreloadConfig,
    validation_dt_s: float = 0.02,
    deadline_s: float = 180.0,
) -> NaiveContactPlan:
    """Compile and check a fresh executable plan from a reviewed teacher seed.

    The teacher supplies contact binding, nominal posture/motion and subgoals.
    This is deliberately not arbitrary online request planning. The existing
    geometric seed is never modified, and commanded preload is archived.
    """
    if (
        not 0.0 < validation_dt_s <= 0.02
        or not math.isfinite(deadline_s)
        or deadline_s <= 0
    ):
        raise ValueError("invalid contact-plan validation budget")
    started = time.monotonic()
    source_hash = stable_hash(
        {k: v.to_dict() for k, v in bundle.phase_trajectories.items()}
    )
    contact = bundle.phase_trajectories["contact_acquisition"].knots[-1]
    from scipy.spatial.transform import Rotation
    by_candidate = {c.candidate_id:c for c in bundle.contact_candidate_set.candidates}
    targets = {by_candidate[a.candidate_id].target_entity_id for a in contact.contact_assignments}
    if len(targets) != 1:
        raise ValueError("nominal grasp load requires one target object")
    obj = next(o for o in task.scene.objects if o.object_id in targets)
    object_com = np.asarray(obj.pose_world[:3]) + Rotation.from_quat(obj.pose_world[3:]).apply(
        task.metadata.get("estimated_com_object", (0., 0., 0.)))
    preload = solve_order9_actuator_aware_nominal_preload(
        morphology=bundle.morphology,
        physical_model=physical_model,
        contact_knot=contact,
        candidate_set=bundle.contact_candidate_set,
        object_mass_kg=object_mass_kg,
        contact_friction=contact_friction,
        contact_stiffness_n_per_m=contact_stiffness_n_per_m,
        config=preload_config,
        object_com_world=object_com,
    )
    if not preload.feasible:
        raise ValueError(f"contact preload infeasible: {preload.rejection_reason}")
    collision_object = build_order9_c3_posture_collision_object(task)
    resolver = Order9PostureTrajectoryResolver(
        physical_model,
        config=Order9PostureResolverConfig(collision_margin_m=NOMINAL_CONTACT_COLLISION_MARGIN_M),
        collision_object=collision_object,
        prefer_native_solver=True,
        require_native_solver=True,
    )
    solver = resolver.ik_solver
    lead_by_anchor = dict(zip((a.anchor_id for a in contact.contact_assignments),
        preload.commanded_inward_lead_m_by_anchor, strict=True))
    compression, body_translation, body_rotation, carried_clearance = solve_nominal_contact_compression(
        morphology=bundle.morphology,
        physical_model=physical_model,
        contact_knot=contact,
        candidate_set=bundle.contact_candidate_set,
        minimum_lead_m=preload.inward_lead_m,
        maximum_lead_m=preload_config.maximum_inward_lead_m,
        carried_knots=tuple(bundle.phase_trajectories[phase].knots[-1]
            for phase in ("contact_acquisition", "lift", "transport", "place")
            if phase in bundle.phase_trajectories),
        collision_object=collision_object, collision_solver=solver,
        lead_by_anchor=lead_by_anchor,
    )
    # The legacy flag also records an intermediate continuation clip, even
    # when final least-squares refinement subsequently succeeds. Check the
    # final displacement and physical joint bounds instead of that history.
    if (
        any(value + 1e-9 < (preload.inward_lead_m if lead_by_anchor is None else lead_by_anchor[anchor])
            for anchor, value in compression.achieved_inward_displacement_m.items())
    ):
        raise ValueError("contact posture did not achieve the requested nominal lead")
    if max(compression.tangential_error_m.values()) > max(
        0.002, 0.25 * preload.inward_lead_m
    ):
        raise ValueError("contact posture has excessive tangential error")
    joint_ids = ordered_global_dock_joint_ids(bundle.morphology, physical_model)
    joint_limits = _global_joint_limits(joint_ids, physical_model)
    anchor_ids = tuple(sorted(a.anchor_id for a in contact.contact_assignments))
    references = resolve_mesh_backed_anchor_references(
        bundle.morphology, physical_model, anchor_ids
    )
    phases = {}
    approach_retiming = None
    release_retiming = None
    maximum_joint_bound_correction = 0.0
    for phase, source in bundle.phase_trajectories.items():
        trajectory = ContactWrenchTrajectory.from_dict(source.to_dict())
        if preload.target_contact_force_world_n:
            from amsrr.training.order9_teacher import upgrade_teacher_trajectory_to_v2
            from types import SimpleNamespace
            forces = dict(zip((a.anchor_id for a in contact.contact_assignments),
                preload.target_contact_force_world_n, strict=True))
            for k in trajectory.knots:
                for assignment in k.contact_assignments:
                    assignment.wrench_frame = 'world'
                    assignment.wrench_lower = assignment.wrench_upper = None
                    assignment.wrench_target = [*forces[assignment.anchor_id], 0.,0.,0.]
            trajectory.contract_version = 'implicit_world_v1'
            trajectory = upgrade_teacher_trajectory_to_v2(trajectory, SimpleNamespace(
                contact_candidate_set=bundle.contact_candidate_set,
                morphology_graph=bundle.morphology))
        duration = float(trajectory.knots[-1].t_rel_s)
        times = [float(k.t_rel_s) for k in trajectory.knots]
        body_offsets, _ = preload_body_curve(phase, times,
            [k.centroidal_target.body_orientation_world for k in trajectory.knots], body_translation)
        requested = [
            [
                float(k.posture_target.joint_pos_target[j])
                + preload_weight(phase, k.t_rel_s / duration)[0]
                * compression.joint_delta_rad.get(j, 0.0)
                for j in joint_ids
            ]
            for k in trajectory.knots
        ]
        if phase == "release":
            times = rest_to_rest_knot_times(times)
            release_retiming = dict(method="inverse_quintic_progress_peak_rate_preserved_v1",
                original_duration_s=duration, duration_s=float(times[-1]))
            trajectory = replace(trajectory, horizon_s=float(times[-1]), dt_s=float(np.min(np.diff(times))))
        positions, velocities, correction = bounded_joint_curve(
            times,
            requested,
            [joint_limits[j][0] for j in joint_ids],
            [joint_limits[j][1] for j in joint_ids],
        )
        if phase == "approach":
            from amsrr.training.order9_articulated_teacher import Order9ArticulatedTeacherConfig
            rate = Order9ArticulatedTeacherConfig().approach_joint_speed_limit_rad_s
            times, positions, velocities, correction, approach_retiming = retime_bounded_joint_curve(
                times, requested, [joint_limits[j][0] for j in joint_ids],
                [joint_limits[j][1] for j in joint_ids], rate)
            trajectory = replace(trajectory, horizon_s=float(times[-1]), dt_s=float(np.min(np.diff(times))))
        centroidal_positions = np.array([k.centroidal_target.com_pos_world for k in trajectory.knots]) + body_offsets
        _, centroidal_velocities, _ = bounded_joint_curve(
            times, centroidal_positions, [-math.inf]*3, [math.inf]*3)
        maximum_joint_bound_correction = max(maximum_joint_bound_correction, correction)
        knots = []
        for index, knot in enumerate(trajectory.knots):
            if time.monotonic() - started > deadline_s:
                raise TimeoutError("contact planning deadline")
            q = dict(zip(joint_ids, positions[index].tolist()))
            qdot = dict(zip(joint_ids, velocities[index].tolist()))
            c = replace(knot.centroidal_target,
                com_pos_world=tuple(centroidal_positions[index]),
                com_vel_world=tuple(centroidal_velocities[index]))
            if np.any(body_rotation):
                from scipy.spatial.transform import Rotation
                weight = preload_weight(phase, knot.t_rel_s / duration)[0]
                rotation = Rotation.from_quat(c.body_orientation_world) * Rotation.from_rotvec(weight * body_rotation)
                c = replace(c, body_orientation_world=tuple(rotation.as_quat()))
            base = base_pose_for_centroidal_target(
                bundle.morphology,
                physical_model,
                q,
                c.com_pos_world,
                c.body_orientation_world,
                kinematics=solver.kinematics,
                rigid_body_builder=solver._rigid_body_builder,
            )
            fk = solver.kinematics.forward(
                bundle.morphology, physical_model, q, base, references
            )
            knots.append(
                replace(
                    knot,
                    t_rel_s=float(times[index]),
                    centroidal_target=c,
                    posture_target=PostureTarget(
                        joint_pos_target=q,
                        joint_vel_target=qdot,
                        free_anchor_pose_targets=dict(fk.anchor_poses_world),
                    ),
                )
            )
        phases[phase] = replace(trajectory, knots=knots)
    module_ids = tuple(sorted(m.module_id for m in bundle.morphology.modules))
    local_ids = tuple(
        j.split(":", 1)[1]
        for j in joint_ids
        if j.startswith(f"module_{module_ids[0]}:")
    )
    reference = NaiveContactPlanReference(
        phase_trajectories=phases,
        module_ids=module_ids,
        joint_ids=local_ids,
        device="cpu",
        dtype=torch.float64,
    )
    checks = validate_teacher_contact_plan(
        reference=reference,
        morphology=bundle.morphology,
        physical_model=physical_model,
        collision_object=collision_object,
        collision_solver=solver,
        anchor_ids=anchor_ids,
        validation_dt_s=validation_dt_s,
        deadline=started + deadline_s,
    )
    if source_hash != stable_hash(
        {k: v.to_dict() for k, v in bundle.phase_trajectories.items()}
    ):
        raise RuntimeError("contact planner changed its teacher seed")
    from amsrr.controllers.grasp_slip_compensation import build_slip_reserve
    slip_reserve = build_slip_reserve(
        morphology=bundle.morphology, physical_model=physical_model,
        contact_knot=contact, candidates=bundle.contact_candidate_set,
        nominal_contact_knot=phases["contact_acquisition"].knots[-1],
        compression=compression, preload=preload, preload_config=preload_config,
    )
    return NaiveContactPlan(
        phases,
        {
            "approach_retiming": approach_retiming,
            "release_retiming": release_retiming,
            "teacher_phase_hash": source_hash,
            "teacher_provenance": bundle.provenance,
            "physical_model_hash": physical_model.stable_hash(),
            "task_spec_hash": stable_hash(task.to_dict()),
            "force_model": ("quasistatic_multi_contact_force_moment_equilibrium"
                if preload.target_contact_force_world_n else "quasistatic_vertical_frictional_support"),
            "exact_wrench_tracking_required": False,
            "preload": preload.to_dict(),
            "slip_reserve": slip_reserve,
            "joint_delta_rad": dict(compression.joint_delta_rad),
            "body_translation_body_m": body_translation.tolist(),
            "body_rotation_body_rad": body_rotation.tolist(),
            "carried_goal_clearance": carried_clearance,
            "compression_solver": compression.version,
            "achieved_inward_displacement_m": dict(
                compression.achieved_inward_displacement_m
            ),
            "tangential_error_m": dict(compression.tangential_error_m),
            "ik_intermediate_saturation_recorded": compression.saturated,
            "maximum_planned_joint_bound_correction_rad": maximum_joint_bound_correction,
            "checks": checks,
            "planning_wall_time_s": time.monotonic() - started,
        },
    )


def validate_teacher_contact_plan(
    *,
    reference,
    morphology,
    physical_model,
    collision_object,
    collision_solver,
    anchor_ids,
    validation_dt_s: float,
    deadline: float,
) -> dict:
    """Check actual interpolated commands, never a separately reconstructed IK.

    Finite sampling is reported as such; it is not a continuous collision or
    full-dynamics certificate. Isaac independently checks realized execution.
    """
    ids = tuple(
        f"module_{m}:{j}" for m in reference.module_ids for j in reference.joint_ids
    )
    limits = _global_joint_limits(ids, physical_model)
    local_models = {j.joint_id: j for j in physical_model.joints}
    from amsrr.training.order9_articulated_teacher import Order9ArticulatedTeacherConfig
    approach_rate = Order9ArticulatedTeacherConfig().approach_joint_speed_limit_rad_s
    count, max_speed, min_margin = 0, 0.0, math.inf
    previous_end = None
    for phase, data in reference._references.items():
        duration = float(data.times_s[-1])
        times = torch.linspace(
            0.0,
            duration,
            math.ceil(duration / validation_dt_s) + 1,
            dtype=reference.dtype,
        )
        sampled = reference._sample(data, times)
        q, qdot = sampled[2].numpy().reshape(len(times), -1), sampled[
            3
        ].numpy().reshape(len(times), -1)
        if previous_end is not None and np.max(np.abs(q[0] - previous_end)) > 1e-5:
            raise ValueError(f"joint discontinuity entering {phase}")
        previous_end = q[-1]
        if not np.isfinite(q).all() or not np.isfinite(qdot).all():
            raise ValueError("nonfinite interpolated contact plan")
        for column, joint_id in enumerate(ids):
            lo, hi = limits[joint_id]
            if np.min(q[:, column]) < lo - 1e-7 or np.max(q[:, column]) > hi + 1e-7:
                raise ValueError(f"joint limit along {phase}: {joint_id}")
            model = local_models[joint_id.split(":", 1)[1]]
            speed_limit = float(model.velocity_limit)
            if phase == "approach":
                speed_limit = min(speed_limit, approach_rate)
            speed = float(np.max(np.abs(qdot[:, column])))
            if speed > speed_limit + 1e-7:
                raise ValueError(f"joint speed along {phase}: {joint_id}: {speed}")
            max_speed = max(max_speed, speed)
        for row, t in enumerate(times.tolist()):
            if time.monotonic() > deadline:
                raise TimeoutError("contact trajectory validation deadline")
            pose, obj = sampled[0][row].tolist(), sampled[4][row].tolist()
            allowed = (
                anchor_ids
                if phase
                in {"contact_acquisition", "lift", "transport", "place", "release"}
                else ()
            )
            scenes = [
                (
                    collision_object.object_id,
                    obj,
                    collision_object.size_m,
                    allowed,
                    True,
                )
            ]
            scenes += [
                (b.box_id, b.pose_world, b.size_m, (), False)
                for b in collision_object.environment_boxes
            ]
            for scene_id, scene_pose, size, anchors, selected in scenes:
                collision_solver.set_collision_scene(
                    morphology=morphology,
                    object_pose_world=scene_pose,
                    object_size_m=size,
                    allowed_anchor_ids=anchors,
                )
                result = collision_solver.check_configuration(
                    morphology=morphology,
                    centroidal_pose_world=tuple(pose),
                    joint_positions_rad=dict(zip(ids, q[row])),
                    exact=False,
                    margin_m=NOMINAL_CONTACT_COLLISION_MARGIN_M,
                    ground_plane_z_m=collision_object.ground_plane_z_m,
                )
                if not _compression_collision_result_accepted(
                    result,
                    selected_contact_scene=selected,
                    collision_margin_m=NOMINAL_CONTACT_COLLISION_MARGIN_M,
                ):
                    raise ValueError(
                        f"contact plan collision {phase} t={t:.3f} scene={scene_id}: {result}"
                    )
                margin = result.get("minimum_clearance_m")
                if margin is not None and math.isfinite(float(margin)):
                    min_margin = min(min_margin, float(margin))
            count += 1
    return {
        "accepted": True,
        "interpolation": NAIVE_CONTACT_INTERPOLATION,
        "sample_count": count,
        "maximum_sample_interval_s": validation_dt_s,
        "maximum_joint_speed_radps": max_speed,
        "minimum_reported_clearance_m": (
            min_margin if math.isfinite(min_margin) else None
        ),
        "scope": "sampled_joint_limits_speed_object_support_ground_self_collision",
        "continuous_collision_certificate": False,
    }
