"""Bounded grasp closure from observed material-point slip, without force feedback.

The planner supplies one checked contact binding and one local joint direction.
Every modified command is checked against joint limits, speeds and the same
native whole-body collision predicates as the nominal plan. This is sampled
command safety, not a continuous collision or exact wrench certificate.
"""
from dataclasses import replace
import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation

GRASP_SLIP_CONTRACT = 'observed_material_slip_bounded_closure_v3'


def _bounded_fk_reserve_scale(positions_at_scale, normals, span_m, *, surface_points=None):
    """Bound the actual FK displacement, including nonlinear IK quantization.

    Only an explicitly checked endpoint is returned. Intermediate commands still
    go through the runtime safety checker; this is not a path certificate.
    """
    origin = np.asarray(positions_at_scale(0.), dtype=float)
    normals = np.asarray(normals, dtype=float)
    surface = None if surface_points is None else np.asarray(surface_points, dtype=float)

    def accepted(scale):
        points = np.asarray(positions_at_scale(scale), dtype=float)
        delta = points - origin
        inward = -(delta * normals).sum(axis=1)
        tangent = delta + inward[:, None] * normals
        if surface is not None:
            total = points - surface
            total_tangent = total - (total * normals).sum(axis=1, keepdims=True) * normals
            if not np.all(np.linalg.norm(total_tangent, axis=1) <= .002):
                return False
        return (np.isfinite(delta).all() and np.all(inward >= -1e-10)
                and np.all(inward <= span_m)
                and np.all(np.linalg.norm(tangent, axis=1) <= .002))

    if accepted(1.):
        return 1.
    lo, hi = 0., 1.
    for _ in range(12):
        middle = .5 * (lo + hi)
        if accepted(middle):
            lo = middle
        else:
            hi = middle
    return lo


GRASP_POSE_CONTRACT = 'observed_object_pose_bounded_posture_v4_native_gn'


class GraspPostureInfeasible(RuntimeError):
    """A bounded posture command cannot satisfy the existing safety limits."""


class ObservedGraspOrientationServo:
    """Smallest regularized body/joint correction for a common grasp motion.

    Offsets are always relative to the uncorrected plan, never measured body
    drift. FK is recentered at the assembled CoM, matching the QPID contract.
    Contact force and teacher future state are not inputs.
    """
    def __init__(self, *, anchor_fk, joint_lower, joint_upper, joint_speeds, native_residual=None,
                 maximum_angle_rad=.20, maximum_offset_m=.030,
                 maximum_joint_offset_rad=.10, maximum_joint_speed_radps=.05,
                 maximum_linear_speed_mps=.020, maximum_angular_speed_radps=.15,
                 response_time_s=.50):
        self.anchor_fk = anchor_fk
        self.native_residual = native_residual
        self.lower = np.asarray(joint_lower, dtype=float)
        self.upper = np.asarray(joint_upper, dtype=float)
        self.speeds = np.asarray(joint_speeds, dtype=float)
        self.scales = np.r_[np.full(3, maximum_offset_m / np.sqrt(3)),
                            np.full(3, maximum_angle_rad / np.sqrt(3)),
                            np.full(len(self.lower), maximum_joint_offset_rad)]
        self.rates = np.r_[np.full(3, maximum_linear_speed_mps / np.sqrt(3)),
                           np.full(3, maximum_angular_speed_radps / np.sqrt(3)),
                           np.minimum(self.speeds, maximum_joint_speed_radps)]
        self.maximum_angle_rad = float(maximum_angle_rad)
        self.maximum_linear_speed_mps = float(maximum_linear_speed_mps)
        self.response_time_s = float(response_time_s)
        from amsrr.feasibility.order9_native_loader import load_order9_posture_native
        self._bounded_lsq = load_order9_posture_native().bounded_least_squares
        if (self.lower.shape != self.upper.shape or self.speeds.shape != self.lower.shape
                or not np.isfinite(np.r_[self.lower, self.upper, self.scales, self.rates,
                                        self.response_time_s]).all()
                or (self.lower >= self.upper).any() or (self.scales <= 0).any()
                or (self.rates <= 0).any() or self.response_time_s <= 0):
            raise ValueError('invalid grasp posture correction bounds')
        self.reset()

    def reset(self):
        self.offset = np.zeros(len(self.scales))
        self.request_rotation = np.zeros(3)
        self.binding = None
        self.regulation_phase = None
        self.regulating = False
        self.diagnostics = {}

    @property
    def translation(self):
        return self.offset[:3]

    @property
    def rotation(self):
        return self.offset[3:6]

    def set_binding(self, binding, *, episode_reset=False):
        if episode_reset or binding != self.binding:
            self.reset()
            self.binding = binding

    def poses(self, body, q, offsets):
        """Batch FK with body offsets; anchor_fk returns CoM-relative poses."""
        offsets = np.atleast_2d(offsets)
        positions, rotations = self.anchor_fk(q[None, :] + offsets[:, 6:])
        rb = Rotation.from_rotvec(offsets[:, 3:6]).as_matrix() @ Rotation.from_quat(body[3:]).as_matrix()
        positions = np.einsum('bij,baj->bai', rb, positions) + body[None, None, :3] + offsets[:, None, :3]
        rotations = rb[:, None] @ rotations
        return positions, rotations

    @staticmethod
    def rigidity_error(positions, rotations, nominal_positions, nominal_rotations):
        # Relative transforms remove the common rigid motion of the grasp.
        p = (positions - positions[:1]) @ rotations[0]
        p0 = (nominal_positions - nominal_positions[:1]) @ nominal_rotations[0]
        r = rotations[0].T @ rotations
        r0 = nominal_rotations[0].T @ nominal_rotations
        return (float(np.max(np.linalg.norm(p - p0, axis=-1))),
                float(np.max(Rotation.from_matrix(r @ r0.transpose(0, 2, 1)).magnitude())))

    def propose(self, body, q, qdot, object_target, object_measured, *, weight, dt, hold=False,
                regulate_orientation=True):
        body, q, qdot, goal, measured = [np.asarray(a, dtype=float) for a in
                                       (body, q, qdot, object_target, object_measured)]
        if not np.isfinite(dt) or dt <= 0 or not np.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError('invalid grasp posture observation interval or weight')
        if (body.shape != (7,) or goal.shape != (7,) or measured.shape != (7,)
                or q.shape != self.lower.shape or qdot.shape != q.shape
                or not np.isfinite(np.r_[body, q, qdot, goal, measured]).all()):
            raise ValueError('invalid grasp posture inputs')
        if weight == 0:
            # A zero-weight rotation is exactly zero; still reject invalid
            # quaternions as the scalar rotation constructor does.
            if not np.any(goal[3:]) or not np.any(measured[3:]):
                raise ValueError('zero norm grasp object quaternion')
            error = np.zeros(3)
        else:
            error = (Rotation.from_quat(goal[3:]) * Rotation.from_quat(measured[3:]).inv()).as_rotvec() * weight
        if weight > 0 and not regulate_orientation:
            # A translation-only recovery preserves the accepted grasp attitude.
            # Position error must not activate an unnecessary rotation near a
            # collision boundary when the object's attitude already meets goal.
            error = self.request_rotation.copy()
        error *= min(1., self.maximum_angle_rad / max(np.linalg.norm(error), 1e-12))
        desired = self.request_rotation + (error - self.request_rotation) * (-math.expm1(-dt / self.response_time_s))
        lo = np.maximum(-self.scales, self.offset - self.rates * dt)
        hi = np.minimum(self.scales, self.offset + self.rates * dt)
        lo[6:] = np.maximum(lo[6:], np.maximum(self.lower - q, self.offset[6:] + (-self.speeds - qdot) * dt))
        hi[6:] = np.minimum(hi[6:], np.minimum(self.upper - q, self.offset[6:] + (self.speeds - qdot) * dt))
        if (hi <= lo).any():
            raise GraspPostureInfeasible('no admissible grasp posture velocity interval')
        if hold:
            if weight != 0 or (self.offset < lo - 1e-12).any() or (self.offset > hi + 1e-12).any():
                raise GraspPostureInfeasible('no admissible held grasp posture')
            self.diagnostics = dict(mode='hold')
            return self.offset.copy(), self.request_rotation.copy()
        if (weight == 0 and np.max(np.abs(self.offset / self.scales)) < 1e-5
                and (lo <= 0).all() and (hi >= 0).all()):
            # Numerical tail only; the zero command still passes runtime safety.
            self.diagnostics = dict(mode='inactive')
            return np.zeros_like(self.offset), desired
        nominal_p, nominal_r = self.poses(body, q, np.zeros_like(self.offset))
        nominal_p, nominal_r = nominal_p[0], nominal_r[0]
        turn = Rotation.from_rotvec(desired).as_matrix()
        target_p = (nominal_p - goal[:3]) @ turn.T + goal[:3]
        target_r = turn @ nominal_r
        if weight > 0:
            # Recover object translation lost through material slip. Carry only
            # the displacement actually accepted by the bounded posture solver:
            # an unreachable target cannot wind up an independent integrator.
            previous_p, previous_r = self.poses(body, q, self.offset)
            previous_turn = Rotation.from_rotvec(self.request_rotation).as_matrix()
            previous_rotated = (nominal_p - goal[:3]) @ previous_turn.T + goal[:3]
            retained_translation = (previous_p[0] - previous_rotated).mean(axis=0)
            translation_step = (goal[:3] - measured[:3]) * weight * (-math.expm1(-dt / self.response_time_s))
            translation_step *= min(1., self.maximum_linear_speed_mps * dt / max(np.linalg.norm(translation_step), 1e-12))
            target_p += retained_translation + translation_step
            if not regulate_orientation:
                # The soft IK objective can leave a requested angle unrealized.
                # Preserve actual accepted hand poses, not the old requested
                # angle; chasing that remainder can block every safe translation.
                target_p = previous_p[0] + translation_step
                target_r = previous_r[0]
        if weight == 0:
            # Decay in hand-pose space, then solve the coupled posture again.
            # Independent clipping of joint/body offsets can open the grasp.
            previous_p, previous_r = self.poses(body, q, self.offset)
            decay = math.exp(-dt / self.response_time_s)
            target_p = nominal_p + (previous_p[0] - nominal_p) * decay
            relative = Rotation.from_matrix(previous_r[0] @ nominal_r.transpose(0, 2, 1)).as_rotvec()
            target_r = Rotation.from_rotvec(relative * decay).as_matrix() @ nominal_r
        def residual_batch(normalized):
            x = np.atleast_2d(normalized) * self.scales
            if self.native_residual is not None:
                return self.native_residual(q, x, np.atleast_2d(normalized), body,
                                            target_p, target_r, nominal_p, nominal_r)
            p, r = self.poses(body, q, x)
            dr = r @ target_r.transpose(0, 2, 1)
            angle = Rotation.from_matrix(dr.reshape(-1, 3, 3)).as_rotvec().reshape(len(x), -1)
            # Both positions and orientations are required. Regularization is
            # dimensionless and penalizes absolute departure from the plan.
            relative_p = np.einsum('baji,baj->bai', np.broadcast_to(r[:, :1], r.shape), p - p[:, :1])
            nominal_relative_p = (nominal_p - nominal_p[:1]) @ nominal_r[0]
            relative_r = r[:, :1].transpose(0, 1, 3, 2) @ r
            nominal_relative_r = nominal_r[0].T @ nominal_r
            relative_angle = Rotation.from_matrix((relative_r @ nominal_relative_r.transpose(0, 2, 1)).reshape(-1, 3, 3)).as_rotvec().reshape(len(x), -1)
            return np.concatenate(((p - target_p).reshape(len(x), -1) / .001,
                                   angle / .01,
                                   (relative_p - nominal_relative_p).reshape(len(x), -1) / .00005,
                                   relative_angle / .0005,
                                   np.atleast_2d(normalized) * .1), axis=1)
        def residual(x):
            return residual_batch(x)[0]
        def jacobian(x):
            h = 1e-5
            samples = np.vstack((x, x + h * np.eye(len(x))))
            r = residual_batch(samples)
            return ((r[1:] - r[0]) / h).T
        # The per-tick trust region is already set by physical velocity bounds.
        # Two bounded Gauss-Newton refinements replace a cold dense TRF solve.
        # Accept only a decrease of the actual nonlinear objective, then the
        # caller independently checks grasp rigidity and whole-body collision.
        lower, upper = lo / self.scales, hi / self.scales
        x = np.clip(self.offset / self.scales, lower, upper)
        evaluations, linear_iterations = 0, 0
        linear_converged = False
        for _ in range(2):
            r, j = residual(x), jacobian(x)
            evaluations += 1
            trial, linear_converged, iterations, _ = self._bounded_lsq(
                j, j @ x - r, lower, upper, 180, 1e-9)
            linear_iterations += iterations
            next_r = residual(trial)
            evaluations += 1
            if not np.isfinite(next_r).all() or next_r @ next_r > r @ r:
                break
            x = trial
        candidate = x * self.scales
        p, r = self.poses(body, q, candidate)
        rigidity = self.rigidity_error(p[0], r[0], nominal_p, nominal_r)
        self.diagnostics = dict(mode='regulate' if weight else 'decay', optimizer_evaluations=evaluations,
            optimizer_method='bounded_gauss_newton_2', linear_solver_converged=bool(linear_converged),
            linear_solver_iterations=linear_iterations, object_error_rad=float(np.linalg.norm(error)),
            object_position_error_m=float(np.linalg.norm(goal[:3] - measured[:3])),
            target_position_error_m=float(np.max(np.linalg.norm(p[0] - target_p, axis=-1))),
            target_orientation_error_rad=float(np.max(Rotation.from_matrix(r[0] @ target_r.transpose(0, 2, 1)).magnitude())),
            relative_grasp_position_error_m=rigidity[0], relative_grasp_angle_error_rad=rigidity[1])
        return candidate, desired

    def command(self, body, offset, dt):
        result = np.asarray(body).copy()
        if not np.any(offset) and not np.any(self.offset):
            # Exact identity transform; keep the quaternion normalization and
            # owned arrays of the general command path.
            result[3:] = Rotation.from_quat(body[3:]).as_quat()
            return result, np.zeros(6), np.zeros_like(offset[6:])
        result[:3] += offset[:3]
        result[3:] = (Rotation.from_rotvec(offset[3:6]) * Rotation.from_quat(body[3:])).as_quat()
        velocity = np.r_[(offset[:3] - self.offset[:3]) / dt,
            (Rotation.from_rotvec(offset[3:6]) * Rotation.from_rotvec(self.offset[3:6]).inv()).as_rotvec() / dt]
        return result, velocity, (offset[6:] - self.offset[6:]) / dt

    def accept(self, offset, request_rotation):
        self.offset = np.asarray(offset).copy()
        self.request_rotation = np.asarray(request_rotation).copy()


class MaterialSlipClosure:
    """Monotone proportional closure: a persistent error cannot wind up."""
    def __init__(self, *, span_m, deadband_m=0.002, rate_mps=0.004):
        if not all(math.isfinite(x) and x >= 0 for x in (span_m, deadband_m, rate_mps)):
            raise ValueError('invalid slip closure bounds')
        self.span_m = span_m
        self.deadband_m = deadband_m
        self.rate_mps = rate_mps
        self.reset()

    def reset(self):
        self.reference = None
        self.closure_m = 0.0
        self.last_slip_m = 0.0
        self.binding = None

    def update(self, points_object, normals_object, *, binding, maintain, contact_present, dt):
        if dt <= 0 or not math.isfinite(dt):
            raise ValueError('invalid slip observation interval')
        points = np.asarray(points_object, dtype=float)
        normals = np.asarray(normals_object, dtype=float)
        if points.shape != normals.shape or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('slip points/normals must be [contact,3]')
        if not np.isfinite(points).all() or not np.isfinite(normals).all():
            raise ValueError('nonfinite slip observation')
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        if np.any(lengths < 1e-8):
            raise ValueError('invalid contact normal')
        normals = normals / lengths
        if self.binding != binding:
            self.reset()
            self.binding = binding
        if self.reference is None or not maintain:
            self.reference = points.copy()
            return self.closure_m
        displacement = points - self.reference
        tangent = displacement - (displacement * normals).sum(1, keepdims=True) * normals
        self.last_slip_m = float(np.max(np.linalg.norm(tangent, axis=1)))
        if contact_present:
            desired = min(self.span_m, max(0., self.last_slip_m - self.deadband_m))
            self.closure_m = min(max(self.closure_m, desired), self.closure_m + self.rate_mps * dt)
        return self.closure_m


def build_slip_reserve(*, morphology, physical_model, contact_knot, nominal_contact_knot, candidates,
                       compression, preload, preload_config):
    """One selected-group reserve, bounded by nominal actuator load estimates."""
    from amsrr.feasibility.articulated_reachability import (
        CentroidalPostureIKSolver, base_pose_for_centroidal_target,
        resolve_mesh_backed_anchor_references,
    )
    from amsrr.robot_model.whole_structure_kinematics import ordered_global_dock_joint_ids
    from amsrr.training.order9_actuator_aware_nominal_preload import _global_joint_actuator_values
    from amsrr.training.order9_virtual_contact_compression import (
        _global_joint_limits, solve_order9_virtual_contact_compression_for_achieved_lead,
    )
    ids = ordered_global_dock_joint_ids(morphology, physical_model)
    q = nominal_contact_knot.posture_target.joint_pos_target
    c = nominal_contact_knot.centroidal_target
    pose = (*c.com_pos_world, *c.body_orientation_world)
    assignments = contact_knot.contact_assignments
    empty = dict(version=GRASP_SLIP_CONTRACT, span_m=0., joint_delta_rad={}, reason=None)
    if any(a.contact_mode.value != 'grasp' for a in assignments):
        return dict(empty, reason='non_grasp_binding')
    solver = CentroidalPostureIKSolver(physical_model)
    refs = resolve_mesh_backed_anchor_references(morphology, physical_model, [a.anchor_id for a in assignments])
    base = base_pose_for_centroidal_target(morphology, physical_model, q, pose[:3], pose[3:], kinematics=solver.kinematics)
    fk = solver.kinematics.forward(morphology, physical_model, q, base, refs)
    js = solver._fixed_centroidal_anchor_jacobians(morphology=morphology,
        centroidal_pose_world=pose, q=q, limits=_global_joint_limits(ids, physical_model),
        references=refs, nominal=fk.anchor_poses_world)
    by_id = {c.candidate_id: c for c in candidates.candidates}
    normals = np.array([by_id[a.candidate_id].normal_world for a in assignments])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    jacobians = np.array([js[a.anchor_id][:3] for a in assignments])
    jn = np.einsum('ai,aij->aj', -normals, jacobians)
    _, peak, _ = _global_joint_actuator_values(ids, physical_model)
    normal_forces = np.asarray(preload.target_normal_force_n_by_anchor)
    if preload.target_contact_force_world_n:
        # A scalar increase of the nominal normal forces is safe only if it
        # is an internal, self-equilibrated squeeze. Extra arbitrary loads
        # would move/rotate the object instead of increasing grasp reserve.
        points = np.array([by_id[a.candidate_id].contact_pose_world[:3] for a in assignments])
        squeezing = -normals * normal_forces[:, None]
        wrench = np.r_[squeezing.sum(0), np.cross(points-points.mean(0), squeezing).sum(0)]
        if np.max(np.abs(wrench)) > 1e-6:
            return dict(empty, reason='normal_increment_is_not_self_equilibrated')
    # The existing opposed vertical-grasp model allocates support in proportion
    # to its normal loads. Include this missing tangential torque when bounding
    # ADDITIONAL closure; this does not claim that the nominal load is measured.
    # support_safety_factor is a friction reserve, not additional payload mass.
    # The controller and this torque estimate both support the nominal weight.
    weight = preload.required_tangential_support_force_n / preload_config.support_safety_factor
    support = weight * normal_forces / normal_forces.sum()
    tau_normal = jn.T @ normal_forces
    tau_support = jacobians[:, 2, :].T @ support
    tau = (np.einsum('aij,ai->j', jacobians, preload.target_contact_force_world_n)
           if preload.target_contact_force_world_n else tau_normal + tau_support)
    limit = peak * preload_config.maximum_peak_effort_utilization
    if np.any(np.abs(tau) > limit + 1e-9):
        return dict(empty, reason='no_estimated_effort_reserve')
    factor = 1.0  # reserve at most one further nominal closure, never arbitrary gain
    for t, slope, bound in zip(tau, tau_normal, limit):
        if slope > 1e-12:
            factor = min(factor, (bound - t) / slope)
        elif slope < -1e-12:
            factor = min(factor, (-bound - t) / slope)
    current = max(compression.achieved_inward_displacement_m.values())
    span = min(preload.inward_lead_m * max(0., factor), preload_config.maximum_inward_lead_m - current)
    if span < preload_config.inward_lead_quantization_m:
        return dict(empty, reason='insufficient_nominal_reserve')
    high = solve_order9_virtual_contact_compression_for_achieved_lead(
        morphology=morphology, physical_model=physical_model, contact_knot=nominal_contact_knot,
        candidate_set=candidates, minimum_achieved_inward_lead_m=span,
        maximum_requested_inward_lead_m=preload_config.maximum_inward_lead_m - current,
        requested_lead_quantization_m=preload_config.inward_lead_quantization_m)
    # A high IK endpoint can exceed a tangential bound while a smaller part of
    # its direction is useful. Admit only the explicitly checked FK fraction,
    # preserving both total and incremental tangential limits.
    direction = {j: high.joint_delta_rad.get(j, 0.) for j in ids}
    def positions_at_scale(scale):
        changed_q = {j: q[j] + scale * direction[j] for j in ids}
        changed_base = base_pose_for_centroidal_target(morphology, physical_model,
            changed_q, pose[:3], pose[3:], kinematics=solver.kinematics)
        changed = solver.kinematics.forward(morphology, physical_model, changed_q, changed_base, refs)
        return np.array([changed.anchor_poses_world[a.anchor_id][:3] for a in assignments])
    surfaces = np.array([contact_knot.posture_target.free_anchor_pose_targets[a.anchor_id][:3]
                         for a in assignments])
    scale = _bounded_fk_reserve_scale(positions_at_scale, normals, span,
                                     surface_points=surfaces)
    if scale <= 0.:
        return dict(empty, reason='no_bounded_fk_reserve')
    extra = -np.einsum('ai,ai->a', positions_at_scale(scale) - positions_at_scale(0.), normals)
    achieved_span = float(extra.max())
    if achieved_span <= 1e-9:
        return dict(empty, reason='no_bounded_fk_reserve')
    direction = {j: scale * value for j, value in direction.items()}
    return dict(version=GRASP_SLIP_CONTRACT, span_m=achieved_span,
        requested_span_m=span, joint_delta_rad=direction,
        reserve_ik_scale=scale, maximum_extra_inward_displacement_m=float(extra.max()),
        estimated_peak_effort_utilization=float(np.max(np.abs(tau + factor * tau_normal) / peak)),
        deadband_m=.002, rate_mps=preload_config.inward_lead_quantization_m / .25,
        normal_points_reference='observed_acquisition_material_points',
        command_safety='native_configuration_check_each_modified_control_sample',
        continuous_collision_certificate=False, reason=None)


class CheckedGraspSlipController:
    active_environments = None
    """Apply the reserve only through the existing position/velocity controller."""
    def __init__(self, *, plan, bundle, physical_model, task, count, dt=.02):
        from amsrr.training.order9_posture_resolver import Order9PostureTrajectoryResolver, Order9PostureResolverConfig
        from amsrr.training.order9_c3_teacher import build_order9_c3_posture_collision_object
        from amsrr.training.order9_virtual_contact_compression import _global_joint_limits
        from amsrr.robot_model.whole_structure_kinematics import ordered_global_dock_joint_ids
        from amsrr.geometry.pose_math import transform_from_pose
        self.reserve = plan['evidence']['slip_reserve']
        if self.reserve['version'] != GRASP_SLIP_CONTRACT:
            raise ValueError('slip reserve contract mismatch')
        self.bundle, self.physical, self.dt = bundle, physical_model, dt
        self.ids = ordered_global_dock_joint_ids(bundle.morphology, physical_model)
        self.limits = _global_joint_limits(self.ids, physical_model)
        local = {j.joint_id: j for j in physical_model.joints}
        self.speeds = np.array([local[j.split(':', 1)[1]].velocity_limit for j in self.ids])
        self.direction = np.array([self.reserve['joint_delta_rad'].get(j, 0.) for j in self.ids])
        self.collision_object = build_order9_c3_posture_collision_object(task)
        self.solver = Order9PostureTrajectoryResolver(physical_model,
            config=Order9PostureResolverConfig(collision_margin_m=.001),
            collision_object=self.collision_object, prefer_native_solver=True, require_native_solver=True).ik_solver
        contact = bundle.phase_trajectories['contact_acquisition'].knots[-1]
        assignments = sorted(contact.contact_assignments, key=lambda a: a.anchor_id)
        self.anchors = tuple(a.anchor_id for a in assignments)
        cs = {c.candidate_id: c for c in bundle.contact_candidate_set.candidates}
        rotation = np.asarray(transform_from_pose(task.scene.objects[0].pose_world).rotation)
        self.normals = np.array([rotation.T @ np.asarray(cs[a.candidate_id].normal_world) for a in assignments])
        self.controllers = [MaterialSlipClosure(span_m=self.reserve['span_m'],
            deadband_m=self.reserve.get('deadband_m', .002), rate_mps=self.reserve.get('rate_mps', .004)) for _ in range(count)]
        self.previous_offsets = np.zeros((count, len(self.ids)))
        from amsrr.feasibility.articulated_reachability import resolve_mesh_backed_anchor_references
        self.references = resolve_mesh_backed_anchor_references(bundle.morphology, physical_model, self.anchors)
        self._bound_anchor_fk = self.solver.kinematics.bind_centroidal_anchor_fk(
            bundle.morphology, self.references, self.ids)
        self.pose_controllers = [ObservedGraspOrientationServo(
            anchor_fk=self._anchor_fk,
            native_residual=self._bound_anchor_fk.residual_batch,
            joint_lower=[self.limits[j][0] for j in self.ids],
            joint_upper=[self.limits[j][1] for j in self.ids],
            joint_speeds=self.speeds) for _ in range(count)]
        self.phase_goal_joints = {name: np.array([
            trajectory.knots[-1].posture_target.joint_pos_target[j] for j in self.ids])
            for name, trajectory in bundle.phase_trajectories.items()}
        self._ground_forecast_cache = None
        self._release_clock = [None for _ in range(count)]
        self.release_decay_progress = [None for _ in range(count)]
        self.release_payload_scale = [1. for _ in range(count)]
        self.trace = []
        self.safety_rejections = {}
        self.terminate_on_rejection = False
        # Build the native contact-pair caches at setup, not on the first
        # corrective control tick. This query does not advance feedback state.
        centroidal = contact.centroidal_target
        self._safe(np.array([contact.posture_target.joint_pos_target[j] for j in self.ids]),
                   np.r_[centroidal.com_pos_world, centroidal.body_orientation_world],
                   np.asarray(task.scene.objects[0].pose_world))

    def _anchor_fk(self, q_samples):
        # Existing C++ batch FK returns base-root poses and the actual CoM.
        # Recenter each q sample independently before applying a body pose.
        return self._bound_anchor_fk(q_samples)

    def reference_elapsed(self, phase_index, phase_elapsed_s):
        """Release accepted feedback before moving along the nominal retreat.

        Only the reference clock pauses. Episode/phase elapsed time, actor
        observations, deadlines and success predicates keep their real clocks.
        """
        from amsrr.training.request_imitation import PHASES
        from amsrr.simulation.order9_tensor_object_task import ORDER9_RELEASE_PAYLOAD_HANDOFF_FRACTION
        result = phase_elapsed_s.clone()
        for i, (phase, elapsed) in enumerate(zip(phase_index.tolist(), phase_elapsed_s.tolist())):
            if PHASES[phase] != 'release':
                self._release_clock[i] = None
                self.release_decay_progress[i] = None
                self.release_payload_scale[i] = 1.
                continue
            if self._release_clock[i] is None:
                self._release_clock[i] = dict(pending=True, delay=0.)
            clock = self._release_clock[i]
            duration = self.bundle.phase_trajectories['release'].knots[-1].t_rel_s
            handoff = min(1., elapsed / max(float(duration) * ORDER9_RELEASE_PAYLOAD_HANDOFF_FRACTION, 1.e-6))
            self.release_payload_scale[i] = 1. - handoff * handoff * (3. - 2. * handoff)
            if clock['pending']:
                progress = min(1., elapsed / max(float(duration), 1e-6))
                self.release_decay_progress[i] = progress
                pose = self.pose_controllers[i]
                ready = (handoff >= 1. and not np.any(pose.offset)
                         and not np.any(self.previous_offsets[i]))
                clock['delay'] = elapsed
                if ready:
                    clock['pending'] = False
                    self.trace.append(dict(environment=i, event='grasp_feedback_released',
                        release_elapsed_s=elapsed, reference_hold_s=clock['delay']))
            result[i] = max(0., elapsed - clock['delay'])
        return result

    def _posture_safe(self, servo, offset, q, body, obj):
        p0, r0 = servo.poses(body, q, np.zeros_like(offset))
        p, r = servo.poses(body, q, offset)
        displacement, angle = servo.rigidity_error(p[0], r[0], p0[0], r0[0])
        if displacement > .0005 or angle > .005:
            return False
        corrected, _, _ = servo.command(body, offset, self.dt)
        return self._safe(q + offset[6:], corrected, obj)

    def _anticipate_ground_clearance(self, servo, candidate, supervisor, goal_body, closure):
        """Keep a carried correction clear of the ground at its planned endpoint.

        A common vertical translation preserves grasp rigidity. Start it while
        descending, using the existing absolute/rate bounds, rather than waiting
        for a ground violation. Actual samples still pass all collision checks.
        """
        if self.collision_object.ground_plane_z_m is None or not np.any(candidate):
            return candidate
        from amsrr.training.request_imitation import PHASES
        q_goal = self.phase_goal_joints[PHASES[supervisor.phase]] + closure + candidate[6:]
        body_goal = np.asarray(goal_body).copy()
        body_goal[:3] -= supervisor.environment_origin.numpy()
        body_goal, _, _ = servo.command(body_goal, candidate, self.dt)
        key = (q_goal.tobytes(), body_goal.tobytes())
        if self._ground_forecast_cache is None or self._ground_forecast_cache[0] != key:
            checked = self.solver.check_configuration(morphology=self.bundle.morphology,
                centroidal_pose_world=tuple(body_goal), joint_positions_rad=dict(zip(self.ids,q_goal)),
                exact=False, margin_m=.001, ground_plane_z_m=self.collision_object.ground_plane_z_m)
            clearance = float(checked['minimum_ground_clearance_m'])
            self._ground_forecast_cache = (key, clearance)
        clearance = self._ground_forecast_cache[1]
        required_z = candidate[2] + max(0., .001 - clearance)
        if required_z > servo.scales[2] + 1e-9:
            raise GraspPostureInfeasible('planned ground clearance exceeds bounded grasp posture')
        result = candidate.copy()
        result[2] = max(result[2], min(required_z, servo.offset[2] + servo.rates[2] * self.dt))
        servo.diagnostics['ground_forecast_clearance_m'] = clearance
        servo.diagnostics['ground_forecast_required_z_offset_m'] = required_z
        return result

    def _safe(self, q, body, obj):
        from amsrr.training.order9_virtual_contact_compression import _compression_collision_result_accepted
        for value, j in zip(q, self.ids):
            if not self.limits[j][0] - 1e-7 <= value <= self.limits[j][1] + 1e-7:
                return False
        scenes = [(self.collision_object.object_id, obj, self.collision_object.size_m, self.anchors, True)]
        scenes += [(b.box_id, b.pose_world, b.size_m, (), False) for b in self.collision_object.environment_boxes]
        for _, pose, size, anchors, selected in scenes:
            self.solver.set_collision_scene(morphology=self.bundle.morphology, object_pose_world=tuple(pose), object_size_m=size, allowed_anchor_ids=anchors)
            result = self.solver.check_configuration(morphology=self.bundle.morphology,
                centroidal_pose_world=tuple(body), joint_positions_rad=dict(zip(self.ids, q)),
                exact=False, margin_m=.001, ground_plane_z_m=self.collision_object.ground_plane_z_m)
            if not _compression_collision_result_accepted(result, selected_contact_scene=selected, collision_margin_m=.001):
                return False
        return True

    def _unloaded_batch(self, host, supervisors):
        """Batch the identity correction before contact; retain scalar fallback.

        Requires exactly zero carried corrections and an admissible zero
        increment. Nonzero recovery keeps the normal solver/collision checks.
        """
        if not bool((host['schedule'] <= 1).all()):
            return False
        servos = self.pose_controllers
        if any(np.any(s.offset) or np.any(s.request_rotation) for s in servos):
            return False
        count = len(servos)
        q = host['q'].numpy().reshape(count, -1)
        qdot = host['qdot'].numpy().reshape(q.shape)
        body, goal, obj = (host[k].numpy() for k in ('body', 'object_target', 'object'))
        if (not np.isfinite(self.dt) or self.dt <= 0
                or any(not np.isfinite(a).all() for a in (q, qdot, body, goal, obj))):
            return False
        lower = np.stack([s.lower for s in servos])
        upper = np.stack([s.upper for s in servos])
        if q.shape != lower.shape or qdot.shape != lower.shape:
            return False
        speeds = np.stack([s.speeds for s in servos])
        scales = np.stack([s.scales[6:] for s in servos])
        rates = np.stack([s.rates[6:] for s in servos])
        lo = np.maximum(np.maximum(-scales, -rates * self.dt),
            np.maximum(lower - q, (-speeds - qdot) * self.dt))
        hi = np.minimum(np.minimum(scales, rates * self.dt),
            np.minimum(upper - q, (speeds - qdot) * self.dt))
        if (np.any(hi <= lo) or np.any(lo > 0) or np.any(hi < 0)
                or np.any(np.abs(qdot) > self.speeds + 1e-7)):
            return False
        # Same scipy quaternion validation/normalization as scalar calls.
        errors = (Rotation.from_quat(goal[:, 3:]) * Rotation.from_quat(obj[:, 3:]).inv()).magnitude()
        normalized_body = Rotation.from_quat(body[:, 3:]).as_quat()
        twist = host['twist'].numpy()
        obj_twist = host['object_twist'].numpy()
        endpoint = ((host['progress'].numpy().astype(np.float64) >= 1. - 1e-6)
            & (np.max(np.abs(twist), axis=1) <= 1e-6)
            & (np.max(np.abs(qdot), axis=1) <= 1e-6))
        settled = ((np.linalg.norm(obj_twist[:, :3], axis=1) <= .05)
            & (np.linalg.norm(obj_twist[:, 3:], axis=1) <= .1))
        position_error = np.linalg.norm(goal[:, :3] - obj[:, :3], axis=1)
        for i, (control, servo, supervisor) in enumerate(zip(self.controllers, servos, supervisors)):
            control.reset()
            servo.set_binding(supervisor.expected_group, episode_reset=supervisor.time_s <= 0.)
            if servo.regulation_phase != supervisor.phase:
                servo.regulation_phase = supervisor.phase
                servo.regulating = False
            final = supervisor.phase in (4, 5, 7)
            orientation_tolerance = supervisor.final_goal_orientation_tolerance_rad if final else .2
            position_tolerance = supervisor.final_goal_position_tolerance_m if final else .052
            if endpoint[i] and settled[i] and (errors[i] > orientation_tolerance or position_error[i] > position_tolerance):
                servo.regulating = True
            servo.offset.fill(0.)
            servo.request_rotation.fill(0.)
            servo.diagnostics = dict(mode='inactive')
            if int(round(supervisor.time_s / self.dt)) % 50 == 0:
                self.trace.append(dict(environment=i, time_s=supervisor.time_s, schedule=int(host['schedule'][i]),
                    slip_m=0., closure_m=0., applied_scale=0.,
                    grasp_pose_rotation_rad=[0.] * 3, grasp_pose_translation_m=[0.] * 3,
                    grasp_pose_target_twist=[0.] * 6, grasp_pose_safety_accepted=True,
                    grasp_pose_joint_offset_rad=[0.] * q.shape[1],
                    grasp_pose_joint_velocity_radps=[0.] * q.shape[1],
                    grasp_pose_requested_rotation_rad=[0.] * 3, grasp_pose_solver=servo.diagnostics))
        self.previous_offsets.fill(0.)
        q += 0.
        qdot += 0.
        body[:, 3:] = normalized_body
        twist[:, :3] += 0.
        twist[:, 3:] = Rotation.from_rotvec(np.zeros((count, 3))).apply(twist[:, 3:]) + 0.
        return True

    @staticmethod
    def _replace_command(target, host):
        return replace(target, **{name: host[key].to(
            device=getattr(target, name).device, dtype=getattr(target, name).dtype).reshape_as(getattr(target, name))
            for name, key in [('desired_robot_root_pose_world', 'body'),
                ('desired_robot_root_twist_world', 'twist'), ('nominal_joint_positions_rad', 'q'),
                ('nominal_joint_velocities_radps', 'qdot')]})

    def _reject_command(self, environment, reason):
        # Interactive callers still fail closed. The request runner explicitly
        # opts into per-environment termination and masks all actuator outputs.
        if not self.terminate_on_rejection:
            raise RuntimeError(reason)
        self.safety_rejections.setdefault(environment, reason)

    def apply(self, target, state, supervisors):
        from amsrr.utils.tensor_snapshot import cpu_snapshot
        host = cpu_snapshot(dict(q=target.nominal_joint_positions_rad,
            qdot=target.nominal_joint_velocities_radps,
            body=target.desired_robot_root_pose_world, twist=target.desired_robot_root_twist_world,
            object_target=target.desired_object_pose_world, object=state.object_pose_world,
            object_twist=state.object_twist_world, schedule=target.contact_schedule_index,
            progress=target.phase_progress, goal=target.phase_goal_robot_root_pose_world))
        all_active = self.active_environments is None or all(self.active_environments)
        if all_active and not self.safety_rejections and self._unloaded_batch(host, supervisors):
            return self._replace_command(target, host)
        q = host['q'].numpy().reshape(len(self.controllers), -1)
        qdot = host['qdot'].numpy().reshape(q.shape)
        body_targets = host['body'].numpy()
        body_twists = host['twist'].numpy()
        object_targets = host['object_target'].numpy()
        object_observed = host['object'].numpy()
        object_twists = host['object_twist'].numpy()
        orientation_errors = (Rotation.from_quat(object_targets[:, 3:])
            * Rotation.from_quat(object_observed[:, 3:]).inv()).magnitude()
        cache = {}
        for i, (control, supervisor) in enumerate(zip(self.controllers, supervisors)):
            if (i in self.safety_rejections or
                    (self.active_environments is not None and not self.active_environments[i])):
                continue
            schedule = int(host['schedule'][i])
            points = supervisor.contact_velocity_observer.previous
            if schedule <= 1:
                control.reset()
                self.previous_offsets[i].fill(0.)
            elif points is not None:
                control.update(points[0].detach().cpu().numpy(), self.normals,
                    binding=supervisor.expected_group, maintain=schedule in (3, 4),
                    contact_present=getattr(supervisor, 'grip_contact_present', False) and schedule == 3, dt=self.dt)
            weight = 1. if schedule == 3 else 0.
            if schedule == 4:
                decay = getattr(self, 'release_decay_progress', None)
                p = (float(host['progress'][i]) if decay is None or decay[i] is None
                     else decay[i])
                weight = 1. - p * p * (3. - 2. * p)
            desired = (control.closure_m / max(self.reserve['span_m'], 1e-12)) * weight
            previous = self.previous_offsets[i]
            offset = desired * self.direction
            extra_velocity = (offset - previous) / self.dt
            if np.any(np.abs(qdot[i] + extra_velocity) > self.speeds + 1e-7):
                # Retain the previous safe command increment, avoiding a jump.
                scale = min(1., float(np.min((self.speeds - np.abs(qdot[i])) / (np.abs(extra_velocity) + 1e-12))))
                offset = previous + max(0., scale) * (offset - previous)
            # With an active body/joint correction, the closure-only posture is
            # never commanded. Check the composed posture below instead; the
            # intermediate geometry may collide while the actual command is safe.
            closure_safety_deferred = bool(np.any(self.pose_controllers[i].offset))
            if np.any(np.abs(offset) > 1e-12) and not closure_safety_deferred:
                body = body_targets[i].copy()
                obj = object_observed[i].copy()
                origin = supervisor.environment_origin.numpy();body[:3] -= origin;obj[:3] -= origin
                key = (q[i].tobytes(), offset.tobytes(), body.tobytes(), obj.tobytes())
                if key not in cache:
                    cache[key] = self._safe(q[i] + offset, body, obj)
                if not cache[key]:
                    if self._safe(q[i] + previous, body, obj):
                        offset = previous.copy()
                        control.closure_m = min(control.closure_m, self.reserve['span_m'] * float(np.linalg.norm(previous)) / max(1e-12, float(np.linalg.norm(self.direction))))
                    else:
                        self.trace.append(dict(error='no_safe_grasp_closure', environment=i,
                            time_s=supervisor.time_s, nominal_body=body.tolist(),
                            object=obj.tolist(), q=q[i].tolist(),
                            proposed_closure=offset.tolist(), previous_closure=previous.tolist()))
                        self._reject_command(i, 'no safe continuous grasp closure command')
                        continue
            q[i] += offset
            qdot[i] += (offset - previous) / self.dt
            self.previous_offsets[i] = offset
            pose_control = self.pose_controllers[i]
            pose_control.set_binding(supervisor.expected_group,
                episode_reset=supervisor.time_s <= 0.)
            if pose_control.regulation_phase != supervisor.phase:
                pose_control.regulation_phase = supervisor.phase
                pose_control.regulating = False
            # Do not oppose transient dynamics while following a moving nominal
            # path. Regulate the object only at a held endpoint;
            # the existing filter/rate bounds release corrections continuously.
            endpoint_hold = (
                float(host['progress'][i]) >= 1. - 1e-6
                and np.max(np.abs(body_twists[i])) <= 1e-6
                and np.max(np.abs(qdot[i])) <= 1e-6
            )
            observed_settled = (np.linalg.norm(object_twists[i, :3]) <= .05
                and np.linalg.norm(object_twists[i, 3:]) <= .1)
            orientation_error = orientation_errors[i]
            orientation_tolerance = (supervisor.final_goal_orientation_tolerance_rad
                if supervisor.phase in (4, 5, 7) else .2)
            position_error = np.linalg.norm(object_targets[i, :3] - object_observed[i, :3])
            position_tolerance = (supervisor.final_goal_position_tolerance_m
                if supervisor.phase in (4, 5, 7) else .052)
            if endpoint_hold and observed_settled and (orientation_error > orientation_tolerance or position_error > position_tolerance):
                pose_control.regulating = True
            pose_weight = weight if schedule == 3 and endpoint_hold and pose_control.regulating else 0.
            if schedule == 3 and not getattr(supervisor, 'grip_contact_present', False):
                pose_weight = 0.
            try:
                candidate, requested_rotation = pose_control.propose(
                    body_targets[i], q[i], qdot[i], object_targets[i], object_observed[i],
                    weight=pose_weight, dt=self.dt,
                    regulate_orientation=orientation_error > orientation_tolerance,
                    hold=(pose_weight == 0 and schedule == 3
                          and getattr(supervisor, 'grip_contact_present', False)
                          and np.linalg.norm(pose_control.offset) > 0))
                candidate = self._anticipate_ground_clearance(pose_control, candidate, supervisor,
                    host['goal'][i].numpy(), offset)
            except GraspPostureInfeasible as exc:
                self.trace.append(dict(error='infeasible_grasp_posture', environment=i,
                    time_s=supervisor.time_s, reason=str(exc), q=q[i].tolist(),
                    qdot=qdot[i].tolist(), previous_offset=pose_control.offset.tolist()))
                self._reject_command(i, str(exc))
                continue
            pose_accepted = True
            if (np.linalg.norm(candidate) > 0 or np.linalg.norm(pose_control.offset) > 0
                    or (closure_safety_deferred and np.any(offset))):
                origin = supervisor.environment_origin.numpy()
                local_body = body_targets[i].copy(); local_body[:3] -= origin
                local_object = object_observed[i].copy(); local_object[:3] -= origin
                pose_accepted = self._posture_safe(pose_control, candidate, q[i], local_body, local_object)
                if not pose_accepted:
                    candidate = pose_control.offset.copy()
                    requested_rotation = pose_control.request_rotation.copy()
                    if not self._posture_safe(pose_control, candidate, q[i], local_body, local_object):
                        self.trace.append(dict(error='no_safe_grasp_posture', environment=i,
                            time_s=supervisor.time_s, nominal_body=body_targets[i].tolist(),
                            object=object_observed[i].tolist(), q=q[i].tolist(),
                            previous_offset=candidate.tolist(), diagnostics=pose_control.diagnostics))
                        self._reject_command(i, 'no safe continuous grasp posture correction')
                        continue
            corrected, velocity, joint_velocity = pose_control.command(body_targets[i], candidate, self.dt)
            if np.any(np.abs(qdot[i] + joint_velocity) > self.speeds + 1e-7):
                self._reject_command(i, 'grasp posture total joint speed exceeds physical limit')
                continue
            q[i] += candidate[6:]
            qdot[i] += joint_velocity
            pose_control.accept(candidate, requested_rotation)
            rotation, translation = pose_control.rotation, pose_control.translation
            body_targets[i] = corrected
            body_twists[i, :3] += velocity[:3]
            body_twists[i, 3:] = Rotation.from_rotvec(rotation).apply(body_twists[i, 3:]) + velocity[3:]
            if int(round(supervisor.time_s / self.dt)) % 50 == 0:
                self.trace.append(dict(environment=i, time_s=supervisor.time_s, schedule=schedule,
                    slip_m=control.last_slip_m, closure_m=control.closure_m,
                    applied_scale=float(np.linalg.norm(offset) / max(1e-12, np.linalg.norm(self.direction))),
                    grasp_pose_rotation_rad=rotation.tolist(), grasp_pose_translation_m=translation.tolist(),
                    grasp_pose_target_twist=velocity.tolist(), grasp_pose_safety_accepted=pose_accepted,
                    grasp_pose_joint_offset_rad=candidate[6:].tolist(),
                    grasp_pose_joint_velocity_radps=joint_velocity.tolist(),
                    grasp_pose_requested_rotation_rad=requested_rotation.tolist(),
                    grasp_pose_solver=pose_control.diagnostics))
        return self._replace_command(target, host)
