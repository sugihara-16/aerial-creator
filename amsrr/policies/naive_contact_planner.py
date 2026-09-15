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
    _compression_collision_result_accepted,
    _global_joint_limits,
    solve_order9_virtual_contact_compression_for_achieved_lead,
)
from amsrr.utils.hashing import stable_hash

NAIVE_CONTACT_PLAN_VERSION = "naive_quasistatic_teacher_contact_plan_v1"
NAIVE_CONTACT_INTERPOLATION = "linear_centroidal_cubic_hermite_joint_v1"


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


class NaiveContactPlanReference(Order9C3NominalTensorReference):
    """The same continuous joint evaluator is used in checks and in Isaac."""

    runtime_version = NAIVE_CONTACT_INTERPOLATION

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.provenance["control_sampling_semantics"] = NAIVE_CONTACT_INTERPOLATION

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
    preload = solve_order9_actuator_aware_nominal_preload(
        morphology=bundle.morphology,
        physical_model=physical_model,
        contact_knot=contact,
        candidate_set=bundle.contact_candidate_set,
        object_mass_kg=object_mass_kg,
        contact_friction=contact_friction,
        contact_stiffness_n_per_m=contact_stiffness_n_per_m,
        config=preload_config,
    )
    if not preload.feasible:
        raise ValueError(f"contact preload infeasible: {preload.rejection_reason}")
    compression = solve_order9_virtual_contact_compression_for_achieved_lead(
        morphology=bundle.morphology,
        physical_model=physical_model,
        contact_knot=contact,
        candidate_set=bundle.contact_candidate_set,
        minimum_achieved_inward_lead_m=preload.inward_lead_m,
        maximum_requested_inward_lead_m=preload_config.maximum_inward_lead_m,
        requested_lead_quantization_m=preload_config.inward_lead_quantization_m,
    )
    # The legacy flag also records an intermediate continuation clip, even
    # when final least-squares refinement subsequently succeeds. Check the
    # final displacement and physical joint bounds instead of that history.
    if (
        min(compression.achieved_inward_displacement_m.values()) + 1e-9
        < preload.inward_lead_m
    ):
        raise ValueError("contact posture did not achieve the requested nominal lead")
    if max(compression.tangential_error_m.values()) > max(
        0.002, 0.25 * preload.inward_lead_m
    ):
        raise ValueError("contact posture has excessive tangential error")
    collision_object = build_order9_c3_posture_collision_object(task)
    resolver = Order9PostureTrajectoryResolver(
        physical_model,
        config=Order9PostureResolverConfig(collision_margin_m=0.001),
        collision_object=collision_object,
        prefer_native_solver=True,
        require_native_solver=True,
    )
    solver = resolver.ik_solver
    joint_ids = ordered_global_dock_joint_ids(bundle.morphology, physical_model)
    joint_limits = _global_joint_limits(joint_ids, physical_model)
    anchor_ids = tuple(sorted(a.anchor_id for a in contact.contact_assignments))
    references = resolve_mesh_backed_anchor_references(
        bundle.morphology, physical_model, anchor_ids
    )
    phases = {}
    maximum_joint_bound_correction = 0.0
    for phase, source in bundle.phase_trajectories.items():
        trajectory = ContactWrenchTrajectory.from_dict(source.to_dict())
        duration = float(trajectory.knots[-1].t_rel_s)
        times = [float(k.t_rel_s) for k in trajectory.knots]
        requested = [
            [
                float(k.posture_target.joint_pos_target[j])
                + preload_weight(phase, k.t_rel_s / duration)[0]
                * compression.joint_delta_rad.get(j, 0.0)
                for j in joint_ids
            ]
            for k in trajectory.knots
        ]
        positions, velocities, correction = bounded_joint_curve(
            times,
            requested,
            [joint_limits[j][0] for j in joint_ids],
            [joint_limits[j][1] for j in joint_ids],
        )
        maximum_joint_bound_correction = max(maximum_joint_bound_correction, correction)
        knots = []
        for index, knot in enumerate(trajectory.knots):
            if time.monotonic() - started > deadline_s:
                raise TimeoutError("contact planning deadline")
            q = dict(zip(joint_ids, positions[index].tolist()))
            qdot = dict(zip(joint_ids, velocities[index].tolist()))
            c = knot.centroidal_target
            base = base_pose_for_centroidal_target(
                bundle.morphology,
                physical_model,
                q,
                c.com_pos_world,
                c.body_orientation_world,
                kinematics=solver.kinematics,
            )
            fk = solver.kinematics.forward(
                bundle.morphology, physical_model, q, base, references
            )
            knots.append(
                replace(
                    knot,
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
    return NaiveContactPlan(
        phases,
        {
            "teacher_phase_hash": source_hash,
            "teacher_provenance": bundle.provenance,
            "physical_model_hash": physical_model.stable_hash(),
            "task_spec_hash": stable_hash(task.to_dict()),
            "force_model": "quasistatic_vertical_frictional_support",
            "exact_wrench_tracking_required": False,
            "preload": preload.to_dict(),
            "joint_delta_rad": dict(compression.joint_delta_rad),
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
                    margin_m=0.001,
                    ground_plane_z_m=collision_object.ground_plane_z_m,
                )
                if not _compression_collision_result_accepted(
                    result,
                    selected_contact_scene=selected,
                    collision_margin_m=0.001,
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
