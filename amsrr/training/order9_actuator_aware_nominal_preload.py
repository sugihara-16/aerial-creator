from __future__ import annotations

"""Actuator/leverage-aware nominal contact preload for Order 9.

The high-level contact poses retain surface semantics.  This module computes
the small inward displacement that a position-controlled joint chain needs in
order to sustain a requested contact load after joint-drive and contact
compliance.  It is evaluated once when a reviewed nominal trajectory is
installed; no simulator contact truth is consumed.
"""

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from amsrr.feasibility.articulated_reachability import (
    CentroidalPostureIKSolver,
    base_pose_for_centroidal_target,
    resolve_mesh_backed_anchor_references,
)
from amsrr.robot_model.whole_structure_kinematics import (
    ordered_global_dock_joint_ids,
)
from amsrr.schemas.common import Pose7D
from amsrr.schemas.contact_candidates import ContactCandidateSet
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.policies import InteractionKnot
from amsrr.training.order9_virtual_contact_compression import (
    _global_joint_limits,
)


ORDER9_ACTUATOR_AWARE_NOMINAL_PRELOAD_VERSION = (
    "order9_nominal_preload_v5_additive_patch_moment"
)


@dataclass(frozen=True)
class Order9ActuatorAwareNominalPreloadConfig:
    minimum_inward_lead_m: float = 0.002
    maximum_inward_lead_m: float = 0.040
    inward_lead_quantization_m: float = 0.001
    support_safety_factor: float = 1.25
    maximum_peak_effort_utilization: float = 0.80
    minimum_anchor_force_fraction: float = 0.50
    model_error_margin_m: float = 0.0
    gravity_mps2: float = 9.80665

    def validate(self) -> None:
        for name in (
            "minimum_inward_lead_m",
            "maximum_inward_lead_m",
            "inward_lead_quantization_m",
            "support_safety_factor",
            "maximum_peak_effort_utilization",
            "minimum_anchor_force_fraction",
            "gravity_mps2",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"nominal preload {name} must be positive")
        if (
            not math.isfinite(float(self.model_error_margin_m))
            or float(self.model_error_margin_m) < 0.0
        ):
            raise ValueError("nominal preload model error margin must be non-negative")
        if self.minimum_inward_lead_m > self.maximum_inward_lead_m:
            raise ValueError("nominal preload lead bounds are reversed")
        if not 0.0 < self.maximum_peak_effort_utilization <= 1.0:
            raise ValueError("nominal preload effort utilization must be in (0, 1]")
        if not 0.0 < self.minimum_anchor_force_fraction <= 1.0:
            raise ValueError("nominal preload anchor fraction must be in (0, 1]")


@dataclass(frozen=True)
class Order9ActuatorAwareNominalPreloadSolution:
    inward_lead_m: float
    compliance_predicted_inward_lead_m: float
    model_error_margin_m: float
    required_tangential_support_force_n: float
    target_normal_force_n_by_anchor: tuple[float, ...]
    predicted_inward_lead_m_by_anchor: tuple[float, ...]
    predicted_joint_torque_nm: tuple[float, ...]
    peak_effort_utilization: float
    continuous_effort_utilization: float
    limiting_joint_id: str
    feasible: bool
    rejection_reason: str | None
    target_contact_force_world_n: tuple[tuple[float, float, float], ...] = ()
    commanded_inward_lead_m_by_anchor: tuple[float, ...] = ()
    equilibrium_residual: float | None = None
    torsion_preload_requested_scale: float | None = None
    torsion_preload_applied_scale: float | None = None
    contact_torsion_radii_m: tuple[float, ...] = ()
    version: str = ORDER9_ACTUATOR_AWARE_NOMINAL_PRELOAD_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "inward_lead_m": self.inward_lead_m,
            "compliance_predicted_inward_lead_m": (
                self.compliance_predicted_inward_lead_m
            ),
            "model_error_margin_m": self.model_error_margin_m,
            "required_tangential_support_force_n": (
                self.required_tangential_support_force_n
            ),
            "target_normal_force_n_by_anchor": list(
                self.target_normal_force_n_by_anchor
            ),
            "predicted_inward_lead_m_by_anchor": list(
                self.predicted_inward_lead_m_by_anchor
            ),
            "predicted_joint_torque_nm": list(self.predicted_joint_torque_nm),
            "peak_effort_utilization": self.peak_effort_utilization,
            "continuous_effort_utilization": (
                self.continuous_effort_utilization
            ),
            "limiting_joint_id": self.limiting_joint_id,
            "feasible": self.feasible,
            "rejection_reason": self.rejection_reason,
            "target_contact_force_world_n": [list(f) for f in self.target_contact_force_world_n],
            "commanded_inward_lead_m_by_anchor": list(self.commanded_inward_lead_m_by_anchor),
            "equilibrium_residual": self.equilibrium_residual,
            "torsion_preload_requested_scale": self.torsion_preload_requested_scale,
            "torsion_preload_applied_scale": self.torsion_preload_applied_scale,
            "contact_torsion_radii_m": list(self.contact_torsion_radii_m),
        }


def solve_order9_actuator_aware_nominal_preload_from_jacobian(
    *,
    inward_normal_joint_jacobian_m: np.ndarray,
    joint_ids: Sequence[str],
    joint_stiffness_nm_per_rad: Sequence[float],
    joint_peak_effort_limit_nm: Sequence[float],
    joint_continuous_effort_limit_nm: Sequence[float],
    object_mass_kg: float,
    contact_friction: float | Sequence[float],
    contact_stiffness_n_per_m: float,
    config: Order9ActuatorAwareNominalPreloadConfig | None = None,
    contact_positions_world: np.ndarray | None = None,
    inward_normals_world: np.ndarray | None = None,
    object_com_world: np.ndarray | None = None,
    contact_position_joint_jacobian: np.ndarray | None = None,
    maximum_contact_force_n: Sequence[float] | None = None,
    contact_torsion_radii_m: Sequence[float] | None = None,
) -> Order9ActuatorAwareNominalPreloadSolution:
    """Resolve balanced contact forces and the corresponding servo lead."""

    resolved = config or Order9ActuatorAwareNominalPreloadConfig()
    resolved.validate()
    jacobian = np.asarray(inward_normal_joint_jacobian_m, dtype=float)
    if jacobian.ndim != 2 or jacobian.shape[0] < 2 or jacobian.shape[1] < 1:
        raise ValueError("nominal preload Jacobian must be [anchor>=2, joint]")
    anchor_count, joint_count = jacobian.shape
    ids = tuple(str(value) for value in joint_ids)
    stiffness = np.asarray(joint_stiffness_nm_per_rad, dtype=float)
    peak = np.asarray(joint_peak_effort_limit_nm, dtype=float)
    continuous = np.asarray(joint_continuous_effort_limit_nm, dtype=float)
    if (
        len(ids) != joint_count
        or stiffness.shape != (joint_count,)
        or peak.shape != (joint_count,)
        or continuous.shape != (joint_count,)
    ):
        raise ValueError("nominal preload joint arrays differ")
    if not np.isfinite(jacobian).all() or not np.isfinite(stiffness).all():
        raise ValueError("nominal preload model contains non-finite values")
    if (
        np.any(stiffness <= 0.0)
        or np.any(peak <= 0.0)
        or np.any(continuous <= 0.0)
        or np.any(continuous > peak)
    ):
        raise ValueError("nominal preload actuator limits are invalid")
    mass = float(object_mass_kg)
    contact_stiffness = float(contact_stiffness_n_per_m)
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("nominal preload object mass must be positive")
    if not math.isfinite(contact_stiffness) or contact_stiffness <= 0.0:
        raise ValueError("nominal preload contact stiffness must be positive")
    friction = np.asarray(
        [float(contact_friction)] * anchor_count
        if isinstance(contact_friction, (int, float))
        else tuple(float(value) for value in contact_friction),
        dtype=float,
    )
    if friction.shape != (anchor_count,) or not np.isfinite(friction).all() or np.any(
        friction <= 0.0
    ):
        raise ValueError("nominal preload contact friction is invalid")

    required_support = (
        mass * float(resolved.gravity_mps2) * float(resolved.support_safety_factor)
    )
    contact_forces = None
    equilibrium_residual = None
    torsion_requested_scale = torsion_applied_scale = None
    baseline_leads = None
    if anchor_count > 2:
        if any(x is None for x in (contact_positions_world, inward_normals_world,
                                  object_com_world, contact_position_joint_jacobian)):
            raise ValueError("multi-contact preload requires contact geometry and COM")
        contact_forces, equilibrium_residual = solve_contact_force_equilibrium(
            positions=contact_positions_world, inward_normals=inward_normals_world,
            center_of_mass=object_com_world, friction=friction / resolved.support_safety_factor,
            required_force=np.array([0., 0., mass * resolved.gravity_mps2]),
            position_jacobian=contact_position_joint_jacobian,
            effort_limits=peak, maximum_utilization=resolved.maximum_peak_effort_utilization,
            minimum_normal_force=resolved.minimum_anchor_force_fraction * required_support / friction.sum(),
            maximum_contact_force_n=maximum_contact_force_n,
        )
        forces = np.einsum('ai,ai->a', contact_forces, inward_normals_world)
        torque = np.einsum('aij,ai->j', contact_position_joint_jacobian, contact_forces)
    else:
        internal_wrench = None
        if contact_positions_world is not None or inward_normals_world is not None:
            positions = np.asarray(contact_positions_world, dtype=float)
            normals = np.asarray(inward_normals_world, dtype=float)
            if (positions.shape != (anchor_count, 3)
                    or normals.shape != (anchor_count, 3)
                    or not np.isfinite(positions).all()
                    or not np.isfinite(normals).all()
                    or not np.allclose(np.linalg.norm(normals, axis=1), 1., atol=1e-5, rtol=0)):
                raise ValueError("normal preload requires finite contact positions and unit normals")
            # A squeeze is an internal load: opposing normal forces must not
            # introduce a net force or couple on the object. Friction still
            # supplies the approximate support capacity used below.
            internal_wrench = np.vstack((normals.T,
                np.cross(positions - positions.mean(axis=0), normals).T))
        forces = _minimum_peak_utilization_force_allocation(
            jacobian=jacobian, friction=friction, peak_effort_limit_nm=peak,
            required_support_force_n=required_support,
            minimum_anchor_force_fraction=float(resolved.minimum_anchor_force_fraction),
            internal_wrench=internal_wrench)
        # Existing two-contact execution uses one common gravity preload.
        # Additional rotational reserve must not reduce either existing command.
        baseline_prediction = np.maximum(jacobian @ ((jacobian.T @ forces) / stiffness)
            + forces / contact_stiffness, 0.)
        quantum = float(resolved.inward_lead_quantization_m)
        baseline_uniform = quantum * math.ceil((max(resolved.minimum_inward_lead_m,
            float(baseline_prediction.max()) + resolved.model_error_margin_m) - 1e-12) / quantum)
        baseline_leads = np.full(anchor_count, baseline_uniform)
        if contact_torsion_radii_m is not None:
            from amsrr.training.soft_contact_preload import patch_normal_forces, bounded_normal_reserve
            if object_com_world is None or internal_wrench is None:
                raise ValueError("finite patch preload requires contact geometry and estimated COM")
            requested, _ = patch_normal_forces(
                positions=positions, normals=normals, center=object_com_world,
                required_force=[0., 0., mass * resolved.gravity_mps2],
                friction=friction / resolved.support_safety_factor,
                radii=contact_torsion_radii_m, normal_jacobian=jacobian,
                effort_limits=peak, minimum_normal_force=resolved.minimum_anchor_force_fraction * required_support / friction.sum())
            forces, torsion_requested_scale, torsion_applied_scale = bounded_normal_reserve(
                forces=forces, requested=requested, jacobian=jacobian, stiffness=stiffness,
                peak=peak, contact_stiffness=contact_stiffness,
                maximum_utilization=resolved.maximum_peak_effort_utilization,
                maximum_lead=resolved.maximum_inward_lead_m,
                quantum=resolved.inward_lead_quantization_m, margin=resolved.model_error_margin_m)
        torque = jacobian.T @ forces
    peak_ratio = np.abs(torque) / peak
    continuous_ratio = np.abs(torque) / continuous
    limiting_index = int(np.argmax(peak_ratio))
    predicted_lead = jacobian @ (torque / stiffness) + forces / contact_stiffness
    predicted_lead = np.maximum(predicted_lead, 0.0)
    compliance_predicted_lead = float(np.max(predicted_lead))
    requested = max(
        float(resolved.minimum_inward_lead_m),
        compliance_predicted_lead + float(resolved.model_error_margin_m),
    )
    quantum = float(resolved.inward_lead_quantization_m)
    requested = quantum * math.ceil((requested - 1.0e-12) / quantum)
    commanded_leads = quantum * np.ceil((np.maximum(resolved.minimum_inward_lead_m,
        predicted_lead + resolved.model_error_margin_m) - 1e-12) / quantum)
    if baseline_leads is not None:
        commanded_leads = np.maximum(commanded_leads, baseline_leads)
    peak_utilization = float(np.max(peak_ratio))
    rejection_reason = None
    if peak_utilization > float(resolved.maximum_peak_effort_utilization) + 1.0e-9:
        rejection_reason = "peak_effort_utilization"
    elif requested > float(resolved.maximum_inward_lead_m) + 1.0e-12:
        rejection_reason = "nominal_inward_lead"
    return Order9ActuatorAwareNominalPreloadSolution(
        inward_lead_m=min(requested, float(resolved.maximum_inward_lead_m)),
        compliance_predicted_inward_lead_m=compliance_predicted_lead,
        model_error_margin_m=float(resolved.model_error_margin_m),
        required_tangential_support_force_n=required_support,
        target_normal_force_n_by_anchor=tuple(float(value) for value in forces),
        predicted_inward_lead_m_by_anchor=tuple(
            float(value) for value in predicted_lead
        ),
        predicted_joint_torque_nm=tuple(float(value) for value in torque),
        peak_effort_utilization=peak_utilization,
        continuous_effort_utilization=float(np.max(continuous_ratio)),
        limiting_joint_id=ids[limiting_index],
        feasible=rejection_reason is None,
        rejection_reason=rejection_reason,
        target_contact_force_world_n=() if contact_forces is None else tuple(tuple(float(x) for x in f) for f in contact_forces),
        commanded_inward_lead_m_by_anchor=tuple(float(x) for x in commanded_leads),
        equilibrium_residual=equilibrium_residual,
        torsion_preload_requested_scale=torsion_requested_scale,
        torsion_preload_applied_scale=torsion_applied_scale,
        contact_torsion_radii_m=() if contact_torsion_radii_m is None else tuple(float(x) for x in contact_torsion_radii_m),
    )


def solve_order9_actuator_aware_nominal_preload(
    *,
    morphology: MorphologyGraph,
    physical_model: PhysicalModel,
    contact_knot: InteractionKnot,
    candidate_set: ContactCandidateSet,
    object_mass_kg: float,
    contact_friction: float,
    contact_stiffness_n_per_m: float,
    config: Order9ActuatorAwareNominalPreloadConfig | None = None,
    object_com_world: np.ndarray | None = None,
) -> Order9ActuatorAwareNominalPreloadSolution:
    """Build the contact Jacobian at the reviewed grasp and resolve preload."""

    posture = contact_knot.posture_target
    centroidal = contact_knot.centroidal_target
    if (
        posture is None
        or posture.joint_pos_target is None
        or centroidal is None
        or centroidal.com_pos_world is None
        or centroidal.body_orientation_world is None
    ):
        raise ValueError("nominal preload requires a resolved contact posture")
    assignments = tuple(contact_knot.contact_assignments)
    if len(assignments) < 2:
        raise ValueError("nominal preload requires at least two contacts")
    ordered_ids = ordered_global_dock_joint_ids(morphology, physical_model)
    q = {
        joint_id: float(posture.joint_pos_target[joint_id])
        for joint_id in ordered_ids
    }
    references = resolve_mesh_backed_anchor_references(
        morphology,
        physical_model,
        tuple(int(value.anchor_id) for value in assignments),
    )
    solver = CentroidalPostureIKSolver(physical_model)
    centroidal_pose: Pose7D = (
        *tuple(float(value) for value in centroidal.com_pos_world),
        *tuple(float(value) for value in centroidal.body_orientation_world),
    )
    base_pose = base_pose_for_centroidal_target(
        morphology,
        physical_model,
        q,
        centroidal_pose[:3],
        centroidal_pose[3:7],
        kinematics=solver.kinematics,
    )
    nominal = solver.kinematics.forward(
        morphology, physical_model, q, base_pose, references
    )
    jacobians = solver._fixed_centroidal_anchor_jacobians(
        morphology=morphology,
        centroidal_pose_world=centroidal_pose,
        q=q,
        limits=_global_joint_limits(ordered_ids, physical_model),
        references=references,
        nominal=nominal.anchor_poses_world,
    )
    candidate_by_id = {
        int(candidate.candidate_id): candidate
        for candidate in candidate_set.candidates
    }
    rows: list[np.ndarray] = []
    for assignment in assignments:
        candidate = candidate_by_id.get(int(assignment.candidate_id))
        if candidate is None:
            raise ValueError("nominal preload contact candidate is missing")
        normal = np.asarray(candidate.normal_world, dtype=float)
        norm = float(np.linalg.norm(normal))
        if not math.isfinite(norm) or norm <= 1.0e-9:
            raise ValueError("nominal preload contact normal is invalid")
        rows.append(
            (-normal / norm)
            @ np.asarray(jacobians[int(assignment.anchor_id)][:3], dtype=float)
        )
    stiffness, peak, continuous = _global_joint_actuator_values(
        ordered_ids, physical_model
    )
    anchors = {a.anchor_id:a for a in morphology.robot_anchors}
    from amsrr.training.soft_contact_preload import nominal_patch_radius
    radii = [nominal_patch_radius(physical_model.urdf_path,
        anchors[a.anchor_id].link_id, tuple(anchors[a.anchor_id].local_pose),
        physical_model.metadata['urdf_hash']) for a in assignments] if len(assignments) == 2 and object_com_world is not None else None
    return solve_order9_actuator_aware_nominal_preload_from_jacobian(
        inward_normal_joint_jacobian_m=np.stack(rows),
        joint_ids=ordered_ids,
        joint_stiffness_nm_per_rad=stiffness,
        joint_peak_effort_limit_nm=peak,
        joint_continuous_effort_limit_nm=continuous,
        object_mass_kg=object_mass_kg,
        contact_friction=contact_friction,
        contact_stiffness_n_per_m=contact_stiffness_n_per_m,
        config=config,
        contact_positions_world=np.array([candidate_by_id[int(a.candidate_id)].contact_pose_world[:3] for a in assignments]),
        inward_normals_world=np.array([-np.asarray(candidate_by_id[int(a.candidate_id)].normal_world) /
            np.linalg.norm(candidate_by_id[int(a.candidate_id)].normal_world) for a in assignments]),
        object_com_world=object_com_world,
        contact_torsion_radii_m=radii,
        contact_position_joint_jacobian=np.stack([jacobians[int(a.anchor_id)][:3] for a in assignments]),
        maximum_contact_force_n=[float(anchors[a.anchor_id].capability['max_force_n']) for a in assignments],
    )


def solve_contact_force_equilibrium(*, positions, inward_normals, center_of_mass,
                                    friction, required_force, position_jacobian,
                                    effort_limits, maximum_utilization,
                                    minimum_normal_force, maximum_contact_force_n=None):
    """Nominal force on object: 6D balance, inscribed friction pyramid, effort.

    Solved only at planning time. No measured contact force is used. The
    inscribed square (|t1|+|t2| <= mu*fn) never overestimates Coulomb friction.
    """
    from scipy.optimize import linprog
    p, n, center, mu, force, jac, limits = [np.asarray(x, dtype=float) for x in
        (positions, inward_normals, center_of_mass, friction, required_force,
         position_jacobian, effort_limits)]
    count = len(p)
    if (p.shape != (count, 3) or n.shape != p.shape or center.shape != (3,)
            or mu.shape != (count,) or force.shape != (3,) or jac.shape != (count, 3, len(limits))
            or not all(np.isfinite(x).all() for x in (p, n, center, mu, force, jac, limits))
            or (mu <= 0).any() or (limits <= 0).any()
            or not np.allclose(np.linalg.norm(n, axis=1), 1., atol=1e-6)):
        raise ValueError("invalid multi-contact equilibrium geometry")
    width = 3 * count + 1
    equilibrium = np.zeros((6, width))
    inequality, upper = [], []
    for i, (point, normal) in enumerate(zip(p, n)):
        block = slice(3*i, 3*i+3)
        equilibrium[:3, block] = np.eye(3)
        x, y, z = point - center
        equilibrium[3:, block] = [[0, -z, y], [z, 0, -x], [-y, x, 0]]
        tangent = np.cross(normal, np.eye(3)[np.argmin(abs(normal))])
        tangent /= np.linalg.norm(tangent)
        other = np.cross(normal, tangent)
        for a, b in ((1,1), (1,-1), (-1,1), (-1,-1)):
            row = np.zeros(width); row[block] = a*tangent + b*other - mu[i]*normal
            inequality.append(row); upper.append(0.)
        row = np.zeros(width); row[block] = -normal
        inequality.append(row); upper.append(-minimum_normal_force)
        if maximum_contact_force_n is not None:
            limit = float(maximum_contact_force_n[i])
            if not math.isfinite(limit) or limit <= 0:
                raise ValueError("invalid contact force capability")
            # Inscribed octahedron bounds the Euclidean force magnitude.
            for sx in (-1.,1.):
                for sy in (-1.,1.):
                    for sz in (-1.,1.):
                        row=np.zeros(width); row[block]=[sx,sy,sz]
                        inequality.append(row); upper.append(limit)
    for j, limit in enumerate(limits):
        for sign in (-1., 1.):
            row = np.r_[sign * jac[:, :, j].ravel(), -limit]
            inequality.append(row); upper.append(0.)
    objective = np.r_[1e-7*n.ravel(), 1.]
    rhs = np.r_[force, np.zeros(3)]
    result = linprog(objective, A_ub=np.asarray(inequality), b_ub=np.asarray(upper),
        A_eq=equilibrium, b_eq=rhs,
        bounds=[(None, None)]*(width-1)+[(0., maximum_utilization)], method='highs')
    if not result.success:
        raise ValueError("contact preload infeasible: multi-contact force/moment equilibrium")
    residual = float(np.max(np.abs(equilibrium @ result.x - rhs)))
    if residual > 1e-6 or np.max(np.asarray(inequality) @ result.x - upper) > 1e-6:
        raise ValueError("contact preload infeasible: equilibrium solver residual")
    return result.x[:-1].reshape(count, 3), residual


def _minimum_peak_utilization_force_allocation(
    *,
    jacobian: np.ndarray,
    friction: np.ndarray,
    peak_effort_limit_nm: np.ndarray,
    required_support_force_n: float,
    minimum_anchor_force_fraction: float,
    internal_wrench: np.ndarray | None = None,
) -> np.ndarray:
    from scipy.optimize import linprog

    anchor_count, joint_count = jacobian.shape
    if internal_wrench is not None:
        internal_wrench = np.asarray(internal_wrench, dtype=float)
        if (internal_wrench.ndim != 2 or internal_wrench.shape[1] != anchor_count
                or not np.isfinite(internal_wrench).all()):
            raise ValueError("invalid internal squeeze wrench matrix")
    # Variables are per-anchor normal force followed by peak utilization.
    objective = np.zeros(anchor_count + 1, dtype=float)
    objective[-1] = 1.0
    objective[:anchor_count] = 1.0e-9
    rows: list[np.ndarray] = []
    upper: list[float] = []
    for joint_index in range(joint_count):
        positive = np.zeros(anchor_count + 1, dtype=float)
        positive[:anchor_count] = jacobian[:, joint_index]
        positive[-1] = -peak_effort_limit_nm[joint_index]
        rows.extend((positive, -positive))
        # Negating the complete row would reverse the utilization coefficient;
        # build the lower-torque inequality explicitly instead.
        rows[-1][-1] = -peak_effort_limit_nm[joint_index]
        upper.extend((0.0, 0.0))
    support = np.zeros(anchor_count + 1, dtype=float)
    support[:anchor_count] = -friction
    rows.append(support)
    upper.append(-float(required_support_force_n))
    equal_force = float(required_support_force_n) / float(np.sum(friction))
    minimum_force = float(minimum_anchor_force_fraction) * equal_force
    result = linprog(
        objective,
        A_ub=np.stack(rows),
        b_ub=np.asarray(upper, dtype=float),
        A_eq=None if internal_wrench is None else np.c_[internal_wrench, np.zeros(len(internal_wrench))],
        b_eq=None if internal_wrench is None else np.zeros(len(internal_wrench)),
        bounds=[(minimum_force, None)] * anchor_count + [(0.0, None)],
        method="highs",
    )
    if not result.success or result.x is None or not np.isfinite(result.x).all():
        raise ValueError(
            "nominal preload force allocation is infeasible: "
            f"{result.message}"
        )
    return np.asarray(result.x[:anchor_count], dtype=float)


def _global_joint_actuator_values(
    ordered_ids: Sequence[str], physical_model: PhysicalModel
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assignments = physical_model.metadata.get("joint_actuator_assignments")
    specs = physical_model.metadata.get("joint_actuator_specs")
    if not isinstance(assignments, dict) or not isinstance(specs, dict):
        raise ValueError("nominal preload lacks actuator provenance")
    stiffness: list[float] = []
    peak: list[float] = []
    continuous: list[float] = []
    for global_id in ordered_ids:
        local_id = str(global_id).split(":", 1)[-1]
        role = assignments.get(local_id)
        spec = specs.get(role) if isinstance(role, str) else None
        drive = spec.get("simulation_drive") if isinstance(spec, dict) else None
        if not isinstance(spec, dict) or not isinstance(drive, dict):
            raise ValueError(f"nominal preload lacks actuator data for {global_id}")
        stiffness.append(float(drive["stiffness"]))
        peak.append(float(spec["peak_torque_nm"]))
        continuous.append(float(spec["continuous_torque_limit_nm"]))
    return np.asarray(stiffness), np.asarray(peak), np.asarray(continuous)


__all__ = [
    "ORDER9_ACTUATOR_AWARE_NOMINAL_PRELOAD_VERSION",
    "Order9ActuatorAwareNominalPreloadConfig",
    "Order9ActuatorAwareNominalPreloadSolution",
    "solve_order9_actuator_aware_nominal_preload",
    "solve_order9_actuator_aware_nominal_preload_from_jacobian",
]
