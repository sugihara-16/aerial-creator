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
    "order9_actuator_leverage_contact_compliance_nominal_preload_v2_calibrated_margin"
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
    forces = _minimum_peak_utilization_force_allocation(
        jacobian=jacobian,
        friction=friction,
        peak_effort_limit_nm=peak,
        required_support_force_n=required_support,
        minimum_anchor_force_fraction=float(
            resolved.minimum_anchor_force_fraction
        ),
    )
    torque = jacobian.T @ forces
    peak_ratio = np.abs(torque) / peak
    continuous_ratio = np.abs(torque) / continuous
    limiting_index = int(np.argmax(peak_ratio))
    compliance = (
        jacobian @ np.diag(1.0 / stiffness) @ jacobian.T
    )
    predicted_lead = compliance @ forces + forces / contact_stiffness
    predicted_lead = np.maximum(predicted_lead, 0.0)
    compliance_predicted_lead = float(np.max(predicted_lead))
    requested = max(
        float(resolved.minimum_inward_lead_m),
        compliance_predicted_lead + float(resolved.model_error_margin_m),
    )
    quantum = float(resolved.inward_lead_quantization_m)
    requested = quantum * math.ceil((requested - 1.0e-12) / quantum)
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
    )


def _minimum_peak_utilization_force_allocation(
    *,
    jacobian: np.ndarray,
    friction: np.ndarray,
    peak_effort_limit_nm: np.ndarray,
    required_support_force_n: float,
    minimum_anchor_force_fraction: float,
) -> np.ndarray:
    from scipy.optimize import linprog

    anchor_count, joint_count = jacobian.shape
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
