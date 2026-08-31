from __future__ import annotations

"""R1-only support-clearance contract and deterministic teacher selection.

The acceptance gate and the generation target are deliberately distinct.  A
trajectory is eligible for R1 materialization only when the complete eight
phase path was part of candidate selection under the 30 mm planning target.
Merely copying a grasp posture into the place phase and checking it afterwards
does not create the required generation certificate.
"""

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.policies import ContactWrenchTrajectory
from amsrr.schemas.contact_candidates import ContactCandidateSet
from amsrr.schemas.runtime import RuntimeObservation
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.order9_c3_nominal_trajectory import (
    Order9C3NominalTrajectory,
    Order9C3NominalWindow,
    _flatten_nominal_windows,
    _phase_trajectories_from_nominal_windows,
)
from amsrr.training.order9_r1_complete_task_audit import (
    ORDER9_R1_COMPLETE_TASK_PHASES,
)
from amsrr.training.order9_r1_complete_task_geometry_v12 import (
    audit_order9_r1_complete_task_geometry_v12,
)
from amsrr.training.order9_r1_nominal_geometry_v11 import (
    Order9R1NominalGeometryV11Contract,
)
from amsrr.training.order9_r1_nominal_retreat import (
    materialize_order9_r1_complete_task_phases,
)
from amsrr.utils.hashing import hash_file, stable_hash

ORDER9_R1_SUPPORT_CLEARANCE_TEACHER_V12_VERSION = (
    "order9_r1_support_clearance_teacher_v12"
)
ORDER9_R1_SUPPORT_CLEARANCE_CERTIFICATE_V1 = (
    "order9_r1_support_clearance_generation_certificate_v1"
)
ORDER9_R1_SUPPORT_CLEARANCE_TEACHER_V12_RELATIVE = Path(
    "configs/training/order9_r1_support_clearance_teacher_v12.json"
)
E_R1_SUPPORT_CLEARANCE_BUFFER = "E_R1_SUPPORT_CLEARANCE_BUFFER"
E_R1_SUPPORT_CLEARANCE_GENERATION_PROOF = "E_R1_SUPPORT_CLEARANCE_GENERATION_PROOF"
E_R1_UNCERTIFIED_RIGID_POSE_COPY = "E_R1_UNCERTIFIED_RIGID_POSE_COPY"


@dataclass(frozen=True)
class Order9R1SupportClearanceTeacherV12Contract:
    config_path: Path
    config_sha256: str
    acceptance_clearance_m: float
    planning_clearance_m: float
    minimum_planning_buffer_m: float
    solver_discretization_reserve_m: float
    maximum_vertical_contact_frame_shift_m: float
    vertical_shift_quantization_m: float
    required_phases: tuple[str, ...]
    maximum_alternate_contact_group_count: int
    require_generation_time_clearance_constraint: bool
    reject_uncertified_rigid_pose_copy: bool
    isaac_allowed_during_generation: bool
    controller_layers_allowed_during_generation: bool
    formal_teacher_collection_authorized: bool

    @property
    def actual_planning_buffer_m(self) -> float:
        return self.planning_clearance_m - self.acceptance_clearance_m

    @property
    def solver_requested_clearance_m(self) -> float:
        return self.planning_clearance_m + self.solver_discretization_reserve_m


def load_order9_r1_support_clearance_teacher_v12_contract(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> Order9R1SupportClearanceTeacherV12Contract:
    repository = Path(repository_root).resolve()
    source = Path(path)
    if not source.is_absolute():
        source = repository / source
    source = source.resolve()
    if repository not in source.parents or not source.is_file():
        raise SchemaValidationError("R1 support-clearance contract path is invalid")
    payload = json.loads(source.read_text(encoding="utf-8"))
    expected_fields = {
        "contract_version",
        "acceptance_clearance_m",
        "planning_clearance_m",
        "minimum_planning_buffer_m",
        "solver_discretization_reserve_m",
        "maximum_vertical_contact_frame_shift_m",
        "vertical_shift_quantization_m",
        "required_phases",
        "maximum_alternate_contact_group_count",
        "require_generation_time_clearance_constraint",
        "reject_uncertified_rigid_pose_copy",
        "isaac_allowed_during_generation",
        "controller_layers_allowed_during_generation",
        "formal_teacher_collection_authorized",
    }
    if (
        set(payload) != expected_fields
        or payload.get("contract_version")
        != ORDER9_R1_SUPPORT_CLEARANCE_TEACHER_V12_VERSION
    ):
        raise SchemaValidationError("R1 support-clearance contract fields changed")
    phases = tuple(str(value) for value in payload["required_phases"])
    values = tuple(
        float(payload[name])
        for name in (
            "acceptance_clearance_m",
            "planning_clearance_m",
            "minimum_planning_buffer_m",
            "solver_discretization_reserve_m",
            "maximum_vertical_contact_frame_shift_m",
            "vertical_shift_quantization_m",
        )
    )
    booleans = tuple(
        payload[name]
        for name in (
            "require_generation_time_clearance_constraint",
            "reject_uncertified_rigid_pose_copy",
            "isaac_allowed_during_generation",
            "controller_layers_allowed_during_generation",
            "formal_teacher_collection_authorized",
        )
    )
    alternate_count = payload["maximum_alternate_contact_group_count"]
    if (
        phases != ORDER9_R1_COMPLETE_TASK_PHASES
        or any(not math.isfinite(value) or value <= 0.0 for value in values)
        or not math.isclose(values[0], 0.0195, rel_tol=0.0, abs_tol=1.0e-12)
        or not math.isclose(values[1], 0.0300, rel_tol=0.0, abs_tol=1.0e-12)
        or values[1] - values[0] + 1.0e-12 < values[2]
        or values[3] >= values[2]
        or values[4] > 0.030 + 1.0e-12
        or values[5] > values[4]
        or isinstance(alternate_count, bool)
        or not 1 <= int(alternate_count) <= 16
        or not all(isinstance(value, bool) for value in booleans)
        or booleans != (True, True, False, False, False)
    ):
        raise SchemaValidationError(
            f"{E_R1_SUPPORT_CLEARANCE_BUFFER}: invalid 19.5/30 mm contract"
        )
    return Order9R1SupportClearanceTeacherV12Contract(
        config_path=source,
        config_sha256=hash_file(source),
        acceptance_clearance_m=values[0],
        planning_clearance_m=values[1],
        minimum_planning_buffer_m=values[2],
        solver_discretization_reserve_m=values[3],
        maximum_vertical_contact_frame_shift_m=values[4],
        vertical_shift_quantization_m=values[5],
        required_phases=phases,
        maximum_alternate_contact_group_count=int(alternate_count),
        require_generation_time_clearance_constraint=booleans[0],
        reject_uncertified_rigid_pose_copy=booleans[1],
        isaac_allowed_during_generation=booleans[2],
        controller_layers_allowed_during_generation=booleans[3],
        formal_teacher_collection_authorized=booleans[4],
    )


def order9_r1_vertical_clearance_shift_candidates(
    audit: Mapping[str, Any],
    *,
    clearance_contract: Order9R1SupportClearanceTeacherV12Contract,
) -> tuple[float, ...]:
    """Return the smallest required 5 mm shift first, then bounded fallbacks."""

    measured = float(
        audit.get("minimum_robot_support_clearance_lower_bound_m", -math.inf)
    )
    quantum = clearance_contract.vertical_shift_quantization_m
    maximum = clearance_contract.maximum_vertical_contact_frame_shift_m
    if math.isfinite(measured):
        deficit = max(
            0.0,
            clearance_contract.planning_clearance_m
            + clearance_contract.solver_discretization_reserve_m
            - measured,
        )
        first_index = max(1, int(math.ceil((deficit - 1.0e-12) / quantum)))
    else:
        first_index = 1
    maximum_index = int(math.floor((maximum + 1.0e-12) / quantum))
    return tuple(index * quantum for index in range(first_index, maximum_index + 1))


def derive_order9_r1_vertical_clearance_nominal(
    nominal: Order9C3NominalTrajectory,
    *,
    vertical_shift_m: float,
) -> Order9C3NominalTrajectory:
    """Move the contact frame vertically without changing the solved Dock q.

    The shift is introduced continuously during contact acquisition.  The
    resulting contact endpoint is an exact rigid translation of the original
    deterministic IK solution, so the same q remains an IK solution.  Lift,
    transport and place then inherit the shifted robot/object relative pose;
    release reverses the shifted contact path back to its unshifted endpoint.
    """

    shift = float(vertical_shift_m)
    if not math.isfinite(shift) or not 0.0 < shift <= 0.030 + 1.0e-12:
        raise ValueError("R1 vertical contact-frame shift must be in (0, 30 mm]")
    contact_windows = tuple(
        value for value in nominal.windows if value.phase == "contact_acquisition"
    )
    if not contact_windows:
        raise SchemaValidationError("R1 clearance shift lacks contact acquisition")
    phase_start = float(contact_windows[0].global_start_time_s)
    phase_end = float(
        contact_windows[-1].global_start_time_s
        + contact_windows[-1].plan.trajectory.horizon_s
    )
    phase_span = max(phase_end - phase_start, 1.0e-12)

    windows = []
    for window in nominal.windows:
        if window.phase != "contact_acquisition":
            windows.append(window)
            continue
        window_start_fraction = min(
            1.0,
            max(0.0, (float(window.global_start_time_s) - phase_start) / phase_span),
        )
        window_end_fraction = min(
            1.0,
            max(
                0.0,
                (
                    float(window.global_start_time_s)
                    + float(window.plan.trajectory.horizon_s)
                    - phase_start
                )
                / phase_span,
            ),
        )
        plan = _shift_teacher_plan(
            window.plan,
            shift_m=shift,
            start_fraction=window_start_fraction,
            end_fraction=window_end_fraction,
        )
        windows.append(replace(window, plan=plan))

    selection = nominal.selection_bundle
    selected_candidates = _selected_group_candidate_ids(selection)
    candidates = _shift_contact_candidate_set(
        selection.contact_candidate_set,
        selected_candidate_ids=selected_candidates,
        shift_m=shift,
    )
    selection_plan = _shift_teacher_plan(
        selection.trajectory_plan,
        shift_m=shift,
        start_fraction=0.0,
        end_fraction=1.0,
    )
    shifted_selection = replace(
        selection,
        contact_candidate_set=candidates,
        trajectory_plan=selection_plan,
    )
    shifted_windows = tuple(windows)
    final_observation = _shift_final_robot_observation(
        nominal.final_observation,
        shift_m=shift,
    )
    return replace(
        nominal,
        selection_bundle=shifted_selection,
        windows=shifted_windows,
        timeline=_flatten_nominal_windows(shifted_windows),
        final_observation=final_observation,
    )


def _shift_teacher_plan(
    plan, *, shift_m: float, start_fraction: float, end_fraction: float
):
    raw = _shift_contact_trajectory(
        plan.raw_trajectory,
        shift_m=shift_m,
        start_fraction=start_fraction,
        end_fraction=end_fraction,
    )
    resolved = _shift_contact_trajectory(
        plan.trajectory,
        shift_m=shift_m,
        start_fraction=start_fraction,
        end_fraction=end_fraction,
    )
    resolution = plan.posture_resolution
    evidence = replace(
        resolution.evidence,
        resolver_version=(
            resolution.evidence.resolver_version + "+r1_vertical_contact_frame_shift_v1"
        ),
        raw_trajectory_hash=stable_hash(raw.to_dict()),
        resolved_trajectory_hash=stable_hash(resolved.to_dict()),
    )
    shifted_resolution = replace(
        resolution,
        raw_trajectory=raw,
        trajectory=resolved,
        evidence=evidence,
    )
    ik = plan.ik_solution
    final_shift = shift_m * end_fraction
    shifted_ik = replace(
        ik,
        base_pose_world=_shift_pose_z(ik.base_pose_world, final_shift),
        centroidal_pose_world=_shift_pose_z(ik.centroidal_pose_world, final_shift),
        anchor_poses_world={
            anchor_id: _shift_pose_z(pose, final_shift)
            for anchor_id, pose in ik.anchor_poses_world.items()
        },
        solver_version=ik.solver_version + "+rigid_translation_preserved_v1",
    )
    configuration_plan = plan.configuration_space_plan
    if configuration_plan is not None:
        states = []
        denominator = max(len(configuration_plan.states) - 1, 1)
        for index, state in enumerate(configuration_plan.states):
            fraction = start_fraction + (
                (end_fraction - start_fraction) * index / denominator
            )
            states.append(
                replace(
                    state,
                    base_pose_world=_shift_pose_z(
                        state.base_pose_world,
                        shift_m * fraction,
                    ),
                )
            )
        configuration_plan = replace(configuration_plan, states=tuple(states))
    return replace(
        plan,
        raw_trajectory=raw,
        trajectory=resolved,
        ik_solution=shifted_ik,
        posture_resolution=shifted_resolution,
        configuration_space_plan=configuration_plan,
        teacher_version=plan.teacher_version + "+r1_clearance_shift_v1",
    )


def _shift_contact_trajectory(
    trajectory: ContactWrenchTrajectory,
    *,
    shift_m: float,
    start_fraction: float,
    end_fraction: float,
) -> ContactWrenchTrajectory:
    knots = []
    horizon = max(float(trajectory.horizon_s), 1.0e-12)
    for knot in trajectory.knots:
        local = min(1.0, max(0.0, float(knot.t_rel_s) / horizon))
        smooth = local * local * (3.0 - 2.0 * local)
        fraction = start_fraction + (end_fraction - start_fraction) * smooth
        payload = knot.to_dict()
        centroidal = payload.get("centroidal_target")
        if isinstance(centroidal, dict) and centroidal.get("com_pos_world") is not None:
            centroidal["com_pos_world"][2] += shift_m * fraction
        posture = payload.get("posture_target")
        if isinstance(posture, dict):
            targets = posture.get("free_anchor_pose_targets")
            if isinstance(targets, dict):
                for pose in targets.values():
                    pose[2] += shift_m * fraction
        knots.append(type(knot).from_dict(payload))
    value = ContactWrenchTrajectory(
        horizon_s=float(trajectory.horizon_s),
        dt_s=float(trajectory.dt_s),
        knots=knots,
        derived_mode_label=(
            f"{trajectory.derived_mode_label or 'unspecified'}:"
            "r1_vertical_contact_frame_shift_v1"
        ),
        contract_version=trajectory.contract_version,
    )
    value.validate()
    return value


def _selected_group_candidate_ids(selection) -> frozenset[int]:
    group_id = selection.trajectory_plan.candidate_group_id
    proposal = next(
        (
            value
            for value in selection.contact_candidate_set.group_proposals
            if value.group_id == group_id
        ),
        None,
    )
    if proposal is None or not proposal.candidate_ids:
        raise SchemaValidationError("R1 clearance shift lacks selected contact group")
    return frozenset(int(value) for value in proposal.candidate_ids)


def _shift_contact_candidate_set(
    candidate_set: ContactCandidateSet,
    *,
    selected_candidate_ids: frozenset[int],
    shift_m: float,
) -> ContactCandidateSet:
    payload = candidate_set.to_dict()
    shifted = 0
    for candidate in payload["candidates"]:
        if int(candidate["candidate_id"]) not in selected_candidate_ids:
            continue
        candidate["contact_pose_world"][2] += shift_m
        candidate["contact_frame_world"][2] += shift_m
        scores = dict(candidate.get("candidate_scores", {}))
        scores["r1_vertical_clearance_shift_m"] = shift_m
        candidate["candidate_scores"] = scores
        shifted += 1
    if shifted != len(selected_candidate_ids):
        raise SchemaValidationError("R1 clearance shift candidate identity differs")
    value = ContactCandidateSet.from_dict(payload)
    value.validate()
    return value


def _shift_final_robot_observation(
    observation: RuntimeObservation,
    *,
    shift_m: float,
) -> RuntimeObservation:
    payload = observation.to_dict()
    for state in payload.get("module_states", []):
        state["pose_world"][2] += shift_m
    value = RuntimeObservation.from_dict(payload)
    value.validate()
    return value


def _shift_pose_z(pose, shift_m: float):
    return (float(pose[0]), float(pose[1]), float(pose[2]) + shift_m, *pose[3:])


def materialize_order9_r1_v12_candidate_phases(
    nominal: Order9C3NominalTrajectory,
    *,
    task_spec: TaskSpec,
    lift_clearance_m: float = 0.30,
    retreat_offset_m: float = 0.10,
    phase_duration_s: Mapping[str, float] | None = None,
) -> dict[str, ContactWrenchTrajectory]:
    """Create the exact complete path used during constrained selection."""

    grasp = _phase_trajectories_from_nominal_windows(nominal.windows)
    return materialize_order9_r1_complete_task_phases(
        phase_trajectories=grasp,
        task_spec=task_spec,
        lift_clearance_m=lift_clearance_m,
        retreat_offset_m=retreat_offset_m,
        phase_duration_s=phase_duration_s,
        nominal_dt_s=0.1,
    )


def audit_order9_r1_v12_candidate_at_planning_clearance(
    nominal: Order9C3NominalTrajectory,
    *,
    task_spec: TaskSpec,
    physical_model_config_path: str | Path,
    geometry_contract: Order9R1NominalGeometryV11Contract,
    clearance_contract: Order9R1SupportClearanceTeacherV12Contract,
    fail_fast: bool = True,
) -> tuple[dict[str, ContactWrenchTrajectory], dict[str, Any]]:
    """Materialize and check all eight phases at the 30 mm generation target."""

    phases = materialize_order9_r1_v12_candidate_phases(
        nominal,
        task_spec=task_spec,
    )
    planning_geometry = replace(
        geometry_contract,
        required_robot_support_clearance_m=(clearance_contract.planning_clearance_m),
    )
    audit = audit_order9_r1_complete_task_geometry_v12(
        phases=phases,
        task_spec=task_spec,
        morphology=nominal.selection_bundle.design_output.target_morphology,
        contact_candidate_set=nominal.selection_bundle.contact_candidate_set,
        physical_model_config_path=physical_model_config_path,
        contract=planning_geometry,
        fail_fast=fail_fast,
    )
    audit.update(
        {
            "teacher_contract_version": (
                ORDER9_R1_SUPPORT_CLEARANCE_TEACHER_V12_VERSION
            ),
            "teacher_contract_sha256": clearance_contract.config_sha256,
            "acceptance_clearance_m": (clearance_contract.acceptance_clearance_m),
            "planning_clearance_m": clearance_contract.planning_clearance_m,
            "planning_buffer_m": clearance_contract.actual_planning_buffer_m,
            "generation_time_clearance_constraint_applied": True,
            "complete_phase_candidate_selection": True,
            "generation_method": (
                "complete_path_hard_constraint_candidate_selection_v1"
            ),
            "vertical_contact_frame_shift_m": 0.0,
            "original_deterministic_ik_solution_rigidly_preserved": True,
            "isaac_invoked": False,
            "controller_layers_invoked": False,
        }
    )
    return phases, audit


def build_order9_r1_support_clearance_generation_certificate(
    audit: Mapping[str, Any],
    *,
    clearance_contract: Order9R1SupportClearanceTeacherV12Contract,
    candidate_id: str,
    candidate_rank: int,
    evaluated_candidate_count: int,
) -> dict[str, Any]:
    if audit.get("accepted") is not True or audit.get("status") != "accepted":
        raise SchemaValidationError(
            f"{E_R1_SUPPORT_CLEARANCE_GENERATION_PROOF}: planning audit rejected"
        )
    checked_phases = tuple(audit.get("checked_phases", ()))
    if checked_phases != clearance_contract.required_phases:
        raise SchemaValidationError(
            f"{E_R1_SUPPORT_CLEARANCE_GENERATION_PROOF}: phase proof incomplete"
        )
    if candidate_rank < 0 or evaluated_candidate_count <= candidate_rank:
        raise ValueError("R1 support-clearance candidate rank is invalid")
    return {
        "certificate_version": ORDER9_R1_SUPPORT_CLEARANCE_CERTIFICATE_V1,
        "candidate_id": str(candidate_id),
        "teacher_contract_sha256": clearance_contract.config_sha256,
        "acceptance_clearance_m": clearance_contract.acceptance_clearance_m,
        "planning_clearance_m": clearance_contract.planning_clearance_m,
        "planning_buffer_m": clearance_contract.actual_planning_buffer_m,
        "required_phases": list(clearance_contract.required_phases),
        "candidate_rank": int(candidate_rank),
        "evaluated_candidate_count": int(evaluated_candidate_count),
        "minimum_robot_support_clearance_lower_bound_m": float(
            audit["minimum_robot_support_clearance_lower_bound_m"]
        ),
        "generation_method": str(audit["generation_method"]),
        "vertical_contact_frame_shift_m": float(
            audit["vertical_contact_frame_shift_m"]
        ),
        "original_deterministic_ik_solution_rigidly_preserved": bool(
            audit["original_deterministic_ik_solution_rigidly_preserved"]
        ),
        "complete_phase_candidate_selection": True,
        "generation_time_clearance_constraint_applied": True,
        "unsupported_rigid_pose_copy": False,
        "planning_audit_semantic_hash": stable_hash(dict(audit)),
        "isaac_invoked": False,
        "controller_layers_invoked": False,
        "formal_teacher_collection_authorized": False,
    }


def require_order9_r1_support_clearance_generation_certificate(
    certificate: Mapping[str, Any] | None,
    *,
    clearance_contract: Order9R1SupportClearanceTeacherV12Contract,
    candidate_id: str,
) -> None:
    """Reject old copied place paths before they can become formal inputs."""

    if certificate is None:
        raise SchemaValidationError(
            f"{E_R1_UNCERTIFIED_RIGID_POSE_COPY}: generation certificate missing"
        )
    try:
        phases = tuple(certificate["required_phases"])
        acceptance = float(certificate["acceptance_clearance_m"])
        planning = float(certificate["planning_clearance_m"])
        buffer = float(certificate["planning_buffer_m"])
        shift = float(certificate["vertical_contact_frame_shift_m"])
    except (KeyError, TypeError, ValueError) as error:
        raise SchemaValidationError(
            f"{E_R1_SUPPORT_CLEARANCE_GENERATION_PROOF}: malformed certificate"
        ) from error
    if (
        certificate.get("certificate_version")
        != ORDER9_R1_SUPPORT_CLEARANCE_CERTIFICATE_V1
        or certificate.get("candidate_id") != candidate_id
        or certificate.get("teacher_contract_sha256")
        != clearance_contract.config_sha256
        or phases != clearance_contract.required_phases
        or not math.isclose(
            acceptance,
            clearance_contract.acceptance_clearance_m,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            planning,
            clearance_contract.planning_clearance_m,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            buffer,
            clearance_contract.actual_planning_buffer_m,
            abs_tol=1.0e-12,
        )
        or certificate.get("complete_phase_candidate_selection") is not True
        or certificate.get("generation_time_clearance_constraint_applied") is not True
        or certificate.get("unsupported_rigid_pose_copy") is not False
        or certificate.get("generation_method")
        not in {
            "complete_path_hard_constraint_candidate_selection_v1",
            "analytic_vertical_contact_frame_shift_v1",
            "analytic_vertical_contact_frame_fine_shift_v13",
            "analytic_planar_contact_frame_shift_v13",
        }
        or not 0.0
        <= shift
        <= clearance_contract.maximum_vertical_contact_frame_shift_m + 1.0e-12
        or certificate.get("original_deterministic_ik_solution_rigidly_preserved")
        is not True
        or certificate.get("isaac_invoked") is not False
        or certificate.get("controller_layers_invoked") is not False
        or certificate.get("formal_teacher_collection_authorized") is not False
    ):
        raise SchemaValidationError(
            f"{E_R1_SUPPORT_CLEARANCE_GENERATION_PROOF}: certificate differs"
        )


def select_order9_r1_maximin_clearance_candidate(
    candidates: Sequence[tuple[Any, Mapping[str, Any]]],
) -> tuple[int, Any, Mapping[str, Any]]:
    """Select the largest complete-path clearance with a stable tie break."""

    admitted = []
    for index, (candidate, audit) in enumerate(candidates):
        if audit.get("accepted") is not True or audit.get("status") != "accepted":
            continue
        clearance = float(
            audit.get("minimum_robot_support_clearance_lower_bound_m", -math.inf)
        )
        if math.isfinite(clearance):
            admitted.append((clearance, -index, index, candidate, audit))
    if not admitted:
        raise SchemaValidationError(
            f"{E_R1_SUPPORT_CLEARANCE_GENERATION_PROOF}: no admitted candidate"
        )
    _clearance, _negative_index, index, candidate, audit = max(admitted)
    return index, candidate, audit


__all__ = [
    "E_R1_SUPPORT_CLEARANCE_BUFFER",
    "E_R1_SUPPORT_CLEARANCE_GENERATION_PROOF",
    "E_R1_UNCERTIFIED_RIGID_POSE_COPY",
    "ORDER9_R1_SUPPORT_CLEARANCE_CERTIFICATE_V1",
    "ORDER9_R1_SUPPORT_CLEARANCE_TEACHER_V12_RELATIVE",
    "ORDER9_R1_SUPPORT_CLEARANCE_TEACHER_V12_VERSION",
    "Order9R1SupportClearanceTeacherV12Contract",
    "audit_order9_r1_v12_candidate_at_planning_clearance",
    "build_order9_r1_support_clearance_generation_certificate",
    "derive_order9_r1_vertical_clearance_nominal",
    "load_order9_r1_support_clearance_teacher_v12_contract",
    "materialize_order9_r1_v12_candidate_phases",
    "require_order9_r1_support_clearance_generation_certificate",
    "order9_r1_vertical_clearance_shift_candidates",
    "select_order9_r1_maximin_clearance_candidate",
]
