from __future__ import annotations

"""Lengthen selected R1 teacher phases without changing their spatial path."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from time import perf_counter
from typing import Any, Mapping

from amsrr.schemas.policies import ContactWrenchTrajectory
from amsrr.schemas.task_spec import TaskSpec
from amsrr.simulation.order9_object_task_runtime import ORDER9_OBJECT_TASK_PHASES
from amsrr.training.order9_c3_nominal_trajectory import (
    Order9C3NominalPhaseArtifact,
    Order9C3NominalTrajectoryArtifact,
    Order9C3NominalTrajectorySetManifest,
    validate_order9_c3_nominal_trajectory_artifact_bytes,
    validate_order9_c3_nominal_trajectory_set_bytes,
)
from amsrr.training.order9_r1_complete_task_audit import (
    audit_order9_r1_complete_task_semantics,
)
from amsrr.training.order9_r1_isaac_calibration import load_order9_r1_isaac_case
from amsrr.training.order9_r1_nominal_retime import _retime_trajectory
from amsrr.training.order9_r1_yaw_branch_repair import (
    _complete_timeline,
    _portable,
    _source_bucket_task_path,
    _write_json_text,
)
from amsrr.utils.hashing import hash_file, stable_hash

ORDER9_R1_PHASE_DURATION_REPAIR_VERSION = "order9_r1_phase_duration_repair_v1"
ORDER9_R1_CONTACT_HANDOFF_REPAIR_VERSION = "order9_r1_contact_handoff_repair_v1"
ORDER9_R1_ENDPOINT_HOLD_REPAIR_VERSION = "order9_r1_endpoint_hold_repair_v1"
ORDER9_R1_PHASE_DURATION_REPAIRS_RELATIVE = Path(
    "configs/training/order9_r1_phase_duration_repairs_v1.json"
)
_PHASES = tuple(value.value for value in ORDER9_OBJECT_TASK_PHASES)


def load_order9_r1_phase_duration_repairs(
    path: str | Path = ORDER9_R1_PHASE_DURATION_REPAIRS_RELATIVE,
    *,
    repository_root: str | Path,
) -> dict[str, dict[str, Any]]:
    """Load hash-bound repairs proven by failure and successful Isaac replays."""

    repository = Path(repository_root).resolve()
    source = Path(path)
    source = source.resolve() if source.is_absolute() else (repository / source).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    repairs = payload.get("repairs")
    if (
        payload.get("repair_version") != "order9_r1_phase_duration_repairs_v1"
        or payload.get("acceptance_or_safety_gate_changed") is not False
        or not isinstance(repairs, dict)
        or not repairs
    ):
        raise ValueError("R1 phase-duration repair registry differs")
    expected_multipliers = {phase: 1.0 for phase in _PHASES}
    expected_multipliers["place"] = 2.0
    result = {}
    for candidate_id, repair in repairs.items():
        if not isinstance(repair, dict):
            raise ValueError("R1 phase-duration repair entry differs")
        failure_binding = repair.get("source_confirmed_isaac_failure") or {}
        diagnostic = repair.get("diagnostic_isaac_success") or {}
        episodes_binding = diagnostic.get("episodes") or {}
        raw_binding = diagnostic.get("raw_rollout") or {}
        bound_paths = []
        for binding in (failure_binding, episodes_binding, raw_binding):
            bound = (repository / str(binding.get("path", ""))).resolve()
            if (
                repository not in bound.parents
                or not bound.is_file()
                or binding.get("sha256") != hash_file(bound)
            ):
                raise ValueError("R1 phase-duration repair binding differs")
            bound_paths.append(bound)
        failure = json.loads(bound_paths[0].read_text(encoding="utf-8"))
        episodes = [
            json.loads(line)
            for line in bound_paths[1].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if (
            repair.get("repair_kind")
            != "confirmed_release_entry_object_drop_place_slowdown"
            or repair.get("phase_duration_multipliers") != expected_multipliers
            or repair.get("spatial_path_changed") is not False
            or repair.get("joint_path_changed") is not False
            or repair.get("acceptance_or_safety_gate_changed") is not False
            or failure.get("candidate_id") != candidate_id
            or failure.get("failure_kind")
            != "independently_repeated_release_entry_object_drop"
            or failure.get("terminal_phase") != "release"
            or failure.get("acceptance_or_safety_gate_changed") is not False
            or len(episodes) != 2
            or any(
                episode.get("task_success") is not True
                or episode.get("safety_failure") is not False
                or int(episode.get("fallback_decision_count", -1)) != 0
                or float((episode.get("metrics") or {}).get("terminal_phase_index", -1.0))
                != 8.0
                for episode in episodes
            )
        ):
            raise ValueError("R1 phase-duration repair evidence differs")
        result[candidate_id] = dict(repair)
    return result


def derive_order9_r1_phase_duration_repair_case(
    *,
    source_case_root: str | Path,
    destination_case_root: str | Path,
    repository_root: str | Path,
    phase_duration_multipliers: Mapping[str, float],
    approach_contact_handoff_fraction: float | None = None,
    phase_endpoint_holds_s: Mapping[str, float] | None = None,
    physical_model_config_path: str | Path = "configs/robot/robot_model.yaml",
) -> dict[str, Any]:
    """Clone, slow, and cheaply admit one private R1 case."""

    started = perf_counter()
    repository = Path(repository_root).resolve()
    source_root = Path(source_case_root).resolve()
    destination = Path(destination_case_root).resolve()
    handoff_fraction = _validated_handoff_fraction(
        approach_contact_handoff_fraction
    )
    endpoint_holds = _validated_endpoint_holds(phase_endpoint_holds_s)
    multipliers = _validated_multipliers(
        phase_duration_multipliers,
        allow_noop=handoff_fraction is not None or any(endpoint_holds.values()),
    )
    if destination.exists():
        raise FileExistsError(destination)
    for name in ("case_manifest.json", "task_spec.json", "nominal_set"):
        if not (source_root / name).exists():
            raise FileNotFoundError(source_root / name)
    source_case = load_order9_r1_isaac_case(
        source_root / "case_manifest.json", repository
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        shutil.rmtree(temporary)
        shutil.copytree(
            source_root,
            temporary,
            ignore=shutil.ignore_patterns("isaac", "reset_bank.pt"),
        )
        task = TaskSpec.from_json(
            (temporary / "task_spec.json").read_text(encoding="utf-8")
        )
        set_path = temporary / "nominal_set/manifest.json"
        nominal_set = Order9C3NominalTrajectorySetManifest.from_json(
            set_path.read_text(encoding="utf-8")
        )
        nominal_set.validate()
        if len(nominal_set.entries) != 1:
            raise ValueError("R1 phase-duration repair set is not singular")
        artifact_path = set_path.parent / nominal_set.entries[0].artifact_path
        artifact = validate_order9_c3_nominal_trajectory_artifact_bytes(artifact_path)
        artifact_root = artifact_path.parent
        source_artifact_sha256 = hash_file(artifact_path)
        phases: dict[str, ContactWrenchTrajectory] = {}
        phase_entries = []
        source_horizons = {}
        for entry in artifact.phase_trajectories:
            path = artifact_root / entry.trajectory_path
            source = ContactWrenchTrajectory.from_json(path.read_text(encoding="utf-8"))
            source_horizons[entry.phase] = float(source.horizon_s)
            multiplier = float(multipliers[entry.phase])
            repaired = (
                source
                if math.isclose(multiplier, 1.0)
                else _retime_trajectory(source, scale=multiplier)
            )
            hold_s = endpoint_holds[entry.phase]
            if hold_s > 0.0:
                repaired = _extend_phase_endpoint_hold(repaired, hold_s=hold_s)
            if not math.isclose(multiplier, 1.0) or hold_s > 0.0:
                _write_json_text(path, repaired.to_dict())
            phases[entry.phase] = repaired
            phase_entries.append(
                Order9C3NominalPhaseArtifact(
                    phase=entry.phase,
                    trajectory_path=entry.trajectory_path,
                    trajectory_sha256=hash_file(path),
                    trajectory_hash=stable_hash(repaired.to_dict()),
                    knot_count=len(repaired.knots),
                    generation_method=(
                        entry.generation_method
                        + (
                            ""
                            if math.isclose(multiplier, 1.0)
                            else "+" + ORDER9_R1_PHASE_DURATION_REPAIR_VERSION
                        )
                        + (
                            ""
                            if hold_s <= 0.0
                            else "+" + ORDER9_R1_ENDPOINT_HOLD_REPAIR_VERSION
                        )
                    ),
                    collision_validation_status="pending_offline_admission",
                )
            )
        if tuple(phases) != _PHASES:
            raise ValueError("R1 phase-duration repair phase order differs")
        handoff = None
        if handoff_fraction is not None:
            source_pair = (
                phases["approach"],
                phases["contact_acquisition"],
            )
            repaired_approach, repaired_contact, handoff = (
                _move_approach_tail_to_contact_acquisition(
                    *source_pair,
                    handoff_fraction=handoff_fraction,
                )
            )
            phases["approach"] = repaired_approach
            phases["contact_acquisition"] = repaired_contact
            entry_by_phase = {
                entry.phase: entry for entry in artifact.phase_trajectories
            }
            for phase in ("approach", "contact_acquisition"):
                path = artifact_root / entry_by_phase[phase].trajectory_path
                _write_json_text(path, phases[phase].to_dict())
            phase_entries = [
                Order9C3NominalPhaseArtifact(
                    phase=entry.phase,
                    trajectory_path=entry.trajectory_path,
                    trajectory_sha256=hash_file(
                        artifact_root / entry.trajectory_path
                    ),
                    trajectory_hash=stable_hash(phases[entry.phase].to_dict()),
                    knot_count=len(phases[entry.phase].knots),
                    generation_method=(
                        entry.generation_method
                        + "+"
                        + ORDER9_R1_CONTACT_HANDOFF_REPAIR_VERSION
                        if entry.phase in {"approach", "contact_acquisition"}
                        else next(
                            value.generation_method
                            for value in phase_entries
                            if value.phase == entry.phase
                        )
                    ),
                    collision_validation_status="pending_offline_admission",
                )
                for entry in artifact.phase_trajectories
            ]
        timeline_path = artifact_root / artifact.timeline_path
        _write_json_text(timeline_path, _complete_timeline(phases))
        provenance = {
            "repair_version": ORDER9_R1_PHASE_DURATION_REPAIR_VERSION,
            "candidate_id": _candidate_id(source_root),
            "source_case_manifest_sha256": hash_file(
                source_root / "case_manifest.json"
            ),
            "source_artifact_sha256": source_artifact_sha256,
            "source_fast_screen_evidence_path": (
                source_case.manifest.fast_screen_evidence.path
            ),
            "source_fast_screen_evidence_sha256": (
                source_case.manifest.fast_screen_evidence.sha256
            ),
            "phase_duration_multipliers": multipliers,
            "phase_endpoint_holds_s": endpoint_holds,
            "approach_contact_handoff_fraction": handoff_fraction,
            "approach_contact_handoff": handoff,
            "source_phase_horizons_s": source_horizons,
            "repaired_phase_horizons_s": {
                phase: float(phases[phase].horizon_s) for phase in _PHASES
            },
            "spatial_path_changed": False,
            "joint_path_changed": False,
            "contact_assignments_changed": handoff_fraction is not None,
            "ik_invoked": False,
            "trajectory_optimization_invoked": False,
            "controller_layers_invoked": False,
            "isaac_invoked": False,
            "training_eligible": False,
        }
        evidence = deepcopy(artifact.selection_evidence)
        evidence["r1_phase_duration_repair"] = provenance
        artifact_payload = artifact.to_dict()
        artifact_payload.update(
            {
                "timeline_sha256": hash_file(timeline_path),
                "duration_s": sum(float(phases[phase].horizon_s) for phase in _PHASES),
                "phase_trajectories": [value.to_dict() for value in phase_entries],
                "proxy_collision_validation_status": "enforced_during_generation",
                "selection_evidence": evidence,
            }
        )
        repaired_artifact = Order9C3NominalTrajectoryArtifact.from_dict(
            artifact_payload
        )
        repaired_artifact.validate()
        _write_json_text(artifact_path, repaired_artifact.to_dict())

        _source_bucket_task_path(source_root, repository)
        fast_screen_path = repository / source_case.manifest.fast_screen_evidence.path
        fast_screen = json.loads(fast_screen_path.read_text(encoding="utf-8"))
        if (
            fast_screen.get("candidate_id") != _candidate_id(source_root)
            or fast_screen.get("accepted") is not True
            or fast_screen.get("eligible_for_full_control_test") is not True
            or fast_screen.get("grasp_pose_reached") is not True
            or int(fast_screen.get("maximum_collision_violating_pair_count", -1)) != 0
            or fast_screen.get("isaac_invoked") is not False
            or fast_screen.get("controller_layers_invoked") is not False
            or fast_screen.get("ik_resolve_invoked") is not False
            or fast_screen.get("trajectory_optimization_invoked") is not False
        ):
            raise ValueError("R1 phase-duration source fast screen is invalid")
        pre_grasp_bytes_preserved = all(
            math.isclose(multipliers[phase], 1.0)
            for phase in ("approach", "contact_acquisition")
        ) and handoff_fraction is None
        lightweight = {
            "audit_version": ORDER9_R1_PHASE_DURATION_REPAIR_VERSION,
            "status": "accepted",
            "checked_phases": list(_PHASES),
            "exact_tracking_to_grasp_screen_inherited": True,
            "inherited_grasp_screen_path": (
                source_case.manifest.fast_screen_evidence.path
            ),
            "inherited_grasp_screen_sha256": (
                source_case.manifest.fast_screen_evidence.sha256
            ),
            "pre_grasp_phase_bytes_preserved": pre_grasp_bytes_preserved,
            "pre_grasp_spatial_path_preserved": True,
            "approach_contact_concatenated_spatial_path_preserved": True,
            "all_phase_spatial_path_preserved": True,
            "post_grasp_spatial_path_preserved": True,
            "post_grasp_joint_path_preserved": True,
            "post_grasp_contact_assignments_preserved": True,
            "post_grasp_velocity_magnitudes_not_increased": True,
            "all_phase_velocity_magnitudes_not_increased": True,
            "minimum_collision_clearance_m": fast_screen.get(
                "minimum_collision_clearance_m"
            ),
            "minimum_normalized_joint_limit_reserve": fast_screen[
                "minimum_normalized_joint_limit_reserve"
            ],
            "maximum_body_tilt_rad": fast_screen["maximum_body_tilt_rad"],
            "isaac_invoked": False,
            "controller_layers_invoked": False,
            "ik_invoked": False,
            "trajectory_optimization_invoked": False,
            "training_eligible": False,
        }
        semantic = audit_order9_r1_complete_task_semantics(
            phases,
            task_spec=task,
            lift_clearance_m=0.30,
            retreat_offset_m=0.10,
            materializer_id=ORDER9_R1_PHASE_DURATION_REPAIR_VERSION,
        )
        audit_path = temporary / "phase_duration_repair_lightweight_audit.json"
        _write_json_text(
            audit_path,
            {
                **lightweight,
                "complete_task_semantic_audit": semantic,
                "repair_provenance": provenance,
            },
        )

        accepted_entries = []
        for entry in phase_entries:
            payload = entry.to_dict()
            payload["collision_validation_status"] = "accepted_offline_admission"
            accepted_entries.append(Order9C3NominalPhaseArtifact.from_dict(payload))
        artifact_payload = repaired_artifact.to_dict()
        artifact_payload["phase_trajectories"] = [
            value.to_dict() for value in accepted_entries
        ]
        artifact_payload["selection_evidence"]["r1_phase_duration_repair"].update(
            {
                "lightweight_audit_path": _portable(
                    destination / audit_path.name, repository
                ),
                "lightweight_audit_sha256": hash_file(audit_path),
                "minimum_collision_clearance_m": lightweight[
                    "minimum_collision_clearance_m"
                ],
                "minimum_normalized_joint_limit_reserve": lightweight[
                    "minimum_normalized_joint_limit_reserve"
                ],
            }
        )
        repaired_artifact = Order9C3NominalTrajectoryArtifact.from_dict(
            artifact_payload
        )
        _write_json_text(artifact_path, repaired_artifact.to_dict())
        validate_order9_c3_nominal_trajectory_artifact_bytes(artifact_path)

        nominal_set.entries[0].artifact_sha256 = hash_file(artifact_path)
        nominal_set.metadata.update(
            {
                "r1_phase_duration_repair": True,
                "r1_phase_duration_repair_version": (
                    ORDER9_R1_PHASE_DURATION_REPAIR_VERSION
                ),
                "phase_duration_multipliers": multipliers,
                "phase_endpoint_holds_s": endpoint_holds,
                "approach_contact_handoff_fraction": handoff_fraction,
                "lightweight_audit_sha256": hash_file(audit_path),
                "training_eligible": False,
            }
        )
        nominal_set.validate()
        _write_json_text(set_path, nominal_set.to_dict())
        collision_path = set_path.parent / "collision_validation.json"
        _write_json_text(
            collision_path,
            {
                "version": ORDER9_R1_PHASE_DURATION_REPAIR_VERSION,
                "all_accepted": True,
                "records": [
                    {
                        "bucket_id": _candidate_id(source_root),
                        "accepted": True,
                        "lightweight_audit_path": _portable(
                            destination / audit_path.name, repository
                        ),
                        "lightweight_audit_sha256": hash_file(audit_path),
                        "minimum_collision_clearance_m": lightweight[
                            "minimum_collision_clearance_m"
                        ],
                        "maximum_collision_violating_pair_count": 0,
                    }
                ],
            },
        )

        case_path = temporary / "case_manifest.json"
        case_payload = json.loads(case_path.read_text(encoding="utf-8"))
        case_payload["nominal_set"]["path"] = _portable(
            destination / "nominal_set/manifest.json", repository
        )
        case_payload["nominal_set"]["sha256"] = hash_file(set_path)
        case_payload["nominal_artifact"]["path"] = _portable(
            destination / "nominal_set" / nominal_set.entries[0].artifact_path,
            repository,
        )
        case_payload["nominal_artifact"]["sha256"] = hash_file(artifact_path)
        case_payload["reset_bank_path"] = _portable(
            destination / "reset_bank.pt", repository
        )
        _write_json_text(case_path, case_payload)

        admission = {
            "admission_version": ORDER9_R1_PHASE_DURATION_REPAIR_VERSION,
            "status": "accepted",
            "candidate_id": _candidate_id(source_root),
            "phase_duration_multipliers": multipliers,
            "phase_endpoint_holds_s": endpoint_holds,
            "approach_contact_handoff_fraction": handoff_fraction,
            "approach_contact_handoff": handoff,
            "case_manifest_path": _portable(
                destination / "case_manifest.json", repository
            ),
            "case_manifest_sha256": hash_file(case_path),
            "nominal_set_sha256": hash_file(set_path),
            "nominal_artifact_sha256": hash_file(artifact_path),
            "reset_bank_status": "pending_runtime_generation",
            "lightweight_audit_sha256": hash_file(audit_path),
            "minimum_collision_clearance_m": lightweight[
                "minimum_collision_clearance_m"
            ],
            "minimum_normalized_joint_limit_reserve": lightweight[
                "minimum_normalized_joint_limit_reserve"
            ],
            "ik_invoked": False,
            "trajectory_optimization_invoked": False,
            "controller_layers_invoked": False,
            "isaac_invoked": False,
            "wall_time_s": perf_counter() - started,
            "training_eligible": False,
            "formal_teacher_collection_authorized": False,
        }
        _write_json_text(temporary / "phase_duration_repair_admission.json", admission)
        os.rename(temporary, destination)
        validate_order9_c3_nominal_trajectory_set_bytes(
            destination / "nominal_set/manifest.json",
            repository_root=repository,
            expected_sha256=admission["nominal_set_sha256"],
        )
        return admission
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def relocate_order9_r1_phase_duration_repair_case(
    *,
    source_case_root: str | Path,
    destination_case_root: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Relocate a verified repair while rebinding every path-dependent hash."""

    repository = Path(repository_root).resolve()
    source_root = Path(source_case_root).resolve()
    destination = Path(destination_case_root).resolve()
    if source_root == destination or destination.exists():
        raise FileExistsError(destination)
    source_case_payload = json.loads(
        (source_root / "case_manifest.json").read_text(encoding="utf-8")
    )
    source_candidate_id = source_case_payload.get("candidate_id")
    if not isinstance(source_candidate_id, str) or not source_candidate_id:
        raise ValueError("R1 relocated phase-duration candidate is invalid")
    source_set_path = source_root / "nominal_set/manifest.json"
    source_set = Order9C3NominalTrajectorySetManifest.from_json(
        source_set_path.read_text(encoding="utf-8")
    )
    source_set.validate()
    if (
        len(source_set.entries) != 1
        or source_case_payload.get("nominal_set", {}).get("sha256")
        != hash_file(source_set_path)
    ):
        raise ValueError("R1 relocated phase-duration source set differs")
    source_artifact_path = source_set_path.parent / source_set.entries[0].artifact_path
    validate_order9_c3_nominal_trajectory_artifact_bytes(source_artifact_path)
    if (
        source_case_payload.get("nominal_artifact", {}).get("sha256")
        != hash_file(source_artifact_path)
    ):
        raise ValueError("R1 relocated phase-duration source artifact differs")
    source_reset_path = source_root / "reset_bank.pt"
    if source_reset_path.is_file():
        import torch

        source_reset = torch.load(
            source_reset_path,
            map_location="cpu",
            weights_only=False,
        )
        source_reference = (
            (source_reset.get("contract") or {}).get("c3_nominal_reference")
            if isinstance(source_reset, dict)
            else None
        )
        source_timeline_path = source_artifact_path.parent / "nominal_timeline.json"
        if (
            not isinstance(source_reference, dict)
            or source_reference.get("set_manifest_path")
            != source_case_payload["nominal_set"]["path"]
            or source_reference.get("set_manifest_sha256")
            != source_case_payload["nominal_set"]["sha256"]
            or source_reference.get("artifact_path")
            != source_case_payload["nominal_artifact"]["path"]
            or source_reference.get("artifact_sha256")
            != source_case_payload["nominal_artifact"]["sha256"]
            or source_reference.get("timeline_sha256")
            != hash_file(source_timeline_path)
        ):
            raise ValueError("R1 relocated source reset-bank binding differs")
    source_relative = _portable(source_root, repository)
    destination_relative = _portable(destination, repository)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        shutil.rmtree(temporary)
        shutil.copytree(source_root, temporary, ignore=shutil.ignore_patterns("isaac"))

        set_path = temporary / "nominal_set/manifest.json"
        nominal_set = Order9C3NominalTrajectorySetManifest.from_json(
            set_path.read_text(encoding="utf-8")
        )
        nominal_set.validate()
        if len(nominal_set.entries) != 1:
            raise ValueError("R1 relocated phase-duration set is not singular")
        artifact_path = set_path.parent / nominal_set.entries[0].artifact_path
        artifact_payload = _replace_portable_root(
            json.loads(artifact_path.read_text(encoding="utf-8")),
            source_relative=source_relative,
            destination_relative=destination_relative,
        )
        _write_json_text(artifact_path, artifact_payload)
        validate_order9_c3_nominal_trajectory_artifact_bytes(artifact_path)

        collision_path = set_path.parent / "collision_validation.json"
        collision_payload = _replace_portable_root(
            json.loads(collision_path.read_text(encoding="utf-8")),
            source_relative=source_relative,
            destination_relative=destination_relative,
        )
        _write_json_text(collision_path, collision_payload)

        nominal_set.entries[0].artifact_sha256 = hash_file(artifact_path)
        _write_json_text(set_path, nominal_set.to_dict())
        nominal_set_sha256 = hash_file(set_path)

        case_path = temporary / "case_manifest.json"
        case_payload = _replace_portable_root(
            json.loads(case_path.read_text(encoding="utf-8")),
            source_relative=source_relative,
            destination_relative=destination_relative,
        )
        case_payload["nominal_set"]["sha256"] = nominal_set_sha256
        case_payload["nominal_artifact"]["sha256"] = hash_file(artifact_path)
        _write_json_text(case_path, case_payload)

        reset_path = temporary / "reset_bank.pt"
        if reset_path.is_file():
            import torch

            reset_payload = torch.load(
                reset_path,
                map_location="cpu",
                weights_only=False,
            )
            contract = (
                reset_payload.get("contract")
                if isinstance(reset_payload, dict)
                else None
            )
            reference = (
                contract.get("c3_nominal_reference")
                if isinstance(contract, dict)
                else None
            )
            if not isinstance(reference, dict):
                raise ValueError("R1 relocated reset-bank reference is absent")
            reference.update(
                {
                    "set_manifest_path": _portable(
                        destination / "nominal_set/manifest.json", repository
                    ),
                    "set_manifest_sha256": nominal_set_sha256,
                    "artifact_path": _portable(
                        destination
                        / "nominal_set"
                        / nominal_set.entries[0].artifact_path,
                        repository,
                    ),
                    "artifact_sha256": hash_file(artifact_path),
                    "timeline_sha256": hash_file(
                        artifact_path.parent / "nominal_timeline.json"
                    ),
                }
            )
            torch.save(reset_payload, reset_path)

        admission_path = temporary / "phase_duration_repair_admission.json"
        admission = _replace_portable_root(
            json.loads(admission_path.read_text(encoding="utf-8")),
            source_relative=source_relative,
            destination_relative=destination_relative,
        )
        admission.update(
            {
                "case_manifest_sha256": hash_file(case_path),
                "nominal_set_sha256": nominal_set_sha256,
                "nominal_artifact_sha256": hash_file(artifact_path),
                "reset_bank_status": (
                    "relocated_hash_reference_rebound"
                    if reset_path.is_file()
                    else "pending_runtime_generation"
                ),
                "relocated_from": {
                    "path": source_relative + "/case_manifest.json",
                    "sha256": hash_file(source_root / "case_manifest.json"),
                },
            }
        )
        _write_json_text(admission_path, admission)
        os.rename(temporary, destination)
        relocated = load_order9_r1_isaac_case(
            destination / "case_manifest.json", repository
        )
        if relocated.manifest.candidate_id != source_candidate_id:
            raise ValueError("R1 relocated phase-duration candidate differs")
        return admission
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _replace_portable_root(
    value: Any,
    *,
    source_relative: str,
    destination_relative: str,
) -> Any:
    if isinstance(value, dict):
        return {
            key: _replace_portable_root(
                item,
                source_relative=source_relative,
                destination_relative=destination_relative,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _replace_portable_root(
                item,
                source_relative=source_relative,
                destination_relative=destination_relative,
            )
            for item in value
        ]
    if isinstance(value, str):
        return value.replace(source_relative, destination_relative)
    return value


def _validated_multipliers(
    values: Mapping[str, float], *, allow_noop: bool = False
) -> dict[str, float]:
    result = {str(key): float(value) for key, value in values.items()}
    if (
        set(result) != set(_PHASES)
        or any(
            not math.isfinite(value) or not 1.0 <= value <= 4.0
            for value in result.values()
        )
        or (
            not allow_noop
            and all(math.isclose(value, 1.0) for value in result.values())
        )
    ):
        raise ValueError("R1 phase-duration multipliers are invalid")
    return {phase: result[phase] for phase in _PHASES}


def _validated_handoff_fraction(value: float | None) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result) or not 0.5 <= result <= 0.95:
        raise ValueError("R1 approach/contact handoff fraction is invalid")
    return result


def _validated_endpoint_holds(
    values: Mapping[str, float] | None,
) -> dict[str, float]:
    raw = {} if values is None else {str(key): float(value) for key, value in values.items()}
    if set(raw) - set(_PHASES):
        raise ValueError("R1 endpoint-hold phase is invalid")
    result = {phase: float(raw.get(phase, 0.0)) for phase in _PHASES}
    if any(not math.isfinite(value) or not 0.0 <= value <= 30.0 for value in result.values()):
        raise ValueError("R1 endpoint-hold duration is invalid")
    return result


def _extend_phase_endpoint_hold(
    trajectory: ContactWrenchTrajectory, *, hold_s: float
) -> ContactWrenchTrajectory:
    """Append a stationary copy of the accepted endpoint."""

    duration = float(hold_s)
    if not math.isfinite(duration) or not 0.0 < duration <= 30.0:
        raise ValueError("R1 endpoint-hold duration is invalid")
    knots = deepcopy(trajectory.knots)
    endpoint = deepcopy(knots[-1])
    if endpoint.centroidal_target is not None:
        endpoint.centroidal_target.com_vel_world = (0.0, 0.0, 0.0)
    if endpoint.posture_target is not None:
        endpoint.posture_target.joint_vel_target = {
            key: 0.0
            for key in (endpoint.posture_target.joint_pos_target or {})
        }
    for target in endpoint.object_targets:
        if target.twist_target_world is not None:
            target.twist_target_world = [0.0] * 6
    step_count = max(1, int(math.ceil(duration / float(trajectory.dt_s))))
    start = float(trajectory.horizon_s)
    for index in range(1, step_count + 1):
        knot = deepcopy(endpoint)
        knot.t_rel_s = min(
            start + duration,
            start + index * float(trajectory.dt_s),
        )
        knots.append(knot)
    repaired = ContactWrenchTrajectory(
        horizon_s=start + duration,
        dt_s=float(trajectory.dt_s),
        knots=knots,
        derived_mode_label=trajectory.derived_mode_label,
        contract_version=trajectory.contract_version,
    )
    repaired.validate()
    return repaired


def _move_approach_tail_to_contact_acquisition(
    approach: ContactWrenchTrajectory,
    contact: ContactWrenchTrajectory,
    *,
    handoff_fraction: float,
) -> tuple[ContactWrenchTrajectory, ContactWrenchTrajectory, dict[str, Any]]:
    """Move an unchanged approach suffix under contact-acquisition semantics."""

    fraction = _validated_handoff_fraction(handoff_fraction)
    if fraction is None or len(approach.knots) < 4 or len(contact.knots) < 2:
        raise ValueError("R1 approach/contact trajectories are too short")
    split_index = int(math.floor(fraction * (len(approach.knots) - 1)))
    split_index = max(1, min(split_index, len(approach.knots) - 2))
    split_time = float(approach.knots[split_index].t_rel_s)
    if not 0.0 < split_time < float(approach.horizon_s):
        raise ValueError("R1 approach/contact handoff time is invalid")

    approach_knots = deepcopy(approach.knots[: split_index + 1])
    for condition in approach_knots[-1].guard_conditions:
        if condition.get("type") == "order9_phase_local_rolling_teacher":
            condition["phase_target_reached"] = "true"

    attach_template = deepcopy(contact.knots[0].contact_assignments)
    moved_knots = deepcopy(approach.knots[split_index:])
    for knot in moved_knots:
        knot.t_rel_s = float(knot.t_rel_s) - split_time
        knot.contact_assignments = deepcopy(attach_template)
        for condition in knot.guard_conditions:
            if condition.get("type") == "order9_phase_local_rolling_teacher":
                condition["phase"] = "contact_acquisition"
                condition["phase_target_reached"] = "false"
    moved_duration = float(moved_knots[-1].t_rel_s)
    contact_tail = deepcopy(contact.knots[1:])
    for knot in contact_tail:
        knot.t_rel_s = moved_duration + float(knot.t_rel_s)
    contact_knots = moved_knots + contact_tail

    repaired_approach = ContactWrenchTrajectory(
        horizon_s=split_time,
        dt_s=float(approach.dt_s),
        knots=approach_knots,
        derived_mode_label=approach.derived_mode_label,
        contract_version=approach.contract_version,
    )
    repaired_contact = ContactWrenchTrajectory(
        horizon_s=moved_duration + float(contact.horizon_s),
        dt_s=min(float(approach.dt_s), float(contact.dt_s)),
        knots=contact_knots,
        derived_mode_label=contact.derived_mode_label,
        contract_version=contact.contract_version,
    )
    repaired_approach.validate()
    repaired_contact.validate()
    return repaired_approach, repaired_contact, {
        "split_index": split_index,
        "source_approach_knot_count": len(approach.knots),
        "source_contact_knot_count": len(contact.knots),
        "repaired_approach_knot_count": len(repaired_approach.knots),
        "repaired_contact_knot_count": len(repaired_contact.knots),
        "source_approach_horizon_s": float(approach.horizon_s),
        "handoff_time_s": split_time,
        "moved_duration_s": moved_duration,
        "repaired_contact_horizon_s": float(repaired_contact.horizon_s),
        "joint_and_spatial_samples_reordered": False,
        "selected_contact_schedule_changed_only_in_moved_suffix": True,
    }


def _candidate_id(root: Path) -> str:
    payload = json.loads((root / "case_manifest.json").read_text(encoding="utf-8"))
    value = payload.get("candidate_id")
    if not isinstance(value, str) or not value:
        raise ValueError("R1 phase-duration repair candidate ID is invalid")
    return value


__all__ = [
    "ORDER9_R1_CONTACT_HANDOFF_REPAIR_VERSION",
    "ORDER9_R1_ENDPOINT_HOLD_REPAIR_VERSION",
    "ORDER9_R1_PHASE_DURATION_REPAIRS_RELATIVE",
    "ORDER9_R1_PHASE_DURATION_REPAIR_VERSION",
    "derive_order9_r1_phase_duration_repair_case",
    "load_order9_r1_phase_duration_repairs",
    "relocate_order9_r1_phase_duration_repair_case",
]
