from __future__ import annotations

"""Derive an R1 teacher path from a nearby admitted yaw branch.

The repair is intentionally a rigid world-frame change only.  It never runs
IK, trajectory optimization, a controller, or Isaac.  Joint coordinates and
all robot/object relative poses are inherited from the admitted reference
path, while the complete path is moved into the target object's task frame.
"""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from time import perf_counter
from typing import Any, Mapping, Sequence

import torch

from amsrr.feasibility.articulated_reachability import (
    CentroidalPostureIKConfig,
    _global_joint_limits,
    resolve_mesh_backed_anchor_references,
)
from amsrr.geometry.pose_math import (
    compose_pose,
    inverse_pose,
    matvec,
    transform_from_pose,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.robot_model.whole_structure_kinematics import (
    ordered_global_dock_joint_ids,
)
from amsrr.schemas.common import Pose7D, SchemaValidationError
from amsrr.schemas.contact_candidates import ContactCandidateSet
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.policies import ContactWrenchTrajectory, InteractionKnot
from amsrr.schemas.task_spec import TaskSpec
from amsrr.simulation.order9_object_task_runtime import ORDER9_OBJECT_TASK_PHASES
from amsrr.training.order9_c3_nominal_trajectory import (
    Order9C3NominalPhaseArtifact,
    Order9C3NominalTrajectoryArtifact,
    Order9C3NominalTrajectorySetManifest,
    validate_order9_c3_nominal_trajectory_artifact_bytes,
    validate_order9_c3_nominal_trajectory_set_bytes,
)
from amsrr.training.order9_posture_resolver import _default_posture_ik_solver
from amsrr.training.order9_posture_resolver import (
    Order9PostureCollisionBox,
    Order9PostureCollisionObject,
)
from amsrr.training.order9_r1_complete_task_audit import (
    audit_order9_r1_complete_task_semantics,
)
from amsrr.training.order9_r1_nominal_retreat import (
    ORDER9_R1_NOMINAL_RETREAT_VERSION,
    materialize_order9_r1_complete_task_phases,
)
from amsrr.training.order9_r1_support_clearance import (
    ORDER9_R1_ROBUST_SUPPORT_CLEARANCE_M,
    build_order9_r1_frozen_support_collision_object,
)
from amsrr.utils.hashing import hash_file, stable_hash

ORDER9_R1_YAW_BRANCH_REPAIR_VERSION = "order9_r1_yaw_branch_repair_v1"
ORDER9_R1_PLANAR_NOMINAL_TRANSFER_VERSION = "order9_r1_planar_nominal_transfer_v1"
ORDER9_R1_PLANAR_SCENE_TRANSFER_VERSION = "order9_r1_planar_scene_transfer_v4"
ORDER9_R1_YAW_BRANCH_LIGHTWEIGHT_AUDIT_VERSION = (
    "order9_r1_yaw_branch_lightweight_audit_v1"
)
_PHASES = tuple(value.value for value in ORDER9_OBJECT_TASK_PHASES)
_MAXIMUM_YAW_DELTA_RAD = math.radians(10.0) + 1.0e-9
_MAXIMUM_OTHER_ROTATION_RAD = math.radians(0.01)
_MINIMUM_NORMALIZED_JOINT_RESERVE = 0.01
_MAXIMUM_BODY_TILT_RAD = math.radians(60.0)
# The R1 user contract explicitly permits +/-30 mm grasp-point position error.
_ANCHOR_POSITION_TOLERANCE_M = 0.030
_ANCHOR_ATTITUDE_TOLERANCE_RAD = math.radians(1.0)


def derive_order9_r1_yaw_branch_case(
    *,
    reference_case_root: str | Path,
    target_case_root: str | Path,
    destination_case_root: str | Path,
    repository_root: str | Path,
    physical_model_config_path: str | Path = "configs/robot/robot_model.yaml",
    _allow_planar_translation: bool = False,
    _preserve_all_phases: bool = False,
    _scene_rigid_transform: bool = False,
) -> dict[str, Any]:
    """Create and admit one private derived case without changing either input."""

    started = perf_counter()
    repository = Path(repository_root).resolve()
    reference_root = Path(reference_case_root).resolve()
    target_root = Path(target_case_root).resolve()
    destination = Path(destination_case_root).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    for root in (reference_root, target_root):
        for name in ("case_manifest.json", "task_spec.json", "nominal_set"):
            if not (root / name).exists():
                raise FileNotFoundError(root / name)

    reference_task = TaskSpec.from_json(
        (reference_root / "task_spec.json").read_text(encoding="utf-8")
    )
    target_task = TaskSpec.from_json(
        (target_root / "task_spec.json").read_text(encoding="utf-8")
    )
    if _scene_rigid_transform and not _allow_planar_translation:
        raise ValueError("R1 scene transfer requires planar translation mode")
    if _scene_rigid_transform:
        scene_delta = _validated_task_scene_rigid_transform(reference_task, target_task)
        translation_world = tuple(float(value) for value in scene_delta[:3])
        yaw_rotation = (0.0, 0.0, 0.0, *scene_delta[3:])
        repair_version = ORDER9_R1_PLANAR_SCENE_TRANSFER_VERSION
    elif _allow_planar_translation:
        scene_delta = None
        translation_world, yaw_rotation = _validated_task_planar_change(
            reference_task, target_task
        )
        repair_version = ORDER9_R1_PLANAR_NOMINAL_TRANSFER_VERSION
    else:
        scene_delta = None
        translation_world = (0.0, 0.0, 0.0)
        yaw_rotation = _validated_task_yaw_rotation(reference_task, target_task)
        repair_version = ORDER9_R1_YAW_BRANCH_REPAIR_VERSION
    reference_object_start, _reference_object_goal = _object_start_and_goal(
        reference_task
    )
    initial_delta = (
        scene_delta
        if scene_delta is not None
        else compose_pose(
            (*translation_world, 0.0, 0.0, 0.0, 1.0),
            _centered_yaw_delta(reference_object_start, yaw_rotation),
        )
    )
    reference_candidate_id = _candidate_id(reference_root)
    target_candidate_id = _candidate_id(target_root)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        shutil.rmtree(temporary)
        shutil.copytree(
            target_root,
            temporary,
            ignore=shutil.ignore_patterns("isaac"),
        )
        target_set_path = temporary / "nominal_set" / "manifest.json"
        target_set = Order9C3NominalTrajectorySetManifest.from_json(
            target_set_path.read_text(encoding="utf-8")
        )
        target_set.validate()
        if len(target_set.entries) != 1:
            raise SchemaValidationError("R1 yaw repair target set is not singular")
        target_artifact_path = (
            target_set_path.parent / target_set.entries[0].artifact_path
        )
        target_artifact = validate_order9_c3_nominal_trajectory_artifact_bytes(
            target_artifact_path
        )

        reference_set_path = reference_root / "nominal_set" / "manifest.json"
        reference_set = Order9C3NominalTrajectorySetManifest.from_json(
            reference_set_path.read_text(encoding="utf-8")
        )
        reference_set.validate()
        if len(reference_set.entries) != 1:
            raise SchemaValidationError("R1 yaw repair reference set is not singular")
        reference_artifact_path = (
            reference_set_path.parent / reference_set.entries[0].artifact_path
        )
        reference_artifact = validate_order9_c3_nominal_trajectory_artifact_bytes(
            reference_artifact_path
        )
        _require_compatible_cases(
            reference_artifact=reference_artifact,
            target_artifact=target_artifact,
            reference_task=reference_task,
            target_task=target_task,
            allow_planar_translation=_allow_planar_translation,
        )

        target_artifact_root = target_artifact_path.parent
        reference_artifact_root = reference_artifact_path.parent
        target_morphology_path = (
            target_artifact_root / target_artifact.task_conditioned_morphology_path
        )
        target_morphology = MorphologyGraph.from_json(
            target_morphology_path.read_text(encoding="utf-8")
        )
        target_morphology.validate()

        reference_candidates_path = (
            reference_artifact_root / reference_artifact.contact_candidate_set_path
        )
        target_candidates_path = (
            target_artifact_root / target_artifact.contact_candidate_set_path
        )
        target_candidates_identity = ContactCandidateSet.from_json(
            target_candidates_path.read_text(encoding="utf-8")
        )
        repaired_candidates = transform_order9_r1_contact_candidates(
            ContactCandidateSet.from_json(
                reference_candidates_path.read_text(encoding="utf-8")
            ),
            delta_pose_world=initial_delta,
            target_task_id=target_candidates_identity.task_id,
            target_morphology_graph_id=target_morphology.graph_id,
            target_set_id=target_candidates_identity.set_id,
        )
        _write_json_text(target_candidates_path, repaired_candidates.to_dict())

        transformed_source_phases: dict[str, ContactWrenchTrajectory] = {}
        reference_entries = {
            value.phase: value for value in reference_artifact.phase_trajectories
        }
        for entry in reference_artifact.phase_trajectories:
            source = reference_artifact_root / entry.trajectory_path
            source_trajectory = ContactWrenchTrajectory.from_json(
                source.read_text(encoding="utf-8")
            )
            transformed_source_phases[entry.phase] = (
                transform_order9_r1_scene_trajectory(
                    source_trajectory,
                    delta_pose_world=initial_delta,
                )
                if _scene_rigid_transform
                else (
                    transform_order9_r1_planar_trajectory(
                        source_trajectory,
                        translation_world=translation_world,
                        yaw_rotation_world=yaw_rotation,
                    )
                    if _allow_planar_translation
                    else transform_order9_r1_trajectory(
                        source_trajectory,
                        yaw_rotation_world=yaw_rotation,
                    )
                )
            )
        repaired_phases = (
            transformed_source_phases
            if _preserve_all_phases
            else materialize_order9_r1_complete_task_phases(
                phase_trajectories={
                    phase: transformed_source_phases[phase] for phase in _PHASES[:2]
                },
                task_spec=target_task,
                lift_clearance_m=0.30,
                retreat_offset_m=0.10,
                phase_duration_s={
                    phase: float(transformed_source_phases[phase].horizon_s)
                    for phase in _PHASES
                },
                nominal_dt_s=0.1,
            )
        )
        repaired_phase_entries: list[Order9C3NominalPhaseArtifact] = []
        for phase in _PHASES:
            repaired = repaired_phases[phase]
            output = target_artifact_root / "phases" / f"{phase}.json"
            _write_json_text(output, repaired.to_dict())
            repaired_phase_entries.append(
                Order9C3NominalPhaseArtifact(
                    phase=phase,
                    trajectory_path=str(output.relative_to(target_artifact_root)),
                    trajectory_sha256=hash_file(output),
                    trajectory_hash=stable_hash(repaired.to_dict()),
                    knot_count=len(repaired.knots),
                    generation_method=(
                        reference_entries[phase].generation_method
                        + "+"
                        + repair_version
                        + (
                            ""
                            if _preserve_all_phases
                            else "+" + ORDER9_R1_NOMINAL_RETREAT_VERSION
                        )
                    ),
                    collision_validation_status="pending_offline_admission",
                )
            )
        if tuple(repaired_phases) != _PHASES:
            raise SchemaValidationError("R1 yaw repair phase order differs")

        timeline_path = target_artifact_root / target_artifact.timeline_path
        _write_json_text(timeline_path, _complete_timeline(repaired_phases))
        repair_provenance = {
            "repair_version": repair_version,
            "reference_candidate_id": reference_candidate_id,
            "target_candidate_id": target_candidate_id,
            "reference_case_manifest_sha256": hash_file(
                reference_root / "case_manifest.json"
            ),
            "target_case_manifest_sha256": hash_file(
                target_root / "case_manifest.json"
            ),
            "reference_artifact_sha256": hash_file(reference_artifact_path),
            "target_source_artifact_sha256": hash_file(
                target_root / "nominal_set" / target_set.entries[0].artifact_path
            ),
            "yaw_rotation_world": list(yaw_rotation),
            "translation_world_m": list(translation_world),
            "initial_centered_delta_pose_world": list(initial_delta),
            "fixed_scene_delta_pose_world": (
                None if scene_delta is None else list(scene_delta)
            ),
            "support_transformed_with_scene": scene_delta is not None,
            "ik_invoked": False,
            "trajectory_optimization_invoked": False,
            "controller_layers_invoked": False,
            "isaac_invoked": False,
            "joint_targets_inherited_byte_semantically": True,
            "object_relative_robot_path_preserved": True,
            "reference_contact_assignment_branch_inherited": True,
            "training_eligible": False,
        }
        selection_evidence = deepcopy(target_artifact.selection_evidence)
        selection_evidence["r1_yaw_branch_repair"] = repair_provenance
        artifact_payload = target_artifact.to_dict()
        artifact_payload.update(
            {
                "contact_candidate_set_sha256": hash_file(target_candidates_path),
                "contact_candidate_set_hash": stable_hash(
                    repaired_candidates.to_dict()
                ),
                "timeline_sha256": hash_file(timeline_path),
                "selected_surface_port_ids": list(
                    reference_artifact.selected_surface_port_ids
                ),
                "selected_candidate_group_id": (
                    reference_artifact.selected_candidate_group_id
                ),
                "duration_s": sum(
                    float(repaired_phases[phase].horizon_s) for phase in _PHASES
                ),
                "phase_trajectories": [
                    value.to_dict() for value in repaired_phase_entries
                ],
                "proxy_collision_validation_status": "enforced_during_generation",
                "selection_evidence": selection_evidence,
            }
        )
        repaired_artifact = Order9C3NominalTrajectoryArtifact.from_dict(
            artifact_payload
        )
        repaired_artifact.validate()
        _write_json_text(target_artifact_path, repaired_artifact.to_dict())
        validate_order9_c3_nominal_trajectory_artifact_bytes(target_artifact_path)

        source_bucket_task_path = _source_bucket_task_path(target_root, repository)
        source_task = TaskSpec.from_json(
            source_bucket_task_path.read_text(encoding="utf-8")
        )
        planar_invariant_audit = (
            audit_order9_r1_scene_transfer_invariants(
                source_phases={
                    phase: ContactWrenchTrajectory.from_json(
                        (
                            reference_artifact_root
                            / reference_entries[phase].trajectory_path
                        ).read_text(encoding="utf-8")
                    )
                    for phase in _PHASES
                },
                target_phases=repaired_phases,
                delta_pose_world=initial_delta,
            )
            if _scene_rigid_transform
            else (
                audit_order9_r1_planar_transfer_invariants(
                    source_phases={
                        phase: ContactWrenchTrajectory.from_json(
                            (
                                reference_artifact_root
                                / reference_entries[phase].trajectory_path
                            ).read_text(encoding="utf-8")
                        )
                        for phase in _PHASES
                    },
                    target_phases=repaired_phases,
                    translation_world=translation_world,
                    yaw_rotation_world=yaw_rotation,
                )
                if _allow_planar_translation
                else None
            )
        )
        physical_path = Path(physical_model_config_path)
        if not physical_path.is_absolute():
            physical_path = repository / physical_path
        inherited_scene_collision = (
            _validated_scene_collision_reference(
                reference_root,
                source_phases={
                    phase: ContactWrenchTrajectory.from_json(
                        (
                            reference_artifact_root
                            / reference_entries[phase].trajectory_path
                        ).read_text(encoding="utf-8")
                    )
                    for phase in _PHASES
                },
            )
            if _scene_rigid_transform
            else None
        )
        lightweight_audit = audit_order9_r1_yaw_branch_path(
            phases=repaired_phases,
            task_spec=target_task,
            source_task_spec=(
                reference_task if _allow_planar_translation else source_task
            ),
            morphology=target_morphology,
            contact_candidate_set=repaired_candidates,
            physical_model_config_path=physical_path,
            inherit_object_relative_collision=_allow_planar_translation,
            scene_delta_pose_world=scene_delta,
            inherited_scene_collision_clearance_m=(
                None
                if inherited_scene_collision is None
                else inherited_scene_collision["minimum_clearance_m"]
            ),
        )
        semantic_audit = (
            _validated_scene_semantic_reference(
                reference_root,
                target_phases=repaired_phases,
                delta_pose_world=initial_delta,
            )
            if _scene_rigid_transform
            else audit_order9_r1_complete_task_semantics(
                repaired_phases,
                task_spec=target_task,
                lift_clearance_m=0.30,
                retreat_offset_m=0.10,
                materializer_id=repair_version,
            )
        )
        audit_payload = {
            **lightweight_audit,
            "planar_transfer_invariant_audit": planar_invariant_audit,
            "complete_task_semantic_audit": semantic_audit,
            "repair_provenance": repair_provenance,
            "inherited_scene_collision_reference": inherited_scene_collision,
        }
        audit_path = temporary / "yaw_branch_repair_lightweight_audit.json"
        _write_json_text(audit_path, audit_payload)

        accepted_entries = []
        for value in repaired_phase_entries:
            payload = value.to_dict()
            payload["collision_validation_status"] = "accepted_offline_admission"
            accepted_entries.append(Order9C3NominalPhaseArtifact.from_dict(payload))
        artifact_payload = repaired_artifact.to_dict()
        artifact_payload["phase_trajectories"] = [
            value.to_dict() for value in accepted_entries
        ]
        artifact_payload["selection_evidence"]["r1_yaw_branch_repair"].update(
            {
                "lightweight_audit_path": _portable(
                    destination / audit_path.name, repository
                ),
                "lightweight_audit_sha256": hash_file(audit_path),
                "minimum_collision_clearance_m": lightweight_audit[
                    "minimum_collision_clearance_m"
                ],
                "minimum_normalized_joint_limit_reserve": lightweight_audit[
                    "minimum_normalized_joint_limit_reserve"
                ],
            }
        )
        repaired_artifact = Order9C3NominalTrajectoryArtifact.from_dict(
            artifact_payload
        )
        _write_json_text(target_artifact_path, repaired_artifact.to_dict())
        validate_order9_c3_nominal_trajectory_artifact_bytes(target_artifact_path)

        target_set.entries[0].artifact_sha256 = hash_file(target_artifact_path)
        target_set.metadata.update(
            {
                "r1_yaw_branch_repair": True,
                "r1_yaw_branch_repair_version": repair_version,
                "reference_candidate_id": reference_candidate_id,
                "lightweight_audit_sha256": hash_file(audit_path),
                "training_eligible": False,
            }
        )
        target_set.validate()
        _write_json_text(target_set_path, target_set.to_dict())
        collision_path = target_set_path.parent / "collision_validation.json"
        _write_json_text(
            collision_path,
            {
                "version": ORDER9_R1_YAW_BRANCH_LIGHTWEIGHT_AUDIT_VERSION,
                "all_accepted": True,
                "records": [
                    {
                        "bucket_id": target_candidate_id,
                        "accepted": True,
                        "lightweight_audit_path": _portable(
                            destination / audit_path.name, repository
                        ),
                        "lightweight_audit_sha256": hash_file(audit_path),
                        "minimum_collision_clearance_m": lightweight_audit[
                            "minimum_collision_clearance_m"
                        ],
                        "maximum_collision_violating_pair_count": 0,
                    }
                ],
            },
        )

        reference_reset_path = reference_root / "reset_bank.pt"
        reset_bank_status = "pending_runtime_generation"
        reset_bank_sha256 = None
        if reference_reset_path.is_file():
            _transform_reset_bank(
                source_path=reference_reset_path,
                destination_path=temporary / "reset_bank.pt",
                yaw_rotation_world=yaw_rotation,
                translation_world=translation_world,
                delta_pose_world=scene_delta,
                morphology_hash=target_morphology.stable_hash(),
                task_spec_hash=stable_hash(target_task.to_dict()),
                nominal_set_path=Path(
                    _portable(destination / "nominal_set" / "manifest.json", repository)
                ),
                nominal_set_sha256=hash_file(target_set_path),
                nominal_artifact_path=Path(
                    _portable(
                        destination
                        / "nominal_set"
                        / target_set.entries[0].artifact_path,
                        repository,
                    )
                ),
                nominal_artifact_sha256=hash_file(target_artifact_path),
                timeline_sha256=hash_file(timeline_path),
            )
            reset_bank_status = "derived_from_reference"
            reset_bank_sha256 = hash_file(temporary / "reset_bank.pt")
        case_path = temporary / "case_manifest.json"
        case_payload = json.loads(case_path.read_text(encoding="utf-8"))
        case_payload["nominal_set"]["path"] = _portable(
            destination / "nominal_set" / "manifest.json", repository
        )
        case_payload["nominal_set"]["sha256"] = hash_file(target_set_path)
        case_payload["nominal_artifact"]["path"] = _portable(
            destination / "nominal_set" / target_set.entries[0].artifact_path,
            repository,
        )
        case_payload["nominal_artifact"]["sha256"] = hash_file(target_artifact_path)
        if _allow_planar_translation:
            case_payload["fast_screen_evidence"] = {
                "artifact_kind": "fast_kinematic_screen",
                "path": _portable(destination / audit_path.name, repository),
                "sha256": hash_file(audit_path),
            }
        case_payload["reset_bank_path"] = _portable(
            destination / "reset_bank.pt", repository
        )
        _write_json_text(case_path, case_payload)

        admission = {
            "admission_version": repair_version,
            "status": "accepted",
            "candidate_id": target_candidate_id,
            "reference_candidate_id": reference_candidate_id,
            "yaw_rotation_world": list(yaw_rotation),
            "translation_world_m": list(translation_world),
            "initial_centered_delta_pose_world": list(initial_delta),
            "fixed_scene_delta_pose_world": (
                None if scene_delta is None else list(scene_delta)
            ),
            "support_transformed_with_scene": scene_delta is not None,
            "case_manifest_path": _portable(
                destination / "case_manifest.json", repository
            ),
            "case_manifest_sha256": hash_file(case_path),
            "nominal_set_sha256": hash_file(target_set_path),
            "nominal_artifact_sha256": hash_file(target_artifact_path),
            "reset_bank_status": reset_bank_status,
            "reset_bank_sha256": reset_bank_sha256,
            "lightweight_audit_sha256": hash_file(audit_path),
            "minimum_collision_clearance_m": lightweight_audit[
                "minimum_collision_clearance_m"
            ],
            "minimum_normalized_joint_limit_reserve": lightweight_audit[
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
        admission_path = temporary / "yaw_branch_repair_admission.json"
        _write_json_text(admission_path, admission)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.rename(temporary, destination)
        validate_order9_c3_nominal_trajectory_set_bytes(
            destination / "nominal_set" / "manifest.json",
            repository_root=repository,
            expected_sha256=admission["nominal_set_sha256"],
        )
        return admission
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def derive_order9_r1_planar_nominal_transfer_case(
    *,
    reference_case_root: str | Path,
    target_task_spec: TaskSpec,
    target_candidate_id: str,
    target_level_id: str,
    target_seed: int,
    destination_case_root: str | Path,
    repository_root: str | Path,
    physical_model_config_path: str | Path = "configs/robot/robot_model.yaml",
) -> dict[str, Any]:
    """Build one diagnostic case by rigidly moving an admitted complete scene."""

    repository = Path(repository_root).resolve()
    reference = Path(reference_case_root).resolve()
    destination = Path(destination_case_root).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if not target_candidate_id or not target_level_id or target_seed < 0:
        raise ValueError("R1 planar transfer target identity is invalid")
    target_task_spec.validate()
    if (
        target_task_spec.metadata.get("r1_calibration_candidate_id")
        != target_candidate_id
    ):
        raise SchemaValidationError("R1 planar target task candidate identity differs")
    reference_task = TaskSpec.from_json(
        (reference / "task_spec.json").read_text(encoding="utf-8")
    )
    target_task_spec = transform_order9_r1_planar_task_scene(
        reference_task,
        target_identity_task=target_task_spec,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.target-", dir=destination.parent
    ) as target_text:
        target = Path(target_text) / "case"
        shutil.copytree(reference, target, ignore=shutil.ignore_patterns("isaac"))
        task_path = target / "task_spec.json"
        _write_json_text(task_path, target_task_spec.to_dict())

        set_path = target / "nominal_set" / "manifest.json"
        set_manifest = Order9C3NominalTrajectorySetManifest.from_json(
            set_path.read_text(encoding="utf-8")
        )
        set_manifest.validate()
        if len(set_manifest.entries) != 1:
            raise SchemaValidationError("R1 planar source nominal set is not singular")
        entry = set_manifest.entries[0]
        old_artifact_path = set_path.parent / entry.artifact_path
        new_artifact_path = (
            set_path.parent / "buckets" / target_candidate_id / "manifest.json"
        )
        if old_artifact_path.parent != new_artifact_path.parent:
            new_artifact_path.parent.parent.mkdir(parents=True, exist_ok=True)
            os.rename(old_artifact_path.parent, new_artifact_path.parent)
        entry.bucket_id = target_candidate_id
        entry.artifact_path = str(new_artifact_path.relative_to(set_path.parent))
        bucket_path = set_path.parent / "bucket_manifest.json"
        bucket_payload = json.loads(bucket_path.read_text(encoding="utf-8"))
        if (
            not isinstance(bucket_payload.get("buckets"), list)
            or len(bucket_payload["buckets"]) != 1
        ):
            raise SchemaValidationError("R1 planar bucket manifest is not singular")
        bucket_payload["buckets"][0]["bucket_id"] = target_candidate_id
        bucket_payload["buckets"][0]["task_spec_sha256"] = hash_file(task_path)
        _write_json_text(bucket_path, bucket_payload)
        set_manifest.bucket_manifest_path = _portable(
            destination / "nominal_set" / "bucket_manifest.json", repository
        )
        set_manifest.bucket_manifest_sha256 = hash_file(bucket_path)
        set_manifest.validate()
        _write_json_text(set_path, set_manifest.to_dict())

        candidates_path = new_artifact_path.parent / "contact_candidate_set.json"
        candidates_payload = json.loads(candidates_path.read_text(encoding="utf-8"))
        candidates_payload["task_id"] = target_task_spec.task_id
        candidates_payload["set_id"] = (
            f"{target_candidate_id}:planar_nominal_transfer_contacts"
        )
        ContactCandidateSet.from_dict(candidates_payload).validate()
        _write_json_text(candidates_path, candidates_payload)

        artifact_payload = json.loads(new_artifact_path.read_text(encoding="utf-8"))
        artifact_payload.update(
            {
                "bucket_id": target_candidate_id,
                "task_spec_sha256": hash_file(task_path),
                "contact_candidate_set_sha256": hash_file(candidates_path),
                "contact_candidate_set_hash": stable_hash(candidates_payload),
            }
        )
        _write_json_text(new_artifact_path, artifact_payload)
        entry.artifact_sha256 = hash_file(new_artifact_path)
        set_manifest.metadata["candidate_id"] = target_candidate_id
        _write_json_text(set_path, set_manifest.to_dict())

        case_path = target / "case_manifest.json"
        case_payload = json.loads(case_path.read_text(encoding="utf-8"))
        case_payload.update(
            {
                "candidate_id": target_candidate_id,
                "level_id": target_level_id,
                "seed": int(target_seed),
            }
        )
        case_payload["task_spec"].update(
            {
                "path": _portable(destination / "task_spec.json", repository),
                "sha256": hash_file(task_path),
            }
        )
        _write_json_text(case_path, case_payload)
        return derive_order9_r1_yaw_branch_case(
            reference_case_root=reference,
            target_case_root=target,
            destination_case_root=destination,
            repository_root=repository,
            physical_model_config_path=physical_model_config_path,
            _allow_planar_translation=True,
            _preserve_all_phases=True,
            _scene_rigid_transform=True,
        )


def transform_order9_r1_planar_task_scene(
    reference_task: TaskSpec,
    *,
    target_identity_task: TaskSpec,
) -> TaskSpec:
    """Move the complete reference scene to the requested target object pose.

    The target task supplies only the candidate identity and desired object-start
    pose.  Object goals, supports, obstacles, and every other world pose are
    inherited from the successful reference and moved by one fixed horizontal
    rigid transform.  This prevents a support from being left behind while the
    robot/object trajectory is moved.
    """

    reference_task.validate()
    target_identity_task.validate()
    reference_start, _reference_goal = _object_start_and_goal(reference_task)
    requested_start, requested_goal = _object_start_and_goal(target_identity_task)
    delta = _bounded_planar_scene_delta(reference_start, requested_start)
    _require_scene_task_compatibility(reference_task, target_identity_task)

    payload = reference_task.to_dict()
    payload["task_id"] = target_identity_task.task_id
    for obj in payload["scene"]["objects"]:
        obj["pose_world"] = list(compose_pose(delta, tuple(obj["pose_world"])))
    for surface in payload["scene"]["environment"]["support_surfaces"]:
        surface["pose_world"] = list(compose_pose(delta, tuple(surface["pose_world"])))
    for obstacle in payload["scene"]["environment"]["obstacles"]:
        obstacle["pose_world"] = list(
            compose_pose(delta, tuple(obstacle["pose_world"]))
        )
    for goal in payload["goals"]:
        if goal.get("target_pose_world") is not None:
            goal["target_pose_world"] = list(
                compose_pose(delta, tuple(goal["target_pose_world"]))
            )

    transformed_goal = next(
        tuple(goal["target_pose_world"])
        for goal in payload["goals"]
        if goal.get("goal_type") == "object_pose"
        and goal.get("target_entity_id")
        == target_identity_task.goals[0].target_entity_id
        and goal.get("target_pose_world") is not None
    )
    metadata = deepcopy(target_identity_task.metadata)
    metadata.update(
        {
            "r1_planar_scene_transfer_version": (
                ORDER9_R1_PLANAR_SCENE_TRANSFER_VERSION
            ),
            "r1_planar_scene_delta_pose_world": list(delta),
            "r1_planar_scene_reference_task_id": reference_task.task_id,
            "r1_requested_unmoved_support_goal_pose_world": list(requested_goal),
            "r1_scene_transformed_goal_pose_world": list(transformed_goal),
            "r1_support_geometry_preserved": True,
            "r1_support_pose_preserved": False,
            "r1_support_pose_transformed_with_scene": True,
            "r1_robot_reset_preserved": False,
            "r1_robot_reset_transformed_with_scene": True,
            "r1_formal_range_selection_eligible": False,
            "r1_formal_teacher_collection_eligible": False,
        }
    )
    payload["metadata"] = metadata
    value = TaskSpec.from_dict(payload)
    value.validate()
    mapped_start, _mapped_goal = _object_start_and_goal(value)
    if not _pose_close(mapped_start, requested_start, position=1.0e-8, angle=1.0e-8):
        raise SchemaValidationError("R1 scene transfer missed requested object pose")
    _validated_task_scene_rigid_transform(reference_task, value)
    return value


def transform_order9_r1_trajectory(
    trajectory: ContactWrenchTrajectory,
    *,
    delta_pose_world: Pose7D | None = None,
    yaw_rotation_world: Pose7D | None = None,
) -> ContactWrenchTrajectory:
    """Apply one rigid transform while preserving all internal coordinates."""

    if (delta_pose_world is None) == (yaw_rotation_world is None):
        raise ValueError(
            "R1 yaw trajectory transform requires exactly one transform mode"
        )
    source = ContactWrenchTrajectory.from_dict(trajectory.to_dict())
    source.validate()
    knots = [
        _transform_knot(
            knot,
            delta_pose_world=delta_pose_world,
            yaw_rotation_world=yaw_rotation_world,
            rotate_contact_wrenches=(source.contract_version == "implicit_world_v1"),
        )
        for knot in source.knots
    ]
    value = ContactWrenchTrajectory(
        horizon_s=float(source.horizon_s),
        dt_s=float(source.dt_s),
        knots=knots,
        derived_mode_label=(
            f"{source.derived_mode_label or 'unspecified'}:"
            f"{ORDER9_R1_YAW_BRANCH_REPAIR_VERSION}"
        ),
        contract_version=source.contract_version,
    )
    value.validate()
    return value


def audit_order9_r1_planar_transfer_invariants(
    *,
    source_phases: Mapping[str, ContactWrenchTrajectory],
    target_phases: Mapping[str, ContactWrenchTrajectory],
    translation_world: Sequence[float],
    yaw_rotation_world: Pose7D,
) -> dict[str, Any]:
    """Prove exact joint inheritance and object-relative pose preservation."""

    if tuple(source_phases) != _PHASES or tuple(target_phases) != _PHASES:
        raise SchemaValidationError("R1 planar invariant phase order differs")
    translation_pose: Pose7D = (
        *tuple(float(value) for value in translation_world),
        0.0,
        0.0,
        0.0,
        1.0,
    )
    maximum_relative_position_error = 0.0
    maximum_relative_attitude_error = 0.0
    maximum_object_mapping_position_error = 0.0
    maximum_object_mapping_attitude_error = 0.0
    checked_knots = 0
    checked_anchor_poses = 0
    for phase in _PHASES:
        source = source_phases[phase]
        target = target_phases[phase]
        if len(source.knots) != len(target.knots):
            raise SchemaValidationError("R1 planar transfer changed knot count")
        for source_knot, target_knot in zip(source.knots, target.knots):
            checked_knots += 1
            if float(source_knot.t_rel_s) != float(target_knot.t_rel_s):
                raise SchemaValidationError("R1 planar transfer changed knot time")
            source_posture = source_knot.posture_target
            target_posture = target_knot.posture_target
            if (source_posture is None) != (target_posture is None):
                raise SchemaValidationError(
                    "R1 planar transfer changed posture presence"
                )
            if source_posture is not None and target_posture is not None:
                if (
                    source_posture.joint_pos_target != target_posture.joint_pos_target
                    or source_posture.joint_vel_target
                    != target_posture.joint_vel_target
                ):
                    raise SchemaValidationError(
                        "R1 planar transfer changed a joint target"
                    )
            source_assignments = [
                (
                    value.slot_id,
                    value.anchor_id,
                    value.candidate_id,
                    value.schedule_state,
                )
                for value in source_knot.contact_assignments
            ]
            target_assignments = [
                (
                    value.slot_id,
                    value.anchor_id,
                    value.candidate_id,
                    value.schedule_state,
                )
                for value in target_knot.contact_assignments
            ]
            if source_assignments != target_assignments:
                raise SchemaValidationError(
                    "R1 planar transfer changed contact assignment identity"
                )
            source_objects = [
                tuple(value.pose_target_world)
                for value in source_knot.object_targets
                if value.pose_target_world is not None
            ]
            target_objects = [
                tuple(value.pose_target_world)
                for value in target_knot.object_targets
                if value.pose_target_world is not None
            ]
            if len(source_objects) != 1 or len(target_objects) != 1:
                raise SchemaValidationError("R1 planar invariant object target differs")
            expected_object = compose_pose(
                translation_pose,
                compose_pose(
                    _centered_yaw_delta(source_objects[0], yaw_rotation_world),
                    source_objects[0],
                ),
            )
            maximum_object_mapping_position_error = max(
                maximum_object_mapping_position_error,
                math.dist(expected_object[:3], target_objects[0][:3]),
            )
            maximum_object_mapping_attitude_error = max(
                maximum_object_mapping_attitude_error,
                _quaternion_distance_rad(expected_object[3:], target_objects[0][3:]),
            )
            pose_pairs = []
            if (
                source_knot.centroidal_target is not None
                and target_knot.centroidal_target is not None
                and source_knot.centroidal_target.com_pos_world is not None
                and source_knot.centroidal_target.body_orientation_world is not None
                and target_knot.centroidal_target.com_pos_world is not None
                and target_knot.centroidal_target.body_orientation_world is not None
            ):
                pose_pairs.append(
                    (
                        (
                            *source_knot.centroidal_target.com_pos_world,
                            *source_knot.centroidal_target.body_orientation_world,
                        ),
                        (
                            *target_knot.centroidal_target.com_pos_world,
                            *target_knot.centroidal_target.body_orientation_world,
                        ),
                    )
                )
            if source_posture is not None and target_posture is not None:
                source_anchors = source_posture.free_anchor_pose_targets or {}
                target_anchors = target_posture.free_anchor_pose_targets or {}
                if set(source_anchors) != set(target_anchors):
                    raise SchemaValidationError(
                        "R1 planar transfer changed free-anchor identity"
                    )
                for anchor_id in sorted(source_anchors):
                    pose_pairs.append(
                        (source_anchors[anchor_id], target_anchors[anchor_id])
                    )
                    checked_anchor_poses += 1
            for source_pose, target_pose in pose_pairs:
                source_relative = compose_pose(
                    inverse_pose(source_objects[0]), tuple(source_pose)
                )
                target_relative = compose_pose(
                    inverse_pose(target_objects[0]), tuple(target_pose)
                )
                maximum_relative_position_error = max(
                    maximum_relative_position_error,
                    math.dist(source_relative[:3], target_relative[:3]),
                )
                maximum_relative_attitude_error = max(
                    maximum_relative_attitude_error,
                    _quaternion_distance_rad(source_relative[3:], target_relative[3:]),
                )
    tolerance = 1.0e-7
    if (
        maximum_relative_position_error > tolerance
        or maximum_relative_attitude_error > tolerance
        or maximum_object_mapping_position_error > tolerance
        or maximum_object_mapping_attitude_error > tolerance
    ):
        raise SchemaValidationError("R1 planar transfer invariant error is excessive")
    return {
        "audit_version": "order9_r1_planar_transfer_invariant_audit_v1",
        "status": "accepted",
        "checked_knot_count": checked_knots,
        "checked_free_anchor_pose_count": checked_anchor_poses,
        "joint_position_targets_exactly_preserved": True,
        "joint_velocity_targets_exactly_preserved": True,
        "contact_assignment_identities_exactly_preserved": True,
        "maximum_object_relative_position_error_m": maximum_relative_position_error,
        "maximum_object_relative_attitude_error_rad": maximum_relative_attitude_error,
        "maximum_object_mapping_position_error_m": (
            maximum_object_mapping_position_error
        ),
        "maximum_object_mapping_attitude_error_rad": (
            maximum_object_mapping_attitude_error
        ),
        "controller_layers_invoked": False,
        "isaac_invoked": False,
    }


def audit_order9_r1_scene_transfer_invariants(
    *,
    source_phases: Mapping[str, ContactWrenchTrajectory],
    target_phases: Mapping[str, ContactWrenchTrajectory],
    delta_pose_world: Pose7D,
) -> dict[str, Any]:
    """Prove that all phase data uses one fixed scene transform."""

    if tuple(source_phases) != _PHASES or tuple(target_phases) != _PHASES:
        raise SchemaValidationError("R1 scene invariant phase order differs")
    checked_knots = 0
    checked_anchor_poses = 0
    maximum_relative_position_error = 0.0
    maximum_relative_attitude_error = 0.0
    for phase in _PHASES:
        source = source_phases[phase]
        target = target_phases[phase]
        expected = transform_order9_r1_scene_trajectory(
            source,
            delta_pose_world=delta_pose_world,
        )
        if expected.to_dict() != target.to_dict():
            raise SchemaValidationError(
                f"R1 scene transfer differs from its fixed transform at {phase}"
            )
        for source_knot, target_knot in zip(source.knots, target.knots):
            checked_knots += 1
            source_objects = [
                tuple(value.pose_target_world)
                for value in source_knot.object_targets
                if value.pose_target_world is not None
            ]
            target_objects = [
                tuple(value.pose_target_world)
                for value in target_knot.object_targets
                if value.pose_target_world is not None
            ]
            if len(source_objects) != 1 or len(target_objects) != 1:
                raise SchemaValidationError("R1 scene invariant object target differs")
            source_posture = source_knot.posture_target
            target_posture = target_knot.posture_target
            source_anchors = (
                {}
                if source_posture is None
                else source_posture.free_anchor_pose_targets or {}
            )
            target_anchors = (
                {}
                if target_posture is None
                else target_posture.free_anchor_pose_targets or {}
            )
            if set(source_anchors) != set(target_anchors):
                raise SchemaValidationError("R1 scene transfer changed anchor identity")
            pose_pairs: list[tuple[Sequence[float], Sequence[float]]] = [
                (source_anchors[key], target_anchors[key])
                for key in sorted(source_anchors)
            ]
            checked_anchor_poses += len(pose_pairs)
            if (
                source_knot.centroidal_target is not None
                and target_knot.centroidal_target is not None
                and source_knot.centroidal_target.com_pos_world is not None
                and source_knot.centroidal_target.body_orientation_world is not None
                and target_knot.centroidal_target.com_pos_world is not None
                and target_knot.centroidal_target.body_orientation_world is not None
            ):
                pose_pairs.append(
                    (
                        (
                            *source_knot.centroidal_target.com_pos_world,
                            *source_knot.centroidal_target.body_orientation_world,
                        ),
                        (
                            *target_knot.centroidal_target.com_pos_world,
                            *target_knot.centroidal_target.body_orientation_world,
                        ),
                    )
                )
            for source_pose, target_pose in pose_pairs:
                source_relative = compose_pose(
                    inverse_pose(source_objects[0]), tuple(source_pose)
                )
                target_relative = compose_pose(
                    inverse_pose(target_objects[0]), tuple(target_pose)
                )
                maximum_relative_position_error = max(
                    maximum_relative_position_error,
                    math.dist(source_relative[:3], target_relative[:3]),
                )
                maximum_relative_attitude_error = max(
                    maximum_relative_attitude_error,
                    _quaternion_distance_rad(source_relative[3:], target_relative[3:]),
                )
    if (
        maximum_relative_position_error > 1.0e-7
        or maximum_relative_attitude_error > 1.0e-7
    ):
        raise SchemaValidationError("R1 scene transfer relative error is excessive")
    return {
        "audit_version": "order9_r1_planar_scene_transfer_invariant_audit_v4",
        "status": "accepted",
        "fixed_delta_pose_world": list(delta_pose_world),
        "checked_knot_count": checked_knots,
        "checked_free_anchor_pose_count": checked_anchor_poses,
        "joint_position_targets_exactly_preserved": True,
        "joint_velocity_targets_exactly_preserved": True,
        "contact_assignment_identities_exactly_preserved": True,
        "maximum_object_relative_position_error_m": maximum_relative_position_error,
        "maximum_object_relative_attitude_error_rad": (maximum_relative_attitude_error),
        "support_transformed_with_same_fixed_delta": True,
        "controller_layers_invoked": False,
        "isaac_invoked": False,
    }


def audit_order9_r1_scene_transfer_case(
    *,
    reference_case_root: str | Path,
    target_task_spec: TaskSpec,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Prove one complete saved case is unchanged by a planar scene move.

    This is the read-only, range-selection form of the scene-transfer audit.
    It authenticates the saved nominal set and its complete collision and task
    certificates, then checks every trajectory target against one fixed SE(2)
    transform.  It never runs IK, a controller, or a simulator.
    """

    repository = Path(repository_root).resolve()
    reference_root = Path(reference_case_root).resolve()
    target_task_spec.validate()
    set_path = reference_root / "nominal_set" / "manifest.json"
    nominal_set = validate_order9_c3_nominal_trajectory_set_bytes(
        set_path,
        repository_root=repository,
    )
    if len(nominal_set.entries) != 1:
        raise SchemaValidationError("R1 scene-transfer source set is not singular")
    artifact_path = set_path.parent / nominal_set.entries[0].artifact_path
    artifact = validate_order9_c3_nominal_trajectory_artifact_bytes(artifact_path)
    reference_task_path = reference_root / "task_spec.json"
    reference_task = TaskSpec.from_json(reference_task_path.read_text(encoding="utf-8"))
    transformed_task = transform_order9_r1_planar_task_scene(
        reference_task,
        target_identity_task=target_task_spec,
    )
    delta = _validated_task_scene_rigid_transform(reference_task, transformed_task)
    source_phases = {
        entry.phase: ContactWrenchTrajectory.from_json(
            (artifact_path.parent / entry.trajectory_path).read_text(encoding="utf-8")
        )
        for entry in artifact.phase_trajectories
    }
    if tuple(source_phases) != _PHASES:
        raise SchemaValidationError("R1 scene-transfer source phases differ")
    target_phases = {
        phase: transform_order9_r1_scene_trajectory(
            trajectory,
            delta_pose_world=delta,
        )
        for phase, trajectory in source_phases.items()
    }
    invariant = audit_order9_r1_scene_transfer_invariants(
        source_phases=source_phases,
        target_phases=target_phases,
        delta_pose_world=delta,
    )
    collision = _validated_scene_collision_reference(
        reference_root,
        source_phases=source_phases,
    )
    semantic = _validated_scene_semantic_reference(
        reference_root,
        target_phases=target_phases,
        delta_pose_world=delta,
    )
    reference_start, reference_goal = _object_start_and_goal(reference_task)
    target_start, target_goal = _object_start_and_goal(transformed_task)
    return {
        "audit_version": "order9_r1_planar_scene_case_audit_v15",
        "status": "accepted",
        "reference_case_root": _portable(reference_root, repository),
        "reference_task_spec_sha256": hash_file(reference_task_path),
        "reference_nominal_set_sha256": hash_file(set_path),
        "reference_nominal_artifact_sha256": hash_file(artifact_path),
        "reference_task_start_pose_world": list(reference_start),
        "reference_task_goal_pose_world": list(reference_goal),
        "target_task_start_pose_world": list(target_start),
        "target_task_goal_pose_world": list(target_goal),
        "source_support_pose_world": list(
            reference_task.scene.environment.support_surfaces[0].pose_world
        ),
        "target_support_pose_world": list(
            transformed_task.scene.environment.support_surfaces[0].pose_world
        ),
        "fixed_scene_delta_pose_world": list(delta),
        "invariant_audit": invariant,
        "collision_certificate": collision,
        "semantic_certificate": semantic,
        "controller_layers_invoked": False,
        "ik_invoked": False,
        "isaac_invoked": False,
        "trajectory_optimization_invoked": False,
        "training_eligible": False,
    }


def transform_order9_r1_scene_trajectory(
    trajectory: ContactWrenchTrajectory,
    *,
    delta_pose_world: Pose7D,
) -> ContactWrenchTrajectory:
    """Move every world target with the same fixed planar scene transform."""

    _require_planar_delta(delta_pose_world)
    source = ContactWrenchTrajectory.from_dict(trajectory.to_dict())
    source.validate()
    value = ContactWrenchTrajectory(
        horizon_s=float(source.horizon_s),
        dt_s=float(source.dt_s),
        knots=[
            _transform_knot(
                knot,
                delta_pose_world=delta_pose_world,
                yaw_rotation_world=None,
                rotate_contact_wrenches=(
                    source.contract_version == "implicit_world_v1"
                ),
            )
            for knot in source.knots
        ],
        derived_mode_label=(
            f"{source.derived_mode_label or 'unspecified'}:"
            f"{ORDER9_R1_PLANAR_SCENE_TRANSFER_VERSION}"
        ),
        contract_version=source.contract_version,
    )
    value.validate()
    return value


def transform_order9_r1_planar_trajectory(
    trajectory: ContactWrenchTrajectory,
    *,
    translation_world: Sequence[float],
    yaw_rotation_world: Pose7D,
) -> ContactWrenchTrajectory:
    """Move every world target with one object-centred planar pose change.

    R1 changes the start and goal x/y coordinates by the same world-frame
    translation and changes their yaw about their respective object centres.
    Applying that operation independently at every knot preserves the robot,
    grasp anchors, free anchors, and CoM targets relative to the object while
    leaving all joint coordinates byte-semantically unchanged.
    """

    if len(translation_world) != 3:
        raise ValueError("R1 planar translation must contain three values")
    translation = tuple(float(value) for value in translation_world)
    if (
        any(not math.isfinite(value) for value in translation)
        or abs(translation[2]) > 1.0e-12
    ):
        raise ValueError("R1 planar translation must be finite and horizontal")
    source = ContactWrenchTrajectory.from_dict(trajectory.to_dict())
    source.validate()
    translation_pose: Pose7D = (*translation, 0.0, 0.0, 0.0, 1.0)
    knots = []
    for knot in source.knots:
        object_poses = [
            tuple(value.pose_target_world)
            for value in knot.object_targets
            if value.pose_target_world is not None
        ]
        if len(object_poses) != 1:
            raise SchemaValidationError(
                "R1 planar transform requires one object pose per knot"
            )
        centered = _centered_yaw_delta(object_poses[0], yaw_rotation_world)
        delta = compose_pose(translation_pose, centered)
        knots.append(
            _transform_knot(
                knot,
                delta_pose_world=delta,
                yaw_rotation_world=None,
                rotate_contact_wrenches=(
                    source.contract_version == "implicit_world_v1"
                ),
            )
        )
    value = ContactWrenchTrajectory(
        horizon_s=float(source.horizon_s),
        dt_s=float(source.dt_s),
        knots=knots,
        derived_mode_label=(
            f"{source.derived_mode_label or 'unspecified'}:"
            f"{ORDER9_R1_PLANAR_NOMINAL_TRANSFER_VERSION}"
        ),
        contract_version=source.contract_version,
    )
    value.validate()
    return value


def transform_order9_r1_contact_candidates(
    candidates: ContactCandidateSet,
    *,
    delta_pose_world: Pose7D,
    target_task_id: str,
    target_morphology_graph_id: str,
    target_set_id: str,
) -> ContactCandidateSet:
    source = ContactCandidateSet.from_dict(candidates.to_dict())
    source.validate()
    rotation = transform_from_pose(delta_pose_world).rotation
    payload = source.to_dict()
    payload.update(
        {
            "task_id": target_task_id,
            "morphology_graph_id": target_morphology_graph_id,
            "set_id": target_set_id,
        }
    )
    for candidate in payload["candidates"]:
        candidate["contact_pose_world"] = list(
            compose_pose(delta_pose_world, tuple(candidate["contact_pose_world"]))
        )
        candidate["contact_frame_world"] = list(
            compose_pose(delta_pose_world, tuple(candidate["contact_frame_world"]))
        )
        candidate["normal_world"] = list(
            matvec(rotation, tuple(candidate["normal_world"]))
        )
        tangent = candidate["tangent_basis_world"]
        candidate["tangent_basis_world"] = list(
            matvec(rotation, tuple(tangent[:3])) + matvec(rotation, tuple(tangent[3:]))
        )
    value = ContactCandidateSet.from_dict(payload)
    value.validate()
    return value


def audit_order9_r1_yaw_branch_path(
    *,
    phases: Mapping[str, ContactWrenchTrajectory],
    task_spec: TaskSpec,
    source_task_spec: TaskSpec,
    morphology: MorphologyGraph,
    contact_candidate_set: ContactCandidateSet,
    physical_model_config_path: str | Path,
    anchor_position_tolerance_m: float = _ANCHOR_POSITION_TOLERANCE_M,
    anchor_attitude_tolerance_rad: float = _ANCHOR_ATTITUDE_TOLERANCE_RAD,
    inherited_collision_phase_names: frozenset[str] = frozenset(),
    inherit_object_relative_collision: bool = False,
    scene_delta_pose_world: Pose7D | None = None,
    inherited_scene_collision_clearance_m: float | None = None,
) -> dict[str, Any]:
    """Run the Isaac-free, optimization-free admission on every phase knot."""

    started = perf_counter()
    if (
        not math.isfinite(float(anchor_position_tolerance_m))
        or not 0.0 < float(anchor_position_tolerance_m) <= 0.030
        or not math.isfinite(float(anchor_attitude_tolerance_rad))
        or not 0.0 < float(anchor_attitude_tolerance_rad) <= 0.10
    ):
        raise ValueError("R1 lightweight audit anchor tolerance is invalid")
    if tuple(phases) != _PHASES:
        raise SchemaValidationError("R1 yaw audit phase order differs")
    if not inherited_collision_phase_names.issubset(_PHASES):
        raise ValueError("R1 lightweight inherited collision phases are invalid")
    if inherited_scene_collision_clearance_m is not None and (
        scene_delta_pose_world is None
        or not math.isfinite(float(inherited_scene_collision_clearance_m))
        or float(inherited_scene_collision_clearance_m) < 0.030 - 1.0e-12
    ):
        raise ValueError("R1 inherited scene collision certificate is invalid")
    physical_model = build_physical_model_from_config(physical_model_config_path)
    collision_object, support_contract = (
        build_order9_r1_frozen_support_collision_object(
            source_task_spec=source_task_spec,
            randomized_task_spec=task_spec,
        )
    )
    if scene_delta_pose_world is not None:
        _require_planar_delta(scene_delta_pose_world)
        source_support = _task_support_collision_box(source_task_spec)
        transformed_support = Order9PostureCollisionBox(
            box_id=source_support.box_id,
            size_m=source_support.size_m,
            pose_world=compose_pose(scene_delta_pose_world, source_support.pose_world),
        )
        collision_object = Order9PostureCollisionObject(
            object_id=collision_object.object_id,
            size_m=collision_object.size_m,
            initial_pose_world=collision_object.initial_pose_world,
            environment_boxes=(transformed_support,),
            ground_plane_z_m=collision_object.ground_plane_z_m,
        )
        support_contract = type(support_contract)(
            version=support_contract.version,
            support_geometry_hash=stable_hash(
                [
                    {
                        "box_id": box.box_id,
                        "size_m": list(box.size_m),
                        "pose_world": list(box.pose_world),
                    }
                    for box in collision_object.environment_boxes
                ]
            ),
            requested_clearance_m=support_contract.requested_clearance_m,
            minimum_admissible_clearance_m=(
                support_contract.minimum_admissible_clearance_m
            ),
        )
    solver = _default_posture_ik_solver(
        physical_model,
        collision_object=collision_object,
        prefer_native=True,
        require_native=True,
        nominal_collision_pair_manifest=None,
        ik_config=CentroidalPostureIKConfig(
            minimum_normalized_joint_limit_reserve=(_MINIMUM_NORMALIZED_JOINT_RESERVE)
        ),
        collision_margin_m=ORDER9_R1_ROBUST_SUPPORT_CLEARANCE_M,
    )
    ordered_ids = ordered_global_dock_joint_ids(morphology, physical_model)
    limits = _global_joint_limits(morphology, physical_model, ordered_ids)
    candidates = {
        int(value.candidate_id): value for value in contact_candidate_set.candidates
    }
    minimum_clearance = math.inf
    minimum_limit_margin = math.inf
    minimum_limit_fraction = math.inf
    maximum_tilt = 0.0
    maximum_anchor_position_error = 0.0
    maximum_anchor_attitude_error = 0.0
    checked_knots = 0
    checked_collision_scenes = 0
    phase_records = []
    for phase in _PHASES:
        trajectory = phases[phase]
        trajectory.validate()
        collision_inherited = (
            inherited_scene_collision_clearance_m is not None
            or phase in inherited_collision_phase_names
        )
        phase_minimum_clearance = math.inf
        for knot_index, knot in enumerate(trajectory.knots):
            checked_knots += 1
            centroidal = knot.centroidal_target
            posture = knot.posture_target
            if (
                centroidal is None
                or centroidal.com_pos_world is None
                or centroidal.body_orientation_world is None
                or posture is None
                or posture.joint_pos_target is None
            ):
                raise SchemaValidationError(
                    f"R1 yaw audit incomplete robot target at {phase}:{knot_index}"
                )
            q = {
                joint_id: float(posture.joint_pos_target[joint_id])
                for joint_id in ordered_ids
            }
            if set(posture.joint_pos_target) != set(ordered_ids):
                raise SchemaValidationError("R1 yaw audit joint identity differs")
            for joint_id, value in q.items():
                lower, upper = limits[joint_id]
                margin = min(value - lower, upper - value)
                fraction = margin / (upper - lower)
                minimum_limit_margin = min(minimum_limit_margin, margin)
                minimum_limit_fraction = min(minimum_limit_fraction, fraction)
                if fraction < _MINIMUM_NORMALIZED_JOINT_RESERVE - 1.0e-12:
                    raise SchemaValidationError(
                        "R1 yaw audit joint reserve is below one percent"
                    )
            tilt = _body_tilt_rad(centroidal.body_orientation_world)
            maximum_tilt = max(maximum_tilt, tilt)
            if tilt > _MAXIMUM_BODY_TILT_RAD + 1.0e-12:
                raise SchemaValidationError("R1 yaw audit body tilt is excessive")

            centroidal_pose: Pose7D = (
                *tuple(float(value) for value in centroidal.com_pos_world),
                *tuple(float(value) for value in centroidal.body_orientation_world),
            )
            assigned_anchor_ids = {
                int(assignment.anchor_id) for assignment in knot.contact_assignments
            }
            if (
                not collision_inherited
                and posture.free_anchor_pose_targets
                and assigned_anchor_ids
            ):
                if not assigned_anchor_ids.issubset(posture.free_anchor_pose_targets):
                    raise SchemaValidationError(
                        "R1 yaw audit assigned anchor lacks a pose target"
                    )
                anchor_ids = tuple(sorted(assigned_anchor_ids))
                references = resolve_mesh_backed_anchor_references(
                    morphology,
                    physical_model,
                    anchor_ids,
                )
                base_pose = solver.kinematics.base_pose_for_centroidal_target(
                    morphology=morphology,
                    q=q,
                    com_pos_world=centroidal.com_pos_world,
                    body_orientation_world=centroidal.body_orientation_world,
                )
                forward = solver.kinematics.forward(
                    morphology,
                    physical_model,
                    q,
                    base_pose,
                    references,
                )
                for anchor_id in anchor_ids:
                    expected = tuple(posture.free_anchor_pose_targets[anchor_id])
                    actual = forward.anchor_poses_world[anchor_id]
                    position_error = math.dist(expected[:3], actual[:3])
                    attitude_error = _quaternion_distance_rad(expected[3:], actual[3:])
                    maximum_anchor_position_error = max(
                        maximum_anchor_position_error, position_error
                    )
                    maximum_anchor_attitude_error = max(
                        maximum_anchor_attitude_error, attitude_error
                    )
                    if position_error > float(
                        anchor_position_tolerance_m
                    ) or attitude_error > float(anchor_attitude_tolerance_rad):
                        raise SchemaValidationError(
                            f"R1 yaw audit anchor target mismatch at "
                            f"{phase}:{knot_index}:{anchor_id} "
                            f"(position={position_error:.9g}, "
                            f"attitude={attitude_error:.9g})"
                        )

            object_targets = [
                value
                for value in knot.object_targets
                if value.object_id == collision_object.object_id
                and value.pose_target_world is not None
            ]
            if len(object_targets) != 1:
                raise SchemaValidationError("R1 yaw audit object target differs")
            allowed_anchors = set()
            for assignment in knot.contact_assignments:
                candidate = candidates.get(int(assignment.candidate_id))
                if candidate is None or int(candidate.anchor_id) != int(
                    assignment.anchor_id
                ):
                    raise SchemaValidationError(
                        "R1 yaw audit contact assignment is not in its candidate set"
                    )
                if (
                    candidate.target_entity_id == collision_object.object_id
                    and assignment.schedule_state
                    in {"attach", "maintain", "slide", "release"}
                ):
                    allowed_anchors.add(int(assignment.anchor_id))
            collision_scenes = (
                []
                if collision_inherited
                else [
                    *(
                        []
                        if inherit_object_relative_collision
                        else [
                            (
                                collision_object.object_id,
                                tuple(object_targets[0].pose_target_world),
                                collision_object.size_m,
                                tuple(sorted(allowed_anchors)),
                            )
                        ]
                    ),
                    *[
                        (box.box_id, box.pose_world, box.size_m, ())
                        for box in collision_object.environment_boxes
                    ],
                ]
            )
            for obstacle_id, pose, size, allowed in collision_scenes:
                solver.set_collision_scene(
                    morphology=morphology,
                    object_pose_world=pose,
                    object_size_m=size,
                    allowed_anchor_ids=allowed,
                )
                collision = solver.check_configuration(
                    morphology=morphology,
                    centroidal_pose_world=centroidal_pose,
                    joint_positions_rad=q,
                    exact=False,
                    margin_m=ORDER9_R1_ROBUST_SUPPORT_CLEARANCE_M,
                    ground_plane_z_m=collision_object.ground_plane_z_m,
                )
                checked_collision_scenes += 1
                clearance = float(collision["minimum_clearance_m"])
                minimum_clearance = min(minimum_clearance, clearance)
                phase_minimum_clearance = min(phase_minimum_clearance, clearance)
                if (
                    collision.get("accepted") is not True
                    or int(collision["violating_pair_count"]) != 0
                ):
                    raise SchemaValidationError(
                        f"R1 yaw audit collision at {phase}:{knot_index} "
                        f"against {obstacle_id} "
                        f"(clearance_m={clearance:.9g}, "
                        f"violating_pair_count={int(collision['violating_pair_count'])})"
                    )
        phase_records.append(
            {
                "phase": phase,
                "knot_count": len(trajectory.knots),
                "minimum_collision_clearance_m": (
                    None if collision_inherited else phase_minimum_clearance
                ),
                "collision_admission_inherited": collision_inherited,
            }
        )
    if inherited_scene_collision_clearance_m is not None:
        minimum_clearance = float(inherited_scene_collision_clearance_m)
    return {
        "audit_version": ORDER9_R1_YAW_BRANCH_LIGHTWEIGHT_AUDIT_VERSION,
        "status": "accepted",
        "checked_phases": list(_PHASES),
        "checked_knot_count": checked_knots,
        "checked_collision_scene_count": checked_collision_scenes,
        "inherited_collision_phase_names": sorted(inherited_collision_phase_names),
        "object_relative_collision_inherited": inherit_object_relative_collision,
        "complete_scene_collision_inherited": (
            inherited_scene_collision_clearance_m is not None
        ),
        "phase_records": phase_records,
        "minimum_collision_clearance_m": minimum_clearance,
        "minimum_joint_limit_margin_rad": minimum_limit_margin,
        "minimum_normalized_joint_limit_reserve": minimum_limit_fraction,
        "maximum_body_tilt_rad": maximum_tilt,
        "maximum_anchor_position_error_m": maximum_anchor_position_error,
        "maximum_anchor_attitude_error_rad": maximum_anchor_attitude_error,
        "anchor_position_tolerance_m": float(anchor_position_tolerance_m),
        "anchor_attitude_tolerance_rad": float(anchor_attitude_tolerance_rad),
        "requested_support_clearance_m": support_contract.requested_clearance_m,
        "support_geometry_hash": support_contract.support_geometry_hash,
        "minimum_admissible_support_clearance_m": (
            support_contract.minimum_admissible_clearance_m
        ),
        "isaac_invoked": False,
        "controller_layers_invoked": False,
        "ik_invoked": False,
        "trajectory_optimization_invoked": False,
        "wall_time_s": perf_counter() - started,
        "training_eligible": False,
    }


def _transform_knot(
    knot: InteractionKnot,
    *,
    delta_pose_world: Pose7D | None,
    yaw_rotation_world: Pose7D | None,
    rotate_contact_wrenches: bool,
) -> InteractionKnot:
    payload = knot.to_dict()
    if yaw_rotation_world is not None:
        object_poses = [
            tuple(value["pose_target_world"])
            for value in payload.get("object_targets", [])
            if value.get("pose_target_world") is not None
        ]
        if len(object_poses) != 1:
            raise SchemaValidationError(
                "R1 centered yaw transform requires one object pose per knot"
            )
        delta_pose_world = _centered_yaw_delta(object_poses[0], yaw_rotation_world)
    if delta_pose_world is None:
        raise AssertionError("R1 yaw transform mode was not resolved")
    rotation = transform_from_pose(delta_pose_world).rotation
    centroidal = payload.get("centroidal_target")
    if centroidal is not None:
        if (
            centroidal.get("com_pos_world") is not None
            and centroidal.get("body_orientation_world") is not None
        ):
            pose = compose_pose(
                delta_pose_world,
                tuple(centroidal["com_pos_world"])
                + tuple(centroidal["body_orientation_world"]),
            )
            centroidal["com_pos_world"] = list(pose[:3])
            centroidal["body_orientation_world"] = list(pose[3:])
        if centroidal.get("com_vel_world") is not None:
            centroidal["com_vel_world"] = list(
                matvec(rotation, tuple(centroidal["com_vel_world"]))
            )
        if centroidal.get("centroidal_wrench_preference") is not None:
            centroidal["centroidal_wrench_preference"] = list(
                _rotate_spatial6(centroidal["centroidal_wrench_preference"], rotation)
            )
    posture = payload.get("posture_target")
    if posture is not None and posture.get("free_anchor_pose_targets") is not None:
        posture["free_anchor_pose_targets"] = {
            key: list(compose_pose(delta_pose_world, tuple(value)))
            for key, value in posture["free_anchor_pose_targets"].items()
        }
    for target in payload.get("object_targets", []):
        if target.get("pose_target_world") is not None:
            target["pose_target_world"] = list(
                compose_pose(delta_pose_world, tuple(target["pose_target_world"]))
            )
        if target.get("twist_target_world") is not None:
            target["twist_target_world"] = list(
                _rotate_spatial6(target["twist_target_world"], rotation)
            )
    if rotate_contact_wrenches:
        for assignment in payload.get("contact_assignments", []):
            for key in ("wrench_target", "wrench_lower", "wrench_upper"):
                if assignment.get(key) is not None:
                    assignment[key] = list(_rotate_spatial6(assignment[key], rotation))
    value = InteractionKnot.from_dict(payload)
    return value


def _validated_task_yaw_rotation(
    reference: TaskSpec,
    target: TaskSpec,
) -> Pose7D:
    reference_start, reference_goal = _object_start_and_goal(reference)
    target_start, target_goal = _object_start_and_goal(target)
    if math.dist(reference_start[:3], target_start[:3]) > 1.0e-8:
        raise SchemaValidationError("R1 yaw repair changed object position")
    delta = compose_pose(target_start, inverse_pose(reference_start))
    rotation = transform_from_pose(delta).rotation
    yaw_delta = math.atan2(rotation[1][0], rotation[0][0])
    tilt_delta = math.acos(max(-1.0, min(1.0, rotation[2][2])))
    if (
        abs(yaw_delta) <= 1.0e-10
        or abs(yaw_delta) > _MAXIMUM_YAW_DELTA_RAD
        or tilt_delta > _MAXIMUM_OTHER_ROTATION_RAD
    ):
        raise SchemaValidationError("R1 yaw repair delta is not a bounded pure yaw")
    yaw_rotation: Pose7D = (0.0, 0.0, 0.0, *delta[3:])
    mapped_start = compose_pose(
        _centered_yaw_delta(reference_start, yaw_rotation), reference_start
    )
    mapped_goal = compose_pose(
        _centered_yaw_delta(reference_goal, yaw_rotation), reference_goal
    )
    if not _pose_close(
        mapped_start, target_start, position=1.0e-7, angle=1.0e-7
    ) or not _pose_close(mapped_goal, target_goal, position=1.0e-7, angle=1.0e-7):
        raise SchemaValidationError("R1 yaw repair does not preserve the task goal")
    return yaw_rotation


def _require_compatible_cases(
    *,
    reference_artifact: Order9C3NominalTrajectoryArtifact,
    target_artifact: Order9C3NominalTrajectoryArtifact,
    reference_task: TaskSpec,
    target_task: TaskSpec,
    allow_planar_translation: bool = False,
) -> None:
    if (
        reference_artifact.structural_hash != target_artifact.structural_hash
        or reference_artifact.physical_model_hash != target_artifact.physical_model_hash
        or reference_artifact.robot_urdf_sha256 != target_artifact.robot_urdf_sha256
        or reference_artifact.selected_surface_port_ids
        != target_artifact.selected_surface_port_ids
    ):
        raise SchemaValidationError("R1 yaw repair cases differ beyond task yaw")
    if allow_planar_translation:
        return
    for key in ("r1_initial_x_offset_m", "r1_initial_y_offset_m"):
        if not math.isclose(
            float(reference_task.metadata[key]),
            float(target_task.metadata[key]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise SchemaValidationError("R1 yaw repair position offsets differ")


def _object_start_and_goal(task: TaskSpec) -> tuple[Pose7D, Pose7D]:
    objects = [value for value in task.scene.objects if value.movable]
    goals = [
        value
        for value in task.goals
        if value.goal_type == "object_pose"
        and value.target_entity_id is not None
        and value.target_pose_world is not None
    ]
    if (
        len(objects) != 1
        or len(goals) != 1
        or goals[0].target_entity_id != objects[0].object_id
    ):
        raise SchemaValidationError("R1 yaw repair task object identity differs")
    return tuple(objects[0].pose_world), tuple(goals[0].target_pose_world)


def _validated_task_planar_change(
    reference: TaskSpec,
    target: TaskSpec,
) -> tuple[tuple[float, float, float], Pose7D]:
    reference_start, reference_goal = _object_start_and_goal(reference)
    target_start, target_goal = _object_start_and_goal(target)
    start_translation = tuple(
        float(target_start[index]) - float(reference_start[index]) for index in range(3)
    )
    goal_translation = tuple(
        float(target_goal[index]) - float(reference_goal[index]) for index in range(3)
    )
    if (
        math.dist(start_translation, goal_translation) > 1.0e-8
        or abs(start_translation[2]) > 1.0e-8
        or math.hypot(*start_translation[:2]) > math.sqrt(2.0) * 0.040 + 1.0e-8
    ):
        raise SchemaValidationError(
            "R1 planar transfer is not one bounded start/goal translation"
        )
    start_delta = compose_pose(target_start, inverse_pose(reference_start))
    goal_delta = compose_pose(target_goal, inverse_pose(reference_goal))
    start_rotation = transform_from_pose(start_delta).rotation
    goal_rotation = transform_from_pose(goal_delta).rotation
    start_yaw = math.atan2(start_rotation[1][0], start_rotation[0][0])
    goal_yaw = math.atan2(goal_rotation[1][0], goal_rotation[0][0])
    start_tilt = math.acos(max(-1.0, min(1.0, start_rotation[2][2])))
    goal_tilt = math.acos(max(-1.0, min(1.0, goal_rotation[2][2])))
    if (
        abs(start_yaw - goal_yaw) > 1.0e-8
        or abs(start_yaw) > math.radians(20.0) + 1.0e-9
        or max(start_tilt, goal_tilt) > _MAXIMUM_OTHER_ROTATION_RAD
    ):
        raise SchemaValidationError(
            "R1 planar transfer is not one bounded start/goal yaw change"
        )
    yaw_rotation: Pose7D = (0.0, 0.0, 0.0, *start_delta[3:])
    for source, expected in (
        (reference_start, target_start),
        (reference_goal, target_goal),
    ):
        mapped = compose_pose(
            (*start_translation, 0.0, 0.0, 0.0, 1.0),
            compose_pose(_centered_yaw_delta(source, yaw_rotation), source),
        )
        if not _pose_close(mapped, expected, position=1.0e-7, angle=1.0e-7):
            raise SchemaValidationError("R1 planar transfer does not map its task")
    return start_translation, yaw_rotation


def _bounded_planar_scene_delta(
    reference_start: Pose7D,
    target_start: Pose7D,
) -> Pose7D:
    if (
        abs(float(target_start[2]) - float(reference_start[2])) > 1.0e-8
        or math.dist(reference_start[:2], target_start[:2])
        > math.sqrt(2.0) * 0.040 + 1.0e-8
    ):
        raise SchemaValidationError("R1 scene transfer object displacement is invalid")
    delta = compose_pose(target_start, inverse_pose(reference_start))
    _require_planar_delta(delta)
    rotation = transform_from_pose(delta).rotation
    yaw = math.atan2(rotation[1][0], rotation[0][0])
    if abs(yaw) > _MAXIMUM_YAW_DELTA_RAD:
        raise SchemaValidationError("R1 scene transfer yaw delta is too large")
    return delta


def _require_planar_delta(delta_pose_world: Pose7D) -> None:
    if len(delta_pose_world) != 7 or any(
        not math.isfinite(float(value)) for value in delta_pose_world
    ):
        raise ValueError("R1 scene transform must be one finite pose")
    rotation = transform_from_pose(delta_pose_world).rotation
    tilt = math.acos(max(-1.0, min(1.0, rotation[2][2])))
    if abs(float(delta_pose_world[2])) > 1.0e-8 or tilt > _MAXIMUM_OTHER_ROTATION_RAD:
        raise SchemaValidationError(
            "R1 scene transform must be horizontal and yaw-only"
        )


def _require_scene_task_compatibility(
    reference: TaskSpec,
    target: TaskSpec,
) -> None:
    if (
        reference.task_type != target.task_type
        or reference.robot_constraints.to_dict() != target.robot_constraints.to_dict()
        or reference.safety.to_dict() != target.safety.to_dict()
        or [value.to_dict() for value in reference.scene.geometry_library]
        != [value.to_dict() for value in target.scene.geometry_library]
        or len(reference.scene.objects) != len(target.scene.objects)
        or len(reference.goals) != len(target.goals)
        or len(reference.scene.environment.support_surfaces)
        != len(target.scene.environment.support_surfaces)
        or len(reference.scene.environment.obstacles)
        != len(target.scene.environment.obstacles)
    ):
        raise SchemaValidationError("R1 scene transfer task structure differs")

    def without_pose(value: Any, key: str) -> dict[str, Any]:
        payload = value.to_dict()
        payload.pop(key, None)
        return payload

    if [without_pose(value, "pose_world") for value in reference.scene.objects] != [
        without_pose(value, "pose_world") for value in target.scene.objects
    ]:
        raise SchemaValidationError("R1 scene transfer object properties differ")
    if [without_pose(value, "target_pose_world") for value in reference.goals] != [
        without_pose(value, "target_pose_world") for value in target.goals
    ]:
        raise SchemaValidationError("R1 scene transfer goal contract differs")
    if [
        without_pose(value, "pose_world")
        for value in reference.scene.environment.support_surfaces
    ] != [
        without_pose(value, "pose_world")
        for value in target.scene.environment.support_surfaces
    ]:
        raise SchemaValidationError("R1 scene transfer support properties differ")
    if [
        without_pose(value, "pose_world")
        for value in reference.scene.environment.obstacles
    ] != [
        without_pose(value, "pose_world")
        for value in target.scene.environment.obstacles
    ]:
        raise SchemaValidationError("R1 scene transfer obstacle properties differ")


def _task_support_collision_box(task: TaskSpec) -> Order9PostureCollisionBox:
    surfaces = task.scene.environment.support_surfaces
    if len(surfaces) != 1:
        raise SchemaValidationError("R1 scene transfer requires one TaskSpec support")
    surface = surfaces[0]
    geometries = {value.geometry_id: value for value in task.scene.geometry_library}
    geometry = geometries.get(surface.geometry_id)
    raw_size = (
        None if geometry is None else (geometry.primitive_params or {}).get("size_m")
    )
    if (
        geometry is None
        or geometry.geometry_type.value != "box"
        or not isinstance(raw_size, list)
        or len(raw_size) != 3
    ):
        raise SchemaValidationError("R1 scene transfer support is not one box")
    return Order9PostureCollisionBox(
        box_id=surface.surface_id,
        size_m=tuple(
            float(raw_size[index]) * float(geometry.scale[index]) for index in range(3)
        ),
        pose_world=tuple(float(value) for value in surface.pose_world),
    )


def _validated_scene_collision_reference(
    reference_root: Path,
    *,
    source_phases: Mapping[str, ContactWrenchTrajectory],
) -> dict[str, Any]:
    """Validate the exact eight-phase certificate inherited by a rigid scene move."""

    audit_path = reference_root / "complete_task_planning_clearance_audit_v14.json"
    certificate_path = (
        reference_root / "support_clearance_generation_certificate_v14.json"
    )
    if not audit_path.is_file() or not certificate_path.is_file():
        raise SchemaValidationError(
            "R1 scene reference lacks its exact collision proof"
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    expected_knots = sum(len(source_phases[phase].knots) for phase in _PHASES)
    clearance = audit.get("minimum_support_or_self_clearance_m")
    if (
        audit.get("accepted") is not True
        or audit.get("status") != "accepted"
        or audit.get("formal_isaac_admission") is not True
        or audit.get("checked_phases") != list(_PHASES)
        or int(audit.get("checked_knot_count", -1)) != expected_knots
        or int(audit.get("failure_count", -1)) != 0
        or not isinstance(clearance, (int, float))
        or float(clearance) < 0.030 - 1.0e-12
        or certificate.get("persisted_artifact_checked") is not True
        or certificate.get("complete_phase_candidate_selection") is not True
        or certificate.get("unsupported_rigid_pose_copy") is not False
        or certificate.get("planning_audit_sha256") != hash_file(audit_path)
        or certificate.get("planning_audit_semantic_hash") != stable_hash(audit)
    ):
        raise SchemaValidationError("R1 scene reference collision proof is invalid")
    return {
        "audit_path": str(audit_path),
        "audit_sha256": hash_file(audit_path),
        "audit_semantic_hash": stable_hash(audit),
        "certificate_path": str(certificate_path),
        "certificate_sha256": hash_file(certificate_path),
        "checked_knot_count": expected_knots,
        "minimum_clearance_m": float(clearance),
        "rigid_planar_distance_invariance_used": True,
    }


def _validated_scene_semantic_reference(
    reference_root: Path,
    *,
    target_phases: Mapping[str, ContactWrenchTrajectory],
    delta_pose_world: Pose7D,
) -> dict[str, Any]:
    """Carry a complete-task proof through a distance-preserving scene move."""

    path = reference_root / "complete_task_semantic_audit.json"
    if not path.is_file():
        raise SchemaValidationError("R1 scene reference lacks its semantic proof")
    source = json.loads(path.read_text(encoding="utf-8"))
    expected_knots = sum(len(target_phases[phase].knots) for phase in _PHASES)
    if (
        source.get("status") != "accepted"
        or source.get("checked_phases") != list(_PHASES)
        or int(source.get("checked_knot_count", -1)) != expected_knots
        or abs(float(source.get("retreat_offset_m", math.nan)) - 0.10) > 1.0e-12
        or abs(float(source.get("measured_retreat_path_length_m", math.nan)) - 0.10)
        > 1.0e-9
    ):
        raise SchemaValidationError("R1 scene reference semantic proof is invalid")
    return {
        "audit_version": "order9_r1_planar_scene_semantic_inheritance_v1",
        "status": "accepted",
        "checked_phases": list(_PHASES),
        "checked_knot_count": expected_knots,
        "phase_boundary_count": 7,
        "retreat_offset_m": 0.10,
        "measured_retreat_path_length_m": float(
            source["measured_retreat_path_length_m"]
        ),
        "fixed_scene_delta_pose_world": list(delta_pose_world),
        "source_semantic_audit_path": str(path),
        "source_semantic_audit_sha256": hash_file(path),
        "source_semantic_audit_hash": stable_hash(source),
        "rigid_planar_length_and_phase_invariance_used": True,
        "controller_layers_invoked": False,
        "ik_resolve_invoked": False,
        "isaac_invoked": False,
        "trajectory_optimization_invoked": False,
        "training_eligible": False,
    }


def _validated_task_scene_rigid_transform(
    reference: TaskSpec,
    target: TaskSpec,
) -> Pose7D:
    """Return and verify the one transform shared by the complete task scene."""

    _require_scene_task_compatibility(reference, target)
    reference_start, _reference_goal = _object_start_and_goal(reference)
    target_start, _target_goal = _object_start_and_goal(target)
    delta = _bounded_planar_scene_delta(reference_start, target_start)
    pose_groups = (
        (
            [value.pose_world for value in reference.scene.objects],
            [value.pose_world for value in target.scene.objects],
        ),
        (
            [
                value.pose_world
                for value in reference.scene.environment.support_surfaces
            ],
            [value.pose_world for value in target.scene.environment.support_surfaces],
        ),
        (
            [value.pose_world for value in reference.scene.environment.obstacles],
            [value.pose_world for value in target.scene.environment.obstacles],
        ),
        (
            [value.target_pose_world for value in reference.goals],
            [value.target_pose_world for value in target.goals],
        ),
    )
    for source_poses, target_poses in pose_groups:
        for source_pose, target_pose in zip(source_poses, target_poses):
            if (source_pose is None) != (target_pose is None):
                raise SchemaValidationError("R1 scene transfer pose presence differs")
            if source_pose is None or target_pose is None:
                continue
            expected = compose_pose(delta, tuple(source_pose))
            if not _pose_close(expected, target_pose, position=1.0e-7, angle=1.0e-7):
                raise SchemaValidationError(
                    "R1 scene transfer did not move every task pose together"
                )
    if (
        target.metadata.get("r1_support_pose_transformed_with_scene") is not True
        or target.metadata.get("r1_robot_reset_transformed_with_scene") is not True
    ):
        raise SchemaValidationError("R1 scene transfer metadata is incomplete")
    return delta


def _centered_yaw_delta(
    object_pose_world: Sequence[float],
    yaw_rotation_world: Pose7D,
) -> Pose7D:
    """Return a transform that rotates orientation around the object's center."""

    source: Pose7D = tuple(float(value) for value in object_pose_world)  # type: ignore[assignment]
    rotation_only = compose_pose(
        yaw_rotation_world,
        (0.0, 0.0, 0.0, *source[3:]),
    )
    target: Pose7D = (*source[:3], *rotation_only[3:])
    return compose_pose(target, inverse_pose(source))


def _complete_timeline(
    phases: Mapping[str, ContactWrenchTrajectory],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    offset_s = 0.0
    for phase in _PHASES:
        trajectory = phases[phase]
        for knot_index, knot in enumerate(trajectory.knots):
            if records and knot_index == 0:
                continue
            records.append(
                {
                    "sample_index": len(records),
                    "global_time_s": offset_s + float(knot.t_rel_s),
                    "phase": phase,
                    "phase_local_time_s": float(knot.t_rel_s),
                    "phase_target_reached": True,
                    "knot": knot.to_dict(),
                }
            )
        offset_s += float(trajectory.horizon_s)
    return records


def _transform_reset_bank(
    *,
    source_path: Path,
    destination_path: Path,
    yaw_rotation_world: Pose7D,
    translation_world: Sequence[float] = (0.0, 0.0, 0.0),
    delta_pose_world: Pose7D | None = None,
    morphology_hash: str,
    task_spec_hash: str,
    nominal_set_path: Path,
    nominal_set_sha256: str,
    nominal_artifact_path: Path,
    nominal_artifact_sha256: str,
    timeline_sha256: str,
) -> None:
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("states"), list):
        raise SchemaValidationError("R1 yaw repair reset bank is invalid")
    rotation_pose = yaw_rotation_world if delta_pose_world is None else delta_pose_world
    rotation = torch.tensor(
        transform_from_pose(rotation_pose).rotation, dtype=torch.float32
    )
    for state in payload["states"]:
        pose_pairs = (
            ("robot_root_pose_local", "object_pose_local"),
            ("object_pose_local", "object_pose_local"),
            ("task_body_start_pose_local", "task_object_start_pose_local"),
            ("task_object_start_pose_local", "task_object_start_pose_local"),
            ("body_pose_local", "reference_object_pose_local"),
            ("reference_object_pose_local", "reference_object_pose_local"),
        )
        original = {
            key: tuple(float(value) for value in state[key].tolist())
            for key in {value for pair in pose_pairs for value in pair}
        }
        for key, center_key in pose_pairs:
            pose_delta = (
                delta_pose_world
                if delta_pose_world is not None
                else compose_pose(
                    (
                        *tuple(float(value) for value in translation_world),
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ),
                    _centered_yaw_delta(original[center_key], yaw_rotation_world),
                )
            )
            state[key] = torch.tensor(
                compose_pose(
                    pose_delta,
                    original[key],
                ),
                dtype=state[key].dtype,
            )
        for key in (
            "robot_root_twist",
            "object_twist",
            "body_twist",
            "reference_object_twist",
        ):
            value = state[key]
            state[key] = torch.cat(
                (rotation.to(value) @ value[:3], rotation.to(value) @ value[3:])
            )
    contract = payload["contract"]
    contract["morphology_graph_hash"] = morphology_hash
    contract["task_spec_hash"] = task_spec_hash
    nominal = contract["c3_nominal_reference"]
    nominal.update(
        {
            "set_manifest_path": str(nominal_set_path),
            "set_manifest_sha256": nominal_set_sha256,
            "artifact_path": str(nominal_artifact_path),
            "artifact_sha256": nominal_artifact_sha256,
            "timeline_sha256": timeline_sha256,
        }
    )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination_path)


def _source_bucket_task_path(case_root: Path, repository: Path) -> Path:
    case = json.loads((case_root / "case_manifest.json").read_text(encoding="utf-8"))
    morphology_path = repository / case["morphology_graph"]["path"]
    task_path = morphology_path.with_name("task_spec.json")
    if not task_path.is_file():
        raise FileNotFoundError(task_path)
    return task_path


def _candidate_id(case_root: Path) -> str:
    payload = json.loads((case_root / "case_manifest.json").read_text(encoding="utf-8"))
    value = str(payload.get("candidate_id", ""))
    if not value:
        raise SchemaValidationError("R1 yaw repair candidate id is empty")
    return value


def _rotate_spatial6(
    values: Sequence[float],
    rotation,
) -> tuple[float, ...]:
    if len(values) != 6:
        raise SchemaValidationError("R1 yaw repair spatial vector is not 6D")
    return tuple(
        matvec(rotation, tuple(float(value) for value in values[:3]))
        + matvec(rotation, tuple(float(value) for value in values[3:]))
    )


def _body_tilt_rad(quaternion: Sequence[float]) -> float:
    rotation = transform_from_pose((0.0, 0.0, 0.0, *quaternion)).rotation
    return math.acos(max(-1.0, min(1.0, rotation[2][2])))


def _quaternion_distance_rad(left: Sequence[float], right: Sequence[float]) -> float:
    dot = abs(sum(float(a) * float(b) for a, b in zip(left, right)))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot / (left_norm * right_norm))))


def _pose_close(
    left: Sequence[float],
    right: Sequence[float],
    *,
    position: float,
    angle: float,
) -> bool:
    return (
        math.dist(left[:3], right[:3]) <= position
        and _quaternion_distance_rad(left[3:], right[3:]) <= angle
    )


def _write_json_text(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _portable(path: Path, repository: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository.resolve()))
    except ValueError:
        return str(path.resolve())


__all__ = [
    "ORDER9_R1_PLANAR_NOMINAL_TRANSFER_VERSION",
    "ORDER9_R1_PLANAR_SCENE_TRANSFER_VERSION",
    "ORDER9_R1_YAW_BRANCH_LIGHTWEIGHT_AUDIT_VERSION",
    "ORDER9_R1_YAW_BRANCH_REPAIR_VERSION",
    "audit_order9_r1_yaw_branch_path",
    "audit_order9_r1_planar_transfer_invariants",
    "audit_order9_r1_scene_transfer_case",
    "audit_order9_r1_scene_transfer_invariants",
    "derive_order9_r1_planar_nominal_transfer_case",
    "derive_order9_r1_yaw_branch_case",
    "transform_order9_r1_contact_candidates",
    "transform_order9_r1_planar_trajectory",
    "transform_order9_r1_planar_task_scene",
    "transform_order9_r1_scene_trajectory",
    "transform_order9_r1_trajectory",
]
