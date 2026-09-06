from __future__ import annotations

"""Held-out R1 confirmation over a complete planar scene transform."""

from copy import deepcopy
from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from amsrr.geometry.pose_math import (
    compose_pose,
    inverse_pose,
    pose_from_transform,
    pose_to_xyz_rpy,
    transform_from_pose,
    transform_from_xyz_rpy,
)
from amsrr.schemas.common import Pose7D, SchemaValidationError
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.order3 import (
    Order3MorphologyPoolEntry,
    Order3MorphologyPoolManifest,
)
from amsrr.schemas.policies import ContactWrenchTrajectory
from amsrr.schemas.task_spec import TaskSpec
from amsrr.simulation.order9_morphology_assets import (
    Order9MorphologyAssetEntry,
    Order9MorphologyAssetManifest,
)
from amsrr.training.order9_c3_nominal_trajectory import (
    validate_order9_c3_nominal_trajectory_artifact_bytes,
    validate_order9_c3_nominal_trajectory_set_bytes,
)
from amsrr.training.order9_r1_yaw_branch_repair import (
    _object_start_and_goal,
    _portable,
    _validated_scene_collision_reference,
    _validated_scene_semantic_reference,
    audit_order9_r1_scene_transfer_invariants,
    transform_order9_r1_planar_task_scene,
    transform_order9_r1_scene_trajectory,
)
from amsrr.training.order9_rollout_buckets import Order9PiLRolloutBucket
from amsrr.utils.hashing import hash_file

ORDER9_R1_HELD_OUT_CONFIRMATION_V16_VERSION = (
    "order9_r1_held_out_scene_confirmation_v16"
)
ORDER9_R1_HELD_OUT_SPLIT = "held_out"
ORDER9_R1_HELD_OUT_BUCKET_COUNT = 14
ORDER9_R1_HELD_OUT_PER_MODULE_COUNT = 2
ORDER9_R1_HELD_OUT_SAMPLE_COUNT = 31
ORDER9_R1_HELD_OUT_LATTICE_COUNT = 27
ORDER9_R1_HELD_OUT_INTERIOR_COUNT = 4
_PHASES = (
    "approach",
    "contact_acquisition",
    "lift",
    "transport",
    "place",
    "release",
    "retreat",
    "settle",
)


@dataclass(frozen=True)
class Order9R1HeldOutPairV16:
    held_out_index: int
    held_out_entry: Order3MorphologyPoolEntry
    asset_entry: Order9MorphologyAssetEntry
    paired_train_bucket: Order9PiLRolloutBucket

    @property
    def bucket_id(self) -> str:
        return (
            f"held-out-{self.held_out_index:06d}-"
            f"{self.held_out_entry.structural_hash[:12]}"
        )


def select_order9_r1_held_out_pairs_v16(
    *,
    pool: Order3MorphologyPoolManifest,
    assets: Order9MorphologyAssetManifest,
    train_buckets: Sequence[Order9PiLRolloutBucket],
) -> tuple[Order9R1HeldOutPairV16, ...]:
    """Bind two never-used morphologies to two fixed tasks per module count."""

    held_out = [
        (index, entry)
        for index, entry in enumerate(pool.entries)
        if entry.split == DatasetSplit.HELD_OUT
    ]
    if len(held_out) != ORDER9_R1_HELD_OUT_BUCKET_COUNT:
        raise SchemaValidationError("R1 held-out morphology count differs")
    by_module: dict[int, list[Order9PiLRolloutBucket]] = {}
    for bucket in train_buckets:
        if bucket.split != DatasetSplit.TRAIN:
            raise SchemaValidationError("R1 held-out pairing received a non-train task")
        by_module.setdefault(int(bucket.module_count), []).append(bucket)
    pairs = []
    used_train_ids: set[str] = set()
    for module_count in range(2, 9):
        entries = [item for item in held_out if item[1].module_count == module_count]
        buckets = sorted(
            by_module.get(module_count, ()), key=lambda item: item.bucket_id
        )
        if (
            len(entries) != ORDER9_R1_HELD_OUT_PER_MODULE_COUNT
            or len(buckets) < ORDER9_R1_HELD_OUT_PER_MODULE_COUNT
        ):
            raise SchemaValidationError("R1 held-out module stratum differs")
        for (held_out_index, entry), bucket in zip(
            sorted(entries, key=lambda item: item[1].structural_hash),
            buckets[:ORDER9_R1_HELD_OUT_PER_MODULE_COUNT],
        ):
            asset = assets.entry_for(entry.morphology_graph)
            if (
                asset.split != DatasetSplit.HELD_OUT
                or asset.structural_hash != entry.structural_hash
                or asset.source_morphology_hash != entry.morphology_graph.stable_hash()
                or asset.module_count != module_count
                or bucket.bucket_id in used_train_ids
            ):
                raise SchemaValidationError("R1 held-out pairing provenance differs")
            used_train_ids.add(bucket.bucket_id)
            pairs.append(
                Order9R1HeldOutPairV16(
                    held_out_index=held_out_index,
                    held_out_entry=entry,
                    asset_entry=asset,
                    paired_train_bucket=bucket,
                )
            )
    if len(pairs) != ORDER9_R1_HELD_OUT_BUCKET_COUNT:
        raise SchemaValidationError("R1 held-out pairing is incomplete")
    return tuple(sorted(pairs, key=lambda item: item.bucket_id))


def build_order9_r1_held_out_source_bucket_v16(
    pair: Order9R1HeldOutPairV16,
    *,
    repository_root: str | Path,
    source_manifest_path: str | Path,
) -> Order9PiLRolloutBucket:
    repository = Path(repository_root).resolve()
    source_manifest = Path(source_manifest_path).resolve()
    graph_path = (repository / pair.asset_entry.morphology_graph_path).resolve()
    usd_path = (repository / pair.asset_entry.usd_path).resolve()
    if (
        hash_file(graph_path) != pair.asset_entry.morphology_graph_sha256
        or hash_file(usd_path) != pair.asset_entry.usd_sha256
    ):
        raise SchemaValidationError("R1 held-out asset bytes changed")
    source = pair.paired_train_bucket
    metadata = deepcopy(source.metadata)
    metadata.pop("accepted_nominal_trajectory", None)
    metadata.update(
        {
            "r1_held_out_confirmation_version": (
                ORDER9_R1_HELD_OUT_CONFIRMATION_V16_VERSION
            ),
            "r1_held_out_source_split": ORDER9_R1_HELD_OUT_SPLIT,
            "r1_held_out_pool_index": pair.held_out_index,
            "r1_held_out_pool_structural_hash": (pair.held_out_entry.structural_hash),
            "r1_held_out_paired_train_bucket_id": source.bucket_id,
        }
    )
    value = Order9PiLRolloutBucket(
        bucket_id=pair.bucket_id,
        split=DatasetSplit.HELD_OUT,
        seed=int(pair.held_out_entry.requested_seed),
        sample_index=int(pair.held_out_index),
        task_id=source.task_id,
        task_spec_path=source.task_spec_path,
        task_spec_sha256=source.task_spec_sha256,
        morphology_graph_path=os.path.relpath(graph_path, source_manifest.parent),
        morphology_graph_sha256=pair.asset_entry.morphology_graph_sha256,
        morphology_hash=pair.held_out_entry.morphology_graph.stable_hash(),
        structural_hash=pair.held_out_entry.structural_hash,
        module_count=pair.held_out_entry.module_count,
        robot_usd_path=pair.asset_entry.usd_path,
        robot_usd_sha256=pair.asset_entry.usd_sha256,
        selected_gripper_friction=source.selected_gripper_friction,
        contact_stiffness_n_per_m=source.contact_stiffness_n_per_m,
        contact_damping_n_s_per_m=source.contact_damping_n_s_per_m,
        estimated_mass_kg=source.estimated_mass_kg,
        estimated_inertia_body=list(source.estimated_inertia_body),
        estimated_com_object=list(source.estimated_com_object),
        randomization_version=source.randomization_version,
        topology_source="held_out_pool_final_confirmation_v16",
        metadata=metadata,
    )
    value.validate()
    return value


def relabel_order9_r1_held_out_task_v16(
    task: TaskSpec,
    *,
    candidate_id: str,
    bucket_id: str,
    structural_hash: str,
    paired_train_bucket_id: str,
) -> TaskSpec:
    payload = task.to_dict()
    payload["task_id"] = f"{task.task_id}:r1-held-out:{structural_hash[:12]}"
    metadata = deepcopy(payload.get("metadata", {}))
    metadata.update(
        {
            "order9_rollout_bucket_id": candidate_id,
            "dataset_split": ORDER9_R1_HELD_OUT_SPLIT,
            "r1_calibration_candidate_id": candidate_id,
            "r1_held_out_confirmation_version": (
                ORDER9_R1_HELD_OUT_CONFIRMATION_V16_VERSION
            ),
            "r1_held_out_source_split": ORDER9_R1_HELD_OUT_SPLIT,
            "r1_held_out_bucket_id": bucket_id,
            "r1_held_out_structural_hash": structural_hash,
            "r1_held_out_paired_train_bucket_id": paired_train_bucket_id,
            "r1_formal_teacher_collection_eligible": False,
        }
    )
    payload["metadata"] = metadata
    value = TaskSpec.from_dict(payload)
    value.validate()
    return value


def _midpoint_object_pose(reference: Pose7D, target: Pose7D) -> Pose7D:
    reference_xyz, reference_rpy = pose_to_xyz_rpy(reference)
    target_xyz, target_rpy = pose_to_xyz_rpy(target)
    if (
        abs(reference_xyz[2] - target_xyz[2]) > 1.0e-8
        or math.dist(reference_rpy[:2], target_rpy[:2]) > 1.0e-8
    ):
        raise SchemaValidationError("R1 held-out target is not planar")
    yaw_delta = math.atan2(
        math.sin(target_rpy[2] - reference_rpy[2]),
        math.cos(target_rpy[2] - reference_rpy[2]),
    )
    midpoint_xyz = (
        0.5 * (reference_xyz[0] + target_xyz[0]),
        0.5 * (reference_xyz[1] + target_xyz[1]),
        reference_xyz[2],
    )
    midpoint_rpy = (
        reference_rpy[0],
        reference_rpy[1],
        reference_rpy[2] + 0.5 * yaw_delta,
    )
    return pose_from_transform(transform_from_xyz_rpy(midpoint_xyz, midpoint_rpy))


def build_order9_r1_held_out_composed_target_v16(
    reference_task: TaskSpec,
    *,
    target_identity_task: TaskSpec,
) -> tuple[TaskSpec, TaskSpec, Pose7D, Pose7D, Pose7D]:
    """Construct ±40 mm/±20 degree targets using two proof substeps.

    The two substeps exist only while constructing and checking the artifact.
    They are not two temporal robot motions and do not add a trajectory phase.
    """

    reference_start, _reference_goal = _object_start_and_goal(reference_task)
    target_start, _target_goal = _object_start_and_goal(target_identity_task)
    midpoint_pose = _midpoint_object_pose(reference_start, target_start)
    midpoint_payload = target_identity_task.to_dict()
    midpoint_payload["task_id"] = f"{target_identity_task.task_id}:proof-midpoint"
    midpoint_payload["scene"]["objects"][0]["pose_world"] = list(midpoint_pose)
    midpoint_metadata = deepcopy(midpoint_payload.get("metadata", {}))
    midpoint_metadata["r1_calibration_candidate_id"] = (
        f"{target_identity_task.metadata['r1_calibration_candidate_id']}__proof_midpoint"
    )
    midpoint_payload["metadata"] = midpoint_metadata
    midpoint_identity = TaskSpec.from_dict(midpoint_payload)
    midpoint_identity.validate()
    midpoint_task = transform_order9_r1_planar_task_scene(
        reference_task,
        target_identity_task=midpoint_identity,
    )
    final_task = transform_order9_r1_planar_task_scene(
        midpoint_task,
        target_identity_task=target_identity_task,
    )
    midpoint_start, _ = _object_start_and_goal(midpoint_task)
    final_start, _ = _object_start_and_goal(final_task)
    delta_one = compose_pose(midpoint_start, inverse_pose(reference_start))
    delta_two = compose_pose(final_start, inverse_pose(midpoint_start))
    overall = compose_pose(final_start, inverse_pose(reference_start))
    composed = compose_pose(delta_two, delta_one)
    composed_rotation = transform_from_pose(composed).rotation
    overall_rotation = transform_from_pose(overall).rotation
    maximum_rotation_matrix_error = max(
        abs(composed_rotation[row][column] - overall_rotation[row][column])
        for row in range(3)
        for column in range(3)
    )
    if (
        math.dist(composed[:3], overall[:3]) > 1.0e-8
        or maximum_rotation_matrix_error > 1.0e-8
        or math.dist(final_start[:3], target_start[:3]) > 1.0e-8
    ):
        raise SchemaValidationError("R1 held-out composed scene transform differs")
    # Each construction substep was already checked by
    # transform_order9_r1_planar_task_scene.  Do not re-apply its 10 degree
    # per-substep limit to the composed 20 degree result.
    return midpoint_task, final_task, delta_one, delta_two, overall


def audit_order9_r1_held_out_composed_scene_case_v16(
    *,
    reference_case_root: str | Path,
    target_identity_task: TaskSpec,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Authenticate one reference and prove one selected-range target."""

    repository = Path(repository_root).resolve()
    reference_root = Path(reference_case_root).resolve()
    reference_task_path = reference_root / "task_spec.json"
    reference_task = TaskSpec.from_json(reference_task_path.read_text(encoding="utf-8"))
    midpoint_task, final_task, delta_one, delta_two, overall = (
        build_order9_r1_held_out_composed_target_v16(
            reference_task,
            target_identity_task=target_identity_task,
        )
    )
    set_path = reference_root / "nominal_set" / "manifest.json"
    nominal_set = validate_order9_c3_nominal_trajectory_set_bytes(
        set_path,
        repository_root=repository,
    )
    if len(nominal_set.entries) != 1:
        raise SchemaValidationError("R1 held-out reference set is not singular")
    artifact_path = set_path.parent / nominal_set.entries[0].artifact_path
    artifact = validate_order9_c3_nominal_trajectory_artifact_bytes(artifact_path)
    source_phases = {
        entry.phase: ContactWrenchTrajectory.from_json(
            (artifact_path.parent / entry.trajectory_path).read_text(encoding="utf-8")
        )
        for entry in artifact.phase_trajectories
    }
    if tuple(source_phases) != _PHASES:
        raise SchemaValidationError("R1 held-out reference phases differ")
    target_phases = {
        phase: transform_order9_r1_scene_trajectory(
            trajectory,
            delta_pose_world=overall,
        )
        for phase, trajectory in source_phases.items()
    }
    invariant = audit_order9_r1_scene_transfer_invariants(
        source_phases=source_phases,
        target_phases=target_phases,
        delta_pose_world=overall,
    )
    collision = _validated_scene_collision_reference(
        reference_root,
        source_phases=source_phases,
    )
    semantic = _validated_scene_semantic_reference(
        reference_root,
        target_phases=target_phases,
        delta_pose_world=overall,
    )
    reference_start, reference_goal = _object_start_and_goal(reference_task)
    final_start, final_goal = _object_start_and_goal(final_task)
    return {
        "audit_version": ORDER9_R1_HELD_OUT_CONFIRMATION_V16_VERSION,
        "status": "accepted",
        "reference_case_root": _portable(reference_root, repository),
        "reference_task_spec_sha256": hash_file(reference_task_path),
        "reference_nominal_set_sha256": hash_file(set_path),
        "reference_nominal_artifact_sha256": hash_file(artifact_path),
        "reference_task_start_pose_world": list(reference_start),
        "reference_task_goal_pose_world": list(reference_goal),
        "target_task_start_pose_world": list(final_start),
        "target_task_goal_pose_world": list(final_goal),
        "target_task_spec": final_task.to_dict(),
        "proof_substep_count": 2,
        "proof_substep_one_delta_pose_world": list(delta_one),
        "proof_substep_two_delta_pose_world": list(delta_two),
        "fixed_scene_delta_pose_world": list(overall),
        "proof_substeps_are_not_temporal_trajectory_phases": True,
        "invariant_audit": invariant,
        "collision_certificate": collision,
        "semantic_certificate": semantic,
        "support_transformed_with_scene": True,
        "controller_layers_invoked": False,
        "ik_invoked": False,
        "isaac_invoked": False,
        "trajectory_optimization_invoked": False,
        "training_eligible": False,
    }


__all__ = [
    "ORDER9_R1_HELD_OUT_BUCKET_COUNT",
    "ORDER9_R1_HELD_OUT_CONFIRMATION_V16_VERSION",
    "ORDER9_R1_HELD_OUT_INTERIOR_COUNT",
    "ORDER9_R1_HELD_OUT_LATTICE_COUNT",
    "ORDER9_R1_HELD_OUT_SAMPLE_COUNT",
    "Order9R1HeldOutPairV16",
    "audit_order9_r1_held_out_composed_scene_case_v16",
    "build_order9_r1_held_out_composed_target_v16",
    "build_order9_r1_held_out_source_bucket_v16",
    "relabel_order9_r1_held_out_task_v16",
    "select_order9_r1_held_out_pairs_v16",
]
