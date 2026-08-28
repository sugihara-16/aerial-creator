from __future__ import annotations

"""Validated proposal/approval contract for Order-9 R1 envelope calibration."""

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

from amsrr.schemas.common import SchemaBase, SchemaValidationError
from amsrr.schemas.order9 import Order9ArtifactBinding
from amsrr.training.order9_r1_randomization import (
    ORDER9_R1_OBJECT_DISTRIBUTION,
    ORDER9_R1_REQUIRED_MODULE_COUNTS,
)
from amsrr.training.order9_r1_fast_screen import (
    ORDER9_R1_FAST_SCREEN_EXECUTION_MODEL,
    ORDER9_R1_FAST_SCREEN_PHASES,
    ORDER9_R1_FAST_SCREEN_VERSION,
)
from amsrr.utils.config import load_config
from amsrr.utils.hashing import hash_file

ORDER9_R1_CALIBRATION_PROTOCOL_VERSION = "order9_r1_calibration_protocol_v1"
ORDER9_R1_CALIBRATION_APPROVAL_RECORD_VERSION = (
    "order9_r1_calibration_approval_record_v1"
)
ORDER9_R1_CALIBRATION_LATTICE_POINT_COUNT = 27

_REQUIRED_APPROVAL_SCOPE = {
    "four_level_numeric_ladder",
    "selection_on_22_train_buckets",
    "single_confirmation_on_14_validation_buckets",
    "confirmation_failure_rejects_v1",
    "fast_screen_before_full_control",
    "one_percent_joint_limit_reserve",
    "sixty_degree_body_tilt_ceiling",
    "formal_collection_140_episodes",
}


@dataclass
class Order9R1CalibrationLevel(SchemaBase):
    level_id: str
    position_half_span_m: float
    yaw_half_span_rad: float

    def validate(self) -> None:
        if not self.level_id:
            raise SchemaValidationError("Order9 R1 calibration level id is empty")
        for name in ("position_half_span_m", "yaw_half_span_rad"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise SchemaValidationError(
                    f"Order9 R1 calibration level {name} must be finite and positive"
                )


@dataclass
class Order9R1CalibrationProtocol(SchemaBase):
    protocol_version: str
    calibration_gate_id: str
    status: Literal["proposed", "approved", "rejected"]
    approval_record: str | None
    stage_id: str
    distribution_id: str
    source_c3_release_id: str
    source_c3_checkpoint_sha256: str
    source_c3_promotion_manifest_sha256: str
    curriculum_schedule_hash: str
    physical_model_hash: str
    source_bucket_manifest: Order9ArtifactBinding
    source_bucket_count: int
    source_module_counts: tuple[int, ...]
    selection_split: str
    selection_bucket_count: int
    confirmation_split: str
    confirmation_bucket_count: int
    confirmation_failure_invalidates_protocol: bool
    coordinate_frame: str
    envelope_center: str
    sampling_method: str
    support_geometry_preserved: bool
    support_pose_preserved: bool
    robot_reset_preserved: bool
    calibration_seed: int
    case_enumeration: str
    random_interior_seed_derivation: str
    calibration_levels: list[Order9R1CalibrationLevel]
    lattice_point_count_per_bucket: int
    random_interior_samples_per_bucket: int
    fast_screen_version: str
    fast_screen_execution_model: str
    fast_screen_phases: tuple[str, ...]
    fast_screen_reuses_resolved_trajectory: bool
    fast_screen_isaac_forbidden: bool
    fast_screen_controller_layers_forbidden: bool
    fast_screen_ik_resolve_forbidden: bool
    fast_screen_trajectory_optimization_forbidden: bool
    fast_screen_applies_to_calibration_and_collection: bool
    full_control_test_requires_fast_screen_pass: bool
    minimum_normalized_joint_limit_reserve: float
    maximum_body_tilt_rad: float
    isaac_replays_per_lattice_case: int
    isaac_replays_per_interior_case: int
    minimum_lattice_pass_rate: float
    minimum_teacher_feasibility_rate: float
    minimum_fast_screen_pass_rate: float
    minimum_isaac_success_rate: float
    minimum_interior_teacher_feasibility_rate: float
    minimum_interior_fast_screen_pass_rate: float
    minimum_interior_isaac_success_rate: float
    minimum_support_projected_com_margin_m: float
    maximum_safety_failure_count: int
    maximum_fallback_rate: float
    stop_on_first_failed_level: bool
    accept_largest_consecutive_passing_level: bool
    extrapolation_beyond_ladder_forbidden: bool
    calibration_records_training_eligible: bool
    collection_episode_count: int
    collection_train_episode_count: int
    collection_validation_episode_count: int
    collection_held_out_episode_count: int
    collection_per_module_episode_count: int
    collection_train_per_module_episode_count: int
    collection_validation_per_module_episode_count: int
    collection_held_out_per_module_episode_count: int
    collection_seed_start: int
    collection_seed_derivation: str
    collection_minimum_success_rate: float
    collection_minimum_split_module_success_rate: float
    collection_maximum_safety_failure_count: int
    collection_maximum_fallback_rate: float
    collection_task_id_uniqueness_required: bool
    collection_seed_uniqueness_required: bool
    collection_pose_hash_uniqueness_required: bool
    collection_cross_split_pose_disjoint_required: bool
    collection_split_module_gate_required: bool
    minimum_free_gib: float
    output_root: str

    def validate(self) -> None:
        if self.protocol_version != ORDER9_R1_CALIBRATION_PROTOCOL_VERSION:
            raise SchemaValidationError(
                "Order9 R1 calibration protocol version mismatch"
            )
        if self.calibration_gate_id != "r1-calibration-gate-v1":
            raise SchemaValidationError("Order9 R1 calibration gate identity mismatch")
        if self.status == "approved":
            if not self.approval_record:
                raise SchemaValidationError(
                    "approved Order9 R1 protocol requires an approval record"
                )
        elif self.approval_record is not None:
            raise SchemaValidationError(
                "unapproved Order9 R1 protocol cannot contain an approval record"
            )
        if self.stage_id != "r1_teacher_trajectory_collection":
            raise SchemaValidationError("Order9 R1 calibration stage identity mismatch")
        if self.distribution_id != ORDER9_R1_OBJECT_DISTRIBUTION:
            raise SchemaValidationError("Order9 R1 calibration distribution mismatch")
        if self.source_c3_release_id != "c3_pi_l_promoted_update18_v1":
            raise SchemaValidationError("Order9 R1 calibration source release mismatch")
        for name in (
            "source_c3_checkpoint_sha256",
            "source_c3_promotion_manifest_sha256",
            "curriculum_schedule_hash",
            "physical_model_hash",
        ):
            _require_sha256(str(getattr(self, name)), name)
        self.source_bucket_manifest.validate()
        if self.source_bucket_manifest.artifact_kind != "c3_accepted_bucket_manifest":
            raise SchemaValidationError(
                "Order9 R1 source bucket artifact role mismatch"
            )
        if self.source_bucket_count < 1:
            raise SchemaValidationError(
                "Order9 R1 source bucket count must be positive"
            )
        if tuple(self.source_module_counts) != ORDER9_R1_REQUIRED_MODULE_COUNTS:
            raise SchemaValidationError("Order9 R1 calibration must cover modules 2--8")
        if (
            self.selection_split != "train"
            or self.confirmation_split != "validation"
            or self.selection_bucket_count < 1
            or self.confirmation_bucket_count < 1
            or self.selection_bucket_count + self.confirmation_bucket_count
            != self.source_bucket_count
            or not self.confirmation_failure_invalidates_protocol
        ):
            raise SchemaValidationError(
                "Order9 R1 calibration selection/confirmation split is invalid"
            )
        if self.coordinate_frame != "world":
            raise SchemaValidationError(
                "Order9 R1 calibration offsets must use world frame"
            )
        if self.envelope_center != "base_task_object_pose_world":
            raise SchemaValidationError(
                "Order9 R1 calibration envelope center mismatch"
            )
        if self.sampling_method != "independent_uniform_v1":
            raise SchemaValidationError(
                "Order9 R1 calibration sampling method mismatch"
            )
        if not all(
            (
                self.support_geometry_preserved,
                self.support_pose_preserved,
                self.robot_reset_preserved,
            )
        ):
            raise SchemaValidationError(
                "Order9 R1 calibration must preserve support and robot reset"
            )
        if self.calibration_seed < 0 or self.collection_seed_start < 0:
            raise SchemaValidationError(
                "Order9 R1 calibration/collection seed is negative"
            )
        if (
            self.collection_seed_start
            <= self.calibration_seed
            < self.collection_seed_start + self.collection_episode_count
        ):
            raise SchemaValidationError(
                "Order9 R1 calibration and collection seeds must be disjoint"
            )
        if (
            self.case_enumeration != "level_split_bucket_sample_replay_lexicographic_v1"
            or self.random_interior_seed_derivation != "r1_derived_seed_v1"
            or self.collection_seed_derivation
            != "sequential_seed_start_plus_episode_index_v1"
        ):
            raise SchemaValidationError(
                "Order9 R1 calibration/collection seed derivation mismatch"
            )
        if len(self.calibration_levels) < 2:
            raise SchemaValidationError(
                "Order9 R1 calibration requires multiple levels"
            )
        level_ids = [level.level_id for level in self.calibration_levels]
        if len(level_ids) != len(set(level_ids)):
            raise SchemaValidationError(
                "Order9 R1 calibration level ids are duplicated"
            )
        previous_position = 0.005
        previous_yaw = math.radians(2.0)
        for level in self.calibration_levels:
            level.validate()
            if (
                level.position_half_span_m <= previous_position
                or level.yaw_half_span_rad <= previous_yaw
            ):
                raise SchemaValidationError(
                    "Order9 R1 calibration levels must strictly expand C3 in order"
                )
            previous_position = level.position_half_span_m
            previous_yaw = level.yaw_half_span_rad
        if (
            self.lattice_point_count_per_bucket
            != ORDER9_R1_CALIBRATION_LATTICE_POINT_COUNT
        ):
            raise SchemaValidationError(
                "Order9 R1 calibration requires the complete 3x3x3 lattice"
            )
        if (
            self.fast_screen_version != ORDER9_R1_FAST_SCREEN_VERSION
            or self.fast_screen_execution_model != ORDER9_R1_FAST_SCREEN_EXECUTION_MODEL
            or tuple(self.fast_screen_phases) != ORDER9_R1_FAST_SCREEN_PHASES
        ):
            raise SchemaValidationError(
                "Order9 R1 fast exact-tracking screen identity mismatch"
            )
        if not all(
            (
                self.fast_screen_reuses_resolved_trajectory,
                self.fast_screen_isaac_forbidden,
                self.fast_screen_controller_layers_forbidden,
                self.fast_screen_ik_resolve_forbidden,
                self.fast_screen_trajectory_optimization_forbidden,
                self.fast_screen_applies_to_calibration_and_collection,
                self.full_control_test_requires_fast_screen_pass,
            )
        ):
            raise SchemaValidationError(
                "Order9 R1 fast-screen/full-control ordering is not fail closed"
            )
        reserve = float(self.minimum_normalized_joint_limit_reserve)
        if not math.isfinite(reserve) or not 0.0 < reserve < 0.5:
            raise SchemaValidationError(
                "Order9 R1 normalized joint-limit reserve must be in (0, 0.5)"
            )
        body_tilt = float(self.maximum_body_tilt_rad)
        if not math.isfinite(body_tilt) or not 0.0 < body_tilt < math.pi / 2.0:
            raise SchemaValidationError(
                "Order9 R1 maximum body tilt must be in (0, pi/2)"
            )
        for name in (
            "random_interior_samples_per_bucket",
            "isaac_replays_per_lattice_case",
            "isaac_replays_per_interior_case",
        ):
            if int(getattr(self, name)) < 1:
                raise SchemaValidationError(f"Order9 R1 {name} must be positive")
        if (
            self.isaac_replays_per_lattice_case != 2
            or self.isaac_replays_per_interior_case != 2
        ):
            raise SchemaValidationError(
                "Order9 R1 calibration v1 requires two Isaac replays per feasible case"
            )
        for name in (
            "minimum_lattice_pass_rate",
            "minimum_teacher_feasibility_rate",
            "minimum_fast_screen_pass_rate",
            "minimum_isaac_success_rate",
            "minimum_interior_teacher_feasibility_rate",
            "minimum_interior_fast_screen_pass_rate",
            "minimum_interior_isaac_success_rate",
            "maximum_fallback_rate",
            "collection_minimum_success_rate",
            "collection_minimum_split_module_success_rate",
            "collection_maximum_fallback_rate",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise SchemaValidationError(f"Order9 R1 {name} must be in [0, 1]")
        if self.minimum_lattice_pass_rate != 1.0:
            raise SchemaValidationError("Order9 R1 calibration lattice must pass 100%")
        if any(
            float(value) < 0.95
            for value in (
                self.minimum_teacher_feasibility_rate,
                self.minimum_fast_screen_pass_rate,
                self.minimum_isaac_success_rate,
                self.minimum_interior_teacher_feasibility_rate,
                self.minimum_interior_fast_screen_pass_rate,
                self.minimum_interior_isaac_success_rate,
                self.collection_minimum_success_rate,
                self.collection_minimum_split_module_success_rate,
            )
        ):
            raise SchemaValidationError("Order9 R1 success gates must be at least 95%")
        if self.maximum_safety_failure_count != 0:
            raise SchemaValidationError(
                "Order9 R1 calibration safety gate must be zero"
            )
        if self.collection_maximum_safety_failure_count != 0:
            raise SchemaValidationError("Order9 R1 collection safety gate must be zero")
        if (
            self.maximum_fallback_rate != 0.0
            or self.collection_maximum_fallback_rate != 0.0
        ):
            raise SchemaValidationError(
                "Order9 R1 calibration/collection fallback gate is zero"
            )
        if (
            not math.isfinite(float(self.minimum_support_projected_com_margin_m))
            or self.minimum_support_projected_com_margin_m < 0.0
        ):
            raise SchemaValidationError(
                "Order9 R1 support projected-CoM margin must be non-negative"
            )
        if not all(
            (
                self.stop_on_first_failed_level,
                self.accept_largest_consecutive_passing_level,
                self.extrapolation_beyond_ladder_forbidden,
            )
        ):
            raise SchemaValidationError(
                "Order9 R1 search/acceptance rule is not fail closed"
            )
        if self.calibration_records_training_eligible:
            raise SchemaValidationError(
                "Order9 R1 calibration records are not training data"
            )
        split_total = (
            self.collection_train_episode_count
            + self.collection_validation_episode_count
            + self.collection_held_out_episode_count
        )
        if (
            self.collection_episode_count < 100
            or split_total != self.collection_episode_count
        ):
            raise SchemaValidationError("Order9 R1 collection count/splits are invalid")
        if any(
            value < len(ORDER9_R1_REQUIRED_MODULE_COUNTS)
            for value in (
                self.collection_train_episode_count,
                self.collection_validation_episode_count,
                self.collection_held_out_episode_count,
            )
        ):
            raise SchemaValidationError(
                "Order9 R1 every collection split must cover all module counts"
            )
        if (
            self.collection_per_module_episode_count != 20
            or self.collection_train_per_module_episode_count != 14
            or self.collection_validation_per_module_episode_count != 3
            or self.collection_held_out_per_module_episode_count != 3
            or self.collection_episode_count
            != len(ORDER9_R1_REQUIRED_MODULE_COUNTS)
            * self.collection_per_module_episode_count
            or self.collection_train_episode_count
            != len(ORDER9_R1_REQUIRED_MODULE_COUNTS)
            * self.collection_train_per_module_episode_count
            or self.collection_validation_episode_count
            != len(ORDER9_R1_REQUIRED_MODULE_COUNTS)
            * self.collection_validation_per_module_episode_count
            or self.collection_held_out_episode_count
            != len(ORDER9_R1_REQUIRED_MODULE_COUNTS)
            * self.collection_held_out_per_module_episode_count
        ):
            raise SchemaValidationError(
                "Order9 R1 collection split-by-module allocation mismatch"
            )
        if not all(
            (
                self.collection_task_id_uniqueness_required,
                self.collection_seed_uniqueness_required,
                self.collection_pose_hash_uniqueness_required,
                self.collection_cross_split_pose_disjoint_required,
                self.collection_split_module_gate_required,
            )
        ):
            raise SchemaValidationError(
                "Order9 R1 collection uniqueness/split-module gates are required"
            )
        if (
            not math.isfinite(float(self.minimum_free_gib))
            or self.minimum_free_gib < 128.0
        ):
            raise SchemaValidationError(
                "Order9 R1 storage floor must be at least 128 GiB"
            )
        if not self.output_root or Path(self.output_root).is_absolute():
            raise SchemaValidationError(
                "Order9 R1 calibration output root must be repository-relative"
            )

    @property
    def teacher_lattice_case_count_per_level(self) -> int:
        return self.selection_bucket_count * self.lattice_point_count_per_bucket

    @property
    def teacher_interior_case_count_per_level(self) -> int:
        return self.selection_bucket_count * self.random_interior_samples_per_bucket

    @property
    def isaac_episode_count_per_level(self) -> int:
        return (
            self.teacher_lattice_case_count_per_level
            * self.isaac_replays_per_lattice_case
            + self.teacher_interior_case_count_per_level
            * self.isaac_replays_per_interior_case
        )

    @property
    def confirmation_teacher_case_count(self) -> int:
        return self.confirmation_bucket_count * (
            self.lattice_point_count_per_bucket
            + self.random_interior_samples_per_bucket
        )

    @property
    def confirmation_isaac_episode_count(self) -> int:
        return self.confirmation_bucket_count * (
            self.lattice_point_count_per_bucket * self.isaac_replays_per_lattice_case
            + self.random_interior_samples_per_bucket
            * self.isaac_replays_per_interior_case
        )


@dataclass
class Order9R1CalibrationApprovalRecord(SchemaBase):
    """Human decision binding the approved v1 bytes to the reviewed proposal."""

    record_version: str
    decision: Literal["approved"]
    approved_by: str
    approved_on: str
    calibration_gate_id: str
    stage_id: str
    approved_protocol: Order9ArtifactBinding
    source_proposal: Order9ArtifactBinding
    source_c3_release_id: str
    source_c3_checkpoint_sha256: str
    approval_scope: tuple[str, ...]
    statement: str

    def validate(self) -> None:
        if self.record_version != ORDER9_R1_CALIBRATION_APPROVAL_RECORD_VERSION:
            raise SchemaValidationError("Order9 R1 approval record version mismatch")
        if self.decision != "approved" or self.approved_by != "repository_user":
            raise SchemaValidationError("Order9 R1 approval decision/actor mismatch")
        try:
            date.fromisoformat(self.approved_on)
        except ValueError as exc:
            raise SchemaValidationError("Order9 R1 approval date is invalid") from exc
        if (
            self.calibration_gate_id != "r1-calibration-gate-v1"
            or self.stage_id != "r1_teacher_trajectory_collection"
            or self.source_c3_release_id != "c3_pi_l_promoted_update18_v1"
        ):
            raise SchemaValidationError("Order9 R1 approval identity mismatch")
        _require_sha256(
            self.source_c3_checkpoint_sha256,
            "Order9 R1 approval source checkpoint",
        )
        self.approved_protocol.validate()
        self.source_proposal.validate()
        if self.approved_protocol.artifact_kind != "approved_calibration_protocol":
            raise SchemaValidationError("Order9 R1 approved protocol role mismatch")
        if self.source_proposal.artifact_kind != "calibration_protocol_proposal":
            raise SchemaValidationError("Order9 R1 source proposal role mismatch")
        if set(self.approval_scope) != _REQUIRED_APPROVAL_SCOPE or len(
            self.approval_scope
        ) != len(_REQUIRED_APPROVAL_SCOPE):
            raise SchemaValidationError("Order9 R1 approval scope is incomplete")
        if not self.statement.strip():
            raise SchemaValidationError("Order9 R1 approval statement is empty")


def load_order9_r1_calibration_protocol(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> Order9R1CalibrationProtocol:
    protocol = Order9R1CalibrationProtocol.from_dict(load_config(path))
    protocol.validate()
    repository = Path(repository_root).resolve()
    bucket_path = _repository_artifact_path(
        repository,
        protocol.source_bucket_manifest.path,
    )
    if hash_file(bucket_path) != protocol.source_bucket_manifest.sha256:
        raise SchemaValidationError("Order9 R1 source bucket manifest SHA-256 mismatch")
    payload = json.loads(bucket_path.read_text(encoding="utf-8"))
    buckets = payload.get("buckets")
    if not isinstance(buckets, list) or len(buckets) != protocol.source_bucket_count:
        raise SchemaValidationError("Order9 R1 source bucket count mismatch")
    module_counts = tuple(sorted({int(bucket["module_count"]) for bucket in buckets}))
    if module_counts != tuple(protocol.source_module_counts):
        raise SchemaValidationError("Order9 R1 source bucket module coverage mismatch")
    split_counts = {
        split: sum(1 for bucket in buckets if bucket.get("split") == split)
        for split in (protocol.selection_split, protocol.confirmation_split)
    }
    if (
        split_counts[protocol.selection_split] != protocol.selection_bucket_count
        or split_counts[protocol.confirmation_split]
        != protocol.confirmation_bucket_count
    ):
        raise SchemaValidationError("Order9 R1 source bucket split count mismatch")
    for split in (protocol.selection_split, protocol.confirmation_split):
        covered = tuple(
            sorted(
                {
                    int(bucket["module_count"])
                    for bucket in buckets
                    if bucket.get("split") == split
                }
            )
        )
        if covered != tuple(protocol.source_module_counts):
            raise SchemaValidationError(
                f"Order9 R1 source bucket {split} module coverage mismatch"
            )
    if (
        payload.get("stage_id") != "c3_pi_l_ppo_arbitrary_morphology"
        or payload.get("curriculum_schedule_hash") != protocol.curriculum_schedule_hash
        or payload.get("topology_randomized") is not True
    ):
        raise SchemaValidationError("Order9 R1 source bucket contract mismatch")
    return protocol


def load_order9_r1_calibration_approval_record(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> Order9R1CalibrationApprovalRecord:
    """Load and verify the approval record plus both protocol byte bindings."""

    source = Path(path).resolve()
    repository = Path(repository_root).resolve()
    record = Order9R1CalibrationApprovalRecord.from_dict(load_config(source))
    record.validate()
    approved_path = _repository_artifact_path(
        repository,
        record.approved_protocol.path,
    )
    proposal_path = _repository_artifact_path(repository, record.source_proposal.path)
    for artifact, artifact_path in (
        (record.approved_protocol, approved_path),
        (record.source_proposal, proposal_path),
    ):
        if hash_file(artifact_path) != artifact.sha256:
            raise SchemaValidationError(
                f"Order9 R1 approval byte binding mismatch: {artifact.artifact_kind}"
            )
    protocol = load_order9_r1_calibration_protocol(
        approved_path,
        repository_root=repository,
    )
    try:
        record_relative = source.relative_to(repository).as_posix()
    except ValueError as exc:
        raise SchemaValidationError(
            "Order9 R1 approval record must be inside the repository"
        ) from exc
    if (
        protocol.status != "approved"
        or protocol.approval_record != record_relative
        or protocol.calibration_gate_id != record.calibration_gate_id
        or protocol.stage_id != record.stage_id
        or protocol.source_c3_release_id != record.source_c3_release_id
        or protocol.source_c3_checkpoint_sha256 != record.source_c3_checkpoint_sha256
    ):
        raise SchemaValidationError(
            "Order9 R1 approved protocol does not match its approval record"
        )
    return record


def _repository_artifact_path(repository: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise SchemaValidationError("Order9 R1 protocol artifact path must be relative")
    resolved = (repository / relative).resolve()
    if resolved != repository and repository not in resolved.parents:
        raise SchemaValidationError(
            "Order9 R1 protocol artifact path escapes repository"
        )
    return resolved


def _require_sha256(value: str, path: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise SchemaValidationError(f"{path} must be a lowercase SHA-256 digest")
