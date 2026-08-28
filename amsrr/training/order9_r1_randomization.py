from __future__ import annotations

"""Hash-bound reachable-pose randomization for the Order-9 R1 ring.

R1 changes only the planar placement and yaw of the canonical object task.
The accepted envelope is intentionally supplied by a calibration manifest;
this module does not turn the historical expanded-object ``+/-0.015 m``
placeholder into an R1 contract.
"""

import math
import random
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_len
from amsrr.schemas.order9 import Order9ArtifactBinding
from amsrr.schemas.task_spec import GeometryType, TaskSpec
from amsrr.training.order9_c3_teacher import (
    build_order9_c3_posture_collision_object,
)
from amsrr.training.order9_r1_fast_screen import (
    ORDER9_R1_FAST_SCREEN_EXECUTION_MODEL,
    ORDER9_R1_FAST_SCREEN_VERSION,
)
from amsrr.utils.hashing import hash_file, stable_hash

ORDER9_R1_DISTRIBUTION_MANIFEST_VERSION = (
    "order9_r1_reachable_pose_distribution_manifest_v1"
)
ORDER9_R1_RANDOMIZATION_VERSION = "order9_r1_reachable_pose_randomization_v1"
ORDER9_R1_OBJECT_DISTRIBUTION = "reachable_pose_expansion"
ORDER9_R1_REQUIRED_MODULE_COUNTS = tuple(range(2, 9))

_CONSERVATIVE_POSITION_LIMIT_M = 0.005
_CONSERVATIVE_YAW_LIMIT_RAD = 2.0 * math.pi / 180.0


@dataclass
class Order9R1ReachablePoseEnvelope(SchemaBase):
    """Accepted offsets from a canonical C3 object task."""

    initial_x_offset_m: tuple[float, float]
    initial_y_offset_m: tuple[float, float]
    initial_yaw_offset_rad: tuple[float, float]

    def validate(self) -> None:
        for name in (
            "initial_x_offset_m",
            "initial_y_offset_m",
            "initial_yaw_offset_rad",
        ):
            bounds = getattr(self, name)
            require_len(bounds, 2, f"Order9R1ReachablePoseEnvelope.{name}")
            lower, upper = (float(value) for value in bounds)
            if (
                not math.isfinite(lower)
                or not math.isfinite(upper)
                or lower > upper
                or lower > 0.0
                or upper < 0.0
            ):
                raise SchemaValidationError(
                    f"Order9R1ReachablePoseEnvelope.{name} must be a finite "
                    "ordered range containing zero"
                )
        planar_limit = max(
            *(abs(value) for value in self.initial_x_offset_m),
            *(abs(value) for value in self.initial_y_offset_m),
        )
        yaw_limit = max(abs(value) for value in self.initial_yaw_offset_rad)
        if planar_limit <= _CONSERVATIVE_POSITION_LIMIT_M:
            raise SchemaValidationError(
                "R1 pose envelope must expand beyond the conservative 5 mm position range"
            )
        if yaw_limit <= _CONSERVATIVE_YAW_LIMIT_RAD:
            raise SchemaValidationError(
                "R1 yaw envelope must expand beyond the conservative 2 degree range"
            )


@dataclass
class Order9R1SupportAudit(SchemaBase):
    """Frozen source-support margins for one R1 object/goal placement."""

    support_geometry_hash: str
    support_center_xy_m: tuple[float, float]
    support_size_xy_m: tuple[float, float]
    start_projected_com_margin_m: float
    goal_projected_com_margin_m: float
    start_full_footprint_margin_m: float
    goal_full_footprint_margin_m: float

    def validate(self) -> None:
        _require_sha256(self.support_geometry_hash, "R1 support geometry hash")
        require_len(self.support_center_xy_m, 2, "R1 support center")
        require_len(self.support_size_xy_m, 2, "R1 support size")
        if any(
            not math.isfinite(float(value))
            for value in (
                *self.support_center_xy_m,
                *self.support_size_xy_m,
                self.start_projected_com_margin_m,
                self.goal_projected_com_margin_m,
                self.start_full_footprint_margin_m,
                self.goal_full_footprint_margin_m,
            )
        ):
            raise SchemaValidationError(
                "Order9 R1 support audit contains non-finite data"
            )
        if any(float(value) <= 0.0 for value in self.support_size_xy_m):
            raise SchemaValidationError("Order9 R1 support audit size must be positive")

    @property
    def minimum_projected_com_margin_m(self) -> float:
        return min(
            self.start_projected_com_margin_m,
            self.goal_projected_com_margin_m,
        )

    @property
    def minimum_full_footprint_margin_m(self) -> float:
        return min(
            self.start_full_footprint_margin_m,
            self.goal_full_footprint_margin_m,
        )


@dataclass
class Order9R1DistributionManifest(SchemaBase):
    """Calibration evidence that authorizes one R1 collection envelope."""

    manifest_version: str
    distribution_id: str
    accepted: bool
    envelope: Order9R1ReachablePoseEnvelope
    coordinate_frame: str
    envelope_center: str
    sampling_method: str
    calibration_seed: int
    calibration_gate_id: str
    accepted_level_id: str
    calibration_gate_approval: Order9ArtifactBinding
    approved_calibration_protocol: Order9ArtifactBinding
    fast_screen_version: str
    fast_screen_execution_model: str
    full_control_test_requires_fast_screen_pass: bool
    calibration_minimum_lattice_pass_rate: float
    calibration_minimum_teacher_feasibility_rate: float
    calibration_minimum_fast_screen_pass_rate: float
    calibration_minimum_isaac_success_rate: float
    calibration_minimum_interior_teacher_feasibility_rate: float
    calibration_minimum_interior_fast_screen_pass_rate: float
    calibration_minimum_interior_isaac_success_rate: float
    calibration_minimum_normalized_joint_limit_reserve: float
    calibration_maximum_body_tilt_rad: float
    calibration_minimum_support_projected_com_margin_m: float
    source_c3_release_id: str
    source_c3_checkpoint_sha256: str
    source_c3_promotion_manifest_sha256: str
    curriculum_schedule_hash: str
    covered_module_counts: tuple[int, ...]
    covered_topology_ids: tuple[str, ...]
    selection_split: str
    selection_bucket_count: int
    confirmation_split: str
    confirmation_bucket_count: int
    lattice_case_count: int
    lattice_pass_count: int
    teacher_feasibility_attempt_count: int
    teacher_feasibility_success_count: int
    fast_screen_attempt_count: int
    fast_screen_success_count: int
    lattice_fast_screen_pass_count: int
    minimum_fast_screen_normalized_joint_limit_reserve: float
    maximum_fast_screen_body_tilt_rad: float
    isaac_attempt_count: int
    isaac_success_count: int
    safety_failure_count: int
    fallback_count: int
    confirmation_lattice_case_count: int
    confirmation_lattice_pass_count: int
    confirmation_teacher_feasibility_attempt_count: int
    confirmation_teacher_feasibility_success_count: int
    confirmation_fast_screen_attempt_count: int
    confirmation_fast_screen_success_count: int
    confirmation_lattice_fast_screen_pass_count: int
    confirmation_minimum_fast_screen_normalized_joint_limit_reserve: float
    confirmation_maximum_fast_screen_body_tilt_rad: float
    confirmation_isaac_attempt_count: int
    confirmation_isaac_success_count: int
    confirmation_safety_failure_count: int
    confirmation_fallback_count: int
    support_audit_attempt_count: int
    minimum_support_projected_com_margin_m: float
    minimum_support_full_footprint_margin_m: float
    confirmation_support_audit_attempt_count: int
    confirmation_minimum_support_projected_com_margin_m: float
    confirmation_minimum_support_full_footprint_margin_m: float
    measured_wall_time_s: float
    canonical_box_geometry_preserved: bool
    object_properties_preserved: bool
    goal_relative_transform_preserved: bool
    support_geometry_preserved: bool
    support_pose_preserved: bool
    robot_reset_preserved: bool
    evidence_artifacts: list[Order9ArtifactBinding]
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.manifest_version != ORDER9_R1_DISTRIBUTION_MANIFEST_VERSION:
            raise SchemaValidationError(
                "Order9 R1 distribution manifest version mismatch"
            )
        if self.distribution_id != ORDER9_R1_OBJECT_DISTRIBUTION:
            raise SchemaValidationError("Order9 R1 distribution identity mismatch")
        if not self.accepted:
            raise SchemaValidationError("Order9 R1 distribution was not accepted")
        self.envelope.validate()
        if self.coordinate_frame != "world":
            raise SchemaValidationError(
                "Order9 R1 pose offsets must use the world frame"
            )
        if self.envelope_center != "base_task_object_pose_world":
            raise SchemaValidationError("Order9 R1 envelope center contract mismatch")
        if self.sampling_method != "independent_uniform_v1":
            raise SchemaValidationError("Order9 R1 sampling method mismatch")
        if (
            self.calibration_seed < 0
            or not self.calibration_gate_id
            or not self.accepted_level_id
        ):
            raise SchemaValidationError("Order9 R1 calibration identity is invalid")
        self.calibration_gate_approval.validate()
        if self.calibration_gate_approval.artifact_kind != "calibration_gate_approval":
            raise SchemaValidationError("Order9 R1 calibration approval role mismatch")
        self.approved_calibration_protocol.validate()
        if (
            self.approved_calibration_protocol.artifact_kind
            != "approved_calibration_protocol"
        ):
            raise SchemaValidationError("Order9 R1 calibration protocol role mismatch")
        if (
            self.fast_screen_version != ORDER9_R1_FAST_SCREEN_VERSION
            or self.fast_screen_execution_model != ORDER9_R1_FAST_SCREEN_EXECUTION_MODEL
            or not self.full_control_test_requires_fast_screen_pass
        ):
            raise SchemaValidationError(
                "Order9 R1 distribution fast-screen identity/order mismatch"
            )
        for name in (
            "calibration_minimum_lattice_pass_rate",
            "calibration_minimum_teacher_feasibility_rate",
            "calibration_minimum_fast_screen_pass_rate",
            "calibration_minimum_isaac_success_rate",
            "calibration_minimum_interior_teacher_feasibility_rate",
            "calibration_minimum_interior_fast_screen_pass_rate",
            "calibration_minimum_interior_isaac_success_rate",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise SchemaValidationError(f"Order9 R1 {name} must be in [0, 1]")
        for name in (
            "source_c3_checkpoint_sha256",
            "source_c3_promotion_manifest_sha256",
            "curriculum_schedule_hash",
        ):
            _require_sha256(str(getattr(self, name)), name)
        if not self.source_c3_release_id:
            raise SchemaValidationError("Order9 R1 source C3 release id is empty")
        if tuple(self.covered_module_counts) != ORDER9_R1_REQUIRED_MODULE_COUNTS:
            raise SchemaValidationError(
                "Order9 R1 calibration must cover every module count from 2 through 8"
            )
        if not self.covered_topology_ids or len(self.covered_topology_ids) != len(
            set(self.covered_topology_ids)
        ):
            raise SchemaValidationError(
                "Order9 R1 calibration topology ids must be non-empty and unique"
            )
        if (
            self.selection_split != "train"
            or self.confirmation_split != "validation"
            or self.selection_bucket_count != 22
            or self.confirmation_bucket_count != 14
        ):
            raise SchemaValidationError(
                "Order9 R1 calibration selection/confirmation evidence mismatch"
            )
        reserve_values = (
            self.calibration_minimum_normalized_joint_limit_reserve,
            self.minimum_fast_screen_normalized_joint_limit_reserve,
            self.confirmation_minimum_fast_screen_normalized_joint_limit_reserve,
        )
        if (
            any(not math.isfinite(float(value)) for value in reserve_values)
            or not 0.0 < self.calibration_minimum_normalized_joint_limit_reserve < 0.5
            or self.minimum_fast_screen_normalized_joint_limit_reserve
            < self.calibration_minimum_normalized_joint_limit_reserve
            or self.confirmation_minimum_fast_screen_normalized_joint_limit_reserve
            < self.calibration_minimum_normalized_joint_limit_reserve
        ):
            raise SchemaValidationError(
                "Order9 R1 fast-screen joint-limit reserve gate failed"
            )
        tilt_values = (
            self.calibration_maximum_body_tilt_rad,
            self.maximum_fast_screen_body_tilt_rad,
            self.confirmation_maximum_fast_screen_body_tilt_rad,
        )
        if (
            any(not math.isfinite(float(value)) for value in tilt_values)
            or not 0.0 < self.calibration_maximum_body_tilt_rad < math.pi / 2.0
            or self.maximum_fast_screen_body_tilt_rad
            > self.calibration_maximum_body_tilt_rad
            or self.confirmation_maximum_fast_screen_body_tilt_rad
            > self.calibration_maximum_body_tilt_rad
        ):
            raise SchemaValidationError("Order9 R1 fast-screen body-tilt gate failed")
        for name in (
            "lattice_case_count",
            "lattice_pass_count",
            "teacher_feasibility_attempt_count",
            "teacher_feasibility_success_count",
            "fast_screen_attempt_count",
            "fast_screen_success_count",
            "lattice_fast_screen_pass_count",
            "isaac_attempt_count",
            "isaac_success_count",
            "safety_failure_count",
            "fallback_count",
            "confirmation_lattice_case_count",
            "confirmation_lattice_pass_count",
            "confirmation_teacher_feasibility_attempt_count",
            "confirmation_teacher_feasibility_success_count",
            "confirmation_fast_screen_attempt_count",
            "confirmation_fast_screen_success_count",
            "confirmation_lattice_fast_screen_pass_count",
            "confirmation_isaac_attempt_count",
            "confirmation_isaac_success_count",
            "confirmation_safety_failure_count",
            "confirmation_fallback_count",
            "support_audit_attempt_count",
            "confirmation_support_audit_attempt_count",
        ):
            if int(getattr(self, name)) < 0:
                raise SchemaValidationError(f"Order9 R1 {name} must be non-negative")
        if (
            self.lattice_case_count < 1
            or self.teacher_feasibility_attempt_count < 1
            or self.isaac_attempt_count < 1
            or self.confirmation_lattice_case_count < 1
            or self.confirmation_teacher_feasibility_attempt_count < 1
            or self.confirmation_isaac_attempt_count < 1
        ):
            raise SchemaValidationError(
                "Order9 R1 calibration requires lattice, interior, and Isaac trials"
            )
        if (
            self.teacher_feasibility_attempt_count <= self.lattice_case_count
            or self.confirmation_teacher_feasibility_attempt_count
            <= self.confirmation_lattice_case_count
            or self.fast_screen_attempt_count != self.teacher_feasibility_success_count
            or self.confirmation_fast_screen_attempt_count
            != self.confirmation_teacher_feasibility_success_count
            or self.isaac_attempt_count != 2 * self.fast_screen_success_count
            or self.confirmation_isaac_attempt_count
            != 2 * self.confirmation_fast_screen_success_count
        ):
            raise SchemaValidationError(
                "Order9 R1 requires fast screening before two Isaac replays per admitted case"
            )
        if (
            self.lattice_pass_count > self.lattice_case_count
            or self.teacher_feasibility_success_count
            > self.teacher_feasibility_attempt_count
            or self.fast_screen_success_count > self.fast_screen_attempt_count
            or self.lattice_fast_screen_pass_count > self.lattice_case_count
            or self.isaac_success_count > self.isaac_attempt_count
            or self.confirmation_lattice_pass_count
            > self.confirmation_lattice_case_count
            or self.confirmation_teacher_feasibility_success_count
            > self.confirmation_teacher_feasibility_attempt_count
            or self.confirmation_fast_screen_success_count
            > self.confirmation_fast_screen_attempt_count
            or self.confirmation_lattice_fast_screen_pass_count
            > self.confirmation_lattice_case_count
            or self.confirmation_isaac_success_count
            > self.confirmation_isaac_attempt_count
        ):
            raise SchemaValidationError("Order9 R1 success count exceeds attempt count")
        if self.lattice_pass_rate < self.calibration_minimum_lattice_pass_rate:
            raise SchemaValidationError("Order9 R1 lattice calibration gate failed")
        if (
            self.teacher_feasibility_rate
            < self.calibration_minimum_teacher_feasibility_rate
        ):
            raise SchemaValidationError(
                "Order9 R1 teacher-feasibility calibration gate failed"
            )
        if (
            self.fast_screen_pass_rate < self.calibration_minimum_fast_screen_pass_rate
            or self.lattice_fast_screen_pass_count != self.lattice_case_count
        ):
            raise SchemaValidationError(
                "Order9 R1 fast exact-tracking screen gate failed"
            )
        if self.isaac_success_rate < self.calibration_minimum_isaac_success_rate:
            raise SchemaValidationError(
                "Order9 R1 Isaac calibration success gate failed"
            )
        if (
            self.interior_teacher_feasibility_rate
            < self.calibration_minimum_interior_teacher_feasibility_rate
            or self.interior_fast_screen_pass_rate
            < self.calibration_minimum_interior_fast_screen_pass_rate
            or self.interior_isaac_success_rate
            < self.calibration_minimum_interior_isaac_success_rate
        ):
            raise SchemaValidationError("Order9 R1 interior calibration gate failed")
        if (
            self.confirmation_lattice_pass_rate
            < self.calibration_minimum_lattice_pass_rate
            or self.confirmation_teacher_feasibility_rate
            < self.calibration_minimum_teacher_feasibility_rate
            or self.confirmation_fast_screen_pass_rate
            < self.calibration_minimum_fast_screen_pass_rate
            or self.confirmation_lattice_fast_screen_pass_count
            != self.confirmation_lattice_case_count
            or self.confirmation_isaac_success_rate
            < self.calibration_minimum_isaac_success_rate
            or self.confirmation_interior_teacher_feasibility_rate
            < self.calibration_minimum_interior_teacher_feasibility_rate
            or self.confirmation_interior_fast_screen_pass_rate
            < self.calibration_minimum_interior_fast_screen_pass_rate
            or self.confirmation_interior_isaac_success_rate
            < self.calibration_minimum_interior_isaac_success_rate
        ):
            raise SchemaValidationError("Order9 R1 confirmation gate failed")
        if any(
            (
                self.safety_failure_count,
                self.fallback_count,
                self.confirmation_safety_failure_count,
                self.confirmation_fallback_count,
            )
        ):
            raise SchemaValidationError(
                "Order9 R1 calibration/confirmation has safety or fallback failure"
            )
        support_values = (
            self.calibration_minimum_support_projected_com_margin_m,
            self.minimum_support_projected_com_margin_m,
            self.minimum_support_full_footprint_margin_m,
            self.confirmation_minimum_support_projected_com_margin_m,
            self.confirmation_minimum_support_full_footprint_margin_m,
        )
        if any(not math.isfinite(float(value)) for value in support_values):
            raise SchemaValidationError("Order9 R1 support evidence is non-finite")
        if (
            self.calibration_minimum_support_projected_com_margin_m < 0.0
            or self.support_audit_attempt_count
            != self.teacher_feasibility_attempt_count
            or self.confirmation_support_audit_attempt_count
            != self.confirmation_teacher_feasibility_attempt_count
            or self.minimum_support_projected_com_margin_m
            < self.calibration_minimum_support_projected_com_margin_m
            or self.confirmation_minimum_support_projected_com_margin_m
            < self.calibration_minimum_support_projected_com_margin_m
        ):
            raise SchemaValidationError("Order9 R1 projected-CoM support gate failed")
        if (
            not math.isfinite(float(self.measured_wall_time_s))
            or self.measured_wall_time_s <= 0
        ):
            raise SchemaValidationError(
                "Order9 R1 calibration wall time must be finite and positive"
            )
        if not all(
            (
                self.canonical_box_geometry_preserved,
                self.object_properties_preserved,
                self.goal_relative_transform_preserved,
                self.support_geometry_preserved,
                self.support_pose_preserved,
                self.robot_reset_preserved,
            )
        ):
            raise SchemaValidationError(
                "Order9 R1 may vary only object/goal position and yaw"
            )
        required_evidence = {
            "calibration_config",
            "teacher_complete_trajectory",
            "fast_kinematic_screen",
            "production_checker",
            "isaac_replay",
            "support_audit",
            "topology_coverage",
        }
        evidence_by_kind = {
            artifact.artifact_kind: artifact for artifact in self.evidence_artifacts
        }
        if (
            len(evidence_by_kind) != len(self.evidence_artifacts)
            or set(evidence_by_kind) != required_evidence
        ):
            raise SchemaValidationError(
                "Order9 R1 calibration evidence artifact roles mismatch"
            )
        for artifact in self.evidence_artifacts:
            artifact.validate()

    @property
    def lattice_pass_rate(self) -> float:
        return float(self.lattice_pass_count) / float(self.lattice_case_count)

    @property
    def teacher_feasibility_rate(self) -> float:
        return float(self.teacher_feasibility_success_count) / float(
            self.teacher_feasibility_attempt_count
        )

    @property
    def isaac_success_rate(self) -> float:
        return float(self.isaac_success_count) / float(self.isaac_attempt_count)

    @property
    def fast_screen_pass_rate(self) -> float:
        return float(self.fast_screen_success_count) / float(
            self.fast_screen_attempt_count
        )

    @property
    def interior_teacher_feasibility_rate(self) -> float:
        interior_attempts = (
            self.teacher_feasibility_attempt_count - self.lattice_case_count
        )
        interior_successes = (
            self.teacher_feasibility_success_count - self.lattice_pass_count
        )
        return float(interior_successes) / float(interior_attempts)

    @property
    def interior_isaac_success_rate(self) -> float:
        interior_attempts = self.isaac_attempt_count - 2 * self.lattice_case_count
        interior_successes = self.isaac_success_count - 2 * self.lattice_pass_count
        return float(interior_successes) / float(interior_attempts)

    @property
    def interior_fast_screen_pass_rate(self) -> float:
        attempts = self.fast_screen_attempt_count - self.lattice_case_count
        successes = self.fast_screen_success_count - self.lattice_fast_screen_pass_count
        return float(successes) / float(attempts)

    @property
    def confirmation_lattice_pass_rate(self) -> float:
        return float(self.confirmation_lattice_pass_count) / float(
            self.confirmation_lattice_case_count
        )

    @property
    def confirmation_teacher_feasibility_rate(self) -> float:
        return float(self.confirmation_teacher_feasibility_success_count) / float(
            self.confirmation_teacher_feasibility_attempt_count
        )

    @property
    def confirmation_isaac_success_rate(self) -> float:
        return float(self.confirmation_isaac_success_count) / float(
            self.confirmation_isaac_attempt_count
        )

    @property
    def confirmation_fast_screen_pass_rate(self) -> float:
        return float(self.confirmation_fast_screen_success_count) / float(
            self.confirmation_fast_screen_attempt_count
        )

    @property
    def confirmation_interior_teacher_feasibility_rate(self) -> float:
        attempts = (
            self.confirmation_teacher_feasibility_attempt_count
            - self.confirmation_lattice_case_count
        )
        successes = (
            self.confirmation_teacher_feasibility_success_count
            - self.confirmation_lattice_pass_count
        )
        return float(successes) / float(attempts)

    @property
    def confirmation_interior_isaac_success_rate(self) -> float:
        attempts = (
            self.confirmation_isaac_attempt_count
            - 2 * self.confirmation_lattice_case_count
        )
        successes = (
            self.confirmation_isaac_success_count
            - 2 * self.confirmation_lattice_pass_count
        )
        return float(successes) / float(attempts)

    @property
    def confirmation_interior_fast_screen_pass_rate(self) -> float:
        attempts = (
            self.confirmation_fast_screen_attempt_count
            - self.confirmation_lattice_case_count
        )
        successes = (
            self.confirmation_fast_screen_success_count
            - self.confirmation_lattice_fast_screen_pass_count
        )
        return float(successes) / float(attempts)


@dataclass(frozen=True)
class LoadedOrder9R1DistributionManifest:
    manifest: Order9R1DistributionManifest
    path: str
    sha256: str


@dataclass
class Order9R1ReachablePoseSample(SchemaBase):
    seed: int
    sample_index: int
    task_spec: TaskSpec
    initial_x_offset_m: float
    initial_y_offset_m: float
    initial_yaw_offset_rad: float
    distribution_manifest_sha256: str
    support_audit: Order9R1SupportAudit
    randomization_version: str = ORDER9_R1_RANDOMIZATION_VERSION

    def validate(self) -> None:
        if self.seed < 0 or self.sample_index < 0:
            raise SchemaValidationError("Order9 R1 sample indices must be non-negative")
        for name in (
            "initial_x_offset_m",
            "initial_y_offset_m",
            "initial_yaw_offset_rad",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise SchemaValidationError(f"Order9 R1 sample {name} must be finite")
        _require_sha256(
            self.distribution_manifest_sha256,
            "Order9R1ReachablePoseSample.distribution_manifest_sha256",
        )
        self.support_audit.validate()


@dataclass(frozen=True)
class Order9R1AppliedPose:
    """Pure SE(2) candidate used by calibration before a distribution exists."""

    task_spec: TaskSpec
    support_audit: Order9R1SupportAudit


class Order9R1ReachablePoseRandomizer:
    """Apply only a calibrated, hash-bound SE(2) placement perturbation."""

    def __init__(self, loaded: LoadedOrder9R1DistributionManifest) -> None:
        loaded.manifest.validate()
        _require_sha256(loaded.sha256, "loaded R1 distribution manifest")
        self.loaded = loaded

    def sample(
        self,
        base_task_spec: TaskSpec,
        *,
        seed: int,
        sample_index: int = 0,
    ) -> Order9R1ReachablePoseSample:
        if seed < 0 or sample_index < 0:
            raise ValueError("Order9 R1 seed and sample_index must be non-negative")
        rng = random.Random(derive_order9_r1_seed(seed, sample_index))
        envelope = self.loaded.manifest.envelope
        return self._apply_offsets(
            base_task_spec,
            seed=seed,
            sample_index=sample_index,
            x_offset_m=rng.uniform(*envelope.initial_x_offset_m),
            y_offset_m=rng.uniform(*envelope.initial_y_offset_m),
            yaw_offset_rad=rng.uniform(*envelope.initial_yaw_offset_rad),
        )

    def calibration_lattice_samples(
        self,
        base_task_spec: TaskSpec,
        *,
        calibration_seed: int,
        case_index_start: int = 0,
    ) -> list[Order9R1ReachablePoseSample]:
        """Return the complete lower/center/upper 3x3x3 calibration lattice."""

        if calibration_seed < 0 or case_index_start < 0:
            raise ValueError("Order9 R1 calibration seed/index must be non-negative")
        envelope = self.loaded.manifest.envelope
        offsets = product(
            (envelope.initial_x_offset_m[0], 0.0, envelope.initial_x_offset_m[1]),
            (envelope.initial_y_offset_m[0], 0.0, envelope.initial_y_offset_m[1]),
            (
                envelope.initial_yaw_offset_rad[0],
                0.0,
                envelope.initial_yaw_offset_rad[1],
            ),
        )
        return [
            self._apply_offsets(
                base_task_spec,
                seed=calibration_seed,
                sample_index=case_index_start + index,
                x_offset_m=x,
                y_offset_m=y,
                yaw_offset_rad=yaw,
            )
            for index, (x, y, yaw) in enumerate(offsets)
        ]

    def _apply_offsets(
        self,
        base_task_spec: TaskSpec,
        *,
        seed: int,
        sample_index: int,
        x_offset_m: float,
        y_offset_m: float,
        yaw_offset_rad: float,
    ) -> Order9R1ReachablePoseSample:
        applied = apply_order9_r1_pose_offsets(
            base_task_spec,
            seed=seed,
            sample_index=sample_index,
            x_offset_m=x_offset_m,
            y_offset_m=y_offset_m,
            yaw_offset_rad=yaw_offset_rad,
            authorization_kind="accepted_distribution_manifest",
            authorization_path=self.loaded.path,
            authorization_sha256=self.loaded.sha256,
        )
        sample = Order9R1ReachablePoseSample(
            seed=seed,
            sample_index=sample_index,
            task_spec=applied.task_spec,
            initial_x_offset_m=x_offset_m,
            initial_y_offset_m=y_offset_m,
            initial_yaw_offset_rad=yaw_offset_rad,
            distribution_manifest_sha256=self.loaded.sha256,
            support_audit=applied.support_audit,
        )
        sample.validate()
        return sample


def apply_order9_r1_pose_offsets(
    base_task_spec: TaskSpec,
    *,
    seed: int,
    sample_index: int,
    x_offset_m: float,
    y_offset_m: float,
    yaw_offset_rad: float,
    authorization_kind: str,
    authorization_path: str,
    authorization_sha256: str,
) -> Order9R1AppliedPose:
    """Apply a hash-authorized planar object/goal transform without simulation."""

    if seed < 0 or sample_index < 0:
        raise ValueError("Order9 R1 seed and sample_index must be non-negative")
    if authorization_kind not in {
        "accepted_distribution_manifest",
        "approved_calibration_protocol",
    }:
        raise ValueError("Order9 R1 pose authorization kind is invalid")
    if not authorization_path:
        raise ValueError("Order9 R1 pose authorization path is empty")
    _require_sha256(authorization_sha256, "Order9 R1 pose authorization")
    for name, value in (
        ("x_offset_m", x_offset_m),
        ("y_offset_m", y_offset_m),
        ("yaw_offset_rad", yaw_offset_rad),
    ):
        if not math.isfinite(float(value)):
            raise ValueError(f"Order9 R1 {name} must be finite")

    task_data = base_task_spec.to_dict()
    object_data = _target_object_data(task_data)
    geometry_data = _geometry_data(task_data, str(object_data["geometry_id"]))
    if geometry_data.get("geometry_type") != GeometryType.BOX.value:
        raise SchemaValidationError(
            "Order9 R1 reachable-pose expansion requires the canonical box family"
        )
    object_pose = object_data.get("pose_world")
    if not isinstance(object_pose, list) or len(object_pose) != 7:
        raise SchemaValidationError("Order9 R1 target object requires pose_world")
    delta = _yaw_quaternion(yaw_offset_rad)
    object_data["pose_world"] = [
        float(object_pose[0]) + x_offset_m,
        float(object_pose[1]) + y_offset_m,
        float(object_pose[2]),
        *_quaternion_multiply(delta, object_pose[3:7]),
    ]
    for goal in task_data["goals"]:
        if goal.get("target_entity_id") != object_data["object_id"]:
            continue
        target_pose = goal.get("target_pose_world")
        if not isinstance(target_pose, list) or len(target_pose) != 7:
            continue
        goal["target_pose_world"] = [
            float(target_pose[0]) + x_offset_m,
            float(target_pose[1]) + y_offset_m,
            float(target_pose[2]),
            *_quaternion_multiply(delta, target_pose[3:7]),
        ]
    task_data["task_id"] = (
        f"{base_task_spec.task_id}_order9_r1_{sample_index:06d}_{seed:08d}"
    )
    randomized_task = TaskSpec.from_dict(task_data)
    support_audit = audit_order9_r1_support(base_task_spec, randomized_task)
    metadata = dict(task_data.get("metadata", {}) or {})
    metadata.update(
        {
            "randomization_family": ORDER9_R1_OBJECT_DISTRIBUTION,
            "randomization_version": ORDER9_R1_RANDOMIZATION_VERSION,
            "randomization_seed": seed,
            "randomization_sample_index": sample_index,
            "r1_authorization_kind": authorization_kind,
            "r1_authorization_path": authorization_path,
            "r1_authorization_sha256": authorization_sha256,
            "r1_initial_x_offset_m": x_offset_m,
            "r1_initial_y_offset_m": y_offset_m,
            "r1_initial_yaw_offset_rad": yaw_offset_rad,
            "r1_object_properties_preserved": True,
            "r1_goal_relative_transform_preserved": True,
            "r1_support_geometry_preserved": True,
            "r1_support_pose_preserved": True,
            "r1_robot_reset_preserved": True,
            "r1_support_geometry_hash": support_audit.support_geometry_hash,
            "r1_support_projected_com_margin_m": (
                support_audit.minimum_projected_com_margin_m
            ),
            "r1_support_full_footprint_margin_m": (
                support_audit.minimum_full_footprint_margin_m
            ),
        }
    )
    if authorization_kind == "accepted_distribution_manifest":
        metadata.update(
            {
                "r1_distribution_manifest_path": authorization_path,
                "r1_distribution_manifest_sha256": authorization_sha256,
            }
        )
    else:
        metadata.update(
            {
                "r1_calibration_protocol_path": authorization_path,
                "r1_calibration_protocol_sha256": authorization_sha256,
            }
        )
    task_data["metadata"] = metadata
    task_spec = TaskSpec.from_dict(task_data)
    return Order9R1AppliedPose(task_spec=task_spec, support_audit=support_audit)


def audit_order9_r1_support(
    base_task_spec: TaskSpec,
    randomized_task_spec: TaskSpec,
) -> Order9R1SupportAudit:
    """Measure sampled start/goal margins against the frozen C3 source support."""

    collision = build_order9_c3_posture_collision_object(base_task_spec)
    if len(collision.environment_boxes) != 1:
        raise SchemaValidationError("Order9 R1 requires one frozen source support")
    support = collision.environment_boxes[0]
    target_id = next(
        (
            goal.target_entity_id
            for goal in randomized_task_spec.goals
            if goal.goal_type == "object_pose"
        ),
        None,
    )
    object_spec = next(
        (
            value
            for value in randomized_task_spec.scene.objects
            if value.object_id == target_id
        ),
        None,
    )
    if object_spec is None:
        raise SchemaValidationError("Order9 R1 support audit target is missing")
    geometry = next(
        (
            value
            for value in randomized_task_spec.scene.geometry_library
            if value.geometry_id == object_spec.geometry_id
        ),
        None,
    )
    if geometry is None or geometry.geometry_type != GeometryType.BOX:
        raise SchemaValidationError("Order9 R1 support audit requires box geometry")
    raw_size = geometry.primitive_params.get("size_m")
    if not isinstance(raw_size, (list, tuple)) or len(raw_size) != 3:
        raise SchemaValidationError("Order9 R1 support audit box size is missing")
    object_size = tuple(
        float(raw_size[index]) * float(geometry.scale[index]) for index in range(3)
    )
    target_poses = [
        goal.target_pose_world
        for goal in randomized_task_spec.goals
        if goal.target_entity_id == object_spec.object_id
        and goal.target_pose_world is not None
    ]
    if len(target_poses) != 1:
        raise SchemaValidationError("Order9 R1 support audit requires one pose goal")
    start_com, start_footprint = _support_pose_margins(
        object_spec.pose_world,
        center_of_mass_object=object_spec.center_of_mass_object or (0.0, 0.0, 0.0),
        object_size_m=object_size,
        support_pose_world=support.pose_world,
        support_size_m=support.size_m,
    )
    goal_com, goal_footprint = _support_pose_margins(
        target_poses[0],
        center_of_mass_object=object_spec.center_of_mass_object or (0.0, 0.0, 0.0),
        object_size_m=object_size,
        support_pose_world=support.pose_world,
        support_size_m=support.size_m,
    )
    audit = Order9R1SupportAudit(
        support_geometry_hash=stable_hash(
            {
                "box_id": support.box_id,
                "size_m": support.size_m,
                "pose_world": support.pose_world,
            }
        ),
        support_center_xy_m=(support.pose_world[0], support.pose_world[1]),
        support_size_xy_m=(support.size_m[0], support.size_m[1]),
        start_projected_com_margin_m=start_com,
        goal_projected_com_margin_m=goal_com,
        start_full_footprint_margin_m=start_footprint,
        goal_full_footprint_margin_m=goal_footprint,
    )
    audit.validate()
    return audit


def load_order9_r1_distribution_manifest(
    path: str | Path,
) -> LoadedOrder9R1DistributionManifest:
    source = Path(path)
    manifest = Order9R1DistributionManifest.from_json(
        source.read_text(encoding="utf-8")
    )
    manifest.validate()
    return LoadedOrder9R1DistributionManifest(
        manifest=manifest,
        path=str(source.resolve()),
        sha256=hash_file(source),
    )


def _target_object_data(task_data: dict[str, Any]) -> dict[str, Any]:
    target_id = next(
        (
            goal.get("target_entity_id")
            for goal in task_data["goals"]
            if goal.get("goal_type") == "object_pose"
        ),
        None,
    )
    for obj in task_data["scene"]["objects"]:
        if obj.get("object_id") == target_id:
            return obj
    raise SchemaValidationError("Order9 R1 requires an object_pose target")


def _geometry_data(task_data: dict[str, Any], geometry_id: str) -> dict[str, Any]:
    for geometry in task_data["scene"]["geometry_library"]:
        if geometry.get("geometry_id") == geometry_id:
            return geometry
    raise SchemaValidationError(f"Order9 R1 geometry {geometry_id!r} is missing")


def _yaw_quaternion(yaw_rad: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0))


def _quaternion_multiply(
    left: tuple[float, float, float, float] | list[float],
    right: tuple[float, float, float, float] | list[float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = (float(value) for value in left)
    rx, ry, rz, rw = (float(value) for value in right)
    output = (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )
    norm = math.sqrt(sum(value * value for value in output))
    if not math.isfinite(norm) or norm <= 0.0:
        raise SchemaValidationError("Order9 R1 encountered an invalid quaternion")
    return tuple(value / norm for value in output)


def _support_pose_margins(
    object_pose_world,
    *,
    center_of_mass_object,
    object_size_m: tuple[float, float, float],
    support_pose_world,
    support_size_m: tuple[float, float, float],
) -> tuple[float, float]:
    object_yaw = _quaternion_yaw(object_pose_world[3:7])
    support_yaw = _quaternion_yaw(support_pose_world[3:7])
    relative_yaw = object_yaw - support_yaw
    cos_yaw = abs(math.cos(relative_yaw))
    sin_yaw = abs(math.sin(relative_yaw))
    object_half_x = 0.5 * (object_size_m[0] * cos_yaw + object_size_m[1] * sin_yaw)
    object_half_y = 0.5 * (object_size_m[0] * sin_yaw + object_size_m[1] * cos_yaw)
    center_dx, center_dy = _xy_in_support_frame(
        object_pose_world[0],
        object_pose_world[1],
        support_pose_world,
        support_yaw,
    )
    rotated_com = _rotate_vector_by_quaternion(
        center_of_mass_object,
        object_pose_world[3:7],
    )
    com_dx, com_dy = _xy_in_support_frame(
        float(object_pose_world[0]) + rotated_com[0],
        float(object_pose_world[1]) + rotated_com[1],
        support_pose_world,
        support_yaw,
    )
    support_half_x = 0.5 * float(support_size_m[0])
    support_half_y = 0.5 * float(support_size_m[1])
    projected_com_margin = min(
        support_half_x - abs(com_dx),
        support_half_y - abs(com_dy),
    )
    full_footprint_margin = min(
        support_half_x - abs(center_dx) - object_half_x,
        support_half_y - abs(center_dy) - object_half_y,
    )
    return projected_com_margin, full_footprint_margin


def _xy_in_support_frame(
    x_world: float,
    y_world: float,
    support_pose_world,
    support_yaw: float,
) -> tuple[float, float]:
    dx = float(x_world) - float(support_pose_world[0])
    dy = float(y_world) - float(support_pose_world[1])
    cosine = math.cos(support_yaw)
    sine = math.sin(support_yaw)
    return cosine * dx + sine * dy, -sine * dx + cosine * dy


def _quaternion_yaw(quaternion) -> float:
    x, y, z, w = (float(value) for value in quaternion)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _rotate_vector_by_quaternion(vector, quaternion) -> tuple[float, float, float]:
    vx, vy, vz = (float(value) for value in vector)
    qx, qy, qz, qw = (float(value) for value in quaternion)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def derive_order9_r1_seed(seed: int, sample_index: int) -> int:
    """Versioned deterministic seed derivation shared by calibration/collection."""

    if seed < 0 or sample_index < 0:
        raise ValueError("Order9 R1 seed inputs must be non-negative")
    return (int(seed) * 0x9E3779B185EBCA87 + int(sample_index)) & ((1 << 64) - 1)


def _require_sha256(value: str, path: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise SchemaValidationError(f"{path} must be a lowercase SHA-256 digest")
