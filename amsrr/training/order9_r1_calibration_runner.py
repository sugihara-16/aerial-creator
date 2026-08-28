from __future__ import annotations

"""Fail-closed execution order for the approved Order-9 R1 calibration.

The runner owns case enumeration and admission.  Heavy work is supplied by
callers, but a full controller/Isaac replay is never callable until the same
candidate has a complete deterministic teacher+IK result and has passed the
CPU-only exact-tracking screen.
"""

import math
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Literal

from amsrr.schemas.common import SchemaBase, SchemaValidationError
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.order9 import Order9ArtifactBinding
from amsrr.schemas.task_spec import TaskSpec
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.training.order9_c3_nominal_trajectory import (
    Order9C3NominalTrajectory,
    generate_order9_c3_nominal_grasp_trajectory,
)
from amsrr.training.order9_r1_calibration import (
    Order9R1CalibrationLevel,
    Order9R1CalibrationProtocol,
    load_order9_r1_calibration_approval_record,
    load_order9_r1_calibration_protocol,
)
from amsrr.training.order9_r1_fast_screen import (
    Order9R1FastScreenConfig,
    Order9R1FastScreenResult,
    screen_order9_r1_nominal_candidate,
)
from amsrr.training.order9_r1_joint_reserve import (
    ORDER9_R1_ANCHOR_POSITION_TOLERANCE_M,
    project_order9_r1_nominal_joint_reserve,
)
from amsrr.training.order9_r1_randomization import (
    Order9R1SupportAudit,
    apply_order9_r1_pose_offsets,
    derive_order9_r1_seed,
)
from amsrr.training.order9_rollout_buckets import (
    Order9PiLRolloutBucket,
    load_order9_pi_l_rollout_bucket_manifest,
    validate_order9_pi_l_rollout_bucket_bytes,
)
from amsrr.utils.hashing import hash_file

ORDER9_R1_CALIBRATION_RUNNER_VERSION = "order9_r1_calibration_runner_v1"
ORDER9_R1_FULL_LAYER_EXECUTION_CHAIN = (
    "deterministic_pi_h_teacher_deterministic_ik_c3_pi_l_"
    "qpid_qp_local_servo_isaac_v1"
)
ORDER9_R1_C3_CHECKPOINT_SHA256 = (
    "6ea412ccdfe983cb2b030b3514bc8982424522673b49def188d547120984357b"
)


@dataclass(frozen=True)
class Order9R1CalibrationCase:
    candidate_id: str
    global_case_index: int
    level_id: str
    split: str
    source_bucket: Order9PiLRolloutBucket
    sample_kind: Literal["lattice", "interior"]
    sample_index: int
    seed: int
    x_offset_m: float
    y_offset_m: float
    yaw_offset_rad: float
    task_spec: TaskSpec
    support_audit: Order9R1SupportAudit
    physical_model_hash: str


@dataclass(frozen=True)
class Order9R1PreparedCandidate:
    """Result of deterministic pi_H teacher generation plus deterministic IK."""

    candidate_id: str
    candidate_task_hash: str
    morphology_hash: str
    physical_model_hash: str
    teacher_trajectory_complete: bool
    failure_reason: str | None
    payload: Any = None

    def validate_for(self, case: Order9R1CalibrationCase) -> None:
        if (
            self.candidate_id != case.candidate_id
            or self.candidate_task_hash != case.task_spec.stable_hash()
            or self.morphology_hash != case.source_bucket.morphology_hash
            or self.physical_model_hash != case.physical_model_hash
        ):
            raise SchemaValidationError(
                "Order9 R1 prepared-candidate identity mismatch"
            )
        if self.teacher_trajectory_complete == (self.failure_reason is not None):
            raise SchemaValidationError(
                "Order9 R1 teacher completion/failure reason is inconsistent"
            )


@dataclass(frozen=True)
class Order9R1FullLayerReplayResult:
    candidate_id: str
    replay_index: int
    success: bool
    safety_failure: bool
    fallback_used: bool
    failure_reason: str | None
    execution_chain: str = ORDER9_R1_FULL_LAYER_EXECUTION_CHAIN
    c3_checkpoint_sha256: str = ORDER9_R1_C3_CHECKPOINT_SHA256
    deterministic_teacher_used: bool = True
    deterministic_ik_used: bool = True
    c3_pi_l_used: bool = True
    qpid_qp_used: bool = True
    local_servo_used: bool = True
    isaac_used: bool = True

    def validate_for(self, case: Order9R1CalibrationCase, replay_index: int) -> None:
        if self.candidate_id != case.candidate_id or self.replay_index != replay_index:
            raise SchemaValidationError("Order9 R1 full-layer replay identity mismatch")
        if (
            self.execution_chain != ORDER9_R1_FULL_LAYER_EXECUTION_CHAIN
            or self.c3_checkpoint_sha256 != ORDER9_R1_C3_CHECKPOINT_SHA256
            or not all(
                (
                    self.deterministic_teacher_used,
                    self.deterministic_ik_used,
                    self.c3_pi_l_used,
                    self.qpid_qp_used,
                    self.local_servo_used,
                    self.isaac_used,
                )
            )
        ):
            raise SchemaValidationError("Order9 R1 full-layer execution chain mismatch")
        if self.success and (
            self.safety_failure or self.fallback_used or self.failure_reason is not None
        ):
            raise SchemaValidationError(
                "successful Order9 R1 replay has failure evidence"
            )
        if not self.success and self.failure_reason is None:
            raise SchemaValidationError("failed Order9 R1 replay lacks a reason")


@dataclass(frozen=True)
class Order9R1CalibrationCaseOutcome:
    case: Order9R1CalibrationCase
    support_gate_passed: bool
    teacher_feasible: bool
    fast_screen: Order9R1FastScreenResult | None
    full_layer_replays: tuple[Order9R1FullLayerReplayResult, ...]
    failure_reason: str | None

    @property
    def passed(self) -> bool:
        return bool(
            self.support_gate_passed
            and self.teacher_feasible
            and self.fast_screen is not None
            and self.fast_screen.accepted
            and self.full_layer_replays
            and all(replay.success for replay in self.full_layer_replays)
            and not any(
                replay.safety_failure or replay.fallback_used
                for replay in self.full_layer_replays
            )
        )


@dataclass
class Order9R1CalibrationLevelResult(SchemaBase):
    level_id: str
    split: str
    bucket_count: int
    case_count: int
    lattice_case_count: int
    interior_case_count: int
    support_gate_success_count: int
    teacher_success_count: int
    fast_screen_attempt_count: int
    fast_screen_success_count: int
    isaac_attempt_count: int
    isaac_success_count: int
    lattice_pass_count: int
    interior_teacher_success_count: int
    interior_fast_screen_attempt_count: int
    interior_fast_screen_success_count: int
    interior_isaac_attempt_count: int
    interior_isaac_success_count: int
    safety_failure_count: int
    fallback_count: int
    minimum_support_projected_com_margin_m: float
    passed: bool
    failed_gates: tuple[str, ...]

    def validate(self) -> None:
        if not self.level_id or self.split not in {"train", "validation"}:
            raise SchemaValidationError("Order9 R1 level-result identity is invalid")
        if (
            self.bucket_count < 1
            or self.case_count != self.lattice_case_count + self.interior_case_count
            or self.case_count < 1
        ):
            raise SchemaValidationError(
                "Order9 R1 level-result case counts are invalid"
            )
        for successes, attempts in (
            (self.support_gate_success_count, self.case_count),
            (self.teacher_success_count, self.case_count),
            (self.fast_screen_success_count, self.fast_screen_attempt_count),
            (self.isaac_success_count, self.isaac_attempt_count),
            (self.lattice_pass_count, self.lattice_case_count),
            (self.interior_teacher_success_count, self.interior_case_count),
            (
                self.interior_fast_screen_success_count,
                self.interior_fast_screen_attempt_count,
            ),
            (self.interior_isaac_success_count, self.interior_isaac_attempt_count),
        ):
            if attempts < 0 or successes < 0 or successes > attempts:
                raise SchemaValidationError("Order9 R1 level-result counts are invalid")
        if not math.isfinite(self.minimum_support_projected_com_margin_m):
            raise SchemaValidationError("Order9 R1 support margin is not finite")
        if self.passed == bool(self.failed_gates):
            raise SchemaValidationError("Order9 R1 level pass/failure gates disagree")

    @staticmethod
    def _rate(successes: int, attempts: int) -> float:
        return 0.0 if attempts == 0 else float(successes) / float(attempts)

    @property
    def teacher_success_rate(self) -> float:
        return self._rate(self.teacher_success_count, self.case_count)

    @property
    def fast_screen_success_rate(self) -> float:
        return self._rate(
            self.fast_screen_success_count, self.fast_screen_attempt_count
        )

    @property
    def isaac_success_rate(self) -> float:
        return self._rate(self.isaac_success_count, self.isaac_attempt_count)

    @property
    def interior_teacher_success_rate(self) -> float:
        return self._rate(self.interior_teacher_success_count, self.interior_case_count)

    @property
    def interior_fast_screen_success_rate(self) -> float:
        return self._rate(
            self.interior_fast_screen_success_count,
            self.interior_fast_screen_attempt_count,
        )

    @property
    def interior_isaac_success_rate(self) -> float:
        return self._rate(
            self.interior_isaac_success_count,
            self.interior_isaac_attempt_count,
        )


@dataclass
class Order9R1CalibrationRunResult(SchemaBase):
    runner_version: str
    status: Literal["accepted", "rejected"]
    approved_protocol: Order9ArtifactBinding
    approval_record: Order9ArtifactBinding
    source_bucket_manifest: Order9ArtifactBinding
    selection_results: list[Order9R1CalibrationLevelResult]
    selected_level_id: str | None
    confirmation_result: Order9R1CalibrationLevelResult | None
    confirmation_attempt_count: int
    accepted_level_id: str | None
    confirmation_failure_stepback_forbidden: bool
    full_control_requires_fast_screen_pass: bool
    case_evidence_count: int
    measured_wall_time_s: float

    def validate(self) -> None:
        if self.runner_version != ORDER9_R1_CALIBRATION_RUNNER_VERSION:
            raise SchemaValidationError("Order9 R1 calibration runner version mismatch")
        for artifact in (
            self.approved_protocol,
            self.approval_record,
            self.source_bucket_manifest,
        ):
            artifact.validate()
        if (
            self.approved_protocol.artifact_kind != "approved_calibration_protocol"
            or self.approval_record.artifact_kind != "calibration_gate_approval"
            or self.source_bucket_manifest.artifact_kind
            != "c3_accepted_bucket_manifest"
        ):
            raise SchemaValidationError("Order9 R1 run artifact roles mismatch")
        if not self.selection_results or self.case_evidence_count < 1:
            raise SchemaValidationError("Order9 R1 run has no selection evidence")
        for result in self.selection_results:
            result.validate()
        if self.confirmation_result is not None:
            self.confirmation_result.validate()
        if (
            self.confirmation_attempt_count not in {0, 1}
            or (self.confirmation_result is None)
            != (self.confirmation_attempt_count == 0)
            or not self.confirmation_failure_stepback_forbidden
            or not self.full_control_requires_fast_screen_pass
            or not math.isfinite(self.measured_wall_time_s)
            or self.measured_wall_time_s <= 0.0
        ):
            raise SchemaValidationError("Order9 R1 run ordering/measurement is invalid")
        accepted = (
            self.selected_level_id is not None
            and self.confirmation_result is not None
            and self.confirmation_result.passed
            and self.accepted_level_id == self.selected_level_id
        )
        if (self.status == "accepted") != accepted:
            raise SchemaValidationError("Order9 R1 run acceptance is inconsistent")
        if self.status == "rejected" and self.accepted_level_id is not None:
            raise SchemaValidationError("rejected Order9 R1 run has an accepted level")


PrepareCandidate = Callable[[Order9R1CalibrationCase], Order9R1PreparedCandidate]
FastScreen = Callable[
    [Order9R1CalibrationCase, Order9R1PreparedCandidate],
    Order9R1FastScreenResult,
]
FullLayerEvaluator = Callable[
    [Order9R1CalibrationCase, Order9R1PreparedCandidate, int],
    Order9R1FullLayerReplayResult,
]
CaseResultSink = Callable[[Order9R1CalibrationCaseOutcome], None]


class Order9R1DeterministicTeacherScreenPipeline:
    """Production adapter for deterministic teacher+IK and the CPU screen.

    Constructing this adapter is read-only. Calling :meth:`prepare` performs
    teacher generation/IK; calling :meth:`screen` only scans that in-memory
    result and never invokes Isaac or a controller.
    """

    def __init__(
        self,
        *,
        repository_root: str | Path,
        source_bucket_manifest_path: str | Path,
        physical_model_config_path: str | Path = "configs/robot/robot_model.yaml",
        minimum_normalized_joint_limit_reserve: float = 0.01,
        maximum_body_tilt_rad: float = math.radians(60.0),
        anchor_position_tolerance_m: float = (ORDER9_R1_ANCHOR_POSITION_TOLERANCE_M),
        enforce_joint_limit_reserve_during_ik: bool = False,
    ) -> None:
        self.repository = Path(repository_root).resolve()
        bucket_source = Path(source_bucket_manifest_path)
        if not bucket_source.is_absolute():
            bucket_source = self.repository / bucket_source
        self.bucket_manifest_path = bucket_source.resolve()
        physical_source = Path(physical_model_config_path)
        if not physical_source.is_absolute():
            physical_source = self.repository / physical_source
        self.physical_model = build_physical_model_from_config(physical_source)
        source_manifest = load_order9_pi_l_rollout_bucket_manifest(
            self.bucket_manifest_path
        )
        if self.physical_model.stable_hash() != source_manifest.physical_model_hash:
            raise SchemaValidationError(
                "Order9 R1 teacher adapter physical-model binding mismatch"
            )
        self.screen_config = Order9R1FastScreenConfig(
            minimum_normalized_joint_limit_reserve=(
                minimum_normalized_joint_limit_reserve
            ),
            maximum_body_tilt_rad=maximum_body_tilt_rad,
        )
        if (
            not math.isfinite(float(anchor_position_tolerance_m))
            or not 0.0 < float(anchor_position_tolerance_m) <= 0.10
        ):
            raise ValueError(
                "anchor_position_tolerance_m must be finite and in (0, 0.10]"
            )
        self.anchor_position_tolerance_m = float(anchor_position_tolerance_m)
        if not isinstance(enforce_joint_limit_reserve_during_ik, bool):
            raise ValueError("enforce_joint_limit_reserve_during_ik must be boolean")
        self.enforce_joint_limit_reserve_during_ik = (
            enforce_joint_limit_reserve_during_ik
        )

    def prepare(self, case: Order9R1CalibrationCase) -> Order9R1PreparedCandidate:
        graph_path = (
            self.bucket_manifest_path.parent / case.source_bucket.morphology_graph_path
        )
        if hash_file(graph_path) != case.source_bucket.morphology_graph_sha256:
            raise SchemaValidationError("Order9 R1 source morphology bytes changed")
        graph = MorphologyGraph.from_json(graph_path.read_text(encoding="utf-8"))
        if graph.stable_hash() != case.source_bucket.morphology_hash:
            raise SchemaValidationError("Order9 R1 source morphology hash changed")
        binding = case.source_bucket.metadata.get("accepted_nominal_trajectory", {})
        preferred = binding.get("selected_surface_port_ids")
        preferred_ports = (
            tuple(int(value) for value in preferred)
            if isinstance(preferred, list) and len(preferred) == 2
            else None
        )
        try:
            nominal = generate_order9_c3_nominal_grasp_trajectory(
                task_spec=case.task_spec,
                structural_target=graph,
                physical_model=self.physical_model,
                preferred_surface_port_ids=preferred_ports,
            )
            if self.enforce_joint_limit_reserve_during_ik:
                nominal = project_order9_r1_nominal_joint_reserve(
                    nominal,
                    task_spec=case.task_spec,
                    physical_model=self.physical_model,
                    minimum_normalized_joint_limit_reserve=(
                        self.screen_config.minimum_normalized_joint_limit_reserve
                    ),
                    anchor_position_tolerance_m=(self.anchor_position_tolerance_m),
                )
        except SchemaValidationError as exc:
            return Order9R1PreparedCandidate(
                candidate_id=case.candidate_id,
                candidate_task_hash=case.task_spec.stable_hash(),
                morphology_hash=case.source_bucket.morphology_hash,
                physical_model_hash=case.physical_model_hash,
                teacher_trajectory_complete=False,
                failure_reason=str(exc),
            )
        return Order9R1PreparedCandidate(
            candidate_id=case.candidate_id,
            candidate_task_hash=case.task_spec.stable_hash(),
            morphology_hash=case.source_bucket.morphology_hash,
            physical_model_hash=case.physical_model_hash,
            teacher_trajectory_complete=True,
            failure_reason=None,
            payload=nominal,
        )

    def screen(
        self,
        case: Order9R1CalibrationCase,
        prepared: Order9R1PreparedCandidate,
    ) -> Order9R1FastScreenResult:
        if not prepared.teacher_trajectory_complete or not isinstance(
            prepared.payload,
            Order9C3NominalTrajectory,
        ):
            raise SchemaValidationError(
                "Order9 R1 screen requires a complete in-memory teacher result"
            )
        return screen_order9_r1_nominal_candidate(
            candidate_id=case.candidate_id,
            task_spec=case.task_spec,
            physical_model=self.physical_model,
            nominal=prepared.payload,
            config=self.screen_config,
        )


def enumerate_order9_r1_calibration_level_cases(
    *,
    protocol_path: str | Path,
    repository_root: str | Path,
    level_id: str,
    split: Literal["train", "validation"],
) -> tuple[Order9R1CalibrationCase, ...]:
    """Build one approved level's immutable candidate list without solvers."""

    repository = Path(repository_root).resolve()
    protocol_source = Path(protocol_path)
    if not protocol_source.is_absolute():
        protocol_source = repository / protocol_source
    protocol_source = protocol_source.resolve()
    try:
        protocol_relative = protocol_source.relative_to(repository)
    except ValueError as exc:
        raise SchemaValidationError(
            "Order9 R1 approved protocol must be inside the repository"
        ) from exc
    protocol = load_order9_r1_calibration_protocol(
        protocol_source,
        repository_root=repository,
    )
    if protocol.status != "approved" or protocol.approval_record is None:
        raise SchemaValidationError("Order9 R1 case enumeration requires approval")
    load_order9_r1_calibration_approval_record(
        repository / protocol.approval_record,
        repository_root=repository,
    )
    level_matches = [
        (index, level)
        for index, level in enumerate(protocol.calibration_levels)
        if level.level_id == level_id
    ]
    if len(level_matches) != 1:
        raise SchemaValidationError("Order9 R1 calibration level is unknown")
    level_index, level = level_matches[0]
    expected_split = (
        protocol.selection_split if split == "train" else protocol.confirmation_split
    )
    if split != expected_split:
        raise SchemaValidationError("Order9 R1 calibration split is invalid")
    bucket_source = (repository / protocol.source_bucket_manifest.path).resolve()
    validate_order9_pi_l_rollout_bucket_bytes(
        bucket_source,
        repository_root=repository,
    )
    manifest = load_order9_pi_l_rollout_bucket_manifest(bucket_source)
    buckets = sorted(
        (bucket for bucket in manifest.buckets if bucket.split.value == split),
        key=lambda bucket: bucket.bucket_id,
    )
    return _enumerate_level_cases(
        protocol=protocol,
        protocol_path=protocol_relative,
        protocol_sha256=hash_file(protocol_source),
        bucket_manifest_path=bucket_source,
        buckets=buckets,
        split=split,
        level=level,
        level_index=level_index,
        split_index=0 if split == protocol.selection_split else 1,
    )


def run_order9_r1_calibration(
    *,
    protocol_path: str | Path,
    repository_root: str | Path,
    prepare_candidate: PrepareCandidate,
    fast_screen: FastScreen,
    evaluate_full_layer: FullLayerEvaluator,
    case_result_sink: CaseResultSink | None = None,
) -> Order9R1CalibrationRunResult:
    """Execute the approved ladder without itself starting learning or collecting data."""

    started = perf_counter()
    repository = Path(repository_root).resolve()
    protocol_source = Path(protocol_path)
    if not protocol_source.is_absolute():
        protocol_source = repository / protocol_source
    protocol_source = protocol_source.resolve()
    try:
        protocol_relative = protocol_source.relative_to(repository)
    except ValueError as exc:
        raise SchemaValidationError(
            "Order9 R1 approved protocol must be inside the repository"
        ) from exc
    protocol = load_order9_r1_calibration_protocol(
        protocol_source,
        repository_root=repository,
    )
    if protocol.status != "approved" or protocol.approval_record is None:
        raise SchemaValidationError(
            "Order9 R1 calibration requires an approved protocol"
        )
    approval_source = (repository / protocol.approval_record).resolve()
    approval = load_order9_r1_calibration_approval_record(
        approval_source,
        repository_root=repository,
    )
    protocol_digest = hash_file(protocol_source)
    if approval.approved_protocol.sha256 != protocol_digest:
        raise SchemaValidationError("Order9 R1 runner protocol approval hash mismatch")

    bucket_source = (repository / protocol.source_bucket_manifest.path).resolve()
    validate_order9_pi_l_rollout_bucket_bytes(
        bucket_source,
        repository_root=repository,
    )
    bucket_manifest = load_order9_pi_l_rollout_bucket_manifest(bucket_source)
    if bucket_manifest.physical_model_hash != protocol.physical_model_hash:
        raise SchemaValidationError("Order9 R1 source physical-model hash mismatch")

    buckets_by_split = {
        split: sorted(
            (
                bucket
                for bucket in bucket_manifest.buckets
                if bucket.split.value == split
            ),
            key=lambda bucket: bucket.bucket_id,
        )
        for split in (protocol.selection_split, protocol.confirmation_split)
    }
    if (
        len(buckets_by_split[protocol.selection_split])
        != protocol.selection_bucket_count
        or len(buckets_by_split[protocol.confirmation_split])
        != protocol.confirmation_bucket_count
    ):
        raise SchemaValidationError("Order9 R1 runner bucket split count mismatch")

    evidence_count = 0
    selection_results: list[Order9R1CalibrationLevelResult] = []
    selected_level: Order9R1CalibrationLevel | None = None
    for level_index, level in enumerate(protocol.calibration_levels):
        result, emitted = _run_level(
            protocol=protocol,
            protocol_path=protocol_relative,
            protocol_sha256=protocol_digest,
            bucket_manifest_path=bucket_source,
            buckets=buckets_by_split[protocol.selection_split],
            split=protocol.selection_split,
            level=level,
            level_index=level_index,
            split_index=0,
            prepare_candidate=prepare_candidate,
            fast_screen=fast_screen,
            evaluate_full_layer=evaluate_full_layer,
            case_result_sink=case_result_sink,
        )
        evidence_count += emitted
        selection_results.append(result)
        if not result.passed:
            break
        selected_level = level

    confirmation: Order9R1CalibrationLevelResult | None = None
    if selected_level is not None:
        selected_index = next(
            index
            for index, value in enumerate(protocol.calibration_levels)
            if value.level_id == selected_level.level_id
        )
        confirmation, emitted = _run_level(
            protocol=protocol,
            protocol_path=protocol_relative,
            protocol_sha256=protocol_digest,
            bucket_manifest_path=bucket_source,
            buckets=buckets_by_split[protocol.confirmation_split],
            split=protocol.confirmation_split,
            level=selected_level,
            level_index=selected_index,
            split_index=1,
            prepare_candidate=prepare_candidate,
            fast_screen=fast_screen,
            evaluate_full_layer=evaluate_full_layer,
            case_result_sink=case_result_sink,
        )
        evidence_count += emitted

    accepted = confirmation is not None and confirmation.passed
    result = Order9R1CalibrationRunResult(
        runner_version=ORDER9_R1_CALIBRATION_RUNNER_VERSION,
        status="accepted" if accepted else "rejected",
        approved_protocol=Order9ArtifactBinding(
            artifact_kind="approved_calibration_protocol",
            path=protocol_relative.as_posix(),
            sha256=protocol_digest,
        ),
        approval_record=Order9ArtifactBinding(
            artifact_kind="calibration_gate_approval",
            path=approval_source.relative_to(repository).as_posix(),
            sha256=hash_file(approval_source),
        ),
        source_bucket_manifest=protocol.source_bucket_manifest,
        selection_results=selection_results,
        selected_level_id=None if selected_level is None else selected_level.level_id,
        confirmation_result=confirmation,
        confirmation_attempt_count=0 if confirmation is None else 1,
        accepted_level_id=(
            selected_level.level_id if accepted and selected_level is not None else None
        ),
        confirmation_failure_stepback_forbidden=True,
        full_control_requires_fast_screen_pass=True,
        case_evidence_count=evidence_count,
        measured_wall_time_s=max(perf_counter() - started, 1.0e-12),
    )
    result.validate()
    return result


def _run_level(
    *,
    protocol: Order9R1CalibrationProtocol,
    protocol_path: Path,
    protocol_sha256: str,
    bucket_manifest_path: Path,
    buckets: list[Order9PiLRolloutBucket],
    split: str,
    level: Order9R1CalibrationLevel,
    level_index: int,
    split_index: int,
    prepare_candidate: PrepareCandidate,
    fast_screen: FastScreen,
    evaluate_full_layer: FullLayerEvaluator,
    case_result_sink: CaseResultSink | None,
) -> tuple[Order9R1CalibrationLevelResult, int]:
    outcomes: list[Order9R1CalibrationCaseOutcome] = []
    cases = _enumerate_level_cases(
        protocol=protocol,
        protocol_path=protocol_path,
        protocol_sha256=protocol_sha256,
        bucket_manifest_path=bucket_manifest_path,
        buckets=buckets,
        split=split,
        level=level,
        level_index=level_index,
        split_index=split_index,
    )
    for case in cases:
        outcome = _run_case(
            case,
            protocol=protocol,
            prepare_candidate=prepare_candidate,
            fast_screen=fast_screen,
            evaluate_full_layer=evaluate_full_layer,
        )
        outcomes.append(outcome)
        if case_result_sink is not None:
            case_result_sink(outcome)
    return _summarize_level(protocol, level, split, len(buckets), outcomes), len(
        outcomes
    )


def _enumerate_level_cases(
    *,
    protocol: Order9R1CalibrationProtocol,
    protocol_path: Path,
    protocol_sha256: str,
    bucket_manifest_path: Path,
    buckets: list[Order9PiLRolloutBucket],
    split: str,
    level: Order9R1CalibrationLevel,
    level_index: int,
    split_index: int,
) -> tuple[Order9R1CalibrationCase, ...]:
    cases: list[Order9R1CalibrationCase] = []
    for bucket_index, bucket in enumerate(buckets):
        task_path = bucket_manifest_path.parent / bucket.task_spec_path
        base_task = TaskSpec.from_json(task_path.read_text(encoding="utf-8"))
        offsets = _case_offsets(protocol, level, level_index, split_index, bucket_index)
        for sample_kind, sample_index, x_value, y_value, yaw_value, ordinal in offsets:
            global_index = (
                ((level_index * 2 + split_index) * protocol.source_bucket_count)
                + bucket_index
            ) * (
                protocol.lattice_point_count_per_bucket
                + protocol.random_interior_samples_per_bucket
            ) + ordinal
            seed = derive_order9_r1_seed(protocol.calibration_seed, global_index)
            applied = apply_order9_r1_pose_offsets(
                base_task,
                seed=seed,
                sample_index=global_index,
                x_offset_m=x_value,
                y_offset_m=y_value,
                yaw_offset_rad=yaw_value,
                authorization_kind="approved_calibration_protocol",
                authorization_path=protocol_path.as_posix(),
                authorization_sha256=protocol_sha256,
            )
            candidate_id = (
                f"{level.level_id}__{split}__{bucket.bucket_id}__"
                f"{sample_kind}_{sample_index:02d}"
            )
            task_payload = applied.task_spec.to_dict()
            task_metadata = dict(task_payload.get("metadata", {}) or {})
            task_metadata.update(
                {
                    "order9_rollout_bucket_id": candidate_id,
                    # The protected C3 rollout implementation only accepts
                    # phase-zero evaluation on its validation execution path.
                    # Preserve the R1 source split separately: this alias is
                    # not the untouched R1 confirmation split.
                    "dataset_split": "validation",
                    "r1_calibration_candidate_id": candidate_id,
                    "r1_calibration_level_id": level.level_id,
                    "r1_calibration_source_bucket_id": bucket.bucket_id,
                    "r1_calibration_source_split": split,
                    "r1_calibration_execution_split": "validation",
                    "r1_calibration_execution_split_alias_contract": (
                        "protected_c3_formal_runner_validation_alias_v1"
                    ),
                }
            )
            task_payload["metadata"] = task_metadata
            candidate_task = TaskSpec.from_dict(task_payload)
            case = Order9R1CalibrationCase(
                candidate_id=candidate_id,
                global_case_index=global_index,
                level_id=level.level_id,
                split=split,
                source_bucket=bucket,
                sample_kind=sample_kind,
                sample_index=sample_index,
                seed=seed,
                x_offset_m=x_value,
                y_offset_m=y_value,
                yaw_offset_rad=yaw_value,
                task_spec=candidate_task,
                support_audit=applied.support_audit,
                physical_model_hash=protocol.physical_model_hash,
            )
            cases.append(case)
    return tuple(cases)


def _case_offsets(protocol, level, level_index, split_index, bucket_index):
    ordinal = 0
    values = (-level.position_half_span_m, 0.0, level.position_half_span_m)
    yaw_values = (-level.yaw_half_span_rad, 0.0, level.yaw_half_span_rad)
    for sample_index, (x_value, y_value, yaw_value) in enumerate(
        (x, y, yaw) for x in values for y in values for yaw in yaw_values
    ):
        yield "lattice", sample_index, x_value, y_value, yaw_value, ordinal
        ordinal += 1
    for sample_index in range(protocol.random_interior_samples_per_bucket):
        seed_index = (
            (level_index * 2 + split_index) * protocol.source_bucket_count
            + bucket_index
        ) * protocol.random_interior_samples_per_bucket + sample_index
        rng = random.Random(
            derive_order9_r1_seed(protocol.calibration_seed, seed_index)
        )
        yield (
            "interior",
            sample_index,
            rng.uniform(-level.position_half_span_m, level.position_half_span_m),
            rng.uniform(-level.position_half_span_m, level.position_half_span_m),
            rng.uniform(-level.yaw_half_span_rad, level.yaw_half_span_rad),
            ordinal,
        )
        ordinal += 1


def _run_case(
    case: Order9R1CalibrationCase,
    *,
    protocol: Order9R1CalibrationProtocol,
    prepare_candidate: PrepareCandidate,
    fast_screen: FastScreen,
    evaluate_full_layer: FullLayerEvaluator,
) -> Order9R1CalibrationCaseOutcome:
    support_pass = (
        case.support_audit.minimum_projected_com_margin_m
        >= protocol.minimum_support_projected_com_margin_m
    )
    if not support_pass:
        return Order9R1CalibrationCaseOutcome(
            case=case,
            support_gate_passed=False,
            teacher_feasible=False,
            fast_screen=None,
            full_layer_replays=(),
            failure_reason="projected center of mass is outside the frozen support",
        )
    prepared = prepare_candidate(case)
    prepared.validate_for(case)
    if not prepared.teacher_trajectory_complete:
        return Order9R1CalibrationCaseOutcome(
            case=case,
            support_gate_passed=True,
            teacher_feasible=False,
            fast_screen=None,
            full_layer_replays=(),
            failure_reason=prepared.failure_reason,
        )
    screen = fast_screen(case, prepared)
    _validate_screen_identity(screen, case)
    if not screen.accepted:
        return Order9R1CalibrationCaseOutcome(
            case=case,
            support_gate_passed=True,
            teacher_feasible=True,
            fast_screen=screen,
            full_layer_replays=(),
            failure_reason="fast exact-tracking screen rejected the candidate",
        )
    replay_count = (
        protocol.isaac_replays_per_lattice_case
        if case.sample_kind == "lattice"
        else protocol.isaac_replays_per_interior_case
    )
    replays = []
    for replay_index in range(replay_count):
        replay = evaluate_full_layer(case, prepared, replay_index)
        replay.validate_for(case, replay_index)
        replays.append(replay)
    failure_reason = next(
        (replay.failure_reason for replay in replays if not replay.success),
        None,
    )
    return Order9R1CalibrationCaseOutcome(
        case=case,
        support_gate_passed=True,
        teacher_feasible=True,
        fast_screen=screen,
        full_layer_replays=tuple(replays),
        failure_reason=failure_reason,
    )


def _validate_screen_identity(
    screen: Order9R1FastScreenResult,
    case: Order9R1CalibrationCase,
) -> None:
    if (
        screen.candidate_id != case.candidate_id
        or screen.candidate_task_hash != case.task_spec.stable_hash()
        or screen.morphology_hash != case.source_bucket.morphology_hash
        or screen.physical_model_hash != case.physical_model_hash
    ):
        raise SchemaValidationError("Order9 R1 fast-screen identity mismatch")
    if any(
        (
            screen.isaac_invoked,
            screen.controller_layers_invoked,
            screen.ik_resolve_invoked,
            screen.trajectory_optimization_invoked,
        )
    ):
        raise SchemaValidationError("Order9 R1 fast screen invoked forbidden work")


def _summarize_level(protocol, level, split, bucket_count, outcomes):
    lattice = [value for value in outcomes if value.case.sample_kind == "lattice"]
    interior = [value for value in outcomes if value.case.sample_kind == "interior"]
    teacher_success = sum(value.teacher_feasible for value in outcomes)
    screen_attempts = teacher_success
    screen_success = sum(
        value.fast_screen is not None and value.fast_screen.accepted
        for value in outcomes
    )
    replays = [replay for value in outcomes for replay in value.full_layer_replays]
    interior_replays = [
        replay for value in interior for replay in value.full_layer_replays
    ]
    failed_gates: list[str] = []

    def gate(name: str, actual: float, minimum: float) -> None:
        if actual < minimum:
            failed_gates.append(name)

    rate = Order9R1CalibrationLevelResult._rate
    lattice_pass = sum(value.passed for value in lattice)
    interior_teacher = sum(value.teacher_feasible for value in interior)
    interior_screen_attempts = interior_teacher
    interior_screen = sum(
        value.fast_screen is not None and value.fast_screen.accepted
        for value in interior
    )
    gate(
        "lattice_pass_rate",
        rate(lattice_pass, len(lattice)),
        protocol.minimum_lattice_pass_rate,
    )
    gate(
        "teacher_feasibility_rate",
        rate(teacher_success, len(outcomes)),
        protocol.minimum_teacher_feasibility_rate,
    )
    gate(
        "fast_screen_pass_rate",
        rate(screen_success, screen_attempts),
        protocol.minimum_fast_screen_pass_rate,
    )
    gate(
        "isaac_success_rate",
        rate(sum(value.success for value in replays), len(replays)),
        protocol.minimum_isaac_success_rate,
    )
    gate(
        "interior_teacher_feasibility_rate",
        rate(interior_teacher, len(interior)),
        protocol.minimum_interior_teacher_feasibility_rate,
    )
    gate(
        "interior_fast_screen_pass_rate",
        rate(interior_screen, interior_screen_attempts),
        protocol.minimum_interior_fast_screen_pass_rate,
    )
    gate(
        "interior_isaac_success_rate",
        rate(sum(value.success for value in interior_replays), len(interior_replays)),
        protocol.minimum_interior_isaac_success_rate,
    )
    minimum_support = min(
        value.case.support_audit.minimum_projected_com_margin_m for value in outcomes
    )
    if minimum_support < protocol.minimum_support_projected_com_margin_m:
        failed_gates.append("projected_com_support_margin")
    safety_count = sum(value.safety_failure for value in replays)
    fallback_count = sum(value.fallback_used for value in replays)
    if safety_count > protocol.maximum_safety_failure_count:
        failed_gates.append("safety_failure_count")
    if rate(fallback_count, len(replays)) > protocol.maximum_fallback_rate:
        failed_gates.append("fallback_rate")
    result = Order9R1CalibrationLevelResult(
        level_id=level.level_id,
        split=split,
        bucket_count=bucket_count,
        case_count=len(outcomes),
        lattice_case_count=len(lattice),
        interior_case_count=len(interior),
        support_gate_success_count=sum(value.support_gate_passed for value in outcomes),
        teacher_success_count=teacher_success,
        fast_screen_attempt_count=screen_attempts,
        fast_screen_success_count=screen_success,
        isaac_attempt_count=len(replays),
        isaac_success_count=sum(value.success for value in replays),
        lattice_pass_count=lattice_pass,
        interior_teacher_success_count=interior_teacher,
        interior_fast_screen_attempt_count=interior_screen_attempts,
        interior_fast_screen_success_count=interior_screen,
        interior_isaac_attempt_count=len(interior_replays),
        interior_isaac_success_count=sum(value.success for value in interior_replays),
        safety_failure_count=safety_count,
        fallback_count=fallback_count,
        minimum_support_projected_com_margin_m=minimum_support,
        passed=not failed_gates,
        failed_gates=tuple(failed_gates),
    )
    result.validate()
    return result
