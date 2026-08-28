from __future__ import annotations

"""Materialize and execute hash-bound real-Isaac R1 calibration cases."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Literal, Sequence

import torch

from amsrr.schemas.common import SchemaBase, SchemaValidationError
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.order9 import Order9ArtifactBinding
from amsrr.training.order9_c3_action_contract import (
    ORDER9_C3_ACTION_CONTRACT_CONTACT_SPACE_PROJECTED,
)
from amsrr.training.order9_c3_nominal_trajectory import (
    Order9C3NominalTrajectorySetEntry,
    Order9C3NominalTrajectorySetManifest,
    validate_order9_c3_nominal_trajectory_artifact_bytes,
    validate_order9_c3_nominal_trajectory_set_bytes,
    write_order9_c3_nominal_trajectory_artifact,
)
from amsrr.training.order9_c3_promotion import (
    load_order9_evaluation_episode_jsonl,
)
from amsrr.training.order9_pi_l_stage_runner import (
    coalesce_order9_same_morphology_collectors,
    run_parallel_order9_collectors,
)
from amsrr.training.order9_r1_calibration_runner import (
    ORDER9_R1_C3_CHECKPOINT_SHA256,
    ORDER9_R1_FULL_LAYER_EXECUTION_CHAIN,
    Order9R1CalibrationCase,
    Order9R1FullLayerReplayResult,
    Order9R1PreparedCandidate,
    enumerate_order9_r1_calibration_level_cases,
)
from amsrr.training.order9_r1_calibration import (
    load_order9_r1_calibration_approval_record,
    load_order9_r1_calibration_protocol,
)
from amsrr.training.order9_r1_fast_screen import Order9R1FastScreenResult
from amsrr.training.order9_r1_nominal_retime import (
    retime_order9_r1_nominal_artifact,
)
from amsrr.utils.hashing import hash_file

ORDER9_R1_ISAAC_CASE_VERSION = "order9_r1_isaac_calibration_case_v1"
ORDER9_R1_CALIBRATION_DECISION_VERSION = "order9_r1_calibration_decision_v1"
ORDER9_R1_EXECUTION_SPLIT_ALIAS_CONTRACT = (
    "protected_c3_formal_runner_validation_alias_v1"
)
ORDER9_R1_C3_ROLLOUT_IMPLEMENTATION_SHA256 = (
    "e813f950dde32818b7375949bfb58baa983dde760e58da22cfbb79cc33b764c2"
)
ORDER9_R1_C3_RELEASE_LEDGER_SHA256 = (
    "7515d537665c0bb10ef6bab02f1d4bb6404e76dbcedf55b8e1b455e224267729"
)
_CHECKPOINT_RELATIVE = Path(
    "artifacts/p4_full/order9/releases/c3_pi_l_promoted_update18_v1/"
    "checkpoint_update_000018.pt"
)
_CONFIG_RELATIVE = Path("configs/training/order9_learning_curriculum.yaml")
_ROLLOUT_SCRIPT_RELATIVE = Path("scripts/order9_vectorized_isaac_rollout.py")
_RELEASE_LEDGER_RELATIVE = Path("for_codex/C3_PROMOTED_UPDATE18_RELEASE_LEDGER.json")
_R1_IMPLEMENTATION_RELATIVES = (
    (
        "r1_calibration_protocol_implementation",
        Path("amsrr/training/order9_r1_calibration.py"),
    ),
    (
        "r1_calibration_enumerator_implementation",
        Path("amsrr/training/order9_r1_calibration_runner.py"),
    ),
    (
        "r1_randomization_implementation",
        Path("amsrr/training/order9_r1_randomization.py"),
    ),
    (
        "r1_fast_screen_implementation",
        Path("amsrr/training/order9_r1_fast_screen.py"),
    ),
    (
        "r1_isaac_wrapper_implementation",
        Path("amsrr/training/order9_r1_isaac_calibration.py"),
    ),
)


@dataclass
class Order9R1IsaacCaseManifest(SchemaBase):
    manifest_version: str
    candidate_id: str
    level_id: str
    split: str
    execution_split: str
    execution_split_alias_contract: str
    source_bucket_id: str
    module_count: int
    morphology_hash: str
    structural_hash: str
    seed: int
    replay_count: int
    task_spec: Order9ArtifactBinding
    morphology_graph: Order9ArtifactBinding
    robot_usd: Order9ArtifactBinding
    nominal_set: Order9ArtifactBinding
    nominal_artifact: Order9ArtifactBinding
    fast_screen_evidence: Order9ArtifactBinding
    approved_protocol: Order9ArtifactBinding
    c3_checkpoint: Order9ArtifactBinding
    c3_rollout_implementation: Order9ArtifactBinding
    c3_release_ledger: Order9ArtifactBinding
    r1_implementations: list[Order9ArtifactBinding]
    c3_promotion_evidence_eligible: bool
    training_eligible: bool
    reset_bank_path: str
    selected_gripper_friction: float
    contact_stiffness_n_per_m: float
    contact_damping_n_s_per_m: float
    estimated_mass_kg: float
    estimated_inertia_body: tuple[float, ...]
    estimated_com_object: tuple[float, ...]

    def validate(self) -> None:
        if self.manifest_version != ORDER9_R1_ISAAC_CASE_VERSION:
            raise SchemaValidationError("Order9 R1 Isaac case version mismatch")
        if (
            not self.candidate_id
            or not self.level_id
            or self.split not in {"train", "validation"}
            or self.execution_split != "validation"
            or self.execution_split_alias_contract
            != ORDER9_R1_EXECUTION_SPLIT_ALIAS_CONTRACT
            or not self.source_bucket_id
            or not 2 <= self.module_count <= 8
            or self.seed < 0
            or self.replay_count != 2
        ):
            raise SchemaValidationError("Order9 R1 Isaac case identity is invalid")
        for artifact in (
            self.task_spec,
            self.morphology_graph,
            self.robot_usd,
            self.nominal_set,
            self.nominal_artifact,
            self.fast_screen_evidence,
            self.approved_protocol,
            self.c3_checkpoint,
            self.c3_rollout_implementation,
            self.c3_release_ledger,
            *self.r1_implementations,
        ):
            artifact.validate()
        expected_roles = (
            "r1_calibration_task_spec",
            "r1_calibration_morphology_graph",
            "r1_calibration_robot_usd",
            "r1_calibration_nominal_set",
            "r1_calibration_nominal_artifact",
            "fast_kinematic_screen",
            "approved_calibration_protocol",
            "c3_promoted_pi_l_checkpoint",
            "c3_protected_rollout_implementation",
            "c3_promoted_release_ledger",
        )
        actual_roles = tuple(
            artifact.artifact_kind
            for artifact in (
                self.task_spec,
                self.morphology_graph,
                self.robot_usd,
                self.nominal_set,
                self.nominal_artifact,
                self.fast_screen_evidence,
                self.approved_protocol,
                self.c3_checkpoint,
                self.c3_rollout_implementation,
                self.c3_release_ledger,
            )
        )
        if actual_roles != expected_roles:
            raise SchemaValidationError("Order9 R1 Isaac artifact roles mismatch")
        if tuple(
            artifact.artifact_kind for artifact in self.r1_implementations
        ) != tuple(kind for kind, _path in _R1_IMPLEMENTATION_RELATIVES):
            raise SchemaValidationError("Order9 R1 implementation roles mismatch")
        if self.c3_checkpoint.sha256 != ORDER9_R1_C3_CHECKPOINT_SHA256:
            raise SchemaValidationError("Order9 R1 Isaac checkpoint binding mismatch")
        if (
            self.c3_rollout_implementation.sha256
            != ORDER9_R1_C3_ROLLOUT_IMPLEMENTATION_SHA256
            or self.c3_release_ledger.sha256 != ORDER9_R1_C3_RELEASE_LEDGER_SHA256
        ):
            raise SchemaValidationError(
                "Order9 R1 protected C3 source binding mismatch"
            )
        if self.c3_promotion_evidence_eligible or self.training_eligible:
            raise SchemaValidationError("Order9 R1 calibration case leaked authority")
        if len(self.estimated_inertia_body) != 6 or len(self.estimated_com_object) != 3:
            raise SchemaValidationError("Order9 R1 Isaac mass-property shape mismatch")
        if not self.reset_bank_path or Path(self.reset_bank_path).is_absolute():
            raise SchemaValidationError("Order9 R1 reset-bank path must be relative")


@dataclass(frozen=True)
class MaterializedOrder9R1IsaacCase:
    manifest_path: Path
    manifest: Order9R1IsaacCaseManifest


@dataclass
class Order9R1CalibrationDecisionManifest(SchemaBase):
    decision_version: str
    status: Literal["rejected"]
    calibration_gate_id: str
    approved_protocol: Order9ArtifactBinding
    approval_record: Order9ArtifactBinding
    source_c3_release_id: str
    source_c3_checkpoint_sha256: str
    selection_split: str
    selection_bucket_count: int
    attempted_level_ids: tuple[str, ...]
    failed_level_id: str
    skipped_higher_level_ids: tuple[str, ...]
    decisive_candidate_id: str
    decisive_source_bucket_id: str
    decisive_sample_kind: str
    decisive_offsets: tuple[float, float, float]
    required_case_count_at_failed_level: int
    attempted_case_count_at_failed_level: int
    unattempted_case_count_at_failed_level: int
    teacher_success_count: int
    fast_screen_success_count: int
    isaac_attempt_count: int
    isaac_success_count: int
    safety_failure_count: int
    fallback_count: int
    failed_gates: tuple[str, ...]
    failure_reason_counts: dict[str, int]
    mathematical_pass_impossible: bool
    stopping_rule: str
    confirmation_attempt_count: int
    accepted_level_id: str | None
    accepted_distribution_published: bool
    formal_collection_authorized: bool
    evidence_artifacts: list[Order9ArtifactBinding]

    def validate(self) -> None:
        if (
            self.decision_version != ORDER9_R1_CALIBRATION_DECISION_VERSION
            or self.status != "rejected"
            or self.calibration_gate_id != "r1-calibration-gate-v1"
            or self.source_c3_release_id != "c3_pi_l_promoted_update18_v1"
            or self.source_c3_checkpoint_sha256 != ORDER9_R1_C3_CHECKPOINT_SHA256
        ):
            raise SchemaValidationError(
                "Order9 R1 calibration decision identity mismatch"
            )
        self.approved_protocol.validate()
        self.approval_record.validate()
        if (
            self.selection_split != "train"
            or self.selection_bucket_count != 22
            or self.attempted_level_ids != ("r1_l1_10mm_5deg",)
            or self.failed_level_id != "r1_l1_10mm_5deg"
            or self.skipped_higher_level_ids
            != (
                "r1_l2_20mm_10deg",
                "r1_l3_30mm_15deg",
                "r1_l4_40mm_20deg",
            )
            or self.decisive_sample_kind != "lattice"
            or self.required_case_count_at_failed_level != 22 * 31
            or self.attempted_case_count_at_failed_level != 1
            or self.unattempted_case_count_at_failed_level != 22 * 31 - 1
        ):
            raise SchemaValidationError("Order9 R1 rejection coverage mismatch")
        if (
            self.teacher_success_count != 1
            or self.fast_screen_success_count != 1
            or self.isaac_attempt_count != 2
            or self.isaac_success_count != 0
            or self.safety_failure_count < 1
            or self.fallback_count != 0
            or set(self.failed_gates)
            != {"minimum_lattice_pass_rate", "maximum_safety_failure_count"}
            or self.failure_reason_counts.get("hard_collision", 0) < 1
            or not self.mathematical_pass_impossible
            or self.stopping_rule
            != "monotonic_mandatory_lattice_or_safety_gate_impossibility_v1"
        ):
            raise SchemaValidationError("Order9 R1 rejection proof mismatch")
        if (
            self.confirmation_attempt_count != 0
            or self.accepted_level_id is not None
            or self.accepted_distribution_published
            or self.formal_collection_authorized
        ):
            raise SchemaValidationError("Order9 R1 rejected decision leaked authority")
        expected_roles = {
            "approved_calibration_protocol",
            "calibration_gate_approval",
            "r1_calibration_case_manifest",
            "r1_calibration_task_spec",
            "r1_calibration_nominal_set",
            "fast_kinematic_screen",
            "r1_isaac_raw_rollout",
            "r1_isaac_evaluation_episodes",
            "r1_isaac_evaluation_log",
            "r1_isaac_reset_bank",
            "c3_protected_rollout_implementation",
            "c3_promoted_release_ledger",
            "r1_calibration_protocol_implementation",
            "r1_calibration_enumerator_implementation",
            "r1_randomization_implementation",
            "r1_fast_screen_implementation",
            "r1_isaac_wrapper_implementation",
        }
        roles = {artifact.artifact_kind for artifact in self.evidence_artifacts}
        if roles != expected_roles or len(roles) != len(self.evidence_artifacts):
            raise SchemaValidationError("Order9 R1 rejection evidence roles mismatch")
        for artifact in self.evidence_artifacts:
            artifact.validate()


def materialize_order9_r1_isaac_case(
    *,
    case: Order9R1CalibrationCase,
    prepared: Order9R1PreparedCandidate,
    screen: Order9R1FastScreenResult,
    output_dir: str | Path,
    repository_root: str | Path,
    approved_protocol_path: str | Path,
    lift_clearance_m: float = 0.3,
    nominal_time_scale: float | None = None,
    nominal_phase_time_scales: dict[str, float] | None = None,
) -> MaterializedOrder9R1IsaacCase:
    """Persist one screen-admitted trajectory as an immutable Isaac input."""

    prepared.validate_for(case)
    if not prepared.teacher_trajectory_complete or prepared.payload is None:
        raise SchemaValidationError("Order9 R1 Isaac input requires a teacher result")
    if (
        not screen.accepted
        or not screen.eligible_for_full_control_test
        or screen.candidate_id != case.candidate_id
        or screen.candidate_task_hash != case.task_spec.stable_hash()
    ):
        raise SchemaValidationError("Order9 R1 Isaac input lacks screen admission")
    repository = Path(repository_root).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Order9 R1 Isaac case output exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(dir=destination.parent, prefix=f".{destination.name}.")
    )
    try:
        task_path = temporary / "task_spec.json"
        _write_new_text(task_path, case.task_spec.to_json(indent=2) + "\n")
        screen_path = temporary / "fast_screen.json"
        _write_new_text(
            screen_path,
            json.dumps(screen.to_dict(), indent=2, sort_keys=True) + "\n",
        )
        source_manifest = _source_bucket_manifest_path(case, repository)
        graph_path = source_manifest.parent / case.source_bucket.morphology_graph_path
        robot_usd_path = _repository_path(repository, case.source_bucket.robot_usd_path)
        if (
            hash_file(graph_path) != case.source_bucket.morphology_graph_sha256
            or hash_file(robot_usd_path) != case.source_bucket.robot_usd_sha256
        ):
            raise SchemaValidationError("Order9 R1 source morphology/USD bytes changed")

        nominal_root = temporary / "nominal_set"
        nominal_root.mkdir()
        bucket_manifest_path = nominal_root / "bucket_manifest.json"
        bucket_payload = {
            "manifest_version": "order9_r1_single_case_bucket_manifest_v1",
            "buckets": [
                {
                    "bucket_id": case.candidate_id,
                    "split": "validation",
                    "r1_calibration_source_split": case.split,
                    "module_count": case.source_bucket.module_count,
                    "structural_hash": case.source_bucket.structural_hash,
                    "task_spec_sha256": hash_file(task_path),
                }
            ],
        }
        _write_new_text(
            bucket_manifest_path,
            json.dumps(bucket_payload, indent=2, sort_keys=True) + "\n",
        )
        source_nominal = case.source_bucket.metadata.get("accepted_nominal_trajectory")
        if not isinstance(source_nominal, dict):
            raise SchemaValidationError("Order9 R1 source nominal binding is missing")
        source_artifact_path = _repository_path(
            repository,
            str(source_nominal.get("artifact_path", "")),
        )
        source_artifact = json.loads(source_artifact_path.read_text(encoding="utf-8"))
        robot_urdf_path = Path(
            str(source_artifact.get("robot_urdf_path", ""))
        ).resolve()
        if not robot_urdf_path.is_file():
            raise SchemaValidationError("Order9 R1 source robot URDF is missing")

        artifact_dir = nominal_root / "buckets" / case.candidate_id
        artifact = write_order9_c3_nominal_trajectory_artifact(
            prepared.payload,
            output_dir=artifact_dir,
            bucket_id=case.candidate_id,
            split=DatasetSplit.VALIDATION,
            task_spec=case.task_spec,
            lift_clearance_m=lift_clearance_m,
            task_spec_sha256=hash_file(task_path),
            structural_hash=case.source_bucket.structural_hash,
            physical_model_hash=case.physical_model_hash,
            robot_urdf_path=robot_urdf_path,
        )
        artifact_path = artifact_dir / "manifest.json"
        retime_admission = None
        if nominal_time_scale is not None or nominal_phase_time_scales is not None:
            retime_admission = retime_order9_r1_nominal_artifact(
                artifact_path,
                time_scale=nominal_time_scale,
                phase_time_scales=nominal_phase_time_scales,
                joint_rate_limit_rad_s=(
                    float(screen.maximum_joint_rate_rad_s)
                    + float(screen.minimum_joint_rate_margin_rad_s)
                ),
            )
            artifact = validate_order9_c3_nominal_trajectory_artifact_bytes(
                artifact_path
            )
        collision_path = nominal_root / "collision_validation.json"
        _write_new_text(
            collision_path,
            json.dumps(
                {
                    "version": "order9_r1_fast_screen_collision_admission_v1",
                    "all_accepted": True,
                    "records": [
                        {
                            "bucket_id": case.candidate_id,
                            "accepted": True,
                            "fast_screen_path": _portable(
                                destination / "fast_screen.json", repository
                            ),
                            "fast_screen_sha256": hash_file(screen_path),
                            "resolved_path_hash": screen.resolved_path_hash,
                            "nominal_retime_admission": (
                                None
                                if retime_admission is None
                                else retime_admission.to_dict()
                            ),
                        }
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        entry = Order9C3NominalTrajectorySetEntry(
            bucket_id=case.candidate_id,
            split=DatasetSplit.VALIDATION,
            module_count=case.source_bucket.module_count,
            structural_hash=case.source_bucket.structural_hash,
            artifact_path=str(artifact_path.relative_to(nominal_root)),
            artifact_sha256=hash_file(artifact_path),
        )
        nominal_manifest = Order9C3NominalTrajectorySetManifest(
            bucket_manifest_path=_portable(
                destination / "nominal_set/bucket_manifest.json",
                repository,
            ),
            bucket_manifest_sha256=hash_file(bucket_manifest_path),
            physical_model_hash=case.physical_model_hash,
            entries=[entry],
            metadata={
                "r1_calibration": True,
                "candidate_id": case.candidate_id,
                "fast_screen_sha256": hash_file(screen_path),
                "nominal_time_scale": nominal_time_scale,
                "nominal_phase_time_scales": nominal_phase_time_scales,
                "nominal_retime_admission": (
                    None
                    if retime_admission is None
                    else retime_admission.to_dict()
                ),
                "training_eligible": False,
            },
        )
        nominal_manifest.validate()
        nominal_manifest_path = nominal_root / "manifest.json"
        _write_new_text(
            nominal_manifest_path,
            nominal_manifest.to_json(indent=2) + "\n",
        )
        protocol_path = Path(approved_protocol_path)
        if not protocol_path.is_absolute():
            protocol_path = repository / protocol_path
        checkpoint_path = repository / _CHECKPOINT_RELATIVE
        reset_path = destination / "reset_bank.pt"
        manifest = Order9R1IsaacCaseManifest(
            manifest_version=ORDER9_R1_ISAAC_CASE_VERSION,
            candidate_id=case.candidate_id,
            level_id=case.level_id,
            split=case.split,
            execution_split="validation",
            execution_split_alias_contract=(ORDER9_R1_EXECUTION_SPLIT_ALIAS_CONTRACT),
            source_bucket_id=case.source_bucket.bucket_id,
            module_count=case.source_bucket.module_count,
            morphology_hash=case.source_bucket.morphology_hash,
            structural_hash=case.source_bucket.structural_hash,
            seed=case.seed,
            replay_count=2,
            task_spec=_binding_at(
                "r1_calibration_task_spec",
                task_path,
                destination / "task_spec.json",
                repository,
            ),
            morphology_graph=_binding_at(
                "r1_calibration_morphology_graph",
                graph_path,
                graph_path,
                repository,
            ),
            robot_usd=_binding_at(
                "r1_calibration_robot_usd",
                robot_usd_path,
                robot_usd_path,
                repository,
            ),
            nominal_set=_binding_at(
                "r1_calibration_nominal_set",
                nominal_manifest_path,
                destination / "nominal_set/manifest.json",
                repository,
            ),
            nominal_artifact=_binding_at(
                "r1_calibration_nominal_artifact",
                artifact_path,
                destination
                / "nominal_set/buckets"
                / case.candidate_id
                / "manifest.json",
                repository,
            ),
            fast_screen_evidence=_binding_at(
                "fast_kinematic_screen",
                screen_path,
                destination / "fast_screen.json",
                repository,
            ),
            approved_protocol=_binding_at(
                "approved_calibration_protocol",
                protocol_path,
                protocol_path,
                repository,
            ),
            c3_checkpoint=_binding_at(
                "c3_promoted_pi_l_checkpoint",
                checkpoint_path,
                checkpoint_path,
                repository,
            ),
            c3_rollout_implementation=_binding_at(
                "c3_protected_rollout_implementation",
                repository / _ROLLOUT_SCRIPT_RELATIVE,
                repository / _ROLLOUT_SCRIPT_RELATIVE,
                repository,
            ),
            c3_release_ledger=_binding_at(
                "c3_promoted_release_ledger",
                repository / _RELEASE_LEDGER_RELATIVE,
                repository / _RELEASE_LEDGER_RELATIVE,
                repository,
            ),
            r1_implementations=[
                _binding_at(kind, repository / path, repository / path, repository)
                for kind, path in _R1_IMPLEMENTATION_RELATIVES
            ],
            c3_promotion_evidence_eligible=False,
            training_eligible=False,
            reset_bank_path=_portable(reset_path, repository),
            selected_gripper_friction=case.source_bucket.selected_gripper_friction,
            contact_stiffness_n_per_m=case.source_bucket.contact_stiffness_n_per_m,
            contact_damping_n_s_per_m=case.source_bucket.contact_damping_n_s_per_m,
            estimated_mass_kg=case.source_bucket.estimated_mass_kg,
            estimated_inertia_body=tuple(case.source_bucket.estimated_inertia_body),
            estimated_com_object=tuple(case.source_bucket.estimated_com_object),
        )
        manifest.validate()
        manifest_path = temporary / "case_manifest.json"
        _write_new_text(manifest_path, manifest.to_json(indent=2) + "\n")
        os.rename(temporary, destination)
        return load_order9_r1_isaac_case(destination / "case_manifest.json", repository)
    except BaseException:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_order9_r1_isaac_case(
    path: str | Path,
    repository_root: str | Path,
) -> MaterializedOrder9R1IsaacCase:
    repository = Path(repository_root).resolve()
    source = Path(path).resolve()
    manifest = Order9R1IsaacCaseManifest.from_json(source.read_text(encoding="utf-8"))
    manifest.validate()
    for artifact in (
        manifest.task_spec,
        manifest.morphology_graph,
        manifest.robot_usd,
        manifest.nominal_set,
        manifest.nominal_artifact,
        manifest.fast_screen_evidence,
        manifest.approved_protocol,
        manifest.c3_checkpoint,
        manifest.c3_rollout_implementation,
        manifest.c3_release_ledger,
        *manifest.r1_implementations,
    ):
        artifact_path = _repository_path(repository, artifact.path)
        if hash_file(artifact_path) != artifact.sha256:
            raise SchemaValidationError(
                f"Order9 R1 Isaac case bytes changed: {artifact.artifact_kind}"
            )
    validate_order9_c3_nominal_trajectory_set_bytes(
        _repository_path(repository, manifest.nominal_set.path),
        repository_root=repository,
        expected_sha256=manifest.nominal_set.sha256,
    )
    return MaterializedOrder9R1IsaacCase(source, manifest)


def order9_r1_isaac_case_command(
    materialized: MaterializedOrder9R1IsaacCase,
    *,
    repository_root: str | Path,
    python_executable: str | Path,
    rollout_steps: int = 15000,
) -> tuple[list[str], Path]:
    repository = Path(repository_root).resolve()
    manifest = materialized.manifest
    case_root = materialized.manifest_path.parent
    evaluation_root = case_root / "isaac"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    raw_path = evaluation_root / "evaluation_rollout.pt"
    jsonl_path = evaluation_root / "evaluation_episodes.jsonl"
    log_path = evaluation_root / "evaluation.log"
    seed = int(manifest.seed % (2**31 - 1))
    command = [
        str(python_executable),
        str(repository / _ROLLOUT_SCRIPT_RELATIVE),
        "--config",
        str(repository / _CONFIG_RELATIVE),
        "--stage",
        "c3_pi_l_ppo_arbitrary_morphology",
        "--pi-l-checkpoint",
        str(_repository_path(repository, manifest.c3_checkpoint.path)),
        "--pi-l-checkpoint-sha256",
        manifest.c3_checkpoint.sha256,
        "--generation-id",
        f"r1_calibration:{manifest.candidate_id}",
        "--output-raw",
        str(raw_path),
        "--split",
        manifest.execution_split,
        "--seed",
        str(seed),
        "--task-spec-json",
        str(_repository_path(repository, manifest.task_spec.path)),
        "--morphology-graph-json",
        str(_repository_path(repository, manifest.morphology_graph.path)),
        "--robot-usd",
        str(_repository_path(repository, manifest.robot_usd.path)),
        "--selected-gripper-friction",
        repr(manifest.selected_gripper_friction),
        "--contact-stiffness",
        repr(manifest.contact_stiffness_n_per_m),
        "--contact-damping",
        repr(manifest.contact_damping_n_s_per_m),
        "--estimated-mass-kg",
        repr(manifest.estimated_mass_kg),
        "--estimated-inertia-body",
        *(repr(value) for value in manifest.estimated_inertia_body),
        "--estimated-com-object",
        *(repr(value) for value in manifest.estimated_com_object),
        "--c3-nominal-set-manifest",
        str(_repository_path(repository, manifest.nominal_set.path)),
        "--c3-nominal-set-sha256",
        manifest.nominal_set.sha256,
        "--c3-nominal-artifact-sha256",
        manifest.nominal_artifact.sha256,
        "--c3-reset-bank",
        str(_repository_path(repository, manifest.reset_bank_path)),
        "--c3-action-contract",
        ORDER9_C3_ACTION_CONTRACT_CONTACT_SPACE_PROJECTED,
        "--num-envs",
        "2",
        "--rollout-steps",
        str(rollout_steps),
        "--evaluation-jsonl",
        str(jsonl_path),
        "--evaluation-episode-count",
        "2",
        "--formal-phase-zero-start",
        "--diagnostic-collision-evidence",
        "--no-tensorboard",
    ]
    return command, log_path


def run_order9_r1_isaac_cases(
    cases: Sequence[MaterializedOrder9R1IsaacCase],
    *,
    repository_root: str | Path,
    python_executable: str | Path = (
        "/home/leus/.local/share/mamba/envs/isaaclab3/bin/python"
    ),
    rollout_steps: int = 15000,
    maximum_parallel_process_count: int = 2,
    persistent_morphology_coalescing: bool = True,
) -> dict[str, tuple[Order9R1FullLayerReplayResult, ...]]:
    if not cases:
        return {}
    repository = Path(repository_root).resolve()
    commands: dict[str, list[str]] = {}
    logs: dict[str, Path] = {}
    morphology: dict[str, str] = {}
    complete: dict[str, tuple[Order9R1FullLayerReplayResult, ...]] = {}
    for case in cases:
        try:
            complete[case.manifest.candidate_id] = validate_order9_r1_isaac_result(
                case,
                repository_root=repository,
            )
            continue
        except (OSError, ValueError, SchemaValidationError):
            pass
        command, log = order9_r1_isaac_case_command(
            case,
            repository_root=repository,
            python_executable=python_executable,
            rollout_steps=rollout_steps,
        )
        name = case.manifest.candidate_id
        commands[name] = command
        logs[name] = log
        morphology[name] = case.manifest.morphology_hash
    if commands:
        if persistent_morphology_coalescing:
            commands, logs, _members = coalesce_order9_same_morphology_collectors(
                commands,
                log_paths=logs,
                morphology_hash_by_name=morphology,
                batch_root=(
                    cases[0].manifest_path.parents[2] / "persistent_process_batches"
                ),
            )
        run_parallel_order9_collectors(
            commands,
            repository_root=repository,
            log_paths=logs,
            maximum_parallel_process_count=maximum_parallel_process_count,
            process_start_stagger_s=3.0,
        )
    for case in cases:
        complete[case.manifest.candidate_id] = validate_order9_r1_isaac_result(
            case,
            repository_root=repository,
        )
    return complete


def validate_order9_r1_isaac_result(
    materialized: MaterializedOrder9R1IsaacCase,
    *,
    repository_root: str | Path,
) -> tuple[Order9R1FullLayerReplayResult, ...]:
    manifest = materialized.manifest
    isaac_root = materialized.manifest_path.parent / "isaac"
    jsonl_path = isaac_root / "evaluation_episodes.jsonl"
    raw_path = isaac_root / "evaluation_rollout.pt"
    episodes = load_order9_evaluation_episode_jsonl(jsonl_path)
    if len(episodes) != manifest.replay_count:
        raise SchemaValidationError("Order9 R1 Isaac replay count mismatch")
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    metadata = raw.get("metadata") if isinstance(raw, dict) else None
    if not isinstance(metadata, dict) or (
        metadata.get("pi_l_checkpoint_sha256") != ORDER9_R1_C3_CHECKPOINT_SHA256
        or metadata.get("c3_action_contract")
        != ORDER9_C3_ACTION_CONTRACT_CONTACT_SPACE_PROJECTED
        or metadata.get("formal_phase_zero_start") is not True
        or metadata.get("generation_id") != f"r1_calibration:{manifest.candidate_id}"
        or metadata.get("deterministic_policy") is not True
        or metadata.get("diagnostic_nominal_qpid_only") is not False
        or metadata.get("diagnostic_action_ablation") is not None
    ):
        raise SchemaValidationError("Order9 R1 raw Isaac execution contract mismatch")
    expected_generation = f"r1_calibration:{manifest.candidate_id}"
    if any(
        episode.split != DatasetSplit.VALIDATION
        or episode.metadata.get("generation_id") != expected_generation
        or episode.metadata.get("formal_phase_zero_start") is not True
        or episode.metadata.get("deterministic_policy") is not True
        for episode in episodes
    ):
        raise SchemaValidationError("Order9 R1 episode execution alias mismatch")
    raw_sha256 = hash_file(raw_path)
    if any(
        episode.source_artifact_sha256 != raw_sha256
        or Path(episode.source_artifact_path).resolve() != raw_path.resolve()
        for episode in episodes
    ):
        raise SchemaValidationError("Order9 R1 episode/raw artifact binding mismatch")
    output = []
    for replay_index, episode in enumerate(episodes):
        success = bool(
            episode.task_success
            and episode.no_fallback_success
            and not episode.safety_failure
            and episode.fallback_decision_count == 0
            and episode.isaac_backed
            and episode.full_mesh_evaluation
        )
        result = Order9R1FullLayerReplayResult(
            candidate_id=manifest.candidate_id,
            replay_index=replay_index,
            success=success,
            safety_failure=episode.safety_failure,
            fallback_used=episode.fallback_decision_count > 0,
            failure_reason=(
                None if success else (episode.failure_reason or "task failed")
            ),
            execution_chain=ORDER9_R1_FULL_LAYER_EXECUTION_CHAIN,
            c3_checkpoint_sha256=manifest.c3_checkpoint.sha256,
        )
        output.append(result)
    return tuple(output)


def finalize_order9_r1_calibration_v1_rejection(
    *,
    decisive_case_manifest_path: str | Path,
    output_path: str | Path,
    repository_root: str | Path,
    approved_protocol_path: str | Path = (
        "configs/training/order9_r1_calibration_protocol_v1.yaml"
    ),
) -> Order9R1CalibrationDecisionManifest:
    """Finalize v1 when a monotonic mandatory gate makes L1 impossible."""

    repository = Path(repository_root).resolve()
    materialized = load_order9_r1_isaac_case(
        decisive_case_manifest_path,
        repository,
    )
    case_manifest = materialized.manifest
    protocol_path = _repository_path(repository, approved_protocol_path)
    protocol = load_order9_r1_calibration_protocol(
        protocol_path,
        repository_root=repository,
    )
    if protocol.status != "approved" or protocol.approval_record is None:
        raise SchemaValidationError("Order9 R1 rejection requires approved protocol")
    approval_path = _repository_path(repository, protocol.approval_record)
    load_order9_r1_calibration_approval_record(
        approval_path,
        repository_root=repository,
    )
    expected_first = enumerate_order9_r1_calibration_level_cases(
        protocol_path=protocol_path,
        repository_root=repository,
        level_id=protocol.calibration_levels[0].level_id,
        split="train",
    )[0]
    task_path = _repository_path(repository, case_manifest.task_spec.path)
    if (
        case_manifest.candidate_id != expected_first.candidate_id
        or case_manifest.source_bucket_id != expected_first.source_bucket.bucket_id
        or case_manifest.task_spec.sha256 != hash_file(task_path)
    ):
        raise SchemaValidationError("Order9 R1 decisive case is not the first case")
    screen_path = _repository_path(
        repository,
        case_manifest.fast_screen_evidence.path,
    )
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    if (
        screen.get("accepted") is not True
        or screen.get("eligible_for_full_control_test") is not True
        or screen.get("candidate_id") != case_manifest.candidate_id
        or screen.get("isaac_invoked") is not False
        or screen.get("controller_layers_invoked") is not False
    ):
        raise SchemaValidationError("Order9 R1 decisive fast screen is invalid")
    replays = validate_order9_r1_isaac_result(
        materialized,
        repository_root=repository,
    )
    if any(replay.success for replay in replays) or not any(
        replay.safety_failure for replay in replays
    ):
        raise SchemaValidationError(
            "Order9 R1 rejection lacks a decisive Isaac safety failure"
        )
    case_root = materialized.manifest_path.parent
    raw_path = case_root / "isaac/evaluation_rollout.pt"
    episodes_path = case_root / "isaac/evaluation_episodes.jsonl"
    log_path = case_root / "isaac/evaluation.log"
    reset_path = _repository_path(repository, case_manifest.reset_bank_path)
    output = Path(output_path)
    if not output.is_absolute():
        output = repository / output
    output = output.resolve()
    failure_counts: dict[str, int] = {}
    episodes = load_order9_evaluation_episode_jsonl(episodes_path)
    for episode in episodes:
        reason = episode.failure_reason or "unknown"
        failure_counts[reason] = failure_counts.get(reason, 0) + 1
    decision = Order9R1CalibrationDecisionManifest(
        decision_version=ORDER9_R1_CALIBRATION_DECISION_VERSION,
        status="rejected",
        calibration_gate_id=protocol.calibration_gate_id,
        approved_protocol=_binding_at(
            "approved_calibration_protocol", protocol_path, protocol_path, repository
        ),
        approval_record=_binding_at(
            "calibration_gate_approval", approval_path, approval_path, repository
        ),
        source_c3_release_id=protocol.source_c3_release_id,
        source_c3_checkpoint_sha256=protocol.source_c3_checkpoint_sha256,
        selection_split=protocol.selection_split,
        selection_bucket_count=protocol.selection_bucket_count,
        attempted_level_ids=(protocol.calibration_levels[0].level_id,),
        failed_level_id=protocol.calibration_levels[0].level_id,
        skipped_higher_level_ids=tuple(
            level.level_id for level in protocol.calibration_levels[1:]
        ),
        decisive_candidate_id=case_manifest.candidate_id,
        decisive_source_bucket_id=case_manifest.source_bucket_id,
        decisive_sample_kind="lattice",
        decisive_offsets=(
            expected_first.x_offset_m,
            expected_first.y_offset_m,
            expected_first.yaw_offset_rad,
        ),
        required_case_count_at_failed_level=(
            protocol.selection_bucket_count
            * (
                protocol.lattice_point_count_per_bucket
                + protocol.random_interior_samples_per_bucket
            )
        ),
        attempted_case_count_at_failed_level=1,
        unattempted_case_count_at_failed_level=(
            protocol.selection_bucket_count
            * (
                protocol.lattice_point_count_per_bucket
                + protocol.random_interior_samples_per_bucket
            )
            - 1
        ),
        teacher_success_count=1,
        fast_screen_success_count=1,
        isaac_attempt_count=len(replays),
        isaac_success_count=sum(replay.success for replay in replays),
        safety_failure_count=sum(replay.safety_failure for replay in replays),
        fallback_count=sum(replay.fallback_used for replay in replays),
        failed_gates=(
            "minimum_lattice_pass_rate",
            "maximum_safety_failure_count",
        ),
        failure_reason_counts=failure_counts,
        mathematical_pass_impossible=True,
        stopping_rule=("monotonic_mandatory_lattice_or_safety_gate_impossibility_v1"),
        confirmation_attempt_count=0,
        accepted_level_id=None,
        accepted_distribution_published=False,
        formal_collection_authorized=False,
        evidence_artifacts=[
            _binding_at(
                "approved_calibration_protocol",
                protocol_path,
                protocol_path,
                repository,
            ),
            _binding_at(
                "calibration_gate_approval",
                approval_path,
                approval_path,
                repository,
            ),
            _binding_at(
                "r1_calibration_case_manifest",
                materialized.manifest_path,
                materialized.manifest_path,
                repository,
            ),
            case_manifest.task_spec,
            case_manifest.nominal_set,
            case_manifest.fast_screen_evidence,
            _binding_at("r1_isaac_raw_rollout", raw_path, raw_path, repository),
            _binding_at(
                "r1_isaac_evaluation_episodes",
                episodes_path,
                episodes_path,
                repository,
            ),
            _binding_at("r1_isaac_evaluation_log", log_path, log_path, repository),
            _binding_at("r1_isaac_reset_bank", reset_path, reset_path, repository),
            case_manifest.c3_rollout_implementation,
            case_manifest.c3_release_ledger,
            *case_manifest.r1_implementations,
        ],
    )
    decision.validate()
    if output.exists():
        existing = load_order9_r1_calibration_decision(output, repository)
        if existing.to_dict() != decision.to_dict():
            raise FileExistsError("Order9 R1 calibration decision output differs")
        return existing
    _write_new_text(output, decision.to_json(indent=2) + "\n")
    return load_order9_r1_calibration_decision(output, repository)


def load_order9_r1_calibration_decision(
    path: str | Path,
    repository_root: str | Path,
) -> Order9R1CalibrationDecisionManifest:
    repository = Path(repository_root).resolve()
    source = Path(path).resolve()
    decision = Order9R1CalibrationDecisionManifest.from_json(
        source.read_text(encoding="utf-8")
    )
    decision.validate()
    for artifact in decision.evidence_artifacts:
        if hash_file(_repository_path(repository, artifact.path)) != artifact.sha256:
            raise SchemaValidationError(
                f"Order9 R1 decision evidence changed: {artifact.artifact_kind}"
            )
    return decision


def _source_bucket_manifest_path(case, repository: Path) -> Path:
    # Every approved case is sourced from the one protected C3 manifest.
    return repository / (
        "artifacts/p4_full/order9/releases/c3_pi_l_promoted_update18_v1/runtime/"
        "rollout_buckets_current_lineage_v5/"
        "manifest_c3_release_smooth_clear300_current_config_min2_preload_screened_v2.json"
    )


def _repository_path(repository: Path, value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (repository / path).resolve()
    if resolved != repository and repository not in resolved.parents:
        raise SchemaValidationError("Order9 R1 artifact path escapes repository")
    return resolved


def _portable(path: Path, repository: Path) -> str:
    try:
        return path.resolve().relative_to(repository).as_posix()
    except ValueError as exc:
        raise SchemaValidationError("Order9 R1 artifact is outside repository") from exc


def _binding_at(
    kind: str,
    bytes_path: Path,
    final_path: Path,
    repository: Path,
) -> Order9ArtifactBinding:
    return Order9ArtifactBinding(
        artifact_kind=kind,
        path=_portable(final_path, repository),
        sha256=hash_file(bytes_path),
    )


def _write_new_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
