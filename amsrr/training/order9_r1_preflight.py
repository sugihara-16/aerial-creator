from __future__ import annotations

"""Fail-closed preparation checks for Order-9 R1 teacher collection."""

import json
import math
import shutil
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

from amsrr.policies.order9_low_level_policy import (
    ORDER9_CATEGORICAL_CONTACT_NORMAL_PI_L_POLICY_VERSION,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaBase, SchemaValidationError
from amsrr.schemas.order9 import Order9PolicyFamily, Order9StageRunStatus
from amsrr.training.order9_c3_nominal_trajectory import (
    validate_order9_c3_nominal_trajectory_set_bytes,
)
from amsrr.training.order9_checkpoints import (
    load_order9_policy_checkpoint,
    order9_model_config_dict,
)
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_pipeline import (
    load_order9_stage_manifest,
    order9_schedule_hash,
    order9_stage_by_id,
    preflight_order9_stage,
)
from amsrr.training.order9_r1_calibration import (
    load_order9_r1_calibration_approval_record,
    load_order9_r1_calibration_protocol,
)
from amsrr.training.order9_r1_randomization import (
    ORDER9_R1_OBJECT_DISTRIBUTION,
    Order9R1ReachablePoseRandomizer,
    load_order9_r1_distribution_manifest,
)
from amsrr.utils.hashing import hash_file

ORDER9_R1_PREFLIGHT_VERSION = "order9_r1_teacher_collection_preflight_v1"
ORDER9_R1_MINIMUM_FREE_GIB = 128.0
ORDER9_R1_STAGE_ID = "r1_teacher_trajectory_collection"
ORDER9_C3_STAGE_ID = "c3_pi_l_ppo_arbitrary_morphology"
ORDER9_C3_RELEASE_ID = "c3_pi_l_promoted_update18_v1"
ORDER9_C3_CHECKPOINT_SHA256 = (
    "6ea412ccdfe983cb2b030b3514bc8982424522673b49def188d547120984357b"
)
ORDER9_C3_PROMOTION_MANIFEST_SHA256 = (
    "366ff7eeb32885db296d8ffbfc0aa61826ecbf93ef3404adc69f05baf33eb923"
)
ORDER9_C3_ACTION_CONTRACT = "contact_space_centroidal_compatible_joint_nullspace_v4"
ORDER9_C3_ACTOR_OBSERVATION_CONTRACT = (
    "task_phase_complete_active_knot_morphology_controller_no_raw_contact_v2"
)

_LEDGER_RELATIVE = Path("for_codex/C3_PROMOTED_UPDATE18_RELEASE_LEDGER.json")
_RELEASE_RELATIVE = Path(
    "artifacts/p4_full/order9/releases/c3_pi_l_promoted_update18_v1"
)
_CURRICULUM_RELATIVE = Path("configs/training/order9_learning_curriculum.yaml")
_ROBOT_CONFIG_RELATIVE = Path("configs/robot/robot_model.yaml")
_ACCEPTED_NOMINAL_MANIFEST = Path(
    "runtime/nominal_trajectories_release_smooth_clear300_all_v16"
    "/manifest_val33_open35_v19.json"
)
_ACCEPTED_BUCKET_MANIFEST = Path(
    "runtime/rollout_buckets_current_lineage_v5"
    "/manifest_c3_release_smooth_clear300_current_config_min2_preload_screened_v2.json"
)
_APPROVED_CALIBRATION_PROTOCOL = Path(
    "configs/training/order9_r1_calibration_protocol_v1.yaml"
)
_FORMAL_CALIBRATION_DECISION = Path(
    "artifacts/p4_full/order9/r1_teacher/calibration_v1/"
    "formal_decision_final_v1/calibration_decision.json"
)


@dataclass
class Order9R1PreflightReport(SchemaBase):
    preflight_version: str
    mode: Literal["calibration", "collection"]
    passed: bool
    ready_for_calibration: bool
    ready_for_collection: bool
    checks: dict[str, bool]
    failures: list[str]
    details: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.preflight_version != ORDER9_R1_PREFLIGHT_VERSION:
            raise SchemaValidationError("Order9 R1 preflight version mismatch")
        if not self.checks:
            raise SchemaValidationError("Order9 R1 preflight has no checks")
        if self.ready_for_collection and not self.ready_for_calibration:
            raise SchemaValidationError(
                "Order9 R1 collection readiness requires calibration readiness"
            )
        expected_passed = (
            self.ready_for_calibration
            if self.mode == "calibration"
            else self.ready_for_collection
        )
        if self.passed != expected_passed:
            raise SchemaValidationError("Order9 R1 preflight readiness is inconsistent")
        if self.passed and self.failures:
            raise SchemaValidationError("passed Order9 R1 preflight contains failures")


def preflight_order9_r1_teacher_collection(
    *,
    repository_root: str | Path,
    mode: Literal["calibration", "collection"] = "calibration",
    distribution_manifest_path: str | Path | None = None,
    calibration_protocol_path: str | Path = _APPROVED_CALIBRATION_PROTOCOL,
    minimum_free_gib: float = ORDER9_R1_MINIMUM_FREE_GIB,
    require_cuda: bool = True,
    prepared_stage_output_path: str | Path | None = None,
) -> Order9R1PreflightReport:
    """Audit C3 and R1 inputs without starting collection or learning.

    ``calibration`` mode proves that a distribution-calibration pilot may
    start.  ``collection`` mode additionally requires accepted, hash-bound R1
    calibration evidence and may publish the normal stage PREPARED manifest.
    """

    if mode not in {"calibration", "collection"}:
        raise ValueError("Order9 R1 preflight mode must be calibration or collection")
    if (
        not math.isfinite(float(minimum_free_gib))
        or minimum_free_gib < ORDER9_R1_MINIMUM_FREE_GIB
    ):
        raise ValueError(
            f"Order9 R1 minimum_free_gib must be at least "
            f"{ORDER9_R1_MINIMUM_FREE_GIB:.1f}"
        )
    if mode == "calibration" and prepared_stage_output_path is not None:
        raise ValueError(
            "calibration preflight cannot publish a stage PREPARED manifest"
        )

    repository = Path(repository_root).resolve()
    release = repository / _RELEASE_RELATIVE
    ledger_path = repository / _LEDGER_RELATIVE
    curriculum_path = repository / _CURRICULUM_RELATIVE
    promotion_path = release / "stage_promoted_or_rejected.json"
    checkpoint_path = release / "checkpoint_update_000018.pt"
    nominal_path = release / _ACCEPTED_NOMINAL_MANIFEST
    bucket_path = release / _ACCEPTED_BUCKET_MANIFEST

    checks: dict[str, bool] = {}
    failures: list[str] = []
    details: dict[str, Any] = {}

    def checked(name: str, operation, *, record_result: bool = True):
        try:
            result = operation()
        except Exception as exc:  # every boundary fails closed into the report
            checks[name] = False
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            return None
        checks[name] = True
        if result is not None and record_result:
            details[name] = result
        return result

    ledger = checked(
        "c3_release_ledger_contract",
        lambda: _load_release_ledger(ledger_path),
        record_result=False,
    )
    if isinstance(ledger, dict):
        checked(
            "c3_release_direct_file_bytes",
            lambda: _verify_ledger_files(repository, ledger),
        )
        checked(
            "c3_release_implementation_inventory",
            lambda: _verify_implementation_inventory(repository, ledger),
        )
        checked(
            "c3_protected_release_permissions",
            lambda: _verify_release_permissions(
                release,
                (checkpoint_path, promotion_path, release / "release_ledger.json"),
            ),
        )

    checked(
        "c3_accepted_nominal_dependencies",
        lambda: _verify_nominal_dependencies(repository, ledger),
    )

    promotion = checked(
        "c3_promotion_manifest",
        lambda: _verify_promotion_manifest(promotion_path),
        record_result=False,
    )
    learning = checked(
        "r1_curriculum_contract",
        lambda: _verify_curriculum(curriculum_path),
        record_result=False,
    )

    if promotion is not None and learning is not None:
        checked(
            "r1_production_stage_lineage_preflight",
            lambda: _run_stage_lineage_preflight(
                learning=learning,
                promotion=promotion,
                ledger_path=ledger_path,
                checkpoint_path=checkpoint_path,
                promotion_path=promotion_path,
                nominal_path=nominal_path,
                bucket_path=bucket_path,
                distribution_manifest_path=None,
                output_path=None,
            ),
        )

    loaded_checkpoint = checked(
        "c3_v9_checkpoint_contract",
        lambda: _verify_checkpoint(checkpoint_path, learning),
        record_result=False,
    )
    if loaded_checkpoint is not None:
        checked(
            "c3_physical_model_binding",
            lambda: _verify_physical_model(
                repository / _ROBOT_CONFIG_RELATIVE,
                loaded_checkpoint.metadata.physical_model_hash,
            ),
        )

    checked(
        "cuda_environment",
        lambda: _verify_cuda(require_cuda=require_cuda),
    )
    checked(
        "storage_headroom",
        lambda: _verify_storage(
            repository,
            learning,
            minimum_free_gib=minimum_free_gib,
        ),
    )

    protocol_source = Path(calibration_protocol_path)
    if not protocol_source.is_absolute():
        protocol_source = repository / protocol_source
    approved_protocol = checked(
        "r1_calibration_protocol_approval",
        lambda: _verify_calibration_protocol_approval(
            protocol_source,
            repository=repository,
            learning=learning,
            ledger=ledger,
        ),
        record_result=False,
    )
    if approved_protocol is not None:
        details["r1_calibration_protocol_approval"] = {
            "path": str(protocol_source.resolve()),
            "sha256": hash_file(protocol_source),
            "calibration_gate_id": approved_protocol.calibration_gate_id,
            "status": approved_protocol.status,
            "selection_bucket_count": approved_protocol.selection_bucket_count,
            "confirmation_bucket_count": approved_protocol.confirmation_bucket_count,
            "level_ids": [
                level.level_id for level in approved_protocol.calibration_levels
            ],
        }

    calibration_decision = None
    calibration_decision_path = repository / _FORMAL_CALIBRATION_DECISION
    if calibration_decision_path.is_file():
        from amsrr.training.order9_r1_isaac_calibration import (
            load_order9_r1_calibration_decision,
        )

        calibration_decision = checked(
            "r1_terminal_calibration_decision",
            lambda: load_order9_r1_calibration_decision(
                calibration_decision_path,
                repository,
            ),
            record_result=False,
        )
        if calibration_decision is not None:
            details["r1_terminal_calibration_decision"] = {
                "path": str(calibration_decision_path),
                "sha256": hash_file(calibration_decision_path),
                "status": calibration_decision.status,
                "failed_level_id": calibration_decision.failed_level_id,
                "accepted_level_id": calibration_decision.accepted_level_id,
                "confirmation_attempt_count": (
                    calibration_decision.confirmation_attempt_count
                ),
                "formal_collection_authorized": (
                    calibration_decision.formal_collection_authorized
                ),
            }

    distribution = None
    if distribution_manifest_path is not None:
        distribution_path = Path(distribution_manifest_path)
        if not distribution_path.is_absolute():
            distribution_path = repository / distribution_path
        distribution = checked(
            "r1_distribution_adapter",
            lambda: _verify_distribution(
                distribution_path,
                repository=repository,
                learning=learning,
                ledger=ledger,
            ),
            record_result=False,
        )
        if distribution is not None:
            details["r1_distribution_adapter"] = {
                "path": distribution.path,
                "sha256": distribution.sha256,
                "envelope": distribution.manifest.envelope.to_dict(),
                "coordinate_frame": distribution.manifest.coordinate_frame,
                "sampling_method": distribution.manifest.sampling_method,
                "calibration_gate_id": distribution.manifest.calibration_gate_id,
                "lattice_pass_rate": distribution.manifest.lattice_pass_rate,
                "teacher_feasibility_rate": (
                    distribution.manifest.teacher_feasibility_rate
                ),
                "isaac_success_rate": distribution.manifest.isaac_success_rate,
                "confirmation_lattice_pass_rate": (
                    distribution.manifest.confirmation_lattice_pass_rate
                ),
                "confirmation_teacher_feasibility_rate": (
                    distribution.manifest.confirmation_teacher_feasibility_rate
                ),
                "confirmation_isaac_success_rate": (
                    distribution.manifest.confirmation_isaac_success_rate
                ),
                "interior_teacher_feasibility_rate": (
                    distribution.manifest.interior_teacher_feasibility_rate
                ),
                "interior_isaac_success_rate": (
                    distribution.manifest.interior_isaac_success_rate
                ),
                "minimum_support_projected_com_margin_m": (
                    distribution.manifest.minimum_support_projected_com_margin_m
                ),
            }
    elif mode == "collection":
        checks["r1_distribution_adapter"] = False
        failures.append(
            "r1_distribution_adapter: accepted calibration manifest is required"
        )
    else:
        checks["r1_distribution_adapter"] = False
        details["r1_distribution_adapter"] = {
            "status": "calibration_required_before_collection",
            "historical_expanded_position_placeholder_reused": False,
        }

    base_check_names = tuple(
        name for name in checks if name != "r1_distribution_adapter"
    )
    ready_for_calibration = (
        bool(base_check_names)
        and all(checks[name] for name in base_check_names)
        and calibration_decision is None
    )
    ready_for_collection = ready_for_calibration and distribution is not None

    if mode == "collection" and ready_for_collection:
        output_path = (
            None
            if prepared_stage_output_path is None
            else Path(prepared_stage_output_path)
        )
        if output_path is not None and not output_path.is_absolute():
            output_path = repository / output_path
        stage_result = checked(
            "r1_collection_stage_prepared_manifest",
            lambda: _run_stage_lineage_preflight(
                learning=learning,
                promotion=promotion,
                ledger_path=ledger_path,
                checkpoint_path=checkpoint_path,
                promotion_path=promotion_path,
                nominal_path=nominal_path,
                bucket_path=bucket_path,
                distribution_manifest_path=distribution.path,
                output_path=output_path,
            ),
        )
        ready_for_collection = stage_result is not None

    passed = ready_for_calibration if mode == "calibration" else ready_for_collection
    if mode == "calibration":
        failures = [
            failure
            for failure in failures
            if not failure.startswith("r1_distribution_adapter:")
        ]
    report = Order9R1PreflightReport(
        preflight_version=ORDER9_R1_PREFLIGHT_VERSION,
        mode=mode,
        passed=passed,
        ready_for_calibration=ready_for_calibration,
        ready_for_collection=ready_for_collection,
        checks=checks,
        failures=failures,
        details=details,
    )
    report.validate()
    return report


def _load_release_ledger(path: Path) -> dict[str, Any]:
    ledger = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "release_id": ORDER9_C3_RELEASE_ID,
        "stage_id": ORDER9_C3_STAGE_ID,
        "final_checkpoint_sha256": ORDER9_C3_CHECKPOINT_SHA256,
        "formal_promotion_manifest_sha256": ORDER9_C3_PROMOTION_MANIFEST_SHA256,
    }
    for name, value in expected.items():
        if ledger.get(name) != value:
            raise SchemaValidationError(f"C3 release ledger {name} mismatch")
    formal = ledger.get("formal_result")
    if not isinstance(formal, dict) or formal != {
        "bucket_count": 14,
        "episode_count": 448,
        "fallback_count": 0,
        "promoted": True,
        "safety_failure_count": 0,
        "success_count": 448,
    }:
        raise SchemaValidationError("C3 release ledger formal result mismatch")
    protected = ledger.get("protected_files")
    if not isinstance(protected, list) or len(protected) != 180:
        raise SchemaValidationError("C3 release ledger protected file count mismatch")
    return ledger


def _verify_ledger_files(repository: Path, ledger: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    for record in ledger["protected_files"]:
        expected = str(record["sha256"])
        expected_size = int(record["size_bytes"])
        for role, key in (
            ("source", "source_path"),
            ("protected_copy", "protected_copy_path"),
        ):
            path = repository / str(record[key])
            if not path.is_file():
                errors.append(f"{role}:missing:{path}")
            elif path.stat().st_size != expected_size:
                errors.append(f"{role}:size:{path}")
            elif hash_file(path) != expected:
                errors.append(f"{role}:sha256:{path}")
    release_ledger_copy = repository / _RELEASE_RELATIVE / "release_ledger.json"
    if hash_file(release_ledger_copy) != hash_file(repository / _LEDGER_RELATIVE):
        errors.append(f"protected_copy:sha256:{release_ledger_copy}")
    if errors:
        raise SchemaValidationError("; ".join(errors[:8]))
    return {"protected_file_count": len(ledger["protected_files"]), "errors": 0}


def _verify_implementation_inventory(
    repository: Path,
    ledger: dict[str, Any],
) -> dict[str, Any]:
    inventory = ledger.get("implementation_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise SchemaValidationError("C3 release implementation inventory is empty")
    errors = []
    for record in inventory:
        path = repository / str(record["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(record["size_bytes"])
            or hash_file(path) != str(record["sha256"])
        ):
            errors.append(str(path))
    if errors:
        raise SchemaValidationError(
            "C3 promotion-critical implementation changed: " + ",".join(errors[:8])
        )
    return {"implementation_file_count": len(inventory), "errors": 0}


def _verify_release_permissions(
    release: Path, files: tuple[Path, ...]
) -> dict[str, Any]:
    if not release.is_dir():
        raise SchemaValidationError("C3 protected release directory is missing")
    writable = [
        str(path)
        for path in (release, *files)
        if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    ]
    if writable:
        raise SchemaValidationError(
            "C3 protected release has writable critical paths: " + ",".join(writable)
        )
    return {"critical_path_count": len(files) + 1, "writable_path_count": 0}


def _verify_nominal_dependencies(
    repository: Path,
    ledger: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(ledger, dict):
        raise SchemaValidationError("C3 ledger is unavailable")
    roots = [
        record
        for record in ledger.get("protected_runtime_directories", [])
        if record.get("role") == "accepted_nominal_set"
    ]
    if len(roots) != 1:
        raise SchemaValidationError("C3 accepted nominal-set ledger entry is ambiguous")
    manifests = []
    for key in ("source_path", "protected_copy_path"):
        manifest = repository / str(roots[0][key]) / _ACCEPTED_NOMINAL_MANIFEST.name
        value = validate_order9_c3_nominal_trajectory_set_bytes(
            manifest,
            repository_root=repository,
        )
        manifests.append(
            {
                "role": key,
                "path": str(manifest),
                "sha256": hash_file(manifest),
                "trajectory_count": len(value.entries),
            }
        )
    if any(item["trajectory_count"] != 43 for item in manifests):
        raise SchemaValidationError("C3 accepted nominal dependency count is not 43")
    return {"validated_sets": manifests}


def _verify_promotion_manifest(path: Path):
    if hash_file(path) != ORDER9_C3_PROMOTION_MANIFEST_SHA256:
        raise SchemaValidationError("C3 protected promotion manifest SHA-256 mismatch")
    manifest = load_order9_stage_manifest(path)
    if (
        manifest.stage_id != ORDER9_C3_STAGE_ID
        or manifest.stage_index != 3
        or manifest.status != Order9StageRunStatus.PROMOTED
        or not manifest.promoted
        or manifest.policy_checkpoint_sha256_by_family.get("pi_l")
        != ORDER9_C3_CHECKPOINT_SHA256
    ):
        raise SchemaValidationError("C3 protected promotion manifest contract mismatch")
    return manifest


def _verify_curriculum(path: Path):
    learning = load_order9_learning_config(path)
    learning.validate()
    stage = order9_stage_by_id(learning, ORDER9_R1_STAGE_ID)
    expected = {
        "stage_index": 4,
        "learning_mode": "collection",
        "learning_target": "dataset",
        "min_modules": 2,
        "max_modules": 8,
        "topology_randomized": True,
        "object_distribution": ORDER9_R1_OBJECT_DISTRIBUTION,
        "deterministic_teacher_required": True,
        "pi_h_output_scope": "full_contact_wrench_trajectory",
        "minimum_episodes": 100,
        "minimum_success_rate": 0.95,
        "maximum_safety_failure_episodes": 0,
    }
    actual = stage.to_dict()
    for name, value in expected.items():
        if actual.get(name) != value:
            raise SchemaValidationError(f"R1 curriculum {name} mismatch")
    return learning


def _run_stage_lineage_preflight(
    *,
    learning,
    promotion,
    ledger_path: Path,
    checkpoint_path: Path,
    promotion_path: Path,
    nominal_path: Path,
    bucket_path: Path,
    distribution_manifest_path: str | Path | None,
    output_path: str | Path | None,
) -> dict[str, Any]:
    inputs: dict[str, str | Path] = {
        "c3_release_ledger": ledger_path,
        "c3_promoted_pi_l_checkpoint": checkpoint_path,
        "c3_promotion_manifest": promotion_path,
        "c3_accepted_nominal_manifest": nominal_path,
        "c3_accepted_bucket_manifest": bucket_path,
    }
    if distribution_manifest_path is not None:
        inputs["r1_distribution_manifest"] = distribution_manifest_path
    prepared, dataset = preflight_order9_stage(
        learning,
        stage_id=ORDER9_R1_STAGE_ID,
        input_artifact_paths=inputs,
        prior_stage_manifests=(promotion,),
        output_path=output_path,
    )
    if dataset is not None:
        raise SchemaValidationError(
            "R1 collection preflight unexpectedly loaded a dataset"
        )
    return {
        "run_id": prepared.run_id,
        "status": prepared.status.value,
        "schedule_hash": prepared.schedule_hash,
        "input_artifact_count": len(prepared.input_artifacts),
        "output_path": None if output_path is None else str(output_path),
    }


def _verify_checkpoint(path: Path, learning):
    expected_schedule = None if learning is None else order9_schedule_hash(learning)
    loaded = load_order9_policy_checkpoint(
        path,
        expected_sha256=ORDER9_C3_CHECKPOINT_SHA256,
        expected_family=Order9PolicyFamily.PI_L,
        expected_schedule_hash=expected_schedule,
    )
    metadata = loaded.metadata
    config = order9_model_config_dict(loaded.model)
    if (
        metadata.policy_version != ORDER9_CATEGORICAL_CONTACT_NORMAL_PI_L_POLICY_VERSION
        or metadata.curriculum_stage_id != ORDER9_C3_STAGE_ID
        or metadata.curriculum_stage_index != 3
        or metadata.action_contract != ORDER9_C3_ACTION_CONTRACT
        or metadata.actor_observation_contract != ORDER9_C3_ACTOR_OBSERVATION_CONTRACT
        or metadata.metadata.get("c3_action_contract")
        != "contact_space_projected_policy_command"
        or config.get("contact_normal_category_count") != 81
        or config.get("contact_normal_category_min_normalized") != -1.0
        or config.get("contact_normal_category_max_normalized") != 1.0
        or config.get("max_modules") != 8
    ):
        raise SchemaValidationError("C3 v9 policy/action/observation contract mismatch")
    return loaded


def _verify_physical_model(path: Path, expected_hash: str) -> dict[str, Any]:
    physical = build_physical_model_from_config(path)
    actual = physical.stable_hash()
    if actual != expected_hash:
        raise SchemaValidationError("C3 physical-model hash mismatch")
    return {"physical_model_hash": actual}


def _verify_cuda(*, require_cuda: bool) -> dict[str, Any]:
    available = torch.cuda.is_available()
    if require_cuda and not available:
        raise SchemaValidationError("CUDA is required for R1 calibration/collection")
    details: dict[str, Any] = {"cuda_available": available}
    if available:
        device = torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        details.update(
            {
                "device_index": device,
                "device_name": torch.cuda.get_device_name(device),
                "free_memory_bytes": int(free_bytes),
                "total_memory_bytes": int(total_bytes),
            }
        )
    return details


def _verify_storage(
    repository: Path,
    learning,
    *,
    minimum_free_gib: float,
) -> dict[str, Any]:
    artifact_root = (
        repository / "artifacts/p4_full/order9"
        if learning is None
        else repository / learning.production_runtime.artifact_root
    )
    usage = shutil.disk_usage(artifact_root)
    required_bytes = int(minimum_free_gib * (1024**3))
    if usage.free < required_bytes:
        raise SchemaValidationError(
            f"R1 storage headroom {usage.free / (1024**3):.1f} GiB is below "
            f"the requested {minimum_free_gib:.1f} GiB"
        )
    return {
        "artifact_root": str(artifact_root),
        "free_bytes": usage.free,
        "free_gib": usage.free / (1024**3),
        "minimum_free_gib": minimum_free_gib,
    }


def _verify_distribution(path: Path, *, repository: Path, learning, ledger):
    if learning is None or not isinstance(ledger, dict):
        raise SchemaValidationError(
            "R1 distribution requires valid curriculum and C3 ledger"
        )
    loaded = load_order9_r1_distribution_manifest(path)
    manifest = loaded.manifest
    if (
        manifest.source_c3_release_id != ledger["release_id"]
        or manifest.source_c3_checkpoint_sha256 != ledger["final_checkpoint_sha256"]
        or manifest.source_c3_promotion_manifest_sha256
        != ledger["formal_promotion_manifest_sha256"]
        or manifest.curriculum_schedule_hash != order9_schedule_hash(learning)
    ):
        raise SchemaValidationError(
            "R1 distribution evidence C3/schedule binding mismatch"
        )
    evidence = (
        manifest.calibration_gate_approval,
        manifest.approved_calibration_protocol,
        *manifest.evidence_artifacts,
    )
    for artifact in evidence:
        artifact_path = _repository_artifact_path(repository, artifact.path)
        if hash_file(artifact_path) != artifact.sha256:
            raise SchemaValidationError(
                f"R1 distribution evidence bytes mismatch: {artifact.artifact_kind}"
            )
    approved_protocol_path = _repository_artifact_path(
        repository,
        manifest.approved_calibration_protocol.path,
    )
    protocol = load_order9_r1_calibration_protocol(
        approved_protocol_path,
        repository_root=repository,
    )
    approval_path = _repository_artifact_path(
        repository,
        manifest.calibration_gate_approval.path,
    )
    approval = load_order9_r1_calibration_approval_record(
        approval_path,
        repository_root=repository,
    )
    if protocol.status != "approved":
        raise SchemaValidationError("R1 calibration protocol is not approved")
    if (
        protocol.calibration_gate_id != manifest.calibration_gate_id
        or protocol.approval_record != manifest.calibration_gate_approval.path
        or approval.approved_protocol.sha256
        != manifest.approved_calibration_protocol.sha256
        or protocol.fast_screen_version != manifest.fast_screen_version
        or protocol.fast_screen_execution_model != manifest.fast_screen_execution_model
        or protocol.full_control_test_requires_fast_screen_pass
        != manifest.full_control_test_requires_fast_screen_pass
        or protocol.calibration_seed != manifest.calibration_seed
        or protocol.selection_split != manifest.selection_split
        or protocol.selection_bucket_count != manifest.selection_bucket_count
        or protocol.confirmation_split != manifest.confirmation_split
        or protocol.confirmation_bucket_count != manifest.confirmation_bucket_count
        or protocol.source_c3_release_id != manifest.source_c3_release_id
        or protocol.source_c3_checkpoint_sha256 != manifest.source_c3_checkpoint_sha256
        or protocol.source_c3_promotion_manifest_sha256
        != manifest.source_c3_promotion_manifest_sha256
        or protocol.curriculum_schedule_hash != manifest.curriculum_schedule_hash
        or protocol.minimum_lattice_pass_rate
        != manifest.calibration_minimum_lattice_pass_rate
        or protocol.minimum_teacher_feasibility_rate
        != manifest.calibration_minimum_teacher_feasibility_rate
        or protocol.minimum_fast_screen_pass_rate
        != manifest.calibration_minimum_fast_screen_pass_rate
        or protocol.minimum_isaac_success_rate
        != manifest.calibration_minimum_isaac_success_rate
        or protocol.minimum_interior_teacher_feasibility_rate
        != manifest.calibration_minimum_interior_teacher_feasibility_rate
        or protocol.minimum_interior_fast_screen_pass_rate
        != manifest.calibration_minimum_interior_fast_screen_pass_rate
        or protocol.minimum_interior_isaac_success_rate
        != manifest.calibration_minimum_interior_isaac_success_rate
        or protocol.minimum_normalized_joint_limit_reserve
        != manifest.calibration_minimum_normalized_joint_limit_reserve
        or protocol.maximum_body_tilt_rad != manifest.calibration_maximum_body_tilt_rad
        or protocol.minimum_support_projected_com_margin_m
        != manifest.calibration_minimum_support_projected_com_margin_m
    ):
        raise SchemaValidationError(
            "R1 distribution does not match its approved calibration protocol"
        )
    accepted_level = next(
        (
            level
            for level in protocol.calibration_levels
            if level.level_id == manifest.accepted_level_id
        ),
        None,
    )
    if accepted_level is None:
        raise SchemaValidationError(
            "R1 accepted level is absent from approved protocol"
        )
    expected_position = (
        -accepted_level.position_half_span_m,
        accepted_level.position_half_span_m,
    )
    expected_yaw = (
        -accepted_level.yaw_half_span_rad,
        accepted_level.yaw_half_span_rad,
    )
    if (
        tuple(manifest.envelope.initial_x_offset_m) != expected_position
        or tuple(manifest.envelope.initial_y_offset_m) != expected_position
        or tuple(manifest.envelope.initial_yaw_offset_rad) != expected_yaw
        or manifest.lattice_case_count != protocol.teacher_lattice_case_count_per_level
        or manifest.teacher_feasibility_attempt_count
        != protocol.teacher_lattice_case_count_per_level
        + protocol.teacher_interior_case_count_per_level
        or manifest.fast_screen_attempt_count
        != manifest.teacher_feasibility_success_count
        or manifest.isaac_attempt_count
        != manifest.fast_screen_success_count * protocol.isaac_replays_per_interior_case
        or manifest.confirmation_lattice_case_count
        != protocol.confirmation_bucket_count * protocol.lattice_point_count_per_bucket
        or manifest.confirmation_teacher_feasibility_attempt_count
        != protocol.confirmation_teacher_case_count
        or manifest.confirmation_fast_screen_attempt_count
        != manifest.confirmation_teacher_feasibility_success_count
        or manifest.confirmation_isaac_attempt_count
        != manifest.confirmation_fast_screen_success_count
        * protocol.isaac_replays_per_interior_case
        or manifest.support_audit_attempt_count
        != manifest.teacher_feasibility_attempt_count
        or manifest.confirmation_support_audit_attempt_count
        != manifest.confirmation_teacher_feasibility_attempt_count
    ):
        raise SchemaValidationError(
            "R1 distribution envelope/case counts do not match approved protocol"
        )
    Order9R1ReachablePoseRandomizer(loaded)
    return loaded


def _verify_calibration_protocol_approval(
    path: Path,
    *,
    repository: Path,
    learning,
    ledger,
):
    path = path.resolve()
    if path != repository and repository not in path.parents:
        raise SchemaValidationError(
            "R1 calibration protocol must be inside the repository"
        )
    if learning is None or not isinstance(ledger, dict):
        raise SchemaValidationError(
            "R1 calibration approval requires valid curriculum and C3 ledger"
        )
    protocol = load_order9_r1_calibration_protocol(
        path,
        repository_root=repository,
    )
    if protocol.status != "approved" or protocol.approval_record is None:
        raise SchemaValidationError("R1 calibration protocol is not approved")
    approval_path = _repository_artifact_path(
        repository,
        protocol.approval_record,
    )
    approval = load_order9_r1_calibration_approval_record(
        approval_path,
        repository_root=repository,
    )
    if (
        approval.approved_protocol.sha256 != hash_file(path)
        or protocol.source_c3_release_id != ledger["release_id"]
        or protocol.source_c3_checkpoint_sha256 != ledger["final_checkpoint_sha256"]
        or protocol.source_c3_promotion_manifest_sha256
        != ledger["formal_promotion_manifest_sha256"]
        or protocol.curriculum_schedule_hash != order9_schedule_hash(learning)
    ):
        raise SchemaValidationError(
            "R1 calibration approval C3/schedule binding mismatch"
        )
    return protocol


def _repository_artifact_path(repository: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise SchemaValidationError(
            "R1 evidence artifact paths must be repository-relative"
        )
    resolved = (repository / relative).resolve()
    if resolved != repository and repository not in resolved.parents:
        raise SchemaValidationError("R1 evidence artifact path escapes the repository")
    return resolved
