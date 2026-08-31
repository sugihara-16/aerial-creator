#!/usr/bin/env python3
from __future__ import annotations

"""Run R1 confirmation with the corrected 19.5/30 mm clearance proof."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

# The R1 IK and geometry matrices are small.  Letting every preparation worker
# start a full OpenBLAS team oversubscribes the 32 logical CPUs and made six
# workers slower than one.  Bind one linear-algebra thread before NumPy/SciPy
# can be imported; process-level parallelism remains controlled below.
for _thread_environment_name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_environment_name] = "1"

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.feasibility.order9_native_loader_v13 import (  # noqa: E402
    activate_order9_posture_native_v13,
)

# Selection must happen before any collision solver can load the default module.
activate_order9_posture_native_v13()

from amsrr.schemas.common import SchemaValidationError  # noqa: E402
from amsrr.training.order9_r1_calibration import (  # noqa: E402
    load_order9_r1_calibration_protocol,
)
from amsrr.training.order9_r1_complete_task_geometry_v13 import (  # noqa: E402
    require_order9_r1_complete_task_geometry_v13,
)
from amsrr.training.order9_r1_complete_task_materialization_v14 import (  # noqa: E402
    ORDER9_R1_PLANNING_CLEARANCE_AUDIT_V14_FILENAME,
    ORDER9_R1_SUPPORT_CLEARANCE_CERTIFICATE_V14_FILENAME,
    finalize_order9_r1_v14_materialized_case,
    promote_order9_r1_v12_materialized_case_v14,
)
from amsrr.training.order9_r1_isaac_calibration import (  # noqa: E402
    load_order9_r1_isaac_case,
    materialize_order9_r1_isaac_case,
)
from amsrr.training.order9_r1_nominal_geometry_v11 import (  # noqa: E402
    ORDER9_R1_NOMINAL_GEOMETRY_V11_RELATIVE,
    load_order9_r1_nominal_geometry_v11_contract,
)
from amsrr.training.order9_r1_range_selection_v10 import (  # noqa: E402
    ORDER9_R1_MINIMUM_LEVEL_ID,
    ORDER9_R1_RANGE_LEVEL_IDS,
    range_case_priority,
)
from amsrr.training.order9_r1_range_selection_v13 import (  # noqa: E402
    ORDER9_R1_RANGE_TEACHER_REPAIRS_V13_RELATIVE,
    ORDER9_R1_TRAIN027_CONFIGURATION_SEED_SOURCE_CANDIDATE_ID,
    ORDER9_R1_TRAIN027_SOURCE_BUCKET_ID,
    Order9R1ExactMarginTeacherScreenPipelineV13,
    _load_bounded_local_contact_repairs,
    _load_configuration_goal_seed_evidence,
)
from amsrr.training.order9_r1_support_clearance_teacher_v12 import (  # noqa: E402
    ORDER9_R1_SUPPORT_CLEARANCE_TEACHER_V12_RELATIVE,
    load_order9_r1_support_clearance_teacher_v12_contract,
    require_order9_r1_support_clearance_generation_certificate,
)
from amsrr.utils.hashing import hash_file  # noqa: E402
from scripts import order9_run_r1_range_selection_v10 as runtime_v10  # noqa: E402
from scripts import order9_run_r1_range_selection_v11 as runtime_v11  # noqa: E402
from scripts import order9_run_r1_range_selection_v12 as runtime_v12  # noqa: E402

RUNNER_VERSION = "order9_r1_range_selection_and_confirmation_v13"
PROTOCOL = REPOSITORY / "configs/training/order9_r1_range_selection_protocol_v13.json"
APPROVAL = REPOSITORY / "for_codex/R1_RANGE_SELECTION_PROTOCOL_V13_APPROVAL.json"
BASE_PROTOCOL = runtime_v11.BASE_PROTOCOL
OUTPUT_ROOT = REPOSITORY / "artifacts/p4_full/order9/r1_teacher/range_selection_v13"
V12_OUTPUT_ROOT = REPOSITORY / "artifacts/p4_full/order9/r1_teacher/range_selection_v12"
RESULT_LEDGER_NAME = "range_selection_and_confirmation_result.json"
LEVEL_IDS = ORDER9_R1_RANGE_LEVEL_IDS
_PREPARATION_PIPELINE = None
_PREPARATION_PIPELINE_KEY = None
_V10_RUN_PROCESSES = runtime_v10._run_processes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--isaac-python",
        default="/home/leus/.local/share/mamba/envs/isaaclab3/bin/python",
    )
    parser.add_argument("--maximum-preparation-process-count", type=int, default=4)
    parser.add_argument("--maximum-parallel-batches", type=int, default=4)
    parser.add_argument("--maximum-wall-time-s", type=float, default=36000.0)
    parser.add_argument("--rollout-steps", type=int, default=15000)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument(
        "--preparation-only-level",
        choices=(ORDER9_R1_MINIMUM_LEVEL_ID, *LEVEL_IDS),
    )
    parser.add_argument(
        "--preparation-only-split",
        choices=("train", "validation"),
        default="train",
    )
    parser.add_argument("--preparation-only-candidate-id")
    parser.add_argument(
        "--maximum-selection-level",
        choices=LEVEL_IDS,
        default=LEVEL_IDS[0],
        help="Highest range level to prove in this bounded run.",
    )
    return parser


def _binding(path: Path) -> dict[str, str]:
    return runtime_v11._binding(path)


def _validate_binding(value: object, *, label: str) -> Path:
    return runtime_v11._validate_binding(value, label=label)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SchemaValidationError(f"R1 v13 JSON object expected: {path}")
    return value


def _load_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = _read_json(PROTOCOL)
    approval = _read_json(APPROVAL)
    if (
        protocol.get("protocol_version") != "order9_r1_range_selection_protocol_v13"
        or protocol.get("status") != "approved"
        or tuple(protocol.get("selection_level_ids", ())) != LEVEL_IDS
        or protocol.get("inherited_minimum_level_id") != ORDER9_R1_MINIMUM_LEVEL_ID
        or protocol.get("selection_split") != "train"
        or int(protocol.get("selection_bucket_count", -1)) != 22
        or protocol.get("confirmation_split") != "validation"
        or int(protocol.get("confirmation_bucket_count", -1)) != 14
        or float(protocol.get("acceptance_clearance_m", -1.0)) != 0.0195
        or float(protocol.get("planning_clearance_m", -1.0)) != 0.0300
        or float(protocol.get("minimum_planning_buffer_m", -1.0)) != 0.0105
        or protocol.get("all_eight_phases_generated_under_constraint") is not True
        or protocol.get("uncertified_rigid_pose_copy_rejected") is not True
        or protocol.get("exact_margin_tolerance_compensated") is not True
        or protocol.get("native_exact_aabb_uses_requested_clearance") is not True
        or protocol.get("reuse_pre_clearance_v11_teacher_rejection") is not False
        or protocol.get("parent_teacher_rejection_reuse_forbidden") is not True
        or protocol.get("selection_requires_current_runner_result") is not True
        or protocol.get("automatic_rotation_contact_retry") is not True
        or protocol.get("automatic_lightweight_contact_retry") is not True
        or protocol.get("automatic_source_configuration_seed_reuse") is not True
        or protocol.get("stale_isaac_evidence_quarantine_required") is not True
        or protocol.get("human_posture_rejections_enforced") is not True
        or int(protocol.get("linear_algebra_thread_count_per_worker", -1)) != 1
        or protocol.get("preparation_worker_start_method") != "spawn"
        or int(protocol.get("torchinductor_compile_threads_per_isaac_process", -1))
        != 8
        or protocol.get("immutable_v12_preparation_reuse_allowed") is not True
        or protocol.get("stop_on_first_failed_level") is not True
        or protocol.get("confirmation_failure_stepback_forbidden") is not True
        or protocol.get("pi_l_actor_command_applied") is not False
        or protocol.get("qpid_qp_enabled_when_isaac_invoked") is not True
        or protocol.get("local_servo_enabled_when_isaac_invoked") is not True
        or protocol.get("formal_teacher_collection_authorized") is not False
    ):
        raise SchemaValidationError("R1 v13 formal range protocol differs")
    if (
        approval.get("record_version")
        != "order9_r1_range_selection_approval_record_v13"
        or approval.get("decision") != "approved"
        or approval.get("approved_by") != "repository_user"
        or approval.get("approved_protocol", {}).get("path")
        != PROTOCOL.relative_to(REPOSITORY).as_posix()
        or approval.get("approved_protocol", {}).get("sha256") != hash_file(PROTOCOL)
        or approval.get("learning_authorized") is not False
        or approval.get("teacher_collection_authorized") is not False
    ):
        raise SchemaValidationError("R1 v13 formal approval differs")
    for label in (
        "base_numeric_protocol",
        "parent_minimum_level_result",
        "parent_v11_rejection",
        "parent_v12_result",
        "source_bucket_manifest",
        "geometry_config",
        "geometry_implementation",
        "complete_task_geometry_audit",
        "support_clearance_config",
        "support_clearance_teacher",
        "teacher_pipeline",
        "early_tilt_pipeline",
        "teacher_hints",
        "teacher_repair_config",
        "human_posture_rejections",
        "materializer",
        "native_collision_base_implementation",
        "native_v13_build_script",
        "isaac_python_native_extension",
        "native_v13_loader",
        "python_collision_wrapper",
        "runner_implementation",
        "base_runner_implementation",
        "streaming_runner_implementation",
        "batched_wrapper_implementation",
        "batched_runtime_implementation",
        "protected_c3_checkpoint",
        "protected_c3_rollout",
        "protected_c3_curriculum",
        "protected_c3_release_ledger",
    ):
        _validate_binding(protocol.get(label), label=label)
    return protocol, approval


def _validate_prepared_case(case, destination: Path) -> dict[str, Any]:
    record = runtime_v10._validate_prepared_case(case, destination)
    clearance = load_order9_r1_support_clearance_teacher_v12_contract(
        ORDER9_R1_SUPPORT_CLEARANCE_TEACHER_V12_RELATIVE,
        repository_root=REPOSITORY,
    )
    audit_path = destination / ORDER9_R1_PLANNING_CLEARANCE_AUDIT_V14_FILENAME
    certificate_path = (
        destination / ORDER9_R1_SUPPORT_CLEARANCE_CERTIFICATE_V14_FILENAME
    )
    audit = _read_json(audit_path)
    certificate = _read_json(certificate_path)
    require_order9_r1_complete_task_geometry_v13(audit)
    require_order9_r1_support_clearance_generation_certificate(
        certificate,
        clearance_contract=clearance,
        candidate_id=case.candidate_id,
    )
    materialized = load_order9_r1_isaac_case(
        destination / "case_manifest.json", REPOSITORY
    )
    nominal_path = REPOSITORY / materialized.manifest.nominal_artifact.path
    if (
        audit.get("candidate_id") != case.candidate_id
        or float(audit.get("planning_clearance_m", -1.0)) != 0.0300
        or audit.get("generation_time_clearance_constraint_applied") is not True
        or audit.get("native_exact_aabb_uses_requested_clearance") is not True
        or certificate.get("persisted_artifact_checked") is not True
        or (
            audit.get("reused_immutable_v12_materialization") is True
            and (
                audit.get("source_nominal_artifact_path")
                != materialized.manifest.nominal_artifact.path
                or audit.get("source_nominal_artifact_sha256")
                != hash_file(nominal_path)
                or certificate.get("source_nominal_artifact_sha256")
                != hash_file(nominal_path)
            )
        )
    ):
        raise SchemaValidationError("R1 v13 persisted planning proof differs")
    evidence = dict(record["preparation_evidence"])
    evidence.update(
        {
            "complete_task_planning_clearance_audit_v14": _binding(audit_path),
            "support_clearance_generation_certificate_v14": _binding(certificate_path),
        }
    )
    retry_path = destination / "automatic_rotation_contact_retry_v1.json"
    if retry_path.is_file():
        retry = _read_json(retry_path)
        if (
            retry.get("retry_version")
            != "order9_r1_automatic_rotation_contact_retry_v1"
            or retry.get("trigger") != "E_R1_GRASP_ROTATION_FORCE_MOMENT_ARM"
            or int(retry.get("bounded_local_contact_maximum_iterations", -1)) != 160
            or retry.get("isaac_invoked") is not False
            or retry.get("controller_layers_invoked") is not False
            or not retry.get("selected_candidate_group_id")
        ):
            raise SchemaValidationError("R1 rotation contact retry differs")
        evidence["automatic_rotation_contact_retry_v1"] = _binding(retry_path)
    lightweight_retry_path = (
        destination / "automatic_lightweight_contact_retry_v1.json"
    )
    if lightweight_retry_path.is_file():
        retry = _read_json(lightweight_retry_path)
        if (
            retry.get("retry_version")
            != "order9_r1_automatic_lightweight_contact_retry_v1"
            or retry.get("trigger")
            not in {
                "sampled local contact search is disabled in lightweight admission",
                "sampled overhead search is disabled in lightweight admission",
            }
            or int(retry.get("bounded_local_contact_maximum_iterations", -1)) != 160
            or retry.get("teacher_trajectory_complete") is not True
            or retry.get("isaac_invoked") is not False
            or retry.get("controller_layers_invoked") is not False
            or retry.get("acceptance_or_safety_gate_changed") is not False
        ):
            raise SchemaValidationError("R1 lightweight contact retry differs")
        evidence["automatic_lightweight_contact_retry_v1"] = _binding(
            lightweight_retry_path
        )
    seed_path = destination / "configuration_goal_seed_evidence_v1.json"
    expected_seeds = _load_configuration_goal_seed_evidence(
        REPOSITORY / ORDER9_R1_RANGE_TEACHER_REPAIRS_V13_RELATIVE,
        repository=REPOSITORY,
    )
    expected_seed = expected_seeds.get(case.candidate_id)
    if expected_seed is not None:
        if not seed_path.is_file() or _read_json(seed_path) != expected_seed:
            raise SchemaValidationError("R1 configuration-goal seed evidence differs")
        evidence["configuration_goal_seed_evidence_v1"] = _binding(seed_path)
    elif seed_path.exists():
        seed_evidence = _read_json(seed_path)
        source_seed = expected_seeds.get(
            ORDER9_R1_TRAIN027_CONFIGURATION_SEED_SOURCE_CANDIDATE_ID
        )
        if (
            source_seed is None
            or case.source_bucket.bucket_id != ORDER9_R1_TRAIN027_SOURCE_BUCKET_ID
            or seed_evidence.get("record_version")
            != "order9_r1_configuration_goal_seed_evidence_v1"
            or seed_evidence.get("candidate_id") != case.candidate_id
            or seed_evidence.get("scope")
            != "configuration_space_goal_initialization_only"
            or seed_evidence.get("source_trajectory")
            != source_seed["source_trajectory"]
            or seed_evidence.get("joint_seed_hash")
            != source_seed["joint_seed_hash"]
            or seed_evidence.get("contact_goal_seed_applied") is not False
            or seed_evidence.get("contact_ik_constraints_changed") is not False
            or seed_evidence.get("acceptance_or_safety_gate_changed") is not False
            or seed_evidence.get("isaac_invoked_during_seed_selection") is not False
            or seed_evidence.get("controller_layers_invoked_during_seed_selection")
            is not False
            or seed_evidence.get("automatic_source_configuration_seed_reuse")
            is not True
            or seed_evidence.get("source_seed_candidate_id")
            != ORDER9_R1_TRAIN027_CONFIGURATION_SEED_SOURCE_CANDIDATE_ID
        ):
            raise SchemaValidationError(
                "unexpected R1 configuration-goal seed evidence"
            )
        evidence["configuration_goal_seed_evidence_v1"] = _binding(seed_path)
    return {
        **record,
        "fast_screen_passed": True,
        "pre_isaac_geometry_passed": True,
        "preparation_evidence": evidence,
    }


def _write_rejection(case, output_root: Path, pipeline) -> dict[str, str]:
    audits = pipeline.clearance_candidate_audits.get(case.candidate_id)
    if not audits:
        return {}
    path = runtime_v10._atomic_json(
        output_root
        / "rejections"
        / case.level_id
        / f"{case.candidate_id}__support_clearance_v13.json",
        {
            "record_version": "order9_r1_support_clearance_rejection_v13",
            "candidate_id": case.candidate_id,
            "accepted": False,
            "isaac_invoked": False,
            "controller_layers_invoked": False,
            "candidate_audits": list(audits),
        },
    )
    return {"support_clearance_rejection_v13": _binding(path)}


def _quarantine_preparation_destination(destination: Path, *, reason: str) -> None:
    """Preserve a superseded or partial preparation instead of deleting it."""

    if not destination.exists():
        return
    output_root = destination.parents[2]
    if destination.parents[1].name != "prepared":
        raise SchemaValidationError("R1 preparation quarantine root differs")
    root = (
        output_root
        / "diagnostics"
        / "preparation_quarantine"
        / destination.name
    )
    root.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while True:
        target = root / f"{reason}__attempt_{attempt:03d}"
        if not target.exists():
            destination.replace(target)
            return
        attempt += 1


def _try_reuse_v12(case, destination: Path) -> dict[str, Any] | None:
    source_root = V12_OUTPUT_ROOT / "prepared" / case.level_id / case.candidate_id
    if not source_root.is_dir():
        return None
    try:
        runtime_v12._validate_prepared_case(case, source_root)
        source = load_order9_r1_isaac_case(
            source_root / "case_manifest.json", REPOSITORY
        )
        geometry = load_order9_r1_nominal_geometry_v11_contract(
            ORDER9_R1_NOMINAL_GEOMETRY_V11_RELATIVE,
            repository_root=REPOSITORY,
        )
        clearance = load_order9_r1_support_clearance_teacher_v12_contract(
            ORDER9_R1_SUPPORT_CLEARANCE_TEACHER_V12_RELATIVE,
            repository_root=REPOSITORY,
        )
        promote_order9_r1_v12_materialized_case_v14(
            source,
            destination=destination,
            task_spec=case.task_spec,
            repository_root=REPOSITORY,
            geometry_contract=geometry,
            clearance_contract=clearance,
        )
        result = _validate_prepared_case(case, destination)
        result["reused_immutable_v12_materialization"] = True
        return result
    except (OSError, RuntimeError, ValueError, SchemaValidationError):
        _quarantine_preparation_destination(
            destination,
            reason="failed_v12_promotion",
        )
        return None


def _prepare_case(arguments) -> dict[str, Any]:
    case, output_root_text = arguments
    output_root = Path(output_root_text)
    destination = output_root / "prepared" / case.level_id / case.candidate_id
    if destination.is_dir():
        try:
            return _validate_prepared_case(case, destination)
        except (OSError, RuntimeError, ValueError, SchemaValidationError):
            _quarantine_preparation_destination(
                destination,
                reason="superseded_invalid_preparation",
            )
    reused = _try_reuse_v12(case, destination)
    if reused is not None:
        return reused

    base = load_order9_r1_calibration_protocol(
        BASE_PROTOCOL,
        repository_root=REPOSITORY,
    )
    geometry = load_order9_r1_nominal_geometry_v11_contract(
        ORDER9_R1_NOMINAL_GEOMETRY_V11_RELATIVE,
        repository_root=REPOSITORY,
    )
    clearance = load_order9_r1_support_clearance_teacher_v12_contract(
        ORDER9_R1_SUPPORT_CLEARANCE_TEACHER_V12_RELATIVE,
        repository_root=REPOSITORY,
    )
    global _PREPARATION_PIPELINE, _PREPARATION_PIPELINE_KEY
    pipeline_key = (
        base.source_bucket_manifest.sha256,
        base.minimum_normalized_joint_limit_reserve,
        base.maximum_body_tilt_rad,
        geometry.config_sha256,
        clearance.config_sha256,
    )
    if _PREPARATION_PIPELINE_KEY != pipeline_key:
        _PREPARATION_PIPELINE = Order9R1ExactMarginTeacherScreenPipelineV13(
            repository_root=REPOSITORY,
            source_bucket_manifest_path=REPOSITORY / base.source_bucket_manifest.path,
            minimum_normalized_joint_limit_reserve=(
                base.minimum_normalized_joint_limit_reserve
            ),
            maximum_body_tilt_rad=base.maximum_body_tilt_rad,
            anchor_position_tolerance_m=0.030,
            enforce_joint_limit_reserve_during_ik=True,
            geometry_contract=geometry,
            clearance_contract=clearance,
            physical_model_config_path=REPOSITORY / "configs/robot/robot_model.yaml",
        )
        _PREPARATION_PIPELINE_KEY = pipeline_key

    record = {
        "candidate_id": case.candidate_id,
        "source_bucket_id": case.source_bucket.bucket_id,
        "module_count": case.source_bucket.module_count,
        "sample_kind": case.sample_kind,
        "sample_index": case.sample_index,
        "teacher_feasible": False,
        "fast_screen_passed": False,
        "screen_passed": False,
        "pre_isaac_geometry_passed": False,
        "prepared": False,
        "preparation_failure_reason": None,
        "preparation_evidence": {},
    }
    try:
        prepared = _PREPARATION_PIPELINE.prepare(case)
        prepared.validate_for(case)
        if not prepared.teacher_trajectory_complete:
            record["preparation_failure_reason"] = prepared.failure_reason
            record["preparation_evidence"] = _write_rejection(
                case, output_root, _PREPARATION_PIPELINE
            )
            return record
        record["teacher_feasible"] = True
        screen = _PREPARATION_PIPELINE.screen(case, prepared)
        record["screen_passed"] = bool(screen.accepted)
        if not screen.accepted or not screen.eligible_for_full_control_test:
            record["preparation_failure_reason"] = ",".join(screen.violation_codes)
            return record
        record["fast_screen_passed"] = True
        materialized = materialize_order9_r1_isaac_case(
            case=case,
            prepared=prepared,
            screen=screen,
            output_dir=destination,
            repository_root=REPOSITORY,
            approved_protocol_path=BASE_PROTOCOL,
        )
        certificate = _PREPARATION_PIPELINE.clearance_generation_certificates.get(
            case.candidate_id
        )
        finalize_order9_r1_v14_materialized_case(
            materialized,
            task_spec=case.task_spec,
            phase_time_scales=runtime_v11.order9_r1_safe_phase_time_scales(
                case.source_bucket.module_count
            ),
            joint_rate_limit_rad_s=(
                float(screen.maximum_joint_rate_rad_s)
                + float(screen.minimum_joint_rate_margin_rad_s)
            ),
            repository_root=REPOSITORY,
            geometry_contract=geometry,
            clearance_contract=clearance,
            clearance_generation_certificate=certificate,
        )
        seed_evidence = _PREPARATION_PIPELINE.configuration_goal_seed_evidence.get(
            case.candidate_id
        )
        if seed_evidence is not None:
            runtime_v10._atomic_json(
                destination / "configuration_goal_seed_evidence_v1.json",
                seed_evidence,
            )
        retry = _PREPARATION_PIPELINE.automatic_rotation_contact_retries.get(
            case.candidate_id
        )
        if retry is not None:
            runtime_v10._atomic_json(
                destination / "automatic_rotation_contact_retry_v1.json",
                retry,
            )
        lightweight_retry = (
            _PREPARATION_PIPELINE.automatic_lightweight_contact_retries.get(
                case.candidate_id
            )
        )
        if lightweight_retry is not None:
            runtime_v10._atomic_json(
                destination / "automatic_lightweight_contact_retry_v1.json",
                lightweight_retry,
            )
        return _validate_prepared_case(case, destination)
    except (OSError, RuntimeError, ValueError, SchemaValidationError) as error:
        evidence = _write_rejection(case, output_root, _PREPARATION_PIPELINE)
        _quarantine_preparation_destination(
            destination,
            reason="failed_generation",
        )
        record["preparation_failure_reason"] = f"{type(error).__name__}: {error}"
        record["preparation_evidence"] = evidence
        return record


def _install_v13_preparation_hooks() -> None:
    runtime_v11._prepare_case = _prepare_case
    runtime_v11._validate_prepared_case = _validate_prepared_case
    runtime_v11._prepare_level = _prepare_level
    runtime_v11.RUNNER_VERSION = RUNNER_VERSION
    runtime_v10.RUNNER_VERSION = RUNNER_VERSION
    runtime_v10._run_processes = _run_processes_with_isaac_abi


def _run_processes_with_isaac_abi(*args, **kwargs):
    """Select the Python 3.12 native module only for Isaac child processes."""

    protocol = _read_json(PROTOCOL)
    extension = _validate_binding(
        protocol.get("isaac_python_native_extension"),
        label="isaac_python_native_extension",
    )
    environment = {
        "AMSRR_ORDER9_POSTURE_NATIVE_PATH": str(extension),
        # Four Isaac processes are the formal maximum.  PyTorch otherwise
        # creates 32 compile workers per process, oversubscribing this
        # 32-logical-CPU host by 4x during every new morphology compile.
        "TORCHINDUCTOR_COMPILE_THREADS": "8",
    }
    previous = {name: os.environ.get(name) for name in environment}
    os.environ.update(environment)
    try:
        return _V10_RUN_PROCESSES(*args, **kwargs)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _prepare_level(
    level_id: str,
    split: str,
    *,
    output_root: Path,
    maximum_process_count: int,
):
    """Prepare at most one work item per worker and supersede bound repairs."""

    cases = list(runtime_v11._corrected_cases(level_id, split))
    expected = (
        runtime_v11.EXPECTED_BUCKET_COUNTS[split] * runtime_v11.SAMPLES_PER_BUCKET
    )
    if len(cases) != expected:
        raise SchemaValidationError("R1 v13 candidate count differs")
    cases.sort(
        key=lambda case: range_case_priority(
            case.candidate_id,
            case.sample_kind,
            case.sample_index,
        )
    )
    repair_ids = _load_bounded_local_contact_repairs(
        REPOSITORY / ORDER9_R1_RANGE_TEACHER_REPAIRS_V13_RELATIVE,
        repository=REPOSITORY,
    )
    records: dict[str, dict[str, Any]] = {}
    decisive = False
    decisive_path = (
        output_root / "preparation" / f"{split}__{level_id}__decisive_in_progress.json"
    )
    if decisive_path.is_file():
        cached = _read_json(decisive_path)
        record = cached.get("decisive_record")
        candidate_id = record.get("candidate_id") if isinstance(record, dict) else None
        case_by_id = {case.candidate_id: case for case in cases}
        case = case_by_id.get(candidate_id)
        if candidate_id in repair_ids:
            print(
                "ORDER9_R1_V13_DECISIVE_PREPARATION_SUPERSEDED="
                + json.dumps(
                    {
                        "level_id": level_id,
                        "split": split,
                        "candidate_id": candidate_id,
                        "teacher_repair_config": str(
                            REPOSITORY / ORDER9_R1_RANGE_TEACHER_REPAIRS_V13_RELATIVE
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            if (
                cached.get("record_version")
                not in {
                    "order9_r1_range_decisive_pre_isaac_v11",
                    "order9_r1_range_decisive_pre_isaac_v13",
                }
                or cached.get("level_id") != level_id
                or cached.get("split") != split
                or cached.get("isaac_invoked") is not False
                or cached.get("controller_layers_invoked") is not False
                or case is None
                or case.sample_kind != "lattice"
                or record.get("prepared") is not False
                or not record.get("preparation_failure_reason")
            ):
                raise SchemaValidationError(
                    "R1 v13 decisive pre-Isaac checkpoint differs"
                )
            decisive_record = dict(record)
            evidence = dict(decisive_record.get("preparation_evidence", {}))
            evidence["decisive_pre_isaac_checkpoint"] = _binding(decisive_path)
            decisive_record["preparation_evidence"] = evidence
            records[case.candidate_id] = decisive_record
            decisive = True
    if not decisive:
        arguments = [(case, str(output_root)) for case in cases]
        maximum_pending = maximum_process_count
        next_index = 0
        # NumPy/SciPy, PyTorch, and the native collision solver have initialized
        # thread pools before this point.  Forking that state can strand workers
        # in inherited futex waits (observed as 20 of 24 workers sleeping while
        # only four made progress).  Spawn clean interpreters so the requested
        # process parallelism is real and deterministic.
        # A child must prepare at most one trajectory.  Reusing a child for a
        # second native planning call left it blocked in a library futex even
        # with the spawn start method.  Recreate the bounded worker set after
        # every wave so no initialized solver state crosses work items.
        while next_index < len(arguments) and not decisive:
            wave = arguments[next_index : next_index + maximum_pending]
            next_index += len(wave)
            with ProcessPoolExecutor(
                max_workers=min(maximum_process_count, len(wave)),
                mp_context=multiprocessing.get_context("spawn"),
            ) as executor:
                futures = {
                    executor.submit(_prepare_case, value): value[0] for value in wave
                }
                while futures:
                    completed, _pending = wait(futures, return_when=FIRST_COMPLETED)
                    for future in completed:
                        case = futures.pop(future)
                        record = future.result()
                        records[case.candidate_id] = record
                        if case.sample_kind == "lattice" and not record["prepared"]:
                            decisive = True
                            runtime_v10._atomic_json(
                                decisive_path,
                                {
                                    "record_version": (
                                        "order9_r1_range_decisive_pre_isaac_v13"
                                    ),
                                    "level_id": level_id,
                                    "split": split,
                                    "completed_count_at_detection": len(records),
                                    "isaac_invoked": False,
                                    "controller_layers_invoked": False,
                                    "decisive_record": record,
                                },
                            )
                    if decisive:
                        for future in futures:
                            future.cancel()
                        break
            print(
                "ORDER9_R1_V13_PREPARATION_PROGRESS="
                + json.dumps(
                    {
                        "level_id": level_id,
                        "split": split,
                        "completed": len(records),
                        "total": len(cases),
                        "admitted": sum(
                            bool(value["prepared"]) for value in records.values()
                        ),
                        "decisive_pre_isaac_rejection": decisive,
                        "maximum_pending": maximum_pending,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    for case in cases:
        records.setdefault(
            case.candidate_id,
            runtime_v11._unattempted_record(
                case,
                "not attempted after decisive pre-Isaac rejection",
            ),
        )
    ordered = [records[case.candidate_id] for case in cases]
    path = runtime_v10._atomic_json(
        output_root / "preparation" / f"{split}__{level_id}.json",
        {
            "record_version": "order9_r1_range_preparation_v13",
            "level_id": level_id,
            "split": split,
            "candidate_count": len(ordered),
            "attempted_count": sum(
                value["preparation_failure_reason"]
                != "not attempted after decisive pre-Isaac rejection"
                for value in ordered
            ),
            "prepared_count": sum(bool(value["prepared"]) for value in ordered),
            "decisive_pre_isaac_rejection": decisive,
            "isaac_invoked": False,
            "controller_layers_invoked": False,
            "entries": ordered,
        },
    )
    print(f"ORDER9_R1_V13_PREPARATION_RESULT={path}", flush=True)
    return cases, ordered


def _run_level(*, level_id, split, output_root, base, protocol, args, deadline):
    cases, preparation = runtime_v11._prepare_level(
        level_id,
        split,
        output_root=output_root,
        maximum_process_count=args.maximum_preparation_process_count,
    )
    if runtime_v11._pre_isaac_gate_failed(preparation):
        entries = runtime_v11._entries_without_isaac(preparation)
        isaac_invoked = False
    else:
        isaac_native = _validate_binding(
            protocol["isaac_python_native_extension"],
            label="isaac_python_native_extension",
        )
        native_environment_name = "AMSRR_ORDER9_POSTURE_NATIVE_PATH"
        previous_native = os.environ.get(native_environment_name)
        os.environ[native_environment_name] = str(isaac_native)
        try:
            entries = runtime_v10._execute_level(
                level_id,
                cases=cases,
                preparation_records=preparation,
                output_root=output_root,
                isaac_python=args.isaac_python,
                rollout_steps=args.rollout_steps,
                maximum_parallel_batches=args.maximum_parallel_batches,
                deadline=deadline,
            )
        finally:
            if previous_native is None:
                os.environ.pop(native_environment_name, None)
            else:
                os.environ[native_environment_name] = previous_native
        isaac_invoked = True
    gate = runtime_v11._gate(base, level_id, entries)
    path = runtime_v10._atomic_json(
        output_root / "results" / f"{split}__{level_id}.json",
        {
            "result_version": "order9_r1_range_level_result_v13",
            "runner_version": RUNNER_VERSION,
            "level_id": level_id,
            "split": split,
            "gate": gate.to_dict(),
            "candidate_count": len(entries),
            "isaac_invoked": isaac_invoked,
            "pi_l_actor_command_applied": False,
            "qpid_qp_applied": isaac_invoked,
            "local_servo_applied": isaac_invoked,
            "training_eligible": False,
            "entries": entries,
        },
    )
    return {
        "level_id": level_id,
        "split": split,
        "passed": gate.passed,
        "gate": gate.to_dict(),
        "result": _binding(path),
        "isaac_invoked": isaac_invoked,
        "controller_layers_invoked": isaac_invoked,
    }, path


def main() -> int:
    runtime_v10._ensure_python_hash_seed_zero()
    args = _parser().parse_args()
    if (
        not 1 <= args.maximum_preparation_process_count <= 24
        or not 1 <= args.maximum_parallel_batches <= 4
        or args.maximum_wall_time_s <= 0.0
        or args.rollout_steps < 1
    ):
        raise ValueError("R1 v13 runtime limits are invalid")
    protocol, approval = _load_contract()
    _install_v13_preparation_hooks()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    base = load_order9_r1_calibration_protocol(
        BASE_PROTOCOL,
        repository_root=REPOSITORY,
    )
    started = time.time()
    deadline = started + min(
        float(args.maximum_wall_time_s),
        float(protocol["maximum_wall_time_s"]),
    )
    if args.preparation_only_candidate_id:
        if not args.preparation_only_level:
            raise ValueError(
                "--preparation-only-candidate-id requires --preparation-only-level"
            )
        matches = [
            case
            for case in runtime_v11._corrected_cases(
                args.preparation_only_level,
                args.preparation_only_split,
            )
            if case.candidate_id == args.preparation_only_candidate_id
        ]
        if len(matches) != 1:
            raise ValueError("R1 preparation-only candidate does not resolve once")
        record = _prepare_case((matches[0], str(output_root)))
        print(
            "ORDER9_R1_V13_SINGLE_PREPARATION=" + json.dumps(record, sort_keys=True),
            flush=True,
        )
        return 0 if record["prepared"] else 1
    if args.preparation_only_level:
        runtime_v11._prepare_level(
            args.preparation_only_level,
            args.preparation_only_split,
            output_root=output_root,
            maximum_process_count=args.maximum_preparation_process_count,
        )
        return 0

    maximum_level_index = LEVEL_IDS.index(args.maximum_selection_level)
    levels_to_run = LEVEL_IDS[: maximum_level_index + 1]
    selection_results = []
    selected_level = ORDER9_R1_MINIMUM_LEVEL_ID
    confirmation = None
    try:
        for level_id in levels_to_run:
            selection, _path = _run_level(
                level_id=level_id,
                split="train",
                output_root=output_root,
                base=base,
                protocol=protocol,
                args=args,
                deadline=deadline,
            )
            selection_results.append(selection)
            if not selection["passed"]:
                break
            selected_level = level_id
        if selected_level == args.maximum_selection_level:
            confirmation, _path = _run_level(
                level_id=selected_level,
                split="validation",
                output_root=output_root,
                base=base,
                protocol=protocol,
                args=args,
                deadline=deadline,
            )
            status = "accepted" if confirmation["passed"] else "rejected"
        else:
            status = "rejected"
    except TimeoutError as error:
        runtime_v10._atomic_json(
            output_root / "incomplete_timeout.json",
            {
                "runner_version": RUNNER_VERSION,
                "status": "incomplete_timeout",
                "selected_level_before_timeout": selected_level,
                "selection_results": selection_results,
                "confirmation_result": confirmation,
                "elapsed_s": time.time() - started,
                "error": str(error),
            },
        )
        return 2

    isaac_invoked = bool(confirmation and confirmation["isaac_invoked"])
    ledger = {
        "ledger_version": "order9_r1_range_selection_result_ledger_v13",
        "runner_version": RUNNER_VERSION,
        "status": status,
        "selected_level_id": selected_level,
        "accepted_level_id": selected_level if status == "accepted" else None,
        "selection_split": "train",
        "selection_bucket_count": 22,
        "confirmation_split": "validation",
        "confirmation_bucket_count": 14,
        "selection_results": selection_results,
        "requested_maximum_selection_level_id": args.maximum_selection_level,
        "selection_results_generated_by_current_runner": True,
        "parent_teacher_rejection_reused": False,
        "confirmation_result": confirmation,
        "acceptance_clearance_m": 0.0195,
        "planning_clearance_m": 0.0300,
        "minimum_planning_buffer_m": 0.0105,
        "generation_certificate_required": True,
        "exact_margin_tolerance_compensated": True,
        "native_exact_aabb_uses_requested_clearance": True,
        "confirmation_failure_stepback_forbidden": True,
        "formal_teacher_collection_authorized": False,
        "training_eligible": False,
        "pi_l_system_retained": True,
        "isaac_invoked": isaac_invoked,
        "pi_l_actor_command_applied": False,
        "qpid_qp_applied": isaac_invoked,
        "local_servo_applied": isaac_invoked,
        "elapsed_s": time.time() - started,
        "bindings": {
            "protocol": _binding(PROTOCOL),
            "approval": _binding(APPROVAL),
            "base_numeric_protocol": _binding(BASE_PROTOCOL),
            "parent_minimum_level_result": protocol["parent_minimum_level_result"],
            "parent_v11_rejection": protocol["parent_v11_rejection"],
            "parent_v12_result": protocol["parent_v12_result"],
            "support_clearance_config": _binding(
                REPOSITORY / ORDER9_R1_SUPPORT_CLEARANCE_TEACHER_V12_RELATIVE
            ),
            "complete_task_geometry_audit": _binding(
                REPOSITORY / "amsrr/training/order9_r1_complete_task_geometry_v13.py"
            ),
            "teacher_pipeline": _binding(
                REPOSITORY / "amsrr/training/order9_r1_range_selection_v13.py"
            ),
            "early_tilt_pipeline": _binding(
                REPOSITORY / "amsrr/training/order9_r1_early_tilt_pipeline.py"
            ),
            "teacher_repair_config": _binding(
                REPOSITORY / ORDER9_R1_RANGE_TEACHER_REPAIRS_V13_RELATIVE
            ),
            "human_posture_rejections": protocol["human_posture_rejections"],
            "materializer": _binding(
                REPOSITORY
                / "amsrr/training/order9_r1_complete_task_materialization_v14.py"
            ),
            "native_v13_build_script": _binding(
                REPOSITORY / "scripts/build_order9_posture_native_v13.sh"
            ),
            "isaac_python_native_extension": protocol["isaac_python_native_extension"],
            "native_v13_loader": _binding(
                REPOSITORY / "amsrr/feasibility/order9_native_loader_v13.py"
            ),
            "runner_implementation": _binding(Path(__file__)),
            "base_runner_implementation": protocol[
                "base_runner_implementation"
            ],
            "streaming_runner_implementation": protocol[
                "streaming_runner_implementation"
            ],
            "protected_c3_checkpoint": protocol["protected_c3_checkpoint"],
            "protected_c3_rollout": protocol["protected_c3_rollout"],
            "protected_c3_curriculum": protocol["protected_c3_curriculum"],
            "protected_c3_release_ledger": protocol["protected_c3_release_ledger"],
        },
        "approval_record_version": approval["record_version"],
    }
    result_path = runtime_v10._atomic_json(
        output_root / RESULT_LEDGER_NAME,
        ledger,
    )
    print(
        "ORDER9_R1_V13_COMPLETE="
        + json.dumps(
            {
                "status": status,
                "selected_level_id": selected_level,
                "result_ledger": str(result_path),
                "result_sha256": hash_file(result_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if status == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
