#!/usr/bin/env python3
from __future__ import annotations

"""Confirm the selected R1 scene range on 14 never-used morphologies."""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import replace
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

for _thread_environment_name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_environment_name] = "1"

import torch

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.feasibility.order9_native_loader_v13 import (  # noqa: E402
    activate_order9_posture_native_v13,
)

activate_order9_posture_native_v13()

from amsrr.schemas.common import SchemaValidationError  # noqa: E402
from amsrr.schemas.datasets import DatasetSplit  # noqa: E402
from amsrr.schemas.order3 import Order3MorphologyPoolManifest  # noqa: E402
from amsrr.schemas.task_spec import TaskSpec  # noqa: E402
from amsrr.simulation.order9_morphology_assets import (  # noqa: E402
    Order9MorphologyAssetManifest,
    validate_order9_morphology_asset_manifest_bytes,
)
from amsrr.training.order9_c3_nominal_trajectory import (  # noqa: E402
    write_order9_c3_nominal_trajectory_artifact,
)
from amsrr.training.order9_r1_calibration import (  # noqa: E402
    load_order9_r1_calibration_protocol,
)
from amsrr.training.order9_r1_calibration_runner import (  # noqa: E402
    Order9R1CalibrationCase,
)
from amsrr.training.order9_r1_complete_task_materialization_v14 import (  # noqa: E402
    ORDER9_R1_PLANNING_CLEARANCE_AUDIT_V14_FILENAME,
    finalize_order9_r1_v14_materialized_case,
)
from amsrr.training.order9_r1_held_out_confirmation_v16 import (  # noqa: E402
    ORDER9_R1_HELD_OUT_BUCKET_COUNT,
    ORDER9_R1_HELD_OUT_CONFIRMATION_V16_VERSION,
    ORDER9_R1_HELD_OUT_INTERIOR_COUNT,
    ORDER9_R1_HELD_OUT_LATTICE_COUNT,
    ORDER9_R1_HELD_OUT_SAMPLE_COUNT,
    Order9R1HeldOutPairV16,
    audit_order9_r1_held_out_composed_scene_case_v16,
    build_order9_r1_held_out_composed_target_v16,
    build_order9_r1_held_out_source_bucket_v16,
    relabel_order9_r1_held_out_task_v16,
    select_order9_r1_held_out_pairs_v16,
)
from amsrr.training.order9_r1_isaac_calibration import (  # noqa: E402
    load_order9_r1_isaac_case,
    materialize_order9_r1_isaac_case,
)
from amsrr.training.order9_r1_nominal_calibration import (  # noqa: E402
    order9_r1_nominal_case_command,
)
from amsrr.training.order9_r1_nominal_geometry_v11 import (  # noqa: E402
    ORDER9_R1_NOMINAL_GEOMETRY_V11_RELATIVE,
    load_order9_r1_nominal_geometry_v11_contract,
)
from amsrr.training.order9_r1_range_selection_v13 import (  # noqa: E402
    Order9R1ExactMarginTeacherScreenPipelineV13,
)
from amsrr.training.order9_r1_safe_timing import (  # noqa: E402
    order9_r1_safe_phase_time_scales,
)
from amsrr.training.order9_r1_support_clearance_teacher_v12 import (  # noqa: E402
    ORDER9_R1_SUPPORT_CLEARANCE_TEACHER_V12_RELATIVE,
    load_order9_r1_support_clearance_teacher_v12_contract,
)
from amsrr.training import (
    order9_r1_yaw_branch_repair as yaw_repair_runtime,
)  # noqa: E402
from amsrr.utils.hashing import hash_file  # noqa: E402
from scripts import order9_run_r1_range_selection_v11 as runtime_v11  # noqa: E402
from scripts import order9_run_r1_range_selection_v13 as runtime_v13  # noqa: E402
from amsrr.training.order9_r1_yaw_branch_repair import (  # noqa: E402
    derive_order9_r1_planar_nominal_transfer_case,
)

RUNNER_VERSION = "order9_r1_held_out_scene_confirmation_runner_v16"
PROTOCOL = REPOSITORY / (
    "configs/training/order9_r1_held_out_scene_confirmation_protocol_v16.json"
)
APPROVAL = REPOSITORY / (
    "for_codex/R1_HELD_OUT_SCENE_CONFIRMATION_PROTOCOL_V16_APPROVAL.json"
)
OUTPUT_ROOT = REPOSITORY / (
    "artifacts/p4_full/order9/r1_teacher/held_out_scene_confirmation_v16"
)
POOL = REPOSITORY / "artifacts/p4_full/order9/morphology_pool.json"
ASSETS = REPOSITORY / (
    "artifacts/p4_full/order9/c3_preparation/"
    "morphology_assets_grasp_frame_mesh_v2/manifest.json"
)
SOURCE_MANIFEST = REPOSITORY / (
    "artifacts/p4_full/order9/releases/c3_pi_l_promoted_update18_v1/runtime/"
    "rollout_buckets_current_lineage_v5/"
    "manifest_c3_release_smooth_clear300_current_config_min2_preload_screened_v2.json"
)
PARENT_SELECTION_LEDGER = REPOSITORY / (
    "for_codex/R1_SCENE_INVARIANT_RANGE_SELECTION_V15_RESULT_LEDGER.json"
)
BASE_PROTOCOL = runtime_v13.BASE_PROTOCOL
WRAPPER = REPOSITORY / "scripts/order9_r1_batched_nominal_compression_rollout.py"
ISAAC_PYTHON = "/home/leus/.local/share/mamba/envs/isaaclab3/bin/python"
REFERENCE_LEVEL = "r1_l2_20mm_10deg"
TARGET_LEVEL = "r1_l4_40mm_20deg"
REFERENCE_SAMPLE_INDEX = 13
ISAAC_SAMPLE_INDICES = (0, 26)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--isaac-python", default=ISAAC_PYTHON)
    parser.add_argument("--preparation-workers", type=int, default=8)
    parser.add_argument("--proof-workers", type=int, default=8)
    parser.add_argument("--maximum-parallel-isaac", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=15000)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--proof-only", action="store_true")
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SchemaValidationError(f"R1 held-out JSON object expected: {path}")
    return value


def _atomic_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_text = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _portable(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY).as_posix()


def _binding(path: Path) -> dict[str, str]:
    return {"path": _portable(path), "sha256": hash_file(path)}


def _validate_binding(value: object, *, label: str) -> Path:
    if not isinstance(value, dict):
        raise SchemaValidationError(f"R1 held-out {label} binding is missing")
    path = (REPOSITORY / str(value.get("path", ""))).resolve()
    if not path.is_file() or hash_file(path) != value.get("sha256"):
        raise SchemaValidationError(f"R1 held-out {label} binding changed")
    return path


def _load_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = _read_json(PROTOCOL)
    approval = _read_json(APPROVAL)
    if (
        protocol.get("protocol_version")
        != "order9_r1_held_out_scene_confirmation_protocol_v16"
        or protocol.get("status") != "approved"
        or protocol.get("source_pool_split") != "held_out"
        or int(protocol.get("held_out_bucket_count", -1))
        != ORDER9_R1_HELD_OUT_BUCKET_COUNT
        or int(protocol.get("held_out_count_per_module_count", -1)) != 2
        or int(protocol.get("candidate_count_per_bucket", -1))
        != ORDER9_R1_HELD_OUT_SAMPLE_COUNT
        or int(protocol.get("candidate_count", -1))
        != ORDER9_R1_HELD_OUT_BUCKET_COUNT * ORDER9_R1_HELD_OUT_SAMPLE_COUNT
        or int(protocol.get("lattice_candidate_count_per_bucket", -1))
        != ORDER9_R1_HELD_OUT_LATTICE_COUNT
        or int(protocol.get("interior_candidate_count_per_bucket", -1))
        != ORDER9_R1_HELD_OUT_INTERIOR_COUNT
        or protocol.get("reference_level_id") != REFERENCE_LEVEL
        or int(protocol.get("reference_lattice_index", -1)) != REFERENCE_SAMPLE_INDEX
        or protocol.get("selected_level_id") != TARGET_LEVEL
        or float(protocol.get("selected_position_half_span_m", -1.0)) != 0.04
        or float(protocol.get("selected_yaw_half_span_rad", -1.0)) != 0.3490658503988659
        or protocol.get("legacy_validation_evidence_reuse_forbidden") is not True
        or protocol.get("complete_scene_transform_required") is not True
        or protocol.get("support_transformed_with_scene") is not True
        or protocol.get("all_eight_phases_required") is not True
        or int(protocol.get("proof_substep_count", -1)) != 2
        or protocol.get("proof_substeps_are_not_temporal_phases") is not True
        or tuple(protocol.get("representative_lattice_indices", ()))
        != ISAAC_SAMPLE_INDICES
        or int(protocol.get("representative_isaac_episode_count", -1)) != 28
        or protocol.get("pi_l_actor_command_applied") is not False
        or protocol.get("qpid_qp_enabled_for_isaac") is not True
        or protocol.get("local_servo_enabled_for_isaac") is not True
        or protocol.get("formal_teacher_collection_authorized") is not False
        or protocol.get("training_eligible") is not False
    ):
        raise SchemaValidationError("R1 held-out protocol differs")
    if (
        approval.get("record_version")
        != "order9_r1_held_out_scene_confirmation_approval_v16"
        or approval.get("decision") != "approved"
        or approval.get("approved_by") != "repository_user"
        or approval.get("approved_protocol", {}).get("path") != _portable(PROTOCOL)
        or approval.get("approved_protocol", {}).get("sha256") != hash_file(PROTOCOL)
        or approval.get("learning_authorized") is not False
        or approval.get("teacher_collection_authorized") is not False
    ):
        raise SchemaValidationError("R1 held-out approval differs")
    for label in (
        "parent_selection_result",
        "morphology_pool",
        "morphology_asset_manifest",
        "source_task_manifest",
        "confirmation_implementation",
        "runner_implementation",
        "scene_transfer_implementation",
        "isaac_wrapper_implementation",
        "protected_c3_checkpoint",
        "protected_c3_rollout",
        "protected_c3_curriculum",
        "protected_c3_release_ledger",
    ):
        _validate_binding(protocol.get(label), label=label)
    return protocol, approval


def _source_cases(level_id: str) -> tuple[Any, ...]:
    cases = tuple(runtime_v11._corrected_cases(level_id, "train"))
    if len(cases) != 682:
        raise SchemaValidationError("R1 held-out paired task population differs")
    return cases


def _pairs_and_cases() -> tuple[
    tuple[Order9R1HeldOutPairV16, ...],
    dict[str, Any],
    dict[str, list[Any]],
]:
    pool = Order3MorphologyPoolManifest.from_json(POOL.read_text(encoding="utf-8"))
    assets = Order9MorphologyAssetManifest.from_json(ASSETS.read_text(encoding="utf-8"))
    validate_order9_morphology_asset_manifest_bytes(
        assets,
        repository_root=REPOSITORY,
        expected_pool_sha256=hash_file(POOL),
    )
    reference_cases = _source_cases(REFERENCE_LEVEL)
    target_cases = _source_cases(TARGET_LEVEL)
    train_buckets = {
        case.source_bucket.bucket_id: case.source_bucket for case in reference_cases
    }
    pairs = select_order9_r1_held_out_pairs_v16(
        pool=pool,
        assets=assets,
        train_buckets=tuple(train_buckets.values()),
    )
    references = {}
    targets = {}
    for pair in pairs:
        bucket_id = pair.paired_train_bucket.bucket_id
        matches = [
            case
            for case in reference_cases
            if case.source_bucket.bucket_id == bucket_id
            and case.sample_kind == "lattice"
            and case.sample_index == REFERENCE_SAMPLE_INDEX
        ]
        level_targets = [
            case for case in target_cases if case.source_bucket.bucket_id == bucket_id
        ]
        if len(matches) != 1 or len(level_targets) != ORDER9_R1_HELD_OUT_SAMPLE_COUNT:
            raise SchemaValidationError("R1 held-out paired case count differs")
        references[pair.bucket_id] = matches[0]
        targets[pair.bucket_id] = sorted(
            level_targets,
            key=lambda case: (case.sample_kind, case.sample_index),
        )
    return pairs, references, targets


def _audit_unusedness(
    *, output_root: Path, pairs: tuple[Order9R1HeldOutPairV16, ...]
) -> Path:
    path = output_root / "unusedness_audit.json"
    if path.is_file():
        payload = _read_json(path)
        if (
            payload.get("status") != "accepted"
            or int(payload.get("held_out_bucket_count", -1)) != 14
            or payload.get("prior_r1_path_match_count") != 0
        ):
            raise SchemaValidationError("R1 held-out cached unusedness audit differs")
        return path
    prefixes = {pair.held_out_entry.structural_hash[:12] for pair in pairs}
    hits = []
    r1_root = REPOSITORY / "artifacts/p4_full/order9/r1_teacher"
    for root_text, directory_names, file_names in os.walk(r1_root):
        root = Path(root_text).resolve()
        if root == output_root or output_root in root.parents:
            directory_names[:] = []
            continue
        for name in (*directory_names, *file_names):
            matches = sorted(prefix for prefix in prefixes if prefix in name)
            if matches:
                hits.append(
                    {
                        "path": _portable(root / name),
                        "structural_hash_prefixes": matches,
                    }
                )
    if hits:
        raise SchemaValidationError("R1 held-out morphology appeared in prior R1 paths")
    return _atomic_json(
        path,
        {
            "audit_version": "order9_r1_held_out_unusedness_audit_v16",
            "status": "accepted",
            "morphology_pool": _binding(POOL),
            "source_pool_split": "held_out",
            "held_out_bucket_count": len(pairs),
            "module_count_histogram": {
                str(module_count): sum(
                    pair.held_out_entry.module_count == module_count for pair in pairs
                )
                for module_count in range(2, 9)
            },
            "structural_hashes": [
                pair.held_out_entry.structural_hash for pair in pairs
            ],
            "prior_r1_path_match_count": 0,
            "legacy_validation_evidence_reused": False,
        },
    )


def _reference_candidate_id(pair: Order9R1HeldOutPairV16) -> str:
    return f"{REFERENCE_LEVEL}__validation__{pair.bucket_id}__lattice_13"


def _target_candidate_id(pair: Order9R1HeldOutPairV16, source_case: Any) -> str:
    return (
        f"{TARGET_LEVEL}__validation__{pair.bucket_id}__"
        f"{source_case.sample_kind}_{source_case.sample_index:02d}"
    )


def _relabel_task(
    task: TaskSpec,
    *,
    pair: Order9R1HeldOutPairV16,
    candidate_id: str,
) -> TaskSpec:
    return relabel_order9_r1_held_out_task_v16(
        task,
        candidate_id=candidate_id,
        bucket_id=pair.bucket_id,
        structural_hash=pair.held_out_entry.structural_hash,
        paired_train_bucket_id=pair.paired_train_bucket.bucket_id,
    )


def _reference_case(
    pair: Order9R1HeldOutPairV16,
    source_case: Any,
) -> Order9R1CalibrationCase:
    source_bucket = build_order9_r1_held_out_source_bucket_v16(
        pair,
        repository_root=REPOSITORY,
        source_manifest_path=SOURCE_MANIFEST,
    )
    candidate_id = _reference_candidate_id(pair)
    task = _relabel_task(source_case.task_spec, pair=pair, candidate_id=candidate_id)
    return replace(
        source_case,
        candidate_id=candidate_id,
        global_case_index=pair.held_out_index,
        split="validation",
        source_bucket=source_bucket,
        seed=int(pair.held_out_entry.requested_seed),
        task_spec=task,
    )


def _write_reference_source_nominal(
    *,
    case: Order9R1CalibrationCase,
    nominal: Any,
    pair: Order9R1HeldOutPairV16,
    output_root: Path,
) -> Path:
    root = output_root / "reference_sources" / pair.bucket_id
    task_path = root / "task_spec.json"
    if root.exists():
        manifest = root / "artifact/manifest.json"
        if (
            manifest.is_file()
            and TaskSpec.from_json(task_path.read_text(encoding="utf-8")).stable_hash()
            == case.task_spec.stable_hash()
        ):
            return manifest
        raise FileExistsError(root)
    root.mkdir(parents=True)
    _atomic_json(task_path, case.task_spec.to_dict())
    urdf = (REPOSITORY / pair.asset_entry.urdf_path).resolve()
    write_order9_c3_nominal_trajectory_artifact(
        nominal,
        output_dir=root / "artifact",
        bucket_id=case.candidate_id,
        split=DatasetSplit.VALIDATION,
        task_spec=case.task_spec,
        lift_clearance_m=0.3,
        task_spec_sha256=hash_file(task_path),
        structural_hash=pair.held_out_entry.structural_hash,
        physical_model_hash=case.physical_model_hash,
        robot_urdf_path=urdf,
    )
    return root / "artifact/manifest.json"


def _prepare_reference_worker(
    pair: Order9R1HeldOutPairV16,
    source_case: Any,
    output_root_text: str,
) -> dict[str, Any]:
    output_root = Path(output_root_text)
    case = _reference_case(pair, source_case)
    destination = output_root / "references" / pair.bucket_id / case.candidate_id
    record_path = output_root / "preparation_records" / f"{pair.bucket_id}.json"
    if record_path.is_file():
        record = _read_json(record_path)
        if record.get("status") != "accepted":
            return record
        runtime_v13._validate_prepared_case(case, destination)
        return record
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
    pipeline = Order9R1ExactMarginTeacherScreenPipelineV13(
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
    started = time.perf_counter()
    try:
        prepared = pipeline.prepare(case)
        prepared.validate_for(case)
        if not prepared.teacher_trajectory_complete:
            raise SchemaValidationError(str(prepared.failure_reason))
        screen = pipeline.screen(case, prepared)
        if not screen.accepted or not screen.eligible_for_full_control_test:
            raise SchemaValidationError(
                "R1 held-out reference fast screen failed: "
                + ",".join(screen.violation_codes)
            )
        source_artifact = _write_reference_source_nominal(
            case=case,
            nominal=prepared.payload,
            pair=pair,
            output_root=output_root,
        )
        case.source_bucket.metadata["accepted_nominal_trajectory"] = {
            "artifact_path": _portable(source_artifact),
            "artifact_sha256": hash_file(source_artifact),
            "r1_held_out_reference": True,
        }
        materialized = materialize_order9_r1_isaac_case(
            case=case,
            prepared=prepared,
            screen=screen,
            output_dir=destination,
            repository_root=REPOSITORY,
            approved_protocol_path=BASE_PROTOCOL,
        )
        finalize_order9_r1_v14_materialized_case(
            materialized,
            task_spec=case.task_spec,
            phase_time_scales=order9_r1_safe_phase_time_scales(
                case.source_bucket.module_count
            ),
            joint_rate_limit_rad_s=(
                float(screen.maximum_joint_rate_rad_s)
                + float(screen.minimum_joint_rate_margin_rad_s)
            ),
            repository_root=REPOSITORY,
            geometry_contract=geometry,
            clearance_contract=clearance,
            clearance_generation_certificate=(
                pipeline.clearance_generation_certificates.get(case.candidate_id)
            ),
        )
        runtime_v13._validate_prepared_case(case, destination)
        clearance_audit = _read_json(
            destination / ORDER9_R1_PLANNING_CLEARANCE_AUDIT_V14_FILENAME
        )
        minimum_clearance = float(
            clearance_audit["minimum_robot_support_clearance_lower_bound_m"]
        )
        if minimum_clearance + 1.0e-12 < 0.030:
            raise SchemaValidationError(
                "R1 held-out reference planning clearance is below 30 mm"
            )
        record = {
            "record_version": "order9_r1_held_out_reference_preparation_v16",
            "status": "accepted",
            "bucket_id": pair.bucket_id,
            "source_pool_split": "held_out",
            "structural_hash": pair.held_out_entry.structural_hash,
            "module_count": pair.held_out_entry.module_count,
            "paired_train_bucket_id": pair.paired_train_bucket.bucket_id,
            "candidate_id": case.candidate_id,
            "teacher_feasible": True,
            "fast_screen_passed": True,
            "complete_task_geometry_passed": True,
            "case_manifest": _binding(destination / "case_manifest.json"),
            "minimum_clearance_m": minimum_clearance,
            "elapsed_s": time.perf_counter() - started,
            "isaac_invoked": False,
            "controller_layers_invoked": False,
            "training_eligible": False,
        }
    except BaseException as error:
        record = {
            "record_version": "order9_r1_held_out_reference_preparation_v16",
            "status": "rejected",
            "bucket_id": pair.bucket_id,
            "source_pool_split": "held_out",
            "structural_hash": pair.held_out_entry.structural_hash,
            "module_count": pair.held_out_entry.module_count,
            "paired_train_bucket_id": pair.paired_train_bucket.bucket_id,
            "candidate_id": case.candidate_id,
            "error": f"{type(error).__name__}: {error}",
            "elapsed_s": time.perf_counter() - started,
            "isaac_invoked": False,
            "controller_layers_invoked": False,
            "training_eligible": False,
        }
    _atomic_json(record_path, record)
    return record


def _prepare_references(
    *,
    pairs: tuple[Order9R1HeldOutPairV16, ...],
    references: dict[str, Any],
    output_root: Path,
    workers: int,
) -> list[dict[str, Any]]:
    results = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        futures = {
            executor.submit(
                _prepare_reference_worker,
                pair,
                references[pair.bucket_id],
                str(output_root),
            ): pair.bucket_id
            for pair in pairs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                "ORDER9_R1_HELD_OUT_PREPARATION="
                + json.dumps(
                    {
                        "bucket_id": result["bucket_id"],
                        "status": result["status"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return sorted(results, key=lambda item: item["bucket_id"])


def _target_identity_task(
    pair: Order9R1HeldOutPairV16,
    source_case: Any,
) -> tuple[str, TaskSpec]:
    candidate_id = _target_candidate_id(pair, source_case)
    return candidate_id, _relabel_task(
        source_case.task_spec,
        pair=pair,
        candidate_id=candidate_id,
    )


def _prove_target_worker(
    pair: Order9R1HeldOutPairV16,
    source_case: Any,
    output_root_text: str,
) -> Path:
    output_root = Path(output_root_text)
    candidate_id, identity = _target_identity_task(pair, source_case)
    proof_path = output_root / "proofs" / pair.bucket_id / f"{candidate_id}.json"
    if proof_path.is_file():
        proof = _read_json(proof_path)
        if (
            proof.get("status") != "accepted"
            or proof.get("candidate_id") != candidate_id
        ):
            raise SchemaValidationError("R1 held-out cached proof differs")
        return proof_path
    reference_root = (
        output_root / "references" / pair.bucket_id / _reference_candidate_id(pair)
    )
    audit = audit_order9_r1_held_out_composed_scene_case_v16(
        reference_case_root=reference_root,
        target_identity_task=identity,
        repository_root=REPOSITORY,
    )
    target_task = TaskSpec.from_dict(audit.pop("target_task_spec"))
    target_path = output_root / "target_tasks" / pair.bucket_id / f"{candidate_id}.json"
    if not target_path.is_file():
        _atomic_json(target_path, target_task.to_dict())
    return _atomic_json(
        proof_path,
        {
            "proof_version": "order9_r1_held_out_scene_equivalence_proof_v16",
            "status": "accepted",
            "candidate_id": candidate_id,
            "bucket_id": pair.bucket_id,
            "source_pool_split": "held_out",
            "structural_hash": pair.held_out_entry.structural_hash,
            "module_count": pair.held_out_entry.module_count,
            "paired_train_bucket_id": pair.paired_train_bucket.bucket_id,
            "sample_kind": source_case.sample_kind,
            "sample_index": source_case.sample_index,
            "target_offsets": [
                source_case.x_offset_m,
                source_case.y_offset_m,
                source_case.yaw_offset_rad,
            ],
            "target_task_spec": _binding(target_path),
            "scene_transfer_audit": audit,
            "legacy_validation_evidence_reused": False,
            "pi_l_actor_command_applied": False,
            "controller_layers_invoked": False,
            "isaac_invoked": False,
            "training_eligible": False,
        },
    )


def _prove_targets(
    *,
    pairs: tuple[Order9R1HeldOutPairV16, ...],
    targets: dict[str, list[Any]],
    output_root: Path,
    workers: int,
) -> Path:
    proof_paths = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        futures = {
            executor.submit(_prove_target_worker, pair, target, str(output_root)): (
                pair.bucket_id,
                target.candidate_id,
            )
            for pair in pairs
            for target in targets[pair.bucket_id]
        }
        for index, future in enumerate(as_completed(futures), start=1):
            proof_paths.append(future.result())
            if index % 25 == 0 or index == len(futures):
                print(
                    f"ORDER9_R1_HELD_OUT_PROOF_PROGRESS={index}/{len(futures)}",
                    flush=True,
                )
    proofs = [_read_json(path) for path in sorted(proof_paths)]
    result_path = output_root / "results/held_out_scene_proofs.json"
    if result_path.is_file():
        return result_path
    by_bucket = {}
    for pair in pairs:
        values = [proof for proof in proofs if proof["bucket_id"] == pair.bucket_id]
        by_bucket[pair.bucket_id] = {
            "module_count": pair.held_out_entry.module_count,
            "candidate_count": len(values),
            "lattice_count": sum(value["sample_kind"] == "lattice" for value in values),
            "interior_count": sum(
                value["sample_kind"] == "interior" for value in values
            ),
            "accepted_count": sum(value["status"] == "accepted" for value in values),
        }
    maximum_position_error = max(
        float(
            proof["scene_transfer_audit"]["invariant_audit"][
                "maximum_object_relative_position_error_m"
            ]
        )
        for proof in proofs
    )
    maximum_attitude_error = max(
        float(
            proof["scene_transfer_audit"]["invariant_audit"][
                "maximum_object_relative_attitude_error_rad"
            ]
        )
        for proof in proofs
    )
    minimum_clearance = min(
        float(
            proof["scene_transfer_audit"]["collision_certificate"][
                "minimum_clearance_m"
            ]
        )
        for proof in proofs
    )
    return _atomic_json(
        result_path,
        {
            "result_version": "order9_r1_held_out_scene_proof_result_v16",
            "status": (
                "accepted"
                if len(proofs) == 14 * 31
                and all(proof["status"] == "accepted" for proof in proofs)
                else "rejected"
            ),
            "source_pool_split": "held_out",
            "bucket_count": len(pairs),
            "candidate_count": len(proofs),
            "lattice_count": sum(proof["sample_kind"] == "lattice" for proof in proofs),
            "interior_count": sum(
                proof["sample_kind"] == "interior" for proof in proofs
            ),
            "accepted_proof_count": sum(
                proof["status"] == "accepted" for proof in proofs
            ),
            "maximum_object_relative_position_error_m": maximum_position_error,
            "maximum_object_relative_attitude_error_rad": maximum_attitude_error,
            "minimum_collision_clearance_m": minimum_clearance,
            "proof_substep_count": 2,
            "proof_substeps_are_not_temporal_trajectory_phases": True,
            "by_bucket": by_bucket,
            "proofs": [_binding(path) for path in sorted(proof_paths)],
            "legacy_validation_evidence_reused": False,
            "isaac_invoked": False,
            "training_eligible": False,
        },
    )


def _replace(arguments: list[str], option: str, value: str) -> None:
    indices = [index for index, item in enumerate(arguments) if item == option]
    if len(indices) != 1 or indices[0] + 1 >= len(arguments):
        raise SchemaValidationError(f"R1 held-out command lacks one {option}")
    arguments[indices[0] + 1] = value


def _held_out_case_task_path(case_root: Path, repository: Path) -> Path:
    """Resolve a held-out task from its case instead of a C3 bucket sibling."""

    root = Path(case_root).resolve()
    task_path = root / "task_spec.json"
    case = _read_json(root / "case_manifest.json")
    task = TaskSpec.from_json(task_path.read_text(encoding="utf-8"))
    binding = case.get("task_spec", {})
    bound_path = (REPOSITORY / str(binding.get("path", ""))).resolve()
    if (
        task.metadata.get("dataset_split") != "held_out"
        or task.metadata.get("r1_calibration_candidate_id") != case.get("candidate_id")
        or bound_path.name != "task_spec.json"
        or OUTPUT_ROOT not in bound_path.parents
        or binding.get("sha256") != hash_file(task_path)
        or Path(repository).resolve() != REPOSITORY
    ):
        raise SchemaValidationError("R1 held-out case task binding differs")
    return task_path


def _materialize_representative(
    *,
    pair: Order9R1HeldOutPairV16,
    source_case: Any,
    output_root: Path,
) -> Path:
    candidate_id, identity = _target_identity_task(pair, source_case)
    destination = output_root / "representatives" / pair.bucket_id / candidate_id
    if destination.is_dir():
        load_order9_r1_isaac_case(destination / "case_manifest.json", REPOSITORY)
        return destination
    reference = (
        output_root / "references" / pair.bucket_id / _reference_candidate_id(pair)
    )
    reference_task = TaskSpec.from_json(
        (reference / "task_spec.json").read_text(encoding="utf-8")
    )
    midpoint, final_task, _delta_one, _delta_two, _overall = (
        build_order9_r1_held_out_composed_target_v16(
            reference_task,
            target_identity_task=identity,
        )
    )
    midpoint_id = str(midpoint.metadata["r1_calibration_candidate_id"])
    intermediate = output_root / "intermediate" / pair.bucket_id / midpoint_id
    original_task_resolver = yaw_repair_runtime._source_bucket_task_path
    yaw_repair_runtime._source_bucket_task_path = _held_out_case_task_path
    try:
        if not intermediate.is_dir():
            derive_order9_r1_planar_nominal_transfer_case(
                reference_case_root=reference,
                target_task_spec=midpoint,
                target_candidate_id=midpoint_id,
                target_level_id=f"{TARGET_LEVEL}_proof_midpoint",
                target_seed=int(
                    pair.held_out_entry.requested_seed + source_case.sample_index
                ),
                destination_case_root=intermediate,
                repository_root=REPOSITORY,
            )
        derive_order9_r1_planar_nominal_transfer_case(
            reference_case_root=intermediate,
            target_task_spec=final_task,
            target_candidate_id=candidate_id,
            target_level_id=TARGET_LEVEL,
            target_seed=int(
                pair.held_out_entry.requested_seed + source_case.sample_index
            ),
            destination_case_root=destination,
            repository_root=REPOSITORY,
        )
    finally:
        yaw_repair_runtime._source_bucket_task_path = original_task_resolver
    load_order9_r1_isaac_case(destination / "case_manifest.json", REPOSITORY)
    return destination


def _prepare_isaac_jobs(
    *,
    pair: Order9R1HeldOutPairV16,
    targets: list[Any],
    output_root: Path,
    isaac_python: str,
    rollout_steps: int,
) -> list[dict[str, Any]]:
    representatives = [
        source_case
        for source_case in targets
        if source_case.sample_kind == "lattice"
        and source_case.sample_index in ISAAC_SAMPLE_INDICES
    ]
    if len(representatives) != 2:
        raise SchemaValidationError("R1 held-out representative set differs")
    prepared_jobs = []
    for source_case in sorted(representatives, key=lambda case: case.sample_index):
        root = _materialize_representative(
            pair=pair,
            source_case=source_case,
            output_root=output_root,
        )
        materialized = load_order9_r1_isaac_case(
            root / "case_manifest.json", REPOSITORY
        )
        command, log_path = order9_r1_nominal_case_command(
            materialized,
            repository_root=REPOSITORY,
            python_executable=isaac_python,
            rollout_steps=rollout_steps,
        )
        indices = [
            index
            for index, value in enumerate(command)
            if Path(value).name == "order9_vectorized_isaac_rollout.py"
        ]
        if len(indices) != 1:
            raise SchemaValidationError("R1 held-out rollout command differs")
        argv = list(command[indices[0] + 1 :])
        _replace(argv, "--num-envs", "1")
        _replace(argv, "--evaluation-episode-count", "1")
        item = {
            "candidate_id": materialized.manifest.candidate_id,
            "sample_index": source_case.sample_index,
            "case_root": root,
        }
        pair_root = output_root / "representatives" / pair.bucket_id
        suffix = f"lattice_{source_case.sample_index:02d}"
        job_path = pair_root / f"isaac_job_{suffix}.json"
        replay_root = pair_root / f"isaac_replay_{suffix}"
        if not job_path.is_file():
            _atomic_json(
                job_path,
                {
                    "version": "order9_r1_held_out_representative_job_v16",
                    "source_pool_split": "held_out",
                    "held_out_bucket_id": pair.bucket_id,
                    "structural_hash": pair.held_out_entry.structural_hash,
                    "r1_diagnostic_replay_count": 1,
                    "r1_nominal_additional_compression_sweep_mm": "5.0",
                    "r1_release_height_offset_m": 0.0,
                    "r1_scene_support_from_task": True,
                    "training_eligible": False,
                    "formal_teacher_collection_authorized": False,
                    "jobs": [
                        {
                            "name": materialized.manifest.candidate_id,
                            "argv": argv,
                            "log_path": str(log_path),
                        }
                    ],
                },
            )
        prepared_jobs.append(
            {
                "bucket_id": pair.bucket_id,
                "structural_hash": pair.held_out_entry.structural_hash,
                "module_count": pair.held_out_entry.module_count,
                "job_path": job_path,
                "output_root": replay_root,
                "items": [item],
                "command": [
                    isaac_python,
                    str(WRAPPER),
                    "--r1-batched-jobs",
                    str(job_path),
                    "--r1-batched-output-root",
                    str(replay_root),
                    "--no-formal-annotation",
                ],
            }
        )
    return prepared_jobs


def _episode_path(job: dict[str, Any], item: dict[str, Any]) -> Path:
    return job["output_root"] / item["candidate_id"] / "isaac/evaluation_episodes.jsonl"


def _run_isaac_job(job: dict[str, Any]) -> tuple[str, int]:
    if all(_episode_path(job, item).is_file() for item in job["items"]):
        return job["bucket_id"], 0
    job_path = Path(job["job_path"])
    log_path = job_path.with_name(f"{job_path.stem}_launcher.log")
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    environment["TORCHINDUCTOR_COMPILE_THREADS"] = "8"
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            job["command"],
            cwd=REPOSITORY,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return job["bucket_id"], completed.returncode


def _validate_isaac_job(job: dict[str, Any]) -> dict[str, Any]:
    entries = []
    for item in job["items"]:
        episode_path = _episode_path(job, item)
        records = [
            json.loads(line)
            for line in episode_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        raw_path = episode_path.with_name("evaluation_rollout.pt")
        raw = torch.load(raw_path, map_location="cpu", weights_only=False)
        task = _read_json(Path(item["case_root"]) / "task_spec.json")
        expected_support = task["scene"]["environment"]["support_surfaces"][0][
            "pose_world"
        ]
        metadata = raw.get("metadata", {})
        actual_support = metadata["task_specs"][0]["scene"]["environment"][
            "support_surfaces"
        ][0]["pose_world"]
        accepted = (
            len(records) == 1
            and records[0].get("task_success") is True
            and records[0].get("safety_failure") is False
            and int(records[0].get("fallback_decision_count", -1)) == 0
            and float(records[0].get("metrics", {}).get("hard_collision", -1.0)) == 0.0
            and float(records[0].get("metrics", {}).get("object_dropped", -1.0)) == 0.0
            and float(records[0].get("metrics", {}).get("timeout", -1.0)) == 0.0
            and metadata.get("r1_scene_support_from_task") is True
            and metadata.get("pi_l_actor_command_applied") is False
            and metadata.get("promotion_evidence_eligible") is False
            and metadata.get("training_eligible") is False
            and actual_support == expected_support
        )
        entries.append(
            {
                "candidate_id": item["candidate_id"],
                "sample_index": item["sample_index"],
                "status": "accepted" if accepted else "rejected",
                "episode_count": len(records),
                "success_count": sum(
                    record.get("task_success") is True for record in records
                ),
                "safety_failure_count": sum(
                    record.get("safety_failure") is True for record in records
                ),
                "fallback_count": sum(
                    int(record.get("fallback_decision_count", 0)) for record in records
                ),
                "task_support_pose_world": expected_support,
                "isaac_support_pose_world": actual_support,
                "episodes": _binding(episode_path),
                "raw_rollout": _binding(raw_path),
            }
        )
    return {
        "bucket_id": job["bucket_id"],
        "structural_hash": job["structural_hash"],
        "module_count": job["module_count"],
        "status": (
            "accepted"
            if all(entry["status"] == "accepted" for entry in entries)
            else "rejected"
        ),
        "episode_count": sum(entry["episode_count"] for entry in entries),
        "success_count": sum(entry["success_count"] for entry in entries),
        "safety_failure_count": sum(entry["safety_failure_count"] for entry in entries),
        "fallback_count": sum(entry["fallback_count"] for entry in entries),
        "entries": entries,
    }


def _run_isaac(
    *, jobs: list[dict[str, Any]], output_root: Path, maximum_parallel: int
) -> Path:
    with ThreadPoolExecutor(max_workers=maximum_parallel) as executor:
        futures = {executor.submit(_run_isaac_job, job): job for job in jobs}
        for future in as_completed(futures):
            bucket_id, returncode = future.result()
            print(
                "ORDER9_R1_HELD_OUT_ISAAC="
                + json.dumps(
                    {"bucket_id": bucket_id, "returncode": returncode},
                    sort_keys=True,
                ),
                flush=True,
            )
    results = [_validate_isaac_job(job) for job in jobs]
    path = output_root / "results/held_out_representative_isaac.json"
    if path.is_file():
        return path
    return _atomic_json(
        path,
        {
            "result_version": "order9_r1_held_out_representative_isaac_v16",
            "status": (
                "accepted"
                if all(result["status"] == "accepted" for result in results)
                else "rejected"
            ),
            "source_pool_split": "held_out",
            "bucket_count": len({result["bucket_id"] for result in results}),
            "process_count": len(results),
            "candidate_count": sum(len(result["entries"]) for result in results),
            "episode_count": sum(result["episode_count"] for result in results),
            "success_count": sum(result["success_count"] for result in results),
            "safety_failure_count": sum(
                result["safety_failure_count"] for result in results
            ),
            "fallback_count": sum(result["fallback_count"] for result in results),
            "pi_l_actor_command_applied": False,
            "qpid_qp_applied": True,
            "local_servo_applied": True,
            "entries": results,
            "training_eligible": False,
        },
    )


def main() -> int:
    args = _parser().parse_args()
    if (
        not 1 <= args.preparation_workers <= 16
        or not 1 <= args.proof_workers <= 16
        or not 1 <= args.maximum_parallel_isaac <= 4
        or args.rollout_steps < 1
        or os.environ.get("PYTHONHASHSEED") != "0"
    ):
        raise ValueError("R1 held-out runtime contract differs")
    protocol, approval = _load_contract()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    pairs, references, targets = _pairs_and_cases()
    unusedness = _audit_unusedness(output_root=output_root, pairs=pairs)
    preparation = _prepare_references(
        pairs=pairs,
        references=references,
        output_root=output_root,
        workers=args.preparation_workers,
    )
    if not all(record.get("status") == "accepted" for record in preparation):
        print(
            "ORDER9_R1_HELD_OUT_PREPARATION_REJECTED="
            + json.dumps(
                {
                    "accepted": sum(
                        record.get("status") == "accepted" for record in preparation
                    ),
                    "total": len(preparation),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 2
    if args.prepare_only:
        print("ORDER9_R1_HELD_OUT_PREPARED=14/14", flush=True)
        return 0
    proof_result = _prove_targets(
        pairs=pairs,
        targets=targets,
        output_root=output_root,
        workers=args.proof_workers,
    )
    proof_summary = _read_json(proof_result)
    if proof_summary.get("status") != "accepted":
        return 3
    jobs = [
        job
        for pair in pairs
        for job in _prepare_isaac_jobs(
            pair=pair,
            targets=targets[pair.bucket_id],
            output_root=output_root,
            isaac_python=args.isaac_python,
            rollout_steps=args.rollout_steps,
        )
    ]
    if args.proof_only:
        print(
            "ORDER9_R1_HELD_OUT_PROVED="
            + json.dumps(
                {"proof_count": 434, "prepared_isaac_job_count": len(jobs)},
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    isaac_result = _run_isaac(
        jobs=jobs,
        output_root=output_root,
        maximum_parallel=args.maximum_parallel_isaac,
    )
    isaac_summary = _read_json(isaac_result)
    status = (
        "accepted"
        if proof_summary.get("status") == "accepted"
        and isaac_summary.get("status") == "accepted"
        and int(isaac_summary.get("episode_count", -1)) == 28
        and int(isaac_summary.get("success_count", -1)) == 28
        and int(isaac_summary.get("safety_failure_count", -1)) == 0
        and int(isaac_summary.get("fallback_count", -1)) == 0
        else "rejected"
    )
    ledger_path = output_root / "held_out_confirmation_result.json"
    if ledger_path.is_file():
        raise FileExistsError(ledger_path)
    ledger = {
        "ledger_version": "order9_r1_held_out_scene_confirmation_result_v16",
        "runner_version": RUNNER_VERSION,
        "status": status,
        "r1_range_confirmed": status == "accepted",
        "accepted_level_id": TARGET_LEVEL if status == "accepted" else None,
        "selected_position_half_span_m": 0.04,
        "selected_yaw_half_span_rad": 0.3490658503988659,
        "source_pool_split": "held_out",
        "held_out_bucket_count": len(pairs),
        "candidate_count": 14 * 31,
        "lattice_candidate_count": 14 * 27,
        "interior_candidate_count": 14 * 4,
        "legacy_validation_evidence_reused": False,
        "unusedness_audit": _binding(unusedness),
        "preparation_records": [
            _binding(output_root / "preparation_records" / f"{pair.bucket_id}.json")
            for pair in pairs
        ],
        "scene_proofs": _binding(proof_result),
        "representative_isaac": _binding(isaac_result),
        "complete_scene_transform_required": True,
        "support_transformed_with_scene": True,
        "proof_substep_count": 2,
        "proof_substeps_are_not_temporal_trajectory_phases": True,
        "pi_l_actor_command_applied": False,
        "qpid_qp_applied_for_isaac": True,
        "local_servo_applied_for_isaac": True,
        "formal_teacher_collection_authorized": False,
        "training_eligible": False,
        "next_entry_point": "r1_teacher_trajectory_collection_preparation",
        "elapsed_s": time.perf_counter() - started,
        "bindings": {
            "protocol": _binding(PROTOCOL),
            "approval": _binding(APPROVAL),
            "parent_selection_result": protocol["parent_selection_result"],
            "morphology_pool": protocol["morphology_pool"],
            "morphology_asset_manifest": protocol["morphology_asset_manifest"],
            "source_task_manifest": protocol["source_task_manifest"],
            "confirmation_implementation": protocol["confirmation_implementation"],
            "runner_implementation": protocol["runner_implementation"],
            "scene_transfer_implementation": protocol["scene_transfer_implementation"],
            "isaac_wrapper_implementation": protocol["isaac_wrapper_implementation"],
            "protected_c3_checkpoint": protocol["protected_c3_checkpoint"],
            "protected_c3_rollout": protocol["protected_c3_rollout"],
            "protected_c3_curriculum": protocol["protected_c3_curriculum"],
            "protected_c3_release_ledger": protocol["protected_c3_release_ledger"],
        },
        "approval_record_version": approval["record_version"],
    }
    _atomic_json(ledger_path, ledger)
    print(
        "ORDER9_R1_HELD_OUT_COMPLETE="
        + json.dumps(
            {
                "status": status,
                "accepted_level_id": ledger["accepted_level_id"],
                "ledger": str(ledger_path),
                "ledger_sha256": hash_file(ledger_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if status == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
