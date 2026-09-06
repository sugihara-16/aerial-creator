from __future__ import annotations

"""Fail-closed preparation for the formal Order-9 R1 teacher collection."""

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping

from amsrr.schemas.common import SchemaValidationError
from amsrr.utils.hashing import hash_file, stable_hash

ORDER9_R1_TEACHER_COLLECTION_PROTOCOL_VERSION = (
    "order9_r1_teacher_collection_protocol_v19"
)
ORDER9_R1_TEACHER_COLLECTION_PLAN_VERSION = "order9_r1_teacher_collection_plan_v19"
ORDER9_R1_TEACHER_COLLECTION_PREFLIGHT_VERSION = (
    "order9_r1_teacher_collection_preflight_v19"
)
SPLIT_ORDER = ("train", "validation", "held_out")
TASK_PHASES = (
    "approach",
    "contact_acquisition",
    "lift",
    "transport",
    "place",
    "release",
    "retreat",
    "settle",
)


def read_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SchemaValidationError(f"JSON object required: {source}")
    return value


def load_order9_r1_teacher_collection_protocol(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    repository = Path(repository_root).resolve()
    protocol = read_json_object(path)
    expected_split = {"train": 14, "validation": 3, "held_out": 3}
    if (
        protocol.get("protocol_version")
        != ORDER9_R1_TEACHER_COLLECTION_PROTOCOL_VERSION
        or protocol.get("status") != "preparation_approved"
        or protocol.get("stage_id") != "r1_teacher_trajectory_collection"
        or protocol.get("collection_launch_authorized") is not False
        or protocol.get("training_authorized") is not False
        or int(protocol.get("total_episode_count", -1)) != 140
        or int(protocol.get("seed_start", -1)) != 29009
        or protocol.get("module_counts") != list(range(2, 9))
        or int(protocol.get("episodes_per_module_count", -1)) != 20
        or protocol.get("split_counts_per_module_count") != expected_split
        or not math.isclose(float(protocol.get("position_half_span_m", -1)), 0.04)
        or not math.isclose(
            float(protocol.get("yaw_half_span_rad", -1)), math.radians(20.0)
        )
        or protocol.get("whole_scene_planar_transfer") is not True
        or protocol.get("support_transformed_with_scene") is not True
        or protocol.get("all_eight_task_phases_required") is not True
        or protocol.get("fast_exact_tracking_screen_required") is not True
        or protocol.get("real_isaac_success_required") is not True
        or protocol.get("pi_l_actor_command_applied") is not False
        or protocol.get("qpid_qp_applied") is not True
        or protocol.get("local_servo_applied") is not True
        or int(protocol.get("minimum_free_storage_gib", -1)) != 128
    ):
        raise SchemaValidationError("R1 teacher collection protocol differs")
    bindings = protocol.get("source_bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise SchemaValidationError("R1 teacher collection bindings are missing")
    for label, binding in bindings.items():
        validate_binding(binding, repository_root=repository, label=label)
    return protocol


def validate_binding(
    binding: object, *, repository_root: str | Path, label: str
) -> Path:
    repository = Path(repository_root).resolve()
    if not isinstance(binding, Mapping):
        raise SchemaValidationError(f"R1 binding is missing: {label}")
    relative = binding.get("path")
    expected = binding.get("sha256")
    if not isinstance(relative, str) or not relative or not isinstance(expected, str):
        raise SchemaValidationError(f"R1 binding is invalid: {label}")
    path = (repository / relative).resolve()
    if repository != path and repository not in path.parents:
        raise SchemaValidationError(f"R1 binding escapes repository: {label}")
    if not path.is_file() or hash_file(path) != expected:
        raise SchemaValidationError(f"R1 bound bytes changed: {label}")
    return path


def build_order9_r1_teacher_collection_plan(
    protocol: Mapping[str, Any], *, repository_root: str | Path
) -> dict[str, Any]:
    repository = Path(repository_root).resolve()
    train_sources = _train_sources(protocol, repository)
    evaluation_sources = _evaluation_sources(protocol, repository)
    entries: list[dict[str, Any]] = []
    global_index = 0
    split_counts = protocol["split_counts_per_module_count"]
    for module_count in protocol["module_counts"]:
        offsets = _latin_hypercube_offsets(
            module_count=int(module_count),
            count=int(protocol["episodes_per_module_count"]),
            position_half_span_m=float(protocol["position_half_span_m"]),
            yaw_half_span_rad=float(protocol["yaw_half_span_rad"]),
            seed=int(protocol["seed_start"]),
        )
        cell_index = 0
        for split in SPLIT_ORDER:
            source_pool = (
                train_sources[int(module_count)]
                if split == "train"
                else evaluation_sources[int(module_count)]
            )
            for split_index in range(int(split_counts[split])):
                seed = int(protocol["seed_start"]) + global_index
                x_offset, y_offset, yaw_offset = offsets[cell_index]
                source = source_pool[split_index % len(source_pool)]
                task_id = f"order9-r1-teacher-m{module_count}-{split}-{split_index:02d}"
                pose_payload = {
                    "x_offset_m": x_offset,
                    "y_offset_m": y_offset,
                    "yaw_offset_rad": yaw_offset,
                }
                entries.append(
                    {
                        "index": global_index,
                        "episode_id": f"r1-teacher-{global_index:06d}",
                        "task_id": task_id,
                        "split": split,
                        "split_index_within_module": split_index,
                        "module_count": int(module_count),
                        "seed": seed,
                        **pose_payload,
                        "pose_condition_hash": stable_hash(pose_payload),
                        "source_bucket_id": source["source_bucket_id"],
                        "source_structural_hash": source["structural_hash"],
                        "source_case_manifest": source["case_manifest"],
                        "source_evidence": source["source_evidence"],
                        "known_repair_profile": source["known_repair_profile"],
                        "task_phases": list(TASK_PHASES),
                        "whole_scene_planar_transfer": True,
                        "support_transformed_with_scene": True,
                        "fast_exact_tracking_screen_required": True,
                        "real_isaac_success_required": True,
                        "pi_l_actor_command_applied": False,
                        "qpid_qp_applied": True,
                        "local_servo_applied": True,
                    }
                )
                global_index += 1
                cell_index += 1
    plan = {
        "plan_version": ORDER9_R1_TEACHER_COLLECTION_PLAN_VERSION,
        "status": "prepared_not_started",
        "collection_launch_authorized": False,
        "training_authorized": False,
        "protocol": {
            "path": _portable(Path(protocol["_path"]), repository),
            "sha256": hash_file(protocol["_path"]),
        },
        "episode_count": len(entries),
        "entries": entries,
    }
    validate_order9_r1_teacher_collection_plan(plan, repository_root=repository)
    return plan


def validate_order9_r1_teacher_collection_plan(
    plan: Mapping[str, Any], *, repository_root: str | Path
) -> None:
    repository = Path(repository_root).resolve()
    entries = plan.get("entries")
    if (
        plan.get("plan_version") != ORDER9_R1_TEACHER_COLLECTION_PLAN_VERSION
        or plan.get("status") != "prepared_not_started"
        or plan.get("collection_launch_authorized") is not False
        or plan.get("training_authorized") is not False
        or int(plan.get("episode_count", -1)) != 140
        or not isinstance(entries, list)
        or len(entries) != 140
    ):
        raise SchemaValidationError("R1 teacher collection plan header differs")
    validate_binding(plan.get("protocol"), repository_root=repository, label="protocol")
    expected_indices = list(range(140))
    if [entry.get("index") for entry in entries] != expected_indices:
        raise SchemaValidationError("R1 teacher collection indices differ")
    for field in ("episode_id", "task_id", "seed", "pose_condition_hash"):
        values = [entry.get(field) for entry in entries]
        if len(set(values)) != len(values):
            raise SchemaValidationError(f"R1 teacher collection {field} is not unique")
    if [entry["seed"] for entry in entries] != list(range(29009, 29149)):
        raise SchemaValidationError("R1 teacher collection seeds differ")
    counts = Counter(
        (entry.get("module_count"), entry.get("split")) for entry in entries
    )
    for module_count in range(2, 9):
        if [counts[(module_count, split)] for split in SPLIT_ORDER] != [14, 3, 3]:
            raise SchemaValidationError("R1 teacher collection stratification differs")
    source_bindings: dict[tuple[str, str], Mapping[str, Any]] = {}
    for entry in entries:
        if (
            entry.get("task_phases") != list(TASK_PHASES)
            or entry.get("whole_scene_planar_transfer") is not True
            or entry.get("support_transformed_with_scene") is not True
            or entry.get("fast_exact_tracking_screen_required") is not True
            or entry.get("real_isaac_success_required") is not True
            or entry.get("pi_l_actor_command_applied") is not False
            or entry.get("qpid_qp_applied") is not True
            or entry.get("local_servo_applied") is not True
        ):
            raise SchemaValidationError(
                "R1 teacher collection execution contract differs"
            )
        if (
            abs(float(entry.get("x_offset_m", math.inf))) > 0.04
            or abs(float(entry.get("y_offset_m", math.inf))) > 0.04
            or abs(float(entry.get("yaw_offset_rad", math.inf)))
            > math.radians(20.0) + 1.0e-12
        ):
            raise SchemaValidationError("R1 teacher collection pose is out of range")
        binding = entry.get("source_case_manifest")
        key = (
            (str(binding.get("path", "")), str(binding.get("sha256", "")))
            if isinstance(binding, Mapping)
            else ("", "")
        )
        source_bindings[key] = binding
    for index, binding in enumerate(source_bindings.values()):
        validate_binding(
            binding, repository_root=repository, label=f"source_case_manifest_{index}"
        )


def build_order9_r1_teacher_collection_preflight(
    *,
    protocol: Mapping[str, Any],
    plan_path: str | Path,
    repository_root: str | Path,
    free_storage_bytes: int,
    isaac_python: str | Path,
    isaac_import_ok: bool,
    active_collection_process_count: int,
    existing_collection_record_count: int = 0,
) -> dict[str, Any]:
    repository = Path(repository_root).resolve()
    plan_source = Path(plan_path).resolve()
    plan = read_json_object(plan_source)
    validate_order9_r1_teacher_collection_plan(plan, repository_root=repository)
    minimum = int(protocol["minimum_free_storage_gib"]) * 1024**3
    python_path = Path(isaac_python).resolve()
    checks = {
        "plan_valid": True,
        "source_and_protected_hashes_valid": True,
        "free_storage_sufficient": free_storage_bytes >= minimum,
        "isaac_python_exists": python_path.is_file(),
        "isaac_import_ok": bool(isaac_import_ok),
        "no_active_collection_process": active_collection_process_count == 0,
        "collection_not_started": existing_collection_record_count == 0,
        "collection_launch_authorized": False,
    }
    ready = all(
        value for key, value in checks.items() if key != "collection_launch_authorized"
    )
    return {
        "preflight_version": ORDER9_R1_TEACHER_COLLECTION_PREFLIGHT_VERSION,
        "status": "ready_awaiting_explicit_launch" if ready else "not_ready",
        "ready_for_explicit_launch": ready,
        "collection_started": existing_collection_record_count > 0,
        "collection_launch_authorized": False,
        "plan": {
            "path": _portable(plan_source, repository),
            "sha256": hash_file(plan_source),
        },
        "episode_count": 140,
        "free_storage_gib": free_storage_bytes / 1024**3,
        "minimum_free_storage_gib": int(protocol["minimum_free_storage_gib"]),
        "existing_collection_record_count": existing_collection_record_count,
        "isaac_python": str(python_path),
        "checks": checks,
        "next_action": "issue_explicit_collection_start_instruction",
    }


def _train_sources(
    protocol: Mapping[str, Any], repository: Path
) -> dict[int, list[dict[str, Any]]]:
    result_path = validate_binding(
        protocol["source_bindings"]["range_selection_v15"],
        repository_root=repository,
        label="range_selection_v15",
    )
    result = read_json_object(result_path)
    level_binding = result.get("levels", {}).get("r1_l2_20mm_10deg")
    level_path = validate_binding(
        level_binding, repository_root=repository, label="range_selection_v15_l2"
    )
    entries = read_json_object(level_path).get("entries")
    if not isinstance(entries, list):
        raise SchemaValidationError("R1 v15 L2 entries are missing")
    selected = [
        entry
        for entry in entries
        if entry.get("sample_kind") == "lattice"
        and int(entry.get("sample_index", -1)) == 13
        and int(entry.get("success_count", -1)) == int(entry.get("episode_count", -2))
        and int(entry.get("safety_failure_count", -1)) == 0
        and int(entry.get("fallback_count", -1)) == 0
    ]
    by_module: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in selected:
        bucket = str(entry["source_bucket_id"])
        candidate = str(entry["candidate_id"])
        case_path = repository / (
            "artifacts/p4_full/order9/r1_teacher/range_selection_v13/prepared/"
            f"r1_l2_20mm_10deg/{candidate}/case_manifest.json"
        )
        case = read_json_object(case_path)
        structural_hash = _case_structural_hash(case, repository)
        evidence_path = validate_binding(
            entry.get("episodes"),
            repository_root=repository,
            label=f"{bucket}_successful_isaac_evidence",
        )
        by_module[int(entry["module_count"])].append(
            {
                "source_bucket_id": bucket,
                "structural_hash": structural_hash,
                "case_manifest": _binding(case_path, repository),
                "source_evidence": _binding(evidence_path, repository),
                "known_repair_profile": "standard_scene_transfer",
            }
        )
    if set(by_module) != set(range(2, 9)) or any(
        not values for values in by_module.values()
    ):
        raise SchemaValidationError("R1 train source strata are incomplete")
    return {
        key: sorted(values, key=lambda item: item["source_bucket_id"])
        for key, values in by_module.items()
    }


def _evaluation_sources(
    protocol: Mapping[str, Any], repository: Path
) -> dict[int, list[dict[str, Any]]]:
    result_path = validate_binding(
        protocol["source_bindings"]["held_out_confirmation_v16"],
        repository_root=repository,
        label="held_out_confirmation_v16",
    )
    result = read_json_object(result_path)
    representative_path = validate_binding(
        result.get("representative_isaac"),
        repository_root=repository,
        label="held_out_representative_isaac",
    )
    representative = read_json_object(representative_path)
    successful_evidence: dict[str, list[dict[str, str]]] = defaultdict(list)
    for result_entry in representative.get("entries", []):
        if not isinstance(result_entry, Mapping):
            raise SchemaValidationError("R1 v16 representative entry is invalid")
        bucket_id = str(result_entry.get("bucket_id", ""))
        nested = result_entry.get("entries")
        if not isinstance(nested, list) or len(nested) != 1:
            raise SchemaValidationError("R1 v16 representative nesting differs")
        candidate = nested[0]
        if (
            result_entry.get("status") == "accepted"
            and int(result_entry.get("success_count", -1)) == 1
            and int(result_entry.get("safety_failure_count", -1)) == 0
            and int(result_entry.get("fallback_count", -1)) == 0
            and isinstance(candidate, Mapping)
            and candidate.get("status") == "accepted"
        ):
            evidence_path = validate_binding(
                candidate.get("episodes"),
                repository_root=repository,
                label=f"{bucket_id}_representative_isaac",
            )
            successful_evidence[bucket_id].append(_binding(evidence_path, repository))
    by_module: dict[int, list[dict[str, Any]]] = defaultdict(list)
    records = result.get("preparation_records")
    if not isinstance(records, list):
        raise SchemaValidationError("R1 v16 preparation records are missing")
    for binding in records:
        record_path = validate_binding(
            binding, repository_root=repository, label="held_out_preparation"
        )
        record = read_json_object(record_path)
        bucket_id = str(record.get("bucket_id", ""))
        if (
            record.get("status") != "accepted"
            or len(successful_evidence[bucket_id]) != 2
        ):
            continue
        case_path = validate_binding(
            record.get("case_manifest"),
            repository_root=repository,
            label="held_out_case",
        )
        by_module[int(record["module_count"])].append(
            {
                "source_bucket_id": str(record["bucket_id"]),
                "structural_hash": str(record["structural_hash"]),
                "case_manifest": _binding(case_path, repository),
                "source_evidence": successful_evidence[bucket_id],
                "known_repair_profile": "standard_scene_transfer",
            }
        )
    if set(by_module) != set(range(2, 9)) or any(
        not values for values in by_module.values()
    ):
        raise SchemaValidationError("R1 evaluation source strata are incomplete")
    return {
        key: sorted(values, key=lambda item: item["source_bucket_id"])
        for key, values in by_module.items()
    }


def _case_structural_hash(case: Mapping[str, Any], repository: Path) -> str:
    binding = case.get("morphology_graph")
    path = validate_binding(
        binding, repository_root=repository, label="morphology_graph"
    )
    value = read_json_object(path)
    structural = case.get("structural_hash") or value.get("metadata", {}).get(
        "structural_hash"
    )
    if not isinstance(structural, str) or len(structural) != 64:
        structural = stable_hash(value)
    return structural


def _latin_hypercube_offsets(
    *,
    module_count: int,
    count: int,
    position_half_span_m: float,
    yaw_half_span_rad: float,
    seed: int,
) -> list[tuple[float, float, float]]:
    randomizer = random.Random(seed + 1009 * module_count)
    axes = []
    for half_span in (position_half_span_m, position_half_span_m, yaw_half_span_rad):
        permutation = list(range(count))
        randomizer.shuffle(permutation)
        axes.append(
            [
                -half_span + 2.0 * half_span * (bin_index + randomizer.random()) / count
                for bin_index in permutation
            ]
        )
    return [tuple(axis[index] for axis in axes) for index in range(count)]


def _binding(path: str | Path, repository: Path) -> dict[str, str]:
    source = Path(path).resolve()
    return {"path": _portable(source, repository), "sha256": hash_file(source)}


def _portable(path: str | Path, repository: Path) -> str:
    return Path(path).resolve().relative_to(repository).as_posix()


def protocol_with_path(protocol: dict[str, Any], path: str | Path) -> dict[str, Any]:
    value = dict(protocol)
    value["_path"] = str(Path(path).resolve())
    return value
