#!/usr/bin/env python3
from __future__ import annotations

"""Replay and repair the two v16 six-module object-drop cases."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import torch

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.schemas.common import SchemaValidationError  # noqa: E402
from amsrr.utils.hashing import hash_file  # noqa: E402

VERSION = "order9_r1_object_drop_repair_v18"
STRUCTURAL_HASH = "58a5c015f4db032c765412d4827d64629ccb6ae4fb2b3ce5feb0d0e6d7a03ece"
BUCKET_ID = "held-out-000045-58a5c015f4db"
SAMPLES = (0, 26)
PROTECTED_ROLLOUT = REPOSITORY / "scripts/order9_vectorized_isaac_rollout.py"
PROTECTED_ROLLOUT_SHA256 = (
    "e813f950dde32818b7375949bfb58baa983dde760e58da22cfbb79cc33b764c2"
)
PROTECTED_CHECKPOINT = REPOSITORY / (
    "artifacts/p4_full/order9/releases/c3_pi_l_promoted_update18_v1/"
    "checkpoint_update_000018.pt"
)
PROTECTED_CHECKPOINT_SHA256 = (
    "6ea412ccdfe983cb2b030b3514bc8982424522673b49def188d547120984357b"
)
V16_ROOT = (
    REPOSITORY
    / (
        "artifacts/p4_full/order9/r1_teacher/held_out_scene_confirmation_v16/"
        "representatives"
    )
    / BUCKET_ID
)
OUTPUT_ROOT = REPOSITORY / (
    "artifacts/p4_full/order9/r1_teacher/diagnostics/"
    "held_out_object_drop_repair_v18_final"
)
WRAPPER = REPOSITORY / "scripts/order9_r1_batched_nominal_compression_rollout.py"
ISAAC_PYTHON = "/home/leus/.local/share/mamba/envs/isaaclab3/bin/python"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-index", type=int, choices=SAMPLES)
    parser.add_argument("--additional-compression-mm", type=float, default=10.0)
    parser.add_argument(
        "--compression-by-contact-mm",
        default="17,23",
        help="Comma-separated compression for active contacts in slot order",
    )
    parser.add_argument("--lift-time-dilation", type=float, default=3.0)
    parser.add_argument("--maintain-vertical-scale", type=float, default=0.1)
    parser.add_argument("--grasp-centering-object-y-mm", type=float, default=30.0)
    parser.add_argument("--grasp-torque-bias-fraction", type=float)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--isaac-python", default=ISAAC_PYTHON)
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_text = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
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


def _binding(path: Path) -> dict[str, str]:
    return {
        "path": path.resolve().relative_to(REPOSITORY).as_posix(),
        "sha256": hash_file(path),
    }


def _source(sample_index: int) -> tuple[Path, Path]:
    suffix = f"lattice_{sample_index:02d}"
    job = V16_ROOT / f"isaac_job_{suffix}.json"
    episode_matches = tuple(
        (V16_ROOT / f"isaac_replay_{suffix}").glob("*/isaac/evaluation_episodes.jsonl")
    )
    if len(episode_matches) != 1:
        raise SchemaValidationError("R1 v18 source episode is not unique")
    return job, episode_matches[0]


def _prepare_job(
    *,
    sample_index: int,
    compression_mm: float,
    lift_time_dilation: float | None,
    grasp_centering_object_y_mm: float,
    grasp_torque_bias_fraction: float | None,
    compression_by_contact_mm: tuple[float, ...] | None,
    maintain_vertical_scale: float | None,
    output_root: Path,
) -> dict[str, Any]:
    source_job, source_episode = _source(sample_index)
    source = json.loads(source_job.read_text(encoding="utf-8"))
    source_records = [
        json.loads(line)
        for line in source_episode.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    jobs = source.get("jobs") if isinstance(source, dict) else None
    expected_suffix = f"__lattice_{sample_index:02d}"
    if (
        len(source_records) != 1
        or source_records[0].get("failure_reason") != "object_dropped"
        or source_records[0].get("task_success") is not False
        or source.get("structural_hash") != STRUCTURAL_HASH
        or source.get("held_out_bucket_id") != BUCKET_ID
        or source.get("r1_nominal_additional_compression_sweep_mm") != "5.0"
        or source.get("r1_scene_support_from_task") is not True
        or source.get("training_eligible") is not False
        or source.get("formal_teacher_collection_authorized") is not False
        or not isinstance(jobs, list)
        or len(jobs) != 1
        or not str(jobs[0].get("name", "")).endswith(expected_suffix)
    ):
        raise SchemaValidationError("R1 v18 source failure contract differs")
    case_root = output_root / f"lattice_{sample_index:02d}"
    replay_root = case_root / "isaac_replay"
    argv = list(jobs[0]["argv"])
    task_indices = [
        index for index, value in enumerate(argv) if value == "--task-spec-json"
    ]
    if len(task_indices) != 1 or task_indices[0] + 1 >= len(argv):
        raise SchemaValidationError("R1 v18 source task argument differs")
    task = json.loads(Path(argv[task_indices[0] + 1]).read_text(encoding="utf-8"))
    objects = task.get("scene", {}).get("objects", [])
    if len(objects) != 1:
        raise SchemaValidationError("R1 v18 source task object differs")
    quaternion = [float(value) for value in objects[0]["pose_world"][3:7]]
    x, y, z, w = quaternion
    object_y_world = (
        2.0 * (x * y - z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z + x * w),
    )
    centering_m = grasp_centering_object_y_mm * 1.0e-3
    grasp_centering = (
        None
        if centering_m == 0.0
        else {
            "translation_world_m": [
                centering_m * component for component in object_y_world
            ]
        }
    )
    payload = {
        **source,
        "version": VERSION,
        "source_failure_job": _binding(source_job),
        "source_failure_episode": _binding(source_episode),
        "repair_kind": "bounded_additional_contact_compression",
        "r1_nominal_additional_compression_sweep_mm": str(compression_mm),
        "r1_diagnostic_lift_time_dilation": lift_time_dilation,
        "r1_diagnostic_grasp_centering": grasp_centering,
        "r1_diagnostic_grasp_torque_bias_fraction": grasp_torque_bias_fraction,
        "r1_diagnostic_contact_compression_by_slot_mm": (
            None
            if compression_by_contact_mm is None
            else list(compression_by_contact_mm)
        ),
        "r1_diagnostic_maintain_vertical_scale": maintain_vertical_scale,
        "training_eligible": False,
        "formal_teacher_collection_authorized": False,
    }
    job_path = _write_json(case_root / "isaac_job.json", payload)
    candidate_id = str(jobs[0]["name"])
    return {
        "sample_index": sample_index,
        "candidate_id": candidate_id,
        "job_path": job_path,
        "replay_root": replay_root,
    }


def _run(job: dict[str, Any], *, isaac_python: str) -> int:
    episode = (
        job["replay_root"] / job["candidate_id"] / "isaac/evaluation_episodes.jsonl"
    )
    if episode.is_file():
        return 0
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "8",
        }
    )
    log = job["job_path"].with_suffix(".launcher.log")
    command = [
        isaac_python,
        str(WRAPPER),
        "--r1-batched-jobs",
        str(job["job_path"]),
        "--r1-batched-output-root",
        str(job["replay_root"]),
        "--no-formal-annotation",
    ]
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode == 0 and not episode.is_file():
        return 1
    return int(completed.returncode)


def _result(job: dict[str, Any]) -> dict[str, Any]:
    episode_path = (
        job["replay_root"] / job["candidate_id"] / "isaac/evaluation_episodes.jsonl"
    )
    raw_path = episode_path.with_name("evaluation_rollout.pt")
    records = [
        json.loads(line)
        for line in episode_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    metadata = raw.get("metadata", {})
    episode = records[0] if len(records) == 1 else {}
    metrics = episode.get("metrics", {})
    accepted = (
        len(records) == 1
        and episode.get("task_success") is True
        and episode.get("safety_failure") is False
        and int(episode.get("fallback_decision_count", -1)) == 0
        and float(metrics.get("hard_collision", -1.0)) == 0.0
        and float(metrics.get("object_dropped", -1.0)) == 0.0
        and float(metrics.get("qp_infeasible_terminal", -1.0)) == 0.0
        and float(metrics.get("timeout", -1.0)) == 0.0
        and metadata.get("pi_l_actor_command_applied") is False
        and metadata.get("r1_scene_support_from_task") is True
    )
    return {
        "sample_index": job["sample_index"],
        "candidate_id": job["candidate_id"],
        "status": "accepted" if accepted else "rejected",
        "episode": episode if len(records) == 1 else None,
        "episodes": _binding(episode_path),
        "raw_rollout": _binding(raw_path),
        "pi_l_actor_command_applied": metadata.get("pi_l_actor_command_applied"),
        "r1_scene_support_from_task": metadata.get("r1_scene_support_from_task"),
        "applied_inward_lead_m": metadata.get(
            "actuator_aware_nominal_preload_applied_inward_lead_m"
        ),
    }


def main() -> int:
    arguments = _parser().parse_args()
    compression_by_contact_mm = None
    if arguments.compression_by_contact_mm is not None:
        compression_by_contact_mm = tuple(
            float(value)
            for value in arguments.compression_by_contact_mm.split(",")
            if value.strip()
        )
    if (
        os.environ.get("PYTHONHASHSEED") != "0"
        or not 5.0 <= arguments.additional_compression_mm <= 80.0
        or not 0.0 <= arguments.grasp_centering_object_y_mm <= 30.0
        or (
            compression_by_contact_mm is not None
            and (
                len(compression_by_contact_mm) != 2
                or any(not 0.0 <= value <= 80.0 for value in compression_by_contact_mm)
            )
        )
        or (
            arguments.grasp_torque_bias_fraction is not None
            and not 0.0 < arguments.grasp_torque_bias_fraction <= 0.20
        )
        or (
            arguments.lift_time_dilation is not None
            and not 1.0 < arguments.lift_time_dilation <= 10.0
        )
        or (
            arguments.maintain_vertical_scale is not None
            and not 0.10 <= arguments.maintain_vertical_scale < 1.0
        )
        or hash_file(PROTECTED_ROLLOUT) != PROTECTED_ROLLOUT_SHA256
        or hash_file(PROTECTED_CHECKPOINT) != PROTECTED_CHECKPOINT_SHA256
    ):
        raise SchemaValidationError("R1 v18 execution contract differs")
    output_root = Path(arguments.output_root).resolve()
    selected = SAMPLES if arguments.sample_index is None else (arguments.sample_index,)
    jobs = [
        _prepare_job(
            sample_index=sample,
            compression_mm=float(arguments.additional_compression_mm),
            lift_time_dilation=arguments.lift_time_dilation,
            grasp_centering_object_y_mm=(arguments.grasp_centering_object_y_mm),
            grasp_torque_bias_fraction=arguments.grasp_torque_bias_fraction,
            compression_by_contact_mm=compression_by_contact_mm,
            maintain_vertical_scale=arguments.maintain_vertical_scale,
            output_root=output_root,
        )
        for sample in selected
    ]
    preparation = {
        "result_version": VERSION,
        "status": "accepted",
        "structural_hash": STRUCTURAL_HASH,
        "sample_indices": list(selected),
        "additional_compression_mm": float(arguments.additional_compression_mm),
        "lift_time_dilation": arguments.lift_time_dilation,
        "grasp_centering_object_y_mm": arguments.grasp_centering_object_y_mm,
        "grasp_torque_bias_fraction": arguments.grasp_torque_bias_fraction,
        "compression_by_contact_mm": (
            None
            if compression_by_contact_mm is None
            else list(compression_by_contact_mm)
        ),
        "maintain_vertical_scale": arguments.maintain_vertical_scale,
        "training_eligible": False,
        "formal_teacher_collection_authorized": False,
        "protected_c3_rollout_sha256": PROTECTED_ROLLOUT_SHA256,
        "protected_c3_checkpoint_sha256": PROTECTED_CHECKPOINT_SHA256,
    }
    _write_json(output_root / "preparation_result.json", preparation)
    if arguments.prepare_only:
        print(json.dumps(preparation, sort_keys=True), flush=True)
        return 0
    for job in jobs:
        returncode = _run(job, isaac_python=arguments.isaac_python)
        if returncode != 0:
            raise RuntimeError("R1 v18 Isaac subprocess failed")
    results = [_result(job) for job in jobs]
    payload = {
        **preparation,
        "status": (
            "accepted"
            if all(result["status"] == "accepted" for result in results)
            else "rejected"
        ),
        "success_count": sum(result["status"] == "accepted" for result in results),
        "results": results,
    }
    _write_json(output_root / "result.json", payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0 if payload["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
