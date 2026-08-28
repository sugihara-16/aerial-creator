#!/usr/bin/env python3
from __future__ import annotations

"""Derive v5 R1 cases from hash-verified v4 controller-free evidence."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.training.order9_c3_nominal_trajectory import (  # noqa: E402
    validate_order9_c3_nominal_trajectory_set_bytes,
)
from amsrr.training.order9_r1_calibration_runner import (  # noqa: E402
    enumerate_order9_r1_calibration_level_cases,
)
from amsrr.training.order9_r1_isaac_calibration import (  # noqa: E402
    load_order9_r1_isaac_case,
)
from amsrr.training.order9_r1_nominal_calibration_v5 import (  # noqa: E402
    load_order9_r1_nominal_calibration_v5_contract,
    write_order9_r1_nominal_v5_case_contract,
)
from amsrr.training.order9_r1_nominal_retime import (  # noqa: E402
    ORDER9_R1_NOMINAL_RETIME_VERSION,
    retime_order9_r1_nominal_artifact,
)
from amsrr.utils.hashing import hash_file  # noqa: E402


V4_RESULT = Path(
    "artifacts/p4_full/order9/r1_teacher/calibration_v6_early_tilt_teacher/"
    "calibration_result_v4.json"
)
V4_RESULT_SHA256 = "881b405c5e739e9939b7885f4f9e22a21c1e747734b65b84cfba77a58aeec0e7"
V4_CASE_ROOT = Path(
    "artifacts/p4_full/order9/r1_teacher/calibration_v6_early_tilt_teacher/"
    "selection/r1_l1_10mm_5deg"
)
CORRECTED_CANDIDATE_ID = (
    "r1_l1_10mm_5deg__train__train-000003-3b95da871f01__lattice_00"
)
CORRECTED_SOURCE = Path(
    "artifacts/p4_full/order9/r1_teacher/diagnostics/"
    "r1_l1_phase_scale_a050_c050_post010/"
) / CORRECTED_CANDIDATE_ID


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maximum-process-count", type=int, default=24)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.maximum_process_count < 1:
        raise ValueError("maximum process count must be positive")
    repository = REPOSITORY.resolve()
    protocol, _approval = load_order9_r1_nominal_calibration_v5_contract(repository)
    if hash_file(repository / V4_RESULT) != V4_RESULT_SHA256:
        raise ValueError("R1 nominal v4 source result bytes changed")
    base_path = repository / protocol["base_numeric_protocol"]["path"]
    cases = enumerate_order9_r1_calibration_level_cases(
        protocol_path=base_path,
        repository_root=repository,
        level_id="r1_l1_10mm_5deg",
        split="train",
    )
    destination_root = repository / protocol["output_root"] / "selection" / (
        "r1_l1_10mm_5deg"
    )
    destination_root.mkdir(parents=True, exist_ok=True)
    arguments = []
    for case in cases:
        source = (
            repository / CORRECTED_SOURCE
            if case.candidate_id == CORRECTED_CANDIDATE_ID
            else repository / V4_CASE_ROOT / case.candidate_id
        )
        arguments.append(
            (
                str(repository),
                str(source),
                str(destination_root / case.candidate_id),
                dict(protocol["phase_time_scales"]),
                case.candidate_id == CORRECTED_CANDIDATE_ID,
            )
        )
    with ProcessPoolExecutor(max_workers=args.maximum_process_count) as executor:
        for completed, result in enumerate(executor.map(_derive_case, arguments), 1):
            if completed % 25 == 0 or completed == len(arguments):
                print(
                    "ORDER9_R1_V5_DERIVATION_PROGRESS="
                    + json.dumps(
                        {"completed": completed, "total": len(arguments)},
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if result is not True:
                raise AssertionError("R1 nominal v5 case derivation failed")
    print(
        "ORDER9_R1_V5_DERIVATION_COMPLETE="
        + json.dumps(
            {
                "case_count": len(cases),
                "source_controller_free_screen_success_count": len(cases),
                "corrected_candidate_count": 1,
                "isaac_invoked": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _derive_case(arguments) -> bool:
    repository_text, source_text, destination_text, phase_scales, already_retimed = (
        arguments
    )
    repository = Path(repository_text)
    source = Path(source_text)
    destination = Path(destination_text)
    if (destination / "case_manifest.json").is_file():
        materialized = load_order9_r1_isaac_case(
            destination / "case_manifest.json", repository
        )
        write_order9_r1_nominal_v5_case_contract(
            materialized,
            repository_root=repository,
        )
        return True
    if destination.exists():
        raise FileExistsError(destination)
    source_case_path = source / "case_manifest.json"
    source_case = json.loads(source_case_path.read_text(encoding="utf-8"))
    for key in ("task_spec", "nominal_set", "nominal_artifact", "fast_screen_evidence"):
        binding = source_case[key]
        path = repository / binding["path"]
        if not path.is_file() or hash_file(path) != binding["sha256"]:
            raise ValueError(f"R1 source case binding changed: {key}")
    source_set_path = repository / source_case["nominal_set"]["path"]
    validate_order9_c3_nominal_trajectory_set_bytes(
        source_set_path,
        repository_root=repository,
        expected_sha256=source_case["nominal_set"]["sha256"],
    )
    screen = json.loads((source / "fast_screen.json").read_text(encoding="utf-8"))
    if (
        screen.get("accepted") is not True
        or screen.get("eligible_for_full_control_test") is not True
        or screen.get("isaac_invoked") is not False
        or screen.get("controller_layers_invoked") is not False
    ):
        raise ValueError("R1 source controller-free screen is not accepted")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(dir=destination.parent, prefix=f".{destination.name}.")
    )
    try:
        shutil.copytree(source / "nominal_set", temporary / "nominal_set")
        shutil.copy2(source / "task_spec.json", temporary / "task_spec.json")
        shutil.copy2(source / "fast_screen.json", temporary / "fast_screen.json")
        candidate_id = str(source_case["candidate_id"])
        artifact_path = (
            temporary / "nominal_set" / "buckets" / candidate_id / "manifest.json"
        )
        if already_retimed:
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            admission = artifact.get("selection_evidence", {}).get(
                "r1_uniform_nominal_retime"
            )
            if (
                not isinstance(admission, dict)
                or admission.get("admission_version")
                != ORDER9_R1_NOMINAL_RETIME_VERSION
                or admission.get("phase_time_scales") != phase_scales
            ):
                raise ValueError("corrected R1 source lacks admitted retiming")
            source_set = json.loads(
                (source / "nominal_set" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            retime_admission = source_set["metadata"]["nominal_retime_admission"]
        else:
            retime_admission = retime_order9_r1_nominal_artifact(
                artifact_path,
                phase_time_scales=phase_scales,
                joint_rate_limit_rad_s=(
                    float(screen["maximum_joint_rate_rad_s"])
                    + float(screen["minimum_joint_rate_margin_rad_s"])
                ),
            ).to_dict()

        collision_path = temporary / "nominal_set" / "collision_validation.json"
        collision = json.loads(collision_path.read_text(encoding="utf-8"))
        collision["version"] = (
            "order9_r1_identical_geometry_retime_collision_admission_v1"
        )
        collision["records"][0]["fast_screen_path"] = _portable(
            destination / "fast_screen.json", repository
        )
        collision["records"][0]["nominal_retime_admission"] = retime_admission
        collision_path.write_text(
            json.dumps(collision, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        set_path = temporary / "nominal_set" / "manifest.json"
        nominal_set = json.loads(set_path.read_text(encoding="utf-8"))
        nominal_set["bucket_manifest_path"] = _portable(
            destination / "nominal_set" / "bucket_manifest.json", repository
        )
        nominal_set["entries"][0]["artifact_sha256"] = hash_file(artifact_path)
        nominal_set["metadata"].update(
            {
                "nominal_phase_time_scales": phase_scales,
                "nominal_retime_admission": retime_admission,
                "controller_free_screen_reused": True,
                "source_case_manifest_path": _portable(source_case_path, repository),
                "source_case_manifest_sha256": hash_file(source_case_path),
                "source_v4_formal_result_sha256": V4_RESULT_SHA256,
            }
        )
        set_path.write_text(
            json.dumps(nominal_set, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        case = source_case
        local = {
            "task_spec": temporary / "task_spec.json",
            "nominal_set": set_path,
            "nominal_artifact": artifact_path,
            "fast_screen_evidence": temporary / "fast_screen.json",
        }
        final = {
            "task_spec": destination / "task_spec.json",
            "nominal_set": destination / "nominal_set" / "manifest.json",
            "nominal_artifact": (
                destination
                / "nominal_set"
                / "buckets"
                / candidate_id
                / "manifest.json"
            ),
            "fast_screen_evidence": destination / "fast_screen.json",
        }
        for key in local:
            case[key]["path"] = _portable(final[key], repository)
            case[key]["sha256"] = hash_file(local[key])
        case["reset_bank_path"] = _portable(destination / "reset_bank.pt", repository)
        for binding in case["r1_implementations"]:
            path = repository / binding["path"]
            binding["sha256"] = hash_file(path)
        case_path = temporary / "case_manifest.json"
        case_path.write_text(
            json.dumps(case, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    materialized = load_order9_r1_isaac_case(
        destination / "case_manifest.json", repository
    )
    write_order9_r1_nominal_v5_case_contract(
        materialized,
        repository_root=repository,
    )
    return True


def _portable(path: Path, repository: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository.resolve()))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
