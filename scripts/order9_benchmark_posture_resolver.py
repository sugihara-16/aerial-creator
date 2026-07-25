#!/usr/bin/env python3
from __future__ import annotations

"""Prepare and benchmark the Order 9 deterministic posture resolver.

Prepare the hash-bound 12-case fixture once on the development PC, then copy
the fixture and repository to the target SBC and rerun without
``--prepare-fixture``.  Timed sections contain only raw-trajectory resolution;
the offline articulated teacher is excluded.
"""

import argparse
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.irg.envelope_extractor import (  # noqa: E402
    InteractionEnvelopeExtractor,
)
from amsrr.irg.irg_builder import IRGBuilder  # noqa: E402
from amsrr.policies.high_level_policy_base import (  # noqa: E402
    HighLevelPolicyContext,
)
from amsrr.robot_model.physical_model_builder import (  # noqa: E402
    build_physical_model_from_config,
)
from amsrr.schemas.common import SchemaValidationError  # noqa: E402
from amsrr.schemas.contact_candidates import ContactCandidateSet  # noqa: E402
from amsrr.schemas.morphology import MorphologyGraph  # noqa: E402
from amsrr.schemas.physical_model import PhysicalModel  # noqa: E402
from amsrr.schemas.policies import ContactWrenchTrajectory  # noqa: E402
from amsrr.schemas.task_spec import TaskSpec  # noqa: E402
from amsrr.training.order9_posture_resolver import (  # noqa: E402
    ORDER9_POSTURE_TRAJECTORY_RESOLVER_VERSION,
    Order9PostureResolverConfig,
    Order9PostureTrajectoryResolver,
)
from amsrr.utils.hashing import hash_file, stable_hash  # noqa: E402


LEGACY_FIXTURE_VERSION = "order9_posture_resolver_benchmark_fixture_v1"
RELOCATABLE_FIXTURE_VERSION = "order9_posture_resolver_benchmark_fixture_v2"
QUANTIZED_FIXTURE_VERSION = "order9_posture_resolver_benchmark_fixture_v3"
FIXTURE_VERSION = "order9_posture_resolver_benchmark_fixture_v4"
REPORT_VERSION = "order9_posture_resolver_benchmark_report_v6_native"
TRAJECTORY_HASH_DECIMAL_PLACES = 9
TRAJECTORY_ABSOLUTE_TOLERANCE = 1.0e-8
DEFAULT_CONFIG = "configs/training/order9_c3a_curation_pilot.yaml"
DEFAULT_FIXTURE = (
    "artifacts/p4_full/order9/posture_resolver/"
    "c3a_12_case_fixture.json"
)
DEFAULT_OUTPUT = (
    "artifacts/p4_full/order9/posture_resolver/"
    "pc_benchmark.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--fixture", default=DEFAULT_FIXTURE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--prepare-fixture", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-rate-hz", type=float, default=20.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    config_path = _resolve(args.config)
    fixture_path = _resolve(args.fixture)
    output_path = _resolve(args.output)
    if args.prepare_fixture:
        started = perf_counter()
        fixture = _prepare_fixture(
            config_path=config_path,
            output_rate_hz=float(args.output_rate_hz),
        )
        _write_json(fixture_path, fixture)
        print(
            "ORDER9_POSTURE_FIXTURE="
            + json.dumps(
                {
                    "path": _portable(fixture_path),
                    "sha256": hash_file(fixture_path),
                    "case_count": len(fixture["cases"]),
                    "preparation_wall_time_s": perf_counter() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if not fixture_path.is_file():
        raise FileNotFoundError(
            f"{fixture_path}; run once with --prepare-fixture on the PC"
        )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    report = _benchmark(
        fixture,
        fixture_path=fixture_path,
        repeats=int(args.repeats),
    )
    _write_json(output_path, report)
    print(
        "ORDER9_POSTURE_BENCHMARK="
        + json.dumps(
            {
                "path": _portable(output_path),
                "sha256": hash_file(output_path),
                "case_count": report["case_count"],
                "mean_cold_s": report[
                    "aggregate"
                ]["mean_cold_s"],
                "maximum_cold_s": report[
                    "aggregate"
                ]["maximum_cold_s"],
                "mean_warm_or_single_s": report[
                    "aggregate"
                ]["mean_warm_or_single_s"],
                "maximum_warm_or_single_s": report[
                    "aggregate"
                ]["maximum_warm_or_single_s"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _prepare_fixture(
    *,
    config_path: Path,
    output_rate_hz: float,
) -> dict[str, object]:
    # Fixture preparation is PC-only and intentionally imported lazily so the
    # target SBC benchmark needs neither the curation viewer nor PyYAML.
    from order9_prepare_c3a_curation_pilot import (
        _case_graph,
        _load_config,
        _pilot_task,
    )
    from amsrr.schemas.order3 import Order3MorphologyPoolManifest
    from amsrr.training.order9_c3_teacher import (
        Order9C3TeacherConfig,
        build_order9_c3_articulated_teacher,
    )

    config = _load_config(config_path)
    sources = config["sources"]
    physical_path = _resolve(sources["robot_model_config"])
    pool_path = _resolve(sources["morphology_pool"])
    task_path = _resolve(sources["canonical_order8_task_bucket"])
    physical = build_physical_model_from_config(physical_path)
    pool = Order3MorphologyPoolManifest.from_json(
        pool_path.read_text(encoding="utf-8")
    )
    if pool.physical_model_hash != physical.stable_hash():
        raise SchemaValidationError(
            "posture benchmark morphology-pool PhysicalModel hash is stale"
        )
    task = _pilot_task(task_path)
    records: list[dict[str, object]] = []
    for index, case in enumerate(config["cases"], start=1):
        structural, _source = _case_graph(
            case,
            config=config,
            pool=pool,
            physical_model=physical,
        )
        bundle = build_order9_c3_articulated_teacher(
            task_spec=task,
            structural_target=structural,
            physical_model=physical,
            config=Order9C3TeacherConfig(
                maximum_surface_pair_attempts=1,
                preferred_surface_port_ids=tuple(
                    int(value) for value in case["surface_port_ids"]
                ),
                preferred_candidate_group_id=str(
                    case["candidate_group_id"]
                ),
            ),
        )
        first_posture = bundle.trajectory.knots[0].posture_target
        if (
            first_posture is None
            or first_posture.joint_pos_target is None
        ):
            raise SchemaValidationError(
                "prepared posture fixture lacks its initial joint state"
            )
        records.append(
            {
                "case_id": str(case["case_id"]),
                "module_count": int(case["module_count"]),
                "topology": str(case["topology"]),
                "structural_hash": str(case["structural_hash"]),
                "morphology_graph": (
                    bundle.design_output.target_morphology.to_dict()
                ),
                "contact_candidate_set": (
                    bundle.contact_candidate_set.to_dict()
                ),
                "raw_pi_h_trajectory": (
                    bundle.trajectory_plan.raw_trajectory.to_dict()
                ),
                "initial_joint_positions_rad": dict(
                    first_posture.joint_pos_target
                ),
                "expected_raw_trajectory_hash": stable_hash(
                    bundle.trajectory_plan.raw_trajectory.to_dict()
                ),
                "expected_resolved_trajectory_hash": stable_hash(
                    bundle.trajectory.to_dict()
                ),
                "expected_resolved_trajectory": (
                    bundle.trajectory.to_dict()
                ),
                "expected_resolved_trajectory_quantized_hash": (
                    _quantized_stable_hash(bundle.trajectory.to_dict())
                ),
            }
        )
        print(
            f"ORDER9_POSTURE_FIXTURE_CASE={index}/"
            f"{len(config['cases'])}:{case['case_id']}",
            flush=True,
        )
    return {
        "fixture_version": FIXTURE_VERSION,
        "resolver_version": ORDER9_POSTURE_TRAJECTORY_RESOLVER_VERSION,
        "resolver_config": {
            "output_rate_hz": output_rate_hz,
            "joint_velocity_fraction": 0.90,
        },
        "source_config_path": _portable(config_path),
        "source_config_sha256": hash_file(config_path),
        "source_pool_path": _portable(pool_path),
        "source_pool_sha256": hash_file(pool_path),
        "robot_model_config_path": _portable(physical_path),
        "robot_model_config_sha256": hash_file(physical_path),
        "physical_model_hash": physical.stable_hash(),
        "physical_model_content_hash": _physical_model_content_hash(physical),
        "task_spec": task.to_dict(),
        "task_spec_hash": task.stable_hash(),
        "case_count": len(records),
        "cases": records,
    }


def _benchmark(
    fixture: dict[str, object],
    *,
    fixture_path: Path,
    repeats: int,
) -> dict[str, object]:
    fixture_version = fixture.get("fixture_version")
    if fixture_version not in {
        LEGACY_FIXTURE_VERSION,
        RELOCATABLE_FIXTURE_VERSION,
        QUANTIZED_FIXTURE_VERSION,
        FIXTURE_VERSION,
    }:
        raise SchemaValidationError("posture benchmark fixture version differs")
    physical_path = _resolve(str(fixture["robot_model_config_path"]))
    if hash_file(physical_path) != fixture["robot_model_config_sha256"]:
        raise SchemaValidationError(
            "posture benchmark robot-model config hash differs"
        )
    physical = build_physical_model_from_config(physical_path)
    physical_model_hash = physical.stable_hash()
    physical_model_content_hash = _physical_model_content_hash(physical)
    if fixture_version in {
        RELOCATABLE_FIXTURE_VERSION,
        QUANTIZED_FIXTURE_VERSION,
        FIXTURE_VERSION,
    }:
        expected_content_hash = fixture.get("physical_model_content_hash")
        if not isinstance(expected_content_hash, str) or not expected_content_hash:
            raise SchemaValidationError(
                "posture benchmark fixture lacks its PhysicalModel content hash"
            )
        physical_identity_matches = (
            physical_model_content_hash == expected_content_hash
        )
    else:
        # The v1 fixture encoded absolute source paths in PhysicalModel hash.
        # It therefore remains valid only at its original project root.
        physical_identity_matches = (
            physical_model_hash == fixture["physical_model_hash"]
        )
    if not physical_identity_matches:
        raise SchemaValidationError(
            "posture benchmark fixture PhysicalModel hash differs"
        )
    task = TaskSpec.from_dict(fixture["task_spec"])
    if task.stable_hash() != fixture["task_spec_hash"]:
        raise SchemaValidationError(
            "posture benchmark fixture TaskSpec hash differs"
        )
    built = IRGBuilder().build_with_scene_graph(task)
    envelope = InteractionEnvelopeExtractor().extract(built.irg)
    config_payload = fixture["resolver_config"]
    resolver_config = Order9PostureResolverConfig(
        output_rate_hz=float(config_payload["output_rate_hz"]),
        joint_velocity_fraction=float(
            config_payload["joint_velocity_fraction"]
        ),
    )
    case_reports = []
    for index, case in enumerate(fixture["cases"], start=1):
        morphology = MorphologyGraph.from_dict(case["morphology_graph"])
        candidates = ContactCandidateSet.from_dict(
            case["contact_candidate_set"]
        )
        raw = ContactWrenchTrajectory.from_dict(
            case["raw_pi_h_trajectory"]
        )
        if (
            stable_hash(raw.to_dict())
            != case["expected_raw_trajectory_hash"]
        ):
            raise SchemaValidationError(
                f"{case['case_id']} raw trajectory hash differs"
            )
        context = HighLevelPolicyContext(
            built.irg,
            envelope,
            morphology,
            candidates,
        )
        resolver = Order9PostureTrajectoryResolver(
            physical,
            config=resolver_config,
        )
        durations = []
        evidences = []
        result_hashes = []
        quantized_result_hashes = []
        last = None
        for _repeat in range(repeats):
            started = perf_counter()
            last = resolver.resolve(
                context=context,
                raw_trajectory=raw,
                initial_joint_positions_rad=case[
                    "initial_joint_positions_rad"
                ],
            )
            durations.append(perf_counter() - started)
            evidences.append(last.evidence)
            result_hashes.append(
                last.evidence.resolved_trajectory_hash
            )
            quantized_result_hashes.append(
                _quantized_stable_hash(last.trajectory.to_dict())
            )
        if len(set(result_hashes)) != 1:
            raise RuntimeError(
                f"{case['case_id']} resolver output is nondeterministic"
            )
        exact_pc_hash_match = (
            result_hashes[0] == case["expected_resolved_trajectory_hash"]
        )
        quantized_pc_hash_match = False
        tolerance_pc_match = False
        maximum_pc_float_difference = None
        if fixture_version == QUANTIZED_FIXTURE_VERSION:
            expected_quantized_hash = case.get(
                "expected_resolved_trajectory_quantized_hash"
            )
            if (
                not isinstance(expected_quantized_hash, str)
                or not expected_quantized_hash
            ):
                raise SchemaValidationError(
                    f"{case['case_id']} lacks its quantized trajectory hash"
                )
            quantized_pc_hash_match = (
                len(set(quantized_result_hashes)) == 1
                and quantized_result_hashes[0] == expected_quantized_hash
            )
        elif fixture_version == FIXTURE_VERSION:
            expected_trajectory = case.get(
                "expected_resolved_trajectory"
            )
            if not isinstance(expected_trajectory, dict):
                raise SchemaValidationError(
                    f"{case['case_id']} lacks its expected trajectory"
                )
            # The fixture binds the pre-promotion resolver identity in the
            # diagnostic label.  Native promotion deliberately advances that
            # identity; numerical compatibility is evaluated on the same
            # trajectory contract with only that provenance label normalized.
            actual_trajectory = last.trajectory.to_dict()
            expected_trajectory = {
                **expected_trajectory,
                "derived_mode_label": actual_trajectory.get(
                    "derived_mode_label"
                ),
            }
            maximum_pc_float_difference = (
                _maximum_float_difference(
                    expected_trajectory,
                    actual_trajectory,
                )
            )
            tolerance_pc_match = (
                maximum_pc_float_difference
                <= TRAJECTORY_ABSOLUTE_TOLERANCE
            )
        if (
            not exact_pc_hash_match
            and not quantized_pc_hash_match
            and not tolerance_pc_match
        ):
            raise RuntimeError(
                f"{case['case_id']} resolved trajectory differs from PC fixture"
            )
        cold_evidence = evidences[0]
        warm_evidence = evidences[-1]
        warm_or_single = durations[1:] if len(durations) > 1 else durations
        case_reports.append(
            {
                "case_id": case["case_id"],
                "module_count": case["module_count"],
                "topology": case["topology"],
                "global_dock_joint_count": len(
                    case["initial_joint_positions_rad"]
                ),
                "durations_s": durations,
                "cold_s": durations[0],
                "cold_resolver_reported_wall_time_s": (
                    cold_evidence.solve_wall_time_s
                ),
                "mean_warm_or_single_s": statistics.fmean(
                    warm_or_single
                ),
                "maximum_warm_or_single_s": max(warm_or_single),
                "raw_trajectory_hash": (
                    last.evidence.raw_trajectory_hash
                ),
                "resolved_trajectory_hash": (
                    last.evidence.resolved_trajectory_hash
                ),
                "resolved_trajectory_quantized_hash": (
                    quantized_result_hashes[-1]
                ),
                "exact_pc_trajectory_hash_match": exact_pc_hash_match,
                "quantized_pc_trajectory_hash_match": (
                    quantized_pc_hash_match
                ),
                "tolerance_pc_trajectory_match": tolerance_pc_match,
                "maximum_pc_trajectory_float_difference": (
                    maximum_pc_float_difference
                ),
                "resolved_knot_count": (
                    last.evidence.resolved_knot_count
                ),
                "cold_solver_iterations": list(
                    cold_evidence.solver_iterations
                ),
                "cold_solver_cache_hits": list(
                    cold_evidence.solver_cache_hits
                ),
                "warm_solver_iterations": list(
                    warm_evidence.solver_iterations
                ),
                "warm_solver_cache_hits": list(
                    warm_evidence.solver_cache_hits
                ),
                "maximum_anchor_position_error_m": (
                    last.evidence.maximum_anchor_position_error_m
                ),
                "maximum_anchor_attitude_error_rad": (
                    last.evidence.maximum_anchor_attitude_error_rad
                ),
                "maximum_joint_rate_rad_s": (
                    last.evidence.maximum_joint_rate_rad_s
                ),
                "minimum_joint_rate_margin_rad_s": (
                    last.evidence.minimum_joint_rate_margin_rad_s
                ),
            }
        )
        print(
            f"ORDER9_POSTURE_BENCHMARK_CASE={index}/"
            f"{len(fixture['cases'])}:{case['case_id']}:"
            f"cold={durations[0]:.6f}s:"
            f"warm={statistics.fmean(warm_or_single):.6f}s",
            flush=True,
        )
    cold_values = [
        float(case["cold_s"])
        for case in case_reports
    ]
    warm_values = [
        float(case["mean_warm_or_single_s"])
        for case in case_reports
    ]
    return {
        "report_version": REPORT_VERSION,
        "fixture_path": _portable(fixture_path),
        "fixture_sha256": hash_file(fixture_path),
        "fixture_version": fixture["fixture_version"],
        "resolver_version": fixture["resolver_version"],
        "resolver_config": fixture["resolver_config"],
        "trajectory_hash_decimal_places": (
            TRAJECTORY_HASH_DECIMAL_PLACES
        ),
        "trajectory_absolute_tolerance": (
            TRAJECTORY_ABSOLUTE_TOLERANCE
        ),
        "physical_model_identity": {
            "expected_hash": fixture["physical_model_hash"],
            "actual_hash": physical_model_hash,
            "content_hash": physical_model_content_hash,
            "relocated_project_root": (
                physical_model_hash != fixture["physical_model_hash"]
            ),
        },
        "repeats": repeats,
        "system": _system_info(),
        "case_count": len(case_reports),
        "cases": case_reports,
        "aggregate": {
            "mean_cold_s": statistics.fmean(cold_values),
            "median_cold_s": statistics.median(cold_values),
            "maximum_cold_s": max(cold_values),
            "minimum_cold_s": min(cold_values),
            "mean_warm_or_single_s": statistics.fmean(warm_values),
            "median_warm_or_single_s": statistics.median(warm_values),
            "maximum_warm_or_single_s": max(warm_values),
            "minimum_warm_or_single_s": min(warm_values),
        },
    }


def _system_info() -> dict[str, object]:
    try:
        import numpy
    except ModuleNotFoundError:
        numpy_version = None
    else:
        numpy_version = numpy.__version__
    cpu_model = None
    total_memory_mib = None
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        for line in cpuinfo.splitlines():
            if line.lower().startswith(("model name", "hardware")):
                cpu_model = line.split(":", 1)[-1].strip()
                break
    except OSError:
        pass
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        total_kib = next(
            int(line.split()[1])
            for line in meminfo.splitlines()
            if line.startswith("MemTotal:")
        )
        total_memory_mib = total_kib / 1024.0
    except (OSError, StopIteration, ValueError):
        pass
    try:
        import resource

        peak_rss = float(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        )
        # Linux reports KiB; macOS reports bytes.
        process_peak_rss_mib = (
            peak_rss / (1024.0 * 1024.0)
            if sys.platform == "darwin"
            else peak_rss / 1024.0
        )
    except (ImportError, ValueError):
        process_peak_rss_mib = None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": cpu_model,
        "python_version": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "total_memory_mib": total_memory_mib,
        "process_peak_rss_mib": process_peak_rss_mib,
        "numpy_version": numpy_version,
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def _physical_model_content_hash(physical_model: PhysicalModel) -> str:
    """Hash a PhysicalModel without installation-root-dependent paths."""

    payload = physical_model.to_dict()
    metadata = dict(payload["metadata"])
    urdf_hash = metadata.get("urdf_hash")
    actuator_hash = metadata.get("joint_actuator_model_hash")
    if not isinstance(urdf_hash, str) or not urdf_hash:
        raise SchemaValidationError("PhysicalModel lacks its URDF content hash")
    if not isinstance(actuator_hash, str) or not actuator_hash:
        raise SchemaValidationError(
            "PhysicalModel lacks its joint-actuator content hash"
        )
    payload["urdf_path"] = f"content-sha256:{urdf_hash}"
    if "joint_actuator_model_path" in metadata:
        metadata["joint_actuator_model_path"] = (
            f"content-sha256:{actuator_hash}"
        )
    payload["metadata"] = metadata
    return stable_hash(payload)


def _quantized_stable_hash(value: object) -> str:
    return stable_hash(_quantize_floats(value))


def _quantize_floats(value: object) -> object:
    if isinstance(value, float):
        quantized = round(value, TRAJECTORY_HASH_DECIMAL_PLACES)
        return 0.0 if quantized == 0.0 else quantized
    if isinstance(value, dict):
        return {
            key: _quantize_floats(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_quantize_floats(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_quantize_floats(item) for item in value)
    return value


def _maximum_float_difference(
    expected: object,
    actual: object,
    *,
    path: str = "$",
) -> float:
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected != actual or type(expected) is not type(actual):
            raise SchemaValidationError(
                f"posture benchmark trajectory differs at {path}"
            )
        return 0.0
    if isinstance(expected, (int, float)) and isinstance(
        actual, (int, float)
    ):
        if isinstance(expected, int) and isinstance(actual, int):
            if expected != actual:
                raise SchemaValidationError(
                    f"posture benchmark trajectory differs at {path}"
                )
            return 0.0
        difference = abs(float(expected) - float(actual))
        if not math.isfinite(difference):
            raise SchemaValidationError(
                f"posture benchmark trajectory is non-finite at {path}"
            )
        return difference
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            raise SchemaValidationError(
                f"posture benchmark trajectory keys differ at {path}"
            )
        return max(
            (
                _maximum_float_difference(
                    expected[key],
                    actual[key],
                    path=f"{path}.{key}",
                )
                for key in expected
            ),
            default=0.0,
        )
    if isinstance(expected, (list, tuple)) and isinstance(
        actual, (list, tuple)
    ):
        if len(expected) != len(actual):
            raise SchemaValidationError(
                f"posture benchmark trajectory length differs at {path}"
            )
        return max(
            (
                _maximum_float_difference(
                    expected_item,
                    actual_item,
                    path=f"{path}[{index}]",
                )
                for index, (expected_item, actual_item) in enumerate(
                    zip(expected, actual, strict=True)
                )
            ),
            default=0.0,
        )
    if type(expected) is not type(actual) or expected != actual:
        raise SchemaValidationError(
            f"posture benchmark trajectory differs at {path}"
        )
    return 0.0


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(resolved)


if __name__ == "__main__":
    raise SystemExit(main())
