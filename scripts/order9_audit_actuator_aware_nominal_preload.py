#!/usr/bin/env python3
from __future__ import annotations

"""Audit actuator-aware nominal preload across a hash-bound C3 bucket set."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.order9_actuator_aware_nominal_preload import (
    Order9ActuatorAwareNominalPreloadConfig,
    solve_order9_actuator_aware_nominal_preload,
)
from amsrr.training.order9_c3_nominal_trajectory import (
    load_order9_c3_accepted_nominal_bundle,
)
from amsrr.training.order9_c3_teacher import build_order9_c3_posture_collision_object
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_posture_resolver import (
    Order9PostureResolverConfig,
    Order9PostureTrajectoryResolver,
)
from amsrr.training.order9_rollout_buckets import (
    load_order9_pi_l_rollout_bucket_manifest,
    validate_order9_pi_l_rollout_bucket_bytes,
)
from amsrr.training.order9_virtual_contact_compression import (
    Order9VirtualContactCompressionConfig,
    limit_order9_virtual_contact_compression_by_collision,
    solve_order9_virtual_contact_compression_for_achieved_lead,
)
from amsrr.simulation.order9_object_task_runtime import Order9ObjectTaskPhase
from amsrr.utils.hashing import hash_file


AUDIT_VERSION = (
    "order9_actuator_aware_nominal_preload_bucket_audit_v3_achieved_displacement"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/order9_learning_curriculum.yaml")
    parser.add_argument("--bucket-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--module-count", type=int, action="append")
    parser.add_argument("--skip-collision", action="store_true")
    args = parser.parse_args()

    config_path = _resolve(args.config)
    manifest_path = _resolve(args.bucket_manifest)
    output = _resolve(args.output)
    if output.exists():
        raise FileExistsError(output)
    learning = load_order9_learning_config(config_path)
    runtime = learning.production_runtime
    physical = build_physical_model_from_config(
        _resolve(runtime.robot_model_config_path)
    )
    validate_order9_pi_l_rollout_bucket_bytes(
        manifest_path, repository_root=REPOSITORY_ROOT
    )
    manifest = load_order9_pi_l_rollout_bucket_manifest(manifest_path)
    selected_counts = None if args.module_count is None else set(args.module_count)
    buckets = tuple(
        bucket
        for bucket in manifest.buckets
        if selected_counts is None or bucket.module_count in selected_counts
    )
    if not buckets:
        raise ValueError("actuator-aware preload audit selected no buckets")

    started = time.perf_counter()
    records: list[dict[str, object]] = []
    for index, bucket in enumerate(buckets, start=1):
        print(
            f"[{index:02d}/{len(buckets):02d}] module={bucket.module_count} "
            f"bucket={bucket.bucket_id}",
            flush=True,
        )
        record: dict[str, object] = {
            "bucket_id": bucket.bucket_id,
            "module_count": bucket.module_count,
            "split": bucket.split.value,
        }
        try:
            task_path = manifest_path.parent / bucket.task_spec_path
            task = TaskSpec.from_json(task_path.read_text(encoding="utf-8"))
            accepted = bucket.metadata.get("accepted_nominal_trajectory")
            if not isinstance(accepted, dict):
                raise ValueError("bucket lacks accepted nominal trajectory binding")
            nominal_manifest = _resolve(str(accepted["set_manifest_path"]))
            bundle = load_order9_c3_accepted_nominal_bundle(
                nominal_manifest,
                bucket_id=bucket.bucket_id,
                repository_root=REPOSITORY_ROOT,
                expected_set_sha256=str(accepted["set_manifest_sha256"]),
                expected_artifact_sha256=str(accepted["artifact_sha256"]),
                expected_structural_hash=bucket.structural_hash,
                expected_task_spec_sha256=bucket.task_spec_sha256,
                expected_physical_model_hash=physical.stable_hash(),
            )
            contact = bundle.phase_trajectories[
                Order9ObjectTaskPhase.CONTACT_ACQUISITION.value
            ]
            solution = solve_order9_actuator_aware_nominal_preload(
                morphology=bundle.morphology,
                physical_model=physical,
                contact_knot=contact.knots[-1],
                candidate_set=bundle.contact_candidate_set,
                object_mass_kg=bucket.estimated_mass_kg,
                contact_friction=bucket.selected_gripper_friction,
                contact_stiffness_n_per_m=bucket.contact_stiffness_n_per_m,
                config=Order9ActuatorAwareNominalPreloadConfig(
                    minimum_inward_lead_m=runtime.c3_virtual_contact_inward_lead_m,
                    maximum_inward_lead_m=runtime.c3_actuator_aware_nominal_preload_maximum_m,
                    inward_lead_quantization_m=runtime.c3_actuator_aware_nominal_preload_quantization_m,
                    support_safety_factor=runtime.c3_actuator_aware_nominal_preload_support_safety_factor,
                    maximum_peak_effort_utilization=runtime.c3_actuator_aware_nominal_preload_maximum_peak_utilization,
                    minimum_anchor_force_fraction=runtime.c3_actuator_aware_nominal_preload_minimum_anchor_fraction,
                    model_error_margin_m=runtime.c3_actuator_aware_nominal_preload_model_error_margin_m,
                ),
            )
            record["preload"] = solution.to_dict()
            if not solution.feasible:
                record["accepted"] = False
                record["rejection_reason"] = solution.rejection_reason
            elif args.skip_collision:
                record["accepted"] = True
                record["collision_screened"] = False
            else:
                compression = solve_order9_virtual_contact_compression_for_achieved_lead(
                    morphology=bundle.morphology,
                    physical_model=physical,
                    contact_knot=contact.knots[-1],
                    candidate_set=bundle.contact_candidate_set,
                    minimum_achieved_inward_lead_m=solution.inward_lead_m,
                    maximum_requested_inward_lead_m=(
                        runtime.c3_actuator_aware_nominal_preload_maximum_m
                    ),
                    requested_lead_quantization_m=(
                        runtime.c3_actuator_aware_nominal_preload_quantization_m
                    ),
                )
                record["compression_solution"] = asdict(compression)
                collision_object = build_order9_c3_posture_collision_object(task)
                resolver = Order9PostureTrajectoryResolver(
                    physical,
                    config=Order9PostureResolverConfig(collision_margin_m=0.001),
                    collision_object=collision_object,
                    prefer_native_solver=True,
                    require_native_solver=True,
                )
                collision = limit_order9_virtual_contact_compression_by_collision(
                    morphology=bundle.morphology,
                    physical_model=physical,
                    contact_knot=contact.knots[-1],
                    solution=compression,
                    collision_object=collision_object,
                    collision_solver=resolver.ik_solver,
                    collision_margin_m=0.001,
                    maximum_action_joint_delta_rad=0.15,
                )
                record["collision_screened"] = True
                record["collision_limit"] = asdict(collision)
                achieved_leads_m = tuple(
                    float(value)
                    for value in compression.achieved_inward_displacement_m.values()
                )
                tangential_errors_m = tuple(
                    float(value) for value in compression.tangential_error_m.values()
                )
                minimum_achieved_lead_m = min(achieved_leads_m)
                maximum_tangential_error_m = max(tangential_errors_m)
                applied_lead_m = (
                    minimum_achieved_lead_m * collision.maximum_action_scale
                )
                minimum_physical_lead_m = max(
                    runtime.c3_virtual_contact_inward_lead_m,
                    solution.compliance_predicted_inward_lead_m,
                )
                record["minimum_achieved_inward_lead_m"] = (
                    minimum_achieved_lead_m
                )
                record["maximum_tangential_error_m"] = maximum_tangential_error_m
                record["collision_limited_inward_lead_m"] = applied_lead_m
                record["minimum_physical_inward_lead_m"] = minimum_physical_lead_m
                record["accepted"] = (
                    applied_lead_m + 1.0e-9 >= minimum_physical_lead_m
                    and maximum_tangential_error_m
                    <= max(0.002, 0.25 * solution.inward_lead_m)
                )
                if not record["accepted"]:
                    if applied_lead_m + 1.0e-9 < minimum_physical_lead_m:
                        record["rejection_reason"] = (
                            "achieved_lead_below_physical_compliance"
                        )
                    else:
                        record["rejection_reason"] = (
                            "compression_tangential_error"
                        )
        except BaseException as exc:
            record["accepted"] = False
            record["rejection_reason"] = f"{type(exc).__name__}: {exc}"
        records.append(record)

    rejected = [record for record in records if not bool(record["accepted"])]
    leads = [
        float(record["preload"]["inward_lead_m"])
        for record in records
        if isinstance(record.get("preload"), dict)
    ]
    summary_by_module: dict[str, dict[str, object]] = {}
    for module_count in sorted({int(record["module_count"]) for record in records}):
        subset = [record for record in records if record["module_count"] == module_count]
        subset_leads = [
            float(record["preload"]["inward_lead_m"])
            for record in subset
            if isinstance(record.get("preload"), dict)
        ]
        summary_by_module[str(module_count)] = {
            "bucket_count": len(subset),
            "accepted_count": sum(bool(record["accepted"]) for record in subset),
            "minimum_inward_lead_m": min(subset_leads) if subset_leads else None,
            "maximum_inward_lead_m": max(subset_leads) if subset_leads else None,
        }
    report = {
        "audit_version": AUDIT_VERSION,
        "config_path": str(config_path),
        "config_sha256": hash_file(config_path),
        "bucket_manifest_path": str(manifest_path),
        "bucket_manifest_sha256": hash_file(manifest_path),
        "physical_model_hash": physical.stable_hash(),
        "bucket_count": len(records),
        "accepted_count": len(records) - len(rejected),
        "rejected_count": len(rejected),
        "minimum_inward_lead_m": min(leads) if leads else None,
        "maximum_inward_lead_m": max(leads) if leads else None,
        "summary_by_module_count": summary_by_module,
        "wall_elapsed_s": time.perf_counter() - started,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "bucket_count", "accepted_count", "rejected_count",
        "minimum_inward_lead_m", "maximum_inward_lead_m",
        "summary_by_module_count", "wall_elapsed_s",
    )}, indent=2, sort_keys=True), flush=True)
    return 0 if not rejected else 2


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
