#!/usr/bin/env python3
from __future__ import annotations

"""Append one focused, split-safe C3 rollout bucket to an existing lineage."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.morphology.random_connected import morphology_structural_hash
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.order3 import Order3MorphologyPoolManifest
from amsrr.schemas.task_spec import TaskSpec
from amsrr.simulation.order9_morphology_assets import (
    Order9MorphologyAssetManifest,
    validate_order9_morphology_asset_manifest_bytes,
)
from amsrr.simulation.order9_object_task_state import load_order9_canonical_reset
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_randomization import Order9ConservativeRandomizer
from amsrr.training.order9_rollout_buckets import (
    ORDER9_C3_BUCKET_PRECHECK_VERSION,
    Order9PiLRolloutBucket,
    Order9PiLRolloutBucketManifest,
    _select_prechecked_c3_graph,
    load_order9_pi_l_rollout_bucket_manifest,
    validate_order9_pi_l_rollout_bucket_bytes,
)
from amsrr.training.order9_teacher import build_order8_grasp_carry_task_spec
from amsrr.utils.hashing import hash_file


EXPANSION_VERSION = "order9_c3_focused_rollout_bucket_expansion_v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument(
        "--split-index",
        type=int,
        required=True,
        help="Module-stratified index within the requested split.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        help="Globally unique randomization index; defaults to max(source)+1.",
    )
    parser.add_argument(
        "--curriculum-config",
        default="configs/training/order9_learning_curriculum.yaml",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.split_index < 0:
        raise ValueError("--split-index must be non-negative")
    source_path = _resolve(args.source_manifest)
    output_path = _resolve(args.output_manifest)
    if source_path.parent != output_path.parent:
        raise ValueError("expanded manifest must remain beside its source manifest")
    if output_path.exists():
        raise FileExistsError(output_path)
    validate_order9_pi_l_rollout_bucket_bytes(
        source_path, repository_root=REPOSITORY_ROOT
    )
    source = load_order9_pi_l_rollout_bucket_manifest(source_path)
    if source.stage_id != "c3_pi_l_ppo_arbitrary_morphology":
        raise ValueError("focused expansion only supports the C3 rollout lineage")

    config = load_order9_learning_config(_resolve(args.curriculum_config))
    physical_model = build_physical_model_from_config(
        _resolve(config.production_runtime.robot_model_config_path)
    )
    if physical_model.stable_hash() != source.physical_model_hash:
        raise ValueError("PhysicalModel differs from the source bucket lineage")
    canonical = load_order9_canonical_reset(
        _resolve(config.production_runtime.canonical_order8_report_path),
        expected_sha256=config.production_runtime.canonical_order8_report_sha256,
    )
    base_task = build_order8_grasp_carry_task_spec(
        object_pose_world=tuple(canonical.object_pose_world),
        object_size_m=(0.30, 0.40, 0.15),
        object_mass_kg=1.0,
        object_friction=0.6,
        required_transport_distance_m=canonical.transport_distance_m,
        support_height_m=config.randomization.support_top_z_m,
        max_contact_force_n=config.hard_checker.qp_force_scale_n,
        max_contact_torque_nm=config.hard_checker.qp_torque_scale_nm,
        selected_gripper_friction=(
            config.randomization.nominal_selected_gripper_friction
        ),
        task_id="order9-vectorized-base",
    )
    expected_base_hash = source.metadata.get("base_task_hash")
    if expected_base_hash != base_task.stable_hash():
        # The immutable bucket lineage predates the patch-moment range label.
        # Reproduce its archived task bytes without reverting the current
        # teacher contract used by new, unrelated lineages.
        legacy_base = TaskSpec.from_dict(base_task.to_dict())
        legacy_base.metadata = {
            **legacy_base.metadata,
            "teacher_version": "order9_natural_contact_teacher_v3_qp_ranges",
        }
        if legacy_base.stable_hash() == expected_base_hash:
            base_task = legacy_base
    if expected_base_hash != base_task.stable_hash():
        raise ValueError("base task differs from the source bucket lineage")

    sample_index = (
        max(bucket.sample_index for bucket in source.buckets) + 1
        if args.sample_index is None
        else int(args.sample_index)
    )
    if sample_index < 0 or any(
        bucket.sample_index == sample_index for bucket in source.buckets
    ):
        raise ValueError("--sample-index must be non-negative and globally unique")
    split = DatasetSplit(args.split)
    base_seed = int(source.metadata["base_seed"])
    sample_seed = base_seed + sample_index
    randomization = Order9ConservativeRandomizer(config.randomization).sample(
        base_task,
        seed=sample_seed,
        sample_index=sample_index,
    )
    task = TaskSpec.from_dict(randomization.task_spec.to_dict())

    pool_path = _resolve_required(source.source_pool_path, "source pool")
    asset_path = _resolve_required(
        source.source_asset_manifest_path, "source asset manifest"
    )
    if hash_file(pool_path) != source.source_pool_sha256:
        raise ValueError("source morphology pool bytes changed")
    if hash_file(asset_path) != source.source_asset_manifest_sha256:
        raise ValueError("source morphology asset manifest bytes changed")
    pool = Order3MorphologyPoolManifest.from_json(
        pool_path.read_text(encoding="utf-8")
    )
    used = {
        bucket.structural_hash
        for bucket in source.buckets
        if bucket.split == split
    }
    graph, topology_metadata = _select_prechecked_c3_graph(
        task=task,
        split=split,
        split_index=args.split_index,
        pool=pool,
        physical_model=physical_model,
        min_modules=2,
        max_modules=8,
        used_structural_hashes=used,
    )
    asset_manifest = Order9MorphologyAssetManifest.from_json(
        asset_path.read_text(encoding="utf-8")
    )
    validate_order9_morphology_asset_manifest_bytes(
        asset_manifest,
        repository_root=REPOSITORY_ROOT,
        expected_pool_sha256=source.source_pool_sha256,
    )
    asset_entry = asset_manifest.entry_for(graph)
    if asset_entry.split != split:
        raise ValueError("selected morphology asset crosses dataset splits")
    robot_usd = _resolve(asset_entry.usd_path)
    structural_hash = morphology_structural_hash(graph)
    bucket_id = f"{split.value}-{sample_index:06d}-{structural_hash[:12]}"
    if any(bucket.bucket_id == bucket_id for bucket in source.buckets):
        raise ValueError(f"expanded bucket already exists: {bucket_id}")

    task.metadata = {
        **task.metadata,
        "order9_c3_articulated_teacher_precheck": topology_metadata[
            "articulated_teacher_precheck"
        ],
        "order9_rollout_bucket_id": bucket_id,
        "dataset_split": split.value,
        "estimated_mass_kg": randomization.estimated_mass_properties.mass_kg,
        "estimated_inertia_body": list(
            randomization.estimated_mass_properties.inertia_kgm2
        ),
        "estimated_com_object": list(
            randomization.estimated_mass_properties.center_of_mass_object
        ),
    }
    task.validate()
    relative_bucket_dir = Path("expanded_buckets") / bucket_id
    bucket_dir = source_path.parent / relative_bucket_dir
    if bucket_dir.exists():
        raise FileExistsError(bucket_dir)
    bucket_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{bucket_id}.", dir=bucket_dir.parent)
    )
    try:
        task_path = temporary_dir / "task_spec.json"
        graph_path = temporary_dir / "morphology_graph.json"
        task_path.write_text(task.to_json(indent=2) + "\n", encoding="utf-8")
        graph_path.write_text(graph.to_json(indent=2) + "\n", encoding="utf-8")
        task_relative = relative_bucket_dir / task_path.name
        graph_relative = relative_bucket_dir / graph_path.name
        expanded = Order9PiLRolloutBucket(
            bucket_id=bucket_id,
            split=split,
            seed=sample_seed,
            sample_index=sample_index,
            task_id=task.task_id,
            task_spec_path=str(task_relative),
            task_spec_sha256=hash_file(task_path),
            morphology_graph_path=str(graph_relative),
            morphology_graph_sha256=hash_file(graph_path),
            morphology_hash=graph.stable_hash(),
            structural_hash=structural_hash,
            module_count=len(graph.modules),
            robot_usd_path=_portable(robot_usd),
            robot_usd_sha256=hash_file(robot_usd),
            selected_gripper_friction=randomization.selected_gripper_friction,
            contact_stiffness_n_per_m=randomization.contact_stiffness_n_per_m,
            contact_damping_n_s_per_m=randomization.contact_damping_n_s_per_m,
            estimated_mass_kg=randomization.estimated_mass_properties.mass_kg,
            estimated_inertia_body=list(
                randomization.estimated_mass_properties.inertia_kgm2
            ),
            estimated_com_object=list(
                randomization.estimated_mass_properties.center_of_mass_object
            ),
            randomization_version=randomization.randomization_version,
            topology_source="split_safe_pool_articulated_teacher_prechecked_v5",
            metadata={
                "split_bucket_index": args.split_index,
                "sampled_values": randomization.sampled_values,
                "true_mass_properties": randomization.true_mass_properties.to_dict(),
                "estimated_mass_properties": (
                    randomization.estimated_mass_properties.to_dict()
                ),
                "topology_provider": topology_metadata,
                "focused_expansion_version": EXPANSION_VERSION,
            },
        )
        expanded.validate()
        os.rename(temporary_dir, bucket_dir)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    buckets = list(source.buckets)
    insertion_index = next(
        (index for index, bucket in enumerate(buckets) if bucket.split != split),
        len(buckets),
    )
    buckets.insert(insertion_index, expanded)
    metadata = dict(source.metadata)
    metadata["train_bucket_count"] = sum(
        bucket.split == DatasetSplit.TRAIN for bucket in buckets
    )
    histogram = {
        split_name: {
            str(module_count): sum(
                bucket.split.value == split_name
                and bucket.module_count == module_count
                for bucket in buckets
            )
            for module_count in range(2, 9)
        }
        for split_name in ("train", "validation")
    }
    metadata["module_count_histogram_by_split"] = histogram
    metadata["focused_bucket_expansion"] = {
        "version": EXPANSION_VERSION,
        "source_manifest_path": _portable(source_path),
        "source_manifest_sha256": hash_file(source_path),
        "split": split.value,
        "split_index": args.split_index,
        "sample_index": sample_index,
        "bucket_id": bucket_id,
        "precheck_version": ORDER9_C3_BUCKET_PRECHECK_VERSION,
    }
    manifest = Order9PiLRolloutBucketManifest(
        stage_id=source.stage_id,
        stage_config_hash=source.stage_config_hash,
        curriculum_schedule_hash=source.curriculum_schedule_hash,
        config_hash=source.config_hash,
        physical_model_hash=source.physical_model_hash,
        topology_randomized=source.topology_randomized,
        buckets=buckets,
        source_pool_path=source.source_pool_path,
        source_pool_sha256=source.source_pool_sha256,
        source_asset_manifest_path=source.source_asset_manifest_path,
        source_asset_manifest_sha256=source.source_asset_manifest_sha256,
        manifest_version=source.manifest_version,
        metadata=metadata,
    )
    manifest.validate()
    temporary_manifest = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.tmp"
    )
    temporary_manifest.write_text(
        manifest.to_json(indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, output_path)
    validate_order9_pi_l_rollout_bucket_bytes(
        output_path, repository_root=REPOSITORY_ROOT
    )
    print(
        "ORDER9_C3_BUCKET_EXPANDED="
        + json.dumps(
            {
                "bucket_id": bucket_id,
                "module_count": expanded.module_count,
                "manifest": str(output_path),
                "manifest_sha256": hash_file(output_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _resolve_required(value: str | None, name: str) -> Path:
    if value is None:
        raise ValueError(f"source manifest lacks {name}")
    return _resolve(value)


def _portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
