#!/usr/bin/env python3
from __future__ import annotations

"""Convert the split-safe Order 9 topology pool into hash-bound Isaac assets."""

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from isaaclab.app import AppLauncher


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool",
        default="artifacts/p4_full/order9/morphology_pool.json",
    )
    parser.add_argument(
        "--robot-model-config",
        default="configs/robot/robot_model.yaml",
    )
    parser.add_argument(
        "--source-urdf",
        default="assets/robots/holon/holon.urdf",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/p4_full/order9/morphology_assets",
    )
    parser.add_argument(
        "--manifest",
        default="artifacts/p4_full/order9/morphology_assets/manifest.json",
    )
    parser.add_argument("--mesh-search-dir", action="append", default=[])
    parser.add_argument("--structural-hash", action="append", default=[])
    parser.add_argument("--force-conversion", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    return parser


args_cli = _parser().parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.order3 import Order3MorphologyPoolManifest
from amsrr.simulation.order9_morphology_assets import (
    Order9MorphologyAssetManifest,
    load_order9_morphology_asset_manifest,
    order9_morphology_asset_entry,
    stage_order9_morphology_urdfs,
    validate_order9_morphology_asset_manifest_bytes,
    write_order9_morphology_asset_manifest,
)
from amsrr.utils.hashing import hash_file


def _resolve(path: str) -> Path:
    value = Path(path)
    return (REPOSITORY_ROOT / value).resolve() if not value.is_absolute() else value.resolve()


def _portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path.resolve())


def main() -> int:
    pool_path = _resolve(args_cli.pool)
    source_urdf = _resolve(args_cli.source_urdf)
    output_root = _resolve(args_cli.output_root)
    manifest_path = _resolve(args_cli.manifest)
    selected_hashes = set(args_cli.structural_hash)
    if manifest_path.is_file() and not args_cli.force_conversion and not selected_hashes:
        existing = load_order9_morphology_asset_manifest(manifest_path)
        validate_order9_morphology_asset_manifest_bytes(
            existing,
            repository_root=REPOSITORY_ROOT,
            expected_pool_sha256=hash_file(pool_path),
        )
        print(
            "ORDER9_MORPHOLOGY_ASSETS="
            + json.dumps(
                {
                    "status": "verified_existing",
                    "manifest": _portable(manifest_path),
                    "manifest_sha256": hash_file(manifest_path),
                    "entry_count": len(existing.entries),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    pool = Order3MorphologyPoolManifest.from_json(
        pool_path.read_text(encoding="utf-8")
    )
    model = build_physical_model_from_config(_resolve(args_cli.robot_model_config))
    if pool.physical_model_hash != model.stable_hash():
        raise RuntimeError("Order9 morphology pool physical-model hash is stale")
    mesh_dirs = [
        _resolve(path) for path in (args_cli.mesh_search_dir or ["module_urdf"])
    ]
    staged = stage_order9_morphology_urdfs(
        pool,
        source_urdf_path=source_urdf,
        output_root=output_root,
        mesh_search_dirs=mesh_dirs,
        structural_hashes=selected_hashes or None,
    )
    entries = []
    for index, item in enumerate(staged, start=1):
        converter = UrdfConverter(
            UrdfConverterCfg(
                asset_path=str(item.urdf_path),
                usd_dir=str(item.usd_directory),
                fix_base=False,
                merge_fixed_joints=False,
                collision_type="Convex Decomposition",
                force_usd_conversion=bool(args_cli.force_conversion),
                joint_drive=UrdfConverterCfg.JointDriveCfg(
                    gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                        stiffness=20.0,
                        damping=1.0,
                    ),
                    target_type="position",
                ),
            )
        )
        entries.append(
            order9_morphology_asset_entry(
                item,
                usd_path=converter.usd_path,
                repository_root=REPOSITORY_ROOT,
            )
        )
        print(
            f"ORDER9_MORPHOLOGY_ASSET_PROGRESS={index}/{len(staged)}:"
            f"{item.structural_hash}",
            flush=True,
        )
    manifest = Order9MorphologyAssetManifest(
        source_pool_path=_portable(pool_path),
        source_pool_sha256=hash_file(pool_path),
        source_pool_version=pool.pool_version,
        source_urdf_path=_portable(source_urdf),
        source_urdf_sha256=hash_file(source_urdf),
        physical_model_hash=model.stable_hash(),
        entries=entries,
        metadata={
            "entry_count": len(entries),
            "pool_entry_count": len(pool.entries),
            "all_pool_entries_converted": (
                not selected_hashes and len(entries) == len(pool.entries)
            ),
            "collision_approximation": "Convex Decomposition",
            "merge_fixed_joints": False,
            "fix_base": False,
            "mesh_search_dirs": [_portable(path) for path in mesh_dirs],
        },
    )
    write_order9_morphology_asset_manifest(manifest_path, manifest)
    validate_order9_morphology_asset_manifest_bytes(
        manifest,
        repository_root=REPOSITORY_ROOT,
        expected_pool_sha256=hash_file(pool_path),
    )
    print(
        "ORDER9_MORPHOLOGY_ASSETS="
        + json.dumps(
            {
                "status": "converted",
                "manifest": _portable(manifest_path),
                "manifest_sha256": hash_file(manifest_path),
                "entry_count": len(entries),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


try:
    _exit_code = main()
finally:
    simulation_app.close()
raise SystemExit(_exit_code)
