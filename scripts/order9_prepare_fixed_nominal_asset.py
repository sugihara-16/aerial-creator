#!/usr/bin/env python3
from __future__ import annotations

"""Generate the graph-exact fixed-topology articulated USD used by C1/C2."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from isaaclab.app import AppLauncher


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot-model-config", default="configs/robot/robot_model.yaml"
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/p4_full/order9/fixed_nominal_asset",
    )
    parser.add_argument(
        "--manifest",
        default="artifacts/p4_full/order9/fixed_nominal_asset/manifest.json",
    )
    parser.add_argument("--force-conversion", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    return parser


args_cli = _parser().parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

from amsrr.morphology.random_connected import morphology_structural_hash
from amsrr.robot_model.fixed_morphology_urdf import (
    write_articulated_morphology_graph_urdf,
)
from amsrr.robot_model.physical_model_builder import (
    build_physical_model_from_config,
)
from amsrr.simulation.order8_natural_contact import (
    build_representative_order8_morphology,
)
from amsrr.simulation.order9_fixed_nominal_asset import (
    Order9FixedNominalAssetManifest,
    load_order9_fixed_nominal_asset_manifest,
    portable_order9_asset_path,
    validate_order9_fixed_nominal_asset_manifest_bytes,
    write_order9_fixed_nominal_asset_manifest,
)
from amsrr.simulation.order9_morphology_assets import order9_usd_bundle_hash
from amsrr.utils.hashing import hash_file


def _resolve(path: str) -> Path:
    value = Path(path)
    return (
        (REPOSITORY_ROOT / value).resolve()
        if not value.is_absolute()
        else value.resolve()
    )


def _write_or_verify(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"immutable Order9 fixed asset differs: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> dict[str, object]:
    model = build_physical_model_from_config(_resolve(args_cli.robot_model_config))
    morphology = build_representative_order8_morphology(model)
    morphology.validate()
    manifest_path = _resolve(args_cli.manifest)
    if manifest_path.is_file() and not args_cli.force_conversion:
        manifest = load_order9_fixed_nominal_asset_manifest(manifest_path)
        usd_path = validate_order9_fixed_nominal_asset_manifest_bytes(
            manifest,
            repository_root=REPOSITORY_ROOT,
            expected_morphology=morphology,
            expected_physical_model_hash=model.stable_hash(),
        )
        return {
            "status": "verified_existing",
            "manifest": portable_order9_asset_path(
                manifest_path, REPOSITORY_ROOT
            ),
            "manifest_sha256": hash_file(manifest_path),
            "source_morphology_hash": morphology.stable_hash(),
            "usd_path": portable_order9_asset_path(usd_path, REPOSITORY_ROOT),
            "usd_sha256": hash_file(usd_path),
        }

    source_urdf = Path(model.urdf_path).resolve()
    graph_hash = morphology.stable_hash()
    bucket = _resolve(args_cli.output_root) / graph_hash
    graph_path = bucket / "morphology_graph.json"
    _write_or_verify(graph_path, morphology.to_json(indent=2) + "\n")
    generated_urdf = (
        bucket
        / f"holon_order9_fixed_nominal_articulated_v3_{graph_hash[:12]}.urdf"
    )
    temporary_urdf = bucket / f".{generated_urdf.name}.generated"
    bucket.mkdir(parents=True, exist_ok=True)
    try:
        write_articulated_morphology_graph_urdf(
            source_urdf,
            temporary_urdf,
            morphology_graph=morphology,
            mesh_search_dirs=[REPOSITORY_ROOT / "module_urdf"],
        )
        _write_or_verify(
            generated_urdf, temporary_urdf.read_text(encoding="utf-8")
        )
    finally:
        try:
            temporary_urdf.unlink()
        except FileNotFoundError:
            pass
    converter = UrdfConverter(
        UrdfConverterCfg(
            asset_path=str(generated_urdf),
            usd_dir=str(bucket / "usd"),
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
    usd_path = Path(converter.usd_path).resolve()
    manifest = Order9FixedNominalAssetManifest(
        source_morphology_hash=graph_hash,
        morphology_structural_hash=morphology_structural_hash(morphology),
        physical_model_hash=model.stable_hash(),
        source_urdf_path=portable_order9_asset_path(
            source_urdf, REPOSITORY_ROOT
        ),
        source_urdf_sha256=hash_file(source_urdf),
        morphology_graph_path=portable_order9_asset_path(
            graph_path, REPOSITORY_ROOT
        ),
        morphology_graph_sha256=hash_file(graph_path),
        generated_urdf_path=portable_order9_asset_path(
            generated_urdf, REPOSITORY_ROOT
        ),
        generated_urdf_sha256=hash_file(generated_urdf),
        usd_path=portable_order9_asset_path(usd_path, REPOSITORY_ROOT),
        usd_sha256=hash_file(usd_path),
        usd_bundle_hash=order9_usd_bundle_hash(usd_path.parent),
    )
    write_order9_fixed_nominal_asset_manifest(manifest_path, manifest)
    validate_order9_fixed_nominal_asset_manifest_bytes(
        manifest,
        repository_root=REPOSITORY_ROOT,
        expected_morphology=morphology,
        expected_physical_model_hash=model.stable_hash(),
    )
    return {
        "status": "converted",
        "manifest": portable_order9_asset_path(manifest_path, REPOSITORY_ROOT),
        "manifest_sha256": hash_file(manifest_path),
        "source_morphology_hash": graph_hash,
        "usd_path": portable_order9_asset_path(usd_path, REPOSITORY_ROOT),
        "usd_sha256": hash_file(usd_path),
    }


try:
    _result = main()
    print("ORDER9_FIXED_NOMINAL_ASSET=" + json.dumps(_result, sort_keys=True))
    _exit_code = 0
finally:
    simulation_app.close()
raise SystemExit(_exit_code)
