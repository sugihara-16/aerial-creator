from __future__ import annotations

from pathlib import Path

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.order3 import Order3MorphologyPoolManifest
from amsrr.simulation.order9_morphology_assets import (
    Order9MorphologyAssetManifest,
    order9_morphology_asset_entry,
    stage_order9_morphology_urdfs,
    validate_order9_morphology_asset_manifest_bytes,
)
from amsrr.utils.hashing import hash_file


POOL_PATH = Path("artifacts/p4_full/order9/morphology_pool.json")
SOURCE_URDF = Path("assets/robots/holon/holon.urdf")


def test_staged_topology_asset_is_structurally_hash_bound(tmp_path) -> None:
    pool = Order3MorphologyPoolManifest.from_json(POOL_PATH.read_text(encoding="utf-8"))
    source_entry = pool.entries[0]
    staged = stage_order9_morphology_urdfs(
        pool,
        source_urdf_path=SOURCE_URDF,
        output_root=tmp_path,
        mesh_search_dirs=("module_urdf",),
        structural_hashes={source_entry.structural_hash},
    )

    assert len(staged) == 1
    assert staged[0].structural_hash == source_entry.structural_hash
    assert staged[0].morphology_graph_path.is_file()
    assert staged[0].urdf_path.is_file()
    assert "articulated_v2" in staged[0].urdf_path.name
    payload = staged[0].urdf_path.read_text(encoding="utf-8")
    assert "module_0__root" in payload
    if staged[0].module_count > 1:
        assert "articulated_graph_edge_" in payload


def test_manifest_validates_every_usd_payload_byte(tmp_path) -> None:
    pool = Order3MorphologyPoolManifest.from_json(POOL_PATH.read_text(encoding="utf-8"))
    source_entry = pool.entries[0]
    staged = stage_order9_morphology_urdfs(
        pool,
        source_urdf_path=SOURCE_URDF,
        output_root=tmp_path / "assets",
        mesh_search_dirs=("module_urdf",),
        structural_hashes={source_entry.structural_hash},
    )[0]
    usd = staged.usd_directory / "robot.usda"
    payload = staged.usd_directory / "payloads" / "geometry.usda"
    payload.parent.mkdir(parents=True)
    usd.write_text("#usda 1.0\n", encoding="utf-8")
    payload.write_text("#usda 1.0\ndef Xform \"geometry\" {}\n", encoding="utf-8")
    asset_entry = order9_morphology_asset_entry(
        staged,
        usd_path=usd,
        repository_root=Path.cwd(),
    )
    manifest = Order9MorphologyAssetManifest(
        source_pool_path=str(POOL_PATH.resolve()),
        source_pool_sha256=hash_file(POOL_PATH),
        source_pool_version=pool.pool_version,
        source_urdf_path=str(SOURCE_URDF.resolve()),
        source_urdf_sha256=hash_file(SOURCE_URDF),
        physical_model_hash=pool.physical_model_hash,
        entries=[asset_entry],
    )

    validate_order9_morphology_asset_manifest_bytes(
        manifest,
        repository_root=Path.cwd(),
        expected_pool_sha256=hash_file(POOL_PATH),
    )
    assert manifest.entry_for(source_entry.morphology_graph) == asset_entry

    payload.write_text("changed\n", encoding="utf-8")
    with pytest.raises(SchemaValidationError, match="bundle bytes changed"):
        validate_order9_morphology_asset_manifest_bytes(
            manifest,
            repository_root=Path.cwd(),
        )
