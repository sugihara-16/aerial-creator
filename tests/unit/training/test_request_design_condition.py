"""Reject mismatches between an upstream planning graph and physical assets."""
from copy import deepcopy
from pathlib import Path

import pytest

from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.order3 import Order3MorphologyPoolManifest
from amsrr.simulation.order9_morphology_assets import (
    Order9MorphologyAssetManifest, order9_morphology_asset_entry,
    stage_order9_morphology_urdfs, write_order9_morphology_asset_manifest,
)
from amsrr.training.request_object_conditions import _load_bound_morphology
from amsrr.utils.hashing import hash_file


@pytest.fixture
def bound_design(tmp_path):
    pool_path = Path('artifacts/p4_full/order9/morphology_pool.json')
    source = Path('assets/robots/holon/holon.urdf')
    pool = Order3MorphologyPoolManifest.from_json(pool_path.read_text())
    graph = pool.entries[0].morphology_graph
    staged = stage_order9_morphology_urdfs(pool, source_urdf_path=source,
        output_root=tmp_path/'assets', mesh_search_dirs=('module_urdf',),
        structural_hashes={pool.entries[0].structural_hash})[0]
    usd = staged.usd_directory/'robot.usda'
    usd.parent.mkdir(parents=True, exist_ok=True)
    usd.write_text('#usda 1.0\n')  # Binding test only; physical execution is separate.
    asset = order9_morphology_asset_entry(staged, usd_path=usd, repository_root=Path.cwd())
    manifest = Order9MorphologyAssetManifest(
        source_pool_path=str(pool_path.resolve()), source_pool_sha256=hash_file(pool_path),
        source_pool_version=pool.pool_version, source_urdf_path=str(source.resolve()),
        source_urdf_sha256=hash_file(source), physical_model_hash=pool.physical_model_hash,
        entries=[asset])
    path = write_order9_morphology_asset_manifest(tmp_path/'manifest.json', manifest)
    binding = dict(path=str(staged.morphology_graph_path),
        sha256=hash_file(staged.morphology_graph_path),
        asset_manifest=dict(path=str(path), sha256=hash_file(path)))
    physical = build_physical_model_from_config('configs/robot/robot_model.yaml')
    return binding, physical, graph, asset


def test_new_design_resolves_its_own_physical_robot(bound_design):
    binding, physical, graph, asset = bound_design
    actual, resolved = _load_bound_morphology(binding, physical)
    assert actual.to_dict() == graph.to_dict()
    assert resolved == asset


def test_design_without_physical_binding_is_rejected(bound_design):
    binding, physical, _, _ = bound_design
    binding.pop('asset_manifest')
    with pytest.raises(ValueError, match='bound physical asset'):
        _load_bound_morphology(binding, physical)


def test_valid_graph_with_different_design_identity_is_rejected(bound_design, tmp_path):
    binding, physical, graph, _ = bound_design
    changed = deepcopy(graph)
    # A graph can retain the same topology while its design identity differs.
    changed.graph_id += ':changed'
    path = tmp_path/'different_graph.json'
    path.write_text(changed.to_json())
    binding.update(path=str(path), sha256=hash_file(path))
    with pytest.raises(ValueError, match='planning and physical morphologies differ'):
        _load_bound_morphology(binding, physical)


def test_modified_physical_asset_is_rejected(bound_design):
    binding, physical, _, asset = bound_design
    Path(asset.usd_path).write_text('#usda 1.0\n# changed\n')
    with pytest.raises(ValueError, match='USD root bytes changed'):
        _load_bound_morphology(binding, physical)
