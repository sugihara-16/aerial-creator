from __future__ import annotations

from pathlib import Path

import pytest

from amsrr.morphology.random_connected import morphology_structural_hash
from amsrr.robot_model.physical_model_builder import (
    build_physical_model_from_config,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.simulation.order8_natural_contact import (
    build_representative_order8_morphology,
)
from amsrr.simulation.order9_fixed_nominal_asset import (
    Order9FixedNominalAssetManifest,
    validate_order9_fixed_nominal_asset_manifest_bytes,
)
from amsrr.simulation.order9_morphology_assets import order9_usd_bundle_hash
from amsrr.utils.hashing import hash_file


def test_fixed_nominal_asset_manifest_binds_all_bytes(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[3]
    model = build_physical_model_from_config(
        repository / "configs/robot/robot_model.yaml"
    )
    morphology = build_representative_order8_morphology(model)
    source = tmp_path / "source.urdf"
    graph = tmp_path / "morphology_graph.json"
    generated = tmp_path / "generated.urdf"
    usd = tmp_path / "usd" / "robot.usda"
    source.write_text("<robot name='source'/>\n", encoding="utf-8")
    graph.write_text(morphology.to_json(indent=2) + "\n", encoding="utf-8")
    generated.write_text("<robot name='generated'/>\n", encoding="utf-8")
    usd.parent.mkdir()
    usd.write_text("#usda 1.0\n", encoding="utf-8")
    manifest = Order9FixedNominalAssetManifest(
        source_morphology_hash=morphology.stable_hash(),
        morphology_structural_hash=morphology_structural_hash(morphology),
        physical_model_hash=model.stable_hash(),
        source_urdf_path=str(source),
        source_urdf_sha256=hash_file(source),
        morphology_graph_path=str(graph),
        morphology_graph_sha256=hash_file(graph),
        generated_urdf_path=str(generated),
        generated_urdf_sha256=hash_file(generated),
        usd_path=str(usd),
        usd_sha256=hash_file(usd),
        usd_bundle_hash=order9_usd_bundle_hash(usd.parent),
    )

    assert validate_order9_fixed_nominal_asset_manifest_bytes(
        manifest,
        repository_root=tmp_path,
        expected_morphology=morphology,
        expected_physical_model_hash=model.stable_hash(),
    ) == usd

    usd.write_text("#usda 1.0\n# changed\n", encoding="utf-8")
    with pytest.raises(SchemaValidationError, match="USD root bytes changed"):
        validate_order9_fixed_nominal_asset_manifest_bytes(
            manifest,
            repository_root=tmp_path,
        )
