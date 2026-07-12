from __future__ import annotations

from pathlib import Path

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.order3 import Order3MorphologyPoolManifest
from amsrr.training.order3_morphology_pool import (
    Order3MorphologyPoolConfig,
    build_order3_morphology_pool,
    write_order3_morphology_pool,
)


MESH_SEARCH_DIRS = ("module_urdf", "module_urdf/mesh")


def _small_config() -> Order3MorphologyPoolConfig:
    return Order3MorphologyPoolConfig(
        master_seed=41,
        min_modules=2,
        max_modules=3,
        train_per_module_count=1,
        validation_per_module_count=1,
        held_out_per_module_count=1,
        two_module_train_per_module_count=1,
        mesh_search_dirs=MESH_SEARCH_DIRS,
    )


def test_order3_pool_is_balanced_deterministic_and_split_disjoint() -> None:
    first = build_order3_morphology_pool(_small_config())
    second = build_order3_morphology_pool(_small_config())

    assert first.to_dict() == second.to_dict()
    assert first.split_counts == {"train": 2, "validation": 2, "held_out": 2}
    assert first.module_count_counts == {
        "2": 3,
        "3": 3,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
    }
    hashes = [entry.structural_hash for entry in first.entries]
    assert len(hashes) == len(set(hashes))
    assert all(entry.feasibility_result.feasible for entry in first.entries)


def test_order3_pool_atomic_write_round_trip(tmp_path: Path) -> None:
    manifest = build_order3_morphology_pool(_small_config())
    destination = tmp_path / "pool.json"

    returned = write_order3_morphology_pool(manifest, destination)
    restored = Order3MorphologyPoolManifest.from_json(destination.read_text(encoding="utf-8"))

    assert returned == str(destination)
    assert restored.to_dict() == manifest.to_dict()
    assert list(tmp_path.glob(".pool.json.*.tmp")) == []


def test_order3_pool_config_rejects_out_of_scope_module_count() -> None:
    with pytest.raises(SchemaValidationError, match="2 <= min_modules"):
        Order3MorphologyPoolConfig(min_modules=1)
