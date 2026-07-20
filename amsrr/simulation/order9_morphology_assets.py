from __future__ import annotations

"""Hash-bound reset-time morphology assets for Order 9 topology buckets."""

import os
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
from typing import Iterable

from amsrr.morphology.random_connected import morphology_structural_hash
from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_non_empty
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.order3 import Order3MorphologyPoolManifest
from amsrr.robot_model.fixed_morphology_urdf import write_fixed_morphology_graph_urdf
from amsrr.utils.hashing import hash_file, stable_hash


ORDER9_MORPHOLOGY_ASSET_MANIFEST_VERSION = "order9_morphology_asset_manifest_v1"


@dataclass
class Order9MorphologyAssetEntry(SchemaBase):
    structural_hash: str
    source_morphology_hash: str
    module_count: int
    split: DatasetSplit
    morphology_graph_path: str
    morphology_graph_sha256: str
    urdf_path: str
    urdf_sha256: str
    usd_path: str
    usd_sha256: str
    usd_bundle_hash: str
    collision_approximation: str = "Convex Decomposition"
    metadata: dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        for name in (
            "structural_hash",
            "source_morphology_hash",
            "morphology_graph_sha256",
            "urdf_sha256",
            "usd_sha256",
            "usd_bundle_hash",
        ):
            _require_sha256(str(getattr(self, name)), name)
        for name in ("morphology_graph_path", "urdf_path", "usd_path"):
            require_non_empty(
                str(getattr(self, name)), f"Order9MorphologyAssetEntry.{name}"
            )
        if not 2 <= self.module_count <= 8:
            raise SchemaValidationError(
                "Order9 morphology asset module_count must lie within [2, 8]"
            )
        if self.collision_approximation != "Convex Decomposition":
            raise SchemaValidationError(
                "Order9 full-mesh shadow assets require convex decomposition"
            )


@dataclass
class Order9MorphologyAssetManifest(SchemaBase):
    source_pool_path: str
    source_pool_sha256: str
    source_pool_version: str
    source_urdf_path: str
    source_urdf_sha256: str
    physical_model_hash: str
    entries: list[Order9MorphologyAssetEntry]
    manifest_version: str = ORDER9_MORPHOLOGY_ASSET_MANIFEST_VERSION
    metadata: dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if self.manifest_version != ORDER9_MORPHOLOGY_ASSET_MANIFEST_VERSION:
            raise SchemaValidationError("Order9 morphology asset manifest version mismatch")
        for name in ("source_pool_path", "source_pool_version", "source_urdf_path"):
            require_non_empty(
                str(getattr(self, name)), f"Order9MorphologyAssetManifest.{name}"
            )
        for name in (
            "source_pool_sha256",
            "source_urdf_sha256",
            "physical_model_hash",
        ):
            _require_sha256(str(getattr(self, name)), name)
        if not self.entries:
            raise SchemaValidationError("Order9 morphology asset manifest is empty")
        hashes = [entry.structural_hash for entry in self.entries]
        if len(hashes) != len(set(hashes)):
            raise SchemaValidationError(
                "Order9 morphology asset manifest repeats a structural hash"
            )

    def entry_for(self, morphology_graph: MorphologyGraph) -> Order9MorphologyAssetEntry:
        structural_hash = morphology_structural_hash(morphology_graph)
        matches = [entry for entry in self.entries if entry.structural_hash == structural_hash]
        if len(matches) != 1:
            raise SchemaValidationError(
                "Order9 morphology asset manifest has no unique topology bucket"
            )
        return matches[0]


@dataclass(frozen=True)
class Order9StagedMorphologyAsset:
    structural_hash: str
    source_morphology_hash: str
    module_count: int
    split: DatasetSplit
    morphology_graph_path: Path
    urdf_path: Path
    usd_directory: Path


def stage_order9_morphology_urdfs(
    pool: Order3MorphologyPoolManifest,
    *,
    source_urdf_path: str | Path,
    output_root: str | Path,
    mesh_search_dirs: Iterable[str | Path] = ("module_urdf",),
    structural_hashes: set[str] | None = None,
) -> list[Order9StagedMorphologyAsset]:
    """Generate deterministic graph JSON/URDF inputs before Kit conversion."""

    pool.validate()
    source = Path(source_urdf_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    root = Path(output_root).resolve()
    selected = [
        entry
        for entry in pool.entries
        if structural_hashes is None or entry.structural_hash in structural_hashes
    ]
    if not selected:
        raise SchemaValidationError("Order9 asset staging selected no morphology")
    staged: list[Order9StagedMorphologyAsset] = []
    seen: set[str] = set()
    for entry in selected:
        graph = entry.morphology_graph
        structural_hash = morphology_structural_hash(graph)
        if structural_hash != entry.structural_hash:
            raise SchemaValidationError(
                "Order9 morphology pool entry has a stale structural hash"
            )
        if structural_hash in seen:
            continue
        seen.add(structural_hash)
        bucket = root / structural_hash
        bucket.mkdir(parents=True, exist_ok=True)
        graph_path = bucket / "morphology_graph.json"
        _write_or_verify_text(graph_path, graph.to_json(indent=2) + "\n")
        urdf_path = bucket / f"holon_order9_{structural_hash[:12]}.urdf"
        temporary = bucket / f".{urdf_path.name}.generated"
        try:
            write_fixed_morphology_graph_urdf(
                source,
                temporary,
                morphology_graph=graph,
                mesh_search_dirs=list(mesh_search_dirs),
            )
            payload = temporary.read_text(encoding="utf-8")
            _write_or_verify_text(urdf_path, payload)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        staged.append(
            Order9StagedMorphologyAsset(
                structural_hash=structural_hash,
                source_morphology_hash=graph.stable_hash(),
                module_count=len(graph.modules),
                split=entry.split,
                morphology_graph_path=graph_path,
                urdf_path=urdf_path,
                usd_directory=bucket / "usd",
            )
        )
    return sorted(staged, key=lambda item: (item.module_count, item.structural_hash))


def order9_morphology_asset_entry(
    staged: Order9StagedMorphologyAsset,
    *,
    usd_path: str | Path,
    repository_root: str | Path,
) -> Order9MorphologyAssetEntry:
    repository = Path(repository_root).resolve()
    usd = Path(usd_path).resolve()
    if not usd.is_file():
        raise FileNotFoundError(usd)
    for path in (staged.morphology_graph_path, staged.urdf_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    graph = MorphologyGraph.from_json(
        staged.morphology_graph_path.read_text(encoding="utf-8")
    )
    if morphology_structural_hash(graph) != staged.structural_hash:
        raise SchemaValidationError("Order9 staged morphology graph changed")
    return Order9MorphologyAssetEntry(
        structural_hash=staged.structural_hash,
        source_morphology_hash=staged.source_morphology_hash,
        module_count=staged.module_count,
        split=staged.split,
        morphology_graph_path=_portable_path(staged.morphology_graph_path, repository),
        morphology_graph_sha256=hash_file(staged.morphology_graph_path),
        urdf_path=_portable_path(staged.urdf_path, repository),
        urdf_sha256=hash_file(staged.urdf_path),
        usd_path=_portable_path(usd, repository),
        usd_sha256=hash_file(usd),
        usd_bundle_hash=order9_usd_bundle_hash(usd.parent),
        metadata={
            "graph_id": graph.graph_id,
            "asset_bucket_directory": _portable_path(usd.parent.parent, repository),
        },
    )


def order9_usd_bundle_hash(directory: str | Path) -> str:
    root = Path(directory).resolve()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise SchemaValidationError("Order9 USD bundle contains no files")
    return stable_hash(
        {str(path.relative_to(root)): hash_file(path) for path in files}
    )


def validate_order9_morphology_asset_manifest_bytes(
    manifest: Order9MorphologyAssetManifest,
    *,
    repository_root: str | Path,
    expected_pool_sha256: str | None = None,
) -> None:
    manifest.validate()
    repository = Path(repository_root).resolve()
    pool = _resolve_portable(manifest.source_pool_path, repository)
    source = _resolve_portable(manifest.source_urdf_path, repository)
    if hash_file(pool) != manifest.source_pool_sha256:
        raise SchemaValidationError("Order9 morphology asset source pool bytes changed")
    if expected_pool_sha256 is not None and manifest.source_pool_sha256 != expected_pool_sha256:
        raise SchemaValidationError("Order9 morphology asset source pool identity mismatch")
    if hash_file(source) != manifest.source_urdf_sha256:
        raise SchemaValidationError("Order9 morphology asset source URDF bytes changed")
    for entry in manifest.entries:
        graph_path = _resolve_portable(entry.morphology_graph_path, repository)
        urdf_path = _resolve_portable(entry.urdf_path, repository)
        usd_path = _resolve_portable(entry.usd_path, repository)
        if hash_file(graph_path) != entry.morphology_graph_sha256:
            raise SchemaValidationError("Order9 morphology graph asset bytes changed")
        if hash_file(urdf_path) != entry.urdf_sha256:
            raise SchemaValidationError("Order9 morphology URDF asset bytes changed")
        if hash_file(usd_path) != entry.usd_sha256:
            raise SchemaValidationError("Order9 morphology USD root bytes changed")
        if order9_usd_bundle_hash(usd_path.parent) != entry.usd_bundle_hash:
            raise SchemaValidationError("Order9 morphology USD bundle bytes changed")
        graph = MorphologyGraph.from_json(graph_path.read_text(encoding="utf-8"))
        if (
            graph.stable_hash() != entry.source_morphology_hash
            or morphology_structural_hash(graph) != entry.structural_hash
        ):
            raise SchemaValidationError("Order9 morphology asset graph identity changed")


def write_order9_morphology_asset_manifest(
    path: str | Path,
    manifest: Order9MorphologyAssetManifest,
) -> Path:
    manifest.validate()
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.to_json(indent=2) + "\n"
    _atomic_replace_text(destination, payload)
    return destination


def load_order9_morphology_asset_manifest(
    path: str | Path,
) -> Order9MorphologyAssetManifest:
    source = Path(path)
    return Order9MorphologyAssetManifest.from_json(source.read_text(encoding="utf-8"))


def _write_or_verify_text(path: Path, payload: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"Order9 immutable asset differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
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
        if path.exists():
            raise FileExistsError(f"Order9 immutable asset appeared concurrently: {path}")
        os.link(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _atomic_replace_text(path: Path, payload: str) -> None:
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


def _portable_path(path: Path, repository: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repository))
    except ValueError:
        return str(resolved)


def _resolve_portable(path: str, repository: Path) -> Path:
    value = Path(path)
    return (repository / value).resolve() if not value.is_absolute() else value.resolve()


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise SchemaValidationError(f"Order9 morphology asset {name} is not SHA-256")


__all__ = [
    "ORDER9_MORPHOLOGY_ASSET_MANIFEST_VERSION",
    "Order9MorphologyAssetEntry",
    "Order9MorphologyAssetManifest",
    "Order9StagedMorphologyAsset",
    "load_order9_morphology_asset_manifest",
    "order9_morphology_asset_entry",
    "order9_usd_bundle_hash",
    "stage_order9_morphology_urdfs",
    "validate_order9_morphology_asset_manifest_bytes",
    "write_order9_morphology_asset_manifest",
]
