from __future__ import annotations

"""Hash-bound fixed-topology articulated robot asset used by Order 9 C1/C2."""

import os
from dataclasses import dataclass
from pathlib import Path
import tempfile

from amsrr.morphology.random_connected import morphology_structural_hash
from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_non_empty
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.simulation.order9_morphology_assets import order9_usd_bundle_hash
from amsrr.utils.hashing import hash_file


ORDER9_FIXED_NOMINAL_ASSET_MANIFEST_VERSION = (
    "order9_fixed_nominal_asset_manifest_v3_articulated_graph_helper_body"
)

ORDER9_FIXED_NOMINAL_ASSEMBLY_KINEMATICS = (
    "graph_dock_connected_child_reroot_v2_explicit_helper_body"
)


@dataclass
class Order9FixedNominalAssetManifest(SchemaBase):
    source_morphology_hash: str
    morphology_structural_hash: str
    physical_model_hash: str
    source_urdf_path: str
    source_urdf_sha256: str
    morphology_graph_path: str
    morphology_graph_sha256: str
    generated_urdf_path: str
    generated_urdf_sha256: str
    usd_path: str
    usd_sha256: str
    usd_bundle_hash: str
    collision_approximation: str = "Convex Decomposition"
    merge_fixed_joints: bool = False
    fix_base: bool = False
    assembly_kinematics: str = ORDER9_FIXED_NOMINAL_ASSEMBLY_KINEMATICS
    manifest_version: str = ORDER9_FIXED_NOMINAL_ASSET_MANIFEST_VERSION

    def validate(self) -> None:
        if self.manifest_version != ORDER9_FIXED_NOMINAL_ASSET_MANIFEST_VERSION:
            raise SchemaValidationError(
                "Order9 fixed-nominal asset manifest version mismatch"
            )
        for name in (
            "source_morphology_hash",
            "morphology_structural_hash",
            "physical_model_hash",
            "source_urdf_sha256",
            "morphology_graph_sha256",
            "generated_urdf_sha256",
            "usd_sha256",
            "usd_bundle_hash",
        ):
            _require_sha256(str(getattr(self, name)), name)
        for name in (
            "source_urdf_path",
            "morphology_graph_path",
            "generated_urdf_path",
            "usd_path",
        ):
            require_non_empty(
                str(getattr(self, name)),
                f"Order9FixedNominalAssetManifest.{name}",
            )
        if self.collision_approximation != "Convex Decomposition":
            raise SchemaValidationError(
                "Order9 fixed-nominal asset requires convex decomposition"
            )
        if self.merge_fixed_joints or self.fix_base:
            raise SchemaValidationError(
                "Order9 fixed-nominal asset articulation settings differ"
            )
        if self.assembly_kinematics != ORDER9_FIXED_NOMINAL_ASSEMBLY_KINEMATICS:
            raise SchemaValidationError(
                "Order9 fixed-nominal asset assembly kinematics differ"
            )


def load_order9_fixed_nominal_asset_manifest(
    path: str | Path,
) -> Order9FixedNominalAssetManifest:
    source = Path(path)
    return Order9FixedNominalAssetManifest.from_json(
        source.read_text(encoding="utf-8")
    )


def write_order9_fixed_nominal_asset_manifest(
    path: str | Path,
    manifest: Order9FixedNominalAssetManifest,
) -> Path:
    manifest.validate()
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.to_json(indent=2) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def validate_order9_fixed_nominal_asset_manifest_bytes(
    manifest: Order9FixedNominalAssetManifest,
    *,
    repository_root: str | Path,
    expected_morphology: MorphologyGraph | None = None,
    expected_physical_model_hash: str | None = None,
) -> Path:
    manifest.validate()
    repository = Path(repository_root).resolve()
    source_urdf = _resolve(manifest.source_urdf_path, repository)
    graph_path = _resolve(manifest.morphology_graph_path, repository)
    generated_urdf = _resolve(manifest.generated_urdf_path, repository)
    usd_path = _resolve(manifest.usd_path, repository)
    for path, expected, label in (
        (source_urdf, manifest.source_urdf_sha256, "source URDF"),
        (graph_path, manifest.morphology_graph_sha256, "morphology graph"),
        (generated_urdf, manifest.generated_urdf_sha256, "generated URDF"),
        (usd_path, manifest.usd_sha256, "USD root"),
    ):
        if not path.is_file() or hash_file(path) != expected:
            raise SchemaValidationError(
                f"Order9 fixed-nominal asset {label} bytes changed"
            )
    if order9_usd_bundle_hash(usd_path.parent) != manifest.usd_bundle_hash:
        raise SchemaValidationError(
            "Order9 fixed-nominal asset USD bundle bytes changed"
        )
    graph = MorphologyGraph.from_json(graph_path.read_text(encoding="utf-8"))
    graph.validate()
    if (
        graph.stable_hash() != manifest.source_morphology_hash
        or morphology_structural_hash(graph) != manifest.morphology_structural_hash
    ):
        raise SchemaValidationError(
            "Order9 fixed-nominal asset graph identity changed"
        )
    if (
        expected_morphology is not None
        and expected_morphology.stable_hash() != manifest.source_morphology_hash
    ):
        raise SchemaValidationError(
            "Order9 fixed-nominal asset does not match the active morphology"
        )
    if (
        expected_physical_model_hash is not None
        and expected_physical_model_hash != manifest.physical_model_hash
    ):
        raise SchemaValidationError(
            "Order9 fixed-nominal asset physical-model identity differs"
        )
    return usd_path


def portable_order9_asset_path(path: str | Path, repository_root: str | Path) -> str:
    resolved = Path(path).resolve()
    repository = Path(repository_root).resolve()
    try:
        return str(resolved.relative_to(repository))
    except ValueError:
        return str(resolved)


def _resolve(path: str, repository: Path) -> Path:
    value = Path(path)
    return (repository / value).resolve() if not value.is_absolute() else value.resolve()


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise SchemaValidationError(
            f"Order9 fixed-nominal asset {name} is not SHA-256"
        )


__all__ = [
    "ORDER9_FIXED_NOMINAL_ASSEMBLY_KINEMATICS",
    "ORDER9_FIXED_NOMINAL_ASSET_MANIFEST_VERSION",
    "Order9FixedNominalAssetManifest",
    "load_order9_fixed_nominal_asset_manifest",
    "portable_order9_asset_path",
    "validate_order9_fixed_nominal_asset_manifest_bytes",
    "write_order9_fixed_nominal_asset_manifest",
]
