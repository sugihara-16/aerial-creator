from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

from amsrr.morphology.random_connected import morphology_structural_hash
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.order3 import (
    ORDER3_DATASET_VERSION,
    ORDER3_POLICY_FAMILY,
    Order3DatasetManifest,
    Order3PolicyTransition,
)
from amsrr.schemas.policies import POLICY_COMMAND_CONTRACT_CENTROIDAL


ORDER3_DATASET_WRITER_VERSION = "order3_dataset_writer_v1"
DEFAULT_ORDER3_DATASET_DIR = "artifacts/p4_full/order3_pi_l_v2/datasets"

_MANIFEST_NAME = "manifest.json"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_LEGACY_P4_3_ROOT = (_REPOSITORY_ROOT / "artifacts" / "p4_3").resolve()
_ACTOR_OBSERVATION_CONTRACT = (
    "runtime_observation_without_privileged_contact_wrench_or_ground_truth_disturbance"
)
_PRIVILEGED_INPUT_CONTRACT = (
    "transition_privileged_disturbance_body_and_non_actor_metrics_only"
)
_FORBIDDEN_ACTOR_KEYS = {
    "contact_force",
    "contact_force_world",
    "contact_impulse",
    "contact_impulse_world",
    "contact_wrench",
    "contact_wrench_world",
    "disturbance_ground_truth",
    "external_disturbance_ground_truth",
    "ground_truth_contact_force",
    "ground_truth_contact_wrench",
    "privileged_disturbance_body",
}


@dataclass(frozen=True)
class Order3DatasetIOResult:
    manifest: Order3DatasetManifest
    manifest_path: str
    transitions: list[Order3PolicyTransition]


def write_order3_dataset(
    transitions: Iterable[Order3PolicyTransition],
    *,
    pool_hash: str,
    physical_model_hash: str,
    config_hash: str,
    output_dir: str | Path = DEFAULT_ORDER3_DATASET_DIR,
    metadata: dict[str, Any] | None = None,
    shard_size: int = 100_000,
    overwrite: bool = False,
) -> Order3DatasetIOResult:
    """Write a deterministic, contract-versioned Order-3 transition dataset.

    The legacy ``artifacts/p4_3`` tree is rejected unconditionally.  Shards are
    written atomically and the manifest is replaced last, so readers never see
    a manifest that references a not-yet-written shard.
    """

    if not isinstance(shard_size, int) or isinstance(shard_size, bool) or shard_size <= 0:
        raise SchemaValidationError("order3 dataset shard_size must be a positive integer")
    destination = Path(output_dir)
    _reject_legacy_p4_3_path(destination)
    records = list(transitions)
    _validate_transition_set(records, require_all_splits=True)

    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Order-3 dataset output directory is not empty: {destination}; "
            "pass overwrite=True only for an existing Order-3 v2 dataset"
        )
    if overwrite and destination.exists() and any(destination.iterdir()):
        _require_existing_order3_dataset(destination)
    destination.mkdir(parents=True, exist_ok=True)

    grouped = _grouped_sorted_transitions(records)
    shard_payloads: dict[str, bytes] = {}
    transition_shards: dict[str, list[str]] = {}
    transition_hashes: dict[str, str] = {}
    for split in DatasetSplit:
        split_records = grouped[split]
        split_paths: list[str] = []
        for shard_index, start in enumerate(range(0, len(split_records), shard_size)):
            shard_records = split_records[start : start + shard_size]
            payload = "".join(f"{record.to_json()}\n" for record in shard_records).encode(
                "utf-8"
            )
            payload_hash = _sha256_bytes(payload)
            # Content-address the shard name so overwrite=True can stage a new
            # generation without invalidating the manifest currently visible to
            # concurrent readers. The manifest is atomically replaced only after
            # every new shard is durable.
            relative_path = (
                f"transitions_{split.value}_{shard_index:05d}_{payload_hash[:12]}.jsonl"
            )
            split_paths.append(relative_path)
            shard_payloads[relative_path] = payload
            transition_hashes[relative_path] = payload_hash
        transition_shards[split.value] = split_paths

    morphology_hashes = {
        split.value: sorted({record.structural_hash for record in grouped[split]})
        for split in DatasetSplit
    }
    transition_counts = {
        split.value: len(grouped[split])
        for split in DatasetSplit
    }
    real_isaac_episode_counts = _real_isaac_episode_counts(records)
    protected_metadata = {
        "writer_version": ORDER3_DATASET_WRITER_VERSION,
        "deterministic_record_order": (
            "structural_hash,episode_id,step_index,time_s,graph_id"
        ),
        "shard_size": shard_size,
        "split_unit": "canonical_structural_hash",
        "actor_observation_contract": _ACTOR_OBSERVATION_CONTRACT,
        "privileged_input_contract": _PRIVILEGED_INPUT_CONTRACT,
        "module_count_counts": _module_count_counts(records),
        "transition_module_count_counts": _transition_module_count_counts(records),
        "p4_full_completion_claim": False,
        "legacy_p4_3_artifact_reused": False,
    }
    manifest_metadata = dict(metadata or {})
    for key, value in protected_metadata.items():
        if key in manifest_metadata and manifest_metadata[key] != value:
            raise SchemaValidationError(
                f"Order-3 dataset metadata cannot override protected key {key!r}"
            )
        manifest_metadata[key] = value

    manifest = Order3DatasetManifest(
        dataset_version=ORDER3_DATASET_VERSION,
        policy_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
        policy_family=ORDER3_POLICY_FAMILY,
        pool_hash=pool_hash,
        physical_model_hash=physical_model_hash,
        config_hash=config_hash,
        transition_shards=transition_shards,
        transition_shard_hashes=transition_hashes,
        transition_counts=transition_counts,
        morphology_hashes=morphology_hashes,
        real_isaac_episode_counts=real_isaac_episode_counts,
        actor_privileged_wrench_inputs=False,
        metadata=manifest_metadata,
    )

    prior_paths = _existing_manifest_shard_paths(destination) if overwrite else set()
    for relative_path in sorted(shard_payloads):
        _atomic_write_bytes(destination / relative_path, shard_payloads[relative_path])
    manifest_path = destination / _MANIFEST_NAME
    _atomic_write_bytes(manifest_path, (manifest.to_json(indent=2) + "\n").encode("utf-8"))
    current_paths = set(shard_payloads)
    for stale_relative_path in sorted(prior_paths - current_paths):
        stale_path = _safe_dataset_child(destination, stale_relative_path)
        if stale_path.is_file():
            stale_path.unlink()

    # Re-read all persisted bytes and re-run every invariant before returning.
    return load_order3_dataset(manifest_path)


def load_order3_dataset(
    path: str | Path = DEFAULT_ORDER3_DATASET_DIR,
) -> Order3DatasetIOResult:
    """Load and independently validate an Order-3 v2 dataset manifest/shards."""

    manifest_path = Path(path)
    if manifest_path.is_dir() or manifest_path.name != _MANIFEST_NAME:
        manifest_path = manifest_path / _MANIFEST_NAME
    _reject_legacy_p4_3_path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Order-3 dataset manifest does not exist: {manifest_path}")
    try:
        manifest = Order3DatasetManifest.from_json(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, SchemaValidationError) as exc:
        raise SchemaValidationError(f"invalid Order-3 dataset manifest: {exc}") from exc
    _validate_manifest_contract_metadata(manifest)

    dataset_root = manifest_path.parent
    records: list[Order3PolicyTransition] = []
    seen_shard_paths: set[str] = set()
    for split in DatasetSplit:
        relative_paths = manifest.transition_shards[split.value]
        if not relative_paths:
            raise SchemaValidationError(
                f"Order3DatasetManifest.{split.value} must contain at least one shard"
            )
        if relative_paths != sorted(relative_paths):
            raise SchemaValidationError(
                f"Order3DatasetManifest.{split.value} shard paths must be sorted"
            )
        for relative_path in relative_paths:
            if relative_path in seen_shard_paths:
                raise SchemaValidationError(f"duplicate Order-3 shard path: {relative_path}")
            seen_shard_paths.add(relative_path)
            shard_path = _safe_dataset_child(dataset_root, relative_path)
            expected_hash = manifest.transition_shard_hashes[relative_path]
            if not _is_sha256(expected_hash):
                raise SchemaValidationError(
                    f"Order-3 shard {relative_path!r} has an invalid sha256 value"
                )
            if not shard_path.is_file() or _sha256_file(shard_path) != expected_hash:
                raise SchemaValidationError(
                    f"Order-3 shard integrity check failed: {relative_path}"
                )
            shard_count = 0
            try:
                with shard_path.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        stripped = line.strip()
                        if not stripped:
                            continue
                        record = Order3PolicyTransition.from_json(stripped)
                        if record.to_json() != stripped:
                            raise SchemaValidationError(
                                "Order-3 shard rows must use deterministic canonical JSON"
                            )
                        if record.split != split:
                            raise SchemaValidationError(
                                f"Order-3 shard {relative_path!r} contains a "
                                f"{record.split.value!r} record"
                            )
                        records.append(record)
                        shard_count += 1
            except (OSError, UnicodeError, json.JSONDecodeError, SchemaValidationError) as exc:
                raise SchemaValidationError(
                    f"invalid Order-3 shard row at {relative_path}:"
                    f"{locals().get('line_number', 0)}: {exc}"
                ) from exc
            if shard_count == 0:
                raise SchemaValidationError(f"Order-3 shard must not be empty: {relative_path}")

    _validate_transition_set(records, require_all_splits=True)
    grouped = _grouped_sorted_transitions(records)
    expected_records = [record for split in DatasetSplit for record in grouped[split]]
    if records != expected_records:
        raise SchemaValidationError("Order-3 dataset records are not in deterministic order")
    expected_counts = {split.value: len(grouped[split]) for split in DatasetSplit}
    if manifest.transition_counts != expected_counts:
        raise SchemaValidationError("Order3DatasetManifest.transition_counts mismatch")
    expected_hashes = {
        split.value: sorted({record.structural_hash for record in grouped[split]})
        for split in DatasetSplit
    }
    if manifest.morphology_hashes != expected_hashes:
        raise SchemaValidationError("Order3DatasetManifest.morphology_hashes mismatch")
    if manifest.real_isaac_episode_counts != _real_isaac_episode_counts(records):
        raise SchemaValidationError("Order3DatasetManifest.real_isaac_episode_counts mismatch")
    if manifest.metadata.get("module_count_counts") != _module_count_counts(records):
        raise SchemaValidationError("Order3DatasetManifest module_count_counts mismatch")
    if manifest.metadata.get("transition_module_count_counts") != _transition_module_count_counts(
        records
    ):
        raise SchemaValidationError(
            "Order3DatasetManifest transition_module_count_counts mismatch"
        )
    return Order3DatasetIOResult(
        manifest=manifest,
        manifest_path=str(manifest_path),
        transitions=records,
    )


def _validate_transition_set(
    records: list[Order3PolicyTransition],
    *,
    require_all_splits: bool,
) -> None:
    if not records:
        raise SchemaValidationError("Order-3 transition dataset must not be empty")
    by_split: dict[DatasetSplit, set[str]] = {split: set() for split in DatasetSplit}
    episode_contracts: dict[str, tuple[DatasetSplit, str, str, bool]] = {}
    episode_steps: dict[str, list[Order3PolicyTransition]] = {}
    record_keys: set[tuple[str, int]] = set()
    for record in records:
        if not isinstance(record, Order3PolicyTransition):
            raise TypeError("Order-3 dataset values must be Order3PolicyTransition instances")
        if record.dataset_version != ORDER3_DATASET_VERSION:
            raise SchemaValidationError("Order-3 transition has an unsupported dataset version")
        if record.policy_contract_version != POLICY_COMMAND_CONTRACT_CENTROIDAL:
            raise SchemaValidationError("Order-3 transition must use centroidal_local_joint_v2")
        if not math.isclose(
            record.time_s,
            record.runtime_observation.time_s,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise SchemaValidationError(
                "Order3PolicyTransition.time_s must match RuntimeObservation.time_s"
            )
        computed_hash = morphology_structural_hash(
            record.runtime_observation.morphology_graph
        )
        module_count = len(record.runtime_observation.morphology_graph.modules)
        if not 2 <= module_count <= 8:
            raise SchemaValidationError(
                "Order-3 transition morphology module count must be in [2, 8]"
            )
        if record.structural_hash != computed_hash:
            raise SchemaValidationError(
                "Order3PolicyTransition.structural_hash does not match morphology graph"
            )
        _validate_actor_privileged_separation(record)
        record_key = (record.episode_id, record.step_index)
        if record_key in record_keys:
            raise SchemaValidationError(
                f"duplicate Order-3 episode/step identity: {record_key}"
            )
        record_keys.add(record_key)
        isaac_backed = _record_is_real_isaac(record)
        episode_contract = (
            record.split,
            record.structural_hash,
            record.graph_id,
            isaac_backed,
        )
        previous_contract = episode_contracts.setdefault(record.episode_id, episode_contract)
        if previous_contract != episode_contract:
            raise SchemaValidationError(
                "Order-3 episode cannot cross split, morphology, graph, or provenance boundaries"
            )
        by_split[record.split].add(record.structural_hash)
        episode_steps.setdefault(record.episode_id, []).append(record)

    split_order = list(DatasetSplit)
    for index, left in enumerate(split_order):
        if require_all_splits and not by_split[left]:
            raise SchemaValidationError(
                f"Order-3 transition dataset split {left.value!r} must not be empty"
            )
        for right in split_order[index + 1 :]:
            overlap = sorted(by_split[left].intersection(by_split[right]))
            if overlap:
                raise SchemaValidationError(
                    "Order-3 morphology structural hashes must be split-disjoint; "
                    f"{left.value}/{right.value} overlap: {overlap}"
                )

    for episode_id, episode_records in episode_steps.items():
        ordered = sorted(episode_records, key=lambda item: (item.step_index, item.time_s))
        if any(
            current.time_s < previous.time_s
            for previous, current in zip(ordered, ordered[1:])
        ):
            raise SchemaValidationError(
                f"Order-3 episode {episode_id!r} has decreasing transition time"
            )
        terminal_indices = [index for index, item in enumerate(ordered) if item.terminal]
        if terminal_indices and terminal_indices != [len(ordered) - 1]:
            raise SchemaValidationError(
                f"Order-3 episode {episode_id!r} terminal must be the final record"
            )


def _validate_actor_privileged_separation(record: Order3PolicyTransition) -> None:
    observation = record.runtime_observation
    if any(contact.wrench_world is not None for contact in observation.contact_states):
        raise SchemaValidationError(
            "Order-3 actor RuntimeObservation must not contain measured contact wrench"
        )
    actor_mappings: tuple[dict[str, Any], ...] = (
        observation.controller_status.metrics,
        observation.task_progress.metrics,
        *(contact.metadata for contact in observation.contact_states),
    )
    for mapping in actor_mappings:
        forbidden = _find_forbidden_actor_key(mapping)
        if forbidden is not None:
            raise SchemaValidationError(
                f"Order-3 actor RuntimeObservation contains privileged field {forbidden!r}"
            )


def _find_forbidden_actor_key(value: Any) -> str | None:
    if isinstance(value, dict):
        for raw_key, item in value.items():
            key = str(raw_key).strip().lower()
            if key in _FORBIDDEN_ACTOR_KEYS or key.startswith("privileged_") or key.startswith(
                "ground_truth_"
            ):
                return key
            nested = _find_forbidden_actor_key(item)
            if nested is not None:
                return nested
    elif isinstance(value, (list, tuple)):
        for item in value:
            nested = _find_forbidden_actor_key(item)
            if nested is not None:
                return nested
    return None


def _grouped_sorted_transitions(
    records: list[Order3PolicyTransition],
) -> dict[DatasetSplit, list[Order3PolicyTransition]]:
    grouped = {split: [] for split in DatasetSplit}
    for record in records:
        grouped[record.split].append(record)
    for split in DatasetSplit:
        grouped[split].sort(
            key=lambda item: (
                item.structural_hash,
                item.episode_id,
                item.step_index,
                item.time_s,
                item.graph_id,
            )
        )
    return grouped


def _record_is_real_isaac(record: Order3PolicyTransition) -> bool:
    value = float(record.metrics.get("isaac_backed", 0.0))
    if not math.isfinite(value):
        raise SchemaValidationError("Order-3 isaac_backed metric must be finite")
    return value > 0.5


def _real_isaac_episode_counts(
    records: list[Order3PolicyTransition],
) -> dict[str, int]:
    episode_flags: dict[tuple[DatasetSplit, str], bool] = {}
    for record in records:
        episode_flags[(record.split, record.episode_id)] = _record_is_real_isaac(record)
    return {
        split.value: sum(
            is_real
            for (episode_split, _), is_real in episode_flags.items()
            if episode_split == split
        )
        for split in DatasetSplit
    }


def _module_count_counts(records: list[Order3PolicyTransition]) -> dict[str, dict[str, int]]:
    morphology_counts: dict[DatasetSplit, dict[int, set[str]]] = {
        split: {module_count: set() for module_count in range(2, 9)}
        for split in DatasetSplit
    }
    for record in records:
        module_count = len(record.runtime_observation.morphology_graph.modules)
        morphology_counts[record.split][module_count].add(record.structural_hash)
    return {
        split.value: {
            str(module_count): len(morphology_counts[split][module_count])
            for module_count in range(2, 9)
        }
        for split in DatasetSplit
    }


def _transition_module_count_counts(
    records: list[Order3PolicyTransition],
) -> dict[str, dict[str, int]]:
    return {
        split.value: {
            str(module_count): sum(
                record.split == split
                and len(record.runtime_observation.morphology_graph.modules) == module_count
                for record in records
            )
            for module_count in range(2, 9)
        }
        for split in DatasetSplit
    }


def _validate_manifest_contract_metadata(manifest: Order3DatasetManifest) -> None:
    expected = {
        "writer_version": ORDER3_DATASET_WRITER_VERSION,
        "split_unit": "canonical_structural_hash",
        "actor_observation_contract": _ACTOR_OBSERVATION_CONTRACT,
        "privileged_input_contract": _PRIVILEGED_INPUT_CONTRACT,
        "p4_full_completion_claim": False,
        "legacy_p4_3_artifact_reused": False,
    }
    for key, value in expected.items():
        if manifest.metadata.get(key) != value:
            raise SchemaValidationError(
                f"Order3DatasetManifest metadata {key!r} does not match the v2 contract"
            )
    if manifest.actor_privileged_wrench_inputs:
        raise SchemaValidationError("Order-3 manifest leaks privileged wrench to actor")


def _require_existing_order3_dataset(destination: Path) -> None:
    manifest_path = destination / _MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileExistsError(
            "overwrite=True requires an existing validated Order-3 dataset manifest"
        )
    # Load first so unrelated/corrupt directories cannot be claimed by overwrite.
    load_order3_dataset(manifest_path)


def _existing_manifest_shard_paths(destination: Path) -> set[str]:
    manifest_path = destination / _MANIFEST_NAME
    if not manifest_path.is_file():
        return set()
    manifest = Order3DatasetManifest.from_json(manifest_path.read_text(encoding="utf-8"))
    return {
        relative_path
        for paths in manifest.transition_shards.values()
        for relative_path in paths
    }


def _reject_legacy_p4_3_path(path: Path) -> None:
    resolved = path.resolve(strict=False)
    if resolved == _LEGACY_P4_3_ROOT or _is_relative_to(resolved, _LEGACY_P4_3_ROOT):
        raise SchemaValidationError(
            "Order-3 v2 datasets must not read from or write into legacy artifacts/p4_3"
        )


def _safe_dataset_child(root: Path, relative_path: str) -> Path:
    candidate_value = Path(relative_path)
    if candidate_value.is_absolute() or ".." in candidate_value.parts:
        raise SchemaValidationError(f"unsafe Order-3 shard path: {relative_path!r}")
    root_resolved = root.resolve(strict=False)
    candidate = (root / candidate_value).resolve(strict=False)
    if candidate == root_resolved or not _is_relative_to(candidate, root_resolved):
        raise SchemaValidationError(f"Order-3 shard escapes dataset root: {relative_path!r}")
    return candidate


def _atomic_write_bytes(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
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


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
