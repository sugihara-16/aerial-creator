from __future__ import annotations

"""Immutable, hash-bound phase reset bank for the Order 9 Isaac hot path.

An arbitrary morphology must never inherit the canonical three-module
``q_close`` pose.  Phase zero may start from a deterministic collision-free
initial state.  Every later phase is unlocked only by a physical transition
that reached that boundary successfully for the same topology/object bucket.
Only simulator state is retained; reward, success, and contact labels are
discarded and recomputed after restore.
"""

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from amsrr.schemas.common import SchemaBase, SchemaValidationError, StrEnum
from amsrr.simulation.order9_object_task_state import Order9IsaacStateSnapshot
from amsrr.utils.hashing import hash_file, stable_hash


ORDER9_PHASE_RESET_BANK_VERSION = "order9_phase_reset_bank_v1"
ORDER9_PHASE_RESET_ENTRY_VERSION = "order9_phase_reset_entry_v1"


class Order9PhaseResetOrigin(StrEnum):
    INITIAL_STATE = "initial_state"
    SUCCESSFUL_TRANSITION = "successful_transition"
    ACCEPTED_ORDER8 = "accepted_order8"


@dataclass
class Order9PhaseResetProvenance(SchemaBase):
    origin: Order9PhaseResetOrigin
    source_artifact_path: str
    source_artifact_sha256: str
    source_snapshot_sha256: str
    controller_config_sha256: str
    trajectory_sha256: str
    policy_checkpoint_sha256_by_family: dict[str, str] = field(default_factory=dict)
    success_evidence_sha256: str | None = None
    physical_state_only: bool = True
    state_labels_reused: bool = False

    def validate(self) -> None:
        if not self.source_artifact_path:
            raise SchemaValidationError(
                "Order9 phase-reset source artifact path must be non-empty"
            )
        for name in (
            "source_artifact_sha256",
            "source_snapshot_sha256",
            "controller_config_sha256",
            "trajectory_sha256",
        ):
            _require_sha256(str(getattr(self, name)), name)
        for family, digest in self.policy_checkpoint_sha256_by_family.items():
            if family not in {"pi_l", "pi_h", "pi_d"}:
                raise SchemaValidationError(
                    "Order9 phase-reset policy family is invalid"
                )
            _require_sha256(digest, f"policy_checkpoint_sha256_by_family.{family}")
        if self.origin == Order9PhaseResetOrigin.SUCCESSFUL_TRANSITION:
            if self.success_evidence_sha256 is None:
                raise SchemaValidationError(
                    "successful phase reset requires success-evidence hash"
                )
            _require_sha256(self.success_evidence_sha256, "success_evidence_sha256")
        elif self.success_evidence_sha256 is not None:
            _require_sha256(self.success_evidence_sha256, "success_evidence_sha256")
        if not self.physical_state_only or self.state_labels_reused:
            raise SchemaValidationError(
                "Order9 phase reset may retain physical state only"
            )


@dataclass
class Order9PhaseResetEntry(SchemaBase):
    entry_id: str
    topology_structural_hash: str
    topology_graph_hash: str
    object_condition_hash: str
    phase_index: int
    generation_index: int
    snapshot: Order9IsaacStateSnapshot
    provenance: Order9PhaseResetProvenance
    entry_version: str = ORDER9_PHASE_RESET_ENTRY_VERSION

    def validate(self) -> None:
        if self.entry_version != ORDER9_PHASE_RESET_ENTRY_VERSION:
            raise SchemaValidationError("Order9 phase-reset entry version mismatch")
        if not self.entry_id:
            raise SchemaValidationError("Order9 phase-reset entry id is empty")
        for name in (
            "topology_structural_hash",
            "topology_graph_hash",
            "object_condition_hash",
        ):
            _require_sha256(str(getattr(self, name)), name)
        if self.phase_index < 0 or self.generation_index < 0:
            raise SchemaValidationError(
                "Order9 phase-reset indices must be non-negative"
            )
        self.snapshot.validate()
        self.provenance.validate()
        if self.snapshot.phase_index != self.phase_index:
            raise SchemaValidationError(
                "Order9 phase-reset snapshot phase identity differs"
            )
        if not math.isclose(self.snapshot.phase_elapsed_s, 0.0, abs_tol=1.0e-9):
            raise SchemaValidationError(
                "Order9 phase-reset snapshot must be captured at phase start"
            )
        if self.provenance.source_snapshot_sha256 != self.snapshot.metadata.get(
            "source_snapshot_sha256"
        ):
            raise SchemaValidationError(
                "Order9 phase-reset sanitized snapshot provenance differs"
            )
        if self.phase_index > 0 and self.provenance.origin not in {
            Order9PhaseResetOrigin.SUCCESSFUL_TRANSITION,
            Order9PhaseResetOrigin.ACCEPTED_ORDER8,
        }:
            raise SchemaValidationError(
                "later Order9 phases require observed successful-transition state"
            )

    @property
    def entry_hash(self) -> str:
        self.validate()
        return stable_hash(self.to_dict())


@dataclass
class Order9PhaseResetBank(SchemaBase):
    bank_id: str
    topology_structural_hash: str
    topology_graph_hash: str
    object_condition_hash: str
    object_id: str
    joint_names: list[str]
    phase_count: int
    entries: list[Order9PhaseResetEntry] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    bank_version: str = ORDER9_PHASE_RESET_BANK_VERSION

    def validate(self) -> None:
        if self.bank_version != ORDER9_PHASE_RESET_BANK_VERSION:
            raise SchemaValidationError("Order9 phase-reset bank version mismatch")
        if not self.bank_id or not self.object_id:
            raise SchemaValidationError(
                "Order9 phase-reset bank/object ids must be non-empty"
            )
        for name in (
            "topology_structural_hash",
            "topology_graph_hash",
            "object_condition_hash",
        ):
            _require_sha256(str(getattr(self, name)), name)
        if self.phase_count < 1:
            raise SchemaValidationError(
                "Order9 phase-reset phase count must be positive"
            )
        if (
            not self.joint_names
            or len(self.joint_names) != len(set(self.joint_names))
            or any(not value for value in self.joint_names)
        ):
            raise SchemaValidationError(
                "Order9 phase-reset joint names must be unique and non-empty"
            )
        entry_ids: set[str] = set()
        entry_hashes: set[str] = set()
        for entry in self.entries:
            entry.validate()
            if not 0 <= entry.phase_index < self.phase_count:
                raise SchemaValidationError(
                    "Order9 phase-reset entry phase lies outside bank"
                )
            if (
                entry.topology_structural_hash != self.topology_structural_hash
                or entry.topology_graph_hash != self.topology_graph_hash
                or entry.object_condition_hash != self.object_condition_hash
            ):
                raise SchemaValidationError(
                    "Order9 phase-reset entry belongs to another bucket"
                )
            if entry.snapshot.object_id != self.object_id:
                raise SchemaValidationError(
                    "Order9 phase-reset entry object identity differs"
                )
            if entry.snapshot.joint_names != self.joint_names:
                raise SchemaValidationError(
                    "Order9 phase-reset entry joint identity differs"
                )
            if entry.entry_id in entry_ids or entry.entry_hash in entry_hashes:
                raise SchemaValidationError(
                    "Order9 phase-reset bank contains a duplicate entry"
                )
            entry_ids.add(entry.entry_id)
            entry_hashes.add(entry.entry_hash)
        _require_json_finite(self.metadata, "metadata")
        if self.metadata.get("state_labels_reused") not in {None, False}:
            raise SchemaValidationError(
                "Order9 phase-reset bank must not reuse state labels"
            )

    @property
    def bank_hash(self) -> str:
        self.validate()
        return stable_hash(self.to_dict())

    @property
    def available_phase_indices(self) -> tuple[int, ...]:
        self.validate()
        return tuple(sorted({entry.phase_index for entry in self.entries}))

    def entries_for_phase(self, phase_index: int) -> tuple[Order9PhaseResetEntry, ...]:
        self.validate()
        if not 0 <= phase_index < self.phase_count:
            raise SchemaValidationError("Order9 reset phase index is invalid")
        return tuple(
            sorted(
                (entry for entry in self.entries if entry.phase_index == phase_index),
                key=lambda item: (item.generation_index, item.entry_id),
            )
        )

    def select_snapshot(
        self, phase_index: int, *, seed: int, selection_index: int
    ) -> Order9IsaacStateSnapshot:
        if seed < 0 or selection_index < 0:
            raise SchemaValidationError(
                "Order9 reset selection indices must be non-negative"
            )
        candidates = self.entries_for_phase(phase_index)
        if not candidates:
            raise SchemaValidationError(
                f"Order9 phase {phase_index} is not unlocked in reset bank"
            )
        digest = stable_hash(
            {
                "bank_hash": self.bank_hash,
                "phase_index": phase_index,
                "seed": seed,
                "selection_index": selection_index,
            }
        )
        selected = candidates[int(digest[:16], 16) % len(candidates)]
        snapshot = Order9IsaacStateSnapshot.from_dict(selected.snapshot.to_dict())
        snapshot.validate()
        return snapshot

    def with_entry(self, entry: Order9PhaseResetEntry) -> "Order9PhaseResetBank":
        candidate = Order9PhaseResetBank.from_dict(self.to_dict())
        candidate.entries.append(
            Order9PhaseResetEntry.from_dict(entry.to_dict())
        )
        candidate.validate()
        return candidate


def build_order9_phase_reset_entry(
    *,
    entry_id: str,
    topology_structural_hash: str,
    topology_graph_hash: str,
    object_condition_hash: str,
    phase_index: int,
    generation_index: int,
    snapshot: Order9IsaacStateSnapshot,
    origin: Order9PhaseResetOrigin,
    source_artifact_path: str | Path,
    expected_source_artifact_sha256: str,
    controller_config_sha256: str,
    trajectory_sha256: str,
    policy_checkpoint_sha256_by_family: Mapping[str, str] | None = None,
    transition_accepted: bool = False,
    success_evidence_sha256: str | None = None,
) -> Order9PhaseResetEntry:
    """Sanitize one boundary snapshot before admitting it to a reset bank."""

    snapshot.validate()
    source = Path(source_artifact_path).resolve()
    if hash_file(source) != expected_source_artifact_sha256:
        raise SchemaValidationError(
            "Order9 phase-reset source artifact hash mismatch"
        )
    if phase_index == 0:
        if origin == Order9PhaseResetOrigin.SUCCESSFUL_TRANSITION:
            raise SchemaValidationError(
                "Order9 phase zero is an initial state, not a transition result"
            )
    elif origin == Order9PhaseResetOrigin.SUCCESSFUL_TRANSITION:
        if not transition_accepted:
            raise SchemaValidationError(
                "Order9 rejected transition cannot populate phase-reset bank"
            )
        if success_evidence_sha256 is None:
            raise SchemaValidationError(
                "Order9 successful transition requires evidence hash"
            )
    elif origin != Order9PhaseResetOrigin.ACCEPTED_ORDER8:
        raise SchemaValidationError(
            "Order9 later phase requires successful transition provenance"
        )
    if snapshot.phase_index != phase_index:
        raise SchemaValidationError(
            "Order9 phase-reset source snapshot phase differs"
        )
    source_snapshot_sha256 = snapshot.snapshot_hash
    sanitized = Order9IsaacStateSnapshot.from_dict(snapshot.to_dict())
    sanitized.phase_elapsed_s = 0.0
    sanitized.metadata = {
        "physical_state_only": True,
        "state_labels_reused": False,
        "source_snapshot_sha256": source_snapshot_sha256,
    }
    sanitized.validate()
    provenance = Order9PhaseResetProvenance(
        origin=origin,
        source_artifact_path=str(source),
        source_artifact_sha256=expected_source_artifact_sha256,
        source_snapshot_sha256=source_snapshot_sha256,
        controller_config_sha256=controller_config_sha256,
        trajectory_sha256=trajectory_sha256,
        policy_checkpoint_sha256_by_family=dict(
            policy_checkpoint_sha256_by_family or {}
        ),
        success_evidence_sha256=success_evidence_sha256,
        physical_state_only=True,
        state_labels_reused=False,
    )
    entry = Order9PhaseResetEntry(
        entry_id=entry_id,
        topology_structural_hash=topology_structural_hash,
        topology_graph_hash=topology_graph_hash,
        object_condition_hash=object_condition_hash,
        phase_index=phase_index,
        generation_index=generation_index,
        snapshot=sanitized,
        provenance=provenance,
    )
    entry.validate()
    return entry


def write_order9_phase_reset_bank(
    path: str | Path, bank: Order9PhaseResetBank
) -> str:
    bank.validate()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(bank.to_json(indent=2))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return hash_file(target)


def load_order9_phase_reset_bank(
    path: str | Path, *, expected_sha256: str
) -> Order9PhaseResetBank:
    source = Path(path)
    if hash_file(source) != expected_sha256:
        raise SchemaValidationError("Order9 phase-reset bank file hash mismatch")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaValidationError(f"failed to load Order9 reset bank: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise SchemaValidationError("Order9 phase-reset bank root must be a mapping")
    bank = Order9PhaseResetBank.from_dict(dict(payload))
    bank.validate()
    return bank


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise SchemaValidationError(f"Order9 {label} must be a SHA-256 digest")


def _require_json_finite(value: object, label: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SchemaValidationError(f"Order9 {label} contains non-finite data")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SchemaValidationError(f"Order9 {label} keys must be strings")
            _require_json_finite(item, f"{label}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _require_json_finite(item, f"{label}[{index}]")
        return
    raise SchemaValidationError(f"Order9 {label} contains non-JSON data")


__all__ = [
    "ORDER9_PHASE_RESET_BANK_VERSION",
    "ORDER9_PHASE_RESET_ENTRY_VERSION",
    "Order9PhaseResetBank",
    "Order9PhaseResetEntry",
    "Order9PhaseResetOrigin",
    "Order9PhaseResetProvenance",
    "build_order9_phase_reset_entry",
    "load_order9_phase_reset_bank",
    "write_order9_phase_reset_bank",
]
