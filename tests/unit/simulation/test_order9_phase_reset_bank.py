from __future__ import annotations

from pathlib import Path

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.simulation.order9_object_task_state import Order9IsaacStateSnapshot
from amsrr.simulation.order9_phase_reset_bank import (
    Order9PhaseResetBank,
    Order9PhaseResetOrigin,
    build_order9_phase_reset_entry,
    load_order9_phase_reset_bank,
    write_order9_phase_reset_bank,
)
from amsrr.utils.hashing import hash_file, stable_hash


_HASH_A = "a" * 64
_HASH_B = "b" * 64
_HASH_C = "c" * 64
_HASH_D = "d" * 64


def _snapshot(phase_index: int) -> Order9IsaacStateSnapshot:
    snapshot = Order9IsaacStateSnapshot(
        simulation_time_s=float(phase_index),
        robot_root_pose_world=[0.0, 0.0, 0.65, 0.0, 0.0, 0.0, 1.0],
        robot_root_twist_world=[0.0] * 6,
        joint_names=["module_0__joint"],
        joint_positions_rad=[0.1 * phase_index],
        joint_velocities_radps=[0.0],
        object_id="object-0",
        object_pose_world=[0.55, 0.0, 0.225, 0.0, 0.0, 0.0, 1.0],
        object_twist_world=[0.0] * 6,
        phase_index=phase_index,
        phase_elapsed_s=0.0,
        command_index=phase_index,
        metadata={
            "success": True,
            "raw_contact_force_n": 12.0,
        },
    )
    snapshot.validate()
    return snapshot


def _source(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "raw_isaac.json"
    path.write_text('{"raw":true}\n', encoding="utf-8")
    return path, hash_file(path)


def _entry(tmp_path: Path, phase_index: int, generation_index: int = 0):
    source, digest = _source(tmp_path)
    successful = phase_index > 0
    return build_order9_phase_reset_entry(
        entry_id=f"phase-{phase_index}-{generation_index}",
        topology_structural_hash=_HASH_A,
        topology_graph_hash=_HASH_B,
        object_condition_hash=_HASH_C,
        phase_index=phase_index,
        generation_index=generation_index,
        snapshot=_snapshot(phase_index),
        origin=(
            Order9PhaseResetOrigin.SUCCESSFUL_TRANSITION
            if successful
            else Order9PhaseResetOrigin.INITIAL_STATE
        ),
        source_artifact_path=source,
        expected_source_artifact_sha256=digest,
        controller_config_sha256=_HASH_D,
        trajectory_sha256=stable_hash({"trajectory": 1}),
        policy_checkpoint_sha256_by_family={"pi_l": "e" * 64},
        transition_accepted=successful,
        success_evidence_sha256=("f" * 64 if successful else None),
    )


def _bank(entries) -> Order9PhaseResetBank:
    bank = Order9PhaseResetBank(
        bank_id="unit-bank",
        topology_structural_hash=_HASH_A,
        topology_graph_hash=_HASH_B,
        object_condition_hash=_HASH_C,
        object_id="object-0",
        joint_names=["module_0__joint"],
        phase_count=8,
        entries=list(entries),
        metadata={"state_labels_reused": False},
    )
    bank.validate()
    return bank


def test_phase_reset_entry_discards_success_and_contact_labels(tmp_path: Path) -> None:
    entry = _entry(tmp_path, 1)

    assert entry.snapshot.metadata == {
        "physical_state_only": True,
        "state_labels_reused": False,
        "source_snapshot_sha256": entry.provenance.source_snapshot_sha256,
    }
    assert entry.provenance.state_labels_reused is False
    assert entry.provenance.physical_state_only is True


def test_later_phase_rejects_unobserved_initial_state(tmp_path: Path) -> None:
    source, digest = _source(tmp_path)
    with pytest.raises(SchemaValidationError, match="successful transition"):
        build_order9_phase_reset_entry(
            entry_id="invalid",
            topology_structural_hash=_HASH_A,
            topology_graph_hash=_HASH_B,
            object_condition_hash=_HASH_C,
            phase_index=2,
            generation_index=0,
            snapshot=_snapshot(2),
            origin=Order9PhaseResetOrigin.INITIAL_STATE,
            source_artifact_path=source,
            expected_source_artifact_sha256=digest,
            controller_config_sha256=_HASH_D,
            trajectory_sha256="e" * 64,
        )


def test_phase_reset_bank_selection_is_deterministic_and_phase_locked(
    tmp_path: Path,
) -> None:
    first = _entry(tmp_path, 1, 0)
    second = _entry(tmp_path, 1, 1)
    bank = _bank([first, second])

    selected_a = bank.select_snapshot(1, seed=9009, selection_index=3)
    selected_b = bank.select_snapshot(1, seed=9009, selection_index=3)

    assert selected_a.to_dict() == selected_b.to_dict()
    assert bank.available_phase_indices == (1,)
    with pytest.raises(SchemaValidationError, match="not unlocked"):
        bank.select_snapshot(2, seed=9009, selection_index=0)


def test_phase_reset_bank_rejects_bucket_identity_mismatch(tmp_path: Path) -> None:
    entry = _entry(tmp_path, 0)
    entry.object_condition_hash = "9" * 64
    with pytest.raises(SchemaValidationError, match="another bucket"):
        _bank([entry])


def test_phase_reset_bank_round_trip_is_file_hash_bound(tmp_path: Path) -> None:
    bank = _bank([_entry(tmp_path, 0), _entry(tmp_path, 1)])
    path = tmp_path / "bank.json"
    digest = write_order9_phase_reset_bank(path, bank)

    loaded = load_order9_phase_reset_bank(path, expected_sha256=digest)

    assert loaded.bank_hash == bank.bank_hash
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(SchemaValidationError, match="file hash mismatch"):
        load_order9_phase_reset_bank(path, expected_sha256=digest)
