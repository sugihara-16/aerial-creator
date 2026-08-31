from __future__ import annotations

from pathlib import Path

from amsrr.training.order9_r1_range_selection_v10 import (
    ORDER9_R1_RANGE_LEVEL_IDS,
)
from amsrr.utils.hashing import hash_file
from scripts import order9_run_r1_range_selection_v10 as runtime_v10
from scripts import order9_run_r1_range_selection_v11 as runtime_v11
from scripts import order9_run_r1_range_selection_v12 as runtime_v12

ROOT = Path(__file__).resolve().parents[3]


def test_v12_formal_contract_is_approved_and_hash_bound() -> None:
    protocol, approval = runtime_v12._load_contract()
    assert tuple(protocol["selection_level_ids"]) == ORDER9_R1_RANGE_LEVEL_IDS
    assert protocol["acceptance_clearance_m"] == 0.0195
    assert protocol["planning_clearance_m"] == 0.0300
    assert protocol["minimum_planning_buffer_m"] == 0.0105
    assert protocol["all_eight_phases_generated_under_constraint"] is True
    assert protocol["uncertified_rigid_pose_copy_rejected"] is True
    assert protocol["reuse_pre_clearance_v11_teacher_rejection"] is True
    assert protocol["formal_teacher_collection_authorized"] is False
    assert approval["learning_authorized"] is False
    assert approval["approved_protocol"]["sha256"] == hash_file(runtime_v12.PROTOCOL)


def test_v12_installs_only_new_preparation_hooks(monkeypatch) -> None:
    original_prepare_level = runtime_v11._prepare_level
    original_run_level = runtime_v11._run_level
    monkeypatch.setattr(runtime_v11, "RUNNER_VERSION", runtime_v11.RUNNER_VERSION)
    monkeypatch.setattr(runtime_v10, "RUNNER_VERSION", runtime_v10.RUNNER_VERSION)
    monkeypatch.setattr(runtime_v11, "_prepare_case", None)
    monkeypatch.setattr(runtime_v11, "_validate_prepared_case", None)
    runtime_v12._install_v12_preparation_hooks()
    assert runtime_v11._prepare_case is runtime_v12._prepare_case
    assert runtime_v11._validate_prepared_case is runtime_v12._validate_prepared_case
    assert runtime_v11._prepare_level is original_prepare_level
    assert runtime_v11._run_level is original_run_level


def test_v12_protected_c3_bindings_remain_current() -> None:
    protocol, _approval = runtime_v12._load_contract()
    for label in (
        "protected_c3_checkpoint",
        "protected_c3_rollout",
        "protected_c3_curriculum",
        "protected_c3_release_ledger",
    ):
        binding = protocol[label]
        assert hash_file(ROOT / binding["path"]) == binding["sha256"]


def test_v12_reuses_only_the_pre_clearance_v11_l2_rejection() -> None:
    protocol, _approval = runtime_v12._load_contract()
    result = runtime_v12._reuse_parent_v11_selection_rejection(protocol)
    assert result["level_id"] == "r1_l2_20mm_10deg"
    assert result["passed"] is False
    assert result["reused_parent_evidence"] is True
    assert result["rejection_precedes_support_clearance_candidate_selection"] is True
    assert result["isaac_invoked"] is False
