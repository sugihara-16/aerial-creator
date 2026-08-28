from __future__ import annotations

import json
from pathlib import Path

from amsrr.training import order9_r1_nominal_calibration_v5 as nominal_v5
from amsrr.training.order9_r1_early_tilt_pipeline import (
    _load_teacher_surface_overrides,
)
from amsrr.utils.hashing import hash_file


REPOSITORY = Path(__file__).resolve().parents[3]


def test_r1_nominal_v5_contract_preserves_gates_and_ten_hour_bound() -> None:
    protocol, approval = nominal_v5.load_order9_r1_nominal_calibration_v5_contract(
        REPOSITORY
    )

    assert approval["decision"] == "approved"
    assert protocol["maximum_wall_time_s"] == 36000
    assert protocol["isaac_required_after_fast_screen"] is True
    assert protocol["replay_count_per_screen_accepted_candidate"] == 2
    assert protocol["pi_l_system_retained"] is True
    assert protocol["pi_l_actor_command_applied"] is False
    assert protocol["formal_teacher_collection_authorized_by_protocol_alone"] is False


def test_r1_nominal_v5_retime_is_geometrically_invariant() -> None:
    protocol, _approval = nominal_v5.load_order9_r1_nominal_calibration_v5_contract(
        REPOSITORY
    )

    assert protocol["phase_time_scales"] == {
        "approach": 0.5,
        "contact_acquisition": 0.5,
        "lift": 0.1,
        "transport": 0.1,
        "place": 0.1,
        "release": 0.1,
        "retreat": 0.1,
        "settle": 0.1,
    }
    assert protocol["geometric_path_changed"] is False
    assert protocol["endpoint_configuration_changed"] is False
    assert protocol["contact_assignments_changed"] is False
    assert protocol["joint_rate_limit_rechecked"] is True


def test_r1_nominal_v5_keeps_v4_override_bytes_and_adds_v2_candidate() -> None:
    v1 = REPOSITORY / "configs/training/order9_r1_teacher_overrides_v1.json"
    v2 = REPOSITORY / "configs/training/order9_r1_teacher_overrides_v2.json"

    assert hash_file(v1) == "a76f9e90efe3c9bf36a8e864f5640167a41a9210ee024c63ba3b075a6a32ae07"
    _bucket, candidates = _load_teacher_surface_overrides(v2)
    options = candidates[
        "r1_l1_10mm_5deg__train__train-000003-3b95da871f01__lattice_00"
    ]
    assert options[0][:2] == ((10, 18), "slot_0:grasp_pair:3")
    payload = json.loads(v2.read_text(encoding="utf-8"))
    assert payload["base_v1_sha256"] == hash_file(v1)
