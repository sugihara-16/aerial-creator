from __future__ import annotations

import os
from pathlib import Path
import sys
import time

import pytest

from amsrr.training import order9_r1_nominal_calibration_v6 as nominal_v6
from amsrr.training.order9_r1_safe_timing import (
    order9_r1_safe_phase_time_scales,
)
from amsrr.utils.hashing import hash_file
from scripts import order9_run_r1_nominal_calibration_v6 as runner_v6

REPOSITORY = Path(__file__).resolve().parents[3]


def test_r1_nominal_v6_contract_repairs_retreat_without_changing_grasp() -> None:
    protocol, approval = nominal_v6.load_order9_r1_nominal_calibration_v6_contract(
        REPOSITORY
    )

    assert approval["decision"] == "approved"
    assert protocol["grasp_geometric_path_changed"] is False
    assert protocol["post_release_retreat_path_changed"] is True
    assert protocol["retreat_offset_m"] == 0.10
    assert protocol["grasp_endpoint_configuration_changed"] is False
    assert protocol["final_endpoint_configuration_changed"] is True
    assert protocol["contact_assignments_changed"] is False
    assert protocol["retreat_isaac_validated"] is True
    assert protocol["complete_task_semantic_audit_required"] is True
    assert (
        protocol["complete_task_materializer_selection"]
        == "explicit_callable_and_identity_v1"
    )
    assert protocol["complete_task_materializer_monkey_patch_forbidden"] is True


def test_r1_nominal_v6_profiles_have_direct_isaac_evidence() -> None:
    protocol, _approval = nominal_v6.load_order9_r1_nominal_calibration_v6_contract(
        REPOSITORY
    )

    evidence = protocol["profile_isaac_evidence"]
    assert {entry["module_count"] for entry in evidence} == set(range(2, 9))
    for entry in evidence:
        assert entry["phase_time_scales"] == order9_r1_safe_phase_time_scales(
            entry["module_count"]
        )
    assert protocol["profile_admission_isaac_episode_count"] == 14
    assert protocol["profile_admission_isaac_success_count"] == 14
    assert protocol["profile_admission_isaac_safety_failure_count"] == 0
    assert protocol["profile_admission_isaac_fallback_count"] == 0
    assert protocol["profile_admission_is_formal_calibration_result"] is False


def test_r1_nominal_v6_preserves_controller_and_completion_gates() -> None:
    protocol, _approval = nominal_v6.load_order9_r1_nominal_calibration_v6_contract(
        REPOSITORY
    )

    assert protocol["maximum_wall_time_s"] == 36000
    assert protocol["maximum_parallel_process_count"] == 4
    assert protocol["isaac_required_after_fast_screen"] is True
    assert protocol["replay_count_per_screen_accepted_candidate"] == 2
    assert protocol["pi_l_system_retained"] is True
    assert protocol["pi_l_actor_command_applied"] is False
    assert protocol["nominal_preload_applied"] is True
    assert protocol["qpid_qp_applied"] is True
    assert protocol["local_servo_applied"] is True
    assert protocol["formal_teacher_collection_authorized_by_protocol_alone"] is False


def test_r1_nominal_v6_preserves_protected_c3_bytes() -> None:
    protocol, _approval = nominal_v6.load_order9_r1_nominal_calibration_v6_contract(
        REPOSITORY
    )

    for key in (
        "protected_c3_curriculum",
        "protected_c3_checkpoint",
        "protected_c3_rollout",
        "protected_c3_release_ledger",
    ):
        binding = protocol[key]
        assert hash_file(REPOSITORY / binding["path"]) == binding["sha256"]


def test_r1_nominal_v6_execution_window_is_persistent_across_resume(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "selection" / "evidence.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("{}\n", encoding="utf-8")
    historical_start = time.time() - 30.0
    evidence.touch()
    evidence.chmod(0o600)
    os.utime(evidence, (historical_start, historical_start))

    first = runner_v6._load_or_create_execution_window(
        tmp_path,
        maximum_wall_time_s=36000.0,
    )
    second = runner_v6._load_or_create_execution_window(
        tmp_path,
        maximum_wall_time_s=36000.0,
    )

    assert first == second
    assert first["resume_resets_window"] is False
    assert float(first["started_unix_s"]) == pytest.approx(historical_start)
    assert float(first["deadline_unix_s"]) == pytest.approx(historical_start + 36000.0)


def test_r1_nominal_v6_deadline_stops_active_collector_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner_v6, "_V6_DEADLINE_UNIX_S", time.time() + 0.25)
    log = tmp_path / "collector.log"
    started = time.monotonic()

    with pytest.raises(runner_v6.Order9R1V6WallTimeExceeded):
        runner_v6._run_parallel_order9_collectors_with_deadline(
            {"probe": [sys.executable, "-c", "import time; time.sleep(30)"]},
            repository_root=tmp_path,
            log_paths={"probe": log},
            maximum_parallel_process_count=1,
        )

    assert time.monotonic() - started < 3.0
    assert "ORDER9_COMMAND=" in log.read_text(encoding="utf-8")


def test_r1_nominal_v6_execution_priority_checks_corners_first() -> None:
    class _Bucket:
        bucket_id = "train-probe"

    class _Case:
        source_bucket = _Bucket()
        sample_kind = "lattice"

        def __init__(self, sample_index: int) -> None:
            self.sample_index = sample_index

    ordered = sorted(
        (_Case(index) for index in range(27)),
        key=runner_v6._v6_execution_case_priority,
    )

    assert [case.sample_index for case in ordered[:8]] == [0, 2, 6, 8, 18, 20, 24, 26]
    assert ordered[-1].sample_index == 13


def test_r1_nominal_v6_preexisting_lattice_failure_is_decisive() -> None:
    class _Replay:
        success = False
        safety_failure = False
        fallback_used = False

    assert runner_v6._is_decisive_preexisting_lattice_failure(
        "candidate__lattice_02",
        (_Replay(),),
    )
    assert not runner_v6._is_decisive_preexisting_lattice_failure(
        "candidate__interior_02",
        (_Replay(),),
    )
