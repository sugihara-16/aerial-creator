from __future__ import annotations

from scripts import order9_run_r1_yaw_repair_bounded_validation_v9 as runner


def test_yaw_repair_jobs_are_partitioned_into_bounded_scenes() -> None:
    chunks = runner._job_chunks(runner._source_payload())

    assert [len(chunk) for chunk in chunks] == [4, 4, 1]
    assert sum(map(len, chunks)) == runner.EXPECTED_CANDIDATE_COUNT
    assert max(map(len, chunks)) == runner.MAXIMUM_CANDIDATES_PER_SCENE


def test_yaw_repair_bounded_result_revalidates_raw_isaac_evidence() -> None:
    entries = runner.validate_result_ledger()

    assert len(entries) == runner.EXPECTED_CANDIDATE_COUNT
    assert sum(entry["episode_count"] for entry in entries) == 18
    assert sum(entry["success_count"] for entry in entries) == 18
    assert all(entry["pi_l_actor_command_applied"] is False for entry in entries)
    assert all(entry["qpid_qp_applied"] is True for entry in entries)
    assert all(entry["local_servo_applied"] is True for entry in entries)
