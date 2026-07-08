from __future__ import annotations

from pathlib import Path

from amsrr.acceptance import P4_0AcceptanceCriteria, run_p4_0_acceptance
from amsrr.logging import read_episode_archives_jsonl
from amsrr.schemas.task_spec import TaskSpec


def test_p4_0_acceptance_simplified_full_pipeline(
    grasp_carry_dict: dict,
    tmp_path: Path,
) -> None:
    base_task = TaskSpec.from_dict(grasp_carry_dict)
    archive_path = tmp_path / "p4_0_acceptance.jsonl"

    report = run_p4_0_acceptance(
        base_task,
        criteria=P4_0AcceptanceCriteria(
            episode_count=12,
            seed=1200,
            source_hash="acceptance-test",
        ),
        archive_path=archive_path,
    )

    assert report.passed, report.failure_reasons
    assert report.episode_count == 12
    assert report.crash_count == 0
    assert report.success_rate == 1.0
    assert report.object_drop_rate == 0.0
    assert report.collision_rate == 0.0
    assert report.qp_infeasible_rate == 0.0
    assert report.archive_count == 12
    assert report.archive_roundtrip_count == 12
    assert report.p2_selected_design_used is True
    assert report.p3_assembly_result_used is True
    assert report.fixed_simple_design_policy_absent is True
    assert report.contact_candidates_generated is True
    assert report.trajectory_generated is True
    assert report.policy_commands_generated is True
    assert report.controller_commands_generated is True
    assert report.archive_complete is True
    assert report.simplified_metrics_recorded is True
    assert report.no_mislabeling_passed is True
    assert report.report_declares_simplified_backend is True
    assert "not Isaac-backed physical success rates" in report.backend_note

    archives = read_episode_archives_jsonl(archive_path)
    assert len(archives) == 12
    assert archives[0].rollout_artifacts["is_p4_full_completion"] is False
    assert archives[0].rollout_artifacts["isaac_backed"] is False
    assert archives[0].rollout_artifacts["physical_success_claim"] is False
    assert archives[0].reproducibility["source_hash"] == "acceptance-test"
    assert type(report).from_json(report.to_json()).to_dict() == report.to_dict()
