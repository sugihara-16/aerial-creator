from __future__ import annotations

from pathlib import Path

from amsrr.acceptance import P2AcceptanceCriteria, P2CompletionCriteria, run_p2_completion
from amsrr.logging import read_episode_archives_jsonl
from amsrr.schemas.task_spec import TaskSpec


def test_p2_completion_milestone_section_24_3(
    grasp_carry_dict: dict,
    tmp_path: Path,
) -> None:
    base_task = TaskSpec.from_dict(grasp_carry_dict)
    archive_path = tmp_path / "p2_completion.jsonl"

    report = run_p2_completion(
        base_task,
        criteria=P2CompletionCriteria(
            acceptance_criteria=P2AcceptanceCriteria(
                episode_count=1000,
                min_valid_design_rate=0.70,
                min_required_slot_coverage=0.90,
                seed=500,
                source_hash="completion-test",
            ),
        ),
        archive_path=archive_path,
    )

    assert report.passed, report.failure_reasons
    assert report.phase_label == "P2"
    assert all(report.completion_checks.values())
    assert report.acceptance_report.passed is True
    assert report.acceptance_report.valid_design_rate >= 0.70
    assert report.acceptance_report.accepted_required_slot_coverage_min >= 0.90
    assert report.acceptance_report.closed_loop_invalid_rejected is True
    assert report.acceptance_report.feasibility_label_valid_count == report.acceptance_report.archive_count
    assert report.acceptance_report.archive_roundtrip_count == 1000
    assert archive_path.exists()

    archives = read_episode_archives_jsonl(archive_path)
    assert len(archives) == 1000
    assert archives[0].reproducibility["source_hash"] == "completion-test"
    assert type(report).from_json(report.to_json()).to_dict() == report.to_dict()
