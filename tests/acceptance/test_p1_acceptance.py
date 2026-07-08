from __future__ import annotations

from pathlib import Path

from amsrr.acceptance import P1AcceptanceCriteria, run_p1_acceptance
from amsrr.logging import read_episode_archives_jsonl
from amsrr.schemas.task_spec import TaskSpec


def test_p1_acceptance_section_24_2(
    grasp_carry_dict: dict,
    tmp_path: Path,
) -> None:
    base_task = TaskSpec.from_dict(grasp_carry_dict)
    archive_path = tmp_path / "p1_acceptance.jsonl"

    report = run_p1_acceptance(
        base_task,
        criteria=P1AcceptanceCriteria(
            episode_count=1000,
            min_success_rate=0.60,
            candidate_sample_count=16,
            seed=300,
            source_hash="acceptance-test",
        ),
        archive_path=archive_path,
    )

    assert report.passed, report.failure_reasons
    assert report.episode_count == 1000
    assert report.success_rate >= 0.60
    assert report.crash_count == 0
    assert report.non_empty_candidate_sample_count == report.candidate_sample_count
    assert min(report.contact_candidate_counts) > 0
    assert report.archive_count == 1000
    assert report.archive_roundtrip_count == 1000
    assert report.metrics["archive_count"] == 1000.0
    assert archive_path.exists()

    archives = read_episode_archives_jsonl(archive_path)
    assert len(archives) == 1000
    assert archives[0].reproducibility["source_hash"] == "acceptance-test"
    assert type(report).from_json(report.to_json()).to_dict() == report.to_dict()
