from __future__ import annotations

from pathlib import Path

from amsrr.acceptance import P3AcceptanceCriteria, run_p3_acceptance
from amsrr.logging import read_episode_archives_jsonl
from amsrr.schemas.task_spec import TaskSpec


def test_p3_acceptance_section_24_4(
    grasp_carry_dict: dict,
    tmp_path: Path,
) -> None:
    base_task = TaskSpec.from_dict(grasp_carry_dict)
    archive_path = tmp_path / "p3_acceptance.jsonl"

    report = run_p3_acceptance(
        base_task,
        criteria=P3AcceptanceCriteria(
            episode_count=1000,
            min_assembly_success_rate=0.70,
            seed=700,
            source_hash="acceptance-test",
        ),
        archive_path=archive_path,
    )

    assert report.passed, report.failure_reasons
    assert report.episode_count == 1000
    assert report.assembly_success_rate >= 0.70
    assert report.crash_count == 0
    assert report.state_match_count == report.assembly_success_count
    assert report.retry_path_tested is True
    assert report.retry_probe_success is True
    assert report.retry_probe_retry_count > 0
    assert report.abort_path_tested is True
    assert report.abort_probe_aborted is True
    assert report.abort_probe_abort_count > 0
    assert report.archive_count == 1000
    assert report.archive_roundtrip_count == 1000
    assert archive_path.exists()

    archives = read_episode_archives_jsonl(archive_path)
    assert len(archives) == 1000
    assert archives[0].assembly_plan is not None
    assert archives[0].metrics["assembly_state_matches_target"] == 1.0
    assert archives[0].reproducibility["source_hash"] == "acceptance-test"
    assert type(report).from_json(report.to_json()).to_dict() == report.to_dict()
