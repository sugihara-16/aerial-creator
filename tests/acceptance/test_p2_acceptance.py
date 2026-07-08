from __future__ import annotations

from pathlib import Path

from amsrr.acceptance import P2AcceptanceCriteria, run_p2_acceptance
from amsrr.logging import read_episode_archives_jsonl
from amsrr.schemas.task_spec import TaskSpec


def test_p2_acceptance_section_24_3(
    grasp_carry_dict: dict,
    tmp_path: Path,
) -> None:
    base_task = TaskSpec.from_dict(grasp_carry_dict)
    archive_path = tmp_path / "p2_acceptance.jsonl"

    report = run_p2_acceptance(
        base_task,
        criteria=P2AcceptanceCriteria(
            episode_count=1000,
            min_valid_design_rate=0.70,
            min_required_slot_coverage=0.90,
            seed=400,
            source_hash="acceptance-test",
        ),
        archive_path=archive_path,
    )

    assert report.passed, report.failure_reasons
    assert report.episode_count == 1000
    assert report.valid_design_rate >= 0.70
    assert report.accepted_required_slot_coverage_min >= 0.90
    assert report.closed_loop_invalid_rejected is True
    assert report.closed_loop_feasibility_label_stored is True
    assert report.feasibility_label_valid_count == report.feasibility_label_archive_count
    assert report.archive_count == 1000
    assert report.archive_roundtrip_count == 1000
    assert report.metrics["archive_count"] == 1000.0
    assert archive_path.exists()

    archives = read_episode_archives_jsonl(archive_path)
    assert len(archives) == 1000
    assert archives[0].reproducibility["source_hash"] == "acceptance-test"
    assert archives[0].feasibility_result is not None
    assert archives[0].feasibility_result.proxy_scores["L_FEASIBLE"] == 1.0
    assert archives[0].metrics["label_L_FEASIBLE"] == 1.0
    assert type(report).from_json(report.to_json()).to_dict() == report.to_dict()
