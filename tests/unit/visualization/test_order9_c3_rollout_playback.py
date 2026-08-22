from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from amsrr.schemas.common import SchemaValidationError
from amsrr.visualization.order9_c3_rollout_playback import (
    Order9C3RolloutPlaybackCase,
    load_order9_c3_evaluation_episode_rows,
    order9_c3_first_episode_indices,
    sample_order9_c3_playback_indices,
    write_order9_c3_rollout_playback_index,
)


def test_rollout_playback_stops_at_first_terminal_before_environment_reset() -> None:
    tensors = {
        "valid": torch.ones((6, 1), dtype=torch.bool),
        "episode_serial": torch.tensor([[0], [0], [0], [1], [1], [1]]),
        "terminal": torch.tensor([[False], [False], [True], [False], [False], [True]]),
        "truncated": torch.zeros((6, 1), dtype=torch.bool),
    }

    assert order9_c3_first_episode_indices(tensors, 0) == (0, 1, 2)


def test_rollout_playback_sampling_preserves_first_and_terminal_frames() -> None:
    time_s = torch.tensor([0.0, 0.02, 0.10, 0.18, 0.20])

    selected = sample_order9_c3_playback_indices(
        (0, 1, 2, 3, 4), time_s, frames_per_second=10
    )

    assert selected == (0, 2, 4)


def test_rollout_playback_episode_rows_and_selector_index(tmp_path: Path) -> None:
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(
        "\n".join(
            json.dumps(
                {
                    "metadata": {"environment_index": environment},
                    "task_success": success,
                    "failure_reason": None if success else "phase_timeout",
                }
            )
            for environment, success in ((0, False), (1, True))
        )
        + "\n",
        encoding="utf-8",
    )
    rows = load_order9_c3_evaluation_episode_rows(episodes)
    viewer0 = tmp_path / "bucket/environment_0.html"
    viewer1 = tmp_path / "bucket/environment_1.html"
    viewer0.parent.mkdir(parents=True)
    viewer0.write_text("env0", encoding="utf-8")
    viewer1.write_text("env1", encoding="utf-8")
    cases = (
        Order9C3RolloutPlaybackCase(
            bucket_id="bucket-31",
            environment_index=0,
            task_success=False,
            failure_reason="phase_timeout",
            duration_s=65.0,
            frame_count=651,
            html_path=viewer0,
            scene_path=viewer0.with_suffix(".scene.json"),
            source_artifact_sha256="a" * 64,
        ),
        Order9C3RolloutPlaybackCase(
            bucket_id="bucket-31",
            environment_index=1,
            task_success=True,
            failure_reason=None,
            duration_s=246.0,
            frame_count=2461,
            html_path=viewer1,
            scene_path=viewer1.with_suffix(".scene.json"),
            source_artifact_sha256="a" * 64,
        ),
    )

    index = write_order9_c3_rollout_playback_index(tmp_path / "index.html", cases)

    assert sorted(rows) == [0, 1]
    assert rows[0]["failure_reason"] == "phase_timeout"
    page = index.read_text(encoding="utf-8")
    assert "保存stateの再生のみ（再validationなし）" in page
    assert "bucket/environment_0.html" in page
    manifest = json.loads(
        index.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["physics_executed_during_playback"] is False
    assert manifest["revalidation_performed"] is False


def test_rollout_playback_rejects_duplicate_episode_environment(
    tmp_path: Path,
) -> None:
    episodes = tmp_path / "episodes.jsonl"
    row = json.dumps({"metadata": {"environment_index": 0}})
    episodes.write_text(f"{row}\n{row}\n", encoding="utf-8")

    with pytest.raises(SchemaValidationError, match="repeats an environment"):
        load_order9_c3_evaluation_episode_rows(episodes)
