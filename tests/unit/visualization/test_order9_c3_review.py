from __future__ import annotations

import json
from pathlib import Path

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.utils.hashing import hash_file
from amsrr.visualization.order9_c3_review import (
    ORDER9_C3_REVIEW_VERSION,
    order9_c3_review_state,
    record_order9_c3_review,
    reset_order9_c3_review_after_recalculation,
)


def _curation_root(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "curation"
    scene = root / "cases/case-1/final.scene.json"
    scene.parent.mkdir(parents=True)
    scene.write_text('{"candidate": 1}\n', encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "review_status": "pending_final_pose_review",
                "cases": [
                    {
                        "case_id": "case-1",
                        "review_status": "pending_user_accept_or_reject",
                        "final_pose": {
                            "scene_path": str(scene),
                            "collision_aware": {"attempt_index": 2},
                            "review_status": (
                                "pending_user_accept_or_reject"
                            ),
                        },
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root, scene


def test_review_actions_are_persistent_and_bound_to_scene(
    tmp_path: Path,
) -> None:
    root, scene = _curation_root(tmp_path)

    initial = order9_c3_review_state(root, "case-1")
    assert initial["review_status"] == "pending_user_accept_or_reject"
    assert initial["recompute_index"] == 2
    assert initial["scene_sha256"] == hash_file(scene)

    accepted = record_order9_c3_review(
        root,
        case_id="case-1",
        action="accept",
        note="mesh clearance visually accepted",
        expected_scene_sha256=hash_file(scene),
    )
    assert accepted["review_status"] == "accepted_by_user"
    assert order9_c3_review_state(root, "case-1")["note"] == (
        "mesh clearance visually accepted"
    )
    decisions = json.loads(
        (root / "review_decisions.json").read_text(encoding="utf-8")
    )
    assert decisions["review_version"] == ORDER9_C3_REVIEW_VERSION
    assert decisions["decisions"]["case-1"]["action"] == "accept"

    with pytest.raises(SchemaValidationError, match="scene hash"):
        record_order9_c3_review(
            root,
            case_id="case-1",
            action="reject",
            expected_scene_sha256="stale",
        )

    reset = reset_order9_c3_review_after_recalculation(
        root,
        case_id="case-1",
        note="try another seed",
        recompute_index=3,
    )
    assert reset["review_status"] == "pending_user_accept_or_reject"
    state = order9_c3_review_state(root, "case-1")
    assert state["recompute_index"] == 3
    assert state["note"] == "try another seed"


def test_review_rejects_unknown_action(tmp_path: Path) -> None:
    root, _scene = _curation_root(tmp_path)

    with pytest.raises(SchemaValidationError, match="unsupported"):
        record_order9_c3_review(
            root,
            case_id="case-1",
            action="maybe",
        )
