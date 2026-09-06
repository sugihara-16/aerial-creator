from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.training.order9_r1_teacher_collection_preparation import (
    build_order9_r1_teacher_collection_plan,
    build_order9_r1_teacher_collection_preflight,
    load_order9_r1_teacher_collection_protocol,
    protocol_with_path,
    validate_order9_r1_teacher_collection_plan,
)

REPOSITORY = Path(__file__).resolve().parents[3]
PROTOCOL = REPOSITORY / (
    "configs/training/order9_r1_teacher_collection_protocol_v19.json"
)


def _plan():
    protocol = protocol_with_path(
        load_order9_r1_teacher_collection_protocol(
            PROTOCOL, repository_root=REPOSITORY
        ),
        PROTOCOL,
    )
    return protocol, build_order9_r1_teacher_collection_plan(
        protocol, repository_root=REPOSITORY
    )


def test_r1_teacher_plan_is_140_unique_stratified_scene_transfers() -> None:
    _protocol, plan = _plan()
    entries = plan["entries"]
    assert len(entries) == 140
    assert [entry["seed"] for entry in entries] == list(range(29009, 29149))
    assert len({entry["task_id"] for entry in entries}) == 140
    assert len({entry["pose_condition_hash"] for entry in entries}) == 140
    for module_count in range(2, 9):
        assert [
            sum(
                entry["module_count"] == module_count and entry["split"] == split
                for entry in entries
            )
            for split in ("train", "validation", "held_out")
        ] == [14, 3, 3]
    assert all(entry["support_transformed_with_scene"] for entry in entries)
    assert all(entry["fast_exact_tracking_screen_required"] for entry in entries)
    assert not any(entry["pi_l_actor_command_applied"] for entry in entries)


def test_r1_teacher_plan_rejects_duplicate_pose_or_changed_source() -> None:
    _protocol, plan = _plan()
    duplicate = deepcopy(plan)
    duplicate["entries"][1]["pose_condition_hash"] = duplicate["entries"][0][
        "pose_condition_hash"
    ]
    with pytest.raises(SchemaValidationError, match="pose_condition_hash"):
        validate_order9_r1_teacher_collection_plan(
            duplicate, repository_root=REPOSITORY
        )
    changed = deepcopy(plan)
    changed["entries"][0]["source_case_manifest"]["sha256"] = "0" * 64
    with pytest.raises(SchemaValidationError, match="bound bytes changed"):
        validate_order9_r1_teacher_collection_plan(changed, repository_root=REPOSITORY)


def test_r1_teacher_preflight_stops_at_explicit_launch_boundary() -> None:
    protocol, plan = _plan()
    with tempfile.TemporaryDirectory(dir=REPOSITORY / "artifacts") as root:
        plan_path = Path(root) / "collection_plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        preflight = build_order9_r1_teacher_collection_preflight(
            protocol=protocol,
            plan_path=plan_path,
            repository_root=REPOSITORY,
            free_storage_bytes=200 * 1024**3,
            isaac_python="/bin/true",
            isaac_import_ok=True,
            active_collection_process_count=0,
        )
    assert preflight["ready_for_explicit_launch"] is True
    assert preflight["collection_started"] is False
    assert preflight["collection_launch_authorized"] is False
    assert preflight["status"] == "ready_awaiting_explicit_launch"
