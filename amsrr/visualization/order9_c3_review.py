from __future__ import annotations

"""Persistent review decisions for the local Order 9 C3 mesh viewer."""

import json
import os
from pathlib import Path
import tempfile
from typing import Any

from amsrr.schemas.common import SchemaValidationError
from amsrr.utils.hashing import hash_file


ORDER9_C3_REVIEW_VERSION = "order9_c3_mesh_review_v1"
ORDER9_C3_REVIEW_ACTIONS = frozenset({"accept", "reject", "recalculate"})


def order9_c3_review_state(
    curation_root: str | Path,
    case_id: str,
) -> dict[str, Any]:
    root = Path(curation_root).resolve()
    manifest = _load_manifest(root)
    case = _case_record(manifest, case_id)
    decisions = _load_decisions(root)
    decision = decisions["decisions"].get(case_id, {})
    final_pose = case.get("final_pose")
    collision_aware = (
        final_pose.get("collision_aware", {})
        if isinstance(final_pose, dict)
        else {}
    )
    default_review_status = (
        final_pose.get("review_status", case["review_status"])
        if isinstance(final_pose, dict)
        else case["review_status"]
    )
    return {
        "case_id": case_id,
        "review_status": str(
            decision.get("review_status", default_review_status)
        ),
        "action": decision.get("action"),
        "note": str(decision.get("note", "")),
        "recompute_index": int(
            decision.get(
                "recompute_index",
                collision_aware.get("attempt_index", 0),
            )
        ),
        "scene_sha256": _case_scene_sha256(root, case),
    }


def record_order9_c3_review(
    curation_root: str | Path,
    *,
    case_id: str,
    action: str,
    note: str = "",
    expected_scene_sha256: str | None = None,
    recompute_index: int | None = None,
) -> dict[str, Any]:
    root = Path(curation_root).resolve()
    action = str(action)
    if action not in ORDER9_C3_REVIEW_ACTIONS:
        raise SchemaValidationError(
            f"unsupported C3 review action {action!r}"
        )
    manifest = _load_manifest(root)
    case = _case_record(manifest, case_id)
    actual_scene_sha256 = _case_scene_sha256(root, case)
    if (
        expected_scene_sha256 is not None
        and expected_scene_sha256 != actual_scene_sha256
    ):
        raise SchemaValidationError(
            "review request scene hash differs from the current candidate"
        )
    decisions = _load_decisions(root)
    previous = decisions["decisions"].get(case_id, {})
    next_recompute_index = int(
        previous.get("recompute_index", 0)
        if recompute_index is None
        else recompute_index
    )
    review_status = {
        "accept": "accepted_by_user",
        "reject": "rejected_by_user",
        "recalculate": "recalculation_requested",
    }[action]
    record = {
        "case_id": case_id,
        "action": action,
        "review_status": review_status,
        "note": str(note),
        "scene_sha256": actual_scene_sha256,
        "recompute_index": next_recompute_index,
    }
    decisions["decisions"][case_id] = record
    _write_json(root / "review_decisions.json", decisions)
    case["review_status"] = review_status
    if isinstance(case.get("final_pose"), dict):
        case["final_pose"]["review_status"] = review_status
    manifest["review_status"] = _overall_review_status(manifest)
    _write_json(root / "manifest.json", manifest)
    return dict(record)


def reset_order9_c3_review_after_recalculation(
    curation_root: str | Path,
    *,
    case_id: str,
    note: str,
    recompute_index: int,
) -> dict[str, Any]:
    root = Path(curation_root).resolve()
    manifest = _load_manifest(root)
    case = _case_record(manifest, case_id)
    scene_sha256 = _case_scene_sha256(root, case)
    decisions = _load_decisions(root)
    record = {
        "case_id": case_id,
        "action": "recalculate",
        "review_status": "pending_user_accept_or_reject",
        "note": str(note),
        "scene_sha256": scene_sha256,
        "recompute_index": int(recompute_index),
    }
    decisions["decisions"][case_id] = record
    _write_json(root / "review_decisions.json", decisions)
    case["review_status"] = record["review_status"]
    if isinstance(case.get("final_pose"), dict):
        case["final_pose"]["review_status"] = record["review_status"]
    manifest["review_status"] = _overall_review_status(manifest)
    _write_json(root / "manifest.json", manifest)
    return dict(record)


def _load_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("cases"), list):
        raise SchemaValidationError("C3 curation manifest is invalid")
    return value


def _load_decisions(root: Path) -> dict[str, Any]:
    path = root / "review_decisions.json"
    if not path.is_file():
        return {
            "review_version": ORDER9_C3_REVIEW_VERSION,
            "decisions": {},
        }
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("review_version") != ORDER9_C3_REVIEW_VERSION
        or not isinstance(value.get("decisions"), dict)
    ):
        raise SchemaValidationError("C3 review decision file is invalid")
    return value


def _case_record(
    manifest: dict[str, Any],
    case_id: str,
) -> dict[str, Any]:
    matches = [
        value
        for value in manifest["cases"]
        if isinstance(value, dict) and value.get("case_id") == case_id
    ]
    if len(matches) != 1:
        raise SchemaValidationError(
            f"C3 curation case {case_id!r} is not uniquely present"
        )
    return matches[0]


def _case_scene_sha256(root: Path, case: dict[str, Any]) -> str:
    final_pose = case.get("final_pose")
    if not isinstance(final_pose, dict):
        raise SchemaValidationError(
            f"C3 case {case.get('case_id')!r} has no final pose"
        )
    path = Path(str(final_pose["scene_path"]))
    if not path.is_absolute():
        repository_path = Path(__file__).resolve().parents[2] / path
        path = repository_path if repository_path.is_file() else root / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return hash_file(path)


def _overall_review_status(manifest: dict[str, Any]) -> str:
    statuses = {
        str(value.get("review_status"))
        for value in manifest["cases"]
        if isinstance(value, dict)
    }
    if statuses == {"accepted_by_user"}:
        return "all_candidates_accepted_by_user"
    if "recalculation_requested" in statuses:
        return "recalculation_requested"
    if "rejected_by_user" in statuses:
        return "contains_user_rejection"
    return "pending_final_pose_review"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = [
    "ORDER9_C3_REVIEW_ACTIONS",
    "ORDER9_C3_REVIEW_VERSION",
    "order9_c3_review_state",
    "record_order9_c3_review",
    "reset_order9_c3_review_after_recalculation",
]
