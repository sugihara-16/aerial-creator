from __future__ import annotations

import json
from pathlib import Path

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.training.order9_r1_preload_collision_repair import (
    load_order9_r1_preload_collision_repair_rules,
)


REPOSITORY = Path(__file__).resolve().parents[3]
CONFIG = REPOSITORY / "configs/training/order9_r1_preload_collision_repair_v1.json"


def test_preload_collision_repair_rules_bind_four_candidates() -> None:
    reason, rules = load_order9_r1_preload_collision_repair_rules(CONFIG)
    assert "nominal_preload_collision_admission" in reason
    assert len(rules) == 4
    assert all(rule.pregrasp_clearance_m == 0.12 for rule in rules.values())
    assert all(rule.collision_margin_m == 0.007 for rule in rules.values())
    assert all(rule.grasp_contact_height_offset_m == 0.0 for rule in rules.values())


def test_preload_collision_repair_rejects_duplicate_ports(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    first = next(iter(payload["candidates"].values()))
    first["selected_surface_port_ids"] = [21, 21]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SchemaValidationError):
        load_order9_r1_preload_collision_repair_rules(path)


def test_preload_collision_repair_accepts_ordered_fallback_options(
    tmp_path: Path,
) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    first = next(iter(payload["candidates"].values()))
    first["fallback_contact_options"] = [
        {
            "selected_surface_port_ids": [17, 20],
            "selected_candidate_group_id": "slot_0:grasp_pair:0",
            "pregrasp_clearance_m": 0.12,
            "collision_margin_m": 0.01,
            "grasp_contact_height_offset_m": 0.03,
        }
    ]
    path = tmp_path / "fallback.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    _reason, rules = load_order9_r1_preload_collision_repair_rules(path)
    rule = next(iter(rules.values()))
    assert len(rule.teacher_options) == 2
    assert rule.teacher_options[1] == (
        (17, 20),
        "slot_0:grasp_pair:0",
        0.12,
        0.01,
        0.03,
        (0.0, 0.0, 0.0),
    )


def test_preload_collision_repair_rejects_repeated_fallback_option(
    tmp_path: Path,
) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    first = next(iter(payload["candidates"].values()))
    first["fallback_contact_options"] = [
        {
            "selected_surface_port_ids": first["selected_surface_port_ids"],
            "selected_candidate_group_id": first[
                "selected_candidate_group_id"
            ],
            "pregrasp_clearance_m": first["pregrasp_clearance_m"],
            "collision_margin_m": first["collision_margin_m"],
            "grasp_contact_height_offset_m": 0.0,
        }
    ]
    path = tmp_path / "repeated.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SchemaValidationError):
        load_order9_r1_preload_collision_repair_rules(path)


def test_preload_collision_repair_accepts_combined_28mm_contact_offset(
    tmp_path: Path,
) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    first = next(iter(payload["candidates"].values()))
    first["grasp_contact_height_offset_m"] = 0.02
    first["grasp_contact_tangent_offset_world_m"] = [0.0, 0.02, 0.0]
    path = tmp_path / "bounded_offset.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    _reason, rules = load_order9_r1_preload_collision_repair_rules(path)
    rule = next(iter(rules.values()))
    assert rule.grasp_contact_tangent_offset_world_m == (0.0, 0.02, 0.0)


def test_preload_collision_repair_rejects_combined_offset_over_30mm(
    tmp_path: Path,
) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    first = next(iter(payload["candidates"].values()))
    first["grasp_contact_height_offset_m"] = 0.03
    first["grasp_contact_tangent_offset_world_m"] = [0.0, 0.01, 0.0]
    path = tmp_path / "oversize_offset.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SchemaValidationError):
        load_order9_r1_preload_collision_repair_rules(path)
