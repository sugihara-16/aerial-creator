from __future__ import annotations

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.order4 import (
    Order4FreeFlightMission,
    Order4FreeFlightWaypoint,
    build_order4_free_flight_mission,
    default_order4_free_flight_mission,
)


def test_order4_default_mission_is_hash_bound_and_round_trips() -> None:
    mission = default_order4_free_flight_mission()

    assert len(mission.waypoints) == 3
    assert mission.final_hover_hold_s == 5.0
    assert len(mission.mission_hash) == 64
    assert Order4FreeFlightMission.from_json(mission.to_json()) == mission


def test_order4_mission_requires_multiple_waypoints_and_matching_hash() -> None:
    with pytest.raises(SchemaValidationError, match="at least two waypoints"):
        build_order4_free_flight_mission(
            mission_id="too-short",
            waypoints=[
                Order4FreeFlightWaypoint(
                    waypoint_id="only",
                    position_offset_world=[0.0, 0.0, 0.0],
                    orientation_rpy_rad=[0.0, 0.0, 0.0],
                )
            ],
        )

    mission = default_order4_free_flight_mission()
    payload = mission.to_dict()
    payload["final_hover_hold_s"] = 6.0
    with pytest.raises(SchemaValidationError, match="mission_hash"):
        Order4FreeFlightMission.from_dict(payload)


def test_order4_waypoint_timeout_covers_transition_and_dwell() -> None:
    with pytest.raises(SchemaValidationError, match="must cover"):
        Order4FreeFlightWaypoint.from_dict(
            {
                "waypoint_id": "invalid-timeout",
                "position_offset_world": [0.0, 0.0, 0.0],
                "orientation_rpy_rad": [0.0, 0.0, 0.0],
                "transition_duration_s": 2.0,
                "dwell_s": 1.0,
                "timeout_s": 2.5,
            }
        )
