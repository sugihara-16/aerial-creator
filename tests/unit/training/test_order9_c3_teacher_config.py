from __future__ import annotations

import pytest

from amsrr.training.order9_c3_teacher import Order9C3TeacherConfig


def test_c3_teacher_config_accepts_pinned_surface_pair_and_group() -> None:
    config = Order9C3TeacherConfig(
        maximum_surface_pair_attempts=1,
        preferred_surface_port_ids=(3, 7),
        preferred_candidate_group_id="slot_0:grasp_pair:0",
    )

    assert config.preferred_surface_port_ids == (3, 7)
    assert config.preferred_candidate_group_id == "slot_0:grasp_pair:0"


@pytest.mark.parametrize(
    "surface_ids",
    [
        (3, 3),
        (-1, 7),
    ],
)
def test_c3_teacher_config_rejects_invalid_pinned_surface_pair(
    surface_ids: tuple[int, int],
) -> None:
    with pytest.raises(ValueError, match="two distinct"):
        Order9C3TeacherConfig(
            preferred_surface_port_ids=surface_ids
        )


def test_c3_teacher_config_rejects_empty_pinned_group() -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        Order9C3TeacherConfig(preferred_candidate_group_id="")
