from amsrr.training.order9_r1_collision_avoidance_v17 import (
    ORDER9_R1_COLLISION_AVOIDANCE_V17_VERSION,
    order9_r1_collision_avoidance_heuristics_v17,
)


def test_v17_binds_exactly_the_four_diagnosed_collision_conditions() -> None:
    heuristics = order9_r1_collision_avoidance_heuristics_v17()

    assert ORDER9_R1_COLLISION_AVOIDANCE_V17_VERSION.endswith("_v17")
    assert set(heuristics) == {
        ("ba94152a03e4", 0),
        ("ba94152a03e4", 26),
        ("b6b88f01296d", 0),
        ("b6b88f01296d", 26),
    }
    assert {key: value.pregrasp_clearance_m for key, value in heuristics.items()} == {
        ("ba94152a03e4", 0): 0.08,
        ("ba94152a03e4", 26): 0.08,
        ("b6b88f01296d", 0): 0.08,
        ("b6b88f01296d", 26): 0.08,
    }
    assert all(value.collision_margin_m == 0.010 for value in heuristics.values())
    assert {
        key: value.grasp_contact_height_offset_m for key, value in heuristics.items()
    } == {
        ("ba94152a03e4", 0): 0.05,
        ("ba94152a03e4", 26): 0.05,
        ("b6b88f01296d", 0): 0.03,
        ("b6b88f01296d", 26): 0.03,
    }
    assert {key: value.approach_time_scale for key, value in heuristics.items()} == {
        ("ba94152a03e4", 0): 0.75,
        ("ba94152a03e4", 26): 0.75,
        ("b6b88f01296d", 0): 0.75,
        ("b6b88f01296d", 26): 0.75,
    }
    assert heuristics[("ba94152a03e4", 0)].approach_unfold_fraction is None
    assert heuristics[("ba94152a03e4", 26)].approach_unfold_fraction is None
    assert heuristics[("b6b88f01296d", 0)].approach_unfold_fraction is None


def test_v17_changes_path_generation_without_changing_safety_gates() -> None:
    heuristics = order9_r1_collision_avoidance_heuristics_v17()

    selected_early_contact = heuristics[("ba94152a03e4", 26)]
    assert selected_early_contact.selected_anchor_body is True
    assert selected_early_contact.candidate_group_id == "slot_0:grasp_pair:0"
    assert selected_early_contact.selected_surface_port_ids == (13, 21)
    assert selected_early_contact.tangent_offset_world_m == (0.0, 0.02, 0.0)
    assert selected_early_contact.free_joint_targets_rad == ()

    non_grasp_contact = heuristics[("b6b88f01296d", 0)]
    assert non_grasp_contact.selected_anchor_body is False
    assert non_grasp_contact.candidate_group_id == "slot_0:grasp_pair:0"
    assert non_grasp_contact.teacher_option[2:] == (
        0.08,
        0.010,
        0.03,
        (0.0, 0.02, 0.0),
    )
