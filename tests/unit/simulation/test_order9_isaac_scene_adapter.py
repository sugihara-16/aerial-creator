from amsrr.simulation.order9_isaac_scene_adapter import (
    _payload_coupling_is_load_bearing,
)


def test_payload_coupling_does_not_treat_contact_intent_as_payload_load() -> None:
    assert not _payload_coupling_is_load_bearing(
        "approach",
        ("approach", "approach"),
    )
    assert not _payload_coupling_is_load_bearing(
        "contact_acquisition",
        ("attach", "attach"),
    )


def test_payload_coupling_is_limited_to_load_bearing_motion_phases() -> None:
    for phase in ("lift", "transport", "place"):
        assert _payload_coupling_is_load_bearing(
            phase,
            ("maintain", "maintain"),
        )
    assert not _payload_coupling_is_load_bearing(
        "release",
        ("release", "release"),
    )
    assert not _payload_coupling_is_load_bearing(
        "lift",
        ("approach", "approach"),
    )
