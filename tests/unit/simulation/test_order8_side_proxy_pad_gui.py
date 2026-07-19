from __future__ import annotations

from scripts.order8_side_proxy_pad_gui import (
    CHILD_FLAG,
    build_isaac_child_command,
    ensure_gui_visualizer,
)


def test_side_proxy_pad_gui_child_command_is_single_module_preview_entrypoint() -> None:
    command = build_isaac_child_command(
        ["--focus-link", "yaw_dock_mech2", "--keep-open-s", "30"],
        micromamba_executable="/opt/micromamba",
        environment_name="isaaclab3",
    )

    assert command[:6] == [
        "/opt/micromamba",
        "run",
        "-n",
        "isaaclab3",
        "python",
        command[5],
    ]
    assert command[6] == CHILD_FLAG
    assert command[-4:] == [
        "--focus-link",
        "yaw_dock_mech2",
        "--keep-open-s",
        "30",
    ]


def test_side_proxy_pad_gui_selects_kit_unless_mode_is_explicit() -> None:
    assert ensure_gui_visualizer(["--keep-open-s", "5"])[-2:] == ["--viz", "kit"]
    assert ensure_gui_visualizer(["--viz", "none"]) == ["--viz", "none"]
    assert ensure_gui_visualizer(["--headless"]) == ["--headless"]
