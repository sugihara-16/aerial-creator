from __future__ import annotations

from pathlib import Path

from amsrr.simulation.order8_state_trace import build_order8_state_trace
from scripts.order8_current_grasp_gui import (
    build_capture_command,
    build_live_physics_command,
    build_replay_command,
)


def _trace() -> dict[str, object]:
    frame = {
        "simulation_time_s": 0.0,
        "phase": "contact_acquisition",
        "modules": {
            "0": {
                "root_pose_world": [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
                "root_twist_world": [0.0] * 6,
                "joint_positions_rad": [0.0],
                "joint_velocities_radps": [0.0],
            }
        },
        "object_pose_world": [0.5, 0.0, 0.075, 0.0, 0.0, 0.0, 1.0],
        "object_twist_world": [0.0] * 6,
    }
    second = {
        **frame,
        "simulation_time_s": 0.02,
        "modules": {
            "0": {
                **frame["modules"]["0"],  # type: ignore[index]
                "joint_positions_rad": [0.01],
            }
        },
    }
    return build_order8_state_trace(
        simulation_dt_s=0.01,
        frame_stride=2,
        graph_id="graph",
        graph_hash="graph-hash",
        config_hash="config-hash",
        source_urdf_sha256="urdf-hash",
        generated_usd_sha256="usd-hash",
        module_ids=[0],
        joint_names_by_module={0: ["joint"]},
        source_probe_argv=[
            "--config",
            "configs/env/isaac_lab.yaml",
            "--steps",
            "100",
            "--order8-natural-contact",
            "--order8-diagnostic-only",
            "--realtime-playback",
            "--viz",
            "kit",
            "--keep-open-after-smoke-s",
            "20",
        ],
        frames=[frame, second],
    )


def test_capture_command_removes_viewer_pacing_and_adds_trace_output(tmp_path) -> None:
    source = [
        "/home/leus/IsaacLab/isaaclab.sh",
        "-p",
        "scripts/p4_control_holon_spawn_probe.py",
        "--order8-natural-contact",
        "--order8-diagnostic-only",
        "--viz",
        "kit",
        "--realtime-playback",
    ]
    trace_path = tmp_path / "trace.json"

    command = build_capture_command(
        source,
        trace_path=trace_path,
        frame_stride=2,
    )

    assert "--viz" not in command
    assert "--realtime-playback" not in command
    assert command[-4:] == [
        "--order8-state-trace-output",
        str(trace_path.resolve()),
        "--order8-state-trace-frame-stride",
        "2",
    ]


def test_replay_command_uses_kit_without_slow_physics_sleep(tmp_path) -> None:
    trace_path = Path(tmp_path / "trace.json")

    command = build_replay_command(
        _trace(),
        trace_path=trace_path,
        speed=1.0,
        loops=3,
        keep_open_s=5.0,
        endpoint_hold_s=1.5,
    )

    assert "--viz" in command
    assert command[command.index("--viz") + 1] == "kit"
    assert "--realtime-playback" not in command
    assert command[command.index("--order8-state-trace-replay") + 1] == str(
        trace_path.resolve()
    )
    assert command[command.index("--order8-state-trace-replay-loops") + 1] == "3"
    assert (
        command[
            command.index("--order8-state-trace-replay-endpoint-hold-s") + 1
        ]
        == "1.5"
    )


def test_live_command_runs_exact_physics_with_kit() -> None:
    source = [
        "/home/leus/IsaacLab/isaaclab.sh",
        "-p",
        "scripts/p4_control_holon_spawn_probe.py",
        "--order8-natural-contact",
        "--order8-diagnostic-only",
        "--order8-state-trace-output",
        "/tmp/trace.json",
        "--order8-state-trace-frame-stride",
        "2",
    ]

    command = build_live_physics_command(source, keep_open_s=7.0)

    assert "--order8-state-trace-output" not in command
    assert "--order8-state-trace-frame-stride" not in command
    assert "--viz" in command
    assert command[command.index("--viz") + 1] == "kit"
    assert "--realtime-playback" in command
    assert command[command.index("--keep-open-after-smoke-s") + 1] == "7.0"
