from __future__ import annotations

import json
from pathlib import Path

import pytest

from amsrr.simulation.order8_state_trace import build_order8_state_trace
from scripts.order8_cone_proxy_lift_replay_gui import (
    CONE_PROXY_FLAG,
    build_cone_proxy_replay_command,
    load_cone_proxy_source_command,
    _replay_status_text,
    require_cone_proxy_trace,
    summarize_trace,
)


def _source_report(tmp_path: Path, *, cone_proxy: bool = True) -> Path:
    path = tmp_path / "report.json"
    command = [
        "/home/leus/IsaacLab/isaaclab.sh",
        "-p",
        "/repo/scripts/p4_control_holon_spawn_probe.py",
        "--order8-diagnostic-only",
    ]
    if cone_proxy:
        command.append(CONE_PROXY_FLAG)
    path.write_text(
        json.dumps(
            {
                "diagnostic_only": True,
                "acceptance_eligible": False,
                "diagnostic_config": {"object_mass_kg": 1.0},
                "probe_command": command,
            }
        ),
        encoding="utf-8",
    )
    return path


def _trace(*, cone_proxy: bool = True) -> dict[str, object]:
    source_argv = [
        "--config",
        "configs/env/isaac_lab.yaml",
        "--order8-natural-contact",
        "--order8-diagnostic-only",
        "--order8-config-json",
        json.dumps({"object_mass_kg": 1.0}),
    ]
    if cone_proxy:
        source_argv.append(CONE_PROXY_FLAG)
    module_0 = {
        "root_pose_world": [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
        "root_twist_world": [0.0] * 6,
        "joint_positions_rad": [0.0],
        "joint_velocities_radps": [0.0],
    }
    module_1 = {
        **module_0,
        "joint_positions_rad": [0.1],
    }
    frame_0 = {
        "simulation_time_s": 0.0,
        "phase": "contact_acquisition",
        "modules": {"0": module_0},
        "object_pose_world": [0.5, 0.0, 0.225, 0.0, 0.0, 0.0, 1.0],
        "object_twist_world": [0.0] * 6,
    }
    frame_1 = {
        "simulation_time_s": 0.02,
        "phase": "lift",
        "modules": {"0": module_1},
        "object_pose_world": [0.5, 0.0, 0.237, 0.0, 0.0, 0.0, 1.0],
        "object_twist_world": [0.0] * 6,
    }
    return build_order8_state_trace(
        simulation_dt_s=0.02,
        frame_stride=2,
        graph_id="graph",
        graph_hash="graph-hash",
        config_hash="config-hash",
        source_urdf_sha256="urdf-hash",
        generated_usd_sha256="usd-hash",
        module_ids=[0],
        joint_names_by_module={0: ["yaw_dock_mech_joint1"]},
        source_probe_argv=source_argv,
        frames=[frame_0, frame_1],
    )


def test_source_and_trace_require_cone_proxy(tmp_path: Path) -> None:
    source = load_cone_proxy_source_command(_source_report(tmp_path))
    assert CONE_PROXY_FLAG in source
    require_cone_proxy_trace(_trace())

    with pytest.raises(ValueError, match="not a cone-proxy"):
        load_cone_proxy_source_command(
            _source_report(tmp_path, cone_proxy=False)
        )
    with pytest.raises(ValueError, match="not a cone-proxy"):
        require_cone_proxy_trace(_trace(cone_proxy=False))


def test_trace_summary_proves_joint_motion_and_object_rise() -> None:
    summary = summarize_trace(_trace())
    assert summary["frame_count"] == 2
    assert summary["source_duration_s"] == pytest.approx(0.02)
    assert summary["maximum_dock_motion_rad"] == pytest.approx(0.1)
    assert summary["maximum_object_com_rise_m"] == pytest.approx(0.012)
    assert summary["final_object_com_rise_m"] == pytest.approx(0.012)


def test_replay_is_balanced_synchronized_and_retains_cone_visual(
    tmp_path: Path,
) -> None:
    command = build_cone_proxy_replay_command(
        _trace(),
        trace_path=tmp_path / "trace.json",
        speed=0.5,
        loops=2,
        endpoint_hold_s=2.0,
        keep_open_s=30.0,
        normal_kit_log=False,
        rendering_mode="balanced",
    )

    assert CONE_PROXY_FLAG in command
    assert "--order8-state-trace-replay-sync-physics" in command
    assert "--realtime-playback" not in command
    assert command[command.index("--rendering_mode") + 1] == "balanced"
    assert "--kit_args=--/log/outputStreamLevel=Error" in command


def test_replay_log_filter_keeps_status_and_hides_routine_startup() -> None:
    assert _replay_status_text("[3.2s] [ext: omni.foo] startup\n") is None
    assert _replay_status_text("ordinary warning\n") is None
    status = _replay_status_text(
        "\r[order8-state-replay] simulation_time=2.00s phase=lift\n"
    )
    assert status == "[order8-state-replay] simulation_time=2.00s phase=lift"
    assert _replay_status_text('{"spawn_passed": true}\n') is None
    assert _replay_status_text(
        json.dumps(
            {
                "spawn_passed": True,
                "order8_state_trace_replay": True,
                "order8_state_trace_replay_cone_proxy_prim_count": 150,
                "order8_state_trace_maximum_physics_dock_joint_delta_rad": 0.1,
                "order8_state_trace_maximum_physics_joint_write_error_rad": 0.0,
                "order8_state_trace_maximum_object_position_write_error_m": 0.0,
            }
        )
    ) == (
        "[order8-state-replay] complete spawn_passed=True cone_pads=150 "
        "physx_dock_delta=5.73deg physx_error=0.000000rad object_error=0.000000m"
    )
