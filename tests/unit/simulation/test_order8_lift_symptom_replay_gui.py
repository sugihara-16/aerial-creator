from __future__ import annotations

import json
from pathlib import Path

import pytest

from amsrr.simulation.order8_state_trace import build_order8_state_trace
from scripts.order8_lift_symptom_replay_gui import (
    build_low_load_replay_command,
    load_one_kg_source_command,
    trace_object_mass_kg,
)


def _source_report(tmp_path: Path, *, mass_kg: float) -> Path:
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps(
            {
                "diagnostic_only": True,
                "acceptance_eligible": False,
                "diagnostic_config": {"object_mass_kg": mass_kg},
                "probe_command": [
                    "/home/leus/IsaacLab/isaaclab.sh",
                    "-p",
                    "/repo/scripts/p4_control_holon_spawn_probe.py",
                    "--order8-diagnostic-only",
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _trace(*, mass_kg: float) -> dict[str, object]:
    module = {
        "root_pose_world": [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
        "root_twist_world": [0.0] * 6,
        "joint_positions_rad": [0.0],
        "joint_velocities_radps": [0.0],
    }
    frame = {
        "simulation_time_s": 0.0,
        "phase": "lift",
        "modules": {"0": module},
        "object_pose_world": [0.5, 0.0, 0.225, 0.0, 0.0, 0.0, 1.0],
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
        source_probe_argv=[
            "--config",
            "configs/env/isaac_lab.yaml",
            "--order8-natural-contact",
            "--order8-diagnostic-only",
            "--order8-config-json",
            json.dumps({"object_mass_kg": mass_kg}),
        ],
        frames=[
            frame,
            {
                **frame,
                "simulation_time_s": 0.02,
            },
        ],
    )


def test_source_and_trace_require_one_kg(tmp_path: Path) -> None:
    command = load_one_kg_source_command(_source_report(tmp_path, mass_kg=1.0))
    assert "--order8-diagnostic-only" in command
    assert trace_object_mass_kg(_trace(mass_kg=1.0)) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="1.0 kg source"):
        load_one_kg_source_command(_source_report(tmp_path, mass_kg=0.9))
    with pytest.raises(ValueError, match="non-1.0 kg trace"):
        trace_object_mass_kg(_trace(mass_kg=0.9))


def test_low_load_replay_uses_visible_balanced_rendering_and_physics_sync(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.json"
    command = build_low_load_replay_command(
        _trace(mass_kg=1.0),
        trace_path=trace_path,
        speed=0.75,
        loops=2,
        endpoint_hold_s=2.0,
        keep_open_s=30.0,
        normal_kit_log=False,
    )

    assert "--realtime-playback" not in command
    assert "--order8-state-trace-replay-sync-physics" in command
    assert command[command.index("--rendering_mode") + 1] == "balanced"
    assert "--kit_args=--/log/outputStreamLevel=Error" in command
    assert command[command.index("--order8-state-trace-replay-speed") + 1] == "0.75"
    assert command[command.index("--order8-state-trace-replay-loops") + 1] == "2"


def test_low_load_replay_rejects_unknown_rendering_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported rendering mode"):
        build_low_load_replay_command(
            _trace(mass_kg=1.0),
            trace_path=tmp_path / "trace.json",
            speed=1.0,
            loops=1,
            endpoint_hold_s=0.0,
            keep_open_s=0.0,
            normal_kit_log=False,
            rendering_mode="invisible",
        )
