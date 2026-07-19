from __future__ import annotations

"""Inspect the current Order-8 grasp in the Isaac GUI.

The default path runs the exact authored-mesh diagnostic with real physics in
Kit.  A faster state-trace replay remains available explicitly, but it is only
a visual aid and cannot replace the live-physics observation or acceptance
evidence.
"""

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from amsrr.simulation.isaac_lab_backend import load_isaac_lab_backend_config
from amsrr.simulation.order8_natural_contact import _run_json_command
from amsrr.simulation.order8_state_trace import load_order8_state_trace


DEFAULT_SOURCE_REPORT = Path("/tmp/order8_safe_open_closure_v307_30s.json")
DEFAULT_TRACE_PATH = Path(
    "artifacts/p4_full/order8_natural_contact/diagnostics/"
    "safe_open_closure_v307_30s_state_trace.json"
)

_TRACE_AND_VIEWER_VALUE_OPTIONS = {
    "--viz",
    "--keep-open-after-smoke-s",
    "--order8-state-trace-output",
    "--order8-state-trace-frame-stride",
    "--order8-state-trace-replay",
    "--order8-state-trace-replay-speed",
    "--order8-state-trace-replay-loops",
    "--order8-state-trace-replay-endpoint-hold-s",
}
_TRACE_AND_VIEWER_FLAG_OPTIONS = {
    "--headless",
    "--realtime-playback",
    "--order8-state-trace-replay-sync-physics",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the current Order-8 grasp diagnostic in the Isaac GUI. "
            "Live physics is the default; trace replay is visual-only."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("live", "replay"),
        default="live",
        help=(
            "Use the exact real-physics diagnostic (default), or explicitly "
            "request the faster acceptance-ineligible trace replay."
        ),
    )
    parser.add_argument(
        "--source-report",
        default=str(DEFAULT_SOURCE_REPORT),
        help=(
            "Prior diagnostic JSON containing the exact probe_command. Used when "
            "the trace is absent, or when --refresh-trace is requested."
        ),
    )
    parser.add_argument("--trace-path", default=str(DEFAULT_TRACE_PATH))
    parser.add_argument(
        "--refresh-trace",
        action="store_true",
        help=(
            "Replay mode only: rerun the headless physical diagnostic and replace "
            "the stored trace before replay."
        ),
    )
    parser.add_argument(
        "--capture-only",
        action="store_true",
        help="Replay mode only: record/validate the trace and do not launch GUI.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=2,
        help="Capture every Nth physics frame; 2 is 50 fps for the current dt.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Replay mode only: wall-clock replay speed multiplier.",
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=3,
        help="Replay mode only: number of trace loops.",
    )
    parser.add_argument(
        "--endpoint-hold-s",
        type=float,
        default=1.5,
        help=(
            "Hold the first and last recorded poses so the small Dock-joint "
            "difference is easier to see."
        ),
    )
    parser.add_argument(
        "--keep-open-s",
        type=float,
        default=5.0,
        help="Keep Kit open after the live run or final replay loop.",
    )
    parser.add_argument("--capture-timeout-s", type=float, default=600.0)
    parser.add_argument(
        "--print-commands",
        action="store_true",
        help="Print the selected live/capture/replay command without executing it.",
    )
    return parser


def _strip_trace_and_viewer_options(argv: list[str]) -> list[str]:
    stripped: list[str] = []
    index = 0
    while index < len(argv):
        value = argv[index]
        if value in _TRACE_AND_VIEWER_FLAG_OPTIONS:
            index += 1
            continue
        if value in _TRACE_AND_VIEWER_VALUE_OPTIONS:
            if index + 1 >= len(argv):
                raise ValueError(f"missing value after {value}")
            index += 2
            continue
        stripped.append(value)
        index += 1
    return stripped


def _source_probe_command(source_report_path: Path) -> list[str]:
    payload = json.loads(source_report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("source diagnostic report must be a JSON object")
    if payload.get("diagnostic_only") is not True:
        raise ValueError("source diagnostic report must be diagnostic-only")
    if payload.get("acceptance_eligible") is not False:
        raise ValueError("source diagnostic report must be acceptance-ineligible")
    command = payload.get("probe_command")
    if not isinstance(command, list) or any(
        not isinstance(value, str) or not value for value in command
    ):
        raise ValueError("source diagnostic report has no valid probe_command")
    return [str(value) for value in command]


def build_capture_command(
    source_command: list[str],
    *,
    trace_path: Path,
    frame_stride: int,
) -> list[str]:
    command = _strip_trace_and_viewer_options(list(source_command))
    if "--order8-diagnostic-only" not in command:
        raise ValueError("capture source must enable --order8-diagnostic-only")
    command.extend(
        [
            "--order8-state-trace-output",
            str(trace_path.resolve()),
            "--order8-state-trace-frame-stride",
            str(int(frame_stride)),
        ]
    )
    return _ensure_isaaclab_environment(command)


def build_live_physics_command(
    source_command: list[str],
    *,
    keep_open_s: float,
) -> list[str]:
    """Launch the exact diagnostic with real physics and the Kit viewer."""

    command = _strip_trace_and_viewer_options(list(source_command))
    if "--order8-diagnostic-only" not in command:
        raise ValueError("live source must enable --order8-diagnostic-only")
    command.extend(["--viz", "kit", "--realtime-playback"])
    if keep_open_s > 0.0:
        command.extend(["--keep-open-after-smoke-s", str(float(keep_open_s))])
    return _ensure_isaaclab_environment(command)


def _launcher_prefix_from_probe_argv(probe_argv: list[str]) -> list[str]:
    try:
        config_index = probe_argv.index("--config")
        backend_config_path = probe_argv[config_index + 1]
    except (ValueError, IndexError):
        backend_config_path = "configs/env/isaac_lab.yaml"
    backend_config = load_isaac_lab_backend_config(backend_config_path)
    isaaclab_path = Path(
        os.path.expandvars(os.path.expanduser(str(backend_config.isaaclab_path)))
    ).resolve()
    return [
        str(isaaclab_path / backend_config.launch_script),
        "-p",
        str(REPO_ROOT / "scripts" / "p4_control_holon_spawn_probe.py"),
    ]


def _ensure_isaaclab_environment(command: list[str]) -> list[str]:
    if command and Path(command[0]).name == "micromamba":
        return command
    try:
        config_index = command.index("--config")
        backend_config_path = command[config_index + 1]
    except (ValueError, IndexError):
        backend_config_path = "configs/env/isaac_lab.yaml"
    backend_config = load_isaac_lab_backend_config(backend_config_path)
    micromamba = shutil.which("micromamba")
    if micromamba is None:
        fallback = Path.home() / ".local" / "bin" / "micromamba"
        if not fallback.is_file():
            raise FileNotFoundError(
                "micromamba is required to launch the configured Isaac Lab environment"
            )
        micromamba = str(fallback)
    return [
        str(Path(micromamba).resolve()),
        "run",
        "-n",
        str(backend_config.micromamba_env),
        *command,
    ]


def build_replay_command(
    trace: dict[str, object],
    *,
    trace_path: Path,
    speed: float,
    loops: int,
    keep_open_s: float,
    endpoint_hold_s: float,
) -> list[str]:
    source_probe_argv = trace.get("source_probe_argv")
    if not isinstance(source_probe_argv, list) or any(
        not isinstance(value, str) for value in source_probe_argv
    ):
        raise ValueError("state trace has no valid source_probe_argv")
    probe_argv = _strip_trace_and_viewer_options(
        [str(value) for value in source_probe_argv]
    )
    if "--order8-diagnostic-only" not in probe_argv:
        raise ValueError("state trace source must be diagnostic-only")
    command = _ensure_isaaclab_environment(
        _launcher_prefix_from_probe_argv(probe_argv) + probe_argv
    )
    command.extend(
        [
            "--viz",
            "kit",
            "--order8-state-trace-replay",
            str(trace_path.resolve()),
            "--order8-state-trace-replay-speed",
            str(float(speed)),
            "--order8-state-trace-replay-loops",
            str(int(loops)),
            "--order8-state-trace-replay-endpoint-hold-s",
            str(float(endpoint_hold_s)),
        ]
    )
    if keep_open_s > 0.0:
        command.extend(["--keep-open-after-smoke-s", str(float(keep_open_s))])
    return command


def _full_source_command_from_trace(trace: dict[str, object]) -> list[str]:
    source_probe_argv = trace.get("source_probe_argv")
    if not isinstance(source_probe_argv, list) or any(
        not isinstance(value, str) for value in source_probe_argv
    ):
        raise ValueError("state trace has no valid source_probe_argv")
    probe_argv = [str(value) for value in source_probe_argv]
    return _launcher_prefix_from_probe_argv(probe_argv) + probe_argv


def _maximum_recorded_dock_motion_rad(trace: dict[str, object]) -> float:
    joint_names_by_module = trace.get("joint_names_by_module")
    frames = trace.get("frames")
    if not isinstance(joint_names_by_module, dict) or not isinstance(frames, list):
        raise ValueError("state trace has no joint-motion data")
    if not frames:
        raise ValueError("state trace has no frames")
    first_frame = frames[0]
    if not isinstance(first_frame, dict) or not isinstance(
        first_frame.get("modules"), dict
    ):
        raise ValueError("state trace first frame has no module states")
    first_modules = first_frame["modules"]
    maximum = 0.0
    for module_key, joint_names in joint_names_by_module.items():
        if not isinstance(joint_names, list):
            raise ValueError("state trace joint-name map is malformed")
        first_state = first_modules.get(str(module_key))
        if not isinstance(first_state, dict) or not isinstance(
            first_state.get("joint_positions_rad"), list
        ):
            raise ValueError("state trace first joint state is malformed")
        first_positions = first_state["joint_positions_rad"]
        for frame in frames:
            if not isinstance(frame, dict) or not isinstance(
                frame.get("modules"), dict
            ):
                raise ValueError("state trace frame is malformed")
            state = frame["modules"].get(str(module_key))
            if not isinstance(state, dict) or not isinstance(
                state.get("joint_positions_rad"), list
            ):
                raise ValueError("state trace joint state is malformed")
            positions = state["joint_positions_rad"]
            for index, joint_name in enumerate(joint_names):
                if "dock" not in str(joint_name).lower():
                    continue
                maximum = max(
                    maximum,
                    abs(float(positions[index]) - float(first_positions[index])),
                )
    return maximum


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.frame_stride <= 0 or args.loops <= 0:
        raise ValueError("frame stride and replay loops must be positive")
    if not math.isfinite(args.speed) or args.speed <= 0.0:
        raise ValueError("replay speed must be finite and positive")
    if not math.isfinite(args.keep_open_s) or args.keep_open_s < 0.0:
        raise ValueError("keep-open duration must be finite and non-negative")
    if not math.isfinite(args.endpoint_hold_s) or args.endpoint_hold_s < 0.0:
        raise ValueError("endpoint hold must be finite and non-negative")
    if not math.isfinite(args.capture_timeout_s) or args.capture_timeout_s <= 0.0:
        raise ValueError("capture timeout must be finite and positive")
    if args.mode == "live" and (args.refresh_trace or args.capture_only):
        raise ValueError(
            "--refresh-trace and --capture-only apply only to --mode replay"
        )

    trace_path = Path(args.trace_path).expanduser().resolve()
    trace: dict[str, object] | None = None
    if trace_path.is_file():
        trace = load_order8_state_trace(trace_path)

    if args.mode == "live":
        if trace is not None:
            source_command = _full_source_command_from_trace(trace)
            print(
                "[order8-current-grasp] Stored real-physics trace contains "
                "maximum Dock-joint motion "
                f"{math.degrees(_maximum_recorded_dock_motion_rad(trace)):.2f} deg.",
                flush=True,
            )
        else:
            source_report_path = Path(args.source_report).expanduser().resolve()
            if not source_report_path.is_file():
                raise FileNotFoundError(
                    "neither the state trace nor source diagnostic report was found"
                )
            source_command = _source_probe_command(source_report_path)
        live_command = build_live_physics_command(
            source_command,
            keep_open_s=float(args.keep_open_s),
        )
        if args.print_commands:
            print("live:", shlex.join(live_command))
            return 0
        print(
            "[order8-current-grasp] Launching the exact 30 s diagnostic with "
            "real physics. It is slower than wall clock; joint motion and contact "
            "must be judged from this mode.",
            flush=True,
        )
        completed = subprocess.run(live_command, check=False)
        return int(completed.returncode)

    capture_command: list[str] | None = None
    if args.refresh_trace or not trace_path.is_file():
        source_report_path = Path(args.source_report).expanduser().resolve()
        if not source_report_path.is_file():
            raise FileNotFoundError(
                "state trace is absent and the source diagnostic report was not "
                f"found: {source_report_path}"
            )
        capture_command = build_capture_command(
            _source_probe_command(source_report_path),
            trace_path=trace_path,
            frame_stride=int(args.frame_stride),
        )
        if args.print_commands:
            print("capture:", shlex.join(capture_command))
        else:
            print(
                "[order8-current-grasp] Capturing the real physics once; "
                "the current 30 s baseline takes several wall-clock minutes.",
                flush=True,
            )
            capture_report = _run_json_command(
                capture_command,
                float(args.capture_timeout_s),
            )
            if not trace_path.is_file():
                raise RuntimeError(
                    "physics capture returned without producing the state trace: "
                    f"{capture_report.get('error', 'unknown error')}"
                )
            print(
                "[order8-current-grasp] Capture complete: "
                f"simulation_time={capture_report.get('order8_natural_contact_simulation_time_s')}s",
                flush=True,
            )

    if args.print_commands and not trace_path.is_file():
        return 0

    trace = load_order8_state_trace(trace_path)
    print(
        "[order8-current-grasp] Visual-only trace replay; physics is not advanced, "
        "rendered articulation motion is not acceptance evidence, and live mode "
        "is required for physical judgement.",
        flush=True,
    )
    if args.capture_only:
        return 0

    replay_command = build_replay_command(
        trace,
        trace_path=trace_path,
        speed=float(args.speed),
        loops=int(args.loops),
        keep_open_s=float(args.keep_open_s),
        endpoint_hold_s=float(args.endpoint_hold_s),
    )
    if args.print_commands:
        print("replay:", shlex.join(replay_command))
        return 0
    completed = subprocess.run(replay_command, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
