from __future__ import annotations

"""Replay a recorded cone-proxy 1 kg trajectory in a low-load Kit GUI.

The exact physical trajectory is captured headlessly once and cached.  Kit
then displays the recorded module, joint, and object state with one
gravity-free/contact-minimized synchronization step per rendered frame.  This
keeps the runtime joint inspector and viewport in sync without rerunning the
expensive contact dynamics in the GUI.  It is diagnostic-only and never
acceptance evidence.
"""

import argparse
from collections import deque
import codecs
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from amsrr.simulation.order8_natural_contact import _run_json_command
from amsrr.simulation.order8_state_trace import load_order8_state_trace
from order8_current_grasp_gui import (
    _maximum_recorded_dock_motion_rad,
    build_capture_command,
)
from order8_lift_symptom_replay_gui import (
    build_low_load_replay_command,
    load_one_kg_source_command,
    trace_object_mass_kg,
)


DEFAULT_SOURCE_REPORT = REPO_ROOT / (
    "artifacts/p4_full/order8_natural_contact/diagnostics/"
    "cone_proxy_pad_effective_gate_slowclose005_v375_30s.json"
)
DEFAULT_TRACE_PATH = REPO_ROOT / (
    "artifacts/p4_full/order8_natural_contact/diagnostics/"
    "cone_proxy_pad_effective_gate_slowclose005_v375_visual_state_trace.json"
)
CONE_PROXY_FLAG = "--order8-diagnostic-cone-proxy-pad"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a cone-proxy 1 kg physical trajectory using a cached, "
            "low-load PhysX-synchronized Kit replay."
        )
    )
    parser.add_argument("--source-report", default=str(DEFAULT_SOURCE_REPORT))
    parser.add_argument("--trace-path", default=str(DEFAULT_TRACE_PATH))
    parser.add_argument(
        "--refresh-trace",
        action="store_true",
        help="Recapture the exact physical trajectory headlessly before replay.",
    )
    parser.add_argument(
        "--capture-only",
        action="store_true",
        help="Capture and validate the trace without opening Kit.",
    )
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--capture-timeout-s", type=float, default=600.0)
    parser.add_argument(
        "--speed",
        type=float,
        default=0.5,
        help=(
            "Replay speed relative to simulation time. The 0.5x default lowers "
            "viewport load while making Dock motion easier to inspect."
        ),
    )
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--endpoint-hold-s", type=float, default=2.0)
    parser.add_argument("--keep-open-s", type=float, default=30.0)
    parser.add_argument(
        "--rendering-mode",
        choices=("balanced", "quality"),
        default="balanced",
        help=(
            "Balanced is the verified non-black default on this workstation; "
            "the known-black performance preset is intentionally unavailable."
        ),
    )
    parser.add_argument(
        "--normal-kit-log",
        action="store_true",
        help="Show ordinary Kit warnings instead of errors only.",
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print the capture or replay command without launching it.",
    )
    return parser


def load_cone_proxy_source_command(path: Path) -> list[str]:
    command = load_one_kg_source_command(path)
    if CONE_PROXY_FLAG not in command:
        raise ValueError("source report is not a cone-proxy physical diagnostic")
    return command


def require_cone_proxy_trace(trace: dict[str, object]) -> None:
    trace_object_mass_kg(trace)
    source_argv = trace.get("source_probe_argv")
    if not isinstance(source_argv, list) or CONE_PROXY_FLAG not in source_argv:
        raise ValueError("state trace is not a cone-proxy physical diagnostic")


def summarize_trace(trace: dict[str, object]) -> dict[str, float | int]:
    frames = trace.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("state trace contains no frames")
    object_heights: list[float] = []
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("state trace frame is malformed")
        pose = frame.get("object_pose_world")
        if not isinstance(pose, list) or len(pose) != 7:
            raise ValueError("state trace object pose is malformed")
        object_heights.append(float(pose[2]))
    initial_height_m = object_heights[0]
    return {
        "frame_count": len(frames),
        "source_duration_s": float(frames[-1]["simulation_time_s"])
        - float(frames[0]["simulation_time_s"]),
        "maximum_dock_motion_rad": _maximum_recorded_dock_motion_rad(trace),
        "maximum_object_com_rise_m": max(object_heights) - initial_height_m,
        "final_object_com_rise_m": object_heights[-1] - initial_height_m,
    }


def build_cone_proxy_replay_command(
    trace: dict[str, object],
    *,
    trace_path: Path,
    speed: float,
    loops: int,
    endpoint_hold_s: float,
    keep_open_s: float,
    normal_kit_log: bool,
    rendering_mode: str,
) -> list[str]:
    require_cone_proxy_trace(trace)
    command = build_low_load_replay_command(
        trace,
        trace_path=trace_path,
        speed=speed,
        loops=loops,
        endpoint_hold_s=endpoint_hold_s,
        keep_open_s=keep_open_s,
        normal_kit_log=normal_kit_log,
        rendering_mode=rendering_mode,
    )
    if CONE_PROXY_FLAG not in command:
        raise ValueError("cone-proxy flag was lost while constructing replay")
    return command


def _replay_status_text(line: str) -> str | None:
    """Return concise user-facing replay output, hiding routine Kit startup."""

    normalized = line.replace("\r", "\n")
    kept: list[str] = []
    for raw_part in normalized.splitlines():
        part = raw_part.strip()
        if "[order8-state-replay]" in part:
            kept.append(part)
            continue
        if not part.startswith("{"):
            continue
        try:
            payload = json.loads(part)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or not payload.get(
            "order8_state_trace_replay"
        ):
            continue
        kept.append(
            "[order8-state-replay] complete "
            f"spawn_passed={bool(payload.get('spawn_passed'))} "
            f"cone_pads={int(payload.get('order8_state_trace_replay_cone_proxy_prim_count', 0))} "
            "physx_dock_delta="
            f"{math.degrees(float(payload.get('order8_state_trace_maximum_physics_dock_joint_delta_rad', 0.0))):.2f}deg "
            "physx_error="
            f"{float(payload.get('order8_state_trace_maximum_physics_joint_write_error_rad', math.inf)):.6f}rad "
            "object_error="
            f"{float(payload.get('order8_state_trace_maximum_object_position_write_error_m', math.inf)):.6f}m"
        )
    return "\n".join(part for part in kept if part) or None


def run_replay_command(
    command: list[str],
    *,
    environment: dict[str, str],
    normal_kit_log: bool,
) -> int:
    if normal_kit_log:
        return int(subprocess.run(command, env=environment, check=False).returncode)
    tail: deque[str] = deque(maxlen=80)
    process = subprocess.Popen(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    assert process.stdout is not None
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    pending = ""

    def handle_line(line: str) -> None:
        tail.append(line.rstrip())
        status = _replay_status_text(line)
        if status is not None:
            print(status, flush=True)

    try:
        while True:
            chunk = process.stdout.read(4096)
            if not chunk:
                break
            pending += decoder.decode(chunk)
            while True:
                separators = [
                    index
                    for index in (pending.find("\r"), pending.find("\n"))
                    if index >= 0
                ]
                if not separators:
                    break
                split_at = min(separators)
                handle_line(pending[:split_at])
                pending = pending[split_at + 1 :]
        pending += decoder.decode(b"", final=True)
        if pending:
            handle_line(pending)
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    returncode = int(process.wait())
    if returncode != 0:
        print(
            "[order8-cone-lift-replay] Kit exited abnormally; recent log follows:",
            file=sys.stderr,
        )
        for line in tail:
            print(line, file=sys.stderr)
    return returncode


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.frame_stride <= 0 or args.loops <= 0:
        raise ValueError("frame stride and loops must be positive")
    for label, value, allow_zero in (
        ("capture timeout", args.capture_timeout_s, False),
        ("replay speed", args.speed, False),
        ("endpoint hold", args.endpoint_hold_s, True),
        ("keep-open duration", args.keep_open_s, True),
    ):
        invalid = not math.isfinite(float(value)) or (
            float(value) < 0.0 if allow_zero else float(value) <= 0.0
        )
        if invalid:
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{label} must be finite and {qualifier}")

    source_report_path = Path(args.source_report).expanduser().resolve()
    trace_path = Path(args.trace_path).expanduser().resolve()
    capture_required = bool(args.refresh_trace or not trace_path.is_file())
    if capture_required:
        if not source_report_path.is_file():
            raise FileNotFoundError(f"source report not found: {source_report_path}")
        capture_command = build_capture_command(
            load_cone_proxy_source_command(source_report_path),
            trace_path=trace_path,
            frame_stride=int(args.frame_stride),
        )
        if args.print_command:
            print("capture:", shlex.join(capture_command))
            return 0
        print(
            "[order8-cone-lift-replay] Capturing the exact cone-proxy physical "
            "run headlessly once. Kit is not opened during capture.",
            flush=True,
        )
        capture_report = _run_json_command(
            capture_command,
            float(args.capture_timeout_s),
        )
        if not trace_path.is_file():
            raise RuntimeError(
                "headless capture returned without a state trace: "
                f"{capture_report.get('error', 'unknown error')}"
            )

    trace = load_order8_state_trace(trace_path)
    require_cone_proxy_trace(trace)
    summary = summarize_trace(trace)
    print(
        "[order8-cone-lift-replay] Validated cached physical trace: "
        f"frames={summary['frame_count']} "
        f"duration={summary['source_duration_s']:.2f}s "
        f"Dock_motion={math.degrees(float(summary['maximum_dock_motion_rad'])):.2f}deg "
        f"object_COM_rise={1000.0 * float(summary['maximum_object_com_rise_m']):.2f}mm",
        flush=True,
    )
    if args.capture_only:
        return 0

    replay_command = build_cone_proxy_replay_command(
        trace,
        trace_path=trace_path,
        speed=float(args.speed),
        loops=int(args.loops),
        endpoint_hold_s=float(args.endpoint_hold_s),
        keep_open_s=float(args.keep_open_s),
        normal_kit_log=bool(args.normal_kit_log),
        rendering_mode=str(args.rendering_mode),
    )
    if args.print_command:
        print("replay:", shlex.join(replay_command))
        return 0
    print(
        "[order8-cone-lift-replay] Opening low-load balanced Kit replay. "
        "The terminal reports independent PhysX Dock motion while the viewport "
        "shows the recorded trajectory; this is visual diagnostic evidence only.",
        flush=True,
    )
    environment = dict(os.environ)
    environment.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    return run_replay_command(
        replay_command,
        environment=environment,
        normal_kit_log=bool(args.normal_kit_log),
    )


if __name__ == "__main__":
    raise SystemExit(main())
