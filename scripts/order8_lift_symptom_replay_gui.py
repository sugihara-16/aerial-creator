from __future__ import annotations

"""Replay the current 1 kg Order-8 lift symptom in a low-load Kit GUI.

The physical state trace is captured headlessly from the exact diagnostic
command. Kit then advances a gravity-free, contact-minimized synchronization
step for each displayed state before reapplying the exact recorded state. The
object, floor, and support colliders are disabled, self-collision remains off,
and graph constraints are disabled; authored inter-module collision geometry
is retained. This is substantially lighter than running the contact simulation
and RTX viewer together, while ensuring that PhysX, Fabric, the viewport, and
the runtime joint inspector see the same state. Replay is diagnostic-only and
is never acceptance evidence.
"""

import argparse
import json
import math
import os
from pathlib import Path
import shlex
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
    build_capture_command,
    build_replay_command,
)


DEFAULT_SOURCE_REPORT = REPO_ROOT / (
    "artifacts/p4_full/order8_natural_contact/diagnostics/"
    "compliant_authored_mesh_mu10_k7000_c75_loaded_rebase_no_accel_"
    "v359_30s.json"
)
DEFAULT_TRACE_PATH = REPO_ROOT / (
    "artifacts/p4_full/order8_natural_contact/diagnostics/"
    "loaded_rebase_1kg_v359_visual_state_trace.json"
)
EXPECTED_OBJECT_MASS_KG = 1.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the current 1 kg Order-8 lift/slip symptom using a "
            "headlessly captured, contact-minimized PhysX-synchronized Kit replay."
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
        default=0.75,
        help="Replay speed relative to recorded simulation time.",
    )
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--endpoint-hold-s", type=float, default=2.0)
    parser.add_argument("--keep-open-s", type=float, default=30.0)
    parser.add_argument(
        "--rendering-mode",
        choices=("performance", "balanced", "quality"),
        default="balanced",
        help=(
            "Kit rendering preset. Balanced is the default because the "
            "performance preset produced a black normal-mesh viewport on the "
            "current workstation."
        ),
    )
    parser.add_argument(
        "--normal-kit-log",
        action="store_true",
        help="Retain normal Kit warning output instead of showing errors only.",
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print the capture/replay command without launching it.",
    )
    return parser


def load_one_kg_source_command(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("source diagnostic report must be a JSON object")
    if payload.get("diagnostic_only") is not True:
        raise ValueError("source diagnostic report must be diagnostic-only")
    if payload.get("acceptance_eligible") is not False:
        raise ValueError("source diagnostic report must be acceptance-ineligible")
    config = payload.get("diagnostic_config")
    if not isinstance(config, dict):
        raise ValueError("source diagnostic report has no diagnostic_config")
    mass_kg = float(config.get("object_mass_kg", math.nan))
    if not math.isclose(mass_kg, EXPECTED_OBJECT_MASS_KG, abs_tol=1.0e-12):
        raise ValueError(
            "lift-symptom GUI requires the 1.0 kg source diagnostic; "
            f"report contains {mass_kg!r} kg"
        )
    command = payload.get("probe_command")
    if not isinstance(command, list) or any(
        not isinstance(value, str) or not value for value in command
    ):
        raise ValueError("source diagnostic report has no valid probe_command")
    return [str(value) for value in command]


def trace_object_mass_kg(trace: dict[str, object]) -> float:
    source_argv = trace.get("source_probe_argv")
    if not isinstance(source_argv, list) or any(
        not isinstance(value, str) for value in source_argv
    ):
        raise ValueError("state trace has no valid source_probe_argv")
    try:
        config_index = source_argv.index("--order8-config-json")
        config = json.loads(str(source_argv[config_index + 1]))
    except (ValueError, IndexError, json.JSONDecodeError) as exc:
        raise ValueError("state trace has no valid Order-8 config JSON") from exc
    if not isinstance(config, dict):
        raise ValueError("state-trace Order-8 config must be an object")
    mass_kg = float(config.get("object_mass_kg", math.nan))
    if not math.isclose(mass_kg, EXPECTED_OBJECT_MASS_KG, abs_tol=1.0e-12):
        raise ValueError(
            "lift-symptom GUI refuses a non-1.0 kg trace; "
            f"trace contains {mass_kg!r} kg"
        )
    return mass_kg


def build_low_load_replay_command(
    trace: dict[str, object],
    *,
    trace_path: Path,
    speed: float,
    loops: int,
    endpoint_hold_s: float,
    keep_open_s: float,
    normal_kit_log: bool,
    rendering_mode: str = "balanced",
) -> list[str]:
    command = build_replay_command(
        trace,
        trace_path=trace_path,
        speed=speed,
        loops=loops,
        keep_open_s=keep_open_s,
        endpoint_hold_s=endpoint_hold_s,
    )
    command.append("--order8-state-trace-replay-sync-physics")
    if rendering_mode not in {"performance", "balanced", "quality"}:
        raise ValueError(f"unsupported rendering mode: {rendering_mode}")
    command.extend(["--rendering_mode", rendering_mode])
    if not normal_kit_log:
        command.append("--kit_args=--/log/outputStreamLevel=Error")
    return command


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
            load_one_kg_source_command(source_report_path),
            trace_path=trace_path,
            frame_stride=int(args.frame_stride),
        )
        if args.print_command:
            print("capture:", shlex.join(capture_command))
            return 0
        print(
            "[order8-lift-replay] Capturing the exact 1.0 kg physical run "
            "headlessly. No GUI is opened during this one-time step.",
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
        print(
            "[order8-lift-replay] Capture complete: "
            f"simulation_time={capture_report.get('order8_natural_contact_simulation_time_s')}s",
            flush=True,
        )

    trace = load_order8_state_trace(trace_path)
    trace_object_mass_kg(trace)
    if args.capture_only:
        print(f"[order8-lift-replay] Validated 1.0 kg trace: {trace_path}")
        return 0

    replay_command = build_low_load_replay_command(
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
        "[order8-lift-replay] Opening low-load Kit replay at 1.0 kg. "
        "Each frame uses one contact-minimized PhysX/Fabric synchronization step; "
        "use it for visual diagnosis only.",
        flush=True,
    )
    environment = dict(os.environ)
    environment.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.execvpe(replay_command[0], replay_command, environment)
    raise AssertionError("os.execvpe unexpectedly returned")


if __name__ == "__main__":
    raise SystemExit(main())
