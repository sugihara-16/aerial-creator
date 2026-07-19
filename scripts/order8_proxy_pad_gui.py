from __future__ import annotations

"""Inspect the current Order-8 diagnostic proxy pads in the Isaac GUI."""

import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from order8_current_grasp_gui import (
    _ensure_isaaclab_environment,
    _source_probe_command,
    _strip_trace_and_viewer_options,
)
from amsrr.schemas.order8 import (
    ORDER8_NATURAL_CONTACT_CONFIG_VERSION,
    Order8NaturalContactConfig,
)


DEFAULT_SOURCE_REPORT = REPO_ROOT / (
    "artifacts/p4_full/order8_natural_contact/diagnostics/"
    "slip60mmps_cumulative30mm_kp200_kd8_v342_30s.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Spawn the two current Order-8 finite-area proxy pads in the Isaac "
            "GUI. This is a diagnostic placement/contact inspection and cannot satisfy "
            "Order-8 acceptance."
        )
    )
    parser.add_argument(
        "--source-report",
        default=str(DEFAULT_SOURCE_REPORT),
        help=(
            "Diagnostic JSON whose recorded probe command contains the proxy-pad "
            "and near-contact fixture configuration."
        ),
    )
    parser.add_argument(
        "--state",
        choices=("grasp", "contact", "qclose", "open"),
        default="grasp",
        help=(
            "Continuously simulate to the recorded two-pad grasp (default; "
            "'contact' is an alias), restore the dynamically fragile q_close "
            "checkpoint, or show the earlier open fixture."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "Override the physics steps before the viewer is held open. The "
            "default is derived from the first recorded grasp for grasp/contact "
            "and is 1 for qclose/open."
        ),
    )
    parser.add_argument(
        "--keep-open-s",
        type=float,
        default=120.0,
        help="Wall-clock seconds to keep the Kit viewer open (default: 120).",
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print the Isaac launch command without executing it.",
    )
    return parser


def _strip_fixed_arity_option(
    command: list[str],
    option: str,
    arity: int,
) -> list[str]:
    stripped: list[str] = []
    index = 0
    while index < len(command):
        if command[index] != option:
            stripped.append(command[index])
            index += 1
            continue
        if index + arity >= len(command):
            raise ValueError(f"proxy-pad source has incomplete {option}")
        index += arity + 1
    return stripped


def source_command_at_qclose(
    source_command: list[str],
    report: dict[str, object],
) -> list[str]:
    """Replace the recorded open fixture with its exact q_close checkpoint."""

    base_pose = report.get("order8_natural_contact_qclose_checkpoint_base_pose")
    joint_positions = report.get(
        "order8_natural_contact_qclose_checkpoint_joint_positions_rad"
    )
    checkpoint_state = report.get(
        "order8_natural_contact_qclose_checkpoint_state"
    )
    if (
        not isinstance(base_pose, list)
        or len(base_pose) != 7
        or not all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in base_pose
        )
        or not isinstance(joint_positions, dict)
        or not joint_positions
        or not isinstance(checkpoint_state, dict)
        or checkpoint_state.get("schema_version")
        != "order8_qclose_checkpoint_state_v1"
    ):
        raise ValueError("proxy-pad source report has no complete q_close checkpoint")

    command = list(source_command)
    for option, arity in (
        ("--order8-diagnostic-near-contact-base-pose", 7),
        ("--order8-diagnostic-near-contact-joint-positions-json", 1),
        ("--order8-diagnostic-near-contact-object-pose", 7),
        ("--order8-diagnostic-qclose-base-pose", 7),
        ("--order8-diagnostic-qclose-joint-positions-json", 1),
        ("--order8-diagnostic-qclose-state-json", 1),
    ):
        command = _strip_fixed_arity_option(command, option, arity)
    command = [
        value
        for value in command
        if value != "--order8-diagnostic-qclose-zero-velocities"
    ]
    command.extend(
        [
            "--order8-diagnostic-qclose-base-pose",
            *(str(float(value)) for value in base_pose),
            "--order8-diagnostic-qclose-joint-positions-json",
            json.dumps(joint_positions, sort_keys=True),
            "--order8-diagnostic-qclose-state-json",
            json.dumps(checkpoint_state, sort_keys=True),
            "--order8-diagnostic-qclose-zero-velocities",
        ]
    )
    return command


def stable_grasp_step_count(report: dict[str, object]) -> int:
    """Return the first continuously simulated two-contact grasp step count."""

    dt = report.get("order8_natural_contact_simulation_dt_s")
    evidence = report.get("order8_natural_contact_step_evidence")
    if (
        not isinstance(dt, (int, float))
        or not math.isfinite(float(dt))
        or float(dt) <= 0.0
        or not isinstance(evidence, list)
    ):
        raise ValueError("proxy-pad source has no valid grasp timeline")
    for sample in evidence:
        if not isinstance(sample, dict) or sample.get("grasp_acquired") is not True:
            continue
        time_s = sample.get("time_s")
        selected_links = sample.get("selected_contact_link_ids")
        if (
            not isinstance(time_s, (int, float))
            or not math.isfinite(float(time_s))
            or float(time_s) < 0.0
            or not isinstance(selected_links, list)
            or len(set(selected_links)) < 2
        ):
            continue
        # The runtime measures evidence before advancing that loop iteration.
        # Include the measured sample and one final step, then freeze physics.
        return int(math.ceil(float(time_s) / float(dt) - 1.0e-9)) + 1
    raise ValueError("proxy-pad source never acquired a two-pad grasp")


def _replace_unique_value_option(
    command: list[str],
    option: str,
    value: str,
) -> None:
    indices = [index for index, item in enumerate(command) if item == option]
    if len(indices) != 1:
        raise ValueError(f"proxy-pad source must contain exactly one {option}")
    index = indices[0]
    if index + 1 >= len(command):
        raise ValueError(f"proxy-pad source has no value after {option}")
    command[index + 1] = value


def upgrade_legacy_order8_config(source_command: list[str]) -> list[str]:
    """Migrate the recorded v10 proxy command to the current v11 schema.

    The proxy remains diagnostic-only.  This narrow migration keeps its
    historical GUI inspector executable after normal Order 8 gained the two
    uniform compliant-contact material fields; it does not make old evidence
    acceptance-eligible.
    """

    command = list(source_command)
    indices = [
        index for index, value in enumerate(command) if value == "--order8-config-json"
    ]
    if not indices:
        return command
    if len(indices) != 1 or indices[0] + 1 >= len(command):
        raise ValueError("proxy-pad source has invalid --order8-config-json")
    index = indices[0] + 1
    payload = json.loads(command[index])
    if not isinstance(payload, dict):
        raise ValueError("proxy-pad source Order 8 config must be an object")
    version = payload.get("config_version")
    if version == ORDER8_NATURAL_CONTACT_CONFIG_VERSION:
        Order8NaturalContactConfig.from_dict(payload)
        return command
    if version != "order8_natural_contact_config_v10":
        raise ValueError(
            "proxy-pad source config is neither current nor the supported legacy v10"
        )
    defaults = Order8NaturalContactConfig()
    payload["config_version"] = ORDER8_NATURAL_CONTACT_CONFIG_VERSION
    payload.setdefault(
        "selected_gripper_compliant_contact_stiffness_n_per_m",
        defaults.selected_gripper_compliant_contact_stiffness_n_per_m,
    )
    payload.setdefault(
        "selected_gripper_compliant_contact_damping_n_s_per_m",
        defaults.selected_gripper_compliant_contact_damping_n_s_per_m,
    )
    migrated = Order8NaturalContactConfig.from_dict(payload)
    command[index] = json.dumps(migrated.to_dict(), sort_keys=True)
    return command


def build_proxy_pad_spawn_command(
    source_command: list[str],
    *,
    steps: int,
    keep_open_s: float,
) -> list[str]:
    """Build a one-shot live-physics spawn command from recorded evidence."""

    if steps <= 0:
        raise ValueError("proxy-pad GUI steps must be positive")
    if not math.isfinite(keep_open_s) or keep_open_s <= 0.0:
        raise ValueError("proxy-pad GUI keep-open duration must be positive")

    command = _strip_trace_and_viewer_options(list(source_command))
    required_flags = (
        "--order8-natural-contact",
        "--order8-diagnostic-only",
        "--order8-diagnostic-proxy-pad",
    )
    missing = [flag for flag in required_flags if flag not in command]
    if missing:
        raise ValueError(
            "proxy-pad GUI source is missing required diagnostic flags: "
            f"{missing}"
        )

    _replace_unique_value_option(command, "--steps", str(int(steps)))
    # Do not use --realtime-playback: only the requested initialization steps
    # are advanced, after which Kit updates rendering without advancing physics.
    command.extend(
        [
            "--viz",
            "kit",
            "--keep-open-after-smoke-s",
            str(float(keep_open_s)),
        ]
    )
    return _ensure_isaaclab_environment(command)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_report = Path(args.source_report).expanduser().resolve()
    if not source_report.is_file():
        raise FileNotFoundError(
            "proxy-pad source report was not found; pass another report produced "
            f"with --proxy-pad: {source_report}"
        )
    payload = json.loads(source_report.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("report"), dict):
        raise ValueError("proxy-pad source report has no runtime report map")
    source_command = _source_probe_command(source_report)
    source_command = upgrade_legacy_order8_config(source_command)
    if args.state == "qclose":
        source_command = source_command_at_qclose(
            source_command,
            payload["report"],
        )
    steps = (
        int(args.steps)
        if args.steps is not None
        else (
            stable_grasp_step_count(payload["report"])
            if args.state in {"grasp", "contact"}
            else 1
        )
    )
    command = build_proxy_pad_spawn_command(
        source_command,
        steps=steps,
        keep_open_s=float(args.keep_open_s),
    )
    state_description = (
        "continuous two-pad grasp"
        if args.state in {"grasp", "contact"}
        else args.state
    )
    print(
        "[order8-proxy-pad-gui] Spawning the exact orange 30 x 30 x 2 mm "
        f"colliders in the recorded {state_description} state. Physics stops "
        f"after {steps} step(s); this view is diagnostic-only.",
        flush=True,
    )
    if args.print_command:
        print(shlex.join(command))
        return 0
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
