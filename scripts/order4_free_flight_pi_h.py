from __future__ import annotations

"""Run the P4-full Order 4 deterministic free-flight pi_H in Isaac Lab."""

import argparse
import json
from pathlib import Path
import secrets
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from amsrr.feasibility.morphology_flight import (
    MorphologyFlightFeasibilityChecker,
    MorphologyFlightFeasibilityConfig,
)
from amsrr.morphology.random_feasible import (
    RandomFeasibleConnectedMorphologyDistribution,
    RandomFeasibleMorphologyConfig,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.order4 import (
    Order4DeterministicPlannerConfig,
    Order4FreeFlightMission,
    Order4FreeFlightWaypoint,
    build_order4_free_flight_mission,
)
from amsrr.simulation.isaac_lab_backend import IsaacLabBackend, load_isaac_lab_backend_config
from amsrr.simulation.order4_free_flight import (
    Order4IsaacFreeFlightConfig,
    Order4IsaacFreeFlightEnv,
)
from amsrr.simulation.random_morphology_takeoff import (
    RandomMorphologyTakeoffConfig,
    RandomMorphologyTakeoffEnv,
)
from amsrr.utils.config import load_config
from amsrr.utils.hashing import hash_file


DEFAULT_CONFIG_PATH = "configs/training/order4_deterministic_pi_h.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample a feasible connected morphology and run floor settle -> "
            "takeoff -> multi-waypoint -> final hover through the deterministic pi_H runtime."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--module-count",
        type=int,
        choices=range(2, 9),
        default=3,
        metavar="{2,3,4,5,6,7,8}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Morphology seed; omitted samples a fresh seed and prints it.",
    )
    parser.add_argument("--max-attempts", type=int, default=256)
    parser.add_argument(
        "--morphology-graph-json-path",
        default=None,
        help="Use an existing current-URDF MorphologyGraph instead of sampling.",
    )
    parser.add_argument("--mission-json-path", default=None)
    parser.add_argument(
        "--waypoint",
        type=float,
        nargs=6,
        action="append",
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="Override mission waypoints as hover-relative xyz/rpy in radians; repeat at least twice.",
    )
    parser.add_argument(
        "--final-hover-hold-s",
        type=float,
        default=None,
        help="Override the automated final hover dwell (default 5 s).",
    )
    parser.add_argument(
        "--endurance",
        action="store_true",
        help="Require a 20 s final hover instead of the 5 s acceptance dwell.",
    )
    parser.add_argument(
        "--pi-l-checkpoint-path",
        default=None,
        help="Optional compatible Order-3 checkpoint; omitted uses deterministic baseline pi_L.",
    )
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--viewer", choices=("kit",), default=None)
    parser.add_argument("--realtime-playback", action="store_true")
    parser.add_argument("--keep-open-after-rollout-s", type=float, default=0.0)
    parser.add_argument(
        "--report-path",
        default="artifacts/p4_full/order4_deterministic_pi_h/report.json",
    )
    parser.add_argument("--print-command", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.max_attempts <= 0:
        parser.error("--max-attempts must be positive")
    if args.viewer is None and (
        args.realtime_playback or args.keep_open_after_rollout_s > 0.0
    ):
        parser.error("real-time playback/post-rollout hold requires --viewer kit")
    if args.viewer is not None and not args.real:
        parser.error("--viewer requires --real")

    config_path = Path(args.config).expanduser().resolve()
    data = load_config(config_path)
    takeoff_config = RandomMorphologyTakeoffConfig.from_dict(data["takeoff"])
    order4_data = data["order4"]
    planner_config = Order4DeterministicPlannerConfig.from_dict(
        order4_data["planner"]
    )
    mission = _mission_from_arguments(args, order4_data["mission"])
    seed = secrets.randbits(63) if args.seed is None else int(args.seed)

    physical_model = build_physical_model_from_config(
        takeoff_config.robot_model_config_path
    )
    feasibility_checker = MorphologyFlightFeasibilityChecker(
        MorphologyFlightFeasibilityConfig(
            mesh_search_dirs=tuple(takeoff_config.mesh_search_dirs)
        )
    )
    if args.morphology_graph_json_path:
        morphology = MorphologyGraph.from_json(
            Path(args.morphology_graph_json_path).read_text(encoding="utf-8")
        )
        sampling_summary: dict[str, Any] = {
            "source": "external_graph",
            "path": str(args.morphology_graph_json_path),
        }
    else:
        distribution = RandomFeasibleConnectedMorphologyDistribution(
            physical_model,
            feasibility_checker=feasibility_checker,
            config=RandomFeasibleMorphologyConfig(
                max_attempts_per_sample=int(args.max_attempts)
            ),
        )
        try:
            sampled = distribution.sample_with_report(
                seed=seed,
                module_count=int(args.module_count),
            )
        except SchemaValidationError as exc:
            parser.error(str(exc))
        morphology = sampled.morphology_graph
        sampling_summary = {
            "source": "random_feasible_connected_distribution",
            "requested_seed": sampled.requested_seed,
            "accepted_proposal_seed": sampled.accepted_proposal_seed,
            "attempt_count": sampled.attempt_count,
            "structural_hash": sampled.structural_hash,
        }

    backend = IsaacLabBackend(
        load_isaac_lab_backend_config(takeoff_config.backend_config_path)
    )
    takeoff_env = RandomMorphologyTakeoffEnv(
        config=takeoff_config,
        backend=backend,
        physical_model=physical_model,
    )
    checkpoint_hash = (
        hash_file(args.pi_l_checkpoint_path)
        if args.pi_l_checkpoint_path is not None
        else None
    )
    env = Order4IsaacFreeFlightEnv(
        config=Order4IsaacFreeFlightConfig(
            mission=mission,
            planner=planner_config,
            pi_l_checkpoint_path=args.pi_l_checkpoint_path,
            expected_pi_l_checkpoint_sha256=checkpoint_hash,
            command_timeout_s=float(order4_data.get("command_timeout_s", 600.0)),
        ),
        takeoff_env=takeoff_env,
        viewer=args.viewer,
        realtime_playback=bool(args.realtime_playback),
        keep_open_after_rollout_s=float(args.keep_open_after_rollout_s),
    )
    result = env.run(morphology, dry_run=not args.real)
    output_path = Path(args.report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.to_dict(), sort_keys=True, indent=2),
        encoding="utf-8",
    )
    summary = {
        "env_version": result.env_version,
        "graph_id": result.graph_id,
        "module_count": len(morphology.modules),
        "seed": seed,
        "sampling": sampling_summary,
        "mission_id": mission.mission_id,
        "mission_hash": mission.mission_hash,
        "waypoint_count": len(mission.waypoints),
        "final_hover_hold_s": mission.final_hover_hold_s,
        "pi_l_source": (
            "checkpoint" if args.pi_l_checkpoint_path is not None else "deterministic_baseline"
        ),
        "dry_run": result.dry_run,
        "attempted": result.attempted,
        "isaac_backed": result.isaac_backed,
        "passed": result.passed,
        "report_validation_failures": result.report_validation_failures,
        "failure_reason": result.failure_reason,
        "report_path": str(output_path),
    }
    if args.print_command and result.dry_run:
        summary["probe_command"] = result.report["probe_command"]
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0 if result.dry_run or result.passed else 1


def _mission_from_arguments(
    args: argparse.Namespace,
    config_payload: dict[str, Any],
) -> Order4FreeFlightMission:
    if args.mission_json_path and args.waypoint:
        raise SchemaValidationError(
            "--mission-json-path and --waypoint are mutually exclusive"
        )
    if args.mission_json_path:
        mission = Order4FreeFlightMission.from_json(
            Path(args.mission_json_path).read_text(encoding="utf-8")
        )
    else:
        waypoint_payloads = list(config_payload["waypoints"])
        if args.waypoint:
            if len(args.waypoint) < 2:
                raise SchemaValidationError("--waypoint must be repeated at least twice")
            waypoint_payloads = [
                {
                    "waypoint_id": f"cli_waypoint_{index}",
                    "position_offset_world": list(values[:3]),
                    "orientation_rpy_rad": list(values[3:]),
                    "transition_duration_s": 2.5,
                    "dwell_s": 0.5,
                    "timeout_s": 9.0,
                }
                for index, values in enumerate(args.waypoint)
            ]
        hold_s = float(config_payload["final_hover_hold_s"])
        if args.final_hover_hold_s is not None:
            hold_s = float(args.final_hover_hold_s)
        if args.endurance:
            hold_s = max(hold_s, 20.0)
        timeout_s = max(
            float(config_payload["mission_timeout_s"]),
            hold_s + 25.0,
        )
        mission = build_order4_free_flight_mission(
            mission_id=str(config_payload["mission_id"]),
            waypoints=[
                Order4FreeFlightWaypoint.from_dict(payload)
                for payload in waypoint_payloads
            ],
            hover_height_delta_m=float(config_payload["hover_height_delta_m"]),
            hover_acquisition_dwell_s=float(
                config_payload["hover_acquisition_dwell_s"]
            ),
            final_hover_hold_s=hold_s,
            mission_timeout_s=timeout_s,
        )
    if args.final_hover_hold_s is not None or args.endurance:
        hold_s = max(
            20.0 if args.endurance else 0.0,
            float(args.final_hover_hold_s)
            if args.final_hover_hold_s is not None
            else mission.final_hover_hold_s,
        )
        mission = build_order4_free_flight_mission(
            mission_id=mission.mission_id,
            waypoints=mission.waypoints,
            hover_height_delta_m=mission.hover_height_delta_m,
            hover_acquisition_dwell_s=mission.hover_acquisition_dwell_s,
            final_hover_hold_s=hold_s,
            mission_timeout_s=max(mission.mission_timeout_s, hold_s + 25.0),
        )
    return mission


if __name__ == "__main__":
    raise SystemExit(main())
