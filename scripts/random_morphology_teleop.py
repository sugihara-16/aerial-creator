from __future__ import annotations

import argparse
import math
import os
import secrets
import sys
from pathlib import Path


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
from amsrr.simulation.isaac_lab_backend import IsaacLabBackend, load_isaac_lab_backend_config
from amsrr.simulation.random_morphology_takeoff import RandomMorphologyTakeoffEnv
from amsrr.simulation.random_morphology_teleop import (
    TELEOP_HELP,
    RandomMorphologyTeleopConfig,
    build_random_morphology_teleop_probe_command,
)
from amsrr.training.random_morphology_takeoff_runner import (
    load_random_morphology_takeoff_runner_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sample a feasible connected morphology, launch Isaac Lab Kit, "
            "take off, hover, and accept terminal keyboard pose commands."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/training/random_morphology_takeoff.yaml",
        help="Order-2 takeoff configuration.",
    )
    parser.add_argument(
        "--module-count",
        type=int,
        choices=range(2, 9),
        default=3,
        metavar="{2,3,4,5,6,7,8}",
        help="Number of connected Holon modules to sample (default: 3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Reproducible sampler seed; omitted uses a fresh random seed.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=256,
        help="Maximum deterministic feasibility-rejection attempts.",
    )
    parser.add_argument(
        "--translation-step-m",
        type=float,
        default=0.05,
        help="Position-target increment per W/S/A/D/R/F keypress.",
    )
    parser.add_argument(
        "--rotation-step-deg",
        type=float,
        default=5.0,
        help="Attitude-target increment per I/K/U/O/J/L keypress.",
    )
    parser.add_argument(
        "--max-roll-pitch-deg",
        type=float,
        default=30.0,
        help="Absolute roll/pitch target safety bound.",
    )
    parser.add_argument(
        "--max-position-lead-m",
        type=float,
        default=0.50,
        help="Maximum target-position distance ahead of the measured robot pose.",
    )
    args = parser.parse_args()

    if not sys.stdin.isatty():
        parser.error("interactive teleoperation requires a terminal (TTY) on stdin")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.max_attempts <= 0:
        parser.error("--max-attempts must be positive")

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    os.chdir(REPO_ROOT)
    _, takeoff_config = load_random_morphology_takeoff_runner_config(config_path)
    teleop_config = RandomMorphologyTeleopConfig(
        translation_step_m=float(args.translation_step_m),
        rotation_step_rad=math.radians(float(args.rotation_step_deg)),
        max_roll_pitch_rad=math.radians(float(args.max_roll_pitch_deg)),
        max_position_lead_m=float(args.max_position_lead_m),
    )
    teleop_config.validate()

    seed = secrets.randbits(63) if args.seed is None else int(args.seed)
    physical_model = build_physical_model_from_config(
        takeoff_config.robot_model_config_path
    )
    feasibility_checker = MorphologyFlightFeasibilityChecker(
        MorphologyFlightFeasibilityConfig(
            mesh_search_dirs=tuple(takeoff_config.mesh_search_dirs)
        )
    )
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

    backend = IsaacLabBackend(
        load_isaac_lab_backend_config(takeoff_config.backend_config_path)
    )
    env = RandomMorphologyTakeoffEnv(
        config=takeoff_config,
        backend=backend,
        physical_model=physical_model,
    )
    command = build_random_morphology_teleop_probe_command(
        env,
        sampled.morphology_graph,
        config=teleop_config,
    )

    print(
        "Sampled feasible morphology: "
        f"modules={args.module_count} seed={seed} "
        f"proposal_seed={sampled.accepted_proposal_seed} "
        f"attempts={sampled.attempt_count} "
        f"structural_hash={sampled.structural_hash}",
        flush=True,
    )
    print("Isaac Lab Kit will open; terminal control starts after takeoff/hover.", flush=True)
    print(TELEOP_HELP, flush=True)

    process_env = os.environ.copy()
    process_env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    process_env.setdefault("WARP_CACHE_PATH", "/tmp/amsrr_warp_cache")
    existing_pythonpath = process_env.get("PYTHONPATH")
    process_env["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not existing_pythonpath
        else f"{REPO_ROOT}{os.pathsep}{existing_pythonpath}"
    )
    os.execvpe(command[0], command, process_env)
    return 1  # pragma: no cover - os.execvpe replaces this process.


if __name__ == "__main__":
    raise SystemExit(main())
