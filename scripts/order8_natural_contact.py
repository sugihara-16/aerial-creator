from __future__ import annotations

"""Run the P4-full Order 8 free-object natural-contact substrate smoke."""

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from amsrr.robot_model.physical_model_builder import (
    build_physical_model_from_config,
)
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.order8 import load_order8_natural_contact_config
from amsrr.simulation.isaac_lab_backend import (
    IsaacLabBackend,
    load_isaac_lab_backend_config,
)
from amsrr.simulation.order8_natural_contact import (
    ORDER8_DEFAULT_GENERATED_USD_DIR,
    Order8IsaacNaturalContactEnv,
    build_representative_order8_morphology,
)
from amsrr.training.p2_inspection_context import default_grasp_carry_task_spec
from amsrr.utils.hashing import hash_file


DEFAULT_CONFIG_PATH = "configs/training/order8_natural_contact.yaml"
DEFAULT_BACKEND_CONFIG_PATH = "configs/env/isaac_lab.yaml"
DEFAULT_REPORT_PATH = Path("artifacts/p4_full/order8_natural_contact/report.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the deterministic three-module Order 8 natural-contact "
            "grasp/lift/transport/place/release smoke."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--backend-config", default=DEFAULT_BACKEND_CONFIG_PATH)
    parser.add_argument("--morphology-graph-json-path", default=None)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--viewer", choices=("kit",), default=None)
    parser.add_argument("--realtime-playback", action="store_true")
    parser.add_argument("--keep-open-after-rollout-s", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--order9-teacher-output", default=None)
    parser.add_argument("--order9-teacher-episode-id", default=None)
    parser.add_argument("--order9-teacher-task-id", default=None)
    parser.add_argument(
        "--order9-teacher-split",
        choices=("train", "validation", "held_out"),
        default="train",
    )
    parser.add_argument("--order9-teacher-low-level-stride", type=int, default=1)
    parser.add_argument("--order9-teacher-high-level-stride", type=int, default=5)
    parser.add_argument("--order9-teacher-window-horizon-s", type=float, default=2.0)
    parser.add_argument("--order9-teacher-window-knot-dt-s", type=float, default=0.1)
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument(
        "--reuse-generated-asset",
        action="store_true",
        help="Reuse the hash-audited generated USD instead of forcing conversion.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.viewer is not None and not args.real:
        parser.error("--viewer requires --real")
    if args.viewer is None and (
        args.realtime_playback or args.keep_open_after_rollout_s > 0.0
    ):
        parser.error(
            "--realtime-playback/--keep-open-after-rollout-s require --viewer kit"
        )
    if args.keep_open_after_rollout_s < 0.0:
        parser.error("--keep-open-after-rollout-s must be non-negative")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.order9_teacher_output is not None and not args.real:
        parser.error("--order9-teacher-output requires --real")
    if (
        args.order9_teacher_low_level_stride < 1
        or args.order9_teacher_high_level_stride < 1
    ):
        parser.error("Order9 teacher strides must be positive")
    if (
        args.order9_teacher_window_horizon_s <= 0.0
        or args.order9_teacher_window_knot_dt_s <= 0.0
    ):
        parser.error("Order9 teacher window values must be positive")

    config_path = Path(args.config).resolve()
    backend_config_path = Path(args.backend_config).resolve()
    config = load_order8_natural_contact_config(config_path)
    backend_config = load_isaac_lab_backend_config(backend_config_path)
    robot_model_config_path = Path(
        backend_config.robot_model_config_path
    ).resolve()
    physical_model = build_physical_model_from_config(robot_model_config_path)
    task_spec = default_grasp_carry_task_spec()
    if args.morphology_graph_json_path is None:
        morphology = build_representative_order8_morphology(physical_model)
        graph_source = "current_symmetric_two_anchor_builder"
    else:
        graph_path = Path(args.morphology_graph_json_path).resolve()
        morphology = MorphologyGraph.from_json(
            graph_path.read_text(encoding="utf-8")
        )
        graph_source = str(graph_path)

    env = Order8IsaacNaturalContactEnv(
        config=config,
        backend=IsaacLabBackend(backend_config),
        physical_model=physical_model,
        backend_config_path=backend_config_path,
        generated_usd_dir=ORDER8_DEFAULT_GENERATED_USD_DIR,
        viewer=args.viewer,
        realtime_playback=bool(args.realtime_playback),
        keep_open_after_rollout_s=float(args.keep_open_after_rollout_s),
        seed=int(args.seed),
        order9_teacher_output=args.order9_teacher_output,
        order9_teacher_episode_id=args.order9_teacher_episode_id,
        order9_teacher_task_id=args.order9_teacher_task_id,
        order9_teacher_split=args.order9_teacher_split,
        order9_teacher_low_level_stride=args.order9_teacher_low_level_stride,
        order9_teacher_high_level_stride=args.order9_teacher_high_level_stride,
        order9_teacher_window_horizon_s=args.order9_teacher_window_horizon_s,
        order9_teacher_window_knot_dt_s=args.order9_teacher_window_knot_dt_s,
        force_convert=not args.reuse_generated_asset,
    )
    probe_command = env.build_probe_command(morphology)
    result = env.run(morphology, dry_run=not args.real)
    result.report["run_provenance"] = {
        "graph_source": graph_source,
        "task_id": task_spec.task_id,
        "task_spec": task_spec.to_dict(),
        "task_spec_hash": task_spec.stable_hash(),
        "graph_id": morphology.graph_id,
        "graph_hash": morphology.stable_hash(),
        "morphology_graph": morphology.to_dict(),
        "config_path": str(config_path),
        "config_file_sha256": hash_file(config_path),
        "config": config.to_dict(),
        "config_hash": config.stable_hash(),
        "backend_config_path": str(backend_config_path),
        "backend_config_file_sha256": hash_file(backend_config_path),
        "backend_config_hash": backend_config.stable_hash(),
        "robot_model_config_path": str(robot_model_config_path),
        "robot_model_config_file_sha256": hash_file(robot_model_config_path),
        "physical_model_hash": physical_model.stable_hash(),
        "source_urdf_path": str(Path(physical_model.urdf_path).resolve()),
        "source_urdf_sha256": env.source_urdf_hash,
        "collision_geometry_content_hash": env.collision_geometry_hash,
        "requested_steps": env.requested_steps,
        "seed": env.seed,
        "simulation_dt_s": env.simulation_dt_s,
        "rollout_budget_s": env.rollout_budget_s,
        "generated_usd_dir": str(Path(env.generated_usd_dir).resolve()),
        "force_convert": env.force_convert,
        "real_requested": bool(args.real),
        "order9_teacher_output": args.order9_teacher_output,
        "order9_teacher_episode_id": args.order9_teacher_episode_id,
        "order9_teacher_task_id": args.order9_teacher_task_id,
        "order9_teacher_split": args.order9_teacher_split,
    }
    output_path = Path(args.report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.to_dict(), sort_keys=True, indent=2),
        encoding="utf-8",
    )
    summary: dict[str, object] = {
        "env_version": result.env_version,
        "graph_id": result.graph_id,
        "graph_hash": result.graph_hash,
        "config_hash": result.config_hash,
        "dry_run": result.dry_run,
        "attempted": result.attempted,
        "isaac_backed": result.isaac_backed,
        "passed": result.passed,
        "report_validation_failures": result.report_validation_failures,
        "failure_reason": result.failure_reason,
        "report_path": str(output_path),
    }
    if args.print_command:
        summary["probe_command"] = probe_command
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0 if result.dry_run or result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
