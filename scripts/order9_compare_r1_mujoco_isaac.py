#!/usr/bin/env python3
from __future__ import annotations

"""Compare one saved R1 Isaac episode with a closed-loop MuJoCo replay."""

import argparse
import json
from pathlib import Path
import time

import torch

from amsrr.simulation.order9_mujoco_trace_replay import (
    build_mujoco_model,
    load_isaac_episode_trace,
    replay_closed_loop_mujoco,
    replay_saved_actuator_trace_mujoco,
    resolve_source_urdf,
    write_comparison_result,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--isaac-rollout", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--environment-index", type=int, default=0)
    parser.add_argument("--episode-serial", type=int, default=0)
    parser.add_argument("--physics-dt-s", type=float, default=0.002)
    parser.add_argument("--control-dt-s", type=float, default=0.02)
    parser.add_argument(
        "--controller-mode",
        choices=("saved_actuator_trace", "closed_loop_qpid"),
        default="saved_actuator_trace",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    # The controller solves one tiny 24-rotor QP at a time.  In this serial
    # diagnostic, intra-operation thread pools cost more than the arithmetic.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    case = json.loads(args.case_manifest.read_text(encoding="utf-8"))
    trace = load_isaac_episode_trace(
        args.isaac_rollout,
        environment_index=args.environment_index,
        episode_serial=args.episode_serial,
    )
    expected_generation = f"r1_nominal_calibration:{case['candidate_id']}"
    if trace.metadata.get("generation_id") != expected_generation:
        raise ValueError("case manifest and Isaac trace candidate identities differ")
    if trace.metadata.get("pi_l_checkpoint_sha256") != case["c3_checkpoint"]["sha256"]:
        raise ValueError("case manifest and Isaac trace checkpoint hashes differ")
    started = time.perf_counter()
    context = build_mujoco_model(
        trace=trace,
        source_urdf=resolve_source_urdf(case),
        mesh_cache_directory=output / "convex_mesh_cache",
        physics_dt_s=args.physics_dt_s,
    )
    setup_wall_s = time.perf_counter() - started
    replay = (
        replay_saved_actuator_trace_mujoco
        if args.controller_mode == "saved_actuator_trace"
        else replay_closed_loop_mujoco
    )
    result = replay(trace=trace, context=context, control_dt_s=args.control_dt_s)
    result["timing"]["setup_wall_s"] = setup_wall_s
    result["case_manifest"] = {
        "path": str(args.case_manifest.resolve()),
        "candidate_id": case["candidate_id"],
        "level_id": case["level_id"],
        "source_bucket_id": case["source_bucket_id"],
        "module_count": case["module_count"],
    }
    result_path = output / "comparison_result.json"
    write_comparison_result(result, result_path)
    print(
        json.dumps(
            {
                "result": str(result_path),
                "mujoco_success": result["acceptance"]["mujoco_success"],
                "simulation_wall_s": result["timing"]["simulation_wall_s"],
                "realtime_factor": result["timing"]["realtime_factor"],
                "throughput_ratio_vs_isaac": result["timing"][
                    "throughput_ratio_vs_isaac_aggregate_env_steps"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
