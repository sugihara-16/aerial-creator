#!/usr/bin/env python3
from __future__ import annotations

"""Repair the two v16 object drops with an alternate stable grasp."""

import argparse
import json
import os
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.schemas.common import SchemaValidationError  # noqa: E402
from amsrr.training.order9_r1_collision_avoidance_v17 import (  # noqa: E402
    Order9R1CollisionAvoidanceHeuristicV17,
)
from scripts import order9_run_r1_collision_avoidance_v17 as v17  # noqa: E402
from scripts import order9_run_r1_held_out_scene_confirmation_v16 as v16  # noqa: E402

VERSION = "order9_r1_object_drop_alternate_grasp_v18"
STRUCTURAL_PREFIX = "58a5c015f4db"
BUCKET_ID = "held-out-000045-58a5c015f4db"
SAMPLES = (0, 26)
OUTPUT_ROOT = REPOSITORY / (
    "artifacts/p4_full/order9/r1_teacher/diagnostics/"
    "held_out_object_drop_alternate_grasp_v18"
)
ISAAC_PYTHON = "/home/leus/.local/share/mamba/envs/isaaclab3/bin/python"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-index", type=int, choices=SAMPLES)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--isaac-python", default=ISAAC_PYTHON)
    parser.add_argument("--prepare-only", action="store_true")
    return parser


GRASP_OPTIONS = (
    ((12, 23), "slot_0:grasp_pair:3", 0.01),
    ((12, 23), "slot_0:grasp_pair:2", 0.01),
)


def _heuristic(
    sample_index: int,
    *,
    ports: tuple[int, int],
    group: str,
    height_offset_m: float,
) -> Order9R1CollisionAvoidanceHeuristicV17:
    group_index = group.rsplit(":", 1)[-1]
    value = Order9R1CollisionAvoidanceHeuristicV17(
        heuristic_id=(
            f"58a5c_lattice{sample_index:02d}_stable_grasp_"
            f"port{ports[0]}_{ports[1]}_group{group_index}_"
            f"height{round(1000.0 * height_offset_m):02d}mm"
        ),
        structural_hash_prefix=STRUCTURAL_PREFIX,
        sample_index=sample_index,
        diagnosed_phase="approach",
        diagnosed_body_name="object_drop_repair",
        selected_anchor_body=False,
        selected_surface_port_ids=ports,
        candidate_group_id=group,
        pregrasp_clearance_m=0.15,
        collision_margin_m=0.030,
        grasp_contact_height_offset_m=height_offset_m,
        approach_time_scale=0.75,
    )
    value.validate()
    return value


def main() -> int:
    arguments = _parser().parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SchemaValidationError("R1 v18 deterministic seed contract differs")
    output_root = Path(arguments.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    # The v17 materializer validates that all generated diagnostic bindings
    # remain under its active output root.
    v17.OUTPUT_ROOT = output_root
    v16.runtime_v13._install_v13_preparation_hooks()
    pairs, references, targets = v16._pairs_and_cases()
    pair_matches = [pair for pair in pairs if pair.bucket_id == BUCKET_ID]
    if len(pair_matches) != 1:
        raise SchemaValidationError("R1 v18 held-out morphology is absent")
    pair = pair_matches[0]
    source_reference = references[BUCKET_ID]
    selected = SAMPLES if arguments.sample_index is None else (arguments.sample_index,)
    reference = None
    selected_ports = None
    selected_group = None
    selected_height_offset_m = None
    jobs = []
    for sample_index in selected:
        target_matches = [
            value
            for value in targets[BUCKET_ID]
            if value.sample_kind == "lattice" and value.sample_index == sample_index
        ]
        if len(target_matches) != 1:
            raise SchemaValidationError("R1 v18 target condition is absent")
        if reference is None:
            rejection_reasons = []
            for ports, group, height_offset_m in GRASP_OPTIONS:
                candidate = _heuristic(
                    sample_index,
                    ports=ports,
                    group=group,
                    height_offset_m=height_offset_m,
                )
                candidate_root = output_root / candidate.heuristic_id
                try:
                    reference = v17._prepare_reference(
                        pair=pair,
                        source_case=source_reference,
                        heuristic=candidate,
                        rule_root=candidate_root,
                    )
                except SchemaValidationError as error:
                    rejection_reasons.append(
                        {
                            "ports": list(ports),
                            "group": group,
                            "height_offset_m": height_offset_m,
                            "reason": str(error),
                        }
                    )
                    continue
                selected_ports = ports
                selected_group = group
                selected_height_offset_m = height_offset_m
                break
            if (
                reference is None
                or selected_ports is None
                or selected_group is None
                or selected_height_offset_m is None
            ):
                v17._write_json(
                    output_root / "lightweight_rejections.json",
                    {"version": VERSION, "rejections": rejection_reasons},
                )
                raise SchemaValidationError("R1 v18 has no admitted short-face grasp")
        assert selected_ports is not None and selected_group is not None
        heuristic = _heuristic(
            sample_index,
            ports=selected_ports,
            group=selected_group,
            height_offset_m=selected_height_offset_m,
        )
        rule_root = output_root / heuristic.heuristic_id
        target = v17._prepare_target(
            pair=pair,
            source_case=target_matches[0],
            heuristic=heuristic,
            reference=reference,
            rule_root=rule_root,
        )
        job = v17._job(
            root=target,
            heuristic=heuristic,
            isaac_python=arguments.isaac_python,
            rollout_steps=15000,
        )
        manifest = json.loads(job["job_path"].read_text(encoding="utf-8"))
        manifest.update(
            {
                "version": VERSION,
                "r1_diagnostic_lift_time_dilation": 3.0,
            }
        )
        v17._write_json(job["job_path"], manifest)
        jobs.append(job)
        print(
            "ORDER9_R1_OBJECT_DROP_GRASP_PREPARED="
            + json.dumps(
                {
                    "sample_index": sample_index,
                    "candidate_id": job["candidate_id"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    preparation = {
        "result_version": VERSION,
        "status": "accepted",
        "sample_indices": list(selected),
        "selected_surface_port_ids": list(selected_ports or ()),
        "candidate_group_id": selected_group,
        "grasp_contact_height_offset_m": selected_height_offset_m,
        "lightweight_screen_passed": True,
        "training_eligible": False,
        "formal_teacher_collection_authorized": False,
    }
    v17._write_json(output_root / "preparation_result.json", preparation)
    if arguments.prepare_only:
        print(json.dumps(preparation, sort_keys=True), flush=True)
        return 0
    results = []
    for job in jobs:
        if v17._run_job(job) != 0:
            raise RuntimeError("R1 v18 Isaac subprocess failed")
        result = v17._result(job)
        results.append(result)
        if result["status"] != "accepted":
            break
    payload = {
        **preparation,
        "status": (
            "accepted"
            if len(results) == len(jobs)
            and all(result["status"] == "accepted" for result in results)
            else "rejected"
        ),
        "success_count": sum(result["status"] == "accepted" for result in results),
        "results": results,
    }
    v17._write_json(output_root / "result.json", payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0 if payload["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
