from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.robot_model.physical_model_builder import (  # noqa: E402
    build_physical_model_from_config,
)
from amsrr.schemas.morphology import MorphologyGraph  # noqa: E402
from amsrr.training.order9_r1_calibration_runner import (  # noqa: E402
    Order9R1PreparedCandidate,
    enumerate_order9_r1_calibration_level_cases,
)
from amsrr.training.order9_r1_clearance_diagnostic import (  # noqa: E402
    write_order9_r1_clearance_diagnostic,
)
from amsrr.training.order9_r1_fast_screen import (  # noqa: E402
    screen_order9_r1_nominal_candidate,
)
from amsrr.training.order9_r1_isaac_calibration import (  # noqa: E402
    materialize_order9_r1_isaac_case,
    run_order9_r1_isaac_cases,
)
from amsrr.training.order9_r1_support_clearance import (  # noqa: E402
    generate_order9_r1_v2_nominal_grasp_trajectory,
    require_order9_r1_v2_support_clearance,
)
from amsrr.utils.hashing import hash_file  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare one approved R1 calibration case against its frozen "
            "source support, then optionally run the full Isaac control chain"
        )
    )
    parser.add_argument(
        "--protocol",
        default="configs/training/order9_r1_calibration_protocol_v1.yaml",
    )
    parser.add_argument("--level-id", default="r1_l1_10mm_5deg")
    parser.add_argument(
        "--split", choices=("train", "validation"), default="train"
    )
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument(
        "--physical-model-config",
        default="configs/robot/robot_model.yaml",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-isaac", action="store_true")
    parser.add_argument(
        "--isaac-python",
        default="/home/leus/.local/share/mamba/envs/isaaclab3/bin/python",
    )
    parser.add_argument("--rollout-steps", type=int, default=15000)
    return parser


def main() -> int:
    args = _parser().parse_args()
    repository = REPOSITORY.resolve()
    cases = enumerate_order9_r1_calibration_level_cases(
        protocol_path=args.protocol,
        repository_root=repository,
        level_id=args.level_id,
        split=args.split,
    )
    if not 0 <= args.case_index < len(cases):
        raise IndexError("R1 support-clearance case index is out of range")
    case = cases[args.case_index]
    physical_path = (repository / args.physical_model_config).resolve()
    physical = build_physical_model_from_config(physical_path)
    bucket_manifest = (
        repository / "artifacts/p4_full/order9/releases/"
        "c3_pi_l_promoted_update18_v1/runtime/"
        "rollout_buckets_current_lineage_v5/manifest.json"
    )
    graph_path = (
        bucket_manifest.parent / case.source_bucket.morphology_graph_path
    ).resolve()
    graph = MorphologyGraph.from_json(graph_path.read_text(encoding="utf-8"))
    accepted_nominal = case.source_bucket.metadata.get(
        "accepted_nominal_trajectory", {}
    )
    preferred = accepted_nominal.get("selected_surface_port_ids")
    preferred_ports = (
        tuple(int(value) for value in preferred)
        if isinstance(preferred, list) and len(preferred) == 2
        else None
    )
    source_task_path = (
        bucket_manifest.parent / case.source_bucket.task_spec_path
    ).resolve()
    from amsrr.schemas.task_spec import TaskSpec

    source_task = TaskSpec.from_json(
        source_task_path.read_text(encoding="utf-8")
    )
    nominal, contract = generate_order9_r1_v2_nominal_grasp_trajectory(
        source_task_spec=source_task,
        randomized_task_spec=case.task_spec,
        structural_target=graph,
        physical_model=physical,
        preferred_surface_port_ids=preferred_ports,
    )
    screen = screen_order9_r1_nominal_candidate(
        candidate_id=case.candidate_id,
        task_spec=case.task_spec,
        physical_model=physical,
        nominal=nominal,
    )
    require_order9_r1_v2_support_clearance(screen=screen, contract=contract)
    prepared = Order9R1PreparedCandidate(
        candidate_id=case.candidate_id,
        candidate_task_hash=case.task_spec.stable_hash(),
        morphology_hash=case.source_bucket.morphology_hash,
        physical_model_hash=case.physical_model_hash,
        teacher_trajectory_complete=True,
        failure_reason=None,
        payload=nominal,
    )
    output = Path(args.output_dir).resolve()
    materialized = materialize_order9_r1_isaac_case(
        case=case,
        prepared=prepared,
        screen=screen,
        output_dir=output,
        repository_root=repository,
        approved_protocol_path=args.protocol,
    )
    contract_payload = {
        "result_version": "order9_r1_frozen_support_preparation_result_v2",
        "candidate_id": case.candidate_id,
        "status": "accepted_for_full_layer_test",
        "support_contract": contract.__dict__,
        "fast_screen": screen.to_dict(),
        "isaac_invoked": False,
        "controller_layers_invoked": False,
        "promotion_evidence_eligible": False,
        "training_eligible": False,
        "input_bindings": {
            "source_task": {
                "path": str(source_task_path),
                "sha256": hash_file(source_task_path),
            },
            "randomized_task": {
                "path": str(output / "task_spec.json"),
                "sha256": hash_file(output / "task_spec.json"),
            },
            "nominal_generator_implementation": {
                "path": str(
                    repository
                    / "amsrr/training/order9_c3_nominal_trajectory.py"
                ),
                "sha256": hash_file(
                    repository
                    / "amsrr/training/order9_c3_nominal_trajectory.py"
                ),
            },
            "support_clearance_implementation": {
                "path": str(
                    repository
                    / "amsrr/training/order9_r1_support_clearance.py"
                ),
                "sha256": hash_file(
                    repository
                    / "amsrr/training/order9_r1_support_clearance.py"
                ),
            },
            "preparation_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": hash_file(Path(__file__).resolve()),
            },
        },
    }
    write_order9_r1_clearance_diagnostic(
        contract_payload, output / "r1_v2_support_preparation_result.json"
    )
    if not args.run_isaac:
        print(output / "r1_v2_support_preparation_result.json")
        return 0

    results = run_order9_r1_isaac_cases(
        [materialized],
        repository_root=repository,
        python_executable=args.isaac_python,
        rollout_steps=args.rollout_steps,
        maximum_parallel_process_count=1,
        persistent_morphology_coalescing=False,
    )[case.candidate_id]
    success_count = sum(result.success for result in results)
    full_payload = {
        "result_version": "order9_r1_frozen_support_full_layer_result_v2",
        "candidate_id": case.candidate_id,
        "status": "accepted" if success_count == len(results) else "rejected",
        "support_contract": contract.__dict__,
        "fast_screen": screen.to_dict(),
        "full_layer": {
            "episode_count": len(results),
            "success_count": success_count,
            "safety_failure_count": sum(
                result.safety_failure for result in results
            ),
            "fallback_decision_count": sum(
                result.fallback_used for result in results
            ),
            "failure_reasons": [result.failure_reason for result in results],
            "execution_chain": results[0].execution_chain,
        },
        "isaac_invoked": True,
        "controller_layers_invoked": True,
        "promotion_evidence_eligible": False,
        "training_eligible": False,
        "teacher_collection_authorized": False,
        "bindings": {
            "case_manifest": {
                "path": str(materialized.manifest_path),
                "sha256": hash_file(materialized.manifest_path),
            },
            "raw_rollout": {
                "path": str(output / "isaac/evaluation_rollout.pt"),
                "sha256": hash_file(output / "isaac/evaluation_rollout.pt"),
            },
            "episodes": {
                "path": str(output / "isaac/evaluation_episodes.jsonl"),
                "sha256": hash_file(
                    output / "isaac/evaluation_episodes.jsonl"
                ),
            },
            "log": {
                "path": str(output / "isaac/evaluation.log"),
                "sha256": hash_file(output / "isaac/evaluation.log"),
            },
            "preparation_result": {
                "path": str(output / "r1_v2_support_preparation_result.json"),
                "sha256": hash_file(
                    output / "r1_v2_support_preparation_result.json"
                ),
            },
        },
    }
    destination = write_order9_r1_clearance_diagnostic(
        full_payload, output / "r1_v2_execution_result.json"
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
