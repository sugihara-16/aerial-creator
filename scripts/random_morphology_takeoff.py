from __future__ import annotations

import argparse
import json
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
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.training.random_morphology_takeoff_runner import (
    RandomMorphologyTakeoffRunner,
    RandomMorphologyTakeoffRunnerConfig,
    load_random_morphology_takeoff_runner_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic floor settle -> takeoff -> hover for a connected morphology."
    )
    parser.add_argument(
        "--config",
        default="configs/training/random_morphology_takeoff.yaml",
        help="Order-2 runner configuration.",
    )
    parser.add_argument(
        "--morphology-graph-json-path",
        default=None,
        help="Optional serialized MorphologyGraph. If omitted, sample the Order-1 distribution.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override Order-1 morphology seed.")
    parser.add_argument("--module-count", type=int, default=None, help="Optionally fix sampled module count to 2-8.")
    parser.add_argument("--max-attempts", type=int, default=None, help="Bounded deterministic feasibility-rejection attempts.")
    parser.add_argument("--real", action="store_true", help="Run the real Isaac probe instead of dry planning.")
    parser.add_argument("--report-path", default=None, help="Override JSON result path.")
    parser.add_argument("--archive-path", default=None, help="Override typed EpisodeArchive JSONL path.")
    args = parser.parse_args()

    runner_config, takeoff_config = load_random_morphology_takeoff_runner_config(args.config)
    seed = runner_config.seed if args.seed is None else args.seed
    module_count = runner_config.module_count if args.module_count is None else args.module_count
    runner_config = RandomMorphologyTakeoffRunnerConfig(
        seed=seed,
        module_count=module_count,
        dry_run=not args.real,
        source_hash=runner_config.source_hash,
        runner_version=runner_config.runner_version,
        report_path=runner_config.report_path,
        archive_path=runner_config.archive_path,
        max_sampling_attempts=(
            runner_config.max_sampling_attempts if args.max_attempts is None else args.max_attempts
        ),
    )
    physical_model = build_physical_model_from_config(takeoff_config.robot_model_config_path)
    feasibility_checker = MorphologyFlightFeasibilityChecker(
        MorphologyFlightFeasibilityConfig(mesh_search_dirs=tuple(takeoff_config.mesh_search_dirs))
    )
    if args.morphology_graph_json_path:
        morphology_graph = MorphologyGraph.from_json(
            Path(args.morphology_graph_json_path).read_text(encoding="utf-8")
        )
        sampling_metadata = {
            "source": "external_morphology_graph_json",
            "path": str(args.morphology_graph_json_path),
        }
    else:
        distribution = RandomFeasibleConnectedMorphologyDistribution(
            physical_model,
            feasibility_checker=feasibility_checker,
            config=RandomFeasibleMorphologyConfig(
                max_attempts_per_sample=runner_config.max_sampling_attempts,
            ),
        )
        try:
            sampled = distribution.sample_with_report(seed=seed, module_count=module_count)
        except SchemaValidationError as exc:
            print(
                json.dumps(
                    {
                        "error": "no_feasible_random_morphology_within_attempt_bound",
                        "seed": seed,
                        "attempt_bound": runner_config.max_sampling_attempts,
                        "detail": str(exc),
                    },
                    sort_keys=True,
                )
            )
            return 1
        morphology_graph = sampled.morphology_graph
        sampling_metadata = {
            "source": "random_feasible_connected_distribution",
            "requested_seed": sampled.requested_seed,
            "accepted_proposal_seed": sampled.accepted_proposal_seed,
            "attempt_count": sampled.attempt_count,
            "duplicate_rejection_count": sampled.duplicate_rejection_count,
            "rejected_violation_counts": sampled.rejected_violation_counts,
            "structural_hash": sampled.structural_hash,
        }

    result = RandomMorphologyTakeoffRunner(
        runner_config=runner_config,
        takeoff_config=takeoff_config,
        feasibility_checker=feasibility_checker,
    ).run(
        morphology_graph,
        report_path=Path(args.report_path) if args.report_path else None,
        archive_path=Path(args.archive_path) if args.archive_path else None,
        sampling_metadata=sampling_metadata,
    )
    print(
        json.dumps(
            {
                "runner_version": result.runner_version,
                "graph_id": result.morphology_graph.graph_id,
                "feasible": result.feasibility_result.feasible,
                "dry_run": result.takeoff_result.dry_run,
                "attempted": result.takeoff_result.attempted,
                "isaac_backed": result.takeoff_result.isaac_backed,
                "unit_contract_passed": result.takeoff_result.unit_contract_passed,
                "real_isaac_passed": result.takeoff_result.real_isaac_passed,
                "failure_reason": result.takeoff_result.failure_reason,
                "sampling_metadata": result.sampling_metadata,
                "metrics": result.takeoff_result.metrics,
                "report_path": args.report_path or runner_config.report_path,
                "archive_path": (
                    (args.archive_path or runner_config.archive_path)
                    if result.archive_episode_id is not None
                    else None
                ),
            },
            sort_keys=True,
        )
    )
    if (
        result.takeoff_result.dry_run
        and result.takeoff_result.unit_contract_passed
        and result.feasibility_result.feasible
    ) or result.takeoff_result.real_isaac_passed:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
