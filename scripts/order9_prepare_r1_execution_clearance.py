from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.robot_model.physical_model_builder import (  # noqa: E402
    build_physical_model_from_config,
)
from amsrr.schemas.common import SchemaValidationError  # noqa: E402
from amsrr.schemas.morphology import MorphologyGraph  # noqa: E402
from amsrr.schemas.task_spec import TaskSpec  # noqa: E402
from amsrr.training.order9_r1_clearance_diagnostic import (  # noqa: E402
    write_order9_r1_clearance_diagnostic,
)
from amsrr.training.order9_r1_execution_clearance import (  # noqa: E402
    build_order9_r1_execution_clearance_contract,
    generate_order9_r1_v3_nominal_grasp_trajectory,
    require_order9_r1_v3_execution_clearance,
)
from amsrr.training.order9_r1_fast_screen import (  # noqa: E402
    screen_order9_r1_nominal_candidate,
)
from amsrr.utils.hashing import hash_file  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare one R1 candidate with measured execution clearance"
    )
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--source-task", required=True)
    parser.add_argument("--randomized-task", required=True)
    parser.add_argument("--morphology-graph", required=True)
    parser.add_argument("--physical-model-config", required=True)
    parser.add_argument("--preferred-surface-port-ids", nargs=2, type=int)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    source_path = Path(args.source_task).resolve()
    randomized_path = Path(args.randomized_task).resolve()
    graph_path = Path(args.morphology_graph).resolve()
    physical_path = Path(args.physical_model_config).resolve()
    source = TaskSpec.from_json(source_path.read_text(encoding="utf-8"))
    randomized = TaskSpec.from_json(
        randomized_path.read_text(encoding="utf-8")
    )
    graph = MorphologyGraph.from_json(graph_path.read_text(encoding="utf-8"))
    physical = build_physical_model_from_config(physical_path)
    _collision, contract = build_order9_r1_execution_clearance_contract(
        source_task_spec=source,
        randomized_task_spec=randomized,
    )
    started = perf_counter()
    status = "rejected"
    failure_reason = None
    screen_payload = None
    try:
        nominal, returned_contract = (
            generate_order9_r1_v3_nominal_grasp_trajectory(
                source_task_spec=source,
                randomized_task_spec=randomized,
                structural_target=graph,
                physical_model=physical,
                preferred_surface_port_ids=(
                    None
                    if args.preferred_surface_port_ids is None
                    else tuple(args.preferred_surface_port_ids)
                ),
            )
        )
        if returned_contract != contract:
            raise SchemaValidationError(
                "R1 execution-clearance contract drifted"
            )
        screen = screen_order9_r1_nominal_candidate(
            candidate_id=args.candidate_id,
            task_spec=randomized,
            physical_model=physical,
            nominal=nominal,
        )
        require_order9_r1_v3_execution_clearance(
            screen=screen,
            contract=contract,
        )
        status = "accepted_for_full_layer_test"
        screen_payload = screen.to_dict()
    except (SchemaValidationError, ValueError) as error:
        failure_reason = str(error)
    payload = {
        "result_version": "order9_r1_execution_clearance_preparation_result_v1",
        "candidate_id": args.candidate_id,
        "status": status,
        "failure_reason": failure_reason,
        "support_contract": contract.__dict__,
        "screen": screen_payload,
        "isaac_invoked": False,
        "controller_layers_invoked": False,
        "promotion_evidence_eligible": False,
        "training_eligible": False,
        "wall_time_s": perf_counter() - started,
        "input_bindings": {
            "source_task": {
                "path": str(source_path),
                "sha256": hash_file(source_path),
            },
            "randomized_task": {
                "path": str(randomized_path),
                "sha256": hash_file(randomized_path),
            },
            "morphology_graph": {
                "path": str(graph_path),
                "sha256": hash_file(graph_path),
            },
            "physical_model_config": {
                "path": str(physical_path),
                "sha256": hash_file(physical_path),
            },
            "implementation": {
                "path": str(
                    REPOSITORY
                    / "amsrr/training/order9_r1_execution_clearance.py"
                ),
                "sha256": hash_file(
                    REPOSITORY
                    / "amsrr/training/order9_r1_execution_clearance.py"
                ),
            },
        },
    }
    destination = write_order9_r1_clearance_diagnostic(payload, args.output)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
