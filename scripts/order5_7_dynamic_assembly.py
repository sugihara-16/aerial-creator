from __future__ import annotations

"""Run the P4-full Orders 5-7 dynamic module assembly round trip."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import secrets
import sys


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
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.simulation.dynamic_assembly import (
    DYNAMIC_ASSEMBLY_ATTACH_ONLY_GATE,
    DYNAMIC_ASSEMBLY_FILTER_FALLBACK_MODE,
    DYNAMIC_ASSEMBLY_MATING_MODES,
    DYNAMIC_ASSEMBLY_PHYSICAL_MATING_MODE,
    DYNAMIC_ASSEMBLY_ROUNDTRIP_GATE,
    DynamicAssemblyIsaacConfig,
    DynamicAssemblyIsaacEnv,
    dynamic_assembly_acceptance_contract,
)
from amsrr.simulation.isaac_lab_backend import IsaacLabBackend, load_isaac_lab_backend_config
from amsrr.utils.config import load_config
from amsrr.utils.hashing import hash_file


DEFAULT_CONFIG_PATH = "configs/training/order5_7_dynamic_assembly.yaml"
DEFAULT_REPORT_DIR = Path("artifacts/p4_full/order5_7_dynamic_assembly")


def default_report_path(
    acceptance_gate: str,
    mating_contact_mode: str = DYNAMIC_ASSEMBLY_FILTER_FALLBACK_MODE,
) -> Path:
    if mating_contact_mode == DYNAMIC_ASSEMBLY_PHYSICAL_MATING_MODE:
        return DEFAULT_REPORT_DIR / f"{acceptance_gate}_report.json"
    return DEFAULT_REPORT_DIR / f"{acceptance_gate}_{mating_contact_mode}_report.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Spawn two floor-initialized Holons and execute staging -> "
            "face-to-face alignment -> axial contact -> external FixedJoint "
            "attach -> unload-gated detach -> stable separation."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--backend-config", default="configs/env/isaac_lab.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=256)
    parser.add_argument("--morphology-graph-json-path", default=None)
    parser.add_argument("--real", action="store_true")
    parser.add_argument(
        "--acceptance-gate",
        choices=(DYNAMIC_ASSEMBLY_ATTACH_ONLY_GATE, DYNAMIC_ASSEMBLY_ROUNDTRIP_GATE),
        default=None,
        help="Run the Order-6 attach-only gate or the Order-7 full round trip.",
    )
    parser.add_argument(
        "--mating-contact-mode",
        choices=tuple(sorted(DYNAMIC_ASSEMBLY_MATING_MODES)),
        default=None,
        help=(
            "Select physical funnel contact or the explicitly separate "
            "selected-Dock-body-pair collision-filter fallback."
        ),
    )
    parser.add_argument("--viewer", choices=("kit",), default=None)
    parser.add_argument("--realtime-playback", action="store_true")
    parser.add_argument("--keep-open-after-rollout-s", type=float, default=0.0)
    parser.add_argument(
        "--report-path",
        default=None,
        help=(
            "Override the JSON result path. By default, attach_only and roundtrip "
            "write separate gate-specific reports."
        ),
    )
    parser.add_argument("--print-command", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.max_attempts <= 0:
        parser.error("--max-attempts must be positive")
    if args.viewer is not None and not args.real:
        parser.error("--viewer requires --real")
    if args.viewer is None and (
        args.realtime_playback or args.keep_open_after_rollout_s > 0.0
    ):
        parser.error("real-time/post-rollout viewing requires --viewer kit")

    config_source_path = Path(args.config).resolve()
    config_file_sha256 = hash_file(config_source_path)
    config_payload = load_config(config_source_path)
    config = DynamicAssemblyIsaacConfig.from_dict(config_payload["dynamic_assembly"])
    if args.acceptance_gate is not None:
        config.acceptance_gate = args.acceptance_gate
    if args.mating_contact_mode is not None:
        config.mating_contact_mode = args.mating_contact_mode
        config.control_bridge = replace(
            config.control_bridge,
            require_selected_pair_contact=(
                args.mating_contact_mode
                != DYNAMIC_ASSEMBLY_FILTER_FALLBACK_MODE
            ),
        )
    config.validate()
    backend_config_source_path = Path(args.backend_config).resolve()
    backend_config_file_sha256 = hash_file(backend_config_source_path)
    backend_config = load_isaac_lab_backend_config(backend_config_source_path)
    physical_model = build_physical_model_from_config(
        backend_config.robot_model_config_path
    )
    seed = secrets.randbits(63) if args.seed is None else int(args.seed)
    if args.morphology_graph_json_path:
        morphology = MorphologyGraph.from_json(
            Path(args.morphology_graph_json_path).read_text(encoding="utf-8")
        )
        sampling = {
            "source": "external_graph",
            "path": str(args.morphology_graph_json_path),
        }
    else:
        distribution = RandomFeasibleConnectedMorphologyDistribution(
            physical_model,
            feasibility_checker=MorphologyFlightFeasibilityChecker(
                MorphologyFlightFeasibilityConfig(
                    mesh_search_dirs=("module_urdf", "module_urdf/mesh")
                )
            ),
            config=RandomFeasibleMorphologyConfig(
                max_attempts_per_sample=int(args.max_attempts)
            ),
        )
        sample = distribution.sample_with_report(seed=seed, module_count=2)
        morphology = sample.morphology_graph
        sampling = {
            "source": "random_feasible_connected_distribution",
            "requested_seed": sample.requested_seed,
            "accepted_proposal_seed": sample.accepted_proposal_seed,
            "attempt_count": sample.attempt_count,
            "structural_hash": sample.structural_hash,
        }
    env = DynamicAssemblyIsaacEnv(
        config=config,
        backend=IsaacLabBackend(backend_config),
        backend_config_path=str(backend_config_source_path),
        viewer=args.viewer,
        realtime_playback=bool(args.realtime_playback),
        keep_open_after_rollout_s=float(args.keep_open_after_rollout_s),
    )
    result = env.run(morphology, dry_run=not args.real)
    run_provenance = {
        "seed": seed,
        "sampling": sampling,
        "acceptance_gate": config.acceptance_gate,
        "mating_contact_mode": config.mating_contact_mode,
        "acceptance_contract": dynamic_assembly_acceptance_contract(
            config.mating_contact_mode
        ),
        "config_path": str(config_source_path),
        "config_file_sha256": config_file_sha256,
        "config_hash": config.stable_hash(),
        "backend_config_path": str(backend_config_source_path),
        "backend_config_file_sha256": backend_config_file_sha256,
        "backend_config_hash": backend_config.stable_hash(),
        "physical_model_hash": physical_model.stable_hash(),
        "graph_id": morphology.graph_id,
        "graph_hash": morphology.stable_hash(),
        "real_requested": bool(args.real),
    }
    result.report["run_provenance"] = run_provenance
    output_path = (
        Path(args.report_path)
        if args.report_path is not None
        else default_report_path(
            config.acceptance_gate,
            config.mating_contact_mode,
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.to_dict(), sort_keys=True, indent=2),
        encoding="utf-8",
    )
    summary = {
        "version": result.version,
        "acceptance_gate": result.acceptance_gate,
        "mating_contact_mode": config.mating_contact_mode,
        "acceptance_contract": dynamic_assembly_acceptance_contract(
            config.mating_contact_mode
        ),
        "graph_id": result.graph_id,
        "graph_hash": result.graph_hash,
        "seed": seed,
        "sampling": sampling,
        "run_provenance": run_provenance,
        "dry_run": result.dry_run,
        "attempted": result.attempted,
        "isaac_backed": result.isaac_backed,
        "attach_passed": result.attach_passed,
        "detach_passed": result.detach_passed,
        "passed": result.passed,
        "report_validation_failures": result.report_validation_failures,
        "failure_reason": result.failure_reason,
        "report_path": str(output_path),
    }
    if args.print_command and result.dry_run:
        summary["probe_command"] = result.report["probe_command"]
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0 if result.dry_run or result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
