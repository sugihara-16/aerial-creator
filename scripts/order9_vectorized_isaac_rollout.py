#!/usr/bin/env python3
from __future__ import annotations

"""Collect one topology/object bucket of real-Isaac Order 9 ``pi_L`` PPO.

The hot path is tensor-only: copied Isaac environments, phase-conditioned
``pi_L``, batched QPID/QP, privileged contact reduction, phase-aware reward,
and a compact raw tensor artifact.  JSONL schema reconstruction is deliberately
left to the post-simulation dataset builder.
"""

import argparse
import contextlib
import gc
import json
import math
from pathlib import Path
import shlex
import sys
import traceback
from types import SimpleNamespace

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from isaaclab.app import AppLauncher
from amsrr.training.order9_c3_action_contract import (
    ORDER9_C3_ACTION_CONTRACT_CONTACT_SPACE_PROJECTED,
    ORDER9_C3_ACTION_CONTRACTS,
    order9_c3_action_contract_uses_full_policy,
    order9_c3_action_contract_uses_contact_space,
)
from amsrr.training.order9_c3_execution_bundle import (
    apply_order9_c3_execution_bundle_to_args,
    load_order9_c3_execution_bundle,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/training/order9_learning_curriculum.yaml"
    )
    parser.add_argument("--stage", default="c2_pi_l_ppo_fixed_conservative")
    parser.add_argument("--pi-l-checkpoint")
    parser.add_argument("--pi-l-checkpoint-sha256")
    parser.add_argument(
        "--c3-execution-bundle",
        help=(
            "Immutable checkpoint/nominal/QPID/reset-bank bundle. When set, "
            "the C3 runtime inputs are loaded from the bundle and conflicting "
            "individual CLI overrides are rejected before Isaac starts."
        ),
    )
    parser.add_argument(
        "--c3-reset-bank",
        help=(
            "Bucket-local accepted-nominal physical phase-reset artifact. It "
            "is generated once and then reused byte-for-byte."
        ),
    )
    parser.add_argument(
        "--c3-wrench-gate-range-scale",
        type=float,
        default=1.0,
        help=(
            "Training-only contact-phase admission scale. The privileged "
            "reward continues to use the exact teacher wrench box."
        ),
    )
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--output-raw", required=True)
    parser.add_argument(
        "--evaluation-jsonl",
        help=(
            "Write deterministic, phase-zero, first-terminal episode evidence "
            "for BC-stage promotion."
        ),
    )
    parser.add_argument("--evaluation-episode-count", type=int, default=100)
    parser.add_argument(
        "--num-envs",
        type=int,
        help="Diagnostic override; production defaults to the selected stage runtime.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        help="Diagnostic override; production defaults to the selected stage runtime.",
    )
    parser.add_argument("--production-topology-shard-index", type=int)
    parser.add_argument("--production-topology-shard-count", type=int)
    parser.add_argument("--production-topology-shard-environment-count", type=int)
    parser.add_argument(
        "--production-state-inheritance-shard",
        action="store_true",
        help=(
            "Collect the long C3 train shard that preserves physical and "
            "recurrent state across real phase transitions."
        ),
    )
    parser.add_argument("--seed", type=int, default=9009)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--env-spacing", type=float, default=3.0)
    parser.add_argument(
        "--robot-usd",
        help="Explicit topology-bucket USD; fixed C1/C2 must match its manifest.",
    )
    parser.add_argument(
        "--fixed-nominal-asset-manifest",
        default="artifacts/p4_full/order9/fixed_nominal_asset/manifest.json",
        help="Hash-bound fixed-morphology USD used by C1/C2.",
    )
    parser.add_argument("--morphology-graph-json")
    parser.add_argument("--task-spec-json")
    parser.add_argument("--c3-nominal-set-manifest")
    parser.add_argument("--c3-nominal-set-sha256")
    parser.add_argument("--c3-nominal-artifact-sha256")
    parser.add_argument(
        "--teacher-dataset-manifest",
        default="artifacts/p4_full/order9/c0_teacher/dataset/manifest.json",
        help=(
            "Checkpoint-bound C0 source for the fixed-nominal C1 active-knot "
            "reference."
        ),
    )
    parser.add_argument("--selected-gripper-friction", type=float)
    parser.add_argument("--contact-stiffness", type=float, default=7500.0)
    parser.add_argument("--contact-damping", type=float, default=75.0)
    parser.add_argument("--collision-contact-offset-m", type=float)
    parser.add_argument("--collision-rest-offset-m", type=float)
    parser.add_argument("--estimated-mass-kg", type=float)
    parser.add_argument("--estimated-inertia-body", nargs=6, type=float)
    parser.add_argument("--estimated-com-object", nargs=3, type=float)
    parser.add_argument(
        "--tensorboard-log-dir",
        help="Override the shared stage TensorBoard directory.",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable live TensorBoard telemetry for diagnostic runs.",
    )
    parser.add_argument(
        "--persistent-bucket-jobs",
        help=(
            "Internal orchestration manifest for sequential same-morphology "
            "bucket collection under one persistent Isaac application. Each "
            "job retains an independent scene, artifact, log, and acceptance "
            "identity; only the Kit process is reused."
        ),
    )
    parser.add_argument(
        "--training-only-continuous-teacher-rollout",
        action="store_true",
        help=(
            "Permit deterministic phase-zero collection on a train bucket "
            "for privileged warm-start targets. The output is never "
            "promotion evidence."
        ),
    )
    parser.add_argument(
        "--canonical-phase-resets",
        choices=("auto", "yes", "no"),
        default="auto",
        help="Use hash-bound Order 8 physical phase starts for its fixed topology.",
    )
    parser.add_argument(
        "--formal-phase-zero-start",
        action="store_true",
        help=(
            "Require the acceptance-eligible end-to-end evaluation start: "
            "the exact t=0 state of phase 0. Unlike diagnostic phase-reset "
            "overrides, this is the formal C3 promotion initial-state contract."
        ),
    )
    parser.add_argument(
        "--diagnostic-initial-phase-index",
        type=int,
        choices=range(8),
        help=(
            "Acceptance-ineligible diagnostic override that initializes every "
            "environment from one canonical phase."
        ),
    )
    parser.add_argument(
        "--diagnostic-initial-phase-progress",
        type=float,
        help=(
            "Acceptance-ineligible override of the initial phase clock. "
            "Used by short, end-of-contact diagnostics."
        ),
    )
    parser.add_argument(
        "--diagnostic-initial-phase-stratum-index",
        type=int,
        choices=range(4),
        help=(
            "Acceptance-ineligible C3 reset-bank stratum override. This lets "
            "nominal-QPID screening start from the boundary-near state of an "
            "explicit diagnostic phase without altering that state's clock."
        ),
    )
    parser.add_argument(
        "--diagnostic-initial-phase-strata",
        help=(
            "Acceptance-ineligible comma-separated PHASE:STRATUM schedule "
            "with one pair per environment. This screens multiple late "
            "phases in one nominal-QPID Isaac launch."
        ),
    )
    parser.add_argument(
        "--diagnostic-learned-phase-strata",
        action="store_true",
        help=(
            "Acceptance-ineligible A/B diagnostic that applies the learned "
            "pi_L to --diagnostic-initial-phase-strata. This may only write "
            "deterministic validation evidence and is never promotion evidence."
        ),
    )
    parser.add_argument(
        "--diagnostic-deterministic-policy",
        action="store_true",
        help=(
            "Use the policy mean in an acceptance-ineligible diagnostic. "
            "With --diagnostic-initial-phase-index it isolates one phase; "
            "without it, the normal all-phase reset distribution is retained."
        ),
    )
    parser.add_argument(
        "--diagnostic-collision-evidence",
        action="store_true",
        help=(
            "Print the first prohibited object/environment contact body and "
            "force per environment. This is acceptance-ineligible telemetry."
        ),
    )
    parser.add_argument(
        "--diagnostic-phase-gate-evidence",
        action="store_true",
        help=(
            "Print final deployable phase-gate geometry and tracking evidence. "
            "This telemetry is acceptance-ineligible."
        ),
    )
    parser.add_argument(
        "--diagnostic-force-observer-trace",
        help=(
            "Write an acceptance-ineligible compact trace containing only "
            "the signals needed for offline force-observer A/B replay."
        ),
    )
    parser.add_argument(
        "--virtual-contact-lead-mm",
        type=float,
        help=(
            "Diagnostic/production override for the execution-side inward "
            "IK servo lead. pi_H contact poses remain on the object surface."
        ),
    )
    parser.add_argument(
        "--diagnostic-virtual-contact-additional-lead-sweep-mm",
        help=(
            "Acceptance-ineligible comma-separated additional inward IK leads. "
            "The values are added to the configured production lead, assigned "
            "to equal-size environment groups, and evaluated only through the "
            "end of lift."
        ),
    )
    parser.add_argument(
        "--training-nominal-preload-deficit-mm",
        help=(
            "Comma-separated train-only deficits subtracted from actuator-aware "
            "nominal preload and stratified across parallel environments."
        ),
    )
    parser.add_argument(
        "--training-nominal-preload-deficit-module-count",
        type=int,
        help="Apply the train-only nominal-preload deficits to this module count.",
    )
    parser.add_argument(
        "--diagnostic-contact-normal-residual-sweep-mm",
        help=(
            "Acceptance-ineligible comma-separated constant signed-normal "
            "pi_L contact residuals. Positive values point inward and negative "
            "values point outward. Values are assigned to equal-size environment "
            "groups while every other deterministic actor output is retained. "
            "The sweep may be a formal phase-zero diagnosis or a short "
            "deterministic diagnostic phase-reset probe."
        ),
    )
    parser.add_argument(
        "--contact-compression-action-span-mm",
        type=float,
        help=(
            "Diagnostic override for the pi_L morphology-invariant inward "
            "contact-compression action span."
        ),
    )
    parser.add_argument(
        "--contact-compression-maximum-joint-delta-mrad",
        type=float,
        help=(
            "Override the deployable local joint trust region applied to the "
            "contact-compression action."
        ),
    )
    parser.add_argument(
        "--diagnostic-nominal-qpid-only",
        action="store_true",
        help="Bypass pi_L commands while retaining nominal IK plus QPID/QP.",
    )
    parser.add_argument(
        "--diagnostic-action-ablation",
        choices=(
            "zero_global",
            "zero_joint",
            "contact_compression_only",
            "contact_compression_plus_centroidal",
            "contact_compression_plus_global",
        ),
        help=(
            "Acceptance-ineligible deterministic A/B test that zeros the "
            "selected pi_L action group while retaining actor recurrence."
        ),
    )
    parser.add_argument(
        "--c3-action-contract",
        choices=ORDER9_C3_ACTION_CONTRACTS,
        help=(
            "Explicit train/evaluation action contract for a matched C3 PPO "
            "branch. Unlike a diagnostic ablation, the sampled full action is "
            "retained for exact on-policy replay while only the contracted "
            "command coordinates are applied. Nominal-QPID diagnostics may "
            "retain the contract solely to load its contract-specific decoder; "
            "the policy action remains bypassed."
        ),
    )
    parser.add_argument(
        "--diagnostic-contact-sweep-json",
        help=(
            "Write acceptance-ineligible PhysX contact-force/load/kinematic-"
            "penetration telemetry for one virtual-contact lead."
        ),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _parse_diagnostic_lead_sweep_mm(raw: str | None) -> tuple[float, ...] | None:
    if raw is None:
        return None
    try:
        values = tuple(float(value.strip()) for value in raw.split(","))
    except ValueError as error:
        raise ValueError(
            "diagnostic virtual-contact lead sweep must be comma-separated numbers"
        ) from error
    if len(values) < 2 or any(
        not math.isfinite(value) or not 0.0 <= value <= 20.0 for value in values
    ):
        raise ValueError(
            "diagnostic virtual-contact lead sweep requires at least two values "
            "in [0, 20] mm"
        )
    if len(set(values)) != len(values):
        raise ValueError("diagnostic virtual-contact lead sweep values must be unique")
    return values


def _parse_diagnostic_contact_normal_residual_sweep_mm(
    raw: str | None,
) -> tuple[float, ...] | None:
    if raw is None:
        return None
    try:
        values = tuple(float(value.strip()) for value in raw.split(","))
    except ValueError as error:
        raise ValueError(
            "diagnostic contact-normal residual sweep must be comma-separated numbers"
        ) from error
    if len(values) < 2 or any(
        not math.isfinite(value) or not -20.0 <= value <= 20.0
        for value in values
    ):
        raise ValueError(
            "diagnostic contact-normal residual sweep requires at least two "
            "values in [-20, 20] mm"
        )
    if len(set(values)) != len(values):
        raise ValueError(
            "diagnostic contact-normal residual sweep values must be unique"
        )
    return values


def _bind_c3_execution_bundle(args):
    if args.c3_execution_bundle is None:
        if args.pi_l_checkpoint is None or args.pi_l_checkpoint_sha256 is None:
            raise ValueError(
                "rollout requires --pi-l-checkpoint and its sha256, or one "
                "--c3-execution-bundle"
            )
        return None
    args.stage = "c3_pi_l_ppo_arbitrary_morphology"
    bundle = load_order9_c3_execution_bundle(
        args.c3_execution_bundle,
        repository_root=REPOSITORY_ROOT,
    )
    apply_order9_c3_execution_bundle_to_args(args, bundle)
    return bundle


args_cli = _parser().parse_args()
c3_execution_bundle_runtime = _bind_c3_execution_bundle(args_cli)
from amsrr.training.order9_training_nominal_preload_deficit import (
    ORDER9_TRAINING_NOMINAL_PRELOAD_DEFICIT_VERSION,
    parse_order9_training_nominal_preload_deficit_mm,
    plan_order9_training_nominal_preload_deficit,
)

training_nominal_preload_deficit_mm = (
    parse_order9_training_nominal_preload_deficit_mm(
        args_cli.training_nominal_preload_deficit_mm
    )
)
diagnostic_virtual_contact_additional_lead_sweep_mm = (
    _parse_diagnostic_lead_sweep_mm(
        args_cli.diagnostic_virtual_contact_additional_lead_sweep_mm
    )
)
diagnostic_contact_normal_residual_sweep_mm = (
    _parse_diagnostic_contact_normal_residual_sweep_mm(
        args_cli.diagnostic_contact_normal_residual_sweep_mm
    )
)
if (args_cli.num_envs is not None and args_cli.num_envs < 1) or (
    args_cli.rollout_steps is not None and args_cli.rollout_steps < 1
):
    raise ValueError("Order9 rollout environment/step counts must be positive")
if args_cli.seed < 0 or args_cli.dt <= 0.0 or args_cli.env_spacing <= 0.0:
    raise ValueError("Order9 rollout seed/dt/spacing is invalid")
if args_cli.evaluation_episode_count < 1:
    raise ValueError("Order9 evaluation episode count must be positive")
if (
    not math.isfinite(args_cli.c3_wrench_gate_range_scale)
    or args_cli.c3_wrench_gate_range_scale < 1.0
):
    raise ValueError("C3 wrench-gate range scale must be finite and at least one")
if (
    args_cli.evaluation_jsonl is not None
    and args_cli.split != "validation"
    and not args_cli.training_only_continuous_teacher_rollout
    and not args_cli.diagnostic_nominal_qpid_only
    and args_cli.diagnostic_action_ablation is None
):
    raise ValueError("Order9 BC promotion evidence must use the validation split")
if args_cli.training_only_continuous_teacher_rollout and (
    args_cli.evaluation_jsonl is None
    or args_cli.split != "train"
    or args_cli.diagnostic_nominal_qpid_only
    or args_cli.diagnostic_action_ablation is not None
):
    raise ValueError(
        "Order9 continuous teacher rollout requires deterministic train "
        "collection with an enabled complete pi_L"
    )
if args_cli.formal_phase_zero_start and (
    args_cli.evaluation_jsonl is None
    or args_cli.split != "validation"
    or args_cli.training_only_continuous_teacher_rollout
    or args_cli.diagnostic_initial_phase_index is not None
    or args_cli.diagnostic_initial_phase_progress is not None
    or args_cli.diagnostic_initial_phase_stratum_index is not None
    or args_cli.diagnostic_initial_phase_strata is not None
):
    raise ValueError(
        "formal phase-zero start requires an unmodified validation evaluation"
    )
if args_cli.diagnostic_initial_phase_progress is not None and (
    args_cli.diagnostic_initial_phase_index is None
    or not math.isfinite(args_cli.diagnostic_initial_phase_progress)
    or not 0.0 <= args_cli.diagnostic_initial_phase_progress <= 1.0
):
    raise ValueError(
        "diagnostic initial phase progress requires a phase and must be in [0, 1]"
    )
if (
    args_cli.diagnostic_initial_phase_stratum_index is not None
    and args_cli.diagnostic_initial_phase_index is None
):
    raise ValueError("diagnostic initial phase stratum requires a phase")
if args_cli.diagnostic_initial_phase_strata is not None and any(
    value is not None
    for value in (
        args_cli.diagnostic_initial_phase_index,
        args_cli.diagnostic_initial_phase_progress,
        args_cli.diagnostic_initial_phase_stratum_index,
    )
):
    raise ValueError(
        "diagnostic phase strata are mutually exclusive with scalar phase overrides"
    )
if args_cli.diagnostic_learned_phase_strata and (
    args_cli.diagnostic_initial_phase_strata is None
    or args_cli.diagnostic_nominal_qpid_only
    or args_cli.split != "validation"
    or (
        args_cli.evaluation_jsonl is None
        and (
            diagnostic_contact_normal_residual_sweep_mm is None
            or not args_cli.diagnostic_deterministic_policy
        )
    )
):
    raise ValueError(
        "learned phase-strata diagnostic requires deterministic validation "
        "evidence or a short deterministic contact-normal probe, explicit "
        "strata, and an enabled pi_L"
    )
if args_cli.diagnostic_initial_phase_strata is not None and not (
    args_cli.diagnostic_nominal_qpid_only or args_cli.diagnostic_learned_phase_strata
):
    raise ValueError(
        "diagnostic phase strata require nominal-QPID-only or the explicit "
        "learned-policy A/B diagnostic"
    )
_topology_shard_values = (
    args_cli.production_topology_shard_index,
    args_cli.production_topology_shard_count,
    args_cli.production_topology_shard_environment_count,
)
if any(value is not None for value in _topology_shard_values):
    if any(value is None for value in _topology_shard_values):
        raise ValueError(
            "production topology shard arguments must be provided together"
        )
    if (
        args_cli.production_topology_shard_count < 2
        or not 0
        <= args_cli.production_topology_shard_index
        < args_cli.production_topology_shard_count
        or args_cli.production_topology_shard_environment_count < 1
    ):
        raise ValueError("production topology shard arguments are invalid")
if (
    args_cli.evaluation_jsonl is not None
    and args_cli.diagnostic_initial_phase_index is not None
    and not args_cli.diagnostic_nominal_qpid_only
):
    raise ValueError("Order9 promotion evaluation must begin at phase zero")
for name in ("contact_stiffness", "contact_damping"):
    if (
        not math.isfinite(float(getattr(args_cli, name)))
        or float(getattr(args_cli, name)) <= 0.0
    ):
        raise ValueError(f"--{name.replace('_', '-')} must be positive")
if (args_cli.collision_contact_offset_m is None) != (
    args_cli.collision_rest_offset_m is None
):
    raise ValueError("collision contact/rest offsets must be provided together")
if args_cli.collision_contact_offset_m is not None and (
    not math.isfinite(args_cli.collision_contact_offset_m)
    or not math.isfinite(args_cli.collision_rest_offset_m)
    or args_cli.collision_rest_offset_m < 0.0
    or args_cli.collision_contact_offset_m <= args_cli.collision_rest_offset_m
):
    raise ValueError("collision offsets must satisfy contact > rest >= 0")
if args_cli.virtual_contact_lead_mm is not None and (
    not math.isfinite(args_cli.virtual_contact_lead_mm)
    or not 0.0 <= args_cli.virtual_contact_lead_mm <= 20.0
):
    raise ValueError("--virtual-contact-lead-mm must be in [0, 20]")
if diagnostic_virtual_contact_additional_lead_sweep_mm is not None and (
    args_cli.evaluation_jsonl is None
    or args_cli.split != "validation"
    or not args_cli.formal_phase_zero_start
    or args_cli.stage != "c3_pi_l_ppo_arbitrary_morphology"
    or args_cli.c3_action_contract != ORDER9_C3_ACTION_CONTRACT_CONTACT_SPACE_PROJECTED
    or args_cli.virtual_contact_lead_mm is not None
    or args_cli.diagnostic_nominal_qpid_only
    or args_cli.diagnostic_action_ablation is not None
    or args_cli.diagnostic_contact_sweep_json is not None
):
    raise ValueError(
        "diagnostic additional-lead sweep requires formal phase-zero C3 "
        "contact-space evaluation and cannot be combined with another action, "
        "lead, or contact-sweep override"
    )
if (training_nominal_preload_deficit_mm is None) != (
    args_cli.training_nominal_preload_deficit_module_count is None
):
    raise ValueError(
        "training nominal-preload deficit values and module count are required together"
    )
if training_nominal_preload_deficit_mm is not None and (
    args_cli.split != "train"
    or args_cli.evaluation_jsonl is not None
    or args_cli.stage != "c3_pi_l_ppo_arbitrary_morphology"
    or args_cli.c3_action_contract
    != ORDER9_C3_ACTION_CONTRACT_CONTACT_SPACE_PROJECTED
    or args_cli.virtual_contact_lead_mm is not None
    or diagnostic_virtual_contact_additional_lead_sweep_mm is not None
    or args_cli.diagnostic_nominal_qpid_only
    or args_cli.diagnostic_action_ablation is not None
):
    raise ValueError(
        "nominal-preload deficit randomization requires an unmodified C3 "
        "contact-space train rollout"
    )
if diagnostic_contact_normal_residual_sweep_mm is not None:
    formal_sweep = bool(
        args_cli.evaluation_jsonl is not None and args_cli.formal_phase_zero_start
    )
    short_phase_probe = bool(
        args_cli.evaluation_jsonl is None
        and not args_cli.formal_phase_zero_start
        and (
            args_cli.diagnostic_initial_phase_index is not None
            or args_cli.diagnostic_initial_phase_strata is not None
        )
        and args_cli.diagnostic_deterministic_policy
    )
    if (
        not (formal_sweep or short_phase_probe)
        or args_cli.split != "validation"
        or args_cli.stage != "c3_pi_l_ppo_arbitrary_morphology"
        or args_cli.c3_action_contract
        != ORDER9_C3_ACTION_CONTRACT_CONTACT_SPACE_PROJECTED
        or args_cli.virtual_contact_lead_mm is not None
        or diagnostic_virtual_contact_additional_lead_sweep_mm is not None
        or args_cli.diagnostic_nominal_qpid_only
        or args_cli.diagnostic_action_ablation is not None
        or args_cli.diagnostic_contact_sweep_json is not None
    ):
        raise ValueError(
            "diagnostic contact-normal residual sweep requires either formal "
            "phase-zero C3 evaluation or a deterministic C3 diagnostic "
            "phase-reset probe, and cannot be combined with another action, "
            "lead, or contact-sweep override"
        )
if args_cli.contact_compression_action_span_mm is not None and (
    not math.isfinite(args_cli.contact_compression_action_span_mm)
    or not 0.0 <= args_cli.contact_compression_action_span_mm <= 20.0
):
    raise ValueError("--contact-compression-action-span-mm must be in [0, 20]")
if args_cli.contact_compression_maximum_joint_delta_mrad is not None and (
    not math.isfinite(args_cli.contact_compression_maximum_joint_delta_mrad)
    or not 0.0 < args_cli.contact_compression_maximum_joint_delta_mrad <= 150.0
):
    raise ValueError(
        "--contact-compression-maximum-joint-delta-mrad must be in (0, 150]"
    )
if args_cli.diagnostic_contact_sweep_json is not None and (
    not args_cli.diagnostic_nominal_qpid_only
    or args_cli.virtual_contact_lead_mm is None
    or args_cli.diagnostic_initial_phase_index != 1
):
    raise ValueError(
        "contact sweep requires nominal-QPID-only, an explicit lead, and "
        "--diagnostic-initial-phase-index 1"
    )
if args_cli.diagnostic_action_ablation is not None and (
    args_cli.evaluation_jsonl is None or args_cli.diagnostic_nominal_qpid_only
):
    raise ValueError(
        "diagnostic action ablation requires learned deterministic evaluation"
    )
if args_cli.c3_action_contract is not None and (
    args_cli.stage != "c3_pi_l_ppo_arbitrary_morphology"
    or args_cli.diagnostic_action_ablation is not None
):
    raise ValueError(
        "explicit C3 action contract requires C3 pi_L and cannot be combined "
        "with a diagnostic action ablation"
    )

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time
from dataclasses import fields, replace

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass
from pxr import PhysxSchema, Sdf, Usd, UsdPhysics, UsdShade

from amsrr.controllers.batched_qpid_controller import BatchedQPIDController
from amsrr.controllers.qpid_controller import QPIDControllerConfig
from amsrr.feasibility.articulated_reachability import (
    resolve_mesh_backed_anchor_references,
)
from amsrr.geometry.pose_math import compose_pose, inverse_pose
from amsrr.geometry.contact_material import resolve_contact_friction
from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.policies.contact_candidate_sampler import ContactCandidateSampler
from amsrr.policies.contact_wrench_trajectory import GraspCarryBaselinePlanner
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.policies.order9_active_knot_features import (
    ORDER9_ACTIVE_ASSIGNMENT_FEATURE_NAMES,
    ORDER9_ACTIVE_KNOT_FEATURE_CONTRACT_VERSION,
    ORDER9_ACTIVE_KNOT_GLOBAL_FEATURE_NAMES,
)
from amsrr.policies.order9_low_level_policy import (
    ORDER9_CONTACT_SPACE_ACTION_NAMES,
    ORDER9_CONTACT_FEEDBACK_SPACE_FEATURE_NAMES,
    ORDER9_CONTACT_SPACE_FEATURE_NAMES,
    ORDER9_CONTACT_FEEDBACK_PI_L_POLICY_VERSION,
    ORDER9_CATEGORICAL_CONTACT_NORMAL_PI_L_POLICY_VERSION,
    ORDER9_CONTACT_SPACE_PI_L_POLICY_VERSION,
    ORDER9_CONTACT_SPACE_PI_L_POLICY_VERSIONS,
    ORDER9_GLOBAL_ACTION_SIZE,
    ORDER9_MAX_CONTACT_SLOTS,
    ORDER9_MORPHOLOGY_INVARIANT_COMPRESSION_PI_L_POLICY_VERSION,
)
from amsrr.policies.order9_tensor_command_decoder import (
    ORDER9_CONTACT_COMPRESSION_ACTION_ADAPTER_VERSION,
    ORDER9_INDEPENDENT_COMPRESSION_ACTION_ADAPTER_VERSION,
    ORDER9_MORPHOLOGY_INVARIANT_COMPRESSION_ACTION_ADAPTER_VERSION,
)
from amsrr.training.order9_contact_space_action import (
    ORDER9_CONTACT_NORMAL_ACTION_QUANTIZATION_VERSION,
    ORDER9_CONTACT_SPACE_ACTION_ADAPTER_VERSION,
    Order9ContactSpaceActionConfig,
    build_order9_deployable_contact_feedback_features,
    build_order9_contact_space_action_basis,
)
from amsrr.robot_model.fixed_morphology_urdf import (
    articulated_morphology_graph_connections,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.robot_model.whole_structure_kinematics import (
    WholeStructureKinematics,
    ordered_global_dock_joint_ids,
)
from amsrr.schemas.common import ContactMode
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.order9 import Order9PolicyFamily
from amsrr.schemas.task_spec import GeometryType, TaskSpec
from amsrr.simulation.order8_natural_contact import (
    build_representative_order8_morphology,
)
from amsrr.simulation.order9_fixed_nominal_asset import (
    load_order9_fixed_nominal_asset_manifest,
    validate_order9_fixed_nominal_asset_manifest_bytes,
)
from amsrr.simulation.order9_actuator_runtime import (
    Order9ActuatorRuntimeValues,
    order9_actuator_runtime_values,
    validate_order9_actuator_readback,
)
from amsrr.simulation.order9_object_task_runtime import (
    ORDER9_OBJECT_TASK_ACTOR_PHASE_INDEX_BY_RUNTIME,
    ORDER9_OBJECT_TASK_ACTOR_PHASE_LABELS,
    ORDER9_OBJECT_TASK_PHASES,
    Order9ObjectTaskPhase,
    Order9ObjectTaskRuntime,
    Order9ObjectTaskRuntimeConfig,
    order9_rollout_initial_phase_indices,
    order9_rollout_initial_stratum_indices,
)
from amsrr.simulation.order9_object_task_state import load_order9_canonical_reset
from amsrr.simulation.order9_tensor_isaac_io import Order9TensorIsaacIO
from amsrr.simulation.order9_tensor_object_task import (
    ORDER9_CONTACT_SCHEDULE_ATTACH,
    ORDER9_CONTACT_SCHEDULE_MAINTAIN,
    ORDER9_CONTACT_SCHEDULE_RELEASE,
    ORDER9_PHASE_SUCCESSOR_REFERENCE_SEMANTICS,
    ORDER9_PAYLOAD_FEEDFORWARD_PHASE_CONTRACT,
    Order9TensorObjectTaskRuntime,
    order9_payload_feedforward_scale,
)
from amsrr.training.order9_curriculum import (
    load_order9_learning_config,
    resolve_order9_stage_runtime,
)
from amsrr.training.order9_pi_l_stage_runner import (
    ORDER9_C3_PHASE_RESET_ROLLOUT_MODE,
    ORDER9_C3_STATE_INHERITANCE_ROLLOUT_MODE,
    order9_c3_state_inheritance_phase_indices,
)
from amsrr.training.order9_curriculum_lineage import (
    load_order9_stage_parent_checkpoint,
)
from amsrr.training.order9_c3_nominal_runtime import (
    ORDER9_C3_FORMAL_PHASE_ZERO_STATE_SOURCE,
    ORDER9_C3_NOMINAL_BOUNDARY_COMPLETION_CONTRACT,
    ORDER9_C3_PHASE_RESET_REFERENCE_VERSION,
    Order9C3NominalTensorReference,
    Order9C3PhaseResetReferences,
    Order9C3PhaseStartReference,
    gate_order9_c3_nominal_boundary_completion,
    parse_order9_c3_diagnostic_phase_strata,
)
from amsrr.training.order9_c3_boundary_sampling import (
    ORDER9_C3_BOUNDARY_TAIL_SAMPLING_CONTRACT,
    ORDER9_C3_TRANSITION_BACKWARD_CURRICULUM_CONTRACT,
    order9_c3_fixed_reset_stratum_index,
    order9_c3_reset_strata_by_phase,
)
from amsrr.training.order9_c3_nominal_trajectory import (
    load_order9_c3_accepted_nominal_bundle,
)
from amsrr.training.order9_c3_teacher import (
    build_order9_c3_posture_collision_object,
)
from amsrr.training.order9_actuator_aware_nominal_preload import (
    ORDER9_ACTUATOR_AWARE_NOMINAL_PRELOAD_VERSION,
    Order9ActuatorAwareNominalPreloadConfig,
    solve_order9_actuator_aware_nominal_preload,
)
from amsrr.training.order9_posture_resolver import (
    Order9PostureResolverConfig,
    Order9PostureTrajectoryResolver,
)
from amsrr.training.order9_virtual_contact_compression import (
    ORDER9_CONTACT_COMPRESSION_COLLISION_SHIELD_VERSION,
    ORDER9_CONTACT_COMPRESSION_MAXIMUM_ACTION_JOINT_DELTA_RAD,
    ORDER9_VIRTUAL_CONTACT_COMPRESSION_VERSION,
    Order9VirtualContactCompressionConfig,
    apply_order9_virtual_contact_compression,
    compression_joint_delta_tensor,
    limit_order9_virtual_contact_compression_by_collision,
    order9_virtual_contact_phase_weight,
    solve_order9_virtual_contact_compression,
    solve_order9_virtual_contact_compression_for_achieved_lead,
)
from amsrr.training.order9_anchor_normal_force_estimator import (
    ORDER9_ANCHOR_NORMAL_FORCE_ESTIMATOR_VERSION,
    ORDER9_ANCHOR_REACTION_DIRECTION_CONTRACT,
    Order9AnchorNormalForceEstimator,
    Order9AnchorNormalForceEstimatorConfig,
    order9_anchor_reaction_normal_world,
    order9_branch_free_motion_baseline_update_mask,
    order9_contact_normal_in_contact_frame,
    order9_projected_inward_wrench_interval,
    order9_required_anchor_normal_force_n,
    shift_link_origin_linear_jacobian_to_point,
)
from amsrr.training.order9_force_observer_replay import (
    ORDER9_FORCE_OBSERVER_TRACE_VERSION,
    write_order9_force_observer_trace,
)
from amsrr.training.order9_deployable_phase_gate import (
    ORDER9_DEPLOYABLE_PHASE_GATE_VERSION,
    Order9DeployablePhaseGate,
    Order9DeployablePhaseGateConfig,
    Order9DeployablePhaseGateInput,
)
from amsrr.training.order9_contact_wrench_reward import (
    ORDER9_CONTACT_WRENCH_REWARD_CONTRACT_VERSION,
    Order9TensorWrenchRangeReference,
)
from amsrr.training.order9_evaluation import (
    ORDER9_EVALUATION_HORIZON_TIMEOUT_CONTRACT,
    Order9EvaluationEpisode,
    order9_evaluation_horizon_timeout_outcome,
    write_order9_evaluation_episodes_jsonl,
)
from amsrr.training.order9_pipeline import order9_schedule_hash, order9_stage_by_id
from amsrr.training.order9_rollout_buckets import ORDER9_C3_STAGE_ID
from amsrr.training.order9_runtime_load import Order9RuntimeLoadMonitor
from amsrr.training.order9_teacher import (
    build_order8_grasp_carry_task_spec,
    calibrate_order9_contact_wrench_moment_envelopes,
    upgrade_teacher_trajectory_to_v2,
)
from amsrr.training.order9_tensor_pi_l_runtime import (
    ORDER9_NOMINAL_ONLY_ACTION_MASK_CONTRACT,
    Order9TensorPiLRuntime,
)
from amsrr.training.order9_tensor_reward import (
    ORDER9_C3_PRIVILEGED_PHASE_SUPERVISION_CONTRACT,
    ORDER9_CONTACT_ACQUISITION_SUCCESS_CONTRACT,
    ORDER9_OBJECT_POSITION_TOLERANCE_M,
    ORDER9_OBJECT_POSE_SUCCESS_CONTRACT,
    ORDER9_RELEASE_JOINT_POSITION_TOLERANCE_RAD,
    ORDER9_RELEASE_SUCCESS_CONTRACT,
    ORDER9_TENSOR_REWARD_TERM_NAMES,
    ORDER9_DEPLOYABLE_NORMAL_CONTACT_QUALITY_REWARD_CONTRACT,
    Order9TensorRewardEngine,
    Order9TensorRewardGateConfig,
    Order9TensorRewardInput,
    Order9TensorRewardState,
)
from amsrr.training.order9_tensorboard import (
    ORDER9_TENSORBOARD_LOGGER_VERSION,
    Order9TensorBoardLogger,
)
from amsrr.training.order9_tensor_teacher_reference import (
    load_order9_nominal_tensor_teacher_reference,
)
from amsrr.training.order9_tensor_rollout_artifact import (
    ORDER9_PRODUCTION_COLLECTOR_VERSION,
    Order9TensorRolloutBuffer,
    write_order9_tensor_rollout_artifact,
)
from amsrr.utils.hashing import hash_file, stable_hash

_RESULT_PREFIX = "ORDER9_ROLLOUT_JSON="
_COLLECTOR_VERSION = ORDER9_PRODUCTION_COLLECTOR_VERSION
_SIMULATOR_VERSION = "isaaclab3_physx_gpu_tensor_runtime"
_C3_RESET_BANK_VERSION = (
    "order9_c3_accepted_nominal_physical_reset_bank_v2_boundary_tail"
)


@configclass
class _Order9RolloutSceneCfg(InteractiveSceneCfg):
    pass


class _PhaseStateBank:
    """Per-environment physical reset states with optional intra-phase strata."""

    def __init__(
        self,
        scene: InteractiveScene,
        phase_count: int,
        *,
        stratum_count: int = 1,
        reset_stratum_indices_by_phase: Sequence[Sequence[int]] | None = None,
    ) -> None:
        if stratum_count < 1:
            raise ValueError("Order9 reset bank stratum count must be positive")
        robot = scene["robot"]
        obj = scene["object"]
        batch = scene.num_envs
        device = torch.device(scene.device)
        dtype = _torch(robot.data.root_pose_w).dtype
        joint_count = _torch(robot.data.joint_pos).shape[1]
        self.available = torch.zeros(
            (batch, phase_count, stratum_count), device=device, dtype=torch.bool
        )
        self.robot_root_pose_local = torch.zeros(
            (batch, phase_count, stratum_count, 7), device=device, dtype=dtype
        )
        self.robot_root_twist = torch.zeros(
            (batch, phase_count, stratum_count, 6), device=device, dtype=dtype
        )
        self.joint_position = torch.zeros(
            (batch, phase_count, stratum_count, joint_count),
            device=device,
            dtype=dtype,
        )
        self.joint_velocity = torch.zeros_like(self.joint_position)
        self.object_pose_local = torch.zeros(
            (batch, phase_count, stratum_count, 7), device=device, dtype=dtype
        )
        self.object_twist = torch.zeros(
            (batch, phase_count, stratum_count, 6), device=device, dtype=dtype
        )
        self.phase_elapsed_s = torch.zeros(
            (batch, phase_count, stratum_count), device=device, dtype=dtype
        )
        self.task_body_start_pose_local = torch.zeros(
            (batch, phase_count, stratum_count, 7), device=device, dtype=dtype
        )
        self.task_object_start_pose_local = torch.zeros(
            (batch, phase_count, stratum_count, 7), device=device, dtype=dtype
        )
        self.task_start_available = torch.zeros_like(self.available)
        self.origins = scene.env_origins
        self._object = obj
        self.phase_count = int(phase_count)
        self.stratum_count = int(stratum_count)
        if reset_stratum_indices_by_phase is None:
            allowed = tuple(
                tuple(range(self.stratum_count)) for _ in range(self.phase_count)
            )
        else:
            allowed = tuple(
                tuple(int(index) for index in indices)
                for indices in reset_stratum_indices_by_phase
            )
        if (
            len(allowed) != self.phase_count
            or any(not indices for indices in allowed)
            or any(len(set(indices)) != len(indices) for indices in allowed)
            or any(
                not 0 <= index < self.stratum_count
                for indices in allowed
                for index in indices
            )
        ):
            raise ValueError("Order9 reset-bank selectable strata are invalid")
        self.reset_stratum_indices_by_phase = allowed

    def capture(
        self,
        scene: InteractiveScene,
        *,
        env_ids: torch.Tensor,
        phase_indices: torch.Tensor,
        stratum_indices: torch.Tensor | None = None,
        phase_elapsed_s: torch.Tensor | None = None,
        task_body_start_pose_world: torch.Tensor | None = None,
        task_object_start_pose_world: torch.Tensor | None = None,
    ) -> None:
        if env_ids.numel() == 0:
            return
        robot = scene["robot"]
        obj = scene["object"]
        strata = (
            torch.zeros_like(phase_indices)
            if stratum_indices is None
            else stratum_indices
        )
        root_pose = _torch(robot.data.root_pose_w)[env_ids].clone()
        object_pose = _object_pose(obj)[env_ids].clone()
        root_pose[:, :3] -= self.origins[env_ids]
        object_pose[:, :3] -= self.origins[env_ids]
        self.robot_root_pose_local[env_ids, phase_indices, strata] = root_pose
        self.robot_root_twist[env_ids, phase_indices, strata] = torch.cat(
            (
                _torch(robot.data.root_lin_vel_w)[env_ids],
                _torch(robot.data.root_ang_vel_w)[env_ids],
            ),
            dim=-1,
        )
        self.joint_position[env_ids, phase_indices, strata] = _torch(
            robot.data.joint_pos
        )[env_ids]
        self.joint_velocity[env_ids, phase_indices, strata] = _torch(
            robot.data.joint_vel
        )[env_ids]
        self.object_pose_local[env_ids, phase_indices, strata] = object_pose
        self.object_twist[env_ids, phase_indices, strata] = _object_twist(obj)[env_ids]
        self.phase_elapsed_s[env_ids, phase_indices, strata] = (
            torch.zeros_like(phase_indices, dtype=root_pose.dtype)
            if phase_elapsed_s is None
            else phase_elapsed_s
        )
        if task_body_start_pose_world is not None:
            body_start = task_body_start_pose_world.clone()
            body_start[:, :3] -= self.origins[env_ids]
            self.task_body_start_pose_local[env_ids, phase_indices, strata] = body_start
        if task_object_start_pose_world is not None:
            object_start = task_object_start_pose_world.clone()
            object_start[:, :3] -= self.origins[env_ids]
            self.task_object_start_pose_local[env_ids, phase_indices, strata] = (
                object_start
            )
        if (
            task_body_start_pose_world is not None
            and task_object_start_pose_world is not None
        ):
            self.task_start_available[env_ids, phase_indices, strata] = True
        self.available[env_ids, phase_indices, strata] = True

    def install(
        self,
        *,
        env_id: int,
        phase_index: int,
        stratum_index: int = 0,
        robot_root_pose_local: torch.Tensor,
        robot_root_twist: torch.Tensor,
        joint_position: torch.Tensor,
        joint_velocity: torch.Tensor,
        object_pose_local: torch.Tensor,
        object_twist: torch.Tensor,
        phase_elapsed_s: float = 0.0,
        task_body_start_pose_local: torch.Tensor | None = None,
        task_object_start_pose_local: torch.Tensor | None = None,
    ) -> None:
        key = (env_id, phase_index, stratum_index)
        self.robot_root_pose_local[key] = robot_root_pose_local
        self.robot_root_twist[key] = robot_root_twist
        self.joint_position[key] = joint_position
        self.joint_velocity[key] = joint_velocity
        self.object_pose_local[key] = object_pose_local
        self.object_twist[key] = object_twist
        self.phase_elapsed_s[key] = float(phase_elapsed_s)
        if task_body_start_pose_local is not None:
            self.task_body_start_pose_local[key] = task_body_start_pose_local
        if task_object_start_pose_local is not None:
            self.task_object_start_pose_local[key] = task_object_start_pose_local
        if (
            task_body_start_pose_local is not None
            and task_object_start_pose_local is not None
        ):
            self.task_start_available[key] = True
        self.available[key] = True

    def select_reset_states(
        self, env_ids: torch.Tensor, episode_serial: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected_phase = torch.zeros_like(env_ids)
        selected_stratum = torch.zeros_like(env_ids)
        for output_index, env_id in enumerate(env_ids.tolist()):
            entries = [
                (int(phase), int(stratum))
                for phase, stratum in torch.nonzero(
                    self.available[env_id], as_tuple=False
                ).tolist()
                if int(stratum) in self.reset_stratum_indices_by_phase[int(phase)]
            ]
            if not entries:
                raise RuntimeError("Order9 reset bank has no available state")
            offset = (int(env_id) + int(episode_serial[env_id].item())) % len(entries)
            selected_phase[output_index] = entries[offset][0]
            selected_stratum[output_index] = entries[offset][1]
        return selected_phase, selected_stratum

    def initial_strata(
        self,
        env_ids: torch.Tensor,
        phase_indices: torch.Tensor,
        *,
        formal_phase_zero_start: bool = False,
    ) -> torch.Tensor:
        values = order9_rollout_initial_stratum_indices(
            phase_indices=tuple(int(value) for value in phase_indices.tolist()),
            phase_count=self.phase_count,
            reset_stratum_indices_by_phase=self.reset_stratum_indices_by_phase,
            formal_phase_zero_start=formal_phase_zero_start,
        )
        strata = torch.tensor(values, device=env_ids.device, dtype=env_ids.dtype)
        if formal_phase_zero_start:
            return strata
        if not bool(self.available[env_ids, phase_indices, strata].all()):
            raise RuntimeError("Order9 initial reset stratum is unavailable")
        return strata

    def state_inheritance_strata(
        self,
        env_ids: torch.Tensor,
        phase_indices: torch.Tensor,
        episode_serial: torch.Tensor,
        *,
        fixed_stratum_index: int | None = None,
    ) -> torch.Tensor:
        """Select either a fixed backward-curriculum state or legacy rotation."""

        if fixed_stratum_index is not None:
            if not 0 <= int(fixed_stratum_index) < self.stratum_count:
                raise ValueError(
                    "Order9 fixed state-inheritance stratum is outside the bank"
                )
            strata = torch.full_like(env_ids, int(fixed_stratum_index))
        else:
            strata = torch.zeros_like(env_ids)
            for output_index, (env_id, phase_index) in enumerate(
                zip(env_ids.tolist(), phase_indices.tolist())
            ):
                allowed = self.reset_stratum_indices_by_phase[int(phase_index)]
                serial = int(episode_serial[int(env_id)].item())
                strata[output_index] = allowed[-1 - (serial % len(allowed))]
        if not bool(self.available[env_ids, phase_indices, strata].all()):
            raise RuntimeError("Order9 state-inheritance reset stratum is unavailable")
        return strata

    def restore(
        self,
        scene: InteractiveScene,
        *,
        env_ids: torch.Tensor,
        phase_indices: torch.Tensor,
        stratum_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if env_ids.numel() == 0:
            empty_pose = torch.empty(
                (0, 7), device=self.origins.device, dtype=self.origins.dtype
            )
            return (
                torch.empty_like(env_ids, dtype=self.origins.dtype),
                empty_pose,
                empty_pose,
            )
        strata = (
            torch.zeros_like(phase_indices)
            if stratum_indices is None
            else stratum_indices
        )
        if not bool(self.available[env_ids, phase_indices, strata].all()):
            raise RuntimeError("Order9 attempted an unavailable phase reset")
        scene.reset(env_ids)
        robot = scene["robot"]
        obj = scene["object"]
        root_pose = self.robot_root_pose_local[env_ids, phase_indices, strata].clone()
        object_pose = self.object_pose_local[env_ids, phase_indices, strata].clone()
        root_pose[:, :3] += self.origins[env_ids]
        object_pose[:, :3] += self.origins[env_ids]
        q = self.joint_position[env_ids, phase_indices, strata]
        qdot = self.joint_velocity[env_ids, phase_indices, strata]
        robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        robot.write_root_velocity_to_sim_index(
            root_velocity=self.robot_root_twist[env_ids, phase_indices, strata],
            env_ids=env_ids,
        )
        robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
        robot.write_joint_velocity_to_sim_index(velocity=qdot, env_ids=env_ids)
        robot.set_joint_position_target_index(target=q, env_ids=env_ids)
        robot.set_joint_velocity_target_index(
            target=torch.zeros_like(qdot), env_ids=env_ids
        )
        robot.set_joint_effort_target_index(
            target=torch.zeros_like(qdot), env_ids=env_ids
        )
        obj.write_root_pose_to_sim_index(root_pose=object_pose, env_ids=env_ids)
        obj.write_root_velocity_to_sim_index(
            root_velocity=self.object_twist[env_ids, phase_indices, strata],
            env_ids=env_ids,
        )
        body_start = self.task_body_start_pose_local[
            env_ids, phase_indices, strata
        ].clone()
        object_start = self.task_object_start_pose_local[
            env_ids, phase_indices, strata
        ].clone()
        body_start[:, :3] += self.origins[env_ids]
        object_start[:, :3] += self.origins[env_ids]
        return (
            self.phase_elapsed_s[env_ids, phase_indices, strata].clone(),
            body_start,
            object_start,
        )


def _c3_maintained_phase_indices() -> tuple[int, ...]:
    return tuple(
        ORDER9_OBJECT_TASK_PHASES.index(phase)
        for phase in (
            Order9ObjectTaskPhase.LIFT,
            Order9ObjectTaskPhase.TRANSPORT,
            Order9ObjectTaskPhase.PLACE,
        )
    )


def _cpu_clone(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu").clone()


def _write_c3_reset_bank(
    path: Path,
    *,
    bank: _PhaseStateBank,
    reset_references: Order9C3PhaseResetReferences,
    contract: dict[str, object],
    stabilization: dict[str, object],
) -> str:
    rows = []
    for phase_index in _c3_maintained_phase_indices():
        for stratum_index in range(bank.stratum_count):
            key = (0, phase_index, stratum_index)
            if not bool(bank.available[key].item()):
                raise RuntimeError(
                    "C3 reset bank contains an unavailable maintained state"
                )
            rows.append(
                {
                    "phase_index": phase_index,
                    "stratum_index": stratum_index,
                    "robot_root_pose_local": _cpu_clone(
                        bank.robot_root_pose_local[key]
                    ),
                    "robot_root_twist": _cpu_clone(bank.robot_root_twist[key]),
                    "joint_position": _cpu_clone(bank.joint_position[key]),
                    "joint_velocity": _cpu_clone(bank.joint_velocity[key]),
                    "object_pose_local": _cpu_clone(bank.object_pose_local[key]),
                    "object_twist": _cpu_clone(bank.object_twist[key]),
                    "phase_elapsed_s": float(bank.phase_elapsed_s[key].item()),
                    "task_body_start_pose_local": _cpu_clone(
                        bank.task_body_start_pose_local[key]
                    ),
                    "task_object_start_pose_local": _cpu_clone(
                        bank.task_object_start_pose_local[key]
                    ),
                    "body_pose_local": _cpu_clone(
                        reset_references.body_pose_local[phase_index, stratum_index]
                    ),
                    "body_twist": _cpu_clone(
                        reset_references.body_twist[phase_index, stratum_index]
                    ),
                    "reference_joint_position": _cpu_clone(
                        reset_references.joint_positions_rad[phase_index, stratum_index]
                    ),
                    "reference_joint_velocity": _cpu_clone(
                        reset_references.joint_velocities_radps[
                            phase_index, stratum_index
                        ]
                    ),
                    "reference_object_pose_local": _cpu_clone(
                        reset_references.object_pose_local[phase_index, stratum_index]
                    ),
                    "reference_object_twist": _cpu_clone(
                        reset_references.object_twist[phase_index, stratum_index]
                    ),
                    "reference_phase_elapsed_s": float(
                        reset_references.phase_elapsed_s[
                            phase_index, stratum_index
                        ].item()
                    ),
                }
            )
    payload = {
        "version": _C3_RESET_BANK_VERSION,
        "contract": dict(contract),
        "phase_count": bank.phase_count,
        "stratum_count": bank.stratum_count,
        "joint_count": int(bank.joint_position.shape[-1]),
        "states": rows,
        "stabilization": dict(stabilization),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{time.time_ns()}")
    torch.save(payload, temporary)
    temporary.replace(path)
    return hash_file(path)


def _load_c3_reset_bank(
    path: Path,
    *,
    bank: _PhaseStateBank,
    reset_references: Order9C3PhaseResetReferences,
    contract: dict[str, object],
) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("version") != _C3_RESET_BANK_VERSION
    ):
        raise RuntimeError(f"Order9 C3 reset bank version differs: {path}")
    if payload.get("contract") != contract:
        raise RuntimeError(f"Order9 C3 reset bank contract differs: {path}")
    if (
        int(payload.get("phase_count", -1)) != bank.phase_count
        or int(payload.get("stratum_count", -1)) != bank.stratum_count
        or int(payload.get("joint_count", -1)) != int(bank.joint_position.shape[-1])
    ):
        raise RuntimeError(f"Order9 C3 reset bank tensor shape differs: {path}")
    rows = payload.get("states")
    expected_rows = len(_c3_maintained_phase_indices()) * bank.stratum_count
    if not isinstance(rows, list) or len(rows) != expected_rows:
        raise RuntimeError(f"Order9 C3 reset bank state count differs: {path}")
    device = bank.available.device
    dtype = bank.robot_root_pose_local.dtype

    def tensor(row: dict[str, object], name: str) -> torch.Tensor:
        value = row.get(name)
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"Order9 C3 reset bank tensor is missing: {name}")
        return value.to(device=device, dtype=dtype)

    seen: set[tuple[int, int]] = set()
    for raw_row in rows:
        if not isinstance(raw_row, dict):
            raise RuntimeError("Order9 C3 reset bank state is not a mapping")
        phase_index = int(raw_row["phase_index"])
        stratum_index = int(raw_row["stratum_index"])
        key = (phase_index, stratum_index)
        if (
            phase_index not in _c3_maintained_phase_indices()
            or not 0 <= stratum_index < bank.stratum_count
            or key in seen
        ):
            raise RuntimeError(f"Order9 C3 reset bank state identity differs: {key}")
        seen.add(key)
        for env_id in range(bank.available.shape[0]):
            bank.install(
                env_id=env_id,
                phase_index=phase_index,
                stratum_index=stratum_index,
                robot_root_pose_local=tensor(raw_row, "robot_root_pose_local"),
                robot_root_twist=tensor(raw_row, "robot_root_twist"),
                joint_position=tensor(raw_row, "joint_position"),
                joint_velocity=tensor(raw_row, "joint_velocity"),
                object_pose_local=tensor(raw_row, "object_pose_local"),
                object_twist=tensor(raw_row, "object_twist"),
                phase_elapsed_s=float(raw_row["phase_elapsed_s"]),
                task_body_start_pose_local=tensor(
                    raw_row, "task_body_start_pose_local"
                ),
                task_object_start_pose_local=tensor(
                    raw_row, "task_object_start_pose_local"
                ),
            )
        reset_references.body_pose_local[phase_index, stratum_index] = tensor(
            raw_row, "body_pose_local"
        )
        reset_references.body_twist[phase_index, stratum_index] = tensor(
            raw_row, "body_twist"
        )
        reset_references.joint_positions_rad[phase_index, stratum_index] = tensor(
            raw_row, "reference_joint_position"
        )
        reset_references.joint_velocities_radps[phase_index, stratum_index] = tensor(
            raw_row, "reference_joint_velocity"
        )
        reset_references.object_pose_local[phase_index, stratum_index] = tensor(
            raw_row, "reference_object_pose_local"
        )
        reset_references.object_twist[phase_index, stratum_index] = tensor(
            raw_row, "reference_object_twist"
        )
        reset_references.phase_elapsed_s[phase_index, stratum_index] = float(
            raw_row["reference_phase_elapsed_s"]
        )
    stabilization = payload.get("stabilization")
    if not isinstance(stabilization, dict) or stabilization.get("passed") is not True:
        raise RuntimeError(
            f"Order9 C3 reset bank stabilization evidence differs: {path}"
        )
    return {
        **stabilization,
        "loaded_from_reset_bank": True,
        "reset_bank_path": str(path.resolve()),
        "reset_bank_sha256": hash_file(path),
    }


def _accepted_nominal_reset_bank_evidence(
    bank: _PhaseStateBank,
    reset_references: Order9C3PhaseResetReferences,
) -> dict[str, object]:
    """Validate static reset snapshots without asking a policy to solve them."""

    tensors = {
        "robot_root_pose_local": bank.robot_root_pose_local,
        "robot_root_twist": bank.robot_root_twist,
        "joint_position": bank.joint_position,
        "joint_velocity": bank.joint_velocity,
        "object_pose_local": bank.object_pose_local,
        "object_twist": bank.object_twist,
        "phase_elapsed_s": bank.phase_elapsed_s,
        "body_pose_local": reset_references.body_pose_local,
        "body_twist": reset_references.body_twist,
        "reference_joint_positions_rad": reset_references.joint_positions_rad,
        "reference_joint_velocities_radps": (reset_references.joint_velocities_radps),
        "reference_object_pose_local": reset_references.object_pose_local,
        "reference_object_twist": reset_references.object_twist,
        "reference_phase_elapsed_s": reset_references.phase_elapsed_s,
    }
    non_finite = [
        name for name, value in tensors.items() if not bool(torch.isfinite(value).all())
    ]
    if non_finite:
        raise RuntimeError(
            f"Order9 C3 accepted-nominal reset bank is non-finite: {non_finite}"
        )
    maintained = _c3_maintained_phase_indices()
    missing = [
        (phase_index, stratum_index)
        for phase_index in maintained
        for stratum_index in range(bank.stratum_count)
        if not bool(bank.available[:, phase_index, stratum_index].all())
    ]
    if missing:
        raise RuntimeError(
            f"Order9 C3 accepted-nominal reset states are unavailable: {missing}"
        )
    return {
        "version": "order9_c3_accepted_nominal_static_reset_v1",
        "passed": True,
        "reset_source": "hash_bound_human_accepted_nominal_trajectory",
        "pi_l_actor_used": False,
        "qpid_used": False,
        "dynamic_stability_gate_required": False,
        "contact_maintenance_is_learning_and_evaluation_target": True,
        "static_validity_checks": [
            "finite_physical_state",
            "complete_phase_strata",
            "accepted_nominal_hash_binding",
            "accepted_offline_collision_review",
        ],
        "admitted_state_count": len(maintained) * bank.stratum_count,
    }


def main() -> dict[str, object]:
    repository = Path(__file__).resolve().parents[1]
    config_path = (repository / args_cli.config).resolve()
    config = load_order9_learning_config(config_path)
    stage = order9_stage_by_id(config, args_cli.stage)
    c3_privileged_phase_supervision = stage.stage_id == ORDER9_C3_STAGE_ID
    evaluation_mode = args_cli.evaluation_jsonl is not None
    if evaluation_mode and args_cli.c3_wrench_gate_range_scale != 1.0:
        raise ValueError("Order9 deterministic evaluation requires exact wrench gate")
    if (
        stage.stage_id != ORDER9_C3_STAGE_ID
        and args_cli.c3_wrench_gate_range_scale != 1.0
    ):
        raise ValueError("wrench-gate curriculum is restricted to C3")
    deterministic_policy = evaluation_mode or bool(
        args_cli.diagnostic_deterministic_policy
    )
    if stage.learning_target.value != "pi_l" or (
        stage.learning_mode.value != "ppo" and not evaluation_mode
    ):
        raise ValueError(
            "vectorized pi_L rollout requires a pi_L PPO stage or evaluation mode"
        )
    if bool(stage.topology_randomized) and args_cli.morphology_graph_json is None:
        raise ValueError("topology-randomized stage requires --morphology-graph-json")
    if stage.stage_id == "c3_pi_l_ppo_arbitrary_morphology" and not all(
        (
            args_cli.c3_nominal_set_manifest,
            args_cli.c3_nominal_set_sha256,
            args_cli.c3_nominal_artifact_sha256,
        )
    ):
        raise ValueError(
            "C3 rollout requires its hash-bound accepted nominal trajectory"
        )
    if (
        stage.stage_id == "c3_pi_l_ppo_arbitrary_morphology"
        and args_cli.c3_reset_bank is None
    ):
        raise ValueError("C3 rollout requires its accepted-nominal reset-bank path")
    configured_runtime = resolve_order9_stage_runtime(config, stage)
    if configured_runtime.rollout_steps_per_environment is None and not evaluation_mode:
        raise ValueError("vectorized pi_L rollout requires a PPO stage runtime")
    runtime_override_used = (
        args_cli.num_envs is not None or args_cli.rollout_steps is not None
    )
    topology_stratified_shard = args_cli.production_topology_shard_index is not None
    state_inheritance_shard = bool(args_cli.production_state_inheritance_shard)
    if state_inheritance_shard and (
        stage.stage_id != ORDER9_C3_STAGE_ID
        or not topology_stratified_shard
        or evaluation_mode
        or not config.production_runtime.c3_state_inheritance_rollouts_enabled
        or args_cli.rollout_steps is not None
        or args_cli.production_topology_shard_environment_count
        != config.production_runtime.c3_state_inheritance_environment_count_per_module
    ):
        raise ValueError(
            "production state-inheritance collection requires its exact C3 "
            "train-shard runtime"
        )
    if topology_stratified_shard:
        if (
            stage.stage_id != "c3_pi_l_ppo_arbitrary_morphology"
            or not stage.topology_randomized
            or args_cli.split != "train"
            or args_cli.num_envs is not None
            or not config.production_runtime.c3_topology_stratified_updates
        ):
            raise ValueError("production topology shards require the C3 train runtime")
        args_cli.num_envs = args_cli.production_topology_shard_environment_count
    elif args_cli.num_envs is None:
        args_cli.num_envs = configured_runtime.environment_count
    if diagnostic_virtual_contact_additional_lead_sweep_mm is not None:
        sweep_count = len(diagnostic_virtual_contact_additional_lead_sweep_mm)
        if (
            int(args_cli.num_envs) % sweep_count != 0
            or int(args_cli.evaluation_episode_count) != int(args_cli.num_envs)
        ):
            raise ValueError(
                "diagnostic additional-lead sweep requires num-envs divisible "
                "by the sweep size and one evaluation episode per environment"
            )
    if diagnostic_contact_normal_residual_sweep_mm is not None:
        sweep_count = len(diagnostic_contact_normal_residual_sweep_mm)
        if (
            int(args_cli.num_envs) % sweep_count != 0
            or int(args_cli.evaluation_episode_count) != int(args_cli.num_envs)
        ):
            raise ValueError(
                "diagnostic contact-normal residual sweep requires num-envs "
                "divisible by the sweep size and one evaluation episode per "
                "environment"
            )
    diagnostic_phase_strata = (
        None
        if args_cli.diagnostic_initial_phase_strata is None
        else parse_order9_c3_diagnostic_phase_strata(
            args_cli.diagnostic_initial_phase_strata,
            environment_count=int(args_cli.num_envs),
        )
    )
    if state_inheritance_shard:
        args_cli.rollout_steps = (
            config.production_runtime.c3_state_inheritance_rollout_steps
        )
    elif args_cli.rollout_steps is None:
        if configured_runtime.rollout_steps_per_environment is None:
            raise ValueError("BC-stage evaluation requires an explicit --rollout-steps")
        args_cli.rollout_steps = configured_runtime.rollout_steps_per_environment
    if evaluation_mode and args_cli.evaluation_episode_count > args_cli.num_envs:
        raise ValueError("evaluation episode count cannot exceed the environment count")
    physical = build_physical_model_from_config(
        repository / config.production_runtime.robot_model_config_path
    )
    actuator_runtime = order9_actuator_runtime_values(physical)
    canonical = load_order9_canonical_reset(
        repository / config.production_runtime.canonical_order8_report_path,
        expected_sha256=config.production_runtime.canonical_order8_report_sha256,
    )
    morphology = (
        MorphologyGraph.from_json(
            Path(args_cli.morphology_graph_json).read_text(encoding="utf-8")
        )
        if args_cli.morphology_graph_json
        else build_representative_order8_morphology(physical)
    )
    morphology.validate()
    robot_asset_manifest = None
    if bool(stage.topology_randomized):
        if args_cli.robot_usd is None:
            raise ValueError("topology-randomized rollout requires --robot-usd")
        robot_usd = Path(args_cli.robot_usd).resolve()
    else:
        robot_asset_manifest_path = (
            repository / args_cli.fixed_nominal_asset_manifest
        ).resolve()
        robot_asset_manifest = load_order9_fixed_nominal_asset_manifest(
            robot_asset_manifest_path
        )
        robot_usd = validate_order9_fixed_nominal_asset_manifest_bytes(
            robot_asset_manifest,
            repository_root=repository,
            expected_morphology=morphology,
            expected_physical_model_hash=physical.stable_hash(),
        )
        if (
            args_cli.robot_usd is not None
            and Path(args_cli.robot_usd).resolve() != robot_usd
        ):
            raise ValueError(
                "fixed rollout --robot-usd differs from its hash-bound manifest"
            )
    if not robot_usd.is_file():
        raise FileNotFoundError(robot_usd)
    task = _load_task(repository, config, canonical)
    target_object, geometry = _target_object_and_geometry(task)
    geometry_kind, geometry_values, object_half_height = _geometry_values(geometry)
    object_mass = float(target_object.mass_kg or 0.0)
    if object_mass <= 0.0 or target_object.inertia_kgm2 is None:
        raise ValueError("Order9 rollout object requires positive mass and inertia")
    object_friction = float(target_object.friction or 0.0)
    friction_resolution = resolve_contact_friction(
        task.metadata,
        target_entity_id=target_object.object_id,
        contact_mode=ContactMode.GRASP,
        target_surface_friction=object_friction,
    )
    selected_friction = (
        float(args_cli.selected_gripper_friction)
        if args_cli.selected_gripper_friction is not None
        else float(friction_resolution.robot_surface_friction or 4.5)
    )
    accepted_nominal_bundle = None
    if bool(stage.topology_randomized):
        bucket_id = task.metadata.get("order9_rollout_bucket_id")
        if not isinstance(bucket_id, str) or not bucket_id:
            raise ValueError("C3 task lacks its rollout bucket identity")
        accepted_nominal_bundle = load_order9_c3_accepted_nominal_bundle(
            Path(str(args_cli.c3_nominal_set_manifest)),
            bucket_id=bucket_id,
            repository_root=repository,
            expected_set_sha256=str(args_cli.c3_nominal_set_sha256),
            expected_artifact_sha256=str(args_cli.c3_nominal_artifact_sha256),
            expected_task_spec_sha256=hash_file(Path(str(args_cli.task_spec_json))),
            expected_physical_model_hash=physical.stable_hash(),
        )
        morphology = accepted_nominal_bundle.morphology
        teacher_trajectory = calibrate_order9_contact_wrench_moment_envelopes(
            accepted_nominal_bundle.trajectory,
            accepted_nominal_bundle.contact_candidate_set,
        )
        candidates = accepted_nominal_bundle.contact_candidate_set
        assignments = _maintain_assignments(teacher_trajectory)
    else:
        teacher_trajectory, assignments, candidates = _teacher_assignments(
            task, morphology
        )
    selected_anchor_ids = tuple(assignment.anchor_id for assignment in assignments)
    if len(selected_anchor_ids) < 2 or len(set(selected_anchor_ids)) != len(
        selected_anchor_ids
    ):
        raise RuntimeError(
            "Order9 grasp teacher must select unique multi-contact anchors"
        )
    articulated_link_ids = tuple(
        link.link_id for link in physical.links if float(link.mass_kg) > 0.0
    )
    physical_robot_body_names = tuple(
        f"module_{module.module_id}__{link.link_id}"
        for module in sorted(morphology.modules, key=lambda value: value.module_id)
        for link in physical.links
        if link.link_id in articulated_link_ids
    )
    internal_robot_body_names: tuple[str, ...] = ()
    source_urdf = (
        Path(robot_asset_manifest.source_urdf_path)
        if robot_asset_manifest is not None
        else Path(physical.urdf_path)
    )
    if not source_urdf.is_absolute():
        source_urdf = repository / source_urdf
    internal_robot_body_names = tuple(
        f"module_{connection.child_module_id}__"
        f"{connection.child_mechanism_joint_id}__reroot_offset_link"
        for connection in articulated_morphology_graph_connections(
            source_urdf,
            morphology_graph=morphology,
        )
        if connection.child_mechanism_joint_id is not None
    )
    robot_body_names_expected = (
        *physical_robot_body_names,
        *internal_robot_body_names,
    )
    scene_cfg = _scene_cfg(
        robot_usd=robot_usd,
        object_kind=geometry_kind,
        geometry_values=geometry_values,
        object_pose=target_object.pose_world,
        object_mass=object_mass,
        object_friction=object_friction,
        support_size=tuple(canonical.metadata["object_support_size_m"]),
        support_pose=tuple(canonical.metadata["object_support_pose_world"]),
        robot_body_names=robot_body_names_expected,
        actuator_runtime=actuator_runtime,
        collision_contact_offset_m=args_cli.collision_contact_offset_m,
        collision_rest_offset_m=args_cli.collision_rest_offset_m,
    )
    sim_utils.create_new_stage()
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=float(args_cli.dt),
            device=str(args_cli.device),
            use_fabric=True,
        )
    )
    load_monitor = Order9RuntimeLoadMonitor(
        sample_interval_s=config.production_runtime.runtime_load_sample_interval_s,
        device=str(args_cli.device),
    )
    load_monitor.start(torch_module=torch)
    setup_started = time.perf_counter()
    scene_cfg.num_envs = int(args_cli.num_envs)
    scene_cfg.env_spacing = float(args_cli.env_spacing)
    scene_cfg.replicate_physics = True
    scene_cfg.lazy_sensor_update = False
    scene = InteractiveScene(scene_cfg)
    selected_names = tuple(
        f"module_{anchor.module_id}__{anchor.link_id}"
        for anchor_id in selected_anchor_ids
        for anchor in morphology.robot_anchors
        if anchor.anchor_id == anchor_id
    )
    if len(selected_names) != len(selected_anchor_ids):
        raise RuntimeError("Order9 selected anchor body identity is incomplete")
    _bind_selected_material(
        sim.stage,
        selected_body_names=selected_names,
        friction=selected_friction,
        stiffness=float(args_cli.contact_stiffness),
        damping=float(args_cli.contact_damping),
    )
    _activate_nested_contact_reports(sim.stage)
    sim.reset()
    scene.reset()
    robot = scene["robot"]
    obj = scene["object"]
    robot_sensor = scene["robot_contact"]
    object_sensor = scene["object_contact"]
    object_mass_properties_readback = _validate_object_mass_properties(
        obj,
        expected_mass_kg=object_mass,
        expected_inertia_kgm2=tuple(target_object.inertia_kgm2),
        expected_com_object=tuple(
            target_object.center_of_mass_object or (0.0, 0.0, 0.0)
        ),
    )
    actuator_readback = validate_order9_actuator_readback(
        robot.joint_names,
        _torch(robot.data.joint_effort_limits)[0].tolist(),
        _torch(robot.data.joint_velocity_limits)[0].tolist(),
        expected=actuator_runtime,
    )
    if set(robot.body_names) != set(robot_body_names_expected):
        raise RuntimeError(
            "Order9 robot USD body identity differs from morphology: "
            f"missing={sorted(set(robot_body_names_expected) - set(robot.body_names))}, "
            f"extra={sorted(set(robot.body_names) - set(robot_body_names_expected))}"
        )
    if object_sensor.contact_view.filter_count != len(robot.body_names):
        raise RuntimeError(
            "Order9 object contact filter count differs from robot bodies"
        )
    robot_sensor_order = tuple(str(value) for value in robot_sensor.body_names)
    if set(robot_sensor_order) != set(robot.body_names):
        raise RuntimeError("Order9 robot contact sensor body identity differs")
    robot_sensor_reorder = torch.tensor(
        [robot_sensor_order.index(name) for name in robot.body_names],
        device=scene.device,
        dtype=torch.long,
    )
    io = Order9TensorIsaacIO(
        morphology_graph=morphology,
        physical_model=physical,
        robot_body_names=robot.body_names,
        robot_joint_names=robot.joint_names,
        # The object ContactSensor emits one force-matrix column per filter in
        # configuration order, which is the morphology-derived body order.
        object_filter_body_names=robot_body_names_expected,
        selected_anchor_ids=selected_anchor_ids,
    )
    wrench_range_reference = Order9TensorWrenchRangeReference(
        trajectory=teacher_trajectory,
        contact_candidate_set=candidates,
        selected_anchor_ids=selected_anchor_ids,
        object_id=target_object.object_id,
        authored_object_pose_world=target_object.pose_world,
        batch_size=scene.num_envs,
        device=scene.device,
        dtype=torch.float32,
    )
    checkpoint = load_order9_stage_parent_checkpoint(
        config,
        stage,
        args_cli.pi_l_checkpoint,
        device=scene.device,
        expected_family=Order9PolicyFamily.PI_L,
    )
    if checkpoint.sha256 != args_cli.pi_l_checkpoint_sha256:
        raise RuntimeError("Order9 pi_L checkpoint SHA-256 mismatch")
    controller = BatchedQPIDController(
        config=QPIDControllerConfig(
            allocation_mode="rigid_body_qp", control_dt_s=float(args_cli.dt)
        )
    )
    contact_space_action_basis = None
    if checkpoint.metadata.policy_version in ORDER9_CONTACT_SPACE_PI_L_POLICY_VERSIONS:
        if accepted_nominal_bundle is None:
            raise ValueError("Order9 v7 pi_L requires its accepted nominal bundle")
        if (
            args_cli.c3_action_contract
            != ORDER9_C3_ACTION_CONTRACT_CONTACT_SPACE_PROJECTED
        ):
            raise ValueError("Order9 v7 pi_L requires its contact-space contract")
        contact_trajectory = accepted_nominal_bundle.phase_trajectories[
            Order9ObjectTaskPhase.CONTACT_ACQUISITION.value
        ]
        contact_space_action_basis = build_order9_contact_space_action_basis(
            morphology=morphology,
            physical_model=physical,
            contact_knot=contact_trajectory.knots[-1],
            candidate_set=accepted_nominal_bundle.contact_candidate_set,
            module_ids=tuple(sorted(module.module_id for module in morphology.modules)),
            local_joint_ids=tuple(
                sorted(
                    {
                        str(port.mechanical_limits["mechanism_joint_id"])
                        for port in physical.dock_ports
                        if port.mechanical_limits.get("mechanism_joint_id")
                    }
                )
            ),
            config=Order9ContactSpaceActionConfig(
                normal_translation_quantization_step_m=(
                    config.optimization.c3_boundary_fine_tune.contact_normal_action_quantization_step_m
                )
            ),
        )
    policy_runtime = Order9TensorPiLRuntime(
        morphology_graph=morphology,
        physical_model=physical,
        policy=checkpoint.model,
        batch_size=scene.num_envs,
        device=scene.device,
        controller=controller,
        policy_frame_origins_world=scene.env_origins,
        active_knot_trajectory=teacher_trajectory,
        contact_space_action_basis=contact_space_action_basis,
    )
    diagnostic_contact_normal_residual_m_by_environment = None
    if diagnostic_contact_normal_residual_sweep_mm is not None:
        repeats = scene.num_envs // len(
            diagnostic_contact_normal_residual_sweep_mm
        )
        diagnostic_contact_normal_residual_m_by_environment = torch.tensor(
            [
                float(value_mm) * 1.0e-3
                for value_mm in diagnostic_contact_normal_residual_sweep_mm
                for _ in range(repeats)
            ],
            device=scene.device,
            dtype=torch.float32,
        )
    c3_nominal_reference = (
        None
        if accepted_nominal_bundle is None
        else Order9C3NominalTensorReference(
            phase_trajectories=(accepted_nominal_bundle.phase_trajectories),
            module_ids=policy_runtime.builder.module_ids,
            joint_ids=policy_runtime.decoder.local_joint_ids,
            device=scene.device,
            dtype=torch.float32,
            provenance=accepted_nominal_bundle.provenance,
        )
    )
    virtual_contact_lead_m = (
        float(args_cli.virtual_contact_lead_mm) * 1.0e-3
        if args_cli.virtual_contact_lead_mm is not None
        else float(config.production_runtime.c3_virtual_contact_inward_lead_m)
    )
    actuator_aware_nominal_preload_solution = None
    if (
        c3_nominal_reference is not None
        and accepted_nominal_bundle is not None
        and args_cli.virtual_contact_lead_mm is None
        and config.production_runtime.c3_actuator_aware_nominal_preload_enabled
    ):
        contact_trajectory = accepted_nominal_bundle.phase_trajectories[
            Order9ObjectTaskPhase.CONTACT_ACQUISITION.value
        ]
        actuator_aware_nominal_preload_solution = (
            solve_order9_actuator_aware_nominal_preload(
                morphology=morphology,
                physical_model=physical,
                contact_knot=contact_trajectory.knots[-1],
                candidate_set=accepted_nominal_bundle.contact_candidate_set,
                object_mass_kg=object_mass,
                contact_friction=selected_friction,
                contact_stiffness_n_per_m=float(args_cli.contact_stiffness),
                config=Order9ActuatorAwareNominalPreloadConfig(
                    minimum_inward_lead_m=virtual_contact_lead_m,
                    maximum_inward_lead_m=float(
                        config.production_runtime.c3_actuator_aware_nominal_preload_maximum_m
                    ),
                    inward_lead_quantization_m=float(
                        config.production_runtime.c3_actuator_aware_nominal_preload_quantization_m
                    ),
                    support_safety_factor=float(
                        config.production_runtime.c3_actuator_aware_nominal_preload_support_safety_factor
                    ),
                    maximum_peak_effort_utilization=float(
                        config.production_runtime.c3_actuator_aware_nominal_preload_maximum_peak_utilization
                    ),
                    minimum_anchor_force_fraction=float(
                        config.production_runtime.c3_actuator_aware_nominal_preload_minimum_anchor_fraction
                    ),
                    model_error_margin_m=float(
                        config.production_runtime.c3_actuator_aware_nominal_preload_model_error_margin_m
                    ),
                ),
            )
        )
        if not actuator_aware_nominal_preload_solution.feasible:
            raise RuntimeError(
                "Order9 actuator-aware nominal preload rejected the contact "
                "assignment: "
                f"{actuator_aware_nominal_preload_solution.rejection_reason}"
            )
        virtual_contact_lead_m = float(
            actuator_aware_nominal_preload_solution.inward_lead_m
        )
    contact_compression_action_span_m = (
        float(args_cli.contact_compression_action_span_mm) * 1.0e-3
        if args_cli.contact_compression_action_span_mm is not None
        else float(
            config.production_runtime.c3_contact_compression_action_span_m
        )
    )
    virtual_contact_solution = None
    virtual_contact_joint_delta = None
    diagnostic_virtual_contact_total_lead_m_by_environment = None
    contact_compression_action_solution = None
    contact_compression_collision_limit = None
    nominal_preload_collision_limit = None
    nominal_preload_applied_inward_lead_m = None
    training_nominal_preload_deficit_plan = None
    contact_compression_action_joint_direction = None
    contact_compression_action_control_joint_id = None
    if (
        c3_nominal_reference is not None
        and diagnostic_virtual_contact_additional_lead_sweep_mm is not None
    ):
        if accepted_nominal_bundle is None:  # pragma: no cover
            raise AssertionError("C3 virtual contact lacks its nominal bundle")
        contact_trajectory = accepted_nominal_bundle.phase_trajectories[
            Order9ObjectTaskPhase.CONTACT_ACQUISITION.value
        ]
        additional_values = diagnostic_virtual_contact_additional_lead_sweep_mm
        total_values = tuple(
            virtual_contact_lead_m + float(value) * 1.0e-3
            for value in additional_values
        )
        if any(
            value
            > float(
                config.production_runtime.c3_actuator_aware_nominal_preload_maximum_m
            )
            for value in total_values
        ):
            raise ValueError(
                "configured plus diagnostic virtual-contact lead exceeds the "
                "nominal preload limit"
            )
        solution_by_total_lead = {
            total_lead: solve_order9_virtual_contact_compression(
                morphology=morphology,
                physical_model=physical,
                contact_knot=contact_trajectory.knots[-1],
                candidate_set=accepted_nominal_bundle.contact_candidate_set,
                config=Order9VirtualContactCompressionConfig(
                    inward_lead_m=total_lead
                ),
            )
            for total_lead in total_values
        }
        repeats = scene.num_envs // len(total_values)
        diagnostic_virtual_contact_total_lead_m_by_environment = tuple(
            total_lead
            for total_lead in total_values
            for _ in range(repeats)
        )
        virtual_contact_joint_delta = torch.stack(
            [
                compression_joint_delta_tensor(
                    solution_by_total_lead[total_lead],
                    module_ids=policy_runtime.builder.module_ids,
                    local_joint_ids=policy_runtime.decoder.local_joint_ids,
                    device=scene.device,
                    dtype=torch.float32,
                )
                for total_lead in diagnostic_virtual_contact_total_lead_m_by_environment
            ]
        )
        virtual_contact_solution = solution_by_total_lead[total_values[0]]
    elif c3_nominal_reference is not None and virtual_contact_lead_m > 0.0:
        if accepted_nominal_bundle is None:  # pragma: no cover
            raise AssertionError("C3 virtual contact lacks its nominal bundle")
        contact_trajectory = accepted_nominal_bundle.phase_trajectories[
            Order9ObjectTaskPhase.CONTACT_ACQUISITION.value
        ]
        if actuator_aware_nominal_preload_solution is None:
            virtual_contact_solution = solve_order9_virtual_contact_compression(
                morphology=morphology,
                physical_model=physical,
                contact_knot=contact_trajectory.knots[-1],
                candidate_set=accepted_nominal_bundle.contact_candidate_set,
                config=Order9VirtualContactCompressionConfig(
                    inward_lead_m=virtual_contact_lead_m
                ),
            )
        else:
            virtual_contact_solution = (
                solve_order9_virtual_contact_compression_for_achieved_lead(
                    morphology=morphology,
                    physical_model=physical,
                    contact_knot=contact_trajectory.knots[-1],
                    candidate_set=accepted_nominal_bundle.contact_candidate_set,
                    minimum_achieved_inward_lead_m=virtual_contact_lead_m,
                    maximum_requested_inward_lead_m=float(
                        config.production_runtime.c3_actuator_aware_nominal_preload_maximum_m
                    ),
                    requested_lead_quantization_m=float(
                        config.production_runtime.c3_actuator_aware_nominal_preload_quantization_m
                    ),
                )
            )
        virtual_contact_joint_delta = compression_joint_delta_tensor(
            virtual_contact_solution,
            module_ids=policy_runtime.builder.module_ids,
            local_joint_ids=policy_runtime.decoder.local_joint_ids,
            device=scene.device,
            dtype=torch.float32,
        )
    if (
        actuator_aware_nominal_preload_solution is not None
        and virtual_contact_solution is not None
        and accepted_nominal_bundle is not None
    ):
        contact_trajectory = accepted_nominal_bundle.phase_trajectories[
            Order9ObjectTaskPhase.CONTACT_ACQUISITION.value
        ]
        collision_object = build_order9_c3_posture_collision_object(task)
        collision_resolver = Order9PostureTrajectoryResolver(
            physical,
            config=Order9PostureResolverConfig(collision_margin_m=0.001),
            collision_object=collision_object,
            prefer_native_solver=True,
            require_native_solver=True,
        )
        nominal_preload_collision_limit = (
            limit_order9_virtual_contact_compression_by_collision(
                morphology=morphology,
                physical_model=physical,
                contact_knot=contact_trajectory.knots[-1],
                solution=virtual_contact_solution,
                collision_object=collision_object,
                collision_solver=collision_resolver.ik_solver,
                collision_margin_m=0.001,
                maximum_action_joint_delta_rad=0.15,
            )
        )
        collision_safe_scale = float(
            nominal_preload_collision_limit.maximum_action_scale
        )
        nominal_preload_applied_inward_lead_m = collision_safe_scale * min(
            float(value)
            for value in virtual_contact_solution.achieved_inward_displacement_m.values()
        )
        minimum_physical_lead_m = max(
            float(config.production_runtime.c3_virtual_contact_inward_lead_m),
            float(
                actuator_aware_nominal_preload_solution.compliance_predicted_inward_lead_m
            ),
        )
        if (
            diagnostic_virtual_contact_additional_lead_sweep_mm is None
            and nominal_preload_applied_inward_lead_m + 1.0e-9
            >= minimum_physical_lead_m
        ):
            # The calibrated margin is desired robustness, while the
            # pre-margin compliance solution is the physical lower bound.
            # Respect the existing deterministic collision checker by
            # continuously clipping the nominal joint displacement.  This is
            # one common rule for every morphology; no module-count switch is
            # introduced.
            virtual_contact_joint_delta = (
                virtual_contact_joint_delta * collision_safe_scale
            )
            virtual_contact_lead_m = nominal_preload_applied_inward_lead_m
        else:
            raise RuntimeError(
                "Order9 actuator-aware nominal preload collision limit falls "
                "below the physical compliance requirement: "
                f"scene={nominal_preload_collision_limit.limiting_scene_id!r}, "
                f"safe_scale={collision_safe_scale}, "
                f"applied_lead_m={nominal_preload_applied_inward_lead_m}, "
                f"minimum_physical_lead_m={minimum_physical_lead_m}"
            )
    if (
        training_nominal_preload_deficit_mm is not None
        and policy_runtime.builder.module_count
        == args_cli.training_nominal_preload_deficit_module_count
    ):
        if (
            actuator_aware_nominal_preload_solution is None
            or virtual_contact_joint_delta is None
            or virtual_contact_joint_delta.ndim != 2
        ):
            raise RuntimeError(
                "training preload deficits require one actuator-aware nominal "
                "compression solution"
            )
        training_nominal_preload_deficit_plan = (
            plan_order9_training_nominal_preload_deficit(
                nominal_lead_m=virtual_contact_lead_m,
                requested_deficit_mm=training_nominal_preload_deficit_mm,
                environment_count=scene.num_envs,
                seed=args_cli.seed,
            )
        )
        deficit_scale = torch.tensor(
            training_nominal_preload_deficit_plan.scale_by_environment,
            device=scene.device,
            dtype=torch.float32,
        )
        virtual_contact_joint_delta = (
            deficit_scale[:, None, None] * virtual_contact_joint_delta[None, :, :]
        )
    if (
        c3_nominal_reference is not None
        and config.production_runtime.c3_contact_compression_action_adapter_enabled
        and args_cli.c3_action_contract
        != ORDER9_C3_ACTION_CONTRACT_CONTACT_SPACE_PROJECTED
    ):
        if accepted_nominal_bundle is None:  # pragma: no cover
            raise AssertionError("C3 contact action lacks its nominal bundle")
        contact_trajectory = accepted_nominal_bundle.phase_trajectories[
            Order9ObjectTaskPhase.CONTACT_ACQUISITION.value
        ]
        contact_compression_action_solution = solve_order9_virtual_contact_compression(
            morphology=morphology,
            physical_model=physical,
            contact_knot=contact_trajectory.knots[-1],
            candidate_set=accepted_nominal_bundle.contact_candidate_set,
            config=Order9VirtualContactCompressionConfig(
                inward_lead_m=contact_compression_action_span_m
            ),
        )
        contact_compression_action_joint_direction = compression_joint_delta_tensor(
            contact_compression_action_solution,
            module_ids=policy_runtime.builder.module_ids,
            local_joint_ids=policy_runtime.decoder.local_joint_ids,
            device=scene.device,
            dtype=torch.float32,
        )
        collision_object = build_order9_c3_posture_collision_object(task)
        collision_resolver = Order9PostureTrajectoryResolver(
            physical,
            config=Order9PostureResolverConfig(collision_margin_m=0.001),
            collision_object=collision_object,
            prefer_native_solver=True,
            require_native_solver=True,
        )
        contact_compression_collision_limit = (
            limit_order9_virtual_contact_compression_by_collision(
                morphology=morphology,
                physical_model=physical,
                contact_knot=contact_trajectory.knots[-1],
                solution=contact_compression_action_solution,
                collision_object=collision_object,
                collision_solver=collision_resolver.ik_solver,
                collision_margin_m=0.001,
                maximum_action_joint_delta_rad=(
                    1.0e-3
                    * float(
                        args_cli.contact_compression_maximum_joint_delta_mrad
                    )
                    if args_cli.contact_compression_maximum_joint_delta_mrad
                    is not None
                    else ORDER9_CONTACT_COMPRESSION_MAXIMUM_ACTION_JOINT_DELTA_RAD
                ),
            )
        )
        flattened_control_index = int(
            contact_compression_action_joint_direction.abs().reshape(-1).argmax().item()
        )
        control_module_index, control_joint_index = divmod(
            flattened_control_index,
            len(policy_runtime.decoder.local_joint_ids),
        )
        contact_compression_action_control_joint_id = (
            f"module_{policy_runtime.builder.module_ids[control_module_index]}:"
            f"{policy_runtime.decoder.local_joint_ids[control_joint_index]}"
        )
    task_runtime_config = Order9ObjectTaskRuntimeConfig()
    if c3_nominal_reference is not None:
        for phase, duration_s in c3_nominal_reference.phase_durations_s.items():
            task_runtime_config.phase_duration_s[phase] = max(
                float(task_runtime_config.phase_duration_s[phase]),
                float(duration_s) + 5.0,
            )
    task_runtime = Order9TensorObjectTaskRuntime(task_runtime_config)
    teacher_reference = None
    if stage.stage_id == "c1_pi_l_bc_fixed_nominal":
        dataset_manifest_sha256 = checkpoint.metadata.input_artifact_hashes.get(
            "dataset_manifest"
        )
        if dataset_manifest_sha256 is None:
            raise ValueError("C1 checkpoint does not bind its C0 dataset manifest")
        teacher_reference = load_order9_nominal_tensor_teacher_reference(
            repository / args_cli.teacher_dataset_manifest,
            expected_dataset_manifest_sha256=dataset_manifest_sha256,
            repository_root=repository,
            module_ids=policy_runtime.builder.module_ids,
            joint_ids=policy_runtime.decoder.local_joint_ids,
            device=scene.device,
            dtype=torch.float32,
        )
        if (
            teacher_reference.provenance["source_graph_hash"]
            != morphology.stable_hash()
        ):
            raise ValueError("C1 teacher reference morphology hash differs")
    canonical_resets = _canonical_resets_enabled(
        morphology, canonical.metadata["source_graph_hash"]
    )
    scalar_task_runtime = Order9ObjectTaskRuntime(canonical, config=task_runtime.config)
    joint_reference_start, joint_reference_end = _canonical_joint_reference_banks(
        scalar_task_runtime,
        module_ids=policy_runtime.builder.module_ids,
        joint_ids=policy_runtime.decoder.local_joint_ids,
        device=torch.device(scene.device),
        dtype=torch.float32,
        articulated_trajectory=(teacher_trajectory if not canonical_resets else None),
    )
    reward_command_joint_indices = torch.tensor(
        [
            io.local_joint_ids.index(joint_id)
            for joint_id in policy_runtime.decoder.local_joint_ids
        ],
        device=scene.device,
        dtype=torch.long,
    )
    contact_phase_index = ORDER9_OBJECT_TASK_PHASES.index(
        Order9ObjectTaskPhase.CONTACT_ACQUISITION
    )
    # Compatibility-only reporting handle.  The former force-feedback preload
    # controller is retired; virtual nominal compression plus pi_L now own the
    # contact command.
    contact_preload = None
    command_isaac_joint_indices = torch.tensor(
        [
            [
                io.local_joint_indices[module_index][io.local_joint_ids.index(joint_id)]
                for joint_id in policy_runtime.decoder.local_joint_ids
            ]
            for module_index, _ in enumerate(policy_runtime.builder.module_ids)
        ],
        device=scene.device,
        dtype=torch.long,
    )
    if bool((command_isaac_joint_indices < 0).any()):
        raise RuntimeError("Order9 commanded Dock joint is absent from Isaac")
    closure_direction = (
        joint_reference_end[contact_phase_index]
        - joint_reference_start[contact_phase_index]
    )
    deployable_anchor_joint_owner_mask = _deployable_anchor_joint_owner_mask(
        morphology=morphology,
        physical_model=physical,
        selected_anchor_ids=selected_anchor_ids,
        module_ids=policy_runtime.builder.module_ids,
        local_joint_ids=policy_runtime.decoder.local_joint_ids,
        contact_joint_positions_rad=joint_reference_end[contact_phase_index],
        closure_direction_rad=(
            closure_direction
            if virtual_contact_joint_delta is None
            else closure_direction
            + (
                virtual_contact_joint_delta.mean(dim=0)
                if virtual_contact_joint_delta.ndim == 3
                else virtual_contact_joint_delta
            )
        ),
    ).to(device=scene.device)
    flattened_anchor_joint_owner_mask = deployable_anchor_joint_owner_mask.flatten(
        start_dim=1
    )
    selected_anchor_body_indices_tensor = torch.tensor(
        io.selected_anchor_body_indices,
        device=scene.device,
        dtype=torch.long,
    )
    jacobian_joint_columns = command_isaac_joint_indices.flatten() + int(
        robot.num_base_dofs
    )
    force_estimator_config = Order9AnchorNormalForceEstimatorConfig(
        baseline_update_alpha=float(
            config.production_runtime.c3_force_estimator_baseline_update_alpha
        ),
        force_filter_alpha=float(
            config.production_runtime.c3_force_estimator_filter_alpha
        ),
        ridge_damping_m2=float(
            config.production_runtime.c3_force_estimator_ridge_damping_m2
        ),
        minimum_normal_jacobian_norm_m=float(
            config.production_runtime.c3_force_estimator_minimum_jacobian_norm_m
        ),
        fit_residual_scale_nm=float(
            config.production_runtime.c3_force_estimator_fit_residual_scale_nm
        ),
    )
    anchor_force_estimator = Order9AnchorNormalForceEstimator(force_estimator_config)
    reward_engine = Order9TensorRewardEngine(
        reward_config=config.reward,
        gate_config=Order9TensorRewardGateConfig(
            wrench_range_gate_scale=float(args_cli.c3_wrench_gate_range_scale)
        ),
        control_dt_s=float(args_cli.dt),
    )
    deployable_phase_gate = Order9DeployablePhaseGate(
        config=Order9DeployablePhaseGateConfig(
            contact_force_estimator_confidence_threshold=float(
                config.production_runtime.c3_force_estimator_minimum_confidence
            )
        ),
        control_dt_s=float(args_cli.dt),
    )
    estimated_mass = torch.full(
        (scene.num_envs,),
        float(args_cli.estimated_mass_kg or object_mass),
        device=scene.device,
        dtype=torch.float32,
    )
    estimated_inertia = (
        torch.tensor(
            args_cli.estimated_inertia_body or target_object.inertia_kgm2,
            device=scene.device,
            dtype=torch.float32,
        )
        .reshape(1, 6)
        .expand(scene.num_envs, -1)
    )
    estimated_com = (
        torch.tensor(
            args_cli.estimated_com_object
            or target_object.center_of_mass_object
            or (0.0, 0.0, 0.0),
            device=scene.device,
            dtype=torch.float32,
        )
        .reshape(1, 3)
        .expand(scene.num_envs, -1)
    )
    lift_clearance = torch.full(
        (scene.num_envs,),
        float(canonical.lift_clearance_m),
        device=scene.device,
        dtype=torch.float32,
    )
    transport_distance = torch.full(
        (scene.num_envs,),
        _transport_distance(task, target_object.object_id),
        device=scene.device,
        dtype=torch.float32,
    )
    selected_mask = torch.ones(
        (scene.num_envs, len(assignments)),
        device=scene.device,
        dtype=torch.bool,
    )
    duration_values = torch.tensor(
        [
            task_runtime.config.phase_duration_s[phase.value]
            for phase in ORDER9_OBJECT_TASK_PHASES
        ],
        device=scene.device,
        dtype=torch.float32,
    )

    def _apply_virtual_contact_target(current_target):
        if virtual_contact_joint_delta is None:
            return current_target
        return apply_order9_virtual_contact_compression(
            current_target,
            phase_index=phase_index,
            phase_progress=current_target.phase_progress,
            joint_delta_rad=virtual_contact_joint_delta,
            phase_duration_s=duration_values.index_select(0, phase_index),
        )

    phase_count = len(ORDER9_OBJECT_TASK_PHASES)
    if c3_nominal_reference is not None:
        reset_strata_by_phase = order9_c3_reset_strata_by_phase(
            phase_labels=tuple(phase.value for phase in ORDER9_OBJECT_TASK_PHASES),
            progress_fractions=(
                config.production_runtime.c3_phase_reset_progress_fractions
            ),
            boundary_tail_phase_labels=(
                config.production_runtime.c3_boundary_tail_phase_labels
            ),
            boundary_tail_progress_fraction=(
                config.production_runtime.c3_boundary_tail_progress_fraction
            ),
        )
    else:
        reset_strata_by_phase = tuple((0,) for _ in range(phase_count))
    bank = _PhaseStateBank(
        scene,
        phase_count,
        stratum_count=(
            len(config.production_runtime.c3_phase_reset_progress_fractions)
            if c3_nominal_reference is not None
            else 1
        ),
        reset_stratum_indices_by_phase=reset_strata_by_phase,
    )
    state_inheritance_fixed_progress = (
        config.production_runtime.c3_state_inheritance_fixed_reset_progress_fraction
        if state_inheritance_shard
        else None
    )
    state_inheritance_fixed_stratum_index = (
        order9_c3_fixed_reset_stratum_index(
            progress_fractions=(
                config.production_runtime.c3_phase_reset_progress_fractions
            ),
            reset_progress_fraction=float(state_inheritance_fixed_progress),
        )
        if state_inheritance_fixed_progress is not None
        else None
    )
    state_inheritance_stratum_policy = (
        "fixed_progress_fraction_on_every_episode"
        if state_inheritance_fixed_stratum_index is not None
        else "latest_selectable_then_reverse_cycle_on_terminal"
    )
    teacher_phase_zero_expected = None
    c3_phase_reset_expected = None
    if canonical_resets:
        _seed_canonical_bank(
            bank,
            scene,
            canonical=canonical,
            task=task,
            robot_joint_names=tuple(robot.joint_names),
        )
        if teacher_reference is not None:
            teacher_phase_zero_expected = _install_teacher_phase_zero(
                bank,
                scene,
                io=io,
                teacher_reference=teacher_reference,
                robot_joint_names=tuple(robot.joint_names),
                task_object_pose_world=tuple(target_object.pose_world),
            )
    else:
        if c3_nominal_reference is not None:
            c3_phase_reset_expected = _install_c3_nominal_phase_bank(
                bank,
                scene,
                sim=sim,
                io=io,
                qpid_runtime=policy_runtime,
                nominal_reference=c3_nominal_reference,
                object_id=target_object.object_id,
                task=task,
                lift_clearance_m=float(canonical.lift_clearance_m),
                retreat_offset_m=float(task_runtime.config.retreat_offset_m),
                phase_duration_s=task_runtime.config.phase_duration_s,
                progress_fractions=(
                    config.production_runtime.c3_phase_reset_progress_fractions
                ),
            )
        else:
            _align_arbitrary_phase_zero(
                scene,
                sim=sim,
                io=io,
                candidate_points=[
                    next(
                        candidate.contact_pose_world
                        for candidate in candidates.candidates
                        if candidate.candidate_id == assignment.candidate_id
                    )
                    for assignment in assignments
                ],
                approach_offset_m=(Order9ObjectTaskRuntimeConfig().approach_offset_m),
            )
            all_ids = torch.arange(scene.num_envs, device=scene.device)
            bank.capture(
                scene,
                env_ids=all_ids,
                phase_indices=torch.zeros_like(all_ids),
            )
    phase_specific_resets_available = bool(
        canonical_resets or c3_phase_reset_expected is not None
    )
    c3_reset_stabilization = None
    if c3_phase_reset_expected is not None:
        reset_bank_path = Path(args_cli.c3_reset_bank).resolve()
        reset_contract = {
            "version": _C3_RESET_BANK_VERSION,
            "morphology_graph_hash": morphology.stable_hash(),
            "physical_model_hash": physical.stable_hash(),
            "robot_usd_sha256": hash_file(robot_usd),
            "task_spec_hash": stable_hash(task.to_dict()),
            "c3_nominal_reference": dict(c3_nominal_reference.provenance),
            "selected_anchor_ids": list(io.selected_anchor_ids),
            "selected_gripper_friction": float(selected_friction),
            "contact_stiffness_n_per_m": float(args_cli.contact_stiffness),
            "contact_damping_n_s_per_m": float(args_cli.contact_damping),
            "estimated_payload_mass_kg": float(estimated_mass[0].item()),
            "estimated_payload_inertia_body": [
                float(value) for value in estimated_inertia[0].tolist()
            ],
            "estimated_payload_com_object": [
                float(value) for value in estimated_com[0].tolist()
            ],
            "control_dt_s": float(args_cli.dt),
            "phase_reset_progress_fractions": list(
                config.production_runtime.c3_phase_reset_progress_fractions
            ),
            "boundary_tail_sampling_contract": (
                ORDER9_C3_BOUNDARY_TAIL_SAMPLING_CONTRACT
            ),
            "boundary_tail_phase_labels": list(
                config.production_runtime.c3_boundary_tail_phase_labels
            ),
            "boundary_tail_progress_fraction": float(
                config.production_runtime.c3_boundary_tail_progress_fraction
            ),
            "reset_stratum_indices_by_phase": [
                list(indices) for indices in reset_strata_by_phase
            ],
            "reset_source": "hash_bound_human_accepted_nominal_trajectory",
            "dynamic_stability_gate_required": False,
        }
        if reset_bank_path.is_file():
            c3_reset_stabilization = _load_c3_reset_bank(
                reset_bank_path,
                bank=bank,
                reset_references=c3_phase_reset_expected,
                contract=reset_contract,
            )
        else:
            c3_reset_stabilization = _accepted_nominal_reset_bank_evidence(
                bank,
                c3_phase_reset_expected,
            )
            reset_bank_sha256 = _write_c3_reset_bank(
                reset_bank_path,
                bank=bank,
                reset_references=c3_phase_reset_expected,
                contract=reset_contract,
                stabilization=c3_reset_stabilization,
            )
            c3_reset_stabilization = {
                **c3_reset_stabilization,
                "loaded_from_reset_bank": False,
                "reset_bank_path": str(reset_bank_path),
                "reset_bank_sha256": reset_bank_sha256,
            }
    all_ids = torch.arange(scene.num_envs, device=scene.device, dtype=torch.long)
    initial_phase_values = (
        [phase for phase, _ in diagnostic_phase_strata]
        if diagnostic_phase_strata is not None
        else (
            order9_c3_state_inheritance_phase_indices(
                range(scene.num_envs),
                [0] * scene.num_envs,
                config.production_runtime.c3_state_inheritance_initial_phase_indices,
            )
            if state_inheritance_shard
            else order9_rollout_initial_phase_indices(
                environment_count=scene.num_envs,
                evaluation_mode=evaluation_mode,
                # The runtime API retains its historical keyword, but C3 now
                # supplies equally authoritative morphology-specific resets.
                canonical_resets=phase_specific_resets_available,
                diagnostic_initial_phase_index=(
                    args_cli.diagnostic_initial_phase_index
                ),
                allow_evaluation_diagnostic_phase=bool(
                    args_cli.diagnostic_nominal_qpid_only
                ),
            )
        )
    )
    phase_index = torch.tensor(
        initial_phase_values,
        device=scene.device,
        dtype=torch.long,
    )
    initial_phase_zero = bool((phase_index == 0).all())
    reset_stratum = (
        torch.tensor(
            [stratum for _, stratum in diagnostic_phase_strata],
            device=scene.device,
            dtype=torch.long,
        )
        if diagnostic_phase_strata is not None
        else (
            torch.full_like(
                all_ids,
                int(args_cli.diagnostic_initial_phase_stratum_index),
            )
            if args_cli.diagnostic_initial_phase_stratum_index is not None
            else (
                bank.state_inheritance_strata(
                    all_ids,
                    phase_index,
                    torch.zeros_like(all_ids),
                    fixed_stratum_index=(
                        state_inheritance_fixed_stratum_index
                    ),
                )
                if state_inheritance_shard
                else bank.initial_strata(
                    all_ids,
                    phase_index,
                    formal_phase_zero_start=args_cli.formal_phase_zero_start,
                )
            )
        )
    )
    formal_phase_zero_reference = None
    if args_cli.formal_phase_zero_start:
        if c3_nominal_reference is None or c3_phase_reset_expected is None:
            raise RuntimeError(
                "formal C3 phase-zero start requires an accepted nominal reference"
            )
        (
            reset_elapsed,
            reset_phase_body_start,
            reset_phase_object_start,
            formal_phase_zero_reference,
        ) = _restore_c3_formal_phase_zero(
            scene,
            sim=sim,
            io=io,
            qpid_runtime=policy_runtime,
            nominal_reference=c3_nominal_reference,
        )
    else:
        (
            reset_elapsed,
            reset_phase_body_start,
            reset_phase_object_start,
        ) = bank.restore(
            scene,
            env_ids=all_ids,
            phase_indices=phase_index,
            stratum_indices=reset_stratum,
        )
    sim.forward()
    scene.update(0.0)
    state = io.gather_state(robot=robot, object_asset=obj)
    if teacher_phase_zero_expected is not None:
        _validate_teacher_phase_zero_alignment(
            state,
            expected=teacher_phase_zero_expected,
        )
    control = policy_runtime.builder.build(
        module_pose_world=state.module_pose_world,
        module_twist_world=state.module_twist_world,
        local_joint_positions_rad=state.local_joint_positions_rad,
    )
    if formal_phase_zero_reference is not None:
        _validate_c3_formal_phase_zero_alignment(
            state,
            control=control,
            expected=formal_phase_zero_reference,
            scene_origins=scene.env_origins,
            io=io,
            reference_joint_ids=c3_nominal_reference.joint_ids,
        )
    elif c3_phase_reset_expected is not None:
        _validate_c3_phase_reset_alignment(
            state,
            control=control,
            expected=c3_phase_reset_expected,
            phase_index=phase_index,
            stratum_index=reset_stratum,
            scene_origins=scene.env_origins,
            io=io,
            reference_joint_ids=c3_nominal_reference.joint_ids,
        )
    phase_start_body_pose = (
        reset_phase_body_start
        if c3_phase_reset_expected is not None
        else control.body_pose_world.clone()
    )
    phase_start_object_pose = (
        reset_phase_object_start
        if c3_phase_reset_expected is not None
        else state.object_pose_world.clone()
    )
    phase_elapsed = (
        reset_elapsed.to(device=scene.device, dtype=torch.float32)
        if c3_phase_reset_expected is not None
        else torch.zeros(scene.num_envs, device=scene.device, dtype=torch.float32)
    )
    if args_cli.diagnostic_initial_phase_progress is not None:
        if c3_nominal_reference is None:
            raise ValueError("diagnostic phase progress requires C3 nominal replay")
        phase_label = ORDER9_OBJECT_TASK_PHASES[
            int(args_cli.diagnostic_initial_phase_index)
        ].value
        phase_elapsed.fill_(
            float(args_cli.diagnostic_initial_phase_progress)
            * float(c3_nominal_reference.phase_durations_s[phase_label])
        )
    episode_time = torch.zeros_like(phase_elapsed)
    episode_serial = torch.zeros(scene.num_envs, device=scene.device, dtype=torch.long)
    episode_step = torch.zeros_like(episode_serial)
    task_object_position_world = torch.tensor(
        target_object.pose_world[:3],
        device=scene.device,
        dtype=phase_elapsed.dtype,
    )
    target = task_runtime.target(
        phase_index=phase_index,
        phase_elapsed_s=phase_elapsed,
        reset_robot_root_pose_world=phase_start_body_pose,
        reset_object_pose_world=phase_start_object_pose,
        reset_joint_positions_rad=joint_reference_start.index_select(0, phase_index),
        phase_end_joint_positions_rad=joint_reference_end.index_select(0, phase_index),
        lift_clearance_m=lift_clearance,
        transport_distance_m=transport_distance,
    )
    target = _condition_target_on_teacher_reference(
        target,
        teacher_reference=teacher_reference,
        phase_index=phase_index,
        scene_origins=scene.env_origins,
        task_object_position_world=task_object_position_world,
    )
    if c3_nominal_reference is not None:
        target = c3_nominal_reference.condition(
            target,
            phase_index=phase_index,
            phase_elapsed_s=phase_elapsed,
            scene_origins=scene.env_origins,
        )
        target = _apply_virtual_contact_target(target)
    reward_state = reward_engine.initial_state(
        object_pose_world=state.object_pose_world,
        desired_object_pose_world=target.phase_goal_object_pose_world,
    )
    deployable_gate_state = deployable_phase_gate.initial_state(
        batch_size=scene.num_envs, device=scene.device
    )
    force_estimator_state = anchor_force_estimator.initial_state(
        batch_size=scene.num_envs,
        anchor_count=len(selected_anchor_ids),
        joint_count=int(command_isaac_joint_indices.numel()),
        device=scene.device,
        dtype=torch.float32,
    )
    release_complete_reset_ids = torch.nonzero(
        phase_index >= ORDER9_OBJECT_TASK_PHASES.index(Order9ObjectTaskPhase.RETREAT),
        as_tuple=False,
    ).flatten()
    if release_complete_reset_ids.numel():
        deployable_gate_state = deployable_phase_gate.assume_release_complete_subset(
            deployable_gate_state, release_complete_reset_ids
        )
    support_top = torch.full_like(
        phase_elapsed, float(config.randomization.support_top_z_m)
    )
    half_height = torch.full_like(phase_elapsed, float(object_half_height))
    translated_tasks = [
        _translated_task(task, scene.env_origins[index], index)
        for index in range(scene.num_envs)
    ]
    split = DatasetSplit(args_cli.split)
    reward_names = ORDER9_TENSOR_REWARD_TERM_NAMES
    buffer = Order9TensorRolloutBuffer(
        _rollout_metadata(
            config=config,
            stage=stage,
            morphology=morphology,
            physical=physical,
            checkpoint_sha256=checkpoint.sha256,
            policy_version=checkpoint.metadata.policy_version,
            tasks=translated_tasks,
            split=split,
            assignments=assignments,
            teacher_trajectory=teacher_trajectory,
            io=io,
            reward_names=reward_names,
            selected_friction=selected_friction,
            canonical_resets=canonical_resets,
            phase_specific_resets_available=(phase_specific_resets_available),
            robot_usd=robot_usd,
            robot_asset_manifest=robot_asset_manifest,
            estimated_mass_kg=float(estimated_mass[0].item()),
            estimated_inertia_body=tuple(
                float(value) for value in estimated_inertia[0].tolist()
            ),
            estimated_com_object=tuple(
                float(value) for value in estimated_com[0].tolist()
            ),
            object_mass_properties_readback=object_mass_properties_readback,
            actuator_readback=actuator_readback,
            teacher_reference=teacher_reference,
            c3_nominal_reference=c3_nominal_reference,
            contact_space_action_basis=contact_space_action_basis,
            c3_reset_stabilization=c3_reset_stabilization,
            reset_strata_by_phase=reset_strata_by_phase,
            phase_duration_s=task_runtime.config.phase_duration_s,
            deterministic_policy=deterministic_policy,
            initial_phase_zero=initial_phase_zero,
            diagnostic_initial_phase_index=(args_cli.diagnostic_initial_phase_index),
            diagnostic_phase_strata=diagnostic_phase_strata,
        )
    )
    tensorboard_logger = None
    tensorboard_update_index = None
    tensorboard_log_dir = None
    if not evaluation_mode and not args_cli.no_tensorboard:
        tensorboard_update_index = _next_ppo_update_index(checkpoint.metadata.metadata)
        tensorboard_log_dir = (
            _tensorboard_log_dir(
                repository,
                artifact_root=config.production_runtime.artifact_root,
                stage_id=stage.stage_id,
                override=args_cli.tensorboard_log_dir,
            )
            / split.value
        )
        generation_environment_steps = configured_runtime.generation_environment_steps
        if generation_environment_steps is None:
            raise ValueError("Order9 TensorBoard PPO generation size is missing")
        tensorboard_logger = Order9TensorBoardLogger(
            tensorboard_log_dir,
            stage_id=stage.stage_id,
            generation_id=args_cli.generation_id,
            split=split.value,
            update_index=tensorboard_update_index,
            generation_environment_steps=generation_environment_steps,
            phase_labels=tuple(phase.value for phase in ORDER9_OBJECT_TASK_PHASES),
            reward_term_names=reward_names,
        )
    setup_elapsed = time.perf_counter() - setup_started
    rollout_started = time.perf_counter()
    terminal_count = 0
    success_count = 0
    evaluation_active = torch.ones(
        scene.num_envs, device=scene.device, dtype=torch.bool
    )
    evaluation_step_count = torch.zeros(
        scene.num_envs, device=scene.device, dtype=torch.long
    )
    evaluation_return = torch.zeros_like(phase_elapsed)
    evaluation_outcomes: list[dict[str, object]] = []
    phase_transition_counts = {
        (
            f"{ORDER9_OBJECT_TASK_PHASES[index].value}->"
            f"{ORDER9_OBJECT_TASK_PHASES[index + 1].value}"
        ): 0
        for index in range(phase_count - 1)
    }
    diagnostic_collision_reported = torch.zeros(
        scene.num_envs, device=scene.device, dtype=torch.bool
    )
    preload_signed_normal_force_n = torch.zeros(
        (scene.num_envs, len(selected_anchor_ids)),
        device=scene.device,
        dtype=torch.float32,
    )
    preload_required_normal_force_n = torch.zeros_like(preload_signed_normal_force_n)
    command_joint_shape = (
        scene.num_envs,
        len(policy_runtime.builder.module_ids),
        len(policy_runtime.decoder.local_joint_ids),
    )
    diagnostic_contact_sweep = args_cli.diagnostic_contact_sweep_json is not None
    diagnostic_sample_count = 0
    diagnostic_force_sum = torch.zeros(
        len(selected_anchor_ids), device=scene.device, dtype=torch.float64
    )
    diagnostic_force_max = torch.zeros_like(diagnostic_force_sum)
    diagnostic_load_sum = torch.zeros((), device=scene.device, dtype=torch.float64)
    diagnostic_load_max = torch.zeros((), device=scene.device, dtype=torch.float64)
    diagnostic_penetration_sum = torch.zeros(
        len(selected_anchor_ids), device=scene.device, dtype=torch.float64
    )
    diagnostic_penetration_max = torch.zeros_like(diagnostic_penetration_sum)
    diagnostic_signed_distance_sum = torch.zeros_like(diagnostic_penetration_sum)
    diagnostic_joint_tracking_error_sum = torch.zeros(
        (), device=scene.device, dtype=torch.float64
    )
    diagnostic_joint_tracking_error_max = torch.zeros_like(
        diagnostic_joint_tracking_error_sum
    )
    force_observer_trace_steps = (
        {} if args_cli.diagnostic_force_observer_trace is not None else None
    )
    selected_anchor_local_pose = torch.tensor(
        [
            next(
                anchor.local_pose
                for anchor in morphology.robot_anchors
                if anchor.anchor_id == anchor_id
            )
            for anchor_id in selected_anchor_ids
        ],
        device=scene.device,
        dtype=torch.float32,
    )
    target_motor_load_proxy = torch.zeros(
        len(selected_anchor_ids), device=scene.device, dtype=torch.float32
    )
    if (
        contact_space_action_basis is not None
        and actuator_aware_nominal_preload_solution is not None
    ):
        ordered_joint_ids = ordered_global_dock_joint_ids(morphology, physical)
        predicted_by_joint_id = dict(
            zip(
                ordered_joint_ids,
                actuator_aware_nominal_preload_solution.predicted_joint_torque_nm,
                strict=True,
            )
        )
        target_joint_load = torch.tensor(
            [
                predicted_by_joint_id[f"module_{module_id}:{joint_id}"]
                for module_id in policy_runtime.builder.module_ids
                for joint_id in policy_runtime.decoder.local_joint_ids
            ],
            device=scene.device,
            dtype=torch.float32,
        ).reshape(1, command_joint_shape[1], command_joint_shape[2])
        target_feedback = build_order9_deployable_contact_feedback_features(
            signed_surface_distance_m=torch.zeros(
                (1, len(selected_anchor_ids)),
                device=scene.device,
                dtype=torch.float32,
            ),
            relative_twist_contact=torch.zeros(
                (1, len(selected_anchor_ids), 6),
                device=scene.device,
                dtype=torch.float32,
            ),
            signed_hardware_joint_load_nm=target_joint_load,
            basis=contact_space_action_basis,
        )
        target_motor_load_proxy = torch.relu(
            target_feedback[0, : len(selected_anchor_ids), 7]
        )
    for rollout_index in range(int(args_cli.rollout_steps)):
        pre_state = state
        pre_target = target
        pre_phase = phase_index.clone()
        pre_actor_phase = _actor_phase_indices(pre_phase)
        pre_elapsed = phase_elapsed.clone()
        pre_time = episode_time.clone()
        pre_serial = episode_serial.clone()
        pre_step = episode_step.clone()
        payload_scale = order9_payload_feedforward_scale(
            pre_phase,
            pre_target.phase_progress,
        )
        payload_active = payload_scale > 0.0
        scaled_estimated_mass = estimated_mass * payload_scale
        scaled_estimated_inertia = estimated_inertia * payload_scale.unsqueeze(-1)
        if args_cli.diagnostic_nominal_qpid_only:
            # This diagnostic proves the accepted nominal trajectory under
            # the production QPID/QP.  Evaluating pi_L first and immediately
            # discarding its command duplicated the rigid-body build and QPID
            # solve on every simulator step without changing any applied
            # command or acceptance decision.
            nominal_step = policy_runtime.compute_nominal_qpid_hold(
                task_target=pre_target,
                state=pre_state,
                estimated_payload_mass_kg=scaled_estimated_mass,
                estimated_payload_inertia_body=scaled_estimated_inertia,
                payload_active=payload_active,
                estimated_payload_com_object=estimated_com,
            )
            policy_step = _nominal_diagnostic_artifact_step(
                nominal_step=nominal_step,
                policy_runtime=policy_runtime,
                module_count=int(pre_state.module_pose_world.shape[1]),
            )
        else:
            # Applied motor effort/current is deployable proprioception.  Read
            # the previous control interval before actor evaluation; raw
            # PhysX contact force is intentionally absent from this actor path.
            actor_hardware_joint_load_nm = (
                _torch(robot.data.applied_torque)
                .index_select(1, command_isaac_joint_indices.flatten())
                .reshape(command_joint_shape)
            )
            contact_feedback_features = None
            if (
                checkpoint.metadata.policy_version
                in {
                    ORDER9_CONTACT_FEEDBACK_PI_L_POLICY_VERSION,
                    ORDER9_CATEGORICAL_CONTACT_NORMAL_PI_L_POLICY_VERSION,
                }
            ):
                pre_wrench_range = wrench_range_reference.resolve(
                    contact_schedule_index=pre_target.contact_schedule_index,
                    object_pose_world=pre_state.object_pose_world,
                )
                pre_selected_body_pose = _torch(
                    robot.data.body_pose_w
                ).index_select(1, selected_anchor_body_indices_tensor)
                pre_selected_body_twist = torch.cat(
                    (
                        _torch(robot.data.body_lin_vel_w).index_select(
                            1, selected_anchor_body_indices_tensor
                        ),
                        _torch(robot.data.body_ang_vel_w).index_select(
                            1, selected_anchor_body_indices_tensor
                        ),
                    ),
                    dim=-1,
                )
                (
                    feedback_surface_distance_m,
                    feedback_relative_twist_contact,
                ) = _deployable_selected_contact_feedback_inputs(
                    selected_body_pose_world=pre_selected_body_pose,
                    selected_body_twist_world=pre_selected_body_twist,
                    selected_anchor_local_pose=selected_anchor_local_pose,
                    object_pose_world=pre_state.object_pose_world,
                    object_twist_world=pre_state.object_twist_world,
                    contact_frame_pose_world=(
                        pre_wrench_range.contact_frame_pose_world
                    ),
                    contact_normal_world=pre_wrench_range.contact_normal_world,
                )
                contact_feedback_features = (
                    build_order9_deployable_contact_feedback_features(
                        signed_surface_distance_m=(
                            feedback_surface_distance_m
                        ),
                        relative_twist_contact=(
                            feedback_relative_twist_contact
                        ),
                        signed_hardware_joint_load_nm=(
                            actor_hardware_joint_load_nm
                        ),
                        basis=contact_space_action_basis,
                    )
                )
            policy_step = policy_runtime.compute(
                time_s=pre_time,
                phase_index=pre_actor_phase,
                task_target=pre_target,
                state=pre_state,
                estimated_payload_mass_kg=scaled_estimated_mass,
                estimated_payload_inertia_body=scaled_estimated_inertia,
                payload_active=payload_active,
                estimated_payload_com_object=estimated_com,
                hardware_joint_load_nm=actor_hardware_joint_load_nm,
                contact_feedback_features=contact_feedback_features,
                contact_compression_joint_direction_rad=(
                    contact_compression_action_joint_direction
                ),
                contact_compression_action_limit=(
                    None
                    if contact_compression_collision_limit is None
                    else float(
                        contact_compression_collision_limit.maximum_action_scale
                    )
                ),
                contact_compression_only_action=(
                    args_cli.diagnostic_action_ablation is None
                    and args_cli.c3_action_contract is None
                    and policy_runtime.builder.module_count
                    in config.production_runtime.c3_contact_compression_only_module_counts
                ),
                joint_only_action=(
                    args_cli.diagnostic_action_ablation is None
                    and args_cli.c3_action_contract is None
                    and policy_runtime.builder.module_count
                    in config.production_runtime.c3_joint_only_module_counts
                ),
                nominal_only_action_mask=(
                    None
                    if (
                        c3_nominal_reference is None
                        or args_cli.c3_action_contract is not None
                    )
                    else (
                        pre_phase
                        >= ORDER9_OBJECT_TASK_PHASES.index(
                            Order9ObjectTaskPhase.RELEASE
                        )
                    )
                    & (
                        pre_phase
                        <= ORDER9_OBJECT_TASK_PHASES.index(
                            Order9ObjectTaskPhase.RETREAT
                        )
                    )
                ),
                deterministic=deterministic_policy,
                diagnostic_action_ablation=(args_cli.diagnostic_action_ablation),
                diagnostic_contact_normal_residual_m=(
                    diagnostic_contact_normal_residual_m_by_environment
                ),
                c3_action_contract=args_cli.c3_action_contract,
            )
        io.apply(
            robot=robot,
            policy_command=policy_step.policy_command,
            controller_result=policy_step.controller_result,
        )
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(float(args_cli.dt))
        state = io.gather_state(robot=robot, object_asset=obj)
        post_control = policy_runtime.builder.build(
            module_pose_world=state.module_pose_world,
            module_twist_world=state.module_twist_world,
            local_joint_positions_rad=state.local_joint_positions_rad,
        )
        robot_net = _torch(robot_sensor.data.net_forces_w).index_select(
            1, robot_sensor_reorder
        )
        object_matrix = _torch(object_sensor.data.force_matrix_w)
        allow_contact = (
            (pre_target.contact_schedule_index == ORDER9_CONTACT_SCHEDULE_ATTACH)
            | (pre_target.contact_schedule_index == ORDER9_CONTACT_SCHEDULE_MAINTAIN)
            | (pre_target.contact_schedule_index == ORDER9_CONTACT_SCHEDULE_RELEASE)
        )
        contact = io.reduce_contacts(
            robot_net_contact_forces_world=robot_net,
            object_force_matrix_world=object_matrix,
            robot_body_linear_velocity_world=_torch(robot.data.body_lin_vel_w),
            robot_body_angular_velocity_world=_torch(robot.data.body_ang_vel_w),
            selected_assignment_mask=selected_mask,
            allow_selected_object_contact=allow_contact,
        )
        hardware_joint_load_nm = (
            _torch(robot.data.applied_torque)
            .index_select(1, command_isaac_joint_indices.flatten())
            .reshape(command_joint_shape)
            .abs()
        )
        if args_cli.diagnostic_collision_evidence:
            newly_observed = (
                contact.prohibited_collision & ~diagnostic_collision_reported
            )
            environment_forces = (
                robot_net + contact.object_contact_forces_by_robot_body_world
            )
            body_pose_world = _torch(robot.data.body_pose_w)
            selected_body_indices = set(io.selected_anchor_body_indices)
            for environment in (
                torch.nonzero(newly_observed, as_tuple=False).flatten().tolist()
            ):
                object_norm = torch.linalg.vector_norm(
                    contact.object_contact_forces_by_robot_body_world[environment],
                    dim=-1,
                )
                environment_norm = torch.linalg.vector_norm(
                    environment_forces[environment], dim=-1
                )
                active_body_indices = (
                    torch.nonzero(
                        (object_norm >= io.contact_force_threshold_n)
                        | (environment_norm >= io.contact_force_threshold_n),
                        as_tuple=False,
                    )
                    .flatten()
                    .tolist()
                )
                print(
                    "ORDER9_COLLISION_DIAGNOSTIC="
                    + json.dumps(
                        {
                            "rollout_index": rollout_index,
                            "environment_index": environment,
                            "runtime_phase_index": int(pre_phase[environment].item()),
                            "phase_elapsed_s": float(pre_elapsed[environment].item()),
                            "prohibited_object_contact": bool(
                                contact.prohibited_object_contact[environment].item()
                            ),
                            "prohibited_environment_contact": bool(
                                contact.prohibited_environment_contact[
                                    environment
                                ].item()
                            ),
                            "active_bodies": [
                                {
                                    "body_index": int(body_index),
                                    "body_name": io.robot_body_names[body_index],
                                    "selected_anchor_body": (
                                        body_index in selected_body_indices
                                    ),
                                    "object_force_norm_n": float(
                                        object_norm[body_index].item()
                                    ),
                                    "environment_force_norm_n": float(
                                        environment_norm[body_index].item()
                                    ),
                                    "body_pose_world": [
                                        float(value)
                                        for value in body_pose_world[
                                            environment, body_index
                                        ].tolist()
                                    ],
                                }
                                for body_index in active_body_indices
                            ],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            diagnostic_collision_reported |= newly_observed
        wrench_range = wrench_range_reference.resolve(
            contact_schedule_index=pre_target.contact_schedule_index,
            object_pose_world=state.object_pose_world,
        )
        selected_contact_wrenches_contact = _selected_contact_wrenches_contact(
            object_sensor,
            io=io,
            contact_frame_pose_world=wrench_range.contact_frame_pose_world,
            selected_assignment_mask=selected_mask,
            batch_size=scene.num_envs,
            filter_count=len(robot.body_names),
            dt_s=float(args_cli.dt),
        )
        wrench_bound_mask = wrench_range.bound_mask & selected_mask
        selected_body_pose = _torch(robot.data.body_pose_w).index_select(
            1,
            torch.tensor(
                io.selected_anchor_body_indices,
                device=scene.device,
                dtype=torch.long,
            ),
        )
        selected_body_twist = torch.cat(
            (
                _torch(robot.data.body_lin_vel_w).index_select(
                    1, selected_anchor_body_indices_tensor
                ),
                _torch(robot.data.body_ang_vel_w).index_select(
                    1, selected_anchor_body_indices_tensor
                ),
            ),
            dim=-1,
        )
        grasp_position = _selected_grasp_frame_positions(
            selected_body_pose, selected_anchor_local_pose
        )
        surface_position = wrench_range.contact_frame_pose_world[..., :3]
        current_surface_normal_world = wrench_range.contact_normal_world
        current_surface_normal_contact = order9_contact_normal_in_contact_frame(
            contact_normal_world=current_surface_normal_world,
            contact_frame_pose_world=(wrench_range.contact_frame_pose_world),
        )
        estimated_reaction_normal_world = order9_anchor_reaction_normal_world(
            contact_normal_world=current_surface_normal_world,
            contact_frame_pose_world=wrench_range.contact_frame_pose_world,
            wrench_lower_contact=wrench_range.lower_contact,
            wrench_upper_contact=wrench_range.upper_contact,
        )
        normal_lower, normal_upper = order9_projected_inward_wrench_interval(
            wrench_lower_contact=wrench_range.lower_contact,
            wrench_upper_contact=wrench_range.upper_contact,
            contact_normal_contact=current_surface_normal_contact,
        )
        (
            signed_surface_distance,
            relative_twist_contact,
        ) = _deployable_selected_contact_feedback_inputs(
            selected_body_pose_world=selected_body_pose,
            selected_body_twist_world=selected_body_twist,
            selected_anchor_local_pose=selected_anchor_local_pose,
            object_pose_world=state.object_pose_world,
            object_twist_world=state.object_twist_world,
            contact_frame_pose_world=wrench_range.contact_frame_pose_world,
            contact_normal_world=current_surface_normal_world,
        )
        selected_relative_normal_velocity_mps = (
            relative_twist_contact[..., :3] * current_surface_normal_contact
        ).sum(dim=-1)
        selected_relative_speed_mps = torch.linalg.vector_norm(
            contact.selected_link_twist_world[..., :3]
            - state.object_twist_world[:, None, :3],
            dim=-1,
        )
        current_command_q = state.local_joint_positions_rad.index_select(
            2, reward_command_joint_indices
        )
        grasp_point_jacobian_world = shift_link_origin_linear_jacobian_to_point(
            body_link_jacobian_world=_torch(
                robot.data.body_link_jacobian_w
            ).index_select(1, selected_anchor_body_indices_tensor),
            body_position_world=selected_body_pose[..., :3],
            point_position_world=grasp_position,
            joint_columns=jacobian_joint_columns,
        )
        signed_applied_joint_torque_nm = _torch(robot.data.applied_torque).index_select(
            1, command_isaac_joint_indices.flatten()
        )
        post_contact_feedback = (
            build_order9_deployable_contact_feedback_features(
                signed_surface_distance_m=signed_surface_distance,
                relative_twist_contact=relative_twist_contact,
                signed_hardware_joint_load_nm=(
                    signed_applied_joint_torque_nm.reshape(command_joint_shape)
                ),
                basis=contact_space_action_basis,
            )
            if contact_space_action_basis is not None
            else None
        )
        selected_motor_load_proxy = (
            torch.relu(
                post_contact_feedback[:, : len(selected_anchor_ids), 7]
            )
            if post_contact_feedback is not None
            else torch.zeros_like(signed_surface_distance)
        )
        gravity_joint_torque_nm = _torch(
            robot.data.gravity_compensation_forces
        ).index_select(1, jacobian_joint_columns)
        estimator_active = (
            pre_target.contact_schedule_index == ORDER9_CONTACT_SCHEDULE_ATTACH
        ) | (pre_target.contact_schedule_index == ORDER9_CONTACT_SCHEDULE_MAINTAIN)
        baseline_joint_update_mask = order9_branch_free_motion_baseline_update_mask(
            approach_mask=(
                pre_phase
                == ORDER9_OBJECT_TASK_PHASES.index(Order9ObjectTaskPhase.APPROACH)
            ),
            contact_acquisition_mask=(pre_phase == contact_phase_index),
            selected_surface_distance_m=signed_surface_distance,
            selected_anchor_mask=selected_mask,
            anchor_joint_owner_mask=flattened_anchor_joint_owner_mask,
            baseline_initialized=(force_estimator_state.baseline_initialized),
            contact_clearance_m=float(
                deployable_phase_gate.config.contact_surface_distance_tolerance_m
            ),
        )
        normal_force_estimate = anchor_force_estimator.step(
            applied_joint_torque_nm=signed_applied_joint_torque_nm,
            gravity_joint_torque_nm=gravity_joint_torque_nm,
            grasp_point_linear_jacobian_world=grasp_point_jacobian_world,
            contact_normal_world=estimated_reaction_normal_world,
            anchor_joint_owner_mask=flattened_anchor_joint_owner_mask,
            selected_anchor_mask=selected_mask,
            baseline_update_mask=baseline_joint_update_mask,
            estimation_active_mask=estimator_active,
            state=force_estimator_state,
        )
        required_estimated_normal_force_n = order9_required_anchor_normal_force_n(
            wrench_lower_contact=wrench_range.lower_contact,
            wrench_upper_contact=wrench_range.upper_contact,
            contact_normal_contact=current_surface_normal_contact,
            wrench_bound_mask=wrench_bound_mask,
            selected_anchor_mask=selected_mask,
            estimated_payload_mass_kg=estimated_mass,
            contact_friction=selected_friction,
            support_safety_factor=float(
                config.production_runtime.c3_force_estimator_support_safety_factor
            ),
        )
        privileged_actual_normal_force_n = (
            (contact.selected_contact_forces_world * current_surface_normal_world)
            .sum(dim=-1)
            .abs()
        ).detach()
        # Wrench-equivalent completion uses the policy-proposed upper inward
        # magnitude, leaving relaxation margin above the lower required hold
        # force.  The full six-axis box remains a reward objective.
        required_normal = normal_upper.clamp_min(0.0)
        preload_signed_normal_force_n = torch.where(
            wrench_bound_mask,
            (
                selected_contact_wrenches_contact[..., :3]
                * (-current_surface_normal_contact)
            )
            .sum(dim=-1)
            .clamp_min(0.0),
            torch.zeros_like(normal_lower),
        ).detach()
        preload_required_normal_force_n = torch.where(
            wrench_bound_mask,
            required_normal,
            torch.zeros_like(required_normal),
        ).detach()
        if diagnostic_contact_sweep:
            compression_weight, _ = order9_virtual_contact_phase_weight(
                pre_phase, pre_target.phase_progress
            )
            sample_mask = (
                (pre_phase == contact_phase_index)
                & (compression_weight >= 0.9)
                & (rollout_index >= max(1, (2 * int(args_cli.rollout_steps)) // 3))
            )
            sample_ids = torch.nonzero(sample_mask, as_tuple=False).flatten()
            if sample_ids.numel():
                sample_count = int(sample_ids.numel())
                diagnostic_sample_count += sample_count
                sampled_force = preload_signed_normal_force_n.index_select(
                    0, sample_ids
                ).to(torch.float64)
                diagnostic_force_sum += sampled_force.sum(dim=0)
                diagnostic_force_max = torch.maximum(
                    diagnostic_force_max, sampled_force.amax(dim=0)
                )
                sampled_load = (
                    hardware_joint_load_nm.index_select(0, sample_ids)
                    .amax(dim=(1, 2))
                    .to(torch.float64)
                )
                diagnostic_load_sum += sampled_load.sum()
                diagnostic_load_max = torch.maximum(
                    diagnostic_load_max, sampled_load.amax()
                )
                penetration = (-signed_surface_distance).clamp_min(0.0)
                sampled_penetration = penetration.index_select(0, sample_ids).to(
                    torch.float64
                )
                diagnostic_penetration_sum += sampled_penetration.sum(dim=0)
                diagnostic_penetration_max = torch.maximum(
                    diagnostic_penetration_max,
                    sampled_penetration.amax(dim=0),
                )
                diagnostic_signed_distance_sum += (
                    signed_surface_distance.index_select(0, sample_ids)
                    .to(torch.float64)
                    .sum(dim=0)
                )
                joint_tracking_error = (
                    (
                        state.local_joint_positions_rad.index_select(
                            2, reward_command_joint_indices
                        ).index_select(0, sample_ids)
                        - pre_target.nominal_joint_positions_rad.index_select(
                            0, sample_ids
                        )
                    )
                    .abs()
                    .amax(dim=(1, 2))
                    .to(torch.float64)
                )
                diagnostic_joint_tracking_error_sum += joint_tracking_error.sum()
                diagnostic_joint_tracking_error_max = torch.maximum(
                    diagnostic_joint_tracking_error_max,
                    joint_tracking_error.amax(),
                )
        allocation = policy_step.controller_result.allocation
        rotor_saturation = allocation.thrust_clipped | allocation.vectoring_clipped
        phase_goal_joint_positions = (
            joint_reference_end.index_select(0, pre_phase)
            if c3_nominal_reference is None
            else c3_nominal_reference.phase_goal_joint_positions(pre_phase)
        )
        reward = reward_engine.step(
            Order9TensorRewardInput(
                phase_index=pre_phase,
                phase_elapsed_s=pre_elapsed + float(args_cli.dt),
                phase_duration_s=duration_values[pre_phase],
                robot_body_pose_world=post_control.body_pose_world,
                robot_body_twist_world=post_control.body_twist_world,
                module_twist_world=state.module_twist_world,
                object_pose_world=state.object_pose_world,
                object_twist_world=state.object_twist_world,
                desired_robot_pose_world=(pre_target.phase_goal_robot_root_pose_world),
                desired_object_pose_world=pre_target.phase_goal_object_pose_world,
                local_joint_positions_rad=(
                    state.local_joint_positions_rad.index_select(
                        2, reward_command_joint_indices
                    )
                ),
                phase_goal_joint_positions_rad=phase_goal_joint_positions,
                selected_contact_forces_world=contact.selected_contact_forces_world,
                selected_contact_wrenches_contact=(selected_contact_wrenches_contact),
                wrench_lower_contact=wrench_range.lower_contact,
                wrench_upper_contact=wrench_range.upper_contact,
                wrench_bound_mask=wrench_bound_mask,
                selected_link_twist_world=contact.selected_link_twist_world,
                selected_contact_mask=contact.selected_contact_mask,
                selected_grasp_frame_normal_error_m=signed_surface_distance,
                selected_relative_normal_velocity_mps=(
                    selected_relative_normal_velocity_mps
                ),
                selected_motor_load_proxy=selected_motor_load_proxy,
                target_motor_load_proxy=target_motor_load_proxy.reshape(
                    1, -1
                ).expand(scene.num_envs, -1),
                contact_preload_complete=(
                    contact_preload.complete
                    if contact_preload is not None
                    else torch.ones_like(pre_phase, dtype=torch.bool)
                ),
                prohibited_collision=contact.prohibited_collision,
                support_top_z_m=support_top,
                object_half_height_m=half_height,
                qp_feasible=allocation.feasible,
                allocation_residual_norm=allocation.residual_norm,
                rotor_thrusts_n=allocation.rotor_thrusts_n,
                rotor_saturation=rotor_saturation,
                joint_torque_bias_nm=(policy_step.policy_command.joint_torque_bias_nm),
            ),
            reward_state,
        )
        deployable_gate = deployable_phase_gate.step(
            Order9DeployablePhaseGateInput(
                phase_index=pre_phase,
                robot_body_pose_world=post_control.body_pose_world,
                robot_body_twist_world=post_control.body_twist_world,
                object_pose_world=state.object_pose_world,
                object_twist_world=state.object_twist_world,
                desired_robot_pose_world=(pre_target.phase_goal_robot_root_pose_world),
                desired_object_pose_world=(pre_target.phase_goal_object_pose_world),
                local_joint_positions_rad=current_command_q,
                phase_goal_joint_positions_rad=phase_goal_joint_positions,
                selected_surface_distance_m=signed_surface_distance,
                selected_relative_speed_mps=selected_relative_speed_mps,
                selected_estimated_normal_force_n=(
                    normal_force_estimate.normal_force_n
                ),
                selected_required_normal_force_n=(required_estimated_normal_force_n),
                selected_force_estimator_confidence=(normal_force_estimate.confidence),
                selected_anchor_mask=selected_mask,
                qp_feasible=allocation.feasible,
            ),
            deployable_gate_state,
        )
        # C3 trains/evaluates pi_L under a teacher trajectory.  Phase timing is
        # supervised with privileged PhysX contact outcome, while exact 6D
        # wrench-box membership remains reward/diagnostic evidence only.  Raw
        # contact truth never enters actor features or QPID.  Later pi_H stages
        # replace this timing supervisor with learned phase decisions plus hard
        # feasibility and safety checks.
        phase_supervision_success = (
            reward.phase_success
            if c3_privileged_phase_supervision
            else deployable_gate.phase_success
        )
        if c3_nominal_reference is not None:
            phase_supervision_success = gate_order9_c3_nominal_boundary_completion(
                phase_supervision_success,
                phase_index=pre_phase,
                phase_progress=pre_target.phase_progress,
            )
        if force_observer_trace_steps is not None:
            trace_values = {
                "applied_joint_torque_nm": signed_applied_joint_torque_nm,
                "gravity_joint_torque_nm": gravity_joint_torque_nm,
                "grasp_point_linear_jacobian_world": (grasp_point_jacobian_world),
                "reaction_normal_world": estimated_reaction_normal_world,
                "selected_anchor_mask": selected_mask,
                "baseline_update_mask": baseline_joint_update_mask,
                "estimation_active_mask": estimator_active,
                "privileged_actual_normal_force_n": (privileged_actual_normal_force_n),
                "required_normal_force_n": required_estimated_normal_force_n,
                "surface_distance_m": signed_surface_distance,
                "relative_speed_mps": selected_relative_speed_mps,
                "qp_feasible": allocation.feasible,
                # QPID allocation diagnostics are additive, acceptance-
                # ineligible trace fields.  They make a terminal
                # ``qp_infeasible`` result attributable to solver convergence,
                # wrench-tracking residual, thrust clipping, or vectoring
                # clipping without changing any command or gate.
                "qpid_desired_wrench_body": (
                    policy_step.controller_result.desired_wrench_body
                ),
                "qpid_achieved_wrench_body": allocation.achieved_wrench_body,
                "qpid_residual_wrench_body": allocation.residual_wrench_body,
                "qpid_residual_norm": allocation.residual_norm,
                "qpid_solver_converged": allocation.solver_converged,
                "qpid_primal_residual_norm": allocation.primal_residual_norm,
                "qpid_dual_residual_norm": allocation.dual_residual_norm,
                "qpid_thrust_clipped": allocation.thrust_clipped,
                "qpid_vectoring_clipped": allocation.vectoring_clipped,
                "qpid_rotor_thrusts_n": allocation.rotor_thrusts_n,
                "qpid_vectoring_targets_rad": (
                    allocation.vectoring_joint_targets_rad
                ),
                "qpid_position_error_world": (
                    policy_step.controller_result.position_error_world
                ),
                "qpid_attitude_error_body": (
                    policy_step.controller_result.attitude_error_body
                ),
                "payload_feedforward_scale": payload_scale,
                "contact_constraint_weight": (
                    torch.zeros_like(payload_scale)
                    if policy_step.contact_constraint_weight is None
                    else policy_step.contact_constraint_weight
                ),
                "phase_index": pre_phase,
                "episode_serial": pre_serial,
                "online_normal_force_n": normal_force_estimate.normal_force_n,
                "online_confidence": normal_force_estimate.confidence,
            }
            for name, value in trace_values.items():
                force_observer_trace_steps.setdefault(name, []).append(
                    value.detach().clone()
                )
        if args_cli.diagnostic_phase_gate_evidence and rollout_index + 1 == int(
            args_cli.rollout_steps
        ):
            diagnostic_joint_error = (
                (current_command_q - joint_reference_end.index_select(0, pre_phase))
                .abs()
                .amax(dim=(-1, -2))
            )
            diagnostic_robot_error = torch.linalg.vector_norm(
                post_control.body_pose_world[:, :3]
                - pre_target.phase_goal_robot_root_pose_world[:, :3],
                dim=-1,
            )
            diagnostic_object_error = torch.linalg.vector_norm(
                state.object_pose_world[:, :3]
                - pre_target.phase_goal_object_pose_world[:, :3],
                dim=-1,
            )
            print(
                "ORDER9_PHASE_GATE_DIAGNOSTIC="
                + json.dumps(
                    {
                        "acceptance_eligible": False,
                        "runtime_phase_indices": sorted(
                            {int(value) for value in pre_phase.tolist()}
                        ),
                        "signed_surface_distance_m_min_by_anchor": [
                            float(value)
                            for value in signed_surface_distance.amin(dim=0).tolist()
                        ],
                        "signed_surface_distance_m_mean_by_anchor": [
                            float(value)
                            for value in signed_surface_distance.mean(dim=0).tolist()
                        ],
                        "signed_surface_distance_m_max_by_anchor": [
                            float(value)
                            for value in signed_surface_distance.amax(dim=0).tolist()
                        ],
                        "released_count": int(deployable_gate.released.sum().item()),
                        "deployable_phase_success_count": int(
                            deployable_gate.phase_success.sum().item()
                        ),
                        "privileged_phase_success_count": int(
                            reward.phase_success.sum().item()
                        ),
                        "qp_feasible_count": int(allocation.feasible.sum().item()),
                        "robot_position_error_m_min_mean_max": [
                            float(diagnostic_robot_error.amin().item()),
                            float(diagnostic_robot_error.mean().item()),
                            float(diagnostic_robot_error.amax().item()),
                        ],
                        "object_position_error_m_min_mean_max": [
                            float(diagnostic_object_error.amin().item()),
                            float(diagnostic_object_error.mean().item()),
                            float(diagnostic_object_error.amax().item()),
                        ],
                        "joint_position_error_rad_min_mean_max": [
                            float(diagnostic_joint_error.amin().item()),
                            float(diagnostic_joint_error.mean().item()),
                            float(diagnostic_joint_error.amax().item()),
                        ],
                        "estimated_normal_force_n_min_mean_max": [
                            float(normal_force_estimate.normal_force_n.amin().item()),
                            float(normal_force_estimate.normal_force_n.mean().item()),
                            float(normal_force_estimate.normal_force_n.amax().item()),
                        ],
                        "estimated_normal_force_n_min_mean_max_by_anchor": [
                            [
                                float(values.amin().item()),
                                float(values.mean().item()),
                                float(values.amax().item()),
                            ]
                            for values in normal_force_estimate.normal_force_n.movedim(
                                0, 1
                            )
                        ],
                        "required_normal_force_n_min_mean_max": [
                            float(required_estimated_normal_force_n.amin().item()),
                            float(required_estimated_normal_force_n.mean().item()),
                            float(required_estimated_normal_force_n.amax().item()),
                        ],
                        "force_estimator_confidence_min_mean_max": [
                            float(normal_force_estimate.confidence.amin().item()),
                            float(normal_force_estimate.confidence.mean().item()),
                            float(normal_force_estimate.confidence.amax().item()),
                        ],
                        "force_estimator_confidence_min_mean_max_by_anchor": [
                            [
                                float(values.amin().item()),
                                float(values.mean().item()),
                                float(values.amax().item()),
                            ]
                            for values in normal_force_estimate.confidence.movedim(0, 1)
                        ],
                        "normal_jacobian_norm_m_min_mean_max_by_anchor": [
                            [
                                float(values.amin().item()),
                                float(values.mean().item()),
                                float(values.amax().item()),
                            ]
                            for values in normal_force_estimate.normal_jacobian_norm_m.movedim(
                                0, 1
                            )
                        ],
                        "privileged_actual_normal_force_n_min_mean_max": [
                            float(privileged_actual_normal_force_n.amin().item()),
                            float(privileged_actual_normal_force_n.mean().item()),
                            float(privileged_actual_normal_force_n.amax().item()),
                        ],
                        "privileged_actual_normal_force_n_min_mean_max_by_anchor": [
                            [
                                float(values.amin().item()),
                                float(values.mean().item()),
                                float(values.amax().item()),
                            ]
                            for values in privileged_actual_normal_force_n.movedim(0, 1)
                        ],
                        "force_estimator_fit_residual_nm_min_mean_max": [
                            float(normal_force_estimate.fit_residual_nm.amin().item()),
                            float(normal_force_estimate.fit_residual_nm.mean().item()),
                            float(normal_force_estimate.fit_residual_nm.amax().item()),
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        last_phase = pre_phase == (phase_count - 1)
        if diagnostic_virtual_contact_additional_lead_sweep_mm is not None:
            diagnostic_terminal_phase = ORDER9_OBJECT_TASK_PHASES.index(
                Order9ObjectTaskPhase.LIFT
            )
            completed_task = (
                phase_supervision_success
                & (pre_phase == diagnostic_terminal_phase)
                & ~reward.terminal_failure
            )
        else:
            completed_task = (
                phase_supervision_success & last_phase & ~reward.terminal_failure
            )
        terminal = reward.terminal_failure | completed_task
        if diagnostic_contact_sweep:
            terminal = torch.zeros_like(terminal)
            completed_task = torch.zeros_like(completed_task)
        terminal_count += int(terminal.sum().item())
        success_count += int(completed_task.sum().item())
        if tensorboard_logger is not None:
            tensorboard_logger.log_rollout_step(
                rollout_index=rollout_index,
                reward=reward.reward,
                reward_terms=reward.terms,
                phase_index=pre_phase,
                statuses={
                    "phase_success": phase_supervision_success,
                    "privileged_phase_success": reward.phase_success,
                    "privileged_wrench_range_satisfied": (
                        reward.wrench_range_satisfied
                    ),
                    "deployable_phase_success": (deployable_gate.phase_success),
                    "deployable_grasp_ready": deployable_gate.grasp_ready,
                    "task_success": completed_task,
                    "terminal": terminal,
                    "hard_collision": reward.hard_collision,
                    "object_dropped": reward.object_dropped,
                    "qp_infeasible_terminal": reward.qp_infeasible_terminal,
                    "timeout": reward.timeout,
                    "qp_feasible": allocation.feasible,
                    "contact_preload_initialized": (
                        contact_preload.initialized
                        if contact_preload is not None
                        else torch.ones_like(pre_phase, dtype=torch.bool)
                    ),
                    "contact_preload_complete": (
                        contact_preload.complete
                        if contact_preload is not None
                        else torch.ones_like(pre_phase, dtype=torch.bool)
                    ),
                },
                elapsed_s=max(time.perf_counter() - rollout_started, 1.0e-12),
                runtime_sample=load_monitor.latest_sample(),
            )
        if evaluation_mode:
            evaluation_step_count[evaluation_active] += 1
            evaluation_return[evaluation_active] += reward.reward[evaluation_active]
            evaluation_terminal = terminal & evaluation_active
            for environment in (
                torch.nonzero(evaluation_terminal, as_tuple=False).flatten().tolist()
            ):
                succeeded = bool(completed_task[environment].item())
                failure_reason = None
                if not succeeded:
                    if bool(reward.hard_collision[environment].item()):
                        failure_reason = "hard_collision"
                    elif bool(reward.object_dropped[environment].item()):
                        failure_reason = "object_dropped"
                    elif bool(reward.qp_infeasible_terminal[environment].item()):
                        failure_reason = "qp_infeasible_terminal"
                    elif bool(reward.timeout[environment].item()):
                        failure_reason = "phase_timeout"
                    else:  # pragma: no cover - terminal causes are exhaustive.
                        failure_reason = "terminal_failure"
                evaluation_outcomes.append(
                    {
                        "environment": environment,
                        "task_success": succeeded,
                        "safety_failure": bool(
                            reward.hard_collision[environment].item()
                            or reward.object_dropped[environment].item()
                            or reward.qp_infeasible_terminal[environment].item()
                        ),
                        "failure_reason": failure_reason,
                        "environment_step_count": int(
                            evaluation_step_count[environment].item()
                        ),
                        "episode_return": float(evaluation_return[environment].item()),
                        "terminal_phase_index": int(
                            pre_actor_phase[environment].item()
                        ),
                        "hard_collision": bool(
                            reward.hard_collision[environment].item()
                        ),
                        "object_dropped": bool(
                            reward.object_dropped[environment].item()
                        ),
                        "qp_infeasible_terminal": bool(
                            reward.qp_infeasible_terminal[environment].item()
                        ),
                        "timeout": bool(reward.timeout[environment].item()),
                        "contact_preload_initialized": bool(
                            contact_preload.initialized[environment].item()
                            if contact_preload is not None
                            else True
                        ),
                        "contact_preload_complete": bool(
                            contact_preload.complete[environment].item()
                            if contact_preload is not None
                            else True
                        ),
                        "contact_preload_frozen_anchor_count": float(
                            contact_preload.frozen_anchor[environment].sum().item()
                            if contact_preload is not None
                            else len(selected_anchor_ids)
                        ),
                        "contact_preload_minimum_load_nm": float(
                            contact_preload.load_nm_by_anchor[environment].amin().item()
                            if contact_preload is not None
                            else 0.0
                        ),
                        "contact_preload_minimum_signed_normal_force_n": float(
                            preload_signed_normal_force_n[environment].amin().item()
                        ),
                        "contact_preload_minimum_required_normal_force_n": float(
                            preload_required_normal_force_n[environment].amin().item()
                        ),
                        "estimated_normal_force_n_minimum": float(
                            normal_force_estimate.normal_force_n[environment]
                            .amin()
                            .item()
                        ),
                        "required_normal_force_n_maximum": float(
                            required_estimated_normal_force_n[environment].amax().item()
                        ),
                        "force_estimator_confidence_minimum": float(
                            normal_force_estimate.confidence[environment].amin().item()
                        ),
                        "force_estimator_fit_residual_nm": float(
                            normal_force_estimate.fit_residual_nm[environment].item()
                        ),
                        "privileged_actual_normal_force_n_minimum": float(
                            privileged_actual_normal_force_n[environment].amin().item()
                        ),
                    }
                )
            evaluation_active &= ~evaluation_terminal
        next_phase_mask = phase_supervision_success & ~last_phase & ~terminal
        if diagnostic_contact_sweep:
            next_phase_mask = torch.zeros_like(next_phase_mask)
        next_ids = torch.nonzero(next_phase_mask, as_tuple=False).flatten()
        deployable_gate_state = deployable_gate.next_state
        force_estimator_state = normal_force_estimate.next_state
        if next_ids.numel():
            deployable_gate_state = deployable_phase_gate.reset_state_subset(
                deployable_gate_state,
                next_ids,
                clear_release_latch=False,
            )
            transitioned_from = pre_phase.index_select(0, next_ids)
            for source_index in range(phase_count - 1):
                count = int((transitioned_from == source_index).sum().item())
                if count:
                    key = (
                        f"{ORDER9_OBJECT_TASK_PHASES[source_index].value}->"
                        f"{ORDER9_OBJECT_TASK_PHASES[source_index + 1].value}"
                    )
                    phase_transition_counts[key] += count
            next_indices = pre_phase[next_ids] + 1
            if c3_nominal_reference is None:
                bank.capture(scene, env_ids=next_ids, phase_indices=next_indices)
            phase_index[next_ids] = next_indices
            phase_elapsed[next_ids] = 0.0
            planned_body_start, planned_object_start = (
                pre_target.planned_successor_start(next_ids)
            )
            phase_start_body_pose[next_ids] = planned_body_start
            phase_start_object_pose[next_ids] = planned_object_start
        continuing = ~terminal & ~next_phase_mask
        phase_elapsed[continuing] += float(args_cli.dt)
        episode_time[~terminal] += float(args_cli.dt)
        episode_step[~terminal] += 1
        target = task_runtime.target(
            phase_index=phase_index,
            phase_elapsed_s=phase_elapsed,
            reset_robot_root_pose_world=phase_start_body_pose,
            reset_object_pose_world=phase_start_object_pose,
            reset_joint_positions_rad=joint_reference_start.index_select(
                0, phase_index
            ),
            phase_end_joint_positions_rad=joint_reference_end.index_select(
                0, phase_index
            ),
            lift_clearance_m=lift_clearance,
            transport_distance_m=transport_distance,
        )
        target = _condition_target_on_teacher_reference(
            target,
            teacher_reference=teacher_reference,
            phase_index=phase_index,
            scene_origins=scene.env_origins,
            task_object_position_world=task_object_position_world,
        )
        if c3_nominal_reference is not None:
            target = c3_nominal_reference.condition(
                target,
                phase_index=phase_index,
                phase_elapsed_s=phase_elapsed,
                scene_origins=scene.env_origins,
            )
            target = _apply_virtual_contact_target(target)
        reward_state = _reset_goal_distance_for_phase_transition(
            reward.next_state,
            env_ids=next_ids,
            object_pose_world=state.object_pose_world,
            desired_object_pose_world=target.phase_goal_object_pose_world,
        )
        evaluation_complete = bool(
            evaluation_mode and not bool(evaluation_active.any())
        )
        final_collection_step = (
            rollout_index + 1 == int(args_cli.rollout_steps) or evaluation_complete
        )
        truncated = torch.zeros_like(terminal)
        bootstrap = torch.zeros_like(phase_elapsed)
        if final_collection_step:
            truncated = ~terminal
            if bool(truncated.any()) and not args_cli.diagnostic_nominal_qpid_only:
                bootstrap_contact_feedback_features = None
                if (
                    checkpoint.metadata.policy_version
                    in {
                        ORDER9_CONTACT_FEEDBACK_PI_L_POLICY_VERSION,
                        ORDER9_CATEGORICAL_CONTACT_NORMAL_PI_L_POLICY_VERSION,
                    }
                ):
                    (
                        bootstrap_surface_distance_m,
                        bootstrap_relative_twist_contact,
                    ) = _deployable_selected_contact_feedback_inputs(
                        selected_body_pose_world=selected_body_pose,
                        selected_body_twist_world=(
                            contact.selected_link_twist_world
                        ),
                        selected_anchor_local_pose=(
                            selected_anchor_local_pose
                        ),
                        object_pose_world=state.object_pose_world,
                        object_twist_world=state.object_twist_world,
                        contact_frame_pose_world=(
                            wrench_range.contact_frame_pose_world
                        ),
                        contact_normal_world=(
                            wrench_range.contact_normal_world
                        ),
                    )
                    bootstrap_contact_feedback_features = (
                        build_order9_deployable_contact_feedback_features(
                            signed_surface_distance_m=(
                                bootstrap_surface_distance_m
                            ),
                            relative_twist_contact=(
                                bootstrap_relative_twist_contact
                            ),
                            signed_hardware_joint_load_nm=(
                                signed_applied_joint_torque_nm.reshape(
                                    command_joint_shape
                                )
                            ),
                            basis=contact_space_action_basis,
                        )
                    )
                bootstrap_payload_scale = order9_payload_feedforward_scale(
                    phase_index,
                    target.phase_progress,
                )
                bootstrap_value = policy_runtime.evaluate_bootstrap_value(
                    time_s=episode_time,
                    phase_index=_actor_phase_indices(phase_index),
                    task_target=target,
                    state=state,
                    estimated_payload_mass_kg=(
                        estimated_mass * bootstrap_payload_scale
                    ),
                    estimated_payload_inertia_body=(
                        estimated_inertia
                        * bootstrap_payload_scale.unsqueeze(-1)
                    ),
                    payload_active=bootstrap_payload_scale > 0.0,
                    estimated_payload_com_object=estimated_com,
                    hardware_joint_load_nm=hardware_joint_load_nm,
                    contact_feedback_features=(
                        bootstrap_contact_feedback_features
                    ),
                )
                bootstrap[truncated] = bootstrap_value[truncated]
        buffer.append(
            _artifact_step(
                valid=torch.ones_like(terminal),
                pre_time=pre_time,
                pre_phase=pre_actor_phase,
                pre_elapsed=pre_elapsed,
                duration=duration_values[pre_phase],
                pre_serial=pre_serial,
                pre_step=pre_step,
                pre_state=pre_state,
                pre_target=pre_target,
                policy_step=policy_step,
                selected_mask=selected_mask,
                contact=contact,
                selected_contact_wrenches_contact=(selected_contact_wrenches_contact),
                wrench_range=wrench_range,
                wrench_bound_mask=wrench_bound_mask,
                reward=reward,
                reward_names=reward_names,
                rotor_saturation=rotor_saturation,
                terminal=terminal,
                truncated=truncated,
                bootstrap=bootstrap,
                post_state=state,
                include_contact_compression_residual_action=(
                    checkpoint.metadata.policy_version
                    == ORDER9_MORPHOLOGY_INVARIANT_COMPRESSION_PI_L_POLICY_VERSION
                ),
                include_contact_space_action=(
                    checkpoint.metadata.policy_version
                    in ORDER9_CONTACT_SPACE_PI_L_POLICY_VERSIONS
                ),
            )
        )
        policy_runtime.finish_transition(
            phase_success=phase_supervision_success,
            terminal_or_reset=terminal,
            current_vectoring_angles_rad=post_control.current_vectoring_angles_rad,
        )
        if evaluation_complete:
            break
        reset_ids = torch.nonzero(terminal, as_tuple=False).flatten()
        if reset_ids.numel() and not final_collection_step:
            preload_signed_normal_force_n[reset_ids] = 0.0
            preload_required_normal_force_n[reset_ids] = 0.0
            deployable_gate_state = deployable_phase_gate.reset_state_subset(
                deployable_gate_state, reset_ids
            )
            force_estimator_state = anchor_force_estimator.reset_state_subset(
                force_estimator_state, reset_ids
            )
            episode_serial[reset_ids] += 1
            if state_inheritance_shard:
                reset_phase_values = order9_c3_state_inheritance_phase_indices(
                    reset_ids.detach().cpu().tolist(),
                    episode_serial.index_select(0, reset_ids).detach().cpu().tolist(),
                    (
                        config.production_runtime.c3_state_inheritance_initial_phase_indices
                    ),
                )
                reset_phases = torch.tensor(
                    reset_phase_values,
                    device=scene.device,
                    dtype=torch.long,
                )
                reset_strata = bank.state_inheritance_strata(
                    reset_ids,
                    reset_phases,
                    episode_serial,
                    fixed_stratum_index=(
                        state_inheritance_fixed_stratum_index
                    ),
                )
            else:
                reset_phases, reset_strata = bank.select_reset_states(
                    reset_ids, episode_serial
                )
            phase_index[reset_ids] = reset_phases
            release_complete_reset_ids = reset_ids[
                reset_phases
                >= ORDER9_OBJECT_TASK_PHASES.index(Order9ObjectTaskPhase.RETREAT)
            ]
            if release_complete_reset_ids.numel():
                deployable_gate_state = (
                    deployable_phase_gate.assume_release_complete_subset(
                        deployable_gate_state,
                        release_complete_reset_ids,
                    )
                )
            episode_time[reset_ids] = 0.0
            episode_step[reset_ids] = 0
            (
                restored_elapsed,
                restored_body_start,
                restored_object_start,
            ) = bank.restore(
                scene,
                env_ids=reset_ids,
                phase_indices=reset_phases,
                stratum_indices=reset_strata,
            )
            phase_elapsed[reset_ids] = restored_elapsed
            sim.forward()
            scene.update(0.0)
            state = io.gather_state(robot=robot, object_asset=obj)
            reset_control = policy_runtime.builder.build(
                module_pose_world=state.module_pose_world,
                module_twist_world=state.module_twist_world,
                local_joint_positions_rad=state.local_joint_positions_rad,
            )
            phase_start_body_pose[reset_ids] = (
                restored_body_start
                if c3_phase_reset_expected is not None
                else reset_control.body_pose_world[reset_ids]
            )
            phase_start_object_pose[reset_ids] = (
                restored_object_start
                if c3_phase_reset_expected is not None
                else state.object_pose_world[reset_ids]
            )
            target = task_runtime.target(
                phase_index=phase_index,
                phase_elapsed_s=phase_elapsed,
                reset_robot_root_pose_world=phase_start_body_pose,
                reset_object_pose_world=phase_start_object_pose,
                reset_joint_positions_rad=joint_reference_start.index_select(
                    0, phase_index
                ),
                phase_end_joint_positions_rad=joint_reference_end.index_select(
                    0, phase_index
                ),
                lift_clearance_m=lift_clearance,
                transport_distance_m=transport_distance,
            )
            target = _condition_target_on_teacher_reference(
                target,
                teacher_reference=teacher_reference,
                phase_index=phase_index,
                scene_origins=scene.env_origins,
                task_object_position_world=task_object_position_world,
            )
            # ``target`` is a batch-wide tensor.  Resetting even one completed
            # environment therefore rebuilds references for every still-live
            # evaluation environment as well.  Reapply the persisted C3
            # nominal contract here just as in the ordinary step path; without
            # this, remaining environments receive one generic task-runtime
            # target and can see a discontinuous CoM jump.
            if c3_nominal_reference is not None:
                target = c3_nominal_reference.condition(
                    target,
                    phase_index=phase_index,
                    phase_elapsed_s=phase_elapsed,
                    scene_origins=scene.env_origins,
                )
                target = _apply_virtual_contact_target(target)
            reward_state = reward_engine.reset_state_subset(
                reward_state,
                reset_ids,
                object_pose_world=state.object_pose_world,
                desired_object_pose_world=target.phase_goal_object_pose_world,
            )
    evaluation_horizon_timeout_count = 0
    if evaluation_mode and bool(evaluation_active.any()):
        horizon_timeout_environments = (
            torch.nonzero(evaluation_active, as_tuple=False).flatten().tolist()
        )
        for environment in horizon_timeout_environments:
            evaluation_outcomes.append(
                order9_evaluation_horizon_timeout_outcome(
                    environment=environment,
                    environment_step_count=int(
                        evaluation_step_count[environment].item()
                    ),
                    episode_return=float(evaluation_return[environment].item()),
                    terminal_phase_index=int(pre_actor_phase[environment].item()),
                )
            )
        evaluation_horizon_timeout_count = len(horizon_timeout_environments)
        evaluation_active[
            torch.tensor(
                horizon_timeout_environments,
                device=evaluation_active.device,
                dtype=torch.long,
            )
        ] = False
    if str(args_cli.device).startswith("cuda"):
        torch.cuda.synchronize()
    rollout_elapsed = time.perf_counter() - rollout_started
    runtime_load = load_monitor.stop(torch_module=torch)
    diagnostic_sweep_payload = None
    if diagnostic_contact_sweep:
        if diagnostic_sample_count < 1 or virtual_contact_solution is None:
            raise RuntimeError("contact sweep produced no full-compression samples")
        divisor = float(diagnostic_sample_count)
        diagnostic_sweep_payload = {
            "contract_version": ("order9_actor_free_virtual_contact_physx_sweep_v1"),
            "acceptance_eligible": False,
            "bucket_id": task.metadata.get("order9_rollout_bucket_id"),
            "module_count": len(morphology.modules),
            "virtual_contact_compression_version": (
                ORDER9_VIRTUAL_CONTACT_COMPRESSION_VERSION
            ),
            "requested_inward_lead_mm": 1000.0 * virtual_contact_lead_m,
            "achieved_free_space_inward_displacement_mm_by_anchor": {
                str(key): 1000.0 * float(value)
                for key, value in (
                    virtual_contact_solution.achieved_inward_displacement_m.items()
                )
            },
            "maximum_virtual_joint_delta_rad": float(
                virtual_contact_solution.maximum_joint_delta_rad
            ),
            "virtual_joint_target_saturated": bool(virtual_contact_solution.saturated),
            "pi_l_command_bypassed": True,
            "raw_physx_used_by_controller": False,
            "sample_count": diagnostic_sample_count,
            "mean_signed_normal_force_n_by_anchor": [
                float(value) for value in (diagnostic_force_sum / divisor).tolist()
            ],
            "maximum_signed_normal_force_n_by_anchor": [
                float(value) for value in diagnostic_force_max.tolist()
            ],
            "mean_maximum_motor_current_equivalent_load_nm": float(
                (diagnostic_load_sum / divisor).item()
            ),
            "maximum_motor_current_equivalent_load_nm": float(
                diagnostic_load_max.item()
            ),
            "mean_kinematic_grasp_frame_penetration_mm_by_anchor": [
                1000.0 * float(value)
                for value in (diagnostic_penetration_sum / divisor).tolist()
            ],
            "maximum_kinematic_grasp_frame_penetration_mm_by_anchor": [
                1000.0 * float(value) for value in diagnostic_penetration_max.tolist()
            ],
            "mean_signed_grasp_frame_surface_distance_mm_by_anchor": [
                1000.0 * float(value)
                for value in (diagnostic_signed_distance_sum / divisor).tolist()
            ],
            "mean_maximum_joint_tracking_error_rad": float(
                (diagnostic_joint_tracking_error_sum / divisor).item()
            ),
            "maximum_joint_tracking_error_rad": float(
                diagnostic_joint_tracking_error_max.item()
            ),
            "contact_force_is_privileged_diagnostic_only": True,
            "kinematic_penetration_semantics": (
                "grasp_contact_frame_signed_depth_below_authored_surface_plane"
            ),
            "rollout_steps": int(args_cli.rollout_steps),
            "environment_count": int(scene.num_envs),
            "dt_s": float(args_cli.dt),
        }
        diagnostic_sweep_payload["payload_hash"] = stable_hash(diagnostic_sweep_payload)
        sweep_path = Path(args_cli.diagnostic_contact_sweep_json).resolve()
        sweep_path.parent.mkdir(parents=True, exist_ok=True)
        sweep_path.write_text(
            json.dumps(diagnostic_sweep_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if force_observer_trace_steps is not None:
        force_observer_trace_path = Path(
            args_cli.diagnostic_force_observer_trace
        ).resolve()
        force_observer_trace_sha256 = write_order9_force_observer_trace(
            force_observer_trace_path,
            metadata={
                "trace_version": ORDER9_FORCE_OBSERVER_TRACE_VERSION,
                "generation_id": args_cli.generation_id,
                "bucket_id": task.metadata.get("order9_rollout_bucket_id"),
                "module_count": len(morphology.modules),
                "checkpoint_sha256": checkpoint.sha256,
                "force_estimator_version": (
                    ORDER9_ANCHOR_NORMAL_FORCE_ESTIMATOR_VERSION
                ),
                "force_estimator_config": {
                    name: getattr(force_estimator_config, name)
                    for name in force_estimator_config.__dataclass_fields__
                },
                "reaction_direction_contract": (
                    ORDER9_ANCHOR_REACTION_DIRECTION_CONTRACT
                ),
                "control_dt_s": float(args_cli.dt),
                "confidence_threshold": float(
                    deployable_phase_gate.config.contact_force_estimator_confidence_threshold
                ),
                "contact_distance_tolerance_m": float(
                    deployable_phase_gate.config.contact_surface_distance_tolerance_m
                ),
                "contact_inward_limit_m": float(
                    deployable_phase_gate.config.contact_surface_inward_limit_m
                ),
                "relative_speed_tolerance_mps": float(
                    deployable_phase_gate.config.contact_relative_speed_tolerance_mps
                ),
                "contact_dwell_s": float(deployable_phase_gate.config.contact_dwell_s),
                "raw_contact_used_by_observer": False,
                "privileged_contact_saved_for_offline_validation_only": True,
            },
            tensors={
                **{
                    name: torch.stack(values)
                    for name, values in force_observer_trace_steps.items()
                },
                "anchor_joint_owner_mask": (
                    flattened_anchor_joint_owner_mask.detach().clone()
                ),
            },
        )
        print(
            "ORDER9_FORCE_OBSERVER_TRACE="
            + json.dumps(
                {
                    "path": str(force_observer_trace_path),
                    "sha256": force_observer_trace_sha256,
                    "step_count": len(
                        force_observer_trace_steps["applied_joint_torque_nm"]
                    ),
                    "environment_count": int(scene.num_envs),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    artifact = buffer.finalize()
    artifact.metadata.update(
        {
            "setup_wall_elapsed_s": float(setup_elapsed),
            "rollout_wall_elapsed_s": float(rollout_elapsed),
            "collection_wall_elapsed_s": float(setup_elapsed + rollout_elapsed),
            "aggregate_env_steps_per_s": (
                artifact.environment_step_count / rollout_elapsed
            ),
            "end_to_end_env_steps_per_s": (
                artifact.environment_step_count / (setup_elapsed + rollout_elapsed)
            ),
            "environment_count": int(scene.num_envs),
            "rollout_steps": int(artifact.step_count),
            "requested_rollout_steps": int(args_cli.rollout_steps),
            "configured_environment_count": configured_runtime.environment_count,
            "configured_rollout_steps_per_environment": (
                configured_runtime.rollout_steps_per_environment
            ),
            "configured_generation_environment_steps": (
                configured_runtime.generation_environment_steps
            ),
            "runtime_override_used": runtime_override_used,
            "c3_train_rollout_mode": (
                ORDER9_C3_STATE_INHERITANCE_ROLLOUT_MODE
                if state_inheritance_shard
                else ORDER9_C3_PHASE_RESET_ROLLOUT_MODE
            ),
            "state_inheritance_initial_phase_indices": (
                list(
                    config.production_runtime.c3_state_inheritance_initial_phase_indices
                )
                if state_inheritance_shard
                else None
            ),
            "state_inheritance_preserves_phase_transitions": bool(
                state_inheritance_shard
            ),
            "state_inheritance_initial_stratum_policy": (
                state_inheritance_stratum_policy
                if state_inheritance_shard
                else None
            ),
            "state_inheritance_fixed_reset_progress_fraction": (
                state_inheritance_fixed_progress
                if state_inheritance_shard
                else None
            ),
            "state_inheritance_transition_backward_curriculum_contract": (
                ORDER9_C3_TRANSITION_BACKWARD_CURRICULUM_CONTRACT
                if state_inheritance_shard
                and state_inheritance_fixed_progress is not None
                else None
            ),
            "state_inheritance_recurrent_state_reset_only_at_episode_boundary": bool(
                state_inheritance_shard
            ),
            "topology_stratified_shard": {
                "enabled": bool(topology_stratified_shard),
                "index": args_cli.production_topology_shard_index,
                "count": args_cli.production_topology_shard_count,
                "environment_count": (
                    args_cli.production_topology_shard_environment_count
                ),
                "configured_total_environment_count": (
                    configured_runtime.environment_count
                ),
            },
            "runtime_load": runtime_load,
            "terminal_count": int(terminal_count),
            "successful_terminal_count": int(success_count),
            "evaluation_horizon_timeout_contract": (
                ORDER9_EVALUATION_HORIZON_TIMEOUT_CONTRACT if evaluation_mode else None
            ),
            "evaluation_horizon_timeout_count": int(evaluation_horizon_timeout_count),
            "phase_transition_counts": dict(phase_transition_counts),
            "evaluation_mode": bool(evaluation_mode),
            "training_only_continuous_teacher_rollout": bool(
                args_cli.training_only_continuous_teacher_rollout
            ),
            "promotion_evidence_eligible": bool(
                evaluation_mode
                and not args_cli.training_only_continuous_teacher_rollout
                and args_cli.virtual_contact_lead_mm is None
                and diagnostic_virtual_contact_additional_lead_sweep_mm is None
                and diagnostic_contact_normal_residual_sweep_mm is None
            ),
            "deterministic_policy": bool(deterministic_policy),
            "initial_phase_zero": bool(initial_phase_zero),
            "formal_phase_zero_start": bool(args_cli.formal_phase_zero_start),
            "formal_phase_zero_state_source": (
                ORDER9_C3_FORMAL_PHASE_ZERO_STATE_SOURCE
                if args_cli.formal_phase_zero_start
                else None
            ),
            "initial_phase_elapsed_s": (
                0.0 if args_cli.formal_phase_zero_start else None
            ),
            "diagnostic_initial_phase_index": (args_cli.diagnostic_initial_phase_index),
            "diagnostic_initial_phase_stratum_index": (
                args_cli.diagnostic_initial_phase_stratum_index
            ),
            "diagnostic_deterministic_all_phases": bool(
                args_cli.diagnostic_deterministic_policy
                and args_cli.diagnostic_initial_phase_index is None
            ),
            "diagnostic_nominal_qpid_only": bool(args_cli.diagnostic_nominal_qpid_only),
            "diagnostic_action_ablation": (args_cli.diagnostic_action_ablation),
            "diagnostic_virtual_contact_additional_lead_sweep_mm": (
                None
                if diagnostic_virtual_contact_additional_lead_sweep_mm is None
                else list(diagnostic_virtual_contact_additional_lead_sweep_mm)
            ),
            "training_nominal_preload_deficit_version": (
                ORDER9_TRAINING_NOMINAL_PRELOAD_DEFICIT_VERSION
                if (
                    training_nominal_preload_deficit_mm is not None
                    and len(morphology.modules)
                    == args_cli.training_nominal_preload_deficit_module_count
                )
                else None
            ),
            "training_nominal_preload_deficit_module_count": (
                args_cli.training_nominal_preload_deficit_module_count
                if training_nominal_preload_deficit_mm is not None
                else None
            ),
            "training_nominal_preload_deficit_mm": (
                None
                if training_nominal_preload_deficit_mm is None
                else list(training_nominal_preload_deficit_mm)
            ),
            "training_nominal_preload_deficit_m_by_environment": (
                None
                if training_nominal_preload_deficit_plan is None
                else list(
                    training_nominal_preload_deficit_plan.deficit_m_by_environment
                )
            ),
            "training_nominal_preload_total_lead_m_by_environment": (
                None
                if training_nominal_preload_deficit_plan is None
                else list(
                    training_nominal_preload_deficit_plan.total_lead_m_by_environment
                )
            ),
            "diagnostic_virtual_contact_total_lead_m_by_environment": (
                None
                if diagnostic_virtual_contact_total_lead_m_by_environment is None
                else list(diagnostic_virtual_contact_total_lead_m_by_environment)
            ),
            "diagnostic_contact_normal_residual_sweep_mm": (
                None
                if diagnostic_contact_normal_residual_sweep_mm is None
                else list(diagnostic_contact_normal_residual_sweep_mm)
            ),
            "diagnostic_contact_normal_residual_m_by_environment": (
                None
                if diagnostic_contact_normal_residual_m_by_environment is None
                else diagnostic_contact_normal_residual_m_by_environment.tolist()
            ),
            "diagnostic_terminal_runtime_phase": (
                Order9ObjectTaskPhase.LIFT.value
                if diagnostic_virtual_contact_additional_lead_sweep_mm is not None
                else None
            ),
            "c3_action_contract": args_cli.c3_action_contract,
            "contact_compression_only_action_applied": bool(
                policy_runtime.builder.module_count
                in config.production_runtime.c3_contact_compression_only_module_counts
                and args_cli.diagnostic_action_ablation is None
                and args_cli.c3_action_contract is None
            ),
            "joint_only_action_applied": bool(
                policy_runtime.builder.module_count
                in config.production_runtime.c3_joint_only_module_counts
                and args_cli.diagnostic_action_ablation is None
                and args_cli.c3_action_contract is None
            ),
            "diagnostic_learned_phase_strata": bool(
                args_cli.diagnostic_learned_phase_strata
            ),
            "virtual_contact_compression_version": (
                ORDER9_VIRTUAL_CONTACT_COMPRESSION_VERSION
                if virtual_contact_solution is not None
                else None
            ),
            "virtual_contact_inward_lead_m": virtual_contact_lead_m,
            "actuator_aware_nominal_preload_version": (
                ORDER9_ACTUATOR_AWARE_NOMINAL_PRELOAD_VERSION
                if actuator_aware_nominal_preload_solution is not None
                else None
            ),
            "actuator_aware_nominal_preload": (
                None
                if actuator_aware_nominal_preload_solution is None
                else actuator_aware_nominal_preload_solution.to_dict()
            ),
            "actuator_aware_nominal_preload_collision_safe_scale": (
                None
                if nominal_preload_collision_limit is None
                else float(nominal_preload_collision_limit.maximum_action_scale)
            ),
            "actuator_aware_nominal_preload_applied_inward_lead_m": (
                nominal_preload_applied_inward_lead_m
            ),
            "raw_physx_used_by_virtual_contact_controller": False,
            "contact_compression_action_adapter_version": (
                _contact_compression_action_adapter_version(
                    policy_runtime.policy.policy_version,
                    c3_action_contract=args_cli.c3_action_contract,
                )
                if contact_compression_action_solution is not None
                else None
            ),
            "contact_compression_action_span_m": (
                None
                if contact_compression_action_solution is None
                else float(contact_compression_action_solution.inward_lead_m)
            ),
            "contact_compression_action_control_joint_id": (
                contact_compression_action_control_joint_id
            ),
            "contact_compression_action_maximum_joint_delta_rad": (
                None
                if contact_compression_collision_limit is None
                else float(
                    contact_compression_collision_limit.maximum_action_joint_delta_rad
                )
            ),
            "contact_compression_collision_shield_version": (
                None
                if contact_compression_collision_limit is None
                else ORDER9_CONTACT_COMPRESSION_COLLISION_SHIELD_VERSION
            ),
            "contact_compression_collision_safe_action_scale": (
                None
                if contact_compression_collision_limit is None
                else float(
                    contact_compression_collision_limit.maximum_action_scale
                )
            ),
            "contact_compression_collision_safe_span_m": (
                None
                if contact_compression_collision_limit is None
                else float(contact_compression_action_solution.inward_lead_m)
                * float(
                    contact_compression_collision_limit.maximum_action_scale
                )
            ),
            "contact_compression_collision_limiting_scene_id": (
                None
                if contact_compression_collision_limit is None
                else contact_compression_collision_limit.limiting_scene_id
            ),
            "contact_compression_action_raw_contact_input": False,
            "anchor_normal_force_estimator_version": (
                ORDER9_ANCHOR_NORMAL_FORCE_ESTIMATOR_VERSION
            ),
            "anchor_reaction_direction_contract": (
                ORDER9_ANCHOR_REACTION_DIRECTION_CONTRACT
            ),
            "anchor_normal_force_estimator_config": {
                name: getattr(force_estimator_config, name)
                for name in force_estimator_config.__dataclass_fields__
            },
            "anchor_normal_force_estimator_raw_contact_input": False,
            "anchor_normal_force_required_support_safety_factor": float(
                config.production_runtime.c3_force_estimator_support_safety_factor
            ),
            "tensorboard_enabled": tensorboard_logger is not None,
            "tensorboard_log_dir": (
                None if tensorboard_log_dir is None else str(tensorboard_log_dir)
            ),
            "tensorboard_update_index": tensorboard_update_index,
            "tensorboard_logger_version": (
                ORDER9_TENSORBOARD_LOGGER_VERSION
                if tensorboard_logger is not None
                else None
            ),
        }
    )
    if tensorboard_logger is not None:
        tensorboard_logger.log_rollout_summary(
            environment_steps=artifact.environment_step_count,
            wall_elapsed_s=rollout_elapsed,
            runtime_load=runtime_load,
        )
        tensorboard_logger.close()
    raw_sha = write_order9_tensor_rollout_artifact(args_cli.output_raw, artifact)
    evaluation_episode_count = 0
    if evaluation_mode:
        requested = int(args_cli.evaluation_episode_count)
        if len(evaluation_outcomes) < requested:
            raise RuntimeError(
                "Order9 evaluation rollout produced "
                f"{len(evaluation_outcomes)} first-terminal episodes; "
                f"{requested} required"
            )
        selected_outcomes = sorted(
            evaluation_outcomes, key=lambda item: int(item["environment"])
        )[:requested]
        raw_path = Path(args_cli.output_raw).resolve()
        evaluation_episodes = []
        for outcome in selected_outcomes:
            environment = int(outcome["environment"])
            succeeded = bool(outcome["task_success"])
            evaluation_episodes.append(
                Order9EvaluationEpisode(
                    episode_id=(
                        f"{args_cli.generation_id}:evaluation:env:{environment:04d}"
                    ),
                    task_id=translated_tasks[environment].task_id,
                    split=split,
                    random_seed=int(args_cli.seed) + environment,
                    task_success=succeeded,
                    no_fallback_success=succeeded,
                    safety_failure=bool(outcome["safety_failure"]),
                    high_level_decision_count=0,
                    fallback_decision_count=0,
                    environment_step_count=int(outcome["environment_step_count"]),
                    isaac_backed=True,
                    full_mesh_evaluation=True,
                    source_artifact_path=str(raw_path),
                    source_artifact_sha256=raw_sha,
                    failure_reason=outcome["failure_reason"],
                    metrics={
                        "episode_return": float(outcome["episode_return"]),
                        "terminal_phase_index": float(outcome["terminal_phase_index"]),
                        "hard_collision": float(outcome["hard_collision"]),
                        "object_dropped": float(outcome["object_dropped"]),
                        "qp_infeasible_terminal": float(
                            outcome["qp_infeasible_terminal"]
                        ),
                        "timeout": float(outcome["timeout"]),
                        "contact_preload_initialized": float(
                            outcome["contact_preload_initialized"]
                        ),
                        "contact_preload_complete": float(
                            outcome["contact_preload_complete"]
                        ),
                        "contact_preload_frozen_anchor_count": float(
                            outcome["contact_preload_frozen_anchor_count"]
                        ),
                        "contact_preload_minimum_load_nm": float(
                            outcome["contact_preload_minimum_load_nm"]
                        ),
                        "contact_preload_minimum_signed_normal_force_n": float(
                            outcome["contact_preload_minimum_signed_normal_force_n"]
                        ),
                        "contact_preload_minimum_required_normal_force_n": float(
                            outcome["contact_preload_minimum_required_normal_force_n"]
                        ),
                        "estimated_normal_force_n_minimum": float(
                            outcome.get("estimated_normal_force_n_minimum", 0.0)
                        ),
                        "required_normal_force_n_maximum": float(
                            outcome.get("required_normal_force_n_maximum", 0.0)
                        ),
                        "force_estimator_confidence_minimum": float(
                            outcome.get("force_estimator_confidence_minimum", 0.0)
                        ),
                        "force_estimator_fit_residual_nm": float(
                            outcome.get("force_estimator_fit_residual_nm", 0.0)
                        ),
                        "privileged_actual_normal_force_n_minimum": float(
                            outcome.get("privileged_actual_normal_force_n_minimum", 0.0)
                        ),
                    },
                    metadata={
                        "environment_index": environment,
                        "generation_id": args_cli.generation_id,
                        "deterministic_policy": True,
                        "initial_phase_index": int(initial_phase_values[environment]),
                        "initial_phase_stratum_index": int(
                            reset_stratum[environment].item()
                        ),
                        "formal_phase_zero_start": bool(
                            args_cli.formal_phase_zero_start
                        ),
                        "formal_phase_zero_state_source": (
                            ORDER9_C3_FORMAL_PHASE_ZERO_STATE_SOURCE
                            if args_cli.formal_phase_zero_start
                            else None
                        ),
                        "initial_phase_elapsed_s": (
                            0.0 if args_cli.formal_phase_zero_start else None
                        ),
                        "first_terminal_only": True,
                        "first_terminal_or_horizon_timeout": True,
                        "evaluation_horizon_timeout_contract": (
                            ORDER9_EVALUATION_HORIZON_TIMEOUT_CONTRACT
                        ),
                        "evaluation_horizon_timeout": bool(
                            outcome.get("evaluation_horizon_timeout", False)
                        ),
                        "raw_contact_actor_input": False,
                        "payload_feedforward_phase_contract": (
                            ORDER9_PAYLOAD_FEEDFORWARD_PHASE_CONTRACT
                        ),
                        "contact_acquisition_success_contract": (
                            ORDER9_CONTACT_ACQUISITION_SUCCESS_CONTRACT
                        ),
                        "wrench_range_gate_scale": float(
                            args_cli.c3_wrench_gate_range_scale
                        ),
                        "wrench_range_hard_gate_enabled": False,
                        "exact_wrench_reward_bounds_preserved": True,
                        "c3_privileged_phase_supervision_contract": (
                            ORDER9_C3_PRIVILEGED_PHASE_SUPERVISION_CONTRACT
                        ),
                        "release_success_contract": (ORDER9_RELEASE_SUCCESS_CONTRACT),
                        "release_joint_position_tolerance_rad": float(
                            ORDER9_RELEASE_JOINT_POSITION_TOLERANCE_RAD
                        ),
                        "object_pose_success_contract": (
                            ORDER9_OBJECT_POSE_SUCCESS_CONTRACT
                        ),
                        "object_position_tolerance_m": float(
                            ORDER9_OBJECT_POSITION_TOLERANCE_M
                        ),
                        "c3_boundary_tail_sampling_contract": (
                            ORDER9_C3_BOUNDARY_TAIL_SAMPLING_CONTRACT
                        ),
                        "controller_side_contact_preload_version": (None),
                        "anchor_normal_force_estimator_version": (
                            ORDER9_ANCHOR_NORMAL_FORCE_ESTIMATOR_VERSION
                        ),
                        "anchor_reaction_direction_contract": (
                            ORDER9_ANCHOR_REACTION_DIRECTION_CONTRACT
                        ),
                        "anchor_normal_force_estimator_raw_contact_input": False,
                    },
                )
            )
        write_order9_evaluation_episodes_jsonl(
            args_cli.evaluation_jsonl, evaluation_episodes
        )
        evaluation_episode_count = len(evaluation_episodes)
    finite = bool(
        torch.isfinite(state.robot_root_pose_world).all()
        and torch.isfinite(state.object_pose_world).all()
    )
    result = {
        "passed": finite,
        "generation_id": args_cli.generation_id,
        "stage_id": stage.stage_id,
        "split": split.value,
        "raw_artifact_path": str(Path(args_cli.output_raw).resolve()),
        "raw_artifact_sha256": raw_sha,
        "environment_count": scene.num_envs,
        "rollout_steps": int(artifact.step_count),
        "requested_rollout_steps": int(args_cli.rollout_steps),
        "environment_steps": artifact.environment_step_count,
        "wall_elapsed_s": rollout_elapsed,
        "setup_wall_elapsed_s": setup_elapsed,
        "collection_wall_elapsed_s": setup_elapsed + rollout_elapsed,
        "aggregate_env_steps_per_s": artifact.environment_step_count / rollout_elapsed,
        "end_to_end_env_steps_per_s": artifact.environment_step_count
        / (setup_elapsed + rollout_elapsed),
        "configured_environment_count": configured_runtime.environment_count,
        "configured_rollout_steps_per_environment": (
            configured_runtime.rollout_steps_per_environment
        ),
        "runtime_override_used": runtime_override_used,
        "c3_train_rollout_mode": (
            ORDER9_C3_STATE_INHERITANCE_ROLLOUT_MODE
            if state_inheritance_shard
            else ORDER9_C3_PHASE_RESET_ROLLOUT_MODE
        ),
        "state_inheritance_initial_phase_indices": (
            list(config.production_runtime.c3_state_inheritance_initial_phase_indices)
            if state_inheritance_shard
            else None
        ),
        "state_inheritance_preserves_phase_transitions": bool(state_inheritance_shard),
        "state_inheritance_initial_stratum_policy": (
            state_inheritance_stratum_policy
            if state_inheritance_shard
            else None
        ),
        "state_inheritance_fixed_reset_progress_fraction": (
            state_inheritance_fixed_progress
            if state_inheritance_shard
            else None
        ),
        "state_inheritance_transition_backward_curriculum_contract": (
            ORDER9_C3_TRANSITION_BACKWARD_CURRICULUM_CONTRACT
            if state_inheritance_shard
            and state_inheritance_fixed_progress is not None
            else None
        ),
        "state_inheritance_recurrent_state_reset_only_at_episode_boundary": bool(
            state_inheritance_shard
        ),
        "topology_stratified_shard": {
            "enabled": bool(topology_stratified_shard),
            "index": args_cli.production_topology_shard_index,
            "count": args_cli.production_topology_shard_count,
            "environment_count": (args_cli.production_topology_shard_environment_count),
            "configured_total_environment_count": (
                configured_runtime.environment_count
            ),
        },
        "runtime_load": {
            name: value for name, value in runtime_load.items() if name != "samples"
        },
        "terminal_count": terminal_count,
        "successful_terminal_count": success_count,
        "phase_transition_counts": dict(phase_transition_counts),
        "evaluation_mode": evaluation_mode,
        "training_only_continuous_teacher_rollout": bool(
            args_cli.training_only_continuous_teacher_rollout
        ),
        "promotion_evidence_eligible": bool(
            evaluation_mode
            and not args_cli.training_only_continuous_teacher_rollout
            and args_cli.virtual_contact_lead_mm is None
            and diagnostic_virtual_contact_additional_lead_sweep_mm is None
            and diagnostic_contact_normal_residual_sweep_mm is None
        ),
        "wrench_range_gate_scale": float(args_cli.c3_wrench_gate_range_scale),
        "wrench_range_hard_gate_enabled": False,
        "exact_wrench_reward_bounds_preserved": True,
        "evaluation_episode_count": evaluation_episode_count,
        "evaluation_horizon_timeout_contract": (
            ORDER9_EVALUATION_HORIZON_TIMEOUT_CONTRACT if evaluation_mode else None
        ),
        "evaluation_horizon_timeout_count": int(evaluation_horizon_timeout_count),
        "evaluation_jsonl": (
            None
            if args_cli.evaluation_jsonl is None
            else str(Path(args_cli.evaluation_jsonl).resolve())
        ),
        "deterministic_policy": deterministic_policy,
        "initial_phase_zero": initial_phase_zero,
        "formal_phase_zero_start": bool(args_cli.formal_phase_zero_start),
        "formal_phase_zero_state_source": (
            ORDER9_C3_FORMAL_PHASE_ZERO_STATE_SOURCE
            if args_cli.formal_phase_zero_start
            else None
        ),
        "initial_phase_elapsed_s": (
            0.0 if args_cli.formal_phase_zero_start else None
        ),
        "diagnostic_initial_phase_index": (args_cli.diagnostic_initial_phase_index),
        "diagnostic_initial_phase_stratum_index": (
            args_cli.diagnostic_initial_phase_stratum_index
        ),
        "diagnostic_deterministic_all_phases": bool(
            args_cli.diagnostic_deterministic_policy
            and args_cli.diagnostic_initial_phase_index is None
        ),
        "diagnostic_nominal_qpid_only": bool(args_cli.diagnostic_nominal_qpid_only),
        "diagnostic_action_ablation": args_cli.diagnostic_action_ablation,
        "diagnostic_virtual_contact_additional_lead_sweep_mm": (
            None
            if diagnostic_virtual_contact_additional_lead_sweep_mm is None
            else list(diagnostic_virtual_contact_additional_lead_sweep_mm)
        ),
        "training_nominal_preload_deficit_version": (
            ORDER9_TRAINING_NOMINAL_PRELOAD_DEFICIT_VERSION
            if (
                training_nominal_preload_deficit_mm is not None
                and len(morphology.modules)
                == args_cli.training_nominal_preload_deficit_module_count
            )
            else None
        ),
        "training_nominal_preload_deficit_module_count": (
            args_cli.training_nominal_preload_deficit_module_count
            if training_nominal_preload_deficit_mm is not None
            else None
        ),
        "training_nominal_preload_deficit_mm": (
            None
            if training_nominal_preload_deficit_mm is None
            else list(training_nominal_preload_deficit_mm)
        ),
        "training_nominal_preload_deficit_m_by_environment": (
            None
            if training_nominal_preload_deficit_plan is None
            else list(training_nominal_preload_deficit_plan.deficit_m_by_environment)
        ),
        "training_nominal_preload_total_lead_m_by_environment": (
            None
            if training_nominal_preload_deficit_plan is None
            else list(training_nominal_preload_deficit_plan.total_lead_m_by_environment)
        ),
        "diagnostic_virtual_contact_total_lead_m_by_environment": (
            None
            if diagnostic_virtual_contact_total_lead_m_by_environment is None
            else list(diagnostic_virtual_contact_total_lead_m_by_environment)
        ),
        "diagnostic_contact_normal_residual_sweep_mm": (
            None
            if diagnostic_contact_normal_residual_sweep_mm is None
            else list(diagnostic_contact_normal_residual_sweep_mm)
        ),
        "diagnostic_contact_normal_residual_m_by_environment": (
            None
            if diagnostic_contact_normal_residual_m_by_environment is None
            else diagnostic_contact_normal_residual_m_by_environment.tolist()
        ),
        "diagnostic_terminal_runtime_phase": (
            Order9ObjectTaskPhase.LIFT.value
            if diagnostic_virtual_contact_additional_lead_sweep_mm is not None
            else None
        ),
        "c3_action_contract": args_cli.c3_action_contract,
        "c3_nominal_only_action_mask_contract": (
            ORDER9_NOMINAL_ONLY_ACTION_MASK_CONTRACT
            if (
                c3_nominal_reference is not None
                and args_cli.c3_action_contract is None
            )
            else None
        ),
        "c3_nominal_only_action_mask_phase_labels": (
            [
                Order9ObjectTaskPhase.RELEASE.value,
                Order9ObjectTaskPhase.RETREAT.value,
            ]
            if (
                c3_nominal_reference is not None
                and args_cli.c3_action_contract is None
            )
            else []
        ),
        "contact_compression_only_action_applied": bool(
            policy_runtime.builder.module_count
            in config.production_runtime.c3_contact_compression_only_module_counts
            and args_cli.diagnostic_action_ablation is None
            and args_cli.c3_action_contract is None
        ),
        "joint_only_action_applied": bool(
            policy_runtime.builder.module_count
            in config.production_runtime.c3_joint_only_module_counts
            and args_cli.diagnostic_action_ablation is None
            and args_cli.c3_action_contract is None
        ),
        "diagnostic_contact_sweep_json": (
            None
            if args_cli.diagnostic_contact_sweep_json is None
            else str(Path(args_cli.diagnostic_contact_sweep_json).resolve())
        ),
        "virtual_contact_compression_version": (
            ORDER9_VIRTUAL_CONTACT_COMPRESSION_VERSION
            if virtual_contact_solution is not None
            else None
        ),
        "virtual_contact_inward_lead_m": virtual_contact_lead_m,
        "actuator_aware_nominal_preload_version": (
            ORDER9_ACTUATOR_AWARE_NOMINAL_PRELOAD_VERSION
            if actuator_aware_nominal_preload_solution is not None
            else None
        ),
        "actuator_aware_nominal_preload": (
            None
            if actuator_aware_nominal_preload_solution is None
            else actuator_aware_nominal_preload_solution.to_dict()
        ),
        "actuator_aware_nominal_preload_collision_safe_scale": (
            None
            if nominal_preload_collision_limit is None
            else float(nominal_preload_collision_limit.maximum_action_scale)
        ),
        "actuator_aware_nominal_preload_applied_inward_lead_m": (
            nominal_preload_applied_inward_lead_m
        ),
        "contact_compression_action_adapter_version": (
            _contact_compression_action_adapter_version(
                policy_runtime.policy.policy_version,
                c3_action_contract=args_cli.c3_action_contract,
            )
            if contact_compression_action_solution is not None
            else None
        ),
        "contact_compression_action_span_m": (
            None
            if contact_compression_action_solution is None
            else float(contact_compression_action_solution.inward_lead_m)
        ),
        "contact_compression_action_control_joint_id": (
            contact_compression_action_control_joint_id
        ),
        "contact_compression_collision_shield_version": (
            None
            if contact_compression_collision_limit is None
            else ORDER9_CONTACT_COMPRESSION_COLLISION_SHIELD_VERSION
        ),
        "contact_compression_collision_safe_action_scale": (
            None
            if contact_compression_collision_limit is None
            else float(contact_compression_collision_limit.maximum_action_scale)
        ),
        "contact_compression_collision_safe_span_m": (
            None
            if contact_compression_collision_limit is None
            else float(contact_compression_action_solution.inward_lead_m)
            * float(contact_compression_collision_limit.maximum_action_scale)
        ),
        "contact_compression_collision_limiting_scene_id": (
            None
            if contact_compression_collision_limit is None
            else contact_compression_collision_limit.limiting_scene_id
        ),
        "contact_compression_action_raw_contact_input": False,
        "tensorboard_enabled": tensorboard_logger is not None,
        "tensorboard_log_dir": (
            None if tensorboard_log_dir is None else str(tensorboard_log_dir)
        ),
        "tensorboard_update_index": tensorboard_update_index,
        "tensorboard_logger_version": (
            ORDER9_TENSORBOARD_LOGGER_VERSION
            if tensorboard_logger is not None
            else None
        ),
        "canonical_phase_resets": canonical_resets,
        "phase_specific_resets_available": phase_specific_resets_available,
        "phase_reset_reference_version": (
            ORDER9_C3_PHASE_RESET_REFERENCE_VERSION
            if c3_phase_reset_expected is not None
            else None
        ),
        "c3_contact_reset_stabilization": c3_reset_stabilization,
        "phase_reset_stratum_count": int(bank.stratum_count),
        "c3_boundary_tail_sampling_contract": (
            ORDER9_C3_BOUNDARY_TAIL_SAMPLING_CONTRACT
            if c3_phase_reset_expected is not None
            else None
        ),
        "controller_side_contact_preload_version": (None),
        "deployable_phase_gate_version": (ORDER9_DEPLOYABLE_PHASE_GATE_VERSION),
        "phase_supervision_source": (
            "privileged_physx_contact_outcome"
            if c3_privileged_phase_supervision
            else "deployable_phase_gate"
        ),
        "c3_privileged_phase_supervision_contract": (
            ORDER9_C3_PRIVILEGED_PHASE_SUPERVISION_CONTRACT
            if c3_privileged_phase_supervision
            else None
        ),
        "anchor_normal_force_estimator_version": (
            ORDER9_ANCHOR_NORMAL_FORCE_ESTIMATOR_VERSION
        ),
        "anchor_reaction_direction_contract": (
            ORDER9_ANCHOR_REACTION_DIRECTION_CONTRACT
        ),
        "anchor_normal_force_estimator_raw_contact_input": False,
        "c3_boundary_tail_phase_labels": (
            list(config.production_runtime.c3_boundary_tail_phase_labels)
            if c3_phase_reset_expected is not None
            else None
        ),
        "c3_boundary_tail_progress_fraction": (
            float(config.production_runtime.c3_boundary_tail_progress_fraction)
            if c3_phase_reset_expected is not None
            else None
        ),
        "phase_reset_selectable_strata_by_phase": (
            [list(indices) for indices in reset_strata_by_phase]
            if c3_phase_reset_expected is not None
            else None
        ),
        "unlocked_phase_indices": [
            index
            for index in range(phase_count)
            if bool(bank.available[:, index].any())
        ],
        "raw_contact_actor_input": False,
        "payload_feedforward_phase_contract": (
            ORDER9_PAYLOAD_FEEDFORWARD_PHASE_CONTRACT
        ),
        "contact_acquisition_success_contract": (
            ORDER9_CONTACT_ACQUISITION_SUCCESS_CONTRACT
        ),
        "release_success_contract": ORDER9_RELEASE_SUCCESS_CONTRACT,
        "release_joint_position_tolerance_rad": float(
            ORDER9_RELEASE_JOINT_POSITION_TOLERANCE_RAD
        ),
        "object_pose_success_contract": ORDER9_OBJECT_POSE_SUCCESS_CONTRACT,
        "object_position_tolerance_m": float(ORDER9_OBJECT_POSITION_TOLERANCE_M),
        "finite_state": finite,
    }
    sim.stop()
    sim.clear_instance()
    if not finite:
        raise RuntimeError("Order9 rollout produced non-finite physical state")
    return result


def _next_ppo_update_index(metadata: dict[str, object]) -> int:
    raw = metadata.get("ppo_update_index")
    if raw is None:
        return 0
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < -1:
        raise ValueError("Order9 parent checkpoint PPO update index is invalid")
    return raw + 1


def _tensorboard_log_dir(
    repository: Path,
    *,
    artifact_root: str,
    stage_id: str,
    override: str | None,
) -> Path:
    value = (
        Path(override)
        if override is not None
        else Path(artifact_root) / "stages" / stage_id / "tensorboard"
    )
    return (
        (repository / value).resolve() if not value.is_absolute() else value.resolve()
    )


def _load_task(repository: Path, config, canonical) -> TaskSpec:
    if args_cli.task_spec_json:
        task = TaskSpec.from_json(
            Path(args_cli.task_spec_json).read_text(encoding="utf-8")
        )
        task.validate()
        return task
    return build_order8_grasp_carry_task_spec(
        object_pose_world=tuple(canonical.object_pose_world),
        object_size_m=(0.30, 0.40, 0.15),
        object_mass_kg=1.0,
        object_friction=0.6,
        required_transport_distance_m=canonical.transport_distance_m,
        support_height_m=config.randomization.support_top_z_m,
        max_contact_force_n=config.hard_checker.qp_force_scale_n,
        max_contact_torque_nm=config.hard_checker.qp_torque_scale_nm,
        selected_gripper_friction=(
            config.randomization.nominal_selected_gripper_friction
        ),
        task_id="order9-vectorized-nominal",
    )


def _target_object_and_geometry(task: TaskSpec):
    target_id = next(
        goal.target_entity_id for goal in task.goals if goal.goal_type == "object_pose"
    )
    obj = next(value for value in task.scene.objects if value.object_id == target_id)
    geometry = next(
        value
        for value in task.scene.geometry_library
        if value.geometry_id == obj.geometry_id
    )
    return obj, geometry


def _geometry_values(geometry):
    params = dict(geometry.primitive_params or {})
    if geometry.geometry_type == GeometryType.BOX:
        size = tuple(float(value) for value in params["size_m"])
        return "box", size, 0.5 * size[2]
    if geometry.geometry_type == GeometryType.SPHERE:
        radius = float(params["radius_m"])
        return "sphere", (radius,), radius
    if geometry.geometry_type == GeometryType.CYLINDER:
        radius, height = float(params["radius_m"]), float(params["height_m"])
        return "cylinder", (radius, height), 0.5 * height
    if geometry.geometry_type == GeometryType.CAPSULE:
        radius, height = float(params["radius_m"]), float(params["height_m"])
        return "capsule", (radius, height), 0.5 * height + radius
    raise ValueError("Order9 rollout currently supports primitive object geometries")


def _collision_properties(*, contact_offset_m, rest_offset_m):
    return sim_utils.CollisionPropertiesCfg(
        contact_offset=contact_offset_m,
        rest_offset=rest_offset_m,
    )


def _object_spawn(
    kind,
    values,
    *,
    mass,
    friction,
    collision_contact_offset_m,
    collision_rest_offset_m,
):
    common = dict(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=2.0,
            enable_gyroscopic_forces=True,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=float(mass)),
        collision_props=_collision_properties(
            contact_offset_m=collision_contact_offset_m,
            rest_offset_m=collision_rest_offset_m,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=float(friction),
            dynamic_friction=float(friction),
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.38, 0.12)),
        activate_contact_sensors=True,
    )
    if kind == "box":
        return sim_utils.CuboidCfg(size=values, **common)
    if kind == "sphere":
        return sim_utils.SphereCfg(radius=values[0], **common)
    if kind == "cylinder":
        return sim_utils.CylinderCfg(radius=values[0], height=values[1], **common)
    return sim_utils.CapsuleCfg(radius=values[0], height=values[1], **common)


def _scene_cfg(
    *,
    robot_usd: Path,
    object_kind: str,
    geometry_values: tuple[float, ...],
    object_pose,
    object_mass: float,
    object_friction: float,
    support_size,
    support_pose,
    robot_body_names,
    actuator_runtime: Order9ActuatorRuntimeValues,
    collision_contact_offset_m: float | None,
    collision_rest_offset_m: float | None,
):
    cfg = _Order9RolloutSceneCfg(num_envs=1, env_spacing=3.0)
    cfg.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(robot_usd),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=2.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=8,
                sleep_threshold=0.0,
                stabilization_threshold=0.001,
            ),
            copy_from_source=False,
            collision_props=_collision_properties(
                contact_offset_m=collision_contact_offset_m,
                rest_offset_m=collision_rest_offset_m,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.65),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "gimbal_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*gimbal.*"],
                stiffness=actuator_runtime.gimbal_stiffness,
                damping=actuator_runtime.gimbal_damping,
                armature=actuator_runtime.gimbal_armature,
                effort_limit_sim=actuator_runtime.gimbal_effort_limit,
                velocity_limit_sim=actuator_runtime.gimbal_velocity_limit,
            ),
            "dock_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*dock_mech.*"],
                stiffness=actuator_runtime.dock_stiffness,
                damping=actuator_runtime.dock_damping,
                armature=actuator_runtime.dock_armature,
                effort_limit_sim=actuator_runtime.dock_effort_limit,
                velocity_limit_sim=actuator_runtime.dock_velocity_limit,
            ),
            "rotor_spinner_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*rotor.*"], stiffness=0.0, damping=0.0
            ),
        },
    )
    cfg.object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=_object_spawn(
            object_kind,
            geometry_values,
            mass=object_mass,
            friction=object_friction,
            collision_contact_offset_m=collision_contact_offset_m,
            collision_rest_offset_m=collision_rest_offset_m,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=tuple(float(value) for value in object_pose[:3]),
            rot=tuple(float(value) for value in object_pose[3:7]),
        ),
    )
    cfg.support = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Support",
        spawn=sim_utils.CuboidCfg(
            size=tuple(float(value) for value in support_size),
            collision_props=_collision_properties(
                contact_offset_m=collision_contact_offset_m,
                rest_offset_m=collision_rest_offset_m,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.18, 0.18, 0.18)
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=tuple(float(value) for value in support_pose[:3]),
            rot=tuple(float(value) for value in support_pose[3:7]),
        ),
    )
    cfg.light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )
    cfg.robot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        debug_vis=False,
        max_contact_data_count_per_prim=16,
    )
    cfg.object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        update_period=0.0,
        debug_vis=False,
        max_contact_data_count_per_prim=max(32, 4 * len(robot_body_names)),
        filter_prim_paths_expr=[
            f"{{ENV_REGEX_NS}}/Robot/{name}" for name in robot_body_names
        ],
    )
    return cfg


def _teacher_assignments(task: TaskSpec, morphology: MorphologyGraph):
    built = IRGBuilder().build_with_scene_graph(task)
    envelope = InteractionEnvelopeExtractor().extract(built.irg)
    candidates = ContactCandidateSampler().sample(
        task_spec=task,
        irg=built.irg,
        interaction_envelope=envelope,
        morphology_graph=morphology,
        geometry_descriptors=built.scene_graph.geometry_descriptors,
    )
    context = HighLevelPolicyContext(built.irg, envelope, morphology, candidates)
    trajectory = upgrade_teacher_trajectory_to_v2(
        GraspCarryBaselinePlanner().plan(context), context
    )
    maintain = next(
        knot
        for knot in trajectory.knots
        if any(
            assignment.schedule_state == "maintain"
            for assignment in knot.contact_assignments
        )
    )
    assignments = sorted(
        (
            assignment
            for assignment in maintain.contact_assignments
            if assignment.schedule_state == "maintain"
        ),
        key=lambda value: value.anchor_id,
    )
    return trajectory, assignments, candidates


def _maintain_assignments(trajectory):
    maintain = next(
        knot
        for knot in trajectory.knots
        if any(
            assignment.schedule_state == "maintain"
            for assignment in knot.contact_assignments
        )
    )
    return sorted(
        (
            assignment
            for assignment in maintain.contact_assignments
            if assignment.schedule_state == "maintain"
        ),
        key=lambda value: value.anchor_id,
    )


def _bind_selected_material(
    stage,
    *,
    selected_body_names,
    friction: float,
    stiffness: float,
    damping: float,
) -> None:
    material_path = "/World/Order9SelectedGripperMaterial"
    cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=friction,
        dynamic_friction=friction,
        restitution=0.0,
        compliant_contact_stiffness=stiffness,
        compliant_contact_damping=damping,
        friction_combine_mode="max",
    )
    cfg.func(material_path, cfg)
    material = UsdShade.Material(stage.GetPrimAtPath(Sdf.Path(material_path)))
    selected = set(selected_body_names)
    count = 0
    for env_id in range(int(args_cli.num_envs)):
        root = stage.GetPrimAtPath(Sdf.Path(f"/World/envs/env_{env_id}/Robot"))
        if not root.IsValid():
            raise RuntimeError("Order9 rollout robot prim is invalid")
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if prim.GetName() not in selected or not prim.HasAPI(
                UsdPhysics.RigidBodyAPI
            ):
                continue
            api = (
                UsdShade.MaterialBindingAPI(prim)
                if prim.HasAPI(UsdShade.MaterialBindingAPI)
                else UsdShade.MaterialBindingAPI.Apply(prim)
            )
            api.Bind(
                material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )
            count += 1
    expected_minimum = int(args_cli.num_envs) * len(selected)
    if count < expected_minimum:
        raise RuntimeError(
            f"Order9 selected material bound {count}, expected at least {expected_minimum}"
        )


def _activate_nested_contact_reports(stage) -> None:
    count = 0
    for env_id in range(int(args_cli.num_envs)):
        root = stage.GetPrimAtPath(Sdf.Path(f"/World/envs/env_{env_id}/Robot"))
        if not root.IsValid():
            raise RuntimeError("Order9 rollout robot prim is invalid")
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
            count += 1
    if count == 0:
        raise RuntimeError("Order9 rollout found no robot contact-report body")


def _canonical_resets_enabled(morphology, source_graph_hash: str) -> bool:
    matches = morphology.stable_hash() == str(source_graph_hash)
    if args_cli.canonical_phase_resets == "yes" and not matches:
        raise ValueError(
            "canonical phase resets cannot be applied to another morphology"
        )
    if args_cli.canonical_phase_resets == "no":
        return False
    return matches


def _seed_canonical_bank(
    bank: _PhaseStateBank,
    scene: InteractiveScene,
    *,
    canonical,
    task: TaskSpec,
    robot_joint_names,
) -> None:
    target_object, _ = _target_object_and_geometry(task)
    canonical_position = canonical.object_pose_world[:3]
    offset = tuple(
        float(target_object.pose_world[index]) - float(canonical_position[index])
        for index in range(3)
    )
    yaw_offset = _yaw(target_object.pose_world[3:7]) - _yaw(
        canonical.object_pose_world[3:7]
    )
    runtime = Order9ObjectTaskRuntime(canonical)
    robot = scene["robot"]
    default_q = _torch(robot.data.default_joint_pos)
    joint_lookup = {name: index for index, name in enumerate(robot_joint_names)}
    device = torch.device(scene.device)
    dtype = default_q.dtype
    for env_id in range(scene.num_envs):
        for phase_index in range(len(ORDER9_OBJECT_TASK_PHASES)):
            reset = runtime.reset_for_phase(
                phase_index,
                object_position_offset_world=offset,
                object_yaw_offset_rad=yaw_offset,
            )
            q = default_q[env_id].clone()
            qdot = torch.zeros_like(q)
            for global_id, value in reset.joint_positions_rad.items():
                name = global_id.replace(":", "__", 1)
                if name in joint_lookup:
                    q[joint_lookup[name]] = float(value)
            bank.install(
                env_id=env_id,
                phase_index=phase_index,
                robot_root_pose_local=torch.tensor(
                    reset.robot_root_pose_world, device=device, dtype=dtype
                ),
                robot_root_twist=torch.zeros(6, device=device, dtype=dtype),
                joint_position=q,
                joint_velocity=qdot,
                object_pose_local=torch.tensor(
                    reset.object_pose_world, device=device, dtype=dtype
                ),
                object_twist=torch.zeros(6, device=device, dtype=dtype),
            )


def _install_teacher_phase_zero(
    bank: _PhaseStateBank,
    scene: InteractiveScene,
    *,
    io: Order9TensorIsaacIO,
    teacher_reference,
    robot_joint_names: tuple[str, ...],
    task_object_pose_world: tuple[float, ...],
) -> dict[str, torch.Tensor]:
    """Install the exact successful C0 phase-zero physical distribution.

    C0 used one articulation per module, whereas the tensor runtime uses one
    rigid fixed-morphology articulation.  Align its root through module 0's
    physical-model frame, then fail closed later if every module does not land
    on the C0 geometry within the rigid-assembly tolerance.
    """

    robot = scene["robot"]
    device = torch.device(scene.device)
    if tuple(teacher_reference.module_ids) != tuple(io.module_ids):
        raise RuntimeError(
            "Order9 C0 initial module identity differs from the fixed USD"
        )
    default_q = _torch(robot.data.default_joint_pos)
    dtype = default_q.dtype
    source_object_position = teacher_reference.initial_object_pose_world[:3]
    task_offset = torch.tensor(
        task_object_pose_world[:3], device=device, dtype=dtype
    ) - source_object_position.to(device=device, dtype=dtype)

    current_root = _torch(robot.data.root_pose_w)[0].detach().cpu().tolist()
    module_zero_index = io.module_ids.index(0)
    current_module_zero = _torch(robot.data.body_pose_w)[
        0, io.module_body_indices[module_zero_index]
    ]
    # The caller validates body identity before this helper.  Remove the copied
    # environment origin so scalar pose composition remains in local world.
    origin_zero = scene.env_origins[0]
    current_root[:3] = (
        (torch.tensor(current_root[:3], device=device, dtype=dtype) - origin_zero)
        .detach()
        .cpu()
        .tolist()
    )
    current_module_zero_local = current_module_zero.clone()
    current_module_zero_local[:3] -= origin_zero
    root_to_module_zero = compose_pose(
        inverse_pose(tuple(float(value) for value in current_root)),
        tuple(float(value) for value in current_module_zero_local.detach().cpu()),
    )

    initial_module_pose = teacher_reference.initial_module_pose_world.to(
        device=device, dtype=dtype
    ).clone()
    initial_module_pose[:, :3] += task_offset
    desired_root_pose = compose_pose(
        tuple(float(value) for value in initial_module_pose[0].detach().cpu()),
        inverse_pose(root_to_module_zero),
    )
    module_zero_twist = teacher_reference.initial_module_twist_world[0].to(
        device=device, dtype=dtype
    )
    root_position = torch.tensor(desired_root_pose[:3], device=device, dtype=dtype)
    root_to_module_world = initial_module_pose[0, :3] - root_position
    root_linear_velocity = module_zero_twist[:3] - torch.linalg.cross(
        module_zero_twist[3:], root_to_module_world, dim=-1
    )
    root_twist = torch.cat((root_linear_velocity, module_zero_twist[3:]))

    joint_lookup = {name: index for index, name in enumerate(robot_joint_names)}
    missing = sorted(
        set(teacher_reference.initial_joint_positions_rad) - set(joint_lookup)
    )
    if missing:
        raise RuntimeError(
            f"Order9 C0 initial joints are missing from fixed USD: {missing}"
        )
    initial_object_pose = teacher_reference.initial_object_pose_world.to(
        device=device, dtype=dtype
    ).clone()
    initial_object_pose[:3] += task_offset
    initial_object_twist = teacher_reference.initial_object_twist_world.to(
        device=device, dtype=dtype
    )
    for env_id in range(scene.num_envs):
        q = default_q[env_id].clone()
        qdot = torch.zeros_like(q)
        for name, value in teacher_reference.initial_joint_positions_rad.items():
            q[joint_lookup[name]] = float(value)
            qdot[joint_lookup[name]] = float(
                teacher_reference.initial_joint_velocities_radps[name]
            )
        bank.install(
            env_id=env_id,
            phase_index=0,
            robot_root_pose_local=torch.tensor(
                desired_root_pose, device=device, dtype=dtype
            ),
            robot_root_twist=root_twist,
            joint_position=q,
            joint_velocity=qdot,
            object_pose_local=initial_object_pose,
            object_twist=initial_object_twist,
        )

    origins = scene.env_origins.to(dtype=dtype)
    expected_module_pose = (
        initial_module_pose.unsqueeze(0).expand(scene.num_envs, -1, -1).clone()
    )
    expected_module_pose[:, :, :3] += origins.unsqueeze(1)
    expected_object_pose = (
        initial_object_pose.unsqueeze(0).expand(scene.num_envs, -1).clone()
    )
    expected_object_pose[:, :3] += origins
    return {
        "module_pose_world": expected_module_pose,
        "module_twist_world": teacher_reference.initial_module_twist_world.to(
            device=device, dtype=dtype
        )
        .unsqueeze(0)
        .expand(scene.num_envs, -1, -1),
        "object_pose_world": expected_object_pose,
        "object_twist_world": initial_object_twist.unsqueeze(0).expand(
            scene.num_envs, -1
        ),
    }


def _validate_teacher_phase_zero_alignment(
    state,
    *,
    expected: dict[str, torch.Tensor],
) -> None:
    module_position_error = torch.linalg.vector_norm(
        state.module_pose_world[..., :3] - expected["module_pose_world"][..., :3],
        dim=-1,
    )
    module_orientation_error = _quaternion_distance_rad(
        state.module_pose_world[..., 3:7],
        expected["module_pose_world"][..., 3:7],
    )
    object_position_error = torch.linalg.vector_norm(
        state.object_pose_world[..., :3] - expected["object_pose_world"][..., :3],
        dim=-1,
    )
    object_orientation_error = _quaternion_distance_rad(
        state.object_pose_world[..., 3:7],
        expected["object_pose_world"][..., 3:7],
    )
    maxima = {
        "module_position_m": float(module_position_error.max().item()),
        "module_orientation_rad": float(module_orientation_error.max().item()),
        "object_position_m": float(object_position_error.max().item()),
        "object_orientation_rad": float(object_orientation_error.max().item()),
    }
    if (
        maxima["module_position_m"] > 2.0e-3
        or maxima["module_orientation_rad"] > 3.0e-3
        or maxima["object_position_m"] > 1.0e-4
        or maxima["object_orientation_rad"] > 1.0e-4
    ):
        raise RuntimeError(
            "Order9 fixed USD cannot reproduce the exact C0 phase-zero "
            f"physical state: {maxima}"
        )


def _quaternion_distance_rad(
    actual: torch.Tensor, expected: torch.Tensor
) -> torch.Tensor:
    # Reset validation often compares identical float32 poses.  Performing the
    # normalization/dot product in float32 gives an acos quantization floor of
    # roughly 1e-3 rad, larger than the physical reset tolerance itself.
    actual = actual.to(torch.float64)
    expected = expected.to(torch.float64)
    actual = actual / actual.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    expected = expected / expected.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    dot = (actual * expected).sum(dim=-1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot)


def _align_arbitrary_phase_zero(
    scene: InteractiveScene,
    *,
    sim: sim_utils.SimulationContext,
    io: Order9TensorIsaacIO,
    candidate_points,
    approach_offset_m: float,
) -> None:
    robot = scene["robot"]
    body_pose = _torch(robot.data.body_pose_w)
    selected = torch.tensor(
        io.selected_anchor_body_indices, device=scene.device, dtype=torch.long
    )
    anchor_centroid = body_pose.index_select(1, selected)[..., :3].mean(dim=1)
    target_local = torch.tensor(
        [pose[:3] for pose in candidate_points],
        device=scene.device,
        dtype=anchor_centroid.dtype,
    ).mean(dim=0)
    target = target_local.unsqueeze(0) + scene.env_origins
    target[:, 0] -= float(approach_offset_m)
    root_pose = _torch(robot.data.root_pose_w).clone()
    root_pose[:, :3] += target - anchor_centroid
    env_ids = torch.arange(scene.num_envs, device=scene.device)
    robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros(
            (scene.num_envs, 6), device=scene.device, dtype=root_pose.dtype
        ),
        env_ids=env_ids,
    )
    sim.forward()
    scene.update(0.0)


def _install_c3_nominal_phase_bank(
    bank: _PhaseStateBank,
    scene: InteractiveScene,
    *,
    sim: sim_utils.SimulationContext,
    io: Order9TensorIsaacIO,
    qpid_runtime: Order9TensorPiLRuntime,
    nominal_reference: Order9C3NominalTensorReference,
    task: TaskSpec,
    object_id: str,
    lift_clearance_m: float,
    retreat_offset_m: float,
    phase_duration_s: Mapping[str, float],
    progress_fractions: Sequence[float],
) -> Order9C3PhaseResetReferences:
    """Install all C3 phase starts from the accepted morphology-specific path."""

    robot = scene["robot"]
    obj = scene["object"]
    env_ids = torch.arange(scene.num_envs, device=scene.device)
    if tuple(nominal_reference.module_ids) != tuple(io.module_ids):
        raise RuntimeError("C3 nominal phase-reset module identity differs")
    object_pose_local = _object_pose(obj).clone()
    object_pose_local[:, :3] -= scene.env_origins
    if not bool(
        torch.allclose(
            object_pose_local,
            object_pose_local[0].unsqueeze(0).expand_as(object_pose_local),
            atol=1.0e-5,
            rtol=0.0,
        )
    ):
        raise RuntimeError("C3 copied environments disagree on object reset pose")
    references = nominal_reference.phase_reset_references(
        object_pose_local=object_pose_local[0],
        lift_clearance_m=float(lift_clearance_m),
        transport_distance_m=_transport_distance(task, object_id),
        retreat_offset_m=float(retreat_offset_m),
        phase_duration_s=phase_duration_s,
        progress_fractions=progress_fractions,
    )
    default_q = _torch(robot.data.default_joint_pos).clone()
    local_index = {joint_id: index for index, joint_id in enumerate(io.local_joint_ids)}
    phase_q = []
    phase_qdot = []
    for phase_index in range(len(ORDER9_OBJECT_TASK_PHASES)):
        phase_q.append([])
        phase_qdot.append([])
        for stratum_index in range(references.stratum_count):
            q = default_q.clone()
            qdot = torch.zeros_like(q)
            reference_q = references.joint_positions_rad[phase_index, stratum_index].to(
                device=scene.device, dtype=q.dtype
            )
            reference_qdot = references.joint_velocities_radps[
                phase_index, stratum_index
            ].to(device=scene.device, dtype=q.dtype)
            for module_row, _module_id in enumerate(io.module_ids):
                for reference_joint, joint_id in enumerate(nominal_reference.joint_ids):
                    if joint_id not in local_index:
                        raise RuntimeError(
                            f"C3 nominal phase-reset joint is unknown: {joint_id}"
                        )
                    robot_index = io.local_joint_indices[module_row][
                        local_index[joint_id]
                    ]
                    if robot_index < 0:
                        raise RuntimeError(
                            f"C3 nominal phase-reset joint was merged: {joint_id}"
                        )
                    q[:, robot_index] = reference_q[module_row, reference_joint]
                    qdot[:, robot_index] = reference_qdot[module_row, reference_joint]
            phase_q[phase_index].append(q)
            phase_qdot[phase_index].append(qdot)

    for phase_index in range(len(ORDER9_OBJECT_TASK_PHASES)):
        for stratum_index in range(references.stratum_count):
            q = phase_q[phase_index][stratum_index]
            qdot = phase_qdot[phase_index][stratum_index]
            robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
            robot.write_joint_velocity_to_sim_index(velocity=qdot, env_ids=env_ids)
            sim.forward()
            scene.update(0.0)
            state = io.gather_state(robot=robot, object_asset=obj)
            control = qpid_runtime.builder.build(
                module_pose_world=state.module_pose_world,
                module_twist_world=state.module_twist_world,
                local_joint_positions_rad=state.local_joint_positions_rad,
            )
            current_root = _torch(robot.data.root_pose_w)
            desired_body_local = references.body_pose_local[phase_index, stratum_index]
            for env_id in range(scene.num_envs):
                root_to_body = compose_pose(
                    inverse_pose(
                        tuple(float(value) for value in current_root[env_id].tolist())
                    ),
                    tuple(
                        float(value)
                        for value in control.body_pose_world[env_id].tolist()
                    ),
                )
                desired_body_world = desired_body_local.clone()
                desired_body_world[:3] += scene.env_origins[env_id]
                desired_root_world = compose_pose(
                    tuple(float(value) for value in desired_body_world.tolist()),
                    inverse_pose(root_to_body),
                )
                desired_root_local = torch.tensor(
                    desired_root_world, device=scene.device, dtype=q.dtype
                )
                desired_root_local[:3] -= scene.env_origins[env_id]
                bank.install(
                    env_id=env_id,
                    phase_index=phase_index,
                    stratum_index=stratum_index,
                    robot_root_pose_local=desired_root_local,
                    robot_root_twist=references.body_twist[
                        phase_index, stratum_index
                    ].to(device=scene.device, dtype=q.dtype),
                    joint_position=q[env_id],
                    joint_velocity=qdot[env_id],
                    object_pose_local=references.object_pose_local[
                        phase_index, stratum_index
                    ].to(device=scene.device, dtype=q.dtype),
                    object_twist=references.object_twist[phase_index, stratum_index].to(
                        device=scene.device, dtype=q.dtype
                    ),
                    phase_elapsed_s=float(
                        references.phase_elapsed_s[phase_index, stratum_index].item()
                    ),
                    task_body_start_pose_local=(
                        references.phase_start_body_pose_local[phase_index].to(
                            device=scene.device, dtype=q.dtype
                        )
                    ),
                    task_object_start_pose_local=(
                        references.phase_start_object_pose_local[phase_index].to(
                            device=scene.device, dtype=q.dtype
                        )
                    ),
                )
    if not bool(bank.available.all()):
        raise RuntimeError("C3 morphology-specific phase-reset bank is incomplete")
    return references


def _restore_c3_formal_phase_zero(
    scene: InteractiveScene,
    *,
    sim: sim_utils.SimulationContext,
    io: Order9TensorIsaacIO,
    qpid_runtime: Order9TensorPiLRuntime,
    nominal_reference: Order9C3NominalTensorReference,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Order9C3PhaseStartReference,
]:
    """Restore the exact accepted approach ``t=0`` state for promotion.

    Phase-reset stratum zero is the first intra-phase training sample (normally
    progress 1/6), not the task start.  Formal end-to-end evidence therefore
    reconstructs the articulation root directly from the accepted nominal
    body/joint state and never indexes the phase-reset bank.
    """

    robot = scene["robot"]
    obj = scene["object"]
    env_ids = torch.arange(scene.num_envs, device=scene.device, dtype=torch.long)
    reference = nominal_reference.phase_start_reference(
        Order9ObjectTaskPhase.APPROACH
    )
    if tuple(nominal_reference.module_ids) != tuple(io.module_ids):
        raise RuntimeError("C3 formal phase-zero module identity differs")

    scene.reset(env_ids)
    q = _torch(robot.data.default_joint_pos).clone()
    qdot = torch.zeros_like(q)
    reference_q = reference.joint_positions_rad.to(device=scene.device, dtype=q.dtype)
    reference_qdot = reference.joint_velocities_radps.to(
        device=scene.device, dtype=q.dtype
    )
    local_index = {joint_id: index for index, joint_id in enumerate(io.local_joint_ids)}
    for module_row, _module_id in enumerate(io.module_ids):
        for reference_joint, joint_id in enumerate(nominal_reference.joint_ids):
            if joint_id not in local_index:
                raise RuntimeError(f"C3 formal phase-zero joint is unknown: {joint_id}")
            robot_index = io.local_joint_indices[module_row][local_index[joint_id]]
            if robot_index < 0:
                raise RuntimeError(f"C3 formal phase-zero joint was merged: {joint_id}")
            q[:, robot_index] = reference_q[module_row, reference_joint]
            qdot[:, robot_index] = reference_qdot[module_row, reference_joint]

    robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=qdot, env_ids=env_ids)
    sim.forward()
    scene.update(0.0)
    state = io.gather_state(robot=robot, object_asset=obj)
    control = qpid_runtime.builder.build(
        module_pose_world=state.module_pose_world,
        module_twist_world=state.module_twist_world,
        local_joint_positions_rad=state.local_joint_positions_rad,
    )
    current_root = _torch(robot.data.root_pose_w)
    root_pose_world = current_root.clone()
    desired_body_world = reference.body_pose_local.to(
        device=scene.device, dtype=q.dtype
    ).unsqueeze(0).expand(scene.num_envs, -1).clone()
    desired_body_world[:, :3] += scene.env_origins
    for env_id in range(scene.num_envs):
        root_to_body = compose_pose(
            inverse_pose(tuple(float(value) for value in current_root[env_id].tolist())),
            tuple(float(value) for value in control.body_pose_world[env_id].tolist()),
        )
        root_pose_world[env_id] = torch.tensor(
            compose_pose(
                tuple(float(value) for value in desired_body_world[env_id].tolist()),
                inverse_pose(root_to_body),
            ),
            device=scene.device,
            dtype=q.dtype,
        )

    body_twist = reference.body_twist.to(device=scene.device, dtype=q.dtype)
    body_twist = body_twist.unsqueeze(0).expand(scene.num_envs, -1).clone()
    object_pose_world = reference.object_pose_local.to(
        device=scene.device, dtype=q.dtype
    ).unsqueeze(0).expand(scene.num_envs, -1).clone()
    object_pose_world[:, :3] += scene.env_origins
    object_twist = reference.object_twist.to(device=scene.device, dtype=q.dtype)
    object_twist = object_twist.unsqueeze(0).expand(scene.num_envs, -1).clone()

    robot.write_root_pose_to_sim_index(root_pose=root_pose_world, env_ids=env_ids)
    robot.write_root_velocity_to_sim_index(root_velocity=body_twist, env_ids=env_ids)
    robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=qdot, env_ids=env_ids)
    robot.set_joint_position_target_index(target=q, env_ids=env_ids)
    robot.set_joint_velocity_target_index(target=torch.zeros_like(qdot), env_ids=env_ids)
    robot.set_joint_effort_target_index(target=torch.zeros_like(qdot), env_ids=env_ids)
    obj.write_root_pose_to_sim_index(root_pose=object_pose_world, env_ids=env_ids)
    obj.write_root_velocity_to_sim_index(root_velocity=object_twist, env_ids=env_ids)
    return (
        torch.zeros(scene.num_envs, device=scene.device, dtype=q.dtype),
        desired_body_world,
        object_pose_world,
        reference,
    )


def _validate_c3_formal_phase_zero_alignment(
    state,
    *,
    control,
    expected: Order9C3PhaseStartReference,
    scene_origins: torch.Tensor,
    io: Order9TensorIsaacIO,
    reference_joint_ids: tuple[str, ...],
) -> None:
    expected_body = expected.body_pose_local.to(
        device=scene_origins.device, dtype=scene_origins.dtype
    ).unsqueeze(0).expand(scene_origins.shape[0], -1).clone()
    expected_body[:, :3] += scene_origins
    expected_object = expected.object_pose_local.to(
        device=scene_origins.device, dtype=scene_origins.dtype
    ).unsqueeze(0).expand(scene_origins.shape[0], -1).clone()
    expected_object[:, :3] += scene_origins
    expected_joint = expected.joint_positions_rad.to(
        device=state.local_joint_positions_rad.device,
        dtype=state.local_joint_positions_rad.dtype,
    ).unsqueeze(0).expand(scene_origins.shape[0], -1, -1)
    reference_indices = torch.tensor(
        [io.local_joint_ids.index(value) for value in reference_joint_ids],
        device=state.local_joint_positions_rad.device,
        dtype=torch.long,
    )
    actual_joint = state.local_joint_positions_rad.index_select(2, reference_indices)
    maxima = {
        "body_position_m": float(
            torch.linalg.vector_norm(
                control.body_pose_world[:, :3] - expected_body[:, :3], dim=-1
            ).max().item()
        ),
        "body_orientation_rad": float(
            _quaternion_distance_rad(
                control.body_pose_world[:, 3:7], expected_body[:, 3:7]
            ).max().item()
        ),
        "object_position_m": float(
            torch.linalg.vector_norm(
                state.object_pose_world[:, :3] - expected_object[:, :3], dim=-1
            ).max().item()
        ),
        "object_orientation_rad": float(
            _quaternion_distance_rad(
                state.object_pose_world[:, 3:7], expected_object[:, 3:7]
            ).max().item()
        ),
        "joint_position_rad": float((actual_joint - expected_joint).abs().max().item()),
    }
    if (
        maxima["body_position_m"] > 2.0e-4
        or maxima["body_orientation_rad"] > 2.0e-4
        or maxima["object_position_m"] > 1.0e-4
        or maxima["object_orientation_rad"] > 1.0e-4
        or maxima["joint_position_rad"] > 1.0e-5
    ):
        raise RuntimeError(f"C3 formal phase-zero installation differs: {maxima}")


def _validate_c3_phase_reset_alignment(
    state,
    *,
    control,
    expected: Order9C3PhaseResetReferences,
    phase_index: torch.Tensor,
    stratum_index: torch.Tensor,
    scene_origins: torch.Tensor,
    io: Order9TensorIsaacIO,
    reference_joint_ids: tuple[str, ...],
) -> None:
    expected_body = expected.body_pose_local[phase_index, stratum_index].clone()
    expected_body[:, :3] += scene_origins
    expected_object = expected.object_pose_local[phase_index, stratum_index].clone()
    expected_object[:, :3] += scene_origins
    expected_joint = expected.joint_positions_rad[phase_index, stratum_index]
    reference_indices = torch.tensor(
        [io.local_joint_ids.index(value) for value in reference_joint_ids],
        device=state.local_joint_positions_rad.device,
        dtype=torch.long,
    )
    actual_joint = state.local_joint_positions_rad.index_select(2, reference_indices)
    if actual_joint.shape != expected_joint.shape:
        raise RuntimeError(
            "C3 phase-reset joint observation shape changed: "
            f"actual={tuple(actual_joint.shape)} "
            f"expected={tuple(expected_joint.shape)}"
        )
    maxima = {
        "body_position_m": float(
            torch.linalg.vector_norm(
                control.body_pose_world[:, :3] - expected_body[:, :3], dim=-1
            )
            .max()
            .item()
        ),
        "body_orientation_rad": float(
            _quaternion_distance_rad(
                control.body_pose_world[:, 3:7], expected_body[:, 3:7]
            )
            .max()
            .item()
        ),
        "object_position_m": float(
            torch.linalg.vector_norm(
                state.object_pose_world[:, :3] - expected_object[:, :3], dim=-1
            )
            .max()
            .item()
        ),
        "object_orientation_rad": float(
            _quaternion_distance_rad(
                state.object_pose_world[:, 3:7], expected_object[:, 3:7]
            )
            .max()
            .item()
        ),
        "joint_position_rad": float((actual_joint - expected_joint).abs().max().item()),
    }
    if (
        maxima["body_position_m"] > 2.0e-4
        or maxima["body_orientation_rad"] > 2.0e-4
        or maxima["object_position_m"] > 1.0e-4
        or maxima["object_orientation_rad"] > 1.0e-4
        or maxima["joint_position_rad"] > 1.0e-5
    ):
        raise RuntimeError(f"C3 phase-reset installation differs: {maxima}")


def _diagnostic_rejected_stabilize_c3_contact_phase_bank(
    bank: _PhaseStateBank,
    scene: InteractiveScene,
    *,
    sim: sim_utils.SimulationContext,
    io: Order9TensorIsaacIO,
    reset_runtime: Order9TensorPiLRuntime,
    task_runtime: Order9TensorObjectTaskRuntime,
    nominal_reference: Order9C3NominalTensorReference,
    reset_references: Order9C3PhaseResetReferences,
    joint_reference_start: torch.Tensor,
    joint_reference_end: torch.Tensor,
    lift_clearance: torch.Tensor,
    transport_distance: torch.Tensor,
    estimated_mass: torch.Tensor,
    estimated_inertia: torch.Tensor,
    estimated_com: torch.Tensor,
    selected_mask: torch.Tensor,
    robot_sensor,
    object_sensor,
    robot_sensor_reorder: torch.Tensor,
    preload_steps: int,
    minimum_selected_contacts: int,
    maximum_object_displacement_m: float,
    maximum_downward_speed_mps: float,
    fixture_free_hold_s: float,
    dt_s: float,
) -> dict[str, object]:
    """Retain the rejected dynamic-reset experiment for offline diagnosis only.

    This function is deliberately not called by the production collector.  The
    accepted C3 contract installs hash-bound physical snapshots from the
    human-accepted nominal trajectory and makes contact maintenance an actor
    rollout/evaluation target, rather than a reset-admission gate.

    Kinematic contact poses can contain zero-force or impulsive configurations.
    Every lift/transport/place intra-phase stratum therefore begins from the
    last 0.1 s of the accepted contact-acquisition path and replays that tail
    over 0.5 s through the production QPID/QP path.  A temporary damped 3-D
    object fixture is active while contact closes.  Its support is then handed
    continuously to payload feedforward before the fixture is removed.  A
    frozen initializer ``pi_L`` then owns the reset-only QPID/QP path; the actor
    being optimized or compared is never consulted.  A state is admitted only
    after contact, collision, QP, displacement, and downward-speed conditions
    have all remained valid for a complete fixture-free hold window.  The exact
    admitted physical state is broadcast to every copied environment and saved
    separately by the caller for byte-identical reuse.
    """

    maintained_phases = (
        ORDER9_OBJECT_TASK_PHASES.index(Order9ObjectTaskPhase.LIFT),
        ORDER9_OBJECT_TASK_PHASES.index(Order9ObjectTaskPhase.TRANSPORT),
        ORDER9_OBJECT_TASK_PHASES.index(Order9ObjectTaskPhase.PLACE),
    )
    entries = tuple(
        (phase_index, stratum_index)
        for phase_index in maintained_phases
        for stratum_index in range(bank.stratum_count)
    )
    if scene.num_envs < len(entries):
        raise RuntimeError(
            "Order9 C3 contact reset stabilization requires at least "
            f"{len(entries)} copied environments; got {scene.num_envs}"
        )
    if preload_steps < 1:
        raise ValueError("Order9 C3 contact reset preload must be positive")
    if not math.isfinite(float(fixture_free_hold_s)) or fixture_free_hold_s <= 0.0:
        raise ValueError("Order9 C3 fixture-free reset hold must be positive")

    device = torch.device(scene.device)
    env_ids = torch.arange(scene.num_envs, device=device, dtype=torch.long)
    entry_index = env_ids % len(entries)
    phase_index = torch.tensor(
        [entries[int(value)][0] for value in entry_index.tolist()],
        device=device,
        dtype=torch.long,
    )
    stratum_index = torch.tensor(
        [entries[int(value)][1] for value in entry_index.tolist()],
        device=device,
        dtype=torch.long,
    )
    (
        phase_elapsed,
        phase_start_body_pose,
        phase_start_object_pose,
    ) = bank.restore(
        scene,
        env_ids=env_ids,
        phase_indices=phase_index,
        stratum_indices=stratum_index,
    )
    # The bank stores the nominal trajectory's instantaneous velocity because
    # it is also the source of rollout reset semantics.  Contact-state
    # construction, however, is a frozen QPID hold: restoring a moving
    # body/object snapshot and then commanding zero velocity creates an
    # artificial impact that can drag the object several centimetres before
    # the controller settles.  Start the reset-only settling window at the
    # exact nominal pose/joint position with every physical velocity zero.
    robot = scene["robot"]
    obj = scene["object"]
    robot.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros(
            (scene.num_envs, 6),
            device=device,
            dtype=phase_elapsed.dtype,
        ),
        env_ids=env_ids,
    )
    zero_joint_velocity = torch.zeros_like(_torch(robot.data.joint_vel))
    robot.write_joint_velocity_to_sim_index(
        velocity=zero_joint_velocity,
        env_ids=env_ids,
    )
    robot.set_joint_velocity_target_index(
        target=zero_joint_velocity,
        env_ids=env_ids,
    )
    obj.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros(
            (scene.num_envs, 6),
            device=device,
            dtype=phase_elapsed.dtype,
        ),
        env_ids=env_ids,
    )
    sim.forward()
    scene.update(0.0)
    state = io.gather_state(robot=scene["robot"], object_asset=scene["object"])
    initial_object_position = state.object_pose_world[:, :3].clone()
    maximum_displacement = torch.zeros(
        scene.num_envs, device=device, dtype=phase_elapsed.dtype
    )
    maximum_downward_speed = torch.zeros_like(maximum_displacement)
    collision_observed = torch.zeros(scene.num_envs, device=device, dtype=torch.bool)
    held_target = task_runtime.target(
        phase_index=phase_index,
        phase_elapsed_s=phase_elapsed,
        reset_robot_root_pose_world=phase_start_body_pose,
        reset_object_pose_world=phase_start_object_pose,
        reset_joint_positions_rad=joint_reference_start.index_select(0, phase_index),
        phase_end_joint_positions_rad=joint_reference_end.index_select(0, phase_index),
        lift_clearance_m=lift_clearance,
        transport_distance_m=transport_distance,
    )
    held_target = nominal_reference.condition(
        held_target,
        phase_index=phase_index,
        phase_elapsed_s=phase_elapsed,
        scene_origins=scene.env_origins,
    )
    contact_phase_index = torch.full_like(
        phase_index,
        ORDER9_OBJECT_TASK_PHASES.index(Order9ObjectTaskPhase.CONTACT_ACQUISITION),
    )
    contact_duration_s = float(
        nominal_reference.phase_durations_s[
            Order9ObjectTaskPhase.CONTACT_ACQUISITION.value
        ]
    )
    contact_tail_duration_s = min(0.1, contact_duration_s)
    contact_replay_start_s = contact_duration_s - contact_tail_duration_s
    contact_replay_duration_s = max(0.5, contact_tail_duration_s)
    payload_handoff_duration_s = 0.5
    # The ten-tail-delta preload was required by the actor-free experiment but
    # is not part of the accepted nominal or frozen-initializer contract.  It
    # creates an approximately one-radian Dock error on larger morphologies and
    # must not be stacked on top of the initializer's learned joint correction.
    contact_preload_tail_fraction = 0.0
    contact_start_target = nominal_reference.condition(
        held_target,
        phase_index=contact_phase_index,
        phase_elapsed_s=torch.full_like(phase_elapsed, contact_replay_start_s),
        scene_origins=scene.env_origins,
    )
    contact_end_target = nominal_reference.condition(
        held_target,
        phase_index=contact_phase_index,
        phase_elapsed_s=torch.full_like(phase_elapsed, contact_duration_s),
        scene_origins=scene.env_origins,
    )
    contact_preload_joint_delta = contact_preload_tail_fraction * (
        contact_end_target.nominal_joint_positions_rad
        - contact_start_target.nominal_joint_positions_rad
    )
    held_target = replace(
        held_target,
        nominal_joint_positions_rad=(
            held_target.nominal_joint_positions_rad + contact_preload_joint_delta
        ),
    )
    contact_translation = (
        held_target.desired_robot_root_pose_world[:, :3]
        - contact_end_target.desired_robot_root_pose_world[:, :3]
    )
    initial_body_pose = contact_start_target.desired_robot_root_pose_world.clone()
    initial_body_pose[:, :3] += contact_translation

    # Begin outside contact and replay the accepted contact-acquisition path
    # through QPID.  Teleporting directly into the nominal final mesh contact
    # can create a one-step PhysX depenetration impulse even when the authored
    # grasp frame is correct.
    robot = scene["robot"]
    initial_q = _torch(robot.data.joint_pos).clone()
    local_index = {joint_id: index for index, joint_id in enumerate(io.local_joint_ids)}
    for module_row, _module_id in enumerate(io.module_ids):
        for reference_joint, joint_id in enumerate(nominal_reference.joint_ids):
            robot_index = io.local_joint_indices[module_row][local_index[joint_id]]
            if robot_index < 0:
                raise RuntimeError(
                    f"C3 QPID reset contact joint was merged: {joint_id}"
                )
            initial_q[:, robot_index] = (
                contact_start_target.nominal_joint_positions_rad[
                    :, module_row, reference_joint
                ]
            )
    robot.write_joint_position_to_sim_index(position=initial_q, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(
        velocity=torch.zeros_like(initial_q), env_ids=env_ids
    )
    robot.set_joint_position_target_index(target=initial_q, env_ids=env_ids)
    robot.set_joint_velocity_target_index(
        target=torch.zeros_like(initial_q), env_ids=env_ids
    )
    sim.forward()
    scene.update(0.0)
    initial_state = io.gather_state(robot=robot, object_asset=scene["object"])
    initial_control = reset_runtime.builder.build(
        module_pose_world=initial_state.module_pose_world,
        module_twist_world=initial_state.module_twist_world,
        local_joint_positions_rad=initial_state.local_joint_positions_rad,
    )
    current_root = _torch(robot.data.root_pose_w)
    desired_root_world_tensor = torch.zeros_like(current_root)
    for env_id in range(scene.num_envs):
        root_to_body = compose_pose(
            inverse_pose(
                tuple(float(value) for value in current_root[env_id].tolist())
            ),
            tuple(
                float(value)
                for value in initial_control.body_pose_world[env_id].tolist()
            ),
        )
        desired_root_world = compose_pose(
            tuple(float(value) for value in initial_body_pose[env_id].tolist()),
            inverse_pose(root_to_body),
        )
        desired_root_world_tensor[env_id] = torch.tensor(
            desired_root_world, device=device, dtype=phase_elapsed.dtype
        )
    robot.write_root_pose_to_sim_index(
        root_pose=desired_root_world_tensor, env_ids=env_ids
    )
    sim.forward()
    scene.update(0.0)
    state = io.gather_state(robot=robot, object_asset=scene["object"])
    last_nominal_step = None
    last_contact = None
    selected_contact_count = torch.zeros(
        scene.num_envs, device=device, dtype=torch.long
    )
    maximum_selected_contact_count = torch.zeros_like(selected_contact_count)
    maximum_selected_contact_force_n = torch.zeros(
        scene.num_envs, device=device, dtype=phase_elapsed.dtype
    )
    qp_feasible_observed = torch.zeros(scene.num_envs, device=device, dtype=torch.bool)
    contact_qp_observed = torch.zeros_like(qp_feasible_observed)
    minimum_contact_qp_displacement = torch.full(
        (scene.num_envs,),
        torch.inf,
        device=device,
        dtype=phase_elapsed.dtype,
    )
    reference_joint_indices = torch.tensor(
        [io.local_joint_ids.index(value) for value in nominal_reference.joint_ids],
        device=device,
        dtype=torch.long,
    )
    contact_trajectory_steps = max(
        1, int(math.ceil(contact_replay_duration_s / float(dt_s)))
    )
    payload_handoff_steps = max(
        1, int(math.ceil(payload_handoff_duration_s / float(dt_s)))
    )
    fixture_free_hold_steps = max(
        1, int(math.ceil(float(fixture_free_hold_s) / float(dt_s)))
    )
    maximum_preload_steps = (
        contact_trajectory_steps
        + payload_handoff_steps
        + fixture_free_hold_steps
        + max(int(preload_steps), 8 * int(preload_steps))
    )
    admitted_snapshots: dict[int, dict[str, object]] = {}
    control = None
    temporary_support_activated = False
    initial_selected_grasp_frames_local: list[list[list[float]]] = []
    selected_anchor_by_id = {
        anchor.anchor_id: anchor for anchor in io.morphology_graph.robot_anchors
    }
    selected_body_pose = _torch(scene["robot"].data.body_pose_w).index_select(
        1,
        torch.tensor(
            io.selected_anchor_body_indices,
            device=device,
            dtype=torch.long,
        ),
    )
    for env_id in range(scene.num_envs):
        env_frames = []
        for anchor_index, anchor_id in enumerate(io.selected_anchor_ids):
            frame_world = compose_pose(
                tuple(
                    float(value)
                    for value in selected_body_pose[env_id, anchor_index].tolist()
                ),
                tuple(
                    float(value)
                    for value in selected_anchor_by_id[anchor_id].local_pose
                ),
            )
            frame_local = list(frame_world)
            for axis in range(3):
                frame_local[axis] -= float(scene.env_origins[env_id, axis].item())
            env_frames.append(frame_local)
        initial_selected_grasp_frames_local.append(env_frames)
    post_fixture_object_delta_xyz = torch.zeros(
        (scene.num_envs, 3), device=device, dtype=phase_elapsed.dtype
    )
    post_fixture_selected_contact_count = torch.zeros_like(selected_contact_count)
    handoff_body_position_error_m = torch.full_like(maximum_displacement, torch.nan)
    handoff_dock_tracking_error_rad = torch.full_like(maximum_displacement, torch.nan)
    handoff_vectoring_tracking_error_rad = torch.full_like(
        maximum_displacement, torch.nan
    )
    handoff_maximum_rotor_thrust_n = torch.full_like(maximum_displacement, torch.nan)
    handoff_allocation_residual_norm = torch.full_like(maximum_displacement, torch.nan)
    handoff_desired_wrench_norm = torch.full_like(maximum_displacement, torch.nan)

    temporary_support_ever_used = False
    consecutive_fixture_free_valid_steps = torch.zeros(
        scene.num_envs, device=device, dtype=torch.long
    )
    maximum_consecutive_fixture_free_valid_steps = torch.zeros_like(
        consecutive_fixture_free_valid_steps
    )
    reset_runtime.finish_transition(
        phase_success=torch.zeros(scene.num_envs, device=device, dtype=torch.bool),
        terminal_or_reset=torch.ones(scene.num_envs, device=device, dtype=torch.bool),
        current_vectoring_angles_rad=initial_control.current_vectoring_angles_rad,
    )

    for preload_index in range(maximum_preload_steps):
        contact_replay_active = preload_index < contact_trajectory_steps
        payload_handoff_index = preload_index - contact_trajectory_steps
        payload_handoff_active = 0 <= payload_handoff_index < payload_handoff_steps
        if contact_replay_active:
            payload_handoff_progress = 0.0
        elif payload_handoff_active:
            payload_handoff_progress = min(
                float(payload_handoff_index + 1) / float(payload_handoff_steps),
                1.0,
            )
        else:
            payload_handoff_progress = 1.0
        fixture_support_fraction = 1.0 - payload_handoff_progress
        if contact_replay_active:
            contact_replay_progress = min(
                float(preload_index + 1) / float(contact_trajectory_steps),
                1.0,
            )
            replay_elapsed = torch.full_like(
                phase_elapsed,
                contact_replay_start_s
                + contact_tail_duration_s * contact_replay_progress,
            )
            nominal_target = nominal_reference.condition(
                held_target,
                phase_index=contact_phase_index,
                phase_elapsed_s=replay_elapsed,
                scene_origins=scene.env_origins,
            )
            replay_pose = nominal_target.desired_robot_root_pose_world.clone()
            replay_pose[:, :3] += contact_translation
            nominal_target = replace(
                nominal_target,
                desired_robot_root_pose_world=replay_pose,
                nominal_joint_positions_rad=(
                    nominal_target.nominal_joint_positions_rad
                    + contact_replay_progress * contact_preload_joint_delta
                ),
            )
        else:
            nominal_target = held_target
        if fixture_support_fraction > 0.0:
            obj = scene["object"]
            gravity_magnitude = max(0.0, -float(sim.cfg.gravity[2]))
            body_mass = _torch(obj.data.body_mass).to(
                device=device, dtype=phase_elapsed.dtype
            )
            position_error = initial_object_position - state.object_pose_world[:, :3]
            desired_acceleration = (
                400.0 * position_error - 40.0 * state.object_twist_world[:, :3]
            )
            gravity_compensation = torch.zeros_like(desired_acceleration)
            gravity_compensation[:, 2] = gravity_magnitude
            support_force = torch.zeros(
                (scene.num_envs, 1, 3),
                device=device,
                dtype=phase_elapsed.dtype,
            )
            support_force[:, 0] = (
                fixture_support_fraction
                * body_mass[:, :1]
                * (desired_acceleration + gravity_compensation).clamp(
                    min=-10.0 * gravity_magnitude,
                    max=10.0 * gravity_magnitude,
                )
            )
            obj.permanent_wrench_composer.set_forces_and_torques_index(
                forces=support_force,
                torques=torch.zeros_like(support_force),
                is_global=True,
            )
            temporary_support_activated = True
            temporary_support_ever_used = True
        else:
            if temporary_support_activated:
                scene["object"].permanent_wrench_composer.reset()
                temporary_support_activated = False
        payload_scale = torch.full_like(estimated_mass, payload_handoff_progress)
        last_nominal_step = reset_runtime.compute(
            time_s=torch.full_like(phase_elapsed, float(preload_index) * float(dt_s)),
            phase_index=_actor_phase_indices(phase_index),
            task_target=nominal_target,
            state=state,
            estimated_payload_mass_kg=estimated_mass * payload_scale,
            estimated_payload_inertia_body=(
                estimated_inertia * payload_scale.unsqueeze(-1)
            ),
            payload_active=payload_scale > 0.0,
            estimated_payload_com_object=estimated_com,
            deterministic=True,
        )
        if payload_handoff_index == 0:
            handoff_body_position_error_m.copy_(
                torch.linalg.vector_norm(
                    last_nominal_step.control_model.body_pose_world[:, :3]
                    - nominal_target.desired_robot_root_pose_world[:, :3],
                    dim=-1,
                )
            )
            handoff_dock_tracking_error_rad.copy_(
                (
                    state.local_joint_positions_rad.index_select(
                        2, reference_joint_indices
                    )
                    - nominal_target.nominal_joint_positions_rad
                )
                .abs()
                .amax(dim=(1, 2))
            )
            handoff_vectoring_tracking_error_rad.copy_(
                (
                    last_nominal_step.control_model.current_vectoring_angles_rad
                    - last_nominal_step.controller_result.allocation.vectoring_joint_targets_rad
                )
                .abs()
                .amax(dim=-1)
            )
            handoff_maximum_rotor_thrust_n.copy_(
                last_nominal_step.controller_result.allocation.rotor_thrusts_n.amax(
                    dim=-1
                )
            )
            handoff_allocation_residual_norm.copy_(
                last_nominal_step.controller_result.allocation.residual_norm
            )
            handoff_desired_wrench_norm.copy_(
                torch.linalg.vector_norm(
                    last_nominal_step.controller_result.desired_wrench_body,
                    dim=-1,
                )
            )
        io.apply(
            robot=scene["robot"],
            policy_command=last_nominal_step.policy_command,
            controller_result=last_nominal_step.controller_result,
        )
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(float(dt_s))
        state = io.gather_state(robot=scene["robot"], object_asset=scene["object"])
        robot_net = _torch(robot_sensor.data.net_forces_w).index_select(
            1, robot_sensor_reorder
        )
        object_matrix = _torch(object_sensor.data.force_matrix_w)
        last_contact = io.reduce_contacts(
            robot_net_contact_forces_world=robot_net,
            object_force_matrix_world=object_matrix,
            robot_body_linear_velocity_world=_torch(scene["robot"].data.body_lin_vel_w),
            robot_body_angular_velocity_world=_torch(
                scene["robot"].data.body_ang_vel_w
            ),
            selected_assignment_mask=selected_mask,
            allow_selected_object_contact=torch.ones(
                scene.num_envs, device=device, dtype=torch.bool
            ),
        )
        selected_contact_count = last_contact.selected_contact_mask.sum(dim=-1)
        maximum_selected_contact_count = torch.maximum(
            maximum_selected_contact_count, selected_contact_count
        )
        maximum_selected_contact_force_n = torch.maximum(
            maximum_selected_contact_force_n,
            torch.linalg.vector_norm(
                last_contact.selected_contact_forces_world, dim=-1
            ).amax(dim=-1),
        )
        collision_observed |= last_contact.prohibited_collision
        qp_feasible = last_nominal_step.controller_result.allocation.feasible
        qp_feasible_observed |= qp_feasible
        current_displacement = torch.linalg.vector_norm(
            state.object_pose_world[:, :3] - initial_object_position,
            dim=-1,
        )
        contact_qp_now = (
            (selected_contact_count >= int(minimum_selected_contacts))
            & qp_feasible
            & ~collision_observed
        )
        contact_qp_observed |= contact_qp_now
        minimum_contact_qp_displacement = torch.minimum(
            minimum_contact_qp_displacement,
            torch.where(
                contact_qp_now,
                current_displacement,
                torch.full_like(current_displacement, torch.inf),
            ),
        )
        current_downward_speed = (-state.object_twist_world[:, 2]).clamp_min(0.0)
        maximum_displacement = torch.maximum(maximum_displacement, current_displacement)
        maximum_downward_speed = torch.maximum(
            maximum_downward_speed, current_downward_speed
        )
        fixture_free_valid_now = (
            (selected_contact_count >= int(minimum_selected_contacts))
            & ~last_contact.prohibited_collision
            & last_nominal_step.controller_result.allocation.feasible
            & (current_displacement <= float(maximum_object_displacement_m))
            & (current_downward_speed <= float(maximum_downward_speed_mps))
            & torch.full_like(
                collision_observed,
                payload_handoff_progress >= 1.0,
                dtype=torch.bool,
            )
        )
        consecutive_fixture_free_valid_steps = torch.where(
            fixture_free_valid_now,
            consecutive_fixture_free_valid_steps + 1,
            torch.zeros_like(consecutive_fixture_free_valid_steps),
        )
        maximum_consecutive_fixture_free_valid_steps = torch.maximum(
            maximum_consecutive_fixture_free_valid_steps,
            consecutive_fixture_free_valid_steps,
        )
        admitted_now = fixture_free_valid_now & (
            consecutive_fixture_free_valid_steps >= fixture_free_hold_steps
        )
        control = reset_runtime.builder.build(
            module_pose_world=state.module_pose_world,
            module_twist_world=state.module_twist_world,
            local_joint_positions_rad=state.local_joint_positions_rad,
        )
        robot = scene["robot"]
        obj = scene["object"]
        root_pose_local = _torch(robot.data.root_pose_w).clone()
        root_pose_local[:, :3] -= scene.env_origins
        object_pose_local = _object_pose(obj).clone()
        object_pose_local[:, :3] -= scene.env_origins
        root_twist = torch.cat(
            (
                _torch(robot.data.root_lin_vel_w),
                _torch(robot.data.root_ang_vel_w),
            ),
            dim=-1,
        )
        joint_position = _torch(robot.data.joint_pos)
        joint_velocity = _torch(robot.data.joint_vel)
        object_twist = _object_twist(obj)
        for entry_id, (reset_phase, reset_stratum) in enumerate(entries):
            if entry_id in admitted_snapshots:
                continue
            candidates = torch.nonzero(
                (entry_index == entry_id) & admitted_now, as_tuple=False
            ).flatten()
            if candidates.numel() == 0:
                continue
            score = current_displacement[candidates] + current_downward_speed[
                candidates
            ] * float(dt_s)
            representative = int(candidates[score.argmin()].item())
            velocity_projection = False
            body_local = control.body_pose_world[representative].clone()
            body_local[:3] -= scene.env_origins[representative]
            body_start_local = phase_start_body_pose[representative].clone()
            body_start_local[:3] -= scene.env_origins[representative]
            object_start_local = phase_start_object_pose[representative].clone()
            object_start_local[:3] -= scene.env_origins[representative]
            admitted_snapshots[entry_id] = {
                "phase": reset_phase,
                "stratum": reset_stratum,
                "representative": representative,
                "preload_steps_used": preload_index + 1,
                "temporary_object_settling_support_used": bool(
                    temporary_support_ever_used
                ),
                "root_pose_local": root_pose_local[representative].clone(),
                "root_twist": (
                    torch.zeros_like(root_twist[representative])
                    if velocity_projection
                    else root_twist[representative].clone()
                ),
                "joint_position": joint_position[representative].clone(),
                "joint_velocity": (
                    torch.zeros_like(joint_velocity[representative])
                    if velocity_projection
                    else joint_velocity[representative].clone()
                ),
                "object_pose_local": object_pose_local[representative].clone(),
                "object_twist": (
                    torch.zeros_like(object_twist[representative])
                    if velocity_projection
                    else object_twist[representative].clone()
                ),
                "body_pose_local": body_local,
                "body_twist": (
                    torch.zeros_like(control.body_twist_world[representative])
                    if velocity_projection
                    else control.body_twist_world[representative].clone()
                ),
                "reference_joint_position": (
                    state.local_joint_positions_rad[representative]
                    .index_select(1, reference_joint_indices)
                    .clone()
                ),
                "reference_joint_velocity": (
                    torch.zeros_like(
                        state.local_joint_velocities_radps[representative].index_select(
                            1, reference_joint_indices
                        )
                    )
                    if velocity_projection
                    else state.local_joint_velocities_radps[representative]
                    .index_select(1, reference_joint_indices)
                    .clone()
                ),
                "phase_elapsed_s": phase_elapsed[representative].clone(),
                "task_body_start_pose_local": body_start_local,
                "task_object_start_pose_local": object_start_local,
                "selected_contact_count": int(
                    selected_contact_count[representative].item()
                ),
                "object_displacement_m": float(
                    current_displacement[representative].item()
                ),
                "downward_speed_mps": float(
                    0.0
                    if velocity_projection
                    else current_downward_speed[representative].item()
                ),
                "pre_projection_downward_speed_mps": float(
                    current_downward_speed[representative].item()
                ),
                "reset_velocity_projection_applied": velocity_projection,
                "peak_settling_displacement_m": float(
                    maximum_displacement[representative].item()
                ),
                "peak_settling_downward_speed_mps": float(
                    maximum_downward_speed[representative].item()
                ),
            }
        if len(admitted_snapshots) == len(entries):
            break
        final_hold_index = (
            preload_index - contact_trajectory_steps - payload_handoff_steps
        )
        if final_hold_index + 1 < int(preload_steps):
            continue
        if payload_handoff_progress >= 1.0:
            post_fixture_object_delta_xyz.copy_(
                state.object_pose_world[:, :3] - initial_object_position
            )
            post_fixture_selected_contact_count.copy_(selected_contact_count)

    scene["object"].permanent_wrench_composer.reset()
    if last_nominal_step is None or last_contact is None:  # pragma: no cover
        raise RuntimeError("Order9 C3 reset stabilization produced no step")
    if len(admitted_snapshots) != len(entries):
        final_selected_body_pose = _torch(scene["robot"].data.body_pose_w).index_select(
            1,
            torch.tensor(
                io.selected_anchor_body_indices,
                device=device,
                dtype=torch.long,
            ),
        )
        final_selected_grasp_frames_local: list[list[list[float]]] = []
        for env_id in range(scene.num_envs):
            env_frames = []
            for anchor_index, anchor_id in enumerate(io.selected_anchor_ids):
                frame_world = compose_pose(
                    tuple(
                        float(value)
                        for value in final_selected_body_pose[
                            env_id, anchor_index
                        ].tolist()
                    ),
                    tuple(
                        float(value)
                        for value in selected_anchor_by_id[anchor_id].local_pose
                    ),
                )
                frame_local = list(frame_world)
                for axis in range(3):
                    frame_local[axis] -= float(scene.env_origins[env_id, axis].item())
                env_frames.append(frame_local)
            final_selected_grasp_frames_local.append(env_frames)
        final_joint_tracking_error = (
            (
                state.local_joint_positions_rad.index_select(2, reference_joint_indices)
                - last_nominal_step.policy_command.joint_position_targets_rad
            )
            .abs()
            .amax(dim=(1, 2))
        )
        missing = []
        for entry_id, (reset_phase, reset_stratum) in enumerate(entries):
            if entry_id in admitted_snapshots:
                continue
            members = torch.nonzero(entry_index == entry_id, as_tuple=False).flatten()
            final_member_displacement = current_displacement[members]
            minimum_final_member = members[final_member_displacement.argmin()]
            minimum_contact_qp = minimum_contact_qp_displacement[members].min()
            representative = int(members[0].item())
            missing.append(
                {
                    "phase": ORDER9_OBJECT_TASK_PHASES[reset_phase].value,
                    "stratum": int(reset_stratum),
                    "candidate_count": int(members.numel()),
                    "maximum_selected_contacts": int(
                        maximum_selected_contact_count[members].max().item()
                    ),
                    "maximum_selected_contact_force_n": float(
                        maximum_selected_contact_force_n[members].max().item()
                    ),
                    "collision_count": int(collision_observed[members].sum().item()),
                    "qp_feasible_count": int(
                        qp_feasible_observed[members].sum().item()
                    ),
                    "contact_qp_count": int(contact_qp_observed[members].sum().item()),
                    "minimum_contact_qp_displacement_m": (
                        None
                        if not bool(torch.isfinite(minimum_contact_qp).item())
                        else float(minimum_contact_qp.item())
                    ),
                    "minimum_final_object_displacement_m": float(
                        final_member_displacement.min().item()
                    ),
                    "minimum_final_object_delta_xyz_m": [
                        float(value)
                        for value in (
                            state.object_pose_world[minimum_final_member, :3]
                            - initial_object_position[minimum_final_member]
                        ).tolist()
                    ],
                    "minimum_final_object_height_m": float(
                        state.object_pose_world[minimum_final_member, 2].item()
                    ),
                    "minimum_final_downward_speed_mps": float(
                        current_downward_speed[members].min().item()
                    ),
                    "post_fixture_object_delta_xyz_m": [
                        float(value)
                        for value in post_fixture_object_delta_xyz[
                            representative
                        ].tolist()
                    ],
                    "post_fixture_selected_contact_count": int(
                        post_fixture_selected_contact_count[representative].item()
                    ),
                    "maximum_final_joint_tracking_error_rad": float(
                        final_joint_tracking_error[representative].item()
                    ),
                    "minimum_handoff_body_position_error_m": float(
                        handoff_body_position_error_m[members]
                        .nan_to_num(nan=torch.inf)
                        .min()
                        .item()
                    ),
                    "minimum_handoff_dock_tracking_error_rad": float(
                        handoff_dock_tracking_error_rad[members]
                        .nan_to_num(nan=torch.inf)
                        .min()
                        .item()
                    ),
                    "minimum_handoff_vectoring_tracking_error_rad": float(
                        handoff_vectoring_tracking_error_rad[members]
                        .nan_to_num(nan=torch.inf)
                        .min()
                        .item()
                    ),
                    "maximum_handoff_rotor_thrust_n": float(
                        handoff_maximum_rotor_thrust_n[members]
                        .nan_to_num()
                        .max()
                        .item()
                    ),
                    "maximum_handoff_allocation_residual_norm": float(
                        handoff_allocation_residual_norm[members]
                        .nan_to_num()
                        .max()
                        .item()
                    ),
                    "maximum_handoff_desired_wrench_norm": float(
                        handoff_desired_wrench_norm[members].nan_to_num().max().item()
                    ),
                    "maximum_consecutive_fixture_free_valid_steps": int(
                        maximum_consecutive_fixture_free_valid_steps[members]
                        .max()
                        .item()
                    ),
                    "initial_selected_grasp_frames_local": (
                        initial_selected_grasp_frames_local[representative]
                    ),
                    "final_selected_grasp_frames_local": (
                        final_selected_grasp_frames_local[representative]
                    ),
                    "initial_object_pose_local": [
                        float(value)
                        for value in bank.object_pose_local[
                            representative, reset_phase, reset_stratum
                        ].tolist()
                    ],
                }
            )
        raise RuntimeError(
            "Order9 C3 contact reset stabilization exhausted its settling "
            f"horizon: {missing}"
        )
    assert control is not None
    admitted_rows: list[dict[str, object]] = []
    for entry_id, (reset_phase, reset_stratum) in enumerate(entries):
        snapshot = admitted_snapshots[entry_id]
        representative = int(snapshot["representative"])
        for env_id in range(scene.num_envs):
            bank.install(
                env_id=env_id,
                phase_index=reset_phase,
                stratum_index=reset_stratum,
                robot_root_pose_local=snapshot["root_pose_local"],
                robot_root_twist=snapshot["root_twist"],
                joint_position=snapshot["joint_position"],
                joint_velocity=snapshot["joint_velocity"],
                object_pose_local=snapshot["object_pose_local"],
                object_twist=snapshot["object_twist"],
                phase_elapsed_s=float(snapshot["phase_elapsed_s"].item()),
                task_body_start_pose_local=snapshot["task_body_start_pose_local"],
                task_object_start_pose_local=snapshot["task_object_start_pose_local"],
            )
        reset_references.body_pose_local[reset_phase, reset_stratum] = snapshot[
            "body_pose_local"
        ]
        reset_references.body_twist[reset_phase, reset_stratum] = snapshot["body_twist"]
        reset_references.joint_positions_rad[reset_phase, reset_stratum] = snapshot[
            "reference_joint_position"
        ]
        reset_references.joint_velocities_radps[reset_phase, reset_stratum] = snapshot[
            "reference_joint_velocity"
        ]
        reset_references.object_pose_local[reset_phase, reset_stratum] = snapshot[
            "object_pose_local"
        ]
        reset_references.object_twist[reset_phase, reset_stratum] = snapshot[
            "object_twist"
        ]
        reset_references.phase_elapsed_s[reset_phase, reset_stratum] = snapshot[
            "phase_elapsed_s"
        ]
        admitted_rows.append(
            {
                "phase": ORDER9_OBJECT_TASK_PHASES[reset_phase].value,
                "stratum": int(reset_stratum),
                "representative_environment": representative,
                "preload_steps_used": int(snapshot["preload_steps_used"]),
                "temporary_object_settling_support_used": snapshot[
                    "temporary_object_settling_support_used"
                ],
                "selected_contact_count": snapshot["selected_contact_count"],
                "object_displacement_m": snapshot["object_displacement_m"],
                "downward_speed_mps": snapshot["downward_speed_mps"],
                "pre_projection_downward_speed_mps": snapshot[
                    "pre_projection_downward_speed_mps"
                ],
                "reset_velocity_projection_applied": snapshot[
                    "reset_velocity_projection_applied"
                ],
                "peak_settling_displacement_m": snapshot[
                    "peak_settling_displacement_m"
                ],
                "peak_settling_downward_speed_mps": snapshot[
                    "peak_settling_downward_speed_mps"
                ],
                "phase_elapsed_s": float(snapshot["phase_elapsed_s"].item()),
            }
        )

    reset_runtime.finish_transition(
        phase_success=torch.zeros(scene.num_envs, device=device, dtype=torch.bool),
        terminal_or_reset=torch.ones(scene.num_envs, device=device, dtype=torch.bool),
        current_vectoring_angles_rad=control.current_vectoring_angles_rad,
    )
    return {
        "version": ("order9_c3_contact_reset_stabilization_v8_frozen_initializer_bank"),
        "passed": True,
        "pi_l_actor_used": True,
        "pi_l_actor_role": "frozen_initializer_reset_generation_only",
        "contact_tail_duration_s": contact_tail_duration_s,
        "contact_tail_replay_duration_s": contact_replay_duration_s,
        "contact_preload_tail_fraction": contact_preload_tail_fraction,
        "payload_handoff_duration_s": payload_handoff_duration_s,
        "payload_handoff_steps": payload_handoff_steps,
        "maximum_contact_preload_joint_delta_rad": float(
            contact_preload_joint_delta.abs().max().item()
        ),
        "final_phase_target_frozen": True,
        "object_fixture_removed_before_admission": True,
        "controller_path": "frozen_initializer_pi_l_qpid_qp",
        "preload_steps": int(preload_steps),
        "maximum_preload_steps": maximum_preload_steps,
        "maximum_preload_duration_s": maximum_preload_steps * float(dt_s),
        "actual_maximum_preload_steps": max(
            int(row["preload_steps_used"]) for row in admitted_rows
        ),
        "minimum_selected_contacts": int(minimum_selected_contacts),
        "maximum_object_displacement_m": float(maximum_object_displacement_m),
        "maximum_downward_speed_mps": float(maximum_downward_speed_mps),
        "fixture_free_hold_s": float(fixture_free_hold_s),
        "fixture_free_hold_steps": int(fixture_free_hold_steps),
        "admitted_state_count": len(admitted_rows),
        "admitted_states": admitted_rows,
    }


def _transport_distance(task: TaskSpec, object_id: str) -> float:
    obj = next(value for value in task.scene.objects if value.object_id == object_id)
    goal = next(
        value
        for value in task.goals
        if value.goal_type == "object_pose" and value.target_entity_id == object_id
    )
    if goal.target_pose_world is None:
        raise ValueError("Order9 object goal has no target pose")
    return float(goal.target_pose_world[0]) - float(obj.pose_world[0])


def _translated_task(task: TaskSpec, origin: torch.Tensor, index: int) -> TaskSpec:
    value = TaskSpec.from_dict(task.to_dict())
    translation = tuple(float(item) for item in origin.tolist())
    value.task_id = f"{task.task_id}:env:{index:04d}"
    for obj in value.scene.objects:
        obj.pose_world = _translate_pose(obj.pose_world, translation)
    for surface in value.scene.environment.support_surfaces:
        surface.pose_world = _translate_pose(surface.pose_world, translation)
    for obstacle in value.scene.environment.obstacles:
        obstacle.pose_world = _translate_pose(obstacle.pose_world, translation)
    for goal in value.goals:
        if goal.target_pose_world is not None:
            goal.target_pose_world = _translate_pose(
                goal.target_pose_world, translation
            )
    value.metadata = {
        **value.metadata,
        "isaac_environment_origin_world": list(translation),
        "isaac_environment_index": index,
    }
    value.validate()
    return value


def _rollout_metadata(
    *,
    config,
    stage,
    morphology,
    physical,
    checkpoint_sha256,
    policy_version,
    tasks,
    split,
    assignments,
    teacher_trajectory,
    io,
    reward_names,
    selected_friction,
    canonical_resets,
    phase_specific_resets_available,
    robot_usd,
    robot_asset_manifest,
    estimated_mass_kg,
    estimated_inertia_body,
    estimated_com_object,
    object_mass_properties_readback,
    actuator_readback,
    teacher_reference,
    c3_nominal_reference,
    contact_space_action_basis,
    c3_reset_stabilization,
    reset_strata_by_phase,
    phase_duration_s,
    deterministic_policy,
    initial_phase_zero,
    diagnostic_initial_phase_index,
    diagnostic_phase_strata,
):
    contact_compression_action_span_m = (
        float(args_cli.contact_compression_action_span_mm) * 1.0e-3
        if args_cli.contact_compression_action_span_mm is not None
        else float(
            config.production_runtime.c3_contact_compression_action_span_m
        )
    )
    thrust_model_hash = str(physical.metadata.get("thrust_model_hash", ""))
    if not thrust_model_hash:
        raise RuntimeError("Order9 PhysicalModel lacks thrust-model provenance")
    return {
        "generation_id": args_cli.generation_id,
        "c3_execution_bundle": (
            None
            if c3_execution_bundle_runtime is None
            else c3_execution_bundle_runtime.provenance()
        ),
        "pi_l_checkpoint_sha256": checkpoint_sha256,
        "c3_reset_stabilization_checkpoint_sha256": None,
        "c3_reset_bank_sha256": (
            c3_reset_stabilization.get("reset_bank_sha256")
            if c3_reset_stabilization is not None
            else None
        ),
        "pi_l_policy_version": policy_version,
        "stage_id": stage.stage_id,
        "stage_config_hash": stable_hash(stage.to_dict()),
        "curriculum_schedule_hash": order9_schedule_hash(config),
        "config_hash": stable_hash(config.to_dict()),
        "morphology_graph": morphology.to_dict(),
        "physical_model_hash": physical.stable_hash(),
        "urdf_hash": hash_file(physical.urdf_path),
        "thrust_model_hash": thrust_model_hash,
        "robot_usd_sha256": hash_file(robot_usd),
        "robot_asset_manifest": (
            None if robot_asset_manifest is None else robot_asset_manifest.to_dict()
        ),
        "simulator_version": _SIMULATOR_VERSION,
        "device": str(args_cli.device),
        "simulator_hash": stable_hash(
            {
                "simulator_version": _SIMULATOR_VERSION,
                "collector_version": _COLLECTOR_VERSION,
                "use_fabric": True,
                "quaternion_layout": "xyzw",
                "control_dt_s": float(args_cli.dt),
                "phase_successor_reference_semantics": (
                    ORDER9_PHASE_SUCCESSOR_REFERENCE_SEMANTICS
                ),
                "payload_feedforward_phase_contract": (
                    ORDER9_PAYLOAD_FEEDFORWARD_PHASE_CONTRACT
                ),
                "contact_acquisition_success_contract": (
                    ORDER9_CONTACT_ACQUISITION_SUCCESS_CONTRACT
                ),
                "wrench_range_gate_scale": float(args_cli.c3_wrench_gate_range_scale),
                "wrench_range_hard_gate_enabled": False,
                "exact_wrench_reward_bounds_preserved": True,
                "release_success_contract": ORDER9_RELEASE_SUCCESS_CONTRACT,
                "c3_boundary_tail_sampling_contract": (
                    ORDER9_C3_BOUNDARY_TAIL_SAMPLING_CONTRACT
                    if c3_nominal_reference is not None
                    else None
                ),
                "controller_side_contact_preload_version": (None),
                "contact_compression_action_adapter_version": (
                    _contact_compression_action_adapter_version(
                        policy_version,
                        c3_action_contract=args_cli.c3_action_contract,
                    )
                    if (
                        c3_nominal_reference is not None
                        and config.production_runtime.c3_contact_compression_action_adapter_enabled
                    )
                    else None
                ),
                "contact_compression_action_span_m": (
                    float(
                        contact_compression_action_span_m
                    )
                    if (
                        c3_nominal_reference is not None
                        and config.production_runtime.c3_contact_compression_action_adapter_enabled
                    )
                    else None
                ),
                "contact_compression_action_raw_contact_input": False,
                "contact_normal_action_quantization_version": (
                    ORDER9_CONTACT_NORMAL_ACTION_QUANTIZATION_VERSION
                    if contact_space_action_basis is not None
                    and contact_space_action_basis.normal_translation_quantization_step_m
                    is not None
                    else None
                ),
                "contact_normal_action_quantization_step_m": (
                    None
                    if contact_space_action_basis is None
                    else contact_space_action_basis.normal_translation_quantization_step_m
                ),
            }
        ),
        "random_seed": int(args_cli.seed),
        "estimated_payload_mass_kg": float(estimated_mass_kg),
        "estimated_payload_inertia_body": list(estimated_inertia_body),
        "estimated_payload_com_object": list(estimated_com_object),
        "object_mass_properties_readback": dict(object_mass_properties_readback),
        "actuator_readback": dict(actuator_readback),
        "teacher_reference": (
            None if teacher_reference is None else dict(teacher_reference.provenance)
        ),
        "c3_nominal_reference": (
            None
            if c3_nominal_reference is None
            else dict(c3_nominal_reference.provenance)
        ),
        "task_specs": [task.to_dict() for task in tasks],
        "environment_splits": [split.value for _ in tasks],
        "assignment_templates_by_environment": [
            [assignment.to_dict() for assignment in assignments] for _ in tasks
        ],
        "active_knot_trajectory_template": teacher_trajectory.to_dict(),
        "active_knot_feature_contract_version": (
            ORDER9_ACTIVE_KNOT_FEATURE_CONTRACT_VERSION
        ),
        "active_knot_global_feature_names": list(
            ORDER9_ACTIVE_KNOT_GLOBAL_FEATURE_NAMES
        ),
        "active_assignment_feature_names": list(ORDER9_ACTIVE_ASSIGNMENT_FEATURE_NAMES),
        "contact_space_feature_names": list(
            ORDER9_CONTACT_FEEDBACK_SPACE_FEATURE_NAMES
            if policy_version
            in {
                ORDER9_CONTACT_FEEDBACK_PI_L_POLICY_VERSION,
                ORDER9_CATEGORICAL_CONTACT_NORMAL_PI_L_POLICY_VERSION,
            }
            else ORDER9_CONTACT_SPACE_FEATURE_NAMES
        ),
        "contact_space_action_names": list(ORDER9_CONTACT_SPACE_ACTION_NAMES),
        "contact_normal_action_distribution": (
            "categorical_81_bins_-20mm_to_20mm"
            if policy_version
            == ORDER9_CATEGORICAL_CONTACT_NORMAL_PI_L_POLICY_VERSION
            else "squashed_normal_continuous"
        ),
        "contact_normal_action_category_count": (
            81
            if policy_version
            == ORDER9_CATEGORICAL_CONTACT_NORMAL_PI_L_POLICY_VERSION
            else None
        ),
        "contact_normal_action_category_step_m": (
            0.0005
            if policy_version
            == ORDER9_CATEGORICAL_CONTACT_NORMAL_PI_L_POLICY_VERSION
            else None
        ),
        "contact_normal_action_quantization_version": (
            ORDER9_CONTACT_NORMAL_ACTION_QUANTIZATION_VERSION
            if contact_space_action_basis is not None
            and contact_space_action_basis.normal_translation_quantization_step_m
            is not None
            else None
        ),
        "contact_normal_action_quantization_step_m": (
            None
            if contact_space_action_basis is None
            else contact_space_action_basis.normal_translation_quantization_step_m
        ),
        "max_contact_slots": ORDER9_MAX_CONTACT_SLOTS,
        "object_id": _target_object_and_geometry(tasks[0])[0].object_id,
        "module_ids": list(io.module_ids),
        "local_joint_ids": list(io.local_joint_ids),
        "command_local_joint_ids": list(policy_command_joint_ids(physical)),
        "rotor_global_ids": [
            f"module_{module_id}:{rotor.rotor_id}"
            for module_id in io.module_ids
            for rotor in sorted(physical.rotors, key=lambda value: value.rotor_id)
        ],
        "vectoring_global_joint_ids": [
            f"module_{module_id}:{rotor.vectoring_joint_ids[0]}"
            for module_id in io.module_ids
            for rotor in sorted(physical.rotors, key=lambda value: value.rotor_id)
        ],
        "reward_term_names": list(reward_names),
        "deployable_normal_contact_quality_reward_contract": (
            ORDER9_DEPLOYABLE_NORMAL_CONTACT_QUALITY_REWARD_CONTRACT
        ),
        "control_dt_s": float(args_cli.dt),
        "raw_contact_actor_input": False,
        "topology_randomized": bool(stage.topology_randomized),
        "collector_version": _COLLECTOR_VERSION,
        "controller_side_contact_preload_version": (None),
        "deployable_phase_gate_version": ORDER9_DEPLOYABLE_PHASE_GATE_VERSION,
        "phase_supervision_source": (
            "privileged_physx_contact_outcome"
            if stage.stage_id == ORDER9_C3_STAGE_ID
            else "deployable_phase_gate"
        ),
        "c3_privileged_phase_supervision_contract": (
            ORDER9_C3_PRIVILEGED_PHASE_SUPERVISION_CONTRACT
            if stage.stage_id == ORDER9_C3_STAGE_ID
            else None
        ),
        "wrench_range_gate_scale": float(args_cli.c3_wrench_gate_range_scale),
        "wrench_range_hard_gate_enabled": False,
        "exact_wrench_reward_bounds_preserved": True,
        "virtual_contact_compression_version": (
            ORDER9_VIRTUAL_CONTACT_COMPRESSION_VERSION
            if c3_nominal_reference is not None
            else None
        ),
        "contact_compression_action_adapter_version": (
            _contact_compression_action_adapter_version(
                policy_version,
                c3_action_contract=args_cli.c3_action_contract,
            )
            if (
                c3_nominal_reference is not None
                and config.production_runtime.c3_contact_compression_action_adapter_enabled
            )
            else None
        ),
        "contact_compression_action_span_m": (
            contact_compression_action_span_m
            if (
                c3_nominal_reference is not None
                and config.production_runtime.c3_contact_compression_action_adapter_enabled
            )
            else None
        ),
        "contact_compression_action_raw_contact_input": False,
        "selected_anchor_ids": list(io.selected_anchor_ids),
        "selected_gripper_friction": float(selected_friction),
        "contact_stiffness_n_per_m": float(args_cli.contact_stiffness),
        "contact_damping_n_s_per_m": float(args_cli.contact_damping),
        "privileged_contact_wrench_reward_contract_version": (
            ORDER9_CONTACT_WRENCH_REWARD_CONTRACT_VERSION
        ),
        "canonical_phase_resets": bool(canonical_resets),
        "phase_specific_resets_available": bool(phase_specific_resets_available),
        "phase_reset_reference_version": (
            ORDER9_C3_PHASE_RESET_REFERENCE_VERSION
            if c3_nominal_reference is not None
            else None
        ),
        "phase_reset_progress_fractions": (
            list(config.production_runtime.c3_phase_reset_progress_fractions)
            if c3_nominal_reference is not None
            else None
        ),
        "phase_reset_stratum_count": (
            len(config.production_runtime.c3_phase_reset_progress_fractions)
            if c3_nominal_reference is not None
            else 1
        ),
        "c3_boundary_tail_sampling_contract": (
            ORDER9_C3_BOUNDARY_TAIL_SAMPLING_CONTRACT
            if c3_nominal_reference is not None
            else None
        ),
        "c3_boundary_tail_phase_labels": (
            list(config.production_runtime.c3_boundary_tail_phase_labels)
            if c3_nominal_reference is not None
            else None
        ),
        "c3_boundary_tail_progress_fraction": (
            float(config.production_runtime.c3_boundary_tail_progress_fraction)
            if c3_nominal_reference is not None
            else None
        ),
        "phase_reset_selectable_strata_by_phase": (
            [list(indices) for indices in reset_strata_by_phase]
            if c3_nominal_reference is not None
            else None
        ),
        "c3_contact_reset_stabilization": c3_reset_stabilization,
        "phase_reset_state_labels_reused": False,
        "phase_successor_reference_semantics": (
            ORDER9_PHASE_SUCCESSOR_REFERENCE_SEMANTICS
        ),
        "payload_feedforward_phase_contract": (
            ORDER9_PAYLOAD_FEEDFORWARD_PHASE_CONTRACT
        ),
        "contact_acquisition_success_contract": (
            ORDER9_CONTACT_ACQUISITION_SUCCESS_CONTRACT
        ),
        "release_success_contract": ORDER9_RELEASE_SUCCESS_CONTRACT,
        "release_joint_position_tolerance_rad": float(
            ORDER9_RELEASE_JOINT_POSITION_TOLERANCE_RAD
        ),
        "object_pose_success_contract": ORDER9_OBJECT_POSE_SUCCESS_CONTRACT,
        "object_position_tolerance_m": float(ORDER9_OBJECT_POSITION_TOLERANCE_M),
        "runtime_phase_labels": [phase.value for phase in ORDER9_OBJECT_TASK_PHASES],
        "actor_phase_labels": list(ORDER9_OBJECT_TASK_ACTOR_PHASE_LABELS),
        "actor_phase_index_by_runtime": list(
            ORDER9_OBJECT_TASK_ACTOR_PHASE_INDEX_BY_RUNTIME
        ),
        "phase_duration_s": dict(phase_duration_s),
        "phase_progress_semantics": "exact_policy_actor_input",
        "c3_nominal_boundary_completion_contract": (
            ORDER9_C3_NOMINAL_BOUNDARY_COMPLETION_CONTRACT
            if c3_nominal_reference is not None
            else None
        ),
        "c3_nominal_only_action_mask_contract": (
            ORDER9_NOMINAL_ONLY_ACTION_MASK_CONTRACT
            if (
                c3_nominal_reference is not None
                and args_cli.c3_action_contract is None
            )
            else None
        ),
        "c3_nominal_only_action_mask_phase_labels": (
            [
                Order9ObjectTaskPhase.RELEASE.value,
                Order9ObjectTaskPhase.RETREAT.value,
            ]
            if (
                c3_nominal_reference is not None
                and args_cli.c3_action_contract is None
            )
            else []
        ),
        "evaluation_mode": bool(args_cli.evaluation_jsonl is not None),
        "training_only_continuous_teacher_rollout": bool(
            args_cli.training_only_continuous_teacher_rollout
        ),
        "promotion_evidence_eligible": bool(
            args_cli.evaluation_jsonl is not None
            and not args_cli.training_only_continuous_teacher_rollout
            and args_cli.virtual_contact_lead_mm is None
            and diagnostic_virtual_contact_additional_lead_sweep_mm is None
            and diagnostic_contact_normal_residual_sweep_mm is None
        ),
        "deterministic_policy": bool(deterministic_policy),
        "initial_phase_zero": bool(initial_phase_zero),
        "formal_phase_zero_start": bool(args_cli.formal_phase_zero_start),
        "formal_phase_zero_state_source": (
            ORDER9_C3_FORMAL_PHASE_ZERO_STATE_SOURCE
            if args_cli.formal_phase_zero_start
            else None
        ),
        "initial_phase_elapsed_s": (
            0.0 if args_cli.formal_phase_zero_start else None
        ),
        "diagnostic_initial_phase_index": diagnostic_initial_phase_index,
        "diagnostic_initial_phase_strata": (
            None
            if diagnostic_phase_strata is None
            else [list(value) for value in diagnostic_phase_strata]
        ),
        "diagnostic_learned_phase_strata": bool(
            args_cli.diagnostic_learned_phase_strata
        ),
        "diagnostic_virtual_contact_additional_lead_sweep_mm": (
            None
            if diagnostic_virtual_contact_additional_lead_sweep_mm is None
            else list(diagnostic_virtual_contact_additional_lead_sweep_mm)
        ),
        "training_nominal_preload_deficit_version": (
            ORDER9_TRAINING_NOMINAL_PRELOAD_DEFICIT_VERSION
            if (
                training_nominal_preload_deficit_mm is not None
                and len(morphology.modules)
                == args_cli.training_nominal_preload_deficit_module_count
            )
            else None
        ),
        "training_nominal_preload_deficit_module_count": (
            args_cli.training_nominal_preload_deficit_module_count
            if training_nominal_preload_deficit_mm is not None
            else None
        ),
        "training_nominal_preload_deficit_mm": (
            None
            if training_nominal_preload_deficit_mm is None
            else list(training_nominal_preload_deficit_mm)
        ),
        "diagnostic_contact_normal_residual_sweep_mm": (
            None
            if diagnostic_contact_normal_residual_sweep_mm is None
            else list(diagnostic_contact_normal_residual_sweep_mm)
        ),
        "diagnostic_terminal_runtime_phase": (
            Order9ObjectTaskPhase.LIFT.value
            if diagnostic_virtual_contact_additional_lead_sweep_mm is not None
            else None
        ),
    }


def _actor_phase_indices(runtime_phase_indices: torch.Tensor) -> torch.Tensor:
    mapping = torch.tensor(
        ORDER9_OBJECT_TASK_ACTOR_PHASE_INDEX_BY_RUNTIME,
        device=runtime_phase_indices.device,
        dtype=runtime_phase_indices.dtype,
    )
    return mapping[runtime_phase_indices.long()]


def _condition_target_on_teacher_reference(
    target,
    *,
    teacher_reference,
    phase_index: torch.Tensor,
    scene_origins: torch.Tensor,
    task_object_position_world: torch.Tensor,
):
    if teacher_reference is None:
        return target
    source_object_position = teacher_reference.desired_object_pose_world[0, 0, :3]
    task_offset = task_object_position_world - source_object_position
    sample = teacher_reference.sample(
        phase_index=phase_index,
        phase_progress=target.phase_progress,
        position_offset_world=scene_origins + task_offset.unsqueeze(0),
    )
    return replace(
        target,
        desired_robot_root_pose_world=sample.desired_body_pose_world,
        desired_robot_root_twist_world=sample.desired_body_twist,
        nominal_joint_positions_rad=sample.nominal_joint_positions_rad,
        nominal_joint_velocities_radps=sample.nominal_joint_velocities_radps,
        desired_object_pose_world=sample.desired_object_pose_world,
        phase_goal_robot_root_pose_world=sample.phase_goal_body_pose_world,
        phase_goal_object_pose_world=sample.phase_goal_object_pose_world,
    )


def _deployable_anchor_joint_owner_mask(
    *,
    morphology: MorphologyGraph,
    physical_model,
    selected_anchor_ids,
    module_ids,
    local_joint_ids,
    contact_joint_positions_rad: torch.Tensor,
    closure_direction_rad: torch.Tensor,
) -> torch.Tensor:
    """Map each selected anchor's normal-force Jacobian to its Dock joints."""

    ordered_global_ids = tuple(
        f"module_{module_id}:{joint_id}"
        for module_id in module_ids
        for joint_id in local_joint_ids
    )
    if tuple(contact_joint_positions_rad.shape) != (
        len(module_ids),
        len(local_joint_ids),
    ) or tuple(closure_direction_rad.shape) != tuple(contact_joint_positions_rad.shape):
        raise RuntimeError("Order9 anchor owner joint reference shape differs")
    positions = {
        global_id: float(contact_joint_positions_rad.flatten()[index].item())
        for index, global_id in enumerate(ordered_global_ids)
    }
    references = resolve_mesh_backed_anchor_references(
        morphology, physical_model, selected_anchor_ids
    )
    kinematics = WholeStructureKinematics().compute(
        morphology,
        physical_model,
        positions,
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        references,
    )
    if set(ordered_global_ids) != set(kinematics.ordered_global_dock_joint_ids):
        raise RuntimeError("Order9 anchor owner Dock identity differs")
    closure_by_id = {
        global_id: float(closure_direction_rad.flatten()[index].item())
        for index, global_id in enumerate(ordered_global_ids)
    }
    anchors = {int(anchor.anchor_id): anchor for anchor in morphology.robot_anchors}
    rows = []
    for raw_anchor_id in selected_anchor_ids:
        anchor_id = int(raw_anchor_id)
        anchor = anchors[anchor_id]
        required_local_id = str(anchor.capability.get("dock_mechanism_joint_id", ""))
        required_global_id = f"module_{anchor.module_id}:{required_local_id}"
        jacobian = kinematics.anchor_jacobians[anchor_id]
        influential = {
            joint_id
            for column, joint_id in enumerate(kinematics.ordered_global_dock_joint_ids)
            if (
                joint_id == required_global_id
                or math.sqrt(sum(float(row[column]) ** 2 for row in jacobian)) > 1.0e-8
            )
            and abs(closure_by_id[joint_id]) > 1.0e-9
        }
        if not influential:
            raise RuntimeError(
                f"Order9 deployable gate anchor {anchor_id} has no moving "
                "influential Dock joint"
            )
        rows.append([global_id in influential for global_id in ordered_global_ids])
    return torch.tensor(rows, dtype=torch.bool).reshape(
        len(selected_anchor_ids), len(module_ids), len(local_joint_ids)
    )


def _canonical_joint_reference_banks(
    runtime: Order9ObjectTaskRuntime,
    *,
    module_ids,
    joint_ids,
    device: torch.device,
    dtype: torch.dtype,
    articulated_trajectory=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize the deterministic active-knot posture reference."""

    if articulated_trajectory is not None:
        return _articulated_joint_reference_banks(
            articulated_trajectory,
            phase_count=runtime.phase_count,
            module_ids=module_ids,
            joint_ids=joint_ids,
            device=device,
            dtype=dtype,
        )
    starts = []
    ends = []
    for phase_index in range(runtime.phase_count):
        reset = runtime.reset_for_phase(phase_index)
        end = runtime.target(
            phase_index,
            runtime.duration_s(phase_index),
            reset=reset,
        )
        phase_start = []
        phase_end = []
        for module_id in module_ids:
            start_row = []
            end_row = []
            for joint_id in joint_ids:
                global_id = f"module_{module_id}:{joint_id}"
                if (
                    global_id not in reset.joint_positions_rad
                    or global_id not in end.nominal_joint_positions_rad
                ):
                    raise RuntimeError(
                        "Order9 deterministic posture reference does not cover "
                        f"{global_id}; arbitrary-morphology reference generation "
                        "must be supplied before this topology is trained"
                    )
                start_row.append(float(reset.joint_positions_rad[global_id]))
                end_row.append(float(end.nominal_joint_positions_rad[global_id]))
            phase_start.append(start_row)
            phase_end.append(end_row)
        starts.append(phase_start)
        ends.append(phase_end)
    return (
        torch.tensor(starts, device=device, dtype=dtype),
        torch.tensor(ends, device=device, dtype=dtype),
    )


def _articulated_joint_reference_banks(
    trajectory,
    *,
    phase_count: int,
    module_ids,
    joint_ids,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map the complete articulated teacher posture into task-phase banks."""

    if not trajectory.knots:
        raise RuntimeError("Order9 articulated teacher trajectory is empty")
    initial_target = trajectory.knots[0].posture_target
    pregrasp_target = next(
        (
            knot.posture_target
            for knot in trajectory.knots
            if any(
                assignment.schedule_state in {"attach", "maintain", "slide"}
                for assignment in knot.contact_assignments
            )
        ),
        None,
    )
    grasp_target = next(
        (
            knot.posture_target
            for knot in reversed(trajectory.knots)
            if any(
                assignment.schedule_state in {"attach", "maintain", "slide"}
                for assignment in knot.contact_assignments
            )
        ),
        None,
    )
    if (
        initial_target is None
        or initial_target.joint_pos_target is None
        or pregrasp_target is None
        or pregrasp_target.joint_pos_target is None
        or grasp_target is None
        or grasp_target.joint_pos_target is None
    ):
        raise RuntimeError(
            "Order9 articulated teacher lacks complete posture references"
        )
    initial = initial_target.joint_pos_target
    pregrasp = pregrasp_target.joint_pos_target
    grasp = grasp_target.joint_pos_target

    def bank_row(values):
        rows = []
        for module_id in module_ids:
            row = []
            for joint_id in joint_ids:
                global_id = f"module_{module_id}:{joint_id}"
                if global_id not in values:
                    raise RuntimeError(
                        "Order9 articulated posture reference does not cover "
                        f"{global_id}"
                    )
                row.append(float(values[global_id]))
            rows.append(row)
        return rows

    initial_row = bank_row(initial)
    pregrasp_row = bank_row(pregrasp)
    grasp_row = bank_row(grasp)
    starts = []
    ends = []
    for phase_index in range(phase_count):
        if phase_index == 0:
            starts.append(initial_row)
            ends.append(pregrasp_row)
        elif phase_index == 1:
            starts.append(pregrasp_row)
            ends.append(grasp_row)
        elif phase_index == ORDER9_OBJECT_TASK_PHASES.index(
            Order9ObjectTaskPhase.RELEASE
        ):
            starts.append(grasp_row)
            ends.append(pregrasp_row)
        elif phase_index > ORDER9_OBJECT_TASK_PHASES.index(
            Order9ObjectTaskPhase.RELEASE
        ):
            starts.append(pregrasp_row)
            ends.append(pregrasp_row)
        else:
            starts.append(grasp_row)
            ends.append(grasp_row)
    return (
        torch.tensor(starts, device=device, dtype=dtype),
        torch.tensor(ends, device=device, dtype=dtype),
    )


def policy_command_joint_ids(physical) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(port.mechanical_limits["mechanism_joint_id"])
                for port in physical.dock_ports
            }
        )
    )


def _artifact_step(**values):
    pre_state = values["pre_state"]
    pre_target = values["pre_target"]
    step = values["policy_step"]
    command = step.policy_command
    allocation = step.controller_result.allocation
    reward = values["reward"]
    post = values["post_state"]
    if (step.active_knot_features is None) != (step.active_assignment_features is None):
        raise RuntimeError("Order9 rollout has an incomplete active-knot actor context")
    payload = {
        "valid": values["valid"],
        "time_s": values["pre_time"],
        "phase_index": values["pre_phase"],
        # Persist the exact phase-progress value consumed by pi_L.  The C3
        # nominal trajectory conditioner intentionally replaces the generic
        # task-timeout progress with progress along the accepted IK reference;
        # recomputing it from ``pre_elapsed / duration`` therefore makes an
        # exact behavior replay impossible, especially during contact
        # acquisition.
        "phase_progress": pre_target.phase_progress,
        "episode_serial": values["pre_serial"],
        "step_index": values["pre_step"],
        "module_pose_world": pre_state.module_pose_world,
        "module_twist_world": pre_state.module_twist_world,
        "local_joint_positions_rad": pre_state.local_joint_positions_rad,
        "local_joint_velocities_radps": pre_state.local_joint_velocities_radps,
        "robot_root_pose_world": pre_state.robot_root_pose_world,
        "robot_root_twist_world": pre_state.robot_root_twist_world,
        "object_pose_world": pre_state.object_pose_world,
        "object_twist_world": pre_state.object_twist_world,
        "desired_body_pose_world": pre_target.desired_robot_root_pose_world,
        "desired_body_twist_reference": pre_target.desired_robot_root_twist_world,
        "desired_object_pose_world": pre_target.desired_object_pose_world,
        "phase_goal_body_pose_world": (pre_target.phase_goal_robot_root_pose_world),
        "phase_goal_object_pose_world": pre_target.phase_goal_object_pose_world,
        "desired_joint_positions_rad": pre_target.nominal_joint_positions_rad,
        "desired_joint_velocities_radps": (pre_target.nominal_joint_velocities_radps),
        "selected_assignment_mask": values["selected_mask"],
        "contact_schedule_index": pre_target.contact_schedule_index,
        "actor_controller_qp_feasible": step.actor_controller_qp_feasible,
        "actor_controller_status_one_hot": step.actor_controller_status_one_hot,
        "actor_allocation_residual_norm": step.actor_allocation_residual_norm,
        "actor_task_success": step.actor_task_success,
        "global_action": step.policy_step.action,
        "joint_action": step.policy_step.joint_action,
        "previous_global_action": step.previous_global_action,
        **(
            {"applied_global_action": step.applied_global_action}
            if values["include_contact_space_action"]
            else {}
        ),
        "recurrent_state_in": step.recurrent_state_in,
        "recurrent_state_out": step.policy_step.recurrent_state,
        "old_log_prob": step.policy_step.log_prob,
        "old_value": step.policy_step.value,
        "privileged_disturbance_body": step.privileged_disturbance_body,
        "command_body_pose_world": command.desired_body_pose_world,
        "command_body_twist": command.desired_body_twist,
        "command_residual_wrench_body": command.residual_wrench_body,
        "command_joint_position_targets_rad": command.joint_position_targets_rad,
        "command_joint_velocity_targets_radps": command.joint_velocity_targets_radps,
        "command_joint_torque_bias_nm": command.joint_torque_bias_nm,
        "controller_desired_wrench_body": step.controller_result.desired_wrench_body,
        "rotor_thrusts_n": allocation.rotor_thrusts_n,
        "vectoring_joint_targets_rad": allocation.vectoring_joint_targets_rad,
        "allocation_residual_norm": allocation.residual_norm,
        "qp_feasible": allocation.feasible,
        "rotor_saturation": values["rotor_saturation"],
        "selected_contact_forces_world": values[
            "contact"
        ].selected_contact_forces_world,
        "selected_contact_wrenches_contact": values[
            "selected_contact_wrenches_contact"
        ],
        "wrench_lower_contact": values["wrench_range"].lower_contact,
        "wrench_upper_contact": values["wrench_range"].upper_contact,
        "wrench_bound_mask": values["wrench_bound_mask"],
        "prohibited_collision": values["contact"].prohibited_collision,
        "reward": reward.reward,
        "reward_terms": torch.stack(
            [reward.terms[name] for name in values["reward_names"]], dim=-1
        ),
        "phase_success": reward.phase_success,
        "terminal": values["terminal"],
        "truncated": values["truncated"],
        "bootstrap_value": values["bootstrap"],
        "post_robot_root_pose_world": post.robot_root_pose_world,
        "post_robot_root_twist_world": post.robot_root_twist_world,
        "post_local_joint_positions_rad": post.local_joint_positions_rad,
        "post_local_joint_velocities_radps": post.local_joint_velocities_radps,
        "post_object_pose_world": post.object_pose_world,
        "post_object_twist_world": post.object_twist_world,
    }
    if values["include_contact_compression_residual_action"]:
        payload["contact_compression_residual_action"] = (
            step.policy_step.contact_compression_residual_action
        )
    if values["include_contact_space_action"]:
        if (
            step.contact_slot_features is None
            or step.contact_slot_owner_module_indices is None
            or step.contact_slot_mask is None
            or step.contact_constraint_weight is None
        ):
            raise RuntimeError("Order9 v7 rollout lacks contact-space actor evidence")
        payload.update(
            {
                "contact_space_residual_action": (
                    step.policy_step.contact_space_residual_action
                ),
                "actor_contact_slot_features": step.contact_slot_features,
                "actor_contact_slot_owner_indices": (
                    step.contact_slot_owner_module_indices
                ),
                "actor_contact_slot_mask": step.contact_slot_mask,
                "contact_constraint_weight": step.contact_constraint_weight,
            }
        )
    if step.active_knot_features is not None:
        payload.update(
            {
                "actor_active_knot_features": step.active_knot_features,
                "actor_active_assignment_features": (step.active_assignment_features),
            }
        )
    return payload


def _contact_compression_action_adapter_version(
    policy_version: str,
    *,
    c3_action_contract: str | None = None,
) -> str:
    if (
        c3_action_contract is not None
        and order9_c3_action_contract_uses_contact_space(c3_action_contract)
    ):
        if policy_version not in ORDER9_CONTACT_SPACE_PI_L_POLICY_VERSIONS:
            raise ValueError(
                "Order9 contact-space adapter requires the v7 pi_L actor"
            )
        return ORDER9_CONTACT_SPACE_ACTION_ADAPTER_VERSION
    if (
        c3_action_contract is not None
        and order9_c3_action_contract_uses_full_policy(c3_action_contract)
    ):
        if (
            policy_version
            != ORDER9_MORPHOLOGY_INVARIANT_COMPRESSION_PI_L_POLICY_VERSION
        ):
            raise ValueError(
                "Order9 independent compression requires the v6 pi_L actor"
            )
        return ORDER9_INDEPENDENT_COMPRESSION_ACTION_ADAPTER_VERSION
    if (
        policy_version
        == ORDER9_MORPHOLOGY_INVARIANT_COMPRESSION_PI_L_POLICY_VERSION
    ):
        return ORDER9_MORPHOLOGY_INVARIANT_COMPRESSION_ACTION_ADAPTER_VERSION
    return ORDER9_CONTACT_COMPRESSION_ACTION_ADAPTER_VERSION


def _nominal_diagnostic_artifact_step(
    *,
    nominal_step,
    policy_runtime: Order9TensorPiLRuntime,
    module_count: int,
):
    """Add zero actor evidence to an actor-free nominal diagnostic step.

    Tensor rollout files retain one fixed schema even for the nominal-QPID
    admission diagnostic.  These fields explicitly record that no actor action,
    recurrence, log probability, or value estimate was consumed.  They are
    diagnostic transport data and are never admitted as on-policy PPO samples.
    """

    batch_size = int(nominal_step.policy_command.desired_body_pose_world.shape[0])
    device = nominal_step.policy_command.desired_body_pose_world.device
    dtype = nominal_step.policy_command.desired_body_pose_world.dtype
    zero_scalar = torch.zeros(batch_size, device=device, dtype=dtype)
    contact_basis = policy_runtime.contact_space_action_basis
    contact_slot_features = (
        None
        if contact_basis is None
        else contact_basis.contact_slot_features.unsqueeze(0).expand(
            batch_size, -1, -1
        )
    )
    if (
        contact_slot_features is not None
        and contact_slot_features.shape[-1]
        < int(policy_runtime.config.contact_space_feature_dim)
    ):
        contact_slot_features = torch.cat(
            (
                contact_slot_features,
                torch.zeros(
                    (
                        batch_size,
                        contact_slot_features.shape[1],
                        int(policy_runtime.config.contact_space_feature_dim)
                        - contact_slot_features.shape[-1],
                    ),
                    device=device,
                    dtype=dtype,
                ),
            ),
            dim=-1,
        )
    contact_slot_owner_module_indices = (
        None
        if contact_basis is None
        else contact_basis.contact_slot_owner_module_indices.unsqueeze(0).expand(
            batch_size, -1
        )
    )
    contact_slot_mask = (
        None
        if contact_basis is None
        else contact_basis.contact_slot_mask.unsqueeze(0).expand(batch_size, -1)
    )
    actor = SimpleNamespace(
        action=torch.zeros(
            (batch_size, ORDER9_GLOBAL_ACTION_SIZE),
            device=device,
            dtype=dtype,
        ),
        joint_action=torch.zeros(
            (
                batch_size,
                module_count,
                3 * int(policy_runtime.config.max_local_joint_slots),
            ),
            device=device,
            dtype=dtype,
        ),
        recurrent_state=torch.zeros_like(policy_runtime.recurrent_state),
        log_prob=zero_scalar.clone(),
        value=zero_scalar.clone(),
        contact_compression_residual_action=zero_scalar.clone(),
        contact_space_residual_action=torch.zeros(
            (
                batch_size,
                int(policy_runtime.config.max_contact_slots),
                6,
            ),
            device=device,
            dtype=dtype,
        ),
    )
    return SimpleNamespace(
        control_model=nominal_step.control_model,
        active_knot_features=torch.zeros(
            (batch_size, len(ORDER9_ACTIVE_KNOT_GLOBAL_FEATURE_NAMES)),
            device=device,
            dtype=dtype,
        ),
        active_assignment_features=torch.zeros(
            (
                batch_size,
                module_count,
                len(ORDER9_ACTIVE_ASSIGNMENT_FEATURE_NAMES),
            ),
            device=device,
            dtype=dtype,
        ),
        # A v7 checkpoint still determines the fixed artifact schema even
        # though this diagnostic bypasses its actor.  Preserve the immutable
        # contact-slot description and use zero constraint weight to record
        # that no projected actor action was applied.
        contact_slot_features=contact_slot_features,
        contact_slot_owner_module_indices=contact_slot_owner_module_indices,
        contact_slot_mask=contact_slot_mask,
        contact_constraint_weight=(
            None
            if contact_basis is None
            else torch.zeros(batch_size, device=device, dtype=dtype)
        ),
        previous_global_action=torch.zeros(
            (batch_size, ORDER9_GLOBAL_ACTION_SIZE),
            device=device,
            dtype=dtype,
        ),
        applied_global_action=torch.zeros(
            (batch_size, ORDER9_GLOBAL_ACTION_SIZE),
            device=device,
            dtype=dtype,
        ),
        recurrent_state_in=torch.zeros_like(policy_runtime.recurrent_state),
        actor_controller_qp_feasible=(policy_runtime.controller_qp_feasible.clone()),
        actor_controller_status_one_hot=(
            policy_runtime.controller_status_one_hot.clone()
        ),
        actor_allocation_residual_norm=(
            policy_runtime.allocation_residual_norm.clone()
        ),
        actor_task_success=policy_runtime.task_success.clone(),
        policy_step=actor,
        policy_command=nominal_step.policy_command,
        controller_result=nominal_step.controller_result,
        privileged_disturbance_body=torch.zeros(
            (batch_size, 6), device=device, dtype=dtype
        ),
    )


def _reset_goal_distance_for_phase_transition(
    state: Order9TensorRewardState,
    *,
    env_ids: torch.Tensor,
    object_pose_world: torch.Tensor,
    desired_object_pose_world: torch.Tensor,
) -> Order9TensorRewardState:
    if env_ids.numel() == 0:
        return state
    values = {field.name: getattr(state, field.name).clone() for field in fields(state)}
    values["previous_object_goal_distance_m"][env_ids] = torch.linalg.vector_norm(
        object_pose_world[env_ids, :3] - desired_object_pose_world[env_ids, :3],
        dim=-1,
    )
    return Order9TensorRewardState(**values)


def _object_pose(obj) -> torch.Tensor:
    # Task/object poses are expressed at the object link/geometry frame.  The
    # separately randomized estimator CoM is transformed from this frame in
    # the controller; exposing PhysX's true CoM pose here would leak it.
    return _torch(obj.data.root_pose_w)


def _selected_contact_wrenches_contact(
    object_sensor,
    *,
    io: Order9TensorIsaacIO,
    contact_frame_pose_world: torch.Tensor,
    selected_assignment_mask: torch.Tensor,
    batch_size: int,
    filter_count: int,
    dt_s: float,
) -> torch.Tensor:
    (
        normal_force,
        normal_point,
        normal_vector,
        _normal_separation,
        normal_count,
        normal_start,
    ) = object_sensor.contact_view.get_contact_data(dt_s)
    normal_force_tensor = wp.to_torch(normal_force).reshape(-1)
    normal_point_tensor = wp.to_torch(normal_point).reshape(-1, 3)
    normal_vector_tensor = wp.to_torch(normal_vector).reshape(-1, 3)
    # PhysX reuses its pair count/start storage for detailed normal-contact
    # and friction queries.  Preserve the normal ranges before the friction
    # call overwrites that shared backend buffer.
    normal_count_tensor = _pair_contact_matrix(
        normal_count,
        batch_size=batch_size,
        filter_count=filter_count,
        label="normal count",
    ).clone()
    normal_start_tensor = _pair_contact_matrix(
        normal_start,
        batch_size=batch_size,
        filter_count=filter_count,
        label="normal start",
    ).clone()
    (
        friction_force,
        friction_point,
        friction_count,
        friction_start,
    ) = object_sensor.contact_view.get_friction_data(dt_s)
    return io.reduce_contact_wrenches(
        normal_force_magnitudes_n=normal_force_tensor,
        normal_points_world=normal_point_tensor,
        normal_vectors_world=normal_vector_tensor,
        normal_contact_counts=normal_count_tensor,
        normal_contact_starts=normal_start_tensor,
        friction_forces_world=wp.to_torch(friction_force).reshape(-1, 3),
        friction_points_world=wp.to_torch(friction_point).reshape(-1, 3),
        friction_contact_counts=_pair_contact_matrix(
            friction_count,
            batch_size=batch_size,
            filter_count=filter_count,
            label="friction count",
        ),
        friction_contact_starts=_pair_contact_matrix(
            friction_start,
            batch_size=batch_size,
            filter_count=filter_count,
            label="friction start",
        ),
        contact_frame_pose_world=contact_frame_pose_world,
        selected_assignment_mask=selected_assignment_mask,
    )


def _pair_contact_matrix(
    value,
    *,
    batch_size: int,
    filter_count: int,
    label: str,
) -> torch.Tensor:
    tensor = wp.to_torch(value)
    expected = batch_size * filter_count
    if tensor.numel() != expected:
        raise RuntimeError(
            f"Order9 raw {label} size differs: {tensor.numel()} != {expected}"
        )
    return tensor.reshape(batch_size, filter_count)


def _validate_object_mass_properties(
    obj,
    *,
    expected_mass_kg: float,
    expected_inertia_kgm2: tuple[float, ...],
    expected_com_object: tuple[float, ...],
) -> dict[str, object]:
    mass = _torch(obj.data.body_mass)[:, 0]
    inertia = _torch(obj.data.body_inertia)[:, 0].reshape(-1, 3, 3)
    com = _torch(obj.data.body_com_pose_b)[:, 0, :3]
    expected_inertia = torch.tensor(
        [
            [
                expected_inertia_kgm2[0],
                expected_inertia_kgm2[1],
                expected_inertia_kgm2[2],
            ],
            [
                expected_inertia_kgm2[1],
                expected_inertia_kgm2[3],
                expected_inertia_kgm2[4],
            ],
            [
                expected_inertia_kgm2[2],
                expected_inertia_kgm2[4],
                expected_inertia_kgm2[5],
            ],
        ],
        device=inertia.device,
        dtype=inertia.dtype,
    )
    expected_eigenvalues = torch.linalg.eigvalsh(expected_inertia)
    actual_eigenvalues = torch.linalg.eigvalsh(inertia)
    expected_com = torch.tensor(
        expected_com_object,
        device=com.device,
        dtype=com.dtype,
    )
    if not torch.allclose(
        mass,
        torch.full_like(mass, float(expected_mass_kg)),
        rtol=5.0e-4,
        atol=1.0e-6,
    ):
        raise RuntimeError("Order9 Isaac object mass readback differs from TaskSpec")
    if not torch.allclose(
        actual_eigenvalues,
        expected_eigenvalues.reshape(1, 3).expand_as(actual_eigenvalues),
        rtol=5.0e-3,
        atol=1.0e-6,
    ):
        raise RuntimeError("Order9 Isaac object inertia readback differs from TaskSpec")
    if not torch.allclose(
        com,
        expected_com.reshape(1, 3).expand_as(com),
        rtol=0.0,
        atol=1.0e-5,
    ):
        raise RuntimeError("Order9 Isaac object CoM readback differs from TaskSpec")
    return {
        "mass_kg": float(mass[0].item()),
        "center_of_mass_object": [float(value) for value in com[0].tolist()],
        "inertia_eigenvalues_kgm2": [
            float(value) for value in actual_eigenvalues[0].tolist()
        ],
        "matches_task_spec": True,
    }


def _object_twist(obj) -> torch.Tensor:
    return torch.cat(
        (_torch(obj.data.root_lin_vel_w), _torch(obj.data.root_ang_vel_w)),
        dim=-1,
    )


def _selected_grasp_frame_positions(
    body_pose_world: torch.Tensor,
    anchor_local_pose: torch.Tensor,
) -> torch.Tensor:
    """Compose only the position of each selected grasp-contact frame."""

    if (
        body_pose_world.ndim != 3
        or body_pose_world.shape[-1] != 7
        or anchor_local_pose.shape != body_pose_world.shape[1:]
    ):
        raise ValueError("selected grasp-frame pose shapes differ")
    vector = (
        anchor_local_pose[:, :3].unsqueeze(0).expand(body_pose_world.shape[0], -1, -1)
    )
    rotated = _quaternion_rotate_vector(body_pose_world[..., 3:7], vector)
    return body_pose_world[..., :3] + rotated


def _deployable_selected_contact_feedback_inputs(
    *,
    selected_body_pose_world: torch.Tensor,
    selected_body_twist_world: torch.Tensor,
    selected_anchor_local_pose: torch.Tensor,
    object_pose_world: torch.Tensor,
    object_twist_world: torch.Tensor,
    contact_frame_pose_world: torch.Tensor,
    contact_normal_world: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve deployable grasp-frame distance and relative twist.

    Hardware obtains the same quantities from base/object state estimation,
    joint encoders/FK, and the planned object contact frames.  No contact
    sensor or PhysX force tensor is involved.
    """

    if selected_body_pose_world.ndim != 3 or selected_body_pose_world.shape[-1] != 7:
        raise ValueError("contact feedback selected body pose shape differs")
    batch, contact_count, _ = selected_body_pose_world.shape
    expected = {
        "selected_body_twist_world": (batch, contact_count, 6),
        "selected_anchor_local_pose": (contact_count, 7),
        "object_pose_world": (batch, 7),
        "object_twist_world": (batch, 6),
        "contact_frame_pose_world": (batch, contact_count, 7),
        "contact_normal_world": (batch, contact_count, 3),
    }
    local_values = locals()
    for name, shape in expected.items():
        if tuple(local_values[name].shape) != shape:
            raise ValueError(f"contact feedback {name} shape differs")
    grasp_position = _selected_grasp_frame_positions(
        selected_body_pose_world, selected_anchor_local_pose
    )
    surface_position = contact_frame_pose_world[..., :3]
    signed_surface_distance = (
        (grasp_position - surface_position) * contact_normal_world
    ).sum(dim=-1)

    body_offset = grasp_position - selected_body_pose_world[..., :3]
    body_point_velocity = selected_body_twist_world[..., :3] + torch.cross(
        selected_body_twist_world[..., 3:6], body_offset, dim=-1
    )
    object_offset = grasp_position - object_pose_world[:, None, :3]
    object_point_velocity = object_twist_world[:, None, :3] + torch.cross(
        object_twist_world[:, None, 3:6].expand(-1, contact_count, -1),
        object_offset,
        dim=-1,
    )
    relative_linear_world = body_point_velocity - object_point_velocity
    relative_angular_world = (
        selected_body_twist_world[..., 3:6]
        - object_twist_world[:, None, 3:6]
    )
    frame_quaternion = contact_frame_pose_world[..., 3:7]
    inverse_quaternion = torch.cat(
        (-frame_quaternion[..., :3], frame_quaternion[..., 3:4]), dim=-1
    )
    relative_linear_contact = _quaternion_rotate_vector(
        inverse_quaternion, relative_linear_world
    )
    relative_angular_contact = _quaternion_rotate_vector(
        inverse_quaternion, relative_angular_world
    )
    return signed_surface_distance, torch.cat(
        (relative_linear_contact, relative_angular_contact), dim=-1
    )


def _quaternion_rotate_vector(
    quaternion: torch.Tensor, vector: torch.Tensor
) -> torch.Tensor:
    """Rotate vectors by normalized XYZW quaternions."""

    if quaternion.shape[:-1] != vector.shape[:-1] or (
        quaternion.shape[-1] != 4 or vector.shape[-1] != 3
    ):
        raise ValueError("quaternion/vector rotation shapes differ")
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    xyz = quaternion[..., :3]
    w = quaternion[..., 3:4]
    first = torch.cross(xyz, vector, dim=-1)
    return vector + 2.0 * torch.cross(xyz, first + w * vector, dim=-1)


def _torch(value) -> torch.Tensor:
    return value.torch if hasattr(value, "torch") else value


def _translate_pose(pose, translation):
    return (
        float(pose[0]) + translation[0],
        float(pose[1]) + translation[1],
        float(pose[2]) + translation[2],
        *tuple(float(value) for value in pose[3:7]),
    )


def _yaw(quaternion) -> float:
    x, y, z, w = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class _TeeTextStream:
    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, value: str) -> int:
        for stream in self._streams:
            stream.write(value)
        return len(value)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _run_and_report() -> None:
    result = main()
    print(_RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)


def _persistent_bucket_jobs(path: str | Path) -> tuple[dict[str, object], ...]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != (
        "order9_persistent_same_morphology_bucket_jobs_v1"
    ):
        raise ValueError("persistent bucket job manifest version differs")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or len(jobs) < 2:
        raise ValueError("persistent bucket job manifest requires multiple jobs")
    normalized: list[dict[str, object]] = []
    names: set[str] = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("persistent bucket job must be an object")
        name = job.get("name")
        argv = job.get("argv")
        log_path = job.get("log_path")
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or not isinstance(argv, list)
            or not argv
            or any(not isinstance(value, str) for value in argv)
            or not isinstance(log_path, str)
            or not log_path
        ):
            raise ValueError("persistent bucket job fields are invalid")
        names.add(name)
        normalized.append(job)
    return tuple(normalized)


_exit_code = 1
try:
    if args_cli.persistent_bucket_jobs is None:
        _run_and_report()
    else:
        _base_launcher = {
            name: getattr(args_cli, name, None)
            for name in ("device", "headless", "enable_cameras", "livestream")
        }
        for _job in _persistent_bucket_jobs(args_cli.persistent_bucket_jobs):
            _job_args = _parser().parse_args(list(_job["argv"]))
            if _job_args.persistent_bucket_jobs is not None:
                raise ValueError("nested persistent bucket jobs are forbidden")
            # AppLauncher resolves environment-derived defaults (notably
            # headless mode and ``cuda`` -> ``cuda:0``) in-place on the first
            # namespace.  Re-parsing the original argv reproduces the raw
            # defaults rather than those resolved values, so each job must
            # inherit the already-running application contract.
            for name, value in _base_launcher.items():
                setattr(_job_args, name, value)
            args_cli = _job_args
            c3_execution_bundle_runtime = _bind_c3_execution_bundle(args_cli)
            _job_log_path = Path(str(_job["log_path"])).resolve()
            _job_log_path.parent.mkdir(parents=True, exist_ok=True)
            with _job_log_path.open("w", encoding="utf-8") as _job_log:
                _tee_stdout = _TeeTextStream(sys.stdout, _job_log)
                _tee_stderr = _TeeTextStream(sys.stderr, _job_log)
                with contextlib.redirect_stdout(_tee_stdout), contextlib.redirect_stderr(
                    _tee_stderr
                ):
                    print(
                        "ORDER9_COMMAND="
                        + shlex.join(
                            [
                                sys.executable,
                                str(Path(__file__).resolve()),
                                *list(_job["argv"]),
                            ]
                        ),
                        flush=True,
                    )
                    _run_and_report()
            # ``main`` clears the SimulationContext.  Releasing the completed
            # scene before creating the next one prevents stale USD/sensor
            # wrappers from retaining the previous stage.
            gc.collect()
    _exit_code = 0
except BaseException as _error:
    print(
        "ORDER9_ROLLOUT_ERROR="
        + json.dumps(
            {"error_type": type(_error).__name__, "error": str(_error)},
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )
    traceback.print_exc()
finally:
    simulation_app.close()
raise SystemExit(_exit_code)
