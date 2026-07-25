#!/usr/bin/env python3
from __future__ import annotations

"""Validate ``raw pi_H -> posture resolver -> migrated pi_L -> QPID`` on CPU.

The input fixture is prepared by ``order9_benchmark_posture_resolver.py``.
This diagnostic deliberately uses exact model/FK states rather than Isaac:

* the raw trajectory must remain joint-target free;
* the resolver must reproduce the fixture's dense nominal trajectory;
* the C3 active-knot initializer must exactly preserve the promoted C2 actor;
* zero policy residual must reproduce the resolver's nominal q/qdot;
* the learned C2 action must remain a bounded correction around that nominal;
* the batched QPID result must remain finite.

Real collision, contact, tracking, and task success still require Isaac.
"""

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
import sys
from typing import Any

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.feasibility.articulated_reachability import (  # noqa: E402
    base_pose_for_centroidal_target,
)
from amsrr.irg.envelope_extractor import (  # noqa: E402
    InteractionEnvelopeExtractor,
)
from amsrr.irg.irg_builder import IRGBuilder  # noqa: E402
from amsrr.policies.high_level_policy_base import (  # noqa: E402
    HighLevelPolicyContext,
)
from amsrr.policies.order9_low_level_policy import (  # noqa: E402
    ORDER9_GLOBAL_ACTION_SIZE,
    Order9ActiveKnotPhaseConditionedActorCritic,
    Order9PhaseConditionedActorCritic,
)
from amsrr.robot_model.physical_model_builder import (  # noqa: E402
    build_physical_model_from_config,
)
from amsrr.robot_model.whole_structure_kinematics import (  # noqa: E402
    WholeStructureKinematics,
)
from amsrr.schemas.common import SchemaValidationError  # noqa: E402
from amsrr.schemas.contact_candidates import (  # noqa: E402
    ContactCandidateSet,
)
from amsrr.schemas.morphology import MorphologyGraph  # noqa: E402
from amsrr.schemas.policies import (  # noqa: E402
    ContactWrenchTrajectory,
    InteractionKnot,
)
from amsrr.schemas.task_spec import TaskSpec  # noqa: E402
from amsrr.simulation.order9_tensor_isaac_io import (  # noqa: E402
    Order9TensorIsaacState,
)
from amsrr.simulation.order9_tensor_object_task import (  # noqa: E402
    ORDER9_CONTACT_SCHEDULE_APPROACH,
    ORDER9_CONTACT_SCHEDULE_ATTACH,
    ORDER9_CONTACT_SCHEDULE_MAINTAIN,
    ORDER9_CONTACT_SCHEDULE_RELEASE,
    Order9TensorObjectTaskTarget,
)
from amsrr.training.order9_checkpoints import (  # noqa: E402
    load_order9_policy_checkpoint,
)
from amsrr.training.order9_posture_resolver import (  # noqa: E402
    Order9PostureResolverConfig,
    Order9PostureTrajectoryResolver,
)
from amsrr.training.order9_tensor_pi_l_runtime import (  # noqa: E402
    Order9TensorPiLRuntime,
)
from amsrr.utils.hashing import hash_file, stable_hash  # noqa: E402


REPORT_VERSION = "order9_posture_pi_l_cpu_integration_v2"
TRAJECTORY_HASH_DECIMAL_PLACES = 9
DEFAULT_FIXTURE = (
    "artifacts/p4_full/order9/posture_resolver/"
    "c3a_12_case_fixture.json"
)
DEFAULT_MIGRATION_MANIFEST = (
    "artifacts/p4_full/order9/c3_preparation/"
    "pi_l_active_knot_initializer_manifest.json"
)
DEFAULT_OUTPUT = (
    "artifacts/p4_full/order9/posture_resolver/"
    "pc_pi_l_integration.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--migration-manifest", default=DEFAULT_MIGRATION_MANIFEST
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> int:
    args = _parser().parse_args()
    fixture_path = _resolve(args.fixture)
    manifest_path = _resolve(args.migration_manifest)
    output_path = _resolve(args.output)
    report = validate_posture_pi_l_integration(
        fixture_path=fixture_path,
        migration_manifest_path=manifest_path,
        device=args.device,
        case_limit=args.case_limit,
    )
    _write_json(output_path, report)
    print(
        "ORDER9_POSTURE_PI_L_INTEGRATION="
        + json.dumps(
            {
                "path": _portable(output_path),
                "sha256": hash_file(output_path),
                "passed": report["passed"],
                "case_count": report["case_count"],
                "knot_count": report["aggregate"]["knot_count"],
                "maximum_c2_c3_action_error": report["aggregate"][
                    "maximum_c2_c3_action_error"
                ],
                "maximum_zero_residual_nominal_error": report["aggregate"][
                    "maximum_zero_residual_nominal_error"
                ],
                "qp_feasible_count": report["aggregate"][
                    "qp_feasible_count"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def validate_posture_pi_l_integration(
    *,
    fixture_path: Path,
    migration_manifest_path: Path,
    device: str | torch.device = "cpu",
    case_limit: int | None = None,
) -> dict[str, Any]:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        migration_manifest_path.read_text(encoding="utf-8")
    )
    cases = list(fixture.get("cases", ()))
    if case_limit is not None:
        if case_limit < 1:
            raise ValueError("case_limit must be positive")
        cases = cases[:case_limit]
    if not cases:
        raise SchemaValidationError(
            "posture/pi_L integration fixture contains no cases"
        )
    if manifest.get("behavior_preserving_initialization") is not True:
        raise SchemaValidationError(
            "C3 pi_L migration is not behavior-preserving"
        )
    source_checkpoint = _resolve(str(manifest["source_checkpoint_path"]))
    target_checkpoint = _resolve(str(manifest["target_checkpoint_path"]))
    if hash_file(source_checkpoint) != manifest["source_checkpoint_sha256"]:
        raise SchemaValidationError("C2 source checkpoint hash differs")
    if hash_file(target_checkpoint) != manifest["target_checkpoint_sha256"]:
        raise SchemaValidationError("C3 initializer checkpoint hash differs")

    runtime_device = torch.device(device)
    source = load_order9_policy_checkpoint(
        source_checkpoint,
        device=runtime_device,
        expected_sha256=manifest["source_checkpoint_sha256"],
    )
    target = load_order9_policy_checkpoint(
        target_checkpoint,
        device=runtime_device,
        expected_sha256=manifest["target_checkpoint_sha256"],
    )
    if not isinstance(source.model, Order9PhaseConditionedActorCritic) or (
        isinstance(
            source.model, Order9ActiveKnotPhaseConditionedActorCritic
        )
    ):
        raise SchemaValidationError(
            "posture/pi_L integration source is not the promoted C2 actor"
        )
    if not isinstance(
        target.model, Order9ActiveKnotPhaseConditionedActorCritic
    ):
        raise SchemaValidationError(
            "posture/pi_L integration target is not the C3 actor"
        )
    shared_parameter_error = _maximum_shared_parameter_error(
        source.model, target.model
    )
    if shared_parameter_error != 0.0:
        raise SchemaValidationError(
            "C2/C3 shared pi_L parameters differ"
        )

    physical_path = _resolve(str(fixture["robot_model_config_path"]))
    physical = build_physical_model_from_config(physical_path)
    if physical.stable_hash() != fixture["physical_model_hash"]:
        raise SchemaValidationError(
            "posture/pi_L fixture PhysicalModel hash differs"
        )
    task = TaskSpec.from_dict(fixture["task_spec"])
    if task.stable_hash() != fixture["task_spec_hash"]:
        raise SchemaValidationError(
            "posture/pi_L fixture TaskSpec hash differs"
        )
    built = IRGBuilder().build_with_scene_graph(task)
    envelope = InteractionEnvelopeExtractor().extract(built.irg)
    resolver_values = fixture["resolver_config"]
    resolver_config = Order9PostureResolverConfig(
        output_rate_hz=float(resolver_values["output_rate_hz"]),
        joint_velocity_fraction=float(
            resolver_values["joint_velocity_fraction"]
        ),
    )

    case_reports = []
    all_action_errors: list[float] = []
    all_command_errors: list[float] = []
    all_zero_errors: list[float] = []
    all_joint_relation_errors: list[float] = []
    all_qp: list[bool] = []
    for case_index, case in enumerate(cases, start=1):
        morphology = MorphologyGraph.from_dict(case["morphology_graph"])
        candidates = ContactCandidateSet.from_dict(
            case["contact_candidate_set"]
        )
        raw = ContactWrenchTrajectory.from_dict(
            case["raw_pi_h_trajectory"]
        )
        _require_raw_pi_h_without_joint_targets(raw)
        context = HighLevelPolicyContext(
            built.irg,
            envelope,
            morphology,
            candidates,
        )
        resolution = Order9PostureTrajectoryResolver(
            physical, config=resolver_config
        ).resolve(
            context=context,
            raw_trajectory=raw,
            initial_joint_positions_rad=case[
                "initial_joint_positions_rad"
            ],
        )
        exact_resolved_hash_match = (
            resolution.evidence.resolved_trajectory_hash
            == case["expected_resolved_trajectory_hash"]
        )
        expected_quantized_hash = case.get(
            "expected_resolved_trajectory_quantized_hash"
        )
        actual_quantized_hash = _quantized_stable_hash(
            resolution.trajectory.to_dict()
        )
        quantized_resolved_hash_match = (
            isinstance(expected_quantized_hash, str)
            and bool(expected_quantized_hash)
            and actual_quantized_hash == expected_quantized_hash
        )
        if (
            resolution.evidence.raw_trajectory_hash
            != case["expected_raw_trajectory_hash"]
            or (
                not exact_resolved_hash_match
                and not quantized_resolved_hash_match
            )
        ):
            raise SchemaValidationError(
                f"{case['case_id']} posture resolution differs from fixture"
            )
        source_runtime = Order9TensorPiLRuntime(
            morphology_graph=morphology,
            physical_model=physical,
            policy=source.model,
            batch_size=1,
            device=runtime_device,
        )
        target_runtime = Order9TensorPiLRuntime(
            morphology_graph=morphology,
            physical_model=physical,
            policy=target.model,
            batch_size=1,
            device=runtime_device,
            active_knot_trajectory=resolution.trajectory,
        )
        _resolved_raw_knots(
            raw, resolution.trajectory
        )
        evaluated_knots = list(resolution.trajectory.knots)
        action_errors: list[float] = []
        command_errors: list[float] = []
        zero_errors: list[float] = []
        joint_relation_errors: list[float] = []
        qp_feasible: list[bool] = []
        for knot in evaluated_knots:
            state, task_target, phase_index = _tensor_inputs(
                task=task,
                morphology=morphology,
                physical_model=physical,
                runtime=target_runtime,
                knot=knot,
                device=runtime_device,
            )
            compute_kwargs = {
                "time_s": torch.tensor(
                    [float(knot.t_rel_s)],
                    device=runtime_device,
                    dtype=torch.float32,
                ),
                "phase_index": torch.tensor(
                    [phase_index],
                    device=runtime_device,
                    dtype=torch.long,
                ),
                "task_target": task_target,
                "state": state,
                "estimated_payload_mass_kg": torch.zeros(
                    (1,), device=runtime_device, dtype=torch.float32
                ),
                "estimated_payload_inertia_body": torch.zeros(
                    (1, 6), device=runtime_device, dtype=torch.float32
                ),
                "payload_active": torch.zeros(
                    (1,), device=runtime_device, dtype=torch.bool
                ),
                "deterministic": True,
            }
            source_step = source_runtime.compute(**compute_kwargs)
            target_step = target_runtime.compute(**compute_kwargs)
            action_error = max(
                _tensor_error(
                    source_step.policy_step.action,
                    target_step.policy_step.action,
                ),
                _tensor_error(
                    source_step.policy_step.joint_action,
                    target_step.policy_step.joint_action,
                ),
                _tensor_error(
                    source_step.policy_step.value,
                    target_step.policy_step.value,
                ),
            )
            command_error = max(
                _tensor_error(
                    source_step.policy_command.desired_body_pose_world,
                    target_step.policy_command.desired_body_pose_world,
                ),
                _tensor_error(
                    source_step.policy_command.desired_body_twist,
                    target_step.policy_command.desired_body_twist,
                ),
                _tensor_error(
                    source_step.policy_command.residual_wrench_body,
                    target_step.policy_command.residual_wrench_body,
                ),
                _tensor_error(
                    source_step.policy_command.joint_position_targets_rad,
                    target_step.policy_command.joint_position_targets_rad,
                ),
                _tensor_error(
                    source_step.policy_command.joint_velocity_targets_radps,
                    target_step.policy_command.joint_velocity_targets_radps,
                ),
                _tensor_error(
                    source_step.policy_command.joint_torque_bias_nm,
                    target_step.policy_command.joint_torque_bias_nm,
                ),
            )
            zero_error = _zero_residual_nominal_error(
                target_runtime,
                task_target=task_target,
                control_model=target_step.control_model,
            )
            relation_error = _learned_joint_relation_error(
                target_runtime,
                step=target_step,
                task_target=task_target,
            )
            _require_finite_pi_l_step(target_step)
            action_errors.append(action_error)
            command_errors.append(command_error)
            zero_errors.append(zero_error)
            joint_relation_errors.append(relation_error)
            qp_feasible.append(
                bool(target_step.controller_result.allocation.feasible.item())
            )
        maximum_action_error = max(action_errors, default=math.inf)
        maximum_command_error = max(command_errors, default=math.inf)
        maximum_zero_error = max(zero_errors, default=math.inf)
        maximum_relation_error = max(
            joint_relation_errors, default=math.inf
        )
        passed = (
            maximum_action_error <= 1.0e-7
            and maximum_command_error <= 1.0e-7
            and maximum_zero_error <= 1.0e-7
            and maximum_relation_error <= 1.0e-6
        )
        case_reports.append(
            {
                "case_id": case["case_id"],
                "module_count": case["module_count"],
                "topology": case["topology"],
                "raw_trajectory_hash": (
                    resolution.evidence.raw_trajectory_hash
                ),
                "resolved_trajectory_hash": (
                    resolution.evidence.resolved_trajectory_hash
                ),
                "resolved_trajectory_quantized_hash": (
                    actual_quantized_hash
                ),
                "exact_resolved_trajectory_hash_match": (
                    exact_resolved_hash_match
                ),
                "quantized_resolved_trajectory_hash_match": (
                    quantized_resolved_hash_match
                ),
                "resolved_knot_count": len(resolution.trajectory.knots),
                "evaluated_resolved_knot_count": len(evaluated_knots),
                "maximum_c2_c3_action_error": maximum_action_error,
                "maximum_c2_c3_command_error": maximum_command_error,
                "maximum_zero_residual_nominal_error": maximum_zero_error,
                "maximum_learned_joint_relation_error": (
                    maximum_relation_error
                ),
                "qp_feasible_count": sum(qp_feasible),
                "qp_evaluated_count": len(qp_feasible),
                "passed": passed,
            }
        )
        all_action_errors.extend(action_errors)
        all_command_errors.extend(command_errors)
        all_zero_errors.extend(zero_errors)
        all_joint_relation_errors.extend(joint_relation_errors)
        all_qp.extend(qp_feasible)
        print(
            f"ORDER9_POSTURE_PI_L_CASE={case_index}/{len(cases)}:"
            f"{case['case_id']}:{'pass' if passed else 'fail'}",
            flush=True,
        )

    aggregate = {
        "knot_count": len(all_action_errors),
        "maximum_c2_c3_action_error": max(
            all_action_errors, default=math.inf
        ),
        "maximum_c2_c3_command_error": max(
            all_command_errors, default=math.inf
        ),
        "maximum_zero_residual_nominal_error": max(
            all_zero_errors, default=math.inf
        ),
        "maximum_learned_joint_relation_error": max(
            all_joint_relation_errors, default=math.inf
        ),
        "qp_feasible_count": sum(all_qp),
        "qp_evaluated_count": len(all_qp),
    }
    passed = all(item["passed"] for item in case_reports)
    if not passed:
        raise RuntimeError(
            "posture/pi_L CPU integration failed one or more cases"
        )
    return {
        "report_version": REPORT_VERSION,
        "passed": passed,
        "fixture_path": _portable(fixture_path),
        "fixture_sha256": hash_file(fixture_path),
        "migration_manifest_path": _portable(
            migration_manifest_path
        ),
        "migration_manifest_sha256": hash_file(
            migration_manifest_path
        ),
        "source_checkpoint_path": _portable(source_checkpoint),
        "source_checkpoint_sha256": source.sha256,
        "target_checkpoint_path": _portable(target_checkpoint),
        "target_checkpoint_sha256": target.sha256,
        "physical_model_hash": physical.stable_hash(),
        "shared_parameter_maximum_error": shared_parameter_error,
        "device": str(runtime_device),
        "case_count": len(case_reports),
        "aggregate": aggregate,
        "cases": case_reports,
        "limitations": [
            "model_fk_cpu_only",
            "no_mesh_collision",
            "no_contact_dynamics",
            "no_controller_tracking",
            "no_grasp_or_transport_success_claim",
        ],
    }


def _tensor_inputs(
    *,
    task: TaskSpec,
    morphology: MorphologyGraph,
    physical_model: object,
    runtime: Order9TensorPiLRuntime,
    knot: InteractionKnot,
    device: torch.device,
) -> tuple[
    Order9TensorIsaacState,
    Order9TensorObjectTaskTarget,
    int,
]:
    posture = knot.posture_target
    centroidal = knot.centroidal_target
    if (
        posture is None
        or posture.joint_pos_target is None
        or posture.joint_vel_target is None
        or centroidal is None
        or centroidal.com_pos_world is None
        or centroidal.com_vel_world is None
        or centroidal.body_orientation_world is None
    ):
        raise SchemaValidationError(
            "resolved trajectory knot lacks complete pi_L references"
        )
    q = posture.joint_pos_target
    qdot = posture.joint_vel_target
    centroidal_pose = (
        *centroidal.com_pos_world,
        *centroidal.body_orientation_world,
    )
    base_pose = base_pose_for_centroidal_target(
        morphology,
        physical_model,
        q,
        centroidal.com_pos_world,
        centroidal.body_orientation_world,
    )
    forward = WholeStructureKinematics().forward(
        morphology,
        physical_model,
        q,
        base_pose,
        (),
    )
    module_ids = runtime.builder.module_ids
    local_joint_ids = runtime.builder.local_joint_ids
    command_joint_ids = runtime.decoder.local_joint_ids
    module_pose = torch.tensor(
        [[forward.module_root_poses_world[module_id] for module_id in module_ids]],
        device=device,
        dtype=torch.float32,
    )
    local_q = torch.tensor(
        [
            [
                [
                    float(
                        q.get(f"module_{module_id}:{joint_id}", 0.0)
                    )
                    for joint_id in local_joint_ids
                ]
                for module_id in module_ids
            ]
        ],
        device=device,
        dtype=torch.float32,
    )
    local_qdot = torch.tensor(
        [
            [
                [
                    float(
                        qdot.get(f"module_{module_id}:{joint_id}", 0.0)
                    )
                    for joint_id in local_joint_ids
                ]
                for module_id in module_ids
            ]
        ],
        device=device,
        dtype=torch.float32,
    )
    nominal_q = torch.tensor(
        [
            [
                [
                    float(q[f"module_{module_id}:{joint_id}"])
                    for joint_id in command_joint_ids
                ]
                for module_id in module_ids
            ]
        ],
        device=device,
        dtype=torch.float32,
    )
    nominal_qdot = torch.tensor(
        [
            [
                [
                    float(qdot[f"module_{module_id}:{joint_id}"])
                    for joint_id in command_joint_ids
                ]
                for module_id in module_ids
            ]
        ],
        device=device,
        dtype=torch.float32,
    )
    object_spec = task.scene.objects[0]
    object_pose = (
        knot.object_targets[0].pose_target_world
        if knot.object_targets
        and knot.object_targets[0].pose_target_world is not None
        else object_spec.pose_world
    )
    schedule_index, phase_index = _schedule_and_phase_index(knot)
    target = Order9TensorObjectTaskTarget(
        desired_robot_root_pose_world=torch.tensor(
            [centroidal_pose], device=device, dtype=torch.float32
        ),
        desired_robot_root_twist_world=torch.tensor(
            [[*centroidal.com_vel_world, 0.0, 0.0, 0.0]],
            device=device,
            dtype=torch.float32,
        ),
        nominal_joint_positions_rad=nominal_q,
        nominal_joint_velocities_radps=nominal_qdot,
        desired_object_pose_world=torch.tensor(
            [object_pose], device=device, dtype=torch.float32
        ),
        phase_goal_robot_root_pose_world=torch.tensor(
            [centroidal_pose], device=device, dtype=torch.float32
        ),
        phase_goal_object_pose_world=torch.tensor(
            [object_pose], device=device, dtype=torch.float32
        ),
        phase_progress=torch.tensor(
            [0.5], device=device, dtype=torch.float32
        ),
        contact_schedule_index=torch.tensor(
            [schedule_index], device=device, dtype=torch.long
        ),
    )
    state = Order9TensorIsaacState(
        module_pose_world=module_pose,
        module_twist_world=torch.zeros(
            (1, len(module_ids), 6),
            device=device,
            dtype=torch.float32,
        ),
        local_joint_positions_rad=local_q,
        local_joint_velocities_radps=local_qdot,
        robot_root_pose_world=torch.tensor(
            [base_pose], device=device, dtype=torch.float32
        ),
        robot_root_twist_world=torch.zeros(
            (1, 6), device=device, dtype=torch.float32
        ),
        object_pose_world=torch.tensor(
            [object_pose], device=device, dtype=torch.float32
        ),
        object_twist_world=torch.zeros(
            (1, 6), device=device, dtype=torch.float32
        ),
    )
    return state, target, phase_index


def _schedule_and_phase_index(
    knot: InteractionKnot,
) -> tuple[int, int]:
    states = {
        assignment.schedule_state for assignment in knot.contact_assignments
    }
    if not states:
        return 0, 7
    if len(states) != 1:
        raise SchemaValidationError(
            "integration knot mixes contact schedule states"
        )
    state = next(iter(states))
    return {
        "approach": (ORDER9_CONTACT_SCHEDULE_APPROACH, 0),
        "attach": (ORDER9_CONTACT_SCHEDULE_ATTACH, 1),
        "maintain": (ORDER9_CONTACT_SCHEDULE_MAINTAIN, 3),
        "slide": (ORDER9_CONTACT_SCHEDULE_MAINTAIN, 3),
        "release": (ORDER9_CONTACT_SCHEDULE_RELEASE, 5),
    }[state]


def _resolved_raw_knots(
    raw: ContactWrenchTrajectory,
    resolved: ContactWrenchTrajectory,
) -> tuple[InteractionKnot, ...]:
    output = []
    for raw_knot in raw.knots:
        matches = [
            knot
            for knot in resolved.knots
            if math.isclose(
                float(knot.t_rel_s),
                float(raw_knot.t_rel_s),
                rel_tol=0.0,
                abs_tol=1.0e-10,
            )
        ]
        if len(matches) != 1:
            raise SchemaValidationError(
                "resolved trajectory does not preserve every raw knot time"
            )
        output.append(matches[0])
    return tuple(output)


def _zero_residual_nominal_error(
    runtime: Order9TensorPiLRuntime,
    *,
    task_target: Order9TensorObjectTaskTarget,
    control_model: object,
) -> float:
    module_count = len(runtime.builder.module_ids)
    decoded = runtime.decoder.decode(
        reference_body_pose_world=(
            task_target.desired_robot_root_pose_world
        ),
        reference_body_twist=task_target.desired_robot_root_twist_world,
        normalized_global_action=torch.zeros(
            (1, ORDER9_GLOBAL_ACTION_SIZE),
            device=runtime.device,
            dtype=runtime.dtype,
        ),
        normalized_joint_action=torch.zeros(
            (
                1,
                module_count,
                3 * runtime.config.max_local_joint_slots,
            ),
            device=runtime.device,
            dtype=runtime.dtype,
        ),
        policy_module_ids=torch.tensor(
            [runtime.builder.module_ids],
            device=runtime.device,
            dtype=torch.long,
        ),
        reference_local_joint_positions_rad=(
            task_target.nominal_joint_positions_rad
        ),
        reference_local_joint_velocities_radps=(
            task_target.nominal_joint_velocities_radps
        ),
        reference_local_joint_mask=torch.ones_like(
            task_target.nominal_joint_positions_rad,
            dtype=torch.bool,
        ),
        total_mass_kg=control_model.total_mass_kg,
    )
    return max(
        _tensor_error(
            decoded.desired_body_pose_world,
            task_target.desired_robot_root_pose_world,
        ),
        _tensor_error(
            decoded.desired_body_twist,
            task_target.desired_robot_root_twist_world,
        ),
        float(decoded.residual_wrench_body.abs().max().item()),
        _tensor_error(
            decoded.joint_position_targets_rad,
            task_target.nominal_joint_positions_rad,
        ),
        _tensor_error(
            decoded.joint_velocity_targets_radps,
            task_target.nominal_joint_velocities_radps,
        ),
        float(decoded.joint_torque_bias_nm.abs().max().item()),
    )


def _learned_joint_relation_error(
    runtime: Order9TensorPiLRuntime,
    *,
    step: object,
    task_target: Order9TensorObjectTaskTarget,
) -> float:
    slot_count = len(runtime.decoder.local_joint_ids)
    maximum_slots = runtime.config.max_local_joint_slots
    action = step.policy_step.joint_action
    expected_q = task_target.nominal_joint_positions_rad + (
        action[:, :, :slot_count]
        * float(runtime.config.joint_position_delta_limit_rad)
    )
    expected_qdot = task_target.nominal_joint_velocities_radps + (
        action[
            :,
            :,
            maximum_slots : maximum_slots + slot_count,
        ]
        * float(runtime.config.joint_velocity_limit_rad_s)
    )
    return max(
        _tensor_error(
            step.policy_command.joint_position_targets_rad, expected_q
        ),
        _tensor_error(
            step.policy_command.joint_velocity_targets_radps,
            expected_qdot,
        ),
    )


def _require_finite_pi_l_step(step: object) -> None:
    tensors = (
        step.policy_step.action,
        step.policy_step.joint_action,
        step.policy_step.value,
        step.policy_command.desired_body_pose_world,
        step.policy_command.desired_body_twist,
        step.policy_command.residual_wrench_body,
        step.policy_command.joint_position_targets_rad,
        step.policy_command.joint_velocity_targets_radps,
        step.policy_command.joint_torque_bias_nm,
        step.controller_result.desired_wrench_body,
        step.controller_result.allocation.rotor_thrusts_n,
        step.controller_result.allocation.vectoring_joint_targets_rad,
    )
    if any(not bool(torch.isfinite(value).all().item()) for value in tensors):
        raise RuntimeError("posture/pi_L integration produced non-finite output")
    if (
        step.active_knot_features is None
        or step.active_assignment_features is None
    ):
        raise RuntimeError(
            "C3 pi_L integration omitted active-knot actor features"
        )


def _require_raw_pi_h_without_joint_targets(
    trajectory: ContactWrenchTrajectory,
) -> None:
    for knot in trajectory.knots:
        posture = knot.posture_target
        if posture is not None and (
            posture.joint_pos_target is not None
            or posture.joint_vel_target is not None
        ):
            raise SchemaValidationError(
                "integration fixture raw pi_H contains joint targets"
            )


def _maximum_shared_parameter_error(
    source: Order9PhaseConditionedActorCritic,
    target: Order9ActiveKnotPhaseConditionedActorCritic,
) -> float:
    source_state = source.state_dict()
    target_state = target.state_dict()
    if not set(source_state).issubset(target_state):
        raise SchemaValidationError(
            "C3 pi_L initializer omits C2 parameters"
        )
    return max(
        (
            float(
                (
                    target_state[name].detach().cpu()
                    - value.detach().cpu()
                )
                .abs()
                .max()
                .item()
            )
            for name, value in source_state.items()
        ),
        default=0.0,
    )


def _tensor_error(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        return math.inf
    return float((left - right).abs().max().item())


def _quantized_stable_hash(value: object) -> str:
    return stable_hash(_quantize_floats(value))


def _quantize_floats(value: object) -> object:
    if isinstance(value, float):
        quantized = round(value, TRAJECTORY_HASH_DECIMAL_PLACES)
        return 0.0 if quantized == 0.0 else quantized
    if isinstance(value, dict):
        return {
            key: _quantize_floats(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_quantize_floats(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_quantize_floats(item) for item in value)
    return value


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def _portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path.resolve())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
