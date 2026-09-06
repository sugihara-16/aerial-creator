from __future__ import annotations

"""Fail-closed preflight for converting formal R1 traces to pi_H records."""

from bisect import bisect_right
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from amsrr.feasibility.contact_wrench_trajectory import (
    ContactWrenchTrajectoryCheckerConfig,
    ContactWrenchTrajectoryFeasibilityChecker,
)
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.policies.order9_high_level_policy import (
    Order9AutoregressiveHighLevelPolicy,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.datasets import (
    DatasetSplit,
    InteractionTrajectoryRecord,
    PolicyBehaviorTrace,
)
from amsrr.schemas.policies import (
    CONTACT_WRENCH_CONTRACT_CONTACT_FRAME,
    ContactWrenchTrajectory,
    ControllerStatus,
    InteractionKnot,
)
from amsrr.schemas.runtime import (
    ModuleRuntimeState,
    ObjectRuntimeState,
    RuntimeObservation,
    TaskProgressState,
)
from amsrr.schemas.task_spec import TaskSpec
from amsrr.simulation.order9_object_task_runtime import (
    ORDER9_OBJECT_TASK_ACTOR_PHASE_COUNT,
    ORDER9_OBJECT_TASK_ACTOR_PHASE_INDEX_BY_RUNTIME,
    ORDER9_OBJECT_TASK_ACTOR_PHASE_LABELS,
    ORDER9_OBJECT_TASK_PHASES,
)
from amsrr.training.order9_c3_nominal_trajectory import (
    Order9C3AcceptedNominalBundle,
    load_order9_c3_accepted_nominal_bundle,
)
from amsrr.training.order9_teacher import (
    compile_high_level_context,
    teacher_interaction_record,
)
from amsrr.training.order9_pi_h_learning import (
    compute_order9_pi_h_behavior_cloning_loss,
)
from amsrr.utils.hashing import hash_file

ORDER9_R1_PI_H_PREFLIGHT_VERSION = "order9_r1_pi_h_dataset_preflight_v20"
ORDER9_R1_PI_H_CONVERSION_VERSION = "order9_r1_pi_h_dataset_conversion_v20"
ORDER9_R1_FORMAL_SUMMARY_VERSION = "order9_r1_teacher_collection_summary_v19"
ORDER9_R1_FORMAL_RECORD_VERSION = "order9_r1_teacher_collection_episode_v19"
ORDER9_R1_FORMAL_EXECUTION_REVISION = "identity_v2"
ORDER9_R1_DECISION_INTERVAL_S = 0.5
ORDER9_R1_PI_H_HORIZON_S = 2.0
ORDER9_R1_PI_H_KNOT_DT_S = 0.25
ORDER9_R1_DECISION_RETURN_GAMMA = 0.99

_PHASE_NAMES = tuple(phase.value for phase in ORDER9_OBJECT_TASK_PHASES)
_ACTOR_TO_RUNTIME_PHASE = {
    actor_label: _PHASE_NAMES[runtime_index]
    for runtime_index, actor_index in enumerate(
        ORDER9_OBJECT_TASK_ACTOR_PHASE_INDEX_BY_RUNTIME
    )
    for actor_label in (ORDER9_OBJECT_TASK_ACTOR_PHASE_LABELS[actor_index],)
}
_REQUIRED_RAW_TENSORS = {
    "valid",
    "time_s",
    "phase_index",
    "phase_progress",
    "step_index",
    "module_pose_world",
    "module_twist_world",
    "local_joint_positions_rad",
    "local_joint_velocities_radps",
    "object_pose_world",
    "object_twist_world",
    "actor_controller_qp_feasible",
    "actor_controller_status_one_hot",
    "actor_allocation_residual_norm",
    "actor_task_success",
    "desired_body_pose_world",
    "desired_body_twist_reference",
    "desired_object_pose_world",
    "global_action",
    "joint_action",
    "contact_space_residual_action",
    "applied_global_action",
    "selected_assignment_mask",
    "prohibited_collision",
    "reward",
    "terminal",
    "truncated",
}


def build_order9_r1_pi_h_teacher_window(
    phase_trajectories: Mapping[str, ContactWrenchTrajectory],
    *,
    actor_phase_label: str,
    phase_progress: float,
    horizon_s: float = ORDER9_R1_PI_H_HORIZON_S,
    knot_dt_s: float = ORDER9_R1_PI_H_KNOT_DT_S,
) -> ContactWrenchTrajectory:
    """Build the joint-free rolling pi_H target at one recorded decision."""

    if tuple(phase_trajectories) != _PHASE_NAMES:
        raise SchemaValidationError(
            "R1 pi_H phase trajectories are incomplete or unordered"
        )
    runtime_phase = _ACTOR_TO_RUNTIME_PHASE.get(actor_phase_label)
    if runtime_phase is None:
        raise SchemaValidationError(
            f"R1 pi_H actor phase is not executable: {actor_phase_label!r}"
        )
    if (
        not math.isfinite(float(phase_progress))
        or not 0.0 <= float(phase_progress) <= 1.0
        or not math.isfinite(float(horizon_s))
        or not math.isfinite(float(knot_dt_s))
        or horizon_s <= 0.0
        or knot_dt_s <= 0.0
        or not math.isclose(
            horizon_s / knot_dt_s,
            round(horizon_s / knot_dt_s),
            abs_tol=1.0e-9,
        )
    ):
        raise SchemaValidationError("R1 pi_H rolling-window parameters are invalid")
    for name, trajectory in phase_trajectories.items():
        trajectory.validate()
        if trajectory.contract_version != CONTACT_WRENCH_CONTRACT_CONTACT_FRAME:
            raise SchemaValidationError(
                f"R1 pi_H phase {name!r} does not use the contact-frame contract"
            )

    phase_index = _PHASE_NAMES.index(runtime_phase)
    phase_time_s = float(phase_progress) * phase_trajectories[runtime_phase].horizon_s
    knots = []
    knot_count = int(round(horizon_s / knot_dt_s)) + 1
    for knot_index in range(knot_count):
        source_phase_index, source_time_s = _advance_phase_time(
            phase_trajectories,
            phase_index=phase_index,
            phase_time_s=phase_time_s,
            delta_s=float(knot_index) * knot_dt_s,
        )
        source_phase = _PHASE_NAMES[source_phase_index]
        knot = _sample_joint_free_knot(phase_trajectories[source_phase], source_time_s)
        knot.t_rel_s = float(knot_index) * knot_dt_s
        knots.append(knot)
    result = ContactWrenchTrajectory(
        horizon_s=float(horizon_s),
        dt_s=float(knot_dt_s),
        knots=knots,
        derived_mode_label=ORDER9_R1_PI_H_CONVERSION_VERSION,
        contract_version=CONTACT_WRENCH_CONTRACT_CONTACT_FRAME,
    )
    result.validate()
    if any(
        knot.posture_target is not None
        and (
            knot.posture_target.joint_pos_target is not None
            or knot.posture_target.joint_vel_target is not None
        )
        for knot in result.knots
    ):
        raise AssertionError("R1 pi_H target retained deterministic IK joint targets")
    return result


def build_order9_r1_pi_h_teacher_window_from_trace(
    phase_trajectories: Mapping[str, ContactWrenchTrajectory],
    *,
    tensors: Mapping[str, torch.Tensor],
    raw_index: int,
    horizon_s: float = ORDER9_R1_PI_H_HORIZON_S,
    knot_dt_s: float = ORDER9_R1_PI_H_KNOT_DT_S,
) -> ContactWrenchTrajectory:
    """Build the rolling target actually presented to QPID during Isaac execution."""

    times = tensors["time_s"][:, 0]
    if not 0 <= raw_index < int(times.numel()):
        raise SchemaValidationError("R1 pi_H trace decision index is invalid")
    knot_count = int(round(horizon_s / knot_dt_s)) + 1
    trace_times = [float(value) for value in times.tolist()]
    start_time = trace_times[raw_index]
    knots: list[InteractionKnot] = []
    for knot_index in range(knot_count):
        target_time = start_time + float(knot_index) * knot_dt_s
        source_index = max(
            raw_index,
            min(len(trace_times) - 1, bisect_right(trace_times, target_time) - 1),
        )
        actor_index = int(tensors["phase_index"][source_index, 0])
        actor_label = ORDER9_OBJECT_TASK_ACTOR_PHASE_LABELS[actor_index]
        runtime_phase = _ACTOR_TO_RUNTIME_PHASE.get(actor_label)
        if runtime_phase is None:
            raise SchemaValidationError(
                f"R1 pi_H future trace phase is not executable: {actor_label!r}"
            )
        phase_progress = float(tensors["phase_progress"][source_index, 0])
        source_time = phase_progress * phase_trajectories[runtime_phase].horizon_s
        payload = _sample_joint_free_knot(
            phase_trajectories[runtime_phase], source_time
        ).to_dict()
        desired_pose = tensors["desired_body_pose_world"][source_index, 0].tolist()
        desired_twist = tensors["desired_body_twist_reference"][
            source_index, 0
        ].tolist()
        centroidal = dict(payload.get("centroidal_target") or {})
        centroidal["com_pos_world"] = [float(value) for value in desired_pose[:3]]
        centroidal["body_orientation_world"] = [
            float(value) for value in desired_pose[3:7]
        ]
        centroidal["com_vel_world"] = [float(value) for value in desired_twist[:3]]
        payload["centroidal_target"] = centroidal
        desired_object = [
            float(value)
            for value in tensors["desired_object_pose_world"][source_index, 0].tolist()
        ]
        object_targets = payload.get("object_targets")
        if not isinstance(object_targets, list) or len(object_targets) != 1:
            raise SchemaValidationError(
                "R1 pi_H source knot must contain one object target"
            )
        object_targets[0]["pose_target_world"] = desired_object
        payload["t_rel_s"] = float(knot_index) * knot_dt_s
        knots.append(InteractionKnot.from_dict(payload))
    result = ContactWrenchTrajectory(
        horizon_s=float(horizon_s),
        dt_s=float(knot_dt_s),
        knots=knots,
        derived_mode_label=ORDER9_R1_PI_H_CONVERSION_VERSION,
        contract_version=CONTACT_WRENCH_CONTRACT_CONTACT_FRAME,
    )
    result.validate()
    return result


def select_order9_r1_pi_h_decision_rows(
    *,
    valid: torch.Tensor,
    time_s: torch.Tensor,
    phase_index: torch.Tensor,
    interval_s: float = ORDER9_R1_DECISION_INTERVAL_S,
) -> tuple[int, ...]:
    """Select fixed-rate decisions while retaining every phase boundary and tail."""

    if valid.ndim != time_s.ndim or valid.ndim != phase_index.ndim or valid.ndim != 1:
        raise SchemaValidationError("R1 pi_H decision tensors must be one-dimensional")
    if not math.isfinite(float(interval_s)) or interval_s <= 0.0:
        raise SchemaValidationError("R1 pi_H decision interval must be positive")
    indices = torch.nonzero(valid, as_tuple=False).flatten().tolist()
    if not indices:
        raise SchemaValidationError("R1 pi_H raw rollout has no valid rows")
    selected = {int(indices[0]), int(indices[-1])}
    last_selected_time = float(time_s[indices[0]])
    previous_phase = int(phase_index[indices[0]])
    for raw_index in indices[1:]:
        current_time = float(time_s[raw_index])
        current_phase = int(phase_index[raw_index])
        if current_phase != previous_phase:
            selected.add(int(raw_index))
            last_selected_time = current_time
        elif current_time - last_selected_time >= interval_s - 1.0e-6:
            selected.add(int(raw_index))
            last_selected_time = current_time
        previous_phase = current_phase
    return tuple(sorted(selected))


def preflight_order9_r1_pi_h_dataset(
    summary_path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Dry-materialize every planned pi_H row without writing a dataset."""

    repository = Path(repository_root).resolve()
    summary_source = Path(summary_path).resolve()
    summary = _read_json(summary_source)
    bindings = summary.get("collection_records")
    if (
        summary.get("summary_version") != ORDER9_R1_FORMAL_SUMMARY_VERSION
        or summary.get("execution_revision") != ORDER9_R1_FORMAL_EXECUTION_REVISION
        or summary.get("status") != "complete"
        or summary.get("training_authorized") is not False
        or int(summary.get("planned_episode_count", -1)) != 140
        or int(summary.get("accepted_episode_count", -1)) != 140
        or not isinstance(bindings, list)
        or len(bindings) != 140
    ):
        raise SchemaValidationError("R1 formal collection summary contract differs")

    checker = ContactWrenchTrajectoryFeasibilityChecker(
        config=ContactWrenchTrajectoryCheckerConfig.warmup_proxy()
    )
    episode_ids: set[str] = set()
    task_ids: set[str] = set()
    pose_hashes: set[str] = set()
    raw_hashes: set[str] = set()
    record_hashes: set[str] = set()
    episode_counts: Counter[str] = Counter()
    record_counts: Counter[str] = Counter()
    module_episode_counts: Counter[int] = Counter()
    module_record_counts: Counter[int] = Counter()
    phase_record_counts: Counter[str] = Counter()
    representative_records: dict[tuple[int, str, str], InteractionTrajectoryRecord] = {}
    maximum_reference_error_m = 0.0
    maximum_joint_action = 0.0
    maximum_global_action = 0.0
    minimum_decisions_per_episode = math.inf
    maximum_decisions_per_episode = 0

    for binding_index, binding in enumerate(bindings):
        record_path = _validated_binding(
            binding,
            repository=repository,
            label=f"collection_record_{binding_index}",
        )
        accepted = _read_json(record_path)
        _validate_accepted_record(accepted)
        episode_id = str(accepted["episode_id"])
        task_id = str(accepted["task_id"])
        split = DatasetSplit(str(accepted["dataset_split"]))
        module_count = int(accepted["module_count"])
        if episode_id in episode_ids or task_id in task_ids:
            raise SchemaValidationError(
                "R1 pi_H episode or task identity is duplicated"
            )
        episode_ids.add(episode_id)
        task_ids.add(task_id)
        pose_hash = str(accepted["pose_condition_hash"])
        if pose_hash in pose_hashes:
            raise SchemaValidationError("R1 pi_H pose condition is duplicated")
        pose_hashes.add(pose_hash)
        case_path = _validated_binding(
            accepted["case_manifest"], repository=repository, label="case_manifest"
        )
        raw_path = _validated_binding(
            accepted["raw_rollout"], repository=repository, label="raw_rollout"
        )
        raw_hash = hash_file(raw_path)
        if raw_hash in raw_hashes:
            raise SchemaValidationError("R1 pi_H raw rollout is duplicated")
        raw_hashes.add(raw_hash)
        case = _read_json(case_path)
        bundle, task = _load_case_context(
            case,
            episode_id=episode_id,
            task_id=task_id,
            split=split,
            module_count=module_count,
            repository=repository,
        )
        payload = torch.load(raw_path, map_location="cpu", weights_only=False)
        metadata, tensors = _validate_raw_rollout(
            payload,
            task=task,
            bundle=bundle,
            accepted=accepted,
        )
        decisions = select_order9_r1_pi_h_decision_rows(
            valid=tensors["valid"][:, 0],
            time_s=tensors["time_s"][:, 0],
            phase_index=tensors["phase_index"][:, 0],
        )
        minimum_decisions_per_episode = min(
            minimum_decisions_per_episode, len(decisions)
        )
        maximum_decisions_per_episode = max(
            maximum_decisions_per_episode, len(decisions)
        )
        returns = _discounted_returns(tensors["reward"][:, 0], tensors["valid"][:, 0])
        base_context = compile_high_level_context(
            task,
            bundle.morphology,
            bundle.contact_candidate_set,
            runtime_observation=None,
        )
        for decision_index, raw_index in enumerate(decisions):
            observation = _runtime_observation(
                metadata=metadata,
                tensors=tensors,
                raw_index=raw_index,
                morphology=bundle.morphology,
            )
            phase_label = observation.task_progress.phase_label
            if phase_label is None:
                raise SchemaValidationError("R1 pi_H actor phase is missing")
            trajectory = build_order9_r1_pi_h_teacher_window_from_trace(
                bundle.phase_trajectories,
                tensors=tensors,
                raw_index=raw_index,
            )
            context = HighLevelPolicyContext(
                irg=base_context.irg,
                interaction_envelope=base_context.interaction_envelope,
                morphology_graph=bundle.morphology,
                contact_candidate_set=bundle.contact_candidate_set,
                runtime_observation=observation,
            )
            record = teacher_interaction_record(
                record_id=(f"{episode_id}:pi_h:{decision_index:06d}"),
                episode_id=episode_id,
                split=split,
                decision_index=decision_index,
                context=context,
                trajectory=trajectory,
                checker=checker,
                decision_return=returns[raw_index],
                teacher_version=ORDER9_R1_PI_H_CONVERSION_VERSION,
            )
            record.terminal = decision_index == len(decisions) - 1
            record.behavior_trace = PolicyBehaviorTrace(
                policy_family="pi_h",
                policy_version=ORDER9_R1_PI_H_CONVERSION_VERSION,
                action_semantics="full_contact_wrench_trajectory_v2_joint_free",
                action_payload={"trajectory": trajectory.to_dict()},
                stochastic=False,
            )
            record.validate()
            digest = record.stable_hash()
            if digest in record_hashes:
                raise SchemaValidationError("R1 pi_H materialized record is duplicated")
            record_hashes.add(digest)
            phase_record_counts[phase_label] += 1
            record_counts[split.value] += 1
            module_record_counts[module_count] += 1
            representative_records.setdefault(
                (module_count, split.value, phase_label), record
            )

        maximum_reference_error_m = max(
            maximum_reference_error_m,
            _maximum_reference_position_error(
                tensors=tensors,
                metadata=metadata,
                bundle=bundle,
            ),
        )
        maximum_joint_action = max(
            maximum_joint_action,
            _maximum_absolute(tensors["joint_action"], tensors["valid"]),
        )
        maximum_global_action = max(
            maximum_global_action,
            _maximum_absolute(tensors["global_action"], tensors["valid"]),
            _maximum_absolute(tensors["applied_global_action"], tensors["valid"]),
            _maximum_absolute(
                tensors["contact_space_residual_action"], tensors["valid"]
            ),
        )
        episode_counts[split.value] += 1
        module_episode_counts[module_count] += 1

    expected_episode_splits = {"train": 98, "validation": 21, "held_out": 21}
    if dict(episode_counts) != expected_episode_splits:
        raise SchemaValidationError("R1 pi_H episode split counts differ")
    if any(module_episode_counts[module_count] != 20 for module_count in range(2, 9)):
        raise SchemaValidationError("R1 pi_H module episode counts differ")
    if set(phase_record_counts) != set(_ACTOR_TO_RUNTIME_PHASE):
        raise SchemaValidationError("R1 pi_H converted records do not cover all phases")
    if len(representative_records) != 168:
        raise SchemaValidationError("R1 pi_H module/split/phase coverage differs")
    representative_forward = _preflight_representative_pi_h_forward(
        tuple(representative_records.values())
    )

    return {
        "preflight_version": ORDER9_R1_PI_H_PREFLIGHT_VERSION,
        "status": "ready_for_pi_h_dataset_conversion",
        "training_authorized": False,
        "source_summary": {
            "path": _portable(summary_source, repository),
            "sha256": hash_file(summary_source),
        },
        "conversion_contract": {
            "conversion_version": ORDER9_R1_PI_H_CONVERSION_VERSION,
            "decision_interval_s": ORDER9_R1_DECISION_INTERVAL_S,
            "trajectory_horizon_s": ORDER9_R1_PI_H_HORIZON_S,
            "trajectory_knot_dt_s": ORDER9_R1_PI_H_KNOT_DT_S,
            "trajectory_knot_count": int(
                round(ORDER9_R1_PI_H_HORIZON_S / ORDER9_R1_PI_H_KNOT_DT_S)
            )
            + 1,
            "decision_return_gamma_per_control_step": (ORDER9_R1_DECISION_RETURN_GAMMA),
            "joint_targets_in_pi_h_teacher_output": False,
            "deterministic_ik_boundary_preserved": True,
            "actor_raw_contact_input": False,
            "pi_l_actor_command_applied": False,
            "qpid_qp_applied": True,
            "local_servo_applied": True,
            "real_isaac_success_required": True,
            "failed_diagnostic_traces_included": False,
        },
        "episode_count": len(episode_ids),
        "episode_split_counts": dict(sorted(episode_counts.items())),
        "module_episode_counts": {
            str(key): module_episode_counts[key] for key in range(2, 9)
        },
        "planned_interaction_record_count": len(record_hashes),
        "record_split_counts": dict(sorted(record_counts.items())),
        "module_record_counts": {
            str(key): module_record_counts[key] for key in range(2, 9)
        },
        "actor_phase_record_counts": dict(sorted(phase_record_counts.items())),
        "minimum_decisions_per_episode": int(minimum_decisions_per_episode),
        "maximum_decisions_per_episode": int(maximum_decisions_per_episode),
        "unique_episode_count": len(episode_ids),
        "unique_task_count": len(task_ids),
        "unique_pose_condition_count": len(pose_hashes),
        "unique_raw_rollout_count": len(raw_hashes),
        "unique_materialized_record_count": len(record_hashes),
        "module_split_phase_representative_count": len(representative_records),
        "representative_pi_h_forward": representative_forward,
        "maximum_nominal_reference_position_error_m": maximum_reference_error_m,
        "maximum_recorded_joint_action_abs": maximum_joint_action,
        "maximum_recorded_pi_l_global_action_abs": maximum_global_action,
        "all_schema_valid": True,
        "all_warmup_c_h_feasible": True,
        "all_real_isaac_accepted": True,
    }


def _advance_phase_time(
    phase_trajectories: Mapping[str, ContactWrenchTrajectory],
    *,
    phase_index: int,
    phase_time_s: float,
    delta_s: float,
) -> tuple[int, float]:
    index = int(phase_index)
    time_s = float(phase_time_s) + float(delta_s)
    while index < len(_PHASE_NAMES) - 1:
        duration = float(phase_trajectories[_PHASE_NAMES[index]].horizon_s)
        if time_s <= duration + 1.0e-9:
            break
        time_s -= duration
        index += 1
    duration = float(phase_trajectories[_PHASE_NAMES[index]].horizon_s)
    return index, min(max(time_s, 0.0), duration)


def _sample_joint_free_knot(
    trajectory: ContactWrenchTrajectory, time_s: float
) -> InteractionKnot:
    times = [float(knot.t_rel_s) for knot in trajectory.knots]
    index = max(0, min(len(times) - 1, bisect_right(times, time_s + 1.0e-9) - 1))
    payload = trajectory.knots[index].to_dict()
    posture = payload.get("posture_target")
    if isinstance(posture, dict):
        posture["joint_pos_target"] = None
        posture["joint_vel_target"] = None
    return InteractionKnot.from_dict(payload)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SchemaValidationError(f"JSON object required: {path}")
    return value


def _validated_binding(binding: object, *, repository: Path, label: str) -> Path:
    if not isinstance(binding, Mapping):
        raise SchemaValidationError(f"R1 pi_H binding is missing: {label}")
    relative = binding.get("path")
    expected = binding.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str):
        raise SchemaValidationError(f"R1 pi_H binding is invalid: {label}")
    path = (repository / relative).resolve()
    if repository != path and repository not in path.parents:
        raise SchemaValidationError(f"R1 pi_H binding escapes repository: {label}")
    if not path.is_file() or hash_file(path) != expected:
        raise SchemaValidationError(f"R1 pi_H bound bytes changed: {label}")
    return path


def _validate_accepted_record(record: Mapping[str, Any]) -> None:
    if (
        record.get("record_version") != ORDER9_R1_FORMAL_RECORD_VERSION
        or record.get("status") != "accepted"
        or record.get("training_eligible") is not True
        or record.get("task_success") is not True
        or record.get("safety_failure") is not False
        or int(record.get("fallback_decision_count", -1)) != 0
        or record.get("pi_l_actor_command_applied") is not False
        or record.get("qpid_qp_applied") is not True
        or record.get("local_servo_applied") is not True
        or record.get("execution_split") != "validation"
        or record.get("execution_split_alias_contract")
        != "protected_c3_formal_runner_validation_alias_v1"
        or record.get("dataset_split") not in {split.value for split in DatasetSplit}
        or not 2 <= int(record.get("module_count", -1)) <= 8
    ):
        raise SchemaValidationError("R1 accepted teacher record contract differs")


def _load_case_context(
    case: Mapping[str, Any],
    *,
    episode_id: str,
    task_id: str,
    split: DatasetSplit,
    module_count: int,
    repository: Path,
) -> tuple[Order9C3AcceptedNominalBundle, TaskSpec]:
    # Protected C3 execution exposes held-out cases through its validation
    # runner alias.  The task and accepted record retain the true dataset split.
    expected_case_split = (
        "validation" if split == DatasetSplit.HELD_OUT else split.value
    )
    if (
        case.get("candidate_id") != episode_id
        or case.get("split") != expected_case_split
        or int(case.get("module_count", -1)) != module_count
        # The case is an input candidate and must remain ineligible by itself.
        # Only the hash-bound, accepted formal collection record promotes the
        # resulting execution to training eligibility.
        or case.get("training_eligible") is not False
    ):
        raise SchemaValidationError("R1 pi_H case identity differs")
    task_path = _validated_binding(
        case.get("task_spec"), repository=repository, label="task_spec"
    )
    task = TaskSpec.from_json(task_path.read_text(encoding="utf-8"))
    task.validate()
    if task.task_id != task_id or task.metadata.get("dataset_split") != split.value:
        raise SchemaValidationError("R1 pi_H task identity or split differs")
    set_path = _validated_binding(
        case.get("nominal_set"), repository=repository, label="nominal_set"
    )
    artifact = case.get("nominal_artifact")
    if not isinstance(artifact, Mapping):
        raise SchemaValidationError("R1 pi_H nominal artifact binding is missing")
    bundle = load_order9_c3_accepted_nominal_bundle(
        set_path,
        bucket_id=episode_id,
        repository_root=repository,
        expected_set_sha256=str(case["nominal_set"]["sha256"]),
        expected_artifact_sha256=str(artifact.get("sha256")),
        expected_structural_hash=str(case.get("structural_hash")),
        expected_task_spec_sha256=hash_file(task_path),
    )
    if (
        len(bundle.morphology.modules) != module_count
        or bundle.contact_candidate_set.task_id != task_id
    ):
        raise SchemaValidationError("R1 pi_H nominal context identity differs")
    return bundle, task


def _validate_raw_rollout(
    payload: object,
    *,
    task: TaskSpec,
    bundle: Order9C3AcceptedNominalBundle,
    accepted: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if not isinstance(payload, dict) or set(payload) != {
        "artifact_version",
        "metadata",
        "tensors",
    }:
        raise SchemaValidationError("R1 pi_H raw rollout payload differs")
    metadata = dict(payload["metadata"])
    tensors = dict(payload["tensors"])
    if not _REQUIRED_RAW_TENSORS.issubset(tensors):
        raise SchemaValidationError("R1 pi_H raw rollout tensors are incomplete")
    valid = tensors["valid"]
    if valid.ndim != 2 or valid.shape[1] != 1 or valid.dtype != torch.bool:
        raise SchemaValidationError("R1 pi_H raw rollout must contain one environment")
    steps = valid.shape[0]
    for name in _REQUIRED_RAW_TENSORS:
        value = tensors[name]
        if not isinstance(value, torch.Tensor) or value.shape[:2] != (steps, 1):
            raise SchemaValidationError(f"R1 pi_H raw tensor shape differs: {name}")
    indices = torch.nonzero(valid[:, 0], as_tuple=False).flatten()
    if indices.numel() != steps:
        raise SchemaValidationError("R1 pi_H raw rollout contains invalid padding rows")
    if tensors["step_index"][:, 0].tolist() != list(range(steps)):
        raise SchemaValidationError("R1 pi_H raw step indices are not contiguous")
    times = tensors["time_s"][:, 0]
    if float(times[0]) != 0.0 or not bool((times[1:] > times[:-1]).all()):
        raise SchemaValidationError("R1 pi_H raw times are not strictly increasing")
    expected_actor_phases = tuple(ORDER9_OBJECT_TASK_ACTOR_PHASE_INDEX_BY_RUNTIME)
    observed_actor_phases = tuple(
        dict.fromkeys(int(value) for value in tensors["phase_index"][:, 0].tolist())
    )
    if observed_actor_phases != expected_actor_phases:
        raise SchemaValidationError(
            "R1 pi_H raw rollout did not execute all phases in order"
        )
    if (
        int(tensors["terminal"][:, 0].sum()) != 1
        or not bool(tensors["terminal"][-1, 0])
        or bool(tensors["truncated"].any())
        or not bool(tensors["actor_task_success"][-1, 0])
        or bool(tensors["prohibited_collision"].any())
        or not bool(tensors["actor_controller_qp_feasible"].all())
    ):
        raise SchemaValidationError("R1 pi_H raw success or safety boundary differs")
    task_specs = metadata.get("task_specs")
    if not isinstance(task_specs, list) or len(task_specs) != 1:
        raise SchemaValidationError("R1 pi_H raw task metadata differs")
    raw_task = TaskSpec.from_dict(task_specs[0])
    expected_raw_task_id = f"{task.task_id}:env:0000"
    if (
        raw_task.task_id != expected_raw_task_id
        or raw_task.metadata.get("dataset_split") != accepted["dataset_split"]
        or metadata.get("environment_splits") != ["validation"]
        or metadata.get("pi_l_actor_command_applied") is not False
        or metadata.get("diagnostic_nominal_qpid_only") is not True
        or metadata.get("raw_contact_actor_input") is not False
        or metadata.get("formal_teacher_collection_authorized") is not True
        or metadata.get("training_eligible") is not False
        or metadata.get("selected_anchor_ids") is None
        or metadata.get("morphology_graph") != bundle.morphology.to_dict()
        or metadata.get("module_ids")
        != sorted(module.module_id for module in bundle.morphology.modules)
    ):
        raise SchemaValidationError("R1 pi_H raw metadata contract differs")
    selected_anchor_ids = metadata["selected_anchor_ids"]
    if len(selected_anchor_ids) != len(set(selected_anchor_ids)) or tensors[
        "selected_assignment_mask"
    ].shape[2] != len(selected_anchor_ids):
        raise SchemaValidationError("R1 pi_H selected-anchor layout differs")
    return metadata, tensors


def _runtime_observation(
    *,
    metadata: Mapping[str, Any],
    tensors: Mapping[str, torch.Tensor],
    raw_index: int,
    morphology,
) -> RuntimeObservation:
    module_ids = tuple(int(value) for value in metadata["module_ids"])
    joint_ids = tuple(str(value) for value in metadata["local_joint_ids"])
    one_hot = tensors["actor_controller_status_one_hot"][raw_index, 0]
    labels = ("ok", "warning", "infeasible", "fault")
    status = ControllerStatus(
        status=labels[int(torch.argmax(one_hot).item())],  # type: ignore[arg-type]
        qp_feasible=bool(tensors["actor_controller_qp_feasible"][raw_index, 0]),
        active_mode="rigid_body_qp",
        metrics={
            "allocation_residual_norm": float(
                tensors["actor_allocation_residual_norm"][raw_index, 0]
            )
        },
    )
    phase_index = int(tensors["phase_index"][raw_index, 0])
    observation = RuntimeObservation(
        time_s=float(tensors["time_s"][raw_index, 0]),
        morphology_graph=type(morphology).from_dict(morphology.to_dict()),
        module_states=[
            ModuleRuntimeState(
                module_id=module_id,
                pose_world=tuple(
                    float(value)
                    for value in tensors["module_pose_world"][
                        raw_index, 0, index
                    ].tolist()
                ),
                twist_world=[
                    float(value)
                    for value in tensors["module_twist_world"][
                        raw_index, 0, index
                    ].tolist()
                ],
                joint_positions={
                    joint_id: float(
                        tensors["local_joint_positions_rad"][
                            raw_index, 0, index, joint_index
                        ]
                    )
                    for joint_index, joint_id in enumerate(joint_ids)
                },
                joint_velocities={
                    joint_id: float(
                        tensors["local_joint_velocities_radps"][
                            raw_index, 0, index, joint_index
                        ]
                    )
                    for joint_index, joint_id in enumerate(joint_ids)
                },
            )
            for index, module_id in enumerate(module_ids)
        ],
        object_states=[
            ObjectRuntimeState(
                object_id=str(metadata["object_id"]),
                pose_world=tuple(
                    float(value)
                    for value in tensors["object_pose_world"][raw_index, 0].tolist()
                ),
                twist_world=[
                    float(value)
                    for value in tensors["object_twist_world"][raw_index, 0].tolist()
                ],
            )
        ],
        contact_states=[],
        controller_status=status,
        task_progress=TaskProgressState(
            phase_label=ORDER9_OBJECT_TASK_ACTOR_PHASE_LABELS[phase_index],
            progress_ratio=float(tensors["phase_progress"][raw_index, 0]),
            success=bool(tensors["actor_task_success"][raw_index, 0]),
            metrics={
                "phase_index": float(phase_index),
                "phase_count": float(ORDER9_OBJECT_TASK_ACTOR_PHASE_COUNT),
            },
        ),
    )
    observation.validate()
    return observation


def _discounted_returns(
    rewards: torch.Tensor,
    valid: torch.Tensor,
    gamma: float = ORDER9_R1_DECISION_RETURN_GAMMA,
) -> dict[int, float]:
    if rewards.ndim != 1 or valid.shape != rewards.shape:
        raise SchemaValidationError("R1 pi_H reward tensors differ")
    result: dict[int, float] = {}
    running = 0.0
    for index in reversed(torch.nonzero(valid, as_tuple=False).flatten().tolist()):
        reward = float(rewards[index])
        if not math.isfinite(reward):
            raise SchemaValidationError("R1 pi_H reward is not finite")
        running = reward + gamma * running
        result[int(index)] = running
    return result


def _maximum_reference_position_error(
    *,
    tensors: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
    bundle: Order9C3AcceptedNominalBundle,
) -> float:
    """Check the stored nominal target plus declared runtime height transform."""

    maximum = 0.0
    actor_to_runtime_index = {
        actor_index: runtime_index
        for runtime_index, actor_index in enumerate(
            ORDER9_OBJECT_TASK_ACTOR_PHASE_INDEX_BY_RUNTIME
        )
    }
    valid_indices = torch.nonzero(tensors["valid"][:, 0], as_tuple=False).flatten()
    sample_indices = valid_indices[
        :: max(1, int(round(0.5 / float(metadata["control_dt_s"]))))
    ]
    for raw_index_tensor in sample_indices:
        raw_index = int(raw_index_tensor)
        actor_index = int(tensors["phase_index"][raw_index, 0])
        runtime_index = actor_to_runtime_index[actor_index]
        trajectory = bundle.phase_trajectories[_PHASE_NAMES[runtime_index]]
        source_time = float(tensors["phase_progress"][raw_index, 0]) * float(
            trajectory.horizon_s
        )
        source_position = _interpolated_position(trajectory, source_time)
        expected = torch.tensor(source_position, dtype=torch.float32)
        scale = metadata.get("r1_diagnostic_maintain_vertical_scale")
        if scale is not None and _PHASE_NAMES[runtime_index] in {
            "lift",
            "transport",
            "place",
        }:
            base = bundle.phase_trajectories["lift"].knots[0].centroidal_target
            if base is None or base.com_pos_world is None:
                raise SchemaValidationError("R1 pi_H lift base target is missing")
            expected[2] = float(base.com_pos_world[2]) + float(scale) * (
                expected[2] - float(base.com_pos_world[2])
            )
        if not bool(torch.isfinite(expected).all()):
            raise SchemaValidationError("R1 pi_H nominal knot lacks a body reference")
        recorded = tensors["desired_body_pose_world"][raw_index, 0, :3]
        error = float(
            torch.linalg.vector_norm(
                recorded.cpu() - expected.to(recorded.dtype)
            ).item()
        )
        maximum = max(maximum, error)
    if maximum > 1.0e-4:
        raise SchemaValidationError(
            "R1 pi_H recorded/nominal reference alignment differs"
        )
    return maximum


def _interpolated_position(
    trajectory: ContactWrenchTrajectory, time_s: float
) -> tuple[float, float, float]:
    times = [float(knot.t_rel_s) for knot in trajectory.knots]
    upper = min(len(times) - 1, bisect_right(times, time_s))
    lower = max(0, upper - 1)
    if upper == lower:
        ratio = 0.0
    else:
        ratio = (float(time_s) - times[lower]) / (times[upper] - times[lower])
    positions = []
    for index in (lower, upper):
        target = trajectory.knots[index].centroidal_target
        if target is None or target.com_pos_world is None:
            raise SchemaValidationError("R1 pi_H nominal knot lacks a body reference")
        positions.append(target.com_pos_world)
    return tuple(
        float(positions[0][axis])
        + ratio * (float(positions[1][axis]) - float(positions[0][axis]))
        for axis in range(3)
    )


def _maximum_absolute(values: torch.Tensor, valid: torch.Tensor) -> float:
    expanded = valid
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    selected = values.masked_select(expanded.expand_as(values))
    maximum = 0.0 if selected.numel() == 0 else float(selected.abs().max().item())
    if not math.isfinite(maximum) or maximum > 1.0e-7:
        raise SchemaValidationError("R1 pi_H trace contains a learned pi_L action")
    return maximum


def _preflight_representative_pi_h_forward(
    records: tuple[InteractionTrajectoryRecord, ...],
) -> dict[str, Any]:
    """Exercise the real pi_H BC tensor path without updating parameters."""

    contexts = [
        HighLevelPolicyContext(
            irg=record.irg,
            interaction_envelope=record.interaction_envelope,
            morphology_graph=record.morphology_graph,
            contact_candidate_set=record.contact_candidate_set,
            runtime_observation=record.runtime_observation,
        )
        for record in records
    ]
    with torch.random.fork_rng(), torch.no_grad():
        torch.manual_seed(0)
        policy = Order9AutoregressiveHighLevelPolicy()
        policy.eval()
        loss = compute_order9_pi_h_behavior_cloning_loss(
            policy,
            contexts,
            [record.trajectory for record in records],
            decision_returns=[record.decision_return for record in records],
        )
    names = (
        "total",
        "assignment",
        "schedule",
        "wrench",
        "timing",
        "centroidal",
        "posture",
        "object_target",
        "priority",
        "guard",
        "value",
    )
    metrics = {name: float(getattr(loss, name).detach().cpu().item()) for name in names}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise SchemaValidationError("R1 pi_H representative BC forward is not finite")
    return {
        "record_count": len(records),
        "all_loss_terms_finite": True,
        "selected_assignment_count": loss.selected_assignment_count,
        "active_wrench_count": loss.active_wrench_count,
        "loss_terms": metrics,
        "parameter_update_performed": False,
    }


def _portable(path: Path, repository: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository))
    except ValueError:
        return str(path.resolve())


__all__ = [
    "ORDER9_R1_DECISION_INTERVAL_S",
    "ORDER9_R1_PI_H_CONVERSION_VERSION",
    "ORDER9_R1_PI_H_HORIZON_S",
    "ORDER9_R1_PI_H_KNOT_DT_S",
    "ORDER9_R1_PI_H_PREFLIGHT_VERSION",
    "build_order9_r1_pi_h_teacher_window",
    "build_order9_r1_pi_h_teacher_window_from_trace",
    "preflight_order9_r1_pi_h_dataset",
    "select_order9_r1_pi_h_decision_rows",
]
