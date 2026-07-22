from __future__ import annotations

"""Compact, hash-bound tensor artifact for real-Isaac Order 9 ``pi_L`` PPO.

The simulator hot path appends GPU tensors only.  Schema objects and compressed
JSONL records are reconstructed after simulation, so collection does not pay a
per-step Python/JSON cost.  This is an internal transport format; the persisted
learning interface remains :class:`LowLevelControlRecord`.
"""

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from amsrr.policies.order9_low_level_policy import (
    ORDER9_GLOBAL_ACTION_SIZE,
    ORDER9_PI_L_POLICY_VERSION,
)
from amsrr.schemas.common import ContactMode, SchemaValidationError
from amsrr.schemas.datasets import (
    DatasetSplit,
    LowLevelControlRecord,
    PolicyBehaviorTrace,
    StageDecisionMasks,
)
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.policies import (
    POLICY_COMMAND_CONTRACT_CENTROIDAL,
    CentroidalTarget,
    ContactAssignment,
    ControllerCommand,
    ControllerStatus,
    InteractionKnot,
    ObjectTarget,
    PolicyCommand,
    PostureTarget,
)
from amsrr.schemas.runtime import (
    ModuleRuntimeState,
    ObjectRuntimeState,
    RuntimeObservation,
    TaskProgressState,
)
from amsrr.schemas.task_spec import TaskSpec
from amsrr.simulation.order9_object_task_runtime import (
    ORDER9_OBJECT_TASK_ADAPTER_ID,
    ORDER9_OBJECT_TASK_ACTOR_PHASE_COUNT,
    ORDER9_OBJECT_TASK_ACTOR_PHASE_INDEX_BY_RUNTIME,
    ORDER9_OBJECT_TASK_ACTOR_PHASE_LABELS,
)
from amsrr.training.order9_ppo import ORDER9_PI_L_ACTION_SEMANTICS
from amsrr.training.order9_ppo import (
    ORDER9_PI_L_GRAPH_JOINT_SUMMARY_NON_FIXED,
)
from amsrr.utils.hashing import hash_file


ORDER9_TENSOR_ROLLOUT_ARTIFACT_VERSION = (
    "order9_tensor_isaac_complete_pi_l_rollout_v7_command_body_pose"
)
ORDER9_PRODUCTION_COLLECTOR_VERSION = (
    "order9_vectorized_isaac_complete_pi_l_collector_v8_command_body_pose"
)


_REQUIRED_TENSORS = {
    "valid",
    "time_s",
    "phase_index",
    "phase_progress",
    "episode_serial",
    "step_index",
    "module_pose_world",
    "module_twist_world",
    "local_joint_positions_rad",
    "local_joint_velocities_radps",
    "robot_root_pose_world",
    "robot_root_twist_world",
    "object_pose_world",
    "object_twist_world",
    "desired_body_pose_world",
    "desired_body_twist_reference",
    "desired_joint_positions_rad",
    "desired_joint_velocities_radps",
    "desired_object_pose_world",
    "phase_goal_body_pose_world",
    "phase_goal_object_pose_world",
    "selected_assignment_mask",
    "contact_schedule_index",
    "actor_controller_qp_feasible",
    "actor_controller_status_one_hot",
    "actor_allocation_residual_norm",
    "actor_task_success",
    "global_action",
    "joint_action",
    "previous_global_action",
    "recurrent_state_in",
    "recurrent_state_out",
    "old_log_prob",
    "old_value",
    "privileged_disturbance_body",
    "command_body_pose_world",
    "command_body_twist",
    "command_residual_wrench_body",
    "command_joint_position_targets_rad",
    "command_joint_velocity_targets_radps",
    "command_joint_torque_bias_nm",
    "controller_desired_wrench_body",
    "rotor_thrusts_n",
    "vectoring_joint_targets_rad",
    "allocation_residual_norm",
    "qp_feasible",
    "rotor_saturation",
    "selected_contact_forces_world",
    "prohibited_collision",
    "reward",
    "reward_terms",
    "phase_success",
    "terminal",
    "truncated",
    "bootstrap_value",
    "post_robot_root_pose_world",
    "post_robot_root_twist_world",
    "post_local_joint_positions_rad",
    "post_local_joint_velocities_radps",
    "post_object_pose_world",
    "post_object_twist_world",
}


@dataclass(frozen=True)
class Order9TensorRolloutArtifact:
    metadata: dict[str, Any]
    tensors: dict[str, torch.Tensor]
    artifact_version: str = ORDER9_TENSOR_ROLLOUT_ARTIFACT_VERSION

    @property
    def step_count(self) -> int:
        return int(self.tensors["valid"].shape[0])

    @property
    def environment_count(self) -> int:
        return int(self.tensors["valid"].shape[1])

    @property
    def environment_step_count(self) -> int:
        return int(self.tensors["valid"].sum().item())

    def validate(self) -> None:
        if self.artifact_version != ORDER9_TENSOR_ROLLOUT_ARTIFACT_VERSION:
            raise SchemaValidationError("Order9 tensor rollout version mismatch")
        if set(self.tensors) != _REQUIRED_TENSORS:
            missing = sorted(_REQUIRED_TENSORS - set(self.tensors))
            extra = sorted(set(self.tensors) - _REQUIRED_TENSORS)
            raise SchemaValidationError(
                f"Order9 tensor rollout fields differ: missing={missing}, extra={extra}"
            )
        valid = self.tensors["valid"]
        if valid.ndim != 2 or valid.dtype != torch.bool or not bool(valid.any()):
            raise SchemaValidationError("Order9 tensor rollout valid mask is invalid")
        steps, environments = valid.shape
        for name, value in self.tensors.items():
            if not isinstance(value, torch.Tensor) or value.shape[:2] != (
                steps,
                environments,
            ):
                raise SchemaValidationError(
                    f"Order9 tensor rollout {name} must begin with [T, B]"
                )
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise SchemaValidationError(
                    f"Order9 tensor rollout {name} contains non-finite values"
                )
        for name in (
            "valid",
            "selected_assignment_mask",
            "actor_controller_qp_feasible",
            "actor_task_success",
            "qp_feasible",
            "rotor_saturation",
            "prohibited_collision",
            "phase_success",
            "terminal",
            "truncated",
        ):
            if self.tensors[name].dtype != torch.bool:
                raise SchemaValidationError(f"Order9 tensor rollout {name} must be bool")
        if bool((self.tensors["terminal"] & self.tensors["truncated"]).any()):
            raise SchemaValidationError(
                "Order9 tensor rollout transition cannot be terminal and truncated"
            )
        required_metadata = {
            "generation_id",
            "pi_l_checkpoint_sha256",
            "stage_id",
            "stage_config_hash",
            "curriculum_schedule_hash",
            "config_hash",
            "morphology_graph",
            "physical_model_hash",
            "urdf_hash",
            "thrust_model_hash",
            "robot_usd_sha256",
            "simulator_version",
            "simulator_hash",
            "device",
            "random_seed",
            "topology_randomized",
            "estimated_payload_mass_kg",
            "estimated_payload_inertia_body",
            "estimated_payload_com_object",
            "task_specs",
            "environment_splits",
            "assignment_templates_by_environment",
            "object_id",
            "module_ids",
            "local_joint_ids",
            "command_local_joint_ids",
            "rotor_global_ids",
            "vectoring_global_joint_ids",
            "reward_term_names",
            "control_dt_s",
            "raw_contact_actor_input",
            "runtime_phase_labels",
            "actor_phase_labels",
            "actor_phase_index_by_runtime",
            "phase_duration_s",
        }
        missing_metadata = sorted(required_metadata - set(self.metadata))
        if missing_metadata:
            raise SchemaValidationError(
                f"Order9 tensor rollout metadata is missing {missing_metadata}"
            )
        if self.metadata["raw_contact_actor_input"] is not False:
            raise SchemaValidationError(
                "Order9 tensor rollout actor must exclude raw contact"
            )
        if tuple(self.metadata["actor_phase_labels"]) != (
            ORDER9_OBJECT_TASK_ACTOR_PHASE_LABELS
        ):
            raise SchemaValidationError(
                "Order9 tensor rollout actor phase vocabulary differs"
            )
        if tuple(self.metadata["actor_phase_index_by_runtime"]) != (
            ORDER9_OBJECT_TASK_ACTOR_PHASE_INDEX_BY_RUNTIME
        ):
            raise SchemaValidationError(
                "Order9 tensor rollout runtime/actor phase mapping differs"
            )
        if bool((self.tensors["phase_index"] < 0).any()) or bool(
            (
                self.tensors["phase_index"]
                >= ORDER9_OBJECT_TASK_ACTOR_PHASE_COUNT
            ).any()
        ):
            raise SchemaValidationError(
                "Order9 tensor rollout actor phase index is invalid"
            )
        _require_sha256(
            str(self.metadata["pi_l_checkpoint_sha256"]),
            "pi_l_checkpoint_sha256",
        )
        _require_sha256(
            str(self.metadata["physical_model_hash"]), "physical_model_hash"
        )
        for name in (
            "stage_config_hash",
            "curriculum_schedule_hash",
            "config_hash",
            "urdf_hash",
            "thrust_model_hash",
            "robot_usd_sha256",
            "simulator_hash",
        ):
            _require_sha256(str(self.metadata[name]), name)
        if not str(self.metadata["stage_id"]):
            raise SchemaValidationError("Order9 tensor rollout stage_id is empty")
        if not str(self.metadata["simulator_version"]):
            raise SchemaValidationError(
                "Order9 tensor rollout simulator_version is empty"
            )
        if not str(self.metadata["device"]):
            raise SchemaValidationError("Order9 tensor rollout device is empty")
        if int(self.metadata["random_seed"]) < 0:
            raise SchemaValidationError(
                "Order9 tensor rollout random_seed must be non-negative"
            )
        if not isinstance(self.metadata["topology_randomized"], bool):
            raise SchemaValidationError(
                "Order9 tensor rollout topology_randomized must be bool"
            )
        estimated_mass = float(self.metadata["estimated_payload_mass_kg"])
        estimated_inertia = self.metadata["estimated_payload_inertia_body"]
        estimated_com = self.metadata["estimated_payload_com_object"]
        if not math.isfinite(estimated_mass) or estimated_mass <= 0.0:
            raise SchemaValidationError(
                "Order9 tensor rollout estimated payload mass is invalid"
            )
        for name, values, width in (
            ("estimated_payload_inertia_body", estimated_inertia, 6),
            ("estimated_payload_com_object", estimated_com, 3),
        ):
            if (
                not isinstance(values, list)
                or len(values) != width
                or any(not math.isfinite(float(value)) for value in values)
            ):
                raise SchemaValidationError(
                    f"Order9 tensor rollout {name} is invalid"
                )
        morphology = MorphologyGraph.from_dict(self.metadata["morphology_graph"])
        morphology.validate()
        module_ids = tuple(int(value) for value in self.metadata["module_ids"])
        if module_ids != tuple(
            sorted(module.module_id for module in morphology.modules)
        ):
            raise SchemaValidationError(
                "Order9 tensor rollout module identity differs from morphology"
            )
        tasks = [TaskSpec.from_dict(value) for value in self.metadata["task_specs"]]
        splits = [DatasetSplit(value) for value in self.metadata["environment_splits"]]
        templates = self.metadata["assignment_templates_by_environment"]
        if not (
            len(tasks) == len(splits) == len(templates) == environments
        ):
            raise SchemaValidationError(
                "Order9 tensor rollout environment metadata width differs"
            )
        for task in tasks:
            task.validate()
        for rows in templates:
            if not isinstance(rows, list):
                raise SchemaValidationError("Order9 assignment templates must be lists")
            for row in rows:
                ContactAssignment.from_dict(row).validate()
        reward_names = [str(value) for value in self.metadata["reward_term_names"]]
        if (
            not reward_names
            or len(reward_names) != len(set(reward_names))
            or self.tensors["reward_terms"].shape[-1] != len(reward_names)
        ):
            raise SchemaValidationError(
                "Order9 tensor rollout reward-term layout differs"
            )
        if not math.isfinite(float(self.metadata["control_dt_s"])) or float(
            self.metadata["control_dt_s"]
        ) <= 0.0:
            raise SchemaValidationError("Order9 tensor rollout dt must be positive")
        self._validate_shapes(module_ids)
        self._validate_boundaries()

    def _validate_shapes(self, module_ids: tuple[int, ...]) -> None:
        tensors = self.tensors
        steps, environments = tensors["valid"].shape
        module_count = len(module_ids)
        local_joint_count = len(self.metadata["local_joint_ids"])
        command_joint_count = len(self.metadata["command_local_joint_ids"])
        rotor_count = len(self.metadata["rotor_global_ids"])
        anchor_count = len(
            {
                int(row["anchor_id"])
                for rows in self.metadata["assignment_templates_by_environment"]
                for row in rows
            }
        )
        expected = {
            "module_pose_world": (steps, environments, module_count, 7),
            "module_twist_world": (steps, environments, module_count, 6),
            "local_joint_positions_rad": (
                steps,
                environments,
                module_count,
                local_joint_count,
            ),
            "local_joint_velocities_radps": (
                steps,
                environments,
                module_count,
                local_joint_count,
            ),
            "desired_body_pose_world": (steps, environments, 7),
            "desired_body_twist_reference": (steps, environments, 6),
            "desired_object_pose_world": (steps, environments, 7),
            "phase_goal_body_pose_world": (steps, environments, 7),
            "phase_goal_object_pose_world": (steps, environments, 7),
            "desired_joint_positions_rad": (
                steps,
                environments,
                module_count,
                command_joint_count,
            ),
            "desired_joint_velocities_radps": (
                steps,
                environments,
                module_count,
                command_joint_count,
            ),
            "global_action": (
                steps,
                environments,
                ORDER9_GLOBAL_ACTION_SIZE,
            ),
            "previous_global_action": (
                steps,
                environments,
                ORDER9_GLOBAL_ACTION_SIZE,
            ),
            "command_body_pose_world": (steps, environments, 7),
            "command_body_twist": (steps, environments, 6),
            "command_residual_wrench_body": (steps, environments, 6),
            "command_joint_position_targets_rad": (
                steps,
                environments,
                module_count,
                command_joint_count,
            ),
            "command_joint_velocity_targets_radps": (
                steps,
                environments,
                module_count,
                command_joint_count,
            ),
            "command_joint_torque_bias_nm": (
                steps,
                environments,
                module_count,
                command_joint_count,
            ),
            "rotor_thrusts_n": (steps, environments, rotor_count),
            "vectoring_joint_targets_rad": (
                steps,
                environments,
                rotor_count,
            ),
            "rotor_saturation": (steps, environments, rotor_count),
            "selected_assignment_mask": (
                steps,
                environments,
                anchor_count,
            ),
            "selected_contact_forces_world": (
                steps,
                environments,
                anchor_count,
                3,
            ),
        }
        for name, shape in expected.items():
            if tuple(tensors[name].shape) != shape:
                raise SchemaValidationError(
                    f"Order9 tensor rollout {name} expected {shape}, got "
                    f"{tuple(tensors[name].shape)}"
                )

    def _validate_boundaries(self) -> None:
        tensors = self.tensors
        valid = tensors["valid"]
        environments = valid.shape[1]
        for environment in range(environments):
            indices = torch.nonzero(valid[:, environment], as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            serials = tensors["episode_serial"][indices, environment].tolist()
            steps = tensors["step_index"][indices, environment].tolist()
            terminals = tensors["terminal"][indices, environment].tolist()
            truncations = tensors["truncated"][indices, environment].tolist()
            by_serial: dict[int, list[int]] = {}
            for offset, serial in enumerate(serials):
                by_serial.setdefault(int(serial), []).append(offset)
            for offsets in by_serial.values():
                episode_steps = [int(steps[offset]) for offset in offsets]
                if episode_steps != list(range(len(episode_steps))):
                    raise SchemaValidationError(
                        "Order9 tensor rollout episode step indices are not contiguous"
                    )
                boundaries = [
                    offset
                    for offset in offsets
                    if bool(terminals[offset]) or bool(truncations[offset])
                ]
                if boundaries and boundaries != [offsets[-1]]:
                    raise SchemaValidationError(
                        "Order9 tensor rollout episode boundary is not final"
                    )


class Order9TensorRolloutBuffer:
    """Append same-layout GPU step tensors and finalize once on CPU."""

    def __init__(self, metadata: Mapping[str, Any]) -> None:
        self.metadata = dict(metadata)
        self._steps: list[dict[str, torch.Tensor]] = []

    def append(self, values: Mapping[str, torch.Tensor]) -> None:
        if set(values) != _REQUIRED_TENSORS:
            raise ValueError("Order9 rollout buffer step fields differ")
        widths = {int(value.shape[0]) for value in values.values()}
        if len(widths) != 1:
            raise ValueError("Order9 rollout buffer environment widths differ")
        self._steps.append(
            {name: value.detach().clone() for name, value in values.items()}
        )

    def finalize(self) -> Order9TensorRolloutArtifact:
        if not self._steps:
            raise ValueError("Order9 rollout buffer is empty")
        artifact = Order9TensorRolloutArtifact(
            metadata=dict(self.metadata),
            tensors={
                name: torch.stack(
                    [step[name] for step in self._steps], dim=0
                ).to(device="cpu")
                for name in sorted(_REQUIRED_TENSORS)
            },
        )
        artifact.validate()
        return artifact


def write_order9_tensor_rollout_artifact(
    path: str | Path, artifact: Order9TensorRolloutArtifact
) -> str:
    artifact.validate()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Order9 raw rollout already exists: {target}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(
                {
                    "artifact_version": artifact.artifact_version,
                    "metadata": artifact.metadata,
                    "tensors": artifact.tensors,
                },
                handle,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return hash_file(target)


def load_order9_tensor_rollout_artifact(
    path: str | Path, *, expected_sha256: str | None = None
) -> Order9TensorRolloutArtifact:
    source = Path(path)
    actual = hash_file(source)
    if expected_sha256 is not None and actual != expected_sha256:
        raise SchemaValidationError("Order9 raw rollout SHA-256 mismatch")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or set(payload) != {
        "artifact_version",
        "metadata",
        "tensors",
    }:
        raise SchemaValidationError("Order9 raw rollout payload keys differ")
    artifact = Order9TensorRolloutArtifact(
        artifact_version=str(payload["artifact_version"]),
        metadata=dict(payload["metadata"]),
        tensors=dict(payload["tensors"]),
    )
    artifact.validate()
    return artifact


def order9_pi_l_records_from_tensor_artifact(
    artifact: Order9TensorRolloutArtifact,
    *,
    record_namespace: str | None = None,
) -> tuple[LowLevelControlRecord, ...]:
    """Reconstruct the existing exact-replay schema after simulation."""

    artifact.validate()
    if record_namespace is not None:
        if not record_namespace or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in record_namespace
        ):
            raise SchemaValidationError(
                "Order9 tensor rollout record namespace is invalid"
            )
    metadata = artifact.metadata
    tensors = artifact.tensors
    morphology = MorphologyGraph.from_dict(metadata["morphology_graph"])
    tasks = [TaskSpec.from_dict(value) for value in metadata["task_specs"]]
    splits = [DatasetSplit(value) for value in metadata["environment_splits"]]
    templates = metadata["assignment_templates_by_environment"]
    module_ids = tuple(int(value) for value in metadata["module_ids"])
    local_joint_ids = tuple(str(value) for value in metadata["local_joint_ids"])
    command_joint_ids = tuple(
        str(value) for value in metadata["command_local_joint_ids"]
    )
    rotor_ids = tuple(str(value) for value in metadata["rotor_global_ids"])
    vectoring_ids = tuple(
        str(value) for value in metadata["vectoring_global_joint_ids"]
    )
    reward_names = tuple(str(value) for value in metadata["reward_term_names"])
    checkpoint = str(metadata["pi_l_checkpoint_sha256"])
    generation = str(metadata["generation_id"])
    object_id = str(metadata["object_id"])
    records: list[LowLevelControlRecord] = []
    valid_rows = torch.nonzero(tensors["valid"], as_tuple=False).tolist()
    for time_index, environment in valid_rows:
        task = tasks[environment]
        phase_index = int(tensors["phase_index"][time_index, environment].item())
        phase_label = ORDER9_OBJECT_TASK_ACTOR_PHASE_LABELS[phase_index]
        serial = int(tensors["episode_serial"][time_index, environment].item())
        step_index = int(tensors["step_index"][time_index, environment].item())
        shard = (
            ""
            if record_namespace is None
            else f":shard:{record_namespace}"
        )
        episode_id = (
            f"{generation}{shard}:env:{environment:04d}:episode:{serial:06d}"
        )
        status = _controller_status(
            tensors["actor_controller_status_one_hot"][time_index, environment],
            qp_feasible=bool(
                tensors["actor_controller_qp_feasible"][time_index, environment]
            ),
            residual=float(
                tensors["actor_allocation_residual_norm"][time_index, environment]
            ),
        )
        observation = _runtime_observation(
            time_s=float(tensors["time_s"][time_index, environment]),
            morphology=morphology,
            module_ids=module_ids,
            local_joint_ids=local_joint_ids,
            module_pose=tensors["module_pose_world"][time_index, environment],
            module_twist=tensors["module_twist_world"][time_index, environment],
            joint_position=tensors["local_joint_positions_rad"][time_index, environment],
            joint_velocity=tensors["local_joint_velocities_radps"][time_index, environment],
            object_id=object_id,
            object_pose=tensors["object_pose_world"][time_index, environment],
            object_twist=tensors["object_twist_world"][time_index, environment],
            controller_status=status,
            phase_index=phase_index,
            phase_progress=float(
                tensors["phase_progress"][time_index, environment]
            ),
            task_success=bool(
                tensors["actor_task_success"][time_index, environment]
            ),
        )
        assignments = _active_assignments(
            templates[environment],
            schedule_index=int(
                tensors["contact_schedule_index"][time_index, environment]
            ),
            selected_mask=tensors["selected_assignment_mask"][
                time_index, environment
            ],
        )
        desired_pose = _float_list(
            tensors["desired_body_pose_world"][time_index, environment]
        )
        desired_twist = _float_list(
            tensors["desired_body_twist_reference"][time_index, environment]
        )
        desired_object = _float_list(
            tensors["desired_object_pose_world"][time_index, environment]
        )
        active_knot = InteractionKnot(
            t_rel_s=0.0,
            contact_assignments=assignments,
            centroidal_target=CentroidalTarget(
                com_pos_world=tuple(desired_pose[:3]),
                com_vel_world=tuple(desired_twist[:3]),
                body_orientation_world=tuple(desired_pose[3:7]),
                centroidal_wrench_preference=_float_list(
                    tensors["controller_desired_wrench_body"][
                        time_index, environment
                    ]
                ),
            ),
            posture_target=PostureTarget(
                joint_pos_target=_joint_target_map(
                    tensors["desired_joint_positions_rad"][
                        time_index, environment
                    ],
                    module_ids=module_ids,
                    joint_ids=command_joint_ids,
                ),
                joint_vel_target=_joint_target_map(
                    tensors["desired_joint_velocities_radps"][
                        time_index, environment
                    ],
                    module_ids=module_ids,
                    joint_ids=command_joint_ids,
                ),
                free_anchor_pose_targets={},
            ),
            object_targets=[
                ObjectTarget(
                    object_id=object_id,
                    pose_target_world=tuple(desired_object),
                )
            ],
            priority_weights={"order9_object_task": 1.0},
            guard_conditions=[
                {"type": "order9_task_phase", "phase_label": phase_label}
            ],
        )
        policy_command = _policy_command(
            tensors=tensors,
            time_index=time_index,
            environment=environment,
            module_ids=module_ids,
            joint_ids=command_joint_ids,
        )
        controller_command = _controller_command(
            tensors=tensors,
            time_index=time_index,
            environment=environment,
            module_ids=module_ids,
            command_joint_ids=command_joint_ids,
            rotor_ids=rotor_ids,
            vectoring_ids=vectoring_ids,
        )
        reward_terms = {
            name: float(
                tensors["reward_terms"][time_index, environment, index]
            )
            for index, name in enumerate(reward_names)
        }
        reward_terms.update(
            {
                "raw_isaac_privileged_evidence": 1.0,
                "raw_contact_actor_input": 0.0,
                "prohibited_collision": float(
                    bool(tensors["prohibited_collision"][time_index, environment])
                ),
                "phase_success": float(
                    bool(tensors["phase_success"][time_index, environment])
                ),
            }
        )
        behavior = PolicyBehaviorTrace(
            policy_family="pi_l",
            policy_version=ORDER9_PI_L_POLICY_VERSION,
            action_semantics=ORDER9_PI_L_ACTION_SEMANTICS,
            action_payload={
                "global_action": _float_list(
                    tensors["global_action"][time_index, environment]
                ),
                "module_ids": list(module_ids),
                "joint_action": tensors["joint_action"][
                    time_index, environment
                ].tolist(),
                "previous_global_action": _float_list(
                    tensors["previous_global_action"][time_index, environment]
                ),
                "privileged_disturbance_body": _float_list(
                    tensors["privileged_disturbance_body"][
                        time_index, environment
                    ]
                ),
                "actor_graph_frame_origin_world": _actor_graph_frame_origin(
                    task
                ),
                "actor_graph_joint_summary_semantics": (
                    ORDER9_PI_L_GRAPH_JOINT_SUMMARY_NON_FIXED
                ),
            },
            stochastic=True,
            policy_checkpoint_sha256=checkpoint,
            old_log_prob=float(tensors["old_log_prob"][time_index, environment]),
            old_value=float(tensors["old_value"][time_index, environment]),
            recurrent_state_in=_float_list(
                tensors["recurrent_state_in"][time_index, environment]
            ),
            recurrent_state_out=_float_list(
                tensors["recurrent_state_out"][time_index, environment]
            ),
        )
        record = LowLevelControlRecord(
            record_id=f"{episode_id}:step:{step_index:06d}",
            episode_id=episode_id,
            task_id=task.task_id,
            split=splits[environment],
            step_index=step_index,
            time_s=float(tensors["time_s"][time_index, environment]),
            trajectory_record_id=f"{episode_id}:trajectory:reference",
            active_trajectory_index=phase_index,
            active_knot_index=0,
            runtime_observation=observation,
            active_knot=active_knot,
            policy_command=policy_command,
            controller_command=controller_command,
            actuator_target_record={
                "source": ORDER9_TENSOR_ROLLOUT_ARTIFACT_VERSION,
                "rotor_thrusts_n": dict(controller_command.rotor_thrusts_n),
                "vectoring_joint_targets": dict(
                    controller_command.vectoring_joint_targets
                ),
                "joint_position_targets": dict(
                    controller_command.joint_position_targets
                ),
                "joint_velocity_targets": dict(
                    controller_command.joint_velocity_targets
                ),
                "joint_torque_bias": dict(controller_command.joint_torque_bias),
            },
            reward_terms=reward_terms,
            reward=float(tensors["reward"][time_index, environment]),
            terminal=bool(tensors["terminal"][time_index, environment]),
            truncated=bool(tensors["truncated"][time_index, environment]),
            bootstrap_value=float(
                tensors["bootstrap_value"][time_index, environment]
            ),
            stage_masks=StageDecisionMasks(low_level_control_mask=True),
            task_type=task.task_type.value,
            task_adapter_id=ORDER9_OBJECT_TASK_ADAPTER_ID,
            phase_index=phase_index,
            phase_count=ORDER9_OBJECT_TASK_ACTOR_PHASE_COUNT,
            behavior_trace=behavior,
        )
        record.validate()
        records.append(record)
    return tuple(records)


def _runtime_observation(
    *,
    time_s: float,
    morphology: MorphologyGraph,
    module_ids: tuple[int, ...],
    local_joint_ids: tuple[str, ...],
    module_pose: torch.Tensor,
    module_twist: torch.Tensor,
    joint_position: torch.Tensor,
    joint_velocity: torch.Tensor,
    object_id: str,
    object_pose: torch.Tensor,
    object_twist: torch.Tensor,
    controller_status: ControllerStatus,
    phase_index: int,
    phase_progress: float,
    task_success: bool,
) -> RuntimeObservation:
    module_states = []
    for index, module_id in enumerate(module_ids):
        module_states.append(
            ModuleRuntimeState(
                module_id=module_id,
                pose_world=tuple(_float_list(module_pose[index])),
                twist_world=_float_list(module_twist[index]),
                joint_positions={
                    joint_id: float(joint_position[index, joint_index])
                    for joint_index, joint_id in enumerate(local_joint_ids)
                },
                joint_velocities={
                    joint_id: float(joint_velocity[index, joint_index])
                    for joint_index, joint_id in enumerate(local_joint_ids)
                },
                health=1.0,
            )
        )
    observation = RuntimeObservation(
        time_s=time_s,
        morphology_graph=MorphologyGraph.from_dict(morphology.to_dict()),
        module_states=module_states,
        object_states=[
            ObjectRuntimeState(
                object_id=object_id,
                pose_world=tuple(_float_list(object_pose)),
                twist_world=_float_list(object_twist),
            )
        ],
        contact_states=[],
        controller_status=controller_status,
        task_progress=TaskProgressState(
            phase_label=ORDER9_OBJECT_TASK_ACTOR_PHASE_LABELS[phase_index],
            progress_ratio=phase_progress,
            success=task_success,
            metrics={
                "phase_index": float(phase_index),
                "phase_count": float(ORDER9_OBJECT_TASK_ACTOR_PHASE_COUNT),
            },
        ),
    )
    observation.validate()
    return observation


def _actor_graph_frame_origin(task: TaskSpec) -> list[float]:
    values = task.metadata.get(
        "isaac_environment_origin_world", [0.0, 0.0, 0.0]
    )
    if (
        not isinstance(values, list)
        or len(values) != 3
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in values
        )
    ):
        raise SchemaValidationError(
            "Order9 task actor graph-frame origin is invalid"
        )
    return [float(value) for value in values]


def _active_assignments(
    raw_templates: Sequence[Mapping[str, Any]],
    *,
    schedule_index: int,
    selected_mask: torch.Tensor,
) -> list[ContactAssignment]:
    schedule_by_index = {
        0: None,
        1: "approach",
        2: "attach",
        3: "maintain",
        4: "release",
    }
    if schedule_index not in schedule_by_index:
        raise SchemaValidationError("Order9 contact schedule index is invalid")
    schedule = schedule_by_index[schedule_index]
    if schedule is None:
        return []
    if len(raw_templates) != selected_mask.numel():
        raise SchemaValidationError("Order9 assignment template/mask widths differ")
    output: list[ContactAssignment] = []
    for enabled, raw in zip(selected_mask.tolist(), raw_templates):
        if not bool(enabled):
            continue
        assignment = ContactAssignment.from_dict(dict(raw))
        assignment.schedule_state = schedule
        if schedule in {"approach", "release"}:
            assignment.wrench_target = None
            assignment.wrench_lower = None
            assignment.wrench_upper = None
        assignment.validate()
        output.append(assignment)
    return output


def _joint_target_map(
    values: torch.Tensor,
    *,
    module_ids: tuple[int, ...],
    joint_ids: tuple[str, ...],
) -> dict[str, float]:
    if values.shape != (len(module_ids), len(joint_ids)):
        raise SchemaValidationError("Order9 active-knot joint target shape differs")
    return {
        f"module_{module_id}:{joint_id}": float(values[module_index, joint_index])
        for module_index, module_id in enumerate(module_ids)
        for joint_index, joint_id in enumerate(joint_ids)
    }


def _policy_command(
    *,
    tensors: Mapping[str, torch.Tensor],
    time_index: int,
    environment: int,
    module_ids: tuple[int, ...],
    joint_ids: tuple[str, ...],
) -> PolicyCommand:
    q = tensors["command_joint_position_targets_rad"][time_index, environment]
    qdot = tensors["command_joint_velocity_targets_radps"][time_index, environment]
    torque = tensors["command_joint_torque_bias_nm"][time_index, environment]
    return PolicyCommand(
        desired_body_pose=_float_list(
            tensors["command_body_pose_world"][time_index, environment]
        ),
        desired_body_twist=_float_list(
            tensors["command_body_twist"][time_index, environment]
        ),
        residual_wrench_body=_float_list(
            tensors["command_residual_wrench_body"][time_index, environment]
        ),
        control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
        joint_position_targets=_joint_mapping(q, module_ids, joint_ids),
        joint_velocity_targets=_joint_mapping(qdot, module_ids, joint_ids),
        joint_torque_bias=_joint_mapping(torque, module_ids, joint_ids),
        priority_weights={"order9_learned_overlay": 1.0},
    )


def _controller_command(
    *,
    tensors: Mapping[str, torch.Tensor],
    time_index: int,
    environment: int,
    module_ids: tuple[int, ...],
    command_joint_ids: tuple[str, ...],
    rotor_ids: tuple[str, ...],
    vectoring_ids: tuple[str, ...],
) -> ControllerCommand:
    feasible = bool(tensors["qp_feasible"][time_index, environment])
    residual = float(tensors["allocation_residual_norm"][time_index, environment])
    status = ControllerStatus(
        status="ok" if feasible else "infeasible",
        qp_feasible=feasible,
        active_mode="rigid_body_qp",
        metrics={"allocation_residual_norm": residual},
    )
    q = tensors["command_joint_position_targets_rad"][time_index, environment]
    qdot = tensors["command_joint_velocity_targets_radps"][time_index, environment]
    torque = tensors["command_joint_torque_bias_nm"][time_index, environment]
    return ControllerCommand(
        rotor_thrusts_n={
            rotor_id: float(value)
            for rotor_id, value in zip(
                rotor_ids,
                tensors["rotor_thrusts_n"][time_index, environment].tolist(),
            )
        },
        vectoring_joint_targets={
            joint_id: float(value)
            for joint_id, value in zip(
                vectoring_ids,
                tensors["vectoring_joint_targets_rad"][
                    time_index, environment
                ].tolist(),
            )
        },
        joint_torque_commands={},
        dock_mechanism_commands=_joint_mapping(
            qdot, module_ids, command_joint_ids
        ),
        controller_status=status,
        control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
        joint_position_targets=_joint_mapping(q, module_ids, command_joint_ids),
        joint_velocity_targets=_joint_mapping(
            qdot, module_ids, command_joint_ids
        ),
        joint_torque_bias=_joint_mapping(
            torque, module_ids, command_joint_ids
        ),
    )


def _controller_status(
    one_hot: torch.Tensor, *, qp_feasible: bool, residual: float
) -> ControllerStatus:
    labels = ("ok", "warning", "infeasible", "fault")
    index = int(torch.argmax(one_hot).item())
    return ControllerStatus(
        status=labels[index],  # type: ignore[arg-type]
        qp_feasible=qp_feasible,
        active_mode="rigid_body_qp",
        metrics={"allocation_residual_norm": residual},
    )


def _joint_mapping(
    values: torch.Tensor,
    module_ids: tuple[int, ...],
    joint_ids: tuple[str, ...],
) -> dict[str, float]:
    if values.shape != (len(module_ids), len(joint_ids)):
        raise SchemaValidationError("Order9 joint command tensor shape differs")
    return {
        f"module_{module_id}:{joint_id}": float(values[module_index, joint_index])
        for module_index, module_id in enumerate(module_ids)
        for joint_index, joint_id in enumerate(joint_ids)
    }


def _float_list(value: torch.Tensor) -> list[float]:
    return [float(item) for item in value.tolist()]


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise SchemaValidationError(f"Order9 tensor rollout {name} is not SHA-256")


__all__ = [
    "ORDER9_PRODUCTION_COLLECTOR_VERSION",
    "ORDER9_TENSOR_ROLLOUT_ARTIFACT_VERSION",
    "Order9TensorRolloutArtifact",
    "Order9TensorRolloutBuffer",
    "load_order9_tensor_rollout_artifact",
    "order9_pi_l_records_from_tensor_artifact",
    "write_order9_tensor_rollout_artifact",
]
