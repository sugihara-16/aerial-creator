from __future__ import annotations

"""Immutable, semantic C3 execution bundles.

The ordinary nominal manifests protect bytes.  This bundle additionally binds
the checkpoint, runtime configuration, action contract, reset banks, collision
admission, and the robot-target transforms which must remain present.  It is
therefore the single runtime input used to prevent a later manifest from
silently selecting an older-but-schema-compatible trajectory lineage.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from amsrr.schemas.common import SchemaValidationError
from amsrr.training.order9_c3_action_contract import ORDER9_C3_ACTION_CONTRACTS
from amsrr.training.order9_c3_nominal_trajectory import (
    Order9C3NominalTrajectoryArtifact,
    validate_order9_c3_nominal_trajectory_artifact_bytes,
    validate_order9_c3_nominal_trajectory_set_bytes,
)
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_pipeline import order9_schedule_hash, order9_stage_by_id
from amsrr.training.order9_rollout_buckets import (
    load_order9_pi_l_rollout_bucket_manifest,
    validate_order9_pi_l_rollout_bucket_bytes,
)
from amsrr.utils.hashing import hash_file, stable_hash


ORDER9_C3_EXECUTION_BUNDLE_VERSION = "order9_c3_execution_bundle_v1"
ORDER9_C3_STAGE_ID = "c3_pi_l_ppo_arbitrary_morphology"


@dataclass(frozen=True)
class Order9C3ExecutionBucketBinding:
    bucket_id: str
    module_count: int
    artifact_sha256: str
    reset_bank_path: Path
    reset_bank_sha256: str
    collision_validation_path: Path
    collision_validation_sha256: str


@dataclass(frozen=True)
class Order9C3ExecutionBundle:
    source_path: Path
    source_sha256: str
    config_path: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    bucket_manifest_path: Path
    bucket_manifest_sha256: str
    nominal_set_manifest_path: Path
    nominal_set_manifest_sha256: str
    action_contract: str
    bucket_bindings: Mapping[str, Order9C3ExecutionBucketBinding]
    payload: Mapping[str, Any]

    def binding(self, bucket_id: str) -> Order9C3ExecutionBucketBinding:
        try:
            return self.bucket_bindings[bucket_id]
        except KeyError as error:
            raise SchemaValidationError(
                f"C3 execution bundle does not admit bucket: {bucket_id}"
            ) from error

    def provenance(self) -> dict[str, Any]:
        return {
            "bundle_version": ORDER9_C3_EXECUTION_BUNDLE_VERSION,
            "bundle_path": str(self.source_path),
            "bundle_sha256": self.source_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "bucket_manifest_sha256": self.bucket_manifest_sha256,
            "nominal_set_manifest_sha256": self.nominal_set_manifest_sha256,
            "action_contract": self.action_contract,
        }


def load_order9_c3_execution_bundle(
    bundle_path: str | Path,
    *,
    repository_root: str | Path,
) -> Order9C3ExecutionBundle:
    """Load and fully validate a C3 runtime bundle before Isaac is launched."""

    repository = Path(repository_root).resolve()
    source = _resolve(bundle_path, repository)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SchemaValidationError("C3 execution bundle must be an object")
    if payload.get("bundle_version") != ORDER9_C3_EXECUTION_BUNDLE_VERSION:
        raise SchemaValidationError("C3 execution bundle version mismatch")
    if payload.get("stage_id") != ORDER9_C3_STAGE_ID:
        raise SchemaValidationError("C3 execution bundle stage mismatch")

    action_contract = _required_string(payload, "action_contract")
    if action_contract not in ORDER9_C3_ACTION_CONTRACTS:
        raise SchemaValidationError("C3 execution bundle action contract is invalid")

    config = _required_mapping(payload, "config")
    checkpoint = _required_mapping(payload, "checkpoint")
    buckets = _required_mapping(payload, "bucket_manifest")
    nominal = _required_mapping(payload, "nominal_set")
    config_path = _bound_file(config, repository, "config")
    checkpoint_path = _bound_file(checkpoint, repository, "checkpoint")
    bucket_manifest_path = _bound_file(buckets, repository, "bucket_manifest")
    nominal_set_manifest_path = _bound_file(nominal, repository, "nominal_set")

    learning = load_order9_learning_config(config_path)
    stage = order9_stage_by_id(learning, ORDER9_C3_STAGE_ID)
    if config.get("semantic_hash") != stable_hash(learning.to_dict()):
        raise SchemaValidationError("C3 execution bundle config semantics changed")
    if config.get("stage_semantic_hash") != stable_hash(stage.to_dict()):
        raise SchemaValidationError("C3 execution bundle stage semantics changed")
    if config.get("schedule_hash") != order9_schedule_hash(learning):
        raise SchemaValidationError("C3 execution bundle schedule changed")
    _validate_checkpoint_metadata(checkpoint_path, checkpoint)

    validate_order9_pi_l_rollout_bucket_bytes(
        bucket_manifest_path, repository_root=repository
    )
    bucket_manifest = load_order9_pi_l_rollout_bucket_manifest(
        bucket_manifest_path
    )
    if (
        bucket_manifest.config_hash != config["semantic_hash"]
        or bucket_manifest.stage_config_hash != config["stage_semantic_hash"]
        or bucket_manifest.curriculum_schedule_hash != config["schedule_hash"]
    ):
        raise SchemaValidationError(
            "C3 execution bucket manifest is not bound to the bundle config"
        )

    nominal_manifest = validate_order9_c3_nominal_trajectory_set_bytes(
        nominal_set_manifest_path,
        repository_root=repository,
        expected_sha256=_required_sha256(nominal, "sha256"),
    )
    nominal_entries = {entry.bucket_id: entry for entry in nominal_manifest.entries}
    rollout_buckets = {bucket.bucket_id: bucket for bucket in bucket_manifest.buckets}

    source_bindings = payload.get("runtime_source_bindings", [])
    if not isinstance(source_bindings, list) or not source_bindings:
        raise SchemaValidationError(
            "C3 execution bundle requires runtime source bindings"
        )
    for index, binding in enumerate(source_bindings):
        if not isinstance(binding, dict):
            raise SchemaValidationError(
                f"C3 runtime source binding {index} is invalid"
            )
        _bound_file(binding, repository, f"runtime_source_bindings[{index}]")

    required_semantics = _required_mapping(payload, "required_nominal_semantics")
    raw_bucket_bindings = payload.get("bucket_bindings")
    if not isinstance(raw_bucket_bindings, list) or not raw_bucket_bindings:
        raise SchemaValidationError("C3 execution bundle has no bucket bindings")
    verified: dict[str, Order9C3ExecutionBucketBinding] = {}
    for raw in raw_bucket_bindings:
        if not isinstance(raw, dict):
            raise SchemaValidationError("C3 execution bucket binding is invalid")
        bucket_id = _required_string(raw, "bucket_id")
        if bucket_id in verified:
            raise SchemaValidationError("C3 execution bundle repeats a bucket")
        if bucket_id not in nominal_entries or bucket_id not in rollout_buckets:
            raise SchemaValidationError(
                f"C3 execution bundle bucket is absent from its manifests: {bucket_id}"
            )
        entry = nominal_entries[bucket_id]
        rollout_bucket = rollout_buckets[bucket_id]
        artifact_sha256 = _required_sha256(raw, "artifact_sha256")
        if entry.artifact_sha256 != artifact_sha256:
            raise SchemaValidationError(
                f"C3 execution nominal artifact changed: {bucket_id}"
            )
        accepted = rollout_bucket.metadata.get("accepted_nominal_trajectory")
        if not isinstance(accepted, dict) or (
            accepted.get("set_manifest_sha256") != nominal["sha256"]
            or accepted.get("artifact_sha256") != artifact_sha256
        ):
            raise SchemaValidationError(
                f"C3 execution rollout bucket selected another nominal: {bucket_id}"
            )
        artifact_path = nominal_set_manifest_path.parent / entry.artifact_path
        artifact = validate_order9_c3_nominal_trajectory_artifact_bytes(
            artifact_path
        )
        _validate_nominal_semantics(
            artifact,
            required=required_semantics,
            bucket_id=bucket_id,
        )

        reset_bank_path = _bound_file(
            _required_mapping(raw, "reset_bank"), repository, "reset_bank"
        )
        _validate_reset_bank(
            reset_bank_path,
            nominal_set_path=nominal_set_manifest_path,
            nominal_set_sha256=nominal["sha256"],
            artifact_sha256=artifact_sha256,
            bucket_id=bucket_id,
        )
        collision = _required_mapping(raw, "collision_validation")
        collision_path = _bound_file(
            collision, repository, "collision_validation"
        )
        _validate_collision_admission(collision_path, bucket_id=bucket_id)
        verified[bucket_id] = Order9C3ExecutionBucketBinding(
            bucket_id=bucket_id,
            module_count=int(raw.get("module_count", -1)),
            artifact_sha256=artifact_sha256,
            reset_bank_path=reset_bank_path,
            reset_bank_sha256=_required_sha256(
                _required_mapping(raw, "reset_bank"), "sha256"
            ),
            collision_validation_path=collision_path,
            collision_validation_sha256=_required_sha256(collision, "sha256"),
        )
        if verified[bucket_id].module_count != int(entry.module_count):
            raise SchemaValidationError(
                f"C3 execution module count changed: {bucket_id}"
            )

    expected_counts = payload.get("module_count_histogram")
    actual_counts: dict[str, int] = {}
    for binding in verified.values():
        key = str(binding.module_count)
        actual_counts[key] = actual_counts.get(key, 0) + 1
    if expected_counts != actual_counts:
        raise SchemaValidationError(
            "C3 execution bundle bucket/module coverage changed"
        )

    return Order9C3ExecutionBundle(
        source_path=source,
        source_sha256=hash_file(source),
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=_required_sha256(checkpoint, "sha256"),
        bucket_manifest_path=bucket_manifest_path,
        bucket_manifest_sha256=_required_sha256(buckets, "sha256"),
        nominal_set_manifest_path=nominal_set_manifest_path,
        nominal_set_manifest_sha256=_required_sha256(nominal, "sha256"),
        action_contract=action_contract,
        bucket_bindings=verified,
        payload=payload,
    )


def apply_order9_c3_execution_bundle_to_args(
    args: Any,
    bundle: Order9C3ExecutionBundle,
) -> None:
    """Hydrate/check one rollout CLI namespace from an immutable bundle."""

    _set_or_require(args, "config", str(bundle.config_path))
    _set_or_require(args, "pi_l_checkpoint", str(bundle.checkpoint_path))
    _set_or_require(args, "pi_l_checkpoint_sha256", bundle.checkpoint_sha256)
    _set_or_require(
        args, "c3_nominal_set_manifest", str(bundle.nominal_set_manifest_path)
    )
    _set_or_require(
        args, "c3_nominal_set_sha256", bundle.nominal_set_manifest_sha256
    )
    _set_or_require(args, "c3_action_contract", bundle.action_contract)
    task_path = getattr(args, "task_spec_json", None)
    if not task_path:
        raise SchemaValidationError(
            "C3 execution bundle requires --task-spec-json to select a bucket"
        )
    task_payload = json.loads(Path(task_path).resolve().read_text(encoding="utf-8"))
    metadata = task_payload.get("metadata")
    bucket_id = (
        metadata.get("order9_rollout_bucket_id")
        if isinstance(metadata, dict)
        else None
    )
    if not isinstance(bucket_id, str) or not bucket_id:
        raise SchemaValidationError("C3 execution bundle task lacks bucket identity")
    binding = bundle.binding(bucket_id)
    _set_or_require(
        args, "c3_nominal_artifact_sha256", binding.artifact_sha256
    )
    _set_or_require(args, "c3_reset_bank", str(binding.reset_bank_path))


def _validate_nominal_semantics(
    artifact: Order9C3NominalTrajectoryArtifact,
    *,
    required: Mapping[str, Any],
    bucket_id: str,
) -> None:
    expected_phases = required.get("phase_sequence")
    actual_phases = [phase.phase for phase in artifact.phase_trajectories]
    if expected_phases != actual_phases:
        raise SchemaValidationError(
            f"C3 execution nominal phase sequence changed: {bucket_id}"
        )
    expected_translation = required.get("rigid_robot_target_translation")
    actual_translation = artifact.selection_evidence.get(
        "rigid_robot_target_translation"
    )
    if not isinstance(expected_translation, dict) or not isinstance(
        actual_translation, dict
    ):
        raise SchemaValidationError(
            f"C3 execution nominal lacks required robot translation: {bucket_id}"
        )
    for key in (
        "repair_version",
        "offset_world_m",
        "phase_z_offset_start_end_m",
        "phase_smooth_z_offset_start_end_m",
        "smooth_offset_boundary_velocity_zero",
        "joint_targets_changed",
        "object_targets_changed",
        "contact_assignments_changed",
        "runtime_planner_enabled",
    ):
        if actual_translation.get(key) != expected_translation.get(key):
            raise SchemaValidationError(
                f"C3 execution nominal translation semantic '{key}' changed: "
                f"{bucket_id}"
            )
    required_tokens = required.get("generation_method_tokens_by_phase")
    if not isinstance(required_tokens, dict):
        raise SchemaValidationError("C3 execution generation tokens are missing")
    methods = {
        phase.phase: phase.generation_method
        for phase in artifact.phase_trajectories
    }
    for phase, tokens in required_tokens.items():
        if phase not in methods or not isinstance(tokens, list) or any(
            not isinstance(token, str) or token not in methods[phase]
            for token in tokens
        ):
            raise SchemaValidationError(
                f"C3 execution nominal transform lineage changed: "
                f"{bucket_id}/{phase}"
            )
    if artifact.final_phase_target_reached is not True:
        raise SchemaValidationError(
            f"C3 execution nominal no longer reaches its final target: {bucket_id}"
        )


def _validate_reset_bank(
    path: Path,
    *,
    nominal_set_path: Path,
    nominal_set_sha256: str,
    artifact_sha256: str,
    bucket_id: str,
) -> None:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    contract = payload.get("contract") if isinstance(payload, dict) else None
    reference = (
        contract.get("c3_nominal_reference")
        if isinstance(contract, dict)
        else None
    )
    if not isinstance(reference, dict) or (
        Path(str(reference.get("set_manifest_path", ""))).name
        != nominal_set_path.name
        or reference.get("set_manifest_sha256") != nominal_set_sha256
        or reference.get("artifact_sha256") != artifact_sha256
    ):
        raise SchemaValidationError(
            f"C3 reset bank selected another nominal: {bucket_id}"
        )


def _validate_checkpoint_metadata(
    path: Path, binding: Mapping[str, Any]
) -> None:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    required = binding.get("required_metadata")
    if not isinstance(metadata, dict) or not isinstance(required, dict):
        raise SchemaValidationError(
            "C3 execution checkpoint metadata binding is missing"
        )
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise SchemaValidationError(
                f"C3 execution checkpoint metadata changed: {key}"
            )


def _validate_collision_admission(path: Path, *, bucket_id: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    matches = (
        [
            record
            for record in records
            if isinstance(record, dict) and record.get("bucket_id") == bucket_id
        ]
        if isinstance(records, list)
        else []
    )
    if (
        payload.get("all_accepted") is not True
        or len(matches) != 1
        or matches[0].get("accepted") is not True
    ):
        raise SchemaValidationError(
            f"C3 execution collision admission failed: {bucket_id}"
        )


def _bound_file(
    binding: Mapping[str, Any], repository: Path, label: str
) -> Path:
    path = _resolve(_required_string(binding, "path"), repository)
    expected = _required_sha256(binding, "sha256")
    if not path.is_file() or hash_file(path) != expected:
        raise SchemaValidationError(f"C3 execution {label} bytes changed")
    return path


def _set_or_require(args: Any, name: str, expected: str) -> None:
    current = getattr(args, name, None)
    if current is not None:
        current_value = str(Path(current).resolve()) if name.endswith(("config", "checkpoint", "manifest", "bank")) else str(current)
        expected_value = str(Path(expected).resolve()) if name.endswith(("config", "checkpoint", "manifest", "bank")) else str(expected)
        if current_value != expected_value:
            raise SchemaValidationError(
                f"C3 execution CLI override differs from bundle: {name}"
            )
    setattr(args, name, expected)


def _required_mapping(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise SchemaValidationError(f"C3 execution bundle lacks {name}")
    return value


def _required_string(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise SchemaValidationError(f"C3 execution bundle lacks {name}")
    return value


def _required_sha256(payload: Mapping[str, Any], name: str) -> str:
    value = _required_string(payload, name)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise SchemaValidationError(f"C3 execution bundle {name} is not sha256")
    return value


def _resolve(path: str | Path, repository: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (repository / value).resolve()


__all__ = [
    "ORDER9_C3_EXECUTION_BUNDLE_VERSION",
    "Order9C3ExecutionBucketBinding",
    "Order9C3ExecutionBundle",
    "apply_order9_c3_execution_bundle_to_args",
    "load_order9_c3_execution_bundle",
]
