from __future__ import annotations

"""Hash-bound reset and snapshot contracts for the Order 9 Isaac task.

The fast training scene starts individual curriculum phases from the accepted
Order 8 trajectory instead of replaying the roughly eighty-second acquisition
prefix for every PPO fragment.  Only simulator state is copied: success and
contact labels are recomputed by the new rollout.  This module deliberately
contains no Isaac imports so the provenance and state contracts can be tested
outside Kit.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_non_empty
from amsrr.utils.hashing import hash_file, stable_hash


ORDER9_CANONICAL_RESET_VERSION = "order9_order8_bound_reset_v1"
ORDER9_ISAAC_SNAPSHOT_VERSION = "order9_isaac_snapshot_v1"


@dataclass
class Order9CanonicalObjectTaskReset(SchemaBase):
    """Minimal physical reset state extracted from accepted Order 8 evidence."""

    source_report_path: str
    source_report_sha256: str
    source_graph_id: str
    robot_root_pose_world: list[float]
    robot_root_twist_world: list[float]
    joint_positions_rad: dict[str, float]
    joint_velocities_radps: dict[str, float]
    object_pose_world: list[float]
    object_twist_world: list[float]
    open_joint_positions_rad: dict[str, float]
    transport_distance_m: float
    lift_clearance_m: float
    reset_version: str = ORDER9_CANONICAL_RESET_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.reset_version != ORDER9_CANONICAL_RESET_VERSION:
            raise SchemaValidationError("Order9 canonical reset version mismatch")
        require_non_empty(
            self.source_report_path,
            "Order9CanonicalObjectTaskReset.source_report_path",
        )
        _require_sha256(self.source_report_sha256, "source_report_sha256")
        require_non_empty(
            self.source_graph_id,
            "Order9CanonicalObjectTaskReset.source_graph_id",
        )
        _finite_vector(self.robot_root_pose_world, 7, "robot_root_pose_world")
        _finite_vector(self.robot_root_twist_world, 6, "robot_root_twist_world")
        _finite_vector(self.object_pose_world, 7, "object_pose_world")
        _finite_vector(self.object_twist_world, 6, "object_twist_world")
        _finite_scalar_map(self.joint_positions_rad, "joint_positions_rad")
        _finite_scalar_map(self.joint_velocities_radps, "joint_velocities_radps")
        _finite_scalar_map(
            self.open_joint_positions_rad,
            "open_joint_positions_rad",
        )
        expected = set(self.joint_positions_rad)
        if set(self.joint_velocities_radps) != expected:
            raise SchemaValidationError(
                "Order9 reset joint position/velocity ids must match"
            )
        if set(self.open_joint_positions_rad) != expected:
            raise SchemaValidationError(
                "Order9 reset open/q_close joint ids must match"
            )
        for name in ("transport_distance_m", "lift_clearance_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise SchemaValidationError(
                    f"Order9 canonical reset {name} must be positive"
                )
        _require_json_finite(self.metadata, "metadata")

    @property
    def reset_hash(self) -> str:
        self.validate()
        return stable_hash(self.to_dict())


@dataclass
class Order9IsaacStateSnapshot(SchemaBase):
    """Portable state shared by the main rollout and isolated shadow worker."""

    simulation_time_s: float
    robot_root_pose_world: list[float]
    robot_root_twist_world: list[float]
    joint_names: list[str]
    joint_positions_rad: list[float]
    joint_velocities_radps: list[float]
    object_id: str
    object_pose_world: list[float]
    object_twist_world: list[float]
    phase_index: int
    phase_elapsed_s: float
    command_index: int
    snapshot_version: str = ORDER9_ISAAC_SNAPSHOT_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.snapshot_version != ORDER9_ISAAC_SNAPSHOT_VERSION:
            raise SchemaValidationError("Order9 Isaac snapshot version mismatch")
        for name in ("simulation_time_s", "phase_elapsed_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise SchemaValidationError(
                    f"Order9 Isaac snapshot {name} must be non-negative"
                )
        if self.phase_index < 0 or self.command_index < 0:
            raise SchemaValidationError(
                "Order9 Isaac snapshot phase/command indices must be non-negative"
            )
        _finite_vector(self.robot_root_pose_world, 7, "robot_root_pose_world")
        _finite_vector(self.robot_root_twist_world, 6, "robot_root_twist_world")
        _finite_vector(self.object_pose_world, 7, "object_pose_world")
        _finite_vector(self.object_twist_world, 6, "object_twist_world")
        require_non_empty(self.object_id, "Order9IsaacStateSnapshot.object_id")
        if (
            not self.joint_names
            or len(self.joint_names) != len(set(self.joint_names))
            or any(not value for value in self.joint_names)
        ):
            raise SchemaValidationError(
                "Order9 Isaac snapshot joint names must be unique and non-empty"
            )
        if not (
            len(self.joint_names)
            == len(self.joint_positions_rad)
            == len(self.joint_velocities_radps)
        ):
            raise SchemaValidationError(
                "Order9 Isaac snapshot joint arrays must have identical lengths"
            )
        _finite_vector(
            self.joint_positions_rad,
            len(self.joint_names),
            "joint_positions_rad",
        )
        _finite_vector(
            self.joint_velocities_radps,
            len(self.joint_names),
            "joint_velocities_radps",
        )
        _require_json_finite(self.metadata, "metadata")

    @property
    def snapshot_hash(self) -> str:
        self.validate()
        return stable_hash(self.to_dict())


def load_order9_canonical_reset(
    report_path: str | Path,
    *,
    expected_sha256: str,
) -> Order9CanonicalObjectTaskReset:
    """Extract the accepted q_close state without trusting caller aggregates."""

    source = Path(report_path).resolve()
    actual_sha256 = hash_file(source)
    if actual_sha256 != expected_sha256:
        raise SchemaValidationError("Order9 canonical Order8 report hash mismatch")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaValidationError(
            f"failed to load canonical Order8 report: {exc}"
        ) from exc
    if not isinstance(payload, Mapping) or payload.get("passed") is not True:
        raise SchemaValidationError(
            "Order9 reset requires a passing canonical Order8 wrapper report"
        )
    failures = payload.get("report_validation_failures")
    report = payload.get("report")
    if failures != [] or not isinstance(report, Mapping):
        raise SchemaValidationError(
            "Order9 reset rejects unvalidated canonical Order8 evidence"
        )
    monitor_result = report.get("order8_natural_contact_monitor_result")
    if (
        report.get("order8_natural_contact_passed") is not True
        or not isinstance(monitor_result, Mapping)
        or monitor_result.get("passed") is not True
        or monitor_result.get("final_phase") != "complete"
    ):
        raise SchemaValidationError(
            "Order9 reset requires a complete Order8 natural-contact run"
        )
    raw_state = report.get("order8_natural_contact_qclose_checkpoint_state")
    if not isinstance(raw_state, Mapping):
        raise SchemaValidationError("canonical Order8 report has no q_close state")
    roots = raw_state.get("module_root_poses")
    root_velocities = raw_state.get("module_root_velocities")
    base_module_id = _base_module_id(report, roots)
    root_pose = _mapping_vector(roots, base_module_id, 7, "module_root_poses")
    root_twist = _mapping_vector(
        root_velocities,
        base_module_id,
        6,
        "module_root_velocities",
    )
    q_close = _mapping_float_map(
        raw_state.get("joint_positions_rad"),
        "joint_positions_rad",
    )
    qdot = _mapping_float_map(
        raw_state.get("joint_velocities_radps"),
        "joint_velocities_radps",
    )
    q_open = _mapping_float_map(
        report.get("order8_natural_contact_contact_closure_open_joint_positions_rad"),
        "contact_closure_open_joint_positions_rad",
    )
    raw_config = report.get("order8_natural_contact_config")
    if not isinstance(raw_config, Mapping):
        raise SchemaValidationError("canonical Order8 report has no frozen config")
    support_pose_world = _sequence_vector(
        report.get("order8_natural_contact_object_support_pose_world"),
        7,
        "object_support_pose_world",
    )
    support_size_m = _sequence_vector(
        report.get("order8_natural_contact_object_support_size_m"),
        3,
        "object_support_size_m",
    )
    if any(value <= 0.0 for value in support_size_m):
        raise SchemaValidationError(
            "canonical Order8 object support size must be positive"
        )
    result = Order9CanonicalObjectTaskReset(
        source_report_path=str(source),
        source_report_sha256=actual_sha256,
        source_graph_id=str(payload.get("graph_id", "")),
        robot_root_pose_world=root_pose,
        robot_root_twist_world=root_twist,
        joint_positions_rad=q_close,
        joint_velocities_radps=qdot,
        object_pose_world=_sequence_vector(
            raw_state.get("object_pose"), 7, "object_pose"
        ),
        object_twist_world=_sequence_vector(
            raw_state.get("object_twist"), 6, "object_twist"
        ),
        open_joint_positions_rad=q_open,
        transport_distance_m=_finite_number(
            raw_config.get("required_transport_distance_m"),
            "required_transport_distance_m",
        ),
        lift_clearance_m=_finite_number(
            raw_config.get("minimum_lift_clearance_m"),
            "minimum_lift_clearance_m",
        ),
        metadata={
            "source_config_hash": str(payload.get("config_hash", "")),
            "source_graph_hash": str(payload.get("graph_hash", "")),
            "source_env_version": str(payload.get("env_version", "")),
            "base_module_id": int(base_module_id),
            "object_support_pose_world": support_pose_world,
            "object_support_size_m": support_size_m,
            "object_support_method": str(
                report.get("order8_natural_contact_object_support_method", "")
            ),
            "state_labels_reused": False,
            "physical_state_only": True,
        },
    )
    result.validate()
    return result


def order9_snapshot_from_mapping(value: Mapping[str, Any]) -> Order9IsaacStateSnapshot:
    snapshot = Order9IsaacStateSnapshot.from_dict(dict(value))
    snapshot.validate()
    return snapshot


def _base_module_id(report: Mapping[str, Any], roots: object) -> int:
    del report
    if not isinstance(roots, Mapping) or not roots:
        raise SchemaValidationError("canonical Order8 module root state is absent")
    try:
        module_ids = sorted(int(key) for key in roots)
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError(
            "canonical Order8 module root ids are invalid"
        ) from exc
    # The accepted representative graph is rooted at module 0.  Inferring the
    # minimum persisted id avoids trusting an unbound convenience field while
    # retaining compatibility with the report's exact state payload.
    raw = module_ids[0]
    if str(raw) not in roots:
        raise SchemaValidationError("canonical Order8 base root state is absent")
    return raw


def _mapping_vector(
    value: object,
    key: int,
    length: int,
    label: str,
) -> list[float]:
    if not isinstance(value, Mapping):
        raise SchemaValidationError(f"canonical Order8 {label} must be a mapping")
    return _sequence_vector(value.get(str(key)), length, f"{label}[{key}]")


def _mapping_float_map(value: object, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise SchemaValidationError(f"canonical Order8 {label} must be non-empty")
    parsed: dict[str, float] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise SchemaValidationError(f"canonical Order8 {label} keys are invalid")
        parsed[key] = _finite_number(item, f"{label}.{key}")
    return parsed


def _sequence_vector(value: object, length: int, label: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SchemaValidationError(f"canonical Order8 {label} must be a vector")
    result = [_finite_number(item, label) for item in value]
    if len(result) != length:
        raise SchemaValidationError(
            f"canonical Order8 {label} must contain {length} values"
        )
    return result


def _finite_number(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SchemaValidationError(f"{label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise SchemaValidationError(f"{label} must be finite")
    return parsed


def _finite_vector(value: Sequence[float], length: int, label: str) -> None:
    if len(value) != length or any(not math.isfinite(float(item)) for item in value):
        raise SchemaValidationError(f"Order9 {label} must contain {length} finite values")


def _finite_scalar_map(value: Mapping[str, float], label: str) -> None:
    if not value or any(
        not isinstance(key, str)
        or not key
        or not math.isfinite(float(item))
        for key, item in value.items()
    ):
        raise SchemaValidationError(f"Order9 {label} must map ids to finite values")


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise SchemaValidationError(f"Order9 {label} must be a SHA-256 digest")


def _require_json_finite(value: object, label: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SchemaValidationError(f"Order9 {label} contains non-finite data")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SchemaValidationError(f"Order9 {label} keys must be strings")
            _require_json_finite(item, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_json_finite(item, f"{label}[{index}]")
        return
    raise SchemaValidationError(f"Order9 {label} contains non-JSON data")


__all__ = [
    "ORDER9_CANONICAL_RESET_VERSION",
    "ORDER9_ISAAC_SNAPSHOT_VERSION",
    "Order9CanonicalObjectTaskReset",
    "Order9IsaacStateSnapshot",
    "load_order9_canonical_reset",
    "order9_snapshot_from_mapping",
]
