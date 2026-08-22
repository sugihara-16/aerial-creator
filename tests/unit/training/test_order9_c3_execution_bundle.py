from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.training.order9_c3_execution_bundle import (
    Order9C3ExecutionBucketBinding,
    Order9C3ExecutionBundle,
    _validate_nominal_semantics,
    apply_order9_c3_execution_bundle_to_args,
)


def _required_semantics() -> dict[str, object]:
    return {
        "phase_sequence": ["release", "retreat", "settle"],
        "rigid_robot_target_translation": {
            "repair_version": "translation_v1",
            "offset_world_m": [0.0, 0.0, 0.0],
            "phase_z_offset_start_end_m": {
                "retreat": [0.3, 0.3],
                "settle": [0.3, 0.3],
            },
            "phase_smooth_z_offset_start_end_m": {"release": [0.0, 0.3]},
            "smooth_offset_boundary_velocity_zero": True,
            "joint_targets_changed": False,
            "object_targets_changed": False,
            "contact_assignments_changed": False,
            "runtime_planner_enabled": False,
        },
        "generation_method_tokens_by_phase": {
            "release": ["translation_v1", "smooth_v1"],
            "retreat": ["translation_v1"],
            "settle": ["translation_v1"],
        },
    }


def _artifact() -> SimpleNamespace:
    required = _required_semantics()
    return SimpleNamespace(
        phase_trajectories=[
            SimpleNamespace(
                phase="release", generation_method="base+translation_v1+smooth_v1"
            ),
            SimpleNamespace(phase="retreat", generation_method="base+translation_v1"),
            SimpleNamespace(phase="settle", generation_method="base+translation_v1"),
        ],
        selection_evidence={
            "rigid_robot_target_translation": required[
                "rigid_robot_target_translation"
            ]
        },
        final_phase_target_reached=True,
    )


def test_semantic_gate_accepts_complete_release_clearance() -> None:
    _validate_nominal_semantics(
        _artifact(), required=_required_semantics(), bucket_id="validation-0"
    )


def test_semantic_gate_rejects_silent_release_clearance_loss() -> None:
    artifact = _artifact()
    artifact.selection_evidence = {}
    with pytest.raises(SchemaValidationError, match="lacks required robot translation"):
        _validate_nominal_semantics(
            artifact, required=_required_semantics(), bucket_id="validation-0"
        )


def test_bundle_hydrates_bucket_specific_runtime_inputs(tmp_path: Path) -> None:
    task = tmp_path / "task.json"
    task.write_text(
        '{"metadata":{"order9_rollout_bucket_id":"validation-0"}}',
        encoding="utf-8",
    )
    reset = tmp_path / "reset.pt"
    reset.write_bytes(b"reset")
    binding = Order9C3ExecutionBucketBinding(
        bucket_id="validation-0",
        module_count=2,
        artifact_sha256="a" * 64,
        reset_bank_path=reset,
        reset_bank_sha256="b" * 64,
        collision_validation_path=tmp_path / "collision.json",
        collision_validation_sha256="c" * 64,
    )
    bundle = Order9C3ExecutionBundle(
        source_path=tmp_path / "bundle.json",
        source_sha256="d" * 64,
        config_path=tmp_path / "config.yaml",
        checkpoint_path=tmp_path / "checkpoint.pt",
        checkpoint_sha256="e" * 64,
        bucket_manifest_path=tmp_path / "buckets.json",
        bucket_manifest_sha256="f" * 64,
        nominal_set_manifest_path=tmp_path / "nominal.json",
        nominal_set_manifest_sha256="1" * 64,
        action_contract="contact_space_projected_policy_command",
        bucket_bindings={binding.bucket_id: binding},
        payload={},
    )
    args = SimpleNamespace(
        config=None,
        pi_l_checkpoint=None,
        pi_l_checkpoint_sha256=None,
        c3_nominal_set_manifest=None,
        c3_nominal_set_sha256=None,
        c3_nominal_artifact_sha256=None,
        c3_reset_bank=None,
        c3_action_contract=None,
        task_spec_json=str(task),
    )
    apply_order9_c3_execution_bundle_to_args(args, bundle)
    assert args.pi_l_checkpoint == str(bundle.checkpoint_path)
    assert args.c3_nominal_artifact_sha256 == "a" * 64
    assert args.c3_reset_bank == str(reset)
