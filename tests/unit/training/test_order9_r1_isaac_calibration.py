from __future__ import annotations

from pathlib import Path

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.order9 import Order9ArtifactBinding
from amsrr.training.order9_r1_calibration_runner import (
    ORDER9_R1_C3_CHECKPOINT_SHA256,
)
from amsrr.training.order9_r1_isaac_calibration import (
    ORDER9_R1_C3_RELEASE_LEDGER_SHA256,
    ORDER9_R1_C3_ROLLOUT_IMPLEMENTATION_SHA256,
    ORDER9_R1_EXECUTION_SPLIT_ALIAS_CONTRACT,
    ORDER9_R1_ISAAC_CASE_VERSION,
    MaterializedOrder9R1IsaacCase,
    Order9R1IsaacCaseManifest,
    order9_r1_isaac_case_command,
)


def _binding(kind: str, digest: str = "a" * 64) -> Order9ArtifactBinding:
    return Order9ArtifactBinding(
        artifact_kind=kind,
        path=f"evidence/{kind}",
        sha256=digest,
    )


def _manifest() -> Order9R1IsaacCaseManifest:
    return Order9R1IsaacCaseManifest(
        manifest_version=ORDER9_R1_ISAAC_CASE_VERSION,
        candidate_id="r1_l1__train__bucket__lattice_00",
        level_id="r1_l1_10mm_5deg",
        split="train",
        execution_split="validation",
        execution_split_alias_contract=ORDER9_R1_EXECUTION_SPLIT_ALIAS_CONTRACT,
        source_bucket_id="train-000000-example",
        module_count=2,
        morphology_hash="b" * 64,
        structural_hash="c" * 64,
        seed=7,
        replay_count=2,
        task_spec=_binding("r1_calibration_task_spec"),
        morphology_graph=_binding("r1_calibration_morphology_graph"),
        robot_usd=_binding("r1_calibration_robot_usd"),
        nominal_set=_binding("r1_calibration_nominal_set"),
        nominal_artifact=_binding("r1_calibration_nominal_artifact"),
        fast_screen_evidence=_binding("fast_kinematic_screen"),
        approved_protocol=_binding("approved_calibration_protocol"),
        c3_checkpoint=_binding(
            "c3_promoted_pi_l_checkpoint", ORDER9_R1_C3_CHECKPOINT_SHA256
        ),
        c3_rollout_implementation=_binding(
            "c3_protected_rollout_implementation",
            ORDER9_R1_C3_ROLLOUT_IMPLEMENTATION_SHA256,
        ),
        c3_release_ledger=_binding(
            "c3_promoted_release_ledger", ORDER9_R1_C3_RELEASE_LEDGER_SHA256
        ),
        r1_implementations=[
            _binding("r1_calibration_protocol_implementation"),
            _binding("r1_calibration_enumerator_implementation"),
            _binding("r1_randomization_implementation"),
            _binding("r1_fast_screen_implementation"),
            _binding("r1_isaac_wrapper_implementation"),
        ],
        c3_promotion_evidence_eligible=False,
        training_eligible=False,
        reset_bank_path="evidence/reset_bank.pt",
        selected_gripper_friction=4.5,
        contact_stiffness_n_per_m=7800.0,
        contact_damping_n_s_per_m=75.0,
        estimated_mass_kg=1.0,
        estimated_inertia_body=(1.0, 0.0, 0.0, 1.0, 0.0, 1.0),
        estimated_com_object=(0.0, 0.0, 0.0),
    )


def test_case_command_uses_protected_c3_validation_execution_alias(tmp_path) -> None:
    manifest = _manifest()
    manifest.validate()
    materialized = MaterializedOrder9R1IsaacCase(
        manifest_path=tmp_path / "case" / "case_manifest.json",
        manifest=manifest,
    )

    command, _log = order9_r1_isaac_case_command(
        materialized,
        repository_root=tmp_path,
        python_executable="python",
    )

    assert command[command.index("--split") + 1] == "validation"
    assert command[command.index("--generation-id") + 1].startswith("r1_calibration:")
    assert "--r1-calibration-evaluation" not in command
    assert "--formal-phase-zero-start" in command


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("execution_split", "train"),
        ("c3_promotion_evidence_eligible", True),
        ("training_eligible", True),
    ),
)
def test_case_manifest_rejects_split_or_authority_leak(field, value) -> None:
    manifest = _manifest()
    setattr(manifest, field, value)

    with pytest.raises(SchemaValidationError):
        manifest.validate()
