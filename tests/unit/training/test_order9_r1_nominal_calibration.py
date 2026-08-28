from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from amsrr.schemas.common import SchemaValidationError
from amsrr.training import order9_r1_nominal_calibration as nominal

REPOSITORY = Path(__file__).resolve().parents[3]


def test_r1_nominal_contract_binds_approved_overlay() -> None:
    protocol, approval = nominal.load_order9_r1_nominal_calibration_contract(
        REPOSITORY
    )

    assert protocol["pi_l_system_retained"] is True
    assert protocol["pi_l_actor_command_applied"] is False
    assert (
        protocol["execution_contract"]
        == nominal.ORDER9_R1_NOMINAL_EXECUTION_CONTRACT
    )
    assert approval["decision"] == "approved"


def test_r1_nominal_command_uses_protected_bypass(
    monkeypatch, tmp_path
) -> None:
    materialized = SimpleNamespace(
        manifest=SimpleNamespace(candidate_id="candidate-1"),
        manifest_path=tmp_path
        / "selection/level/candidate-1/case_manifest.json",
    )

    def source_command(*_args, **_kwargs):
        return [
            "python",
            "rollout.py",
            "--generation-id",
            "old",
        ], tmp_path / "log"

    monkeypatch.setattr(
        nominal,
        "order9_r1_isaac_case_command",
        source_command,
    )

    command, _log = nominal.order9_r1_nominal_case_command(
        materialized,
        repository_root=tmp_path,
        python_executable="python",
    )

    assert command[command.index("--generation-id") + 1] == (
        "r1_nominal_calibration:candidate-1"
    )
    assert command.count("--diagnostic-nominal-qpid-only") == 1


def test_r1_nominal_zero_action_gate_rejects_policy_command() -> None:
    tensors = {
        name: torch.zeros((2, 3), dtype=torch.float32)
        for name in nominal._ZERO_ACTION_TENSORS
    }
    nominal._require_zero_policy_actions(tensors)

    tensors["joint_action"][0, 0] = 0.001
    with pytest.raises(SchemaValidationError, match="joint_action"):
        nominal._require_zero_policy_actions(tensors)


def test_r1_nominal_replay_contract_forbids_applied_pi_l() -> None:
    result = nominal.Order9R1NominalReplayResult(
        candidate_id="candidate-1",
        replay_index=0,
        success=True,
        safety_failure=False,
        fallback_used=False,
        failure_reason=None,
    )
    result.validate()

    with pytest.raises(SchemaValidationError, match="contract"):
        nominal.Order9R1NominalReplayResult(
            candidate_id="candidate-1",
            replay_index=0,
            success=True,
            safety_failure=False,
            fallback_used=False,
            failure_reason=None,
            pi_l_actor_command_applied=True,
        ).validate()
