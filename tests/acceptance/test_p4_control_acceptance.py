from __future__ import annotations

from pathlib import Path

from amsrr.acceptance import P4ControlSmokeResult, run_p4_control_acceptance
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training import (
    P4_0FullPipelineRunner,
    P4_0FullPipelineRunnerConfig,
    load_p4_0_full_pipeline_runner_config,
)


def _p4_control_archive(grasp_carry_dict: dict, tmp_path: Path):
    base_task = TaskSpec.from_dict(grasp_carry_dict)
    _, distribution_config, policy_config, env_config = load_p4_0_full_pipeline_runner_config(
        "configs/training/p4_0_grasp_carry.yaml"
    )
    runner = P4_0FullPipelineRunner(
        base_task,
        runner_config=P4_0FullPipelineRunnerConfig(
            episode_count=1,
            seed=1400,
            source_hash="p4-control-acceptance-test",
        ),
        distribution_config=distribution_config,
        policy_config=policy_config,
        env_config=env_config,
    )
    archive = runner.run(archive_path=tmp_path / "p4_control_source.jsonl").archives[0]
    archive.rollout_artifacts = {
        "phase": "P4-control",
        "backend": "isaac_lab",
        "is_p4_full_completion": False,
        "isaac_backed": True,
        "physical_success_claim": False,
        "note": "low-level flight validation only",
    }
    archive.metrics["p4_full_completion"] = 0.0
    archive.metrics["isaac_backed"] = 1.0
    archive.actuator_target_records = [
        {
            "time_s": 0.0,
            "backend": "isaac_lab",
            "morphology_graph_id": "p4-control-test",
            "command_index": 0,
            "actuator_targets": [],
            "clipped_targets": [],
            "missing_actuators": [],
            "unsupported_actuators": [],
            "allocation_residual_norm": 0.0,
            "qp_status": "ok",
            "metrics": {
                "allocation_residual_norm": 0.0,
                "clipped_target_count": 0.0,
                "missing_actuator_count": 0.0,
                "unsupported_actuator_count": 0.0,
            },
        }
    ]
    return archive


def test_p4_control_fast_gate_does_not_complete_without_real_isaac_smoke(
    grasp_carry_dict: dict,
    tmp_path: Path,
) -> None:
    archive = _p4_control_archive(grasp_carry_dict, tmp_path)

    report = run_p4_control_acceptance([archive])

    assert report.fast_gate_passed is True
    assert report.real_isaac_smoke_passed is False
    assert report.completion_passed is False
    assert "P4-control real Isaac smoke gate has not passed" in report.failure_reasons
    assert report.metrics["completion_passed"] == 0.0


def test_p4_control_completion_requires_all_real_isaac_smokes(
    grasp_carry_dict: dict,
    tmp_path: Path,
) -> None:
    archive = _p4_control_archive(grasp_carry_dict, tmp_path)
    smoke_results = [
        P4ControlSmokeResult("single_module_hover", attempted=True, passed=True, isaac_backed=True),
        P4ControlSmokeResult("fixed_morphology_hover", attempted=True, passed=True, isaac_backed=True),
        P4ControlSmokeResult("fixed_morphology_waypoint", attempted=True, passed=True, isaac_backed=True),
    ]

    report = run_p4_control_acceptance([archive], smoke_results=smoke_results)

    assert report.fast_gate_passed is True
    assert report.real_isaac_smoke_passed is True
    assert report.completion_passed is True
    assert report.failure_reasons == []
    assert report.passed_smoke_names == [
        "fixed_morphology_hover",
        "fixed_morphology_waypoint",
        "single_module_hover",
    ]
    assert type(report).from_json(report.to_json()).to_dict() == report.to_dict()
