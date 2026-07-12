from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from amsrr.feasibility.morphology_flight import collision_geometry_content_hash
from amsrr.schemas.common import SchemaValidationError
from amsrr.simulation.isaac_lab_backend import load_isaac_lab_backend_config
from amsrr.simulation.order3_rollout_condition import (
    order3_terminal_evidence_start_s,
    order3_tracking_window_start_s,
)
from amsrr.training.order3_pipeline_runner import (
    ORDER3_PIPELINE_RUNNER_VERSION,
    Order3PipelineAcceptanceConfig,
    Order3PipelineMode,
    Order3PipelinePathsConfig,
    Order3PipelineRunner,
    Order3PipelineStage,
    load_order3_pipeline_runner_config,
)
from amsrr.training.random_morphology_takeoff_runner import (
    load_random_morphology_takeoff_runner_config,
)
from amsrr.utils.hashing import hash_file


@pytest.fixture()
def pipeline_runner(tmp_path: Path) -> Order3PipelineRunner:
    artifact_root = tmp_path / "order3"
    config_path = tmp_path / "order3.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "phase": "P4-full-order3",
                "pipeline": {
                    "artifact_root": str(artifact_root),
                    "pool_manifest_path": str(artifact_root / "pool.json"),
                    "bc_dataset_dir": str(artifact_root / "datasets" / "bc"),
                    "ppo_dataset_dir": str(artifact_root / "datasets" / "ppo"),
                    "evaluation_path": str(artifact_root / "evaluation" / "episodes.jsonl"),
                    "takeoff_config_path": "configs/training/order2_5_centroidal_control.yaml",
                    "report_dir": str(artifact_root / "rollouts"),
                    "command_timeout_s": 123.0,
                    "dataset_shard_size": 32,
                },
                "pool": {
                    "master_seed": 9300,
                    "min_modules": 2,
                    "max_modules": 2,
                    "train_per_module_count": 1,
                    "validation_per_module_count": 1,
                    "held_out_per_module_count": 1,
                    "two_module_train_per_module_count": 1,
                    "max_attempts_per_sample": 64,
                    "robot_model_config_path": "configs/robot/robot_model.yaml",
                    "mesh_search_dirs": ["module_urdf"],
                },
                "policy": {"recurrent_hidden_dim": 128},
                "training": {"seed": 3011},
                "curriculum": {
                    "stages": [
                        {
                            "name": "in_air_hover",
                            "floor_takeoff": False,
                            "translation_waypoints": False,
                            "attitude_waypoints": False,
                            "initial_state_randomization_scale": 0.25,
                            "model_randomization_scale": 0.0,
                            "disturbance_scale": 0.0,
                        },
                        {
                            "name": "in_air_waypoints",
                            "floor_takeoff": False,
                            "translation_waypoints": True,
                            "attitude_waypoints": True,
                            "initial_state_randomization_scale": 0.5,
                            "model_randomization_scale": 0.05,
                            "disturbance_scale": 0.0,
                        },
                        {
                            "name": "floor_takeoff_hover",
                            "floor_takeoff": True,
                            "translation_waypoints": False,
                            "attitude_waypoints": False,
                            "initial_state_randomization_scale": 0.0,
                            "model_randomization_scale": 0.1,
                            "disturbance_scale": 1.0,
                        },
                    ]
                },
                "acceptance": {"p4_full_completion_claim": False},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    runner = Order3PipelineRunner.from_config_path(config_path)
    runner.build_pool()
    return runner


def test_order3_pipeline_config_loads_strict_stage_contract(
    pipeline_runner: Order3PipelineRunner,
) -> None:
    config = load_order3_pipeline_runner_config(pipeline_runner.config.config_path)

    assert config.runner_version == ORDER3_PIPELINE_RUNNER_VERSION
    assert config.phase == "P4-full-order3"
    assert config.recurrent_hidden_dim == 128
    assert config.ppo_orchestration_updates == 40
    assert [stage.name for stage in config.curriculum.stages] == [
        "in_air_hover",
        "in_air_waypoints",
        "floor_takeoff_hover",
    ]
    assert config.pipeline.command_timeout_s == 123.0
    assert config.acceptance.p4_full_completion_claim is False
    conditions = pipeline_runner.curriculum_conditions(replicates_per_stage=2)
    assert len(conditions) == 6
    assert len({condition.condition_hash for condition in conditions}) == 6
    assert {condition.task_mode for condition in conditions} == {
        "hover",
        "waypoint",
        "takeoff",
    }
    in_air_conditions = [
        condition for condition in conditions if condition.task_mode != "takeoff"
    ]
    assert all(
        any(abs(value) > 0.0 for value in condition.initial_position_offset_world)
        for condition in in_air_conditions
    )
    assert all(
        any(abs(value) > 0.0 for value in condition.initial_linear_velocity_world)
        for condition in in_air_conditions
    )
    assert all(
        condition.initial_position_offset_world == [0.0, 0.0, 0.0]
        for condition in conditions
        if condition.task_mode == "takeoff"
    )
    evaluation_conditions = pipeline_runner.evaluation_conditions()
    assert len(evaluation_conditions) == 6
    assert {
        (condition.task_mode, condition.stage_id.endswith("randomized"))
        for condition in evaluation_conditions
    } == {
        (task_mode, randomized)
        for task_mode in ("hover", "waypoint", "takeoff")
        for randomized in (False, True)
    }

    # The persisted round-trip is authoritative because nested ``Any``
    # metadata can normalize tuples to lists during JSON serialization.
    persisted = pipeline_runner.build_pool(overwrite=True)
    reloaded = pipeline_runner.load_pool()
    assert persisted.stable_hash() == reloaded.stable_hash()
    assert persisted.to_json() == reloaded.to_json()

    with pytest.raises(SchemaValidationError, match="legacy artifacts/p4_3"):
        Order3PipelinePathsConfig(artifact_root="artifacts/p4_3/order3")
    with pytest.raises(SchemaValidationError, match="implemented acceptance gate"):
        Order3PipelineAcceptanceConfig(held_out_aggregate_success_min=0.5)


def test_order3_full_bc_dry_plan_covers_pool_and_builds_commands(
    pipeline_runner: Order3PipelineRunner,
) -> None:
    plan = pipeline_runner.plan_bc_rollouts(
        mode=Order3PipelineMode.FULL,
        real=False,
    )

    assert plan.stage == Order3PipelineStage.BC_ROLLOUTS
    assert len(plan.commands) == 3
    assert plan.pool_hash == pipeline_runner.load_pool().stable_hash()
    assert plan.full_pool_coverage is True
    assert plan.execution_requested is False
    assert plan.p4_full_completion_claim is False
    for command in plan.commands:
        assert command.argv[0]
        assert command.argv[1].endswith("scripts/random_morphology_takeoff.py")
        assert "--morphology-graph-json-path" in command.argv
        assert "--real" not in command.argv
        assert Path(command.graph_path).is_file()
        staged = command.argv[command.argv.index("--report-path") + 1]
        assert staged != command.report_path
        assert staged.endswith(".order3-staging")


def test_order3_smoke_commands_require_explicit_pool_graph_and_bind_checkpoint(
    pipeline_runner: Order3PipelineRunner,
) -> None:
    full = pipeline_runner.plan_bc_rollouts(
        mode=Order3PipelineMode.FULL,
        real=False,
    )
    graph_path = full.commands[0].graph_path

    with pytest.raises(SchemaValidationError, match="requires at least one explicit"):
        pipeline_runner.plan_bc_rollouts(mode=Order3PipelineMode.SMOKE)

    plan = pipeline_runner.plan_learned_rollouts(
        mode=Order3PipelineMode.SMOKE,
        graph_paths=[graph_path],
        checkpoint_path="/tmp/order3-does-not-exist.pt",
        checkpoint_sha256="a" * 64,
        real=False,
    )

    assert len(plan.commands) == 1
    assert plan.full_pool_coverage is False
    command = plan.commands[0].argv
    assert command[1].endswith("scripts/order3_morphology_pi_l.py")
    assert "learned-rollout-one" in command
    assert command[command.index("--checkpoint-sha256") + 1] == "a" * 64
    assert ("a" * 64) in Path(plan.commands[0].report_path).name
    assert "--stochastic" in command
    assert "--raw-report" not in command
    assert plan.condition_hashes == [plan.commands[0].condition_hash]
    assert "--real" not in command


def test_order3_ppo_conditions_are_fresh_per_graph_and_update(
    pipeline_runner: Order3PipelineRunner,
) -> None:
    base_condition = pipeline_runner.curriculum_conditions(
        stage_ids=["in_air_hover"]
    )[0]
    first = pipeline_runner.plan_learned_rollouts(
        mode=Order3PipelineMode.FULL,
        checkpoint_path="/tmp/order3-seeded-plan.pt",
        checkpoint_sha256="d" * 64,
        real=False,
        conditions=[base_condition],
        condition_seed_namespace="ppo_update_0000",
    )
    second = pipeline_runner.plan_learned_rollouts(
        mode=Order3PipelineMode.FULL,
        checkpoint_path="/tmp/order3-seeded-plan.pt",
        checkpoint_sha256="d" * 64,
        real=False,
        conditions=[base_condition],
        condition_seed_namespace="ppo_update_0001",
    )

    assert len(first.condition_hashes) == len(first.commands) == 3
    assert set(first.condition_hashes).isdisjoint(second.condition_hashes)
    first_conditions = [
        command.argv[command.argv.index("--rollout-condition-json") + 1]
        for command in first.commands
    ]
    assert len(set(first_conditions)) == 3


def test_order3_dry_plan_is_atomic_and_never_executes(
    pipeline_runner: Order3PipelineRunner,
    tmp_path: Path,
) -> None:
    def forbidden_executor(argv, timeout_s):  # pragma: no cover - failure assertion.
        raise AssertionError((argv, timeout_s))

    pipeline_runner.command_executor = forbidden_executor
    plan = pipeline_runner.plan_bc_rollouts(
        mode=Order3PipelineMode.FULL,
        real=False,
    )
    plan_path = tmp_path / "plans" / "bc.json"

    result = pipeline_runner.execute_plan(plan, plan_path=plan_path)

    assert result == {
        "executed": False,
        "command_count": 3,
        "report_hashes": {},
        "p4_full_completion_claim": False,
    }
    persisted = json.loads(plan_path.read_text(encoding="utf-8"))
    assert persisted["execution_requested"] is False
    assert persisted["full_pool_coverage"] is True


def test_order3_evaluation_builds_paired_condition_bound_dry_commands(
    pipeline_runner: Order3PipelineRunner,
) -> None:
    full = pipeline_runner.plan_bc_rollouts(
        mode=Order3PipelineMode.FULL,
        real=False,
    )
    graph_path = full.commands[0].graph_path
    condition = pipeline_runner.evaluation_conditions()[0]

    learned = pipeline_runner.plan_learned_evaluation_rollouts(
        mode=Order3PipelineMode.SMOKE,
        graph_paths=[graph_path],
        checkpoint_path="/tmp/order3-evaluation.pt",
        checkpoint_sha256="b" * 64,
        conditions=[condition],
        real=False,
    )
    baseline = pipeline_runner.plan_baseline_evaluation_rollouts(
        mode=Order3PipelineMode.SMOKE,
        graph_paths=[graph_path],
        conditions=[condition],
        real=False,
    )

    assert learned.stage == Order3PipelineStage.EVALUATE_LEARNED
    assert baseline.stage == Order3PipelineStage.EVALUATE_BASELINE
    assert learned.condition_hashes == baseline.condition_hashes == [
        condition.condition_hash
    ]
    assert "--stochastic" not in learned.commands[0].argv
    assert "--raw-report" in learned.commands[0].argv
    assert "--raw-report" in baseline.commands[0].argv
    assert "--rollout-condition-json" in learned.commands[0].argv
    assert "baseline-rollout-one" in baseline.commands[0].argv
    assert ("b" * 64) in learned.commands[0].report_path
    assert ("b" * 64) not in baseline.commands[0].report_path


def test_order3_learned_evaluation_propagates_gui_options(
    pipeline_runner: Order3PipelineRunner,
    tmp_path: Path,
) -> None:
    graph_path = pipeline_runner.plan_bc_rollouts(
        mode=Order3PipelineMode.FULL,
        real=False,
    ).commands[0].graph_path
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"order3-gui-checkpoint")
    condition = pipeline_runner.evaluation_conditions()[0]

    plan = pipeline_runner.plan_learned_evaluation_rollouts(
        mode=Order3PipelineMode.SMOKE,
        graph_paths=[graph_path],
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=hash_file(checkpoint_path),
        conditions=[condition],
        real=True,
        viewer="kit",
        realtime_playback=True,
        keep_open_after_rollout_s=8.0,
    )

    command = plan.commands[0].argv
    assert command[command.index("--viewer") + 1] == "kit"
    assert "--realtime-playback" in command
    assert command[command.index("--keep-open-after-rollout-s") + 1] == "8.0"
    assert "--real" in command

    with pytest.raises(SchemaValidationError, match="requires real Isaac"):
        pipeline_runner.plan_learned_evaluation_rollouts(
            mode=Order3PipelineMode.SMOKE,
            graph_paths=[graph_path],
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=hash_file(checkpoint_path),
            conditions=[condition],
            real=False,
            viewer="kit",
        )


def test_order3_evaluation_episode_builder_pairs_raw_reports(
    pipeline_runner: Order3PipelineRunner,
    tmp_path: Path,
) -> None:
    entry = pipeline_runner.load_pool().entries[0]
    condition = next(
        value
        for value in pipeline_runner.evaluation_conditions()
        if value.stage_id == "evaluation_hover_nominal"
    )
    checkpoint_sha256 = "c" * 64
    terminal_metrics = {
        "position_error_m": 0.0,
        "attitude_error_rad": 0.0,
        "linear_velocity_error_mps": 0.0,
        "angular_velocity_error_rad_s": 0.0,
        "within_tolerance_duration_s": 1.0,
        "takeoff_height_gain_ratio": None,
    }
    _, takeoff_config = load_random_morphology_takeoff_runner_config(
        pipeline_runner.config.pipeline.takeoff_config_path
    )
    physical_model = pipeline_runner.physical_model
    backend_hash = load_isaac_lab_backend_config(
        takeoff_config.backend_config_path
    ).stable_hash()
    collision_hash = collision_geometry_content_hash(
        physical_model,
        mesh_search_dirs=takeoff_config.mesh_search_dirs,
    )

    def write_report(path: Path, *, learned: bool, cost: float) -> None:
        path.write_text(
            json.dumps(
                {
                    "isaac_backed": True,
                    "order3_report_validation_failures": [],
                    "order3_task_mode": condition.task_mode,
                    "order3_rollout_task_mode": condition.task_mode,
                    "order3_structural_hash": entry.structural_hash,
                    "random_morphology_takeoff_module_count": entry.module_count,
                    "order3_rollout_condition": condition.to_dict(),
                    "order3_rollout_condition_hash": condition.condition_hash,
                    "order3_rollout_seed_applied": {
                        "seed": condition.seed,
                        "python_random": True,
                        "torch": True,
                    },
                    "order3_privileged_external_wrench_body": list(
                        condition.external_wrench_body
                    ),
                    "order3_disturbance_start_s": condition.disturbance_start_s,
                    "order3_disturbance_duration_s": condition.disturbance_duration_s,
                    "order3_condition_realization": {
                        "condition_hash": condition.condition_hash,
                        "task_mode": condition.task_mode,
                        "requested_initial_root_pose_world": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                        "applied_initial_root_pose_world": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                        "requested_initial_twist_world": [0.0] * 6,
                        "applied_initial_twist_world": [0.0] * 6,
                        "requested_mass_scale": condition.mass_scale,
                        "applied_mass_scale": condition.mass_scale,
                        "requested_inertia_scale": condition.inertia_scale,
                        "applied_inertia_scale": condition.inertia_scale,
                        "requested_thrust_scale": condition.thrust_scale,
                        "applied_thrust_scale": condition.thrust_scale,
                        "mass_randomization_applied": True,
                        "inertia_randomization_applied": True,
                        "thrust_randomization_applied": True,
                        "initial_state_applied": True,
                        "final_target_pose_world": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                        "final_target_twist_world": [0.0] * 6,
                    },
                    "order3_terminal_evidence_start_s": (
                        order3_terminal_evidence_start_s(condition)
                    ),
                    "order3_terminal_evidence_completed": True,
                    "order3_tracking_window_start_s": (
                        order3_tracking_window_start_s(condition)
                    ),
                    "order3_tracking_window_end_s": 1.0,
                    "order3_tracking_window_sample_count": 10,
                    "random_morphology_takeoff_backend_config_hash": backend_hash,
                    "random_morphology_takeoff_physical_model_hash": (
                        physical_model.stable_hash()
                    ),
                    "random_morphology_takeoff_collision_geometry_hash": collision_hash,
                    "order3_terminal_metrics": terminal_metrics,
                    "order3_free_flight_terminal_metrics": terminal_metrics,
                    "order3_free_flight_tracking_cost": cost,
                    "order3_free_flight_success": True,
                    "order3_qp_infeasible": False,
                    "order3_hard_collision": False,
                    "order3_non_finite_state": False,
                    "order3_unsupported_actuator": False,
                    "order3_pi_l_rollout": learned,
                    "order3_pi_l_checkpoint_sha256": (
                        checkpoint_sha256 if learned else None
                    ),
                    "order3_deterministic_baseline_rollout": not learned,
                    "order3_fallback_used": False,
                    "order3_fallback_reason": None,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    learned_path = tmp_path / "learned.json"
    baseline_path = tmp_path / "baseline.json"
    write_report(learned_path, learned=True, cost=0.8)
    write_report(baseline_path, learned=False, cost=1.0)

    episodes, output_path = pipeline_runner.build_evaluation_episodes(
        mode=Order3PipelineMode.SMOKE,
        learned_report_paths=[learned_path],
        baseline_report_paths=[baseline_path],
        checkpoint_sha256=checkpoint_sha256,
        output_path=tmp_path / "episodes.json",
    )

    assert Path(output_path).is_file()
    assert len(episodes) == 1
    assert episodes[0].condition_hash == condition.condition_hash
    assert episodes[0].learned_report_sha256 is not None
    assert episodes[0].deterministic_baseline_report_sha256 is not None
    assert episodes[0].isaac_backed is True


def test_order3_smoke_mode_cannot_invoke_acceptance(
    pipeline_runner: Order3PipelineRunner,
) -> None:
    with pytest.raises(SchemaValidationError, match="smoke mode"):
        pipeline_runner.evaluate_acceptance(
            mode=Order3PipelineMode.SMOKE,
            dataset_manifest_path="missing-dataset.json",
            checkpoint_path="missing-checkpoint.pt",
            checkpoint_sha256="a" * 64,
            episodes_path="missing-episodes.json",
            artifact_metadata_path="missing-metadata.json",
        )
