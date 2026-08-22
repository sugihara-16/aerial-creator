from __future__ import annotations

import json
from pathlib import Path

import pytest

import amsrr.training.order9_pi_l_stage_runner as stage_runner_module
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.datasets import DatasetSplit
from amsrr.training.order9_curriculum import load_order9_learning_config
from amsrr.training.order9_pi_l_stage_runner import (
    ORDER9_ROLLOUT_RESULT_PREFIX,
    allocate_order9_topology_shard_environments,
    allocate_order9_topology_stratified_environments,
    coalesce_order9_same_morphology_collectors,
    load_order9_rollout_result,
    order9_c3_state_inheritance_environment_steps,
    order9_c3_state_inheritance_phase_indices,
    order9_c3_training_wrench_gate_range_scale,
    order9_pi_l_collector_command,
    required_order9_update_count,
    resolve_order9_extended_update_budget,
    run_parallel_order9_collectors,
    select_order9_c3_state_inheritance_buckets,
    select_order9_pi_l_rollout_buckets,
    select_order9_pi_l_topology_stratified_buckets,
    validate_order9_c3_generation_boundary_coverage,
)
from amsrr.training.order9_rollout_buckets import (
    Order9PiLRolloutBucket,
    Order9PiLRolloutBucketManifest,
)


def _bucket(
    split: DatasetSplit, sample_index: int, *, module_count: int = 3
) -> Order9PiLRolloutBucket:
    return Order9PiLRolloutBucket(
        bucket_id=f"{split.value}-{sample_index}",
        split=split,
        seed=100 + sample_index,
        sample_index=sample_index,
        task_id=f"task-{split.value}-{sample_index}",
        task_spec_path=f"task-{sample_index}.json",
        task_spec_sha256="a" * 64,
        morphology_graph_path=f"graph-{sample_index}.json",
        morphology_graph_sha256="b" * 64,
        morphology_hash="c" * 64,
        structural_hash="d" * 64,
        module_count=module_count,
        robot_usd_path="robot.usda",
        robot_usd_sha256="e" * 64,
        selected_gripper_friction=4.5,
        contact_stiffness_n_per_m=7500.0,
        contact_damping_n_s_per_m=75.0,
        estimated_mass_kg=1.0,
        estimated_inertia_body=[0.01, 0.0, 0.0, 0.02, 0.0, 0.03],
        estimated_com_object=[0.0, 0.0, 0.0],
        randomization_version="unit",
        topology_source="unit",
    )


def _manifest() -> Order9PiLRolloutBucketManifest:
    config = load_order9_learning_config()
    return Order9PiLRolloutBucketManifest(
        stage_id="c2_pi_l_ppo_fixed_conservative",
        stage_config_hash="1" * 64,
        curriculum_schedule_hash="2" * 64,
        config_hash="3" * 64,
        physical_model_hash="4" * 64,
        topology_randomized=False,
        buckets=[
            _bucket(DatasetSplit.TRAIN, 2),
            _bucket(DatasetSplit.VALIDATION, 9),
            _bucket(DatasetSplit.TRAIN, 0),
            _bucket(DatasetSplit.VALIDATION, 8),
            _bucket(DatasetSplit.TRAIN, 1),
        ],
        metadata={"config_version": config.curriculum.schedule_version},
    )


def test_required_update_count_covers_c2_target() -> None:
    assert required_order9_update_count(3_000_000, 65_536) == 46
    assert 45 * 65_536 < 3_000_000 <= 46 * 65_536
    with pytest.raises(ValueError, match="positive"):
        required_order9_update_count(0, 65_536)


def test_parallel_collectors_stagger_concurrent_process_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[str] = []
    sleeps: list[float] = []

    class _Process:
        def __init__(self, command: list[str], **_: object) -> None:
            started.append(command[0])

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(stage_runner_module.subprocess, "Popen", _Process)
    monkeypatch.setattr(stage_runner_module.time, "sleep", sleeps.append)

    elapsed_s, return_codes = run_parallel_order9_collectors(
        {"train": ["train"], "validation": ["validation"]},
        repository_root=tmp_path,
        log_paths={"train": "train.log", "validation": "validation.log"},
        maximum_parallel_process_count=2,
        process_start_stagger_s=2.5,
    )

    assert elapsed_s >= 0.0
    assert started == ["train", "validation"]
    assert sleeps == [2.5]
    assert return_codes == {"train": 0, "validation": 0}


def test_same_morphology_collectors_share_one_persistent_process(
    tmp_path: Path,
) -> None:
    script = tmp_path / "order9_vectorized_isaac_rollout.py"
    commands = {
        "bucket-a": [
            "taskset",
            "-c",
            "0-7",
            "python",
            str(script),
            "--generation-id",
            "a",
        ],
        "bucket-b": [
            "taskset",
            "-c",
            "0-7",
            "python",
            str(script),
            "--generation-id",
            "b",
        ],
        "bucket-c": [
            "taskset",
            "-c",
            "0-7",
            "python",
            str(script),
            "--generation-id",
            "c",
        ],
    }
    logs = {name: tmp_path / name / "evaluation.log" for name in commands}

    grouped, grouped_logs, members = coalesce_order9_same_morphology_collectors(
        commands,
        log_paths=logs,
        morphology_hash_by_name={
            "bucket-a": "a" * 64,
            "bucket-b": "a" * 64,
            "bucket-c": "c" * 64,
        },
        batch_root=tmp_path / "batches",
    )

    batch_name = "morphology_" + "a" * 16
    assert set(grouped) == {batch_name, "bucket-c"}
    assert members == {
        batch_name: ("bucket-a", "bucket-b"),
        "bucket-c": ("bucket-c",),
    }
    assert grouped["bucket-c"] == commands["bucket-c"]
    assert grouped_logs["bucket-c"] == logs["bucket-c"].resolve()
    manifest_path = Path(grouped[batch_name][-1])
    assert grouped[batch_name][-2] == "--persistent-bucket-jobs"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["morphology_hash"] == "a" * 64
    assert [job["name"] for job in payload["jobs"]] == ["bucket-a", "bucket-b"]
    assert payload["jobs"][0]["argv"] == ["--generation-id", "a"]
    assert payload["jobs"][1]["log_path"] == str(logs["bucket-b"].resolve())


@pytest.mark.parametrize("stagger_s", [-1.0, float("nan"), float("inf")])
def test_parallel_collectors_reject_invalid_start_stagger(
    tmp_path: Path, stagger_s: float
) -> None:
    with pytest.raises(ValueError, match="stagger"):
        run_parallel_order9_collectors(
            {"train": ["train"], "validation": ["validation"]},
            repository_root=tmp_path,
            log_paths={"train": "train.log", "validation": "validation.log"},
            maximum_parallel_process_count=2,
            process_start_stagger_s=stagger_s,
        )


def test_extended_update_budget_preserves_configured_stage_target() -> None:
    assert resolve_order9_extended_update_budget(
        3_000_000,
        65_536,
        0,
    ) == (46, 46, 3_014_656)
    assert resolve_order9_extended_update_budget(
        3_000_000,
        65_536,
        4,
    ) == (46, 50, 3_276_800)
    with pytest.raises(ValueError, match="non-negative"):
        resolve_order9_extended_update_budget(3_000_000, 65_536, -1)


def test_bucket_selection_rotates_each_split_independently() -> None:
    manifest = _manifest()
    selected = [select_order9_pi_l_rollout_buckets(manifest, index) for index in range(4)]
    assert [train.sample_index for train, _ in selected] == [0, 1, 2, 0]
    assert [validation.sample_index for _, validation in selected] == [8, 9, 8, 9]
    with pytest.raises(ValueError, match="non-negative"):
        select_order9_pi_l_rollout_buckets(manifest, -1)


def test_topology_stratified_selection_covers_every_module_count() -> None:
    manifest = _manifest()
    manifest.buckets = [
        *[
            _bucket(DatasetSplit.TRAIN, module_count, module_count=module_count)
            for module_count in range(2, 9)
        ],
        *[
            _bucket(
                DatasetSplit.TRAIN,
                module_count + 10,
                module_count=module_count,
            )
            for module_count in range(2, 9)
        ],
        _bucket(DatasetSplit.VALIDATION, 30, module_count=2),
    ]

    train0, validation0 = select_order9_pi_l_topology_stratified_buckets(
        manifest, 0
    )
    train1, _ = select_order9_pi_l_topology_stratified_buckets(manifest, 1)

    assert [bucket.module_count for bucket in train0] == list(range(2, 9))
    assert [bucket.sample_index for bucket in train0] == list(range(2, 9))
    assert [bucket.sample_index for bucket in train1] == list(range(12, 19))
    assert validation0.sample_index == 30


def test_topology_environment_allocation_preserves_1024_budget() -> None:
    allocation = allocate_order9_topology_stratified_environments(
        1024, list(range(2, 9)), update_index=0
    )
    assert allocation == {2: 147, 3: 147, 4: 146, 5: 146, 6: 146, 7: 146, 8: 146}
    assert sum(allocation.values()) == 1024


def test_c3_state_inheritance_budget_and_phase_seeds_are_deterministic() -> None:
    config = load_order9_learning_config()
    assert order9_c3_state_inheritance_environment_steps(
        config, minimum_module_count=2, maximum_module_count=8
    ) == 172_032
    assert order9_c3_state_inheritance_environment_steps(
        config, minimum_module_count=2, maximum_module_count=5
    ) == 98_304
    assert order9_c3_state_inheritance_phase_indices(
        range(6), [0] * 6, [1, 2, 3]
    ) == (1, 2, 3, 1, 2, 3)
    assert order9_c3_state_inheritance_phase_indices(
        [1, 4], [1, 3], [1, 2, 3]
    ) == (3, 2)


def test_state_inheritance_selector_covers_focused_topologies() -> None:
    manifest = _manifest()
    manifest.buckets = [
        *[
            _bucket(
                DatasetSplit.TRAIN,
                module_count + replica * 10,
                module_count=module_count,
            )
            for module_count in range(2, 6)
            for replica in range(4)
        ],
        _bucket(DatasetSplit.VALIDATION, 30, module_count=2),
    ]

    selected = select_order9_c3_state_inheritance_buckets(
        manifest,
        2,
        minimum_module_count=2,
        maximum_module_count=5,
        topology_count_by_module={2: 0, 3: 0, 4: 0, 5: 4},
    )

    assert [bucket.module_count for bucket in selected] == [5, 5, 5, 5]
    assert len({bucket.bucket_id for bucket in selected if bucket.module_count == 5}) == 4

    with pytest.raises(ValueError, match="topology counts"):
        select_order9_c3_state_inheritance_buckets(
            manifest,
            0,
            minimum_module_count=2,
            maximum_module_count=5,
            topology_count_by_module={2: 0, 3: 0, 4: 0, 5: 0},
        )


def test_topology_selector_can_sample_two_structures_per_module_count() -> None:
    manifest = _manifest()
    manifest.buckets = [
        *[
            _bucket(
                DatasetSplit.TRAIN,
                module_count + replica * 10,
                module_count=module_count,
            )
            for module_count in range(2, 9)
            for replica in range(2)
        ],
        _bucket(DatasetSplit.VALIDATION, 30, module_count=2),
    ]

    train, _ = select_order9_pi_l_topology_stratified_buckets(
        manifest, 0, topologies_per_module_count=2
    )
    allocation = allocate_order9_topology_shard_environments(
        1024, len(train), update_index=0
    )

    assert [bucket.module_count for bucket in train] == [
        value for value in range(2, 9) for _ in range(2)
    ]
    assert len(train) == 14
    assert sorted(allocation) == [73] * 12 + [74] * 2
    assert sum(allocation) == 1024


def test_c3_boundary_coverage_is_aggregated_without_rejecting_failed_topology() -> None:
    summaries = {
        "train_m02": {
            "phase_transition_counts": {
                "place->release": 5,
                "release->retreat": 3,
                "retreat->settle": 79,
            }
        },
        "train_m03": {
            "phase_transition_counts": {
                "place->release": 0,
                "release->retreat": 0,
                "retreat->settle": 0,
            }
        },
        "validation": {
            "phase_transition_counts": {
                "place->release": 0,
                "release->retreat": 0,
                "retreat->settle": 0,
            }
        },
    }

    assert validate_order9_c3_generation_boundary_coverage(summaries) == {
        "place->release": 5,
        "release->retreat": 3,
        "retreat->settle": 79,
    }


def test_c3_boundary_coverage_preserves_zero_success_as_training_evidence() -> None:
    assert validate_order9_c3_generation_boundary_coverage(
        {
            "train_m03": {
                "phase_transition_counts": {
                    "place->release": 0,
                    "release->retreat": 0,
                    "retreat->settle": 0,
                }
            }
        }
    ) == {
        "place->release": 0,
        "release->retreat": 0,
        "retreat->settle": 0,
    }


def test_rollout_result_parser_uses_last_structured_result(tmp_path: Path) -> None:
    log = tmp_path / "rollout.log"
    expected = {"passed": True, "generation_id": "generation-1"}
    log.write_text(
        "startup\n"
        + ORDER9_ROLLOUT_RESULT_PREFIX
        + json.dumps({"passed": False})
        + "\n"
        + ORDER9_ROLLOUT_RESULT_PREFIX
        + json.dumps(expected)
        + "\n",
        encoding="utf-8",
    )
    assert load_order9_rollout_result(log) == expected
    missing = tmp_path / "missing.log"
    missing.write_text("startup only\n", encoding="utf-8")
    with pytest.raises(SchemaValidationError, match="missing"):
        load_order9_rollout_result(missing)


def test_collector_command_binds_accepted_nominal_reset_bank(
    tmp_path: Path,
) -> None:
    bucket = _bucket(DatasetSplit.TRAIN, 0)
    kwargs = {
        "python_executable": "python",
        "repository_root": tmp_path,
        "config_path": "config.yaml",
        "stage_id": "c3_pi_l_ppo_arbitrary_morphology",
        "parent_checkpoint_path": "behavior.pt",
        "parent_checkpoint_sha256": "a" * 64,
        "generation_id": "generation-0",
        "output_raw_path": "rollout.pt",
        "bucket": bucket,
        "bucket_manifest_path": "manifest.json",
        "c3_reset_bank_path": "reset-bank.pt",
    }
    command = order9_pi_l_collector_command(**kwargs)
    assert command[command.index("--c3-reset-bank") + 1] == str(
        tmp_path / "reset-bank.pt"
    )


def test_c3_wrench_gate_curriculum_and_command_binding(tmp_path: Path) -> None:
    config = load_order9_learning_config()
    assert order9_c3_training_wrench_gate_range_scale(config, 3) == 1.0
    assert order9_c3_training_wrench_gate_range_scale(config, 4) == 1.0
    assert order9_c3_training_wrench_gate_range_scale(config, 5) == 1.0
    assert order9_c3_training_wrench_gate_range_scale(config, 9) == 1.0
    assert order9_c3_training_wrench_gate_range_scale(config, 20) == 1.0

    command = order9_pi_l_collector_command(
        python_executable="python",
        repository_root=tmp_path,
        config_path="config.yaml",
        stage_id="c3_pi_l_ppo_arbitrary_morphology",
        parent_checkpoint_path="behavior.pt",
        parent_checkpoint_sha256="a" * 64,
        generation_id="generation-0",
        output_raw_path="rollout.pt",
        bucket=_bucket(DatasetSplit.TRAIN, 0),
        bucket_manifest_path="manifest.json",
        c3_wrench_gate_range_scale=4.0,
    )
    assert command[command.index("--c3-wrench-gate-range-scale") + 1] == "4.0"


def test_collector_command_binds_production_topology_shard(tmp_path: Path) -> None:
    command = order9_pi_l_collector_command(
        python_executable="python",
        repository_root=tmp_path,
        config_path="config.yaml",
        stage_id="c3_pi_l_ppo_arbitrary_morphology",
        parent_checkpoint_path="behavior.pt",
        parent_checkpoint_sha256="a" * 64,
        generation_id="generation-0",
        output_raw_path="rollout.pt",
        bucket=_bucket(DatasetSplit.TRAIN, 0),
        bucket_manifest_path="manifest.json",
        topology_shard_index=3,
        topology_shard_count=7,
        topology_shard_environment_count=146,
    )

    assert command[-6:] == [
        "--production-topology-shard-index",
        "3",
        "--production-topology-shard-count",
        "7",
        "--production-topology-shard-environment-count",
        "146",
    ]


def test_collector_command_binds_state_inheritance_shard(tmp_path: Path) -> None:
    command = order9_pi_l_collector_command(
        python_executable="python",
        repository_root=tmp_path,
        config_path="config.yaml",
        stage_id="c3_pi_l_ppo_arbitrary_morphology",
        parent_checkpoint_path="behavior.pt",
        parent_checkpoint_sha256="a" * 64,
        generation_id="generation-0",
        output_raw_path="rollout.pt",
        bucket=_bucket(DatasetSplit.TRAIN, 0),
        bucket_manifest_path="manifest.json",
        topology_shard_index=14,
        topology_shard_count=21,
        topology_shard_environment_count=6,
        c3_state_inheritance=True,
    )

    assert command[-1] == "--production-state-inheritance-shard"


def test_collector_command_binds_c3_action_contract(tmp_path: Path) -> None:
    command = order9_pi_l_collector_command(
        python_executable="python",
        repository_root=tmp_path,
        config_path="config.yaml",
        stage_id="c3_pi_l_ppo_arbitrary_morphology",
        parent_checkpoint_path="behavior.pt",
        parent_checkpoint_sha256="a" * 64,
        generation_id="generation-0",
        output_raw_path="rollout.pt",
        bucket=_bucket(DatasetSplit.TRAIN, 0),
        bucket_manifest_path="manifest.json",
        c3_action_contract="contact_compression_plus_centroidal",
    )

    assert command[-2:] == [
        "--c3-action-contract",
        "contact_compression_plus_centroidal",
    ]


def test_collector_command_binds_training_nominal_preload_deficit(
    tmp_path: Path,
) -> None:
    command = order9_pi_l_collector_command(
        python_executable="python",
        repository_root=tmp_path,
        config_path="config.yaml",
        stage_id="c3_pi_l_ppo_arbitrary_morphology",
        parent_checkpoint_path="behavior.pt",
        parent_checkpoint_sha256="a" * 64,
        generation_id="generation-0",
        output_raw_path="rollout.pt",
        bucket=_bucket(DatasetSplit.TRAIN, 0, module_count=7),
        bucket_manifest_path="manifest.json",
        c3_training_nominal_preload_deficit_mm="0,2,4,6,8",
        c3_training_nominal_preload_deficit_module_count=7,
    )
    assert command[-4:] == [
        "--training-nominal-preload-deficit-mm",
        "0,2,4,6,8",
        "--training-nominal-preload-deficit-module-count",
        "7",
    ]
