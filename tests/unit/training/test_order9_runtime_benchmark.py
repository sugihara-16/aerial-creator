from __future__ import annotations

import pytest

from amsrr.simulation.isaac_lab_backend import IsaacLabBackendConfig
from amsrr.training.order9_curriculum import Order9RuntimeBenchmarkConfig
from amsrr.training.order9_isaac_benchmark_runner import (
    run_order9_isaac_runtime_benchmark,
)
from amsrr.training.order9_runtime_benchmark import (
    Order9RuntimeBenchmarkReport,
    benchmark_order9_runtime,
    load_order8_timing_reference,
)


def test_benchmark_selects_measured_throughput_and_archives_order8_reference() -> None:
    clock = _Clock()
    per_step = {32: 0.020, 64: 0.025, 128: 0.080}
    config = Order9RuntimeBenchmarkConfig(
        environment_count_candidates=[32, 64, 128],
        initial_environment_count=64,
        minimum_aggregate_env_steps_per_s=500.0,
        warmup_steps=2,
        measurement_steps=10,
        maximum_wall_time_s=10.0,
    )
    reference = load_order8_timing_reference()

    report = benchmark_order9_runtime(
        config,
        lambda count: _Environment(count, clock, per_step[count]),
        order8_reference=reference,
        clock=clock,
    )
    roundtrip = Order9RuntimeBenchmarkReport.from_json(report.to_json())

    assert report.passed
    assert report.selected_environment_count == 64
    assert [sample.passed_throughput_gate for sample in report.samples] == [True, True, True]
    assert roundtrip.to_dict() == report.to_dict()
    assert reference.physics_step_count == 6445
    assert reference.simulation_dt_s == pytest.approx(0.02)
    assert reference.wall_elapsed_s is None


def test_benchmark_rejects_non_isaac_environment() -> None:
    clock = _Clock()
    config = Order9RuntimeBenchmarkConfig(
        environment_count_candidates=[1],
        initial_environment_count=1,
        minimum_aggregate_env_steps_per_s=1.0,
        warmup_steps=1,
        measurement_steps=1,
        maximum_wall_time_s=1.0,
    )

    report = benchmark_order9_runtime(
        config,
        lambda count: _Environment(count, clock, 0.01, isaac_backed=False),
        clock=clock,
    )

    assert not report.passed
    assert report.samples[0].failure_reason is not None
    assert "requires an Isaac-backed env" in report.samples[0].failure_reason


def test_isaac_parent_runner_validates_child_contract() -> None:
    config = Order9RuntimeBenchmarkConfig(
        environment_count_candidates=[32, 64],
        initial_environment_count=64,
        minimum_aggregate_env_steps_per_s=500.0,
        warmup_steps=2,
        measurement_steps=10,
        maximum_wall_time_s=10.0,
    )

    def child(command, timeout):
        del timeout
        count = int(command[command.index("--num-envs") + 1])
        aggregate = {32: 700.0, 64: 1200.0}[count]
        return {
            "attempted": True,
            "isaac_backed": True,
            "finite_state": True,
            "backend_version": "isaac-test",
            "device": "cuda:0",
            "environment_count": count,
            "warmup_steps": 2,
            "measurement_steps": 10,
            "control_dt_s": 0.02,
            "wall_elapsed_s": count * 10 / aggregate,
            "aggregate_env_steps_per_s": aggregate,
            "per_environment_steps_per_s": aggregate / count,
            "topology_bucketed": True,
            "phase_specific_resets": True,
            "per_step_json_logging": False,
            "raw_contact_actor_input": False,
            "unchanged_order8_acceptance_replaced": False,
            "tensorized_pi_l_inference": True,
        }

    report = run_order9_isaac_runtime_benchmark(
        config,
        backend_config=IsaacLabBackendConfig(),
        command_runner=child,
    )

    assert report.passed
    assert report.selected_environment_count == 64
    assert all(sample.isaac_backed for sample in report.samples)


def test_isaac_parent_runner_rejects_physics_only_child_payload() -> None:
    config = Order9RuntimeBenchmarkConfig(
        environment_count_candidates=[32],
        initial_environment_count=32,
        minimum_aggregate_env_steps_per_s=1.0,
        warmup_steps=1,
        measurement_steps=1,
        maximum_wall_time_s=10.0,
    )

    def child(command, timeout):
        del command, timeout
        return {
            "attempted": True,
            "isaac_backed": True,
            "finite_state": True,
            "backend_version": "isaac-test",
            "device": "cuda:0",
            "environment_count": 32,
            "warmup_steps": 1,
            "measurement_steps": 1,
            "control_dt_s": 0.02,
            "wall_elapsed_s": 0.01,
            "aggregate_env_steps_per_s": 3200.0,
            "per_environment_steps_per_s": 100.0,
            "topology_bucketed": True,
            "phase_specific_resets": True,
            "per_step_json_logging": False,
            "raw_contact_actor_input": False,
            "unchanged_order8_acceptance_replaced": False,
            "tensorized_pi_l_inference": False,
        }

    report = run_order9_isaac_runtime_benchmark(
        config,
        backend_config=IsaacLabBackendConfig(),
        command_runner=child,
    )

    assert not report.passed
    assert report.samples[0].failure_reason == "child_contract_validation_failed"


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class _Environment:
    backend_version = "test_isaac_backend_v1"
    device = "cuda:0"
    topology_bucketed = True
    phase_specific_resets = True

    def __init__(
        self,
        num_envs: int,
        clock: _Clock,
        step_duration: float,
        *,
        isaac_backed: bool = True,
    ) -> None:
        self.num_envs = num_envs
        self.clock = clock
        self.step_duration = step_duration
        self.isaac_backed = isaac_backed

    def reset(self) -> None:
        pass

    def step(self) -> None:
        self.clock.value += self.step_duration

    def synchronize(self) -> None:
        pass

    def close(self) -> None:
        pass
