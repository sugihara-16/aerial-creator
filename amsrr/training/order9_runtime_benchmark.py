from __future__ import annotations

"""Measured throughput gate for the vectorized Isaac Order 9 runtime."""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Callable, Protocol

from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_non_empty
from amsrr.training.order9_curriculum import Order9RuntimeBenchmarkConfig
from amsrr.utils.hashing import stable_hash


ORDER9_RUNTIME_BENCHMARK_VERSION = "order9_production_collector_benchmark_v3"
ORDER8_CANONICAL_REPORT_PATH = (
    "artifacts/p4_full/order8_natural_contact/order8_mu4p5_dt20ms_full_v406.json"
)


class Order9BenchmarkEnvironment(Protocol):
    """Minimum non-logging vector runtime used by the benchmark and collector."""

    num_envs: int
    isaac_backed: bool
    backend_version: str
    device: str
    topology_bucketed: bool
    phase_specific_resets: bool

    def reset(self) -> None:
        ...

    def step(self) -> None:
        ...

    def synchronize(self) -> None:
        ...

    def close(self) -> None:
        ...


@dataclass
class Order8TimingReference(SchemaBase):
    report_path: str
    simulated_duration_s: float
    physics_step_count: int
    simulation_dt_s: float
    wall_elapsed_s: float | None
    single_environment_steps_per_s: float | None
    source_note: str

    def validate(self) -> None:
        require_non_empty(self.report_path, "Order8TimingReference.report_path")
        if self.simulated_duration_s <= 0.0 or self.physics_step_count <= 0:
            raise SchemaValidationError("Order8 timing reference must contain a completed run")
        if self.simulation_dt_s <= 0.0:
            raise SchemaValidationError("Order8 timing reference dt must be positive")
        if self.wall_elapsed_s is not None and self.wall_elapsed_s <= 0.0:
            raise SchemaValidationError("Order8 timing wall time must be positive")
        if self.single_environment_steps_per_s is not None:
            if self.single_environment_steps_per_s <= 0.0:
                raise SchemaValidationError("Order8 timing throughput must be positive")
        require_non_empty(self.source_note, "Order8TimingReference.source_note")


@dataclass
class Order9RuntimeBenchmarkSample(SchemaBase):
    environment_count: int
    attempted: bool
    isaac_backed: bool
    backend_version: str
    device: str
    warmup_steps: int
    measurement_steps: int
    wall_elapsed_s: float
    aggregate_env_steps_per_s: float
    per_environment_steps_per_s: float
    passed_throughput_gate: bool
    topology_bucketed: bool
    phase_specific_resets: bool
    failure_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.environment_count < 1:
            raise SchemaValidationError("Order9 benchmark environment_count must be positive")
        for name in ("warmup_steps", "measurement_steps"):
            if getattr(self, name) < 0:
                raise SchemaValidationError(f"Order9 benchmark {name} must be non-negative")
        for name in (
            "wall_elapsed_s",
            "aggregate_env_steps_per_s",
            "per_environment_steps_per_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise SchemaValidationError(f"Order9 benchmark {name} must be finite/non-negative")
        if self.passed_throughput_gate and (
            not self.attempted
            or not self.isaac_backed
            or not self.topology_bucketed
            or not self.phase_specific_resets
        ):
            raise SchemaValidationError(
                "Order9 throughput pass requires real Isaac and both production optimizations"
            )
        if not self.attempted and self.failure_reason is None:
            raise SchemaValidationError("unattempted Order9 benchmark requires failure_reason")


@dataclass
class Order9RuntimeBenchmarkReport(SchemaBase):
    benchmark_version: str
    config_hash: str
    samples: list[Order9RuntimeBenchmarkSample]
    selected_environment_count: int | None
    minimum_aggregate_env_steps_per_s: float
    passed: bool
    order8_reference: Order8TimingReference | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        require_non_empty(self.benchmark_version, "Order9RuntimeBenchmarkReport.version")
        require_non_empty(self.config_hash, "Order9RuntimeBenchmarkReport.config_hash")
        if not self.samples:
            raise SchemaValidationError("Order9 benchmark report requires samples")
        counts = [sample.environment_count for sample in self.samples]
        if len(counts) != len(set(counts)):
            raise SchemaValidationError("Order9 benchmark sample counts must be unique")
        passing = {
            sample.environment_count
            for sample in self.samples
            if sample.passed_throughput_gate
        }
        if self.passed:
            if self.selected_environment_count not in passing:
                raise SchemaValidationError(
                    "passing Order9 benchmark must select a passing environment count"
                )
        elif self.selected_environment_count is not None:
            raise SchemaValidationError(
                "failed Order9 benchmark cannot select an environment count"
            )


def benchmark_order9_runtime(
    config: Order9RuntimeBenchmarkConfig,
    environment_factory: Callable[[int], Order9BenchmarkEnvironment],
    *,
    order8_reference: Order8TimingReference | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> Order9RuntimeBenchmarkReport:
    """Benchmark every configured environment count without per-step logging."""

    config.validate()
    samples: list[Order9RuntimeBenchmarkSample] = []
    for count in config.environment_count_candidates:
        environment: Order9BenchmarkEnvironment | None = None
        started = clock()
        try:
            environment = environment_factory(count)
            _validate_environment_contract(environment, count)
            environment.reset()
            for _ in range(config.warmup_steps):
                environment.step()
            environment.synchronize()
            measurement_started = clock()
            for _ in range(config.measurement_steps):
                environment.step()
                if clock() - started > config.maximum_wall_time_s:
                    raise TimeoutError("Order9 benchmark exceeded maximum_wall_time_s")
            environment.synchronize()
            elapsed = max(clock() - measurement_started, 1.0e-12)
            aggregate = count * config.measurement_steps / elapsed
            samples.append(
                Order9RuntimeBenchmarkSample(
                    environment_count=count,
                    attempted=True,
                    isaac_backed=bool(environment.isaac_backed),
                    backend_version=str(environment.backend_version),
                    device=str(environment.device),
                    warmup_steps=config.warmup_steps,
                    measurement_steps=config.measurement_steps,
                    wall_elapsed_s=elapsed,
                    aggregate_env_steps_per_s=aggregate,
                    per_environment_steps_per_s=aggregate / count,
                    passed_throughput_gate=(
                        environment.isaac_backed
                        and aggregate >= config.minimum_aggregate_env_steps_per_s
                    ),
                    topology_bucketed=bool(environment.topology_bucketed),
                    phase_specific_resets=bool(environment.phase_specific_resets),
                )
            )
        except Exception as exc:
            elapsed = max(clock() - started, 0.0)
            samples.append(
                Order9RuntimeBenchmarkSample(
                    environment_count=count,
                    attempted=environment is not None,
                    isaac_backed=bool(
                        environment is not None and environment.isaac_backed
                    ),
                    backend_version=(
                        str(environment.backend_version)
                        if environment is not None
                        else "unavailable"
                    ),
                    device=(
                        str(environment.device)
                        if environment is not None
                        else "unavailable"
                    ),
                    warmup_steps=config.warmup_steps,
                    measurement_steps=0,
                    wall_elapsed_s=elapsed,
                    aggregate_env_steps_per_s=0.0,
                    per_environment_steps_per_s=0.0,
                    passed_throughput_gate=False,
                    topology_bucketed=bool(
                        environment is not None and environment.topology_bucketed
                    ),
                    phase_specific_resets=bool(
                        environment is not None and environment.phase_specific_resets
                    ),
                    failure_reason=f"{type(exc).__name__}:{exc}",
                )
            )
        finally:
            if environment is not None:
                environment.close()
    passing = [sample for sample in samples if sample.passed_throughput_gate]
    # Select measured throughput, not the largest nominal batch.  The initial
    # count is only a starting hypothesis and remains in the report.
    selected = (
        max(
            passing,
            key=lambda sample: (
                sample.aggregate_env_steps_per_s,
                -abs(sample.environment_count - config.initial_environment_count),
            ),
        ).environment_count
        if passing
        else None
    )
    return Order9RuntimeBenchmarkReport(
        benchmark_version=ORDER9_RUNTIME_BENCHMARK_VERSION,
        config_hash=stable_hash(config.to_dict()),
        samples=samples,
        selected_environment_count=selected,
        minimum_aggregate_env_steps_per_s=config.minimum_aggregate_env_steps_per_s,
        passed=selected is not None,
        order8_reference=order8_reference,
        metadata={
            "initial_environment_count": config.initial_environment_count,
            "selection_rule": "maximum_measured_aggregate_throughput_then_initial_count_proximity",
            "per_step_json_logging": False,
        },
    )


def load_order8_timing_reference(
    path: str | Path = ORDER8_CANONICAL_REPORT_PATH,
    *,
    wall_elapsed_s: float | None = None,
) -> Order8TimingReference:
    """Extract the canonical Order 8 timing without inventing absent wall time."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    report = payload.get("report", payload)
    monitor = report.get("order8_natural_contact_monitor_result", {})
    steps = int(monitor.get("step_count", 0))
    duration = float(report.get("order8_natural_contact_simulation_time_s", 0.0))
    if duration <= 0.0:
        duration = float(monitor.get("duration_s", 0.0))
    dt = float(
        report.get(
            "order8_natural_contact_simulation_dt_s",
            duration / steps if steps > 0 else 0.0,
        )
    )
    throughput = steps / wall_elapsed_s if wall_elapsed_s is not None else None
    return Order8TimingReference(
        report_path=str(source),
        simulated_duration_s=duration,
        physics_step_count=steps,
        simulation_dt_s=dt,
        wall_elapsed_s=wall_elapsed_s,
        single_environment_steps_per_s=throughput,
        source_note=(
            "canonical_v406_records_simulated_time_and_steps_but_no_wall_clock"
            if wall_elapsed_s is None
            else "canonical_v406_with_separately_measured_wall_clock"
        ),
    )


def write_order9_runtime_benchmark_report(
    path: str | Path,
    report: Order9RuntimeBenchmarkReport,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report.to_json(indent=2), encoding="utf-8")


def _validate_environment_contract(
    environment: Order9BenchmarkEnvironment,
    expected_count: int,
) -> None:
    if environment.num_envs != expected_count:
        raise SchemaValidationError("Order9 benchmark environment count mismatch")
    if not environment.isaac_backed:
        raise SchemaValidationError("Order9 production benchmark requires an Isaac-backed env")
    if not environment.topology_bucketed or not environment.phase_specific_resets:
        raise SchemaValidationError(
            "Order9 production benchmark requires topology buckets and phase resets"
        )
    require_non_empty(environment.backend_version, "Order9 benchmark backend_version")
    require_non_empty(environment.device, "Order9 benchmark device")
