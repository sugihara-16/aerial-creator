from __future__ import annotations

"""Parent-process launcher for the real vectorized Order 9 Isaac benchmark."""

import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable, Sequence

from amsrr.simulation.isaac_lab_backend import (
    IsaacLabBackendConfig,
    load_isaac_lab_backend_config,
)
from amsrr.training.order9_curriculum import Order9RuntimeBenchmarkConfig
from amsrr.training.order9_runtime_benchmark import (
    ORDER9_RUNTIME_BENCHMARK_VERSION,
    Order9RuntimeBenchmarkReport,
    Order9RuntimeBenchmarkSample,
    load_order8_timing_reference,
    write_order9_runtime_benchmark_report,
)
from amsrr.utils.hashing import stable_hash


ORDER9_ISAAC_BENCHMARK_RUNNER_VERSION = "order9_isaac_benchmark_runner_v2"
_JSON_PREFIX = "ORDER9_BENCHMARK_JSON="
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


CommandRunner = Callable[[Sequence[str], float], dict[str, Any]]


def run_order9_isaac_runtime_benchmark(
    config: Order9RuntimeBenchmarkConfig,
    *,
    backend_config: IsaacLabBackendConfig | None = None,
    command_runner: CommandRunner | None = None,
    output_path: str | Path | None = None,
) -> Order9RuntimeBenchmarkReport:
    config.validate()
    backend = backend_config or load_isaac_lab_backend_config(
        _REPOSITORY_ROOT / "configs/env/isaac_lab.yaml"
    )
    runner = command_runner or _run_child_command
    samples: list[Order9RuntimeBenchmarkSample] = []
    for environment_count in config.environment_count_candidates:
        command = order9_isaac_benchmark_command(
            config,
            backend,
            environment_count=environment_count,
        )
        try:
            payload = runner(command, config.maximum_wall_time_s)
            samples.append(_sample_from_payload(payload, config, environment_count))
        except Exception as exc:
            samples.append(
                Order9RuntimeBenchmarkSample(
                    environment_count=environment_count,
                    attempted=True,
                    isaac_backed=False,
                    backend_version="isaac_child_failed",
                    device=backend.device,
                    warmup_steps=config.warmup_steps,
                    measurement_steps=0,
                    wall_elapsed_s=0.0,
                    aggregate_env_steps_per_s=0.0,
                    per_environment_steps_per_s=0.0,
                    passed_throughput_gate=False,
                    topology_bucketed=False,
                    phase_specific_resets=False,
                    failure_reason=f"{type(exc).__name__}:{exc}",
                )
            )
    passing = [sample for sample in samples if sample.passed_throughput_gate]
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
    report = Order9RuntimeBenchmarkReport(
        benchmark_version=ORDER9_RUNTIME_BENCHMARK_VERSION,
        config_hash=stable_hash(config.to_dict()),
        samples=samples,
        selected_environment_count=selected,
        minimum_aggregate_env_steps_per_s=config.minimum_aggregate_env_steps_per_s,
        passed=selected is not None,
        order8_reference=load_order8_timing_reference(),
        metadata={
            "runner_version": ORDER9_ISAAC_BENCHMARK_RUNNER_VERSION,
            "initial_environment_count": config.initial_environment_count,
            "selection_rule": "maximum_measured_aggregate_throughput_then_initial_count_proximity",
            "training_approximation": "cached_order8_morphology_plus_free_box_support_contact",
            "tensorized_pi_l_inference_required": (
                config.require_tensorized_pi_l_inference
            ),
            "full_order8_acceptance_replaced": False,
            "per_step_json_logging": False,
        },
    )
    if output_path is not None:
        write_order9_runtime_benchmark_report(output_path, report)
    return report


def order9_isaac_benchmark_command(
    config: Order9RuntimeBenchmarkConfig,
    backend: IsaacLabBackendConfig,
    *,
    environment_count: int,
) -> list[str]:
    isaaclab_root = Path(os.path.expandvars(backend.isaaclab_path)).expanduser()
    launch_script = isaaclab_root / backend.launch_script
    child = _REPOSITORY_ROOT / "scripts/order9_vectorized_isaac_benchmark.py"
    command = [
        str(launch_script),
        "-p",
        str(child),
        "--num-envs",
        str(environment_count),
        "--warmup-steps",
        str(config.warmup_steps),
        "--measurement-steps",
        str(config.measurement_steps),
        "--dt",
        str(config.control_dt_s),
        "--device",
        backend.device,
    ]
    micromamba = _micromamba_executable()
    if micromamba is not None:
        command = [
            str(micromamba),
            "run",
            "-n",
            backend.micromamba_env,
            *command,
        ]
    return command


def _run_child_command(command: Sequence[str], timeout_s: float) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=_REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_s,
    )
    if completed.returncode != 0:
        tail = "\n".join((completed.stderr or completed.stdout).splitlines()[-12:])
        raise RuntimeError(
            f"Order9 Isaac benchmark child exited {completed.returncode}: {tail}"
        )
    lines = [
        line[len(_JSON_PREFIX) :]
        for line in completed.stdout.splitlines()
        if line.startswith(_JSON_PREFIX)
    ]
    if len(lines) != 1:
        raise RuntimeError("Order9 Isaac benchmark child emitted no unique JSON payload")
    payload = json.loads(lines[0])
    if not isinstance(payload, dict):
        raise RuntimeError("Order9 Isaac benchmark child payload must be a mapping")
    return payload


def _sample_from_payload(
    payload: dict[str, Any],
    config: Order9RuntimeBenchmarkConfig,
    expected_count: int,
) -> Order9RuntimeBenchmarkSample:
    count = int(payload.get("environment_count", 0))
    warmup = int(payload.get("warmup_steps", -1))
    steps = int(payload.get("measurement_steps", -1))
    dt = float(payload.get("control_dt_s", 0.0))
    if count != expected_count or warmup != config.warmup_steps or steps != config.measurement_steps:
        raise RuntimeError("Order9 Isaac child benchmark request/response mismatch")
    if abs(dt - config.control_dt_s) > 1.0e-12:
        raise RuntimeError("Order9 Isaac child control dt mismatch")
    aggregate = float(payload.get("aggregate_env_steps_per_s", 0.0))
    contract_ok = all(
        (
            payload.get("attempted") is True,
            payload.get("isaac_backed") is True,
            payload.get("finite_state") is True,
            payload.get("topology_bucketed") is True,
            payload.get("phase_specific_resets") is True,
            payload.get("per_step_json_logging") is False,
            payload.get("raw_contact_actor_input") is False,
            payload.get("unchanged_order8_acceptance_replaced") is False,
            (
                payload.get("tensorized_pi_l_inference") is True
                if config.require_tensorized_pi_l_inference
                else True
            ),
        )
    )
    return Order9RuntimeBenchmarkSample(
        environment_count=count,
        attempted=True,
        isaac_backed=bool(payload.get("isaac_backed", False)),
        backend_version=str(payload.get("backend_version", "unknown")),
        device=str(payload.get("device", "unknown")),
        warmup_steps=warmup,
        measurement_steps=steps,
        wall_elapsed_s=float(payload.get("wall_elapsed_s", 0.0)),
        aggregate_env_steps_per_s=aggregate,
        per_environment_steps_per_s=float(
            payload.get("per_environment_steps_per_s", 0.0)
        ),
        passed_throughput_gate=(
            contract_ok and aggregate >= config.minimum_aggregate_env_steps_per_s
        ),
        topology_bucketed=bool(payload.get("topology_bucketed", False)),
        phase_specific_resets=bool(payload.get("phase_specific_resets", False)),
        failure_reason=None if contract_ok else "child_contract_validation_failed",
        metadata={
            key: value
            for key, value in payload.items()
            if key
            not in {
                "environment_count",
                "attempted",
                "isaac_backed",
                "backend_version",
                "device",
                "warmup_steps",
                "measurement_steps",
                "wall_elapsed_s",
                "aggregate_env_steps_per_s",
                "per_environment_steps_per_s",
                "topology_bucketed",
                "phase_specific_resets",
            }
        },
    )


def _micromamba_executable() -> Path | None:
    candidate = shutil.which("micromamba")
    return Path(candidate) if candidate is not None else None
