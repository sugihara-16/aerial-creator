#!/usr/bin/env python3
from __future__ import annotations

"""Run protected nominal R1 with deterministic extra contact compression."""

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import torch

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from amsrr.schemas.common import SchemaValidationError  # noqa: E402
from amsrr.simulation.order9_tensor_object_task import (  # noqa: E402
    ORDER9_CONTACT_SCHEDULE_ATTACH,
    ORDER9_CONTACT_SCHEDULE_MAINTAIN,
    ORDER9_CONTACT_SCHEDULE_RELEASE,
)
from amsrr.training.order9_r1_pi_l_isolated_action_diagnostic import (  # noqa: E402
    protected_rollout_source_without_process_exit,
)
from amsrr.training.order9_tensor_pi_l_runtime import (  # noqa: E402
    Order9TensorPiLRuntime,
)
from amsrr.utils.hashing import hash_file  # noqa: E402

PROTECTED_ROLLOUT = REPOSITORY / "scripts/order9_vectorized_isaac_rollout.py"
PROTECTED_ROLLOUT_SHA256 = (
    "e813f950dde32818b7375949bfb58baa983dde760e58da22cfbb79cc33b764c2"
)
SWEEP_OPTION = "--r1-nominal-additional-compression-sweep-mm"
WRAPPER_VERSION = "order9_r1_deterministic_nominal_compression_sweep_v1"


def _value(arguments: list[str], option: str) -> str:
    matches = [index for index, value in enumerate(arguments) if value == option]
    if len(matches) != 1 or matches[0] + 1 >= len(arguments):
        raise ValueError(f"R1 compression sweep requires exactly one {option}")
    return arguments[matches[0] + 1]


def _remove_value(arguments: list[str], option: str) -> tuple[str, list[str]]:
    value = _value(arguments, option)
    index = arguments.index(option)
    return value, arguments[:index] + arguments[index + 2 :]


def _parse_sweep(raw: str, *, environment_count: int) -> tuple[float, ...]:
    try:
        values = tuple(float(value.strip()) for value in raw.split(","))
    except ValueError as error:
        raise ValueError(
            "R1 compression sweep must be comma-separated numbers"
        ) from error
    if len(values) != environment_count or any(
        not math.isfinite(value) or not 0.0 <= value <= 10.0 for value in values
    ):
        raise ValueError(
            "R1 compression sweep requires one value per environment " "in [0, 10] mm"
        )
    return values


def _compression_scale(task_target) -> torch.Tensor:
    progress = task_target.phase_progress.clamp(0.0, 1.0)
    smooth = progress * progress * (3.0 - 2.0 * progress)
    schedule = task_target.contact_schedule_index
    return torch.where(
        schedule == ORDER9_CONTACT_SCHEDULE_ATTACH,
        smooth,
        torch.where(
            schedule == ORDER9_CONTACT_SCHEDULE_MAINTAIN,
            torch.ones_like(progress),
            torch.where(
                schedule == ORDER9_CONTACT_SCHEDULE_RELEASE,
                1.0 - smooth,
                torch.zeros_like(progress),
            ),
        ),
    )


def _additional_joint_delta(runtime, task_target, values_m: torch.Tensor):
    basis = runtime.contact_space_action_basis
    if basis is None:
        raise RuntimeError("R1 compression sweep requires contact-space geometry")
    moved = basis.to(
        device=task_target.nominal_joint_positions_rad.device,
        dtype=task_target.nominal_joint_positions_rad.dtype,
    )
    batch = task_target.nominal_joint_positions_rad.shape[0]
    if values_m.shape != (batch,):
        raise RuntimeError("R1 compression sweep environment count changed")
    physical = torch.zeros(
        (batch, moved.contact_slot_mask.numel(), 6),
        device=values_m.device,
        dtype=values_m.dtype,
    )
    active = moved.contact_slot_mask.unsqueeze(0).expand(batch, -1)
    physical[:, :, 0] = torch.where(
        active,
        values_m.unsqueeze(1).expand_as(physical[:, :, 0]),
        physical[:, :, 0],
    )
    flat = torch.einsum(
        "ij,bj->bi",
        moved.contact_to_joint_position,
        physical.reshape(batch, -1),
    )
    return flat.reshape_as(task_target.nominal_joint_positions_rad) * (
        _compression_scale(task_target).reshape(batch, 1, 1)
    )


def _close_simulation_application(namespace: dict[str, Any]) -> None:
    simulation_app = namespace.get("simulation_app")
    if simulation_app is not None:
        simulation_app.close()


def main() -> int:
    sweep_raw, arguments = _remove_value(list(sys.argv[1:]), SWEEP_OPTION)
    if hash_file(PROTECTED_ROLLOUT) != PROTECTED_ROLLOUT_SHA256:
        raise SchemaValidationError("protected C3 rollout bytes changed")
    if "--diagnostic-nominal-qpid-only" not in arguments:
        raise ValueError("R1 compression sweep requires the nominal pi_L bypass")
    if "--formal-phase-zero-start" not in arguments:
        raise ValueError("R1 compression sweep requires formal phase-zero execution")
    environment_count = int(_value(arguments, "--num-envs"))
    values_mm = _parse_sweep(sweep_raw, environment_count=environment_count)
    output_pairs = _output_pairs(arguments)

    original = Order9TensorPiLRuntime.compute_nominal_qpid_hold
    calls = 0

    def wrapped(runtime, **kwargs):
        nonlocal calls
        target = kwargs["task_target"]
        values_m = torch.tensor(
            [value * 1.0e-3 for value in values_mm],
            device=target.nominal_joint_positions_rad.device,
            dtype=target.nominal_joint_positions_rad.dtype,
        )
        delta = _additional_joint_delta(runtime, target, values_m)
        kwargs["task_target"] = replace(
            target,
            nominal_joint_positions_rad=(target.nominal_joint_positions_rad + delta),
        )
        calls += 1
        return original(runtime, **kwargs)

    namespace: dict[str, Any] = {
        "__file__": str(PROTECTED_ROLLOUT),
        "__name__": "__main__",
        "__package__": None,
        "__cached__": None,
    }
    source = protected_rollout_source_without_process_exit(
        PROTECTED_ROLLOUT.read_text(encoding="utf-8")
    )
    previous_argv = sys.argv
    Order9TensorPiLRuntime.compute_nominal_qpid_hold = wrapped
    try:
        sys.argv = [str(PROTECTED_ROLLOUT), *arguments]
        exec(compile(source, str(PROTECTED_ROLLOUT), "exec"), namespace)
        exit_code = int(namespace.get("_exit_code", 1))
        if exit_code != 0:
            return exit_code
        if calls < 1:
            raise RuntimeError("R1 compression sweep did not reach nominal control")
        for output_raw, episodes in output_pairs:
            _annotate(
                raw_path=output_raw,
                episode_path=episodes,
                values_mm=values_mm,
                call_count=calls,
            )
    finally:
        sys.argv = previous_argv
        Order9TensorPiLRuntime.compute_nominal_qpid_hold = original
        _close_simulation_application(namespace)
    print(
        "ORDER9_R1_NOMINAL_COMPRESSION_SWEEP="
        + ",".join(str(raw) for raw, _episodes in output_pairs),
        flush=True,
    )
    return 0


def _output_pairs(arguments: list[str]) -> tuple[tuple[Path, Path], ...]:
    if "--persistent-bucket-jobs" not in arguments:
        return (
            (
                Path(_value(arguments, "--output-raw")).resolve(),
                Path(_value(arguments, "--evaluation-jsonl")).resolve(),
            ),
        )
    manifest_path = Path(_value(arguments, "--persistent-bucket-jobs")).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = payload.get("jobs") if isinstance(payload, dict) else None
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("R1 compression persistent job manifest is invalid")
    result = []
    for job in jobs:
        argv = job.get("argv") if isinstance(job, dict) else None
        if not isinstance(argv, list) or not all(
            isinstance(value, str) for value in argv
        ):
            raise ValueError("R1 compression persistent job arguments are invalid")
        result.append(
            (
                Path(_value(argv, "--output-raw")).resolve(),
                Path(_value(argv, "--evaluation-jsonl")).resolve(),
            )
        )
    return tuple(result)


def _annotate(
    *,
    raw_path: Path,
    episode_path: Path,
    values_mm: tuple[float, ...],
    call_count: int,
) -> None:
    payload = torch.load(raw_path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    if not isinstance(metadata, dict):
        raise SchemaValidationError("R1 compression sweep raw artifact is invalid")
    annotation = {
        "r1_nominal_compression_diagnostic_version": WRAPPER_VERSION,
        "r1_nominal_additional_compression_mm_by_environment": list(values_mm),
        "r1_nominal_compression_control_call_count": int(call_count),
        "r1_nominal_compression_wrapper_sha256": hash_file(Path(__file__)),
        "r1_protected_rollout_sha256": PROTECTED_ROLLOUT_SHA256,
        "pi_l_actor_command_applied": False,
        "promotion_evidence_eligible": False,
        "training_eligible": False,
        "formal_teacher_collection_authorized": False,
    }
    metadata.update(annotation)
    _atomic_torch_save(payload, raw_path)
    raw_sha = hash_file(raw_path)
    records = [
        json.loads(line)
        for line in episode_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != len(values_mm):
        raise SchemaValidationError("R1 compression sweep episode count changed")
    for index, record in enumerate(records):
        record["source_artifact_path"] = str(raw_path)
        record["source_artifact_sha256"] = raw_sha
        record.setdefault("metadata", {}).update(annotation)
        record["metadata"]["r1_nominal_additional_compression_mm"] = values_mm[index]
    _atomic_text_write(
        episode_path,
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
    )


def _atomic_torch_save(payload: Any, destination: Path) -> None:
    descriptor, temporary_text = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_text)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text_write(destination: Path, value: str) -> None:
    descriptor, temporary_text = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
