from __future__ import annotations

"""Live TensorBoard telemetry for Order 9 rollout and PPO updates."""

import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


ORDER9_TENSORBOARD_LOGGER_VERSION = "order9_tensorboard_live_v1"
_PPO_LIVE_STEP_STRIDE = 1_000_000
_SYSTEM_SCALAR_FIELDS = (
    "gpu_memory_used_mib",
    "gpu_utilization_percent",
    "gpu_memory_utilization_percent",
    "gpu_power_draw_w",
    "gpu_temperature_c",
    "process_rss_mib",
    "system_load_1m",
    "system_load_per_cpu_1m",
    "system_memory_used_mib",
    "system_memory_available_mib",
)


class Order9TensorBoardLogger:
    """Append one immutable Order 9 generation to a shared stage log."""

    def __init__(
        self,
        log_dir: str | Path,
        *,
        stage_id: str,
        generation_id: str,
        split: str,
        update_index: int,
        generation_environment_steps: int,
        phase_labels: Sequence[str],
        reward_term_names: Sequence[str],
        writer_factory: Callable[[Path], Any] | None = None,
    ) -> None:
        if not stage_id or not generation_id or not split:
            raise ValueError("Order9 TensorBoard run identity must be non-empty")
        if update_index < 0 or generation_environment_steps < 1:
            raise ValueError("Order9 TensorBoard update/step counts are invalid")
        self.phase_labels = tuple(str(value) for value in phase_labels)
        self.reward_term_names = tuple(str(value) for value in reward_term_names)
        if (
            not self.phase_labels
            or len(set(self.phase_labels)) != len(self.phase_labels)
            or not self.reward_term_names
            or len(set(self.reward_term_names)) != len(self.reward_term_names)
        ):
            raise ValueError("Order9 TensorBoard labels must be non-empty and unique")
        self.log_dir = Path(log_dir).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        factory = writer_factory or _summary_writer
        self._writer = factory(self.log_dir)
        self.stage_id = stage_id
        self.generation_id = generation_id
        self.split = split
        self.update_index = int(update_index)
        self.generation_environment_steps = int(generation_environment_steps)
        self._metric_names = ("total_reward", *self.reward_term_names)
        self._rollout_sums = [0.0] * len(self._metric_names)
        self._rollout_count = 0
        self._phase_sums = [
            [0.0] * len(self._metric_names) for _ in self.phase_labels
        ]
        self._phase_counts = [0] * len(self.phase_labels)
        self._status_counts: dict[str, float] = {}
        self._closed = False
        metadata = {
            "logger_version": ORDER9_TENSORBOARD_LOGGER_VERSION,
            "stage_id": self.stage_id,
            "generation_id": self.generation_id,
            "split": self.split,
            "update_index": self.update_index,
            "generation_environment_steps": self.generation_environment_steps,
            "phase_labels": list(self.phase_labels),
            "reward_term_names": list(self.reward_term_names),
        }
        self._writer.add_text(
            "run/metadata",
            "```json\n" + json.dumps(metadata, indent=2, sort_keys=True) + "\n```",
            global_step=self._rollout_step_base,
        )
        self._writer.flush()

    @property
    def _rollout_step_base(self) -> int:
        return self.update_index * self.generation_environment_steps

    def log_rollout_step(
        self,
        *,
        rollout_index: int,
        reward: torch.Tensor,
        reward_terms: Mapping[str, torch.Tensor],
        phase_index: torch.Tensor,
        statuses: Mapping[str, torch.Tensor],
        elapsed_s: float,
        runtime_sample: Mapping[str, Any] | None = None,
    ) -> int:
        """Write current and running reward/phase statistics and flush them."""

        self._require_open()
        if rollout_index < 0 or not math.isfinite(elapsed_s) or elapsed_s <= 0.0:
            raise ValueError("Order9 TensorBoard rollout progress is invalid")
        if reward.ndim != 1 or phase_index.shape != reward.shape:
            raise ValueError("Order9 TensorBoard reward/phase shape mismatch")
        if set(reward_terms) != set(self.reward_term_names):
            raise ValueError("Order9 TensorBoard reward term names differ")
        batch_size = int(reward.shape[0])
        if batch_size < 1:
            raise ValueError("Order9 TensorBoard rollout batch is empty")
        ordered_terms = [reward_terms[name] for name in self.reward_term_names]
        if any(value.shape != reward.shape for value in ordered_terms):
            raise ValueError("Order9 TensorBoard reward term shape mismatch")
        phase = phase_index.to(device=reward.device, dtype=torch.long)
        if bool(((phase < 0) | (phase >= len(self.phase_labels))).any().item()):
            raise ValueError("Order9 TensorBoard phase index is invalid")
        metric_matrix = torch.stack((reward, *ordered_terms), dim=-1).double()
        phase_one_hot = torch.nn.functional.one_hot(
            phase, num_classes=len(self.phase_labels)
        ).to(dtype=metric_matrix.dtype)
        step_sums = metric_matrix.sum(dim=0)
        phase_sums = phase_one_hot.transpose(0, 1).matmul(metric_matrix)
        phase_counts = phase_one_hot.sum(dim=0)
        packed = torch.cat(
            (step_sums, phase_sums.reshape(-1), phase_counts), dim=0
        ).detach().cpu().tolist()
        metric_count = len(self._metric_names)
        cursor = 0
        current_sums = [float(value) for value in packed[cursor : cursor + metric_count]]
        cursor += metric_count
        current_phase_sums = []
        for _ in self.phase_labels:
            current_phase_sums.append(
                [float(value) for value in packed[cursor : cursor + metric_count]]
            )
            cursor += metric_count
        current_phase_counts = [
            int(round(float(value))) for value in packed[cursor:]
        ]
        self._rollout_count += batch_size
        for index, value in enumerate(current_sums):
            self._rollout_sums[index] += value
        for phase_id, count in enumerate(current_phase_counts):
            self._phase_counts[phase_id] += count
            for metric_id, value in enumerate(current_phase_sums[phase_id]):
                self._phase_sums[phase_id][metric_id] += value

        global_step = self._rollout_step_base + (rollout_index + 1) * batch_size
        for index, name in enumerate(self._metric_names):
            self._writer.add_scalar(
                f"reward/step_mean/{name}",
                current_sums[index] / batch_size,
                global_step,
            )
            self._writer.add_scalar(
                f"reward/rollout_mean/{name}",
                self._rollout_sums[index] / self._rollout_count,
                global_step,
            )
        for phase_id, label in enumerate(self.phase_labels):
            current_count = current_phase_counts[phase_id]
            running_count = self._phase_counts[phase_id]
            self._writer.add_scalar(
                f"rollout/phase_occupancy/{label}",
                current_count / batch_size,
                global_step,
            )
            for metric_id, name in enumerate(self._metric_names):
                if current_count:
                    self._writer.add_scalar(
                        f"reward/phase_step_mean/{label}/{name}",
                        current_phase_sums[phase_id][metric_id] / current_count,
                        global_step,
                    )
                if running_count:
                    self._writer.add_scalar(
                        f"reward/phase_rollout_mean/{label}/{name}",
                        self._phase_sums[phase_id][metric_id] / running_count,
                        global_step,
                    )

        for name, value in statuses.items():
            if value.shape != reward.shape:
                raise ValueError(f"Order9 TensorBoard status {name!r} shape mismatch")
            count = float(value.to(dtype=torch.float64).sum().detach().cpu().item())
            self._status_counts[name] = self._status_counts.get(name, 0.0) + count
            self._writer.add_scalar(
                f"rollout/rate/{name}", count / batch_size, global_step
            )
            self._writer.add_scalar(
                f"rollout/count/{name}_cumulative",
                self._status_counts[name],
                global_step,
            )
        collected_steps = (rollout_index + 1) * batch_size
        self._writer.add_scalar(
            "performance/rollout_environment_steps_per_s",
            collected_steps / elapsed_s,
            global_step,
        )
        self._writer.add_scalar(
            "performance/rollout_wall_elapsed_s", elapsed_s, global_step
        )
        self._log_system_sample(runtime_sample, global_step)
        self._writer.flush()
        return global_step

    def log_rollout_summary(
        self,
        *,
        environment_steps: int,
        wall_elapsed_s: float,
        runtime_load: Mapping[str, Any],
    ) -> None:
        self._require_open()
        if environment_steps < 1 or wall_elapsed_s <= 0.0:
            raise ValueError("Order9 TensorBoard rollout summary is invalid")
        global_step = self._rollout_step_base + environment_steps
        self._writer.add_scalar(
            "performance/rollout_environment_steps_per_s",
            environment_steps / wall_elapsed_s,
            global_step,
        )
        self._writer.add_scalar(
            "performance/rollout_wall_elapsed_s", wall_elapsed_s, global_step
        )
        self._log_numeric_mapping("system/summary", runtime_load, global_step)
        self._writer.flush()

    def log_ppo_minibatch(
        self,
        *,
        optimizer_step: int,
        metrics: Mapping[str, Any],
        runtime_sample: Mapping[str, Any] | None = None,
    ) -> int:
        self._require_open()
        if optimizer_step < 1:
            raise ValueError("Order9 TensorBoard optimizer step must be positive")
        global_step = self.update_index * _PPO_LIVE_STEP_STRIDE + optimizer_step
        self._log_numeric_mapping("ppo/minibatch", metrics, global_step)
        self._log_system_sample(runtime_sample, global_step)
        self._writer.flush()
        return global_step

    def log_ppo_update(
        self,
        *,
        metrics: Mapping[str, Any],
        environment_steps: int,
        wall_elapsed_s: float,
        runtime_load: Mapping[str, Any],
    ) -> None:
        self._require_open()
        if environment_steps < 1 or wall_elapsed_s <= 0.0:
            raise ValueError("Order9 TensorBoard PPO summary is invalid")
        self._log_numeric_mapping("ppo/update", metrics, self.update_index)
        cumulative_environment_steps = self._rollout_step_base + environment_steps
        self._writer.add_scalar(
            "performance/ppo_environment_steps_per_s",
            environment_steps / wall_elapsed_s,
            cumulative_environment_steps,
        )
        self._writer.add_scalar(
            "performance/ppo_wall_elapsed_s",
            wall_elapsed_s,
            cumulative_environment_steps,
        )
        self._log_numeric_mapping(
            "system/ppo_summary", runtime_load, cumulative_environment_steps
        )
        self._writer.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._writer.flush()
        self._writer.close()
        self._closed = True

    def _log_system_sample(
        self, sample: Mapping[str, Any] | None, global_step: int
    ) -> None:
        if not sample:
            return
        for name in _SYSTEM_SCALAR_FIELDS:
            value = sample.get(name)
            if _is_finite_number(value):
                self._writer.add_scalar(
                    f"system/live/{name}", float(value), global_step
                )

    def _log_numeric_mapping(
        self, prefix: str, values: Mapping[str, Any], global_step: int
    ) -> None:
        for name, value in values.items():
            if _is_finite_number(value):
                self._writer.add_scalar(f"{prefix}/{name}", float(value), global_step)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Order9 TensorBoard logger is closed")


def _summary_writer(log_dir: Path) -> Any:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:  # pragma: no cover - environment-specific guard.
        raise RuntimeError(
            "TensorBoard is required for live Order9 logging; run in isaaclab3"
        ) from error
    return SummaryWriter(log_dir=str(log_dir), max_queue=10, flush_secs=2)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


__all__ = ["ORDER9_TENSORBOARD_LOGGER_VERSION", "Order9TensorBoardLogger"]
