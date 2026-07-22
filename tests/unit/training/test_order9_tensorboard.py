from __future__ import annotations

from pathlib import Path

import pytest
import torch

from amsrr.training.order9_tensor_reward import ORDER9_TENSOR_REWARD_TERM_NAMES
from amsrr.training.order9_tensorboard import Order9TensorBoardLogger


class _Writer:
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.scalars: list[tuple[str, float, int]] = []
        self.text: list[tuple[str, str, int]] = []
        self.flush_count = 0
        self.closed = False

    def add_scalar(self, tag: str, value: float, global_step: int) -> None:
        self.scalars.append((tag, float(value), int(global_step)))

    def add_text(self, tag: str, value: str, global_step: int) -> None:
        self.text.append((tag, value, int(global_step)))

    def flush(self) -> None:
        self.flush_count += 1

    def close(self) -> None:
        self.closed = True


def test_tensorboard_logger_flushes_live_reward_phase_and_ppo_metrics(
    tmp_path: Path,
) -> None:
    writer: _Writer | None = None

    def factory(log_dir: Path) -> _Writer:
        nonlocal writer
        writer = _Writer(log_dir)
        return writer

    logger = Order9TensorBoardLogger(
        tmp_path / "tensorboard",
        stage_id="c2_pi_l_ppo_fixed_conservative",
        generation_id="generation-3",
        split="train",
        update_index=3,
        generation_environment_steps=32,
        phase_labels=("approach", "lift"),
        reward_term_names=ORDER9_TENSOR_REWARD_TERM_NAMES,
        writer_factory=factory,
    )
    assert writer is not None
    reward = torch.tensor([1.0, 3.0, 5.0, 7.0])
    terms = {
        name: torch.full((4,), float(index + 1))
        for index, name in enumerate(ORDER9_TENSOR_REWARD_TERM_NAMES)
    }
    global_step = logger.log_rollout_step(
        rollout_index=0,
        reward=reward,
        reward_terms=terms,
        phase_index=torch.tensor([0, 0, 1, 1]),
        statuses={
            "phase_success": torch.tensor([True, False, True, False]),
            "task_success": torch.tensor([False, False, True, False]),
            "qp_feasible": torch.tensor([True, True, True, False]),
        },
        elapsed_s=2.0,
        runtime_sample={
            "gpu_utilization_percent": 75.0,
            "gpu_uuid": "not-a-scalar",
        },
    )

    assert global_step == 100
    scalars = {(tag, step): value for tag, value, step in writer.scalars}
    assert scalars[("reward/step_mean/total_reward", 100)] == pytest.approx(4.0)
    assert scalars[
        ("reward/phase_step_mean/approach/total_reward", 100)
    ] == pytest.approx(2.0)
    assert scalars[
        ("reward/phase_step_mean/lift/total_reward", 100)
    ] == pytest.approx(6.0)
    assert scalars[("rollout/rate/phase_success", 100)] == pytest.approx(0.5)
    assert scalars[("system/live/gpu_utilization_percent", 100)] == 75.0
    assert ("system/live/gpu_uuid", 100) not in scalars

    logger.log_ppo_minibatch(
        optimizer_step=2,
        metrics={"actor_loss": 0.25, "approximate_kl": 0.01},
        runtime_sample={"gpu_memory_used_mib": 2048.0},
    )
    logger.log_ppo_update(
        metrics={"actor_loss": 0.20, "early_stopped_for_kl": False},
        environment_steps=32,
        wall_elapsed_s=4.0,
        runtime_load={"gpu_utilization_percent_mean": 60.0},
    )
    logger.close()

    scalars = {(tag, step): value for tag, value, step in writer.scalars}
    assert scalars[("ppo/minibatch/actor_loss", 3_000_002)] == 0.25
    assert scalars[("ppo/update/actor_loss", 3)] == 0.20
    assert scalars[("performance/ppo_environment_steps_per_s", 128)] == 8.0
    assert writer.flush_count >= 4
    assert writer.closed is True


def test_tensorboard_logger_rejects_reward_contract_drift(tmp_path: Path) -> None:
    logger = Order9TensorBoardLogger(
        tmp_path,
        stage_id="c2",
        generation_id="generation-0",
        split="train",
        update_index=0,
        generation_environment_steps=4,
        phase_labels=("approach",),
        reward_term_names=("expected",),
        writer_factory=_Writer,
    )

    with pytest.raises(ValueError, match="term names differ"):
        logger.log_rollout_step(
            rollout_index=0,
            reward=torch.ones(1),
            reward_terms={"unexpected": torch.ones(1)},
            phase_index=torch.zeros(1, dtype=torch.long),
            statuses={},
            elapsed_s=1.0,
        )
    logger.close()
