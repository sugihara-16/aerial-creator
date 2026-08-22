#!/usr/bin/env python3
from __future__ import annotations

"""Summarize rewards from one already-collected Order 9 C3 generation."""

import argparse
import gc
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from amsrr.training.order9_c3_stagewise_curriculum import (
    write_order9_c3_stagewise_json,
)
from amsrr.training.order9_tensor_rollout_artifact import (
    load_order9_tensor_rollout_artifact,
)
from amsrr.utils.hashing import hash_file


SUMMARY_CONTRACT = "order9_c3_collected_train_rollout_reward_summary_v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output")
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest_path = _resolve(args.dataset_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    generation_root = manifest_path.parent.parent
    output_path = (
        _resolve(args.output)
        if args.output is not None
        else generation_root / "training_rollout_reward_summary.json"
    )
    behavior_sha = str(manifest["behavior_checkpoint_sha256"])
    generation_id = str(manifest["generation_id"])
    generation_index = int(generation_id.rsplit(":", 1)[-1])
    shards = [
        value for value in manifest["shards"] if value["split"] == "train"
    ]
    if not shards:
        raise ValueError("Order 9 C3 generation contains no train shards")

    groups: dict[str, _Accumulator] = {
        "all_train": _Accumulator(),
        "phase_reset_train": _Accumulator(),
        "state_inheritance_train": _Accumulator(),
    }
    rows: list[dict[str, object]] = []
    reward_names: tuple[str, ...] | None = None
    for shard in shards:
        raw_path = Path(str(shard["path"])).resolve()
        artifact = load_order9_tensor_rollout_artifact(
            raw_path,
            expected_sha256=str(shard["sha256"]),
        )
        if artifact.metadata.get("pi_l_checkpoint_sha256") != behavior_sha:
            raise ValueError("C3 train rollout checkpoint binding differs")
        names = tuple(str(value) for value in artifact.metadata["reward_term_names"])
        if reward_names is None:
            reward_names = names
        elif names != reward_names:
            raise ValueError("C3 train rollout reward layout differs between shards")
        tensors = artifact.tensors
        valid = tensors["valid"]
        if int(valid.sum().item()) != int(shard["environment_step_count"]):
            raise ValueError("C3 train rollout valid-step count differs")
        row_accumulator = _Accumulator()
        row_accumulator.add(tensors=tensors, valid=valid, reward_names=names)
        row = row_accumulator.to_dict(reward_names=names)
        row.update(
            {
                "shard_id": raw_path.stem,
                "module_count": int(shard["module_count"]),
                "rollout_mode": str(shard.get("rollout_mode", "phase_reset")),
                "environment_count": int(shard["environment_count"]),
                "environment_step_count": int(shard["environment_step_count"]),
                "raw_artifact_path": str(raw_path),
                "raw_artifact_sha256": str(shard["sha256"]),
            }
        )
        rows.append(row)
        groups["all_train"].merge(row_accumulator)
        mode_group = (
            "state_inheritance_train"
            if shard.get("rollout_mode") == "continuous_state_inheritance"
            else "phase_reset_train"
        )
        groups[mode_group].merge(row_accumulator)
        module_group = f"module_{int(shard['module_count']):02d}_train"
        groups.setdefault(module_group, _Accumulator()).merge(row_accumulator)
        del artifact, tensors, valid
        gc.collect()

    assert reward_names is not None
    payload = {
        "contract": SUMMARY_CONTRACT,
        "generation_id": generation_id,
        "generation_index": generation_index,
        "evaluated_checkpoint_role": "pre_update_behavior_parent",
        "evaluated_behavior_checkpoint_sha256": behavior_sha,
        "evaluated_behavior_update_index": (
            None if generation_index == 0 else generation_index - 1
        ),
        "dataset_manifest_path": str(manifest_path),
        "dataset_manifest_sha256": hash_file(manifest_path),
        "bucket_cycle_index": generation_index % 2,
        "reward_term_names": list(reward_names),
        "aggregates": {
            name: value.to_dict(reward_names=reward_names)
            for name, value in sorted(groups.items())
        },
        "shards": rows,
    }
    write_order9_c3_stagewise_json(output_path, payload)
    console_summary = {
        "bucket_cycle_index": payload["bucket_cycle_index"],
        "evaluated_behavior_checkpoint_sha256": payload[
            "evaluated_behavior_checkpoint_sha256"
        ],
        "evaluated_behavior_update_index": payload[
            "evaluated_behavior_update_index"
        ],
        "generation_index": payload["generation_index"],
        "module_aggregates": {
            name: {
                "phase_success_rate": values["phase_success_rate"],
                "reward_mean": values["reward_mean"],
            }
            for name, values in payload["aggregates"].items()
            if name.startswith("module_")
        },
    }
    print(
        "ORDER9_C3_GENERATION_REWARD="
        + json.dumps(console_summary, sort_keys=True)
    )
    print(f"ORDER9_C3_GENERATION_REWARD_PATH={output_path}")
    return 0


class _Accumulator:
    def __init__(self) -> None:
        self.count = 0
        self.reward_sum = 0.0
        self.reward_square_sum = 0.0
        self.reward_term_sums: list[float] | None = None
        self.rate_sums = {
            "phase_success_rate": 0.0,
            "qp_feasible_rate": 0.0,
            "prohibited_collision_rate": 0.0,
            "rotor_saturation_rate": 0.0,
            "terminal_step_rate": 0.0,
            "truncated_step_rate": 0.0,
            "actor_task_success_rate": 0.0,
        }
        self.terminal_count = 0
        self.successful_terminal_count = 0

    def add(self, *, tensors, valid, reward_names: tuple[str, ...]) -> None:
        import torch

        count = int(valid.sum().item())
        reward = tensors["reward"][valid].to(dtype=torch.float64)
        reward_terms = tensors["reward_terms"][valid].to(dtype=torch.float64)
        if reward_terms.shape[-1] != len(reward_names):
            raise ValueError("C3 train rollout reward tensor layout differs")
        self.count += count
        self.reward_sum += float(reward.sum().item())
        self.reward_square_sum += float(reward.square().sum().item())
        term_sums = [float(value) for value in reward_terms.sum(dim=0).tolist()]
        if self.reward_term_sums is None:
            self.reward_term_sums = term_sums
        else:
            self.reward_term_sums = [
                left + right
                for left, right in zip(self.reward_term_sums, term_sums, strict=True)
            ]
        for output_name, tensor_name in (
            ("phase_success_rate", "phase_success"),
            ("qp_feasible_rate", "qp_feasible"),
            ("prohibited_collision_rate", "prohibited_collision"),
            ("rotor_saturation_rate", "rotor_saturation"),
            ("terminal_step_rate", "terminal"),
            ("truncated_step_rate", "truncated"),
            ("actor_task_success_rate", "actor_task_success"),
        ):
            self.rate_sums[output_name] += float(
                tensors[tensor_name][valid].to(dtype=torch.float64).sum().item()
            )
        terminal = tensors["terminal"][valid]
        self.terminal_count += int(terminal.sum().item())
        self.successful_terminal_count += int(
            (terminal & tensors["actor_task_success"][valid]).sum().item()
        )

    def merge(self, other: "_Accumulator") -> None:
        self.count += other.count
        self.reward_sum += other.reward_sum
        self.reward_square_sum += other.reward_square_sum
        if other.reward_term_sums is not None:
            if self.reward_term_sums is None:
                self.reward_term_sums = list(other.reward_term_sums)
            else:
                self.reward_term_sums = [
                    left + right
                    for left, right in zip(
                        self.reward_term_sums,
                        other.reward_term_sums,
                        strict=True,
                    )
                ]
        for name, value in other.rate_sums.items():
            self.rate_sums[name] += value
        self.terminal_count += other.terminal_count
        self.successful_terminal_count += other.successful_terminal_count

    def to_dict(self, *, reward_names: tuple[str, ...]) -> dict[str, object]:
        if self.count <= 0 or self.reward_term_sums is None:
            raise ValueError("C3 train rollout reward accumulator is empty")
        mean = self.reward_sum / self.count
        variance = max(0.0, self.reward_square_sum / self.count - mean * mean)
        return {
            "environment_step_count": self.count,
            "reward_mean": mean,
            "reward_std": variance**0.5,
            "reward_term_means": {
                name: value / self.count
                for name, value in zip(
                    reward_names, self.reward_term_sums, strict=True
                )
            },
            **{
                name: value / self.count
                for name, value in self.rate_sums.items()
            },
            "terminal_count": self.terminal_count,
            "successful_terminal_count": self.successful_terminal_count,
            "terminal_success_rate": (
                0.0
                if self.terminal_count == 0
                else self.successful_terminal_count / self.terminal_count
            ),
        }


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = REPOSITORY_ROOT / value
    return value.resolve()


if __name__ == "__main__":
    raise SystemExit(main())
