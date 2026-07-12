from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

from amsrr.feasibility.morphology_flight import (
    MorphologyFlightFeasibilityChecker,
    MorphologyFlightFeasibilityConfig,
)
from amsrr.morphology.random_feasible import (
    RandomFeasibleConnectedMorphologyDistribution,
    RandomFeasibleMorphologyConfig,
)
from amsrr.robot_model.physical_model_builder import build_physical_model_from_config
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.order3 import (
    ORDER3_POOL_VERSION,
    Order3MorphologyPoolEntry,
    Order3MorphologyPoolManifest,
)
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.utils.hashing import stable_hash


@dataclass(frozen=True)
class Order3MorphologyPoolConfig:
    master_seed: int = 3000
    min_modules: int = 2
    max_modules: int = 8
    train_per_module_count: int = 8
    validation_per_module_count: int = 2
    held_out_per_module_count: int = 2
    two_module_train_per_module_count: int = 4
    max_attempts_per_sample: int = 256
    robot_model_config_path: str = "configs/robot/robot_model.yaml"
    mesh_search_dirs: tuple[str, ...] = ("module_urdf",)

    def __post_init__(self) -> None:
        if self.master_seed < 0:
            raise SchemaValidationError("Order3MorphologyPoolConfig.master_seed must be non-negative")
        if not 2 <= self.min_modules <= self.max_modules <= 8:
            raise SchemaValidationError(
                "Order3MorphologyPoolConfig requires 2 <= min_modules <= max_modules <= 8"
            )
        for name in (
            "train_per_module_count",
            "validation_per_module_count",
            "held_out_per_module_count",
            "two_module_train_per_module_count",
            "max_attempts_per_sample",
        ):
            if int(getattr(self, name)) <= 0:
                raise SchemaValidationError(f"Order3MorphologyPoolConfig.{name} must be positive")
        if not self.robot_model_config_path:
            raise SchemaValidationError(
                "Order3MorphologyPoolConfig.robot_model_config_path must be non-empty"
            )
        if not self.mesh_search_dirs:
            raise SchemaValidationError(
                "Order3MorphologyPoolConfig.mesh_search_dirs must not be empty"
            )
        if (
            self.two_module_train_per_module_count
            + self.validation_per_module_count
            + self.held_out_per_module_count
            > 8
        ):
            raise SchemaValidationError(
                "Order3MorphologyPoolConfig requests more than the eight canonical "
                "two-module morphologies"
            )

    @property
    def total_per_module_count(self) -> int:
        return (
            self.train_per_module_count
            + self.validation_per_module_count
            + self.held_out_per_module_count
        )

    def train_count(self, module_count: int) -> int:
        return (
            self.two_module_train_per_module_count
            if module_count == 2
            else self.train_per_module_count
        )

    def total_count(self, module_count: int) -> int:
        return (
            self.train_count(module_count)
            + self.validation_per_module_count
            + self.held_out_per_module_count
        )


def build_order3_morphology_pool(
    config: Order3MorphologyPoolConfig | None = None,
    *,
    physical_model: PhysicalModel | None = None,
    feasibility_checker: MorphologyFlightFeasibilityChecker | None = None,
) -> Order3MorphologyPoolManifest:
    cfg = config or Order3MorphologyPoolConfig()
    model = physical_model or build_physical_model_from_config(cfg.robot_model_config_path)
    checker = feasibility_checker or MorphologyFlightFeasibilityChecker(
        MorphologyFlightFeasibilityConfig(mesh_search_dirs=cfg.mesh_search_dirs)
    )
    distribution = RandomFeasibleConnectedMorphologyDistribution(
        model,
        feasibility_checker=checker,
        config=RandomFeasibleMorphologyConfig(
            max_attempts_per_sample=cfg.max_attempts_per_sample,
        ),
    )
    entries: list[Order3MorphologyPoolEntry] = []
    for module_count in range(cfg.min_modules, cfg.max_modules + 1):
        sampling_seed = _module_count_seed(cfg.master_seed, module_count)
        reports = distribution.sample_reports(
            seed=sampling_seed,
            count=cfg.total_count(module_count),
            module_count=module_count,
        )
        for index, report in enumerate(reports):
            split = _split_for_index(index, module_count, cfg)
            entries.append(
                Order3MorphologyPoolEntry(
                    split=split,
                    module_count=module_count,
                    structural_hash=report.structural_hash,
                    requested_seed=report.requested_seed,
                    accepted_proposal_seed=report.accepted_proposal_seed,
                    morphology_graph=report.morphology_graph,
                    feasibility_result=report.feasibility_result,
                    sampling_metadata={
                        "attempt_count": report.attempt_count,
                        "duplicate_rejection_count": report.duplicate_rejection_count,
                        "rejected_violation_counts": report.rejected_violation_counts,
                        "module_count_sampling_seed": sampling_seed,
                        "split_index": index,
                    },
                )
            )
    entries.sort(key=lambda item: (item.module_count, item.split.value, item.structural_hash))
    split_counts = {
        split.value: sum(entry.split == split for entry in entries)
        for split in DatasetSplit
    }
    module_count_counts = {
        str(module_count): sum(entry.module_count == module_count for entry in entries)
        for module_count in range(2, 9)
    }
    config_hash = stable_hash(cfg)
    return Order3MorphologyPoolManifest(
        pool_version=ORDER3_POOL_VERSION,
        master_seed=cfg.master_seed,
        physical_model_hash=model.stable_hash(),
        config_hash=config_hash,
        entries=entries,
        split_counts=split_counts,
        module_count_counts=module_count_counts,
        metadata={
            "split_unit": "canonical_structural_hash",
            "module_count_balanced": True,
            "min_modules": cfg.min_modules,
            "max_modules": cfg.max_modules,
            "train_per_module_count": cfg.train_per_module_count,
            "two_module_train_per_module_count": cfg.two_module_train_per_module_count,
            "validation_per_module_count": cfg.validation_per_module_count,
            "held_out_per_module_count": cfg.held_out_per_module_count,
            "max_attempts_per_sample": cfg.max_attempts_per_sample,
        },
    )


def write_order3_morphology_pool(
    manifest: Order3MorphologyPoolManifest,
    path: str | Path,
) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.to_json(indent=2) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return str(destination)


def _split_for_index(
    index: int,
    module_count: int,
    config: Order3MorphologyPoolConfig,
) -> DatasetSplit:
    train_count = config.train_count(module_count)
    if index < train_count:
        return DatasetSplit.TRAIN
    if index < train_count + config.validation_per_module_count:
        return DatasetSplit.VALIDATION
    return DatasetSplit.HELD_OUT


def _module_count_seed(master_seed: int, module_count: int) -> int:
    return master_seed * 1009 + module_count * 9176 + 31
