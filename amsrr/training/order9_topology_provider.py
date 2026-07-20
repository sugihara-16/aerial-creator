from __future__ import annotations

"""Split-safe topology buckets before learned ``pi_D`` becomes active."""

import random
from dataclasses import dataclass, field
from pathlib import Path

from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.morphology.random_connected import morphology_structural_hash
from amsrr.policies.design_policy_base import DesignPolicyContext
from amsrr.schemas.common import SchemaBase, SchemaValidationError
from amsrr.schemas.datasets import DatasetSplit
from amsrr.schemas.feasibility import FeasibilityResult
from amsrr.schemas.morphology import DesignOutput, MorphologyGraph
from amsrr.schemas.order3 import Order3MorphologyPoolEntry, Order3MorphologyPoolManifest
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.order9_design_teacher_dataset import (
    ORDER9_PI_D_TEACHER_VERSION,
    build_order9_task_conditioned_design_teacher,
)
from amsrr.utils.hashing import hash_file, stable_hash


ORDER9_TOPOLOGY_PROVIDER_VERSION = "order9_split_safe_topology_provider_v1"


@dataclass
class Order9TopologySample(SchemaBase):
    split: DatasetSplit
    sample_index: int
    seed: int
    module_count: int
    source_structural_hash: str
    source_morphology_hash: str
    task_hash: str
    design_output: DesignOutput
    feasibility_result: FeasibilityResult
    provider_version: str = ORDER9_TOPOLOGY_PROVIDER_VERSION
    metadata: dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if self.provider_version != ORDER9_TOPOLOGY_PROVIDER_VERSION:
            raise SchemaValidationError("Order9 topology provider version mismatch")
        if self.sample_index < 0 or self.seed < 0:
            raise SchemaValidationError("Order9 topology seed/index must be non-negative")
        if not 2 <= self.module_count <= 8:
            raise SchemaValidationError("Order9 topology module count must be in [2, 8]")
        for name in ("source_structural_hash", "source_morphology_hash", "task_hash"):
            _require_sha256(str(getattr(self, name)), name)
        graph = self.design_output.target_morphology
        if len(graph.modules) != self.module_count:
            raise SchemaValidationError("Order9 topology sample module count changed")
        if morphology_structural_hash(graph) != self.source_structural_hash:
            raise SchemaValidationError("Order9 topology sample structure changed")
        if not self.feasibility_result.feasible:
            raise SchemaValidationError("Order9 topology sample must pass deterministic feasibility")

    @property
    def morphology_graph(self) -> MorphologyGraph:
        return self.design_output.target_morphology


class Order9CurriculumTopologyProvider:
    """Select a pool structure, then add anchors through the production grammar."""

    def __init__(
        self,
        pool: Order3MorphologyPoolManifest,
        *,
        physical_model: PhysicalModel,
        pool_path: str | Path | None = None,
        pool_sha256: str | None = None,
    ) -> None:
        pool.validate()
        physical_model.validate()
        if pool.physical_model_hash != physical_model.stable_hash():
            raise SchemaValidationError("Order9 topology pool physical model is stale")
        if pool_path is not None:
            source = Path(pool_path)
            actual = hash_file(source)
            if pool_sha256 is not None and actual != pool_sha256:
                raise SchemaValidationError("Order9 topology pool byte hash mismatch")
            self.pool_reference = str(source.resolve())
            self.pool_sha256 = actual
        else:
            self.pool_reference = "in_memory:order3_morphology_pool"
            self.pool_sha256 = stable_hash(pool.to_dict())
        self.pool = pool
        self.physical_model = physical_model
        self._entries = tuple(pool.entries)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        physical_model: PhysicalModel,
        expected_sha256: str | None = None,
    ) -> "Order9CurriculumTopologyProvider":
        source = Path(path)
        pool = Order3MorphologyPoolManifest.from_json(
            source.read_text(encoding="utf-8")
        )
        return cls(
            pool,
            physical_model=physical_model,
            pool_path=source,
            pool_sha256=expected_sha256,
        )

    def sample(
        self,
        task_spec: TaskSpec,
        *,
        split: DatasetSplit,
        seed: int,
        sample_index: int,
        min_modules: int = 2,
        max_modules: int = 8,
        module_count: int | None = None,
    ) -> Order9TopologySample:
        if seed < 0 or sample_index < 0:
            raise ValueError("Order9 topology seed/index must be non-negative")
        if not 2 <= min_modules <= max_modules <= 8:
            raise ValueError("Order9 topology module bounds must lie in [2, 8]")
        rng = random.Random(_selection_seed(seed, sample_index, task_spec.stable_hash()))
        selected_count = (
            rng.randint(min_modules, max_modules)
            if module_count is None
            else int(module_count)
        )
        if not min_modules <= selected_count <= max_modules:
            raise ValueError("Order9 requested module count lies outside the stage")
        candidates = [
            entry
            for entry in self._entries
            if entry.split == split and entry.module_count == selected_count
        ]
        if not candidates:
            raise SchemaValidationError(
                "Order9 topology pool has no entry for the requested split/module count"
            )
        candidates.sort(key=lambda item: item.structural_hash)
        entry = candidates[rng.randrange(len(candidates))]
        return self._task_conditioned_sample(
            task_spec,
            entry=entry,
            split=split,
            seed=seed,
            sample_index=sample_index,
        )

    def _task_conditioned_sample(
        self,
        task_spec: TaskSpec,
        *,
        entry: Order3MorphologyPoolEntry,
        split: DatasetSplit,
        seed: int,
        sample_index: int,
    ) -> Order9TopologySample:
        irg = IRGBuilder().build(task_spec)
        envelope = InteractionEnvelopeExtractor().extract(irg)
        context = DesignPolicyContext(
            task_spec,
            irg,
            self.physical_model,
            envelope,
        )
        _, design, feasibility = build_order9_task_conditioned_design_teacher(
            context,
            entry.morphology_graph,
        )
        sample = Order9TopologySample(
            split=split,
            sample_index=sample_index,
            seed=seed,
            module_count=entry.module_count,
            source_structural_hash=entry.structural_hash,
            source_morphology_hash=entry.morphology_graph.stable_hash(),
            task_hash=task_spec.stable_hash(),
            design_output=design,
            feasibility_result=feasibility,
            metadata={
                "provider_version": ORDER9_TOPOLOGY_PROVIDER_VERSION,
                "source_pool_path": self.pool_reference,
                "source_pool_sha256": self.pool_sha256,
                "source_pool_version": self.pool.pool_version,
                "source_requested_seed": entry.requested_seed,
                "source_accepted_proposal_seed": entry.accepted_proposal_seed,
                "task_conditioning_teacher": ORDER9_PI_D_TEACHER_VERSION,
                "structural_split_owner": split.value,
                "learned_pi_d_used": False,
            },
        )
        sample.validate()
        return sample


def _selection_seed(seed: int, sample_index: int, task_hash: str) -> int:
    return int(
        stable_hash(
            {
                "seed": seed,
                "sample_index": sample_index,
                "task_hash": task_hash,
                "provider_version": ORDER9_TOPOLOGY_PROVIDER_VERSION,
            }
        )[:16],
        16,
    )


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise SchemaValidationError(f"Order9 topology {name} is not SHA-256")


__all__ = [
    "ORDER9_TOPOLOGY_PROVIDER_VERSION",
    "Order9CurriculumTopologyProvider",
    "Order9TopologySample",
]
