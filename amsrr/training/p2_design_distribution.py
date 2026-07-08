from __future__ import annotations

from dataclasses import dataclass

from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.p1_task_distribution import (
    P1GraspCarryTaskDistribution,
    P1TaskDistributionConfig,
    P1TaskSample,
)
from amsrr.utils.config import load_config


@dataclass
class P2DesignDistributionConfig(P1TaskDistributionConfig):
    """P2 grasp/carry design distribution.

    The field set intentionally matches P1 randomization for now so P2 can
    reuse the same TaskSpec perturbations before policy/controller simulation.
    """


@dataclass
class P2DesignTaskSample(P1TaskSample):
    pass


def load_p2_design_distribution_config(path: str) -> P2DesignDistributionConfig:
    data = load_config(path)
    return P2DesignDistributionConfig.from_dict(data.get("distribution", data))


class P2GraspCarryDesignDistribution:
    def __init__(
        self,
        base_task_spec: TaskSpec,
        config: P2DesignDistributionConfig | None = None,
    ) -> None:
        self.base_task_spec = base_task_spec
        self.config = config or P2DesignDistributionConfig()
        self._p1_distribution = P1GraspCarryTaskDistribution(base_task_spec, self.config)

    def sample(self, *, seed: int, sample_index: int = 0) -> P2DesignTaskSample:
        p1_sample = self._p1_distribution.sample(seed=seed, sample_index=sample_index)
        task_data = p1_sample.task_spec.to_dict()
        task_data["task_id"] = f"{self.base_task_spec.task_id}_p2_{sample_index:04d}"
        metadata = dict(task_data.get("metadata", {}) or {})
        metadata["randomization_family"] = "p2_design_grasp_carry"
        metadata["design_evaluation_phase"] = "P2"
        task_data["metadata"] = metadata
        return P2DesignTaskSample(
            task_spec=TaskSpec.from_dict(task_data),
            seed=seed,
            sample_index=sample_index,
            sampled_values=p1_sample.sampled_values,
        )
