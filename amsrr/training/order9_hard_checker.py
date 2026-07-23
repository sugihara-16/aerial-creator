from __future__ import annotations

"""Factory binding the approved production C_H configuration to runtime."""

from amsrr.feasibility.contact_wrench_hybrid import (
    HybridContactWrenchPhysicsEvaluator,
    LightweightContactQPConfig,
    LightweightContactWrenchQPEvaluator,
    ShadowTrajectoryRolloutBackend,
)
from amsrr.feasibility.articulated_reachability import (
    ArticulatedTrajectoryReachabilityEvaluator,
)
from amsrr.feasibility.contact_wrench_trajectory import (
    ContactWrenchTrajectoryCheckerConfig,
    ContactWrenchTrajectoryFeasibilityChecker,
)
from amsrr.training.order9_curriculum import Order9HardCheckerConfig
from amsrr.robot_model.physical_model_builder import (
    build_physical_model_from_config,
)
from amsrr.schemas.physical_model import PhysicalModel


def build_order9_production_hard_checker(
    shadow_backend: ShadowTrajectoryRolloutBackend,
    *,
    config: Order9HardCheckerConfig | None = None,
    physical_model: PhysicalModel | None = None,
) -> ContactWrenchTrajectoryFeasibilityChecker:
    resolved = config or Order9HardCheckerConfig()
    resolved.validate()
    resolved_physical_model = physical_model or build_physical_model_from_config(
        "configs/robot/robot_model.yaml"
    )
    qp = LightweightContactWrenchQPEvaluator(
        LightweightContactQPConfig(
            force_scale_n=resolved.qp_force_scale_n,
            torque_scale_nm=resolved.qp_torque_scale_nm,
            solver_absolute_tolerance=resolved.qp_solver_absolute_tolerance,
            solver_relative_tolerance=resolved.qp_solver_relative_tolerance,
            solver_max_iterations=resolved.qp_solver_max_iterations,
        )
    )
    evaluator = HybridContactWrenchPhysicsEvaluator(
        shadow_backend=shadow_backend,
        qp_evaluator=qp,
    )
    return ContactWrenchTrajectoryFeasibilityChecker(
        config=ContactWrenchTrajectoryCheckerConfig(
            evaluation_mode="production",
            qp_residual_threshold=resolved.qp_residual_threshold,
            wrench_residual_threshold=resolved.wrench_residual_threshold,
            require_reachability_evaluation=True,
        ),
        physics_evaluator=evaluator,
        reachability_evaluator=ArticulatedTrajectoryReachabilityEvaluator(
            resolved_physical_model
        ),
    )


__all__ = ["build_order9_production_hard_checker"]
