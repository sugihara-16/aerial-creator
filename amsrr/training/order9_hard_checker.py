from __future__ import annotations

"""Factory binding the approved production C_H configuration to runtime."""

from amsrr.feasibility.contact_wrench_hybrid import (
    HybridContactWrenchPhysicsEvaluator,
    LightweightContactQPConfig,
    LightweightContactWrenchQPEvaluator,
    ShadowTrajectoryRolloutBackend,
)
from amsrr.feasibility.contact_wrench_trajectory import (
    ContactWrenchTrajectoryCheckerConfig,
    ContactWrenchTrajectoryFeasibilityChecker,
)
from amsrr.training.order9_curriculum import Order9HardCheckerConfig


def build_order9_production_hard_checker(
    shadow_backend: ShadowTrajectoryRolloutBackend,
    *,
    config: Order9HardCheckerConfig | None = None,
) -> ContactWrenchTrajectoryFeasibilityChecker:
    resolved = config or Order9HardCheckerConfig()
    resolved.validate()
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
        ),
        physics_evaluator=evaluator,
    )


__all__ = ["build_order9_production_hard_checker"]
