from __future__ import annotations

"""Execution-error-aware R1 support-clearance preparation contract."""

from dataclasses import dataclass
import math

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.order9_c3_nominal_trajectory import (
    Order9C3NominalTrajectory,
)
from amsrr.training.order9_r1_fast_screen import Order9R1FastScreenResult
from amsrr.training.order9_r1_support_clearance import (
    build_order9_r1_frozen_support_collision_object,
    generate_order9_r1_frozen_support_nominal_grasp_trajectory,
)

ORDER9_R1_EXECUTION_CLEARANCE_VERSION = (
    "order9_r1_frozen_support_execution_error_clearance_v3"
)
ORDER9_R1_REQUIRED_RETAINED_CLEARANCE_M = 0.005
ORDER9_R1_MEASURED_NOMINAL_TRACKING_CLEARANCE_CONSUMPTION_M = (
    0.011802887795739854 - 0.0033279385195250506
)
ORDER9_R1_MEASURED_PI_L_CLEARANCE_CONSUMPTION_M = 0.006825809539512066
ORDER9_R1_MEASURED_MINIMUM_REQUIRED_CLEARANCE_M = (
    ORDER9_R1_REQUIRED_RETAINED_CLEARANCE_M
    + ORDER9_R1_MEASURED_NOMINAL_TRACKING_CLEARANCE_CONSUMPTION_M
    + ORDER9_R1_MEASURED_PI_L_CLEARANCE_CONSUMPTION_M
)
# 20.301 mm is rounded upward past the existing +20 mm posture alternative,
# so the planner must actually select a more separated solution when the
# nominal contact posture cannot retain the measured execution reserve.
ORDER9_R1_EXECUTION_ROBUST_CLEARANCE_M = 0.022
ORDER9_R1_COLLISION_FEASIBILITY_TOLERANCE_M = 0.0002


@dataclass(frozen=True)
class Order9R1ExecutionClearanceContract:
    version: str
    support_geometry_hash: str
    requested_clearance_m: float
    minimum_admissible_clearance_m: float
    retained_clearance_m: float
    measured_nominal_tracking_consumption_m: float
    measured_pi_l_consumption_m: float
    measured_minimum_required_clearance_m: float

    def __post_init__(self) -> None:
        if self.version != ORDER9_R1_EXECUTION_CLEARANCE_VERSION:
            raise ValueError("R1 execution-clearance version mismatch")
        if len(self.support_geometry_hash) != 64:
            raise ValueError("R1 execution-clearance support hash is invalid")
        values = (
            self.requested_clearance_m,
            self.minimum_admissible_clearance_m,
            self.retained_clearance_m,
            self.measured_nominal_tracking_consumption_m,
            self.measured_pi_l_consumption_m,
            self.measured_minimum_required_clearance_m,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("R1 execution-clearance values must be positive")
        if (
            self.requested_clearance_m
            < self.measured_minimum_required_clearance_m
            or self.minimum_admissible_clearance_m > self.requested_clearance_m
        ):
            raise ValueError("R1 execution-clearance reserve is insufficient")


def build_order9_r1_execution_clearance_contract(
    *, source_task_spec: TaskSpec, randomized_task_spec: TaskSpec
):
    collision, frozen = build_order9_r1_frozen_support_collision_object(
        source_task_spec=source_task_spec,
        randomized_task_spec=randomized_task_spec,
    )
    contract = Order9R1ExecutionClearanceContract(
        version=ORDER9_R1_EXECUTION_CLEARANCE_VERSION,
        support_geometry_hash=frozen.support_geometry_hash,
        requested_clearance_m=ORDER9_R1_EXECUTION_ROBUST_CLEARANCE_M,
        minimum_admissible_clearance_m=(
            ORDER9_R1_EXECUTION_ROBUST_CLEARANCE_M
            - ORDER9_R1_COLLISION_FEASIBILITY_TOLERANCE_M
        ),
        retained_clearance_m=ORDER9_R1_REQUIRED_RETAINED_CLEARANCE_M,
        measured_nominal_tracking_consumption_m=(
            ORDER9_R1_MEASURED_NOMINAL_TRACKING_CLEARANCE_CONSUMPTION_M
        ),
        measured_pi_l_consumption_m=(
            ORDER9_R1_MEASURED_PI_L_CLEARANCE_CONSUMPTION_M
        ),
        measured_minimum_required_clearance_m=(
            ORDER9_R1_MEASURED_MINIMUM_REQUIRED_CLEARANCE_M
        ),
    )
    return collision, contract


def generate_order9_r1_v3_nominal_grasp_trajectory(
    *,
    source_task_spec: TaskSpec,
    randomized_task_spec: TaskSpec,
    structural_target: MorphologyGraph,
    physical_model: PhysicalModel,
    preferred_surface_port_ids: tuple[int, int] | None = None,
) -> tuple[Order9C3NominalTrajectory, Order9R1ExecutionClearanceContract]:
    collision, contract = build_order9_r1_execution_clearance_contract(
        source_task_spec=source_task_spec,
        randomized_task_spec=randomized_task_spec,
    )
    nominal = generate_order9_r1_frozen_support_nominal_grasp_trajectory(
        randomized_task_spec=randomized_task_spec,
        structural_target=structural_target,
        physical_model=physical_model,
        collision_object=collision,
        preferred_surface_port_ids=preferred_surface_port_ids,
        collision_margin_m=contract.requested_clearance_m,
    )
    return nominal, contract


def require_order9_r1_v3_execution_clearance(
    *,
    screen: Order9R1FastScreenResult,
    contract: Order9R1ExecutionClearanceContract,
) -> None:
    clearance = screen.minimum_collision_clearance_m
    if (
        not screen.accepted
        or not screen.eligible_for_full_control_test
        or clearance is None
        or not math.isfinite(float(clearance))
        or float(clearance) + 1.0e-12 < contract.minimum_admissible_clearance_m
    ):
        raise SchemaValidationError(
            "R1 candidate lacks measured execution-error support clearance"
        )


__all__ = [
    "ORDER9_R1_EXECUTION_CLEARANCE_VERSION",
    "ORDER9_R1_EXECUTION_ROBUST_CLEARANCE_M",
    "ORDER9_R1_MEASURED_MINIMUM_REQUIRED_CLEARANCE_M",
    "Order9R1ExecutionClearanceContract",
    "build_order9_r1_execution_clearance_contract",
    "generate_order9_r1_v3_nominal_grasp_trajectory",
    "require_order9_r1_v3_execution_clearance",
]
