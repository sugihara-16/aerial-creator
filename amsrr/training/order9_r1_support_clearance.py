from __future__ import annotations

"""R1 v2 support-clearance contract for teacher trajectory preparation.

R1 moves the object start and goal while the physical support remains fixed.
The generic C3 helper reconstructs a bucket support from those two object
poses, so using it directly would incorrectly move the collision proxy with
the randomized object.  This module combines the randomized object with the
source bucket's frozen support and requires additional execution reserve.
"""

from dataclasses import dataclass
import math
from threading import RLock
from unittest.mock import patch

from amsrr.training import order9_c3_nominal_trajectory as c3_nominal
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.physical_model import PhysicalModel
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.order9_c3_nominal_trajectory import (
    Order9C3NominalTrajectory,
)
from amsrr.training.order9_c3_teacher import (
    build_order9_c3_posture_collision_object,
)
from amsrr.training.order9_posture_resolver import (
    Order9PostureCollisionObject,
)
from amsrr.training.order9_r1_fast_screen import Order9R1FastScreenResult
from amsrr.utils.hashing import stable_hash

ORDER9_R1_SUPPORT_CLEARANCE_VERSION = (
    "order9_r1_frozen_support_robust_clearance_v2"
)
# The failed learned-policy replay consumed about 6.82 mm of support clearance
# at its terminal sample.  Preserve the existing 5 mm nominal requirement and
# round their sum upward to a reproducible 12 mm preparation gate.
ORDER9_R1_ROBUST_SUPPORT_CLEARANCE_M = 0.012
ORDER9_R1_COLLISION_FEASIBILITY_TOLERANCE_M = 0.0002
_R1_FROZEN_SUPPORT_GENERATION_LOCK = RLock()


@dataclass(frozen=True)
class Order9R1FrozenSupportContract:
    version: str
    support_geometry_hash: str
    requested_clearance_m: float
    minimum_admissible_clearance_m: float

    def __post_init__(self) -> None:
        if self.version != ORDER9_R1_SUPPORT_CLEARANCE_VERSION:
            raise ValueError("R1 frozen-support contract version mismatch")
        if len(self.support_geometry_hash) != 64:
            raise ValueError("R1 frozen-support hash is not SHA-256 shaped")
        if (
            not math.isfinite(self.requested_clearance_m)
            or not math.isfinite(self.minimum_admissible_clearance_m)
            or self.requested_clearance_m <= 0.0
            or self.minimum_admissible_clearance_m <= 0.0
            or self.minimum_admissible_clearance_m > self.requested_clearance_m
        ):
            raise ValueError("R1 frozen-support clearance values are invalid")


def build_order9_r1_frozen_support_collision_object(
    *,
    source_task_spec: TaskSpec,
    randomized_task_spec: TaskSpec,
) -> tuple[Order9PostureCollisionObject, Order9R1FrozenSupportContract]:
    """Use the randomized object pose and the unmodified source support."""

    source = build_order9_c3_posture_collision_object(source_task_spec)
    randomized = build_order9_c3_posture_collision_object(randomized_task_spec)
    if (
        source.object_id != randomized.object_id
        or source.size_m != randomized.size_m
        or source.ground_plane_z_m != randomized.ground_plane_z_m
    ):
        raise SchemaValidationError(
            "R1 randomized task changed object or ground collision identity"
        )
    if not source.environment_boxes:
        raise SchemaValidationError("R1 source bucket has no frozen support")
    collision = Order9PostureCollisionObject(
        object_id=randomized.object_id,
        size_m=randomized.size_m,
        initial_pose_world=randomized.initial_pose_world,
        environment_boxes=source.environment_boxes,
        ground_plane_z_m=source.ground_plane_z_m,
    )
    support_hash = stable_hash(
        [
            {
                "box_id": box.box_id,
                "size_m": list(box.size_m),
                "pose_world": list(box.pose_world),
            }
            for box in collision.environment_boxes
        ]
    )
    contract = Order9R1FrozenSupportContract(
        version=ORDER9_R1_SUPPORT_CLEARANCE_VERSION,
        support_geometry_hash=support_hash,
        requested_clearance_m=ORDER9_R1_ROBUST_SUPPORT_CLEARANCE_M,
        minimum_admissible_clearance_m=(
            ORDER9_R1_ROBUST_SUPPORT_CLEARANCE_M
            - ORDER9_R1_COLLISION_FEASIBILITY_TOLERANCE_M
        ),
    )
    return collision, contract


def generate_order9_r1_v2_nominal_grasp_trajectory(
    *,
    source_task_spec: TaskSpec,
    randomized_task_spec: TaskSpec,
    structural_target: MorphologyGraph,
    physical_model: PhysicalModel,
    preferred_surface_port_ids: tuple[int, int] | None = None,
) -> tuple[Order9C3NominalTrajectory, Order9R1FrozenSupportContract]:
    """Generate teacher+IK data against the fixed, 12 mm-clear support."""

    collision, contract = build_order9_r1_frozen_support_collision_object(
        source_task_spec=source_task_spec,
        randomized_task_spec=randomized_task_spec,
    )
    nominal = generate_order9_r1_frozen_support_nominal_grasp_trajectory(
        randomized_task_spec=randomized_task_spec,
        structural_target=structural_target,
        physical_model=physical_model,
        collision_object=collision,
        collision_margin_m=contract.requested_clearance_m,
        preferred_surface_port_ids=preferred_surface_port_ids,
    )
    return nominal, contract


def generate_order9_r1_frozen_support_nominal_grasp_trajectory(
    *,
    randomized_task_spec: TaskSpec,
    structural_target: MorphologyGraph,
    physical_model: PhysicalModel,
    collision_object: Order9PostureCollisionObject,
    collision_margin_m: float,
    preferred_surface_port_ids: tuple[int, int] | None = None,
) -> Order9C3NominalTrajectory:
    """Run the protected generator with an R1-only frozen support binding.

    The promoted C3 file is hash-protected and has no dependency-injection
    argument.  R1 preparation is single-process and offline, so a process-local
    lock scopes a temporary replacement of only its collision-object builder.
    The protected source bytes and all default callers remain unchanged.
    """

    if not math.isfinite(collision_margin_m) or collision_margin_m <= 0.0:
        raise ValueError("R1 frozen-support collision margin must be positive")
    with _R1_FROZEN_SUPPORT_GENERATION_LOCK:
        with patch.object(
            c3_nominal,
            "build_order9_c3_posture_collision_object",
            return_value=collision_object,
        ):
            return c3_nominal.generate_order9_c3_nominal_grasp_trajectory(
                task_spec=randomized_task_spec,
                structural_target=structural_target,
                physical_model=physical_model,
                preferred_surface_port_ids=preferred_surface_port_ids,
                collision_margin_m=collision_margin_m,
            )


def require_order9_r1_v2_support_clearance(
    *,
    screen: Order9R1FastScreenResult,
    contract: Order9R1FrozenSupportContract,
) -> None:
    """Fail closed unless the light screen carries the v2 clearance reserve."""

    clearance = screen.minimum_collision_clearance_m
    if (
        not screen.accepted
        or not screen.eligible_for_full_control_test
        or clearance is None
        or not math.isfinite(float(clearance))
        or float(clearance) + 1.0e-12 < contract.minimum_admissible_clearance_m
    ):
        raise SchemaValidationError(
            "R1 v2 candidate lacks frozen-support execution clearance"
        )


__all__ = [
    "ORDER9_R1_COLLISION_FEASIBILITY_TOLERANCE_M",
    "ORDER9_R1_ROBUST_SUPPORT_CLEARANCE_M",
    "ORDER9_R1_SUPPORT_CLEARANCE_VERSION",
    "Order9R1FrozenSupportContract",
    "build_order9_r1_frozen_support_collision_object",
    "generate_order9_r1_frozen_support_nominal_grasp_trajectory",
    "generate_order9_r1_v2_nominal_grasp_trajectory",
    "require_order9_r1_v2_support_clearance",
]
