from __future__ import annotations

"""R1 range preparation using one support and raised side contacts."""

from threading import RLock
from unittest.mock import patch

from amsrr.training import order9_c3_nominal_trajectory as c3_nominal
from amsrr.training import order9_c3_teacher as c3_teacher
from amsrr.training.order9_r1_calibration_runner import (
    Order9R1CalibrationCase,
    Order9R1PreparedCandidate,
)
from amsrr.training.order9_r1_nominal_geometry_v11 import (
    Order9R1NominalGeometryV11Contract,
    build_order9_r1_explicit_support_collision_object_v11,
)
from amsrr.training.order9_r1_range_selection_v10 import (
    Order9R1RangeTeacherScreenPipelineV10,
)

ORDER9_R1_RANGE_SELECTION_V11_VERSION = "order9_r1_range_selection_v11"
_R1_V11_GENERATION_LOCK = RLock()


class Order9R1RangeTeacherScreenPipelineV11(Order9R1RangeTeacherScreenPipelineV10):
    """Generate against the Isaac support and raise every side grasp >=20 mm."""

    pipeline_version = ORDER9_R1_RANGE_SELECTION_V11_VERSION

    def __init__(
        self,
        *,
        geometry_contract: Order9R1NominalGeometryV11Contract,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.geometry_contract = geometry_contract

    def prepare(self, case: Order9R1CalibrationCase) -> Order9R1PreparedCandidate:
        collision = build_order9_r1_explicit_support_collision_object_v11(
            case.task_spec,
            contract=self.geometry_contract,
        )
        original_offset = c3_teacher._offset_horizontal_grasp_candidates

        def raised_offset(candidate_set, *, height_offset_m, tangent_offset_world_m):
            return original_offset(
                candidate_set,
                height_offset_m=max(
                    float(height_offset_m),
                    self.geometry_contract.grasp_contact_height_offset_m,
                ),
                tangent_offset_world_m=tangent_offset_world_m,
            )

        # The historical generator imports its support builder at module load.
        # Keep its protected bytes unchanged and bind the R1-only call under a
        # process-local lock.
        with _R1_V11_GENERATION_LOCK:
            with patch.object(
                c3_nominal,
                "build_order9_c3_posture_collision_object",
                return_value=collision,
            ), patch.object(
                c3_teacher,
                "_offset_horizontal_grasp_candidates",
                side_effect=raised_offset,
            ):
                return super().prepare(case)


__all__ = [
    "ORDER9_R1_RANGE_SELECTION_V11_VERSION",
    "Order9R1RangeTeacherScreenPipelineV11",
]
