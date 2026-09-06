from __future__ import annotations

"""Deterministic R1 teacher heuristics for the v16 early collisions."""

from dataclasses import dataclass
import math
from typing import Mapping

from amsrr.schemas.common import SchemaValidationError

ORDER9_R1_COLLISION_AVOIDANCE_V17_VERSION = (
    "order9_r1_collision_avoidance_heuristic_v17"
)


@dataclass(frozen=True)
class Order9R1CollisionAvoidanceHeuristicV17:
    heuristic_id: str
    structural_hash_prefix: str
    sample_index: int
    diagnosed_phase: str
    diagnosed_body_name: str
    selected_anchor_body: bool
    selected_surface_port_ids: tuple[int, int]
    candidate_group_id: str
    pregrasp_clearance_m: float
    collision_margin_m: float
    grasp_contact_height_offset_m: float
    tangent_offset_world_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    approach_time_scale: float = 0.75
    free_joint_targets_rad: tuple[tuple[str, float], ...] = ()
    approach_unfold_fraction: tuple[float, float] | None = None

    def validate(self) -> None:
        if (
            not self.heuristic_id
            or len(self.structural_hash_prefix) != 12
            or any(
                value not in "0123456789abcdef" for value in self.structural_hash_prefix
            )
            or self.sample_index not in (0, 26)
            or self.diagnosed_phase not in ("approach", "contact_acquisition")
            or not self.diagnosed_body_name
            or len(set(self.selected_surface_port_ids)) != 2
            or min(self.selected_surface_port_ids) < 0
            or not self.candidate_group_id.startswith("slot_0:grasp_pair:")
            or not 0.08 <= self.pregrasp_clearance_m <= 0.30
            or not 0.005 <= self.collision_margin_m <= 0.030
            or not 0.0 <= self.grasp_contact_height_offset_m <= 0.08
            or len(self.tangent_offset_world_m) != 3
            or any(not math.isfinite(value) for value in self.tangent_offset_world_m)
            or math.dist(self.tangent_offset_world_m, (0.0, 0.0, 0.0)) > 0.03
            or self.approach_time_scale not in (0.75, 1.0)
            or len({joint_id for joint_id, _value in self.free_joint_targets_rad})
            != len(self.free_joint_targets_rad)
            or any(
                not joint_id.startswith("module_")
                or not math.isfinite(value)
                or abs(value) > 1.2
                for joint_id, value in self.free_joint_targets_rad
            )
            or (
                self.approach_unfold_fraction is not None
                and (
                    len(self.free_joint_targets_rad) != 1
                    or len(self.approach_unfold_fraction) != 2
                    or not 0.0
                    < self.approach_unfold_fraction[0]
                    < self.approach_unfold_fraction[1]
                    < 1.0
                )
            )
        ):
            raise SchemaValidationError("R1 collision-avoidance heuristic is invalid")

    @property
    def teacher_option(
        self,
    ) -> tuple[tuple[int, int], str, float, float, float, tuple[float, float, float]]:
        self.validate()
        return (
            self.selected_surface_port_ids,
            self.candidate_group_id,
            self.pregrasp_clearance_m,
            self.collision_margin_m,
            self.grasp_contact_height_offset_m,
            self.tangent_offset_world_m,
        )


_HEURISTICS = (
    # Port 13/21 is the first alternative pair that passes the complete
    # deterministic IK/path checks while avoiding the collision-prone 12/21
    # posture.  Keep this as a different teacher candidate, not as a runtime
    # correction of the rejected path.
    Order9R1CollisionAvoidanceHeuristicV17(
        heuristic_id="ba941_lattice00_alternate_port13_21",
        structural_hash_prefix="ba94152a03e4",
        sample_index=0,
        diagnosed_phase="approach",
        diagnosed_body_name="module_3__gimbal_link2",
        selected_anchor_body=False,
        selected_surface_port_ids=(13, 21),
        candidate_group_id="slot_0:grasp_pair:0",
        pregrasp_clearance_m=0.08,
        collision_margin_m=0.010,
        grasp_contact_height_offset_m=0.05,
        tangent_offset_world_m=(0.0, 0.02, 0.0),
    ),
    # The target conditions differ only by a planar scene transform, so the
    # same admitted pair and object-relative trajectory are transferred.
    Order9R1CollisionAvoidanceHeuristicV17(
        heuristic_id="ba941_lattice26_alternate_port13_21",
        structural_hash_prefix="ba94152a03e4",
        sample_index=26,
        diagnosed_phase="approach",
        diagnosed_body_name="module_5__pitch_dock_mech2",
        selected_anchor_body=True,
        selected_surface_port_ids=(13, 21),
        candidate_group_id="slot_0:grasp_pair:0",
        pregrasp_clearance_m=0.08,
        collision_margin_m=0.010,
        grasp_contact_height_offset_m=0.05,
        tangent_offset_world_m=(0.0, 0.02, 0.0),
    ),
    # Raising the group-2 contact posture by 10 mm did not remove the contact.
    # The alternate X-face assignment reduced it to about 0.52 N and placed
    # the offending body consistently on the object's +Y side.  Move both
    # contacts 20 mm along that face so the body stays outside the swept volume.
    *(
        Order9R1CollisionAvoidanceHeuristicV17(
            heuristic_id=f"b6b88_lattice{sample_index:02d}_tangent_y20",
            structural_hash_prefix="b6b88f01296d",
            sample_index=sample_index,
            diagnosed_phase="contact_acquisition",
            diagnosed_body_name="module_4__yaw_dock_mech2",
            selected_anchor_body=False,
            selected_surface_port_ids=(12, 27),
            candidate_group_id="slot_0:grasp_pair:0",
            pregrasp_clearance_m=0.08,
            collision_margin_m=0.010,
            grasp_contact_height_offset_m=0.03,
            tangent_offset_world_m=(0.0, 0.02, 0.0),
        )
        for sample_index in (0, 26)
    ),
)


def order9_r1_collision_avoidance_heuristics_v17() -> (
    Mapping[tuple[str, int], Order9R1CollisionAvoidanceHeuristicV17]
):
    result = {}
    for heuristic in _HEURISTICS:
        heuristic.validate()
        key = (heuristic.structural_hash_prefix, heuristic.sample_index)
        if key in result:
            raise SchemaValidationError("R1 collision heuristic identity is duplicated")
        result[key] = heuristic
    if len(result) != 4:
        raise SchemaValidationError("R1 collision heuristic set is incomplete")
    return result


__all__ = [
    "ORDER9_R1_COLLISION_AVOIDANCE_V17_VERSION",
    "Order9R1CollisionAvoidanceHeuristicV17",
    "order9_r1_collision_avoidance_heuristics_v17",
]
