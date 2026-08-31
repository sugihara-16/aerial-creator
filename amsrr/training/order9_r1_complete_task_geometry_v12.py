from __future__ import annotations

"""Conservative two-stage acceleration for the R1 eight-phase mesh audit."""

import math
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import patch

from amsrr.feasibility.order9_posture_collision import (
    CollisionAwareNativeCentroidalPostureIKSolver,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.contact_candidates import ContactCandidateSet
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.policies import ContactWrenchTrajectory
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training import order9_r1_complete_task_geometry_v11 as geometry_v11
from amsrr.training.order9_r1_complete_task_audit import (
    ORDER9_R1_COMPLETE_TASK_PHASES,
)
from amsrr.training.order9_r1_complete_task_geometry_v11 import (
    audit_order9_r1_complete_task_geometry_v11,
)
from amsrr.training.order9_r1_nominal_geometry_v11 import (
    Order9R1NominalGeometryV11Contract,
)

ORDER9_R1_COMPLETE_TASK_GEOMETRY_V12_VERSION = (
    "order9_r1_complete_eight_phase_convex_proof_exact_fallback_v12"
)
ORDER9_R1_COMPLETE_TASK_GEOMETRY_V12_FILENAME = "complete_task_geometry_audit_v12.json"
_ANCHOR_ATTITUDE_TOLERANCE_RAD = math.radians(3.0)


def audit_order9_r1_complete_task_geometry_v12(
    *,
    phases: Mapping[str, ContactWrenchTrajectory],
    task_spec: TaskSpec,
    morphology: MorphologyGraph,
    contact_candidate_set: ContactCandidateSet,
    physical_model_config_path: str | Path,
    contract: Order9R1NominalGeometryV11Contract,
    fail_fast: bool = True,
) -> dict[str, Any]:
    """Prove separation with convex supersets, using STL only when unresolved."""

    original = CollisionAwareNativeCentroidalPostureIKSolver.check_configuration
    counters = {
        "convex_proof_scene_count": 0,
        "exact_fallback_scene_count": 0,
    }

    def two_stage_check(
        solver,
        *,
        morphology,
        centroidal_pose_world,
        joint_positions_rad,
        exact,
        margin_m,
        ground_plane_z_m=None,
    ):
        if not exact:
            return original(
                solver,
                morphology=morphology,
                centroidal_pose_world=centroidal_pose_world,
                joint_positions_rad=joint_positions_rad,
                exact=False,
                margin_m=margin_m,
                ground_plane_z_m=ground_plane_z_m,
            )
        proxy = original(
            solver,
            morphology=morphology,
            centroidal_pose_world=centroidal_pose_world,
            joint_positions_rad=joint_positions_rad,
            exact=False,
            margin_m=margin_m,
            ground_plane_z_m=ground_plane_z_m,
        )
        if _convex_proxy_proves_admission(proxy, required_clearance_m=margin_m):
            counters["convex_proof_scene_count"] += 1
            return {
                **proxy,
                "r1_geometry_proof": "convex_mesh_superset",
                "r1_exact_stl_fallback_invoked": False,
            }
        counters["exact_fallback_scene_count"] += 1
        exact_result = original(
            solver,
            morphology=morphology,
            centroidal_pose_world=centroidal_pose_world,
            joint_positions_rad=joint_positions_rad,
            exact=True,
            margin_m=margin_m,
            ground_plane_z_m=ground_plane_z_m,
        )
        return _complete_exact_result(
            exact_result,
            required_clearance_m=margin_m,
        )

    with patch.object(
        CollisionAwareNativeCentroidalPostureIKSolver,
        "check_configuration",
        new=two_stage_check,
    ), patch.object(
        geometry_v11,
        "_ANCHOR_ATTITUDE_TOLERANCE_RAD",
        _ANCHOR_ATTITUDE_TOLERANCE_RAD,
    ):
        result = audit_order9_r1_complete_task_geometry_v11(
            phases=phases,
            task_spec=task_spec,
            morphology=morphology,
            contact_candidate_set=contact_candidate_set,
            physical_model_config_path=physical_model_config_path,
            contract=contract,
            fail_fast=fail_fast,
        )
    result.update(
        {
            "audit_version": ORDER9_R1_COMPLETE_TASK_GEOMETRY_V12_VERSION,
            "collision_geometry_mode": (
                "convex_stl_superset_proof_then_exact_stl_triangle_mesh_fallback"
            ),
            "convex_geometry_is_conservative_mesh_superset": True,
            "exact_stl_fallback_required_when_convex_proof_unresolved": True,
            "anchor_position_tolerance_m": 0.030,
            "anchor_attitude_tolerance_rad": _ANCHOR_ATTITUDE_TOLERANCE_RAD,
            **counters,
        }
    )
    return result


def require_order9_r1_complete_task_geometry_v12(audit: Mapping[str, Any]) -> None:
    computed = int(audit.get("computed_collision_scene_count", -1))
    convex = int(audit.get("convex_proof_scene_count", -1))
    exact = int(audit.get("exact_fallback_scene_count", -1))
    if (
        audit.get("audit_version") != ORDER9_R1_COMPLETE_TASK_GEOMETRY_V12_VERSION
        or audit.get("accepted") is not True
        or audit.get("status") != "accepted"
        or tuple(audit.get("checked_phases", ())) != ORDER9_R1_COMPLETE_TASK_PHASES
        or audit.get("isaac_invoked") is not False
        or audit.get("controller_layers_invoked") is not False
        or audit.get("formal_isaac_admission") is not True
        or audit.get("convex_geometry_is_conservative_mesh_superset") is not True
        or not math.isclose(
            float(audit.get("anchor_position_tolerance_m", -1.0)),
            0.030,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            float(audit.get("anchor_attitude_tolerance_rad", -1.0)),
            _ANCHOR_ATTITUDE_TOLERANCE_RAD,
            abs_tol=1.0e-12,
        )
        or computed < 1
        or convex < 0
        or exact < 0
        or convex + exact != computed
    ):
        examples = audit.get("failure_examples", [])
        suffix = "" if not examples else f": {examples[0]}"
        raise SchemaValidationError(
            "R1 v12 eight-phase geometry admission rejected" + suffix
        )


def _convex_proxy_proves_admission(
    result: Mapping[str, Any],
    *,
    required_clearance_m: float,
) -> bool:
    """A separated convex superset proves its enclosed STL mesh is separated."""

    robot = float(result.get("minimum_robot_clearance_m", -math.inf))
    obstacle = float(result.get("minimum_object_clearance_m", -math.inf))
    ground = result.get("minimum_ground_clearance_m")
    return bool(
        robot > 0.0
        and obstacle + 1.0e-12 >= float(required_clearance_m)
        and (
            ground is None
            # v11 uses the task obstacle margin for robot/box clearance and
            # the independent zero-clearance rule for the ground plane.
            or float(ground) + 1.0e-12 >= 0.0
        )
        and int(result.get("selected_contact_penetration_violating_pair_count", 0)) == 0
    )


def _complete_exact_result(
    result: Mapping[str, Any],
    *,
    required_clearance_m: float,
) -> dict[str, Any]:
    """Expose collisions even when a 32-entry diagnostic list is saturated."""

    completed = dict(result)
    pairs = [dict(value) for value in (result.get("worst_pairs") or [])]
    robot_collisions = int(result.get("robot_colliding_pair_count", 0))
    object_collisions = int(result.get("object_colliding_pair_count", 0))
    object_clearance = float(result.get("minimum_object_clearance_m", math.inf))
    if robot_collisions and not any(
        value.get("second_kind") == "robot" and value.get("colliding") is True
        for value in pairs
    ):
        pairs.append(
            {
                "second_kind": "robot",
                "clearance_m": min(0.0, float(result["minimum_robot_clearance_m"])),
                "colliding": True,
                "aggregate_pair_count": robot_collisions,
            }
        )
    if object_clearance + 1.0e-12 < float(required_clearance_m) and not any(
        value.get("second_kind") == "object"
        and (
            value.get("colliding") is True
            or float(value.get("clearance_m", math.inf)) + 1.0e-12
            < float(required_clearance_m)
        )
        for value in pairs
    ):
        pairs.append(
            {
                "second_kind": "object",
                "clearance_m": object_clearance,
                "colliding": object_collisions > 0,
                "aggregate_pair_count": object_collisions,
            }
        )
    completed.update(
        {
            "worst_pairs": pairs,
            "r1_geometry_proof": "exact_stl_triangle_mesh_fallback",
            "r1_exact_stl_fallback_invoked": True,
        }
    )
    return completed


__all__ = [
    "ORDER9_R1_COMPLETE_TASK_GEOMETRY_V12_FILENAME",
    "ORDER9_R1_COMPLETE_TASK_GEOMETRY_V12_VERSION",
    "audit_order9_r1_complete_task_geometry_v12",
    "require_order9_r1_complete_task_geometry_v12",
]
