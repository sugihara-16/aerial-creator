from __future__ import annotations

"""R1 v13 clearance audit with exact requested-margin fallback."""

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
from amsrr.training.order9_r1_complete_task_geometry_v12 import (
    _complete_exact_result,
    _convex_proxy_proves_admission,
)
from amsrr.training.order9_r1_nominal_geometry_v11 import (
    Order9R1NominalGeometryV11Contract,
)

ORDER9_R1_COMPLETE_TASK_GEOMETRY_V13_VERSION = (
    "order9_r1_complete_eight_phase_convex_exact_margin_v13"
)
ORDER9_R1_COMPLETE_TASK_GEOMETRY_V13_FILENAME = "complete_task_geometry_audit_v13.json"
_ANCHOR_ATTITUDE_TOLERANCE_RAD = math.radians(3.0)


def _requested_margin_two_stage_check(
    original,
    counters: dict[str, int],
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
    tolerance = float(solver.collision_config.collision_feasibility_tolerance_m)
    exact_result = original(
        solver,
        morphology=morphology,
        centroidal_pose_world=centroidal_pose_world,
        joint_positions_rad=joint_positions_rad,
        exact=True,
        margin_m=float(margin_m) + tolerance,
        ground_plane_z_m=ground_plane_z_m,
    )
    if not math.isclose(
        float(exact_result.get("effective_collision_margin_m", -1.0)),
        float(margin_m),
        abs_tol=1.0e-12,
    ):
        raise SchemaValidationError(
            "R1 v13 exact fallback did not enforce the requested margin"
        )
    return {
        **_complete_exact_result(
            exact_result,
            required_clearance_m=margin_m,
        ),
        "r1_geometry_proof": "exact_stl_requested_margin",
        "r1_exact_stl_fallback_invoked": True,
    }


def audit_order9_r1_complete_task_geometry_v13(
    *,
    phases: Mapping[str, ContactWrenchTrajectory],
    task_spec: TaskSpec,
    morphology: MorphologyGraph,
    contact_candidate_set: ContactCandidateSet,
    physical_model_config_path: str | Path,
    contract: Order9R1NominalGeometryV11Contract,
    fail_fast: bool = True,
) -> dict[str, Any]:
    """Use convex-hull proof, then exact mesh at the undiminished margin."""

    original = CollisionAwareNativeCentroidalPostureIKSolver.check_configuration
    counters = {
        "convex_proof_scene_count": 0,
        "exact_fallback_scene_count": 0,
    }

    def two_stage_check(solver, **values):
        return _requested_margin_two_stage_check(
            original,
            counters,
            solver,
            **values,
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
            "audit_version": ORDER9_R1_COMPLETE_TASK_GEOMETRY_V13_VERSION,
            "collision_geometry_mode": (
                "convex_stl_superset_then_exact_stl_requested_margin"
            ),
            "convex_geometry_is_conservative_mesh_superset": True,
            "exact_stl_fallback_required_when_convex_proof_unresolved": True,
            "exact_margin_tolerance_compensated": True,
            "native_exact_aabb_uses_requested_clearance": True,
            "anchor_position_tolerance_m": 0.030,
            "anchor_attitude_tolerance_rad": _ANCHOR_ATTITUDE_TOLERANCE_RAD,
            **counters,
        }
    )
    return result


def require_order9_r1_complete_task_geometry_v13(
    audit: Mapping[str, Any],
) -> None:
    computed = int(audit.get("computed_collision_scene_count", -1))
    convex = int(audit.get("convex_proof_scene_count", -1))
    exact = int(audit.get("exact_fallback_scene_count", -1))
    if (
        audit.get("audit_version") != ORDER9_R1_COMPLETE_TASK_GEOMETRY_V13_VERSION
        or audit.get("accepted") is not True
        or audit.get("status") != "accepted"
        or tuple(audit.get("checked_phases", ())) != ORDER9_R1_COMPLETE_TASK_PHASES
        or audit.get("isaac_invoked") is not False
        or audit.get("controller_layers_invoked") is not False
        or audit.get("formal_isaac_admission") is not True
        or audit.get("convex_geometry_is_conservative_mesh_superset") is not True
        or audit.get("exact_margin_tolerance_compensated") is not True
        or audit.get("native_exact_aabb_uses_requested_clearance") is not True
        or computed < 1
        or convex < 0
        or exact < 0
        or convex + exact != computed
    ):
        examples = audit.get("failure_examples", [])
        suffix = "" if not examples else f": {examples[0]}"
        raise SchemaValidationError(
            "R1 v13 exact-margin geometry admission rejected" + suffix
        )


__all__ = [
    "ORDER9_R1_COMPLETE_TASK_GEOMETRY_V13_FILENAME",
    "ORDER9_R1_COMPLETE_TASK_GEOMETRY_V13_VERSION",
    "audit_order9_r1_complete_task_geometry_v13",
    "require_order9_r1_complete_task_geometry_v13",
]
