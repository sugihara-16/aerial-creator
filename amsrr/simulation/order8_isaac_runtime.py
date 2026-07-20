from __future__ import annotations

"""Real-Isaac producer for the P4-full Order 8 natural-contact smoke.

Isaac modules are imported only inside :func:`run_order8_isaac_runtime`, so
normal unit-test collection stays independent of the simulator installation.
"""

from bisect import bisect_right
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Collection, Mapping, Sequence

from amsrr.controllers.centroidal_admittance import (
    CentroidalAdmittanceConfig,
    CentroidalAdmittanceController,
    CentroidalExternalWrenchEstimate,
    CentroidalExternalWrenchEstimator,
    CentroidalExternalWrenchEstimatorConfig,
)
from amsrr.controllers.controller_base import PayloadCoupling
from amsrr.controllers.qpid_controller import QPIDTrackingProfile
from amsrr.geometry.pose_math import (
    compose_pose,
    inverse_pose,
    matmul,
    matvec,
    pose_from_transform,
    pose_to_xyz_rpy,
    transform_from_pose,
    transform_from_xyz_rpy,
    transpose,
)
from amsrr.geometry.contact_material import combine_friction
from amsrr.schemas.common import Pose7D, SchemaValidationError
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.order8 import Order8NaturalContactPhase
from amsrr.schemas.policies import ControllerStatus
from amsrr.schemas.runtime import (
    ModuleRuntimeState,
    ObjectRuntimeState,
    RuntimeObservation,
    TaskProgressState,
)
from amsrr.simulation.order8_side_proxy_pad import (
    build_order8_side_proxy_pad_specs,
    load_order8_side_proxy_pad_preview_config,
)
from amsrr.training.order9_teacher import (
    build_order8_grasp_carry_task_spec,
    compile_high_level_context,
)

ORDER8_ISAAC_REPORT_VERSION = "order8_natural_contact_isaac_report_v1"
ORDER8_PROGRESS_PREFIX = "[order8-natural-contact]"
# The implicit-drive damping is a simulator-side numerical gain, not the raw
# AK MIT Kd field.  The hardware-relevant authority remains the independently
# audited applied torque/current/speed envelope.  Keep a finite guard against
# accidental pathological values while allowing an overdamped numerical drive.
ORDER8_SIMULATION_DRIVE_DAMPING_MAX_NMS_PER_RAD = 50.0
ORDER8_CONTACT_STALL_RATED_TORQUE_FRACTION = 0.10
ORDER8_POST_FIRST_ARREST_CREEP_MULTIPLIER = 3.0
ORDER8_SELECTED_GRIPPER_MATERIAL_PATH = (
    "/World/Order8/Materials/SelectedGripperPhysicsMaterial"
)
ORDER8_SELECTED_GRIPPER_FRICTION_COMBINE_MODE = "max"
ORDER8_DIAGNOSTIC_PROXY_PAD_PRIM_NAME = "Order8SelectedContactProxyPad"
ORDER8_DIAGNOSTIC_PROXY_PAD_TANGENTIAL_SIZE_M = 0.030
ORDER8_DIAGNOSTIC_PROXY_PAD_THICKNESS_M = 0.002
ORDER8_DIAGNOSTIC_PROXY_PAD_MESH_CLEARANCE_M = 0.001
ORDER8_DIAGNOSTIC_PROXY_PAD_SURFACE_BAND_M = 0.003
ORDER8_DIAGNOSTIC_CONE_PROXY_PAD_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs/training/order8_side_proxy_pad_preview.yaml"
)
ORDER8_OBJECT_SUPPORT_PATH = "/World/Order8/ObjectSupport"
# ``compute_floor_contact_placement`` leaves 2 mm below the authored neutral
# collision geometry.  That is suitable for reset but remains inside a typical
# PhysX contact-offset band.  Contact acquisition is airborne after takeoff, so
# add a recorded margin while remaining inside the approved +/-50 mm face
# region.  This is simulator geometry planning, not a motor-limit relaxation.
ORDER8_GRASP_ADDITIONAL_FLOOR_CLEARANCE_M = 0.010
ORDER8_NEAR_CONTACT_DIAGNOSTIC_WARMUP_S = 0.50
ORDER8_PRELIFT_RELATIVE_SPEED_FRACTION = 0.50
# A positive geometric gap at this scale is well outside the penetration-noise
# floor used by the Order-8 monitor while remaining small relative to the
# planned 100 mm lift.  It confirms that the support can no longer carry the
# object without using privileged support-contact truth as a controller input.
ORDER8_OBJECT_LIFT_OFF_CLEARANCE_M = 0.001
ORDER8_DIAGNOSTIC_PAYLOAD_COUPLING_FULL = "full"
ORDER8_DIAGNOSTIC_PAYLOAD_COUPLING_FORCE_ONLY = "translational_force_only"
ORDER8_DIAGNOSTIC_PAYLOAD_COUPLING_FORCE_OFFSET = (
    "translational_force_and_com_offset_moment"
)
ORDER8_DIAGNOSTIC_PAYLOAD_COUPLING_COMPONENT_MODES = (
    ORDER8_DIAGNOSTIC_PAYLOAD_COUPLING_FULL,
    ORDER8_DIAGNOSTIC_PAYLOAD_COUPLING_FORCE_ONLY,
    ORDER8_DIAGNOSTIC_PAYLOAD_COUPLING_FORCE_OFFSET,
)
ORDER8_DIAGNOSTIC_ANCHOR_HOLD_TASK_GAIN_PER_S = 10.0
ORDER8_DIAGNOSTIC_LOADED_STATE_REBASE_MIN_HOLD_S = 0.50
# Order 8 is a natural-contact/controller substrate smoke, not a general grasp
# planner.  Once the known grasp-ready pose is reached, drive the selected
# diagnostic joints with one fixed whole-structure velocity ratio at a small
# speed.  The authored-mesh free-object A/B at the retained 30 mm
# grasp-reference displacement gate found that 0.020 rad/s failed during
# LIFT (30.38 mm), while 0.005 rad/s completed lift, transport, place, and
# contact-free release with 3.09 mm maximum displacement.  Keep the slower
# value as the deterministic fallback; this remains far below the AK40-10
# velocity envelope and changes no learned-policy interface.
# The ratio is solved once at closure onset toward the known grasp pose; it is
# never recomputed from contact geometry.  A later learned pi_L owns general
# whole-structure posture planning.
ORDER8_SIMPLE_CLOSURE_JOINT_SPEED_RADPS = 0.005
# Opening no longer has to build or retain contact load.  Keep it independent
# from the deliberately slow closing creep so the deterministic fallback can
# clear the object before the planner's bounded RELEASE timeout.  This is the
# formerly stable nominal joint-motion rate and remains two orders of
# magnitude below the configured AK40-10 velocity envelope.
ORDER8_SIMPLE_RELEASE_JOINT_SPEED_RADPS = 0.020


def _clip(value: float, lower: float, upper: float) -> float:
    """Return ``value`` clipped to the inclusive finite scalar interval."""

    return max(float(lower), min(float(upper), float(value)))


def _vector_world_to_pose_local(
    pose_world: Sequence[float],
    vector_world: Sequence[float],
) -> tuple[float, float, float]:
    """Rotate a free vector from world coordinates into a pose-local frame."""

    if len(pose_world) != 7 or not all(
        math.isfinite(float(value)) for value in pose_world
    ):
        raise SchemaValidationError("vector frame pose must be a finite Pose7D")
    if len(vector_world) != 3 or not all(
        math.isfinite(float(value)) for value in vector_world
    ):
        raise SchemaValidationError("world vector must contain three finite values")
    world_from_local = transform_from_pose(
        tuple(float(value) for value in pose_world)  # type: ignore[arg-type]
    ).rotation
    return tuple(
        float(value)
        for value in matvec(
            transpose(world_from_local),
            tuple(float(value) for value in vector_world),
        )
    )


def _vector_pose_local_to_world(
    pose_world: Sequence[float],
    vector_local: Sequence[float],
) -> tuple[float, float, float]:
    """Rotate a free vector from a pose-local frame into world coordinates."""

    if len(pose_world) != 7 or not all(
        math.isfinite(float(value)) for value in pose_world
    ):
        raise SchemaValidationError("vector frame pose must be a finite Pose7D")
    if len(vector_local) != 3 or not all(
        math.isfinite(float(value)) for value in vector_local
    ):
        raise SchemaValidationError("local vector must contain three finite values")
    world_from_local = transform_from_pose(
        tuple(float(value) for value in pose_world)  # type: ignore[arg-type]
    ).rotation
    return tuple(
        float(value)
        for value in matvec(
            world_from_local,
            tuple(float(value) for value in vector_local),
        )
    )


def _dominant_signed_vector_axis(vector: Sequence[float]) -> str:
    """Return the signed axis carrying the largest absolute vector component."""

    if len(vector) != 3 or not all(math.isfinite(float(value)) for value in vector):
        raise SchemaValidationError("dominant-axis vector must contain three finite values")
    values = tuple(float(value) for value in vector)
    axis = max(range(3), key=lambda index: abs(values[index]))
    if abs(values[axis]) <= 1.0e-12:
        return "stationary"
    return ("+" if values[axis] > 0.0 else "-") + "xyz"[axis]


def _kit_visualizer_requested(args: Any) -> bool:
    """Return whether Isaac Lab's normalized visualizer selection includes Kit.

    ``AppLauncher`` stores ``--viz kit`` in ``args.visualizer`` as a sequence.
    Retain the legacy ``args.viz`` fallback for direct/unit callers, but do not
    infer GUI state from the attribute name used on the command line.
    """

    raw_selection = getattr(args, "visualizer", None)
    if raw_selection is None:
        raw_selection = getattr(args, "viz", None)
    if raw_selection is None:
        return False
    if isinstance(raw_selection, str):
        selections = raw_selection.split(",")
    else:
        try:
            selections = tuple(raw_selection)
        except TypeError:
            selections = (raw_selection,)
    return any(str(selection).strip().lower() == "kit" for selection in selections)


def _diagnostic_force_stop_ready(
    *,
    contact_configuration_latched: bool,
    contact_force_scale: float,
    stop_force_scale: float,
    grasp_acquired: bool,
) -> bool:
    """Stop partial ramps immediately, but require stable grasp at full scale."""

    if (
        not contact_configuration_latched
        or float(contact_force_scale) + 1.0e-12 < float(stop_force_scale)
    ):
        return False
    return bool(
        float(stop_force_scale) < 1.0 - 1.0e-12 or grasp_acquired
    )


def _prelift_relative_speed_threshold_mps(
    *,
    maintained_contact_slip_limit_mps: float,
) -> float:
    """Reserve deterministic velocity margin before maintained-contact motion."""

    limit = float(maintained_contact_slip_limit_mps)
    if not math.isfinite(limit) or limit <= 0.0:
        raise SchemaValidationError(
            "Order8 maintained-contact slip limit must be finite and positive"
        )
    return ORDER8_PRELIFT_RELATIVE_SPEED_FRACTION * limit


def _advance_loaded_state_rebase_settle_dwell(
    previous_dwell_s: float,
    *,
    relative_speed_mps_by_anchor: Mapping[int, float],
    selected_anchor_ids: Sequence[int],
    speed_threshold_mps: float,
    dt_s: float,
) -> float:
    """Advance a continuous post-rebase settle dwell from kinematic state.

    This gate consumes only full-link/object relative speed.  It deliberately
    does not use raw Isaac contact, per-patch force, or contact-wrench
    decomposition, so the same one-shot setpoint rule can later be driven by
    ordinary robot/object state estimation.
    """

    previous = float(previous_dwell_s)
    dt = float(dt_s)
    if not math.isfinite(previous) or previous < 0.0:
        raise SchemaValidationError(
            "Order8 loaded-state settle dwell must be finite and non-negative"
        )
    if not math.isfinite(dt) or dt <= 0.0:
        raise SchemaValidationError(
            "Order8 loaded-state settle dt must be finite and positive"
        )
    settled = _contact_force_hold_settled(
        relative_speed_mps_by_anchor,
        selected_anchor_ids=selected_anchor_ids,
        speed_threshold_mps=float(speed_threshold_mps),
    )
    return previous + dt if settled else 0.0


def _loaded_state_rebase_acceleration_bias_scale(
    nominal_scale: float,
    *,
    rebase_settle_active: bool,
) -> float:
    """Remove micro-lift acceleration while holding the loaded hover state.

    Payload gravity feed-forward is a separate command channel and remains
    active.  Only the transient lift-acceleration bias is suppressed while
    the one-shot measured state is settling.
    """

    scale = float(nominal_scale)
    if not math.isfinite(scale) or scale < 0.0 or scale > 1.0:
        raise SchemaValidationError(
            "Order8 lift acceleration bias scale must be finite and in [0, 1]"
        )
    return 0.0 if rebase_settle_active else scale


def _fixed_whole_structure_closure_velocity_targets(
    *,
    ordered_joint_ids: Sequence[str],
    one_shot_velocity_targets_radps: Mapping[str, float],
    maximum_speed_radps: float,
    fixed_joint_ids: Collection[str] = (),
) -> dict[str, float]:
    """Normalize a one-shot whole-structure IK direction into a fixed command.

    IK is evaluated exactly once against the known terminal anchor pose.  The
    returned ratio is then held constant and contains no receding target,
    contact geometry, or raw-contact feedback.  It is integrated from the
    previous position target by
    :func:`_apply_simple_joint_velocity_command`.
    """

    joint_ids = tuple(str(joint_id) for joint_id in ordered_joint_ids)
    if not joint_ids or len(set(joint_ids)) != len(joint_ids):
        raise SchemaValidationError(
            "Order8 simple closure requires unique ordered Dock joint ids"
        )
    if set(one_shot_velocity_targets_radps) != set(joint_ids):
        raise SchemaValidationError(
            "Order8 fixed closure one-shot velocity map must cover every Dock joint"
        )
    fixed_ids = {str(joint_id) for joint_id in fixed_joint_ids}
    unknown_fixed_ids = fixed_ids.difference(joint_ids)
    if unknown_fixed_ids:
        raise SchemaValidationError(
            "Order8 fixed closure fixed-joint ids are not in the Dock vector: "
            + ", ".join(sorted(unknown_fixed_ids))
        )
    speed = float(maximum_speed_radps)
    if not math.isfinite(speed) or speed <= 0.0:
        raise SchemaValidationError(
            "Order8 simple closure speed must be finite and positive"
        )
    raw_targets = {
        joint_id: (
            0.0
            if joint_id in fixed_ids
            else float(one_shot_velocity_targets_radps[joint_id])
        )
        for joint_id in joint_ids
    }
    for joint_id, value in raw_targets.items():
        if not math.isfinite(value):
            raise SchemaValidationError(
                f"Order8 fixed closure one-shot velocity for {joint_id!r} "
                "must be finite"
            )
    peak = max((abs(value) for value in raw_targets.values()), default=0.0)
    if peak <= 1.0e-9:
        raise SchemaValidationError(
            "Order8 fixed closure one-shot IK produced no usable joint motion"
        )
    scale = speed / peak
    return {
        joint_id: value * scale
        for joint_id, value in raw_targets.items()
    }


def _position_preload_joint_ids_by_anchor(
    *,
    ordered_joint_ids: Sequence[str],
    closure_velocity_targets_radps: Mapping[str, float],
    influential_joint_ids_by_anchor: Mapping[int, Sequence[str]],
    fixed_joint_ids: Collection[str] = (),
    motion_epsilon_radps: float = 1.0e-9,
) -> dict[int, tuple[str, ...]]:
    """Select the moving closure joints whose load belongs to each side.

    A fixed diagnostic joint or a kinematically influential joint with zero
    closure velocity cannot build the approved positional preload, so neither
    may satisfy a side's load gate.  Shared moving joints remain in every
    affected side; the velocity scheduler below stops them conservatively as
    soon as any owning side freezes.
    """

    joint_ids = tuple(str(joint_id) for joint_id in ordered_joint_ids)
    if not joint_ids or len(set(joint_ids)) != len(joint_ids):
        raise SchemaValidationError(
            "Order8 position preload requires unique ordered Dock joint ids"
        )
    if set(closure_velocity_targets_radps) != set(joint_ids):
        raise SchemaValidationError(
            "Order8 position preload closure velocity must cover every Dock joint"
        )
    anchor_ids = tuple(int(anchor_id) for anchor_id in influential_joint_ids_by_anchor)
    if len(anchor_ids) < 2 or len(set(anchor_ids)) != len(anchor_ids):
        raise SchemaValidationError(
            "Order8 position preload requires at least two unique anchors"
        )
    fixed = {str(joint_id) for joint_id in fixed_joint_ids}
    unknown_fixed = fixed.difference(joint_ids)
    if unknown_fixed:
        raise SchemaValidationError(
            "Order8 position preload fixed joints are not in the Dock vector: "
            + ", ".join(sorted(unknown_fixed))
        )
    epsilon = float(motion_epsilon_radps)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise SchemaValidationError(
            "Order8 position preload motion epsilon must be finite and positive"
        )
    velocities = {
        joint_id: float(closure_velocity_targets_radps[joint_id])
        for joint_id in joint_ids
    }
    if any(not math.isfinite(value) for value in velocities.values()):
        raise SchemaValidationError(
            "Order8 position preload closure velocities must be finite"
        )

    result: dict[int, tuple[str, ...]] = {}
    known = set(joint_ids)
    for anchor_id in anchor_ids:
        influential = tuple(
            str(joint_id)
            for joint_id in influential_joint_ids_by_anchor[anchor_id]
        )
        if not influential or len(set(influential)) != len(influential):
            raise SchemaValidationError(
                "Order8 position preload influential joints must be unique/nonempty"
            )
        unknown = set(influential).difference(known)
        if unknown:
            raise SchemaValidationError(
                "Order8 position preload influential joints are not in the Dock "
                "vector: " + ", ".join(sorted(unknown))
            )
        moving = tuple(
            joint_id
            for joint_id in joint_ids
            if joint_id in influential
            and joint_id not in fixed
            and abs(velocities[joint_id]) > epsilon
        )
        if not moving:
            raise SchemaValidationError(
                f"Order8 position preload anchor {anchor_id} has no moving closure joint"
            )
        result[anchor_id] = moving
    return result


def _load_limited_position_preload_velocity_targets(
    *,
    ordered_joint_ids: Sequence[str],
    closure_velocity_targets_radps: Mapping[str, float],
    preload_joint_ids_by_anchor: Mapping[int, Sequence[str]],
    frozen_anchor_ids: Collection[int],
    maximum_speed_radps: float,
    fixed_joint_ids: Collection[str] = (),
) -> dict[str, float]:
    """Continue the fixed closure ratio and freeze load-complete branches."""

    joint_ids = tuple(str(joint_id) for joint_id in ordered_joint_ids)
    scaled = _fixed_whole_structure_closure_velocity_targets(
        ordered_joint_ids=joint_ids,
        one_shot_velocity_targets_radps=closure_velocity_targets_radps,
        maximum_speed_radps=float(maximum_speed_radps),
        fixed_joint_ids=fixed_joint_ids,
    )
    anchor_ids = {int(anchor_id) for anchor_id in preload_joint_ids_by_anchor}
    frozen = {int(anchor_id) for anchor_id in frozen_anchor_ids}
    unknown_frozen = frozen.difference(anchor_ids)
    if unknown_frozen:
        raise SchemaValidationError(
            "Order8 position preload frozen anchor ids are unknown: "
            + ", ".join(str(value) for value in sorted(unknown_frozen))
        )
    owners_by_joint: dict[str, set[int]] = {joint_id: set() for joint_id in joint_ids}
    for anchor_id, raw_joint_ids in preload_joint_ids_by_anchor.items():
        selected = {str(joint_id) for joint_id in raw_joint_ids}
        unknown = selected.difference(joint_ids)
        if not selected or unknown:
            raise SchemaValidationError(
                "Order8 position preload side joint sets must be nonempty subsets "
                "of the Dock vector"
            )
        for joint_id in selected:
            owners_by_joint[joint_id].add(int(anchor_id))
    return {
        joint_id: (
            scaled[joint_id]
            if owners_by_joint[joint_id]
            and owners_by_joint[joint_id].isdisjoint(frozen)
            else 0.0
        )
        for joint_id in joint_ids
    }


def _apply_simple_joint_velocity_command(
    joint_result: Any,
    joint_vector: Any,
    *,
    velocity_targets_radps: Mapping[str, float],
    previous_position_targets_rad: Mapping[str, float],
    dt_s: float,
    zero_torque_bias: bool,
) -> Any:
    """Integrate a joint velocity command from the previous position target."""

    expected = tuple(str(joint_id) for joint_id in joint_vector.joint_ids)
    if len(set(expected)) != len(expected):
        raise SchemaValidationError(
            "Order8 simple velocity command joint ids must be unique"
        )
    if set(velocity_targets_radps) != set(expected):
        raise SchemaValidationError(
            "Order8 simple velocity command must cover exactly the Dock joints"
        )
    if set(previous_position_targets_rad) != set(expected):
        raise SchemaValidationError(
            "Order8 previous position targets must cover exactly the Dock joints"
        )
    dt = float(dt_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise SchemaValidationError(
            "Order8 simple velocity command dt must be finite and positive"
        )
    positions: dict[str, float] = {}
    velocities: dict[str, float] = {}
    for index, joint_id in enumerate(expected):
        requested_velocity = float(velocity_targets_radps[joint_id])
        if not math.isfinite(requested_velocity):
            raise SchemaValidationError(
                f"Order8 simple velocity command for {joint_id!r} is non-finite"
            )
        limit = joint_vector.limits[index]
        bounded_velocity = _clip(
            requested_velocity,
            -float(limit.max_velocity_radps),
            float(limit.max_velocity_radps),
        )
        previous_target = float(previous_position_targets_rad[joint_id])
        if not math.isfinite(previous_target):
            raise SchemaValidationError(
                f"Order8 previous position target for {joint_id!r} is non-finite"
            )
        target_position = _clip(
            previous_target + bounded_velocity * dt,
            float(limit.position_lower_rad),
            float(limit.position_upper_rad),
        )
        applied_velocity = (target_position - previous_target) / dt
        positions[joint_id] = target_position
        velocities[joint_id] = applied_velocity

    policy = joint_result.policy_command
    torque_bias = (
        {joint_id: 0.0 for joint_id in expected}
        if zero_torque_bias
        else {
            joint_id: float(policy.joint_torque_bias[joint_id])
            for joint_id in expected
        }
    )
    simple_policy = replace(
        policy,
        joint_position_targets=positions,
        joint_velocity_targets=velocities,
        joint_torque_bias=torque_bias,
    )
    simple_policy.validate()
    return replace(joint_result, policy_command=simple_policy)


def _zero_joint_torque_bias(joint_result: Any, joint_vector: Any) -> Any:
    """Disable the offset-torque path while preserving position/velocity intent."""

    expected = tuple(str(joint_id) for joint_id in joint_vector.joint_ids)
    if not expected or len(set(expected)) != len(expected):
        raise SchemaValidationError(
            "Order8 zero torque-bias override requires unique Dock joint ids"
        )
    policy = joint_result.policy_command
    if set(policy.joint_torque_bias) != set(expected):
        raise SchemaValidationError(
            "Order8 zero torque-bias override must cover every Dock joint"
        )
    zeros = {joint_id: 0.0 for joint_id in expected}
    zero_policy = replace(policy, joint_torque_bias=zeros)
    zero_policy.validate()
    torque_mapping = replace(
        joint_result.torque_mapping,
        unclipped_joint_torque_bias=dict(zeros),
        joint_torque_bias=dict(zeros),
        clipped_joint_ids=(),
    )
    diagnostics = replace(
        joint_result.diagnostics,
        torque_clipped_joint_ids=(),
    )
    return replace(
        joint_result,
        policy_command=zero_policy,
        torque_mapping=torque_mapping,
        diagnostics=diagnostics,
    )


def _apply_closure_direction_joint_torque_bias(
    joint_result: Any,
    joint_vector: Any,
    *,
    closure_velocity_targets_radps: Mapping[str, float],
    selected_joint_ids: Collection[str],
    magnitude_nm: float,
) -> Any:
    """Apply equal-magnitude offset torque along the fixed closure direction.

    This helper is used only by the explicit Order-8 diagnostic A/B.  A raw
    positive torque on every joint would open one side because joint-axis
    signs differ, so the scalar magnitude is signed by the already-established
    simple-closure velocity map.  Non-selected and zero-direction joints stay
    at exactly zero bias.
    """

    expected = tuple(str(joint_id) for joint_id in joint_vector.joint_ids)
    if not expected or len(set(expected)) != len(expected):
        raise SchemaValidationError(
            "Order8 directional torque-bias override requires unique Dock joint ids"
        )
    if set(closure_velocity_targets_radps) != set(expected):
        raise SchemaValidationError(
            "Order8 directional torque-bias closure map must cover every Dock joint"
        )
    selected = {str(joint_id) for joint_id in selected_joint_ids}
    if not selected or not selected.issubset(expected):
        raise SchemaValidationError(
            "Order8 directional torque-bias selected ids must be a nonempty "
            "subset of the Dock vector"
        )
    magnitude = float(magnitude_nm)
    if not math.isfinite(magnitude) or magnitude <= 0.0:
        raise SchemaValidationError(
            "Order8 directional torque-bias magnitude must be finite and positive"
        )

    values = {joint_id: 0.0 for joint_id in expected}
    limits_by_id = {
        joint_id: float(joint_vector.limits[index].max_torque_nm)
        for index, joint_id in enumerate(expected)
    }
    for joint_id in sorted(selected):
        direction = float(closure_velocity_targets_radps[joint_id])
        if not math.isfinite(direction) or abs(direction) <= 1.0e-12:
            raise SchemaValidationError(
                "Order8 directional torque-bias selected joint has no finite "
                f"closure direction: {joint_id}"
            )
        if magnitude > limits_by_id[joint_id] + 1.0e-12:
            raise SchemaValidationError(
                "Order8 directional torque-bias magnitude exceeds the active "
                f"limit for {joint_id}"
            )
        values[joint_id] = math.copysign(magnitude, direction)

    policy = joint_result.policy_command
    if set(policy.joint_torque_bias) != set(expected):
        raise SchemaValidationError(
            "Order8 directional torque-bias policy must cover every Dock joint"
        )
    biased_policy = replace(policy, joint_torque_bias=dict(values))
    biased_policy.validate()
    torque_mapping = replace(
        joint_result.torque_mapping,
        unclipped_joint_torque_bias=dict(values),
        joint_torque_bias=dict(values),
        clipped_joint_ids=(),
    )
    diagnostics = replace(
        joint_result.diagnostics,
        torque_clipped_joint_ids=(),
    )
    return replace(
        joint_result,
        policy_command=biased_policy,
        torque_mapping=torque_mapping,
        diagnostics=diagnostics,
    )


def _hold_joint_subset_positions(
    joint_result: Any,
    joint_vector: Any,
    *,
    position_targets_rad: Mapping[str, float],
) -> Any:
    """Override a selected joint subset with absolute position hold commands."""

    expected = tuple(str(joint_id) for joint_id in joint_vector.joint_ids)
    target_ids = {str(joint_id) for joint_id in position_targets_rad}
    unknown_ids = target_ids.difference(expected)
    if unknown_ids:
        raise SchemaValidationError(
            "Order8 fixed-position joint ids are not in the Dock vector: "
            + ", ".join(sorted(unknown_ids))
        )
    if not target_ids:
        return joint_result

    policy = joint_result.policy_command
    positions = dict(policy.joint_position_targets)
    velocities = dict(policy.joint_velocity_targets)
    torque_bias = dict(policy.joint_torque_bias)
    limits_by_id = {
        joint_id: joint_vector.limits[index]
        for index, joint_id in enumerate(expected)
    }
    for joint_id in sorted(target_ids):
        requested_target = float(position_targets_rad[joint_id])
        if not math.isfinite(requested_target):
            raise SchemaValidationError(
                f"Order8 fixed position target for {joint_id!r} is non-finite"
            )
        limit = limits_by_id[joint_id]
        positions[joint_id] = _clip(
            requested_target,
            float(limit.position_lower_rad),
            float(limit.position_upper_rad),
        )
        velocities[joint_id] = 0.0
        torque_bias[joint_id] = 0.0

    fixed_policy = replace(
        policy,
        joint_position_targets=positions,
        joint_velocity_targets=velocities,
        joint_torque_bias=torque_bias,
    )
    fixed_policy.validate()
    return replace(joint_result, policy_command=fixed_policy)


def _joint_velocity_targets_toward_positions(
    joint_vector: Any,
    *,
    target_positions_rad: Mapping[str, float],
    maximum_speed_radps: float,
    dt_s: float,
) -> dict[str, float]:
    """Return a bounded direct joint-space release command."""

    expected = tuple(str(joint_id) for joint_id in joint_vector.joint_ids)
    if set(target_positions_rad) != set(expected):
        raise SchemaValidationError(
            "Order8 release targets must cover exactly the Dock joints"
        )
    speed = float(maximum_speed_radps)
    dt = float(dt_s)
    if not math.isfinite(speed) or speed <= 0.0:
        raise SchemaValidationError(
            "Order8 release speed must be finite and positive"
        )
    if not math.isfinite(dt) or dt <= 0.0:
        raise SchemaValidationError(
            "Order8 release dt must be finite and positive"
        )
    velocities: dict[str, float] = {}
    for index, joint_id in enumerate(expected):
        target = float(target_positions_rad[joint_id])
        measured = float(joint_vector.positions_rad[index])
        if not math.isfinite(target) or not math.isfinite(measured):
            raise SchemaValidationError(
                f"Order8 release state for {joint_id!r} must be finite"
            )
        velocities[joint_id] = _clip(
            (target - measured) / dt,
            -speed,
            speed,
        )
    return velocities


def _rebased_manipulation_base_poses(
    measured_grasp_pose: Pose7D,
    *,
    transport_distance_m: float,
) -> tuple[Pose7D, Pose7D, Pose7D, Pose7D]:
    """Build lift/transport/place/retreat from the measured full 6D grasp pose."""

    lift_pose = _offset_pose(measured_grasp_pose, dz=0.15)
    transport_pose = _offset_pose(
        lift_pose,
        dx=float(transport_distance_m),
    )
    place_pose = _offset_pose(
        measured_grasp_pose,
        dx=float(transport_distance_m),
    )
    retreat_pose = _offset_pose(place_pose, dx=-0.10, dz=0.20)
    return lift_pose, transport_pose, place_pose, retreat_pose


def _object_relative_inward_preload_pose(
    *,
    anchor_pose_world: Pose7D,
    object_pose_world: Pose7D,
    inward_normal_object: Sequence[float],
    preload_distance_m: float,
) -> Pose7D:
    """Build a bounded post-arrest anchor target in the moving object frame."""

    for name, pose in (
        ("anchor_pose_world", anchor_pose_world),
        ("object_pose_world", object_pose_world),
    ):
        if len(pose) != 7 or not all(math.isfinite(float(value)) for value in pose):
            raise SchemaValidationError(f"Order8 {name} must be a finite Pose7D")
    normal = tuple(float(value) for value in inward_normal_object)
    if len(normal) != 3 or not all(math.isfinite(value) for value in normal):
        raise SchemaValidationError(
            "Order8 inward preload normal must contain three finite values"
        )
    normal_norm = math.sqrt(sum(value * value for value in normal))
    if normal_norm <= 1.0e-12:
        raise SchemaValidationError(
            "Order8 inward preload normal must have non-zero magnitude"
        )
    distance = float(preload_distance_m)
    if not math.isfinite(distance) or distance <= 0.0:
        raise SchemaValidationError(
            "Order8 inward preload distance must be finite and positive"
        )
    unit_normal = tuple(value / normal_norm for value in normal)
    anchor_pose_object = compose_pose(
        inverse_pose(object_pose_world),
        anchor_pose_world,
    )
    return (
        *tuple(
            float(anchor_pose_object[index]) + distance * unit_normal[index]
            for index in range(3)
        ),
        *tuple(float(value) for value in anchor_pose_object[3:7]),
    )


def _advance_contact_yield_blend(
    current_blend: float,
    *,
    yield_requested: bool,
    dt_s: float,
    ramp_down_s: float,
    ramp_up_s: float,
) -> float:
    """Advance the normal-control to yield-control blend in ``[0, 1]``."""

    values = {
        "current_blend": current_blend,
        "dt_s": dt_s,
        "ramp_down_s": ramp_down_s,
        "ramp_up_s": ramp_up_s,
    }
    if any(not math.isfinite(float(value)) for value in values.values()):
        raise SchemaValidationError(
            "Order8 contact-yield blend inputs must be finite"
        )
    if not 0.0 <= float(current_blend) <= 1.0:
        raise SchemaValidationError(
            "Order8 contact-yield blend must remain in [0, 1]"
        )
    if float(dt_s) <= 0.0 or float(ramp_down_s) <= 0.0 or float(ramp_up_s) <= 0.0:
        raise SchemaValidationError(
            "Order8 contact-yield timing values must be positive"
        )
    duration = float(ramp_down_s if yield_requested else ramp_up_s)
    direction = 1.0 if yield_requested else -1.0
    return min(
        max(float(current_blend) + direction * float(dt_s) / duration, 0.0),
        1.0,
    )


def _contact_yield_tracking_profile(
    blend: float,
    *,
    integrator_decay_rate_per_s: float,
) -> QPIDTrackingProfile:
    """Keep ordinary centroidal tracking while admittance moves its target.

    Compliance is expressed solely as a bounded target pose/twist along the
    selected horizontal contact axis.  Height and attitude regulation must not
    be weakened as a side effect of contact acquisition.
    """

    if (
        not math.isfinite(float(blend))
        or not 0.0 <= float(blend) <= 1.0
        or not math.isfinite(float(integrator_decay_rate_per_s))
        or float(integrator_decay_rate_per_s) < 0.0
    ):
        raise SchemaValidationError(
            "Order8 contact-yield tracking profile inputs are invalid"
        )
    profile = QPIDTrackingProfile(
        proportional_gain_scale=1.0,
        integral_gain_scale=1.0,
        derivative_gain_scale=1.0,
        integrator_accumulation_scale=1.0,
        integrator_decay_rate_per_s=0.0,
    )
    profile.validate()
    return profile


def _contact_yield_joint_drive_gains(
    blend: float,
    *,
    nominal_stiffness_nm_per_rad: float,
    nominal_damping_nms_per_rad: float,
    yield_stiffness_scale: float,
    yield_damping_nms_per_rad: float,
) -> tuple[float, float]:
    """Blend simulator-only Dock impedance without changing policy commands."""

    values = {
        "blend": blend,
        "nominal_stiffness_nm_per_rad": nominal_stiffness_nm_per_rad,
        "nominal_damping_nms_per_rad": nominal_damping_nms_per_rad,
        "yield_stiffness_scale": yield_stiffness_scale,
        "yield_damping_nms_per_rad": yield_damping_nms_per_rad,
    }
    if any(not math.isfinite(float(value)) for value in values.values()):
        raise SchemaValidationError(
            "Order8 contact-yield joint drive gains must be finite"
        )
    if not 0.0 <= float(blend) <= 1.0:
        raise SchemaValidationError(
            "Order8 contact-yield joint drive blend must remain in [0, 1]"
        )
    if (
        float(nominal_stiffness_nm_per_rad) <= 0.0
        or float(nominal_damping_nms_per_rad) <= 0.0
        or not 0.0 < float(yield_stiffness_scale) <= 1.0
        or float(yield_damping_nms_per_rad) <= 0.0
    ):
        raise SchemaValidationError(
            "Order8 contact-yield joint drive gains must be positive and "
            "the stiffness scale must not exceed one"
        )
    yielded_stiffness = (
        float(nominal_stiffness_nm_per_rad) * float(yield_stiffness_scale)
    )
    stiffness = (
        (1.0 - float(blend)) * float(nominal_stiffness_nm_per_rad)
        + float(blend) * yielded_stiffness
    )
    damping = (
        (1.0 - float(blend)) * float(nominal_damping_nms_per_rad)
        + float(blend) * float(yield_damping_nms_per_rad)
    )
    return stiffness, damping


def _torque_bias_limit_with_peak_window(
    *,
    continuous_torque_nm: float,
    peak_torque_nm: float,
    elapsed_since_qclose_s: float | None,
    peak_window_s: float | None,
) -> float:
    """Return a bounded diagnostic torque-bias limit with a smooth handoff.

    Production always passes ``peak_window_s=None`` and therefore uses the
    continuous rating.  A diagnostic window holds the actuator peak rating,
    then linearly returns to the continuous rating over at most its final
    0.25 s.  The articulation hard effort clamp remains the independent final
    authority for position-drive plus offset-torque output.
    """

    continuous = float(continuous_torque_nm)
    peak = float(peak_torque_nm)
    if (
        not math.isfinite(continuous)
        or not math.isfinite(peak)
        or continuous <= 0.0
        or peak < continuous
    ):
        raise SchemaValidationError(
            "Order8 torque limits must be finite with 0 < continuous <= peak"
        )
    if peak_window_s is None:
        return continuous
    window = float(peak_window_s)
    if not math.isfinite(window) or window <= 0.0:
        raise SchemaValidationError(
            "Order8 diagnostic peak-torque window must be finite and positive"
        )
    if elapsed_since_qclose_s is None:
        return continuous
    elapsed = float(elapsed_since_qclose_s)
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise SchemaValidationError(
            "Order8 elapsed time since q_close must be finite and non-negative"
        )
    if elapsed >= window:
        return continuous
    ramp_down_s = min(0.25, 0.5 * window)
    ramp_started_s = window - ramp_down_s
    if elapsed <= ramp_started_s:
        return peak
    alpha = (elapsed - ramp_started_s) / ramp_down_s
    scheduled = peak + alpha * (continuous - peak)
    # Avoid a sub-nanotorque interpolation residue being reported as a bridge
    # clip exactly at the continuous-rating handoff.  Simulation timestamps
    # accumulate dt and can leave a slightly larger residue than a direct
    # arithmetic boundary sample.
    if scheduled <= continuous + 1.0e-9:
        return continuous
    if scheduled >= peak - 1.0e-12:
        return peak
    return scheduled


def _actuator_mapping_with_torque_bias_limit(
    actuator_mapping: Any,
    *,
    active_limit_nm: float,
) -> Any:
    """Synchronize an explicitly scheduled Dock effort-bias limit.

    ``ActuatorMapping`` normally and correctly clips Dock effort bias at the
    continuous rating.  The Order-8 diagnostic peak window also raises the
    joint-controller limit, so leaving the bridge at the continuous value
    caused a second, contradictory clip and made the diagnostic path
    impossible to exercise.  This helper changes only channels that explicitly
    support ``joint_effort_bias`` and never permits a value above their recorded
    actuator peak.  Position, velocity, articulation effort, and the independent
    runtime torque/current/speed audits are unchanged.
    """

    limit = float(active_limit_nm)
    if not math.isfinite(limit) or limit <= 0.0:
        raise SchemaValidationError(
            "active actuator-mapping torque-bias limit must be finite and positive"
        )
    updated_channels = []
    updated_count = 0
    for channel in actuator_mapping.channels:
        if "joint_effort_bias" not in channel.supported_command_types:
            updated_channels.append(channel)
            continue
        continuous_raw = channel.metadata.get("continuous_torque_limit_nm")
        peak_raw = channel.metadata.get("peak_torque_limit_nm")
        if not isinstance(continuous_raw, (int, float)) or not isinstance(
            peak_raw, (int, float)
        ):
            raise SchemaValidationError(
                f"Dock actuator channel {channel.actuator_id!r} lacks finite "
                "continuous/peak torque metadata"
            )
        continuous = float(continuous_raw)
        peak = float(peak_raw)
        if (
            not math.isfinite(continuous)
            or not math.isfinite(peak)
            or continuous <= 0.0
            or peak < continuous
            or limit < continuous - 1.0e-12
            or limit > peak + 1.0e-12
        ):
            raise SchemaValidationError(
                f"active torque-bias limit {limit} is outside the recorded "
                f"[{continuous}, {peak}] Nm envelope for {channel.actuator_id!r}"
            )
        metadata = dict(channel.metadata)
        metadata["continuous_torque_limit_nm"] = limit
        metadata["active_torque_bias_limit_nm"] = limit
        updated_channels.append(replace(channel, metadata=metadata))
        updated_count += 1
    if updated_count == 0:
        raise SchemaValidationError(
            "active torque-bias mapping requires at least one Dock effort-bias channel"
        )
    metadata = dict(actuator_mapping.metadata)
    metadata["active_torque_bias_limit_nm"] = limit
    metadata["active_torque_bias_channel_count"] = updated_count
    return replace(
        actuator_mapping,
        channels=updated_channels,
        metadata=metadata,
    )


def _base_translation_speed_limit_for_phase(
    phase: Order8NaturalContactPhase,
    *,
    free_motion_limit_mps: float,
    maintained_contact_limit_mps: float,
) -> float:
    """Keep centroidal motion slow through every contact-required phase."""

    free_limit = float(free_motion_limit_mps)
    contact_limit = float(maintained_contact_limit_mps)
    if (
        not math.isfinite(free_limit)
        or not math.isfinite(contact_limit)
        or free_limit <= 0.0
        or contact_limit <= 0.0
        or contact_limit > free_limit
    ):
        raise SchemaValidationError(
            "Order8 base speed limits must be positive with contact <= free motion"
        )
    if phase in {
        Order8NaturalContactPhase.CONTACT_ACQUISITION,
        Order8NaturalContactPhase.LIFT,
        Order8NaturalContactPhase.TRANSPORT,
        Order8NaturalContactPhase.PLACE,
    }:
        return contact_limit
    return free_limit


def _payload_load_transfer_scale_from_external_wrench(
    *,
    external_wrench_body: Sequence[float],
    body_pose_world: Pose7D,
    lift_start_external_force_world_z_n: float,
    payload_mass_kg: float,
    gravity_mps2: float,
) -> tuple[float, float, float]:
    """Estimate the carried payload share from aggregate centroidal load.

    The centroidal estimator reports the net external wrench acting on the
    robot morphology.  At the start of LIFT the support still carries the
    object, so latch that external vertical force as the zero-load baseline.
    A subsequent *increase in downward force* is the object load transferred
    through the two natural contacts.  This remains an aggregate observation:
    no per-contact wrench decomposition or privileged Isaac contact is used.

    Returns ``(scale, current_force_world_z_n, transferred_load_n)``.
    """

    if len(external_wrench_body) != 6 or not all(
        math.isfinite(float(value)) for value in external_wrench_body
    ):
        raise SchemaValidationError(
            "payload load-transfer external wrench must contain six finite values"
        )
    baseline = float(lift_start_external_force_world_z_n)
    mass = float(payload_mass_kg)
    gravity = float(gravity_mps2)
    if not math.isfinite(baseline):
        raise SchemaValidationError(
            "payload load-transfer lift-start force must be finite"
        )
    if not math.isfinite(mass) or mass <= 0.0:
        raise SchemaValidationError(
            "payload load-transfer mass must be finite and positive"
        )
    if not math.isfinite(gravity) or gravity <= 0.0:
        raise SchemaValidationError(
            "payload load-transfer gravity must be finite and positive"
        )
    force_world = _vector_pose_local_to_world(
        body_pose_world,
        tuple(float(value) for value in external_wrench_body[:3]),
    )
    force_world_z = float(force_world[2])
    transferred_load_n = max(0.0, baseline - force_world_z)
    scale = min(1.0, transferred_load_n / (mass * gravity))
    return scale, force_world_z, transferred_load_n


def _payload_feedforward_scale_for_phase(
    phase: Order8NaturalContactPhase,
    *,
    phase_elapsed_s: float,
    transition_duration_s: float,
    estimated_lift_transfer_scale: float | None = None,
    measured_lift_transfer_scale: float | None = None,
    previous_scale: float = 0.0,
    dt_s: float | None = None,
    lift_off_confirmed: bool = False,
) -> float:
    """Track bounded commanded/observed payload transfer through LIFT.

    LIFT motion starts immediately and ordinary QPID pose feedback creates the
    initial upward load that unloads the object support.  The same bounded
    phase-entry progress used by that upward motion is a feed-forward floor, so
    a supported payload cannot deadlock an observed-load-only follower below
    full weight.  Aggregate centroidal load and measured object rise remain
    monotonic lower bounds/audits, while geometric lift-off promotes the target
    to full payload support.  A bounded slew prevents a one-step model change.
    RELEASE retains its time ramp.
    """

    elapsed = float(phase_elapsed_s)
    duration = float(transition_duration_s)
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise SchemaValidationError(
            "Order8 payload feed-forward phase elapsed time must be finite and non-negative"
        )
    if not math.isfinite(duration) or duration <= 0.0:
        raise SchemaValidationError(
            "Order8 payload feed-forward transition duration must be finite and positive"
        )
    if phase == Order8NaturalContactPhase.LIFT:
        if (
            estimated_lift_transfer_scale is None
            or measured_lift_transfer_scale is None
            or dt_s is None
        ):
            raise SchemaValidationError(
                "Order8 lift payload feed-forward requires estimated/measured "
                "load-transfer scales and dt"
            )
        estimated = float(estimated_lift_transfer_scale)
        measured = float(measured_lift_transfer_scale)
        previous = float(previous_scale)
        dt = float(dt_s)
        if not math.isfinite(estimated) or not 0.0 <= estimated <= 1.0:
            raise SchemaValidationError(
                "Order8 estimated lift load-transfer scale must be finite and in [0, 1]"
            )
        if not math.isfinite(measured) or not 0.0 <= measured <= 1.0:
            raise SchemaValidationError(
                "Order8 measured lift load-transfer scale must be finite and in [0, 1]"
            )
        if not math.isfinite(previous) or not 0.0 <= previous <= 1.0:
            raise SchemaValidationError(
                "Order8 previous payload feed-forward scale must be finite and in [0, 1]"
            )
        if not math.isfinite(dt) or dt <= 0.0:
            raise SchemaValidationError(
                "Order8 payload feed-forward dt must be finite and positive"
            )
        commanded_progress = _contact_motion_entry_speed_scale(
            phase,
            phase_elapsed_s=elapsed,
            transition_duration_s=duration,
        )
        target = max(previous, commanded_progress, estimated, measured)
        if lift_off_confirmed:
            target = 1.0
        maximum_step = min(1.0, dt / duration)
        return min(target, previous + maximum_step)
    if phase in {
        Order8NaturalContactPhase.TRANSPORT,
        Order8NaturalContactPhase.PLACE,
    }:
        return 1.0
    if phase == Order8NaturalContactPhase.RELEASE:
        alpha = min(1.0, elapsed / duration)
        return 1.0 - alpha
    return 0.0


def _lift_acceleration_bias_scale_for_phase(
    phase: Order8NaturalContactPhase,
    *,
    commanded_lift_progress_scale: float,
    lift_off_elapsed_s: float | None,
    lift_off_scale: float | None,
    removal_duration_s: float,
) -> float:
    """Schedule a transient upward inertial bias around payload lift-off.

    Before geometric lift-off, the bias follows the same bounded progress as
    the LIFT trajectory and payload feed-forward schedule.  At the first
    verified 1 mm lift-off event, the current scale is latched and then
    removed with a cubic smoothstep.  The zero slope at both ends avoids a
    step in commanded acceleration derivative.  Every non-LIFT phase receives
    exactly zero bias.
    """

    progress = float(commanded_lift_progress_scale)
    removal_duration = float(removal_duration_s)
    if not math.isfinite(progress) or not 0.0 <= progress <= 1.0:
        raise SchemaValidationError(
            "Order8 lift-acceleration commanded progress must be finite and in [0, 1]"
        )
    if not math.isfinite(removal_duration) or removal_duration <= 0.0:
        raise SchemaValidationError(
            "Order8 lift-acceleration removal duration must be finite and positive"
        )
    if (lift_off_elapsed_s is None) != (lift_off_scale is None):
        raise SchemaValidationError(
            "Order8 lift-acceleration lift-off elapsed time and scale must be paired"
        )
    elapsed: float | None = None
    latched_scale: float | None = None
    if lift_off_elapsed_s is not None and lift_off_scale is not None:
        elapsed = float(lift_off_elapsed_s)
        latched_scale = float(lift_off_scale)
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise SchemaValidationError(
                "Order8 lift-acceleration lift-off elapsed time must be finite and non-negative"
            )
        if not math.isfinite(latched_scale) or not 0.0 <= latched_scale <= 1.0:
            raise SchemaValidationError(
                "Order8 lift-acceleration lift-off scale must be finite and in [0, 1]"
            )
    if phase != Order8NaturalContactPhase.LIFT:
        return 0.0
    if elapsed is None or latched_scale is None:
        return progress
    removal_progress = min(1.0, elapsed / removal_duration)
    smooth_progress = removal_progress * removal_progress * (
        3.0 - 2.0 * removal_progress
    )
    return latched_scale * (1.0 - smooth_progress)


def _diagnostic_prelift_controller_restore_ready(
    *,
    enabled: bool,
    grasp_pose_rebased: bool,
    centroidal_yield_blend: float,
    joint_drive_yield_blend: float,
    admittance_active: bool,
    base_linear_speed_mps: float,
    base_speed_limit_mps: float,
) -> bool:
    """Gate a diagnostic LIFT until the ordinary controller is restored.

    This is an acceptance-ineligible isolation switch.  It makes the intended
    ordering explicit without silently changing the production transition:
    measured grasp rebase, complete centroidal/joint-drive gain restoration,
    no contact admittance, and a slow common base motion must all precede the
    normal contact dwell that authorizes LIFT.
    """

    if not enabled:
        return True
    blend = float(centroidal_yield_blend)
    joint_blend = float(joint_drive_yield_blend)
    speed = float(base_linear_speed_mps)
    speed_limit = float(base_speed_limit_mps)
    if not all(math.isfinite(value) for value in (blend, joint_blend, speed)):
        raise SchemaValidationError(
            "Order8 diagnostic prelift controller state must be finite"
        )
    if not math.isfinite(speed_limit) or speed_limit <= 0.0:
        raise SchemaValidationError(
            "Order8 diagnostic prelift base-speed limit must be finite and positive"
        )
    if not 0.0 <= blend <= 1.0 or not 0.0 <= joint_blend <= 1.0:
        raise SchemaValidationError(
            "Order8 diagnostic prelift yield blends must be in [0, 1]"
        )
    return bool(
        grasp_pose_rebased
        and blend <= 1.0e-12
        and joint_blend <= 1.0e-12
        and not admittance_active
        and speed <= speed_limit
    )


def _diagnostic_delayed_lift_bias_progress_scale(
    phase: Order8NaturalContactPhase,
    *,
    enabled: bool,
    phase_elapsed_s: float,
    bias_delay_s: float,
    transition_duration_s: float,
    normal_commanded_progress_scale: float,
) -> float:
    """Delay only the extra LIFT bias in a diagnostic separation run."""

    elapsed = float(phase_elapsed_s)
    delay = float(bias_delay_s)
    duration = float(transition_duration_s)
    normal_progress = float(normal_commanded_progress_scale)
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise SchemaValidationError(
            "Order8 diagnostic LIFT phase elapsed time must be finite and non-negative"
        )
    if not math.isfinite(delay) or delay < 0.0:
        raise SchemaValidationError(
            "Order8 diagnostic LIFT-bias delay must be finite and non-negative"
        )
    if not math.isfinite(duration) or duration <= 0.0:
        raise SchemaValidationError(
            "Order8 diagnostic LIFT-bias transition duration must be positive"
        )
    if not math.isfinite(normal_progress) or not 0.0 <= normal_progress <= 1.0:
        raise SchemaValidationError(
            "Order8 diagnostic normal LIFT progress must be in [0, 1]"
        )
    if not enabled or phase != Order8NaturalContactPhase.LIFT:
        return normal_progress
    if elapsed <= delay:
        return 0.0
    return _contact_motion_entry_speed_scale(
        phase,
        phase_elapsed_s=elapsed - delay,
        transition_duration_s=duration,
    )


def _lift_acceleration_force_bias_world(
    *,
    payload_mass_kg: float,
    lift_payload_acceleration_mps2: float,
    scale: float,
) -> tuple[float, float, float]:
    """Return the transient world-vertical payload inertial force."""

    mass = float(payload_mass_kg)
    acceleration = float(lift_payload_acceleration_mps2)
    bounded_scale = float(scale)
    if not math.isfinite(mass) or mass <= 0.0:
        raise SchemaValidationError(
            "Order8 lift-acceleration payload mass must be finite and positive"
        )
    if not math.isfinite(acceleration) or acceleration <= 0.0:
        raise SchemaValidationError(
            "Order8 lift-acceleration must be finite and positive"
        )
    if not math.isfinite(bounded_scale) or not 0.0 <= bounded_scale <= 1.0:
        raise SchemaValidationError(
            "Order8 lift-acceleration scale must be finite and in [0, 1]"
        )
    return (0.0, 0.0, mass * acceleration * bounded_scale)


def _measured_object_lift_transfer_scale(
    *,
    qclose_object_pose: Pose7D,
    current_object_pose: Pose7D,
    transfer_distance_m: float,
) -> float:
    """Map measured object COM rise since q_close to payload support share."""

    distance = float(transfer_distance_m)
    if not math.isfinite(distance) or distance <= 0.0:
        raise SchemaValidationError(
            "Order8 payload load-transfer distance must be finite and positive"
        )
    for name, pose in (
        ("qclose_object_pose", qclose_object_pose),
        ("current_object_pose", current_object_pose),
    ):
        if len(pose) != 7 or not all(math.isfinite(float(value)) for value in pose):
            raise SchemaValidationError(f"Order8 {name} must be a finite Pose7D")
    rise_m = max(
        0.0,
        float(current_object_pose[2]) - float(qclose_object_pose[2]),
    )
    return min(1.0, rise_m / distance)


def _project_object_rotation_state(
    pose_world: Sequence[float],
    twist_world: Sequence[float],
    *,
    locked_orientation_xyzw: Sequence[float],
) -> tuple[Pose7D, tuple[float, float, float, float, float, float], float, float]:
    """Project only object orientation/angular velocity for fault isolation.

    Translation and linear velocity are passed through exactly.  The returned
    angular deviation and speed describe the state immediately before the
    projection, so the diagnostic remains auditable rather than hiding the
    rotational motion it removes.
    """

    pose = tuple(float(value) for value in pose_world)
    twist = tuple(float(value) for value in twist_world)
    locked = tuple(float(value) for value in locked_orientation_xyzw)
    if (
        len(pose) != 7
        or len(twist) != 6
        or len(locked) != 4
        or not all(math.isfinite(value) for value in (*pose, *twist, *locked))
    ):
        raise SchemaValidationError(
            "Order8 object-rotation projection requires finite pose/twist/quaternion"
        )
    current_norm = math.sqrt(sum(value * value for value in pose[3:7]))
    locked_norm = math.sqrt(sum(value * value for value in locked))
    if current_norm <= 1.0e-12 or locked_norm <= 1.0e-12:
        raise SchemaValidationError(
            "Order8 object-rotation projection requires nonzero quaternions"
        )
    current_unit = tuple(value / current_norm for value in pose[3:7])
    locked_unit = tuple(value / locked_norm for value in locked)
    quaternion_dot = abs(
        sum(
            current * target
            for current, target in zip(current_unit, locked_unit, strict=True)
        )
    )
    angular_deviation_rad = 2.0 * math.acos(_clip(quaternion_dot, 0.0, 1.0))
    angular_speed_rad_s = _norm(twist[3:])
    projected_pose: Pose7D = (
        pose[0],
        pose[1],
        pose[2],
        locked_unit[0],
        locked_unit[1],
        locked_unit[2],
        locked_unit[3],
    )
    projected_twist = (twist[0], twist[1], twist[2], 0.0, 0.0, 0.0)
    return (
        projected_pose,
        projected_twist,
        angular_deviation_rad,
        angular_speed_rad_s,
    )


def _natural_contact_payload_coupling(
    *,
    control_body_pose_world: Pose7D,
    object_com_pose_world: Pose7D,
    object_mass_kg: float,
    object_size_m: Sequence[float],
    load_transfer_scale: float,
    contact_model: str,
) -> PayloadCoupling | None:
    """Build the QPID payload model from the measured free-object state.

    The object remains a free rigid body and no constraint is authored.  Once
    verified natural contact begins transferring load, QPID nevertheless needs
    the equivalent payload force *and moment* at the robot centroidal frame.
    The coupling mass/inertia is scaled continuously during transfer, while the
    measured object COM pose supplies the current lever arm and orientation.
    """

    scale = float(load_transfer_scale)
    mass = float(object_mass_kg)
    size = tuple(float(value) for value in object_size_m)
    if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
        raise SchemaValidationError(
            "Order8 payload load-transfer scale must be finite and in [0, 1]"
        )
    if not math.isfinite(mass) or mass <= 0.0:
        raise SchemaValidationError("Order8 payload mass must be finite and positive")
    if len(size) != 3 or not all(
        math.isfinite(value) and value > 0.0 for value in size
    ):
        raise SchemaValidationError(
            "Order8 payload size must contain three finite positive values"
        )
    if not isinstance(contact_model, str) or not contact_model:
        raise SchemaValidationError("Order8 payload contact model must be non-empty")
    if scale == 0.0:
        return None

    effective_mass = mass * scale
    sx, sy, sz = size
    inertia_object = (
        (effective_mass * (sy * sy + sz * sz) / 12.0, 0.0, 0.0),
        (0.0, effective_mass * (sx * sx + sz * sz) / 12.0, 0.0),
        (0.0, 0.0, effective_mass * (sx * sx + sy * sy) / 12.0),
    )
    body_from_object_pose = compose_pose(
        inverse_pose(control_body_pose_world),
        object_com_pose_world,
    )
    body_from_object_rotation = transform_from_pose(body_from_object_pose).rotation
    inertia_body = matmul(
        matmul(body_from_object_rotation, inertia_object),
        transpose(body_from_object_rotation),
    )
    coupling = PayloadCoupling(
        payload_id="order8_free_object",
        contact_model=contact_model,
        mass_kg=effective_mass,
        inertia_body=[
            inertia_body[0][0],
            inertia_body[0][1],
            inertia_body[0][2],
            inertia_body[1][1],
            inertia_body[1][2],
            inertia_body[2][2],
        ],
        com_offset_body=tuple(body_from_object_pose[:3]),
        coupling_mode="natural_contact_verified_grasp_ramped_payload_v2",
    )
    coupling.validate()
    return coupling


def _diagnostic_payload_coupling_component_view(
    coupling: PayloadCoupling,
    *,
    component_mode: str,
) -> PayloadCoupling:
    """Return an acceptance-ineligible component view of one payload model.

    The payload mass and therefore its complete translational force contribution
    remain identical in every enabled view.  Zeroing the lever arm removes only
    ``r x F``; zeroing inertia removes only the payload rotational-inertia term.
    This isolates QPID wrench components without changing the production
    payload model or bypassing the normal payload-coupling allocation path.
    """

    coupling.validate()
    mode = str(component_mode)
    if mode not in ORDER8_DIAGNOSTIC_PAYLOAD_COUPLING_COMPONENT_MODES:
        raise SchemaValidationError(
            "Order8 diagnostic payload coupling component mode must be one of "
            f"{ORDER8_DIAGNOSTIC_PAYLOAD_COUPLING_COMPONENT_MODES!r}"
        )
    if mode == ORDER8_DIAGNOSTIC_PAYLOAD_COUPLING_FULL:
        return coupling
    if mode == ORDER8_DIAGNOSTIC_PAYLOAD_COUPLING_FORCE_ONLY:
        return replace(
            coupling,
            inertia_body=[0.0] * 6,
            com_offset_body=(0.0, 0.0, 0.0),
            coupling_mode="diagnostic_translational_payload_force_only_v1",
        )
    return replace(
        coupling,
        inertia_body=[0.0] * 6,
        coupling_mode=(
            "diagnostic_translational_payload_force_and_com_offset_moment_v1"
        ),
    )


def _diagnostic_payload_coupling_component_flags(
    component_mode: str,
) -> dict[str, bool]:
    """Describe the exact payload terms retained by a diagnostic view."""

    mode = str(component_mode)
    if mode not in ORDER8_DIAGNOSTIC_PAYLOAD_COUPLING_COMPONENT_MODES:
        raise SchemaValidationError(
            "Order8 diagnostic payload coupling component mode must be one of "
            f"{ORDER8_DIAGNOSTIC_PAYLOAD_COUPLING_COMPONENT_MODES!r}"
        )
    return {
        "translational_force": True,
        "com_offset_moment": mode
        != ORDER8_DIAGNOSTIC_PAYLOAD_COUPLING_FORCE_ONLY,
        "rotational_inertia": mode
        == ORDER8_DIAGNOSTIC_PAYLOAD_COUPLING_FULL,
    }


def _contact_motion_entry_speed_scale(
    phase: Order8NaturalContactPhase,
    *,
    phase_elapsed_s: float,
    transition_duration_s: float,
) -> float:
    """Ramp maintained-contact motion immediately after each phase entry.

    In particular, LIFT must begin unloading the object support before payload
    feed-forward can represent carried mass.  Holding the pose while adding a
    time-driven payload model creates a false gravity force and moment.
    """

    elapsed = float(phase_elapsed_s)
    duration = float(transition_duration_s)
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise SchemaValidationError(
            "Order8 contact-motion phase elapsed time must be finite and non-negative"
        )
    if not math.isfinite(duration) or duration <= 0.0:
        raise SchemaValidationError(
            "Order8 contact-motion transition duration must be finite and positive"
        )
    if phase in {
        Order8NaturalContactPhase.LIFT,
        Order8NaturalContactPhase.TRANSPORT,
        Order8NaturalContactPhase.PLACE,
    }:
        return min(1.0, elapsed / duration)
    return 1.0


def _contact_required_motion_safety_authorized(
    *,
    nominal_command_dwell_complete: bool,
    privileged_grasp_dwell_acquired: bool,
) -> bool:
    """Fail closed before lift without exposing raw truth to policy/QPID.

    Geometry, relative motion, and Dock load remain the nominal planner-ready
    signal.  The privileged contact monitor is a separate deterministic safety
    interlock: contact-required motion is not released until it has verified
    the required two-link raw-contact dwell.
    """

    return bool(nominal_command_dwell_complete and privileged_grasp_dwell_acquired)


def _whole_structure_runtime_observation(
    *,
    time_s: float,
    morphology_graph: MorphologyGraph,
    module_states_by_id: Mapping[int, ModuleRuntimeState],
    controller_status: ControllerStatus,
    phase_label: str,
) -> RuntimeObservation:
    """Build one controller observation for the complete connected morphology."""

    expected_module_ids = tuple(
        sorted(int(module.module_id) for module in morphology_graph.modules)
    )
    observed_module_ids = tuple(sorted(int(value) for value in module_states_by_id))
    if not expected_module_ids:
        raise SchemaValidationError(
            "Order8 whole-structure observation requires at least one module"
        )
    if observed_module_ids != expected_module_ids:
        raise SchemaValidationError(
            "Order8 whole-structure observation must cover exactly the "
            f"connected morphology modules: expected={expected_module_ids}, "
            f"observed={observed_module_ids}"
        )
    ordered_states = [
        module_states_by_id[module_id] for module_id in expected_module_ids
    ]
    mismatched_state_ids = [
        (module_id, int(state.module_id))
        for module_id, state in zip(expected_module_ids, ordered_states, strict=True)
        if int(state.module_id) != module_id
    ]
    if mismatched_state_ids:
        raise SchemaValidationError(
            "Order8 whole-structure observation state ids do not match their keys: "
            f"{mismatched_state_ids}"
        )
    return RuntimeObservation(
        time_s=float(time_s),
        morphology_graph=morphology_graph,
        module_states=ordered_states,
        object_states=[],
        contact_states=[],
        controller_status=controller_status,
        task_progress=TaskProgressState(phase_label=str(phase_label)),
    )


def _centroidal_measured_joint_reference(
    *,
    expected_joint_ids: Sequence[str],
    actuator_position_targets: Mapping[str, float],
    measured_joint_positions: Mapping[str, float],
) -> dict[str, float]:
    """Build the quasi-static QPID shape from measured, not commanded, joints.

    Dock position commands are executed independently from the centroidal
    thruster QP.  In particular, a provisional object contact may prevent a
    joint from reaching its next command.  Feeding that *unreached* command into
    the target rigid-body model makes QPID compensate a shape change that never
    happened and can drive the free base away from its settled grasp pose.

    The target model therefore uses the measured absolute Dock state for the
    current morphology and applies only the planner's base-pose target.  The
    actuator bridge still receives ``actuator_position_targets`` unchanged.
    This is the intended quasi-static separation: QPID remains unaware of the
    commanded joint motion while retaining morphology-correct mass, inertia,
    rotor geometry, and base-to-centroidal-frame conversion.
    """

    ordered_ids = tuple(str(joint_id) for joint_id in expected_joint_ids)
    expected = set(ordered_ids)
    if (
        not ordered_ids
        or len(expected) != len(ordered_ids)
        or any(not joint_id for joint_id in ordered_ids)
    ):
        raise SchemaValidationError(
            "Order8 centroidal geometric joint ids must be non-empty and unique"
        )
    for label, values in (
        ("actuator position targets", actuator_position_targets),
        ("measured joint positions", measured_joint_positions),
    ):
        if set(values) != expected:
            raise SchemaValidationError(
                f"Order8 centroidal {label} must cover exactly the Dock joints"
            )
        if not all(math.isfinite(float(value)) for value in values.values()):
            raise SchemaValidationError(
                f"Order8 centroidal {label} must contain finite values"
            )
    return {
        joint_id: float(measured_joint_positions[joint_id]) for joint_id in ordered_ids
    }


def _dock_joint_armature_setting(
    *,
    simulation_drive: Mapping[str, object],
    diagnostic_override_kg_m2: object | None,
    diagnostic_only: bool,
) -> tuple[float, float | None, str]:
    """Resolve a simulator-only Dock armature without changing motor limits."""

    configured_present = "armature_kg_m2" in simulation_drive
    configured = float(simulation_drive.get("armature_kg_m2", 0.0))
    if not math.isfinite(configured) or configured < 0.0:
        raise SchemaValidationError(
            "configured Dock joint armature must be finite and non-negative"
        )
    if diagnostic_override_kg_m2 is None:
        return (
            configured,
            None,
            (
                "joint_actuator_simulation_drive_armature_kg_m2_v1"
                if configured_present
                else "implicit_zero_armature_v1"
            ),
        )
    if not diagnostic_only:
        raise SchemaValidationError(
            "Dock joint armature override requires diagnostic-only mode"
        )
    diagnostic = float(diagnostic_override_kg_m2)
    if not math.isfinite(diagnostic) or diagnostic <= 0.0:
        raise SchemaValidationError(
            "diagnostic Dock joint armature must be finite and positive"
        )
    return (
        diagnostic,
        diagnostic,
        "acceptance_ineligible_diagnostic_armature_override_v1",
    )


@dataclass(frozen=True)
class _SelectedMeshLocalAABB:
    module_id: int
    link_id: str
    primitive_id: str
    geometry_ref: str
    minimum_local: tuple[float, float, float]
    maximum_local: tuple[float, float, float]
    surface_sample_points_local: tuple[tuple[float, float, float], ...] = ()


@dataclass(frozen=True)
class _Order8DiagnosticProxyPadSpec:
    """Finite selected-surface collider attached to one existing rigid link."""

    module_id: int
    link_id: str
    center_local: tuple[float, float, float]
    orientation_local_xyzw: tuple[float, float, float, float]
    size_m: tuple[float, float, float]
    mesh_surface_projection_m: float
    inner_face_projection_m: float
    outer_face_projection_m: float
    tangential_surface_span_m: tuple[float, float]
    surface_sample_count: int
    near_surface_sample_count: int


@dataclass(frozen=True)
class _Order8DiagnosticConeProxyPadSpec:
    """One approved cone micro-pad attached to a selected Dock rigid link."""

    module_id: int
    link_id: str
    pad_id: str
    center_local: tuple[float, float, float]
    representative_surface_point_local: tuple[float, float, float]
    orientation_local_xyzw: tuple[float, float, float, float]
    size_m: tuple[float, float, float]
    outward_normal_local: tuple[float, float, float]
    axial_band_index: int
    circumferential_segment_index: int
    inner_face_surface_gap_m: float
    surface_fit_max_gap_m: float
    source_geometry_refs: tuple[str, ...]


@dataclass(frozen=True)
class _MeshAwareStagingPlan:
    base_pose_world: Pose7D
    retreat_distance_m: float
    predicted_clearance_m: float
    approach_axis_world: tuple[float, float, float]


@dataclass(frozen=True)
class _MeshAwareAnchorOpeningPlan:
    anchor_poses_base: dict[int, Pose7D]
    outward_distance_m_by_anchor: dict[int, float]
    predicted_clearance_m_by_anchor: dict[int, float]


@dataclass(frozen=True)
class _FloorClearGraspBasePlan:
    base_pose_world: Pose7D
    unconstrained_base_pose_world: Pose7D
    vertical_correction_m: float
    normal_correction_m_by_anchor: dict[int, float]
    tangential_correction_m_by_anchor: dict[int, tuple[float, float]]


@dataclass(frozen=True)
class _QCloseCheckpointState:
    """Acceptance-ineligible exact simulator state for fast force debugging."""

    module_root_poses: dict[int, Pose7D]
    module_root_velocities: dict[int, tuple[float, float, float, float, float, float]]
    joint_positions_rad: dict[str, float]
    joint_velocities_radps: dict[str, float]
    object_pose: Pose7D
    object_twist: tuple[float, float, float, float, float, float]
    anchor_hold_poses_base: dict[int, Pose7D]


@dataclass(frozen=True)
class _IsaacContactVectorTelemetry:
    """Privileged vector telemetry used only for contact fault isolation."""

    valid: bool
    normal_force_world_by_link: dict[str, tuple[float, float, float]]
    normal_force_application_point_world_by_link: dict[
        str, tuple[float, float, float]
    ]
    friction_force_world_by_link: dict[str, tuple[float, float, float]]
    contact_force_matrix_world_by_link: dict[str, tuple[float, float, float]]
    body_linear_velocity_world_by_link: dict[str, tuple[float, float, float]]
    body_contact_velocity_world_by_link: dict[str, tuple[float, float, float]]
    object_contact_velocity_world_by_link: dict[str, tuple[float, float, float]]
    relative_contact_velocity_world_by_link: dict[
        str, tuple[float, float, float]
    ]
    tangential_slip_velocity_world_by_link: dict[
        str, tuple[float, float, float]
    ]
    tangential_slip_contact_point_world_by_link: dict[
        str, tuple[float, float, float]
    ]
    tangential_slip_contact_normal_world_by_link: dict[
        str, tuple[float, float, float]
    ]


def _finite_checkpoint_vector(
    value: object,
    *,
    length: int,
    label: str,
) -> tuple[float, ...]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != length
        or not all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            for item in value
        )
    ):
        raise RuntimeError(
            f"Order8 q_close checkpoint {label} must contain {length} finite numbers"
        )
    return tuple(float(item) for item in value)


def _finite_checkpoint_map(
    value: object,
    *,
    vector_length: int,
    label: str,
) -> dict[int, tuple[float, ...]]:
    if not isinstance(value, dict) or not value:
        raise RuntimeError(f"Order8 q_close checkpoint {label} must be a non-empty map")
    parsed: dict[int, tuple[float, ...]] = {}
    for raw_key, raw_vector in value.items():
        try:
            key = int(raw_key)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Order8 q_close checkpoint {label} keys must be integer ids"
            ) from exc
        if str(key) != str(raw_key) and raw_key != key:
            raise RuntimeError(
                f"Order8 q_close checkpoint {label} keys must be canonical integer ids"
            )
        parsed[key] = _finite_checkpoint_vector(
            raw_vector,
            length=vector_length,
            label=f"{label}[{key}]",
        )
    return parsed


def _finite_checkpoint_scalar_map(
    value: object,
    *,
    label: str,
) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise RuntimeError(f"Order8 q_close checkpoint {label} must be a non-empty map")
    parsed: dict[str, float] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
        ):
            raise RuntimeError(
                f"Order8 q_close checkpoint {label} must map ids to finite numbers"
            )
        parsed[key] = float(item)
    return parsed


def _parse_qclose_checkpoint_state(raw: object) -> _QCloseCheckpointState | None:
    if raw is None:
        return None
    try:
        payload = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Order8 diagnostic q_close checkpoint state must be valid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Order8 diagnostic q_close checkpoint state must be a map")
    if payload.get("schema_version") != "order8_qclose_checkpoint_state_v1":
        raise RuntimeError(
            "Order8 diagnostic q_close checkpoint state version mismatch"
        )
    module_root_poses = _finite_checkpoint_map(
        payload.get("module_root_poses"),
        vector_length=7,
        label="module_root_poses",
    )
    module_root_velocities = _finite_checkpoint_map(
        payload.get("module_root_velocities"),
        vector_length=6,
        label="module_root_velocities",
    )
    anchor_hold_poses_base = _finite_checkpoint_map(
        payload.get("anchor_hold_poses_base"),
        vector_length=7,
        label="anchor_hold_poses_base",
    )
    return _QCloseCheckpointState(
        module_root_poses={
            key: tuple(value)  # type: ignore[dict-item]
            for key, value in module_root_poses.items()
        },
        module_root_velocities={
            key: tuple(value)  # type: ignore[dict-item]
            for key, value in module_root_velocities.items()
        },
        joint_positions_rad=_finite_checkpoint_scalar_map(
            payload.get("joint_positions_rad"),
            label="joint_positions_rad",
        ),
        joint_velocities_radps=_finite_checkpoint_scalar_map(
            payload.get("joint_velocities_radps"),
            label="joint_velocities_radps",
        ),
        object_pose=tuple(  # type: ignore[arg-type]
            _finite_checkpoint_vector(
                payload.get("object_pose"),
                length=7,
                label="object_pose",
            )
        ),
        object_twist=tuple(  # type: ignore[arg-type]
            _finite_checkpoint_vector(
                payload.get("object_twist"),
                length=6,
                label="object_twist",
            )
        ),
        anchor_hold_poses_base={
            key: tuple(value)  # type: ignore[dict-item]
            for key, value in anchor_hold_poses_base.items()
        },
    )


def _qclose_checkpoint_state_to_dict(
    state: _QCloseCheckpointState,
) -> dict[str, object]:
    return {
        "schema_version": "order8_qclose_checkpoint_state_v1",
        "module_root_poses": {
            str(module_id): list(pose)
            for module_id, pose in sorted(state.module_root_poses.items())
        },
        "module_root_velocities": {
            str(module_id): list(velocity)
            for module_id, velocity in sorted(state.module_root_velocities.items())
        },
        "joint_positions_rad": dict(sorted(state.joint_positions_rad.items())),
        "joint_velocities_radps": dict(sorted(state.joint_velocities_radps.items())),
        "object_pose": list(state.object_pose),
        "object_twist": list(state.object_twist),
        "anchor_hold_poses_base": {
            str(anchor_id): list(pose)
            for anchor_id, pose in sorted(state.anchor_hold_poses_base.items())
        },
    }


def format_order8_progress(phase: str, simulation_time_s: float) -> str:
    if not phase:
        raise SchemaValidationError("Order8 progress phase must be non-empty")
    if not math.isfinite(float(simulation_time_s)) or simulation_time_s < 0.0:
        raise SchemaValidationError(
            "Order8 progress time must be finite and non-negative"
        )
    return (
        f"{ORDER8_PROGRESS_PREFIX} simulation_time={float(simulation_time_s):.3f}s "
        f"phase={phase}"
    )


def _apply_order8_seed(seed: int, *, torch: Any) -> dict[str, object]:
    """Apply the declared rollout seed to every stochastic host backend used."""

    import random

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise SchemaValidationError("Order8 seed must be a non-negative integer")
    try:
        import numpy
    except ImportError as exc:  # pragma: no cover - Isaac runtime always ships NumPy.
        raise RuntimeError(
            "Order8 deterministic rollout requires NumPy seeding"
        ) from exc
    random.seed(seed)
    numpy.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        torch.cuda.manual_seed_all(seed)
    return {
        "seed": seed,
        "python_random": True,
        "torch": True,
        "torch_cuda": cuda_available,
        "numpy": True,
    }


def run_order8_isaac_runtime(
    *,
    args: Any,
    sim_utils: Any,
    SimulationContext: Any,
    Articulation: Any,
    ArticulationCfg: Any,
    ImplicitActuatorCfg: Any,
    RigidObject: Any,
    RigidObjectCfg: Any,
    usd_path: Any,
    urdf_path: Any,
    physical_model: Any,
    morphology_graph: Any,
    config: Any,
    gimbal_stiffness: float,
    gimbal_damping: float,
    dock_stiffness: float,
    dock_damping: float,
    backend_config_hash: str,
    collision_approximation_evidence: dict[str, object],
    device: str,
) -> dict[str, object]:
    """Run one deterministic natural-contact substrate episode.

    The acceptance path never writes the object after spawn and authors no
    object constraint.  The acceptance-ineligible exact-state diagnostic may
    restore the measured free-object state once after simulator reset.
    Three independent module articulations remain articulated and are connected
    only at occupied graph DockEdges by exact external FixedJoints.  The
    acceptance-ineligible force-path fixture additionally fixes the base-module
    frame and object to world so centroidal flight and free-object transients
    cannot mask Dock impedance behavior during a short diagnostic.
    """

    import torch
    import warp as wp
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    from amsrr.controllers.actuator_mapping import build_actuator_mapping
    from amsrr.controllers.controller_base import ControllerContext
    from amsrr.controllers.isaac_controller_bridge import IsaacControllerBridge
    from amsrr.controllers.natural_contact_joint_controller import (
        DockJointVector,
        NaturalContactJointController,
        NaturalContactJointControllerConfig,
        position_drive_peak_effort_lead_rad,
    )
    from amsrr.controllers.qpid_controller import QPIDController, QPIDControllerConfig
    from amsrr.controllers.rigid_body_model import RigidBodyControlModelBuilder
    from amsrr.feasibility.morphology_flight import (
        collision_geometry_content_hash,
    )
    from amsrr.policies.deterministic_natural_contact_planner import (
        DeterministicNaturalContactPlanner,
        NaturalContactAnchorSelection,
        NaturalContactPlannerConfig,
        NaturalContactPlannerFeedback,
        ORDER8_FREE_MORPH_ANCHOR_ORIENTATION_WEIGHT,
    )
    from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
    from amsrr.robot_model.gripper_surfaces import (
        select_opposing_gripper_surface_pair,
    )
    from amsrr.robot_model.whole_structure_kinematics import (
        MeshBackedAnchorReference,
        WholeStructureKinematics,
        ordered_global_dock_joint_ids,
    )
    from amsrr.robot_model.urdf_loader import load_urdf
    from amsrr.robot_model.urdf_transforms import link_poses_in_root_frame
    from amsrr.schemas.common import ContactMode
    from amsrr.schemas.contact_candidates import ContactCandidate, ContactCandidateSet
    from amsrr.schemas.order8 import (
        ORDER8_NATURAL_CONTACT_MODEL,
        ORDER8_NATURAL_CONTACT_OBSERVATION_VERSION,
        ORDER8_RAW_CONTACT_TRUTH_ROLE,
        Order8NaturalContactObservation,
    )
    from amsrr.schemas.policies import (
        POLICY_COMMAND_CONTRACT_CENTROIDAL,
        InteractionKnot,
        PolicyCommand,
    )
    from amsrr.simulation.dynamic_dock_constraint import (
        build_dynamic_dock_constraint_spec,
        fixed_joint_identity_failures,
        preauthor_disabled_fixed_joint,
    )
    from amsrr.simulation.natural_contact_evidence import (
        NaturalContactEvidenceMonitor,
    )
    from amsrr.simulation.random_morphology_takeoff import (
        ORDER2_FLOOR_POSE_WORLD,
        ORDER2_FLOOR_SIZE_M,
        compute_floor_contact_placement,
    )
    from amsrr.utils.hashing import hash_directory_manifest, hash_file, stable_hash

    config.validate()
    diagnostic_only = bool(getattr(args, "order8_diagnostic_only", False))
    diagnostic_separated_lift_transition = bool(
        getattr(args, "order8_diagnostic_separated_lift_transition", False)
    )
    diagnostic_lift_bias_delay_s = float(
        getattr(args, "order8_diagnostic_lift_bias_delay_s", 0.0)
    )
    diagnostic_disable_payload_feedforward = bool(
        getattr(args, "order8_diagnostic_disable_payload_feedforward", False)
    )
    diagnostic_payload_coupling_component_mode = str(
        getattr(
            args,
            "order8_diagnostic_payload_coupling_component_mode",
            ORDER8_DIAGNOSTIC_PAYLOAD_COUPLING_FULL,
        )
    )
    if diagnostic_separated_lift_transition and not diagnostic_only:
        raise RuntimeError(
            "Order8 separated LIFT transition requires diagnostic-only mode"
        )
    if (
        not math.isfinite(diagnostic_lift_bias_delay_s)
        or diagnostic_lift_bias_delay_s < 0.0
    ):
        raise RuntimeError(
            "Order8 diagnostic LIFT-bias delay must be finite and non-negative"
        )
    if diagnostic_lift_bias_delay_s > 0.0 and not (
        diagnostic_separated_lift_transition
    ):
        raise RuntimeError(
            "Order8 diagnostic LIFT-bias delay requires separated transition mode"
        )
    if diagnostic_disable_payload_feedforward and not (
        diagnostic_separated_lift_transition
    ):
        raise RuntimeError(
            "Order8 payload feed-forward A/B requires separated transition mode"
        )
    if (
        diagnostic_payload_coupling_component_mode
        not in ORDER8_DIAGNOSTIC_PAYLOAD_COUPLING_COMPONENT_MODES
    ):
        raise RuntimeError(
            "Order8 diagnostic payload coupling component mode must be one of "
            f"{ORDER8_DIAGNOSTIC_PAYLOAD_COUPLING_COMPONENT_MODES!r}"
        )
    if (
        diagnostic_payload_coupling_component_mode
        != ORDER8_DIAGNOSTIC_PAYLOAD_COUPLING_FULL
        and not diagnostic_separated_lift_transition
    ):
        raise RuntimeError(
            "Order8 diagnostic payload component isolation requires separated "
            "transition mode"
        )
    if (
        diagnostic_disable_payload_feedforward
        and diagnostic_payload_coupling_component_mode
        != ORDER8_DIAGNOSTIC_PAYLOAD_COUPLING_FULL
    ):
        raise RuntimeError(
            "Order8 disabled payload feed-forward and component isolation are "
            "mutually exclusive"
        )
    diagnostic_legacy_proxy_pad_enabled = bool(
        getattr(args, "order8_diagnostic_proxy_pad", False)
    )
    diagnostic_cone_proxy_pad_enabled = bool(
        getattr(args, "order8_diagnostic_cone_proxy_pad", False)
    )
    if diagnostic_legacy_proxy_pad_enabled and diagnostic_cone_proxy_pad_enabled:
        raise RuntimeError("legacy and cone-only Order8 proxy pads are mutually exclusive")
    diagnostic_proxy_pad_enabled = bool(
        diagnostic_legacy_proxy_pad_enabled or diagnostic_cone_proxy_pad_enabled
    )
    if diagnostic_proxy_pad_enabled and not diagnostic_only:
        raise RuntimeError("Order8 proxy pads require diagnostic-only mode")
    state_trace_output_raw = getattr(args, "order8_state_trace_output", None)
    state_trace_replay_raw = getattr(args, "order8_state_trace_replay", None)
    state_trace_replay_sync_physics = bool(
        getattr(args, "order8_state_trace_replay_sync_physics", False)
    )
    if state_trace_output_raw is not None and state_trace_replay_raw is not None:
        raise RuntimeError(
            "Order8 state-trace capture and replay are mutually exclusive"
        )
    if (
        state_trace_output_raw is not None or state_trace_replay_raw is not None
    ) and not diagnostic_only:
        raise RuntimeError(
            "Order8 state-trace capture/replay requires diagnostic-only mode"
        )
    if state_trace_replay_sync_physics and state_trace_replay_raw is None:
        raise RuntimeError(
            "Order8 state-trace PhysX synchronization requires replay mode"
        )
    if (
        diagnostic_legacy_proxy_pad_enabled
        and state_trace_replay_raw is not None
    ):
        raise RuntimeError(
            "Order8 legacy proxy pads require live physics and cannot be used "
            "in state replay"
        )
    if (
        diagnostic_cone_proxy_pad_enabled
        and state_trace_replay_raw is not None
        and not state_trace_replay_sync_physics
    ):
        raise RuntimeError(
            "Order8 cone proxy state replay requires contact-minimized PhysX "
            "synchronization"
        )
    state_trace_output_path = (
        None
        if state_trace_output_raw is None
        else Path(str(state_trace_output_raw)).expanduser().resolve()
    )
    state_trace_replay_path = (
        None
        if state_trace_replay_raw is None
        else Path(str(state_trace_replay_raw)).expanduser().resolve()
    )
    state_trace_frame_stride = int(
        getattr(args, "order8_state_trace_frame_stride", 2)
    )
    state_trace_replay_speed = float(
        getattr(args, "order8_state_trace_replay_speed", 1.0)
    )
    state_trace_replay_loops = int(
        getattr(args, "order8_state_trace_replay_loops", 1)
    )
    state_trace_replay_endpoint_hold_s = float(
        getattr(args, "order8_state_trace_replay_endpoint_hold_s", 0.0)
    )
    if state_trace_frame_stride <= 0:
        raise RuntimeError("Order8 state-trace frame stride must be positive")
    if not math.isfinite(state_trace_replay_speed) or state_trace_replay_speed <= 0.0:
        raise RuntimeError("Order8 state-trace replay speed must be positive")
    if state_trace_replay_loops <= 0:
        raise RuntimeError("Order8 state-trace replay loops must be positive")
    if (
        not math.isfinite(state_trace_replay_endpoint_hold_s)
        or state_trace_replay_endpoint_hold_s < 0.0
    ):
        raise RuntimeError(
            "Order8 state-trace endpoint hold must be finite and non-negative"
        )
    diagnostic_continue_after_force_ramp = bool(
        getattr(args, "order8_diagnostic_continue_after_force_ramp", False)
    )
    if diagnostic_continue_after_force_ramp and not diagnostic_only:
        raise RuntimeError(
            "Order8 diagnostic continuation requires diagnostic-only mode"
        )
    diagnostic_force_fixture = bool(
        getattr(args, "order8_diagnostic_force_fixture", False)
    )
    diagnostic_world_fixed_object_requested = bool(
        getattr(args, "order8_diagnostic_world_fixed_object", False)
    )
    diagnostic_lock_object_rotation = bool(
        getattr(args, "order8_diagnostic_lock_object_rotation", False)
    )
    diagnostic_anchor_hold_joint_correction = bool(
        getattr(
            args,
            "order8_diagnostic_anchor_hold_joint_correction",
            False,
        )
    )
    diagnostic_loaded_state_rebase = bool(
        getattr(
            args,
            "order8_diagnostic_loaded_state_rebase",
            False,
        )
    )
    diagnostic_kinematic_base_isolation = bool(
        getattr(args, "order8_diagnostic_kinematic_base_isolation", False)
    )
    if diagnostic_force_fixture and not diagnostic_only:
        raise RuntimeError(
            "Order8 diagnostic force fixture requires diagnostic-only mode"
        )
    if diagnostic_world_fixed_object_requested and not diagnostic_only:
        raise RuntimeError(
            "Order8 world-fixed object fixture requires diagnostic-only mode"
        )
    if diagnostic_lock_object_rotation and not diagnostic_only:
        raise RuntimeError(
            "Order8 object-rotation lock requires diagnostic-only mode"
        )
    if diagnostic_lock_object_rotation and diagnostic_world_fixed_object_requested:
        raise RuntimeError(
            "Order8 object-rotation lock cannot be combined with a world-fixed object"
        )
    if diagnostic_anchor_hold_joint_correction and not diagnostic_only:
        raise RuntimeError(
            "Order8 anchor-hold joint correction requires diagnostic-only mode"
        )
    if diagnostic_anchor_hold_joint_correction and diagnostic_lock_object_rotation:
        raise RuntimeError(
            "Order8 anchor-hold joint correction and object-rotation lock are "
            "mutually exclusive causal diagnostics"
        )
    if diagnostic_loaded_state_rebase and not diagnostic_only:
        raise RuntimeError(
            "Order8 loaded-state rebase requires diagnostic-only mode"
        )
    if diagnostic_loaded_state_rebase and not diagnostic_separated_lift_transition:
        raise RuntimeError(
            "Order8 loaded-state rebase requires the separated LIFT transition"
        )
    if diagnostic_loaded_state_rebase and not diagnostic_continue_after_force_ramp:
        raise RuntimeError(
            "Order8 loaded-state rebase requires continuation after force ramp"
        )
    if diagnostic_loaded_state_rebase and diagnostic_anchor_hold_joint_correction:
        raise RuntimeError(
            "Order8 loaded-state rebase and continuous anchor-hold joint "
            "correction are mutually exclusive"
        )
    if diagnostic_loaded_state_rebase and diagnostic_lock_object_rotation:
        raise RuntimeError(
            "Order8 loaded-state rebase and object-rotation lock are mutually "
            "exclusive causal diagnostics"
        )
    diagnostic_precontact_pose_raw = getattr(
        args, "order8_diagnostic_precontact_base_pose", None
    )
    diagnostic_precontact_base_pose: Pose7D | None = None
    if diagnostic_precontact_pose_raw is not None:
        if not diagnostic_only or diagnostic_force_fixture:
            raise RuntimeError(
                "Order8 precontact fixture requires diagnostic-only mode and "
                "cannot be combined with the force fixture"
            )
        diagnostic_precontact_base_pose = tuple(
            float(value) for value in diagnostic_precontact_pose_raw
        )
        if len(diagnostic_precontact_base_pose) != 7 or not all(
            math.isfinite(value) for value in diagnostic_precontact_base_pose
        ):
            raise RuntimeError("Order8 diagnostic precontact pose must be finite")
        quaternion_norm = math.sqrt(
            sum(value * value for value in diagnostic_precontact_base_pose[3:7])
        )
        if abs(quaternion_norm - 1.0) > 1.0e-3:
            raise RuntimeError(
                "Order8 diagnostic precontact pose quaternion must be unit length"
            )
    diagnostic_precontact_fixture = diagnostic_precontact_base_pose is not None
    diagnostic_near_contact_pose_raw = getattr(
        args, "order8_diagnostic_near_contact_base_pose", None
    )
    diagnostic_near_contact_joint_positions_raw = getattr(
        args, "order8_diagnostic_near_contact_joint_positions_json", None
    )
    diagnostic_near_contact_object_pose_raw = getattr(
        args, "order8_diagnostic_near_contact_object_pose", None
    )
    near_contact_components_present = (
        diagnostic_near_contact_pose_raw is not None,
        diagnostic_near_contact_joint_positions_raw is not None,
        diagnostic_near_contact_object_pose_raw is not None,
    )
    if any(near_contact_components_present) and not all(
        near_contact_components_present
    ):
        raise RuntimeError(
            "Order8 near-contact fixture requires base pose, complete Dock "
            "state, and object pose together"
        )
    diagnostic_near_contact_base_pose: Pose7D | None = None
    diagnostic_near_contact_object_pose: Pose7D | None = None
    diagnostic_near_contact_joint_positions: dict[str, float] = {}
    if all(near_contact_components_present):
        if (
            not diagnostic_only
            or diagnostic_force_fixture
            or diagnostic_precontact_fixture
        ):
            raise RuntimeError(
                "Order8 near-contact fixture requires diagnostic-only mode and "
                "is mutually exclusive with precontact/force fixtures"
            )
        diagnostic_near_contact_base_pose = tuple(
            float(value) for value in diagnostic_near_contact_pose_raw
        )
        diagnostic_near_contact_object_pose = tuple(
            float(value) for value in diagnostic_near_contact_object_pose_raw
        )
        for name, pose in (
            ("base", diagnostic_near_contact_base_pose),
            ("object", diagnostic_near_contact_object_pose),
        ):
            if len(pose) != 7 or not all(math.isfinite(value) for value in pose):
                raise RuntimeError(
                    f"Order8 diagnostic near-contact {name} pose must be finite"
                )
            quaternion_norm = math.sqrt(
                sum(value * value for value in pose[3:7])
            )
            if abs(quaternion_norm - 1.0) > 1.0e-3:
                raise RuntimeError(
                    "Order8 diagnostic near-contact "
                    f"{name} pose quaternion must be unit length"
                )
        try:
            joint_payload = json.loads(
                str(diagnostic_near_contact_joint_positions_raw)
            )
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Order8 diagnostic near-contact joint positions must be valid JSON"
            ) from exc
        if not isinstance(joint_payload, dict) or not joint_payload:
            raise RuntimeError(
                "Order8 diagnostic near-contact joint positions must be a "
                "non-empty map"
            )
        for joint_id, value in joint_payload.items():
            if (
                not isinstance(joint_id, str)
                or not joint_id
                or not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise RuntimeError(
                    "Order8 diagnostic near-contact joint positions must map "
                    "non-empty ids to finite numbers"
                )
            diagnostic_near_contact_joint_positions[joint_id] = float(value)
    diagnostic_near_contact_fixture = (
        diagnostic_near_contact_base_pose is not None
    )
    if diagnostic_kinematic_base_isolation and not (
        diagnostic_near_contact_fixture or diagnostic_precontact_fixture
    ):
        raise RuntimeError(
            "Order8 kinematic base isolation requires a near-contact or "
            "precontact diagnostic fixture"
        )
    diagnostic_qclose_pose_raw = getattr(
        args, "order8_diagnostic_qclose_base_pose", None
    )
    diagnostic_qclose_joint_positions_raw = getattr(
        args, "order8_diagnostic_qclose_joint_positions_json", None
    )
    if (diagnostic_qclose_pose_raw is None) != (
        diagnostic_qclose_joint_positions_raw is None
    ):
        raise RuntimeError(
            "Order8 q_close fixture requires both base pose and joint positions"
        )
    diagnostic_qclose_base_pose: Pose7D | None = None
    diagnostic_qclose_joint_positions: dict[str, float] = {}
    if diagnostic_qclose_pose_raw is not None:
        if (
            not diagnostic_only
            or diagnostic_force_fixture
            or diagnostic_precontact_fixture
            or diagnostic_near_contact_fixture
        ):
            raise RuntimeError(
                "Order8 q_close fixture requires diagnostic-only mode and is "
                "mutually exclusive with other fixtures"
            )
        diagnostic_qclose_base_pose = tuple(
            float(value) for value in diagnostic_qclose_pose_raw
        )
        if len(diagnostic_qclose_base_pose) != 7 or not all(
            math.isfinite(value) for value in diagnostic_qclose_base_pose
        ):
            raise RuntimeError("Order8 diagnostic q_close pose must be finite")
        quaternion_norm = math.sqrt(
            sum(value * value for value in diagnostic_qclose_base_pose[3:7])
        )
        if abs(quaternion_norm - 1.0) > 1.0e-3:
            raise RuntimeError(
                "Order8 diagnostic q_close pose quaternion must be unit length"
            )
        try:
            joint_payload = json.loads(str(diagnostic_qclose_joint_positions_raw))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Order8 diagnostic q_close joint positions must be valid JSON"
            ) from exc
        if not isinstance(joint_payload, dict) or not joint_payload:
            raise RuntimeError(
                "Order8 diagnostic q_close joint positions must be a non-empty map"
            )
        for joint_id, value in joint_payload.items():
            if (
                not isinstance(joint_id, str)
                or not joint_id
                or not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise RuntimeError(
                    "Order8 diagnostic q_close joint positions must map non-empty "
                    "ids to finite numbers"
                )
            diagnostic_qclose_joint_positions[joint_id] = float(value)
    diagnostic_qclose_fixture = diagnostic_qclose_base_pose is not None
    diagnostic_qclose_checkpoint_state = _parse_qclose_checkpoint_state(
        getattr(args, "order8_diagnostic_qclose_state_json", None)
    )
    diagnostic_qclose_zero_velocities = bool(
        getattr(args, "order8_diagnostic_qclose_zero_velocities", False)
    )
    if diagnostic_qclose_checkpoint_state is not None and not diagnostic_qclose_fixture:
        raise RuntimeError(
            "Order8 exact q_close state requires the paired base pose and "
            "joint-position checkpoint"
        )
    if (
        diagnostic_qclose_zero_velocities
        and diagnostic_qclose_checkpoint_state is None
    ):
        raise RuntimeError(
            "Order8 zero-velocity q_close replay requires an exact checkpoint"
        )
    if diagnostic_continue_after_force_ramp and (
        not (
            diagnostic_precontact_fixture
            or diagnostic_near_contact_fixture
            or (
                diagnostic_qclose_fixture
                and diagnostic_qclose_checkpoint_state is not None
            )
        )
        or diagnostic_force_fixture
        or diagnostic_world_fixed_object_requested
    ):
        raise RuntimeError(
            "Order8 diagnostic continuation requires a free-object precontact "
            "or near-contact fixture, or exact q_close checkpoint"
        )
    diagnostic_object_width_padding_m = float(
        getattr(args, "order8_diagnostic_object_width_padding_m", 0.0)
    )
    if (
        not math.isfinite(diagnostic_object_width_padding_m)
        or diagnostic_object_width_padding_m < 0.0
    ):
        raise RuntimeError(
            "Order8 diagnostic object width padding must be finite and " "non-negative"
        )
    if diagnostic_object_width_padding_m > 0.0 and not diagnostic_force_fixture:
        raise RuntimeError(
            "Order8 diagnostic object width padding requires force fixture"
        )
    runtime_object_size_m = tuple(float(value) for value in config.object_size_m)
    if diagnostic_object_width_padding_m > 0.0:
        runtime_object_size_m = (
            runtime_object_size_m[0],
            runtime_object_size_m[1] + diagnostic_object_width_padding_m,
            runtime_object_size_m[2],
        )
    diagnostic_stop_force_scale = float(
        getattr(args, "order8_diagnostic_stop_force_scale", 0.40)
    )
    if not 0.0 < diagnostic_stop_force_scale <= 1.0:
        raise RuntimeError("Order8 diagnostic stop force scale must be in (0, 1]")
    diagnostic_profile_output = getattr(args, "order8_diagnostic_profile_output", None)
    if diagnostic_profile_output is not None and not diagnostic_only:
        raise RuntimeError(
            "Order8 diagnostic runtime profiling requires diagnostic-only mode"
        )
    diagnostic_force_anchor_ids_raw = getattr(
        args, "order8_diagnostic_force_anchor_id", None
    )
    diagnostic_force_anchor_ids = (
        None
        if diagnostic_force_anchor_ids_raw is None
        else tuple(int(value) for value in diagnostic_force_anchor_ids_raw)
    )
    if diagnostic_force_anchor_ids is not None and (
        not diagnostic_only
        or not diagnostic_force_anchor_ids
        or len(set(diagnostic_force_anchor_ids)) != len(diagnostic_force_anchor_ids)
        or any(value < 0 for value in diagnostic_force_anchor_ids)
    ):
        raise RuntimeError(
            "Order8 diagnostic force-anchor ids require diagnostic-only mode "
            "and must be unique non-negative ids"
        )
    seed_application = _apply_order8_seed(int(args.order8_seed), torch=torch)
    if len(morphology_graph.modules) != 3:
        raise RuntimeError("Order8 representative smoke requires exactly three modules")
    if len(morphology_graph.dock_edges) != 2:
        raise RuntimeError("Order8 representative smoke requires a two-edge tree")
    grasp_anchors = sorted(
        (
            anchor
            for anchor in morphology_graph.robot_anchors
            if anchor.anchor_type == "grasp"
        ),
        key=lambda anchor: anchor.anchor_id,
    )
    if len(grasp_anchors) != 2:
        raise RuntimeError(
            "Order8 representative smoke requires exactly two grasp anchors"
        )
    selected_pair = select_opposing_gripper_surface_pair(
        morphology_graph,
        physical_model,
    )
    selected_surfaces = (selected_pair.first, selected_pair.second)
    if len({surface.module_id for surface in selected_surfaces}) != 2:
        raise RuntimeError("Order8 selected gripper surfaces must use distinct modules")
    anchor_by_module = {anchor.module_id: anchor for anchor in grasp_anchors}
    anchor_references: list[MeshBackedAnchorReference] = []
    for surface in selected_surfaces:
        anchor = anchor_by_module.get(surface.module_id)
        if anchor is None:
            raise RuntimeError("selected gripper surface has no matching grasp anchor")
        anchor_references.append(
            MeshBackedAnchorReference(anchor=anchor, surface=surface)
        )
    selected_gripper_body_key_by_anchor = {
        int(reference.anchor.anchor_id): (
            int(reference.surface.module_id),
            str(reference.surface.mechanism_link_id),
        )
        for reference in anchor_references
    }
    selected_gripper_local_aabbs = _selected_gripper_mesh_local_aabbs(
        selected_surfaces,
        urdf_path=urdf_path,
    )
    diagnostic_proxy_pad_specs: tuple[
        _Order8DiagnosticProxyPadSpec | _Order8DiagnosticConeProxyPadSpec, ...
    ] = (
        _selected_gripper_cone_proxy_pad_specs(
            selected_surfaces,
            urdf_path=urdf_path,
        )
        if diagnostic_cone_proxy_pad_enabled
        else _selected_gripper_proxy_pad_specs(
            selected_surfaces,
            selected_gripper_local_aabbs,
            physical_model,
        )
        if diagnostic_legacy_proxy_pad_enabled
        else ()
    )
    selected_gripper_contact_local_surfaces = (
        _cone_proxy_pad_surface_local_meshes(diagnostic_proxy_pad_specs)
        if diagnostic_cone_proxy_pad_enabled
        else selected_gripper_local_aabbs
    )

    module_ids = sorted(module.module_id for module in morphology_graph.modules)
    singleton_graphs = {
        module_id: _singleton_graph(morphology_graph, module_id)
        for module_id in module_ids
    }
    placement = compute_floor_contact_placement(
        morphology_graph,
        physical_model,
        mesh_search_dirs=("module_urdf", "module_urdf/mesh"),
        floor_z_m=0.0,
        clearance_m=0.002,
    )
    source_urdf_model = load_urdf(physical_model.urdf_path)
    link_poses_root = link_poses_in_root_frame(source_urdf_model)
    module_frame_link_id = str(
        source_urdf_model.metadata.get("baselink", {}).get("name", "fc")
    )
    if module_frame_link_id not in link_poses_root:
        raise RuntimeError(
            f"Order8 module frame {module_frame_link_id!r} is missing from URDF"
        )
    root_to_module_frame = link_poses_root[module_frame_link_id]
    module_frame_to_root = inverse_pose(root_to_module_frame)
    floor_root_pose = tuple(float(value) for value in placement.root_pose_world)
    floor_base_pose = compose_pose(floor_root_pose, root_to_module_frame)
    hover_base_pose = _offset_pose(floor_base_pose, dz=0.50)
    module_by_id = {module.module_id: module for module in morphology_graph.modules}
    pair_center_design = tuple(
        0.5
        * (
            float(selected_pair.first.connect_frame_design[index])
            + float(selected_pair.second.connect_frame_design[index])
        )
        for index in range(3)
    )
    initial_pair_center = compose_pose(
        floor_base_pose,
        (*pair_center_design, 0.0, 0.0, 0.0, 1.0),
    )
    object_support_height_m = float(config.object_support_height_m)
    object_pose: Pose7D = (
        float(initial_pair_center[0]) + float(config.initial_object_standoff_m),
        float(initial_pair_center[1]),
        object_support_height_m + 0.5 * float(runtime_object_size_m[2]),
        0.0,
        0.0,
        0.0,
        1.0,
    )
    if diagnostic_near_contact_object_pose is not None:
        object_pose = diagnostic_near_contact_object_pose
    elif diagnostic_qclose_checkpoint_state is not None:
        object_pose = diagnostic_qclose_checkpoint_state.object_pose
    unconstrained_grasp_base_pose: Pose7D = (
        float(object_pose[0]) - float(pair_center_design[0]),
        float(object_pose[1]) - float(pair_center_design[1]),
        float(object_pose[2]) - float(pair_center_design[2]),
        *tuple(float(value) for value in floor_base_pose[3:7]),
    )
    grasp_base_plan = _floor_clear_grasp_base_plan(
        floor_base_pose=floor_base_pose,
        unconstrained_grasp_base_pose=unconstrained_grasp_base_pose,
        inward_normal_world_by_anchor={
            int(anchor_by_module[surface.module_id].anchor_id): tuple(
                float(value)
                for value in (
                    selected_pair.first_inward_axis_design
                    if surface is selected_pair.first
                    else selected_pair.second_inward_axis_design
                )
            )
            for surface in selected_surfaces
        },
        tangential_tolerance_m=float(config.contact_tangential_tolerance_m),
        additional_floor_clearance_m=(ORDER8_GRASP_ADDITIONAL_FLOOR_CLEARANCE_M),
    )
    grasp_base_pose = grasp_base_plan.base_pose_world
    initial_robot_base_pose = (
        grasp_base_pose
        if diagnostic_force_fixture
        else (
            diagnostic_qclose_base_pose
            if diagnostic_qclose_fixture
            else (
                diagnostic_near_contact_base_pose
                if diagnostic_near_contact_fixture
                else (
                    diagnostic_precontact_base_pose
                    if diagnostic_precontact_fixture
                    else hover_base_pose if diagnostic_only else floor_base_pose
                )
            )
        )
    )
    if diagnostic_qclose_fixture or diagnostic_near_contact_fixture:
        expected_fixture_joint_ids = ordered_global_dock_joint_ids(
            morphology_graph,
            physical_model,
        )
        fixture_joint_positions = (
            diagnostic_qclose_joint_positions
            if diagnostic_qclose_fixture
            else diagnostic_near_contact_joint_positions
        )
        if set(fixture_joint_positions) != set(expected_fixture_joint_ids):
            raise RuntimeError(
                "Order8 diagnostic checkpoint joint map does not cover the exact "
                "whole-structure Dock state"
            )
        if (
            diagnostic_qclose_fixture
            and diagnostic_qclose_checkpoint_state is not None
        ):
            exact_state = diagnostic_qclose_checkpoint_state
            if set(exact_state.module_root_poses) != set(module_ids) or set(
                exact_state.module_root_velocities
            ) != set(module_ids):
                raise RuntimeError(
                    "Order8 exact q_close state must cover every module root"
                )
            if set(exact_state.joint_positions_rad) != set(
                expected_fixture_joint_ids
            ) or set(exact_state.joint_velocities_radps) != set(
                expected_fixture_joint_ids
            ):
                raise RuntimeError(
                    "Order8 exact q_close state must cover the complete Dock q/qdot"
                )
            if set(exact_state.anchor_hold_poses_base) != {
                int(anchor.anchor_id) for anchor in grasp_anchors
            }:
                raise RuntimeError(
                    "Order8 exact q_close state must cover both selected anchors"
                )
            if any(
                not math.isclose(
                    exact_state.joint_positions_rad[joint_id],
                    diagnostic_qclose_joint_positions[joint_id],
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
                for joint_id in expected_fixture_joint_ids
            ):
                raise RuntimeError(
                    "Order8 exact q_close state disagrees with the paired Dock state"
                )
        if (
            diagnostic_qclose_checkpoint_state is not None
            and not diagnostic_qclose_zero_velocities
        ):
            exact_state = diagnostic_qclose_checkpoint_state
            initial_module_poses = dict(exact_state.module_root_poses)
            initial_module_frame_poses = {
                module_id: compose_pose(root_pose, root_to_module_frame)
                for module_id, root_pose in initial_module_poses.items()
            }
        else:
            # A static force-path replay deliberately reconstructs root poses
            # from the morphology graph and q_close joint positions.  Dynamic
            # checkpoint roots contain small compliant-constraint errors; if
            # those poses are restored with zero velocity, the fixed Dock
            # constraints release the stored error as an artificial impact.
            checkpoint_kinematics = WholeStructureKinematics().forward(
                morphology_graph,
                physical_model,
                fixture_joint_positions,
                initial_robot_base_pose,
                anchor_references,
            )
            initial_module_frame_poses = dict(
                checkpoint_kinematics.module_root_poses_world
            )
            initial_module_poses = {
                module_id: compose_pose(
                    initial_module_frame_poses[module_id],
                    module_frame_to_root,
                )
                for module_id in module_ids
            }
    else:
        initial_module_frame_poses = {
            module_id: compose_pose(
                initial_robot_base_pose,
                module_by_id[module_id].pose_in_design_frame,
            )
            for module_id in module_ids
        }
        initial_module_poses = {
            module_id: compose_pose(
                initial_module_frame_poses[module_id],
                module_frame_to_root,
            )
            for module_id in module_ids
        }
    lift_base_pose = _offset_pose(grasp_base_pose, dz=0.15)
    transport_base_pose = _offset_pose(
        lift_base_pose, dx=config.required_transport_distance_m
    )
    place_base_pose = _offset_pose(
        grasp_base_pose, dx=config.required_transport_distance_m
    )
    retreat_base_pose = _offset_pose(place_base_pose, dx=-0.10, dz=0.20)

    actuator_specs = physical_model.metadata.get("joint_actuator_specs", {})
    dock_spec = (
        actuator_specs.get("dock", {}) if isinstance(actuator_specs, dict) else {}
    )
    drive_spec = (
        dock_spec.get("simulation_drive", {}) if isinstance(dock_spec, dict) else {}
    )
    dock_continuous_torque_nm = float(dock_spec.get("continuous_torque_limit_nm", 1.3))
    dock_peak_torque_nm = float(dock_spec.get("peak_torque_nm", 4.1))
    dock_peak_current_a = float(dock_spec.get("peak_current_a", 7.3))
    dock_effort_limit = dock_peak_torque_nm
    if not (
        math.isfinite(dock_continuous_torque_nm)
        and math.isfinite(dock_peak_torque_nm)
        and math.isfinite(dock_peak_current_a)
        and 0.0 < dock_continuous_torque_nm <= dock_peak_torque_nm
        and dock_peak_current_a > 0.0
    ):
        raise RuntimeError("Order8 AK40-10 torque/current limits are invalid")
    configured_dock_velocity_limit = float(
        drive_spec.get("safe_velocity_limit_rad_s", 3.0)
    )
    (
        dock_armature_kg_m2,
        diagnostic_dock_armature_kg_m2,
        dock_armature_source,
    ) = _dock_joint_armature_setting(
        simulation_drive=drive_spec,
        diagnostic_override_kg_m2=getattr(
            args, "order8_diagnostic_dock_armature_kg_m2", None
        ),
        diagnostic_only=diagnostic_only,
    )
    diagnostic_dock_velocity_limit_raw = getattr(
        args, "order8_diagnostic_dock_velocity_limit_rad_s", None
    )
    if diagnostic_dock_velocity_limit_raw is not None:
        if not diagnostic_only:
            raise RuntimeError(
                "Order8 Dock velocity-limit override requires diagnostic mode"
            )
        diagnostic_dock_velocity_limit = float(diagnostic_dock_velocity_limit_raw)
        if (
            not math.isfinite(diagnostic_dock_velocity_limit)
            or diagnostic_dock_velocity_limit <= 0.0
            or diagnostic_dock_velocity_limit > configured_dock_velocity_limit + 1.0e-12
        ):
            raise RuntimeError(
                "Order8 diagnostic Dock velocity limit must be positive and no "
                "greater than the configured actuator limit"
            )
        dock_velocity_limit = diagnostic_dock_velocity_limit
    else:
        diagnostic_dock_velocity_limit = None
        dock_velocity_limit = configured_dock_velocity_limit
    contact_joint_velocity_limit = float(config.contact_joint_velocity_limit_radps)
    if (
        not math.isfinite(contact_joint_velocity_limit)
        or contact_joint_velocity_limit <= 0.0
        or contact_joint_velocity_limit > configured_dock_velocity_limit + 1.0e-12
    ):
        raise RuntimeError(
            "Order8 contact joint velocity limit must be positive and no "
            "greater than the configured AK40-10 simulation limit"
        )
    if diagnostic_dock_velocity_limit is not None:
        contact_joint_velocity_limit = min(
            contact_joint_velocity_limit,
            diagnostic_dock_velocity_limit,
        )
    diagnostic_contact_closure_joint_speed_raw = getattr(
        args,
        "order8_diagnostic_contact_closure_joint_speed_radps",
        None,
    )
    if diagnostic_contact_closure_joint_speed_raw is not None:
        if not diagnostic_only:
            raise RuntimeError(
                "Order8 closure-speed override requires diagnostic-only mode"
            )
        contact_closure_joint_speed_radps = float(
            diagnostic_contact_closure_joint_speed_raw
        )
        ordinary_closure_speed_radps = min(
            ORDER8_SIMPLE_CLOSURE_JOINT_SPEED_RADPS,
            contact_joint_velocity_limit,
        )
        if (
            not math.isfinite(contact_closure_joint_speed_radps)
            or contact_closure_joint_speed_radps <= 0.0
            or contact_closure_joint_speed_radps
            > ordinary_closure_speed_radps + 1.0e-12
        ):
            raise RuntimeError(
                "Order8 diagnostic closure speed must be positive and no "
                "greater than the ordinary closure speed"
            )
    else:
        contact_closure_joint_speed_radps = min(
            ORDER8_SIMPLE_CLOSURE_JOINT_SPEED_RADPS,
            contact_joint_velocity_limit,
        )
    release_joint_speed_radps = min(
        ORDER8_SIMPLE_RELEASE_JOINT_SPEED_RADPS,
        contact_joint_velocity_limit,
    )
    diagnostic_peak_torque_window_raw = getattr(
        args, "order8_diagnostic_peak_torque_window_s", None
    )
    if diagnostic_peak_torque_window_raw is not None:
        if not diagnostic_only:
            raise RuntimeError(
                "Order8 peak-torque window requires diagnostic-only mode"
            )
        diagnostic_peak_torque_window_s = float(diagnostic_peak_torque_window_raw)
        if (
            not math.isfinite(diagnostic_peak_torque_window_s)
            or diagnostic_peak_torque_window_s <= 0.0
        ):
            raise RuntimeError(
                "Order8 diagnostic peak-torque window must be finite and positive"
            )
    else:
        diagnostic_peak_torque_window_s = None
    diagnostic_post_grasp_joint_torque_bias_raw = getattr(
        args, "order8_diagnostic_post_grasp_joint_torque_bias_nm", None
    )
    if diagnostic_post_grasp_joint_torque_bias_raw is not None:
        if not diagnostic_only:
            raise RuntimeError(
                "Order8 post-grasp joint torque-bias override requires "
                "diagnostic-only mode"
            )
        diagnostic_post_grasp_joint_torque_bias_nm = float(
            diagnostic_post_grasp_joint_torque_bias_raw
        )
        if (
            not math.isfinite(diagnostic_post_grasp_joint_torque_bias_nm)
            or diagnostic_post_grasp_joint_torque_bias_nm <= 0.0
            or diagnostic_post_grasp_joint_torque_bias_nm
            > dock_continuous_torque_nm + 1.0e-12
        ):
            raise RuntimeError(
                "Order8 diagnostic post-grasp joint torque bias must be finite, "
                "positive, and no greater than the AK40-10 continuous rating"
            )
    else:
        diagnostic_post_grasp_joint_torque_bias_nm = None
    diagnostic_disable_slip_speed_safe_hold = bool(
        getattr(args, "order8_diagnostic_disable_slip_speed_safe_hold", False)
    )
    if diagnostic_disable_slip_speed_safe_hold and not diagnostic_only:
        raise RuntimeError(
            "Order8 slip-speed safe-hold disable requires diagnostic-only mode"
        )
    diagnostic_disable_all_safe_hold = bool(
        getattr(args, "order8_diagnostic_disable_all_safe_hold", False)
    )
    if diagnostic_disable_all_safe_hold and not diagnostic_only:
        raise RuntimeError(
            "Order8 all-safe-hold disable requires diagnostic-only mode"
        )

    sim_utils.create_new_stage()
    sim = SimulationContext(sim_utils.SimulationCfg(dt=float(args.dt), device=device))
    kit_viewer_active = _kit_visualizer_requested(args)
    if diagnostic_only and kit_viewer_active:
        # Keep the selected Dock mechanisms large enough in the viewport to
        # visually audit the live-physics closure.  The former distant scene
        # camera made slow joint motion unnecessarily difficult to judge.
        sim.set_camera_view(
            eye=[
                float(object_pose[0]) + 0.85,
                float(object_pose[1]) + 1.15,
                float(object_pose[2]) + 1.00,
            ],
            target=[
                float(object_pose[0]),
                float(object_pose[1]),
                float(object_pose[2]) + 0.10,
            ],
        )
    else:
        sim.set_camera_view(
            eye=[2.5, 2.5, 1.8], target=[object_pose[0], object_pose[1], 0.5]
        )
    floor_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=float(config.floor_friction),
        dynamic_friction=float(config.floor_friction),
        restitution=0.0,
    )
    ground_cfg = sim_utils.CuboidCfg(
        size=ORDER2_FLOOR_SIZE_M,
        # PhysX GPU contact views cannot use a bare static collider as a
        # filter.  A fixed kinematic rigid body preserves the immovable-floor
        # mechanics while making the explicit robot-environment safety view
        # supported and eliminating one warning per sensor body.
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            kinematic_enabled=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=not state_trace_replay_sync_physics,
        ),
        physics_material=floor_material,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.18, 0.18)),
    )
    ground_cfg.func(
        "/World/defaultGroundPlane",
        ground_cfg,
        translation=ORDER2_FLOOR_POSE_WORLD[:3],
    )
    # The object remains a free rigid body.  This fixed platform only raises
    # the work plane so articulated robot links retain useful floor clearance
    # during contact acquisition.  Its footprint covers both the initial and
    # planned place poses while leaving a small lateral overhang for the two
    # opposing Dock surfaces.
    object_support_size_m = (
        float(runtime_object_size_m[0])
        + float(config.required_transport_distance_m)
        + 0.05,
        max(0.05, float(runtime_object_size_m[1]) - 0.04),
        object_support_height_m,
    )
    object_support_pose_world: Pose7D = (
        float(object_pose[0]) + 0.5 * float(config.required_transport_distance_m),
        float(object_pose[1]),
        0.5 * object_support_height_m,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    support_cfg = sim_utils.CuboidCfg(
        size=object_support_size_m,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            kinematic_enabled=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=not state_trace_replay_sync_physics,
        ),
        physics_material=floor_material,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.24, 0.24, 0.28)
        ),
    )
    support_cfg.func(
        ORDER8_OBJECT_SUPPORT_PATH,
        support_cfg,
        translation=object_support_pose_world[:3],
    )
    if not diagnostic_only or state_trace_replay_path is not None or kit_viewer_active:
        sim_utils.DistantLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75),
        ).func(
            "/World/Light",
            sim_utils.DistantLightCfg(
                intensity=3000.0,
                color=(0.75, 0.75, 0.75),
            ),
        )
    if state_trace_replay_sync_physics and kit_viewer_active:
        # The low-load replay must remain visible even if the active RTX preset
        # has no usable environment illumination for the imported materials.
        # A uniform dome avoids a black normal-mesh viewport without touching
        # physics, collision, or the recorded trajectory.
        replay_dome_light_cfg = sim_utils.DomeLightCfg(
            intensity=1200.0,
            color=(0.92, 0.94, 1.0),
        )
        replay_dome_light_cfg.func(
            "/World/Order8/ReplayDomeLight",
            replay_dome_light_cfg,
        )

    roots = {module_id: f"/World/Order8/Module_{module_id}" for module_id in module_ids}
    robots: dict[int, Any] = {}
    movable_joint_ids = tuple(
        joint.joint_id for joint in physical_model.joints if joint.joint_type != "fixed"
    )
    solver_position_iteration_count = (
        8 if diagnostic_continue_after_force_ramp else (4 if diagnostic_only else 8)
    )
    solver_velocity_iteration_count = (
        8 if diagnostic_continue_after_force_ramp else (2 if diagnostic_only else 8)
    )
    for module_id in module_ids:
        pose = initial_module_poses[module_id]
        checkpoint_root_velocity = (
            diagnostic_qclose_checkpoint_state.module_root_velocities[module_id]
            if (
                diagnostic_qclose_checkpoint_state is not None
                and not diagnostic_qclose_zero_velocities
            )
            else (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        )
        initial_joint_position_map = (
            {
                joint_id: float(
                    fixture_joint_positions.get(
                        f"module_{module_id}:{joint_id}",
                        0.0,
                    )
                )
                for joint_id in movable_joint_ids
            }
            if diagnostic_qclose_fixture or diagnostic_near_contact_fixture
            else {".*": 0.0}
        )
        initial_joint_velocity_map = (
            {
                joint_id: float(
                    diagnostic_qclose_checkpoint_state.joint_velocities_radps.get(
                        f"module_{module_id}:{joint_id}",
                        0.0,
                    )
                )
                for joint_id in movable_joint_ids
            }
            if (
                diagnostic_qclose_checkpoint_state is not None
                and not diagnostic_qclose_zero_velocities
            )
            else {".*": 0.0}
        )
        cfg = ArticulationCfg(
            prim_path=roots[module_id],
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(usd_path),
                activate_contact_sensors=not state_trace_replay_sync_physics,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=state_trace_replay_sync_physics,
                    max_depenetration_velocity=2.0,
                    enable_gyroscopic_forces=True,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=(solver_position_iteration_count),
                    solver_velocity_iteration_count=(solver_velocity_iteration_count),
                    sleep_threshold=0.0,
                    stabilization_threshold=0.001,
                ),
                copy_from_source=False,
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=pose[:3],
                rot=pose[3:7],
                lin_vel=checkpoint_root_velocity[:3],
                ang_vel=checkpoint_root_velocity[3:],
                joint_pos=initial_joint_position_map,
                joint_vel=initial_joint_velocity_map,
            ),
            actuators={
                "gimbal_joints": ImplicitActuatorCfg(
                    joint_names_expr=[".*gimbal.*"],
                    stiffness=float(gimbal_stiffness),
                    damping=float(gimbal_damping),
                ),
                "dock_joints": ImplicitActuatorCfg(
                    joint_names_expr=[".*dock_mech.*"],
                    stiffness=float(dock_stiffness),
                    damping=float(dock_damping),
                    armature=dock_armature_kg_m2,
                    effort_limit_sim=dock_effort_limit,
                    velocity_limit_sim=dock_velocity_limit,
                ),
                "rotor_spinner_joints": ImplicitActuatorCfg(
                    joint_names_expr=[".*rotor.*"], stiffness=0.0, damping=0.0
                ),
            },
        )
        robots[module_id] = Articulation(cfg)

    # Keep the free object's floor interaction at the baseline material.  A
    # separate friction/compliance material is bound only to the two selected
    # authored Dock collision surfaces.  This models a uniform thin coating
    # without making the object artificially difficult to move on its support
    # during joint-only closure; actual lift margin remains an acceptance gate.
    selected_gripper_material_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=float(config.selected_gripper_friction),
        dynamic_friction=float(config.selected_gripper_friction),
        restitution=0.0,
        compliant_contact_stiffness=float(
            config.selected_gripper_compliant_contact_stiffness_n_per_m
        ),
        compliant_contact_damping=float(
            config.selected_gripper_compliant_contact_damping_n_s_per_m
        ),
        friction_combine_mode=ORDER8_SELECTED_GRIPPER_FRICTION_COMBINE_MODE,
    )
    selected_gripper_material_cfg.func(
        ORDER8_SELECTED_GRIPPER_MATERIAL_PATH,
        selected_gripper_material_cfg,
    )
    selected_gripper_material_body_path_by_key = {
        (int(surface.module_id), str(surface.mechanism_link_id)): (
            _resolve_rigid_body_path(
                sim.stage,
                roots[surface.module_id],
                surface.mechanism_link_id,
            )
        )
        for surface in selected_surfaces
    }
    selected_gripper_material_body_paths = sorted(
        set(selected_gripper_material_body_path_by_key.values())
    )
    if len(selected_gripper_material_body_paths) != len(selected_surfaces):
        raise RuntimeError(
            "Order8 selected gripper surfaces must resolve to distinct rigid bodies"
        )
    selected_gripper_material = UsdShade.Material(
        sim.stage.GetPrimAtPath(Sdf.Path(ORDER8_SELECTED_GRIPPER_MATERIAL_PATH))
    )
    if not selected_gripper_material.GetPrim().IsValid():
        raise RuntimeError("Order8 selected gripper physics material is invalid")
    selected_gripper_compliant_contact_stiffness = (
        selected_gripper_material.GetPrim()
        .GetAttribute("physxMaterial:compliantContactStiffness")
        .Get()
    )
    selected_gripper_compliant_contact_damping = (
        selected_gripper_material.GetPrim()
        .GetAttribute("physxMaterial:compliantContactDamping")
        .Get()
    )
    selected_gripper_compliant_contact_audit_passed = bool(
        selected_gripper_compliant_contact_stiffness is not None
        and selected_gripper_compliant_contact_damping is not None
        and math.isclose(
            float(selected_gripper_compliant_contact_stiffness),
            float(config.selected_gripper_compliant_contact_stiffness_n_per_m),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
        and math.isclose(
            float(selected_gripper_compliant_contact_damping),
            float(config.selected_gripper_compliant_contact_damping_n_s_per_m),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
    )
    if not selected_gripper_compliant_contact_audit_passed:
        raise RuntimeError(
            "Order8 selected authored Dock material did not retain the "
            "configured PhysX compliant-contact spring"
        )
    diagnostic_proxy_pad_initial_authored_collision_prims = [
        prim
        for body_path in selected_gripper_material_body_paths
        for prim in Usd.PrimRange(
            sim.stage.GetPrimAtPath(Sdf.Path(body_path)),
            Usd.TraverseInstanceProxies(),
        )
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    diagnostic_proxy_pad_deinstanced_prim_paths: list[str] = []
    if diagnostic_cone_proxy_pad_enabled:
        instance_root_paths: set[str] = set()
        for collision_prim in diagnostic_proxy_pad_initial_authored_collision_prims:
            current = collision_prim
            while current.IsValid() and current.IsInstanceProxy():
                current = current.GetParent()
            if current.IsValid() and current.IsInstance():
                instance_root_paths.add(current.GetPath().pathString)
        for instance_root_path in sorted(instance_root_paths):
            instance_root = sim.stage.GetPrimAtPath(Sdf.Path(instance_root_path))
            instance_root.SetInstanceable(False)
            if instance_root.IsInstance():
                raise RuntimeError(
                    "Order8 cone proxy failed to de-instance selected geometry: "
                    f"{instance_root_path}"
                )
            diagnostic_proxy_pad_deinstanced_prim_paths.append(instance_root_path)
    diagnostic_proxy_pad_authored_collision_paths = sorted(
        prim.GetPath().pathString
        for body_path in selected_gripper_material_body_paths
        for prim in Usd.PrimRange(
            sim.stage.GetPrimAtPath(Sdf.Path(body_path)),
            Usd.TraverseInstanceProxies(),
        )
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    )
    diagnostic_proxy_pad_disabled_authored_collision_paths: list[str] = []
    if diagnostic_cone_proxy_pad_enabled:
        for collision_path in diagnostic_proxy_pad_authored_collision_paths:
            collision_prim = sim.stage.GetPrimAtPath(Sdf.Path(collision_path))
            collision_api = UsdPhysics.CollisionAPI(collision_prim)
            collision_api.CreateCollisionEnabledAttr(False).Set(False)
            enabled = collision_api.GetCollisionEnabledAttr().Get()
            if enabled is not False:
                raise RuntimeError(
                    "Order8 cone proxy failed to disable selected authored collision: "
                    f"{collision_path}"
                )
            diagnostic_proxy_pad_disabled_authored_collision_paths.append(
                collision_path
            )
    diagnostic_proxy_pad_prim_paths: list[str] = []
    for proxy_spec in diagnostic_proxy_pad_specs:
        body_key = (int(proxy_spec.module_id), str(proxy_spec.link_id))
        body_path = selected_gripper_material_body_path_by_key.get(body_key)
        if body_path is None:
            raise RuntimeError(
                "Order8 proxy pad has no selected rigid-body path: "
                f"module_{proxy_spec.module_id}:{proxy_spec.link_id}"
            )
        proxy_id = getattr(proxy_spec, "pad_id", None)
        proxy_name = (
            ORDER8_DIAGNOSTIC_PROXY_PAD_PRIM_NAME
            if proxy_id is None
            else f"{ORDER8_DIAGNOSTIC_PROXY_PAD_PRIM_NAME}_{proxy_id}"
        )
        proxy_path = f"{body_path}/{proxy_name}"
        if sim.stage.GetPrimAtPath(Sdf.Path(proxy_path)).IsValid():
            raise RuntimeError(f"Order8 proxy-pad prim already exists: {proxy_path}")
        cube = UsdGeom.Cube.Define(sim.stage, Sdf.Path(proxy_path))
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.20, 0.05)])
        xformable = UsdGeom.Xformable(cube.GetPrim())
        xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(*proxy_spec.center_local)
        )
        qx, qy, qz, qw = proxy_spec.orientation_local_xyzw
        xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(float(qw), Gf.Vec3d(float(qx), float(qy), float(qz)))
        )
        xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(*proxy_spec.size_m)
        )
        proxy_collision_api = UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        proxy_collision_api.CreateCollisionEnabledAttr(True).Set(True)
        proxy_binding_api = UsdShade.MaterialBindingAPI.Apply(cube.GetPrim())
        proxy_binding_api.Bind(
            selected_gripper_material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )
        if (
            not cube.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
            or cube.GetPrim().HasAPI(UsdPhysics.RigidBodyAPI)
        ):
            raise RuntimeError(
                "Order8 proxy pad must be a collider on the existing selected "
                f"rigid body, not an independent rigid body: {proxy_path}"
            )
        diagnostic_proxy_pad_prim_paths.append(proxy_path)
    selected_gripper_material_collision_prim_paths: list[str] = []
    for body_path in selected_gripper_material_body_paths:
        body_prim = sim.stage.GetPrimAtPath(Sdf.Path(body_path))
        body_binding_api = (
            UsdShade.MaterialBindingAPI(body_prim)
            if body_prim.HasAPI(UsdShade.MaterialBindingAPI)
            else UsdShade.MaterialBindingAPI.Apply(body_prim)
        )
        body_binding_api.Bind(
            selected_gripper_material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )
        collision_paths = sorted(
            prim.GetPath().pathString
            for prim in Usd.PrimRange(body_prim, Usd.TraverseInstanceProxies())
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        )
        if not collision_paths:
            raise RuntimeError(
                "Order8 selected gripper body has no collision prim: " f"{body_path}"
            )
        selected_gripper_material_collision_prim_paths.extend(collision_paths)
    selected_gripper_material_binding_failures: list[str] = []
    for collision_path in selected_gripper_material_collision_prim_paths:
        collision_prim = sim.stage.GetPrimAtPath(Sdf.Path(collision_path))
        bound_material, _binding_relation = UsdShade.MaterialBindingAPI(
            collision_prim
        ).ComputeBoundMaterial(materialPurpose="physics")
        if (
            not bound_material.GetPrim().IsValid()
            or bound_material.GetPath().pathString
            != ORDER8_SELECTED_GRIPPER_MATERIAL_PATH
        ):
            selected_gripper_material_binding_failures.append(collision_path)
    if selected_gripper_material_binding_failures:
        raise RuntimeError(
            "Order8 selected gripper physics-material binding audit failed: "
            f"{selected_gripper_material_binding_failures}"
        )
    diagnostic_proxy_pad_missing_collision_paths = sorted(
        set(diagnostic_proxy_pad_prim_paths)
        - set(selected_gripper_material_collision_prim_paths)
    )
    if diagnostic_proxy_pad_missing_collision_paths:
        raise RuntimeError(
            "Order8 proxy pads are absent from selected-body collision traversal: "
            f"{diagnostic_proxy_pad_missing_collision_paths}"
        )
    diagnostic_proxy_pad_exclusive_under_penetration_limit = bool(
        diagnostic_cone_proxy_pad_enabled
        and diagnostic_proxy_pad_specs
        and diagnostic_proxy_pad_authored_collision_paths
        and set(diagnostic_proxy_pad_disabled_authored_collision_paths)
        == set(diagnostic_proxy_pad_authored_collision_paths)
    ) or bool(
        diagnostic_legacy_proxy_pad_enabled
        and diagnostic_proxy_pad_specs
        and all(
            float(spec.outer_face_projection_m)
            - float(spec.mesh_surface_projection_m)
            > float(config.max_penetration_m) + 1.0e-12
            for spec in diagnostic_proxy_pad_specs
            if isinstance(spec, _Order8DiagnosticProxyPadSpec)
        )
    )
    if (
        diagnostic_proxy_pad_enabled
        and not diagnostic_proxy_pad_exclusive_under_penetration_limit
    ):
        raise RuntimeError(
            "Order8 proxy-pad contact representation is not exclusive from "
            "the selected authored collision mesh"
        )

    object_cfg = RigidObjectCfg(
        prim_path="/World/Order8/Object",
        spawn=sim_utils.CuboidCfg(
            size=runtime_object_size_m,
            activate_contact_sensors=not state_trace_replay_sync_physics,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=state_trace_replay_sync_physics,
                max_depenetration_velocity=2.0,
                enable_gyroscopic_forces=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=float(config.object_mass_kg)),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=not state_trace_replay_sync_physics,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=float(config.object_friction),
                dynamic_friction=float(config.object_friction),
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.55, 0.85)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=object_pose[:3],
            rot=object_pose[3:7],
            lin_vel=(
                diagnostic_qclose_checkpoint_state.object_twist[:3]
                if (
                    diagnostic_qclose_checkpoint_state is not None
                    and not diagnostic_qclose_zero_velocities
                )
                else (0.0, 0.0, 0.0)
            ),
            ang_vel=(
                diagnostic_qclose_checkpoint_state.object_twist[3:]
                if (
                    diagnostic_qclose_checkpoint_state is not None
                    and not diagnostic_qclose_zero_velocities
                )
                else (0.0, 0.0, 0.0)
            ),
        ),
    )
    object_asset = RigidObject(object_cfg)
    # Any post-spawn object pose write must be routed through this explicit
    # counter.  Acceptance keeps it at zero; the exact-state diagnostic may
    # perform one measured checkpoint restore after reset.
    object_root_pose_write_count = 0

    physical_ports = {port.port_id: port for port in physical_model.dock_ports}
    graph_ports = {port.port_global_id: port for port in morphology_graph.ports}
    constraints: list[tuple[Any, Any]] = []
    for edge in morphology_graph.dock_edges:
        src_port = graph_ports[edge.src_port_id]
        dst_port = graph_ports[edge.dst_port_id]
        src_body_path = _resolve_rigid_body_path(
            sim.stage,
            roots[edge.src_module_id],
            physical_ports[src_port.port_local_id].parent_link,
        )
        dst_body_path = _resolve_rigid_body_path(
            sim.stage,
            roots[edge.dst_module_id],
            physical_ports[dst_port.port_local_id].parent_link,
        )
        spec = build_dynamic_dock_constraint_spec(
            morphology_graph,
            physical_model,
            edge_id=edge.edge_id,
            leader_module_id=edge.src_module_id,
            follower_module_id=edge.dst_module_id,
            leader_body_path=src_body_path,
            follower_body_path=dst_body_path,
            constraint_root_path="/World/Order8/AssemblyConstraints",
        )
        joint = preauthor_disabled_fixed_joint(sim.stage, spec)
        joint.GetJointEnabledAttr().Set(not state_trace_replay_sync_physics)
        # Occupied docking surfaces are already rigidly constrained.  Filtering
        # only this exact body pair avoids self-collision without touching either
        # selected free gripper/object pair.
        leader_prim = sim.stage.GetPrimAtPath(Sdf.Path(spec.leader_body_path))
        UsdPhysics.FilteredPairsAPI.Apply(
            leader_prim
        ).CreateFilteredPairsRel().AddTarget(Sdf.Path(spec.follower_body_path))
        constraints.append((spec, joint))

    diagnostic_world_fixed_joint = None
    diagnostic_world_fixed_body_path: str | None = None
    diagnostic_world_fixed_object_joint = None
    if diagnostic_force_fixture or diagnostic_kinematic_base_isolation:
        diagnostic_world_fixed_body_path = _resolve_rigid_body_path(
            sim.stage,
            roots[morphology_graph.base_module_id],
            module_frame_link_id,
        )
        diagnostic_world_fixed_joint = _preauthor_disabled_world_fixed_body(
            sim.stage,
            prim_path="/World/Order8/DiagnosticConstraints/BaseFrameToWorld",
            body_path=diagnostic_world_fixed_body_path,
            body_pose_world=initial_module_frame_poses[morphology_graph.base_module_id],
        )
    if diagnostic_force_fixture or diagnostic_world_fixed_object_requested:
        diagnostic_world_fixed_object_joint = _preauthor_disabled_world_fixed_body(
            sim.stage,
            prim_path="/World/Order8/DiagnosticConstraints/ObjectToWorld",
            body_path="/World/Order8/Object",
            body_pose_world=object_pose,
        )

    # Isaac Lab's generic contact-sensor activation stops walking a subtree at
    # its first rigid body.  Imported URDF articulations contain nested rigid
    # bodies, so explicitly author the PhysX reporting API on every body before
    # the tensor views are initialized.  Without this, the contact view silently
    # drops most sensor paths and its pair layout no longer matches our stable
    # global-link identity table.
    if state_trace_replay_sync_physics:
        contact_report_body_counts = {module_id: 0 for module_id in module_ids}
        object_contact_report_body_count = 0
    else:
        contact_report_body_counts = {
            module_id: _activate_nested_contact_reports(
                sim.stage,
                root_prim_path=roots[module_id],
            )
            for module_id in module_ids
        }
        object_contact_report_body_count = _activate_nested_contact_reports(
            sim.stage,
            root_prim_path="/World/Order8/Object",
        )

    sim.reset()
    sim_dt = float(sim.get_physics_dt())
    if diagnostic_qclose_fixture or diagnostic_near_contact_fixture:
        # SimulationContext.reset() advances the newly spawned constrained
        # articulations once.  Re-apply the mutually consistent graph-FK root
        # poses and measured joint state before the first diagnostic
        # observation so that reset warm-up cannot corrupt the checkpoint.
        for module_id, robot in robots.items():
            root_pose_tensor = torch.tensor(
                [initial_module_poses[module_id]],
                dtype=torch.float32,
                device=sim.device,
            )
            restored_root_velocity = torch.tensor(
                [
                    (
                        diagnostic_qclose_checkpoint_state.module_root_velocities[
                            module_id
                        ]
                        if (
                            diagnostic_qclose_checkpoint_state is not None
                            and not diagnostic_qclose_zero_velocities
                        )
                        else (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                    )
                ],
                dtype=torch.float32,
                device=sim.device,
            )
            joint_positions = torch.tensor(
                [
                    [
                        float(
                            fixture_joint_positions.get(
                                f"module_{module_id}:{joint_name}",
                                0.0,
                            )
                        )
                        for joint_name in robot.joint_names
                    ]
                ],
                dtype=torch.float32,
                device=sim.device,
            )
            restored_joint_velocities = torch.tensor(
                [
                    [
                        (
                            float(
                                diagnostic_qclose_checkpoint_state.joint_velocities_radps.get(
                                    f"module_{module_id}:{joint_name}",
                                    0.0,
                                )
                            )
                            if (
                                diagnostic_qclose_checkpoint_state is not None
                                and not diagnostic_qclose_zero_velocities
                            )
                            else 0.0
                        )
                        for joint_name in robot.joint_names
                    ]
                ],
                dtype=torch.float32,
                device=sim.device,
            )
            robot.write_root_pose_to_sim_index(root_pose=root_pose_tensor)
            robot.write_root_velocity_to_sim_index(root_velocity=restored_root_velocity)
            robot.write_joint_position_to_sim_index(position=joint_positions)
            robot.write_joint_velocity_to_sim_index(velocity=restored_joint_velocities)
        if diagnostic_near_contact_fixture:
            object_pose_tensor = torch.tensor(
                [diagnostic_near_contact_object_pose],
                dtype=torch.float32,
                device=sim.device,
            )
            object_asset.write_root_pose_to_sim(object_pose_tensor)
            object_asset.write_root_velocity_to_sim(
                torch.zeros((1, 6), dtype=torch.float32, device=sim.device)
            )
            object_root_pose_write_count += 1
        elif diagnostic_qclose_checkpoint_state is not None:
            object_pose_tensor = torch.tensor(
                [diagnostic_qclose_checkpoint_state.object_pose],
                dtype=torch.float32,
                device=sim.device,
            )
            object_twist_tensor = torch.tensor(
                [
                    (
                        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                        if diagnostic_qclose_zero_velocities
                        else diagnostic_qclose_checkpoint_state.object_twist
                    )
                ],
                dtype=torch.float32,
                device=sim.device,
            )
            object_asset.write_root_pose_to_sim(object_pose_tensor)
            object_asset.write_root_velocity_to_sim(object_twist_tensor)
            object_root_pose_write_count += 1
    for robot in robots.values():
        robot.update(sim_dt)
    object_asset.update(sim_dt)
    if state_trace_replay_path is not None:
        from amsrr.simulation.order8_state_trace import load_order8_state_trace

        state_trace = load_order8_state_trace(state_trace_replay_path)
        _validate_order8_state_trace_runtime_binding(
            state_trace,
            graph_id=morphology_graph.graph_id,
            graph_hash=morphology_graph.stable_hash(),
            config_hash=config.stable_hash(),
            source_urdf_sha256=hash_file(urdf_path),
            generated_usd_sha256=hash_file(usd_path),
            module_ids=module_ids,
            robots=robots,
        )
        replay_metrics = _replay_order8_state_trace(
            state_trace,
            robots=robots,
            object_asset=object_asset,
            sim=sim,
            torch=torch,
            wp=wp,
            speed=state_trace_replay_speed,
            loops=state_trace_replay_loops,
            endpoint_hold_s=state_trace_replay_endpoint_hold_s,
            sync_physics=state_trace_replay_sync_physics,
        )
        return {
            "spawn_passed": True,
            "isaac_backed": True,
            "order8_state_trace_replay": True,
            "order8_state_trace_replay_passed": True,
            "order8_state_trace_replay_version": state_trace["schema_version"],
            "order8_state_trace_path": str(state_trace_replay_path),
            "order8_state_trace_hash": state_trace["trace_payload_hash"],
            "order8_state_trace_diagnostic_only": True,
            "order8_state_trace_acceptance_eligible": False,
            "order8_state_trace_replay_advances_physics": (
                state_trace_replay_sync_physics
            ),
            "order8_state_trace_replay_sync_physics": (
                state_trace_replay_sync_physics
            ),
            "order8_state_trace_replay_sync_method": (
                "gravity_free_contact_minimized_exact_state_reapply_v1"
                if state_trace_replay_sync_physics
                else "kinematic_forward_only_v1"
            ),
            "order8_state_trace_replay_collisions_enabled": (
                True
            ),
            "order8_state_trace_replay_object_collisions_enabled": (
                not state_trace_replay_sync_physics
            ),
            "order8_state_trace_replay_floor_support_collisions_enabled": (
                not state_trace_replay_sync_physics
            ),
            "order8_state_trace_replay_robot_self_collisions_enabled": False,
            "order8_state_trace_replay_authored_cross_module_collisions_enabled": (
                True
            ),
            "order8_state_trace_replay_cone_proxy_visual_enabled": (
                diagnostic_cone_proxy_pad_enabled
            ),
            "order8_state_trace_replay_cone_proxy_prim_count": (
                len(diagnostic_proxy_pad_prim_paths)
                if diagnostic_cone_proxy_pad_enabled
                else 0
            ),
            "order8_state_trace_replay_cone_proxy_object_collision_enabled": False,
            "order8_state_trace_replay_gravity_enabled": (
                not state_trace_replay_sync_physics
            ),
            "order8_state_trace_replay_graph_constraints_enabled": (
                not state_trace_replay_sync_physics
            ),
            "order8_state_trace_replay_speed": state_trace_replay_speed,
            "order8_state_trace_replay_loops": state_trace_replay_loops,
            "order8_state_trace_replay_endpoint_hold_s": (
                state_trace_replay_endpoint_hold_s
            ),
            **replay_metrics,
        }
    diagnostic_world_fixed_pose: Pose7D | None = None
    if diagnostic_world_fixed_joint is not None:
        diagnostic_world_fixed_pose, _ = _module_frame_pose_twist(
            robots[morphology_graph.base_module_id],
            module_frame_link_id=module_frame_link_id,
        )
        _enable_world_fixed_body_at_pose(
            diagnostic_world_fixed_joint,
            diagnostic_world_fixed_pose,
        )
    diagnostic_world_fixed_object_pose: Pose7D | None = None
    if diagnostic_world_fixed_object_joint is not None:
        diagnostic_world_fixed_object_pose = tuple(
            float(value) for value in _object_state(object_asset)["pose"]
        )
        _enable_world_fixed_body_at_pose(
            diagnostic_world_fixed_object_joint,
            diagnostic_world_fixed_object_pose,
        )

    rigid_body_paths: list[str] = []
    body_identity: list[str] = []
    body_lookup: list[tuple[int, str]] = []
    for module_id in module_ids:
        for path in _rigid_body_paths(sim.stage, roots[module_id]):
            local_name = _canonical_rigid_body_local_name(
                path,
                physical_model,
            )
            rigid_body_paths.append(path)
            body_identity.append(f"module_{module_id}:{local_name}")
            body_lookup.append((module_id, local_name))
    all_robot_rigid_body_paths = list(rigid_body_paths)
    selected_link_ids = [
        f"module_{surface.module_id}:{surface.mechanism_link_id}"
        for surface in selected_surfaces
    ]
    if diagnostic_only:
        selected_rows = [
            (path, identity, lookup)
            for path, identity, lookup in zip(
                rigid_body_paths,
                body_identity,
                body_lookup,
                strict=True,
            )
            if identity in selected_link_ids
        ]
        if len(selected_rows) != len(selected_link_ids):
            raise RuntimeError(
                "Order8 diagnostic could not isolate both selected Dock bodies"
            )
        rigid_body_paths = [row[0] for row in selected_rows]
        body_identity = [row[1] for row in selected_rows]
        body_lookup = [row[2] for row in selected_rows]
    object_path = "/World/Order8/Object"
    robot_object_contact_capacity = max(
        1024
        if diagnostic_cone_proxy_pad_enabled
        else 256
        if diagnostic_only
        else 64,
        len(rigid_body_paths) * 16,
    )
    contact_view = sim.physics_manager.get_physics_sim_view().create_rigid_contact_view(
        rigid_body_paths,
        filter_patterns=[[object_path] for _ in rigid_body_paths],
        max_contact_data_count=robot_object_contact_capacity,
    )
    _require_contact_view_layout(
        contact_view,
        label="robot_object",
        expected_sensor_count=len(rigid_body_paths),
        expected_filter_count=1,
    )
    object_floor_view = (
        sim.physics_manager.get_physics_sim_view().create_rigid_contact_view(
            [object_path],
            filter_patterns=[
                ["/World/defaultGroundPlane", ORDER8_OBJECT_SUPPORT_PATH]
            ],
            max_contact_data_count=32,
        )
    )
    _require_contact_view_layout(
        object_floor_view,
        label="object_support_environment",
        expected_sensor_count=1,
        expected_filter_count=2,
    )
    robot_environment_contact_view = (
        sim.physics_manager.get_physics_sim_view().create_rigid_contact_view(
            all_robot_rigid_body_paths,
            filter_patterns=[
                ["/World/defaultGroundPlane", ORDER8_OBJECT_SUPPORT_PATH]
                for _ in all_robot_rigid_body_paths
            ],
            max_contact_data_count=max(
                256,
                8 * len(all_robot_rigid_body_paths),
            ),
        )
    )
    _require_contact_view_layout(
        robot_environment_contact_view,
        label="robot_support_environment",
        expected_sensor_count=len(all_robot_rigid_body_paths),
        expected_filter_count=2,
    )

    qpid_config = QPIDControllerConfig(
        allocation_mode=str(args.allocation_mode),
        control_dt_s=sim_dt,
    )
    contact_centering_qpid_config = replace(
        qpid_config,
        xy_p_gain=float(config.contact_centering_xy_p_gain),
        xy_d_gain=float(config.contact_centering_xy_d_gain),
        roll_pitch_p_gain=float(config.contact_centering_roll_pitch_p_gain),
        roll_pitch_d_gain=float(config.contact_centering_roll_pitch_d_gain),
    )
    controller = QPIDController(config=qpid_config)
    contact_centering_controller = QPIDController(config=contact_centering_qpid_config)
    centroidal_target_builder = RigidBodyControlModelBuilder()
    external_wrench_estimator_config = CentroidalExternalWrenchEstimatorConfig(
        gravity_mps2=float(qpid_config.gravity_mps2),
        wrench_filter_time_constant_s=float(
            config.contact_external_wrench_filter_time_constant_s
        ),
        bias_filter_time_constant_s=float(
            config.contact_external_wrench_bias_time_constant_s
        ),
    )
    external_wrench_estimator = CentroidalExternalWrenchEstimator(
        external_wrench_estimator_config
    )
    contact_admittance_config = CentroidalAdmittanceConfig(
        force_deadband_n=float(config.contact_admittance_force_deadband_n),
        torque_deadband_nm=float(config.contact_admittance_torque_deadband_nm),
        linear_admittance_mps_per_n=float(
            config.contact_admittance_linear_gain_mps_per_n
        ),
        angular_admittance_radps_per_nm=float(
            config.contact_admittance_angular_gain_radps_per_nm
        ),
        maximum_linear_speed_mps=float(
            config.contact_admittance_max_linear_speed_mps
        ),
        maximum_angular_speed_radps=float(
            config.contact_admittance_max_angular_speed_radps
        ),
        maximum_translation_offset_m=float(
            config.contact_admittance_max_translation_offset_m
        ),
    )
    contact_admittance_controller = CentroidalAdmittanceController(
        contact_admittance_config
    )
    bridge = IsaacControllerBridge()
    component_mappings = {
        module_id: build_actuator_mapping(singleton_graphs[module_id], physical_model)
        for module_id in module_ids
    }
    last_status = {
        module_id: ControllerStatus(status="ok", qp_feasible=True)
        for module_id in module_ids
    }
    previous_controller_command = None

    joint_model = {joint.joint_id: joint for joint in physical_model.joints}
    expected_joint_ids = ordered_global_dock_joint_ids(
        morphology_graph,
        physical_model,
    )
    applied_dock_armature_kg_m2_by_joint = {
        joint_id: _global_joint_tensor_value(
            robots,
            joint_id,
            field_name="joint_armature",
        )
        for joint_id in expected_joint_ids
    }
    if any(
        not math.isclose(
            applied,
            dock_armature_kg_m2,
            rel_tol=1.0e-6,
            abs_tol=1.0e-9,
        )
        for applied in applied_dock_armature_kg_m2_by_joint.values()
    ):
        raise RuntimeError(
            "Isaac Dock joint armature does not match the requested setting"
        )
    joint_limits = tuple(
        _dock_limit(
            joint_model[global_id.split(":", 1)[1]],
            dock_spec,
            velocity_limit_override_radps=contact_joint_velocity_limit,
        )
        for global_id in expected_joint_ids
    )
    joint_controller_config = NaturalContactJointControllerConfig(
        control_dt_s=sim_dt,
        max_position_command_lead_rad=position_drive_peak_effort_lead_rad(
            stiffness_nm_per_rad=float(dock_stiffness),
            peak_effort_nm=float(dock_effort_limit),
        ),
        reachability_absolute_tolerance=float(
            config.simultaneous_reachability_absolute_tolerance
        ),
    )
    low_level = NaturalContactJointController(joint_controller_config)
    diagnostic_anchor_hold_low_level = NaturalContactJointController(
        replace(
            joint_controller_config,
            # The outer loop must preserve the measured q_close shape in the
            # Jacobian null space.  Pulling unused columns toward URDF neutral
            # would be a second posture policy and would confound this causal
            # diagnostic.
            task_error_gain_per_s=(
                ORDER8_DIAGNOSTIC_ANCHOR_HOLD_TASK_GAIN_PER_S
            ),
            neutral_posture_gain_per_s=0.0,
            nullspace_velocity_damping=0.0,
        )
    )
    full_actuator_mapping = build_actuator_mapping(
        morphology_graph,
        physical_model,
    )

    selections: list[NaturalContactAnchorSelection] = []
    candidates: list[ContactCandidate] = []
    for candidate_id, surface in enumerate(selected_surfaces):
        anchor = anchor_by_module.get(surface.module_id)
        if anchor is None:
            raise RuntimeError("selected gripper surface has no matching grasp anchor")
        slot_id = int(anchor.associated_contact_slot_ids[0])
        normal = (
            selected_pair.first_inward_axis_design
            if surface is selected_pair.first
            else selected_pair.second_inward_axis_design
        )
        selections.append(
            NaturalContactAnchorSelection(
                anchor_id=anchor.anchor_id,
                slot_id=slot_id,
                candidate_id=candidate_id,
                dock_link_id=f"module_{surface.module_id}:{surface.mechanism_link_id}",
                inward_normal_world=tuple(float(value) for value in normal),
            )
        )
        candidates.append(
            ContactCandidate(
                candidate_id=candidate_id,
                slot_id=slot_id,
                anchor_id=anchor.anchor_id,
                target_entity_id="order8_object",
                region_id=f"order8_object_face:{candidate_id}",
                contact_pose_world=object_pose,
                contact_frame_world=object_pose,
                # ``normal`` is the robot-on-object inward command direction;
                # ContactCandidate stores the target surface's outward normal.
                normal_world=tuple(-float(value) for value in normal),
                tangent_basis_world=_tangent_basis(normal),
                contact_mode=ContactMode.GRASP,
                friction=combine_friction(
                    float(config.object_friction),
                    float(config.selected_gripper_friction),
                    ORDER8_SELECTED_GRIPPER_FRICTION_COMBINE_MODE,
                ),
                patch_area_m2=0.01,
                candidate_scores={
                    "deterministic_order8": 1.0,
                    "material_contract_applied": 1.0,
                    "material_target_surface_friction": float(
                        config.object_friction
                    ),
                    "material_robot_surface_friction": float(
                        config.selected_gripper_friction
                    ),
                    "material_effective_friction": combine_friction(
                        float(config.object_friction),
                        float(config.selected_gripper_friction),
                        ORDER8_SELECTED_GRIPPER_FRICTION_COMBINE_MODE,
                    ),
                    "material_friction_combine_mode_code": 3.0,
                },
                unary_valid=True,
            )
        )
    selected_anchor_ids = tuple(
        sorted(int(selection.anchor_id) for selection in selections)
    )
    contact_admittance_axis_world = _unit(
        (
            float(selections[0].inward_normal_world[0]),
            float(selections[0].inward_normal_world[1]),
            0.0,
        )
    )
    selected_contact_joint_id_by_anchor = {
        int(anchor_by_module[surface.module_id].anchor_id): (
            f"module_{surface.module_id}:{surface.mechanism_joint_id}"
        )
        for surface in selected_surfaces
    }
    if set(selected_contact_joint_id_by_anchor) != set(selected_anchor_ids):
        raise RuntimeError(
            "Order8 selected contact-joint map must cover every grasp anchor"
        )
    dock_rated_torque_nm = dock_continuous_torque_nm
    contact_stall_selected_joint_load_threshold_nm = (
        ORDER8_CONTACT_STALL_RATED_TORQUE_FRACTION * dock_rated_torque_nm
    )
    if not (
        math.isfinite(contact_stall_selected_joint_load_threshold_nm)
        and 0.0 < contact_stall_selected_joint_load_threshold_nm < dock_rated_torque_nm
    ):
        raise RuntimeError(
            "Order8 selected Dock contact-load threshold must be inside the "
            "AK40-10 rated-torque envelope"
        )
    contact_position_preload_load_threshold_nm = float(
        config.contact_position_preload_load_threshold_nm
    )
    if not (
        contact_stall_selected_joint_load_threshold_nm
        < contact_position_preload_load_threshold_nm
        <= dock_rated_torque_nm
    ):
        raise RuntimeError(
            "Order8 position-preload load threshold must be above the q_close "
            "trigger and no greater than the AK40-10 continuous rating"
        )
    if diagnostic_force_anchor_ids is not None and not set(
        diagnostic_force_anchor_ids
    ).issubset(selected_anchor_ids):
        raise RuntimeError(
            "Order8 diagnostic force-anchor ids must select only active "
            "natural-contact anchors"
        )
    force_ramp_anchor_ids = (
        selected_anchor_ids
        if diagnostic_force_anchor_ids is None
        else tuple(sorted(diagnostic_force_anchor_ids))
    )
    selected_gripper_local_aabbs_by_anchor = {}
    for selection in selections:
        module_text, link_id = selection.dock_link_id.split(":", 1)
        module_id = int(module_text[len("module_") :])
        matching_bounds = tuple(
            bounds
            for bounds in selected_gripper_contact_local_surfaces
            if bounds.module_id == module_id and bounds.link_id == link_id
        )
        if not matching_bounds:
            raise RuntimeError(
                "Order8 selected anchor has no matching active collision-surface "
                f"bounds: {selection.anchor_id}"
            )
        selected_gripper_local_aabbs_by_anchor[int(selection.anchor_id)] = (
            matching_bounds
        )
    order9_teacher_output_raw = getattr(args, "order9_teacher_output", None)
    order9_teacher_output_path = (
        None
        if order9_teacher_output_raw is None
        else Path(str(order9_teacher_output_raw)).resolve()
    )
    if order9_teacher_output_path is not None and diagnostic_only:
        raise RuntimeError("Order9 teacher capture rejects Order8 diagnostic-only runs")
    order9_teacher_task_id = str(
        getattr(args, "order9_teacher_task_id", None)
        or (
            f"order9-c0-task-{int(args.order8_seed):06d}"
            if order9_teacher_output_path is not None
            else "order8-natural-contact-smoke"
        )
    )
    candidate_set = ContactCandidateSet(
        set_id=f"{order9_teacher_task_id}:selected-pair",
        task_id=order9_teacher_task_id,
        morphology_graph_id=morphology_graph.graph_id,
        candidates=candidates,
        candidate_mask=[True] * len(candidates),
        slot_coverage={
            slot_id: [
                selection.candidate_id
                for selection in selections
                if selection.slot_id == slot_id
            ]
            for slot_id in sorted({selection.slot_id for selection in selections})
        },
        pairwise_conflict_matrix=[[False] * len(candidates) for _ in candidates],
        pairwise_compatibility_score=[[1.0] * len(candidates) for _ in candidates],
        group_proposals=[],
        assignment_feasibility_cache={},
        sampler_version="order8_mesh_backed_selected_pair_v2_material_combine",
    )
    order8_task_spec = build_order8_grasp_carry_task_spec(
        object_pose_world=object_pose,
        object_size_m=tuple(float(value) for value in config.object_size_m),
        object_mass_kg=float(config.object_mass_kg),
        object_friction=float(config.object_friction),
        required_transport_distance_m=float(config.required_transport_distance_m),
        support_height_m=float(config.object_support_height_m),
        max_contact_force_n=float(config.max_force_per_contact_n),
        max_contact_torque_nm=float(config.max_torque_per_contact_nm),
        task_id=order9_teacher_task_id,
        selected_gripper_friction=float(config.selected_gripper_friction),
        friction_combine_mode=ORDER8_SELECTED_GRIPPER_FRICTION_COMBINE_MODE,
    )
    policy_context = compile_high_level_context(
        order8_task_spec,
        morphology_graph,
        candidate_set,
    )
    max_steps = max(1, int(args.steps))
    requested_simulation_duration_s = float(max_steps) * sim_dt
    planner_timeout_s = (
        requested_simulation_duration_s + 2.0 * sim_dt
        if diagnostic_disable_all_safe_hold
        else None
    )
    planner_config = NaturalContactPlannerConfig(
        phase_timeout_s=(
            float(planner_timeout_s)
            if planner_timeout_s is not None
            else NaturalContactPlannerConfig.phase_timeout_s
        ),
        contact_acquisition_timeout_s=(
            float(planner_timeout_s)
            if planner_timeout_s is not None
            else float(config.contact_acquisition_timeout_s)
        ),
        normal_force_target_per_contact_n=float(
            config.normal_force_target_per_contact_n
        ),
    )
    planner = DeterministicNaturalContactPlanner(
        selections,
        config=planner_config,
    )
    order9_teacher_collector = None
    if order9_teacher_output_path is not None:
        from amsrr.schemas.datasets import DatasetSplit
        from amsrr.training.order9_teacher_collection import (
            Order9TeacherCollectionConfig,
            Order9TeacherEpisodeCollector,
        )

        order9_teacher_collector = Order9TeacherEpisodeCollector(
            task_spec=order8_task_spec,
            morphology_graph=morphology_graph,
            contact_candidate_set=candidate_set,
            config=Order9TeacherCollectionConfig(
                episode_id=str(
                    getattr(args, "order9_teacher_episode_id", None)
                    or f"order9-c0-episode-{int(args.order8_seed):06d}"
                ),
                split=DatasetSplit(
                    str(getattr(args, "order9_teacher_split", "train"))
                ),
                low_level_stride=int(
                    getattr(args, "order9_teacher_low_level_stride", 1)
                ),
                high_level_stride=int(
                    getattr(args, "order9_teacher_high_level_stride", 5)
                ),
                window_horizon_s=float(
                    getattr(args, "order9_teacher_window_horizon_s", 2.0)
                ),
                window_knot_dt_s=float(
                    getattr(args, "order9_teacher_window_knot_dt_s", 0.1)
                ),
            ),
        )
    suppressed_safe_hold_reason_counts: dict[str, int] = {}
    suppressed_safe_hold_first_time_s_by_reason: dict[str, float] = {}

    def request_safe_hold_or_record(*, time_s: float, reason: str) -> None:
        if diagnostic_disable_all_safe_hold:
            suppressed_safe_hold_reason_counts[reason] = (
                suppressed_safe_hold_reason_counts.get(reason, 0) + 1
            )
            suppressed_safe_hold_first_time_s_by_reason.setdefault(
                reason, float(time_s)
            )
            return
        planner.request_safe_hold(time_s=time_s, reason=reason)

    monitor = NaturalContactEvidenceMonitor(config)
    whole_structure_kinematics = WholeStructureKinematics()

    phase_trace = [Order8NaturalContactPhase.RESET.value]
    state_trace_frames: list[dict[str, object]] = []
    state_trace_step_counter = 0
    step_evidence: list[dict[str, object]] = []
    planner_transitions: list[dict[str, object]] = []
    current_time_s = 0.0
    if state_trace_output_path is not None:
        state_trace_frames.append(
            _capture_order8_state_trace_frame(
                simulation_time_s=current_time_s,
                phase=Order8NaturalContactPhase.RESET.value,
                robots=robots,
                object_asset=object_asset,
            )
        )
    command_index = 0
    qp_infeasible_count = 0
    controller_failure_count = 0
    missing_count = 0
    unsupported_count = 0
    clipped_count = 0
    unresolved_count = 0
    observed_joint_ids: set[str] = set()
    position_commanded_ids: set[str] = set()
    velocity_commanded_ids: set[str] = set()
    torque_commanded_ids: set[str] = set()
    raw_invalid_count = 0
    raw_saturation_count = 0
    raw_contact_failure_reasons: list[str] = []
    robot_environment_contact_step_count = 0
    robot_environment_unsafe_contact_step_count = 0
    robot_environment_first_unsafe_contact_time_s: float | None = None
    post_release_selected_contact_count = 0
    payload_feedforward_active_count = 0
    payload_feedforward_peak_scale = 0.0
    payload_feedforward_max_scale_step = 0.0
    last_payload_feedforward_scale = 0.0
    last_payload_feedforward_target_scale = 0.0
    last_payload_coupling: dict[str, object] | None = None
    last_full_payload_coupling: dict[str, object] | None = None
    measured_payload_lift_transfer_peak_scale = 0.0
    estimated_payload_lift_transfer_peak_scale = 0.0
    last_estimated_payload_lift_transfer_scale = 0.0
    lift_start_external_force_world_z_n: float | None = None
    last_lift_external_force_world_z_n: float | None = None
    last_estimated_payload_transferred_load_n = 0.0
    payload_load_observer_valid_step_count = 0
    payload_load_observer_invalid_step_count = 0
    payload_lift_off_confirmed_time_s: float | None = None
    diagnostic_loaded_state_rebase_triggered_time_s: float | None = None
    diagnostic_loaded_state_rebase_completed_time_s: float | None = None
    diagnostic_loaded_state_rebase_hold_base_pose: Pose7D | None = None
    diagnostic_loaded_state_rebase_centroidal_pose: Pose7D | None = None
    diagnostic_loaded_state_rebase_joint_targets_rad: dict[str, float] = {}
    diagnostic_loaded_state_rebase_active_step_count = 0
    diagnostic_loaded_state_rebase_acceleration_bias_suppressed_step_count = 0
    diagnostic_loaded_state_rebase_suppressed_acceleration_bias_peak_scale = 0.0
    diagnostic_loaded_state_rebase_settled_dwell_s = 0.0
    diagnostic_loaded_state_rebase_relative_speed_mps_at_trigger_by_anchor: dict[
        int, float
    ] = {}
    diagnostic_loaded_state_rebase_relative_speed_mps_at_completion_by_anchor: dict[
        int, float
    ] = {}
    diagnostic_loaded_state_rebase_cumulative_slip_m_at_trigger_by_link: dict[
        str, float
    ] = {}
    diagnostic_loaded_state_rebase_cumulative_slip_m_at_completion_by_link: dict[
        str, float
    ] = {}
    max_payload_feedforward_lead_over_observed_scale = 0.0
    last_payload_commanded_lift_progress_scale = 0.0
    payload_commanded_lift_progress_peak_scale = 0.0
    max_payload_feedforward_lag_behind_commanded_progress_scale = 0.0
    lift_acceleration_bias_active_count = 0
    lift_acceleration_bias_non_lift_active_count = 0
    lift_acceleration_bias_policy_command_active_count = 0
    lift_acceleration_bias_peak_scale = 0.0
    last_lift_acceleration_bias_scale = 0.0
    last_lift_acceleration_bias_commanded_progress_scale = 0.0
    lift_acceleration_bias_lift_off_scale: float | None = None
    lift_acceleration_bias_removal_complete_time_s: float | None = None
    lift_acceleration_bias_peak_force_world_z_n = 0.0
    lift_acceleration_bias_peak_residual_force_body_norm_n = 0.0
    last_lift_acceleration_bias_force_world_z_n = 0.0
    last_lift_acceleration_residual_wrench_body = [0.0] * 6
    payload_load_transfer_distance_m = float(
        config.contact_base_translation_speed_limit_mps
    ) * float(config.payload_load_transfer_s)
    if not math.isfinite(payload_load_transfer_distance_m) or (
        payload_load_transfer_distance_m <= 0.0
    ):
        raise RuntimeError(
            "Order8 measured payload load-transfer distance must be positive"
        )
    morphology_aware_module_root_target_count = 0
    first_transport_object_pose: Pose7D | None = None
    last_evidence = None
    last_control_result = None
    failure_reason: str | None = None
    commanded_base_target = initial_robot_base_pose
    max_base_target_step_m = 0.0
    max_contact_base_target_step_m = 0.0
    max_joint_position_command_lead_rad = 0.0
    max_joint_velocity_command_radps = 0.0
    max_observed_joint_limit_violation_rad = 0.0
    max_diagnostic_pitch_hold_error_rad = 0.0
    last_base_terminal_tracking_error_m = 0.0
    last_base_command_tracking_error_m = 0.0
    last_joint_positions: dict[str, float] = {}
    diagnostic_stop_reached = False
    last_selected_normal_force_n_by_link = {
        link_id: 0.0 for link_id in selected_link_ids
    }
    max_selected_normal_force_n_by_link = {
        link_id: 0.0 for link_id in selected_link_ids
    }
    last_selected_contact_normal_force_world_by_link = {
        link_id: (0.0, 0.0, 0.0) for link_id in selected_link_ids
    }
    last_selected_contact_application_point_world_by_link = {
        link_id: (0.0, 0.0, 0.0) for link_id in selected_link_ids
    }
    last_selected_friction_force_world_by_link = {
        link_id: (0.0, 0.0, 0.0) for link_id in selected_link_ids
    }
    last_selected_contact_force_matrix_world_by_link = {
        link_id: (0.0, 0.0, 0.0) for link_id in selected_link_ids
    }
    last_selected_body_linear_velocity_world_by_link = {
        link_id: (0.0, 0.0, 0.0) for link_id in selected_link_ids
    }
    last_selected_body_contact_velocity_world_by_link = {
        link_id: (0.0, 0.0, 0.0) for link_id in selected_link_ids
    }
    last_selected_object_contact_velocity_world_by_link = {
        link_id: (0.0, 0.0, 0.0) for link_id in selected_link_ids
    }
    last_selected_relative_contact_velocity_world_by_link = {
        link_id: (0.0, 0.0, 0.0) for link_id in selected_link_ids
    }
    last_selected_tangential_slip_velocity_world_by_link = {
        link_id: (0.0, 0.0, 0.0) for link_id in selected_link_ids
    }
    last_selected_tangential_slip_velocity_object_by_link = {
        link_id: (0.0, 0.0, 0.0) for link_id in selected_link_ids
    }
    last_selected_slip_contact_point_world_by_link = {
        link_id: (0.0, 0.0, 0.0) for link_id in selected_link_ids
    }
    last_selected_slip_contact_normal_world_by_link = {
        link_id: (0.0, 0.0, 0.0) for link_id in selected_link_ids
    }
    signed_cumulative_slip_displacement_world_m_by_link = {
        link_id: (0.0, 0.0, 0.0) for link_id in selected_link_ids
    }
    signed_cumulative_slip_displacement_object_m_by_link = {
        link_id: (0.0, 0.0, 0.0) for link_id in selected_link_ids
    }
    diagnostic_cumulative_slip_path_m_by_link = {
        link_id: 0.0 for link_id in selected_link_ids
    }
    diagnostic_lift_transition_stage = (
        "prelift_not_started"
        if diagnostic_separated_lift_transition
        else "disabled"
    )
    diagnostic_contact_point_vertical_velocity_bounds_mps_by_stage: dict[
        str, dict[str, dict[str, float]]
    ] = {}
    slip_vector_step_telemetry: list[dict[str, object]] = []
    max_selected_friction_force_magnitude_n_by_link = {
        link_id: 0.0 for link_id in selected_link_ids
    }
    max_abs_selected_friction_vertical_force_n_by_link = {
        link_id: 0.0 for link_id in selected_link_ids
    }
    contact_vector_telemetry_invalid_step_count = 0
    latest_dock_actuator_telemetry: dict[str, dict[str, object]] = {}
    diagnostic_post_grasp_joint_torque_bias_active_step_count = 0
    diagnostic_post_grasp_joint_torque_bias_joint_ids: set[str] = set()
    diagnostic_post_grasp_joint_torque_bias_last_map_nm: dict[str, float] = {}
    diagnostic_object_rotation_lock_orientation_xyzw: tuple[
        float, float, float, float
    ] | None = None
    diagnostic_object_rotation_projection_step_count = 0
    diagnostic_object_rotation_projection_max_deviation_rad = 0.0
    diagnostic_object_rotation_projection_max_angular_speed_rad_s = 0.0
    diagnostic_anchor_hold_joint_correction_active_step_count = 0
    diagnostic_anchor_hold_joint_correction_initial_targets_rad: dict[
        str, float
    ] = {}
    diagnostic_anchor_hold_joint_correction_last_targets_rad: dict[
        str, float
    ] = {}
    diagnostic_anchor_hold_joint_correction_last_velocity_targets_radps: dict[
        str, float
    ] = {}
    diagnostic_anchor_hold_joint_correction_joint_ids: set[str] = set()
    diagnostic_anchor_hold_joint_correction_max_translation_error_m = 0.0
    diagnostic_anchor_hold_joint_correction_max_attitude_error_rad = 0.0
    diagnostic_anchor_hold_joint_correction_max_target_offset_rad = 0.0
    diagnostic_anchor_hold_joint_correction_max_target_step_rad = 0.0
    diagnostic_anchor_hold_joint_correction_max_command_speed_radps = 0.0
    diagnostic_anchor_hold_joint_correction_last_reachability_status = "inactive"
    diagnostic_anchor_hold_joint_correction_max_reachability_residual = 0.0
    dock_actuator_telemetry_maxima = {
        "abs_position_error_rad": 0.0,
        "abs_measured_velocity_radps": 0.0,
        "abs_requested_unclipped_torque_bias_nm": 0.0,
        "abs_requested_limited_torque_bias_nm": 0.0,
        "abs_isaac_effort_target_nm": 0.0,
        "abs_estimated_position_drive_torque_nm": 0.0,
        "abs_estimated_total_drive_torque_nm": 0.0,
        "abs_isaac_computed_torque_nm": 0.0,
        "abs_isaac_applied_torque_nm": 0.0,
        "estimated_current_a": 0.0,
    }
    dock_actuator_envelope_violation_step_count = 0
    diagnostic_peak_torque_active_step_count = 0
    diagnostic_peak_torque_max_limit_nm = dock_continuous_torque_nm
    contact_force_impedance_active_step_count = 0
    contact_force_position_preload_active_step_count = 0
    latched_joint_position_hold_step_count = 0
    max_contact_force_impedance_joint_count = 0
    contact_force_impedance_peak_clipped_joint_ids: set[str] = set()
    contact_force_impedance_position_clipped_joint_ids: set[str] = set()
    contact_joint_drive_damping_scheduled = False
    contact_joint_drive_damping_targets: dict[str, float] = {}
    # Initial floor/takeoff commands run before contact-phase state is created.
    # Keep the normal whole-morphology QPID active until that later state
    # machine starts.
    contact_centering_active = False
    contact_motion_qpid_gain_scheduled = False
    previous_external_wrench_centroidal_model = None
    last_external_wrench_estimate = CentroidalExternalWrenchEstimate(
        valid=False,
        wrench_body=(0.0,) * 6,
        raw_wrench_body=(0.0,) * 6,
        bias_wrench_body=(0.0,) * 6,
        force_norm_n=0.0,
        torque_norm_nm=0.0,
        failure_reason="estimator_not_initialized",
    )
    contact_yield_requested = False
    contact_yield_triggered_time_s: float | None = None
    contact_yield_blend = 0.0
    contact_yield_active_step_count = 0
    contact_yield_full_step_count = 0
    contact_yield_restore_step_count = 0
    contact_yield_load_dwell_s_by_anchor = {
        anchor_id: 0.0 for anchor_id in selected_anchor_ids
    }
    selected_contact_raw_joint_torque_nm_by_anchor = {
        anchor_id: 0.0 for anchor_id in selected_anchor_ids
    }
    selected_contact_raw_joint_load_nm_by_anchor = {
        anchor_id: 0.0 for anchor_id in selected_anchor_ids
    }
    selected_contact_damping_drive_torque_nm_by_anchor = {
        anchor_id: 0.0 for anchor_id in selected_anchor_ids
    }
    selected_contact_joint_load_nm_by_anchor = {
        anchor_id: 0.0 for anchor_id in selected_anchor_ids
    }
    contact_stall_joint_load_nm_by_anchor = {
        anchor_id: 0.0 for anchor_id in selected_anchor_ids
    }
    contact_yield_minimum_pi_scale = 1.0
    contact_yield_maximum_external_force_n = 0.0
    contact_yield_maximum_external_torque_nm = 0.0
    contact_yield_maximum_translation_offset_m = 0.0
    contact_yield_trigger_anchor_ids: set[int] = set()
    contact_load_detection_armed_step_count = 0
    contact_load_detection_armed_time_s: float | None = None
    contact_admittance_requested = False
    contact_admittance_triggered_time_s: float | None = None
    contact_admittance_trigger_anchor_ids: set[int] = set()
    contact_yield_grasp_pose_rebased = False
    contact_yield_grasp_pose_rebase_time_s: float | None = None
    contact_yield_grasp_pose: Pose7D | None = None
    contact_yield_estimator_valid_step_count = 0
    contact_yield_estimator_invalid_step_count = 0
    last_contact_admittance_twist = (0.0,) * 6
    last_contact_admittance_translation_offset_world = (0.0, 0.0, 0.0)
    contact_yield_joint_drive_requested = False
    contact_yield_joint_drive_triggered_time_s: float | None = None
    contact_yield_joint_drive_blend = 0.0
    contact_yield_joint_drive_active_step_count = 0
    contact_yield_joint_drive_write_count = 0
    contact_yield_joint_drive_restore_write_count = 0
    contact_yield_joint_drive_minimum_stiffness_nm_per_rad = float(dock_stiffness)
    contact_yield_joint_drive_maximum_damping_nms_per_rad = float(dock_damping)
    contact_yield_joint_drive_last_stiffness_nm_per_rad = float(dock_stiffness)
    contact_yield_joint_drive_last_damping_nms_per_rad = float(dock_damping)
    contact_yield_joint_drive_stiffness_targets: dict[str, float] = {
        joint_id: float(dock_stiffness) for joint_id in expected_joint_ids
    }
    contact_yield_joint_drive_damping_targets: dict[str, float] = {
        joint_id: float(dock_damping) for joint_id in expected_joint_ids
    }

    def module_runtime_state(module_id: int) -> ModuleRuntimeState:
        robot = robots[module_id]
        module_pose, module_twist = _module_frame_pose_twist(
            robot,
            module_frame_link_id=module_frame_link_id,
        )
        return ModuleRuntimeState(
            module_id=module_id,
            pose_world=module_pose,
            twist_world=list(module_twist),
            joint_positions=_joint_state_dict(robot),
            joint_velocities=_joint_velocity_dict(robot),
        )

    def whole_structure_observation() -> RuntimeObservation:
        return _whole_structure_runtime_observation(
            time_s=current_time_s,
            morphology_graph=morphology_graph,
            module_states_by_id={
                module_id: module_runtime_state(module_id) for module_id in module_ids
            },
            controller_status=last_status[morphology_graph.base_module_id],
            phase_label=planner.phase.value,
        )

    def order9_teacher_observations(
        *,
        evidence: Any | None,
        raw_contact_valid: bool,
        phase_transitioned_now: bool = False,
    ) -> tuple[RuntimeObservation, RuntimeObservation]:
        """Build actor-safe state and a separate privileged reward view."""

        base = whole_structure_observation()
        measured_object = _object_state(object_asset)
        object_runtime_state = ObjectRuntimeState(
            object_id="order8_object",
            pose_world=tuple(float(value) for value in measured_object["pose"]),
            twist_world=[float(value) for value in measured_object["twist"]],
        )
        phase = planner.phase
        phase_timeout = (
            float(planner_config.contact_acquisition_timeout_s)
            if phase == Order8NaturalContactPhase.CONTACT_ACQUISITION
            else float(planner_config.phase_timeout_s)
        )
        progress_ratio = (
            1.0
            if phase == Order8NaturalContactPhase.COMPLETE
            else min(
                1.0,
                (
                    0.0
                    if phase_transitioned_now
                    else max(0.0, current_time_s - phase_started_s)
                )
                / phase_timeout,
            )
        )
        actor_progress = TaskProgressState(
            phase_label=phase.value,
            progress_ratio=progress_ratio,
            success=phase == Order8NaturalContactPhase.COMPLETE,
            # A raw-monitor reason is intentionally not persisted in actor data.
            failure_reason=None,
            metrics={},
        )
        actor = RuntimeObservation(
            time_s=float(current_time_s),
            morphology_graph=morphology_graph,
            module_states=base.module_states,
            object_states=[object_runtime_state],
            contact_states=[],
            controller_status=base.controller_status,
            task_progress=actor_progress,
        )
        reward_metrics: dict[str, float] = {
            "grasp_data_available": 0.0,
            "contact_data_available": 0.0,
            "slip_data_available": 0.0,
            "collision_data_available": 0.0,
        }
        reward_failure_reason = None
        if evidence is not None and raw_contact_valid:
            required_contacts = max(1, int(config.required_distinct_dock_links))
            reward_metrics.update(
                {
                    "grasp_data_available": 1.0,
                    "contact_data_available": 1.0,
                    "grasp_maintenance": min(
                        1.0,
                        float(evidence.selected_distinct_contact_count)
                        / float(required_contacts),
                    ),
                    "slip_data_available": 1.0,
                    "slip_speed_mps": float(
                        evidence.max_tangential_slip_speed_mps
                    ),
                    "collision_data_available": 1.0,
                    "hard_collision": 1.0
                    if (
                        bool(evidence.unintended_contact_link_ids)
                        or float(evidence.max_penetration_m)
                        > float(config.max_penetration_m) + 1.0e-12
                    )
                    else 0.0,
                }
            )
            reward_failure_reason = (
                ",".join(evidence.failure_reasons)
                if evidence.failure_reasons
                else None
            )
        reward = RuntimeObservation(
            time_s=actor.time_s,
            morphology_graph=morphology_graph,
            module_states=actor.module_states,
            object_states=actor.object_states,
            contact_states=[],
            controller_status=actor.controller_status,
            task_progress=TaskProgressState(
                phase_label=phase.value,
                progress_ratio=progress_ratio,
                success=actor_progress.success,
                failure_reason=reward_failure_reason,
                metrics=reward_metrics,
            ),
        )
        return actor, reward

    def full_joint_vector(
        *,
        neutral_positions: dict[str, float] | None = None,
        torque_bias_limit_nm: float | None = None,
    ) -> DockJointVector:
        nonlocal max_observed_joint_limit_violation_rad, last_joint_positions
        positions: list[float] = []
        velocities: list[float] = []
        observed_positions: dict[str, float] = {}
        for joint_index, global_id in enumerate(expected_joint_ids):
            module_text, local_id = global_id.split(":", 1)
            module_id = int(module_text[len("module_") :])
            position_map = _joint_state_dict(robots[module_id])
            velocity_map = _joint_velocity_dict(robots[module_id])
            if local_id not in position_map or local_id not in velocity_map:
                raise RuntimeError(f"Order8 missing observed Dock joint {global_id}")
            observed_joint_ids.add(global_id)
            raw_position = float(position_map[local_id])
            limit = joint_limits[joint_index]
            violation = max(
                limit.position_lower_rad - raw_position,
                raw_position - limit.position_upper_rad,
                0.0,
            )
            max_observed_joint_limit_violation_rad = max(
                max_observed_joint_limit_violation_rad,
                violation,
            )
            if violation > float(config.joint_limit_state_tolerance_rad):
                raise RuntimeError(
                    f"Order8 observed Dock joint {global_id!r} at "
                    f"{raw_position:.9f} rad outside "
                    f"[{limit.position_lower_rad:.9f}, "
                    f"{limit.position_upper_rad:.9f}] rad by "
                    f"{violation:.9f} rad"
                )
            clamped_position = min(
                max(raw_position, limit.position_lower_rad),
                limit.position_upper_rad,
            )
            positions.append(clamped_position)
            observed_positions[global_id] = raw_position
            velocities.append(float(velocity_map[local_id]))
        last_joint_positions = observed_positions
        active_limits = joint_limits
        if torque_bias_limit_nm is not None:
            torque_limit = float(torque_bias_limit_nm)
            if (
                not math.isfinite(torque_limit)
                or torque_limit < dock_continuous_torque_nm
                or torque_limit > dock_peak_torque_nm
            ):
                raise RuntimeError(
                    "Order8 active torque-bias limit must remain inside the "
                    "AK40-10 continuous/peak envelope"
                )
            active_limits = tuple(
                replace(limit, max_torque_nm=torque_limit) for limit in joint_limits
            )
        return DockJointVector(
            joint_ids=expected_joint_ids,
            positions_rad=tuple(positions),
            velocities_radps=tuple(velocities),
            neutral_positions_rad=tuple(
                float((neutral_positions or {}).get(joint_id, 0.0))
                for joint_id in expected_joint_ids
            ),
            limits=active_limits,
        )

    def apply_commands(
        joint_result: Any,
        base_target: Pose7D,
        *,
        centroidal_measured_joint_positions: Mapping[str, float],
        payload_feedforward_scale: float,
        centroidal_force_bias_world: Sequence[float] = (0.0, 0.0, 0.0),
        actuator_torque_bias_limit_nm: float | None = None,
        zero_thrust: bool = False,
        tracking_profile: QPIDTrackingProfile | None = None,
        admittance_active: bool = False,
        external_wrench_estimate: CentroidalExternalWrenchEstimate | None = None,
        order9_teacher_trajectory: Any | None = None,
        base_twist_world: tuple[float, float, float, float, float, float] = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    ) -> None:
        nonlocal command_index, qp_infeasible_count, controller_failure_count
        nonlocal missing_count, unsupported_count, clipped_count, unresolved_count
        nonlocal payload_feedforward_active_count
        nonlocal payload_feedforward_peak_scale
        nonlocal payload_feedforward_max_scale_step
        nonlocal last_payload_feedforward_scale
        nonlocal last_payload_coupling
        nonlocal last_full_payload_coupling
        nonlocal morphology_aware_module_root_target_count
        nonlocal latest_dock_actuator_telemetry
        nonlocal previous_controller_command
        nonlocal dock_actuator_envelope_violation_step_count
        nonlocal state_trace_step_counter
        nonlocal last_contact_admittance_twist
        nonlocal last_contact_admittance_translation_offset_world
        nonlocal contact_yield_maximum_translation_offset_m
        nonlocal last_lift_acceleration_residual_wrench_body
        nonlocal lift_acceleration_bias_policy_command_active_count
        nonlocal lift_acceleration_bias_peak_residual_force_body_norm_n
        nonlocal object_root_pose_write_count
        nonlocal diagnostic_object_rotation_lock_orientation_xyzw
        nonlocal diagnostic_object_rotation_projection_step_count
        nonlocal diagnostic_object_rotation_projection_max_deviation_rad
        nonlocal diagnostic_object_rotation_projection_max_angular_speed_rad_s
        if (
            diagnostic_lock_object_rotation
            and planner.phase
            in {
                Order8NaturalContactPhase.LIFT,
                Order8NaturalContactPhase.TRANSPORT,
                Order8NaturalContactPhase.PLACE,
            }
        ):
            current_object_state = _object_state(object_asset)
            if diagnostic_object_rotation_lock_orientation_xyzw is None:
                diagnostic_object_rotation_lock_orientation_xyzw = tuple(
                    float(value) for value in current_object_state["pose"][3:7]
                )
            (
                projected_pose,
                projected_twist,
                angular_deviation_rad,
                angular_speed_rad_s,
            ) = _project_object_rotation_state(
                current_object_state["pose"],
                current_object_state["twist"],
                locked_orientation_xyzw=(
                    diagnostic_object_rotation_lock_orientation_xyzw
                ),
            )
            object_asset.write_root_pose_to_sim(
                torch.tensor(
                    [projected_pose],
                    dtype=torch.float32,
                    device=sim.device,
                )
            )
            object_asset.write_root_velocity_to_sim(
                torch.tensor(
                    [projected_twist],
                    dtype=torch.float32,
                    device=sim.device,
                )
            )
            sim.forward()
            object_asset.update(sim_dt)
            object_root_pose_write_count += 1
            diagnostic_object_rotation_projection_step_count += 1
            diagnostic_object_rotation_projection_max_deviation_rad = max(
                diagnostic_object_rotation_projection_max_deviation_rad,
                angular_deviation_rad,
            )
            diagnostic_object_rotation_projection_max_angular_speed_rad_s = max(
                diagnostic_object_rotation_projection_max_angular_speed_rad_s,
                angular_speed_rad_s,
            )
        source = joint_result.policy_command
        feedforward_scale = float(payload_feedforward_scale)
        if (
            not math.isfinite(feedforward_scale)
            or feedforward_scale < 0.0
            or feedforward_scale > 1.0
        ):
            raise RuntimeError(
                "Order8 payload feed-forward scale must be finite and in [0, 1]"
            )
        if feedforward_scale > 0.0:
            payload_feedforward_active_count += 1
        payload_feedforward_peak_scale = max(
            payload_feedforward_peak_scale,
            feedforward_scale,
        )
        payload_feedforward_max_scale_step = max(
            payload_feedforward_max_scale_step,
            abs(feedforward_scale - last_payload_feedforward_scale),
        )
        last_payload_feedforward_scale = feedforward_scale
        force_bias_world = tuple(
            float(value) for value in centroidal_force_bias_world
        )
        if len(force_bias_world) != 3 or not all(
            math.isfinite(value) for value in force_bias_world
        ):
            raise RuntimeError(
                "Order8 centroidal force bias must contain three finite world-frame values"
            )
        force_bias_active = any(abs(value) > 0.0 for value in force_bias_world)
        commanded_joint_positions = dict(source.joint_position_targets)
        if set(commanded_joint_positions) != set(expected_joint_ids):
            raise RuntimeError(
                "Order8 commanded Dock positions must cover the full "
                "whole-structure kinematics state"
            )
        measured_joint_positions = _centroidal_measured_joint_reference(
            expected_joint_ids=expected_joint_ids,
            actuator_position_targets=commanded_joint_positions,
            measured_joint_positions=centroidal_measured_joint_positions,
        )
        commanded_structure_kinematics = whole_structure_kinematics.forward(
            morphology_graph,
            physical_model,
            measured_joint_positions,
            base_target,
            anchor_references,
        )
        module_root_targets = commanded_structure_kinematics.module_root_poses_world
        if set(module_root_targets) != set(module_ids):
            raise RuntimeError(
                "Order8 morphology-aware module root targets must cover "
                "exactly the spawned modules"
            )
        morphology_aware_module_root_target_count += 1
        position_commanded_ids.update(source.joint_position_targets)
        velocity_commanded_ids.update(source.joint_velocity_targets)
        torque_commanded_ids.update(source.joint_torque_bias)

        observation = whole_structure_observation()
        actual_states_by_id = {
            int(state.module_id): state for state in observation.module_states
        }
        target_states_by_id: dict[int, ModuleRuntimeState] = {}
        for module_id in module_ids:
            module_target = tuple(
                float(value) for value in module_root_targets[module_id]
            )
            prefix = f"module_{module_id}:"
            position_targets = {
                key: value
                for key, value in measured_joint_positions.items()
                if key.startswith(prefix)
            }
            velocity_targets = {
                key: value
                for key, value in source.joint_velocity_targets.items()
                if key.startswith(prefix)
            }
            actual_state = actual_states_by_id[module_id]
            target_joint_positions = dict(actual_state.joint_positions)
            target_joint_positions.update(
                {
                    key.split(":", 1)[1]: float(value)
                    for key, value in position_targets.items()
                }
            )
            target_joint_velocities = dict(actual_state.joint_velocities)
            target_joint_velocities.update(
                {
                    key.split(":", 1)[1]: float(value)
                    for key, value in velocity_targets.items()
                }
            )
            target_states_by_id[module_id] = ModuleRuntimeState(
                module_id=module_id,
                pose_world=module_target,
                twist_world=list(base_twist_world),
                joint_positions=target_joint_positions,
                joint_velocities=target_joint_velocities,
            )

        target_observation = _whole_structure_runtime_observation(
            time_s=current_time_s,
            morphology_graph=morphology_graph,
            module_states_by_id=target_states_by_id,
            controller_status=last_status[morphology_graph.base_module_id],
            phase_label=planner.phase.value,
        )
        target_centroidal_model = centroidal_target_builder.build(
            morphology_graph,
            physical_model,
            target_observation,
        )
        actual_centroidal_model = None
        if feedforward_scale > 0.0 or admittance_active or force_bias_active:
            actual_centroidal_model = centroidal_target_builder.build(
                morphology_graph,
                physical_model,
                observation,
            )
        payload_coupling = None
        if feedforward_scale > 0.0:
            if actual_centroidal_model is None:
                raise RuntimeError(
                    "Order8 payload feed-forward lacks a centroidal state"
                )
            measured_object_pose = tuple(_object_state(object_asset)["pose"])
            full_payload_coupling = _natural_contact_payload_coupling(
                control_body_pose_world=actual_centroidal_model.body_pose_world,
                object_com_pose_world=measured_object_pose,
                object_mass_kg=float(config.object_mass_kg),
                object_size_m=config.object_size_m,
                load_transfer_scale=feedforward_scale,
                contact_model=str(config.contact_model),
            )
            if full_payload_coupling is None:
                raise RuntimeError(
                    "positive payload feed-forward scale produced no coupling"
                )
            last_full_payload_coupling = full_payload_coupling.to_dict()
            payload_coupling = _diagnostic_payload_coupling_component_view(
                full_payload_coupling,
                component_mode=diagnostic_payload_coupling_component_mode,
            )
            last_payload_coupling = payload_coupling.to_dict()
        desired_body_pose = target_centroidal_model.body_pose_world
        desired_body_twist = tuple(
            float(value) for value in target_centroidal_model.body_twist_world
        )
        if admittance_active:
            if actual_centroidal_model is None:
                raise RuntimeError(
                    "Order8 contact admittance lacks a centroidal state"
                )
            admittance_command = contact_admittance_controller.update(
                nominal_pose_world=target_centroidal_model.body_pose_world,
                current_pose_world=actual_centroidal_model.body_pose_world,
                estimate=(
                    external_wrench_estimate
                    if external_wrench_estimate is not None
                    else last_external_wrench_estimate
                ),
                dt_s=sim_dt,
                active=True,
                linear_projection_axis_world=contact_admittance_axis_world,
                angular_admittance_enabled=False,
            )
            desired_body_pose = admittance_command.desired_body_pose
            desired_body_twist = tuple(
                float(desired_body_twist[index])
                + float(admittance_command.desired_body_twist[index])
                for index in range(6)
            )
            last_contact_admittance_twist = (
                admittance_command.desired_body_twist
            )
            last_contact_admittance_translation_offset_world = (
                admittance_command.translation_offset_world
            )
            contact_yield_maximum_translation_offset_m = max(
                contact_yield_maximum_translation_offset_m,
                _norm(admittance_command.translation_offset_world),
            )
        else:
            contact_admittance_controller.reset()
            last_contact_admittance_twist = (0.0,) * 6
            last_contact_admittance_translation_offset_world = (0.0, 0.0, 0.0)
        residual = [0.0] * 6
        if force_bias_active:
            if actual_centroidal_model is None:
                raise RuntimeError(
                    "Order8 centroidal force bias lacks a centroidal state"
                )
            residual[:3] = _vector_world_to_pose_local(
                actual_centroidal_model.body_pose_world,
                force_bias_world,
            )
            lift_acceleration_bias_policy_command_active_count += 1
            lift_acceleration_bias_peak_residual_force_body_norm_n = max(
                lift_acceleration_bias_peak_residual_force_body_norm_n,
                _norm(residual[:3]),
            )
        last_lift_acceleration_residual_wrench_body = list(residual)
        policy = PolicyCommand(
            desired_body_pose=desired_body_pose,
            desired_body_twist=list(desired_body_twist),
            residual_wrench_body=residual,
            control_contract_version=POLICY_COMMAND_CONTRACT_CENTROIDAL,
            joint_position_targets=dict(source.joint_position_targets),
            joint_velocity_targets=dict(source.joint_velocity_targets),
            joint_torque_bias=dict(source.joint_torque_bias),
        )
        active_controller = (
            contact_centering_controller
            if contact_motion_qpid_gain_scheduled
            else controller
        )
        command = active_controller.compute(
            ControllerContext(
                runtime_observation=observation,
                morphology_graph=morphology_graph,
                physical_model=physical_model,
                active_knot=InteractionKnot(
                    t_rel_s=current_time_s,
                    contact_assignments=[],
                ),
                policy_command=policy,
                previous_command=previous_controller_command,
                control_dt_s=sim_dt,
                payload_coupling=payload_coupling,
            ),
            tracking_profile=tracking_profile,
        )
        bridge_torque_limit_nm = (
            dock_continuous_torque_nm
            if actuator_torque_bias_limit_nm is None
            else float(actuator_torque_bias_limit_nm)
        )
        active_actuator_mapping = (
            full_actuator_mapping
            if math.isclose(
                bridge_torque_limit_nm,
                dock_continuous_torque_nm,
                abs_tol=1.0e-12,
            )
            else _actuator_mapping_with_torque_bias_limit(
                full_actuator_mapping,
                active_limit_nm=bridge_torque_limit_nm,
            )
        )
        record = bridge.convert(
            command,
            active_actuator_mapping,
            time_s=current_time_s,
            command_index=command_index,
        )
        if order9_teacher_trajectory is not None:
            if order9_teacher_collector is None:
                raise RuntimeError(
                    "Order9 teacher trajectory supplied without an active collector"
                )
            order9_teacher_collector.record_command(
                trajectory=order9_teacher_trajectory,
                policy_command=policy,
                controller_command=command,
                actuator_target_record=record.to_dict(),
                decision_dt_s=sim_dt,
            )
        for module_id in module_ids:
            application = _apply_record(
                robots[module_id],
                record,
                physical_model,
                module_id=module_id,
                device=device,
            )
            if zero_thrust:
                robots[module_id].permanent_wrench_composer.reset()
            unresolved_count += int(application["unresolved_target_count"])
        previous_controller_command = command
        for module_id in module_ids:
            last_status[module_id] = command.controller_status
        qp_infeasible_count += int(not command.controller_status.qp_feasible)
        controller_failure_count += int(
            command.controller_status.status in {"fault", "infeasible"}
        )
        missing_count += len(record.missing_actuators)
        unsupported_count += len(record.unsupported_actuators)
        clipped_count += len(record.clipped_targets)
        for robot in robots.values():
            robot.write_data_to_sim()
        sim.step()
        for robot in robots.values():
            robot.update(sim_dt)
        object_asset.update(sim_dt)
        state_trace_step_counter += 1
        if (
            state_trace_output_path is not None
            and state_trace_step_counter % state_trace_frame_stride == 0
        ):
            state_trace_frames.append(
                _capture_order8_state_trace_frame(
                    simulation_time_s=current_time_s + sim_dt,
                    phase=planner.phase.value,
                    robots=robots,
                    object_asset=object_asset,
                )
            )
        torque_mapping = getattr(joint_result, "torque_mapping", None)
        unclipped_torque_bias = (
            dict(torque_mapping.unclipped_joint_torque_bias)
            if torque_mapping is not None
            else dict(source.joint_torque_bias)
        )
        latest_dock_actuator_telemetry = _dock_joint_actuator_telemetry(
            robots,
            expected_joint_ids,
            requested_position_targets=source.joint_position_targets,
            requested_velocity_targets=source.joint_velocity_targets,
            requested_unclipped_torque_bias=unclipped_torque_bias,
            requested_limited_torque_bias=source.joint_torque_bias,
            peak_torque_nm=dock_peak_torque_nm,
            peak_current_a=dock_peak_current_a,
        )
        telemetry_maximum_sources = {
            "abs_position_error_rad": "position_error_rad",
            "abs_measured_velocity_radps": "measured_velocity_radps",
            "abs_requested_unclipped_torque_bias_nm": (
                "requested_unclipped_torque_bias_nm"
            ),
            "abs_requested_limited_torque_bias_nm": (
                "requested_limited_torque_bias_nm"
            ),
            "abs_isaac_effort_target_nm": "isaac_effort_target_nm",
            "abs_estimated_position_drive_torque_nm": (
                "estimated_position_drive_torque_nm"
            ),
            "abs_estimated_total_drive_torque_nm": ("estimated_total_drive_torque_nm"),
            "abs_isaac_computed_torque_nm": "isaac_computed_torque_nm",
            "abs_isaac_applied_torque_nm": "isaac_applied_torque_nm",
            "estimated_current_a": "estimated_current_a",
        }
        for maximum_key, source_key in telemetry_maximum_sources.items():
            dock_actuator_telemetry_maxima[maximum_key] = max(
                float(dock_actuator_telemetry_maxima[maximum_key]),
                max(
                    (
                        abs(float(values[source_key]))
                        for values in latest_dock_actuator_telemetry.values()
                    ),
                    default=0.0,
                ),
            )
        if (
            _telemetry_max_abs(
                latest_dock_actuator_telemetry, "measured_velocity_radps"
            )
            > configured_dock_velocity_limit + 1.0e-6
            or _telemetry_max_abs(
                latest_dock_actuator_telemetry, "isaac_applied_torque_nm"
            )
            > dock_peak_torque_nm + 1.0e-6
            or _telemetry_max_abs(latest_dock_actuator_telemetry, "estimated_current_a")
            > dock_peak_current_a + 1.0e-6
        ):
            dock_actuator_envelope_violation_step_count += 1
        command_index += 1
        if args.realtime_playback:
            time.sleep(max(0.0, sim_dt))

    initial_joint_vector = full_joint_vector()
    initial_kinematics = whole_structure_kinematics.compute(
        morphology_graph,
        physical_model,
        _global_dock_position_map(initial_joint_vector),
        _module_frame_pose_twist(
            robots[morphology_graph.base_module_id],
            module_frame_link_id=module_frame_link_id,
        )[0],
        anchor_references,
    )
    if initial_kinematics.ordered_global_dock_joint_ids != expected_joint_ids:
        raise RuntimeError(
            "Order8 whole-structure Jacobian columns do not match the full Dock state"
        )
    zero_tasks = _anchor_task_linearizations(
        initial_kinematics,
        selections,
        desired_anchor_poses=initial_kinematics.anchor_poses_world,
        wrench_targets={
            selection.anchor_id: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            for selection in selections
        },
    )
    initial_joint_result = low_level.compute(initial_joint_vector, zero_tasks)
    joint_position_reference_by_id = dict(
        initial_joint_result.policy_command.joint_position_targets
    )
    diagnostic_pitch_hold_positions_rad = (
        {
            str(joint_id): float(position)
            for joint_id, position in zip(
                initial_joint_vector.joint_ids,
                initial_joint_vector.positions_rad,
                strict=True,
            )
            if str(joint_id).split(":", 1)[-1].startswith(
                "pitch_dock_mech_joint"
            )
        }
        if diagnostic_only
        else {}
    )
    last_control_result = initial_joint_result
    last_kinematics = initial_kinematics
    initial_base_root_pose = _module_frame_pose_twist(
        robots[morphology_graph.base_module_id],
        module_frame_link_id=module_frame_link_id,
    )[0]
    pregrasp_hold_anchor_poses_base = {
        anchor_id: compose_pose(
            inverse_pose(initial_base_root_pose),
            anchor_pose,
        )
        for anchor_id, anchor_pose in initial_kinematics.anchor_poses_world.items()
    }
    preliminary_grasp_shape_kinematics = whole_structure_kinematics.forward(
        morphology_graph,
        physical_model,
        _global_dock_position_map(initial_joint_vector),
        grasp_base_pose,
        anchor_references,
    )
    preliminary_grasp_body_poses = {
        (
            reference.surface.module_id,
            reference.surface.mechanism_link_id,
        ): compose_pose(
            preliminary_grasp_shape_kinematics.anchor_poses_world[
                reference.anchor.anchor_id
            ],
            inverse_pose(reference.anchor.local_pose),
        )
        for reference in anchor_references
    }
    nominal_face_center_poses = _desired_anchor_poses(
        selections,
        object_pose,
        config.object_size_m,
        pregrasp=False,
        inward_overtravel_m=0.0,
        orientation_by_anchor_id={
            anchor_id: pose[3:]
            for anchor_id, pose in (
                preliminary_grasp_shape_kinematics.anchor_poses_world.items()
            )
        },
    )
    preliminary_surface_points_world = {
        anchor_id: _gripper_object_surface_sample_query_from_body_poses(
            bounds,
            preliminary_grasp_body_poses,
            object_pose,
            runtime_object_size_m,
        )[1]
        for anchor_id, bounds in selected_gripper_local_aabbs_by_anchor.items()
    }
    object_approach_axis_world = _unit(
        (
            float(object_pose[0]) - float(initial_pair_center[0]),
            float(object_pose[1]) - float(initial_pair_center[1]),
            0.0,
        )
    )
    mesh_pair_base_centering_correction_world = (
        (0.0, 0.0, 0.0)
        if diagnostic_qclose_fixture or diagnostic_force_fixture
        else _horizontal_mesh_pair_centering_correction_world(
            surface_point_world_by_anchor=preliminary_surface_points_world,
            nominal_contact_pose_world_by_anchor=nominal_face_center_poses,
            approach_axis_world=object_approach_axis_world,
            maximum_correction_m=float(config.contact_tangential_tolerance_m),
        )
    )
    grasp_base_pose = _offset_pose(
        grasp_base_pose,
        dx=mesh_pair_base_centering_correction_world[0],
        dy=mesh_pair_base_centering_correction_world[1],
        dz=mesh_pair_base_centering_correction_world[2],
    )
    lift_base_pose = _offset_pose(grasp_base_pose, dz=0.15)
    transport_base_pose = _offset_pose(
        lift_base_pose, dx=config.required_transport_distance_m
    )
    place_base_pose = _offset_pose(
        grasp_base_pose, dx=config.required_transport_distance_m
    )
    retreat_base_pose = _offset_pose(place_base_pose, dx=-0.10, dz=0.20)
    grasp_shape_kinematics = whole_structure_kinematics.forward(
        morphology_graph,
        physical_model,
        _global_dock_position_map(initial_joint_vector),
        grasp_base_pose,
        anchor_references,
    )
    grasp_body_poses = {
        (
            reference.surface.module_id,
            reference.surface.mechanism_link_id,
        ): compose_pose(
            grasp_shape_kinematics.anchor_poses_world[reference.anchor.anchor_id],
            inverse_pose(reference.anchor.local_pose),
        )
        for reference in anchor_references
    }
    staging_plan = _mesh_aware_staging_plan(
        selected_gripper_local_aabbs,
        grasp_body_poses,
        grasp_base_pose=grasp_base_pose,
        object_pose=object_pose,
        object_size=config.object_size_m,
        required_clearance_m=float(config.pregrasp_mesh_clearance_m),
        maximum_retreat_m=float(config.initial_object_standoff_m),
    )
    approach_base_pose = staging_plan.base_pose_world
    opening_plan = _mesh_aware_anchor_opening_plan(
        selected_gripper_local_aabbs,
        grasp_body_poses,
        anchor_id_by_module_link={
            (
                reference.surface.module_id,
                reference.surface.mechanism_link_id,
            ): reference.anchor.anchor_id
            for reference in anchor_references
        },
        anchor_pose_world_by_id=dict(grasp_shape_kinematics.anchor_poses_world),
        inward_normal_world_by_anchor={
            selection.anchor_id: selection.inward_normal_world
            for selection in selections
        },
        grasp_base_pose=grasp_base_pose,
        object_pose=object_pose,
        object_size=config.object_size_m,
        required_clearance_m=float(config.pregrasp_mesh_clearance_m),
        maximum_opening_m=float(config.initial_object_standoff_m),
    )
    contact_anchor_targets_world = _desired_anchor_poses(
        selections,
        object_pose,
        config.object_size_m,
        pregrasp=False,
        inward_overtravel_m=float(config.contact_closure_inward_overtravel_m),
        orientation_by_anchor_id={
            anchor_id: pose[3:]
            for anchor_id, pose in (grasp_shape_kinematics.anchor_poses_world.items())
        },
    )
    reference_object_pose = tuple(float(value) for value in object_pose)
    from amsrr.geometry.pose_math import matvec, transform_from_pose

    reference_object_from_world_rotation = transform_from_pose(
        inverse_pose(reference_object_pose)
    ).rotation
    contact_inward_normal_object_by_anchor = {
        int(selection.anchor_id): _unit(
            matvec(
                reference_object_from_world_rotation,
                tuple(float(value) for value in selection.inward_normal_world),
            )
        )
        for selection in selections
    }
    contact_terminal_anchor_targets = {
        anchor_id: compose_pose(
            inverse_pose(grasp_base_pose),
            pose_world,
        )
        for anchor_id, pose_world in contact_anchor_targets_world.items()
    }
    release_terminal_anchor_targets = dict(opening_plan.anchor_poses_base)
    commanded_anchor_targets_base = dict(pregrasp_hold_anchor_poses_base)
    pregrasp_open_anchor_poses_base: dict[int, Pose7D] | None = None
    contact_axial_aligned = False
    contact_axial_overlap_at_latch_m: float | None = None
    contact_axial_hold_base_pose: Pose7D | None = None
    contact_axial_settle_dwell_s = 0.0
    contact_axial_settle_position_tolerance_m = min(
        float(config.pregrasp_position_tolerance_m),
        float(config.contact_tangential_tolerance_m),
    )
    contact_axial_settle_base_speed_tolerance_mps = float(
        config.pregrasp_linear_speed_tolerance_mps
    )
    contact_mesh_clearance_reacquire_threshold_m = float(
        config.contact_surface_arm_clearance_m
    ) + float(config.contact_penetration_noise_floor_m)
    contact_side_closure_enabled = False
    contact_mesh_precenter_complete = False
    contact_mesh_precenter_dwell_s = 0.0
    contact_mesh_precenter_completed_time_s: float | None = None
    grasp_hold_anchor_poses_base: dict[int, Pose7D] | None = None
    if diagnostic_only:
        # The diagnostic starts already airborne and executes only a short
        # stabilization hold.  It is deliberately not an acceptance takeoff.
        diagnostic_hold_steps = (
            0
            if (
                diagnostic_qclose_checkpoint_state is not None
                or diagnostic_near_contact_fixture
            )
            else max(1, int(math.ceil(0.10 / sim_dt)))
        )
        for _ in range(diagnostic_hold_steps):
            apply_commands(
                initial_joint_result,
                initial_robot_base_pose,
                centroidal_measured_joint_positions=_global_dock_position_map(
                    full_joint_vector()
                ),
                payload_feedforward_scale=0.0,
            )
            current_time_s += sim_dt
    else:
        # Floor settle with explicit all-Dock q/qdot/tau=0 and no rotor force.
        floor_steps = max(1, int(math.ceil(1.0 / sim_dt)))
        for _ in range(floor_steps):
            apply_commands(
                initial_joint_result,
                floor_base_pose,
                centroidal_measured_joint_positions=_global_dock_position_map(
                    full_joint_vector()
                ),
                payload_feedforward_scale=0.0,
                zero_thrust=True,
            )
            current_time_s += sim_dt

        # Deterministic takeoff/hover remains outside the contact phase monitor.
        takeoff_steps = max(1, int(math.ceil(2.0 / sim_dt)))
        takeoff_velocity = tuple(
            (float(hover_base_pose[index]) - float(floor_base_pose[index]))
            / (float(takeoff_steps) * sim_dt)
            for index in range(3)
        )
        for index in range(takeoff_steps):
            alpha = min(1.0, float(index + 1) / float(takeoff_steps))
            target = _interpolate_pose(floor_base_pose, hover_base_pose, alpha)
            apply_commands(
                initial_joint_result,
                target,
                centroidal_measured_joint_positions=_global_dock_position_map(
                    full_joint_vector()
                ),
                payload_feedforward_scale=0.0,
                base_twist_world=(*takeoff_velocity, 0.0, 0.0, 0.0),
            )
            current_time_s += sim_dt
        hover_steps = max(
            1,
            int(math.ceil(float(config.hover_dwell_s) / sim_dt)),
        )
        for _ in range(hover_steps):
            apply_commands(
                initial_joint_result,
                hover_base_pose,
                centroidal_measured_joint_positions=_global_dock_position_map(
                    full_joint_vector()
                ),
                payload_feedforward_scale=0.0,
            )
            current_time_s += sim_dt

    phase_started_s = current_time_s
    nonprivileged_settle_dwell_s = 0.0
    nonprivileged_contact_command_dwell_s = 0.0
    prelift_relative_motion_settle_achieved = False
    prelift_relative_speed_threshold_mps = _prelift_relative_speed_threshold_mps(
        maintained_contact_slip_limit_mps=(
            config.max_tangential_slip_speed_mps
        )
    )
    nonprivileged_release_command_dwell_s = 0.0
    nonprivileged_contact_force_ramp_elapsed_s_by_anchor = {
        int(selection.anchor_id): 0.0 for selection in selections
    }
    nonprivileged_contact_stall_dwell_s_by_anchor = {
        int(selection.anchor_id): 0.0 for selection in selections
    }
    nonprivileged_contact_stall_dwell_s = 0.0
    nonprivileged_contact_configuration_dwell_s = 0.0
    last_contact_surface_load_arrest_candidates = {
        int(selection.anchor_id): False for selection in selections
    }
    contact_stall_latched_anchor_poses_base: dict[int, Pose7D] = {}
    # Preserve the physical arrest in world coordinates while the centroidal
    # frame is translated to bring the remaining side into contact.  A base-
    # frame-only snapshot would move the supposedly held contact with the base.
    contact_stall_latched_anchor_poses_world: dict[int, Pose7D] = {}
    contact_stall_latched_mesh_clearance_m_by_anchor: dict[int, float] = {}
    contact_stall_latched_tangential_offset_m_by_anchor: dict[
        int, tuple[float, float]
    ] = {}
    contact_reacquired_hold_anchor_poses_base: dict[int, Pose7D] = {}
    contact_stall_latched = False
    contact_configuration_latched = False
    contact_configuration_latched_time_s: float | None = None
    post_qclose_joint_settle_complete = bool(diagnostic_force_fixture)
    post_qclose_joint_settle_dwell_s = (
        float(config.contact_stall_dwell_s)
        if post_qclose_joint_settle_complete
        else 0.0
    )
    post_qclose_joint_speed_threshold_radps = min(
        0.02,
        0.2 * float(contact_joint_velocity_limit),
    )
    post_qclose_max_measured_joint_speed_radps = 0.0
    post_qclose_position_rebase_step_count = 0
    # The historical mesh-tracking IK preload remains disabled.  The active
    # v8 path below continues the one-shot fixed closure direction directly in
    # joint space and freezes each side from damping-compensated actuator load.
    post_qclose_geometric_preload_complete = True
    post_qclose_geometric_preload_anchor_poses_object: dict[int, Pose7D] = {}
    post_qclose_geometric_preload_commanded_anchor_targets_world: dict[
        int, Pose7D
    ] = {}
    post_qclose_geometric_preload_surface_point_local_by_anchor: dict[
        int, tuple[float, float, float]
    ] = {}
    post_qclose_geometric_preload_initial_surface_point_object_by_anchor: dict[
        int, tuple[float, float, float]
    ] = {}
    post_qclose_geometric_preload_current_surface_point_world_by_anchor: dict[
        int, tuple[float, float, float]
    ] = {}
    kinematic_consistency_actual_module_pose_world_by_id: dict[int, Pose7D] = {}
    kinematic_consistency_predicted_module_pose_world_by_id: dict[int, Pose7D] = {}
    kinematic_consistency_module_position_error_m_by_id: dict[int, float] = {}
    kinematic_consistency_module_attitude_error_rad_by_id: dict[int, float] = {}
    kinematic_consistency_max_module_position_error_m = 0.0
    kinematic_consistency_max_module_attitude_error_rad = 0.0
    kinematic_consistency_anchor_position_error_m_by_anchor: dict[int, float] = {}
    kinematic_consistency_anchor_attitude_error_rad_by_anchor: dict[int, float] = {}
    kinematic_consistency_max_anchor_position_error_m = 0.0
    kinematic_consistency_max_anchor_attitude_error_rad = 0.0
    kinematic_consistency_predicted_surface_point_world_by_anchor: dict[
        int, tuple[float, float, float]
    ] = {}
    kinematic_consistency_surface_point_error_m_by_anchor: dict[int, float] = {}
    kinematic_consistency_max_surface_point_error_m = 0.0
    post_qclose_geometric_preload_tracking_error_m = 0.0
    post_qclose_geometric_preload_achieved_inward_displacement_m_by_anchor = {
        int(selection.anchor_id): 0.0 for selection in selections
    }
    post_qclose_geometric_preload_terminal_error_m = 0.0
    post_qclose_geometric_preload_settle_dwell_s = float(
        config.contact_stall_dwell_s
    )
    post_qclose_geometric_preload_active = False
    post_qclose_geometric_preload_active_step_count = 0
    post_qclose_geometric_preload_measured_position_reference_step_count = 0
    contact_closure_measured_position_reference_step_count = 0
    post_qclose_geometric_preload_load_arrest_candidates = {
        int(selection.anchor_id): False for selection in selections
    }
    post_qclose_geometric_preload_completion_source: str | None = (
        "not_applicable_superseded_by_load_limited_position_preload_v2"
    )
    contact_closure_reason: str | None = None
    qclose_base_pose_snapshot: Pose7D | None = None
    qclose_joint_positions_snapshot: dict[str, float] = {}
    qclose_object_pose_snapshot: Pose7D | None = None
    qclose_checkpoint_state_snapshot: dict[str, object] | None = None
    contact_centering_target_base_pose: Pose7D | None = None
    contact_centering_offset_world = (0.0, 0.0, 0.0)
    contact_closure_common_translation_world = (0.0, 0.0, 0.0)
    contact_closure_common_translation_active_step_count = 0
    max_contact_closure_common_translation_m = 0.0
    simple_closure_velocity_targets_radps: dict[str, float] = {}
    simple_closure_open_joint_positions_rad: dict[str, float] = {}
    simple_closure_position_targets_rad: dict[str, float] = {}
    simple_release_position_targets_rad: dict[str, float] = {}
    simple_closure_initialized_time_s: float | None = None
    simple_closure_active_step_count = 0
    simple_release_active_step_count = 0
    contact_position_preload_complete = bool(
        diagnostic_force_fixture or diagnostic_qclose_fixture
    )
    contact_position_preload_position_targets_rad: dict[str, float] = {}
    contact_position_preload_velocity_targets_radps: dict[str, float] = {}
    contact_position_preload_joint_ids_by_anchor: dict[int, tuple[str, ...]] = {}
    contact_position_preload_load_nm_by_anchor = {
        int(selection.anchor_id): 0.0 for selection in selections
    }
    contact_position_preload_max_load_nm_by_anchor = {
        int(selection.anchor_id): 0.0 for selection in selections
    }
    contact_position_preload_load_dwell_s_by_anchor = {
        int(selection.anchor_id): 0.0 for selection in selections
    }
    contact_position_preload_frozen_anchor_ids: set[int] = (
        set(selected_anchor_ids) if contact_position_preload_complete else set()
    )
    contact_position_preload_frozen_time_s_by_anchor: dict[int, float] = {}
    contact_position_preload_active_step_count = 0
    contact_position_preload_completion_source: str | None = (
        "diagnostic_fixture_bypass_without_closure_direction"
        if contact_position_preload_complete
        else None
    )
    committed_contact_centering_offset_world = (0.0, 0.0, 0.0)
    latched_contact_centering_offset_world: tuple[float, float, float] | None = None
    contact_centering_hold_anchor_poses_base: dict[int, Pose7D] | None = None
    contact_centering_unlatched_anchor_id: int | None = None
    contact_backed_off_anchor_hold_poses_world: dict[int, Pose7D] = {}
    contact_pursued_anchor_id: int | None = None
    contact_sequential_transfer_origin_base_pose: Pose7D | None = None
    contact_sequential_transfer_limit_m: float | None = None
    contact_individual_latch_hold_base_pose: Pose7D | None = None
    contact_centering_settle_dwell_s = 0.0
    contact_centering_cycle_count = 0
    max_contact_centering_offset_m = 0.0
    max_contact_centering_tilt_rad = 0.0
    max_contact_centering_measured_tilt_rad = 0.0
    contact_centering_active = False
    contact_centering_active_step_count = 0
    contact_continuous_balance_active_step_count = 0
    contact_sequential_reacquire_active_step_count = 0
    contact_sequential_centroidal_nudge_active_step_count = 0
    contact_sequential_latched_transfer_active_step_count = 0
    contact_sequential_joint_position_hold_step_count = 0
    contact_object_follow_active_step_count = 0
    contact_provisional_surface_settle_active_step_count = 0
    contact_motion_safety_interlock_blocked_step_count = 0
    max_contact_object_translation_m = 0.0
    max_contact_base_retarget_translation_m = 0.0
    contact_axial_gain_scheduled_step_count = 0
    contact_clearance_sync_active_step_count = 0
    post_first_arrest_creep_active_step_count = 0
    post_first_arrest_centroidal_transfer_active_step_count = 0
    max_post_first_arrest_centroidal_transfer_m = 0.0
    max_contact_clearance_imbalance_m = 0.0
    contact_force_scale = 0.0
    contact_force_scale_by_anchor = {
        int(selection.anchor_id): 0.0 for selection in selections
    }
    max_contact_force_scale_by_anchor = {
        int(selection.anchor_id): 0.0 for selection in selections
    }
    last_contact_stall_command_error_m_by_anchor = {
        int(selection.anchor_id): 0.0 for selection in selections
    }
    last_contact_stall_anchor_speed_mps_by_anchor = {
        int(selection.anchor_id): 0.0 for selection in selections
    }
    last_anchor_object_relative_speed_mps_by_anchor = {
        int(selection.anchor_id): 0.0 for selection in selections
    }
    max_anchor_object_relative_speed_mps_by_anchor = {
        int(selection.anchor_id): 0.0 for selection in selections
    }
    last_anchor_object_normal_relative_speed_mps_by_anchor = {
        int(selection.anchor_id): 0.0 for selection in selections
    }
    max_anchor_object_normal_relative_speed_mps_by_anchor = {
        int(selection.anchor_id): 0.0 for selection in selections
    }
    filtered_anchor_object_normal_relative_velocity_mps_by_anchor = {
        int(selection.anchor_id): 0.0 for selection in selections
    }
    last_anchor_object_filtered_normal_relative_speed_mps_by_anchor = {
        int(selection.anchor_id): 0.0 for selection in selections
    }
    max_anchor_object_filtered_normal_relative_speed_mps_by_anchor = {
        int(selection.anchor_id): 0.0 for selection in selections
    }
    previous_gripper_surface_clearance_m_by_anchor: dict[int, float] = {}
    filtered_gripper_surface_clearance_rate_mps_by_anchor = {
        int(selection.anchor_id): 0.0 for selection in selections
    }
    last_gripper_surface_clearance_rate_mps_by_anchor = {
        int(selection.anchor_id): 0.0 for selection in selections
    }
    last_filtered_gripper_surface_clearance_rate_mps_by_anchor = {
        int(selection.anchor_id): 0.0 for selection in selections
    }
    max_filtered_gripper_surface_clearance_rate_mps_by_anchor = {
        int(selection.anchor_id): 0.0 for selection in selections
    }
    last_contact_stall_selected_joint_load_nm_by_anchor = {
        int(selection.anchor_id): 0.0 for selection in selections
    }
    pregrasp_achieved_mesh_clearance_m = 0.0
    pregrasp_reachability_gate_passed = False
    pregrasp_reachability_gate_source = "not_evaluated"
    diagnostic_near_contact_initial_surface_clearance_m_by_anchor: dict[
        int, float
    ] = {}
    diagnostic_near_contact_warmup_complete = (
        not diagnostic_near_contact_fixture
    )
    diagnostic_near_contact_warmup_completed_time_s: float | None = None
    diagnostic_near_contact_estimator_reset_count = 0
    if diagnostic_force_fixture or diagnostic_qclose_fixture:
        # Fault-isolation fixture: bypass takeoff, staging, axial insertion,
        # contact-arrest inference, and simultaneous reacquire.  The real
        # three-module articulation, authored Dock meshes, free object,
        # position drive, torque-bias channel, and contact measurement remain.
        # The neutral force fixture additionally fixes the base and object;
        # the measured q_close checkpoint leaves both dynamic.  Both states
        # are explicitly acceptance-ineligible and exist only to expose the
        # force-path mechanics in a few simulated seconds.
        fixture_joint_vector = full_joint_vector()
        fixture_base_pose, _fixture_base_twist = _module_frame_pose_twist(
            robots[morphology_graph.base_module_id],
            module_frame_link_id=module_frame_link_id,
        )
        last_kinematics = whole_structure_kinematics.compute(
            morphology_graph,
            physical_model,
            _global_dock_position_map(fixture_joint_vector),
            fixture_base_pose,
            anchor_references,
        )
        fixture_anchor_poses_base = {
            anchor_id: compose_pose(
                inverse_pose(fixture_base_pose),
                anchor_pose,
            )
            for anchor_id, anchor_pose in (last_kinematics.anchor_poses_world.items())
        }
        if (
            diagnostic_qclose_checkpoint_state is not None
            and not diagnostic_qclose_zero_velocities
        ):
            fixture_anchor_poses_base = dict(
                diagnostic_qclose_checkpoint_state.anchor_hold_poses_base
            )
        fixture_object_state = _object_state(object_asset)
        fixture_gripper_body_poses = _selected_gripper_body_poses(
            selected_gripper_local_aabbs,
            robots,
        )
        fixture_surface_clearances = {
            anchor_id: _gripper_object_surface_sample_query_from_body_poses(
                bounds,
                fixture_gripper_body_poses,
                fixture_object_state["pose"],
                runtime_object_size_m,
            )[0]
            for anchor_id, bounds in (selected_gripper_local_aabbs_by_anchor.items())
        }
        pregrasp_open_anchor_poses_base = dict(fixture_anchor_poses_base)
        contact_axial_aligned = True
        contact_axial_overlap_at_latch_m = (
            _minimum_gripper_object_axial_overlap_from_body_poses(
                selected_gripper_local_aabbs,
                fixture_gripper_body_poses,
                fixture_object_state["pose"],
                runtime_object_size_m,
                axis_world=staging_plan.approach_axis_world,
            )
        )
        contact_axial_hold_base_pose = fixture_base_pose
        contact_axial_settle_dwell_s = 0.0
        contact_side_closure_enabled = False
        contact_mesh_precenter_complete = True
        contact_mesh_precenter_dwell_s = float(config.contact_stall_dwell_s)
        contact_mesh_precenter_completed_time_s = current_time_s
        contact_stall_latched_anchor_poses_base = dict(fixture_anchor_poses_base)
        contact_stall_latched_anchor_poses_world = dict(
            last_kinematics.anchor_poses_world
        )
        contact_stall_latched_mesh_clearance_m_by_anchor = dict(
            fixture_surface_clearances
        )
        contact_stall_latched = True
        contact_configuration_latched = True
        contact_configuration_latched_time_s = current_time_s
        contact_closure_reason = (
            "diagnostic_force_fixture_preloaded_q_close"
            if diagnostic_force_fixture
            else "diagnostic_measured_qclose_checkpoint_preloaded"
        )
        qclose_base_pose_snapshot = fixture_base_pose
        qclose_joint_positions_snapshot = {
            joint_id: float(position)
            for joint_id, position in zip(
                fixture_joint_vector.joint_ids,
                fixture_joint_vector.positions_rad,
                strict=True,
            )
        }
        qclose_object_pose_snapshot = tuple(fixture_object_state["pose"])
        if diagnostic_qclose_checkpoint_state is not None:
            qclose_checkpoint_state_snapshot = _qclose_checkpoint_state_to_dict(
                diagnostic_qclose_checkpoint_state
            )
        # Exact q_close replay uses the same production contract: restore and
        # hold the measured geometry, then exercise only the independent
        # offset-torque path.  Re-solving a new geometric target would no
        # longer be an exact continuation of the captured state.
        grasp_hold_anchor_poses_base = dict(fixture_anchor_poses_base)
        commanded_anchor_targets_base = dict(fixture_anchor_poses_base)
        contact_individual_latch_hold_base_pose = fixture_base_pose
        contact_centering_target_base_pose = fixture_base_pose
        latched_contact_centering_offset_world = (0.0, 0.0, 0.0)
        contact_yield_requested = True
        contact_yield_triggered_time_s = current_time_s
        contact_yield_blend = 1.0
        contact_yield_trigger_anchor_ids.update(selected_anchor_ids)
        contact_admittance_requested = True
        contact_admittance_triggered_time_s = current_time_s
        contact_admittance_trigger_anchor_ids.update(selected_anchor_ids)
        contact_yield_joint_drive_requested = False
        contact_yield_joint_drive_triggered_time_s = None
        contact_yield_joint_drive_blend = 0.0
        joint_position_reference_by_id = {
            joint_id: float(position)
            for joint_id, position in zip(
                fixture_joint_vector.joint_ids,
                fixture_joint_vector.positions_rad,
                strict=True,
            )
        }
        commanded_base_target = fixture_base_pose
        fixture_body_targets = {
            phase: fixture_base_pose for phase in Order8NaturalContactPhase
        }
        fixture_feedback = NaturalContactPlannerFeedback(
            time_s=current_time_s,
            hover_ready=True,
            simultaneous_reachability_passed=True,
            pregrasp_aligned=False,
            contact_command_dwell_complete=False,
            lift_clearance_reached=False,
            transport_distance_reached=False,
            intended_place_pose_reached=False,
            release_command_dwell_complete=False,
            retreat_clearance_reached=False,
            post_release_settle_complete=False,
            desired_body_pose_by_phase=fixture_body_targets,
            desired_anchor_pose_by_id=last_kinematics.anchor_poses_world,
            contact_force_scale=0.0,
            contact_force_scale_by_anchor_id={
                anchor_id: 0.0 for anchor_id in selected_anchor_ids
            },
            anchor_pose_priority_by_id={
                anchor_id: 1.0 for anchor_id in selected_anchor_ids
            },
        )
        planner.observe(fixture_feedback)
        planner.observe(replace(fixture_feedback, pregrasp_aligned=True))
        phase_trace.extend(
            [
                Order8NaturalContactPhase.APPROACH.value,
                Order8NaturalContactPhase.CONTACT_ACQUISITION.value,
            ]
        )
        planner_transitions.extend(
            {
                "from_phase": transition.from_phase.value,
                "to_phase": transition.to_phase.value,
                "time_s": transition.time_s,
                "reason": transition.reason,
            }
            for transition in planner.transitions
        )
    elif diagnostic_near_contact_fixture:
        # Fast fault-isolation fixture: restore a measured collision-free
        # state from the open-to-near-contact part of ordinary acquisition.
        # Module roots are reconstructed from graph FK and all velocities are
        # zero, so the fixture contains neither saved constraint strain nor a
        # contact impulse.  It resumes the same fixed joint-velocity q_close
        # path and can never satisfy acceptance.
        fixture_joint_vector = full_joint_vector()
        fixture_base_pose, _fixture_base_twist = _module_frame_pose_twist(
            robots[morphology_graph.base_module_id],
            module_frame_link_id=module_frame_link_id,
        )
        last_kinematics = whole_structure_kinematics.compute(
            morphology_graph,
            physical_model,
            _global_dock_position_map(fixture_joint_vector),
            fixture_base_pose,
            anchor_references,
        )
        fixture_anchor_poses_base = {
            anchor_id: compose_pose(inverse_pose(fixture_base_pose), anchor_pose)
            for anchor_id, anchor_pose in last_kinematics.anchor_poses_world.items()
        }
        fixture_object_state = _object_state(object_asset)
        fixture_gripper_body_poses = _selected_gripper_body_poses(
            selected_gripper_local_aabbs,
            robots,
        )
        fixture_surface_clearances = {
            anchor_id: _gripper_object_surface_sample_query_from_body_poses(
                bounds,
                fixture_gripper_body_poses,
                fixture_object_state["pose"],
                runtime_object_size_m,
            )[0]
            for anchor_id, bounds in selected_gripper_local_aabbs_by_anchor.items()
        }
        diagnostic_near_contact_initial_surface_clearance_m_by_anchor = dict(
            fixture_surface_clearances
        )
        if any(
            clearance <= 0.0
            for clearance in fixture_surface_clearances.values()
        ):
            raise RuntimeError(
                "Order8 reconstructed near-contact fixture is not collision-free"
            )
        diagnostic_open_fixture_clearance_limit_m = (
            float(config.pregrasp_mesh_clearance_m)
            + float(config.contact_penetration_noise_floor_m)
        )
        if (
            max(fixture_surface_clearances.values())
            > diagnostic_open_fixture_clearance_limit_m
        ):
            raise RuntimeError(
                "Order8 reconstructed near-contact fixture is outside the "
                "diagnostic open-to-contact region"
            )
        pregrasp_open_anchor_poses_base = dict(fixture_anchor_poses_base)
        pregrasp_achieved_mesh_clearance_m = min(fixture_surface_clearances.values())
        pregrasp_reachability_gate_passed = True
        pregrasp_reachability_gate_source = (
            "acceptance_ineligible_measured_collision_free_open_contact_fixture_v2"
        )
        contact_axial_aligned = True
        contact_axial_overlap_at_latch_m = (
            _minimum_gripper_object_axial_overlap_from_body_poses(
                selected_gripper_local_aabbs,
                fixture_gripper_body_poses,
                fixture_object_state["pose"],
                runtime_object_size_m,
                axis_world=staging_plan.approach_axis_world,
            )
        )
        contact_axial_hold_base_pose = grasp_base_pose
        contact_axial_settle_dwell_s = float(config.contact_stall_dwell_s)
        # The restored fixture is close to the object, but it remains
        # collision-free.  Let graph constraints, nominal Dock drives, and the
        # external-wrench estimator settle before beginning closure so a
        # checkpoint-restore impulse cannot masquerade as first contact.
        contact_side_closure_enabled = False
        contact_mesh_precenter_complete = True
        contact_mesh_precenter_dwell_s = float(config.contact_stall_dwell_s)
        contact_mesh_precenter_completed_time_s = current_time_s
        commanded_anchor_targets_base = dict(fixture_anchor_poses_base)
        joint_position_reference_by_id = {
            joint_id: float(position)
            for joint_id, position in zip(
                fixture_joint_vector.joint_ids,
                fixture_joint_vector.positions_rad,
                strict=True,
            )
        }
        commanded_base_target = fixture_base_pose
        # This fixture is near contact but explicitly collision-free.  Do not
        # inherit the historical near-surface-only centroidal yield from its
        # source checkpoint: the production controller now keeps pose P/I
        # active until a non-privileged proximity-plus-load event is observed.
        # Admittance likewise remains off until that same physical signature.
        contact_yield_requested = False
        contact_yield_triggered_time_s = None
        contact_yield_blend = 0.0
        contact_yield_joint_drive_requested = False
        contact_yield_joint_drive_triggered_time_s = None
        contact_yield_joint_drive_blend = 0.0
        fixture_body_targets = {
            phase: fixture_base_pose for phase in Order8NaturalContactPhase
        }
        fixture_feedback = NaturalContactPlannerFeedback(
            time_s=current_time_s,
            hover_ready=True,
            simultaneous_reachability_passed=True,
            pregrasp_aligned=False,
            contact_command_dwell_complete=False,
            lift_clearance_reached=False,
            transport_distance_reached=False,
            intended_place_pose_reached=False,
            release_command_dwell_complete=False,
            retreat_clearance_reached=False,
            post_release_settle_complete=False,
            desired_body_pose_by_phase=fixture_body_targets,
            desired_anchor_pose_by_id=last_kinematics.anchor_poses_world,
            contact_force_scale=0.0,
            contact_force_scale_by_anchor_id={
                anchor_id: 0.0 for anchor_id in selected_anchor_ids
            },
            anchor_pose_priority_by_id={
                anchor_id: 1.0 for anchor_id in selected_anchor_ids
            },
        )
        planner.observe(fixture_feedback)
        planner.observe(replace(fixture_feedback, pregrasp_aligned=True))
        phase_trace.extend(
            [
                Order8NaturalContactPhase.APPROACH.value,
                Order8NaturalContactPhase.CONTACT_ACQUISITION.value,
            ]
        )
        planner_transitions.extend(
            {
                "from_phase": transition.from_phase.value,
                "to_phase": transition.to_phase.value,
                "time_s": transition.time_s,
                "reason": transition.reason,
            }
            for transition in planner.transitions
        )
    elif diagnostic_precontact_fixture:
        # Reuse a measured, collision-free post-axial pose from an earlier
        # full-sequence diagnostic, then execute the current final-base-settle,
        # simultaneous surface-region closure, and force paths.  The object
        # remains free and q_close is deliberately not preloaded.  In
        # particular, this fixture must not bypass the v3 base-settle gate.
        fixture_joint_vector = full_joint_vector()
        fixture_base_pose, _fixture_base_twist = _module_frame_pose_twist(
            robots[morphology_graph.base_module_id],
            module_frame_link_id=module_frame_link_id,
        )
        last_kinematics = whole_structure_kinematics.compute(
            morphology_graph,
            physical_model,
            _global_dock_position_map(fixture_joint_vector),
            fixture_base_pose,
            anchor_references,
        )
        fixture_anchor_poses_base = {
            anchor_id: compose_pose(inverse_pose(fixture_base_pose), anchor_pose)
            for anchor_id, anchor_pose in last_kinematics.anchor_poses_world.items()
        }
        fixture_object_state = _object_state(object_asset)
        fixture_gripper_body_poses = _selected_gripper_body_poses(
            selected_gripper_local_aabbs,
            robots,
        )
        fixture_surface_clearances = {
            anchor_id: _gripper_object_surface_sample_query_from_body_poses(
                bounds,
                fixture_gripper_body_poses,
                fixture_object_state["pose"],
                runtime_object_size_m,
            )[0]
            for anchor_id, bounds in selected_gripper_local_aabbs_by_anchor.items()
        }
        pregrasp_open_anchor_poses_base = dict(fixture_anchor_poses_base)
        pregrasp_achieved_mesh_clearance_m = min(fixture_surface_clearances.values())
        pregrasp_reachability_gate_passed = True
        pregrasp_reachability_gate_source = (
            "acceptance_ineligible_measured_post_axial_pose_fixture_v1"
        )
        contact_axial_aligned = True
        contact_axial_overlap_at_latch_m = (
            _minimum_gripper_object_axial_overlap_from_body_poses(
                selected_gripper_local_aabbs,
                fixture_gripper_body_poses,
                fixture_object_state["pose"],
                runtime_object_size_m,
                axis_world=staging_plan.approach_axis_world,
            )
        )
        contact_axial_hold_base_pose = fixture_base_pose
        contact_axial_settle_dwell_s = 0.0
        contact_side_closure_enabled = False
        commanded_anchor_targets_base = dict(fixture_anchor_poses_base)
        joint_position_reference_by_id = {
            joint_id: float(position)
            for joint_id, position in zip(
                fixture_joint_vector.joint_ids,
                fixture_joint_vector.positions_rad,
                strict=True,
            )
        }
        commanded_base_target = fixture_base_pose
        fixture_body_targets = {
            phase: fixture_base_pose for phase in Order8NaturalContactPhase
        }
        fixture_feedback = NaturalContactPlannerFeedback(
            time_s=current_time_s,
            hover_ready=True,
            simultaneous_reachability_passed=True,
            pregrasp_aligned=False,
            contact_command_dwell_complete=False,
            lift_clearance_reached=False,
            transport_distance_reached=False,
            intended_place_pose_reached=False,
            release_command_dwell_complete=False,
            retreat_clearance_reached=False,
            post_release_settle_complete=False,
            desired_body_pose_by_phase=fixture_body_targets,
            desired_anchor_pose_by_id=last_kinematics.anchor_poses_world,
            contact_force_scale=0.0,
            contact_force_scale_by_anchor_id={
                anchor_id: 0.0 for anchor_id in selected_anchor_ids
            },
            anchor_pose_priority_by_id={
                anchor_id: 1.0 for anchor_id in selected_anchor_ids
            },
        )
        planner.observe(fixture_feedback)
        planner.observe(replace(fixture_feedback, pregrasp_aligned=True))
        phase_trace.extend(
            [
                Order8NaturalContactPhase.APPROACH.value,
                Order8NaturalContactPhase.CONTACT_ACQUISITION.value,
            ]
        )
        planner_transitions.extend(
            {
                "from_phase": transition.from_phase.value,
                "to_phase": transition.to_phase.value,
                "time_s": transition.time_s,
                "reason": transition.reason,
            }
            for transition in planner.transitions
        )
    last_print_s = -math.inf
    previous_phase = planner.phase
    # The exact q_close diagnostic starts with contact already latched, so its
    # first sample can enter slip telemetry before the normal per-step phase
    # target is calculated below.  Seed the nominal target from the command
    # initialized by the selected fixture (or the ordinary initial pose).
    nominal_base_target = commanded_base_target
    # A preloaded diagnostic fixture may satisfy the positional-preload gate
    # on its first loop iteration, before the ordinary per-step phase maps are
    # rebuilt below.  Seed both maps with a stationary command so that this
    # short-path transition has the same well-defined target as the longer
    # rollout.
    desired_body_pose_by_phase = {
        phase: commanded_base_target for phase in Order8NaturalContactPhase
    }
    desired_body_linear_velocity_by_phase = {
        phase: (0.0, 0.0, 0.0) for phase in Order8NaturalContactPhase
    }
    runtime_profiler = None
    if diagnostic_profile_output is not None:
        import cProfile

        runtime_profiler = cProfile.Profile()
        runtime_profiler.enable()
    for _ in range(max_steps):
        phase = planner.phase
        object_state = _object_state(object_asset)
        live_grasp_base_pose = _pose_following_object_motion(
            reference_object_pose,
            tuple(object_state["pose"]),
            grasp_base_pose,
        )
        live_contact_anchor_targets_world = {
            anchor_id: _pose_following_object_motion(
                reference_object_pose,
                tuple(object_state["pose"]),
                target_pose,
            )
            for anchor_id, target_pose in contact_anchor_targets_world.items()
        }
        if (
            phase == Order8NaturalContactPhase.CONTACT_ACQUISITION
            and not contact_configuration_latched
        ):
            contact_object_follow_active_step_count += 1
            max_contact_object_translation_m = max(
                max_contact_object_translation_m,
                _position_distance(reference_object_pose, tuple(object_state["pose"])),
            )
            max_contact_base_retarget_translation_m = max(
                max_contact_base_retarget_translation_m,
                _position_distance(grasp_base_pose, live_grasp_base_pose),
            )
        contact_measurement, contact_vector_telemetry = _measure_robot_object_contacts(
            contact_view,
            sim_dt=sim_dt,
            sensor_body_paths=rigid_body_paths,
            body_identity=body_identity,
            body_lookup=body_lookup,
            robots=robots,
            object_state=object_state,
            selected_link_ids=set(selected_link_ids),
            wp=wp,
            torch=torch,
        )
        contact_vector_telemetry_invalid_step_count += int(
            not contact_vector_telemetry.valid
        )
        last_selected_contact_normal_force_world_by_link = dict(
            contact_vector_telemetry.normal_force_world_by_link
        )
        last_selected_contact_application_point_world_by_link = dict(
            contact_vector_telemetry.normal_force_application_point_world_by_link
        )
        selected_contact_point_object_m_by_link = {
            link_id: tuple(
                float(value)
                for value in compose_pose(
                    inverse_pose(tuple(object_state["pose"])),
                    (
                        *last_selected_contact_application_point_world_by_link[
                            link_id
                        ],
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ),
                )[:3]
            )
            for link_id in contact_measurement.selected_contact_link_ids
            if link_id in last_selected_contact_application_point_world_by_link
        }
        last_selected_friction_force_world_by_link = dict(
            contact_vector_telemetry.friction_force_world_by_link
        )
        last_selected_contact_force_matrix_world_by_link = dict(
            contact_vector_telemetry.contact_force_matrix_world_by_link
        )
        last_selected_body_linear_velocity_world_by_link = dict(
            contact_vector_telemetry.body_linear_velocity_world_by_link
        )
        last_selected_body_contact_velocity_world_by_link = dict(
            contact_vector_telemetry.body_contact_velocity_world_by_link
        )
        last_selected_object_contact_velocity_world_by_link = dict(
            contact_vector_telemetry.object_contact_velocity_world_by_link
        )
        last_selected_relative_contact_velocity_world_by_link = dict(
            contact_vector_telemetry.relative_contact_velocity_world_by_link
        )
        last_selected_tangential_slip_velocity_world_by_link = dict(
            contact_vector_telemetry.tangential_slip_velocity_world_by_link
        )
        last_selected_tangential_slip_velocity_object_by_link = {
            link_id: _vector_world_to_pose_local(
                object_state["pose"],
                last_selected_tangential_slip_velocity_world_by_link[link_id],
            )
            for link_id in selected_link_ids
        }
        last_selected_slip_contact_point_world_by_link = dict(
            contact_vector_telemetry.tangential_slip_contact_point_world_by_link
        )
        last_selected_slip_contact_normal_world_by_link = dict(
            contact_vector_telemetry.tangential_slip_contact_normal_world_by_link
        )
        max_selected_friction_force_magnitude_n_by_link = {
            link_id: max(
                max_selected_friction_force_magnitude_n_by_link[link_id],
                _norm(last_selected_friction_force_world_by_link[link_id]),
            )
            for link_id in selected_link_ids
        }
        max_abs_selected_friction_vertical_force_n_by_link = {
            link_id: max(
                max_abs_selected_friction_vertical_force_n_by_link[link_id],
                abs(last_selected_friction_force_world_by_link[link_id][2]),
            )
            for link_id in selected_link_ids
        }
        last_selected_normal_force_n_by_link = {
            link_id: sum(
                float(patch.normal_force_n)
                for patch in contact_measurement.raw_contact_patches
                if patch.robot_link_id == link_id
            )
            for link_id in selected_link_ids
        }
        max_selected_normal_force_n_by_link = {
            link_id: max(
                max_selected_normal_force_n_by_link[link_id],
                last_selected_normal_force_n_by_link[link_id],
            )
            for link_id in selected_link_ids
        }
        floor_contact = _contact_view_active(
            object_floor_view,
            sim_dt=sim_dt,
            wp=wp,
            torch=torch,
            force_threshold_n=float(config.contact_normal_force_threshold_n),
        )
        robot_environment_contact = _contact_view_active(
            robot_environment_contact_view,
            sim_dt=sim_dt,
            wp=wp,
            torch=torch,
            force_threshold_n=float(config.contact_normal_force_threshold_n),
        )
        robot_environment_contact_step_count += int(robot_environment_contact)
        robot_environment_unsafe_contact = bool(
            robot_environment_contact
            and phase
            in {
                Order8NaturalContactPhase.CONTACT_ACQUISITION,
                Order8NaturalContactPhase.LIFT,
                Order8NaturalContactPhase.TRANSPORT,
                Order8NaturalContactPhase.PLACE,
                Order8NaturalContactPhase.RELEASE,
                Order8NaturalContactPhase.RETREAT,
                Order8NaturalContactPhase.SETTLE,
            }
        )
        if robot_environment_unsafe_contact:
            robot_environment_unsafe_contact_step_count += 1
            if robot_environment_first_unsafe_contact_time_s is None:
                robot_environment_first_unsafe_contact_time_s = current_time_s
            request_safe_hold_or_record(
                time_s=current_time_s,
                reason="unintended_robot_floor_or_object_support_contact",
            )
        raw_invalid_count += int(not contact_measurement.raw_contact_valid)
        raw_saturation_count += int(contact_measurement.raw_contact_saturated)
        for reason in contact_measurement.failure_reasons:
            if reason not in raw_contact_failure_reasons:
                raw_contact_failure_reasons.append(reason)
        from amsrr.simulation.order8_contact_measurement import (
            object_bottom_clearance_m,
        )

        object_bottom_clearance = object_bottom_clearance_m(
            object_pose_world=object_state["pose"],
            object_size_m=runtime_object_size_m,
            floor_height_m=object_support_height_m,
        )
        if (
            first_transport_object_pose is None
            and phase == Order8NaturalContactPhase.TRANSPORT
        ):
            first_transport_object_pose = tuple(object_state["pose"])
        transport_distance = (
            0.0
            if first_transport_object_pose is None
            else _position_distance(object_state["pose"], first_transport_object_pose)
        )
        selected_gripper_body_poses = _selected_gripper_body_poses(
            selected_gripper_local_aabbs,
            robots,
        )
        measured_selected_anchor_poses_world_by_anchor = {
            int(reference.anchor.anchor_id): compose_pose(
                selected_gripper_body_poses[
                    (
                        reference.surface.module_id,
                        reference.surface.mechanism_link_id,
                    )
                ],
                reference.anchor.local_pose,
            )
            for reference in anchor_references
        }
        gripper_clearance = _gripper_object_clearance_from_body_poses(
            selected_gripper_local_aabbs,
            selected_gripper_body_poses,
            object_state["pose"],
            runtime_object_size_m,
        )
        gripper_clearance_m_by_anchor = {
            anchor_id: _gripper_object_clearance_from_body_poses(
                bounds,
                selected_gripper_body_poses,
                object_state["pose"],
                runtime_object_size_m,
            )
            for anchor_id, bounds in (selected_gripper_local_aabbs_by_anchor.items())
        }
        gripper_surface_query_by_anchor = {
            anchor_id: _gripper_object_surface_sample_query_from_body_poses(
                bounds,
                selected_gripper_body_poses,
                object_state["pose"],
                runtime_object_size_m,
            )
            for anchor_id, bounds in (selected_gripper_local_aabbs_by_anchor.items())
        }
        gripper_surface_clearance_m_by_anchor = {
            anchor_id: query[0]
            for anchor_id, query in gripper_surface_query_by_anchor.items()
        }
        current_gripper_surface_clearance_rate_mps_by_anchor = {
            anchor_id: (
                0.0
                if anchor_id not in previous_gripper_surface_clearance_m_by_anchor
                else (
                    gripper_surface_clearance_m_by_anchor[anchor_id]
                    - previous_gripper_surface_clearance_m_by_anchor[anchor_id]
                )
                / sim_dt
            )
            for anchor_id in selected_anchor_ids
        }
        previous_gripper_surface_clearance_m_by_anchor = dict(
            gripper_surface_clearance_m_by_anchor
        )
        filtered_gripper_surface_clearance_rate_mps_by_anchor = {
            anchor_id: _first_order_low_pass(
                filtered_gripper_surface_clearance_rate_mps_by_anchor[anchor_id],
                current_gripper_surface_clearance_rate_mps_by_anchor[anchor_id],
                dt_s=sim_dt,
                time_constant_s=float(config.contact_stall_dwell_s),
            )
            for anchor_id in selected_anchor_ids
        }
        current_filtered_gripper_surface_clearance_speed_mps_by_anchor = {
            anchor_id: abs(
                filtered_gripper_surface_clearance_rate_mps_by_anchor[anchor_id]
            )
            for anchor_id in selected_anchor_ids
        }
        gripper_surface_application_point_world_by_anchor = {
            anchor_id: query[1]
            for anchor_id, query in gripper_surface_query_by_anchor.items()
        }
        # The full-mesh query remains diagnostic/safety evidence only.  The
        # nominal closure command below is joint-space constant velocity and
        # never tracks this changing closest-point sample.
        contact_control_surface_point_world_by_anchor = dict(
            gripper_surface_application_point_world_by_anchor
        )
        post_qclose_geometric_preload_current_surface_point_world_by_anchor = (
            {
                anchor_id: compose_pose(
                    selected_gripper_body_poses[
                        selected_gripper_body_key_by_anchor[anchor_id]
                    ],
                    (*point_local, 0.0, 0.0, 0.0, 1.0),
                )[:3]
                for anchor_id, point_local in (
                    post_qclose_geometric_preload_surface_point_local_by_anchor.items()
                )
            }
            if post_qclose_geometric_preload_surface_point_local_by_anchor
            else dict(contact_control_surface_point_world_by_anchor)
        )
        gripper_object_surface_normal_world_by_anchor = {
            anchor_id: query[2]
            for anchor_id, query in gripper_surface_query_by_anchor.items()
        }
        selected_gripper_body_pose_twist_by_anchor = {}
        for selection in selections:
            module_text, link_id = selection.dock_link_id.split(":", 1)
            module_id = int(module_text.removeprefix("module_"))
            selected_gripper_body_pose_twist_by_anchor[int(selection.anchor_id)] = (
                _module_frame_pose_twist(
                    robots[module_id],
                    module_frame_link_id=link_id,
                )
            )
        from amsrr.simulation.order8_contact_measurement import (
            relative_point_normal_velocity_mps,
            relative_point_speed_mps,
        )

        current_anchor_object_relative_speed_mps_by_anchor = {}
        current_anchor_object_normal_relative_velocity_mps_by_anchor = {}
        current_anchor_object_normal_relative_speed_mps_by_anchor = {}
        for anchor_id in selected_anchor_ids:
            body_pose, body_twist = selected_gripper_body_pose_twist_by_anchor[
                anchor_id
            ]
            current_anchor_object_relative_speed_mps_by_anchor[anchor_id] = (
                relative_point_speed_mps(
                    body_reference_pose_world=body_pose,
                    body_twist_world=body_twist,
                    object_reference_pose_world=object_state["pose"],
                    object_twist_world=object_state["twist"],
                    point_world=(
                        contact_control_surface_point_world_by_anchor[anchor_id]
                    ),
                )
            )
            current_anchor_object_normal_relative_velocity_mps_by_anchor[anchor_id] = (
                relative_point_normal_velocity_mps(
                    body_reference_pose_world=body_pose,
                    body_twist_world=body_twist,
                    object_reference_pose_world=object_state["pose"],
                    object_twist_world=object_state["twist"],
                    point_world=(
                        contact_control_surface_point_world_by_anchor[anchor_id]
                    ),
                    surface_normal_world=(
                        gripper_object_surface_normal_world_by_anchor[anchor_id]
                    ),
                )
            )
            current_anchor_object_normal_relative_speed_mps_by_anchor[anchor_id] = abs(
                current_anchor_object_normal_relative_velocity_mps_by_anchor[anchor_id]
            )
        filtered_anchor_object_normal_relative_velocity_mps_by_anchor = {
            anchor_id: _first_order_low_pass(
                filtered_anchor_object_normal_relative_velocity_mps_by_anchor[
                    anchor_id
                ],
                current_anchor_object_normal_relative_velocity_mps_by_anchor[anchor_id],
                dt_s=sim_dt,
                time_constant_s=float(config.contact_stall_dwell_s),
            )
            for anchor_id in selected_anchor_ids
        }
        current_anchor_object_filtered_normal_relative_speed_mps_by_anchor = {
            anchor_id: abs(
                filtered_anchor_object_normal_relative_velocity_mps_by_anchor[anchor_id]
            )
            for anchor_id in selected_anchor_ids
        }
        gripper_axial_overlap_m = _minimum_gripper_object_axial_overlap_from_body_poses(
            selected_gripper_local_aabbs,
            selected_gripper_body_poses,
            object_state["pose"],
            runtime_object_size_m,
            axis_world=staging_plan.approach_axis_world,
        )
        elapsed_since_qclose_s = (
            None
            if contact_configuration_latched_time_s is None
            else max(0.0, current_time_s - contact_configuration_latched_time_s)
        )
        active_torque_bias_limit_nm = _torque_bias_limit_with_peak_window(
            continuous_torque_nm=dock_continuous_torque_nm,
            peak_torque_nm=dock_peak_torque_nm,
            elapsed_since_qclose_s=elapsed_since_qclose_s,
            peak_window_s=diagnostic_peak_torque_window_s,
        )
        diagnostic_peak_torque_active = (
            active_torque_bias_limit_nm > dock_continuous_torque_nm + 1.0e-12
        )
        diagnostic_peak_torque_active_step_count += int(diagnostic_peak_torque_active)
        diagnostic_peak_torque_max_limit_nm = max(
            diagnostic_peak_torque_max_limit_nm,
            active_torque_bias_limit_nm,
        )
        joint_vector = full_joint_vector(
            torque_bias_limit_nm=active_torque_bias_limit_nm
        )
        if diagnostic_pitch_hold_positions_rad:
            measured_positions_by_id = {
                str(joint_id): float(position)
                for joint_id, position in zip(
                    joint_vector.joint_ids,
                    joint_vector.positions_rad,
                    strict=True,
                )
            }
            max_diagnostic_pitch_hold_error_rad = max(
                max_diagnostic_pitch_hold_error_rad,
                max(
                    abs(
                        measured_positions_by_id[joint_id]
                        - fixed_position
                    )
                    for joint_id, fixed_position in (
                        diagnostic_pitch_hold_positions_rad.items()
                    )
                ),
            )
        if (
            diagnostic_near_contact_fixture
            and not diagnostic_near_contact_warmup_complete
            and current_time_s + 1.0e-12
            >= ORDER8_NEAR_CONTACT_DIAGNOSTIC_WARMUP_S
        ):
            # Reset once after graph constraints and the free base have
            # numerically settled.  The next no-contact sample initializes the
            # estimator bias before closure/load is allowed to arm admittance.
            external_wrench_estimator.reset()
            previous_external_wrench_centroidal_model = None
            last_external_wrench_estimate = CentroidalExternalWrenchEstimate(
                valid=False,
                wrench_body=(0.0,) * 6,
                raw_wrench_body=(0.0,) * 6,
                bias_wrench_body=(0.0,) * 6,
                force_norm_n=0.0,
                torque_norm_nm=0.0,
                failure_reason="near_contact_fixture_post_settle_bias_reset",
            )
            diagnostic_near_contact_warmup_complete = True
            diagnostic_near_contact_warmup_completed_time_s = current_time_s
            diagnostic_near_contact_estimator_reset_count += 1
        external_wrench_observation = whole_structure_observation()
        current_external_wrench_centroidal_model = (
            centroidal_target_builder.build(
                morphology_graph,
                physical_model,
                external_wrench_observation,
            )
        )
        if previous_external_wrench_centroidal_model is not None:
            last_external_wrench_estimate = external_wrench_estimator.estimate(
                previous_model=previous_external_wrench_centroidal_model,
                current_model=current_external_wrench_centroidal_model,
                applied_controller_command=previous_controller_command,
                dt_s=sim_dt,
                calibrate_bias=bool(
                    not contact_admittance_requested
                    and not contact_configuration_latched
                    and not contact_side_closure_enabled
                    and phase
                    in {
                        Order8NaturalContactPhase.RESET,
                        Order8NaturalContactPhase.APPROACH,
                        Order8NaturalContactPhase.CONTACT_ACQUISITION,
                    }
                ),
            )
            contact_yield_estimator_valid_step_count += int(
                last_external_wrench_estimate.valid
            )
            contact_yield_estimator_invalid_step_count += int(
                not last_external_wrench_estimate.valid
            )
        previous_external_wrench_centroidal_model = (
            current_external_wrench_centroidal_model
        )
        selected_contact_raw_joint_torque_nm_by_anchor = {
            anchor_id: _global_joint_tensor_value(
                    robots,
                    selected_contact_joint_id_by_anchor[anchor_id],
                    field_name="applied_torque",
                )
            for anchor_id in selected_anchor_ids
        }
        selected_contact_raw_joint_load_nm_by_anchor = {
            anchor_id: abs(torque_nm)
            for anchor_id, torque_nm in (
                selected_contact_raw_joint_torque_nm_by_anchor.items()
            )
        }
        selected_contact_damping_drive_torque_nm_by_anchor = {
            anchor_id: float(
                latest_dock_actuator_telemetry.get(
                    selected_contact_joint_id_by_anchor[anchor_id],
                    {},
                ).get("estimated_damping_drive_torque_nm", 0.0)
            )
            for anchor_id in selected_anchor_ids
        }
        selected_contact_joint_load_nm_by_anchor = {
            anchor_id: _damping_compensated_joint_load_nm(
                applied_torque_nm=(
                    selected_contact_raw_joint_torque_nm_by_anchor[anchor_id]
                ),
                estimated_damping_drive_torque_nm=(
                    selected_contact_damping_drive_torque_nm_by_anchor[anchor_id]
                ),
            )
            for anchor_id in selected_anchor_ids
        }
        applied_dock_load_nm_by_joint = {
            joint_id: _damping_compensated_joint_load_nm(
                applied_torque_nm=_global_joint_tensor_value(
                    robots,
                    joint_id,
                    field_name="applied_torque",
                ),
                estimated_damping_drive_torque_nm=float(
                    latest_dock_actuator_telemetry.get(joint_id, {}).get(
                        "estimated_damping_drive_torque_nm",
                        0.0,
                    )
                ),
            )
            for joint_id in expected_joint_ids
        }
        whole_structure_dock_drive_load_nm = max(
            applied_dock_load_nm_by_joint.values(),
            default=0.0,
        )
        (
            per_anchor_influential_dock_load_nm,
            contact_stall_influential_joint_ids_by_anchor,
        ) = _per_anchor_influential_dock_loads(
            selected_anchor_ids,
            ordered_joint_ids=last_kinematics.ordered_global_dock_joint_ids,
            anchor_jacobians=last_kinematics.anchor_jacobians,
            applied_joint_load_nm=applied_dock_load_nm_by_joint,
            required_joint_id_by_anchor=selected_contact_joint_id_by_anchor,
        )
        base_root_pose, base_module_twist = _module_frame_pose_twist(
            robots[morphology_graph.base_module_id],
            module_frame_link_id=module_frame_link_id,
        )
        base_linear_speed_mps = _norm(base_module_twist[:3])
        contact_preload_ready = bool(
            phase == Order8NaturalContactPhase.CONTACT_ACQUISITION
            and contact_configuration_latched
            and post_qclose_joint_settle_complete
            and post_qclose_geometric_preload_complete
            and contact_position_preload_complete
        )
        prelift_relative_motion_settled = _contact_force_hold_settled(
            current_anchor_object_relative_speed_mps_by_anchor,
            selected_anchor_ids=selected_anchor_ids,
            speed_threshold_mps=prelift_relative_speed_threshold_mps,
        )
        diagnostic_prelift_controller_restore_ready = (
            _diagnostic_prelift_controller_restore_ready(
                enabled=diagnostic_separated_lift_transition,
                grasp_pose_rebased=contact_yield_grasp_pose_rebased,
                centroidal_yield_blend=contact_yield_blend,
                joint_drive_yield_blend=contact_yield_joint_drive_blend,
                admittance_active=contact_admittance_requested,
                base_linear_speed_mps=base_linear_speed_mps,
                base_speed_limit_mps=float(
                    config.pregrasp_linear_speed_tolerance_mps
                ),
            )
        )
        contact_command_ready = bool(
            contact_preload_ready
            and prelift_relative_motion_settled
            and diagnostic_prelift_controller_restore_ready
        )
        observation = Order8NaturalContactObservation(
            observation_version=ORDER8_NATURAL_CONTACT_OBSERVATION_VERSION,
            phase=phase,
            time_s=current_time_s,
            step_dt_s=sim_dt,
            object_id="order8_object",
            selected_dock_link_ids=list(selected_link_ids),
            raw_contact_patches=contact_measurement.raw_contact_patches,
            selected_contact_point_object_m_by_link={
                link_id: list(point)
                for link_id, point in sorted(
                    selected_contact_point_object_m_by_link.items()
                )
            },
            raw_contact_valid=bool(contact_measurement.raw_contact_valid),
            raw_contact_saturated=bool(contact_measurement.raw_contact_saturated),
            object_bottom_clearance_m=object_bottom_clearance,
            object_floor_contact=bool(floor_contact),
            object_linear_speed_mps=_norm(object_state["twist"][:3]),
            object_vertical_velocity_world_mps=float(object_state["twist"][2]),
            object_angular_speed_rad_s=_norm(object_state["twist"][3:]),
            transport_distance_m=transport_distance,
            gripper_object_clearance_m=gripper_clearance,
            controller_qp_feasible=all(
                status.qp_feasible for status in last_status.values()
            ),
            simultaneous_qclose_acquired=contact_configuration_latched,
            # In the separated diagnostic, raw contact dwell begins only after
            # measured closed-configuration rebase, ordinary-gain restore,
            # slow common-base motion, and full relative-motion settle.  The
            # production path remains unchanged until this isolation result is
            # reviewed and explicitly promoted.
            grasp_confirmation_ready=contact_command_ready,
            missing_actuator_target_count=missing_count,
            unsupported_actuator_target_count=unsupported_count,
            clipped_actuator_target_count=clipped_count,
            unresolved_actuator_target_count=unresolved_count,
        )
        last_evidence = monitor.observe(observation)
        maintained_slip_active = bool(
            last_evidence.gate_results.get(
                "maintained_contact_slip_enforcement_active",
                False,
            )
        )
        maintained_contact_links = set(last_evidence.selected_contact_link_ids)
        if diagnostic_separated_lift_transition:
            diagnostic_phase_elapsed_s = max(
                0.0, current_time_s - phase_started_s
            )
            if phase == Order8NaturalContactPhase.CONTACT_ACQUISITION:
                if not contact_yield_grasp_pose_rebased:
                    diagnostic_lift_transition_stage = "preload_before_qpid_restore"
                elif not diagnostic_prelift_controller_restore_ready:
                    diagnostic_lift_transition_stage = "qpid_restore"
                else:
                    diagnostic_lift_transition_stage = "prelift_contact_dwell"
            elif phase == Order8NaturalContactPhase.LIFT:
                if (
                    diagnostic_loaded_state_rebase_triggered_time_s is not None
                    and diagnostic_loaded_state_rebase_completed_time_s is None
                ):
                    diagnostic_lift_transition_stage = (
                        "loaded_state_rebase_settle"
                    )
                elif diagnostic_loaded_state_rebase_completed_time_s is not None:
                    diagnostic_lift_transition_stage = (
                        "main_lift_after_loaded_state_rebase"
                    )
                elif diagnostic_disable_payload_feedforward:
                    diagnostic_lift_transition_stage = (
                        "ordinary_lift_pose_only_payload_feedforward_disabled"
                    )
                elif diagnostic_phase_elapsed_s <= diagnostic_lift_bias_delay_s:
                    diagnostic_lift_transition_stage = (
                        "ordinary_lift_payload_feedforward_only"
                    )
                else:
                    diagnostic_lift_transition_stage = "extra_lift_bias_ramp"
            else:
                diagnostic_lift_transition_stage = phase.value
        else:
            diagnostic_lift_transition_stage = "disabled"
        current_selected_contact_links = set(
            contact_measurement.selected_contact_link_ids
        )
        if diagnostic_separated_lift_transition and current_selected_contact_links:
            stage_bounds = (
                diagnostic_contact_point_vertical_velocity_bounds_mps_by_stage.setdefault(
                    diagnostic_lift_transition_stage,
                    {},
                )
            )
            for link_id in sorted(
                set(selected_link_ids).intersection(current_selected_contact_links)
            ):
                dock_velocity_z = float(
                    last_selected_body_contact_velocity_world_by_link[link_id][2]
                )
                object_velocity_z = float(
                    last_selected_object_contact_velocity_world_by_link[link_id][2]
                )
                link_bounds = stage_bounds.setdefault(
                    link_id,
                    {
                        "dock_min": dock_velocity_z,
                        "dock_max": dock_velocity_z,
                        "object_min": object_velocity_z,
                        "object_max": object_velocity_z,
                    },
                )
                link_bounds["dock_min"] = min(
                    link_bounds["dock_min"], dock_velocity_z
                )
                link_bounds["dock_max"] = max(
                    link_bounds["dock_max"], dock_velocity_z
                )
                link_bounds["object_min"] = min(
                    link_bounds["object_min"], object_velocity_z
                )
                link_bounds["object_max"] = max(
                    link_bounds["object_max"], object_velocity_z
                )
        for link_id in selected_link_ids:
            if not maintained_slip_active or link_id not in maintained_contact_links:
                continue
            velocity_world = (
                last_selected_tangential_slip_velocity_world_by_link[link_id]
            )
            velocity_object = (
                last_selected_tangential_slip_velocity_object_by_link[link_id]
            )
            signed_cumulative_slip_displacement_world_m_by_link[link_id] = tuple(
                float(
                    signed_cumulative_slip_displacement_world_m_by_link[link_id][
                        axis
                    ]
                )
                + float(velocity_world[axis]) * sim_dt
                for axis in range(3)
            )
            signed_cumulative_slip_displacement_object_m_by_link[link_id] = tuple(
                float(
                    signed_cumulative_slip_displacement_object_m_by_link[link_id][
                        axis
                    ]
                )
                + float(velocity_object[axis]) * sim_dt
                for axis in range(3)
            )
            diagnostic_cumulative_slip_path_m_by_link[link_id] += (
                _norm(velocity_world) * sim_dt
            )
        if contact_configuration_latched or phase in {
            Order8NaturalContactPhase.LIFT,
            Order8NaturalContactPhase.TRANSPORT,
            Order8NaturalContactPhase.PLACE,
        }:
            slip_vector_step_telemetry.append(
                {
                    "time_s": float(current_time_s),
                    "phase": phase.value,
                    "diagnostic_lift_transition_stage": (
                        diagnostic_lift_transition_stage
                    ),
                    "diagnostic_prelift_controller_restore_ready": bool(
                        diagnostic_prelift_controller_restore_ready
                    ),
                    "base_linear_speed_mps": float(base_linear_speed_mps),
                    "base_module_twist_world": list(base_module_twist),
                    "centroidal_body_pose_world": list(
                        current_external_wrench_centroidal_model.body_pose_world
                    ),
                    "centroidal_body_twist_world": list(
                        current_external_wrench_centroidal_model.body_twist_world
                    ),
                    "commanded_base_target_world": list(commanded_base_target),
                    "nominal_base_target_world": list(nominal_base_target),
                    "maintained_contact_slip_enforcement_active": (
                        maintained_slip_active
                    ),
                    "contact_link_ids": sorted(maintained_contact_links),
                    "tangential_velocity_world_mps_by_link": {
                        link_id: list(
                            last_selected_tangential_slip_velocity_world_by_link[
                                link_id
                            ]
                        )
                        for link_id in selected_link_ids
                    },
                    "tangential_velocity_object_mps_by_link": {
                        link_id: list(
                            last_selected_tangential_slip_velocity_object_by_link[
                                link_id
                            ]
                        )
                        for link_id in selected_link_ids
                    },
                    "relative_velocity_world_mps_by_link": {
                        link_id: list(
                            last_selected_relative_contact_velocity_world_by_link[
                                link_id
                            ]
                        )
                        for link_id in selected_link_ids
                    },
                    "dock_contact_point_velocity_world_mps_by_link": {
                        link_id: list(
                            last_selected_body_contact_velocity_world_by_link[
                                link_id
                            ]
                        )
                        for link_id in selected_link_ids
                    },
                    "object_contact_point_velocity_world_mps_by_link": {
                        link_id: list(
                            last_selected_object_contact_velocity_world_by_link[
                                link_id
                            ]
                        )
                        for link_id in selected_link_ids
                    },
                    "contact_point_world_by_link": {
                        link_id: list(
                            last_selected_slip_contact_point_world_by_link[link_id]
                        )
                        for link_id in selected_link_ids
                    },
                    "contact_normal_world_by_link": {
                        link_id: list(
                            last_selected_slip_contact_normal_world_by_link[link_id]
                        )
                        for link_id in selected_link_ids
                    },
                    "signed_cumulative_displacement_world_m_by_link": {
                        link_id: list(
                            signed_cumulative_slip_displacement_world_m_by_link[
                                link_id
                            ]
                        )
                        for link_id in selected_link_ids
                    },
                    "signed_cumulative_displacement_object_m_by_link": {
                        link_id: list(
                            signed_cumulative_slip_displacement_object_m_by_link[
                                link_id
                            ]
                        )
                        for link_id in selected_link_ids
                    },
                    "diagnostic_cumulative_path_m_by_link": dict(
                        diagnostic_cumulative_slip_path_m_by_link
                    ),
                    "monitor_cumulative_path_m_by_link": dict(
                        last_evidence.contact_point_slip_displacement_m_by_link
                    ),
                    "object_pose_world": list(object_state["pose"]),
                    "object_twist_world": list(object_state["twist"]),
                    "payload_feedforward_scale": float(
                        last_payload_feedforward_scale
                    ),
                    "payload_feedforward_target_scale": float(
                        last_payload_feedforward_target_scale
                    ),
                    "payload_commanded_lift_progress_scale": float(
                        last_payload_commanded_lift_progress_scale
                    ),
                    "lift_acceleration_bias_scale": float(
                        last_lift_acceleration_bias_scale
                    ),
                    "lift_acceleration_bias_commanded_progress_scale": float(
                        last_lift_acceleration_bias_commanded_progress_scale
                    ),
                    "lift_acceleration_bias_force_world_z_n": float(
                        last_lift_acceleration_bias_force_world_z_n
                    ),
                    "lift_acceleration_residual_wrench_body": list(
                        last_lift_acceleration_residual_wrench_body
                    ),
                    "estimated_payload_load_transfer_scale": float(
                        last_estimated_payload_lift_transfer_scale
                    ),
                    "estimated_payload_load_transfer_peak_scale": float(
                        estimated_payload_lift_transfer_peak_scale
                    ),
                    "measured_payload_lift_transfer_peak_scale": float(
                        measured_payload_lift_transfer_peak_scale
                    ),
                    "lift_external_force_world_z_n": (
                        None
                        if last_lift_external_force_world_z_n is None
                        else float(last_lift_external_force_world_z_n)
                    ),
                    "object_bottom_clearance_m": float(object_bottom_clearance),
                    "lift_off_confirmed": bool(
                        payload_lift_off_confirmed_time_s is not None
                    ),
                    "loaded_state_rebase_triggered": bool(
                        diagnostic_loaded_state_rebase_triggered_time_s is not None
                    ),
                    "loaded_state_rebase_completed": bool(
                        diagnostic_loaded_state_rebase_completed_time_s is not None
                    ),
                    "loaded_state_rebase_settled_dwell_s": float(
                        diagnostic_loaded_state_rebase_settled_dwell_s
                    ),
                }
            )
        step_evidence.append(last_evidence.to_dict())
        if (
            phase
            in {
                Order8NaturalContactPhase.RETREAT,
                Order8NaturalContactPhase.SETTLE,
            }
            and last_evidence.selected_contact_exists
        ):
            post_release_selected_contact_count += 1

        nominal_base_target = _base_target_for_phase(
            phase,
            hover_base_pose=hover_base_pose,
            approach_base_pose=approach_base_pose,
            grasp_base_pose=(
                grasp_base_pose
                if contact_configuration_latched
                else live_grasp_base_pose
            ),
            lift_base_pose=lift_base_pose,
            transport_base_pose=transport_base_pose,
            place_base_pose=place_base_pose,
            retreat_base_pose=retreat_base_pose,
        )
        current_anchor_poses_base = {
            anchor_id: compose_pose(
                inverse_pose(base_root_pose),
                measured_selected_anchor_poses_world_by_anchor[anchor_id],
            )
            for anchor_id in selected_anchor_ids
        }
        last_kinematics = whole_structure_kinematics.compute(
            morphology_graph,
            physical_model,
            _global_dock_position_map(joint_vector),
            base_root_pose,
            anchor_references,
        )
        if last_kinematics.ordered_global_dock_joint_ids != joint_vector.joint_ids:
            raise RuntimeError(
                "Order8 whole-structure Jacobian columns changed during runtime"
            )
        kinematic_consistency_actual_module_pose_world_by_id = {
            module_id: _module_frame_pose_twist(
                robots[module_id],
                module_frame_link_id=module_frame_link_id,
            )[0]
            for module_id in module_ids
        }
        kinematic_consistency_predicted_module_pose_world_by_id = dict(
            last_kinematics.module_root_poses_world
        )
        kinematic_consistency_module_position_error_m_by_id = {
            module_id: _position_distance(
                kinematic_consistency_actual_module_pose_world_by_id[module_id],
                kinematic_consistency_predicted_module_pose_world_by_id[module_id],
            )
            for module_id in module_ids
        }
        kinematic_consistency_module_attitude_error_rad_by_id = {
            module_id: _norm(
                _rotation_error_world(
                    kinematic_consistency_actual_module_pose_world_by_id[module_id],
                    kinematic_consistency_predicted_module_pose_world_by_id[module_id],
                )
            )
            for module_id in module_ids
        }
        kinematic_consistency_max_module_position_error_m = max(
            kinematic_consistency_max_module_position_error_m,
            max(
                kinematic_consistency_module_position_error_m_by_id.values(),
                default=0.0,
            ),
        )
        kinematic_consistency_max_module_attitude_error_rad = max(
            kinematic_consistency_max_module_attitude_error_rad,
            max(
                kinematic_consistency_module_attitude_error_rad_by_id.values(),
                default=0.0,
            ),
        )
        kinematic_consistency_anchor_position_error_m_by_anchor = {
            anchor_id: _position_distance(
                measured_selected_anchor_poses_world_by_anchor[anchor_id],
                last_kinematics.anchor_poses_world[anchor_id],
            )
            for anchor_id in selected_anchor_ids
        }
        kinematic_consistency_anchor_attitude_error_rad_by_anchor = {
            anchor_id: _norm(
                _rotation_error_world(
                    measured_selected_anchor_poses_world_by_anchor[anchor_id],
                    last_kinematics.anchor_poses_world[anchor_id],
                )
            )
            for anchor_id in selected_anchor_ids
        }
        kinematic_consistency_max_anchor_position_error_m = max(
            kinematic_consistency_max_anchor_position_error_m,
            max(
                kinematic_consistency_anchor_position_error_m_by_anchor.values(),
                default=0.0,
            ),
        )
        kinematic_consistency_max_anchor_attitude_error_rad = max(
            kinematic_consistency_max_anchor_attitude_error_rad,
            max(
                kinematic_consistency_anchor_attitude_error_rad_by_anchor.values(),
                default=0.0,
            ),
        )
        if post_qclose_geometric_preload_surface_point_local_by_anchor:
            reference_by_anchor = {
                int(reference.anchor.anchor_id): reference
                for reference in anchor_references
            }
            kinematic_consistency_predicted_surface_point_world_by_anchor = {}
            for anchor_id, point_local in (
                post_qclose_geometric_preload_surface_point_local_by_anchor.items()
            ):
                point_in_anchor = compose_pose(
                    inverse_pose(reference_by_anchor[anchor_id].anchor.local_pose),
                    (*point_local, 0.0, 0.0, 0.0, 1.0),
                )
                kinematic_consistency_predicted_surface_point_world_by_anchor[
                    anchor_id
                ] = compose_pose(
                    last_kinematics.anchor_poses_world[anchor_id],
                    point_in_anchor,
                )[:3]
            kinematic_consistency_surface_point_error_m_by_anchor = {
                anchor_id: _norm(
                    tuple(
                        float(
                            kinematic_consistency_predicted_surface_point_world_by_anchor[
                                anchor_id
                            ][axis]
                        )
                        - float(
                            post_qclose_geometric_preload_current_surface_point_world_by_anchor[
                                anchor_id
                            ][axis]
                        )
                        for axis in range(3)
                    )
                )
                for anchor_id in sorted(
                    kinematic_consistency_predicted_surface_point_world_by_anchor
                )
            }
            kinematic_consistency_max_surface_point_error_m = max(
                kinematic_consistency_max_surface_point_error_m,
                max(
                    kinematic_consistency_surface_point_error_m_by_anchor.values(),
                    default=0.0,
                ),
            )
        if contact_preload_ready and not contact_yield_grasp_pose_rebased:
            if contact_axial_hold_base_pose is None:
                raise RuntimeError(
                    "Order8 positional preload was acquired without an axial hold pose"
                )
            # Rebase the normal centroidal controller from non-privileged
            # load-limited positional-preload evidence before restoring pose
            # P/I.  Raw
            # contact truth is not used for this control-mode transition.
            contact_yield_requested = False
            contact_admittance_requested = False
            # Retain the frozen absolute targets acquired by the slow
            # load-limited positional lead.  Replacing them with measured q
            # would erase the normal-force margin immediately before payload
            # transfer.  Offset torque remains zero; the AK40-10
            # torque/current/speed audits remain active.
            contact_yield_joint_drive_requested = False
            grasp_hold_anchor_poses_base = dict(current_anchor_poses_base)
            commanded_anchor_targets_base = dict(current_anchor_poses_base)
            planner_anchor_references = {
                anchor_id: compose_pose(base_root_pose, pose_base)
                for anchor_id, pose_base in current_anchor_poses_base.items()
            }
            terminal_anchor_references = dict(planner_anchor_references)
            contact_yield_grasp_pose_rebased = True
            contact_yield_grasp_pose_rebase_time_s = current_time_s
            contact_yield_grasp_pose = (
                current_external_wrench_centroidal_model.body_pose_world
            )
            # Rebuild every downstream waypoint from the complete measured
            # 6D grasp pose.  Applying only a translation offset would command
            # the small acquired roll/pitch back to nominal during lift and
            # create a tangential contact transient.
            grasp_base_pose = base_root_pose
            (
                lift_base_pose,
                transport_base_pose,
                place_base_pose,
                retreat_base_pose,
            ) = _rebased_manipulation_base_poses(
                grasp_base_pose,
                transport_distance_m=config.required_transport_distance_m,
            )
            contact_individual_latch_hold_base_pose = base_root_pose
            contact_centering_target_base_pose = base_root_pose
            commanded_base_target = base_root_pose
            nominal_base_target = base_root_pose
            desired_body_pose_by_phase[
                Order8NaturalContactPhase.CONTACT_ACQUISITION
            ] = base_root_pose
            desired_body_linear_velocity_by_phase[
                Order8NaturalContactPhase.CONTACT_ACQUISITION
            ] = (0.0, 0.0, 0.0)
            latched_contact_centering_offset_world = tuple(
                float(base_root_pose[index])
                - float(contact_axial_hold_base_pose[index])
                for index in range(3)
            )
            contact_admittance_controller.reset()
            for qpid in (contact_centering_controller, controller):
                qpid.reset_integrators()
        if phase == Order8NaturalContactPhase.APPROACH:
            achieved_shape_at_grasp = whole_structure_kinematics.forward(
                morphology_graph,
                physical_model,
                _global_dock_position_map(joint_vector),
                grasp_base_pose,
                anchor_references,
            )
            achieved_body_poses_at_grasp = {
                (
                    reference.surface.module_id,
                    reference.surface.mechanism_link_id,
                ): compose_pose(
                    achieved_shape_at_grasp.anchor_poses_world[
                        reference.anchor.anchor_id
                    ],
                    inverse_pose(reference.anchor.local_pose),
                )
                for reference in anchor_references
            }
            pregrasp_achieved_mesh_clearance_m = (
                _gripper_object_clearance_from_body_poses(
                    selected_gripper_local_aabbs,
                    achieved_body_poses_at_grasp,
                    object_pose,
                    runtime_object_size_m,
                )
            )
        pregrasp_base_aligned = _position_distance(
            nominal_base_target, base_root_pose
        ) <= float(
            config.pregrasp_position_tolerance_m
        ) and base_linear_speed_mps <= float(
            config.pregrasp_linear_speed_tolerance_mps
        )
        if (
            phase == Order8NaturalContactPhase.CONTACT_ACQUISITION
            and pregrasp_open_anchor_poses_base is not None
            and not contact_axial_aligned
            and gripper_axial_overlap_m
            >= float(config.contact_axial_min_mesh_overlap_m)
            and _base_hold_settled(
                live_grasp_base_pose,
                base_root_pose,
                base_linear_speed_mps=base_linear_speed_mps,
                position_tolerance_m=contact_axial_settle_position_tolerance_m,
                speed_tolerance_mps=float(config.pregrasp_linear_speed_tolerance_mps),
            )
        ):
            contact_axial_aligned = True
            contact_axial_overlap_at_latch_m = gripper_axial_overlap_m
            # The final precontact centroidal pose is object-relative.  Keep
            # following measured free-object motion through the normal command
            # rate limiter until q_close rather than freezing a stale world
            # pose after the first provisional contact.
            contact_axial_hold_base_pose = live_grasp_base_pose
        if (
            phase == Order8NaturalContactPhase.CONTACT_ACQUISITION
            and contact_axial_hold_base_pose is not None
            and not contact_configuration_latched
        ):
            contact_axial_hold_base_pose = live_grasp_base_pose
        if (
            phase == Order8NaturalContactPhase.CONTACT_ACQUISITION
            and contact_axial_hold_base_pose is not None
            and not contact_side_closure_enabled
        ):
            if (
                diagnostic_near_contact_fixture
                and not diagnostic_near_contact_warmup_complete
            ):
                contact_axial_settle_dwell_s = 0.0
            elif (
                diagnostic_near_contact_fixture
                and last_external_wrench_estimate.valid
                and base_linear_speed_mps
                <= contact_axial_settle_base_speed_tolerance_mps
            ):
                # The restored state intentionally includes the displacement
                # admitted by the yielded centroidal controller.  Requiring it
                # to return to the nominal object-relative pose would erase
                # precisely the state this fixture is meant to continue.
                contact_axial_settle_dwell_s += sim_dt
            elif _position_distance(
                live_grasp_base_pose,
                commanded_base_target,
            ) <= contact_axial_settle_position_tolerance_m and _base_hold_settled(
                live_grasp_base_pose,
                base_root_pose,
                base_linear_speed_mps=base_linear_speed_mps,
                position_tolerance_m=(contact_axial_settle_position_tolerance_m),
                speed_tolerance_mps=(contact_axial_settle_base_speed_tolerance_mps),
            ):
                contact_axial_settle_dwell_s += sim_dt
            else:
                contact_axial_settle_dwell_s = 0.0
            if contact_axial_settle_dwell_s >= float(config.contact_stall_dwell_s):
                contact_side_closure_enabled = True
        contact_region_joint_closure_active = bool(
            phase == Order8NaturalContactPhase.CONTACT_ACQUISITION
            and contact_side_closure_enabled
            and contact_axial_hold_base_pose is not None
            and not contact_configuration_latched
        )
        # Admittance is centroidal only.  Every Dock joint retains its nominal
        # implicit position-drive gains throughout contact acquisition.
        contact_yield_joint_drive_requested = False
        contact_load_detection_armed = bool(
            contact_region_joint_closure_active
            and all(
                selected_contact_joint_id_by_anchor[anchor_id]
                in latest_dock_actuator_telemetry
                for anchor_id in selected_anchor_ids
            )
        )
        if contact_load_detection_armed:
            contact_load_detection_armed_step_count += 1
            if contact_load_detection_armed_time_s is None:
                contact_load_detection_armed_time_s = current_time_s
        contact_yield_load_candidates_by_anchor = (
            _selected_anchor_surface_load_arrest_candidates(
                selected_anchor_ids,
                mesh_clearance_m_by_anchor=gripper_surface_clearance_m_by_anchor,
                selected_joint_load_nm_by_anchor=(
                    selected_contact_joint_load_nm_by_anchor
                ),
                mesh_clearance_arm_threshold_m=(
                    float(config.contact_surface_arm_clearance_m)
                    + float(config.contact_penetration_noise_floor_m)
                ),
                selected_joint_load_threshold_nm=(
                    contact_stall_selected_joint_load_threshold_nm
                ),
            )
            if contact_load_detection_armed
            else {anchor_id: False for anchor_id in selected_anchor_ids}
        )
        # Near-surface geometry alone is not contact.  Keep full centroidal
        # height/attitude/pose tracking through free-space closure and enable
        # the bounded horizontal-axis admittance only from the non-privileged
        # proximity-plus-damping-compensated actuator-load signature of at
        # least one surface.
        contact_yield_near_surface_candidates_by_anchor = {
            anchor_id: bool(
                gripper_surface_clearance_m_by_anchor[anchor_id]
                <= float(config.contact_surface_arm_clearance_m)
                + float(config.contact_penetration_noise_floor_m)
            )
            for anchor_id in selected_anchor_ids
        }
        contact_yield_candidates_by_anchor = {
            anchor_id: bool(contact_yield_load_candidates_by_anchor[anchor_id])
            for anchor_id in selected_anchor_ids
        }
        for anchor_id, candidate in contact_yield_candidates_by_anchor.items():
            if candidate:
                contact_yield_load_dwell_s_by_anchor[anchor_id] += sim_dt
            else:
                contact_yield_load_dwell_s_by_anchor[anchor_id] = 0.0
        # The first detected terminal-joint contact load enables the approved
        # centroidal admittance immediately.  The longer simultaneous dwell
        # below is reserved for q_close, not for delaying compliance onset.
        newly_triggered_yield_anchor_ids = {
            anchor_id
            for anchor_id, candidate in contact_yield_candidates_by_anchor.items()
            if candidate
        }
        if newly_triggered_yield_anchor_ids and not contact_yield_requested:
            contact_yield_requested = True
            contact_yield_triggered_time_s = current_time_s
        contact_yield_trigger_anchor_ids.update(newly_triggered_yield_anchor_ids)
        newly_triggered_admittance_anchor_ids = {
            anchor_id
            for anchor_id in newly_triggered_yield_anchor_ids
            if last_external_wrench_estimate.valid
        }
        if (
            newly_triggered_admittance_anchor_ids
            and not contact_admittance_requested
        ):
            contact_admittance_requested = True
            contact_admittance_triggered_time_s = current_time_s
        contact_admittance_trigger_anchor_ids.update(
            newly_triggered_admittance_anchor_ids
        )
        if (
            contact_yield_joint_drive_requested
            and contact_configuration_latched
            and post_qclose_joint_settle_complete
        ):
            # Low joint impedance is useful while discovering the first
            # simultaneous arrest, but it must not remain the bottleneck for
            # the subsequent bounded geometric preload.  After measured q_close
            # velocity has settled, restore the ordinary virtual drive smoothly
            # while centroidal P/I remains yielded and contact torque bias stays
            # at zero.  PhysX effort/current/speed limits still bound the motion.
            contact_yield_joint_drive_requested = False
        if (
            contact_yield_joint_drive_requested
            and phase
            in {
                Order8NaturalContactPhase.RETREAT,
                Order8NaturalContactPhase.SETTLE,
                Order8NaturalContactPhase.COMPLETE,
            }
        ):
            # Fail-safe restore for an externally forced phase transition that
            # bypasses the normal stable-grasp rebase path.
            contact_yield_joint_drive_requested = False
        previous_contact_yield_blend = contact_yield_blend
        contact_yield_blend = _advance_contact_yield_blend(
            contact_yield_blend,
            yield_requested=contact_yield_requested,
            dt_s=sim_dt,
            ramp_down_s=float(config.contact_yield_ramp_down_s),
            ramp_up_s=float(config.contact_yield_ramp_up_s),
        )
        contact_yield_tracking_profile = _contact_yield_tracking_profile(
            contact_yield_blend,
            integrator_decay_rate_per_s=float(
                config.contact_yield_integrator_decay_rate_per_s
            ),
        )
        contact_yield_active_step_count += int(contact_yield_blend > 1.0e-12)
        contact_yield_full_step_count += int(
            contact_yield_blend >= 1.0 - 1.0e-12
        )
        contact_yield_restore_step_count += int(
            not contact_yield_requested
            and previous_contact_yield_blend > contact_yield_blend
        )
        contact_yield_minimum_pi_scale = min(
            contact_yield_minimum_pi_scale,
            contact_yield_tracking_profile.proportional_gain_scale,
        )
        previous_contact_yield_joint_drive_blend = (
            contact_yield_joint_drive_blend
        )
        contact_yield_joint_drive_blend = _advance_contact_yield_blend(
            contact_yield_joint_drive_blend,
            yield_requested=contact_yield_joint_drive_requested,
            dt_s=sim_dt,
            ramp_down_s=float(config.contact_yield_ramp_down_s),
            ramp_up_s=float(config.contact_yield_ramp_up_s),
        )
        (
            requested_contact_joint_drive_stiffness,
            requested_contact_joint_drive_damping,
        ) = _contact_yield_joint_drive_gains(
            contact_yield_joint_drive_blend,
            nominal_stiffness_nm_per_rad=float(dock_stiffness),
            nominal_damping_nms_per_rad=float(dock_damping),
            yield_stiffness_scale=float(
                config.contact_yield_joint_drive_stiffness_scale
            ),
            yield_damping_nms_per_rad=float(
                config.contact_yield_joint_drive_damping_nms_per_rad
            ),
        )
        if (
            not math.isclose(
                requested_contact_joint_drive_stiffness,
                contact_yield_joint_drive_last_stiffness_nm_per_rad,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            or not math.isclose(
                requested_contact_joint_drive_damping,
                contact_yield_joint_drive_last_damping_nms_per_rad,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
        ):
            (
                contact_yield_joint_drive_stiffness_targets,
                contact_yield_joint_drive_damping_targets,
            ) = _schedule_contact_joint_drive_impedance(
                robots,
                expected_joint_ids,
                stiffness_nm_per_rad=(
                    requested_contact_joint_drive_stiffness
                ),
                damping_nms_per_rad=requested_contact_joint_drive_damping,
                maximum_stiffness_nm_per_rad=float(dock_stiffness),
                maximum_damping_nms_per_rad=(
                    ORDER8_SIMULATION_DRIVE_DAMPING_MAX_NMS_PER_RAD
                ),
            )
            contact_yield_joint_drive_write_count += 1
            contact_yield_joint_drive_restore_write_count += int(
                contact_yield_joint_drive_blend
                < previous_contact_yield_joint_drive_blend
            )
            contact_yield_joint_drive_last_stiffness_nm_per_rad = (
                requested_contact_joint_drive_stiffness
            )
            contact_yield_joint_drive_last_damping_nms_per_rad = (
                requested_contact_joint_drive_damping
            )
        contact_yield_joint_drive_active_step_count += int(
            contact_yield_joint_drive_blend > 1.0e-12
        )
        contact_yield_joint_drive_minimum_stiffness_nm_per_rad = min(
            contact_yield_joint_drive_minimum_stiffness_nm_per_rad,
            requested_contact_joint_drive_stiffness,
        )
        contact_yield_joint_drive_maximum_damping_nms_per_rad = max(
            contact_yield_joint_drive_maximum_damping_nms_per_rad,
            requested_contact_joint_drive_damping,
        )
        if contact_yield_blend > 1.0e-12 and last_external_wrench_estimate.valid:
            contact_yield_maximum_external_force_n = max(
                contact_yield_maximum_external_force_n,
                last_external_wrench_estimate.force_norm_n,
            )
            contact_yield_maximum_external_torque_nm = max(
                contact_yield_maximum_external_torque_nm,
                last_external_wrench_estimate.torque_norm_nm,
            )
        # Superseded one-sided-freeze/recenter modes remain readable for old
        # diagnostic artifacts but are not part of the v3 acquisition path.
        contact_centering_active = False
        contact_continuous_balance_active = False
        post_first_arrest_centroidal_transfer_active = False
        sequential_free_shape_nudge_active = False
        sequential_latched_transfer_active = False
        sequential_centroidal_transfer_active = False
        # Contact acquisition must use an anchor-specific local signal.  A
        # maximum over every Jacobian-influential upstream Dock aliases the
        # same graph-constraint/impedance transient into both grippers and can
        # therefore fabricate a simultaneous q_close while neither authored
        # mesh is touching the object.  Use the mechanism joint that directly
        # carries each selected Dock link for first-contact/yield/q_close and
        # geometric-preload completion.  The influential-chain loads remain
        # separately observed and are still used by the post-grasp whole-
        # structure preload/stability gate.
        contact_stall_joint_load_nm_by_anchor = dict(
            selected_contact_joint_load_nm_by_anchor
        )
        # Do not freeze a merely near/load-bearing configuration.  The sampled
        # mesh proxy can enter its millimetre noise band before PhysX reports a
        # physical patch, and an upstream Dock can carry reaction load while
        # an anchor still has normal closure remaining.  Freezing on those two
        # signals stranded the command several millimetres before contact.
        # Continue bounded creep until the stricter simultaneous q_close gate
        # also observes low object-relative speed on both sides.  Acquisition
        # contact may separate/reacquire, so no intermediate pose is latched.
        contact_provisional_surface_settle_active = False
        contact_provisional_surface_settle_active_step_count += int(
            contact_provisional_surface_settle_active
        )
        contact_centering_active_step_count += int(contact_centering_active)
        contact_continuous_balance_active_step_count += int(
            contact_continuous_balance_active
        )
        contact_sequential_reacquire_active_step_count += int(
            contact_pursued_anchor_id is not None
        )
        contact_sequential_centroidal_nudge_active_step_count += int(
            sequential_free_shape_nudge_active
        )
        contact_sequential_latched_transfer_active_step_count += int(
            sequential_latched_transfer_active
        )
        contact_axial_gain_scheduled = bool(
            phase == Order8NaturalContactPhase.CONTACT_ACQUISITION
            and not contact_axial_aligned
        )
        contact_axial_gain_scheduled_step_count += int(contact_axial_gain_scheduled)
        # The higher horizontal-gain bank is used for collision-free mesh-open
        # axial insertion and the non-privileged clearance-balancing outer
        # loop.  QPID still owns only centroidal thrust/vectoring; Dock motion
        # remains the independent quasi-static servo path.
        contact_motion_qpid_gain_scheduled = bool(
            contact_axial_gain_scheduled or contact_region_joint_closure_active
        )
        if contact_axial_hold_base_pose is None:
            contact_centering_target_base_pose = None
            contact_centering_offset_world = (0.0, 0.0, 0.0)
        elif (
            contact_configuration_latched
            and contact_individual_latch_hold_base_pose is not None
        ):
            # q_close is a complete measured centroidal pose.  Reconstructing
            # it from only the translation offset would discard the small
            # underactuated tilt and inject an attitude step exactly when the
            # positional preload begins.
            contact_centering_target_base_pose = contact_individual_latch_hold_base_pose
            contact_centering_offset_world = tuple(
                float(contact_centering_target_base_pose[index])
                - float(contact_axial_hold_base_pose[index])
                for index in range(3)
            )
        elif latched_contact_centering_offset_world is not None:
            contact_centering_target_base_pose = _offset_pose(
                contact_axial_hold_base_pose,
                dx=latched_contact_centering_offset_world[0],
                dy=latched_contact_centering_offset_world[1],
                dz=latched_contact_centering_offset_world[2],
            )
            contact_centering_offset_world = latched_contact_centering_offset_world
        elif contact_region_joint_closure_active:
            # Hold/follow only the known object-relative centroidal grasp pose.
            # Joint closure is an independent constant-velocity test-driver
            # command; no mesh error is fed back into the QPID target.
            contact_closure_common_translation_world = (0.0, 0.0, 0.0)
            contact_centering_target_base_pose = live_grasp_base_pose
            contact_centering_offset_world = tuple(
                float(contact_centering_target_base_pose[index])
                - float(contact_axial_hold_base_pose[index])
                for index in range(3)
            )
        elif sequential_centroidal_transfer_active:
            if (
                contact_pursued_anchor_id is None
                or contact_sequential_transfer_origin_base_pose is None
                or contact_sequential_transfer_limit_m is None
            ):
                raise RuntimeError(
                    "Order8 sequential centroidal nudge lacks an active target"
                )
            # The remaining normal direction is locally singular for Dock-only
            # motion.  First move the frozen released shape a few millimetres
            # to establish the opposite arrest.  Then retain that arrest as a
            # world-frame anchor task while translating the centroidal frame
            # toward the final side.  Both transfers stop on the same
            # non-privileged drive-load arrest gate.
            sequential_position_target = _post_first_arrest_centroidal_transfer_pose(
                contact_sequential_transfer_origin_base_pose,
                inward_normal_world=next(
                    selection.inward_normal_world
                    for selection in selections
                    if int(selection.anchor_id) == contact_pursued_anchor_id
                ),
                maximum_transfer_m=contact_sequential_transfer_limit_m,
            )
            contact_centering_target_base_pose = _underactuated_contact_centering_pose(
                sequential_position_target,
                hold_pose=contact_sequential_transfer_origin_base_pose,
                current_pose=base_root_pose,
                current_linear_velocity_world=base_module_twist[:3],
                speed_limit_mps=min(
                    float(config.contact_base_translation_speed_limit_mps),
                    float(config.contact_surface_creep_speed_limit_mps)
                    * ORDER8_POST_FIRST_ARREST_CREEP_MULTIPLIER,
                ),
                slowdown_distance_m=float(config.contact_near_surface_slowdown_m),
                position_deadband_m=contact_axial_settle_position_tolerance_m,
                xy_p_gain=float(contact_centering_qpid_config.xy_p_gain),
                xy_d_gain=float(contact_centering_qpid_config.xy_d_gain),
                gravity_mps2=float(contact_centering_qpid_config.gravity_mps2),
                max_tilt_rad=float(config.contact_centering_max_tilt_rad),
            )
            contact_centering_offset_world = tuple(
                float(contact_centering_target_base_pose[index])
                - float(contact_axial_hold_base_pose[index])
                for index in range(3)
            )
        elif contact_centering_active or contact_continuous_balance_active:
            if (
                contact_centering_active
                and contact_centering_unlatched_anchor_id is not None
                and contact_individual_latch_hold_base_pose is not None
            ):
                # Freeze the complete articulated shape and move away from the
                # singly loaded side.  The unlatched side's inward normal is
                # exactly that release/recenter direction for an opposing pair.
                contact_centering_position_target = (
                    _post_first_arrest_centroidal_transfer_pose(
                        contact_individual_latch_hold_base_pose,
                        inward_normal_world=next(
                            selection.inward_normal_world
                            for selection in selections
                            if int(selection.anchor_id)
                            == contact_centering_unlatched_anchor_id
                        ),
                        maximum_transfer_m=float(
                            config.contact_clearance_sync_full_slowdown_m
                        ),
                    )
                )
            else:
                contact_centering_position_target = _contact_centering_base_pose(
                    contact_axial_hold_base_pose,
                    base_root_pose,
                    mesh_clearance_m_by_anchor=(gripper_surface_clearance_m_by_anchor),
                    inward_normal_world_by_anchor={
                        selection.anchor_id: selection.inward_normal_world
                        for selection in selections
                    },
                    max_offset_m=float(config.contact_centering_max_offset_m),
                )
            contact_centering_target_base_pose = _underactuated_contact_centering_pose(
                contact_centering_position_target,
                hold_pose=contact_axial_hold_base_pose,
                current_pose=base_root_pose,
                current_linear_velocity_world=base_module_twist[:3],
                speed_limit_mps=(
                    min(
                        float(config.contact_base_translation_speed_limit_mps),
                        float(config.contact_surface_creep_speed_limit_mps)
                        * ORDER8_POST_FIRST_ARREST_CREEP_MULTIPLIER,
                    )
                    if contact_centering_active
                    else float(config.contact_base_translation_speed_limit_mps)
                ),
                slowdown_distance_m=float(config.contact_near_surface_slowdown_m),
                position_deadband_m=(contact_axial_settle_position_tolerance_m),
                xy_p_gain=float(contact_centering_qpid_config.xy_p_gain),
                xy_d_gain=float(contact_centering_qpid_config.xy_d_gain),
                gravity_mps2=float(contact_centering_qpid_config.gravity_mps2),
                max_tilt_rad=float(config.contact_centering_max_tilt_rad),
            )
            contact_centering_offset_world = tuple(
                float(contact_centering_target_base_pose[index])
                - float(contact_axial_hold_base_pose[index])
                for index in range(3)
            )
        elif (
            post_first_arrest_centroidal_transfer_active
            and contact_individual_latch_hold_base_pose is not None
        ):
            # Dock-only closure is nearly singular after the first loaded
            # arrest in the representative morphology.  Translate the whole
            # structure toward the unlatched side while independent Dock IK
            # holds the arrested anchor at a fixed world pose.  QPID remains
            # centroidal and joint-unaware, and no raw contact truth is used.
            unlatched_anchor_id = next(
                anchor_id
                for anchor_id in selected_anchor_ids
                if anchor_id not in contact_stall_latched_anchor_poses_world
            )
            transfer_position_target = _post_first_arrest_centroidal_transfer_pose(
                contact_individual_latch_hold_base_pose,
                inward_normal_world=next(
                    selection.inward_normal_world
                    for selection in selections
                    if int(selection.anchor_id) == unlatched_anchor_id
                ),
                maximum_transfer_m=float(config.contact_centering_max_offset_m),
            )
            contact_centering_target_base_pose = _underactuated_contact_centering_pose(
                transfer_position_target,
                hold_pose=contact_individual_latch_hold_base_pose,
                current_pose=base_root_pose,
                current_linear_velocity_world=base_module_twist[:3],
                speed_limit_mps=min(
                    float(config.contact_base_translation_speed_limit_mps),
                    float(config.contact_surface_creep_speed_limit_mps)
                    * ORDER8_POST_FIRST_ARREST_CREEP_MULTIPLIER,
                ),
                slowdown_distance_m=float(config.contact_near_surface_slowdown_m),
                position_deadband_m=(contact_axial_settle_position_tolerance_m),
                xy_p_gain=float(contact_centering_qpid_config.xy_p_gain),
                xy_d_gain=float(contact_centering_qpid_config.xy_d_gain),
                gravity_mps2=float(contact_centering_qpid_config.gravity_mps2),
                max_tilt_rad=float(config.contact_centering_max_tilt_rad),
            )
            max_post_first_arrest_centroidal_transfer_m = max(
                max_post_first_arrest_centroidal_transfer_m,
                _position_distance(
                    contact_individual_latch_hold_base_pose,
                    base_root_pose,
                ),
            )
            contact_centering_offset_world = tuple(
                float(contact_centering_target_base_pose[index])
                - float(contact_axial_hold_base_pose[index])
                for index in range(3)
            )
        elif contact_individual_latch_hold_base_pose is not None:
            # Once both sides have independently arrested, hold the complete
            # measured centroidal pose through simultaneous reacquisition.
            contact_centering_target_base_pose = contact_individual_latch_hold_base_pose
            contact_centering_offset_world = tuple(
                float(contact_centering_target_base_pose[index])
                - float(contact_axial_hold_base_pose[index])
                for index in range(3)
            )
        else:
            contact_centering_target_base_pose = _offset_pose(
                contact_axial_hold_base_pose,
                dx=committed_contact_centering_offset_world[0],
                dy=committed_contact_centering_offset_world[1],
                dz=committed_contact_centering_offset_world[2],
            )
            contact_centering_offset_world = committed_contact_centering_offset_world
        unilateral_release_centering_settled = bool(
            contact_centering_active
            and contact_centering_unlatched_anchor_id is not None
            and contact_centering_target_base_pose is not None
            and _base_hold_settled(
                contact_centering_target_base_pose,
                base_root_pose,
                base_linear_speed_mps=base_linear_speed_mps,
                position_tolerance_m=contact_axial_settle_position_tolerance_m,
                speed_tolerance_mps=float(
                    config.contact_stall_anchor_speed_threshold_mps
                ),
            )
            and all(
                selected_contact_joint_load_nm_by_anchor[anchor_id]
                <= 0.5 * contact_stall_selected_joint_load_threshold_nm
                for anchor_id in contact_stall_latched_anchor_poses_base
            )
            and _norm(
                _rotation_error_world(
                    base_root_pose,
                    contact_individual_latch_hold_base_pose,
                )
            )
            <= float(config.contact_centering_max_tilt_rad)
        )
        geometric_centering_settled = bool(
            contact_centering_active
            and contact_centering_unlatched_anchor_id is None
            and _contact_pair_centering_settled(
                gripper_surface_clearance_m_by_anchor,
                base_linear_speed_mps=base_linear_speed_mps,
                speed_tolerance_mps=float(
                    config.contact_stall_anchor_speed_threshold_mps
                ),
                imbalance_tolerance_m=float(config.contact_surface_arm_clearance_m),
                measured_tilt_rad=_norm(
                    _rotation_error_world(
                        base_root_pose,
                        contact_axial_hold_base_pose,
                    )
                ),
                max_tilt_rad=float(config.contact_centering_max_tilt_rad),
            )
        )
        if unilateral_release_centering_settled or geometric_centering_settled:
            contact_centering_settle_dwell_s += sim_dt
        elif contact_centering_active:
            contact_centering_settle_dwell_s = 0.0
        if contact_centering_active and contact_centering_settle_dwell_s >= float(
            config.contact_stall_dwell_s
        ):
            if contact_axial_hold_base_pose is None:
                raise RuntimeError(
                    "Order8 contact centering settled without axial hold"
                )
            committed_contact_centering_offset_world = tuple(
                float(base_root_pose[index])
                - float(contact_axial_hold_base_pose[index])
                for index in range(3)
            )
            contact_centering_offset_world = committed_contact_centering_offset_world
            # The achieved underactuated centering pose includes a small
            # bounded roll/pitch component.  Preserve that complete measured
            # pose when leaving the centering cycle; reconstructing only its
            # translation would inject an attitude step before reacquisition.
            contact_centering_target_base_pose = base_root_pose
            contact_centering_hold_anchor_poses_base = None
            if contact_centering_unlatched_anchor_id is not None:
                # Hold the now-backed-off side at its released world pose and
                # pursue only the opposite side.  The old contact latch itself
                # is cleared because it must be earned again after release.
                contact_backed_off_anchor_hold_poses_world = {
                    anchor_id: measured_selected_anchor_poses_world_by_anchor[
                        anchor_id
                    ]
                    for anchor_id in contact_stall_latched_anchor_poses_base
                }
                contact_pursued_anchor_id = contact_centering_unlatched_anchor_id
                contact_sequential_transfer_origin_base_pose = base_root_pose
                contact_sequential_transfer_limit_m = min(
                    float(config.contact_centering_max_offset_m),
                    ORDER8_POST_FIRST_ARREST_CREEP_MULTIPLIER
                    * float(config.contact_clearance_sync_full_slowdown_m),
                )
                contact_stall_latched_anchor_poses_base.clear()
                contact_stall_latched_anchor_poses_world.clear()
                contact_stall_latched_mesh_clearance_m_by_anchor.clear()
                contact_reacquired_hold_anchor_poses_base.clear()
                for anchor_id in selected_anchor_ids:
                    nonprivileged_contact_stall_dwell_s_by_anchor[anchor_id] = 0.0
                nonprivileged_contact_stall_dwell_s = 0.0
                commanded_anchor_targets_base = dict(current_anchor_poses_base)
                joint_position_reference_by_id = {
                    joint_id: float(position)
                    for joint_id, position in zip(
                        joint_vector.joint_ids,
                        joint_vector.positions_rad,
                        strict=True,
                    )
                }
                contact_centering_unlatched_anchor_id = None
            contact_individual_latch_hold_base_pose = base_root_pose
            commanded_base_target = base_root_pose
            desired_body_pose_by_phase[
                Order8NaturalContactPhase.CONTACT_ACQUISITION
            ] = base_root_pose
            desired_body_linear_velocity_by_phase[
                Order8NaturalContactPhase.CONTACT_ACQUISITION
            ] = (0.0, 0.0, 0.0)
            for qpid in (contact_centering_controller, controller):
                qpid.reset_integrators()
            contact_centering_settle_dwell_s = 0.0
            contact_centering_active = False
            contact_centering_cycle_count += 1
        max_contact_centering_offset_m = max(
            max_contact_centering_offset_m,
            _norm(contact_centering_offset_world),
        )
        post_first_arrest_centroidal_transfer_active_step_count += int(
            post_first_arrest_centroidal_transfer_active
        )
        contact_centering_measured_tilt_rad = (
            0.0
            if contact_axial_hold_base_pose is None
            else _norm(
                _rotation_error_world(
                    contact_axial_hold_base_pose,
                    base_root_pose,
                )
            )
        )
        max_contact_centering_measured_tilt_rad = max(
            max_contact_centering_measured_tilt_rad,
            contact_centering_measured_tilt_rad,
        )
        if (
            contact_axial_hold_base_pose is not None
            and contact_centering_target_base_pose is not None
        ):
            max_contact_centering_tilt_rad = max(
                max_contact_centering_tilt_rad,
                _norm(
                    _rotation_error_world(
                        contact_axial_hold_base_pose,
                        contact_centering_target_base_pose,
                    )
                ),
            )
        terminal_body_pose_by_phase = {
            item: _base_target_for_phase(
                item,
                hover_base_pose=hover_base_pose,
                approach_base_pose=approach_base_pose,
                grasp_base_pose=(
                    grasp_base_pose
                    if contact_configuration_latched
                    else live_grasp_base_pose
                ),
                lift_base_pose=lift_base_pose,
                transport_base_pose=transport_base_pose,
                place_base_pose=place_base_pose,
                retreat_base_pose=retreat_base_pose,
            )
            for item in Order8NaturalContactPhase
        }
        if contact_axial_hold_base_pose is not None:
            terminal_body_pose_by_phase[
                Order8NaturalContactPhase.CONTACT_ACQUISITION
            ] = contact_centering_target_base_pose
        if (
            latched_contact_centering_offset_world is not None
            and not contact_yield_grasp_pose_rebased
        ):
            for item in (
                Order8NaturalContactPhase.LIFT,
                Order8NaturalContactPhase.TRANSPORT,
                Order8NaturalContactPhase.PLACE,
                Order8NaturalContactPhase.RELEASE,
                Order8NaturalContactPhase.RETREAT,
                Order8NaturalContactPhase.SETTLE,
                Order8NaturalContactPhase.COMPLETE,
            ):
                terminal_body_pose_by_phase[item] = _offset_pose(
                    terminal_body_pose_by_phase[item],
                    dx=latched_contact_centering_offset_world[0],
                    dy=latched_contact_centering_offset_world[1],
                    dz=latched_contact_centering_offset_world[2],
                )
        contact_base_target_speed_limit_mps = float(
            config.contact_base_translation_speed_limit_mps
        )
        if sequential_latched_transfer_active:
            # Once one surface is physically arrested, move the centroidal
            # command no faster than the final surface-creep tier.  The
            # underactuated tilt helper shapes attitude only; this outer
            # command limiter prevents a millimetre-scale terminal target from
            # becoming a 10 mm/s centroidal step.
            contact_base_target_speed_limit_mps = min(
                contact_base_target_speed_limit_mps,
                float(config.contact_surface_creep_speed_limit_mps),
            )
        elif (
            post_first_arrest_centroidal_transfer_active
            or contact_centering_active
            or sequential_free_shape_nudge_active
        ):
            contact_base_target_speed_limit_mps = min(
                contact_base_target_speed_limit_mps,
                float(config.contact_surface_creep_speed_limit_mps)
                * ORDER8_POST_FIRST_ARREST_CREEP_MULTIPLIER,
            )
        desired_body_pose_by_phase = {
            item: _advance_pose_toward(
                commanded_base_target,
                terminal_body_pose_by_phase[item],
                max_translation_step_m=(
                    _base_translation_speed_limit_for_phase(
                        item,
                        free_motion_limit_mps=float(
                            config.base_translation_speed_limit_mps
                        ),
                        maintained_contact_limit_mps=(
                            contact_base_target_speed_limit_mps
                        ),
                    )
                    * sim_dt
                ),
            )
            for item in Order8NaturalContactPhase
        }
        desired_body_linear_velocity_by_phase = {
            item: tuple(
                (
                    float(desired_body_pose_by_phase[item][index])
                    - float(commanded_base_target[index])
                )
                / sim_dt
                for index in range(3)
            )
            for item in Order8NaturalContactPhase
        }
        current_contact_tangential_offset_m_by_anchor = {
            anchor_id: _contact_region_tangential_offsets_m(
                current_anchor_pose_world=(
                    *contact_control_surface_point_world_by_anchor[anchor_id],
                    *measured_selected_anchor_poses_world_by_anchor[anchor_id][3:7],
                ),
                nominal_anchor_pose_world=(
                    live_contact_anchor_targets_world[anchor_id]
                ),
                object_pose_world=tuple(object_state["pose"]),
                inward_normal_object=(
                    contact_inward_normal_object_by_anchor[anchor_id]
                ),
            )
            for anchor_id in selected_anchor_ids
        }
        mesh_precenter_candidate = bool(
            phase == Order8NaturalContactPhase.CONTACT_ACQUISITION
            and contact_side_closure_enabled
            and not contact_configuration_latched
            and all(
                abs(component_m)
                <= float(config.contact_tangential_tolerance_m) + 1.0e-9
                for anchor_id in selected_anchor_ids
                for component_m in (
                    current_contact_tangential_offset_m_by_anchor[anchor_id]
                )
            )
            and all(
                abs(
                    gripper_surface_clearance_m_by_anchor[anchor_id]
                    - float(config.contact_near_surface_slowdown_m)
                )
                <= float(config.contact_surface_arm_clearance_m) + 1.0e-9
                and current_anchor_object_relative_speed_mps_by_anchor[anchor_id]
                <= float(config.contact_stall_anchor_speed_threshold_mps)
                for anchor_id in selected_anchor_ids
            )
        )
        if not contact_mesh_precenter_complete:
            if mesh_precenter_candidate:
                contact_mesh_precenter_dwell_s += sim_dt
            else:
                contact_mesh_precenter_dwell_s = 0.0
            if contact_mesh_precenter_dwell_s >= float(config.contact_stall_dwell_s):
                contact_mesh_precenter_complete = True
                contact_mesh_precenter_completed_time_s = current_time_s
        live_contact_precenter_targets_world = {
            anchor_id: _contact_precenter_nominal_pose(
                live_contact_anchor_targets_world[anchor_id],
                object_pose_world=tuple(object_state["pose"]),
                inward_normal_object=(
                    contact_inward_normal_object_by_anchor[anchor_id]
                ),
                inward_overtravel_m=float(
                    config.contact_closure_inward_overtravel_m
                ),
                clearance_m=float(config.contact_near_surface_slowdown_m),
            )
            for anchor_id in selected_anchor_ids
        }
        live_contact_region_targets_world = {
            anchor_id: _contact_region_pose_target(
                current_anchor_pose_world=(
                    measured_selected_anchor_poses_world_by_anchor[anchor_id]
                ),
                current_surface_point_world=(
                    contact_control_surface_point_world_by_anchor[anchor_id]
                ),
                nominal_anchor_pose_world=(
                    live_contact_anchor_targets_world[anchor_id]
                    if contact_mesh_precenter_complete
                    else live_contact_precenter_targets_world[anchor_id]
                ),
                object_pose_world=tuple(object_state["pose"]),
                inward_normal_object=(
                    contact_inward_normal_object_by_anchor[anchor_id]
                ),
                tangential_tolerance_m=float(config.contact_tangential_tolerance_m),
            )
            for anchor_id in selected_anchor_ids
        }
        last_anchor_object_relative_speed_mps_by_anchor = dict(
            current_anchor_object_relative_speed_mps_by_anchor
        )
        max_anchor_object_relative_speed_mps_by_anchor = {
            anchor_id: max(
                max_anchor_object_relative_speed_mps_by_anchor[anchor_id],
                current_anchor_object_relative_speed_mps_by_anchor[anchor_id],
            )
            for anchor_id in selected_anchor_ids
        }
        last_anchor_object_normal_relative_speed_mps_by_anchor = dict(
            current_anchor_object_normal_relative_speed_mps_by_anchor
        )
        max_anchor_object_normal_relative_speed_mps_by_anchor = {
            anchor_id: max(
                max_anchor_object_normal_relative_speed_mps_by_anchor[anchor_id],
                current_anchor_object_normal_relative_speed_mps_by_anchor[anchor_id],
            )
            for anchor_id in selected_anchor_ids
        }
        last_anchor_object_filtered_normal_relative_speed_mps_by_anchor = dict(
            current_anchor_object_filtered_normal_relative_speed_mps_by_anchor
        )
        max_anchor_object_filtered_normal_relative_speed_mps_by_anchor = {
            anchor_id: max(
                max_anchor_object_filtered_normal_relative_speed_mps_by_anchor[
                    anchor_id
                ],
                current_anchor_object_filtered_normal_relative_speed_mps_by_anchor[
                    anchor_id
                ],
            )
            for anchor_id in selected_anchor_ids
        }
        last_gripper_surface_clearance_rate_mps_by_anchor = dict(
            current_gripper_surface_clearance_rate_mps_by_anchor
        )
        last_filtered_gripper_surface_clearance_rate_mps_by_anchor = dict(
            current_filtered_gripper_surface_clearance_speed_mps_by_anchor
        )
        max_filtered_gripper_surface_clearance_rate_mps_by_anchor = {
            anchor_id: max(
                max_filtered_gripper_surface_clearance_rate_mps_by_anchor[anchor_id],
                current_filtered_gripper_surface_clearance_speed_mps_by_anchor[
                    anchor_id
                ],
            )
            for anchor_id in selected_anchor_ids
        }
        contact_anchor_tier_speed_limit_mps_by_anchor = (
            _contact_anchor_target_speed_limits_mps(
                base_limit_mps=float(config.anchor_translation_speed_limit_mps),
                mesh_clearance_m_by_anchor=(gripper_surface_clearance_m_by_anchor),
                near_mesh_clearance_m=float(config.contact_near_surface_slowdown_m),
                surface_arm_clearance_m=float(config.contact_surface_arm_clearance_m),
                surface_creep_speed_limit_mps=float(
                    config.contact_surface_creep_speed_limit_mps
                ),
            )
        )
        contact_clearance_sync_enabled = bool(
            phase == Order8NaturalContactPhase.CONTACT_ACQUISITION
            and contact_side_closure_enabled
            and not contact_configuration_latched
            and contact_centering_hold_anchor_poses_base is None
            and not contact_stall_latched_anchor_poses_base
            and contact_centering_cycle_count == 0
        )
        contact_anchor_target_speed_limit_mps_by_anchor = (
            _clearance_synchronized_contact_anchor_target_speed_limits_mps(
                contact_anchor_tier_speed_limit_mps_by_anchor,
                mesh_clearance_m_by_anchor=(gripper_surface_clearance_m_by_anchor),
                deadband_m=float(config.contact_clearance_sync_deadband_m),
                full_slowdown_m=float(config.contact_clearance_sync_full_slowdown_m),
                minimum_speed_scale=float(
                    config.contact_clearance_sync_minimum_speed_scale
                ),
            )
            if contact_clearance_sync_enabled
            else contact_anchor_tier_speed_limit_mps_by_anchor
        )
        contact_anchor_target_speed_limit_mps_by_anchor = (
            _accelerate_unlatched_anchor_after_first_arrest(
                contact_anchor_target_speed_limit_mps_by_anchor,
                latched_anchor_ids=set(contact_stall_latched_anchor_poses_base),
                maximum_speed_mps=float(config.anchor_translation_speed_limit_mps),
                creep_speed_mps=float(config.contact_surface_creep_speed_limit_mps),
                multiplier=ORDER8_POST_FIRST_ARREST_CREEP_MULTIPLIER,
            )
        )
        if (
            phase == Order8NaturalContactPhase.CONTACT_ACQUISITION
            and contact_mesh_precenter_complete
            and not contact_configuration_latched
        ):
            # The clear tangential precenter ends at the near-surface boundary.
            # From there both authored meshes use the same creep ceiling until
            # q_close; retaining the faster near tier created a one-sided impact
            # and a transient AK40 velocity-envelope violation.
            contact_anchor_target_speed_limit_mps_by_anchor = {
                anchor_id: min(
                    speed,
                    float(config.contact_surface_creep_speed_limit_mps),
                )
                for anchor_id, speed in (
                    contact_anchor_target_speed_limit_mps_by_anchor.items()
                )
            }
        if (
            phase == Order8NaturalContactPhase.CONTACT_ACQUISITION
            and contact_centering_cycle_count > 0
            and not contact_stall_latched_anchor_poses_base
            and not contact_configuration_latched
        ):
            # After a unilateral release/recenter cycle the sampled mesh
            # clearance ordering is known to be unreliable inside the convex-
            # decomposition noise band.  Retry both sides at a bounded creep
            # floor instead of allowing that metric to starve one side.
            post_recenter_creep_floor = min(
                float(config.anchor_translation_speed_limit_mps),
                float(config.contact_surface_creep_speed_limit_mps)
                * ORDER8_POST_FIRST_ARREST_CREEP_MULTIPLIER,
            )
            contact_anchor_target_speed_limit_mps_by_anchor = {
                anchor_id: max(speed, post_recenter_creep_floor)
                for anchor_id, speed in (
                    contact_anchor_target_speed_limit_mps_by_anchor.items()
                )
            }
        post_first_arrest_creep_active_step_count += int(
            0 < len(contact_stall_latched_anchor_poses_base) < len(selected_anchor_ids)
        )
        contact_clearance_sync_active_step_count += int(
            contact_clearance_sync_enabled
            and contact_anchor_target_speed_limit_mps_by_anchor
            != contact_anchor_tier_speed_limit_mps_by_anchor
        )
        current_contact_clearance_imbalance_m = max(
            gripper_surface_clearance_m_by_anchor.values()
        ) - min(gripper_surface_clearance_m_by_anchor.values())
        max_contact_clearance_imbalance_m = max(
            max_contact_clearance_imbalance_m,
            current_contact_clearance_imbalance_m,
        )

        if phase == Order8NaturalContactPhase.RESET:
            terminal_anchor_targets_base = dict(pregrasp_hold_anchor_poses_base)
            commanded_anchor_targets_base = dict(pregrasp_hold_anchor_poses_base)
        elif phase == Order8NaturalContactPhase.APPROACH:
            # Opening is planned from the selected collision meshes themselves.
            # It can run while the base moves to staging because every selected
            # body is commanded away from the object.
            terminal_anchor_targets_base = dict(opening_plan.anchor_poses_base)
            commanded_anchor_targets_base = {
                anchor_id: _advance_pose_toward(
                    commanded_anchor_targets_base.get(
                        anchor_id,
                        current_anchor_poses_base[anchor_id],
                    ),
                    target_pose,
                    max_translation_step_m=(
                        float(config.anchor_translation_speed_limit_mps) * sim_dt
                    ),
                )
                for anchor_id, target_pose in (terminal_anchor_targets_base.items())
            }
        elif (
            phase == Order8NaturalContactPhase.CONTACT_ACQUISITION
            and grasp_hold_anchor_poses_base is not None
        ):
            # q_close is the first simultaneous measured surface-region
            # configuration.  Hold that achieved geometry absolutely until
            # the later slow positional preload starts.  The offset-torque
            # channel remains zero.
            terminal_anchor_targets_base = dict(grasp_hold_anchor_poses_base)
            commanded_anchor_targets_base = dict(grasp_hold_anchor_poses_base)
        elif phase == Order8NaturalContactPhase.CONTACT_ACQUISITION:
            if pregrasp_open_anchor_poses_base is None:
                raise RuntimeError(
                    "Order8 CONTACT entered without a latched mesh-clear pregrasp"
                )
            if not contact_side_closure_enabled:
                # Keep the open articulated shape while QPID owns the slow
                # axial insertion and its subsequent settle dwell.  Joint IK
                # must not absorb centroidal error or close during that dwell.
                terminal_anchor_targets_base = dict(pregrasp_open_anchor_poses_base)
                commanded_anchor_targets_base = dict(pregrasp_open_anchor_poses_base)
            else:
                if contact_centering_hold_anchor_poses_base is not None:
                    # Recenter the centroidal pose with the complete articulated
                    # shape frozen.  Running the far-side IK concurrently makes
                    # the per-module QPID targets internally oppose the desired
                    # base translation in the connected morphology.
                    terminal_anchor_targets_base = dict(
                        contact_centering_hold_anchor_poses_base
                    )
                    commanded_anchor_targets_base = dict(
                        contact_centering_hold_anchor_poses_base
                    )
                else:
                    # Rebase the measured-object-following mesh-contact targets
                    # through the current centroidal frame.  The actual
                    # authored-mesh sample point, rather than the Dock connect
                    # frame, is softly driven toward the preferred object-face
                    # centre.  The configured +/- tolerance remains a hard
                    # q_close region, not a no-correction deadband.
                    terminal_anchor_targets_base = {
                        anchor_id: compose_pose(
                            inverse_pose(base_root_pose),
                            target_pose_world,
                        )
                        for anchor_id, target_pose_world in (
                            live_contact_region_targets_world.items()
                        )
                    }
                    all_individual_latches_acquired = set(
                        contact_stall_latched_anchor_poses_base
                    ) == set(selected_anchor_ids)
                    if all_individual_latches_acquired:
                        for anchor_id in selected_anchor_ids:
                            if (
                                anchor_id
                                not in contact_reacquired_hold_anchor_poses_base
                                and gripper_surface_clearance_m_by_anchor[anchor_id]
                                <= contact_mesh_clearance_reacquire_threshold_m
                            ):
                                # Snapshot exactly once on entry to the shared
                                # reacquire band.  A fixed measured target gives
                                # the coupled IK a restorative error if motion
                                # of the opposite side perturbs this anchor.
                                contact_reacquired_hold_anchor_poses_base[anchor_id] = (
                                    current_anchor_poses_base[anchor_id]
                                )
                    next_commanded_anchor_targets_base: dict[int, Pose7D] = {}
                    for anchor_id, target_pose in terminal_anchor_targets_base.items():
                        previous_command = commanded_anchor_targets_base.get(
                            anchor_id,
                            current_anchor_poses_base[anchor_id],
                        )
                        next_target = _alternating_reacquire_anchor_target(
                            previous_command=previous_command,
                            terminal_target=target_pose,
                            individual_latched_pose=(
                                compose_pose(
                                    inverse_pose(base_root_pose),
                                    contact_backed_off_anchor_hold_poses_world[
                                        anchor_id
                                    ],
                                )
                                if anchor_id
                                in contact_backed_off_anchor_hold_poses_world
                                else (
                                    None
                                    if anchor_id
                                    not in contact_stall_latched_anchor_poses_world
                                    else compose_pose(
                                        inverse_pose(base_root_pose),
                                        contact_stall_latched_anchor_poses_world[
                                            anchor_id
                                        ],
                                    )
                                )
                            ),
                            reacquired_hold_pose=(
                                contact_reacquired_hold_anchor_poses_base.get(anchor_id)
                            ),
                            all_individual_latches_acquired=(
                                all_individual_latches_acquired
                            ),
                            max_translation_step_m=(
                                contact_anchor_target_speed_limit_mps_by_anchor[
                                    anchor_id
                                ]
                                * sim_dt
                            ),
                        )
                        next_commanded_anchor_targets_base[anchor_id] = next_target
                    commanded_anchor_targets_base = next_commanded_anchor_targets_base
        elif phase == Order8NaturalContactPhase.RELEASE:
            terminal_anchor_targets_base = dict(release_terminal_anchor_targets)
            commanded_anchor_targets_base = {
                anchor_id: _advance_pose_toward(
                    commanded_anchor_targets_base.get(
                        anchor_id,
                        current_anchor_poses_base[anchor_id],
                    ),
                    target_pose,
                    max_translation_step_m=(
                        float(config.anchor_translation_speed_limit_mps) * sim_dt
                    ),
                )
                for anchor_id, target_pose in (terminal_anchor_targets_base.items())
            }
        elif grasp_hold_anchor_poses_base is not None:
            terminal_anchor_targets_base = dict(grasp_hold_anchor_poses_base)
            commanded_anchor_targets_base = dict(grasp_hold_anchor_poses_base)
        else:
            terminal_anchor_targets_base = dict(current_anchor_poses_base)
            commanded_anchor_targets_base = dict(current_anchor_poses_base)

        planner_anchor_references = {
            anchor_id: compose_pose(base_root_pose, pose_base)
            for anchor_id, pose_base in commanded_anchor_targets_base.items()
        }
        terminal_anchor_references = {
            anchor_id: compose_pose(base_root_pose, pose_base)
            for anchor_id, pose_base in terminal_anchor_targets_base.items()
        }
        if phase == Order8NaturalContactPhase.CONTACT_ACQUISITION:
            # Acquisition is position-primary.  All Dock joints may morph to
            # close the two authored mesh surfaces, so a stale pregrasp
            # quaternion must not pull the chain back toward neutral while the
            # normal target advances by only tens of micrometres per tick.
            planner_anchor_references = (
                _contact_anchor_references_with_measured_orientation(
                    planner_anchor_references,
                    measured_selected_anchor_poses_world_by_anchor,
                )
            )

        # The Order 8 closure test driver is joint-space only.  Authored mesh
        # samples remain evidence inputs and never become a receding IK task.
        contact_surface_translation_task_active = False
        current_translation_task_poses_world = {
            anchor_id: (
                (
                    *contact_control_surface_point_world_by_anchor[anchor_id],
                    *measured_selected_anchor_poses_world_by_anchor[anchor_id][3:7],
                )
                if contact_surface_translation_task_active
                else measured_selected_anchor_poses_world_by_anchor[anchor_id]
            )
            for anchor_id in selected_anchor_ids
        }
        desired_translation_task_poses_world = {
            anchor_id: (
                _rigid_point_pose_following_anchor_target(
                    current_anchor_pose_world=(
                        measured_selected_anchor_poses_world_by_anchor[anchor_id]
                    ),
                    current_point_world=(
                        contact_control_surface_point_world_by_anchor[anchor_id]
                    ),
                    desired_anchor_pose_world=planner_anchor_references[anchor_id],
                )
                if contact_surface_translation_task_active
                else planner_anchor_references[anchor_id]
            )
            for anchor_id in selected_anchor_ids
        }
        tentative_command_errors_m = {
            anchor_id: _position_distance(
                current_translation_task_poses_world[anchor_id],
                desired_translation_task_poses_world[anchor_id],
            )
            for anchor_id in selected_anchor_ids
        }
        tentative_terminal_errors_m = {
            anchor_id: _position_distance(
                planner_anchor_references[anchor_id],
                terminal_anchor_references[anchor_id],
            )
            for anchor_id in selected_anchor_ids
        }
        contact_stall_candidates_by_anchor = (
            _selected_anchor_surface_load_settle_candidates(
                selected_anchor_ids,
                object_normal_relative_speed_mps_by_anchor=(
                    current_anchor_object_filtered_normal_relative_speed_mps_by_anchor
                ),
                mesh_clearance_m_by_anchor=gripper_surface_clearance_m_by_anchor,
                selected_joint_load_nm_by_anchor=(
                    contact_stall_joint_load_nm_by_anchor
                ),
                anchor_speed_threshold_mps=float(
                    config.contact_stall_anchor_speed_threshold_mps
                ),
                mesh_clearance_arm_threshold_m=(
                    float(config.contact_surface_arm_clearance_m)
                    + float(config.contact_penetration_noise_floor_m)
                ),
                selected_joint_load_threshold_nm=(
                    contact_stall_selected_joint_load_threshold_nm
                ),
            )
            if (
                phase == Order8NaturalContactPhase.CONTACT_ACQUISITION
                and not contact_configuration_latched
                and contact_side_closure_enabled
                and contact_load_detection_armed
            )
            else {anchor_id: False for anchor_id in selected_anchor_ids}
        )
        contact_surface_load_arrest_candidates = (
            _selected_anchor_surface_load_arrest_candidates(
                selected_anchor_ids,
                mesh_clearance_m_by_anchor=gripper_surface_clearance_m_by_anchor,
                selected_joint_load_nm_by_anchor=(
                    contact_stall_joint_load_nm_by_anchor
                ),
                mesh_clearance_arm_threshold_m=(
                    float(config.contact_surface_arm_clearance_m)
                    + float(config.contact_penetration_noise_floor_m)
                ),
                selected_joint_load_threshold_nm=(
                    contact_stall_selected_joint_load_threshold_nm
                ),
            )
            if (
                phase == Order8NaturalContactPhase.CONTACT_ACQUISITION
                and not contact_configuration_latched
                and contact_side_closure_enabled
                and contact_load_detection_armed
            )
            else {anchor_id: False for anchor_id in selected_anchor_ids}
        )
        last_contact_surface_load_arrest_candidates = dict(
            contact_surface_load_arrest_candidates
        )
        if (
            phase == Order8NaturalContactPhase.CONTACT_ACQUISITION
            and not contact_configuration_latched
        ):
            for anchor_id in selected_anchor_ids:
                last_contact_stall_command_error_m_by_anchor[anchor_id] = (
                    tentative_command_errors_m[anchor_id]
                )
                last_contact_stall_anchor_speed_mps_by_anchor[anchor_id] = (
                    current_anchor_object_relative_speed_mps_by_anchor[anchor_id]
                )
                last_contact_stall_selected_joint_load_nm_by_anchor[anchor_id] = (
                    contact_stall_joint_load_nm_by_anchor[anchor_id]
                )
                if contact_stall_candidates_by_anchor[anchor_id]:
                    nonprivileged_contact_stall_dwell_s_by_anchor[anchor_id] += sim_dt
                else:
                    nonprivileged_contact_stall_dwell_s_by_anchor[anchor_id] = 0.0

            nonprivileged_contact_stall_dwell_s = min(
                nonprivileged_contact_stall_dwell_s_by_anchor.values(),
                default=0.0,
            )
            # q_close is an arrest event, not the later stable-grasp proof.
            # Still, a single-cycle graph-constraint or drive transient must
            # not disable centroidal pose control and freeze a false contact.
            # Require the simultaneous proximity/load signature for the
            # existing short contact-stall dwell.  At the bounded 2 mm/s creep
            # rate this adds at most 0.2 mm of commanded overtravel.
            simultaneous_region_arrest_candidate = bool(
                all(contact_surface_load_arrest_candidates.values())
            )
            if simultaneous_region_arrest_candidate:
                nonprivileged_contact_configuration_dwell_s += sim_dt
            else:
                nonprivileged_contact_configuration_dwell_s = 0.0
            if (
                nonprivileged_contact_configuration_dwell_s + 1.0e-12
                >= float(config.contact_stall_dwell_s)
            ):
                # A one-sided provisional contact remains free to separate and
                # reacquire.  Only a simultaneous two-sided surface/load event
                # snapshots the complete measured articulated configuration.
                contact_stall_latched_anchor_poses_base = dict(
                    current_anchor_poses_base
                )
                contact_stall_latched_anchor_poses_world = dict(
                    measured_selected_anchor_poses_world_by_anchor
                )
                contact_stall_latched_mesh_clearance_m_by_anchor = dict(
                    gripper_surface_clearance_m_by_anchor
                )
                contact_stall_latched_tangential_offset_m_by_anchor = dict(
                    current_contact_tangential_offset_m_by_anchor
                )
                contact_stall_latched = True
                contact_configuration_latched = True
                contact_configuration_latched_time_s = current_time_s
                contact_closure_reason = "dynamic_simultaneous_surface_region_arrest"
                joint_position_reference_by_id = {
                    joint_id: float(position)
                    for joint_id, position in zip(
                        joint_vector.joint_ids,
                        joint_vector.positions_rad,
                        strict=True,
                    )
                }
                contact_individual_latch_hold_base_pose = base_root_pose
                contact_centering_target_base_pose = base_root_pose
                commanded_base_target = base_root_pose
                desired_body_pose_by_phase[
                    Order8NaturalContactPhase.CONTACT_ACQUISITION
                ] = base_root_pose
                desired_body_linear_velocity_by_phase[
                    Order8NaturalContactPhase.CONTACT_ACQUISITION
                ] = (0.0, 0.0, 0.0)
                for qpid in (contact_centering_controller, controller):
                    qpid.reset_integrators()
                qclose_base_pose_snapshot = base_root_pose
                qclose_joint_positions_snapshot = {
                    joint_id: float(position)
                    for joint_id, position in zip(
                        joint_vector.joint_ids,
                        joint_vector.positions_rad,
                        strict=True,
                    )
                }
                qclose_object_pose_snapshot = tuple(object_state["pose"])
                qclose_checkpoint_state_snapshot = _qclose_checkpoint_state_to_dict(
                    _QCloseCheckpointState(
                        module_root_poses={
                            module_id: tuple(_tensor_row(robot.data.root_pose_w))
                            for module_id, robot in robots.items()
                        },
                        module_root_velocities={
                            module_id: tuple(_tensor_row(robot.data.root_vel_w))
                            for module_id, robot in robots.items()
                        },
                        joint_positions_rad=dict(qclose_joint_positions_snapshot),
                        joint_velocities_radps={
                            joint_id: float(velocity)
                            for joint_id, velocity in zip(
                                joint_vector.joint_ids,
                                joint_vector.velocities_radps,
                                strict=True,
                            )
                        },
                        object_pose=tuple(object_state["pose"]),
                        object_twist=tuple(object_state["twist"]),
                        anchor_hold_poses_base=dict(current_anchor_poses_base),
                    )
                )
                if contact_axial_hold_base_pose is None:
                    raise RuntimeError(
                        "Order8 contact configuration latched without axial hold"
                    )
                latched_contact_centering_offset_world = tuple(
                    float(base_root_pose[index])
                    - float(contact_axial_hold_base_pose[index])
                    for index in range(3)
                )

        if (
            phase == Order8NaturalContactPhase.CONTACT_ACQUISITION
            and contact_configuration_latched
            and grasp_hold_anchor_poses_base is None
        ):
            # Latch the achieved q_close geometry.  Joint position and offset
            # torque remain independent MIT-mode command channels: position
            # holds the measured contact shape and torque supplies the wrench.
            grasp_hold_anchor_poses_base = dict(current_anchor_poses_base)
            joint_position_reference_by_id = {
                joint_id: float(position)
                for joint_id, position in zip(
                    joint_vector.joint_ids,
                    joint_vector.positions_rad,
                    strict=True,
                )
            }
            planner_anchor_references = {
                anchor_id: compose_pose(base_root_pose, pose_base)
                for anchor_id, pose_base in commanded_anchor_targets_base.items()
            }
            terminal_anchor_references = {
                anchor_id: compose_pose(base_root_pose, pose_base)
                for anchor_id, pose_base in grasp_hold_anchor_poses_base.items()
            }

        if (
            phase == Order8NaturalContactPhase.CONTACT_ACQUISITION
            and contact_configuration_latched
            and not post_qclose_joint_settle_complete
        ):
            # q_close starts as a simultaneous surface/load arrest event.  The
            # mechanism is still moving at that instant, so immediately
            # freezing the absolute positions turns the residual kinetic
            # motion into a large position-servo/contact impulse.  During this
            # short acquisition-only settle window, continuously rebase the
            # absolute position channel to the measured q while retaining a
            # zero velocity target.  Centroidal P/I remains yielded and the
            # offset-torque channel stays at zero.
            joint_position_reference_by_id = {
                joint_id: float(position)
                for joint_id, position in zip(
                    joint_vector.joint_ids,
                    joint_vector.positions_rad,
                    strict=True,
                )
            }
            grasp_hold_anchor_poses_base = dict(current_anchor_poses_base)
            commanded_anchor_targets_base = dict(current_anchor_poses_base)
            planner_anchor_references = {
                anchor_id: compose_pose(base_root_pose, pose_base)
                for anchor_id, pose_base in current_anchor_poses_base.items()
            }
            terminal_anchor_references = dict(planner_anchor_references)
            post_qclose_position_rebase_step_count += 1
            current_max_joint_speed_radps = max(
                (abs(float(value)) for value in joint_vector.velocities_radps),
                default=0.0,
            )
            post_qclose_max_measured_joint_speed_radps = max(
                post_qclose_max_measured_joint_speed_radps,
                current_max_joint_speed_radps,
            )
            post_qclose_surface_motion_settled = _contact_force_hold_settled(
                current_filtered_gripper_surface_clearance_speed_mps_by_anchor,
                selected_anchor_ids=selected_anchor_ids,
                speed_threshold_mps=float(
                    config.contact_stall_anchor_speed_threshold_mps
                ),
            )
            post_qclose_surface_region_retained = all(
                gripper_surface_clearance_m_by_anchor[anchor_id]
                <= float(config.contact_surface_arm_clearance_m)
                + float(config.contact_closure_inward_overtravel_m)
                for anchor_id in selected_anchor_ids
            )
            if (
                current_max_joint_speed_radps
                <= post_qclose_joint_speed_threshold_radps + 1.0e-12
                and base_linear_speed_mps
                <= float(config.pregrasp_linear_speed_tolerance_mps) + 1.0e-12
                and post_qclose_surface_motion_settled
                and post_qclose_surface_region_retained
            ):
                post_qclose_joint_settle_dwell_s += sim_dt
            else:
                post_qclose_joint_settle_dwell_s = 0.0
            if (
                post_qclose_joint_settle_dwell_s + 1.0e-12
                >= float(config.contact_stall_dwell_s)
            ):
                post_qclose_joint_settle_complete = True
                # The settled measured state, rather than the first arrest
                # sample, is the q_close used to initialize positional preload.
                qclose_base_pose_snapshot = base_root_pose
                qclose_joint_positions_snapshot = dict(
                    joint_position_reference_by_id
                )
                qclose_object_pose_snapshot = tuple(object_state["pose"])
                qclose_checkpoint_state_snapshot = _qclose_checkpoint_state_to_dict(
                    _QCloseCheckpointState(
                        module_root_poses={
                            module_id: tuple(_tensor_row(robot.data.root_pose_w))
                            for module_id, robot in robots.items()
                        },
                        module_root_velocities={
                            module_id: tuple(_tensor_row(robot.data.root_vel_w))
                            for module_id, robot in robots.items()
                        },
                        joint_positions_rad=dict(qclose_joint_positions_snapshot),
                        joint_velocities_radps={
                            joint_id: float(velocity)
                            for joint_id, velocity in zip(
                                joint_vector.joint_ids,
                                joint_vector.velocities_radps,
                                strict=True,
                            )
                        },
                        object_pose=tuple(object_state["pose"]),
                        object_twist=tuple(object_state["twist"]),
                        anchor_hold_poses_base=dict(current_anchor_poses_base),
                    )
                )

        post_qclose_geometric_preload_active = bool(
            phase == Order8NaturalContactPhase.CONTACT_ACQUISITION
            and contact_configuration_latched
            and post_qclose_joint_settle_complete
            and not post_qclose_geometric_preload_complete
        )
        post_qclose_geometric_preload_active_step_count += int(
            post_qclose_geometric_preload_active
        )
        if post_qclose_geometric_preload_active:
            if not post_qclose_geometric_preload_surface_point_local_by_anchor:
                post_qclose_geometric_preload_surface_point_local_by_anchor = {
                    anchor_id: compose_pose(
                        inverse_pose(
                            selected_gripper_body_poses[
                                selected_gripper_body_key_by_anchor[anchor_id]
                            ]
                        ),
                        (
                            *contact_control_surface_point_world_by_anchor[
                                anchor_id
                            ],
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                        ),
                    )[:3]
                    for anchor_id in selected_anchor_ids
                }
                post_qclose_geometric_preload_current_surface_point_world_by_anchor = {
                    anchor_id: compose_pose(
                        selected_gripper_body_poses[
                            selected_gripper_body_key_by_anchor[anchor_id]
                        ],
                        (*point_local, 0.0, 0.0, 0.0, 1.0),
                    )[:3]
                    for anchor_id, point_local in (
                        post_qclose_geometric_preload_surface_point_local_by_anchor.items()
                    )
                }
                post_qclose_geometric_preload_initial_surface_point_object_by_anchor = {
                    anchor_id: compose_pose(
                        inverse_pose(tuple(object_state["pose"])),
                        (
                            *post_qclose_geometric_preload_current_surface_point_world_by_anchor[
                                anchor_id
                            ],
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                        ),
                    )[:3]
                    for anchor_id in selected_anchor_ids
                }
            if not post_qclose_geometric_preload_anchor_poses_object:
                post_qclose_geometric_preload_anchor_poses_object = {
                    anchor_id: _object_relative_inward_preload_pose(
                        anchor_pose_world=(
                            *post_qclose_geometric_preload_current_surface_point_world_by_anchor[
                                anchor_id
                            ],
                            *measured_selected_anchor_poses_world_by_anchor[
                                anchor_id
                            ][3:7],
                        ),
                        object_pose_world=tuple(object_state["pose"]),
                        inward_normal_object=(
                            contact_inward_normal_object_by_anchor[anchor_id]
                        ),
                        preload_distance_m=float(
                            config.contact_closure_inward_overtravel_m
                        ),
                    )
                    for anchor_id in selected_anchor_ids
                }
                post_qclose_geometric_preload_commanded_anchor_targets_world = {
                    anchor_id: (
                        *post_qclose_geometric_preload_current_surface_point_world_by_anchor[
                            anchor_id
                        ],
                        *measured_selected_anchor_poses_world_by_anchor[anchor_id][
                            3:7
                        ],
                    )
                    for anchor_id in selected_anchor_ids
                }
            live_preload_terminal_targets_world = {
                anchor_id: compose_pose(
                    tuple(object_state["pose"]),
                    post_qclose_geometric_preload_anchor_poses_object[anchor_id],
                )
                for anchor_id in selected_anchor_ids
            }
            post_qclose_geometric_preload_commanded_anchor_targets_world = {
                anchor_id: _advance_pose_toward(
                    post_qclose_geometric_preload_commanded_anchor_targets_world[
                        anchor_id
                    ],
                    live_preload_terminal_targets_world[anchor_id],
                    max_translation_step_m=(
                        float(config.contact_surface_creep_speed_limit_mps)
                        * sim_dt
                    ),
                )
                for anchor_id in selected_anchor_ids
            }
            commanded_anchor_targets_base = {
                anchor_id: compose_pose(
                    inverse_pose(base_root_pose),
                    target_pose,
                )
                for anchor_id, target_pose in (
                    post_qclose_geometric_preload_commanded_anchor_targets_world.items()
                )
            }
            planner_anchor_references = dict(
                post_qclose_geometric_preload_commanded_anchor_targets_world
            )
            terminal_anchor_references = dict(
                live_preload_terminal_targets_world
            )
            post_qclose_geometric_preload_terminal_error_m = max(
                (
                    _position_distance(
                        post_qclose_geometric_preload_commanded_anchor_targets_world[
                            anchor_id
                        ],
                        live_preload_terminal_targets_world[anchor_id],
                    )
                    for anchor_id in selected_anchor_ids
                ),
                default=math.inf,
            )
            post_qclose_geometric_preload_tracking_error_m = max(
                (
                    _position_distance(
                        (
                            *post_qclose_geometric_preload_current_surface_point_world_by_anchor[
                                anchor_id
                            ],
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                        ),
                        post_qclose_geometric_preload_commanded_anchor_targets_world[
                            anchor_id
                        ],
                    )
                    for anchor_id in selected_anchor_ids
                ),
                default=math.inf,
            )
            post_qclose_geometric_preload_achieved_inward_displacement_m_by_anchor = {
                anchor_id: sum(
                    (
                        float(current_point_object[index])
                        - float(
                            post_qclose_geometric_preload_initial_surface_point_object_by_anchor[
                                anchor_id
                            ][index]
                        )
                    )
                    * float(contact_inward_normal_object_by_anchor[anchor_id][index])
                    for index in range(3)
                )
                for anchor_id, current_point_object in (
                    (
                        anchor_id,
                        compose_pose(
                            inverse_pose(tuple(object_state["pose"])),
                            (
                                *post_qclose_geometric_preload_current_surface_point_world_by_anchor[
                                    anchor_id
                                ],
                                0.0,
                                0.0,
                                0.0,
                                1.0,
                            ),
                        )[:3],
                    )
                    for anchor_id in selected_anchor_ids
                )
            }
            current_max_joint_speed_radps = max(
                (abs(float(value)) for value in joint_vector.velocities_radps),
                default=0.0,
            )
            preload_motion_settled = _contact_force_hold_settled(
                current_anchor_object_filtered_normal_relative_speed_mps_by_anchor,
                selected_anchor_ids=selected_anchor_ids,
                speed_threshold_mps=float(
                    config.contact_stall_anchor_speed_threshold_mps
                ),
            )
            post_qclose_geometric_preload_load_arrest_candidates = (
                _selected_anchor_surface_load_arrest_candidates(
                    selected_anchor_ids,
                    mesh_clearance_m_by_anchor=(
                        gripper_surface_clearance_m_by_anchor
                    ),
                    selected_joint_load_nm_by_anchor=(
                        contact_stall_joint_load_nm_by_anchor
                    ),
                    mesh_clearance_arm_threshold_m=(
                        float(config.contact_surface_arm_clearance_m)
                        + float(config.contact_penetration_noise_floor_m)
                    ),
                    selected_joint_load_threshold_nm=(
                        contact_stall_selected_joint_load_threshold_nm
                    ),
                )
            )
            preload_geometry_or_load_acquired = bool(
                post_qclose_geometric_preload_tracking_error_m
                <= float(config.anchor_reference_terminal_tolerance_m)
                or all(
                    post_qclose_geometric_preload_load_arrest_candidates.values()
                )
            )
            if (
                post_qclose_geometric_preload_terminal_error_m <= 1.0e-9
                and preload_geometry_or_load_acquired
                and current_max_joint_speed_radps
                <= post_qclose_joint_speed_threshold_radps + 1.0e-12
                and base_linear_speed_mps
                <= float(config.pregrasp_linear_speed_tolerance_mps) + 1.0e-12
                and preload_motion_settled
            ):
                post_qclose_geometric_preload_settle_dwell_s += sim_dt
            else:
                post_qclose_geometric_preload_settle_dwell_s = 0.0
            if (
                post_qclose_geometric_preload_settle_dwell_s + 1.0e-12
                >= float(config.contact_stall_dwell_s)
            ):
                post_qclose_geometric_preload_complete = True
                post_qclose_geometric_preload_active = False
                post_qclose_geometric_preload_completion_source = (
                    "simultaneous_nonprivileged_surface_load_arrest"
                    if all(
                        post_qclose_geometric_preload_load_arrest_candidates.values()
                    )
                    else "measured_mesh_material_point_tracking"
                )
                contact_closure_reason = (
                    "dynamic_simultaneous_surface_region_arrest_then_"
                    "object_relative_geometric_preload"
                )
                grasp_hold_anchor_poses_base = dict(current_anchor_poses_base)
                commanded_anchor_targets_base = dict(current_anchor_poses_base)
                planner_anchor_references = {
                    anchor_id: compose_pose(base_root_pose, pose_base)
                    for anchor_id, pose_base in current_anchor_poses_base.items()
                }
                terminal_anchor_references = dict(planner_anchor_references)
                qclose_base_pose_snapshot = base_root_pose
                qclose_joint_positions_snapshot = {
                    joint_id: float(position)
                    for joint_id, position in zip(
                        joint_vector.joint_ids,
                        joint_vector.positions_rad,
                        strict=True,
                    )
                }
                qclose_object_pose_snapshot = tuple(object_state["pose"])
                qclose_checkpoint_state_snapshot = _qclose_checkpoint_state_to_dict(
                    _QCloseCheckpointState(
                        module_root_poses={
                            module_id: tuple(_tensor_row(robot.data.root_pose_w))
                            for module_id, robot in robots.items()
                        },
                        module_root_velocities={
                            module_id: tuple(_tensor_row(robot.data.root_vel_w))
                            for module_id, robot in robots.items()
                        },
                        joint_positions_rad=dict(qclose_joint_positions_snapshot),
                        joint_velocities_radps={
                            joint_id: float(velocity)
                            for joint_id, velocity in zip(
                                joint_vector.joint_ids,
                                joint_vector.velocities_radps,
                                strict=True,
                            )
                        },
                        object_pose=tuple(object_state["pose"]),
                        object_twist=tuple(object_state["twist"]),
                        anchor_hold_poses_base=dict(current_anchor_poses_base),
                    )
                )

        if (
            contact_yield_grasp_pose_rebased
            and contact_yield_blend <= 1.0e-12
            and contact_yield_joint_drive_blend <= 1.0e-12
            and not contact_joint_drive_damping_scheduled
        ):
            contact_joint_drive_damping_targets = _schedule_contact_joint_drive_damping(
                robots,
                expected_joint_ids,
                nominal_damping_nms_per_rad=float(dock_damping),
                damping_multiplier=float(config.contact_joint_drive_damping_multiplier),
                maximum_damping_nms_per_rad=(
                    ORDER8_SIMULATION_DRIVE_DAMPING_MAX_NMS_PER_RAD
                ),
            )
            contact_joint_drive_damping_scheduled = True

        max_anchor_position_error_m = max(
            (
                _position_distance(
                    (
                        (
                            *post_qclose_geometric_preload_current_surface_point_world_by_anchor[
                                anchor_id
                            ],
                            *measured_selected_anchor_poses_world_by_anchor[
                                anchor_id
                            ][3:7],
                        )
                        if (
                            contact_surface_translation_task_active
                            or post_qclose_geometric_preload_active
                        )
                        else measured_selected_anchor_poses_world_by_anchor[anchor_id]
                    ),
                    (
                        target_pose
                        if post_qclose_geometric_preload_active
                        else (
                            _rigid_point_pose_following_anchor_target(
                                current_anchor_pose_world=(
                                    measured_selected_anchor_poses_world_by_anchor[
                                        anchor_id
                                    ]
                                ),
                                current_point_world=(
                                    contact_control_surface_point_world_by_anchor[
                                        anchor_id
                                    ]
                                ),
                                desired_anchor_pose_world=target_pose,
                            )
                            if contact_surface_translation_task_active
                            else target_pose
                        )
                    ),
                )
                for anchor_id, target_pose in planner_anchor_references.items()
            ),
            default=math.inf,
        )
        max_anchor_reference_terminal_error_m = max(
            (
                _position_distance(
                    planner_anchor_references[anchor_id],
                    terminal_pose,
                )
                for anchor_id, terminal_pose in terminal_anchor_references.items()
            ),
            default=math.inf,
        )
        anchor_command_tracking_complete = bool(
            max_anchor_position_error_m
            <= float(config.anchor_command_tracking_tolerance_m)
            and max_anchor_reference_terminal_error_m
            <= float(config.anchor_reference_terminal_tolerance_m)
        )
        minimum_achieved_pregrasp_clearance_m = max(
            0.0,
            float(config.pregrasp_mesh_clearance_m)
            - float(config.anchor_command_tracking_tolerance_m),
        )
        achieved_endpoint_reachability = bool(
            phase == Order8NaturalContactPhase.APPROACH
            and anchor_command_tracking_complete
            and pregrasp_achieved_mesh_clearance_m
            >= minimum_achieved_pregrasp_clearance_m
        )
        if phase == Order8NaturalContactPhase.APPROACH:
            differential_reachability = bool(last_control_result.reachability.passed)
            pregrasp_reachability_gate_passed = bool(
                differential_reachability or achieved_endpoint_reachability
            )
            if differential_reachability and achieved_endpoint_reachability:
                pregrasp_reachability_gate_source = (
                    "differential_and_achieved_mesh_clear_endpoint"
                )
            elif differential_reachability:
                pregrasp_reachability_gate_source = (
                    "differential_whole_structure_jacobian"
                )
            elif achieved_endpoint_reachability:
                pregrasp_reachability_gate_source = "achieved_mesh_clear_endpoint"
            else:
                pregrasp_reachability_gate_source = "not_reachable"
        pregrasp_aligned = bool(
            pregrasp_base_aligned
            and anchor_command_tracking_complete
            and pregrasp_achieved_mesh_clearance_m
            >= minimum_achieved_pregrasp_clearance_m
            and pregrasp_reachability_gate_passed
        )
        if (
            phase == Order8NaturalContactPhase.APPROACH
            and pregrasp_aligned
            and pregrasp_open_anchor_poses_base is None
        ):
            # Latch the achieved, measured open morphology.  This is the q_open
            # analogue: it is geometry- and state-dependent, not a binary angle.
            pregrasp_open_anchor_poses_base = dict(current_anchor_poses_base)
            commanded_anchor_targets_base = dict(pregrasp_open_anchor_poses_base)
            # q_open is the achieved, measured articulated configuration, not
            # the ideal mesh-opening IK endpoint.  The direct release command
            # returns to the joint positions captured at this same instant;
            # its terminal/dwell gate must therefore use the matching measured
            # anchor poses.  Mixing the ideal opening plan with measured q_open
            # can leave a permanent multi-centimetre error even after every
            # joint has returned correctly.
            release_terminal_anchor_targets = dict(
                pregrasp_open_anchor_poses_base
            )

        contact_position_preload_active = bool(
            phase == Order8NaturalContactPhase.CONTACT_ACQUISITION
            and contact_configuration_latched
            and post_qclose_joint_settle_complete
            and not contact_position_preload_complete
        )
        contact_position_preload_active_step_count += int(
            contact_position_preload_active
        )
        if contact_position_preload_active:
            if not simple_closure_velocity_targets_radps:
                raise RuntimeError(
                    "Order8 position preload requires the fixed closure direction"
                )
            if not contact_position_preload_joint_ids_by_anchor:
                contact_position_preload_joint_ids_by_anchor = (
                    _position_preload_joint_ids_by_anchor(
                        ordered_joint_ids=(
                            last_kinematics.ordered_global_dock_joint_ids
                        ),
                        closure_velocity_targets_radps=(
                            simple_closure_velocity_targets_radps
                        ),
                        influential_joint_ids_by_anchor=(
                            contact_stall_influential_joint_ids_by_anchor
                        ),
                        fixed_joint_ids=diagnostic_pitch_hold_positions_rad,
                    )
                )
            if not contact_position_preload_position_targets_rad:
                # Start from the settled measured q_close hold.  Subsequent
                # targets integrate only from this previous absolute target;
                # measured-q rebasing would erase the load-producing lead.
                contact_position_preload_position_targets_rad = dict(
                    joint_position_reference_by_id
                )
            for anchor_id in selected_anchor_ids:
                side_load_nm = max(
                    applied_dock_load_nm_by_joint[joint_id]
                    for joint_id in contact_position_preload_joint_ids_by_anchor[
                        anchor_id
                    ]
                )
                contact_position_preload_load_nm_by_anchor[anchor_id] = side_load_nm
                contact_position_preload_max_load_nm_by_anchor[anchor_id] = max(
                    contact_position_preload_max_load_nm_by_anchor[anchor_id],
                    side_load_nm,
                )
                if anchor_id in contact_position_preload_frozen_anchor_ids:
                    continue
                if (
                    side_load_nm + 1.0e-12
                    >= contact_position_preload_load_threshold_nm
                ):
                    contact_position_preload_load_dwell_s_by_anchor[
                        anchor_id
                    ] += sim_dt
                else:
                    contact_position_preload_load_dwell_s_by_anchor[anchor_id] = 0.0
                if (
                    contact_position_preload_load_dwell_s_by_anchor[anchor_id]
                    + 1.0e-12
                    >= float(config.contact_stall_dwell_s)
                ):
                    contact_position_preload_frozen_anchor_ids.add(anchor_id)
                    contact_position_preload_frozen_time_s_by_anchor[
                        anchor_id
                    ] = current_time_s
            if set(contact_position_preload_frozen_anchor_ids) == set(
                selected_anchor_ids
            ):
                contact_position_preload_complete = True
                contact_position_preload_active = False
                contact_position_preload_velocity_targets_radps = {
                    joint_id: 0.0
                    for joint_id in last_kinematics.ordered_global_dock_joint_ids
                }
                contact_position_preload_completion_source = (
                    "per_anchor_damping_compensated_moving_chain_load_dwell"
                )
                contact_closure_reason = (
                    "dynamic_simultaneous_surface_region_arrest_then_"
                    "load_limited_position_preload"
                )

        for anchor_id in selected_anchor_ids:
            if (
                phase == Order8NaturalContactPhase.CONTACT_ACQUISITION
                and contact_configuration_latched
                and post_qclose_joint_settle_complete
            ):
                nonprivileged_contact_force_ramp_elapsed_s_by_anchor[
                    anchor_id
                ] += sim_dt
            elif phase == Order8NaturalContactPhase.CONTACT_ACQUISITION:
                nonprivileged_contact_force_ramp_elapsed_s_by_anchor[anchor_id] = 0.0
        unmasked_contact_force_scale_by_anchor = {}
        for anchor_id in selected_anchor_ids:
            if phase in {
                Order8NaturalContactPhase.LIFT,
                Order8NaturalContactPhase.TRANSPORT,
                Order8NaturalContactPhase.PLACE,
            }:
                scale = 1.0
            elif phase == Order8NaturalContactPhase.CONTACT_ACQUISITION:
                if anchor_id in contact_position_preload_frozen_anchor_ids:
                    scale = 1.0
                else:
                    scale = _clip(
                        contact_position_preload_load_nm_by_anchor[anchor_id]
                        / contact_position_preload_load_threshold_nm,
                        0.0,
                        1.0,
                    )
            else:
                scale = 0.0
            unmasked_contact_force_scale_by_anchor[anchor_id] = scale
        contact_force_scale_by_anchor = {
            anchor_id: (
                unmasked_contact_force_scale_by_anchor[anchor_id]
                if anchor_id in force_ramp_anchor_ids
                else 0.0
            )
            for anchor_id in selected_anchor_ids
        }
        max_contact_force_scale_by_anchor = {
            anchor_id: max(
                max_contact_force_scale_by_anchor[anchor_id],
                contact_force_scale_by_anchor[anchor_id],
            )
            for anchor_id in selected_anchor_ids
        }
        contact_force_scale = min(
            (
                contact_force_scale_by_anchor[anchor_id]
                for anchor_id in force_ramp_anchor_ids
            ),
            default=0.0,
        )
        all_individual_latches_acquired = set(
            contact_stall_latched_anchor_poses_base
        ) == set(selected_anchor_ids)
        all_reacquired_holds_acquired = set(
            contact_reacquired_hold_anchor_poses_base
        ) == set(selected_anchor_ids)
        anchor_pose_priority_by_id = {
            anchor_id: _contact_anchor_pose_priority(
                phase=phase,
                contact_configuration_latched=contact_configuration_latched,
                anchor_individually_latched=(
                    anchor_id in contact_stall_latched_anchor_poses_base
                ),
                all_individual_latches_acquired=(all_individual_latches_acquired),
                anchor_reacquired=(
                    anchor_id in contact_reacquired_hold_anchor_poses_base
                ),
                all_reacquired_holds_acquired=(all_reacquired_holds_acquired),
            )
            for anchor_id in selected_anchor_ids
        }
        if contact_command_ready:
            nonprivileged_contact_command_dwell_s += sim_dt
        elif phase == Order8NaturalContactPhase.CONTACT_ACQUISITION:
            nonprivileged_contact_command_dwell_s = 0.0
        nominal_contact_command_dwell_complete = bool(
            phase == Order8NaturalContactPhase.CONTACT_ACQUISITION
            and contact_command_ready
            and nonprivileged_contact_command_dwell_s >= float(config.contact_dwell_s)
        )
        prelift_relative_motion_settle_achieved = bool(
            prelift_relative_motion_settle_achieved
            or nominal_contact_command_dwell_complete
        )
        contact_required_motion_safety_authorized = (
            _contact_required_motion_safety_authorized(
                nominal_command_dwell_complete=(nominal_contact_command_dwell_complete),
                privileged_grasp_dwell_acquired=bool(last_evidence.grasp_acquired),
            )
        )
        contact_motion_safety_interlock_blocked_step_count += int(
            nominal_contact_command_dwell_complete
            and not contact_required_motion_safety_authorized
        )
        if (
            phase == Order8NaturalContactPhase.RELEASE
            and anchor_command_tracking_complete
        ):
            nonprivileged_release_command_dwell_s += sim_dt
        elif phase == Order8NaturalContactPhase.RELEASE:
            nonprivileged_release_command_dwell_s = 0.0
        phase_elapsed_s = max(0.0, current_time_s - phase_started_s)
        last_base_terminal_tracking_error_m = _position_distance(
            nominal_base_target,
            base_root_pose,
        )
        last_base_command_tracking_error_m = _position_distance(
            commanded_base_target,
            base_root_pose,
        )
        object_speed = _norm(object_state["twist"][:3])
        object_angular_speed = _norm(object_state["twist"][3:])
        if (
            phase == Order8NaturalContactPhase.SETTLE
            and object_speed <= float(config.settle_linear_speed_mps)
            and object_angular_speed <= float(config.settle_angular_speed_rad_s)
            and gripper_clearance >= float(config.gripper_retreat_clearance_m)
        ):
            nonprivileged_settle_dwell_s += sim_dt
        else:
            nonprivileged_settle_dwell_s = 0.0
        place_object_target = _offset_pose(
            object_pose,
            dx=config.required_transport_distance_m,
        )
        planner.observe(
            NaturalContactPlannerFeedback(
                time_s=current_time_s,
                hover_ready=True,
                simultaneous_reachability_passed=(pregrasp_reachability_gate_passed),
                pregrasp_aligned=pregrasp_aligned,
                contact_command_dwell_complete=(
                    contact_required_motion_safety_authorized
                ),
                lift_clearance_reached=(
                    object_bottom_clearance >= float(config.minimum_lift_clearance_m)
                ),
                transport_distance_reached=(
                    transport_distance >= float(config.required_transport_distance_m)
                ),
                intended_place_pose_reached=(
                    _position_distance(object_state["pose"], place_object_target)
                    <= 0.05
                ),
                release_command_dwell_complete=(
                    phase == Order8NaturalContactPhase.RELEASE
                    and nonprivileged_release_command_dwell_s
                    >= float(config.release_contact_free_dwell_s)
                ),
                retreat_clearance_reached=(
                    gripper_clearance >= float(config.gripper_retreat_clearance_m)
                ),
                post_release_settle_complete=(
                    nonprivileged_settle_dwell_s
                    >= float(config.post_release_settle_dwell_s)
                ),
                desired_body_pose_by_phase=desired_body_pose_by_phase,
                desired_body_linear_velocity_by_phase=(
                    desired_body_linear_velocity_by_phase
                ),
                desired_anchor_pose_by_id=planner_anchor_references,
                contact_force_scale=contact_force_scale,
                contact_force_scale_by_anchor_id=(contact_force_scale_by_anchor),
                anchor_pose_priority_by_id=anchor_pose_priority_by_id,
                desired_object_pose_by_phase={
                    Order8NaturalContactPhase.LIFT: _offset_pose(object_pose, dz=0.15),
                    Order8NaturalContactPhase.TRANSPORT: _offset_pose(
                        object_pose,
                        dx=config.required_transport_distance_m,
                        dz=0.15,
                    ),
                    Order8NaturalContactPhase.PLACE: place_object_target,
                },
            )
        )
        if last_evidence.hard_failure:
            request_safe_hold_or_record(
                time_s=current_time_s,
                reason=",".join(last_evidence.failure_reasons)
                or "raw_monitor_hard_failure",
            )
        if order9_teacher_collector is not None:
            teacher_actor_observation, teacher_reward_observation = (
                order9_teacher_observations(
                    evidence=last_evidence,
                    raw_contact_valid=bool(
                        observation.raw_contact_valid
                        and not observation.raw_contact_saturated
                    ),
                    phase_transitioned_now=planner.phase != phase,
                )
            )
            order9_teacher_collector.observe_state(
                actor_observation=teacher_actor_observation,
                reward_observation=teacher_reward_observation,
            )
        trajectory = planner.plan(policy_context)
        if planner.phase == Order8NaturalContactPhase.APPROACH:
            contact_motion_subphase = "mesh_open_staging"
        elif planner.phase == Order8NaturalContactPhase.CONTACT_ACQUISITION:
            if not contact_axial_aligned:
                contact_motion_subphase = "axial_insert_open_shape"
            elif not contact_side_closure_enabled:
                contact_motion_subphase = "axial_settle_open_shape"
            elif not contact_mesh_precenter_complete:
                contact_motion_subphase = "authored_mesh_tangential_precenter_clear"
            elif not contact_configuration_latched:
                contact_motion_subphase = (
                    "provisional_surface_load_settle"
                    if contact_provisional_surface_settle_active
                    else "object_follow_surface_region_joint_close"
                )
            elif not post_qclose_joint_settle_complete:
                contact_motion_subphase = "measured_qclose_velocity_settle"
            elif not contact_position_preload_complete:
                contact_motion_subphase = "load_limited_position_preload"
            else:
                contact_motion_subphase = "frozen_position_preload_settle"
        else:
            contact_motion_subphase = "inactive"
        progress_message = (
            format_order8_progress(planner.phase.value, current_time_s)
            + f" contact_motion={contact_motion_subphase}"
            + f" base_terminal_error={last_base_terminal_tracking_error_m:.4f}m"
            + f" base_command_error={last_base_command_tracking_error_m:.4f}m"
            + f" base_speed={base_linear_speed_mps:.4f}mps"
            + f" lift_transition_stage={diagnostic_lift_transition_stage}"
            + " reachability="
            + ("pass" if pregrasp_reachability_gate_passed else "fail")
            + " differential_reachability="
            + ("pass" if last_control_result.reachability.passed else "fail")
            + f" reachability_residual={last_control_result.reachability.relative_residual:.4f}"
            + f" achieved_mesh_clearance={pregrasp_achieved_mesh_clearance_m:.4f}m"
            + f" axial_mesh_overlap={gripper_axial_overlap_m:.4f}m"
            + f" axial_settle_dwell={contact_axial_settle_dwell_s:.3f}s"
            + " mesh_precenter="
            + ("complete" if contact_mesh_precenter_complete else "seeking")
            + f" mesh_precenter_dwell={contact_mesh_precenter_dwell_s:.3f}s"
            + f" anchor_error={max_anchor_position_error_m:.4f}m"
            + f" anchor_terminal_error={max_anchor_reference_terminal_error_m:.4f}m"
            + " contact_configuration="
            + (contact_closure_reason or "seeking")
            + f" contact_stall_dwell={nonprivileged_contact_stall_dwell_s:.3f}s"
            + " contact_configuration_dwell="
            + f"{nonprivileged_contact_configuration_dwell_s:.3f}s"
            + " qclose_velocity_settle="
            + ("complete" if post_qclose_joint_settle_complete else "active")
            + f" qclose_velocity_settle_dwell={post_qclose_joint_settle_dwell_s:.3f}s"
            + " qclose_max_joint_speed="
            + f"{post_qclose_max_measured_joint_speed_radps:.4f}radps"
            + " geometric_preload="
            + (
                "complete"
                if post_qclose_geometric_preload_complete
                else (
                    "active"
                    if post_qclose_geometric_preload_active
                    else "pending"
                )
            )
            + " geometric_preload_terminal_error="
            + (
                f"{post_qclose_geometric_preload_terminal_error_m:.4f}m"
                if math.isfinite(post_qclose_geometric_preload_terminal_error_m)
                else "inf"
            )
            + " geometric_preload_tracking_error="
            + (
                f"{post_qclose_geometric_preload_tracking_error_m:.4f}m"
                if math.isfinite(post_qclose_geometric_preload_tracking_error_m)
                else "inf"
            )
            + " geometric_preload_inward_displacement_by_anchor="
            + ",".join(
                f"{anchor_id}:"
                f"{post_qclose_geometric_preload_achieved_inward_displacement_m_by_anchor[anchor_id]:.4f}"
                for anchor_id in selected_anchor_ids
            )
            + " geometric_preload_settle_dwell="
            + f"{post_qclose_geometric_preload_settle_dwell_s:.3f}s"
            + " contact_centering="
            + (
                "shape_frozen"
                if contact_centering_active
                else ("continuous" if contact_continuous_balance_active else "inactive")
            )
            + " post_first_arrest_transfer="
            + ("active" if post_first_arrest_centroidal_transfer_active else "inactive")
            + " contact_centering_offset="
            + f"{_norm(contact_centering_offset_world):.4f}m"
            + " contact_centering_target_delta_xyz="
            + (
                "none"
                if contact_axial_hold_base_pose is None
                or contact_centering_target_base_pose is None
                else ",".join(
                    f"{float(contact_centering_target_base_pose[index]) - float(contact_axial_hold_base_pose[index]):.5f}"
                    for index in range(3)
                )
            )
            + " contact_centering_measured_delta_xyz="
            + (
                "none"
                if contact_axial_hold_base_pose is None
                else ",".join(
                    f"{float(base_root_pose[index]) - float(contact_axial_hold_base_pose[index]):.5f}"
                    for index in range(3)
                )
            )
            + " contact_centering_tilt="
            + (
                "0.0000rad"
                if contact_axial_hold_base_pose is None
                or contact_centering_target_base_pose is None
                else f"{_norm(_rotation_error_world(contact_axial_hold_base_pose, contact_centering_target_base_pose)):.4f}rad"
            )
            + " contact_centering_measured_tilt="
            + f"{contact_centering_measured_tilt_rad:.4f}rad"
            + " contact_centering_cycles="
            + f"{contact_centering_cycle_count}"
            + " contact_centering_settle_dwell="
            + f"{contact_centering_settle_dwell_s:.3f}s"
            + " contact_pursued_anchor="
            + (
                "none"
                if contact_pursued_anchor_id is None
                else str(contact_pursued_anchor_id)
            )
            + " contact_backed_off_anchor_ids="
            + (
                "none"
                if not contact_backed_off_anchor_hold_poses_world
                else ",".join(
                    str(anchor_id)
                    for anchor_id in sorted(contact_backed_off_anchor_hold_poses_world)
                )
            )
            + " contact_stall_dwell_by_anchor="
            + ",".join(
                f"{anchor_id}:{nonprivileged_contact_stall_dwell_s_by_anchor[anchor_id]:.3f}"
                for anchor_id in selected_anchor_ids
            )
            + " contact_surface_load_arrest_candidate_by_anchor="
            + ",".join(
                f"{anchor_id}:{int(last_contact_surface_load_arrest_candidates[anchor_id])}"
                for anchor_id in selected_anchor_ids
            )
            + " contact_stall_error_by_anchor="
            + ",".join(
                f"{anchor_id}:{last_contact_stall_command_error_m_by_anchor[anchor_id]:.4f}"
                for anchor_id in selected_anchor_ids
            )
            + " contact_stall_object_relative_speed_by_anchor="
            + ",".join(
                f"{anchor_id}:{last_contact_stall_anchor_speed_mps_by_anchor[anchor_id]:.4f}"
                for anchor_id in selected_anchor_ids
            )
            + " anchor_object_relative_speed_by_anchor="
            + ",".join(
                f"{anchor_id}:{current_anchor_object_relative_speed_mps_by_anchor[anchor_id]:.4f}"
                for anchor_id in selected_anchor_ids
            )
            + " prelift_relative_motion_settled="
            + ("yes" if prelift_relative_motion_settled else "no")
            + " prelift_controller_restore_ready="
            + (
                "yes"
                if diagnostic_prelift_controller_restore_ready
                else "no"
            )
            + " prelift_relative_speed_threshold="
            + f"{prelift_relative_speed_threshold_mps:.4f}mps"
            + " anchor_object_normal_relative_speed_by_anchor="
            + ",".join(
                f"{anchor_id}:{current_anchor_object_normal_relative_speed_mps_by_anchor[anchor_id]:.4f}"
                for anchor_id in selected_anchor_ids
            )
            + " anchor_object_filtered_normal_relative_speed_by_anchor="
            + ",".join(
                f"{anchor_id}:{current_anchor_object_filtered_normal_relative_speed_mps_by_anchor[anchor_id]:.4f}"
                for anchor_id in selected_anchor_ids
            )
            + " contact_surface_filtered_clearance_rate_by_anchor="
            + ",".join(
                f"{anchor_id}:{current_filtered_gripper_surface_clearance_speed_mps_by_anchor[anchor_id]:.4f}"
                for anchor_id in selected_anchor_ids
            )
            + " contact_joint_load_by_anchor="
            + ",".join(
                f"{anchor_id}:{selected_contact_joint_load_nm_by_anchor[anchor_id]:.3f}"
                for anchor_id in selected_anchor_ids
            )
            + " contact_joint_raw_load_by_anchor="
            + ",".join(
                f"{anchor_id}:{selected_contact_raw_joint_load_nm_by_anchor[anchor_id]:.3f}"
                for anchor_id in selected_anchor_ids
            )
            + " contact_joint_damping_drive_by_anchor="
            + ",".join(
                f"{anchor_id}:{selected_contact_damping_drive_torque_nm_by_anchor[anchor_id]:.3f}"
                for anchor_id in selected_anchor_ids
            )
            + f" whole_structure_dock_drive_load={whole_structure_dock_drive_load_nm:.3f}Nm"
            + " contact_stall_gate_load_by_anchor="
            + ",".join(
                f"{anchor_id}:{contact_stall_joint_load_nm_by_anchor[anchor_id]:.3f}"
                for anchor_id in selected_anchor_ids
            )
            + " contact_anchor_target_speed_limit_by_anchor="
            + ",".join(
                f"{anchor_id}:{contact_anchor_target_speed_limit_mps_by_anchor[anchor_id]:.4f}"
                for anchor_id in selected_anchor_ids
            )
            + " contact_surface_clearance_by_anchor="
            + ",".join(
                f"{anchor_id}:{gripper_surface_clearance_m_by_anchor[anchor_id]:.6f}"
                for anchor_id in selected_anchor_ids
            )
            + " centroidal_yield="
            + (
                "requested"
                if contact_yield_requested
                else ("restoring" if contact_yield_blend > 1.0e-12 else "inactive")
            )
            + f" centroidal_yield_blend={contact_yield_blend:.3f}"
            + " centroidal_admittance="
            + ("active" if contact_admittance_requested else "inactive")
            + " joint_drive_yield="
            + (
                "requested"
                if contact_yield_joint_drive_requested
                else (
                    "restoring"
                    if contact_yield_joint_drive_blend > 1.0e-12
                    else "inactive"
                )
            )
            + f" joint_drive_yield_blend={contact_yield_joint_drive_blend:.3f}"
            + " contact_dock_drive="
            + (
                f"{contact_yield_joint_drive_last_stiffness_nm_per_rad:.1f}Nm/rad/"
                f"{contact_yield_joint_drive_last_damping_nms_per_rad:.1f}Nms/rad"
            )
            + " external_wrench="
            + (
                f"{last_external_wrench_estimate.force_norm_n:.2f}N/"
                f"{last_external_wrench_estimate.torque_norm_nm:.2f}Nm"
                if last_external_wrench_estimate.valid
                else "invalid"
            )
            + f" payload_ff={last_payload_feedforward_scale:.3f}"
            + f" payload_ff_target={last_payload_feedforward_target_scale:.3f}"
            + " payload_cmd_progress="
            + f"{last_payload_commanded_lift_progress_scale:.3f}"
            + " lift_accel_bias="
            + (
                f"{last_lift_acceleration_bias_scale:.3f}/"
                f"{last_lift_acceleration_bias_force_world_z_n:.3f}N"
            )
            + " payload_load_est="
            + f"{last_estimated_payload_lift_transfer_scale:.3f}"
            + " payload_liftoff="
            + (
                "yes"
                if payload_lift_off_confirmed_time_s is not None
                else "no"
            )
            + " loaded_state_rebase="
            + (
                "complete"
                if diagnostic_loaded_state_rebase_completed_time_s is not None
                else (
                    "settling"
                    if diagnostic_loaded_state_rebase_triggered_time_s is not None
                    else ("armed" if diagnostic_loaded_state_rebase else "disabled")
                )
            )
            + " loaded_state_settle_dwell="
            + f"{diagnostic_loaded_state_rebase_settled_dwell_s:.3f}s"
            + " admittance_twist="
            + "/".join(f"{value:.4f}" for value in last_contact_admittance_twist)
            + f" contact_force_scale={contact_force_scale:.3f}"
            + " contact_force_hold_settled="
            + (
                "yes"
                if _contact_force_hold_settled(
                    current_filtered_gripper_surface_clearance_speed_mps_by_anchor,
                    selected_anchor_ids=selected_anchor_ids,
                    speed_threshold_mps=float(
                        config.contact_stall_anchor_speed_threshold_mps
                    ),
                )
                else "no"
            )
            + f" contact_command_dwell={nonprivileged_contact_command_dwell_s:.3f}s"
            + " contact_motion_safety_authorized="
            + ("yes" if contact_required_motion_safety_authorized else "no")
            + " contact_force_scale_by_anchor="
            + ",".join(
                f"{anchor_id}:{contact_force_scale_by_anchor[anchor_id]:.3f}"
                for anchor_id in selected_anchor_ids
            )
            + " preload_load_by_anchor="
            + ",".join(
                f"{anchor_id}:{contact_position_preload_load_nm_by_anchor[anchor_id]:.3f}Nm"
                for anchor_id in selected_anchor_ids
            )
            + " preload_frozen="
            + ",".join(
                str(anchor_id)
                for anchor_id in sorted(contact_position_preload_frozen_anchor_ids)
            )
            + " anchor_pose_priority_by_anchor="
            + ",".join(
                f"{anchor_id}:{anchor_pose_priority_by_id[anchor_id]:.3f}"
                for anchor_id in selected_anchor_ids
            )
            + " dock_torque_bias_unclipped_max="
            + f"{_telemetry_max_abs(latest_dock_actuator_telemetry, 'requested_unclipped_torque_bias_nm'):.3f}Nm"
            + " dock_torque_bias_active_limit="
            + f"{active_torque_bias_limit_nm:.3f}Nm"
            + " dock_torque_bias_limited_max="
            + f"{_telemetry_max_abs(latest_dock_actuator_telemetry, 'requested_limited_torque_bias_nm'):.3f}Nm"
            + " dock_position_drive_estimate_max="
            + f"{_telemetry_max_abs(latest_dock_actuator_telemetry, 'estimated_position_drive_torque_nm'):.3f}Nm"
            + " dock_applied_torque_max="
            + f"{_telemetry_max_abs(latest_dock_actuator_telemetry, 'isaac_applied_torque_nm'):.3f}Nm"
            + f" selected_contacts={last_evidence.selected_distinct_contact_count}"
            + f" measured_contact_force={last_evidence.max_force_per_selected_contact_n:.3f}N"
            + " selected_force_by_link="
            + ",".join(
                f"{link_id}:{last_selected_normal_force_n_by_link[link_id]:.3f}N"
                for link_id in selected_link_ids
            )
            + " selected_normal_vector_by_link="
            + ",".join(
                f"{link_id}:"
                + "/".join(
                    f"{value:.3f}"
                    for value in last_selected_contact_normal_force_world_by_link[
                        link_id
                    ]
                )
                + "N"
                for link_id in selected_link_ids
            )
            + " selected_friction_vector_by_link="
            + ",".join(
                f"{link_id}:"
                + "/".join(
                    f"{value:.3f}"
                    for value in last_selected_friction_force_world_by_link[link_id]
                )
                + "N"
                for link_id in selected_link_ids
            )
            + " selected_body_velocity_by_link="
            + ",".join(
                f"{link_id}:"
                + "/".join(
                    f"{value:.4f}"
                    for value in last_selected_body_linear_velocity_world_by_link[
                        link_id
                    ]
                )
                + "mps"
                for link_id in selected_link_ids
            )
            + " selected_contact_point_velocity_by_link="
            + ",".join(
                f"{link_id}:"
                + "/".join(
                    f"{value:.4f}"
                    for value in last_selected_body_contact_velocity_world_by_link[
                        link_id
                    ]
                )
                + "mps"
                for link_id in selected_link_ids
            )
            + f" contact_slip={last_evidence.max_tangential_slip_speed_mps:.4f}mps"
            + " contact_point_slip="
            + f"{last_evidence.max_contact_point_slip_displacement_m:.4f}m"
            + f" joint_limit_violation={max_observed_joint_limit_violation_rad:.6f}rad"
        )
        if planner.phase != previous_phase:
            phase_trace.append(planner.phase.value)
            transition = planner.transitions[-1]
            planner_transitions.append(
                {
                    "from_phase": transition.from_phase.value,
                    "to_phase": transition.to_phase.value,
                    "time_s": transition.time_s,
                    "reason": transition.reason,
                }
            )
            previous_phase = planner.phase
            phase_started_s = current_time_s
            print(
                progress_message,
                file=__import__("sys").stderr,
                flush=True,
            )
        elif current_time_s - last_print_s >= 1.0:
            print(
                progress_message,
                file=__import__("sys").stderr,
                flush=True,
            )
            last_print_s = current_time_s
        if (
            diagnostic_only
            and not diagnostic_continue_after_force_ramp
            and _diagnostic_force_stop_ready(
                contact_configuration_latched=contact_configuration_latched,
                contact_force_scale=contact_force_scale,
                stop_force_scale=diagnostic_stop_force_scale,
                grasp_acquired=last_evidence.grasp_acquired,
            )
        ):
            diagnostic_stop_reached = True
            failure_reason = (
                "diagnostic_stop_force_scale_reached:" f"{contact_force_scale:.6f}"
            )
            break
        if (
            planner.phase == Order8NaturalContactPhase.COMPLETE
            and not diagnostic_disable_all_safe_hold
        ):
            break
        if planner.phase == Order8NaturalContactPhase.SAFE_HOLD:
            if diagnostic_disable_all_safe_hold:
                raise RuntimeError(
                    "Order8 diagnostic all-safe-hold suppression was bypassed"
                )
            failure_reason = planner.failure_reason or "order8_safe_hold"
            break
        tasks = _anchor_tasks_from_planner_trajectory(
            last_kinematics,
            selections,
            trajectory,
            current_anchor_poses_world=(
                measured_selected_anchor_poses_world_by_anchor
            ),
            task_application_points_world=None,
            wrench_application_points_world=None,
        )
        if contact_region_joint_closure_active:
            if not simple_closure_velocity_targets_radps:
                # Solve one fixed whole-structure direction toward the known
                # terminal anchor pose.  After this first closure step the
                # ratio is frozen; there is no receding IK or mesh tracking.
                tasks = _anchor_task_linearizations(
                    last_kinematics,
                    selections,
                    desired_anchor_poses=terminal_anchor_references,
                    wrench_targets={
                        anchor_id: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                        for anchor_id in selected_anchor_ids
                    },
                    orientation_task_weight=(
                        ORDER8_FREE_MORPH_ANCHOR_ORIENTATION_WEIGHT
                    ),
                    current_anchor_poses_world=(
                        measured_selected_anchor_poses_world_by_anchor
                    ),
                    task_application_points_world=None,
                    wrench_application_points_world=None,
                )
            else:
                tasks = []
        elif post_qclose_geometric_preload_active:
            tasks = _anchor_task_linearizations(
                last_kinematics,
                selections,
                desired_anchor_poses=dict(
                    post_qclose_geometric_preload_commanded_anchor_targets_world
                ),
                wrench_targets={
                    anchor_id: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                    for anchor_id in selected_anchor_ids
                },
                orientation_task_weight=(
                    ORDER8_FREE_MORPH_ANCHOR_ORIENTATION_WEIGHT
                ),
                current_anchor_poses_world=(
                    measured_selected_anchor_poses_world_by_anchor
                ),
                task_application_points_world=(
                    post_qclose_geometric_preload_current_surface_point_world_by_anchor
                ),
                desired_task_application_points_world={
                    anchor_id: tuple(target_pose[:3])
                    for anchor_id, target_pose in (
                        post_qclose_geometric_preload_commanded_anchor_targets_world.items()
                    )
                },
                wrench_application_points_world=(
                    post_qclose_geometric_preload_current_surface_point_world_by_anchor
                ),
            )
        elif sequential_free_shape_nudge_active:
            tasks = []
        elif sequential_latched_transfer_active:
            tasks = _sequential_latched_anchor_hold_tasks(
                tasks,
                latched_anchor_ids=set(contact_stall_latched_anchor_poses_world),
            )
        else:
            tasks = _sequential_reacquire_anchor_tasks(
                tasks,
                pursued_anchor_id=contact_pursued_anchor_id,
            )
        diagnostic_anchor_hold_joint_correction_active = bool(
            diagnostic_anchor_hold_joint_correction
            and contact_configuration_latched
            and contact_position_preload_complete
            and planner.phase
            in {
                Order8NaturalContactPhase.LIFT,
                Order8NaturalContactPhase.TRANSPORT,
                Order8NaturalContactPhase.PLACE,
            }
        )
        if diagnostic_anchor_hold_joint_correction_active:
            if grasp_hold_anchor_poses_base is None:
                raise RuntimeError(
                    "Order8 diagnostic anchor-hold correction lacks the measured "
                    "q_close anchor poses"
                )
            correction_base_target = _base_target_from_planner_trajectory(trajectory)
            correction_anchor_targets_world = {
                anchor_id: compose_pose(correction_base_target, pose_base)
                for anchor_id, pose_base in grasp_hold_anchor_poses_base.items()
            }
            tasks = _anchor_task_linearizations(
                last_kinematics,
                selections,
                desired_anchor_poses=correction_anchor_targets_world,
                wrench_targets={
                    anchor_id: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                    for anchor_id in selected_anchor_ids
                },
                orientation_task_weight=(
                    ORDER8_FREE_MORPH_ANCHOR_ORIENTATION_WEIGHT
                ),
                current_anchor_poses_world=(
                    measured_selected_anchor_poses_world_by_anchor
                ),
            )
            task_anchor_ids = {int(task.anchor_id) for task in tasks}
            if task_anchor_ids != set(selected_anchor_ids):
                raise RuntimeError(
                    "Order8 diagnostic anchor-hold correction requires one "
                    "simultaneous task for every selected anchor"
                )
            diagnostic_anchor_hold_joint_correction_max_translation_error_m = max(
                diagnostic_anchor_hold_joint_correction_max_translation_error_m,
                max(
                    (_norm(task.task_error[:3]) for task in tasks),
                    default=0.0,
                ),
            )
            diagnostic_anchor_hold_joint_correction_max_attitude_error_rad = max(
                diagnostic_anchor_hold_joint_correction_max_attitude_error_rad,
                max(
                    (_norm(task.task_error[3:]) for task in tasks),
                    default=0.0,
                ),
            )
        low_level_position_reference = joint_position_reference_by_id
        if post_qclose_geometric_preload_active:
            # Contact acquisition is a feedback-driven differential-IK
            # fallback, not a one-shot q_grasp trajectory.  Integrating from
            # the previous position command lets effort-limited drives build a
            # persistent lead before *or* after q_close when an external Dock
            # constraint deflects one joint.  Across several articulations
            # that accumulated lead becomes internal servo stress and can
            # mimic contact or move the mesh point away from the requested
            # task direction even though the instantaneous Jacobian is
            # correct.  Recede from measured q every cycle so the position
            # channel supplies only bounded velocity-like stabilization while
            # the task error remains the sole integrator.
            low_level_position_reference = {
                joint_id: float(position)
                for joint_id, position in zip(
                    joint_vector.joint_ids,
                    joint_vector.positions_rad,
                    strict=True,
                )
            }
            post_qclose_geometric_preload_measured_position_reference_step_count += 1
        active_low_level = (
            diagnostic_anchor_hold_low_level
            if diagnostic_anchor_hold_joint_correction_active
            else low_level
        )
        last_control_result = active_low_level.compute(
            joint_vector,
            tasks,
            position_reference_rad=low_level_position_reference,
        )
        if diagnostic_anchor_hold_joint_correction_active:
            if not diagnostic_anchor_hold_joint_correction_initial_targets_rad:
                diagnostic_anchor_hold_joint_correction_initial_targets_rad = dict(
                    joint_position_reference_by_id
                )
            diagnostic_anchor_hold_joint_correction_active_step_count += 1
            diagnostic_anchor_hold_joint_correction_joint_ids.update(
                last_control_result.diagnostics.task_influential_joint_ids
            )
            diagnostic_anchor_hold_joint_correction_last_reachability_status = (
                last_control_result.reachability.status
            )
            diagnostic_anchor_hold_joint_correction_max_reachability_residual = max(
                diagnostic_anchor_hold_joint_correction_max_reachability_residual,
                float(last_control_result.reachability.residual_norm),
            )
        if contact_region_joint_closure_active:
            if not simple_closure_velocity_targets_radps:
                simple_closure_velocity_targets_radps = (
                    _fixed_whole_structure_closure_velocity_targets(
                        ordered_joint_ids=(
                            last_kinematics.ordered_global_dock_joint_ids
                        ),
                        one_shot_velocity_targets_radps=(
                            last_control_result.policy_command.joint_velocity_targets
                        ),
                        maximum_speed_radps=contact_closure_joint_speed_radps,
                        fixed_joint_ids=diagnostic_pitch_hold_positions_rad,
                    )
                )
                simple_closure_open_joint_positions_rad = {
                    joint_id: float(position)
                    for joint_id, position in zip(
                        joint_vector.joint_ids,
                        joint_vector.positions_rad,
                        strict=True,
                    )
                }
                simple_closure_position_targets_rad = dict(
                    simple_closure_open_joint_positions_rad
                )
                # This is the exact achieved q_open from which the monotonic
                # closure starts.  Axial insertion can move the articulated
                # shape after the earlier approach latch, so snapshot the
                # matching measured anchor poses here as the direct-release
                # terminal instead of mixing two different configurations.
                release_terminal_anchor_targets = dict(
                    current_anchor_poses_base
                )
                simple_closure_initialized_time_s = current_time_s
            last_control_result = _apply_simple_joint_velocity_command(
                last_control_result,
                joint_vector,
                velocity_targets_radps=simple_closure_velocity_targets_radps,
                previous_position_targets_rad=(
                    simple_closure_position_targets_rad
                ),
                dt_s=sim_dt,
                zero_torque_bias=True,
            )
            simple_closure_position_targets_rad = dict(
                last_control_result.policy_command.joint_position_targets
            )
            simple_closure_active_step_count += 1
        elif contact_position_preload_active:
            contact_position_preload_velocity_targets_radps = (
                _load_limited_position_preload_velocity_targets(
                    ordered_joint_ids=(
                        last_kinematics.ordered_global_dock_joint_ids
                    ),
                    closure_velocity_targets_radps=(
                        simple_closure_velocity_targets_radps
                    ),
                    preload_joint_ids_by_anchor=(
                        contact_position_preload_joint_ids_by_anchor
                    ),
                    frozen_anchor_ids=(
                        contact_position_preload_frozen_anchor_ids
                    ),
                    maximum_speed_radps=float(
                        config.contact_position_preload_joint_speed_radps
                    ),
                    fixed_joint_ids=diagnostic_pitch_hold_positions_rad,
                )
            )
            last_control_result = _apply_simple_joint_velocity_command(
                last_control_result,
                joint_vector,
                velocity_targets_radps=(
                    contact_position_preload_velocity_targets_radps
                ),
                previous_position_targets_rad=(
                    contact_position_preload_position_targets_rad
                ),
                dt_s=sim_dt,
                zero_torque_bias=True,
            )
            contact_position_preload_position_targets_rad = dict(
                last_control_result.policy_command.joint_position_targets
            )
        elif (
            planner.phase == Order8NaturalContactPhase.RELEASE
            and simple_closure_open_joint_positions_rad
        ):
            release_velocity_targets = _joint_velocity_targets_toward_positions(
                joint_vector,
                target_positions_rad=simple_closure_open_joint_positions_rad,
                maximum_speed_radps=release_joint_speed_radps,
                dt_s=sim_dt,
            )
            for joint_id in diagnostic_pitch_hold_positions_rad:
                release_velocity_targets[joint_id] = 0.0
            if not simple_release_position_targets_rad:
                simple_release_position_targets_rad = dict(
                    last_control_result.policy_command.joint_position_targets
                )
            last_control_result = _apply_simple_joint_velocity_command(
                last_control_result,
                joint_vector,
                velocity_targets_radps=release_velocity_targets,
                previous_position_targets_rad=simple_release_position_targets_rad,
                dt_s=sim_dt,
                zero_torque_bias=True,
            )
            simple_release_position_targets_rad = dict(
                last_control_result.policy_command.joint_position_targets
            )
            simple_release_active_step_count += 1
        # The legacy mesh-tracking preload remains inactive.  The active
        # preload above advances only the fixed closure ratio and then holds
        # its load-limited absolute targets through carriage.
        contact_force_position_preload_active = bool(
            contact_position_preload_active
        )
        latched_joint_position_hold_active = bool(
            contact_configuration_latched
            and not contact_force_position_preload_active
            and not diagnostic_anchor_hold_joint_correction_active
            and planner.phase
            in {
                Order8NaturalContactPhase.CONTACT_ACQUISITION,
                Order8NaturalContactPhase.LIFT,
                Order8NaturalContactPhase.TRANSPORT,
                Order8NaturalContactPhase.PLACE,
            }
        )
        contact_force_position_preload_active_step_count += int(
            contact_force_position_preload_active
        )
        if sequential_free_shape_nudge_active:
            last_control_result = _hold_latched_joint_positions(
                last_control_result,
                joint_vector,
                position_reference_rad=joint_position_reference_by_id,
            )
            nominal_joint_position_reference_by_id = dict(
                joint_position_reference_by_id
            )
            contact_sequential_joint_position_hold_step_count += 1
        elif latched_joint_position_hold_active:
            last_control_result = _hold_latched_joint_positions(
                last_control_result,
                joint_vector,
                position_reference_rad=joint_position_reference_by_id,
            )
            latched_joint_position_hold_step_count += 1
        if (
            diagnostic_pitch_hold_positions_rad
            and not diagnostic_anchor_hold_joint_correction_active
        ):
            last_control_result = _hold_joint_subset_positions(
                last_control_result,
                joint_vector,
                position_targets_rad=diagnostic_pitch_hold_positions_rad,
            )
        if contact_configuration_latched:
            # ContactAssignment wrench targets remain semantic high-level
            # intent, but the approved Order-8 local-joint path realizes the
            # grasp only through load-limited position preload.  No
            # Jacobian-transpose/contact-wrench offset torque reaches Isaac in
            # the production path.
            last_control_result = _zero_joint_torque_bias(
                last_control_result,
                joint_vector,
            )
            if (
                diagnostic_post_grasp_joint_torque_bias_nm is not None
                and contact_position_preload_complete
                and planner.phase
                in {
                    Order8NaturalContactPhase.CONTACT_ACQUISITION,
                    Order8NaturalContactPhase.LIFT,
                    Order8NaturalContactPhase.TRANSPORT,
                    Order8NaturalContactPhase.PLACE,
                }
            ):
                selected_bias_joint_ids = {
                    joint_id
                    for joint_ids in contact_position_preload_joint_ids_by_anchor.values()
                    for joint_id in joint_ids
                }
                last_control_result = _apply_closure_direction_joint_torque_bias(
                    last_control_result,
                    joint_vector,
                    closure_velocity_targets_radps=(
                        simple_closure_velocity_targets_radps
                    ),
                    selected_joint_ids=selected_bias_joint_ids,
                    magnitude_nm=diagnostic_post_grasp_joint_torque_bias_nm,
                )
                diagnostic_post_grasp_joint_torque_bias_active_step_count += 1
                diagnostic_post_grasp_joint_torque_bias_joint_ids.update(
                    selected_bias_joint_ids
                )
                diagnostic_post_grasp_joint_torque_bias_last_map_nm = dict(
                    last_control_result.policy_command.joint_torque_bias
                )
        if diagnostic_anchor_hold_joint_correction_active:
            corrected_position_targets = dict(
                last_control_result.policy_command.joint_position_targets
            )
            corrected_velocity_targets = dict(
                last_control_result.policy_command.joint_velocity_targets
            )
            diagnostic_anchor_hold_joint_correction_max_target_offset_rad = max(
                diagnostic_anchor_hold_joint_correction_max_target_offset_rad,
                max(
                    (
                        abs(
                            float(corrected_position_targets[joint_id])
                            - float(
                                diagnostic_anchor_hold_joint_correction_initial_targets_rad[
                                    joint_id
                                ]
                            )
                        )
                        for joint_id in joint_vector.joint_ids
                    ),
                    default=0.0,
                ),
            )
            diagnostic_anchor_hold_joint_correction_max_target_step_rad = max(
                diagnostic_anchor_hold_joint_correction_max_target_step_rad,
                max(
                    (
                        abs(
                            float(corrected_position_targets[joint_id])
                            - float(joint_position_reference_by_id[joint_id])
                        )
                        for joint_id in joint_vector.joint_ids
                    ),
                    default=0.0,
                ),
            )
            diagnostic_anchor_hold_joint_correction_max_command_speed_radps = max(
                diagnostic_anchor_hold_joint_correction_max_command_speed_radps,
                max(
                    (abs(float(value)) for value in corrected_velocity_targets.values()),
                    default=0.0,
                ),
            )
            diagnostic_anchor_hold_joint_correction_last_targets_rad = (
                corrected_position_targets
            )
            diagnostic_anchor_hold_joint_correction_last_velocity_targets_radps = (
                corrected_velocity_targets
            )
        nominal_joint_position_reference_by_id = dict(
            last_control_result.policy_command.joint_position_targets
        )
        # q_close and the subsequent load-limited positional lead are absolute
        # targets.  Residual anchor-pose errors may not ratchet either hold;
        # release returns toward the measured open configuration captured at
        # closure onset.
        if not latched_joint_position_hold_active:
            joint_position_reference_by_id = nominal_joint_position_reference_by_id
        max_joint_position_command_lead_rad = max(
            max_joint_position_command_lead_rad,
            max(
                (
                    abs(
                        float(
                            last_control_result.policy_command.joint_position_targets[
                                joint_id
                            ]
                        )
                        - float(measured_position)
                    )
                    for joint_id, measured_position in zip(
                        joint_vector.joint_ids,
                        joint_vector.positions_rad,
                        strict=True,
                    )
                ),
                default=0.0,
            ),
        )
        max_joint_velocity_command_radps = max(
            max_joint_velocity_command_radps,
            max(
                (
                    abs(float(value))
                    for value in (
                        last_control_result.policy_command.joint_velocity_targets.values()
                    )
                ),
                default=0.0,
            ),
        )
        base_target = _base_target_from_planner_trajectory(trajectory)
        base_twist = _base_twist_from_planner_trajectory(trajectory)
        contact_motion_entry_speed_scale = _contact_motion_entry_speed_scale(
            planner.phase,
            phase_elapsed_s=max(0.0, current_time_s - phase_started_s),
            transition_duration_s=float(config.payload_load_transfer_s),
        )
        last_payload_commanded_lift_progress_scale = (
            contact_motion_entry_speed_scale
            if planner.phase == Order8NaturalContactPhase.LIFT
            else 0.0
        )
        payload_commanded_lift_progress_peak_scale = max(
            payload_commanded_lift_progress_peak_scale,
            last_payload_commanded_lift_progress_scale,
        )
        if contact_motion_entry_speed_scale < 1.0:
            base_target = _interpolate_pose(
                commanded_base_target,
                base_target,
                contact_motion_entry_speed_scale,
            )
            base_twist = (
                *(
                    float(value) * contact_motion_entry_speed_scale
                    for value in base_twist[:3]
                ),
                0.0,
                0.0,
                0.0,
            )
        max_base_target_step_m = max(
            max_base_target_step_m,
            _position_distance(commanded_base_target, base_target),
        )
        if planner.phase == Order8NaturalContactPhase.CONTACT_ACQUISITION:
            max_contact_base_target_step_m = max(
                max_contact_base_target_step_m,
                _position_distance(commanded_base_target, base_target),
            )
        commanded_base_target = base_target
        estimated_lift_transfer_scale = None
        measured_lift_transfer_scale = None
        lift_off_confirmed = False
        if planner.phase == Order8NaturalContactPhase.LIFT:
            if qclose_object_pose_snapshot is None:
                raise RuntimeError(
                    "Order8 LIFT requires the measured q_close object pose"
                )
            if last_external_wrench_estimate.valid:
                current_external_force_world = _vector_pose_local_to_world(
                    current_external_wrench_centroidal_model.body_pose_world,
                    last_external_wrench_estimate.wrench_body[:3],
                )
                current_external_force_world_z_n = float(
                    current_external_force_world[2]
                )
                if lift_start_external_force_world_z_n is None:
                    lift_start_external_force_world_z_n = (
                        current_external_force_world_z_n
                    )
                (
                    raw_estimated_lift_transfer_scale,
                    last_lift_external_force_world_z_n,
                    last_estimated_payload_transferred_load_n,
                ) = _payload_load_transfer_scale_from_external_wrench(
                    external_wrench_body=last_external_wrench_estimate.wrench_body,
                    body_pose_world=(
                        current_external_wrench_centroidal_model.body_pose_world
                    ),
                    lift_start_external_force_world_z_n=(
                        lift_start_external_force_world_z_n
                    ),
                    payload_mass_kg=float(config.object_mass_kg),
                    gravity_mps2=float(qpid_config.gravity_mps2),
                )
                estimated_payload_lift_transfer_peak_scale = max(
                    estimated_payload_lift_transfer_peak_scale,
                    raw_estimated_lift_transfer_scale,
                )
                payload_load_observer_valid_step_count += 1
            else:
                payload_load_observer_invalid_step_count += 1
            last_estimated_payload_lift_transfer_scale = (
                estimated_payload_lift_transfer_peak_scale
            )
            estimated_lift_transfer_scale = (
                estimated_payload_lift_transfer_peak_scale
            )
            measured_lift_transfer_scale = _measured_object_lift_transfer_scale(
                qclose_object_pose=qclose_object_pose_snapshot,
                current_object_pose=tuple(object_state["pose"]),
                transfer_distance_m=payload_load_transfer_distance_m,
            )
            measured_payload_lift_transfer_peak_scale = max(
                measured_payload_lift_transfer_peak_scale,
                measured_lift_transfer_scale,
            )
            # Once the naturally grasped object has transferred a fraction of
            # its weight, do not chatter compensation back down on millimetre
            # contact/floor bounce during the same lift.
            measured_lift_transfer_scale = measured_payload_lift_transfer_peak_scale
            lift_off_confirmed = bool(
                object_bottom_clearance >= ORDER8_OBJECT_LIFT_OFF_CLEARANCE_M
            )
            if (
                lift_off_confirmed
                and payload_lift_off_confirmed_time_s is None
            ):
                payload_lift_off_confirmed_time_s = current_time_s
                lift_acceleration_bias_lift_off_scale = float(
                    last_lift_acceleration_bias_scale
                    if diagnostic_separated_lift_transition
                    else last_payload_commanded_lift_progress_scale
                )
                if diagnostic_loaded_state_rebase:
                    # Capture the first naturally support-clear loaded state
                    # exactly once.  The object remains completely free: this
                    # writes only controller setpoints and never object pose,
                    # velocity, force, or a robot-object constraint.
                    diagnostic_loaded_state_rebase_triggered_time_s = (
                        current_time_s
                    )
                    diagnostic_loaded_state_rebase_hold_base_pose = tuple(
                        float(value) for value in base_root_pose
                    )
                    diagnostic_loaded_state_rebase_centroidal_pose = tuple(
                        float(value)
                        for value in (
                            current_external_wrench_centroidal_model.body_pose_world
                        )
                    )
                    diagnostic_loaded_state_rebase_joint_targets_rad = {
                        str(joint_id): float(position)
                        for joint_id, position in zip(
                            joint_vector.joint_ids,
                            joint_vector.positions_rad,
                            strict=True,
                        )
                    }
                    joint_position_reference_by_id = dict(
                        diagnostic_loaded_state_rebase_joint_targets_rad
                    )
                    nominal_joint_position_reference_by_id = dict(
                        diagnostic_loaded_state_rebase_joint_targets_rad
                    )
                    contact_position_preload_position_targets_rad = dict(
                        diagnostic_loaded_state_rebase_joint_targets_rad
                    )
                    for joint_id in tuple(diagnostic_pitch_hold_positions_rad):
                        diagnostic_pitch_hold_positions_rad[joint_id] = float(
                            diagnostic_loaded_state_rebase_joint_targets_rad[joint_id]
                        )
                    diagnostic_loaded_state_rebase_relative_speed_mps_at_trigger_by_anchor = dict(
                        current_anchor_object_relative_speed_mps_by_anchor
                    )
                    diagnostic_loaded_state_rebase_cumulative_slip_m_at_trigger_by_link = dict(
                        last_evidence.contact_point_slip_displacement_m_by_link
                    )
                    commanded_base_target = (
                        diagnostic_loaded_state_rebase_hold_base_pose
                    )
                    nominal_base_target = (
                        diagnostic_loaded_state_rebase_hold_base_pose
                    )
                    for qpid in (contact_centering_controller, controller):
                        qpid.reset_integrators()
            last_payload_feedforward_target_scale = max(
                float(last_payload_commanded_lift_progress_scale),
                float(estimated_lift_transfer_scale),
                float(measured_lift_transfer_scale),
                1.0 if lift_off_confirmed else 0.0,
            )
        elif planner.phase in {
            Order8NaturalContactPhase.TRANSPORT,
            Order8NaturalContactPhase.PLACE,
        }:
            last_payload_feedforward_target_scale = 1.0
        elif planner.phase == Order8NaturalContactPhase.RELEASE:
            last_payload_feedforward_target_scale = max(
                0.0,
                1.0
                - max(0.0, current_time_s - phase_started_s)
                / float(config.payload_load_transfer_s),
            )
        else:
            last_payload_feedforward_target_scale = 0.0
        payload_feedforward_scale = _payload_feedforward_scale_for_phase(
            planner.phase,
            phase_elapsed_s=max(0.0, current_time_s - phase_started_s),
            transition_duration_s=float(config.payload_load_transfer_s),
            estimated_lift_transfer_scale=estimated_lift_transfer_scale,
            measured_lift_transfer_scale=measured_lift_transfer_scale,
            previous_scale=last_payload_feedforward_scale,
            dt_s=(sim_dt if planner.phase == Order8NaturalContactPhase.LIFT else None),
            lift_off_confirmed=lift_off_confirmed,
        )
        if diagnostic_disable_payload_feedforward:
            payload_feedforward_scale = 0.0
        if planner.phase == Order8NaturalContactPhase.LIFT:
            observed_transfer_scale = max(
                estimated_payload_lift_transfer_peak_scale,
                measured_payload_lift_transfer_peak_scale,
                1.0 if lift_off_confirmed else 0.0,
            )
            max_payload_feedforward_lead_over_observed_scale = max(
                max_payload_feedforward_lead_over_observed_scale,
                payload_feedforward_scale - observed_transfer_scale,
            )
            max_payload_feedforward_lag_behind_commanded_progress_scale = max(
                max_payload_feedforward_lag_behind_commanded_progress_scale,
                last_payload_commanded_lift_progress_scale
                - payload_feedforward_scale,
            )
        lift_off_elapsed_s = (
            None
            if payload_lift_off_confirmed_time_s is None
            else max(0.0, current_time_s - payload_lift_off_confirmed_time_s)
        )
        diagnostic_loaded_state_rebase_active = bool(
            diagnostic_loaded_state_rebase
            and diagnostic_loaded_state_rebase_triggered_time_s is not None
            and diagnostic_loaded_state_rebase_completed_time_s is None
        )
        last_lift_acceleration_bias_commanded_progress_scale = (
            _diagnostic_delayed_lift_bias_progress_scale(
                planner.phase,
                enabled=diagnostic_separated_lift_transition,
                phase_elapsed_s=max(0.0, current_time_s - phase_started_s),
                bias_delay_s=diagnostic_lift_bias_delay_s,
                transition_duration_s=float(config.payload_load_transfer_s),
                normal_commanded_progress_scale=(
                    last_payload_commanded_lift_progress_scale
                ),
            )
        )
        lift_acceleration_bias_scale = _lift_acceleration_bias_scale_for_phase(
            planner.phase,
            commanded_lift_progress_scale=(
                last_lift_acceleration_bias_commanded_progress_scale
            ),
            lift_off_elapsed_s=lift_off_elapsed_s,
            lift_off_scale=lift_acceleration_bias_lift_off_scale,
            removal_duration_s=float(config.lift_acceleration_bias_removal_s),
        )
        nominal_lift_acceleration_bias_scale = lift_acceleration_bias_scale
        lift_acceleration_bias_scale = _loaded_state_rebase_acceleration_bias_scale(
            nominal_lift_acceleration_bias_scale,
            rebase_settle_active=diagnostic_loaded_state_rebase_active,
        )
        if diagnostic_loaded_state_rebase_active:
            diagnostic_loaded_state_rebase_acceleration_bias_suppressed_step_count += 1
            diagnostic_loaded_state_rebase_suppressed_acceleration_bias_peak_scale = max(
                diagnostic_loaded_state_rebase_suppressed_acceleration_bias_peak_scale,
                nominal_lift_acceleration_bias_scale,
            )
        lift_acceleration_force_bias_world = (
            _lift_acceleration_force_bias_world(
                payload_mass_kg=float(config.object_mass_kg),
                lift_payload_acceleration_mps2=(
                    float(config.lift_payload_acceleration_mps2)
                ),
                scale=lift_acceleration_bias_scale,
            )
        )
        if lift_acceleration_bias_scale > 0.0:
            lift_acceleration_bias_active_count += 1
            if planner.phase != Order8NaturalContactPhase.LIFT:
                lift_acceleration_bias_non_lift_active_count += 1
        lift_acceleration_bias_peak_scale = max(
            lift_acceleration_bias_peak_scale,
            lift_acceleration_bias_scale,
        )
        last_lift_acceleration_bias_scale = lift_acceleration_bias_scale
        last_lift_acceleration_bias_force_world_z_n = float(
            lift_acceleration_force_bias_world[2]
        )
        lift_acceleration_bias_peak_force_world_z_n = max(
            lift_acceleration_bias_peak_force_world_z_n,
            last_lift_acceleration_bias_force_world_z_n,
        )
        if (
            planner.phase == Order8NaturalContactPhase.LIFT
            and lift_off_elapsed_s is not None
            and lift_off_elapsed_s
            >= float(config.lift_acceleration_bias_removal_s)
            and lift_acceleration_bias_removal_complete_time_s is None
        ):
            lift_acceleration_bias_removal_complete_time_s = current_time_s
        if diagnostic_loaded_state_rebase_active:
            if (
                diagnostic_loaded_state_rebase_hold_base_pose is None
                or not diagnostic_loaded_state_rebase_joint_targets_rad
            ):
                raise RuntimeError(
                    "Order8 loaded-state rebase became active without a complete "
                    "captured controller state"
                )
            diagnostic_loaded_state_rebase_active_step_count += 1
            diagnostic_loaded_state_rebase_settled_dwell_s = (
                _advance_loaded_state_rebase_settle_dwell(
                    diagnostic_loaded_state_rebase_settled_dwell_s,
                    relative_speed_mps_by_anchor=(
                        current_anchor_object_relative_speed_mps_by_anchor
                    ),
                    selected_anchor_ids=selected_anchor_ids,
                    speed_threshold_mps=prelift_relative_speed_threshold_mps,
                    dt_s=sim_dt,
                )
            )
            loaded_state_elapsed_s = max(
                0.0,
                current_time_s
                - float(diagnostic_loaded_state_rebase_triggered_time_s),
            )
            # Freeze the trajectory at the one-time measured loaded pose and
            # hold the one-time measured loaded q.  The existing closure-
            # direction offset torque is retained by the position-hold helper.
            base_target = diagnostic_loaded_state_rebase_hold_base_pose
            commanded_base_target = diagnostic_loaded_state_rebase_hold_base_pose
            nominal_base_target = diagnostic_loaded_state_rebase_hold_base_pose
            base_twist = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            last_control_result = _hold_latched_joint_positions(
                last_control_result,
                joint_vector,
                position_reference_rad=(
                    diagnostic_loaded_state_rebase_joint_targets_rad
                ),
            )
            if (
                loaded_state_elapsed_s + 1.0e-12
                >= ORDER8_DIAGNOSTIC_LOADED_STATE_REBASE_MIN_HOLD_S
                and diagnostic_loaded_state_rebase_settled_dwell_s + 1.0e-12
                >= float(config.contact_stall_dwell_s)
            ):
                diagnostic_loaded_state_rebase_completed_time_s = current_time_s
                diagnostic_loaded_state_rebase_relative_speed_mps_at_completion_by_anchor = dict(
                    current_anchor_object_relative_speed_mps_by_anchor
                )
                diagnostic_loaded_state_rebase_cumulative_slip_m_at_completion_by_link = dict(
                    last_evidence.contact_point_slip_displacement_m_by_link
                )
        apply_commands(
            last_control_result,
            base_target,
            centroidal_measured_joint_positions=_global_dock_position_map(joint_vector),
            payload_feedforward_scale=payload_feedforward_scale,
            centroidal_force_bias_world=lift_acceleration_force_bias_world,
            actuator_torque_bias_limit_nm=active_torque_bias_limit_nm,
            tracking_profile=contact_yield_tracking_profile,
            admittance_active=contact_admittance_requested,
            external_wrench_estimate=last_external_wrench_estimate,
            base_twist_world=base_twist,
            zero_thrust=diagnostic_kinematic_base_isolation,
            order9_teacher_trajectory=trajectory,
        )
        current_time_s += sim_dt

    if order9_teacher_collector is not None and order9_teacher_collector.pending_command:
        # A step-budget exit has no next-loop privileged contact query.  Close
        # only the observable state transition and mark raw-contact reward
        # terms unavailable instead of reusing stale truth.
        teacher_actor_observation, teacher_reward_observation = (
            order9_teacher_observations(
                evidence=None,
                raw_contact_valid=False,
            )
        )
        order9_teacher_collector.observe_state(
            actor_observation=teacher_actor_observation,
            reward_observation=teacher_reward_observation,
        )

    if runtime_profiler is not None:
        runtime_profiler.disable()
        profile_path = Path(str(diagnostic_profile_output)).resolve()
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_profiler.dump_stats(profile_path)

    result = monitor.finalize()
    if (
        planner.phase == Order8NaturalContactPhase.COMPLETE
        and result.final_phase != Order8NaturalContactPhase.COMPLETE
    ):
        # Record the terminal phase without adding a physics step.  The monitor
        # requires the same already-acquired gates and therefore cannot turn a
        # failing episode into a pass.  The loop exits before applying another
        # physics step when the planner enters COMPLETE, so ``current_time_s``
        # still equals the last observation timestamp here.  Put the phase-only
        # record at the next control boundary to preserve the monitor's strict
        # monotonic-time contract.
        terminal_observation = replace(
            observation,
            time_s=float(observation.time_s) + sim_dt,
            phase=Order8NaturalContactPhase.COMPLETE,
        )
        last_evidence = monitor.observe(terminal_observation)
        step_evidence.append(last_evidence.to_dict())
        result = monitor.finalize()
    if failure_reason is None and not result.passed:
        failure_reason = planner.failure_reason or (
            result.failure_reasons[0]
            if result.failure_reasons
            else "order8_monitor_gate_failed"
        )

    constraint_failures = [
        failure
        for spec, _joint in constraints
        for failure in fixed_joint_identity_failures(sim.stage, spec)
    ]
    object_constraint_references = _object_constraint_references(
        sim.stage,
        object_root_path=object_path,
    )
    object_constraint_prim_paths = sorted(
        {reference[0] for reference in object_constraint_references}
    )
    collision_info = collision_approximation_evidence
    order9_teacher_episode_manifest_path: str | None = None
    if order9_teacher_collector is not None:
        from amsrr.training.order9_teacher_collection import (
            write_order9_teacher_episode,
        )

        teacher_success = bool(
            result.passed
            and not constraint_failures
            and not object_constraint_references
        )
        teacher_failure_reason = None
        if not teacher_success:
            teacher_failure_reason = failure_reason or (
                "constraint_identity_failure"
                if constraint_failures
                else "object_constraint_detected"
                if object_constraint_references
                else "order8_teacher_episode_failed"
            )
        lowered_failure = (teacher_failure_reason or "").lower()
        teacher_episode_result = order9_teacher_collector.finalize(
            success=teacher_success,
            failure_reason=teacher_failure_reason,
            release_valid=bool(
                result.release_contact_free_acquired and result.settle_acquired
            ),
            object_dropped=bool(result.object_dropped),
            hard_collision=bool(
                result.unintended_contact_count > 0
                or result.max_penetration_m
                > float(config.max_penetration_m) + 1.0e-12
            ),
            timeout=bool(
                "timeout" in lowered_failure
                or (not teacher_success and command_index >= max_steps)
            ),
            qp_infeasible_terminal=bool(
                not teacher_success
                and (
                    "qp" in lowered_failure
                    or "controller" in lowered_failure
                    or controller_failure_count > 0
                )
            ),
        )
        written_manifest = write_order9_teacher_episode(
            teacher_episode_result,
            order9_teacher_output_path,
            random_seed=int(args.order8_seed),
            robot_model_hash=physical_model.stable_hash(),
            urdf_hash=hash_file(urdf_path),
            thrust_model_hash=stable_hash(
                [rotor.to_dict() for rotor in physical_model.rotors]
            ),
            config_hash=config.stable_hash(),
            simulator_version="isaac_lab_order8_natural_contact",
            simulator_hash=stable_hash(
                {
                    "backend_config_hash": backend_config_hash,
                    "collision_approximation": collision_info,
                    "device": device,
                    "simulation_dt_s": sim_dt,
                }
            ),
            metadata={
                "teacher_task_id": order9_teacher_task_id,
                "source_graph_hash": morphology_graph.stable_hash(),
                "source_order8_result_hash": result.stable_hash(),
                "raw_contact_actor_input": False,
                "privileged_contact_role": "reward_and_safety_only",
                "full_mesh_acceptance_replaced": False,
            },
        )
        order9_teacher_episode_manifest_path = str(written_manifest)
    state_trace_payload: dict[str, object] | None = None
    if state_trace_output_path is not None:
        from amsrr.simulation.order8_state_trace import (
            build_order8_state_trace,
            write_order8_state_trace,
        )

        if (
            not state_trace_frames
            or float(state_trace_frames[-1]["simulation_time_s"])
            < current_time_s - 0.5 * sim_dt
        ):
            state_trace_frames.append(
                _capture_order8_state_trace_frame(
                    simulation_time_s=current_time_s,
                    phase=planner.phase.value,
                    robots=robots,
                    object_asset=object_asset,
                )
            )
        state_trace_payload = build_order8_state_trace(
            simulation_dt_s=sim_dt,
            frame_stride=state_trace_frame_stride,
            graph_id=morphology_graph.graph_id,
            graph_hash=morphology_graph.stable_hash(),
            config_hash=config.stable_hash(),
            source_urdf_sha256=hash_file(urdf_path),
            generated_usd_sha256=hash_file(usd_path),
            module_ids=module_ids,
            joint_names_by_module={
                module_id: tuple(robots[module_id].joint_names)
                for module_id in module_ids
            },
            source_probe_argv=sys.argv[1:],
            frames=state_trace_frames,
        )
        write_order8_state_trace(state_trace_output_path, state_trace_payload)
    report = {
        "spawn_passed": True,
        "isaac_backed": True,
        "command_applied": command_index > 0,
        "command_probe_passed": bool(
            not diagnostic_only and result.passed and not constraint_failures
        ),
        "command_returncode": (
            0
            if not diagnostic_only and result.passed and not constraint_failures
            else 1
        ),
        "order8_natural_contact_enabled": True,
        "order9_teacher_collection_enabled": order9_teacher_collector is not None,
        "order9_teacher_episode_manifest_path": (
            order9_teacher_episode_manifest_path
        ),
        "order8_natural_contact_object_support_method": (
            "free_object_on_fixed_raised_platform_without_pose_constraint_v1"
        ),
        "order8_natural_contact_object_support_path": ORDER8_OBJECT_SUPPORT_PATH,
        "order8_natural_contact_object_support_height_m": object_support_height_m,
        "order8_natural_contact_object_support_size_m": list(
            object_support_size_m
        ),
        "order8_natural_contact_object_support_pose_world": list(
            object_support_pose_world
        ),
        "order8_natural_contact_object_support_covers_planned_place_pose": True,
        "order8_state_trace_recorded": state_trace_payload is not None,
        "order8_state_trace_path": (
            None
            if state_trace_output_path is None
            else str(state_trace_output_path)
        ),
        "order8_state_trace_hash": (
            None
            if state_trace_payload is None
            else state_trace_payload["trace_payload_hash"]
        ),
        "order8_state_trace_frame_count": len(state_trace_frames),
        "order8_state_trace_acceptance_eligible": False,
        "order8_natural_contact_diagnostic_only": diagnostic_only,
        "order8_natural_contact_diagnostic_force_fixture": (diagnostic_force_fixture),
        "order8_natural_contact_diagnostic_precontact_fixture": (
            diagnostic_precontact_fixture
        ),
        "order8_natural_contact_diagnostic_near_contact_fixture": (
            diagnostic_near_contact_fixture
        ),
        "order8_natural_contact_diagnostic_qclose_fixture": (diagnostic_qclose_fixture),
        "order8_natural_contact_diagnostic_qclose_zero_velocities": (
            diagnostic_qclose_zero_velocities
        ),
        "order8_natural_contact_diagnostic_continue_after_force_ramp": (
            diagnostic_continue_after_force_ramp
        ),
        "order8_natural_contact_diagnostic_separated_lift_transition": (
            diagnostic_separated_lift_transition
        ),
        "order8_natural_contact_diagnostic_lift_bias_delay_s": (
            diagnostic_lift_bias_delay_s
        ),
        "order8_natural_contact_diagnostic_payload_feedforward_disabled": (
            diagnostic_disable_payload_feedforward
        ),
        "order8_natural_contact_diagnostic_payload_coupling_component_mode": (
            diagnostic_payload_coupling_component_mode
        ),
        "order8_natural_contact_diagnostic_payload_coupling_component_flags": (
            _diagnostic_payload_coupling_component_flags(
                diagnostic_payload_coupling_component_mode
            )
        ),
        "order8_natural_contact_diagnostic_lift_transition_stage": (
            diagnostic_lift_transition_stage
        ),
        "order8_natural_contact_diagnostic_prelift_controller_restore_ready": (
            diagnostic_prelift_controller_restore_ready
        ),
        "order8_natural_contact_diagnostic_loaded_state_rebase": (
            diagnostic_loaded_state_rebase
        ),
        "order8_natural_contact_diagnostic_loaded_state_rebase_method": (
            "geometric_liftoff_one_shot_measured_base_and_all_dock_target_"
            "rebase_then_kinematic_relative_speed_settle_v1"
            if diagnostic_loaded_state_rebase
            else "disabled"
        ),
        "order8_natural_contact_diagnostic_loaded_state_rebase_raw_contact_input": False,
        "order8_natural_contact_diagnostic_loaded_state_rebase_object_pose_write": False,
        "order8_natural_contact_diagnostic_loaded_state_rebase_direct_object_force": False,
        "order8_natural_contact_diagnostic_loaded_state_rebase_object_constraint": False,
        "order8_natural_contact_diagnostic_loaded_state_rebase_continuous_joint_correction": False,
        "order8_natural_contact_diagnostic_loaded_state_rebase_trigger_clearance_m": (
            ORDER8_OBJECT_LIFT_OFF_CLEARANCE_M
        ),
        "order8_natural_contact_diagnostic_loaded_state_rebase_min_hold_s": (
            ORDER8_DIAGNOSTIC_LOADED_STATE_REBASE_MIN_HOLD_S
        ),
        "order8_natural_contact_diagnostic_loaded_state_rebase_speed_threshold_mps": (
            prelift_relative_speed_threshold_mps
        ),
        "order8_natural_contact_diagnostic_loaded_state_rebase_required_speed_dwell_s": (
            float(config.contact_stall_dwell_s)
        ),
        "order8_natural_contact_diagnostic_loaded_state_rebase_triggered_time_s": (
            diagnostic_loaded_state_rebase_triggered_time_s
        ),
        "order8_natural_contact_diagnostic_loaded_state_rebase_completed_time_s": (
            diagnostic_loaded_state_rebase_completed_time_s
        ),
        "order8_natural_contact_diagnostic_loaded_state_rebase_active_step_count": (
            diagnostic_loaded_state_rebase_active_step_count
        ),
        "order8_natural_contact_diagnostic_loaded_state_rebase_acceleration_bias_suppressed_step_count": (
            diagnostic_loaded_state_rebase_acceleration_bias_suppressed_step_count
        ),
        "order8_natural_contact_diagnostic_loaded_state_rebase_suppressed_acceleration_bias_peak_scale": (
            diagnostic_loaded_state_rebase_suppressed_acceleration_bias_peak_scale
        ),
        "order8_natural_contact_diagnostic_loaded_state_rebase_settled_dwell_s": (
            diagnostic_loaded_state_rebase_settled_dwell_s
        ),
        "order8_natural_contact_diagnostic_loaded_state_rebase_hold_base_pose": (
            None
            if diagnostic_loaded_state_rebase_hold_base_pose is None
            else list(diagnostic_loaded_state_rebase_hold_base_pose)
        ),
        "order8_natural_contact_diagnostic_loaded_state_rebase_centroidal_pose": (
            None
            if diagnostic_loaded_state_rebase_centroidal_pose is None
            else list(diagnostic_loaded_state_rebase_centroidal_pose)
        ),
        "order8_natural_contact_diagnostic_loaded_state_rebase_joint_targets_rad": dict(
            sorted(diagnostic_loaded_state_rebase_joint_targets_rad.items())
        ),
        "order8_natural_contact_diagnostic_loaded_state_rebase_relative_speed_mps_at_trigger_by_anchor": {
            str(anchor_id): float(value)
            for anchor_id, value in sorted(
                diagnostic_loaded_state_rebase_relative_speed_mps_at_trigger_by_anchor.items()
            )
        },
        "order8_natural_contact_diagnostic_loaded_state_rebase_relative_speed_mps_at_completion_by_anchor": {
            str(anchor_id): float(value)
            for anchor_id, value in sorted(
                diagnostic_loaded_state_rebase_relative_speed_mps_at_completion_by_anchor.items()
            )
        },
        "order8_natural_contact_diagnostic_loaded_state_rebase_cumulative_slip_m_at_trigger_by_link": dict(
            sorted(
                diagnostic_loaded_state_rebase_cumulative_slip_m_at_trigger_by_link.items()
            )
        ),
        "order8_natural_contact_diagnostic_loaded_state_rebase_cumulative_slip_m_at_completion_by_link": dict(
            sorted(
                diagnostic_loaded_state_rebase_cumulative_slip_m_at_completion_by_link.items()
            )
        ),
        "order8_natural_contact_diagnostic_precontact_base_pose": (
            None
            if diagnostic_precontact_base_pose is None
            else list(diagnostic_precontact_base_pose)
        ),
        "order8_natural_contact_diagnostic_near_contact_base_pose": (
            None
            if diagnostic_near_contact_base_pose is None
            else list(diagnostic_near_contact_base_pose)
        ),
        "order8_natural_contact_diagnostic_near_contact_object_pose": (
            None
            if diagnostic_near_contact_object_pose is None
            else list(diagnostic_near_contact_object_pose)
        ),
        "order8_natural_contact_diagnostic_near_contact_joint_positions_rad": (
            dict(sorted(diagnostic_near_contact_joint_positions.items()))
        ),
        "order8_natural_contact_diagnostic_near_contact_initial_surface_clearance_m_by_anchor": {
            str(anchor_id): float(clearance)
            for anchor_id, clearance in sorted(
                diagnostic_near_contact_initial_surface_clearance_m_by_anchor.items()
            )
        },
        "order8_natural_contact_diagnostic_near_contact_warmup_duration_s": (
            ORDER8_NEAR_CONTACT_DIAGNOSTIC_WARMUP_S
            if diagnostic_near_contact_fixture
            else None
        ),
        "order8_natural_contact_diagnostic_near_contact_warmup_complete": (
            diagnostic_near_contact_warmup_complete
        ),
        "order8_natural_contact_diagnostic_near_contact_warmup_completed_time_s": (
            diagnostic_near_contact_warmup_completed_time_s
        ),
        "order8_natural_contact_diagnostic_near_contact_estimator_reset_count": (
            diagnostic_near_contact_estimator_reset_count
        ),
        "order8_natural_contact_diagnostic_world_fixed_base": bool(
            diagnostic_world_fixed_joint is not None
        ),
        "order8_natural_contact_diagnostic_kinematic_base_isolation": (
            diagnostic_kinematic_base_isolation
        ),
        "order8_natural_contact_diagnostic_world_fixed_body_path": (
            diagnostic_world_fixed_body_path
        ),
        "order8_natural_contact_diagnostic_world_fixed_pose": (
            None
            if diagnostic_world_fixed_pose is None
            else list(diagnostic_world_fixed_pose)
        ),
        "order8_natural_contact_diagnostic_world_fixed_object": bool(
            diagnostic_world_fixed_object_joint is not None
        ),
        "order8_natural_contact_diagnostic_world_fixed_object_pose": (
            None
            if diagnostic_world_fixed_object_pose is None
            else list(diagnostic_world_fixed_object_pose)
        ),
        "order8_natural_contact_diagnostic_object_width_padding_m": (
            diagnostic_object_width_padding_m
        ),
        "order8_natural_contact_diagnostic_profile_output": (
            None
            if diagnostic_profile_output is None
            else str(diagnostic_profile_output)
        ),
        "order8_natural_contact_diagnostic_dock_velocity_limit_rad_s": (
            diagnostic_dock_velocity_limit
        ),
        "order8_natural_contact_dock_velocity_limit_sim_rad_s": (dock_velocity_limit),
        "order8_natural_contact_contact_joint_velocity_limit_command_rad_s": (
            contact_joint_velocity_limit
        ),
        "order8_natural_contact_contact_joint_velocity_limit_basis": (
            "fixed_whole_structure_previous_target_integrated_velocity_and_"
            "simulator_"
            "consistent_below_ak40_10_"
            "configured_speed_limit_v2"
        ),
        "order8_natural_contact_diagnostic_dock_armature_kg_m2": (
            diagnostic_dock_armature_kg_m2
        ),
        "order8_natural_contact_diagnostic_peak_torque_window_s": (
            diagnostic_peak_torque_window_s
        ),
        "order8_natural_contact_diagnostic_peak_torque_active_step_count": (
            diagnostic_peak_torque_active_step_count
        ),
        "order8_natural_contact_diagnostic_peak_torque_max_limit_nm": (
            diagnostic_peak_torque_max_limit_nm
        ),
        "order8_natural_contact_diagnostic_peak_torque_limit_schedule": (
            "peak_hold_then_final_0p25s_linear_return_to_continuous_v1"
            if diagnostic_peak_torque_window_s is not None
            else "disabled_continuous_rating_v1"
        ),
        "order8_natural_contact_diagnostic_post_grasp_joint_torque_bias_nm": (
            diagnostic_post_grasp_joint_torque_bias_nm
        ),
        "order8_natural_contact_diagnostic_post_grasp_joint_torque_bias_method": (
            "fixed_closure_direction_equal_magnitude_after_load_preload_v1"
            if diagnostic_post_grasp_joint_torque_bias_nm is not None
            else "disabled"
        ),
        "order8_natural_contact_diagnostic_post_grasp_joint_torque_bias_active_step_count": (
            diagnostic_post_grasp_joint_torque_bias_active_step_count
        ),
        "order8_natural_contact_diagnostic_post_grasp_joint_torque_bias_joint_ids": (
            sorted(diagnostic_post_grasp_joint_torque_bias_joint_ids)
        ),
        "order8_natural_contact_diagnostic_post_grasp_joint_torque_bias_last_map_nm": dict(
            sorted(diagnostic_post_grasp_joint_torque_bias_last_map_nm.items())
        ),
        "order8_natural_contact_diagnostic_disable_slip_speed_safe_hold": (
            diagnostic_disable_slip_speed_safe_hold
        ),
        "order8_natural_contact_contact_slip_speed_safe_hold_enabled": (
            False
        ),
        "order8_natural_contact_diagnostic_disable_all_safe_hold": (
            diagnostic_disable_all_safe_hold
        ),
        "order8_natural_contact_diagnostic_all_safe_hold_suppression_method": (
            "record_all_evidence_suppress_supervisor_transitions_extend_planner_"
            "timeouts_and_run_complete_step_budget_v1"
            if diagnostic_disable_all_safe_hold
            else "disabled"
        ),
        "order8_natural_contact_diagnostic_requested_simulation_duration_s": (
            requested_simulation_duration_s
        ),
        "order8_natural_contact_diagnostic_full_step_budget_reached": bool(
            current_time_s >= requested_simulation_duration_s - 0.5 * sim_dt
        ),
        "order8_natural_contact_diagnostic_suppressed_safe_hold_request_count": (
            sum(suppressed_safe_hold_reason_counts.values())
        ),
        "order8_natural_contact_diagnostic_suppressed_safe_hold_reason_counts": dict(
            sorted(suppressed_safe_hold_reason_counts.items())
        ),
        "order8_natural_contact_diagnostic_suppressed_safe_hold_first_time_s_by_reason": dict(
            sorted(suppressed_safe_hold_first_time_s_by_reason.items())
        ),
        "order8_natural_contact_diagnostic_lock_object_rotation": (
            diagnostic_lock_object_rotation
        ),
        "order8_natural_contact_diagnostic_object_rotation_lock_method": (
            "lift_entry_orientation_per_step_state_projection_translation_free_v1"
            if diagnostic_lock_object_rotation
            else "disabled"
        ),
        "order8_natural_contact_diagnostic_object_rotation_lock_orientation_xyzw": (
            None
            if diagnostic_object_rotation_lock_orientation_xyzw is None
            else list(diagnostic_object_rotation_lock_orientation_xyzw)
        ),
        "order8_natural_contact_diagnostic_object_rotation_projection_step_count": (
            diagnostic_object_rotation_projection_step_count
        ),
        "order8_natural_contact_diagnostic_object_rotation_projection_max_deviation_rad": (
            diagnostic_object_rotation_projection_max_deviation_rad
        ),
        "order8_natural_contact_diagnostic_object_rotation_projection_max_angular_speed_rad_s": (
            diagnostic_object_rotation_projection_max_angular_speed_rad_s
        ),
        "order8_natural_contact_diagnostic_anchor_hold_joint_correction": (
            diagnostic_anchor_hold_joint_correction
        ),
        "order8_natural_contact_diagnostic_anchor_hold_joint_correction_method": (
            "qclose_rigid_grasp_following_commanded_centroidal_path_two_anchor_"
            "full_dock_dls_absolute_position_outer_loop_v2"
            if diagnostic_anchor_hold_joint_correction
            else "disabled"
        ),
        "order8_natural_contact_diagnostic_anchor_hold_joint_correction_raw_contact_input": False,
        "order8_natural_contact_diagnostic_anchor_hold_joint_correction_qpid_input": False,
        "order8_natural_contact_diagnostic_anchor_hold_joint_correction_task_gain_per_s": (
            ORDER8_DIAGNOSTIC_ANCHOR_HOLD_TASK_GAIN_PER_S
        ),
        "order8_natural_contact_diagnostic_anchor_hold_joint_correction_dls_damping": (
            diagnostic_anchor_hold_low_level.config.dls_damping
        ),
        "order8_natural_contact_diagnostic_anchor_hold_joint_correction_max_position_lead_rad": (
            diagnostic_anchor_hold_low_level.config.max_position_command_lead_rad
        ),
        "order8_natural_contact_diagnostic_anchor_hold_joint_correction_active_step_count": (
            diagnostic_anchor_hold_joint_correction_active_step_count
        ),
        "order8_natural_contact_diagnostic_anchor_hold_joint_correction_joint_ids": (
            sorted(diagnostic_anchor_hold_joint_correction_joint_ids)
        ),
        "order8_natural_contact_diagnostic_anchor_hold_joint_correction_initial_targets_rad": dict(
            sorted(
                diagnostic_anchor_hold_joint_correction_initial_targets_rad.items()
            )
        ),
        "order8_natural_contact_diagnostic_anchor_hold_joint_correction_last_targets_rad": dict(
            sorted(diagnostic_anchor_hold_joint_correction_last_targets_rad.items())
        ),
        "order8_natural_contact_diagnostic_anchor_hold_joint_correction_last_velocity_targets_radps": dict(
            sorted(
                diagnostic_anchor_hold_joint_correction_last_velocity_targets_radps.items()
            )
        ),
        "order8_natural_contact_diagnostic_anchor_hold_joint_correction_max_translation_error_m": (
            diagnostic_anchor_hold_joint_correction_max_translation_error_m
        ),
        "order8_natural_contact_diagnostic_anchor_hold_joint_correction_max_attitude_error_rad": (
            diagnostic_anchor_hold_joint_correction_max_attitude_error_rad
        ),
        "order8_natural_contact_diagnostic_anchor_hold_joint_correction_max_target_offset_rad": (
            diagnostic_anchor_hold_joint_correction_max_target_offset_rad
        ),
        "order8_natural_contact_diagnostic_anchor_hold_joint_correction_max_target_step_rad": (
            diagnostic_anchor_hold_joint_correction_max_target_step_rad
        ),
        "order8_natural_contact_diagnostic_anchor_hold_joint_correction_max_command_speed_radps": (
            diagnostic_anchor_hold_joint_correction_max_command_speed_radps
        ),
        "order8_natural_contact_diagnostic_anchor_hold_joint_correction_last_reachability_status": (
            diagnostic_anchor_hold_joint_correction_last_reachability_status
        ),
        "order8_natural_contact_diagnostic_anchor_hold_joint_correction_max_reachability_residual": (
            diagnostic_anchor_hold_joint_correction_max_reachability_residual
        ),
        "order8_natural_contact_dock_armature_kg_m2": dock_armature_kg_m2,
        "order8_natural_contact_dock_armature_source": dock_armature_source,
        "order8_natural_contact_applied_dock_armature_kg_m2_by_joint": dict(
            sorted(applied_dock_armature_kg_m2_by_joint.items())
        ),
        "order8_natural_contact_runtime_object_size_m": list(runtime_object_size_m),
        "order8_natural_contact_qclose_checkpoint_base_pose": (
            None
            if qclose_base_pose_snapshot is None
            else list(qclose_base_pose_snapshot)
        ),
        "order8_natural_contact_qclose_checkpoint_joint_positions_rad": dict(
            sorted(qclose_joint_positions_snapshot.items())
        ),
        "order8_natural_contact_qclose_checkpoint_object_pose": (
            None
            if qclose_object_pose_snapshot is None
            else list(qclose_object_pose_snapshot)
        ),
        "order8_natural_contact_qclose_checkpoint_state": (
            qclose_checkpoint_state_snapshot
        ),
        "order8_natural_contact_acceptance_eligible": not diagnostic_only,
        "order8_natural_contact_diagnostic_mode": (
            "preloaded_neutral_grasp_force_fixture_v1"
            if diagnostic_force_fixture
            else (
                (
                    (
                        "static_measured_qclose_checkpoint_force_fixture_v1"
                        if diagnostic_qclose_zero_velocities
                        else "exact_measured_qclose_checkpoint_force_fixture_v2"
                    )
                    if diagnostic_qclose_checkpoint_state is not None
                    else "approximate_measured_qclose_checkpoint_force_fixture_v1"
                )
                if diagnostic_qclose_fixture
                else (
                    (
                        (
                            "measured_collision_free_near_contact_continuation_v1"
                            if diagnostic_continue_after_force_ramp
                            else "measured_collision_free_near_contact_fixture_v1"
                        )
                        if diagnostic_near_contact_fixture
                        else (
                            "measured_post_axial_precontact_continuation_v1"
                            if diagnostic_continue_after_force_ramp
                            else "measured_post_axial_precontact_fixture_v1"
                        )
                    )
                    if (
                        diagnostic_near_contact_fixture
                        or diagnostic_precontact_fixture
                    )
                    else (
                        "accelerated_contact_force_isolation_v1"
                        if diagnostic_only
                        else "disabled"
                    )
                )
            )
        ),
        "order8_natural_contact_diagnostic_stop_force_scale": (
            diagnostic_stop_force_scale
            if diagnostic_only and not diagnostic_continue_after_force_ramp
            else None
        ),
        "order8_natural_contact_diagnostic_force_anchor_ids": (
            list(force_ramp_anchor_ids)
            if diagnostic_force_anchor_ids is not None
            else None
        ),
        "order8_natural_contact_diagnostic_stop_reached": (diagnostic_stop_reached),
        "order8_natural_contact_diagnostic_observed_monitor_passed": bool(
            result.passed
        ),
        "order8_natural_contact_solver_position_iteration_count": (
            solver_position_iteration_count
        ),
        "order8_natural_contact_solver_velocity_iteration_count": (
            solver_velocity_iteration_count
        ),
        "order8_natural_contact_report_version": ORDER8_ISAAC_REPORT_VERSION,
        "order8_natural_contact_passed": bool(
            not diagnostic_only and result.passed and not constraint_failures
        ),
        "order8_natural_contact_contact_model": ORDER8_NATURAL_CONTACT_MODEL,
        "order8_natural_contact_config": config.to_dict(),
        "order8_natural_contact_config_hash": config.stable_hash(),
        "order8_natural_contact_graph_id": morphology_graph.graph_id,
        "order8_natural_contact_graph_hash": morphology_graph.stable_hash(),
        "order8_natural_contact_module_count": len(module_ids),
        "order8_natural_contact_robot_anchor_count": len(grasp_anchors),
        "order8_natural_contact_seed": int(args.order8_seed),
        "order8_natural_contact_seed_applied": seed_application,
        "order8_natural_contact_backend_config_hash": backend_config_hash,
        "order8_natural_contact_physical_model_hash": physical_model.stable_hash(),
        "order8_natural_contact_collision_geometry_content_hash": collision_geometry_content_hash(
            physical_model,
            mesh_search_dirs=("module_urdf", "module_urdf/mesh"),
        ),
        "order8_natural_contact_force_usd_conversion": bool(args.force_convert),
        "order8_natural_contact_dock_collision_type": "Convex Decomposition",
        "order8_natural_contact_dock_collision_approximation_token": collision_info.get(
            "requested_approximation_token"
        ),
        "order8_natural_contact_dock_collision_approximation_verified": collision_info.get(
            "verified"
        )
        is True,
        "order8_natural_contact_dock_collision_composed_prim_count": int(
            collision_info.get("composed_prim_count", 0)
        ),
        "order8_natural_contact_source_urdf_sha256": hash_file(
            physical_model.urdf_path
        ),
        "order8_natural_contact_generated_usd_sha256": hash_file(usd_path),
        "order8_natural_contact_generated_usd_bundle_hash": hash_directory_manifest(
            usd_path.parent
        ),
        "order8_natural_contact_requested_steps": int(args.steps),
        "order8_natural_contact_simulation_dt_s": sim_dt,
        "order8_natural_contact_simulation_time_s": current_time_s,
        "order8_natural_contact_executed_steps": command_index,
        "order8_natural_contact_qpid_config": asdict(qpid_config),
        "order8_natural_contact_qpid_config_hash": stable_hash(qpid_config),
        "order8_natural_contact_contact_centering_qpid_config": asdict(
            contact_centering_qpid_config
        ),
        "order8_natural_contact_contact_centering_qpid_config_hash": (
            stable_hash(contact_centering_qpid_config)
        ),
        "order8_natural_contact_external_wrench_estimator_config": asdict(
            external_wrench_estimator_config
        ),
        "order8_natural_contact_external_wrench_estimator_config_hash": stable_hash(
            external_wrench_estimator_config
        ),
        "order8_natural_contact_contact_admittance_config": asdict(
            contact_admittance_config
        ),
        "order8_natural_contact_contact_admittance_config_hash": stable_hash(
            contact_admittance_config
        ),
        "order8_natural_contact_contact_yield_method": (
            "first_damping_compensated_terminal_joint_surface_load_enables_"
            "contact_axis_centroidal_admittance_with_full_height_attitude_"
            "pose_tracking_v9"
        ),
        "order8_natural_contact_contact_yield_trigger_method": (
            "any_selected_terminal_joint_damping_compensated_load_plus_mesh_"
            "proximity_after_closure_armed_latched_until_verified_grasp_v7"
        ),
        "order8_natural_contact_contact_yield_raw_contact_input": False,
        "order8_natural_contact_contact_yield_per_contact_wrench_input": False,
        "order8_natural_contact_contact_yield_external_wrench_scope": (
            "aggregate_centroidal_only_v1"
        ),
        "order8_natural_contact_contact_yield_triggered_time_s": (
            contact_yield_triggered_time_s
        ),
        "order8_natural_contact_contact_yield_load_dwell_s_by_anchor": {
            str(anchor_id): float(dwell_s)
            for anchor_id, dwell_s in sorted(
                contact_yield_load_dwell_s_by_anchor.items()
            )
        },
        "order8_natural_contact_contact_yield_trigger_anchor_ids": sorted(
            contact_yield_trigger_anchor_ids
        ),
        "order8_natural_contact_contact_load_detection_arm_method": (
            "fixed_whole_structure_closure_with_complete_previous_step_"
            "actuator_telemetry_and_nominal_dock_impedance_v3"
        ),
        "order8_natural_contact_contact_load_detection_armed_step_count": (
            contact_load_detection_armed_step_count
        ),
        "order8_natural_contact_contact_load_detection_armed_time_s": (
            contact_load_detection_armed_time_s
        ),
        "order8_natural_contact_contact_admittance_trigger_method": (
            "any_selected_terminal_joint_damping_compensated_load_plus_mesh_"
            "proximity_v4"
        ),
        "order8_natural_contact_contact_admittance_triggered_time_s": (
            contact_admittance_triggered_time_s
        ),
        "order8_natural_contact_contact_admittance_trigger_anchor_ids": sorted(
            contact_admittance_trigger_anchor_ids
        ),
        "order8_natural_contact_contact_admittance_final_active": (
            contact_admittance_requested
        ),
        "order8_natural_contact_contact_yield_active_step_count": (
            contact_yield_active_step_count
        ),
        "order8_natural_contact_contact_yield_full_step_count": (
            contact_yield_full_step_count
        ),
        "order8_natural_contact_contact_yield_restore_step_count": (
            contact_yield_restore_step_count
        ),
        "order8_natural_contact_contact_yield_final_blend": contact_yield_blend,
        "order8_natural_contact_contact_yield_minimum_pi_scale": (
            contact_yield_minimum_pi_scale
        ),
        "order8_natural_contact_contact_yield_estimator_valid_step_count": (
            contact_yield_estimator_valid_step_count
        ),
        "order8_natural_contact_contact_yield_estimator_invalid_step_count": (
            contact_yield_estimator_invalid_step_count
        ),
        "order8_natural_contact_contact_yield_maximum_external_force_n": (
            contact_yield_maximum_external_force_n
        ),
        "order8_natural_contact_contact_yield_maximum_external_torque_nm": (
            contact_yield_maximum_external_torque_nm
        ),
        "order8_natural_contact_contact_yield_maximum_translation_offset_m": (
            contact_yield_maximum_translation_offset_m
        ),
        "order8_natural_contact_contact_yield_last_admittance_twist": list(
            last_contact_admittance_twist
        ),
        "order8_natural_contact_contact_yield_last_translation_offset_world": list(
            last_contact_admittance_translation_offset_world
        ),
        "order8_natural_contact_contact_yield_grasp_pose_rebased": (
            contact_yield_grasp_pose_rebased
        ),
        "order8_natural_contact_contact_yield_grasp_pose_rebase_time_s": (
            contact_yield_grasp_pose_rebase_time_s
        ),
        "order8_natural_contact_contact_yield_grasp_centroidal_pose": (
            None
            if contact_yield_grasp_pose is None
            else list(contact_yield_grasp_pose)
        ),
        "order8_natural_contact_contact_yield_grasp_rebase_method": (
            "nonprivileged_load_limited_position_preload_measured_full_6d_"
            "centroidal_with_zero_offset_torque_then_pi_restore_v6"
        ),
        "order8_natural_contact_contact_yield_joint_drive_method": (
            "disabled_nominal_dock_implicit_impedance_preserved_v7"
        ),
        "order8_natural_contact_contact_yield_joint_drive_trigger_method": (
            "disabled_v7"
        ),
        "order8_natural_contact_contact_yield_joint_drive_raw_contact_input": False,
        "order8_natural_contact_contact_yield_joint_drive_scope": (
            "none_v2"
        ),
        "order8_natural_contact_contact_yield_joint_drive_triggered_time_s": (
            contact_yield_joint_drive_triggered_time_s
        ),
        "order8_natural_contact_contact_yield_joint_drive_final_blend": (
            contact_yield_joint_drive_blend
        ),
        "order8_natural_contact_contact_yield_joint_drive_nominal_stiffness_nm_per_rad": (
            float(dock_stiffness)
        ),
        "order8_natural_contact_contact_yield_joint_drive_nominal_damping_nms_per_rad": (
            float(dock_damping)
        ),
        "order8_natural_contact_contact_yield_joint_drive_stiffness_scale": (
            float(config.contact_yield_joint_drive_stiffness_scale)
        ),
        "order8_natural_contact_contact_yield_joint_drive_target_damping_nms_per_rad": (
            float(config.contact_yield_joint_drive_damping_nms_per_rad)
        ),
        "order8_natural_contact_contact_yield_joint_drive_active_step_count": (
            contact_yield_joint_drive_active_step_count
        ),
        "order8_natural_contact_contact_yield_joint_drive_write_count": (
            contact_yield_joint_drive_write_count
        ),
        "order8_natural_contact_contact_yield_joint_drive_restore_write_count": (
            contact_yield_joint_drive_restore_write_count
        ),
        "order8_natural_contact_contact_yield_joint_drive_minimum_stiffness_nm_per_rad": (
            contact_yield_joint_drive_minimum_stiffness_nm_per_rad
        ),
        "order8_natural_contact_contact_yield_joint_drive_maximum_damping_nms_per_rad": (
            contact_yield_joint_drive_maximum_damping_nms_per_rad
        ),
        "order8_natural_contact_contact_yield_joint_drive_final_stiffness_nm_per_rad": (
            contact_yield_joint_drive_last_stiffness_nm_per_rad
        ),
        "order8_natural_contact_contact_yield_joint_drive_final_damping_nms_per_rad": (
            contact_yield_joint_drive_last_damping_nms_per_rad
        ),
        "order8_natural_contact_contact_yield_joint_drive_stiffness_targets_nm_per_rad": (
            dict(sorted(contact_yield_joint_drive_stiffness_targets.items()))
        ),
        "order8_natural_contact_contact_yield_joint_drive_damping_targets_nms_per_rad": (
            dict(sorted(contact_yield_joint_drive_damping_targets.items()))
        ),
        "order8_natural_contact_contact_axial_qpid_gain_schedule": (
            "mesh_open_axial_insert_uses_centering_horizontal_gain_bank_v1"
        ),
        "order8_natural_contact_contact_axial_gain_scheduled_step_count": (
            contact_axial_gain_scheduled_step_count
        ),
        "order8_natural_contact_joint_controller_config": asdict(
            joint_controller_config
        ),
        "order8_natural_contact_joint_controller_config_hash": stable_hash(
            joint_controller_config
        ),
        "order8_natural_contact_joint_position_reference_mode": (
            "one_shot_whole_structure_ik_direction_previous_target_integrated_"
            "fixed_velocity_ratio_with_diagnostic_absolute_pitch_hold_until_"
            "load_qclose_then_slow_load_limited_previous_target_preload_and_"
            "measured_qopen_direct_return_v12"
        ),
        "order8_natural_contact_contact_closure_driver": (
            "known_pose_one_shot_whole_structure_ik_direction_then_fixed_"
            "yaw_velocity_ratio_with_pitch_hold_load_arrest_v5"
        ),
        "order8_natural_contact_contact_closure_general_grasp_planner": False,
        "order8_natural_contact_contact_closure_velocity_targets_radps": dict(
            sorted(simple_closure_velocity_targets_radps.items())
        ),
        "order8_natural_contact_contact_closure_open_joint_positions_rad": dict(
            sorted(simple_closure_open_joint_positions_rad.items())
        ),
        "order8_natural_contact_contact_closure_last_position_targets_rad": dict(
            sorted(simple_closure_position_targets_rad.items())
        ),
        "order8_natural_contact_diagnostic_pitch_hold_joint_positions_rad": dict(
            sorted(diagnostic_pitch_hold_positions_rad.items())
        ),
        "order8_natural_contact_diagnostic_pitch_hold_max_error_rad": (
            max_diagnostic_pitch_hold_error_rad
        ),
        "order8_natural_contact_diagnostic_pitch_hold_method": (
            "absolute_initial_position_zero_velocity_zero_torque_bias_v1"
            if diagnostic_pitch_hold_positions_rad
            else "disabled_non_diagnostic_v1"
        ),
        "order8_natural_contact_contact_closure_initialized_time_s": (
            simple_closure_initialized_time_s
        ),
        "order8_natural_contact_contact_closure_active_step_count": (
            simple_closure_active_step_count
        ),
        "order8_natural_contact_simple_release_active_step_count": (
            simple_release_active_step_count
        ),
        "order8_natural_contact_contact_closure_joint_speed_radps": min(
            contact_closure_joint_speed_radps,
            contact_joint_velocity_limit,
        ),
        "order8_natural_contact_release_joint_speed_radps": (
            release_joint_speed_radps
        ),
        "order8_natural_contact_diagnostic_contact_closure_joint_speed_override_radps": (
            None
            if diagnostic_contact_closure_joint_speed_raw is None
            else contact_closure_joint_speed_radps
        ),
        "order8_natural_contact_max_joint_position_command_lead_rad": (
            max_joint_position_command_lead_rad
        ),
        "order8_natural_contact_max_joint_velocity_command_radps": (
            max_joint_velocity_command_radps
        ),
        "order8_natural_contact_planner_config": asdict(planner_config),
        "order8_natural_contact_planner_config_hash": stable_hash(planner_config),
        "order8_natural_contact_base_target_speed_limit_mps": float(
            config.base_translation_speed_limit_mps
        ),
        "order8_natural_contact_contact_base_target_speed_limit_mps": float(
            config.contact_base_translation_speed_limit_mps
        ),
        "order8_natural_contact_contact_base_target_speed_limit_phases": [
            Order8NaturalContactPhase.CONTACT_ACQUISITION.value,
            Order8NaturalContactPhase.LIFT.value,
            Order8NaturalContactPhase.TRANSPORT.value,
            Order8NaturalContactPhase.PLACE.value,
        ],
        "order8_natural_contact_contact_axial_min_mesh_overlap_m": float(
            config.contact_axial_min_mesh_overlap_m
        ),
        "order8_natural_contact_contact_axial_overlap_method": (
            "selected_urdf_mesh_world_aabb_approach_axis_projection_v1"
        ),
        "order8_natural_contact_contact_axial_overlap_at_latch_m": (
            contact_axial_overlap_at_latch_m
        ),
        "order8_natural_contact_grasp_base_pose_method": (
            "normal_aligned_floor_clear_tangential_contact_region_v1"
        ),
        "order8_natural_contact_floor_base_pose": list(floor_base_pose),
        "order8_natural_contact_unconstrained_grasp_base_pose": list(
            grasp_base_plan.unconstrained_base_pose_world
        ),
        "order8_natural_contact_grasp_base_pose": list(grasp_base_plan.base_pose_world),
        "order8_natural_contact_grasp_base_vertical_correction_m": float(
            grasp_base_plan.vertical_correction_m
        ),
        "order8_natural_contact_grasp_additional_floor_clearance_m": (
            ORDER8_GRASP_ADDITIONAL_FLOOR_CLEARANCE_M
        ),
        "order8_natural_contact_grasp_base_normal_correction_m_by_anchor": {
            str(anchor_id): float(value)
            for anchor_id, value in sorted(
                grasp_base_plan.normal_correction_m_by_anchor.items()
            )
        },
        "order8_natural_contact_grasp_base_tangential_correction_m_by_anchor": {
            str(anchor_id): [float(value) for value in values]
            for anchor_id, values in sorted(
                grasp_base_plan.tangential_correction_m_by_anchor.items()
            )
        },
        "order8_natural_contact_contact_axial_hold_method": (
            "measured_free_object_relative_floor_clear_contact_region_base_pose_"
            "with_rate_limited_retarget_v4"
        ),
        "order8_natural_contact_contact_axial_hold_base_pose": (
            None
            if contact_axial_hold_base_pose is None
            else list(contact_axial_hold_base_pose)
        ),
        "order8_natural_contact_contact_axial_settle_dwell_s": (
            contact_axial_settle_dwell_s
        ),
        "order8_natural_contact_contact_axial_settle_position_tolerance_m": (
            contact_axial_settle_position_tolerance_m
        ),
        "order8_natural_contact_contact_axial_settle_base_speed_tolerance_mps": (
            contact_axial_settle_base_speed_tolerance_mps
        ),
        "order8_natural_contact_contact_side_closure_enabled": (
            contact_side_closure_enabled
        ),
        "order8_natural_contact_contact_anchor_target_speed_limit_mps": (
            float(config.anchor_translation_speed_limit_mps)
        ),
        "order8_natural_contact_contact_near_anchor_target_speed_limit_mps": (
            0.2 * float(config.anchor_translation_speed_limit_mps)
        ),
        "order8_natural_contact_contact_near_anchor_slowdown_error_m": (
            float(config.contact_near_surface_slowdown_m)
        ),
        "order8_natural_contact_contact_surface_anchor_target_speed_limit_mps": (
            float(config.contact_surface_creep_speed_limit_mps)
        ),
        "order8_natural_contact_contact_surface_anchor_speed_boundary_m": (
            float(config.contact_surface_arm_clearance_m)
        ),
        "order8_natural_contact_contact_anchor_target_speed_schedule": (
            "nonprivileged_three_tier_precenter_then_symmetric_creep_close_"
            "with_opposing_clearance_synchronization_v8"
        ),
        "order8_natural_contact_contact_clearance_sync_method": (
            "closer_surface_linear_slowdown_farther_surface_full_tier_speed_v1"
        ),
        "order8_natural_contact_contact_clearance_sync_deadband_m": float(
            config.contact_clearance_sync_deadband_m
        ),
        "order8_natural_contact_contact_clearance_sync_full_slowdown_m": float(
            config.contact_clearance_sync_full_slowdown_m
        ),
        "order8_natural_contact_contact_clearance_sync_minimum_speed_scale": float(
            config.contact_clearance_sync_minimum_speed_scale
        ),
        "order8_natural_contact_contact_clearance_sync_active_step_count": (
            contact_clearance_sync_active_step_count
        ),
        "order8_natural_contact_post_first_arrest_creep_method": (
            "unlatched_anchor_only_bounded_creep_acceleration_v1"
        ),
        "order8_natural_contact_post_first_arrest_creep_multiplier": (
            ORDER8_POST_FIRST_ARREST_CREEP_MULTIPLIER
        ),
        "order8_natural_contact_post_first_arrest_creep_active_step_count": (
            post_first_arrest_creep_active_step_count
        ),
        "order8_natural_contact_post_first_arrest_centroidal_transfer_method": (
            "unlatched_inward_axis_bounded_centroidal_translation_with_world_fixed_latched_anchor_v1"
        ),
        "order8_natural_contact_post_first_arrest_centroidal_transfer_active_step_count": (
            post_first_arrest_centroidal_transfer_active_step_count
        ),
        "order8_natural_contact_post_first_arrest_centroidal_transfer_speed_limit_mps": float(
            min(
                float(config.contact_base_translation_speed_limit_mps),
                float(config.contact_surface_creep_speed_limit_mps)
                * ORDER8_POST_FIRST_ARREST_CREEP_MULTIPLIER,
            )
        ),
        "order8_natural_contact_post_first_arrest_centroidal_transfer_max_observed_m": (
            max_post_first_arrest_centroidal_transfer_m
        ),
        "order8_natural_contact_max_contact_clearance_imbalance_m": (
            max_contact_clearance_imbalance_m
        ),
        "order8_natural_contact_pregrasp_staging_method": (
            "selected_urdf_mesh_aabb_axial_retreat_bisection_v1"
        ),
        "order8_natural_contact_pregrasp_mesh_clearance_target_m": float(
            config.pregrasp_mesh_clearance_m
        ),
        "order8_natural_contact_pregrasp_mesh_clearance_predicted_m": (
            staging_plan.predicted_clearance_m
        ),
        "order8_natural_contact_pregrasp_staging_retreat_distance_m": (
            staging_plan.retreat_distance_m
        ),
        "order8_natural_contact_pregrasp_approach_axis_world": list(
            staging_plan.approach_axis_world
        ),
        "order8_natural_contact_pregrasp_anchor_target_source": (
            "selected_urdf_mesh_aabb_outward_opening_in_base_frame_v1"
        ),
        "order8_natural_contact_pregrasp_opening_distance_m_by_anchor": {
            str(anchor_id): distance
            for anchor_id, distance in sorted(
                opening_plan.outward_distance_m_by_anchor.items()
            )
        },
        "order8_natural_contact_pregrasp_opening_clearance_m_by_anchor": {
            str(anchor_id): clearance
            for anchor_id, clearance in sorted(
                opening_plan.predicted_clearance_m_by_anchor.items()
            )
        },
        "order8_natural_contact_pregrasp_minimum_achieved_mesh_clearance_m": (
            minimum_achieved_pregrasp_clearance_m
        ),
        "order8_natural_contact_pregrasp_achieved_mesh_clearance_m": (
            pregrasp_achieved_mesh_clearance_m
        ),
        "order8_natural_contact_pregrasp_reachability_gate_passed": (
            pregrasp_reachability_gate_passed
        ),
        "order8_natural_contact_pregrasp_reachability_gate_source": (
            pregrasp_reachability_gate_source
        ),
        "order8_natural_contact_pregrasp_open_configuration_latched": (
            pregrasp_open_anchor_poses_base is not None
        ),
        "order8_natural_contact_contact_axial_alignment_latched": (
            contact_axial_aligned
        ),
        "order8_natural_contact_contact_motion_sequence": (
            "mesh_open_then_floor_clear_object_relative_base_settle_then_"
            "known_grasp_ready_pose_then_one_shot_whole_structure_direction_"
            "fixed_velocity_close_until_simultaneous_load_qclose_then_slow_"
            "per_side_load_limited_position_preload_v25"
        ),
        "order8_natural_contact_contact_mesh_precenter_method": (
            "one_shot_direction_seed_only_not_completion_gate_v3"
        ),
        "order8_natural_contact_contact_mesh_precenter_clearance_m": float(
            config.contact_near_surface_slowdown_m
        ),
        "order8_natural_contact_contact_mesh_precenter_tangential_tolerance_m": float(
            config.contact_tangential_tolerance_m
        ),
        "order8_natural_contact_mesh_pair_base_centering_method": (
            "horizontal_approach_axis_mean_authored_mesh_patch_centering_v1"
        ),
        "order8_natural_contact_mesh_pair_base_centering_correction_world": list(
            mesh_pair_base_centering_correction_world
        ),
        "order8_natural_contact_contact_mesh_precenter_complete": (
            contact_mesh_precenter_complete
        ),
        "order8_natural_contact_contact_mesh_precenter_dwell_s": (
            contact_mesh_precenter_dwell_s
        ),
        "order8_natural_contact_contact_mesh_precenter_completed_time_s": (
            contact_mesh_precenter_completed_time_s
        ),
        "order8_natural_contact_contact_centering_method": (
            "known_object_relative_centroidal_pose_hold_without_closure_mesh_"
            "feedback_v3"
        ),
        "order8_natural_contact_contact_centering_joint_motion_mode": (
            "all_docks_fixed_one_shot_velocity_ratio_without_receding_"
            "geometry_feedback_v4"
        ),
        "order8_natural_contact_contact_closure_common_translation_world": list(
            contact_closure_common_translation_world
        ),
        "order8_natural_contact_contact_closure_common_translation_active_step_count": (
            contact_closure_common_translation_active_step_count
        ),
        "order8_natural_contact_contact_closure_max_common_translation_m": (
            max_contact_closure_common_translation_m
        ),
        "order8_natural_contact_contact_individual_arrest_centroidal_hold": (
            "disabled_provisional_contact_may_separate_until_simultaneous_qclose_v1"
        ),
        "order8_natural_contact_contact_post_arrest_shape_hold_activation": (
            "simultaneous_qclose_only_v1"
        ),
        "order8_natural_contact_contact_centering_settle_gate": (
            "object_relative_final_base_pose_and_speed_dwell_before_joint_close_v1"
        ),
        "order8_natural_contact_contact_centering_raw_contact_input": False,
        "order8_natural_contact_contact_centering_max_offset_limit_m": float(
            config.contact_centering_max_offset_m
        ),
        "order8_natural_contact_contact_centering_max_tilt_limit_rad": float(
            config.contact_centering_max_tilt_rad
        ),
        "order8_natural_contact_contact_centering_tilt_source": (
            "not_used_in_surface_region_joint_only_close_v1"
        ),
        "order8_natural_contact_contact_centering_active_step_count": (
            contact_centering_active_step_count
        ),
        "order8_natural_contact_contact_continuous_balance_active_step_count": (
            contact_continuous_balance_active_step_count
        ),
        "order8_natural_contact_contact_sequential_reacquire_active_step_count": (
            contact_sequential_reacquire_active_step_count
        ),
        "order8_natural_contact_contact_sequential_centroidal_nudge_active_step_count": (
            contact_sequential_centroidal_nudge_active_step_count
        ),
        "order8_natural_contact_contact_sequential_latched_transfer_active_step_count": (
            contact_sequential_latched_transfer_active_step_count
        ),
        "order8_natural_contact_contact_sequential_joint_position_hold_step_count": (
            contact_sequential_joint_position_hold_step_count
        ),
        "order8_natural_contact_contact_sequential_reacquire_final_pursued_anchor_id": (
            contact_pursued_anchor_id
        ),
        "order8_natural_contact_contact_centering_cycle_count": (
            contact_centering_cycle_count
        ),
        "order8_natural_contact_contact_centering_max_observed_offset_m": (
            max_contact_centering_offset_m
        ),
        "order8_natural_contact_contact_centering_max_observed_tilt_rad": (
            max_contact_centering_tilt_rad
        ),
        "order8_natural_contact_contact_centering_max_measured_tilt_rad": (
            max_contact_centering_measured_tilt_rad
        ),
        "order8_natural_contact_contact_centering_latched_offset_world": (
            None
            if latched_contact_centering_offset_world is None
            else list(latched_contact_centering_offset_world)
        ),
        "order8_natural_contact_anchor_reference_frame": (
            "measured_free_object_relative_authored_mesh_contact_rebased_through_"
            "measured_base_v4"
        ),
        "order8_natural_contact_contact_tangential_region_method": (
            "authored_mesh_sample_componentwise_tangential_region_with_"
            "pair_mean_base_centering_v3"
        ),
        "order8_natural_contact_contact_tangential_tolerance_m": float(
            config.contact_tangential_tolerance_m
        ),
        "order8_natural_contact_provisional_contact_separation_allowed": True,
        "order8_natural_contact_contact_slip_enforcement_phase": (
            "diagnostic_all_safe_hold_disabled_evidence_only_v1"
            if diagnostic_disable_all_safe_hold
            else "grasp_latched_object_frame_contact_point_displacement_v1"
        ),
        "order8_natural_contact_contact_slip_measurement_method": (
            "force_weighted_selected_contact_centroid_object_frame_"
            "displacement_norm_from_grasp_confirmation_v1"
        ),
        "order8_natural_contact_contact_break_enforcement_phase": (
            "after_verified_two_contact_grasp_dwell_until_planned_release_v2"
        ),
        "order8_natural_contact_max_provisional_acquisition_slip_speed_mps": (
            result.max_provisional_acquisition_slip_speed_mps
        ),
        "order8_natural_contact_object_motion_retargeting_enabled": True,
        "order8_natural_contact_object_motion_retarget_source": (
            "measured_free_object_pose_read_only_v1"
        ),
        "order8_natural_contact_object_follow_active_step_count": (
            contact_object_follow_active_step_count
        ),
        "order8_natural_contact_max_observed_pre_qclose_object_translation_m": (
            max_contact_object_translation_m
        ),
        "order8_natural_contact_max_observed_base_retarget_translation_m": (
            max_contact_base_retarget_translation_m
        ),
        "order8_natural_contact_object_follow_pose_write_count": 0,
        "order8_natural_contact_morphology_aware_module_root_targets": True,
        "order8_natural_contact_module_root_target_source": (
            "whole_structure_fk_of_measured_absolute_dock_state_and_"
            "planner_base_pose_v4"
        ),
        "order8_natural_contact_module_frame_link_id": module_frame_link_id,
        "order8_natural_contact_spawn_pose_conversion": (
            "graph_module_frame_to_urdf_root_v1"
        ),
        "order8_natural_contact_runtime_module_pose_source": (
            "isaac_named_module_frame_link_pose_and_twist_v1"
        ),
        "order8_natural_contact_qpid_centroidal_target_source": (
            "single_full_morphology_rigid_body_model_from_planner_base_pose_"
            "and_measured_absolute_dock_state_v5"
        ),
        "order8_natural_contact_qpid_joint_motion_assumption": (
            "quasi_static_measured_shape_without_commanded_joint_motion_"
            "compensation_even_during_slow_preload_v2"
        ),
        "order8_natural_contact_qpid_unreached_joint_target_compensation": False,
        "order8_natural_contact_contact_force_joint_impedance_mode": (
            (
                "diagnostic_post_grasp_closure_direction_offset_torque_v1"
                if diagnostic_post_grasp_joint_torque_bias_nm is not None
                else "per_anchor_load_limited_position_preload_with_zero_offset_torque_v6"
            )
        ),
        "order8_natural_contact_contact_force_joint_impedance_raw_contact_input": (
            False
        ),
        "order8_natural_contact_contact_force_joint_impedance_active_step_count": (
            contact_force_impedance_active_step_count
        ),
        "order8_natural_contact_contact_force_position_preload_method": (
            "fixed_closure_ratio_previous_target_integration_per_anchor_"
            "damping_compensated_load_dwell_and_freeze_v3"
        ),
        # Retained only for older report readers; the active v8 command is
        # joint-space and the correctly dimensioned field follows.
        "order8_natural_contact_contact_force_position_preload_speed_limit_mps": (
            0.0
        ),
        "order8_natural_contact_contact_position_preload_joint_speed_radps": (
            float(config.contact_position_preload_joint_speed_radps)
        ),
        "order8_natural_contact_contact_position_preload_load_threshold_nm": (
            contact_position_preload_load_threshold_nm
        ),
        "order8_natural_contact_contact_position_preload_complete": (
            contact_position_preload_complete
        ),
        "order8_natural_contact_contact_position_preload_completion_source": (
            contact_position_preload_completion_source
        ),
        "order8_natural_contact_contact_position_preload_joint_ids_by_anchor": {
            str(anchor_id): list(joint_ids)
            for anchor_id, joint_ids in sorted(
                contact_position_preload_joint_ids_by_anchor.items()
            )
        },
        "order8_natural_contact_contact_position_preload_velocity_targets_radps": dict(
            sorted(contact_position_preload_velocity_targets_radps.items())
        ),
        "order8_natural_contact_contact_position_preload_position_targets_rad": dict(
            sorted(contact_position_preload_position_targets_rad.items())
        ),
        "order8_natural_contact_contact_position_preload_load_nm_by_anchor": {
            str(anchor_id): value
            for anchor_id, value in sorted(
                contact_position_preload_load_nm_by_anchor.items()
            )
        },
        "order8_natural_contact_contact_position_preload_max_load_nm_by_anchor": {
            str(anchor_id): value
            for anchor_id, value in sorted(
                contact_position_preload_max_load_nm_by_anchor.items()
            )
        },
        "order8_natural_contact_contact_position_preload_load_dwell_s_by_anchor": {
            str(anchor_id): value
            for anchor_id, value in sorted(
                contact_position_preload_load_dwell_s_by_anchor.items()
            )
        },
        "order8_natural_contact_contact_position_preload_frozen_anchor_ids": sorted(
            contact_position_preload_frozen_anchor_ids
        ),
        "order8_natural_contact_contact_position_preload_frozen_time_s_by_anchor": {
            str(anchor_id): value
            for anchor_id, value in sorted(
                contact_position_preload_frozen_time_s_by_anchor.items()
            )
        },
        "order8_natural_contact_contact_position_preload_active_step_count": (
            contact_position_preload_active_step_count
        ),
        "order8_natural_contact_contact_force_position_preload_active_step_count": (
            contact_force_position_preload_active_step_count
        ),
        "order8_natural_contact_latched_joint_position_hold_method": (
            "diagnostic_qclose_rigid_grasp_following_commanded_centroidal_path_"
            "two_anchor_full_dock_dls_position_outer_loop_v2"
            if diagnostic_anchor_hold_joint_correction
            else (
                "absolute_final_load_limited_preload_target_with_diagnostic_"
                "closure_direction_torque_bias_v1"
                if diagnostic_post_grasp_joint_torque_bias_nm is not None
                else "absolute_final_load_limited_preload_target_with_zero_torque_bias_v6"
            )
        ),
        "order8_natural_contact_latched_joint_position_hold_step_count": (
            latched_joint_position_hold_step_count
        ),
        "order8_natural_contact_contact_force_joint_impedance_max_joint_count": (
            max_contact_force_impedance_joint_count
        ),
        "order8_natural_contact_contact_force_joint_impedance_peak_clipped_joint_ids": (
            sorted(contact_force_impedance_peak_clipped_joint_ids)
        ),
        "order8_natural_contact_contact_force_joint_impedance_position_clipped_joint_ids": (
            sorted(contact_force_impedance_position_clipped_joint_ids)
        ),
        "order8_natural_contact_contact_joint_drive_damping_multiplier": (
            float(config.contact_joint_drive_damping_multiplier)
        ),
        "order8_natural_contact_contact_joint_drive_nominal_damping_nms_per_rad": (
            float(dock_damping)
        ),
        "order8_natural_contact_contact_joint_drive_maximum_damping_nms_per_rad": (
            ORDER8_SIMULATION_DRIVE_DAMPING_MAX_NMS_PER_RAD
        ),
        "order8_natural_contact_contact_joint_drive_damping_limit_basis": (
            "simulator_numerical_gain_with_independent_ak40_torque_current_"
            "and_speed_envelope_audit_v1"
        ),
        "order8_natural_contact_contact_joint_drive_damping_scheduled": (
            contact_joint_drive_damping_scheduled
        ),
        "order8_natural_contact_contact_joint_drive_damping_targets_nms_per_rad": (
            dict(sorted(contact_joint_drive_damping_targets.items()))
        ),
        "order8_natural_contact_morphology_aware_module_root_target_count": (
            morphology_aware_module_root_target_count
        ),
        "order8_natural_contact_max_base_target_step_m": max_base_target_step_m,
        "order8_natural_contact_max_contact_base_target_step_m": (
            max_contact_base_target_step_m
        ),
        "order8_natural_contact_max_observed_joint_limit_violation_rad": (
            max_observed_joint_limit_violation_rad
        ),
        "order8_natural_contact_last_base_terminal_tracking_error_m": (
            last_base_terminal_tracking_error_m
        ),
        "order8_natural_contact_last_base_command_tracking_error_m": (
            last_base_command_tracking_error_m
        ),
        "order8_natural_contact_last_joint_positions_rad": last_joint_positions,
        "order8_natural_contact_last_anchor_position_error_m": (
            max_anchor_position_error_m
        ),
        "order8_natural_contact_last_anchor_reference_terminal_error_m": (
            max_anchor_reference_terminal_error_m
        ),
        "order8_natural_contact_last_base_linear_speed_mps": (base_linear_speed_mps),
        "order8_natural_contact_contact_command_dwell_s": (
            nonprivileged_contact_command_dwell_s
        ),
        "order8_natural_contact_prelift_relative_motion_settle_method": (
            "all_selected_nonprivileged_surface_point_object_relative_speed_"
            "below_half_slip_limit_for_contact_dwell_v2"
        ),
        "order8_natural_contact_prelift_relative_speed_threshold_mps": (
            prelift_relative_speed_threshold_mps
        ),
        "order8_natural_contact_prelift_relative_motion_settle_achieved": (
            prelift_relative_motion_settle_achieved
        ),
        "order8_natural_contact_contact_force_hold_settle_gate_method": (
            "per_anchor_damping_compensated_moving_chain_load_dwell_then_"
            "full_link_object_relative_speed_dwell_v8"
        ),
        "order8_natural_contact_contact_force_hold_settle_raw_contact_input": False,
        "order8_natural_contact_grasp_confirmation_method": (
            "load_limited_position_preload_freeze_and_full_relative_motion_"
            "settle_then_privileged_safety_interlocked_raw_contact_dwell_v10"
        ),
        "order8_natural_contact_contact_required_motion_safety_interlock": (
            "privileged_two_selected_link_raw_contact_dwell_v1"
        ),
        "order8_natural_contact_contact_motion_safety_interlock_actor_input": False,
        "order8_natural_contact_contact_motion_safety_interlock_qpid_command": False,
        "order8_natural_contact_contact_motion_safety_interlock_blocked_step_count": (
            contact_motion_safety_interlock_blocked_step_count
        ),
        "order8_natural_contact_anchor_object_relative_speed_source": (
            "selected_authored_mesh_sample_point_link_object_kinematic_twist_v2"
        ),
        "order8_natural_contact_anchor_object_normal_relative_speed_source": (
            "selected_authored_mesh_sample_point_object_obb_surface_normal_"
            "projected_link_object_kinematic_twist_v1"
        ),
        "order8_natural_contact_contact_force_hold_speed_reference_frame": (
            "first_order_low_pass_of_signed_sampled_mesh_to_observed_object_obb_"
            "clearance_rate_v3"
        ),
        "order8_natural_contact_contact_force_hold_speed_filter_time_constant_s": (
            float(config.contact_stall_dwell_s)
        ),
        "order8_natural_contact_contact_force_hold_speed_filter_raw_contact_input": (
            False
        ),
        "order8_natural_contact_contact_closure_detection": (
            "simultaneous_selected_terminal_joint_load_dwell_then_measured_"
            "qclose_and_privileged_contact_validation_v18"
        ),
        "order8_natural_contact_contact_anchor_orientation_task_weight": float(
            ORDER8_FREE_MORPH_ANCHOR_ORIENTATION_WEIGHT
        ),
        "order8_natural_contact_contact_anchor_task_hierarchy": (
            "contact_translation_primary_measured_orientation_zero_error_then_"
            "verified_absolute_joint_hold_v2"
        ),
        "order8_natural_contact_contact_terminal_inward_overtravel_m": (
            float(config.contact_closure_inward_overtravel_m)
        ),
        "order8_natural_contact_provisional_surface_load_settle_method": (
            "one_sided_contact_may_separate_continuous_bounded_creep_until_"
            "simultaneous_nonprivileged_surface_load_qclose_v3"
        ),
        "order8_natural_contact_provisional_surface_load_settle_raw_contact_input": (
            False
        ),
        "order8_natural_contact_provisional_surface_load_settle_active_step_count": (
            contact_provisional_surface_settle_active_step_count
        ),
        "order8_natural_contact_contact_closure_raw_contact_input": False,
        "order8_natural_contact_contact_terminal_target_snapshotted": (
            contact_terminal_anchor_targets is not None
        ),
        "order8_natural_contact_release_terminal_target_snapshotted": (
            release_terminal_anchor_targets is not None
        ),
        "order8_natural_contact_release_terminal_target_source": (
            "measured_closure_start_qopen_anchor_poses_base_v2"
            if simple_closure_initialized_time_s is not None
            else (
                "measured_pregrasp_qopen_anchor_poses_base_v1"
                if pregrasp_open_anchor_poses_base is not None
                else "ideal_opening_plan_before_qopen_latch"
            )
        ),
        "order8_natural_contact_contact_configuration_latched": (
            contact_configuration_latched
        ),
        "order8_natural_contact_post_qclose_joint_settle_complete": (
            post_qclose_joint_settle_complete
        ),
        "order8_natural_contact_post_qclose_joint_settle_dwell_s": (
            post_qclose_joint_settle_dwell_s
        ),
        "order8_natural_contact_post_qclose_joint_speed_threshold_radps": (
            post_qclose_joint_speed_threshold_radps
        ),
        "order8_natural_contact_post_qclose_max_measured_joint_speed_radps": (
            post_qclose_max_measured_joint_speed_radps
        ),
        "order8_natural_contact_post_qclose_position_rebase_step_count": (
            post_qclose_position_rebase_step_count
        ),
        "order8_natural_contact_post_qclose_geometric_preload_complete": (
            post_qclose_geometric_preload_complete
        ),
        "order8_natural_contact_post_qclose_geometric_preload_distance_m": float(
            config.contact_closure_inward_overtravel_m
        ),
        "order8_natural_contact_post_qclose_geometric_preload_terminal_error_m": (
            post_qclose_geometric_preload_terminal_error_m
        ),
        "order8_natural_contact_post_qclose_geometric_preload_tracking_error_m": (
            post_qclose_geometric_preload_tracking_error_m
        ),
        "order8_natural_contact_post_qclose_geometric_preload_achieved_inward_"
        "displacement_m_by_anchor": {
            str(anchor_id): value
            for anchor_id, value in sorted(
                post_qclose_geometric_preload_achieved_inward_displacement_m_by_anchor.items()
            )
        },
        "order8_natural_contact_post_qclose_geometric_preload_initial_surface_"
        "point_object_by_anchor": {
            str(anchor_id): list(point)
            for anchor_id, point in sorted(
                post_qclose_geometric_preload_initial_surface_point_object_by_anchor.items()
            )
        },
        "order8_natural_contact_post_qclose_geometric_preload_terminal_surface_"
        "point_object_by_anchor": {
            str(anchor_id): list(pose[:3])
            for anchor_id, pose in sorted(
                post_qclose_geometric_preload_anchor_poses_object.items()
            )
        },
        "order8_natural_contact_post_qclose_geometric_preload_current_surface_"
        "point_world_by_anchor": {
            str(anchor_id): list(point)
            for anchor_id, point in sorted(
                post_qclose_geometric_preload_current_surface_point_world_by_anchor.items()
            )
        },
        "order8_natural_contact_post_qclose_geometric_preload_settle_dwell_s": (
            post_qclose_geometric_preload_settle_dwell_s
        ),
        "order8_natural_contact_post_qclose_geometric_preload_active_step_count": (
            post_qclose_geometric_preload_active_step_count
        ),
        "order8_natural_contact_post_qclose_geometric_preload_measured_position_"
        "reference_step_count": (
            post_qclose_geometric_preload_measured_position_reference_step_count
        ),
        "order8_natural_contact_contact_closure_measured_position_reference_"
        "step_count": contact_closure_measured_position_reference_step_count,
        "order8_natural_contact_post_qclose_geometric_preload_load_arrest_"
        "candidate_by_anchor": {
            str(anchor_id): value
            for anchor_id, value in sorted(
                post_qclose_geometric_preload_load_arrest_candidates.items()
            )
        },
        "order8_natural_contact_post_qclose_geometric_preload_completion_source": (
            post_qclose_geometric_preload_completion_source
        ),
        "order8_natural_contact_post_qclose_geometric_preload_method": (
            "not_applicable_replaced_by_joint_space_load_limited_preload_v5"
        ),
        "order8_natural_contact_post_qclose_settle_method": (
            "measured_absolute_joint_position_rebase_zero_velocity_target_"
            "before_slow_previous_target_position_preload_v2"
        ),
        "order8_natural_contact_contact_closure_reason": contact_closure_reason,
        "order8_natural_contact_contact_stall_latched": contact_stall_latched,
        "order8_natural_contact_contact_stall_dwell_s": (
            nonprivileged_contact_stall_dwell_s
        ),
        "order8_natural_contact_contact_configuration_dwell_s": (
            nonprivileged_contact_configuration_dwell_s
        ),
        "order8_natural_contact_contact_surface_load_arrest_candidate_by_anchor": {
            str(anchor_id): value
            for anchor_id, value in sorted(
                last_contact_surface_load_arrest_candidates.items()
            )
        },
        "order8_natural_contact_contact_surface_load_arrest_selected_joint_source": (
            "per_anchor_terminal_mechanism_joint_isaac_applied_torque_or_"
            "hardware_current_torque_estimate_v3"
        ),
        "order8_natural_contact_contact_stall_dwell_s_by_anchor": {
            str(anchor_id): value
            for anchor_id, value in sorted(
                nonprivileged_contact_stall_dwell_s_by_anchor.items()
            )
        },
        "order8_natural_contact_contact_stall_latched_anchor_ids": sorted(
            contact_stall_latched_anchor_poses_base
        ),
        "order8_natural_contact_contact_stall_command_error_m_by_anchor": {
            str(anchor_id): value
            for anchor_id, value in sorted(
                last_contact_stall_command_error_m_by_anchor.items()
            )
        },
        "order8_natural_contact_contact_stall_anchor_speed_mps_by_anchor": {
            str(anchor_id): value
            for anchor_id, value in sorted(
                last_contact_stall_anchor_speed_mps_by_anchor.items()
            )
        },
        "order8_natural_contact_last_anchor_object_relative_speed_mps_by_anchor": {
            str(anchor_id): value
            for anchor_id, value in sorted(
                last_anchor_object_relative_speed_mps_by_anchor.items()
            )
        },
        "order8_natural_contact_max_anchor_object_relative_speed_mps_by_anchor": {
            str(anchor_id): value
            for anchor_id, value in sorted(
                max_anchor_object_relative_speed_mps_by_anchor.items()
            )
        },
        "order8_natural_contact_last_anchor_object_normal_relative_speed_mps_by_anchor": {
            str(anchor_id): value
            for anchor_id, value in sorted(
                last_anchor_object_normal_relative_speed_mps_by_anchor.items()
            )
        },
        "order8_natural_contact_max_anchor_object_normal_relative_speed_mps_by_anchor": {
            str(anchor_id): value
            for anchor_id, value in sorted(
                max_anchor_object_normal_relative_speed_mps_by_anchor.items()
            )
        },
        "order8_natural_contact_last_anchor_object_filtered_normal_relative_speed_mps_by_anchor": {
            str(anchor_id): value
            for anchor_id, value in sorted(
                last_anchor_object_filtered_normal_relative_speed_mps_by_anchor.items()
            )
        },
        "order8_natural_contact_max_anchor_object_filtered_normal_relative_speed_mps_by_anchor": {
            str(anchor_id): value
            for anchor_id, value in sorted(
                max_anchor_object_filtered_normal_relative_speed_mps_by_anchor.items()
            )
        },
        "order8_natural_contact_last_contact_mesh_surface_clearance_rate_mps_by_anchor": {
            str(anchor_id): value
            for anchor_id, value in sorted(
                last_gripper_surface_clearance_rate_mps_by_anchor.items()
            )
        },
        "order8_natural_contact_last_contact_mesh_surface_filtered_clearance_speed_mps_by_anchor": {
            str(anchor_id): value
            for anchor_id, value in sorted(
                last_filtered_gripper_surface_clearance_rate_mps_by_anchor.items()
            )
        },
        "order8_natural_contact_max_contact_mesh_surface_filtered_clearance_speed_mps_by_anchor": {
            str(anchor_id): value
            for anchor_id, value in sorted(
                max_filtered_gripper_surface_clearance_rate_mps_by_anchor.items()
            )
        },
        "order8_natural_contact_contact_stall_selected_joint_id_by_anchor": {
            str(anchor_id): joint_id
            for anchor_id, joint_id in sorted(
                selected_contact_joint_id_by_anchor.items()
            )
        },
        "order8_natural_contact_contact_stall_selected_joint_load_nm_by_anchor": {
            str(anchor_id): value
            for anchor_id, value in sorted(
                last_contact_stall_selected_joint_load_nm_by_anchor.items()
            )
        },
        "order8_natural_contact_last_selected_joint_raw_applied_torque_nm_by_anchor": {
            str(anchor_id): value
            for anchor_id, value in sorted(
                selected_contact_raw_joint_torque_nm_by_anchor.items()
            )
        },
        "order8_natural_contact_last_selected_joint_raw_applied_load_nm_by_anchor": {
            str(anchor_id): value
            for anchor_id, value in sorted(
                selected_contact_raw_joint_load_nm_by_anchor.items()
            )
        },
        "order8_natural_contact_last_selected_joint_damping_drive_torque_nm_by_anchor": {
            str(anchor_id): value
            for anchor_id, value in sorted(
                selected_contact_damping_drive_torque_nm_by_anchor.items()
            )
        },
        "order8_natural_contact_contact_stall_selected_joint_load_threshold_nm": (
            contact_stall_selected_joint_load_threshold_nm
        ),
        "order8_natural_contact_contact_stall_selected_joint_load_source": (
            "absolute_per_anchor_terminal_mechanism_joint_isaac_applied_"
            "torque_minus_estimated_virtual_drive_damping_torque_v4"
        ),
        "order8_natural_contact_contact_stall_influential_joint_ids_by_anchor": {
            str(anchor_id): list(joint_ids)
            for anchor_id, joint_ids in sorted(
                contact_stall_influential_joint_ids_by_anchor.items()
            )
        },
        "order8_natural_contact_contact_stall_speed_reference_frame": (
            "first_order_low_pass_selected_mesh_sample_point_object_normal_"
            "relative_speed_v2"
        ),
        "order8_natural_contact_contact_configuration_base_speed_tolerance_mps": (
            contact_axial_settle_base_speed_tolerance_mps
        ),
        "order8_natural_contact_contact_configuration_base_speed_gate": (
            "world_base_linear_speed_with_object_relative_target_follow_v1"
        ),
        "order8_natural_contact_contact_mesh_clearance_arm_threshold_m": float(
            config.contact_surface_arm_clearance_m
        ),
        "order8_natural_contact_contact_mesh_clearance_reacquire_tolerance_m": float(
            config.contact_penetration_noise_floor_m
        ),
        "order8_natural_contact_contact_mesh_surface_distance_method": (
            "sampled_urdf_collision_mesh_surface_to_observed_object_obb_v1"
        ),
        "order8_natural_contact_contact_wrench_application_mapping": (
            (
                "diagnostic_fixed_closure_direction_local_joint_offset_torque_v1"
                if diagnostic_post_grasp_joint_torque_bias_nm is not None
                else "high_level_semantic_only_local_joint_offset_torque_forced_zero_v4"
            )
        ),
        "order8_natural_contact_contact_wrench_application_raw_contact_input": False,
        "order8_natural_contact_contact_mesh_surface_sample_count": sum(
            len(bounds.surface_sample_points_local)
            for bounds in selected_gripper_local_aabbs
        ),
        "order8_natural_contact_contact_mesh_surface_clearance_at_latch_m_by_anchor": {
            str(anchor_id): value
            for anchor_id, value in sorted(
                contact_stall_latched_mesh_clearance_m_by_anchor.items()
            )
        },
        "order8_natural_contact_contact_tangential_offset_at_latch_m_by_anchor": {
            str(anchor_id): list(offsets_m)
            for anchor_id, offsets_m in sorted(
                contact_stall_latched_tangential_offset_m_by_anchor.items()
            )
        },
        "order8_natural_contact_last_contact_mesh_surface_clearance_m_by_anchor": {
            str(anchor_id): value
            for anchor_id, value in sorted(
                gripper_surface_clearance_m_by_anchor.items()
            )
        },
        "order8_natural_contact_grasp_hold_anchor_target_source": (
            "simultaneous_surface_region_qclose_measured_anchor_poses_in_base_frame_v3"
        ),
        "order8_natural_contact_grasp_hold_anchor_count": (
            0
            if grasp_hold_anchor_poses_base is None
            else len(grasp_hold_anchor_poses_base)
        ),
        "order8_natural_contact_contact_force_ramp_elapsed_s": min(
            nonprivileged_contact_force_ramp_elapsed_s_by_anchor.values(),
            default=0.0,
        ),
        "order8_natural_contact_contact_force_ramp_elapsed_s_by_anchor": {
            str(anchor_id): value
            for anchor_id, value in sorted(
                nonprivileged_contact_force_ramp_elapsed_s_by_anchor.items()
            )
        },
        "order8_natural_contact_last_contact_force_scale": contact_force_scale,
        "order8_natural_contact_last_contact_force_scale_by_anchor": {
            str(anchor_id): value
            for anchor_id, value in sorted(contact_force_scale_by_anchor.items())
        },
        "order8_natural_contact_max_contact_force_scale_by_anchor": {
            str(anchor_id): value
            for anchor_id, value in sorted(max_contact_force_scale_by_anchor.items())
        },
        "order8_natural_contact_release_command_dwell_s": (
            nonprivileged_release_command_dwell_s
        ),
        "order8_natural_contact_actuator_mapping_hash": (
            full_actuator_mapping.stable_hash()
        ),
        "order8_natural_contact_component_actuator_mapping_hashes": {
            str(module_id): component_mappings[module_id].stable_hash()
            for module_id in module_ids
        },
        "order8_natural_contact_free_object": True,
        "order8_natural_contact_object_kinematic": False,
        "order8_natural_contact_object_root_pose_write_count": (
            object_root_pose_write_count
        ),
        "order8_natural_contact_object_root_pose_write_audit_method": (
            "instrumented_post_spawn_object_pose_write_counter_v1"
        ),
        "order8_natural_contact_object_constraint_created": bool(
            object_constraint_references
        ),
        "order8_natural_contact_object_constraint_stage_audit_method": (
            "usd_physics_joint_body_target_scan_v1"
        ),
        "order8_natural_contact_object_constraint_reference_count": len(
            object_constraint_references
        ),
        "order8_natural_contact_object_constraint_prim_paths": (
            object_constraint_prim_paths
        ),
        "order8_natural_contact_pre_contact_object_pose_hold": False,
        "order8_natural_contact_kinematic_payload_attach_used": False,
        "order8_natural_contact_dynamic_assembly_filter_fallback_used": False,
        "order8_natural_contact_selected_surface_actual_dock_mesh": (
            not diagnostic_proxy_pad_enabled
        ),
        "order8_natural_contact_selected_surface_contact_representation": (
            "diagnostic_cone_surface_micro_pad_v1"
            if diagnostic_cone_proxy_pad_enabled
            else "diagnostic_finite_area_proxy_pad_v1"
            if diagnostic_legacy_proxy_pad_enabled
            else "authored_dock_collision_mesh_compliant_contact_v2"
        ),
        "order8_natural_contact_diagnostic_proxy_pad_enabled": (
            diagnostic_proxy_pad_enabled
        ),
        "order8_natural_contact_diagnostic_proxy_pad_acceptance_eligible": False,
        "order8_natural_contact_diagnostic_proxy_pad_method": (
            "approved_cone_only_local_triangle_tiles_authored_collision_disabled_v1"
            if diagnostic_cone_proxy_pad_enabled
            else "sampled_mesh_outer_face_connect_frame_aligned_retained_mesh_v1"
            if diagnostic_legacy_proxy_pad_enabled
            else "not_enabled"
        ),
        "order8_natural_contact_diagnostic_proxy_pad_prim_paths": (
            diagnostic_proxy_pad_prim_paths
        ),
        "order8_natural_contact_diagnostic_proxy_pad_specs": [
            asdict(spec) for spec in diagnostic_proxy_pad_specs
        ],
        "order8_natural_contact_diagnostic_proxy_pad_retained_authored_mesh": True,
        "order8_natural_contact_diagnostic_proxy_pad_authored_collision_enabled": (
            not diagnostic_cone_proxy_pad_enabled
        ),
        "order8_natural_contact_diagnostic_proxy_pad_authored_collision_paths": (
            diagnostic_proxy_pad_authored_collision_paths
        ),
        "order8_natural_contact_diagnostic_proxy_pad_disabled_authored_collision_paths": (
            diagnostic_proxy_pad_disabled_authored_collision_paths
        ),
        "order8_natural_contact_diagnostic_proxy_pad_deinstanced_prim_paths": (
            diagnostic_proxy_pad_deinstanced_prim_paths
        ),
        "order8_natural_contact_diagnostic_proxy_pad_independent_rigid_body_count": 0,
        "order8_natural_contact_diagnostic_proxy_pad_exclusive_under_penetration_limit": (
            diagnostic_proxy_pad_exclusive_under_penetration_limit
        ),
        "order8_natural_contact_diagnostic_proxy_pad_missing_collision_paths": (
            diagnostic_proxy_pad_missing_collision_paths
        ),
        "order8_natural_contact_selected_dock_link_ids": selected_link_ids,
        "order8_natural_contact_selected_contact_pair_count": len(selected_link_ids),
        "order8_natural_contact_selected_surface_module_ids": [
            surface.module_id for surface in selected_surfaces
        ],
        "order8_natural_contact_selected_surface_port_global_ids": [
            surface.port_global_id for surface in selected_surfaces
        ],
        "order8_natural_contact_selected_surface_geometry_refs": sorted(
            primitive.geometry_ref
            for surface in selected_surfaces
            for primitive in surface.collision_primitives
            if primitive.geometry_ref is not None
        ),
        "order8_natural_contact_selected_gripper_material_method": (
            "cone_micro_pad_only_compliant_material_v1"
            if diagnostic_cone_proxy_pad_enabled
            else "legacy_diagnostic_proxy_and_authored_mesh_compliant_material_v4"
            if diagnostic_legacy_proxy_pad_enabled
            else "selected_authored_dock_mesh_compliant_material_v3"
        ),
        "order8_natural_contact_selected_gripper_material_path": (
            ORDER8_SELECTED_GRIPPER_MATERIAL_PATH
        ),
        "order8_natural_contact_selected_gripper_static_friction": float(
            config.selected_gripper_friction
        ),
        "order8_natural_contact_selected_gripper_dynamic_friction": float(
            config.selected_gripper_friction
        ),
        "order8_natural_contact_selected_gripper_friction_combine_mode": (
            ORDER8_SELECTED_GRIPPER_FRICTION_COMBINE_MODE
        ),
        "order8_natural_contact_selected_gripper_compliant_contact_enabled": True,
        "order8_natural_contact_selected_gripper_compliant_contact_stiffness_n_per_m": (
            float(selected_gripper_compliant_contact_stiffness)
        ),
        "order8_natural_contact_selected_gripper_compliant_contact_damping_n_s_per_m": (
            float(selected_gripper_compliant_contact_damping)
        ),
        "order8_natural_contact_selected_gripper_compliant_contact_audit_passed": (
            selected_gripper_compliant_contact_audit_passed
        ),
        "order8_natural_contact_selected_gripper_material_binding_strength": (
            "strongerThanDescendants"
        ),
        "order8_natural_contact_selected_gripper_material_body_paths": (
            selected_gripper_material_body_paths
        ),
        "order8_natural_contact_selected_gripper_material_collision_prim_paths": (
            selected_gripper_material_collision_prim_paths
        ),
        "order8_natural_contact_selected_gripper_material_collision_prim_count": len(
            selected_gripper_material_collision_prim_paths
        ),
        "order8_natural_contact_selected_gripper_material_binding_audit_passed": (
            not selected_gripper_material_binding_failures
        ),
        "order8_natural_contact_last_selected_normal_force_n_by_link": dict(
            sorted(last_selected_normal_force_n_by_link.items())
        ),
        "order8_natural_contact_max_selected_normal_force_n_by_link": dict(
            sorted(max_selected_normal_force_n_by_link.items())
        ),
        "order8_natural_contact_contact_vector_telemetry_role": (
            "privileged_diagnostic_only_not_actor_or_qpid_input_v2"
        ),
        "order8_natural_contact_contact_vector_telemetry_invalid_step_count": (
            contact_vector_telemetry_invalid_step_count
        ),
        "order8_natural_contact_last_selected_contact_normal_force_world_n_by_link": {
            link_id: list(vector)
            for link_id, vector in sorted(
                last_selected_contact_normal_force_world_by_link.items()
            )
        },
        "order8_natural_contact_last_selected_contact_application_point_world_by_link": {
            link_id: list(point)
            for link_id, point in sorted(
                last_selected_contact_application_point_world_by_link.items()
            )
        },
        "order8_natural_contact_last_selected_friction_force_world_n_by_link": {
            link_id: list(vector)
            for link_id, vector in sorted(
                last_selected_friction_force_world_by_link.items()
            )
        },
        "order8_natural_contact_last_selected_contact_force_matrix_world_n_by_link": {
            link_id: list(vector)
            for link_id, vector in sorted(
                last_selected_contact_force_matrix_world_by_link.items()
            )
        },
        "order8_natural_contact_last_selected_body_linear_velocity_world_mps_by_link": {
            link_id: list(vector)
            for link_id, vector in sorted(
                last_selected_body_linear_velocity_world_by_link.items()
            )
        },
        "order8_natural_contact_last_selected_dock_contact_point_velocity_world_mps_by_link": {
            link_id: list(vector)
            for link_id, vector in sorted(
                last_selected_body_contact_velocity_world_by_link.items()
            )
        },
        "order8_natural_contact_last_selected_object_contact_point_velocity_world_mps_by_link": {
            link_id: list(vector)
            for link_id, vector in sorted(
                last_selected_object_contact_velocity_world_by_link.items()
            )
        },
        "order8_natural_contact_last_selected_relative_contact_velocity_world_mps_by_link": {
            link_id: list(vector)
            for link_id, vector in sorted(
                last_selected_relative_contact_velocity_world_by_link.items()
            )
        },
        "order8_natural_contact_last_selected_tangential_slip_velocity_world_mps_by_link": {
            link_id: list(vector)
            for link_id, vector in sorted(
                last_selected_tangential_slip_velocity_world_by_link.items()
            )
        },
        "order8_natural_contact_last_selected_tangential_slip_velocity_object_mps_by_link": {
            link_id: list(vector)
            for link_id, vector in sorted(
                last_selected_tangential_slip_velocity_object_by_link.items()
            )
        },
        "order8_natural_contact_last_selected_slip_contact_point_world_by_link": {
            link_id: list(point)
            for link_id, point in sorted(
                last_selected_slip_contact_point_world_by_link.items()
            )
        },
        "order8_natural_contact_last_selected_slip_contact_normal_world_by_link": {
            link_id: list(vector)
            for link_id, vector in sorted(
                last_selected_slip_contact_normal_world_by_link.items()
            )
        },
        "order8_natural_contact_signed_cumulative_slip_displacement_world_m_by_link": {
            link_id: list(vector)
            for link_id, vector in sorted(
                signed_cumulative_slip_displacement_world_m_by_link.items()
            )
        },
        "order8_natural_contact_signed_cumulative_slip_displacement_object_m_by_link": {
            link_id: list(vector)
            for link_id, vector in sorted(
                signed_cumulative_slip_displacement_object_m_by_link.items()
            )
        },
        "order8_natural_contact_signed_cumulative_slip_dominant_world_axis_by_link": {
            link_id: _dominant_signed_vector_axis(vector)
            for link_id, vector in sorted(
                signed_cumulative_slip_displacement_world_m_by_link.items()
            )
        },
        "order8_natural_contact_signed_cumulative_slip_dominant_object_axis_by_link": {
            link_id: _dominant_signed_vector_axis(vector)
            for link_id, vector in sorted(
                signed_cumulative_slip_displacement_object_m_by_link.items()
            )
        },
        "order8_natural_contact_diagnostic_cumulative_slip_path_m_by_link": dict(
            sorted(diagnostic_cumulative_slip_path_m_by_link.items())
        ),
        "order8_natural_contact_diagnostic_contact_point_vertical_velocity_bounds_mps_by_stage": (
            diagnostic_contact_point_vertical_velocity_bounds_mps_by_stage
        ),
        "order8_natural_contact_slip_vector_step_telemetry": (
            slip_vector_step_telemetry
        ),
        "order8_natural_contact_max_selected_friction_force_magnitude_n_by_link": dict(
            sorted(max_selected_friction_force_magnitude_n_by_link.items())
        ),
        "order8_natural_contact_max_abs_selected_friction_vertical_force_n_by_link": dict(
            sorted(max_abs_selected_friction_vertical_force_n_by_link.items())
        ),
        "order8_natural_contact_gripper_clearance_geometry": (
            "urdf_collision_mesh_local_aabb_world_aabb_v1"
        ),
        "order8_natural_contact_last_measured_object_pose": list(
            object_state["pose"]
        ),
        "order8_natural_contact_last_measured_object_twist": list(
            object_state["twist"]
        ),
        "order8_natural_contact_last_measured_base_module_pose": list(base_root_pose),
        "order8_natural_contact_last_measured_selected_anchor_poses_world": {
            str(anchor_id): list(
                measured_selected_anchor_poses_world_by_anchor[anchor_id]
            )
            for anchor_id in selected_anchor_ids
        },
        "order8_natural_contact_kinematic_consistency_method": (
            "same_cycle_measured_module_anchor_and_latched_mesh_material_point_"
            "versus_graph_constrained_urdf_fk_v1"
        ),
        "order8_natural_contact_kinematic_consistency_actual_module_pose_world_by_id": {
            str(module_id): list(pose)
            for module_id, pose in sorted(
                kinematic_consistency_actual_module_pose_world_by_id.items()
            )
        },
        "order8_natural_contact_kinematic_consistency_predicted_module_pose_world_by_id": {
            str(module_id): list(pose)
            for module_id, pose in sorted(
                kinematic_consistency_predicted_module_pose_world_by_id.items()
            )
        },
        "order8_natural_contact_kinematic_consistency_module_position_error_m_by_id": {
            str(module_id): value
            for module_id, value in sorted(
                kinematic_consistency_module_position_error_m_by_id.items()
            )
        },
        "order8_natural_contact_kinematic_consistency_module_attitude_error_rad_by_id": {
            str(module_id): value
            for module_id, value in sorted(
                kinematic_consistency_module_attitude_error_rad_by_id.items()
            )
        },
        "order8_natural_contact_kinematic_consistency_max_module_position_error_m": (
            kinematic_consistency_max_module_position_error_m
        ),
        "order8_natural_contact_kinematic_consistency_max_module_attitude_error_rad": (
            kinematic_consistency_max_module_attitude_error_rad
        ),
        "order8_natural_contact_kinematic_consistency_anchor_position_error_m_by_anchor": {
            str(anchor_id): value
            for anchor_id, value in sorted(
                kinematic_consistency_anchor_position_error_m_by_anchor.items()
            )
        },
        "order8_natural_contact_kinematic_consistency_anchor_attitude_error_rad_by_anchor": {
            str(anchor_id): value
            for anchor_id, value in sorted(
                kinematic_consistency_anchor_attitude_error_rad_by_anchor.items()
            )
        },
        "order8_natural_contact_kinematic_consistency_max_anchor_position_error_m": (
            kinematic_consistency_max_anchor_position_error_m
        ),
        "order8_natural_contact_kinematic_consistency_max_anchor_attitude_error_rad": (
            kinematic_consistency_max_anchor_attitude_error_rad
        ),
        "order8_natural_contact_kinematic_consistency_predicted_surface_point_world_by_anchor": {
            str(anchor_id): list(point)
            for anchor_id, point in sorted(
                kinematic_consistency_predicted_surface_point_world_by_anchor.items()
            )
        },
        "order8_natural_contact_kinematic_consistency_surface_point_error_m_by_anchor": {
            str(anchor_id): value
            for anchor_id, value in sorted(
                kinematic_consistency_surface_point_error_m_by_anchor.items()
            )
        },
        "order8_natural_contact_kinematic_consistency_max_surface_point_error_m": (
            kinematic_consistency_max_surface_point_error_m
        ),
        "order8_natural_contact_gripper_clearance_mesh_aabb_count": len(
            selected_gripper_local_aabbs
        ),
        "order8_natural_contact_contact_report_body_counts": {
            str(module_id): contact_report_body_counts[module_id]
            for module_id in module_ids
        },
        "order8_natural_contact_object_contact_report_body_count": (
            object_contact_report_body_count
        ),
        "order8_natural_contact_robot_object_contact_view_sensor_count": int(
            contact_view.sensor_count
        ),
        "order8_natural_contact_robot_object_contact_view_filter_count": int(
            contact_view.filter_count
        ),
        "order8_natural_contact_robot_object_contact_view_capacity": int(
            contact_view.max_contact_data_count
        ),
        "order8_natural_contact_debug_command_mask_enabled": False,
        "order8_natural_contact_dock_joint_structural_lock_count": 0,
        "order8_natural_contact_whole_structure_kinematics_used": True,
        "order8_natural_contact_anchor_jacobian_column_count": len(
            last_kinematics.ordered_global_dock_joint_ids
        ),
        "order8_natural_contact_anchor_jacobian_ids": sorted(
            last_kinematics.anchor_jacobians
        ),
        "order8_natural_contact_dock_joint_physical_dof_count": len(expected_joint_ids),
        "order8_natural_contact_dock_joint_expected_ids": list(expected_joint_ids),
        "order8_natural_contact_dock_joint_observed_ids": sorted(observed_joint_ids),
        "order8_natural_contact_dock_joint_position_commanded_ids": sorted(
            position_commanded_ids
        ),
        "order8_natural_contact_dock_joint_velocity_commanded_ids": sorted(
            velocity_commanded_ids
        ),
        "order8_natural_contact_dock_joint_torque_bias_commanded_ids": sorted(
            torque_commanded_ids
        ),
        "order8_natural_contact_dock_torque_bias_limit_nm": float(
            dock_continuous_torque_nm
        ),
        "order8_natural_contact_dock_torque_bias_limit_basis": (
            "ak40_10_continuous_torque_limit_v1"
        ),
        "order8_natural_contact_dock_continuous_torque_nm": float(
            dock_continuous_torque_nm
        ),
        "order8_natural_contact_dock_peak_torque_nm": float(dock_peak_torque_nm),
        "order8_natural_contact_dock_peak_current_a": float(dock_peak_current_a),
        "order8_natural_contact_dock_actuator_telemetry_method": (
            "requested_unclipped_limited_isaac_target_computed_applied_speed_"
            "and_linear_current_estimate_v2"
        ),
        "order8_natural_contact_dock_current_estimate_method": (
            "absolute_applied_torque_linear_peak_ratio_v1"
        ),
        "order8_natural_contact_dock_actuator_envelope_violation_step_count": (
            dock_actuator_envelope_violation_step_count
        ),
        "order8_natural_contact_dock_actuator_envelope_audit_passed": (
            dock_actuator_envelope_violation_step_count == 0
        ),
        "order8_natural_contact_last_dock_actuator_telemetry": (
            latest_dock_actuator_telemetry
        ),
        "order8_natural_contact_dock_actuator_telemetry_maxima": (
            dock_actuator_telemetry_maxima
        ),
        "order8_natural_contact_qp_infeasible_count": qp_infeasible_count,
        "order8_natural_contact_controller_failure_count": controller_failure_count,
        "order8_natural_contact_last_controller_statuses": {
            str(module_id): last_status[module_id].to_dict() for module_id in module_ids
        },
        "order8_natural_contact_missing_actuator_target_count": missing_count,
        "order8_natural_contact_unsupported_actuator_target_count": unsupported_count,
        "order8_natural_contact_clipped_actuator_target_count": clipped_count,
        "order8_natural_contact_unresolved_actuator_target_count": unresolved_count,
        "order8_natural_contact_ordered_phase_trace": phase_trace,
        "order8_natural_contact_planner_transitions": planner_transitions,
        "order8_natural_contact_monitor_result": result.to_dict(),
        "order8_natural_contact_monitor_result_hash": stable_hash(result),
        "order8_natural_contact_step_evidence": step_evidence,
        "order8_natural_contact_raw_contact_truth_role": ORDER8_RAW_CONTACT_TRUTH_ROLE,
        "order8_natural_contact_raw_contact_truth_actor_input": False,
        "order8_natural_contact_raw_contact_truth_qpid_command": False,
        "order8_natural_contact_raw_contact_invalid_count": raw_invalid_count,
        "order8_natural_contact_raw_contact_saturation_count": raw_saturation_count,
        "order8_natural_contact_robot_environment_contact_method": (
            "all_robot_rigid_bodies_against_floor_and_object_support_v1"
        ),
        "order8_natural_contact_robot_environment_contact_step_count": (
            robot_environment_contact_step_count
        ),
        "order8_natural_contact_robot_environment_unsafe_contact_step_count": (
            robot_environment_unsafe_contact_step_count
        ),
        "order8_natural_contact_robot_environment_first_unsafe_contact_time_s": (
            robot_environment_first_unsafe_contact_time_s
        ),
        "order8_natural_contact_last_raw_contact_valid": (
            contact_measurement.raw_contact_valid
        ),
        "order8_natural_contact_last_raw_contact_saturated": (
            contact_measurement.raw_contact_saturated
        ),
        "order8_natural_contact_last_raw_contact_count": (
            contact_measurement.raw_contact_count
        ),
        "order8_natural_contact_last_raw_contact_capacity": (
            contact_measurement.raw_contact_capacity
        ),
        "order8_natural_contact_raw_contact_failure_reasons": (
            raw_contact_failure_reasons
        ),
        "order8_natural_contact_unintended_contact_count": result.unintended_contact_count,
        "order8_natural_contact_object_drop_count": int(result.object_dropped),
        "order8_natural_contact_post_release_selected_contact_count": post_release_selected_contact_count,
        "order8_natural_contact_payload_feedforward_active_count": payload_feedforward_active_count,
        "order8_natural_contact_payload_feedforward_method": (
            "verified_grasp_shared_commanded_lift_progress_and_centroidal_"
            "load_observer_known_payload_qpid_coupling_v7"
        ),
        "order8_natural_contact_payload_feedforward_transition_duration_s": float(
            config.payload_load_transfer_s
        ),
        "order8_natural_contact_payload_feedforward_peak_scale": (
            payload_feedforward_peak_scale
        ),
        "order8_natural_contact_payload_feedforward_max_scale_step": (
            payload_feedforward_max_scale_step
        ),
        "order8_natural_contact_last_payload_feedforward_scale": (
            last_payload_feedforward_scale
        ),
        "order8_natural_contact_last_payload_feedforward_target_scale": (
            last_payload_feedforward_target_scale
        ),
        "order8_natural_contact_payload_commanded_lift_progress_method": (
            "shared_lift_phase_elapsed_over_payload_transfer_duration_v1"
        ),
        "order8_natural_contact_last_payload_commanded_lift_progress_scale": (
            last_payload_commanded_lift_progress_scale
        ),
        "order8_natural_contact_payload_commanded_lift_progress_peak_scale": (
            payload_commanded_lift_progress_peak_scale
        ),
        "order8_natural_contact_payload_load_transfer_distance_m": (
            payload_load_transfer_distance_m
        ),
        "order8_natural_contact_measured_payload_lift_transfer_peak_scale": (
            measured_payload_lift_transfer_peak_scale
        ),
        "order8_natural_contact_estimated_payload_lift_transfer_peak_scale": (
            estimated_payload_lift_transfer_peak_scale
        ),
        "order8_natural_contact_last_estimated_payload_lift_transfer_scale": (
            last_estimated_payload_lift_transfer_scale
        ),
        "order8_natural_contact_payload_lift_start_external_force_world_z_n": (
            lift_start_external_force_world_z_n
        ),
        "order8_natural_contact_payload_last_external_force_world_z_n": (
            last_lift_external_force_world_z_n
        ),
        "order8_natural_contact_payload_last_estimated_transferred_load_n": (
            last_estimated_payload_transferred_load_n
        ),
        "order8_natural_contact_payload_load_observer_valid_step_count": (
            payload_load_observer_valid_step_count
        ),
        "order8_natural_contact_payload_load_observer_invalid_step_count": (
            payload_load_observer_invalid_step_count
        ),
        "order8_natural_contact_payload_load_observer_method": (
            "aggregate_centroidal_external_vertical_force_delta_from_lift_start_"
            "normalized_by_known_payload_weight_v1"
        ),
        "order8_natural_contact_payload_load_observer_raw_contact_input": False,
        "order8_natural_contact_payload_lift_off_clearance_m": (
            ORDER8_OBJECT_LIFT_OFF_CLEARANCE_M
        ),
        "order8_natural_contact_payload_lift_off_confirmed_time_s": (
            payload_lift_off_confirmed_time_s
        ),
        "order8_natural_contact_payload_feedforward_max_lead_over_observed_scale": (
            max_payload_feedforward_lead_over_observed_scale
        ),
        "order8_natural_contact_payload_feedforward_max_lag_behind_commanded_"
        "progress_scale": (
            max_payload_feedforward_lag_behind_commanded_progress_scale
        ),
        "order8_natural_contact_payload_load_transfer_observation": (
            "aggregate_centroidal_external_vertical_load_with_monotonic_"
            "object_com_rise_and_geometric_liftoff_audit_v2"
        ),
        "order8_natural_contact_payload_load_transfer_driver": (
            "slew_limited_max_commanded_lift_progress_observed_load_after_"
            "verified_grasp_v3"
        ),
        "order8_natural_contact_last_payload_coupling": last_payload_coupling,
        "order8_natural_contact_last_full_payload_coupling": (
            last_full_payload_coupling
        ),
        "order8_natural_contact_payload_feedforward_coupling_mode": (
            "natural_contact_verified_grasp_ramped_payload_v2"
        ),
        "order8_natural_contact_payload_feedforward_inertia_method": (
            "oriented_cuboid_com_inertia_rotated_into_measured_centroidal_frame_v1"
        ),
        "order8_natural_contact_payload_feedforward_object_constraint": False,
        "order8_natural_contact_lift_acceleration_bias_method": (
            "known_payload_mass_times_shared_lift_progress_world_vertical_"
            "policy_command_residual_wrench_v1"
        ),
        "order8_natural_contact_lift_acceleration_bias_qpid_application": (
            "policy_command_residual_wrench_body_centroidal_only_v1"
        ),
        "order8_natural_contact_lift_acceleration_bias_raw_contact_input": False,
        "order8_natural_contact_lift_acceleration_bias_payload_mass_kg": float(
            config.object_mass_kg
        ),
        "order8_natural_contact_lift_payload_acceleration_mps2": float(
            config.lift_payload_acceleration_mps2
        ),
        "order8_natural_contact_lift_acceleration_bias_removal_s": float(
            config.lift_acceleration_bias_removal_s
        ),
        "order8_natural_contact_lift_acceleration_bias_removal_method": (
            "cubic_smoothstep_zero_endpoint_slope_v1"
        ),
        "order8_natural_contact_lift_acceleration_bias_active_count": (
            lift_acceleration_bias_active_count
        ),
        "order8_natural_contact_lift_acceleration_bias_non_lift_active_count": (
            lift_acceleration_bias_non_lift_active_count
        ),
        "order8_natural_contact_lift_acceleration_bias_policy_command_active_count": (
            lift_acceleration_bias_policy_command_active_count
        ),
        "order8_natural_contact_lift_acceleration_bias_peak_scale": (
            lift_acceleration_bias_peak_scale
        ),
        "order8_natural_contact_last_lift_acceleration_bias_scale": (
            last_lift_acceleration_bias_scale
        ),
        "order8_natural_contact_last_lift_acceleration_bias_commanded_progress_scale": (
            last_lift_acceleration_bias_commanded_progress_scale
        ),
        "order8_natural_contact_lift_acceleration_bias_lift_off_scale": (
            lift_acceleration_bias_lift_off_scale
        ),
        "order8_natural_contact_lift_acceleration_bias_removal_complete_time_s": (
            lift_acceleration_bias_removal_complete_time_s
        ),
        "order8_natural_contact_lift_acceleration_bias_peak_force_world_z_n": (
            lift_acceleration_bias_peak_force_world_z_n
        ),
        "order8_natural_contact_lift_acceleration_bias_peak_residual_force_"
        "body_norm_n": (
            lift_acceleration_bias_peak_residual_force_body_norm_n
        ),
        "order8_natural_contact_last_lift_acceleration_bias_force_world_z_n": (
            last_lift_acceleration_bias_force_world_z_n
        ),
        "order8_natural_contact_last_lift_acceleration_residual_wrench_body": (
            last_lift_acceleration_residual_wrench_body
        ),
        "order8_natural_contact_contact_motion_entry_speed_ramp_method": (
            "immediate_linear_lift_and_maintained_contact_phase_entry_ramp_v6"
        ),
        "order8_natural_contact_contact_motion_entry_speed_ramp_duration_s": float(
            config.payload_load_transfer_s
        ),
        "order8_natural_contact_constraint_identity_failures": constraint_failures,
        "order8_natural_contact_scope": "deterministic_natural_contact_substrate_only",
        "order8_natural_contact_p4_full_completion_claim": False,
        "order8_natural_contact_order9_full_taskspec_claim": False,
        "order8_natural_contact_learned_policy_success_claim": False,
        "order8_natural_contact_failure_reason": failure_reason,
        "generated_urdf_sha256": hash_file(urdf_path),
        "generated_urdf_path": str(urdf_path),
        "generated_usd_sha256": hash_file(usd_path),
        "generated_usd_bundle_hash": hash_directory_manifest(usd_path.parent),
        "usd_path": str(usd_path),
        "realtime_playback": bool(args.realtime_playback),
        "keep_open_after_smoke_s": float(args.keep_open_after_smoke_s),
    }
    return report


def _capture_order8_state_trace_frame(
    *,
    simulation_time_s: float,
    phase: str,
    robots: Mapping[int, Any],
    object_asset: Any,
) -> dict[str, object]:
    """Capture only simulator state needed for diagnostic visual replay."""

    object_pose = _tensor_row(object_asset.data.root_pose_w)
    object_velocity_tensor = getattr(object_asset.data, "root_vel_w", None)
    object_twist = (
        _tensor_row(object_velocity_tensor)
        if object_velocity_tensor is not None
        else _tensor_row(object_asset.data.root_lin_vel_w)
        + _tensor_row(object_asset.data.root_ang_vel_w)
    )
    return {
        "simulation_time_s": float(simulation_time_s),
        "phase": str(phase),
        "modules": {
            str(module_id): {
                "root_pose_world": _tensor_row(robot.data.root_pose_w),
                "root_twist_world": _tensor_row(robot.data.root_vel_w),
                "joint_positions_rad": _tensor_row(robot.data.joint_pos),
                "joint_velocities_radps": _tensor_row(robot.data.joint_vel),
            }
            for module_id, robot in sorted(robots.items())
        },
        "object_pose_world": object_pose,
        "object_twist_world": object_twist,
    }


def _validate_order8_state_trace_runtime_binding(
    state_trace: Mapping[str, object],
    *,
    graph_id: str,
    graph_hash: str,
    config_hash: str,
    source_urdf_sha256: str,
    generated_usd_sha256: str,
    module_ids: Sequence[int],
    robots: Mapping[int, Any],
) -> None:
    """Reject replay against different geometry, configuration, or indexing."""

    expected_scalars = {
        "graph_id": str(graph_id),
        "graph_hash": str(graph_hash),
        "config_hash": str(config_hash),
        "source_urdf_sha256": str(source_urdf_sha256),
        "generated_usd_sha256": str(generated_usd_sha256),
    }
    for field_name, expected in expected_scalars.items():
        if state_trace.get(field_name) != expected:
            raise RuntimeError(
                f"Order8 state-trace {field_name} does not match the replay scene"
            )
    expected_module_ids = [int(value) for value in sorted(module_ids)]
    if state_trace.get("module_ids") != expected_module_ids:
        raise RuntimeError("Order8 state-trace module ids do not match the scene")
    expected_joint_names = {
        str(module_id): [str(name) for name in robots[module_id].joint_names]
        for module_id in expected_module_ids
    }
    if state_trace.get("joint_names_by_module") != expected_joint_names:
        raise RuntimeError(
            "Order8 state-trace joint ordering does not match the scene"
        )


def _replay_order8_state_trace(
    state_trace: Mapping[str, object],
    *,
    robots: Mapping[int, Any],
    object_asset: Any,
    sim: Any,
    torch: Any,
    wp: Any,
    speed: float,
    loops: int,
    endpoint_hold_s: float,
    sync_physics: bool,
) -> dict[str, object]:
    """Replay states against wall time, dropping late GUI frames.

    The compatibility synchronization mode advances one gravity-free,
    contact-minimized step per rendered frame. It exists because a direct
    PhysX tensor write plus ``forward()`` is not reflected by every Kit/Fabric
    GUI configuration. The exact recorded state is re-applied after that step.
    """

    frames_raw = state_trace["frames"]
    assert isinstance(frames_raw, list)
    frames = [frame for frame in frames_raw if isinstance(frame, dict)]
    times = [float(frame["simulation_time_s"]) for frame in frames]
    source_duration_s = times[-1] - times[0]
    playback_duration_s = source_duration_s / float(speed)
    rendered_frame_count = 0
    dropped_frame_count = 0
    maximum_joint_write_error_rad = 0.0
    maximum_physics_joint_write_error_rad = 0.0
    maximum_physics_dock_joint_delta_rad = 0.0
    maximum_root_position_write_error_m = 0.0
    maximum_object_position_write_error_m = 0.0
    physics_joint_reference_by_module: dict[str, list[float]] | None = None
    wall_started = time.monotonic()
    last_status_print_s = -math.inf
    first_object_pose = frames[0]["object_pose_world"]
    assert isinstance(first_object_pose, list)
    sim.set_camera_view(
        eye=[
            float(first_object_pose[0]) + 0.85,
            float(first_object_pose[1]) + 1.15,
            float(first_object_pose[2]) + 1.00,
        ],
        target=[
            float(first_object_pose[0]),
            float(first_object_pose[1]),
            float(first_object_pose[2]) + 0.10,
        ],
    )
    maximum_recorded_dock_joint_delta_rad = _order8_trace_maximum_dock_delta_rad(
        state_trace,
        frames,
    )

    for loop_index in range(int(loops)):
        first_application = _apply_order8_state_trace_frame(
            frames[0],
            robots=robots,
            object_asset=object_asset,
            sim=sim,
            torch=torch,
            wp=wp,
            update_dt_s=float(state_trace["simulation_dt_s"]),
            sync_physics=sync_physics,
        )
        rendered_frame_count += 1
        maximum_joint_write_error_rad = max(
            maximum_joint_write_error_rad,
            float(first_application["maximum_joint_position_error_rad"]),
        )
        maximum_physics_joint_write_error_rad = max(
            maximum_physics_joint_write_error_rad,
            float(first_application["maximum_physics_joint_position_error_rad"]),
        )
        first_physics_positions = first_application[
            "physics_joint_positions_rad_by_module"
        ]
        assert isinstance(first_physics_positions, dict)
        if physics_joint_reference_by_module is None:
            physics_joint_reference_by_module = {
                str(module_key): [float(value) for value in values]
                for module_key, values in first_physics_positions.items()
            }
        maximum_root_position_write_error_m = max(
            maximum_root_position_write_error_m,
            float(first_application["maximum_root_position_error_m"]),
        )
        maximum_object_position_write_error_m = max(
            maximum_object_position_write_error_m,
            float(first_application["object_position_error_m"]),
        )
        _pump_order8_state_trace_hold(sim, float(endpoint_hold_s))
        loop_started = time.monotonic()
        rendered_index = 0
        while rendered_index < len(frames) - 1:
            wall_elapsed_s = time.monotonic() - loop_started
            source_elapsed_s = min(
                source_duration_s,
                wall_elapsed_s * float(speed),
            )
            target_source_time_s = times[0] + source_elapsed_s
            frame_index = max(
                0,
                min(
                    len(frames) - 1,
                    bisect_right(times, target_source_time_s) - 1,
                ),
            )
            if wall_elapsed_s >= playback_duration_s:
                frame_index = len(frames) - 1
            if frame_index > rendered_index:
                dropped_frame_count += max(0, frame_index - rendered_index - 1)
                frame = frames[frame_index]
                application = _apply_order8_state_trace_frame(
                    frame,
                    robots=robots,
                    object_asset=object_asset,
                    sim=sim,
                    torch=torch,
                    wp=wp,
                    update_dt_s=float(state_trace["simulation_dt_s"]),
                    sync_physics=sync_physics,
                )
                maximum_joint_write_error_rad = max(
                    maximum_joint_write_error_rad,
                    float(application["maximum_joint_position_error_rad"]),
                )
                maximum_physics_joint_write_error_rad = max(
                    maximum_physics_joint_write_error_rad,
                    float(application["maximum_physics_joint_position_error_rad"]),
                )
                physics_positions = application[
                    "physics_joint_positions_rad_by_module"
                ]
                assert isinstance(physics_positions, dict)
                assert physics_joint_reference_by_module is not None
                maximum_physics_dock_joint_delta_rad = max(
                    maximum_physics_dock_joint_delta_rad,
                    _order8_physics_dock_delta_rad(
                        state_trace,
                        reference_by_module=physics_joint_reference_by_module,
                        current_by_module=physics_positions,
                    ),
                )
                maximum_root_position_write_error_m = max(
                    maximum_root_position_write_error_m,
                    float(application["maximum_root_position_error_m"]),
                )
                maximum_object_position_write_error_m = max(
                    maximum_object_position_write_error_m,
                    float(application["object_position_error_m"]),
                )
                rendered_frame_count += 1
                rendered_index = frame_index
                now_s = time.monotonic()
                if (
                    now_s - last_status_print_s >= 0.10
                    or rendered_index == len(frames) - 1
                ):
                    print(
                        "\r[order8-state-replay] "
                        f"loop={loop_index + 1}/{loops} "
                        f"simulation_time={float(frame['simulation_time_s']):.2f}s "
                        f"phase={frame['phase']} "
                        "recorded_dock_delta_max="
                        f"{math.degrees(maximum_recorded_dock_joint_delta_rad):.2f}deg "
                        "physx_dock_delta_max="
                        f"{math.degrees(maximum_physics_dock_joint_delta_rad):.2f}deg "
                        "physx_error="
                        f"{maximum_physics_joint_write_error_rad:.6f}rad",
                        end="",
                        flush=True,
                    )
                    last_status_print_s = now_s
            if rendered_index >= len(frames) - 1:
                break
            next_source_delta_s = max(
                0.0,
                times[rendered_index + 1] - target_source_time_s,
            )
            time.sleep(min(0.005, next_source_delta_s / float(speed)))
        _pump_order8_state_trace_hold(sim, float(endpoint_hold_s))
    print(flush=True)
    return {
        "order8_state_trace_source_duration_s": source_duration_s,
        "order8_state_trace_wall_elapsed_s": time.monotonic() - wall_started,
        "order8_state_trace_source_frame_count": len(frames),
        "order8_state_trace_rendered_frame_count": rendered_frame_count,
        "order8_state_trace_dropped_frame_count": dropped_frame_count,
        "order8_state_trace_endpoint_hold_s": float(endpoint_hold_s),
        "order8_state_trace_maximum_recorded_dock_joint_delta_rad": (
            maximum_recorded_dock_joint_delta_rad
        ),
        "order8_state_trace_maximum_joint_write_error_rad": (
            maximum_joint_write_error_rad
        ),
        "order8_state_trace_maximum_physics_joint_write_error_rad": (
            maximum_physics_joint_write_error_rad
        ),
        "order8_state_trace_maximum_physics_dock_joint_delta_rad": (
            maximum_physics_dock_joint_delta_rad
        ),
        "order8_state_trace_maximum_root_position_write_error_m": (
            maximum_root_position_write_error_m
        ),
        "order8_state_trace_maximum_object_position_write_error_m": (
            maximum_object_position_write_error_m
        ),
    }


def _apply_order8_state_trace_frame(
    frame: Mapping[str, object],
    *,
    robots: Mapping[int, Any],
    object_asset: Any,
    sim: Any,
    torch: Any,
    wp: Any,
    update_dt_s: float,
    sync_physics: bool,
) -> dict[str, object]:
    modules = frame["modules"]
    assert isinstance(modules, dict)

    def write_recorded_state() -> None:
        for module_id, robot in sorted(robots.items()):
            module_state = modules[str(module_id)]
            assert isinstance(module_state, dict)
            joint_positions = torch.tensor(
                [module_state["joint_positions_rad"]],
                dtype=torch.float32,
                device=sim.device,
            )
            joint_velocities = torch.tensor(
                [
                    [0.0] * len(module_state["joint_velocities_radps"])
                    if sync_physics
                    else module_state["joint_velocities_radps"]
                ],
                dtype=torch.float32,
                device=sim.device,
            )
            robot.write_root_pose_to_sim_index(
                root_pose=torch.tensor(
                    [module_state["root_pose_world"]],
                    dtype=torch.float32,
                    device=sim.device,
                )
            )
            robot.write_root_velocity_to_sim_index(
                root_velocity=torch.tensor(
                    [
                        [0.0] * 6
                        if sync_physics
                        else module_state["root_twist_world"]
                    ],
                    dtype=torch.float32,
                    device=sim.device,
                )
            )
            robot.write_joint_position_to_sim_index(position=joint_positions)
            robot.write_joint_velocity_to_sim_index(velocity=joint_velocities)
            if sync_physics:
                robot.set_joint_position_target_index(target=joint_positions)
                robot.set_joint_velocity_target_index(
                    target=torch.zeros_like(joint_positions)
                )
                robot.set_joint_effort_target_index(
                    target=torch.zeros_like(joint_positions)
                )
        object_asset.write_root_pose_to_sim(
            torch.tensor(
                [frame["object_pose_world"]],
                dtype=torch.float32,
                device=sim.device,
            )
        )
        object_asset.write_root_velocity_to_sim(
            torch.tensor(
                [[0.0] * 6 if sync_physics else frame["object_twist_world"]],
                dtype=torch.float32,
                device=sim.device,
            )
        )

    write_recorded_state()
    if sync_physics:
        for robot in robots.values():
            robot.permanent_wrench_composer.reset()
            robot.write_data_to_sim()
        # This is a visual synchronization step, not a replay of recorded
        # dynamics. Object/environment collisions, gravity, self-collision, and
        # graph constraints were disabled before reset; targets equal the
        # recorded joint state. Authored cross-module geometry remains enabled.
        sim.step(render=False)
        write_recorded_state()
    sim.forward()
    for robot in robots.values():
        robot.update(float(update_dt_s))
    object_asset.update(float(update_dt_s))
    sim.render()
    maximum_joint_error_rad = 0.0
    maximum_physics_joint_error_rad = 0.0
    maximum_root_position_error_m = 0.0
    physics_joint_positions_by_module: dict[str, list[float]] = {}
    for module_id, robot in sorted(robots.items()):
        module_state = modules[str(module_id)]
        assert isinstance(module_state, dict)
        actual_joint_positions = _tensor_row(robot.data.joint_pos)
        expected_joint_positions = [
            float(value) for value in module_state["joint_positions_rad"]
        ]
        physics_joint_positions = _tensor_row(
            wp.to_torch(robot.root_view.get_dof_positions())
        )
        physics_joint_positions_by_module[str(module_id)] = physics_joint_positions
        maximum_joint_error_rad = max(
            maximum_joint_error_rad,
            max(
                (
                    abs(actual - expected)
                    for actual, expected in zip(
                        actual_joint_positions,
                        expected_joint_positions,
                        strict=True,
                    )
                ),
                default=0.0,
            ),
        )
        maximum_physics_joint_error_rad = max(
            maximum_physics_joint_error_rad,
            max(
                (
                    abs(actual - expected)
                    for actual, expected in zip(
                        physics_joint_positions,
                        expected_joint_positions,
                        strict=True,
                    )
                ),
                default=0.0,
            ),
        )
        actual_root_pose = _tensor_row(robot.data.root_pose_w)
        expected_root_pose = [float(value) for value in module_state["root_pose_world"]]
        maximum_root_position_error_m = max(
            maximum_root_position_error_m,
            math.sqrt(
                sum(
                    (actual_root_pose[index] - expected_root_pose[index]) ** 2
                    for index in range(3)
                )
            ),
        )
    actual_object_pose = _tensor_row(object_asset.data.root_pose_w)
    expected_object_pose = [float(value) for value in frame["object_pose_world"]]
    object_position_error_m = math.sqrt(
        sum(
            (actual_object_pose[index] - expected_object_pose[index]) ** 2
            for index in range(3)
        )
    )
    return {
        "maximum_joint_position_error_rad": maximum_joint_error_rad,
        "maximum_physics_joint_position_error_rad": (
            maximum_physics_joint_error_rad
        ),
        "physics_joint_positions_rad_by_module": (
            physics_joint_positions_by_module
        ),
        "maximum_root_position_error_m": maximum_root_position_error_m,
        "object_position_error_m": object_position_error_m,
    }


def _order8_physics_dock_delta_rad(
    state_trace: Mapping[str, object],
    *,
    reference_by_module: Mapping[str, object],
    current_by_module: Mapping[str, object],
) -> float:
    """Return the maximum independently read PhysX Dock displacement."""

    joint_names_by_module = state_trace["joint_names_by_module"]
    assert isinstance(joint_names_by_module, dict)
    maximum = 0.0
    for module_key, joint_names in joint_names_by_module.items():
        assert isinstance(joint_names, list)
        reference = reference_by_module[str(module_key)]
        current = current_by_module[str(module_key)]
        assert isinstance(reference, list)
        assert isinstance(current, list)
        for index, joint_name in enumerate(joint_names):
            if "dock" not in str(joint_name).lower():
                continue
            maximum = max(
                maximum,
                abs(float(current[index]) - float(reference[index])),
            )
    return maximum


def _order8_trace_maximum_dock_delta_rad(
    state_trace: Mapping[str, object],
    frames: Sequence[Mapping[str, object]],
) -> float:
    joint_names_by_module = state_trace["joint_names_by_module"]
    assert isinstance(joint_names_by_module, dict)
    first_modules = frames[0]["modules"]
    assert isinstance(first_modules, dict)
    maximum = 0.0
    for module_key, joint_names in joint_names_by_module.items():
        assert isinstance(joint_names, list)
        first_state = first_modules[str(module_key)]
        assert isinstance(first_state, dict)
        first_positions = first_state["joint_positions_rad"]
        assert isinstance(first_positions, list)
        for frame in frames:
            modules = frame["modules"]
            assert isinstance(modules, dict)
            state = modules[str(module_key)]
            assert isinstance(state, dict)
            positions = state["joint_positions_rad"]
            assert isinstance(positions, list)
            for index, joint_name in enumerate(joint_names):
                if "dock" not in str(joint_name).lower():
                    continue
                maximum = max(
                    maximum,
                    abs(float(positions[index]) - float(first_positions[index])),
                )
    return maximum


def _pump_order8_state_trace_hold(sim: Any, duration_s: float) -> None:
    deadline_s = time.monotonic() + max(0.0, float(duration_s))
    while time.monotonic() < deadline_s:
        sim.render()
        time.sleep(min(1.0 / 60.0, max(0.0, deadline_s - time.monotonic())))


def _singleton_graph(graph: Any, module_id: int) -> Any:
    from amsrr.schemas.morphology import ControlGroup, MorphologyGraph

    modules = [
        replace(
            module,
            is_base=True,
            pose_in_design_frame=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        )
        for module in graph.modules
        if module.module_id == module_id
    ]
    if len(modules) != 1:
        raise RuntimeError(f"Order8 cannot resolve module {module_id}")
    return MorphologyGraph(
        graph_id=f"{graph.graph_id}:order8-component:{module_id}",
        modules=modules,
        ports=[
            replace(port, occupied=False)
            for port in graph.ports
            if port.module_id == module_id
        ],
        dock_edges=[],
        robot_anchors=[
            anchor for anchor in graph.robot_anchors if anchor.module_id == module_id
        ],
        control_groups=[
            ControlGroup(f"component:{module_id}", [module_id], "order8_component")
        ],
        base_module_id=module_id,
        is_closed_loop=False,
    )


def _resolve_rigid_body_path(stage: Any, root: str, local_name: str) -> str:
    from pxr import UsdPhysics

    matches = [
        prim.GetPath().pathString
        for prim in stage.Traverse()
        if prim.GetPath().pathString.startswith(root.rstrip("/") + "/")
        and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        and (
            str(prim.GetName()) == local_name
            or str(prim.GetName()).endswith("__" + local_name)
        )
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Order8 body {local_name!r} below {root} resolved to {matches}"
        )
    return matches[0]


def _preauthor_disabled_world_fixed_body(
    stage: Any,
    *,
    prim_path: str,
    body_path: str,
    body_pose_world: Pose7D,
) -> Any:
    """Preauthor a disabled world constraint for an ineligible diagnostic."""

    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    if not prim_path.startswith("/") or not body_path.startswith("/"):
        raise SchemaValidationError("world-fixed diagnostic paths must be absolute")
    if len(body_pose_world) != 7 or not all(
        math.isfinite(float(value)) for value in body_pose_world
    ):
        raise SchemaValidationError(
            "world-fixed diagnostic pose must be a finite Pose7D"
        )
    body_prim = stage.GetPrimAtPath(Sdf.Path(body_path))
    if not body_prim.IsValid() or not body_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(
            f"world-fixed diagnostic body is not a rigid body: {body_path}"
        )

    UsdGeom.Scope.Define(stage, Sdf.Path(prim_path.rsplit("/", 1)[0]))
    joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(prim_path))
    joint.CreateJointEnabledAttr(False).Set(False)
    joint.CreateExcludeFromArticulationAttr(True).Set(True)
    joint.CreateCollisionEnabledAttr(True).Set(True)
    # An absent body0 relationship denotes the static world frame.  Match its
    # local joint frame to the requested initial world pose and body1's frame
    # to the rigid-body origin.
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body_path)])
    joint.CreateLocalPos0Attr().Set(
        Gf.Vec3f(*(float(value) for value in body_pose_world[:3]))
    )
    x, y, z, w = (float(value) for value in body_pose_world[3:7])
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(w, x, y, z))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    return joint


def _enable_world_fixed_body_at_pose(joint: Any, body_pose_world: Pose7D) -> None:
    """Match the world frame to an observed body pose, then enable the joint."""

    from pxr import Gf

    if len(body_pose_world) != 7 or not all(
        math.isfinite(float(value)) for value in body_pose_world
    ):
        raise SchemaValidationError(
            "world-fixed diagnostic pose must be a finite Pose7D"
        )
    joint.GetLocalPos0Attr().Set(
        Gf.Vec3f(*(float(value) for value in body_pose_world[:3]))
    )
    x, y, z, w = (float(value) for value in body_pose_world[3:7])
    joint.GetLocalRot0Attr().Set(Gf.Quatf(w, x, y, z))
    joint.GetJointEnabledAttr().Set(True)
    from omni.physx import get_physx_simulation_interface

    get_physx_simulation_interface().flush_changes()


def _rigid_body_paths(stage: Any, root: str) -> list[str]:
    from pxr import UsdPhysics

    return sorted(
        prim.GetPath().pathString
        for prim in stage.Traverse()
        if prim.GetPath().pathString.startswith(root.rstrip("/") + "/")
        and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    )


def _activate_nested_contact_reports(stage: Any, *, root_prim_path: str) -> int:
    """Enable zero-threshold PhysX reporting on every nested rigid body."""

    from pxr import PhysxSchema, UsdPhysics

    root_prefix = root_prim_path.rstrip("/") + "/"
    applied_count = 0
    for prim in stage.Traverse():
        prim_path = prim.GetPath().pathString
        if prim_path != root_prim_path and not prim_path.startswith(root_prefix):
            continue
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        report_api.CreateThresholdAttr().Set(0.0)
        applied_count += 1
    if applied_count == 0:
        raise RuntimeError(
            "Order8 found no rigid body for contact reporting below "
            f"{root_prim_path}"
        )
    return applied_count


def _require_contact_view_layout(
    contact_view: Any,
    *,
    label: str,
    expected_sensor_count: int,
    expected_filter_count: int,
) -> None:
    """Fail before rollout if PhysX drops any requested sensor/filter body."""

    actual_sensor_count = int(contact_view.sensor_count)
    actual_filter_count = int(contact_view.filter_count)
    if actual_sensor_count != expected_sensor_count:
        raise RuntimeError(
            f"Order8 {label} contact-view sensor count mismatch: "
            f"{actual_sensor_count} != {expected_sensor_count}"
        )
    if actual_filter_count != expected_filter_count:
        raise RuntimeError(
            f"Order8 {label} contact-view filter count mismatch: "
            f"{actual_filter_count} != {expected_filter_count}"
        )


def _object_constraint_references(
    stage: Any,
    *,
    object_root_path: str,
) -> list[tuple[str, str, str]]:
    """List USD physics-joint body references to the free object subtree."""

    from pxr import UsdPhysics

    object_prefix = object_root_path.rstrip("/") + "/"
    references: list[tuple[str, str, str]] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        for relationship_name, relationship in (
            ("body0", joint.GetBody0Rel()),
            ("body1", joint.GetBody1Rel()),
        ):
            for target in relationship.GetTargets():
                target_path = target.pathString
                if target_path == object_root_path or target_path.startswith(
                    object_prefix
                ):
                    references.append(
                        (
                            prim.GetPath().pathString,
                            relationship_name,
                            target_path,
                        )
                    )
    return sorted(references)


def _canonical_rigid_body_local_name(path: str, physical_model: Any) -> str:
    imported_name = str(path).rsplit("/", 1)[-1]
    matches = [
        str(link.link_id)
        for link in physical_model.links
        if imported_name == str(link.link_id)
        or imported_name.endswith("__" + str(link.link_id))
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Order8 imported rigid body {path!r} cannot be mapped to one PhysicalModel link: {matches}"
        )
    return matches[0]


def _dock_limit(
    joint: Any,
    dock_spec: dict[str, Any],
    *,
    velocity_limit_override_radps: float | None = None,
) -> Any:
    from amsrr.controllers.natural_contact_joint_controller import DockJointLimit

    lower = float(joint.limit_lower if joint.limit_lower is not None else -math.pi)
    upper = float(joint.limit_upper if joint.limit_upper is not None else math.pi)
    velocity = min(
        abs(float(joint.velocity_limit if joint.velocity_limit is not None else 3.0)),
        float(
            dock_spec.get("simulation_drive", {}).get("safe_velocity_limit_rad_s", 3.0)
        ),
    )
    if velocity_limit_override_radps is not None:
        override = float(velocity_limit_override_radps)
        if not math.isfinite(override) or override <= 0.0:
            raise SchemaValidationError(
                "Dock joint velocity-limit override must be finite and positive"
            )
        velocity = min(velocity, override)
    torque = min(
        abs(float(joint.effort_limit if joint.effort_limit is not None else 4.1)),
        float(dock_spec.get("continuous_torque_limit_nm", 1.3)),
    )
    return DockJointLimit(lower, upper, velocity, torque)


def _joint_state_dict(robot: Any) -> dict[str, float]:
    values = _tensor_row(robot.data.joint_pos.torch)
    return {name: float(values[index]) for index, name in enumerate(robot.joint_names)}


def _joint_velocity_dict(robot: Any) -> dict[str, float]:
    values = _tensor_row(robot.data.joint_vel.torch)
    return {name: float(values[index]) for index, name in enumerate(robot.joint_names)}


def _global_joint_tensor_value(
    robots: Mapping[int, Any],
    global_joint_id: str,
    *,
    field_name: str,
) -> float:
    """Read one finite articulation field without treating contact as input."""

    module_text, separator, local_joint_id = str(global_joint_id).partition(":")
    if not separator or not module_text.startswith("module_") or not local_joint_id:
        raise SchemaValidationError(
            f"invalid global joint id for actuator observation: {global_joint_id!r}"
        )
    try:
        module_id = int(module_text.removeprefix("module_"))
    except ValueError as exc:
        raise SchemaValidationError(
            f"invalid module id for actuator observation: {global_joint_id!r}"
        ) from exc
    robot = robots.get(module_id)
    if robot is None:
        raise SchemaValidationError(f"actuator observation has no module {module_id}")
    resolved_name = _resolve_name(robot.joint_names, local_joint_id)
    if resolved_name is None:
        raise SchemaValidationError(
            f"actuator observation cannot resolve joint {global_joint_id!r}"
        )
    field = getattr(robot.data, field_name, None)
    if field is None:
        raise SchemaValidationError(
            f"actuator observation requires articulation field {field_name!r}"
        )
    values = _tensor_row(field)
    value = float(values[robot.joint_names.index(resolved_name)])
    if not math.isfinite(value):
        raise SchemaValidationError(
            f"actuator observation {field_name!r} must be finite"
        )
    return value


def _schedule_contact_joint_drive_impedance(
    robots: Mapping[int, Any],
    expected_joint_ids: Sequence[str],
    *,
    stiffness_nm_per_rad: float,
    damping_nms_per_rad: float,
    maximum_stiffness_nm_per_rad: float,
    maximum_damping_nms_per_rad: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """Write bounded simulator-only impedance to every articulated Dock joint.

    This changes neither the policy's absolute position/velocity targets nor
    its independent torque bias.  It only changes how Isaac realizes those
    commands while contact is being acquired.  Applied torque/current/speed
    remain independently audited against the AK40-10 envelope.
    """

    values = {
        "stiffness_nm_per_rad": stiffness_nm_per_rad,
        "damping_nms_per_rad": damping_nms_per_rad,
        "maximum_stiffness_nm_per_rad": maximum_stiffness_nm_per_rad,
        "maximum_damping_nms_per_rad": maximum_damping_nms_per_rad,
    }
    if any(
        not math.isfinite(float(value)) or float(value) <= 0.0
        for value in values.values()
    ):
        raise SchemaValidationError(
            "contact joint drive impedance values must be finite and positive"
        )
    if float(stiffness_nm_per_rad) > float(maximum_stiffness_nm_per_rad):
        raise SchemaValidationError(
            "scheduled contact joint drive stiffness exceeds the configured "
            "simulation limit"
        )
    if float(damping_nms_per_rad) > float(maximum_damping_nms_per_rad):
        raise SchemaValidationError(
            "scheduled contact joint drive damping exceeds the configured "
            "simulation limit"
        )
    expected = tuple(str(joint_id) for joint_id in expected_joint_ids)
    if len(set(expected)) != len(expected):
        raise SchemaValidationError(
            "contact joint drive impedance ids must be globally unique"
        )

    joint_indices_by_module: dict[int, list[int]] = {}
    stiffness_targets: dict[str, float] = {}
    damping_targets: dict[str, float] = {}
    for joint_id in expected:
        module_prefix, separator, local_joint_id = joint_id.partition(":")
        if not separator or not module_prefix.startswith("module_"):
            raise SchemaValidationError(
                f"invalid global Dock joint id for impedance schedule: {joint_id!r}"
            )
        try:
            module_id = int(module_prefix.removeprefix("module_"))
        except ValueError as exc:
            raise SchemaValidationError(
                f"invalid module id in Dock impedance schedule: {joint_id!r}"
            ) from exc
        robot = robots.get(module_id)
        if robot is None:
            raise SchemaValidationError(
                f"Dock impedance schedule has no robot for module {module_id}"
            )
        matches = [
            index
            for index, name in enumerate(robot.joint_names)
            if str(name) == local_joint_id
        ]
        if len(matches) != 1:
            raise SchemaValidationError(
                f"Dock impedance schedule could not resolve {joint_id!r} exactly once"
            )
        joint_indices_by_module.setdefault(module_id, []).append(matches[0])
        stiffness_targets[joint_id] = float(stiffness_nm_per_rad)
        damping_targets[joint_id] = float(damping_nms_per_rad)

    for module_id, joint_indices in sorted(joint_indices_by_module.items()):
        robots[module_id].write_joint_stiffness_to_sim_index(
            stiffness=float(stiffness_nm_per_rad),
            joint_ids=joint_indices,
        )
        robots[module_id].write_joint_damping_to_sim_index(
            damping=float(damping_nms_per_rad),
            joint_ids=joint_indices,
        )
    return stiffness_targets, damping_targets


def _schedule_contact_joint_drive_damping(
    robots: Mapping[int, Any],
    expected_joint_ids: Sequence[str],
    *,
    nominal_damping_nms_per_rad: float,
    damping_multiplier: float,
    maximum_damping_nms_per_rad: float,
) -> dict[str, float]:
    """Schedule simulator-side implicit damping on every Dock joint after q_close.

    Isaac applies this damping inside the implicit actuator solve, avoiding the
    one-control-step delay of feeding measured velocity back as a new velocity
    target.  This numerical gain consumes no contact truth.  Hardware fidelity
    is enforced separately on the resulting applied torque, current estimate,
    and joint speed rather than by equating this gain to the raw MIT Kd field.
    """

    if (
        not math.isfinite(float(nominal_damping_nms_per_rad))
        or nominal_damping_nms_per_rad <= 0.0
    ):
        raise SchemaValidationError(
            "contact joint nominal drive damping must be finite and positive"
        )
    if not math.isfinite(float(damping_multiplier)) or damping_multiplier < 1.0:
        raise SchemaValidationError(
            "contact joint drive damping multiplier must be finite and at least one"
        )
    if (
        not math.isfinite(float(maximum_damping_nms_per_rad))
        or maximum_damping_nms_per_rad <= 0.0
    ):
        raise SchemaValidationError(
            "contact joint maximum drive damping must be finite and positive"
        )
    expected = tuple(str(joint_id) for joint_id in expected_joint_ids)
    if len(set(expected)) != len(expected):
        raise SchemaValidationError(
            "contact joint drive damping ids must be globally unique"
        )
    scheduled_damping = float(nominal_damping_nms_per_rad) * float(damping_multiplier)
    if not math.isfinite(scheduled_damping):
        raise SchemaValidationError(
            "scheduled contact joint drive damping must be finite"
        )
    if scheduled_damping > float(maximum_damping_nms_per_rad):
        raise SchemaValidationError(
            "scheduled contact joint drive damping exceeds the configured simulation limit"
        )

    joint_indices_by_module: dict[int, list[int]] = {}
    targets: dict[str, float] = {}
    for joint_id in expected:
        module_prefix, separator, local_joint_id = joint_id.partition(":")
        if not separator or not module_prefix.startswith("module_"):
            raise SchemaValidationError(
                f"invalid global Dock joint id for damping schedule: {joint_id!r}"
            )
        try:
            module_id = int(module_prefix.removeprefix("module_"))
        except ValueError as exc:
            raise SchemaValidationError(
                f"invalid module id in Dock damping schedule: {joint_id!r}"
            ) from exc
        robot = robots.get(module_id)
        if robot is None:
            raise SchemaValidationError(
                f"Dock damping schedule has no robot for module {module_id}"
            )
        matches = [
            index
            for index, name in enumerate(robot.joint_names)
            if str(name) == local_joint_id
        ]
        if len(matches) != 1:
            raise SchemaValidationError(
                f"Dock damping schedule could not resolve {joint_id!r} exactly once"
            )
        joint_indices_by_module.setdefault(module_id, []).append(matches[0])
        targets[joint_id] = scheduled_damping

    for module_id, joint_indices in sorted(joint_indices_by_module.items()):
        robots[module_id].write_joint_damping_to_sim_index(
            damping=scheduled_damping,
            joint_ids=joint_indices,
        )
    return targets


def _hold_latched_joint_positions(
    joint_result: Any,
    joint_vector: Any,
    *,
    position_reference_rad: Mapping[str, float],
) -> Any:
    """Hold the measured q_close without integrating residual IK error.

    The whole-structure Jacobian remains active for ``J.T @ wrench``.  Its
    pose-task output is deliberately discarded after q_close because a
    compliant fixed-Dock graph and the simulated articulation state need not
    have identical instantaneous root transforms.  Feeding that residual back
    into an absolute position reference ratchets the grasp shape even when the
    requested contact wrench is zero.
    """

    expected = tuple(str(joint_id) for joint_id in joint_vector.joint_ids)
    expected_set = set(expected)
    if len(expected_set) != len(expected):
        raise SchemaValidationError("latched q_close joint ids must be globally unique")
    frozen = {
        str(joint_id): float(value)
        for joint_id, value in position_reference_rad.items()
    }
    if set(frozen) != expected_set or not all(
        math.isfinite(value) for value in frozen.values()
    ):
        raise SchemaValidationError(
            "latched q_close position reference must cover exactly the Dock "
            "joints with finite values"
        )
    policy = joint_result.policy_command
    for name, values in (
        ("joint_position_targets", policy.joint_position_targets),
        ("joint_velocity_targets", policy.joint_velocity_targets),
        ("joint_torque_bias", policy.joint_torque_bias),
    ):
        if set(values) != expected_set:
            raise SchemaValidationError(
                f"latched q_close {name} must cover exactly the Dock joints"
            )
    held_policy = replace(
        policy,
        joint_position_targets={joint_id: frozen[joint_id] for joint_id in expected},
        joint_velocity_targets={joint_id: 0.0 for joint_id in expected},
    )
    held_policy.validate()
    return replace(joint_result, policy_command=held_policy)


def _dock_joint_actuator_telemetry(
    robots: Mapping[int, Any],
    expected_joint_ids: tuple[str, ...],
    *,
    requested_position_targets: Mapping[str, float],
    requested_velocity_targets: Mapping[str, float],
    requested_unclipped_torque_bias: Mapping[str, float],
    requested_limited_torque_bias: Mapping[str, float],
    peak_torque_nm: float,
    peak_current_a: float,
) -> dict[str, dict[str, object]]:
    expected = set(expected_joint_ids)
    for name, values in (
        ("requested_position_targets", requested_position_targets),
        ("requested_velocity_targets", requested_velocity_targets),
        (
            "requested_unclipped_torque_bias",
            requested_unclipped_torque_bias,
        ),
        ("requested_limited_torque_bias", requested_limited_torque_bias),
    ):
        if set(values) != expected:
            raise SchemaValidationError(
                f"Order8 {name} must cover exactly the Dock joint ids"
            )

    telemetry: dict[str, dict[str, object]] = {}
    tensor_fields = (
        "joint_pos",
        "joint_vel",
        "joint_pos_target",
        "joint_vel_target",
        "joint_effort_target",
        "computed_torque",
        "applied_torque",
        "joint_stiffness",
        "joint_damping",
        "joint_effort_limits",
    )
    values_by_module: dict[int, dict[str, list[float]]] = {}
    for global_joint_id in expected_joint_ids:
        module_text, local_joint_id = global_joint_id.split(":", 1)
        module_id = int(module_text[len("module_") :])
        robot = robots.get(module_id)
        if robot is None:
            raise RuntimeError(f"Order8 telemetry cannot resolve module {module_id}")
        resolved_name = _resolve_name(robot.joint_names, local_joint_id)
        if resolved_name is None:
            raise RuntimeError(
                "Order8 telemetry cannot resolve Dock joint " f"{global_joint_id!r}"
            )
        if module_id not in values_by_module:
            values_by_module[module_id] = {}
            for field_name in tensor_fields:
                field = getattr(robot.data, field_name, None)
                if field is None:
                    raise RuntimeError(
                        "Order8 telemetry requires Isaac articulation field "
                        f"{field_name!r}"
                    )
                values_by_module[module_id][field_name] = _tensor_row(field)
        joint_index = robot.joint_names.index(resolved_name)
        fields = values_by_module[module_id]
        telemetry[global_joint_id] = _dock_joint_actuator_telemetry_entry(
            requested_position_target_rad=float(
                requested_position_targets[global_joint_id]
            ),
            requested_velocity_target_radps=float(
                requested_velocity_targets[global_joint_id]
            ),
            requested_unclipped_torque_bias_nm=float(
                requested_unclipped_torque_bias[global_joint_id]
            ),
            requested_limited_torque_bias_nm=float(
                requested_limited_torque_bias[global_joint_id]
            ),
            measured_position_rad=float(fields["joint_pos"][joint_index]),
            measured_velocity_radps=float(fields["joint_vel"][joint_index]),
            isaac_position_target_rad=float(fields["joint_pos_target"][joint_index]),
            isaac_velocity_target_radps=float(fields["joint_vel_target"][joint_index]),
            isaac_effort_target_nm=float(fields["joint_effort_target"][joint_index]),
            isaac_computed_torque_nm=float(fields["computed_torque"][joint_index]),
            isaac_applied_torque_nm=float(fields["applied_torque"][joint_index]),
            stiffness_nm_per_rad=float(fields["joint_stiffness"][joint_index]),
            damping_nms_per_rad=float(fields["joint_damping"][joint_index]),
            effort_limit_sim_nm=float(fields["joint_effort_limits"][joint_index]),
            peak_torque_nm=peak_torque_nm,
            peak_current_a=peak_current_a,
        )
        telemetry[global_joint_id]["module_id"] = module_id
        telemetry[global_joint_id]["resolved_joint_name"] = resolved_name
    return telemetry


def _dock_joint_actuator_telemetry_entry(
    *,
    requested_position_target_rad: float,
    requested_velocity_target_radps: float,
    requested_unclipped_torque_bias_nm: float,
    requested_limited_torque_bias_nm: float,
    measured_position_rad: float,
    measured_velocity_radps: float,
    isaac_position_target_rad: float,
    isaac_velocity_target_radps: float,
    isaac_effort_target_nm: float,
    isaac_computed_torque_nm: float,
    isaac_applied_torque_nm: float,
    stiffness_nm_per_rad: float,
    damping_nms_per_rad: float,
    effort_limit_sim_nm: float,
    peak_torque_nm: float | None = None,
    peak_current_a: float | None = None,
) -> dict[str, object]:
    values = {
        "requested_position_target_rad": requested_position_target_rad,
        "requested_velocity_target_radps": requested_velocity_target_radps,
        "requested_unclipped_torque_bias_nm": (requested_unclipped_torque_bias_nm),
        "requested_limited_torque_bias_nm": requested_limited_torque_bias_nm,
        "measured_position_rad": measured_position_rad,
        "measured_velocity_radps": measured_velocity_radps,
        "isaac_position_target_rad": isaac_position_target_rad,
        "isaac_velocity_target_radps": isaac_velocity_target_radps,
        "isaac_effort_target_nm": isaac_effort_target_nm,
        "isaac_computed_torque_nm": isaac_computed_torque_nm,
        "isaac_applied_torque_nm": isaac_applied_torque_nm,
        "stiffness_nm_per_rad": stiffness_nm_per_rad,
        "damping_nms_per_rad": damping_nms_per_rad,
        "effort_limit_sim_nm": effort_limit_sim_nm,
    }
    if not all(math.isfinite(float(value)) for value in values.values()):
        raise SchemaValidationError(
            "Order8 Dock actuator telemetry values must be finite"
        )
    if (peak_torque_nm is None) != (peak_current_a is None):
        raise SchemaValidationError(
            "Order8 Dock current estimate requires both peak torque and current"
        )
    estimated_current_a = None
    if peak_torque_nm is not None and peak_current_a is not None:
        peak_torque = float(peak_torque_nm)
        peak_current = float(peak_current_a)
        if (
            not math.isfinite(peak_torque)
            or not math.isfinite(peak_current)
            or peak_torque <= 0.0
            or peak_current <= 0.0
        ):
            raise SchemaValidationError(
                "Order8 Dock peak torque/current must be finite and positive"
            )
        # No simulated current sensor exists.  This conservative, explicit
        # torque-proportional estimate audits the manufacturer peak-current
        # envelope without pretending it is hardware telemetry.
        estimated_current_a = (
            abs(float(isaac_applied_torque_nm)) / peak_torque * peak_current
        )
    position_error = isaac_position_target_rad - measured_position_rad
    velocity_error = isaac_velocity_target_radps - measured_velocity_radps
    estimated_position_drive = stiffness_nm_per_rad * position_error
    estimated_damping_drive = damping_nms_per_rad * velocity_error
    estimated_total = (
        estimated_position_drive + estimated_damping_drive + isaac_effort_target_nm
    )
    result = {
        **{key: float(value) for key, value in values.items()},
        "position_error_rad": float(position_error),
        "velocity_error_radps": float(velocity_error),
        "estimated_position_drive_torque_nm": float(estimated_position_drive),
        "estimated_damping_drive_torque_nm": float(estimated_damping_drive),
        "estimated_total_drive_torque_nm": float(estimated_total),
        "torque_bias_limited": not math.isclose(
            requested_unclipped_torque_bias_nm,
            requested_limited_torque_bias_nm,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ),
    }
    if estimated_current_a is not None:
        result["estimated_current_a"] = float(estimated_current_a)
        result["current_estimate_method"] = (
            "absolute_applied_torque_linear_peak_ratio_v1"
        )
    return result


def _telemetry_max_abs(
    telemetry: Mapping[str, Mapping[str, object]],
    field_name: str,
) -> float:
    return max(
        (
            abs(float(values[field_name]))
            for values in telemetry.values()
            if field_name in values
        ),
        default=0.0,
    )


def _module_frame_pose_twist(
    robot: Any,
    *,
    module_frame_link_id: str,
) -> tuple[Pose7D, tuple[float, float, float, float, float, float]]:
    """Read the PhysicalModel module frame, not the articulation root/COM."""

    body_name = _resolve_name(robot.body_names, module_frame_link_id)
    if body_name is None:
        raise RuntimeError(
            f"Order8 cannot resolve module frame body {module_frame_link_id!r}"
        )
    body_index = robot.body_names.index(body_name)
    position = _tensor_body_row(robot.data.body_pos_w, body_index)
    orientation = _tensor_body_row(robot.data.body_quat_w, body_index)
    linear_tensor = getattr(
        robot.data,
        "body_link_lin_vel_w",
        robot.data.body_lin_vel_w,
    )
    angular_tensor = getattr(
        robot.data,
        "body_link_ang_vel_w",
        robot.data.body_ang_vel_w,
    )
    linear = _tensor_body_row(linear_tensor, body_index)
    angular = _tensor_body_row(angular_tensor, body_index)
    return (
        tuple(position + orientation),  # type: ignore[return-value]
        tuple(linear + angular),  # type: ignore[return-value]
    )


def _tensor_row(tensor: Any) -> list[float]:
    if hasattr(tensor, "torch"):
        tensor = tensor.torch
    row = tensor[0] if getattr(tensor, "ndim", 1) > 1 else tensor
    return [float(value) for value in row.detach().cpu().tolist()]


def _tensor_body_row(tensor: Any, index: int) -> list[float]:
    if hasattr(tensor, "torch"):
        tensor = tensor.torch
    return [float(value) for value in tensor[0, index].detach().cpu().tolist()]


def _resolve_name(names: list[str], local_id: str) -> str | None:
    if local_id in names:
        return local_id
    matches = [name for name in names if name.endswith("__" + local_id)]
    return matches[0] if len(matches) == 1 else None


def _apply_record(
    robot: Any,
    record: Any,
    physical_model: Any,
    *,
    module_id: int,
    device: str,
) -> dict[str, int]:
    import torch

    rotor_by_id = {rotor.rotor_id: rotor for rotor in physical_model.rotors}
    force_ids: list[int] = []
    force_rows: list[list[float]] = []
    torque_rows: list[list[float]] = []
    pos_ids: list[int] = []
    pos_targets: list[float] = []
    vel_ids: list[int] = []
    vel_targets: list[float] = []
    effort_ids: list[int] = []
    effort_targets: list[float] = []
    unresolved = 0
    for target in record.actuator_targets:
        if int(target.metadata.get("module_id", module_id)) != module_id:
            continue
        local_id = str(target.metadata.get("local_id", target.command_key))
        if target.actuator_type == "rotor_thrust":
            rotor = rotor_by_id.get(local_id)
            body_name = _resolve_name(robot.body_names, local_id)
            if rotor is None or body_name is None:
                unresolved += 1
                continue
            force_ids.append(robot.body_names.index(body_name))
            force_rows.append(
                [
                    float(axis) * float(target.target_value)
                    for axis in rotor.thrust_axis_local
                ]
            )
            torque_rows.append(
                [
                    float(axis)
                    * float(rotor.reaction_torque_coeff_nm_per_n)
                    * float(target.target_value)
                    for axis in rotor.thrust_axis_local
                ]
            )
        elif target.actuator_type in {
            "vectoring_joint_position",
            "dock_joint_position",
            "joint_position",
        }:
            name = _resolve_name(robot.joint_names, local_id)
            if name is None:
                unresolved += 1
                continue
            pos_ids.append(robot.joint_names.index(name))
            pos_targets.append(float(target.target_value))
        elif target.actuator_type == "joint_velocity":
            name = _resolve_name(robot.joint_names, local_id)
            if name is None:
                unresolved += 1
                continue
            vel_ids.append(robot.joint_names.index(name))
            vel_targets.append(float(target.target_value))
        elif target.actuator_type in {"joint_effort", "joint_effort_bias"}:
            name = _resolve_name(robot.joint_names, local_id)
            if name is None:
                unresolved += 1
                continue
            effort_ids.append(robot.joint_names.index(name))
            effort_targets.append(float(target.target_value))
        else:
            unresolved += 1
    if force_ids:
        robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=torch.tensor([force_rows], dtype=torch.float32, device=device),
            torques=torch.tensor([torque_rows], dtype=torch.float32, device=device),
            body_ids=torch.tensor(force_ids, dtype=torch.int32, device=device),
            is_global=False,
        )
    if pos_ids:
        robot.set_joint_position_target_index(
            target=torch.tensor([pos_targets], dtype=torch.float32, device=device),
            joint_ids=torch.tensor(pos_ids, dtype=torch.int32, device=device),
        )
    if vel_ids:
        robot.set_joint_velocity_target_index(
            target=torch.tensor([vel_targets], dtype=torch.float32, device=device),
            joint_ids=torch.tensor(vel_ids, dtype=torch.int32, device=device),
        )
    if effort_ids:
        robot.set_joint_effort_target_index(
            target=torch.tensor([effort_targets], dtype=torch.float32, device=device),
            joint_ids=torch.tensor(effort_ids, dtype=torch.int32, device=device),
        )
    return {"unresolved_target_count": unresolved}


def _global_dock_position_map(joint_vector: Any) -> dict[str, float]:
    joint_ids = tuple(joint_vector.joint_ids)
    positions = tuple(joint_vector.positions_rad)
    if len(joint_ids) != len(positions):
        raise SchemaValidationError(
            "Dock joint ids and positions must have the same length"
        )
    if len(set(joint_ids)) != len(joint_ids):
        raise SchemaValidationError("Dock joint ids must be unique")
    return {
        str(joint_id): float(position)
        for joint_id, position in zip(joint_ids, positions, strict=True)
    }


def _anchor_task_linearizations(
    kinematics: Any,
    selections: list[Any],
    *,
    desired_anchor_poses: dict[int, Pose7D],
    wrench_targets: dict[int, tuple[float, ...]],
    task_priorities: dict[int, float] | None = None,
    orientation_task_weight: float = 0.05,
    current_anchor_poses_world: Mapping[int, Pose7D] | None = None,
    task_application_points_world: (
        Mapping[int, tuple[float, float, float]] | None
    ) = None,
    desired_task_application_points_world: (
        Mapping[int, tuple[float, float, float]] | None
    ) = None,
    wrench_application_points_world: (
        Mapping[int, tuple[float, float, float]] | None
    ) = None,
) -> list[Any]:
    from amsrr.controllers.natural_contact_joint_controller import (
        AnchorTaskLinearization,
    )

    selected_ids = {int(selection.anchor_id) for selection in selections}
    if len(selected_ids) != len(selections):
        raise SchemaValidationError(
            "selected natural-contact anchor ids must be unique"
        )
    if set(desired_anchor_poses) != selected_ids:
        raise SchemaValidationError(
            "desired anchor poses must cover exactly the selected anchors"
        )
    if set(wrench_targets) != selected_ids:
        raise SchemaValidationError(
            "wrench targets must cover exactly the selected anchors"
        )
    if (
        current_anchor_poses_world is not None
        and set(current_anchor_poses_world) != selected_ids
    ):
        raise SchemaValidationError(
            "current anchor poses must cover exactly the selected anchors"
        )
    if (
        task_application_points_world is not None
        and set(task_application_points_world) != selected_ids
    ):
        raise SchemaValidationError(
            "task application points must cover exactly the selected anchors"
        )
    if (
        desired_task_application_points_world is not None
        and set(desired_task_application_points_world) != selected_ids
    ):
        raise SchemaValidationError(
            "desired task application points must cover exactly the selected anchors"
        )
    if (
        desired_task_application_points_world is not None
        and task_application_points_world is None
    ):
        raise SchemaValidationError(
            "desired task application points require current task application points"
        )
    if (
        wrench_application_points_world is not None
        and set(wrench_application_points_world) != selected_ids
    ):
        raise SchemaValidationError(
            "wrench application points must cover exactly the selected anchors"
        )
    if task_priorities is None:
        task_priorities = {anchor_id: 1.0 for anchor_id in selected_ids}
    if set(task_priorities) != selected_ids or any(
        not math.isfinite(float(value)) or not 0.0 < float(value) <= 1.0
        for value in task_priorities.values()
    ):
        raise SchemaValidationError(
            "task priorities must cover exactly the selected anchors with "
            "finite values in (0, 1]"
        )
    if (
        not math.isfinite(float(orientation_task_weight))
        or not 0.0 < float(orientation_task_weight) <= 1.0
    ):
        raise SchemaValidationError(
            "orientation task weight must be finite and in (0, 1]"
        )
    if set(kinematics.anchor_poses_world) != selected_ids:
        raise SchemaValidationError(
            "whole-structure FK must cover exactly the selected anchors"
        )
    if set(kinematics.anchor_jacobians) != selected_ids:
        raise SchemaValidationError(
            "whole-structure Jacobians must cover exactly the selected anchors"
        )

    joint_count = len(kinematics.ordered_global_dock_joint_ids)
    tasks = []
    for selection in sorted(selections, key=lambda item: item.anchor_id):
        anchor_id = int(selection.anchor_id)
        current_pose = (
            kinematics.anchor_poses_world[anchor_id]
            if current_anchor_poses_world is None
            else current_anchor_poses_world[anchor_id]
        )
        desired_pose = desired_anchor_poses[anchor_id]
        jacobian = kinematics.anchor_jacobians[anchor_id]
        if len(jacobian) != 6 or any(len(row) != joint_count for row in jacobian):
            raise SchemaValidationError(
                f"anchor {anchor_id} Jacobian must be 6x{joint_count}"
            )
        current_task_point = tuple(float(value) for value in current_pose[:3])
        desired_task_point = tuple(float(value) for value in desired_pose[:3])
        if task_application_points_world is not None:
            current_task_point = tuple(
                float(value) for value in task_application_points_world[anchor_id]
            )
            if len(current_task_point) != 3 or not all(
                math.isfinite(value) for value in current_task_point
            ):
                raise SchemaValidationError(
                    f"anchor {anchor_id} task application point must contain "
                    "three finite values"
                )
            jacobian = _spatial_jacobian_at_world_point(
                jacobian,
                origin_world=tuple(float(value) for value in current_pose[:3]),
                point_world=current_task_point,
            )
            if desired_task_application_points_world is None:
                desired_task_point = _rigid_point_pose_following_anchor_target(
                    current_anchor_pose_world=current_pose,
                    current_point_world=current_task_point,
                    desired_anchor_pose_world=desired_pose,
                )[:3]
            else:
                desired_task_point = tuple(
                    float(value)
                    for value in desired_task_application_points_world[anchor_id]
                )
                if len(desired_task_point) != 3 or not all(
                    math.isfinite(value) for value in desired_task_point
                ):
                    raise SchemaValidationError(
                        f"anchor {anchor_id} desired task application point must "
                        "contain three finite values"
                    )
        error = tuple(
            desired_task_point[index] - current_task_point[index]
            for index in range(3)
        ) + _rotation_error_world(current_pose, desired_pose)
        wrench = tuple(float(value) for value in wrench_targets[anchor_id])
        if len(wrench) != 6 or not all(math.isfinite(value) for value in wrench):
            raise SchemaValidationError(
                f"anchor {anchor_id} wrench target must contain six finite values"
            )
        if wrench_application_points_world is not None:
            application_point = tuple(
                float(value) for value in wrench_application_points_world[anchor_id]
            )
            if len(application_point) != 3 or not all(
                math.isfinite(value) for value in application_point
            ):
                raise SchemaValidationError(
                    f"anchor {anchor_id} wrench application point must contain "
                    "three finite values"
                )
            moment_arm = tuple(
                application_point[index] - current_task_point[index]
                for index in range(3)
            )
            shifted_moment = _cross(moment_arm, wrench[:3])
            wrench = (
                *wrench[:3],
                *tuple(
                    float(wrench[3 + index]) + shifted_moment[index]
                    for index in range(3)
                ),
            )
        priority = float(task_priorities[anchor_id])
        tasks.append(
            AnchorTaskLinearization(
                anchor_id=anchor_id,
                task_error=error,
                jacobian=jacobian,
                wrench_bias=wrench,
                task_weights=(
                    priority,
                    priority,
                    priority,
                    float(orientation_task_weight) * priority,
                    float(orientation_task_weight) * priority,
                    float(orientation_task_weight) * priority,
                ),
            )
        )
    return tasks


def _anchor_tasks_from_planner_trajectory(
    kinematics: Any,
    selections: list[Any],
    trajectory: Any,
    *,
    current_anchor_poses_world: Mapping[int, Pose7D] | None = None,
    task_application_points_world: (
        Mapping[int, tuple[float, float, float]] | None
    ) = None,
    desired_task_application_points_world: (
        Mapping[int, tuple[float, float, float]] | None
    ) = None,
    wrench_application_points_world: (
        Mapping[int, tuple[float, float, float]] | None
    ) = None,
) -> list[Any]:
    if not trajectory.knots:
        raise SchemaValidationError("planner trajectory must contain an active knot")
    knot = trajectory.knots[0]
    posture = knot.posture_target
    desired = (
        {}
        if posture is None or posture.free_anchor_pose_targets is None
        else dict(posture.free_anchor_pose_targets)
    )
    if not desired:
        if knot.contact_assignments:
            raise SchemaValidationError(
                "planner contact assignments require matching anchor pose targets"
            )
        return []
    wrench_targets: dict[int, tuple[float, ...]] = {}
    task_priorities: dict[int, float] = {}
    for assignment in knot.contact_assignments:
        if assignment.anchor_id in wrench_targets:
            raise SchemaValidationError(
                "planner active knot contains duplicate anchor assignments"
            )
        if assignment.wrench_target is None:
            raise SchemaValidationError(
                f"planner anchor {assignment.anchor_id} lacks a wrench target"
            )
        wrench_targets[int(assignment.anchor_id)] = tuple(
            float(value) for value in assignment.wrench_target
        )
        task_priorities[int(assignment.anchor_id)] = float(
            getattr(assignment, "priority", 1.0)
        )
    return _anchor_task_linearizations(
        kinematics,
        selections,
        desired_anchor_poses=desired,
        wrench_targets=wrench_targets,
        task_priorities=task_priorities,
        orientation_task_weight=float(
            getattr(knot, "priority_weights", {}).get(
                "anchor_orientation",
                0.05,
            )
        ),
        current_anchor_poses_world=current_anchor_poses_world,
        task_application_points_world=task_application_points_world,
        desired_task_application_points_world=(
            desired_task_application_points_world
        ),
        wrench_application_points_world=wrench_application_points_world,
    )


def _spatial_jacobian_at_world_point(
    jacobian_at_origin: Sequence[Sequence[float]],
    *,
    origin_world: Sequence[float],
    point_world: Sequence[float],
) -> tuple[tuple[float, ...], ...]:
    """Shift a world-frame geometric Jacobian from an origin to a rigid point."""

    rows = tuple(tuple(float(value) for value in row) for row in jacobian_at_origin)
    origin = tuple(float(value) for value in origin_world)
    point = tuple(float(value) for value in point_world)
    if (
        len(rows) != 6
        or not rows
        or any(len(row) != len(rows[0]) for row in rows)
        or len(origin) != 3
        or len(point) != 3
        or not all(math.isfinite(value) for row in rows for value in row)
        or not all(math.isfinite(value) for value in (*origin, *point))
    ):
        raise SchemaValidationError(
            "surface-point Jacobian shift requires finite 6xN rows and two 3D points"
        )
    offset = tuple(point[index] - origin[index] for index in range(3))
    shifted_linear_rows = [list(row) for row in rows[:3]]
    for column in range(len(rows[0])):
        angular = (rows[3][column], rows[4][column], rows[5][column])
        point_velocity = _cross(angular, offset)
        for axis in range(3):
            shifted_linear_rows[axis][column] += point_velocity[axis]
    return (
        *(tuple(float(value) for value in row) for row in shifted_linear_rows),
        rows[3],
        rows[4],
        rows[5],
    )


def _rigid_point_pose_following_anchor_target(
    *,
    current_anchor_pose_world: Pose7D,
    current_point_world: Sequence[float],
    desired_anchor_pose_world: Pose7D,
) -> Pose7D:
    """Carry a rigid material point through an anchor pose target.

    Policy posture commands remain anchor-frame poses.  During final closure,
    however, differential IK translates an authored Dock-mesh surface sample.
    Carrying the sampled point through the desired anchor transform keeps the
    task error and the shifted Jacobian tied to the same physical point.
    """

    point = tuple(float(value) for value in current_point_world)
    if len(point) != 3 or not all(math.isfinite(value) for value in point):
        raise SchemaValidationError(
            "rigid point following requires three finite world coordinates"
        )
    point_in_anchor = compose_pose(
        inverse_pose(current_anchor_pose_world),
        (*point, 0.0, 0.0, 0.0, 1.0),
    )
    desired_point_pose = compose_pose(desired_anchor_pose_world, point_in_anchor)
    return (
        *tuple(float(value) for value in desired_point_pose[:3]),
        *tuple(float(value) for value in desired_anchor_pose_world[3:7]),
    )


def _base_target_from_planner_trajectory(trajectory: Any) -> Pose7D:
    if not trajectory.knots:
        raise SchemaValidationError("planner trajectory must contain an active knot")
    target = trajectory.knots[0].centroidal_target
    if (
        target is None
        or target.com_pos_world is None
        or target.body_orientation_world is None
    ):
        raise SchemaValidationError(
            "planner active knot must contain a complete centroidal pose target"
        )
    return (
        *(float(value) for value in target.com_pos_world),
        *(float(value) for value in target.body_orientation_world),
    )


def _base_twist_from_planner_trajectory(
    trajectory: Any,
) -> tuple[float, float, float, float, float, float]:
    if not trajectory.knots:
        raise SchemaValidationError("planner trajectory must contain an active knot")
    target = trajectory.knots[0].centroidal_target
    if target is None:
        raise SchemaValidationError(
            "planner active knot must contain a centroidal target"
        )
    linear = target.com_vel_world or (0.0, 0.0, 0.0)
    return (
        *(float(value) for value in linear),
        0.0,
        0.0,
        0.0,
    )


def _rotation_error_world(
    current_pose: Pose7D, desired_pose: Pose7D
) -> tuple[float, float, float]:
    from amsrr.geometry.pose_math import (
        matmul,
        quat_from_matrix,
        transform_from_pose,
        transpose,
    )

    current_rotation = transform_from_pose(current_pose).rotation
    desired_rotation = transform_from_pose(desired_pose).rotation
    desired_from_current = matmul(desired_rotation, transpose(current_rotation))
    qx, qy, qz, qw = quat_from_matrix(desired_from_current)
    vector_norm = math.sqrt(qx * qx + qy * qy + qz * qz)
    if vector_norm <= 1.0e-12:
        return (0.0, 0.0, 0.0)
    angle = 2.0 * math.atan2(vector_norm, qw)
    scale = angle / vector_norm
    return (scale * qx, scale * qy, scale * qz)


def _base_target_for_phase(
    phase: Any,
    *,
    hover_base_pose: Pose7D,
    approach_base_pose: Pose7D,
    grasp_base_pose: Pose7D,
    lift_base_pose: Pose7D,
    transport_base_pose: Pose7D,
    place_base_pose: Pose7D,
    retreat_base_pose: Pose7D,
) -> Pose7D:
    if phase == Order8NaturalContactPhase.RESET:
        return hover_base_pose
    if phase == Order8NaturalContactPhase.APPROACH:
        return approach_base_pose
    if phase == Order8NaturalContactPhase.CONTACT_ACQUISITION:
        return grasp_base_pose
    if phase == Order8NaturalContactPhase.LIFT:
        return lift_base_pose
    if phase == Order8NaturalContactPhase.TRANSPORT:
        return transport_base_pose
    if phase in {Order8NaturalContactPhase.PLACE, Order8NaturalContactPhase.RELEASE}:
        return place_base_pose
    return retreat_base_pose


def _floor_clear_grasp_base_plan(
    *,
    floor_base_pose: Pose7D,
    unconstrained_grasp_base_pose: Pose7D,
    inward_normal_world_by_anchor: Mapping[int, tuple[float, float, float]],
    tangential_tolerance_m: float,
    additional_floor_clearance_m: float = 0.0,
) -> _FloorClearGraspBasePlan:
    """Keep the connected robot out of the floor without changing closure.

    The object rests on the floor, but the representative opposing surfaces
    have horizontal normals and a two-dimensional contact region.  Aligning
    the anchor-pair *centre* with the object centre in all three axes lowered
    the complete morphology below its collision-derived floor pose.  QPID
    consequently pressed the batteries into the floor while the Dock servos
    tried to close the pair.  Preserve the collision-derived base height
    instead: the resulting vertical offset is tangential to both faces and is
    valid only when it lies inside the configured contact region.

    This helper deliberately fails closed for a surface whose normal has a
    vertical component.  Such a case needs a different whole-body contact
    plan; silently converting a normal error into a tangential allowance would
    weaken the grasp geometry.
    """

    floor = tuple(float(value) for value in floor_base_pose)
    unconstrained = tuple(float(value) for value in unconstrained_grasp_base_pose)
    if (
        len(floor) != 7
        or len(unconstrained) != 7
        or not all(math.isfinite(value) for value in (*floor, *unconstrained))
    ):
        raise SchemaValidationError(
            "Order8 floor/grasp base poses must be finite Pose7D values"
        )
    tolerance = float(tangential_tolerance_m)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise SchemaValidationError(
            "Order8 grasp tangential tolerance must be finite and positive"
        )
    additional_clearance = float(additional_floor_clearance_m)
    if not math.isfinite(additional_clearance) or additional_clearance < 0.0:
        raise SchemaValidationError(
            "Order8 additional grasp floor clearance must be finite and non-negative"
        )
    if not inward_normal_world_by_anchor or any(
        not isinstance(anchor_id, int) or isinstance(anchor_id, bool) or anchor_id < 0
        for anchor_id in inward_normal_world_by_anchor
    ):
        raise SchemaValidationError(
            "Order8 floor-clear grasp plan requires non-negative anchor ids"
        )

    target_z = max(
        float(unconstrained[2]),
        float(floor[2]) + additional_clearance,
    )
    correction = (0.0, 0.0, target_z - float(unconstrained[2]))
    normal_corrections: dict[int, float] = {}
    tangential_corrections: dict[int, tuple[float, float]] = {}
    for anchor_id, raw_normal in sorted(inward_normal_world_by_anchor.items()):
        normal = _unit(tuple(float(value) for value in raw_normal))
        tangent_values = _tangent_basis(normal)
        tangents = (tuple(tangent_values[:3]), tuple(tangent_values[3:6]))
        normal_correction = sum(correction[index] * normal[index] for index in range(3))
        tangential = tuple(
            sum(correction[index] * tangent[index] for index in range(3))
            for tangent in tangents
        )
        if abs(normal_correction) > 1.0e-9:
            raise SchemaValidationError(
                "Order8 floor-clear correction changes a selected surface normal; "
                "a different contact plan is required"
            )
        if any(abs(value) > tolerance + 1.0e-12 for value in tangential):
            raise SchemaValidationError(
                "Order8 floor-clear correction exceeds the selected surface "
                "tangential contact region"
            )
        normal_corrections[int(anchor_id)] = normal_correction
        tangential_corrections[int(anchor_id)] = tangential

    base_pose: Pose7D = (
        float(unconstrained[0]),
        float(unconstrained[1]),
        target_z,
        *tuple(float(value) for value in unconstrained[3:7]),
    )
    return _FloorClearGraspBasePlan(
        base_pose_world=base_pose,
        unconstrained_base_pose_world=unconstrained,  # type: ignore[arg-type]
        vertical_correction_m=correction[2],
        normal_correction_m_by_anchor=normal_corrections,
        tangential_correction_m_by_anchor=tangential_corrections,
    )


def _desired_anchor_poses(
    selections: list[Any],
    object_pose: Pose7D,
    object_size: list[float],
    *,
    pregrasp: bool,
    inward_overtravel_m: float = 0.0,
    orientation_by_anchor_id: dict[int, tuple[float, ...]] | None = None,
) -> dict[int, Pose7D]:
    if (
        not isinstance(inward_overtravel_m, (int, float))
        or isinstance(inward_overtravel_m, bool)
        or not math.isfinite(float(inward_overtravel_m))
        or float(inward_overtravel_m) < 0.0
    ):
        raise SchemaValidationError(
            "desired anchor inward_overtravel_m must be finite and non-negative"
        )
    result = {}
    for selection in selections:
        normal = selection.inward_normal_world
        support = 0.5 * sum(
            abs(float(normal[index])) * float(object_size[index]) for index in range(3)
        )
        support += 0.05 if pregrasp else 0.0
        position = tuple(
            float(object_pose[index])
            - support * float(normal[index])
            + float(inward_overtravel_m) * float(normal[index])
            for index in range(3)
        )
        orientation = (
            tuple(
                float(value) for value in orientation_by_anchor_id[selection.anchor_id]
            )
            if orientation_by_anchor_id is not None
            else (0.0, 0.0, 0.0, 1.0)
        )
        if len(orientation) != 4:
            raise SchemaValidationError(
                "desired anchor orientation must be a quaternion"
            )
        result[selection.anchor_id] = (*position, *orientation)
    return result


def _pose_following_object_motion(
    reference_object_pose: Pose7D,
    current_object_pose: Pose7D,
    reference_world_pose: Pose7D,
) -> Pose7D:
    """Move a reference with the observed free object without writing its state."""

    return compose_pose(
        current_object_pose,
        compose_pose(inverse_pose(reference_object_pose), reference_world_pose),
    )


def _horizontal_mesh_pair_centering_correction_world(
    *,
    surface_point_world_by_anchor: Mapping[
        int, tuple[float, float, float]
    ],
    nominal_contact_pose_world_by_anchor: Mapping[int, Pose7D],
    approach_axis_world: tuple[float, float, float],
    maximum_correction_m: float,
) -> tuple[float, float, float]:
    """Centre the pair mean along the horizontal object-approach tangent.

    The two selected Dock meshes are different authored CAD parts, so their
    first physical contact patches need not coincide with identical connect
    frames.  A single centroidal translation cannot make both patches equal,
    but it can centre their mean and leave equal-and-opposite residuals inside
    the approved contact region.  Vertical correction is deliberately excluded:
    floor clearance already determines the feasible common contact height.
    """

    expected = set(surface_point_world_by_anchor)
    if expected != set(nominal_contact_pose_world_by_anchor) or len(expected) < 2:
        raise SchemaValidationError(
            "mesh pair centering requires matching maps with at least two anchors"
        )
    if (
        not math.isfinite(float(maximum_correction_m))
        or float(maximum_correction_m) <= 0.0
    ):
        raise SchemaValidationError(
            "mesh pair centering maximum correction must be finite and positive"
        )
    axis = tuple(float(value) for value in approach_axis_world)
    horizontal_axis = _unit((axis[0], axis[1], 0.0))
    mean_offset_m = sum(
        sum(
            (
                float(surface_point_world_by_anchor[anchor_id][index])
                - float(nominal_contact_pose_world_by_anchor[anchor_id][index])
            )
            * horizontal_axis[index]
            for index in range(3)
        )
        for anchor_id in expected
    ) / float(len(expected))
    correction_m = min(
        max(-mean_offset_m, -float(maximum_correction_m)),
        float(maximum_correction_m),
    )
    return tuple(correction_m * value for value in horizontal_axis)


def _contact_precenter_nominal_pose(
    nominal_contact_pose_world: Pose7D,
    *,
    object_pose_world: Pose7D,
    inward_normal_object: tuple[float, float, float],
    inward_overtravel_m: float,
    clearance_m: float,
) -> Pose7D:
    """Place the preferred mesh point clear of the face for tangential motion.

    The normal contact target already includes ``inward_overtravel_m``.  Before
    normal closure, retract that overtravel plus ``clearance_m`` along the
    object-following inward normal.  This lets every Dock joint solve the
    tangential mesh-centering motion without dragging a loaded collision mesh
    across the object.  No object state is written and no collision is disabled.
    """

    for name, value in (
        ("inward_overtravel_m", inward_overtravel_m),
        ("clearance_m", clearance_m),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise SchemaValidationError(
                f"contact mesh precenter {name} must be finite and non-negative"
            )
    from amsrr.geometry.pose_math import matvec, transform_from_pose

    normal_world = _unit(
        matvec(
            transform_from_pose(object_pose_world).rotation,
            _unit(tuple(float(value) for value in inward_normal_object)),
        )
    )
    retreat_m = float(inward_overtravel_m) + float(clearance_m)
    return _offset_pose(
        nominal_contact_pose_world,
        dx=-retreat_m * normal_world[0],
        dy=-retreat_m * normal_world[1],
        dz=-retreat_m * normal_world[2],
    )


def _contact_region_pose_target(
    *,
    current_anchor_pose_world: Pose7D,
    current_surface_point_world: tuple[float, float, float] | None = None,
    nominal_anchor_pose_world: Pose7D,
    object_pose_world: Pose7D,
    inward_normal_object: tuple[float, float, float],
    tangential_tolerance_m: float,
) -> Pose7D:
    """Target a preferred mesh contact point inside a tolerated face region.

    ``nominal_anchor_pose_world`` names the preferred object-face point.  When
    ``current_surface_point_world`` is provided, translation is solved for the
    actual selected authored-mesh sample instead of incorrectly treating the
    Dock connect frame as the collision point.  Within the component-wise
    ``+/- tolerance`` its measured tangential coordinate is retained; outside
    the region it is pulled only to the nearest boundary.  Pair-level mean
    centring is handled separately by the centroidal grasp-pose correction.
    Normal closure remains exact.

    The optional surface point preserves the helper's historical connect-frame
    behaviour for geometry-only callers, but the real Order-8 runtime always
    supplies its non-privileged selected-mesh sample.
    """

    if (
        not math.isfinite(float(tangential_tolerance_m))
        or float(tangential_tolerance_m) <= 0.0
    ):
        raise SchemaValidationError(
            "contact-region tangential tolerance must be finite and positive"
        )
    _unit(tuple(float(value) for value in inward_normal_object))
    surface_point = (
        tuple(float(value) for value in current_anchor_pose_world[:3])
        if current_surface_point_world is None
        else tuple(float(value) for value in current_surface_point_world)
    )
    if len(surface_point) != 3 or not all(
        math.isfinite(value) for value in surface_point
    ):
        raise SchemaValidationError(
            "contact-region surface point must contain three finite values"
        )
    tangential_offsets_m = _contact_region_tangential_offsets_m(
        current_anchor_pose_world=(*surface_point, *current_anchor_pose_world[3:7]),
        nominal_anchor_pose_world=nominal_anchor_pose_world,
        object_pose_world=object_pose_world,
        inward_normal_object=inward_normal_object,
    )
    if not all(math.isfinite(value) for value in tangential_offsets_m):
        raise SchemaValidationError(
            "contact-region tangential offsets must be finite"
        )
    normal = _unit(tuple(float(value) for value in inward_normal_object))
    tangent_values = _tangent_basis(normal)
    tangents_object = (
        tuple(tangent_values[:3]),
        tuple(tangent_values[3:6]),
    )
    object_rotation = transform_from_pose(object_pose_world).rotation
    tangents_world = tuple(
        _unit(matvec(object_rotation, tangent)) for tangent in tangents_object
    )
    allowed_tangent_world = [0.0, 0.0, 0.0]
    for tangent_world, component in zip(
        tangents_world, tangential_offsets_m, strict=True
    ):
        bounded = min(
            max(component, -float(tangential_tolerance_m)),
            float(tangential_tolerance_m),
        )
        for index in range(3):
            allowed_tangent_world[index] += bounded * tangent_world[index]
    desired_surface_point_world = tuple(
        float(nominal_anchor_pose_world[index]) + allowed_tangent_world[index]
        for index in range(3)
    )
    translation = tuple(
        desired_surface_point_world[index] - surface_point[index]
        for index in range(3)
    )
    return (
        *tuple(
            float(current_anchor_pose_world[index]) + translation[index]
            for index in range(3)
        ),
        *tuple(float(value) for value in nominal_anchor_pose_world[3:7]),
    )


def _contact_region_tangential_offsets_m(
    *,
    current_anchor_pose_world: Pose7D,
    nominal_anchor_pose_world: Pose7D,
    object_pose_world: Pose7D,
    inward_normal_object: tuple[float, float, float],
) -> tuple[float, float]:
    """Return the two object-face tangential offsets from nominal contact."""

    normal = _unit(tuple(float(value) for value in inward_normal_object))
    tangent_values = _tangent_basis(normal)
    tangents = (tuple(tangent_values[:3]), tuple(tangent_values[3:6]))
    object_from_world = inverse_pose(object_pose_world)
    current_object = compose_pose(object_from_world, current_anchor_pose_world)
    nominal_object = compose_pose(object_from_world, nominal_anchor_pose_world)
    delta = tuple(
        float(current_object[index]) - float(nominal_object[index])
        for index in range(3)
    )
    return tuple(
        sum(delta[index] * tangent[index] for index in range(3)) for tangent in tangents
    )


def _contact_anchor_references_with_measured_orientation(
    desired_anchor_poses_world: Mapping[int, Pose7D],
    measured_anchor_poses_world: Mapping[int, Pose7D],
) -> dict[int, Pose7D]:
    """Make contact acquisition translational while retaining a 6D task shape.

    Dock closure necessarily rotates the articulated chain.  Reusing the
    neutral/pregrasp quaternion as a simultaneous target made the regularizing
    orientation rows oppose the much smaller normal-creep command.  During
    acquisition the measured quaternion is therefore the zero-error reference;
    tangential translation remains bounded by the contact-region helper and
    the achieved absolute joint configuration is frozen after verified grasp.
    """

    desired_ids = set(desired_anchor_poses_world)
    if not desired_ids or desired_ids != set(measured_anchor_poses_world):
        raise SchemaValidationError(
            "contact anchor desired/measured poses must have identical non-empty coverage"
        )
    result: dict[int, Pose7D] = {}
    for anchor_id in sorted(desired_ids):
        desired = tuple(float(value) for value in desired_anchor_poses_world[anchor_id])
        measured = tuple(
            float(value) for value in measured_anchor_poses_world[anchor_id]
        )
        if (
            len(desired) != 7
            or len(measured) != 7
            or not all(math.isfinite(value) for value in (*desired, *measured))
        ):
            raise SchemaValidationError(
                "contact anchor desired/measured poses must be finite Pose7D values"
            )
        result[int(anchor_id)] = (*desired[:3], *measured[3:7])
    return result


def _measure_robot_object_contacts(
    contact_view: Any,
    *,
    sim_dt: float,
    sensor_body_paths: list[str],
    body_identity: list[str],
    body_lookup: list[tuple[int, str]],
    robots: dict[int, Any],
    object_state: dict[str, list[float]],
    selected_link_ids: set[str],
    wp: Any,
    torch: Any,
) -> tuple[Any, _IsaacContactVectorTelemetry]:
    from amsrr.simulation.order8_contact_measurement import (
        measure_order8_raw_contacts,
    )

    (
        force_buffer,
        point_buffer,
        normal_buffer,
        separation_buffer,
        count_buffer,
        start_buffer,
    ) = contact_view.get_contact_data(sim_dt)
    counts = wp.to_torch(count_buffer).reshape(-1).to(torch.int64)
    starts = wp.to_torch(start_buffer).reshape(-1).to(torch.int64)
    forces = wp.to_torch(force_buffer).reshape(-1)
    points = wp.to_torch(point_buffer).reshape(-1, 3)
    normals = wp.to_torch(normal_buffer).reshape(-1, 3)
    separations = wp.to_torch(separation_buffer).reshape(-1)
    (
        friction_force_buffer,
        _friction_impulse_buffer,
        friction_count_buffer,
        friction_start_buffer,
    ) = contact_view.get_friction_data(sim_dt)
    friction_forces = wp.to_torch(friction_force_buffer).reshape(-1, 3)
    friction_counts = wp.to_torch(friction_count_buffer).reshape(-1).to(torch.int64)
    friction_starts = wp.to_torch(friction_start_buffer).reshape(-1).to(torch.int64)
    contact_force_matrix = wp.to_torch(
        contact_view.get_contact_force_matrix(sim_dt)
    ).reshape(-1, 3)
    capacity = int(contact_view.max_contact_data_count)
    body_com_poses: dict[str, list[float]] = {}
    body_twists: dict[str, list[float]] = {}
    active_sensor_indices = {
        index
        for index, count in enumerate(counts.detach().cpu().tolist())
        if int(count) > 0
    }
    for sensor_index in sorted(active_sensor_indices):
        sensor_path = sensor_body_paths[sensor_index]
        module_id, local_name = body_lookup[sensor_index]
        robot = robots[module_id]
        body_name = _resolve_name(robot.body_names, local_name)
        if body_name is None:
            continue
        body_index = robot.body_names.index(body_name)
        com_pos_tensor = getattr(robot.data, "body_com_pos_w", robot.data.body_pos_w)
        linear_tensor = getattr(
            robot.data,
            "body_com_lin_vel_w",
            robot.data.body_lin_vel_w,
        )
        angular_tensor = getattr(
            robot.data,
            "body_com_ang_vel_w",
            robot.data.body_ang_vel_w,
        )
        body_com_poses[sensor_path] = _tensor_body_row(
            com_pos_tensor, body_index
        ) + _tensor_body_row(robot.data.body_quat_w, body_index)
        body_twists[sensor_path] = _tensor_body_row(
            linear_tensor, body_index
        ) + _tensor_body_row(angular_tensor, body_index)
    measurement = measure_order8_raw_contacts(
        sensor_body_ids=sensor_body_paths,
        sensor_global_link_ids=body_identity,
        selected_global_link_ids=sorted(selected_link_ids),
        object_body_id="order8_object",
        contact_counts=[int(value) for value in counts.detach().cpu().tolist()],
        start_indices=[int(value) for value in starts.detach().cpu().tolist()],
        patch_forces_n=[float(value) for value in forces.detach().cpu().tolist()],
        patch_points_world=[
            [float(value) for value in row] for row in points.detach().cpu().tolist()
        ],
        patch_normals_world=[
            [float(value) for value in row] for row in normals.detach().cpu().tolist()
        ],
        patch_separations_m=[
            float(value) for value in separations.detach().cpu().tolist()
        ],
        raw_capacity=capacity,
        body_com_poses_world=body_com_poses,
        body_twists_world=body_twists,
        object_com_pose_world=object_state["pose"],
        object_twist_world=object_state["twist"],
    )
    vector_telemetry = _contact_vector_telemetry_from_flat_buffers(
        body_identity=body_identity,
        selected_link_ids=selected_link_ids,
        contact_counts=[int(value) for value in counts.detach().cpu().tolist()],
        contact_starts=[int(value) for value in starts.detach().cpu().tolist()],
        normal_force_magnitudes_n=[
            float(value) for value in forces.detach().cpu().tolist()
        ],
        contact_normals_world=[
            tuple(float(value) for value in row)
            for row in normals.detach().cpu().tolist()
        ],
        contact_points_world=[
            tuple(float(value) for value in row)
            for row in points.detach().cpu().tolist()
        ],
        friction_counts=[
            int(value) for value in friction_counts.detach().cpu().tolist()
        ],
        friction_starts=[
            int(value) for value in friction_starts.detach().cpu().tolist()
        ],
        friction_forces_world=[
            tuple(float(value) for value in row)
            for row in friction_forces.detach().cpu().tolist()
        ],
        contact_force_matrix_world=[
            tuple(float(value) for value in row)
            for row in contact_force_matrix.detach().cpu().tolist()
        ],
    )
    maximum_slip_kinematics_by_link = (
        _maximum_tangential_slip_kinematics_by_link(
            measurement.patch_kinematics,
            selected_link_ids=selected_link_ids,
        )
    )
    zero_vector = (0.0, 0.0, 0.0)
    vector_telemetry = replace(
        vector_telemetry,
        body_linear_velocity_world_by_link={
            link_id: tuple(
                float(value)
                for value in body_twists.get(sensor_body_paths[index], [0.0] * 6)[:3]
            )
            for index, link_id in enumerate(body_identity)
            if link_id in selected_link_ids
        },
        body_contact_velocity_world_by_link={
            link_id: (
                maximum_slip_kinematics_by_link[
                    link_id
                ].body_contact_velocity_world_mps
                if link_id in maximum_slip_kinematics_by_link
                else zero_vector
            )
            for link_id in sorted(selected_link_ids)
        },
        object_contact_velocity_world_by_link={
            link_id: (
                maximum_slip_kinematics_by_link[
                    link_id
                ].object_contact_velocity_world_mps
                if link_id in maximum_slip_kinematics_by_link
                else zero_vector
            )
            for link_id in sorted(selected_link_ids)
        },
        relative_contact_velocity_world_by_link={
            link_id: (
                maximum_slip_kinematics_by_link[link_id].relative_velocity_world_mps
                if link_id in maximum_slip_kinematics_by_link
                else zero_vector
            )
            for link_id in sorted(selected_link_ids)
        },
        tangential_slip_velocity_world_by_link={
            link_id: (
                maximum_slip_kinematics_by_link[
                    link_id
                ].tangential_velocity_world_mps
                if link_id in maximum_slip_kinematics_by_link
                else zero_vector
            )
            for link_id in sorted(selected_link_ids)
        },
        tangential_slip_contact_point_world_by_link={
            link_id: (
                maximum_slip_kinematics_by_link[link_id].contact_point_world
                if link_id in maximum_slip_kinematics_by_link
                else zero_vector
            )
            for link_id in sorted(selected_link_ids)
        },
        tangential_slip_contact_normal_world_by_link={
            link_id: (
                maximum_slip_kinematics_by_link[link_id].contact_normal_world
                if link_id in maximum_slip_kinematics_by_link
                else zero_vector
            )
            for link_id in sorted(selected_link_ids)
        },
    )
    if not measurement.raw_contact_valid:
        suspicious_ranges = [
            {
                "pair_index": index,
                "start": int(start),
                "count": int(count),
            }
            for index, (start, count) in enumerate(
                zip(
                    starts.detach().cpu().tolist(),
                    counts.detach().cpu().tolist(),
                    strict=True,
                )
            )
            if int(count) > 0 or int(start) > capacity
        ]
        print(
            f"{ORDER8_PROGRESS_PREFIX} invalid_raw_contact "
            f"capacity={capacity} buffer_length={int(forces.numel())} "
            f"ranges={suspicious_ranges} "
            f"reasons={list(measurement.failure_reasons)}",
            file=__import__("sys").stderr,
            flush=True,
        )
    return measurement, vector_telemetry


def _maximum_tangential_slip_kinematics_by_link(
    patch_kinematics: Sequence[Any],
    *,
    selected_link_ids: Collection[str],
) -> dict[str, Any]:
    """Select the deterministic maximum-slip patch for each selected link.

    The acceptance monitor integrates the maximum scalar patch speed on each
    link.  Selecting the vector from that same patch keeps diagnostic signed
    displacement directly comparable with the unchanged scalar safety path.
    Input patch order is deterministic; an exact speed tie retains the first
    patch rather than introducing a second ordering rule.
    """

    selected = set(selected_link_ids)
    result: dict[str, Any] = {}
    maximum_speed: dict[str, float] = {}
    for kinematics in patch_kinematics:
        link_id = str(kinematics.robot_link_id)
        if link_id not in selected:
            continue
        speed = _norm(kinematics.tangential_velocity_world_mps)
        if link_id not in result or speed > maximum_speed[link_id]:
            result[link_id] = kinematics
            maximum_speed[link_id] = speed
    return result


def _contact_vector_telemetry_from_flat_buffers(
    *,
    body_identity: Sequence[str],
    selected_link_ids: set[str],
    contact_counts: Sequence[int],
    contact_starts: Sequence[int],
    normal_force_magnitudes_n: Sequence[float],
    contact_normals_world: Sequence[Sequence[float]],
    contact_points_world: Sequence[Sequence[float]],
    friction_counts: Sequence[int],
    friction_starts: Sequence[int],
    friction_forces_world: Sequence[Sequence[float]],
    contact_force_matrix_world: Sequence[Sequence[float]],
) -> _IsaacContactVectorTelemetry:
    """Aggregate privileged PhysX force vectors by selected robot link.

    Contact normals and friction forces come from separate PhysX buffers with
    independent active ranges.  This helper deliberately keeps those vectors
    separate: it is diagnostic evidence only and cannot become a contact-wrench
    command or a normal actor observation.
    """

    selected = set(selected_link_ids)
    zero = (0.0, 0.0, 0.0)
    normal_by_link = {link_id: zero for link_id in sorted(selected)}
    normal_point_by_link = {link_id: zero for link_id in sorted(selected)}
    friction_by_link = {link_id: zero for link_id in sorted(selected)}
    matrix_by_link = {link_id: zero for link_id in sorted(selected)}
    velocity_by_link = {link_id: zero for link_id in sorted(selected)}

    def finite_vector(value: Sequence[float]) -> bool:
        return len(value) == 3 and all(math.isfinite(float(item)) for item in value)

    pair_count = len(body_identity)
    valid = bool(
        selected
        and selected.issubset(set(body_identity))
        and len(contact_counts) == pair_count
        and len(contact_starts) == pair_count
        and len(friction_counts) == pair_count
        and len(friction_starts) == pair_count
        and len(contact_force_matrix_world) == pair_count
        and len(normal_force_magnitudes_n) == len(contact_normals_world)
        and len(normal_force_magnitudes_n) == len(contact_points_world)
        and all(finite_vector(value) for value in contact_normals_world)
        and all(finite_vector(value) for value in contact_points_world)
        and all(finite_vector(value) for value in friction_forces_world)
        and all(finite_vector(value) for value in contact_force_matrix_world)
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in (
                *contact_counts,
                *contact_starts,
                *friction_counts,
                *friction_starts,
            )
        )
        and all(math.isfinite(float(value)) for value in normal_force_magnitudes_n)
    )
    if not valid:
        return _IsaacContactVectorTelemetry(
            valid=False,
            normal_force_world_by_link=normal_by_link,
            normal_force_application_point_world_by_link=normal_point_by_link,
            friction_force_world_by_link=friction_by_link,
            contact_force_matrix_world_by_link=matrix_by_link,
            body_linear_velocity_world_by_link=velocity_by_link,
            body_contact_velocity_world_by_link=velocity_by_link,
            object_contact_velocity_world_by_link=velocity_by_link,
            relative_contact_velocity_world_by_link=velocity_by_link,
            tangential_slip_velocity_world_by_link=velocity_by_link,
            tangential_slip_contact_point_world_by_link=normal_point_by_link,
            tangential_slip_contact_normal_world_by_link=velocity_by_link,
        )

    for pair_index, link_id in enumerate(body_identity):
        if link_id not in selected:
            continue
        contact_start = int(contact_starts[pair_index])
        contact_stop = contact_start + int(contact_counts[pair_index])
        friction_start = int(friction_starts[pair_index])
        friction_stop = friction_start + int(friction_counts[pair_index])
        if (
            contact_stop > len(normal_force_magnitudes_n)
            or contact_stop > len(contact_normals_world)
            or friction_stop > len(friction_forces_world)
        ):
            return _IsaacContactVectorTelemetry(
                valid=False,
                normal_force_world_by_link=normal_by_link,
                normal_force_application_point_world_by_link=normal_point_by_link,
                friction_force_world_by_link=friction_by_link,
                contact_force_matrix_world_by_link=matrix_by_link,
                body_linear_velocity_world_by_link=velocity_by_link,
                body_contact_velocity_world_by_link=velocity_by_link,
                object_contact_velocity_world_by_link=velocity_by_link,
                relative_contact_velocity_world_by_link=velocity_by_link,
                tangential_slip_velocity_world_by_link=velocity_by_link,
                tangential_slip_contact_point_world_by_link=normal_point_by_link,
                tangential_slip_contact_normal_world_by_link=velocity_by_link,
            )
        normal_by_link[link_id] = tuple(
            sum(
                float(normal_force_magnitudes_n[index])
                * float(contact_normals_world[index][axis])
                for index in range(contact_start, contact_stop)
            )
            for axis in range(3)
        )
        force_sum = sum(
            abs(float(normal_force_magnitudes_n[index]))
            for index in range(contact_start, contact_stop)
        )
        if force_sum > 0.0:
            normal_point_by_link[link_id] = tuple(
                sum(
                    abs(float(normal_force_magnitudes_n[index]))
                    * float(contact_points_world[index][axis])
                    for index in range(contact_start, contact_stop)
                )
                / force_sum
                for axis in range(3)
            )
        friction_by_link[link_id] = tuple(
            sum(
                float(friction_forces_world[index][axis])
                for index in range(friction_start, friction_stop)
            )
            for axis in range(3)
        )
        matrix_by_link[link_id] = tuple(
            float(value) for value in contact_force_matrix_world[pair_index]
        )

    return _IsaacContactVectorTelemetry(
        valid=True,
        normal_force_world_by_link=normal_by_link,
        normal_force_application_point_world_by_link=normal_point_by_link,
        friction_force_world_by_link=friction_by_link,
        contact_force_matrix_world_by_link=matrix_by_link,
        body_linear_velocity_world_by_link=velocity_by_link,
        body_contact_velocity_world_by_link=velocity_by_link,
        object_contact_velocity_world_by_link=velocity_by_link,
        relative_contact_velocity_world_by_link=velocity_by_link,
        tangential_slip_velocity_world_by_link=velocity_by_link,
        tangential_slip_contact_point_world_by_link=normal_point_by_link,
        tangential_slip_contact_normal_world_by_link=velocity_by_link,
    )


def _contact_view_active(
    contact_view: Any, *, sim_dt: float, wp: Any, torch: Any, force_threshold_n: float
) -> bool:
    matrix = wp.to_torch(contact_view.get_contact_force_matrix(sim_dt)).reshape(-1, 3)
    return bool(
        torch.linalg.vector_norm(matrix, dim=-1).max().detach().cpu()
        >= force_threshold_n
    )


def _object_state(object_asset: Any) -> dict[str, list[float]]:
    pose_tensor = getattr(
        object_asset.data,
        "root_com_pose_w",
        object_asset.data.root_pose_w,
    )
    twist_tensor = getattr(
        object_asset.data,
        "root_com_vel_w",
        None,
    )
    return {
        "pose": _tensor_row(pose_tensor),
        "twist": (
            _tensor_row(twist_tensor)
            if twist_tensor is not None
            else _tensor_row(object_asset.data.root_lin_vel_w)
            + _tensor_row(object_asset.data.root_ang_vel_w)
        ),
    }


def _selected_gripper_mesh_local_aabbs(
    selected_surfaces: tuple[Any, Any],
    *,
    urdf_path: Any,
) -> tuple[_SelectedMeshLocalAABB, ...]:
    """Load selected collision meshes as link-frame local AABBs once per run."""

    from pathlib import Path

    from amsrr.feasibility.morphology_flight import (
        _ascii_stl_vertices,
        _binary_stl_vertices,
        _record_local_aabb,
        _resolve_mesh,
        _transform_aabb,
        _urdf_collision_records,
    )

    source_path = Path(urdf_path)
    records_by_link: dict[str, list[Any]] = {}
    for record in _urdf_collision_records(source_path):
        records_by_link.setdefault(record.link_id, []).append(record)

    result: list[_SelectedMeshLocalAABB] = []
    for surface in selected_surfaces:
        primitives = tuple(
            sorted(surface.collision_primitives, key=lambda item: item.primitive_id)
        )
        records = tuple(records_by_link.get(surface.mechanism_link_id, ()))
        if len(records) != len(primitives):
            raise SchemaValidationError(
                "selected Dock mesh collision count differs between URDF and PhysicalModel: "
                f"{surface.mechanism_link_id} ({len(records)} != {len(primitives)})"
            )
        for primitive, record in zip(primitives, records, strict=True):
            if (
                primitive.primitive_type not in {"mesh", "convex"}
                or record.geometry_type != "mesh"
                or primitive.geometry_ref is None
                or record.mesh_ref is None
                or _mesh_ref_basename(primitive.geometry_ref)
                != _mesh_ref_basename(record.mesh_ref)
            ):
                raise SchemaValidationError(
                    "selected Dock collision does not resolve to the expected URDF mesh: "
                    f"{primitive.primitive_id}"
                )
            geometry_bounds = _record_local_aabb(record, source_path, ())
            link_bounds = _transform_aabb(
                geometry_bounds,
                record.local_transform,
            )
            mesh_path = _resolve_mesh(
                record.mesh_ref,
                source_path.parent,
                (),
            )
            surface_sample_points_local = _stl_surface_samples_link_local(
                mesh_path,
                mesh_scale=record.mesh_scale,
                geometry_to_link=record.local_transform,
                binary_vertex_reader=_binary_stl_vertices,
                ascii_vertex_reader=_ascii_stl_vertices,
            )
            result.append(
                _SelectedMeshLocalAABB(
                    module_id=int(surface.module_id),
                    link_id=str(surface.mechanism_link_id),
                    primitive_id=str(primitive.primitive_id),
                    geometry_ref=str(record.mesh_ref),
                    minimum_local=tuple(float(value) for value in link_bounds.lower),
                    maximum_local=tuple(float(value) for value in link_bounds.upper),
                    surface_sample_points_local=(surface_sample_points_local),
                )
            )
    if not result:
        raise SchemaValidationError(
            "selected gripper surfaces contain no URDF mesh AABBs"
        )
    return tuple(result)


def _selected_gripper_cone_proxy_pad_specs(
    selected_surfaces: Sequence[Any],
    *,
    urdf_path: str | Path,
) -> tuple[_Order8DiagnosticConeProxyPadSpec, ...]:
    """Bind the visually approved cone micro-pad geometry to selected modules."""

    surfaces = tuple(selected_surfaces)
    if len(surfaces) != 2 or len(
        {(int(surface.module_id), str(surface.mechanism_link_id)) for surface in surfaces}
    ) != 2:
        raise SchemaValidationError(
            "Order8 cone proxy pads require exactly two distinct selected rigid links"
        )
    geometry_config = load_order8_side_proxy_pad_preview_config(
        ORDER8_DIAGNOSTIC_CONE_PROXY_PAD_CONFIG_PATH
    )
    if (
        geometry_config.acceptance_eligible
        or not geometry_config.visual_approval_recorded
        or not geometry_config.contact_runtime_enabled
    ):
        raise SchemaValidationError(
            "Order8 cone proxy-pad geometry lacks diagnostic runtime approval"
        )
    geometry_specs = build_order8_side_proxy_pad_specs(
        urdf_path=urdf_path,
        config=geometry_config,
    )
    geometry_by_link: dict[str, list[Any]] = {}
    for geometry_spec in geometry_specs:
        geometry_by_link.setdefault(str(geometry_spec.link_id), []).append(
            geometry_spec
        )
    result: list[_Order8DiagnosticConeProxyPadSpec] = []
    for surface in surfaces:
        link_id = str(surface.mechanism_link_id)
        selected_geometry = geometry_by_link.get(link_id, [])
        if not selected_geometry:
            raise SchemaValidationError(
                "Order8 cone proxy-pad geometry has no approved tiles for "
                f"module_{surface.module_id}:{link_id}"
            )
        for geometry_spec in selected_geometry:
            result.append(
                _Order8DiagnosticConeProxyPadSpec(
                    module_id=int(surface.module_id),
                    link_id=link_id,
                    pad_id=str(geometry_spec.pad_id),
                    center_local=tuple(geometry_spec.center_local),
                    representative_surface_point_local=tuple(
                        geometry_spec.representative_surface_point_local
                    ),
                    orientation_local_xyzw=tuple(
                        geometry_spec.orientation_local_xyzw
                    ),
                    size_m=tuple(geometry_spec.size_m),
                    outward_normal_local=tuple(
                        geometry_spec.outward_normal_local
                    ),
                    axial_band_index=int(geometry_spec.axial_band_index),
                    circumferential_segment_index=int(
                        geometry_spec.circumferential_segment_index
                    ),
                    inner_face_surface_gap_m=float(
                        geometry_spec.inner_face_surface_gap_m
                    ),
                    surface_fit_max_gap_m=float(
                        geometry_spec.surface_fit_max_gap_m
                    ),
                    source_geometry_refs=tuple(geometry_spec.geometry_refs),
                )
            )
    return tuple(
        sorted(
            result,
            key=lambda item: (
                item.module_id,
                item.link_id,
                item.axial_band_index,
                item.circumferential_segment_index,
            ),
        )
    )


def _cone_proxy_pad_surface_local_meshes(
    specs: Sequence[_Order8DiagnosticConeProxyPadSpec],
) -> tuple[_SelectedMeshLocalAABB, ...]:
    """Expose the active proxy-box surfaces to the non-contact q_close gate.

    Cone proxy mode disables the authored selected-body collision meshes.  Its
    geometric stop detector must therefore measure the same physical boxes,
    otherwise an already contacting pad can still look several millimetres
    clear and the integrated position target continues closing.  The samples
    below are derived solely from known collider geometry (eight corners and
    six face centres); no Isaac contact truth enters the control path.
    """

    from itertools import product

    from amsrr.geometry.pose_math import add3, matvec, transform_from_pose

    result: list[_SelectedMeshLocalAABB] = []
    for spec in specs:
        if any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in spec.size_m
        ):
            raise SchemaValidationError(
                f"Order8 cone proxy pad has invalid size: {spec.pad_id}"
            )
        pad_from_local = transform_from_pose(
            (*spec.center_local, *spec.orientation_local_xyzw)
        )
        half = tuple(0.5 * float(value) for value in spec.size_m)
        points_pad = [
            tuple(sign[axis] * half[axis] for axis in range(3))
            for sign in product((-1.0, 1.0), repeat=3)
        ]
        for axis in range(3):
            for sign in (-1.0, 1.0):
                point = [0.0, 0.0, 0.0]
                point[axis] = sign * half[axis]
                points_pad.append(tuple(point))
        points_local = tuple(
            tuple(
                float(value)
                for value in add3(
                    pad_from_local.translation,
                    matvec(pad_from_local.rotation, point_pad),
                )
            )
            for point_pad in points_pad
        )
        result.append(
            _SelectedMeshLocalAABB(
                module_id=int(spec.module_id),
                link_id=str(spec.link_id),
                primitive_id=f"diagnostic_cone_proxy:{spec.pad_id}",
                geometry_ref="diagnostic_cone_proxy_box",
                minimum_local=tuple(
                    min(point[axis] for point in points_local)
                    for axis in range(3)
                ),
                maximum_local=tuple(
                    max(point[axis] for point in points_local)
                    for axis in range(3)
                ),
                surface_sample_points_local=points_local,
            )
        )
    if not result:
        raise SchemaValidationError(
            "Order8 cone proxy geometric gate requires at least one pad"
        )
    return tuple(result)


def _selected_gripper_proxy_pad_specs(
    selected_surfaces: Sequence[Any],
    local_meshes: Sequence[_SelectedMeshLocalAABB],
    physical_model: Any,
) -> tuple[_Order8DiagnosticProxyPadSpec, ...]:
    """Fit one deterministic finite pad to each selected mesh outer face.

    The pad frame is the authored connect-frame orientation expressed in the
    selected mechanism link.  Its inner face is kept 1 mm beyond the sampled
    mesh maximum and its 2 mm thickness places the contact face 3 mm beyond
    that maximum.  Since the Order-8 penetration ceiling is 2 mm, an otherwise
    valid diagnostic contact cannot also reach the retained authored mesh.
    """

    surfaces = tuple(selected_surfaces)
    meshes = tuple(local_meshes)
    if len(surfaces) != 2 or len(
        {(int(surface.module_id), str(surface.mechanism_link_id)) for surface in surfaces}
    ) != 2:
        raise SchemaValidationError(
            "Order8 proxy pads require exactly two distinct selected rigid links"
        )
    joints_by_id = {str(joint.joint_id): joint for joint in physical_model.joints}
    if len(joints_by_id) != len(physical_model.joints):
        raise SchemaValidationError(
            "Order8 proxy-pad construction requires unique PhysicalModel joint ids"
        )
    specs: list[_Order8DiagnosticProxyPadSpec] = []
    for surface in surfaces:
        connect_joint = joints_by_id.get(str(surface.port_local_id))
        if connect_joint is None or str(connect_joint.parent_link) != str(
            surface.mechanism_link_id
        ):
            raise SchemaValidationError(
                "Order8 selected surface does not resolve to its authored "
                f"connect-frame joint: module_{surface.module_id}:"
                f"{surface.port_local_id}"
            )
        connect_pose_local = pose_from_transform(
            transform_from_xyz_rpy(
                connect_joint.origin_xyz,
                connect_joint.origin_rpy,
            )
        )
        connect_rotation_local = transform_from_pose(connect_pose_local).rotation
        axes_local = tuple(
            tuple(float(value) for value in matvec(connect_rotation_local, axis))
            for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        )
        samples = tuple(
            point
            for mesh in meshes
            if int(mesh.module_id) == int(surface.module_id)
            and str(mesh.link_id) == str(surface.mechanism_link_id)
            for point in mesh.surface_sample_points_local
        )
        if not samples:
            raise SchemaValidationError(
                "Order8 proxy-pad selected mesh contains no surface samples: "
                f"module_{surface.module_id}:{surface.mechanism_link_id}"
            )
        projections = tuple(
            tuple(
                sum(float(point[index]) * float(axis[index]) for index in range(3))
                for point in samples
            )
            for axis in axes_local
        )
        mesh_surface_projection = max(projections[0])
        near_indices = tuple(
            index
            for index, value in enumerate(projections[0])
            if mesh_surface_projection - float(value)
            <= ORDER8_DIAGNOSTIC_PROXY_PAD_SURFACE_BAND_M + 1.0e-12
        )
        if not near_indices:
            raise SchemaValidationError(
                "Order8 proxy-pad outer-face sampling produced no candidates"
            )
        tangential_ranges = tuple(
            (
                min(projections[axis_index][index] for index in near_indices),
                max(projections[axis_index][index] for index in near_indices),
            )
            for axis_index in (1, 2)
        )
        tangential_spans = tuple(
            float(upper) - float(lower) for lower, upper in tangential_ranges
        )
        if any(
            span + 1.0e-12 < ORDER8_DIAGNOSTIC_PROXY_PAD_TANGENTIAL_SIZE_M
            for span in tangential_spans
        ):
            raise SchemaValidationError(
                "Order8 proxy pad does not fit inside the sampled outer-face "
                f"region: spans={tangential_spans}"
            )
        center_coordinates = (
            mesh_surface_projection
            + ORDER8_DIAGNOSTIC_PROXY_PAD_MESH_CLEARANCE_M
            + 0.5 * ORDER8_DIAGNOSTIC_PROXY_PAD_THICKNESS_M,
            0.5 * (tangential_ranges[0][0] + tangential_ranges[0][1]),
            0.5 * (tangential_ranges[1][0] + tangential_ranges[1][1]),
        )
        center_local = tuple(
            sum(
                float(center_coordinates[axis_index])
                * float(axes_local[axis_index][coordinate_index])
                for axis_index in range(3)
            )
            for coordinate_index in range(3)
        )
        specs.append(
            _Order8DiagnosticProxyPadSpec(
                module_id=int(surface.module_id),
                link_id=str(surface.mechanism_link_id),
                center_local=center_local,
                orientation_local_xyzw=tuple(
                    float(value) for value in connect_pose_local[3:7]
                ),
                size_m=(
                    ORDER8_DIAGNOSTIC_PROXY_PAD_THICKNESS_M,
                    ORDER8_DIAGNOSTIC_PROXY_PAD_TANGENTIAL_SIZE_M,
                    ORDER8_DIAGNOSTIC_PROXY_PAD_TANGENTIAL_SIZE_M,
                ),
                mesh_surface_projection_m=float(mesh_surface_projection),
                inner_face_projection_m=(
                    float(mesh_surface_projection)
                    + ORDER8_DIAGNOSTIC_PROXY_PAD_MESH_CLEARANCE_M
                ),
                outer_face_projection_m=(
                    float(mesh_surface_projection)
                    + ORDER8_DIAGNOSTIC_PROXY_PAD_MESH_CLEARANCE_M
                    + ORDER8_DIAGNOSTIC_PROXY_PAD_THICKNESS_M
                ),
                tangential_surface_span_m=tuple(tangential_spans),
                surface_sample_count=len(samples),
                near_surface_sample_count=len(near_indices),
            )
        )
    return tuple(sorted(specs, key=lambda item: (item.module_id, item.link_id)))


def _stl_surface_samples_link_local(
    mesh_path: Any,
    *,
    mesh_scale: tuple[float, float, float],
    geometry_to_link: Any,
    binary_vertex_reader: Any,
    ascii_vertex_reader: Any,
    maximum_sample_count: int = 4096,
) -> tuple[tuple[float, float, float], ...]:
    """Return deterministic triangle samples in the collision link frame.

    A mesh AABB includes holes and concavities and therefore cannot arm a
    contact-stop detector.  Vertices plus triangle centroids preserve actual
    surface occupancy while keeping the 200 Hz distance query bounded.
    """

    import struct

    from amsrr.geometry.pose_math import add3, matvec

    data = mesh_path.read_bytes()
    if len(data) >= 84:
        triangle_count = struct.unpack_from("<I", data, 80)[0]
        expected_size = 84 + triangle_count * 50
    else:
        triangle_count = 0
        expected_size = -1
    if expected_size == len(data):
        raw_vertices = tuple(binary_vertex_reader(data, triangle_count))
    else:
        raw_vertices = tuple(ascii_vertex_reader(data, mesh_path))
    if len(raw_vertices) < 3 or len(raw_vertices) % 3 != 0:
        raise SchemaValidationError(
            f"selected Dock STL has incomplete triangles: {mesh_path}"
        )
    triangle_count = len(raw_vertices) // 3
    sampled_triangle_count = max(1, min(triangle_count, maximum_sample_count // 4))
    triangle_indices = tuple(
        min(
            triangle_count - 1,
            (sample_index * triangle_count) // sampled_triangle_count,
        )
        for sample_index in range(sampled_triangle_count)
    )
    points: list[tuple[float, float, float]] = []
    for triangle_index in triangle_indices:
        triangle = tuple(
            tuple(
                float(raw_vertices[3 * triangle_index + vertex_index][axis])
                * float(mesh_scale[axis])
                for axis in range(3)
            )
            for vertex_index in range(3)
        )
        centroid = tuple(
            sum(vertex[axis] for vertex in triangle) / 3.0 for axis in range(3)
        )
        for point in (*triangle, centroid):
            transformed = add3(
                geometry_to_link.translation,
                matvec(geometry_to_link.rotation, point),
            )
            if not all(math.isfinite(float(value)) for value in transformed):
                raise SchemaValidationError(
                    f"selected Dock STL contains a non-finite sample: {mesh_path}"
                )
            points.append(tuple(float(value) for value in transformed))
    if not points:
        raise SchemaValidationError(
            f"selected Dock STL contains no surface samples: {mesh_path}"
        )
    return tuple(points)


def _mesh_ref_basename(mesh_ref: str) -> str:
    normalized = str(mesh_ref).replace("\\", "/").split("?", 1)[0]
    return normalized.rsplit("/", 1)[-1].lower()


def _gripper_object_clearance_from_body_poses(
    local_aabbs: tuple[_SelectedMeshLocalAABB, ...],
    body_pose_by_module_link: dict[tuple[int, str], Pose7D],
    object_pose: Pose7D,
    object_size: list[float],
) -> float:
    from amsrr.simulation.order8_contact_measurement import (
        gripper_object_clearance_m,
    )

    return gripper_object_clearance_m(
        gripper_aabbs_world=_selected_mesh_world_aabbs(
            local_aabbs,
            body_pose_by_module_link,
        ),
        object_pose_world=object_pose,
        object_size_m=object_size,
    )


def _gripper_object_surface_sample_clearance_from_body_poses(
    local_meshes: tuple[_SelectedMeshLocalAABB, ...],
    body_pose_by_module_link: dict[tuple[int, str], Pose7D],
    object_pose: Pose7D,
    object_size: list[float],
) -> float:
    return _gripper_object_surface_sample_query_from_body_poses(
        local_meshes,
        body_pose_by_module_link,
        object_pose,
        object_size,
    )[0]


def _gripper_object_surface_sample_query_from_body_poses(
    local_meshes: tuple[_SelectedMeshLocalAABB, ...],
    body_pose_by_module_link: dict[tuple[int, str], Pose7D],
    object_pose: Pose7D,
    object_size: list[float],
) -> tuple[
    float,
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """Approximate true mesh/box separation without simulator contact truth.

    Selected URDF collision-mesh surface samples are transformed into the
    observed object's frame and measured against its oriented box.  A value of
    zero means an actual sampled mesh surface has reached the box volume; holes
    inside a mesh AABB do not arm the gate.  The returned world point is the
    deterministic mesh sample nearest the object surface.  Final closure uses
    it as the differential-IK translation point, and planned contact wrench
    mapping uses the same point so the Jacobian and wrench are expressed
    consistently.  The final item is the observed box's outward surface normal
    at that sample and supports a post-``q_close`` normal-motion gate without
    simulator contact normals.
    """

    import numpy as np

    from amsrr.geometry.pose_math import transform_from_pose

    if len(object_size) != 3 or any(
        not math.isfinite(float(value)) or float(value) <= 0.0 for value in object_size
    ):
        raise SchemaValidationError(
            "Order8 object_size must contain three finite positive values"
        )
    object_from_world = transform_from_pose(inverse_pose(object_pose))
    half_extent = 0.5 * np.asarray(object_size, dtype=float)
    minimum = math.inf
    selected_point_world: tuple[float, float, float] | None = None
    selected_point_object: tuple[float, float, float] | None = None
    for mesh in local_meshes:
        if not mesh.surface_sample_points_local:
            raise SchemaValidationError(
                "selected Dock mesh lacks surface samples: " f"{mesh.primitive_id}"
            )
        key = (mesh.module_id, mesh.link_id)
        body_pose = body_pose_by_module_link.get(key)
        if body_pose is None:
            raise SchemaValidationError(
                f"missing world pose for selected Dock body module_{key[0]}:{key[1]}"
            )
        body_from_link = transform_from_pose(body_pose)
        points_link = np.asarray(mesh.surface_sample_points_local, dtype=float)
        rotation_body = np.asarray(body_from_link.rotation, dtype=float)
        translation_body = np.asarray(body_from_link.translation, dtype=float)
        points_world = points_link @ rotation_body.T + translation_body
        rotation_object = np.asarray(object_from_world.rotation, dtype=float)
        translation_object = np.asarray(object_from_world.translation, dtype=float)
        points_object = points_world @ rotation_object.T + translation_object
        outside = np.maximum(np.abs(points_object) - half_extent, 0.0)
        distances = np.linalg.norm(outside, axis=1)
        inside = np.all(np.abs(points_object) <= half_extent, axis=1)
        if bool(np.any(inside)):
            # Multiple authored surface samples can lie slightly inside the
            # box after contact.  Choose the least-penetrating one, rather than
            # the STL-order-dependent first zero-distance sample.
            inside_margin = np.min(
                half_extent[None, :] - np.abs(points_object),
                axis=1,
            )
            signed_distance = np.where(inside, -inside_margin, -math.inf)
            point_index = int(np.argmax(signed_distance))
            mesh_minimum = 0.0
        else:
            point_index = int(np.argmin(distances))
            mesh_minimum = float(distances[point_index])
        if mesh_minimum < minimum:
            minimum = mesh_minimum
            selected_point_world = tuple(
                float(value) for value in points_world[point_index]
            )
            selected_point_object = tuple(
                float(value) for value in points_object[point_index]
            )
    if (
        not math.isfinite(minimum)
        or selected_point_world is None
        or selected_point_object is None
    ):
        raise SchemaValidationError(
            "selected gripper surfaces contain no finite mesh samples"
        )
    point_object = np.asarray(selected_point_object, dtype=float)
    outside = np.maximum(np.abs(point_object) - half_extent, 0.0)
    if float(np.linalg.norm(outside)) > 1.0e-12:
        # At an edge/corner, the closest-point vector is the unambiguous
        # geometric normal of the box distance field.
        normal_object = np.sign(point_object) * outside
        normal_object /= np.linalg.norm(normal_object)
    else:
        # For a sample on or slightly inside the OBB, use its nearest face.
        # ``argmin`` provides deterministic axis selection for an exact tie.
        face_axis = int(np.argmin(half_extent - np.abs(point_object)))
        normal_object = np.zeros(3, dtype=float)
        normal_object[face_axis] = 1.0 if point_object[face_axis] >= 0.0 else -1.0
    world_from_object = transform_from_pose(object_pose)
    normal_world_array = (
        np.asarray(world_from_object.rotation, dtype=float) @ normal_object
    )
    normal_world_array /= np.linalg.norm(normal_world_array)
    normal_world = tuple(float(value) for value in normal_world_array)
    return max(0.0, minimum), selected_point_world, normal_world


def _selected_mesh_world_aabbs(
    local_aabbs: tuple[_SelectedMeshLocalAABB, ...],
    body_pose_by_module_link: dict[tuple[int, str], Pose7D],
) -> tuple[Any, ...]:
    """Resolve selected URDF collision-mesh bounds at measured body poses."""

    from amsrr.simulation.order8_contact_measurement import (
        oriented_cuboid_world_aabb,
    )

    world_aabbs: list[Any] = []
    for bounds in local_aabbs:
        key = (bounds.module_id, bounds.link_id)
        body_pose = body_pose_by_module_link.get(key)
        if body_pose is None:
            raise SchemaValidationError(
                f"missing world pose for selected Dock body module_{key[0]}:{key[1]}"
            )
        center_local = tuple(
            0.5 * (bounds.minimum_local[index] + bounds.maximum_local[index])
            for index in range(3)
        )
        size_local = tuple(
            bounds.maximum_local[index] - bounds.minimum_local[index]
            for index in range(3)
        )
        center_world = compose_pose(
            body_pose,
            (*center_local, 0.0, 0.0, 0.0, 1.0),
        )
        world_aabbs.append(
            oriented_cuboid_world_aabb(
                pose_world=center_world,
                size_m=size_local,
            )
        )
    if not world_aabbs:
        raise SchemaValidationError(
            "selected gripper surfaces contain no world mesh AABBs"
        )
    return tuple(world_aabbs)


def _minimum_gripper_object_axial_overlap_from_body_poses(
    local_aabbs: tuple[_SelectedMeshLocalAABB, ...],
    body_pose_by_module_link: dict[tuple[int, str], Pose7D],
    object_pose: Pose7D,
    object_size: list[float],
    *,
    axis_world: tuple[float, float, float],
) -> float:
    """Return the least selected-mesh/object overlap along an approach axis.

    Positive values mean every selected mesh AABB has entered the object's
    projected interval by that distance.  Negative values retain the axial
    separation, so this is a direct insertion gate rather than a root-tracking
    proxy.  Raw contact truth is deliberately not an input.
    """

    from amsrr.simulation.order8_contact_measurement import (
        oriented_cuboid_world_aabb,
    )

    axis = _unit(axis_world)
    object_bounds = oriented_cuboid_world_aabb(
        pose_world=object_pose,
        size_m=object_size,
    )
    object_interval = _aabb_projection_interval(object_bounds, axis)
    overlaps = []
    for bounds in _selected_mesh_world_aabbs(
        local_aabbs,
        body_pose_by_module_link,
    ):
        mesh_interval = _aabb_projection_interval(bounds, axis)
        overlaps.append(
            min(mesh_interval[1], object_interval[1])
            - max(mesh_interval[0], object_interval[0])
        )
    return min(overlaps)


def _aabb_projection_interval(
    bounds: Any,
    axis: tuple[float, float, float],
) -> tuple[float, float]:
    minimum = sum(
        float(axis[index])
        * float(
            bounds.minimum_world[index]
            if axis[index] >= 0.0
            else bounds.maximum_world[index]
        )
        for index in range(3)
    )
    maximum = sum(
        float(axis[index])
        * float(
            bounds.maximum_world[index]
            if axis[index] >= 0.0
            else bounds.minimum_world[index]
        )
        for index in range(3)
    )
    return minimum, maximum


def _mesh_aware_staging_plan(
    local_aabbs: tuple[_SelectedMeshLocalAABB, ...],
    body_pose_by_module_link_at_grasp: dict[tuple[int, str], Pose7D],
    *,
    grasp_base_pose: Pose7D,
    object_pose: Pose7D,
    object_size: list[float],
    required_clearance_m: float,
    maximum_retreat_m: float,
) -> _MeshAwareStagingPlan:
    """Retreat the neutral morphology until every selected mesh is clear.

    Connect frames do not encode protruding collision geometry.  This solver
    therefore translates the actual selected URDF mesh AABBs opposite the
    grasp-frame +x approach axis and finds the minimum deterministic retreat
    satisfying the configured object clearance.
    """

    from amsrr.geometry.pose_math import matvec, transform_from_pose

    for name, value in (
        ("required_clearance_m", required_clearance_m),
        ("maximum_retreat_m", maximum_retreat_m),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise SchemaValidationError(
                f"Order8 mesh-aware staging {name} must be finite and positive"
            )
    approach_axis = _unit(
        matvec(
            transform_from_pose(grasp_base_pose).rotation,
            (1.0, 0.0, 0.0),
        )
    )

    def body_poses_at_retreat(
        distance_m: float,
    ) -> dict[tuple[int, str], Pose7D]:
        offset = tuple(-float(distance_m) * value for value in approach_axis)
        return {
            key: _offset_pose(
                pose,
                dx=offset[0],
                dy=offset[1],
                dz=offset[2],
            )
            for key, pose in body_pose_by_module_link_at_grasp.items()
        }

    def clearance(distance_m: float) -> float:
        return _gripper_object_clearance_from_body_poses(
            local_aabbs,
            body_poses_at_retreat(distance_m),
            object_pose,
            object_size,
        )

    target = float(required_clearance_m)
    initial_clearance = clearance(0.0)
    if initial_clearance >= target:
        retreat = 0.0
        predicted = initial_clearance
    else:
        upper = float(maximum_retreat_m)
        maximum_clearance = clearance(upper)
        if maximum_clearance < target:
            raise SchemaValidationError(
                "Order8 selected Dock meshes cannot reach the configured "
                "pregrasp clearance within the initial object standoff"
            )
        lower = 0.0
        for _ in range(64):
            midpoint = 0.5 * (lower + upper)
            if clearance(midpoint) >= target:
                upper = midpoint
            else:
                lower = midpoint
        retreat = upper
        predicted = clearance(retreat)

    translation = tuple(-retreat * value for value in approach_axis)
    staging_pose = _offset_pose(
        grasp_base_pose,
        dx=translation[0],
        dy=translation[1],
        dz=translation[2],
    )
    return _MeshAwareStagingPlan(
        base_pose_world=staging_pose,
        retreat_distance_m=retreat,
        predicted_clearance_m=predicted,
        approach_axis_world=approach_axis,
    )


def _mesh_aware_anchor_opening_plan(
    local_aabbs: tuple[_SelectedMeshLocalAABB, ...],
    body_pose_by_module_link_at_grasp: dict[tuple[int, str], Pose7D],
    *,
    anchor_id_by_module_link: dict[tuple[int, str], int],
    anchor_pose_world_by_id: dict[int, Pose7D],
    inward_normal_world_by_anchor: dict[int, tuple[float, float, float]],
    grasp_base_pose: Pose7D,
    object_pose: Pose7D,
    object_size: list[float],
    required_clearance_m: float,
    maximum_opening_m: float,
) -> _MeshAwareAnchorOpeningPlan:
    """Open each selected mesh outward until it clears the object AABB."""

    for name, value in (
        ("required_clearance_m", required_clearance_m),
        ("maximum_opening_m", maximum_opening_m),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise SchemaValidationError(
                f"Order8 mesh-aware opening {name} must be finite and positive"
            )
    bounds_by_key: dict[tuple[int, str], list[_SelectedMeshLocalAABB]] = {}
    for bounds in local_aabbs:
        bounds_by_key.setdefault((bounds.module_id, bounds.link_id), []).append(bounds)
    if set(bounds_by_key) != set(anchor_id_by_module_link):
        raise SchemaValidationError(
            "Order8 mesh-aware opening body/anchor coverage mismatch"
        )

    anchor_poses_base: dict[int, Pose7D] = {}
    distances: dict[int, float] = {}
    clearances: dict[int, float] = {}
    grasp_from_world = inverse_pose(grasp_base_pose)
    for key in sorted(bounds_by_key):
        anchor_id = anchor_id_by_module_link[key]
        body_pose = body_pose_by_module_link_at_grasp.get(key)
        anchor_pose = anchor_pose_world_by_id.get(anchor_id)
        inward = inward_normal_world_by_anchor.get(anchor_id)
        if body_pose is None or anchor_pose is None or inward is None:
            raise SchemaValidationError(
                "Order8 mesh-aware opening is missing a selected body, anchor, or normal"
            )
        outward = tuple(-value for value in _unit(inward))
        selected_bounds = tuple(bounds_by_key[key])

        def clearance(distance_m: float) -> float:
            offset = tuple(float(distance_m) * value for value in outward)
            shifted_body = _offset_pose(
                body_pose,
                dx=offset[0],
                dy=offset[1],
                dz=offset[2],
            )
            return _gripper_object_clearance_from_body_poses(
                selected_bounds,
                {key: shifted_body},
                object_pose,
                object_size,
            )

        target = float(required_clearance_m)
        if clearance(0.0) >= target:
            distance = 0.0
        else:
            lower = 0.0
            upper = float(maximum_opening_m)
            if clearance(upper) < target:
                raise SchemaValidationError(
                    "Order8 selected Dock mesh cannot reach the configured "
                    "pregrasp opening clearance"
                )
            for _ in range(64):
                midpoint = 0.5 * (lower + upper)
                if clearance(midpoint) >= target:
                    upper = midpoint
                else:
                    lower = midpoint
            distance = upper
        predicted = clearance(distance)
        offset = tuple(distance * value for value in outward)
        opened_anchor_world = _offset_pose(
            anchor_pose,
            dx=offset[0],
            dy=offset[1],
            dz=offset[2],
        )
        anchor_poses_base[anchor_id] = compose_pose(
            grasp_from_world,
            opened_anchor_world,
        )
        distances[anchor_id] = distance
        clearances[anchor_id] = predicted

    return _MeshAwareAnchorOpeningPlan(
        anchor_poses_base=anchor_poses_base,
        outward_distance_m_by_anchor=distances,
        predicted_clearance_m_by_anchor=clearances,
    )


def _gripper_object_clearance(
    local_aabbs: tuple[_SelectedMeshLocalAABB, ...],
    robots: dict[int, Any],
    object_pose: Pose7D,
    object_size: list[float],
) -> float:
    return _gripper_object_clearance_from_body_poses(
        local_aabbs,
        _selected_gripper_body_poses(local_aabbs, robots),
        object_pose,
        object_size,
    )


def _selected_gripper_body_poses(
    local_aabbs: tuple[_SelectedMeshLocalAABB, ...],
    robots: dict[int, Any],
) -> dict[tuple[int, str], Pose7D]:
    body_poses: dict[tuple[int, str], Pose7D] = {}
    for bounds in local_aabbs:
        key = (bounds.module_id, bounds.link_id)
        if key in body_poses:
            continue
        robot = robots[bounds.module_id]
        body_name = _resolve_name(robot.body_names, bounds.link_id)
        if body_name is None:
            raise RuntimeError(
                f"Order8 cannot resolve selected Dock body module_{bounds.module_id}:"
                f"{bounds.link_id}"
            )
        body_index = robot.body_names.index(body_name)
        body_poses[key] = tuple(
            _tensor_body_row(robot.data.body_pos_w, body_index)
            + _tensor_body_row(robot.data.body_quat_w, body_index)
        )
    return body_poses


def _mean_root_pose(robots: dict[int, Any]) -> Pose7D:
    poses = [_tensor_row(robot.data.root_pose_w.torch) for robot in robots.values()]
    position = tuple(
        sum(pose[index] for pose in poses) / len(poses) for index in range(3)
    )
    return (*position, *poses[0][3:7])


def _tangent_basis(normal: Any) -> list[float]:
    n = tuple(float(value) for value in normal)
    reference = (0.0, 0.0, 1.0) if abs(n[2]) < 0.9 else (0.0, 1.0, 0.0)
    first = _unit(_cross(reference, n))
    second = _unit(_cross(n, first))
    return [*first, *second]


def _offset_pose(
    pose: Pose7D, *, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0
) -> Pose7D:
    return (float(pose[0]) + dx, float(pose[1]) + dy, float(pose[2]) + dz, *pose[3:7])


def _interpolate_pose(start: Pose7D, target: Pose7D, alpha: float) -> Pose7D:
    value = min(max(float(alpha), 0.0), 1.0)
    return tuple(
        (
            (1.0 - value) * float(start[index]) + value * float(target[index])
            if index < 3
            else float(target[index])
        )
        for index in range(7)
    )  # type: ignore[return-value]


def _advance_pose_toward(
    current: Pose7D,
    target: Pose7D,
    *,
    max_translation_step_m: float,
) -> Pose7D:
    """Rate-limit a deterministic centroidal reference without using contact truth."""

    if not math.isfinite(max_translation_step_m) or max_translation_step_m <= 0.0:
        raise SchemaValidationError(
            "Order8 max_translation_step_m must be finite and positive"
        )
    distance = _position_distance(current, target)
    if distance <= max_translation_step_m:
        return tuple(float(value) for value in target)  # type: ignore[return-value]
    return _interpolate_pose(
        current,
        target,
        max_translation_step_m / max(distance, 1.0e-12),
    )


def _alternating_reacquire_anchor_target(
    *,
    previous_command: Pose7D,
    terminal_target: Pose7D,
    individual_latched_pose: Pose7D | None,
    reacquired_hold_pose: Pose7D | None,
    all_individual_latches_acquired: bool,
    max_translation_step_m: float,
) -> Pose7D:
    """Hold in-band measured poses while only the out-of-band side closes."""

    if individual_latched_pose is not None and not all_individual_latches_acquired:
        # Before both sides have individually arrested, retain the first side
        # at the pose measured when its arrest was established.
        return individual_latched_pose
    if all_individual_latches_acquired and reacquired_hold_pose is not None:
        # Reusing an earlier individual-arrest pose after the base has settled
        # is stale, while replacing the target with the current pose every step
        # cannot restore coupled drift.  Hold the one-time in-band snapshot.
        return reacquired_hold_pose
    return _advance_pose_toward(
        previous_command,
        terminal_target,
        max_translation_step_m=max_translation_step_m,
    )


def _sequential_reacquire_anchor_tasks(
    tasks: Sequence[Any],
    *,
    pursued_anchor_id: int | None,
) -> list[Any]:
    """Keep every Dock variable while activating only the pursued anchor task.

    The two opposing end-effector pose tasks are generally not independently
    realizable once one branch is held at a world-fixed pose and the centroidal
    frame is held quasi-statically.  Sequential reacquisition therefore removes
    only the inactive *task rows*; it never masks or removes a Dock joint column.
    """

    ordered = list(tasks)
    if pursued_anchor_id is None:
        return ordered
    selected = [
        task
        for task in ordered
        if int(getattr(task, "anchor_id")) == int(pursued_anchor_id)
    ]
    if len(selected) != 1:
        raise SchemaValidationError(
            "Order8 sequential reacquire pursued anchor must select exactly " "one task"
        )
    return selected


def _sequential_latched_anchor_hold_tasks(
    tasks: Sequence[Any],
    *,
    latched_anchor_ids: set[int],
) -> list[Any]:
    """Activate only the world-held anchor while the base moves the peer."""

    if len(latched_anchor_ids) != 1:
        raise SchemaValidationError(
            "Order8 sequential latched transfer requires exactly one held anchor"
        )
    held_anchor_id = next(iter(latched_anchor_ids))
    selected = [
        task for task in tasks if int(getattr(task, "anchor_id")) == held_anchor_id
    ]
    if len(selected) != 1:
        raise SchemaValidationError(
            "Order8 sequential latched transfer must select exactly one task"
        )
    return selected


def _contact_anchor_pose_priority(
    *,
    phase: Order8NaturalContactPhase,
    contact_configuration_latched: bool,
    anchor_individually_latched: bool,
    all_individual_latches_acquired: bool,
    anchor_reacquired: bool,
    all_reacquired_holds_acquired: bool,
) -> float:
    """Keep an established contact safety-critical through reacquisition."""

    if (
        phase == Order8NaturalContactPhase.CONTACT_ACQUISITION
        and not contact_configuration_latched
        and anchor_individually_latched
        and not all_individual_latches_acquired
    ):
        return 1.0
    if (
        phase == Order8NaturalContactPhase.CONTACT_ACQUISITION
        and not contact_configuration_latched
        and all_individual_latches_acquired
        and anchor_reacquired
        and not all_reacquired_holds_acquired
    ):
        return 0.5
    return 1.0


def _contact_force_scale_for_phase(
    *,
    phase: Order8NaturalContactPhase,
    ramp_elapsed_s: float,
    ramp_duration_s: float,
) -> float:
    """Ramp during acquisition, hold through carriage, release deliberately."""

    if ramp_duration_s <= 0.0:
        raise ValueError("contact force ramp duration must be positive")
    if phase == Order8NaturalContactPhase.CONTACT_ACQUISITION:
        return min(max(float(ramp_elapsed_s) / float(ramp_duration_s), 0.0), 1.0)
    if phase in {
        Order8NaturalContactPhase.LIFT,
        Order8NaturalContactPhase.TRANSPORT,
        Order8NaturalContactPhase.PLACE,
    }:
        return 1.0
    return 0.0


def _first_order_low_pass(
    previous_value: float,
    sample_value: float,
    *,
    dt_s: float,
    time_constant_s: float,
) -> float:
    """Deterministic first-order filter for signed kinematic velocity.

    Contact-solver micro-oscillation reverses sign at the physics rate.  The
    nominal settle gate therefore filters the signed normal component before
    taking its magnitude; filtering an already absolute speed would retain the
    oscillation bias.  Safety evidence remains unfiltered and privileged.
    """

    values = (previous_value, sample_value, dt_s, time_constant_s)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("low-pass values must be finite")
    if float(dt_s) <= 0.0 or float(time_constant_s) <= 0.0:
        raise ValueError("low-pass dt and time constant must be positive")
    alpha = 1.0 - math.exp(-float(dt_s) / float(time_constant_s))
    return float(previous_value) + alpha * (float(sample_value) - float(previous_value))


def _contact_force_hold_settled(
    surface_clearance_speed_mps_by_anchor: Mapping[int, float],
    *,
    selected_anchor_ids: Sequence[int],
    speed_threshold_mps: float,
) -> bool:
    """Non-privileged mesh-clearance-rate gate after the force ramp."""

    if speed_threshold_mps < 0.0 or not math.isfinite(speed_threshold_mps):
        raise ValueError(
            "contact force hold speed threshold must be finite/nonnegative"
        )
    selected = tuple(int(anchor_id) for anchor_id in selected_anchor_ids)
    return bool(selected) and all(
        anchor_id in surface_clearance_speed_mps_by_anchor
        and math.isfinite(float(surface_clearance_speed_mps_by_anchor[anchor_id]))
        and float(surface_clearance_speed_mps_by_anchor[anchor_id])
        <= speed_threshold_mps + 1.0e-12
        for anchor_id in selected
    )


def _should_recenter_contact_pair(
    mesh_clearance_m_by_anchor: Mapping[int, float],
    *,
    engagement_clearance_m: float,
    imbalance_clearance_m: float,
) -> bool:
    """Request a shape-frozen balance cycle before asymmetric contact.

    The decision is intentionally limited to sampled authored-mesh geometry.
    It does not consume raw PhysX contact state or force.
    """

    clearances = tuple(
        float(mesh_clearance_m_by_anchor[anchor_id])
        for anchor_id in sorted(mesh_clearance_m_by_anchor)
    )
    if len(clearances) != 2:
        raise SchemaValidationError(
            "Order8 contact recentering requires exactly two mesh clearances"
        )
    for name, value in (
        ("engagement_clearance_m", engagement_clearance_m),
        ("imbalance_clearance_m", imbalance_clearance_m),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise SchemaValidationError(
                f"Order8 contact recentering {name} must be finite and positive"
            )
    if any(not math.isfinite(value) or value < 0.0 for value in clearances):
        raise SchemaValidationError(
            "Order8 contact recentering clearances must be finite and non-negative"
        )
    return min(clearances) <= float(engagement_clearance_m) and max(clearances) - min(
        clearances
    ) > float(imbalance_clearance_m)


def _contact_pair_centering_settled(
    mesh_clearance_m_by_anchor: Mapping[int, float],
    *,
    base_linear_speed_mps: float,
    speed_tolerance_mps: float,
    imbalance_tolerance_m: float,
    measured_tilt_rad: float,
    max_tilt_rad: float,
) -> bool:
    """Evaluate the achieved geometric purpose of a centering cycle.

    The clearance-derived target is recomputed while the base moves, so exact
    tracking of that moving pose is not the terminal objective.  Commit the
    measured pose once the two authored-mesh clearances are balanced, motion is
    settled, and the measured attitude remains within the configured bound.
    """

    clearances = tuple(
        float(mesh_clearance_m_by_anchor[anchor_id])
        for anchor_id in sorted(mesh_clearance_m_by_anchor)
    )
    if len(clearances) != 2:
        raise SchemaValidationError(
            "contact centering settle requires exactly two mesh clearances"
        )
    positive_values = {
        "speed_tolerance_mps": speed_tolerance_mps,
        "imbalance_tolerance_m": imbalance_tolerance_m,
        "max_tilt_rad": max_tilt_rad,
    }
    if any(
        not math.isfinite(float(value)) or float(value) <= 0.0
        for value in positive_values.values()
    ):
        raise SchemaValidationError(
            "contact centering settle tolerances must be finite and positive"
        )
    non_negative_values = (
        *clearances,
        float(base_linear_speed_mps),
        float(measured_tilt_rad),
    )
    if any(not math.isfinite(value) or value < 0.0 for value in non_negative_values):
        raise SchemaValidationError(
            "contact centering settle observations must be finite and non-negative"
        )
    return bool(
        max(clearances) - min(clearances) <= float(imbalance_tolerance_m)
        and float(base_linear_speed_mps) <= float(speed_tolerance_mps)
        and float(measured_tilt_rad) <= float(max_tilt_rad)
    )


def _contact_centering_base_pose(
    hold_pose: Pose7D,
    current_pose: Pose7D,
    *,
    mesh_clearance_m_by_anchor: Mapping[int, float],
    inward_normal_world_by_anchor: Mapping[int, tuple[float, float, float]],
    max_offset_m: float,
) -> Pose7D:
    """Balance an opposing mesh pair with bounded centroidal translation.

    This is the deterministic ``pi_H`` fallback's coordination step after one
    side has established a kinematic closure arrest.  It uses only sampled
    authored-mesh clearance to the observed object pose.  Raw Isaac contact
    truth is deliberately absent.

    The correction is half of the two clearances' difference, added to the
    *measured current* offset along the farther side's inward normal.  Adding
    it to the current offset avoids the one-half steady-state error produced
    by repeatedly applying the difference directly to the original hold pose.
    The returned orientation and all orthogonal translation components remain
    those of ``hold_pose``.
    """

    anchor_ids = tuple(sorted(int(key) for key in mesh_clearance_m_by_anchor))
    if len(anchor_ids) != 2 or set(inward_normal_world_by_anchor) != set(anchor_ids):
        raise SchemaValidationError(
            "Order8 contact centering requires exactly one normal for each "
            "of two mesh clearances"
        )
    if not math.isfinite(float(max_offset_m)) or float(max_offset_m) <= 0.0:
        raise SchemaValidationError(
            "Order8 contact centering max_offset_m must be finite and positive"
        )
    clearances = {
        anchor_id: float(mesh_clearance_m_by_anchor[anchor_id])
        for anchor_id in anchor_ids
    }
    if any(not math.isfinite(value) or value < 0.0 for value in clearances.values()):
        raise SchemaValidationError(
            "Order8 contact centering clearances must be finite and non-negative"
        )
    unit_normals = {
        anchor_id: _unit(
            tuple(float(value) for value in inward_normal_world_by_anchor[anchor_id])
        )
        for anchor_id in anchor_ids
    }
    if (
        sum(
            unit_normals[anchor_ids[0]][index] * unit_normals[anchor_ids[1]][index]
            for index in range(3)
        )
        > -0.9
    ):
        raise SchemaValidationError(
            "Order8 contact centering requires an opposing gripper pair"
        )

    far_anchor_id = max(
        anchor_ids,
        key=lambda anchor_id: (clearances[anchor_id], anchor_id),
    )
    near_anchor_id = next(
        anchor_id for anchor_id in anchor_ids if anchor_id != far_anchor_id
    )
    centering_axis = unit_normals[far_anchor_id]
    current_offset = tuple(
        float(current_pose[index]) - float(hold_pose[index]) for index in range(3)
    )
    current_axis_offset = sum(
        current_offset[index] * centering_axis[index] for index in range(3)
    )
    clearance_correction = 0.5 * (
        clearances[far_anchor_id] - clearances[near_anchor_id]
    )
    target_axis_offset = min(
        max(current_axis_offset + clearance_correction, -float(max_offset_m)),
        float(max_offset_m),
    )
    return _offset_pose(
        hold_pose,
        dx=target_axis_offset * centering_axis[0],
        dy=target_axis_offset * centering_axis[1],
        dz=target_axis_offset * centering_axis[2],
    )


def _post_first_arrest_centroidal_transfer_pose(
    first_arrest_base_pose: Pose7D,
    *,
    inward_normal_world: tuple[float, float, float],
    maximum_transfer_m: float,
) -> Pose7D:
    """Add a bounded centroidal DOF toward the still-unlatched surface.

    The already-loaded anchor is held separately at its measured world pose.
    Moving the base along the remaining anchor's inward axis therefore relieves
    a Dock-only kinematic singularity without asking QPID to model joint motion.
    """

    if not math.isfinite(float(maximum_transfer_m)) or float(maximum_transfer_m) <= 0.0:
        raise SchemaValidationError(
            "post-first-arrest centroidal maximum_transfer_m must be finite "
            "and positive"
        )
    inward = _unit(tuple(float(value) for value in inward_normal_world))
    return _offset_pose(
        first_arrest_base_pose,
        dx=float(maximum_transfer_m) * inward[0],
        dy=float(maximum_transfer_m) * inward[1],
        dz=float(maximum_transfer_m) * inward[2],
    )


def _sequential_centroidal_transfer_limit_m(
    *,
    observed_clearance_m: float,
    clearance_margin_m: float,
    maximum_transfer_m: float,
) -> float:
    """Snapshot a bounded final transfer from measured surface geometry.

    The sampled mesh clearance is non-privileged geometry available to the
    runtime controller.  Adding one slowdown-band margin lets the load/current
    arrest gate engage without deriving the move from raw contact truth.
    """

    if (
        not math.isfinite(float(observed_clearance_m))
        or float(observed_clearance_m) < 0.0
        or not math.isfinite(float(clearance_margin_m))
        or float(clearance_margin_m) <= 0.0
        or not math.isfinite(float(maximum_transfer_m))
        or float(maximum_transfer_m) <= 0.0
    ):
        raise SchemaValidationError(
            "sequential centroidal transfer geometry must be finite and bounded"
        )
    return min(
        float(maximum_transfer_m),
        max(
            float(clearance_margin_m),
            float(observed_clearance_m) + float(clearance_margin_m),
        ),
    )


def _underactuated_contact_centering_pose(
    position_target: Pose7D,
    *,
    hold_pose: Pose7D,
    current_pose: Pose7D,
    current_linear_velocity_world: Sequence[float],
    speed_limit_mps: float,
    slowdown_distance_m: float,
    position_deadband_m: float,
    xy_p_gain: float,
    xy_d_gain: float,
    gravity_mps2: float,
    max_tilt_rad: float,
) -> Pose7D:
    """Add bounded roll/pitch needed by the one-axis-vectoring airframe.

    Order 8's opposing Dock clearances differ along world ``y`` in the
    representative scene, while each Holon rotor has only one vectoring axis.
    A horizontal pose wrench at fixed level attitude is therefore not enough.
    This outer-loop conversion uses the same QPID XY pose/velocity errors to
    tilt the centroidal target so thrust supplies the otherwise underactuated
    horizontal component.  It remains a centroidal pose command; no contact
    wrench or raw simulator contact truth is consumed.
    """

    values = {
        "speed_limit_mps": speed_limit_mps,
        "slowdown_distance_m": slowdown_distance_m,
        "position_deadband_m": position_deadband_m,
        "xy_p_gain": xy_p_gain,
        "xy_d_gain": xy_d_gain,
        "gravity_mps2": gravity_mps2,
        "max_tilt_rad": max_tilt_rad,
    }
    if any(
        not math.isfinite(float(value)) or float(value) <= 0.0
        for value in values.values()
    ):
        raise SchemaValidationError(
            "Order8 underactuated centering parameters must be finite and positive"
        )
    if len(tuple(current_linear_velocity_world)) != 3 or any(
        not math.isfinite(float(value)) for value in current_linear_velocity_world
    ):
        raise SchemaValidationError(
            "Order8 underactuated centering velocity must contain three finite values"
        )
    horizontal_error = (
        float(position_target[0]) - float(current_pose[0]),
        float(position_target[1]) - float(current_pose[1]),
    )
    distance = math.hypot(*horizontal_error)
    _, hold_rpy = pose_to_xyz_rpy(hold_pose)
    if distance <= float(position_deadband_m):
        target_rpy = hold_rpy
    else:
        direction = (
            horizontal_error[0] / distance,
            horizontal_error[1] / distance,
        )
        speed = float(speed_limit_mps) * min(
            distance / float(slowdown_distance_m),
            1.0,
        )
        desired_velocity = (
            speed * direction[0],
            speed * direction[1],
        )
        desired_acceleration = (
            float(xy_p_gain) * horizontal_error[0]
            + float(xy_d_gain)
            * (desired_velocity[0] - float(current_linear_velocity_world[0])),
            float(xy_p_gain) * horizontal_error[1]
            + float(xy_d_gain)
            * (desired_velocity[1] - float(current_linear_velocity_world[1])),
        )
        yaw = float(hold_rpy[2])
        tilt_roll = (
            desired_acceleration[0] * math.sin(yaw)
            - desired_acceleration[1] * math.cos(yaw)
        ) / float(gravity_mps2)
        tilt_pitch = (
            desired_acceleration[0] * math.cos(yaw)
            + desired_acceleration[1] * math.sin(yaw)
        ) / float(gravity_mps2)
        tilt_norm = math.hypot(tilt_roll, tilt_pitch)
        if tilt_norm > float(max_tilt_rad):
            scale = float(max_tilt_rad) / tilt_norm
            tilt_roll *= scale
            tilt_pitch *= scale
        target_rpy = (
            float(hold_rpy[0]) + tilt_roll,
            float(hold_rpy[1]) + tilt_pitch,
            yaw,
        )
    return pose_from_transform(
        transform_from_xyz_rpy(
            tuple(float(value) for value in position_target[:3]),
            target_rpy,
        )
    )


def _per_anchor_influential_dock_loads(
    anchor_ids: Sequence[int],
    *,
    ordered_joint_ids: Sequence[str],
    anchor_jacobians: Mapping[int, Sequence[Sequence[float]]],
    applied_joint_load_nm: Mapping[str, float],
    required_joint_id_by_anchor: Mapping[int, str],
    influence_epsilon: float = 1.0e-8,
) -> tuple[dict[int, float], dict[int, tuple[str, ...]]]:
    """Attribute measured Dock load to each anchor's kinematic chain.

    The all-Dock controller intentionally permits every joint to contribute to
    grasp morphing, but that does not make a load on one branch evidence for a
    different branch.  A Jacobian column is included for an anchor only when
    it changes that anchor's 6D pose.  The selected terminal mechanism joint
    is always retained as a fail-closed coverage check, including at a
    kinematic singularity where its instantaneous column can be zero.
    """

    ordered_anchors = tuple(int(anchor_id) for anchor_id in anchor_ids)
    if len(ordered_anchors) < 2 or len(set(ordered_anchors)) != len(ordered_anchors):
        raise SchemaValidationError(
            "Order8 per-anchor Dock load requires at least two unique anchors"
        )
    joints = tuple(str(joint_id) for joint_id in ordered_joint_ids)
    if not joints or len(set(joints)) != len(joints):
        raise SchemaValidationError(
            "Order8 per-anchor Dock load requires unique ordered joints"
        )
    if set(anchor_jacobians) != set(ordered_anchors):
        raise SchemaValidationError(
            "Order8 per-anchor Dock-load Jacobians must cover selected anchors"
        )
    if set(applied_joint_load_nm) != set(joints):
        raise SchemaValidationError(
            "Order8 per-anchor applied Dock loads must cover ordered joints"
        )
    if set(required_joint_id_by_anchor) != set(ordered_anchors):
        raise SchemaValidationError(
            "Order8 per-anchor terminal joints must cover selected anchors"
        )
    epsilon = float(influence_epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise SchemaValidationError(
            "Order8 Dock-load influence epsilon must be finite and positive"
        )
    loads = {joint_id: float(applied_joint_load_nm[joint_id]) for joint_id in joints}
    if any(not math.isfinite(value) or value < 0.0 for value in loads.values()):
        raise SchemaValidationError(
            "Order8 applied Dock loads must be finite and non-negative"
        )

    load_by_anchor: dict[int, float] = {}
    joint_ids_by_anchor: dict[int, tuple[str, ...]] = {}
    for anchor_id in ordered_anchors:
        jacobian = tuple(
            tuple(float(value) for value in row) for row in anchor_jacobians[anchor_id]
        )
        if len(jacobian) != 6 or any(len(row) != len(joints) for row in jacobian):
            raise SchemaValidationError(
                "Order8 per-anchor Dock-load Jacobian must be 6 x joint_count"
            )
        if any(not math.isfinite(value) for row in jacobian for value in row):
            raise SchemaValidationError(
                "Order8 per-anchor Dock-load Jacobian must be finite"
            )
        required_joint_id = str(required_joint_id_by_anchor[anchor_id])
        if required_joint_id not in loads:
            raise SchemaValidationError(
                "Order8 per-anchor terminal joint must belong to ordered joints"
            )
        influential = tuple(
            joint_id
            for column, joint_id in enumerate(joints)
            if joint_id == required_joint_id
            or math.sqrt(sum(row[column] ** 2 for row in jacobian)) > epsilon
        )
        joint_ids_by_anchor[anchor_id] = influential
        load_by_anchor[anchor_id] = max(loads[joint_id] for joint_id in influential)
    return load_by_anchor, joint_ids_by_anchor


def _selected_anchor_joint_load_candidates(
    anchor_ids: Sequence[int],
    *,
    selected_joint_load_nm_by_anchor: Mapping[int, float],
    selected_joint_load_threshold_nm: float,
) -> dict[int, bool]:
    """Detect terminal-joint load without simulator contact or mesh geometry."""

    ordered_ids = tuple(int(anchor_id) for anchor_id in anchor_ids)
    if len(ordered_ids) < 2 or len(set(ordered_ids)) != len(ordered_ids):
        raise SchemaValidationError(
            "Order8 joint-load detection requires at least two unique anchor ids"
        )
    if set(selected_joint_load_nm_by_anchor) != set(ordered_ids):
        raise SchemaValidationError(
            "Order8 joint-load observations must cover exactly the selected anchors"
        )
    threshold = float(selected_joint_load_threshold_nm)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise SchemaValidationError(
            "Order8 joint-load threshold must be finite and positive"
        )
    if any(
        not math.isfinite(float(selected_joint_load_nm_by_anchor[anchor_id]))
        or float(selected_joint_load_nm_by_anchor[anchor_id]) < 0.0
        for anchor_id in ordered_ids
    ):
        raise SchemaValidationError(
            "Order8 joint-load observations must be finite and non-negative"
        )
    return {
        anchor_id: bool(
            float(selected_joint_load_nm_by_anchor[anchor_id]) >= threshold
        )
        for anchor_id in ordered_ids
    }


def _damping_compensated_joint_load_nm(
    *,
    applied_torque_nm: float,
    estimated_damping_drive_torque_nm: float,
) -> float:
    """Remove virtual-drive damping from the contact-load observation.

    Isaac's applied joint torque includes the implicit velocity-drive term.
    During closure startup that term can exceed the small contact threshold
    even while the Dock mesh is still in free space.  Position-drive and
    torque-bias contributions remain observable because they are also how a
    real contact reaction appears at this actuator boundary.
    """

    applied = float(applied_torque_nm)
    damping = float(estimated_damping_drive_torque_nm)
    if not math.isfinite(applied) or not math.isfinite(damping):
        raise SchemaValidationError(
            "Order8 damping-compensated joint load inputs must be finite"
        )
    return abs(applied - damping)


def _selected_anchor_surface_load_arrest_candidates(
    anchor_ids: Sequence[int],
    *,
    mesh_clearance_m_by_anchor: Mapping[int, float],
    selected_joint_load_nm_by_anchor: Mapping[int, float],
    mesh_clearance_arm_threshold_m: float,
    selected_joint_load_threshold_nm: float,
) -> dict[int, bool]:
    """Detect per-side q_close arrest without privileged contact truth.

    The arrest signal deliberately excludes velocity.  Velocity is expected
    to settle *after* the measured joint configuration is held; requiring it
    beforehand creates a control deadlock in which the position target keeps
    advancing into the object.  The selected load is the terminal Dock's
    actuator torque/current-equivalent observation, while the broader
    per-anchor kinematic-chain load remains part of the post-arrest stable-
    grasp gate.
    """

    return _selected_anchor_surface_load_settle_candidates(
        anchor_ids,
        object_normal_relative_speed_mps_by_anchor={
            int(anchor_id): 0.0 for anchor_id in anchor_ids
        },
        mesh_clearance_m_by_anchor=mesh_clearance_m_by_anchor,
        selected_joint_load_nm_by_anchor=selected_joint_load_nm_by_anchor,
        anchor_speed_threshold_mps=1.0,
        mesh_clearance_arm_threshold_m=mesh_clearance_arm_threshold_m,
        selected_joint_load_threshold_nm=selected_joint_load_threshold_nm,
    )


def _all_selected_anchor_surface_load_settled(
    anchor_ids: Sequence[int],
    *,
    object_normal_relative_speed_mps_by_anchor: Mapping[int, float],
    mesh_clearance_m_by_anchor: Mapping[int, float],
    selected_joint_load_nm_by_anchor: Mapping[int, float],
    anchor_speed_threshold_mps: float,
    mesh_clearance_arm_threshold_m: float,
    selected_joint_load_threshold_nm: float,
) -> bool:
    """Detect a settled two-sided surface/load state without contact truth.

    Speeds are measured relative to the free object's observed pose so object
    motion cannot masquerade as a settled Dock closure.  The authored-mesh
    proximity and actuator load/current-equivalent gates reject a free-space
    stop.  Raw contact truth remains excluded, and temporal dwell/latching are
    intentionally owned by the caller.
    """

    candidates = _selected_anchor_surface_load_settle_candidates(
        anchor_ids,
        object_normal_relative_speed_mps_by_anchor=(
            object_normal_relative_speed_mps_by_anchor
        ),
        mesh_clearance_m_by_anchor=mesh_clearance_m_by_anchor,
        selected_joint_load_nm_by_anchor=selected_joint_load_nm_by_anchor,
        anchor_speed_threshold_mps=anchor_speed_threshold_mps,
        mesh_clearance_arm_threshold_m=mesh_clearance_arm_threshold_m,
        selected_joint_load_threshold_nm=selected_joint_load_threshold_nm,
    )
    return all(candidates.values())


def _selected_anchor_surface_load_settle_candidates(
    anchor_ids: Sequence[int],
    *,
    object_normal_relative_speed_mps_by_anchor: Mapping[int, float],
    mesh_clearance_m_by_anchor: Mapping[int, float],
    selected_joint_load_nm_by_anchor: Mapping[int, float],
    anchor_speed_threshold_mps: float,
    mesh_clearance_arm_threshold_m: float,
    selected_joint_load_threshold_nm: float,
) -> dict[int, bool]:
    """Return each selected Dock's non-privileged surface/load settle gate.

    Each anchor receives an independent candidate and dwell so a provisional
    contact can disappear and reset without being frozen.  The final q_close
    gate requires every candidate simultaneously.  Inputs remain only
    object-relative motion, authored mesh geometry, and Dock actuator
    load/current observations; raw Isaac contact truth is excluded.  The load
    gate prevents a slow free-space endpoint from being mislabeled as contact.
    """

    ordered_ids = tuple(int(anchor_id) for anchor_id in anchor_ids)
    if len(ordered_ids) < 2 or len(set(ordered_ids)) != len(ordered_ids):
        raise SchemaValidationError(
            "Order8 stall detection requires at least two unique anchor ids"
        )
    expected = set(ordered_ids)
    if set(object_normal_relative_speed_mps_by_anchor) != expected:
        raise SchemaValidationError(
            "Order8 stall anchor speeds must cover exactly the selected anchors"
        )
    if set(mesh_clearance_m_by_anchor) != expected:
        raise SchemaValidationError(
            "Order8 stall mesh clearances must cover exactly the selected anchors"
        )
    if set(selected_joint_load_nm_by_anchor) != expected:
        raise SchemaValidationError(
            "Order8 stall selected-joint loads must cover exactly the selected anchors"
        )
    for name, value in (
        ("anchor_speed_threshold_mps", anchor_speed_threshold_mps),
        ("mesh_clearance_arm_threshold_m", mesh_clearance_arm_threshold_m),
        ("selected_joint_load_threshold_nm", selected_joint_load_threshold_nm),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise SchemaValidationError(
                f"Order8 stall {name} must be finite and positive"
            )
    for label, values in (
        ("anchor normal speed", object_normal_relative_speed_mps_by_anchor),
        ("mesh clearance", mesh_clearance_m_by_anchor),
        ("selected-joint load", selected_joint_load_nm_by_anchor),
    ):
        if any(
            not math.isfinite(float(values[anchor_id]))
            or float(values[anchor_id]) < 0.0
            for anchor_id in ordered_ids
        ):
            raise SchemaValidationError(
                f"Order8 stall {label} values must be finite and non-negative"
            )
    return {
        anchor_id: bool(
            float(object_normal_relative_speed_mps_by_anchor[anchor_id])
            <= float(anchor_speed_threshold_mps)
            and float(mesh_clearance_m_by_anchor[anchor_id])
            <= float(mesh_clearance_arm_threshold_m)
            and float(selected_joint_load_nm_by_anchor[anchor_id])
            >= float(selected_joint_load_threshold_nm)
        )
        for anchor_id in ordered_ids
    }


def _base_hold_settled(
    target_pose: Pose7D,
    measured_pose: Pose7D,
    *,
    base_linear_speed_mps: float,
    position_tolerance_m: float,
    speed_tolerance_mps: float,
) -> bool:
    """Non-contact gate for serializing axial insertion and side closure."""

    return bool(
        math.isfinite(float(base_linear_speed_mps))
        and math.isfinite(float(position_tolerance_m))
        and math.isfinite(float(speed_tolerance_mps))
        and float(position_tolerance_m) > 0.0
        and float(speed_tolerance_mps) > 0.0
        and _position_distance(target_pose, measured_pose)
        <= float(position_tolerance_m)
        and float(base_linear_speed_mps) <= float(speed_tolerance_mps)
    )


def _contact_anchor_target_speed_limit_mps(
    *,
    base_limit_mps: float,
    mesh_clearance_m: float,
    near_mesh_clearance_m: float,
    surface_arm_clearance_m: float,
    surface_creep_speed_limit_mps: float,
) -> float:
    """Apply far, near-surface, and final-creep closure speed tiers."""

    if (
        not math.isfinite(float(base_limit_mps))
        or float(base_limit_mps) <= 0.0
        or not math.isfinite(float(mesh_clearance_m))
        or float(mesh_clearance_m) < 0.0
        or not math.isfinite(float(near_mesh_clearance_m))
        or float(near_mesh_clearance_m) <= 0.0
        or not math.isfinite(float(surface_arm_clearance_m))
        or float(surface_arm_clearance_m) <= 0.0
        or not math.isfinite(float(surface_creep_speed_limit_mps))
        or float(surface_creep_speed_limit_mps) <= 0.0
        or float(surface_arm_clearance_m) > float(near_mesh_clearance_m)
        or float(surface_creep_speed_limit_mps) > 0.2 * float(base_limit_mps)
    ):
        raise SchemaValidationError(
            "contact anchor speed schedule requires ordered finite positive "
            "limits and non-negative mesh clearance"
        )
    if float(mesh_clearance_m) <= float(surface_arm_clearance_m):
        return float(surface_creep_speed_limit_mps)
    if float(mesh_clearance_m) <= float(near_mesh_clearance_m):
        return 0.2 * float(base_limit_mps)
    return float(base_limit_mps)


def _accelerate_unlatched_anchor_after_first_arrest(
    speed_limit_mps_by_anchor: Mapping[int, float],
    *,
    latched_anchor_ids: set[int],
    maximum_speed_mps: float,
    creep_speed_mps: float,
    multiplier: float,
) -> dict[int, float]:
    """Accelerate only the remaining side after the first loaded arrest.

    The already-arrested anchor is held by its measured pose elsewhere.  This
    schedule therefore shortens the unloaded second-side search without
    increasing the velocity of the maintained contact.  It consumes no raw
    simulator contact truth.
    """

    speeds = {
        int(anchor_id): float(value)
        for anchor_id, value in speed_limit_mps_by_anchor.items()
    }
    anchor_ids = set(speeds)
    if len(anchor_ids) < 2 or not latched_anchor_ids.issubset(anchor_ids):
        raise SchemaValidationError(
            "post-arrest creep schedule requires at least two anchors and a "
            "valid latched subset"
        )
    for name, value in (
        ("maximum_speed_mps", maximum_speed_mps),
        ("creep_speed_mps", creep_speed_mps),
        ("multiplier", multiplier),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise SchemaValidationError(
                f"post-arrest creep {name} must be finite and positive"
            )
    if any(not math.isfinite(value) or value <= 0.0 for value in speeds.values()):
        raise SchemaValidationError(
            "post-arrest creep input speeds must be finite and positive"
        )
    if not 0 < len(latched_anchor_ids) < len(anchor_ids):
        return speeds
    accelerated_floor = min(
        float(maximum_speed_mps),
        float(creep_speed_mps) * float(multiplier),
    )
    return {
        anchor_id: (
            speed
            if anchor_id in latched_anchor_ids
            else min(float(maximum_speed_mps), max(speed, accelerated_floor))
        )
        for anchor_id, speed in speeds.items()
    }


def _contact_anchor_target_speed_limits_mps(
    *,
    base_limit_mps: float,
    mesh_clearance_m_by_anchor: Mapping[int, float],
    near_mesh_clearance_m: float,
    surface_arm_clearance_m: float,
    surface_creep_speed_limit_mps: float,
) -> dict[int, float]:
    """Schedule each opposing anchor from its own authored-mesh clearance.

    A shared minimum clearance would slow the farther anchor together with the
    near one and preserve the very asymmetry that the schedule is meant to
    remove.  Per-anchor scheduling lets the farther side catch up while the
    near side approaches conservatively; it consumes no raw contact state.
    """

    clearances = {
        int(anchor_id): float(clearance)
        for anchor_id, clearance in mesh_clearance_m_by_anchor.items()
    }
    if len(clearances) != 2:
        raise SchemaValidationError(
            "contact anchor speed scheduling requires exactly two mesh clearances"
        )
    return {
        anchor_id: _contact_anchor_target_speed_limit_mps(
            base_limit_mps=base_limit_mps,
            mesh_clearance_m=clearance,
            near_mesh_clearance_m=near_mesh_clearance_m,
            surface_arm_clearance_m=surface_arm_clearance_m,
            surface_creep_speed_limit_mps=surface_creep_speed_limit_mps,
        )
        for anchor_id, clearance in sorted(clearances.items())
    }


def _clearance_synchronized_contact_anchor_target_speed_limits_mps(
    tier_speed_limit_mps_by_anchor: Mapping[int, float],
    *,
    mesh_clearance_m_by_anchor: Mapping[int, float],
    deadband_m: float,
    full_slowdown_m: float,
    minimum_speed_scale: float,
) -> dict[int, float]:
    """Slow only the closer surface as an opposing pair becomes imbalanced.

    The three-tier mesh-proximity schedule bounds each surface's absolute
    closing speed.  This second, non-privileged coordination layer compensates
    unequal articulated response before the discrete centroidal recenter
    fallback is needed: the farther surface keeps its tier speed, while the
    closer one is linearly reduced between the deadband and full-slowdown
    imbalance.  No raw contact or force state enters this calculation.
    """

    speeds = {
        int(anchor_id): float(speed)
        for anchor_id, speed in tier_speed_limit_mps_by_anchor.items()
    }
    clearances = {
        int(anchor_id): float(clearance)
        for anchor_id, clearance in mesh_clearance_m_by_anchor.items()
    }
    if len(speeds) != 2 or set(speeds) != set(clearances):
        raise SchemaValidationError(
            "clearance synchronization requires the same opposing anchor pair"
        )
    if any(
        not math.isfinite(value) or value <= 0.0 for value in speeds.values()
    ) or any(not math.isfinite(value) or value < 0.0 for value in clearances.values()):
        raise SchemaValidationError(
            "clearance synchronization requires finite positive speeds and "
            "non-negative clearances"
        )
    if (
        not math.isfinite(float(deadband_m))
        or float(deadband_m) <= 0.0
        or not math.isfinite(float(full_slowdown_m))
        or float(full_slowdown_m) <= float(deadband_m)
        or not math.isfinite(float(minimum_speed_scale))
        or not 0.0 < float(minimum_speed_scale) <= 1.0
    ):
        raise SchemaValidationError(
            "clearance synchronization requires ordered positive geometry "
            "thresholds and a speed scale in (0, 1]"
        )
    nearer_anchor_id = min(clearances, key=lambda anchor_id: clearances[anchor_id])
    farther_anchor_id = max(clearances, key=lambda anchor_id: clearances[anchor_id])
    imbalance_m = clearances[farther_anchor_id] - clearances[nearer_anchor_id]
    if imbalance_m <= float(deadband_m):
        return dict(sorted(speeds.items()))
    fraction = min(
        1.0,
        (imbalance_m - float(deadband_m))
        / (float(full_slowdown_m) - float(deadband_m)),
    )
    speed_scale = 1.0 - fraction * (1.0 - float(minimum_speed_scale))
    speeds[nearer_anchor_id] *= speed_scale
    return dict(sorted(speeds.items()))


def _position_distance(left: Any, right: Any) -> float:
    return math.sqrt(
        sum((float(left[index]) - float(right[index])) ** 2 for index in range(3))
    )


def _norm(values: Any) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))


def _add(left: Any, right: Any) -> list[float]:
    return [float(left[index]) + float(right[index]) for index in range(3)]


def _subtract(left: Any, right: Any) -> list[float]:
    return [float(left[index]) - float(right[index]) for index in range(3)]


def _cross(left: Any, right: Any) -> list[float]:
    return [
        float(left[1]) * float(right[2]) - float(left[2]) * float(right[1]),
        float(left[2]) * float(right[0]) - float(left[0]) * float(right[2]),
        float(left[0]) * float(right[1]) - float(left[1]) * float(right[0]),
    ]


def _unit(values: Any) -> tuple[float, float, float]:
    length = _norm(values)
    if length <= 1.0e-12:
        raise SchemaValidationError("cannot normalize a zero vector")
    return tuple(float(value) / length for value in values)


__all__ = [
    "ORDER8_ISAAC_REPORT_VERSION",
    "format_order8_progress",
    "run_order8_isaac_runtime",
]
