from __future__ import annotations

import torch
from torch import nn

from amsrr.policies.learned_low_level_policy import (
    PI_L_FEATURE_NAMES,
    PI_L_TARGET_NAMES,
    LearnedLowLevelPolicy,
    overlay_learned_pi_l_subset,
    pi_l_feature_vector,
)
from amsrr.policies.low_level_policy_base import BaselineLowLevelPolicy, LowLevelPolicyContext
from amsrr.schemas.morphology import ModuleNode, MorphologyGraph
from amsrr.schemas.physical_model import ModuleCapabilityToken, PhysicalModel
from amsrr.schemas.policies import (
    CentroidalTarget,
    ContactWrenchTrajectory,
    ControllerStatus,
    InteractionKnot,
    ObjectTarget,
)
from amsrr.schemas.runtime import (
    ModuleRuntimeState,
    ObjectRuntimeState,
    RuntimeObservation,
    TaskProgressState,
)


class _FixedOutput(nn.Module):
    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.values = values

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tensor([self.values], dtype=torch.float32)


class _WrongShape(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.zeros((1, len(PI_L_TARGET_NAMES) - 1), dtype=torch.float32)


def test_learned_pi_l_merges_only_a_bounded_policy_command_subset() -> None:
    context = _context()
    baseline = BaselineLowLevelPolicy().command(context)
    features = pi_l_feature_vector(context)
    policy = LearnedLowLevelPolicy(
        model=_FixedOutput([100.0] * len(PI_L_TARGET_NAMES)),
        feature_mean=features,
        feature_std=[1.0] * len(PI_L_FEATURE_NAMES),
        feature_ood_scale=[1.0] * len(PI_L_FEATURE_NAMES),
    )

    command = policy.command(context)

    assert policy.last_diagnostics.used_learned_delta is True
    assert command.desired_body_twist is not None
    assert baseline.desired_body_twist is not None
    assert command.desired_body_twist == [value + 0.5 for value in baseline.desired_body_twist]
    assert command.desired_body_pose is not None
    assert baseline.desired_body_pose is not None
    assert command.desired_body_pose[:3] == tuple(
        value + delta
        for value, delta in zip(
            baseline.desired_body_pose[:3],
            (1.75, 1.35, 0.40),
        )
    )
    assert command.desired_body_pose[3:] == baseline.desired_body_pose[3:]
    assert command.residual_wrench_body is not None
    assert baseline.residual_wrench_body is not None
    assert command.residual_wrench_body[:3] == [value + 4.0 for value in baseline.residual_wrench_body[:3]]
    assert command.residual_wrench_body[3:] == [value + 0.5 for value in baseline.residual_wrench_body[3:]]
    assert command.desired_anchor_pose_offsets == baseline.desired_anchor_pose_offsets
    assert command.joint_position_bias == baseline.joint_position_bias
    assert command.joint_velocity_bias == baseline.joint_velocity_bias
    assert command.contact_tracking_bias == baseline.contact_tracking_bias
    assert command.priority_weights == baseline.priority_weights


def test_learned_pi_l_falls_back_on_shape_nan_ood_and_controller_infeasible() -> None:
    context = _context()
    features = pi_l_feature_vector(context)
    baseline = BaselineLowLevelPolicy().command(context)

    shape_policy = _policy(_WrongShape(), features)
    assert shape_policy.command(context).to_dict() == baseline.to_dict()
    assert shape_policy.last_diagnostics.fallback_reason == "model_output_shape"

    nan_policy = _policy(_FixedOutput([float("nan")] * len(PI_L_TARGET_NAMES)), features)
    assert nan_policy.command(context).to_dict() == baseline.to_dict()
    assert nan_policy.last_diagnostics.fallback_reason == "non_finite_model_output"

    ood_context = _context(time_s=100.0)
    ood_policy = _policy(_FixedOutput([0.0] * len(PI_L_TARGET_NAMES)), features)
    assert ood_policy.command(ood_context).to_dict() == BaselineLowLevelPolicy().command(ood_context).to_dict()
    assert ood_policy.last_diagnostics.fallback_reason == "feature_ood"

    infeasible_context = _context(
        controller_status=ControllerStatus(status="infeasible", qp_feasible=False)
    )
    infeasible_policy = _policy(_FixedOutput([0.0] * len(PI_L_TARGET_NAMES)), features)
    assert infeasible_policy.command(infeasible_context).to_dict() == BaselineLowLevelPolicy().command(
        infeasible_context
    ).to_dict()
    assert infeasible_policy.last_diagnostics.fallback_reason == "controller_infeasible"

    non_finite_context = _context(
        controller_status=ControllerStatus(
            status="ok",
            qp_feasible=True,
            metrics={"allocation_residual_norm": float("nan")},
        )
    )
    non_finite_policy = _policy(_FixedOutput([0.0] * len(PI_L_TARGET_NAMES)), features)
    assert non_finite_policy.command(non_finite_context).to_dict() == BaselineLowLevelPolicy().command(
        non_finite_context
    ).to_dict()
    assert non_finite_policy.last_diagnostics.fallback_reason == "non_finite_features"


def test_runtime_overlay_preserves_all_nonlearned_deterministic_intent() -> None:
    template = BaselineLowLevelPolicy().command(_context())
    learned = template.to_dict()
    learned["desired_body_twist"] = [0.25] * 6
    learned["priority_weights"] = {"must_not_replace": 99.0}
    learned["desired_anchor_pose_offsets"] = {}

    overlaid = overlay_learned_pi_l_subset(
        template,
        type(template).from_dict(learned),
        blend_factor=0.25,
    )

    assert template.desired_body_twist is not None
    assert overlaid.desired_body_twist == [
        value + 0.25 * (0.25 - value)
        for value in template.desired_body_twist
    ]
    assert overlaid.desired_body_pose is not None
    assert template.desired_body_pose is not None
    assert overlaid.desired_body_pose[3:] == template.desired_body_pose[3:]
    assert overlaid.desired_anchor_pose_offsets == template.desired_anchor_pose_offsets
    assert overlaid.contact_tracking_bias == template.contact_tracking_bias
    assert overlaid.priority_weights == template.priority_weights


def _policy(model: nn.Module, feature_mean: list[float]) -> LearnedLowLevelPolicy:
    return LearnedLowLevelPolicy(
        model=model,
        feature_mean=feature_mean,
        feature_std=[1.0] * len(PI_L_FEATURE_NAMES),
        feature_ood_scale=[1.0] * len(PI_L_FEATURE_NAMES),
    )


def _context(
    *,
    time_s: float = 0.25,
    controller_status: ControllerStatus | None = None,
) -> LowLevelPolicyContext:
    status = controller_status or ControllerStatus(
        status="ok",
        qp_feasible=True,
        metrics={"allocation_residual_norm": 0.05},
    )
    capability = ModuleCapabilityToken(
        module_type="holon",
        aggregate_mass_norm=1.0,
        aggregate_inertia_features=[0.0] * 6,
        rotor_count=4,
        port_count=4,
        thrust_min_features=[0.0] * 4,
        thrust_max_features=[1.0] * 4,
        thrust_to_weight_ratio_est=2.0,
        dock_port_type_counts=[2, 2],
        has_vectoring=True,
        has_dock_mechanism=True,
    )
    morphology = MorphologyGraph(
        graph_id="morphology-1",
        modules=[
            ModuleNode(
                module_id=0,
                module_type="holon",
                pose_in_design_frame=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                role_id="base",
                is_base=True,
                capability_token=capability,
            )
        ],
        ports=[],
        dock_edges=[],
        robot_anchors=[],
        control_groups=[],
        base_module_id=0,
        is_closed_loop=False,
    )
    observation = RuntimeObservation(
        time_s=time_s,
        morphology_graph=morphology,
        module_states=[
            ModuleRuntimeState(
                module_id=0,
                pose_world=(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                twist_world=[0.0] * 6,
            )
        ],
        object_states=[
            ObjectRuntimeState(
                object_id="box",
                pose_world=(0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0),
                twist_world=[0.0] * 6,
            )
        ],
        contact_states=[],
        controller_status=status,
        task_progress=TaskProgressState(progress_ratio=0.25),
    )
    knot = InteractionKnot(
        t_rel_s=0.0,
        contact_assignments=[],
        centroidal_target=CentroidalTarget(
            com_pos_world=(0.1, 0.0, 1.0),
            com_vel_world=(0.1, 0.0, 0.0),
            body_orientation_world=(0.0, 0.0, 0.0, 1.0),
        ),
        object_targets=[
            ObjectTarget(
                object_id="box",
                pose_target_world=(0.7, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0),
                twist_target_world=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            )
        ],
    )
    return LowLevelPolicyContext(
        runtime_observation=observation,
        morphology_graph=morphology,
        physical_model=PhysicalModel(
            model_id="physical-model-1",
            urdf_path="module_urdf/holon.urdf",
            links=[],
            joints=[],
            rotors=[],
            dock_ports=[],
            collision_primitives=[],
            aggregate_mass_kg=1.0,
            aggregate_inertia_body=[0.0] * 6,
        ),
        contact_wrench_trajectory=ContactWrenchTrajectory(
            horizon_s=1.0,
            dt_s=0.01,
            knots=[knot],
        ),
        active_knot=knot,
        controller_status=status,
    )
