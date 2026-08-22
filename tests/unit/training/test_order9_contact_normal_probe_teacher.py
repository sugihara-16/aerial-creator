from __future__ import annotations

from amsrr.training.order9_contact_normal_probe_teacher import (
    Order9ContactNormalProbeCandidate,
    order9_contact_normal_head_named_parameters,
    select_order9_contact_normal_preload_deficit_teacher_rows,
    select_order9_contact_normal_probe_teacher_candidate,
    validate_order9_contact_normal_probe_teacher_target_lineage,
)
from amsrr.schemas.common import SchemaValidationError
import pytest
from amsrr.policies.order9_low_level_policy import (
    Order9CategoricalContactNormalLowLevelPolicyConfig,
    Order9CategoricalContactNormalPhaseConditionedActorCritic,
)


def _candidate(
    residual_mm: float,
    force_n: float,
    *,
    terminal: bool = False,
    collision: bool = False,
) -> Order9ContactNormalProbeCandidate:
    return Order9ContactNormalProbeCandidate(
        residual_mm=residual_mm,
        environment_index=int(residual_mm + 20.0),
        evaluated_step_count=64,
        mean_reward=1.0 + residual_mm / 100.0,
        weak_anchor_normal_force_n=force_n,
        qp_feasible_fraction=1.0,
        prohibited_collision=collision,
        rotor_saturation=False,
        terminal=terminal,
    )


def test_probe_teacher_selects_smallest_safe_force_satisfying_residual() -> None:
    selected = select_order9_contact_normal_probe_teacher_candidate(
        (
            _candidate(-2.0, 2.5),
            _candidate(0.0, 1.8),
            _candidate(2.0, 2.2, terminal=True),
            _candidate(4.0, 2.3),
            _candidate(8.0, 3.0),
        ),
        required_weak_anchor_normal_force_n=2.061,
    )

    assert selected is not None
    assert selected.residual_mm == 4.0


def test_probe_teacher_excludes_state_without_safe_force_support() -> None:
    selected = select_order9_contact_normal_probe_teacher_candidate(
        (
            _candidate(0.0, 1.8),
            _candidate(4.0, 2.3, collision=True),
            _candidate(8.0, 2.5, terminal=True),
        ),
        required_weak_anchor_normal_force_n=2.061,
    )

    assert selected is None


def test_probe_teacher_trainable_boundary_is_only_categorical_normal_head() -> None:
    policy = Order9CategoricalContactNormalPhaseConditionedActorCritic(
        Order9CategoricalContactNormalLowLevelPolicyConfig(
            graph_hidden_dim=8,
            graph_message_layers=1,
            recurrent_hidden_dim=12,
            max_local_joint_slots=4,
        )
    )

    names = {name for name, _ in order9_contact_normal_head_named_parameters(policy)}

    assert names
    assert all(name.startswith("contact_normal_category_logits.") for name in names)
    assert "actor_mean.weight" not in names
    assert "joint_decoder.2.weight" not in names


def test_probe_teacher_accepts_source_or_direct_ppo_child_only() -> None:
    source = "a" * 64
    validate_order9_contact_normal_probe_teacher_target_lineage(
        probe_source_checkpoint_sha256=source,
        target_checkpoint_sha256=source,
        target_parent_checkpoint_sha256=None,
    )
    validate_order9_contact_normal_probe_teacher_target_lineage(
        probe_source_checkpoint_sha256=source,
        target_checkpoint_sha256="b" * 64,
        target_parent_checkpoint_sha256=source,
    )
    with pytest.raises(SchemaValidationError, match="direct child"):
        validate_order9_contact_normal_probe_teacher_target_lineage(
            probe_source_checkpoint_sha256=source,
            target_checkpoint_sha256="b" * 64,
            target_parent_checkpoint_sha256="c" * 64,
        )


def test_preload_deficit_teacher_selects_positive_and_retention_rows() -> None:
    import torch

    valid = torch.ones(6, 3, dtype=torch.bool)
    phases = torch.tensor(
        [
            [0, 0, 0],
            [1, 1, 1],
            [1, 1, 1],
            [2, 2, 2],
            [2, 2, 2],
            [3, 3, 3],
        ]
    )
    examples, retention = (
        select_order9_contact_normal_preload_deficit_teacher_rows(
            valid=valid,
            phase_index=phases,
            category_index_by_environment=torch.tensor([40, 44, 52]),
            deficit_environment_mask=torch.tensor([False, True, True]),
            target_phase_indices=(1, 2),
        )
    )

    assert len(examples) == 4
    assert len(retention) == 2
    assert {int(value["target_category_index"]) for value in examples} == {44, 52}
    assert {float(value["target_residual_mm"]) for value in examples} == {2.0, 6.0}
    assert {int(value["phase_index"]) for value in retention} == {1, 2}
