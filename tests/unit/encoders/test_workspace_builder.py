from __future__ import annotations

import pytest

from amsrr.encoders.interaction_envelope_encoder import InteractionEnvelopeEncoder
from amsrr.encoders.workspace_builder import (
    SharedInteractionWorkspaceBuilder,
    empty_workspace_token_group,
    workspace_token_group_from_encoder_output,
)
from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.task_spec import TaskSpec
from amsrr.schemas.workspace import REQUIRED_WORKSPACE_GROUPS, WorkspaceTokenGroup, tensor_shape


def _envelope_group(grasp_carry_dict: dict):
    task = TaskSpec.from_dict(grasp_carry_dict)
    irg = IRGBuilder().build(task)
    envelope = InteractionEnvelopeExtractor().extract(irg)
    encoded = InteractionEnvelopeEncoder(d_model=8, max_tokens=24).encode(envelope)
    return workspace_token_group_from_encoder_output("interaction_envelope", encoded)


def test_workspace_builder_assembles_required_group_slices(grasp_carry_dict: dict) -> None:
    envelope_group = _envelope_group(grasp_carry_dict)
    workspace = SharedInteractionWorkspaceBuilder(d_model=8).build(
        {"interaction_envelope": envelope_group},
        metadata={"context": "unit"},
    )

    assert tensor_shape(workspace.tokens) == (1, 24, 8)
    assert tensor_shape(workspace.mask) == (1, 24)
    assert set(REQUIRED_WORKSPACE_GROUPS).issubset(workspace.group_slices)
    assert workspace.group_slices["task"] == slice(0, 0)
    assert workspace.group_slices["interaction_envelope"] == slice(0, 24)
    assert workspace.group_slices["capability"] == slice(24, 24)
    assert workspace.group_masks["interaction_envelope"] == envelope_group.mask
    assert workspace.metadata["context"] == "unit"


def test_workspace_builder_supports_optional_contact_candidate_group() -> None:
    candidates = WorkspaceTokenGroup(
        group_name="contact_candidates",
        tokens=[[[0.5, 0.25]]],
        mask=[[True]],
        token_type_ids=[[101]],
        source_type_ids=[[70]],
        source_ids=[[42]],
    )
    workspace = SharedInteractionWorkspaceBuilder(d_model=2, include_contact_candidates=True).build(
        {"contact_candidates": candidates}
    )

    assert workspace.group_slices["contact_candidates"] == slice(0, 1)
    assert workspace.source_ids == [[42]]


def test_workspace_builder_rejects_mismatched_d_model(grasp_carry_dict: dict) -> None:
    group = _envelope_group(grasp_carry_dict)
    with pytest.raises(SchemaValidationError, match="d_model"):
        SharedInteractionWorkspaceBuilder(d_model=16).build({"interaction_envelope": group})


def test_empty_workspace_token_group_contract() -> None:
    group = empty_workspace_token_group("runtime", batch_size=2, d_model=4)
    assert group.mask == [[], []]
    assert group.d_model == 4
