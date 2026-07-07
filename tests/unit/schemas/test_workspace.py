from __future__ import annotations

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.workspace import (
    LearnedQuerySpec,
    SharedInteractionWorkspace,
    WorkspaceTokenGroup,
    recommended_learned_query_specs,
)


def _group_slices() -> dict[str, slice]:
    return {
        "task": slice(0, 1),
        "geometry": slice(1, 2),
        "irg": slice(2, 2),
        "interaction_envelope": slice(2, 2),
        "morphology": slice(2, 2),
        "runtime": slice(2, 2),
        "capability": slice(2, 2),
    }


def _group_masks() -> dict[str, list[list[bool]]]:
    return {
        "task": [[True]],
        "geometry": [[True]],
        "irg": [[]],
        "interaction_envelope": [[]],
        "morphology": [[]],
        "runtime": [[]],
        "capability": [[]],
    }


def test_shared_interaction_workspace_tensor_shapes() -> None:
    workspace = SharedInteractionWorkspace(
        tokens=[[[0.1, 0.2], [0.3, 0.4]]],
        mask=[[True, True]],
        token_type_ids=[[1, 2]],
        source_type_ids=[[10, 20]],
        source_ids=[[100, 200]],
        group_slices=_group_slices(),
        group_masks=_group_masks(),
    )

    roundtrip = SharedInteractionWorkspace.from_json(workspace.to_json())
    assert roundtrip.to_dict() == workspace.to_dict()


def test_padded_tensor_masks() -> None:
    with pytest.raises(SchemaValidationError, match="mask"):
        SharedInteractionWorkspace(
            tokens=[[[0.1, 0.2], [0.3, 0.4]]],
            mask=[[True]],
            token_type_ids=[[1, 2]],
            source_type_ids=[[10, 20]],
            source_ids=[[100, 200]],
            group_slices=_group_slices(),
            group_masks=_group_masks(),
        )


def test_workspace_rejects_group_mask_mismatch() -> None:
    masks = _group_masks()
    masks["geometry"] = [[False]]
    with pytest.raises(SchemaValidationError, match="group_masks"):
        SharedInteractionWorkspace(
            tokens=[[[0.1, 0.2], [0.3, 0.4]]],
            mask=[[True, True]],
            token_type_ids=[[1, 2]],
            source_type_ids=[[10, 20]],
            source_ids=[[100, 200]],
            group_slices=_group_slices(),
            group_masks=masks,
        )


def test_workspace_token_group_shapes() -> None:
    group = WorkspaceTokenGroup(
        group_name="interaction_envelope",
        tokens=[[[0.1, 0.2], [0.3, 0.4]]],
        mask=[[True, True]],
        token_type_ids=[[1, 2]],
        source_type_ids=[[40, 40]],
        source_ids=[[-1, 0]],
    )

    assert group.group_name == "interaction_envelope"


def test_learned_query_spec_contract() -> None:
    specs = recommended_learned_query_specs(d_model=16)
    assert {spec.query_name for spec in specs} == {"design", "high_level", "low_level", "critic", "feasibility"}
    assert all(spec.d_model == 16 for spec in specs)

    with pytest.raises(SchemaValidationError, match="unknown groups"):
        LearnedQuerySpec(
            query_name="design",
            num_queries=1,
            d_model=16,
            allowed_token_groups=["not_a_group"],
        )
