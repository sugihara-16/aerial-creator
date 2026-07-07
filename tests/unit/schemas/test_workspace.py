from __future__ import annotations

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.workspace import SharedInteractionWorkspace


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


def test_shared_interaction_workspace_tensor_shapes() -> None:
    workspace = SharedInteractionWorkspace(
        tokens=[[[0.1, 0.2], [0.3, 0.4]]],
        mask=[[True, True]],
        token_type_ids=[[1, 2]],
        source_type_ids=[[10, 20]],
        source_ids=[[100, 200]],
        group_slices=_group_slices(),
        group_masks={},
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
            group_masks={},
        )

