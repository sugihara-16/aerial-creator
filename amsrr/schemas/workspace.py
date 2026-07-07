from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from amsrr.schemas.common import SchemaBase, SchemaValidationError


REQUIRED_WORKSPACE_GROUPS = (
    "task",
    "geometry",
    "irg",
    "interaction_envelope",
    "morphology",
    "runtime",
    "capability",
)


def tensor_shape(value: Any) -> tuple[int, ...]:
    """Return a lightweight shape tuple for torch/numpy tensors or nested sequences."""

    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(dim) for dim in shape)
    if isinstance(value, (list, tuple)):
        if not value:
            return (0,)
        child_shape = tensor_shape(value[0])
        return (len(value),) + child_shape
    return ()


@dataclass
class LearnedQuerySpec(SchemaBase):
    query_name: str
    num_queries: int
    d_model: int
    allowed_token_groups: list[str]

    def validate(self) -> None:
        if self.query_name not in {"design", "high_level", "low_level", "critic", "feasibility"}:
            raise SchemaValidationError("LearnedQuerySpec.query_name is invalid")
        if self.num_queries <= 0:
            raise SchemaValidationError("LearnedQuerySpec.num_queries must be positive")
        if self.d_model <= 0:
            raise SchemaValidationError("LearnedQuerySpec.d_model must be positive")


@dataclass
class SharedInteractionWorkspace(SchemaBase):
    tokens: Any
    mask: Any
    token_type_ids: Any
    source_type_ids: Any
    source_ids: Any
    group_slices: dict[str, slice]
    group_masks: dict[str, Any]
    query_outputs: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        token_shape = tensor_shape(self.tokens)
        if len(token_shape) != 3:
            raise SchemaValidationError("SharedInteractionWorkspace.tokens must have shape [B, N_total, d_model]")
        batch, total_tokens, _ = token_shape
        expected_2d = (batch, total_tokens)
        for name in ("mask", "token_type_ids", "source_type_ids", "source_ids"):
            actual = tensor_shape(getattr(self, name))
            if actual != expected_2d:
                raise SchemaValidationError(f"SharedInteractionWorkspace.{name} must have shape {expected_2d}, got {actual}")
        missing_groups = [name for name in REQUIRED_WORKSPACE_GROUPS if name not in self.group_slices]
        if missing_groups:
            raise SchemaValidationError(f"SharedInteractionWorkspace.group_slices missing groups: {missing_groups}")
        for group_name, group_slice in self.group_slices.items():
            start = 0 if group_slice.start is None else group_slice.start
            stop = total_tokens if group_slice.stop is None else group_slice.stop
            if start < 0 or stop < start or stop > total_tokens:
                raise SchemaValidationError(f"SharedInteractionWorkspace.group_slices[{group_name!r}] is out of range")

