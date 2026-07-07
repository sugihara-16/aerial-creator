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
OPTIONAL_WORKSPACE_GROUPS = ("contact_candidates",)
WORKSPACE_GROUPS = REQUIRED_WORKSPACE_GROUPS + OPTIONAL_WORKSPACE_GROUPS


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
        if not self.allowed_token_groups:
            raise SchemaValidationError("LearnedQuerySpec.allowed_token_groups must not be empty")
        unknown_groups = [name for name in self.allowed_token_groups if name not in WORKSPACE_GROUPS]
        if unknown_groups:
            raise SchemaValidationError(f"LearnedQuerySpec.allowed_token_groups has unknown groups: {unknown_groups}")


@dataclass
class WorkspaceTokenGroup(SchemaBase):
    group_name: str
    tokens: Any
    mask: Any
    token_type_ids: Any
    source_type_ids: Any
    source_ids: Any
    d_model: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.group_name not in WORKSPACE_GROUPS:
            raise SchemaValidationError(f"WorkspaceTokenGroup.group_name is unknown: {self.group_name!r}")
        token_shape = tensor_shape(self.tokens)
        if len(token_shape) == 3:
            batch, total_tokens, inferred_d_model = token_shape
            if self.d_model is not None and self.d_model != inferred_d_model:
                raise SchemaValidationError("WorkspaceTokenGroup.d_model does not match token feature dimension")
        elif len(token_shape) == 2 and token_shape[1] == 0 and self.d_model is not None:
            batch, total_tokens = token_shape
            inferred_d_model = self.d_model
        else:
            raise SchemaValidationError("WorkspaceTokenGroup.tokens must have shape [B, N, d_model]")
        if inferred_d_model <= 0:
            raise SchemaValidationError("WorkspaceTokenGroup.d_model must be positive")
        expected_2d = (batch, total_tokens)
        for name in ("mask", "token_type_ids", "source_type_ids", "source_ids"):
            actual = tensor_shape(getattr(self, name))
            if actual != expected_2d:
                raise SchemaValidationError(f"WorkspaceTokenGroup.{name} must have shape {expected_2d}, got {actual}")


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
            if group_name not in WORKSPACE_GROUPS:
                raise SchemaValidationError(f"SharedInteractionWorkspace.group_slices has unknown group {group_name!r}")
            if group_slice.step not in (None, 1):
                raise SchemaValidationError(f"SharedInteractionWorkspace.group_slices[{group_name!r}] step must be None or 1")
            start = 0 if group_slice.start is None else group_slice.start
            stop = total_tokens if group_slice.stop is None else group_slice.stop
            if start < 0 or stop < start or stop > total_tokens:
                raise SchemaValidationError(f"SharedInteractionWorkspace.group_slices[{group_name!r}] is out of range")
            if group_name not in self.group_masks:
                raise SchemaValidationError(f"SharedInteractionWorkspace.group_masks missing group {group_name!r}")
            expected_group_shape = (batch, stop - start)
            actual_group_shape = tensor_shape(self.group_masks[group_name])
            if actual_group_shape != expected_group_shape:
                raise SchemaValidationError(
                    f"SharedInteractionWorkspace.group_masks[{group_name!r}] must have shape {expected_group_shape}, got {actual_group_shape}"
                )
            if isinstance(self.mask, list) and isinstance(self.group_masks[group_name], list):
                for batch_idx in range(batch):
                    if self.group_masks[group_name][batch_idx] != self.mask[batch_idx][start:stop]:
                        raise SchemaValidationError(
                            f"SharedInteractionWorkspace.group_masks[{group_name!r}] does not match workspace mask slice"
                        )
        if self.query_outputs is not None:
            for query_name, query_output in self.query_outputs.items():
                if query_name not in {"design", "high_level", "low_level", "critic", "feasibility"}:
                    raise SchemaValidationError(f"SharedInteractionWorkspace.query_outputs has unknown query {query_name!r}")
                output_shape = tensor_shape(query_output)
                if len(output_shape) != 3 or output_shape[0] != batch:
                    raise SchemaValidationError(
                        f"SharedInteractionWorkspace.query_outputs[{query_name!r}] must have shape [B, Q, d_model]"
                    )


def recommended_learned_query_specs(d_model: int, *, num_queries: int = 1) -> list[LearnedQuerySpec]:
    if d_model <= 0:
        raise SchemaValidationError("recommended_learned_query_specs.d_model must be positive")
    if num_queries <= 0:
        raise SchemaValidationError("recommended_learned_query_specs.num_queries must be positive")
    return [
        LearnedQuerySpec(
            query_name="design",
            num_queries=num_queries,
            d_model=d_model,
            allowed_token_groups=["task", "geometry", "irg", "interaction_envelope", "morphology", "capability"],
        ),
        LearnedQuerySpec(
            query_name="high_level",
            num_queries=num_queries,
            d_model=d_model,
            allowed_token_groups=[
                "task",
                "geometry",
                "irg",
                "interaction_envelope",
                "morphology",
                "contact_candidates",
                "runtime",
                "capability",
            ],
        ),
        LearnedQuerySpec(
            query_name="low_level",
            num_queries=num_queries,
            d_model=d_model,
            allowed_token_groups=["interaction_envelope", "morphology", "runtime", "capability"],
        ),
        LearnedQuerySpec(
            query_name="critic",
            num_queries=num_queries,
            d_model=d_model,
            allowed_token_groups=list(WORKSPACE_GROUPS),
        ),
        LearnedQuerySpec(
            query_name="feasibility",
            num_queries=num_queries,
            d_model=d_model,
            allowed_token_groups=["geometry", "irg", "interaction_envelope", "morphology", "contact_candidates", "capability"],
        ),
    ]
