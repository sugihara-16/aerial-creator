from __future__ import annotations

from typing import Any

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.workspace import (
    REQUIRED_WORKSPACE_GROUPS,
    WORKSPACE_GROUPS,
    SharedInteractionWorkspace,
    WorkspaceTokenGroup,
    tensor_shape,
)


class SharedInteractionWorkspaceBuilder:
    """Assemble per-modality token groups into the shared workspace contract."""

    def __init__(self, *, d_model: int, include_contact_candidates: bool = False) -> None:
        if d_model <= 0:
            raise SchemaValidationError("SharedInteractionWorkspaceBuilder.d_model must be positive")
        self.d_model = d_model
        self.group_order = list(REQUIRED_WORKSPACE_GROUPS)
        if include_contact_candidates:
            self.group_order.append("contact_candidates")

    def build(
        self,
        groups: dict[str, WorkspaceTokenGroup | Any],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> SharedInteractionWorkspace:
        normalized = {
            name: _coerce_group(name, group, self.d_model)
            for name, group in groups.items()
        }
        unknown = [name for name in normalized if name not in WORKSPACE_GROUPS]
        if unknown:
            raise SchemaValidationError(f"Unknown workspace token groups: {unknown}")
        missing_required = [name for name in REQUIRED_WORKSPACE_GROUPS if name not in self.group_order]
        if missing_required:
            raise SchemaValidationError(f"Workspace builder group order missing required groups: {missing_required}")

        batch_size = self._infer_batch_size(normalized)
        all_groups: dict[str, WorkspaceTokenGroup] = {}
        for name in self.group_order:
            all_groups[name] = normalized.get(name) or empty_workspace_token_group(
                name,
                batch_size=batch_size,
                d_model=self.d_model,
            )

        tokens: list[list[list[float]]] = [[] for _ in range(batch_size)]
        mask: list[list[bool]] = [[] for _ in range(batch_size)]
        token_type_ids: list[list[int]] = [[] for _ in range(batch_size)]
        source_type_ids: list[list[int]] = [[] for _ in range(batch_size)]
        source_ids: list[list[int]] = [[] for _ in range(batch_size)]
        group_slices: dict[str, slice] = {}
        group_masks: dict[str, list[list[bool]]] = {}

        cursor = 0
        for name in self.group_order:
            group = all_groups[name]
            self._validate_group_compatibility(group, batch_size)
            width = tensor_shape(group.mask)[1]
            group_slices[name] = slice(cursor, cursor + width)
            group_masks[name] = _copy_2d_bool(group.mask)
            for batch_idx in range(batch_size):
                tokens[batch_idx].extend(_copy_2d_tokens(group.tokens, batch_idx))
                mask[batch_idx].extend(list(group.mask[batch_idx]))
                token_type_ids[batch_idx].extend(list(group.token_type_ids[batch_idx]))
                source_type_ids[batch_idx].extend(list(group.source_type_ids[batch_idx]))
                source_ids[batch_idx].extend(list(group.source_ids[batch_idx]))
            cursor += width

        return SharedInteractionWorkspace(
            tokens=tokens,
            mask=mask,
            token_type_ids=token_type_ids,
            source_type_ids=source_type_ids,
            source_ids=source_ids,
            group_slices=group_slices,
            group_masks=group_masks,
            metadata={
                "workspace_builder": "SharedInteractionWorkspaceBuilder",
                "d_model": self.d_model,
                "group_order": list(self.group_order),
                **(metadata or {}),
            },
        )

    def _infer_batch_size(self, groups: dict[str, WorkspaceTokenGroup]) -> int:
        batch_size: int | None = None
        for group in groups.values():
            shape = tensor_shape(group.mask)
            if len(shape) != 2:
                raise SchemaValidationError(f"WorkspaceTokenGroup {group.group_name!r} mask must be rank 2")
            if batch_size is None:
                batch_size = shape[0]
            elif batch_size != shape[0]:
                raise SchemaValidationError("All workspace token groups must have the same batch size")
        return batch_size or 1

    def _validate_group_compatibility(self, group: WorkspaceTokenGroup, batch_size: int) -> None:
        shape = tensor_shape(group.tokens)
        if len(shape) == 3:
            group_batch, _, group_d_model = shape
        elif len(shape) == 2 and shape[1] == 0 and group.d_model is not None:
            group_batch, group_d_model = shape[0], group.d_model
        else:
            raise SchemaValidationError(f"WorkspaceTokenGroup {group.group_name!r} tokens must have rank 3")
        if group_batch != batch_size:
            raise SchemaValidationError(f"WorkspaceTokenGroup {group.group_name!r} batch size mismatch")
        if group_d_model != self.d_model:
            raise SchemaValidationError(f"WorkspaceTokenGroup {group.group_name!r} d_model mismatch")


def empty_workspace_token_group(group_name: str, *, batch_size: int, d_model: int) -> WorkspaceTokenGroup:
    if batch_size <= 0:
        raise SchemaValidationError("empty_workspace_token_group.batch_size must be positive")
    if d_model <= 0:
        raise SchemaValidationError("empty_workspace_token_group.d_model must be positive")
    return WorkspaceTokenGroup(
        group_name=group_name,
        tokens=[[] for _ in range(batch_size)],
        mask=[[] for _ in range(batch_size)],
        token_type_ids=[[] for _ in range(batch_size)],
        source_type_ids=[[] for _ in range(batch_size)],
        source_ids=[[] for _ in range(batch_size)],
        d_model=d_model,
        metadata={"empty": True},
    )


def workspace_token_group_from_encoder_output(
    group_name: str,
    encoder_output: Any,
    *,
    d_model: int | None = None,
) -> WorkspaceTokenGroup:
    token_shape = tensor_shape(encoder_output.tokens)
    inferred_d_model = d_model
    if len(token_shape) == 3:
        inferred_d_model = token_shape[2]
    if inferred_d_model is None:
        raise SchemaValidationError("Cannot infer d_model from encoder output")
    return WorkspaceTokenGroup(
        group_name=group_name,
        tokens=encoder_output.tokens,
        mask=encoder_output.mask,
        token_type_ids=encoder_output.token_type_ids,
        source_type_ids=encoder_output.source_type_ids,
        source_ids=encoder_output.source_ids,
        d_model=inferred_d_model,
        metadata=getattr(encoder_output, "metadata", {}),
    )


def _coerce_group(group_name: str, group: WorkspaceTokenGroup | Any, d_model: int) -> WorkspaceTokenGroup:
    if isinstance(group, WorkspaceTokenGroup):
        if group.group_name != group_name:
            raise SchemaValidationError(
                f"Workspace group dict key {group_name!r} does not match group_name {group.group_name!r}"
            )
        return group
    return workspace_token_group_from_encoder_output(group_name, group, d_model=d_model)


def _copy_2d_bool(value: Any) -> list[list[bool]]:
    return [[bool(item) for item in row] for row in value]


def _copy_2d_tokens(tokens: Any, batch_idx: int) -> list[list[float]]:
    return [[float(item) for item in token] for token in tokens[batch_idx]]
