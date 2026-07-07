from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

from amsrr.schemas.common import ContactMode, SchemaBase, SchemaValidationError
from amsrr.schemas.interaction_envelope import (
    CapabilityRequirement,
    DurationRequirement,
    EnvelopeBranchOption,
    InteractionEnvelope,
    PrecisionRequirement,
    TargetRegionSet,
    WrenchSpaceRequirement,
)
from amsrr.schemas.workspace import tensor_shape


TOKEN_TYPE_IDS = {
    "summary": 1,
    "contact_mode": 2,
    "target_region_set": 3,
    "wrench_requirement": 4,
    "precision_requirement": 5,
    "duration_requirement": 6,
    "capability_requirement": 7,
    "branch_option": 8,
}

SOURCE_TYPE_INTERACTION_ENVELOPE = 40
CONTACT_MODE_ORDER = list(ContactMode)


@dataclass
class InteractionEnvelopeEncoderOutput(SchemaBase):
    """Token group contract for the interaction_envelope workspace slice."""

    tokens: list[list[list[float]]]
    mask: list[list[bool]]
    token_type_ids: list[list[int]]
    source_type_ids: list[list[int]]
    source_ids: list[list[int]]
    group_slice: slice
    group_mask: list[list[bool]]
    metadata: dict[str, int | str | bool] = field(default_factory=dict)

    def validate(self) -> None:
        token_shape = tensor_shape(self.tokens)
        if len(token_shape) != 3:
            raise SchemaValidationError("InteractionEnvelopeEncoderOutput.tokens must have shape [B, N, d_model]")
        batch, total_tokens, _ = token_shape
        expected = (batch, total_tokens)
        for name in ("mask", "token_type_ids", "source_type_ids", "source_ids", "group_mask"):
            actual = tensor_shape(getattr(self, name))
            if actual != expected:
                raise SchemaValidationError(f"InteractionEnvelopeEncoderOutput.{name} must have shape {expected}, got {actual}")
        start = 0 if self.group_slice.start is None else self.group_slice.start
        stop = total_tokens if self.group_slice.stop is None else self.group_slice.stop
        if start < 0 or stop < start or stop > total_tokens:
            raise SchemaValidationError("InteractionEnvelopeEncoderOutput.group_slice is out of range")


@dataclass(frozen=True)
class _TokenRecord:
    token_type: str
    source_id: int
    features: list[float]


class InteractionEnvelopeEncoder:
    """Deterministic scalar-token encoder contract for InteractionEnvelope.

    This is the P0 contract layer: it creates padded token tensors, masks, token
    type ids, and source ids. Learned MLP parameters belong to later model code.
    """

    def __init__(
        self,
        *,
        d_model: int = 16,
        max_tokens: int | None = None,
        backend_type: str | None = None,
    ) -> None:
        if d_model <= 0:
            raise SchemaValidationError("InteractionEnvelopeEncoder.d_model must be positive")
        if max_tokens is not None and max_tokens <= 0:
            raise SchemaValidationError("InteractionEnvelopeEncoder.max_tokens must be positive")
        self.d_model = d_model
        self.max_tokens = max_tokens
        self.backend_type = backend_type or "mlp_embedding"

    def encode(self, envelope: InteractionEnvelope) -> InteractionEnvelopeEncoderOutput:
        return self.encode_batch([envelope])

    def encode_batch(self, envelopes: list[InteractionEnvelope]) -> InteractionEnvelopeEncoderOutput:
        if not envelopes:
            raise SchemaValidationError("InteractionEnvelopeEncoder.encode_batch requires at least one envelope")
        records_by_batch = [self._records_for_envelope(envelope) for envelope in envelopes]
        observed_tokens = max(len(records) for records in records_by_batch)
        total_tokens = self.max_tokens or observed_tokens
        if observed_tokens > total_tokens:
            raise SchemaValidationError(
                f"InteractionEnvelopeEncoder max_tokens={total_tokens} is smaller than required token count {observed_tokens}"
            )

        batch_tokens: list[list[list[float]]] = []
        batch_mask: list[list[bool]] = []
        batch_token_type_ids: list[list[int]] = []
        batch_source_type_ids: list[list[int]] = []
        batch_source_ids: list[list[int]] = []

        for records in records_by_batch:
            tokens = [_pad_features(record.features, self.d_model) for record in records]
            mask = [True] * len(records)
            token_type_ids = [TOKEN_TYPE_IDS[record.token_type] for record in records]
            source_type_ids = [SOURCE_TYPE_INTERACTION_ENVELOPE] * len(records)
            source_ids = [record.source_id for record in records]
            while len(tokens) < total_tokens:
                tokens.append([0.0] * self.d_model)
                mask.append(False)
                token_type_ids.append(0)
                source_type_ids.append(0)
                source_ids.append(-1)
            batch_tokens.append(tokens)
            batch_mask.append(mask)
            batch_token_type_ids.append(token_type_ids)
            batch_source_type_ids.append(source_type_ids)
            batch_source_ids.append(source_ids)

        return InteractionEnvelopeEncoderOutput(
            tokens=batch_tokens,
            mask=batch_mask,
            token_type_ids=batch_token_type_ids,
            source_type_ids=batch_source_type_ids,
            source_ids=batch_source_ids,
            group_slice=slice(0, total_tokens),
            group_mask=batch_mask,
            metadata={
                "encoder": "InteractionEnvelopeEncoder",
                "backend_type": self.backend_type,
                "d_model": self.d_model,
                "max_tokens": total_tokens,
                "padded": self.max_tokens is not None,
            },
        )

    def _records_for_envelope(self, envelope: InteractionEnvelope) -> list[_TokenRecord]:
        records = [
            _TokenRecord(
                "summary",
                -1,
                [
                    float(envelope.required_contact_count_range[0]),
                    float(envelope.required_contact_count_range[1]),
                    float(len(envelope.required_contact_modes)),
                    float(len(envelope.target_region_sets)),
                    float(len(envelope.wrench_space_requirements)),
                    float(len(envelope.precision_requirements)),
                    float(len(envelope.duration_requirements)),
                    float(len(envelope.capability_requirements)),
                    1.0 if envelope.support_ratio_requirements is not None else 0.0,
                    _optional_feature(envelope.vertical_thrust_ratio_limit),
                    _stable_scalar(envelope.task_id),
                ],
            )
        ]
        records.extend(
            _TokenRecord("contact_mode", idx, self._contact_mode_features(mode))
            for idx, mode in enumerate(envelope.required_contact_modes)
        )
        records.extend(
            _TokenRecord("target_region_set", idx, self._target_region_features(item))
            for idx, item in enumerate(envelope.target_region_sets)
        )
        records.extend(
            _TokenRecord("wrench_requirement", idx, self._wrench_features(item))
            for idx, item in enumerate(envelope.wrench_space_requirements)
        )
        records.extend(
            _TokenRecord("precision_requirement", idx, self._precision_features(item))
            for idx, item in enumerate(envelope.precision_requirements)
        )
        records.extend(
            _TokenRecord("duration_requirement", idx, self._duration_features(item))
            for idx, item in enumerate(envelope.duration_requirements)
        )
        records.extend(
            _TokenRecord("capability_requirement", idx, self._capability_features(item))
            for idx, item in enumerate(envelope.capability_requirements)
        )
        records.extend(
            _TokenRecord("branch_option", idx, self._branch_features(item))
            for idx, item in enumerate(envelope.branch_options)
        )
        return records

    @staticmethod
    def _contact_mode_features(mode: ContactMode) -> list[float]:
        return [1.0 if mode == candidate else 0.0 for candidate in CONTACT_MODE_ORDER] + [_stable_scalar(mode.value)]

    @staticmethod
    def _target_region_features(item: TargetRegionSet) -> list[float]:
        return [
            _stable_scalar(item.entity_id),
            float(len(item.region_ids)),
            float(len(item.region_types)),
            _mean_stable_scalar(item.region_ids),
            _mean_stable_scalar(item.region_types),
        ]

    @staticmethod
    def _wrench_features(item: WrenchSpaceRequirement) -> list[float]:
        lower_norm = _norm_or_zero(item.wrench_lower)
        upper_norm = _norm_or_zero(item.wrench_upper)
        target_norm = _norm_or_zero(item.target_wrench)
        return [
            _stable_scalar(item.applies_to),
            _stable_scalar(item.effect),
            lower_norm,
            upper_norm,
            target_norm,
            1.0 if item.wrench_lower is not None else 0.0,
            1.0 if item.wrench_upper is not None else 0.0,
            1.0 if item.target_wrench is not None else 0.0,
            float(item.priority),
        ]

    @staticmethod
    def _precision_features(item: PrecisionRequirement) -> list[float]:
        q_values = item.tolerance_q or []
        return [
            _stable_scalar(item.target),
            _optional_feature(item.tolerance_pos_m),
            _optional_feature(item.tolerance_rot_rad),
            float(len(q_values)),
            _norm_or_zero(q_values),
        ]

    @staticmethod
    def _duration_features(item: DurationRequirement) -> list[float]:
        return [
            _stable_scalar(item.phase_label or "__task__"),
            _optional_feature(item.min_duration_s),
            _optional_feature(item.max_duration_s),
        ]

    @staticmethod
    def _capability_features(item: CapabilityRequirement) -> list[float]:
        return [
            _stable_scalar(item.capability_type),
            _optional_feature(item.min_force_n),
            _optional_feature(item.min_torque_nm),
            _optional_feature(item.pose_accuracy_m),
            _optional_feature(item.pose_accuracy_rad),
            _optional_feature(item.stiffness_requirement),
        ]

    @staticmethod
    def _branch_features(item: EnvelopeBranchOption) -> list[float]:
        return [
            _stable_scalar(item.branch_id),
            _stable_scalar(item.label),
            float(len(item.required_contact_modes)),
            _mean_stable_scalar([mode.value for mode in item.required_contact_modes]),
        ]


def _stable_scalar(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)


def _mean_stable_scalar(values: list[str]) -> float:
    if not values:
        return 0.0
    return sum(_stable_scalar(value) for value in values) / float(len(values))


def _optional_feature(value: float | None) -> float:
    return 0.0 if value is None else float(value)


def _norm_or_zero(values: list[float] | None) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(float(item) * float(item) for item in values))


def _pad_features(values: list[float], d_model: int) -> list[float]:
    if len(values) >= d_model:
        return [float(item) for item in values[:d_model]]
    return [float(item) for item in values] + [0.0] * (d_model - len(values))
