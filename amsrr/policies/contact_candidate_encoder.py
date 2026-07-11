from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

from amsrr.schemas.common import ContactMode, SchemaBase, SchemaValidationError
from amsrr.schemas.contact_candidates import ContactCandidate, ContactCandidateSet
from amsrr.schemas.workspace import tensor_shape


CONTACT_CANDIDATE_TOKEN_TYPE = 1
CONTACT_GROUP_TOKEN_TYPE = 2
SOURCE_TYPE_CONTACT_CANDIDATE = 50
SOURCE_TYPE_CONTACT_GROUP = 51

CONTACT_MODE_ORDER = list(ContactMode)
GROUP_TYPE_ORDER = [
    "grasp_pair",
    "multi_grasp",
    "perch_set",
    "support_set",
    "locomotion_stance",
]

CONTACT_CANDIDATE_FEATURE_NAMES = [
    "contact_pose_x",
    "contact_pose_y",
    "contact_pose_z",
    "normal_x",
    "normal_y",
    "normal_z",
    "friction",
    "friction_known",
    "patch_area_m2",
    "unary_valid",
    "candidate_mask",
    "unary_violation_count",
    "candidate_score_count",
    "candidate_score_mean",
    "candidate_score_max",
    "pairwise_conflict_fraction",
    "pairwise_compatibility_mean",
    "group_membership_count",
    "slot_coverage_count",
    "target_entity_hash",
    "region_hash",
] + [f"contact_mode_{mode.value}" for mode in CONTACT_MODE_ORDER]

CONTACT_GROUP_FEATURE_NAMES = [
    "group_score",
    "group_size",
    "group_violation_count",
    "known_member_fraction",
    "valid_member_fraction",
    "pairwise_conflict_fraction",
    "pairwise_compatibility_mean",
] + [f"group_type_{group_type}" for group_type in GROUP_TYPE_ORDER]

DEFAULT_CONTACT_CANDIDATE_D_MODEL = 48


@dataclass
class ContactCandidateEncoderOutput(SchemaBase):
    """Padded contact-candidate/group tokens with original source IDs.

    Candidate tokens retain the schema ``candidate_id`` in ``source_ids``.
    Group IDs are strings, so their source-id slot is ``-1`` and the exact IDs
    and member candidate IDs are carried separately.
    """

    tokens: list[list[list[float]]]
    mask: list[list[bool]]
    token_type_ids: list[list[int]]
    source_type_ids: list[list[int]]
    source_ids: list[list[int]]
    candidate_counts: list[int]
    group_counts: list[int]
    candidate_ids: list[list[int]]
    group_ids: list[list[str | None]]
    group_candidate_ids: list[list[list[int]]]
    candidate_width: int
    group_width: int
    d_model: int
    metadata: dict[str, int | str | bool] = field(default_factory=dict)

    def validate(self) -> None:
        shape = tensor_shape(self.tokens)
        if len(shape) != 3:
            raise SchemaValidationError(
                "ContactCandidateEncoderOutput.tokens must have shape [B, N, d_model]"
            )
        batch_size, total_width, feature_width = shape
        if feature_width != self.d_model or self.d_model <= 0:
            raise SchemaValidationError(
                "ContactCandidateEncoderOutput.d_model must match token feature width"
            )
        if total_width != self.candidate_width + self.group_width:
            raise SchemaValidationError(
                "ContactCandidateEncoderOutput token width must match candidate/group widths"
            )
        expected = (batch_size, total_width)
        for name in ("mask", "token_type_ids", "source_type_ids", "source_ids"):
            actual = tensor_shape(getattr(self, name))
            if actual != expected:
                raise SchemaValidationError(
                    f"ContactCandidateEncoderOutput.{name} must have shape {expected}, got {actual}"
                )
        if len(self.candidate_counts) != batch_size or len(self.group_counts) != batch_size:
            raise SchemaValidationError(
                "ContactCandidateEncoderOutput counts must match batch size"
            )
        if tensor_shape(self.candidate_ids) != (batch_size, self.candidate_width):
            raise SchemaValidationError(
                "ContactCandidateEncoderOutput.candidate_ids has an invalid shape"
            )
        if tensor_shape(self.group_ids) != (batch_size, self.group_width):
            raise SchemaValidationError(
                "ContactCandidateEncoderOutput.group_ids has an invalid shape"
            )
        if len(self.group_candidate_ids) != batch_size or any(
            len(row) != self.group_width for row in self.group_candidate_ids
        ):
            raise SchemaValidationError(
                "ContactCandidateEncoderOutput.group_candidate_ids has an invalid shape"
            )
        if any(count < 0 or count > self.candidate_width for count in self.candidate_counts):
            raise SchemaValidationError(
                "ContactCandidateEncoderOutput.candidate_counts is out of range"
            )
        if any(count < 0 or count > self.group_width for count in self.group_counts):
            raise SchemaValidationError(
                "ContactCandidateEncoderOutput.group_counts is out of range"
            )

    def candidate_tokens(self, batch_index: int = 0) -> list[list[float]]:
        count = self.candidate_counts[batch_index]
        return self.tokens[batch_index][:count]

    def candidate_valid_mask(self, batch_index: int = 0) -> list[bool]:
        count = self.candidate_counts[batch_index]
        return self.mask[batch_index][:count]

    def group_tokens(self, batch_index: int = 0) -> list[list[float]]:
        count = self.group_counts[batch_index]
        start = self.candidate_width
        return self.tokens[batch_index][start : start + count]

    def group_valid_mask(self, batch_index: int = 0) -> list[bool]:
        count = self.group_counts[batch_index]
        start = self.candidate_width
        return self.mask[batch_index][start : start + count]


class ContactCandidateEncoder:
    """Deterministic Version-1 tokenization for learned pi_H ranking heads."""

    def __init__(self, *, d_model: int = DEFAULT_CONTACT_CANDIDATE_D_MODEL) -> None:
        minimum_width = max(
            len(CONTACT_CANDIDATE_FEATURE_NAMES),
            len(CONTACT_GROUP_FEATURE_NAMES) + len(CONTACT_CANDIDATE_FEATURE_NAMES),
        )
        if d_model < minimum_width:
            raise SchemaValidationError(
                f"ContactCandidateEncoder.d_model must be at least {minimum_width}"
            )
        self.d_model = d_model

    def encode(self, candidate_set: ContactCandidateSet) -> ContactCandidateEncoderOutput:
        return self.encode_batch([candidate_set])

    def encode_batch(
        self,
        candidate_sets: list[ContactCandidateSet],
    ) -> ContactCandidateEncoderOutput:
        if not candidate_sets:
            raise SchemaValidationError(
                "ContactCandidateEncoder.encode_batch requires at least one candidate set"
            )
        for candidate_set in candidate_sets:
            candidate_set.validate()
            candidate_ids = [candidate.candidate_id for candidate in candidate_set.candidates]
            if not candidate_ids:
                raise SchemaValidationError(
                    "ContactCandidateEncoder requires at least one contact candidate"
                )
            if len(candidate_ids) != len(set(candidate_ids)):
                raise SchemaValidationError(
                    "ContactCandidateEncoder requires unique ContactCandidate.candidate_id values"
                )
            group_ids = [group.group_id for group in candidate_set.group_proposals]
            if len(group_ids) != len(set(group_ids)):
                raise SchemaValidationError(
                    "ContactCandidateEncoder requires unique group proposal IDs"
                )

        candidate_width = max(len(candidate_set.candidates) for candidate_set in candidate_sets)
        group_width = max(len(candidate_set.group_proposals) for candidate_set in candidate_sets)
        total_width = candidate_width + group_width

        batch_tokens: list[list[list[float]]] = []
        batch_mask: list[list[bool]] = []
        batch_token_types: list[list[int]] = []
        batch_source_types: list[list[int]] = []
        batch_source_ids: list[list[int]] = []
        batch_candidate_ids: list[list[int]] = []
        batch_group_ids: list[list[str | None]] = []
        batch_group_candidate_ids: list[list[list[int]]] = []
        candidate_counts: list[int] = []
        group_counts: list[int] = []

        for candidate_set in candidate_sets:
            candidate_counts.append(len(candidate_set.candidates))
            group_counts.append(len(candidate_set.group_proposals))
            candidate_by_id = {
                candidate.candidate_id: candidate for candidate in candidate_set.candidates
            }
            candidate_index_by_id = {
                candidate.candidate_id: index
                for index, candidate in enumerate(candidate_set.candidates)
            }
            candidate_features = [
                self._candidate_features(
                    candidate_set,
                    candidate,
                    candidate_index=index,
                )
                for index, candidate in enumerate(candidate_set.candidates)
            ]
            group_features = [
                self._group_features(
                    candidate_set,
                    group_index=index,
                    candidate_by_id=candidate_by_id,
                    candidate_index_by_id=candidate_index_by_id,
                    candidate_features=candidate_features,
                )
                for index, _ in enumerate(candidate_set.group_proposals)
            ]
            group_masks = [
                self._group_is_valid(
                    candidate_set,
                    group_index=index,
                    candidate_by_id=candidate_by_id,
                    candidate_index_by_id=candidate_index_by_id,
                )
                for index, _ in enumerate(candidate_set.group_proposals)
            ]

            tokens = [_pad(features, self.d_model) for features in candidate_features]
            mask = [
                bool(candidate_set.candidate_mask[index] and candidate.unary_valid)
                for index, candidate in enumerate(candidate_set.candidates)
            ]
            token_types = [CONTACT_CANDIDATE_TOKEN_TYPE] * len(candidate_set.candidates)
            source_types = [SOURCE_TYPE_CONTACT_CANDIDATE] * len(candidate_set.candidates)
            source_ids = [candidate.candidate_id for candidate in candidate_set.candidates]
            candidate_ids = list(source_ids)
            while len(tokens) < candidate_width:
                tokens.append([0.0] * self.d_model)
                mask.append(False)
                token_types.append(0)
                source_types.append(0)
                source_ids.append(-1)
                candidate_ids.append(-1)

            tokens.extend(_pad(features, self.d_model) for features in group_features)
            mask.extend(group_masks)
            token_types.extend([CONTACT_GROUP_TOKEN_TYPE] * len(group_features))
            source_types.extend([SOURCE_TYPE_CONTACT_GROUP] * len(group_features))
            source_ids.extend([-1] * len(group_features))
            group_ids: list[str | None] = [
                group.group_id for group in candidate_set.group_proposals
            ]
            group_candidate_ids = [
                list(group.candidate_ids) for group in candidate_set.group_proposals
            ]
            while len(group_ids) < group_width:
                tokens.append([0.0] * self.d_model)
                mask.append(False)
                token_types.append(0)
                source_types.append(0)
                source_ids.append(-1)
                group_ids.append(None)
                group_candidate_ids.append([])

            if len(tokens) != total_width:
                raise SchemaValidationError(
                    "ContactCandidateEncoder internal padding width mismatch"
                )
            if not all(math.isfinite(value) for token in tokens for value in token):
                raise SchemaValidationError(
                    "ContactCandidateEncoder produced a non-finite token"
                )
            batch_tokens.append(tokens)
            batch_mask.append(mask)
            batch_token_types.append(token_types)
            batch_source_types.append(source_types)
            batch_source_ids.append(source_ids)
            batch_candidate_ids.append(candidate_ids)
            batch_group_ids.append(group_ids)
            batch_group_candidate_ids.append(group_candidate_ids)

        return ContactCandidateEncoderOutput(
            tokens=batch_tokens,
            mask=batch_mask,
            token_type_ids=batch_token_types,
            source_type_ids=batch_source_types,
            source_ids=batch_source_ids,
            candidate_counts=candidate_counts,
            group_counts=group_counts,
            candidate_ids=batch_candidate_ids,
            group_ids=batch_group_ids,
            group_candidate_ids=batch_group_candidate_ids,
            candidate_width=candidate_width,
            group_width=group_width,
            d_model=self.d_model,
            metadata={
                "encoder": "ContactCandidateEncoder",
                "backend_type": "deterministic_feature_tokens",
                "candidate_feature_count": len(CONTACT_CANDIDATE_FEATURE_NAMES),
                "group_feature_count": len(CONTACT_GROUP_FEATURE_NAMES),
                "preserves_original_candidate_ids": True,
            },
        )

    def _candidate_features(
        self,
        candidate_set: ContactCandidateSet,
        candidate: ContactCandidate,
        *,
        candidate_index: int,
    ) -> list[float]:
        score_values = [float(value) for value in candidate.candidate_scores.values()]
        conflict_row = candidate_set.pairwise_conflict_matrix[candidate_index]
        compatibility_row = candidate_set.pairwise_compatibility_score[candidate_index]
        denominator = float(max(1, len(candidate_set.candidates) - 1))
        group_membership_count = sum(
            1
            for group in candidate_set.group_proposals
            if candidate.candidate_id in group.candidate_ids
        )
        slot_coverage_count = len(candidate_set.slot_coverage.get(candidate.slot_id, []))
        values = [
            float(candidate.contact_pose_world[0]),
            float(candidate.contact_pose_world[1]),
            float(candidate.contact_pose_world[2]),
            float(candidate.normal_world[0]),
            float(candidate.normal_world[1]),
            float(candidate.normal_world[2]),
            0.0 if candidate.friction is None else float(candidate.friction),
            0.0 if candidate.friction is None else 1.0,
            float(candidate.patch_area_m2),
            1.0 if candidate.unary_valid else 0.0,
            1.0 if candidate_set.candidate_mask[candidate_index] else 0.0,
            float(len(candidate.unary_violation_codes)),
            float(len(score_values)),
            sum(score_values) / float(len(score_values)) if score_values else 0.0,
            max(score_values) if score_values else 0.0,
            sum(1.0 for value in conflict_row if value) / denominator,
            sum(float(value) for value in compatibility_row)
            / float(max(1, len(compatibility_row))),
            float(group_membership_count),
            float(slot_coverage_count),
            _stable_scalar(candidate.target_entity_id),
            _stable_scalar(candidate.region_id),
        ]
        values.extend(
            1.0 if candidate.contact_mode == mode else 0.0
            for mode in CONTACT_MODE_ORDER
        )
        return values

    def _group_features(
        self,
        candidate_set: ContactCandidateSet,
        *,
        group_index: int,
        candidate_by_id: dict[int, ContactCandidate],
        candidate_index_by_id: dict[int, int],
        candidate_features: list[list[float]],
    ) -> list[float]:
        group = candidate_set.group_proposals[group_index]
        known_ids = [
            candidate_id
            for candidate_id in group.candidate_ids
            if candidate_id in candidate_by_id
        ]
        member_count = max(1, len(group.candidate_ids))
        valid_count = sum(
            1
            for candidate_id in known_ids
            if candidate_by_id[candidate_id].unary_valid
            and candidate_set.candidate_mask[candidate_index_by_id[candidate_id]]
        )
        conflict_fraction, compatibility_mean = _group_pairwise_summary(
            candidate_set,
            known_ids,
            candidate_index_by_id,
        )
        aggregate = _mean_features(
            [candidate_features[candidate_index_by_id[candidate_id]] for candidate_id in known_ids],
            len(CONTACT_CANDIDATE_FEATURE_NAMES),
        )
        values = [
            float(group.group_score),
            float(len(group.candidate_ids)),
            float(len(group.group_violation_codes)),
            float(len(known_ids)) / float(member_count),
            float(valid_count) / float(member_count),
            conflict_fraction,
            compatibility_mean,
        ]
        values.extend(1.0 if group.group_type == name else 0.0 for name in GROUP_TYPE_ORDER)
        values.extend(aggregate)
        return values

    def _group_is_valid(
        self,
        candidate_set: ContactCandidateSet,
        *,
        group_index: int,
        candidate_by_id: dict[int, ContactCandidate],
        candidate_index_by_id: dict[int, int],
    ) -> bool:
        group = candidate_set.group_proposals[group_index]
        if not group.candidate_ids or group.group_violation_codes:
            return False
        if len(group.candidate_ids) != len(set(group.candidate_ids)):
            return False
        for candidate_id in group.candidate_ids:
            candidate = candidate_by_id.get(candidate_id)
            if candidate is None:
                return False
            candidate_index = candidate_index_by_id[candidate_id]
            if not candidate.unary_valid or not candidate_set.candidate_mask[candidate_index]:
                return False
        conflict_fraction, _ = _group_pairwise_summary(
            candidate_set,
            group.candidate_ids,
            candidate_index_by_id,
        )
        return conflict_fraction == 0.0


def _group_pairwise_summary(
    candidate_set: ContactCandidateSet,
    candidate_ids: list[int],
    candidate_index_by_id: dict[int, int],
) -> tuple[float, float]:
    pairs: list[tuple[bool, float]] = []
    unique_ids = list(dict.fromkeys(candidate_ids))
    for left_position, left_id in enumerate(unique_ids):
        left_index = candidate_index_by_id[left_id]
        for right_id in unique_ids[left_position + 1 :]:
            right_index = candidate_index_by_id[right_id]
            pairs.append(
                (
                    bool(candidate_set.pairwise_conflict_matrix[left_index][right_index]),
                    float(candidate_set.pairwise_compatibility_score[left_index][right_index]),
                )
            )
    if not pairs:
        return 0.0, 1.0
    return (
        sum(1.0 for conflict, _ in pairs if conflict) / float(len(pairs)),
        sum(score for _, score in pairs) / float(len(pairs)),
    )


def _mean_features(rows: list[list[float]], width: int) -> list[float]:
    if not rows:
        return [0.0] * width
    return [
        sum(float(row[index]) for row in rows) / float(len(rows))
        for index in range(width)
    ]


def _pad(values: list[float], width: int) -> list[float]:
    if len(values) > width:
        raise SchemaValidationError(
            f"ContactCandidateEncoder feature width {len(values)} exceeds d_model={width}"
        )
    return [float(value) for value in values] + [0.0] * (width - len(values))


def _stable_scalar(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)
