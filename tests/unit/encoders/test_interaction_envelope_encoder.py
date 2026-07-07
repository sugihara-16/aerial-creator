from __future__ import annotations

from amsrr.encoders.interaction_envelope_encoder import InteractionEnvelopeEncoder, TOKEN_TYPE_IDS
from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.schemas.task_spec import TaskSpec
from amsrr.schemas.workspace import tensor_shape


def _grasp_carry_envelope(grasp_carry_dict: dict):
    task = TaskSpec.from_dict(grasp_carry_dict)
    irg = IRGBuilder().build(task)
    return InteractionEnvelopeExtractor().extract(irg)


def test_interaction_envelope_encoder_contract(grasp_carry_dict: dict) -> None:
    envelope = _grasp_carry_envelope(grasp_carry_dict)
    encoder = InteractionEnvelopeEncoder(d_model=12, max_tokens=32)
    output = encoder.encode(envelope)

    assert tensor_shape(output.tokens) == (1, 32, 12)
    assert tensor_shape(output.mask) == (1, 32)
    assert tensor_shape(output.token_type_ids) == (1, 32)
    assert tensor_shape(output.source_type_ids) == (1, 32)
    assert tensor_shape(output.source_ids) == (1, 32)
    assert output.group_slice == slice(0, 32)
    assert output.group_mask == output.mask
    assert output.metadata["backend_type"] == "mlp_embedding"

    valid_count = sum(output.mask[0])
    expected_count = (
        1
        + len(envelope.required_contact_modes)
        + len(envelope.target_region_sets)
        + len(envelope.wrench_space_requirements)
        + len(envelope.precision_requirements)
        + len(envelope.duration_requirements)
        + len(envelope.capability_requirements)
        + len(envelope.branch_options)
    )
    assert valid_count == expected_count
    assert output.token_type_ids[0][0] == TOKEN_TYPE_IDS["summary"]
    assert output.source_ids[0][0] == -1
    assert output.mask[0][valid_count] is False
    assert output.tokens[0][valid_count] == [0.0] * 12

    roundtrip = type(output).from_json(output.to_json())
    assert roundtrip.to_dict() == output.to_dict()


def test_interaction_envelope_encoder_batch_padding(grasp_carry_dict: dict) -> None:
    envelope = _grasp_carry_envelope(grasp_carry_dict)
    output = InteractionEnvelopeEncoder(d_model=8).encode_batch([envelope, envelope])

    assert tensor_shape(output.tokens)[0] == 2
    assert output.mask[0] == output.mask[1]
    assert output.token_type_ids[0] == output.token_type_ids[1]
