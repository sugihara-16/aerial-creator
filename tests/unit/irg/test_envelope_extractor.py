from __future__ import annotations

import pytest

from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.schemas.common import ContactMode
from amsrr.schemas.interaction_envelope import InteractionEnvelope
from amsrr.schemas.task_spec import TaskSpec
from tests.unit.irg.test_irg_builder import _free_flight_task, _locomotion_task, _perching_task, _valve_task


def test_interaction_envelope_extract(grasp_carry_dict: dict) -> None:
    task = TaskSpec.from_dict(grasp_carry_dict)
    irg = IRGBuilder().build(task)
    envelope = InteractionEnvelopeExtractor().extract(irg)

    assert envelope.task_id == "grasp_carry_box_001"
    assert envelope.required_contact_count_range == (2, 4)
    assert envelope.required_contact_modes == [ContactMode.GRASP, ContactMode.SUPPORT]
    assert len(envelope.target_region_sets) == 1
    assert envelope.target_region_sets[0].entity_id == "box_01"
    assert len(envelope.target_region_sets[0].region_ids) == 6
    assert set(envelope.target_region_sets[0].region_types) == {"face"}

    effects = {item.effect for item in envelope.wrench_space_requirements}
    assert {
        "inward_grasp_force",
        "frictional_no_slip_proxy",
        "payload_support_force",
        "object_pose_tracking_effect",
    }.issubset(effects)
    payload = next(item for item in envelope.wrench_space_requirements if item.effect == "payload_support_force")
    assert payload.wrench_lower == pytest.approx([0.0, 0.0, 9.80665, 0.0, 0.0, 0.0])

    object_pose_precision = next(item for item in envelope.precision_requirements if item.target == "object_pose")
    assert object_pose_precision.tolerance_pos_m == pytest.approx(0.05)
    assert object_pose_precision.tolerance_rot_rad == pytest.approx(0.20)
    assert envelope.duration_requirements[0].max_duration_s == pytest.approx(30.0)
    assert envelope.capability_requirements[0].capability_type == "grasp"

    roundtrip = InteractionEnvelope.from_json(envelope.to_json())
    assert roundtrip.to_dict() == envelope.to_dict()


def test_interaction_envelope_extracts_all_task_families(grasp_carry_dict: dict) -> None:
    tasks = [
        TaskSpec.from_dict(grasp_carry_dict),
        _free_flight_task(),
        _valve_task(),
        _perching_task(),
        _locomotion_task(),
    ]
    builder = IRGBuilder()
    extractor = InteractionEnvelopeExtractor()

    for task in tasks:
        envelope = extractor.extract(builder.build(task))
        assert envelope.task_id == task.task_id
        assert envelope.duration_requirements
        if task.task_type.value == "free_flight_navigation":
            assert envelope.required_contact_count_range == (0, 0)
            assert envelope.required_contact_modes == []
        else:
            assert envelope.required_contact_count_range[1] >= 1
            assert envelope.required_contact_modes
            assert envelope.target_region_sets
