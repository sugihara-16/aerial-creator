from __future__ import annotations

from pathlib import Path

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.schemas.policies import ContactWrenchTrajectory
from amsrr.schemas.task_spec import TaskSpec
from amsrr.simulation.order9_object_task_runtime import ORDER9_OBJECT_TASK_PHASES
from amsrr.training.order9_c3_nominal_trajectory import (
    materialize_order9_c3_complete_task_phases,
    validate_order9_c3_nominal_trajectory_artifact_bytes,
)
from amsrr.training.order9_r1_complete_task_audit import (
    E_RETREAT_OFFSET,
    ORDER9_R1_COMPLETE_TASK_AUDIT_VERSION,
    audit_order9_r1_complete_task_artifact,
    audit_order9_r1_materializer_parameter_effect,
)
from amsrr.training.order9_r1_nominal_retreat import (
    ORDER9_R1_NOMINAL_RETREAT_VERSION,
    materialize_order9_r1_complete_task_phases,
)

REPOSITORY = Path(__file__).resolve().parents[3]
CASE_ROOT = REPOSITORY / (
    "artifacts/p4_full/order9/r1_teacher/diagnostics/"
    "r1_v6_short_retreat_profile_v1/"
    "r1_l1_10mm_5deg__train__train-000013-42f605dc0437__lattice_00"
)
ARTIFACT = CASE_ROOT / (
    "nominal_set/buckets/"
    "r1_l1_10mm_5deg__train__train-000013-42f605dc0437__lattice_00/manifest.json"
)


def _inputs() -> tuple[TaskSpec, dict[str, ContactWrenchTrajectory]]:
    task = TaskSpec.from_json(
        (CASE_ROOT / "task_spec.json").read_text(encoding="utf-8")
    )
    artifact = validate_order9_c3_nominal_trajectory_artifact_bytes(ARTIFACT)
    entries = {value.phase: value for value in artifact.phase_trajectories}
    phases = {
        phase: ContactWrenchTrajectory.from_json(
            (ARTIFACT.parent / entries[phase].trajectory_path).read_text(
                encoding="utf-8"
            )
        )
        for phase in ("approach", "contact_acquisition")
    }
    return task, phases


def test_historical_whole_approach_retreat_is_rejected_before_isaac() -> None:
    task, inputs = _inputs()
    phases = materialize_order9_c3_complete_task_phases(
        phase_trajectories=inputs,
        task_spec=task,
        lift_clearance_m=0.3,
        retreat_offset_m=0.1,
        phase_duration_s={phase.value: 1.0 for phase in ORDER9_OBJECT_TASK_PHASES},
    )

    with pytest.raises(SchemaValidationError, match=E_RETREAT_OFFSET):
        from amsrr.training.order9_r1_complete_task_audit import (
            audit_order9_r1_complete_task_semantics,
        )

        audit_order9_r1_complete_task_semantics(
            phases,
            task_spec=task,
            lift_clearance_m=0.3,
            retreat_offset_m=0.1,
            materializer_id="historical_c3_materializer",
        )


def test_repaired_complete_task_and_parameter_effect_are_accepted() -> None:
    task, inputs = _inputs()
    evidence = audit_order9_r1_materializer_parameter_effect(
        materialize_order9_r1_complete_task_phases,
        materializer_id=ORDER9_R1_NOMINAL_RETREAT_VERSION,
        phase_trajectories=inputs,
        task_spec=task,
        lift_clearance_m=0.3,
        phase_duration_s={phase.value: 1.0 for phase in ORDER9_OBJECT_TASK_PHASES},
    )

    assert evidence["status"] == "accepted"
    assert evidence["retreat_offsets_m"] == [0.05, 0.10]
    assert evidence["measured_endpoint_effect_m"] == pytest.approx(0.05)
    assert evidence["isaac_invoked"] is False
    assert evidence["controller_layers_invoked"] is False


def test_persisted_repaired_artifact_passes_all_eight_phase_audit() -> None:
    task, _inputs_value = _inputs()
    evidence = audit_order9_r1_complete_task_artifact(
        ARTIFACT,
        task_spec=task,
        lift_clearance_m=0.3,
        retreat_offset_m=0.1,
        materializer_id=ORDER9_R1_NOMINAL_RETREAT_VERSION,
    )

    assert evidence["audit_version"] == ORDER9_R1_COMPLETE_TASK_AUDIT_VERSION
    assert evidence["status"] == "accepted"
    assert evidence["checked_phases"] == [
        phase.value for phase in ORDER9_OBJECT_TASK_PHASES
    ]
    assert evidence["measured_retreat_path_length_m"] == pytest.approx(0.1)
