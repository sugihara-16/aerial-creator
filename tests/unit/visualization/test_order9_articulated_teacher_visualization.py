from __future__ import annotations

import json
from pathlib import Path

from amsrr.irg.envelope_extractor import InteractionEnvelopeExtractor
from amsrr.irg.irg_builder import IRGBuilder
from amsrr.morphology.grasp_carry_designs import (
    GraspCarryMorphologyVariant,
    build_grasp_carry_variant_design_output,
)
from amsrr.policies.contact_candidate_sampler import ContactCandidateSampler
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.robot_model.physical_model_builder import (
    build_physical_model_from_config,
)
from amsrr.schemas.task_spec import TaskSpec
from amsrr.training.order9_c3_teacher import _neutral_runtime_observation
from amsrr.visualization.order9_articulated_teacher import (
    ORDER9_ARTICULATED_TEACHER_ANIMATION_VERSION,
    build_order9_articulated_teacher_animation_payload,
    write_order9_articulated_teacher_animation,
)
from tests.unit.training.test_order9_articulated_teacher import (
    _production_teacher,
)


def test_articulated_teacher_animation_contains_fk_object_and_joint_targets(
    tmp_path: Path,
    grasp_carry_dict: dict,
) -> None:
    task = TaskSpec.from_dict(grasp_carry_dict)
    built = IRGBuilder().build_with_scene_graph(task)
    envelope = InteractionEnvelopeExtractor().extract(built.irg)
    physical = build_physical_model_from_config(
        "configs/robot/robot_model.yaml"
    )
    design = build_grasp_carry_variant_design_output(
        task,
        built.irg,
        physical,
        variant=GraspCarryMorphologyVariant.SYMMETRIC_TWO_ANCHOR_GRASP,
    )
    morphology = design.target_morphology
    candidates = ContactCandidateSampler().sample(
        task_spec=task,
        irg=built.irg,
        interaction_envelope=envelope,
        morphology_graph=morphology,
        geometry_descriptors=built.scene_graph.geometry_descriptors,
    )
    plan = _production_teacher(task, physical).plan(
        HighLevelPolicyContext(
            built.irg,
            envelope,
            morphology,
            candidates,
            runtime_observation=_neutral_runtime_observation(
                morphology,
                physical,
                task,
                phase_label="contact_acquisition",
            ),
        ),
        initial_object_poses_world={
            obj.object_id: obj.pose_world for obj in task.scene.objects
        },
    )

    payload = build_order9_articulated_teacher_animation_payload(
        task_spec=task,
        morphology=morphology,
        contact_candidate_set=candidates,
        trajectory_plan=plan,
        physical_model=physical,
        frames_per_second=4,
        source_metadata={"bucket_id": "unit"},
    )
    summary = payload["summary"]
    frames = payload["frames"]
    assert payload["animation_version"] == (
        ORDER9_ARTICULATED_TEACHER_ANIMATION_VERSION
    )
    assert summary["morphology"]["module_count"] == len(morphology.modules)
    assert summary["object"]["geometry_type"] == "box"
    assert summary["source_metadata"]["bucket_id"] == "unit"
    assert summary["ik"]["feasible"]
    assert summary["grasp_joint_positions"]
    assert len(frames) >= len(plan.trajectory.knots)
    assert all(
        len(frame["modules"]) == len(morphology.modules)
        for frame in frames
    )
    assert all(len(frame["contacts"]) == 2 for frame in frames)
    grasp_time = float(summary["trajectory"]["grasp_time_s"])
    grasp_frame = min(
        frames,
        key=lambda frame: abs(float(frame["time_s"]) - grasp_time),
    )
    initial_error = max(
        contact["position_error_m"] for contact in frames[0]["contacts"]
    )
    endpoint_error = max(
        contact["position_error_m"] for contact in grasp_frame["contacts"]
    )
    assert endpoint_error < initial_error

    html_path = tmp_path / "teacher.html"
    artifacts = write_order9_articulated_teacher_animation(
        payload,
        html_path=html_path,
    )
    assert artifacts.html_path == html_path
    assert artifacts.summary_path == tmp_path / "teacher.summary.json"
    page = html_path.read_text(encoding="utf-8")
    archived = json.loads(
        artifacts.summary_path.read_text(encoding="utf-8")
    )
    assert "運動学的teacher指令" in page
    assert "grasp_joint_positions" in page
    assert "<script src=" not in page
    assert archived == summary
