from __future__ import annotations

import json
from dataclasses import dataclass
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from amsrr.feasibility.order9_native_loader_v13 import (
    order9_posture_native_v13_path,
)
from amsrr.schemas.common import SchemaValidationError
from amsrr.training.order9_r1_complete_task_geometry_v13 import (
    _requested_margin_two_stage_check,
)
from amsrr.training import order9_r1_range_selection_v13 as range_v13
from amsrr.training import order9_r1_early_tilt_pipeline as early_tilt
from amsrr.training.order9_r1_range_selection_v10 import (
    load_minimum_level_teacher_hints,
)
from amsrr.utils.hashing import hash_file
from scripts import order9_run_r1_range_selection_v10 as runtime_v10

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "configs/training/order9_r1_range_selection_protocol_v13.json"
APPROVAL = ROOT / "for_codex/R1_RANGE_SELECTION_PROTOCOL_V13_APPROVAL.json"
RUNNER = ROOT / "scripts/order9_run_r1_range_selection_v13.py"
PIPELINE = ROOT / "amsrr/training/order9_r1_range_selection_v13.py"


def test_v13_native_build_identity_and_broad_phase_are_current() -> None:
    extension = order9_posture_native_v13_path()
    generated = extension.parent / "order9_posture_native_v13.cpp"

    assert extension.is_file()
    assert "broad_phase_clearance > activation_distance" in generated.read_text(
        encoding="utf-8"
    )


def test_v13_exact_fallback_enforces_undiminished_requested_margin() -> None:
    calls = []

    def original(_solver, **values):
        calls.append(dict(values))
        if not values["exact"]:
            return {
                "minimum_robot_clearance_m": 0.10,
                "minimum_object_clearance_m": 0.019430590652131215,
                "minimum_ground_clearance_m": 0.10,
                "selected_contact_penetration_violating_pair_count": 0,
            }
        return {
            "effective_collision_margin_m": values["margin_m"] - 0.0002,
            "minimum_robot_clearance_m": 0.10,
            "minimum_object_clearance_m": 0.031,
            "minimum_ground_clearance_m": 0.10,
            "robot_colliding_pair_count": 0,
            "object_colliding_pair_count": 0,
            "selected_contact_penetration_violating_pair_count": 0,
            "worst_pairs": [],
        }

    counters = {"convex_proof_scene_count": 0, "exact_fallback_scene_count": 0}
    solver = SimpleNamespace(
        collision_config=SimpleNamespace(collision_feasibility_tolerance_m=0.0002)
    )
    result = _requested_margin_two_stage_check(
        original,
        counters,
        solver,
        morphology=object(),
        centroidal_pose_world=(0.0,) * 7,
        joint_positions_rad={},
        exact=True,
        margin_m=0.0195,
    )

    assert calls[0]["exact"] is False
    assert calls[1]["exact"] is True
    assert calls[1]["margin_m"] == pytest.approx(0.0197)
    assert result["effective_collision_margin_m"] == pytest.approx(0.0195)
    assert result["r1_exact_stl_fallback_invoked"] is True
    assert counters == {
        "convex_proof_scene_count": 0,
        "exact_fallback_scene_count": 1,
    }


def test_v13_exact_fallback_fails_closed_on_wrong_effective_margin() -> None:
    def original(_solver, **values):
        if not values["exact"]:
            return {
                "minimum_robot_clearance_m": 0.10,
                "minimum_object_clearance_m": 0.0194,
                "minimum_ground_clearance_m": 0.10,
                "selected_contact_penetration_violating_pair_count": 0,
            }
        return {"effective_collision_margin_m": 0.0193}

    solver = SimpleNamespace(
        collision_config=SimpleNamespace(collision_feasibility_tolerance_m=0.0002)
    )
    with pytest.raises(SchemaValidationError, match="requested margin"):
        _requested_margin_two_stage_check(
            original,
            {"convex_proof_scene_count": 0, "exact_fallback_scene_count": 0},
            solver,
            morphology=object(),
            centroidal_pose_world=(0.0,) * 7,
            joint_positions_rad={},
            exact=True,
            margin_m=0.0195,
        )


def test_v13_protocol_separates_generation_and_acceptance_thresholds() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    approval = json.loads(APPROVAL.read_text(encoding="utf-8"))

    assert protocol["acceptance_clearance_m"] == pytest.approx(0.0195)
    assert protocol["planning_clearance_m"] == pytest.approx(0.0300)
    assert protocol["minimum_planning_buffer_m"] == pytest.approx(0.0105)
    assert protocol["uncertified_rigid_pose_copy_rejected"] is True
    assert protocol["exact_margin_tolerance_compensated"] is True
    assert protocol["native_exact_aabb_uses_requested_clearance"] is True
    assert protocol["reuse_pre_clearance_v11_teacher_rejection"] is False
    assert protocol["parent_teacher_rejection_reuse_forbidden"] is True
    assert protocol["selection_requires_current_runner_result"] is True
    assert protocol["automatic_rotation_contact_retry"] is True
    assert protocol["automatic_source_configuration_seed_reuse"] is True
    assert protocol["stale_isaac_evidence_quarantine_required"] is True
    assert protocol["human_posture_rejections_enforced"] is True
    assert protocol["linear_algebra_thread_count_per_worker"] == 1
    assert protocol["preparation_worker_start_method"] == "spawn"
    assert approval["approved_protocol"]["sha256"] == hash_file(PROTOCOL)
    assert approval["learning_authorized"] is False
    assert approval["teacher_collection_authorized"] is False


def test_v13_cached_isaac_evidence_is_bound_to_current_teacher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_id = "r1_l2_20mm_10deg__train__source__lattice_00"
    preparation_root = tmp_path / "prepared" / "r1_l2_20mm_10deg"
    set_manifest = preparation_root / candidate_id / "nominal_set" / "manifest.json"
    bucket_root = (
        preparation_root
        / candidate_id
        / "nominal_set"
        / "buckets"
        / candidate_id
    )
    artifact_manifest = bucket_root / "manifest.json"
    timeline = bucket_root / "nominal_timeline.json"
    for path, payload in (
        (set_manifest, "set"),
        (artifact_manifest, "artifact"),
        (timeline, "timeline"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    monkeypatch.setattr(runtime_v10, "REPOSITORY", tmp_path)
    metadata = {
        "c3_nominal_reference": {
            "set_manifest_path": set_manifest.relative_to(tmp_path).as_posix(),
            "set_manifest_sha256": hash_file(set_manifest),
            "artifact_path": artifact_manifest.relative_to(tmp_path).as_posix(),
            "artifact_sha256": hash_file(artifact_manifest),
            "timeline_sha256": hash_file(timeline),
        }
    }

    bindings = runtime_v10._require_current_nominal_reference(
        candidate_id=candidate_id,
        metadata=metadata,
        preparation_root=preparation_root,
    )
    assert bindings["nominal_timeline"]["sha256"] == hash_file(timeline)

    timeline.write_text("changed teacher", encoding="utf-8")
    with pytest.raises(SchemaValidationError, match="nominal_timeline"):
        runtime_v10._require_current_nominal_reference(
            candidate_id=candidate_id,
            metadata=metadata,
            preparation_root=preparation_root,
        )


def test_v13_stale_isaac_evidence_is_quarantined_without_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_id = "r1_l2_20mm_10deg__train__source__lattice_00"
    preparation_root = tmp_path / "prepared" / "r1_l2_20mm_10deg"
    source = tmp_path / "isaac" / "r1_l2_20mm_10deg" / "single" / candidate_id
    candidate_output = source / candidate_id
    candidate_output.mkdir(parents=True)
    original = candidate_output / "isaac" / "evaluation_episodes.jsonl"
    original.parent.mkdir()
    original.write_text("old evidence", encoding="utf-8")
    monkeypatch.setattr(runtime_v10, "REPOSITORY", tmp_path)

    manifest_binding = runtime_v10._quarantine_invalid_candidate_output(
        candidate_id=candidate_id,
        evidence_root=source,
        preparation_root=preparation_root,
        failure_reason="stale teacher hash",
    )

    assert manifest_binding is not None
    assert not candidate_output.exists()
    manifest = json.loads(
        (tmp_path / manifest_binding["path"]).read_text(encoding="utf-8")
    )
    quarantined = tmp_path / manifest["quarantine_path"]
    assert (quarantined / "isaac" / "evaluation_episodes.jsonl").read_text(
        encoding="utf-8"
    ) == "old evidence"
    assert manifest["destructive_deletion_used"] is False


def test_v13_hash_bound_human_rejection_forbids_train027_pair_19_22() -> None:
    path = ROOT / "configs/training/order9_c3_human_posture_rejections_v2.json"
    rejections = range_v13._load_human_posture_rejections(
        path,
        repository=ROOT,
    )
    exclusions = {
        source_bucket_id: value["excluded_surface_port_id_pairs"]
        for source_bucket_id, value in rejections.items()
    }

    with pytest.raises(SchemaValidationError, match="human-rejected contact pair"):
        range_v13._assert_pair_allowed(
            "train-000027-7fe33c662d36",
            (19, 22),
            exclusions=exclusions,
            label="test option",
        )
    range_v13._assert_pair_allowed(
        "train-000027-7fe33c662d36",
        (21, 29),
        exclusions=exclusions,
        label="test option",
    )


def test_v13_teacher_hints_do_not_reintroduce_human_rejected_pairs() -> None:
    hints = load_minimum_level_teacher_hints(
        ROOT / "configs/training/order9_r1_range_teacher_hints_v10.json",
        repository_root=ROOT,
    )
    train027 = "train-000027-7fe33c662d36"
    options = (
        option
        for candidate_id, candidate_options in hints.items()
        if f"__{train027}__" in candidate_id
        for option in candidate_options
    )

    assert all(frozenset(option[0]) != frozenset((19, 22)) for option in options)


def test_v13_configuration_branch_seeds_are_hash_bound_and_finite() -> None:
    seeds = range_v13._load_configuration_goal_seeds(
        ROOT / "configs/training/order9_r1_range_teacher_repairs_v13.json",
        repository=ROOT,
    )

    prefix = (
        "r1_l2_20mm_10deg__train__"
        "train-000027-7fe33c662d36__lattice_"
    )
    assert set(seeds) >= {prefix + "02", prefix + "24"}
    assert seeds[prefix + "02"]
    assert seeds[prefix + "24"]
    assert (
        range_v13.ORDER9_R1_TRAIN027_CONFIGURATION_SEED_SOURCE_CANDIDATE_ID
        in seeds
    )
    assert "automatic_source_configuration_seed_reuse" in PIPELINE.read_text(
        encoding="utf-8"
    )
    assert "configuration_goal_joint_seed_positions_rad" in PIPELINE.read_text(
        encoding="utf-8"
    )


def test_v13_nearby_same_morphology_seed_overrides_are_hash_bound() -> None:
    repairs_path = (
        ROOT / "configs/training/order9_r1_range_teacher_repairs_v13.json"
    )
    payload = json.loads(repairs_path.read_text(encoding="utf-8"))
    overrides = payload["automatic_configuration_goal_seed_overrides"]
    prefix = (
        "r1_l2_20mm_10deg__train__"
        "train-000027-7fe33c662d36__"
    )
    expected = {
        prefix + "lattice_14": prefix + "lattice_13",
        prefix + "lattice_23": prefix + "lattice_22",
        prefix + "interior_02": prefix + "lattice_08",
    }
    superseded_candidate_id = prefix + "lattice_17"
    interpolated_candidate_id = prefix + "lattice_10"
    interpolated_source_ids = [
        prefix + "lattice_09",
        prefix + "lattice_11",
    ]
    confirmed = payload["candidate_repairs"][interpolated_candidate_id]

    evidence = range_v13._load_configuration_goal_seed_evidence(
        repairs_path,
        repository=ROOT,
    )
    assert {
        candidate_id: value["source_candidate_id"]
        for candidate_id, value in overrides.items()
        if "source_candidate_id" in value
    } == expected
    for candidate_id, source_candidate_id in expected.items():
        assert evidence[candidate_id]["automatic_source_configuration_seed_reuse"]
        assert (
            evidence[candidate_id]["source_seed_candidate_id"]
            == source_candidate_id
        )
        assert evidence[candidate_id]["contact_goal_seed_applied"] is False
        assert evidence[candidate_id]["contact_ik_constraints_changed"] is False
        assert evidence[candidate_id]["acceptance_or_safety_gate_changed"] is False
    assert superseded_candidate_id not in overrides
    assert (
        payload["candidate_repairs"][superseded_candidate_id]["repair_kind"]
        == "confirmed_isaac_collision_teacher_option_reselection"
    )
    assert "source_seed_candidate_id" not in evidence[superseded_candidate_id]
    assert evidence[superseded_candidate_id]["isaac_invoked_during_seed_selection"]
    assert confirmed["repair_kind"] == "confirmed_cpu_teacher_option_selection"
    assert confirmed["teacher_candidate_override"]["selected_surface_port_ids"] == [
        21,
        29,
    ]
    assert (
        confirmed["teacher_candidate_override"]["candidate_group_id"]
        == "slot_0:grasp_pair:3"
    )
    seed_path = ROOT / confirmed["configuration_goal_seed_trajectory"]["path"]
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    assert all(
        f"/{source_candidate_id}/" in item["path"]
        for source_candidate_id, item in zip(
            interpolated_source_ids, seed["source_trajectories"], strict=True
        )
    )
    assert evidence[interpolated_candidate_id]["contact_goal_seed_applied"] is False
    assert evidence[interpolated_candidate_id]["contact_ik_constraints_changed"] is False
    assert (
        evidence[interpolated_candidate_id]["acceptance_or_safety_gate_changed"]
        is False
    )
    assert interpolated_candidate_id in range_v13._load_confirmed_teacher_option_overrides(
        repairs_path
    )
    assert (
        "or case.candidate_id in self.configuration_goal_seeds"
        in PIPELINE.read_text(encoding="utf-8")
    )


def test_v13_confirmed_isaac_collision_reseed_requires_bound_replays() -> None:
    repairs_path = (
        ROOT / "configs/training/order9_r1_range_teacher_repairs_v13.json"
    )
    candidate_id = (
        "r1_l2_20mm_10deg__train__"
        "train-000002-0e86a4ff3e71__interior_02"
    )

    repairs = range_v13._load_bounded_local_contact_repairs(
        repairs_path,
        repository=ROOT,
    )
    evidence = range_v13._load_configuration_goal_seed_evidence(
        repairs_path,
        repository=ROOT,
    )[candidate_id]

    assert candidate_id in repairs
    assert evidence["scope"] == "configuration_space_goal_initialization_only"
    assert evidence["contact_goal_seed_applied"] is False
    assert evidence["contact_ik_constraints_changed"] is False
    assert evidence["acceptance_or_safety_gate_changed"] is False
    assert evidence["isaac_invoked_during_seed_selection"] is True
    assert evidence["controller_layers_invoked_during_seed_selection"] is True
    assert "source_confirmed_isaac_failure" in evidence


def test_v13_superseded_preparation_is_quarantined_not_deleted(tmp_path) -> None:
    from scripts import order9_run_r1_range_selection_v13 as runner_v13

    destination = tmp_path / "prepared" / "r1_l2_20mm_10deg" / "candidate"
    destination.mkdir(parents=True)
    (destination / "evidence.json").write_text("preserve", encoding="utf-8")

    runner_v13._quarantine_preparation_destination(
        destination,
        reason="test",
    )

    preserved = (
        tmp_path
        / "diagnostics"
        / "preparation_quarantine"
        / "candidate"
        / "test__attempt_000"
        / "evidence.json"
    )
    assert not destination.exists()
    assert preserved.read_text(encoding="utf-8") == "preserve"


def test_early_tilt_forwards_human_rejection_to_every_generation_attempt(
    monkeypatch,
) -> None:
    observed = {}

    def generate(**kwargs):
        observed.update(kwargs)
        return "nominal"

    monkeypatch.setattr(
        early_tilt,
        "generate_order9_c3_nominal_grasp_trajectory",
        generate,
    )
    result = early_tilt._generate_with_safe_surface_fallback(
        task_spec=object(),
        structural_target=object(),
        physical_model=object(),
        preferred_surface_port_id_options=(),
        preferred_candidate_options=(((19, 22), "group", 0.12, 0.007),),
        maximum_body_tilt_rad=1.0,
        strict_preferred_candidate_options=True,
        excluded_surface_port_id_pairs=((19, 22),),
    )

    assert result == "nominal"
    assert observed["excluded_surface_port_id_pairs"] == ((19, 22),)


def test_v13_current_runner_cannot_skip_the_20mm_selection_level() -> None:
    runner = RUNNER.read_text(encoding="utf-8")

    assert "_reuse_parent_v11_selection_rejection(protocol)" not in runner
    assert "for level_id in levels_to_run:" in runner
    assert 'split="train"' in runner
    assert '"selection_results_generated_by_current_runner": True' in runner
    assert '"parent_teacher_rejection_reused": False' in runner


def test_v13_rotation_retry_disables_both_nested_lightweight_contexts() -> None:
    pipeline = PIPELINE.read_text(encoding="utf-8")

    assert "patch.object(\n            range_v12," in pipeline
    assert "patch.object(\n            range_v10," in pipeline
    assert "E_R1_GRASP_ROTATION_FORCE_MOMENT_ARM" in pipeline
    assert "automatic_rotation_contact_retries" in pipeline
    assert '"sampled overhead search is disabled"' in pipeline


def test_v13_lightweight_search_omission_triggers_bounded_retry_evidence() -> None:
    pipeline = PIPELINE.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")

    assert "automatic_lightweight_contact_retries" in pipeline
    assert (
        "sampled local contact search is disabled in lightweight admission"
        in pipeline
    )
    assert "automatic_lightweight_contact_retry_v1.json" in runner
    assert '"acceptance_or_safety_gate_changed": False' in pipeline


def test_v13_preparation_spawns_clean_workers_after_native_initialization() -> None:
    runner = RUNNER.read_text(encoding="utf-8")

    assert 'multiprocessing.get_context("spawn")' in runner
    assert "mp_context=" in runner
    assert "every wave so no initialized solver state crosses work items" in runner
    assert "wave = arguments[next_index : next_index + maximum_pending]" in runner


def test_v13_contact_acquisition_lead_resegments_only_approach_suffix(
    monkeypatch,
) -> None:
    @dataclass(frozen=True)
    class Window:
        phase: str
        plan: str = "plan"

    @dataclass(frozen=True)
    class Nominal:
        windows: tuple[Window, ...]
        timeline: tuple[str, ...]

    monkeypatch.setattr(
        range_v13,
        "_flatten_nominal_windows",
        lambda windows: tuple(window.phase for window in windows),
    )
    monkeypatch.setattr(
        range_v13,
        "_relabel_order9_r1_plan_contact_phase",
        lambda plan, **_kwargs: plan,
    )
    nominal = Nominal(
        windows=(
            Window("approach"),
            Window("approach"),
            Window("approach"),
            Window("approach"),
            Window("contact_acquisition"),
        ),
        timeline=(),
    )

    shifted = range_v13._resegment_order9_r1_contact_acquisition(
        nominal,
        lead_window_count=3,
    )

    assert shifted.timeline == (
        "approach",
        "contact_acquisition",
        "contact_acquisition",
        "contact_acquisition",
        "contact_acquisition",
    )


def test_v13_isaac_child_uses_bound_python312_native_extension(monkeypatch) -> None:
    from scripts import order9_run_r1_range_selection_v13 as runner_v13

    variable = "AMSRR_ORDER9_POSTURE_NATIVE_PATH"
    compile_variable = "TORCHINDUCTOR_COMPILE_THREADS"
    previous = os.environ.get(variable)
    previous_compile = os.environ.get(compile_variable)
    observed = {}

    def inspect_environment(*args, **kwargs):
        observed["path"] = os.environ[variable]
        observed["compile_threads"] = os.environ[compile_variable]
        return {"done": 0}

    monkeypatch.setattr(runner_v13, "_V10_RUN_PROCESSES", inspect_environment)
    result = runner_v13._run_processes_with_isaac_abi()

    assert result == {"done": 0}
    assert "order9_posture_native_v13_py312" in observed["path"]
    assert observed["compile_threads"] == "8"
    assert os.environ.get(variable) == previous
    assert os.environ.get(compile_variable) == previous_compile
