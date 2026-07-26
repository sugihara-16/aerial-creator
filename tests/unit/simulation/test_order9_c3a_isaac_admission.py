from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    REPOSITORY_ROOT / "scripts/order9_c3a_isaac_admission.py"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "order9_c3a_isaac_admission",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_accepted_cases_are_hash_bound_with_exact_root_and_joint_sets() -> None:
    module = _module()
    manifest_path = (
        REPOSITORY_ROOT
        / "artifacts/p4_full/order9/"
        "c3a_contact_penetration_batch_001/manifest.json"
    )
    review_path = manifest_path.with_name("review_decisions.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for case in manifest["cases"]:
        bound = module._load_bound_case(
            manifest_path=manifest_path,
            review_path=review_path,
            case_id=case["case_id"],
            robot_model_config_path=(
                REPOSITORY_ROOT / "configs/robot/robot_model.yaml"
            ),
        )
        assert bound["root_pose_conversion"]["passed"] is True
        assert (
            0.0608
            < bound["root_pose_conversion"]["position_error_m"]
            < 0.0611
        )
        assert bound["root_pose_conversion"]["attitude_error_rad"] == 0.0
        assert (
            bound["root_pose_world"]
            != bound["ik_base_pose_world"]
        )
        assert len(bound["joint_positions"]) == 4 * case["module_count"]
        assert len(bound["expected_movable_joint_names"]) == (
            12 * case["module_count"]
        )
        assert set(bound["joint_positions"]) == set(
            bound["expected_dock_joint_names"]
        )
        assert (
            bound["joint_solution_hash"]
            == case["final_pose"]["collision_aware"][
                "joint_solution_hash"
            ]
        )
        assert bound["physical_contact_force_threshold_n"] == 0.5
        assert bound["order8_max_force_per_contact_n"] == 30.0
        assert (
            bound["physical_contact_penetration_noise_floor_m"]
            == 0.0001
        )


def test_admission_declares_production_runtime_collision_scope() -> None:
    module = _module()
    assert module.PRODUCTION_COLLISION_FILTER_SEMANTICS == (
        "production_random_morphology_takeoff_all_same_module_"
        "plus_intended_dock_v1"
    )


def test_cross_module_contact_matrix_includes_every_body_pair() -> None:
    module = _module()
    assert module._included_contact_matrix_indices(
        module_pair=(0, 1),
        sensor_paths=("/Robot/module_0__a",),
        filter_paths=("/Robot/module_1__a", "/Robot/module_1__b"),
    ) == (0, 1)


def test_fixed_child_pose_is_inferred_from_observed_parent() -> None:
    module = _module()
    requested_parent = (1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0)
    requested_child = (1.0, 2.1, 3.0, 0.0, 0.0, 0.0, 1.0)
    observed_parent = (
        4.0,
        5.0,
        6.0,
        0.0,
        0.0,
        2.0**-0.5,
        2.0**-0.5,
    )
    observed_child = module._infer_fixed_child_pose_world(
        requested_parent_pose_world=requested_parent,
        requested_child_pose_world=requested_child,
        observed_parent_pose_world=observed_parent,
    )
    assert observed_child[:3] == [3.9, 5.0, 6.0]
    assert module._attitude_error(
        observed_child[3:],
        observed_parent[3:],
    ) == 0.0


def test_order8_noise_floors_classify_physical_contact_patches() -> None:
    module = _module()
    speculative = module._physical_contact_patch_evidence(
        patch_forces_n=(0.49,),
        patch_separations_m=(-0.00009,),
        force_threshold_n=0.5,
        penetration_noise_floor_m=0.0001,
    )
    assert speculative["finite"] is True
    assert speculative["physical_contact"] is False

    by_force = module._physical_contact_patch_evidence(
        patch_forces_n=(0.5,),
        patch_separations_m=(0.001,),
        force_threshold_n=0.5,
        penetration_noise_floor_m=0.0001,
    )
    assert by_force["physical_contact"] is True
    assert by_force["force_threshold_met"] is True

    by_penetration = module._physical_contact_patch_evidence(
        patch_forces_n=(0.0,),
        patch_separations_m=(-0.0001,),
        force_threshold_n=0.5,
        penetration_noise_floor_m=0.0001,
    )
    assert by_penetration["physical_contact"] is True
    assert by_penetration["penetration_noise_floor_exceeded"] is True

    nonfinite = module._physical_contact_patch_evidence(
        patch_forces_n=(float("nan"),),
        patch_separations_m=(0.0,),
        force_threshold_n=0.5,
        penetration_noise_floor_m=0.0001,
    )
    assert nonfinite["finite"] is False
    assert nonfinite["physical_contact"] is False


def test_batch_summary_marks_settle_drift_as_diagnostic_only(
    tmp_path: Path,
) -> None:
    module = _module()
    manifest_path = tmp_path / "source_manifest.json"
    review_path = tmp_path / "review.json"
    output_root = tmp_path / "admission"
    manifest_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "case-1",
                        "structural_hash": "a" * 64,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    review_path.write_text(
        json.dumps(
            {
                "decisions": {
                    "case-1": {
                        "scene_sha256": "b" * 64,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    report_path = output_root / "cases/case-1/admission.json"
    module._write_report(
        report_path,
        {
            "status": "pass",
            "passed": True,
            "failure_reasons": [],
            "method": {
                "trajectory_path_status": "not_run_missing_bound_path"
            },
            "resource_usage": {"wall_time_s": 1.0},
            "settle": {
                "maximum_selected_force_n": {
                    "module_0__yaw_dock_mech1": 31.0,
                },
                "order8_selected_contact_force_limit_n": 30.0,
                "final_state_drift": {"passed": False},
                "final_state_drift_gate_role": (
                    "diagnostic_only_not_an_admission_gate"
                ),
            },
        },
    )
    summary_path = module._write_batch_summary(
        output_root=output_root,
        manifest_path=manifest_path,
        review_path=review_path,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["passed_count"] == 1
    assert summary["static_pose_collision_contact_pass_count"] == 1
    assert summary["production_isaac_admission_eligible_count"] == 0
    assert (
        summary["cases"][0]["result_semantics"]
        == "static_pose_collision_contact_pass"
    )
    assert (
        summary["cases"][0][
            "order8_selected_contact_force_limit_passed"
        ]
        is False
    )
    assert (
        summary["method"]["final_state_drift_gate_role"]
        == "diagnostic_only_not_an_admission_gate"
    )
    assert summary["method"]["trajectory_reachability"].startswith(
        "out_of_scope"
    )
