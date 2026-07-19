from __future__ import annotations

import json
from pathlib import Path

import amsrr.acceptance.order8_acceptance as acceptance_module
from amsrr.acceptance.order8_acceptance import run_order8_acceptance
from amsrr.robot_model.physical_model_builder import (
    build_physical_model_from_config,
)
from amsrr.schemas.order8 import Order8NaturalContactConfig
from amsrr.simulation.isaac_lab_backend import (
    IsaacLabBackend,
    load_isaac_lab_backend_config,
)
from amsrr.simulation.order8_natural_contact import (
    ORDER8_NATURAL_CONTACT_ENV_VERSION,
    Order8IsaacNaturalContactEnv,
    Order8IsaacNaturalContactResult,
)
from amsrr.training.p2_inspection_context import default_grasp_carry_task_spec
from amsrr.utils.hashing import hash_directory_manifest, hash_file


def test_order8_acceptance_rebuilds_hash_bound_local_inputs(
    tmp_path: Path, monkeypatch
) -> None:
    path = _write_minimal_bound_result(tmp_path)
    monkeypatch.setattr(
        acceptance_module,
        "order8_natural_contact_report_failures",
        lambda *args, **kwargs: [],
    )

    report = run_order8_acceptance(path)

    assert report.artifact_loaded is True
    assert report.real_isaac_passed is True
    assert report.evidence_integrity_passed is True
    assert report.no_mislabeling_passed is True
    assert report.completion_passed is True
    assert report.failures == []
    assert report.artifacts["order8_result_sha256"] == hash_file(path)


def test_order8_acceptance_rejects_provenance_tampering_and_mislabeling(
    tmp_path: Path, monkeypatch
) -> None:
    path = _write_minimal_bound_result(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["report"]["run_provenance"]["task_spec_hash"] = "e" * 64
    payload["report"]["run_provenance"]["graph_hash"] = "f" * 64
    payload["report"]["run_provenance"]["simulation_dt_s"] = 0.01
    payload["report"]["order8_natural_contact_p4_full_completion_claim"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    Path(payload["report"]["usd_path"]).write_text(
        "#usda 1.0\n# tampered\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        acceptance_module,
        "order8_natural_contact_report_failures",
        lambda *args, **kwargs: [],
    )

    report = run_order8_acceptance(path)

    assert report.completion_passed is False
    assert report.evidence_integrity_passed is False
    assert report.no_mislabeling_passed is False
    assert "graph_hash_mismatch" in report.failures
    assert "task_spec_hash_mismatch" in report.failures
    assert "requested_steps_budget_mismatch" in report.failures
    assert "generated_usd_hash_mismatch" in report.failures
    assert "generated_usd_bundle_hash_mismatch" in report.failures
    assert "order8_result_mislabeled" in report.failures


def test_order8_acceptance_rejects_missing_or_dry_run_artifact(tmp_path: Path) -> None:
    missing = run_order8_acceptance(tmp_path / "missing.json")
    assert missing.artifact_loaded is False
    assert missing.completion_passed is False
    assert "artifact_missing" in missing.failures

    env = _env()
    dry = env.run(dry_run=True)
    path = tmp_path / "dry.json"
    path.write_text(dry.to_json(), encoding="utf-8")
    report = run_order8_acceptance(path)
    assert report.real_isaac_passed is False
    assert report.completion_passed is False
    assert "result_is_not_attempted_real_isaac" in report.failures


def _env() -> Order8IsaacNaturalContactEnv:
    backend_config = load_isaac_lab_backend_config("configs/env/isaac_lab.yaml")
    physical_model = build_physical_model_from_config(
        backend_config.robot_model_config_path
    )
    return Order8IsaacNaturalContactEnv(
        config=Order8NaturalContactConfig(),
        backend=IsaacLabBackend(backend_config),
        physical_model=physical_model,
    )


def _write_minimal_bound_result(tmp_path: Path) -> Path:
    env = _env()
    graph = env.representative_morphology()
    config_path = Path("configs/training/order8_natural_contact.yaml").resolve()
    backend_path = Path("configs/env/isaac_lab.yaml").resolve()
    robot_path = Path(env.backend.config.robot_model_config_path).resolve()
    task_spec = default_grasp_carry_task_spec()
    generated_root = tmp_path / "generated"
    generated_urdf_path = generated_root / "resolved_urdf" / "holon.urdf"
    generated_usd_path = generated_root / "holon" / "holon.usda"
    generated_urdf_path.parent.mkdir(parents=True)
    generated_usd_path.parent.mkdir(parents=True)
    generated_urdf_path.write_text("<robot name='order8-test'/>", encoding="utf-8")
    generated_usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    generated_usd_hash = hash_file(generated_usd_path)
    generated_bundle_hash = hash_directory_manifest(generated_usd_path.parent)
    report = {
        "order8_natural_contact_config": env.config.to_dict(),
        "order8_natural_contact_scope": "deterministic_natural_contact_substrate_only",
        "order8_natural_contact_p4_full_completion_claim": False,
        "order8_natural_contact_order9_full_taskspec_claim": False,
        "order8_natural_contact_learned_policy_success_claim": False,
        "generated_urdf_path": str(generated_urdf_path),
        "generated_urdf_sha256": hash_file(generated_urdf_path),
        "usd_path": str(generated_usd_path),
        "generated_usd_sha256": generated_usd_hash,
        "generated_usd_bundle_hash": generated_bundle_hash,
        "order8_natural_contact_generated_usd_sha256": generated_usd_hash,
        "order8_natural_contact_generated_usd_bundle_hash": generated_bundle_hash,
        "run_provenance": {
            "task_id": task_spec.task_id,
            "task_spec": task_spec.to_dict(),
            "task_spec_hash": task_spec.stable_hash(),
            "graph_id": graph.graph_id,
            "graph_hash": graph.stable_hash(),
            "morphology_graph": graph.to_dict(),
            "config_path": str(config_path),
            "config_file_sha256": hash_file(config_path),
            "config": env.config.to_dict(),
            "config_hash": env.config.stable_hash(),
            "backend_config_path": str(backend_path),
            "backend_config_file_sha256": hash_file(backend_path),
            "backend_config_hash": env.backend.config.stable_hash(),
            "robot_model_config_path": str(robot_path),
            "robot_model_config_file_sha256": hash_file(robot_path),
            "physical_model_hash": env.physical_model.stable_hash(),
            "source_urdf_path": str(Path(env.physical_model.urdf_path).resolve()),
            "source_urdf_sha256": hash_file(env.physical_model.urdf_path),
            "collision_geometry_content_hash": env.collision_geometry_hash,
            "requested_steps": env.requested_steps,
            "seed": env.seed,
            "simulation_dt_s": env.simulation_dt_s,
            "rollout_budget_s": env.rollout_budget_s,
            "generated_usd_dir": str(generated_root),
            "real_requested": True,
        },
    }
    result = Order8IsaacNaturalContactResult(
        env_version=ORDER8_NATURAL_CONTACT_ENV_VERSION,
        graph_id=graph.graph_id,
        graph_hash=graph.stable_hash(),
        config_hash=env.config.stable_hash(),
        dry_run=False,
        attempted=True,
        isaac_backed=True,
        passed=True,
        report_validation_failures=[],
        report=report,
        failure_reason=None,
    )
    path = tmp_path / "order8.json"
    path.write_text(result.to_json(), encoding="utf-8")
    return path
