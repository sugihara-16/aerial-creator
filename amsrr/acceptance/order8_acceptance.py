from __future__ import annotations

"""Artifact-level acceptance gate for the Order 8 natural-contact smoke."""

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

from amsrr.feasibility.morphology_flight import collision_geometry_content_hash
from amsrr.robot_model.physical_model_builder import (
    build_physical_model_from_config,
)
from amsrr.schemas.common import SchemaBase, SchemaValidationError
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.order8 import Order8NaturalContactConfig
from amsrr.simulation.isaac_lab_backend import load_isaac_lab_backend_config
from amsrr.simulation.order8_natural_contact import (
    ORDER8_NATURAL_CONTACT_ENV_VERSION,
    Order8IsaacNaturalContactResult,
    order8_natural_contact_report_failures,
    validate_representative_order8_morphology,
)
from amsrr.training.p2_inspection_context import default_grasp_carry_task_spec
from amsrr.utils.config import load_config
from amsrr.utils.hashing import hash_directory_manifest, hash_file


@dataclass
class Order8AcceptanceReport(SchemaBase):
    artifact_loaded: bool
    real_isaac_passed: bool
    evidence_integrity_passed: bool
    no_mislabeling_passed: bool
    completion_passed: bool
    failures: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)


def run_order8_acceptance(
    artifact_path: str | Path = "artifacts/p4_full/order8_natural_contact/report.json",
    *,
    expected_artifact_sha256: str | None = None,
) -> Order8AcceptanceReport:
    path = Path(artifact_path)
    failures: list[str] = []
    result = _load_result(path, failures)
    artifact_loaded = result is not None
    if expected_artifact_sha256 is not None:
        if not path.is_file() or hash_file(path) != expected_artifact_sha256:
            failures.append("artifact_sha256_mismatch")
    if result is None:
        return _acceptance_report(
            path=path,
            artifact_loaded=False,
            real_isaac_passed=False,
            evidence_integrity_passed=False,
            no_mislabeling_passed=False,
            failures=failures,
        )

    if result.env_version != ORDER8_NATURAL_CONTACT_ENV_VERSION:
        failures.append("env_version_mismatch")
    if result.dry_run or not result.attempted or not result.isaac_backed:
        failures.append("result_is_not_attempted_real_isaac")
    if result.report_validation_failures:
        failures.append("wrapper_report_validation_failed")
    provenance = result.report.get("run_provenance")
    if not isinstance(provenance, dict):
        failures.append("run_provenance_missing")
        provenance = {}
    _validate_task_provenance(provenance, failures)

    config = _load_bound_config(result.report, provenance, failures)
    graph = _load_bound_graph(provenance, failures)
    physical_model = None
    backend_hash: str | None = None
    collision_hash: str | None = None
    source_urdf_hash: str | None = None
    if config is not None and graph is not None:
        physical_model, backend_hash = _load_bound_models(provenance, failures)
        if physical_model is not None:
            try:
                validate_representative_order8_morphology(
                    graph,
                    physical_model=physical_model,
                )
            except (SchemaValidationError, ValueError):
                failures.append("representative_morphology_mismatch")
            try:
                collision_hash = collision_geometry_content_hash(
                    physical_model,
                    mesh_search_dirs=("module_urdf", "module_urdf/mesh"),
                )
                source_urdf_hash = hash_file(physical_model.urdf_path)
            except (OSError, ValueError):
                failures.append("local_collision_provenance_unavailable")

    report_failures: list[str] = []
    if config is not None and graph is not None and physical_model is not None:
        requested_steps = provenance.get("requested_steps")
        seed = provenance.get("seed")
        simulation_dt_s = provenance.get("simulation_dt_s")
        rollout_budget_s = provenance.get("rollout_budget_s")
        if not _positive_int(requested_steps):
            failures.append("requested_steps_provenance_invalid")
        if not _finite_positive(simulation_dt_s):
            failures.append("simulation_dt_provenance_invalid")
        if not _finite_positive(rollout_budget_s):
            failures.append("rollout_budget_provenance_invalid")
        if (
            _positive_int(requested_steps)
            and _finite_positive(simulation_dt_s)
            and _finite_positive(rollout_budget_s)
            and requested_steps
            != int(math.ceil(float(rollout_budget_s) / float(simulation_dt_s)))
        ):
            failures.append("requested_steps_budget_mismatch")
        report_failures = order8_natural_contact_report_failures(
            result.report,
            morphology_graph=graph,
            config=config,
            physical_model=physical_model,
            expected_backend_config_hash=backend_hash,
            expected_collision_geometry_hash=collision_hash,
            expected_source_urdf_hash=source_urdf_hash,
            requested_steps=(
                requested_steps
                if _positive_int(requested_steps)
                else None
            ),
            expected_seed=(
                seed
                if isinstance(seed, int)
                and not isinstance(seed, bool)
                and seed >= 0
                else None
            ),
            expected_simulation_dt_s=(
                float(simulation_dt_s)
                if _finite_positive(simulation_dt_s)
                else None
            ),
        )
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            failures.append("seed_provenance_invalid")
        failures.extend(f"report:{failure}" for failure in report_failures)

    _validate_source_and_collision_provenance(
        provenance,
        physical_model=physical_model,
        collision_hash=collision_hash,
        failures=failures,
    )
    _validate_generated_artifacts(result.report, provenance, failures)

    _validate_wrapper_binding(result, config, graph, provenance, failures)
    no_mislabeling = bool(
        result.report.get("order8_natural_contact_scope")
        == "deterministic_natural_contact_substrate_only"
        and result.report.get("order8_natural_contact_p4_full_completion_claim")
        is False
        and result.report.get(
            "order8_natural_contact_order9_full_taskspec_claim"
        )
        is False
        and result.report.get(
            "order8_natural_contact_learned_policy_success_claim"
        )
        is False
    )
    if not no_mislabeling:
        failures.append("order8_result_mislabeled")

    real_isaac_passed = bool(
        result.passed
        and result.attempted
        and result.isaac_backed
        and not result.dry_run
        and not result.report_validation_failures
        and not report_failures
    )
    integrity_failure_prefixes = (
        "artifact_",
        "run_provenance",
        "config_",
        "task_",
        "graph_",
        "backend_",
        "physical_model_",
        "robot_model_",
        "representative_",
        "local_collision_",
        "source_",
        "collision_",
        "generated_",
        "seed_",
        "simulation_dt_",
        "rollout_budget_",
        "requested_steps_",
        "wrapper_",
        "report:",
    )
    evidence_integrity = not any(
        failure.startswith(integrity_failure_prefixes) for failure in failures
    )
    return _acceptance_report(
        path=path,
        artifact_loaded=artifact_loaded,
        real_isaac_passed=real_isaac_passed,
        evidence_integrity_passed=evidence_integrity,
        no_mislabeling_passed=no_mislabeling,
        failures=failures,
    )


def _load_result(
    path: Path, failures: list[str]
) -> Order8IsaacNaturalContactResult | None:
    if not path.is_file():
        failures.append("artifact_missing")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return Order8IsaacNaturalContactResult.from_dict(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, SchemaValidationError, TypeError):
        failures.append("artifact_invalid")
        return None


def _load_bound_config(
    report: dict[str, Any],
    provenance: dict[str, Any],
    failures: list[str],
) -> Order8NaturalContactConfig | None:
    payload = report.get("order8_natural_contact_config")
    try:
        config = Order8NaturalContactConfig.from_dict(payload)
    except (SchemaValidationError, TypeError):
        failures.append("config_payload_invalid")
        return None
    if provenance.get("config") != config.to_dict():
        failures.append("config_provenance_payload_mismatch")
    if provenance.get("config_hash") != config.stable_hash():
        failures.append("config_provenance_hash_mismatch")
    path_raw = provenance.get("config_path")
    file_hash = provenance.get("config_file_sha256")
    if not isinstance(path_raw, str) or not path_raw:
        failures.append("config_source_path_invalid")
    else:
        path = Path(path_raw)
        if not path.is_file() or hash_file(path) != file_hash:
            failures.append("config_source_hash_mismatch")
        else:
            try:
                source = Order8NaturalContactConfig.from_dict(
                    load_config(path).get("order8", {})
                )
                if source != config:
                    failures.append("config_source_payload_mismatch")
            except (OSError, ValueError, SchemaValidationError, TypeError):
                failures.append("config_source_payload_invalid")
    return config


def _validate_task_provenance(
    provenance: dict[str, Any],
    failures: list[str],
) -> None:
    expected = default_grasp_carry_task_spec()
    if provenance.get("task_id") != expected.task_id:
        failures.append("task_id_mismatch")
    if provenance.get("task_spec") != expected.to_dict():
        failures.append("task_spec_payload_mismatch")
    if provenance.get("task_spec_hash") != expected.stable_hash():
        failures.append("task_spec_hash_mismatch")


def _load_bound_graph(
    provenance: dict[str, Any], failures: list[str]
) -> MorphologyGraph | None:
    try:
        graph = MorphologyGraph.from_dict(provenance.get("morphology_graph"))
    except (SchemaValidationError, TypeError):
        failures.append("graph_payload_invalid")
        return None
    if provenance.get("graph_id") != graph.graph_id:
        failures.append("graph_id_mismatch")
    if provenance.get("graph_hash") != graph.stable_hash():
        failures.append("graph_hash_mismatch")
    return graph


def _load_bound_models(
    provenance: dict[str, Any], failures: list[str]
) -> tuple[Any | None, str | None]:
    backend_path_raw = provenance.get("backend_config_path")
    backend_file_hash = provenance.get("backend_config_file_sha256")
    backend_hash: str | None = None
    backend_config = None
    if not isinstance(backend_path_raw, str) or not backend_path_raw:
        failures.append("backend_config_path_invalid")
    else:
        backend_path = Path(backend_path_raw)
        if not backend_path.is_file() or hash_file(backend_path) != backend_file_hash:
            failures.append("backend_config_file_hash_mismatch")
        else:
            try:
                backend_config = load_isaac_lab_backend_config(backend_path)
                backend_hash = backend_config.stable_hash()
                if provenance.get("backend_config_hash") != backend_hash:
                    failures.append("backend_config_hash_mismatch")
            except (OSError, ValueError, SchemaValidationError, TypeError):
                failures.append("backend_config_invalid")

    robot_path_raw = provenance.get("robot_model_config_path")
    robot_file_hash = provenance.get("robot_model_config_file_sha256")
    if not isinstance(robot_path_raw, str) or not robot_path_raw:
        failures.append("robot_model_config_path_invalid")
        return None, backend_hash
    robot_path = Path(robot_path_raw)
    if not robot_path.is_file() or hash_file(robot_path) != robot_file_hash:
        failures.append("robot_model_config_file_hash_mismatch")
        return None, backend_hash
    if backend_config is not None and str(robot_path) != str(
        Path(backend_config.robot_model_config_path).resolve()
    ):
        failures.append("robot_model_backend_path_mismatch")
    try:
        physical_model = build_physical_model_from_config(robot_path)
    except (OSError, ValueError, SchemaValidationError, TypeError):
        failures.append("physical_model_rebuild_failed")
        return None, backend_hash
    if provenance.get("physical_model_hash") != physical_model.stable_hash():
        failures.append("physical_model_hash_mismatch")
    return physical_model, backend_hash


def _validate_wrapper_binding(
    result: Order8IsaacNaturalContactResult,
    config: Order8NaturalContactConfig | None,
    graph: MorphologyGraph | None,
    provenance: dict[str, Any],
    failures: list[str],
) -> None:
    if provenance.get("real_requested") is not True:
        failures.append("run_provenance_not_real")
    if config is not None and result.config_hash != config.stable_hash():
        failures.append("wrapper_config_hash_mismatch")
    if graph is not None:
        if result.graph_id != graph.graph_id:
            failures.append("wrapper_graph_id_mismatch")
        if result.graph_hash != graph.stable_hash():
            failures.append("wrapper_graph_hash_mismatch")


def _validate_source_and_collision_provenance(
    provenance: dict[str, Any],
    *,
    physical_model: Any | None,
    collision_hash: str | None,
    failures: list[str],
) -> None:
    if physical_model is None:
        return
    expected_source_path = Path(physical_model.urdf_path).resolve()
    source_path_raw = provenance.get("source_urdf_path")
    if not isinstance(source_path_raw, str) or not source_path_raw:
        failures.append("source_urdf_path_invalid")
    else:
        source_path = Path(source_path_raw).resolve()
        if source_path != expected_source_path or not source_path.is_file():
            failures.append("source_urdf_path_mismatch")
        elif provenance.get("source_urdf_sha256") != hash_file(source_path):
            failures.append("source_urdf_hash_mismatch")
    if (
        collision_hash is None
        or provenance.get("collision_geometry_content_hash") != collision_hash
    ):
        failures.append("collision_geometry_provenance_mismatch")


def _validate_generated_artifacts(
    report: dict[str, Any],
    provenance: dict[str, Any],
    failures: list[str],
) -> None:
    generated_root_raw = provenance.get("generated_usd_dir")
    if not isinstance(generated_root_raw, str) or not generated_root_raw:
        failures.append("generated_usd_dir_invalid")
        return
    generated_root = Path(generated_root_raw).resolve()

    urdf_path = _bound_generated_path(
        report.get("generated_urdf_path"),
        generated_root,
        "generated_urdf_path",
        failures,
    )
    if urdf_path is not None:
        actual_hash = hash_file(urdf_path)
        if report.get("generated_urdf_sha256") != actual_hash:
            failures.append("generated_urdf_hash_mismatch")

    usd_path = _bound_generated_path(
        report.get("usd_path"),
        generated_root,
        "generated_usd_path",
        failures,
    )
    if usd_path is None:
        return
    actual_usd_hash = hash_file(usd_path)
    if report.get("generated_usd_sha256") != actual_usd_hash or report.get(
        "order8_natural_contact_generated_usd_sha256"
    ) != actual_usd_hash:
        failures.append("generated_usd_hash_mismatch")
    actual_bundle_hash = hash_directory_manifest(usd_path.parent)
    if report.get("generated_usd_bundle_hash") != actual_bundle_hash or report.get(
        "order8_natural_contact_generated_usd_bundle_hash"
    ) != actual_bundle_hash:
        failures.append("generated_usd_bundle_hash_mismatch")


def _bound_generated_path(
    path_raw: object,
    generated_root: Path,
    label: str,
    failures: list[str],
) -> Path | None:
    if not isinstance(path_raw, str) or not path_raw:
        failures.append(f"{label}_invalid")
        return None
    path = Path(path_raw).resolve()
    if not path.is_file() or not path.is_relative_to(generated_root):
        failures.append(f"{label}_invalid")
        return None
    return path


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _finite_positive(value: object) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _acceptance_report(
    *,
    path: Path,
    artifact_loaded: bool,
    real_isaac_passed: bool,
    evidence_integrity_passed: bool,
    no_mislabeling_passed: bool,
    failures: list[str],
) -> Order8AcceptanceReport:
    unique_failures = list(dict.fromkeys(failures))
    completion = bool(
        artifact_loaded
        and real_isaac_passed
        and evidence_integrity_passed
        and no_mislabeling_passed
        and not unique_failures
    )
    return Order8AcceptanceReport(
        artifact_loaded=artifact_loaded,
        real_isaac_passed=real_isaac_passed,
        evidence_integrity_passed=evidence_integrity_passed,
        no_mislabeling_passed=no_mislabeling_passed,
        completion_passed=completion,
        failures=unique_failures,
        metrics={
            "artifact_loaded": float(artifact_loaded),
            "real_isaac_passed": float(real_isaac_passed),
            "evidence_integrity_passed": float(evidence_integrity_passed),
            "no_mislabeling_passed": float(no_mislabeling_passed),
            "completion_passed": float(completion),
        },
        artifacts={
            "order8_result": str(path),
            "order8_result_sha256": hash_file(path) if path.is_file() else "",
        },
    )
