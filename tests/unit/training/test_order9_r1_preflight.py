from __future__ import annotations

from pathlib import Path

import pytest

from amsrr.schemas.common import SchemaValidationError
from amsrr.training.order9_r1_preflight import (
    ORDER9_R1_MINIMUM_FREE_GIB,
    ORDER9_R1_PREFLIGHT_VERSION,
    Order9R1PreflightReport,
    _load_release_ledger,
    _repository_artifact_path,
    _verify_calibration_protocol_approval,
    _verify_curriculum,
    preflight_order9_r1_teacher_collection,
)


def test_r1_preflight_report_distinguishes_calibration_from_collection() -> None:
    report = Order9R1PreflightReport(
        preflight_version=ORDER9_R1_PREFLIGHT_VERSION,
        mode="calibration",
        passed=True,
        ready_for_calibration=True,
        ready_for_collection=False,
        checks={"immutable_c3_base": True, "r1_distribution_adapter": False},
        failures=[],
    )

    assert (
        Order9R1PreflightReport.from_json(report.to_json()).to_dict()
        == report.to_dict()
    )

    report.mode = "collection"
    with pytest.raises(SchemaValidationError, match="readiness is inconsistent"):
        report.validate()


def test_r1_preflight_cannot_weaken_storage_floor(tmp_path) -> None:
    with pytest.raises(ValueError, match="must be at least"):
        preflight_order9_r1_teacher_collection(
            repository_root=tmp_path,
            minimum_free_gib=ORDER9_R1_MINIMUM_FREE_GIB - 1.0,
            require_cuda=False,
        )


def test_r1_evidence_paths_cannot_escape_repository(tmp_path) -> None:
    assert _repository_artifact_path(tmp_path, "artifacts/evidence.json") == (
        tmp_path / "artifacts/evidence.json"
    )
    with pytest.raises(SchemaValidationError, match="repository-relative"):
        _repository_artifact_path(tmp_path, "/tmp/evidence.json")
    with pytest.raises(SchemaValidationError, match="escapes"):
        _repository_artifact_path(tmp_path, "../evidence.json")


def test_r1_calibration_preflight_requires_hash_bound_user_approval() -> None:
    repository = Path(".").resolve()
    learning = _verify_curriculum(
        repository / "configs/training/order9_learning_curriculum.yaml"
    )
    ledger = _load_release_ledger(
        repository / "for_codex/C3_PROMOTED_UPDATE18_RELEASE_LEDGER.json"
    )

    protocol = _verify_calibration_protocol_approval(
        repository / "configs/training/order9_r1_calibration_protocol_v1.yaml",
        repository=repository,
        learning=learning,
        ledger=ledger,
    )

    assert protocol.status == "approved"
    assert protocol.selection_bucket_count == 22
    assert protocol.confirmation_bucket_count == 14
