from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch

REPOSITORY = Path(__file__).resolve().parents[3]
SOURCE = REPOSITORY / "scripts/order9_run_r1_batched_nominal_calibration_v7.py"


def _module():
    spec = importlib.util.spec_from_file_location("order9_r1_batched_runner", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _job(case_root: Path) -> dict[str, object]:
    return {
        "name": "candidate-a",
        "argv": [
            "--output-raw",
            str(case_root / "isaac/evaluation_rollout.pt"),
        ],
        "log_path": str(case_root / "isaac/evaluation.log"),
    }


def test_existing_batch_candidate_count_reads_bound_metadata(tmp_path: Path) -> None:
    module = _module()
    case_root = tmp_path / "candidate-a"
    isaac = case_root / "isaac"
    isaac.mkdir(parents=True)
    (case_root / "case_manifest.json").write_text(
        json.dumps({"candidate_id": "candidate-a"}) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {"metadata": {"r1_batched_candidate_count": 4}, "tensors": {}},
        isaac / "evaluation_rollout.pt",
    )

    assert module._existing_batch_candidate_count(case_root / "case_manifest.json") == 4


def test_valid_case_returns_true_for_successful_bound_batch(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    monkeypatch.setattr(module, "load_order9_r1_isaac_case", lambda *_args: object())
    monkeypatch.setattr(
        module,
        "validate_order9_r1_nominal_v7_isaac_result",
        lambda *_args, **_kwargs: (SimpleNamespace(success=True),),
    )
    monkeypatch.setattr(module, "hash_file", lambda _path: "bound-sha")
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: {
            "metadata": {
                "r1_batched_nominal_runtime_version": (
                    "order9_r1_environment_wise_nominal_isaac_batch_v1"
                ),
                "r1_batched_nominal_wrapper_sha256": "bound-sha",
                "r1_batched_nominal_runtime_sha256": "bound-sha",
                "r1_batched_candidate_count": 31,
            }
        },
    )

    assert module._valid_case(tmp_path / "case_manifest.json", {}) is True


def test_prior_failed_batch_remains_single_retry_after_restart(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    monkeypatch.setattr(module, "REPOSITORY", tmp_path)
    monkeypatch.setattr(module, "_valid_case", lambda *_args, **_kwargs: False)

    case_root = tmp_path / "candidate-a"
    case_root.mkdir()
    manifest = case_root / "case_manifest.json"
    manifest.write_text(
        json.dumps({"candidate_id": "candidate-a"}) + "\n",
        encoding="utf-8",
    )
    quarantine = tmp_path / "quarantine/candidate-a"
    quarantine.mkdir(parents=True)
    (quarantine / "attempt_000.json").write_text(
        json.dumps(
            {"reason": ("bounded batch failed; " + module.SINGLE_RETRY_REASON_TOKEN)}
        )
        + "\n",
        encoding="utf-8",
    )
    source_manifest = tmp_path / "persistent_bucket_jobs.json"
    source_manifest.write_text(
        json.dumps({"jobs": [_job(case_root)]}) + "\n",
        encoding="utf-8",
    )

    pending = module._prepare_pending_manifests(
        "source-a",
        source_manifest,
        protocol={"interrupted_attempt_quarantine_root": "quarantine"},
        output_root=tmp_path / "runs",
    )

    assert len(pending) == 1
    pending_path, count = pending[0]
    payload = json.loads(pending_path.read_text(encoding="utf-8"))
    assert count == 1
    assert payload["execution_kind"] == "single_candidate_retry"
    assert payload["pending_job_count"] == 1


def test_prior_large_batch_failure_restarts_as_bounded_retry(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    monkeypatch.setattr(module, "REPOSITORY", tmp_path)
    monkeypatch.setattr(module, "_valid_case", lambda *_args, **_kwargs: False)

    case_root = tmp_path / "candidate-a"
    case_root.mkdir()
    (case_root / "case_manifest.json").write_text(
        json.dumps({"candidate_id": "candidate-a"}) + "\n",
        encoding="utf-8",
    )
    quarantine = tmp_path / "quarantine/candidate-a"
    quarantine.mkdir(parents=True)
    (quarantine / "attempt_000.json").write_text(
        json.dumps({"reason": module.BOUNDED_RETRY_REASON_TOKEN}) + "\n",
        encoding="utf-8",
    )
    source_manifest = tmp_path / "persistent_bucket_jobs.json"
    source_manifest.write_text(
        json.dumps({"jobs": [_job(case_root)]}) + "\n",
        encoding="utf-8",
    )

    pending = module._prepare_pending_manifests(
        "source-a",
        source_manifest,
        protocol={"interrupted_attempt_quarantine_root": "quarantine"},
        output_root=tmp_path / "runs",
    )

    assert len(pending) == 1
    pending_path, count = pending[0]
    payload = json.loads(pending_path.read_text(encoding="utf-8"))
    assert count == 1
    assert payload["execution_kind"] == "bounded_batch_retry"


def test_excluded_candidate_manifest_is_sorted_unique(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "excluded.json"
    path.write_text(
        json.dumps({"candidate_ids": ["candidate-a", "candidate-b"]}) + "\n",
        encoding="utf-8",
    )

    assert module._load_excluded_candidate_ids(path) == frozenset(
        {"candidate-a", "candidate-b"}
    )


def test_excluded_candidate_is_not_scheduled(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    monkeypatch.setattr(module, "REPOSITORY", tmp_path)
    monkeypatch.setattr(module, "_valid_case", lambda *_args, **_kwargs: False)
    case_a = tmp_path / "candidate-a"
    case_b = tmp_path / "candidate-b"
    for case in (case_a, case_b):
        case.mkdir()
        (case / "case_manifest.json").write_text(
            json.dumps({"candidate_id": case.name}) + "\n",
            encoding="utf-8",
        )
    source_manifest = tmp_path / "persistent_bucket_jobs.json"
    job_a = _job(case_a)
    job_b = _job(case_b)
    job_b["name"] = "candidate-b"
    source_manifest.write_text(
        json.dumps({"jobs": [job_a, job_b]}) + "\n",
        encoding="utf-8",
    )

    pending = module._prepare_pending_manifests(
        "source-a",
        source_manifest,
        protocol={"interrupted_attempt_quarantine_root": "quarantine"},
        output_root=tmp_path / "runs",
        excluded_candidate_ids=frozenset({"candidate-a"}),
    )

    assert len(pending) == 1
    payload = json.loads(pending[0][0].read_text(encoding="utf-8"))
    assert [job["name"] for job in payload["jobs"]] == ["candidate-b"]
