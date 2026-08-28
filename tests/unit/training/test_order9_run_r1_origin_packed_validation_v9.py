from __future__ import annotations

import importlib.util
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[3]
SOURCE = REPOSITORY / "scripts/order9_run_r1_origin_packed_validation_v9.py"


def _module():
    spec = importlib.util.spec_from_file_location("order9_r1_v9_runner", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_existing_single_retry_is_reused_before_new_isaac(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    candidate_id = "candidate-a"
    single_root = tmp_path / "single_retries" / candidate_id
    single_root.mkdir(parents=True)
    calls: list[Path] = []

    def complete(candidate_ids, *, output_root):
        assert candidate_ids == (candidate_id,)
        calls.append(output_root)
        return output_root == single_root

    monkeypatch.setattr(module, "_is_complete", complete)

    assert module._existing_retry_evidence_root(
        candidate_id,
        source_id="source-a",
        output_root=tmp_path,
    ) == single_root
    assert calls == [single_root]


def test_existing_bounded_retry_is_reused_when_single_is_absent(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    candidate_id = "candidate-a"
    bounded_root = tmp_path / "bounded_retries/source-a/chunk-000"
    (bounded_root / candidate_id).mkdir(parents=True)

    monkeypatch.setattr(
        module,
        "_is_complete",
        lambda candidate_ids, *, output_root: (
            candidate_ids == (candidate_id,) and output_root == bounded_root
        ),
    )

    assert module._existing_retry_evidence_root(
        candidate_id,
        source_id="source-a",
        output_root=tmp_path,
    ) == bounded_root
