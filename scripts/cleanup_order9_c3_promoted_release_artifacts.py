#!/usr/bin/env python3
"""Audit and remove superseded Order-9 C3 artifacts after promotion.

The command is dry-run by default.  ``--apply`` is accepted only after every
file in the promoted-release ledger has an intact protected copy and no
candidate contains a protected source file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


C3_RELATIVE = Path(
    "artifacts/p4_full/order9/stages/c3_pi_l_ppo_arbitrary_morphology"
)
LEDGER_RELATIVE = Path("for_codex/C3_PROMOTED_UPDATE18_RELEASE_LEDGER.json")
REPORT_RELATIVE = Path("for_codex/C3_ARTIFACT_CLEANUP_REPORT.json")

KEEP_C3_CHILDREN = frozenset(
    {
        "actuator_aware_nominal_preload_v2",
        "actuator_aware_nominal_preload_v3",
        "actuator_aware_nominal_preload_v4",
        "contact_residual_initializers",
        "contact_space_initializers",
        "diagnostics",
        "execution_bundles",
        "morphology_invariant_compression_initializers",
        "nominal_trajectories_release_smooth_clear300_all_v16",
        "nominal_trajectories_val33_open35_smooth_clear300_v18",
        "rollout_buckets_current_lineage_v5",
        "training_lineages",
    }
)

ACCEPTED_NOMINAL_MANIFEST_RELATIVE = Path(
    "nominal_trajectories_release_smooth_clear300_all_v16"
    "/manifest_val33_open35_v19.json"
)
KEEP_LINEAGES = frozenset(
    {
        "contact_space_projected_qpid_physical_feasibility_from_zero_command_v1",
        "incremental_full_action_min2_clear300_from_update3_v1",
        "incremental_full_action_min2_clear300_preload_screened_from_update3_v2",
    }
)
KEEP_CONTACT_SPACE_CHILDREN = frozenset(
    {"modules_2_3_normal_contact_quality_min2_v1"}
)
KEEP_INCREMENTAL_CHILDREN = frozenset(
    {
        "modules_2_4",
        "modules_2_5",
        "modules_2_6",
        "modules_2_7",
        "modules_2_7_ppo_plus_probe_teacher_v1_from_update16",
        "modules_2_8_teacher_pretrain_then_ppo_v1_from_update17",
        "orchestrator_logs",
        "phase_reset_banks",
        "stagewise_runner_state.json",
    }
)
KEEP_AUDIT_CHILDREN = frozenset(
    {"preload_audit_modules2_8_achieved_refined_v3.json"}
)
KEEP_DIAGNOSTIC_CHILDREN = frozenset(
    {"bucket40_multistate_signed_normal_probe_v1"}
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _allocated_size(path: Path) -> int:
    if path.is_symlink() or path.is_file():
        return path.lstat().st_blocks * 512
    return sum(
        child.lstat().st_blocks * 512
        for child in path.rglob("*")
        if not child.is_symlink() or child.exists()
    ) + path.lstat().st_blocks * 512


def _children_not_in(root: Path, keep: frozenset[str]) -> Iterable[Path]:
    if not root.is_dir():
        raise RuntimeError(f"required cleanup root is missing: {root}")
    return (child for child in root.iterdir() if child.name not in keep)


def _deduplicate(targets: Iterable[tuple[Path, str]]) -> list[tuple[Path, str]]:
    ordered = sorted(targets, key=lambda item: (len(item[0].parts), str(item[0])))
    kept: list[tuple[Path, str]] = []
    for path, reason in ordered:
        if any(parent == path or parent in path.parents for parent, _ in kept):
            continue
        kept.append((path, reason))
    return kept


def _candidate_targets(c3: Path) -> list[tuple[Path, str]]:
    training = c3 / "training_lineages"
    contact_space = (
        training
        / "contact_space_projected_qpid_physical_feasibility_from_zero_command_v1"
    )
    incremental = (
        training
        / "incremental_full_action_min2_clear300_preload_screened_from_update3_v2"
    )
    audit = training / "incremental_full_action_min2_clear300_from_update3_v1"
    diagnostics = c3 / "diagnostics"

    targets: list[tuple[Path, str]] = []
    targets.extend(
        (path, "superseded C3 top-level experiment or generated artifact")
        for path in _children_not_in(c3, KEEP_C3_CHILDREN)
    )
    targets.extend(
        (path, "superseded PPO lineage")
        for path in _children_not_in(training, KEEP_LINEAGES)
    )
    targets.extend(
        (path, "superseded contact-space branch")
        for path in _children_not_in(contact_space, KEEP_CONTACT_SPACE_CHILDREN)
    )
    targets.extend(
        (path, "superseded incremental-training branch")
        for path in _children_not_in(incremental, KEEP_INCREMENTAL_CHILDREN)
    )
    targets.extend(
        (path, "superseded preload audit")
        for path in _children_not_in(audit, KEEP_AUDIT_CHILDREN)
    )
    targets.extend(
        (path, "superseded diagnostic")
        for path in _children_not_in(diagnostics, KEEP_DIAGNOSTIC_CHILDREN)
    )

    retained_lineage_roots = (
        contact_space / "modules_2_3_normal_contact_quality_min2_v1",
        *(incremental / name for name in KEEP_INCREMENTAL_CHILDREN),
    )
    for root in retained_lineage_roots:
        if not root.is_dir():
            continue
        targets.extend(
            (path, "reproducible raw rollout tensor directory")
            for path in root.rglob("raw")
            if path.is_dir()
        )
        targets.extend(
            (path, "reproducible evaluation rollout tensor")
            for path in root.rglob("evaluation_rollout.pt")
            if path.is_file()
        )
    return _deduplicate(targets)


def _load_and_verify_ledger(
    repository: Path,
) -> tuple[dict[str, Any], set[Path], list[dict[str, str]]]:
    ledger_path = repository / LEDGER_RELATIVE
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    protected_sources: set[Path] = set()
    errors: list[dict[str, str]] = []
    for record in ledger["protected_files"]:
        source = (repository / record["source_path"]).resolve()
        copy = (repository / record["protected_copy_path"]).resolve()
        expected = record["sha256"]
        protected_sources.add(source)
        for role, path in (("source", source), ("protected_copy", copy)):
            if not path.is_file():
                errors.append({"role": role, "path": str(path), "error": "missing"})
            elif _sha256(path) != expected:
                errors.append(
                    {"role": role, "path": str(path), "error": "sha256_mismatch"}
                )
    return ledger, protected_sources, errors


def _verify_transitive_runtime_dependencies(
    repository: Path,
    ledger: dict[str, Any],
) -> list[dict[str, str]]:
    """Validate hash-bound files reached through the accepted nominal set.

    A direct file ledger is insufficient when a retained set manifest points
    to a component directory outside its own tree.  Run the production byte
    validator for both the source and protected runtime copies so cleanup
    cannot omit such a transitive dependency.
    """

    import sys

    repository_text = str(repository)
    if repository_text not in sys.path:
        sys.path.insert(0, repository_text)

    from amsrr.training.order9_c3_nominal_trajectory import (
        validate_order9_c3_nominal_trajectory_set_bytes,
    )

    errors: list[dict[str, str]] = []
    for record in ledger["protected_runtime_directories"]:
        if record["role"] != "accepted_nominal_set":
            continue
        for role, root_key in (
            ("source", "source_path"),
            ("protected_copy", "protected_copy_path"),
        ):
            manifest = (
                repository
                / record[root_key]
                / ACCEPTED_NOMINAL_MANIFEST_RELATIVE.name
            )
            try:
                validate_order9_c3_nominal_trajectory_set_bytes(
                    manifest,
                    repository_root=repository,
                )
            except Exception as exc:  # fail closed with a reportable reason
                errors.append(
                    {
                        "role": role,
                        "path": str(manifest),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return errors


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    else:
        raise RuntimeError(f"cleanup target disappeared: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path, default=REPORT_RELATIVE)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    repository = args.repository_root.resolve()
    if not (repository / ".git").exists():
        raise RuntimeError(f"not a repository root: {repository}")
    c3 = (repository / C3_RELATIVE).resolve()
    if c3 != repository / C3_RELATIVE or not c3.is_dir():
        raise RuntimeError(f"unexpected C3 artifact root: {c3}")

    ledger, protected_sources, ledger_errors = _load_and_verify_ledger(repository)
    transitive_runtime_errors = _verify_transitive_runtime_dependencies(
        repository,
        ledger,
    )
    targets = _candidate_targets(c3)
    containment_errors: list[dict[str, str]] = []
    for target, _ in targets:
        resolved = target.resolve()
        if resolved == c3 or c3 not in resolved.parents:
            containment_errors.append(
                {"target": str(target), "error": "outside_exact_c3_root"}
            )
        for source in protected_sources:
            if source == resolved or resolved in source.parents:
                containment_errors.append(
                    {
                        "target": str(target),
                        "protected_source": str(source),
                        "error": "contains_protected_source",
                    }
                )

    records = [
        {
            "path": str(path.relative_to(repository)),
            "reason": reason,
            "allocated_size_bytes": _allocated_size(path),
            "kind": "directory" if path.is_dir() and not path.is_symlink() else "file",
        }
        for path, reason in targets
    ]
    pre_delete_bytes = sum(int(item["allocated_size_bytes"]) for item in records)
    safe = (
        not ledger_errors
        and not transitive_runtime_errors
        and not containment_errors
    )
    if args.apply and not safe:
        raise RuntimeError("cleanup safety audit failed; refusing deletion")

    deleted: list[str] = []
    if args.apply:
        for path, _ in targets:
            _remove(path)
            deleted.append(str(path.relative_to(repository)))

    report = {
        "report_version": "order9_c3_artifact_cleanup_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "applied" if args.apply else "dry_run",
        "release_id": ledger["release_id"],
        "protected_release_ledger": str(LEDGER_RELATIVE),
        "protected_file_count": len(ledger["protected_files"]),
        "safety_audit_passed": safe,
        "ledger_errors": ledger_errors,
        "transitive_runtime_errors": transitive_runtime_errors,
        "containment_errors": containment_errors,
        "retained_c3_children": sorted(KEEP_C3_CHILDREN),
        "retained_training_lineages": sorted(KEEP_LINEAGES),
        "target_count": len(records),
        "target_allocated_size_bytes": pre_delete_bytes,
        "targets": records,
        "deleted_count": len(deleted),
        "deleted_paths": deleted,
        "deletion_recovery": (
            "Not directly recoverable. Promotion-critical files are retained in the "
            "protected release; raw tensors are reproducible from retained inputs."
        ),
    }
    report_path = args.report
    if not report_path.is_absolute():
        report_path = repository / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: report[k] for k in (
        "mode",
        "release_id",
        "protected_file_count",
        "safety_audit_passed",
        "target_count",
        "target_allocated_size_bytes",
        "deleted_count",
    )}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
