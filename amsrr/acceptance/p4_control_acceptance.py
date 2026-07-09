from __future__ import annotations

from dataclasses import dataclass, field

from amsrr.logging import EpisodeArchive
from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_non_empty


P4_CONTROL_REQUIRED_SMOKES = (
    "single_module_hover",
    "fixed_morphology_hover",
    "fixed_morphology_waypoint",
)


@dataclass
class P4ControlSmokeResult(SchemaBase):
    smoke_name: str
    attempted: bool
    passed: bool
    skipped: bool = False
    isaac_backed: bool = False
    backend: str = "isaac_lab"
    skip_reason: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        require_non_empty(self.smoke_name, "P4ControlSmokeResult.smoke_name")
        require_non_empty(self.backend, "P4ControlSmokeResult.backend")
        if self.passed and (not self.attempted or self.skipped):
            raise SchemaValidationError("P4ControlSmokeResult cannot pass when not attempted or skipped")


@dataclass
class P4ControlAcceptanceReport(SchemaBase):
    fast_gate_passed: bool
    real_isaac_smoke_passed: bool
    completion_passed: bool
    archive_count: int
    smoke_result_count: int
    controller_commands_saved: bool
    runtime_observations_saved: bool
    actuator_target_records_saved: bool
    controller_metrics_saved: bool
    actuator_metrics_saved: bool
    no_full_completion_claim: bool
    required_smoke_names: list[str]
    passed_smoke_names: list[str]
    skipped_smoke_names: list[str]
    metrics: dict[str, float] = field(default_factory=dict)
    failure_reasons: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.archive_count < 0:
            raise SchemaValidationError("P4ControlAcceptanceReport.archive_count must be non-negative")
        if self.smoke_result_count < 0:
            raise SchemaValidationError("P4ControlAcceptanceReport.smoke_result_count must be non-negative")
        for smoke_name in self.required_smoke_names:
            require_non_empty(smoke_name, "P4ControlAcceptanceReport.required_smoke_names")


def run_p4_control_acceptance(
    archives: list[EpisodeArchive],
    *,
    smoke_results: list[P4ControlSmokeResult] | None = None,
) -> P4ControlAcceptanceReport:
    """Evaluate the split P4-control gate without running Isaac itself."""

    smoke_results = smoke_results or []
    controller_commands_saved = _all_archives(archives, lambda archive: bool(archive.controller_commands))
    runtime_observations_saved = _all_archives(archives, lambda archive: bool(archive.runtime_observations))
    actuator_target_records_saved = _all_archives(archives, lambda archive: bool(archive.actuator_target_records))
    controller_metrics_saved = _all_archives(archives, _controller_metrics_saved)
    actuator_metrics_saved = _all_archives(archives, _actuator_metrics_saved)
    no_full_completion_claim = _all_archives(archives, _no_full_completion_claim)

    fast_failure_reasons = _fast_failure_reasons(
        archives=archives,
        controller_commands_saved=controller_commands_saved,
        runtime_observations_saved=runtime_observations_saved,
        actuator_target_records_saved=actuator_target_records_saved,
        controller_metrics_saved=controller_metrics_saved,
        actuator_metrics_saved=actuator_metrics_saved,
        no_full_completion_claim=no_full_completion_claim,
    )
    fast_gate_passed = not fast_failure_reasons
    real_isaac_smoke_passed = _real_isaac_smoke_passed(smoke_results)
    smoke_failure_reasons = _smoke_failure_reasons(smoke_results)
    completion_passed = fast_gate_passed and real_isaac_smoke_passed
    failure_reasons = list(fast_failure_reasons)
    if fast_gate_passed and not real_isaac_smoke_passed:
        failure_reasons.append("P4-control real Isaac smoke gate has not passed")
    failure_reasons.extend(smoke_failure_reasons)

    passed_smoke_names = sorted(
        result.smoke_name
        for result in smoke_results
        if result.passed and result.attempted and result.isaac_backed and not result.skipped
    )
    skipped_smoke_names = sorted(result.smoke_name for result in smoke_results if result.skipped)
    return P4ControlAcceptanceReport(
        fast_gate_passed=fast_gate_passed,
        real_isaac_smoke_passed=real_isaac_smoke_passed,
        completion_passed=completion_passed,
        archive_count=len(archives),
        smoke_result_count=len(smoke_results),
        controller_commands_saved=controller_commands_saved,
        runtime_observations_saved=runtime_observations_saved,
        actuator_target_records_saved=actuator_target_records_saved,
        controller_metrics_saved=controller_metrics_saved,
        actuator_metrics_saved=actuator_metrics_saved,
        no_full_completion_claim=no_full_completion_claim,
        required_smoke_names=list(P4_CONTROL_REQUIRED_SMOKES),
        passed_smoke_names=passed_smoke_names,
        skipped_smoke_names=skipped_smoke_names,
        metrics={
            "fast_gate_passed": 1.0 if fast_gate_passed else 0.0,
            "real_isaac_smoke_passed": 1.0 if real_isaac_smoke_passed else 0.0,
            "completion_passed": 1.0 if completion_passed else 0.0,
            "archive_count": float(len(archives)),
            "smoke_result_count": float(len(smoke_results)),
        },
        failure_reasons=failure_reasons,
    )


def _all_archives(archives: list[EpisodeArchive], predicate) -> bool:
    return bool(archives) and all(predicate(archive) for archive in archives)


def _controller_metrics_saved(archive: EpisodeArchive) -> bool:
    for command in archive.controller_commands:
        metrics = command.controller_status.metrics
        if "residual_norm" not in metrics and "allocation_residual_norm" not in metrics:
            return False
        if "clipped" not in metrics and "clipped_target_count" not in metrics:
            return False
    return bool(archive.controller_commands)


def _actuator_metrics_saved(archive: EpisodeArchive) -> bool:
    for record in archive.actuator_target_records:
        if not isinstance(record, dict):
            return False
        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            return False
        required = {"allocation_residual_norm", "clipped_target_count", "missing_actuator_count"}
        if not required.issubset(metrics):
            return False
    return bool(archive.actuator_target_records)


def _no_full_completion_claim(archive: EpisodeArchive) -> bool:
    artifacts = archive.rollout_artifacts
    return (
        artifacts.get("phase") == "P4-control"
        and artifacts.get("is_p4_full_completion") is False
        and artifacts.get("physical_success_claim") is False
        and archive.metrics.get("p4_full_completion", 0.0) == 0.0
    )


def _real_isaac_smoke_passed(smoke_results: list[P4ControlSmokeResult]) -> bool:
    results_by_name = {result.smoke_name: result for result in smoke_results}
    for smoke_name in P4_CONTROL_REQUIRED_SMOKES:
        result = results_by_name.get(smoke_name)
        if result is None:
            return False
        if not (result.attempted and result.passed and result.isaac_backed and not result.skipped):
            return False
    return True


def _fast_failure_reasons(
    *,
    archives: list[EpisodeArchive],
    controller_commands_saved: bool,
    runtime_observations_saved: bool,
    actuator_target_records_saved: bool,
    controller_metrics_saved: bool,
    actuator_metrics_saved: bool,
    no_full_completion_claim: bool,
) -> list[str]:
    reasons: list[str] = []
    if not archives:
        reasons.append("P4-control produced no archives")
    if not controller_commands_saved:
        reasons.append("P4-control archives are missing ControllerCommand records")
    if not runtime_observations_saved:
        reasons.append("P4-control archives are missing RuntimeObservation records")
    if not actuator_target_records_saved:
        reasons.append("P4-control archives are missing actuator target records")
    if not controller_metrics_saved:
        reasons.append("P4-control archives are missing controller residual/clipping metrics")
    if not actuator_metrics_saved:
        reasons.append("P4-control archives are missing actuator target metrics")
    if not no_full_completion_claim:
        reasons.append("P4-control archives incorrectly claim P4 full completion or physical success")
    return reasons


def _smoke_failure_reasons(smoke_results: list[P4ControlSmokeResult]) -> list[str]:
    reasons: list[str] = []
    results_by_name = {result.smoke_name: result for result in smoke_results}
    for smoke_name in P4_CONTROL_REQUIRED_SMOKES:
        result = results_by_name.get(smoke_name)
        if result is None:
            reasons.append(f"P4-control missing real Isaac smoke result: {smoke_name}")
            continue
        if result.skipped:
            reasons.append(f"P4-control real Isaac smoke skipped: {smoke_name}")
        elif not result.attempted:
            reasons.append(f"P4-control real Isaac smoke not attempted: {smoke_name}")
        elif not result.isaac_backed:
            reasons.append(f"P4-control smoke was not Isaac-backed: {smoke_name}")
        elif not result.passed:
            reasons.append(f"P4-control real Isaac smoke failed: {smoke_name}")
    return reasons
