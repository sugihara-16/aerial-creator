from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from amsrr.acceptance.p2_acceptance import P2AcceptanceCriteria, P2AcceptanceReport, run_p2_acceptance
from amsrr.schemas.common import SchemaBase, SchemaValidationError
from amsrr.schemas.task_spec import TaskSpec


@dataclass
class P2CompletionCriteria(SchemaBase):
    acceptance_criteria: P2AcceptanceCriteria = field(default_factory=P2AcceptanceCriteria)
    phase_label: str = "P2"

    def validate(self) -> None:
        if self.phase_label != "P2":
            raise SchemaValidationError("P2CompletionCriteria.phase_label must be 'P2'")


@dataclass
class P2CompletionReport(SchemaBase):
    phase_label: str
    passed: bool
    acceptance_report: P2AcceptanceReport
    completion_checks: dict[str, bool] = field(default_factory=dict)
    failure_reasons: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.phase_label != "P2":
            raise SchemaValidationError("P2CompletionReport.phase_label must be 'P2'")
        if self.passed and self.failure_reasons:
            raise SchemaValidationError("P2CompletionReport cannot pass with failure_reasons")
        if self.passed and not all(self.completion_checks.values()):
            raise SchemaValidationError("P2CompletionReport cannot pass with failed completion checks")


def run_p2_completion(
    base_task_spec: TaskSpec,
    *,
    criteria: P2CompletionCriteria | None = None,
    archive_path: str | Path | None = None,
) -> P2CompletionReport:
    """Run the complete P2 milestone gate for v0.4 Section 24.3."""

    criteria = criteria or P2CompletionCriteria()
    acceptance_report = run_p2_acceptance(
        base_task_spec,
        criteria=criteria.acceptance_criteria,
        archive_path=archive_path,
    )
    checks = _completion_checks(acceptance_report)
    failure_reasons = list(acceptance_report.failure_reasons)
    for check_name, passed in checks.items():
        if not passed and not any(check_name in reason for reason in failure_reasons):
            failure_reasons.append(f"P2 completion check failed: {check_name}")
    return P2CompletionReport(
        phase_label=criteria.phase_label,
        passed=all(checks.values()),
        acceptance_report=acceptance_report,
        completion_checks=checks,
        failure_reasons=failure_reasons,
    )


def _completion_checks(report: P2AcceptanceReport) -> dict[str, bool]:
    return {
        "p2_acceptance_passed": report.passed,
        "valid_design_rate_gate": report.valid_design_rate >= report.min_valid_design_rate,
        "required_slot_coverage_gate": (
            report.accepted_design_count > 0
            and report.accepted_required_slot_coverage_min >= report.min_required_slot_coverage
        ),
        "closed_loop_invalid_rejection_gate": report.closed_loop_invalid_rejected,
        "feasibility_labels_stored_gate": (
            report.closed_loop_feasibility_label_stored
            and report.feasibility_label_archive_count > 0
            and report.feasibility_label_valid_count == report.feasibility_label_archive_count
        ),
    }
