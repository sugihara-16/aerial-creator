from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

from amsrr.feasibility.morphology_flight import collision_geometry_content_hash
from amsrr.morphology.random_connected import morphology_structural_hash
from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_len, require_non_empty
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.order3 import ORDER3_ACTION_SIZE, Order3PolicyCheckpointMetadata
from amsrr.schemas.order3_rollout_condition import (
    Order3RolloutCondition,
    build_order3_rollout_condition,
)
from amsrr.schemas.policies import POLICY_COMMAND_CONTRACT_CENTROIDAL, PolicyCommand
from amsrr.simulation.order3_rollout_condition import (
    ORDER3_FREE_FLIGHT_REPORT_VERSION,
    Order3ConditionRealization,
    order3_terminal_evidence_start_s,
    order3_tracking_window_start_s,
)
from amsrr.simulation.random_morphology_takeoff import (
    FIXED_DOCK_JOINT_POSITION_TOLERANCE_RAD,
    RandomMorphologyTakeoffEnv,
    RandomMorphologyTakeoffResult,
    random_morphology_takeoff_result_from_report,
)
from amsrr.utils.hashing import hash_file


ORDER3_ISAAC_POLICY_ROLLOUT_VERSION = "order3_isaac_policy_rollout_v1"


@dataclass
class Order3IsaacPolicyRolloutConfig(SchemaBase):
    checkpoint_path: str
    expected_checkpoint_sha256: str
    stochastic: bool = False
    record_policy_transitions: bool = True
    external_wrench_body: list[float] = field(default_factory=lambda: [0.0] * 6)
    disturbance_start_s: float = 3.0
    disturbance_duration_s: float = 0.0
    rollout_condition: Order3RolloutCondition | None = None
    rollout_version: str = ORDER3_ISAAC_POLICY_ROLLOUT_VERSION
    p4_full_completion_claim: bool = False

    def validate(self) -> None:
        require_non_empty(self.checkpoint_path, "Order3IsaacPolicyRolloutConfig.checkpoint_path")
        require_non_empty(
            self.expected_checkpoint_sha256,
            "Order3IsaacPolicyRolloutConfig.expected_checkpoint_sha256",
        )
        if not _is_sha256(self.expected_checkpoint_sha256):
            raise SchemaValidationError(
                "Order3IsaacPolicyRolloutConfig.expected_checkpoint_sha256 must be sha256"
            )
        require_len(
            self.external_wrench_body,
            6,
            "Order3IsaacPolicyRolloutConfig.external_wrench_body",
        )
        if not all(math.isfinite(float(value)) for value in self.external_wrench_body):
            raise SchemaValidationError(
                "Order3IsaacPolicyRolloutConfig.external_wrench_body must be finite"
            )
        if self.disturbance_start_s < 0.0 or self.disturbance_duration_s < 0.0:
            raise SchemaValidationError(
                "Order3IsaacPolicyRolloutConfig disturbance timing must be non-negative"
            )
        if self.rollout_version != ORDER3_ISAAC_POLICY_ROLLOUT_VERSION:
            raise SchemaValidationError(
                f"Order3IsaacPolicyRolloutConfig.rollout_version must be "
                f"{ORDER3_ISAAC_POLICY_ROLLOUT_VERSION!r}"
            )
        if self.p4_full_completion_claim:
            raise SchemaValidationError("Order3 free-flight rollout is not P4 full completion")

    def effective_condition(self) -> Order3RolloutCondition:
        return self.rollout_condition or build_order3_rollout_condition(
            stage_id="legacy_takeoff_compatibility",
            task_mode="takeoff",
            seed=0,
            external_wrench_body=self.external_wrench_body,
            disturbance_start_s=self.disturbance_start_s,
            disturbance_duration_s=self.disturbance_duration_s,
        )


@dataclass
class Order3DeterministicBaselineRolloutConfig(SchemaBase):
    external_wrench_body: list[float] = field(default_factory=lambda: [0.0] * 6)
    disturbance_start_s: float = 3.0
    disturbance_duration_s: float = 0.0
    rollout_condition: Order3RolloutCondition | None = None
    p4_full_completion_claim: bool = False

    def validate(self) -> None:
        require_len(
            self.external_wrench_body,
            6,
            "Order3DeterministicBaselineRolloutConfig.external_wrench_body",
        )
        if not all(math.isfinite(float(value)) for value in self.external_wrench_body):
            raise SchemaValidationError("Order3 baseline disturbance wrench must be finite")
        if self.disturbance_start_s < 0.0 or self.disturbance_duration_s < 0.0:
            raise SchemaValidationError("Order3 baseline disturbance timing must be non-negative")
        if self.p4_full_completion_claim:
            raise SchemaValidationError("Order3 deterministic baseline is not P4 full completion")

    def effective_condition(self) -> Order3RolloutCondition:
        return self.rollout_condition or build_order3_rollout_condition(
            stage_id="legacy_takeoff_baseline_compatibility",
            task_mode="takeoff",
            seed=0,
            external_wrench_body=self.external_wrench_body,
            disturbance_start_s=self.disturbance_start_s,
            disturbance_duration_s=self.disturbance_duration_s,
        )


@dataclass
class Order3IsaacPolicyRolloutResult(SchemaBase):
    rollout_version: str
    graph_id: str
    checkpoint_sha256: str
    takeoff_result: RandomMorphologyTakeoffResult
    policy_decision_count: int
    policy_applied_count: int
    fallback_count: int
    transition_trace_count: int
    report_validation_failures: list[str]
    task_mode: str = "takeoff"
    rollout_condition: Order3RolloutCondition | None = None
    terminal_metrics: dict[str, Any] | None = None
    condition_realization: Order3ConditionRealization | None = None
    p4_full_completion_claim: bool = False

    def validate(self) -> None:
        if self.rollout_version != ORDER3_ISAAC_POLICY_ROLLOUT_VERSION:
            raise SchemaValidationError("Order3IsaacPolicyRolloutResult version mismatch")
        require_non_empty(self.graph_id, "Order3IsaacPolicyRolloutResult.graph_id")
        if not _is_sha256(self.checkpoint_sha256):
            raise SchemaValidationError("Order3IsaacPolicyRolloutResult checkpoint hash is invalid")
        for name in (
            "policy_decision_count",
            "policy_applied_count",
            "fallback_count",
            "transition_trace_count",
        ):
            if int(getattr(self, name)) < 0:
                raise SchemaValidationError(
                    f"Order3IsaacPolicyRolloutResult.{name} must be non-negative"
                )
        if self.policy_applied_count + self.fallback_count != self.policy_decision_count:
            raise SchemaValidationError(
                "Order3IsaacPolicyRolloutResult policy counts must partition decisions"
            )
        if self.takeoff_result.real_isaac_passed and self.report_validation_failures:
            raise SchemaValidationError(
                "passing Order3 rollout cannot contain report validation failures"
            )
        if self.task_mode not in {"hover", "waypoint", "takeoff"}:
            raise SchemaValidationError("Order3IsaacPolicyRolloutResult.task_mode is invalid")
        if self.rollout_condition is not None:
            if self.task_mode != self.rollout_condition.task_mode:
                raise SchemaValidationError(
                    "Order3 rollout result task mode must match its condition"
                )
            if self.condition_realization is not None and (
                self.condition_realization.condition_hash
                != self.rollout_condition.condition_hash
            ):
                raise SchemaValidationError(
                    "Order3 rollout condition realization hash mismatch"
                )
        if (
            self.task_mode != "takeoff"
            and self.takeoff_result.real_isaac_passed
            and self.terminal_metrics is None
        ):
            raise SchemaValidationError(
                "Order3 in-air rollout result requires terminal metrics"
            )
        if self.p4_full_completion_claim:
            raise SchemaValidationError("Order3 free-flight result is not P4 full completion")


class Order3IsaacPolicyRolloutEnv:
    """Run a versioned Order-3 pi_L under a hash-bound Isaac rollout condition."""

    def __init__(
        self,
        *,
        config: Order3IsaacPolicyRolloutConfig,
        takeoff_env: RandomMorphologyTakeoffEnv,
        viewer: str | None = None,
        realtime_playback: bool = False,
        keep_open_after_rollout_s: float = 0.0,
    ) -> None:
        if viewer not in {None, "kit"}:
            raise ValueError("Order3 viewer must be None or 'kit'")
        if keep_open_after_rollout_s < 0.0:
            raise ValueError("keep_open_after_rollout_s must be non-negative")
        if viewer is None and (realtime_playback or keep_open_after_rollout_s > 0.0):
            raise ValueError(
                "Order3 realtime playback and post-rollout hold require viewer='kit'"
            )
        self.config = config
        self.takeoff_env = takeoff_env
        self.viewer = viewer
        self.realtime_playback = realtime_playback
        self.keep_open_after_rollout_s = keep_open_after_rollout_s
        if (
            self.takeoff_env.config.control_contract_version
            != POLICY_COMMAND_CONTRACT_CENTROIDAL
        ):
            raise SchemaValidationError(
                "Order3IsaacPolicyRolloutEnv requires centroidal_local_joint_v2 takeoff"
            )
        checkpoint_path = Path(self.config.checkpoint_path)
        if checkpoint_path.is_file():
            actual_hash = hash_file(checkpoint_path)
            if actual_hash != self.config.expected_checkpoint_sha256:
                raise SchemaValidationError("Order3 rollout checkpoint sha256 mismatch")

    def build_probe_command(self, morphology_graph: MorphologyGraph) -> list[str]:
        command = self.takeoff_env.build_probe_command(morphology_graph)
        command.extend(
            [
                "--order3-pi-l-checkpoint-path",
                self.config.checkpoint_path,
                "--order3-rollout-condition-json",
                self.config.effective_condition().to_canonical_json(),
            ]
        )
        if self.config.stochastic:
            command.append("--order3-pi-l-stochastic")
        if not self.config.record_policy_transitions:
            command.append("--no-order3-record-policy-transitions")
        if self.viewer is not None:
            command.extend(["--viz", self.viewer])
        if self.realtime_playback:
            command.append("--realtime-playback")
        if self.keep_open_after_rollout_s > 0.0:
            command.extend(
                ["--keep-open-after-smoke-s", str(self.keep_open_after_rollout_s)]
            )
        return command

    def run(
        self,
        morphology_graph: MorphologyGraph,
        *,
        dry_run: bool = True,
    ) -> Order3IsaacPolicyRolloutResult:
        dry_contract = self.takeoff_env.run(morphology_graph, dry_run=True)
        condition = self.config.effective_condition()
        if dry_run:
            dry_contract.report = {"probe_command": self.build_probe_command(morphology_graph)}
            return Order3IsaacPolicyRolloutResult(
                rollout_version=self.config.rollout_version,
                graph_id=morphology_graph.graph_id,
                checkpoint_sha256=self.config.expected_checkpoint_sha256,
                takeoff_result=dry_contract,
                policy_decision_count=0,
                policy_applied_count=0,
                fallback_count=0,
                transition_trace_count=0,
                report_validation_failures=[],
                task_mode=condition.task_mode,
                rollout_condition=condition,
                terminal_metrics=(
                    None
                    if condition.task_mode == "takeoff"
                    else {
                        "position_error_m": 0.0,
                        "attitude_error_rad": 0.0,
                        "linear_velocity_error_mps": 0.0,
                        "angular_velocity_error_rad_s": 0.0,
                        "within_tolerance_duration_s": 0.0,
                        "takeoff_height_gain_ratio": None,
                    }
                ),
            )
        availability = self.takeoff_env.backend.availability()
        if not availability.available:
            result = RandomMorphologyTakeoffResult(
                graph_id=morphology_graph.graph_id,
                attempted=False,
                dry_run=False,
                isaac_backed=False,
                unit_contract_passed=True,
                real_isaac_passed=False,
                placement=dry_contract.placement,
                metrics={**dry_contract.metrics, "isaac_backend_available": False},
                failure_reason=",".join(availability.missing_reasons),
            )
            return self._result(morphology_graph, result, ["isaac_backend_unavailable"])
        try:
            report = self.takeoff_env.command_executor(
                self.build_probe_command(morphology_graph),
                self.takeoff_env.config.command_timeout_s,
            )
        except Exception as exc:  # pragma: no cover - environment-specific subprocess failure.
            result = RandomMorphologyTakeoffResult(
                graph_id=morphology_graph.graph_id,
                attempted=True,
                dry_run=False,
                isaac_backed=True,
                unit_contract_passed=True,
                real_isaac_passed=False,
                placement=dry_contract.placement,
                metrics=dry_contract.metrics,
                failure_reason=str(exc),
            )
            return self._result(morphology_graph, result, ["probe_execution_failed"])
        collision_hash = collision_geometry_content_hash(
            self.takeoff_env.physical_model,
            mesh_search_dirs=self.takeoff_env.config.mesh_search_dirs,
        )
        if condition.task_mode == "takeoff":
            takeoff_result = random_morphology_takeoff_result_from_report(
                morphology_graph,
                placement=self.takeoff_env.placement_for(morphology_graph),
                report=report,
                expected_backend_config_hash=self.takeoff_env.backend.config.stable_hash(),
                expected_physical_model_hash=self.takeoff_env.physical_model.stable_hash(),
                expected_collision_geometry_hash=collision_hash,
                expected_config=self.takeoff_env.config,
                unit_metrics=dry_contract.metrics,
                expected_learned_policy=True,
            )
        else:
            takeoff_result = _order3_in_air_result_from_report(
                morphology_graph,
                report=report,
                dry_contract=dry_contract,
            )
        failures = _order3_report_failures(
            report,
            expected_checkpoint_sha256=self.config.expected_checkpoint_sha256,
            record_transitions=self.config.record_policy_transitions,
        )
        failures.extend(
            order3_condition_report_failures(
                report,
                expected_condition=condition,
            )
        )
        failures.extend(
            order3_provenance_report_failures(
                report,
                expected_backend_config_hash=self.takeoff_env.backend.config.stable_hash(),
                expected_physical_model_hash=self.takeoff_env.physical_model.stable_hash(),
                expected_collision_geometry_hash=collision_hash,
            )
        )
        if report.get("order3_structural_hash") != morphology_structural_hash(
            morphology_graph
        ):
            failures.append("mismatch:order3_structural_hash")
        if condition.task_mode != "takeoff":
            failures.extend(_order3_in_air_report_failures(report))
        failures = sorted(set(failures))
        if failures:
            takeoff_result.real_isaac_passed = False
            takeoff_result.failure_reason = "order3_report_validation_failed:" + ",".join(failures)
        takeoff_result.metrics["order3_report_validation_failures"] = failures
        return self._result(morphology_graph, takeoff_result, failures)

    def _result(
        self,
        morphology_graph: MorphologyGraph,
        takeoff_result: RandomMorphologyTakeoffResult,
        failures: list[str],
    ) -> Order3IsaacPolicyRolloutResult:
        report = takeoff_result.report
        return Order3IsaacPolicyRolloutResult(
            rollout_version=self.config.rollout_version,
            graph_id=morphology_graph.graph_id,
            checkpoint_sha256=self.config.expected_checkpoint_sha256,
            takeoff_result=takeoff_result,
            policy_decision_count=int(report.get("order3_pi_l_policy_decision_count", 0)),
            policy_applied_count=int(report.get("order3_pi_l_policy_applied_count", 0)),
            fallback_count=int(report.get("order3_pi_l_fallback_count", 0)),
            transition_trace_count=len(report.get("order3_pi_l_transition_traces", [])),
            report_validation_failures=list(failures),
            task_mode=self.config.effective_condition().task_mode,
            rollout_condition=self.config.effective_condition(),
            terminal_metrics=(
                report.get("order3_terminal_metrics")
                if isinstance(report.get("order3_terminal_metrics"), dict)
                else None
            ),
            condition_realization=_parse_condition_realization(report),
        )


class Order3DeterministicBaselineRolloutEnv:
    """Paired deterministic-v2 rollout under the exact same Order-3 condition."""

    def __init__(
        self,
        *,
        config: Order3DeterministicBaselineRolloutConfig,
        takeoff_env: RandomMorphologyTakeoffEnv,
    ) -> None:
        self.config = config
        self.takeoff_env = takeoff_env
        if takeoff_env.config.control_contract_version != POLICY_COMMAND_CONTRACT_CENTROIDAL:
            raise SchemaValidationError(
                "Order3 deterministic baseline requires centroidal_local_joint_v2"
            )

    def build_probe_command(self, morphology_graph: MorphologyGraph) -> list[str]:
        return [
            *self.takeoff_env.build_probe_command(morphology_graph),
            "--order3-deterministic-baseline-evaluation",
            "--order3-rollout-condition-json",
            self.config.effective_condition().to_canonical_json(),
        ]

    def run(
        self,
        morphology_graph: MorphologyGraph,
        *,
        dry_run: bool = True,
    ) -> RandomMorphologyTakeoffResult:
        dry_contract = self.takeoff_env.run(morphology_graph, dry_run=True)
        if dry_run:
            dry_contract.report = {"probe_command": self.build_probe_command(morphology_graph)}
            return dry_contract
        availability = self.takeoff_env.backend.availability()
        if not availability.available:
            return RandomMorphologyTakeoffResult(
                graph_id=morphology_graph.graph_id,
                attempted=False,
                dry_run=False,
                isaac_backed=False,
                unit_contract_passed=True,
                real_isaac_passed=False,
                placement=dry_contract.placement,
                metrics={**dry_contract.metrics, "isaac_backend_available": False},
                failure_reason=",".join(availability.missing_reasons),
            )
        report = self.takeoff_env.command_executor(
            self.build_probe_command(morphology_graph),
            self.takeoff_env.config.command_timeout_s,
        )
        collision_hash = collision_geometry_content_hash(
            self.takeoff_env.physical_model,
            mesh_search_dirs=self.takeoff_env.config.mesh_search_dirs,
        )
        condition = self.config.effective_condition()
        if condition.task_mode == "takeoff":
            result = random_morphology_takeoff_result_from_report(
                morphology_graph,
                placement=self.takeoff_env.placement_for(morphology_graph),
                report=report,
                expected_backend_config_hash=self.takeoff_env.backend.config.stable_hash(),
                expected_physical_model_hash=self.takeoff_env.physical_model.stable_hash(),
                expected_collision_geometry_hash=collision_hash,
                expected_config=self.takeoff_env.config,
                unit_metrics=dry_contract.metrics,
                expected_learned_policy=False,
            )
        else:
            result = _order3_in_air_result_from_report(
                morphology_graph,
                report=report,
                dry_contract=dry_contract,
            )
        failures: list[str] = []
        if report.get("order3_deterministic_baseline_rollout") is not True:
            failures.append("missing:order3_deterministic_baseline_rollout")
        failures.extend(
            order3_condition_report_failures(report, expected_condition=condition)
        )
        failures.extend(
            order3_provenance_report_failures(
                report,
                expected_backend_config_hash=self.takeoff_env.backend.config.stable_hash(),
                expected_physical_model_hash=self.takeoff_env.physical_model.stable_hash(),
                expected_collision_geometry_hash=collision_hash,
            )
        )
        if report.get("order3_structural_hash") != morphology_structural_hash(
            morphology_graph
        ):
            failures.append("mismatch:order3_structural_hash")
        if condition.task_mode != "takeoff":
            failures.extend(_order3_in_air_report_failures(report))
        if report.get("asset_cache_reuse_enabled") is not True:
            failures.append("asset_cache_reuse_disabled")
        if failures:
            result.real_isaac_passed = False
            result.failure_reason = "order3_baseline_report_validation_failed:" + ",".join(failures)
        result.metrics["order3_baseline_report_validation_failures"] = failures
        return result


def _order3_in_air_result_from_report(
    morphology_graph: MorphologyGraph,
    *,
    report: dict[str, Any],
    dry_contract: RandomMorphologyTakeoffResult,
) -> RandomMorphologyTakeoffResult:
    passed = bool(report.get("order3_free_flight_passed"))
    return RandomMorphologyTakeoffResult(
        graph_id=morphology_graph.graph_id,
        attempted=True,
        dry_run=False,
        isaac_backed=report.get("isaac_backed") is True,
        unit_contract_passed=True,
        real_isaac_passed=passed,
        placement=dry_contract.placement,
        metrics={
            **dry_contract.metrics,
            "order3_free_flight_passed": passed,
            "order3_task_mode": report.get("order3_rollout_task_mode"),
            "order3_floor_evidence_claimed": False,
        },
        report=dict(report),
        failure_reason=None if passed else "order3_free_flight_report_failed",
    )


def order3_condition_report_failures(
    report: dict[str, Any],
    *,
    expected_condition: Order3RolloutCondition,
) -> list[str]:
    failures: list[str] = []
    if report.get("order3_rollout_condition") != expected_condition.to_dict():
        failures.append("mismatch:order3_rollout_condition")
    if report.get("order3_rollout_condition_hash") != expected_condition.condition_hash:
        failures.append("mismatch:order3_rollout_condition_hash")
    if report.get("order3_rollout_task_mode") != expected_condition.task_mode:
        failures.append("mismatch:order3_rollout_task_mode")
    if report.get("order3_task_mode") != expected_condition.task_mode:
        failures.append("mismatch:order3_task_mode")
    if report.get("order3_report_validation_failures") != []:
        failures.append("producer:order3_report_validation_failures")
    seed_evidence = report.get("order3_rollout_seed_applied")
    if not isinstance(seed_evidence, dict) or seed_evidence.get("seed") != expected_condition.seed:
        failures.append("mismatch:order3_rollout_seed_applied")
    elif any(seed_evidence.get(name) is not True for name in ("python_random", "torch")):
        failures.append("incomplete:order3_rollout_seed_applied")
    if report.get("order3_privileged_external_wrench_body") != list(
        expected_condition.external_wrench_body
    ):
        failures.append("mismatch:order3_privileged_external_wrench_body")
    if report.get("order3_disturbance_start_s") != expected_condition.disturbance_start_s:
        failures.append("mismatch:order3_disturbance_start_s")
    if (
        report.get("order3_disturbance_duration_s")
        != expected_condition.disturbance_duration_s
    ):
        failures.append("mismatch:order3_disturbance_duration_s")
    realization = _parse_condition_realization(report)
    if realization is None:
        failures.append("invalid:order3_condition_realization")
    elif realization.condition_hash != expected_condition.condition_hash:
        failures.append("mismatch:order3_condition_realization_hash")
    elif any(
        (
            realization.task_mode != expected_condition.task_mode,
            realization.requested_mass_scale != expected_condition.mass_scale,
            realization.applied_mass_scale != expected_condition.mass_scale,
            realization.requested_inertia_scale != expected_condition.inertia_scale,
            realization.applied_inertia_scale != expected_condition.inertia_scale,
            realization.requested_thrust_scale != expected_condition.thrust_scale,
            realization.applied_thrust_scale != expected_condition.thrust_scale,
            not realization.mass_randomization_applied,
            not realization.inertia_randomization_applied,
            not realization.thrust_randomization_applied,
            not realization.initial_state_applied,
        )
    ):
        failures.append("mismatch:order3_condition_realization")
    expected_terminal_start = order3_terminal_evidence_start_s(expected_condition)
    if not _finite_float_equal(
        report.get("order3_terminal_evidence_start_s"), expected_terminal_start
    ):
        failures.append("mismatch:order3_terminal_evidence_start_s")
    if report.get("order3_terminal_evidence_completed") is not True:
        failures.append("incomplete:order3_terminal_evidence")
    expected_tracking_start = order3_tracking_window_start_s(expected_condition)
    if not _finite_float_equal(
        report.get("order3_tracking_window_start_s"), expected_tracking_start
    ):
        failures.append("mismatch:order3_tracking_window_start_s")
    tracking_end = report.get("order3_tracking_window_end_s")
    if (
        not isinstance(tracking_end, (int, float))
        or isinstance(tracking_end, bool)
        or not math.isfinite(float(tracking_end))
        or float(tracking_end) + 1.0e-12 < expected_tracking_start
    ):
        failures.append("invalid:order3_tracking_window_end_s")
    tracking_count = report.get("order3_tracking_window_sample_count")
    if (
        not isinstance(tracking_count, int)
        or isinstance(tracking_count, bool)
        or tracking_count <= 0
    ):
        failures.append("invalid:order3_tracking_window_sample_count")
    return failures


# Backward-compatible private name retained for existing callers/tests.
_order3_condition_report_failures = order3_condition_report_failures


def order3_provenance_report_failures(
    report: Mapping[str, Any],
    *,
    expected_backend_config_hash: str,
    expected_physical_model_hash: str,
    expected_collision_geometry_hash: str,
) -> list[str]:
    """Validate the simulator/physics identity bound to an Order-3 report."""

    expected = {
        "random_morphology_takeoff_backend_config_hash": expected_backend_config_hash,
        "random_morphology_takeoff_physical_model_hash": expected_physical_model_hash,
        "random_morphology_takeoff_collision_geometry_hash": (
            expected_collision_geometry_hash
        ),
    }
    return [
        f"mismatch:{key}"
        for key, value in expected.items()
        if report.get(key) != value
    ]


def _order3_in_air_report_failures(report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if report.get("order3_free_flight_report_version") != ORDER3_FREE_FLIGHT_REPORT_VERSION:
        failures.append("mismatch:order3_free_flight_report_version")
    if report.get("order3_free_flight_floor_initialization") is not False:
        failures.append("in_air_floor_initialization_claim")
    if report.get("order3_free_flight_floor_evidence_claim") is not False:
        failures.append("in_air_floor_evidence_claim")
    if report.get("isaac_backed") is not True:
        failures.append("missing:isaac_backed")
    if report.get("order3_free_flight_success") != report.get(
        "order3_free_flight_passed"
    ):
        failures.append("mismatch:order3_free_flight_success")
    tracking_cost = report.get("order3_free_flight_tracking_cost")
    if (
        not isinstance(tracking_cost, (int, float))
        or isinstance(tracking_cost, bool)
        or not math.isfinite(float(tracking_cost))
        or float(tracking_cost) < 0.0
    ):
        failures.append("invalid:order3_free_flight_tracking_cost")
    structural_hash = report.get("order3_structural_hash")
    if not isinstance(structural_hash, str) or not _is_sha256(structural_hash):
        failures.append("invalid:order3_structural_hash")
    terminal = report.get("order3_terminal_metrics")
    required = {
        "position_error_m",
        "attitude_error_rad",
        "linear_velocity_error_mps",
        "angular_velocity_error_rad_s",
        "within_tolerance_duration_s",
        "takeoff_height_gain_ratio",
    }
    if not isinstance(terminal, dict) or set(terminal) != required:
        failures.append("invalid:order3_terminal_metrics")
    elif any(
        not isinstance(terminal[key], (int, float))
        or isinstance(terminal[key], bool)
        or not math.isfinite(float(terminal[key]))
        for key in required - {"takeoff_height_gain_ratio"}
    ) or terminal["takeoff_height_gain_ratio"] is not None:
        failures.append("invalid:order3_terminal_metrics")
    if report.get("order3_free_flight_terminal_metrics") != terminal:
        failures.append("mismatch:order3_free_flight_terminal_metrics")
    for key in (
        "order3_free_flight_qp_infeasible_count",
        "order3_free_flight_hard_collision_count",
        "order3_free_flight_non_finite_state_count",
        "order3_free_flight_unsupported_actuator_count",
    ):
        _non_negative_int(report, key, failures)
    aliases = (
        ("order3_qp_infeasible", "order3_free_flight_qp_infeasible_count"),
        ("order3_hard_collision", "order3_free_flight_hard_collision_count"),
        ("order3_non_finite_state", "order3_free_flight_non_finite_state_count"),
        (
            "order3_unsupported_actuator",
            "order3_free_flight_unsupported_actuator_count",
        ),
    )
    for boolean_key, count_key in aliases:
        count = report.get(count_key)
        expected = isinstance(count, int) and not isinstance(count, bool) and count > 0
        if report.get(boolean_key) is not expected:
            failures.append(f"mismatch:{boolean_key}")
    if report.get("random_morphology_takeoff_fixed_dock_neutral_hold_passed") is not True:
        failures.append("fixed_dock_neutral_hold_failed")
    dock_joint_count = report.get("random_morphology_takeoff_fixed_dock_joint_count")
    if (
        not isinstance(dock_joint_count, int)
        or isinstance(dock_joint_count, bool)
        or dock_joint_count <= 0
    ):
        failures.append("invalid:random_morphology_takeoff_fixed_dock_joint_count")
    tolerance = report.get(
        "random_morphology_takeoff_dock_joint_position_tolerance_rad"
    )
    if (
        not isinstance(tolerance, (int, float))
        or isinstance(tolerance, bool)
        or not math.isclose(
            float(tolerance),
            FIXED_DOCK_JOINT_POSITION_TOLERANCE_RAD,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        failures.append(
            "mismatch:random_morphology_takeoff_dock_joint_position_tolerance_rad"
        )
    for key in (
        "random_morphology_takeoff_max_abs_dock_joint_position_rad",
        "random_morphology_takeoff_final_max_abs_dock_joint_position_rad",
    ):
        value = report.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) > FIXED_DOCK_JOINT_POSITION_TOLERANCE_RAD
        ):
            failures.append(f"exceeds:{key}")
    for key in (
        "random_morphology_takeoff_max_abs_dock_position_target_rad",
        "random_morphology_takeoff_max_abs_dock_velocity_target_rad_s",
        "random_morphology_takeoff_max_abs_dock_torque_bias_nm",
    ):
        value = report.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or abs(float(value)) > 1.0e-12
        ):
            failures.append(f"nonzero:{key}")
    return failures


def _parse_condition_realization(
    report: dict[str, Any],
) -> Order3ConditionRealization | None:
    try:
        return Order3ConditionRealization.from_dict(
            report.get("order3_condition_realization")
        )
    except (SchemaValidationError, TypeError):
        return None


def _order3_report_failures(
    report: dict[str, Any],
    *,
    expected_checkpoint_sha256: str,
    record_transitions: bool,
) -> list[str]:
    failures: list[str] = []

    def exact(key: str, expected: Any) -> None:
        if key not in report:
            failures.append(f"missing:{key}")
        elif type(report[key]) is not type(expected) or report[key] != expected:
            failures.append(f"mismatch:{key}")

    exact("order3_pi_l_rollout", True)
    exact("order3_pi_l_checkpoint_sha256", expected_checkpoint_sha256)
    exact("asset_cache_reuse_enabled", True)
    for key in ("asset_cache_key", "generated_urdf_sha256"):
        value = report.get(key)
        if not isinstance(value, str) or not _is_sha256(value):
            failures.append(f"invalid:{key}")
    metadata_value = report.get("order3_pi_l_checkpoint_metadata")
    try:
        metadata = Order3PolicyCheckpointMetadata.from_dict(metadata_value)
    except (SchemaValidationError, TypeError):
        metadata = None
        failures.append("invalid:order3_pi_l_checkpoint_metadata")
    if metadata is not None and metadata.actor_uses_privileged_wrench:
        failures.append("actor_privileged_wrench")
    decisions = _non_negative_int(report, "order3_pi_l_policy_decision_count", failures)
    applied = _non_negative_int(report, "order3_pi_l_policy_applied_count", failures)
    fallback = _non_negative_int(report, "order3_pi_l_fallback_count", failures)
    if decisions is not None and decisions <= 0:
        failures.append("no_policy_decisions")
    if None not in (decisions, applied, fallback) and applied + fallback != decisions:
        failures.append("policy_count_partition")
    traces = report.get("order3_pi_l_transition_traces")
    if not isinstance(traces, list):
        failures.append("invalid:order3_pi_l_transition_traces")
        traces = []
    if record_transitions and decisions is not None and len(traces) != decisions:
        failures.append("transition_trace_count")
    bootstrap = report.get("order3_pi_l_final_bootstrap_value")
    if bootstrap is not None and (
        not isinstance(bootstrap, (int, float))
        or isinstance(bootstrap, bool)
        or not math.isfinite(float(bootstrap))
    ):
        failures.append("invalid:order3_pi_l_final_bootstrap_value")
    runtime_rows = report.get("random_morphology_takeoff_runtime_observations")
    policy_rows = report.get("random_morphology_takeoff_policy_commands")
    if not isinstance(runtime_rows, list) or not isinstance(policy_rows, list):
        failures.append("missing_aligned_control_rows")
        return failures
    last_step = -1
    recurrent_width: int | None = None
    for trace in traces:
        if not isinstance(trace, dict):
            failures.append("invalid_transition_trace")
            continue
        step_index = trace.get("step_index")
        if not isinstance(step_index, int) or isinstance(step_index, bool):
            failures.append("invalid_transition_step")
            continue
        if step_index <= last_step or not 0 <= step_index < len(runtime_rows):
            failures.append("transition_step_alignment")
            continue
        last_step = step_index
        for key, width in (
            ("target_pose_world", 7),
            ("target_twist", 6),
            ("previous_action", ORDER3_ACTION_SIZE),
            ("action", ORDER3_ACTION_SIZE),
            ("privileged_disturbance_body", 6),
        ):
            value = trace.get(key)
            if (
                not isinstance(value, list)
                or len(value) != width
                or not all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in value)
            ):
                failures.append(f"invalid_trace:{key}")
        action = trace.get("action")
        if isinstance(action, list) and any(abs(float(value)) > 1.0 for value in action):
            failures.append("unbounded_trace_action")
        recurrent = trace.get("recurrent_state_in")
        if not isinstance(recurrent, list) or not recurrent:
            failures.append("invalid_trace:recurrent_state_in")
        elif recurrent_width is None:
            recurrent_width = len(recurrent)
        elif len(recurrent) != recurrent_width:
            failures.append("recurrent_state_width")
        command = PolicyCommand.from_dict(policy_rows[step_index])
        if command.desired_body_pose != tuple(trace.get("target_pose_world", [])):
            failures.append("centroidal_pose_not_passed_through")
        if command.control_contract_version != POLICY_COMMAND_CONTRACT_CENTROIDAL:
            failures.append("trace_policy_contract")
        if command.contact_tracking_bias or command.joint_position_bias or command.joint_velocity_bias:
            failures.append("deprecated_policy_output")
    return sorted(set(failures))


def _non_negative_int(report: dict[str, Any], key: str, failures: list[str]) -> int | None:
    value = report.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        failures.append(f"invalid:{key}")
        return None
    return value


def _finite_float_equal(left: Any, right: float) -> bool:
    return (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and math.isfinite(float(left))
        and math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1.0e-9)
    )


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


__all__ = [
    "ORDER3_ISAAC_POLICY_ROLLOUT_VERSION",
    "Order3DeterministicBaselineRolloutConfig",
    "Order3DeterministicBaselineRolloutEnv",
    "Order3IsaacPolicyRolloutConfig",
    "Order3IsaacPolicyRolloutEnv",
    "Order3IsaacPolicyRolloutResult",
    "order3_condition_report_failures",
    "order3_provenance_report_failures",
]
