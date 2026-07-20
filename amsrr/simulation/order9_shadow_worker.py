from __future__ import annotations

"""Persistent worker contract for real-Isaac Order 9 shadow rollouts."""

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
import selectors
import subprocess
import sys
import time
from typing import Any, Mapping, Protocol, Sequence, TextIO

from amsrr.feasibility.contact_wrench_hybrid import (
    ShadowCollisionSample,
    ShadowKnotObservation,
)
from amsrr.morphology.random_connected import morphology_structural_hash
from amsrr.policies.high_level_policy_base import HighLevelPolicyContext
from amsrr.schemas.common import SchemaBase, SchemaValidationError, require_non_empty
from amsrr.schemas.contact_candidates import ContactCandidateSet
from amsrr.schemas.interaction_envelope import InteractionEnvelope
from amsrr.schemas.irg import InteractionRequirementGraph
from amsrr.schemas.morphology import MorphologyGraph
from amsrr.schemas.policies import ContactWrenchTrajectory
from amsrr.schemas.runtime import RuntimeObservation
from amsrr.utils.hashing import stable_hash


ORDER9_SHADOW_STATE_EXPORT_VERSION = "order9_shadow_state_export_v1"
ORDER9_PERSISTENT_ISAAC_SHADOW_DRIVER_VERSION = (
    "order9_persistent_isaac_shadow_driver_v1"
)
ORDER9_SHADOW_RPC_VERSION = "order9_shadow_rpc_v1"
_REQUEST_PREFIX = "ORDER9_SHADOW_REQUEST="
_RESPONSE_PREFIX = "ORDER9_SHADOW_RESPONSE="


@dataclass
class Order9ShadowStateExport(SchemaBase):
    state_id: str
    topology_structural_hash: str
    simulation_time_s: float
    simulation_state: dict[str, Any]
    controller_state: dict[str, Any]
    pi_l_state: dict[str, Any]
    pi_l_checkpoint_sha256: str
    export_version: str = ORDER9_SHADOW_STATE_EXPORT_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        require_non_empty(self.state_id, "Order9ShadowStateExport.state_id")
        if self.export_version != ORDER9_SHADOW_STATE_EXPORT_VERSION:
            raise SchemaValidationError("Order9 shadow state export version mismatch")
        _require_sha256(
            self.topology_structural_hash,
            "Order9 shadow topology structural hash",
        )
        _require_sha256(
            self.pi_l_checkpoint_sha256,
            "Order9 shadow pi_L checkpoint hash",
        )
        if (
            not math.isfinite(float(self.simulation_time_s))
            or self.simulation_time_s < 0.0
        ):
            raise SchemaValidationError(
                "Order9 shadow simulation time must be finite and non-negative"
            )
        _require_json_finite(self.simulation_state, "simulation_state")
        _require_json_finite(self.controller_state, "controller_state")
        _require_json_finite(self.pi_l_state, "pi_l_state")
        _require_json_finite(self.metadata, "metadata")

    @property
    def state_digest(self) -> str:
        return stable_hash(self.to_dict())


class Order9MainStateExporter(Protocol):
    def export_shadow_state(
        self,
        context: HighLevelPolicyContext,
    ) -> Order9ShadowStateExport:
        ...


class Order9ShadowWorkerTransport(Protocol):
    @property
    def transport_version(self) -> str:
        ...

    def request(
        self,
        operation: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        ...


class Order9ShadowWorkerExecutor(Protocol):
    """Physics-owning half of the isolated shadow worker.

    The executor is instantiated after Isaac has started.  It owns exactly one
    copied simulator/controller/policy state and must never reference the main
    training environment.  The JSON-line service below owns identity checks;
    the executor owns state restoration and physical execution.
    """

    @property
    def worker_version(self) -> str:
        ...

    @property
    def pi_l_checkpoint_sha256(self) -> str:
        ...

    def synchronize(self, state: Order9ShadowStateExport) -> None:
        ...

    def describe(self) -> Mapping[str, Any]:
        ...

    def execute(
        self,
        *,
        state_digest: str,
        context: HighLevelPolicyContext,
        trajectory: ContactWrenchTrajectory,
    ) -> Sequence[ShadowKnotObservation]:
        ...

    def reset(self, *, state_digest: str) -> None:
        ...

    def close(self) -> None:
        ...


class Order9ShadowWorkerService:
    """Strict state machine behind the persistent JSON-line worker RPC."""

    def __init__(self, executor: Order9ShadowWorkerExecutor) -> None:
        if not executor.worker_version:
            raise ValueError("Order9 shadow executor version must be non-empty")
        _require_sha256(
            executor.pi_l_checkpoint_sha256,
            "Order9 shadow executor pi_L checkpoint hash",
        )
        self.executor = executor
        self._state: Order9ShadowStateExport | None = None
        self._state_digest: str | None = None
        self._closed = False

    def handle(
        self,
        operation: str,
        payload: Mapping[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        if self._closed:
            return _rejected_worker_result(operation, "worker_closed"), True
        try:
            if operation == "describe":
                return self._describe(), False
            if operation == "synchronize":
                return self._synchronize(payload), False
            if operation == "execute":
                return self._execute(payload), False
            if operation == "reset":
                return self._reset(payload), False
            if operation == "shutdown":
                self._shutdown()
                return {
                    "operation": operation,
                    "accepted": True,
                    "worker_version": self.executor.worker_version,
                }, True
            return _rejected_worker_result(operation, "unsupported_operation"), False
        except Exception as exc:
            # The client treats any rejection as a fail-closed C_H result.  Do
            # not let a malformed request terminate the persistent Isaac app.
            return _rejected_worker_result(
                operation,
                f"{type(exc).__name__}:{exc}",
            ), False

    def close(self) -> None:
        if not self._closed:
            self._shutdown()

    def _synchronize(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        raw_state = payload.get("state")
        if not isinstance(raw_state, Mapping):
            raise SchemaValidationError("shadow synchronize requires a state mapping")
        state = Order9ShadowStateExport.from_dict(dict(raw_state))
        state.validate()
        digest = str(payload.get("state_digest", ""))
        checkpoint = str(payload.get("pi_l_checkpoint_sha256", ""))
        if digest != state.state_digest:
            raise SchemaValidationError("shadow synchronize state digest mismatch")
        if checkpoint != state.pi_l_checkpoint_sha256:
            raise SchemaValidationError("shadow synchronize state checkpoint mismatch")
        if checkpoint != self.executor.pi_l_checkpoint_sha256:
            raise SchemaValidationError("shadow executor loaded a different pi_L checkpoint")
        if self._state_digest is not None:
            raise RuntimeError("shadow synchronize requires reset of the previous copy")
        self.executor.synchronize(state)
        self._state = state
        self._state_digest = digest
        return {
            "operation": "synchronize",
            "accepted": True,
            "state_digest": digest,
            "pi_l_checkpoint_sha256": checkpoint,
            "worker_version": self.executor.worker_version,
        }

    def _describe(self) -> dict[str, Any]:
        descriptor = dict(self.executor.describe())
        _require_json_finite(descriptor, "shadow_worker_descriptor")
        return {
            "operation": "describe",
            "accepted": True,
            "worker_version": self.executor.worker_version,
            "pi_l_checkpoint_sha256": self.executor.pi_l_checkpoint_sha256,
            "descriptor": descriptor,
        }

    def _execute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        state = self._state
        digest = str(payload.get("state_digest", ""))
        if state is None or self._state_digest is None:
            raise RuntimeError("shadow execute requires synchronized copied state")
        if digest != self._state_digest:
            raise SchemaValidationError("shadow execute state digest mismatch")
        checkpoint = str(payload.get("pi_l_checkpoint_sha256", ""))
        if checkpoint != state.pi_l_checkpoint_sha256:
            raise SchemaValidationError("shadow execute checkpoint mismatch")
        raw_trajectory = payload.get("trajectory")
        raw_context = payload.get("context")
        if not isinstance(raw_trajectory, Mapping) or not isinstance(raw_context, Mapping):
            raise SchemaValidationError(
                "shadow execute requires trajectory and context mappings"
            )
        trajectory = ContactWrenchTrajectory.from_dict(dict(raw_trajectory))
        trajectory.validate()
        context = _context_from_dict(raw_context)
        proposal_hash = str(payload.get("proposal_hash", ""))
        if proposal_hash != trajectory.stable_hash():
            raise SchemaValidationError("shadow execute proposal hash mismatch")
        if (
            morphology_structural_hash(context.morphology_graph)
            != state.topology_structural_hash
        ):
            raise SchemaValidationError("shadow execute topology hash mismatch")
        observations = tuple(
            self.executor.execute(
                state_digest=digest,
                context=context,
                trajectory=trajectory,
            )
        )
        if len(observations) != len(trajectory.knots) or any(
            not isinstance(item, ShadowKnotObservation) for item in observations
        ):
            raise RuntimeError(
                "shadow executor must return one typed observation per knot"
            )
        return {
            "operation": "execute",
            "accepted": True,
            "state_digest": digest,
            "pi_l_checkpoint_sha256": checkpoint,
            "proposal_hash": proposal_hash,
            "worker_version": self.executor.worker_version,
            "observations": [
                _shadow_observation_to_dict(item) for item in observations
            ],
        }

    def _reset(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        digest = str(payload.get("state_digest", ""))
        if self._state_digest is None or digest != self._state_digest:
            raise SchemaValidationError("shadow reset state digest mismatch")
        self.executor.reset(state_digest=digest)
        self._state = None
        self._state_digest = None
        return {
            "operation": "reset",
            "accepted": True,
            "state_digest": digest,
            "worker_version": self.executor.worker_version,
        }

    def _shutdown(self) -> None:
        try:
            if self._state_digest is not None:
                self.executor.reset(state_digest=self._state_digest)
        finally:
            self._state = None
            self._state_digest = None
            self.executor.close()
            self._closed = True


def run_order9_shadow_worker_rpc(
    executor: Order9ShadowWorkerExecutor,
    *,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> int:
    """Serve prefixed RPC lines until an authenticated shutdown or EOF."""

    source = input_stream or sys.stdin
    destination = output_stream or sys.stdout
    service = Order9ShadowWorkerService(executor)
    try:
        for line in source:
            if not line.startswith(_REQUEST_PREFIX):
                continue
            request_id: object = None
            operation = "invalid"
            stop = False
            try:
                request = json.loads(line[len(_REQUEST_PREFIX) :])
                if not isinstance(request, Mapping):
                    raise SchemaValidationError("shadow RPC request must be a mapping")
                request_id = request.get("request_id")
                if request.get("rpc_version") != ORDER9_SHADOW_RPC_VERSION:
                    raise SchemaValidationError("shadow RPC version mismatch")
                if not isinstance(request_id, int) or request_id < 0:
                    raise SchemaValidationError("shadow RPC request id is invalid")
                operation = str(request.get("operation", ""))
                payload = request.get("payload")
                if not operation or not isinstance(payload, Mapping):
                    raise SchemaValidationError(
                        "shadow RPC operation/payload is invalid"
                    )
                result, stop = service.handle(operation, payload)
            except Exception as exc:
                result = _rejected_worker_result(
                    operation,
                    f"{type(exc).__name__}:{exc}",
                )
            response = {
                "rpc_version": ORDER9_SHADOW_RPC_VERSION,
                "request_id": request_id,
                "result": result,
            }
            _require_json_finite(response, "shadow_rpc_response")
            destination.write(
                _RESPONSE_PREFIX + json.dumps(response, sort_keys=True) + "\n"
            )
            destination.flush()
            if stop:
                return 0
        return 0
    finally:
        service.close()


class PersistentIsaacShadowDriver:
    """Bind exact main-state export to one topology-bucket Isaac worker."""

    def __init__(
        self,
        *,
        state_exporter: Order9MainStateExporter,
        transport: Order9ShadowWorkerTransport,
        pi_l_checkpoint_sha256: str,
        worker_version: str,
    ) -> None:
        _require_sha256(pi_l_checkpoint_sha256, "Order9 shadow checkpoint hash")
        if not worker_version:
            raise ValueError("Order9 shadow worker_version must be non-empty")
        if not transport.transport_version:
            raise ValueError("Order9 shadow transport_version must be non-empty")
        self.state_exporter = state_exporter
        self.transport = transport
        self.pi_l_checkpoint_sha256 = pi_l_checkpoint_sha256
        self.worker_version = worker_version
        self._pending_state: Order9ShadowStateExport | None = None
        self._synchronized_digest: str | None = None

    @property
    def driver_version(self) -> str:
        return (
            f"{ORDER9_PERSISTENT_ISAAC_SHADOW_DRIVER_VERSION}:"
            f"{self.worker_version}:{self.transport.transport_version}"
        )

    def main_state_digest(self, context: HighLevelPolicyContext) -> str:
        state = self.state_exporter.export_shadow_state(context)
        state.validate()
        if state.pi_l_checkpoint_sha256 != self.pi_l_checkpoint_sha256:
            raise SchemaValidationError(
                "Order9 shadow state is not bound to the configured pi_L checkpoint"
            )
        self._pending_state = state
        return state.state_digest

    def synchronize_shadow(self, context: HighLevelPolicyContext) -> None:
        del context
        state = self._pending_state
        if state is None:
            raise RuntimeError(
                "Order9 shadow synchronize requires a preceding main-state digest"
            )
        response = self.transport.request(
            "synchronize",
            {
                "state": state.to_dict(),
                "state_digest": state.state_digest,
                "pi_l_checkpoint_sha256": self.pi_l_checkpoint_sha256,
            },
        )
        _require_accepted_response(response, "synchronize")
        if response.get("state_digest") != state.state_digest:
            raise RuntimeError("Order9 shadow worker synchronized the wrong state")
        if response.get("pi_l_checkpoint_sha256") != self.pi_l_checkpoint_sha256:
            raise RuntimeError("Order9 shadow worker loaded the wrong pi_L checkpoint")
        self._synchronized_digest = state.state_digest

    def execute_shadow(
        self,
        *,
        context: HighLevelPolicyContext,
        trajectory: ContactWrenchTrajectory,
    ) -> tuple[ShadowKnotObservation, ...]:
        state = self._pending_state
        if state is None or self._synchronized_digest != state.state_digest:
            raise RuntimeError("Order9 shadow execute requires synchronized state")
        proposal_hash = trajectory.stable_hash()
        response = self.transport.request(
            "execute",
            {
                "state_digest": state.state_digest,
                "pi_l_checkpoint_sha256": self.pi_l_checkpoint_sha256,
                "proposal_hash": proposal_hash,
                "trajectory": trajectory.to_dict(),
                "context": _context_to_dict(context),
            },
        )
        _require_accepted_response(response, "execute")
        expected_echoes = {
            "state_digest": state.state_digest,
            "pi_l_checkpoint_sha256": self.pi_l_checkpoint_sha256,
            "proposal_hash": proposal_hash,
        }
        for name, expected in expected_echoes.items():
            if response.get(name) != expected:
                raise RuntimeError(
                    f"Order9 shadow execute response has wrong {name}"
                )
        raw_observations = response.get("observations")
        if not isinstance(raw_observations, list):
            raise RuntimeError(
                "Order9 shadow execute response requires observation rows"
            )
        observations = tuple(
            _shadow_observation_from_dict(item) for item in raw_observations
        )
        if len(observations) != len(trajectory.knots):
            raise RuntimeError(
                "Order9 shadow worker returned the wrong knot observation count"
            )
        return observations

    def reset_shadow(self) -> None:
        digest = self._synchronized_digest
        try:
            if digest is not None:
                response = self.transport.request(
                    "reset",
                    {"state_digest": digest},
                )
                _require_accepted_response(response, "reset")
                if response.get("state_digest") != digest:
                    raise RuntimeError(
                        "Order9 shadow worker reset the wrong copied state"
                    )
        finally:
            self._synchronized_digest = None
            self._pending_state = None


class JsonLineSubprocessShadowTransport:
    """Persistent prefixed JSON-line transport tolerant of Isaac log output."""

    transport_version = "order9_prefixed_jsonline_subprocess_v1"

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path,
        timeout_s: float = 30.0,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        if not command or not all(str(value) for value in command):
            raise ValueError("Order9 shadow worker command must be non-empty")
        if not math.isfinite(float(timeout_s)) or timeout_s <= 0.0:
            raise ValueError("Order9 shadow worker timeout must be positive")
        self.command = [str(value) for value in command]
        self.cwd = str(Path(cwd).resolve())
        self.timeout_s = float(timeout_s)
        self.environment = (
            None
            if environment is None
            else {**os.environ, **{str(key): str(value) for key, value in environment.items()}}
        )
        self._process: subprocess.Popen[str] | None = None
        self._request_index = 0

    def request(
        self,
        operation: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if not operation:
            raise ValueError("Order9 shadow RPC operation must be non-empty")
        process = self._ensure_process()
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("Order9 shadow worker pipes are unavailable")
        request_id = self._request_index
        self._request_index += 1
        request = {
            "rpc_version": ORDER9_SHADOW_RPC_VERSION,
            "request_id": request_id,
            "operation": operation,
            "payload": dict(payload),
        }
        _require_json_finite(request, "shadow_rpc_request")
        process.stdin.write(_REQUEST_PREFIX + json.dumps(request, sort_keys=True) + "\n")
        process.stdin.flush()
        deadline = time.monotonic() + self.timeout_s
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Order9 shadow worker exited with code {process.returncode}"
                    )
                remaining = max(0.0, deadline - time.monotonic())
                if not selector.select(timeout=remaining):
                    break
                line = process.stdout.readline()
                if line == "":
                    continue
                if not line.startswith(_RESPONSE_PREFIX):
                    continue
                response = json.loads(line[len(_RESPONSE_PREFIX) :])
                if not isinstance(response, dict):
                    raise RuntimeError("Order9 shadow RPC response must be a mapping")
                if response.get("rpc_version") != ORDER9_SHADOW_RPC_VERSION:
                    raise RuntimeError("Order9 shadow RPC version mismatch")
                if response.get("request_id") != request_id:
                    raise RuntimeError("Order9 shadow RPC request id mismatch")
                result = response.get("result")
                if not isinstance(result, dict):
                    raise RuntimeError("Order9 shadow RPC result must be a mapping")
                return result
        finally:
            selector.close()
        raise TimeoutError(
            f"Order9 shadow worker timed out during {operation!r}"
        )

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None:
            try:
                if process.stdin is not None:
                    request = {
                        "rpc_version": ORDER9_SHADOW_RPC_VERSION,
                        "request_id": self._request_index,
                        "operation": "shutdown",
                        "payload": {},
                    }
                    process.stdin.write(
                        _REQUEST_PREFIX + json.dumps(request, sort_keys=True) + "\n"
                    )
                    process.stdin.flush()
                process.wait(timeout=min(self.timeout_s, 5.0))
            except (BrokenPipeError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
        if process.stdin is not None:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()

    def __enter__(self) -> "JsonLineSubprocessShadowTransport":
        self._ensure_process()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        self._process = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            env=self.environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        return self._process


def _context_to_dict(context: HighLevelPolicyContext) -> dict[str, Any]:
    return {
        "irg": context.irg.to_dict(),
        "interaction_envelope": context.interaction_envelope.to_dict(),
        "morphology_graph": context.morphology_graph.to_dict(),
        "contact_candidate_set": context.contact_candidate_set.to_dict(),
        "runtime_observation": (
            None
            if context.runtime_observation is None
            else context.runtime_observation.to_dict()
        ),
    }


def _context_from_dict(value: Mapping[str, Any]) -> HighLevelPolicyContext:
    required = {
        "irg": InteractionRequirementGraph,
        "interaction_envelope": InteractionEnvelope,
        "morphology_graph": MorphologyGraph,
        "contact_candidate_set": ContactCandidateSet,
    }
    parsed: dict[str, Any] = {}
    for name, schema in required.items():
        raw = value.get(name)
        if not isinstance(raw, Mapping):
            raise SchemaValidationError(f"shadow context requires {name} mapping")
        parsed[name] = schema.from_dict(dict(raw))
        parsed[name].validate()
    raw_observation = value.get("runtime_observation")
    observation = None
    if raw_observation is not None:
        if not isinstance(raw_observation, Mapping):
            raise SchemaValidationError(
                "shadow runtime_observation must be a mapping or null"
            )
        observation = RuntimeObservation.from_dict(dict(raw_observation))
        observation.validate()
    return HighLevelPolicyContext(
        irg=parsed["irg"],
        interaction_envelope=parsed["interaction_envelope"],
        morphology_graph=parsed["morphology_graph"],
        contact_candidate_set=parsed["contact_candidate_set"],
        runtime_observation=observation,
    )


def _shadow_observation_to_dict(
    observation: ShadowKnotObservation,
) -> dict[str, Any]:
    return {
        "controller_qp_residual": float(observation.controller_qp_residual),
        "contact_wrench_residual": float(observation.contact_wrench_residual),
        "collision_samples": [
            {
                "entity_a": item.entity_a,
                "entity_b": item.entity_b,
                "signed_distance_m": float(item.signed_distance_m),
                "candidate_id": item.candidate_id,
                "anchor_id": item.anchor_id,
                "target_entity_id": item.target_entity_id,
                "task_allowed": item.task_allowed,
                "allowance_reason": item.allowance_reason,
            }
            for item in observation.collision_samples
        ],
        "collision_free_clearance_m": float(
            observation.collision_free_clearance_m
        ),
        "finite_state": bool(observation.finite_state),
        "main_state_unchanged": bool(observation.main_state_unchanged),
        "metrics": {
            str(key): float(item) for key, item in observation.metrics.items()
        },
    }


def _shadow_observation_from_dict(value: object) -> ShadowKnotObservation:
    if not isinstance(value, Mapping):
        raise RuntimeError("Order9 shadow knot observation must be a mapping")
    raw_samples = value.get("collision_samples", [])
    if not isinstance(raw_samples, list):
        raise RuntimeError("Order9 shadow collision_samples must be a list")
    samples: list[ShadowCollisionSample] = []
    for raw in raw_samples:
        if not isinstance(raw, Mapping):
            raise RuntimeError("Order9 shadow collision sample must be a mapping")
        samples.append(
            ShadowCollisionSample(
                entity_a=str(raw.get("entity_a", "")),
                entity_b=str(raw.get("entity_b", "")),
                signed_distance_m=float(raw.get("signed_distance_m", 0.0)),
                candidate_id=(
                    None
                    if raw.get("candidate_id") is None
                    else int(raw["candidate_id"])
                ),
                anchor_id=(
                    None
                    if raw.get("anchor_id") is None
                    else int(raw["anchor_id"])
                ),
                target_entity_id=(
                    None
                    if raw.get("target_entity_id") is None
                    else str(raw["target_entity_id"])
                ),
                task_allowed=bool(raw.get("task_allowed", False)),
                allowance_reason=(
                    None
                    if raw.get("allowance_reason") is None
                    else str(raw["allowance_reason"])
                ),
            )
        )
    raw_metrics = value.get("metrics", {})
    if not isinstance(raw_metrics, Mapping):
        raise RuntimeError("Order9 shadow observation metrics must be a mapping")
    return ShadowKnotObservation(
        controller_qp_residual=float(value.get("controller_qp_residual", 1.0)),
        contact_wrench_residual=float(value.get("contact_wrench_residual", 1.0)),
        collision_samples=tuple(samples),
        collision_free_clearance_m=float(
            value.get("collision_free_clearance_m", 0.0)
        ),
        finite_state=bool(value.get("finite_state", False)),
        main_state_unchanged=bool(value.get("main_state_unchanged", True)),
        metrics={str(key): float(item) for key, item in raw_metrics.items()},
    )


def _require_accepted_response(response: Mapping[str, Any], operation: str) -> None:
    if response.get("operation") != operation or response.get("accepted") is not True:
        raise RuntimeError(f"Order9 shadow worker rejected {operation!r}")


def _rejected_worker_result(operation: str, reason: str) -> dict[str, Any]:
    return {
        "operation": str(operation),
        "accepted": False,
        "error": str(reason),
    }


def _require_sha256(value: str, name: str) -> None:
    if (
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaValidationError(f"{name} must be a SHA-256 digest")


def _require_json_finite(value: object, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SchemaValidationError(f"{path} must contain only finite values")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SchemaValidationError(f"{path} keys must be strings")
            _require_json_finite(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_json_finite(item, f"{path}[{index}]")
        return
    raise SchemaValidationError(f"{path} contains a non-JSON value")


__all__ = [
    "ORDER9_PERSISTENT_ISAAC_SHADOW_DRIVER_VERSION",
    "ORDER9_SHADOW_RPC_VERSION",
    "ORDER9_SHADOW_STATE_EXPORT_VERSION",
    "JsonLineSubprocessShadowTransport",
    "Order9MainStateExporter",
    "Order9ShadowStateExport",
    "Order9ShadowWorkerExecutor",
    "Order9ShadowWorkerService",
    "Order9ShadowWorkerTransport",
    "PersistentIsaacShadowDriver",
    "run_order9_shadow_worker_rpc",
]
